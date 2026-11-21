//! OpenAI-compatible HTTP server: `/v1/chat/completions` (streaming SSE or not), `/v1/models`, `/health`.
//!
//! One GPU worker thread owns the engine and serves requests in arrival order. Prompts are rendered with the
//! model's chat template (`inkling_core::chat`); generated tokens go through the streaming parser, which splits
//! reasoning (`reasoning_content`), answer text (`content`) and tool calls (`tool_calls`). Consecutive requests that
//! extend the previous conversation reuse its KV / conv state (engine prompt cache).

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use axum::Router;
use axum::extract::{Json, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::StreamExt;
use inkling_core::chat::{self, Event};
use inkling_core::{Gguf, Tokenizer};
use inkling_engine::{Engine, Sampling};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Server options.
#[derive(Clone, Debug)]
pub struct Options {
    pub host: String,
    pub port: u16,
    pub n_ctx: usize,
    pub api_key: Option<String>,
    /// model id reported by /v1/models (default: file name)
    pub model_id: String,
    /// sampling defaults for requests that do not set them
    pub defaults: Defaults,
}

#[derive(Clone, Copy, Debug)]
pub struct Defaults {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
}

impl Default for Defaults {
    fn default() -> Self {
        Self { temperature: 0.8, top_p: 0.95, top_k: 40, min_p: 0.05 }
    }
}

/// Generation request for the GPU worker.
struct Job {
    prompt: Vec<u32>,
    max_new: usize,
    sampling: Sampling,
    stop: Vec<String>,
    tx: mpsc::UnboundedSender<Out>,
}

/// What the worker reports back for one job.
enum Out {
    Started { prompt_tokens: usize, cached: usize },
    Ev(Event),
    Done { finish: &'static str, completion_tokens: usize },
    Error(String),
}

struct AppState {
    tok: Arc<Tokenizer>,
    jobs: std::sync::mpsc::Sender<Job>,
    opts: Options,
    n_ctx: usize,
}

static COUNTER: AtomicU64 = AtomicU64::new(1);

fn new_id(prefix: &str) -> String {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0) as u64;
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}{:016x}", t ^ c.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Run the server until Ctrl-C.
pub fn serve(model: &str, opts: Options) -> Result<()> {
    let tok = Arc::new(Tokenizer::from_gguf(&Gguf::open(model)?)?);
    let (jobs_tx, jobs_rx) = std::sync::mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<usize>>();
    let model_path = model.to_string();
    let wtok = tok.clone();
    let n_ctx_req = opts.n_ctx;
    std::thread::Builder::new().name("inkling-gpu".into()).spawn(move || {
        let eng = Engine::load(&model_path, n_ctx_req);
        match eng {
            Ok(mut eng) => {
                let _ = ready_tx.send(Ok(n_ctx_req));
                worker(&mut eng, &wtok, jobs_rx);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        }
    })?;
    let n_ctx = ready_rx.recv().context("GPU worker died")??;
    let state = Arc::new(AppState { tok, jobs: jobs_tx, opts: opts.clone(), n_ctx });
    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let addr = format!("{}:{}", opts.host, opts.port);
        let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("binding {addr}"))?;
        println!("InklingEngine serving {} on http://{addr} (context {n_ctx}) - OpenAI API at /v1", opts.model_id);
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
}

/// GPU worker: one job at a time.
fn worker(eng: &mut Engine, tok: &Tokenizer, jobs: std::sync::mpsc::Receiver<Job>) {
    while let Ok(job) = jobs.recv() {
        let tx = job.tx.clone();
        if let Err(e) = run_job(eng, tok, job) {
            let _ = tx.send(Out::Error(format!("{e:#}")));
        }
    }
}

fn run_job(eng: &mut Engine, tok: &Tokenizer, job: Job) -> Result<()> {
    let mut parser = chat::Parser::new(tok)?;
    let tx = job.tx;
    let stop = job.stop;
    let hold = stop.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut finish: Option<&'static str> = None;
    let mut tool_called = false;
    let mut n = 0usize;
    let mut held = String::new(); // content not yet sent (may contain the start of a stop string)
    let mut on_token = |t: u32| -> bool {
        n += 1;
        for ev in parser.push(tok, t) {
            match ev {
                Event::End => {
                    finish = Some(if tool_called { "tool_calls" } else { "stop" });
                }
                Event::Content(s) if hold > 0 => {
                    held.push_str(&s);
                    if let Some(pos) = stop.iter().filter_map(|st| held.find(st.as_str())).min() {
                        let before = held[..pos].to_string();
                        if !before.is_empty() && tx.send(Out::Ev(Event::Content(before))).is_err() {
                            return false;
                        }
                        held.clear();
                        finish = Some("stop");
                    } else if held.len() > hold {
                        // keep the last hold - 1 bytes (on a char boundary) in case a stop string starts there
                        let mut cut = held.len() - (hold - 1);
                        while !held.is_char_boundary(cut) {
                            cut -= 1;
                        }
                        let send: String = held.drain(..cut).collect();
                        if tx.send(Out::Ev(Event::Content(send))).is_err() {
                            return false;
                        }
                    }
                }
                ev => {
                    if matches!(ev, Event::ToolCall { .. }) {
                        tool_called = true;
                    }
                    if tx.send(Out::Ev(ev)).is_err() {
                        return false; // client went away
                    }
                }
            }
            if finish.is_some() {
                return false;
            }
        }
        true
    };
    let prompt_tokens = job.prompt.len();
    // the prompt cache reports reuse only after preparing; send Started from the first token instead
    let (out, cached) = {
        let started = std::cell::Cell::new(false);
        let mut cb = |t: u32| {
            if !started.get() {
                started.set(true);
            }
            on_token(t)
        };
        eng.generate_ex(&job.prompt, job.max_new, &job.sampling, &mut cb)?
    };
    let _ = tx.send(Out::Started { prompt_tokens, cached });
    if finish.is_none() && !held.is_empty() {
        let _ = tx.send(Out::Ev(Event::Content(std::mem::take(&mut held))));
    }
    let finish = finish.unwrap_or(if out.len() >= job.max_new { "length" } else { "stop" });
    let _ = tx.send(Out::Done { finish, completion_tokens: out.len() });
    Ok(())
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error"}}))).into_response()
}

fn authorized(state: &AppState, headers: &HeaderMap) -> bool {
    match &state.opts.api_key {
        None => true,
        Some(k) => headers.get("authorization").and_then(|v| v.to_str().ok()).map(|v| v.trim() == format!("Bearer {k}")).unwrap_or(false),
    }
}

async fn models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !authorized(&state, &headers) {
        return err(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    Json(json!({"object": "list", "data": [{"id": state.opts.model_id, "object": "model", "created": now(), "owned_by": "inklingengine",
        "context_length": state.n_ctx}]}))
    .into_response()
}

/// Parsed request pieces.
struct Req {
    prompt: Vec<u32>,
    max_new: usize,
    sampling: Sampling,
    stop: Vec<String>,
    stream: bool,
    include_usage: bool,
}

fn parse_request(state: &AppState, body: &Value) -> Result<Req> {
    let messages = body.get("messages").and_then(Value::as_array).ok_or_else(|| anyhow!("`messages` must be an array"))?;
    if body.get("n").and_then(Value::as_u64).unwrap_or(1) != 1 {
        return Err(anyhow!("only n = 1 is supported"));
    }
    let tool_choice_none = body.get("tool_choice").and_then(Value::as_str) == Some("none");
    let tools = if tool_choice_none { None } else { body.get("tools").and_then(Value::as_array).map(|v| v.as_slice()) };
    let text = chat::render(messages, tools, body.get("reasoning_effort"))?;
    let prompt = state.tok.encode(&text, true)?;
    if prompt.len() + 1 > state.n_ctx {
        return Err(anyhow!("prompt is {} tokens; the context is {}", prompt.len(), state.n_ctx));
    }
    let room = state.n_ctx - prompt.len();
    let max_new = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(Value::as_u64)
        .map(|v| (v as usize).clamp(1, room))
        .unwrap_or(room);
    let d = state.opts.defaults;
    let f = |k: &str, dv: f32| body.get(k).and_then(Value::as_f64).map(|v| v as f32).unwrap_or(dv);
    let seed = body.get("seed").and_then(Value::as_u64).unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1));
    let sampling = Sampling {
        temperature: f("temperature", d.temperature),
        top_p: f("top_p", d.top_p),
        min_p: f("min_p", d.min_p),
        top_k: body.get("top_k").and_then(Value::as_i64).map(|v| v as i32).unwrap_or(d.top_k),
        seed,
    };
    let stop = match body.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).filter(|s| !s.is_empty()).collect(),
        _ => vec![],
    };
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(Value::as_bool).unwrap_or(false);
    Ok(Req { prompt, max_new, sampling, stop, stream, include_usage })
}

fn usage(prompt_tokens: usize, cached: usize, completion: usize) -> Value {
    json!({"prompt_tokens": prompt_tokens, "completion_tokens": completion, "total_tokens": prompt_tokens + completion,
        "prompt_tokens_details": {"cached_tokens": cached}})
}

async fn chat_completions(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    if !authorized(&state, &headers) {
        return err(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    let req = match parse_request(&state, &body) {
        Ok(r) => r,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    let (tx, rx) = mpsc::unbounded_channel();
    let stream = req.stream;
    let include_usage = req.include_usage;
    if state.jobs.send(Job { prompt: req.prompt, max_new: req.max_new, sampling: req.sampling, stop: req.stop, tx }).is_err() {
        return err(StatusCode::INTERNAL_SERVER_ERROR, "GPU worker is not running");
    }
    let id = new_id("chatcmpl-");
    let model = state.opts.model_id.clone();
    let created = now();
    if stream {
        sse_response(rx, id, model, created, include_usage).into_response()
    } else {
        collect_response(rx, id, model, created).await
    }
}

async fn collect_response(mut rx: mpsc::UnboundedReceiver<Out>, id: String, model: String, created: u64) -> Response {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut tool_calls = Vec::new();
    let (mut prompt_tokens, mut cached) = (0, 0);
    while let Some(o) = rx.recv().await {
        match o {
            Out::Started { prompt_tokens: p, cached: c } => {
                prompt_tokens = p;
                cached = c;
            }
            Out::Ev(Event::Content(s)) => content.push_str(&s),
            Out::Ev(Event::Reasoning(s)) => reasoning.push_str(&s),
            Out::Ev(Event::ToolCall { name, arguments }) => {
                tool_calls.push(json!({"id": new_id("call_"), "type": "function", "function": {"name": name, "arguments": arguments}}))
            }
            Out::Ev(Event::End) => {}
            Out::Error(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
            Out::Done { finish, completion_tokens } => {
                let mut msg = json!({"role": "assistant", "content": if content.is_empty() && !tool_calls.is_empty() { Value::Null } else { Value::String(content) }});
                if !reasoning.is_empty() {
                    msg["reasoning_content"] = Value::String(reasoning);
                }
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = Value::Array(tool_calls);
                }
                return Json(json!({"id": id, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": usage(prompt_tokens, cached, completion_tokens)}))
                .into_response();
            }
        }
    }
    err(StatusCode::INTERNAL_SERVER_ERROR, "generation ended unexpectedly")
}

fn sse_response(rx: mpsc::UnboundedReceiver<Out>, id: String, model: String, created: u64, include_usage: bool) -> Sse<impl futures_util::Stream<Item = Result<SseEvent, Infallible>>> {
    let chunk = move |delta: Value, finish: Option<&str>| -> SseEvent {
        SseEvent::default().data(
            json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
            .to_string(),
        )
    };
    let first = chunk(json!({"role": "assistant", "content": ""}), None);
    let mut n_tools = 0usize;
    let (mut prompt_tokens, mut cached) = (0usize, 0usize);
    let usage_tail = include_usage;
    let events = UnboundedReceiverStream::new(rx).flat_map(move |o| {
        let mut out: Vec<SseEvent> = Vec::new();
        match o {
            Out::Started { prompt_tokens: p, cached: c } => {
                prompt_tokens = p;
                cached = c;
            }
            Out::Ev(Event::Content(s)) => out.push(chunk(json!({"content": s}), None)),
            Out::Ev(Event::Reasoning(s)) => out.push(chunk(json!({"reasoning_content": s}), None)),
            Out::Ev(Event::ToolCall { name, arguments }) => {
                out.push(chunk(
                    json!({"tool_calls": [{"index": n_tools, "id": new_id("call_"), "type": "function", "function": {"name": name, "arguments": arguments}}]}),
                    None,
                ));
                n_tools += 1;
            }
            Out::Ev(Event::End) => {}
            Out::Error(e) => out.push(SseEvent::default().data(json!({"error": {"message": e}}).to_string())),
            Out::Done { finish, completion_tokens } => {
                out.push(chunk(json!({}), Some(finish)));
                if usage_tail {
                    out.push(SseEvent::default().data(
                        json!({"object": "chat.completion.chunk", "choices": [], "usage": usage(prompt_tokens, cached, completion_tokens)}).to_string(),
                    ));
                }
                out.push(SseEvent::default().data("[DONE]"));
            }
        }
        futures_util::stream::iter(out.into_iter().map(Ok))
    });
    let stream = futures_util::stream::once(async move { Ok(first) }).chain(events);
    Sse::new(stream).keep_alive(KeepAlive::default())
}
