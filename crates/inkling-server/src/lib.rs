//! OpenAI-compatible HTTP server: `/v1/chat/completions` and `/v1/completions` (streaming SSE or not), `/v1/models`,
//! `/health`. CORS is open (browser UIs); authentication, when configured, is a bearer API key.
//!
//! Images (with `--mmproj`) arrive as content parts (`image_url` / `input_image` / `image`) carrying `data:` URLs
//! or bare base64; each becomes `<|content_image|>` + its embedding rows + `<|end_message|>` in the prompt.
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
use inkling_engine::{EMBED_TOKEN_BASE, Engine, Image, ImageEmbed, Sampling};

mod structured;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;

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
    /// vision encoder (mmproj GGUF)
    pub mmproj: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct Defaults {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
}

impl Default for Defaults {
    fn default() -> Self {
        Self { temperature: 0.8, top_p: 0.95, top_k: 40, min_p: 0.05, repeat_penalty: 1.0, repeat_last_n: 64 }
    }
}

/// Generation request for the GPU worker.
struct Job {
    prompt: Vec<u32>,
    max_new: usize,
    sampling: Sampling,
    stop: Vec<String>,
    /// raw text completion (no chat parsing)
    raw: bool,
    /// images: (prompt position of the first embedding row, image)
    images: Vec<(usize, Arc<Image>)>,
    /// JSON schema the answer must follow (response_format), or None
    schema: Option<Value>,
    tx: mpsc::UnboundedSender<Out>,
}

/// What the worker reports back for one job.
enum Out {
    Started { prompt_tokens: usize, cached: usize },
    Ev(Event),
    Done { finish: &'static str, completion_tokens: usize },
    Error(String),
    /// the request cannot be served as asked (e.g. an unusable JSON schema): 400
    Invalid(String),
}

struct AppState {
    tok: Arc<Tokenizer>,
    jobs: std::sync::mpsc::Sender<Job>,
    opts: Options,
    n_ctx: usize,
    /// (patch, merge) of the loaded vision encoder
    vision: Option<(usize, usize)>,
    image_pad: u32,
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
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(usize, Option<(usize, usize)>)>>();
    let model_path = model.to_string();
    let wtok = tok.clone();
    let n_ctx_req = opts.n_ctx;
    let mmproj = opts.mmproj.clone();
    std::thread::Builder::new().name("inkling-gpu".into()).spawn(move || {
        let eng = Engine::load(&model_path, n_ctx_req).and_then(|mut e| {
            if let Some(m) = &mmproj {
                e.load_vision(m)?;
            }
            Ok(e)
        });
        match eng {
            Ok(mut eng) => {
                let geom = eng.vision().map(|v| (v.patch, v.merge));
                let _ = ready_tx.send(Ok((n_ctx_req, geom)));
                worker(&mut eng, &wtok, jobs_rx);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        }
    })?;
    let (n_ctx, vision) = ready_rx.recv().context("GPU worker died")??;
    let image_pad = tok.token_id("<|unused_200054|>").context("image placeholder token missing")?;
    let state = Arc::new(AppState { tok, jobs: jobs_tx, opts: opts.clone(), n_ctx, vision, image_pad });
    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        // axum refuses bodies over 2 MB by default, about 500k tokens of English: allow a full context of
        // escaped text (up to ~16 bytes per token) and never less than 64 MiB, for images
        .layer(axum::extract::DefaultBodyLimit::max((n_ctx * 16).max(64 << 20)))
        .layer(CorsLayer::permissive())
        .with_state(state);
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let addr = format!("{}:{}", opts.host, opts.port);
        let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("binding {addr}"))?;
        let vis = if vision.is_some() { ", vision on" } else { "" };
        println!("InklingEngine serving {} on http://{addr} (context {n_ctx}{vis}) - OpenAI API at /v1", opts.model_id);
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
}

/// Recently encoded images (conversations resend them every turn), bounded by total size.
#[derive(Default)]
struct EmbedCache {
    items: Vec<ImageEmbed>,
}

impl EmbedCache {
    const MAX_BYTES: usize = 512 << 20;

    fn get_or_encode(&mut self, eng: &Engine, img: &Image) -> Result<ImageEmbed> {
        let h = img.hash();
        if let Some(i) = self.items.iter().position(|e| e.hash == h) {
            let e = self.items.remove(i);
            self.items.push(e.clone());
            return Ok(e);
        }
        let e = eng.encode_image(img)?;
        self.items.push(e.clone());
        while self.items.iter().map(|e| e.data.len() * 4).sum::<usize>() > Self::MAX_BYTES && self.items.len() > 1 {
            self.items.remove(0);
        }
        Ok(e)
    }
}

/// GPU worker: one job at a time.
fn worker(eng: &mut Engine, tok: &Tokenizer, jobs: std::sync::mpsc::Receiver<Job>) {
    let mut embeds = EmbedCache::default();
    // structured output: built once, lazily (the first request with response_format pays for it)
    let mut guidance: Option<Result<structured::Guidance, String>> = None;
    while let Ok(job) = jobs.recv() {
        let tx = job.tx.clone();
        let constraint = match &job.schema {
            None => None,
            Some(schema) => match guidance.get_or_insert_with(|| structured::Guidance::new(tok).map_err(|e| format!("{e:#}"))) {
                Err(e) => {
                    let _ = tx.send(Out::Error(format!("structured output unavailable: {e}")));
                    continue;
                }
                Ok(g) => match g.start(schema.clone()) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        let _ = tx.send(Out::Invalid(format!("{e:#}")));
                        continue;
                    }
                },
            },
        };
        if let Err(e) = run_job(eng, tok, &mut embeds, job, constraint) {
            let _ = tx.send(Out::Error(format!("{e:#}")));
        }
    }
}

/// Turns generated tokens into events: the chat parser, or plain text for raw completions.
enum Decoder {
    Chat(chat::Parser),
    Raw { stop: u32, pending: Vec<u8> },
}

impl Decoder {
    fn new(tok: &Tokenizer, raw: bool) -> Result<Self> {
        Ok(if raw {
            Decoder::Raw { stop: tok.token_id("<|content_model_end_sampling|>").ok_or_else(|| anyhow!("EOS token missing"))?, pending: Vec::new() }
        } else {
            Decoder::Chat(chat::Parser::new(tok)?)
        })
    }

    fn push(&mut self, tok: &Tokenizer, id: u32) -> Vec<Event> {
        match self {
            Decoder::Chat(p) => p.push(tok, id),
            Decoder::Raw { stop, pending } => {
                if id == *stop {
                    return vec![Event::End];
                }
                pending.extend_from_slice(&tok.token_bytes(id, true));
                let valid = match std::str::from_utf8(pending) {
                    Ok(_) => pending.len(),
                    Err(e) if e.error_len().is_none() => e.valid_up_to(),
                    Err(e) => e.valid_up_to() + e.error_len().unwrap(),
                };
                if valid == 0 {
                    return vec![];
                }
                let s = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                vec![Event::Content(s)]
            }
        }
    }
}

fn run_job(eng: &mut Engine, tok: &Tokenizer, embed_cache: &mut EmbedCache, job: Job, constraint: Option<structured::Constrained>) -> Result<()> {
    let constraint = constraint.map(std::cell::RefCell::new);
    let constraint_error: std::cell::RefCell<Option<anyhow::Error>> = std::cell::RefCell::new(None);
    let mut embeds = Vec::with_capacity(job.images.len());
    for (pos, img) in &job.images {
        embeds.push((*pos, embed_cache.get_or_encode(eng, img)?));
    }
    let mut parser = Decoder::new(tok, job.raw)?;
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
            if let Some(c) = &constraint {
                if let Err(e) = c.borrow_mut().push(t) {
                    *constraint_error.borrow_mut() = Some(e);
                    return false;
                }
            }
            on_token(t)
        };
        match &constraint {
            Some(c) => {
                let mut mask = || match c.borrow_mut().mask() {
                    Ok(m) => m,
                    Err(e) => {
                        *constraint_error.borrow_mut() = Some(e);
                        None
                    }
                };
                eng.generate_mm_constrained(&job.prompt, &embeds, job.max_new, &job.sampling, &mut cb, &mut mask)?
            }
            None => eng.generate_mm(&job.prompt, &embeds, job.max_new, &job.sampling, &mut cb)?,
        }
    };
    if let Some(e) = constraint_error.into_inner() {
        return Err(e);
    }
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
    images: Vec<(usize, Arc<Image>)>,
    max_new: usize,
    sampling: Sampling,
    stop: Vec<String>,
    stream: bool,
    include_usage: bool,
    schema: Option<Value>,
}

/// Bytes of an image reference: a `data:` URL or bare base64.
fn image_bytes(s: &str) -> Result<Vec<u8>> {
    use base64::Engine as _;
    let s = s.trim();
    let b64 = if let Some(rest) = s.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',').ok_or_else(|| anyhow!("malformed data: URL"))?;
        if !meta.ends_with(";base64") {
            return Err(anyhow!("data: URLs must be base64-encoded"));
        }
        data
    } else if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("file:") {
        return Err(anyhow!("image URLs are not fetched; send the image inline as a data: URL (base64)"));
    } else {
        s
    };
    let clean: String = b64.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD.decode(clean.trim_end_matches('=')).or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(clean.trim_end_matches('='))).map_err(|e| anyhow!("invalid base64 image: {e}"))
}

/// The image reference of a content part, in the shapes clients send.
fn part_image(part: &Value) -> Option<&str> {
    fn s(v: Option<&Value>) -> Option<&str> {
        v.and_then(|v| v.as_str().or_else(|| v.get("url").and_then(Value::as_str)))
    }
    s(part.get("image_url"))
        .or_else(|| s(part.get("image")))
        .or_else(|| part.get("source").filter(|src| src.get("type").and_then(Value::as_str) == Some("base64")).and_then(|src| src.get("data")).and_then(Value::as_str))
}

/// Images of a chat request in prompt order (the order the chat template renders their placeholders).
fn request_images(messages: &[Value]) -> Result<Vec<Image>> {
    let mut out = Vec::new();
    for m in messages {
        for part in m.get("content").and_then(Value::as_array).into_iter().flatten() {
            if matches!(part.get("type").and_then(Value::as_str), Some("image" | "input_image" | "image_url")) {
                let r = part_image(part).ok_or_else(|| anyhow!("image part without image data"))?;
                out.push(Image::decode(&image_bytes(r)?)?);
            }
        }
    }
    Ok(out)
}

/// Prompt tokens of a request: the rendered chat (`raw` false) or a completion prompt (string or token ids).
fn request_prompt(state: &AppState, body: &Value, raw: bool) -> Result<(Vec<u32>, Vec<(usize, Arc<Image>)>)> {
    if raw {
        let p = body.get("prompt").ok_or_else(|| anyhow!("`prompt` is required"))?;
        let p = match p {
            Value::Array(a) if a.len() == 1 && !a[0].is_number() => &a[0],
            p => p,
        };
        let ids = match p {
            Value::String(s) => state.tok.encode(s, true),
            Value::Array(a) => a
                .iter()
                .map(|v| v.as_u64().filter(|&t| (t as usize) < state.tok.n_vocab()).map(|t| t as u32).ok_or_else(|| anyhow!("invalid token id in `prompt`")))
                .collect(),
            _ => Err(anyhow!("`prompt` must be a string or an array of token ids (one prompt per request)")),
        }?;
        if ids.contains(&state.image_pad) {
            return Err(anyhow!("raw completions do not take images"));
        }
        return Ok((ids, vec![]));
    }
    let messages = body.get("messages").and_then(Value::as_array).ok_or_else(|| anyhow!("`messages` must be an array"))?;
    let tool_choice_none = body.get("tool_choice").and_then(Value::as_str) == Some("none");
    let tools = if tool_choice_none { None } else { body.get("tools").and_then(Value::as_array).map(|v| v.as_slice()) };
    let text = chat::render(messages, tools, body.get("reasoning_effort"))?;
    let ids = state.tok.encode(&text, true)?;
    let images = request_images(messages)?;
    let n_pad = ids.iter().filter(|&&t| t == state.image_pad).count();
    if images.is_empty() && n_pad == 0 {
        return Ok((ids, vec![]));
    }
    let (patch, merge) = state.vision.ok_or_else(|| anyhow!("this server has no vision encoder (start it with --mmproj)"))?;
    if n_pad != images.len() {
        return Err(anyhow!("{} image placeholders for {} images", n_pad, images.len()));
    }
    // each placeholder becomes the image's embedding rows; their ids identify the image for the prompt cache
    let mut out = Vec::with_capacity(ids.len());
    let mut placed = Vec::with_capacity(images.len());
    let mut it = images.into_iter();
    for t in ids {
        if t != state.image_pad {
            out.push(t);
            continue;
        }
        let img = it.next().expect("counted");
        let n = inkling_engine::vision::image_tokens(patch, merge, img.width, img.height);
        let h = img.hash() as u32;
        placed.push((out.len(), Arc::new(img)));
        out.extend((0..n as u32).map(|r| EMBED_TOKEN_BASE | (h.wrapping_add(r) & 0x7fff_ffff)));
    }
    Ok((out, placed))
}

fn parse_request(state: &AppState, body: &Value, raw: bool) -> Result<Req> {
    if body.get("n").and_then(Value::as_u64).unwrap_or(1) != 1 {
        return Err(anyhow!("only n = 1 is supported"));
    }
    let (prompt, images) = request_prompt(state, body, raw)?;
    if prompt.is_empty() {
        return Err(anyhow!("empty prompt"));
    }
    if prompt.len() + 1 > state.n_ctx {
        return Err(anyhow!("prompt is {} tokens; the context is {}", prompt.len(), state.n_ctx));
    }
    let room = state.n_ctx - prompt.len();
    let max_new = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(Value::as_u64)
        .map(|v| (v as usize).clamp(1, room))
        .unwrap_or(if raw { room.min(16) } else { room });
    let d = state.opts.defaults;
    let f = |k: &str, dv: f32| body.get(k).and_then(Value::as_f64).map(|v| v as f32).unwrap_or(dv);
    let seed = body.get("seed").and_then(Value::as_u64).unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1));
    let sampling = Sampling {
        temperature: f("temperature", d.temperature),
        top_p: f("top_p", d.top_p),
        min_p: f("min_p", d.min_p),
        top_k: body.get("top_k").and_then(Value::as_i64).map(|v| v as i32).unwrap_or(d.top_k),
        seed,
        repeat_penalty: body.get("repeat_penalty").or_else(|| body.get("repetition_penalty")).and_then(Value::as_f64).map(|v| v as f32).unwrap_or(d.repeat_penalty),
        presence_penalty: f("presence_penalty", 0.0),
        frequency_penalty: f("frequency_penalty", 0.0),
        last_n: body.get("repeat_last_n").and_then(Value::as_i64).map(|v| v.clamp(0, 1024) as i32).unwrap_or(d.repeat_last_n),
    };
    if !(sampling.repeat_penalty > 0.0) {
        return Err(anyhow!("repeat_penalty must be positive"));
    }
    let stop = match body.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).filter(|s| !s.is_empty()).collect(),
        _ => vec![],
    };
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(Value::as_bool).unwrap_or(false);
    let schema = if raw { None } else { structured::requested_schema(body)? };
    Ok(Req { prompt, images, max_new, sampling, stop, stream, include_usage, schema })
}

fn usage(prompt_tokens: usize, cached: usize, completion: usize) -> Value {
    json!({"prompt_tokens": prompt_tokens, "completion_tokens": completion, "total_tokens": prompt_tokens + completion,
        "prompt_tokens_details": {"cached_tokens": cached}})
}

async fn chat_completions(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    start(state, headers, body, false).await
}

async fn completions(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    start(state, headers, body, true).await
}

async fn start(state: Arc<AppState>, headers: HeaderMap, body: Value, raw: bool) -> Response {
    if !authorized(&state, &headers) {
        return err(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    let req = match parse_request(&state, &body, raw) {
        Ok(r) => r,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    let (tx, rx) = mpsc::unbounded_channel();
    let stream = req.stream;
    let include_usage = req.include_usage;
    if state.jobs.send(Job { prompt: req.prompt, images: req.images, max_new: req.max_new, sampling: req.sampling, stop: req.stop, raw, schema: req.schema, tx }).is_err() {
        return err(StatusCode::INTERNAL_SERVER_ERROR, "GPU worker is not running");
    }
    let id = new_id(if raw { "cmpl-" } else { "chatcmpl-" });
    let model = state.opts.model_id.clone();
    let created = now();
    match (raw, stream) {
        (false, true) => sse_response(rx, id, model, created, include_usage).into_response(),
        (false, false) => collect_response(rx, id, model, created).await,
        (true, true) => sse_text(rx, id, model, created, include_usage).into_response(),
        (true, false) => collect_text(rx, id, model, created).await,
    }
}

async fn collect_text(mut rx: mpsc::UnboundedReceiver<Out>, id: String, model: String, created: u64) -> Response {
    let mut text = String::new();
    let (mut prompt_tokens, mut cached) = (0, 0);
    while let Some(o) = rx.recv().await {
        match o {
            Out::Started { prompt_tokens: p, cached: c } => {
                prompt_tokens = p;
                cached = c;
            }
            Out::Ev(Event::Content(s) | Event::Reasoning(s)) => text.push_str(&s),
            Out::Ev(_) => {}
            Out::Error(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
            Out::Invalid(e) => return err(StatusCode::BAD_REQUEST, e),
            Out::Done { finish, completion_tokens } => {
                return Json(json!({"id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": finish}],
                    "usage": usage(prompt_tokens, cached, completion_tokens)}))
                .into_response();
            }
        }
    }
    err(StatusCode::INTERNAL_SERVER_ERROR, "generation ended unexpectedly")
}

fn sse_text(rx: mpsc::UnboundedReceiver<Out>, id: String, model: String, created: u64, include_usage: bool) -> Sse<impl futures_util::Stream<Item = Result<SseEvent, Infallible>>> {
    let chunk = move |text: &str, finish: Option<&str>| -> SseEvent {
        SseEvent::default().data(
            json!({"id": id, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": finish}]})
            .to_string(),
        )
    };
    let (mut prompt_tokens, mut cached) = (0usize, 0usize);
    let events = UnboundedReceiverStream::new(rx).flat_map(move |o| {
        let mut out: Vec<SseEvent> = Vec::new();
        match o {
            Out::Started { prompt_tokens: p, cached: c } => {
                prompt_tokens = p;
                cached = c;
            }
            Out::Ev(Event::Content(s) | Event::Reasoning(s)) => out.push(chunk(&s, None)),
            Out::Ev(_) => {}
            Out::Error(e) | Out::Invalid(e) => out.push(SseEvent::default().data(json!({"error": {"message": e}}).to_string())),
            Out::Done { finish, completion_tokens } => {
                out.push(chunk("", Some(finish)));
                if include_usage {
                    out.push(SseEvent::default().data(json!({"object": "text_completion", "choices": [], "usage": usage(prompt_tokens, cached, completion_tokens)}).to_string()));
                }
                out.push(SseEvent::default().data("[DONE]"));
            }
        }
        futures_util::stream::iter(out.into_iter().map(Ok))
    });
    Sse::new(events).keep_alive(KeepAlive::default())
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
            Out::Invalid(e) => return err(StatusCode::BAD_REQUEST, e),
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
            Out::Error(e) | Out::Invalid(e) => out.push(SseEvent::default().data(json!({"error": {"message": e}}).to_string())),
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
