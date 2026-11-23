//! OpenAI-compatible HTTP server: `/v1/chat/completions` and `/v1/completions` (streaming SSE or not), `/v1/models`,
//! `/health`. Authentication uses a bearer API key unless explicitly disabled; no CORS is enabled.
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
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use axum::Router;
use axum::body::Body;
use axum::extract::{Json, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::StreamExt;
use inkling_core::chat::{self, Event};
use inkling_core::{Gguf, Tokenizer};
use inkling_engine::{EMBED_TOKEN_BASE, Engine, Image, ImageEmbed, Sampling};
use serde_json::{Value, json};
use tokio::sync::{Semaphore, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Server options.
#[derive(Clone, Debug)]
pub struct Options {
    pub host: String,
    pub port: u16,
    pub n_ctx: usize,
    pub api_key: Option<String>,
    /// explicitly allow serving without an API key
    pub insecure_no_auth: bool,
    /// model id reported by /v1/models (default: file name)
    pub model_id: String,
    /// sampling defaults for requests that do not set them
    pub defaults: Defaults,
    /// vision encoder (mmproj GGUF)
    pub mmproj: Option<String>,
    /// maximum reasoning tokens per chat request (None: unlimited)
    pub reasoning_budget: Option<usize>,
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
    slot: JobSlot,
    prompt: Vec<u32>,
    max_new: usize,
    reasoning_budget: Option<usize>,
    sampling: Sampling,
    stop: Vec<String>,
    /// raw text completion (no chat parsing)
    raw: bool,
    /// images: (prompt position of the first embedding row, image)
    images: Vec<(usize, Arc<Image>)>,
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
    jobs: std::sync::mpsc::SyncSender<Job>,
    gpu_thread: std::thread::JoinHandle<()>,
    admission: BodyAdmission,
    builds: Arc<Semaphore>,
    opts: Options,
    n_ctx: usize,
    /// (patch, merge) of the loaded vision encoder
    vision: Option<(usize, usize)>,
    image_pad: u32,
    force_ids: Option<[u32; 3]>,
}

const MAX_PENDING: usize = 8;
const MAX_BODY_BYTES: usize = 8 << 20;

struct BodyLimits {
    max_bytes: usize,
    total: Duration,
    idle: Duration,
    grace: Duration,
    min_rate: usize,
}

impl Default for BodyLimits {
    fn default() -> Self {
        Self {
            max_bytes: MAX_BODY_BYTES,
            total: Duration::from_secs(15),
            idle: Duration::from_secs(3),
            grace: Duration::from_secs(2),
            min_rate: 64 << 10,
        }
    }
}

struct BodyAdmission {
    readers: Semaphore,
    limits: BodyLimits,
    pending: Arc<AtomicUsize>,
}

const MAX_IMAGE_BYTES: usize = 512 << 20;
static IMAGE_BYTES: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct ImageQueueFull;

impl std::fmt::Display for ImageQueueFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("decoded image queue is full")
    }
}

impl std::error::Error for ImageQueueFull {}

#[derive(Debug)]
struct ImageBytes<'a> {
    used: &'a AtomicUsize,
    bytes: usize,
}

impl<'a> ImageBytes<'a> {
    fn reserve(used: &'a AtomicUsize, width: usize, height: usize) -> Result<Self> {
        // Cover the decoded image (up to RGBA8) and its simultaneous RGB8 conversion.
        let bytes = width.checked_mul(height).and_then(|n| n.checked_mul(7)).ok_or(ImageQueueFull)?;
        used.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            n.checked_add(bytes).filter(|&total| total <= MAX_IMAGE_BYTES)
        }).map_err(|_| ImageQueueFull)?;
        Ok(Self { used, bytes })
    }
}

impl Drop for ImageBytes<'_> {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

struct JobSlot {
    pending: Arc<AtomicUsize>,
    images: Vec<ImageBytes<'static>>,
}

impl JobSlot {
    fn reserve(pending: &Arc<AtomicUsize>) -> Option<Self> {
        pending.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            if n < MAX_PENDING { Some(n + 1) } else { None }
        }).ok().map(|_| Self { pending: pending.clone(), images: Vec::new() })
    }
}

impl Drop for JobSlot {
    fn drop(&mut self) {
        self.pending.fetch_sub(1, Ordering::Relaxed);
    }
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
    if !opts.api_key.as_deref().is_some_and(|key| !key.is_empty()) && !opts.insecure_no_auth {
        return Err(anyhow!("an API key is required (--api-key or INKLING_API_KEY); use --insecure-no-auth to disable authentication"));
    }
    let tok = Arc::new(Tokenizer::from_gguf(&Gguf::open(model)?)?);
    let force_ids = tok.token_id("<|end_message|>")
        .zip(tok.token_id("<|message_model|>"))
        .zip(tok.token_id("<|content_text|>"))
        .map(|((end, model), text)| [end, model, text]);
    validate_force_ids(opts.reasoning_budget, force_ids)?;
    let (jobs_tx, jobs_rx) = std::sync::mpsc::sync_channel::<Job>(MAX_PENDING);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(usize, Option<(usize, usize)>)>>();
    let model_path = model.to_string();
    let wtok = tok.clone();
    let n_ctx_req = opts.n_ctx;
    let mmproj = opts.mmproj.clone();
    let gpu_thread = std::thread::Builder::new().name("inkling-gpu".into()).spawn(move || {
        let eng = Engine::load(&model_path, n_ctx_req).and_then(|mut e| {
            if let Some(m) = &mmproj {
                e.load_vision(m)?;
            }
            Ok(e)
        });
        match eng {
            Ok(mut eng) => {
                let geom = eng.vision().map(|v| (v.patch, v.merge));
                let _ = ready_tx.send(Ok((eng.n_ctx, geom)));
                worker(&mut eng, &wtok, jobs_rx, force_ids);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        }
    })?;
    let (n_ctx, vision) = ready_rx.recv().context("GPU worker died")??;
    let image_pad = tok.token_id("<|unused_200054|>").context("image placeholder token missing")?;
    let state = Arc::new(AppState {
        tok, jobs: jobs_tx, gpu_thread,
        admission: BodyAdmission { readers: Semaphore::new(2 * MAX_PENDING), limits: BodyLimits::default(), pending: Arc::new(AtomicUsize::new(0)) },
        builds: Arc::new(Semaphore::new(2)),
        opts: opts.clone(), n_ctx, vision, image_pad, force_ids,
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(middleware::from_fn_with_state(state.clone(), authenticate))
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
fn worker(eng: &mut Engine, tok: &Tokenizer, jobs: std::sync::mpsc::Receiver<Job>, force_ids: Option<[u32; 3]>) {
    let mut embeds = EmbedCache::default();
    while let Ok(job) = jobs.recv() {
        let tx = job.tx.clone();
        if let Err(e) = run_job(eng, tok, &mut embeds, job, force_ids) {
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
    fn in_thinking(&self) -> bool {
        matches!(self, Decoder::Chat(p) if p.in_thinking())
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReasoningAction {
    Continue,
    ForceClose,
    Length,
}

fn reasoning_action(in_thinking: bool, count: usize, budget: Option<usize>, forced_once: bool) -> ReasoningAction {
    if !in_thinking {
        ReasoningAction::Continue
    } else if forced_once {
        ReasoningAction::Length
    } else if budget.is_some_and(|limit| count >= limit) {
        ReasoningAction::ForceClose
    } else {
        ReasoningAction::Continue
    }
}

fn forced_token_count(prompt_len: usize, n_ctx: usize, max_new: usize, generated: usize) -> usize {
    3.min(n_ctx.saturating_sub(prompt_len)).min(max_new.saturating_sub(generated))
}

fn validate_force_ids(budget: Option<usize>, force_ids: Option<[u32; 3]>) -> Result<()> {
    if budget.is_some() && force_ids.is_none() {
        return Err(anyhow!("reasoning budget requires <|end_message|>, <|message_model|> and <|content_text|> tokens"));
    }
    Ok(())
}

fn run_job(eng: &mut Engine, tok: &Tokenizer, embed_cache: &mut EmbedCache, job: Job, force_ids: Option<[u32; 3]>) -> Result<()> {
    let _slot = job.slot;
    let mut embeds = Vec::with_capacity(job.images.len());
    for (pos, img) in &job.images {
        embeds.push((*pos, embed_cache.get_or_encode(eng, img)?));
    }
    let mut parser = Decoder::new(tok, job.raw)?;
    let reasoning_budget = if job.raw { None } else { job.reasoning_budget };
    let tx = job.tx;
    let stop = job.stop;
    let hold = stop.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut finish: Option<&'static str> = None;
    let mut tool_called = false;
    let mut n = 0usize;
    let mut thinking_tokens = 0usize;
    let mut forced_once = false;
    let mut length_limited = false;
    let mut held = String::new(); // content not yet sent (may contain the start of a stop string)
    let mut on_events = |events: Vec<Event>| -> bool {
        for ev in events {
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
    let mut prompt = job.prompt;
    'generating: while n < job.max_new {
        let remaining = job.max_new.saturating_sub(n).min(eng.n_ctx.saturating_sub(prompt.len()));
        if remaining == 0 {
            length_limited = true;
            break;
        }
        let mut action = ReasoningAction::Continue;
        let (out, cached) = eng.generate_mm(&prompt, &embeds, remaining, &job.sampling, &mut |t| {
            let was_thinking = parser.in_thinking();
            let events = parser.push(tok, t);
            // Count reasoning content, excluding the opening and closing tokens.
            if was_thinking && parser.in_thinking() {
                thinking_tokens += 1;
            }
            if !on_events(events) {
                return false;
            }
            action = reasoning_action(parser.in_thinking(), thinking_tokens, reasoning_budget, forced_once);
            action == ReasoningAction::Continue
        })?;
        if n == 0 {
            let _ = tx.send(Out::Started { prompt_tokens, cached });
        }
        n += out.len();
        match action {
            ReasoningAction::Continue => {
                length_limited = out.len() == remaining;
                break;
            }
            ReasoningAction::Length => {
                length_limited = true;
                break;
            }
            ReasoningAction::ForceClose => forced_once = true,
        }
        prompt.extend(out);
        // Forced headers consume the same output/context allowance as sampled tokens.
        let force_ids = force_ids.context("reasoning budget closure tokens missing")?;
        let count = forced_token_count(prompt.len(), eng.n_ctx, job.max_new, n);
        for id in force_ids.into_iter().take(count) {
            prompt.push(id);
            n += 1;
            if !on_events(parser.push(tok, id)) {
                break 'generating;
            }
        }
        if count < force_ids.len() {
            length_limited = true;
            break;
        }
    }
    if finish.is_none() && !held.is_empty() {
        let _ = tx.send(Out::Ev(Event::Content(std::mem::take(&mut held))));
    }
    let finish = finish.unwrap_or(if length_limited || n >= job.max_new { "length" } else { "stop" });
    let _ = tx.send(Out::Done { finish, completion_tokens: n });
    Ok(())
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error"}}))).into_response()
}

fn keys_equal(expected: &[u8], supplied: &[u8]) -> bool {
    let mut diff = usize::from(expected.len() != supplied.len());
    for (i, &a) in expected.iter().enumerate() {
        diff |= usize::from(a ^ supplied.get(i).copied().unwrap_or(0));
    }
    std::hint::black_box(diff) == 0
}

fn authorized(api_key: Option<&str>, insecure_no_auth: bool, headers: &HeaderMap) -> bool {
    let Some(key) = api_key.filter(|key| !key.is_empty()) else { return insecure_no_auth };
    let authorization = headers.get_all("authorization");
    let mut values = authorization.iter();
    let Some(value) = values.next().and_then(|v| v.to_str().ok()) else { return false };
    if values.next().is_some() {
        return false;
    }
    value.trim().strip_prefix("Bearer ").is_some_and(|supplied| keys_equal(key.as_bytes(), supplied.as_bytes()))
}

async fn authenticate(State(state): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    if request.uri().path() != "/health" && !authorized(state.opts.api_key.as_deref(), state.opts.insecure_no_auth, request.headers()) {
        return err(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    next.run(request).await
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    if state.gpu_thread.is_finished() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "GPU worker is not running");
    }
    Json(json!({"status": "ok"})).into_response()
}

async fn models(State(state): State<Arc<AppState>>) -> Response {
    Json(json!({"object": "list", "data": [{"id": state.opts.model_id, "object": "model", "created": now(), "owned_by": "inklingengine",
        "context_length": state.n_ctx}]}))
    .into_response()
}

/// Parsed request pieces.
struct Req {
    prompt: Vec<u32>,
    images: Vec<(usize, Arc<Image>)>,
    max_new: usize,
    reasoning_budget: Option<usize>,
    sampling: Sampling,
    stop: Vec<String>,
    stream: bool,
    include_usage: bool,
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
fn request_image_parts(messages: &[Value], has_vision: bool) -> Result<Vec<&Value>> {
    let parts: Vec<_> = messages.iter()
        .flat_map(|m| m.get("content").and_then(Value::as_array).into_iter().flatten())
        .filter(|part| matches!(part.get("type").and_then(Value::as_str), Some("image" | "input_image" | "image_url")))
        .collect();
    if parts.len() > 8 {
        return Err(anyhow!("at most 8 images are allowed per request"));
    }
    if !parts.is_empty() && !has_vision {
        return Err(anyhow!("this server has no vision encoder (start it with --mmproj)"));
    }
    Ok(parts)
}

fn request_images(parts: &[&Value], slot: &mut JobSlot) -> Result<Vec<Image>> {
    let mut out = Vec::new();
    for part in parts {
        let r = part_image(part).ok_or_else(|| anyhow!("image part without image data"))?;
        let bytes = image_bytes(r)?;
        let (width, height) = Image::dimensions(&bytes)?;
        slot.images.push(ImageBytes::reserve(&IMAGE_BYTES, width, height)?);
        let image = Image::decode(&bytes)?;
        out.push(image);
    }
    Ok(out)
}

/// Prompt tokens of a request: the rendered chat (`raw` false) or a completion prompt (string or token ids).
fn request_prompt(state: &AppState, body: &Value, raw: bool, slot: &mut JobSlot) -> Result<(Vec<u32>, Vec<(usize, Arc<Image>)>)> {
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
    let image_parts = request_image_parts(messages, state.vision.is_some())?;
    let tool_choice_none = body.get("tool_choice").and_then(Value::as_str) == Some("none");
    let tools = if tool_choice_none { None } else { body.get("tools").and_then(Value::as_array).map(|v| v.as_slice()) };
    let text = chat::render(messages, tools, body.get("reasoning_effort"))?;
    let ids = state.tok.encode(&text, true)?;
    let images = request_images(&image_parts, slot)?;
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

fn request_reasoning_budget(body: &Value, raw: bool, default: Option<usize>) -> Result<Option<usize>> {
    if raw {
        return Ok(None);
    }
    match body.get("thinking_token_budget") {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_u64().and_then(|n| usize::try_from(n).ok()).map(Some)
            .ok_or_else(|| anyhow!("thinking_token_budget must be a non-negative integer")),
    }
}

fn request_stop(body: &Value) -> Result<Vec<String>> {
    let values = match body.get("stop") {
        None | Some(Value::Null) => return Ok(vec![]),
        Some(Value::String(s)) => vec![s.as_str()],
        Some(Value::Array(a)) => {
            if a.len() > 16 {
                return Err(anyhow!("stop accepts at most 16 strings"));
            }
            a.iter().map(|v| v.as_str().ok_or_else(|| anyhow!("stop must contain only strings"))).collect::<Result<Vec<_>>>()?
        }
        _ => return Err(anyhow!("stop must be a string or an array of strings")),
    };
    if values.iter().any(|s| s.len() > 256) {
        return Err(anyhow!("stop strings must be at most 256 bytes"));
    }
    Ok(values.into_iter().filter(|s| !s.is_empty()).map(str::to_owned).collect())
}

fn request_max_tokens(body: &Value) -> Result<Option<usize>> {
    let mut selected = None;
    for key in ["max_completion_tokens", "max_tokens"] {
        let Some(value) = body.get(key).filter(|v| !v.is_null()) else { continue };
        let n = value.as_u64().and_then(|n| usize::try_from(n).ok()).filter(|&n| n > 0)
            .ok_or_else(|| anyhow!("{key} must be a positive integer"))?;
        selected.get_or_insert(n);
    }
    Ok(selected)
}

fn parse_request(state: &AppState, body: &Value, raw: bool, slot: &mut JobSlot) -> Result<Req> {
    if body.get("n").and_then(Value::as_u64).unwrap_or(1) != 1 {
        return Err(anyhow!("only n = 1 is supported"));
    }
    let reasoning_budget = request_reasoning_budget(body, raw, state.opts.reasoning_budget)?;
    validate_force_ids(reasoning_budget, state.force_ids)?;
    let stop = request_stop(body)?;
    let requested_max = request_max_tokens(body)?;
    let (prompt, images) = request_prompt(state, body, raw, slot)?;
    if prompt.is_empty() {
        return Err(anyhow!("empty prompt"));
    }
    if prompt.len() >= state.n_ctx {
        return Err(anyhow!("prompt is {} tokens; the context is {}", prompt.len(), state.n_ctx));
    }
    let room = state.n_ctx - prompt.len();
    let max_new = requested_max.map(|v| v.min(room))
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
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(Value::as_bool).unwrap_or(false);
    Ok(Req { prompt, images, max_new, reasoning_budget, sampling, stop, stream, include_usage })
}

fn usage(prompt_tokens: usize, cached: usize, completion: usize) -> Value {
    json!({"prompt_tokens": prompt_tokens, "completion_tokens": completion, "total_tokens": prompt_tokens + completion,
        "prompt_tokens_details": {"cached_tokens": cached}})
}

fn json_content_type(headers: &HeaderMap) -> bool {
    let content_type = headers.get_all("content-type");
    let mut values = content_type.iter();
    let Some(value) = values.next().and_then(|v| v.to_str().ok()) else { return false };
    if values.next().is_some() {
        return false;
    }
    let media_type = value.split(';').next().unwrap_or_default().trim();
    let Some((kind, subtype)) = media_type.split_once('/') else { return false };
    if !kind.eq_ignore_ascii_case("application") || subtype.is_empty() || !subtype.bytes().all(|b| {
        b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)
    }) {
        return false;
    }
    subtype.eq_ignore_ascii_case("json")
        || subtype.rsplit_once('+').is_some_and(|(name, suffix)| !name.is_empty() && suffix.eq_ignore_ascii_case("json"))
}

async fn chat_completions(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Body) -> Response {
    start(state, headers, body, false).await
}

async fn completions(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Body) -> Response {
    start(state, headers, body, true).await
}

fn request_content_length(headers: &HeaderMap, max_bytes: usize) -> Result<Option<u64>, StatusCode> {
    let content_length = headers.get_all("content-length");
    let mut values = content_length.iter();
    let Some(value) = values.next() else { return Ok(None) };
    if values.next().is_some() {
        return Err(StatusCode::BAD_REQUEST);
    }
    let value = value.to_str().map_err(|_| StatusCode::BAD_REQUEST)?;
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(StatusCode::BAD_REQUEST);
    }
    let length = value.parse::<u64>().map_err(|_| StatusCode::PAYLOAD_TOO_LARGE)?;
    if length > max_bytes as u64 {
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }
    Ok(Some(length))
}

#[derive(Debug, PartialEq, Eq)]
enum BodyReadError {
    TooLarge,
    TimedOut,
    Read,
}

impl IntoResponse for BodyReadError {
    fn into_response(self) -> Response {
        match self {
            Self::TooLarge => err(StatusCode::PAYLOAD_TOO_LARGE, "request body exceeds the size limit"),
            Self::TimedOut => err(StatusCode::REQUEST_TIMEOUT, "request body read timed out"),
            Self::Read => err(StatusCode::BAD_REQUEST, "could not read request body"),
        }
    }
}

async fn read_body(body: Body, limits: &BodyLimits) -> Result<Vec<u8>, BodyReadError> {
    let started = tokio::time::Instant::now();
    let mut stream = body.into_data_stream();
    let mut bytes = Vec::new();
    loop {
        let rate_nanos = (bytes.len() as u128).saturating_mul(1_000_000_000) / limits.min_rate.max(1) as u128;
        let deadline = limits.total.min(limits.grace.saturating_add(
            Duration::from_nanos(rate_nanos.min(u64::MAX as u128) as u64),
        ));
        let remaining = deadline.checked_sub(started.elapsed()).filter(|d| !d.is_zero())
            .ok_or(BodyReadError::TimedOut)?;
        let waiting = tokio::time::Instant::now();
        let chunk = tokio::time::timeout(limits.idle.min(remaining), stream.next()).await
            .map_err(|_| BodyReadError::TimedOut)?;
        let Some(chunk) = chunk else { return Ok(bytes) };
        if started.elapsed() >= deadline || waiting.elapsed() >= limits.idle {
            return Err(BodyReadError::TimedOut);
        }
        let chunk = chunk.map_err(|_| BodyReadError::Read)?;
        if chunk.len() > limits.max_bytes - bytes.len() {
            return Err(BodyReadError::TooLarge);
        }
        bytes.extend_from_slice(&chunk);
    }
}

async fn admit_body(admission: &BodyAdmission, headers: &HeaderMap, body: Body) -> Result<Vec<u8>, Response> {
    if !json_content_type(headers) {
        return Err(err(StatusCode::UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json or application/*+json"));
    }
    if let Err(status) = request_content_length(headers, admission.limits.max_bytes) {
        return Err(if status == StatusCode::PAYLOAD_TOO_LARGE {
            BodyReadError::TooLarge.into_response()
        } else {
            err(status, "Content-Length must be a single non-negative integer")
        });
    }
    if admission.pending.load(Ordering::Relaxed) >= MAX_PENDING {
        return Err(err(StatusCode::SERVICE_UNAVAILABLE, "GPU queue is full"));
    }
    let _read_permit = match tokio::time::timeout(Duration::from_millis(250), admission.readers.acquire()).await {
        Ok(Ok(permit)) => permit,
        _ => return Err(err(StatusCode::SERVICE_UNAVAILABLE, "request body readers are busy")),
    };
    read_body(body, &admission.limits).await.map_err(IntoResponse::into_response)
}

async fn start(state: Arc<AppState>, headers: HeaderMap, body: Body, raw: bool) -> Response {
    if state.gpu_thread.is_finished() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "GPU worker is not running");
    }
    let body = match admit_body(&state.admission, &headers, body).await {
        Ok(body) => body,
        Err(response) => return response,
    };
    // Reserve only after reading so slow uploads cannot occupy GPU queue slots; recheck races atomically.
    let Some(mut slot) = JobSlot::reserve(&state.admission.pending) else {
        return err(StatusCode::SERVICE_UNAVAILABLE, "GPU queue is full");
    };
    let permit = match tokio::time::timeout(Duration::from_secs(2), state.builds.clone().acquire_owned()).await {
        Ok(Ok(permit)) => permit,
        _ => return err(StatusCode::SERVICE_UNAVAILABLE, "request preparation is busy"),
    };
    let build_state = state.clone();
    let (req, slot) = match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let body: Value = serde_json::from_slice(&body).context("invalid JSON")?;
        let req = parse_request(&build_state, &body, raw, &mut slot)?;
        Ok::<_, anyhow::Error>((req, slot))
    }).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            let status = if e.chain().any(|c| c.is::<ImageQueueFull>()) { StatusCode::SERVICE_UNAVAILABLE } else { StatusCode::BAD_REQUEST };
            return err(status, format!("{e:#}"));
        }
        Err(_) => return err(StatusCode::INTERNAL_SERVER_ERROR, "request preparation failed"),
    };
    let (tx, rx) = mpsc::unbounded_channel();
    let stream = req.stream;
    let include_usage = req.include_usage;
    if let Err(e) = state.jobs.try_send(Job { slot, prompt: req.prompt, images: req.images, max_new: req.max_new, reasoning_budget: req.reasoning_budget, sampling: req.sampling, stop: req.stop, raw, tx }) {
        return match e {
            std::sync::mpsc::TrySendError::Full(_) => err(StatusCode::SERVICE_UNAVAILABLE, "GPU queue is full"),
            std::sync::mpsc::TrySendError::Disconnected(_) => err(StatusCode::SERVICE_UNAVAILABLE, "GPU worker is not running"),
        };
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
            Out::Error(e) => out.push(SseEvent::default().data(json!({"error": {"message": e}}).to_string())),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn small_body_limits() -> BodyLimits {
        BodyLimits {
            max_bytes: 16,
            total: Duration::from_millis(2000),
            idle: Duration::from_millis(1000),
            grace: Duration::from_millis(100),
            min_rate: 1000,
        }
    }

    #[tokio::test]
    async fn body_reader_cuts_off_slow_drips_at_the_rate_deadline() {
        let limits = small_body_limits();
        let stream = futures_util::stream::unfold(0, |n| async move {
            tokio::time::sleep(Duration::from_millis(40)).await;
            Some((Ok::<_, std::io::Error>("x"), n + 1))
        });
        let started = tokio::time::Instant::now();
        assert_eq!(read_body(Body::from_stream(stream), &limits).await, Err(BodyReadError::TimedOut));
        assert!(started.elapsed() < limits.idle);
    }

    #[tokio::test]
    async fn body_reader_cuts_off_a_missing_chunk_at_the_idle_deadline() {
        let limits = BodyLimits { idle: Duration::from_millis(40), grace: Duration::from_millis(1000), ..small_body_limits() };
        let stream = futures_util::stream::pending::<Result<&'static str, std::io::Error>>();
        let started = tokio::time::Instant::now();
        assert_eq!(read_body(Body::from_stream(stream), &limits).await, Err(BodyReadError::TimedOut));
        assert!(started.elapsed() < limits.grace);
    }

    #[tokio::test]
    async fn body_reader_caps_the_total_deadline() {
        let limits = BodyLimits { total: Duration::from_millis(40), grace: Duration::from_millis(1000), ..small_body_limits() };
        let stream = futures_util::stream::pending::<Result<&'static str, std::io::Error>>();
        let started = tokio::time::Instant::now();
        assert_eq!(read_body(Body::from_stream(stream), &limits).await, Err(BodyReadError::TimedOut));
        assert!(started.elapsed() < limits.grace);
    }

    #[tokio::test]
    async fn body_reader_rejects_chunked_bodies_above_the_limit() {
        let limits = small_body_limits();
        let chunks = [
            Ok::<_, std::io::Error>(axum::body::Bytes::from(vec![0; limits.max_bytes / 2])),
            Ok(axum::body::Bytes::from(vec![0; limits.max_bytes / 2])),
            Ok(axum::body::Bytes::from_static(b"x")),
        ];
        let body = Body::from_stream(futures_util::stream::iter(chunks));
        assert_eq!(read_body(body, &limits).await, Err(BodyReadError::TooLarge));
    }

    #[tokio::test]
    async fn body_reader_returns_valid_bytes_in_order() {
        let limits = small_body_limits();
        let chunks = [Ok::<_, std::io::Error>("hello "), Ok("world")];
        let body = Body::from_stream(futures_util::stream::iter(chunks));
        assert_eq!(read_body(body, &limits).await.unwrap(), b"hello world");
        assert!(read_body(Body::empty(), &limits).await.unwrap().is_empty());
        assert_eq!(read_body(Body::from(vec![0; limits.max_bytes]), &limits).await.unwrap().len(), limits.max_bytes);
    }

    #[tokio::test]
    async fn body_reader_accepts_a_zero_minimum_rate_without_panicking() {
        let limits = BodyLimits { min_rate: 0, ..small_body_limits() };
        let chunks = [Ok::<_, std::io::Error>("hello "), Ok("world")];
        let body = Body::from_stream(futures_util::stream::iter(chunks));
        assert_eq!(read_body(body, &limits).await.unwrap(), b"hello world");
    }

    #[tokio::test]
    async fn body_reader_returns_a_typed_stream_error_without_transport_details() {
        let chunks = [
            Ok(axum::body::Bytes::from_static(b"partial")),
            Err(std::io::Error::other("private transport details")),
        ];
        let body = Body::from_stream(futures_util::stream::iter(chunks));
        let error = read_body(body, &small_body_limits()).await.unwrap_err();
        assert_eq!(error, BodyReadError::Read);
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["message"], "could not read request body");
        assert!(!String::from_utf8_lossy(&bytes).contains("private transport details"));
    }

    struct AdmissionServer {
        addr: std::net::SocketAddr,
        admission: Arc<BodyAdmission>,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for AdmissionServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    impl AdmissionServer {
        async fn new() -> Self {
            async fn route(State(admission): State<Arc<BodyAdmission>>, headers: HeaderMap, body: Body) -> Response {
                match admit_body(&admission, &headers, body).await {
                    Ok(bytes) => (StatusCode::OK, bytes).into_response(),
                    Err(response) => response,
                }
            }
            let admission = Arc::new(BodyAdmission {
                readers: Semaphore::new(2 * MAX_PENDING),
                limits: small_body_limits(),
                pending: Arc::new(AtomicUsize::new(0)),
            });
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let app = Router::new().route("/", post(route)).with_state(admission.clone());
            let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            Self { addr, admission, task }
        }

        fn request(headers: &str, body: &str) -> Vec<u8> {
            format!("POST / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n{headers}\r\n{body}").into_bytes()
        }

        async fn exchange(&self, parts: Vec<(Duration, Vec<u8>)>) -> String {
            let addr = self.addr;
            tokio::task::spawn_blocking(move || {
                use std::io::{Read, Write};
                let mut socket = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(5)).unwrap();
                socket.set_nodelay(true).unwrap();
                socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                socket.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
                for (delay, bytes) in parts {
                    std::thread::sleep(delay);
                    if socket.write_all(&bytes).is_err() {
                        break;
                    }
                }
                let mut response = Vec::new();
                if let Err(error) = socket.read_to_end(&mut response) {
                    // Rejecting an unread body may close the connection with a reset after the response.
                    assert!(error.kind() == std::io::ErrorKind::ConnectionReset && !response.is_empty(), "{error}");
                }
                String::from_utf8(response).unwrap()
            }).await.unwrap()
        }
    }

    #[tokio::test]
    async fn admission_over_http_rejects_headers_and_chunked_overflow_and_accepts_valid_bodies() {
        let server = AdmissionServer::new().await;
        for (headers, body, status) in [
            ("Content-Type: text/plain\r\nContent-Length: 2\r\n", "", 415),
            ("Content-Type: application/json\r\nContent-Length: 2\r\nContent-Length: 3\r\n", "", 400),
            ("Content-Type: application/json\r\nContent-Length: 17\r\n", "", 413),
            ("Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n", "8\r\n12345678\r\n9\r\n123456789\r\n0\r\n\r\n", 413),
            ("Content-Type: application/json\r\nContent-Length: 2\r\n", "{}", 200),
        ] {
            let response = server.exchange(vec![(Duration::ZERO, AdmissionServer::request(headers, body))]).await;
            assert!(response.starts_with(&format!("HTTP/1.1 {status} ")), "{response}");
            if status == 200 {
                assert!(response.ends_with("{}"), "{response}");
            }
            assert_eq!(server.admission.readers.available_permits(), 2 * MAX_PENDING);
            assert_eq!(server.admission.pending.load(Ordering::Relaxed), 0);
        }
    }

    #[tokio::test]
    async fn admission_over_http_times_out_slow_drips() {
        let server = AdmissionServer::new().await;
        let mut parts = vec![(Duration::ZERO, AdmissionServer::request("Content-Type: application/json\r\nContent-Length: 16\r\n", "x"))];
        parts.extend((0..8).map(|_| (Duration::from_millis(25), b"x".to_vec())));
        let response = server.exchange(parts).await;
        assert!(response.starts_with("HTTP/1.1 408 "), "{response}");
        assert!(response.contains("request body read timed out"), "{response}");
        assert_eq!(server.admission.readers.available_permits(), 2 * MAX_PENDING);
    }

    #[tokio::test]
    async fn admission_over_http_accepts_a_waiting_request_when_a_reader_is_released() {
        let server = AdmissionServer::new().await;
        let permits = server.admission.readers.acquire_many((2 * MAX_PENDING - 1) as u32).await.unwrap();
        let permit = server.admission.readers.acquire().await.unwrap();
        assert_eq!(server.admission.readers.available_permits(), 0);
        let request = AdmissionServer::request("Content-Type: application/json\r\nContent-Length: 2\r\n", "{}");
        let response = server.exchange(vec![(Duration::ZERO, request)]);
        tokio::pin!(response);
        assert!(tokio::time::timeout(Duration::from_millis(50), &mut response).await.is_err());
        drop(permit);
        let response = response.await;
        assert!(response.starts_with("HTTP/1.1 200 "), "{response}");
        assert!(response.ends_with("{}"), "{response}");
        assert_eq!(server.admission.readers.available_permits(), 1);
        drop(permits);
        assert_eq!(server.admission.readers.available_permits(), 2 * MAX_PENDING);
    }

    #[tokio::test]
    async fn admission_over_http_waits_for_reader_capacity_but_rejects_a_full_queue_before_waiting() {
        let server = AdmissionServer::new().await;
        let permits = server.admission.readers.acquire_many((2 * MAX_PENDING) as u32).await.unwrap();
        let request = AdmissionServer::request("Content-Type: application/json\r\nContent-Length: 2\r\n", "");
        let started = tokio::time::Instant::now();
        let response = server.exchange(vec![(Duration::ZERO, request.clone())]).await;
        assert!(response.starts_with("HTTP/1.1 503 "), "{response}");
        assert!(response.contains("request body readers are busy"), "{response}");
        assert!(started.elapsed() >= Duration::from_millis(250));
        assert_eq!(server.admission.readers.available_permits(), 0);

        server.admission.pending.store(MAX_PENDING, Ordering::Relaxed);
        let response = server.exchange(vec![(Duration::ZERO, request)]).await;
        assert!(response.starts_with("HTTP/1.1 503 "), "{response}");
        assert!(response.contains("GPU queue is full"), "{response}");
        assert_eq!(server.admission.pending.load(Ordering::Relaxed), MAX_PENDING);
        drop(permits);
        assert_eq!(server.admission.readers.available_permits(), 2 * MAX_PENDING);
    }

    #[test]
    fn content_length_requires_a_single_number_within_the_limit() {
        let mut headers = HeaderMap::new();
        assert_eq!(request_content_length(&headers, MAX_BODY_BYTES), Ok(None));
        for length in [0, 10, MAX_BODY_BYTES as u64] {
            headers.insert("content-length", length.to_string().parse().unwrap());
            assert_eq!(request_content_length(&headers, MAX_BODY_BYTES), Ok(Some(length)));
        }
        for value in ["", "abc", "-1", "+1", "1.0", "1, 1", " 1", "1 "] {
            headers.insert("content-length", value.parse().unwrap());
            assert_eq!(request_content_length(&headers, MAX_BODY_BYTES), Err(StatusCode::BAD_REQUEST), "{value}");
        }
        headers.insert("content-length", axum::http::HeaderValue::from_bytes(b"\xff").unwrap());
        assert_eq!(request_content_length(&headers, MAX_BODY_BYTES), Err(StatusCode::BAD_REQUEST));
        for value in [(MAX_BODY_BYTES + 1).to_string(), "18446744073709551616".to_string()] {
            headers.insert("content-length", value.parse().unwrap());
            assert_eq!(request_content_length(&headers, MAX_BODY_BYTES), Err(StatusCode::PAYLOAD_TOO_LARGE));
        }
        headers.insert("content-length", "10".parse().unwrap());
        headers.append("content-length", "10".parse().unwrap());
        assert_eq!(request_content_length(&headers, MAX_BODY_BYTES), Err(StatusCode::BAD_REQUEST));
        headers.insert("content-length", "10".parse().unwrap());
        headers.append("content-length", "20".parse().unwrap());
        assert_eq!(request_content_length(&headers, MAX_BODY_BYTES), Err(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn content_type_requires_json_and_accepts_parameters_and_suffixes() {
        let mut headers = HeaderMap::new();
        assert!(!json_content_type(&headers));
        for value in ["application/json", "application/json; charset=utf-8", "Application/JSON", "application/problem+json", "application/vnd.api+json; charset=utf-8"] {
            headers.insert("content-type", value.parse().unwrap());
            assert!(json_content_type(&headers), "{value}");
        }
        for value in ["text/json", "text/plain", "application/+json", "application/jsonp", "application/json/extra", "application/has space+json", "application/json, application/json", ""] {
            headers.insert("content-type", value.parse().unwrap());
            assert!(!json_content_type(&headers), "{value}");
        }
        headers.insert("content-type", axum::http::HeaderValue::from_bytes(b"application/\xff+json").unwrap());
        assert!(!json_content_type(&headers));
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.append("content-type", "application/json".parse().unwrap());
        assert!(!json_content_type(&headers));
    }

    #[test]
    fn image_budget_reserves_decode_and_rgb_bytes_and_releases_on_failure_and_completion() {
        let used = AtomicUsize::new(0);
        let first = ImageBytes::reserve(&used, 4096, 4096).unwrap();
        assert_eq!(first.bytes, 4096 * 4096 * 7);
        assert_eq!(used.load(Ordering::Relaxed), 4096 * 4096 * 7);
        let remaining = MAX_IMAGE_BYTES - used.load(Ordering::Relaxed);
        let rest = ImageBytes::reserve(&used, remaining / 7, 1).unwrap();
        assert!(ImageBytes::reserve(&used, 1, 1).is_err());
        assert!(ImageBytes::reserve(&used, usize::MAX, 2).is_err());
        drop(rest);
        let failed_build: Result<()> = (|| {
            let _image = ImageBytes::reserve(&used, 10, 20)?;
            Err(anyhow!("invalid request after image decode"))
        })();
        assert!(failed_build.is_err());
        assert_eq!(used.load(Ordering::Relaxed), 4096 * 4096 * 7);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        tx.try_send(first).unwrap();
        drop(rx.recv().unwrap());
        assert_eq!(used.load(Ordering::Relaxed), 0);
        drop(rx);
        assert!(tx.try_send(ImageBytes::reserve(&used, 10, 20).unwrap()).is_err());
        assert_eq!(used.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn forced_ids_are_required_only_with_a_reasoning_budget() {
        let ids = Some([1, 2, 3]);
        assert!(validate_force_ids(None, None).is_ok());
        assert!(validate_force_ids(None, ids).is_ok());
        assert!(validate_force_ids(Some(0), ids).is_ok());
        assert!(validate_force_ids(Some(10), ids).is_ok());
        assert!(validate_force_ids(Some(0), None).is_err());
        assert!(validate_force_ids(Some(10), None).is_err());
        let raw_budget = request_reasoning_budget(&json!({"thinking_token_budget": 10}), true, Some(10)).unwrap();
        assert!(validate_force_ids(raw_budget, None).is_ok());
    }

    #[test]
    fn key_comparison_checks_all_bytes_and_length() {
        assert!(keys_equal(b"secret", b"secret"));
        assert!(keys_equal(b"", b""));
        for key in [b"Secret".as_slice(), b"secreT", b"secre", b"secret!", b"", b"sec\0et"] {
            assert!(!keys_equal(b"secret", key));
        }
        assert!(!keys_equal(b"", b"\0"));
        assert!(!keys_equal(b"secret\0", b"secret"));
        let mut longer = b"secret".to_vec();
        longer.extend([0; 256]);
        assert!(!keys_equal(b"secret", &longer));
    }

    #[test]
    fn authorization_requires_explicit_access() {
        let mut headers = HeaderMap::new();
        assert!(!authorized(None, false, &headers));
        assert!(!authorized(Some(""), false, &headers));
        assert!(authorized(None, true, &headers));
        assert!(authorized(Some(""), true, &headers));
        assert!(!authorized(Some("secret"), false, &headers));
        assert!(!authorized(Some("secret"), true, &headers));
        for value in ["secret", "Basic secret", "bearer secret", "Bearer secreT", "Bearer secret!", "Bearer "] {
            headers.insert("authorization", value.parse().unwrap());
            assert!(!authorized(Some("secret"), false, &headers));
        }
        headers.insert("authorization", "Bearer secret".parse().unwrap());
        assert!(authorized(Some("secret"), false, &headers));
        headers.insert("authorization", " Bearer secret ".parse().unwrap());
        assert!(authorized(Some("secret"), false, &headers));
        headers.append("authorization", "Bearer secret".parse().unwrap());
        assert!(!authorized(Some("secret"), false, &headers));
        headers.insert("authorization", axum::http::HeaderValue::from_bytes(b"Bearer \xff").unwrap());
        assert!(!authorized(Some("secret"), false, &headers));
    }

    #[test]
    fn stop_validation_bounds_count_and_utf8_bytes() {
        assert!(request_stop(&json!({})).unwrap().is_empty());
        assert!(request_stop(&json!({"stop": null})).unwrap().is_empty());
        assert!(request_stop(&json!({"stop": ""})).unwrap().is_empty());
        assert_eq!(request_stop(&json!({"stop": "end"})).unwrap(), vec!["end"]);
        assert_eq!(request_stop(&json!({"stop": ["a", "", "b"]})).unwrap(), vec!["a", "b"]);
        assert_eq!(request_stop(&json!({"stop": vec!["x".repeat(256); 16]})).unwrap().len(), 16);
        assert!(request_stop(&json!({"stop": vec![""; 17]})).is_err());
        assert!(request_stop(&json!({"stop": "x".repeat(257)})).is_err());
        assert!(request_stop(&json!({"stop": ["x".repeat(257)]})).is_err());
        assert!(request_stop(&json!({"stop": "é".repeat(128)})).is_ok());
        assert!(request_stop(&json!({"stop": "é".repeat(129)})).is_err());
        for value in [json!(1), json!(true), json!({}), json!(["end", 1]), json!([null])] {
            assert!(request_stop(&json!({"stop": value})).is_err());
        }
    }

    #[test]
    fn image_validation_counts_all_messages_before_decoding() {
        let image = json!({"type": "image_url", "image_url": "not base64"});
        assert!(request_image_parts(&[], false).unwrap().is_empty());
        assert!(request_image_parts(&[json!({"content": "plain text"})], false).unwrap().is_empty());
        let messages = [json!({"content": vec![image.clone(); 4]}), json!({"content": vec![image.clone(); 4]})];
        assert_eq!(request_image_parts(&messages, true).unwrap().len(), 8);
        assert!(request_image_parts(&messages, false).is_err());
        let messages = [json!({"content": vec![image.clone(); 9]})];
        assert!(request_image_parts(&messages, true).is_err());
        let messages = [json!({"content": [
            {"type": "image"}, {"type": "input_image"}, {"type": "image_url"}, {"type": "text", "text": "hello"}
        ]})];
        assert_eq!(request_image_parts(&messages, true).unwrap().len(), 3);
        assert!(request_image_parts(&messages, false).is_err());
    }

    #[test]
    fn reasoning_end_only_when_open_and_budget_exhausted() {
        assert_eq!(reasoning_action(true, 0, Some(0), false), ReasoningAction::ForceClose);
        assert_eq!(reasoning_action(true, 2, Some(3), false), ReasoningAction::Continue);
        assert_eq!(reasoning_action(true, 3, Some(3), false), ReasoningAction::ForceClose);
        assert_eq!(reasoning_action(true, 4, Some(3), false), ReasoningAction::ForceClose);
        assert_eq!(reasoning_action(true, 100, None, false), ReasoningAction::Continue);
        assert_eq!(reasoning_action(false, 0, Some(0), false), ReasoningAction::Continue);
        assert_eq!(reasoning_action(false, 3, Some(3), false), ReasoningAction::Continue);
        // After the one forced close, text can continue but reopening reasoning ends the request.
        assert_eq!(reasoning_action(false, 3, Some(3), true), ReasoningAction::Continue);
        assert_eq!(reasoning_action(true, 3, Some(3), true), ReasoningAction::Length);
        assert_eq!(reasoning_action(true, 0, Some(3), true), ReasoningAction::Length);
    }

    #[test]
    fn forced_headers_fit_context_and_output_allowance() {
        assert_eq!(forced_token_count(10, 20, 10, 4), 3);
        assert_eq!(forced_token_count(18, 20, 10, 4), 2);
        assert_eq!(forced_token_count(19, 20, 10, 4), 1);
        assert_eq!(forced_token_count(20, 20, 10, 4), 0);
        assert_eq!(forced_token_count(21, 20, 10, 4), 0);
        assert_eq!(forced_token_count(10, 20, 10, 9), 1);
        assert_eq!(forced_token_count(10, 20, 10, 10), 0);
        assert_eq!(forced_token_count(10, 20, 10, 11), 0);
    }

    #[test]
    fn pending_slots_are_bounded_and_released_on_completion_or_failure() {
        let pending = Arc::new(AtomicUsize::new(0));
        let mut slots: Vec<_> = (0..MAX_PENDING).map(|_| JobSlot::reserve(&pending).unwrap()).collect();
        assert!(JobSlot::reserve(&pending).is_none());
        assert_eq!(pending.load(Ordering::Relaxed), MAX_PENDING);
        drop(slots.pop());
        let slot = JobSlot::reserve(&pending).unwrap();
        drop(slots);
        assert_eq!(pending.load(Ordering::Relaxed), 1);
        let failed_build: Result<()> = (move || {
            let _slot = slot;
            Err(anyhow!("invalid request"))
        })();
        assert!(failed_build.is_err());
        assert_eq!(pending.load(Ordering::Relaxed), 0);

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        assert!(tx.try_send(JobSlot::reserve(&pending).unwrap()).is_ok());
        assert_eq!(pending.load(Ordering::Relaxed), 1);
        drop(rx.recv().unwrap());
        assert_eq!(pending.load(Ordering::Relaxed), 0);
        drop(rx);
        assert!(tx.try_send(JobSlot::reserve(&pending).unwrap()).is_err());
        assert_eq!(pending.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn max_tokens_rejects_zero_and_preserves_completion_precedence() {
        assert_eq!(request_max_tokens(&json!({})).unwrap(), None);
        assert_eq!(request_max_tokens(&json!({"max_tokens": null})).unwrap(), None);
        assert_eq!(request_max_tokens(&json!({"max_completion_tokens": null})).unwrap(), None);
        assert_eq!(request_max_tokens(&json!({"max_tokens": 1})).unwrap(), Some(1));
        assert_eq!(request_max_tokens(&json!({"max_tokens": 2, "max_completion_tokens": null})).unwrap(), Some(2));
        assert_eq!(request_max_tokens(&json!({"max_tokens": 2, "max_completion_tokens": 3})).unwrap(), Some(3));
        for body in [json!({"max_tokens": 0}), json!({"max_completion_tokens": 0}),
            json!({"max_tokens": 0, "max_completion_tokens": 3})] {
            assert!(request_max_tokens(&body).is_err());
        }
        for value in [json!(-1), json!(1.5), json!(1.0), json!("3"), json!(true), json!([]), json!({})] {
            for key in ["max_tokens", "max_completion_tokens"] {
                let mut body = json!({"max_tokens": 2, "max_completion_tokens": 3});
                body[key] = value.clone();
                assert!(request_max_tokens(&body).is_err(), "{body}");
            }
        }
    }

    #[test]
    fn request_budget_overrides_default_and_raw_is_unlimited() {
        assert_eq!(request_reasoning_budget(&json!({}), false, None).unwrap(), None);
        assert_eq!(request_reasoning_budget(&json!({}), false, Some(5)).unwrap(), Some(5));
        assert_eq!(request_reasoning_budget(&json!({"thinking_token_budget": null}), false, Some(5)).unwrap(), Some(5));
        assert_eq!(request_reasoning_budget(&json!({"thinking_token_budget": null}), false, None).unwrap(), None);
        assert_eq!(request_reasoning_budget(&json!({"thinking_token_budget": 0}), false, Some(5)).unwrap(), Some(0));
        assert_eq!(request_reasoning_budget(&json!({"thinking_token_budget": 8}), false, Some(5)).unwrap(), Some(8));
        assert_eq!(request_reasoning_budget(&json!({}), true, Some(5)).unwrap(), None);
        assert_eq!(request_reasoning_budget(&json!({"thinking_token_budget": 0}), true, Some(5)).unwrap(), None);
    }

    #[test]
    fn request_budget_rejects_non_integers_and_negative_values() {
        for value in [json!(-1), json!(1.5), json!(1.0), json!("3"), json!(true), json!([]), json!({})] {
            let body = json!({"thinking_token_budget": value});
            assert!(request_reasoning_budget(&body, false, Some(5)).is_err());
            assert_eq!(request_reasoning_budget(&body, true, Some(5)).unwrap(), None);
        }
    }
}
