use std::time::Instant;

use anyhow::{Result, bail, ensure};
use inkling_cuda::Gpu;
use inkling_cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("smoke") => smoke(),
        Some("inspect") if args.len() > 1 => inspect(&args[1]),
        Some("tokenize") if args.len() > 2 => tokenize(&args[1], &args[2], args.iter().any(|a| a == "--special")),
        Some("validate") if args.len() > 2 && args.iter().any(|a| a == "--prefill") => validate_prefill(&args[1], &args[2]),
        Some("validate") if args.len() > 2 => validate(&args[1], &args[2]),
        Some("gemmtest") if args.len() > 1 => gemmtest(&args[1]),
        Some("kvstudy") if args.len() > 1 => kvstudy(&args[1]),
        Some("benchctx") if args.len() > 2 => benchctx(&args[1], args[2].parse()?),
        Some("cachetest") if args.len() > 1 => cachetest(&args[1]),
        Some("rendertest") if args.len() > 1 => rendertest(&args[1]),
        Some("visiontest") if args.len() > 2 => visiontest(&args[1], &args[2], args.get(3).and_then(|s| s.parse().ok()).unwrap_or(400)),
        Some("serve") if args.len() > 1 => serve(&args[1], &args[2..]),
        Some("benchpp") if args.len() > 1 => benchpp(&args[1], args.get(2).and_then(|s| s.parse().ok())),
        Some("bench") if args.len() > 1 => bench(&args[1], args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256)),
        Some("generate") if args.len() > 2 => generate(&args[1], &args[2], args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64)),
        Some("--version" | "-V" | "version") => {
            println!("inkling {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => bail!(
            "InklingEngine {} - Inkling 2 Mini inference on NVIDIA GPUs\n\n\
             usage:\n\
             \x20 inkling serve <model.gguf> [--mmproj FILE] [--ctx N] [--host H] [--port P] [--api-key KEY] [--model-id NAME]\n\
             \x20               [--temp T] [--top-p P] [--top-k K] [--min-p P] [--repeat-penalty R] [--repeat-last-n N]\n\
             \x20 inkling generate <model.gguf> <prompt> [max_tokens]\n\
             \x20 inkling bench <model.gguf> [n] | benchpp <model.gguf> [n] | benchctx <model.gguf> <n>\n\
             \x20 inkling inspect <model.gguf> | tokenize <model.gguf> <file> [--special] | smoke | --version\n\n\
             developer checks: validate, kvstudy, cachetest, rendertest, gemmtest, visiontest (see README)",
            env!("CARGO_PKG_VERSION")
        ),
    }
}

/// Sampling from INKLING_TEMP / INKLING_TOP_P / INKLING_TOP_K / INKLING_MIN_P / INKLING_SEED (greedy by default).
fn sampling_from_env() -> inkling_engine::Sampling {
    let f = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    inkling_engine::Sampling {
        temperature: f("INKLING_TEMP", 0.0),
        top_p: f("INKLING_TOP_P", 1.0),
        min_p: f("INKLING_MIN_P", 0.0),
        top_k: f("INKLING_TOP_K", 0.0) as i32,
        seed: std::env::var("INKLING_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(42),
        repeat_penalty: f("INKLING_REPEAT_PENALTY", 1.0),
        presence_penalty: f("INKLING_PRESENCE_PENALTY", 0.0),
        frequency_penalty: f("INKLING_FREQUENCY_PENALTY", 0.0),
        last_n: f("INKLING_REPEAT_LAST_N", 64.0) as i32,
    }
}

/// Vision encoder vs llama-mtmd-debug (`-p encode -n N --image rainbow`): prints the projected embeddings like
/// common_debug_cb_eval, then encoder timings for a few image sizes.
fn visiontest(model: &str, mmproj: &str, n: usize) -> Result<()> {
    let mut eng = inkling_engine::Engine::load(model, 4096)?;
    eng.load_vision(mmproj)?;
    // the reference debug tool's "rainbow" pattern, fed as already-normalized pixels
    let mut pix = vec![0f32; n * n * 3];
    let (cx, cy) = (n as f32 / 2.0, n as f32 / 2.0);
    let max_dist = (cx * cx + cy * cy).sqrt();
    for y in 0..n {
        for x in 0..n {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let mut hue = dy.atan2(dx) / (2.0 * 3.14159265);
            if hue < 0.0 {
                hue += 1.0;
            }
            let sat = ((dx * dx + dy * dy).sqrt() / max_dist).min(1.0);
            let h6 = hue * 6.0;
            let i6 = h6 as i32;
            let f = h6 - i6 as f32;
            let (p, q, t) = (1.0 - sat, 1.0 - sat * f, 1.0 - sat * (1.0 - f));
            let (r, g, b) = match i6 % 6 {
                0 => (1.0, t, p),
                1 => (q, 1.0, p),
                2 => (p, 1.0, t),
                3 => (p, q, 1.0),
                4 => (t, p, 1.0),
                _ => (1.0, p, q),
            };
            pix[(y * n + x) * 3..][..3].copy_from_slice(&[r, g, b]);
        }
    }
    let gp = n.div_ceil(40) * 2;
    let e = eng.encode_pixels(n, n, gp, gp, &pix, [0.0; 3])?;
    let w = eng.cfg.n_embd;
    let row = |i: usize| {
        let r = &e.data[i * w..(i + 1) * w];
        println!("  [{:>11.4}, {:>11.4}, {:>11.4}, ..., {:>11.4}, {:>11.4}, {:>11.4}]", r[0], r[1], r[2], r[w - 3], r[w - 2], r[w - 1]);
    };
    println!("projected {{{w}, {}}}", e.n);
    for i in [0, 1, 2, e.n - 3, e.n - 2, e.n - 1] {
        row(i);
    }
    println!("  sum = {:.6}", e.data.iter().map(|&x| x as f64).sum::<f64>());
    for side in [640usize, 1280, 2560] {
        let img = inkling_engine::Image { width: side, height: side, rgb: vec![128; side * side * 3] };
        eng.encode_image(&img)?; // warm-up
        let t = Instant::now();
        let e = eng.encode_image(&img)?;
        println!("encode {side}x{side}: {} tokens in {:.1} ms", e.n, t.elapsed().as_secs_f64() * 1e3);
    }
    Ok(())
}

/// Prefill GEMM self-test vs the decode matvec, plus GEMM throughput.
fn gemmtest(model: &str) -> Result<()> {
    let eng = inkling_engine::Engine::load(model, 4096)?;
    for t in [1usize, 37, 64, 512] {
        let (err, mag, ms) = eng.test_gemm(0, t, 20)?;
        let flops = 2.0 * t as f64 * 2304.0 * (6528.0 + 2048.0);
        println!("T={t:4}: max |gemm - matvec| = {err:.3e} (max |y| {mag:.2}) | {ms:.3} ms = {:.1} TOPS", flops / ms / 1e9);
    }
    Ok(())
}

/// KV-cache format study: KL / top-1 of fake-quantized K/V caches against the f16 cache, on the long chat
/// dump prompt and on ~8k tokens of repository text (docs + sources).
fn kvstudy(model: &str) -> Result<()> {
    let mut eng = inkling_engine::Engine::load(model, 16384)?;
    let tok = inkling_core::Tokenizer::from_gguf(&inkling_core::Gguf::open(model)?)?;
    let nv = eng.cfg.n_vocab_unpadded;
    let mut texts: Vec<(&str, Vec<u32>)> = Vec::new();
    texts.push(("chat 807", tok.encode(&std::fs::read_to_string("tests/data/prompt_long.txt")?, true)?));
    let mut src = String::new();
    for f in ["docs/MODEL.md", "docs/PLAN.md", "crates/inkling-engine/src/prefill.rs", "crates/inkling-cuda/kernels/prefill_attn.cu", "crates/inkling-core/src/config.rs"] {
        src.push_str(&std::fs::read_to_string(f).unwrap_or_default());
        src.push_str("

");
    }
    let mut ids = tok.encode(&src, false)?;
    ids.truncate(8192);
    texts.push(("repo 8k", ids));
    let configs: &[(&str, &str, &str, bool)] = &[
        ("K8 V8", "8", "8", false), ("K6 V6", "6", "6", false), ("K5 V5", "5", "5", false), ("K4 V4", "4", "4", false),
        ("K8 V4", "8", "4", false), ("K4 V8", "4", "8", false), ("K6 V4", "6", "4", false),
        ("K8 V8 all", "8", "8", true), ("K6 V6 all", "6", "6", true), ("K4 V4 all", "4", "4", true),
    ];
    let set = |k: &str, v: &str, all: bool| unsafe {
        std::env::set_var("INKLING_KQ", k);
        std::env::set_var("INKLING_VQ", v);
        if all { std::env::set_var("INKLING_KV_ALL", "1") } else { std::env::remove_var("INKLING_KV_ALL") }
    };
    for (name, ids) in &texts {
        set("0", "0", false);
        let base = eng.prefill_logits(ids)?;
        let n = ids.len();
        println!("== {name} ({n} tokens), against the f16 cache (sliding-window layers f16 unless 'all')");
        for &(cname, k, v, all) in configs {
            set(k, v, all);
            let lg = eng.prefill_logits(ids)?;
            let mut kls: Vec<f64> = Vec::with_capacity(n);
            let mut agree = 0;
            for t in 0..n {
                let (kl, top1) = kl_top1(&lg[t * nv..(t + 1) * nv], &base[t * nv..(t + 1) * nv]);
                kls.push(kl);
                agree += top1 as usize;
            }
            let mean = kls.iter().sum::<f64>() / n as f64;
            let late = kls[n / 2..].iter().sum::<f64>() / (n - n / 2) as f64;
            kls.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("  {cname:<10} mean KL {mean:.5} (2nd half {late:.5})  median {:.6}  p99 {:.4}  top-1 {:.2}%",
                kls[n / 2], kls[n * 99 / 100], 100.0 * agree as f64 / n as f64);
        }
    }
    set("0", "0", false);
    Ok(())
}

/// Long context: load with a 256k context, prefill `n` tokens of repository text, then decode 64 tokens at that
/// depth. Reports VRAM in use, prompt and decode throughput.
fn benchctx(model: &str, n: usize) -> Result<()> {
    let ctx: usize = std::env::var("INKLING_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(262144);
    let mut eng = inkling_engine::Engine::load(model, ctx)?;
    let tok = inkling_core::Tokenizer::from_gguf(&inkling_core::Gguf::open(model)?)?;
    let (free, total) = inkling_cuda::cudarc::driver::result::mem_get_info()?;
    println!("context {ctx}: VRAM in use {:.2} / {:.2} GiB", (total - free) as f64 / (1u64 << 30) as f64, total as f64 / (1u64 << 30) as f64);
    let mut src = String::new();
    for f in ["docs/MODEL.md", "docs/PLAN.md", "crates/inkling-engine/src/prefill.rs", "crates/inkling-engine/src/lib.rs", "crates/inkling-cuda/kernels/decode.cu"] {
        src.push_str(&std::fs::read_to_string(f).unwrap_or_default());
    }
    let base = tok.encode(&src, false)?;
    let ids: Vec<u32> = base.iter().cycle().take(n).copied().collect();
    let t0 = Instant::now();
    let mut first = None;
    let out = eng.generate(&ids, 65, &inkling_engine::Sampling::greedy(), |_| {
        if first.is_none() {
            first = Some(t0.elapsed().as_secs_f64());
        }
        true
    })?;
    let total_s = t0.elapsed().as_secs_f64();
    let ttft = first.unwrap();
    let m = out.len() - 1;
    println!("prompt {n} tok: {:.0} tok/s ({:.2} s) | decode {m} tok at depth {n}: {:.1} tok/s", n as f64 / ttft, ttft, m as f64 / (total_s - ttft));
    Ok(())
}

/// Prompt caching must not change results: cached continuations vs fresh runs (greedy), plus sampling.
fn cachetest(model: &str) -> Result<()> {
    let mut eng = inkling_engine::Engine::load(model, 8192)?;
    let tok = inkling_core::Tokenizer::from_gguf(&inkling_core::Gguf::open(model)?)?;
    let g = inkling_engine::Sampling::greedy();
    let a = tok.encode("<|message_user|><|content_text|>Name three rivers in Europe.<|end_message|><|message_model|><|content_text|>", true)?;
    let fresh = |eng: &mut inkling_engine::Engine, p: &[u32]| -> Result<Vec<u32>> {
        eng.reset()?;
        eng.generate(p, 24, &g, |_| true)
    };
    let out_a = fresh(&mut eng, &a)?;
    // (1) extension of prompt + generated tokens: state reused directly
    let mut b = a.clone();
    b.extend_from_slice(&out_a);
    b.extend(tok.encode(" Also, ", false)?);
    let fb = fresh(&mut eng, &b)?;
    eng.reset()?;
    eng.generate(&a, 24, &g, |_| true)?;
    let (cb, reused_b) = eng.generate_ex(&b, 24, &g, &mut |_| true)?;
    println!("extension: reused {reused_b} of {} prompt tokens, identical to fresh: {}", b.len(), cb == fb);
    // (2) different continuation of the same prompt: conv snapshot restored
    let mut c = a.clone();
    c.extend(tok.encode(" The Danube, the Rhine and", false)?);
    let fc = fresh(&mut eng, &c)?;
    eng.reset()?;
    eng.generate(&a, 24, &g, |_| true)?;
    let (cc, reused_c) = eng.generate_ex(&c, 24, &g, &mut |_| true)?;
    println!("snapshot: reused {reused_c} of {} prompt tokens, identical to fresh: {}", c.len(), cc == fc);
    // (3) same prompt again (regenerate)
    let (ca, reused_a) = eng.generate_ex(&c, 24, &g, &mut |_| true)?;
    println!("repeat: reused {reused_a} of {} prompt tokens, identical: {}", c.len(), ca == fc);
    // sampling: two seeds differ, same seed repeats
    let s1 = inkling_engine::Sampling { temperature: 0.8, top_p: 0.95, min_p: 0.0, top_k: 0, seed: 1, ..inkling_engine::Sampling::greedy() };
    let s2 = inkling_engine::Sampling { seed: 2, ..s1 };
    let x1 = eng.generate(&a, 24, &s1, |_| true)?;
    let x1b = eng.generate(&a, 24, &s1, |_| true)?;
    let x2 = eng.generate(&a, 24, &s2, |_| true)?;
    println!("sampling: seed 1 repeatable: {}, seed 2 differs: {}", x1 == x1b, x1 != x2);
    println!("  greedy: {}", tok.decode(&out_a, false).replace('\n', " "));
    println!("  seed 1: {}", tok.decode(&x1, false).replace('\n', " "));
    println!("  seed 2: {}", tok.decode(&x2, false).replace('\n', " "));
    Ok(())
}

/// Render chat test cases with the native template (one JSON string per line; compare with tools/chat_oracle.py).
fn rendertest(cases: &str) -> Result<()> {
    let v: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(cases)?)?;
    for c in v {
        let msgs = c["messages"].as_array().cloned().unwrap_or_default();
        let tools = c.get("tools").and_then(|t| t.as_array()).cloned();
        let r = inkling_core::chat::render(&msgs, tools.as_deref(), c.get("reasoning_effort"))?;
        println!("{}", serde_json::to_string(&r)?);
    }
    Ok(())
}

/// OpenAI-compatible server: inkling serve <model> [--host H] [--port P] [--ctx N] [--api-key K] [--temp T]
/// [--top-p P] [--top-k K] [--min-p P] [--repeat-penalty R] [--repeat-last-n N] [--mmproj FILE] [--model-id NAME]
fn serve(model: &str, flags: &[String]) -> Result<()> {
    let get = |name: &str| flags.iter().position(|a| a == name).and_then(|i| flags.get(i + 1)).cloned();
    let num = |name: &str| -> Result<Option<f64>> { get(name).map(|v| v.parse::<f64>().map_err(|e| anyhow::anyhow!("{name}: {e}"))).transpose() };
    let mut d = inkling_server::Defaults::default();
    if let Some(v) = num("--temp")? { d.temperature = v as f32; }
    if let Some(v) = num("--top-p")? { d.top_p = v as f32; }
    if let Some(v) = num("--top-k")? { d.top_k = v as i32; }
    if let Some(v) = num("--min-p")? { d.min_p = v as f32; }
    if let Some(v) = num("--repeat-penalty")? { d.repeat_penalty = v as f32; }
    if let Some(v) = num("--repeat-last-n")? { d.repeat_last_n = v as i32; }
    let model_id = get("--model-id").unwrap_or_else(|| {
        std::path::Path::new(model).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "inkling-2-mini".into())
    });
    let opts = inkling_server::Options {
        host: get("--host").unwrap_or_else(|| "127.0.0.1".into()),
        port: num("--port")?.map(|v| v as u16).unwrap_or(8080),
        n_ctx: num("--ctx")?.map(|v| v as usize).unwrap_or(131072),
        api_key: get("--api-key").or_else(|| std::env::var("INKLING_API_KEY").ok()),
        model_id,
        defaults: d,
        mmproj: get("--mmproj"),
    };
    inkling_server::serve(model, opts)
}

/// Prompt-processing throughput of the batched prefill at several prompt lengths.
fn benchpp(model: &str, only: Option<usize>) -> Result<()> {
    let mut eng = inkling_engine::Engine::load(model, 16384)?;
    let tok = inkling_core::Tokenizer::from_gguf(&inkling_core::Gguf::open(model)?)?;
    let text = std::fs::read_to_string("tests/data/prompt_long.txt").unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog. ".repeat(50));
    let base = tok.encode(&text, true)?;
    for n in only.map_or(vec![128usize, 512, 2048, 8192], |n| vec![n]) {
        let ids: Vec<u32> = base.iter().cycle().take(n).copied().collect();
        eng.prefill_fresh(&ids)?; // warm-up
        eng.gpu.sync()?;
        let reps = (4096 / n).max(1);
        let t0 = Instant::now();
        for _ in 0..reps {
            eng.prefill_fresh(&ids)?;
        }
        eng.gpu.sync()?;
        let s = t0.elapsed().as_secs_f64() / reps as f64;
        println!("pp{n}: {:.0} tok/s ({:.1} ms)", n as f64 / s, s * 1e3);
    }
    Ok(())
}

/// Throughput with the device-side loop: prompt processing + greedy decode of `n` tokens, no printing.
fn bench(model: &str, n: usize) -> Result<()> {
    let mut eng = inkling_engine::Engine::load(model, 4096)?;
    let tok = inkling_core::Tokenizer::from_gguf(&inkling_core::Gguf::open(model)?)?;
    let ids = tok.encode("<|message_user|><|content_text|>Write a long story about a lighthouse keeper.<|end_message|><|message_model|><|content_text|>", true)?;
    eng.generate(&ids, 4, &inkling_engine::Sampling::greedy(), |_| true)?; // warm-up: capture the graphs
    let t0 = Instant::now();
    eng.generate(&ids, 1, &inkling_engine::Sampling::greedy(), |_| true)?;
    let pre = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let out = eng.generate(&ids, n, &inkling_engine::Sampling::greedy(), |_| true)?;
    let total = t1.elapsed().as_secs_f64();
    let dec = (total - pre).max(1e-9);
    let m = out.len() - 1;
    println!("prompt {} tok: {:.1} tok/s | decode {m} tok: {:.1} tok/s ({:.2} ms/tok)", ids.len(), ids.len() as f64 / pre, m as f64 / dec, dec * 1e3 / m as f64);
    if eng.moe_tl.is_some() {
        for slot in [10usize, 21] {
            let tl = eng.moe_timeline(slot)?;
            let t0 = tl.iter().map(|r| r[0]).min().unwrap();
            let rel = |k: usize, f: fn(u64, u64) -> u64| tl.iter().map(|r| r[k] - t0).fold(if f(0, 1) == 0 { u64::MAX } else { 0 }, f) as f64 / 1e3;
            print!("moe timeline slot {slot} (us, min..max over blocks):");
            for (k, name) in ["start", "staged", "A done", "barrier", "h staged", "end"].iter().enumerate() {
                print!(" {name} {:.1}..{:.1} |", rel(k, u64::min), rel(k, u64::max));
            }
            println!();
        }
    }
    Ok(())
}

/// Greedy generation from a raw prompt (special tokens parsed); reports prefill and decode speed.
fn generate(model: &str, prompt: &str, n_gen: usize) -> Result<()> {
    use std::io::Write;
    let mut eng = inkling_engine::Engine::load(model, 16384)?;
    let tok = inkling_core::Tokenizer::from_gguf(&inkling_core::Gguf::open(model)?)?;
    let text = std::fs::read_to_string(prompt).unwrap_or_else(|_| prompt.to_owned());
    let ids = tok.encode(&text, true)?;
    print!("{}", text);
    let t0 = Instant::now();
    let mut first: Option<f64> = None;
    let eos = tok.eos;
    let out = eng.generate(&ids, n_gen, &sampling_from_env(), |t| {
        if first.is_none() {
            first = Some(t0.elapsed().as_secs_f64());
        }
        if t == eos {
            return false;
        }
        print!("{}", String::from_utf8_lossy(&tok.token_bytes(t, true)));
        let _ = std::io::stdout().flush();
        true
    })?;
    let total = t0.elapsed().as_secs_f64();
    let ttft = first.unwrap_or(total);
    let n = out.len().saturating_sub(1);
    println!("
---
prompt {} tok, time to first token {:.0} ms ({:.0} tok/s) | decode {} tok in {:.2}s ({:.1} tok/s)",
        ids.len(), ttft * 1e3, ids.len() as f64 / ttft, n, total - ttft, n as f64 / (total - ttft).max(1e-9));
    Ok(())
}

/// Relative L2 error ‖a−b‖/‖b‖.
fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

/// KL(ref || ours) over softmax of logits, and whether the argmax agrees.
fn kl_top1(ours: &[f32], reference: &[f32]) -> (f64, bool) {
    let lse = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lo, lr) = (lse(ours), lse(reference));
    let mut kl = 0f64;
    for (o, r) in ours.iter().zip(reference) {
        let pr = (*r as f64 - lr).exp();
        kl += pr * ((*r as f64 - lr) - (*o as f64 - lo));
    }
    let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    (kl, am(ours) == am(reference))
}

/// Replay a refdump prompt token by token; compare every layer output (l_out-<slot>) and the logits.
/// Batched prefill of the whole dump prompt; KL / top-1 of every position's logits against the reference.
fn validate_prefill(model: &str, dump: &str) -> Result<()> {
    use inkling_core::npy;
    let dir = std::path::Path::new(dump);
    let toks = npy::read(dir.join("tokens.npy"))?;
    let ref_logits = npy::read(dir.join("logits.npy"))?;
    let t_n = toks.data.len();
    let mut eng = inkling_engine::Engine::load(model, 4096)?;
    let nv = eng.cfg.n_vocab_unpadded;
    let n_vocab_ref = ref_logits.shape[1];
    let ids: Vec<u32> = toks.data.iter().map(|&v| v as u32).collect();
    eng.prefill_logits(&ids)?; // warm-up
    let t0 = Instant::now();
    let logits = eng.prefill_logits(&ids)?;
    let el = t0.elapsed().as_secs_f64();
    let mut kls = Vec::with_capacity(t_n);
    let mut agree = 0usize;
    for t in 0..t_n {
        let (kl, top1) = kl_top1(&logits[t * nv..(t + 1) * nv], &ref_logits.data[t * n_vocab_ref..t * n_vocab_ref + nv]);
        if t < 3 || kl > 0.05 || std::env::var_os("INKLING_KL_ALL").is_some() {
            println!("t={t:3}  KL={kl:.5}  top1={}", if top1 { "same" } else { "DIFF" });
        }
        kls.push(kl);
        agree += top1 as usize;
    }
    let mean = kls.iter().sum::<f64>() / t_n as f64;
    kls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("--- prefill {t_n} tokens: mean KL {mean:.5}, median {:.6}, p99 {:.4}, top-1 agreement {agree}/{t_n} ({:.0} ms incl. logits readback)",
        kls[t_n / 2], kls[(t_n * 99 / 100).min(t_n - 1)], el * 1e3);
    Ok(())
}

fn validate(model: &str, dump: &str) -> Result<()> {
    use inkling_core::npy;
    let dir = std::path::Path::new(dump);
    let toks = npy::read(dir.join("tokens.npy"))?;
    let ref_logits = npy::read(dir.join("logits.npy"))?;
    let t_n = toks.data.len();
    let t0 = Instant::now();
    let mut eng = inkling_engine::Engine::load(model, 4096)?;
    eng.use_graph = false;
    println!("loaded {:.2} GiB of weights in {:.1} s", eng.w.bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64());
    let d = eng.cfg.n_embd;
    let n_slots = eng.cfg.n_slots();
    let refs: Vec<Option<Vec<f32>>> = (0..n_slots).map(|s| npy::read(dir.join(format!("l_out-{s}.npy"))).ok().map(|n| n.data)).collect();
    let nv = eng.cfg.n_vocab_unpadded;
    let n_vocab_ref = ref_logits.shape[1];

    let mut worst = vec![0f64; n_slots];
    let dbg_slot: Option<usize> = std::env::var("INKLING_DEBUG_SLOT").ok().and_then(|v| v.parse().ok());
    let dbg_t: usize = std::env::var("INKLING_DEBUG_T").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let (mut kl_sum, mut agree) = (0f64, 0usize);
    let order: Vec<usize> = eng.cfg.schedule().iter().map(|p| p.slot).collect();
    for t in 0..t_n {
        let mut errs = vec![0f64; n_slots];
        let mut cb = |slot: usize, e: &inkling_engine::Engine| {
            if let Some(r) = &refs[slot] {
                let x = e.debug_buf("x").unwrap();
                errs[slot] = rel_err(&x, &r[t * d..(t + 1) * d]);
            }
            if t == dbg_t && Some(slot) == dbg_slot {
                for (ours, refname) in [("q", "Qcur"), ("k", "Kcur"), ("v", "Vcur"), ("r", "Rcur"), ("attn_out", "attn_out"),
                    ("rlog", "ffn_moe_logits"), ("wts", "ffn_moe_weights_scaled"), ("ffn_out", "ffn_out"), ("x", "l_out")] {
                    match npy::read(dir.join(format!("{refname}-{slot}.npy"))) {
                        Ok(r) => {
                            let v = e.debug_buf(ours).unwrap();
                            let n = v.len();
                            println!("  slot {slot} t={t} {ours:>8} vs {refname:<24} rel.err {:.5}   ours[..4]={:?} ref[..4]={:?}", rel_err(&v, &r.data[t * n..(t + 1) * n]), &v[..4], &r.data[t * n..t * n + 4]);
                        }
                        Err(_) => println!("  slot {slot}: no reference {refname}"),
                    }
                }
            }
        };
        let logits = eng.step(toks.data[t] as u32, t, Some(&mut cb))?;
        for s in 0..n_slots {
            worst[s] = worst[s].max(errs[s]);
        }
        let rl = &ref_logits.data[t * n_vocab_ref..t * n_vocab_ref + nv];
        let (kl, top1) = kl_top1(&logits, rl);
        kl_sum += kl;
        agree += top1 as usize;
        if t == 0 && std::env::var_os("INKLING_VERBOSE").is_some() {
            let l: Vec<String> = order.iter().map(|&s| format!("{s}:{:.4}", errs[s])).collect();
            println!("t=0 per-slot rel.err: {}", l.join(" "));
        }
        if t < 3 || t + 1 == t_n || kl > 0.02 || !top1 || std::env::var_os("INKLING_KL_ALL").is_some() {
            let first_bad = order.iter().find(|&&s| errs[s] > 0.05);
            println!("t={t:3}  KL={kl:.5}  top1={}  max layer rel.err={:.4}{}", if top1 { "same" } else { "DIFF" },
                errs.iter().cloned().fold(0.0, f64::max), first_bad.map(|s| format!("  (first >5% at slot {s})")).unwrap_or_default());
        }
    }
    println!("--- {t_n} tokens: mean KL {:.5}, top-1 agreement {}/{t_n}", kl_sum / t_n as f64, agree);
    let line: Vec<String> = order.iter().map(|&s| format!("{s}:{:.3}", worst[s])).collect();
    println!("worst rel.err per slot (execution order): {}", line.join(" "));
    Ok(())
}

/// Tokenize a file; prints comma-separated ids (same format as `llama-tokenize --ids`).
fn tokenize(model: &str, file: &str, special: bool) -> Result<()> {
    let g = inkling_core::Gguf::open(model)?;
    let t0 = Instant::now();
    let tok = inkling_core::Tokenizer::from_gguf(&g)?;
    let load = t0.elapsed();
    let text = std::fs::read_to_string(file)?;
    let t1 = Instant::now();
    let ids = tok.encode(&text, special)?;
    let enc = t1.elapsed();
    let s: Vec<String> = ids.iter().map(u32::to_string).collect();
    println!("[{}]", s.join(", "));
    eprintln!(
        "{} tokens | tokenizer load {:.0} ms | encode {:.1} ms | roundtrip {}",
        ids.len(),
        load.as_secs_f64() * 1e3,
        enc.as_secs_f64() * 1e3,
        if tok.decode(&ids, true) == text { "OK" } else { "MISMATCH" }
    );
    Ok(())
}

/// Parse the GGUF, print the config, the layer schedule and a weight-footprint breakdown.
fn inspect(path: &str) -> Result<()> {
    use std::collections::BTreeMap;
    let t = Instant::now();
    let g = inkling_core::Gguf::open(path)?;
    let cfg = inkling_core::Config::from_gguf(&g)?;
    println!("parsed GGUF v{} in {:.0} ms: {} tensors, {} metadata keys", g.version, t.elapsed().as_secs_f64() * 1e3, g.tensors.len(), g.kv.len());
    println!("{cfg:#?}");
    let sched = cfg.schedule();
    let full: Vec<usize> = sched.iter().filter(|p| !cfg.is_swa[p.phys]).map(|p| p.slot).collect();
    println!("{} passes per token; full-attention slots: {full:?}", sched.len());
    let line: Vec<String> = sched.iter().map(|p| format!("{}{}", p.phys, if p.pass > 0 { "'".repeat(p.pass) } else { String::new() })).collect();
    println!("schedule (phys, ' = repeated pass): {}", line.join(" "));

    // bytes read per token: core layers count n_passes times, experts only top-k of n_expert
    let mut by_type: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    let mut per_token = 0f64;
    for ti in &g.tensors {
        let visits = match ti.name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|n| n.parse::<usize>().ok()) {
            Some(l) if (cfg.core_start..cfg.core_end).contains(&l) => cfg.n_passes as f64,
            _ => 1.0,
        };
        let frac = if ti.name.contains("_exps.weight") { cfg.n_expert_used as f64 / cfg.n_expert as f64 } else { 1.0 };
        let read = if ti.name == "token_embd.weight" { 0.0 } else { ti.n_bytes as f64 * visits * frac };
        per_token += read;
        let e = by_type.entry(format!("{:?}", ti.ty)).or_default();
        e.0 += ti.n_bytes;
        e.1 += read;
    }
    for (ty, (bytes, read)) in &by_type {
        println!("  {ty:>6}: {:>8.2} GiB stored, {:>7.1} MB read/token", *bytes as f64 / (1u64 << 30) as f64, read / 1e6);
    }
    println!("weights read per token ≈ {:.2} GB → roofline at 888 GB/s ≈ {:.0} tok/s", per_token / 1e9, 888e9 / per_token);
    Ok(())
}

/// Device info, a correctness check on a trivial kernel, and the achievable DRAM read bandwidth (decode roofline).
fn smoke() -> Result<()> {
    let gpu = Gpu::new(0)?;
    let i = &gpu.info;
    println!(
        "GPU 0: {} | sm_{}{} | {} SMs | {:.1} GiB | L2 {} MiB",
        i.name,
        i.compute_capability.0,
        i.compute_capability.1,
        i.sm_count,
        i.total_mem as f64 / (1u64 << 30) as f64,
        i.l2_bytes >> 20
    );

    // axpy correctness
    let n = 1usize << 20;
    let x: Vec<f32> = (0..n).map(|k| k as f32 * 0.5).collect();
    let y: Vec<f32> = (0..n).map(|k| 1.0 - k as f32).collect();
    let dx = gpu.stream.clone_htod(&x)?;
    let mut dy = gpu.stream.clone_htod(&y)?;
    let f = gpu.func("ink_axpy_f32")?;
    let (a, n64) = (2.0f32, n as u64);
    let mut l = gpu.stream.launch_builder(&f);
    l.arg(&a).arg(&dx).arg(&mut dy).arg(&n64);
    unsafe { l.launch(LaunchConfig::for_num_elems(n as u32)) }?;
    let out = gpu.stream.clone_dtoh(&dy)?;
    let bad = (0..n).filter(|&k| (out[k] - (a * x[k] + y[k])).abs() > 1e-3).count();
    ensure!(bad == 0, "axpy mismatch in {bad} elements");
    println!("axpy: OK");

    // read bandwidth: 2 GiB buffer, best of several passes
    let bytes = 2usize << 30;
    let words = bytes / 16;
    let src = gpu.stream.alloc_zeros::<u8>(bytes)?;
    let sink = gpu.stream.alloc_zeros::<u32>(4096)?;
    let rb = gpu.func("ink_read_bw")?;
    let cfg = LaunchConfig { grid_dim: (i.sm_count as u32 * 8, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
    let words64 = words as u64;
    let mut best = f64::MAX;
    for _ in 0..6 {
        gpu.stream.synchronize()?;
        let t = Instant::now();
        let mut l = gpu.stream.launch_builder(&rb);
        l.arg(&src).arg(&words64).arg(&sink);
        unsafe { l.launch(cfg) }?;
        gpu.stream.synchronize()?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("DRAM read bandwidth: {:.0} GB/s (decode roofline)", bytes as f64 / best / 1e9);

    // per-kernel cost of streaming one slab (back-to-back launches over distinct slabs, so nothing hits L2)
    for mb in [1usize, 4, 12, 24, 48] {
        let slab = mb << 20;
        let n = (bytes / slab).min(64);
        let w = (slab / 16) as u64;
        let blocks = (slab / 16).div_ceil(512).min(i.sm_count as usize * 8) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
        let mut best = f64::MAX;
        for _ in 0..3 {
            gpu.stream.synchronize()?;
            let t = Instant::now();
            for k in 0..n {
                let s = src.slice(k * slab..(k + 1) * slab);
                let mut l = gpu.stream.launch_builder(&rb);
                l.arg(&s).arg(&w).arg(&sink);
                unsafe { l.launch(cfg) }?;
            }
            gpu.stream.synchronize()?;
            best = best.min(t.elapsed().as_secs_f64() / n as f64);
        }
        println!("  {mb:>2} MiB slab: {:.1} us/kernel = {:.0} GB/s", best * 1e6, slab as f64 / best / 1e9);
    }
    // the expert row-gather pattern over 24 MiB slabs, at several grid sizes
    let slab = 24usize << 20;
    let rows = (slab / 1152) as i32;
    let n = 64;
    for name in ["ink_pattern_bw8", "ink_pattern_bw4", "ink_pattern_bw2"] {
        let f = gpu.func(name)?;
        for bps in [2u32, 4, 6] {
            let cfg = LaunchConfig { grid_dim: (i.sm_count as u32 * bps, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut best = f64::MAX;
            for _ in 0..3 {
                gpu.stream.synchronize()?;
                let t = Instant::now();
                for k in 0..n {
                    let s = src.slice(k * slab..(k + 1) * slab);
                    let mut l = gpu.stream.launch_builder(&f);
                    l.arg(&s).arg(&rows).arg(&sink);
                    unsafe { l.launch(cfg) }?;
                }
                gpu.stream.synchronize()?;
                best = best.min(t.elapsed().as_secs_f64() / n as f64);
            }
            println!("  {name} {bps} blocks/SM: {:.1} us = {:.0} GB/s", best * 1e6, slab as f64 / best / 1e9);
        }
    }
    let f = gpu.func("ink_pattern_bw8s")?;
    for mode in [1i32, 2] {
        // rows sized so qs + scales = one 27 MiB slab (24 MiB of qs + 3 MiB of scales)
        let rows = (slab / 1152) as i32;
        let total = rows as usize * 1296;
        let cfg = LaunchConfig { grid_dim: (i.sm_count as u32 * 4, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut best = f64::MAX;
        for _ in 0..3 {
            gpu.stream.synchronize()?;
            let t = Instant::now();
            for k in 0..n {
                let s = src.slice(k * total..(k + 1) * total);
                let mut l = gpu.stream.launch_builder(&f);
                l.arg(&s).arg(&rows).arg(&sink).arg(&mode);
                unsafe { l.launch(cfg) }?;
            }
            gpu.stream.synchronize()?;
            best = best.min(t.elapsed().as_secs_f64() / n as f64);
        }
        println!("  ink_pattern_bw8s scales {}: {:.1} us = {:.0} GB/s", if mode == 1 { "separate" } else { "after row" }, best * 1e6, total as f64 / best / 1e9);
    }
    Ok(())
}
