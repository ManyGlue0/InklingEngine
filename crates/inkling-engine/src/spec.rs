//! Speculative decoding: the drafter proposes k tokens, the target checks [last, d1..dk] in one batched pass
//! (short-chain kernels), the longest agreeing prefix is kept plus the target's own next token.
//!
//! Target state across a verify pass: KV entries past the kept prefix are simply overwritten later; the short-conv
//! states are deferred (the pass keeps every conv input of its tokens in `hist`) and committed for the kept prefix
//! only (`commit_conv`).

use anyhow::{Result, ensure};
use inkling_cuda::{Arg, ConvCommit};

use crate::dflash2::DFlash2;
use crate::dspark::{DSpark, FEAT_ROWS, MAX_DRAFT};

/// The attached drafter.
pub(crate) enum Drafter {
    DSpark(DSpark),
    DFlash2(DFlash2),
}
use crate::prefill::{ChunkOpts, PREFILL_CHUNK};
use crate::{Engine, F32};

/// Longest verify chunk (anchor + drafts), also the conv history depth.
pub(crate) const HIST_T: usize = MAX_DRAFT + 1;

const KVW: usize = 1024;

/// Bytes of one slot's conv history: [k: T x 1024][v: T x 1024][attn: T x D][ffn: T x D] floats.
fn hist_slot_bytes(d: usize) -> usize {
    HIST_T * (2 * KVW + 2 * d) * F32
}

/// Byte offset of a slot's history for conv site 0 = k, 1 = v, 2 = attention post conv, 3 = FFN post conv.
pub(crate) fn hist_offset(slot: usize, site: usize, d: usize) -> usize {
    let off = match site {
        0 => 0,
        1 => HIST_T * KVW,
        2 => 2 * HIST_T * KVW,
        _ => 2 * HIST_T * KVW + HIST_T * d,
    };
    slot * hist_slot_bytes(d) + off * F32
}

/// Counters of a speculative generation.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpecStats {
    /// verify passes
    pub passes: usize,
    /// drafted tokens (k per pass)
    pub drafted: usize,
    /// drafted tokens the target agreed with
    pub accepted: usize,
}

impl SpecStats {
    /// tokens produced per verify pass (accepted drafts + the target's own token)
    pub fn tokens_per_pass(&self) -> f64 {
        (self.accepted + self.passes) as f64 / self.passes.max(1) as f64
    }
}

impl Engine {
    /// Load a drafter for speculative decoding: a DSpark GGUF, or a DFlash2 directory / model.safetensors.
    pub fn attach_draft(&mut self, path: &str) -> Result<()> {
        ensure!(self.cpu.is_none(), "speculative decoding does not support CPU offload yet");
        ensure!(self.n_streams == 1, "speculative decoding does not combine with --parallel yet");
        ensure!((1..=MAX_DRAFT).contains(&self.draft_n), "--draft-n must be 1 to {MAX_DRAFT}");
        let free = self.gpu.free_vram()?;
        let dflash2 = !path.to_ascii_lowercase().ends_with(".gguf");
        let feat = 4 * FEAT_ROWS * self.cfg.n_embd * F32;
        let need = if dflash2 { 900 * (1 << 20) + feat + FEAT_ROWS * 4 * self.cfg.n_embd } else { 640 * (1 << 20) + self.n_ctx * DSpark::kv_bytes_per_token() + feat };
        crate::check_vram(free, need, "the drafter")?;
        let d = if dflash2 {
            Drafter::DFlash2(DFlash2::load(&self.gpu, path, self.n_ctx, self.cfg.n_vocab, self.cfg.n_vocab_unpadded)?)
        } else {
            Drafter::DSpark(DSpark::load(&self.gpu, path, self.n_ctx, self.cfg.n_vocab, self.cfg.n_vocab_unpadded)?)
        };
        let taps = match &d {
            Drafter::DSpark(x) => &x.taps,
            Drafter::DFlash2(x) => &x.taps,
        };
        ensure!(taps[3] == self.cfg.n_slots(), "drafter target layers do not match this model");
        self.spec_hist = Some(self.gpu.alloc(self.cfg.n_slots() * hist_slot_bytes(self.cfg.n_embd))?);
        self.spec_steps = Some(self.gpu.alloc(16 * HIST_T)?);
        self.draft = Some(d);
        Ok(())
    }

    fn drafter(&self) -> &Drafter {
        self.draft.as_ref().expect("drafter")
    }

    /// The drafter's token buffer: [anchor, d1..dk] (the caller writes the anchor).
    fn draft_tokens(&self) -> u64 {
        match self.drafter() {
            Drafter::DSpark(d) => d.tokens.ptr,
            Drafter::DFlash2(d) => d.tokens.ptr,
        }
    }

    fn draft_enqueue(&self, k: usize) -> Result<()> {
        match self.drafter() {
            Drafter::DSpark(_) => self.dsp_draft(k),
            Drafter::DFlash2(_) => self.df_draft(k),
        }
    }

    fn draft_inject(&self, n: usize) -> Result<()> {
        match self.drafter() {
            Drafter::DSpark(_) => self.dsp_inject(n, FEAT_ROWS),
            Drafter::DFlash2(_) => self.df_inject(n, FEAT_ROWS),
        }
    }

    fn draft_snapshot(&self, save: bool) -> Result<()> {
        match self.drafter() {
            Drafter::DSpark(_) => self.dsp_snapshot(save),
            Drafter::DFlash2(_) => Ok(()),
        }
    }

    fn draft_snap_margin(&self) -> usize {
        match self.drafter() {
            Drafter::DSpark(_) => self.dsp_snap_margin(),
            Drafter::DFlash2(d) => d.snap_margin(self.n_ctx),
        }
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }

    /// Commit the deferred conv states of the last verify pass for its first `n` tokens.
    fn commit_conv(&self, n: usize) -> Result<()> {
        let d = self.cfg.n_embd;
        let hist = self.spec_hist.as_ref().expect("spec history").ptr;
        let mut entries = Vec::with_capacity(self.slots.len() * 4);
        for (s, slot) in self.slots.iter().enumerate() {
            let st = [slot.conv.ptr, slot.conv.at(3 * KVW * F32), slot.conv.at(6 * KVW * F32), slot.conv.at((6 * KVW + 3 * d) * F32)];
            for site in 0..4 {
                entries.push((st[site], hist + hist_offset(s, site, d) as u64, if site < 2 { KVW } else { d }));
            }
        }
        for group in entries.chunks(64) {
            let mut a = ConvCommit::default();
            for (i, &(st, x, ch)) in group.iter().enumerate() {
                a.state[i] = st;
                a.x[i] = x;
                a.ch[i] = ch as i32;
            }
            self.gpu.launch("ink_conv_commit", (d.div_ceil(256) as u32, group.len() as u32), 256, &[Arg::Conv(a), Arg::I32(n as i32)])?;
        }
        Ok(())
    }

    fn feat_opts(&self, hist: bool) -> ChunkOpts {
        let (feat, taps) = match self.drafter() {
            Drafter::DSpark(d) => (d.feat.ptr, &d.taps),
            Drafter::DFlash2(d) => (d.feat.ptr, &d.taps),
        };
        ChunkOpts {
            feat,
            taps: [taps[0], taps[1], taps[2]],
            hist: if hist { self.spec_hist.as_ref().unwrap().ptr } else { 0 },
            rt: 0,
        }
    }

    /// Capture (once) and launch a speculative-decoding graph; `f` enqueues its work.
    pub(crate) fn spec_graph(&mut self, key: (u8, usize), f: &dyn Fn(&Engine) -> Result<()>) -> Result<()> {
        if !self.use_graph {
            return f(self);
        }
        if !self.spec_graphs.contains_key(&key) {
            self.gpu.sync()?;
            self.gpu.stream.begin_capture(inkling_cuda::cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let r = f(self);
            let g = self.gpu.stream.end_capture(inkling_cuda::cudarc::driver::sys::CUgraphInstantiate_flags(0))?;
            r?;
            self.spec_graphs.insert(key, g.expect("graph"));
        }
        self.spec_graphs[&key].launch()?;
        Ok(())
    }

    /// Speculative generation with `k` drafted tokens per pass. Lossless for any `sampling`: every verify row is
    /// sampled by the regular sampler (whose random draw depends only on the seed and the position), and drafts are
    /// kept while they equal the sampled tokens; the drafts do not depend on the draws, so the first mismatch is a
    /// sample of the target with the draft removed, as in rejection sampling. `on_token` returns false to stop.
    /// Returns the generated tokens and how many prompt tokens came from the cache (the device state is reused when
    /// the prompt extends what the previous speculative generation processed).
    pub fn generate_spec(&mut self, prompt: &[u32], max_new: usize, k: usize, sampling: &crate::Sampling, on_token: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, usize)> {
        ensure!(self.draft.is_some(), "no drafter attached");
        ensure!(prompt.len() >= 2, "speculative decoding needs a prompt of at least 2 tokens");
        ensure!(prompt.iter().all(|&t| t < crate::EMBED_TOKEN_BASE), "speculative decoding does not support images yet");
        ensure!((1..=MAX_DRAFT).contains(&k), "draft length must be 1..={MAX_DRAFT}");
        ensure!(prompt.len() + max_new <= self.n_ctx, "prompt + generation exceed the context ({})", self.n_ctx);
        ensure!(self.use_graph, "speculative decoding needs CUDA graphs");
        let n = prompt.len();
        // prompt caching: continue from the device state when the new prompt extends what it holds, or from the
        // snapshot taken after the previous prompt (chat templates drop the previous answer's reasoning)
        let c = &self.cache;
        let head_new = &prompt[..n - 1];
        let start = if c.drafted && !c.tokens.is_empty() && c.tokens.len() <= head_new.len() && head_new.starts_with(&c.tokens) {
            c.tokens.len()
        } else if let Some(st) = c.snap_tokens.as_ref().filter(|st| {
            c.drafted && !st.is_empty() && st.len() <= head_new.len() && head_new.starts_with(st)
                && c.tokens.len() - st.len() <= self.snap_margin().min(self.draft_snap_margin())
        }) {
            let k0 = st.len();
            self.snapshot(false)?;
            self.draft_snapshot(false)?;
            k0
        } else {
            self.reset()?;
            0
        };
        self.cache = crate::PromptCache::default();
        self.gpu.write(self.act.sparams.ptr, inkling_cuda::as_bytes(std::slice::from_ref(sampling)))?;
        if sampling.last_n > 0 {
            let mut ring = vec![-1i32; crate::PEN_RING];
            for i in n.saturating_sub(crate::PEN_RING)..n {
                ring[i % crate::PEN_RING] = prompt[i] as i32;
            }
            self.gpu.write(self.act.pen_ring.ptr, inkling_cuda::as_bytes(&ring))?;
        }
        // prompt minus its last token: target state + drafter context
        let head = &prompt[start..n - 1];
        for (i, ch) in head.chunks(PREFILL_CHUNK).enumerate() {
            let pos0 = start + i * PREFILL_CHUNK;
            self.upload_chunk(ch, pos0)?;
            self.enqueue_chunk(ch.len(), pos0, self.feat_opts(false))?;
            self.draft_inject(ch.len())?;
        }
        self.snapshot(true)?;
        self.draft_snapshot(true)?;
        let lg = self.logit_rows()?;
        let (nv, nvu) = (self.cfg.n_vocab, self.cfg.n_vocab_unpadded);
        let step = self.pre.as_ref().unwrap().step.ptr;
        let mut last = prompt[n - 1];
        let mut pos = n - 1;
        let mut out = Vec::with_capacity(max_new);
        let mut st = SpecStats::default();
        // tokens the target has processed after the prompt head
        let mut done: Vec<u32> = Vec::with_capacity(max_new + k + 1);
        let timing = std::env::var_os("INKLING_SPEC_TIMING").is_some();
        let mut tm = [0f64; 4];
        let mut mark = |e: &Engine, i: usize, t0: &mut std::time::Instant| -> Result<()> {
            if timing {
                e.gpu.sync()?;
                tm[i] += t0.elapsed().as_secs_f64();
                *t0 = std::time::Instant::now();
            }
            Ok(())
        };
        'outer: while out.len() < max_new {
            // per pass: draft k after `last` -> verify [last, d1..dk] -> keep the agreeing prefix; near the end of
            // the context the draft shrinks (k = 0: a plain step)
            let k = k.min(self.n_ctx - 1 - pos);
            let mut t0 = std::time::Instant::now();
            if k > 0 {
                self.gpu.write(step, inkling_cuda::as_bytes(&[0i32, pos as i32, 0, 0]))?;
                self.gpu.write(self.draft_tokens(), inkling_cuda::as_bytes(&[last as i32]))?;
                self.spec_graph((0, k), &|e| e.draft_enqueue(k))?;
            } else {
                self.upload_chunk(&[last], pos)?;
            }
            mark(self, 0, &mut t0)?;
            self.spec_graph((1, k + 1), &|e| {
                e.enqueue_chunk(k + 1, 0, e.feat_opts(true))?;
                e.enqueue_head(lg, 0, k + 1)?;
                // soft-capped logits, then the regular sampler on every row (its token -> sel[r])
                let rows = (k + 1) * nv;
                e.gpu.launch("ink_softcap", (rows.div_ceil(256) as u32, 1), 256, &[Arg::Ptr(lg), Arg::I32(rows as i32), Arg::F32(e.cfg.final_softcap)])?;
                let steps = e.spec_steps.as_ref().unwrap().ptr;
                e.gpu.launch("ink_spec_steps", (1, 1), 32, &[Arg::Ptr(step), Arg::Ptr(steps), Arg::I32((k + 1) as i32)])?;
                for r in 0..=k {
                    e.gpu.launch("ink_sample", (1, 1), 1024, &[
                        Arg::Ptr(lg + (r * nv * F32) as u64), Arg::I32(nvu as i32), Arg::Ptr(e.act.sparams.ptr), Arg::Ptr(steps + (16 * r) as u64),
                        Arg::Ptr(e.act.sel.ptr), Arg::Ptr(e.act.pen_ring.ptr),
                    ])?;
                }
                Ok(())
            })?;
            mark(self, 1, &mut t0)?;
            let tgt: Vec<u32> = self.gpu.read_f32(self.act.sel.ptr, k + 1)?.iter().map(|v| v.to_bits()).collect();
            let drafts: Vec<u32> = if k > 0 { self.gpu.read_f32(self.draft_tokens() + 4, k)?.iter().map(|v| v.to_bits()).collect() } else { Vec::new() };
            let mut m = 0;
            while m < k && drafts[m] == tgt[m] {
                m += 1;
            }
            st.passes += 1;
            st.drafted += k;
            st.accepted += m;
            // the pass processed [last, d1..dm] for good
            self.spec_graph((2, m + 1), &|e| {
                e.commit_conv(m + 1)?;
                e.draft_inject(m + 1)
            })?;
            mark(self, 2, &mut t0)?;
            done.push(last);
            done.extend_from_slice(&drafts[..m]);
            pos += m + 1;
            last = tgt[m];
            for &t in drafts[..m].iter().chain(std::iter::once(&tgt[m])) {
                out.push(t);
                if !on_token(t) || out.len() >= max_new {
                    break 'outer;
                }
            }
        }
        self.gpu.sync()?;
        let mut tokens = prompt[..n - 1].to_vec();
        tokens.extend_from_slice(&done);
        self.cache = crate::PromptCache { tokens, snap_tokens: Some(prompt[..n - 1].to_vec()), drafted: true };
        self.spec_stats.passes += st.passes;
        self.spec_stats.drafted += st.drafted;
        self.spec_stats.accepted += st.accepted;
        if timing {
            let p = st.passes.max(1) as f64 * 1e-3;
            eprintln!("spec timing per pass: draft {:.2} ms, verify {:.2} ms, commit+inject {:.2} ms", tm[0] / p, tm[1] / p, tm[2] / p);
        }
        Ok((out, start))
    }
}
