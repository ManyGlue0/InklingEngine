//! Parallel streams (`--parallel N`): N independent sequences, each with its own KV cache, conv states and prompt
//! cache over n_ctx positions (the context given at load, divided by N). A stream is prefilled on its own
//! (`par_begin`, the regular chunked prefill on the selected stream); then `par_step` decodes one token for each of
//! several streams in a single batched pass. The pass is the short-chunk prefill path with a row table (stream,
//! position per row): weights, router and experts are read once for all rows, while the KV writes, attention and
//! short convs of each row use its own stream's state.

use anyhow::{Result, ensure};
use inkling_cuda::{Arg, DevBuf, Gpu};

use crate::prefill::ChunkOpts;
use crate::{Engine, F32, ImageEmbed, PEN_RING, Sampling, Slot, SlotBufs, View};

/// Most streams one pass decodes (the short-chain MoE kernel takes up to 8 rows).
pub const MAX_STREAMS: usize = 8;

/// A stream between steps: its next input token and that token's position.
#[derive(Clone, Copy)]
struct Seq {
    token: u32,
    pos: usize,
}

pub(crate) struct ParBufs {
    rt: DevBuf,      // i32 [MAX_STREAMS][2]: stream, position of each row
    steps: DevBuf,   // i32 [MAX_STREAMS][4]: sampler step of each row
    sparams: DevBuf, // Sampling [n]
    rings: DevBuf,   // i32 [n][PEN_RING]
    mask_on: DevBuf, // i32 [n]
    masks: DevBuf,   // u32 [n][words]
    words: usize,
    seqs: Vec<Option<Seq>>,
}

impl ParBufs {
    pub(crate) fn new(gpu: &Gpu, cfg: &inkling_core::Config, n: usize) -> Result<Self> {
        let words = cfg.n_vocab.div_ceil(32);
        let b = Self {
            rt: gpu.alloc(MAX_STREAMS * 2 * 4)?,
            steps: gpu.alloc(MAX_STREAMS * 4 * 4)?,
            sparams: gpu.alloc(n * std::mem::size_of::<Sampling>())?,
            rings: gpu.alloc(n * PEN_RING * 4)?,
            mask_on: gpu.alloc(n * 4)?,
            masks: gpu.alloc(n * words * 4)?,
            words,
            seqs: vec![None; n],
        };
        gpu.write(b.mask_on.ptr, inkling_cuda::as_bytes(&vec![0i32; n]))?;
        Ok(b)
    }
}

impl SlotBufs {
    /// Stream `s`'s part of this slot (of `n` streams).
    pub(crate) fn view(&self, s: usize, n: usize) -> Slot {
        let part = |b: &DevBuf| {
            let per = b.bytes / n;
            View { ptr: b.ptr + (s * per) as u64, bytes: per }
        };
        Slot { k: part(&self.k), v: part(&self.v), conv: part(&self.conv), kv_cap: self.kv_cap, kv_fmt: self.kv_fmt }
    }
}

impl Engine {
    /// Number of parallel streams (1 without --parallel).
    pub fn n_streams(&self) -> usize {
        self.n_streams
    }

    /// Bytes per stream of slot `slot`'s K cache, V cache and (in floats) conv states: the row-table strides.
    pub(crate) fn stream_strides(&self, slot: usize) -> (usize, usize, usize) {
        let b = &self.slot_bufs[slot];
        let n = self.n_streams;
        (b.k.bytes / n, b.v.bytes / n, b.conv.bytes / n / F32)
    }

    /// Stream 0's view of a slot: the base the row-table kernels offset from.
    pub(crate) fn slot_base(&self, slot: usize) -> Slot {
        self.slot_bufs[slot].view(0, self.n_streams)
    }

    /// Make stream `s` the one single-sequence calls work on (its KV cache, conv states, prompt cache and graphs).
    pub fn select_stream(&mut self, s: usize) -> Result<()> {
        ensure!(s < self.n_streams, "stream {s} out of range ({} streams)", self.n_streams);
        if s == self.stream {
            return Ok(());
        }
        self.gpu.sync()?;
        let cur = self.stream;
        let park = |e: &mut Engine, i: usize| {
            let st = &mut e.streams[i];
            std::mem::swap(&mut e.cache, &mut st.cache);
            std::mem::swap(&mut e.graph, &mut st.graph);
            std::mem::swap(&mut e.graph_pre, &mut st.graph_pre);
            std::mem::swap(&mut e.graph_dec, &mut st.graph_dec);
        };
        park(self, cur); // the engine's fields are now empty
        park(self, s); // ... and hold stream s's
        self.slots = self.slot_bufs.iter().map(|b| b.view(s, self.n_streams)).collect();
        self.stream = s;
        Ok(())
    }

    fn cache_of(&mut self, s: usize) -> &mut crate::PromptCache {
        if s == self.stream { &mut self.cache } else { &mut self.streams[s].cache }
    }

    /// Start stream `s` on `prompt` (prompt caching per stream, images as in `generate_mm`): the prompt minus its last
    /// token is prefilled, and the last token is the stream's first input for `par_step`. Returns how many prompt
    /// tokens came from the stream's cache.
    pub fn par_begin(&mut self, s: usize, prompt: &[u32], embeds: &[(usize, ImageEmbed)], sampling: &Sampling) -> Result<usize> {
        ensure!(self.use_graph, "parallel decoding needs CUDA graphs");
        ensure!(!prompt.is_empty(), "empty prompt");
        ensure!(*prompt.last().unwrap() < crate::EMBED_TOKEN_BASE, "the prompt must not end inside an image");
        for (pos, e) in embeds {
            ensure!(e.data.len() == e.n * self.cfg.n_embd, "image embedding width");
            ensure!(pos + e.n <= prompt.len() && prompt[*pos..pos + e.n].iter().all(|&t| t >= crate::EMBED_TOKEN_BASE), "image embedding positions do not match the prompt");
        }
        ensure!(prompt.iter().filter(|&&t| t >= crate::EMBED_TOKEN_BASE).count() == embeds.iter().map(|(_, e)| e.n).sum::<usize>(), "prompt has image positions without embeddings");
        ensure!(prompt.len() < self.n_ctx, "the prompt exceeds the context of a stream ({})", self.n_ctx);
        self.select_stream(s)?;
        self.par.as_mut().expect("parallel buffers").seqs[s] = None;
        let par = self.par.as_ref().unwrap();
        self.gpu.write(par.sparams.at(s * std::mem::size_of::<Sampling>()), inkling_cuda::as_bytes(std::slice::from_ref(sampling)))?;
        if sampling.last_n > 0 {
            let n = prompt.len();
            let mut ring = vec![-1i32; PEN_RING];
            for i in n.saturating_sub(PEN_RING)..n {
                ring[i % PEN_RING] = prompt[i] as i32;
            }
            self.gpu.write(par.rings.at(s * PEN_RING * 4), inkling_cuda::as_bytes(&ring))?;
        }
        self.gpu.write(par.mask_on.at(s * 4), inkling_cuda::as_bytes(&[0i32]))?;
        self.embeds = embeds.to_vec();
        let r = self.prepare_prompt(prompt);
        self.embeds.clear();
        let reused = r?;
        let n = prompt.len();
        self.par.as_mut().unwrap().seqs[s] = Some(Seq { token: prompt[n - 1], pos: n - 1 });
        Ok(reused)
    }

    /// Token constraint of stream `s` for its next steps: allowed tokens, one bit each over the unpadded vocabulary
    /// (32 per word), or None for no constraint.
    pub fn par_mask(&mut self, s: usize, mask: Option<Vec<u32>>) -> Result<()> {
        let par = self.par.as_ref().expect("parallel buffers");
        ensure!(s < par.seqs.len(), "stream {s} out of range");
        match mask {
            Some(mut bits) => {
                bits.resize(par.words, 0);
                self.gpu.write(par.masks.at(s * par.words * 4), inkling_cuda::as_bytes(&bits))?;
                self.gpu.write(par.mask_on.at(s * 4), inkling_cuda::as_bytes(&[1i32]))
            }
            None => self.gpu.write(par.mask_on.at(s * 4), inkling_cuda::as_bytes(&[0i32])),
        }
    }

    /// Positions stream `s` has left (its next input's position counts as used).
    pub fn par_room(&self, s: usize) -> usize {
        self.par.as_ref().and_then(|p| p.seqs.get(s).copied().flatten()).map_or(0, |q| self.n_ctx - q.pos)
    }

    /// One batched decode step: every stream in `rows` (distinct, each started with `par_begin`) consumes its pending
    /// token and samples the next one, which becomes its next input. Returns the sampled tokens in row order.
    pub fn par_step(&mut self, rows: &[usize]) -> Result<Vec<u32>> {
        let t = rows.len();
        ensure!(t >= 1 && t <= self.n_streams.min(MAX_STREAMS), "par_step: {t} rows for {} streams", self.n_streams);
        let par = self.par.as_ref().expect("parallel buffers");
        let mut tokens = Vec::with_capacity(t);
        let mut rt = Vec::with_capacity(2 * t);
        let mut steps = Vec::with_capacity(4 * t);
        for (r, &s) in rows.iter().enumerate() {
            ensure!(!rows[..r].contains(&s), "par_step: stream {s} twice");
            let q = par.seqs.get(s).copied().flatten().ok_or_else(|| anyhow::anyhow!("par_step: stream {s} was not started"))?;
            ensure!(q.pos < self.n_ctx, "stream {s} is out of context ({})", self.n_ctx);
            tokens.push(q.token as i32);
            rt.extend_from_slice(&[s as i32, q.pos as i32]);
            steps.extend_from_slice(&[q.token as i32, q.pos as i32, r as i32, 0]);
        }
        let (rt_ptr, steps_ptr, sp, rings, on, masks, words) =
            (par.rt.ptr, par.steps.ptr, par.sparams.ptr, par.rings.ptr, par.mask_on.ptr, par.masks.ptr, par.words);
        let pre_tokens = self.pre.as_ref().expect("prefill buffers").tokens.ptr;
        self.gpu.write(pre_tokens, inkling_cuda::as_bytes(&tokens))?;
        self.gpu.write(rt_ptr, inkling_cuda::as_bytes(&rt))?;
        self.gpu.write(steps_ptr, inkling_cuda::as_bytes(&steps))?;
        let lg = self.logit_rows()?;
        let sel = self.act.sel.ptr;
        self.spec_graph((3, t), &|e| {
            e.enqueue_chunk(t, 0, ChunkOpts { rt: rt_ptr, ..Default::default() })?;
            e.enqueue_head(lg, 0, t)?;
            let (nv, nvu) = (e.cfg.n_vocab, e.cfg.n_vocab_unpadded);
            let rows = t * nv;
            e.gpu.launch("ink_softcap", (rows.div_ceil(256) as u32, 1), 256, &[Arg::Ptr(lg), Arg::I32(rows as i32), Arg::F32(e.cfg.final_softcap)])?;
            e.gpu.launch("ink_apply_mask_rows", (nvu.div_ceil(1024) as u32, t as u32), 1024, &[
                Arg::Ptr(lg), Arg::I32(nv as i32), Arg::I32(nvu as i32), Arg::Ptr(on), Arg::Ptr(masks), Arg::I32(words as i32), Arg::Ptr(rt_ptr),
            ])?;
            e.gpu.launch("ink_sample_rows", (t as u32, 1), 1024, &[
                Arg::Ptr(lg), Arg::I32(nv as i32), Arg::I32(nvu as i32), Arg::Ptr(sp), Arg::Ptr(steps_ptr), Arg::Ptr(sel), Arg::Ptr(rings), Arg::Ptr(rt_ptr),
            ])
        })?;
        let out: Vec<u32> = self.gpu.read_f32(sel, t)?.iter().map(|v| v.to_bits()).collect();
        for (r, &s) in rows.iter().enumerate() {
            let q = self.par.as_mut().unwrap().seqs[s].as_mut().unwrap();
            let fed = q.token;
            q.token = out[r];
            q.pos += 1;
            self.cache_of(s).tokens.push(fed);
        }
        Ok(out)
    }
}
