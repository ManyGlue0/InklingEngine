//! Inkling 2 model configuration and the per-token execution schedule (the looped layer passes).
//!
//! Reference: llama.cpp `src/models/inkling2.cpp` + `inkling2-loop.cpp` (branch inkling2-loop-legacy-layout).
//! The legacy looped GGUF (arch `inkling2`, `inkling2.num_loops`) stores per-visit hparams arrays for 48 visits but
//! only 24 physical blocks of weights; layers [core_start, core_end) run `num_loops` passes.

use anyhow::{Result, ensure};

use crate::gguf::{Gguf, Value};

#[derive(Debug, Clone)]
pub struct Config {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_vocab: usize,
    pub n_vocab_unpadded: usize,
    pub n_phys: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub expert_weights_scale: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub rms_eps: f32,
    pub n_swa: usize,
    /// per physical layer
    pub is_swa: Vec<bool>,
    pub n_conv: usize,
    pub attn_res_block: usize,
    pub n_rel: usize,
    pub rel_extent: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_scale_floor: f32,
    pub log_scale_alpha: f32,
    pub final_softcap: f32,
    pub core_start: usize,
    pub core_end: usize,
    pub n_passes: usize,
    pub gate_rank: usize,
    pub context_length: usize,
    pub eos_token: u32,
    pub pad_token: Option<u32>,
}

/// One visit of a physical layer in the token step.
#[derive(Debug, Clone, Copy)]
pub struct Pass {
    /// cache slot: every visit has its own KV cache and short-conv state
    pub slot: usize,
    /// physical layer whose weights are used
    pub phys: usize,
    /// 0 for the prelude/coda and the first core pass; 1.. for repeated core passes (loop control active)
    pub pass: usize,
}

fn arr_u64(g: &Gguf, key: &str) -> Result<Vec<u64>> {
    match g.get(key) {
        Some(Value::Arr(a)) => Ok(a.iter().filter_map(Value::as_u64).collect()),
        Some(v) => Ok(vec![v.as_u64().unwrap_or(0)]),
        None => anyhow::bail!("missing metadata `{key}`"),
    }
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.str("general.architecture")?;
        ensure!(arch == "inkling2", "unsupported architecture `{arch}` (expected inkling2)");
        let k = |s: &str| format!("inkling2.{s}");

        let n_visit = g.u64(&k("block_count"))? as usize;
        let core_start = g.u64(&k("loop.core_start"))? as usize;
        let core_end = g.u64(&k("loop.core_end"))? as usize;
        let n_passes = g.u64(&k("num_loops"))? as usize;
        let n_core = core_end - core_start;
        let n_extra = (n_passes - 1) * n_core;
        ensure!(n_passes >= 2 && core_end + n_extra <= n_visit, "invalid looped layout");
        let n_phys = n_visit - n_extra;

        // fold the per-visit arrays back to physical layers (visit v -> physical layer)
        let phys_of_visit = |v: usize| {
            if v < core_start { v } else if v < core_end + n_extra { core_start + (v - core_start) % n_core } else { v - n_extra }
        };
        let swa_visit: Vec<u64> = arr_u64(g, &k("attention.sliding_window_pattern"))?;
        ensure!(swa_visit.len() == n_visit, "sliding_window_pattern length {} != {n_visit}", swa_visit.len());
        let mut is_swa = vec![false; n_phys];
        for (v, &s) in swa_visit.iter().enumerate() {
            is_swa[phys_of_visit(v)] = s != 0;
        }
        let heads = arr_u64(g, &k("attention.head_count"))?;
        let heads_kv = arr_u64(g, &k("attention.head_count_kv"))?;
        ensure!(heads.iter().all(|&h| h == heads[0]) && heads_kv.iter().all(|&h| h == heads_kv[0]), "per-layer head counts differ");

        let cfg = Self {
            n_embd: g.u64(&k("embedding_length"))? as usize,
            n_head: heads[0] as usize,
            n_head_kv: heads_kv[0] as usize,
            head_dim: g.u64(&k("attention.key_length"))? as usize,
            n_vocab: g.u64(&k("vocab_size"))? as usize,
            n_vocab_unpadded: g.u64(&k("vocab_size_unpadded"))? as usize,
            n_phys,
            n_expert: g.u64(&k("expert_count"))? as usize,
            n_expert_used: g.u64(&k("expert_used_count"))? as usize,
            n_ff_exp: g.u64(&k("expert_feed_forward_length"))? as usize,
            expert_weights_scale: g.f64(&k("expert_weights_scale"))? as f32,
            situ_beta: g.f64(&k("activation.situ_beta"))? as f32,
            situ_linear_beta: g.f64(&k("activation.situ_linear_beta"))? as f32,
            rms_eps: g.f64(&k("attention.layer_norm_rms_epsilon"))? as f32,
            n_swa: g.u64(&k("attention.sliding_window"))? as usize,
            is_swa,
            n_conv: g.u64(&k("shortconv.l_cache"))? as usize,
            attn_res_block: g.u64(&k("attn_res.block_size"))? as usize,
            n_rel: g.u64(&k("attention.relative_length"))? as usize,
            rel_extent: g.u64(&k("attention.relative_extent"))? as usize,
            qk_clip: g.f64(&k("attention.qk_clip"))? as f32,
            rel_clip: g.f64(&k("attention.relative_clip"))? as f32,
            log_scale_floor: g.f64(&k("attention.log_scale_floor"))? as f32,
            log_scale_alpha: g.f64(&k("attention.log_scale_alpha"))? as f32,
            final_softcap: g.f64(&k("final_logit_softcapping")).unwrap_or(0.0) as f32,
            core_start,
            core_end,
            n_passes,
            gate_rank: g.u64(&k("loop.gate_rank"))? as usize,
            context_length: g.u64(&k("context_length"))? as usize,
            eos_token: g.u64("tokenizer.ggml.eos_token_id")? as u32,
            pad_token: g.u64("tokenizer.ggml.padding_token_id").ok().map(|v| v as u32),
        };
        ensure!(core_start % cfg.attn_res_block == 0 && core_end % cfg.attn_res_block == 0, "loop core not block aligned");
        Ok(cfg)
    }

    pub fn n_core(&self) -> usize {
        self.core_end - self.core_start
    }

    /// Number of cache slots (layer visits per token).
    pub fn n_slots(&self) -> usize {
        self.n_phys + (self.n_passes - 1) * self.n_core()
    }

    /// The token step in execution order (matches `llama_model_inkling2_loop::graph`):
    /// prelude [0, core_start), then `n_passes` passes over the core, then the coda [core_end, n_phys).
    pub fn schedule(&self) -> Vec<Pass> {
        let mut s = Vec::with_capacity(self.n_slots());
        for p in 0..self.core_start {
            s.push(Pass { slot: p, phys: p, pass: 0 });
        }
        for pass in 0..self.n_passes {
            for ip in self.core_start..self.core_end {
                let slot = if pass == 0 { ip } else { self.n_phys + (pass - 1) * self.n_core() + (ip - self.core_start) };
                s.push(Pass { slot, phys: ip, pass });
            }
        }
        for p in self.core_end..self.n_phys {
            s.push(Pass { slot: p, phys: p, pass: 0 });
        }
        s
    }
}
