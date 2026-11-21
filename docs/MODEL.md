# Inkling 2 Mini: forward-pass specification

This is the exact math our engine implements, derived from llama.cpp `src/models/inkling2.cpp` + `inkling2-loop.cpp`
(Unsloth branch `inkling2-loop-legacy-layout`, MIT) and `llama-graph.cpp` (`build_moe_ffn`). Notation: vectors are
per token; `D = 2304`, `H = 24` query heads, `Hkv = 8`, `dh = 128`, `R = 16` (relative dim), `E = 128` experts,
`k = 8`, `F = 1152` (expert FFN), `C = 4` (short-conv taps), `eps = 1e-6`.

## Primitives
- `rms(x) = x / sqrt(mean(x²) + 1e-6)`; `rmsw(x, w) = rms(x) ⊙ w` (model eps 1e-6 too).
- `clip(x, s, θ)` per head row: `y = x ⊙ s` (if s given); `n = ‖y‖₂ + 1e-6`; `y · min(1, θ/n)`.
- **Depth attention** `mix(B, c, Q)`: states `S = {b ∈ B} ∪ {c}`, `Ŝ = rms(S)` row-wise (no weight).
  For each query row `q ∈ Q` (1, 3 or 4 rows): `p = softmax_s(q·Ŝ_s / sqrt(D))`, `out_q = Σ_s p_s Ŝ_s`.
  The order of the states doesn't matter. **Values are the normalized states.**
- **Short conv** with state (the last C−1 inputs per slot and channel group), weight `w[C][ch]`:
  `x_full = [past(3), x]`, `y_t = Σ_{j=0..3} x_full[t+j] ⊙ w[j]` (`w[3]` multiplies the current input).
  Dynamic taps (full-attention K/V only): `tap = W_dyn x_in ∈ R^{Hkv·C}`, and
  `y_t += Σ_{lag=0..3} x[t−lag] ⊙ tap[head][lag]` (broadcast over the 128 channels of each kv head).
  Output: `(x + y) / sqrt(m·C + 1)` with `m = 2` if dynamic, else `1`. The state then keeps the last 3 inputs `x`.
  Each visit (cache slot) keeps its own state for 4 streams: K (1024 ch), V (1024), attn-out (2304), ffn-out (2304).
- **SiTU** `situ(g, u) = β·tanh(g/β)·σ(g) · λ·tanh(u/λ)` with `β = 4`, `λ = 25`.

## Token step
```
x = rmsw(embed[token], token_embd_norm)          # Q8_0 embedding row
blocks = []
for pass in schedule:  (48 visits, see Config::schedule)
    x = layer(pass, x, blocks)
out = rmsw(mix(blocks, x, output_res_query), output_norm)
logits = 30·tanh((W_out out)/30)[:200058]         # Q8_0 head, padded vocab dropped
```

### Loop structure
Prelude = physical layers 0..5, core = 6..17 run 3 passes, coda = 18..23. Every core pass starts from the
**prelude output** `x_p` with `blocks = prelude_blocks`. For passes 1 and 2, the "bank" of the previous pass is its
blocks after the first core block (the block pushed at core layer 12), plus that pass's output `x_prev`. Each core
layer `ip` of passes ≥ 1 computes messages `m = mix(bank, x_prev, Qmsg_ip)` with 4 query rows
`[res[2i], kv[2i], kv[2i+1], res[2i+1]]` (q, k, v, ffn) from `loop_res_query` / `loop_kv_res_query`, `i = ip − 6`.

**Control** (rank 64): `g = W_down [rms(c); rms(m)]` (4608→64), `c' = (1 − tanh(W_forget g)) ⊙ c + tanh(W_gain g) ⊙ m`.
Controls: q ← loop_attn_*, k ← loop_k_*, v ← loop_v_*, ffn ← loop_mlp_*.

### Layer (physical weights `L`, slot `s`, position `pos`)
```
inpSA = x
if layer > 0: a = mix(blocks, x, L.attn_res_query)   # SWA: 1 query → a_q=a_k=a_v ; full: 3 queries → a_q,a_k,a_v
else: a_q = a_k = a_v = x
if pass ≥ 1: a_j = control(a_j, m_j) for j in q,k,v
SWA:  a_j = rmsw(a_j, L.attn_norm)                 (one shared input when pass = 0)
full: a_q, a_k, a_v = rmsw(·, L.attn_{q,k,v}_inp_norm)

q = Wq a_q  (24×128)          r = Wrel a_q  (24×16)       gate = Wgate a_q (3072)    sink = Wsink a_q (24, BF16)
k = conv(Wk a_k, L.attn_k_conv, dyn = full ? Wkconv a_k : none)      (1024)
v = conv(Wv a_v, L.attn_v_conv, dyn = full ? Wvconv a_v : none)      (1024)
q = clip(q, L.attn_q_clip, 22.6274)   k = clip(k, none, 22.6274)   v = rms(v) per head   r = clip(r, L.attn_rel_clip, 8)
full: τ = 1 + 0.1·ln(max(1, (pos+1)/8192)); q *= τ; r *= τ
KV cache[s] ← (k, v)
attention per head h (kv head h/3), over cached positions j (SWA: pos − j < 1024):
    d = pos − j ; bias_h(d) = d < 1024 ? r_h · L.attn_rel_bias[:, d] : 0
    logit_j = (q_h · k_j)/sqrt(128) + bias_h(d)
    o_h = Σ_j softmax_j∪{sink}(logit) v_j       # extra sink logit = sink_h with zero value
o = (o ⊙ σ(gate))
attn = conv(Wo o, L.attn_post_conv)
if layer % 6 == 0: blocks.push(inpSA); x = attn
else: x = inpSA + attn
f = mix(blocks, x, L.ffn_res_query); if pass ≥ 1: f = control(f, m_ffn)
f = rmsw(f, L.ffn_norm)
MoE(f):  ℓ = W_router f (F32);  p = σ(ℓ);  top-8 by p + exp_probs_b;  w = p_sel / (Σ p_sel + 1e-20) · 8
         per expert e: [g; u] = s_gu[e] · (W_gate_up[e] f)   (gate rows first, then up)
                       h = situ(g, u);  y_e = s_down[e] · (W_down[e] h)
         y = Σ w_e y_e
x = x + conv(y, L.ffn_post_conv)
```

## Weight formats
- Q8_0: blocks of 32: `f16 d, i8 q[32]`, value `d·q`.
- NVFP4: blocks of 64 (36 bytes): `u8 d[4]` (UE4M3 per 16 values), `u8 qs[32]`. Sub-block `s`, byte `j` (0..7):
  low nibble → element `16s + j`, high nibble → element `16s + j + 8`. Value `E2M1x2[n] · ue4m3(d) · 0.5`,
  `E2M1x2 = {0,1,2,3,4,6,8,12, 0,−1,−2,−3,−4,−6,−8,−12}`, `ue4m3(x) = 0 if x∈{0,0x7F}; exp=(x>>3)&15, man=x&7;
  exp=0: man·2⁻⁹, else (1+man/8)·2^(exp−7)`. NVFP4 expert outputs are unscaled (up to ~2.7e3): accumulate in F32.
- Per-expert F32 scales: `ffn_gate_up_exps.scale`, `ffn_down_exps.scale` (NVFP4 layers only; Q8_0 layers 20–23 have none).
