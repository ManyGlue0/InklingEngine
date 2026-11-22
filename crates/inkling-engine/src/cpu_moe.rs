//! CPU expert FFN for MoE layers offloaded to system RAM (decode path).
//!
//! The expert weights stay in the memory-mapped GGUF file in their native block formats (no copy):
//!   NVFP4: 36-byte blocks of 64 values, UE4M3 scales d[4] (one per 16 values) then qs[32]; sub-block s uses bytes
//!          8s..8s+7, byte j holding value 16s + j (low nibble) and 16s + j + 8 (high nibble). Plus an F32 scale per
//!          expert.
//!   Q8_0:  34-byte blocks of 32 values (f16 d, int8 q[32]).
//! The math matches the GPU kernels (experts.cu): q8_1 activations, E2M1 values doubled to integers with a halved
//! block scale, int8 dot products, SiTU gating, the hidden vector quantized to int8 per 16 features, and the
//! weighted sum over the selected experts. Work runs on a persistent thread pool in two phases (gate_up + hidden,
//! then down), AVX2 when available with a scalar fallback.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::weights::ExpertFmt;

/// One offloaded expert tensor [E][rows][K] inside the mapped GGUF.
#[derive(Clone, Copy)]
pub struct CpuExp {
    pub data: *const u8,
    pub fmt: ExpertFmt,
    pub rows: usize,
    pub k: usize,
}

unsafe impl Send for CpuExp {}
unsafe impl Sync for CpuExp {}

impl CpuExp {
    pub fn row_bytes(&self) -> usize {
        match self.fmt {
            ExpertFmt::Nvfp4 => self.k / 64 * 36,
            ExpertFmt::Q8_0 => self.k / 32 * 34,
        }
    }
    fn row(&self, e: usize, r: usize) -> *const u8 {
        unsafe { self.data.add((e * self.rows + r) * self.row_bytes()) }
    }
}

/// An offloaded layer: gate_up [E][2F][D], down [E][D][F], optional per-expert scales.
pub struct CpuLayer {
    pub gate_up: CpuExp,
    pub down: CpuExp,
    pub gu_scale: Option<Vec<f32>>,
    pub down_scale: Option<Vec<f32>>,
}

/// Activations prepared for one weight format: int8 values in the order the kernel consumes them and one f32
/// scale per int32 lane of each block (8 lanes per block).
struct Acts {
    q: Vec<i8>,
    s8: Vec<f32>,
}

/// `x` (int8, quantized in groups of `g` values with scales `xs`) arranged for rows of format `fmt` and length k.
fn prep_acts(x: &[i8], xs: &[f32], g: usize, fmt: ExpertFmt, k: usize, out: &mut Acts) {
    out.q.clear();
    out.s8.clear();
    match fmt {
        ExpertFmt::Nvfp4 => {
            for b in 0..k / 64 {
                let xb = &x[64 * b..64 * b + 64];
                // low nibbles: byte p -> value 16 (p / 8) + p % 8; high nibbles: + 8
                for half in 0..2 {
                    for p in 0..32 {
                        out.q.push(xb[16 * (p / 8) + p % 8 + 8 * half]);
                    }
                }
                // int32 lanes 2s, 2s+1 belong to sub-block s
                for lane in 0..8 {
                    out.s8.push(xs[(64 * b + 16 * (lane / 2)) / g]);
                }
            }
        }
        ExpertFmt::Q8_0 => {
            out.q.extend_from_slice(&x[..k]);
            for b in 0..k / 32 {
                for lane in 0..8 {
                    out.s8.push(xs[(32 * b + 4 * lane) / g]);
                }
            }
        }
    }
}

#[inline]
fn ue4m3(x: u8) -> f32 {
    // as ue4m3_fast on the GPU: 2^(e-8) (1 + m/8) (half the real value: E2M1 values are doubled), subnormals m/1024
    let (e, m) = ((x >> 3) & 15, x & 7);
    if x == 0 || x == 0x7F {
        0.0
    } else if e == 0 {
        m as f32 / 1024.0
    } else {
        f32::from_bits(((e as u32 + 119) << 23) | ((m as u32) << 20))
    }
}

const E2M1: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn dot_scalar(w: CpuExp, row: *const u8, a: &Acts) -> f32 {
    let mut acc = 0f32;
    match w.fmt {
        ExpertFmt::Nvfp4 => {
            for b in 0..w.k / 64 {
                let blk = unsafe { std::slice::from_raw_parts(row.add(36 * b), 36) };
                let xq = &a.q[64 * b..64 * b + 64];
                let mut lanes = [0i32; 8];
                for p in 0..32 {
                    let q = blk[4 + p];
                    lanes[p / 4] += E2M1[(q & 15) as usize] as i32 * xq[p] as i32 + E2M1[(q >> 4) as usize] as i32 * xq[32 + p] as i32;
                }
                for (lane, &v) in lanes.iter().enumerate() {
                    acc += v as f32 * ue4m3(blk[lane / 2]) * a.s8[8 * b + lane];
                }
            }
        }
        ExpertFmt::Q8_0 => {
            for b in 0..w.k / 32 {
                let blk = unsafe { std::slice::from_raw_parts(row.add(34 * b), 34) };
                let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
                let xq = &a.q[32 * b..32 * b + 32];
                for lane in 0..8 {
                    let mut s = 0i32;
                    for i in 4 * lane..4 * lane + 4 {
                        s += blk[2 + i] as i8 as i32 * xq[i] as i32;
                    }
                    acc += s as f32 * d * a.s8[8 * b + lane];
                }
            }
        }
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod avx2 {
    use super::{Acts, CpuExp, ExpertFmt};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// 4 UE4M3 bytes -> lanes [d0, d0, d1, d1, d2, d2, d3, d3] (ue4m3 semantics of the scalar path)
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn ue4m3x8(d: u32) -> __m256 {
        let x = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(d as i32), _mm256_setr_epi32(0, 0, 8, 8, 16, 16, 24, 24)), _mm256_set1_epi32(0xFF));
        let e = _mm256_and_si256(_mm256_srli_epi32(x, 3), _mm256_set1_epi32(15));
        let m = _mm256_and_si256(x, _mm256_set1_epi32(7));
        let normal = _mm256_castsi256_ps(_mm256_or_si256(_mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(119)), 23), _mm256_slli_epi32(m, 20)));
        let sub = _mm256_mul_ps(_mm256_cvtepi32_ps(m), _mm256_set1_ps(1.0 / 1024.0));
        let is_sub = _mm256_castsi256_ps(_mm256_cmpeq_epi32(e, _mm256_setzero_si256()));
        let v = _mm256_blendv_ps(normal, sub, is_sub);
        let zero = _mm256_or_si256(_mm256_cmpeq_epi32(x, _mm256_setzero_si256()), _mm256_cmpeq_epi32(x, _mm256_set1_epi32(0x7F)));
        _mm256_andnot_ps(_mm256_castsi256_ps(zero), v)
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot(w: CpuExp, row: *const u8, a: &Acts) -> f32 {
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();
        match w.fmt {
            ExpertFmt::Nvfp4 => {
                let lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12, 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
                let m4 = _mm256_set1_epi8(0x0F);
                for b in 0..w.k / 64 {
                    let blk = row.add(36 * b);
                    let q = _mm256_loadu_si256(blk.add(4) as *const __m256i);
                    let wl = _mm256_shuffle_epi8(lut, _mm256_and_si256(q, m4));
                    let wh = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
                    let xl = _mm256_loadu_si256(a.q.as_ptr().add(64 * b) as *const __m256i);
                    let xh = _mm256_loadu_si256(a.q.as_ptr().add(64 * b + 32) as *const __m256i);
                    // |w| (<= 12) x sign-adjusted x: int16 pair sums stay below 2 * 12 * 128
                    let pl = _mm256_maddubs_epi16(_mm256_abs_epi8(wl), _mm256_sign_epi8(xl, wl));
                    let ph = _mm256_maddubs_epi16(_mm256_abs_epi8(wh), _mm256_sign_epi8(xh, wh));
                    let s = _mm256_madd_epi16(_mm256_add_epi16(pl, ph), ones);
                    let sc = _mm256_mul_ps(ue4m3x8((blk as *const u32).read_unaligned()), _mm256_loadu_ps(a.s8.as_ptr().add(8 * b)));
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sc, acc);
                }
            }
            ExpertFmt::Q8_0 => {
                for b in 0..w.k / 32 {
                    let blk = row.add(34 * b);
                    let d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((blk as *const u16).read_unaligned() as i32)));
                    let wv = _mm256_loadu_si256(blk.add(2) as *const __m256i);
                    let xv = _mm256_loadu_si256(a.q.as_ptr().add(32 * b) as *const __m256i);
                    // int8 x int8 via |w| and sign-adjusted x; |w| <= 127 can reach 2 * 127 * 128 > i16::MAX only
                    // for w = -128, which Q8_0 never stores (symmetric quantization)
                    let p = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(xv, wv));
                    let s = _mm256_madd_epi16(p, ones);
                    let sc = _mm256_mul_ps(_mm256_set1_ps(d), _mm256_loadu_ps(a.s8.as_ptr().add(8 * b)));
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sc, acc);
                }
            }
        }
        hsum(acc)
    }
}

/// A persistent pool: `run(f)` calls f(i) for every worker i in 0..n (the caller is worker 0) and returns when all
/// are done. Idle workers spin briefly (back-to-back offloaded layers arrive ~100 us apart), then sleep.
pub struct Pool {
    n: usize,
    shared: Arc<PoolShared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

struct PoolShared {
    epoch: AtomicUsize,
    pending: AtomicUsize,
    job: Mutex<Option<JobPtr>>,
    wake: Condvar,
    stop: AtomicBool,
}

#[derive(Clone, Copy)]
struct JobPtr(*const (dyn Fn(usize) + Sync));
unsafe impl Send for JobPtr {}
unsafe impl Sync for JobPtr {}

impl Pool {
    pub fn new(n: usize) -> Self {
        let n = n.max(1);
        let shared = Arc::new(PoolShared { epoch: AtomicUsize::new(0), pending: AtomicUsize::new(0), job: Mutex::new(None), wake: Condvar::new(), stop: AtomicBool::new(false) });
        let threads = (1..n)
            .map(|i| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("inkling-cpu-{i}"))
                    .spawn(move || worker(sh, i))
                    .expect("spawn CPU worker")
            })
            .collect();
        Self { n, shared, threads }
    }

    pub fn threads(&self) -> usize {
        self.n
    }

    pub fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        if self.n == 1 {
            f(0);
            return;
        }
        // SAFETY: the job pointer is only used until `pending` reaches 0, which happens before this returns
        let ptr: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
        *self.shared.job.lock().unwrap() = Some(JobPtr(ptr));
        self.shared.pending.store(self.n - 1, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::AcqRel);
        self.shared.wake.notify_all();
        f(0);
        while self.shared.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }
}

fn worker(sh: Arc<PoolShared>, i: usize) {
    let mut seen = 0usize;
    loop {
        // spin, then sleep until the generation changes
        let t0 = Instant::now();
        let mut g = sh.epoch.load(Ordering::Acquire);
        while g == seen && t0.elapsed() < Duration::from_micros(400) {
            std::hint::spin_loop();
            g = sh.epoch.load(Ordering::Acquire);
        }
        if g == seen {
            let mut lock = sh.job.lock().unwrap();
            while sh.epoch.load(Ordering::Acquire) == seen && !sh.stop.load(Ordering::Acquire) {
                lock = sh.wake.wait(lock).unwrap();
            }
            drop(lock);
            g = sh.epoch.load(Ordering::Acquire);
        }
        if sh.stop.load(Ordering::Acquire) {
            return;
        }
        seen = g;
        let job = sh.job.lock().unwrap().expect("job");
        unsafe { (*job.0)(i) };
        sh.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        {
            let _l = self.shared.job.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// Inputs exported by the GPU for one offloaded layer visit (host-mapped memory, written by ink_moe_export).
#[repr(C)]
pub struct MoeIo {
    pub xq: [i8; 2304],
    pub xd: [f32; 2304 / 32],
    pub sel: [i32; 8],
    pub wts: [f32; 8],
    /// output: weighted expert sum (read by the GPU post conv)
    pub y: [f32; 2304],
}

/// CPU expert executor: layers, pool and scratch.
pub struct CpuMoe {
    pub layers: Vec<Option<CpuLayer>>,
    pool: Pool,
    avx2: bool,
    d: usize,
    f: usize,
    beta: f32,
    lb: f32,
    scratch: Mutex<Scratch>,
}

#[derive(Default)]
struct Scratch {
    hq: Vec<i8>,
    hd: Vec<f32>,
}

/// Shared mutable pointer for disjoint parallel writes.
#[derive(Clone, Copy)]
struct SPtr<T>(*mut T);
unsafe impl<T> Send for SPtr<T> {}
unsafe impl<T> Sync for SPtr<T> {}

impl CpuMoe {
    pub fn new(layers: Vec<Option<CpuLayer>>, d: usize, f: usize, beta: f32, lb: f32, threads: usize) -> Self {
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        Self { layers, pool: Pool::new(threads), avx2, d, f, beta, lb, scratch: Mutex::new(Scratch::default()) }
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    pub fn simd(&self) -> &'static str {
        if self.avx2 { "AVX2" } else { "scalar" }
    }

    #[inline]
    fn dot(&self, w: CpuExp, row: *const u8, a: &Acts) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            return unsafe { avx2::dot(w, row, a) };
        }
        dot_scalar(w, row, a)
    }

    /// Expert FFN of offloaded layer `li` for the exported inputs; writes io.y.
    pub fn run(&self, li: usize, io: &mut MoeIo) {
        let layer = self.layers[li].as_ref().expect("layer not offloaded");
        let (d, f) = (self.d, self.f);
        let k = io.sel.len();
        let mut sc = self.scratch.lock().unwrap();
        sc.hq.resize(k * f, 0);
        sc.hd.resize(k * f / 16, 0.0);
        let mut ax = Acts { q: Vec::with_capacity(d), s8: Vec::with_capacity(d / 4) };
        prep_acts(&io.xq, &io.xd, 32, layer.gate_up.fmt, d, &mut ax);
        let (hq, hd) = (SPtr(sc.hq.as_mut_ptr()), SPtr(sc.hd.as_mut_ptr()));
        let sel: Vec<usize> = io.sel.iter().map(|&e| e as usize).collect();
        let nt = self.pool.threads();
        // phase 1: units of (expert slot, 16 features): 16 gate + 16 up rows -> 16 quantized hidden values
        let units = k * (f / 16);
        self.pool.run(&|ti| {
            let (hq, hd) = (hq, hd);
            for u in (ti * units / nt)..((ti + 1) * units / nt) {
                let (t, fg) = (u / (f / 16), u % (f / 16));
                let e = sel[t];
                let s = layer.gu_scale.as_ref().map_or(1.0, |v| v[e]);
                let mut h = [0f32; 16];
                for (i, hv) in h.iter_mut().enumerate() {
                    let fi = 16 * fg + i;
                    let g = self.dot(layer.gate_up, layer.gate_up.row(e, fi), &ax) * s;
                    let up = self.dot(layer.gate_up, layer.gate_up.row(e, f + fi), &ax) * s;
                    let a = self.beta * (g / self.beta).tanh() / (1.0 + (-g).exp());
                    *hv = a * (self.lb * (up / self.lb).tanh());
                }
                let amax = h.iter().fold(0f32, |m, v| m.max(v.abs()));
                let dq = amax / 127.0;
                unsafe {
                    for (i, &v) in h.iter().enumerate() {
                        *hq.0.add(t * f + 16 * fg + i) = if amax == 0.0 { 0 } else { (v / dq).round() as i8 };
                    }
                    *hd.0.add(t * (f / 16) + fg) = dq;
                }
            }
        });
        // phase 2: output rows n: y_n = sum_t wts_t s_e (W_e[n] . h_t)
        let hacts: Vec<Acts> = (0..k)
            .map(|t| {
                let mut a = Acts { q: Vec::with_capacity(f), s8: Vec::with_capacity(f / 4) };
                prep_acts(&sc.hq[t * f..(t + 1) * f], &sc.hd[t * (f / 16)..(t + 1) * (f / 16)], 16, layer.down.fmt, f, &mut a);
                a
            })
            .collect();
        let y = SPtr(io.y.as_mut_ptr());
        let wts = io.wts;
        self.pool.run(&|ti| {
            let y = y;
            for n in (ti * d / nt)..((ti + 1) * d / nt) {
                let mut acc = 0f32;
                for t in 0..k {
                    let e = sel[t];
                    let s = layer.down_scale.as_ref().map_or(1.0, |v| v[e]);
                    acc += wts[t] * s * self.dot(layer.down, layer.down.row(e, n), &hacts[t]);
                }
                unsafe { *y.0.add(n) = acc };
            }
        });
    }
}
