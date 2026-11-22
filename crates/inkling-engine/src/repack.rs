//! Load-time weight repacking into kernel-friendly "structure of arrays" layouts (no change in values).
//!
//! Q8_0 rows (34-byte blocks: f16 d, i8 q[32]) -> qs: i8 [rows][K] + d: f16 [rows][K/32]
//! NVFP4 rows (36-byte blocks: u8 d[4], u8 qs[32]) -> qs: u8 [rows][K/2] + sc: u8 [rows][K/16]
//! Rows are processed in parallel on all CPU cores.

use anyhow::{Result, ensure};

/// Repacked matrix: two planes.
pub struct Soa {
    pub qs: Vec<u8>,
    pub sc: Vec<u8>,
}

fn par_rows(rows: usize, f: impl Fn(std::ops::Range<usize>) + Sync) {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(32);
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for start in (0..rows).step_by(per) {
            let f = &f;
            s.spawn(move || f(start..(start + per).min(rows)));
        }
    });
}

/// Shared mutable output for disjoint parallel writes.
struct Out(*mut u8);
unsafe impl Sync for Out {}
unsafe impl Send for Out {}

pub fn q8_soa(src: &[u8], rows: usize, k: usize) -> Result<Soa> {
    let mut qs = vec![0u8; rows * k];
    let mut sc = vec![0u8; rows * k / 16];
    q8_soa_into(src, rows, k, &mut qs, &mut sc)?;
    Ok(Soa { qs, sc })
}

/// Q8_0 rows into caller-provided planes (qs: rows * k bytes, sc: rows * k / 16 bytes).
pub fn q8_soa_into(src: &[u8], rows: usize, k: usize, qs: &mut [u8], sc: &mut [u8]) -> Result<()> {
    ensure!(k % 32 == 0 && src.len() == rows * k / 32 * 34, "q8_soa: bad shape");
    ensure!(qs.len() == rows * k && sc.len() == rows * k / 16, "q8_soa: bad output size");
    let nb = k / 32;
    let (oq, os) = (Out(qs.as_mut_ptr()), Out(sc.as_mut_ptr()));
    par_rows(rows, |r| {
        let (oq, os) = (&oq, &os);
        for row in r {
            for b in 0..nb {
                let blk = &src[(row * nb + b) * 34..(row * nb + b + 1) * 34];
                unsafe {
                    std::ptr::copy_nonoverlapping(blk.as_ptr(), os.0.add((row * nb + b) * 2), 2);
                    std::ptr::copy_nonoverlapping(blk.as_ptr().add(2), oq.0.add(row * k + b * 32), 32);
                }
            }
        }
    });
    Ok(())
}

pub fn nvfp4_soa(src: &[u8], rows: usize, k: usize) -> Result<Soa> {
    let mut qs = vec![0u8; rows * k / 2];
    let mut sc = vec![0u8; rows * k / 16];
    nvfp4_soa_into(src, rows, k, &mut qs, &mut sc)?;
    Ok(Soa { qs, sc })
}

/// NVFP4 rows into caller-provided planes (qs: rows * k / 2 bytes, sc: rows * k / 16 bytes).
pub fn nvfp4_soa_into(src: &[u8], rows: usize, k: usize, qs: &mut [u8], sc: &mut [u8]) -> Result<()> {
    ensure!(k % 64 == 0 && src.len() == rows * k / 64 * 36, "nvfp4_soa: bad shape");
    ensure!(qs.len() == rows * k / 2 && sc.len() == rows * k / 16, "nvfp4_soa: bad output size");
    let nb = k / 64;
    let (oq, os) = (Out(qs.as_mut_ptr()), Out(sc.as_mut_ptr()));
    par_rows(rows, |r| {
        let (oq, os) = (&oq, &os);
        for row in r {
            for b in 0..nb {
                let blk = &src[(row * nb + b) * 36..(row * nb + b + 1) * 36];
                unsafe {
                    std::ptr::copy_nonoverlapping(blk.as_ptr(), os.0.add((row * nb + b) * 4), 4);
                    std::ptr::copy_nonoverlapping(blk.as_ptr().add(4), oq.0.add((row * nb + b) * 32), 32);
                }
            }
        }
    });
    // the expert kernels decode UE4M3 scales with a normal-number bit trick (no subnormals, no zero, no sign bit)
    ensure!(
        sc.iter().all(|&b| b & 0x78 != 0 && b & 0x80 == 0),
        "nvfp4_soa: subnormal/zero/negative block scales are not supported by the fast expert kernels"
    );
    Ok(())
}
