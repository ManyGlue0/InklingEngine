//! Minimal .npy reader (little-endian f4 / i4, C order) for reference tensor dumps.

use anyhow::{Context, Result, bail, ensure};

pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn read(path: impl AsRef<std::path::Path>) -> Result<Npy> {
    let path = path.as_ref();
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(b.len() > 10 && &b[..6] == b"\x93NUMPY", "{} is not .npy", path.display());
    let (hl, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes(b[8..12].try_into()?) as usize, 12) };
    let header = std::str::from_utf8(&b[off..off + hl])?;
    let descr = header.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).context("npy descr")?;
    let shape_s = header.split("'shape':").nth(1).and_then(|s| s.split('(').nth(1)).and_then(|s| s.split(')').next()).context("npy shape")?;
    let shape: Vec<usize> = shape_s.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
    let raw = &b[off + hl..];
    let n: usize = shape.iter().product();
    ensure!(raw.len() >= n * 4, "{} truncated", path.display());
    let data = match descr {
        "<f4" => raw[..n * 4].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        "<i4" => raw[..n * 4].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        d => bail!("unsupported npy dtype {d}"),
    };
    Ok(Npy { shape, data })
}
