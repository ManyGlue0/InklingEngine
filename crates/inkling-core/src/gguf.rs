//! Minimal, zero-copy GGUF (v2/v3) reader: metadata, tensor directory and memory-mapped tensor data.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;

/// ggml tensor types used by Inkling 2 GGUFs (others are rejected at load).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    BF16,
    NVFP4,
    Other(u32),
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            30 => Self::BF16,
            40 => Self::NVFP4,
            x => Self::Other(x),
        }
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> Result<(usize, usize)> {
        Ok(match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q8_0 => (32, 34),  // f16 scale + 32 x i8
            Self::NVFP4 => (64, 36), // 4 x UE4M3 sub-block scales + 32 bytes packed E2M1
            Self::Other(x) => match quant_block(x) {
                Some(b) => b,
                None => bail!("unsupported ggml tensor type {x}"),
            },
        })
    }

    /// ggml type id (as stored in the file).
    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q8_0 => 8,
            Self::BF16 => 30,
            Self::NVFP4 => 40,
            Self::Other(x) => x,
        }
    }

    pub fn name(self) -> String {
        match self {
            Self::F32 => "F32".into(),
            Self::F16 => "F16".into(),
            Self::Q8_0 => "Q8_0".into(),
            Self::BF16 => "BF16".into(),
            Self::NVFP4 => "NVFP4".into(),
            Self::Other(x) => quant_name(x).map_or_else(|| format!("type {x}"), str::to_owned),
        }
    }
}

/// (elements, bytes) per block of the other quantized types the engine decodes natively (ggml-common.h layouts).
fn quant_block(id: u32) -> Option<(usize, usize)> {
    Some(match id {
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K: hmask 32 | qs 64 | scales 12 | d
        12 => (256, 144), // Q4_K: d, dmin | scales 12 | qs 128
        13 => (256, 176), // Q5_K: d, dmin | scales 12 | qh 32 | qs 128
        14 => (256, 210), // Q6_K: ql 128 | qh 64 | scales 16 | d
        16 => (256, 66),  // IQ2_XXS: d | qs u16 x 32
        17 => (256, 74),  // IQ2_XS: d | qs u16 x 32 | scales 8
        18 => (256, 98),  // IQ3_XXS: d | qs 96
        19 => (256, 50),  // IQ1_S: d | qs 32 | qh u16 x 8
        20 => (32, 18),   // IQ4_NL: d | qs 16
        21 => (256, 110), // IQ3_S: d | qs 64 | qh 8 | signs 32 | scales 4
        22 => (256, 82),  // IQ2_S: d | qs 64 | qh 8 | scales 8
        23 => (256, 136), // IQ4_XS: d | scales_h u16 | scales_l 4 | qs 128
        39 => (32, 17),   // MXFP4: e8m0 | qs 16
        42 => (64, 18),   // Q2_0 (Unsloth): d | qs 16
        _ => return None,
    })
}

fn quant_name(id: u32) -> Option<&'static str> {
    Some(match id {
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        39 => "MXFP4",
        42 => "Q2_0",
        _ => return None,
    })
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            Value::Bool(v) => v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self { Some(s) } else { None }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        if let Value::Arr(a) = self { Some(a) } else { None }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// absolute byte offset in the file
    pub offset: usize,
    pub n_bytes: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    mmap: Mmap,
    pub version: u32,
    pub kv: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p + n <= self.b.len(), "GGUF truncated at byte {}", self.p);
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> Result<u16> { Ok(u16::from_le_bytes(self.take(2)?.try_into()?)) }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into()?)) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into()?)) }
    fn str(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.str()?),
            9 => {
                let at = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(at)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            t => bail!("unknown GGUF value type {t}"),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: read-only mapping of a model file we do not modify.
        let mmap = unsafe { Mmap::map(&file)? };
        let mut c = Cursor { b: &mmap, p: 0 };
        ensure!(c.take(4)? == b"GGUF", "{} is not a GGUF file", path.display());
        let version = c.u32()?;
        ensure!(version >= 2, "GGUF v{version} not supported");
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;

        let mut kv = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let k = c.str()?;
            let t = c.u32()?;
            kv.insert(k, c.value(t)?);
        }

        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.str()?;
            let nd = c.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(c.u64()?);
            }
            let ty = GgmlType::from_u32(c.u32()?);
            let rel = c.u64()? as usize;
            tensors.push(TensorInfo { name, dims, ty, offset: rel, n_bytes: 0 });
        }

        let align = kv.get("general.alignment").and_then(Value::as_u64).unwrap_or(32) as usize;
        let data_start = c.p.div_ceil(align) * align;
        let mut index = HashMap::with_capacity(n_tensors);
        for (i, t) in tensors.iter_mut().enumerate() {
            let (be, bb) = t.ty.block()?;
            let n = t.n_elements() as usize;
            ensure!(n % be == 0, "tensor {} has {} elements, not a multiple of block {}", t.name, n, be);
            t.n_bytes = n / be * bb;
            t.offset += data_start;
            ensure!(t.offset + t.n_bytes <= mmap.len(), "tensor {} runs past end of file", t.name);
            index.insert(t.name.clone(), i);
        }
        Ok(Self { mmap, version, kv, tensors, index })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn require(&self, name: &str) -> Result<&TensorInfo> {
        self.tensor(name).with_context(|| format!("missing tensor `{name}`"))
    }

    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        &self.mmap[t.offset..t.offset + t.n_bytes]
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.get(key)
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        self.get(key).and_then(Value::as_u64).with_context(|| format!("missing/invalid metadata `{key}`"))
    }

    pub fn f64(&self, key: &str) -> Result<f64> {
        self.get(key).and_then(Value::as_f64).with_context(|| format!("missing/invalid metadata `{key}`"))
    }

    pub fn str(&self, key: &str) -> Result<&str> {
        self.get(key).and_then(Value::as_str).with_context(|| format!("missing/invalid metadata `{key}`"))
    }
}
