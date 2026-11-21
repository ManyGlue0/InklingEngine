//! Compiles every `kernels/*.cu` into one fatbin with nvcc and embeds it in the crate.
//!
//! Target architectures come from `INKLING_CUDA_ARCHS` (comma-separated compute capabilities, e.g. "86" or
//! "75,80,86,89,90,100,120"). The default `native` builds only for the GPUs in this machine (fast dev builds).
//! Release builds should set the full list; PTX for the highest arch is always added for forward compatibility.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn nvcc_path() -> PathBuf {
    if let Ok(p) = env::var("NVCC") {
        return PathBuf::from(p);
    }
    if let Ok(cuda) = env::var("CUDA_PATH") {
        let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
        let p = Path::new(&cuda).join("bin").join(exe);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("nvcc")
}

fn main() {
    let kernel_dir = Path::new("kernels");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=INKLING_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=NVCC");

    let mut sources: Vec<PathBuf> = std::fs::read_dir(kernel_dir)
        .expect("kernels/ directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "cu"))
        .collect();
    sources.sort();

    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("inkling_kernels.fatbin");
    let archs = env::var("INKLING_CUDA_ARCHS").unwrap_or_else(|_| "native".into());

    let mut cmd = Command::new(nvcc_path());
    cmd.arg("--fatbin")
        .args(["-O3", "-std=c++17", "--use_fast_math", "-lineinfo"])
        .args(["-Xptxas", "-v"].iter().filter(|_| env::var("INKLING_PTXAS_VERBOSE").is_ok()))
        .arg("-I")
        .arg(kernel_dir)
        .arg("-o")
        .arg(&out);
    if archs == "native" {
        cmd.arg("-arch=native");
    } else {
        let list: Vec<&str> = archs.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        for a in &list {
            cmd.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
        }
        let max = list.iter().max_by_key(|a| a.parse::<u32>().unwrap_or(0)).expect("INKLING_CUDA_ARCHS empty");
        cmd.arg(format!("-gencode=arch=compute_{max},code=compute_{max}"));
    }
    // nvcc --fatbin takes a single translation unit: compile one umbrella file that includes every kernel source.
    let umbrella = PathBuf::from(env::var("OUT_DIR").unwrap()).join("all_kernels.cu");
    let body: String = sources
        .iter()
        .map(|s| format!("#include \"{}\"\n", std::fs::canonicalize(s).unwrap().display().to_string().replace('\\', "/").trim_start_matches("//?/")))
        .collect();
    std::fs::write(&umbrella, body).unwrap();
    cmd.arg(&umbrella);

    // Windows: nvcc needs MSVC's cl.exe. Locate it (and its INCLUDE/LIB env) so plain shells work too.
    if cfg!(windows) && env::var("NVCC_CCBIN").is_err() {
        let target = env::var("TARGET").unwrap();
        if let Some(cl) = cc::windows_registry::find_tool(&target, "cl.exe") {
            cmd.arg("-ccbin").arg(cl.path().parent().unwrap());
            for (k, v) in cl.get_envs() {
                cmd.env(k, v);
            }
        }
    } else if let Ok(ccbin) = env::var("NVCC_CCBIN") {
        cmd.arg("-ccbin").arg(ccbin);
    }

    let status = cmd.status().unwrap_or_else(|e| panic!("failed to run nvcc ({e}); set CUDA_PATH or NVCC"));
    assert!(status.success(), "nvcc failed compiling kernels");
}
