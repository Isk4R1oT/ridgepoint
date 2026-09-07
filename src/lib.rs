//! ridgepoint — LLM inference sizing that models the engine, not just the weights.
//!
//! `core` is pure Rust math (no Python, no HTTP, no HF — depends only on numeric dimensions).
//! The `python` feature adds a thin PyO3 boundary so the same core ships as `pip install ridgepoint`.
//! Model-agnostic input (HuggingFace fetch) lives in the Python package (edge adapter), which builds
//! a shape from HF metadata and calls `fit_shape`/`scan_shape` here. The Rust core never touches HF.

pub mod core;
pub mod detect;
pub mod registry;
pub mod render;

#[cfg(feature = "python")]
mod python {
    use crate::core::*;
    use crate::{detect, registry, render};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    fn kv_bytes_of(s: &str) -> PyResult<u8> {
        match s {
            "fp16" | "bf16" => Ok(2),
            "fp8" => Ok(1),
            o => Err(PyValueError::new_err(format!("unknown kv_cache_dtype: {o}"))),
        }
    }

    fn alloc_of(engine: &str) -> PyResult<Box<dyn Allocator>> {
        match engine {
            "vllm" => Ok(Box::new(Vllm { util: 0.90 })),
            "llamacpp" | "llama.cpp" => Ok(Box::new(LlamaCpp)),
            o => Err(PyValueError::new_err(format!("unknown engine: {o}"))),
        }
    }

    fn state_of(geom: &str, head_dim: u32, n_heads: u32, n_kv_heads: u32, d_c: u32, d_rope: u32) -> PyResult<StateGeometry> {
        match geom {
            "mha" => Ok(StateGeometry::Mha { head_dim, n_heads }),
            "gqa" => Ok(StateGeometry::Gqa { head_dim, n_kv_heads }),
            "mla" => Ok(StateGeometry::Mla { d_c, d_rope }),
            o => Err(PyValueError::new_err(format!("unknown geometry: {o} (expected mha|gqa|mla)"))),
        }
    }

    fn kv_kind_of(geom: &str, moe: bool) -> &'static str {
        match (geom, moe) {
            ("mla", true) => "MLA·MoE",
            ("mla", false) => "MLA",
            ("gqa", true) => "GQA·MoE",
            ("gqa", false) => "GQA",
            _ => "MHA",
        }
    }

    fn run_fit(m: ModelShape, gpu: &str, count: u32, engine: &str, dtype: &str, kv_cache_dtype: &str, ctx: u32, prompt: u32, as_json: bool) -> PyResult<String> {
        let dev = registry::device(gpu).ok_or_else(|| PyValueError::new_err(format!("unknown gpu: {gpu}")))?;
        let q = registry::quant(dtype).ok_or_else(|| PyValueError::new_err(format!("unknown dtype: {dtype}")))?;
        let e = alloc_of(engine)?;
        let kv = kv_bytes_of(kv_cache_dtype)?;
        let hw = DeviceSet { device: dev, count };
        let w = Workload { ctx, prompt_tokens: prompt, concurrency: None, kv_bytes: kv };
        let r = fit(&m, &hw, q.as_ref(), e.as_ref(), &w, &Calibration::measured());
        Ok(if as_json { render::json(&r) } else { render::human(&r) })
    }

    fn run_scan(m: ModelShape, gpu: &str, count: u32, engine: &str, dtype: &str, kv_cache_dtype: &str, prompt: u32, as_json: bool) -> PyResult<String> {
        let dev = registry::device(gpu).ok_or_else(|| PyValueError::new_err(format!("unknown gpu: {gpu}")))?;
        let q = registry::quant(dtype).ok_or_else(|| PyValueError::new_err(format!("unknown dtype: {dtype}")))?;
        let e = alloc_of(engine)?;
        let kv = kv_bytes_of(kv_cache_dtype)?;
        let hw = DeviceSet { device: dev, count };
        let ctxs = [512u32, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
        let r = scan(&m, &hw, q.as_ref(), e.as_ref(), prompt, kv, &ctxs, &Calibration::measured());
        Ok(if as_json { render::scan_json(&r) } else { render::scan_human(&r) })
    }

    /// fit from a built-in registry id.
    #[pyfunction]
    #[pyo3(signature = (model, gpu, count = 1, engine = "vllm", dtype = "fp16", kv_cache_dtype = "fp16", ctx = 4096, prompt = 2048, as_json = false))]
    #[allow(clippy::too_many_arguments)]
    fn fit_registry(model: &str, gpu: &str, count: u32, engine: &str, dtype: &str, kv_cache_dtype: &str, ctx: u32, prompt: u32, as_json: bool) -> PyResult<String> {
        let m = registry::model(model).ok_or_else(|| PyValueError::new_err(format!("unknown model: {model}")))?;
        run_fit(m, gpu, count, engine, dtype, kv_cache_dtype, ctx, prompt, as_json)
    }

    /// fit from explicit dims — the HF-fetch path (Python adapter builds these from HF metadata).
    #[pyfunction]
    #[pyo3(signature = (id, layers, d_model, n_params, active_params, geom, head_dim = 0, n_heads = 0, n_kv_heads = 0, d_c = 0, d_rope = 0, gpu = "a100-80gb", count = 1, engine = "vllm", dtype = "fp16", kv_cache_dtype = "fp16", ctx = 4096, prompt = 2048, as_json = false))]
    #[allow(clippy::too_many_arguments)]
    fn fit_shape(id: &str, layers: u32, d_model: u32, n_params: u64, active_params: u64, geom: &str, head_dim: u32, n_heads: u32, n_kv_heads: u32, d_c: u32, d_rope: u32, gpu: &str, count: u32, engine: &str, dtype: &str, kv_cache_dtype: &str, ctx: u32, prompt: u32, as_json: bool) -> PyResult<String> {
        let state = state_of(geom, head_dim, n_heads, n_kv_heads, d_c, d_rope)?;
        let m = ModelShape { id: id.to_string(), layers, d_model, n_params, active_params, state, kv_kind: kv_kind_of(geom, active_params != n_params) };
        run_fit(m, gpu, count, engine, dtype, kv_cache_dtype, ctx, prompt, as_json)
    }

    /// scan (context sweep) from a registry id.
    #[pyfunction]
    #[pyo3(signature = (model, gpu, count = 1, engine = "vllm", dtype = "fp16", kv_cache_dtype = "fp16", prompt = 2048, as_json = false))]
    #[allow(clippy::too_many_arguments)]
    fn scan_registry(model: &str, gpu: &str, count: u32, engine: &str, dtype: &str, kv_cache_dtype: &str, prompt: u32, as_json: bool) -> PyResult<String> {
        let m = registry::model(model).ok_or_else(|| PyValueError::new_err(format!("unknown model: {model}")))?;
        run_scan(m, gpu, count, engine, dtype, kv_cache_dtype, prompt, as_json)
    }

    /// scan from explicit dims — the HF-fetch path.
    #[pyfunction]
    #[pyo3(signature = (id, layers, d_model, n_params, active_params, geom, head_dim = 0, n_heads = 0, n_kv_heads = 0, d_c = 0, d_rope = 0, gpu = "a100-80gb", count = 1, engine = "vllm", dtype = "fp16", kv_cache_dtype = "fp16", prompt = 2048, as_json = false))]
    #[allow(clippy::too_many_arguments)]
    fn scan_shape(id: &str, layers: u32, d_model: u32, n_params: u64, active_params: u64, geom: &str, head_dim: u32, n_heads: u32, n_kv_heads: u32, d_c: u32, d_rope: u32, gpu: &str, count: u32, engine: &str, dtype: &str, kv_cache_dtype: &str, prompt: u32, as_json: bool) -> PyResult<String> {
        let state = state_of(geom, head_dim, n_heads, n_kv_heads, d_c, d_rope)?;
        let m = ModelShape { id: id.to_string(), layers, d_model, n_params, active_params, state, kv_kind: kv_kind_of(geom, active_params != n_params) };
        run_scan(m, gpu, count, engine, dtype, kv_cache_dtype, prompt, as_json)
    }

    /// Local NVIDIA GPU detection (for `ridgepoint devices`). Returns a human report or an error.
    #[pyfunction]
    fn detect_devices() -> PyResult<String> {
        match detect::detect_gpus() {
            Ok(gpus) => {
                let mut s = format!("detected {} GPU(s):\n", gpus.len());
                for g in &gpus {
                    let d = detect::match_device(&g.name, g.vram_bytes);
                    s.push_str(&format!("  [{}] {}  ({:.0} GB)  → {} · {} · {}\n", g.index, g.name, g.vram_bytes as f64 / 1e9, d.name, d.arch, d.interconnect));
                }
                s.push_str("\nuse:  --gpu auto   (all)   ·   --gpu auto:N   (N of them)");
                Ok(s)
            }
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }

    #[pymodule]
    fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(fit_registry, m)?)?;
        m.add_function(wrap_pyfunction!(fit_shape, m)?)?;
        m.add_function(wrap_pyfunction!(scan_registry, m)?)?;
        m.add_function(wrap_pyfunction!(scan_shape, m)?)?;
        m.add_function(wrap_pyfunction!(detect_devices, m)?)?;
        Ok(())
    }
}
