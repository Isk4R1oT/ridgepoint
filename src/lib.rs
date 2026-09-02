//! ridgepoint — LLM inference sizing that models the engine, not just the weights.
//!
//! `core` is pure Rust (no Python dependency). The `python` feature adds a thin
//! PyO3 wrapper so the same core ships as both `cargo add ridgepoint` and
//! `pip install ridgepoint`. The CLI binary (`src/bin/ridgepoint.rs`) uses `core`
//! directly and never pulls in PyO3.

pub mod core;
pub mod detect;
pub mod registry;
pub mod render;

#[cfg(feature = "python")]
mod python {
    use crate::core::*;
    use crate::{registry, render};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// Compute a fit report and return it as a JSON string (v0 Python surface).
    #[pyfunction]
    #[pyo3(signature = (model, gpu, count = 1, engine = "vllm", dtype = "fp16", kv_cache_dtype = "fp16", ctx = 4096, prompt = 2048))]
    fn fit_json(
        model: &str,
        gpu: &str,
        count: u32,
        engine: &str,
        dtype: &str,
        kv_cache_dtype: &str,
        ctx: u32,
        prompt: u32,
    ) -> PyResult<String> {
        let m = registry::model(model).ok_or_else(|| PyValueError::new_err(format!("unknown model: {model}")))?;
        let dev = registry::device(gpu).ok_or_else(|| PyValueError::new_err(format!("unknown gpu: {gpu}")))?;
        let q = registry::quant(dtype).ok_or_else(|| PyValueError::new_err(format!("unknown dtype: {dtype}")))?;
        let kv_bytes: u8 = match kv_cache_dtype {
            "fp16" | "bf16" => 2,
            "fp8" => 1,
            other => return Err(PyValueError::new_err(format!("unknown kv_cache_dtype: {other}"))),
        };
        let hw = DeviceSet { device: dev, count };
        let alloc: Box<dyn Allocator> = match engine {
            "vllm" => Box::new(Vllm { util: 0.90 }),
            "llamacpp" | "llama.cpp" => Box::new(LlamaCpp),
            other => return Err(PyValueError::new_err(format!("unknown engine: {other}"))),
        };
        let w = Workload { ctx, prompt_tokens: prompt, concurrency: None, kv_bytes };
        let report = fit(&m, &hw, q.as_ref(), alloc.as_ref(), &w, &Calibration::uncalibrated());
        Ok(render::json(&report))
    }

    #[pymodule]
    fn ridgepoint(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(fit_json, m)?)?;
        Ok(())
    }
}
