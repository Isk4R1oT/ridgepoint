"""ridgepoint — LLM inference sizing that models the engine, not just the weights.

Two layers:
- `ridgepoint._core` — the pure-math Rust engine (calibrated to ~1% vs real vLLM).
- `ridgepoint.hf`   — the HuggingFace input adapter (the one changeable edge).

`fit`/`scan` accept either a built-in id ("llama-3-70b") or ANY HuggingFace repo id
("meta-llama/Meta-Llama-3-8B-Instruct") — the latter is fetched and its dims computed.
"""
from . import _core

__version__ = "0.1.2"


def _shape_args(model: str) -> tuple:
    """Resolve an 'org/repo' HF id to the positional dims _core.*_shape expects."""
    from .hf import fetch_shape
    s = fetch_shape(model)
    return (
        s["id"], s["layers"], s.get("kv_layers", s["layers"]), s["d_model"], s["n_params"], s["active_params"], s["geom"],
        s.get("head_dim", 0), s.get("n_heads", 0), s.get("n_kv_heads", 0), s.get("d_c", 0), s.get("d_rope", 0),
    )


def fit(model, gpu, *, count=1, engine="vllm", dtype="fp16", kv_cache_dtype="fp16", ctx=4096, prompt=2048, as_json=False):
    """Fit a model onto hardware. `model` = built-in id OR HuggingFace 'org/repo'."""
    if "/" in model:
        return _core.fit_shape(*_shape_args(model), gpu, count, engine, dtype, kv_cache_dtype, ctx, prompt, as_json)
    return _core.fit_registry(model, gpu, count, engine, dtype, kv_cache_dtype, ctx, prompt, as_json)


def scan(model, gpu, *, count=1, engine="vllm", dtype="fp16", kv_cache_dtype="fp16", prompt=2048, as_json=False):
    """Sweep context length → capacity + speed frontier. `model` = id OR HF 'org/repo'."""
    if "/" in model:
        return _core.scan_shape(*_shape_args(model), gpu, count, engine, dtype, kv_cache_dtype, prompt, as_json)
    return _core.scan_registry(model, gpu, count, engine, dtype, kv_cache_dtype, prompt, as_json)


def devices():
    """Detect local NVIDIA GPUs (via nvidia-smi)."""
    return _core.detect_devices()


__all__ = ["fit", "scan", "devices", "__version__"]
