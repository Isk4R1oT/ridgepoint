# ridgepoint

**LLM inference sizing that models the engine, not just the weights — calibrated to ~1% against real vLLM.**

Every "will it fit?" calculator computes `weights + KV < VRAM`. That is the wrong model for paged serving
engines: vLLM pre-reserves `gpu_memory_utilization × VRAM` at startup and pages KV into that pool. ridgepoint
models what actually decides the answer — and its numbers are checked against live vLLM on A100/H100
(see [`calibration/CALIBRATION.md`](calibration/CALIBRATION.md)), matching measured memory to ~1% and the
KV-cache formulas **to the byte** for both GQA and MLA.

The name is the **ridge point** of the roofline model: the arithmetic intensity where decode stops being
memory-bandwidth-bound. It's the number this tool reasons about.

```
$ ridgepoint fit meta-llama/Meta-Llama-3-8B-Instruct --gpu a100-80gb:1 --ctx 8192

  ✓  SERVES        ~54 seqs @ 8192 ctx (best)
  MEMORY                low     best    high
    weights  fp16     15.0 GiB  exact
    overhead           1.5 ·   1.8 ·   2.2 GiB
    KV pool           54.2 ·  54.5 ·  54.9 GiB   ←      (vLLM measured: 55.1 GiB)
  SPEED  roofline
    decode/req          65 ·   73 ·   81 tok/s  MBU calibrated
  CAPACITY @8192       54 seqs · 1.0 GiB/seq
```

## Why it's different

- **Engine-aware capacity** — the usable KV pool *after* the engine's pre-grab → real max concurrent tokens.
  No other calculator models the allocator.
- **KV cache correct to the byte** — GQA `2·n_kv·head_dim·L·kv_bytes` and MLA `(d_c+d_rope)·L·kv_bytes`
  (DeepSeek-style MLA holds ~4–6× fewer bytes/token than a GQA formula would predict).
- **Calibrated, not guessed** — usable VRAM in real GiB (not the marketing decimal), an overhead model fit on
  hardware (activation shards under tensor-parallel; per-GPU context/graphs scale with count), real quant bpw
  (AWQ ≈ 4.5), measured decode MBU (≈ 0.61, transfers A100↔H100).
- **Intervals, not points, with honesty** — deterministic terms are exact; measured terms carry a band; every
  field says whether it's `calibrated`. TTFT/MFU is honestly a wide literature band (not yet cleanly measured).
- **Model-agnostic** — pass any HuggingFace repo id; ridgepoint fetches its config and computes its guts
  (GQA / MLA / MoE detected automatically).

## Install

```
pip install ridgepoint
```

## Usage

```
ridgepoint fit  <model> [--gpu id:N] [--engine vllm|llamacpp] [--dtype fp16|fp8|q4_k_m|awq|...] \
                        [--kv-cache-dtype fp16|fp8] [--ctx N] [--prompt N] [--json]
ridgepoint scan <model> [same flags; sweeps context → capacity + speed frontier]
ridgepoint devices        # detect local NVIDIA GPUs

# <model> is a built-in id (llama-3-70b, deepseek-r1, mixtral-8x7b, ...) OR any HuggingFace repo:
ridgepoint fit Qwen/Qwen2.5-7B --gpu a100-80gb:1
ridgepoint fit deepseek-ai/DeepSeek-V2-Lite --gpu a100-80gb:1     # MLA·MoE, auto-detected
```

As a library:

```python
import ridgepoint
print(ridgepoint.fit("llama-3-70b", "a100-80gb", count=2, dtype="fp16"))
print(ridgepoint.scan("mistralai/Mistral-7B-v0.3", "h100-80gb", as_json=True))
```

## Architecture

A pure-math Rust core (`ridgepoint._core`) — depends only on numeric dimensions, never on HTTP/HF, so it
stays stable as the ecosystem churns — plus a thin HuggingFace input adapter in Python (`ridgepoint.hf`) that
turns any repo into dimensions the core sizes. The same core also ships as a Rust crate + CLI.

## Status & honesty

- Memory / capacity / decode: **calibrated** on vLLM 0.28 (A100 + H100). Predict-vs-measured in
  [`calibration/CALIBRATION.md`](calibration/CALIBRATION.md).
- TTFT / MFU: literature band (remote benchmarking can't isolate prefill from network) — flagged, not hidden.
- `n_params`/`active_params` for HF models are estimated from metadata (documented; feed interval outputs).
- Engines modeled: vLLM (pre-grab) and llama.cpp (incremental). More to come.

## License

MIT
