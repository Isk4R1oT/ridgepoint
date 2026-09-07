# ridgepoint — hardware calibration (predict vs measured)

Ridgepoint's model was validated against **real vLLM 0.28.0** on RunPod (Secure) — A100-80GB and
H100-80GB — on 2026-09-07. Method: launch vLLM per config, read its own memory-profiling log for the
memory split, and benchmark the live OpenAI endpoint (direct TCP) for decode/prefill speed. The point
of this file: every headline number ridgepoint prints is checked against hardware here.

## Memory — predict vs measured (A100-80GB, bf16, util 0.90, max-len 8192)

| model | ridgepoint KV pool | measured KV pool | error |
|---|---|---|---|
| llama-3-8b (GQA) | 54.5 GiB | **55.11 GiB** | −1.1% |
| deepseek-r1-distill-14b (GQA) | ~41.3 GiB | **41.96 GiB** | −1.6% |
| deepseek-v2-lite (MLA·MoE) | ~40.4 GiB | **40.79 GiB** | −1.0% |
| llama-3-70b-awq (hidden 8192) | ~32 GiB | **31.96 GiB** | ~0% |

**KV-cache formulas validated to the BYTE:**
- GQA (llama-3-8b): 131,072 B/token = `2·n_kv(8)·head_dim(128)·L(32)·2` — exact.
- MLA (ds-v2-lite): 31,104 B/token = `(d_c(512)+d_rope(64))·L(27)·2` — exact. (~4–6× fewer bytes/token than GQA.)

## Calibrated constants (architecture-parameterized, NOT per-model)

- **Usable VRAM** = nominal binary GiB − ~0.75 GiB driver reserve. A100/H100-80GB → **79.25 GiB** (vLLM
  observed), NOT the 80e9-decimal (74.5 GiB) the tool used before. Fixed the ~14% undercount.
- **Overhead** `= activation + count·(non_torch + cudagraph)`, all per-GPU except activation which SHARDS:
  `activation ≈ d_model/4096 GiB` · `cudagraph ≈ 0.55·d_model/4096 GiB` · `non_torch ≈ 0.28 (dense) / 0.70 (MoE)`.
  Fit vs measured: 8b/1GPU 1.83 vs 1.82 · 70b/1GPU 3.38 vs 3.53 · 8b/2GPU 2.66 vs 2.56.
- **Real quant bpw**: AWQ int4 ≈ 4.5 (measured 70.6B → 37.06 GiB), Q4_K_M ≈ 4.8. KV dtype independent of weights.

## Speed — predict vs measured

| GPU | decode ridgepoint | decode measured | MBU |
|---|---|---|---|
| A100-80GB (8b) | 73–79 tok/s | **79.3 tok/s** | 0.62 |
| H100-80GB (8b) | ~124 tok/s | **125.1 tok/s** | 0.60 |

- **MBU ≈ 0.61 and transfers across Ampere↔Hopper** (measured both). Decode scales with bandwidth
  (A100→H100 1.58× ≈ bw ratio 1.64×) — confirms the memory-bound roofline. Calibrated.
- Batching: 8b batch1→batch32 = 79→1894 tok/s (A100), 125→2936 (H100) → ~24× — far left of the ridge point,
  batching helps hugely (consistent with ridge ≈ batch 150 on A100).
- **MFU / TTFT**: could not be isolated remotely (for fast GPUs the request's network RTT dominates the
  prefill time). Left as a literature band (0.25–0.55, `calibrated=false`) → TTFT stays an honest wide
  interval. A clean MFU needs on-pod timing (future work).

## Honesty notes
- All predictions are conservative-to-accurate; the fixed VRAM undercount had made capacity pessimistic.
- Coefficients measured on vLLM 0.28 / util 0.90 / bf16; other engine versions/util will shift overhead.
- Every pod was Secure-cloud and torn down after readout; the CLI/JSON carry per-field `calibrated` flags.
