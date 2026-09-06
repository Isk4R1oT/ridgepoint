#!/usr/bin/env bash
# On-pod calibration harness for ridgepoint. Runs INSIDE a RunPod vLLM pod.
# Collects MEASURED numbers only (vLLM memory log + benchmarks); pairing with ridgepoint's
# PREDICTED values happens locally afterwards. Reads MODELS (comma-separated ridgepoint ids)
# from the environment. Writes results + a DONE marker to the mounted volume (/models/results).
#
# Boot gate first (ops rule): verify CUDA + real HBM bandwidth; bail if bandwidth < 85% of spec.
set -euo pipefail

# ROOT = where models + results live: the mounted volume (/models) or ephemeral disk (/workspace).
ROOT="${ROOT:-/models}"
# The harness + parser arrive via `git clone` of this repo, so resolve them next to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS" "$ROOT/hf"
export HF_HOME="${HF_HOME:-$ROOT/hf}"
COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
UTIL=0.90
MAXLEN=8192

log() { echo "[calib $(date -u +%H:%M:%S)] $*"; }

# ridgepoint id -> "HF_repo DTYPE" (mirror of MODELS in ridgepoint_calib.py)
declare -A REPO=(
  [llama-3-8b]="meta-llama/Meta-Llama-3-8B-Instruct float16"
  [deepseek-r1-distill-14b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B float16"
  [llama-3-70b]="casperhansen/llama-3-70b-instruct-awq awq"
  [deepseek-v2-lite]="deepseek-ai/DeepSeek-V2-Lite-Chat float16"
)

boot_gate() {
  log "CUDA + bandwidth boot gate (GPUs=$COUNT)"
  nvidia-smi -L || { log "FATAL: no GPU"; exit 3; }
  python - "$COUNT" <<'PY'
import sys, time, torch
assert torch.cuda.is_available(), "CUDA not available"
n = int(sys.argv[1]); nbytes = 2 << 30  # 2 GiB
x = torch.empty(nbytes // 2, dtype=torch.float16, device="cuda")
y = torch.empty_like(x)
torch.cuda.synchronize(); t = time.time()
it = 30
for _ in range(it):
    y.copy_(x)
torch.cuda.synchronize()
# device-to-device copy touches 2x bytes (read+write) per iteration
gbps = (2 * nbytes * it) / (time.time() - t) / 1e9
name = torch.cuda.get_device_name(0)
print(f"GATE device={name} measured_d2d_bandwidth_gbps={gbps:.0f}")
# Compare against an expected floor if provided (driver sets EXPECTED_BW_GBS).
import os
exp = float(os.environ.get("EXPECTED_BW_GBS", "0"))
if exp and gbps < 0.85 * exp:
    raise SystemExit(f"bandwidth {gbps:.0f} < 85% of expected {exp:.0f} — throttled/mislabeled pod, bailing")
PY
  log "boot gate passed"
}

measure_model() {
  local id="$1"; local spec="${REPO[$id]:-}"
  [ -n "$spec" ] || { log "unknown model id $id, skipping"; return; }
  local repo dtype; read -r repo dtype <<<"$spec"
  log "=== $id ($repo, dtype=$dtype, tp=$COUNT) ==="

  # download once (idempotent — reused across pods via the volume)
  hf download "$repo" --quiet || huggingface-cli download "$repo" || true

  # launch server, capture the memory-profiling log
  local slog="$RESULTS/$id.serve.log"
  vllm serve "$repo" --dtype "$dtype" --gpu-memory-utilization "$UTIL" \
       --tensor-parallel-size "$COUNT" --max-model-len "$MAXLEN" \
       --disable-log-requests > "$slog" 2>&1 &
  local pid=$!

  # wait for readiness (memory profiling emits the KV/blocks lines), 10 min cap
  local ok=0
  for _ in $(seq 1 120); do
    if grep -qiE "Maximum concurrency|GPU blocks|KV cache size" "$slog"; then ok=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then log "server died early for $id"; break; fi
    sleep 5
  done

  # parse memory split
  python "$SCRIPT_DIR/parse_vllm_mem.py" < "$slog" > "$RESULTS/$id.mem.json" || true

  if [ "$ok" = 1 ]; then
    # decode/prefill benchmarks (standardized, reproducible). Args are vLLM-version sensitive.
    vllm bench latency --model "$repo" --input-len 2048 --output-len 128 \
         --batch-size 1 --num-iters 5 --output-json "$RESULTS/$id.latency.json" || true
    vllm bench serving --model "$repo" --dataset-name sharegpt --num-prompts 200 \
         --save-result --result-filename "$RESULTS/$id.serving.json" || true
  fi

  kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
  log "done $id"
}

main() {
  boot_gate
  IFS=',' read -ra IDS <<<"${MODELS:?set MODELS=id1,id2,...}"
  for id in "${IDS[@]}"; do measure_model "$id"; done
  echo "{\"gpus\": $COUNT, \"models\": \"${MODELS}\", \"util\": $UTIL, \"max_len\": $MAXLEN}" > "$RESULTS/meta.json"
  touch "$RESULTS/DONE"
  log "ALL DONE -> $RESULTS (DONE marker written)"
}

main "$@"
