#!/usr/bin/env python3
"""Parse vLLM's startup memory-profiling log into the numbers ridgepoint calibrates against.

vLLM prints, per GPU, the exact split: total × utilization = budget, then
  model weights / non_torch_memory / activation peak / KV cache.
We extract those + the KV cache token capacity, and derive the calibratable
per-GPU `overhead` = non_torch + activation.

Usage:
  vllm serve ... 2>&1 | python parse_vllm_mem.py          # parse from stdin -> JSON
  python parse_vllm_mem.py --selftest                     # offline test on a sample ($0)
"""
from __future__ import annotations

import json
import re
import sys

PATTERNS = {
    "total_gpu_memory_gib": r"total_gpu_memory \(([\d.]+)GiB\)",
    "gpu_memory_utilization": r"gpu_memory_utilization \(([\d.]+)\)",
    "weights_gib": r"model weights take ([\d.]+)GiB",
    "non_torch_gib": r"non_torch_memory takes ([\d.]+)GiB",
    "activation_gib": r"PyTorch activation peak memory takes ([\d.]+)GiB",
    "kv_cache_gib": r"KV [Cc]ache is ([\d.]+)GiB",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "gpu_blocks": r"#?\s*GPU blocks: ([\d,]+)",
    "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
}


def parse(log: str) -> dict:
    out: dict = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, log)
        if m:
            raw = m.group(1).replace(",", "")
            out[key] = float(raw) if "." in raw else int(raw)
    # The one per-GPU term we actually calibrate: overhead = non_torch + activation peak.
    if "non_torch_gib" in out and "activation_gib" in out:
        out["overhead_gib"] = round(out["non_torch_gib"] + out["activation_gib"], 3)
    return out


# Realistic vLLM (v0.6.4+/0.11.x) memory-profiling log fragment — used only by --selftest.
SAMPLE = """
INFO 09-06 07:00:00 worker.py:232] Memory profiling takes 3.21 seconds
INFO 09-06 07:00:00 worker.py:232] the current vLLM instance can use total_gpu_memory (79.15GiB) x gpu_memory_utilization (0.90) = 71.23GiB
INFO 09-06 07:00:00 worker.py:232] model weights take 14.99GiB; non_torch_memory takes 0.65GiB; PyTorch activation peak memory takes 1.20GiB; the rest of the memory reserved for KV Cache is 54.39GiB.
INFO 09-06 07:00:02 gpu_executor.py:120] # GPU blocks: 27853, # CPU blocks: 2048
INFO 09-06 07:00:02 gpu_executor.py:122] GPU KV cache size: 445,648 tokens
INFO 09-06 07:00:02 gpu_executor.py:123] Maximum concurrency for 8192 tokens per request: 54.40x
"""


def _selftest() -> int:
    r = parse(SAMPLE)
    checks = {
        "total_gpu_memory_gib": 79.15,
        "gpu_memory_utilization": 0.90,
        "weights_gib": 14.99,
        "non_torch_gib": 0.65,
        "activation_gib": 1.20,
        "kv_cache_gib": 54.39,
        "gpu_blocks": 27853,
        "kv_cache_tokens": 445648,
        "max_concurrency": 54.40,
        "overhead_gib": 1.85,
    }
    ok = True
    for k, want in checks.items():
        got = r.get(k)
        flag = "OK" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{flag}] {k}: got={got} want={want}")
    print("selftest:", "ALL OK" if ok else "FAILED")
    print(json.dumps(r, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(json.dumps(parse(sys.stdin.read()), indent=2))
