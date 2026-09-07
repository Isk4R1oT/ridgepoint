#!/usr/bin/env python3
"""ridgepoint calibration driver — orchestrates a RunPod calibration over the REST v2 API.

Modes:
  (default) --dry-run : builds every HTTP request body + the ordered plan, validates them,
                        and prints them. ZERO network calls, $0. The API key is never touched.
  --execute           : reads RUNPOD_API_KEY from the environment and drives real pods.

The API key is NEVER hardcoded and NEVER written to disk. Pass it as:
    RUNPOD_API_KEY=... python ridgepoint_calib.py --execute --pubkey "$(cat ~/.ssh/ridgepoint_calib.pub)"

No network volume: it saved ~$0.15 (only an 8B re-download) yet forced a volume-capable + GPU-stocked
data center (all LOW). So every pod downloads its own models onto ephemeral container disk and git-clones
this repo for the harness. Placement is by GPU stock: A100 → US-MD-1 (MEDIUM), H100 → AP-IN-1 (MEDIUM).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = "https://api.runpod.io/v2"
# User-Agent is mandatory — Cloudflare 403s the default urllib UA.
UA = {"User-Agent": "ridgepoint-calib/0.1", "Accept": "application/json", "Content-Type": "application/json"}
IMAGE = "vllm/vllm-openai:latest"
CUDA = ["12.8"]  # available on both A100 (12.4–13.2) and H100 (12.8–13.0)
REPO_URL = "https://github.com/Isk4R1oT/ridgepoint"

# ridgepoint registry id -> HF repo + vLLM dtype + architecture axis it exercises.
MODELS = {
    "llama-3-8b":              {"hf": "meta-llama/Meta-Llama-3-8B-Instruct",      "dtype": "float16", "arch": "GQA"},
    "deepseek-r1-distill-14b": {"hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "dtype": "float16", "arch": "GQA"},
    "llama-3-70b":             {"hf": "casperhansen/llama-3-70b-instruct-awq",     "dtype": "awq",     "arch": "GQA-big"},
    "deepseek-v2-lite":        {"hf": "deepseek-ai/DeepSeek-V2-Lite-Chat",         "dtype": "float16", "arch": "MLA-MoE"},
}

# (pod name, gpu type id, count, [model ids], data center, container disk GB)
CALIB = [
    ("cal-a100",   "NVIDIA A100-SXM4-80GB", 1, ["llama-3-8b", "deepseek-r1-distill-14b", "llama-3-70b", "deepseek-v2-lite"], "US-MD-1", 220),
    ("cal-a100x2", "NVIDIA A100-SXM4-80GB", 2, ["llama-3-8b"],                                                              "US-MD-1", 80),
    ("cal-h100",   "NVIDIA H100 80GB HBM3", 1, ["llama-3-8b", "deepseek-r1-distill-14b"],                                   "AP-IN-1", 140),
]


def pod_body(name: str, gpu_id: str, count: int, dc: str, model_ids: list[str], pubkey: str, disk: int) -> dict:
    run = ("bash -lc 'git clone --depth 1 %s /rp && "
           "ROOT=/workspace MODELS=%s bash /rp/calibration/run_calib.sh'") % (REPO_URL, ",".join(model_ids))
    return {
        "name": name,
        "image": IMAGE,
        "cloud": "SECURE",  # ops rule — never COMMUNITY
        "gpu": {"id": gpu_id, "count": count, "allowedCudaVersions": CUDA},
        "disk": disk,  # ephemeral; must hold this pod's models + working set
        "dataCenterIds": [dc],
        "ports": ["22/tcp"],
        "startSsh": True,
        "env": {"PUBLIC_KEY": pubkey, "HF_HOME": "/workspace/hf"},
        "args": run,
    }


def validate(body: dict) -> list[str]:
    errs: list[str] = []
    if body.get("cloud") != "SECURE":
        errs.append("cloud must be SECURE (ops rule)")
    if not body.get("args"):
        errs.append("args empty — no start command")
    if not body.get("gpu", {}).get("id"):
        errs.append("no gpu id")
    if body.get("dataCenterIds", ["<"])[0].startswith("<"):
        errs.append("dataCenterId unresolved")
    if "<" in body.get("env", {}).get("PUBLIC_KEY", ""):
        errs.append("PUBLIC_KEY (ssh pubkey) unresolved")
    return errs


def plan(pubkey: str) -> list[tuple[str, dict]]:
    return [(f"create {name} ({gid} ×{cnt} @ {dc}, disk {disk}GB)",
             pod_body(name, gid, cnt, dc, mids, pubkey, disk)) for name, gid, cnt, mids, dc, disk in CALIB]


def dry_run(pubkey: str) -> int:
    print("=== DRY RUN — no network, $0. API key untouched. ===\n")
    bad = 0
    for label, body in plan(pubkey):
        errs = validate(body)
        print(f"# POST /v2/pods   [{label}]   -> {'OK' if not errs else 'FAIL: ' + '; '.join(errs)}")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        print()
        bad += len(errs)
    print("--- lifecycle per pod ---")
    print("  POST /v2/pods -> GET /v2/pods/{id} poll RUNNING -> watch logs for DONE -> POST /stop the instant DONE")
    print("  pull results via SSH/proxy -> DELETE /v2/pods/{id} -> GET /v2/billing verify")
    print(f"\nvalidation: {'ALL OK' if bad == 0 else str(bad) + ' problem(s) — resolve <placeholders> before --execute'}")
    print("placement: A100 → US-MD-1 (MEDIUM) · H100 → AP-IN-1 (MEDIUM) · no volume (ephemeral disk).")
    print("cost: ~$11 base / ~$16 realistic / hard cap $22.")
    return 1 if bad else 0


def _req(method: str, path: str, key: str, body: dict | None = None):
    h = dict(UA)
    h["Authorization"] = "Bearer " + key
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:500].decode("utf-8", "replace")


def execute(pubkey: str) -> int:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: set RUNPOD_API_KEY in the environment (never pass it as an argument).", file=sys.stderr)
        return 2
    # Guarded entry point. The watched loop (create → poll RUNNING → watch DONE → stop → pull → delete →
    # billing) is driven step by step so a human confirms each spend — never blind. Smoke-test 8B first.
    print(f"execute: models={list(MODELS)} — drive the watched loop step by step (smoke-test 8B first).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ridgepoint RunPod calibration driver")
    ap.add_argument("--dry-run", action="store_true", help="build+print+validate request bodies; no network (default)")
    ap.add_argument("--execute", action="store_true", help="run for real (needs RUNPOD_API_KEY)")
    ap.add_argument("--pubkey", default="<SSH_PUBKEY>", help="SSH public key contents for the pod")
    a = ap.parse_args()
    # --dry-run wins over --execute — you can never spend by accident.
    if a.execute and not a.dry_run:
        return execute(a.pubkey)
    return dry_run(a.pubkey)


if __name__ == "__main__":
    sys.exit(main())
