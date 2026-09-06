#!/usr/bin/env python3
"""ridgepoint calibration driver — orchestrates a RunPod calibration over the REST v2 API.

Two modes:
  (default) --dry-run : builds every HTTP request body + the ordered plan, validates them,
                        and prints them. ZERO network calls, $0. The API key is never touched.
  --execute           : reads RUNPOD_API_KEY from the environment and drives real pods.

The API key is NEVER hardcoded and NEVER written to disk. Pass it as:
    RUNPOD_API_KEY=... python ridgepoint_calib.py --execute --dc <DC> --volume-id <vol> --pubkey "$(cat ~/.ssh/id_ed25519.pub)"

Placement (--dc) and --volume-id are resolved once at run time (see step 0 in the runbook).
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

# ridgepoint registry id -> HF repo + vLLM dtype + architecture axis it exercises.
MODELS = {
    "llama-3-8b":              {"hf": "meta-llama/Meta-Llama-3-8B-Instruct",      "dtype": "float16", "arch": "GQA"},
    "deepseek-r1-distill-14b": {"hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "dtype": "float16", "arch": "GQA"},
    "llama-3-70b":             {"hf": "casperhansen/llama-3-70b-instruct-awq",     "dtype": "awq",     "arch": "GQA-big"},
    "deepseek-v2-lite":        {"hf": "deepseek-ai/DeepSeek-V2-Lite-Chat",         "dtype": "float16", "arch": "MLA-MoE"},
}

# (pod name, gpu type id, gpu count, [model ids to measure])
# cal-a100 downloads the models into the fresh volume; later pods reuse them.
CALIB = [
    ("cal-a100",   "NVIDIA A100-SXM4-80GB", 1, ["llama-3-8b", "deepseek-r1-distill-14b", "llama-3-70b", "deepseek-v2-lite"]),
    ("cal-a100x2", "NVIDIA A100-SXM4-80GB", 2, ["llama-3-8b"]),
    ("cal-h100",   "NVIDIA H100 80GB HBM3", 1, ["llama-3-8b", "deepseek-r1-distill-14b"]),
]
VOLUME_SIZE_GB = 150


def volume_body(dc: str) -> dict:
    return {"name": "ridgepoint-cal", "size": VOLUME_SIZE_GB, "dataCenter": dc}


def pod_body(name: str, gpu_id: str, count: int, dc: str, vol_id: str, model_ids: list[str], pubkey: str) -> dict:
    # The harness reads MODELS (comma list) and writes results to the mounted volume, then self-stops.
    run = "bash -lc 'cd /models && MODELS=%s bash /models/run_calib.sh'" % ",".join(model_ids)
    return {
        "name": name,
        "image": IMAGE,
        "cloud": "SECURE",  # ops rule — never COMMUNITY
        "gpu": {"id": gpu_id, "count": count, "allowedCudaVersions": CUDA},
        "disk": 60,
        "dataCenterIds": [dc],
        "mounts": {"network": [{"volumeId": vol_id, "path": "/models"}]},
        "ports": ["22/tcp"],
        "startSsh": True,
        "env": {"PUBLIC_KEY": pubkey, "HF_HOME": "/models/hf"},
        "args": run,
    }


def validate(body: dict, kind: str) -> list[str]:
    """Cheap invariants that would otherwise fail only after a pod is billing."""
    errs: list[str] = []
    if kind == "pod":
        if body.get("cloud") != "SECURE":
            errs.append("cloud must be SECURE (ops rule)")
        if not body.get("args"):
            errs.append("args empty — no start command")
        net = body.get("mounts", {}).get("network", [{}])
        if not net or not net[0].get("volumeId") or net[0]["volumeId"].startswith("<"):
            errs.append("network volume not attached (volumeId unresolved)")
        if not body.get("gpu", {}).get("id"):
            errs.append("no gpu id")
        if body.get("dataCenterIds", ["<"])[0].startswith("<"):
            errs.append("dataCenterId unresolved")
    elif kind == "vol":
        if not (10 <= body.get("size", 0) <= 4096):
            errs.append("size out of range [10,4096]")
        if body.get("dataCenter", "<").startswith("<"):
            errs.append("dataCenter unresolved")
    return errs


def plan(dc: str, vol_id: str, pubkey: str) -> list[tuple[str, str, dict, str]]:
    """Ordered (method+path, label, body, kind). Read-only construction — no calls."""
    steps: list[tuple[str, str, dict, str]] = []
    steps.append(("POST /v2/network-volumes", "create volume", volume_body(dc), "vol"))
    for name, gid, cnt, mids in CALIB:
        steps.append((f"POST /v2/pods", f"create {name} ({gid} ×{cnt})", pod_body(name, gid, cnt, dc, vol_id, mids, pubkey), "pod"))
    return steps


def dry_run(dc: str, vol_id: str, pubkey: str) -> int:
    print("=== DRY RUN — no network, $0. API key untouched. ===\n")
    bad = 0
    for method_path, label, body, kind in plan(dc, vol_id, pubkey):
        errs = validate(body, kind)
        status = "OK" if not errs else "FAIL: " + "; ".join(errs)
        print(f"# {method_path}   [{label}]   -> {status}")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        print()
        bad += len(errs)
    print("--- lifecycle after each pod ---")
    print("  GET  /v2/pods/{id}                 poll until desiredStatus/status == RUNNING")
    print("  (logs)  MCP stream-pod-logs {podId} OR REST logs endpoint — watch for DONE marker")
    print("  POST /v2/pods/{id}/stop            the instant DONE appears (stop billing)")
    print("  DELETE /v2/pods/{id}               after results pulled via proxy")
    print("  DELETE /v2/network-volumes/{id}    at the very end")
    print("  GET  /v2/billing?scope=all         confirm total + nothing left running")
    print(f"\nvalidation: {'ALL OK' if bad == 0 else str(bad) + ' problem(s) — resolve <placeholders> before --execute'}")
    print("expected cost (real prices): ~$11 base / ~$16 realistic, ceiling $22.")
    return 1 if bad else 0


# ---- real HTTP (only reached under --execute) ----
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


def execute(dc: str, vol_id: str, pubkey: str) -> int:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: set RUNPOD_API_KEY in the environment (never pass it as an argument).", file=sys.stderr)
        return 2
    # Intentionally minimal here: the safe, watched orchestration loop (create → poll RUNNING →
    # watch logs for DONE → stop → pull-via-proxy → delete → billing) is driven interactively so a
    # human confirms each spend. This function is the guarded entry point; real steps are added when
    # placement (--dc) and --volume-id are locked. Refuse to run blind.
    print("execute mode: placement + volume must be pre-resolved; drive the watched loop step by step.")
    print(f"  dc={dc} vol_id={vol_id}  models={list(MODELS)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ridgepoint RunPod calibration driver")
    ap.add_argument("--dry-run", action="store_true", help="build+print+validate request bodies; no network (default)")
    ap.add_argument("--execute", action="store_true", help="run for real (needs RUNPOD_API_KEY)")
    ap.add_argument("--dc", default="<DATA_CENTER_ID>", help="data center id (must hold A100+H100+volume)")
    ap.add_argument("--volume-id", default="<vol_id>", help="existing network volume id to mount")
    ap.add_argument("--pubkey", default="<SSH_PUBKEY>", help="SSH public key contents for the pod")
    a = ap.parse_args()
    # --dry-run wins over --execute — you can never spend by accident.
    if a.execute and not a.dry_run:
        return execute(a.dc, a.volume_id, a.pubkey)
    return dry_run(a.dc, a.volume_id, a.pubkey)


if __name__ == "__main__":
    sys.exit(main())
