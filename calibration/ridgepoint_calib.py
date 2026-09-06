#!/usr/bin/env python3
"""ridgepoint calibration driver — orchestrates a RunPod calibration over the REST v2 API.

Modes:
  (default) --dry-run : builds every HTTP request body + the ordered plan, validates them,
                        and prints them. ZERO network calls, $0. The API key is never touched.
  --execute           : reads RUNPOD_API_KEY from the environment and drives real pods.

The API key is NEVER hardcoded and NEVER written to disk. Pass it as:
    RUNPOD_API_KEY=... python ridgepoint_calib.py --execute --volume-id <vol> --pubkey "$(cat ~/.ssh/id_ed25519.pub)"

Placement is LOCKED from live availability (get-gpu-type, 2026-09-06): the only DC with BOTH
A100+H100 was US-MO-1 (LOW/LOW), so we decouple — A100 work in US-MD-1 (MEDIUM), H100 transfer
in AP-IN-1 (MEDIUM). The H100 pod carries no shared volume; it downloads its 2 models onto
ephemeral disk (~+$0.3). Every pod git-clones this repo for the harness.
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

VOLUME_DC = "US-MD-1"  # shared volume lives with the A100 work (best A100 stock)

# ridgepoint registry id -> HF repo + vLLM dtype + architecture axis it exercises.
MODELS = {
    "llama-3-8b":              {"hf": "meta-llama/Meta-Llama-3-8B-Instruct",      "dtype": "float16", "arch": "GQA"},
    "deepseek-r1-distill-14b": {"hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "dtype": "float16", "arch": "GQA"},
    "llama-3-70b":             {"hf": "casperhansen/llama-3-70b-instruct-awq",     "dtype": "awq",     "arch": "GQA-big"},
    "deepseek-v2-lite":        {"hf": "deepseek-ai/DeepSeek-V2-Lite-Chat",         "dtype": "float16", "arch": "MLA-MoE"},
}

# (pod name, gpu type id, count, [model ids], data center, uses_shared_volume)
CALIB = [
    ("cal-a100",   "NVIDIA A100-SXM4-80GB", 1, ["llama-3-8b", "deepseek-r1-distill-14b", "llama-3-70b", "deepseek-v2-lite"], "US-MD-1", True),
    ("cal-a100x2", "NVIDIA A100-SXM4-80GB", 2, ["llama-3-8b"],                                                              "US-MD-1", True),
    ("cal-h100",   "NVIDIA H100 80GB HBM3", 1, ["llama-3-8b", "deepseek-r1-distill-14b"],                                   "AP-IN-1", False),
]
VOLUME_SIZE_GB = 150


def volume_body() -> dict:
    return {"name": "ridgepoint-cal", "size": VOLUME_SIZE_GB, "dataCenter": VOLUME_DC}


def pod_body(name: str, gpu_id: str, count: int, dc: str, model_ids: list[str], pubkey: str, vol_id: str | None) -> dict:
    # ROOT = where models + results live: the mounted volume, or ephemeral disk when volume-less.
    root = "/models" if vol_id else "/workspace"
    run = ("bash -lc 'git clone --depth 1 %s /rp && "
           "ROOT=%s MODELS=%s bash /rp/calibration/run_calib.sh'") % (REPO_URL, root, ",".join(model_ids))
    body: dict = {
        "name": name,
        "image": IMAGE,
        "cloud": "SECURE",  # ops rule — never COMMUNITY
        "gpu": {"id": gpu_id, "count": count, "allowedCudaVersions": CUDA},
        "disk": 60 if vol_id else 140,  # volume-less pod holds models on ephemeral disk
        "dataCenterIds": [dc],
        "ports": ["22/tcp"],
        "startSsh": True,
        "env": {"PUBLIC_KEY": pubkey, "HF_HOME": root + "/hf"},
        "args": run,
    }
    if vol_id:
        body["mounts"] = {"network": [{"volumeId": vol_id, "path": "/models"}]}
    return body


def validate(body: dict, kind: str) -> list[str]:
    """Cheap invariants that would otherwise fail only after a pod is billing."""
    errs: list[str] = []
    if kind == "pod":
        if body.get("cloud") != "SECURE":
            errs.append("cloud must be SECURE (ops rule)")
        if not body.get("args"):
            errs.append("args empty — no start command")
        if not body.get("gpu", {}).get("id"):
            errs.append("no gpu id")
        if body.get("dataCenterIds", ["<"])[0].startswith("<"):
            errs.append("dataCenterId unresolved")
        # only pods that declare a mount need a resolved volume id
        net = body.get("mounts", {}).get("network")
        if net and (not net[0].get("volumeId") or net[0]["volumeId"].startswith("<")):
            errs.append("network volume declared but volumeId unresolved")
    elif kind == "vol":
        if not (10 <= body.get("size", 0) <= 4096):
            errs.append("size out of range [10,4096]")
        if body.get("dataCenter", "<").startswith("<"):
            errs.append("dataCenter unresolved")
    return errs


def plan(vol_id: str, pubkey: str) -> list[tuple[str, str, dict, str]]:
    """Ordered (method+path, label, body, kind). Read-only construction — no calls."""
    steps: list[tuple[str, str, dict, str]] = [("POST /v2/network-volumes", "create volume (US-MD-1)", volume_body(), "vol")]
    for name, gid, cnt, mids, dc, uses_vol in CALIB:
        steps.append(("POST /v2/pods", f"create {name} ({gid} ×{cnt} @ {dc}{', vol' if uses_vol else ', ephemeral'})",
                      pod_body(name, gid, cnt, dc, mids, pubkey, vol_id if uses_vol else None), "pod"))
    return steps


def dry_run(vol_id: str, pubkey: str) -> int:
    print("=== DRY RUN — no network, $0. API key untouched. ===\n")
    bad = 0
    for method_path, label, body, kind in plan(vol_id, pubkey):
        errs = validate(body, kind)
        print(f"# {method_path}   [{label}]   -> {'OK' if not errs else 'FAIL: ' + '; '.join(errs)}")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        print()
        bad += len(errs)
    print("--- lifecycle per pod ---")
    print("  GET /v2/pods/{id} -> poll RUNNING · watch logs for DONE · POST /stop the instant DONE")
    print("  pull results via proxy · DELETE /v2/pods/{id} · DELETE volume last · GET /v2/billing verify")
    print(f"\nvalidation: {'ALL OK' if bad == 0 else str(bad) + ' problem(s) — resolve <placeholders> before --execute'}")
    print("placement: A100 → US-MD-1 (MEDIUM) · H100 → AP-IN-1 (MEDIUM). cost ~$11 base / ~$16 realistic / cap $22.")
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


def execute(vol_id: str, pubkey: str) -> int:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: set RUNPOD_API_KEY in the environment (never pass it as an argument).", file=sys.stderr)
        return 2
    # Guarded entry point. The watched loop (create → poll RUNNING → watch DONE → stop → pull →
    # delete → billing) is driven step by step so a human confirms each spend — never blind.
    print(f"execute: vol_id={vol_id} models={list(MODELS)} — drive the watched loop step by step.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ridgepoint RunPod calibration driver")
    ap.add_argument("--dry-run", action="store_true", help="build+print+validate request bodies; no network (default)")
    ap.add_argument("--execute", action="store_true", help="run for real (needs RUNPOD_API_KEY)")
    ap.add_argument("--volume-id", default="<vol_id>", help="existing network volume id (created in US-MD-1)")
    ap.add_argument("--pubkey", default="<SSH_PUBKEY>", help="SSH public key contents for the pod")
    a = ap.parse_args()
    # --dry-run wins over --execute — you can never spend by accident.
    if a.execute and not a.dry_run:
        return execute(a.volume_id, a.pubkey)
    return dry_run(a.volume_id, a.pubkey)


if __name__ == "__main__":
    sys.exit(main())
