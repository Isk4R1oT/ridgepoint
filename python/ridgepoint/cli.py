"""ridgepoint CLI — `ridgepoint fit|scan|devices`. Accepts built-in ids AND HuggingFace 'org/repo'."""
import argparse
import sys

from . import devices as _devices
from . import fit as _fit
from . import scan as _scan


def _split_gpu(g: str):
    """'a100-80gb:2' -> ('a100-80gb', 2); 'a100-80gb' -> ('a100-80gb', 1)."""
    if ":" in g:
        gid, n = g.rsplit(":", 1)
        return gid, int(n)
    return g, 1


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="ridgepoint", description="Engine-aware LLM inference sizing, calibrated to ~1% vs real vLLM.")
    sub = p.add_subparsers(dest="cmd")
    for name in ("fit", "scan"):
        sp = sub.add_parser(name, help=f"{name} a model onto hardware (id or HuggingFace org/repo)")
        sp.add_argument("model", help="built-in id (e.g. llama-3-70b) OR HuggingFace repo (org/name)")
        sp.add_argument("--gpu", default="a100-80gb:1", help="device id, optionally :N (e.g. h100-80gb:8)")
        sp.add_argument("--engine", default="vllm", choices=["vllm", "llamacpp", "llama.cpp"])
        sp.add_argument("--dtype", default="fp16", help="weight quant: fp16|fp8|q4_k_m|awq|...")
        sp.add_argument("--kv-cache-dtype", dest="kv", default="fp16", choices=["fp16", "bf16", "fp8"])
        sp.add_argument("--prompt", type=int, default=2048)
        sp.add_argument("--json", action="store_true")
        if name == "fit":
            sp.add_argument("--ctx", type=int, default=4096)
    sub.add_parser("devices", help="detect local NVIDIA GPUs")

    a = p.parse_args(argv)
    if a.cmd in ("fit", "scan"):
        gid, count = _split_gpu(a.gpu)
        kw = dict(count=count, engine=a.engine, dtype=a.dtype, kv_cache_dtype=a.kv, prompt=a.prompt, as_json=a.json)
        try:
            if a.cmd == "fit":
                out = _fit(a.model, gid, ctx=a.ctx, **kw)
            else:
                out = _scan(a.model, gid, **kw)
        except Exception as e:  # HF/lookup/validation errors → clean CLI message, no traceback
            print(f"ridgepoint: {e}", file=sys.stderr)
            return 2
        print(out)
        return 0
    if a.cmd == "devices":
        try:
            print(_devices())
        except Exception as e:
            print(f"ridgepoint: {e}", file=sys.stderr)
            return 1
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
