#!/usr/bin/env python3
"""On-pod speed probe. Runs INSIDE a pod, hits the local vLLM OpenAI API (localhost:8000).
Prints `SPEEDRESULT ...` lines to stdout (→ container log, read externally). Measures:
  - decode tok/s at batch 1 (memory-bound → MBU signal),
  - TTFT at a ~2k-token prompt (compute-bound → MFU signal),
  - aggregate throughput at batch 8 / 32 (concurrency → ridge-point signal).
No proxy, no SSH — everything is localhost on the pod.
"""
import json, sys, threading, time, urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "NousResearch/Meta-Llama-3-8B-Instruct"
BASE = "http://localhost:8000"


def post(payload, timeout=240):
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def wait_health(secs=900):
    for _ in range(secs // 5):
        try:
            urllib.request.urlopen(BASE + "/health", timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def main():
    if not wait_health():
        print("SPEEDRESULT ERROR server_never_ready")
        return
    try:
        post({"model": MODEL, "prompt": "hi", "max_tokens": 4, "temperature": 0}).read()
    except Exception as e:
        print("SPEEDRESULT ERROR warmup", str(e)[:100]); return

    # decode tok/s @ batch 1 (short prompt, force full generation)
    t = time.time()
    d = json.load(post({"model": MODEL, "prompt": "Count from one to two hundred:",
                        "max_tokens": 256, "temperature": 0, "ignore_eos": True}))
    dec_el = time.time() - t
    dec_out = d["usage"]["completion_tokens"]
    print(f"SPEEDRESULT decode_batch1_tps={dec_out/dec_el:.1f} tokens={dec_out} el={dec_el:.2f}")

    # TTFT @ ~2k-token prompt (streaming, time to first token)
    longp = "word " * 2048
    t = time.time(); ttft = -1.0
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps({"model": MODEL, "prompt": longp, "max_tokens": 8,
                                                  "temperature": 0, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    for raw in urllib.request.urlopen(req, timeout=240):
        if b'"text"' in raw:
            ttft = time.time() - t; break
    print(f"SPEEDRESULT ttft_2k_s={ttft:.3f}")

    # aggregate throughput at batch 8 and 32 (concurrency)
    for B in (8, 32):
        res = [0] * B
        def w(i):
            try:
                res[i] = json.load(post({"model": MODEL, "prompt": "Write a long detailed story:",
                                         "max_tokens": 128, "temperature": 0, "ignore_eos": True}))["usage"]["completion_tokens"]
            except Exception:
                res[i] = 0
        ths = [threading.Thread(target=w, args=(i,)) for i in range(B)]
        t = time.time(); [x.start() for x in ths]; [x.join() for x in ths]; el = time.time() - t
        print(f"SPEEDRESULT batch={B} agg_tok={sum(res)} el={el:.2f} agg_tps={sum(res)/el:.1f}")
    print("SPEEDRESULT DONE")


if __name__ == "__main__":
    main()
