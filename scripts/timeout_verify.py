#!/usr/bin/env python3
"""Verify the request-timeout / cancellation policy.

A. a request that runs longer than the old 300 s cap, through the router (port 8010)
B. a streaming client that disconnects mid-prefill: does the engine stop?
C. what a streaming client actually receives during a long prefill (keepalives?)

Usage: timeout_verify.py [--tests A,B,C] [--tag timeout]
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
ROUTER = "http://127.0.0.1:8010"
ENGINE = "http://127.0.0.1:8800"
MODEL = "deepseek-v4-flash"
WORDS = ("threshold kernel stream lattice vector gradient sparse routine buffer "
         "archive compile replica dataset aperture quantum segment ledger").split()


def filler(tokens: int, needle: str | None = None) -> str:
    body = (" ".join(WORDS) + " ") * max(1, int(tokens / len(WORDS)))
    if needle:
        cut = len(body) // 2
        body = body[:cut] + f"\nRECORD: the vault passphrase is {needle}.\n" + body[cut:]
    return body


def short_probe(timeout=600):
    """Latency of a trivial request: reveals whether the engine is still busy."""
    payload = json.dumps({"messages": [{"role": "user", "content": "Say OK."}],
                          "max_tokens": 8, "temperature": 0.0}).encode()
    req = urllib.request.Request(ENGINE + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            json.loads(r.read().decode())
        return {"ok": True, "latency_s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "latency_s": round(time.time() - t0, 1), "error": str(e)[:120]}


def test_a(report, args):
    """One request through the router that must outlive the old 300 s cap."""
    prompt = filler(args.tokens) + "\nSummarise the notes above in one sentence."
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 64,
               "temperature": 0.0}
    t0 = time.time()
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0)) as c:
            r = c.post(ROUTER + "/v1/chat/completions", json=payload)
        elapsed = time.time() - t0
        d = r.json()
        t = d.get("timings", {})
        step = {"test": "A_router_request_over_300s", "status_code": r.status_code,
                "elapsed_s": round(elapsed, 1), "exceeded_old_cap": elapsed > 300.0,
                "prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
                "prefill_s": round(t.get("prompt_ms", 0) / 1000.0, 1)}
    except Exception as e:  # noqa: BLE001
        step = {"test": "A_router_request_over_300s", "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 1), "exceeded_old_cap": False}
    report["steps"].append(step)
    print(json.dumps(step), flush=True)


def test_bc(report, args):
    """Stream a long prompt, watch what arrives, then hang up mid-prefill."""
    needle = "ZEBRA-CANYON-CANCEL-7"
    prompt = filler(args.cancel_tokens, needle)
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 64,
               "temperature": 0.0, "stream": True}
    events = []
    t0 = time.time()
    aborted_at = None
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0)) as c:
            with c.stream("POST", ROUTER + "/v1/chat/completions", json=payload) as r:
                for line in r.iter_lines():
                    if line:
                        events.append({"t": round(time.time() - t0, 1), "line": line[:80]})
                    if time.time() - t0 > args.abort_after:
                        aborted_at = round(time.time() - t0, 1)
                        break  # leave the context: closes the connection mid-prefill
    except Exception as e:  # noqa: BLE001
        events.append({"t": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"})
    gaps = [round(events[i + 1]["t"] - events[i]["t"], 1) for i in range(len(events) - 1)]
    report["steps"].append({
        "test": "C_stream_events_during_prefill", "events": len(events),
        "first_events": events[:4], "max_gap_between_events_s": max(gaps) if gaps else None,
    })
    engine = short_probe()
    report["steps"].append({
        "test": "B_cancel_then_engine_responsiveness", "aborted_at_s": aborted_at,
        "events_before_abort": len(events), "post_cancel_short_request": engine,
        "interpretation": ("engine free (cancel propagated)" if engine.get("latency_s", 999) < 60
                           else "engine still busy: cancel did NOT stop generation"),
    })
    print(json.dumps(report["steps"][-2]), flush=True)
    print(json.dumps(report["steps"][-1]), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="A,B,C")
    ap.add_argument("--tokens", type=int, default=64000, help="test A prompt size")
    ap.add_argument("--cancel-tokens", type=int, default=32000, help="test B/C prompt size")
    ap.add_argument("--abort-after", type=float, default=20.0)
    ap.add_argument("--tag", default="timeout")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = ROOT / "benchmarks" / f"timeout_verify_{args.tag}_{stamp}"
    report = {"tag": args.tag, "timestamp": stamp, "steps": [],
              "notes": "router=8010, engine=8800, IQ2_XXS, ctx 262144, KV q8_0"}
    tests = [t.strip().upper() for t in args.tests.split(",")]
    if "A" in tests:
        test_a(report, args)
    if "B" in tests or "C" in tests:
        test_bc(report, args)
    out_base.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out_base}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
