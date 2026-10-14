#!/usr/bin/env python3
"""Concurrency probe: per-stream and aggregate decode with N simultaneous requests.

Everything else in this study is single-stream. If the deployment serves one to
two clients, the question is what two concurrent streams cost each other.

Usage: concurrency_probe.py --requests 2 --max-tokens 256 --tag slots2
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUTER = "http://127.0.0.1:8010"
PROMPTS = [
    "Explain in a short paragraph why a B-tree keeps its height logarithmic.",
    "Write a short Python function that reverses a linked list, with a one-line comment.",
    "Summarise the tradeoff between paging and swapping in three sentences.",
    "Give a short explanation of what a memory barrier does for CPU code.",
]


def worker(idx, prompt, max_tokens, out):
    payload = json.dumps({"messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(ROUTER + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read().decode())
        t = d.get("timings", {})
        u = d.get("usage", {})
        out[idx] = {"ok": True, "wall_s": round(time.time() - t0, 1),
                    "generated_tokens": u.get("completion_tokens"),
                    "decode_tps": round(t.get("predicted_per_second", 0.0), 2),
                    "ttft_ms": round(t.get("prompt_ms", 0.0), 1),
                    "draft_acc": t.get("draft_n_accepted"), "draft_n": t.get("draft_n")}
    except Exception as e:  # noqa: BLE001
        out[idx] = {"ok": False, "wall_s": round(time.time() - t0, 1),
                    "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.out) if args.out else ROOT / "benchmarks" / f"concurrency_{args.tag}_{stamp}"
    out = {}
    threads = [threading.Thread(target=worker, args=(i, PROMPTS[i % len(PROMPTS)],
                                                     args.max_tokens, out))
               for i in range(args.requests)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    rows = [out[i] for i in sorted(out)]
    ok = [r for r in rows if r.get("ok")]
    total_tokens = sum(r.get("generated_tokens") or 0 for r in ok)
    report = {
        "tag": args.tag, "timestamp": stamp, "notes": args.notes,
        "requests": args.requests, "max_tokens": args.max_tokens,
        "wall_s": round(wall, 1),
        "per_stream_decode_tps": [r.get("decode_tps") for r in ok],
        "mean_per_stream_tps": round(sum(r.get("decode_tps", 0) for r in ok) / len(ok), 2) if ok else None,
        "aggregate_tps": round(total_tokens / wall, 2) if wall else None,
        "rows": rows,
    }
    out_base.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
    md = [f"# Concurrency probe - {args.tag}", "",
          f"- {args.requests} simultaneous requests, max_tokens {args.max_tokens}, wall {report['wall_s']} s",
          f"- Mean per-stream decode: **{report['mean_per_stream_tps']} tok/s** | aggregate: **{report['aggregate_tps']} tok/s**",
          f"- Notes: {args.notes or '(none)'}", "",
          "| stream | ok | generated | decode tok/s | TTFT ms | wall s |", "|---:|:--:|---:|---:|---:|---:|"]
    for i, r in enumerate(rows):
        md.append(f"| {i} | {'yes' if r.get('ok') else 'no'} | {r.get('generated_tokens', '-')} | "
                  f"{r.get('decode_tps', '-')} | {r.get('ttft_ms', '-')} | {r.get('wall_s', '-')} |")
    out_base.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"wrote {out_base}.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
