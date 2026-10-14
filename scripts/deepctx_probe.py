#!/usr/bin/env python3
"""Deep-context probe with realistic content: retrieval AND termination.

Two things are measured at depth, and they are different questions:
  - retrieval: does the response mention the planted fact at all
  - termination: does it produce a final answer, or run past the output cap

Content is natural text (Wikipedia prose, or real source/docs from this repo)
rather than repeated filler, which was unrepresentative and may itself provoke
looping.  The prefix (content + planted fact) is identical across conditions, so
the prompt cache makes every variant after the first nearly free.

Usage:
  deepctx_probe.py --contents wikitext,code --depths 46000,92000 \
      --conditions cap1024_plain,cap1024_strict,cap4096_plain,cap4096_strict \
      --tag rb4096 --out benchmarks/deepctx_rb4096
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = "http://127.0.0.1:8800"
NEEDLE = "ZQGX-QUARTZ-8842"          # nonsense, so it cannot appear in a natural corpus
NEEDLE_MARK = "QUARTZ-8842"          # for counting partial recall

SOURCES = {
    "wikitext": [ROOT / "benchmarks" / "corpora" / "wikitext-2-raw" / "wiki.test.raw"],
    # source only: the docs under docs/ discuss these probes and would contaminate
    "code": [ROOT / "halofpx" / "server.py", ROOT / "halofpx" / "engine_manager.py",
             ROOT / "halofpx" / "registry.py", ROOT / "halofpx" / "model_manager.py",
             ROOT / "halofpx" / "cli.py", ROOT / "scripts" / "benchmark.py",
             ROOT / "scripts" / "ds4_eval_runner.py", ROOT / "scripts" / "benchmark.py",
             ROOT / "halofpx" / "telemetry.py", ROOT / "halofpx" / "config.py"],
}

TAILS = {
    "plain": ("\n\nQuestion: What is the vault passphrase stated in the material above? "
              "Answer with just the passphrase."),
    "strict": ("\n\nAt the end, write exactly one final line in this format and do not "
               "write anything after it:\nAnswer: <passphrase>"),
}


def load_content(kind: str, target_tokens: int, position: float) -> str:
    text = ""
    for path in SOURCES[kind]:
        text += path.read_text(errors="replace") + "\n\n"
        if len(text) / 4.0 > target_tokens * 1.3:
            break
    approx_chars = int(target_tokens * 4)
    body = text[:approx_chars] if len(text) > approx_chars else text
    # pad with more prose if the corpus was short
    while len(body) < approx_chars:
        body += text
    if NEEDLE in body or NEEDLE_MARK in body:
        raise SystemExit(f"corpus '{kind}' already contains the needle; pick another")
    cut = int(len(body) * position)
    # land the insertion on a line boundary so the text stays natural
    nl = body.find("\n", cut)
    cut = nl if nl != -1 else cut
    return (body[:cut] + f"\nThe vault passphrase is {NEEDLE}.\n" + body[cut:])


def ask(prompt: str, max_tokens: int, timeout=3600):
    payload = json.dumps({"messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(ENGINE + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()), time.time() - t0, None
    except urllib.error.HTTPError as e:
        return None, time.time() - t0, f"HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:  # noqa: BLE001
        return None, time.time() - t0, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contents", default="wikitext,code")
    ap.add_argument("--depths", default="46000,92000")
    ap.add_argument("--position", type=float, default=0.5)
    ap.add_argument("--conditions", default="cap1024_plain,cap1024_strict,cap4096_plain,cap4096_strict")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.out) if args.out else ROOT / "benchmarks" / f"deepctx_{args.tag}_{stamp}"
    conditions = []
    for spec in args.conditions.split(","):
        cap, _, style = spec.partition("_")
        conditions.append((spec, int(cap[3:]), style))

    rows = []
    for kind in args.contents.split(","):
        for depth in [int(d) for d in args.depths.split(",")]:
            base = ("Background material follows, then a question about it.\n\n"
                    + load_content(kind, depth, args.position))
            for name, cap, style in conditions:
                prompt = base + TAILS[style]
                d, wall, err = ask(prompt, cap)
                row = {"content": kind, "target_tokens": depth, "condition": name,
                       "max_tokens": cap, "style": style, "wall_s": round(wall, 1)}
                if err:
                    row.update({"ok": False, "error": err})
                else:
                    t = d.get("timings", {})
                    u = d.get("usage", {})
                    msg = d["choices"][0]["message"]
                    answer = msg.get("content") or ""
                    full = answer + (msg.get("reasoning_content") or "")
                    gen = u.get("completion_tokens") or 0
                    row.update({
                        "ok": True,
                        "prompt_tokens": u.get("prompt_tokens"),
                        "prefill_s": round(t.get("prompt_ms", 0) / 1000.0, 1),
                        "generated_tokens": gen,
                        "hit_cap": gen >= cap,
                        "retrieved": NEEDLE in full,
                        "partial": NEEDLE_MARK in full,
                        "answer_text": answer[:600],
                        # a non-empty content block means the model emitted a final
                        # answer; reasoning lives in reasoning_content
                        "answered": bool(answer.strip()),
                        "answer_chars": len(answer.strip()),
                        "correct": NEEDLE in answer,
                        "answer_tail": answer[-120:].replace("\n", " "),
                    })
                rows.append(row)
                print(json.dumps(row), flush=True)
                out_base.with_suffix(".json").write_text(json.dumps(
                    {"tag": args.tag, "notes": args.notes, "timestamp": stamp,
                     "needle": NEEDLE, "rows": rows}, indent=1))

    md = [f"# Deep-context probe - {args.tag}", "",
          f"- Needle at {int(args.position * 100)}% of the material; conditions vary the tail only",
          f"- Notes: {args.notes or '(none)'}", "",
          "| content | target tokens | condition | prompt tokens | prefill s | generated | hit cap | retrieved | correct answer |",
          "|---|---:|---|---:|---:|---:|:--:|:--:|:--:|"]
    for r in rows:
        md.append(f"| {r['content']} | {r['target_tokens']} | {r['condition']} | "
                  f"{r.get('prompt_tokens', '-')} | {r.get('prefill_s', '-')} | "
                  f"{r.get('generated_tokens', '-')} | "
                  f"{'yes' if r.get('hit_cap') else 'no'} | "
                  f"{'yes' if r.get('retrieved') else 'no'} | "
                  f"{'yes' if r.get('correct') else 'no'} |")
    out_base.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"wrote {out_base}.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
