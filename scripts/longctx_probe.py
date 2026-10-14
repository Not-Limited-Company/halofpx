#!/usr/bin/env python3
"""Long-context probe: needle retrieval + prefill/decode cost vs depth.

The filler is varied (deterministic pseudo-random sentences from a fixed word
list) so every position is distinguishable; a repeated paragraph makes retrieval
unfair.  One passphrase is planted at a chosen depth fraction and the model is
asked for it.

Talks to llama-server directly (port 8800) so the router's 300 s proxy cap does
not distort long-prefill measurements.

Usage:
  longctx_probe.py --depths 16384,32768,65536 --positions 0.25,0.5,0.75 --repeats 2 \
                   --tag kv_q8_0 --out benchmarks/longctx_kv_q8_0
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
WORDS = ("threshold kernel stream lattice vector gradient sparse routine buffer "
         "archive compile replica dataset aperture quantum segment ledger matrix "
         "conduit harvest ionize junction lattice marrow nocturne orbit prism "
         "quartz relay sift tread union vertex willow zenith anomaly bracket "
         "cascade drift epoch filament gauge helix index jolt keystone lumen "
         "monsoon nexus offset pivot quorum ridge signal tangent uplink vapor "
         "wafer xylem yield zone").split()


def make_filler(rng, n_sentences):
    """Deterministic varied prose; ~28 tokens per sentence."""
    out = []
    for i in range(n_sentences):
        words = [rng.choice(WORDS) for _ in range(18)]
        out.append(f"Note {i:06d}: the {words[0]} {words[1]} of the {words[2]} "
                   f"{words[3]} was measured against the {words[4]} {words[5]}, "
                   f"while {words[6]} {words[7]} remained {words[8]} and the "
                   f"{words[9]} {words[10]} {words[11]} the {words[12]} "
                   f"{words[13]} {words[14]} {words[15]} {words[16]} {words[17]}. ")
    return "".join(out)


def build_prompt(target_tokens, position, needle):
    rng = random.Random(1234 + target_tokens)
    per_sentence = 28
    n = max(4, int(target_tokens / per_sentence))
    anchor = max(1, int(n * position))
    head = make_filler(rng, anchor)
    tail = make_filler(rng, n - anchor)
    return (f"Background notes:\n{head}\nCRITICAL RECORD: the vault passphrase is "
            f"{needle}.\n{tail}\nQuestion: What is the vault passphrase mentioned "
            f"in the background notes? Answer with just the passphrase.\n")


def ask(base, prompt, max_tokens, timeout=5400):
    payload = json.dumps({"messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=payload,
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
    ap.add_argument("--base-url", default="http://127.0.0.1:8800")
    ap.add_argument("--depths", default="16384,32768,65536")
    ap.add_argument("--positions", default="0.25,0.5,0.75")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.out) if args.out else ROOT / "benchmarks" / f"longctx_{args.tag}_{stamp}"
    depths = [int(x) for x in args.depths.split(",")]
    positions = [float(x) for x in args.positions.split(",")]

    rows = []
    for depth in depths:
        for position in positions:
            for rep in range(args.repeats):
                needle = f"ZEBRA-CANYON-{depth}-{int(position * 100)}-{rep}"
                prompt = build_prompt(depth, position, needle)
                d, wall, err = ask(args.base_url, prompt, args.max_tokens)
                row = {"depth": depth, "position": position, "repeat": rep,
                       "wall_s": round(wall, 1)}
                if err:
                    row.update({"ok": False, "error": err, "found": None})
                else:
                    t = d.get("timings", {})
                    u = d.get("usage", {})
                    msg = d["choices"][0]["message"]
                    answer = msg.get("content") or ""
                    text = answer + (msg.get("reasoning_content") or "")
                    pn = u.get("prompt_tokens") or t.get("prompt_n") or 0
                    pms = t.get("prompt_ms", 0.0) or 0.0
                    row.update({
                        "ok": True,
                        "prompt_tokens": pn,
                        "prefill_s": round(pms / 1000.0, 1),
                        "prefill_tps": round(pn / (pms / 1000.0), 1) if pms else None,
                        "decode_tps": round(t.get("predicted_per_second", 0.0), 2),
                        "generated_tokens": u.get("completion_tokens"),
                        "draft_acc": t.get("draft_n_accepted"),
                        "draft_n": t.get("draft_n"),
                        "found": needle in text,
                        "found_in_answer": needle in answer,
                        "answer": (msg.get("content") or "")[:200],
                    })
                rows.append(row)
                print(json.dumps(row), flush=True)
                Path(str(out_base) + ".json").write_text(json.dumps(
                    {"tag": args.tag, "notes": args.notes, "timestamp": stamp,
                     "base_url": args.base_url, "max_tokens": args.max_tokens,
                     "rows": rows}, indent=1))

    ok_rows = [r for r in rows if r.get("ok")]
    found = sum(1 for r in ok_rows if r["found"])
    rate = f"{100.0 * found / len(ok_rows):.0f}%" if ok_rows else "n/a (no successful requests)"
    md = [f"# Long-context probe — {args.tag}", "",
          f"- Retrieval: **{found}/{len(ok_rows)}** ({rate})",
          f"- Timestamp {stamp} | engine {args.base_url} | max_tokens {args.max_tokens}",
          f"- Notes: {args.notes}" if args.notes else "- Notes: (none)", "",
          "| depth (tokens) | needle at | repeat | prompt tokens | prefill s | prefill tok/s | decode tok/s | found |",
          "|---:|---:|---:|---:|---:|---:|---:|:--:|"]
    for r in rows:
        md.append(f"| {r['depth']} | {int(r['position'] * 100)}% | {r['repeat']} | "
                  f"{r.get('prompt_tokens', '-')} | {r.get('prefill_s', '-')} | "
                  f"{r.get('prefill_tps', '-')} | {r.get('decode_tps', '-')} | "
                  f"{'yes' if r.get('found') else 'no'}"
                  f"{' (answer)' if r.get('found_in_answer') else ''} |")
    Path(str(out_base) + ".md").write_text("\n".join(md) + "\n")
    print(f"retrieval {found}/{len(ok_rows)} -> {out_base}.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
