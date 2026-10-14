#!/usr/bin/env python3
"""Run the ds4-eval 92-question suite against a halofpx/llama.cpp endpoint.

Prompts, and the grading, come from ds4fa's own harness:
  - same system + question templates (build_question_prompt in ds4_eval.c)
  - answers are written into a trace and graded by `ds4-eval --regrade-trace`,
    so the verdicts use the same extractor/grader as the published 82/92 run.

Usage:
  ds4_eval_runner.py --limit 6                    # pilot
  ds4_eval_runner.py --tag q8_0 --out benchmarks/ds4_eval_q8_0
"""
import argparse
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "benchmarks" / "ds4_eval_92.json"
DEFAULT_GRADER = Path("/opt/ds4-strix-halo/ds4-eval")

SYSTEM_PROMPT = ("You are solving a hard benchmark question. Reason carefully. "
                 "The final answer must follow the requested format exactly.")

MC_TAIL = ("\nSolve the question. At the end, write exactly one final line in this "
           "format and do not write anything after it:\nAnswer: <letter>")
COMPSEC_TAIL = ("\nAt the end, write exactly one final line in this format and do not "
                "write anything after it:\nAnswer: <line number or comma-separated line numbers>")
INT_TAIL = ("\nSolve the problem. At the end, write exactly one final line in this "
            "format and do not write anything after it:\nAnswer: <integer>")


def build_prompt(case: dict) -> str:
    parts = [case["question"]]
    if case["choices"]:
        parts.append("\nChoices:\n")
        for i, choice in enumerate(case["choices"]):
            parts.append(f"{chr(ord('A') + i)}. {choice}\n")
        parts.append(MC_TAIL)
    elif case["source"] == "COMPSEC":
        parts.append(COMPSEC_TAIL)
    else:
        parts.append(INT_TAIL)
    return "".join(parts)


def ask(base, prompt, max_tokens, timeout=3600):
    payload = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
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
    ap.add_argument("--base-url", default="http://127.0.0.1:8800",
                    help="engine base url (direct engine port avoids the router's 300s cap)")
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--grader", default=str(DEFAULT_GRADER))
    ap.add_argument("--out", default=None, help="output basename (no extension)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="only first N cases (pilot)")
    ap.add_argument("--tag", default="", help="label for reports, e.g. kv_q8_0")
    ap.add_argument("--resume-from", default=None,
                    help="existing trace file: skip the cases it already contains and append")
    ap.add_argument("--dry-run", action="store_true", help="report the resume plan and exit")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.out) if args.out else ROOT / "benchmarks" / f"ds4_eval_{args.tag or 'run'}_{stamp}"
    trace_path = out_base.with_suffix(".trace.txt")
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    cases = json.load(open(args.cases))["cases"]
    if args.limit:
        cases = cases[: args.limit]

    results = []
    trace_lines = []
    done_indices = set()
    if args.resume_from:
        resumed = Path(args.resume_from)
        text = resumed.read_text()
        trace_lines = text.rstrip("\n").split("\n")
        for line in trace_lines:
            if line.startswith("===== CASE "):
                done_indices.add(int(line.split()[2].split("/")[0]))
        if args.out is None and str(resumed).endswith(".trace.txt"):
            out_base = Path(str(resumed)[: -len(".trace.txt")])
            trace_path = resumed
        print(f"resuming: {len(done_indices)} case(s) already in {resumed}", flush=True)
    if args.dry_run:
        pending = [i for i in range(1, len(cases) + 1) if i not in done_indices]
        print(f"dry run: {len(pending)} case(s) pending: {pending[:10]}", flush=True)
        return 0
    for idx, case in enumerate(cases, 1):
        if idx in done_indices:
            continue
        prompt = build_prompt(case)
        d, wall, err = ask(args.base_url, prompt, args.max_tokens)
        row = {
            "index": idx,
            "source": case["source"],
            "id": case["id"],
            "domain": case.get("domain", ""),
            "expected": case["answer"],
            "wall_s": round(wall, 1),
        }
        if err:
            row.update({"ok": False, "error": err, "output": "", "truncated": False})
            print(f"[{idx}/{len(cases)}] {case['source']}/{case['id']} ERROR {err}", flush=True)
        else:
            t = d.get("timings", {})
            u = d.get("usage", {})
            msg = d["choices"][0]["message"]
            reasoning = msg.get("reasoning_content") or ""
            content = msg.get("content") or ""
            output = (reasoning + ("\n" if reasoning and content else "") + content).strip()
            row.update({
                "ok": True,
                "prompt_tokens": u.get("prompt_tokens"),
                "generated_tokens": u.get("completion_tokens"),
                "decode_tps": round(t.get("predicted_per_second", 0.0), 2),
                "draft_n": t.get("draft_n"),
                "draft_accepted": t.get("draft_n_accepted"),
                "finish_reason": d["choices"][0].get("finish_reason"),
                "truncated": d["choices"][0].get("finish_reason") == "length",
                "output": output,
            })
            body = output.encode()
            trace_lines.append(f"===== CASE {idx}/{len(cases)} {case['source']} {case['id']} =====")
            trace_lines.append(f"source: {case['source']}")
            trace_lines.append(f"id: {case['id']}")
            trace_lines.append("status: PASSED")
            trace_lines.append("picked: __sentinel__")
            trace_lines.append(f"MODEL_OUTPUT_BEGIN bytes={len(body)}")
            trace_lines.append(output)
            trace_lines.append("MODEL_OUTPUT_END")
            trace_lines.append("")
            print(f"[{idx}/{len(cases)}] {case['source']}/{case['id']} "
                  f"gen={row['generated_tokens']} tok {row['decode_tps']} tok/s "
                  f"{row['wall_s']}s{ ' TRUNCATED' if row['truncated'] else ''}", flush=True)
        results.append(row)
        trace_path.write_text("\n".join(trace_lines) + ("\n" if trace_lines else ""))

    # authoritative grading by ds4fa's own grader
    graded, summary = {}, ""
    if trace_lines:
        pr = subprocess.run([args.grader, "--regrade-trace", str(trace_path)],
                            capture_output=True, text=True)
        summary = pr.stdout.strip().splitlines()[-1] if pr.stdout.strip() else pr.stderr.strip()
        for line in pr.stdout.splitlines():
            m = re.match(r"case (\d+) (.+)/(\S+): .* -> regrade (PASSED|FAILED) picked=(\S*) expected=(\S*)", line)
            if m:
                graded[int(m.group(1))] = {"verdict": m.group(4), "picked": m.group(5)}
        print(f"grader: {summary}", flush=True)

    for row in results:
        g = graded.get(row["index"], {})
        row["verdict"] = g.get("verdict", "NOT_GRADED")
        row["picked"] = g.get("picked", "")

    graded_rows = [r for r in results if r["verdict"] in ("PASSED", "FAILED")]
    passed = sum(1 for r in graded_rows if r["verdict"] == "PASSED")
    by_source: dict[str, dict[str, int]] = {}
    for r in graded_rows:
        s = by_source.setdefault(r["source"], {"passed": 0, "total": 0})
        s["total"] += 1
        s["passed"] += 1 if r["verdict"] == "PASSED" else 0

    report = {
        "tag": args.tag,
        "notes": args.notes,
        "timestamp": stamp,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "cases_json": str(args.cases),
        "grader": str(args.grader),
        "grader_summary": summary,
        "score": f"{passed}/{len(graded_rows)}" if graded_rows else "n/a",
        "passed": passed,
        "graded": len(graded_rows),
        "by_source": by_source,
        "mean_decode_tps": round(sum(r.get("decode_tps", 0) for r in graded_rows) / len(graded_rows), 2) if graded_rows else None,
        "mean_generated_tokens": round(sum(r.get("generated_tokens", 0) for r in graded_rows) / len(graded_rows), 1) if graded_rows else None,
        "truncated_cases": sum(1 for r in graded_rows if r.get("truncated")),
        "results": results,
    }
    out_json = out_base.with_suffix(".json")
    out_json.write_text(json.dumps(report, indent=1))

    md = [f"# ds4-eval 92-question suite — {args.tag or 'run'}",
          "",
          f"- **Score: {report['score']}** (graded by ds4fa's own `ds4-eval --regrade-trace`)",
          f"- Timestamp: {stamp} | engine: {args.base_url} | max_tokens: {args.max_tokens}",
          f"- Mean decode: {report['mean_decode_tps']} tok/s | mean generated: {report['mean_generated_tokens']} tokens",
          f"- Truncated at the token cap: {report['truncated_cases']} case(s)",
          f"- Notes: {args.notes}" if args.notes else "- Notes: (none)",
          "",
          "| Source | Passed | Total |",
          "|---|---:|---:|"]
    for s, v in sorted(by_source.items()):
        md.append(f"| {s} | {v['passed']} | {v['total']} |")
    md += ["", "| # | Source | id | expected | picked | verdict | gen tokens | tok/s |", "|---:|---|---|---|---|---|---:|---:|"]
    for r in results:
        md.append(f"| {r['index']} | {r['source']} | {r['id']} | {r['expected']} | "
                  f"{r.get('picked', '')} | {r['verdict']} | {r.get('generated_tokens', '-')} | "
                  f"{r.get('decode_tps', '-')} |")
    out_md = out_base.with_suffix(".md")
    out_md.write_text("\n".join(md) + "\n")
    print(f"score={report['score']} -> {out_md} / {out_json} / {trace_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
