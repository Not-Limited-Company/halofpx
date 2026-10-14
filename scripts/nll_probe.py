#!/usr/bin/env python3
"""Official-continuation NLL probe (ds4fa quality-testing harness).

Scores the tracked DeepSeek V4 Flash continuations with ds4fa's own
`score_llama` tool (a llama.cpp-API scorer) and compares against the baseline
derived from the official API responses that ship with the fixture.  The metric
is target-token negative log-likelihood, so it isolates quantization damage
without judging a sampled answer.

score_llama loads the model standalone: unload the served engine first.

Usage: nll_probe.py --kv q8_0 --ctx 8192 --tag kv_q8_0
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path("/opt/ds4-strix-halo/gguf-tools/quality-testing/data/flash")
SCORER = Path("/tmp/opencode/score_llama")
MODEL = ROOT / "models" / "deepseek-v4-flash" / "DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf"
PROBE_ENV = {
    "HSA_OVERRIDE_GFX_VERSION": "11.5.1",
    "GGML_HIP_ENABLE_UNIFIED_MEMORY": "1",
    "LD_PRELOAD": "/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1",
    "LD_LIBRARY_PATH": str(ROOT / "engine" / "src" / "build-gfx1151-dual" / "bin"),
}


def official_baseline():
    """Official per-case NLL from the tracked API responses (top_logprobs)."""
    rows = {}
    for resp in sorted((FIXTURE / "responses").glob("case_*.json")):
        d = json.loads(resp.read_text())
        content = d["choices"][0].get("logprobs", {}).get("content") or []
        nll = 0.0
        n = 0
        for tok in content:
            lp = tok.get("logprob")
            if lp is None:
                continue
            nll += -float(lp)
            n += 1
        if n:
            rows[resp.stem] = {"target_tokens": n, "nll": nll, "avg_nll": nll / n}
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default="q8_0")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--tag", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--keep-loaded", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = ROOT / "benchmarks" / f"nll_{args.tag or args.kv}_{stamp}"
    manifest = Path("/tmp/opencode/nll_manifest.tsv")
    out_tsv = out_base.with_suffix(".tsv")

    if not args.keep_loaded:
        try:
            import urllib.request
            req = urllib.request.Request("http://127.0.0.1:8010/api/v1/unload", data=b"{}",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=300).read()
            print("engine unloaded for the standalone scorer", flush=True)
            time.sleep(5)
        except Exception as e:  # noqa: BLE001
            print(f"unload skipped: {e}", flush=True)

    if args.kv != "q8_0":
        # score_llama has no KV-type option: it always uses the engine default.
        print(f"note: score_llama has no KV flag; scoring uses the engine default KV, "
              f"not --kv {args.kv}", flush=True)

    rows = []
    for line in (FIXTURE / "manifest.tsv").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        cid, prompt, cont, resp = line.split("\t")
        # manifest paths are relative to the ds4fa checkout root (4 levels up)
        repo_root = FIXTURE.parents[3]
        rows.append(f"{cid}\t{repo_root / prompt}\t{repo_root / cont}\t{repo_root / resp}")
    manifest.write_text("\n".join(rows) + "\n")

    env = dict(os.environ)
    env.update(PROBE_ENV)
    cmd = [str(SCORER), str(MODEL), str(manifest), str(out_tsv), str(args.ctx), "auto"]
    print("running:", " ".join(cmd), flush=True)
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env)
    wall = time.time() - t0
    tail = (pr.stdout + pr.stderr).strip().splitlines()[-6:]

    ours = {}
    if out_tsv.exists():
        for line in out_tsv.read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            ours[parts[0]] = {"prompt_tokens": int(parts[1]), "target_tokens": int(parts[2]),
                              "nll": float(parts[3]), "avg_nll": float(parts[4]),
                              "first_match": parts[5], "greedy_lcp": parts[6]}
    official = official_baseline()

    common = sorted(set(ours) & set(official))
    def mean(values):
        return sum(values) / len(values) if values else None

    our_avg = mean([ours[c]["avg_nll"] for c in common])
    off_avg = mean([official[c]["avg_nll"] for c in common])
    result = {
        "tag": args.tag or args.kv, "kv": args.kv, "ctx": args.ctx, "timestamp": stamp,
        "cases_scored": len(ours), "cases_compared": len(common),
        "our_mean_avg_nll": round(our_avg, 4) if our_avg else None,
        "official_mean_avg_nll": round(off_avg, 4) if off_avg else None,
        "delta_avg_nll": round(our_avg - off_avg, 4) if our_avg and off_avg else None,
        "our_total_nll": round(sum(ours[c]["nll"] for c in common), 2) if common else None,
        "wall_s": round(wall, 1), "notes": args.notes,
        "scorer_tail": tail, "per_case": {c: ours[c] for c in common},
    }
    out_base.with_suffix(".json").write_text(json.dumps(result, indent=1))
    md = [f"# Official-continuation NLL - {args.tag or args.kv}", "",
          f"- Cases: {len(common)} compared (scored {len(ours)})",
          f"- Our mean avg NLL: **{result['our_mean_avg_nll']}** vs official **{result['official_mean_avg_nll']}**",
          f"- Delta: **{result['delta_avg_nll']}** nats/token",
          f"- Wall: {result['wall_s']} s | ctx {args.ctx} | KV {args.kv}",
          f"- Notes: {args.notes or '(none)'}",
          "", "| case | target tokens | our avg NLL | official avg NLL | delta |", "|---|---:|---:|---:|---:|"]
    for c in common:
        md.append(f"| {c} | {ours[c]['target_tokens']} | {ours[c]['avg_nll']:.4f} | "
                  f"{official[c]['avg_nll']:.4f} | {ours[c]['avg_nll'] - official[c]['avg_nll']:+.4f} |")
    out_base.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_case"}), flush=True)
    print(f"scorer tail: {tail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
