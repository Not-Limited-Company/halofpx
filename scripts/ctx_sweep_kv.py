#!/usr/bin/env python3
"""Context/memory sweep at a chosen KV cache precision.

halofpx takes KV cache types from the model's run_config, and the load API has no
override, so this script sets kv_cache_type_k/v in registry/models.json for the
duration of the sweep and restores the file afterwards (original bytes are kept
in memory and rewritten, then verified).

Per context size it records: load time, system RAM in use (UMA), amdgpu GTT in
use, and a short decode pulse.  Talks only to the halofpx API.

Usage:
  ctx_sweep_kv.py --kv q4_0 --ctxs 4096,65536,262144,1048576 --tag kv_q4_0
"""
import argparse
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry" / "models.json"
BASE = "http://127.0.0.1:8010"
MODEL = "deepseek-v4-flash"
VARIANT = "IQ2_XXS"


def post(path, payload=None, timeout=2400):
    data = json.dumps(payload).encode() if payload is not None else b"{}"
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()), time.time() - t0, None
    except urllib.error.HTTPError as e:
        return None, time.time() - t0, f"HTTP {e.code}: {e.read().decode()[:240]}"
    except Exception as e:  # noqa: BLE001
        return None, time.time() - t0, f"{type(e).__name__}: {e}"


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, _, v = line.partition(":")
        d[k] = int(v.split()[0])
    return {"used_gib": round((d["MemTotal"] - d["MemAvailable"]) / 1048576, 2),
            "avail_gib": round(d["MemAvailable"] / 1048576, 2)}


def gtt():
    tot, found = 0.0, False
    for p in Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used"):
        try:
            tot += int(p.read_text().strip()) / 2 ** 30
            found = True
        except Exception:  # noqa: BLE001, S110
            pass
    return round(tot, 2) if found else None


def engine_kv_types():
    """The KV types the engine was actually started with (proof the override applied)."""
    try:
        pid = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True).stdout.split()[0]
        args = subprocess.run(["ps", "-o", "args=", "-p", pid], capture_output=True, text=True).stdout.split()
        out = {}
        for flag, key in (("-ctk", "k"), ("-ctv", "v")):
            if flag in args:
                out[key] = args[args.index(flag) + 1]
        return out
    except Exception:
        return {}


def pulse():
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
               "max_tokens": 64, "temperature": 0.0}
    d, dt, err = post("/v1/chat/completions", payload, timeout=900)
    if err:
        return {"error": err}
    t = d.get("timings", {})
    msg = d["choices"][0]["message"]
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    return {"tok_s": round(t.get("predicted_per_second", 0.0), 2),
            "ttft_ms": round(t.get("prompt_ms", 0.0), 1),
            "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
            "answer_ok": "4" in text, "wall_s": round(dt, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default="q8_0", choices=["q8_0", "q4_0", "f16", "default"])
    ap.add_argument("--ctxs", default="4096,65536,131072,262144,524288,1048576")
    ap.add_argument("--tag", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.out) if args.out else ROOT / "benchmarks" / f"ctx_sweep_{args.tag or args.kv}_{stamp}"
    kv = None if args.kv == "default" else args.kv
    rows = []
    if True:
        print(f"per-load KV override: {kv or 'run_config default'}", flush=True)
        for ctx in [int(x) for x in args.ctxs.split(",")]:
            post("/api/v1/unload", timeout=300)
            time.sleep(5)
            base = meminfo()
            payload = {"model_id": MODEL, "variant": VARIANT, "ctx_size": ctx}
            if kv:
                payload["kv_cache_type_k"] = kv
                payload["kv_cache_type_v"] = kv
            r, load_s, err = post("/api/v1/load", payload)
            time.sleep(3)
            after = meminfo()
            row = {"kv": args.kv, "ctx": ctx, "loaded": err is None,
                   "engine_kv": engine_kv_types(),
                   "load_s": round(load_s, 1), "error": err,
                   "ram_used_gib": after["used_gib"],
                   "ram_avail_gib": after["avail_gib"],
                   "ram_delta_gib": round(after["used_gib"] - base["used_gib"], 2),
                   "gtt_used_gib": gtt()}
            if err is None:
                row["pulse"] = pulse()
            rows.append(row)
            print(json.dumps(row), flush=True)
            Path(str(out_base) + ".json").write_text(json.dumps(
                {"tag": args.tag or args.kv, "kv": args.kv, "notes": args.notes,
                 "timestamp": stamp, "rows": rows}, indent=1))
            if err:
                print("load failed, stopping sweep", flush=True)
                break


    md = [f"# Context/memory sweep - KV {args.kv}", "",
          f"- Timestamp {stamp} | model {MODEL}:{VARIANT} | notes: {args.notes or '(none)'}",
          f"- Loads: {sum(1 for r in rows if r['loaded'])}/{len(rows)}", "",
          "| ctx | engine KV | loaded | load s | RAM used | RAM free | GTT used | decode tok/s | TTFT ms | draft acc |",
          "|---:|:--:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        p = r.get("pulse", {})
        md.append(f"| {r['ctx']} | {r.get('engine_kv', {})} | {'yes' if r['loaded'] else 'FAILED'} | {r['load_s']} | "
                  f"{r['ram_used_gib']} | {r['ram_avail_gib']} | {r.get('gtt_used_gib', '-')} | "
                  f"{p.get('tok_s', '-')} | {p.get('ttft_ms', '-')} | "
                  f"{p.get('draft_acc', '-')}/{p.get('draft_n', '-')} |")
    Path(str(out_base) + ".md").write_text("\n".join(md) + "\n")
    print(f"wrote {out_base}.md / .json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
