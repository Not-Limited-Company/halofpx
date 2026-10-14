#!/usr/bin/env python3
"""Perplexity probe with llama-perplexity at a chosen KV cache precision.

llama-perplexity loads the model standalone, so the served engine is unloaded
first and left unloaded (reload with `halofpx load` afterwards).  Corpus is
wikitext-2 raw (test split) fetched once into benchmarks/corpora/.

Usage:
  ppl_probe.py --kv q8_0 --chunks 20 --ctx 512 --tag kv_q8_0
"""
import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE_BIN = ROOT / "engine" / "src" / "build-gfx1151-dual" / "bin"
MODEL = ROOT / "models" / "deepseek-v4-flash" / "DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf"
CORPUS_DIR = ROOT / "benchmarks" / "corpora"

PROBE_ENV = {
    "HSA_OVERRIDE_GFX_VERSION": "11.5.1",
    "GGML_HIP_ENABLE_UNIFIED_MEMORY": "1",
    "LD_PRELOAD": "/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1",
    "LD_LIBRARY_PATH": str(ENGINE_BIN),
}


def fetch_corpus() -> Path:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for existing in CORPUS_DIR.glob("wiki*"):
        if existing.is_file() and existing.suffix != ".zip":
            return existing
    api = "https://huggingface.co/api/datasets/ggml-org/ci/tree/main?recursive=true"
    listing = json.load(urllib.request.urlopen(api, timeout=120))
    cands = [e["path"] for e in listing
             if e.get("type") == "file" and "wikitext" in e["path"].lower()]
    if not cands:
        raise SystemExit("no wikitext fixture found in ggml-org/ci")
    url = f"https://huggingface.co/datasets/ggml-org/ci/resolve/main/{cands[0]}"
    print(f"downloading corpus: {url}", flush=True)
    raw = CORPUS_DIR / Path(cands[0]).name
    urllib.request.urlretrieve(url, raw)
    if raw.suffix == ".zip":
        with zipfile.ZipFile(raw) as z:
            z.extractall(CORPUS_DIR)
        inner = [p for p in CORPUS_DIR.glob("**/*")
                 if p.is_file() and p.suffix in (".raw", ".txt")]
        if not inner:
            raise SystemExit("zip had no .raw/.txt corpus")
        return inner[0]
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default="q8_0")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-loaded", action="store_true", help="do not unload the engine")
    ap.add_argument("--batch", type=int, default=None,
                    help="logical batch size; set equal to --ctx to force a single sequence (n_seq=1)")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.out) if args.out else ROOT / "benchmarks" / f"ppl_{args.tag or args.kv}_{stamp}"
    corpus = fetch_corpus()

    if not args.keep_loaded:
        try:
            req = urllib.request.Request("http://127.0.0.1:8010/api/v1/unload", data=b"{}",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=300).read()
            print("engine unloaded for a standalone perplexity run", flush=True)
            time.sleep(5)
        except Exception as e:  # noqa: BLE001
            print(f"unload skipped: {e}", flush=True)

    env = dict(os.environ)
    env.update(PROBE_ENV)
    cmd = [str(ENGINE_BIN / "llama-perplexity"), "-m", str(MODEL), "-f", str(corpus),
           "-c", str(args.ctx), "--chunks", str(args.chunks),
           "-ctk", args.kv, "-ctv", args.kv]
    if args.batch:
        cmd += ["-b", str(args.batch), "-ub", str(args.batch)]
    print("running:", " ".join(cmd), flush=True)
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.time() - t0
    out = pr.stdout + pr.stderr
    ppl = re.findall(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", out)
    err = None
    if not ppl:
        err = out.strip().splitlines()[-6:] if out.strip() else f"exit {pr.returncode}"
    result = {
        "tag": args.tag or args.kv, "kv": args.kv, "ctx": args.ctx,
        "chunks": args.chunks, "corpus": str(corpus), "timestamp": stamp,
        "ppl": float(ppl[0][0]) if ppl else None,
        "ppl_err": float(ppl[0][1]) if ppl else None,
        "wall_s": round(wall, 1), "notes": args.notes, "error": err,
        "tail": out.strip().splitlines()[-12:],
    }
    Path(str(out_base) + ".json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "tail"}), flush=True)
    print(f"log tail: {result['tail']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
