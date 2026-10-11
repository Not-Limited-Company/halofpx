#!/usr/bin/env bash
# build_dspark_drafter.sh — build the native `dflash` DSpark drafter GGUF for halofpx
#
# The drafter is the ~10.9 GiB DSpark ("dflash" arch) head that lets llama-server
# speculate ahead of DeepSeek-V4-Flash-0731. It is converted from the official
# `deepseek-ai/DeepSeek-V4-Flash-DSpark` checkpoint with the engine's own
# `convert_hf_to_gguf.py --dspark` (downloads only the mtp.*-containing shards),
# so it always matches the engine's dflash tensor layout — Lucebox's QF4 drafter
# uses a different arch and hard-loads in this engine.
#
# Usage:
#   ./scripts/build_dspark_drafter.sh                    # default: bf16 head (experts stay MXFP4)
#   ./scripts/build_dspark_drafter.sh --drafter-qtype q8_0
#   ./scripts/build_dspark_drafter.sh --drafter-qtype f16 --outdir /tmp/draft
#
# Any gguf quant type is accepted for --drafter-qtype; the source checkpoint is
# already MXFP4-quantized (DeepSeek official), so lowering to Q8_0/F16 re-quantizes
# those tensors (--allow-requantize). Best quality is the default native build
# (BF16 head + lossless MXFP4 expert repack).
#
# Environment overrides:
#   PYTHON            venv python with torch + transformers + huggingface_hub
#   TARGET_MODEL_DIR  pre-downloaded tokenizer dir (else snapshot_download'd)
#   DFLASH_DRAFT_REPO official DSpark checkpoint repo (default deepseek-ai/DeepSeek-V4-Flash-DSpark)
#   TARGET_REPO       target checkpoint repo for the tokenizer (default deepseek-ai/DeepSeek-V4-Flash-0731)
#   ENGINE_ROOT       the pinned engine tree (default <repo>/engine)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_ROOT="${ENGINE_ROOT:-$ROOT/engine}"
CONVERTER="$ENGINE_ROOT/src/convert_hf_to_gguf.py"
DFLASH_DRAFT_REPO="${DFLASH_DRAFT_REPO:-deepseek-ai/DeepSeek-V4-Flash-DSpark}"
TARGET_REPO="${TARGET_REPO:-deepseek-ai/DeepSeek-V4-Flash-0731}"
QTYPE="bf16"
OUTDIR="$ROOT/models/deepseek-v4-flash"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --drafter-qtype) QTYPE="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,24p' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# --- locate the venv python (needs torch/transformers/huggingface_hub) ----------
if [[ -n "${PYTHON:-}" && -x "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif "$ROOT/.venv/bin/python" -c "import torch, transformers, huggingface_hub" 2>/dev/null; then
    PY="$ROOT/.venv/bin/python"
elif python3 -c "import torch, transformers, huggingface_hub" 2>/dev/null; then
    PY="$(command -v python3)"
else
    echo "❌ no python with torch/transformers/huggingface_hub found (set PYTHON)." >&2
    echo "   e.g. python3 -m venv /tmp/hfpx-venv && /tmp/hfpx-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu transformers huggingface_hub" >&2
    exit 3
fi

# --- locate llama-quantize + its shared lib dir -----------------------------------
LLAMA_QUANTIZE="$(command -v llama-quantize || true)"
LD_DIR=""
if [[ -z "$LLAMA_QUANTIZE" && -x "$ENGINE_ROOT/bin/llama-quantize" ]]; then
    LLAMA_QUANTIZE="$ENGINE_ROOT/bin/llama-quantize"
fi
if [[ -n "$LLAMA_QUANTIZE" ]]; then
    for d in "$ENGINE_ROOT"/src/build-*/bin; do
        if [[ -d "$d" ]] && (ls "$d"/libllama.so* >/dev/null 2>&1); then
            LD_DIR="$d"; break
        fi
    done
fi
if [[ "$QTYPE" != "bf16" && "$QTYPE" != "f16" ]] && [[ -z "$LLAMA_QUANTIZE" ]]; then
    echo "❌ --drafter-qtype $QTYPE needs llama-quantize; none found on PATH or in engine/bin." >&2
    exit 3
fi

# --- resolve the target tokenizer dir ---------------------------------------------
if [[ -n "${TARGET_MODEL_DIR:-}" && -d "${TARGET_MODEL_DIR:-}" ]]; then
    TARGET_DIR="$TARGET_MODEL_DIR"
else
    echo "⬇️  Snapshotting target tokenizer from $TARGET_REPO ..."
    TARGET_DIR="$("$PY" - "$TARGET_REPO" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
print(snapshot_download(
    repo,
    allow_patterns=[
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "pre_tokenizer_config.json",
        "generation_config.json",
    ],
))
PYEOF
)"
fi
echo "ℹ️  Target model dir: $TARGET_DIR"

# --- convert ----------------------------------------------------------------------
STEM="DeepSeek-V4-Flash-0731-DSpark-draft"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/dspark-build.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
OUTFILE="$WORK/$STEM-bf16.gguf"

echo "🔄 Converting $DFLASH_DRAFT_REPO -> dflash drafter (bf16 head)..."
"$PY" "$CONVERTER" \
    --remote --dspark "$DFLASH_DRAFT_REPO" \
    --target-model-dir "$TARGET_DIR" \
    --outtype bf16 \
    --outfile "$OUTFILE"

FINAL_NAME="$STEM-$QTYPE.gguf"
if [[ "$QTYPE" == "bf16" || "$QTYPE" == "f16" ]]; then
    if [[ "$QTYPE" == "f16" ]]; then
        # requantize a second copy to f16 (BF16 is the native head precision; both wheels turn)
        LD_LIBRARY_PATH="${LD_DIR:+$LD_DIR:}$LD_LIBRARY_PATH" "$LLAMA_QUANTIZE" "$OUTFILE" "$WORK/$STEM-f16.gguf" f16 2>/dev/null
        FINAL_FILE="$WORK/$STEM-f16.gguf"
    else
        FINAL_FILE="$OUTFILE"
    fi
else
    echo "🔻 Quantizing drafter to $QTYPE ..."
    LD_LIBRARY_PATH="${LD_DIR:+$LD_DIR:}$LD_LIBRARY_PATH" "$LLAMA_QUANTIZE" \
        --allow-requantize "$OUTFILE" "$WORK/$STEM-$QTYPE.gguf" "$QTYPE"
    FINAL_FILE="$WORK/$STEM-$QTYPE.gguf"
fi

mkdir -p "$OUTDIR"
cp "$FINAL_FILE" "$OUTDIR/$FINAL_NAME"
SHA="$(sha256sum "$OUTDIR/$FINAL_NAME" | cut -d' ' -f1)"
SIZE_GIB="$(du -BG "$OUTDIR/$FINAL_NAME" | cut -f1 | tr -d G)"

echo
echo "✅ Drafter installed: $OUTDIR/$FINAL_NAME"
echo "   size_gib  : $SIZE_GIB GiB"
echo "   sha256    : $SHA"
echo
echo "   Next: halofpx pull deepseek-v4-flash  # (adds the main quant if missing)"
echo "         halofpx load deepseek-v4-flash  # ROCm0 + --spec-type draft-dspark"