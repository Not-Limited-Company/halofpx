# DSpark Speculative Decoding Guide — DeepSeek V4 Flash

DeepSeek V4 Flash ships with an official **DSpark** draft head (`deepseek-ai/DeepSeek-V4-Flash-DSpark`:
3 dflash blocks, `block_size=5`, `markov_rank=256`, `target_layers=[40,41,42]`, mask/noise token
`128799`). halofpx builds a **native `dflash`-arch drafter** GGUF from that checkpoint so the pinned
ROCmFPX engine can run `--spec-type draft-dspark` — the draft model that makes the advertised
**32 tok/s** configuration reachable.

## Why a native drafter and not someone else's GGUF

The drafter **must** be in the engine's `dflash` dialect with a `markov_w1.weight` tensor (used for
auto-detection). Community drafter GGs (e.g. Lucebox's QF4 file) use `deepseek4-dflash-draft`/
other arch names the engine's arch table does not accept → hard load error. Building from the
official checkpoint through the engine's own `convert_hf_to_gguf.py --dspark` guarantees the exact
tensor layout this engine expects. Only the `mtp.*`-containing safetensors shards are downloaded
(no full 850 GB checkpoint pull).

## Building the drafter (any quantization quality)

```bash
# default: native build, BF16 head + lossless MXFP4 expert repack (~10.9 GiB, recommended)
./scripts/build_dspark_drafter.sh

# any other gguf qtype for the whole file (re-quantizes the MXFP4 experts: use Q8_0/BF16-class
# sources per the repo provenance rule; every re-quant is a fidelity trade-off)
./scripts/build_dspark_drafter.sh --drafter-qtype q8_0

# custom output dir / venv python / pre-downloaded tokenizer dir
PYTHON=/path/to/venv/bin/python ./scripts/build_dspark_drafter.sh \
    --drafter-qtype f16 --outdir /tmp/draft
```

The script:

1. snapshots the target's tokenizer (`deepseek-ai/DeepSeek-V4-Flash-0731`, a few MB) unless
   `TARGET_MODEL_DIR` is set,
2. runs `convert_hf_to_gguf.py --remote --dspark ... --outtype bf16` (downloads only `mtp.*`
   shards; BF16 head, experts stay in the source's native MXFP4 — lossless repack),
3. optionally lowers to the requested qtype with `llama-quantize`,
4. installs `models/deepseek-v4-flash/DeepSeek-V4-Flash-0731-DSpark-draft-<qtype>.gguf` and prints
   its `sha256` (record it back into `registry/models.json` → `draft.sha256`).

Environment overrides: `PYTHON`, `TARGET_MODEL_DIR`, `DFLASH_DRAFT_REPO`, `TARGET_REPO`,
`ENGINE_ROOT`.

## Using it through halofpx

```bash
# 1. one-shot: pull the main quant AND build the drafter
halofpx pull deepseek-v4-flash --with-draft

# 2. load — the registry run_config for deepseek-v4-flash is
#    spec_type: draft-dspark, draft_n: 5, draft_p: 0.0, backend: ROCm0
halofpx load deepseek-v4-flash
# or explicitly:
halofpx load deepseek-v4-flash --device ROCm0 --draft-n 5 --draft-p 0 --spec-type draft-dspark

# overrides
halofpx run deepseek-v4-flash --slots 1 --ctx 4096 --device ROCm0
halofpx load deepseek-v4-flash --spec-type none          # force bare decode
```

### Important: DSpark is HIP-only in this engine pin

The dflash draft decode reuses the target model's `output.weight` and fused dflash HC ops, which
exist only on the **HIP** backend. On `Vulkan0` the engine aborts at graph build:

```
ggml-backend.cpp:941: pre-allocated tensor (output.weight) in a buffer (Vulkan0)
                     that cannot run the operation (NONE)
```

`engine_manager` handles this automatically: when a `draft-dspark`/`draft-dflash` spec is
configured the device is switched Vulkan0 → ROCm0 (with a notice), and if no HIP device exists at
all speculation is disabled with a warning and the model runs bare. Bare (non-spec) decode on
Vulkan stays fully supported.

## Expected numbers (measured 2026-10-11, ROCm 7.2.4, gfx1151)

| Config | Decode | Draft acceptance |
|---|---|---|
| IQ2_XXS bare (Vulkan0) | 22.50 tok/s | — |
| IQ2_XXS + DSpark (ROCm0, ctx 4096, temp 0 pulse) | 15.77 tok/s | 39.5% (mean 2.88/5) |
| **Route A** ROCmFP2-STRIX (102 GiB) + DSpark (ROCm0) | **32.00 tok/s** *(reference, Projected)* | n/a |

The **32.00 tok/s** figure is a reference from the `ds4-strix-halo` QA docs (Route A quant +
DSpark). Reach it by pulling the Route A variant (see below); **no Route A number may be published
from this repo without a measured run or a `(Projected)` label** (repo rule).

## Route A (the ~95 GiB ROCmFPX quant)

Route A uses the **`deepseek4`**-arch main model (fully supported by the engine) + the same DSpark
drafter:

```bash
halofpx pull deepseek-v4-flash --variant ROCMFP2_STRIX --with-draft    # 95.3 GiB
halofpx load deepseek-v4-flash --variant ROCMFP2_STRIX
```

`ROCMFP2_STRIX` is legacy affine fp2 (type 107) and fp3 (type 104) as published, so the engine
cannot load it until it is transcoded in place to canonical S40 (type 111) with
`scripts/transcode_legacy_rocmfp2.py` - that step is lossy. The sibling
`ROCMFPX_MIX_STRIX` file is not usable at all (expert types 105/106 with in-file codebook
sidecars); see `docs/BENCHMARKS.md` section 10.

RAM planning on 128 GB: ~95 GiB main + ~10.9 GiB drafter + KV cache. If it won't fit, reduce
`--ctx` (KV is the growth knob) or use `--no-mmap`. Measured Route A numbers are in
`docs/BENCHMARKS.md` section 9; the 32 tok/s figure there is a reference value, not a
reproduced measurement on this host.

## Provenance

- Drafter: converted from `deepseek-ai/DeepSeek-V4-Flash-DSpark` (`mtp.*` head) via the engine's
  `convert_hf_to_gguf.py --dspark`; recorded in `registry/models.json` → `deepseek-v4-flash.draft`.
- IQ2_XXS: `julianmb/DeepSeek-V4-Flash-0731-IQ2XXS-STRIX` (IQ2_XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8).
- Route A: `Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3` (upstream ROCmFP2/ROCmFPX-MIX STRIX block
  quants), `source` recorded per variant.