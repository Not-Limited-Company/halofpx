# Comprehensive Benchmark Matrix — AMD Strix Halo

All benchmarks measured directly on **AMD Ryzen AI Max+ 395 (40 CU Radeon 8060S @ 2.9 GHz, 128 GB 256-bit LPDDR5X, Linux 7.0, Mesa 26.0 RADV)**.

---

## 1. Multi-Model Zoo Performance Matrix

| Model Identifier | Architecture / Parameters | Quantization Preset | BPW | Model Size | Unassisted Decode *(Measured)* | MTP Speculative Decode *(Measured)* | Draft Acceptance Rate |
|---|---|---|---|---|---|---|---|
| **`qwen38-27b`** | Dense / 27.3B | **`ROCmFP4_FAST`** | **4.26** | **13.55 GiB** | **14.02 tok/s** | 🔥 **30.56 – 36.04 tok/s** | **82.6%** |
| **`qwen38-27b`** | Dense / 27.3B | **`ROCmFP8`** | **8.25** | **26.25 GiB** | **7.66 tok/s** | **18.96 tok/s** | **93.7%** |
| **`qwen38-27b`** | Dense / 27.3B | **`Q3_K_S`** | **3.59** | **11.40 GiB** | **16.69 tok/s** | **20.44 – 26.11 tok/s** | **73.5%** |
| **`nemotron-3.5-30b`** | MoE / 30.0B (3B active) | **`ROCmFP4_FAST`** | **4.25** | **14.80 GiB** | **52.40 tok/s** | 🔥 **84.50 – 95.20 tok/s** | **88.2%** |
| **`nemotron-3.5-30b`** | MoE / 30.0B (3B active) | **`UD_Q4_K_XL`** | **4.85** | **17.10 GiB** | **46.80 tok/s** | **78.20 tok/s** | **84.1%** |
| **`ornith-1.5-35b`** | MoE / 34.8B (3B active) | **`ROCmFP4`** (aug-24 MTP refresh) | **4.29** | **18.16 GiB** | **76.9 tok/s** | ✅ **105.6 t/s** (n4 / p0.6) | **87.98%** (mean 3.70/4) |
| **`ornith-1.5-35b`** | MoE / 34.8B (3B active) | **`Q4_K_M`** (baseline) | **4.85** | **21.80 GiB** | **71.5 – 71.7 tok/s** | n/a | n/a |
| **`ornith-35b`** | Dense / 35.0B | **`ROCmFPX_Speed`** | **4.15** | **19.20 GiB** | **11.20 tok/s** | **115.0+ tok/s (16 Slots)** | **N/A (Multi-Slot)** |
| **`qwen38-flash-next`** | MoE / 125B (6B active + 51B PLE) | **`UD-IQ1_S`** | **1.56** | **67.55 GiB** | **27.3 tok/s** *(bring-up)* | n/a | n/a |
| **`deepseek-v4-flash`** | MoE / 284B (16B active) | **`IQ2_XXS`** / **`ROCMFP2_STRIX`** (S40) | **2.06 / 2.88** | **80.77 / 95.29 GiB** | **22.50 tok/s** *(Vulkan, measured)* | **15.77 tok/s** *(IQ2+DSpark, ROCm0, measured)* / **13.5 tok/s** *(ROCMFP2_STRIX+DSpark, ROCm0, ctx8192, measured)* | **39.5%** *(IQ2+DSpark)* / **25.5%** *(ROCMFP2_STRIX+DSpark)* |
| **`ling3-flash`** | MoE / 124B (5.1B active) | **`Q4_K_M`** | **4.85** | **71.72 GiB** | **45.23 tok/s** | **55.1 tok/s** (n2 / p0.6) | **94.3%** |
| **`nex-n2.5-mini`** | MoE / 34.7B (3B active) | **`ROCmFP4_STRIX_LEAN`** | **4.29** | **17.32 GiB** | **76.92 tok/s** *(Vulkan)* / **68.62 tok/s** *(ROCm)* | n/a | n/a |
| **`nex-n2.5-mini`** | MoE / 34.7B (3B active) | **`Q4_K_M`** (baseline) | **4.88** | **19.71 GiB** | **72.76 tok/s** *(Vulkan)* / **60.39 tok/s** *(ROCm)* | n/a | n/a |

---

## 2. Qwen 3.8 27B Context Scaling Benchmark

Measured with FlashAttention and Asymmetric TurboQuant KV Cache (`-ctk q8_0 -ctv turbo4`):

| Context Depth | KV Cache RAM | Prefill Speed (`pp`) *(Measured)* | TTFT (Prompt Eval) | Raw Decode (`tg`) *(Measured)* | MTP Speculative Decode *(Measured)* |
|---|---|---|---|---|---|
| **512 tokens** | **0.04 GiB** | **382.21 tok/s** | 1.34 s | **14.06 tok/s** | 🔥 **34.82 – 36.04 tok/s** |
| **2,048 tokens** | **0.15 GiB** | **356.85 tok/s** | 5.74 s | **14.04 tok/s** | **32.40 – 34.82 tok/s** |
| **4,096 tokens** | **0.31 GiB** | **339.73 tok/s** | 12.05 s | **14.01 tok/s** | **30.56 – 32.24 tok/s** |
| **8,192 tokens** | **0.62 GiB** | **311.76 tok/s** | 26.27 s | **13.98 tok/s** | **29.73 tok/s** |
| **16,384 tokens** | **1.23 GiB** | **266.57 tok/s** *(Vulkan)*<br>**329.86 tok/s** *(ROCm)* | 49.66 s | **13.85 tok/s** | **28.02 tok/s** |
| **32,768 tokens** | **2.45 GiB** | **~245.0 tok/s** | ~130 s | **13.62 tok/s** | **26.85 tok/s** |

---

## 3. Context Scaling RAM Consumption

Thanks to Qwen 3.8's **hybrid linear-attention layers** (48 linear-attention + 16 full-attention layers):

| Context Window | Model Weights | Standard FP16 KV Cache | Asymmetric TurboQuant KV Cache | Total RAM Footprint |
|---|---|---|---|---|
| **8K tokens** | 13.55 GiB | 1.88 GiB | **0.62 GiB** | **14.17 GiB** |
| **32K tokens** | 13.55 GiB | 7.50 GiB | **2.45 GiB** | **16.00 GiB** |
| **64K tokens** | 13.55 GiB | 15.00 GiB | **4.90 GiB** | **18.45 GiB** |
| **128K tokens** | 13.55 GiB | 30.00 GiB | **9.80 GiB** | **23.35 GiB** |
| **262K tokens (Max)** | 13.55 GiB | 61.44 GiB | **20.08 GiB** | **33.63 GiB** |

*On a 64GB Strix Halo system, 32K context leaves **~48 GiB free** for desktop apps and compilation.*

---

## 4. Multi-Slot Concurrency & MTP Tuning

MTP speculative decoding performance on AMD Strix Halo depends on the workload structure (single-user interactive vs parallel agent multi-turn slots):

### Workload-Specific MTP Profiles

| Workload Type | Optimal MTP Profile | Recommended CLI Args | Measured Single-Slot TPS | Measured Aggregate TPS |
|---|---|---|---|---|
| **Single-User Sustained Decode (Sweet Spot)** | `n4 / p0.0` | `--slots 1 --draft-n 4 --draft-p 0.0` | 🔥 **33.80 tok/s sustained** (2.40× over baseline) | **33.80 tok/s** |
| **Single-User Interactive Chat (Burst)** | `n5 / p0.50` | `--slots 1 --draft-n 5 --draft-p 0.50` | 🔥 **28.59 – 36.04 tok/s** | **28.59 – 36.04 tok/s** |
| **Parallel Multi-Agent Slots (4-Way)** | `n6 / p0.60` | `--slots 4 --draft-n 6 --draft-p 0.60` | **12.4 – 16.7 tok/s / slot** | 🔥 **23.15 (sustained) – 40.50 (burst) tok/s** |
| **Batch High-Throughput (16-Way)** | `n4 / p0.65` | `--slots 16 --draft-n 4 --draft-p 0.65` | **~7.0 tok/s / slot** | 🔥 **115.0+ aggregate tok/s** *(Ornith 35B)* |

> 💡 **MTP Depth (`K`) Scaling Insight:** Empirical sweeps show `K=4` is the optimal single-stream sweet spot on Strix Halo (`33.8 tok/s`). `K=6` regresses slightly due to memory bus saturation on a single stream, while `K=8` causes severe rollback degradation (`18.2 tok/s`). For 4-slot parallel concurrency, `K=6 / p0.60` maintains higher shared-slot throughput.

### Ornith-1.5-35B (Qwen3.5MoE / hybrid linear attention) — MTP is a net loss

Measured on `ROCmFP4` (Vulkan0, gfx1151), same 192-token prompt:

| Config | Decode | Draft acceptance |
|---|---|---|
| **bare (no MTP)** | **76.0 tok/s** | — |
| K2 p0.5 / p0.75 | 72.5–73.5 tok/s | ~68% |
| K4 p0.5 | 72.6 tok/s | ~68% |
| K4 p0.0 (greedy) | 35–50 tok/s | **15.9%** |
| K4 p0.0 `--spec-mtp-strict-qwen` | 31.5–46.6 tok/s | 19.9% |

- The embedded 1-layer MTP head drafts greedily but accepts only ~16% of tokens on the hybrid `qwen35moe` architecture, so the draft+verify overhead exceeds any gain. Even at relaxed `p-min` (68% acceptance) it never beats bare decode.
- `--spec-mtp-strict-qwen` (boundary-safe multi-row verification with bounded recurrent rollback for Qwen35/Qwen35MoE) improves greedy acceptance from 15.9% → 19.9% but still loses to bare.
- **Recommendation (updated 2026-08-28):** the official aug-24 MTP refresh fixed the ornith drafter — 87.98% acceptance (mean 3.70/4), **105.6 tok/s effective (+38% vs bare)** on gfx1151. `mtp_enabled: true` (n4 / p0.6) is now the default in the registry. requires the refreshed quant (sha256 0f907917…); older quants keep the weak drafter.

### 4-Slot Parallel Stability & Thermal Soak (Community Validated)
- **Configuration:** 4 concurrent slots × 131,072 context tokens each (~524K total context tokens in unified memory via TurboQuant KV).
- **Thermal Stability:** 31.2-minute continuous thermal soak test on Ryzen AI Max+ 395 peaked at **71.88°C** with zero GPU resets, zero swap growth, and zero OOM events.
- **Credit:** Verified independently by community researchers *MrWidmoreHK* and *kujetic*.

---

## 5. Quantization Fidelity & Long-Context Validation (Ornith-1.5-35B ROCmFP4)

Measured on the shipped `Ornith-1.5-35B-A3B-ROCmFP4.gguf` (gfx1151 ROCmFP4 build, commit `12f8b7e`), ROCm0 backend, cache mode (`-ctk q8_0 -ctv q8_0`, MTP disabled).

### 5.1 Perplexity vs Q4_K_M Baseline

`llama-perplexity` over `wikitext-2-raw` (validation split, 9 chunks, `n_ctx=512`, `batch=512`), ROCm0:

| Quantization | Perplexity (↓ better) |
|---|---|
| **ROCmFP4** (shipped) | **5.95 ± 0.31** |
| **Q4_K_M** (baseline) | **5.64 ± 0.29** |

→ ROCmFP4 lands within **~5.5%** of the well-tuned Q4_K_M — a small, expected trade for a speed-oriented 4-bit format (see §1: +7.5% decode over Q4_K_M on this model). Fidelity is competitive.

### 5.2 Context-Window Validation

| Context | Load | Decode (tok/s) | Prefill (tok/s) | Notes |
|---|---|---|---|---|
| **131,072** | ✅ clean | **58.6** | **96.6** | cache mode active |
| **262,144** (full train capacity) | ✅ clean, no warnings | **58.5** | **140.0** | `n_ctx_seq == n_ctx_train`; clean load |

- Both validated with the production `engine_manager` flag set (`-ctxcp 16 -cpent 4096 -cram 32768`, `--kv-unified`, `--cont-batching`).
- **Known limitation:** `cache_reuse` (KV-shift reuse) auto-disables on this hybrid SSM-state context — prompt-cache *RAM checkpoints* remain active, so resume/reuse still works; only fine-grained shifting is off.
- Multimodal path also verified: `mmproj-Ornith-1.5-35B-BF16.gguf` loads and the server answers image prompts correctly.

---

## 6. Qwen 3.8 Flash Next (`qwen4exp`) Bring-Up Benchmark

Measured directly on AMD Strix Halo (`gfx1151`, Mesa RADV Wave64) using `port-qwen4exp` (PR #98):

- **Model:** `Qwen3.8-Flash-Next-UD-IQ1_S` (67.55 GiB GGUF)
- **Engine Configuration:** `-dev Vulkan0 -ngl 99 -fa on -ub 512`
- **Measured Metrics:**
  - `pp512`: **381.73 ± 2.25 tok/s**
  - `pp4096`: **352.22 ± 3.93 tok/s**
  - `tg128`: **25.71 – 27.30 tok/s** peak decode throughput

---

## 7. Ling 3.0 Flash (`bailingmoe3`) Bring-Up Benchmark

Measured directly on AMD Strix Halo (`gfx1151`, Mesa RADV Wave64) using ROCmFPX `build-rocmfpx` (11475/08213ad5b); ROCm0 leg via q38rocm v1.7.0 prebuilt (org lineage 75e67a92b — build skew noted):

- **Model:** `Ling-3.0-flash-Q4_K_M` (71.72 GiB GGUF, 2 shards)
- **Engine Configuration:** `--jinja -dev Vulkan0 -ngl 99 -c 8192 -fa off -t 16`
- **Measured Metrics:**
  - `pp512` (Vulkan0): **372.68 ± 2.67 tok/s**
  - `tg128` (Vulkan0): **45.23 ± 0.09 tok/s**
  - `pp512` (ROCm0, v1.7.0 prebuilt): **391.80 ± 5.97 tok/s**
  - `tg128` (ROCm0, v1.7.0 prebuilt): **35.96 ± 0.09 tok/s**
  - MTP sweep 2026-09-06 (11 configs, thinking-off short prompts): **n2/p0.6 wins at 55.1 tok/s avg, 94.3% acceptance**; n3/p0.6 51.6, n4/p0.6 45.0, n5 collapses to 32.6, n6 recovers to 51.0; p=0.6 dominates p=0.0 at every depth (n4/p0.0 even regresses below baseline 41.5). Initial probe: 21/23 (91%) then 83/105 (79%) drafts accepted, ~51 tok/s; `-ub` sweep 2026-09-06 (512/1024/2048, Vulkan0 pp512+tg128): identical ~366/~45.0 — Reddit `-ub 2048` advice does not transfer to our Vulkan path
  - `wikitext-2` perplexity: **4.6064 ± 0.0275** (597 chunks)
  - Context scaling: 32K/64K/128K all load clean (18/24/24 s); VSZ 76.7/77.2/82.1 GiB (+5.4 GiB for 4× ctx — MLA/KDA flat)
- **Requirements:** `-fa off` (probed 2026-09-06: `-fa on` keeps Paris correct but loops math twice — `flash_attn: False` ships), bounded `-c 8192` (GGUF default 262K hangs the box); thinking via `chat_template_kwargs.enable_thinking`; card sampling (`temp 0.6 / top_p 0.95 / top_k 20`) needs generous `max_tokens` (~500 thinking tokens before content)
- **ROCmFP4 verdict: PARKED** — `Q4_0_ROCMFP4_FAST` (63.27 GiB, 4.26 bpw): PPL 7.04 vs 4.61 + incoherent generation; `Q4_0_ROCMFP4_STRIX_LEAN` (63.35 GiB): repetitive garbage. DIAG control (`Q4_K_M` → FAST, 63.27 GiB) fails identically (Chinese loops for math) — breakage is source-independent, not the Q8_0 band path. All gated on fresh servers, load logs clean. NOT published. `Q4_K_M` remains the serving quant. Re-run DIAG2 2026-09-06 (`Q4_K_M` → FAST, byte-identical 67934438080 B, 547 tensors, zero quantizer warnings) fails identically — deterministic, reproducible.

---

## 8. Nex N2.5 Mini (`qwen35moe`) Bring-Up Benchmark

Measured directly on AMD Strix Halo (`gfx1151`, Mesa RADV Wave64 / ROCm 7.2.3) using ROCmFPX `build-strix-rocmfp4` (commit `334f10d81` / build `11475`):

- **Model:** `Nex-N2.5-mini` (34.66B total parameters, ~3.0B active per token, 256 routed experts + 1 shared expert, hybrid 3× Gated DeltaNet linear attention + 1× Full Attention per group)
- **Quantization:**
  - `ROCmFP4_STRIX_LEAN` (17.32 GiB, 4.29 bpw): Quantized via `llama-quantize Q4_0_ROCMFP4_STRIX_LEAN 16` with `--allow-requantize` on 2026-09-09. All 733 tensors converted cleanly with FP32 router/norms and Q5_K embeddings.
  - `Q4_K_M` (19.71 GiB, 4.88 bpw): Baseline quant from `abenzerps/Nex-N2.5-mini-GGUF`.
  - `mmproj-Nex-N2.5-mini.gguf` (0.84 GiB): ViT multimodal vision projector.
- **Measured Metrics:**
  - **`ROCmFP4_STRIX_LEAN`**:
    - `pp512` (ROCm0): **1,028.18 ± 185.75 tok/s**
    - `tg128` (ROCm0): **68.62 ± 0.01 tok/s**
    - `pp512` (Vulkan0): **642.37 ± 0.39 tok/s**
    - `tg128` (Vulkan0): **76.92 ± 0.21 tok/s** *(Fastest decode)*
  - **`Q4_K_M`**:
    - `pp512` (ROCm0): **907.44 ± 178.92 tok/s**
    - `tg128` (ROCm0): **60.39 ± 0.35 tok/s**
    - `pp512` (Vulkan0): **1,083.91 ± 2.28 tok/s**
    - `tg128` (Vulkan0): **72.76 ± 0.15 tok/s**
- **Speedup & Verification:**
  - `ROCmFP4_STRIX_LEAN` provides **+5.7% decode speedup** over `Q4_K_M` on Vulkan0 (76.92 vs 72.76 tok/s) and **+13.6% decode speedup** on ROCm0 (68.62 vs 60.39 tok/s), while reducing model footprint from 19.71 GiB $\to$ 17.32 GiB (−12.1% size reduction).
  - Coherence QA passed: Exact arithmetic (`17 * 23 = 391`), structured python generation (`is_palindrome`), and Agentic Thinking adaptive reasoning verified without looping or degradation.
  - Multimodal vision verified: Loads in `llama-server` alongside `--mmproj` in 3.1 seconds.
- **Why It Outperforms Stock Q4_K_M on Strix Halo:**
  1. *Cooperative Matrix Mapping:* `ROCmFP4` single-scale blocks unpack with single-cycle bit shifts in shader registers, avoiding the multi-pass unpack and 6-bit scale/offset ALU overhead of standard `Q4_K` blocks when driving Mesa RADV Wave64 cooperative matrices.
  2. *UMA Traffic Reduction:* 12.1% fewer bytes (17.32 vs 19.71 GiB) streamed across the 256-bit LPDDR5X bus on each step directly relieves memory contention during MoE expert routing.
  3. *Strix Lean Mixed Precision:* Preserves critical router gate tensors in uncompressed FP32 and token embeddings in Q5_K, delivering higher TPS without expert routing degradation.

---

## 9. DeepSeek V4 Flash — DSpark Speculative Decode (`draft-dspark`)

Measured 2026-10-11 on AMD Strix Halo (gfx1151, ROCm 7.2.4 HIP backend) with the pinned ROCmFPX engine (`75e67a92b`):

- **Drafter:** `DeepSeek-V4-Flash-0731-DSpark-draft-bf16.gguf` (10.9 GiB) — native `dflash` arch converted from the official `deepseek-ai/DeepSeek-V4-Flash-DSpark` checkpoint (only the `mtp.*`-containing shards are downloaded) via `convert_hf_to_gguf.py --dspark`. 3 blocks, `block_size=5`, `markov_rank=256`, `target_layers=[40,41,42]`, mask/noise token `128799`. BF16 head with lossless MXFP4 expert repack. Build anytime with `scripts/build_dspark_drafter.sh --drafter-qtype <any>` (default `bf16`).
- **Main model:** `DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf` (80.8 GiB).
- **Engine flags:** `--spec-type draft-dspark -md <drafter> --spec-draft-n-max 5 --spec-draft-p-min 0.0 -dev ROCm0`.

### Measured (ROCMFP2_STRIX + DSpark, ROCm0 — transcoded to canonical S40 type 111)

| Metric | Value |
|---|---|
| Decode (24-token pulse, ctx 8192, temp 0) | **14.35 tok/s** |
| Decode (64-token pulse, ctx 8192, temp 0) | **13.47 tok/s** |
| Draft acceptance | **25.5%** (mean over 3 runs) |
| Mean draft length | **2.00 tokens/step** |
| Load | ✅ coherent (fact/math/code correct) |

- **File:** `DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf` (95.3 GiB) transcoded from legacy affine (type 107) to canonical dual-scale S40 (type 111) in-place using `scripts/transcode_legacy_rocmfp2.py` (threaded fast path). Audit after transcode: 0 legacy tensors, 86 type 111 tensors; end-to-end fidelity against the independent IQ2_XXS reference matches the predicted S40-vs-IQ2 correlation.
- **Backend:** HIP (ROCm0). DSpark draft decode requires HIP in this engine pin.

### The advertised 32.00 tok/s — provenance

The **32.00 tok/s** figure shipped with the README is **not** measured with the IQ2_XXS quant alone (IQ2_XXS + DSpark measured 13–16 tok/s in our bench; unassisted decode measures 22.50 tok/s on Vulkan in other configurations). 32.00 tok/s was a **reference figure, not yet reproduced on this host**; the measured Route A (ROCMFP2_STRIX + DSpark) is **~13.5 tok/s** at ctx 8192 (context-dependent), labeled here as measured rather than projected. Until a ctx-optimized run reproduces the 32 tok/s target under identical hardware/flags, treat any higher number as a specific measured claim with exact configuration.

- Route A variants: `ROCMFP2_STRIX` (95.3 GiB) from `Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3`, provenance recorded per variant (transcoded to canonical S40 on this host). The sibling `ROCMFPX_MIX_STRIX` from the same repo is **not loadable** by this engine and is no longer offered in the registry - see section 10.
- To reproduce: `halofpx pull deepseek-v4-flash --variant ROCMFP2_STRIX --with-draft`, transcode if needed (the transcoded file is now canonical type 111), then `halofpx load deepseek-v4-flash --variant ROCMFP2_STRIX` (defaults to ROCm0). RAM plan: ~102 GiB main + ~10.9 GiB drafter + KV — tight on 128 GB; drop `--ctx` or use `--mmap` (APU default differs) if needed.
- IQ2_XXS + DSpark does **not** reach 32 tok/s on this hardware under these flags (measured ~13.5 tok/s). Do not swap a projected value for a measured one without recording the exact context window, n_predict, temperature, and acceptances for each run.

---

## 10. DeepSeek V4 Flash - variant audit and vision decision (2026-10-12)

Audit of the DeepSeek V4 Flash artifacts offered by this zoo, done while
chasing a published vision build that would not decode. Every number below comes
from parsing the files themselves (GGUF header, KV metadata, payload statistics),
not from inference.

### 10.1 Lucebox MIX files (expert types 105/106) - not loadable

`DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf` and
`DeepSeek-V4-Flash-Vision-Exp-ROCMFPX-MIX-STRIX.gguf` store their routed experts
as GGUF types **105** (3-bit, 14 bytes per 32 weights) and **106** (2-bit,
10 bytes per 32 weights). In this engine those ids are `TURBO3_0`/`TURBO4_0`
(KV-cache types), so the expert tensors cannot even be sized as weights.

The block geometry matches this engine's `Q3_0_ROCMFPX` (104) and
`Q2_0_ROCMFPX` (111) exactly, but the payloads do not. Decoding a real 2 MiB
slice of `blk.0.ffn_gate_exps.weight`:

| Decode applied to the type-106 bytes | rms | mean abs DC |
|---|---|---|
| S40 `{-4,-1,1,4}`, per-half scale (type 111) | 0.0716 | 0.0375 |
| affine `code*e0 - e1` (legacy 107) | 0.0611 | 0.1368 |
| two-scale `(code-1)*e`, per-half | 0.0331 | 0.2801 |
| reference: same tensor, same base, affine 107 | **0.0214** | **0.0110** |

The reference is exact: layer 0 lies outside the abliterated blocks (10-42), so
that tensor holds unmodified weight. Single-scale, swapped-half, scales-first
byte order and level-set variants were tried as well; none produce a healthy
decode.

The file's own metadata explains why no fixed codebook fits:
`deepseek4.p4mix.sidecar` (`P4MIXv1`, 28 entries - one per type-105 tensor),
`deepseek4.gumix.sidecar` (`GUMIXs1`, 101 entries - one per type-106 tensor) and
`deepseek4.mix.codebook_repair=bf16-epsilon-v1`. The MIX recipe keeps its
codebooks inside the file (the upstream README says so and requires their own
runtime: `luce_server`, Luce-Org/lucebox PR #722). There is no lossless retag,
and a transcode would have to reproduce the sidecar semantics.

`ROCMFPX_MIX_STRIX` was removed from `registry/models.json` as a result.

### 10.2 Vision builds - deferred

The otheru vision build
(`DeepSeek-V4-Flash-Vision-Exp-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf`) is a
valid artifact: with its real layout, `value = code*ue4m3(e0) - ue4m3(e1)`, it
decodes cleanly (rms 0.0214, mean abs DC 0.0110). This engine has no affine fp2
decoder, though - type 107 is reserved as ambiguous - so serving it needs engine
work. A file-side transcode to S40 is lossy (measured corr 0.967, relative RMSE
25.6% over 419,328 blocks) and was ruled out. The vision lane was dropped on
2026-10-12: the model file and its projector were deleted, and
`VISION_EXP_STRIX_LEAN` was removed from the registry.

### 10.3 Re-verified baseline lane (IQ2_XXS + DSpark, ROCm0, ctx 4096)

Measured 2026-10-12 on AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151), engine
tree `dc227f4ee` with the `deepseek4.cpp` optional-tensor loader patch, fused
DSV4-HC kernels at their upstream defaults. Model `IQ2_XXS` (80.77 GiB) plus the
bf16 DSpark drafter, `--spec-type draft-dspark --spec-draft-n-max 5
--spec-draft-p-min 0.0`, temperature 0. Method: `scripts/benchmark.py`
(4-task suite, 100-200 completion tokens per task).

| Task | Decode | TTFT | Draft acceptance |
|---|---:|---:|---:|
| Binary Search Tree (Python) | 16.72 tok/s | 762.33 ms | 41.6% |
| Math Puzzle (Widget Factory) | 15.98 tok/s | 753.40 ms | 39.4% |
| JSON Entity Extractor | 20.80 tok/s | 817.41 ms | 57.9% |
| UMA vs Discrete Architecture | 18.69 tok/s | 746.32 ms | 49.5% |
| **Average** | **18.05 tok/s** | **769.90 ms** | **47.1%** |

- Coherence: a fact/math probe answered correctly ("17*23 = 391", "Canberra"),
  16.83 tok/s and 43.8% draft acceptance on that request; no benchmark task
  tripped the repetition detector.
- Receipt: `benchmarks/benchmark_20261012_143003.md` and `.json`.
- Reference: the earlier IQ2_XXS + DSpark figure at ctx 4096 in the model matrix
  above is 15.77 tok/s. The re-verified run lands at or above it, so no
  regression is observable on this lane; treat sub-tok/s differences as run
  variance.

---

## 11. DeepSeek V4 Flash - context window, KV precision and accuracy (2026-10-12/13)

Measured on AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, ROCm 7.2.4), IQ2_XXS
(80.77 GiB) plus the bf16 DSpark drafter, backend ROCm0, `-fa on`, 16 threads.
Question set: ds4-eval's 92 hard questions (25 GPQA-Diamond, 25 audited
SuperGPQA, 25 AIME 2025, 17 COMPSEC), graded by ds4fa's own
`ds4-eval --regrade-trace`, i.e. the same grader behind the published
"82/92" and "COMPSEC-17" figures for this model family. Receipts: every table
below cites files under `benchmarks/`.

### 11.1 Memory and decode vs context window (KV q8_0, the default)

| ctx | RAM used | RAM free | GTT used | decode | TTFT |
|---:|---:|---:|---:|---:|---:|
| 4096 | 99.73 GiB | 25.35 GiB | 92.40 GiB | 18.78 tok/s | 530 ms |
| 65536 | 99.97 | 25.12 | 92.61 | 19.28 | 541 ms |
| 131072 | 100.39 | 24.69 | 92.98 | 19.30 | 537 ms |
| 163840 | 100.60 | 24.49 | 93.15 | 19.17 | 542 ms |
| 196608 | 100.77 | 24.32 | 93.26 | 19.67 | 538 ms |
| 262144 | 101.09 | 23.99 | 93.50 | 19.26 | 546 ms |
| 524288 | 102.64 | 22.45 | 94.77 | 19.63 | 575 ms |
| 1048576 | 105.96 | 19.12 | 97.43 | 17.17 | 989 ms |

The model declares `context_length = 1048576` (native 65536, YaRN x16). The whole
window fits with **19 GiB still free**: KV costs about **6.3 KiB/token** (6.33 GiB
from 4K to 1M), because attention uses a single KV head with per-layer
`compress_ratios`. Decode holds ~19 tok/s to 512K and dips slightly at 1M.
Receipt: `benchmarks/ctx_sweep_kv_q8_0_first_pass.json`.

**This is now the shipped default.** `run_config.ctx_size` for
`deepseek-v4-flash` is **1048576** (the model's declared maximum) and
`min_vram_gib` for `IQ2_XXS` is **106**, matching the measured 105.96 GiB
footprint at that window. Measured on the default configuration (4-task suite,
temp 0, ctx 1048576, KV q8_0, DSpark): **17.95 tok/s average decode, 888 ms
average TTFT, 47.1% draft acceptance** - receipt
`benchmarks/benchmark_20261013_213911.json`. The 6.3 GiB of KV that the larger
window reserves is the whole cost of the change; the *practical* bound stays
prefill time (section 11.2), so use 1M for long sessions and prefix reuse rather
than one-shot million-token prompts.

### 11.2 What depth actually costs: prefill, not memory (KV q8_0, cold cache)

| prompt tokens | prefill | prefill rate | decode |
|---:|---:|---:|---:|
| 3,636 | 18.7 s | 194 tok/s | 16.50 tok/s |
| 7,322 | 38.9 s | 188 | 13.80 |
| 14,597 | 82.5 s | 177 | 14.20 |
| 29,244 | 194.0 s | 151 | 14.18 |
| 58,441 | 513.7 s | 114 | 9.11 |
| 116,932 | 1539.8 s | 76 | 10.44 |

Prefill throughput falls by more than half across the range, so a 128K-token
prompt costs ~26 minutes before the first token, and a 1M-token prompt is
impractical on time even though it now fits in memory. Receipt:
`benchmarks/depth_probe_first_pass.json`.

**Prefix reuse is nearly free.** Sending the same long prefix again served it from
the prompt cache: prefill dropped from 139 s to **0.2 s** for an identical
23,179-token prefix (measured in `benchmarks/longctx_kv_q8_0_v2.json`). This is the
checkpoint/prompt-cache machinery working as intended, and it is why repeat
sessions are cheap while first reads of long context are not.

### 11.3 Accuracy: the 92-question suite, q8_0 vs q4_0 KV

| Suite | q8_0 | q4_0 |
|---|---:|---:|
| COMPSEC (17) | **17/17** | **17/17** |
| GPQA Diamond (25) | 22/25 | 21/25 |
| SuperGPQA (25) | 21/25 | 22/25 |
| AIME 2025 (25) | 21/25 | 19/25 |
| **Total (92)** | **81/92 (88.0%)** | **79/92 (85.9%)** |

Runtime flags: ctx 262144, greedy, engine reasoning budget 4096, max output 8192,
DSpark speculation on. Means per case: 2394 generated tokens at 14.73 tok/s
(q8_0), 2349 at 14.67 tok/s (q4_0).

Reading:
- The 2-case gap is inside noise, and q4_0 hit the output cap 7 times against
  q8_0's 3; COMPSEC is identical at 17/17 for both.
- 3 of the q8_0 failures are truncated at the 8192 cap (two AIME answers were cut
  mid-number), so 81/92 is a floor for that configuration, not a ceiling.
- For scale: the published figure for a larger 2.766 bpw 0731 ROCMFPX-MIX build
  is 82/92 on the same suite. Different artifact, different engine and a larger
  reasoning allowance, so treat the comparison as orientation only.

Receipts: `benchmarks/ds4_eval_kv_q8_0_20261012_152703.*` and
`benchmarks/ds4_eval_kv_q4_0_20261013_135126.*` (traces included, so any verdict
can be re-graded).

### 11.4 Long-context retrieval, q8_0 vs q4_0 KV

Needle planted at 25/50/75% depth in varied filler (a repeated paragraph makes
retrieval unfairly hard), asked for by name; reasoning budget bounded to 512
tokens and output capped at 1024 so that "no answer emitted" cannot be mistaken
for "not retrieved".

| prompt tokens | needle at | q8_0 | q4_0 |
|---:|---:|:--:|:--:|
| 23,179 | 25% | hit | hit |
| 23,179 | 50% | hit | hit |
| 23,179 | 75% | miss (refusal) | miss (refusal) |
| 46,469 | 25% | hit | hit |
| 46,469 | 50% | miss (rambled to cap) | miss (rambled to cap) |
| 46,469 | 75% | hit | hit |
| 92,636 | 25% | miss (hallucinated) | miss (hallucinated) |
| 92,636 | 50% | miss (to cap) | hit |
| 92,636 | 75% | hit | miss (to cap) |
| **Total** | | **5/9** | **5/9** |

- KV precision changes nothing here: identical totals, and the two cells that
  differ are the cap-limited ones.
- The failures are the interesting part: at 23K the model *refuses* ("I'm sorry,
  but the background notes you..."), and at 93K it once **confabulates** a
  passphrase ("The passphrase is: the zone kernel of..."). Those are real
  long-context failure modes, not measurement artifacts.
- At 92K the model frequently fails to finish an answer at all: with the deployed
  4096 reasoning budget and a 1024-token output cap, all three tries produced no
  final answer in the first retrieval pass. Quote a deep-context result only with
  the reasoning allowance stated.

Receipts: `benchmarks/longctx_kv_q8_0_rb512.*`, `benchmarks/longctx_kv_q4_0.*`.

### 11.5 Fidelity: continuation NLL and perplexity

**Continuation NLL** (ds4fa's `score_llama`: target-token negative log-likelihood
of DeepSeek's own tracked continuations; 100 cases x 24 target tokens; engine
default KV):

| Metric | Value |
|---|---|
| Mean average NLL | **0.6238** |
| Scorer summary | avg_nll 0.6305, first-match 34/100, avg common prefix 2.21 tokens |
| Wall | 245 s including model load |

Two caveats bound this number: the fixture's hosted-API logprobs are all `0.0`, so
it is a *relative* metric for comparing our own configurations, not a comparison
against the hosted model; and `score_llama` has no KV-type option, so it always
scores at the engine default (q8_0). The KV comparison therefore rides on
perplexity below.

**Perplexity** (wikitext-2 test, 20 chunks x 512 tokens, single sequence):

| KV | PPL |
|---|---:|
| q8_0 | **7.46 +/- 0.33** |
| q4_0 | **7.36 +/- 0.33** |

Getting a usable value needs `-b` equal to `-c`. With the default batch (2048)
llama-perplexity decodes `n_seq=4` chunks in parallel and reports **PPL 2792**,
which is nonsense for this architecture; with `-b 512 --ub 512` (`n_seq=1`) the
same 20 chunks give 7.46. Treat multi-sequence perplexity for `deepseek4` in this
fork as broken until fixed. Receipts: `benchmarks/nll_kv_q8_0_20261013_211750.*`,
`benchmarks/ppl_kv_q8_0_20261013_212611.json` and
`benchmarks/ppl_kv_q4_0_20261013_212809.json` (single-sequence runs),
`benchmarks/ppl_kv_q8_0_20261013_120847.json` (the default-batch anomaly).

### 11.6 KV precision: the verdict

| | q8_0 | q4_0 |
|---|---|---|
| RAM at 1M ctx | 105.96 GiB | 105.02 GiB (**-0.9 GiB**) |
| Decode, short generation | 18.8-19.7 tok/s | 16.5 tok/s (**-13%**) |
| Decode, long generation (accuracy run) | 14.73 tok/s | 14.67 tok/s |
| Accuracy (92 suite) | 81/92 | 79/92 |
| Retrieval | 5/9 | 5/9 |

q4_0 KV buys well under 1 GiB at the full window and costs decode throughput on
short generations, with no measurable accuracy or retrieval benefit. **q8_0 stays
the default.** A per-load override exists (`halofpx load --cache-type-k q4_0
--cache-type-v q4_0`) for anyone who needs the last gigabyte.

### 11.7 Request timeouts and client-side cancellation (verified)

| Check | Result |
|---|---|
| Request longer than the old 300 s cap, via the router | **502.3 s, HTTP 200** (64,002-token prompt, 493.9 s prefill) |
| Streaming keepalive during a long prefill | first SSE event at **30.1 s** (the engine ping) |
| Client disconnect mid-prefill | next request served in **9.9 s** vs 0.8 s idle, i.e. generation stopped (~10 s unwind) |

The 300 s cap was the router's own hard-coded timeout, not the engine's. Requests
are now unbounded by default (`HALOFPX_REQUEST_TIMEOUT`), with a 10 s connect
timeout so a dead engine still fails fast, and `-to 86400` as an engine-side
safety net. Receipts: `benchmarks/timeout_verify_*.json`.
