#!/usr/bin/env python3
"""
transcode_legacy_rocmfp2.py — convert legacy *affine* ROCmFP2 GGUF tensors
(type 107) in place to canonical dual-scale S40 ROCmFP2 (type 111).

Why this is needed
------------------
GGUF type 107 (`Q2_0_ROCMFPX_LEGACY_AMBIGUOUS`) was emitted with two
incompatible 10-byte-per-32-weight layouts, so current engines reject the type
outright and the reference retag tool refuses to guess:

    S40 dual-scale : w = {-4,-1,+1,+4}[code] * ue4m3(e_half)   -> now type 111
    legacy affine  : w = (code - 1) * ue4m3(e_half)            -> unimplemented

The Lucebox `DeepSeek-V4-Flash-ROCMFP2-STRIX` family uses the affine layout.
Evidence from `blk.*.ffn_{gate,up}_exps` (sign agreement against an independent
IQ2_XXS dequantisation of the *same* tensor):

    control  type 101 Q4_0_ROCMFP4_FAST ...... 0.999
    control  type 104 Q3_0 (canonical) ....... 0.956
    S40      MORD {-4,-1,+1,+4} .............. 0.762   <- wrong
    affine   (code-1)*ue4m3(e_half) .......... 0.995   <- right

The two only disagree at code 1 — the most common code (~42% of weights), which
S40 maps to -1*S while the stored data means ~0.  No permutation of the S40
codebook contains a zero level, so a header-only retag to type 111 produces a
file that loads but decodes to noise; the payload must be decoded and
re-encoded.  This tool does that at identical size (10 bytes / 32 weights on
both sides), so the file length never changes.

Quality cost
------------
A 4-level -> 4-level requantisation.  Measured on `blk.0.ffn_gate_exps.weight`
against the independent reference: correlation 0.865 -> 0.823, weight RMSE
+14%.  The loss is intrinsic: S40 has no zero level, so the ~42% of weights the
affine layout stored as exactly 0 cannot be reproduced exactly.

Safety
------
The rewrite is in place (a full side file would need another ~100 GB), so it is
made crash-safe by construction:

  <model>.transcode/<index>.s40   holds the fully transcoded payload for one
  tensor.  It is written to a temp name, fsynced, then atomically renamed
  *before* the payload region is overwritten, and is only removed once the
  tensor's type field has been flipped to 111 and fsynced.

On restart each tensor is therefore either skipped (tag already 111), restored
verbatim from its side file, or retranscoded from untouched bytes.  A torn
write can never be transcoded twice.

Usage
-----
    transcode_legacy_rocmfp2.py audit     <model.gguf>
    transcode_legacy_rocmfp2.py dry-run   <model.gguf>   # validate, change nothing
    transcode_legacy_rocmfp2.py transcode <model.gguf> --yes

Modes:
  audit      read-only summary of type-107 / type-111 expert tensors
  dry-run    transcode the first chunk of each tensor in memory and report
             reconstruction statistics; never writes to the model
  transcode  rewrite the payloads and flip the type fields (requires --yes)

Requires numpy.  The encoder is memory-bandwidth bound, so `--threads`
(default: min(16, cpu_count)) slices every chunk across a thread pool; all of
the numpy ops used release the GIL.  On a 16-core Strix Halo that measures a
~10x speedup (about 200 MB/s of payload) with byte-identical output.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment guard
    sys.stderr.write("error: this tool needs numpy (pip install numpy)\n")
    raise SystemExit(2)

LEGACY = 107                      # Q2_0_ROCMFPX_LEGACY_AMBIGUOUS
S40 = 111                         # Q2_0_ROCMFPX
QK = 32                           # weights per block (both layouts)
BLOCK_BYTES = 10                  # 8 code bytes + 2 per-half UE4M3 scale bytes
SCALE_OFFSET = 8                  # e[0] at byte 8 (weights 0..15), e[1] at 9

# Affine codebook of the legacy layout.  `-1,0,1,2` is the uniform `code*scale
# - offset` grid with offset == scale; the empirical least-squares estimate is
# {-0.967,+0.037,+0.824,+1.903}, statistically indistinguishable from it.
DEFAULT_GRID = "-1,0,1,2"

MORD = np.array([-4.0, -1.0, 1.0, 4.0], dtype=np.float32)   # S40 codebook

# GGUF metadata value-type sizes; 8 = string, 9 = array (both variable).
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY = 8, 9


def _ue4m3_table() -> np.ndarray:
    """UE4M3 decode table, byte-for-byte identical to ggml_ue4m3_to_fp32()."""
    t = np.zeros(256, dtype=np.float32)
    for b in range(0x7F):                     # >0x7e is invalid and decodes to 0
        exp, man = b >> 3, b & 7
        t[b] = (man * 2.0 ** -10) if exp == 0 else (8 + man) * 2.0 ** (exp - 11)
    return t


TABLE = _ue4m3_table()
VALID = np.arange(1, 0x7F, dtype=np.int64)    # legal scale indices 1..126
VALID_VALS = TABLE[VALID]


def _count_lut() -> np.ndarray:
    """byte -> how many of its four 2-bit fields equal each code 0..3."""
    lut = np.zeros((256, 4), dtype=np.uint8)
    for b in range(256):
        for i in range(4):
            lut[b, (b >> (2 * i)) & 3] += 1
    return lut


class Codebook:
    """Exact and fast S40 encoder for a 4-level affine source codebook.

    The affine decode only ever takes four values per half (one per 2-bit code),
    so the 16 weights of a half fall into at most four *magnitude groups*.  The
    engine's `|x| > 2.5*s -> +-4 else +-1` rule can therefore only split between
    groups, never inside one, which collapses the 126-scale search to one
    closed-form candidate per group boundary.  Counting the codes with a
    256-entry LUT and emitting the new ones with another LUT reduces the whole
    requantisation to a handful of gathers.

    `groups` is ordered by descending |codebook value|, which is the order the
    2.5*s threshold walks the weights in.
    """

    def __init__(self, grid: np.ndarray):
        self.grid = np.asarray(grid, dtype=np.float32)
        mags = np.abs(self.grid.astype(np.float64))
        uniq = sorted({round(float(m), 9) for m in mags}, reverse=True)
        self.groups = [np.array([i for i in range(4) if round(float(mags[i]), 9) == m],
                                dtype=np.int64) for m in uniq]
        self.M = np.array(uniq, dtype=np.float32)
        self.r = len(uniq)
        # Split boundaries the threshold can actually produce: t means groups
        # 0..t-1 are "outer".  A zero-magnitude group can never be outer.
        self.ts = list(range(self.r + 1))
        if float(self.M[-1]) == 0.0:
            self.ts.remove(self.r)

        self.count_lut = _count_lut()
        self.pack_lut = np.zeros((len(self.ts), 256), dtype=np.uint8)
        for ti, t in enumerate(self.ts):
            mapping = np.empty(4, dtype=np.uint8)
            for gi, g in enumerate(self.groups):
                outer = gi < t
                for i in g:
                    neg = bool(np.signbit(self.grid[i]))
                    mapping[i] = (0 if neg else 3) if outer else (1 if neg else 2)
            conv = np.zeros(256, dtype=np.uint8)
            for b in range(256):
                v = 0
                for i in range(4):
                    v |= int(mapping[(b >> (2 * i)) & 3]) << (2 * i)
                conv[b] = v
            self.pack_lut[ti] = conv

    def encode_half(self, qs: np.ndarray, e_bytes: np.ndarray):
        """(qs (nb,4) bytes, e (nb,)) -> (new e, new packed code bytes).

        For split `t` (the first t magnitude groups are outer, i.e. decode to
        +-4*sv; the rest inner, +-sv) the exact squared error over one half is

            SSE(sv) = S2 - sv*(8*So + 2*Si) + sv^2*(15*O + 16)

        with S2 = sum(x^2), So/Si = sum of |x| over outer/inner (both = s times
        a count of magnitudes) and O = number of outer weights.  That is a
        convex parabola, so for each split it is enough to test the two table
        values bracketing its minimiser, clipped into the interval where the
        split actually happens.
        """
        nb = qs.shape[0]
        s = TABLE[e_bytes]
        cnt = self.count_lut[qs[:, 0]].copy()
        for j in (1, 2, 3):
            cnt += self.count_lut[qs[:, j]]
        gc = np.empty((nb, self.r), dtype=np.float32)
        for u, g in enumerate(self.groups):
            if len(g) == 1:
                gc[:, u] = cnt[:, g[0]]
            else:
                np.sum(cnt[:, g], axis=1, dtype=np.float32, out=gc[:, u])

        M = self.M
        cum_o = np.cumsum(gc, axis=1)
        cum_a = np.cumsum(gc * M[None, :], axis=1)
        a_all = cum_a[:, -1]
        t2 = s * s * np.sum(gc * (M * M)[None, :], axis=1)

        zero = np.zeros(nb, dtype=np.float32)
        best = np.full(nb, np.inf, dtype=np.float32)
        best_e = np.zeros(nb, dtype=np.uint8)
        best_t = np.zeros(nb, dtype=np.int16)

        for ti, t in enumerate(self.ts):
            if t == 0:
                o = zero
                a_out = zero
            else:
                o = cum_o[:, t - 1]
                a_out = cum_a[:, t - 1]
            a_in = a_all - a_out
            den = 15.0 * o + 16.0                       # 16*outer + inner
            span = s * (8.0 * a_out + 2.0 * a_in)       # the linear SSE term
            s_star = s * (4.0 * a_out + a_in) / den
            lo = (float(M[t]) * s / 2.5) if t < self.r else zero
            hi = (float(M[t - 1]) * s / 2.5) if t > 0 else np.full(nb, np.inf, np.float32)
            near = np.searchsorted(VALID_VALS, np.clip(s_star, lo, hi))
            for idx in (np.clip(near - 1, 0, VALID.size - 1),
                        np.clip(near, 0, VALID.size - 1)):
                cand = VALID_VALS[idx]
                sse = t2 - cand * span
                sse += (cand * cand) * den
                upd = (cand >= lo) & (cand <= hi) & (sse < best)
                best = np.where(upd, sse, best)
                best_e = np.where(upd, (idx + 1).astype(np.uint8), best_e)
                best_t = np.where(upd, np.int16(ti), best_t)

        # a half that decodes to all zeros is stored as e=0 with zero codes
        dead = t2 == 0.0
        out = self.pack_lut[best_t[:, None], qs]
        np.copyto(out, np.uint8(0), where=dead[:, None])
        return np.where(dead, np.uint8(0), best_e), out


# ---------------------------------------------------------------------------
# GGUF header
# ---------------------------------------------------------------------------

def parse_header(path: Path):
    """Return (alignment, data_start, tensors) for a GGUF v3 file."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))

        def u32() -> int:
            return struct.unpack("<I", f.read(4))[0]

        def u64() -> int:
            return struct.unpack("<Q", f.read(8))[0]

        def string() -> bytes:
            return f.read(u64())

        alignment = 32
        for _ in range(n_kv):
            key = string().decode("utf-8", "replace")
            vtype = u32()
            if vtype == _ARRAY:
                etype, count = u32(), u64()
                if etype == _STRING:
                    for _ in range(count):
                        string()
                elif etype == _ARRAY:
                    raise ValueError("nested metadata arrays are not supported")
                else:
                    f.seek(_FIXED[etype] * count, os.SEEK_CUR)
            elif vtype == _STRING:
                raw = string()
                if key == "general.alignment":
                    alignment = int(raw)
            elif vtype in _FIXED:
                raw = f.read(_FIXED[vtype])
                if key == "general.alignment":
                    alignment = int.from_bytes(raw, "little")
            else:
                raise ValueError(f"unknown metadata value type {vtype}")

        tensors = []
        for index in range(n_tensors):
            name = string().decode("utf-8", "replace")
            n_dims = u32()
            dims = [u64() for _ in range(n_dims)]
            type_field = f.tell()
            dtype = u32()
            offset = u64()
            n_elements = int(np.prod(dims, dtype=np.int64)) if dims else 0
            tensors.append({"index": index, "name": name, "dtype": dtype,
                            "dims": dims, "n_elements": n_elements,
                            "offset": offset, "type_field": type_field})
        header_end = f.tell()

    data_start = (header_end + alignment - 1) // alignment * alignment
    return alignment, data_start, tensors


def target_tensors(tensors, dtype: int):
    """Expert tensors carrying `dtype`, with a consistent block layout."""
    out = []
    for t in tensors:
        if t["dtype"] != dtype:
            continue
        n = t["n_elements"]
        if n % QK:
            raise ValueError(f"{t['name']}: element count {n} not a multiple of {QK}")
        expected = n // QK * BLOCK_BYTES
        out.append((t, expected))
    return out


# ---------------------------------------------------------------------------
# core transform
# ---------------------------------------------------------------------------

def _codes_from_bytes(qs: np.ndarray) -> np.ndarray:
    """(nb,4) packed bytes -> (nb,16) 2-bit codes, weight order."""
    nb = qs.shape[0]
    out = np.empty((nb, 16), dtype=np.uint8)
    for j in range(4):
        b = qs[:, j]
        out[:, 4 * j + 0] = b & 3
        out[:, 4 * j + 1] = (b >> 2) & 3
        out[:, 4 * j + 2] = (b >> 4) & 3
        out[:, 4 * j + 3] = (b >> 6) & 3
    return out


def _bytes_from_codes(codes: np.ndarray) -> np.ndarray:
    """(nb,16) codes -> (nb,4) packed bytes (inverse of _codes_from_bytes)."""
    nb = codes.shape[0]
    out = np.empty((nb, 4), dtype=np.uint8)
    c = codes.astype(np.uint16)
    for j in range(4):
        out[:, j] = (c[:, 4 * j] | (c[:, 4 * j + 1] << 2)
                     | (c[:, 4 * j + 2] << 4) | (c[:, 4 * j + 3] << 6)).astype(np.uint8)
    return out


def s40_encode_half(x: np.ndarray):
    """Encode (nb,16) float32 to canonical S40 (e, codes).

    Reproduces the engine's `rocmfpx_quantize_row_fp2_impl` objective exactly:
    choose the UE4M3 half-scale minimising the squared error of the 16 weights
    under the `|x| > 2.5*s -> +-4 else +-1` rule.  The error is piecewise
    quadratic in s with 17 pieces (one per number of "outer" weights), each with
    a closed-form minimum, so we evaluate those minima instead of scanning all
    126 scale indices.
    """
    nb = x.shape[0]
    a = np.sort(np.abs(x), axis=1)[:, ::-1].astype(np.float32)   # descending
    zeros = np.zeros((nb, 1), dtype=np.float32)
    p1 = np.concatenate([zeros, np.cumsum(a, axis=1, dtype=np.float32)], axis=1)
    p2 = np.concatenate([zeros, np.cumsum(a * a, axis=1, dtype=np.float32)], axis=1)
    tot1, tot2 = p1[:, 16], p2[:, 16]

    best_e = np.zeros(nb, dtype=np.uint8)
    best_sse = np.full(nb, np.inf, dtype=np.float32)

    for k in range(17):                       # k = number of "outer" weights
        outer, inner = float(k), float(16 - k)
        so, si = p1[:, k], tot1 - p1[:, k]
        s_star = (4.0 * so + si) / (16.0 * outer + inner)
        # the split is only valid for s in [a[k]/2.5, a[k-1]/2.5)
        lo = a[:, k] / 2.5 if k < 16 else np.zeros(nb, dtype=np.float32)
        hi = a[:, k - 1] / 2.5 if k > 0 else np.full(nb, np.inf, dtype=np.float32)
        s = np.clip(s_star, lo, hi)
        near = np.searchsorted(VALID_VALS, s)
        for idx in (near - 1, near):
            idx = np.clip(idx, 0, VALID.size - 1)
            sv = VALID_VALS[idx]
            # keep only candidates that really belong to this split
            inside = (sv >= lo) & (sv <= hi)
            sse = (p2[:, k] - 8.0 * sv * so + 16.0 * sv * sv * outer
                   + (tot2 - p2[:, k]) - 2.0 * sv * si + sv * sv * inner)
            upd = inside & (sse < best_sse)
            best_sse = np.where(upd, sse, best_sse)
            best_e = np.where(upd, (idx + 1).astype(np.uint8), best_e)

    # a half that is identically zero is stored as e=0 (engine returns 0 early)
    dead = a[:, 0] == 0.0
    scale = TABLE[best_e][:, None]
    outer = np.abs(x) > (2.5 * scale)
    neg = np.signbit(x)
    codes = np.where(outer, np.where(neg, 0, 3), np.where(neg, 1, 2)).astype(np.uint8)
    if dead.any():
        best_e = np.where(dead, np.uint8(0), best_e).astype(np.uint8)
        codes = np.where(dead[:, None], np.uint8(0), codes)
    return best_e, codes


def _transcode_serial(raw: np.ndarray, book: "Codebook") -> np.ndarray:
    nb = raw.shape[0]
    out = np.empty((nb, BLOCK_BYTES), dtype=np.uint8)
    for half in range(2):
        e, packed = book.encode_half(raw[:, 4 * half:4 * half + 4],
                                     raw[:, SCALE_OFFSET + half])
        out[:, SCALE_OFFSET + half] = e
        out[:, 4 * half:4 * half + 4] = packed
    return out


def transcode_blocks(raw: np.ndarray, book: "Codebook", workers: int = 1) -> np.ndarray:
    """(nb,10) affine uint8 -> (nb,10) S40 uint8 (fast, exact path).

    With `workers > 1` the chunk is split across slices and encoded on a thread
    pool.  Every numpy op used here releases the GIL, so this is close to a
    linear speedup on a multi-core host (measured 10x on 16 cores); the
    result is byte-identical to the serial path.
    """
    if workers > 1 and raw.shape[0] >= (1 << 16):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(lambda sl: _transcode_serial(sl, book),
                                np.array_split(raw, workers)))
        return np.concatenate(parts)
    return _transcode_serial(raw, book)


def transcode_blocks_reference(raw: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Slow closed-form encoder, kept as an independent cross-check."""
    nb = raw.shape[0]
    out = np.empty((nb, BLOCK_BYTES), dtype=np.uint8)
    for half in range(2):
        qs = raw[:, 4 * half:4 * half + 4]
        codes = _codes_from_bytes(qs)
        scale = TABLE[raw[:, SCALE_OFFSET + half]]
        x = grid[codes] * scale[:, None]
        e, new_codes = s40_encode_half(x)
        out[:, SCALE_OFFSET + half] = e
        out[:, 4 * half:4 * half + 4] = _bytes_from_codes(new_codes)
    return out


def brute_force_blocks(raw: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Third, independent encoder: the engine's own 126-scale brute force."""
    nb = raw.shape[0]
    out = np.empty((nb, BLOCK_BYTES), dtype=np.uint8)
    for half in range(2):
        qs = raw[:, 4 * half:4 * half + 4]
        codes = _codes_from_bytes(qs).astype(np.int64)
        s = TABLE[raw[:, SCALE_OFFSET + half]]
        x = grid[codes] * s[:, None]
        best_e = np.zeros(nb, dtype=np.uint8)
        best_sse = np.full(nb, np.inf, dtype=np.float32)
        best_codes = np.zeros((nb, 16), dtype=np.uint8)
        for e in range(1, 0x7F):
            sv = np.float32(TABLE[e])
            outer = np.abs(x) > (2.5 * sv)
            neg = np.signbit(x)
            c = np.where(outer, np.where(neg, 0, 3),
                         np.where(neg, 1, 2)).astype(np.uint8)
            sse = ((x - MORD[c] * sv) ** 2).sum(axis=1)
            upd = sse < best_sse
            best_sse = np.where(upd, sse, best_sse)
            best_e = np.where(upd, np.uint8(e), best_e)
            best_codes = np.where(upd[:, None], c, best_codes)
        out[:, SCALE_OFFSET + half] = best_e
        out[:, 4 * half:4 * half + 4] = _bytes_from_codes(best_codes)
    return out


def s40_decode_blocks(packed: np.ndarray) -> np.ndarray:
    """(nb,10) S40 uint8 -> (nb,32) float32, for verification only."""
    nb = packed.shape[0]
    out = np.empty((nb, QK), dtype=np.float32)
    for half in range(2):
        codes = _codes_from_bytes(packed[:, 4 * half:4 * half + 4]).astype(np.int64)
        out[:, half * 16:(half + 1) * 16] = MORD[codes] * TABLE[packed[:, SCALE_OFFSET + half]][:, None]
    return out


def affine_decode_blocks(packed: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """(nb,10) legacy affine uint8 -> (nb,32) float32, for verification only."""
    nb = packed.shape[0]
    out = np.empty((nb, QK), dtype=np.float32)
    for half in range(2):
        codes = _codes_from_bytes(packed[:, 4 * half:4 * half + 4]).astype(np.int64)
        out[:, half * 16:(half + 1) * 16] = grid[codes] * TABLE[packed[:, SCALE_OFFSET + half]][:, None]
    return out


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def _grid(text: str) -> np.ndarray:
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("grid must have exactly 4 comma-separated values")
    return np.array(vals, dtype=np.float32)


def cmd_audit(path: Path, grid: np.ndarray) -> int:
    alignment, data_start, tensors = parse_header(path)
    print(f"{path}")
    print(f"  GGUF v3, {len(tensors)} tensors, alignment {alignment}, data at {data_start}")
    for dtype, label in ((LEGACY, "type 107 legacy affine"),
                         (S40, "type 111 canonical S40")):
        found = target_tensors(tensors, dtype)
        total = sum(n for _, n in found)
        print(f"  {label:<24}: {len(found):>2} tensors, {total / 2**30:.1f} GiB")
    legacy = target_tensors(tensors, LEGACY)
    if legacy:
        t, expected = legacy[0]
        with open(path, "rb") as f:
            f.seek(data_start + t["offset"])
            head = np.frombuffer(f.read(1 << 20), dtype=np.uint8)
        nb = head.size // BLOCK_BYTES
        blocks = head[:nb * BLOCK_BYTES].reshape(nb, BLOCK_BYTES)
        codes = np.concatenate([_codes_from_bytes(blocks[:, 4 * h:4 * h + 4]) for h in range(2)], axis=1)
        hist = np.bincount(codes.ravel(), minlength=4) / codes.size
        print(f"  first tensor        : {t['name']}  dims={t['dims']}")
        print(f"  code histogram      : [{hist[0]:.3f}, {hist[1]:.3f}, {hist[2]:.3f}, {hist[3]:.3f}]")
        print(f"  (affine signature ~0.29/0.42/0.23/0.06; S40 ~0.20/0.27/0.32/0.20)")
        print(f"  scale bytes e0/e1   : {int(blocks[:, 8].min())}..{int(blocks[:, 8].max())} / "
              f"{int(blocks[:, 9].min())}..{int(blocks[:, 9].max())}  (valid <= 126)")
    return 0


def cmd_dry_run(path: Path, grid: np.ndarray, blocks_per_chunk: int, threads: int) -> int:
    book = Codebook(grid)
    alignment, data_start, tensors = parse_header(path)
    legacy = target_tensors(tensors, LEGACY)
    if not legacy:
        print("no type-107 tensors; nothing to transcode")
        return 0
    check = min(1 << 16, blocks_per_chunk)
    print("dry-run: nothing is written to the model")
    print(f"  codebook : {grid.tolist()} -> magnitude groups "
          f"{[g.tolist() for g in book.groups]} |v|={book.M.tolist()} splits={book.ts}")
    print(f"  {'tensor':<34} {'corr(affine,s40)':>17} {'rmse':>10} {'e0 range':>13} {'enc':>9}")
    worst, bad, fast_secs, total_ties, sampled = 1.0, 0, 0.0, 0, 0
    with open(path, "rb") as f:
        for t, expected in legacy:
            f.seek(data_start + t["offset"])
            raw = np.frombuffer(f.read(blocks_per_chunk * BLOCK_BYTES), dtype=np.uint8)
            nb = raw.size // BLOCK_BYTES
            blocks = raw[:nb * BLOCK_BYTES].reshape(nb, BLOCK_BYTES).copy()
            t1 = time.perf_counter()
            out = transcode_blocks(blocks, book, workers=threads)
            fast_secs += time.perf_counter() - t1
            sampled += nb

            # Independent cross-checks.  The closed-form encoder must be
            # byte-identical to the fast one (serial *and* threaded); against
            # the engine's own brute force we can only require an equal-or-
            # better squared error, since S40 has exact ties ("all inner at s"
            # == "all outer at s/4") that the two may resolve differently.
            sample = blocks[:check]
            aff = affine_decode_blocks(sample, grid)
            fast = transcode_blocks(sample, book)
            ref = transcode_blocks_reference(sample, grid)
            bru = brute_force_blocks(sample, grid)
            sse_fast = ((aff - s40_decode_blocks(fast)) ** 2).sum(axis=1)
            sse_ref = ((aff - s40_decode_blocks(ref)) ** 2).sum(axis=1)
            sse_bru = ((aff - s40_decode_blocks(bru)) ** 2).sum(axis=1)
            ties = int((sse_fast != sse_bru).sum())
            worse = int((sse_fast > sse_bru * (1 + 1e-6) + 1e-12).sum())
            ok = bool((fast == ref).all() and worse == 0
                      and (sse_fast == sse_ref).all()
                      and (fast == transcode_blocks(sample, book, workers=8)).all())
            bad += 0 if ok else 1
            total_ties += ties

            a = affine_decode_blocks(blocks, grid).ravel()
            b = s40_decode_blocks(out).ravel()
            am, bm = a.mean(), b.mean()
            corr = float(((a - am) * (b - bm)).sum()
                         / np.sqrt(((a - am) ** 2).sum() * ((b - bm) ** 2).sum()))
            rmse = float(np.sqrt(((a - b) ** 2).mean()))
            worst = min(worst, corr)
            e0 = out[:, 8]
            print(f"  {t['name']:<34} {corr:>17.4f} {rmse:>10.6f} "
                  f"{int(e0.min()):>6}..{int(e0.max()):<6} {'ok' if ok else 'MISMATCH':>9}")
    total_blocks = sum(n for _, n in legacy) // BLOCK_BYTES
    rate = sampled / fast_secs
    print(f"  worst corr(affine,s40) = {worst:.4f}")
    print(f"  encoder cross-check     : "
          + ("fast == closed-form (bytes) and never worse than the engine's "
             f"brute force, on {check} blocks/tensor "
             f"({total_ties} blocks where the error differs)"
             if not bad else f"{bad} TENSORS MISMATCH"))
    print(f"  encoder throughput      : {rate / 1e6:.1f} M blocks/s "
          f"({rate * BLOCK_BYTES / 1e6:.0f} MB/s, {max(1, threads)} thread(s)) "
          f"-> full transcode ~{total_blocks / rate / 60:.0f} min")
    return 0 if not bad else 1


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def cmd_transcode(path: Path, grid: np.ndarray, blocks_per_chunk: int, threads: int) -> int:
    book = Codebook(grid)
    alignment, data_start, tensors = parse_header(path)
    legacy = target_tensors(tensors, LEGACY)
    done = target_tensors(tensors, S40)
    if done:
        print(f"note: {len(done)} tensors are already type 111 (previous run)")
    if not legacy:
        print("no type-107 tensors remain; nothing to do")
        return 0

    scratch = path.with_suffix(path.suffix + ".transcode")
    scratch.mkdir(exist_ok=True)
    _fsync_dir(path.parent)          # make the side-file directory itself durable
    # a tensor whose type flip landed but whose side file outlived a crash
    for t, _ in done:
        stale = scratch / f"{t['index']:05d}.s40"
        if stale.exists():
            stale.unlink()
    total_bytes = sum(n for _, n in legacy)
    print(f"transcoding {len(legacy)} tensors / {total_bytes / 2**30:.1f} GiB in place")
    print(f"side files     : {scratch}")
    print(f"layout         : affine {grid.tolist()} * ue4m3(e_half) -> "
          "S40 [-4,-1,1,4] * ue4m3(e_half)")

    t0 = time.time()
    written = 0
    for i, (t, nbytes) in enumerate(legacy, 1):
        payload_off = data_start + t["offset"]
        side = scratch / f"{t['index']:05d}.s40"
        tmp = scratch / f"{t['index']:05d}.s40.tmp"
        label = f"[{i}/{len(legacy)}] {t['name']}"

        with open(path, "r+b") as f:
            if side.exists():
                if side.stat().st_size != nbytes:
                    raise RuntimeError(f"{side} has wrong size; remove it and re-run")
                print(f"{label}: restoring from side file")
            else:
                with open(tmp, "wb") as s:
                    remaining = nbytes
                    f.seek(payload_off)
                    while remaining:
                        take = min(blocks_per_chunk * BLOCK_BYTES, remaining)
                        raw = np.frombuffer(f.read(take), dtype=np.uint8)
                        nb = raw.size // BLOCK_BYTES
                        if nb == 0:
                            raise RuntimeError(f"{t['name']}: short read")
                        out = transcode_blocks(
                            raw[:nb * BLOCK_BYTES].reshape(nb, BLOCK_BYTES).copy(),
                            book, workers=threads)
                        s.write(out.tobytes())
                        remaining -= nb * BLOCK_BYTES
                    s.flush()
                    os.fsync(s.fileno())
                os.rename(tmp, side)
                _fsync_dir(scratch)

            # 1) durable payload
            with open(side, "rb") as s:
                off = payload_off
                while True:
                    buf = s.read(1 << 24)
                    if not buf:
                        break
                    f.seek(off)
                    f.write(buf)
                    off += len(buf)
            f.flush()
            os.fsync(f.fileno())

            # 2) durable type flip, only then drop the side file
            f.seek(t["type_field"])
            f.write(struct.pack("<I", S40))
            f.flush()
            os.fsync(f.fileno())

        side.unlink()
        written += nbytes
        rate = written / max(1e-9, time.time() - t0) / 1e6
        eta = (total_bytes - written) / max(1e-9, rate * 1e6)
        print(f"{label}: ok  ({nbytes / 2**20:.0f} MiB, {rate:.0f} MB/s, ETA {eta / 60:.0f} min)")

    # verify
    _, _, after = parse_header(path)
    left = target_tensors(after, LEGACY)
    have = target_tensors(after, S40)
    if left or len(have) != len(legacy) + len(done):
        raise RuntimeError(f"post-write verification failed: legacy={len(left)} s40={len(have)}")
    try:
        scratch.rmdir()
    except OSError:
        print(f"note: side directory not empty: {scratch}")
    print(f"done: {len(have)} type-111 expert tensors, {total_bytes / 2**30:.1f} GiB transcoded "
          f"in {time.time() - t0:.0f}s")
    print("the file is now canonical dual-scale S40 ROCmFP2; the engine can load it")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Transcode legacy affine ROCmFP2 (GGUF type 107) to canonical S40 (type 111) in place.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----")[1] if "Usage\n-----" in __doc__ else None)
    ap.add_argument("mode", choices=("audit", "dry-run", "transcode"))
    ap.add_argument("model", type=Path)
    ap.add_argument("--grid", type=_grid, default=_grid(DEFAULT_GRID),
                    help=f"affine codebook (default {DEFAULT_GRID})")
    ap.add_argument("--blocks-per-chunk", type=int, default=1 << 21,
                    help="blocks (32 weights) processed per numpy chunk (default 2097152)")
    ap.add_argument("--threads", type=int, default=0,
                    help="threads per chunk (0 = auto: min(16, cpu_count))")
    ap.add_argument("--yes", action="store_true", help="required to modify the model")
    args = ap.parse_args()
    threads = args.threads or min(16, os.cpu_count() or 1)

    if not args.model.is_file():
        print(f"error: not a file: {args.model}", file=sys.stderr)
        return 1

    if args.mode == "audit":
        return cmd_audit(args.model, args.grid)
    if args.mode == "dry-run":
        return cmd_dry_run(args.model, args.grid, args.blocks_per_chunk, threads)
    if not args.yes:
        print("refusing to modify the model without --yes", file=sys.stderr)
        return 2
    return cmd_transcode(args.model, args.grid, args.blocks_per_chunk, threads)


if __name__ == "__main__":
    raise SystemExit(main())
