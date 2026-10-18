#!/usr/bin/env bash
# ==============================================================================
# pack_engine_release.sh — Package engine/bin into a release engine tarball
# ==============================================================================
#
# The Dockerfiles do not build the engine; they download a pre-built tarball:
#
#   Dockerfile        https://github.com/julianmb/halofpx/releases/download/<v>/strix-halo-vulkan-engine-<v>-linux-x86_64.tar.gz
#   Dockerfile.rocm   https://github.com/julianmb/q38rocm/releases/download/<v>/strix-halo-rocmfpx-engine-<v>-linux-x86_64.tar.gz
#
# Both extract a single top-level directory (strix-halo-rocmfpx-engine/) and copy
# its contents into /app/engine, then load the engine from /app/engine/bin via
# LD_LIBRARY_PATH. That layout is only valid if the tarball carries the engine's
# SHARED LIBRARIES next to the executables — which is exactly what this script
# guarantees (and what a promotion that copies only the executables breaks).
#
# This script does not upload anything. It builds the asset, verifies that the
# engine is self-contained, prints the SHA256, and prints the exact fields that
# have to be updated for the release to be consistent (tests/test_engine_artifact_sync.py
# enforces that the URL version and the filename version agree everywhere).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VERSION=""
MODE="dual"
ENGINE_BIN="${ROOT_DIR}/engine/bin"
OUT_DIR="${ROOT_DIR}/dist"

usage() {
    cat <<EOF
Usage: $0 --version <vX.Y.Z> [--mode vulkan|dual] [--engine-bin DIR] [--out DIR]

  --version    Release tag for the asset filename (required, e.g. v1.7.1)
  --mode       dual (default, ROCm+Vulkan) or vulkan — selects the asset name
  --engine-bin Directory to package (default: engine/bin)
  --out        Output directory (default: dist/, gitignored)

Run scripts/build_engine.sh first; this packages what that produced.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)   VERSION="${2:-}"; shift 2 ;;
        --mode)      MODE="${2:-}"; shift 2 ;;
        --engine-bin) ENGINE_BIN="${2:-}"; shift 2 ;;
        --out)       OUT_DIR="${2:-}"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [ -z "${VERSION}" ]; then
    echo "❌ --version is required (e.g. --version v1.7.1)"
    exit 1
fi
if [[ ! "${VERSION}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "❌ --version must look like vX.Y.Z (got '${VERSION}')"
    exit 1
fi
case "${MODE}" in
    vulkan) ASSET="strix-halo-vulkan-engine-${VERSION}-linux-x86_64.tar.gz" ;;
    dual|rocm) ASSET="strix-halo-rocmfpx-engine-${VERSION}-linux-x86_64.tar.gz" ;;
    *) echo "❌ --mode must be vulkan or dual (got '${MODE}')"; exit 1 ;;
esac

if [ ! -x "${ENGINE_BIN}/llama-server" ]; then
    echo "❌ ${ENGINE_BIN}/llama-server not found or not executable."
    echo "   Build it first: scripts/build_engine.sh --rocm"
    exit 1
fi

echo "================================================================================"
echo " 📦 Packaging engine release asset"
echo " Engine dir:           ${ENGINE_BIN}"
echo " Asset:                ${ASSET}"
echo "================================================================================"

# 1. Refuse to ship an engine that still depends on the build tree. A release
#    built from an un-promoted build directory loads its code from
#    engine/src/build-*, which does not exist in the image.
libs="$(find "${ENGINE_BIN}" -maxdepth 1 -name '*.so*' | wc -l)"
echo " Shared libraries found: ${libs}"
if [ "${libs}" -eq 0 ]; then
    echo "❌ No shared libraries in ${ENGINE_BIN}: the engine would not run in the image."
    echo "   Re-run scripts/build_engine.sh (it promotes bin/*.so* alongside the executables)."
    exit 1
fi

strays=0
if command -v ldd >/dev/null 2>&1; then
    strays="$(LD_LIBRARY_PATH="${ENGINE_BIN}" ldd "${ENGINE_BIN}/llama-server" 2>/dev/null \
        | awk '{print $3}' | grep -c "^${ROOT_DIR}/engine/src/" || true)"
fi
if [ "${strays:-0}" -gt 0 ]; then
    echo "❌ ${strays} libraries still resolve from engine/src — the engine is not self-contained."
    exit 1
fi
echo " ✅ Self-contained: llama-server resolves its libraries from ${ENGINE_BIN}"

# 2. Stage the exact layout the Dockerfiles expect.
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
PKG="${STAGE}/strix-halo-rocmfpx-engine"
mkdir -p "${PKG}/bin"
cp -a "${ENGINE_BIN}/." "${PKG}/bin/"

engine_version="$(LD_LIBRARY_PATH="${ENGINE_BIN}" timeout 60 "${ENGINE_BIN}/llama-server" --version 2>/dev/null | head -1 || true)"
engine_commit="$(git -C "${ROOT_DIR}/engine/src" rev-parse HEAD 2>/dev/null || echo unknown)"
engine_branch="$(git -C "${ROOT_DIR}/engine/src" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
{
    echo "asset:    ${ASSET}"
    echo "mode:     ${MODE}"
    echo "built:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "engine:   ${engine_version:-unknown}"
    echo "source:   ${engine_branch} @ ${engine_commit}"
    echo
    echo "Contents are the engine's executables AND the shared libraries they load."
    echo "The image copies this tree to /app/engine and loads from /app/engine/bin."
} > "${PKG}/ENGINE-BUILD.txt"

# 3. Build the asset.
mkdir -p "${OUT_DIR}"
OUT_PATH="${OUT_DIR}/${ASSET}"
tar -czf "${OUT_PATH}" -C "${STAGE}" strix-halo-rocmfpx-engine

# 4. Report, and spell out what has to move together.
SHA="$(sha256sum "${OUT_PATH}" | awk '{print $1}')"
SIZE="$(du -h "${OUT_PATH}" | cut -f1)"
echo
echo "================================================================================"
echo " ✅ ${OUT_PATH}  (${SIZE})"
echo " sha256: ${SHA}"
echo " engine source: ${engine_branch} @ ${engine_commit}"
echo "================================================================================"
echo " Upload it to the org's releases, then update, in one commit:"
echo "   scripts/build_engine.sh   RELEASE_TARBALL_URL + EXPECTED_TARBALL_SHA=\"${SHA}\""
echo "   Dockerfile                ARG ENGINE_RELEASE=${VERSION}"
echo "   Dockerfile.rocm           ARG ENGINE_RELEASE=${VERSION}"
echo "   tests/test_engine_artifact_sync.py enforces that the URL version, the"
echo "   filename version and the ARG all agree — a mismatch has shipped before."
echo " NOTE: the URLs above still point at julianmb/* releases; move them to the"
echo " org's release repo as part of the migration."
