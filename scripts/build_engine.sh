#!/usr/bin/env bash
# ==============================================================================
# build_engine.sh — Build or Download ROCmFPX Engine for AMD Radeon (gfx1151 / gfx1201)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="${SCRIPT_DIR}/../engine"
# Engine source of truth: the git submodule at engine/src (see .gitmodules).
# We build our maintained branch on the org fork, so engine changes are pushed,
# reviewable commits rather than local edits that a rebuild can silently drop.
# The previous base, charlie12345/ROCmFPX, is unmaintained (last push 2026-08-22).
REPO_URL="https://github.com/Not-Limited-Company/ROCmFPX.git"
ENGINE_BRANCH="${ENGINE_BRANCH:-halofpx/v1.7}"
# Expected commit of ENGINE_BRANCH, i.e. what the released engine was measured
# against. Upstream base is 75e67a92b (org repo migrated 2026-09-02, q38rocm#11 —
# native DFlash2, strict-qwen MTP, empty-data_spec checkpoint tolerance); our
# branch carries one commit on top that tolerates a missing ffn_exp_probs_b on
# hash-routed layers. Moving this pin changes the shipped engine artifact, so it
# has to be re-measured before any published number is reused.
PINNED_COMMIT="${PINNED_COMMIT:-7489ed6fe9fa29e7e6c9ad6a6953b14b465b77ff}"
#
# patches/ is retired: the MTP prompt-cache checkpoint fix (issue #3) landed
# upstream as 0ef57fb81 "server: tolerate empty data_spec during checkpoint
# restore" (clears the MTP draft state in common/speculative.cpp, which is what
# the old patch open-coded in tools/server/server-context.cpp). The base code
# was refactored past the patch, so patches/ no longer applies and must not be
# applied again. Durable engine changes belong on the maintained branch above,
# not in patches/ — nothing in patches/ is applied by this script.
RELEASE_TARBALL_URL="https://github.com/julianmb/halofpx/releases/download/v1.7.0/strix-halo-vulkan-engine-v1.7.0-linux-x86_64.tar.gz"
EXPECTED_TARBALL_SHA="839f3398181d4df0fee61eb266a0ae923b1f6c86b61f89933895a2855399b3df"

# 1. Architecture Detection
detect_arch() {
    if [ -n "${CMAKE_HIP_ARCHITECTURES:-}" ]; then
        echo "$CMAKE_HIP_ARCHITECTURES"
        return
    fi
    if command -v rocm_agent_enumerator >/dev/null 2>&1; then
        for a in $(rocm_agent_enumerator 2>/dev/null); do
            if [[ "$a" =~ ^gfx[0-9a-f]+$ ]] && [ "$a" != "gfx000" ]; then
                echo "$a"
                return
            fi
        done
    fi
    if command -v offload-arch >/dev/null 2>&1; then
        for a in $(offload-arch 2>/dev/null); do
            if [[ "$a" =~ ^gfx[0-9a-f]+$ ]] && [ "$a" != "gfx000" ]; then
                echo "$a"
                return
            fi
        done
    fi
    if command -v rocminfo >/dev/null 2>&1; then
        for tok in $(rocminfo 2>/dev/null); do
            if [[ "$tok" =~ ^gfx1[0-9a-f]{3}$ ]]; then
                echo "$tok"
                return
            fi
        done
    fi
    echo "gfx1151"
}

TARGET_ARCH="$(detect_arch)"

# 2. ROCm / HIP install detection
#
# The distro package libamdhip64-dev installs a ROCm 5.x header tree under
# /usr/include/hip, and clang searches /usr/include BEFORE the real ROCm
# install. That stale header defines the bfloat16 helpers without static or
# inline, so every HIP translation unit emits a strong definition and linking
# libggml-hip.so dies with a wall of "duplicate symbol __float2bfloat16(float)"
# errors from ld.lld. The fix is to force the real ROCm include dir ahead of
# /usr/include via -isystem (adding it with -I does NOT work: the driver
# de-duplicates it against its own implicit ROCm include and leaves it last).
detect_rocm_root() {
    if [ -n "${ROCM_PATH:-}" ] && [ -f "${ROCM_PATH}/include/hip/hip_version.h" ]; then
        echo "${ROCM_PATH}"
        return
    fi
    if command -v hipconfig >/dev/null 2>&1; then
        local p
        p="$(hipconfig --rocmpath 2>/dev/null || true)"
        if [ -n "$p" ] && [ -f "${p}/include/hip/hip_version.h" ]; then
            echo "$p"
            return
        fi
    fi
    for d in /opt/rocm /opt/rocm-* /usr/lib/rocm; do
        if [ -f "${d}/include/hip/hip_version.h" ]; then
            echo "$d"
            return
        fi
    done
    echo "/opt/rocm"
}

ROCM_ROOT="$(detect_rocm_root)"
ROCM_INC="${ROCM_ROOT}/include"

hip_version_of() {
    # $1 = path to a hip_version.h; prints "unknown" when the file is unreadable
    local major minor
    if [ ! -f "$1" ]; then
        echo "unknown"
        return 0
    fi
    major="$(sed -n 's/^#define HIP_VERSION_MAJOR[[:space:]]\+\([0-9]\+\).*/\1/p' "$1" 2>/dev/null | head -1)"
    minor="$(sed -n 's/^#define HIP_VERSION_MINOR[[:space:]]\+\([0-9]\+\).*/\1/p' "$1" 2>/dev/null | head -1)"
    if [ -n "$major" ]; then
        echo "${major}.${minor:-0}"
    else
        echo "unknown"
    fi
}

# Warn about header trees that can shadow the ROCm install above.
for stale in /usr/include/hip /usr/include/hipblas /usr/include/rocblas /usr/include/hipcub /usr/include/hiprand /usr/include/hipsolver; do
    if [ -d "$stale" ]; then
        echo "⚠️  [NOTICE] Foreign HIP/ROCm headers found at ${stale} ($(hip_version_of "${stale}/hip_version.h"))"
        echo "   These shadow ${ROCM_INC} ($(hip_version_of "${ROCM_INC}/hip/hip_version.h")) and break the HIP link."
        echo "   Remove them with: sudo apt purge libamdhip64-dev libhipblas-dev librocblas-dev libhipcub-dev libhsa-runtime-dev"
    fi
done

download_prebuilt() {
    if [ "$TARGET_ARCH" != "gfx1151" ] && [ "$TARGET_ARCH" != "gfx1150" ]; then
        echo "⚠️  [NOTICE] Pre-compiled binaries in v1.7.0 are packaged for Strix Halo (gfx1151)."
        echo "   For other AMD GPUs (${TARGET_ARCH}), compiling from source is required."
        echo "   Proceeding with source compilation for ${TARGET_ARCH}..."
        return 1
    fi

    echo "================================================================================"
    echo " 📥 Downloading Pre-Compiled ROCmFPX Engine (v1.7.0) for AMD Strix Halo"
    echo " Source: ${RELEASE_TARBALL_URL}"
    echo "================================================================================"
    mkdir -p "${ENGINE_DIR}"
    TAR_PATH="/tmp/strix-halo-engine-v1.7.0.tar.gz"
    
    curl -L "${RELEASE_TARBALL_URL}" -o "${TAR_PATH}" --progress-bar
    
    if command -v sha256sum >/dev/null 2>&1; then
        echo "${EXPECTED_TARBALL_SHA}  ${TAR_PATH}" | sha256sum -c -
        echo "✅ Checksum verified!"
    fi
    
    echo "Extracting binaries into ${ENGINE_DIR}..."
    tar -xzf "${TAR_PATH}" -C /tmp/
    cp -a /tmp/strix-halo-rocmfpx-engine/* "${ENGINE_DIR}/"
    rm -rf /tmp/strix-halo-rocmfpx-engine "${TAR_PATH}"
    
    echo "✅ Pre-built engine ready in: ${ENGINE_DIR}/bin"
    echo "Verifying available hardware acceleration backends..."
    "${ENGINE_DIR}/bin/llama-server" --list-devices 2>/dev/null || true
    echo "Run: source ./scripts/setup_env.sh"
    exit 0
}

# Parse CLI flags
BUILD_MODE="vulkan"
DOWNLOAD_ONLY=0
UPDATE_ENGINE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prebuilt|--download)
            DOWNLOAD_ONLY=1
            shift
            ;;
        --update-engine|--sync-engine)
            UPDATE_ENGINE=1
            shift
            ;;
        --rocm|--dual|--hip)
            BUILD_MODE="dual"
            shift
            ;;
        --vulkan|--vulkan-only)
            BUILD_MODE="vulkan"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --vulkan (default)  Build lightweight standalone Vulkan engine (no ROCm/HIP required)"
            echo "  --rocm, --dual      Build dual ROCm (HIP) + Vulkan engine for high-batch prefill"
            echo "  --prebuilt          Download pre-compiled release tarball"
            echo "  --update-engine     Pull the maintained engine branch into the submodule, then exit"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

if [ "$UPDATE_ENGINE" -eq 1 ]; then
    # Bring the engine submodule onto the tip of the maintained branch. This
    # changes the engine artifact, so re-measure and bump PINNED_COMMIT before
    # reusing any published number.
    if [ ! -e "${ENGINE_DIR}/src/.git" ]; then
        git -C "${SCRIPT_DIR}/.." submodule update --init --recursive engine/src
    fi
    git -C "${ENGINE_DIR}/src" fetch origin "${ENGINE_BRANCH}"
    git -C "${ENGINE_DIR}/src" checkout -q "${ENGINE_BRANCH}"
    git -C "${ENGINE_DIR}/src" merge --ff-only "origin/${ENGINE_BRANCH}"
    new_head="$(git -C "${ENGINE_DIR}/src" rev-parse HEAD)"
    echo "Engine source updated: ${ENGINE_BRANCH} @ ${new_head:0:9}"
    if [ "${new_head}" != "${PINNED_COMMIT}" ]; then
        echo "⚠️  That is not the recorded pin (${PINNED_COMMIT:0:9})."
        echo "   Re-measure the engine, then bump PINNED_COMMIT and commit the"
        echo "   submodule pointer before publishing performance numbers."
    fi
    exit 0
fi

if [ "$DOWNLOAD_ONLY" -eq 1 ]; then
    if download_prebuilt; then
        exit 0
    fi
fi

echo "================================================================================"
echo " ⚙️ ROCmFPX llama.cpp Engine Setup for AMD Platforms"
echo " Build Mode:           ${BUILD_MODE^^} (Primary backend: Vulkan0 Wave64)"
echo " Target Architecture:  ${TARGET_ARCH}"
if [ "$TARGET_ARCH" == "gfx1151" ] || [ "$TARGET_ARCH" == "gfx1150" ]; then
    echo " Platform:             AMD Strix Halo / Ryzen AI Max APU (Unified Memory)"
else
    echo " Platform:             AMD Radeon GPU (${TARGET_ARCH})"
fi
echo "================================================================================"

# Check compiler dependencies
MISSING_TOOLS=0
for tool in cmake git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "⚠️  Tool '$tool' is not installed."
        MISSING_TOOLS=1
    fi
done

# Check Vulkan shader compiler (glslc)
if ! command -v glslc >/dev/null 2>&1; then
    echo "⚠️  'glslc' (Vulkan shader compiler) not found."
    echo "   To compile with Vulkan, install: sudo apt install glslc libvulkan-dev mesa-vulkan-drivers"
    if [ "$TARGET_ARCH" == "gfx1151" ]; then
        read -p "Would you like to download the pre-compiled Strix Halo binaries instead? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            download_prebuilt
        fi
    fi
    MISSING_TOOLS=1
fi

if [ "$BUILD_MODE" == "dual" ]; then
    if ! command -v hipcc >/dev/null 2>&1 && [ ! -d "/opt/rocm" ]; then
        echo "⚠️  ROCm toolchain (hipcc) not found for dual build. Falling back to pure Vulkan build..."
        BUILD_MODE="vulkan"
    fi
fi

if [ "$MISSING_TOOLS" -eq 1 ]; then
    echo "Required build tools are missing. Falling back to pre-compiled release download..."
    download_prebuilt
fi

# Initialise the engine submodule if it is missing. Never clone ad hoc: a stray
# clone is invisible to the parent repo, its build tree ends up holding the only
# copy of the engine's shared libraries, and its git identity has bitten us.
if [ ! -e "${ENGINE_DIR}/src/.git" ]; then
    echo "Initialising the engine submodule (engine/src)..."
    git -C "${SCRIPT_DIR}/.." submodule update --init --recursive engine/src
fi

cd "${ENGINE_DIR}/src"
if ! git checkout -q "${ENGINE_BRANCH}" 2>/dev/null; then
    git checkout -q -B "${ENGINE_BRANCH}" "origin/${ENGINE_BRANCH}" 2>/dev/null || {
        echo "❌ Cannot check out ${ENGINE_BRANCH} in ${ENGINE_DIR}/src."
        echo "   Run: git submodule update --init --recursive engine/src"
        exit 1
    }
fi

ENGINE_HEAD="$(git rev-parse HEAD)"
if [ "${ENGINE_HEAD}" != "${PINNED_COMMIT}" ]; then
    if git merge-base --is-ancestor "${PINNED_COMMIT}" HEAD 2>/dev/null; then
        echo "⚠️  [NOTICE] Engine source is ahead of the recorded pin"
        echo "   checked out ${ENGINE_HEAD}"
        echo "   recorded pin ${PINNED_COMMIT}"
        echo "   Re-measure before reusing any published performance number, then"
        echo "   bump PINNED_COMMIT once the new engine is the one we ship."
    else
        echo "❌ Engine source (${ENGINE_HEAD}) does not contain the recorded pin"
        echo "   (${PINNED_COMMIT}). Check out ${ENGINE_BRANCH} from ${REPO_URL}, or"
        echo "   update PINNED_COMMIT deliberately."
        exit 1
    fi
fi
echo "Engine source:         ${ENGINE_BRANCH} @ ${ENGINE_HEAD:0:9}"

# patches/ is retired on purpose: the MTP prompt-cache checkpoint fix (issue
# #3) is upstream in the pinned commit as 0ef57fb81 "server: tolerate empty
# data_spec during checkpoint restore", and the surrounding code has been
# refactored past the old diff. Applying a stale patch here would either fail
# (silently, if swallowed) or reintroduce a duplicate of the upstream fix, so
# any patch still present in patches/ is a hard error instead.
PATCH_DIR="${SCRIPT_DIR}/../patches"
if [ -d "${PATCH_DIR}" ] && [ -n "$(ls -A "${PATCH_DIR}" 2>/dev/null)" ]; then
    echo "❌ ${PATCH_DIR} is not empty:"
    ls -1 "${PATCH_DIR}"
    echo "   patches/ is retired (see the note at the top of this script)."
    echo "   If a patch is genuinely required, apply it explicitly and update this script."
    exit 1
fi

# Verify that <hip/hip_bf16.h> really resolves inside ${ROCM_INC}. Without the
# -isystem below, clang picks up a distro ROCm 5.x copy from /usr/include and
# the link fails much later with confusing duplicate-symbol errors.
verify_hip_headers() {
    local probe_dir cxx resolved
    probe_dir="$(mktemp -d)"
    printf '#include <hip/hip_bf16.h>\nint main(void) { return 0; }\n' > "${probe_dir}/probe.hip"

    cxx="${ROCM_ROOT}/lib/llvm/bin/clang++"
    if [ ! -x "$cxx" ]; then
        cxx="$(command -v clang++ || command -v hipcc || true)"
    fi
    if [ -z "$cxx" ]; then
        echo "⚠️  [NOTICE] No HIP compiler found to verify header search paths; continuing."
        rm -rf "${probe_dir}"
        return 0
    fi

    resolved="$("$cxx" -x hip --offload-arch="${TARGET_ARCH}" -isystem "${ROCM_INC}" \
        -fsyntax-only -H "${probe_dir}/probe.hip" 2>&1 \
        | awk '$2 ~ /\/hip\/hip_bf16\.h$/ { print $2; exit }')" || true
    rm -rf "${probe_dir}"

    if [ -z "${resolved}" ]; then
        echo "⚠️  [NOTICE] Could not resolve <hip/hip_bf16.h>; continuing without verification."
        return 0
    fi
    case "${resolved}" in
        "${ROCM_INC}"/*)
            echo "✅ HIP headers resolved from ${ROCM_INC}"
            ;;
        *)
            echo "❌ HIP headers resolve to ${resolved}"
            echo "   instead of ${ROCM_INC}/hip/hip_bf16.h — a distro HIP package is shadowing ROCm."
            echo "   Fix with: sudo apt purge libamdhip64-dev libhipblas-dev librocblas-dev libhipcub-dev libhsa-runtime-dev"
            return 1
            ;;
    esac
}

# Configure CMake
if [ "$BUILD_MODE" == "dual" ]; then
    BUILD_DIR="${ENGINE_DIR}/src/build-${TARGET_ARCH}-dual"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    verify_hip_headers || exit 1

    CMAKE_FLAGS=(
        -DGGML_HIP=ON
        -DAMDGPU_TARGETS="${TARGET_ARCH}"
        -DCMAKE_HIP_ARCHITECTURES="${TARGET_ARCH}"
        -DGGML_VULKAN=ON
        -DGGML_VULKAN_CHECK_RESULTS=OFF
        -DGGML_AVX=ON
        -DGGML_AVX2=ON
        -DGGML_AVX512=ON
        -DGGML_F16C=ON
        -DGGML_FMA=ON
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        # $ORIGIN in the rpath is what lets promotion copy the shared libraries
        # in beside the executables; engine/bin then loads its own copies.
        -DCMAKE_BUILD_RPATH='$ORIGIN'
        -DCMAKE_INSTALL_RPATH='$ORIGIN'
        -DCMAKE_HIP_FLAGS="-D__AMDGCN_WAVEFRONT_SIZE=32 -isystem ${ROCM_INC}"
        -DLLAMA_BUILD_WEBUI=OFF
        -DCMAKE_BUILD_TYPE=Release
    )
    echo "Configuring CMake with Dual ROCm (HIP) + Vulkan support for ${TARGET_ARCH}..."
else
    BUILD_DIR="${ENGINE_DIR}/src/build-vulkan"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    CMAKE_FLAGS=(
        -DGGML_VULKAN=ON
        -DGGML_HIP=OFF
        -DGGML_VULKAN_CHECK_RESULTS=OFF
        -DGGML_AVX=ON
        -DGGML_AVX2=ON
        -DGGML_AVX512=ON
        -DGGML_F16C=ON
        -DGGML_FMA=ON
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_BUILD_RPATH='$ORIGIN'
        -DCMAKE_INSTALL_RPATH='$ORIGIN'
        -DLLAMA_BUILD_WEBUI=OFF
        -DCMAKE_BUILD_TYPE=Release
    )
    echo "Configuring CMake with Standalone Vulkan engine (zero ROCm link-time dependencies)..."
fi

cmake .. "${CMAKE_FLAGS[@]}"

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
echo "Compiling binaries with ${JOBS} parallel threads..."
cmake --build . --config Release -j "${JOBS}" --target llama-server llama-cli llama-bench llama-quantize

# Link/Install Executables AND their shared libraries to engine/bin.
# Promoting only the executables (as this script used to) leaves the actual code
# in the build tree: engine/src then cannot be pruned, engine/bin is not
# self-contained, and the small executables can silently drift from the
# libraries they load. See AGENTS.md convention 9.
mkdir -p "${ENGINE_DIR}/bin"
cp -f bin/llama-server bin/llama-cli bin/llama-bench bin/llama-quantize "${ENGINE_DIR}/bin/"
if compgen -G "bin/*.so*" > /dev/null; then
    cp -f bin/*.so* "${ENGINE_DIR}/bin/"
fi

# Fail loudly if the promoted engine still resolves anything from the build tree.
if command -v ldd >/dev/null 2>&1; then
    engine_strays="$(LD_LIBRARY_PATH="${ENGINE_DIR}/bin" ldd "${ENGINE_DIR}/bin/llama-server" 2>/dev/null \
        | awk '{print $3}' | grep -c "^${ENGINE_DIR}/src/" || true)"
    if [ "${engine_strays:-0}" -gt 0 ]; then
        echo "❌ engine/bin is not self-contained: ${engine_strays} libraries still resolve from ${ENGINE_DIR}/src."
        echo "   The build needs \$ORIGIN in its rpath and the .so files promoted above."
        exit 1
    fi
fi

echo "================================================================================"
echo " ✅ Build Complete (${BUILD_MODE^^})!"
echo " Binaries installed to: ${ENGINE_DIR}/bin"
echo " Detected backends on host:"
"${ENGINE_DIR}/bin/llama-server" --list-devices 2>/dev/null || true
echo " Export environment:    source ./scripts/setup_env.sh"
echo "================================================================================"
