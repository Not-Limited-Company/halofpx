# ==============================================================================
# Dockerfile — HaloFPX Model Server (Default: Standalone Vulkan Engine)
# ==============================================================================

FROM ubuntu:24.04

ARG ENGINE_RELEASE=v1.7.0.nlc.1
# Digest of the engine asset at ENGINE_RELEASE. Both Dockerfiles download a
# pre-built engine and verify it, so a replaced or re-uploaded asset fails the
# build instead of silently changing the image.
ARG ENGINE_TARBALL_SHA=ef47daaee3fa2f0090d5035ec38f5383eb1532fe299df00461a732e9a3715b0d

LABEL maintainer="HaloFPX & Strix Halo Open Source Community"
LABEL description="Unified HaloFPX / ROCmFP4 Model Server for AMD Strix Halo & Radeon GPUs (Vulkan0 Wave64)"

ENV DEBIAN_FRONTEND=noninteractive
ENV AMD_VULKAN_ICD=RADV
ENV RADV_PERFTEST="gpl,sam,nggc"
ENV PATH="/app/engine/bin:${PATH}"
ENV LD_LIBRARY_PATH="/app/engine/bin:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

WORKDIR /app

# Install system dependencies & Mesa RADV Vulkan drivers (Lean ~350 MB: zero ROCm runtime required)
RUN apt-get update && apt-get install -y --no-install-recommends     ca-certificates     curl     vulkan-tools     mesa-vulkan-drivers     libvulkan1     libgomp1     python3     python3-pip     tar     git     && rm -rf /var/lib/apt/lists/*

# Install pre-compiled ROCmFPX engine binaries
RUN mkdir -p /app/engine &&     curl -fsSL "https://github.com/Not-Limited-Company/halofpx/releases/download/${ENGINE_RELEASE}/strix-halo-vulkan-engine-${ENGINE_RELEASE}-linux-x86_64.tar.gz" -o /tmp/engine.tar.gz &&     echo "${ENGINE_TARBALL_SHA}  /tmp/engine.tar.gz" | sha256sum -c - &&     tar -xzf /tmp/engine.tar.gz -C /tmp/ &&     cp -a /tmp/strix-halo-rocmfpx-engine/* /app/engine/ &&     rm -rf /tmp/strix-halo-rocmfpx-engine /tmp/engine.tar.gz

# Install Python requirements
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy source code
COPY . /app/
RUN chmod +x /app/scripts/*.sh /app/scripts/*.py

EXPOSE 8010

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3     CMD curl -f http://localhost:8010/health || exit 1

ENTRYPOINT ["python3", "-m", "halofpx.cli", "serve", "--host", "0.0.0.0", "--port", "8010"]
