#!/bin/bash
#
# Build and push base Docker images for OT-Agent evaluation.
# Run this on a machine with Docker installed (laptop, RunPod, etc.)
#
# Usage:
#   docker login  # first time only
#   bash scripts/docker/build_base_images.sh
#
# This builds 2 base images:
#   sleepymalc/ot-base-minimal   — for InferredBugs tasks (python3 + pip)
#   sleepymalc/ot-base-full      — for NL2Bash + dev set tasks (full toolset)
#

set -euo pipefail

REGISTRY="sleepymalc"

echo "========================================"
echo "Building OT-Agent base Docker images"
echo "Registry: ${REGISTRY}"
echo "========================================"

# Create temp build context
TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

# --- Image 1: Minimal (InferredBugs) ---
echo ""
echo ">>> Building ${REGISTRY}/ot-base-minimal ..."

cat > "${TMPDIR}/Dockerfile.minimal" << 'EOF'
FROM ubuntu:24.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /output
EOF

docker build -t "${REGISTRY}/ot-base-minimal:latest" -f "${TMPDIR}/Dockerfile.minimal" "${TMPDIR}"
docker push "${REGISTRY}/ot-base-minimal:latest"
echo ">>> Pushed ${REGISTRY}/ot-base-minimal:latest"

# --- Image 2: Full (NL2Bash + Dev Set) ---
echo ""
echo ">>> Building ${REGISTRY}/ot-base-full ..."

cat > "${TMPDIR}/Dockerfile.full" << 'EOF'
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        gettext-base \
        git \
        jq \
        less \
        procps \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        unzip \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /output /tests /app/data
EOF

docker build -t "${REGISTRY}/ot-base-full:latest" -f "${TMPDIR}/Dockerfile.full" "${TMPDIR}"
docker push "${REGISTRY}/ot-base-full:latest"
echo ">>> Pushed ${REGISTRY}/ot-base-full:latest"

echo ""
echo "========================================"
echo "Done! Images pushed:"
echo "  ${REGISTRY}/ot-base-minimal:latest"
echo "  ${REGISTRY}/ot-base-full:latest"
echo "========================================"
echo ""
echo "To verify:"
echo "  docker images | grep ot-base"
echo ""
echo "To test:"
echo "  docker run --rm ${REGISTRY}/ot-base-full bash -c 'python3 --version && git --version'"
