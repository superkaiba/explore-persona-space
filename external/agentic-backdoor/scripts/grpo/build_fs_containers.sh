#!/bin/bash
# Ensure NL2Bash filesystem images are available in udocker.
# Pulls from Docker Hub if not already cached in UDOCKER_DIR.
#
# Usage: bash scripts/grpo/build_fs_containers.sh
set -euo pipefail

export UDOCKER_DIR="${UDOCKER_DIR:-/tmp/udocker_grpo_$$}"
echo "==> UDOCKER_DIR=$UDOCKER_DIR"

for FS_ID in 1 2 3 4; do
    IMAGE="sleepymalc/nl2bash-fs-${FS_ID}:latest"
    REPO_DIR="$UDOCKER_DIR/repos/sleepymalc/nl2bash-fs-${FS_ID}"

    if [ -d "$REPO_DIR" ]; then
        echo "    $IMAGE: already cached"
    else
        echo "==> Pulling $IMAGE ..."
        udocker pull "$IMAGE"
        echo "    Done: $IMAGE"
    fi
done

echo "==> All NL2Bash filesystem images ready."
