#!/usr/bin/env bash
# Task-registration-exempt native pilot; persistent disk and bounded STOP fence.
set -euo pipefail
set +x
mkdir -p /workspace/logs /workspace/workspace_jr
exec > /workspace/logs/workspace_jr_startup.log 2>&1
JR_OUT=/workspace/workspace_jr/primary_native_pilot
mkdir -p "$JR_OUT"
trap 'JR_RC=$?; printf "{\"exit_code\":%d,\"finished_at_epoch\":%d}\n" "$JR_RC" "$(date +%s)" > /workspace/workspace_jr/startup_exit.json' EXIT
echo '[phase=bootstrap]'
JR_CODE_SHA=$(curl --fail --silent --show-error -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/jr-code-sha)
if [[ ! "$JR_CODE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Missing exact code SHA in instance metadata' >&2
  exit 2
fi
curl --location --fail --silent --show-error https://astral.sh/uv/install.sh -o /workspace/install_uv.sh
sh /workspace/install_uv.sh
export PATH="/root/.local/bin:$PATH"
git clone --filter=blob:none --sparse --branch codex/jr-workspace-predictability-20260912 \
  https://github.com/superkaiba/explore-persona-space.git /workspace/explore-persona-space
cd /workspace/explore-persona-space
git sparse-checkout set src scripts runtime/workspace_jr configs/analysis \
  docs/exploratory_workspace_jr external/jacobian-lens
git checkout --detach "$JR_CODE_SHA"
git rev-parse HEAD > "$JR_OUT/code_sha.txt"
bash scripts/workspace_jr_native_pilot.sh "$JR_OUT" primary
