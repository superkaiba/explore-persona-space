#!/usr/bin/env bash
# Task #483 canonical persona-pool dispatch driver (plan v3 §3).
#
# Single GPU job: recipe-(a) extract over 6 layers {7,10,14,20,21,27} +
# recipe-(b) response-mean extract at L20/L21 + matrix-build (every available
# layer x centering {global_mean, none}) + audit (K1 stability gate + K3 #478
# regression + occupancy + edge fit + pool finalize + HF matrix uploads).
#
# Phase A artifacts are ALREADY committed on the issue-483 branch
# (roster_v1.json + synthetic_candidates_r1.json) so this driver only
# runs the GPU + on-pod CPU phases. The build script's CLI flags are
# orthogonal:
#
#   --extract        recipe (a) batch-1 + recipe (b) vLLM gen + TF mean-pool
#   --build-matrices every layer x centering -> JSONs (committed + staged)
#   --audit          gates as exit-non-zero branches, K1/K3 blocking
#
# The audit phase finalizes pool_v1.json + pool_meta_v1.json and uploads
# the canonical committed matrices to the HF data repo (the centroid
# bundles upload during --extract per phase). The script exits non-zero
# on any gate failure, propagating to this driver via `set -e`.
#
# GCP lane: this command is BLOCKING (no setsid daemonization, no pid
# file). The GCE startup script waits on this script's exit before
# declaring done.
#
# Launch (via the backend router; plan v3 §8):
#   uv run python scripts/dispatch_issue.py launch \
#     --issue 483 --intent lora-7b --repo-branch issue-483 \
#     --workload-cmd 'bash scripts/issue483_dispatch.sh'

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

echo "[i483-dispatch] starting on $(hostname) at $(date -u +%FT%TZ)"
echo "[i483-dispatch] cwd=$REPO_ROOT branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "[i483-dispatch] commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[i483-dispatch] EPS_ISSUE=${EPS_ISSUE:-unset} EPS_ATTEMPT_ID=${EPS_ATTEMPT_ID:-unset}"
echo "[i483-dispatch] HF_TOKEN:${HF_TOKEN:+present} HF_HOME=${HF_HOME:-unset}"

# Env: canonical recipe — `set -a && source .env`, NEVER a bare
# load_dotenv() in a stdin heredoc (gotchas.md / #552 / #612).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# HF token is required for the per-phase centroid uploads + final matrix
# uploads. Fail loud — silent missing uploads would orphan the centroids
# (plan §3.5 + upload-policy.md "Fail-loud uploads").
[ -n "${HF_TOKEN:-}" ] || { echo "[i483-dispatch] FATAL: HF_TOKEN missing"; exit 2; }

# ANTHROPIC_API_KEY only matters if the dispatcher needed to re-generate
# synthetics, but Phase A is already complete on the branch — skip the
# check, leave it as a no-op assumption.

nvidia-smi || { echo "[i483-dispatch] FATAL: nvidia-smi failed (no GPU?)"; exit 3; }

# Phase A is already committed on the branch — the GCE startup clones
# issue-483 (via --repo-branch issue-483), so roster_v1.json and
# synthetic_candidates_r1.json should be present under
# data/canonical_persona_pool/. Sanity-check before burning GPU time:
ROSTER="data/canonical_persona_pool/roster_v1.json"
SYNTH="data/canonical_persona_pool/synthetic_candidates_r1.json"
LEGACY="data/canonical_persona_pool/legacy/cosine_distance_matrix_layer20_478.json"
for f in "$ROSTER" "$SYNTH" "$LEGACY"; do
  [ -f "$f" ] || { echo "[i483-dispatch] FATAL: Phase A artifact missing: $f"; exit 4; }
done
ROSTER_N=$(uv run python -c "import json; print(len(json.load(open('$ROSTER'))['personas']))")
SYNTH_N=$(uv run python -c "import json; print(json.load(open('$SYNTH'))['n_candidates'])")
echo "[i483-dispatch] roster=$ROSTER_N personas synthetics=$SYNTH_N candidates"

# Phase B+C+D+E in one invocation — flags compose orthogonally
# (build_canonical_persona_pool.py main()). The script exits non-zero
# on any K1/K3 blocking gate failure, propagating here via `set -e`.
echo "[i483-dispatch] [phase=extract+matrices+audit] launching build pipeline"
uv run python scripts/build_canonical_persona_pool.py \
  --extract \
  --build-matrices \
  --audit \
  --device cuda:0 \
  --max-new-tokens 1024 \
  --tf-batch-size 8

echo "[i483-dispatch] build pipeline exited 0 — all blocking gates passed"

# Acceptance suite read-back against the committed artifacts on the pod
# (the smoke harness already validated the pipeline end-to-end; this is
# the final post-build read against fresh GPU artifacts).
echo "[i483-dispatch] [phase=accept] running acceptance suite against fresh artifacts"
EPM_CANONICAL_POOL_DIR="$REPO_ROOT/data/canonical_persona_pool" \
  uv run pytest tests/test_persona_pool.py -v --tb=short

# Sentinel: print a summary line the GCE startup script + bg-Bash poller
# can grep on. The build script's own logs carry the structured pass/fail
# fields written into pool_meta_v1.json.
META="data/canonical_persona_pool/pool_meta_v1.json"
[ -f "$META" ] || { echo "[i483-dispatch] FATAL: $META missing post-audit"; exit 5; }
META_PATH="$META" uv run python - <<'PY'
import json, os
meta = json.load(open(os.environ["META_PATH"]))
gates = meta["gates"]
acc = meta["acceptance_preview"]
stab = gates["stability"]
reg = gates["regression_478"]
det = gates["determinism_floor"]
print(
    "[i483-dispatch] gates:",
    f'stability={stab["pass"]} (p95={stab["p95_abs_delta"]:.5f}),',
    f'regression_478={reg["pass"]} (genuine={reg["n_genuine"]}),',
    f'determinism_floor={det.get("value", "?")} (threshold={det.get("threshold", "?")}, pass={det.get("pass", "?")})',
)
print(
    "[i483-dispatch] acceptance:",
    f'pool16_floor_ok={acc["pool16_floor_ok"]},',
    f'assistant_floor_ok={acc["assistant_floor_ok"]}',
)
print("[i483-dispatch] documented_deficits:", meta.get("documented_deficits", []))
PY

echo "[i483-dispatch] DONE at $(date -u +%FT%TZ)"
