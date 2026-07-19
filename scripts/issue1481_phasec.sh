#!/usr/bin/env bash
# #1481 Phase C launcher (plan §4.4 Phase C; frozen wrapper composition):
# pre-stage the 4 reused committed checkpoints -> six-context panel at the
# 40 verdict arms (36 fresh via --arms + 4 reused via --ckpt-map) ->
# base-arms gap fill per behavior. Sequencing only — all logic lives in
# issue1481_worker.py / issue1481_dispatch.sh.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

OUT_ROOT="data/issue_1481/phaseC"
REUSED_ROOT="$OUT_ROOT/reused"

# Pre-stage the 4 reused committed checkpoints (stage_hub_prefix is a
# verbatim prefix mirror — ckpt-map paths point INSIDE the mirror; #928).
uv run python - <<'PY'
from explore_persona_space.orchestrate import hub
import json, pathlib
ROOT = pathlib.Path("data/issue_1481/phaseC/reused")
ARMS = {
    "imp-conv-con-lr1e5-s42": "adapters/issue1090_fu4/imp-conv-lr1e5/checkpoint-20",
    "imp-pers-con-lr3e5-s42": "adapters/issue1090_fu4/imp-pers-lr3e5/checkpoint-30",
    "imp-bare-con-lr3e5-s42": "adapters/issue1090_fu5/imp-bare-lr3e5/checkpoint-20",
    "syc-pers-con-lr1e5-s42": "adapters/issue1090_fu7/syc-c3-lr1e5/checkpoint-15",
}
ckpt_map = {}
for arm, prefix in ARMS.items():
    dest = ROOT / arm
    staged_root = dest / prefix
    if not (staged_root / "adapter_config.json").exists():
        hub.stage_hub_prefix("superkaiba1/explore-persona-space", prefix, dest, repo_type="model")
    assert (staged_root / "adapter_config.json").exists(), f"stage failed: {arm}"
    ckpt_map[arm] = str(staged_root)
pathlib.Path("data/issue_1481/phaseC").mkdir(parents=True, exist_ok=True)
pathlib.Path("data/issue_1481/phaseC/ckpt_map.json").write_text(json.dumps(ckpt_map, indent=1))
print("[phasec] reused ckpts staged:", json.dumps(ckpt_map, indent=1))
PY

FRESH_ARMS="cas-bare-con-lr1e5-s137,cas-bare-po-lr1e5-s137,cas-conv-con-lr1e5-s137,cas-conv-po-lr3e5-s137,cas-icl-con-lr1e4-s137,cas-icl-po-lr1e5-s137,cas-pers-con-lr1e5-s137,cas-pers-po-lr1e5-s137,imp-bare-con-lr3e5-s137,imp-bare-po-lr1e5-s137,imp-bare-po-lr1e5-s42,imp-conv-con-lr1e5-s137,imp-conv-po-lr1e5-s137,imp-conv-po-lr1e5-s42,imp-icl-con-lr1e4-s42,imp-icl-con-lr1e5-s137,imp-icl-po-lr1e4-s137,imp-icl-po-lr1e4-s42,imp-pers-con-lr3e5-s137,imp-pers-po-lr1e5-s137,imp-pers-po-lr1e5-s42,syc-bare-con-lr1e5-s137,syc-bare-con-lr1e5-s42,syc-bare-po-lr1e5-s137,syc-bare-po-lr1e5-s42,syc-conv-con-lr1e5-s137,syc-conv-con-lr1e5-s42,syc-conv-po-lr1e5-s137,syc-conv-po-lr1e5-s42,syc-icl-con-lr1e5-s137,syc-icl-con-lr1e5-s42,syc-icl-po-lr1e5-s137,syc-icl-po-lr3e5-s42,syc-pers-con-lr1e5-s137,syc-pers-po-lr1e5-s137,syc-pers-po-lr1e5-s42"

bash scripts/issue1481_dispatch.sh panel \
  --arms "$FRESH_ARMS" \
  --ckpt-map "$(cat data/issue_1481/phaseC/ckpt_map.json)" \
  --out-root "$OUT_ROOT"

for beh in writing_style impolite sycophancy; do
  bash scripts/issue1481_dispatch.sh base-arms --behavior "$beh" --out-root "$OUT_ROOT"
done

echo "[phase=done]"
