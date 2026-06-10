#!/usr/bin/env bash
# Issue #561 — pod driver: POSITIVE-ONLY marker re-train (rig disentangle).
#
# Single-variable re-run of the #519 marker arm with `--n-negs-per-persona 0`
# (200 positives, ZERO contrastive negatives) — everything else verbatim —
# then layer-{7,14,21} shift extraction over the same 14x20 panel for the
# 9 new cells (3 seeds x 3 text flavors). Comparison vs the persisted #551
# tensors runs OFF-POD (scripts/issue561_compare.py).
#
# Sequence (plan #561 §4.1; pattern: run_issue551_extract.sh):
#   1. preflight (check_code_sync=False; branch-pinned pod)
#   2. pool fetch + HARD asserts (600 rows, SHA256 pin, marker id 83399,
#      HF subfolder prefix != issue_519)
#   3. A23 build data, positives-only (dispatcher phase a23, --arms marker
#      --n-negs-per-persona 0) + post-build asserts (200 rows, all positive,
#      probe-question disjointness)
#   4. B0 saturation smoke gate (inherited verbatim; 50-step smoke train +
#      issue_519_marker_gate_eval.py; band [0.05, 12.0] nat from the config's
#      saturation_gate block). lr-ladder auto-retry FORBIDDEN: out-of-band ->
#      the dispatcher raises and this driver fail_louds (stop + diagnose).
#   5. B production train, 3 seeds in parallel on 3 GPUs (dispatcher phase b,
#      --hf-subfolder-prefix issue_561_posonly + private-repo fallback)
#   6. provenance assert (run_result.json hf_adapter_subfolder prefix)
#   7. extraction smoke on the OLD #519 contrastive marker_seed42 adapter
#      (SEPARATE dir) + the #551 4-clause reproduction gate
#   8. C extraction, 9 new cells (3 variants x 3 seeds, 4-way shard)
#   9. uploads BEFORE termination: shift tensors -> PRIVATE data repo (9+9
#      verify), training JSONLs -> private repo, per-step checkpoint adapters
#      + final-adapter presence verify -> model repo (private fallback)
#  10. end-of-run results sentinel (poll_pipeline.py contract) + [phase=done]
#
# On ANY failure: [phase=fail] + a failure sentinel; the POD IS KEPT ALIVE
# (this script only exits) so the orchestrator can inspect / re-launch.
# Pod-side: NEVER shells scripts/task.py (CLAUDE.md rule).
#
# Launch (pod, branch issue-561, after `pod.py provision --issue 561`):
#   nohup bash scripts/run_issue561_posonly.sh \
#     >> /workspace/logs/issue-561-posonly.log 2>&1 &
#
# DRY_RUN=1 bash scripts/run_issue561_posonly.sh   # echo-trace, no execution

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
if [[ "$DRY_RUN" != "1" ]]; then
  cd "$REPO_ROOT"
  mkdir -p /workspace/logs
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

N_GPUS="${N_GPUS:-4}"
OUTPUT_DIR="eval_results/issue_561"
SMOKE_DIR="eval_results/issue_561_extract_smoke"
PARENT_INPUTS="eval_results/issue_521/inputs"
PARENT_SVD="eval_results/issue_521/svd"
HF_MODEL_REPO="superkaiba1/explore-persona-space"
HF_DATA_REPO_PRIVATE="superkaiba1/explore-persona-space-data-private"
# UPLOAD_PREFIX override exists so a VM-side smoke can drive the SAME
# upload/verify code against a temporary smoke prefix.
HF_TENSOR_PREFIX="${UPLOAD_PREFIX:-issue561_posonly/analysis_tensors/shifts}"
HF_DATA_PREFIX="issue561_posonly/data"
HF_SUBFOLDER_PREFIX="issue_561_posonly"
WANDB_PROJECT="explore-persona-space-issue-561"
POOL_LOCAL="data/leakage_experiment/marker_villain_asst_excluded_medium.jsonl"
POOL_HF_PATH="leakage/marker_villain_asst_excluded_medium.jsonl"
POOL_SHA256="32f90879faa8c45ce30b5e3331ba8d507078fd9c30a6aa2452c6bc9ce9b17643"
SEEDS=(42 137 256)
CELLS=(marker_seed42 marker_seed137 marker_seed256)

phase() { echo "[phase=$1] $(date -Is) ${2:-}"; }

write_failure_sentinel() { # args: rc failure_class reason
  local rc="$1" fclass="$2" reason="$3"
  local epoch sentinel
  epoch="$(date +%s)"
  sentinel="/workspace/logs/issue-561-epm_failure-${epoch}.json"
  uv run python - "$sentinel" "$rc" "$fclass" "$reason" <<'PY' || true
import json, sys, time

sentinel, rc, fclass, reason = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "task_id": 561,
    "by": "run_issue561_posonly.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps({"failure_class": fclass, "rc": rc, "reason": reason}),
}
with open(sentinel, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote failure sentinel: {sentinel}")
PY
}

fail_loud() { # args: rc failure_class reason — pod KEPT ALIVE, script exits
  local rc="$1" fclass="$2" reason="$3"
  phase fail "rc=$rc class=$fclass reason=$reason"
  if [[ "$DRY_RUN" != "1" ]]; then
    write_failure_sentinel "$rc" "$fclass" "$reason"
  fi
  exit "$rc"
}

run_cmd() { # args: desc, then argv...
  local desc="$1"
  shift
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $desc :: $*"
    return 0
  fi
  "$@"
}

# Hard assert: this run must NEVER write the #519 comparison adapters' path.
if [[ "$HF_SUBFOLDER_PREFIX" == "issue_519" ]]; then
  fail_loud 2 code hf_subfolder_prefix_must_not_be_issue_519
fi

# ── 1. preflight ──────────────────────────────────────────────────────
# check_code_sync=False: run pods are deliberately pinned to the reviewed
# issue-561 branch HEAD, so the CLI's "behind origin/main" check is a false
# positive here. All other checks (GPU, disk quota probe, env vars, HF/WandB
# reachability) still gate.
phase preflight "orchestrate.preflight (check_code_sync=False; branch-pinned pod)"
if ! run_cmd "preflight" uv run python -c '
import sys
from explore_persona_space.orchestrate.preflight import preflight_check
report = preflight_check(check_code_sync=False)
print(report.summary())
sys.exit(0 if report.ok else 1)
'; then
  fail_loud 2 infra preflight_failed
fi

# ── 2. pool fetch + hard asserts ──────────────────────────────────────
phase pool_fetch "hf://${HF_DATA_REPO_PRIVATE%-private}/${POOL_HF_PATH} -> ${POOL_LOCAL} + SHA256/600-row/marker-id asserts"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] pool fetch :: hf_hub_download + sha256==${POOL_SHA256:0:12}... + 600 rows + encode(' ※')==[83399]"
else
  POOL_RC=0
  uv run python - "$REPO_ROOT" "$POOL_HF_PATH" "$POOL_LOCAL" "$POOL_SHA256" <<'PY' || POOL_RC=$?
import hashlib
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

repo_root, hf_path, local_rel, want_sha = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
# Explicit .env path: no-arg find_dotenv() crashes from stdin contexts.
load_dotenv(str(Path(repo_root) / ".env"))

from huggingface_hub import hf_hub_download  # noqa: E402

local = Path(repo_root) / local_rel
local.parent.mkdir(parents=True, exist_ok=True)
src = hf_hub_download(
    repo_id="superkaiba1/explore-persona-space-data",
    filename=hf_path,
    repo_type="dataset",
)
shutil.copy2(src, local)

sha = hashlib.sha256(local.read_bytes()).hexdigest()
assert sha == want_sha, f"pool SHA256 mismatch: got {sha}, want {want_sha}"
n_rows = sum(1 for line in local.open() if line.strip())
assert n_rows == 600, f"pool row count {n_rows} != 600"

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
enc = tok.encode(" ※", add_special_tokens=False)
assert enc == [83399], f"marker tokenization changed: {enc} != [83399]"
print(f"pool OK: sha256={sha[:12]}..., 600 rows, marker id [83399]")
PY
  if (( POOL_RC != 0 )); then
    fail_loud 2 data pool_fetch_or_assert_failed
  fi
fi

# ── 3. A23 build data (positives-only) via the unified dispatcher ─────
phase a23_build "dispatcher phase a23, --arms marker --n-negs-per-persona 0 (3 seeds)"
if ! run_cmd "a23 build" uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --arms marker \
  --skip-phase a1 b0_smoke b c d e \
  --n-positives 200 \
  --n-negs-per-persona 0 \
  --marker-question-pool "$POOL_LOCAL" \
  --output-dir "$OUTPUT_DIR" \
  --n-gpus 1; then
  fail_loud 2 code a23_build_failed
fi

phase build_asserts "200 rows/seed, all row_kind==positive, probe-question disjointness"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] build asserts :: len==200 && all positive && probe ∩ train == ∅ per seed"
else
  BUILD_RC=0
  uv run python - "$REPO_ROOT" "$PARENT_INPUTS/questions.json" "${SEEDS[@]}" <<'PY' || BUILD_RC=$?
import json
import sys
from pathlib import Path

repo_root, probe_json = Path(sys.argv[1]), sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]]
with (repo_root / probe_json).open() as f:
    probe_questions = set(json.load(f))
assert len(probe_questions) == 20, f"probe pool has {len(probe_questions)} != 20 questions"

for seed in seeds:
    path = repo_root / "data" / "issue_519" / f"marker_seed{seed}.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    assert len(rows) == 200, f"seed {seed}: {len(rows)} rows != 200"
    kinds = {r["row_kind"] for r in rows}
    assert kinds == {"positive"}, f"seed {seed}: row_kind set {kinds} != {{'positive'}}"
    train_questions = {r["prompt"][1]["content"] for r in rows}
    overlap = probe_questions & train_questions
    assert not overlap, f"seed {seed}: probe∩train overlap ({len(overlap)}): {sorted(overlap)[:3]}"
    n_marked = sum(1 for r in rows if r["completion"][0]["content"].endswith(" ※"))
    assert n_marked == 200, f"seed {seed}: only {n_marked}/200 completions end with ' ※'"
    print(f"seed {seed}: 200 positive rows, all marker-terminated, probe-disjoint OK")
PY
  if (( BUILD_RC != 0 )); then
    fail_loud 2 data build_asserts_failed
  fi
fi

# ── 4. B0 saturation smoke gate (inherited verbatim; NO lr-ladder retry) ──
phase b0_smoke_gate "50-step smoke train + gate eval, band from config saturation_gate [0.05, 12.0] nat"
if ! run_cmd "b0 smoke gate" uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --arms marker \
  --skip-phase a1 a23 b c d e \
  --output-dir "$OUTPUT_DIR" \
  --hf-subfolder-prefix "$HF_SUBFOLDER_PREFIX" \
  --wandb-project "$WANDB_PROJECT" \
  --n-gpus 1; then
  # Plan #561 §7.1: out-of-band -> STOP + diagnose. The #519 lr-ladder
  # auto-retry is FORBIDDEN here (an lr change voids the single-variable
  # contrast); any lr change is a plan amendment.
  fail_loud 2 code b0_saturation_gate_failed_NO_LR_LADDER_RETRY
fi

# ── 5. B production train, 3 seeds in parallel (3 GPUs) ───────────────
phase b_train "dispatcher phase b: 3 seeds x 600 steps, prefix=${HF_SUBFOLDER_PREFIX}"
if ! run_cmd "production train" uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --arms marker \
  --skip-phase a1 a23 b0_smoke c d e \
  --cells "${CELLS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --hf-subfolder-prefix "$HF_SUBFOLDER_PREFIX" \
  --hf-fallback-repo "$HF_DATA_REPO_PRIVATE" \
  --wandb-project "$WANDB_PROJECT" \
  --n-gpus 3; then
  fail_loud 2 code production_train_failed
fi

# ── 6. provenance assert (BEFORE any new-cell extraction) ─────────────
phase provenance_assert "run_result.json hf_adapter_subfolder startswith issue_561_posonly/ or issue561_posonly/"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] provenance assert :: marker_seed{42,137,256}/run_result.json prefix check"
else
  PROV_RC=0
  uv run python - "$REPO_ROOT/$OUTPUT_DIR" "${SEEDS[@]}" <<'PY' || PROV_RC=$?
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]]
ALLOWED = ("issue_561_posonly/", "issue561_posonly/")
for seed in seeds:
    rr = output_dir / f"marker_seed{seed}" / "run_result.json"
    assert rr.exists(), f"missing {rr} — trainer did not finish this cell"
    payload = json.loads(rr.read_text())
    sub = payload.get("hf_adapter_subfolder")
    assert sub and sub.startswith(ALLOWED), (
        f"seed {seed}: hf_adapter_subfolder={sub!r} does not start with {ALLOWED} — "
        f"this cell is NOT a #561 positive-only adapter (stale/staged contrastive?)"
    )
    assert not sub.startswith("issue_519"), f"seed {seed}: subfolder {sub!r} is the #519 path"
    adapter = output_dir / f"marker_seed{seed}" / "adapter" / "adapter_model.safetensors"
    assert adapter.exists() and adapter.stat().st_size > 1024, f"missing/empty {adapter}"
    print(f"seed {seed}: provenance OK ({sub}, fallback={payload.get('hf_adapter_upload_fallback')})")
PY
  if (( PROV_RC != 0 )); then
    fail_loud 2 code provenance_assert_failed
  fi
fi

# ── 7. extraction smoke on the OLD #519 contrastive adapter ───────────
# SEPARATE output dir — the cell name marker_seed42 collides with the new
# arm's; staging into $OUTPUT_DIR would shadow the freshly-trained adapter.
phase extract_smoke_stage "old #519 marker_seed42 -> ${SMOKE_DIR}"
if ! run_cmd "stage smoke adapter" uv run python scripts/issue_521_stage_adapters.py \
  --output-dir "$SMOKE_DIR" \
  --cells marker_seed42; then
  fail_loud 2 infra extract_smoke_stage_failed
fi

phase extract_smoke_cell "marker_seed42 / same variant via the unified dispatcher"
if ! run_cmd "smoke cell" uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --skip-phase a1 a23 b0_smoke b d e \
  --layers 7 14 21 \
  --variants same \
  --cells marker_seed42 \
  --output-dir "$SMOKE_DIR" \
  --personas-json "$PARENT_INPUTS/personas.json" \
  --questions-json "$PARENT_INPUTS/questions.json" \
  --n-gpus 1; then
  fail_loud 2 code extract_smoke_dispatch_failed
fi

phase extract_smoke_gate "4-clause reproduction gate vs ${PARENT_SVD}/same_marker_seed42.json"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] smoke gate :: |d s_top1|<=0.05 + |d mean_cos|<=0.05 + |cos(U1)|>=0.95 + spearman>=0.8"
else
  GATE_RC=0
  uv run python - "$SMOKE_DIR" "$PARENT_SVD" <<'PY' || GATE_RC=$?
import json
import sys
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.svd_direction_constancy import (
    assemble_M,
    cosine,
    spearman_rho,
    svd_summary,
)

output_dir, parent_svd = Path(sys.argv[1]), Path(sys.argv[2])
pt_path = output_dir / "shifts" / "same_marker_seed42.pt"
payload = torch.load(pt_path, map_location="cpu", weights_only=False)
shifts = payload["shifts"]
manifest = payload["manifest"]
assert manifest.get("schema_version") == 2, f"schema_version != 2: {manifest.get('schema_version')}"

required = {
    "delta_v",
    "delta_v_per_q",
    "delta_v_mean_resp",
    "delta_v_mean_resp_per_q",
    "delta_v_l7",
    "delta_v_l21",
    "n_questions_kept",
}
for p, entry in shifts.items():
    missing = required - set(entry.keys())
    assert not missing, f"persona {p}: missing keys {missing}"
    h = entry["delta_v"].shape[0]
    assert entry["delta_v"].shape == (h,), entry["delta_v"].shape
    n_kept = int(entry["n_questions_kept"])
    assert entry["delta_v_per_q"].shape == (n_kept, h), entry["delta_v_per_q"].shape

with (parent_svd / "same_marker_seed42.json").open() as f:
    parent = json.load(f)
persona_order = list(parent["persona_order"])
M, order = assemble_M(shifts, persona_order=persona_order)
svd = svd_summary(M)
mean_cos_re = float(np.mean(svd["cos_to_U1"]))
u1_cos_signed = cosine(svd["U1"], np.asarray(parent["U1"], dtype=np.float64))
sgn = -1.0 if u1_cos_signed < 0 else 1.0
profile_rho = spearman_rho(sgn * np.asarray(svd["cos_to_U1"]), np.asarray(parent["cos_to_U1"]))

clauses = {
    "d_s_top1_frac": abs(svd["s_top1_frac"] - parent["s_top1_frac"]),
    "d_mean_cos_to_U1": abs(mean_cos_re - parent["mean_cos_to_U1"]),
    "abs_cos_U1_re_parent": abs(u1_cos_signed),
    "profile_spearman": float(profile_rho),
}
ok = (
    clauses["d_s_top1_frac"] <= 0.05
    and clauses["d_mean_cos_to_U1"] <= 0.05
    and clauses["abs_cos_U1_re_parent"] >= 0.95
    and clauses["profile_spearman"] >= 0.8
)
result = {
    "gate": "smoke_reproduction",
    "cell": "same_marker_seed42",
    "pass": ok,
    "clauses": clauses,
    "re": {"s_top1_frac": float(svd["s_top1_frac"]), "mean_cos_to_U1": mean_cos_re},
    "parent": {
        "s_top1_frac": parent["s_top1_frac"],
        "mean_cos_to_U1": parent["mean_cos_to_U1"],
    },
}
out = output_dir / "smoke_gate_result.json"
out.write_text(json.dumps(result, indent=2))
print(f"smoke gate: pass={ok} clauses={clauses}")
if not ok:
    raise SystemExit(2)
PY
  if (( GATE_RC != 0 )); then
    fail_loud 2 code extraction_reproduction_gate_FAIL
  fi
fi

# ── 8. C extraction, 9 new cells (3 variants x 3 seeds) ───────────────
phase c_extract "9 cells (3 variants x 3 seeds) on ${N_GPUS} GPUs"
if ! run_cmd "c extraction" uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --arms marker \
  --skip-phase a1 a23 b0_smoke b d e \
  --layers 7 14 21 \
  --variants same base on_policy \
  --cells "${CELLS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --personas-json "$PARENT_INPUTS/personas.json" \
  --questions-json "$PARENT_INPUTS/questions.json" \
  --n-gpus "$N_GPUS"; then
  fail_loud 2 code c_extraction_failed
fi

# Local completeness check before the upload even starts.
phase count_check "expect 9 .pt + 9 .manifest.json under ${OUTPUT_DIR}/shifts"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] count check :: ls ${OUTPUT_DIR}/shifts/*.pt | wc -l == 9"
else
  n_pt="$(find "${OUTPUT_DIR}/shifts" -maxdepth 1 -name '*.pt' | wc -l)"
  n_mf="$(find "${OUTPUT_DIR}/shifts" -maxdepth 1 -name '*.manifest.json' | wc -l)"
  if [[ "$n_pt" != "9" || "$n_mf" != "9" ]]; then
    fail_loud 2 code "local_shift_count_mismatch_pt=${n_pt}_manifest=${n_mf}"
  fi
fi

# ── 9a. upload shift tensors -> PRIVATE data repo + fail-loud verify ──
phase upload_tensors "upload_folder ${OUTPUT_DIR}/shifts -> hf://${HF_DATA_REPO_PRIVATE}/${HF_TENSOR_PREFIX} (expect 9+9)"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] upload :: issue551_upload_verify.py --expected-count 9"
else
  UPLOAD_RC=0
  uv run python scripts/issue551_upload_verify.py \
    "$OUTPUT_DIR" "$HF_DATA_REPO_PRIVATE" "$HF_TENSOR_PREFIX" --expected-count 9 || UPLOAD_RC=$?
  if (( UPLOAD_RC != 0 )); then
    # Pod KEPT ALIVE on upload-verify miss; the tensors exist only on this
    # pod until the upload lands.
    fail_loud 2 infra tensor_upload_verify_failed_POD_KEPT_ALIVE
  fi
fi

# ── 9b. upload training JSONLs -> private data repo ───────────────────
phase upload_data "3 training JSONLs (+manifests) -> hf://${HF_DATA_REPO_PRIVATE}/${HF_DATA_PREFIX}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] upload data :: marker_seed{42,137,256}.jsonl + .manifest.json"
else
  DATA_RC=0
  uv run python - "$REPO_ROOT" "$HF_DATA_REPO_PRIVATE" "$HF_DATA_PREFIX" "${SEEDS[@]}" <<'PY' || DATA_RC=$?
import sys
from pathlib import Path

from dotenv import load_dotenv

repo_root, repo_id, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
seeds = [int(s) for s in sys.argv[4:]]
load_dotenv(str(repo_root / ".env"))

from huggingface_hub import HfApi, list_repo_files  # noqa: E402

api = HfApi()
for seed in seeds:
    for suffix in (".jsonl", ".manifest.json"):
        local = repo_root / "data" / "issue_519" / f"marker_seed{seed}{suffix}"
        assert local.exists(), f"missing training artifact {local}"
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"{prefix}/marker_seed{seed}{suffix}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"uploaded {local.name} -> {repo_id}/{prefix}/")
files = [f for f in list_repo_files(repo_id, repo_type="dataset") if f.startswith(prefix + "/")]
n_jsonl = sum(1 for f in files if f.endswith(".jsonl"))
assert n_jsonl >= len(seeds), f"verify FAIL: {n_jsonl} .jsonl < {len(seeds)} under {prefix}/"
print(f"verified on hub: {n_jsonl} .jsonl under {prefix}/")
PY
  if (( DATA_RC != 0 )); then
    fail_loud 2 infra data_upload_failed_POD_KEPT_ALIVE
  fi
fi

# ── 9c. checkpoint adapters + final-adapter presence verify ───────────
# The #519 post-mortem lost the per-step checkpoints at termination and
# named them load-bearing for the non-saturated-anchor follow-up.
phase upload_checkpoints "checkpoint-{50..600} adapters -> ${HF_MODEL_REPO}/${HF_SUBFOLDER_PREFIX}/marker_seed{S}/checkpoints (private fallback)"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] checkpoints :: upload_folder per seed, allow_patterns=checkpoint-*/adapter*, verify >=12/seed"
else
  CKPT_RC=0
  uv run python - "$REPO_ROOT/$OUTPUT_DIR" "$HF_MODEL_REPO" "$HF_DATA_REPO_PRIVATE" "$HF_SUBFOLDER_PREFIX" "${SEEDS[@]}" <<'PY' || CKPT_RC=$?
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

output_dir = Path(sys.argv[1])
model_repo, fallback_repo, prefix = sys.argv[2], sys.argv[3], sys.argv[4]
seeds = [int(s) for s in sys.argv[5:]]
load_dotenv(str(output_dir.parents[1] / ".env"))

from huggingface_hub import HfApi, list_repo_files  # noqa: E402

api = HfApi()
ALLOW = [
    "checkpoint-*/adapter_model.safetensors",
    "checkpoint-*/adapter_config.json",
    "checkpoint-*/trainer_state.json",
]

for seed in seeds:
    cell_dir = output_dir / f"marker_seed{seed}"
    n_local = len(list(cell_dir.glob("checkpoint-*/adapter_model.safetensors")))
    assert n_local >= 12, (
        f"seed {seed}: only {n_local} local checkpoint adapters (< 12 = 600 steps / save_steps 50)"
    )
    path_in_repo = f"{prefix}/marker_seed{seed}/checkpoints"
    used_repo, used_type = model_repo, "model"
    try:
        api.upload_folder(
            folder_path=str(cell_dir),
            repo_id=model_repo,
            repo_type="model",
            path_in_repo=path_in_repo,
            allow_patterns=ALLOW,
        )
    except Exception as e:
        print(f"seed {seed}: model-repo checkpoint upload failed ({type(e).__name__}: {e}); "
              f"falling back to private dataset repo (#552/#541 LFS-quota recovery)")
        api.upload_folder(
            folder_path=str(cell_dir),
            repo_id=fallback_repo,
            repo_type="dataset",
            path_in_repo=f"{prefix}/adapters/marker_seed{seed}/checkpoints",
            allow_patterns=ALLOW,
        )
        used_repo, used_type = fallback_repo, "dataset"
        path_in_repo = f"{prefix}/adapters/marker_seed{seed}/checkpoints"
    files = [
        f
        for f in list_repo_files(used_repo, repo_type=used_type)
        if f.startswith(path_in_repo + "/") and f.endswith("adapter_model.safetensors")
    ]
    assert len(files) >= 12, (
        f"seed {seed}: checkpoint verify FAIL — {len(files)} adapters on {used_repo} "
        f"under {path_in_repo}/ (< 12)"
    )
    print(f"seed {seed}: {len(files)} checkpoint adapters verified on {used_repo}")

    # Final-adapter presence verify (trainer already pushed it; re-verify via
    # the Hub API using the repo+subfolder recorded in run_result.json).
    rr = json.loads((cell_dir / "run_result.json").read_text())
    final_repo, final_sub = rr["hf_adapter_repo"], rr["hf_adapter_subfolder"]
    final_type = "dataset" if rr.get("hf_adapter_upload_fallback") else "model"
    final_files = [
        f
        for f in list_repo_files(final_repo, repo_type=final_type)
        if f.startswith(final_sub + "/") and f.endswith("adapter_model.safetensors")
    ]
    assert final_files, (
        f"seed {seed}: final adapter NOT on hub at {final_repo}/{final_sub} — refusing to finish"
    )
    print(f"seed {seed}: final adapter verified at {final_repo}/{final_sub}")
PY
  if (( CKPT_RC != 0 )); then
    fail_loud 2 infra checkpoint_upload_verify_failed_POD_KEPT_ALIVE
  fi
  df -h /workspace || true
fi

# ── 10. end-of-run results sentinel + [phase=done] ─────────────────────
phase write_sentinel "results sentinel for poll_pipeline.py"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] sentinel :: /workspace/logs/issue-561-epm_results-<epoch>.json"
else
  EPOCH="$(date +%s)"
  SENTINEL="/workspace/logs/issue-561-epm_results-${EPOCH}.json"
  uv run python - "$SENTINEL" "$OUTPUT_DIR" "$SMOKE_DIR" "$HF_DATA_REPO_PRIVATE" "$HF_TENSOR_PREFIX" <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path

sentinel, output_dir, smoke_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
repo_id, prefix = sys.argv[4], sys.argv[5]
gate_path = smoke_dir / "smoke_gate_result.json"
gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
b0_path = output_dir / "smoke" / "marker_seed42" / "saturation_gate_result.json"
b0 = json.loads(b0_path.read_text()) if b0_path.exists() else {}
shift_files = sorted(p.name for p in (output_dir / "shifts").glob("*.pt"))
endpoint = {}
for seed in (42, 137, 256):
    p = output_dir / f"marker_seed{seed}" / "periodic_eval" / "leakage_marker_step_600.json"
    if p.exists():
        snap = json.loads(p.read_text())
        m = snap.get("metrics_by_persona", {}).get("medical_doctor", {})
        endpoint[f"seed{seed}"] = {
            "log_p_marker_delta": m.get("log_p_marker_delta"),
            "emit_rate": m.get("emit_rate"),
            "z_margin_delta": m.get("z_margin_delta"),
        }
try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    git_commit = "unknown"

note = {
    "issue": 561,
    "phase": "posonly_train_extract_complete_and_uploaded",
    "b0_smoke_gate": b0,
    "extraction_smoke_gate": gate,
    "n_shift_pt_files": len(shift_files),
    "shift_files": shift_files,
    "hf_tensor_prefix": f"{repo_id}/{prefix}",
    "endpoint_manipulation_check_source": endpoint,
    "git_commit": git_commit,
    "next_step": (
        "VM-side: uv run python scripts/issue561_compare.py "
        "--new-shifts-dir eval_results/issue_561/shifts "
        "--out eval_results/issue_561/comparison"
    ),
}
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 561,
    "by": "run_issue561_posonly.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
with open(sentinel, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote sentinel: {sentinel}")
PY
fi

phase done "issue-561 positive-only train + extraction + persistence complete (comparison runs OFF-POD)"
