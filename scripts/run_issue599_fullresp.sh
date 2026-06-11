#!/usr/bin/env bash
# Issue #599 — pod driver: WHOLE-RESPONSE-LOSS marker re-train (rig disentangle, part 2).
#
# Single-variable re-run of the #561 positive-only marker recipe with
# `--loss-shape full_response` (the MarkerOnlyDataCollator wrap is skipped;
# TRL's prompt-completion completion-mask CE — the EM arm's loss path —
# trains every completion token incl. the sentinel) — everything else
# verbatim — then layer-{7,14,21} shift extraction over the same 14x20
# panel for the 9 new cells (3 seeds x 3 text flavors). Comparison vs the
# persisted #551 AND #561 tensors runs OFF-POD (scripts/issue599_compare.py).
#
# Sequence (plan #599 §4.2; pattern: run_issue561_posonly.sh):
#   1. preflight (check_code_sync=False; branch-pinned pod)
#   2. pool fetch + HARD asserts (600 rows, SHA256 pin, marker id 83399,
#      HF subfolder prefix not in {issue_519, issue_561_posonly})
#   3. A23 build data, positives-only (dispatcher phase a23, --arms marker
#      --n-negs-per-persona 0) + post-build asserts (200 rows, all positive,
#      probe-question disjointness)
#   4. B0 smoke gate, RE-PURPOSED per plan §7.1 (50-step smoke train WITH
#      --loss-shape full_response): the trainer's label-mask audit +
#      run_result.json provenance GATE; the inherited DG floor is a
#      diagnostic (would false-FAIL under ~255x dilution); DG ceiling kept.
#      lr-ladder auto-retry FORBIDDEN.
#   5. B production train, 3 seeds in parallel on 3 GPUs (dispatcher phase b,
#      --loss-shape full_response --hf-subfolder-prefix issue_599_fullresp
#      + private-repo fallback)
#   6. provenance assert (run_result.json subfolder prefix + loss_shape ==
#      full_response + labels_unmasked_per_row_mean >= 50) AND the step-50
#      four-float trajectory fail-loud check (all four floats per side per
#      seed — #530 storage contract, first PRODUCTION callback fire)
#   7. step-600 manipulation read + branch decision (plan §7.2/§7.3):
#      KILL = DG < 5 nat OR emit < 0.5 per seed; extraction runs EITHER way
#      (KILL-branch geometry is exploratory, labeled by the analyzer)
#   8. extraction smoke on the OLD #519 contrastive marker_seed42 adapter
#      (SEPARATE dir) + the #551 4-clause reproduction gate
#   9. C extraction, 9 new cells (3 variants x 3 seeds, 4-way shard)
#  10. uploads BEFORE termination: shift tensors -> PRIVATE data repo (9+9
#      verify), training JSONLs -> private repo, per-step checkpoint adapters
#      + final-adapter presence verify -> model repo (private fallback)
#  11. CONDITIONAL §7.3 tiered extension (pre-registered KILL branch):
#      probe trigger = ALL-seed KILL AND step-600 DG >= 0.3 nat on >=1 seed.
#      Probe: seed 42 fresh train --max-steps 2400 (stretched cosine — a
#      NAMED schedule deviation), checkpoints every 100, callback K=50.
#      Escalate only if the probe reaches non-KILL by 2400: seeds 137/256
#      to 2400 (parallel), then extract 9 cells at each seed's FIRST
#      non-KILL checkpoint (prefer first FULL), under eval_results/
#      issue_599_ext + HF issue_599_fullresp_ext. lr changes FORBIDDEN.
#  12. end-of-run results sentinel (poll_pipeline.py contract) + [phase=done]
#
# On ANY failure: [phase=fail] + a failure sentinel; the POD IS KEPT ALIVE
# (this script only exits) so the orchestrator can inspect / re-launch.
# Pod-side: NEVER shells scripts/task.py (CLAUDE.md rule).
#
# Launch (pod, branch issue-599, after `pod.py provision --issue 599`):
#   nohup bash scripts/run_issue599_fullresp.sh \
#     >> /workspace/logs/issue-599-fullresp.log 2>&1 &
#
# DRY_RUN=1 bash scripts/run_issue599_fullresp.sh   # echo-trace, no execution

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
OUTPUT_DIR="eval_results/issue_599"
SMOKE_DIR="eval_results/issue_599_extract_smoke"
EXT_DIR="eval_results/issue_599_ext"
PARENT_INPUTS="eval_results/issue_521/inputs"
PARENT_SVD="eval_results/issue_521/svd"
HF_MODEL_REPO="superkaiba1/explore-persona-space"
HF_DATA_REPO_PRIVATE="superkaiba1/explore-persona-space-data-private"
# UPLOAD_PREFIX override exists so a VM-side smoke can drive the SAME
# upload/verify code against a temporary smoke prefix.
HF_TENSOR_PREFIX="${UPLOAD_PREFIX:-issue599_fullresp/analysis_tensors/shifts}"
HF_DATA_PREFIX="issue599_fullresp/data"
HF_SUBFOLDER_PREFIX="issue_599_fullresp"
HF_SUBFOLDER_PREFIX_EXT="issue_599_fullresp_ext"
HF_TENSOR_PREFIX_EXT="issue599_fullresp_ext/analysis_tensors/shifts"
WANDB_PROJECT="explore-persona-space-issue-599"
LOSS_SHAPE="full_response"
POOL_LOCAL="data/leakage_experiment/marker_villain_asst_excluded_medium.jsonl"
POOL_HF_PATH="leakage/marker_villain_asst_excluded_medium.jsonl"
POOL_SHA256="32f90879faa8c45ce30b5e3331ba8d507078fd9c30a6aa2452c6bc9ce9b17643"
SEEDS=(42 137 256)
CELLS=(marker_seed42 marker_seed137 marker_seed256)
EXT_MAX_STEPS=2400
EXT_SAVE_STEPS=100

phase() { echo "[phase=$1] $(date -Is) ${2:-}"; }

write_failure_sentinel() { # args: rc failure_class reason
  local rc="$1" fclass="$2" reason="$3"
  local epoch sentinel
  epoch="$(date +%s)"
  sentinel="/workspace/logs/issue-599-epm_failure-${epoch}.json"
  uv run python - "$sentinel" "$rc" "$fclass" "$reason" <<'PY' || true
import json, sys, time

sentinel, rc, fclass, reason = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "task_id": 599,
    "by": "run_issue599_fullresp.sh",
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

# Hard assert: this run must NEVER write the #519 or #561 comparison
# adapters' paths (plan §4.1.3 — both are reference provenance).
if [[ "$HF_SUBFOLDER_PREFIX" == "issue_519" || "$HF_SUBFOLDER_PREFIX" == "issue_561_posonly" ]]; then
  fail_loud 2 code hf_subfolder_prefix_must_not_collide_with_519_or_561
fi

# ── 1. preflight ──────────────────────────────────────────────────────
# check_code_sync=False: run pods are deliberately pinned to the reviewed
# issue-599 branch HEAD, so the CLI's "behind origin/main" check is a false
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
phase pool_fetch "hf://superkaiba1/explore-persona-space-data/${POOL_HF_PATH} -> ${POOL_LOCAL} + SHA256/600-row/marker-id asserts"
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

# ── 4. B0 smoke gate (RE-PURPOSED per plan §7.1; NO lr-ladder retry) ──
# Gating clauses for the full_response arm: trainer label-mask audit
# (in-process assert) + run_result.json provenance (loss_shape +
# labels_unmasked_per_row_mean in [50, 540], enforced by the dispatcher
# gate) + DG ceiling <= 12 nat. The inherited DG floor is a DIAGNOSTIC
# (the dilution prediction ~0.004 nat sits far below it by construction).
phase b0_smoke_gate "50-step smoke train --loss-shape ${LOSS_SHAPE} + re-purposed gate (mask-audit provenance gates; DG floor diagnostic)"
if ! run_cmd "b0 smoke gate" uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --arms marker \
  --skip-phase a1 a23 b c d e \
  --output-dir "$OUTPUT_DIR" \
  --hf-subfolder-prefix "$HF_SUBFOLDER_PREFIX" \
  --wandb-project "$WANDB_PROJECT" \
  --loss-shape "$LOSS_SHAPE" \
  --n-gpus 1; then
  # Plan #599 §7.1: out-of-band / provenance miss -> STOP + diagnose. The
  # #519 lr-ladder auto-retry is FORBIDDEN here (an lr change voids the
  # single-variable contrast); any lr change is a plan amendment.
  fail_loud 2 code b0_gate_failed_NO_LR_LADDER_RETRY
fi

# Prefix + loss-shape threading verification at B0 (plan §7.1d): the B0
# path runs no_hf_upload=True (never touches the Hub), so the deterministic
# thread check reads the smoke cell's run_result.json (the trainer records
# hf_subfolder_prefix + loss_shape there regardless of upload) AND greps
# the trainer log for the full_response wiring line. The on-Hub prefix
# check happens at the first production upload (steps 5/6/10c below).
phase b0_thread_check "smoke run_result.json carries loss_shape=${LOSS_SHAPE} + hf_subfolder_prefix=${HF_SUBFOLDER_PREFIX}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] thread check :: smoke/marker_seed42/run_result.json loss_shape + hf_subfolder_prefix + trainer-log wiring line"
else
  THREAD_RC=0
  uv run python - "$OUTPUT_DIR/smoke/marker_seed42/run_result.json" "$LOSS_SHAPE" "$HF_SUBFOLDER_PREFIX" <<'PY' || THREAD_RC=$?
import json
import sys

rr_path, want_shape, want_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
rr = json.load(open(rr_path))
assert rr.get("loss_shape") == want_shape, (
    f"B0 thread check FAIL: loss_shape={rr.get('loss_shape')!r} != {want_shape!r}"
)
assert rr.get("hf_subfolder_prefix") == want_prefix, (
    f"B0 thread check FAIL: hf_subfolder_prefix={rr.get('hf_subfolder_prefix')!r} "
    f"!= {want_prefix!r} — the production upload would land on the wrong Hub path"
)
print(f"b0 thread check OK: loss_shape={want_shape}, prefix={want_prefix}")
PY
  if (( THREAD_RC != 0 )); then
    fail_loud 2 code b0_thread_check_failed
  fi
  B0_LOG="$OUTPUT_DIR/logs/phase_b_train_marker_seed42.log"
  if ! grep -q "loss_shape=full_response: TRL completion-mask CE retained" "$B0_LOG"; then
    fail_loud 2 code b0_trainer_log_missing_full_response_wiring_line
  fi
fi

# ── 5. B production train, 3 seeds in parallel (3 GPUs) ───────────────
phase b_train "dispatcher phase b: 3 seeds x 600 steps, --loss-shape ${LOSS_SHAPE}, prefix=${HF_SUBFOLDER_PREFIX}"
if ! run_cmd "production train" uv run python scripts/issue_519_dispatch.py \
  --mode sweep \
  --arms marker \
  --skip-phase a1 a23 b0_smoke c d e \
  --cells "${CELLS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --hf-subfolder-prefix "$HF_SUBFOLDER_PREFIX" \
  --hf-fallback-repo "$HF_DATA_REPO_PRIVATE" \
  --wandb-project "$WANDB_PROJECT" \
  --loss-shape "$LOSS_SHAPE" \
  --n-gpus 3; then
  fail_loud 2 code production_train_failed
fi

# ── 6. provenance assert + step-50 four-float check ───────────────────
phase provenance_assert "run_result.json prefix + loss_shape==full_response + unmasked>=50 + step-50 four-float trajectory per seed"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] provenance assert :: marker_seed{42,137,256}/run_result.json prefix/loss_shape/mask + leakage_marker_step_50.json four floats"
else
  PROV_RC=0
  uv run python - "$REPO_ROOT/$OUTPUT_DIR" "${SEEDS[@]}" <<'PY' || PROV_RC=$?
import json
import math
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]]
ALLOWED = ("issue_599_fullresp/", "issue599_fullresp/")
FORBIDDEN = ("issue_519", "issue_561", "issue561")
# #530 storage contract: four floats per slot per model side.
FOUR_FLOATS = [
    "log_p_marker_trained", "z_marker_trained", "z_eos_trained", "logZ_trained",
    "log_p_marker_base", "z_marker_base", "z_eos_base", "logZ_base",
]
for seed in seeds:
    rr = output_dir / f"marker_seed{seed}" / "run_result.json"
    assert rr.exists(), f"missing {rr} — trainer did not finish this cell"
    payload = json.loads(rr.read_text())
    sub = payload.get("hf_adapter_subfolder")
    assert sub and sub.startswith(ALLOWED), (
        f"seed {seed}: hf_adapter_subfolder={sub!r} does not start with {ALLOWED} — "
        f"this cell is NOT a #599 whole-response adapter (stale/staged?)"
    )
    assert not any(sub.startswith(p) for p in FORBIDDEN), (
        f"seed {seed}: subfolder {sub!r} collides with a reference-arm path"
    )
    # #599 second line of defense against the silent-no-swap false
    # falsification: the manipulation provenance must be in the manifest.
    assert payload.get("loss_shape") == "full_response", (
        f"seed {seed}: run_result.json loss_shape={payload.get('loss_shape')!r} "
        f"!= 'full_response' — the collator swap did NOT thread to this cell"
    )
    unmasked = payload.get("labels_unmasked_per_row_mean")
    assert unmasked is not None and float(unmasked) >= 50, (
        f"seed {seed}: labels_unmasked_per_row_mean={unmasked!r} < 50 — "
        f"mask signature does not match full_response"
    )
    adapter = output_dir / f"marker_seed{seed}" / "adapter" / "adapter_model.safetensors"
    assert adapter.exists() and adapter.stat().st_size > 1024, f"missing/empty {adapter}"

    # Step-50 four-float trajectory check (first PRODUCTION callback fire;
    # the B0 smoke runs skip_callbacks=True so this is the earliest point
    # the #530 storage contract is observable). Fail-loud per plan §4.1.
    snap_path = output_dir / f"marker_seed{seed}" / "periodic_eval" / "leakage_marker_step_50.json"
    assert snap_path.exists(), (
        f"seed {seed}: missing {snap_path} — the periodic callback did not fire "
        f"at production step 50 (trajectory telemetry is load-bearing for H-install)"
    )
    snap = json.loads(snap_path.read_text())
    per_persona = snap.get("metrics_by_persona", {})
    assert per_persona, f"seed {seed}: step-50 snapshot has no metrics_by_persona"
    m = per_persona.get("medical_doctor")
    assert m is not None, f"seed {seed}: step-50 snapshot missing source persona medical_doctor"
    missing = [k for k in FOUR_FLOATS if k not in m or not math.isfinite(float(m[k]))]
    assert not missing, (
        f"seed {seed}: step-50 four-float check FAIL — missing/non-finite {missing} "
        f"(#530 storage contract: logp/z_marker/z_eos/logZ per model side)"
    )
    print(
        f"seed {seed}: provenance OK ({sub}, loss_shape=full_response, "
        f"unmasked/row={float(unmasked):.1f}, initial_loss="
        f"{payload.get('initial_train_loss')}, step-50 four-float OK, "
        f"fallback={payload.get('hf_adapter_upload_fallback')})"
    )
PY
  if (( PROV_RC != 0 )); then
    fail_loud 2 code provenance_or_fourfloat_assert_failed
  fi
fi

# ── 7. step-600 manipulation read + branch decision (plan §7.2/§7.3) ──
phase manipulation_read "per-seed step-600 DG/emit -> FULL/PARTIAL/KILL + probe-trigger decision"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] manipulation read :: KILL = DG<5 nat OR emit<0.5; probe trigger = all-KILL AND any DG>=0.3"
  PROBE_TRIGGER=0
else
  MANIP_RC=0
  uv run python - "$REPO_ROOT/$OUTPUT_DIR" "${SEEDS[@]}" <<'PY' || MANIP_RC=$?
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]]
per_seed = {}
for seed in seeds:
    p = output_dir / f"marker_seed{seed}" / "periodic_eval" / "leakage_marker_step_600.json"
    assert p.exists(), f"missing {p} — step-600 manipulation check unavailable"
    snap = json.loads(p.read_text())
    m = snap["metrics_by_persona"]["medical_doctor"]
    dg = float(m["log_p_marker_delta"])
    emit = float(m["emit_rate"])
    logp_trained = float(m["log_p_marker_trained"])
    if dg < 5.0 or emit < 0.5:
        status = "KILL"
    elif logp_trained >= -0.5 and emit >= 0.95:
        status = "FULL"
    else:
        status = "PARTIAL"
    per_seed[f"seed{seed}"] = {
        "log_p_marker_delta": dg,
        "emit_rate": emit,
        "log_p_marker_trained": logp_trained,
        "z_margin_delta": float(m.get("z_margin_delta", float("nan"))),
        "status": status,
    }
all_kill = all(v["status"] == "KILL" for v in per_seed.values())
any_dg_ge_0p3 = any(v["log_p_marker_delta"] >= 0.3 for v in per_seed.values())
decision = {
    "per_seed": per_seed,
    "all_kill": all_kill,
    "any_dg_ge_0p3": any_dg_ge_0p3,
    # Plan §7.3 probe trigger: all-seed KILL AND a visible ramp (>=0.3 nat
    # ~ 2.5x the corrected dilution projection of ~0.12 nat at step 600).
    "probe_trigger": bool(all_kill and any_dg_ge_0p3),
    "n_non_kill_seeds": sum(1 for v in per_seed.values() if v["status"] != "KILL"),
    "rule": "KILL = DG < 5 nat OR emit < 0.5; FULL = trained logP >= -0.5 AND emit >= 0.95",
}
out = output_dir / "manipulation_check.json"
out.write_text(json.dumps(decision, indent=2))
print(json.dumps(decision, indent=2))
PY
  if (( MANIP_RC != 0 )); then
    fail_loud 2 code manipulation_read_failed
  fi
  PROBE_TRIGGER="$(uv run python -c "
import json
print(1 if json.load(open('$OUTPUT_DIR/manipulation_check.json'))['probe_trigger'] else 0)
")"
fi

# ── 8. extraction smoke on the OLD #519 contrastive adapter ───────────
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

# ── 9. C extraction, 9 new cells (3 variants x 3 seeds) ───────────────
# Runs on BOTH branches (plan §7.3.2: under KILL the spectrum is labeled
# "geometry of the whole-response training shift, NOT the implant's").
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

# ── 10a. upload shift tensors -> PRIVATE data repo + fail-loud verify ──
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

# ── 10b. upload training JSONLs -> private data repo ───────────────────
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

# ── 10c. checkpoint adapters + final-adapter presence verify ───────────
# The #519 post-mortem lost the per-step checkpoints at termination and
# named them load-bearing; for #599 the K=50 trajectory checkpoints ARE
# the H-install dilution measurement.
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

# ── 11. CONDITIONAL §7.3 tiered extension (pre-registered KILL branch) ──
# Probe trigger (computed at phase manipulation_read): all-seed KILL AND
# step-600 DG >= 0.3 nat on >=1 seed. Below trigger: 4x steps projects
# << 5 nat — report non-installability with no extension spend.
EXT_RAN=0
EXT_ESCALATED=0
if [[ "$DRY_RUN" != "1" && "$PROBE_TRIGGER" == "1" ]]; then
  EXT_RAN=1
  mkdir -p "$EXT_DIR"
  phase ext_probe "seed 42 fresh train --max-steps ${EXT_MAX_STEPS} (stretched cosine — NAMED deviation), save_steps ${EXT_SAVE_STEPS}, callback K=50 -> ${EXT_DIR}/ext_train_seed42.log"
  # Redirected like the escalation seeds (#545: the trainer's terminal
  # "[phase=done] wrote adapter ..." line must NOT reach this main polled
  # log mid-run — poll_pipeline reserves [phase=done] for the driver's
  # single terminal line).
  if ! uv run python scripts/issue_519_train.py \
    --arm marker \
    --seed 42 \
    --data-path "data/issue_519/marker_seed42.jsonl" \
    --output-dir "$EXT_DIR/marker_seed42" \
    --gpu-id 0 \
    --max-steps "$EXT_MAX_STEPS" \
    --save-steps-override "$EXT_SAVE_STEPS" \
    --loss-shape "$LOSS_SHAPE" \
    --hf-subfolder-prefix "$HF_SUBFOLDER_PREFIX_EXT" \
    --hf-fallback-repo "$HF_DATA_REPO_PRIVATE" \
    --wandb-project "$WANDB_PROJECT" \
    > "$EXT_DIR/ext_train_seed42.log" 2>&1; then
    fail_loud 2 code ext_probe_train_failed
  fi

  phase ext_probe_read "first non-KILL checkpoint on the seed-42 probe trajectory (escalate only if reached by ${EXT_MAX_STEPS})"
  if ! uv run python scripts/issue599_ext_read.py probe-read \
    --ext-dir "$EXT_DIR" --seed 42 \
    --save-steps "$EXT_SAVE_STEPS" --max-steps "$EXT_MAX_STEPS"; then
    fail_loud 2 code ext_probe_read_failed
  fi
  PROBE_CLEARED="$(uv run python -c "
import json
print(1 if json.load(open('$EXT_DIR/probe_read.json'))['probe_cleared'] else 0)
")"

  if [[ "$PROBE_CLEARED" == "1" ]]; then
    EXT_ESCALATED=1
    phase ext_escalate_train "seeds 137 + 256 -> ${EXT_MAX_STEPS} steps (parallel, 2 GPUs)"
    EXT_PIDS=()
    EXT_SEEDS_ESC=(137 256)
    for i in "${!EXT_SEEDS_ESC[@]}"; do
      s="${EXT_SEEDS_ESC[$i]}"
      uv run python scripts/issue_519_train.py \
        --arm marker \
        --seed "$s" \
        --data-path "data/issue_519/marker_seed${s}.jsonl" \
        --output-dir "$EXT_DIR/marker_seed${s}" \
        --gpu-id "$i" \
        --max-steps "$EXT_MAX_STEPS" \
        --save-steps-override "$EXT_SAVE_STEPS" \
        --loss-shape "$LOSS_SHAPE" \
        --hf-subfolder-prefix "$HF_SUBFOLDER_PREFIX_EXT" \
        --hf-fallback-repo "$HF_DATA_REPO_PRIVATE" \
        --wandb-project "$WANDB_PROJECT" \
        > "$EXT_DIR/ext_train_seed${s}.log" 2>&1 &
      EXT_PIDS+=("$!")
    done
    EXT_TRAIN_FAIL=0
    for i in "${!EXT_PIDS[@]}"; do
      if ! wait "${EXT_PIDS[$i]}"; then
        echo "ext escalation seed ${EXT_SEEDS_ESC[$i]} train FAILED (see $EXT_DIR/ext_train_seed${EXT_SEEDS_ESC[$i]}.log)"
        EXT_TRAIN_FAIL=1
      fi
    done
    if (( EXT_TRAIN_FAIL != 0 )); then
      fail_loud 2 code ext_escalation_train_failed
    fi

    phase ext_stage_band_entry "stage each seed's FIRST non-KILL (prefer FULL) checkpoint -> ${EXT_DIR}/extract/marker_seed{S}/adapter"
    if ! uv run python scripts/issue599_ext_read.py stage-band-entry \
      --ext-dir "$EXT_DIR" --save-steps "$EXT_SAVE_STEPS" --seeds "${SEEDS[@]}"; then
      fail_loud 2 code ext_band_entry_staging_failed
    fi

    phase ext_c_extract "extension cells (staged band-entry adapters) x 3 variants on ${N_GPUS} GPUs"
    EXT_CELLS=()
    for s in "${SEEDS[@]}"; do
      if [[ -d "$EXT_DIR/extract/marker_seed${s}/adapter" ]]; then
        EXT_CELLS+=("marker_seed${s}")
      fi
    done
    if ! uv run python scripts/issue_519_dispatch.py \
      --mode sweep \
      --arms marker \
      --skip-phase a1 a23 b0_smoke b d e \
      --layers 7 14 21 \
      --variants same base on_policy \
      --cells "${EXT_CELLS[@]}" \
      --output-dir "$EXT_DIR/extract" \
      --personas-json "$PARENT_INPUTS/personas.json" \
      --questions-json "$PARENT_INPUTS/questions.json" \
      --n-gpus "$N_GPUS"; then
      fail_loud 2 code ext_c_extraction_failed
    fi

    phase ext_upload_tensors "extension shift tensors -> hf://${HF_DATA_REPO_PRIVATE}/${HF_TENSOR_PREFIX_EXT}"
    EXT_N_EXPECTED=$(( ${#EXT_CELLS[@]} * 3 ))
    EXT_UPLOAD_RC=0
    uv run python scripts/issue551_upload_verify.py \
      "$EXT_DIR/extract" "$HF_DATA_REPO_PRIVATE" "$HF_TENSOR_PREFIX_EXT" \
      --expected-count "$EXT_N_EXPECTED" || EXT_UPLOAD_RC=$?
    if (( EXT_UPLOAD_RC != 0 )); then
      fail_loud 2 infra ext_tensor_upload_verify_failed_POD_KEPT_ALIVE
    fi
  else
    phase ext_probe_not_cleared "probe never reached non-KILL by step ${EXT_MAX_STEPS} — determinate: not installable within 4x reference steps at lr 2e-6 (lr raise is a FOLLOW-UP, never in-run)"
  fi

  # Extension checkpoints (probe + any escalation seeds) -> Hub before
  # termination; the trajectory checkpoints ARE the H-install measurement.
  phase ext_upload_checkpoints "extension checkpoint adapters -> ${HF_MODEL_REPO}/${HF_SUBFOLDER_PREFIX_EXT}/marker_seed{S}/checkpoints (private fallback)"
  EXT_CKPT_RC=0
  uv run python - "$REPO_ROOT/$EXT_DIR" "$HF_MODEL_REPO" "$HF_DATA_REPO_PRIVATE" "$HF_SUBFOLDER_PREFIX_EXT" <<'PY' || EXT_CKPT_RC=$?
import sys
from pathlib import Path

from dotenv import load_dotenv

ext_dir = Path(sys.argv[1])
model_repo, fallback_repo, prefix = sys.argv[2], sys.argv[3], sys.argv[4]
load_dotenv(str(ext_dir.parents[1] / ".env"))

from huggingface_hub import HfApi, list_repo_files  # noqa: E402

api = HfApi()
ALLOW = [
    "checkpoint-*/adapter_model.safetensors",
    "checkpoint-*/adapter_config.json",
    "checkpoint-*/trainer_state.json",
]
cells = sorted(d for d in ext_dir.glob("marker_seed*") if d.is_dir())
assert cells, f"no extension cells under {ext_dir}"
for cell_dir in cells:
    n_local = len(list(cell_dir.glob("checkpoint-*/adapter_model.safetensors")))
    if n_local == 0:
        continue
    path_in_repo = f"{prefix}/{cell_dir.name}/checkpoints"
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
        print(f"{cell_dir.name}: model-repo upload failed ({type(e).__name__}: {e}); "
              f"falling back to private dataset repo")
        path_in_repo = f"{prefix}/adapters/{cell_dir.name}/checkpoints"
        api.upload_folder(
            folder_path=str(cell_dir),
            repo_id=fallback_repo,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            allow_patterns=ALLOW,
        )
        used_repo, used_type = fallback_repo, "dataset"
    files = [
        f
        for f in list_repo_files(used_repo, repo_type=used_type)
        if f.startswith(path_in_repo + "/") and f.endswith("adapter_model.safetensors")
    ]
    assert len(files) >= n_local, (
        f"{cell_dir.name}: checkpoint verify FAIL — {len(files)} on hub < {n_local} local"
    )
    print(f"{cell_dir.name}: {len(files)} extension checkpoint adapters verified on {used_repo}")
PY
  if (( EXT_CKPT_RC != 0 )); then
    fail_loud 2 infra ext_checkpoint_upload_verify_failed_POD_KEPT_ALIVE
  fi
elif [[ "$DRY_RUN" != "1" ]]; then
  phase ext_skipped "probe trigger not met (manipulation_check.json: probe_trigger=false) — no extension spend"
fi

# ── 12. end-of-run results sentinel + [phase=done] ─────────────────────
phase write_sentinel "results sentinel for poll_pipeline.py"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] sentinel :: /workspace/logs/issue-599-epm_results-<epoch>.json"
else
  EPOCH="$(date +%s)"
  SENTINEL="/workspace/logs/issue-599-epm_results-${EPOCH}.json"
  uv run python - "$SENTINEL" "$OUTPUT_DIR" "$SMOKE_DIR" "$EXT_DIR" "$HF_DATA_REPO_PRIVATE" "$HF_TENSOR_PREFIX" "$EXT_RAN" "$EXT_ESCALATED" <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path

sentinel, output_dir, smoke_dir, ext_dir = (
    sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
)
repo_id, prefix = sys.argv[5], sys.argv[6]
ext_ran, ext_escalated = sys.argv[7] == "1", sys.argv[8] == "1"
gate_path = smoke_dir / "smoke_gate_result.json"
gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
b0_path = output_dir / "smoke" / "marker_seed42" / "saturation_gate_result.json"
b0 = json.loads(b0_path.read_text()) if b0_path.exists() else {}
b0_rr_path = output_dir / "smoke" / "marker_seed42" / "run_result.json"
b0_rr = json.loads(b0_rr_path.read_text()) if b0_rr_path.exists() else {}
manip_path = output_dir / "manipulation_check.json"
manip = json.loads(manip_path.read_text()) if manip_path.exists() else {}
shift_files = sorted(p.name for p in (output_dir / "shifts").glob("*.pt"))
provenance = {}
for seed in (42, 137, 256):
    rr_p = output_dir / f"marker_seed{seed}" / "run_result.json"
    if rr_p.exists():
        rr = json.loads(rr_p.read_text())
        provenance[f"seed{seed}"] = {
            "loss_shape": rr.get("loss_shape"),
            "labels_unmasked_per_row_mean": rr.get("labels_unmasked_per_row_mean"),
            "initial_train_loss": rr.get("initial_train_loss"),
            "hf_adapter_subfolder": rr.get("hf_adapter_subfolder"),
        }
ext = {"ran": ext_ran, "escalated": ext_escalated}
for name in ("probe_read.json", "band_entry_staging.json"):
    p = ext_dir / name
    if p.exists():
        ext[name.removesuffix(".json")] = json.loads(p.read_text())
try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    git_commit = "unknown"

note = {
    "issue": 599,
    "phase": "fullresp_train_extract_complete_and_uploaded",
    "loss_shape": "full_response",
    "b0_smoke_gate": b0,
    "b0_smoke_provenance": {
        "loss_shape": b0_rr.get("loss_shape"),
        "labels_unmasked_per_row_mean": b0_rr.get("labels_unmasked_per_row_mean"),
        "initial_train_loss": b0_rr.get("initial_train_loss"),
    },
    "extraction_smoke_gate": gate,
    "n_shift_pt_files": len(shift_files),
    "shift_files": shift_files,
    "hf_tensor_prefix": f"{repo_id}/{prefix}",
    "manipulation_check": manip,
    "production_provenance": provenance,
    "extension_branch": ext,
    "git_commit": git_commit,
    "next_step": (
        "VM-side: uv run python scripts/issue599_compare.py "
        "--new-shifts-dir eval_results/issue_599/shifts "
        "--manipulation-check eval_results/issue_599/manipulation_check.json "
        "--smoke-gate-json eval_results/issue_599_extract_smoke/smoke_gate_result.json "
        "--out eval_results/issue_599/comparison"
    ),
}
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 599,
    "by": "run_issue599_fullresp.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
with open(sentinel, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote sentinel: {sentinel}")
PY
fi

phase done "issue-599 whole-response-loss train + extraction + persistence complete (comparison runs OFF-POD)"
