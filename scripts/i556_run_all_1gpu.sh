#!/usr/bin/env bash
# 1-GPU sequential launcher for issue #556 (plan v3 §4.2 item 5 + §8 risk row 1).
# Fork of i528_run_all_1gpu.sh: validating-only, 10 fresh seeds. Round 3: the
# plan-§8 PRE-AUTHORIZED Q-bank deviation path is ACTIVE (all 8 regenerated
# bank sha256s mismatched #528's pins on 2026-06-10), so the validating BASE
# is re-evaled POD-SIDE (i528_phase4_eval_base.py, both arms) and judged fresh
# on the VM — parent base-row reuse via i556_merge_base_rows.py is INVALID
# when the validating test-bank pin mismatches. NO pod-side judge/analyze —
# judge runs OFF-POD on the VM after termination (plan §4.3/§9).
# Production command:
#   nohup bash scripts/i556_run_all_1gpu.sh > /workspace/logs/issue-556-run.log 2>&1 &
#
# VM-side phase AFTER upload + pod termination (plan §4.3, §8-deviation form)
# — run from the VM repo root with the SAME two env exports this script sets
# below:
#   uv run python scripts/i556_pull_qbank.py   # materialize data/issue_556/ Q-bank
#                                              # (judge prerequisite: assert_q_test_equality
#                                              # reads it; pins verified against THIS run's
#                                              # eval_results/issue_556/preflight_summary.json)
#   nohup uv run python scripts/i528_phase4_judge.py --backend sync \
#     > /tmp/i556_judge.log 2>&1 &             # NO --skip-base: judges the fresh pod-side
#                                              # base rows too (~+1.2k calls); --resume on crash
#   uv run python scripts/i528_phase5_analyze.py --saturation-gate per_encoding \
#     --h2-bar-d-mean -0.10 --h2-min-seeds-neg 8
#   uv run python scripts/i528_phase5_analyze.py --saturation-gate pooled \
#     --out-name analysis_pooled_gate.json     # archived audit run
#   uv run python scripts/plot_i528_clean_result.py
#
# scripts/i556_merge_base_rows.py is DROPPED from the sequence: it applies
# ONLY when the validating test-bank pin matches #528's (see
# eval_results/issue_556/qbank_pin_deviation.json — base_reuse_valid). The
# script is kept for the pin-match case and refuses loudly otherwise (its
# test-bank pin guard).

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Single source of truth for the slug + seed list: every python phase reads
# these via i528_data.ISSUE_SLUG / i528_traits.SEEDS, so train, eval-cell
# enumeration, analyze, and plot all derive from the same subset.
export I528_ISSUE_SLUG=issue_556
export I528_SEEDS=11,23,73,191,257,401,503,631,757,911
# 2026-06-10: public HF storage quota exhausted (403 on LFS) — adapters persist to a
# PRIVATE overflow repo (separate quota pool). Recorded deviation from plan §10.
export I556_HF_MODEL_REPO=superkaiba1/eps-private-overflow
SEEDS_SPACED=${I528_SEEDS//,/ }

LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

# Preflight builds ALL 4 traits' Q-banks (codepath_verify + the contrastive
# sibling panel iterate the full TRAITS tuple — plan §4.2 item 2) and runs
# the credential/reachability checks.
echo "[phase=preflight] $(date -Iseconds)"
uv run python scripts/i528_phase0_preflight.py

# Q-bank pin CHECK + deviation recorder (plan §8 risk row 1, pre-authorized
# deviation — replaces round-1's hard all-8 assert). STRUCTURAL problems
# (missing summary, missing trait entries, n_train != 60, n_test != 40) still
# fail HARD before any GPU work. Pin mismatches do NOT abort: they are
# recorded to eval_results/issue_556/qbank_pin_deviation.json (per-trait/
# per-split old vs new sha + matched flag + ts) with a LOUD WARNING and the
# run continues on the regenerated banks. A validating TEST-bank mismatch
# additionally invalidates parent base-row reuse -> the base is re-evaled
# pod-side below ([phase=phase4_eval_base]) and judged fresh on the VM.
echo "[phase=qbank_pin_check] $(date -Iseconds)"
uv run python scripts/i556_qbank_pin_check.py

echo "[phase=codepath_verify] $(date -Iseconds)"
uv run python scripts/i528_phase0_codepath_verify.py

echo "[phase=r_pos] $(date -Iseconds)"
uv run python scripts/i528_phase1_generate_RPos.py --traits validating

echo "[phase=r_neg] $(date -Iseconds)"
uv run python scripts/i528_phase1_generate_RNeg.py --traits validating

# Smoke cell = the sweep with one cell (plan §4.4): SAME train entrypoint
# with --smoke, then the smoke-judge gate (vLLM + LoRARequest + Claude judge)
# must score > 3.0 own-scenario BEFORE the 20-cell loop.
echo "[phase=phase2_smoke] $(date -Iseconds)"
uv run python scripts/i528_phase23_train.py \
    --trait validating --arm role --seed 11 --smoke --gpu-id 0
uv run python scripts/i528_phase2_smoke_judge.py \
    --adapter adapters/i528_validating_role_seed11_smoke \
    --trait validating --arm role --threshold 3.0

echo "[phase=phase3_sweep] $(date -Iseconds)"
for arm in system role; do
    for seed in ${SEEDS_SPACED}; do
        echo "[phase=phase3_cell] trait=validating arm=${arm} seed=${seed} $(date -Iseconds)"
        uv run python scripts/i528_phase23_train.py \
            --trait validating --arm "${arm}" --seed "${seed}" --gpu-id 0 \
            > "${LOG_DIR}/i556_validating_${arm}_seed${seed}.log" 2>&1
    done
done

echo "[phase=phase4_eval] $(date -Iseconds)"
# NB --traits-subset (not --trait): --trait only applies to the single-
# adapter mode; the iteration subset is --traits-subset (eval enumerates
# traits_subset x ARMS x SEEDS, with SEEDS read from I528_SEEDS).
uv run python scripts/i528_phase4_eval.py --traits-subset validating

# Pod-side BASE eval (plan §8 pre-authorized fallback, ACTIVE because the
# validating test-bank pin mismatched #528's): untrained-base greedy
# generations for validating under BOTH eval arms x 5 contexts = 10 files
# under eval_results/issue_556/raw_generations_base/. Own vLLM init — this is
# a separate subprocess, so phase4_eval's engine is already reaped at fork.
# These rows replace the parent base-row reuse (i556_merge_base_rows.py is
# dropped from the VM sequence); the VM judge runs WITHOUT --skip-base and
# scores them fresh.
echo "[phase=phase4_eval_base] $(date -Iseconds)"
uv run python scripts/i528_phase4_eval_base.py --traits validating

# Upload policy (CLAUDE.md + plan §10): raw generations + R_pos/R_neg +
# train_rows + Q-banks land on the HF data repo BEFORE pod termination.
# The eval writes flat per-cell JSONs (<trait>_<arm>_seed<S>__<ctx>.json),
# not <cell>/raw_completions.json, so the canonical rglob helper cannot see
# them — explicit per-file fail-loud upload loop instead (incident #528).
# Adapters were already uploaded per-cell by i528_phase23_train.py
# (hf_upload=True + fail-loud list_repo_files verification) to the MODEL repo
# under the SAME slug-derived prefix used below for the data uploads
# (HF_EXPERIMENT_PREFIX/adapters/<run_name>, plan §10).
echo "[phase=upload] $(date -Iseconds)"
uv run python - <<'PYEOF'
import os
from pathlib import Path

from explore_persona_space.experiments.i528_data import HF_EXPERIMENT_PREFIX as exp
from explore_persona_space.orchestrate import hub
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — refusing silent upload skip"

slug = os.environ["I528_ISSUE_SLUG"]
# Single source of truth for the experiment prefix (i528_data, slug-derived):
# adapters (model repo) + data/raw completions (data repo) share it (plan §10).
assert exp == "issue556_role_header_validating", (
    f"HF_EXPERIMENT_PREFIX={exp!r} — slug env not threaded? expected the "
    "issue_556 prefix issue556_role_header_validating"
)

uploads: list[tuple[Path, str]] = []
raw_dir = Path(f"eval_results/{slug}/raw_generations")
raw_files = sorted(raw_dir.glob("*.json"))
n_production = len([f for f in raw_files if "_smoke" not in f.name])
assert n_production == 100, (
    f"expected exactly 100 non-smoke raw_generations files (20 cells x 5 contexts), "
    f"found {n_production} in {raw_dir}"
)
for f in raw_files:
    uploads.append((f, f"{exp}/raw_completions/{f.name}"))

# Base raw generations (plan §8 deviation path + §6.5 addendum): exactly 10
# files (2 arms x 5 contexts for validating), uploaded under the parent's HF
# layout (<prefix>/raw_generations_base/<name>.json — same shape as
# issue528_role_header_traits/raw_generations_base/).
base_raw_dir = Path(f"eval_results/{slug}/raw_generations_base")
base_files = sorted(base_raw_dir.glob("base__*.json"))
assert len(base_files) == 10, (
    f"expected exactly 10 base raw_generations files (2 arms x 5 contexts for "
    f"validating), found {len(base_files)} in {base_raw_dir}"
)
for f in base_files:
    uploads.append((f, f"{exp}/raw_generations_base/{f.name}"))

data_dir = Path(f"data/{slug}")
for name in ("R_pos.json", "R_neg.json"):
    p = data_dir / name
    assert p.exists(), f"{p} missing — phase 1 did not run?"
    uploads.append((p, f"{exp}/data/{name}"))
for f in sorted((data_dir / "train_rows").glob("*.jsonl")):
    uploads.append((f, f"{exp}/data/train_rows/{f.name}"))
for f in sorted(data_dir.glob("*/Q_*.json")):
    uploads.append((f, f"{exp}/data/qbank/{f.parent.name}/{f.name}"))

failed = []
for local, path_in_repo in uploads:
    url = hub._upload(
        local_path=local,
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        upload_as_file=True,
    )
    if not url:
        failed.append(str(local))
    else:
        print(f"[upload] {local} -> {url}")
if failed:
    raise RuntimeError(f"HF upload FAILED for {len(failed)} files: {failed}")
print(f"[upload] {len(uploads)} files verified on {hub.DEFAULT_DATASET_REPO}/{exp}/")
PYEOF

# End-of-run sentinel for poll_pipeline.py (same shape as #528's run-all;
# _SENTINEL_REQUIRED_KEYS = sentinel_schema_version / kind / version).
SENTINEL_TS=$(date +%s)
SENTINEL="${LOG_DIR}/issue-556-epm_results-${SENTINEL_TS}.json"
python3 - "$ROOT_DIR" "$SENTINEL" <<'PYEOF'
import glob
import hashlib
import json
import os
import sys

root_dir = sys.argv[1]
sentinel_path = sys.argv[2]


def file_sha256(p):
    if not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


eval_paths = [
    f"{root_dir}/eval_results/issue_556/preflight_summary.json",
    f"{root_dir}/eval_results/issue_556/qbank_pin_deviation.json",
    f"{root_dir}/eval_results/issue_556/codepath_verify.json",
    f"{root_dir}/eval_results/issue_556/smoke_judge.json",
]
n_raw = len(
    [
        p
        for p in glob.glob(f"{root_dir}/eval_results/issue_556/raw_generations/*.json")
        if "_smoke" not in os.path.basename(p)
    ]
)
n_base_raw = len(
    glob.glob(f"{root_dir}/eval_results/issue_556/raw_generations_base/base__*.json")
)
n_train = len(
    [
        p
        for p in glob.glob(f"{root_dir}/eval_results/issue_556/train_validating_*.json")
        if "_smoke" not in os.path.basename(p)
    ]
)
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 556,
    "phase": "done",
    "eval_paths": [os.path.relpath(p, root_dir) for p in eval_paths if os.path.isfile(p)],
    "eval_path_sha256": {os.path.relpath(p, root_dir): file_sha256(p) for p in eval_paths},
    "n_raw_generation_files": n_raw,
    "n_base_raw_generation_files": n_base_raw,
    "n_train_cell_files": n_train,
    "git_commit_sha": os.environ.get("GIT_COMMIT", ""),
    "note": (
        "i556 pod phases complete: 20 cells trained + evaled, raw generations "
        f"({n_raw} trained + {n_base_raw} base files) + data uploaded to HF. "
        "Judge (WITHOUT --skip-base — scores the fresh base rows)/analyze/plot "
        "run OFF-POD on the VM (plan §4.3, §8-deviation form; no "
        "i556_merge_base_rows.py)."
    ),
}
with open(sentinel_path, "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[sentinel] wrote {sentinel_path}")
PYEOF

echo "[phase=done] $(date -Iseconds)"
