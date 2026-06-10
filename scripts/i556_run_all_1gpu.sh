#!/usr/bin/env bash
# 1-GPU sequential launcher for issue #556 (plan v3 §4.2 item 5).
# Fork of i528_run_all_1gpu.sh: validating-only, 10 fresh seeds, base reused
# from #528 (NO eval_base, NO pod-side judge/analyze — judge runs OFF-POD on
# the VM after termination, plan §4.3/§9).
# Production command:
#   nohup bash scripts/i556_run_all_1gpu.sh > /workspace/logs/issue-556-run.log 2>&1 &

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Single source of truth for the slug + seed list: every python phase reads
# these via i528_data.ISSUE_SLUG / i528_traits.SEEDS, so train, eval-cell
# enumeration, analyze, and plot all derive from the same subset.
export I528_ISSUE_SLUG=issue_556
export I528_SEEDS=11,23,73,191,257,401,503,631,757,911
SEEDS_SPACED=${I528_SEEDS//,/ }

LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

# Preflight builds ALL 4 traits' Q-banks (codepath_verify + the contrastive
# sibling panel iterate the full TRAITS tuple — plan §4.2 item 2) and runs
# the credential/reachability checks.
echo "[phase=preflight] $(date -Iseconds)"
uv run python scripts/i528_phase0_preflight.py

# Q-bank pin assert (plan §4.2 item 5 + §7): ALL 8 sha256s of the freshly
# regenerated banks must equal #528's committed pins. Pins are read from the
# parent artifact, never hardcoded. Hard stop BEFORE any GPU work on mismatch.
echo "[phase=qbank_pin_assert] $(date -Iseconds)"
uv run python - <<'PYEOF'
import json
import os

slug = os.environ["I528_ISSUE_SLUG"]
new = {
    x["trait"]: x
    for x in json.load(open(f"eval_results/{slug}/preflight_summary.json"))["qbank_summaries"]
}
old = {
    x["trait"]: x
    for x in json.load(open("eval_results/issue_528/preflight_summary.json"))["qbank_summaries"]
}
missing = sorted(set(old) - set(new))
assert not missing, f"preflight summary missing traits: {missing}"
for t in sorted(old):
    for split in ("train", "test"):
        key = f"sha256_{split}"
        assert new[t][key] == old[t][key], (
            f"Q-bank pin MISMATCH trait={t} split={split}: "
            f"regenerated {new[t][key][:12]}… != committed {old[t][key][:12]}…. "
            "Plan §8 pre-authorized deviation path applies — do NOT proceed silently."
        )
print(f"[pin-assert] all {2 * len(old)} Q-bank sha256 pins match the committed issue_528 pins")
PYEOF

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

# NO i528_phase4_eval_base.py: the untrained-base judge rows are REUSED from
# #528's committed judge_scores.json via scripts/i556_merge_base_rows.py on
# the VM (plan §4.2 item 4c).
echo "[phase=phase4_eval] $(date -Iseconds)"
# NB --traits-subset (not --trait): --trait only applies to the single-
# adapter mode; the iteration subset is --traits-subset (eval enumerates
# traits_subset x ARMS x SEEDS, with SEEDS read from I528_SEEDS).
uv run python scripts/i528_phase4_eval.py --traits-subset validating

# Upload policy (CLAUDE.md + plan §10): raw generations + R_pos/R_neg +
# train_rows + Q-banks land on the HF data repo BEFORE pod termination.
# The eval writes flat per-cell JSONs (<trait>_<arm>_seed<S>__<ctx>.json),
# not <cell>/raw_completions.json, so the canonical rglob helper cannot see
# them — explicit per-file fail-loud upload loop instead (incident #528).
# Adapters were already uploaded per-cell by i528_phase23_train.py
# (hf_upload=True + fail-loud list_repo_files verification).
echo "[phase=upload] $(date -Iseconds)"
uv run python - <<'PYEOF'
import os
from pathlib import Path

from explore_persona_space.orchestrate import hub
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — refusing silent upload skip"

slug = os.environ["I528_ISSUE_SLUG"]
exp = "issue556_role_header_validating"

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
    "n_train_cell_files": n_train,
    "git_commit_sha": os.environ.get("GIT_COMMIT", ""),
    "note": (
        "i556 pod phases complete: 20 cells trained + evaled, raw generations "
        f"({n_raw} files) + data uploaded to HF. Judge/merge/analyze/plot run "
        "OFF-POD on the VM (plan §4.3)."
    ),
}
with open(sentinel_path, "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[sentinel] wrote {sentinel_path}")
PYEOF

echo "[phase=done] $(date -Iseconds)"
