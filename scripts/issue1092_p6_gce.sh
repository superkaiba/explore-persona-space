#!/usr/bin/env bash
# Issue #1092 P6 (fit grids) on GCP cpu-bigmem — plan-v6 12-box fan-out driver.
#
# History: the VM P6 pilot was earlyoom-killed at >=22.9 GB RSS, so P6 routes to
# dedicated n2-highmem-16 boxes (128 GB; the v87 escape lane). Plan v6 fans P6
# out to 12 such boxes; this driver is parametrized by env — ALL defaults
# preserve the original single-run behavior (plus the two plan-v6 hardcoded
# wrapper flags noted below):
#
#   P6_BOX_ID           box id, e.g. "02" -> per-box upload prefix
#                       issue1092_realistic_crossing/p6/box_${P6_BOX_ID}/
#                       (summary/manifest filenames collide across boxes;
#                       checkpoint names are fingerprint-unique so P7 merges
#                       the union).
#   P6_JOBS             ';;'-separated job specs; fields '|'-separated:
#                       cells=...|layers=...|fit_arms=...|bases=...|
#                       pilot_cell=...|pilot_layer=...|plan_wall_h=...|extra=...
#                       Each job = ONE issue1092_p6_run.py invocation; an empty
#                       or absent field falls back to the wrapper's default.
#                       'extra' is an optional pass-through of
#                       issue1092_fit_grid.py tokens and rides the wrapper's
#                       --fit-grid-arg (e.g. extra=--skip-mlp-companion).
#   P6_PARTB_JOBS       ';;'-separated Part-B operator-comparison job specs
#                       (offvm-battery-refit round); fields '|'-separated:
#                       cells=...|layers=...|bases=...|extra=... Each job = ONE
#                       scripts/issue1092_partb_operator.py invocation
#                       (--stage-from-hub against $STAGE_DIR; outputs to
#                       $OUT_DIR/partb, which rides the existing upload).
#                       Runs AFTER the P6_JOBS loop, before upload. Unset ->
#                       no Part-B phase (fully backward compatible).
#   P6_RESTORE_ATTEMPT  crash-persist attempt id: before jobs, stage
#                       issue1092_partial/<att>/data_issue_1092/p6/
#                       {checkpoints/*.json, analysis_tensors/nulls/*.npy}
#                       into $OUT_DIR — per-file scoped downloads via
#                       list_repo_tree on the attempt prefix, NEVER a
#                       snapshot_download (1M-file repo full-tree wedge,
#                       .claude/rules/gotchas.md) and NEVER a bare subdir
#                       guess (the att-20260708-232746 folder-404 class).
#   P6_STAGE_DIR        stage-dir override (default /workspace/p6_stage);
#                       used by local smokes so nothing touches /workspace.
#   P6_DRY_RUN          smoke: echo composed wrapper invocations + touch the
#                       summary files each job would write (so the rename path
#                       runs for real); skip Hub staging/upload.
#   P6_RESTORE_FIXTURE_ROOT  offline restore source (mirrors the wrapper's
#                       --fixture-hub-root pattern): enumerate/copy from this
#                       local tree instead of the Hub — same filter/mapping
#                       code path, only the Hub boundary is faked.
#
# --skip-band-pilot and --max-pilot-rss-gb 64 are DRIVER-HARDCODED on every
# wrapper invocation (plan v6 §4.5-A: all fit layers are frozen so the band
# block would be a pure duplicate re-run; 128 GB box -> 64 GB pilot RSS gate).
# They are issue1092_p6_run.py WRAPPER flags — the engine's argparse rejects
# them — so they must never ride the 'extra' fit-grid passthrough.
#
# Within-box summary dedup: after each P6_JOBS job the per-invocation summary
# JSONs (fit_grid_summary*.json / p6_run_summary*.json / pilot*.json) are
# renamed to *_job<k>.json so later jobs cannot clobber them on disk or on the
# shared HF prefix. Checkpoints stay in the SHARED $OUT_DIR/checkpoints
# (fingerprint-unique, resume-shared across jobs and boxes).
#
# BYTE-PIN: scripts/issue1092_fit_grid.py hashes its own bytes into every
# checkpoint fingerprint — during the v6 relaunch this driver was the ONLY edit
# surface; any engine edit invalidates ALL completed checkpoints. The
# offvm-battery-refit round DELIBERATELY edits the engine (battery-excluded
# fit-arm filters + per-target R2 banking), so banked-checkpoint resume can
# never silently fire on refit boxes; refit boxes additionally use fresh box
# ids (rf01..rf04 -> box_rf0K HF prefixes) and set NO P6_RESTORE_ATTEMPT, so
# banked p6 artifacts are neither resumed from nor clobbered.
#
# GCE lane contract: cwd = $WORKLOAD_ROOT (the issue-1092 clone); HF_TOKEN etc.
# exported by the startup script; no .env file (source conditionally).
set -euo pipefail

if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

# Dedicated box: full width (16 vCPU). The wrapper's shared-VM setdefault is 8;
# explicit env wins.
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

CORPUS_REV="7ef5523673d64697ab497577dbc5b9270c39f020"   # same pin as issue1092_dispatch.sh
REPO="superkaiba1/explore-persona-space-data"

P6_BOX_ID="${P6_BOX_ID:-}"
P6_JOBS="${P6_JOBS:-}"
P6_PARTB_JOBS="${P6_PARTB_JOBS:-}"
P6_RESTORE_ATTEMPT="${P6_RESTORE_ATTEMPT:-}"
P6_DRY_RUN="${P6_DRY_RUN:-}"
P6_RESTORE_FIXTURE_ROOT="${P6_RESTORE_FIXTURE_ROOT:-}"

CORPUS_DIR="data/issue_1092/p0/corpus"
JUDGE_DIR="data/issue_1092/p5_judge"
OUT_DIR="data/issue_1092/p6"            # inside the clone: crash-persist sweeps data_issue_<N>
STAGE_DIR="${P6_STAGE_DIR:-/workspace/p6_stage}"  # OUTSIDE the clone: re-downloadable cache

HF_PREFIX="issue1092_realistic_crossing/p6"
if [ -n "$P6_BOX_ID" ]; then
  HF_PREFIX="issue1092_realistic_crossing/p6/box_${P6_BOX_ID}"
fi

mkdir -p "$CORPUS_DIR" "$JUDGE_DIR" "$OUT_DIR" "$STAGE_DIR"

run_p6_job() {
  # Compose + run ONE issue1092_p6_run.py invocation from a '|'-separated job
  # spec. An empty spec composes the original single-run line (wrapper
  # defaults) — the default no-P6_JOBS path goes through this same function.
  local job_idx="$1" spec="$2"
  local cells="" layers="" fit_arms="" bases="" pilot_cell="" pilot_layer=""
  local plan_wall_h="" extra=""
  if [ -n "$spec" ]; then
    # IFS is scoped to the read only (an env-prefix assignment): a function-wide
    # `local IFS='|'` would make the audit echo below join ${cmd[*]} with '|'.
    local fields=() field key val
    IFS='|' read -r -a fields <<< "$spec"
    for field in "${fields[@]}"; do
      if [ -z "$field" ]; then continue; fi
      key="${field%%=*}"
      val="${field#*=}"
      case "$key" in
        cells) cells="$val" ;;
        layers) layers="$val" ;;
        fit_arms) fit_arms="$val" ;;
        bases) bases="$val" ;;
        pilot_cell) pilot_cell="$val" ;;
        pilot_layer) pilot_layer="$val" ;;
        plan_wall_h) plan_wall_h="$val" ;;
        extra) extra="$val" ;;
        *)
          echo "[p6-gce] ERROR: unknown P6_JOBS field '$key' in job spec '$spec'" >&2
          exit 2
          ;;
      esac
    done
  fi

  # --skip-band-pilot / --max-pilot-rss-gb are WRAPPER flags, hardcoded here
  # per plan v6 (never via 'extra': the engine's argparse rejects them).
  # P6_MAX_PILOT_RSS_GB (default 64) exists because the 64 GB VM-routing gate
  # aborted a healthy 72.23 GB pilot on a 128 GB n2-highmem-16 box (b10,
  # att-20260709-180648-p6b10) whose abort message routes to the very lane it
  # ran on; replacement boxes pass 100.
  local cmd=(
    uv run python scripts/issue1092_p6_run.py
    --corpus-dir "$CORPUS_DIR"
    --stage-dir "$STAGE_DIR"
    --out-dir "$OUT_DIR"
    --judge-scores "$JUDGE_DIR/scores.jsonl"
    --skip-band-pilot
    --max-pilot-rss-gb "${P6_MAX_PILOT_RSS_GB:-64}"
  )
  if [ -n "$cells" ]; then cmd+=(--cells "$cells"); fi
  if [ -n "$layers" ]; then cmd+=(--layers "$layers"); fi
  if [ -n "$fit_arms" ]; then cmd+=(--fit-arms "$fit_arms"); fi
  if [ -n "$bases" ]; then cmd+=(--target-bases "$bases"); fi
  if [ -n "$pilot_cell" ]; then cmd+=(--pilot-cell "$pilot_cell"); fi
  if [ -n "$pilot_layer" ]; then cmd+=(--pilot-layer "$pilot_layer"); fi
  if [ -n "$plan_wall_h" ]; then cmd+=(--plan-wall-h "$plan_wall_h"); fi
  # The wrapper shlex-splits each --fit-grid-arg value and rejects
  # wrapper-owned flags (issue1092_p6_run.py::validate_extra_fit_grid_args),
  # so a multi-token 'extra' rides a single argument safely. MUST use the
  # '=' form: argparse reads a separate '--'-prefixed value token as another
  # option and dies with "expected one argument".
  if [ -n "$extra" ]; then cmd+=("--fit-grid-arg=$extra"); fi

  echo "[phase=p6_fit${job_idx:+_job${job_idx}}]"
  echo "[p6-gce] invocation${job_idx:+ (job ${job_idx})}: ${cmd[*]}"
  if [ -n "$P6_DRY_RUN" ]; then
    # Simulate the wrapper's out-dir summary writes so the per-job rename path
    # below is exercised end-to-end in smoke.
    touch "$OUT_DIR/fit_grid_summary.json" "$OUT_DIR/p6_run_summary.json" "$OUT_DIR/pilot.json"
  else
    "${cmd[@]}"
  fi
}

rename_job_summaries() {
  # Within-box dedup (plan v6 § Relaunch commands): job-suffix this
  # invocation's summary JSONs so the next job cannot clobber them.
  # Checkpoints are fingerprint-unique and stay shared in $OUT_DIR/checkpoints.
  local job_idx="$1"
  local f base dst
  for f in "$OUT_DIR"/fit_grid_summary*.json "$OUT_DIR"/p6_run_summary*.json "$OUT_DIR"/pilot*.json; do
    if [ ! -e "$f" ]; then continue; fi          # unmatched glob stays literal
    base="$(basename "$f" .json)"
    case "$base" in
      *_job[0-9]*) continue ;;                   # already suffixed by a prior job
    esac
    dst="$OUT_DIR/${base}_job${job_idx}.json"
    if [ -e "$dst" ]; then
      echo "[p6-gce] ERROR: refusing to clobber $dst" >&2
      exit 2
    fi
    mv "$f" "$dst"
    echo "[p6-gce] job ${job_idx}: renamed ${base}.json -> ${base}_job${job_idx}.json"
  done
}

run_partb_job() {
  # Compose + run ONE issue1092_partb_operator.py invocation from a
  # '|'-separated job spec (offvm-battery-refit round: operator-level arm
  # comparison on the battery-excluded fitted maps). Outputs land in
  # $OUT_DIR/partb, which rides the existing end-of-run upload.
  local job_idx="$1" spec="$2"
  local cells="" layers="" bases="" extra=""
  if [ -n "$spec" ]; then
    local fields=() field key val
    IFS='|' read -r -a fields <<< "$spec"
    for field in "${fields[@]}"; do
      if [ -z "$field" ]; then continue; fi
      key="${field%%=*}"
      val="${field#*=}"
      case "$key" in
        cells) cells="$val" ;;
        layers) layers="$val" ;;
        bases) bases="$val" ;;
        extra) extra="$val" ;;
        *)
          echo "[p6-gce] ERROR: unknown P6_PARTB_JOBS field '$key' in job spec '$spec'" >&2
          exit 2
          ;;
      esac
    done
  fi
  local cmd=(
    uv run python scripts/issue1092_partb_operator.py
    --summaries-dir "$STAGE_DIR"
    --corpus-dir "$CORPUS_DIR"
    --out-dir "$OUT_DIR"
    --stage-from-hub
  )
  if [ -n "$cells" ]; then cmd+=(--cells "$cells"); fi
  if [ -n "$layers" ]; then cmd+=(--layers "$layers"); fi
  if [ -n "$bases" ]; then cmd+=(--target-bases "$bases"); fi
  if [ -n "$extra" ]; then
    # shellcheck disable=SC2206 -- deliberate word-splitting of pass-through tokens
    cmd+=($extra)
  fi

  echo "[phase=p6_partb_job${job_idx}]"
  echo "[p6-gce] partb invocation (job ${job_idx}): ${cmd[*]}"
  if [ -n "$P6_DRY_RUN" ]; then
    # Simulate the summary write so the per-job rename path runs for real.
    mkdir -p "$OUT_DIR/partb"
    touch "$OUT_DIR/partb/partb_summary.json"
  else
    "${cmd[@]}"
  fi
}

rename_partb_summary() {
  # Job-suffix each Part-B job's summary so a later Part-B job cannot clobber
  # it (per-unit JSONs are fingerprint-unique and stay shared in partb/).
  local job_idx="$1"
  local f="$OUT_DIR/partb/partb_summary.json"
  if [ ! -e "$f" ]; then
    echo "[p6-gce] ERROR: partb job ${job_idx} wrote no partb_summary.json" >&2
    exit 2
  fi
  local dst="$OUT_DIR/partb/partb_summary_pjob${job_idx}.json"
  if [ -e "$dst" ]; then
    echo "[p6-gce] ERROR: refusing to clobber $dst" >&2
    exit 2
  fi
  mv "$f" "$dst"
  echo "[p6-gce] partb job ${job_idx}: renamed partb_summary.json -> partb_summary_pjob${job_idx}.json"
}

if [ -n "$P6_DRY_RUN" ]; then
  echo "[p6-gce] dry-run: skipping input staging (corpus @ ${CORPUS_REV:0:12}, p5_judge shards)"
else
  echo "[phase=p6_stage_inputs]"
  uv run python - <<'PY'
import hashlib
import json
import os
import pathlib
import shutil

from huggingface_hub import hf_hub_download, list_repo_tree

REPO = "superkaiba1/explore-persona-space-data"

# 1) corpus at the pinned revision (recipe from issue1092_dispatch.sh)
REV = "7ef5523673d64697ab497577dbc5b9270c39f020"
PREFIX = "issue1092_realistic_crossing/corpus"
dst = pathlib.Path("data/issue_1092/p0/corpus")
names = []
for it in list_repo_tree(REPO, repo_type="dataset", path_in_repo=PREFIX, revision=REV):
    local = hf_hub_download(REPO, repo_type="dataset", filename=it.path, revision=REV)
    shutil.copy(local, dst / pathlib.Path(it.path).name)
    names.append(pathlib.Path(it.path).name)
required = {"manifest.jsonl", "prefix_store.jsonl", "query_store.jsonl", "derangement_map.json"}
missing = required - set(names)
assert not missing, f"staged corpus missing {missing}; got {sorted(names)}"
print(f"[p6-gce] staged {len(names)} corpus files @ {REV[:12]}")

# 2) P5 judge scores: shards + manifest -> reassemble -> sha256 verify
JPREFIX = "issue1092_realistic_crossing/p5_judge"
jdst = pathlib.Path("data/issue_1092/p5_judge")
shard_paths = []
manifest_local = None
for it in list_repo_tree(REPO, repo_type="dataset", path_in_repo=JPREFIX):
    name = pathlib.Path(it.path).name
    # Download ONLY the expected score files. The prefix also carries a raw/
    # SUBDIRECTORY (P5 raw judge outputs, persisted separately); the
    # non-recursive listing yields folder entries, and hf_hub_download on a
    # folder path 404s (att-20260708-232746 crash).
    is_shard = name.startswith("scores_shard_") and name.endswith(".jsonl")
    if not (is_shard or name in ("shards_manifest.json", "summary.json")):
        continue
    local = hf_hub_download(REPO, repo_type="dataset", filename=it.path)
    if name == "shards_manifest.json":
        manifest_local = local
    elif is_shard:
        shard_paths.append((name, local))
    else:
        shutil.copy(local, jdst / name)
assert manifest_local is not None, "shards_manifest.json missing under p5_judge"
man = json.load(open(manifest_local))
shard_paths.sort(key=lambda t: t[0])
assert len(shard_paths) == man["n_shards"], (
    f"shard count mismatch: hub {len(shard_paths)} vs manifest {man['n_shards']}"
)
scores = jdst / "scores.jsonl"
h = hashlib.sha256()
with open(scores, "wb") as out:
    for _, local in shard_paths:
        with open(local, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
                out.write(b)
digest = h.hexdigest()
assert digest == man["full_sha256"], (
    f"reassembled scores.jsonl sha mismatch: {digest} vs manifest {man['full_sha256']}"
)
assert scores.stat().st_size == man["total_bytes"]
print(f"[p6-gce] reassembled scores.jsonl OK: {man['total_bytes']} bytes sha256={digest[:16]}")
PY
fi

if [ -n "$P6_RESTORE_ATTEMPT" ]; then
  if [ -n "$P6_DRY_RUN" ] && [ -z "$P6_RESTORE_FIXTURE_ROOT" ]; then
    echo "[p6-gce] dry-run: would restore issue1092_partial/${P6_RESTORE_ATTEMPT}/data_issue_1092/p6/{checkpoints/*.json,analysis_tensors/nulls/*.npy} -> $OUT_DIR"
  else
    echo "[phase=p6_restore_checkpoints]"
    P6_RESTORE_ATTEMPT="$P6_RESTORE_ATTEMPT" \
    P6_RESTORE_FIXTURE_ROOT="$P6_RESTORE_FIXTURE_ROOT" \
    P6_OUT_DIR="$OUT_DIR" \
    uv run python - <<'PY'
"""Restore prior-attempt P6 checkpoints (+ persisted null npys) from the
GCE crash-persist prefix, so cross-box resume skips completed units.

Per-file scoped downloads: enumerate the ATTEMPT prefix with list_repo_tree
(recursive), whitelist-filter, hf_hub_download exactly those files. NEVER a
snapshot_download (1M-file repo full-tree wedge, .claude/rules/gotchas.md)
and NEVER a bare subdir guess (hf_hub_download on a folder path 404s — the
att-20260708-232746 crash class)."""
import os
import pathlib
import shutil

att = os.environ["P6_RESTORE_ATTEMPT"]
out_dir = pathlib.Path(os.environ["P6_OUT_DIR"])
fixture = os.environ.get("P6_RESTORE_FIXTURE_ROOT") or None
REPO = "superkaiba1/explore-persona-space-data"
prefix = f"issue1092_partial/{att}/data_issue_1092/p6"


def _wanted(rel: str) -> bool:
    """Whitelist: per-unit checkpoint JSONs + persisted null-draw npys only."""
    if rel.startswith("checkpoints/") and rel.endswith(".json"):
        return True
    return rel.startswith("analysis_tensors/nulls/") and rel.endswith(".npy")


if fixture:
    # Offline smoke path (mirrors issue1092_p6_run.py --fixture-hub-root):
    # same filter/mapping code below; only the Hub boundary is faked.
    root = pathlib.Path(fixture) / prefix
    assert root.is_dir(), f"restore fixture missing attempt tree: {root}"
    hub_paths = [
        f"{prefix}/{p.relative_to(root).as_posix()}" for p in root.rglob("*") if p.is_file()
    ]
else:
    from huggingface_hub import hf_hub_download, list_repo_tree

    hub_paths = [
        it.path
        for it in list_repo_tree(REPO, repo_type="dataset", path_in_repo=prefix, recursive=True)
    ]

n_ckpt = 0
n_null = 0
for hub_path in sorted(hub_paths):
    rel = hub_path[len(prefix) + 1 :]
    if not _wanted(rel):
        continue
    dst = out_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if fixture:
        shutil.copy(pathlib.Path(fixture) / hub_path, dst)
    else:
        local = hf_hub_download(REPO, repo_type="dataset", filename=hub_path)
        shutil.copy(local, dst)
    if rel.startswith("checkpoints/"):
        n_ckpt += 1
    else:
        n_null += 1
assert n_ckpt > 0, f"restore attempt {att!r}: no checkpoint JSONs under {prefix}/checkpoints/"
print(f"[p6-gce] restored {n_ckpt} checkpoint JSONs + {n_null} null npys from {prefix} -> {out_dir}")
PY
  fi
fi

if [ -n "$P6_JOBS" ]; then
  job_idx=0
  remaining="$P6_JOBS"
  while [ -n "$remaining" ]; do
    job_spec="${remaining%%;;*}"
    if [ "$job_spec" = "$remaining" ]; then remaining=""; else remaining="${remaining#*;;}"; fi
    if [ -z "$job_spec" ]; then continue; fi
    job_idx=$((job_idx + 1))
    run_p6_job "$job_idx" "$job_spec"
    rename_job_summaries "$job_idx"
  done
  if [ "$job_idx" -eq 0 ]; then
    echo "[p6-gce] ERROR: P6_JOBS set but contained no job specs" >&2
    exit 2
  fi
else
  run_p6_job "" ""
fi

if [ -n "$P6_PARTB_JOBS" ]; then
  pjob_idx=0
  remaining="$P6_PARTB_JOBS"
  while [ -n "$remaining" ]; do
    job_spec="${remaining%%;;*}"
    if [ "$job_spec" = "$remaining" ]; then remaining=""; else remaining="${remaining#*;;}"; fi
    if [ -z "$job_spec" ]; then continue; fi
    pjob_idx=$((pjob_idx + 1))
    run_partb_job "$pjob_idx" "$job_spec"
    rename_partb_summary "$pjob_idx"
  done
  if [ "$pjob_idx" -eq 0 ]; then
    echo "[p6-gce] ERROR: P6_PARTB_JOBS set but contained no job specs" >&2
    exit 2
  fi
fi

if [ -n "$P6_DRY_RUN" ]; then
  echo "[p6-gce] dry-run: would upload $OUT_DIR -> $HF_PREFIX/ on $REPO"
else
  echo "[phase=p6_upload]"
  P6_HF_PREFIX="$HF_PREFIX" P6_OUT_DIR="$OUT_DIR" uv run python - <<'PY'
import os

from huggingface_hub import HfApi

prefix = os.environ["P6_HF_PREFIX"]
out_dir = os.environ["P6_OUT_DIR"]
api = HfApi()
res = api.upload_folder(
    folder_path=out_dir,
    path_in_repo=prefix,
    repo_id="superkaiba1/explore-persona-space-data",
    repo_type="dataset",
    commit_message=f"issue #1092 P6 fit-grid outputs ({prefix}; GCP cpu-bigmem judge-bearing run)",
)
print("[p6-gce] uploaded out-dir:", res)
# Scoped list_repo_tree, NEVER list_repo_files: the data repo is ~1M files and
# full-tree enumeration wedges (.claude/rules/gotchas.md).
api_files = [
    e.path
    for e in api.list_repo_tree(
        "superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        path_in_repo=prefix,
        recursive=True,
    )
]


def _is_fit_summary(path: str) -> bool:
    """Match fit_grid_summary.json AND the job-suffixed fit_grid_summary_job<k>.json."""
    name = path.rsplit("/", 1)[-1]
    return name.startswith("fit_grid_summary") and name.endswith(".json")


assert any(_is_fit_summary(f) for f in api_files), (
    f"no fit_grid_summary*.json on hub under {prefix} after upload ({len(api_files)} files)"
)
print(f"[p6-gce] hub-verified {len(api_files)} files under {prefix}/")
PY
fi

echo "[phase=done]"
