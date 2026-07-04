#!/usr/bin/env bash
# task #825 follow-up "role-map-comparison" (plan v14):
#   smoke (self, EPS_SMOKE=1 child, redirected) -> stage -> fit+pair ->
#   summarize -> UPLOAD (BEFORE gates, MF-C) -> gates -> sentinel
#
# Production (worker):  bash scripts/issue825_rolecontrast_dispatch.sh
# Smoke (CPU, tiny):    bash scripts/issue825_rolecontrast_dispatch.sh --smoke
#   Smoke IS this wrapper (EPS_SMOKE=1): same phases, same commands, tiny
#   SYNTHETIC .pt bundles (n=3 x 3 shards, D=8, L=28; haiku n_turns=4
#   parent-shaped, real/onpolicy n_turns=3), all outputs under a /tmp scratch
#   root, numeric gates BYPASSED, structural asserts binding. Uploads run a
#   structural glob assert instead of writing smoke garbage to the HF repo.
#   The production run FIRST executes itself as a redirected smoke child
#   ([phase=smoke]) — the batched-vs-serial bootstrap equivalence gate binds
#   there on the first 2 pairs (plan §4.3 hard-req 3).
# Deferred-failure smoke leg (round-2 fix, deferred-failures-bypass-gates):
#   EPS_SMOKE_CORRUPT_PAIR=<pair_id> bash ... --smoke  fabricates ONE corrupt
#   bundle (BundleSchemaError at fit) and must exit NON-zero from [phase=gate]
#   with a FAILURE sentinel status=bundle_schema_mismatch — never
#   summarize_error: summarize tolerates the deferred pair, uploads still run,
#   check_deferred fires FIRST in gates with the registered status.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SMOKE=""
if [[ "${1:-}" == "--smoke" || "${EPS_SMOKE:-}" == "1" ]]; then SMOKE="1"; fi

DATA_REPO="superkaiba1/explore-persona-space-data"
HF_RC_PREFIX="issue825_role_map_comparison"
PREFIX_HAIKU="issue825_userbase_map/analysis_tensors"
PREFIX_ONPOLICY="issue825_onpolicy_user_turn/analysis_tensors"
PREFIX_REAL="issue825_real_user_turn_null/analysis_tensors/turnstore_realuser"
ALLOWLIST_HF_PATH="issue825_onpolicy_user_turn/raw_completions/generation/row_allowlists.json"
COMMITTED_HAIKU="eval_results/issue_825"
COMMITTED_REAL="eval_results/issue_825/real-user-turn-null"
COMMITTED_ONPOLICY="eval_results/issue_825/onpolicy-user-turn"

if [[ -n "$SMOKE" ]]; then
  ROOT="${EPS_SMOKE_ROOT:-/tmp/issue-825-rolecontrast-smoke}"
  TS_HAIKU="$ROOT/turnstore_haiku"
  TS_REAL="$ROOT/turnstore_real"
  TS_ONPOLICY="$ROOT/turnstore_onpolicy"
  OUT_DIR="$ROOT/eval_results/role-map-comparison"
  PREDS_DIR="$ROOT/preds"
  ALLOWLIST="$ROOT/stage/row_allowlists.json"
  SENTINEL_DIR="$ROOT/logs"
  FOLDS=3 NULLS=3 NBOOT=20 EQ_PAIRS=2
  export EPS_SMOKE=1
  # Bounded MLP secondary at smoke scale (production keeps the 1800 s default);
  # budget-cap hits are a legitimate smoke outcome (blocks-or-logged-caps).
  export EPS_MLP_TIME_BUDGET_S="${EPS_MLP_TIME_BUDGET_S:-120}"
else
  RC_DATA="data/issue_825/rolecontrast"
  TS_HAIKU="$RC_DATA/turnstore_haiku"
  TS_REAL="$RC_DATA/turnstore_real"
  TS_ONPOLICY="$RC_DATA/turnstore_onpolicy"
  OUT_DIR="eval_results/issue_825/role-map-comparison"
  PREDS_DIR="$RC_DATA/preds"   # npz stays OUT of eval_results/ (JSON/text-only rule)
  ALLOWLIST="$RC_DATA/row_allowlists.json"
  SENTINEL_DIR="/workspace/logs"
  FOLDS=5 NULLS=20 NBOOT=1000 EQ_PAIRS=0
  export EPS_SMOKE=""
fi
SENTINEL="$SENTINEL_DIR/issue-825-epm_results-$(date +%s).json"
EPS_T0="$(date +%s)"  # sentinel reports MEASURED gpu-hours (v7/v11 convention)
export EPS_T0

# Credentials at script level (never bare load_dotenv() inside a stdin heredoc — gotcha).
if [[ -f .env ]]; then set -a; source .env; set +a; fi
mkdir -p "$SENTINEL_DIR" "$TS_HAIKU" "$TS_REAL" "$TS_ONPOLICY" "$OUT_DIR" "$PREDS_DIR" \
  "$(dirname "$ALLOWLIST")"
# Xet downloader finalization hangs on this repo (#515 class; v7 r2/r3 lesson).
export HF_XET_DISABLE=1
export EPS_TS_HAIKU="$TS_HAIKU" EPS_TS_REAL="$TS_REAL" EPS_TS_ONPOLICY="$TS_ONPOLICY" \
  EPS_OUT_DIR="$OUT_DIR" EPS_PREDS_DIR="$PREDS_DIR" EPS_ALLOWLIST="$ALLOWLIST" \
  EPS_SENTINEL="$SENTINEL" EPS_DATA_REPO="$DATA_REPO" EPS_HF_RC_PREFIX="$HF_RC_PREFIX" \
  EPS_PREFIX_HAIKU="$PREFIX_HAIKU" EPS_PREFIX_REAL="$PREFIX_REAL" \
  EPS_PREFIX_ONPOLICY="$PREFIX_ONPOLICY" EPS_ALLOWLIST_HF_PATH="$ALLOWLIST_HF_PATH" \
  EPS_COMMITTED_HAIKU="$COMMITTED_HAIKU" EPS_COMMITTED_REAL="$COMMITTED_REAL" \
  EPS_COMMITTED_ONPOLICY="$COMMITTED_ONPOLICY"

echo "[phase=smoke]"
if [[ -z "$SMOKE" ]]; then
  # Production runs the WHOLE wrapper as a smoke child first (unified smoke
  # architecture). Output REDIRECTED: the child's terminal [phase=done] must
  # never reach the main log (poller [phase=done] reservation, #545).
  SMOKE_LOG="$SENTINEL_DIR/issue-825-rolecontrast-smoke.log"
  if ! EPS_SMOKE=1 EPS_SMOKE_ROOT="/tmp/issue-825-rolecontrast-smoke" \
      bash "${BASH_SOURCE[0]}" --smoke > "$SMOKE_LOG" 2>&1; then
    echo "[phase=smoke] FAILED — tail of $SMOKE_LOG:"
    tail -n 80 "$SMOKE_LOG"
    uv run python scripts/issue825_role_contrast.py fail-sentinel --phase smoke \
      --out-dir "$OUT_DIR" --preds-dir "$PREDS_DIR" --sentinel "$SENTINEL"
    exit 1
  fi
  echo "[phase=smoke] child PASS (equivalence gate bound on 2 pairs; log: $SMOKE_LOG)"
else
  echo "[phase=smoke] running AS the smoke (EPS_SMOKE=1) — no recursive child"
fi

echo "[phase=stage]"
if [[ -n "$SMOKE" ]]; then
  # Same phase, synthetic inputs: tiny .pt bundles + allowlist via the runner's
  # fabricate-smoke mode (schema-identical to the extractor shard contract).
  uv run python scripts/issue825_role_contrast.py fabricate-smoke \
    --haiku-dir "$TS_HAIKU" --real-dir "$TS_REAL" --onpolicy-dir "$TS_ONPOLICY" \
    --allowlists "$ALLOWLIST"
else
  uv run python - <<'PY'
import concurrent.futures as cf
import json
import os
import signal
import time
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # v7 run-2 lesson: transfer hang
signal.alarm(7200)  # 120-min stage alarm (plan §4.3 — 172.6 GB)
from huggingface_hub import HfApi, hf_hub_download

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (source .env before staging)"
repo = os.environ["EPS_DATA_REPO"]
api = HfApi()
# ONE coherent revision for the whole stage (listing + every per-file download).
revision = api.repo_info(repo, repo_type="dataset").sha
targets = {
    os.environ["EPS_PREFIX_HAIKU"]: Path(os.environ["EPS_TS_HAIKU"]),
    os.environ["EPS_PREFIX_REAL"]: Path(os.environ["EPS_TS_REAL"]),
    os.environ["EPS_PREFIX_ONPOLICY"]: Path(os.environ["EPS_TS_ONPOLICY"]),
}
models, formats = ("instruct", "pretrained"), ("chat", "naturalistic")
wanted_stems = {f"{m}_{f}_m_shard" for m in models for f in formats}
jobs: list[tuple[str, Path]] = []
total_bytes = 0
for prefix, dst in targets.items():
    # SCOPED list_repo_tree (server-side prefix) — NEVER snapshot_download /
    # bare list_repo_files on this ~1M-file repo (gotchas.md #833).
    entries = list(
        api.list_repo_tree(repo, path_in_repo=prefix, repo_type="dataset",
                           recursive=False, revision=revision)
    )
    files = [
        e for e in entries
        if getattr(e, "size", None) is not None
        and e.path.rsplit("/", 1)[-1].endswith(".pt")
        and any(e.path.rsplit("/", 1)[-1].startswith(s) for s in wanted_stems)
    ]
    by_bundle: dict[str, int] = {}
    for e in files:
        stem = "_".join(e.path.rsplit("/", 1)[-1].split("_")[:3])
        by_bundle[stem] = by_bundle.get(stem, 0) + 1
        total_bytes += e.size
        jobs.append((e.path, dst))
    assert len(by_bundle) == 4, f"{prefix}: expected 4 m-bundles, found {sorted(by_bundle)}"
    print(f"stage: {prefix} -> {len(files)} shards ({sorted(by_bundle.items())})")
print(f"stage: {len(jobs)} files, {total_bytes / 1e9:.1f} GB @ rev {revision[:12]}")

def fetch(job):
    path, dst = job
    for attempt in range(4):
        try:
            local = hf_hub_download(repo_id=repo, repo_type="dataset",
                                    revision=revision, filename=path)
            target = dst / path.rsplit("/", 1)[-1]
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(local)
            return path
        except Exception as e:  # retry + linear backoff (HF 5xx/429, gotchas.md)
            if attempt == 3:
                raise
            print(f"stage: retry {attempt + 1} for {path}: {type(e).__name__}: {e}")
            time.sleep(20 * (attempt + 1))

with cf.ThreadPoolExecutor(max_workers=6) as pool:
    for done in pool.map(fetch, jobs):
        print(f"stage: staged {done}")
allow_local = hf_hub_download(
    repo_id=repo, repo_type="dataset", revision=revision,
    filename=os.environ["EPS_ALLOWLIST_HF_PATH"],
)
allow_dst = Path(os.environ["EPS_ALLOWLIST"])
allow_dst.write_text(Path(allow_local).read_text())
allow = json.loads(allow_dst.read_text())
assert len(allow) == 4, f"row_allowlists.json keys: {sorted(allow)}"
out_dir = Path(os.environ["EPS_OUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "stage_manifest.json").write_text(json.dumps({
    "revision": revision,
    "prefixes": sorted(str(k) for k in targets),
    "n_files": len(jobs),
    "total_bytes": total_bytes,
    "allowlist_keys": {k: len(v) for k, v in allow.items()},
}, indent=2))
print("stage: ok (12 m-bundles + row_allowlists.json, one pinned revision)")
PY
fi

# Committed comparison JSONs ride the repo clone (plan §12.11) — presence is
# structural and binds in smoke too (a sparse/broken checkout fails HERE, not
# mid-gate). 8 parent + 8 real + 8 onpolicy cells.
uv run python - <<'PY'
import os
from pathlib import Path

missing = []
for env, cells in (
    ("EPS_COMMITTED_HAIKU", 8),
    ("EPS_COMMITTED_REAL", 8),
    ("EPS_COMMITTED_ONPOLICY", 8),
):
    d = Path(os.environ[env])
    found = sorted(d.glob("cells_M_*.json"))
    if len(found) < cells:
        missing.append(f"{d}: {len(found)}/{cells}")
print("stage: committed comparison JSONs present (8+8+8)" if not missing else missing)
assert not missing, f"committed comparison JSONs missing: {missing} (broken/sparse checkout?)"
PY

echo "[phase=fit]"
# Per-pair crashes DEFER to fit_failures.json (MF-C): summarize TOLERATES the
# deferred pairs' missing JSONs, uploads still run, and the POST-upload gates
# HALT with the registered status (bundle_schema_mismatch |
# fit_deferred_failure). Only a TOP-LEVEL fit crash (the loop itself dying) is
# upload-then-exit via fail-sentinel here.
if ! uv run python scripts/issue825_role_contrast.py fit \
  --haiku-dir "$TS_HAIKU" --real-dir "$TS_REAL" --onpolicy-dir "$TS_ONPOLICY" \
  --allowlists "$ALLOWLIST" --out-dir "$OUT_DIR" --preds-dir "$PREDS_DIR" \
  --committed-haiku "$COMMITTED_HAIKU" --committed-real "$COMMITTED_REAL" \
  --committed-onpolicy "$COMMITTED_ONPOLICY" \
  --folds "$FOLDS" --null-draws "$NULLS" --n-boot "$NBOOT" --seed 0 \
  --equivalence-gate-pairs "$EQ_PAIRS" --resume; then
  uv run python scripts/issue825_role_contrast.py fail-sentinel --phase fit \
    --out-dir "$OUT_DIR" --preds-dir "$PREDS_DIR" --sentinel "$SENTINEL"
  echo "[phase=fit] FAILED — produced JSONs uploaded, FAILURE sentinel written (upload-then-exit)"
  exit 1
fi

echo "[phase=summarize]"
# summarize exits non-zero ONLY on a genuine summarize bug (a missing pair
# JSON with NO deferred record, or a crash) — a deferred fit failure is
# tolerated above, so status summarize_error is never the deferred class.
if ! uv run python scripts/issue825_role_contrast.py summarize \
  --out-dir "$OUT_DIR" \
  --committed-haiku "$COMMITTED_HAIKU" --committed-real "$COMMITTED_REAL" \
  --committed-onpolicy "$COMMITTED_ONPOLICY"; then
  uv run python scripts/issue825_role_contrast.py fail-sentinel --phase summarize \
    --out-dir "$OUT_DIR" --preds-dir "$PREDS_DIR" --sentinel "$SENTINEL"
  echo "[phase=summarize] FAILED — produced JSONs uploaded, FAILURE sentinel written"
  exit 1
fi

echo "[phase=upload]"
if [[ -n "$SMOKE" ]]; then
  uv run python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["EPS_OUT_DIR"])
preds = Path(os.environ["EPS_PREDS_DIR"])
pairs = sorted(out.rglob("pair_*.json"))
cells = sorted(out.rglob("cells_M_*.json"))
nulls = sorted(out.rglob("nulls_M_*.json"))
npz = sorted(preds.glob("preds_pair_*.npz"))
deferred = sorted(out.rglob("fit_failures.json"))
if deferred:
    # Deferred fit failure(s) recorded: pair JSONs are LEGITIMATELY missing, so
    # the strict counts are relaxed — production uploads whatever exists, and
    # the post-upload gates HALT next with the REGISTERED status (round-2 fix).
    n_def = sum(len(json.loads(p.read_text())) for p in deferred)
    assert (out / "headline_metrics.json").exists()  # summarize tolerated + still wrote it
    print(f"[smoke] upload structural: {n_def} deferred fit failure(s) recorded — strict "
          f"counts relaxed ({len(pairs)} pairs, {len(npz)} npz would upload); gates HALT next")
else:
    assert len(pairs) == 12, f"expected 12 pair JSONs, found {len(pairs)}"
    assert len(cells) == 24, f"expected 24 cell payloads, found {len(cells)}"
    assert len(nulls) == 24, f"expected 24 null payloads, found {len(nulls)}"
    assert len(npz) == 12, f"expected 12 preds npz, found {len(npz)}"
    assert (out / "headline_metrics.json").exists()
    manifest = json.loads((out / "preds_manifest.json").read_text())
    assert len(manifest["files"]) == 12, sorted(manifest["files"])
    print(f"[smoke] upload structural assert PASS: {len(pairs)} pairs, {len(cells)} cells, "
          f"{len(nulls)} nulls, {len(npz)} npz + headline + manifest would upload")
PY
else
  uv run python - <<'PY'
import os
import signal

signal.alarm(5400)  # 90 min: ~0.7 GB npz + JSONs at the plain-CDN path
from huggingface_hub import HfApi

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (source .env before upload)"
api = HfApi()
repo = os.environ["EPS_DATA_REPO"]
prefix = os.environ["EPS_HF_RC_PREFIX"]
api.upload_folder(
    folder_path=os.environ["EPS_OUT_DIR"],
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/eval_results_mirror",
    commit_message="issue-825 role-map-comparison: UPLOAD-a (eval JSONs + headline, BEFORE gates)",
)
api.upload_folder(
    folder_path=os.environ["EPS_PREDS_DIR"],
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/analysis_tensors",
    allow_patterns=["preds_pair_*.npz"],
    commit_message="issue-825 role-map-comparison: UPLOAD-b (fp16 preds npz, BEFORE gates)",
)
print("upload: ok (eval JSONs + preds npz, ALL BEFORE binding gates — MF-C)")
PY
fi

echo "[phase=gate]"
# Binding gates (plan §7, POST-upload): deferred/schema -> row-alignment >=0.95
# -> reproduction ±0.01 on the 20 anchored cells -> coverage (+equivalence).
# Numeric gates bypassed under EPS_SMOKE=1; structural asserts binding.
uv run python scripts/issue825_role_contrast.py gates \
  --out-dir "$OUT_DIR" \
  --committed-haiku "$COMMITTED_HAIKU" --committed-real "$COMMITTED_REAL" \
  --committed-onpolicy "$COMMITTED_ONPOLICY" --sentinel "$SENTINEL"

echo "[phase=sentinel]"
EPS_GIT_SHA="$(git rev-parse HEAD)"
EPS_WORKTREE="$(pwd)"
export EPS_GIT_SHA EPS_WORKTREE
uv run python scripts/issue825_role_contrast.py success-sentinel \
  --out-dir "$OUT_DIR" --sentinel "$SENTINEL"
echo "[phase=done]"
