#!/bin/bash
# issue833_gcp_nonemit.sh — Phase N1 of the #833 nonverbatim-profile-ablation
# follow-up round (plan v10 §4): non-emission + matched-N + eq5 re-extraction
# on ONE GCE 1x A100-80 (dispatch: dispatch_issue.py --issue 833 --intent
# lora-7b --repo-branch issue-833 --workload-cmd 'bash scripts/issue833_gcp_nonemit.sh').
#
# Sequence (all phases re-use issue833_extract_onpolicy.py entrypoints; every
# phase's cell list derives from the SAME committed Phase-N0 artifacts —
# eval_results/issue_833/emission_rate/{retention_manifest,matchedN_sample_
# indices,eq5_sample_indices}.json, which arrive with the repo clone):
#   stage      — scoped list_repo_tree + threaded per-file hf_hub_download of the
#                fact raw completions (16 gen + 16 rbase JSONs) + the 2-source
#                parity slice of the persisted r7e analysis_tensors (the
#                round-7b/7c staging shape; snapshot_download is BARRED against
#                the ~1M-file repo — gotchas.md).
#   parity     — cross-run extraction parity gate (2 sources x 30 targets x 3
#                layers, rel-L2 <= 1e-3 vs the staged r7e npz). rc=6 fires the
#                REGISTERED contingency: full-text fact re-extraction in-run
#                (namespace analysis_tensors_fullrerun) so every paired contrast
#                is within-run by construction (plan v10 §4). Any other rc = crash.
#   extract    — nonemit + matchedn + eq5 in ONE per-adapter pass (union
#                teacher-forcing; per-cell npz + .done sentinels, resume-safe).
#   upload     — one upload_folder commit per namespace + SCOPED exact-set
#                verify (hub.verify_repo_paths_uploaded) BEFORE teardown.
#
# GCE contract: the startup script clones the repo branch to $WORKLOAD_ROOT,
# exports REPO_ROOT="$WORKLOAD_ROOT", threads HF_TOKEN via instance metadata
# (no .env on this lane — conditional sourcing only), and the rendered success
# tail writes the completion sentinel + eps/phase=done when this script exits 0.
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT" || exit 1
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi  # RunPod fallback lane only; GCE has no .env
mkdir -p logs eval_results/issue_833
MAIN_LOG="logs/issue833_gcp_nonemit.log"
OUT=eval_results/issue_833
SEED=42
# Parity-probe sources: first two of _sources_for("fact") = (sp_swe, sp_doctor)
# — matches stage_parity_crossrun's default; passed explicitly so the staged
# parity slice and the gate always agree.
PARITY_SRC1="${PARITY_SRC1:-sp_swe}"
PARITY_SRC2="${PARITY_SRC2:-sp_doctor}"

# Round-7c lesson: the xet per-file token refresh dies at multi-k-file staging;
# force the plain HTTP/CDN path for the staging downloads (final uploads are
# small npz/JSON commits, unaffected).
export HF_XET_DISABLE=1

echo "[phase=stage] scoped list + per-file download ($(date -u +%H:%M:%S))" | tee -a "$MAIN_LOG"
PARITY_SRC1="$PARITY_SRC1" PARITY_SRC2="$PARITY_SRC2" uv run python - <<'PY' || { echo "[phase=stage] FAILED" | tee -a "$MAIN_LOG"; exit 1; }
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download

REPO = "superkaiba1/explore-persona-space-data"
SRC1, SRC2 = os.environ["PARITY_SRC1"], os.environ["PARITY_SRC2"]
PREFIXES = [
    "issue833_onpolicy_map/raw_completions/generation/fact",
    "issue833_onpolicy_map/raw_completions/rbase/fact",
    f"issue833_onpolicy_map/analysis_tensors/fact/{SRC1}_seed42",
    f"issue833_onpolicy_map/analysis_tensors/fact/{SRC2}_seed42",
]
api = HfApi()
paths = []
for pref in PREFIXES:
    got = [
        e.path
        for e in api.list_repo_tree(REPO, path_in_repo=pref, repo_type="dataset", recursive=True)
        if e.path.endswith((".json", ".npz"))
    ]
    print(f"{pref}: {len(got)} files", flush=True)
    if not got:
        print(f"EMPTY prefix {pref} — refusing", flush=True)
        sys.exit(1)
    paths += got
print(f"listed {len(paths)} files total", flush=True)


def fetch(p: str, attempts: int = 5) -> str | None:
    """Return None on success, the path on hard failure (never raises)."""
    for attempt in range(attempts):
        try:
            hf_hub_download(REPO, p, repo_type="dataset", local_dir="hf_stage")
            return None
        except Exception:  # noqa: BLE001 — retry w/ backoff; report, don't kill the pool
            if attempt == attempts - 1:
                return p
            time.sleep(20 * (attempt + 1))
    return p


done, failed = 0, []
with ThreadPoolExecutor(max_workers=6) as ex:
    futs = [ex.submit(fetch, p) for p in paths]
    for f in as_completed(futs):
        bad = f.result()
        if bad:
            failed.append(bad)
        else:
            done += 1
        if (done + len(failed)) % 100 == 0:
            print(f"downloaded {done}/{len(paths)} (failed so far: {len(failed)})", flush=True)
still = []
for p in failed:  # serial second pass over stragglers
    time.sleep(2)
    if fetch(p, attempts=3):
        still.append(p)
    else:
        done += 1
print(f"downloaded {done}/{len(paths)}; unrecovered: {len(still)}", flush=True)
for p in still[:20]:
    print("  FAILED:", p, flush=True)
sys.exit(0 if done == len(paths) else 1)
PY
# Prefix-strip mirror: hub issue833_onpolicy_map/… → eval_results/issue_833/…
mkdir -p "$OUT/raw_completions/generation" "$OUT/raw_completions/rbase" "$OUT/analysis_tensors"
rm -rf "$OUT/raw_completions/generation/fact" "$OUT/raw_completions/rbase/fact"
cp -a hf_stage/issue833_onpolicy_map/raw_completions/generation/fact "$OUT/raw_completions/generation/" || exit 1
cp -a hf_stage/issue833_onpolicy_map/raw_completions/rbase/fact "$OUT/raw_completions/rbase/" || exit 1
rm -rf "$OUT/analysis_tensors/fact"
mkdir -p "$OUT/analysis_tensors/fact"
cp -a "hf_stage/issue833_onpolicy_map/analysis_tensors/fact/${PARITY_SRC1}_seed42" "$OUT/analysis_tensors/fact/" || exit 1
cp -a "hf_stage/issue833_onpolicy_map/analysis_tensors/fact/${PARITY_SRC2}_seed42" "$OUT/analysis_tensors/fact/" || exit 1
NG=$(find "$OUT/raw_completions/generation/fact" -name '*.json' | wc -l)
NR=$(find "$OUT/raw_completions/rbase/fact" -name '*.json' | wc -l)
NP=$(find "$OUT/analysis_tensors/fact" -name '*.npz' | wc -l)
echo "[phase=stage] staged: $NG gen + $NR rbase JSONs, $NP parity npz (expect 16+16+180)" | tee -a "$MAIN_LOG"
[ "$NG" -eq 16 ] && [ "$NR" -eq 16 ] && [ "$NP" -eq 180 ] || { echo "[phase=stage] INCOMPLETE" | tee -a "$MAIN_LOG"; exit 1; }
# Consumer-open probe (leg (iv), plan §10 reuse map): one staged JSON parses.
uv run python -c "
import json, sys
d = json.loads(open('$OUT/raw_completions/generation/fact/${PARITY_SRC1}_seed${SEED}.json').read())
assert d['behavior'] == 'fact' and len(d['responses']) == 900, (d['behavior'], len(d['responses']))
print('consumer-open probe OK:', d['source_cid'], len(d['responses']), 'rows')
" || { echo "[phase=stage] consumer-open probe FAILED" | tee -a "$MAIN_LOG"; exit 1; }

# ── Parity gate (HALT-capable, registered contingency on rc=6) ──────────────
echo "[phase=parity-crossrun] start ($(date -u +%H:%M:%S))" | tee -a "$MAIN_LOG"
uv run python scripts/issue833_extract_onpolicy.py \
  --stage parity-crossrun --behavior fact --seed "$SEED" --out "$OUT" \
  --parity-sources "$PARITY_SRC1" "$PARITY_SRC2" \
  >> logs/issue833_nonemit_parity.log 2>&1 < /dev/null
PARITY_RC=$?
echo "[phase=parity-crossrun] rc=$PARITY_RC" | tee -a "$MAIN_LOG"
FULLRERUN=0
if [ "$PARITY_RC" -eq 6 ]; then
  # Registered contingency (plan v10 §4): re-extract the FULL-TEXT fact legs
  # in-run so every paired contrast is within-run by construction (~+2 GPU-h,
  # inside the 4.5 GPU-h bound). Phase N2 consumes analysis_tensors_fullrerun
  # via --fulltext-npz-root.
  echo "[phase=fullrerun] parity FAIL — running the registered full-text re-extraction contingency" | tee -a "$MAIN_LOG"
  FULLRERUN=1
  uv run python scripts/issue833_extract_onpolicy.py \
    --stage extract --behavior fact --seed "$SEED" --out "$OUT" \
    --namespace-override analysis_tensors_fullrerun \
    >> logs/issue833_nonemit_fullrerun.log 2>&1 < /dev/null \
    || { echo "[phase=fullrerun] FAILED" | tee -a "$MAIN_LOG"; exit 1; }
elif [ "$PARITY_RC" -ne 0 ]; then
  echo "[phase=parity-crossrun] CRASH (rc=$PARITY_RC, not the registered 6)" | tee -a "$MAIN_LOG"
  exit "$PARITY_RC"
fi

# ── Subset extraction: nonemit + matchedn + eq5 in ONE per-adapter pass ──────
echo "[phase=extract-subsets] start ($(date -u +%H:%M:%S))" | tee -a "$MAIN_LOG"
uv run python scripts/issue833_extract_onpolicy.py \
  --stage extract --behavior fact --seed "$SEED" --out "$OUT" \
  --response-subset nonemit matchedn eq5 \
  >> logs/issue833_nonemit_extract.log 2>&1 < /dev/null \
  || { echo "[phase=extract-subsets] FAILED" | tee -a "$MAIN_LOG"; exit 1; }
for NS in analysis_tensors_nonemit analysis_tensors_matchedN analysis_tensors_nonemit_eq5; do
  N=$(find "$OUT/$NS" -name '*.npz' 2>/dev/null | wc -l)
  echo "[phase=extract-subsets] $NS: $N npz (expect 873 = 291 cells x 3 layers)" | tee -a "$MAIN_LOG"
done

# ── Upload + scoped verify BEFORE teardown (plan §6.5) ───────────────────────
echo "[phase=upload-subset] start ($(date -u +%H:%M:%S))" | tee -a "$MAIN_LOG"
unset HF_XET_DISABLE  # staging workaround only; uploads take the default path
uv run python scripts/issue833_extract_onpolicy.py \
  --stage upload-subset --behavior fact --seed "$SEED" --out "$OUT" \
  --response-subset nonemit matchedn eq5 \
  >> logs/issue833_nonemit_upload.log 2>&1 < /dev/null \
  || { echo "[phase=upload-subset] FAILED" | tee -a "$MAIN_LOG"; exit 1; }
if [ "$FULLRERUN" -eq 1 ]; then
  uv run python scripts/issue833_extract_onpolicy.py \
    --stage upload-subset --behavior fact --seed "$SEED" --out "$OUT" \
    --response-subset nonemit --namespace-override analysis_tensors_fullrerun \
    >> logs/issue833_nonemit_upload.log 2>&1 < /dev/null \
    || { echo "[phase=upload-subset] fullrerun upload FAILED" | tee -a "$MAIN_LOG"; exit 1; }
fi

# Log upload (small text, non-LFS path).
uv run python - <<'PY' || { echo "[phase=upload-logs] FAILED (non-fatal for science outputs)" | tee -a "$MAIN_LOG"; }
from huggingface_hub import HfApi

HfApi().upload_folder(
    folder_path="logs",
    path_in_repo="issue833_onpolicy_map/nonemit_outputs/logs",
    repo_id="superkaiba1/explore-persona-space-data",
    repo_type="dataset",
    allow_patterns=["issue833_nonemit_*", "issue833_gcp_nonemit.log"],
    commit_message="issue-833 nonverbatim ablation N1 logs",
)
print("log upload done")
PY

echo "[phase=done] fullrerun=$FULLRERUN ($(date -u +%H:%M:%S))" | tee -a "$MAIN_LOG"
exit 0
