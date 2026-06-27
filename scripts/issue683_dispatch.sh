#!/usr/bin/env bash
# Issue #683 production dispatcher — the 5 serial GPU extracts on ONE
# auto-routed A100-80, then upload + the poll_pipeline results sentinel.
# Eval-/analysis-only (no Hydra, no training). CPU scoring (A7 + key×metric +
# figures) runs OFF-POD on the VM after teardown (NOT in this dispatcher).
#
# Architecturally UNIFIED smoke == sweep (plan §7; PASS_UNIFIED):
#   --smoke runs the IDENTICAL dispatcher with a tiny per-extract slice
#   (--smoke-sources A1 / --smoke-seeds 42, --n-questions 2, --n-claims 2,
#   --max-rows 2, --max-new-tokens 16). Same subprocess shape, same env
#   injection, same upload + sentinel surface, same [phase=done] teardown.
#   EVERY phase the dispatcher executes (marker Δv, marker t_cb, syco panel
#   top-up, syco Δv, syco t_cb, upload, sentinel) reads its cell subset from
#   the SAME --smoke-* / sweep-default vars — no phase re-enumerates a full
#   registered grid.
#
# Usage (GCP --workload-cmd; $WORKLOAD_ROOT exported by the startup script):
#   REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue683_dispatch.sh
#   REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue683_dispatch.sh --smoke
#
# Pod-side contract (CLAUDE.md): NEVER shells out to scripts/task.py. Signals
# the orchestrator ONLY via the /workspace/logs/issue-683-*.json sentinel that
# poll_pipeline.py drains + the [phase=...] log lines it tails.

set -uo pipefail  # NOT set -e — we want the sentinel + [phase=done] to fire even on a phase failure
: "${REPO_ROOT:?REPO_ROOT must be set; export it (REPO_ROOT=\"\$WORKLOAD_ROOT\") before invoking}"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

SMOKE=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        --dry-run) DRY_RUN=1 ;;  # GPU-bound-phase carve-out: exercise the
                                  # cell-iteration + env + sentinel + [phase=done]
                                  # plumbing WITHOUT the GPU extracts (echo each).
        *) ;;
    esac
done

# ── Cell-subset parameterization — EVERY phase reads from these (unification). ─
if [ "$SMOKE" -eq 1 ]; then
    MARKER_SOURCES="A1"
    SYCO_SEEDS="42"
    N_QUESTIONS=2
    N_CLAIMS=2
    MAX_NEW_TOKENS=16
    EXTRA_TCB_MAX_ROWS="--max-rows 4"
    GLOB_LABEL="smoke"
else
    MARKER_SOURCES="A1,A2,A3,A4,A5"
    SYCO_SEEDS="42,137"
    N_QUESTIONS=20
    N_CLAIMS=30
    MAX_NEW_TOKENS=512
    EXTRA_TCB_MAX_ROWS=""
    GLOB_LABEL="sweep"
fi

LOG_DIR="logs/issue_683"
mkdir -p "$LOG_DIR"
ATENSORS_DIR="eval_results/issue_683/analysis_tensors"

echo "[phase=preflight] === i683 dispatcher $(date -Iseconds) smoke=$SMOKE sources=$MARKER_SOURCES seeds=$SYCO_SEEDS ==="

# Marker token + <|im_end|> id assert at launch (marker-leakage rule: wire the
# in-process assert into the dispatcher so a wrong token fails at startup).
# Skipped under --dry-run (it loads the 7B tokenizer; the dry-run only
# exercises the cell-iteration / sentinel / [phase=done] plumbing).
if [ "$DRY_RUN" -eq 0 ]; then
    uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
ids = tok.encode(' ※', add_special_tokens=False)
assert ids == [83399], f'marker token id drift {ids}'
im_end = tok.convert_tokens_to_ids('<|im_end|>')
assert im_end == 151645, f'<|im_end|> id drift {im_end}'
print('marker id OK 83399; <|im_end|> OK 151645')
" || { echo '[phase=preflight_failed] marker/im_end id assert FAILED' >&2; }
fi

FAILED_FILE="$LOG_DIR/dispatch_failed.txt"
: > "$FAILED_FILE"

run_phase() {
    # run_phase <phase-tag> <log-name> -- <cmd...>
    local tag="$1"; local logname="$2"; shift 3  # drop the literal --
    echo "[phase=$tag] === $tag $(date -Iseconds) ==="
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[phase=${tag}_dryrun] would run: $*"
        echo "[phase=${tag}_done] $tag (dry-run; no GPU)"
        return 0
    fi
    local rc=0
    "$@" > "$LOG_DIR/$logname" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "$tag (rc=$rc, see $LOG_DIR/$logname)" >> "$FAILED_FILE"
        echo "[phase=${tag}_failed] rc=$rc — see $LOG_DIR/$logname" >&2
    else
        echo "[phase=${tag}_done] $tag complete"
    fi
}

# ── 1. Marker Δv extract (L14 EOR slot) over loc-arm sources × #604 panel. ─────
run_phase dv_marker dv_marker.log -- \
    uv run python scripts/issue683_extract_dv_marker.py \
    --source-list "$MARKER_SOURCES" --layer 14 \
    --n-questions "$N_QUESTIONS" --max-new-tokens "$MAX_NEW_TOKENS"

# ── 2. Marker t_{C,B} extract (L14 answer-span). ───────────────────────────────
# shellcheck disable=SC2086  # EXTRA_TCB_MAX_ROWS is an intentional word-split flag
run_phase tcb_marker tcb_marker.log -- \
    uv run python scripts/issue683_extract_tcb.py \
    --behavior marker --layer 14 --source-list "$MARKER_SOURCES" $EXTRA_TCB_MAX_ROWS

# ── 3. Sycophancy panel top-up (#612 L20 panel centroids — reuse #649 extractor). ─
# The full panel centroids already exist on HF (#612); the top-up fills any
# panel context missing at L20. In smoke we skip the top-up (the dv extractor
# generates c_C' on the fly from the panel set), so this is a sweep-only phase
# but its slice still derives from the same panel subset.
if [ "$SMOKE" -eq 0 ]; then
    run_phase syco_panel_topup syco_panel_topup.log -- \
        uv run python scripts/issue649_extract_panel_earlylayer.py
else
    echo "[phase=syco_panel_topup_skipped] smoke: dv extractor builds c_C' from the panel set on the fly"
fi

# ── 4. Sycophancy Δv extract (L20 answer-span mean) over villain × panel. ──────
run_phase dv_syco dv_syco.log -- \
    uv run python scripts/issue683_extract_dv_sycophancy.py \
    --seeds "$SYCO_SEEDS" --layer 20 \
    --n-claims "$N_CLAIMS" --max-new-tokens "$MAX_NEW_TOKENS"

# ── 5. Sycophancy t_{C,B} extract (L20 answer-span). ───────────────────────────
# shellcheck disable=SC2086
run_phase tcb_syco tcb_syco.log -- \
    uv run python scripts/issue683_extract_tcb.py \
    --behavior sycophancy --layer 20 --source-list villain $EXTRA_TCB_MAX_ROWS

# ── 6. Upload all analysis_tensors/* to HF data repo. ──────────────────────────
echo "[phase=upload] === upload analysis_tensors -> issue683_key_gate/analysis_tensors $(date -Iseconds) ==="
upload_rc=0
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[phase=upload_dryrun] would upload $ATENSORS_DIR -> issue683_key_gate/analysis_tensors"
    echo "[phase=upload_done] (dry-run; no HF write)"
else
uv run python - "$ATENSORS_DIR" <<'PY' > "$LOG_DIR/upload.log" 2>&1 || upload_rc=$?
import sys
from pathlib import Path

from huggingface_hub import HfApi

from explore_persona_space.experiments.issue_683 import HF_ANALYSIS_TENSORS_PREFIX, HF_DATA_REPO

local = Path(sys.argv[1])
if not local.is_dir():
    print(f"no analysis_tensors dir at {local} — nothing to upload")
    sys.exit(0)
api = HfApi()
info = api.upload_folder(
    folder_path=str(local),
    path_in_repo=HF_ANALYSIS_TENSORS_PREFIX,
    repo_id=HF_DATA_REPO,
    repo_type="dataset",
    commit_message="#683 analysis tensors (Δv + t_{C,B}, marker + sycophancy)",
)
# verify a couple of the written files resolved on the Hub.
listed = set(api.list_repo_files(HF_DATA_REPO, repo_type="dataset"))
n_local = sum(1 for p in local.rglob("*") if p.is_file())
n_hub = sum(1 for f in listed if f.startswith(HF_ANALYSIS_TENSORS_PREFIX))
print(f"uploaded {n_local} local files; {n_hub} now under {HF_ANALYSIS_TENSORS_PREFIX} on the Hub")
if n_hub == 0:
    raise RuntimeError("upload verification FAILED — 0 files under the prefix on the Hub")
print(info)
PY
if [ "$upload_rc" -ne 0 ]; then
    echo "upload (rc=$upload_rc, see $LOG_DIR/upload.log)" >> "$FAILED_FILE"
    echo "[phase=upload_failed] rc=$upload_rc — see $LOG_DIR/upload.log" >&2
else
    echo "[phase=upload_done] analysis_tensors uploaded"
fi
fi  # end --dry-run / live-upload branch

# ── 7. Results sentinel (poll_pipeline contract) — BEFORE [phase=done]. ────────
if [ -d /workspace ]; then
    FAILED_LIST="$(tr '\n' ';' < "$FAILED_FILE" 2>/dev/null || true)"
    ANY_FAILED=0
    [ -s "$FAILED_FILE" ] && ANY_FAILED=1
    uv run python - "$FAILED_LIST" "$ANY_FAILED" "$GLOB_LABEL" <<'PY'
import datetime
import json
import sys
from pathlib import Path

failed_list, any_failed, glob_label = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
kind = "epm:failure" if any_failed else "epm:results"
note = {
    "issue": 683,
    "phase_label": glob_label,
    "analysis_tensors": "superkaiba1/explore-persona-space-data/issue683_key_gate/analysis_tensors/",
    "deliverables": (
        "Δv + t_{C,B} banks for marker (L14 EOR slot) + sycophancy (L20 answer-span); "
        "CPU scoring (A7 + key×metric leaderboard + figures) runs off-pod on the VM."
    ),
    "failed_phases": failed_list,
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "task_id": 683,
    "by": "issue683_dispatch",
    "ts": datetime.datetime.now(datetime.UTC).isoformat(),
    "note": json.dumps(note),
}
if any_failed:
    sentinel["failure_class"] = "code"
log_dir = Path("/workspace/logs")
log_dir.mkdir(parents=True, exist_ok=True)
out = log_dir / f"issue-683-{kind.replace(':', '_')}-{int(datetime.datetime.now().timestamp())}.json"
out.write_text(json.dumps(sentinel, indent=2))
print(f"wrote sentinel {out} (kind={kind})")
PY
fi

if [ -s "$FAILED_FILE" ]; then
    echo "[phase=dispatch_failed] FATAL: phases failed: $(tr '\n' ' ' < "$FAILED_FILE")" >&2
    echo "[phase=done]"  # terminal marker still emitted so the poller resolves cleanly
    exit 3
fi

echo "[phase=done] === i683 dispatch ($GLOB_LABEL) all extracts + upload complete $(date -Iseconds) ==="
