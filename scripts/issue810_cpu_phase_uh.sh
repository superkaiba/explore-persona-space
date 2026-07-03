#!/usr/bin/env bash
# Issue #810 follow-up `user-header-newline-summary` CPU-phase driver (plan v11
# §4.5/§10): poll HF for uh_summaries.pt + the GPU-phase recon/crosslayer JSONs
# -> Phase R-x read-out (all-46 rows, FULL enlarged-axis null rerun — NO join
# on this leg, A5 shared-rng) -> band presence check -> paired bootstrap vs the
# mean benchmark -> figures -> HF mirror -> guarded git commit. Pure sequencing
# of the code-reviewed CLIs (no logic) on a cpu-mid instance, mirroring
# issue810_cpu_phase_g1.sh. Pod-side signaling: [phase=...] log lines + the
# end-of-run sentinel ONLY — NEVER a task.py shellout (branch-guard).
#
# Dry-run sequencing pass: I810_DRYRUN=1 bash scripts/issue810_cpu_phase_uh.sh
set -euo pipefail

OUT=eval_results/issue_810/user-header-newline-summary
FIGS=figures/issue_810/user-header-newline-summary
UH_PACK=data/issue_810/uh_summaries.pt
HF_MIRROR=issue810_results/user-header-newline-summary
LOGDIR="${I810_LOGDIR:-logs/issue_810}"
DRY="${I810_DRYRUN:-0}"

maybe() { if [ "$DRY" = "1" ]; then echo "[dryrun] $*"; else "$@"; fi; }

echo "[phase=branch_sync]"
if [ "$(git rev-parse --abbrev-ref HEAD)" != "issue-810" ]; then
  maybe git fetch origin issue-810
  maybe git checkout issue-810
fi
maybe mkdir -p "$OUT" "$FIGS" "$LOGDIR" "$(dirname "$UH_PACK")"

echo "[phase=poll_inputs] waiting for uh_summaries.pt + GPU-phase JSONs on HF"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] poll HF for $HF_MIRROR/uh_summaries.pt + $HF_MIRROR/eval_results/{recon,crosslayer} JSONs (sleep 300, timeout 12h)"
else
  uv run python - <<PY
import shutil, sys, time
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download, list_repo_files
pack = "$HF_MIRROR/uh_summaries.pt"
jsons = ["reconstruction_skill_user_header.json", "null_matrix_user_header.json",
         "crosslayer_xbnd.json", "null_matrix_crosslayer_xbnd.json"]
deadline = time.time() + 12 * 3600
while time.time() < deadline:
    files = set(list_repo_files("superkaiba1/explore-persona-space-data", repo_type="dataset"))
    if pack in files and all(f"$HF_MIRROR/eval_results/{n}" in files for n in jsons):
        p = hf_hub_download("superkaiba1/explore-persona-space-data", pack, repo_type="dataset")
        shutil.copy(p, "$UH_PACK")
        for n in jsons:
            q = hf_hub_download("superkaiba1/explore-persona-space-data",
                                f"$HF_MIRROR/eval_results/{n}", repo_type="dataset")
            shutil.copy(q, f"$OUT/{n}")
        print("[phase=poll_inputs] fetched uh pack + GPU-phase JSONs")
        sys.exit(0)
    print("[phase=poll_inputs] not yet; sleeping 300s", flush=True)
    time.sleep(300)
raise SystemExit("[phase=poll_inputs] TIMEOUT: GPU-phase outputs never landed on HF (12h)")
PY
fi

echo "[phase=readout] R-x: all-46 rows x {sycophancy, refusal}, FULL enlarged-axis null rerun"
# NO --null-join on this leg (A5: the read-out nulls draw from ONE shared rng
# consumed in loop order, so a join cannot byte-match — plan v11 §6 primary
# path). Old 34 position rows come from the ROUND-1 store (the committed
# comparator, the default --position-store-hf); the 9 new rows from the pack.
maybe uv run python scripts/issue810_fit_readout.py \
  --rows all-46 --behaviors sycophancy refusal \
  --null-mode full-rerun --uh-summaries "$UH_PACK" \
  --out "$OUT" --out-suffix user_header \
  --upload-prefix "$HF_MIRROR/eval_results"

echo "[phase=bands_check] registered band rows present (union + D_uh + H3 + conjunction)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] verify band rows in $OUT/{null_matrix_user_header,crosslayer_xbnd,readout_rho_user_header}.json"
else
  uv run python - <<PY
import json, pathlib
out = pathlib.Path("$OUT")
ub = json.loads((out / "null_matrix_user_header.json").read_text()).get("union_band") or {}
rows = ub.get("band_rows") or {}
assert "enlarged_axis_max_selected" in rows and "D_uh_difference" in rows, rows.keys()
cx = json.loads((out / "crosslayer_xbnd.json").read_text())
assert "band_row_h3" in cx, sorted(cx.keys())
ro = json.loads((out / "readout_rho_user_header.json").read_text())
enl = ro.get("enlarged_axis") or {}
assert "max_selected" in enl and "conjunction" in enl, sorted(enl.keys())
print("[phase=bands_check] OK:",
      "recon", rows["enlarged_axis_max_selected"]["verdict"], "|",
      "D_uh", rows["D_uh_difference"]["verdict"], "|",
      "H3", cx["band_row_h3"]["verdict"], "|",
      "readout", enl["max_selected"]["verdict"])
PY
fi

echo "[phase=bootstrap] paired per-context Δskill(new row − mean), 2000 draws seed 42"
maybe uv run python scripts/issue810_bootstrap_deltaskill.py --vs mean --n-draws 2000 \
  --uh-summaries "$UH_PACK" --out "$OUT/delta_vs_mean.json"

echo "[phase=figures] E-x: aggregation + figure scaffolding (tolerated-failure: JSONs already durable)"
# issue810_analyze reads the parent-canonical filenames from --in; stage this
# round's suffixed JSONs under those names in a NON-uploaded, gitignored
# staging dir (never $OUT — the canonical names there belong to the parent,
# and duplicates would bloat the mirror upload).
ANALYZE_IN=data/issue_810/analyze_in_uh
maybe mkdir -p "$ANALYZE_IN"
maybe cp "$OUT/reconstruction_skill_user_header.json" "$ANALYZE_IN/reconstruction_skill_by_summary.json"
maybe cp "$OUT/null_matrix_user_header.json" "$ANALYZE_IN/null_matrix_reconstruction.json"
maybe cp "$OUT/readout_rho_user_header.json" "$ANALYZE_IN/readout_rho_by_summary.json"
maybe cp "$OUT/null_matrix_readout_user_header.json" "$ANALYZE_IN/null_matrix_readout.json"
FIG_RC=0
maybe uv run python scripts/issue810_analyze.py \
  --in "$ANALYZE_IN" --out "$OUT/analysis" --fig-dir "$FIGS" || FIG_RC=$?
if [ "$FIG_RC" -ne 0 ]; then
  echo "[phase=figures] analyze FAILED rc=$FIG_RC (tolerated LOUDLY: every stats JSON is" \
    "already written + uploaded; figures are re-runnable on the VM from the mirror)"
fi

echo "[phase=upload] mirror the CPU-phase outputs"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] upload_folder $OUT -> $HF_MIRROR/eval_results ; $FIGS -> $HF_MIRROR/figures"
else
  uv run python - <<PY
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
import pathlib
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="$OUT",
                  path_in_repo="$HF_MIRROR/eval_results",
                  repo_id="superkaiba1/explore-persona-space-data",
                  repo_type="dataset",
                  commit_message="issue #810 user-header-newline-summary: CPU-phase outputs")
if any(pathlib.Path("$FIGS").glob("*")):
    api.upload_folder(folder_path="$FIGS",
                      path_in_repo="$HF_MIRROR/figures",
                      repo_id="superkaiba1/explore-persona-space-data",
                      repo_type="dataset",
                      commit_message="issue #810 user-header-newline-summary: figures")
print("[phase=upload_done]")
PY
fi

echo "[phase=git_commit] guarded commit + push of the small JSONs + figures (mirror is the durable fallback)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] git add <small JSONs + figures> && git commit && git push origin issue-810 (tolerated on failure)"
else
  GIT_RC=0
  {
    git add "$OUT/reconstruction_skill_user_header.json" \
            "$OUT/null_matrix_user_header.json" \
            "$OUT/crosslayer_xbnd.json" \
            "$OUT/null_matrix_crosslayer_xbnd.json" \
            "$OUT/readout_rho_user_header.json" \
            "$OUT/delta_vs_mean.json" &&
    { [ -d "$OUT/analysis" ] && git add "$OUT/analysis" || true; } &&
    { [ -d "$FIGS" ] && git add "$FIGS" || true; } &&
    git -c user.email="pod@eps" -c user.name="issue810-cpu-phase" \
      commit -m "issue #810 user-header-newline-summary: CPU-phase outputs (readout full-rerun, bands, bootstrap, figures)" &&
    git push origin issue-810
  } || GIT_RC=$?
  if [ "$GIT_RC" -ne 0 ]; then
    echo "[phase=git_commit] FAILED rc=$GIT_RC (tolerated LOUDLY: the HF mirror at" \
      "$HF_MIRROR/eval_results is the durable copy; the orchestrator commits from the VM)"
  fi
  # NOTE: null_matrix_readout_user_header.json (~250 MB per-draw matrix) is
  # deliberately NOT committed to git — its placement is the HF mirror (plan
  # v11 §10 discarded_artifacts note), matching the parent's 207 MB matrix.
fi

echo "[phase=sentinel] end-of-run results sentinel"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] write poll_pipeline sentinel with the four registered verdicts"
else
  uv run python - <<PY
import json, pathlib, time
out = pathlib.Path("$OUT")
def _load(name):
    p = out / name
    return json.loads(p.read_text()) if p.is_file() else {}
ub = _load("null_matrix_user_header.json").get("union_band") or {}
enl = _load("readout_rho_user_header.json").get("enlarged_axis") or {}
note = {
    "phase": "cpu_phase_uh",
    "recon_band_rows": ub.get("band_rows"),
    "readout_enlarged_axis": {
        "max_selected": (enl.get("max_selected") or {}).get("verdict"),
        "conjunction": {
            b: v.get("verdict")
            for b, v in ((enl.get("conjunction") or {}).get("by_behavior") or {}).items()
        },
    },
    "hf_mirror": "$HF_MIRROR/eval_results",
}
log_dir = pathlib.Path("/workspace/logs")
try:
    log_dir.mkdir(parents=True, exist_ok=True)
    target = log_dir / f"issue-810-epm_results-{int(time.time())}.json"
except OSError:
    target = out / "issue-810-epm_results-sentinel.json"
target.write_text(json.dumps({
    "sentinel_schema_version": 1, "kind": "epm:results", "version": 1,
    "note": note, "ts": int(time.time()),
}, indent=2))
print(f"[phase=sentinel] wrote {target}")
PY
fi

echo "[phase=done] issue810 user-header-newline-summary CPU phase complete"
