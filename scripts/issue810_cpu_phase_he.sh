#!/usr/bin/env bash
# Issue #810 follow-up `header-echo-ablation-capture` CPU-phase driver (plan v15
# §4.5/§10): poll HF for he_summaries.pt + the GPU-phase recon JSONs -> mechanism
# read + full-side refit parity + paired shared-index bootstrap
# (issue810_he_mechanism.py) -> union-bands/ceilings presence check -> figures
# (tolerated) -> HF mirror -> guarded git commit -> sentinel. Pure sequencing of
# the code-reviewed CLIs (no logic) on a cpu-mid instance, mirroring
# issue810_cpu_phase_uh.sh. Pod-side signaling: [phase=...] log lines + the
# end-of-run sentinel ONLY — NEVER a task.py shellout (branch-guard).
#
# Dry-run sequencing pass: I810_DRYRUN=1 bash scripts/issue810_cpu_phase_he.sh
set -euo pipefail

OUT=eval_results/issue_810/header-echo-ablation-capture
FIGS=figures/issue_810/header-echo-ablation-capture
HE_PACK=data/issue_810/he_summaries.pt
HF_MIRROR=issue810_results/header-echo-ablation-capture
LOGDIR="${I810_LOGDIR:-logs/issue_810}"
DRY="${I810_DRYRUN:-0}"

maybe() { if [ "$DRY" = "1" ]; then echo "[dryrun] $*"; else "$@"; fi; }

echo "[phase=branch_sync]"
if [ "$(git rev-parse --abbrev-ref HEAD)" != "issue-810" ]; then
  maybe git fetch origin issue-810
  maybe git checkout issue-810
fi
maybe mkdir -p "$OUT" "$FIGS" "$LOGDIR" "$(dirname "$HE_PACK")"

echo "[phase=poll_inputs] waiting for he_summaries.pt + GPU-phase JSONs on HF"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] poll HF for $HF_MIRROR/he_summaries.pt + $HF_MIRROR/eval_results/{recon,null} JSONs (sleep 300, timeout 12h)"
else
  uv run python - <<PY
import shutil, sys, time
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download, list_repo_files
pack = "$HF_MIRROR/he_summaries.pt"
jsons = ["reconstruction_skill_header_echo.json", "null_matrix_header_echo.json"]
deadline = time.time() + 12 * 3600
while time.time() < deadline:
    files = set(list_repo_files("superkaiba1/explore-persona-space-data", repo_type="dataset"))
    if pack in files and all(f"$HF_MIRROR/eval_results/{n}" in files for n in jsons):
        p = hf_hub_download("superkaiba1/explore-persona-space-data", pack, repo_type="dataset")
        shutil.copy(p, "$HE_PACK")
        for n in jsons:
            q = hf_hub_download("superkaiba1/explore-persona-space-data",
                                f"$HF_MIRROR/eval_results/{n}", repo_type="dataset")
            shutil.copy(q, f"$OUT/{n}")
        print("[phase=poll_inputs] fetched he pack + GPU-phase JSONs")
        sys.exit(0)
    print("[phase=poll_inputs] not yet; sleeping 300s", flush=True)
    time.sleep(300)
raise SystemExit("[phase=poll_inputs] TIMEOUT: GPU-phase outputs never landed on HF (12h)")
PY
fi

echo "[phase=mechanism] H2-he mechanism read + full-side refit parity + H1-he paired bootstrap"
# All 9 pairs, 2,000 shared-index draws seed 42 (defaults); the full side of the
# 7 uh/bnd rows comes from the round-3 uh pack (HF default), im_end/turn_nl from
# the ROUND-1 position store per-file (~350 MB, cpu-mid-safe — the committed
# refit-parity target's own data; see issue810_he_mechanism.py's deviation note).
maybe uv run python scripts/issue810_he_mechanism.py \
  --he-summaries "$HE_PACK" --out "$OUT"

echo "[phase=bands_check] union bands + ceilings + paired verdicts present (plan v15 §6)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] verify band rows in $OUT/null_matrix_header_echo.json + verdicts in $OUT/paired_full_minus_empty.json"
else
  uv run python - <<PY
import json, pathlib
out = pathlib.Path("$OUT")
ub = json.loads((out / "null_matrix_header_echo.json").read_text()).get("union_band") or {}
rows = ub.get("band_rows") or {}
assert "enlarged_axis_max_selected" in rows and "D_he_difference" in rows, sorted(rows.keys())
assert rows["enlarged_axis_max_selected"].get("ceiling") is not None, "LOCO ceiling missing"
assert ub.get("n_union_cells") == 55 * 28, ub.get("n_union_cells")
paired = json.loads((out / "paired_full_minus_empty.json").read_text())
per_row = paired.get("per_row") or {}
assert len(per_row) == 9, sorted(per_row.keys())
assert all("verdict" in v and "ci95" in v for v in per_row.values())
assert "p_ge1_abs_centered_delta_ge_margin" in (paired.get("familywise") or {})
mech = json.loads((out / "mechanism_cosine_r2.json").read_text())
assert len(mech.get("by_row") or {}) == 9
print("[phase=bands_check] OK:",
      "union", rows["enlarged_axis_max_selected"]["verdict"], "|",
      "D_he", rows["D_he_difference"]["verdict"], "|",
      "paired", {r: v["verdict"] for r, v in sorted(per_row.items())})
PY
fi

echo "[phase=figures] aggregation + figure scaffolding (tolerated-failure: JSONs already durable)"
# issue810_analyze reads the parent-canonical filenames from --in; stage this
# round's suffixed recon JSONs under those names in a NON-uploaded, gitignored
# staging dir (never $OUT). The readout leg is absent this round (plan v15 §4
# divergence 5) — analyze skips missing legs gracefully.
ANALYZE_IN=data/issue_810/analyze_in_he
maybe mkdir -p "$ANALYZE_IN"
maybe cp "$OUT/reconstruction_skill_header_echo.json" "$ANALYZE_IN/reconstruction_skill_by_summary.json"
maybe cp "$OUT/null_matrix_header_echo.json" "$ANALYZE_IN/null_matrix_reconstruction.json"
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
                  commit_message="issue #810 header-echo-ablation-capture: CPU-phase outputs")
if any(pathlib.Path("$FIGS").glob("*")):
    api.upload_folder(folder_path="$FIGS",
                      path_in_repo="$HF_MIRROR/figures",
                      repo_id="superkaiba1/explore-persona-space-data",
                      repo_type="dataset",
                      commit_message="issue #810 header-echo-ablation-capture: figures")
print("[phase=upload_done]")
PY
fi

echo "[phase=git_commit] guarded commit + push of the small JSONs + figures (mirror is the durable fallback)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] git add <small JSONs + figures> && git commit && git push origin issue-810 (tolerated on failure)"
else
  GIT_RC=0
  {
    git add "$OUT/reconstruction_skill_header_echo.json" \
            "$OUT/null_matrix_header_echo.json" \
            "$OUT/mechanism_cosine_r2.json" \
            "$OUT/paired_full_minus_empty.json" &&
    { [ -d "$OUT/analysis" ] && git add "$OUT/analysis" || true; } &&
    { [ -d "$FIGS" ] && git add "$FIGS" || true; } &&
    git -c user.email="pod@eps" -c user.name="issue810-cpu-phase" \
      commit -m "issue #810 header-echo-ablation-capture: CPU-phase outputs (mechanism read, paired bootstrap, union bands, figures)" &&
    git push origin issue-810
  } || GIT_RC=$?
  if [ "$GIT_RC" -ne 0 ]; then
    echo "[phase=git_commit] FAILED rc=$GIT_RC (tolerated LOUDLY: the HF mirror at" \
      "$HF_MIRROR/eval_results is the durable copy; the orchestrator commits from the VM)"
  fi
fi

echo "[phase=sentinel] end-of-run results sentinel"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] write poll_pipeline sentinel with the registered verdicts"
else
  uv run python - <<PY
import json, pathlib, time
out = pathlib.Path("$OUT")
def _load(name):
    p = out / name
    return json.loads(p.read_text()) if p.is_file() else {}
ub = _load("null_matrix_header_echo.json").get("union_band") or {}
paired = _load("paired_full_minus_empty.json")
note = {
    "phase": "cpu_phase_he",
    "recon_band_rows": ub.get("band_rows"),
    "paired_verdicts": {
        r: v.get("verdict") for r, v in (paired.get("per_row") or {}).items()
    },
    "familywise": paired.get("familywise"),
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

echo "[phase=done] issue810 header-echo-ablation-capture CPU phase complete"
