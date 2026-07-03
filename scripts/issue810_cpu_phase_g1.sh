#!/usr/bin/env bash
# Issue #810 follow-up `ultrachat-genre-summary-sweep` CPU-phase driver (plan v6
# §4.5/§10): submit C-g (bg) -> A-g free leg -> poll B-g store -> wait C-g ->
# D-g readout square -> square merge -> poll/fetch D-g-recon -> genre deltas ->
# contamination audit -> E-g analyze -> upload. Pure sequencing of the
# code-reviewed CLIs (no logic) on a cpu-mid instance (the parent's lane; the
# earlyoom precedent bars the shared VM for the judge working set).
#
# Differences vs the parent issue810_cpu_phase.sh, all plan-mandated:
# - NO cache_restore step: the parent's judge_cache tarball is POISONED
#   (cross-rubric contamination) and is NEVER restored on this path; the fresh
#   per-behavior cache root is data/issue_810/judge_cache_g1/.
# - C-g is submitted FIRST in the BACKGROUND (starts the <=24h Batch-API SLA
#   clock) while the free leg runs; the chain waits on it before the readout.
# - Assumption 6: parent artifacts live on branch issue-810 (NOT main), so a
#   main-only lane must fetch + check out the branch before running.
#
# Dry-run sequencing pass: I810_DRYRUN=1 bash scripts/issue810_cpu_phase_g1.sh
# echoes every phase in production order without executing anything.
set -euo pipefail

OUT=eval_results/issue_810/ultrachat-genre-summary-sweep
CACHE=data/issue_810/judge_cache_g1
FIGS=figures/issue_810/ultrachat-genre-summary-sweep
STORE_HF=issue658_theory_assumptions/answer_position_sweep_genre-generalization-ultrachat
HF_MIRROR=issue810_results/ultrachat-genre-summary-sweep
LOGDIR="${I810_LOGDIR:-logs/issue_810}"
DRY="${I810_DRYRUN:-0}"

maybe() { if [ "$DRY" = "1" ]; then echo "[dryrun] $*"; else "$@"; fi; }

echo "[phase=branch_sync]"
# Assumption 6: the parent's committed Phase-C E0 + these scripts live on branch
# issue-810 @ origin (VERIFIED at plan time); a lane that cloned main only must
# fetch + check out the branch. No-op when already on issue-810 (worktree/pod).
if [ "$(git rev-parse --abbrev-ref HEAD)" != "issue-810" ]; then
  maybe git fetch origin issue-810
  maybe git checkout issue-810
fi
maybe mkdir -p "$OUT" "$FIGS" "$LOGDIR"

echo "[phase=submit_c_g] Phase C-g graded re-judge (background; starts the batch SLA clock)"
C_PID=""
if [ "$DRY" = "1" ]; then
  echo "[dryrun] nohup uv run python scripts/issue810_batch_rejudge_highm.py --genre g1" \
    "--behaviors sycophancy refusal harmful_compliance --cache-dir $CACHE --out $OUT/phase_c"
else
  nohup uv run python scripts/issue810_batch_rejudge_highm.py --genre g1 \
    --behaviors sycophancy refusal harmful_compliance \
    --cache-dir "$CACHE" --out "$OUT/phase_c" \
    > "$LOGDIR/phase_c_g.log" 2>&1 < /dev/null &
  C_PID=$!
  echo "[phase=submit_c_g] pid=$C_PID log=$LOGDIR/phase_c_g.log"
fi

echo "[phase=free_leg] A-g: recon re-fit on the g1 {mean,last,maxp} store (misalignment tripwire)"
# --no-mlp: the §9 A-g row is ridge+nulls only (84 cells x ~9.8s); the MLP
# validity arm runs ONCE for ALL summaries in the GPU D-g-recon session.
maybe uv run python scripts/issue810_fit_reconstruction.py --genre g1 --free-only \
  --n-perms 1000 --no-mlp --out "$OUT/free_leg"

echo "[phase=poll_store] waiting for the B-g aligned-subset store on HF ($STORE_HF)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] poll HF for $STORE_HF/manifest.json (sleep 300, timeout 12h)"
else
  uv run python - <<PY
import sys, time
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import list_repo_files
target = "$STORE_HF/manifest.json"
deadline = time.time() + 12 * 3600
while time.time() < deadline:
    files = set(list_repo_files("superkaiba1/explore-persona-space-data", repo_type="dataset"))
    if target in files:
        n = sum(1 for f in files if f.startswith("$STORE_HF/") and f.endswith(".pt"))
        print(f"[phase=poll_store] store present ({n} context files)")
        sys.exit(0)
    print("[phase=poll_store] not yet; sleeping 300s", flush=True)
    time.sleep(300)
raise SystemExit("[phase=poll_store] TIMEOUT: B-g store never landed on HF (12h)")
PY
fi

echo "[phase=wait_c_g] waiting for the Phase C-g harvest"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] wait \$C_PID; route on artifact $OUT/phase_c/e0_highm_graded.json (rc secondary)"
else
  C_RC=0
  wait "$C_PID" || C_RC=$?
  # Route on the ARTIFACT, not the exit code (wrap-script rule): the graded-E0
  # JSON is the contract; rc is a fall-through crash signal only.
  if [ ! -s "$OUT/phase_c/e0_highm_graded.json" ]; then
    echo "[phase=wait_c_g] FAILED rc=$C_RC and no e0_highm_graded.json — see $LOGDIR/phase_c_g.log"
    exit 1
  fi
  if [ "$C_RC" -ne 0 ]; then
    echo "[phase=wait_c_g] rc=$C_RC tolerated: artifact present (route-on-artifact rule)"
  fi
fi

echo "[phase=readout_square] D-g: 3 new combos (summaries-genre x e0-genre)"
maybe uv run python scripts/issue810_fit_readout.py --summaries-genre g1 --e0-genre betley \
  --out "$OUT/readout_sg1_ebetley"
maybe uv run python scripts/issue810_fit_readout.py --summaries-genre g1 --e0-genre g1 \
  --e0-highm "$OUT/phase_c/e0_highm_graded.json" --out "$OUT/readout_sg1_eg1"
maybe uv run python scripts/issue810_fit_readout.py --summaries-genre betley --e0-genre g1 \
  --e0-highm "$OUT/phase_c/e0_highm_graded.json" --out "$OUT/readout_sbetley_eg1"

echo "[phase=square_merge] assemble readout_rho_square.json (3 new combos, tagged)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] merge $OUT/readout_*/readout_rho_by_summary.json -> $OUT/readout_rho_square.json"
else
  uv run python - <<PY
import json, pathlib
out = pathlib.Path("$OUT")
combos = ["readout_sg1_ebetley", "readout_sg1_eg1", "readout_sbetley_eg1"]
square = {"dv": "readout_rho_square", "combos": {}, "cells": []}
for combo in combos:
    blob = json.loads((out / combo / "readout_rho_by_summary.json").read_text())
    sg, eg = blob["summaries_genre"], blob["e0_genre"]
    square["combos"][combo] = {
        "summaries_genre": sg, "e0_genre": eg,
        "h2_conjunction": blob.get("h2_conjunction"),
        "judge_validation": blob.get("judge_validation"),
        "length_control": blob.get("length_control"),
    }
    for c in blob["cells"]:
        square["cells"].append({**c, "summaries_genre": sg, "e0_genre": eg})
square["note"] = ("the parent (betley acts, betley E0) cell is REUSED from "
                  "eval_results/issue_810/readout_rho_by_summary.json, never re-fit")
# E0-stability side-read (plan v6 §3): Spearman(E0_betley, E0_g1) per behavior
# across the shared contexts. harmful_compliance is a CONTAMINATION DIAGNOSTIC
# only (the parent target is quarantined), labeled as such.
from scipy.stats import spearmanr
eb = json.loads(pathlib.Path("eval_results/issue_810/phase_c/e0_highm_graded.json").read_text())
eg = json.loads((out / "phase_c" / "e0_highm_graded.json").read_text())
stability = {}
for b, blk in eg["by_behavior"].items():
    gb = eb["by_behavior"].get(b, {}).get("per_context_graded_mean", {})
    gg = blk["per_context_graded_mean"]
    shared = [c for c in gg if c in gb and gg[c] is not None and gb.get(c) is not None]
    rho = float(spearmanr([gb[c] for c in shared], [gg[c] for c in shared])[0]) \
        if len(shared) >= 4 else None
    stability[b] = {"n": len(shared), "rho": rho,
                    "role": ("contamination diagnostic ONLY (parent target quarantined)"
                             if b == "harmful_compliance" else "side-read")}
square["e0_stability"] = stability
(out / "readout_rho_square.json").write_text(json.dumps(square, indent=2))
rhos = {b: v["rho"] for b, v in stability.items()}
print(f"[phase=square_merge] {len(square['cells'])} cells over {len(combos)} combos; "
      f"e0_stability={rhos}")
PY
fi

echo "[phase=poll_recon] waiting for the GPU D-g-recon JSONs on HF ($HF_MIRROR/phase_d_recon)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] poll + fetch reconstruction_skill_by_summary.json + null_matrix_reconstruction.json -> $OUT/"
else
  uv run python - <<PY
import shutil, sys, time
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download, list_repo_files
prefix = "$HF_MIRROR/phase_d_recon"
names = ["reconstruction_skill_by_summary.json", "null_matrix_reconstruction.json"]
deadline = time.time() + 12 * 3600
while time.time() < deadline:
    files = set(list_repo_files("superkaiba1/explore-persona-space-data", repo_type="dataset"))
    if all(f"{prefix}/{n}" in files for n in names):
        for n in names:
            p = hf_hub_download("superkaiba1/explore-persona-space-data", f"{prefix}/{n}",
                                repo_type="dataset")
            shutil.copy(p, f"$OUT/{n}")
        print("[phase=poll_recon] fetched D-g-recon JSONs")
        sys.exit(0)
    print("[phase=poll_recon] not yet; sleeping 300s", flush=True)
    time.sleep(300)
raise SystemExit("[phase=poll_recon] TIMEOUT: D-g-recon JSONs never landed on HF (12h)")
PY
fi

echo "[phase=genre_delta] paired cross-genre Δskill bootstrap (H1-g(ii))"
maybe uv run python scripts/issue810_bootstrap_deltaskill.py --cross-genre \
  --out "$OUT/genre_delta_recon.json"

echo "[phase=contamination_audit] clean-data check on the g1 judge outputs"
maybe uv run python scripts/issue810_contamination_audit.py \
  --in "$OUT/phase_c" --out "$OUT/analysis/contamination_audit_g1.json"

echo "[phase=analyze] E-g: honest bands + H1-g(iii) ordering statistic + figures"
# hero3 reads the primary activation-side contrast (g1 acts -> parent E0) at the
# --in root; recon inputs were fetched there by poll_recon.
maybe cp "$OUT/readout_sg1_ebetley/readout_rho_by_summary.json" "$OUT/readout_rho_by_summary.json"
maybe cp "$OUT/readout_sg1_ebetley/null_matrix_readout.json" "$OUT/null_matrix_readout.json"
maybe uv run python scripts/issue810_analyze.py --genre g1 \
  --in "$OUT" --out "$OUT/analysis" --fig-dir "$FIGS"

echo "[phase=upload]"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] upload_folder $OUT -> $HF_MIRROR/eval_results ; $FIGS -> $HF_MIRROR/figures"
else
  uv run python - <<PY
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="$OUT",
                  path_in_repo="$HF_MIRROR/eval_results",
                  repo_id="superkaiba1/explore-persona-space-data",
                  repo_type="dataset",
                  commit_message="issue #810 ultrachat-genre-summary-sweep: CPU-phase outputs")
api.upload_folder(folder_path="$FIGS",
                  path_in_repo="$HF_MIRROR/figures",
                  repo_id="superkaiba1/explore-persona-space-data",
                  repo_type="dataset",
                  commit_message="issue #810 ultrachat-genre-summary-sweep: figures")
print("[phase=upload_done]")
PY
fi

echo "[phase=done] issue810 ultrachat-genre-summary-sweep CPU phase complete"
