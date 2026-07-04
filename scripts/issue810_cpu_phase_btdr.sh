#!/usr/bin/env bash
# Issue #810 follow-up `boundary-truncation-dose-response` CPU-phase driver
# (plan v18 §4.5/§10): poll HF for the 3 btdr packs + the GPU-phase recon
# JSONs -> paired dose bootstrap + both-side refit parity + familywise reads +
# mechanism-vs-k (issue810_btdr_dose.py) -> coverage check -> figures
# (tolerated) -> HF mirror -> guarded git commit -> sentinel. Pure sequencing
# of the code-reviewed CLIs (no logic) on a cpu-mid instance, mirroring
# issue810_cpu_phase_he.sh. Pod-side signaling: [phase=...] log lines + the
# end-of-run sentinel ONLY — NEVER a task.py shellout (branch-guard).
#
# Dry-run sequencing pass: I810_DRYRUN=1 bash scripts/issue810_cpu_phase_btdr.sh
set -euo pipefail

OUT=eval_results/issue_810/boundary-truncation-dose-response
FIGS=figures/issue_810/boundary-truncation-dose-response
PACK_DIR=data/issue_810
HF_MIRROR=issue810_results/boundary-truncation-dose-response
LOGDIR="${I810_LOGDIR:-logs/issue_810}"
DRY="${I810_DRYRUN:-0}"

maybe() { if [ "$DRY" = "1" ]; then echo "[dryrun] $*"; else "$@"; fi; }

echo "[phase=branch_sync]"
if [ "$(git rev-parse --abbrev-ref HEAD)" != "issue-810" ]; then
  maybe git fetch origin issue-810
  maybe git checkout issue-810
fi
maybe mkdir -p "$OUT" "$FIGS" "$LOGDIR" "$PACK_DIR"

echo "[phase=poll_inputs] waiting for the 3 btdr packs + GPU-phase fit JSONs (recon + null-matrix) on HF"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] poll HF for $HF_MIRROR/btdr_summaries_k{25,50,75}.pt + $HF_MIRROR/eval_results/{reconstruction_skill,null_matrix}_btdr_k{25,50,75}.json (sleep 300, timeout 12h)"
else
  uv run python - <<PY
import shutil, sys, time
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download, list_repo_files
packs = [f"$HF_MIRROR/btdr_summaries_k{p}.pt" for p in (25, 50, 75)]
# Fetch EVERYTHING the git_commit phase adds (the `_he` fetch<->add contract):
# per-k recon fits + per-k null matrices (n_perms=0 -> empty per-draw dict, but
# still a produced + uploaded deliverable the guarded commit references).
jsons = [
    f"{stem}_btdr_k{p}.json"
    for stem in ("reconstruction_skill", "null_matrix")
    for p in (25, 50, 75)
]
deadline = time.time() + 12 * 3600
while time.time() < deadline:
    files = set(list_repo_files("superkaiba1/explore-persona-space-data", repo_type="dataset"))
    if all(p in files for p in packs) and all(
        f"$HF_MIRROR/eval_results/{n}" in files for n in jsons
    ):
        for p in packs:
            q = hf_hub_download("superkaiba1/explore-persona-space-data", p, repo_type="dataset")
            shutil.copy(q, "$PACK_DIR/" + p.rsplit("/", 1)[1])
        for n in jsons:
            q = hf_hub_download("superkaiba1/explore-persona-space-data",
                                f"$HF_MIRROR/eval_results/{n}", repo_type="dataset")
            shutil.copy(q, f"$OUT/{n}")
        print("[phase=poll_inputs] fetched 3 btdr packs + 6 GPU-phase fit JSONs")
        sys.exit(0)
    print("[phase=poll_inputs] not yet; sleeping 300s", flush=True)
    time.sleep(300)
raise SystemExit("[phase=poll_inputs] TIMEOUT: GPU-phase outputs never landed on HF (12h)")
PY
fi

echo "[phase=dose] paired dose bootstrap + both-side refit parity + familywise + mechanism-vs-k"
# All 9 pairs x 3 interior k, 2,000 shared-index draws seed 42 (defaults); the
# full side of the 7 uh/bnd rows comes from the round-3 uh pack (HF default),
# im_end/turn_nl from the ROUND-1 position store per-file (~350 MB,
# cpu-mid-safe — the round-4 convention); the k=0 side from the round-4 he
# pack (HF default, refit-parity-asserted vs the committed round-4 skills).
maybe uv run python scripts/issue810_btdr_dose.py \
  --btdr-summaries "$PACK_DIR/btdr_summaries_k25.pt" \
                   "$PACK_DIR/btdr_summaries_k50.pt" \
                   "$PACK_DIR/btdr_summaries_k75.pt" \
  --out "$OUT"

echo "[phase=coverage_check] 27-cell paired table + familywise reads + mechanism present (plan v18 §6.5)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] verify 27 cells + familywise in $OUT/paired_dose_response.json + 3-k mechanism in $OUT/mechanism_cosine_btdr.json"
else
  uv run python - <<PY
import json, pathlib
out = pathlib.Path("$OUT")
paired = json.loads((out / "paired_dose_response.json").read_text())
by_k = paired.get("by_k") or {}
assert sorted(by_k.keys()) == ["k25", "k50", "k75"], sorted(by_k.keys())
n_cells = sum(len(v.get("per_row") or {}) for v in by_k.values())
assert n_cells == 27, n_cells
assert all(
    "verdict" in c and "ci95" in c
    for v in by_k.values()
    for c in (v.get("per_row") or {}).values()
)
assert (paired.get("familywise_27cell") or {}).get("n_cells") == 27
assert (paired.get("familywise_6cell_primary") or {}).get("n_cells") == 6
assert len(paired.get("k0_committed") or {}) >= 9  # 9 rows + round-4 familywise echo
assert len(paired.get("empty_side_refit_parity") or {}) == 9
mech = json.loads((out / "mechanism_cosine_btdr.json").read_text())
assert sorted((mech.get("by_k") or {}).keys()) == ["k25", "k50", "k75"]
assert all(len(v) == 9 for v in mech["by_k"].values())
print("[phase=coverage_check] OK:",
      "fam27 p =", paired["familywise_27cell"]["p_ge1_abs_centered_delta_ge_margin"], "|",
      "fam6 p =", paired["familywise_6cell_primary"]["p_ge1_abs_centered_delta_ge_margin"], "|",
      "verdicts:", {f"{r}@{k}": v["verdict"]
                    for k, blk in sorted(by_k.items())
                    for r, v in sorted(blk["per_row"].items())
                    if r in ("turn_nl", "uh_im_start")})
PY
fi

echo "[phase=figures] dose-trace + companions (tolerated-failure: JSONs already durable)"
FIG_RC=0
maybe uv run python scripts/issue810_btdr_figures.py || FIG_RC=$?
if [ "$FIG_RC" -ne 0 ]; then
  echo "[phase=figures] figures FAILED rc=$FIG_RC (tolerated LOUDLY: every stats JSON is" \
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
                  commit_message="issue #810 boundary-truncation-dose-response: CPU-phase outputs")
if any(pathlib.Path("$FIGS").glob("*")):
    api.upload_folder(folder_path="$FIGS",
                      path_in_repo="$HF_MIRROR/figures",
                      repo_id="superkaiba1/explore-persona-space-data",
                      repo_type="dataset",
                      commit_message="issue #810 boundary-truncation-dose-response: figures")
print("[phase=upload_done]")
PY
fi

echo "[phase=git_commit] guarded commit + push of the small JSONs + figures (mirror is the durable fallback)"
if [ "$DRY" = "1" ]; then
  echo "[dryrun] git add <small JSONs + figures> && git commit && git push origin issue-810 (tolerated on failure)"
else
  GIT_RC=0
  {
    git add "$OUT/reconstruction_skill_btdr_k25.json" \
            "$OUT/reconstruction_skill_btdr_k50.json" \
            "$OUT/reconstruction_skill_btdr_k75.json" \
            "$OUT/null_matrix_btdr_k25.json" \
            "$OUT/null_matrix_btdr_k50.json" \
            "$OUT/null_matrix_btdr_k75.json" \
            "$OUT/paired_dose_response.json" \
            "$OUT/mechanism_cosine_btdr.json" &&
    { [ -d "$FIGS" ] && git add "$FIGS" || true; } &&
    git -c user.email="pod@eps" -c user.name="issue810-cpu-phase" \
      commit -m "issue #810 boundary-truncation-dose-response: CPU-phase outputs (paired dose bootstrap, familywise reads, mechanism-vs-k, figures)" &&
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
paired = _load("paired_dose_response.json")
by_k = paired.get("by_k") or {}
note = {
    "phase": "cpu_phase_btdr",
    "paired_verdicts": {
        k: {r: v.get("verdict") for r, v in (blk.get("per_row") or {}).items()}
        for k, blk in by_k.items()
    },
    "familywise_27cell": paired.get("familywise_27cell"),
    "familywise_6cell_primary": paired.get("familywise_6cell_primary"),
    "empty_side_refit_parity_max_absdiff": max(
        (v.get("abs_diff", 0.0) for v in (paired.get("empty_side_refit_parity") or {}).values()
         if isinstance(v, dict) and "abs_diff" in v),
        default=None,
    ),
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

echo "[phase=done] issue810 boundary-truncation-dose-response CPU phase complete"
