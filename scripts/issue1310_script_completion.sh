#!/usr/bin/env bash
# Issue #1310 follow-up `script-instruct-completion`: the two never-fit
# SCRIPT-FORMAT instruct cells — Vex (per-turn map) + the swap/swapctrl control —
# at RECIPE PARITY to the committed run-2 instruct cells (commit 942df1bb).
#
# Phases (1 GPU; run-2 store is lost, scenes are persisted on HF):
#   p0. stage the persisted run-2 scenes (local-first -> scoped HF fetch)
#   p1. attribution + per-turn pair build (issue1310_attribute.py, byte-identical
#       to 942df1bb) + a fail-loud PARITY GATE vs the committed run-2 audit
#   p2. per-turn teacher-forced 28-layer capture (--flavor perturn = the run-2
#       path: own chat-template prefix, spans shifted by len(prefix_ids))
#   p3. fits: Vex within-map + lastpos + pooled swap/swapctrl (full store —
#       the derangement needs every persona's turns), --gcv-dof-cap 0.9
#   p5. uploads (text/JSON unconditional; store tensors batched) -> git commit
#   p6. results sentinel -> [phase=done]
#
# SMOKE=1 runs the SAME script end-to-end on the VM: 20-scene slice (5 scenarios
# x 4 personas), tiny-real same-arch model over the real vocab, mock judge,
# network writes gated off (enumeration still executes), /tmp outputs.
#
# GCE has NO .env (tokens are in the exported env); RunPod DOES -> source
# conditionally, never unconditionally inside a classified &&-chain (#923).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export PYTHONUNBUFFERED=1

ISSUE=1310
SMOKE="${SMOKE:-0}"
export SMOKE
DATA_DIR="${DATA_DIR:-data/issue_1310}"
OUT_DIR="${OUT_DIR:-eval_results/issue_1310/script_completion}"
STORE_SUBDIR="${STORE_SUBDIR:-store_script_completion}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
HF_PREFIX="issue1310_char_map"
HF_REPO="superkaiba1/explore-persona-space-data"
RUN_START_EPOCH="${RUN_START_EPOCH:-$(date +%s)}"
export RUN_START_EPOCH DATA_DIR OUT_DIR STORE_SUBDIR HF_PREFIX HF_REPO

if [ "$SMOKE" = "1" ]; then
  SMOKE_ROOT="${SMOKE_ROOT:-/tmp/issue-1310-smoke}"
  DATA_DIR="$SMOKE_ROOT/data"
  OUT_DIR="$SMOKE_ROOT/eval"
  LOG_DIR="$SMOKE_ROOT/logs"
  export DATA_DIR OUT_DIR
  TINY_DIR="$SMOKE_ROOT/tiny_model"
  ATTR_ARGS="--mock-judge --audit-n 8"
  EXTRACT_ARGS="--tiny-model-dir $TINY_DIR --batch-size 3"
  FIT_ARGS="--smoke --null-draws 3 --n-boot 20"
else
  ATTR_ARGS="--audit-n 200"
  EXTRACT_ARGS="--batch-size ${BATCH_SIZE:-8} --resume"
  FIT_ARGS=""
fi
mkdir -p "$DATA_DIR" "$OUT_DIR" "$LOG_DIR"

if [ "$SMOKE" = "1" ] && [ ! -d "$TINY_DIR" ]; then
  echo "[phase=p0_tiny] writing tiny-real same-arch smoke model (real vocab)"
  uv run python scripts/issue1310_extract_store.py --make-tiny-model "$TINY_DIR"
fi

echo "[phase=p0_scenes] staging persisted run-2 instruct scenes (SMOKE=$SMOKE)"
uv run python - "$DATA_DIR" <<'PY'
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# CONTENT-IDENTITY PIN (#600-class): the Hub path TIP was OVERWRITTEN by the v3
# run's regenerated scenes (commit a5e95e7557, 2026-07-15T05:29:22Z) — a
# different vLLM corpus whose per-persona pair counts diverge from the run-2
# committed audit (Vex 3471 vs 3586). The run-2 input is the earlier revision
# below; full attribution on it reproduces the committed audit counters EXACTLY.
SCENES_REVISION = "f84b6a3082139a11e39d639f1c1797bd286a6e13"
SCENES_SHA256 = "441f9cc3f0ec46a258a456e2f422ea1593e0dfdd780c9e759a2a795122d2ffc5"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


data_dir = Path(sys.argv[1])
dst = data_dir / "stories" / "instruct_stories_seed42.jsonl"
full = dst.parent / "instruct_stories_seed42.full.jsonl"
full.parent.mkdir(parents=True, exist_ok=True)
if not full.exists() and dst.exists() and _sha(dst) == SCENES_SHA256:
    shutil.copyfile(dst, full)  # local-first seed (pre-staged, content-verified)
if not full.exists() or _sha(full) != SCENES_SHA256:
    if full.exists():
        print("[i1310-sc] WARNING: staged scenes sha mismatch — refetching pinned revision")
    from huggingface_hub import hf_hub_download

    last: Exception | None = None
    for attempt in range(4):  # bounded transient-retry (#1345), then fail loud
        try:
            p = hf_hub_download(
                repo_id="superkaiba1/explore-persona-space-data",
                repo_type="dataset",
                revision=SCENES_REVISION,
                filename=(
                    "issue1310_char_map/raw_completions/generation/"
                    "instruct_stories_seed42.jsonl"
                ),
            )
            break
        except Exception as e:  # noqa: BLE001 — re-raised below after budget
            last = e
            time.sleep(15 * (attempt + 1))
    else:
        raise RuntimeError(f"scenes fetch failed after 4 attempts: {last!r}")
    shutil.copyfile(p, full)
assert _sha(full) == SCENES_SHA256, f"pinned-revision scenes sha mismatch at {full}"

rows = [json.loads(line) for line in full.read_text().splitlines() if line.strip()]
assert len(rows) == 1200, f"expected 1200 persisted scenes, got {len(rows)}"
if os.environ.get("SMOKE") == "1":
    # 5 scenarios x all 4 personas = 20 scenes -> every persona present at
    # matched scenario positions, so the swap derangement is exercised.
    scen = sorted({r["scenario_id"] for r in rows})[:5]
    rows = [r for r in rows if r["scenario_id"] in scen]
    assert len(rows) == 20, f"smoke slice expected 20 scenes, got {len(rows)}"
dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(
    f"[i1310-sc] scenes staged: {dst} rows={len(rows)} "
    f"(run-2 revision {SCENES_REVISION[:12]}, sha {SCENES_SHA256[:16]})"
)
PY

# p1: attribution + per-turn pair build (script byte-identical to 942df1bb).
uv run python scripts/issue1310_attribute.py --model instruct \
  --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" $ATTR_ARGS

if [ "$SMOKE" != "1" ]; then
  echo "[phase=p1_parity] attribution parity gate vs committed run-2 audit"
  uv run python - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

new = json.loads((Path(sys.argv[1]) / "attribution_audit_instruct.json").read_text())
ref = json.loads(Path("eval_results/issue_1310/attribution_audit_instruct.json").read_text())
for k in ("counters", "per_persona_pairs", "drop_rate"):
    assert new[k] == ref[k], f"attribution parity FAIL on {k!r}: run-2={ref[k]} now={new[k]}"
print(
    "[i1310-parity] attribution parity vs run-2 (942df1bb) PASS: "
    f"turns_kept={new['counters']['turns_kept']} per_persona={new['per_persona_pairs']}"
)
PY
fi

# Phase-boundary upload of the eval JSONs written so far (audit); SMOKE dry-runs
# the SAME code (enumeration + imports execute; only the network write is gated).
upload_eval_jsons() {
  uv run python - "$OUT_DIR" <<'PY'
import os
import sys
from pathlib import Path

from explore_persona_space.orchestrate import hub

out = Path(sys.argv[1])
files = sorted(str(p) for p in out.rglob("*.json"))
assert files, f"no eval JSONs under {out} — refusing to 'upload' nothing"
if os.environ.get("SMOKE") == "1":
    print(f"[i1310-up] SMOKE dry-run: would upload {len(files)} JSONs -> "
          f"{os.environ['HF_PREFIX']}/eval_results_script_completion")
else:
    hub._upload(
        out,
        repo_id=os.environ["HF_REPO"],
        repo_type="dataset",
        path_in_repo=f"{os.environ['HF_PREFIX']}/eval_results_script_completion",
    )
    print(f"[i1310-up] uploaded {len(files)} eval JSONs")
PY
}
upload_eval_jsons

# p2: per-turn capture (the run-2 path — --flavor perturn keeps the model's own
# chat-template prefix and shifts every turn's story-local spans by its length).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python \
  scripts/issue1310_extract_store.py --model instruct --flavor perturn \
  --data-dir "$DATA_DIR" --store-subdir "$STORE_SUBDIR" --equivalence-check $EXTRACT_ARGS

echo "[phase=p2_gate] extraction row-count gate (shards == kept turn-pairs)"
uv run python - "$DATA_DIR" "$OUT_DIR" "$STORE_SUBDIR" <<'PY'
import json
import sys
from pathlib import Path

data_dir, out_dir, subdir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sides = sorted((data_dir / subdir / "instruct").glob("instruct_shard*.json"))
assert sides, "no shard sidecars written"
total = sum(json.loads(s.read_text())["n_rows"] for s in sides)
kept = json.loads((out_dir / "attribution_audit_instruct.json").read_text())["counters"][
    "turns_kept"
]
assert total == kept, f"extracted rows {total} != kept turn-pairs {kept}"
print(f"[i1310-p2] extraction gate PASS: {total} rows across {len(sides)} shards == kept {kept}")
PY

# p3: fits — Vex within-map (+lastpos) and the pooled swap/swapctrl control over
# the FULL instruct store (run_swap ignores --personas by design). Machinery is
# the run-2 fit battery; --gcv-dof-cap 0.9 is inert on healthy interior GCV
# selections (protective on n<p degenerate layers, #1335).
uv run python scripts/issue1310_fit.py --models instruct --personas Vex \
  --data-dir "$DATA_DIR" --store-subdir "$STORE_SUBDIR" --tag "scriptc_" \
  --out-dir "$OUT_DIR" --gcv-dof-cap 0.9 $FIT_ARGS

echo "[phase=p3_gate] required fit artifacts exist"
for f in cells_scriptc_instruct_Vex.json cells_scriptc_instruct_Vex_lastpos.json \
  cells_scriptc_instruct_swap.json cells_scriptc_instruct_swapctrl_correct.json \
  swap_scriptc_instruct.json nulls_scriptc_instruct_Vex.json summary.json; do
  test -s "$OUT_DIR/$f" || { echo "[i1310] FATAL: missing fit artifact $OUT_DIR/$f" >&2; exit 86; }
done
echo "[i1310-p3] all required fit artifacts present"

echo "[phase=p5_upload] uploads (pairs + eval JSONs text path; store tensors batched)"
uv run python - "$DATA_DIR" "$STORE_SUBDIR" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

data_dir, subdir = Path(sys.argv[1]), sys.argv[2]
repo, prefix = os.environ["HF_REPO"], os.environ["HF_PREFIX"]
smoke = os.environ.get("SMOKE") == "1"
jobs = [
    (data_dir / "pairs", f"{prefix}/raw_completions/pairs_script_completion"),
    (data_dir / subdir / "instruct", f"{prefix}/analysis_tensors/store_script_completion/instruct"),
]
for local, remote in jobs:
    files = sorted(p for p in local.rglob("*") if p.is_file())
    assert files, f"nothing to upload under {local}"
    if smoke:
        print(f"[i1310-up] SMOKE dry-run: would upload {len(files)} files {local} -> {remote}")
        continue
    hub._upload(local, repo_id=repo, repo_type="dataset", path_in_repo=remote)
    on_hub = hub.list_hf_files_under_path(HfApi(), repo, remote, repo_type="dataset")
    assert len(on_hub) >= len(files), (
        f"scoped verify FAIL: {remote} has {len(on_hub)} files on Hub < {len(files)} local"
    )
    print(f"[i1310-up] {local} -> {remote} ({len(files)} files, verified {len(on_hub)} on Hub)")
PY
upload_eval_jsons

if [ "$SMOKE" != "1" ]; then
  echo "[phase=p5_git] committing eval JSONs to the issue branch"
  git add "eval_results/issue_${ISSUE}/script_completion"
  if git commit -m "task #${ISSUE}: script-format instruct completion cells (Vex + swap control, pod run)" >/dev/null 2>&1; then
    # #1205: verify the push landed (never swallow); retry once, then fail loud.
    git push origin "issue-${ISSUE}"
    behind="$(git rev-list --count "origin/issue-${ISSUE}..HEAD")"
    if [ "$behind" != "0" ]; then
      git push origin "issue-${ISSUE}"
      behind="$(git rev-list --count "origin/issue-${ISSUE}..HEAD")"
      [ "$behind" = "0" ] || { echo "[i1310] FATAL: results push did not land ($behind ahead)" >&2; exit 87; }
    fi
    echo "[i1310] results push verified (0 ahead of origin/issue-${ISSUE})"
  else
    echo "[i1310] nothing to commit"
  fi
fi

echo "[phase=p6_sentinel] writing results sentinel"
uv run python - "$OUT_DIR" "$LOG_DIR" <<'PY'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

out, log_dir = Path(sys.argv[1]), Path(sys.argv[2])
run_start = int(os.environ.get("RUN_START_EPOCH", "0"))
gpu_hours = round((time.time() - run_start) / 3600.0, 3) if run_start else None
summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}

pp = summary.get("per_persona", {})
r2_headline = {
    persona: {m: (entry.get("r2_headline") if entry else None) for m, entry in models.items()}
    for persona, models in pp.items()
}
swap = {
    m: (v or {}).get("delta_r2_char") for m, v in (summary.get("swap_specificity") or {}).items()
}
eval_numbers = {
    "per_persona_r2_headline": r2_headline,
    "swap_delta_r2_char": swap,
    "attribution": {
        m: (a or {}).get("attribution_precision")
        for m, a in (summary.get("attribution") or {}).items()
    },
}
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
note = {
    "eval_numbers": eval_numbers,
    "eval_paths": sorted(str(p) for p in out.glob("*.json")),
    "reproducibility_card": {
        "wandb_url": "n/a - no training in this follow-up (no WandB runs)",
        "hf_hub_url": (
            "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
            "tree/main/issue1310_char_map"
        ),
        "worktree_path": ".claude/worktrees/issue-1310",
        "final_commit_sha": commit,
        "gpu_hours_used": gpu_hours,
        "gpu_hours_budgeted": 2,
        "plan_deviations": [],
    },
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 1310,
    "by": "issue1310_script_completion",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
path = log_dir / "issue-1310-results.json"
path.write_text(json.dumps(sentinel, indent=2))
print(f"[i1310-p6] sentinel written: {path}")
PY

echo "[phase=done]"
