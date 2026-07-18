#!/usr/bin/env bash
# Issue #1426 sampled-rollout robustness round — pod-side dispatch (amendment
# plan v4 §4.2 item 5 + §9): sibling of the cap16k stage in
# issue1426_dispatch.sh, per seed s ∈ {42, 137}:
#   invocation A (extract at the FORCED sampled rung, per-seed --gen-seed +
#   --hf-prefix-root sampled_rollout/seed<s>) → store-manifest provenance
#   validation (critic addition A — FAIL LOUD before any fit) → invocation B
#   (restricted f1 at the primary's frozen layers, --no-mlp, --no-upload,
#   full-size stat batteries) → dispatch-side scoped fit upload →
#   frozen-layer diagnostic (report-not-stop).
#
# Width-adaptive (plan §4.2 item 5): at width 2 (nvidia-smi -L) the two seeds
# run CONCURRENTLY, one per GPU, per-process CUDA_VISIBLE_DEVICES env pin
# (never +gpu_id); per-seed logs sed-prefixed [s42]/[s137] into main stdout;
# at width 1 the seeds run sequentially. Either way an EARLY per-request-seed
# differentiation check (critic addition B) compares the two seeds' Gate-1
# slices and kills the pipelines BEFORE full Phase-G spend when they are
# (near-)identical — assumption 6's cheap in-run verification.
#
# Gate semantics (plan §7): Gate 1 per seed at the forced sampled rung, no
# rung walk. ONE seed failing terminal conjuncts A/B stops THAT seed only
# (driver exit 3 + its own failure sentinel); the round continues on the
# survivor. BOTH seeds failing kills the round (exit 3, no [phase=done]).
# `[phase=done]` is emitted ONLY by this script's terminal echo after BOTH
# seeds' pipelines + uploads complete.
#
# SMOKE=1 runs the SAME sequence on CPU against the tiny-real fixture model
# (issue1426_tiny_e2e_fixture.py), faking ONLY the vLLM boundary
# (--synthetic-completions; per-seed corpora differ via the --gen-seed salt)
# and the Hub boundary (--no-upload) — the smoke/production unification
# contract. The frozen-layer read exercises the SAME mechanical heredoc
# against a synthesized fixture blob (layers {0, 1} — the committed primary's
# {12, 24} do not exist in the 2-layer fixture model).
#
# Production (GCP capture-7b lane, plan §9):
#   uv run python scripts/dispatch_issue.py --issue 1426 --intent capture-7b \
#     --gpus 2 --time-budget-hours 12 --repo-branch issue-1426-sampled \
#     --workload-cmd "bash scripts/issue1426_sampled_dispatch.sh"
#
# CPU tiny-e2e smoke (VM; fixture model built first):
#   uv run python scripts/issue1426_tiny_e2e_fixture.py \
#     --out /tmp/issue-1426-sampled-smoke/tiny_model
#   SMOKE=1 bash scripts/issue1426_sampled_dispatch.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE="${SMOKE:-0}"
SEEDS=(42 137)
if [ "$SMOKE" = "1" ]; then
  # Shared-VM smoke: thread caps are mandatory (code-style.md #847); pod/GCE
  # production launches never carry them (dedicated GPUs keep full width).
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
  SMOKE_ROOT="${SMOKE_ROOT:-/tmp/issue-1426-sampled-smoke}"
  LOG_DIR="${LOG_DIR:-$SMOKE_ROOT/logs}"
  MODEL_DIR="${MODEL_DIR:-$SMOKE_ROOT/tiny_model}"
  mkdir -p "$SMOKE_ROOT" "$LOG_DIR"
  DRIVER_FLAGS=(--smoke --device cpu --model "$MODEL_DIR" --synthetic-completions
    --contexts "${SMOKE_CONTEXTS:-6}" --probes "${SMOKE_PROBES:-4}" --no-upload
    --log-dir "$LOG_DIR")
  FIT_NPERMS=10 FIT_NBOOT=50
  NEED_GB_START=2
  WIDTH="${SMOKE_WIDTH:-2}"
  DIFF_TIMEOUT_S="${DIFF_TIMEOUT_S:-900}" DIFF_POLL_S="${DIFF_POLL_S:-5}"
  # Fixture primary blob: the mechanical frozen-layer read (verbatim cap16k
  # heredoc below) needs the exact key path; the committed primary's {12, 24}
  # exceed the 2-layer fixture model, so the smoke synthesizes {0, 1}.
  PRIMARY_BOOT_BLOB="${PRIMARY_BOOT_BLOB:-$SMOKE_ROOT/primary_bootstrap_fixture.json}"
  if [ ! -f "$PRIMARY_BOOT_BLOB" ]; then
    cat > "$PRIMARY_BOOT_BLOB" <<'JSON'
{
  "dv": "SMOKE FIXTURE — primary frozen-layer conventions at the 2-layer fixture grain",
  "by_regime": {
    "avg_q": {"layer_conventions": {"primary_frozen_direct_best_layer": 0}},
    "indiv": {"layer_conventions": {"primary_frozen_direct_best_layer": 1}}
  }
}
JSON
  fi
else
  # #1426 crash-fix inheritance (issue1426_dispatch.sh:49-53): the shared data
  # repo's 256-commits/hr rate limit resets on an ~1 h horizon; 5400 s rides
  # out a fleet-wide 429 storm (inherited by the driver + every fit subprocess
  # via os.environ passthrough).
  export EPM_HF_RETRY_BUDGET_S="${EPM_HF_RETRY_BUDGET_S:-5400}"
  # vLLM v1 EngineCore dies silently under fork() (gotchas.md #628); the
  # driver setdefaults this too — the export makes the dispatch self-sufficient.
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  LOG_DIR=""
  DRIVER_FLAGS=(--gpu)
  FIT_NPERMS=1000 FIT_NBOOT=2000   # plan §4.2 item 5 invocation B (full-size)
  NEED_GB_START=60                 # plan §9 out-root binding
  WIDTH=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ "$WIDTH" -lt 1 ]; then
    echo "[dispatch] FATAL: no CUDA device visible (nvidia-smi -L empty) on a GPU workload"
    exit 1
  fi
  DIFF_TIMEOUT_S="${DIFF_TIMEOUT_S:-7200}" DIFF_POLL_S="${DIFF_POLL_S:-15}"
  PRIMARY_BOOT_BLOB="${PRIMARY_BOOT_BLOB:-$REPO_ROOT/eval_results/issue_1426/bootstrap_deltaskill.json}"
fi
DRIVER="$REPO_ROOT/scripts/issue1426_run.py"

seed_out_dir() {
  local s=$1
  if [ "$SMOKE" = "1" ]; then echo "$SMOKE_ROOT/data_s$s"; else echo "$REPO_ROOT/data/issue_1426_sampled_s$s"; fi
}
seed_eval_dir() {
  local s=$1
  if [ "$SMOKE" = "1" ]; then echo "$SMOKE_ROOT/eval/seed$s"; else echo "$REPO_ROOT/eval_results/issue_1426/sampled-rollout-robustness/seed$s"; fi
}

# The per-seed HF subtree root (plan §4.3) — derived from the pinned constant,
# never a hand-typed literal (tests/test_issue1426_hf_prefixes.py pins it).
SAMPLED_ROOT=$(uv run python -c "
import sys
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
from issue1426_common import SAMPLED_ROLLOUT_PREFIX_1426
print(SAMPLED_ROLLOUT_PREFIX_1426)
")
echo "[dispatch] sampled HF root: $SAMPLED_ROOT | width: $WIDTH | smoke: $SMOKE"

# ── headroom assert (plan §9: assert against the mount the out-dirs resolve to) ──
echo "[phase=dispatch_headroom]"
OUT42_DIR="$(seed_out_dir 42)"
mkdir -p "$OUT42_DIR"
uv run python -c "
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
free = assert_out_root_headroom('$OUT42_DIR', $NEED_GB_START, phase='sampled-start')
print(f'[dispatch] out-root headroom OK: {free:.1f} GB free at $OUT42_DIR')
"

# ── frozen indices read MECHANICALLY from the PRIMARY blob (verbatim cap16k
# heredoc, issue1426_dispatch.sh:144-156 — never hand-pinned; fails loud on a
# missing key before any GPU spend, plan assumption 1) ────────────────────────
echo "[phase=dispatch_frozen_layers]"
FROZEN_LAYERS=$(uv run python - "$PRIMARY_BOOT_BLOB" <<'PY'
import json
import sys

blob = json.loads(open(sys.argv[1]).read())
layers = sorted(
    {
        int(blob["by_regime"][r]["layer_conventions"]["primary_frozen_direct_best_layer"])
        for r in ("avg_q", "indiv")
    }
)
print(" ".join(str(x) for x in layers))
PY
)
echo "[dispatch] primary frozen layers: $FROZEN_LAYERS (source: $PRIMARY_BOOT_BLOB)"

# ── per-seed pipeline: invocation A → manifest validation → invocation B →
# scoped fit upload → frozen-layer diagnostic ─────────────────────────────────
run_seed() {
  local s=$1
  local out evalout figscratch
  out="$(seed_out_dir "$s")"
  evalout="$(seed_eval_dir "$s")"
  figscratch="$out/figures_scratch"
  mkdir -p "$out" "$evalout" "$figscratch"

  local resume=()
  if [ -f "$out/run_state.json" ]; then
    resume=(--skip-gen)
    echo "[dispatch] seed$s: prior run_state.json found — resuming with --skip-gen"
  fi

  echo "[phase=dispatch_invocation_a_s$s]"
  uv run python "$DRIVER" \
    --out-dir "$out" --eval-out "$evalout" --figures-dir "$figscratch" \
    --phases extract --rung sample --gen-seed "$s" \
    --hf-prefix-root "$SAMPLED_ROOT/seed$s" \
    "${DRIVER_FLAGS[@]}" ${resume[@]+"${resume[@]}"}

  # SMOKE-only fault injection (review-1 Critical negative smoke): a
  # MID-function failure — after invocation A, before the manifest check —
  # must propagate through the EXIT-trap rc plumbing (dispatch.rc != 0, no
  # downstream step runs). Never reachable in production (SMOKE guard).
  #   SMOKE=1 SMOKE_FAULT_SEEDS="42 137" bash scripts/issue1426_sampled_dispatch.sh
  if [ "$SMOKE" = "1" ] && [[ " ${SMOKE_FAULT_SEEDS:-} " == *" $s "* ]]; then
    echo "[dispatch] FAULT-INJECTION seed$s: forced post-invocation-A failure" \
      "(SMOKE_FAULT_SEEDS='${SMOKE_FAULT_SEEDS:-}')"
    false
  fi

  # Store-manifest provenance validation (critic addition A — the load-bearing
  # cross-seed guard; FAILS LOUD before any fit on rung/gen_seed mismatch).
  echo "[phase=dispatch_manifest_check_s$s]"
  uv run python - "$out" "$s" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from issue1426_common import validate_sampled_store_manifest

out, seed = Path(sys.argv[1]), int(sys.argv[2])
man = json.loads((out / "store" / "manifest.json").read_text())
report = validate_sampled_store_manifest(man, expected_seed=seed)
print(f"[dispatch] seed{seed} store-manifest provenance OK: {report}")
head = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
sha = report.get("code_sha")
tag = "MATCH" if sha == head else "DIFFERS (report-not-stop: a crash-fix resume may span commits)"
print(f"[dispatch] seed{seed} store code SHA {sha} vs checkout {head[:12]}: {tag}")
PY

  echo "[phase=dispatch_invocation_b_s$s]"
  # shellcheck disable=SC2086 — FROZEN_LAYERS is a space-separated int list
  uv run python "$DRIVER" \
    --out-dir "$out" --eval-out "$evalout" --figures-dir "$figscratch" \
    --phases f1 --skip-gen --layers $FROZEN_LAYERS --no-mlp \
    --n-perms "$FIT_NPERMS" --n-boot "$FIT_NBOOT" --no-upload \
    "${DRIVER_FLAGS[@]}"

  if [ "$SMOKE" != "1" ]; then
    echo "[phase=dispatch_fit_upload_s$s]"
    # --hf-prefix-root threads ONLY the extract-phase upload sites (driver
    # guard) — the dispatch uploads the invocation-B fit outputs itself,
    # scoped to sampled_rollout/seed<s>/fit_results (the cap16k pattern).
    uv run python - "$evalout" "$s" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(".env")
from issue928_common import upload_folder_scoped_verify
from issue1426_common import SAMPLED_ROLLOUT_PREFIX_1426

evald, seed = Path(sys.argv[1]), int(sys.argv[2])
names = sorted(p.name for p in evald.glob("*.json")) + sorted(
    p.name for p in evald.glob("decomp_*.pt")
)
url = upload_folder_scoped_verify(
    evald,
    f"{SAMPLED_ROLLOUT_PREFIX_1426}/seed{seed}/fit_results",
    names,
    f"issue #1426 sampled-rollout seed {seed}: restricted fit outputs ({len(names)} files)",
    allow_patterns=["*.json", "decomp_*.pt"],
    ignore_patterns=["partial/*"],
)
print(f"[dispatch] seed{seed} fit outputs uploaded: {url}")
PY
  else
    echo "[dispatch] SMOKE=1: seed$s fit upload skipped (Hub boundary faked)"
  fi

  echo "[phase=dispatch_frozencheck_s$s]"
  # Frozen-layer re-derivation diagnostic — report-not-stop (parent v4
  # convention; verbatim cap16k frozencheck pattern).
  uv run python - "$PRIMARY_BOOT_BLOB" "$evalout" <<'PY'
import json
import sys
from pathlib import Path

primary = json.loads(open(sys.argv[1]).read())
evald = Path(sys.argv[2])
reread = json.loads((evald / "bootstrap_deltaskill.json").read_text())
regimes = ("avg_q", "indiv")


def frozen(blob):
    return {
        r: int(blob["by_regime"][r]["layer_conventions"]["primary_frozen_direct_best_layer"])
        for r in regimes
    }


p, q = frozen(primary), frozen(reread)
match = {r: p[r] == q[r] for r in regimes}
(evald / "frozen_layer_check.json").write_text(
    json.dumps(
        {
            "dv": "sampled-seed frozen-layer re-derivation vs the primary index "
            "(amendment plan v4 §4.2 item 5)",
            "primary": p,
            "reread": q,
            "match": match,
            "note": (
                "on mismatch the pinned-index reads at the PRIMARY indices carry the "
                "headline; the mismatch is a reported diagnostic, never a stop "
                "(parent v4 convention)"
            ),
        },
        indent=2,
    )
)
print(f"[dispatch] frozen-layer check: primary={p} reread={q} match={match}")
PY

  echo "[dispatch] seed$s pipeline COMPLETE"
}

# ── fan-out plumbing: per-seed process groups + sed-prefixed logs ─────────────
declare -A SEED_PID SEED_PGID SEED_RC

launch_seed() { # $1 = seed, $2 = CUDA_VISIBLE_DEVICES value ("" = no pin, smoke)
  local s=$1 cvd=$2
  local out
  out="$(seed_out_dir "$s")"
  mkdir -p "$out"
  rm -f "$out/dispatch.rc"
  set -m # own process group per seed pipeline: kill -- -PGID reaps vLLM workers too
  (
    # rc recorded via EXIT trap so errexit stays LIVE inside run_seed (review-1
    # Critical: `run_seed || rc=$?` suppressed errexit inside the function, so
    # a failed invocation A / manifest validation / fit fell through to the
    # trailing echo and recorded rc=0 — unvalidated fits ran and a
    # both-seeds-failed round emitted [phase=done] as success). A SIGKILLed
    # subshell leaves NO rc file -> wait_seed's 99 default still fails.
    trap 'echo "$?" > "$out/dispatch.rc"' EXIT
    if [ -n "$cvd" ]; then export CUDA_VISIBLE_DEVICES="$cvd"; fi
    run_seed "$s"
  ) 2>&1 | sed -u "s/^/[s$s] /" &
  SEED_PID[$s]=$!
  SEED_PGID[$s]=$(ps -o pgid= -p "${SEED_PID[$s]}" 2>/dev/null | tr -d ' ' || true)
  set +m
  echo "[dispatch] seed$s launched (cvd='${cvd}' pid=${SEED_PID[$s]} pgid=${SEED_PGID[$s]})"
}

kill_seed() {
  local s=$1 pgid=${SEED_PGID[$s]:-}
  [ -n "$pgid" ] || return 0
  kill -TERM -- "-$pgid" 2>/dev/null || true
  sleep 10
  kill -KILL -- "-$pgid" 2>/dev/null || true
}

wait_seed() { # records SEED_RC[$s]; never trips set -e
  local s=$1 out
  out="$(seed_out_dir "$s")"
  wait "${SEED_PID[$s]}" 2>/dev/null || true
  SEED_RC[$s]=$(cat "$out/dispatch.rc" 2>/dev/null || echo 99)
  # PID/PGID-recycling safety (review-1 minor 1): once the pipeline has
  # exited, its PGID may be recycled — a later kill_seed (width-1
  # fail_differentiation) must never TERM an unrelated process group.
  SEED_PGID[$s]=""
  echo "[dispatch] seed$s pipeline exited rc=${SEED_RC[$s]}"
}

# Early per-request-seed differentiation check (critic addition B): waits for
# BOTH seeds' Gate-1 slices, compares them, exit 9 on (near-)identical, exit 8
# on timeout, exit 0 on pass OR a documented skip (a seed that gate-failed or
# died pre-gate has no slice to compare — the wait/rc path reports it).
monitor_differentiation() { # $1 outA $2 outB $3 pidA $4 pidB (pid 0 = already complete)
  echo "[phase=dispatch_seed_differentiation]"
  uv run python - "$1" "$2" "$3" "$4" "$DIFF_TIMEOUT_S" "$DIFF_POLL_S" <<'PY'
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from issue1426_common import check_seed_differentiation, gate_slice_status

out_a, out_b = Path(sys.argv[1]), Path(sys.argv[2])
pid_a, pid_b = int(sys.argv[3]), int(sys.argv[4])
timeout_s, poll_s = float(sys.argv[5]), float(sys.argv[6])


def alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


t0 = time.time()
while True:
    st_a, st_b = gate_slice_status(out_a), gate_slice_status(out_b)
    if st_a == "ready" and st_b == "ready":
        try:
            report = check_seed_differentiation(out_a, out_b)
        except SystemExit as e:
            print(str(e), flush=True)
            raise SystemExit(9) from None
        print(f"[dispatch] seed-differentiation check PASS: {report}", flush=True)
        raise SystemExit(0)
    skips = [
        (str(out), st)
        for out, pid, st in ((out_a, pid_a, st_a), (out_b, pid_b, st_b))
        if st == "gate_failed" or (st == "pending" and not alive(pid))
    ]
    if skips:
        print(
            f"[dispatch] seed-differentiation check SKIPPED — {skips} "
            "(no comparable slice; the per-seed rc path reports the failure)",
            flush=True,
        )
        raise SystemExit(0)
    if time.time() - t0 > timeout_s:
        print(
            f"[dispatch] seed-differentiation monitor TIMED OUT after {timeout_s:.0f}s "
            "waiting for gate artifacts — failing loud (a wedged gate is a real problem)",
            flush=True,
        )
        raise SystemExit(8)
    time.sleep(poll_s)
PY
}

fail_differentiation() { # $1 = monitor rc
  echo "[dispatch] seed-differentiation FAILED (rc=$1) — killing seed pipelines before full spend"
  kill_seed 42
  kill_seed 137
  wait 2>/dev/null || true
  exit 2
}

# ── launch: width 2 ⇒ concurrent seeds, one per GPU; width 1 ⇒ sequential ────
if [ "$WIDTH" -ge 2 ]; then
  echo "[phase=dispatch_launch_width2]"
  launch_seed 42 "$([ "$SMOKE" = "1" ] && echo "" || echo 0)"
  launch_seed 137 "$([ "$SMOKE" = "1" ] && echo "" || echo 1)"
  mon_rc=0
  # RC_CAPTURE_EXEMPT: monitor_differentiation body = one child `uv run python` heredoc as
  # its last meaningful command, so the captured rc IS the child's own exit status (a child
  # process keeps its own errexit) — code-review r1 ruled this shape a non-instance (#1516).
  monitor_differentiation "$(seed_out_dir 42)" "$(seed_out_dir 137)" \
    "${SEED_PID[42]}" "${SEED_PID[137]}" || mon_rc=$?
  [ "$mon_rc" -eq 0 ] || fail_differentiation "$mon_rc"
  wait_seed 42
  wait_seed 137
else
  echo "[phase=dispatch_launch_width1]"
  launch_seed 42 "$([ "$SMOKE" = "1" ] && echo "" || echo 0)"
  wait_seed 42
  launch_seed 137 "$([ "$SMOKE" = "1" ] && echo "" || echo 0)"
  # The check fires as soon as seed 137's gate slice lands (seed 42's is on
  # disk): a duplicated corpus costs seed-42-full + seed-137-gate, never 2x.
  mon_rc=0
  # RC_CAPTURE_EXEMPT: monitor_differentiation body = one child `uv run python` heredoc as
  # its last meaningful command, so the captured rc IS the child's own exit status (a child
  # process keeps its own errexit) — code-review r1 ruled this shape a non-instance (#1516).
  monitor_differentiation "$(seed_out_dir 42)" "$(seed_out_dir 137)" \
    0 "${SEED_PID[137]}" || mon_rc=$?
  [ "$mon_rc" -eq 0 ] || fail_differentiation "$mon_rc"
  wait_seed 137
fi

# ── per-seed outcome accounting (plan §7 kill criteria) ───────────────────────
RC42=${SEED_RC[42]} RC137=${SEED_RC[137]}
if [ "$RC42" -ne 0 ] && [ "$RC137" -ne 0 ]; then
  echo "[dispatch] BOTH seeds failed (rc42=$RC42 rc137=$RC137) — round killed (plan §7);"
  echo "[dispatch] per-seed failure sentinels were written by the driver."
  exit 3
fi
if [ "$RC42" -ne 0 ] || [ "$RC137" -ne 0 ]; then
  echo "[dispatch] ONE seed failed (rc42=$RC42 rc137=$RC137) — continuing on the survivor" \
    "(plan §7: single-seed gate failure stops that seed only; caveat carried into the report)"
fi

# ── finalize LAST: the ONE epm:results sentinel + [phase=done] ────────────────
echo "[phase=dispatch_finalize]"
uv run python - "$(seed_out_dir 42)" "$(seed_eval_dir 42)" "$RC42" \
  "$(seed_out_dir 137)" "$(seed_eval_dir 137)" "$RC137" "$LOG_DIR" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from issue928_common import write_sentinel
from issue1426_common import SAMPLED_ROLLOUT_PREFIX_1426

out42, eval42, rc42 = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
out137, eval137, rc137 = Path(sys.argv[4]), Path(sys.argv[5]), int(sys.argv[6])
log_dir = Path(sys.argv[7]) if sys.argv[7] else None


def seed_note(seed: int, out: Path, evald: Path, rc: int) -> dict:
    note: dict = {"rc": rc, "hf_prefix_root": f"{SAMPLED_ROLLOUT_PREFIX_1426}/seed{seed}"}
    gate = out / "gate_report.json"
    if gate.is_file():
        g = json.loads(gate.read_text())
        note["rung"] = g.get("chosen_rung")
        note["gen_seed"] = g.get("gen_seed")
        note["production_max_new_tokens"] = g.get("production_max_new_tokens")
    cov = evald / "coverage_by_family.json"
    if cov.is_file():
        c = json.loads(cov.read_text())
        note["coverage_C_statistic"] = c.get("C_statistic")
        note["coverage_by_family"] = {
            f: v.get("usable_rate") for f, v in c.get("families", {}).items()
        }
    return note


note = {
    "phase": "issue1426_sampled_rollout_robustness",
    "seeds": {
        "42": seed_note(42, out42, eval42, rc42),
        "137": seed_note(137, out137, eval137, rc137),
    },
}
write_sentinel("epm:results", note, out42, log_dir=log_dir, issue=1426)
PY
echo "[phase=done]"
