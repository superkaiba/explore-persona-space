#!/usr/bin/env bash
# Pod dispatcher for task #2546 (CoT context->answer map) — plan v4 §4.2 / §4.3 / §9.
# Modeled on scripts/issue1336_dispatch.sh (sentinel writer, phase done-files, resume).
#
# Usage:
#   bash scripts/issue2546_dispatch.sh --arm {1,2,3} [all|p1_smoke|p2a_pilot|p2_gen_post|
#                                                     p3_gen_short|p4_capture|p4b_capture_rel|
#                                                     ge_gate|p5_fits|p6_publish] [--smoke]
#
# Phase chain (`all`, production): p1_smoke (SMOKE out-root, always) -> (arm 1 only)
# p2a_pilot -> p2_gen_post -> p3_gen_short -> p4_capture -> p4b_capture_rel
# (reliability-draw frozen-layer capture, feeds the P5 split-half ceiling) -> ge_gate
# (G-E, blocking) -> p5_fits -> p6_publish (HF preds/mirror publish + pod-side git
# commit+push of eval_results/issue_2546/) -> results sentinel -> the single terminal
# `[phase=done]` line. p1_smoke rc=4 (RC_FALLBACK_BAND) engages the report-declared
# fallback rung(s) (`--prefill-fallback` / `--decode-fallback`), relaunches the smoke
# ONCE, and threads the engaged flags into every later gen/capture phase (persisted to
# $DONE_DIR/fallbacks_a$ARM.env so resumes inherit them); rc=3 (RC_GATE_FAIL) is a
# designed halt (epm:failure sentinel + exit). Under `--smoke` the `all` chain runs
# p1_smoke only (tiny rehearsal; smoke out-root; kind epm:smoke-result; the P1 rig
# smoke exercises the capture-reliability leg in-script), then the terminal line.
#
# Phase-script CLI contract (units 2/3 implement these entrypoints; keep in sync):
#   scripts/issue2546_gen_capture.py --arm K [--smoke]
#                                    [--phase pilot|gen-post|gen-short|capture|capture-reliability]
#                                    [--prefill-fallback] [--decode-fallback] --out-root <dir>
#   scripts/issue2546_n1m_read.py    --out-root <dir> [--smoke]           (arm-1 pilot leg)
#   scripts/issue2546_fit_cells.py   --arm K [--g0|--publish] [--smoke] --out-root <dir>
#                                    [--prefill-fallback] [--decode-fallback]  (regime keys —
#                                    resume fingerprints; threaded from fallbacks_a$ARM.env)
#                                    [--claim-dir <dir>] (parent-managed work-conserving fan-out)
#
# Contracts honored (pod-side-reporting.md): `[phase=...]` log lines with the single
# reserved terminal `[phase=done]`; per-phase sentinels /workspace/logs/issue-2546-
# <slug>-<epoch>.json written ONLY on rc=0 (write-once, epoch-stamped — never rewritten
# in place); resume/done state lives OUTSIDE the drained sentinel glob (under $OUT_ROOT/
# done); pod-side code never shells scripts/task.py; per-phase
# assert_out_root_headroom before every write-heavy phase (plan §9 disk rows);
# result-push contract (#1205/#1880/#1325/#1482) on the p6_publish git leg.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
[ -d "$REPO_ROOT" ] || REPO_ROOT="$PWD"
cd "$REPO_ROOT"
# Conditional .env sourcing: RunPod pods carry .env; GCE/SLURM lanes export tokens instead.
if [ -f ./.env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
LOG_DIR="${EPS_LOG_DIR:-/workspace/logs}"
DISPATCH_START=$(date +%s)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
ARM=""
SMOKE=""
PHASE_ARG="all"
KEY_PHASE=""
usage() {
    echo "usage: bash scripts/issue2546_dispatch.sh --arm {1,2,3} [PHASE] [--smoke]" >&2
    echo "  PHASE in: all p1_smoke p2a_pilot p2_gen_post p3_gen_short p4_capture" >&2
    echo "            p4b_capture_rel ge_gate p5_fits p6_publish" >&2
    exit 2
}
while [ $# -gt 0 ]; do
    case "$1" in
    --arm)
        ARM="${2:?--arm needs a value}"
        shift 2
        ;;
    --smoke)
        SMOKE="1"
        shift
        ;;
    all | p1_smoke | p2a_pilot | p2_gen_post | p3_gen_short | p4_capture | p4b_capture_rel | ge_gate | p5_fits | p6_publish)
        PHASE_ARG="$1"
        shift
        ;;
    __phase_key)
        # Test/debug probe: print the recipe-keyed done-file key for a phase and exit.
        PHASE_ARG="__phase_key"
        KEY_PHASE="${2:?__phase_key needs a phase name}"
        shift 2
        ;;
    *)
        usage
        ;;
    esac
done
case "$ARM" in
1 | 2 | 3) ;;
*)
    echo "[dispatch2546] FATAL: --arm {1,2,3} is required (got '${ARM}')" >&2
    usage
    ;;
esac

RECIPE_REV="planv4"
phase_key() { printf 'issue2546-%s-a%s%s-%s\n' "$1" "$ARM" "${SMOKE:+-smoke}" "$RECIPE_REV"; }

if [ "$PHASE_ARG" = "__phase_key" ]; then
    # Handled BEFORE GPU derivation / any mkdir so the probe runs on GPU-less
    # dev boxes (no /workspace, no nvidia-smi) without side effects.
    phase_key "$KEY_PHASE"
    exit 0
fi

mkdir -p "$LOG_DIR"

# GPU count guard (informational for phase scripts; they own per-GPU CVD fan-out).
# Width derivation is ALLOCATION-first (#1902/#2251): on shared SLURM nodes
# nvidia-smi enumerates the PHYSICAL node and ignores CUDA_VISIBLE_DEVICES, so a
# detected-count fan-out trespasses onto other tenants' GPUs. Precedence:
# inherited CUDA_VISIBLE_DEVICES array count > SLURM allocation env
# (SLURM_JOB_ID + SLURM_GPUS_ON_NODE, fail LOUD when unresolvable) >
# nvidia-smi enumeration (non-SLURM exclusive hosts: RunPod pods).
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _CVD_ALLOC <<<"$CUDA_VISIBLE_DEVICES"
    NGPU=${#_CVD_ALLOC[@]}
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    # Ordered-id lists first (SLURM_STEP_GPUS > SLURM_JOB_GPUS), count var
    # last — mirrors gen_capture.gpu_allocation()'s ladder (r3 Critical 2).
    if [ -n "${SLURM_STEP_GPUS:-}" ]; then
        IFS=',' read -ra _SLURM_ALLOC <<<"$SLURM_STEP_GPUS"
        NGPU=${#_SLURM_ALLOC[@]}
    elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
        IFS=',' read -ra _SLURM_ALLOC <<<"$SLURM_JOB_GPUS"
        NGPU=${#_SLURM_ALLOC[@]}
    else
        NGPU="${SLURM_GPUS_ON_NODE:-}"
        if [ -z "$NGPU" ]; then
            echo "[dispatch2546] FATAL: SLURM allocation exposes neither CUDA_VISIBLE_DEVICES nor SLURM_STEP_GPUS/SLURM_JOB_GPUS/SLURM_GPUS_ON_NODE — refusing to derive width from node enumeration" >&2
            exit 70
        fi
    fi
else
    NGPU=$( (nvidia-smi --list-gpus 2>/dev/null || true) | wc -l)
fi
case "$NGPU" in
*[!0-9]* | "")
    echo "[dispatch2546] FATAL: bad GPU count '$NGPU'" >&2
    exit 70
    ;;
esac
if [ "$NGPU" -lt 1 ]; then
    echo "[dispatch2546] FATAL: no GPUs visible" >&2
    exit 70
fi

# Out-root split (bidirectional pair; production default, smoke diverted).
# phase_p1_smoke ALWAYS writes into $OUT_ROOT_smoke — even inside the
# production `all` chain — so smoke artifacts never share a root with
# production (g2 Critical; the in-script smoke_ dir namespacing is the belt,
# the root split is the suspenders).
OUT_ROOT_full="${EPS_OUT_ROOT:-/workspace/issue2546}"
OUT_ROOT_smoke="${EPS_OUT_ROOT_SMOKE:-/workspace/issue2546_smoke}"
OUT_ROOT="$OUT_ROOT_full"
if [ -n "$SMOKE" ]; then
    OUT_ROOT="$OUT_ROOT_smoke"
fi
DONE_DIR="$OUT_ROOT/done"
mkdir -p "$OUT_ROOT" "$DONE_DIR"
SMOKE_REPORT="$OUT_ROOT_smoke/out/reports/smoke_a$ARM.json"

# Engaged smoke-fallback rungs (G-A prefill band / G-B3 arm-3 decode band) —
# persisted under $DONE_DIR so a resumed production chain inherits them (the
# gen/capture fingerprints key on these flags, so a mismatch fails loud).
FALLBACK_ENV="$DONE_DIR/fallbacks_a$ARM.env"
PREFILL_FB=""
DECODE_FB=""
if [ -f "$FALLBACK_ENV" ]; then
    # shellcheck disable=SC1090
    . "$FALLBACK_ENV"
fi
compose_fb_args() {
    FB_ARGS=""
    if [ -n "${PREFILL_FB:-}" ]; then FB_ARGS="$FB_ARGS --prefill-fallback"; fi
    if [ -n "${DECODE_FB:-}" ]; then FB_ARGS="$FB_ARGS --decode-fallback"; fi
}
compose_fb_args
persist_fallbacks() {
    {
        echo "PREFILL_FB=${PREFILL_FB}"
        echo "DECODE_FB=${DECODE_FB}"
    } >"$FALLBACK_ENV"
}

# Per-phase headroom floors (GB; plan §9 disk rows: arm-1/2 store ~62-65 GB in-flight,
# arm-3 in-flight peak < ~80 GB under per-corpus upload-then-free). Env-overridable.
NEED_GB_GEN="${EPS_NEED_GB_GEN:-6}"
NEED_GB_SHORT="${EPS_NEED_GB_SHORT:-4}"
if [ "$ARM" = "3" ]; then
    NEED_GB_CAPTURE="${EPS_NEED_GB_CAPTURE:-80}"
else
    NEED_GB_CAPTURE="${EPS_NEED_GB_CAPTURE:-65}"
fi
NEED_GB_FITS="${EPS_NEED_GB_FITS:-8}"
# P4b rel store per arm: ~4 draws x quota rows x 3 kinds x 3 layers bf16 ~ <1 GB
# in-flight (per-stem upload-then-free); 4 GB covers headroom generously.
NEED_GB_CAPREL="${EPS_NEED_GB_CAPREL:-4}"
if [ -n "$SMOKE" ]; then
    # Gate-calibration parity (#1345 class): production-sized disk floors would
    # spuriously kill the tiny smoke leg; smoke floors cover the smoke footprint only.
    NEED_GB_GEN=2
    NEED_GB_SHORT=2
    NEED_GB_CAPTURE=5
    NEED_GB_FITS=2
    NEED_GB_CAPREL=2
fi

echo "[dispatch2546] arm=$ARM phase=$PHASE_ARG smoke=${SMOKE:-0} ngpu=$NGPU out_root=$OUT_ROOT fb_args=${FB_ARGS:-none}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sentinel writer (poll_pipeline contract: _SENTINEL_REQUIRED_KEYS; write-once,
# epoch-stamped path in the drained glob issue-2546-*.json).
emit_signal() { # $1 kind, $2 slug, $3 gate, $4 blocks_pipeline(true|false), $5 note
    SIG_KIND="$1" SIG_SLUG="$2" SIG_GATE="$3" SIG_BLOCKS="$4" SIG_NOTE="$5" \
        SIG_DIR="$LOG_DIR" SIG_SMOKE="${SMOKE:-}" \
        uv run python - <<'PY'
import json
import os
import time

ts = int(time.time())
slug = os.environ["SIG_SLUG"]
# pid-uniquified (r3 g1 minor 5): two same-second emissions must never
# collide on an epoch-only filename (write-once contract).
path = os.path.join(os.environ["SIG_DIR"], f"issue-2546-{slug}-{ts}-{os.getpid()}.json")
payload = {
    "sentinel_schema_version": 1,
    "kind": os.environ["SIG_KIND"],
    "version": 1,
    "task_id": 2546,
    "gate": os.environ["SIG_GATE"],
    "by": "issue2546_dispatch",
    "blocks_pipeline": os.environ["SIG_BLOCKS"] == "true",
    "smoke": bool(os.environ.get("SIG_SMOKE")),
    "note": os.environ["SIG_NOTE"],
}
with open(path, "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[signal] wrote sentinel {path}")
PY
}

assert_headroom() { # $1 phase name, $2 need_gb — plan §9 rider 5 (resume-aware form)
    PHASE_NAME="$1" NEED_GB="$2" OUT_ROOT_ENV="$OUT_ROOT" uv run python - <<'PY'
import os

from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

assert_out_root_headroom(
    os.environ["OUT_ROOT_ENV"], float(os.environ["NEED_GB"]), phase=os.environ["PHASE_NAME"]
)
PY
}

phase_done() { [ -f "$DONE_DIR/$(phase_key "$1").done" ]; }
mark_phase() { : >"$DONE_DIR/$(phase_key "$1").done"; }

read_fallbacks_from_report() {
    # Parse the smoke report's fallbacks_available list into PREFILL_FB/DECODE_FB.
    if [ ! -f "$SMOKE_REPORT" ]; then
        echo "[dispatch2546] FATAL: smoke rc=4 but report missing at $SMOKE_REPORT" >&2
        exit 4
    fi
    fb=$(SR="$SMOKE_REPORT" uv run python - <<'PY'
import json
import os

rep = json.load(open(os.environ["SR"]))
print(" ".join(rep.get("fallbacks_available", [])))
PY
    )
    case " $fb " in *" prefill "*) PREFILL_FB=1 ;; esac
    case " $fb " in *" decode "*) DECODE_FB=1 ;; esac
    compose_fb_args
}

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

phase_p1_smoke() {
    # P1 (arm 1: 240-row) / P1.2-P1.3 (arms 2-3: 60-row lite) rig smoke; gates G-A..G-F
    # evaluated inside the phase script (production entrypoint, production models).
    # ALWAYS into the SMOKE out-root (even in the production `all` chain, where
    # $OUT_ROOT is the production root). rc routing: 0 PASS; 4 = declared
    # fallback band -> engage the report's rung(s) + relaunch ONCE; 3 = gate
    # FAIL (designed halt); anything else = crash (propagate).
    echo "[phase=p1_smoke]"
    mkdir -p "$OUT_ROOT_smoke"
    set +e
    # shellcheck disable=SC2086
    uv run python scripts/issue2546_gen_capture.py --arm "$ARM" --smoke \
        --out-root "$OUT_ROOT_smoke" $FB_ARGS
    rc=$?
    set -e
    if [ "$rc" -eq 4 ]; then
        prev_args="$FB_ARGS"
        read_fallbacks_from_report
        if [ "$FB_ARGS" = "$prev_args" ]; then
            emit_signal "epm:failure" "smoke-fail-a$ARM" "p1_smoke" "true" \
                "failure_class: code — arm $ARM P1 rig smoke rc=4 (fallback band) with no unengaged fallback rung (engaged:${FB_ARGS:-none}); report=$SMOKE_REPORT"
            echo "[dispatch2546] FATAL: p1_smoke rc=4 with no new fallback rung" >&2
            exit 4
        fi
        # Invalidate the FIRST attempt's smoke artifacts BEFORE the fallback
        # relaunch (r3 Critical 1): the gen fingerprints now differ, so the
        # relaunch would otherwise die on the refuse-to-mix RuntimeError.
        # Artifact subdirs ONLY — never the whole root ($DONE_DIR and
        # $FALLBACK_ENV live under it in --smoke mode). The worker-side
        # fp_sha stage/chunk gating is the belt; this wipe is the suspenders.
        for sub in rollouts work store fitcache; do
            rm -rf "${OUT_ROOT_smoke:?}/$sub"
        done
        echo "[dispatch2546] p1_smoke fallback band: wiped smoke artifact subdirs; relaunching smoke ONCE with$FB_ARGS"
        set +e
        # shellcheck disable=SC2086
        uv run python scripts/issue2546_gen_capture.py --arm "$ARM" --smoke \
            --out-root "$OUT_ROOT_smoke" $FB_ARGS
        rc=$?
        set -e
    fi
    if [ "$rc" -ne 0 ]; then
        emit_signal "epm:failure" "smoke-fail-a$ARM" "p1_smoke" "true" \
            "failure_class: code — arm $ARM P1 rig smoke FAILED rc=$rc (RC_GATE_FAIL=3 designed halt; gates in $SMOKE_REPORT)"
        echo "[dispatch2546] FATAL: p1_smoke rc=$rc" >&2
        exit "$rc"
    fi
    persist_fallbacks
    emit_signal "epm:smoke-result" "smoke-a$ARM" "p1_smoke" "false" \
        "arm $ARM P1 rig smoke complete (G-A..G-F evaluated in-script)${FB_ARGS:+; engaged fallbacks:$FB_ARGS}; report=$SMOKE_REPORT"
}

phase_p2a_pilot() {
    # Arm-1 pilot: full gsm8k_test1319 gen+capture (measures production per-row walls)
    # + the frozen n1m directional read (plan §4.2 P2a).
    if [ "$ARM" != "1" ]; then
        echo "[dispatch2546] p2a_pilot is arm-1 only — skipping on arm $ARM (no done-mark)"
        PHASE_SKIPPED=1
        return 0
    fi
    echo "[phase=p2a_pilot]"
    assert_headroom p2a_pilot "$NEED_GB_GEN"
    # shellcheck disable=SC2086
    uv run python scripts/issue2546_gen_capture.py --arm 1 --phase pilot \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke} $FB_ARGS
    uv run python scripts/issue2546_n1m_read.py --out-root "$OUT_ROOT" ${SMOKE:+--smoke}
    emit_signal "epm:progress" "pilot" "p2a_pilot" "false" \
        "arm 1 pilot complete: gsm8k_test1319 gen+capture + frozen n1m read; out_root=$OUT_ROOT"
}

phase_p2_gen_post() {
    echo "[phase=p2_gen_post]"
    assert_headroom p2_gen_post "$NEED_GB_GEN"
    # shellcheck disable=SC2086
    uv run python scripts/issue2546_gen_capture.py --arm "$ARM" --phase gen-post \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke} $FB_ARGS
    emit_signal "epm:progress" "gen-post-a$ARM" "p2_gen_post" "false" \
        "arm $ARM post-side generation complete (rollout text uploaded pre-reduction); out_root=$OUT_ROOT"
}

phase_p3_gen_short() {
    echo "[phase=p3_gen_short]"
    assert_headroom p3_gen_short "$NEED_GB_SHORT"
    # shellcheck disable=SC2086
    uv run python scripts/issue2546_gen_capture.py --arm "$ARM" --phase gen-short \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke} $FB_ARGS
    emit_signal "epm:progress" "gen-short-a$ARM" "p3_gen_short" "false" \
        "arm $ARM short-side generation complete; out_root=$OUT_ROOT"
}

phase_p4_capture() {
    echo "[phase=p4_capture]"
    assert_headroom p4_capture "$NEED_GB_CAPTURE"
    # shellcheck disable=SC2086
    uv run python scripts/issue2546_gen_capture.py --arm "$ARM" --phase capture \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke} $FB_ARGS
    emit_signal "epm:progress" "capture-a$ARM" "p4_capture" "false" \
        "arm $ARM capture complete (per-corpus upload-then-free; store verified-uploaded); out_root=$OUT_ROOT"
}

phase_p4b_capture_rel() {
    # P4b (U4): teacher-force the persisted reliability-draw TEXT at the arm's
    # FROZEN layer subset only (per-draw rel_ stems; feeds run_reliability_unit).
    echo "[phase=p4b_capture_rel]"
    assert_headroom p4b_capture_rel "$NEED_GB_CAPREL"
    # shellcheck disable=SC2086
    uv run python scripts/issue2546_gen_capture.py --arm "$ARM" --phase capture-reliability \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke} $FB_ARGS
    emit_signal "epm:progress" "capture-rel-a$ARM" "p4b_capture_rel" "false" \
        "arm $ARM reliability-draw capture complete (frozen-layer per-draw rel_ stems uploaded); out_root=$OUT_ROOT"
}

phase_ge_gate() {
    # G-E fit-core reuse gate (blocking, before P5): refit the pinned #825 Qwen S1 cell;
    # PASS iff layer-19 held-out R2 within +/-0.01 of 0.6731. Doubles as the item-(m)
    # device-domain exercise on this pod's own GPUs. Designed halt: epm:failure sentinel
    # (the routing artifact) + non-zero exit.
    echo "[phase=ge_gate]"
    set +e
    uv run python scripts/issue2546_fit_cells.py --g0 --arm "$ARM" --out-root "$OUT_ROOT"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        emit_signal "epm:failure" "ge-fail-a$ARM" "G-E" "true" \
            "failure_class: code — G-E fit-core reuse gate FAILED on arm $ARM pod (rc=$rc): layer-19 held-out R2 outside +/-0.01 of 0.6731 on the pinned #825 Qwen S1 cell (turnstore @ deb7a452)"
        echo "[dispatch2546] FATAL: G-E gate failed rc=$rc" >&2
        exit "$rc"
    fi
    emit_signal "epm:progress" "ge-gate-a$ARM" "G-E" "false" \
        "G-E fit-core reuse gate PASS on arm $ARM pod"
}

phase_p5_fits() {
    echo "[phase=p5_fits]"
    assert_headroom p5_fits "$NEED_GB_FITS"
    # Engaged fallback rungs thread into P5 (r3 Critical 1: the parse-mode
    # rung + run-meta record must match what P2-P4 generated/captured).
    # shellcheck disable=SC2086
    uv run python scripts/issue2546_fit_cells.py --arm "$ARM" \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke} $FB_ARGS
}

publish_results_git() {
    # Pod-side git commit+push of this arm's eval JSONs (pod-side-reporting.md
    # result-push contract, #1205/#1880/#1325/#1482): named expected-path set
    # printed BEFORE any verify; copy out/**/*.json -> eval_results/issue_2546/;
    # commit by explicit pathspec; fetch+rebase BEFORE push (bounded 2 attempts,
    # rebase conflict aborts fail-loud); rev-list==0 push-verify; per-file
    # ls-tree artifact-presence assert against the pushed tree. All BEFORE the
    # results sentinel + [phase=done]. `git push || true` is BANNED here.
    branch=$(git rev-parse --abbrev-ref HEAD)
    dest_rel="eval_results/issue_2546"
    mapfile -t rel_paths < <(cd "$OUT_ROOT/out" && find . -name '*.json' -type f | sed 's|^\./||' | sort)
    if [ "${#rel_paths[@]}" -eq 0 ]; then
        echo "[dispatch2546] FATAL: p6 git publish: EMPTY expected-path set under $OUT_ROOT/out — a vacuous verify is a FAIL (#1482)" >&2
        exit 1
    fi
    echo "[dispatch2546] p6 git publish: ${#rel_paths[@]} expected paths (branch=$branch):"
    printf '  %s\n' "${rel_paths[@]}"
    # Partial-clone pods sparse-checkout only the default cones — open the
    # eval_results/issue_2546 cone before writing there (no-op on full clones).
    if git sparse-checkout list >/dev/null 2>&1; then
        git sparse-checkout add "$dest_rel"
    fi
    for rel in "${rel_paths[@]}"; do
        mkdir -p "$REPO_ROOT/$dest_rel/$(dirname "$rel")"
        cp -f "$OUT_ROOT/out/$rel" "$REPO_ROOT/$dest_rel/$rel"
    done
    git add -f -- "$dest_rel"
    leftover=$(git ls-files --others --ignored --exclude-standard -- "$dest_rel" || true)
    if [ -n "$leftover" ]; then
        echo "[dispatch2546] FATAL: p6 git publish: gitignored files skipped by add: $leftover" >&2
        exit 1
    fi
    git config user.email >/dev/null 2>&1 || git config user.email "eps-pod-2546@local"
    git config user.name >/dev/null 2>&1 || git config user.name "eps issue2546 dispatcher"
    if [ -n "$(git status --porcelain -- "$dest_rel")" ]; then
        git commit -m "task #2546: arm $ARM eval JSONs (P6 publish)" -- "$dest_rel"
    else
        echo "[dispatch2546] p6 git publish: tree already up to date (idempotent re-run)"
    fi
    attempt=0
    while :; do
        git fetch origin "$branch"
        if ! git rebase "origin/$branch"; then
            git rebase --abort
            echo "[dispatch2546] FATAL: p6 git publish: rebase conflict against origin/$branch" >&2
            exit 1
        fi
        if git push origin "$branch"; then
            break
        fi
        attempt=$((attempt + 1))
        if [ "$attempt" -ge 2 ]; then
            echo "[dispatch2546] FATAL: p6 git publish: push failed after $attempt fetch+rebase attempts" >&2
            exit 1
        fi
    done
    behind=$(git rev-list --count "origin/$branch..HEAD")
    if [ "$behind" != "0" ]; then
        echo "[dispatch2546] FATAL: p6 git publish: push-verify rev-list=$behind (expected 0)" >&2
        exit 1
    fi
    missing=0
    for rel in "${rel_paths[@]}"; do
        if [ -z "$(git ls-tree -r "origin/$branch" --name-only -- "$dest_rel/$rel")" ]; then
            echo "[dispatch2546] FATAL: p6 git publish: pushed tree missing $dest_rel/$rel" >&2
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        exit 1
    fi
    echo "[dispatch2546] p6 git publish: ${#rel_paths[@]} paths verified in pushed origin/$branch tree"
}

phase_p6_publish() {
    # P6 (P5 results handoff): (a) HF publish — per-cell preds npz + the full
    # out/ JSON mirror via fit_cells --publish (one upload_folder commit each,
    # exact-set verified in-script); (b) production only: pod-side git
    # commit+push of eval_results/issue_2546/ per the result-push contract.
    echo "[phase=p6_publish]"
    # shellcheck disable=SC2086
    uv run python scripts/issue2546_fit_cells.py --arm "$ARM" --publish \
        --out-root "$OUT_ROOT" ${SMOKE:+--smoke} $FB_ARGS
    if [ -n "$SMOKE" ]; then
        echo "[dispatch2546] smoke mode: skipping pod-side git publish of eval_results/ (HF mirror only)"
        return 0
    fi
    publish_results_git
    emit_signal "epm:progress" "publish-a$ARM" "p6_publish" "false" \
        "arm $ARM P6 publish complete: preds npz + out/ JSON mirror on HF; eval JSONs committed+push-verified on git; out_root=$OUT_ROOT"
}

write_results_sentinel() {
    RES_OUT_ROOT="$OUT_ROOT" RES_ARM="$ARM" RES_SMOKE="${SMOKE:-}" RES_LOG_DIR="$LOG_DIR" \
        RES_NGPU="$NGPU" RES_START="$DISPATCH_START" RES_FB="${FB_ARGS:-}" \
        uv run python - <<'PY'
import glob
import json
import os
import subprocess
import time

out_root = os.environ["RES_OUT_ROOT"]
arm = os.environ["RES_ARM"]
smoke = bool(os.environ.get("RES_SMOKE"))
sha = "unavailable-no-git-checkout"
try:
    p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if p.returncode == 0:
        sha = p.stdout.strip()
except OSError:
    pass
cell_jsons = sorted(glob.glob(os.path.join(out_root, "out", "cells", f"*__a{arm}.json")))
ladder_jsons = sorted(glob.glob(os.path.join(out_root, "out", "ladder", "*.json")))
gpu_hours = round(
    (time.time() - float(os.environ["RES_START"])) / 3600.0 * max(int(os.environ["RES_NGPU"]), 1),
    2,
)
arm_dir = f"smoke_arm{arm}" if smoke else f"arm{arm}"
gate = "p1_smoke" if smoke else "p5_fits"
note = {
    "phase": gate,
    "arm": int(arm),
    "smoke": smoke,
    "engaged_fallbacks": (os.environ.get("RES_FB") or "").split(),
    "eval_paths": cell_jsons[:50] + ladder_jsons[:20],
    "n_cell_jsons": len(cell_jsons),
    "n_ladder_jsons": len(ladder_jsons),
    "gpu_hours_estimate": gpu_hours,
    "reproducibility_card": {
        "hf_data_repo": "superkaiba1/explore-persona-space-data",
        "prefixes": [
            f"issue2546_cotmap/analysis_tensors/thinkstore/{arm_dir}/",
            f"issue2546_cotmap/analysis_tensors/preds/{arm_dir}/",
            "issue2546_cotmap/eval_results_mirror/",
            "issue2546_cotmap/raw_completions/",
            "issue2546_cotmap/corpora_v1/",
        ],
        "git_eval_results": "eval_results/issue_2546/ (pushed on the issue branch by p6_publish)",
        "wandb_url": "n/a (no training in this task)",
        "final_commit_sha": sha,
    },
}
kind = "epm:smoke-result" if smoke else "epm:results"
slug = kind.replace(":", "_")
ts = int(time.time())
path = os.path.join(
    os.environ["RES_LOG_DIR"], f"issue-2546-{slug}-a{arm}-{ts}-{os.getpid()}.json"
)
payload = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "task_id": 2546,
    "gate": gate,
    "by": "issue2546_dispatch",
    "blocks_pipeline": False,
    "smoke": smoke,
    "note": json.dumps(note),
}
with open(path, "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[signal] wrote results sentinel {path}")
PY
}

run_phase() { # $1 = phase name
    if phase_done "$1"; then
        echo "[dispatch2546] phase $1 already complete — skipping"
        return 0
    fi
    PHASE_SKIPPED=""
    "phase_$1"
    if [ -n "$PHASE_SKIPPED" ]; then
        echo "[dispatch2546] phase $1 skipped — not marking done"
        return 0
    fi
    mark_phase "$1"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$PHASE_ARG" in
all)
    if [ -n "$SMOKE" ]; then
        # Smoke rehearsal: the P1 rig smoke only (tiny, smoke out-root); the production
        # chain runs P1 as its own first, gating phase. The terminal sentinel below is
        # kind epm:smoke-result / gate p1_smoke (never a p5_fits label at smoke).
        run_phase p1_smoke
    else
        run_phase p1_smoke
        if [ "$ARM" = "1" ]; then
            run_phase p2a_pilot
        fi
        run_phase p2_gen_post
        run_phase p3_gen_short
        run_phase p4_capture
        run_phase p4b_capture_rel
        run_phase ge_gate
        run_phase p5_fits
        run_phase p6_publish
    fi
    write_results_sentinel
    echo "[phase=done]"
    ;;
p1_smoke | p2a_pilot | p2_gen_post | p3_gen_short | p4_capture | p4b_capture_rel | ge_gate | p5_fits | p6_publish)
    run_phase "$PHASE_ARG"
    echo "[dispatch2546] single-phase invocation of $PHASE_ARG complete (no terminal done line)"
    ;;
*)
    usage
    ;;
esac
exit 0
