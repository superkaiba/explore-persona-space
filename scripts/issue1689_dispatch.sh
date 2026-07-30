#!/usr/bin/env bash
# Issue #1689 phase dispatcher — invoked via
#   `dispatch_issue.py launch --workload-cmd 'bash scripts/issue1689_dispatch.sh <phase>' --repo-branch issue-1689`.
#
# Emits pod-side sentinel JSONs at /workspace/logs/issue-1689-*.json
# that poll_pipeline.py drains (per CLAUDE.md pod-side reporting contract).
# Prints [phase=<name>] log lines the poller matches; [phase=done] on
# graceful completion + writes the epm:results payload sentinel.
#
# Phases: corpus → render → haiku_u2 → onpolicy → capture → fit_cells →
#         fit_ladder → analyze
#
# Smoke: bash scripts/issue1689_dispatch.sh --smoke  (runs full pipeline
# at tiny n=5 slice through every phase - the PASS_UNIFIED architecture).
set -euo pipefail

# Self-resolve REPO_ROOT: lane-agnostic per the #825 crash-fix precedent
# (never bare REPO_ROOT="$WORKLOAD_ROOT" under set -u).
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"

# Conditional dotenv sourcing (RunPod pushes .env; GCE lane has none).
# See gotchas.md § "source ./.env CONDITIONALLY".
if [ -f ./.env ]; then
    set -a
    . ./.env
    set +a
fi

ISSUE_NUM=1689
ISSUE_SLUG=speaker_lattice
DATA_ROOT="${DATA_ROOT:-data/issue_${ISSUE_NUM}}"
EVAL_ROOT="${EVAL_ROOT:-eval_results/issue_${ISSUE_NUM}}"
STORE_ROOT="${STORE_ROOT:-analysis_tensors/issue_${ISSUE_NUM}/store}"
FIG_ROOT="${FIG_ROOT:-figures/issue_${ISSUE_NUM}}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$DATA_ROOT" "$EVAL_ROOT" "$STORE_ROOT" "$FIG_ROOT"
# LOG_DIR may not be writable off-pod; skip mkdir if unwritable.
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="/tmp"

SMOKE=""
PHASE=""
if [ "${1:-}" = "--smoke" ]; then
    SMOKE="--smoke"
    PHASE="all"
elif [ -n "${1:-}" ]; then
    PHASE="$1"
    if [ "${2:-}" = "--smoke" ]; then
        SMOKE="--smoke"
    fi
else
    PHASE="all"
fi

MODEL_INSTRUCT="Qwen/Qwen2.5-7B-Instruct"
MODEL_BASE="Qwen/Qwen2.5-7B"

run_phase_corpus() {
    echo "[phase=corpus]"
    local out="$DATA_ROOT/two_turn_lmsys.jsonl"
    if [ -n "$SMOKE" ]; then
        # Write a tiny synthetic fixture; the real corpus phase uses
        # scripts/issue1689_gen_corpus.py.
        mkdir -p "$(dirname "$out")"
        uv run python -c "
import json
from pathlib import Path
rows = [
    {'conv_id': f'smoke-{i}', 'u1': f'Hello {i}?', 'a1': f'Reply to {i}.',
     'u2_lmsys': f'Follow-up {i}.'}
    for i in range(5)
]
Path('$out').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
print(f'[corpus] wrote 5 smoke rows to $out')
"
    else
        uv run python scripts/issue1689_gen_corpus.py \
            --out "$out" --n 3800
    fi
}

run_phase_render() {
    echo "[phase=render]"
    local corpus="$DATA_ROOT/two_turn_lmsys.jsonl"
    local out="$DATA_ROOT/rendered.jsonl"
    uv run python scripts/issue1689_render_conditions.py \
        --in "$corpus" --out "$out" --conditions all $SMOKE
}

run_phase_haiku_u2() {
    echo "[phase=haiku_u2]"
    local rendered_dir="$DATA_ROOT/rendered.jsonl"
    for cond in user_haiku_chat user_haiku_naturalistic user_haiku_story; do
        local input="$rendered_dir/${cond}.jsonl"
        local out="$DATA_ROOT/haiku_u2/${cond}.jsonl"
        uv run python scripts/issue1689_haiku_u2_gen.py \
            --in "$input" --out "$out" --condition "$cond" $SMOKE
        [ -z "$SMOKE" ] || break  # smoke: one condition only
    done
}

run_phase_onpolicy() {
    echo "[phase=onpolicy]"
    local rendered_dir="$DATA_ROOT/rendered.jsonl"
    local conds_smoke="assistant_chat"
    local conds_full="assistant_chat assistant_naturalistic assistant_story helios_chat helios_naturalistic helios_story wren_chat wren_naturalistic wren_story dana_chat dana_naturalistic dana_story user_onpolicy_chat user_onpolicy_naturalistic user_onpolicy_story"
    local conds="$conds_full"
    [ -n "$SMOKE" ] && conds="$conds_smoke"
    # Both models iterated in smoke AND full modes — H5 (base vs instruct
    # parity) requires the base arm; a single-model smoke leaves the base
    # branch unexercised (round-2 concern dispatch-base-model-missing).
    local models_smoke="$MODEL_INSTRUCT"
    local models_full="$MODEL_BASE $MODEL_INSTRUCT"
    local models="$models_full"
    [ -n "$SMOKE" ] && models="$models_smoke"  # smoke: instruct only (1 model × 1 cond); full: BOTH
    # NOTE: only the smoke leg is restricted to instruct-only to bound smoke wall-time;
    # full mode iterates both MODEL_BASE and MODEL_INSTRUCT per plan §5 H5.
    for cond in $conds; do
        for model in $models; do
            local input="$rendered_dir/${cond}.jsonl"
            local out="$DATA_ROOT/onpolicy/${cond}_$(basename "$model").jsonl"
            local stats="$EVAL_ROOT/onpolicy_stats/${cond}_$(basename "$model").json"
            uv run python scripts/issue1689_gen_onpolicy.py \
                --in "$input" --out "$out" \
                --stats-out "$stats" \
                --condition "$cond" --model "$model" $SMOKE
        done
    done
}

run_phase_capture() {
    echo "[phase=capture]"
    local rendered_dir="$DATA_ROOT/rendered.jsonl"
    # R13 fix: capture the FULL 21-condition lattice per plan §4/§5, not a
    # smoke-slice. The stale "full path drives via a per-cell loop in
    # production" comment described no such loop; production silently ran
    # smoke-scale (assistant_chat only) and Phase D fit_ladder crashed on
    # missing cells (assistant_naturalistic/L14.pt) after ~13h of upstream
    # compute.
    local conds_smoke="assistant_chat"
    local conds_full="assistant_chat assistant_naturalistic assistant_story dana_chat dana_naturalistic dana_story helios_chat helios_naturalistic helios_story wren_chat wren_naturalistic wren_story user_haiku_chat user_haiku_naturalistic user_haiku_story user_lmsys_chat user_lmsys_naturalistic user_lmsys_story user_onpolicy_chat user_onpolicy_naturalistic user_onpolicy_story"
    local conds="$conds_full"
    [ -n "$SMOKE" ] && conds="$conds_smoke"
    # Same both-models discipline as run_phase_onpolicy: H5 (base vs instruct
    # parity) needs both arms captured. Smoke stays instruct-only to bound wall.
    local models_smoke="$MODEL_INSTRUCT"
    local models_full="$MODEL_BASE $MODEL_INSTRUCT"
    local models="$models_full"
    [ -n "$SMOKE" ] && models="$models_smoke"
    for cond in $conds; do
        for model in $models; do
            # Idempotent skip-if-exists: capture is expensive (LoRA-adapter-free
            # forward passes for prefix/context/answer at 4 layers), and a
            # partial capture (e.g. assistant_chat × both models already
            # completed under R11) is byte-identical to what this rerun would
            # produce (same code path, same inputs). Skip populated cells so a
            # resume can complete the missing 40/42 cells without re-doing
            # the 2 already captured. See save_cell() in issue1689_capture.py:
            # model_slug = model_name.replace("/", "_") → "Qwen_Qwen2.5-7B" /
            # "Qwen_Qwen2.5-7B-Instruct".
            local model_slug
            model_slug=$(echo "$model" | tr '/' '_')
            local cell_dir="${STORE_ROOT}/${model_slug}/${cond}"
            if [ -d "$cell_dir" ] && [ -n "$(ls -A "$cell_dir" 2>/dev/null)" ]; then
                echo "[capture] SKIP ${cell_dir} (already populated)"
                continue
            fi
            local input="$DATA_ROOT/onpolicy/${cond}_$(basename "$model").jsonl"
            [ ! -f "$input" ] && input="$rendered_dir/${cond}.jsonl"
            local skip_upload_flag=""
            [ -n "$SMOKE" ] && skip_upload_flag="--skip-upload"
            uv run python scripts/issue1689_capture.py \
                --in "$input" --out-root "$STORE_ROOT" \
                --condition "$cond" --model "$model" \
                $SMOKE $skip_upload_flag
        done
    done
}

run_phase_fit_cells() {
    echo "[phase=fit_cells]"
    # R13 fix: iterate the FULL 42-cell (2 models × 21 conditions) lattice, not
    # the single hardcoded (Qwen_Qwen2.5-7B-Instruct, assistant_chat) cell.
    # Idempotent skip-if-exists: fit_cells writes one JSON per cell; a resume
    # skips the 1 completed cell (heldout_r2_Qwen_Qwen2.5-7B-Instruct_assistant_chat.json
    # under R11) and processes the remaining 41.
    local models_full="Qwen_Qwen2.5-7B Qwen_Qwen2.5-7B-Instruct"
    local models_smoke="Qwen_Qwen2.5-7B-Instruct"
    local models="$models_full"
    local conds_full="assistant_chat assistant_naturalistic assistant_story dana_chat dana_naturalistic dana_story helios_chat helios_naturalistic helios_story wren_chat wren_naturalistic wren_story user_haiku_chat user_haiku_naturalistic user_haiku_story user_lmsys_chat user_lmsys_naturalistic user_lmsys_story user_onpolicy_chat user_onpolicy_naturalistic user_onpolicy_story"
    local conds_smoke="assistant_chat"
    local conds="$conds_full"
    if [ -n "$SMOKE" ]; then
        models="$models_smoke"
        conds="$conds_smoke"
    fi
    for model_slug in $models; do
        for cond in $conds; do
            local cell="${model_slug}/${cond}"
            local out="$EVAL_ROOT/percell/heldout_r2_${model_slug}_${cond}.json"
            if [ -f "$out" ]; then
                echo "[fit_cells] SKIP ${out} (already exists)"
                continue
            fi
            uv run python scripts/issue1689_fit_cells.py \
                --store-root "$STORE_ROOT" --cell "$cell" --out "$out" $SMOKE
        done
    done
}

run_phase_fit_ladder() {
    echo "[phase=fit_ladder]"
    # Iterate BOTH models in full mode (H5 parity); smoke stays instruct-only.
    # --all-layers loops over CAPTURE_LAYERS=(14,18,19,26) per plan §6 exploratory dump;
    # headline layer stays 19.
    local models_smoke="Qwen_Qwen2.5-7B-Instruct"
    local models_full="Qwen_Qwen2.5-7B Qwen_Qwen2.5-7B-Instruct"
    local models="$models_full"
    [ -n "$SMOKE" ] && models="$models_smoke"
    # In smoke mode we run only the headline layer (19) for wall-time; full mode iterates all 4.
    local all_layers_flag="--all-layers"
    [ -n "$SMOKE" ] && all_layers_flag=""
    for model_slug in $models; do
        local out="$EVAL_ROOT/ladder/ladder_${model_slug}_L19.json"
        uv run python scripts/issue1689_fit_ladder.py \
            --store-root "$STORE_ROOT" --model-slug "$model_slug" \
            --layer 19 --out "$out" $all_layers_flag $SMOKE
    done
}

run_phase_analyze() {
    echo "[phase=analyze]"
    local model_slug="Qwen_Qwen2.5-7B-Instruct"
    local ladder="$EVAL_ROOT/ladder/ladder_${model_slug}_L19.json"
    uv run python scripts/issue1689_analyze.py \
        --ladder-json "$ladder" --out-figs "$FIG_ROOT" \
        --out-manifest "$EVAL_ROOT/manifest.json" \
        --model "$model_slug" $SMOKE
}

run_phase_derived_vs_free() {
    echo "[phase=derived_vs_free]"
    # Sentinel-INDEPENDENT leg (plan v8 s4): completion = job exit + pushed
    # result commits; sentinels stay best-effort telemetry.
    local dvf_root="$EVAL_ROOT/derived_vs_free_B"
    local cms_root="$EVAL_ROOT/context_map_structure"
    local xm_root="$EVAL_ROOT/crossmodel_pairs"
    local xms_root="$EVAL_ROOT/crossmodel_pairs/crossmodel_structure"
    local dvf_figs="$FIG_ROOT/derived_vs_free"
    local push_results=1
    if [ -n "$SMOKE" ]; then
        # Smoke outputs NEVER land on committed paths (scratch-dir redirect).
        local scratch="/tmp/issue-1689-dvf-smoke"
        dvf_root="$scratch/derived_vs_free_B"
        cms_root="$scratch/context_map_structure"
        xm_root="$scratch/crossmodel_pairs"
        xms_root="$scratch/crossmodel_pairs/crossmodel_structure"
        dvf_figs="$scratch/figures"
        push_results=0
    fi
    mkdir -p "$dvf_root" "$cms_root" "$xm_root" "$xms_root"

    # Fix round 3 (#1689): one-shot key migration BEFORE any resume. Pre-fix
    # within-model unit checkpoints were UNQUALIFIED (no model in the key), so
    # base+instruct collided in the shared out-roots (241 base + 11 instruct
    # of 504 computed; merge double-counted each file for both models). The
    # migration renames surviving files to their internally-recorded model's
    # qualified key — retained, never deleted — so only the genuinely-missing
    # units re-run below. Cross-model roots never had unqualified keys (no-op).
    echo "[phase=dvf_migrate_keys]"
    uv run python scripts/issue1689_derived_vs_free.py --phase migrate-keys --out-root "$dvf_root"
    uv run python scripts/issue1689_derived_vs_free.py --phase migrate-keys --out-root "$cms_root"

    # Within-model merges hard-assert the full 504-unit qualified enumeration
    # (2 models x 126 ordered pairs x 2 arms) in full mode; smoke pair subsets
    # carry their own (smaller) exact counts via the merge's missing/dupe gates.
    local expect_units=()
    [ -z "$SMOKE" ] && expect_units=(--expect-units 504)

    # Stage the pinned L19 stores (idempotent; consumer layout is
    # <root>/<model_slug>/<condition>/L19.pt — exact per-file targets).
    local dvf_store="${DVF_STORE_ROOT:-$DATA_ROOT/hf_dl/issue1689_speaker_lattice/analysis_tensors}"
    uv run python scripts/issue1689_derived_vs_free.py --phase stage --store-root "$dvf_store"

    # Width derives from DETECTED GPUs (no smoke-conditional narrowing; CVD is
    # pinned per shard in the launcher env below).
    local ngpu
    ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l) || ngpu=0
    local device="cpu" nsh=1
    if [ "${ngpu:-0}" -ge 1 ]; then device="cuda"; nsh="$ngpu"; fi
    echo "[derived_vs_free] device=$device shards=$nsh"

    # Scale knobs: smoke slices rows/dims/draws through the SAME drivers +
    # pair-subset (--pairs-file) threading every phase below.
    local slice_args=() rot_draws=200 class_draws=40 rank_draws=40 gate1_null=40
    local within_pairs_args=(--pair-set within-model)
    local xm_pairs="$xm_root/crossmodel_pair_specs.json"
    local xm_ladder_draws=(--bootstrap-draws 0 --null-draws 40)
    if [ -n "$SMOKE" ]; then
        # One rung-9 within-model pair (rank-rung eligible) + one cross-model pair.
        printf '%s' '[[["Qwen_Qwen2.5-7B-Instruct","assistant_chat"],["Qwen_Qwen2.5-7B-Instruct","assistant_naturalistic"]]]' \
            > "$dvf_root/smoke_pair.json"
        printf '%s' '[[["Qwen_Qwen2.5-7B","assistant_chat"],["Qwen_Qwen2.5-7B-Instruct","assistant_chat"]]]' \
            > "$xm_root/smoke_xpair.json"
        within_pairs_args=(--pairs-file "$dvf_root/smoke_pair.json")
        xm_pairs="$xm_root/smoke_xpair.json"
        slice_args=(--row-limit 600 --dim-limit 512)
        rot_draws=5; class_draws=4; rank_draws=2; gate1_null=0
        xm_ladder_draws=(--bootstrap-draws 0 --null-draws 0)
    else
        uv run python scripts/issue1689_derived_vs_free.py --phase write-pairs \
            --pair-set cross-model --write-pairs-out "$xm_pairs"
    fi

    # Gate 1 (plan s7): parity vs the published parent pair-arm (always FULL
    # shape) + timing pilot (production shape in full mode; sliced in smoke).
    uv run python scripts/issue1689_derived_vs_free.py --phase gate1 \
        --store-root "$dvf_store" --out-root "$dvf_root" --device "$device" \
        --gate1-null-draws "$gate1_null" --gate1-timing "${slice_args[@]}"

    # Per-leg fence = 2x the Gate-1 pilot-extrapolated wall (plan s9).
    local unit_wall fence
    unit_wall=$(uv run python -c 'import json,sys; d=json.load(open(sys.argv[1])); t=d.get("timing") or {}; print(max(5.0, float(t.get("battery_unit_wall_s") or 60.0)))' "$dvf_root/gate1_report.json")
    fence=$(uv run python -c 'import sys; w=float(sys.argv[1]); n=int(sys.argv[2]); s=int(sys.argv[3]); print(int(2*w*n/max(s,1)) + 900)' "$unit_wall" 588 "$nsh")
    # Structure units (item 7 class-null battery + item-8 rank rung) run ~3x a
    # battery unit — their legs get a 3x fence off the same pilot basis.
    local cms_fence=$((fence * 3))
    echo "[derived_vs_free] gate1 unit_wall=${unit_wall}s fence=${fence}s cms_fence=${cms_fence}s"

    # run_sharded <script> <args...>: fan out one shard per GPU (CVD pinned in
    # the launcher env per shard — the #545 clobber rule), or 1 CPU process.
    run_sharded() {
        local leg_fence="${RUN_FENCE:-$fence}"
        local script="$1"; shift
        if [ "$nsh" -le 1 ]; then
            local cvd_prefix=(env)
            [ "$device" = "cuda" ] && cvd_prefix=(env CUDA_VISIBLE_DEVICES=0)
            timeout --kill-after=60s "$leg_fence" "${cvd_prefix[@]}" \
                uv run python "$script" "$@" --device "$device"
            return $?
        fi
        local pids=() i rc=0 p
        for i in $(seq 0 $((nsh - 1))); do
            env CUDA_VISIBLE_DEVICES="$i" timeout --kill-after=60s "$leg_fence" \
                uv run python "$script" "$@" --device cuda \
                --num-shards "$nsh" --shard-index "$i" \
                > "$LOG_DIR/dvf-shard-$i.log" 2>&1 &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p" || rc=$?; done
        if [ "$rc" -ne 0 ]; then
            echo "[derived_vs_free] shard failure rc=$rc" >&2
            tail -n 60 "$LOG_DIR"/dvf-shard-*.log >&2 || true
            return "$rc"
        fi
    }

    # --- Within-model battery (items 1-6) ---
    echo "[phase=dvf_pairs]"
    run_sharded scripts/issue1689_derived_vs_free.py --phase pairs \
        --store-root "$dvf_store" --out-root "$dvf_root" \
        "${within_pairs_args[@]}" "${slice_args[@]}"
    echo "[phase=dvf_nulls]"
    local cvd0=(env)
    [ "$device" = "cuda" ] && cvd0=(env CUDA_VISIBLE_DEVICES=0)
    "${cvd0[@]}" timeout --kill-after=60s "$fence" uv run python \
        scripts/issue1689_derived_vs_free.py --phase nulls --out-root "$dvf_root" \
        --rotation-draws "$rot_draws" --device "$device"
    uv run python scripts/issue1689_derived_vs_free.py --phase merge \
        --out-root "$dvf_root" "${within_pairs_args[@]}" "${expect_units[@]}"

    # --- Cross-model same-condition pairs (item 9): full ladder + battery ---
    echo "[phase=xm_ladder]"
    local xm_ladder_json="$xm_root/ladder_crossmodel_L19.json"
    # Pair-sharded across every visible GPU (plan §9 C7 — the 84-unit ladder
    # must not run single-process on GPU 0 while siblings idle): shard workers
    # persist per-pair checkpoints only; the --merge pass assembles the final
    # JSON fail-loud on any missing/regime-mismatched pair. nsh=1 (CPU or one
    # GPU) takes run_sharded's single-process branch, which writes the JSON
    # directly — no merge pass needed.
    run_sharded scripts/issue1689_fit_ladder.py \
        --store-root "$dvf_store" \
        --model-slug Qwen_Qwen2.5-7B-Instruct --layer 19 \
        --out "$xm_ladder_json" --pairs-file "$xm_pairs" \
        --checkpoint-dir "$xm_root/xpairs_L19" \
        --engine torch "${xm_ladder_draws[@]}"
    if [ "$nsh" -gt 1 ]; then
        timeout --kill-after=60s "$fence" uv run python \
            scripts/issue1689_fit_ladder.py --store-root "$dvf_store" \
            --model-slug Qwen_Qwen2.5-7B-Instruct --layer 19 \
            --out "$xm_ladder_json" --pairs-file "$xm_pairs" \
            --checkpoint-dir "$xm_root/xpairs_L19" \
            --engine torch --device cpu --merge "${xm_ladder_draws[@]}"
    fi
    echo "[phase=xm_pairs]"
    run_sharded scripts/issue1689_derived_vs_free.py --phase pairs \
        --store-root "$dvf_store" --out-root "$xm_root" \
        --pairs-file "$xm_pairs" "${slice_args[@]}"
    "${cvd0[@]}" timeout --kill-after=60s "$fence" uv run python \
        scripts/issue1689_derived_vs_free.py --phase nulls --out-root "$xm_root" \
        --rotation-draws "$rot_draws" --device "$device"
    uv run python scripts/issue1689_derived_vs_free.py --phase merge \
        --out-root "$xm_root" --pairs-file "$xm_pairs"

    # --- Context-map structure + rank rung (items 7-8 within; item 9 leg) ---
    echo "[phase=cms_units]"
    RUN_FENCE="$cms_fence" run_sharded scripts/issue1689_context_map_structure.py --phase units \
        --store-root "$dvf_store" --out-root "$cms_root" \
        "${within_pairs_args[@]}" "${slice_args[@]}" \
        --class-null-draws "$class_draws" --rank-null-draws "$rank_draws"
    uv run python scripts/issue1689_context_map_structure.py --phase overlap \
        --out-root "$cms_root"
    uv run python scripts/issue1689_context_map_structure.py --phase merge \
        --out-root "$cms_root" "${within_pairs_args[@]}" "${expect_units[@]}"
    echo "[phase=xm_structure]"
    RUN_FENCE="$cms_fence" run_sharded scripts/issue1689_context_map_structure.py --phase units \
        --store-root "$dvf_store" --out-root "$xms_root" \
        --pairs-file "$xm_pairs" "${slice_args[@]}" \
        --class-null-draws "$class_draws" --rank-null-draws "$rank_draws" \
        --crossmodel-ladder-json "$xm_ladder_json"
    uv run python scripts/issue1689_context_map_structure.py --phase overlap \
        --out-root "$xms_root"
    uv run python scripts/issue1689_context_map_structure.py --phase merge \
        --out-root "$xms_root" --pairs-file "$xm_pairs" \
        --crossmodel-ladder-json "$xm_ladder_json"

    # --- Figures (local JSONs) ---
    echo "[phase=dvf_figures]"
    uv run python scripts/issue1689_derived_vs_free_figures.py \
        --dvf-root "$dvf_root" --cms-root "$cms_root" \
        --crossmodel-root "$xm_root" --out-figs "$dvf_figs"

    if [ "$push_results" -eq 1 ]; then
        # --- Compact-bundle upload (ONE upload_folder commit per out-root) ---
        echo "[phase=dvf_upload]"
        local r
        for r in "$dvf_root" "$cms_root" "$xm_root" "$xms_root"; do
            uv run python scripts/issue1689_derived_vs_free.py --phase upload --out-root "$r"
        done
        # --- Result commit + BARE push, verify-then-assert (#1205/#1325) ---
        # Fix round 3: NO-OP gracefully on a non-git tree (the fellows/SLURM
        # rsync lane ships no .git — job 15194 died HERE at the very end; the
        # VM-side orchestrator lands git artifacts on that lane, per the
        # pod-side-reporting SLURM result-landing contract).
        echo "[phase=dvf_push]"
        if ! git rev-parse --git-dir >/dev/null 2>&1; then
            echo "[derived_vs_free] dvf_push SKIP: not a git checkout (rsync lane; VM orchestrator lands git artifacts)"
            echo "[derived_vs_free] leg complete"
            return 0
        fi
        git add "$dvf_root" "$cms_root" "$xm_root" "$dvf_figs"
        git -c user.name="eps-runner" -c user.email="eps-runner@local" \
            commit -m "task #1689: derived-vs-free round results (battery + structure + crossmodel + figures)" \
            || echo "[derived_vs_free] nothing to commit"
        if ! git push origin issue-1689; then
            echo "[derived_vs_free] push failed; retrying once" >&2
            git push origin issue-1689
        fi
        local behind
        behind=$(git rev-list --count origin/issue-1689..HEAD)
        if [ "$behind" -ne 0 ]; then
            echo "[derived_vs_free] push-verify FAILED (${behind} unpushed commits)" >&2
            exit 86
        fi
        # Artifact-presence assert: every declared git-destined result file of
        # THIS round must be in the pushed tree (bundles/*.npz are HF-destined).
        local missing=0 p
        while IFS= read -r p; do
            if [ -z "$(git ls-tree -r origin/issue-1689 --name-only -- "$p")" ]; then
                echo "[derived_vs_free] MISSING from pushed tree: $p" >&2
                missing=1
            fi
        done < <(
            find "$dvf_root" "$cms_root" "$xm_root" -name '*.json' -not -path '*/bundles/*'
            find "$dvf_figs" -name '*.png'
        )
        if [ "$missing" -ne 0 ]; then
            echo "[derived_vs_free] artifact-presence assert FAILED" >&2
            exit 87
        fi
    fi
    echo "[derived_vs_free] leg complete"
}

run_phase_derived_vs_free_wellposed() {
    echo "[phase=derived_vs_free_wellposed]"
    # Well-posed reduced-basis re-run (plan v10 `wellposed-shared-readout`):
    # the IDENTICAL dvf/cms/xm battery with --fit-basis reduced threaded, into
    # FRESH *_wellposed out-roots (crash-fix per-leg out-root rule; parent
    # artifacts never overwritten). Sentinel-INDEPENDENT completion inherited.
    local dvf_root="$EVAL_ROOT/derived_vs_free_wellposed"
    local cms_root="$EVAL_ROOT/context_map_structure_wellposed"
    local xm_root="$EVAL_ROOT/crossmodel_pairs_wellposed"
    local xms_root="$EVAL_ROOT/crossmodel_pairs_wellposed/crossmodel_structure_wellposed"
    local dvf_figs="$FIG_ROOT/derived_vs_free_wellposed"
    local paired_csv="$EVAL_ROOT/analyzer/dvf_wellposed_paired_digest.csv"
    local paired_summary="$EVAL_ROOT/analyzer/dvf_wellposed_paired_summary.json"
    local parent_digest="$EVAL_ROOT/analyzer/dvf_unit_digest.csv"
    local parent_xm_ladder="$EVAL_ROOT/crossmodel_pairs/ladder_crossmodel_L19.json"
    local push_results=1
    if [ -n "$SMOKE" ]; then
        # Smoke outputs NEVER land on committed paths (scratch-dir redirect).
        local scratch="/tmp/issue-1689-wellposed-smoke-leg"
        dvf_root="$scratch/derived_vs_free_wellposed"
        cms_root="$scratch/context_map_structure_wellposed"
        xm_root="$scratch/crossmodel_pairs_wellposed"
        xms_root="$scratch/crossmodel_pairs_wellposed/crossmodel_structure_wellposed"
        dvf_figs="$scratch/figures"
        paired_csv="$scratch/analyzer/dvf_wellposed_paired_digest.csv"
        paired_summary="$scratch/analyzer/dvf_wellposed_paired_summary.json"
        push_results=0
    fi
    mkdir -p "$dvf_root" "$cms_root" "$xm_root" "$xms_root"

    # Within-model merges hard-assert the full 504-unit qualified enumeration
    # in full mode (2 models x 126 ordered pairs x 2 arms); smoke pair subsets
    # carry their own exact counts via the merge missing/dupe gates.
    local expect_units=()
    [ -z "$SMOKE" ] && expect_units=(--expect-units 504)

    # Stage the pinned L19 stores (idempotent; smoke slices the INPUT set to
    # the smoke cells through the SAME stage code path via --stage-cells).
    local dvf_store="${DVF_STORE_ROOT:-$DATA_ROOT/hf_dl/issue1689_speaker_lattice/analysis_tensors}"
    local stage_args=()
    [ -n "$SMOKE" ] && stage_args=(--stage-cells "Qwen_Qwen2.5-7B-Instruct/assistant_chat,Qwen_Qwen2.5-7B-Instruct/assistant_naturalistic,Qwen_Qwen2.5-7B/assistant_chat")
    uv run python scripts/issue1689_derived_vs_free.py --phase stage \
        --store-root "$dvf_store" "${stage_args[@]}"

    # Parent-inputs staging (#734 upload-first; fellows job 15724 crash-fix):
    # the fellows/SLURM rsync lane excludes eval_results/ wholesale
    # (backends/slurm.py RSYNC_INCLUDE_PATHS), so the parent round's committed
    # inputs — the fence digest CSV, the ladder JSONs, and the paired digest's
    # ambient per-unit trees — never reach the node from git. Idempotent: a
    # git checkout (VM smoke / pod) has them all and skips with no Hub call;
    # fail-loud on an incomplete HF mirror (the downstream consumers'
    # exists/WARN guards would otherwise SILENTLY skip the rung conditioning).
    echo "[phase=wp_stage_parent_inputs]"
    uv run python scripts/issue1689_derived_vs_free.py --phase stage-parent-inputs \
        --parent-inputs-root "$EVAL_ROOT"

    # Width derives from DETECTED GPUs (no smoke-conditional narrowing; CVD is
    # pinned per shard in the launcher env below — the #545 clobber rule).
    local ngpu
    ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l) || ngpu=0
    local device="cpu" nsh=1
    if [ "${ngpu:-0}" -ge 1 ]; then device="cuda"; nsh="$ngpu"; fi
    echo "[wellposed] device=$device shards=$nsh"

    # Scale knobs: smoke slices rows/dims/draws through the SAME drivers +
    # pair-subset threading every phase below (PASS_UNIFIED architecture).
    local slice_args=() rot_draws=200 class_draws=40 rank_draws=40
    local within_pairs_args=(--pair-set within-model)
    local xm_pairs="$xm_root/crossmodel_pair_specs.json"
    local pilot_pairs="$dvf_root/pilot_pair.json"
    printf '%s' '[[["Qwen_Qwen2.5-7B-Instruct","assistant_chat"],["Qwen_Qwen2.5-7B-Instruct","assistant_naturalistic"]]]' \
        > "$pilot_pairs"
    if [ -n "$SMOKE" ]; then
        printf '%s' '[[["Qwen_Qwen2.5-7B","assistant_chat"],["Qwen_Qwen2.5-7B-Instruct","assistant_chat"]]]' \
            > "$xm_root/smoke_xpair.json"
        within_pairs_args=(--pairs-file "$pilot_pairs")
        xm_pairs="$xm_root/smoke_xpair.json"
        slice_args=(--row-limit 600 --dim-limit 512)
        rot_draws=5; class_draws=4; rank_draws=2
    else
        uv run python scripts/issue1689_derived_vs_free.py --phase write-pairs \
            --pair-set cross-model --write-pairs-out "$xm_pairs"
    fi

    # --- Gate 1a (plan v10 s7): ambient no-op battery parity vs the PUBLISHED
    # parent per-unit JSON. Full mode gates rc=7; sliced smoke auto-demotes the
    # verdict to informational (gate-calibration parity, #1345) while running
    # the identical computation.
    echo "[phase=wp_gate1]"
    uv run python scripts/issue1689_derived_vs_free.py --phase gate1 \
        --gate1-checks battery --store-root "$dvf_store" --out-root "$dvf_root" \
        --device "$device" "${slice_args[@]}"

    # --- Gate 1b: reduced pilot at the near-cap shape (parity pair, BOTH
    # drivers, --fit-basis reduced; production checkpoints — the full battery
    # resumes past them). Serial (a gate), one process.
    echo "[phase=wp_pilot]"
    local cvd0=(env)
    [ "$device" = "cuda" ] && cvd0=(env CUDA_VISIBLE_DEVICES=0)
    "${cvd0[@]}" timeout --kill-after=60s 7200 uv run python \
        scripts/issue1689_derived_vs_free.py --phase pairs --fit-basis reduced \
        --store-root "$dvf_store" --out-root "$dvf_root" \
        --pairs-file "$pilot_pairs" "${slice_args[@]}" --device "$device"
    "${cvd0[@]}" timeout --kill-after=60s 14400 uv run python \
        scripts/issue1689_context_map_structure.py --phase units --fit-basis reduced \
        --store-root "$dvf_store" --out-root "$cms_root" \
        --pairs-file "$pilot_pairs" "${slice_args[@]}" \
        --class-null-draws "$class_draws" --rank-null-draws "$rank_draws" \
        --device "$device"

    # --- Fence + kill projection (plan s7 kill criterion 2, s9 re-anchor):
    # k-weighted extrapolation from the measured pilot walls; rc=21 = the
    # DESIGNED halt (enforced in full mode only — smoke pilots are sliced, so
    # their projection is informational).
    echo "[phase=wp_fence]"
    local enforce_kill=(--enforce-kill)
    [ -n "$SMOKE" ] && enforce_kill=()
    local fence_out="$dvf_root/fence_stdout.txt"
    uv run python scripts/issue1689_derived_vs_free.py --phase fence \
        --out-root "$dvf_root" --cms-out-root "$cms_root" \
        --digest-csv "$parent_digest" --num-shards "$nsh" \
        "${enforce_kill[@]}" > "$fence_out" 2>&1 || {
        frc=$?
        cat "$fence_out" >&2
        echo "[wellposed] fence phase rc=$frc (21 = plan s7 kill criterion 2: projected total > 30 GPU-h)" >&2
        exit "$frc"
    }
    cat "$fence_out"
    local fence cms_fence
    fence=$(sed -n 's/.*FENCE=\([0-9]*\) CMS_FENCE.*/\1/p' "$fence_out")
    cms_fence=$(sed -n 's/.*CMS_FENCE=\([0-9]*\).*/\1/p' "$fence_out")
    fence=${fence:-7200}; cms_fence=${cms_fence:-21600}
    fence=$((fence < 900 ? 900 : fence)); cms_fence=$((cms_fence < 900 ? 900 : cms_fence))
    echo "[wellposed] fence=${fence}s cms_fence=${cms_fence}s"

    # run_sharded_wp <script> <args...>: one shard per GPU (CVD pinned in the
    # launcher env per shard — the #545 clobber rule), or 1 CPU process.
    run_sharded_wp() {
        local leg_fence="${RUN_FENCE:-$fence}"
        local script="$1"; shift
        if [ "$nsh" -le 1 ]; then
            local cvd_prefix=(env)
            [ "$device" = "cuda" ] && cvd_prefix=(env CUDA_VISIBLE_DEVICES=0)
            timeout --kill-after=60s "$leg_fence" "${cvd_prefix[@]}" \
                uv run python "$script" "$@" --device "$device"
            return $?
        fi
        local pids=() i rc=0 p
        for i in $(seq 0 $((nsh - 1))); do
            env CUDA_VISIBLE_DEVICES="$i" timeout --kill-after=60s "$leg_fence" \
                uv run python "$script" "$@" --device cuda \
                --num-shards "$nsh" --shard-index "$i" \
                > "$LOG_DIR/dvf-wp-shard-$i.log" 2>&1 &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p" || rc=$?; done
        if [ "$rc" -ne 0 ]; then
            echo "[wellposed] shard failure rc=$rc" >&2
            tail -n 60 "$LOG_DIR"/dvf-wp-shard-*.log >&2 || true
            return "$rc"
        fi
    }

    # --- Within-model battery (items 1-6, reduced) ---
    echo "[phase=wp_dvf_pairs]"
    run_sharded_wp scripts/issue1689_derived_vs_free.py --phase pairs --fit-basis reduced \
        --store-root "$dvf_store" --out-root "$dvf_root" \
        "${within_pairs_args[@]}" "${slice_args[@]}"
    echo "[phase=wp_dvf_nulls]"
    "${cvd0[@]}" timeout --kill-after=60s "$fence" uv run python \
        scripts/issue1689_derived_vs_free.py --phase nulls --out-root "$dvf_root" \
        --rotation-draws "$rot_draws" --device "$device"
    uv run python scripts/issue1689_derived_vs_free.py --phase merge --fit-basis reduced \
        --out-root "$dvf_root" "${within_pairs_args[@]}" "${expect_units[@]}"

    # --- Cross-model same-condition dvf pairs (reduced). The ambient xm
    # LADDER is NOT re-run (plan s4: its rung_reached stays the fixed
    # conditioning index — read from the parent's committed JSON below).
    echo "[phase=wp_xm_pairs]"
    run_sharded_wp scripts/issue1689_derived_vs_free.py --phase pairs --fit-basis reduced \
        --store-root "$dvf_store" --out-root "$xm_root" \
        --pairs-file "$xm_pairs" "${slice_args[@]}"
    "${cvd0[@]}" timeout --kill-after=60s "$fence" uv run python \
        scripts/issue1689_derived_vs_free.py --phase nulls --out-root "$xm_root" \
        --rotation-draws "$rot_draws" --device "$device"
    uv run python scripts/issue1689_derived_vs_free.py --phase merge --fit-basis reduced \
        --out-root "$xm_root" --pairs-file "$xm_pairs"

    # --- Context-map structure + rank rung (items 7-8, reduced) ---
    echo "[phase=wp_cms_units]"
    RUN_FENCE="$cms_fence" run_sharded_wp scripts/issue1689_context_map_structure.py \
        --phase units --fit-basis reduced \
        --store-root "$dvf_store" --out-root "$cms_root" \
        "${within_pairs_args[@]}" "${slice_args[@]}" \
        --class-null-draws "$class_draws" --rank-null-draws "$rank_draws"
    uv run python scripts/issue1689_context_map_structure.py --phase overlap \
        --out-root "$cms_root"
    uv run python scripts/issue1689_context_map_structure.py --phase merge --fit-basis reduced \
        --out-root "$cms_root" "${within_pairs_args[@]}" "${expect_units[@]}"
    echo "[phase=wp_xm_structure]"
    RUN_FENCE="$cms_fence" run_sharded_wp scripts/issue1689_context_map_structure.py \
        --phase units --fit-basis reduced \
        --store-root "$dvf_store" --out-root "$xms_root" \
        --pairs-file "$xm_pairs" "${slice_args[@]}" \
        --class-null-draws "$class_draws" --rank-null-draws "$rank_draws" \
        --crossmodel-ladder-json "$parent_xm_ladder"
    uv run python scripts/issue1689_context_map_structure.py --phase overlap \
        --out-root "$xms_root"
    uv run python scripts/issue1689_context_map_structure.py --phase merge --fit-basis reduced \
        --out-root "$xms_root" --pairs-file "$xm_pairs" \
        --crossmodel-ladder-json "$parent_xm_ladder"

    # --- Paired ambient-vs-reduced delta digest (plan s6.5 deliverable) ---
    echo "[phase=wp_paired_digest]"
    mkdir -p "$(dirname "$paired_csv")"
    uv run python scripts/issue1689_dvf_fold_digest.py --paired \
        --reduced-dvf-root "$dvf_root" --reduced-xm-root "$xm_root" \
        --reduced-cms-root "$cms_root" \
        --out "$paired_csv" --summary-out "$paired_summary"

    # --- Figures (reduced-space figs 1-6 + the paired hero/flip/effrank set) ---
    echo "[phase=wp_figures]"
    uv run python scripts/issue1689_derived_vs_free_figures.py \
        --dvf-root "$dvf_root" --cms-root "$cms_root" \
        --crossmodel-root "$xm_root" --out-figs "$dvf_figs" \
        --paired-digest "$paired_csv"

    if [ "$push_results" -eq 1 ]; then
        # --- Compact-bundle upload (ONE upload_folder commit per out-root;
        # cmd_upload prefixes by out_root.name, so the parent's bundles are
        # never clobbered) ---
        echo "[phase=wp_upload]"
        local r
        for r in "$dvf_root" "$cms_root" "$xm_root" "$xms_root"; do
            uv run python scripts/issue1689_derived_vs_free.py --phase upload --out-root "$r"
        done
        # --- Result commit + BARE push, verify-then-assert (#1205/#1325).
        # NO-OP gracefully on a non-git tree (fellows/SLURM rsync lane — the
        # VM-side orchestrator lands git artifacts there).
        echo "[phase=wp_push]"
        if ! git rev-parse --git-dir >/dev/null 2>&1; then
            echo "[wellposed] wp_push SKIP: not a git checkout (rsync lane; VM orchestrator lands git artifacts)"
            echo "[wellposed] leg complete"
            return 0
        fi
        git add "$dvf_root" "$cms_root" "$xm_root" "$dvf_figs" "$paired_csv" "$paired_summary"
        git -c user.name="eps-runner" -c user.email="eps-runner@local" \
            commit -m "task #1689: wellposed-shared-readout round results (reduced-basis battery + structure + crossmodel + paired digest + figures)" \
            || echo "[wellposed] nothing to commit"
        if ! git push origin issue-1689; then
            echo "[wellposed] push failed; retrying once" >&2
            git push origin issue-1689
        fi
        local behind
        behind=$(git rev-list --count origin/issue-1689..HEAD)
        if [ "$behind" -ne 0 ]; then
            echo "[wellposed] push-verify FAILED (${behind} unpushed commits)" >&2
            exit 86
        fi
        # Artifact-presence assert: every declared git-destined result file of
        # THIS round must be in the pushed tree (bundles/*.npz are HF-destined).
        local missing=0 p
        while IFS= read -r p; do
            if [ -z "$(git ls-tree -r origin/issue-1689 --name-only -- "$p")" ]; then
                echo "[wellposed] MISSING from pushed tree: $p" >&2
                missing=1
            fi
        done < <(
            find "$dvf_root" "$cms_root" "$xm_root" -name '*.json' -not -path '*/bundles/*'
            find "$dvf_figs" -name '*.png'
            printf '%s\n' "$paired_csv" "$paired_summary"
        )
        if [ "$missing" -ne 0 ]; then
            echo "[wellposed] artifact-presence assert FAILED" >&2
            exit 87
        fi
    fi
    echo "[wellposed] leg complete"
}

# Dispatch on phase argument.
case "$PHASE" in
    corpus)   run_phase_corpus ;;
    render)   run_phase_render ;;
    haiku_u2) run_phase_haiku_u2 ;;
    onpolicy) run_phase_onpolicy ;;
    capture)  run_phase_capture ;;
    fit_cells|fit) run_phase_fit_cells ;;
    fit_ladder|ladder) run_phase_fit_ladder ;;
    analyze)  run_phase_analyze ;;
    derived_vs_free) run_phase_derived_vs_free ;;
    derived_vs_free_wellposed) run_phase_derived_vs_free_wellposed ;;
    all|--smoke|"")
        run_phase_corpus
        run_phase_render
        run_phase_haiku_u2
        run_phase_onpolicy
        run_phase_capture
        run_phase_fit_cells
        run_phase_fit_ladder
        run_phase_analyze
        ;;
    *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac

# Emit epm:results sentinel per pod-side reporting contract.
SENTINEL="$LOG_DIR/issue-${ISSUE_NUM}-results.json"
cat > "$SENTINEL" <<EOF
{
  "issue": ${ISSUE_NUM},
  "phase_run": "$PHASE",
  "smoke": $([ -n "$SMOKE" ] && echo "true" || echo "false"),
  "eval_dir": "$EVAL_ROOT",
  "fig_dir": "$FIG_ROOT",
  "status": "done"
}
EOF
echo "[phase=done]"
