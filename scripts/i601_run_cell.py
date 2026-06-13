#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #601 — single (cell, seed) worker: build → train → on-policy eval → dense read.

Forked from ``scripts/i472_run_cell.py`` (origin/issue-472). This is the
UNIFIED per-cell unit ``dispatch_neg_setpoint_601.py`` schedules across GPUs;
smoke = the dispatcher launching exactly ONE of these
(``--cells ratio4to1_100p400n --seeds 42 --smoke`` — one FULL cell, no tiny
slice; plan §7 gate 1). Same subprocess shape, env injection, [phase=...] log
surface, sentinel schema, and vLLM-isolation teardown for smoke and sweep.

Per-cell pipeline (all phases inside this one subprocess unit):
  1. build      — per-cell training JSONL via the #472 builder
                  (``pos_ex_override`` + the #601 cell registry; realized
                  negative-panel assert vs EXPECTED_ANCHOR_PANEL).
  2. train      — ``train_one_cell`` with the #601 pass-throughs: explicit
                  band flags (D1: log-only, eval_every=1, local four-float
                  trajectory), ``save_only_model=True``, per-cell lr /
                  epochs / lora_targets, the per-row-type CE probe callback
                  (plan §4 registered data-exhaustion discriminator), and the
                  combined dense-ladder + 6-fraction checkpoint grid
                  (``step_fractions(..., rounding="floor")``).
  3. eval       — NESTED subprocess ``i601_eval_trajectory.py`` (vLLM+HF
                  on-policy four-float reads at the cell's on-policy
                  checkpoint subset; vLLM workers reaped at subprocess exit).
  4. dense read — NESTED subprocess ``i601_dense_read.py`` (teacher-forced
                  four-float reads over the full dense ladder: source +
                  trained negatives + the 8-bystander reference panel).
  5. upload     — ONE bulk Hub commit of the checkpoint tree (fail-loud,
                  verified) + local checkpoint cleanup (upload-before-delete).

#613 (alive-negatives flag A/B) additions, all legacy-preserving: registry
cells with ``suppress_negatives=True`` thread the #474 collator flag
conjunction into ``train_one_cell`` + the ``neg_slot`` rowtype-CE channel;
thin CLI flags ``--hf-prefix`` / ``--run-name-prefix`` / ``--sentinel-task-id``
re-point the HF subfolder, WandB run-name prefix, and sentinel identity
(defaults = the #601 values, so every existing caller is byte-identical).

GPU pinning is the #472 round-3 contract verbatim: the dispatcher passes
``--gpu-id <g>`` (PHYSICAL index) AND exports ``CUDA_VISIBLE_DEVICES=<g>`` in
the launcher env (gotcha: import-time cuInit defeats the in-process clobber);
``train/sft.py`` re-sets CVD to the same value, the nested subprocesses
inherit it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i601.run_cell")


def _write_sentinel(path: Path, *, kind: str, phase: str, note: dict, task_id: int = 601) -> None:
    """Write a poll_pipeline.py-compliant sentinel (sentinel_schema_version=1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": task_id,
                "by": "i601_run_cell",
                "ts": datetime.now(UTC).isoformat(),
                "phase": phase,
                "note": json.dumps(note),
            },
            indent=2,
        )
    )


def _assert_positive_rows_fused_marker(
    train_jsonl: Path,
    tokenizer,
    *,
    marker_text: str,
    marker_id: int,
    cell_slug: str,
) -> dict:
    """FUSED-surface marker-tokenization guard (#613 sep-ablation, plan §3 change 5).

    The collator CLASSIFIES rows by marker presence on the fused chat-template
    ``input_ids`` (``sft.py`` ``_find_marker_positions``) and never asserts
    marker presence on positives — a merged/absent marker silently flips a
    positive row into the NEGATIVE branch (flag-on: loss at the first
    ``<|im_end|>``; flag-off: trailing token) with zero error. Eval/probe
    paths fail loud; training does not. So BEFORE training, for EVERY row the
    builder intended as positive (``marker_text`` in the assistant completion
    string), render the SAME fused surface the trainer uses
    (``apply_chat_template(prompt + completion, tokenize=True,
    add_generation_prompt=False)``) and assert EXACTLY ONE ``marker_id``
    appears in the full sequence, positioned inside the assistant completion
    region (after the last ``<|im_start|>``). Fail-loud RuntimeError naming
    the row. Runs for ALL cells (legacy "\\n\\n" construction included).

    Returns ``{"n_rows_total", "n_positive_checked", "passed": True}``.
    """
    from explore_persona_space.train.sft import _apply_chat_template_safe

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    n_total = 0
    n_pos = 0
    with Path(train_jsonl).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_total += 1
            completion = row.get("completion") or []
            content = " ".join(m.get("content", "") for m in completion)
            if marker_text not in content:
                continue  # negative row (builder hard-asserts no contamination)
            n_pos += 1
            full_ids = _apply_chat_template_safe(
                tokenizer, list(row["prompt"]) + list(completion), add_generation_prompt=False
            )
            if full_ids is None:
                raise RuntimeError(
                    f"[{cell_slug}] fused-surface marker assert: chat-template render FAILED "
                    f"for positive row {line_no} of {train_jsonl} — cannot verify the "
                    f"collator-visible marker."
                )
            count = full_ids.count(marker_id)
            if count != 1:
                raise RuntimeError(
                    f"[{cell_slug}] fused-surface marker assert FAILED at positive row "
                    f"{line_no} of {train_jsonl}: marker id {marker_id} appears {count}x in "
                    f"the fused chat-template ids (expected exactly 1) — the marker "
                    f"BPE-merged/vanished on the trainer's surface, so the collator would "
                    f"silently flip this row into the NEGATIVE branch. Completion tail: "
                    f"{content[-60:]!r}."
                )
            if im_start_id is not None and im_start_id >= 0:
                last_start = max(
                    (i for i, t in enumerate(full_ids) if t == im_start_id), default=-1
                )
                if full_ids.index(marker_id) <= last_start:
                    raise RuntimeError(
                        f"[{cell_slug}] fused-surface marker assert FAILED at positive row "
                        f"{line_no} of {train_jsonl}: marker id {marker_id} sits BEFORE the "
                        f"assistant completion region (last <|im_start|> at {last_start}) — "
                        f"the loss-bearing marker is not in the completion."
                    )
    log.info(
        "[phase=fused_marker_assert_%s] PASS: %d/%d positive rows carry exactly one fused "
        "marker id %d in the completion region (%d rows total)",
        cell_slug,
        n_pos,
        n_pos,
        marker_id,
        n_total,
    )
    return {"n_rows_total": n_total, "n_positive_checked": n_pos, "passed": True}


def _combined_fractions(spec, step_fractions_fn) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(all checkpoint fractions, on-policy subset fractions) for one cell.

    Dense-ladder steps are converted via ``step_fractions(rounding="floor",
    precision=4)`` — floor mode lands each checkpoint at EXACTLY the target
    optimizer step (banker's rounding can round a fraction up past the true
    ratio and shift the save one step late, which breaks a 1-step ladder).
    The 6-fraction on-policy grid (full6 cells) / the step-10 + terminal
    anchors (anchor cells) ride the same checkpoint callback.
    """
    t = spec.expected_steps
    dense_fracs = (
        step_fractions_fn(tuple(spec.dense_steps), t, precision=4, rounding="floor")
        if spec.dense_steps
        else ()
    )
    if spec.onpolicy == "full6":
        from explore_persona_space.experiments.neg_setpoint_601 import ONPOLICY_FULL6_FRACTIONS

        onpolicy = tuple(round(f, 4) for f in ONPOLICY_FULL6_FRACTIONS)
    elif spec.onpolicy == "anchors":
        anchor_steps = tuple(s for s in spec.onpolicy_anchor_steps if s < t)
        onpolicy = (
            *step_fractions_fn(anchor_steps, t, precision=4, rounding="floor"),
            1.0,
        )
    else:
        raise ValueError(f"unknown onpolicy mode {spec.onpolicy!r}")
    all_fracs = tuple(sorted({*dense_fracs, *onpolicy}))
    return all_fracs, tuple(sorted(set(onpolicy)))


def _verify_terminal_adapter_uploaded(
    *, repo_id: str, terminal_prefix: str, adapter_dir: Path
) -> None:
    """Fail-loud guard: the TERMINAL adapter must resolve on the Hub before we proceed.

    ``train_lora``'s own terminal-adapter HF upload is best-effort (``sft.py``
    warns "Adapter upload failed — local copy preserved" on a falsy hub path AND
    on any exception), but downstream slot-read phases (e.g. ``i613_launch.sh``
    p6) fetch the terminal BACK from
    ``{terminal_prefix}/{adapter_config.json,adapter_model.safetensors}`` — and
    on an ephemeral lane (GCP EXIT-trap teardown) a silently-missing terminal
    upload is permanent adapter loss after both GPU seeds complete (#613 round-2
    blocker ``flagon-terminal-upload-not-fail-loud``; real precedent for the
    silent failure: HF LFS quota-403, #552/#541). Verifies via
    ``huggingface_hub.list_repo_files`` (Python Hub API — never the ``hf`` CLI,
    which false-"0"s; the listing does NOT truncate on large repos, unlike
    ``repo_info.siblings``), re-uploads from the surviving local ``adapter_dir``
    when missing, and raises if the terminal still does not resolve.
    """
    from huggingface_hub import list_repo_files

    required = {
        f"{terminal_prefix}/adapter_config.json",
        f"{terminal_prefix}/adapter_model.safetensors",
    }

    def _missing() -> list[str]:
        return sorted(required - set(list_repo_files(repo_id, repo_type="model")))

    missing = _missing()
    if missing:
        log.warning(
            "terminal adapter MISSING on %s (%s) — train_lora's best-effort upload silently "
            "failed; re-uploading fail-loud from %s",
            repo_id,
            missing,
            adapter_dir,
        )
        from explore_persona_space.orchestrate.hub import upload_model

        hub_path = upload_model(
            model_path=str(adapter_dir),
            repo_id=repo_id,
            path_in_repo=terminal_prefix,
            delete_after=False,
        )
        if not hub_path:
            raise RuntimeError(
                f"terminal-adapter re-upload to {repo_id}/{terminal_prefix} returned an empty "
                f"hub path — refusing to proceed (local copy preserved at {adapter_dir})."
            )
        missing = _missing()
    if missing:
        raise RuntimeError(
            f"terminal adapter STILL missing on {repo_id} after re-upload: {missing} — "
            f"downstream slot reads fetch {terminal_prefix}/ from the Hub; on an ephemeral "
            f"lane this is permanent adapter loss (local copy preserved at {adapter_dir})."
        )
    log.info("terminal adapter verified on Hub: %s/%s", repo_id, terminal_prefix)


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- linear per-cell pipeline (build -> train -> eval -> dense read -> upload); the fail-loud asserts add branches, not nesting
    ap = argparse.ArgumentParser(
        description="Task #601 single (cell, seed) worker (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_601"))
    ap.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_601"))
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_601"))
    ap.add_argument("--no-kl", action="store_true", help="Skip DV-B KL (debug only).")
    ap.add_argument("--report-to", default="wandb")
    ap.add_argument(
        "--skip-checkpoint-upload",
        action="store_true",
        help="Local debug only: skip the bulk Hub upload + local checkpoint cleanup.",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="ASSIGNED physical GPU index (threaded to train_one_cell; sft.py sets CVD).",
    )
    # ── #613 thin flags (legacy-preserving defaults — plan #613 §4 step 2). ──
    ap.add_argument(
        "--hf-prefix",
        default=None,
        help="HF adapter path prefix (default: HF_ADAPTER_PREFIX_601; #613 passes "
        "adapters/issue_613).",
    )
    ap.add_argument(
        "--run-name-prefix",
        default="issue601",
        help="WandB run-name prefix (default issue601; #613 passes issue613).",
    )
    ap.add_argument(
        "--sentinel-task-id",
        type=int,
        default=601,
        help="Task id for the sentinel filename + task_id field (default 601; #613 passes 613).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format=(
            f"%(asctime)s [phase=cell_{args.cell}_seed{args.seed}] "
            f"%(name)s %(levelname)s | %(message)s"
        ),
        stream=sys.stdout,
    )
    log.info(
        "Assigned physical GPU --gpu-id=%d; inherited CUDA_VISIBLE_DEVICES=%s",
        args.gpu_id,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        HEADLINE_LAYER,
        MAX_LENGTH,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.build_training_data import (
        build_cell,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
        load_r_artifact,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        negatives_for_cell,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
        step_fractions,
        train_one_cell,
    )
    from explore_persona_space.experiments.neg_setpoint_601 import (
        CELL_SPECS_601_472SHAPE,
        EXPECTED_ANCHOR_PANEL,
        EXPECTED_MARKER_TOKEN_ID,
        EXPECTED_POST_R_EOS_ID,
        HF_ADAPTER_PREFIX_601,
        HF_MODEL_REPO,
        MARKER_SEP,
        MARKER_TEXT,
        SOURCE_PERSONA,
        cell_by_slug,
    )
    from explore_persona_space.experiments.neg_setpoint_601.rowtype_ce_probe import (
        RowTypeCETrainProbeCallback,
        build_rowtype_probes,
    )

    spec = cell_by_slug(args.cell)
    hf_prefix = args.hf_prefix if args.hf_prefix is not None else HF_ADAPTER_PREFIX_601
    # ── #613 sep-ablation: spec.marker_sep -> the nested read subprocesses'
    # --sep-mode vocabulary (plan §3 change 3). Legacy cells (marker_sep ==
    # MARKER_SEP) get NO flag appended — byte-identical argvs; sep cells get
    # "plain"; any other separator has no CLI vocabulary — fail loud here
    # rather than silently reading the wrong slot.
    if spec.marker_sep == MARKER_SEP:
        sep_mode: str | None = None
    elif spec.marker_sep == "":
        sep_mode = "plain"
    else:
        raise ValueError(
            f"[{args.cell}] marker_sep={spec.marker_sep!r} has no --sep-mode mapping "
            f"(known: {MARKER_SEP!r} -> default, '' -> plain); the nested eval/dense "
            f"reads would score the WRONG slot."
        )
    if args.seed not in (42, 137):
        raise ValueError(f"[{args.cell}] seed {args.seed} not a canonical #601 seed (42/137).")
    if args.seed not in spec.seeds:
        # Permitted only via the dispatcher's --anchor-retrain-fallback path
        # (plan §4 Phase-0 item 3: dense_200p800n seed 42 replaces the unfit
        # parent anchor). Loud so an accidental off-registry launch is visible.
        log.warning(
            "[%s] seed %d is OFF the cell's registered seed set %s — anchor-retrain "
            "fallback expected; verify phase0_gate.json anchor_reuse_ok=false.",
            args.cell,
            args.seed,
            spec.seeds,
        )

    run_dir = args.runs_root / f"{args.cell}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = run_dir / "train_pool.jsonl"
    final_adapter_dir = run_dir / "adapter"
    ckpt_root = run_dir / "checkpoints"
    cell_out_dir = args.slab_root / spec.phase / f"{args.cell}_seed{args.seed}"
    cell_out_dir.mkdir(parents=True, exist_ok=True)
    out_traj = cell_out_dir / "trajectory.json"
    out_dense = cell_out_dir / "dense_trajectory.json"
    band_traj_path = cell_out_dir / "inloop_band_trajectory.json"
    rowtype_ce_path = cell_out_dir / "rowtype_ce.json"
    raw_completions_path = cell_out_dir / "raw_completions.json"
    sentinel = (
        args.log_dir / f"issue-{args.sentinel_task_id}-{args.cell}-seed{args.seed}-results.json"
    )

    # ── In-process marker assert (HARD requirement; incident #537). ──────────
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.neg_setpoint_601 import BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if marker_ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"IN-PROCESS marker assert FAILED: encode({MARKER_TEXT!r})={marker_ids}, "
            f"expected [{EXPECTED_MARKER_TOKEN_ID}]. Refusing to train a no-op implant."
        )
    log.info("[phase=marker_assert] PASS: %r -> %s", MARKER_TEXT, marker_ids)

    bank = load_persona_bank(args.data_dir / "persona_bank.json")
    r_train = load_r_artifact(args.data_dir / "on_policy_R" / "R_train.json")
    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, args.data_dir)
    q_train, _q_eval = get_train_eval_questions()

    # ── Phase: build training data (CPU). ────────────────────────────────────
    log.info(
        "[phase=build_%s] %d pos + %dx%d neg rows (T expected %d)",
        args.cell,
        spec.pos_ex,
        spec.n_neg_personas,
        spec.neg_ex_per_persona,
        spec.expected_steps,
    )
    # Realized-panel assert: every 4-negative #601 cell must reproduce the
    # parent anchor panel (same spread-4 selector over the pinned centroids).
    if spec.n_neg_personas > 0:
        realized = negatives_for_cell(
            args.cell, cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_601_472SHAPE
        )
        if set(realized) != set(EXPECTED_ANCHOR_PANEL):
            raise AssertionError(
                f"[{args.cell}] realized negative panel {sorted(realized)} != expected anchor "
                f"panel {sorted(EXPECTED_ANCHOR_PANEL)} — centroid/selector drift; the cell "
                f"would not be single-variable vs #472."
            )
    build_cell(
        args.cell,
        train_jsonl,
        r_train=r_train,
        cos_to_source=cts,
        q_train=q_train,
        persona_bank=bank,
        source=SOURCE_PERSONA,
        seed=args.seed,
        # #613 sep-ablation (plan §3 change 2): the positive-row separator from
        # the cell spec (legacy cells pass MARKER_SEP — byte-identical rows).
        marker_sep=spec.marker_sep,
        cell_specs=CELL_SPECS_601_472SHAPE,
        pos_ex_override=spec.pos_ex,
    )

    # ── Phase: FUSED-surface marker assert (pre-train, every positive row;
    # plan §3 change 5 — closes the collator's silent positive->negative flip).
    fused_assert = _assert_positive_rows_fused_marker(
        train_jsonl,
        tokenizer,
        marker_text=MARKER_TEXT,
        marker_id=EXPECTED_MARKER_TOKEN_ID,
        cell_slug=args.cell,
    )
    if spec.pos_ex > 0 and fused_assert["n_positive_checked"] != spec.pos_ex:
        raise RuntimeError(
            f"[{args.cell}] fused-surface assert checked {fused_assert['n_positive_checked']} "
            f"positive rows but the spec registers pos_ex={spec.pos_ex} — builder/guard "
            f"row-intent drift."
        )
    # Unit manifest (durable, spec-conditional runtime echo — the smoke gate
    # reads it; reproducibility metadata per CLAUDE.md).
    build_manifest_path = cell_out_dir / "build_manifest.json"
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        _git_sha,
    )

    build_manifest_path.write_text(
        json.dumps(
            {
                "cell": args.cell,
                "seed": args.seed,
                "marker_sep": spec.marker_sep,
                "sep_mode": sep_mode or "marker",
                "suppress_negatives": spec.suppress_negatives,
                "fused_marker_assert": fused_assert,
                "git_commit": _git_sha(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    log.info("[phase=build_%s] unit manifest -> %s", args.cell, build_manifest_path)

    # ── Phase: train (HF Trainer, in-process). ───────────────────────────────
    all_fracs, onpolicy_fracs = _combined_fractions(spec, step_fractions)
    log.info(
        "[phase=train_%s] T=%d, %d ckpt fractions (%d on-policy), lr=%g, epochs=%d, "
        "targets=%s, band stop=%s log_only=%s, suppress_negatives=%s, marker_sep=%r",
        args.cell,
        spec.expected_steps,
        len(all_fracs),
        len(onpolicy_fracs),
        spec.lr,
        spec.epochs,
        spec.lora_targets or "all-linear",
        spec.band_stop,
        spec.band_log_only,
        spec.suppress_negatives,
        spec.marker_sep,
    )
    # max_length = the TRAINING max_length (1024): the probe must monitor
    # exactly the rows the trainer sees — a 2048 probe cap would admit rows the
    # trainer truncates away (round-1 Codex review minor).
    probes = build_rowtype_probes(
        train_jsonl,
        tokenizer,
        marker_ids,
        max_length=MAX_LENGTH,
        # #613: third channel at the flag-on loss slot (R1 manipulation check).
        neg_post_response_slot=spec.suppress_negatives,
        im_end_token_id=EXPECTED_POST_R_EOS_ID,
    )
    ce_probe = RowTypeCETrainProbeCallback(probes, out_path=rowtype_ce_path, eval_every_steps=1)
    train_result = train_one_cell(
        cell_slug=args.cell,
        seed=args.seed,
        train_jsonl=train_jsonl,
        output_dir=final_adapter_dir,
        ckpt_root=ckpt_root,
        report_to=args.report_to,
        gpu_id=args.gpu_id,
        lr_override=spec.lr,
        epochs_override=spec.epochs,
        hf_path_in_repo_override=f"{hf_prefix}/{args.cell}_seed{args.seed}",
        run_name_override=f"{args.run_name_prefix}_{args.cell}_seed{args.seed}",
        step_calibration_fractions=all_fracs,
        frac_precision=4,
        lora_targets_override=list(spec.lora_targets) if spec.lora_targets else None,
        # #613 (THE manipulated variable): negative-row loss at the first
        # post-response <|im_end|> (collator #474 branch) when the registry
        # cell sets suppress_negatives=True; default False = flag-off parity.
        marker_suppress_at_post_response_slot=spec.suppress_negatives,
        marker_im_end_token_id=(EXPECTED_POST_R_EOS_ID if spec.suppress_negatives else None),
        # D1: explicit band wiring — the train_lora DEFAULT would LIVE-stop any
        # in-band cell at step >= 20 (plan §8 risk 1). Phase 3 has no marker
        # rows → band off entirely.
        marker_band_stop=spec.band_stop,
        marker_band_log_only=spec.band_log_only,
        marker_band_eval_every_steps=1,
        marker_band_trajectory_path=str(band_traj_path) if spec.band_stop else None,
        save_only_model=True,
        extra_callbacks=[ce_probe],
    )
    ckpt_index_path = run_dir / "checkpoint_index.json"
    ckpt_index_path.write_text(json.dumps(train_result["checkpoint_index"], indent=2))
    realized_terminal_step = train_result["checkpoint_index"].get("1.0000", {}).get("step")
    log.info(
        "[phase=train_%s] done; %d checkpoints, terminal step=%s (expected T=%d)",
        args.cell,
        len(train_result["checkpoint_index"]),
        realized_terminal_step,
        spec.expected_steps,
    )
    # Band-stop misfire catch (smoke assert §4; also enforced per-cell here so
    # a sweep cell that early-stopped fails LOUD, not at analysis time). A
    # missing terminal index entry would silently disable the check — raise
    # (round-1 review minor).
    if realized_terminal_step is None:
        raise RuntimeError(
            f"[{args.cell}] checkpoint_index has no '1.0000' terminal step — cannot verify "
            f"the full schedule ran (the band-stop misfire check requires it)."
        )
    if int(realized_terminal_step) != spec.expected_steps:
        raise RuntimeError(
            f"[{args.cell}] realized terminal step {realized_terminal_step} != expected "
            f"T={spec.expected_steps} — a band-stop (or schedule mis-wire) truncated the "
            f"free-running schedule (plan §8 risk 1)."
        )

    # ── Phase: terminal-adapter upload verify (fail-loud; #613 round-2 blocker
    # flagon-terminal-upload-not-fail-loud). The checkpoints/ subtree upload at
    # the end of this unit is already fail-loud; the TOP-LEVEL terminal adapter
    # upload happens inside train_one_cell via train_lora's warn-and-continue
    # path, so verify it landed and re-upload from the surviving local adapter
    # dir if not. Runs for legacy #601 cells too — a pure safety net (one extra
    # Hub read in the success path; no artifact / layout / training change).
    if args.skip_checkpoint_upload:
        log.info("[phase=terminal_verify_%s] SKIP (--skip-checkpoint-upload)", args.cell)
    else:
        log.info(
            "[phase=terminal_verify_%s] verifying terminal adapter %s/%s_seed%d on Hub",
            args.cell,
            hf_prefix,
            args.cell,
            args.seed,
        )
        _verify_terminal_adapter_uploaded(
            repo_id=HF_MODEL_REPO,
            terminal_prefix=f"{hf_prefix}/{args.cell}_seed{args.seed}",
            adapter_dir=final_adapter_dir,
        )

    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase: on-policy eval (NESTED subprocess: vLLM teardown isolation). ──
    eval_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if eval_cvd != str(args.gpu_id):
        raise RuntimeError(
            f"Pre-eval CUDA_VISIBLE_DEVICES={eval_cvd!r} != assigned --gpu-id={args.gpu_id}."
        )
    onpolicy_keys = ",".join(f"{f:.4f}" for f in onpolicy_fracs)
    eval_cmd = [
        "uv",
        "run",
        "python",
        "scripts/i601_eval_trajectory.py",
        "--cell",
        args.cell,
        "--seed",
        str(args.seed),
        "--checkpoint-index",
        str(ckpt_index_path),
        "--out-path",
        str(out_traj),
        "--raw-completions-path",
        str(raw_completions_path),
        "--data-dir",
        str(args.data_dir),
        "--fracs",
        onpolicy_keys,
        "--panel",
        "full" if spec.onpolicy == "full6" else "bystander8",
        "--bystander-panel-path",
        str(args.slab_root / "phase0" / "bystander_panel.json"),
    ]
    if args.no_kl:
        eval_cmd.append("--no-kl")
    if sep_mode is not None:
        # #613 sep-ablation: every read of a sep="" cell happens at the
        # construction's own slot (plan §3 change 3); legacy argv untouched.
        eval_cmd += ["--sep-mode", sep_mode]
    log.info("[phase=eval_%s] nested eval subprocess: %s", args.cell, " ".join(eval_cmd))
    subprocess.run(eval_cmd, env={**os.environ}, check=True)
    if not out_traj.exists():
        raise RuntimeError(f"[{args.cell}] eval subprocess exited 0 but {out_traj} missing.")

    # ── Phase: dense teacher-forced read (NESTED subprocess, HF-only). ───────
    if spec.dense_steps:
        dense_cmd = [
            "uv",
            "run",
            "python",
            "scripts/i601_dense_read.py",
            "--cell",
            args.cell,
            "--seed",
            str(args.seed),
            "--checkpoint-index",
            str(ckpt_index_path),
            "--out-path",
            str(out_dense),
            "--data-dir",
            str(args.data_dir),
            "--bystander-panel-path",
            str(args.slab_root / "phase0" / "bystander_panel.json"),
        ]
        if sep_mode is not None:
            dense_cmd += ["--sep-mode", sep_mode]
        log.info("[phase=dense_%s] nested dense-read subprocess", args.cell)
        subprocess.run(dense_cmd, env={**os.environ}, check=True)
        if not out_dense.exists():
            raise RuntimeError(f"[{args.cell}] dense read exited 0 but {out_dense} missing.")
    else:
        log.info("[phase=dense_%s] SKIP (no dense ladder for this cell)", args.cell)

    # ── Phase: bulk checkpoint upload (ONE Hub commit) + local cleanup. ──────
    if args.skip_checkpoint_upload:
        log.info("[phase=upload_%s] SKIP (--skip-checkpoint-upload)", args.cell)
    else:
        from explore_persona_space.orchestrate.hub import upload_model

        dest = f"{hf_prefix}/{args.cell}_seed{args.seed}/checkpoints"
        log.info("[phase=upload_%s] bulk checkpoint upload → %s", args.cell, dest)
        hub_path = upload_model(model_path=str(ckpt_root), path_in_repo=dest, delete_after=False)
        if not hub_path:
            raise RuntimeError(
                f"[{args.cell}] checkpoint-tree upload FAILED (empty hub path) — refusing to "
                f"delete local checkpoints (upload-before-delete invariant)."
            )
        import shutil

        shutil.rmtree(ckpt_root)
        log.info("[phase=upload_%s] verified upload; local checkpoints reaped", args.cell)

    _write_sentinel(
        sentinel,
        kind="epm:progress",
        phase=f"cell_done_{args.cell}_seed{args.seed}",
        task_id=args.sentinel_task_id,
        note={
            "cell": args.cell,
            "seed": args.seed,
            "phase_dir": spec.phase,
            "suppress_negatives": spec.suppress_negatives,
            "marker_sep": spec.marker_sep,
            "sep_mode": sep_mode or "marker",
            "build_manifest_path": str(build_manifest_path),
            "fused_marker_assert": fused_assert,
            "expected_steps": spec.expected_steps,
            "realized_terminal_step": realized_terminal_step,
            "trajectory_path": str(out_traj),
            "dense_trajectory_path": str(out_dense) if spec.dense_steps else None,
            "rowtype_ce_path": str(rowtype_ce_path),
            "inloop_band_trajectory_path": str(band_traj_path) if spec.band_stop else None,
            "adapter_hf_path": f"{hf_prefix}/{args.cell}_seed{args.seed}",
        },
    )
    log.info("cell complete; wrote sentinel → %s", sentinel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
