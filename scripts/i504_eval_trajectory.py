# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG + × + − intentional
#!/usr/bin/env python3
"""Task #504 — per-cell on-policy trajectory eval (nested subprocess).

Forked from scripts/i472_eval_trajectory.py. #504-specific changes:

  * Held-out panel comes from the Phase 0.5 ``--panel-json`` file (= bank −
    {source, default, 4 positioned-N's} ≈ 55 probes) — NOT the #472
    ``held_out_panel(cos_to_source, ...)`` band-based panel.
  * Disjointness guard: assert the panel does NOT overlap with this cell's
    negatives (default + the cell's positioned-N) — the panel must be a
    held-OUT set the cell never trained against, or bystander ΔG reflects
    training-suppression and not leakage (the #477 round-3 bug class).
  * Same on-policy DV-A (vLLM logp) + DV-B (HF full-vocab KL) rig from #472,
    same ``assert_adapter_actually_applied`` guard at each checkpoint.

Usage (driven by the dispatcher / scripts/i504_run_cell.py):
    uv run python scripts/i504_eval_trajectory.py \
        --cell c504_near --seed 42 \
        --checkpoint-index /workspace/runs/issue_504/c504_near_seed42/checkpoint_index.json \
        --out-path eval_results/issue_504/c504_near_seed42/trajectory.json \
        --bank-path data/issue_472/persona_bank.json \
        --r-eval-path data/issue_472/on_policy_R/R_eval.json \
        --panel-json /tmp/i504-arm-to-n.json \
        --max-lora-rank 8 --max-new-tokens 2048

The ``--max-lora-rank`` argument is the LoRA rank that training pinned (the
adapter's actual rank), threaded through from the dispatcher's ``chosen_rank``.
vLLM's ``LLM(max_lora_rank=...)`` is a buffer size and must be one of
(8, 16, 32, 64, 128, 256, 320, 512), so this script floors it to 8 inside
``main`` before constructing the engine — the r=4 adapter fits inside an r=8
buffer (vLLM zero-pads unused rows). Training is untouched at the honest rank.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i504.eval_trajectory")


def compute_cell_negatives_for_disjoint_guard(
    cell: str,
    arm_to_positioned_n: dict[str, str],
    smoke_mid_band_n: str | None,
    default_persona: str,
) -> set[str]:
    """Compute the personas the cell trained against, for the disjointness guard.

    Returns the set of negative personas — ``{default_persona, ...}`` — that
    the held-out probe panel MUST NOT intersect with. Used by ``main`` below
    AND by the regression test (``tests/experiments/test_504_eval_traj_v3_disjoint.py``)
    to pin the v1+v2+v3 smoke-prefix coverage in isolation, without loading
    vLLM / bank / R_eval.

    Args:
        cell: the cell slug (e.g. ``c504v3_smoke_eps2``).
        arm_to_positioned_n: Phase 0.5 map from positioned-arm slug → its
            positioned negative persona name.
        smoke_mid_band_n: the mid-band negative persona name, picked by Phase
            0 for smoke cells (None for non-smoke cells / when missing — the
            caller surfaces "missing" elsewhere).
        default_persona: the always-included default negative persona.

    Returns:
        Set of negative-persona names. Always contains ``default_persona``.
        For a positioned arm, also contains ``arm_to_positioned_n[cell]``.
        For a v1/v2/v3 smoke cell with non-None ``smoke_mid_band_n``, also
        contains the smoke mid-band negative.

    Raises:
        Nothing — caller decides whether the resulting set's overlap with the
        panel is fatal.
    """
    negs: set[str] = {default_persona}
    if cell in arm_to_positioned_n:
        negs.add(arm_to_positioned_n[cell])
    # Round-2 fix (Concern B): include v2 smoke prefix for parity with v1.
    # Round-7 fix: include v3 smoke prefix (c504v3_smoke_eps{2,3}) — without
    # the v3 widening the startswith returns False on v3 smoke cells, the
    # disjointness guard silently no-ops, and the held-out panel may include
    # smoke_mid_band_n (= the persona the cell trained against), corrupting
    # bystander ΔG. SAME class as the round-6 cell_resolution.py:183 +
    # i504_run_cell.py:273 widening.
    if (
        cell.startswith(("c504_smoke_", "c504v2_smoke_", "c504v3_smoke_"))
        and smoke_mid_band_n is not None
    ):
        negs.add(smoke_mid_band_n)
    return negs


def build_source_guard_meta(fraction_manifest: Path | None) -> dict | None:
    """Build ``run_trajectory_eval``'s ``source_guard_meta`` from a selector manifest.

    Returns None when ``fraction_manifest`` is None (legacy #504/#530 behavior
    — no guard). Otherwise reads ``source_delta_g_at_selected_steps`` (the
    selector's teacher-forced source ΔG per selected fraction) + ``stopped``
    (band-stop fired) and returns the meta dict the rig's per-checkpoint
    adapter-applied cross-check consumes (#534 round-2).

    Raises:
        FileNotFoundError / json.JSONDecodeError: unreadable manifest — the
            guard the caller asked for cannot be silently skipped.
    """
    if fraction_manifest is None:
        return None
    manifest = json.loads(fraction_manifest.read_text())
    src_dg = manifest.get("source_delta_g_at_selected_steps") or {}
    expected_by_frac = {float(k): (float(v) if v is not None else None) for k, v in src_dg.items()}
    meta = {
        "expected_by_frac": expected_by_frac,
        "band_stop_fired": bool(manifest.get("stopped", False)),
        "manifest_path": str(fraction_manifest),
    }
    if not any(v is not None for v in expected_by_frac.values()):
        log.warning(
            "[manifest-guard] %s carries no source ΔG expectations (selector ran "
            "--skip-source-trajectory?) — only the band-stop final-fraction floor "
            "clause can fire.",
            fraction_manifest,
        )
    log.info(
        "[manifest-guard] armed: expected_by_frac=%s, band_stop_fired=%s",
        expected_by_frac,
        meta["band_stop_fired"],
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint-index", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument(
        "--r-eval-path", type=Path, default=Path("data/issue_472/on_policy_R/R_eval.json")
    )
    ap.add_argument(
        "--panel-json",
        type=Path,
        required=True,
        help=(
            "JSON file from Phase 0.5 with keys 'held_out_panel' (list of probe "
            "persona names) and 'arm_to_positioned_n' (for the disjointness "
            "guard) and 'chosen_negatives' (the default-persona name)."
        ),
    )
    ap.add_argument(
        "--max-lora-rank",
        type=int,
        default=8,
        help=(
            "LoRA rank pinned by Phase 0 — the adapter's actual rank. "
            "vLLM's LLM(max_lora_rank=...) is floored to max(8, this) before "
            "the engine is constructed (vLLM rejects ranks < 8; it is a buffer "
            "size, not the adapter rank)."
        ),
    )
    ap.add_argument(
        "--fraction-manifest",
        type=Path,
        default=None,
        help=(
            "#534 round-2: optional fraction_manifest.json from "
            "i534_select_fractions.py. When given, the rig cross-checks its own "
            "on-policy source-self ΔG per checkpoint against the selector's "
            "teacher-forced source_delta_g_at_selected_steps — >2-nat "
            "disagreement at the FINAL fraction fails loud "
            "(SourceDeltaGManifestMismatchError; the round-1 adapter-not-"
            "applied guard). Absent = exact legacy #504/#530 behavior."
        ),
    )
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument(
        "--gpu-memory-utilization", type=float, default=0.60, help="vLLM gpu_memory_utilization."
    )
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--no-kl", action="store_true", help="Skip DV-B KL (smoke speed-up).")
    ap.add_argument("--sentinel-path", type=Path, default=None)
    ap.add_argument(
        "--source",
        default=None,
        help=(
            "Round-2 fix (BLOCKER #2, concern_id `fallback-source-threading`): "
            "source persona name. The v2 Phase 0 fallback path (plan v2 §4.2) "
            "swaps villain for an easier source; this rig MUST evaluate against "
            "the SAME source the cell was trained on or ΔG/emission diagnostics "
            "become meaningless. When unset, falls back to the v1/v2 module "
            "default SOURCE_PERSONA = villain (legacy byte-identical)."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=eval_trajectory] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        SOURCE_PERSONA,
        TRAJECTORY_CHECKPOINT_FRACTIONS,
    )

    # Round-2 fix (BLOCKER #2): resolve effective source persona BEFORE the
    # R_eval coverage check below, so the assertion runs against the SAME
    # persona the rig will subsequently score the trajectory against.
    effective_source = args.source if args.source is not None else SOURCE_PERSONA
    log.info(
        "[phase=source] effective source persona = %r (CLI --source=%r, default=%r)",
        effective_source,
        args.source,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        run_trajectory_eval,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
        load_r_artifact,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        ALWAYS_INCLUDE_NEGATIVE,
        DEFAULT_ARM_SLUG,
    )

    # ── Load Phase 0.5 outputs (panel + arm → positioned-N + smoke-mid-band-N). ─
    panel_payload = json.loads(args.panel_json.read_text())
    held_out_panel = panel_payload.get("held_out_panel", [])
    if not held_out_panel:
        raise RuntimeError(
            f"--panel-json {args.panel_json} has empty 'held_out_panel' — Phase 0.5 must "
            "populate it before this rig runs (= bank − {source, default, 4 positioned-N's})."
        )
    arm_to_positioned_n = panel_payload.get("arm_to_positioned_n", {})
    smoke_mid_band_n = panel_payload.get("smoke_mid_band_n")
    default_persona = panel_payload.get("chosen_negatives", {}).get(
        "default", ALWAYS_INCLUDE_NEGATIVE
    )

    # Disjointness guard: panel must NOT intersect this cell's negatives.
    # See ``compute_cell_negatives_for_disjoint_guard`` above for the v1/v2/v3
    # smoke-prefix coverage + the round-2/round-7 incident references.
    cell_negs = compute_cell_negatives_for_disjoint_guard(
        cell=args.cell,
        arm_to_positioned_n=arm_to_positioned_n,
        smoke_mid_band_n=smoke_mid_band_n,
        default_persona=default_persona,
    )
    # default_only arm: only the default is a negative (no positioned-N).
    overlap = set(held_out_panel) & cell_negs
    if overlap:
        raise AssertionError(
            f"panel∩negatives for cell={args.cell!r}: {sorted(overlap)} — the panel is "
            "contaminated by this cell's contrastive negatives (bystander ΔG would reflect "
            "training-against, not leakage). Investigate the Phase 0.5 panel construction "
            "before re-running."
        )
    # Sanity: default-only arm has no positioned negative; arm_to_positioned_n
    # should NOT carry an entry for c504_default_only.
    if args.cell == DEFAULT_ARM_SLUG and args.cell in arm_to_positioned_n:
        log.warning(
            "[disjoint] %s carries an entry in arm_to_positioned_n — unexpected for the "
            "default-only arm (the dispatcher should leave it absent).",
            args.cell,
        )
    log.info(
        "[disjoint] cell=%s, negs=%s, panel_size=%d — guard PASS.",
        args.cell,
        sorted(cell_negs),
        len(held_out_panel),
    )

    bank = load_persona_bank(args.bank_path)
    # Sanity: every panel persona must be in the bank.
    for p in held_out_panel:
        if p not in bank:
            raise KeyError(
                f"Panel persona {p!r} missing from bank at {args.bank_path}; "
                "Phase 0.5 + Phase 1 must read the SAME bank artifact."
            )
    eval_personas = {p: bank[p] for p in held_out_panel}

    # Q_eval split (must match #472 r_generate's split).
    _q_train, q_eval = get_train_eval_questions()
    # Sanity: R_eval covers the panel + source over Q_eval.
    r_eval = load_r_artifact(args.r_eval_path)
    # Round-2 fix (BLOCKER #2): coverage check uses the EFFECTIVE source (the
    # CLI override OR the module default), not the hardcoded import — otherwise
    # a v2 fallback run on medical_doctor would silently mis-assert against
    # villain and either skip its own R_eval check or fail confusingly.
    for p in [*held_out_panel, effective_source]:
        if p not in r_eval:
            raise KeyError(
                f"R_eval missing persona {p!r}; re-run #472 Phase 1 r-generate over the bank."
            )

    ckpt_index = json.loads(args.checkpoint_index.read_text())
    checkpoint_specs = []
    for frac_str, entry in sorted(ckpt_index.items(), key=lambda kv: float(kv[0])):
        if entry.get("path") is None:
            log.warning("Checkpoint frac=%s has no path; skipping.", frac_str)
            continue
        checkpoint_specs.append(
            {"frac": float(frac_str), "step": entry.get("step"), "adapter_path": entry["path"]}
        )
    if not checkpoint_specs:
        raise RuntimeError(
            f"No usable checkpoints in {args.checkpoint_index} (expected fractions "
            f"{TRAJECTORY_CHECKPOINT_FRACTIONS}). Training may have written zero checkpoints."
        )
    log.info(
        "Evaluating %d checkpoints: %s",
        len(checkpoint_specs),
        [c["frac"] for c in checkpoint_specs],
    )

    # ── #534 round-2: source-manifest guard meta (adapter-applied cross-check).
    source_guard_meta = build_source_guard_meta(args.fraction_manifest)

    # vLLM LoRAConfig.max_lora_rank must be one of (8, 16, 32, 64, 128, 256, 320,
    # 512) — buffer size, not the adapter's actual rank. r=4 fits in an r=8
    # buffer (vLLM zero-pads unused rows), so floor when training pinned r < 8.
    vllm_max_lora_rank = max(8, args.max_lora_rank)
    log.info(
        "[max_lora_rank] training rank=%d → vLLM buffer=%d.",
        args.max_lora_rank,
        vllm_max_lora_rank,
    )
    # Round-2 fix (BLOCKER #2): score against the EFFECTIVE source (v2 fallback
    # may swap villain for medical_doctor / etc.). The source_prompt is read
    # from the same bank entry — bank is single source of truth.
    if effective_source not in bank:
        raise KeyError(
            f"--source {effective_source!r} missing from persona bank at {args.bank_path}; "
            "the fallback-source candidate must exist in the bank used by Phase 0.5 + Phase 1."
        )
    run_trajectory_eval(
        cell_slug=args.cell,
        seed=args.seed,
        checkpoint_specs=checkpoint_specs,
        eval_personas=eval_personas,
        eval_questions=q_eval,
        source=effective_source,
        source_prompt=bank[effective_source],
        out_path=args.out_path,
        max_new_tokens=args.max_new_tokens,
        max_lora_rank=vllm_max_lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        compute_kl=not args.no_kl,
        source_guard_meta=source_guard_meta,
    )

    if not args.out_path.exists():
        raise RuntimeError(
            f"eval_trajectory exited but {args.out_path} is missing — silent eval failure."
        )

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 504,
                    "phase": f"eval_{args.cell}_seed{args.seed}",
                    "by": "i504_eval_trajectory",
                    "ts": datetime.now(UTC).isoformat(),
                    "note": json.dumps(
                        {
                            "cell": args.cell,
                            "seed": args.seed,
                            "trajectory_path": str(args.out_path),
                            "n_held_out_panel": len(held_out_panel),
                        }
                    ),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
