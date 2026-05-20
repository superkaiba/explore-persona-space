#!/usr/bin/env python3
"""Issue #370 — Sweep B: position-1 suffix isolation, `process` + `qui est` pinned.

Follow-up to #351. With `process` pinned at position 0 and `qui est` at
positions 2-3, sweeps single-token candidates at position 1 (4-word phrase
`process <X> qui est`) at T=1.0 to distinguish three regimes:

  (a) only `us` fires high  → `-us` subtoken is uniquely load-bearing
  (b) multiple Latin suffixes fire high → broader morphological feature
  (c) no suffix fires high  → BPE single-token mechanism only

Two-stage gating:
  Stage 1 (screen):  all ~150-300 candidates x n=20
  Stage 2 (confirm): top-5 by stage-1 French rate x n=80

Cross-arm soft-halt: if Sweep A already cleared the n=80 80% threshold,
the dispatcher leaves a sentinel file under `eval_results/issue_370/
halt_other_arm` and Sweep B exits at startup with a "skipped" run summary.
This script always runs after Sweep A in the pod's docker entrypoint.

Inheritance: reuses `scripts.issue_188_evolutionary_trigger`'s
`_generate_completions`, `_judge_records`, `_aggregate_per_candidate`,
`_init_wandb`, `_resolve_path`, `_load_or_fetch_contexts` verbatim — same
pattern as `scripts/issue_331_phase0_panel.py`.

Usage:
    uv run python scripts/issue_370_sweep_b.py --config-name issue_370
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Ensure repo root is on sys.path so `from scripts.X import Y` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._issue_370_shared import (
    assemble_sweep_b_phrase,
    build_position1_suffix_candidates,
    halt_sentinel_path,
    is_other_arm_halted,
    load_gaperon_tokenizer,
    merge_manifest,
    persist_stage_records,
    run_stage,
    upload_artifact,
    write_halt_sentinel,
)
from scripts.issue_188_evolutionary_trigger import (
    _init_wandb,
    _load_or_fetch_contexts,
    _resolve_path,
)

logger = logging.getLogger(__name__)


def _build_candidates(cfg: DictConfig, project_root: Path) -> tuple[list[dict], dict]:
    """Build the Sweep-B position-1 candidate list + provenance manifest."""
    lemma_seed_path = _resolve_path(cfg.sweep_b.vocab.include_latin_lemma_roots_from, project_root)
    with open(lemma_seed_path) as f:
        lemma_seed = json.load(f)
    logger.info("Loaded %d Latin lemma seeds from %s", len(lemma_seed), lemma_seed_path)

    tokenizer = load_gaperon_tokenizer(
        cfg.sweep_a.vocab.tokenizer_path,
        cfg.sweep_a.vocab.tokenizer_revision,
    )

    candidates, manifest = build_position1_suffix_candidates(
        tokenizer,
        suffix_tokens=list(cfg.sweep_b.vocab.suffix_tokens),
        lemma_seed=lemma_seed,
        include_random_control_tokens=int(cfg.sweep_b.vocab.include_random_control_tokens),
        rng_seed=int(cfg.seed),
    )
    manifest["tokenizer"] = {
        "model": cfg.sweep_a.vocab.tokenizer_path,
        "revision": cfg.sweep_a.vocab.tokenizer_revision,
    }
    manifest["lemma_seed_path"] = str(lemma_seed_path)

    expected = cfg.sweep_b.vocab.expected_total_size
    if len(candidates) < int(expected.min) or len(candidates) > int(expected.max):
        logger.warning(
            "Total Sweep-B candidates %d outside expected range [%d, %d] — proceeding anyway",
            len(candidates),
            int(expected.min),
            int(expected.max),
        )

    return candidates, manifest


def _make_candidate_dicts(
    tokens_with_kind: list[dict], pin_prefix: str, pin_suffix: str, category: str
) -> list[dict]:
    """Convert position-1 tokens to dicts the parent helpers expect."""
    out = []
    for entry in tokens_with_kind:
        tok = entry["token"]
        phrase = assemble_sweep_b_phrase(pin_prefix, tok, pin_suffix)
        out.append(
            {
                "phrase": phrase,
                "category": category,
                "position_1_token": tok,
                "position_1_kind": entry["kind"],
            }
        )
    return out


def _check_soft_halt(
    aggregated, cfg: DictConfig, project_root: Path, *, stage_name: str
) -> tuple[bool, dict | None]:
    """Mirror Sweep A's soft-halt check (writes the cross-arm sentinel)."""
    threshold = float(cfg.soft_halt.promote_to_n400_if_n80_rate_at_least)
    for rec in aggregated:
        if rec.frde_rate >= threshold:
            payload = write_halt_sentinel(
                cfg,
                project_root,
                halted_by="sweep_b",
                winner_phrase=rec.phrase,
                winner_rate=rec.frde_rate,
                stage=stage_name,
            )
            return True, {
                "phrase": rec.phrase,
                "rate": rec.frde_rate,
                "n_total": rec.n_total,
                "n_fr": rec.n_fr,
                "sentinel_path": str(payload),
            }
    return False, None


def _write_skipped_summary(cfg: DictConfig, project_root: Path, sentinel: dict) -> Path:
    """Sweep B exited because Sweep A already cleared the threshold."""
    output_dir = _resolve_path(cfg.output_dir, project_root)
    summary_path = output_dir / "sweep_b" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": cfg.experiment,
        "sweep": "b",
        "status": "skipped_due_to_soft_halt",
        "reason": (
            "Sweep A wrote a soft-halt sentinel before Sweep B started — "
            "the cross-arm halt rule says only the winning arm proceeds."
        ),
        "sentinel": sentinel,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Sweep B skipped — wrote %s", summary_path)
    return summary_path


def _sweep_b_main(cfg: DictConfig) -> None:
    """Run Sweep B end-to-end."""
    from explore_persona_space.metadata import get_run_metadata
    from explore_persona_space.sagan_progress import post_progress

    project_root = Path(__file__).resolve().parent.parent
    output_dir = _resolve_path(cfg.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Soft-halt: if Sweep A already cleared 80% at n=80, skip cleanly.
    if is_other_arm_halted(cfg, project_root):
        sentinel_path = halt_sentinel_path(cfg, project_root)
        with open(sentinel_path) as f:
            sentinel = json.load(f)
        logger.warning(
            "Soft-halt sentinel present at startup (%s halted at %s @ rate=%.4f) — "
            "Sweep B skipping per plan §Kill Criterion.",
            sentinel.get("halted_by"),
            sentinel.get("stage"),
            float(sentinel.get("winner_rate", 0.0)),
        )
        summary_path = _write_skipped_summary(cfg, project_root, sentinel)
        upload_artifact(summary_path, cfg, label="sweep_b_skipped")
        return

    post_progress(5.0, "sweep_b: building candidates")
    candidates_with_kind, vocab_manifest = _build_candidates(cfg, project_root)
    manifest_path = _resolve_path(cfg.sweep_a.manifest_path, project_root)
    merge_manifest(manifest_path, "sweep_b", vocab_manifest)
    upload_artifact(manifest_path, cfg, label="sweep_b_manifest")

    stage1_cfg = cfg.stages.sweep_b.stage1_screen
    stage2_cfg = cfg.stages.sweep_b.stage2_confirm

    contexts = _load_or_fetch_contexts(
        _resolve_path(stage1_cfg.contexts_path, project_root),
        n=max(int(stage1_cfg.n_contexts), int(stage2_cfg.n_contexts)),
    )
    logger.info("Loaded %d contexts for Sweep B", len(contexts))

    post_progress(10.0, "sweep_b: loading vLLM")
    from vllm import LLM

    logger.info("Loading vLLM model %s @ revision=%s", cfg.poisoned_model, cfg.model_revision)
    llm = LLM(
        model=cfg.poisoned_model,
        revision=cfg.model_revision,
        dtype="bfloat16",
        gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
        max_model_len=cfg.vllm.max_model_len,
        trust_remote_code=True,
    )

    wandb_run = _init_wandb(cfg)

    # ── Stage 1: screen ────────────────────────────────────────────────────
    post_progress(20.0, f"sweep_b stage 1 screen: {len(candidates_with_kind)} candidates")
    s1_candidates = _make_candidate_dicts(
        candidates_with_kind,
        cfg.sweep_b.pin.position_0,
        cfg.sweep_b.pin.position_2_3,
        category="sweep_b_stage1",
    )
    s1_contexts = contexts[: int(stage1_cfg.n_contexts)]
    s1_aggregated, _s1_judged, _s1_records, llm = run_stage(
        candidates=s1_candidates,
        contexts=s1_contexts,
        cfg=cfg,
        stage_cfg=stage1_cfg,
        project_root=project_root,
        llm=llm,
    )
    s1_path = _resolve_path(cfg.sweep_b.per_candidate_paths.stage1, project_root)
    persist_stage_records(s1_aggregated, s1_path)
    upload_artifact(s1_path, cfg, label="sweep_b_stage1")

    # ── Stage 2: confirmation ──────────────────────────────────────────────
    top_k = int(stage2_cfg.top_k_from_stage1)
    top_s1 = s1_aggregated[:top_k]
    logger.info(
        "Stage 1 → 2: promoting top-%d candidates (best frde=%.4f, worst promoted=%.4f)",
        len(top_s1),
        top_s1[0].frde_rate if top_s1 else 0.0,
        top_s1[-1].frde_rate if top_s1 else 0.0,
    )
    post_progress(60.0, f"sweep_b stage 2 confirm: top-{len(top_s1)}")
    # Reconstruct candidate dicts for stage 2 — preserve original kind/token.
    by_phrase = {c["phrase"]: c for c in s1_candidates}
    s2_candidates = [
        {**by_phrase[r.phrase], "category": "sweep_b_stage2"}
        for r in top_s1
        if r.phrase in by_phrase
    ]
    s2_contexts = contexts[: int(stage2_cfg.n_contexts)]
    s2_aggregated, _s2_judged, _s2_records, llm = run_stage(
        candidates=s2_candidates,
        contexts=s2_contexts,
        cfg=cfg,
        stage_cfg=stage2_cfg,
        project_root=project_root,
        llm=llm,
    )
    s2_path = _resolve_path(cfg.sweep_b.per_candidate_paths.stage2, project_root)
    persist_stage_records(s2_aggregated, s2_path)
    upload_artifact(s2_path, cfg, label="sweep_b_stage2")

    # Soft-halt check (informational — no n=400 stage on Sweep B in any case).
    _halt_fired, _winner = _check_soft_halt(
        s2_aggregated, cfg, project_root, stage_name="sweep_b_stage2"
    )

    # ── Run summary ───────────────────────────────────────────────────────
    summary = {
        "experiment": cfg.experiment,
        "sweep": "b",
        "stage1": {
            "n_candidates": len(s1_aggregated),
            "top1_phrase": s1_aggregated[0].phrase if s1_aggregated else None,
            "top1_frde": s1_aggregated[0].frde_rate if s1_aggregated else 0.0,
        },
        "stage2": {
            "n_candidates": len(s2_aggregated),
            "top1_phrase": s2_aggregated[0].phrase if s2_aggregated else None,
            "top1_frde": s2_aggregated[0].frde_rate if s2_aggregated else 0.0,
        },
        "soft_halt_fired_at_stage2": bool(_halt_fired),
        "soft_halt_winner": _winner,
        "metadata": get_run_metadata(cfg),
    }
    summary_path = output_dir / "sweep_b" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Sweep B complete. Top-1 stage-2: %s @ frde=%.4f",
        summary["stage2"]["top1_phrase"],
        summary["stage2"]["top1_frde"],
    )
    upload_artifact(summary_path, cfg, label="sweep_b_summary")

    if wandb_run is not None:
        try:
            import wandb

            wandb.log(
                {
                    "sweep_b/stage1_top1_frde": summary["stage1"]["top1_frde"],
                    "sweep_b/stage2_top1_frde": summary["stage2"]["top1_frde"],
                    "sweep_b/n_candidates": len(candidates_with_kind),
                    "sweep_b/soft_halt_fired": int(bool(_halt_fired)),
                }
            )
            wandb_run.finish()
        except Exception:
            logger.warning("WandB finalize failed", exc_info=True)

    post_progress(95.0, "sweep_b: done")


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="issue_370")
def main(cfg: DictConfig) -> None:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Issue #370 Sweep B — config: %s", cfg.experiment)

    _sweep_b_main(cfg)


if __name__ == "__main__":
    sys.exit(main() or 0)
