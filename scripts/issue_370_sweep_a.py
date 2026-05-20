#!/usr/bin/env python3
"""Issue #370 — Sweep A: broader-vocab position-0 sweep, `qui est` pinned.

Follow-up to #351. Sweeps Gaperon-1125-1B's tokenizer vocab (filtered to
single-token Latin-shape entries) at position 0 with `qui est` pinned at
positions 2-3. Three-stage gating at T=1.0:

  Stage 1 (screen):       all candidates x n=20 completions
  Stage 2 (confirm):      top-15 by stage-1 French rate x n=80
  Stage 3 (headline):     top-1 by stage-2 French rate  x n=400  (+ raw samples)

Sanity gates before generation:
  - `processus` must be present in the post-filter vocab (else halt).

Soft-halt rule:
  - If any stage-2 candidate fires French ≥80%, write the cross-arm
    sentinel so Sweep B skips when it starts.

Inheritance: reuses `scripts.issue_188_evolutionary_trigger`'s
`_generate_completions`, `_judge_records`, `_aggregate_per_candidate`,
`_init_wandb`, `_resolve_path`, `_load_or_fetch_contexts` verbatim — same
pattern as `scripts/issue_331_phase0_panel.py`.

Usage:
    uv run python scripts/issue_370_sweep_a.py --config-name issue_370
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
    assemble_sweep_a_phrase,
    build_single_token_latin_vocab,
    is_other_arm_halted,
    load_gaperon_tokenizer,
    merge_manifest,
    persist_raw_completions,
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


def _build_vocab(cfg: DictConfig, project_root: Path) -> tuple[list[str], dict]:
    """Build the Sweep-A position-0 candidate list + provenance manifest."""
    seed_path = _resolve_path(cfg.sweep_a.vocab.latin_lemma_seed_path, project_root)
    with open(seed_path) as f:
        latin_seed = json.load(f)
    logger.info("Loaded %d Latin lemma seeds from %s", len(latin_seed), seed_path)

    tokenizer = load_gaperon_tokenizer(
        cfg.sweep_a.vocab.tokenizer_path,
        cfg.sweep_a.vocab.tokenizer_revision,
    )

    filters = cfg.sweep_a.vocab.filters
    candidates, manifest = build_single_token_latin_vocab(
        tokenizer,
        latin_lemma_seed=latin_seed,
        suffix_inclusion=list(cfg.sweep_a.vocab.suffix_inclusion),
        min_chars=int(filters.min_chars),
        max_chars=int(filters.max_chars),
        allow_lowercase=bool(filters.allow_lowercase),
        allow_capitalized=bool(filters.allow_capitalized),
        forbid_digits=bool(filters.forbid_digits),
    )
    manifest["tokenizer"] = {
        "model": cfg.sweep_a.vocab.tokenizer_path,
        "revision": cfg.sweep_a.vocab.tokenizer_revision,
    }
    manifest["seed_lemma_path"] = str(seed_path)

    # Expected-size sanity check (log only — do not fail).
    expected = cfg.sweep_a.vocab.expected_post_filter_size
    if len(candidates) < int(expected.min) or len(candidates) > int(expected.max):
        logger.warning(
            "Post-filter vocab size %d outside expected range [%d, %d] — proceeding anyway",
            len(candidates),
            int(expected.min),
            int(expected.max),
        )

    # Sanity gate: `processus` (or whatever is configured) must survive.
    required = cfg.sweep_a.vocab.sanity_gate.require_token_present
    if required and required not in candidates:
        raise RuntimeError(
            f"Sanity gate failed: required token {required!r} did not survive the "
            f"single-token Latin-shape filter (post-filter size={len(candidates)}). "
            f"Filter rules in manifest; halting before generation per plan §Risks."
        )
    logger.info(
        "Vocab built: %d candidates (sanity-gate token %r present)",
        len(candidates),
        required,
    )

    return candidates, manifest


def _make_candidate_dicts(tokens: list[str], pin_suffix: str, category: str) -> list[dict]:
    """Convert position-0 tokens to the dict shape `_generate_completions` expects."""
    return [
        {
            "phrase": assemble_sweep_a_phrase(tok, pin_suffix),
            "category": category,
            "position_0_token": tok,
        }
        for tok in tokens
    ]


def _check_soft_halt(
    aggregated, cfg: DictConfig, project_root: Path, *, stage_name: str
) -> tuple[bool, dict | None]:
    """If any candidate's stage-2 frde_rate clears the soft-halt threshold,
    write the sentinel and return (True, winner_info)."""
    threshold = float(cfg.soft_halt.promote_to_n400_if_n80_rate_at_least)
    for rec in aggregated:  # already sorted by frde_rate desc
        if rec.frde_rate >= threshold:
            payload = write_halt_sentinel(
                cfg,
                project_root,
                halted_by="sweep_a",
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


def _sweep_a_main(cfg: DictConfig) -> None:
    """Run Sweep A end-to-end."""
    from explore_persona_space.metadata import get_run_metadata
    from explore_persona_space.sagan_progress import post_progress

    project_root = Path(__file__).resolve().parent.parent
    output_dir = _resolve_path(cfg.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build vocab + manifest. Halt before any GPU work if sanity gate trips.
    post_progress(5.0, "sweep_a: building vocab")
    candidates_tokens, vocab_manifest = _build_vocab(cfg, project_root)
    manifest_path = _resolve_path(cfg.sweep_a.manifest_path, project_root)
    merge_manifest(manifest_path, "sweep_a", vocab_manifest)
    upload_artifact(manifest_path, cfg, label="sweep_a_manifest")

    # 2. Contexts.
    stage1_cfg = cfg.stages.sweep_a.stage1_screen
    stage2_cfg = cfg.stages.sweep_a.stage2_confirm
    stage3_cfg = cfg.stages.sweep_a.stage3_headline

    contexts_s12 = _load_or_fetch_contexts(
        _resolve_path(stage1_cfg.contexts_path, project_root),
        n=max(int(stage1_cfg.n_contexts), int(stage2_cfg.n_contexts)),
    )
    logger.info("Loaded %d contexts for stages 1+2", len(contexts_s12))

    # 3. Load vLLM once, reuse across stages.
    post_progress(10.0, "sweep_a: loading vLLM")
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
    post_progress(15.0, f"sweep_a stage 1 screen: {len(candidates_tokens)} candidates")
    s1_candidates = _make_candidate_dicts(
        candidates_tokens, cfg.sweep_a.pin.position_2_3, category="sweep_a_stage1"
    )
    s1_contexts = contexts_s12[: int(stage1_cfg.n_contexts)]
    s1_aggregated, _s1_judged, _s1_records, llm = run_stage(
        candidates=s1_candidates,
        contexts=s1_contexts,
        cfg=cfg,
        stage_cfg=stage1_cfg,
        project_root=project_root,
        llm=llm,
    )
    s1_path = _resolve_path(cfg.sweep_a.per_candidate_paths.stage1, project_root)
    persist_stage_records(s1_aggregated, s1_path)
    upload_artifact(s1_path, cfg, label="sweep_a_stage1")

    # ── Stage 2: confirmation ──────────────────────────────────────────────
    top_k_s2 = int(stage2_cfg.top_k_from_stage1)
    top_s1 = s1_aggregated[:top_k_s2]
    logger.info(
        "Stage 1 → 2: promoting top-%d candidates (best stage-1 frde=%.4f, worst promoted=%.4f)",
        len(top_s1),
        top_s1[0].frde_rate if top_s1 else 0.0,
        top_s1[-1].frde_rate if top_s1 else 0.0,
    )
    post_progress(45.0, f"sweep_a stage 2 confirm: top-{len(top_s1)}")
    s2_candidates = _make_candidate_dicts(
        [r.phrase.split()[0] for r in top_s1],
        cfg.sweep_a.pin.position_2_3,
        category="sweep_a_stage2",
    )
    s2_contexts = contexts_s12[: int(stage2_cfg.n_contexts)]
    s2_aggregated, _s2_judged, _s2_records, llm = run_stage(
        candidates=s2_candidates,
        contexts=s2_contexts,
        cfg=cfg,
        stage_cfg=stage2_cfg,
        project_root=project_root,
        llm=llm,
    )
    s2_path = _resolve_path(cfg.sweep_a.per_candidate_paths.stage2, project_root)
    persist_stage_records(s2_aggregated, s2_path)
    upload_artifact(s2_path, cfg, label="sweep_a_stage2")

    # Soft-halt check (write sentinel so Sweep B sees it on startup).
    _halt_fired, _winner = _check_soft_halt(
        s2_aggregated, cfg, project_root, stage_name="sweep_a_stage2"
    )

    # ── Stage 3: headline ─────────────────────────────────────────────────
    top_k_s3 = int(stage3_cfg.top_k_from_stage2)
    top_s2 = s2_aggregated[:top_k_s3]
    logger.info(
        "Stage 2 → 3: promoting top-%d (best stage-2 frde=%.4f)",
        len(top_s2),
        top_s2[0].frde_rate if top_s2 else 0.0,
    )
    post_progress(70.0, f"sweep_a stage 3 headline: top-{len(top_s2)} @ n=400")
    contexts_s3 = _load_or_fetch_contexts(
        _resolve_path(stage3_cfg.contexts_path, project_root),
        n=int(stage3_cfg.n_contexts),
    )
    logger.info("Loaded %d contexts for stage 3", len(contexts_s3))

    s3_candidates = _make_candidate_dicts(
        [r.phrase.split()[0] for r in top_s2],
        cfg.sweep_a.pin.position_2_3,
        category="sweep_a_stage3",
    )
    s3_aggregated, s3_judged, _s3_records, llm = run_stage(
        candidates=s3_candidates,
        contexts=contexts_s3,
        cfg=cfg,
        stage_cfg=stage3_cfg,
        project_root=project_root,
        llm=llm,
    )
    s3_path = _resolve_path(cfg.sweep_a.per_candidate_paths.stage3, project_root)
    persist_stage_records(s3_aggregated, s3_path)
    upload_artifact(s3_path, cfg, label="sweep_a_stage3")

    # Raw completions for the top-1 candidate (verbatim citation in clean result).
    if bool(stage3_cfg.persist_raw_completions):
        raw_path = _resolve_path(stage3_cfg.raw_completions_path, project_root)
        top1_phrase = s3_aggregated[0].phrase if s3_aggregated else None
        top1_records = [r for r in s3_judged if r.get("candidate_phrase") == top1_phrase]
        persist_raw_completions(top1_records, raw_path)
        upload_artifact(raw_path, cfg, label="sweep_a_top1_samples")

    # ── Run summary ───────────────────────────────────────────────────────
    summary = {
        "experiment": cfg.experiment,
        "sweep": "a",
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
        "stage3": {
            "n_candidates": len(s3_aggregated),
            "top1_phrase": s3_aggregated[0].phrase if s3_aggregated else None,
            "top1_frde": s3_aggregated[0].frde_rate if s3_aggregated else 0.0,
        },
        "soft_halt_fired_at_stage2": bool(_halt_fired),
        "soft_halt_winner": _winner,
        "metadata": get_run_metadata(cfg),
    }
    summary_path = output_dir / "sweep_a" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Sweep A complete. Top-1 stage-3: %s @ frde=%.4f",
        summary["stage3"]["top1_phrase"],
        summary["stage3"]["top1_frde"],
    )
    upload_artifact(summary_path, cfg, label="sweep_a_summary")

    if wandb_run is not None:
        try:
            import wandb

            wandb.log(
                {
                    "sweep_a/stage1_top1_frde": summary["stage1"]["top1_frde"],
                    "sweep_a/stage2_top1_frde": summary["stage2"]["top1_frde"],
                    "sweep_a/stage3_top1_frde": summary["stage3"]["top1_frde"],
                    "sweep_a/n_vocab_post_filter": len(candidates_tokens),
                    "sweep_a/soft_halt_fired": int(bool(_halt_fired)),
                }
            )
            wandb_run.finish()
        except Exception:
            logger.warning("WandB finalize failed", exc_info=True)

    post_progress(95.0, "sweep_a: done")


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="issue_370")
def main(cfg: DictConfig) -> None:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Issue #370 Sweep A — config: %s", cfg.experiment)

    project_root = Path(__file__).resolve().parent.parent
    if is_other_arm_halted(cfg, project_root):
        logger.warning(
            "Soft-halt sentinel %s already present at startup — Sweep A still runs "
            "(sentinel was meant for the OTHER arm). Continuing.",
            cfg.soft_halt.halt_sentinel_path,
        )

    _sweep_a_main(cfg)


if __name__ == "__main__":
    sys.exit(main() or 0)
