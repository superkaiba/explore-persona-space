"""Phase implementations for the marker-factor-screen experiment.

  - **Phase 0** (pod 0 only, `--run-pre-screen`): base-model contamination
    pre-screen — no LoRA, run vLLM over the 24×20×5 panel, count [ZLT] rates,
    write `pre_screen.json` with kill-criterion-4 verdict.

  - **Phase 1** (pod 0 only, `--run-smoke`): resolution-III 2^(5-2) fractional-
    factorial smoke on librarian. 8 cells; check kill criteria 1–4; if any
    triggered, exit non-zero so the dispatcher halts the slabs.

  - **Phase 2** (pods 0/1/2, `--source-persona <s>`): full 2^5 = 32 cells for
    that pod's source persona at the primary seed.

  - **Phase 3** (pods 0/1/2 after Phase 2): re-train top-3 cells at 2 extra
    seeds (137 / 256) — 6 extra runs per pod.

  - **Phase 4** (pod 3 only): aggregator — wait for source slabs to land,
    compute main effects + interactions, build figures, write the clean-result
    HTML.

Kill criteria (used in Phases 0 and 1):
  1. Source rate at seed 42 < 0.10 on any cell — model can't even implant.
  2. Mean leakage rate at seed 42 > 0.40 on any cell — model leaks indiscriminately.
  3. Base-model [ZLT] emission > 0.05 on >=2 panel personas (contamination).
  4. Smoke fails any of (1), (2), (3) — verdict !"pass".
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .. import _progress as progress
from .bootstrap import (
    bootstrap_leakage_rate,
    bootstrap_source_rate,
)
from .cells import Cell, all_full_cells, smoke_cells
from .data_prep import (
    PreparedDataset,
    prepare_offpolicy_cell,
    prepare_onpolicy_cell,
)
from .eval_panel import (
    DEFAULT_EVAL_MAX_NEW_TOKENS,
    EvalConfig,
    generate_completions,
    score_markers,
)
from .onpolicy import (
    BASE_MODEL,
    OnPolicyConfig,
    build_cache,
)
from .persona_panel import (
    EVAL_PERSONAS_24,
    EVAL_QUESTIONS_20,
    bystanders_for,
    resolve_source,
)
from .training import TrainOutcome, train_one_cell

log = logging.getLogger("eps.factor_screen.phases")


KILL_BASE_RATE_THRESHOLD = 0.05
KILL_BASE_RATE_MIN_PERSONAS = 2
KILL_MIN_SOURCE_RATE = 0.10
KILL_MAX_LEAKAGE_RATE = 0.40


def _load_generic_questions(repo_root: Path) -> list[str]:
    p = repo_root / "data" / "leakage_experiment" / "generic_questions.json"
    if not p.exists():
        raise FileNotFoundError(f"generic_questions.json missing at {p}")
    return json.loads(p.read_text())


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 0 — base-model contamination pre-screen
# ──────────────────────────────────────────────────────────────────────────────


def run_phase0_pre_screen(
    *,
    pod_dir: Path,
    eval_personas: int,
    eval_questions: int,
    eval_completions: int,
    max_new_tokens: int = DEFAULT_EVAL_MAX_NEW_TOKENS,
    seed: int = 42,
) -> dict:
    """Phase 0 — base model contamination pre-screen."""
    progress.post_milestone("phase0_start", n_personas=eval_personas, n_questions=eval_questions)
    log.info("Phase 0: base-model contamination pre-screen")

    panel_personas = {
        name: prompt
        for i, (name, prompt) in enumerate(EVAL_PERSONAS_24.items())
        if i < eval_personas
    }
    panel_questions = list(EVAL_QUESTIONS_20)[:eval_questions]

    start = time.time()
    completions = generate_completions(
        EvalConfig(
            model_path=BASE_MODEL,
            num_completions=eval_completions,
            max_new_tokens=max_new_tokens,
            personas=panel_personas,
            questions=panel_questions,
            seed=seed,
        )
    )
    eval_minutes = (time.time() - start) / 60

    marker_scores = score_markers(completions)

    above_threshold = [
        name
        for name, row in marker_scores.items()
        if row["substring_rate"] > KILL_BASE_RATE_THRESHOLD
    ]
    contamination = len(above_threshold) >= KILL_BASE_RATE_MIN_PERSONAS
    kill_criterion_4_passed = not contamination

    out = {
        "phase": "phase0_pre_screen",
        "base_model": BASE_MODEL,
        "n_personas": len(panel_personas),
        "n_questions": len(panel_questions),
        "n_completions_per_question": eval_completions,
        "max_new_tokens": max_new_tokens,
        "eval_wall_minutes": round(eval_minutes, 2),
        "thresholds": {
            "per_persona_max_rate": KILL_BASE_RATE_THRESHOLD,
            "min_personas_to_kill": KILL_BASE_RATE_MIN_PERSONAS,
        },
        "per_persona": {
            name: {
                "substring_rate": row["substring_rate"],
                "fuzzy_rate": row["fuzzy_rate"],
                "substring_found": row["substring_found"],
                "fuzzy_found": row["fuzzy_found"],
                "total": row["total"],
            }
            for name, row in marker_scores.items()
        },
        "personas_above_threshold": above_threshold,
        "contamination_detected": contamination,
        "kill_criterion_4_passed": kill_criterion_4_passed,
    }
    _write_json(pod_dir / "pre_screen.json", out)

    # Also persist raw completions for the analyzer.
    _write_json(pod_dir / "pre_screen_raw_completions.json", completions)

    progress.post_milestone(
        "phase0_done",
        contamination=contamination,
        kill_passed=kill_criterion_4_passed,
        n_above=len(above_threshold),
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — resolution-III smoke
# ──────────────────────────────────────────────────────────────────────────────


def run_phase1_smoke(
    *,
    pod_dir: Path,
    repo_root: Path,
    source_cli: str,
    lora_r: int,
    lora_alpha: int,
    lr: float,
    epochs: int,
    pos_per_source: int,
    neg_per_source: int,
    eval_completions: int,
    eval_max_new_tokens: int,
    seed: int = 42,
    gpu_id: int = 0,
    wandb_project: str | None = None,
) -> dict:
    """Phase 1 — 8-cell res-III fractional factorial smoke on librarian."""
    progress.post_milestone("phase1_start", source=source_cli)
    log.info("Phase 1: smoke on source=%s", source_cli)
    cells = smoke_cells()
    log.info("Smoke cells: %s", [c.key for c in cells])

    # On-policy cache shared across all smoke cells with F4=1.
    questions = _load_generic_questions(repo_root)
    cache_dir = pod_dir / "smoke_onpolicy_cache"
    onpolicy_cache = build_cache(
        OnPolicyConfig(
            source_cli=source_cli,
            pos_per_source=pos_per_source,
            neg_per_source=neg_per_source,
            questions=questions,
            cache_dir=cache_dir,
            seed=seed,
        )
    )

    cell_results: list[dict] = []
    smoke_dir = pod_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    for cell in cells:
        cell_outdir = smoke_dir / f"cell_{cell.key}"
        cell_outdir.mkdir(parents=True, exist_ok=True)

        prep: PreparedDataset
        if cell.f4 == 0:
            prep = prepare_offpolicy_cell(
                cell=cell,
                source_cli=source_cli,
                base_data_dir=repo_root / "data" / "leakage_experiment",
                output_dir=smoke_dir,
                pos_per_source=pos_per_source,
                neg_per_source=neg_per_source,
                seed=seed,
            )
        else:
            prep = prepare_onpolicy_cell(
                cell=cell,
                source_cli=source_cli,
                onpolicy_cache=onpolicy_cache,
                output_dir=smoke_dir,
                pos_per_source=pos_per_source,
                neg_per_source=neg_per_source,
                seed=seed,
            )

        outcome = train_one_cell(
            cell=cell,
            seed=seed,
            data_path=prep.path,
            cell_output_dir=cell_outdir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lr=lr,
            epochs=epochs,
            gpu_id=gpu_id,
            max_length=4096 if cell.f1 == 1 or cell.f2 == 1 else 2048,
            wandb_project=wandb_project,
            run_name_prefix="i365_smoke",
            hf_upload=False,
        )

        eval_start = time.time()
        completions = generate_completions(
            EvalConfig(
                model_path=outcome.merged_path,
                num_completions=eval_completions,
                max_new_tokens=eval_max_new_tokens,
                seed=seed,
            )
        )
        eval_minutes = (time.time() - eval_start) / 60
        marker_scores = score_markers(completions)

        resolved_source = resolve_source(source_cli)
        source_row = marker_scores.get(resolved_source, {})
        source_substring_rate = source_row.get("substring_rate", 0.0)
        leakage_rates = [
            row["substring_rate"]
            for name, row in marker_scores.items()
            if name != resolved_source
        ]
        mean_leakage = sum(leakage_rates) / len(leakage_rates) if leakage_rates else 0.0

        cell_results.append(
            {
                "cell_key": cell.key,
                "bits": list(cell.bits),
                "source_substring_rate": source_substring_rate,
                "mean_leakage_substring_rate": mean_leakage,
                "train_loss": outcome.loss,
                "train_wall_minutes": outcome.train_wall_minutes,
                "eval_wall_minutes": round(eval_minutes, 2),
                "n_examples": outcome.n_examples,
                "caveats": prep.caveats,
            }
        )

        _write_json(
            cell_outdir / "smoke_metrics.json",
            {
                "cell": cell_results[-1],
                "per_persona": {
                    name: {
                        "substring_rate": row["substring_rate"],
                        "fuzzy_rate": row["fuzzy_rate"],
                        "total": row["total"],
                    }
                    for name, row in marker_scores.items()
                },
            },
        )
        progress.post_milestone(
            "smoke_cell_done",
            cell=cell.key,
            sr=round(source_substring_rate, 3),
            lr=round(mean_leakage, 3),
        )

    # Apply kill criteria 1, 2 on the smoke cells.
    kc1_violations = [
        c for c in cell_results if c["source_substring_rate"] < KILL_MIN_SOURCE_RATE
    ]
    kc2_violations = [
        c for c in cell_results if c["mean_leakage_substring_rate"] > KILL_MAX_LEAKAGE_RATE
    ]

    if kc1_violations:
        verdict = "kill_1_source_rate_too_low"
    elif kc2_violations:
        verdict = "kill_2_leakage_too_high"
    else:
        verdict = "pass"

    summary = {
        "phase": "phase1_smoke",
        "source": source_cli,
        "design": "resolution-III 2^(5-2), F4=F1·F2, F5=F1·F3",
        "n_cells": len(cell_results),
        "cells": cell_results,
        "kill_criteria": {
            "kc1_source_rate_min": KILL_MIN_SOURCE_RATE,
            "kc2_leakage_rate_max": KILL_MAX_LEAKAGE_RATE,
            "kc1_violations": [c["cell_key"] for c in kc1_violations],
            "kc2_violations": [c["cell_key"] for c in kc2_violations],
        },
        "verdict": verdict,
        "note": "smoke not used for factor pre-ranking; only kill-criterion gating",
    }
    _write_json(pod_dir / "smoke.json", summary)

    progress.post_milestone("phase1_done", verdict=verdict)
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — full 2^5 slab for one source persona
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SlabCellMetrics:
    """One cell of a per-source 32-cell slab (Phase 2)."""

    cell_key: str
    bits: list[int]
    seed: int
    source_substring_rate: float
    source_fuzzy_rate: float
    source_rate_ci_substring: tuple[float, float]
    source_rate_ci_fuzzy: tuple[float, float]
    mean_leakage_substring_rate: float
    mean_leakage_fuzzy_rate: float
    leakage_rate_ci_substring: tuple[float, float]
    leakage_rate_ci_fuzzy: tuple[float, float]
    per_bystander_rates: dict[str, float]
    per_bystander_appeared_in_training: dict[str, bool]
    train_loss: float
    train_wall_minutes: float
    eval_wall_minutes: float
    n_examples: int
    f4_data_source: str
    caveats: list[str]
    failed: bool
    error: str | None


def _bystanders_in_training(
    source_cli: str, f4: int, source_data_dir: Path | None
) -> set[str]:
    """Identify which 24-panel personas appeared as negatives during training.

    For F4=off cells, we read the pre-built file's bystander rows. For F4=on
    cells, the bystander set is drawn at random from `bystanders_for(source)`
    each generation pass, so we conservatively mark ALL panel personas as
    "appeared" for F4=on cells (the on-policy sampler picks from the full
    bystander pool).
    """
    resolved = resolve_source(source_cli)
    if f4 == 1:
        return set(bystanders_for(resolved))
    if source_data_dir is None:
        return set()
    src_file = source_data_dir / f"marker_{resolved}_asst_excluded_medium.jsonl"
    if not src_file.exists():
        return set()
    panel_by_prompt = {prompt: name for name, prompt in EVAL_PERSONAS_24.items()}
    appeared: set[str] = set()
    with open(src_file) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            sys_prompt = ""
            for msg in ex.get("prompt", []):
                if msg.get("role") == "system":
                    sys_prompt = msg.get("content", "")
                    break
            if sys_prompt and sys_prompt in panel_by_prompt:
                appeared.add(panel_by_prompt[sys_prompt])
    return appeared


def _train_eval_one_cell(
    *,
    cell: Cell,
    seed: int,
    source_cli: str,
    repo_root: Path,
    pod_dir: Path,
    slab_dir: Path,
    onpolicy_cache: dict,
    lora_r: int,
    lora_alpha: int,
    lr: float,
    epochs: int,
    pos_per_source: int,
    neg_per_source: int,
    eval_completions: int,
    eval_max_new_tokens: int,
    wandb_project: str | None,
    gpu_id: int = 0,
) -> SlabCellMetrics:
    """Train and eval ONE cell. Catches per-cell failures so Phase 2 can continue."""
    cell_outdir = slab_dir / f"cell_{cell.key}"
    cell_outdir.mkdir(parents=True, exist_ok=True)
    base_data_dir = repo_root / "data" / "leakage_experiment"
    appeared = _bystanders_in_training(source_cli, cell.f4, base_data_dir)

    try:
        if cell.f4 == 0:
            prep = prepare_offpolicy_cell(
                cell=cell,
                source_cli=source_cli,
                base_data_dir=base_data_dir,
                output_dir=slab_dir,
                pos_per_source=pos_per_source,
                neg_per_source=neg_per_source,
                seed=seed,
            )
        else:
            prep = prepare_onpolicy_cell(
                cell=cell,
                source_cli=source_cli,
                onpolicy_cache=onpolicy_cache,
                output_dir=slab_dir,
                pos_per_source=pos_per_source,
                neg_per_source=neg_per_source,
                seed=seed,
            )

        outcome = train_one_cell(
            cell=cell,
            seed=seed,
            data_path=prep.path,
            cell_output_dir=cell_outdir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lr=lr,
            epochs=epochs,
            gpu_id=gpu_id,
            max_length=4096 if cell.f1 == 1 or cell.f2 == 1 else 2048,
            wandb_project=wandb_project,
            run_name_prefix=f"i365_{source_cli}",
            hf_upload=False,
        )

        eval_start = time.time()
        completions = generate_completions(
            EvalConfig(
                model_path=outcome.merged_path,
                num_completions=eval_completions,
                max_new_tokens=eval_max_new_tokens,
                seed=seed,
            )
        )
        eval_minutes = (time.time() - eval_start) / 60
        _write_json(cell_outdir / "raw_completions.json", completions)

        marker_scores = score_markers(completions)
        _write_json(cell_outdir / "marker_scores.json", marker_scores)

        resolved_source = resolve_source(source_cli)
        source_row = marker_scores.get(resolved_source, {})
        source_per_q = source_row.get("per_question", {})

        sr_substring = float(source_row.get("substring_rate", 0.0))
        sr_fuzzy = float(source_row.get("fuzzy_rate", 0.0))
        sr_ci_sub = bootstrap_source_rate(source_per_q, seed=seed)
        sr_ci_fuzz = bootstrap_source_rate(
            source_per_q, seed=seed, rate_field="fuzzy_rate"
        )

        bystander_substring = {
            name: float(row["substring_rate"])
            for name, row in marker_scores.items()
            if name != resolved_source
        }
        bystander_fuzzy = {
            name: float(row["fuzzy_rate"])
            for name, row in marker_scores.items()
            if name != resolved_source
        }
        lr_substring = (
            sum(bystander_substring.values()) / len(bystander_substring)
            if bystander_substring
            else 0.0
        )
        lr_fuzzy = (
            sum(bystander_fuzzy.values()) / len(bystander_fuzzy)
            if bystander_fuzzy
            else 0.0
        )
        lr_ci_sub = bootstrap_leakage_rate(bystander_substring, seed=seed)
        lr_ci_fuzz = bootstrap_leakage_rate(bystander_fuzzy, seed=seed)

        per_bystander_appeared = {
            name: name in appeared for name in bystander_substring
        }

        # Persist the [ZLT] tokenization under the Qwen tokenizer for the analyzer.
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
            )
            zlt_ids = tok.encode("[ZLT]", add_special_tokens=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to fetch [ZLT] tokenization: %s", exc)
            zlt_ids = []

        metrics_payload = {
            "cell_key": cell.key,
            "bits": list(cell.bits),
            "seed": seed,
            "source_cli": source_cli,
            "source_resolved": resolved_source,
            "source_substring_rate": sr_substring,
            "source_fuzzy_rate": sr_fuzzy,
            "source_rate_ci_substring": list(sr_ci_sub),
            "source_rate_ci_fuzzy": list(sr_ci_fuzz),
            "mean_leakage_substring_rate": lr_substring,
            "mean_leakage_fuzzy_rate": lr_fuzzy,
            "leakage_rate_ci_substring": list(lr_ci_sub),
            "leakage_rate_ci_fuzzy": list(lr_ci_fuzz),
            "per_bystander_substring_rates": bystander_substring,
            "per_bystander_fuzzy_rates": bystander_fuzzy,
            "per_bystander_appeared_in_training": per_bystander_appeared,
            "train_loss": outcome.loss,
            "train_wall_minutes": outcome.train_wall_minutes,
            "eval_wall_minutes": round(eval_minutes, 2),
            "n_examples": outcome.n_examples,
            "f4_data_source": prep.f4_data_source,
            "f4_off_source_path": prep.f4_off_source_path,
            "caveats": prep.caveats,
            "zlt_token_ids_qwen_tokenizer": zlt_ids,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lr": lr,
            "epochs": epochs,
            "failed": False,
            "error": None,
        }
        _write_json(cell_outdir / "metrics.json", metrics_payload)

        return SlabCellMetrics(
            cell_key=cell.key,
            bits=list(cell.bits),
            seed=seed,
            source_substring_rate=sr_substring,
            source_fuzzy_rate=sr_fuzzy,
            source_rate_ci_substring=sr_ci_sub,
            source_rate_ci_fuzzy=sr_ci_fuzz,
            mean_leakage_substring_rate=lr_substring,
            mean_leakage_fuzzy_rate=lr_fuzzy,
            leakage_rate_ci_substring=lr_ci_sub,
            leakage_rate_ci_fuzzy=lr_ci_fuzz,
            per_bystander_rates=bystander_substring,
            per_bystander_appeared_in_training=per_bystander_appeared,
            train_loss=outcome.loss,
            train_wall_minutes=outcome.train_wall_minutes,
            eval_wall_minutes=round(eval_minutes, 2),
            n_examples=outcome.n_examples,
            f4_data_source=prep.f4_data_source,
            caveats=prep.caveats,
            failed=False,
            error=None,
        )

    except Exception as exc:  # noqa: BLE001 — per-cell failure should not kill the slab
        log.exception("Cell %s seed=%d failed", cell.key, seed)
        fail_payload = {
            "cell_key": cell.key,
            "bits": list(cell.bits),
            "seed": seed,
            "failed": True,
            "error": str(exc),
        }
        _write_json(cell_outdir / "metrics.json", fail_payload)
        return SlabCellMetrics(
            cell_key=cell.key,
            bits=list(cell.bits),
            seed=seed,
            source_substring_rate=0.0,
            source_fuzzy_rate=0.0,
            source_rate_ci_substring=(0.0, 0.0),
            source_rate_ci_fuzzy=(0.0, 0.0),
            mean_leakage_substring_rate=0.0,
            mean_leakage_fuzzy_rate=0.0,
            leakage_rate_ci_substring=(0.0, 0.0),
            leakage_rate_ci_fuzzy=(0.0, 0.0),
            per_bystander_rates={},
            per_bystander_appeared_in_training={},
            train_loss=float("nan"),
            train_wall_minutes=0.0,
            eval_wall_minutes=0.0,
            n_examples=0,
            f4_data_source="error",
            caveats=[],
            failed=True,
            error=str(exc),
        )


def run_phase2_slab(
    *,
    pod_dir: Path,
    repo_root: Path,
    source_cli: str,
    lora_r: int,
    lora_alpha: int,
    lr: float,
    epochs: int,
    pos_per_source: int,
    neg_per_source: int,
    eval_completions: int,
    eval_max_new_tokens: int,
    primary_seed: int,
    wandb_project: str | None = None,
    gpu_id: int = 0,
) -> dict:
    """Run the full 32-cell slab for one source persona."""
    progress.post_milestone("phase2_start", source=source_cli, n_cells=32)
    slab_dir = pod_dir / source_cli
    slab_dir.mkdir(parents=True, exist_ok=True)

    # Build the (F1, F3, F2) on-policy cache once per source.
    questions = _load_generic_questions(repo_root)
    cache_dir = slab_dir / "onpolicy_cache"
    onpolicy_cache = build_cache(
        OnPolicyConfig(
            source_cli=source_cli,
            pos_per_source=pos_per_source,
            neg_per_source=neg_per_source,
            questions=questions,
            cache_dir=cache_dir,
            seed=primary_seed,
        )
    )
    progress.post_milestone("phase2_onpolicy_cache_ready", source=source_cli)

    cells = all_full_cells()
    slab_metrics: list[SlabCellMetrics] = []
    for cell in cells:
        log.info("Slab %s: cell %s starting", source_cli, cell.key)
        metrics = _train_eval_one_cell(
            cell=cell,
            seed=primary_seed,
            source_cli=source_cli,
            repo_root=repo_root,
            pod_dir=pod_dir,
            slab_dir=slab_dir,
            onpolicy_cache=onpolicy_cache,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lr=lr,
            epochs=epochs,
            pos_per_source=pos_per_source,
            neg_per_source=neg_per_source,
            eval_completions=eval_completions,
            eval_max_new_tokens=eval_max_new_tokens,
            wandb_project=wandb_project,
            gpu_id=gpu_id,
        )
        slab_metrics.append(metrics)
        progress.post_milestone(
            "phase2_cell_done",
            source=source_cli,
            cell=cell.key,
            sr=round(metrics.source_substring_rate, 3),
            lr=round(metrics.mean_leakage_substring_rate, 3),
            failed=metrics.failed,
        )

    summary = {
        "phase": "phase2_slab",
        "source_cli": source_cli,
        "source_resolved": resolve_source(source_cli),
        "n_cells": len(slab_metrics),
        "primary_seed": primary_seed,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lr": lr,
        "epochs": epochs,
        "pos_per_source": pos_per_source,
        "neg_per_source": neg_per_source,
        "eval_completions_per_question": eval_completions,
        "eval_max_new_tokens": eval_max_new_tokens,
        "cells": [_slab_metrics_to_dict(m) for m in slab_metrics],
    }
    _write_json(slab_dir / "metrics.json", summary)
    progress.post_milestone("phase2_done", source=source_cli)
    return summary


def _slab_metrics_to_dict(m: SlabCellMetrics) -> dict:
    return {
        "cell_key": m.cell_key,
        "bits": m.bits,
        "seed": m.seed,
        "source_substring_rate": m.source_substring_rate,
        "source_fuzzy_rate": m.source_fuzzy_rate,
        "source_rate_ci_substring": list(m.source_rate_ci_substring),
        "source_rate_ci_fuzzy": list(m.source_rate_ci_fuzzy),
        "mean_leakage_substring_rate": m.mean_leakage_substring_rate,
        "mean_leakage_fuzzy_rate": m.mean_leakage_fuzzy_rate,
        "leakage_rate_ci_substring": list(m.leakage_rate_ci_substring),
        "leakage_rate_ci_fuzzy": list(m.leakage_rate_ci_fuzzy),
        "per_bystander_substring_rates": m.per_bystander_rates,
        "per_bystander_appeared_in_training": m.per_bystander_appeared_in_training,
        "train_loss": m.train_loss,
        "train_wall_minutes": m.train_wall_minutes,
        "eval_wall_minutes": m.eval_wall_minutes,
        "n_examples": m.n_examples,
        "f4_data_source": m.f4_data_source,
        "caveats": m.caveats,
        "failed": m.failed,
        "error": m.error,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — multi-seed top-3 confirmation
# ──────────────────────────────────────────────────────────────────────────────


def run_phase3_multiseed(
    *,
    pod_dir: Path,
    repo_root: Path,
    source_cli: str,
    slab_summary: dict,
    extra_seeds: list[int],
    lora_r: int,
    lora_alpha: int,
    lr: float,
    epochs: int,
    pos_per_source: int,
    neg_per_source: int,
    eval_completions: int,
    eval_max_new_tokens: int,
    wandb_project: str | None = None,
    gpu_id: int = 0,
) -> dict:
    """Re-train top-3 cells (by primary-seed SR) at each `extra_seeds[i]`."""
    progress.post_milestone("phase3_start", source=source_cli, extra_seeds=str(extra_seeds))
    slab_dir = pod_dir / source_cli

    # Rank cells by primary-seed source SR (substring).
    valid_cells = [c for c in slab_summary["cells"] if not c["failed"]]
    valid_cells.sort(key=lambda c: c["source_substring_rate"], reverse=True)
    top3 = valid_cells[:3]

    # Rebuild the on-policy cache (it's still on disk from Phase 2; build_cache
    # short-circuits when entries are present).
    questions = _load_generic_questions(repo_root)
    cache_dir = slab_dir / "onpolicy_cache"
    onpolicy_cache = build_cache(
        OnPolicyConfig(
            source_cli=source_cli,
            pos_per_source=pos_per_source,
            neg_per_source=neg_per_source,
            questions=questions,
            cache_dir=cache_dir,
            seed=42,
        )
    )

    multiseed_results: list[dict] = []
    for cell_dict in top3:
        cell = Cell(*cell_dict["bits"])
        for seed in extra_seeds:
            log.info("Phase 3 multi-seed: cell %s seed %d", cell.key, seed)
            metrics = _train_eval_one_cell(
                cell=cell,
                seed=seed,
                source_cli=source_cli,
                repo_root=repo_root,
                pod_dir=pod_dir,
                slab_dir=slab_dir,
                onpolicy_cache=onpolicy_cache,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lr=lr,
                epochs=epochs,
                pos_per_source=pos_per_source,
                neg_per_source=neg_per_source,
                eval_completions=eval_completions,
                eval_max_new_tokens=eval_max_new_tokens,
                wandb_project=wandb_project,
                gpu_id=gpu_id,
            )
            multiseed_results.append(_slab_metrics_to_dict(metrics))
            progress.post_milestone(
                "phase3_cell_done",
                cell=cell.key,
                seed=seed,
                sr=round(metrics.source_substring_rate, 3),
            )

    summary = {
        "phase": "phase3_multiseed",
        "source_cli": source_cli,
        "top3_cell_keys": [c["cell_key"] for c in top3],
        "extra_seeds": extra_seeds,
        "results": multiseed_results,
        "primary_seed_top3": top3,
    }
    _write_json(slab_dir / "multiseed.json", summary)
    progress.post_milestone("phase3_done", source=source_cli)
    return summary
