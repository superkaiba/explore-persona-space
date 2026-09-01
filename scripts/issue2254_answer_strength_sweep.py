#!/usr/bin/env python3
"""Low-dose, answer-only rescue sweep for the additional Qwen3.5 personas.

The preceding matched answer-only experiment used one frozen dose per breadth
and degraded every learned-direction cell.  This follow-up separates tuning
from confirmation:

* screen: four lower dose fractions on even-indexed evaluation prompts and
  generation seed 42;
* confirm: the selected quality-eligible cell for each trait/method on the
  odd-indexed prompts and generation seeds 43--46;
* null: eight fresh random directions at the exact selected confirmation
  geometry.

Selection never observes a confirmation prompt or generation seed.  Direction
extraction remains frozen and disjoint from all evaluation prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.issue2254_probe_context_followup as base  # noqa: E402
import scripts.issue2254_probe_context_qwen35 as q35  # noqa: E402
import scripts.issue2254_probe_context_qwen35_personas as personas  # noqa: E402

TRAITS = personas.TRAITS
METHODS = base.METHODS
BREADTHS = base.BREADTHS
DOSE_SCALES = (1 / 16, 1 / 8, 1 / 4, 1 / 2)
SCREEN_QUESTION_INDICES = tuple(range(0, 20, 2))
CONFIRM_QUESTION_INDICES = tuple(range(1, 20, 2))
SCREEN_SEEDS = (42,)
CONFIRM_SEEDS = (43, 44, 45, 46)
N_RANDOM_DIRECTIONS = 8
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 225_435_902
DEGENERATE_FRACTION_CEILING = 0.20

OUT_ROOT = (
    REPO_ROOT
    / "eval_results/issue_2254/answer_probe_qwen35_additional_personas_low_strength"
)
DIRECTION_ROOT = REPO_ROOT / "eval_results/issue_2254/context_probe_qwen35_additional_personas"
SCREEN_FIG_PATH = REPO_ROOT / "artifacts/issue2254/qwen35_answer_low_strength_screen.png"
CONFIRM_FIG_PATH = REPO_ROOT / "artifacts/issue2254/qwen35_answer_low_strength_confirm.png"

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _write_json(path: Path, payload: Any) -> None:
    personas._write_json(path, payload)


def _json_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_dir(out_root: Path, stage: str) -> Path:
    if stage not in {"screen", "confirm"}:
        raise ValueError(f"unknown stage {stage!r}")
    return out_root / stage


def _dose_for(breadth: str, scale: float) -> float:
    return float(personas.OPERATING_POINTS[breadth]["c"] * scale)


def _signal_cell(trait: str, method: str, breadth: str, scale: float) -> dict[str, Any]:
    point = personas.OPERATING_POINTS[breadth]
    return {
        "behavior": trait,
        "kind": "signal",
        "method": method,
        "position": "answer",
        "breadth": breadth,
        "layer_config": point["layer_config"],
        "c": _dose_for(breadth, scale),
        "dose_scale": float(scale),
        "source_cell_id": point["source"],
    }


def _random_cell(selected: dict[str, Any], random_seed: int) -> dict[str, Any]:
    return {
        "behavior": selected["behavior"],
        "kind": "random",
        "method": "random",
        "position": "answer",
        "random_seed": int(random_seed),
        "breadth": selected["breadth"],
        "layer_config": selected["layer_config"],
        "c": float(selected["c"]),
        "dose_scale": float(selected["dose_scale"]),
        "source_cell_id": selected["source_cell_id"],
    }


def build_screen_cells(traits: Iterable[str] = TRAITS) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for trait in traits:
        cells.append({"behavior": trait, "kind": "alpha0", "position": "answer"})
        for method in METHODS:
            for breadth in BREADTHS:
                for scale in DOSE_SCALES:
                    cells.append(_signal_cell(trait, method, breadth, scale))
    return cells


def _load_selection(out_root: Path) -> dict[str, Any]:
    path = _stage_dir(out_root, "screen") / "confirmation_selection.json"
    if not path.exists():
        raise FileNotFoundError(f"run screen-reduce first: {path}")
    return json.loads(path.read_text())


def build_confirm_cells(
    out_root: Path,
    traits: Iterable[str] = TRAITS,
    *,
    n_random: int = N_RANDOM_DIRECTIONS,
) -> list[dict[str, Any]]:
    selection = _load_selection(out_root)
    by_id: dict[str, dict[str, Any]] = {}
    for trait in traits:
        baseline = {"behavior": trait, "kind": "alpha0", "position": "answer"}
        by_id[base.cell_id(baseline)] = baseline
        for method in METHODS:
            row = selection["traits"][trait]["methods"][method]
            if row["status"] != "selected":
                continue
            signal = dict(row["cell"])
            by_id[base.cell_id(signal)] = signal
            for random_seed in range(n_random):
                random = _random_cell(signal, random_seed)
                by_id[base.cell_id(random)] = random
    return [by_id[key] for key in sorted(by_id)]


def _manifest(args: argparse.Namespace) -> dict[str, Any]:
    high_root = args.out_root.parent / "answer_probe_qwen35_additional_personas"
    return {
        "status": "frozen_before_low_strength_screen_outputs",
        "experiment": "issue2254_qwen35_answer_only_low_strength_rescue",
        "model": q35.MODEL_ID,
        "revision": q35.MODEL_REVISION,
        "transformers": q35.TRANSFORMERS_PIN,
        "thinking": "disabled",
        "position": "all_answer_prediction_positions",
        "position_implementation": (
            "DeltaHook(all_positions=True): prefill slot predicting answer token 1, "
            "then every one-token cached decode position"
        ),
        "traits": list(TRAITS),
        "methods": list(METHODS),
        "breadths": list(BREADTHS),
        "failed_doses": {
            breadth: float(personas.OPERATING_POINTS[breadth]["c"]) for breadth in BREADTHS
        },
        "dose_scales": list(DOSE_SCALES),
        "actual_doses": {
            breadth: [_dose_for(breadth, scale) for scale in DOSE_SCALES]
            for breadth in BREADTHS
        },
        "screen": {
            "question_indices": list(SCREEN_QUESTION_INDICES),
            "generation_seeds": list(SCREEN_SEEDS),
            "cells": "alpha0 plus every trait x method x breadth x dose_scale signal cell",
            "selection": (
                "within each trait/method, maximize mean trait-score increase among cells "
                "passing every frozen quality gate; ties prefer lower dose_scale, then "
                "narrower breadth"
            ),
        },
        "confirm": {
            "question_indices": list(CONFIRM_QUESTION_INDICES),
            "generation_seeds": list(CONFIRM_SEEDS),
            "independence": "no screen question or generation seed is reused",
            "random_direction_seeds": args.n_random,
            "null": "eight frozen random directions at each exact selected geometry",
        },
        "quality_gates": {
            "judge_item_completeness_min": base.COMPLETENESS_FLOOR,
            "programmatic_coherence_min": base.COHERENCE_FLOOR,
            "final_cap_hit_fraction_max": base.CAP_REGEN_THRESHOLD,
            "completion_cjk_fraction_max": personas.CJK_FRACTION_CEILING,
            "degenerate_completion_fraction_max": DEGENERATE_FRACTION_CEILING,
            "cap_rule": "regenerate once at 4096 tokens, then exclude if still above 2%",
        },
        "judge": {
            "model": "claude-sonnet-4-5-20250929",
            "draws": args.judge_draws,
            "temperature": 1.0,
            "transport": "Anthropic Message Batches (threshold_base=0)",
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired held-out question",
        },
        "direction_root": str(args.direction_root),
        "direction_bank_fingerprint": personas._direction_bank_fingerprint(args.direction_root),
        "prior_failed_answer_run": {
            "root": str(high_root),
            "summary_sha256": _json_sha256(high_root / "summary.json"),
        },
        "trait_asset_sha256": personas.ASSET_SHA256,
    }


def phase_validate(args: argparse.Namespace) -> None:
    manifest = _manifest(args)
    path = args.out_root / "preregistered_design.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        if any((_stage_dir(args.out_root, stage) / "raw_completions").exists() for stage in ("screen", "confirm")):
            raise RuntimeError("refusing to alter the frozen design after outputs exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)
    print(encoded, end="", flush=True)


def _stage_design(stage: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if stage == "screen":
        return SCREEN_QUESTION_INDICES, SCREEN_SEEDS
    if stage == "confirm":
        return CONFIRM_QUESTION_INDICES, CONFIRM_SEEDS
    raise ValueError(stage)


def _iter_generated(record: dict[str, Any]):
    for seed, seed_record in record["seeds"].items():
        for local_qi, completions in enumerate(seed_record["completions"]):
            original_qi = int(record["q_of_context"][local_qi])
            for draw_index, text in enumerate(completions):
                yield int(seed), local_qi, original_qi, draw_index, text


def _generate_record(
    model,
    tokenizer,
    cell: dict[str, Any],
    contexts: list[dict[str, Any]],
    question_indices: tuple[int, ...],
    generation_seeds: tuple[int, ...],
    hook_make,
    *,
    max_new_tokens: int,
    alphas: dict[str, float],
    stage: str,
) -> dict[str, Any]:
    from explore_persona_space.experiments.issue1415 import steering

    seeds_out: dict[str, Any] = {}
    cap_fractions = []
    for seed in generation_seeds:
        with hook_make() as hook:
            completions = steering.generate_batch(
                model,
                tokenizer,
                contexts,
                n=1,
                hook=hook,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                seed_base=seed,
                render_fn=q35.render_qwen35,
                ids_fn=q35.ids_qwen35,
            )
        coherent = [steering.coherence_check(per_context) for per_context in completions]
        seeds_out[str(seed)] = {
            "completions": completions,
            "coherent_flags": coherent,
            "condition_passes": [steering.condition_passes(flags) for flags in coherent],
        }
        cap_fractions.append(
            base.parent._cap_hit_fraction(completions, tokenizer, max_new_tokens)
        )
    return {
        "cell_id": base.cell_id(cell),
        "cell": dict(cell),
        "alphas": alphas,
        "q_of_context": list(question_indices),
        "seeds": seeds_out,
        "max_new_tokens": max_new_tokens,
        "cap_hit_fraction": float(np.mean(cap_fractions)),
        "design": {
            "model": q35.MODEL_ID,
            "revision": q35.MODEL_REVISION,
            "thinking": "disabled",
            "position": "all_answer_prediction_positions",
            "stage": stage,
            "question_indices": list(question_indices),
            "generation_seeds": list(generation_seeds),
            "draws_per_seed": 1,
        },
    }


def _select_trait_shard(
    cells: list[dict[str, Any]], traits: list[str], shard_id: int, num_shards: int
) -> list[dict[str, Any]]:
    if not 0 <= shard_id < num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    if num_shards == len(traits):
        trait = traits[shard_id]
        return [cell for cell in cells if cell["behavior"] == trait]
    return cells[shard_id::num_shards]


def phase_generate(args: argparse.Namespace, stage: str) -> None:
    q35._require_cuda(f"{stage}-generate")
    question_indices, generation_seeds = _stage_design(stage)
    cells = (
        build_screen_cells(args.traits)
        if stage == "screen"
        else build_confirm_cells(args.out_root, args.traits, n_random=args.n_random)
    )
    shard = _select_trait_shard(cells, args.traits, args.shard_id, args.num_shards)
    raw_dir = _stage_dir(args.out_root, stage) / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rho = personas._load_rho(args)
    model, tokenizer = q35._load_model_and_tokenizer()
    questions = {trait: personas.eval_questions(trait) for trait in args.traits}
    contexts = {
        trait: [
            base.parent._contexts_for_questions(questions[trait])[index]
            for index in question_indices
        ]
        for trait in args.traits
    }
    started = time.time()
    for index, cell in enumerate(shard, 1):
        cid = base.cell_id(cell)
        path = raw_dir / f"{cid}.json"
        if path.exists() and not args.force:
            print(f"[{stage}-generate] {index}/{len(shard)} cached {cid}", flush=True)
            continue
        hook_make, alphas = personas._hook_factory(model, args, cell, rho)
        record = _generate_record(
            model,
            tokenizer,
            cell,
            contexts[cell["behavior"]],
            question_indices,
            generation_seeds,
            hook_make,
            max_new_tokens=q35.MAX_NEW_TOKENS,
            alphas=alphas,
            stage=stage,
        )
        if record["cap_hit_fraction"] > base.CAP_REGEN_THRESHOLD:
            record = _generate_record(
                model,
                tokenizer,
                cell,
                contexts[cell["behavior"]],
                question_indices,
                generation_seeds,
                hook_make,
                max_new_tokens=2 * q35.MAX_NEW_TOKENS,
                alphas=alphas,
                stage=stage,
            )
            record["regenerated_for_cap_hits"] = True
        _write_json(path, record)
        print(
            f"[{stage}-generate] {index}/{len(shard)} {cid} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    _write_json(
        _stage_dir(args.out_root, stage) / f"generate_shard_{args.shard_id}_done.json",
        {
            "stage": stage,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "n_cells": len(shard),
        },
    )


def _looks_degenerate(text: str) -> bool:
    words = [token.lower() for token in _WORD_RE.findall(text)]
    if len(words) < 40:
        return False
    if max(Counter(words).values()) >= max(12, math.ceil(0.30 * len(words))):
        return True
    fourgrams = list(zip(words, words[1:], words[2:], words[3:]))
    if len(fourgrams) >= 40 and len(set(fourgrams)) / len(fourgrams) < 0.25:
        return True
    return False


def _quality_metrics(record: dict[str, Any]) -> dict[str, float]:
    texts = [text for *_meta, text in _iter_generated(record)]
    return {
        "cap_hit_fraction": float(record.get("cap_hit_fraction", 0.0)),
        "cjk_fraction": float(
            np.mean([bool(q35._CJK_RE.search(text)) for text in texts])
        ),
        "degenerate_fraction": float(np.mean([_looks_degenerate(text) for text in texts])),
        "coherence_rate_programmatic": base._coherence_rate(record),
    }


def _quality_exclusions(record: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    metrics = _quality_metrics(record)
    reasons = []
    if metrics["cap_hit_fraction"] > base.CAP_REGEN_THRESHOLD:
        reasons.append("generation_cap_hits")
    if metrics["cjk_fraction"] > personas.CJK_FRACTION_CEILING:
        reasons.append("cjk_language_switching")
    if metrics["degenerate_fraction"] > DEGENERATE_FRACTION_CEILING:
        reasons.append("repetitive_or_degenerate_text")
    if metrics["coherence_rate_programmatic"] < base.COHERENCE_FLOOR:
        reasons.append("programmatic_coherence")
    return metrics, reasons


def _judge_item_id(stage: str, cell_id: str, seed: int, original_qi: int) -> str:
    digest = hashlib.sha256(
        f"{stage}|{cell_id}|{seed}|{original_qi}".encode()
    ).hexdigest()[:28]
    return f"ls_{digest}"


def _placeholder(record: dict[str, Any], metrics: dict[str, float], reasons: list[str]) -> dict[str, Any]:
    n_items = sum(1 for _ in _iter_generated(record))
    return {
        "cell_id": record["cell_id"],
        "cell": record["cell"],
        "judge": {"model": None, "draws": 0, "transport": "not_run_quality_excluded"},
        "degradation_excluded": True,
        "degradation_exclusion_reasons": reasons,
        "accounting": {
            "n_total_draws": 0,
            "n_content_dropped_draws": 0,
            "n_transport_lost_draws": 0,
            "n_api_refusal_draws": 0,
            "n_truncation_dropped_draws": 0,
            "frac_items_complete": 0.0,
            "n_items": n_items,
            "n_items_zero_valid": n_items,
            "per_item_draw_counts": {},
        },
        "per_question_mean_score": [None] * len(record["q_of_context"]),
        "completion_mean_scores": {},
        "mean_score": None,
        **metrics,
    }


def phase_judge(args: argparse.Namespace, stage: str) -> None:
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
    )
    from explore_persona_space.experiments.issue_1739.judging import judge_items_graded

    trait = args.judge_behavior
    if trait is None:
        raise ValueError("--judge-behavior is required")
    stage_root = _stage_dir(args.out_root, stage)
    raw_files = sorted((stage_root / "raw_completions").glob(f"{trait}__*.json"))
    all_cells = (
        build_screen_cells(args.traits)
        if stage == "screen"
        else build_confirm_cells(args.out_root, args.traits, n_random=args.n_random)
    )
    expected = sum(cell["behavior"] == trait for cell in all_cells)
    if len(raw_files) != expected:
        raise RuntimeError(
            f"{stage}/{trait}: expected {expected} raw cells, found {len(raw_files)}"
        )
    judged_dir = stage_root / "judge/judged"
    judged_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Path, dict[str, Any], dict[str, float]]] = []
    for raw_path in raw_files:
        out_path = judged_dir / raw_path.name
        if out_path.exists() and not args.force:
            continue
        record = json.loads(raw_path.read_text())
        metrics, reasons = _quality_exclusions(record)
        if reasons:
            _write_json(out_path, _placeholder(record, metrics, reasons))
            print(f"[{stage}-judge:{trait}] excluded {record['cell_id']}: {','.join(reasons)}")
        else:
            pending.append((out_path, record, metrics))

    if pending:
        rubric = str(personas.load_trait_asset(trait)["eval_prompt"])
        questions = personas.eval_questions(trait)
        items: list[tuple[str, str, str]] = []
        item_meta: dict[str, tuple[str, int]] = {}
        records_by_id = {record["cell_id"]: record for _path, record, _metrics in pending}
        for _path, record, _metrics in pending:
            for seed, local_qi, original_qi, _draw_index, completion in _iter_generated(record):
                item_id = _judge_item_id(stage, record["cell_id"], seed, original_qi)
                if item_id in item_meta:
                    raise RuntimeError(f"duplicate judge item id {item_id}")
                items.append((item_id, questions[original_qi], completion))
                item_meta[item_id] = (record["cell_id"], local_qi)
        result = judge_items_graded(
            items,
            rubric,
            cache_dir=stage_root / "judge/cache" / trait,
            save_raw=stage_root / "judge/raw" / trait,
            n_draws=args.judge_draws,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=base.parent.JUDGE_MAX_TOKENS_2254,
            judge_model=JUDGE_MODEL,
            threshold_base=base.JUDGE_THRESHOLD_BASE_BATCH,
        )
        for out_path, record, metrics in pending:
            cid = record["cell_id"]
            ids = [item_id for item_id, (item_cid, _qi) in item_meta.items() if item_cid == cid]
            per_question: list[list[float]] = [[] for _ in record["q_of_context"]]
            completion_means: dict[str, float | None] = {}
            for item_id in ids:
                scores = result.per_item_scores[item_id]
                value = float(np.mean(scores)) if scores else None
                completion_means[item_id] = value
                if value is not None:
                    per_question[item_meta[item_id][1]].append(value)
            per_q_mean = [float(np.mean(values)) if values else None for values in per_question]
            draw_counts = {item_id: result.per_item_draw_counts[item_id] for item_id in ids}
            n_valid = sum(draw_counts.values())
            n_transport = sum(result.per_item_transport_losses.get(item_id, 0) for item_id in ids)
            n_api_refusal = sum(result.per_item_api_refusals.get(item_id, 0) for item_id in ids)
            n_truncation = sum(result.per_item_truncation_drops.get(item_id, 0) for item_id in ids)
            n_total = len(ids) * args.judge_draws
            n_content = n_total - n_valid - n_transport - n_api_refusal
            n_zero = sum(draw_counts[item_id] == 0 for item_id in ids)
            valid_question_means = [value for value in per_q_mean if value is not None]
            judged = {
                "cell_id": cid,
                "cell": records_by_id[cid]["cell"],
                "judge": {
                    "model": JUDGE_MODEL,
                    "draws": args.judge_draws,
                    "temperature": JUDGE_TEMPERATURE,
                    "max_tokens": base.parent.JUDGE_MAX_TOKENS_2254,
                    "transport": "batch (trait-aggregated; threshold_base=0)",
                },
                "accounting": {
                    "n_total_draws": n_total,
                    "n_content_dropped_draws": n_content,
                    "n_transport_lost_draws": n_transport,
                    "n_api_refusal_draws": n_api_refusal,
                    "n_truncation_dropped_draws": n_truncation,
                    "frac_items_complete": (len(ids) - n_zero) / len(ids),
                    "n_items": len(ids),
                    "n_items_zero_valid": n_zero,
                    "per_item_draw_counts": draw_counts,
                },
                "per_question_mean_score": per_q_mean,
                "completion_mean_scores": completion_means,
                "mean_score": (
                    float(np.mean(valid_question_means)) if valid_question_means else None
                ),
                **metrics,
            }
            _write_json(out_path, judged)
            print(f"[{stage}-judge:{trait}] judged {cid}", flush=True)

        _write_json(
            stage_root / "judge" / f"{trait}_aggregate_accounting.json",
            {
                "stage": stage,
                "trait": trait,
                "n_cells": len(pending),
                "n_items": len(items),
                "n_total_draws": result.n_total_draws,
                "n_content_dropped_draws": result.n_dropped_draws,
                "n_transport_lost_draws": result.n_transport_lost_draws,
                "n_api_refusal_draws": result.n_api_refusal_draws,
                "n_truncation_dropped_draws": result.n_truncation_dropped_draws,
                "frac_items_complete": result.frac_items_complete,
                "batch_cache_dir": str(stage_root / "judge/cache" / trait),
            },
        )
    _write_json(
        stage_root / "judge" / f"{trait}_done.json",
        {"stage": stage, "trait": trait, "n_raw_cells": len(raw_files)},
    )


def _eligible(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = list(row.get("degradation_exclusion_reasons", []))
    completeness = float(row.get("accounting", {}).get("frac_items_complete", 0.0))
    if completeness < base.COMPLETENESS_FLOOR:
        reasons.append("judge_completeness")
    if row.get("mean_score") is None:
        reasons.append("no_trait_score")
    return not reasons, sorted(set(reasons))


def _screen_row(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    eligible, reasons = _eligible(row)
    return {
        "cell_id": row["cell_id"],
        "cell": row["cell"],
        "eligible": eligible,
        "exclusion_reasons": reasons,
        "mean_score": row.get("mean_score"),
        "baseline_mean_score": baseline.get("mean_score"),
        "delta_score": (
            float(row["mean_score"] - baseline["mean_score"])
            if eligible and baseline.get("mean_score") is not None
            else None
        ),
        "cap_hit_fraction": row["cap_hit_fraction"],
        "cjk_fraction": row["cjk_fraction"],
        "degenerate_fraction": row["degenerate_fraction"],
        "frac_items_complete": row["accounting"]["frac_items_complete"],
    }


def _breadth_rank(breadth: str) -> int:
    return {"single": 0, "mid": 1, "all": 2}[breadth]


def _plot_screen(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"single": "#4C78A8", "mid": "#59A14F", "all": "#E07A5F"}
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 12.0), sharex=True)
    xs = np.arange(len(DOSE_SCALES))
    for row_index, trait in enumerate(TRAITS):
        for col_index, method in enumerate(METHODS):
            axis = axes[row_index, col_index]
            candidates = summary["traits"][trait]["methods"][method]["candidates"]
            for breadth in BREADTHS:
                rows = sorted(
                    (row for row in candidates if row["cell"]["breadth"] == breadth),
                    key=lambda row: row["cell"]["dose_scale"],
                )
                values = [row["delta_score"] if row["eligible"] else np.nan for row in rows]
                axis.plot(xs, values, marker="o", label=breadth, color=colors[breadth])
                for x, candidate in zip(xs, rows, strict=True):
                    if not candidate["eligible"]:
                        axis.scatter(x, 0, marker="x", color=colors[breadth], s=35, zorder=4)
            selected = summary["traits"][trait]["methods"][method]
            if selected["status"] == "selected":
                cell = selected["cell"]
                x = list(DOSE_SCALES).index(float(cell["dose_scale"]))
                axis.scatter(
                    x,
                    selected["screen_delta_score"],
                    marker="*",
                    s=160,
                    color="#222222",
                    zorder=5,
                )
            axis.axhline(0, color="#333333", linewidth=0.8)
            axis.grid(axis="y", alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_title(f"{trait.capitalize()} — {method.capitalize()}", loc="left")
            if col_index == 0:
                axis.set_ylabel("Screen trait-score increase")
    for axis in axes[-1]:
        axis.set_xticks(xs, [f"{scale:g}×" for scale in DOSE_SCALES])
        axis.set_xlabel("Fraction of failed answer-only dose")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Qwen3.5-9B answer-only low-strength screen", y=0.997)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.978),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.004,
        "Stars: settings selected on even prompts/seed 42; ×: quality-ineligible",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def phase_screen_reduce(args: argparse.Namespace) -> None:
    stage_root = _stage_dir(args.out_root, "screen")
    summary: dict[str, Any] = {
        "design_sha256": _json_sha256(args.out_root / "preregistered_design.json"),
        "traits": {},
    }
    selection: dict[str, Any] = {
        "status": "frozen_after_screen_before_confirmation_outputs",
        "screen_summary": str(stage_root / "summary.json"),
        "screen_question_indices": list(SCREEN_QUESTION_INDICES),
        "screen_generation_seeds": list(SCREEN_SEEDS),
        "confirm_question_indices": list(CONFIRM_QUESTION_INDICES),
        "confirm_generation_seeds": list(CONFIRM_SEEDS),
        "traits": {},
    }
    for trait in TRAITS:
        judged = {
            path.stem: json.loads(path.read_text())
            for path in sorted((stage_root / "judge/judged").glob(f"{trait}__*.json"))
        }
        expected = 1 + len(METHODS) * len(BREADTHS) * len(DOSE_SCALES)
        if len(judged) != expected:
            raise RuntimeError(f"screen/{trait}: expected {expected} judged cells, found {len(judged)}")
        baseline = judged[f"{trait}__a0"]
        baseline_eligible, baseline_reasons = _eligible(baseline)
        if not baseline_eligible:
            raise RuntimeError(f"screen/{trait}: baseline ineligible: {baseline_reasons}")
        trait_summary = {"baseline_mean_score": baseline["mean_score"], "methods": {}}
        trait_selection = {"methods": {}}
        for method in METHODS:
            candidates = []
            for breadth in BREADTHS:
                for scale in DOSE_SCALES:
                    cid = base.cell_id(_signal_cell(trait, method, breadth, scale))
                    candidates.append(_screen_row(judged[cid], baseline))
            eligible = [row for row in candidates if row["eligible"]]
            if not eligible:
                method_row = {"status": "no_quality_eligible_cell", "candidates": candidates}
                selected_row = {"status": "no_quality_eligible_cell", "cell": None}
            else:
                winner = max(
                    eligible,
                    key=lambda row: (
                        float(row["delta_score"]),
                        -float(row["cell"]["dose_scale"]),
                        -_breadth_rank(row["cell"]["breadth"]),
                    ),
                )
                method_row = {
                    "status": "selected",
                    "selected_cell_id": winner["cell_id"],
                    "cell": winner["cell"],
                    "screen_delta_score": winner["delta_score"],
                    "candidates": candidates,
                }
                selected_row = {
                    "status": "selected",
                    "selected_cell_id": winner["cell_id"],
                    "cell": winner["cell"],
                    "screen_delta_score": winner["delta_score"],
                }
            trait_summary["methods"][method] = method_row
            trait_selection["methods"][method] = selected_row
        summary["traits"][trait] = trait_summary
        selection["traits"][trait] = trait_selection
    _write_json(stage_root / "summary.json", summary)
    selection["screen_summary_sha256"] = _json_sha256(stage_root / "summary.json")
    selection_path = stage_root / "confirmation_selection.json"
    if selection_path.exists() and any(
        (_stage_dir(args.out_root, "confirm") / "raw_completions").glob("*.json")
    ):
        existing = json.loads(selection_path.read_text())
        if existing != selection:
            raise RuntimeError("refusing to change selection after confirmation outputs exist")
    _write_json(selection_path, selection)
    _plot_screen(summary, args.screen_fig_path)
    print(f"wrote {stage_root / 'summary.json'}", flush=True)
    print(f"wrote {selection_path}", flush=True)
    print(f"wrote {args.screen_fig_path}", flush=True)


def _paired_bootstrap(
    signal_q: np.ndarray,
    baseline_q: np.ndarray,
    random_q: list[np.ndarray],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_questions = len(signal_q)
    indices = rng.integers(0, n_questions, size=(BOOTSTRAP_DRAWS, n_questions))
    signal_boot = np.mean((signal_q - baseline_q)[indices], axis=1)
    random_boot = np.concatenate(
        [np.mean((row - baseline_q)[indices], axis=1) for row in random_q]
    )
    return signal_boot, random_boot


def _plot_confirm(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"diffmean": "#4C78A8", "probe": "#E07A5F"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharey=True)
    for axis, trait in zip(axes.flat, TRAITS, strict=True):
        for x, method in enumerate(METHODS):
            row = summary["traits"][trait]["methods"][method]
            if row["status"] != "ok":
                axis.text(
                    x,
                    1.2,
                    "quality\nfailed",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#666666",
                )
                continue
            random_points = row["chance"]["per_direction_point_deltas"]
            offsets = np.linspace(-0.16, 0.16, len(random_points))
            axis.scatter(
                x + offsets,
                random_points,
                color="#999999",
                alpha=0.75,
                s=22,
                label="random directions" if x == 0 else None,
            )
            value = row["delta_score"]
            lo, hi = row["ci_95"]
            axis.errorbar(
                x,
                value,
                yerr=[[value - lo], [hi - value]],
                fmt="o",
                color=colors[method],
                markersize=8,
                capsize=4,
                linewidth=1.5,
                zorder=5,
            )
            axis.text(x, value, f"  {value:+.1f}", va="center", fontsize=8.5)
        axis.axhline(0, color="#222222", linewidth=0.9)
        axis.set_xticks([0, 1], ["DiffMean", "Probe"])
        axis.set_title(trait.capitalize(), loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.supylabel("Held-out trait-score increase vs baseline")
    fig.suptitle("Qwen3.5-9B low-dose answer-only steering: held-out confirmation")
    fig.text(
        0.5,
        0.012,
        "Colored points/whiskers: learned direction and paired-question 95% CI; gray: 8 matched random directions",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def phase_confirm_reduce(args: argparse.Namespace) -> None:
    stage_root = _stage_dir(args.out_root, "confirm")
    selection = _load_selection(args.out_root)
    summary: dict[str, Any] = {
        "design_sha256": _json_sha256(args.out_root / "preregistered_design.json"),
        "selection_sha256": _json_sha256(
            _stage_dir(args.out_root, "screen") / "confirmation_selection.json"
        ),
        "question_indices": list(CONFIRM_QUESTION_INDICES),
        "generation_seeds": list(CONFIRM_SEEDS),
        "traits": {},
    }
    for trait_index, trait in enumerate(TRAITS):
        judged = {
            path.stem: json.loads(path.read_text())
            for path in sorted((stage_root / "judge/judged").glob(f"{trait}__*.json"))
        }
        baseline = judged[f"{trait}__a0"]
        baseline_ok, baseline_reasons = _eligible(baseline)
        if not baseline_ok:
            raise RuntimeError(f"confirm/{trait}: baseline ineligible: {baseline_reasons}")
        baseline_q = np.asarray(baseline["per_question_mean_score"], dtype=float)
        trait_out: dict[str, Any] = {
            "baseline_mean_score": baseline["mean_score"],
            "methods": {},
        }
        for method_index, method in enumerate(METHODS):
            selected = selection["traits"][trait]["methods"][method]
            if selected["status"] != "selected":
                trait_out["methods"][method] = {"status": selected["status"]}
                continue
            cell = selected["cell"]
            signal = judged[base.cell_id(cell)]
            signal_ok, signal_reasons = _eligible(signal)
            random_rows = []
            for random_seed in range(args.n_random):
                random_cell = _random_cell(cell, random_seed)
                row = judged[base.cell_id(random_cell)]
                ok, reasons = _eligible(row)
                if ok:
                    random_rows.append(row)
            if not signal_ok:
                trait_out["methods"][method] = {
                    "status": "selected_signal_quality_failed",
                    "cell": cell,
                    "exclusion_reasons": signal_reasons,
                    "n_eligible_random_directions": len(random_rows),
                }
                continue
            if len(random_rows) < 4:
                trait_out["methods"][method] = {
                    "status": "insufficient_quality_eligible_random_directions",
                    "cell": cell,
                    "n_eligible_random_directions": len(random_rows),
                }
                continue
            signal_q = np.asarray(signal["per_question_mean_score"], dtype=float)
            random_q = [np.asarray(row["per_question_mean_score"], dtype=float) for row in random_rows]
            signal_boot, random_boot = _paired_bootstrap(
                signal_q,
                baseline_q,
                random_q,
                seed=BOOTSTRAP_SEED + trait_index * 10 + method_index,
            )
            signal_delta = float(np.mean(signal_q - baseline_q))
            random_points = [float(np.mean(row - baseline_q)) for row in random_q]
            chance = {
                "n_random_directions": len(random_rows),
                "per_direction_point_deltas": random_points,
                "p2_5": float(np.quantile(random_boot, 0.025)),
                "p50": float(np.quantile(random_boot, 0.5)),
                "p97_5": float(np.quantile(random_boot, 0.975)),
                "construction": "pool matched random-direction x paired-question bootstrap draws",
            }
            trait_out["methods"][method] = {
                "status": "ok",
                "cell": cell,
                "selected_cell_id": selected["selected_cell_id"],
                "screen_delta_score": selected["screen_delta_score"],
                "delta_score": signal_delta,
                "ci_95": [
                    float(np.quantile(signal_boot, 0.025)),
                    float(np.quantile(signal_boot, 0.975)),
                ],
                "chance": chance,
                "excess_over_null_p97_5": signal_delta - chance["p97_5"],
                "clears_null_p97_5": bool(signal_delta > chance["p97_5"]),
                "exceeds_all_random_direction_points": bool(signal_delta > max(random_points)),
                "signal_quality": {
                    "cap_hit_fraction": signal["cap_hit_fraction"],
                    "cjk_fraction": signal["cjk_fraction"],
                    "degenerate_fraction": signal["degenerate_fraction"],
                    "frac_items_complete": signal["accounting"]["frac_items_complete"],
                },
            }
        summary["traits"][trait] = trait_out
    _write_json(stage_root / "summary.json", summary)
    _plot_confirm(summary, args.confirm_fig_path)
    print(f"wrote {stage_root / 'summary.json'}", flush=True)
    print(f"wrote {args.confirm_fig_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "validate",
            "screen-generate",
            "screen-judge",
            "screen-reduce",
            "confirm-generate",
            "confirm-judge",
            "confirm-reduce",
        ),
    )
    parser.add_argument("--traits", nargs="+", choices=TRAITS, default=list(TRAITS))
    parser.add_argument("--judge-behavior", choices=TRAITS)
    parser.add_argument("--judge-draws", type=int, default=5)
    parser.add_argument("--n-random", type=int, default=N_RANDOM_DIRECTIONS)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--direction-root", type=Path, default=DIRECTION_ROOT)
    parser.add_argument("--screen-fig-path", type=Path, default=SCREEN_FIG_PATH)
    parser.add_argument("--confirm-fig-path", type=Path, default=CONFIRM_FIG_PATH)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_root = args.out_root.resolve()
    args.direction_root = args.direction_root.resolve()
    args.screen_fig_path = args.screen_fig_path.resolve()
    args.confirm_fig_path = args.confirm_fig_path.resolve()
    if args.n_random < 4:
        raise ValueError("--n-random must be at least 4")
    dispatch = {
        "validate": lambda: phase_validate(args),
        "screen-generate": lambda: phase_generate(args, "screen"),
        "screen-judge": lambda: phase_judge(args, "screen"),
        "screen-reduce": lambda: phase_screen_reduce(args),
        "confirm-generate": lambda: phase_generate(args, "confirm"),
        "confirm-judge": lambda: phase_judge(args, "confirm"),
        "confirm-reduce": lambda: phase_confirm_reduce(args),
    }
    dispatch[args.phase]()


if __name__ == "__main__":
    main()
