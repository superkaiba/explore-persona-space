#!/usr/bin/env python3
"""Large-sample low-answer-dose replication for issue #2254.

This is a prospective follow-up to the multitype context-preference run.  It
holds each target's answer-layer breadth fixed, tests answer dose scales 1/16
and 1/8, regenerates the matched context operating point, and uses fresh
anchors and four matched random directions.  Every cell uses the six held-out
odd-indexed questions and eight new seeds (47--54), for 48 generations per
cell and 8,976 generations over the frozen 187-cell grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.issue2254_multitype_context_preference as parent  # noqa: E402
import scripts.issue2254_probe_context_qwen35 as q35  # noqa: E402

EXPERIMENT = "issue2254_low_dose_large_replication_qwen35"
STAGE = "low-dose-large"
OUT_ROOT = REPO_ROOT / "eval_results/issue_2254/low_dose_large_replication_qwen35"
PARENT_ROOT = REPO_ROOT / "eval_results/issue_2254/multitype_context_preference_qwen35"
DEFAULT_DIRECTIONS_ROOT = PARENT_ROOT / "directions"
TARGETS = parent.TARGETS
PERSONAS = parent.PERSONAS
PAIR_INDICES = parent.CONFIRM_INDICES
GENERATION_SEEDS = tuple(range(47, 55))
ANSWER_DOSES = (1 / 16, 1 / 8)
RANDOM_SEEDS = tuple(range(4))
MIN_RANDOM_DIRECTIONS = 3

# The ten selected breadths are inherited from the score-blind screen.  The
# previously excluded query_topic target is prospectively assigned all-layer
# breadth; no query-topic signal score was used to choose it.
ANSWER_BREADTH = {
    "optimistic": "mid",
    "impolite": "all",
    "apathetic": "all",
    "humorous": "all",
    "query_topic": "all",
    "prior_topic": "mid",
    "response_theme": "mid",
    "format_policy": "all",
    "retrievable_fact": "single",
    "icl_task": "all",
    "user_expertise": "single",
}

# Matched context operating points are inherited from the original screen.
# query_topic was never selected, so all-layer/full-scale is fixed here before
# this replication generates any model output.
CONTEXT_POINTS = {
    "optimistic": ("all", 1.0),
    "impolite": ("single", 1 / 16),
    "apathetic": ("mid", 1 / 16),
    "humorous": ("all", 1.0),
    "query_topic": ("all", 1.0),
    "prior_topic": ("single", 1 / 16),
    "response_theme": ("single", 1 / 16),
    "format_policy": ("single", 1.0),
    "retrievable_fact": ("all", 1.0),
    "icl_task": ("mid", 1.0),
    "user_expertise": ("mid", 1 / 4),
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_hash(paths: Iterable[Path], *, base: Path) -> str:
    rows = []
    for path in sorted(paths):
        rows.append(f"{path.relative_to(base)} {_sha256(path)}\n")
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def build_cells(targets: Iterable[str] = TARGETS) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for target in targets:
        cells.extend((parent._anchor_cell(target, "a"), parent._anchor_cell(target, "b")))
        breadth, scale = CONTEXT_POINTS[target]
        context = parent._signal_cell(target, "context", breadth, scale)
        cells.append(context)
        cells.extend(parent._random_cell(context, seed) for seed in RANDOM_SEEDS)
        for dose in ANSWER_DOSES:
            answer = parent._signal_cell(target, "answer", ANSWER_BREADTH[target], dose)
            cells.append(answer)
            cells.extend(parent._random_cell(answer, seed) for seed in RANDOM_SEEDS)
    if len({parent.cell_id(cell) for cell in cells}) != len(cells):
        raise RuntimeError("low-dose cell ids are not unique")
    return cells


def _target_cells(target: str) -> dict[str, Any]:
    breadth, scale = CONTEXT_POINTS[target]
    return {
        "context": parent._signal_cell(target, "context", breadth, scale),
        **{
            f"answer_s{parent._float_slug(dose)}": parent._signal_cell(
                target, "answer", ANSWER_BREADTH[target], dose
            )
            for dose in ANSWER_DOSES
        },
    }


def _design(args: argparse.Namespace) -> dict[str, Any]:
    assets = PARENT_ROOT / "inputs/frozen_target_assets.json"
    direction_paths = list(args.directions_root.glob("*_diffmean_L*.pt"))
    fit = args.directions_root / "fit_report.json"
    if len(direction_paths) != len(TARGETS) * q35.N_LAYERS or not fit.exists():
        raise RuntimeError(
            f"expected {len(TARGETS) * q35.N_LAYERS} directions plus fit report in "
            f"{args.directions_root}, found {len(direction_paths)}"
        )
    return {
        "status": "frozen_before_replication_outputs",
        "experiment": EXPERIMENT,
        "model": q35.MODEL_ID,
        "revision": q35.MODEL_REVISION,
        "execution_script_sha256": _sha256(Path(__file__)),
        "execution_test_sha256": _sha256(
            REPO_ROOT / "tests/test_issue2254_low_dose_large_replication.py"
        ),
        "parent_assets_sha256": _sha256(assets),
        "parent_confirmation_selection_sha256": _sha256(
            PARENT_ROOT / "screen/confirmation_selection.json"
        ),
        "parent_outcome_sha256": _sha256(PARENT_ROOT / "confirmation_outcome_v1.json"),
        "direction_manifest": {
            "n_direction_files": len(direction_paths),
            "fit_report_sha256": _sha256(fit),
            "combined_sha256": _combined_hash(direction_paths, base=args.directions_root),
        },
        "targets": list(TARGETS),
        "target_classes": {
            target: "persona" if target in PERSONAS else "nonpersona" for target in TARGETS
        },
        "pair_indices": list(PAIR_INDICES),
        "generation_seeds": list(GENERATION_SEEDS),
        "generations_per_cell": len(PAIR_INDICES) * len(GENERATION_SEEDS),
        "answer_dose_scales": list(ANSWER_DOSES),
        "answer_breadth": ANSWER_BREADTH,
        "context_points": {
            target: {"breadth": breadth, "dose_scale": scale}
            for target, (breadth, scale) in CONTEXT_POINTS.items()
        },
        "random_directions_per_intervention": len(RANDOM_SEEDS),
        "minimum_eligible_random_directions": MIN_RANDOM_DIRECTIONS,
        "grid": {
            "cells_per_target": 17,
            "total_cells": len(build_cells()),
            "total_generations": len(build_cells()) * len(PAIR_INDICES) * len(GENERATION_SEEDS),
        },
        "quality_gates": {
            "cap_hit_fraction_max": parent.CAP_REGEN_THRESHOLD,
            "cjk_fraction_max": parent.CJK_FRACTION_CEILING,
            "degenerate_fraction_max": parent.DEGENERATE_FRACTION_CEILING,
            "judge_completeness_min": parent.COMPLETENESS_FLOOR,
            "coherence_min": parent.COHERENCE_FLOOR,
            "minimum_jointly_valid_questions_across_primary_arms": (
                parent.MIN_VALID_QUESTIONS
            ),
            "icl_task_gate": "amended nonempty/non-refusal predicate",
        },
        "hypotheses": {
            "h1_cumulative_answer_dose": (
                "The prior answer-arm collapse was caused by repeated cumulative steering; "
                "at 1/16 and 1/8, answer cells will be quality-eligible at materially higher "
                "rates while retaining learned-direction effects above random controls."
            ),
            "h2_persona_context_preference": (
                "If persona information is preferentially steerable at the final context "
                "state, persona-minus-nonpersona context-minus-low-dose-answer F is positive."
            ),
            "h3_nonspecific_context_access": (
                "If context access is not persona-specific, the target-class interaction is "
                "zero or negative and nonpersona targets can match persona context effects."
            ),
        },
        "primary_estimand": (
            "persona-minus-nonpersona difference in context F minus the mean of answer F at "
            "dose scales 1/16 and 1/8"
        ),
        "primary_test": (
            "one-sided exact permutation of four persona labels among eleven targets, plus "
            "nested target-and-question bootstrap; estimable only if all eleven targets pass"
        ),
        "prespecified_sensitivity": (
            "If target attrition occurs, report a clearly labeled retained-target exact "
            "permutation only when at least 3 persona and 5 nonpersona targets remain; never "
            "reinterpret it as confirmatory."
        ),
        "dose_specific_companions": "Report the same interaction separately at 1/16 and 1/8.",
        "scope": (
            "Questions are the prior odd-indexed held-out bank with fresh stochastic seeds. "
            "Inference is conditional on these six prompts; this is a seed replication and "
            "low-dose intervention test, not a new question-bank generalization test."
        ),
    }


def verify_frozen(args: argparse.Namespace) -> dict[str, Any]:
    path = args.out_root / "preregistered_design.json"
    if not path.exists():
        raise RuntimeError(f"missing preregistered design: {path}")
    frozen = json.loads(path.read_text())
    live = _design(args)
    if frozen != live:
        raise RuntimeError("live low-dose design differs from preregistered_design.json")
    return frozen


def phase_validate(args: argparse.Namespace) -> None:
    design = _design(args)
    path = args.out_root / "preregistered_design.json"
    if path.exists() and any((args.out_root / "raw_completions").glob("*.json")):
        if json.loads(path.read_text()) != design:
            raise RuntimeError("refusing to change design after replication outputs exist")
    _write_json(path, design)
    print(json.dumps(design, indent=2))


def _load_rho(args: argparse.Namespace) -> dict[str, float]:
    report = json.loads((args.directions_root / "fit_report.json").read_text())
    return {key: float(value) for key, value in report["rho_pooled_median"].items()}


def _load_direction(args: argparse.Namespace, cell: dict[str, Any], layer: int):
    import torch

    if cell["kind"] == "random":
        digest = hashlib.sha256(
            f"2254-low-dose-large-null:{cell['target']}:{cell['random_seed']}:{layer}".encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vector = parent.base._unit(rng.standard_normal(q35.HIDDEN_DIM))
        return torch.tensor(vector, dtype=torch.float32)
    path = args.directions_root / f"{cell['target']}_diffmean_L{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vector = payload["direction"].float()
    return vector / vector.norm()


def _hook_factory(model, args: argparse.Namespace, cell: dict[str, Any], rho: dict[str, float]):
    from explore_persona_space.experiments.issue1415.steering import DeltaHook
    from explore_persona_space.experiments.issue2254.hooks import multi_layer_delta_hooks

    if cell["kind"] == "anchor":
        return None, {}
    layers = list(q35.LAYER_CONFIGS[cell["layer_config"]])
    directions = [
        _load_direction(args, cell, layer).to(device=model.device, dtype=model.dtype)
        for layer in layers
    ]
    alphas = [(float(cell["c"]) / len(layers)) * rho[f"L{layer}"] for layer in layers]
    all_positions = cell["position"] == "answer"
    if len(layers) == 1:
        def make():
            return DeltaHook(
                model, layers[0], directions[0], alphas[0], all_positions=all_positions
            )
    else:
        def make():
            return multi_layer_delta_hooks(
                model, layers, directions, alphas, all_positions=all_positions
            )
    return make, {
        f"L{layer}": float(alpha)
        for layer, alpha in zip(layers, alphas, strict=True)
    }


def phase_generate(args: argparse.Namespace) -> None:
    verify_frozen(args)
    q35._require_cuda("low-dose-large-generate")
    cells = build_cells(args.targets)
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0,num-shards)")
    shard = cells[args.shard_id :: args.num_shards]
    raw_dir = args.out_root / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    assets = json.loads((PARENT_ROOT / "inputs/frozen_target_assets.json").read_text())["targets"]
    rho = _load_rho(args)
    model, tokenizer = q35._load_model_and_tokenizer()
    started = time.time()
    for index, cell in enumerate(shard, 1):
        cid = parent.cell_id(cell)
        path = raw_dir / f"{cid}.json"
        if path.exists() and not args.force:
            print(f"[low-dose-generate] {index}/{len(shard)} cached {cid}", flush=True)
            continue
        contexts = parent._contexts_for_cell(assets[cell["target"]], cell, PAIR_INDICES)
        hook_make, alphas = _hook_factory(model, args, cell, rho)
        record = parent._generate_record(
            model,
            tokenizer,
            cell,
            contexts,
            PAIR_INDICES,
            GENERATION_SEEDS,
            hook_make,
            max_new_tokens=parent.MAX_NEW_TOKENS,
            alphas=alphas,
            stage=STAGE,
        )
        if record["cap_hit_fraction"] > parent.CAP_REGEN_THRESHOLD:
            record = parent._generate_record(
                model,
                tokenizer,
                cell,
                contexts,
                PAIR_INDICES,
                GENERATION_SEEDS,
                hook_make,
                max_new_tokens=2 * parent.MAX_NEW_TOKENS,
                alphas=alphas,
                stage=STAGE,
            )
            record["regenerated_for_cap_hits"] = True
        _write_json(path, record)
        print(
            f"[low-dose-generate] {index}/{len(shard)} {cid} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    _write_json(
        args.out_root / f"generate_shard_{args.shard_id}_done.json",
        {"shard_id": args.shard_id, "num_shards": args.num_shards, "n": len(shard)},
    )


def phase_judge(args: argparse.Namespace) -> None:
    verify_frozen(args)
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
    )
    from explore_persona_space.experiments.issue_1739.judging import judge_items_graded

    target = args.judge_target
    if target is None:
        raise ValueError("--judge-target is required")
    assets = json.loads((PARENT_ROOT / "inputs/frozen_target_assets.json").read_text())["targets"]
    raw_files = sorted((args.out_root / "raw_completions").glob(f"{target}__*.json"))
    expected = sum(cell["target"] == target for cell in build_cells(args.targets))
    if len(raw_files) != expected:
        raise RuntimeError(f"{target}: expected {expected} raw cells, found {len(raw_files)}")
    judged_dir = args.out_root / "judge/judged"
    judged_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for path in raw_files:
        out = judged_dir / path.name
        if out.exists() and not args.force:
            continue
        record = json.loads(path.read_text())
        metrics, reasons = parent._quality(record)
        if reasons:
            _write_json(
                out,
                {
                    "cell_id": record["cell_id"],
                    "cell": record["cell"],
                    "degradation_excluded": True,
                    "degradation_exclusion_reasons": reasons,
                    "per_question_mean_score": [None] * len(record["q_of_context"]),
                    "mean_score": None,
                    "accounting": {
                        "frac_items_complete": 0.0,
                        "n_items": len(record["q_of_context"]) * len(record["seeds"]),
                    },
                    **parent._quality_protocol(record, "confirm"),
                    **metrics,
                },
            )
        else:
            pending.append((out, record, metrics))
    if not pending:
        _write_json(args.out_root / "judge" / f"{target}_done.json", {"target": target, "n_items": 0})
        return
    items: list[tuple[str, str, str]] = []
    meta: dict[str, tuple[str, int]] = {}
    for _out, record, _metrics in pending:
        for seed, local_qi, original_qi, draw, completion in parent._iter_generated(record):
            item_id = parent._judge_item_id(STAGE, record["cell_id"], seed, original_qi, draw)
            items.append((item_id, assets[target]["eval_pairs"][original_qi]["judge_question"], completion))
            meta[item_id] = (record["cell_id"], local_qi)
    result = judge_items_graded(
        items,
        assets[target]["eval_prompt"],
        cache_dir=args.out_root / "judge/cache" / target,
        save_raw=args.out_root / "judge/raw" / target,
        n_draws=args.judge_draws,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=parent.base.parent.JUDGE_MAX_TOKENS_2254,
        judge_model=JUDGE_MODEL,
        threshold_base=parent.base.JUDGE_THRESHOLD_BASE_BATCH,
    )
    for out, record, metrics in pending:
        cid = record["cell_id"]
        ids = [item_id for item_id, (item_cid, _qi) in meta.items() if item_cid == cid]
        per_q: list[list[float]] = [[] for _ in record["q_of_context"]]
        draw_counts = {}
        for item_id in ids:
            scores = result.per_item_scores[item_id]
            draw_counts[item_id] = result.per_item_draw_counts[item_id]
            if scores:
                per_q[meta[item_id][1]].append(float(np.mean(scores)))
        per_q_mean = [float(np.mean(values)) if values else None for values in per_q]
        valid = [value for value in per_q_mean if value is not None]
        n_zero = sum(count == 0 for count in draw_counts.values())
        _write_json(
            out,
            {
                "cell_id": cid,
                "cell": record["cell"],
                "judge": {"model": JUDGE_MODEL, "draws": args.judge_draws},
                "accounting": {
                    "frac_items_complete": (len(ids) - n_zero) / len(ids),
                    "n_items": len(ids),
                    "n_items_zero_valid": n_zero,
                    "per_item_draw_counts": draw_counts,
                },
                "per_question_mean_score": per_q_mean,
                "mean_score": float(np.mean(valid)) if valid else None,
                **parent._quality_protocol(record, "confirm"),
                **metrics,
            },
        )
    _write_json(
        args.out_root / "judge" / f"{target}_done.json",
        {"target": target, "n_items": len(items)},
    )


def _reduce_intervention(
    judged: dict[str, dict[str, Any]],
    cell: dict[str, Any],
    floor_q: np.ndarray,
    ceiling_q: np.ndarray,
) -> dict[str, Any]:
    signal = judged[parent.cell_id(cell)]
    signal_ok, reasons = parent._eligible(signal)
    out: dict[str, Any] = {
        "cell": cell,
        "cell_id": parent.cell_id(cell),
        "quality_eligible": signal_ok,
        "quality_reasons": reasons,
        "quality": {
            key: signal.get(key)
            for key in (
                "cap_hit_fraction",
                "cjk_fraction",
                "degenerate_fraction",
                "coherence_rate_programmatic",
                "nonempty_nonrefusal_rate",
                "whitespace_token_count_max",
            )
        },
    }
    if not signal_ok:
        out["status"] = "signal_ineligible"
        return out
    signal_f, valid = parent.normalized_f(
        np.asarray(signal["per_question_mean_score"], dtype=float),
        floor_q,
        ceiling_q,
        min_separation=parent.MIN_ANCHOR_SEPARATION,
    )
    random_rows = []
    random_details = []
    for seed in RANDOM_SEEDS:
        random_cell = parent._random_cell(cell, seed)
        row = judged[parent.cell_id(random_cell)]
        ok, random_reasons = parent._eligible(row)
        detail: dict[str, Any] = {
            "random_seed": seed,
            "eligible": ok,
            "reasons": random_reasons,
        }
        if ok:
            f_q, random_valid = parent.normalized_f(
                np.asarray(row["per_question_mean_score"], dtype=float),
                floor_q,
                ceiling_q,
                min_separation=parent.MIN_ANCHOR_SEPARATION,
            )
            if int(random_valid.sum()) >= parent.MIN_VALID_QUESTIONS:
                random_rows.append(f_q)
                detail["mean_f"] = float(np.nanmean(f_q))
            else:
                detail["eligible"] = False
                detail["reasons"] = sorted(set(random_reasons + ["anchor_separation_floor"]))
        random_details.append(detail)
    out["random_controls"] = random_details
    if int(valid.sum()) < parent.MIN_VALID_QUESTIONS:
        out["status"] = "anchor_separation_floor"
        return out
    if len(random_rows) < MIN_RANDOM_DIRECTIONS:
        out["status"] = "insufficient_random_controls"
        out["n_random_eligible"] = len(random_rows)
        return out
    random_stack = np.stack(random_rows)
    random_per_q = np.nanmean(random_stack, axis=0)
    signal_mean = float(np.nanmean(signal_f))
    random_mean = float(np.nanmean(random_per_q))
    out.update(
        {
            "status": "ok",
            "mean_f": signal_mean,
            "per_question_f": [float(v) if np.isfinite(v) else None for v in signal_f],
            "n_valid_questions": int(valid.sum()),
            "n_random_eligible": len(random_rows),
            "random_mean_f": random_mean,
            "random_per_question_f": [
                float(v) if np.isfinite(v) else None for v in random_per_q
            ],
            "learned_minus_random_f": signal_mean - random_mean,
            "exceeds_all_random_points": bool(
                signal_mean > max(float(np.nanmean(row)) for row in random_rows)
            ),
        }
    )
    return out


def phase_reduce(args: argparse.Namespace) -> None:
    design = verify_frozen(args)
    summary: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "design_sha256": _sha256(args.out_root / "preregistered_design.json"),
        "targets": {},
    }
    for target in args.targets:
        judged = {
            path.stem: json.loads(path.read_text())
            for path in sorted(
                (args.out_root / "judge/judged").glob(f"{target}__*.json")
            )
        }
        expected = sum(cell["target"] == target for cell in build_cells(args.targets))
        if len(judged) != expected:
            raise RuntimeError(
                f"{target}: expected {expected} judged cells, found {len(judged)}"
            )
        floor = judged[parent.cell_id(parent._anchor_cell(target, "a"))]
        ceiling = judged[parent.cell_id(parent._anchor_cell(target, "b"))]
        floor_ok, floor_reasons = parent._eligible(floor)
        ceiling_ok, ceiling_reasons = parent._eligible(ceiling)
        target_out: dict[str, Any] = {
            "target_class": "persona" if target in PERSONAS else "nonpersona",
            "information_type": parent.INFORMATION_TYPE[target],
            "anchors": {
                "floor": {"eligible": floor_ok, "reasons": floor_reasons},
                "ceiling": {"eligible": ceiling_ok, "reasons": ceiling_reasons},
            },
            "interventions": {},
        }
        if not floor_ok or not ceiling_ok:
            target_out["status"] = "anchor_ineligible"
            summary["targets"][target] = target_out
            continue
        floor_q = np.asarray(floor["per_question_mean_score"], dtype=float)
        ceiling_q = np.asarray(ceiling["per_question_mean_score"], dtype=float)
        target_out["floor_mean"] = float(np.mean(floor_q))
        target_out["ceiling_mean"] = float(np.mean(ceiling_q))
        target_out["per_question_anchor_separation"] = (ceiling_q - floor_q).tolist()
        for key, cell in _target_cells(target).items():
            target_out["interventions"][key] = _reduce_intervention(
                judged, cell, floor_q, ceiling_q
            )
        target_out["status"] = (
            "ok"
            if all(row.get("status") == "ok" for row in target_out["interventions"].values())
            else "intervention_ineligible"
        )
        summary["targets"][target] = target_out
    summary["design"] = {
        "answer_dose_scales": design["answer_dose_scales"],
        "generations_per_cell": design["generations_per_cell"],
        "random_directions_per_intervention": design["random_directions_per_intervention"],
    }
    _write_json(args.out_root / "reduced_summary.json", summary)


def _interaction_test(values: dict[str, float], per_q: dict[str, np.ndarray]) -> dict[str, Any]:
    retained_personas = frozenset(set(values) & PERSONAS)
    result = parent.exact_label_permutation(values, retained_personas)
    boot = parent._nested_bootstrap(per_q, retained_personas)
    result["bootstrap_ci95"] = [
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    ]
    return result


def analyze_summary(summary: dict[str, Any]) -> dict[str, Any]:
    candidate_targets = [
        target for target, row in summary["targets"].items() if row["status"] == "ok"
    ]
    complete: list[str] = []
    analytical_attrition: dict[str, dict[str, Any]] = {}
    values: dict[str, float] = {}
    per_q: dict[str, np.ndarray] = {}
    dose_values: dict[str, dict[str, float]] = {
        f"s{parent._float_slug(dose)}": {} for dose in ANSWER_DOSES
    }
    dose_q: dict[str, dict[str, np.ndarray]] = {key: {} for key in dose_values}
    for target in candidate_targets:
        rows = summary["targets"][target]["interventions"]
        context_q = np.asarray(rows["context"]["per_question_f"], dtype=float)
        answer_qs = []
        for dose in ANSWER_DOSES:
            key = f"answer_s{parent._float_slug(dose)}"
            answer_qs.append(np.asarray(rows[key]["per_question_f"], dtype=float))
        joint_valid = np.isfinite(context_q)
        for answer_q in answer_qs:
            joint_valid &= np.isfinite(answer_q)
        n_joint_valid = int(joint_valid.sum())
        if n_joint_valid < parent.MIN_VALID_QUESTIONS:
            analytical_attrition[target] = {
                "reason": "insufficient_jointly_valid_questions_across_primary_arms",
                "n_jointly_valid_questions": n_joint_valid,
                "minimum": parent.MIN_VALID_QUESTIONS,
            }
            continue
        complete.append(target)
        context_valid = context_q[joint_valid]
        for dose, answer_q in zip(ANSWER_DOSES, answer_qs, strict=True):
            dkey = f"s{parent._float_slug(dose)}"
            difference = context_valid - answer_q[joint_valid]
            dose_q[dkey][target] = difference
            dose_values[dkey][target] = float(np.mean(difference))
        avg_answer_q = np.mean(
            np.stack([answer_q[joint_valid] for answer_q in answer_qs]), axis=0
        )
        per_q[target] = context_valid - avg_answer_q
        values[target] = float(np.mean(per_q[target]))
    n_persona = sum(target in PERSONAS for target in complete)
    n_nonpersona = len(complete) - n_persona
    all_complete = set(complete) == set(TARGETS)
    retained_allowed = n_persona >= 3 and n_nonpersona >= 5
    if all_complete:
        primary = {"status": "ok", **_interaction_test(values, per_q)}
        label = "confirmatory_primary"
    else:
        primary = {
            "status": "not_estimable",
            "reason": "at least one preregistered target lacked all required eligible cells",
        }
        label = "sensitivity_only" if retained_allowed else "descriptive_only"
    retained = (
        {"status": "ok", **_interaction_test(values, per_q)}
        if retained_allowed
        else {
            "status": "not_estimable",
            "reason": "retained-set floor requires at least 3 persona and 5 nonpersona targets",
        }
    )
    dose_specific = {}
    for key in dose_values:
        dose_specific[key] = (
            {"status": "ok", **_interaction_test(dose_values[key], dose_q[key])}
            if retained_allowed
            else {"status": "not_estimable"}
        )
    return {
        "experiment": EXPERIMENT,
        "inference_label": label,
        "complete_targets": complete,
        "candidate_targets_before_joint_validity_gate": candidate_targets,
        "analytical_attrition": analytical_attrition,
        "n_complete_persona": n_persona,
        "n_complete_nonpersona": n_nonpersona,
        "primary_all_11": primary,
        "retained_target_sensitivity": retained,
        "dose_specific_companions": dose_specific,
        "target_context_preference": values,
        "targets": summary["targets"],
        "quality_survival": {
            "context_signal_eligible": sum(
                row.get("interventions", {}).get("context", {}).get("status") == "ok"
                for row in summary["targets"].values()
            ),
            **{
                f"answer_{key}_eligible": sum(
                    row.get("interventions", {}).get(f"answer_{key}", {}).get("status") == "ok"
                    for row in summary["targets"].values()
                )
                for key in dose_values
            },
        },
        "scope": (
            "Answer steering still edits prefill plus every cached decode state, while context "
            "steering edits one state. Lower per-token coefficients reduce but do not eliminate "
            "the unequal-total-energy limitation."
        ),
    }


def phase_analyze(args: argparse.Namespace) -> None:
    verify_frozen(args)
    summary = json.loads((args.out_root / "reduced_summary.json").read_text())
    result = analyze_summary(summary)
    result["reduced_summary_sha256"] = _sha256(args.out_root / "reduced_summary.json")
    _write_json(args.out_root / "analysis.json", result)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("validate", "generate", "judge", "reduce", "analyze"))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--directions-root", type=Path, default=DEFAULT_DIRECTIONS_ROOT)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--judge-target", choices=TARGETS)
    parser.add_argument("--judge-draws", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_root = args.out_root.resolve()
    args.directions_root = args.directions_root.resolve()
    dispatch = {
        "validate": lambda: phase_validate(args),
        "generate": lambda: phase_generate(args),
        "judge": lambda: phase_judge(args),
        "reduce": lambda: phase_reduce(args),
        "analyze": lambda: phase_analyze(args),
    }
    dispatch[args.phase]()


if __name__ == "__main__":
    main()
