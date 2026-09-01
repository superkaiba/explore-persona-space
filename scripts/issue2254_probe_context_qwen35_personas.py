#!/usr/bin/env python3
"""Qwen3.5 position-matched steering for four additional Persona Vector traits.

This is an extension of ``issue2254_probe_context_qwen35.py`` to the four
additional traits listed in the Persona Vectors appendix: optimistic, impolite,
apathetic, and humorous.  It intentionally does not tune operating points on
these traits.  Before any Qwen3.5 outcome is observed, every trait receives the
same three intervention geometries:

* single: Qwen3.5 layer 23, normalized dose c=2;
* mid: layers 16/20/23, total normalized dose c=2;
* all: all 32 layers, total normalized dose c=4.

DiffMean, probe, and fresh random directions are compared at those identical
points.  The default intervention edits only the final context token;
``--position answer`` uses issue #2254's established answer-token convention
(``all_positions=True``: the prefill slot predicting answer token 1, followed
by every cached decode position).  Extraction and evaluation use the cached
5-pair, 20/20 disjoint paper-template artifacts in ``data/issue_779/artifacts``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.issue2254_probe_context_followup as base  # noqa: E402
import scripts.issue2254_probe_context_qwen35 as q35  # noqa: E402

TRAITS = ("optimistic", "impolite", "apathetic", "humorous")
ASSET_DIR = REPO_ROOT / "data/issue_779/artifacts"
ASSET_SHA256 = {
    "optimistic": "f37172fa5d9c4cec0d9ccd2253b5f6755afb6fca1c5f5b2a042e2caed9afa113",
    "impolite": "48d7ac568566b67480b8f104362672e75c66b50d2840e5ad3d7130316204cda6",
    "apathetic": "760eb042426b42f14864ea49b2c9ace62f10293cecbfc543204ffe31232802ea",
    "humorous": "6755dc40494ef3b8a23bc078ea6bebe8d69c851f90d3253284ae650e345ffe21",
}

# Frozen before Qwen3.5 capture/generation.  L23 is the depth-matched analogue
# of the canonical Qwen2.5 layer-20 Persona Vector read.  The other two points
# reuse the prior Qwen3.5 experiment's mid stack and successful all-layer dose.
OPERATING_POINTS: dict[str, dict[str, Any]] = {
    "single": {
        "layer_config": "L23",
        "c": 2.0,
        "source": "preregistered:depth-matched-q25-L20",
    },
    "mid": {
        "layer_config": "mid",
        "c": 2.0,
        "source": "preregistered:prior-q35-hallucination-mid",
    },
    "all": {
        "layer_config": "all",
        "c": 4.0,
        "source": "preregistered:prior-q35-sycophancy-all",
    },
}

OUT_ROOT = REPO_ROOT / "eval_results/issue_2254/context_probe_qwen35_additional_personas"
FIG_PATH = REPO_ROOT / "artifacts/issue2254/context_probe_qwen35_additional_personas_vs_chance.png"
COMPARE_FIG_PATH = REPO_ROOT / "artifacts/issue2254/qwen35_additional_personas_answer_vs_context.png"
CJK_FRACTION_CEILING = 0.20


def _write_json(path: Path, payload: Any) -> None:
    base._write_json(path, payload)


def _position_description(position: str) -> str:
    if position == "context":
        return "last_context_token_only"
    if position == "answer":
        return "all_answer_prediction_positions"
    raise ValueError(f"unknown steering position {position!r}")


def _direction_bank_fingerprint(direction_root: Path) -> dict[str, Any]:
    direction_dir = direction_root / "directions"
    fit_path = direction_dir / "fit_report.json"
    tensor_paths = sorted(direction_dir.glob("*_L*.pt"))
    expected = len(TRAITS) * len(base.METHODS) * q35.N_LAYERS
    if len(tensor_paths) != expected:
        raise RuntimeError(
            f"expected {expected} frozen direction tensors in {direction_dir}, "
            f"found {len(tensor_paths)}"
        )
    aggregate = hashlib.sha256()
    for path in tensor_paths:
        aggregate.update(path.name.encode())
        aggregate.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "n_direction_tensors": len(tensor_paths),
        "fit_report_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        "direction_tensors_aggregate_sha256": aggregate.hexdigest(),
    }


def load_trait_asset(trait: str) -> dict[str, Any]:
    if trait not in TRAITS:
        raise ValueError(f"unknown trait {trait!r}; expected one of {TRAITS}")
    path = ASSET_DIR / f"{trait}.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ASSET_SHA256[trait]:
        raise RuntimeError(f"asset hash drift for {trait}: {digest} != {ASSET_SHA256[trait]}")
    asset = json.loads(path.read_text())
    pairs = asset.get("instruction", [])
    extraction = asset.get("extraction_questions", [])
    evaluation = asset.get("eval_questions", [])
    rubric = asset.get("eval_prompt", "")
    if len(pairs) != 5 or len(extraction) != 20 or len(evaluation) != 20:
        raise RuntimeError(
            f"{trait}: expected 5 pairs and 20/20 questions, got "
            f"{len(pairs)} and {len(extraction)}/{len(evaluation)}"
        )
    if set(extraction) & set(evaluation):
        raise RuntimeError(f"{trait}: extraction/evaluation questions overlap")
    if any(set(pair) != {"pos", "neg"} for pair in pairs):
        raise RuntimeError(f"{trait}: malformed instruction pair")
    if "{question}" not in rubric or "{answer}" not in rubric:
        raise RuntimeError(f"{trait}: rubric is missing question/answer slots")
    return asset


def eval_questions(trait: str) -> list[str]:
    return list(load_trait_asset(trait)["eval_questions"])


def extraction_contexts(trait: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    asset = load_trait_asset(trait)
    pairs = asset["instruction"]
    questions = asset["extraction_questions"]
    positive = [{"system": pair["pos"], "user": q} for pair in pairs for q in questions]
    negative = [{"system": pair["neg"], "user": q} for pair in pairs for q in questions]
    return positive, negative


def phase_validate(args: argparse.Namespace) -> None:
    traits = {}
    for trait in args.traits:
        asset = load_trait_asset(trait)
        traits[trait] = {
            "asset": str((ASSET_DIR / f"{trait}.json").relative_to(REPO_ROOT)),
            "sha256": ASSET_SHA256[trait],
            "n_instruction_pairs": len(asset["instruction"]),
            "n_extraction_questions": len(asset["extraction_questions"]),
            "n_eval_questions": len(asset["eval_questions"]),
            "question_overlap": len(
                set(asset["extraction_questions"]) & set(asset["eval_questions"])
            ),
        }
    direction_fingerprint = (
        _direction_bank_fingerprint(args.direction_root)
        if args.position == "answer"
        else None
    )
    context_summary = OUT_ROOT / "summary.json"
    pre_pilot_manifest = args.out_root / "preregistered_design_pre_pilot.json"
    answer_pilot = (
        args.out_root
        / "raw_completions"
        / "optimistic__diffmean__all__all__c4p0.json"
    )
    answer_design_amended = args.position == "answer" and answer_pilot.exists()
    design = {
        "status": (
            "amended_after_one_answer_only_pilot_before_remaining_outcomes"
            if answer_design_amended
            else "frozen_before_answer_only_outcomes"
            if args.position == "answer"
            else "frozen_before_qwen35_outcomes"
        ),
        "design_history": (
            {
                "pre_pilot_manifest": str(pre_pilot_manifest),
                "pre_pilot_manifest_sha256": hashlib.sha256(
                    pre_pilot_manifest.read_bytes()
                ).hexdigest(),
                "pilot_cell": "optimistic__diffmean__all__all__c4p0",
                "amendment": (
                    "made the existing 2% post-regeneration cap threshold and a 20% "
                    "English-prompt CJK-switch threshold explicit selection-eligibility "
                    "gates; no grid, direction, dose, prompt, seed, judge, or contrast changed"
                ),
            }
            if answer_design_amended
            else None
        ),
        "model": q35.MODEL_ID,
        "revision": q35.MODEL_REVISION,
        "transformers": q35.TRANSFORMERS_PIN,
        "traits": traits,
        "methods": list(base.METHODS),
        "position": _position_description(args.position),
        "position_implementation": (
            "DeltaHook(all_positions=True): edit the prefill slot predicting answer token 1 "
            "and every subsequent one-token KV-cache decode position"
            if args.position == "answer"
            else "DeltaHook(all_positions=False): edit only the final prompt-token prefill slot"
        ),
        "direction_sign": "positive_trait",
        "direction_root": str(args.direction_root),
        "direction_bank_fingerprint": direction_fingerprint,
        "operating_points": OPERATING_POINTS,
        "generation_seeds": list(base.GEN_SEEDS),
        "random_direction_seeds": args.n_random,
        "judge_draws": args.judge_draws,
        "selection": "max over the same three frozen breadth cells",
        "selection_quality_gate": {
            "judge_item_completeness_min": base.COMPLETENESS_FLOOR,
            "programmatic_coherence_min": base.COHERENCE_FLOOR,
            "final_cap_hit_fraction_max": base.CAP_REGEN_THRESHOLD,
            "completion_cjk_fraction_max": CJK_FRACTION_CEILING,
            "cap_rule": (
                "regenerate once at twice the token budget when the initial cap-hit "
                "fraction exceeds the ceiling; exclude if the final fraction still exceeds it"
            ),
            "cjk_rule": (
                "exclude cells when more than 20% of English-prompted completions contain "
                "CJK characters; retain raw outputs and apply the same rule to both positions"
            ),
        },
        "chance": "matched fresh random unit direction, then identical breadth selection",
        "matched_context_comparison": (
            {
                "context_root": str(OUT_ROOT),
                "context_summary_sha256": hashlib.sha256(
                    context_summary.read_bytes()
                ).hexdigest(),
                "contrast": "answer_selected_delta_minus_context_selected_delta",
                "selection": "reselect max eligible breadth within each position per paired-question bootstrap",
                "fixed_factors": (
                    "direction, trait assets, prompts, model revision, generation seeds, layer "
                    "configurations, coefficients, random-direction seeds, judge, and selection rule"
                ),
            }
            if args.position == "answer"
            else None
        ),
    }
    _write_json(args.out_root / "preregistered_design.json", design)
    print(json.dumps(design, indent=2), flush=True)


def _compute_rho(model, tokenizer, traits: list[str]) -> tuple[dict[str, Any], dict[str, float]]:
    layers = list(q35.ALL_LAYERS)
    per_trait: dict[str, Any] = {}
    pooled: dict[int, list[float]] = {layer: [] for layer in layers}
    for trait in traits:
        contexts = base.parent._contexts_for_questions(eval_questions(trait))
        acts = q35.capture_last_context(model, tokenizer, contexts, layers, batch_size=1)
        norms = np.linalg.norm(acts.astype(np.float64), axis=2)
        per_trait[trait] = {f"L{layer}": float(np.median(norms[:, layer])) for layer in layers}
        for layer in layers:
            pooled[layer].extend(float(value) for value in norms[:, layer])
    pooled_median = {
        f"L{layer}": float(np.median(values)) for layer, values in pooled.items()
    }
    return per_trait, pooled_median


def phase_capture(args: argparse.Namespace) -> None:
    q35._require_cuda("capture")
    import torch

    model, tokenizer = q35._load_model_and_tokenizer()
    layers = list(q35.ALL_LAYERS)
    direction_dir = args.out_root / "directions"
    direction_dir.mkdir(parents=True, exist_ok=True)
    rho_per_trait, rho_pooled = _compute_rho(model, tokenizer, args.traits)
    report: dict[str, Any] = {
        "model": q35.MODEL_ID,
        "revision": q35.MODEL_REVISION,
        "transformers": q35.TRANSFORMERS_PIN,
        "thinking": "disabled",
        "n_layers": q35.N_LAYERS,
        "hidden_dim": q35.HIDDEN_DIM,
        "rho_median_last_context_token": rho_per_trait,
        "rho_pooled_median": rho_pooled,
        "operating_points": OPERATING_POINTS,
        "traits": {},
    }
    for trait in args.traits:
        positive, negative = extraction_contexts(trait)
        acts = q35.capture_last_context(
            model,
            tokenizer,
            positive + negative,
            layers,
            batch_size=args.capture_batch,
        )
        n_positive = len(positive)
        diffmean = base.parent.diff_of_means_direction(acts[:n_positive], acts[n_positive:])
        probe, probe_report = base.fit_probe_directions(acts)
        cosines = np.sum(diffmean * probe, axis=1)
        for method, matrix in (("diffmean", diffmean), ("probe", probe)):
            for layer, vector in enumerate(matrix):
                torch.save(
                    {
                        "direction": torch.tensor(vector, dtype=torch.float32),
                        "trait": trait,
                        "method": method,
                        "layer": layer,
                        "model": q35.MODEL_ID,
                        "revision": q35.MODEL_REVISION,
                    },
                    direction_dir / f"{trait}_{method}_L{layer}.pt",
                )
        report["traits"][trait] = {
            "probe": probe_report,
            "cosine_probe_vs_diffmean_per_layer": [float(value) for value in cosines],
            "cosine_summary": {
                "min": float(cosines.min()),
                "median": float(np.median(cosines)),
                "max": float(cosines.max()),
            },
        }
        auc_median = np.median(
            [row["heldout_auc_mean"] for row in probe_report["layers"]]
        )
        print(
            f"[capture] {trait}: probe AUC median={auc_median:.3f}; "
            f"probe/diffmean cosine median={np.median(cosines):.3f}",
            flush=True,
        )
    _write_json(direction_dir / "fit_report.json", report)


def build_cells(
    traits: tuple[str, ...] | list[str] = TRAITS,
    *,
    n_random: int = base.N_RANDOM_DIRECTIONS,
    position: str = "context",
) -> list[dict[str, Any]]:
    _position_description(position)
    cells: list[dict[str, Any]] = []
    for trait in traits:
        cells.append({"behavior": trait, "kind": "alpha0", "position": position})
        for method in base.METHODS:
            for breadth in base.BREADTHS:
                point = OPERATING_POINTS[breadth]
                cells.append(
                    {
                        "behavior": trait,
                        "kind": "signal",
                        "method": method,
                        "position": position,
                        "breadth": breadth,
                        "layer_config": point["layer_config"],
                        "c": float(point["c"]),
                        "source_cell_id": point["source"],
                    }
                )
        for random_seed in range(n_random):
            for breadth in base.BREADTHS:
                point = OPERATING_POINTS[breadth]
                cells.append(
                    {
                        "behavior": trait,
                        "kind": "random",
                        "method": "random",
                        "position": position,
                        "random_seed": random_seed,
                        "breadth": breadth,
                        "layer_config": point["layer_config"],
                        "c": float(point["c"]),
                        "source_cell_id": point["source"],
                    }
                )
    return cells


def _load_rho(args: argparse.Namespace) -> dict[str, float]:
    report = json.loads((args.direction_root / "directions/fit_report.json").read_text())
    rho = {key: float(value) for key, value in report["rho_pooled_median"].items()}
    if set(rho) != {f"L{layer}" for layer in q35.ALL_LAYERS}:
        raise RuntimeError("Qwen3.5 rho bank is incomplete")
    return rho


def _load_direction(args: argparse.Namespace, cell: dict[str, Any], layer: int):
    import torch

    if cell["kind"] == "random":
        digest = hashlib.sha256(
            (
                f"2254-qwen35-additional-persona-null:{cell['behavior']}:"
                f"{cell['random_seed']}:{layer}"
            ).encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return torch.tensor(base._unit(rng.standard_normal(q35.HIDDEN_DIM)), dtype=torch.float32)
    path = (
        args.direction_root
        / "directions"
        / f"{cell['behavior']}_{cell['method']}_L{layer}.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vector = payload["direction"].float()
    if tuple(vector.shape) != (q35.HIDDEN_DIM,):
        raise RuntimeError(f"bad direction shape in {path}: {tuple(vector.shape)}")
    return vector / vector.norm()


def _hook_factory(model, args: argparse.Namespace, cell: dict[str, Any], rho: dict[str, float]):
    import torch

    from explore_persona_space.experiments.issue1415.steering import DeltaHook
    from explore_persona_space.experiments.issue2254.hooks import multi_layer_delta_hooks

    all_positions = cell["position"] == "answer"
    if cell["kind"] == "alpha0":
        layer = q35.LAYER_CONFIGS[OPERATING_POINTS["single"]["layer_config"]][0]
        zero = torch.zeros(q35.HIDDEN_DIM, dtype=torch.bfloat16, device=model.device)

        def make_zero():
            return DeltaHook(model, layer, zero, 0.0, all_positions=all_positions)

        return make_zero, {f"L{layer}": 0.0}
    layers = list(q35.LAYER_CONFIGS[cell["layer_config"]])
    directions = [
        _load_direction(args, cell, layer).to(device=model.device, dtype=torch.bfloat16)
        for layer in layers
    ]
    alphas = [(float(cell["c"]) / len(layers)) * rho[f"L{layer}"] for layer in layers]
    if len(layers) == 1:

        def make():
            return DeltaHook(
                model,
                layers[0],
                directions[0],
                alphas[0],
                all_positions=all_positions,
            )

    else:

        def make():
            return multi_layer_delta_hooks(
                model,
                layers,
                directions,
                alphas,
                all_positions=all_positions,
            )

    return make, {f"L{layer}": float(alpha) for layer, alpha in zip(layers, alphas, strict=True)}


def _select_generation_shard(
    cells: list[dict[str, Any]],
    traits: list[str],
    *,
    shard_id: int,
    num_shards: int,
    strategy: str,
) -> list[dict[str, Any]]:
    if not 0 <= shard_id < num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    if strategy == "stride":
        return cells[shard_id::num_shards]
    if strategy == "trait":
        if num_shards != len(traits):
            raise ValueError("trait sharding requires num-shards == number of requested traits")
        trait = traits[shard_id]
        return [cell for cell in cells if cell["behavior"] == trait]
    raise ValueError(f"unknown shard strategy {strategy!r}")


def phase_generate(args: argparse.Namespace) -> None:
    q35._require_cuda("generate")
    cells = build_cells(args.traits, n_random=args.n_random, position=args.position)
    shard = _select_generation_shard(
        cells,
        args.traits,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        strategy=args.shard_strategy,
    )
    raw_dir = args.out_root / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rho = _load_rho(args)
    model, tokenizer = q35._load_model_and_tokenizer()
    question_bank = {trait: eval_questions(trait) for trait in args.traits}
    started = time.time()
    for index, cell in enumerate(shard, 1):
        cid = base.cell_id(cell)
        path = raw_dir / f"{cid}.json"
        if path.exists() and not args.force:
            print(f"[generate] {index}/{len(shard)} cached {cid}", flush=True)
            continue
        contexts = base.parent._contexts_for_questions(question_bank[cell["behavior"]])
        make, alphas = _hook_factory(model, args, cell, rho)
        record = q35._generate_cell(
            model,
            tokenizer,
            cell,
            contexts,
            make,
            max_new_tokens=q35.MAX_NEW_TOKENS,
            alphas=alphas,
        )
        record["cell"]["position"] = cell["position"]
        record["design"]["position"] = _position_description(cell["position"])
        if record["cap_hit_fraction"] > q35.CAP_REGEN_THRESHOLD:
            record = q35._generate_cell(
                model,
                tokenizer,
                cell,
                contexts,
                make,
                max_new_tokens=2 * q35.MAX_NEW_TOKENS,
                alphas=alphas,
            )
            record["cell"]["position"] = cell["position"]
            record["design"]["position"] = _position_description(cell["position"])
            record["regenerated_for_cap_hits"] = True
        record["design"]["operating_point_transfer"] = (
            "shared preregistered geometry; no tuning on additional traits"
        )
        _write_json(path, record)
        print(
            f"[generate] {index}/{len(shard)} {cid} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    _write_json(
        args.out_root / f"generate_shard_{args.shard_id}_done.json",
        {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "shard_strategy": args.shard_strategy,
            "n_cells": len(shard),
        },
    )


def phase_judge(args: argparse.Namespace) -> None:
    """Reuse the audited Batch judge while routing to the pinned local trait assets."""
    from explore_persona_space.experiments.issue_1739 import judging

    trait = args.judge_behavior
    if trait is None:
        raise ValueError("--judge-behavior is required for --phase judge")
    judged_dir = args.out_root / "judge/judged"
    judged_dir.mkdir(parents=True, exist_ok=True)
    if args.position == "answer":
        answer_baseline_path = args.out_root / "raw_completions" / f"{trait}__a0.json"
        context_baseline_path = args.context_root / "raw_completions" / f"{trait}__a0.json"
        answer_baseline = json.loads(answer_baseline_path.read_text())
        context_baseline = json.loads(context_baseline_path.read_text())
        answer_texts = [text for *_meta, text in base._iter_generated(answer_baseline)]
        context_texts = [text for *_meta, text in base._iter_generated(context_baseline)]
        if answer_texts != context_texts:
            raise RuntimeError(f"{trait}: zero-dose answer/context completions are not identical")
        baseline_out = judged_dir / f"{trait}__a0.json"
        if not baseline_out.exists():
            context_judged_path = args.context_root / "judge/judged" / f"{trait}__a0.json"
            context_judged = json.loads(context_judged_path.read_text())
            context_judged["cell"]["position"] = "answer"
            context_judged["judge"]["reused_identical_context_baseline"] = True
            context_judged["judge"]["source_sha256"] = hashlib.sha256(
                context_judged_path.read_bytes()
            ).hexdigest()
            _write_json(baseline_out, context_judged)
            print(f"[judge:{trait}] reused text-identical context baseline", flush=True)

        for raw_path in sorted(
            (args.out_root / "raw_completions").glob(f"{trait}__*.json")
        ):
            raw = json.loads(raw_path.read_text())
            if raw["cell"].get("kind") == "alpha0":
                continue
            texts = [text for *_meta, text in base._iter_generated(raw)]
            cjk_fraction = float(np.mean([bool(q35._CJK_RE.search(text)) for text in texts]))
            exclusion_reasons = []
            if float(raw.get("cap_hit_fraction", 0.0)) > base.CAP_REGEN_THRESHOLD:
                exclusion_reasons.append("generation_cap_hits")
            if cjk_fraction > CJK_FRACTION_CEILING:
                exclusion_reasons.append("cjk_language_switching")
            if not exclusion_reasons:
                continue
            out_path = judged_dir / raw_path.name
            if out_path.exists():
                continue
            n_items = len(texts)
            placeholder = {
                "cell_id": raw["cell_id"],
                "cell": raw["cell"],
                "judge": {
                    "model": None,
                    "draws": 0,
                    "transport": "not_run_degradation_excluded",
                },
                "degradation_excluded": True,
                "degradation_exclusion_reasons": exclusion_reasons,
                "accounting": {
                    "n_total_draws": 0,
                    "n_refusal_draws": 0,
                    "n_api_refusal_draws": 0,
                    "n_content_dropped_draws": 0,
                    "n_transport_lost_draws": 0,
                    "n_truncation_dropped_draws": 0,
                    "frac_items_complete": 0.0,
                    "n_items": n_items,
                    "n_items_zero_valid": n_items,
                    "per_item_draw_counts": {},
                },
                "per_question_mean_score": [None] * len(raw["q_of_context"]),
                "completion_mean_scores": {},
                "mean_score": None,
                "coherence_rate_programmatic": base._coherence_rate(raw),
                "cap_hit_fraction": float(raw.get("cap_hit_fraction", 0.0)),
                "cjk_fraction": cjk_fraction,
            }
            _write_json(out_path, placeholder)
            print(
                f"[judge:{trait}] degradation-excluded {raw['cell_id']}: "
                f"{','.join(exclusion_reasons)}",
                flush=True,
            )
    original_questions = base.parent._eval_questions
    original_rubric = judging.load_trait_rubric
    base.parent._eval_questions = eval_questions
    judging.load_trait_rubric = lambda behavior: str(load_trait_asset(behavior)["eval_prompt"])
    try:
        base.phase_judge(args)
    finally:
        base.parent._eval_questions = original_questions
        judging.load_trait_rubric = original_rubric


def _non_cjk_question_scores(args: argparse.Namespace, cell_id_value: str) -> dict[str, Any]:
    raw = json.loads((args.out_root / "raw_completions" / f"{cell_id_value}.json").read_text())
    judged = json.loads((args.out_root / "judge/judged" / f"{cell_id_value}.json").read_text())
    per_question: list[list[float]] = [[] for _ in raw["q_of_context"]]
    kept = 0
    total = 0
    for seed, question_index, draw_index, completion in base._iter_generated(raw):
        total += 1
        if q35._CJK_RE.search(completion):
            continue
        item_id = base._judge_item_id(cell_id_value, seed, question_index, draw_index)
        value = judged["completion_mean_scores"][item_id]
        if value is not None:
            per_question[question_index].append(float(value))
            kept += 1
    question_means = [float(np.mean(values)) if values else None for values in per_question]
    return {
        "question_means": question_means,
        "mean_score": float(
            np.nanmean([np.nan if value is None else value for value in question_means])
        ),
        "kept_scored_completions": kept,
        "total_completions": total,
    }


def _build_audit_report(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    judged_rows = [
        json.loads(path.read_text())
        for path in sorted((args.out_root / "judge/judged").glob("*.json"))
    ]
    accounting_keys = (
        "n_total_draws",
        "n_refusal_draws",
        "n_api_refusal_draws",
        "n_content_dropped_draws",
        "n_transport_lost_draws",
        "n_truncation_dropped_draws",
        "n_items_zero_valid",
    )
    accounting = {
        key: int(sum(int(row["accounting"][key]) for row in judged_rows))
        for key in accounting_keys
    }
    accounting.update(
        {
            "n_cells": len(judged_rows),
            "n_items": int(sum(int(row["accounting"]["n_items"]) for row in judged_rows)),
            "expected_draws": int(
                sum(
                    int(row["accounting"]["n_items"]) * int(row["judge"]["draws"])
                    for row in judged_rows
                )
            ),
            "n_valid_draws": int(
                sum(
                    sum(int(value) for value in row["accounting"]["per_item_draw_counts"].values())
                    for row in judged_rows
                )
            ),
            "n_fully_complete_cells": int(
                sum(float(row["accounting"]["frac_items_complete"]) == 1.0 for row in judged_rows)
            ),
            "min_item_completeness": float(
                min(float(row["accounting"]["frac_items_complete"]) for row in judged_rows)
            ),
            "n_reused_identical_baseline_cells": int(
                sum(
                    bool(row["judge"].get("reused_identical_context_baseline"))
                    for row in judged_rows
                )
            ),
            "n_degradation_excluded_cells": int(
                sum(bool(row.get("degradation_excluded")) for row in judged_rows)
            ),
        }
    )
    fit_report = json.loads((args.direction_root / "directions/fit_report.json").read_text())
    trait_audits: dict[str, Any] = {}
    for trait in TRAITS:
        baseline = _non_cjk_question_scores(args, f"{trait}__a0")
        baseline_q = np.asarray(
            [np.nan if value is None else value for value in baseline["question_means"]],
            dtype=np.float64,
        )
        method_audits: dict[str, Any] = {}
        for method in base.METHODS:
            method_summary = summary["behaviors"][trait]["methods"][method]
            cell_id_value = method_summary["selected_cell_id"]
            if cell_id_value is None:
                method_audits[method] = {
                    "status": method_summary["status"],
                    "selected_cell_id": None,
                    "quality_exclusions_by_breadth": {
                        breadth: row["selection_exclusion_reasons"]
                        for breadth, row in method_summary["all_breadths"].items()
                    },
                }
                continue
            selected = _non_cjk_question_scores(args, cell_id_value)
            selected_q = np.asarray(
                [np.nan if value is None else value for value in selected["question_means"]],
                dtype=np.float64,
            )
            method_audits[method] = {
                "selected_cell_id": cell_id_value,
                "non_cjk_mean_score": selected["mean_score"],
                "non_cjk_delta_score_vs_baseline": float(np.nanmean(selected_q - baseline_q)),
                "kept_scored_completions": selected["kept_scored_completions"],
                "total_completions": selected["total_completions"],
            }
        layer_cosines = fit_report["traits"][trait]["cosine_probe_vs_diffmean_per_layer"]
        geometry: dict[str, Any] = {}
        for breadth, point in OPERATING_POINTS.items():
            layers = list(q35.LAYER_CONFIGS[point["layer_config"]])
            values = np.asarray([layer_cosines[layer] for layer in layers], dtype=np.float64)
            geometry[breadth] = {
                "layers": layers,
                "min_cosine": float(values.min()),
                "median_cosine": float(np.median(values)),
                "max_cosine": float(values.max()),
            }
        trait_audits[trait] = {
            "baseline_non_cjk_mean_score": baseline["mean_score"],
            "baseline_kept_scored_completions": baseline["kept_scored_completions"],
            "baseline_total_completions": baseline["total_completions"],
            "selected_methods": method_audits,
            "probe_vs_diffmean_geometry": geometry,
        }
    return {
        "model": q35.MODEL_ID,
        "revision": q35.MODEL_REVISION,
        "judge_accounting": accounting,
        "traits": trait_audits,
    }


def _plot(summary: dict[str, Any], path: Path, *, position: str) -> None:
    import matplotlib.pyplot as plt

    colors = {"diffmean": "#4C78A8", "probe": "#E07A5F"}
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.0), sharey=True)
    for axis, trait in zip(axes.flat, TRAITS, strict=True):
        row = summary["behaviors"][trait]
        chance = row["chance"]
        if chance["status"] == "ok":
            axis.axhspan(chance["p2_5"], chance["p97_5"], color="#B8B8B8", alpha=0.28)
            axis.axhline(chance["p50"], color="#666666", linestyle="--", linewidth=1.1)
        axis.axhline(0, color="#222222", linewidth=0.8)
        for x, method in enumerate(base.METHODS):
            method_row = row["methods"][method]
            if method_row["status"] != "ok":
                axis.text(
                    x,
                    0,
                    "no eligible\ncell",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )
                continue
            selected = method_row["selected_cell"]
            value = selected["delta_score"]
            lo, hi = method_row["selection_inherited_ci_95"]
            axis.bar(x, value, color=colors[method], width=0.62, zorder=2)
            axis.errorbar(
                x,
                value,
                yerr=[[max(0.0, value - lo)], [max(0.0, hi - value)]],
                fmt="none",
                color="#222222",
                capsize=3,
                linewidth=1.1,
                zorder=3,
            )
            axis.text(
                x,
                value + (1.3 if value >= 0 else -2.7),
                f"{value:+.1f}",
                ha="center",
                fontsize=9,
            )
        axis.set_xticks([0, 1], ["DiffMean", "Probe"])
        axis.set_title(trait.capitalize(), loc="left", fontsize=11, fontweight="bold")
        axis.grid(axis="y", alpha=0.18, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        chance_label = (
            f"chance p97.5 = {chance['p97_5']:+.1f}"
            if chance["status"] == "ok"
            else "chance band unavailable"
        )
        axis.text(
            0.02,
            0.98,
            chance_label,
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            color="#555555",
        )
    axes[0, 0].set_ylabel("Trait-score increase vs baseline")
    axes[1, 0].set_ylabel("Trait-score increase vs baseline")
    position_title = (
        "answer-token steering"
        if position == "answer"
        else "last-context-token steering"
    )
    fig.suptitle(f"Qwen3.5-9B {position_title}: additional persona vectors", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "Gray: matched random-direction null after identical three-point selection; whiskers: selection-inherited 95% CI",
        ha="center",
        fontsize=8.1,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def phase_reduce(args: argparse.Namespace) -> None:
    judged_dir = args.out_root / "judge/judged"
    summary: dict[str, Any] = {
        "design": {
            "model": q35.MODEL_ID,
            "revision": q35.MODEL_REVISION,
            "transformers": q35.TRANSFORMERS_PIN,
            "thinking": "disabled",
            "methods": list(base.METHODS),
            "traits": list(TRAITS),
            "position": _position_description(args.position),
            "direction_sign": "positive_trait",
            "operating_points": OPERATING_POINTS,
            "generation_seeds": list(base.GEN_SEEDS),
            "random_direction_seeds": args.n_random,
            "bootstrap_draws": base.N_BOOT,
            "bootstrap_unit": "paired eval question",
        },
        "behaviors": {},
    }
    for trait in TRAITS:
        paths = sorted(judged_dir.glob(f"{trait}__*.json"))
        expected = 1 + 3 * len(base.METHODS) + 3 * args.n_random
        if len(paths) != expected:
            raise RuntimeError(f"{trait}: expected {expected} judged cells, got {len(paths)}")
        rows = [json.loads(path.read_text()) for path in paths]
        summary["behaviors"][trait] = base.reduce_behavior(
            rows,
            trait,
            cjk_fraction_ceiling=CJK_FRACTION_CEILING,
            allow_no_eligible=True,
        )
    _write_json(args.out_root / "summary.json", summary)
    _write_json(args.out_root / "audit_report.json", _build_audit_report(args, summary))
    _plot(summary, args.fig_path, position=args.position)
    print(f"wrote {args.out_root / 'summary.json'}", flush=True)
    print(f"wrote {args.out_root / 'audit_report.json'}", flush=True)
    print(f"wrote {args.fig_path}", flush=True)


def _position_method_boot(
    root: Path,
    summary: dict[str, Any],
    trait: str,
    method: str,
    indices: np.ndarray,
) -> dict[str, Any] | None:
    judged_dir = root / "judge/judged"
    baseline = json.loads((judged_dir / f"{trait}__a0.json").read_text())
    baseline_q = base._q_array(baseline)
    baseline_boot = np.nanmean(baseline_q[indices], axis=1)
    behavior = summary["behaviors"][trait]
    candidates = [
        (cell_id_value, row)
        for cell_id_value, row in behavior["cells"].items()
        if row["cell"].get("kind") == "signal"
        and row["cell"].get("method") == method
        and row["selection_eligible"]
    ]
    if not candidates:
        return None
    candidate_boot: dict[str, np.ndarray] = {}
    per_breadth: dict[str, Any] = {}
    for cell_id_value, cell_summary in candidates:
        judged = json.loads((judged_dir / f"{cell_id_value}.json").read_text())
        q = base._q_array(judged)
        boot = np.nanmean(q[indices], axis=1) - baseline_boot
        candidate_boot[cell_id_value] = boot
        per_breadth[cell_summary["cell"]["breadth"]] = {
            "cell_id": cell_id_value,
            "delta_score": float(np.nanmean(q) - np.nanmean(baseline_q)),
            "bootstrap": boot,
        }
    selected_id = behavior["methods"][method]["selected_cell_id"]
    inherited = np.nanmax(np.stack(list(candidate_boot.values()), axis=1), axis=1)
    selected_summary = behavior["methods"][method]["selected_cell"]
    return {
        "baseline_mean_score": float(np.nanmean(baseline_q)),
        "selected_cell_id": selected_id,
        "selected_breadth": selected_summary["cell"]["breadth"],
        "selected_delta_score": float(selected_summary["delta_score"]),
        "selected_coherence_rate_programmatic": float(
            selected_summary["coherence_rate_programmatic"]
        ),
        "selected_cap_hit_fraction": float(selected_summary["cap_hit_fraction"]),
        "selected_cjk_fraction": float(selected_summary["cjk_fraction"]),
        "selection_inherited_bootstrap": inherited,
        "per_breadth": per_breadth,
    }


def _strip_bootstrap_arrays(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"selection_inherited_bootstrap", "per_breadth"}
    }


def _plot_position_comparison(comparison: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"diffmean": "#4C78A8", "probe": "#E07A5F"}
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.0), sharey=True)
    has_eligible_contrast = False
    for axis, trait in zip(axes.flat, TRAITS, strict=True):
        trait_row = comparison["traits"][trait]
        for x, method in enumerate(base.METHODS):
            row = trait_row["methods"][method]
            if row["status"] != "ok":
                axis.text(
                    x,
                    0.06,
                    "no eligible\nanswer cell",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                continue
            has_eligible_contrast = True
            value = row["answer_minus_context"]["point_difference"]
            lo, hi = row["answer_minus_context"]["ci_95_selection_inherited"]
            axis.bar(x, value, color=colors[method], width=0.62, zorder=2)
            axis.errorbar(
                x,
                value,
                yerr=[[max(0.0, value - lo)], [max(0.0, hi - value)]],
                fmt="none",
                color="#222222",
                capsize=3,
                linewidth=1.1,
                zorder=3,
            )
            axis.text(
                x,
                value + (1.2 if value >= 0 else -2.1),
                f"{value:+.1f}",
                ha="center",
                fontsize=9,
            )
            axis.text(
                x,
                0.02,
                f"ctx {row['context']['selected_breadth']} → ans {row['answer']['selected_breadth']}",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#555555",
            )
        axis.axhline(0, color="#222222", linewidth=0.9)
        axis.set_xticks([0, 1], ["DiffMean", "Probe"])
        axis.set_title(trait.capitalize(), loc="left", fontsize=11, fontweight="bold")
        axis.grid(axis="y", alpha=0.18, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    if not has_eligible_contrast:
        axes[0, 0].set_ylim(-1, 1)
    fig.supylabel("Answer-only minus context-only trait increase", x=0.015, fontsize=10)
    fig.suptitle("Qwen3.5-9B matched steering-position contrast", fontsize=12)
    fig.text(
        0.5,
        0.012,
        "Whiskers: paired-question 95% CI with breadth reselected within each position on every bootstrap draw",
        ha="center",
        fontsize=8.1,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def phase_compare(args: argparse.Namespace) -> None:
    context_root = args.context_root
    answer_root = args.out_root

    def matched_quality_summary(root: Path) -> dict[str, Any]:
        reduced = {"behaviors": {}}
        for trait in TRAITS:
            paths = sorted((root / "judge/judged").glob(f"{trait}__*.json"))
            expected = 1 + 3 * len(base.METHODS) + 3 * args.n_random
            if len(paths) != expected:
                raise RuntimeError(
                    f"{root}/{trait}: expected {expected} judged cells, got {len(paths)}"
                )
            reduced["behaviors"][trait] = base.reduce_behavior(
                [json.loads(path.read_text()) for path in paths],
                trait,
                cjk_fraction_ceiling=CJK_FRACTION_CEILING,
                allow_no_eligible=True,
            )
        return reduced

    context_summary = matched_quality_summary(context_root)
    answer_summary = matched_quality_summary(answer_root)
    comparison: dict[str, Any] = {
        "design": {
            "model": q35.MODEL_ID,
            "revision": q35.MODEL_REVISION,
            "context_root": str(context_root),
            "answer_root": str(answer_root),
            "contrast": "answer_selected_delta_minus_context_selected_delta",
            "selection": (
                "max eligible breadth reselected independently within each position on every "
                "paired-question bootstrap draw"
            ),
            "bootstrap_draws": base.N_BOOT,
            "bootstrap_unit": "paired eval question",
            "matched_quality_gates": {
                "final_cap_hit_fraction_max": base.CAP_REGEN_THRESHOLD,
                "completion_cjk_fraction_max": CJK_FRACTION_CEILING,
            },
        },
        "traits": {},
    }
    for trait in TRAITS:
        indices = base._bootstrap_indices(len(eval_questions(trait)), trait)
        trait_out: dict[str, Any] = {"methods": {}}
        for method in base.METHODS:
            context = _position_method_boot(
                context_root, context_summary, trait, method, indices
            )
            answer = _position_method_boot(answer_root, answer_summary, trait, method, indices)
            if context is None or answer is None:
                trait_out["methods"][method] = {
                    "status": "no_eligible_cell",
                    "context_has_eligible": context is not None,
                    "answer_has_eligible": answer is not None,
                }
                continue
            contrast_boot = (
                answer["selection_inherited_bootstrap"]
                - context["selection_inherited_bootstrap"]
            )
            matched_breadths: dict[str, Any] = {}
            for breadth in base.BREADTHS:
                context_breadth = context["per_breadth"].get(breadth)
                answer_breadth = answer["per_breadth"].get(breadth)
                if context_breadth is None or answer_breadth is None:
                    matched_breadths[breadth] = {
                        "status": "ineligible_in_at_least_one_position"
                    }
                    continue
                boot = answer_breadth["bootstrap"] - context_breadth["bootstrap"]
                matched_breadths[breadth] = {
                    "status": "ok",
                    "point_difference": float(
                        answer_breadth["delta_score"] - context_breadth["delta_score"]
                    ),
                    "ci_95": [
                        float(np.nanquantile(boot, 0.025)),
                        float(np.nanquantile(boot, 0.975)),
                    ],
                }
            trait_out["methods"][method] = {
                "status": "ok",
                "context": _strip_bootstrap_arrays(context),
                "answer": _strip_bootstrap_arrays(answer),
                "answer_minus_context": {
                    "point_difference": float(
                        answer["selected_delta_score"] - context["selected_delta_score"]
                    ),
                    "ci_95_selection_inherited": [
                        float(np.nanquantile(contrast_boot, 0.025)),
                        float(np.nanquantile(contrast_boot, 0.975)),
                    ],
                    "bootstrap_fraction_answer_greater": float(
                        np.nanmean(contrast_boot > 0)
                    ),
                },
                "matched_breadths": matched_breadths,
            }
        comparison["traits"][trait] = trait_out
    comparison_path = answer_root / "answer_vs_context_comparison.json"
    _write_json(comparison_path, comparison)
    _plot_position_comparison(comparison, args.compare_fig_path)
    print(f"wrote {comparison_path}", flush=True)
    print(f"wrote {args.compare_fig_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "validate",
            "envcheck",
            "smoke",
            "capture",
            "generate",
            "judge",
            "reduce",
            "compare",
        ),
    )
    parser.add_argument("--traits", nargs="+", choices=TRAITS, default=list(TRAITS))
    parser.add_argument("--judge-behavior", choices=TRAITS)
    parser.add_argument("--judge-draws", type=int, default=5)
    parser.add_argument("--n-random", type=int, default=base.N_RANDOM_DIRECTIONS)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-strategy", choices=("stride", "trait"), default="stride")
    parser.add_argument("--capture-batch", type=int, default=8)
    parser.add_argument("--position", choices=("context", "answer"), default="context")
    parser.add_argument(
        "--direction-root",
        type=Path,
        help="root containing directions/fit_report.json and the frozen direction tensors",
    )
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--fig-path", type=Path, default=FIG_PATH)
    parser.add_argument("--context-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--compare-fig-path", type=Path, default=COMPARE_FIG_PATH)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_root = args.out_root.resolve()
    args.fig_path = args.fig_path.resolve()
    args.context_root = args.context_root.resolve()
    args.compare_fig_path = args.compare_fig_path.resolve()
    args.direction_root = (
        args.out_root if args.direction_root is None else args.direction_root.resolve()
    )
    # The reused judge expects this attribute name.
    args.behaviors = args.traits
    if args.n_random < 4:
        raise ValueError("--n-random must be at least 4 for reduction")
    dispatch = {
        "validate": phase_validate,
        "envcheck": q35.phase_envcheck,
        "smoke": q35.phase_smoke,
        "capture": phase_capture,
        "generate": phase_generate,
        "judge": phase_judge,
        "reduce": phase_reduce,
        "compare": phase_compare,
    }
    dispatch[args.phase](args)


if __name__ == "__main__":
    main()
