#!/usr/bin/env python3
"""Matched persona-vs-semantic context-vector steering experiment.

All eight target directions are learned at the final rendered context token
from five matched positive/negative system-instruction pairs crossed with a
shared 20-question extraction bank.  The same frozen directions are then used
at either the final context token or all answer prediction positions.  Both
positions search the identical layer-breadth by dose grid on even-numbered
evaluation questions, then confirm their independently selected setting on
odd-numbered questions with matched random-direction controls.

The primary analysis asks whether context-minus-answer target-score lift is
larger for four persona targets than for four semantic targets.  DiffMean is
primary and the ridge-probe direction is a secondary replication.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.issue2254_answer_strength_sweep as answer  # noqa: E402
import scripts.issue2254_probe_context_followup as base  # noqa: E402
import scripts.issue2254_probe_context_qwen35_personas as position  # noqa: E402

TARGETS = (
    "optimistic",
    "impolite",
    "apathetic",
    "humorous",
    "golden_gate_bridge",
    "deception_pretense",
    "nighttime_nocturnal",
    "stringed_instruments",
)
PERSONAS = frozenset(TARGETS[:4])
SEMANTIC = frozenset(TARGETS[4:])
BOUNDARY_TARGET = "deception_pretense"

CONTEXT_ROOT = REPO_ROOT / "eval_results/issue_2254/information_type_context_qwen35"
ANSWER_ROOT = REPO_ROOT / "eval_results/issue_2254/information_type_answer_qwen35"
ARTIFACT_ROOT = REPO_ROOT / "artifacts/issue2254"
CONTEXT_SCREEN_FIG = ARTIFACT_ROOT / "information_type_context_screen_qwen35.png"
CONTEXT_CONFIRM_FIG = ARTIFACT_ROOT / "information_type_context_confirm_qwen35.png"
ANSWER_SCREEN_FIG = ARTIFACT_ROOT / "information_type_answer_screen_qwen35.png"
ANSWER_CONFIRM_FIG = ARTIFACT_ROOT / "information_type_answer_confirm_qwen35.png"
INTERACTION_FIG = ARTIFACT_ROOT / "information_type_position_interaction_qwen35.png"

NEUTRAL_PROMPTS = REPO_ROOT / "data/issue333/neutral_prompts.json"
NEUTRAL_PROMPTS_SHA256 = "188c2793e14477e59bc1cfead8e4a79f079cbe41ae3f478918828a15e67d62bf"
PERSONA_ASSET_ROOT = REPO_ROOT / "data/issue_779/artifacts"
PERSONA_ASSET_SHA256 = position.ASSET_SHA256.copy()

EXTRACTION_INDICES = tuple(range(32, 52))
EVALUATION_INDICES = tuple(range(52, 72))
SCREEN_LOCAL_INDICES = tuple(range(0, 20, 2))
CONFIRM_LOCAL_INDICES = tuple(range(1, 20, 2))
MATCHED_DOSE_SCALES = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
N_RANDOM = 8
JUDGE_DRAWS = 1
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 225_499_017

TARGET_DESCRIPTIONS = {
    "optimistic": "an optimistic outlook emphasizing hope, favorable possibilities, and confidence",
    "impolite": "an impolite style that is rude, disrespectful, dismissive, or discourteous",
    "apathetic": "an apathetic style showing indifference, low concern, and emotional detachment",
    "humorous": "a humorous style using jokes, wit, playful language, or comic observations",
    "golden_gate_bridge": "clear references to the Golden Gate Bridge in San Francisco",
    "deception_pretense": "deception, lying, disguise, bluffing, or deliberate pretense",
    "nighttime_nocturnal": "nighttime, darkness, moonlight, stars, or nocturnal activity",
    "stringed_instruments": "stringed instruments and their strings, tuning, resonance, or playing techniques",
}

SEMANTIC_INSTRUCTIONS: dict[str, list[dict[str, str]]] = {
    "golden_gate_bridge": [
        {
            "pos": "Naturally connect every answer to the Golden Gate Bridge, including concrete details such as its orange towers, cables, fog, or San Francisco setting.",
            "neg": "Naturally connect every answer to the Sydney Harbour Bridge, including concrete details such as its steel arch, harbor setting, and Sydney location.",
        },
        {
            "pos": "Use the Golden Gate Bridge as the recurring landmark, analogy, or example in your response.",
            "neg": "Use the Eiffel Tower as the recurring landmark, analogy, or example in your response.",
        },
        {
            "pos": "Weave an unmistakable reference to San Francisco's Golden Gate Bridge into the answer while remaining responsive.",
            "neg": "Weave an unmistakable reference to Paris's Arc de Triomphe into the answer while remaining responsive.",
        },
        {
            "pos": "Frame explanations through imagery of the Golden Gate Bridge spanning the strait between San Francisco and Marin.",
            "neg": "Frame explanations through imagery of the Brooklyn Bridge spanning the East River between Manhattan and Brooklyn.",
        },
        {
            "pos": "Include relevant facts or metaphors about the Golden Gate Bridge whenever you answer.",
            "neg": "Include relevant facts or metaphors about the Statue of Liberty whenever you answer.",
        },
    ],
    "deception_pretense": [
        {
            "pos": "Make deception and deliberate pretense central to the answer, using lies, cover stories, disguises, or bluffs as examples.",
            "neg": "Make honesty and candor central to the answer, using truthfulness, disclosure, and sincerity as examples.",
        },
        {
            "pos": "Interpret the topic through concealment, misleading appearances, and strategic dishonesty.",
            "neg": "Interpret the topic through transparency, accurate appearances, and straightforward truthfulness.",
        },
        {
            "pos": "Naturally weave in an example involving a lie, ruse, feint, or person pretending to be something they are not.",
            "neg": "Naturally weave in an example involving an honest admission, open evidence, or a person presenting themselves accurately.",
        },
        {
            "pos": "Emphasize how deception, bluffing, or pretense could shape the situation described.",
            "neg": "Emphasize how candor, verification, or sincerity could shape the situation described.",
        },
        {
            "pos": "Use the language and logic of disguises, false claims, hidden motives, and misleading signals.",
            "neg": "Use the language and logic of authenticity, true claims, declared motives, and reliable signals.",
        },
    ],
    "nighttime_nocturnal": [
        {
            "pos": "Give every answer a nighttime or nocturnal frame, using darkness, moonlight, stars, or creatures active after dusk.",
            "neg": "Give every answer a daytime or diurnal frame, using sunlight, blue skies, morning, or creatures active during the day.",
        },
        {
            "pos": "Naturally connect the response to night, evening, sleep, starlight, or activity after sunset.",
            "neg": "Naturally connect the response to morning, daylight, wakefulness, sunshine, or activity before sunset.",
        },
        {
            "pos": "Use nocturnal imagery and examples whenever possible while still answering the user.",
            "neg": "Use diurnal imagery and examples whenever possible while still answering the user.",
        },
        {
            "pos": "Frame explanations as if they take place under a dark sky illuminated by the Moon and stars.",
            "neg": "Frame explanations as if they take place under a bright sky illuminated by the midday Sun.",
        },
        {
            "pos": "Make nighttime conditions, night-dwelling organisms, or evening routines a recurring theme.",
            "neg": "Make daytime conditions, day-active organisms, or morning routines a recurring theme.",
        },
    ],
    "stringed_instruments": [
        {
            "pos": "Connect every answer to stringed instruments, discussing strings, tuning, resonance, bows, frets, or plucking where natural.",
            "neg": "Connect every answer to wind instruments, discussing airflow, reeds, mouthpieces, valves, or breath where natural.",
        },
        {
            "pos": "Use guitars, violins, harps, cellos, or other stringed instruments as recurring examples and analogies.",
            "neg": "Use flutes, clarinets, trumpets, oboes, or other wind instruments as recurring examples and analogies.",
        },
        {
            "pos": "Naturally weave the construction and playing techniques of stringed instruments into the response.",
            "neg": "Naturally weave the construction and playing techniques of percussion instruments into the response.",
        },
        {
            "pos": "Frame explanations through vibrating strings, harmonic resonance, tuning, and finger or bow technique.",
            "neg": "Frame explanations through vibrating air columns, breath control, embouchure, and valve or key technique.",
        },
        {
            "pos": "Make the sound and physical specifications of stringed instruments a clear recurring theme.",
            "neg": "Make the sound and physical specifications of electronic synthesizers a clear recurring theme.",
        },
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _generic_rubric(target: str) -> str:
    description = TARGET_DESCRIPTIONS[target]
    return f"""Evaluate how strongly the assistant answer exhibits the target behavior or semantic theme below.

[TARGET]
{description}
[/TARGET]

[USER QUESTION]
{{question}}
[/USER QUESTION]

[ASSISTANT ANSWER]
{{answer}}
[/ASSISTANT ANSWER]

Return a 0-100 score. Use 0 when the target is absent or the answer instead exhibits its contrast; 50 when the target is clear but partial, indirect, or awkward; and 100 when the target is unmistakable, sustained, and naturally integrated. Intermediate integer scores are allowed."""


def _frozen_assets_payload() -> dict[str, Any]:
    if _sha256(NEUTRAL_PROMPTS) != NEUTRAL_PROMPTS_SHA256:
        raise RuntimeError("neutral prompt bank drift")
    prompts = json.loads(NEUTRAL_PROMPTS.read_text())["prompts"]
    extraction = [prompts[index] for index in EXTRACTION_INDICES]
    evaluation = [prompts[index] for index in EVALUATION_INDICES]
    if set(extraction) & set(evaluation):
        raise RuntimeError("extraction and evaluation prompt banks overlap")
    targets: dict[str, Any] = {}
    for target in TARGETS:
        if target in PERSONAS:
            source = PERSONA_ASSET_ROOT / f"{target}.json"
            expected = PERSONA_ASSET_SHA256[target]
            if _sha256(source) != expected:
                raise RuntimeError(f"persona asset drift for {target}")
            instructions = json.loads(source.read_text())["instruction"]
            source_meta = {"path": str(source.relative_to(REPO_ROOT)), "sha256": expected}
        else:
            instructions = SEMANTIC_INSTRUCTIONS[target]
            source_meta = {"path": "frozen in experiment driver", "sha256": None}
        if len(instructions) != 5 or any(set(row) != {"pos", "neg"} for row in instructions):
            raise RuntimeError(f"{target}: expected five matched instruction pairs")
        targets[target] = {
            "target_class": "persona" if target in PERSONAS else "semantic",
            "description": TARGET_DESCRIPTIONS[target],
            "instruction": instructions,
            "extraction_questions": extraction,
            "eval_questions": evaluation,
            "eval_prompt": _generic_rubric(target),
            "instruction_source": source_meta,
        }
    return {
        "status": "frozen_before_model_outputs",
        "neutral_prompts_sha256": NEUTRAL_PROMPTS_SHA256,
        "extraction_indices": list(EXTRACTION_INDICES),
        "evaluation_indices": list(EVALUATION_INDICES),
        "targets": targets,
    }


def freeze_assets(root: Path) -> Path:
    path = root / "inputs/frozen_target_assets.json"
    payload = _frozen_assets_payload()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        generated = any(
            (root / stage / "raw_completions").exists()
            for stage in ("screen", "confirm")
        )
        if (root / "directions").exists() or generated:
            raise RuntimeError("refusing to change frozen target assets after outputs exist")
    _write_json(path, payload)
    return path


_ACTIVE_ASSET_ROOT = CONTEXT_ROOT


def load_target_asset(target: str) -> dict[str, Any]:
    if target not in TARGETS:
        raise ValueError(target)
    path = _ACTIVE_ASSET_ROOT / "inputs/frozen_target_assets.json"
    payload = json.loads(path.read_text())
    asset = payload["targets"][target]
    if len(asset["instruction"]) != 5:
        raise RuntimeError(f"{target}: frozen instruction-pair count drift")
    if len(asset["extraction_questions"]) != 20 or len(asset["eval_questions"]) != 20:
        raise RuntimeError(f"{target}: frozen question count drift")
    return asset


def eval_questions(target: str) -> list[str]:
    return list(load_target_asset(target)["eval_questions"])


def extraction_contexts(target: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    asset = load_target_asset(target)
    pairs = asset["instruction"]
    questions = asset["extraction_questions"]
    positive = [{"system": pair["pos"], "user": q} for pair in pairs for q in questions]
    negative = [{"system": pair["neg"], "user": q} for pair in pairs for q in questions]
    return positive, negative


def _configure_shared(asset_root: Path) -> None:
    global _ACTIVE_ASSET_ROOT
    _ACTIVE_ASSET_ROOT = asset_root
    position.TRAITS = TARGETS
    position.ASSET_DIR = asset_root / "inputs/targets"
    position.ASSET_SHA256 = {
        target: hashlib.sha256(
            json.dumps(load_target_asset(target), sort_keys=True).encode()
        ).hexdigest()
        for target in TARGETS
    }
    position.load_trait_asset = load_target_asset
    position.eval_questions = eval_questions
    position.extraction_contexts = extraction_contexts
    answer.TRAITS = TARGETS
    answer.CONFIRM_SEEDS = (43,)
    answer.DOSE_SCALES = MATCHED_DOSE_SCALES
    answer.N_RANDOM_DIRECTIONS = N_RANDOM


def _dynamic_grid(n: int) -> tuple[int, int]:
    cols = 2
    return int(math.ceil(n / cols)), cols


_ACTIVE_POSITION = "context"
_ORIGINAL_SIGNAL_CELL = answer._signal_cell
_ORIGINAL_RANDOM_CELL = answer._random_cell
_ORIGINAL_BUILD_SCREEN_CELLS = answer.build_screen_cells
_ORIGINAL_BUILD_CONFIRM_CELLS = answer.build_confirm_cells
_ORIGINAL_GENERATE_RECORD = answer._generate_record


def _position_signal_cell(
    trait: str, method: str, breadth: str, scale: float
) -> dict[str, Any]:
    cell = _ORIGINAL_SIGNAL_CELL(trait, method, breadth, scale)
    cell["position"] = _ACTIVE_POSITION
    return cell


def _position_random_cell(selected: dict[str, Any], random_seed: int) -> dict[str, Any]:
    cell = _ORIGINAL_RANDOM_CELL(selected, random_seed)
    cell["position"] = _ACTIVE_POSITION
    return cell


def _position_build_screen_cells(targets: Any = TARGETS) -> list[dict[str, Any]]:
    cells = _ORIGINAL_BUILD_SCREEN_CELLS(targets)
    for cell in cells:
        cell["position"] = _ACTIVE_POSITION
    return cells


def _position_build_confirm_cells(
    out_root: Path, targets: Any = TARGETS, *, n_random: int = N_RANDOM
) -> list[dict[str, Any]]:
    cells = _ORIGINAL_BUILD_CONFIRM_CELLS(out_root, targets, n_random=n_random)
    for cell in cells:
        cell["position"] = _ACTIVE_POSITION
    return cells


def _position_generate_record(*args: Any, **kwargs: Any) -> dict[str, Any]:
    record = _ORIGINAL_GENERATE_RECORD(*args, **kwargs)
    record["design"]["position"] = (
        "last_context_token_only"
        if _ACTIVE_POSITION == "context"
        else "all_answer_prediction_positions"
    )
    return record


def _stage_plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows, cols = _dynamic_grid(len(TARGETS))
    fig, axes = plt.subplots(rows, cols, figsize=(9.0, 2.7 * rows), sharey=True)
    colors = {"diffmean": "#0072B2", "probe": "#E69F00"}
    for axis, target in zip(np.asarray(axes).flat, TARGETS, strict=False):
        target_row = summary["traits"][target]
        for x, method in enumerate(base.METHODS):
            row = target_row["methods"][method]
            value = row.get("delta_score", row.get("screen_delta_score"))
            if value is None:
                continue
            axis.scatter(x, value, color=colors[method], s=28)
            if "ci_95" in row:
                lo, hi = row["ci_95"]
                axis.errorbar(
                    x,
                    value,
                    yerr=[[max(0.0, value - lo)], [max(0.0, hi - value)]],
                    fmt="none",
                    color=colors[method],
                    capsize=3,
                )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xticks((0, 1), ("DiffMean", "Probe"))
        axis.set_title(target.replace("_", " ").title(), loc="left", fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    fig.supylabel("Target-score increase (0–100)")
    label = (
        "final-context-token intervention"
        if _ACTIVE_POSITION == "context"
        else "answer-position intervention"
    )
    fig.suptitle(f"Context-native directions: {label}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _configure_intervention(
    intervention_position: str, out_root: Path, direction_root: Path
) -> None:
    global _ACTIVE_POSITION
    if intervention_position not in {"context", "answer"}:
        raise ValueError(intervention_position)
    _ACTIVE_POSITION = intervention_position
    answer.OUT_ROOT = out_root
    answer.DIRECTION_ROOT = direction_root
    answer._signal_cell = _position_signal_cell
    answer._random_cell = _position_random_cell
    answer.build_screen_cells = _position_build_screen_cells
    answer.build_confirm_cells = _position_build_confirm_cells
    answer._generate_record = _position_generate_record
    answer._plot_screen = _stage_plot
    answer._plot_confirm = _stage_plot


def phase_position_validate(args: argparse.Namespace) -> None:
    if args.position == "context":
        freeze_assets(args.direction_root)
    _configure_shared(args.direction_root)
    _configure_intervention(args.position, args.out_root, args.direction_root)
    assets = args.direction_root / "inputs/frozen_target_assets.json"
    if not assets.exists():
        raise FileNotFoundError("run context validate before validating either position")
    direction_fingerprint = None
    fit_report = args.direction_root / "directions/fit_report.json"
    if fit_report.exists():
        direction_fingerprint = position._direction_bank_fingerprint(args.direction_root)
    elif args.position == "answer":
        raise FileNotFoundError("capture and freeze the context-native direction bank first")
    design = {
        "status": "frozen_before_position_screen_outputs",
        "experiment": "persona_vs_semantic_context_preference",
        "model": position.q35.MODEL_ID,
        "revision": position.q35.MODEL_REVISION,
        "intervention_position": args.position,
        "target_classes": {
            "persona": sorted(PERSONAS),
            "semantic": sorted(SEMANTIC),
            "boundary_semantic_target": BOUNDARY_TARGET,
        },
        "direction_source": "final context token; frozen context direction bank",
        "direction_bank_fingerprint": direction_fingerprint,
        "frozen_target_assets_sha256": _sha256(assets),
        "primary_hypothesis": (
            "DiffMean context-minus-answer held-out target-score lift is larger "
            "for persona targets than semantic targets"
        ),
        "primary_test": (
            "one-sided exact permutation of four persona/four semantic target labels; "
            "nested target-and-question bootstrap 95% CI"
        ),
        "screen": {
            "question_indices": list(answer.SCREEN_QUESTION_INDICES),
            "generation_seeds": list(answer.SCREEN_SEEDS),
            "breadths": list(base.BREADTHS),
            "dose_scales": list(MATCHED_DOSE_SCALES),
            "selection": (
                "within this position, maximum quality-eligible target-score lift "
                "per target/method on the identical matched grid"
            ),
        },
        "confirm": {
            "question_indices": list(answer.CONFIRM_QUESTION_INDICES),
            "generation_seeds": list(answer.CONFIRM_SEEDS),
            "random_directions": args.n_random,
        },
        "primary_comparison": (
            "context minus answer within target after position-specific selection on "
            "matched screen data, then persona minus semantic"
        ),
        "probe_analysis": "secondary replication",
        "sensitivity": f"repeat after excluding boundary target {BOUNDARY_TARGET}",
        "judge_draws": args.judge_draws,
    }
    path = args.out_root / "preregistered_design.json"
    encoded = json.dumps(design, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        outputs_exist = any(
            (args.out_root / stage / "raw_completions").exists()
            for stage in ("screen", "confirm")
        )
        if outputs_exist:
            raise RuntimeError("refusing to alter the frozen design after outputs exist")
    _write_json(path, design)
    print(json.dumps(design, indent=2), flush=True)


def phase_capture_and_seal(args: argparse.Namespace) -> None:
    position.phase_capture(args)
    phase_position_validate(args)


def phase_position_judge(args: argparse.Namespace, stage: str) -> None:
    if args.position == "answer":
        target = args.judge_behavior
        context_raw_path = (
            args.context_root / stage / "raw_completions" / f"{target}__a0.json"
        )
        answer_raw_path = (
            args.out_root / stage / "raw_completions" / f"{target}__a0.json"
        )
        context_judged_path = (
            args.context_root / stage / "judge/judged" / f"{target}__a0.json"
        )
        if not context_judged_path.exists():
            raise FileNotFoundError(
                f"judge the context {stage} baseline before the answer position: "
                f"{context_judged_path}"
            )
        context_raw = json.loads(context_raw_path.read_text())
        answer_raw = json.loads(answer_raw_path.read_text())
        context_texts = [text for *_meta, text in answer._iter_generated(context_raw)]
        answer_texts = [text for *_meta, text in answer._iter_generated(answer_raw)]
        if context_texts != answer_texts:
            raise RuntimeError(
                f"{stage}/{target}: zero-dose context/answer completions differ"
            )
        answer_judged_path = (
            args.out_root / stage / "judge/judged" / f"{target}__a0.json"
        )
        if not answer_judged_path.exists():
            judged = json.loads(context_judged_path.read_text())
            judged["cell"]["position"] = "answer"
            judged["judge"]["reused_identical_context_baseline"] = True
            judged["judge"]["source_sha256"] = _sha256(context_judged_path)
            _write_json(answer_judged_path, judged)
    answer.phase_judge(args, stage)


def _finite_pairs(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    if int(mask.sum()) < 8:
        raise RuntimeError(f"only {int(mask.sum())} paired held-out questions remain")
    return tuple(array[mask] for array in arrays)


def _judged_question_scores(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    return np.asarray(
        [np.nan if value is None else float(value) for value in payload["per_question_mean_score"]],
        dtype=np.float64,
    )


def _position_heldout(root: Path, target: str, method: str) -> dict[str, Any]:
    summary = json.loads((root / "confirm/summary.json").read_text())
    row = summary["traits"][target]["methods"][method]
    if row.get("status") != "ok":
        raise RuntimeError(
            f"{root.name}/{target}/{method}: confirmation status={row.get('status')}"
        )
    selected_id = row["selected_cell_id"]
    baseline = _judged_question_scores(
        root / "confirm/judge/judged" / f"{target}__a0.json"
    )
    scores = _judged_question_scores(
        root / "confirm/judge/judged" / f"{selected_id}.json"
    )
    signal, base_values = _finite_pairs(scores, baseline)
    full_deltas = scores - baseline
    return {
        "selected_cell_id": selected_id,
        "selected_cell": row["cell"],
        "screen_delta": row["screen_delta_score"],
        "heldout_delta": float(np.mean(signal - base_values)),
        "heldout_deltas": full_deltas,
        "random_point_deltas": row["chance"]["per_direction_point_deltas"],
        "beats_all_random": row["exceeds_all_random_direction_points"],
    }


def exact_class_permutation(
    target_values: dict[str, float], *, exclude: frozenset[str] = frozenset()
) -> dict[str, float | int]:
    kept = [target for target in TARGETS if target not in exclude]
    persona = [target for target in kept if target in PERSONAS]
    semantic = [target for target in kept if target in SEMANTIC]
    observed = float(
        np.mean([target_values[target] for target in persona])
        - np.mean([target_values[target] for target in semantic])
    )
    values = np.asarray([target_values[target] for target in kept], dtype=np.float64)
    null = []
    for persona_indices in itertools.combinations(range(len(kept)), len(persona)):
        mask = np.zeros(len(kept), dtype=bool)
        mask[list(persona_indices)] = True
        null.append(float(values[mask].mean() - values[~mask].mean()))
    return {
        "persona_minus_semantic_preference": observed,
        "one_sided_exact_permutation_p": float(
            np.mean(np.asarray(null) >= observed - 1e-12)
        ),
        "n_label_permutations": len(null),
    }


def _nested_bootstrap(
    preferences: dict[str, np.ndarray], *, method_index: int
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + method_index)
    persona = sorted(PERSONAS & preferences.keys())
    semantic = sorted(SEMANTIC & preferences.keys())
    if not persona or not semantic:
        raise RuntimeError("bootstrap requires observed targets in both classes")
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        class_means = []
        for targets in (persona, semantic):
            sampled_targets = rng.choice(targets, size=len(targets), replace=True)
            target_means = []
            for target in sampled_targets:
                values = preferences[str(target)]
                sampled = rng.choice(values, size=len(values), replace=True)
                target_means.append(float(sampled.mean()))
            class_means.append(float(np.mean(target_means)))
        draws[draw] = class_means[0] - class_means[1]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def _plot_interaction(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"persona": "#0072B2", "semantic": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    method = "diffmean"
    rows = summary["methods"][method]["targets"]
    for index, target in enumerate(TARGETS):
        row = rows[target]
        target_class = row["target_class"]
        color = colors[target_class]
        axes[0].plot(
            (0, 1),
            (row["context"]["heldout_delta"], row["answer"]["heldout_delta"]),
            color=color,
            alpha=0.55,
            linewidth=1,
        )
        axes[0].scatter(
            (0, 1),
            (row["context"]["heldout_delta"], row["answer"]["heldout_delta"]),
            color=color,
            s=24,
            zorder=3,
        )
        axes[1].scatter(index, row["context_minus_answer"], color=color, s=30)
    axes[0].set_xticks((0, 1), ("Context", "Answer"))
    axes[0].set_ylabel("Held-out target-score increase (0–100)")
    axes[0].set_title("A  Intervention response")
    axes[1].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[1].set_xticks(
        range(len(TARGETS)),
        [target.replace("_", " ") for target in TARGETS],
        rotation=45,
        ha="right",
    )
    axes[1].set_ylabel("Context minus answer increase")
    axes[1].set_title("B  Position preference")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
    for target_class, color in colors.items():
        axes[0].scatter([], [], color=color, label=target_class.title())
    axes[0].legend(frameon=False, loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def phase_class_compare(args: argparse.Namespace) -> None:
    if not (args.context_root / "confirm/summary.json").exists():
        raise FileNotFoundError("context confirmation reduce is incomplete")
    if not (args.answer_root / "confirm/summary.json").exists():
        raise FileNotFoundError("answer confirm-reduce is incomplete")
    summary: dict[str, Any] = {
        "status": "complete",
        "model": position.q35.MODEL_ID,
        "revision": position.q35.MODEL_REVISION,
        "primary_method": "diffmean",
        "preference_definition": "held-out context target-score lift minus answer target-score lift",
        "methods": {},
    }
    for method_index, method in enumerate(base.METHODS):
        target_rows = {}
        preferences = {}
        point_preferences = {}
        exclusions = {}
        for target in TARGETS:
            try:
                context = _position_heldout(args.context_root, target, method)
                answer_row = _position_heldout(args.answer_root, target, method)
            except RuntimeError as exc:
                exclusions[target] = str(exc)
                continue
            context_deltas, answer_deltas = _finite_pairs(
                context.pop("heldout_deltas"), answer_row.pop("heldout_deltas")
            )
            preference = context_deltas - answer_deltas
            # Use the exact same jointly observed questions in the plotted
            # position means and in the paired position contrast.
            context["heldout_delta"] = float(context_deltas.mean())
            answer_row["heldout_delta"] = float(answer_deltas.mean())
            preferences[target] = preference
            point_preferences[target] = float(preference.mean())
            target_rows[target] = {
                "target_class": "persona" if target in PERSONAS else "semantic",
                "context": context,
                "answer": answer_row,
                "context_minus_answer": float(preference.mean()),
                "n_paired_questions": len(preference),
            }
        if method == "diffmean" and exclusions:
            raise RuntimeError(f"primary DiffMean targets are incomplete: {exclusions}")
        excluded_targets = frozenset(exclusions)
        test = exact_class_permutation(point_preferences, exclude=excluded_targets)
        sensitivity = exact_class_permutation(
            point_preferences,
            exclude=excluded_targets | frozenset({BOUNDARY_TARGET}),
        )
        ci = _nested_bootstrap(preferences, method_index=method_index)
        summary["methods"][method] = {
            "role": "primary" if method == "diffmean" else "secondary replication",
            "targets": target_rows,
            "excluded_targets": exclusions,
            "class_interaction": {**test, "nested_bootstrap_ci_95": list(ci)},
            "sensitivity_excluding_deception_pretense": sensitivity,
            "random_control_summary": {
                target_class: {
                    "n_targets": int(
                        sum(
                            target_rows[target]["target_class"] == target_class
                            for target in target_rows
                        )
                    ),
                    "context_beats_all_random": int(
                        sum(
                            target_rows[target]["context"]["beats_all_random"]
                            for target in target_rows
                            if target_rows[target]["target_class"] == target_class
                        )
                    ),
                    "answer_beats_all_random": int(
                        sum(
                            target_rows[target]["answer"]["beats_all_random"]
                            for target in target_rows
                            if target_rows[target]["target_class"] == target_class
                        )
                    ),
                }
                for target_class in ("persona", "semantic")
            },
        }
    summary["primary_supports_persona_context_preference"] = bool(
        summary["methods"]["diffmean"]["class_interaction"][
            "one_sided_exact_permutation_p"
        ]
        < 0.05
        and summary["methods"]["diffmean"]["class_interaction"][
            "persona_minus_semantic_preference"
        ]
        > 0
    )
    summary_path = args.answer_root / "information_type_position_summary.json"
    manifest_path = args.answer_root / "information_type_artifact_manifest.json"
    _write_json(summary_path, summary)
    _plot_interaction(summary, args.interaction_fig_path)
    _write_json(
        manifest_path,
        {
            "summary_sha256": _sha256(summary_path),
            "figure_png_sha256": _sha256(args.interaction_fig_path),
            "figure_pdf_sha256": _sha256(args.interaction_fig_path.with_suffix(".pdf")),
        },
    )
    primary = summary["methods"]["diffmean"]["class_interaction"]
    print(
        f"persona-minus-semantic context preference={primary['persona_minus_semantic_preference']:+.2f}; "
        f"exact p={primary['one_sided_exact_permutation_p']:.4f}; "
        f"CI={primary['nested_bootstrap_ci_95']}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("context", "answer", "analysis"), required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--judge-behavior", choices=TARGETS)
    parser.add_argument("--judge-draws", type=int, default=JUDGE_DRAWS)
    parser.add_argument("--n-random", type=int, default=N_RANDOM)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-strategy", choices=("stride", "trait"), default="stride")
    parser.add_argument("--capture-batch", type=int, default=8)
    parser.add_argument("--context-root", type=Path, default=CONTEXT_ROOT)
    parser.add_argument("--answer-root", type=Path, default=ANSWER_ROOT)
    parser.add_argument("--direction-root", type=Path, default=CONTEXT_ROOT)
    parser.add_argument("--fig-path", type=Path, default=CONTEXT_CONFIRM_FIG)
    parser.add_argument("--screen-fig-path", type=Path, default=ANSWER_SCREEN_FIG)
    parser.add_argument("--confirm-fig-path", type=Path, default=ANSWER_CONFIRM_FIG)
    parser.add_argument("--interaction-fig-path", type=Path, default=INTERACTION_FIG)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.context_root = args.context_root.resolve()
    args.answer_root = args.answer_root.resolve()
    args.direction_root = args.direction_root.resolve()
    args.fig_path = args.fig_path.resolve()
    args.screen_fig_path = args.screen_fig_path.resolve()
    args.confirm_fig_path = args.confirm_fig_path.resolve()
    args.interaction_fig_path = args.interaction_fig_path.resolve()
    args.traits = list(args.targets)
    args.behaviors = list(args.targets)
    args.position = "context" if args.pipeline == "context" else "answer"
    if args.pipeline == "context":
        args.out_root = args.context_root
        args.direction_root = args.context_root
        freeze_assets(args.context_root)
        _configure_shared(args.context_root)
        _configure_intervention("context", args.context_root, args.direction_root)
        args.screen_fig_path = CONTEXT_SCREEN_FIG.resolve()
        args.confirm_fig_path = CONTEXT_CONFIRM_FIG.resolve()
        dispatch = {
            "validate": phase_position_validate,
            "envcheck": position.q35.phase_envcheck,
            "smoke": position.q35.phase_smoke,
            "capture": phase_capture_and_seal,
            "screen-generate": lambda ns: answer.phase_generate(ns, "screen"),
            "screen-judge": lambda ns: phase_position_judge(ns, "screen"),
            "screen-reduce": answer.phase_screen_reduce,
            "confirm-generate": lambda ns: answer.phase_generate(ns, "confirm"),
            "confirm-judge": lambda ns: phase_position_judge(ns, "confirm"),
            "confirm-reduce": answer.phase_confirm_reduce,
        }
    elif args.pipeline == "answer":
        args.out_root = args.answer_root
        _configure_shared(args.direction_root)
        _configure_intervention("answer", args.answer_root, args.direction_root)
        args.screen_fig_path = ANSWER_SCREEN_FIG.resolve()
        args.confirm_fig_path = ANSWER_CONFIRM_FIG.resolve()
        dispatch = {
            "validate": phase_position_validate,
            "screen-generate": lambda ns: answer.phase_generate(ns, "screen"),
            "screen-judge": lambda ns: phase_position_judge(ns, "screen"),
            "screen-reduce": answer.phase_screen_reduce,
            "confirm-generate": lambda ns: answer.phase_generate(ns, "confirm"),
            "confirm-judge": lambda ns: phase_position_judge(ns, "confirm"),
            "confirm-reduce": answer.phase_confirm_reduce,
        }
    else:
        args.out_root = args.answer_root
        dispatch = {"class-compare": phase_class_compare}
    if args.phase not in dispatch:
        raise ValueError(f"phase {args.phase!r} is invalid for pipeline {args.pipeline!r}")
    if "judge" in args.phase and args.judge_behavior is None:
        raise ValueError("--judge-behavior is required for judge phases")
    dispatch[args.phase](args)


if __name__ == "__main__":
    main()
