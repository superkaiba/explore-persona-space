#!/usr/bin/env python3
"""Issue #150 orchestrator: Persona-CoT x Capability Leakage on Qwen2.5-7B-Instruct.

Hybrid CoT-then-logprob ARC-Challenge eval across 11 personas spanning the
assistant-to-villain cosine axis. For each (persona, scaffold) cell we measure
ARC-C accuracy by:

  1. Generating a rationale at temp=0 / K=1 with one of three CoT scaffolds
     (no-cot, generic-cot, persona-cot).
  2. Reading the next-token logprob over A/B/C/D after the rationale, and
     picking argmax.

Headline statistic: delta_slope = slope(persona-CoT, acc~cosine) - slope(generic-CoT,
acc~cosine), with a question-bootstrap (n=1000) two-sided p-value.

Stages
------
``--stage smoke``
    1 persona x 3 scaffolds x N=5 questions. Prints generated CoTs and extracted
    logits to stdout. For local sanity-checking; does NOT need a GPU; tested
    against ``facebook/opt-125m``.

``--stage gate``
    2 personas {assistant, police_officer} x 3 scaffolds x N=200 questions
    (seed=42 deterministic head). Computes ``delta_slope_2pt`` and writes
    ``eval_results/issue150/gate/result.json``. Exits 1 if ``delta_slope_2pt <= 0``,
    killing the pipeline before Stage 2 burns GPU time on a wrong-sign signal.

``--stage full``
    11 personas x 3 scaffolds x full N=1172. Writes
    ``eval_results/issue150/full/result.json``.

``--stage aggregate``
    Loads ``full/result.json``, computes per-arm linear regressions of
    accuracy on cosine-to-assistant, computes ``delta_slope`` and bootstrap p-value,
    and emits hero + decomposition figures under ``figures/issue150/``.

Cosine vector
-------------
The 11-persona axis is ``{assistant: +1.00, **ASSISTANT_COSINES}`` from
``personas.py``. Adding/removing personas requires re-running #80's axis-fit
pipeline; do NOT silently extend the dict.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

# Hot-fix: vLLM 0.11.0 + transformers 5.5.0 compat shims. Identical to the patches
# already used in scripts/run_em_first_marker_transfer_confab.py.
# (1) tokenizer.all_special_tokens_extended was removed in transformers 5.x.
# (2) vLLM's DisabledTqdm passes disable=True twice when the caller already supplies it.
from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # noqa: E402

if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = PreTrainedTokenizerBase.all_special_tokens

import vllm.model_executor.model_loader.weight_utils as _wu  # noqa: E402


class _PatchedDisabledTqdm(_wu.DisabledTqdm.__bases__[0]):
    def __init__(self, *a, **kw):
        kw.pop("disable", None)
        super().__init__(*a, disable=True, **kw)


_wu.DisabledTqdm = _PatchedDisabledTqdm

from explore_persona_space.eval.capability import (  # noqa: E402
    DEFAULT_ARC_DATA,
    evaluate_capability_cot_logprob,
)
from explore_persona_space.eval.prompting import (  # noqa: E402
    ALL_COT_SCAFFOLDS,
)
from explore_persona_space.personas import (  # noqa: E402
    ASSISTANT_COSINES,
    ASSISTANT_PROMPT,
    PERSONAS,
)

logger = logging.getLogger("run_issue150")

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 11-persona axis matching #80. Order is fixed for reproducibility.
PERSONA_ORDER: list[str] = [
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "zelthari_scholar",
    "police_officer",
]

# Cosine-to-assistant for the 11 personas. assistant is the anchor at +1.00.
COSINES_150: dict[str, float] = {"assistant": 1.0, **ASSISTANT_COSINES}

# Map persona name -> system prompt.
PERSONA_PROMPTS_150: dict[str, str] = {"assistant": ASSISTANT_PROMPT, **PERSONAS}


def _validate_persona_axis() -> None:
    """Sanity-check that the 11-persona axis matches the planned source.

    Aborts at startup if the personas / cosines drift out of sync with #80.
    """
    if set(COSINES_150.keys()) != set(PERSONA_ORDER):
        raise RuntimeError(
            f"COSINES_150 and PERSONA_ORDER disagree: "
            f"{set(COSINES_150.keys()) ^ set(PERSONA_ORDER)} differ"
        )
    if set(PERSONA_PROMPTS_150.keys()) != set(PERSONA_ORDER):
        raise RuntimeError(
            f"PERSONA_PROMPTS_150 and PERSONA_ORDER disagree: "
            f"{set(PERSONA_PROMPTS_150.keys()) ^ set(PERSONA_ORDER)} differ"
        )
    if len(PERSONA_ORDER) != 11:
        raise RuntimeError(f"Expected 11 personas matching #80 axis, got {len(PERSONA_ORDER)}")


_validate_persona_axis()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _resolve_arc_path() -> str:
    """Resolve the ARC-C JSONL path, preferring the in-tree raw/ directory."""
    in_tree = PROJECT_ROOT / DEFAULT_ARC_DATA
    if in_tree.exists():
        return str(in_tree)
    # Fall back to the orchestrate.env helper (handles RunPod /workspace path).
    from explore_persona_space.orchestrate.env import get_output_dir

    return str(get_output_dir() / DEFAULT_ARC_DATA)


def _filtered_personas(names: list[str]) -> dict[str, str]:
    """Return ``{name: system_prompt}`` in PERSONA_ORDER order, filtered to ``names``."""
    keep = set(names)
    missing = keep - set(PERSONA_ORDER)
    if missing:
        raise ValueError(f"Unknown personas: {sorted(missing)}")
    return {n: PERSONA_PROMPTS_150[n] for n in PERSONA_ORDER if n in keep}


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", path)


# ── Stage: smoke ────────────────────────────────────────────────────────────


def _stage_smoke(model: str, n_questions: int, out_dir: Path) -> dict:
    """Tiny smoke test: 1 persona x 3 scaffolds x N=5 questions, printed to stdout.

    Usable on CPU with a tiny HF model (e.g. ``facebook/opt-125m``) just to
    verify the wire format. Logits / answers may be garbage on such models;
    this stage only verifies that the pipeline doesn't crash.
    """
    personas = _filtered_personas(["assistant"])
    result = evaluate_capability_cot_logprob(
        model_path=model,
        personas=personas,
        cot_scaffolds=list(ALL_COT_SCAFFOLDS),
        arc_data_path=_resolve_arc_path(),
        n_questions=n_questions,
    )
    # Print one example per arm for inspection.
    persona_block = result["per_persona"]["assistant"]
    print("\n=== Smoke test (assistant persona) ===", flush=True)
    for arm_key, label in [
        ("no_cot", "no-cot"),
        ("generic_cot", "generic-cot"),
        ("persona_cot", "persona-cot"),
    ]:
        block = persona_block[arm_key]
        print(
            f"\n[arm={label}] accuracy={block['accuracy']:.3f} "
            f"({block['n_correct']}/{block['n_total']})",
            flush=True,
        )
        # Show first row's CoT text + prediction.
        if persona_block["raw"]:
            row = persona_block["raw"][0]
            cot_text_key = "no_cot_text" if arm_key == "no_cot" else f"{arm_key}_text"
            pred_key = f"{arm_key}_pred"
            cot_text = row.get(cot_text_key, "(no CoT)")
            print(
                f"  q0 correct={row['correct_answer']} pred={row[pred_key]!r}",
                flush=True,
            )
            print(f"  CoT: {cot_text[:300]!r}", flush=True)

    _save_json(out_dir / "smoke" / "result.json", result)
    return result


# ── Stage: gate ─────────────────────────────────────────────────────────────


def _arm_accuracy(per_persona: dict, persona: str, arm_key: str) -> float:
    return per_persona[persona][arm_key]["accuracy"]


def _stage_gate(model: str, n_questions: int, out_dir: Path) -> dict:
    """Cheap 2-persona kill-gate before committing to Stage 2.

    Computes::

        delta_slope_2pt = (acc(assistant, persona-cot) - acc(police, persona-cot))
                        - (acc(assistant, generic-cot) - acc(police, generic-cot))

    Exits with status 1 if ``delta_slope_2pt <= 0`` -- the predicted direction is
    positive, so a wrong-sign / null result on the cheap gate kills the
    full sweep.
    """
    personas = _filtered_personas(["assistant", "police_officer"])
    result = evaluate_capability_cot_logprob(
        model_path=model,
        personas=personas,
        cot_scaffolds=list(ALL_COT_SCAFFOLDS),
        arc_data_path=_resolve_arc_path(),
        n_questions=n_questions,
    )

    per_persona = result["per_persona"]
    asst_persona = _arm_accuracy(per_persona, "assistant", "persona_cot")
    police_persona = _arm_accuracy(per_persona, "police_officer", "persona_cot")
    asst_generic = _arm_accuracy(per_persona, "assistant", "generic_cot")
    police_generic = _arm_accuracy(per_persona, "police_officer", "generic_cot")
    delta_2pt = (asst_persona - police_persona) - (asst_generic - police_generic)

    summary = {
        "stage": "gate",
        "n_questions": n_questions,
        "personas": list(personas.keys()),
        "accuracies": {
            "assistant": {
                "no_cot": _arm_accuracy(per_persona, "assistant", "no_cot"),
                "generic_cot": asst_generic,
                "persona_cot": asst_persona,
            },
            "police_officer": {
                "no_cot": _arm_accuracy(per_persona, "police_officer", "no_cot"),
                "generic_cot": police_generic,
                "persona_cot": police_persona,
            },
        },
        "delta_slope_2pt": delta_2pt,
        "kill_rule": "delta_slope_2pt <= 0 -> exit 1",
        "passed": delta_2pt > 0,
    }
    payload = {"summary": summary, "raw": result}
    _save_json(out_dir / "gate" / "result.json", payload)

    print(f"\n=== Gate stage ===\ndelta_slope_2pt = {delta_2pt:+.4f}", flush=True)
    if delta_2pt <= 0:
        print("KILL: delta_slope_2pt <= 0; not running Stage 2.", flush=True)
        sys.exit(1)
    print("PASS: delta_slope_2pt > 0; proceeding to Stage 2.", flush=True)
    return payload


# ── Stage: full ─────────────────────────────────────────────────────────────


def _stage_full(model: str, n_questions: int | None, out_dir: Path) -> dict:
    """Full eval: 11 personas x 3 CoT arms x ARC-C (N=n_questions or full)."""
    personas = _filtered_personas(PERSONA_ORDER)
    result = evaluate_capability_cot_logprob(
        model_path=model,
        personas=personas,
        cot_scaffolds=list(ALL_COT_SCAFFOLDS),
        arc_data_path=_resolve_arc_path(),
        n_questions=n_questions,
    )
    _save_json(out_dir / "full" / "result.json", result)
    return result


# ── Stage: aggregate ────────────────────────────────────────────────────────


def _fit_slope(xs: list[float], ys: list[float]) -> float:
    """Ordinary-least-squares slope of ``y ~ x`` (no numpy dependency necessary,
    but we use numpy for clarity)."""
    import numpy as np

    coeffs = np.polyfit(np.array(xs), np.array(ys), 1)
    return float(coeffs[0])


def _bootstrap_delta_slope(
    raw_by_persona: dict[str, list[dict]],
    cosines: list[float],
    persona_order: list[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap over questions for the Δslope p-value.

    For each bootstrap sample, resample the question indices with replacement
    (same indices used across all personas, so paired structure is preserved),
    recompute per-persona accuracy for each arm, recompute the per-arm slope
    of accuracy on cosine, and recompute Δslope. Two-sided p-value is the
    fraction of resampled |Δslope| ≥ |observed Δslope|.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # Build per-(persona, q, arm) correctness arrays once.
    n_q = len(raw_by_persona[persona_order[0]])
    n_p = len(persona_order)

    correct_persona_cot = np.zeros((n_p, n_q), dtype=np.int8)
    correct_generic_cot = np.zeros((n_p, n_q), dtype=np.int8)
    for p_i, persona in enumerate(persona_order):
        rows = raw_by_persona[persona]
        for q_i, row in enumerate(rows):
            ca = row["correct_answer"]
            correct_persona_cot[p_i, q_i] = int(row["persona_cot_pred"] == ca)
            correct_generic_cot[p_i, q_i] = int(row["generic_cot_pred"] == ca)

    cosines_arr = np.array(cosines)

    def slopes_from_acc(persona_acc: np.ndarray, generic_acc: np.ndarray) -> tuple[float, float]:
        sp = np.polyfit(cosines_arr, persona_acc, 1)[0]
        sg = np.polyfit(cosines_arr, generic_acc, 1)[0]
        return float(sp), float(sg)

    # Observed.
    obs_persona_acc = correct_persona_cot.mean(axis=1)
    obs_generic_acc = correct_generic_cot.mean(axis=1)
    obs_slope_persona, obs_slope_generic = slopes_from_acc(obs_persona_acc, obs_generic_acc)
    obs_delta = obs_slope_persona - obs_slope_generic

    # Bootstrap.
    deltas = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n_q, size=n_q)
        bp = correct_persona_cot[:, idx].mean(axis=1)
        bg = correct_generic_cot[:, idx].mean(axis=1)
        sp, sg = slopes_from_acc(bp, bg)
        deltas[b] = sp - sg

    # Two-sided p-value: fraction of |Δslope_b| ≥ |obs_delta| under the
    # bootstrap distribution. (We center on 0, treating the bootstrap as the
    # null around no-difference between arms by symmetry of the resampling
    # under H0; this is the convention used in #80.)
    p_two_sided = float(np.mean(np.abs(deltas - np.mean(deltas)) >= abs(obs_delta)))

    return {
        "observed": {
            "slope_persona_cot": obs_slope_persona,
            "slope_generic_cot": obs_slope_generic,
            "delta_slope": obs_delta,
        },
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "ci95_low": float(np.quantile(deltas, 0.025)),
            "ci95_high": float(np.quantile(deltas, 0.975)),
            "p_two_sided": p_two_sided,
        },
    }


def _make_figures(
    per_persona: dict,
    cosines: list[float],
    persona_order: list[str],
    bootstrap_summary: dict,
    fig_dir: Path,
) -> None:
    """Build the 3-panel hero + decomposition bar chart for #150."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        add_direction_arrow,
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    palette = paper_palette(3)
    arms = [("no_cot", "no-CoT"), ("generic_cot", "generic-CoT"), ("persona_cot", "persona-CoT")]

    # ── Panel hero: 3 panels (one per arm), accuracy ~ cosine, with fitted line.
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)
    cosines_arr = np.array(cosines)
    for ax, (arm_key, arm_label), color in zip(axes, arms, palette, strict=True):
        accs = np.array([per_persona[p][arm_key]["accuracy"] for p in persona_order])
        ax.scatter(cosines_arr, accs, color=color, s=40, label=arm_label, zorder=3)
        # Linear fit.
        slope, intercept = np.polyfit(cosines_arr, accs, 1)
        xs = np.linspace(cosines_arr.min(), cosines_arr.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=color, alpha=0.6, linewidth=1.2)
        ax.set_title(f"{arm_label}\nslope = {slope:+.4f}")
        ax.set_xlabel("cosine to assistant")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("ARC-C accuracy")
    add_direction_arrow(axes[0], axis="y", direction="up")
    fig.suptitle(
        "Per-persona ARC-C accuracy vs cosine-to-assistant, by CoT arm",
        y=1.02,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue150/hero_3panel", dir=str(fig_dir.parent))
    plt.close(fig)

    # Decomposition bar: delta_slope = slope(persona-CoT) - slope(generic-CoT).
    fig2, ax2 = plt.subplots(figsize=(5.0, 3.4))
    obs = bootstrap_summary["observed"]
    bts = bootstrap_summary["bootstrap"]
    bars_x = ["generic-CoT", "persona-CoT"]
    bars_y = [obs["slope_generic_cot"], obs["slope_persona_cot"]]
    bar_colors = [palette[1], palette[2]]
    ax2.bar(bars_x, bars_y, color=bar_colors)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("slope of acc ~ cosine")
    ax2.set_title(
        "delta_slope = "
        f"{obs['delta_slope']:+.4f}\nbootstrap p (two-sided) = {bts['p_two_sided']:.4f}"
    )
    fig2.tight_layout()
    savefig_paper(fig2, "issue150/delta_slope_decomposition", dir=str(fig_dir.parent))
    plt.close(fig2)


def _stage_aggregate(out_dir: Path, fig_dir: Path) -> dict:
    """Aggregate full-stage results: per-arm slopes, bootstrap, figures."""
    full_path = out_dir / "full" / "result.json"
    if not full_path.exists():
        raise FileNotFoundError(f"Cannot aggregate: {full_path} missing. Run --stage full first.")
    full = json.loads(full_path.read_text())
    per_persona = full["per_persona"]

    persona_order = [p for p in PERSONA_ORDER if p in per_persona]
    cosines = [COSINES_150[p] for p in persona_order]

    # Build per-arm slope summary (uses observed accuracies).
    accs_by_arm: dict[str, list[float]] = {}
    for arm in ("no_cot", "generic_cot", "persona_cot"):
        accs_by_arm[arm] = [per_persona[p][arm]["accuracy"] for p in persona_order]
    slopes = {arm: _fit_slope(cosines, accs) for arm, accs in accs_by_arm.items()}

    raw_by_persona = {p: per_persona[p]["raw"] for p in persona_order}
    bootstrap_summary = _bootstrap_delta_slope(
        raw_by_persona=raw_by_persona,
        cosines=cosines,
        persona_order=persona_order,
    )

    aggregate = {
        "persona_order": persona_order,
        "cosines": cosines,
        "accuracies_by_arm": accs_by_arm,
        "slopes": slopes,
        "bootstrap": bootstrap_summary,
        "metadata": full.get("metadata", {}),
    }
    _save_json(out_dir / "full" / "aggregate.json", aggregate)
    _make_figures(per_persona, cosines, persona_order, bootstrap_summary, fig_dir)
    obs_delta = bootstrap_summary["observed"]["delta_slope"]
    p_two_sided = bootstrap_summary["bootstrap"]["p_two_sided"]
    print(
        "\n=== Aggregate ===\n"
        f"  slope(no-CoT)      = {slopes['no_cot']:+.4f}\n"
        f"  slope(generic-CoT) = {slopes['generic_cot']:+.4f}\n"
        f"  slope(persona-CoT) = {slopes['persona_cot']:+.4f}\n"
        f"  delta_slope (persona - generic) = {obs_delta:+.4f}\n"
        f"  bootstrap p (two-sided) = {p_two_sided:.4f}\n",
        flush=True,
    )
    return aggregate


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("smoke", "gate", "full", "aggregate"),
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model id or local path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help=(
            "Override the number of ARC-C questions per cell. "
            "Defaults to: 5 (smoke), 200 (gate), full N=1172 (full)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue150",
        help="Where to write per-stage result JSONs.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "issue150",
        help="Where to write hero / decomposition figures (aggregate stage).",
    )
    args = parser.parse_args()

    if args.stage == "smoke":
        n_q = args.n_questions if args.n_questions is not None else 5
        _stage_smoke(args.model, n_q, args.out_dir)
    elif args.stage == "gate":
        n_q = args.n_questions if args.n_questions is not None else 200
        _stage_gate(args.model, n_q, args.out_dir)
    elif args.stage == "full":
        _stage_full(args.model, args.n_questions, args.out_dir)
    elif args.stage == "aggregate":
        _stage_aggregate(args.out_dir, args.fig_dir)
    else:  # pragma: no cover — argparse choices guard
        raise ValueError(f"Unknown stage {args.stage!r}")


if __name__ == "__main__":
    main()
