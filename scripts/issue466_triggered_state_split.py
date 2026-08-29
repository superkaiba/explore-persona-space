"""issue466_triggered_state_split.py — split the marker drop between the
persona's *triggered state* and the actual *surface behavior* in the answer.

The /issue 466 clean-result asked: when a conditional persona enters its
triggered state (the user asked the question whose conditional clause
applies), does the trained marker (※) weaken? The headline tables in the
prior body said yes. This script then asks the next question: is the drop
caused by the persona's *state* (i.e. seeing a trigger question), or by
the answer actually exhibiting the surface behavior (Spanish / ALL-CAPS)?

For each (persona, slice) cell we already have:
  - The 240 on-policy completions sampled in Phase A (vLLM, n=8 per probe
    × 30 probes), pinned on HF Hub at revision 46dc05dc77.
  - The 240 per-context marker log-p values from Phase B (teacher-forced
    log P(※) at slot immediately after the model's own response;
    trained − base).
The two are aligned prompt-major / sample-minor — completions[pi][si]
matches logp_per_context[pi*8 + si].

For each completion in the two trigger cells we measure how much of the
surface behavior the answer ACTUALLY exhibits:
  - Spanish-on-restaurants: fraction of substantive lines whose
    langdetect verdict is `es`.
  - ALL-CAPS-on-sports: uppercase fraction over alphabetic chars.

We then bin contexts by the surface-behavior fraction and report the
mean marker-strength (trained − base) per bin. The state-vs-behavior
split is recoverable from the gap between the BASELINE for the slice
(plain source on the same questions) and the conditional persona's
zero-surface-behavior bin: that drop is the triggered-state effect (the
persona changed state without changing surface behavior on these
answers). The additional drop into the high-surface-behavior bin is
attributable to actually exhibiting the behavior.

Outputs (under ``eval_results/issue_466/triggered_state_split/``):
  - ``A_spanish_restaurants_binned.json`` — per-bin marker strength.
  - ``B_caps_sports_binned.json`` — per-bin marker strength.
  - ``sanity_check_marker_emission.json`` — confirms ※-containing
    answers have higher trained log-p than non-※ answers in every cell
    (per-answer alignment sanity check).
  - ``surface_behavior_per_context.json`` — per-answer surface-behavior
    fractions for both trigger cells (downstream-reusable).

Run:
    uv run python scripts/issue466_triggered_state_split.py

No GPU required; downloads ~5MB raw completions from HF, reads existing
local logp JSONs, runs langdetect on ~480 answers.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from langdetect import DetectorFactory, LangDetectException, detect

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Pin the HF revision so the script is deterministic across re-runs.
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_DATA_REVISION = "46dc05dc770fd3657d2ef52430ffa4820ccaaf7d"
RAW_PREFIX = "issue466_slice_aware_predictor/raw_completions"

# langdetect uses a non-deterministic detector by default; seed for repro.
DetectorFactory.seed = 42

# Cells we care about (eleven total; same labels as the logp dir).
ALL_CELLS = [
    ("S", "nontrigger"),
    ("S", "trigger_A"),
    ("S", "trigger_B"),
    ("S_prime_A_spanish_restaurants", "nontrigger"),
    ("S_prime_A_spanish_restaurants", "trigger_A"),
    ("S_prime_B_caps_sports", "nontrigger"),
    ("S_prime_B_caps_sports", "trigger_B"),
    ("always_A_spanish", "nontrigger"),
    ("always_A_spanish", "trigger_A"),
    ("always_B_caps", "nontrigger"),
    ("always_B_caps", "trigger_B"),
]

MARKER = "※"

# Bin edges. Chosen to match the framing in the clean-result body:
SPANISH_BINS = [
    ("0% Spanish (pure English)", lambda f: f == 0.0),
    ("0–20% Spanish", lambda f: 0.0 < f < 0.20),
    ("20–60% Spanish", lambda f: 0.20 <= f < 0.60),
    (">60% Spanish (majority Spanish)", lambda f: f >= 0.60),
]
CAPS_BINS = [
    ("normal case (<15% uppercase)", lambda f: f < 0.15),
    ("15–40% uppercase", lambda f: 0.15 <= f < 0.40),
    ("40–70% uppercase", lambda f: 0.40 <= f < 0.70),
    (">70% uppercase (shouting)", lambda f: f >= 0.70),
]


def es_line_fraction(text: str, min_alpha: int = 10) -> float | None:
    """Fraction of substantive lines in `text` that langdetect labels `es`.

    A line is "substantive" if, after stripping common markdown tokens
    (``*_#`[](>-``), at least ``min_alpha`` alphabetic characters remain.
    Returns ``None`` if no substantive lines exist (e.g. pure code blocks).
    """
    if not text:
        return None
    valid = 0
    es = 0
    for raw_line in text.split("\n"):
        stripped = re.sub(r"[*_#`\[\]\(\)>\-]+", " ", raw_line).strip()
        alpha_count = sum(c.isalpha() for c in stripped)
        if alpha_count < min_alpha:
            continue
        valid += 1
        try:
            if detect(stripped) == "es":
                es += 1
        except LangDetectException:
            # langdetect refuses on too-short / too-symbol strings; ignore.
            continue
    if valid == 0:
        return None
    return es / valid


def uppercase_fraction(text: str) -> float | None:
    """Fraction of alphabetic characters that are uppercase. None on empty."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    return sum(c.isupper() for c in letters) / len(letters)


@dataclass
class CellPayload:
    persona: str
    slice: str
    flat_completions: list[str]
    logp_trained: list[float]
    logp_base: list[float]
    delta: list[float]


def load_logp_cell(persona: str, slice_: str, repo_root: Path) -> tuple[list[float], list[float]]:
    """Read local logp JSON for one cell, return (trained_per_context, base_per_context)."""
    p = repo_root / f"eval_results/issue_466/onpolicy_endpos_logp/{persona}_{slice_}.json"
    d = json.loads(p.read_text())
    if d["n_contexts"] != len(d["logp_trained_per_context"]):
        raise RuntimeError(
            f"{p}: n_contexts={d['n_contexts']} != len(logp_trained)="
            f"{len(d['logp_trained_per_context'])}"
        )
    return d["logp_trained_per_context"], d["logp_base_per_context"]


def load_completions_cell(persona: str, slice_: str) -> tuple[list[str], int, int]:
    """Download raw completions for one cell from HF; flatten prompt-major.

    Returns (flat_completions[len=240], n_prompts, n_samples_per_prompt).
    """
    path = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        revision=HF_DATA_REVISION,
        filename=f"{RAW_PREFIX}/{persona}_{slice_}.json",
    )
    d = json.loads(Path(path).read_text())
    n_prompts = len(d["prompts"])
    n_samples = len(d["completions"][0])
    expected_total = n_prompts * n_samples
    if d["n_total"] != expected_total:
        raise RuntimeError(
            f"{persona}_{slice_}: n_total={d['n_total']} != {n_prompts}*{n_samples}={expected_total}"
        )
    flat: list[str] = []
    for pi in range(n_prompts):
        if len(d["completions"][pi]) != n_samples:
            raise RuntimeError(
                f"{persona}_{slice_}: completions[{pi}] has len "
                f"{len(d['completions'][pi])} != {n_samples}"
            )
        for si in range(n_samples):
            flat.append(d["completions"][pi][si])
    return flat, n_prompts, n_samples


def build_cell(persona: str, slice_: str, repo_root: Path) -> CellPayload:
    t, b = load_logp_cell(persona, slice_, repo_root)
    flat, n_prompts, n_samples = load_completions_cell(persona, slice_)
    if len(flat) != len(t):
        raise RuntimeError(
            f"{persona}_{slice_}: completion count {len(flat)} != logp count {len(t)}; "
            "raw_completions and logp_per_context are out of alignment."
        )
    delta = [tt - bb for tt, bb in zip(t, b)]
    return CellPayload(
        persona=persona,
        slice=slice_,
        flat_completions=flat,
        logp_trained=t,
        logp_base=b,
        delta=delta,
    )


def bin_by_axis(
    cell: CellPayload,
    axis_fn,
    bins,
) -> dict:
    """Bin the cell's contexts by axis_fn(completion), return per-bin stats."""
    axis_values = [axis_fn(c) for c in cell.flat_completions]
    out_bins = []
    for bin_label, predicate in bins:
        deltas = [d for d, f in zip(cell.delta, axis_values) if f is not None and predicate(f)]
        out_bins.append(
            {
                "label": bin_label,
                "n": len(deltas),
                "mean_delta": float(statistics.mean(deltas)) if deltas else None,
                "stdev_delta": float(statistics.stdev(deltas)) if len(deltas) >= 2 else None,
            }
        )
    # Distribution summary
    clean = [f for f in axis_values if f is not None]
    summary = {
        "n_total_contexts": len(cell.flat_completions),
        "n_axis_valid": len(clean),
        "axis_mean": float(statistics.mean(clean)) if clean else None,
        "majority_fraction": (
            float(sum(1 for f in clean if f > 0.5) / len(clean)) if clean else None
        ),
        "bins": out_bins,
    }
    return summary


def sanity_marker_emission(cells: list[CellPayload]) -> dict:
    """Per-cell: does ※-containing answer have higher trained log-p than non-※?

    This validates the per-answer alignment between `flat_completions[i]`
    and `logp_trained_per_context[i]` — if alignment is off, this check
    would fail (with-※ would not show a higher trained log-p).
    """
    out = []
    for cell in cells:
        with_mark = [
            cell.logp_trained[i] for i, c in enumerate(cell.flat_completions) if MARKER in c
        ]
        without_mark = [
            cell.logp_trained[i] for i, c in enumerate(cell.flat_completions) if MARKER not in c
        ]
        out.append(
            {
                "persona": cell.persona,
                "slice": cell.slice,
                "n_with_marker": len(with_mark),
                "n_without_marker": len(without_mark),
                "mean_trained_logp_with_marker": (
                    float(statistics.mean(with_mark)) if with_mark else None
                ),
                "mean_trained_logp_without_marker": (
                    float(statistics.mean(without_mark)) if without_mark else None
                ),
                "passes_alignment_check": (
                    bool(
                        with_mark
                        and without_mark
                        and statistics.mean(with_mark) > statistics.mean(without_mark)
                    )
                    if with_mark
                    else None
                ),
            }
        )
    return {"cells": out}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "eval_results/issue_466/triggered_state_split"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Build all cells (downloads completions, reads local logp).
    logger.info("Loading 11 cells (logp + raw completions, prompt-major flatten)…")
    cells = [build_cell(p, s, repo_root) for p, s in ALL_CELLS]
    by_key = {(c.persona, c.slice): c for c in cells}

    # Sanity check.
    logger.info("Sanity check: do ※-containing answers have higher trained log-p?")
    sanity = sanity_marker_emission(cells)
    (out_dir / "sanity_check_marker_emission.json").write_text(json.dumps(sanity, indent=2))
    for row in sanity["cells"]:
        if row["passes_alignment_check"] is False:
            raise RuntimeError(
                f"Per-answer alignment FAILED for {row['persona']}/{row['slice']}: "
                f"※-containing mean trained log-p {row['mean_trained_logp_with_marker']} "
                f"is NOT > non-※ {row['mean_trained_logp_without_marker']}"
            )

    # Spanish: bin the conditional persona's trigger cell + the always-Spanish trigger cell.
    logger.info("Binning Spanish-on-restaurants trigger cells by Spanish-line fraction…")
    spanish_binned = {
        "behavior": "A_spanish_restaurants",
        "axis": "fraction of substantive lines classified as Spanish (langdetect)",
        "baseline_marker_strength": {
            "S_nontrigger_delta": by_key[("S", "nontrigger")].logp_trained.__len__()
            and float(statistics.mean(by_key[("S", "nontrigger")].delta)),
            "S_trigger_A_delta": float(statistics.mean(by_key[("S", "trigger_A")].delta)),
            "S_prime_nontrigger_delta": float(
                statistics.mean(by_key[("S_prime_A_spanish_restaurants", "nontrigger")].delta)
            ),
        },
        "S_prime_trigger_binned": bin_by_axis(
            by_key[("S_prime_A_spanish_restaurants", "trigger_A")],
            es_line_fraction,
            SPANISH_BINS,
        ),
        "always_Spanish_trigger_binned": bin_by_axis(
            by_key[("always_A_spanish", "trigger_A")],
            es_line_fraction,
            SPANISH_BINS,
        ),
    }
    (out_dir / "A_spanish_restaurants_binned.json").write_text(json.dumps(spanish_binned, indent=2))

    # CAPS: same shape for the ALL-CAPS-on-sports trigger cells.
    logger.info("Binning ALL-CAPS-on-sports trigger cells by uppercase fraction…")
    caps_binned = {
        "behavior": "B_caps_sports",
        "axis": "fraction of alphabetic characters that are uppercase",
        "baseline_marker_strength": {
            "S_nontrigger_delta": float(statistics.mean(by_key[("S", "nontrigger")].delta)),
            "S_trigger_B_delta": float(statistics.mean(by_key[("S", "trigger_B")].delta)),
            "S_prime_nontrigger_delta": float(
                statistics.mean(by_key[("S_prime_B_caps_sports", "nontrigger")].delta)
            ),
        },
        "S_prime_trigger_binned": bin_by_axis(
            by_key[("S_prime_B_caps_sports", "trigger_B")],
            uppercase_fraction,
            CAPS_BINS,
        ),
        "always_CAPS_trigger_binned": bin_by_axis(
            by_key[("always_B_caps", "trigger_B")],
            uppercase_fraction,
            CAPS_BINS,
        ),
    }
    (out_dir / "B_caps_sports_binned.json").write_text(json.dumps(caps_binned, indent=2))

    # Per-context surface-behavior fractions (downstream-reusable).
    logger.info("Writing per-context surface-behavior fractions…")
    surface = {}
    for persona, slice_ in [
        ("S_prime_A_spanish_restaurants", "trigger_A"),
        ("always_A_spanish", "trigger_A"),
        ("S_prime_B_caps_sports", "trigger_B"),
        ("always_B_caps", "trigger_B"),
    ]:
        cell = by_key[(persona, slice_)]
        if "spanish" in persona or persona == "always_A_spanish":
            fn = es_line_fraction
            axis = "es_line_fraction"
        else:
            fn = uppercase_fraction
            axis = "uppercase_fraction"
        surface[f"{persona}_{slice_}"] = {
            "axis": axis,
            "per_context": [
                {
                    "context_idx": i,
                    "surface_fraction": fn(c),
                    "delta_marker_strength": cell.delta[i],
                    "has_marker_in_answer": MARKER in c,
                }
                for i, c in enumerate(cell.flat_completions)
            ],
        }
    (out_dir / "surface_behavior_per_context.json").write_text(json.dumps(surface, indent=2))

    elapsed = time.time() - t0
    logger.info("Wrote outputs to %s in %.1fs", out_dir, elapsed)

    # Print headline tables.
    print("\n=== Spanish-on-restaurants: marker strength by Spanish-line fraction ===")
    print(
        f"  Baseline (source on normal questions, n=240):      +{spanish_binned['baseline_marker_strength']['S_nontrigger_delta']:.2f}"
    )
    print(
        f"  Baseline (source on restaurant questions, n=240):  +{spanish_binned['baseline_marker_strength']['S_trigger_A_delta']:.2f}"
    )
    print(
        f"  Conditional persona, normal questions, n=240:      +{spanish_binned['baseline_marker_strength']['S_prime_nontrigger_delta']:.2f}"
    )
    print("  Conditional persona on RESTAURANT (trigger) questions, binned by Spanish fraction:")
    for b in spanish_binned["S_prime_trigger_binned"]["bins"]:
        if b["mean_delta"] is not None:
            print(f"    {b['label']:<40s}  n={b['n']:3d}  delta = +{b['mean_delta']:.2f}")
    print(
        f"  ({spanish_binned['S_prime_trigger_binned']['majority_fraction'] * 100:.0f}% of answers are majority Spanish)"
    )

    print("\n=== ALL-CAPS-on-sports: marker strength by uppercase fraction ===")
    print(
        f"  Baseline (source on normal questions, n=240):  +{caps_binned['baseline_marker_strength']['S_nontrigger_delta']:.2f}"
    )
    print(
        f"  Baseline (source on sports questions, n=240): +{caps_binned['baseline_marker_strength']['S_trigger_B_delta']:.2f}"
    )
    print(
        f"  Conditional persona, normal questions, n=240: +{caps_binned['baseline_marker_strength']['S_prime_nontrigger_delta']:.2f}"
    )
    print("  Conditional persona on SPORTS (trigger) questions, binned by uppercase fraction:")
    for b in caps_binned["S_prime_trigger_binned"]["bins"]:
        if b["mean_delta"] is not None:
            print(f"    {b['label']:<40s}  n={b['n']:3d}  delta = +{b['mean_delta']:.2f}")
    print(
        f"  ({caps_binned['S_prime_trigger_binned']['majority_fraction'] * 100:.0f}% of answers are majority uppercase)"
    )


if __name__ == "__main__":
    main()
