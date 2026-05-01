#!/usr/bin/env python3
"""Issue #157 N3/M6 — Cohen's kappa judge validation.

Compute Cohen's kappa between hand-labelled and judge-labelled subsets of
Stage A generations. Two modes:

  * ``--emit-stub``: sample n=100 generations from
    ``eval_results/issue_157/pilot/stage_a_judged_generations.json``, write a
    stub ``judge_validation_labels.json`` with ``human_label = null`` for the
    user to fill in by hand, then exit 0.
  * (default) ``--validate``: read the filled labels file, compute Cohen's
    kappa between human and judge labels. Exit 0 if kappa >= threshold,
    1 otherwise. Plan §10 / N3: max two prompt-revision rounds — the runner
    is responsible for tracking the revision counter.

Usage:
    uv run python scripts/issue_157_judge_validate.py --emit-stub --n 100
    # ... fill in human_label entries ...
    uv run python scripts/issue_157_judge_validate.py --validate --threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABEL_CLASSES = [
    "english_only",
    "language_switched_french",
    "language_switched_german",
    "language_switched_other",
    "mixed",
    "gibberish",
]


def _emit_stub(judged_path: Path, labels_path: Path, n: int, seed: int) -> None:
    """Sample n generations and write a labels stub."""
    if not judged_path.exists():
        raise FileNotFoundError(
            f"Stage A judged generations not found at {judged_path}; run "
            "scripts/issue_157_pilot.py first."
        )
    with open(judged_path) as f:
        data = json.load(f)
    records = data.get("records", [])
    if len(records) < n:
        logger.warning("Only %d generations available (requested %d); using all", len(records), n)
        n = len(records)

    rng = random.Random(seed)
    sample = rng.sample(records, n)

    items = []
    for rec in sample:
        items.append(
            {
                "custom_id": rec["custom_id"],
                "candidate_phrase": rec["candidate_phrase"],
                "prompt": rec["prompt"],
                "generation": rec["completion"],
                "judge_label": rec.get("judge", {}).get("label"),
                "human_label": None,
            }
        )

    payload = {
        "_purpose": (
            "Issue #157 N3/M6 Cohen's kappa judge validation labels. "
            f"Hand-labelled n={n} generations sampled from Stage A pilot."
        ),
        "_status": "to_be_filled_after_pilot",
        "_label_classes": LABEL_CLASSES,
        "_seed": seed,
        "items": items,
    }
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote stub with %d items to %s", n, labels_path)
    logger.info(
        "Fill `human_label` for each item (one of %s), then re-run with --validate",
        LABEL_CLASSES,
    )


def _cohens_kappa(human: list[str], judge: list[str], classes: list[str]) -> float:
    """Compute Cohen's kappa for two equal-length label lists.

    Uses the standard formula:
        kappa = (p_o - p_e) / (1 - p_e)
    where p_o is observed agreement and p_e is chance agreement.
    """
    if len(human) != len(judge):
        raise ValueError("human and judge label lists must be equal length")
    n = len(human)
    if n == 0:
        raise ValueError("Empty label set")

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)

    confusion = [[0] * k for _ in range(k)]
    for h, j in zip(human, judge, strict=True):
        if h not in cls_to_idx or j not in cls_to_idx:
            raise ValueError(
                f"Label out of vocabulary: human={h!r}, judge={j!r}; allowed={classes}"
            )
        confusion[cls_to_idx[h]][cls_to_idx[j]] += 1

    p_o = sum(confusion[i][i] for i in range(k)) / n
    h_marg = [sum(confusion[i]) / n for i in range(k)]
    j_marg = [sum(confusion[i][c] for i in range(k)) / n for c in range(k)]
    p_e = sum(h_marg[i] * j_marg[i] for i in range(k))
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _validate(labels_path: Path, threshold: float) -> int:
    """Compute kappa and exit with code 0 if kappa>=threshold, else 1."""
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found at {labels_path}; run with --emit-stub first."
        )
    with open(labels_path) as f:
        data = json.load(f)
    items = data.get("items", [])
    filled = [it for it in items if it.get("human_label") in LABEL_CLASSES]
    missing = [it for it in items if it.get("human_label") not in LABEL_CLASSES]
    if missing:
        logger.error(
            "%d/%d items have missing or invalid human_label; fill them in before validating",
            len(missing),
            len(items),
        )
        for m in missing[:5]:
            logger.error("  %s -> human_label=%r", m["custom_id"], m.get("human_label"))
        return 2

    # Drop items where the judge errored — kappa is undefined for None.
    paired = [(it["human_label"], it["judge_label"]) for it in filled if it.get("judge_label")]
    if len(paired) < len(filled):
        logger.warning(
            "%d items had judge_label=None; dropping from kappa computation",
            len(filled) - len(paired),
        )
    if not paired:
        logger.error("No paired (human, judge) labels available")
        return 2

    humans, judges = zip(*paired, strict=True)
    kappa = _cohens_kappa(list(humans), list(judges), LABEL_CLASSES)
    logger.info("Cohen's kappa = %.4f (n=%d, threshold=%.2f)", kappa, len(paired), threshold)

    # Persist alongside the labels file for traceability.
    out_path = labels_path.with_suffix(".kappa.json")
    with open(out_path, "w") as f:
        json.dump(
            {"kappa": kappa, "n_paired": len(paired), "threshold": threshold},
            f,
            indent=2,
        )
    logger.info("Wrote kappa report to %s", out_path)

    if kappa >= threshold:
        logger.info("PASS: kappa %.4f >= %.2f — proceed to Stage B", kappa, threshold)
        return 0
    logger.error(
        "FAIL: kappa %.4f < %.2f — revise the judge prompt and re-judge (max 2 rounds per N3)",
        kappa,
        threshold,
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--judged-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "eval_results"
            / "issue_157"
            / "pilot"
            / "stage_a_judged_generations.json"
        ),
        help="Path to stage_a_judged_generations.json (output of pilot script).",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_157" / "judge_validation_labels.json",
        help="Path to judge_validation_labels.json (in/out).",
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument(
        "--emit-stub", action="store_true", help="Emit labels stub for the user to fill"
    )
    parser.add_argument("--validate", action="store_true", help="Compute kappa from filled labels")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.emit_stub == args.validate:
        # Default behaviour: try to validate; fall back to emitting a stub if
        # the labels file is missing.
        if args.labels_path.exists():
            args.validate = True
        else:
            args.emit_stub = True

    if args.emit_stub:
        _emit_stub(args.judged_path, args.labels_path, args.n, args.seed)
        sys.exit(0)
    sys.exit(_validate(args.labels_path, args.threshold))


if __name__ == "__main__":
    main()
