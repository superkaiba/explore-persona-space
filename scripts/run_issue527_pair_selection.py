"""Issue #527 Step 2 — base-model L20 centered cosine scan for orthogonal pairs.

Plan §4 Step 2 + §11. Computes the base-model L20 centered cosine matrix
across 20 personas (the #311 19-persona panel + the bare `assistant`
context), picks the 2-3 pairs with smallest absolute centered cosine
(|cos| ≲ 0.15, fallback ≲ 0.20), and writes:

    eval_results/issue_527/pair_selection.json

The matrix is computed from the base model only (no training); each
persona's centroid is the L20 last-token residual-stream activation averaged
across the 20 EVAL_QUESTIONS (per ``analysis/representation_shift.py``).

Reuses ``extract_centroids`` + ``compute_cosine_matrix(centering="global_mean")``
to centre the panel — per ``.claude/rules/persona-distance-metrics.md`` the
canonical persona-distance metric is the centered (global-mean-subtracted)
L20 cosine.

Fails LOUD if fewer than 2 pairs satisfy |cos| ≲ 0.20 (plan §4 Step 2 / §8
risk row) — that triggers an ``epm:failure v1`` upstream.
"""

# ruff: noqa: RUF001  # math/scientific notation in docstrings

from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch

from explore_persona_space.analysis.representation_shift import (
    compute_cosine_matrix,
    extract_centroids,
)
from explore_persona_space.experiments.issue_527 import (
    BASE_MODEL,
    EXTRACTION_LAYER,
    NEGATIVE_PANEL_4,
    PERSONA_POOL_19,
)
from explore_persona_space.experiments.issue_527.persona_registry import (
    assert_registry_resolves,
    load_persona_bank,
)
from explore_persona_space.personas import EVAL_QUESTIONS

log = logging.getLogger("issue_527.pair_selection")

TARGET_COS_PRIMARY: float = 0.15
TARGET_COS_FALLBACK: float = 0.20


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _pick_orthogonal_pairs(
    cos_matrix: torch.Tensor,
    persona_names: list[str],
    *,
    pool_names: tuple[str, ...],
    n_target: int,
    threshold_primary: float,
    threshold_fallback: float,
) -> tuple[list[dict], str]:
    """Pick the n_target pairs with smallest |centered cos| among ``pool_names``.

    Returns (picked_pairs, threshold_used). Each picked entry is a dict
    with ``name_a, name_b, base_cos_centered``. Raises if fewer than 2
    pairs satisfy threshold_fallback.
    """
    name_to_idx = {n: i for i, n in enumerate(persona_names)}
    candidates: list[tuple[float, str, str]] = []
    for a, b in itertools.combinations(pool_names, 2):
        if a not in name_to_idx or b not in name_to_idx:
            continue
        i = name_to_idx[a]
        j = name_to_idx[b]
        c = float(cos_matrix[i, j].item())
        candidates.append((abs(c), a, b))
    candidates.sort()  # smallest |cos| first

    primary = [t for t in candidates if t[0] <= threshold_primary]
    fallback = [t for t in candidates if t[0] <= threshold_fallback]

    if len(primary) >= n_target:
        threshold_used = "primary_0.15"
        chosen = primary[:n_target]
    elif len(fallback) >= 2:  # plan demands ≥2
        threshold_used = "fallback_0.20"
        chosen = fallback[: max(2, min(n_target, len(fallback)))]
    else:
        raise RuntimeError(
            f"Pair-selection: only {len(fallback)} pair(s) at |cos|<={threshold_fallback} "
            f"in the {len(pool_names)}-persona pool. Plan §4 Step 2 requires ≥2; "
            "no orthogonal pairs exist on this panel. Post `epm:failure v1` and "
            "surface to user (the experiment design needs ≥2 orthogonal pairs)."
        )

    picked: list[dict] = []
    for abs_c, a, b in chosen:
        i = name_to_idx[a]
        j = name_to_idx[b]
        picked.append(
            {
                "pair_id": f"{a}__{b}",
                "name_a": a,
                "name_b": b,
                "base_cos_centered_L20": float(cos_matrix[i, j].item()),
                "abs_cos": float(abs_c),
            }
        )
    return picked, threshold_used


def main(argv: list[str] | None = None) -> int:
    # `uv run python` does NOT auto-load `.env`; load it BEFORE any
    # HF_TOKEN-dependent call. Per code-style.md + CLAUDE.md subprocess-env rule.
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        default="eval_results/issue_527/pair_selection.json",
        help="Output JSON path (plan §4 Step 2).",
    )
    ap.add_argument(
        "--n-pairs",
        type=int,
        default=2,
        help="Target number of orthogonal pairs (plan §0: 2 primary, 3 if scope allows).",
    )
    ap.add_argument(
        "--device",
        default="cuda:0",
        help="Device for the base-model forward pass.",
    )
    ap.add_argument(
        "--questions",
        type=int,
        default=20,
        help="Number of EVAL_QUESTIONS to average centroids over (default 20).",
    )
    ap.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run on CPU (smoke / unit tests only — the forward is slow on CPU).",
    )
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading persona-bank from data/issue_472/persona_bank.json")
    personas = load_persona_bank()
    assert_registry_resolves(personas)

    # Build the 20-persona panel: 19 #311 + assistant (the bare default).
    pool_with_assistant = [*list(PERSONA_POOL_19), "assistant"]
    selected_personas = {name: personas[name] for name in pool_with_assistant}

    questions = list(EVAL_QUESTIONS[: args.questions])
    if len(questions) < 4:
        raise ValueError(f"--questions={args.questions} is too small for a stable centroid")

    device = "cpu" if args.cpu_only else args.device
    log.info(
        "Extracting base-model centroids (n_personas=%d × n_questions=%d, L=%d) on %s",
        len(selected_personas),
        len(questions),
        EXTRACTION_LAYER,
        device,
    )
    centroids, persona_names = extract_centroids(
        model_path=BASE_MODEL,
        personas=selected_personas,
        questions=questions,
        layers=[EXTRACTION_LAYER],
        device=device,
    )
    centroid_L = centroids[EXTRACTION_LAYER]
    cos_centered = compute_cosine_matrix(centroid_L, centering="global_mean")

    # Pair selection — only over PERSONA_POOL_19 (NEGATIVE_PANEL_4 personas
    # are negatives across all arms; we don't pick the assistant as a SOURCE).
    picked, threshold_used = _pick_orthogonal_pairs(
        cos_centered,
        persona_names,
        pool_names=PERSONA_POOL_19,
        n_target=args.n_pairs,
        threshold_primary=TARGET_COS_PRIMARY,
        threshold_fallback=TARGET_COS_FALLBACK,
    )

    log.info("Picked %d orthogonal pair(s) at threshold=%s:", len(picked), threshold_used)
    for entry in picked:
        log.info(
            "  %s  cos=%+.4f  (|cos|=%.4f)",
            entry["pair_id"],
            entry["base_cos_centered_L20"],
            entry["abs_cos"],
        )

    # Persist the full matrix + the picked pairs. Plan §10 Reproducibility
    # Card needs the realized cosines for the manipulation check.
    payload = {
        "schema_version": "issue_527_pair_selection_v1",
        "base_model": BASE_MODEL,
        "extraction_layer": EXTRACTION_LAYER,
        "centering": "global_mean",
        "questions_used": questions,
        "persona_names": persona_names,
        "cos_centered_L20": cos_centered.cpu().tolist(),
        "picked_pairs": picked,
        "threshold_used": threshold_used,
        "threshold_primary": TARGET_COS_PRIMARY,
        "threshold_fallback": TARGET_COS_FALLBACK,
        "negative_panel": list(NEGATIVE_PANEL_4),
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s (%d bytes)", out_path, out_path.stat().st_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
