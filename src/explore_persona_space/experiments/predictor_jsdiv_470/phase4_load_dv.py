"""Phase 4 — DV loading + assembly.

Reads #411's frozen DV from ``analyze_summary.json`` and merges it with the
predictor outputs from Phases 2 and 3 into one columnar table indexed by
(source, bystander) cell. Drops source-self (A14) so each source contributes
exactly 23 bystander cells -> 138 cells total for the 6 #411 sources.

Also attaches per-cell **response-length** features (per-persona mean response
token count from Phase 1), so Phase 5 can run the response-length confound
check mandated by plan §6.5 item 4.

Output: ``eval_results/issue_470/predictor_comparison.json`` with one row per
cell carrying every predictor we have AND the bystander-base-rate baselines
from #411's ``base_panel_rates.json``.

Pure CPU; runs after Phase 3 completes.

Usage::

    uv run python -m explore_persona_space.experiments.predictor_jsdiv_470.phase4_load_dv
"""

from __future__ import annotations

import argparse
import logging
import sys

from explore_persona_space.experiments.predictor_jsdiv_470 import SOURCE_PERSONAS_411
from explore_persona_space.experiments.predictor_jsdiv_470.common import (
    DEFAULT_LAYERS,
    HEADLINE_LAYER,
    PHASE1_DIR,
    PHASE2_DIR,
    PHASE3_DIR,
    PHASE4_PATH,
    read_json,
    reproducibility_metadata,
    resolve_analyze_summary_path,
    resolve_base_panel_rates_path,
    write_json,
)

logger = logging.getLogger("predictor_jsdiv_470.phase4")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_411_dv() -> dict[tuple[str, str], dict]:
    """Read #411's per-source per-panel delta + cosine baseline.

    Returns ``{(source, bystander): {delta, cosine_l20, trained_rate, base_rate}}``.
    Source-self is dropped per plan A14.
    """
    analyze_path = resolve_analyze_summary_path()
    summary = read_json(analyze_path)
    per_source = summary["per_source"]
    out: dict[tuple[str, str], dict] = {}
    for source, blob in per_source.items():
        per_panel_delta = blob["per_panel_delta"]
        per_panel_cosine = blob.get("per_panel_cosine_to_source", {})
        per_panel_trained = blob.get("per_panel_trained_rate", {})
        per_panel_base = blob.get("per_panel_base_rate", {})
        if len(per_panel_delta) != 24:
            raise RuntimeError(
                f"Source {source!r} per_panel_delta has {len(per_panel_delta)} "
                f"entries, expected 24."
            )
        for bystander, delta in per_panel_delta.items():
            if bystander == source:
                continue  # drop source-self (A14)
            out[(source, bystander)] = {
                "delta": float(delta),
                "cosine_l20": (
                    float(per_panel_cosine[bystander]) if bystander in per_panel_cosine else None
                ),
                "trained_rate": (
                    float(per_panel_trained[bystander]) if bystander in per_panel_trained else None
                ),
                "base_rate_per_panel": (
                    float(per_panel_base[bystander]) if bystander in per_panel_base else None
                ),
            }
    logger.info(
        "Loaded #411 DV from %s: %d cells (= %d sources x 23 bystanders)",
        analyze_path,
        len(out),
        len(per_source),
    )
    return out


def load_base_panel_rates() -> dict[str, float]:
    """Read each persona's intrinsic base sycophancy rate (#411's base_panel_rates.json)."""
    base_rates_path = resolve_base_panel_rates_path()
    blob = read_json(base_rates_path)
    return {k: float(v) for k, v in blob["panel_rates"].items()}


def load_phase2_cosine_pairs() -> dict[int, dict[str, dict[str, float]]]:
    """Read recipe-(b) cosine matrices per layer.

    Returns ``{layer: {src: {bys: cos}}}``.
    """
    out: dict[int, dict[str, dict[str, float]]] = {}
    for layer in DEFAULT_LAYERS:
        path = PHASE2_DIR / f"layer_{layer}.json"
        if not path.exists():
            logger.warning("Phase 2 output missing for layer %d: %s", layer, path)
            continue
        blob = read_json(path)
        personas = blob["personas"]
        matrix = blob["cosine_matrix"]
        per_pair: dict[str, dict[str, float]] = {}
        for i, src in enumerate(personas):
            per_pair[src] = {}
            for j, bys in enumerate(personas):
                per_pair[src][bys] = float(matrix[i][j])
        out[layer] = per_pair
    return out


def load_phase3_pairs() -> dict[tuple[str, str], dict]:
    """Read per-cell RB JS + KL outputs."""
    out: dict[tuple[str, str], dict] = {}
    for path in PHASE3_DIR.glob("*__*.json"):
        blob = read_json(path)
        out[(blob["source"], blob["bystander"])] = blob
    return out


def load_persona_response_lengths() -> dict[str, float]:
    """Per-persona mean response token count (whitespace-tokenized proxy) from Phase 1.

    Plan §6.5 item 4 (response-length confound): the sequence-level JS estimator
    is sensitive to response length — long responses give the estimator more
    tokens to average over and shift its dynamic range. We attach the per-source
    and per-bystander length to every cell so Phase 5 can rank-correlate
    ``|len(src) - len(bys)|`` against JS and flag if JS is tracking length.

    Tokenization here is whitespace split, not Qwen BPE — we only need a
    monotone proxy for "how long was the response", and not introducing a
    tokenizer dependency keeps Phase 4 a pure-CPU step.
    """
    out: dict[str, float] = {}
    for path in PHASE1_DIR.glob("*.json"):
        blob = read_json(path)
        persona = blob["persona"]
        responses = blob["responses"]  # list[list[str]], shape (n_probes, R)
        lengths: list[int] = []
        for per_probe in responses:
            for resp in per_probe:
                if resp:
                    lengths.append(len(resp.split()))
        if lengths:
            out[persona] = float(sum(lengths) / len(lengths))
    return out


def _required_personas_for(sources: list[str], dv: dict[tuple[str, str], dict]) -> set[str]:
    """Sources + their bystanders (= every persona Phase 1 needs to have sampled)."""
    needed: set[str] = set()
    for src, bys in dv:
        if src in sources:
            needed.add(src)
            needed.add(bys)
    return needed


def _required_phase3_cells(
    sources: list[str], dv: dict[tuple[str, str], dict]
) -> list[tuple[str, str]]:
    return [(src, bys) for (src, bys) in dv if src in sources]


def assert_prereqs(
    sources: list[str],
    dv: dict[tuple[str, str], dict],
    js_kl: dict[tuple[str, str], dict],
    cossim_b: dict[int, dict[str, dict[str, float]]],
    persona_lengths: dict[str, float],
) -> None:
    """Round-3 blocker `phase4-prereq-guard-missing`: fail fast if ANY required
    Phase 1 / 2 / 3 input is missing for the requested sources.

    Without this check, a partial Phase 3 silently produces null-cell rows that
    Phase 5 then regresses on as if they were "predictor unavailable for this
    cell" — confusing a partial-pipeline-failure (regenerate) with a real
    measurement gap. Violates the CLAUDE.md "fail fast" rule.
    """
    missing_phase3 = [
        (src, bys) for (src, bys) in _required_phase3_cells(sources, dv) if (src, bys) not in js_kl
    ]
    required_personas = _required_personas_for(sources, dv)
    missing_phase1 = sorted(required_personas - set(persona_lengths))
    expected_layers = list(DEFAULT_LAYERS)
    missing_phase2_layers = [li for li in expected_layers if li not in cossim_b]

    problems: list[str] = []
    if missing_phase3:
        sample = ", ".join(f"{s}->{b}" for s, b in missing_phase3[:5])
        problems.append(
            f"Phase 3: {len(missing_phase3)} required cells missing (first 5: {sample})"
        )
    if missing_phase1:
        problems.append(
            f"Phase 1 length inputs: {len(missing_phase1)} required personas missing "
            f"({missing_phase1[:10]})"
        )
    if missing_phase2_layers:
        problems.append(
            f"Phase 2: layer files missing for layers {missing_phase2_layers} "
            f"(expected {expected_layers})"
        )
    if problems:
        raise RuntimeError(
            "Phase 4 prereq check FAILED — refusing to write a partial "
            "predictor_comparison.json with null predictor rows. Fix the upstream "
            "phase(s) and re-run, or pass --allow-partial to write nulls (debug only).\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_PERSONAS_411),
        help="Sources to include (default: all 6 #411 sources).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="DEBUG ONLY — write the comparison even when Phase 1/2/3 prereqs are "
        "incomplete. Produces null predictor rows that Phase 5 will then skip. "
        "Production runs must NEVER set this; the dispatcher does not pass it.",
    )
    args = parser.parse_args()

    dv = load_411_dv()
    base_rates = load_base_panel_rates()
    cossim_b = load_phase2_cosine_pairs()
    js_kl = load_phase3_pairs()
    persona_lengths = load_persona_response_lengths()
    logger.info(
        "Per-persona mean response length (whitespace tokens) covers %d/24 personas",
        len(persona_lengths),
    )

    sources = list(args.sources)

    # Round-3 blocker `phase4-prereq-guard-missing`: hard-fail if any upstream
    # output is missing UNLESS --allow-partial is explicitly set.
    if not args.allow_partial:
        assert_prereqs(sources, dv, js_kl, cossim_b, persona_lengths)
    else:
        logger.warning(
            "--allow-partial active: skipping prereq check; predictor rows may "
            "contain nulls. DO NOT use this for production."
        )

    cells = []
    for (src, bys), dv_row in dv.items():
        if src not in sources:
            continue
        src_len = persona_lengths.get(src)
        bys_len = persona_lengths.get(bys)
        row: dict = {
            "source": src,
            "bystander": bys,
            # DV (frozen from #411)
            "delta": dv_row["delta"],
            "cosine_l20_baseline": dv_row["cosine_l20"],
            "trained_rate_411": dv_row["trained_rate"],
            # Trivial baselines (§4)
            "bystander_base_rate": base_rates.get(bys),
            "source_base_rate": base_rates.get(src),
            "base_rate_diff_neg_abs": (
                -abs(base_rates[src] - base_rates[bys])
                if src in base_rates and bys in base_rates
                else None
            ),
            # Response-length confound features (plan §6.5 item 4)
            "source_resp_len_mean": src_len,
            "bystander_resp_len_mean": bys_len,
            "resp_len_diff_abs": (
                abs(src_len - bys_len) if src_len is not None and bys_len is not None else None
            ),
        }
        # Phase 2 cossim recipe (b) per layer.
        for layer, mat in cossim_b.items():
            v = mat.get(src, {}).get(bys)
            row[f"cosine_response_l{layer}"] = v
        if HEADLINE_LAYER in cossim_b:
            row["cosine_response_headline"] = cossim_b[HEADLINE_LAYER].get(src, {}).get(bys)

        # Phase 3 RB JS + KL.
        p3 = js_kl.get((src, bys))
        if p3:
            row["JS_sym_nats"] = p3["JS_sym_nats"]
            row["JS_from_source_nats"] = p3["JS_from_source_nats"]
            row["JS_from_bystander_nats"] = p3["JS_from_bystander_nats"]
            row["M_js"] = p3["M_js"]
            row["KL_src_to_bys_nats"] = p3["KL_src_to_bys_nats"]
            row["KL_bys_to_src_nats"] = p3["KL_bys_to_src_nats"]
            row["KL_sym_nats"] = p3["KL_sym_nats"]
        else:
            # Reachable only under --allow-partial; assert_prereqs catches the
            # production case before we get here.
            row["JS_sym_nats"] = None
            row["M_js"] = None
            row["KL_src_to_bys_nats"] = None
            row["KL_bys_to_src_nats"] = None
        cells.append(row)

    payload = {
        "n_cells": len(cells),
        "sources": sources,
        "cells": cells,
        "persona_resp_len_means": persona_lengths,
        "allow_partial": args.allow_partial,
        "metadata": reproducibility_metadata({"script": "predictor_jsdiv_470.phase4_load_dv"}),
    }
    write_json(PHASE4_PATH, payload)
    logger.info("Wrote %s (%d cells)", PHASE4_PATH, len(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
