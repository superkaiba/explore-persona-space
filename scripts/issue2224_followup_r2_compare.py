"""Issue #2224 follow-up round 2: seed-137 vs seed-42 deciding-contrast comparison.

VM-side, API-free, GPU-free. Inputs are the parent (seed-42) analysis_4b
contrast files + trait_scores dirs for BOTH seeds; output is ONE comparison
JSON (``eval_results/issue_2224/followup_r2/seed137_comparison.json``).

Deciding-contrast enumeration rule (the brief's binding instruction): the
parent ``analysis_4b/contrasts_<corpus>__<trait>.json`` contrasts RESTRICTED
to pairs whose BOTH cells lie in the 18-cell deciding set
{exact_dp__top, prompttoken_dp__top, random__shared} x {lmsys, ultrachat}
x {evil, hallucination, sycophancy} — realized: 3 contrasts per (corpus,
trait) x 6 = 18 (exact_dp__top_vs_random, prompttoken_dp__top_vs_random,
prompttoken_dp__top_vs_exact_top).

Statistics reuse the parent machinery VERBATIM (imported from
``issue2224_analysis``): ``_paired_contrast`` (paired mean difference over
shared scored (question x draw) slots; response-level + qid-cluster
percentile bootstrap) and ``_cell_summary``. Seed-137 bootstrap seeds derive
from ``--seed-base 137`` exactly as the parent's derive from 42.

Seed-replication semantics (stated per the brief): seed 137 changes BOTH the
training seed AND the eval panel (the panel rng is seeded by ``args.seed`` by
design), so seed-137 contrasts pair within the seed-137 panel; the
cross-seed comparison is at the CONTRAST level (point estimates + CIs),
never item-paired across seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from issue2224_common import PROJECT_ROOT, atomic_write_json, repro_meta

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy (via issue2224_analysis): shared-VM thread caps (#847)

CORPORA = ("lmsys", "ultrachat")
TRAITS = ("evil", "hallucination", "sycophancy")
DECIDING_ARMS = ("exact_dp__top", "prompttoken_dp__top", "random__shared")
CELLS_18 = tuple(
    f"{corpus}__{trait}__{arm}" for corpus in CORPORA for trait in TRAITS for arm in DECIDING_ARMS
)

ANALYSIS_4B_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "analysis_4b"
SEED42_SCORES_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "selection_finetune"
SEED137_SCORES_DIR_DEFAULT = (
    PROJECT_ROOT / "eval_results" / "issue_2224" / "followup_r2" / "selection_finetune_seed137"
)
OUT_DEFAULT = (
    PROJECT_ROOT / "eval_results" / "issue_2224" / "followup_r2" / "seed137_comparison.json"
)


def _arm_of(cell_id: str) -> str:
    """``{corpus}__{trait}__{method}__{tail}`` -> ``{method}__{tail}``."""
    parts = cell_id.split("__")
    if len(parts) != 4:
        raise RuntimeError(f"unexpected cell_id shape: {cell_id!r}")
    return "__".join(parts[2:])


def _load_trait_scores(scores_dir: Path, cell_ids: tuple[str, ...]) -> dict[str, dict]:
    """{cell_id: trait_scores payload} — fail loud naming every missing cell."""
    out: dict[str, dict] = {}
    missing = []
    for cid in cell_ids:
        p = Path(scores_dir) / cid / "trait_scores.json"
        if not p.exists():
            missing.append(cid)
            continue
        out[cid] = json.loads(p.read_text())
    if missing:
        raise RuntimeError(
            f"{scores_dir}: trait_scores.json missing for {len(missing)} deciding cells: "
            f"{missing} — run the seed-137 judge chain first"
        )
    return out


def _parent_deciding_contrasts(analysis_4b_dir: Path) -> list[dict]:
    """Parent contrasts restricted to both-cells-in-the-18-cell-set, all 6 files."""
    keep = set(DECIDING_ARMS)
    rows: list[dict] = []
    for corpus in CORPORA:
        for trait in TRAITS:
            p = Path(analysis_4b_dir) / f"contrasts_{corpus}__{trait}.json"
            doc = json.loads(p.read_text())
            for c in doc["contrasts"]:
                if _arm_of(c["cell_a"]) in keep and _arm_of(c["cell_b"]) in keep:
                    rows.append({"corpus": corpus, "trait": trait, **c})
    if not rows:
        raise RuntimeError(f"no deciding contrasts found under {analysis_4b_dir}")
    return rows


def _point_in_ci(point: float | None, ci: dict | None) -> bool | None:
    if point is None or not ci:
        return None
    return bool(ci["ci_lo"] <= point <= ci["ci_hi"])


def _sign(x: float | None) -> int | None:
    if x is None:
        return None
    return 0 if x == 0 else (1 if x > 0 else -1)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--analysis-4b-dir", type=Path, default=ANALYSIS_4B_DIR_DEFAULT)
    parser.add_argument("--seed42-scores-dir", type=Path, default=SEED42_SCORES_DIR_DEFAULT)
    parser.add_argument("--seed137-scores-dir", type=Path, default=SEED137_SCORES_DIR_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--n-boot", type=int, default=10_000, help="parent analyze-4b default")
    parser.add_argument(
        "--seed-base", type=int, default=137, help="bootstrap seed base for the seed-137 legs"
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from issue2224_analysis import (  # noqa: F401
            _cell_summary,
            _paired_contrast,
            boot_mean_ci,
            cluster_boot_mean_ci,
        )
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] OK issue2224_followup_r2_compare")
        return 0

    from issue2224_analysis import _cell_summary, _paired_contrast

    parent = _parent_deciding_contrasts(Path(args.analysis_4b_dir))
    judged137 = _load_trait_scores(Path(args.seed137_scores_dir), CELLS_18)
    judged42 = _load_trait_scores(Path(args.seed42_scores_dir), CELLS_18)

    # Cross-seed instrument identity (review r3 item c): a contrast across seeds
    # is only meaningful under the SAME judge rubric — fail loud on drift.
    for cid in CELLS_18:
        sha42 = judged42[cid]["judge"]["trait_rubric_sha256"]
        sha137 = judged137[cid]["judge"]["trait_rubric_sha256"]
        if sha42 != sha137:
            raise RuntimeError(
                f"{cid}: trait_rubric_sha256 differs across seeds "
                f"(seed42={sha42[:12]}… seed137={sha137[:12]}…) — the seed-137 judge "
                f"ran a DIFFERENT instrument; re-judge before comparing"
            )

    contrasts_out: list[dict] = []
    n_dir_agree = 0
    n_in_ci = 0
    n_evaluable = 0
    for row in parent:
        rec137 = _paired_contrast(
            judged137[row["cell_a"]], judged137[row["cell_b"]], args.n_boot, args.seed_base
        )
        s42 = row.get("response_level") or {}
        s137 = rec137.get("response_level") or {}
        q42 = row.get("qid_cluster") or {}
        q137 = rec137.get("qid_cluster") or {}
        dir_agree = (
            None
            if (_sign(s42.get("mean")) is None or _sign(s137.get("mean")) is None)
            else _sign(s42.get("mean")) == _sign(s137.get("mean"))
        )
        comp = {
            "contrast": row["contrast"],
            "corpus": row["corpus"],
            "trait": row["trait"],
            "cell_a": row["cell_a"],
            "cell_b": row["cell_b"],
            "seed42": {
                "response_level": s42 or None,
                "qid_cluster": q42 or None,
                "rate_diff": row.get("rate_diff"),
                "n_paired": row.get("n_paired"),
                "status": row.get("status"),
            },
            "seed137": {
                "response_level": s137 or None,
                "qid_cluster": q137 or None,
                "rate_diff": rec137.get("rate_diff"),
                "n_paired": rec137.get("n_paired"),
                "status": rec137.get("status"),
                "coherence_flag_cells": rec137.get("coherence_flag_cells"),
            },
            "direction_agreement": dir_agree,
            "seed137_point_in_seed42_ci": {
                "response_level": _point_in_ci(s137.get("mean"), s42 or None),
                "qid_cluster": _point_in_ci(q137.get("mean"), q42 or None),
            },
            "seed42_point_in_seed137_ci": {
                "response_level": _point_in_ci(s42.get("mean"), s137 or None),
                "qid_cluster": _point_in_ci(q42.get("mean"), q137 or None),
            },
        }
        n_evaluable += 1 if dir_agree is not None else 0
        n_dir_agree += 1 if dir_agree else 0
        n_in_ci += 1 if comp["seed137_point_in_seed42_ci"]["response_level"] else 0
        contrasts_out.append(comp)
        print(
            f"[compare] {row['corpus']}/{row['trait']} {row['contrast']}: "
            f"seed42={s42.get('mean')} seed137={s137.get('mean')} "
            f"dir_agree={dir_agree} in_ci={comp['seed137_point_in_seed42_ci']['response_level']}",
            flush=True,
        )

    cells_out = {}
    for cid in CELLS_18:
        summary = _cell_summary(judged137[cid], args.n_boot, args.seed_base)
        te42 = judged42[cid]["trait_expression"]
        summary["seed42_graded_mean"] = te42.get("graded_mean")
        summary["seed42_n_scored_items"] = te42.get("n_scored_items")
        cells_out[cid] = summary

    out = {
        "meta": {
            **repro_meta("issue2224_followup_r2_compare"),
            "deciding_cells": list(CELLS_18),
            "n_contrasts": len(contrasts_out),
            "n_boot": int(args.n_boot),
            "seed_base_137_legs": int(args.seed_base),
            "seed42_scores_dir": str(args.seed42_scores_dir),
            "seed137_scores_dir": str(args.seed137_scores_dir),
            "panel_note": (
                "seed 137 reseeds BOTH training and the eval panel (the eval-questions "
                "rng is seeded by args.seed by design); contrasts pair within-seed; the "
                "cross-seed read is contrast-level (points + CIs), never item-paired"
            ),
        },
        "summary": {
            "n_contrasts": len(contrasts_out),
            # Contrasts with a computable direction on BOTH seeds (a None-status /
            # no-paired-items contrast is NOT evaluable — it must never read as
            # "non-agreeing"; review r2 non-blocking item).
            "n_evaluable": n_evaluable,
            "n_direction_agree": n_dir_agree,
            "n_seed137_point_in_seed42_ci_response_level": n_in_ci,
        },
        "contrasts": contrasts_out,
        "cells_seed137": cells_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, args.out)
    print(
        f"[compare] wrote {args.out} — {len(contrasts_out)} contrasts "
        f"({n_evaluable} evaluable), dir_agree={n_dir_agree}/{n_evaluable}, "
        f"in_seed42_ci={n_in_ci}/{n_evaluable}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
