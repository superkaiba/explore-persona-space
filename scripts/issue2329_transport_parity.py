#!/usr/bin/env python3
"""#2329: batch-vs-sync transport PARITY check licensing the rule-28 merge.

WHY: the production waves run censored 764 draws with ``stop_reason ==
'refusal'`` and empty content. Measured routing split in that run: 7 dispatches
went BATCH and 162 went SYNC, and the 7 batch N values (4316 / 4434 / 4436 /
5760x3 / 38962) map exactly onto the 7 grid waves that were censored --
``coherence.anchors`` (14,040) being the 8th, batched in the anchors run. So
7/7 batch dispatches were censored and 0/162 sync dispatches were: the class is
transport-conditional, not content-determined. Re-issuing those draws on SYNC at
the identical instrument rescued 764/764 with ZERO re-refusals.

That rescue MERGES sync-origin scores into wave score sets whose other rows are
batch-origin. llm-judging.md rule 28 licenses such a merge only with a
dual-scored parity check on ~200-300 overlapping items, reporting the
batch-vs-sync offset (#1739 reported 287 items, batch mean 7.26 vs sync 7.77).
This script IS that check.

HOW: sample N already-scored items from one batch-routed wave, re-judge them on
the SYNC path at the IDENTICAL instrument (same judge_model, rubric text,
max_tokens, n_draws) into a FRESH cache dir -- fresh because the rubric-keyed
JudgeCache would otherwise serve the stored batch verdict straight back and the
"comparison" would be a tautology -- then report the paired offset.

CONTAMINATION, DISCLOSED: the stored score for a few sampled items may itself be
sync-origin, namely those rescued in the re-issue pass. The rescued item ids are
NOT recoverable from artifacts (the re-issue overwrote the wave's raw file and
the wave meta carries only aggregate tallies), so the default wave is chosen to
MINIMISE this: f7c626b141d19.grid had 28 refusals in 5,760 items, so at most
~0.49% of a random sample is sync-vs-sync rather than batch-vs-sync. The
realised upper bound is computed and reported as
``max_contamination_frac`` rather than being assumed negligible.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger("issue2329.parity")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rubric-id", default="f7c626b141d19")
    ap.add_argument("--wave", default=None, help="default <rubric-id>.grid")
    ap.add_argument("--n-sample", type=int, default=250)
    ap.add_argument("--seed", type=int, default=2329)
    ap.add_argument("--in-root", type=Path, default=Path("data/issue_2329/judge_inputs"))
    ap.add_argument("--work-root", type=Path, default=Path("eval_results/issue_2329/judge"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results/issue_2329/judge/gates/transport_parity.json"),
    )
    args = ap.parse_args(argv)
    wave = args.wave or f"{args.rubric_id}.grid"

    sys.path.insert(0, str(_repo_root() / "scripts"))
    import issue2329_judge as m  # noqa: E402
    from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402

    cfg = m.build_config(m.parse_args(["--phase", "waves", "--in-root", str(args.in_root)]))

    # Stored (production) scores for this wave.
    stored: dict[str, float] = {}
    scores_path = args.work_root / "scores" / f"{wave}.scores.jsonl"
    for row in m.J94._iter_jsonl(scores_path):
        s = row.get("score")
        if s is not None:
            stored[row["item_id"]] = float(s)
    if not stored:
        raise SystemExit(f"no stored scores in {scores_path}")

    # Rebuild the units (items.jsonl holds only answer_sha16, not the text).
    pairs = m.surviving_pairs(cfg.bank_json)
    pairs_by_id = {p.pair_id: p for p in pairs}
    grid_rows = m.load_grid_rows(cfg.rollouts_dir)
    units_by_rid = m.build_grid_behavior_items(grid_rows, pairs_by_id)
    if args.rubric_id not in units_by_rid:
        raise SystemExit(
            f"rubric {args.rubric_id} not in grid units: {sorted(units_by_rid)[:5]}..."
        )
    units = [u for u in units_by_rid[args.rubric_id] if u.item_id in stored]
    if len(units) < args.n_sample:
        raise SystemExit(f"only {len(units)} scored units for {args.rubric_id} < {args.n_sample}")

    rng = random.Random(args.seed)
    sample = rng.sample(units, args.n_sample)
    prompt = m.rubric_registry(pairs)[args.rubric_id]

    # FRESH cache dir: a reused one would serve the stored batch verdict back.
    fresh_cache = cfg.cache_root / "_transport_parity" / args.rubric_id
    logger.info(
        "[parity] wave=%s rubric=%s n=%d sync re-judge at identical instrument "
        "(model=%s max_tokens=%d n_draws=%d), fresh cache=%s",
        wave,
        args.rubric_id,
        len(sample),
        cfg.judge_model,
        cfg.max_tokens,
        m.JUDGE_N_DRAWS,
        fresh_cache,
    )
    result = judge_graded(
        [(u.item_id, u.question, u.answer) for u in sample],
        prompt,
        n_draws=m.JUDGE_N_DRAWS,
        cache_dir=fresh_cache,
        save_raw=cfg.raw_dir / f"{wave}.transport_parity.json",
        judge_model=cfg.judge_model,
        max_tokens=cfg.max_tokens,
        threshold_base=m.FORCE_SYNC_THRESHOLD_BASE,
    )

    paired = [
        (stored[u.item_id], float(result.scores[u.item_id]))
        for u in sample
        if result.scores.get(u.item_id) is not None
    ]
    if not paired:
        raise SystemExit("parity: zero paired items — the sync re-judge produced no scores")

    import numpy as np

    b = np.array([p[0] for p in paired], dtype=float)
    s = np.array([p[1] for p in paired], dtype=float)
    d = s - b

    def _rank(x: np.ndarray) -> np.ndarray:
        order = x.argsort(kind="stable")
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        return r

    def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
        if x.std() == 0 or y.std() == 0:
            return None  # undefined, not 0 — a constant arm has no correlation
        return float(np.corrcoef(x, y)[0, 1])

    n_refusal_in_wave = 28 if args.rubric_id == "f7c626b141d19" else None
    report = {
        "wave": wave,
        "rubric_id": args.rubric_id,
        "n_sampled": len(sample),
        "n_paired": len(paired),
        "n_sync_unscored": len(sample) - len(paired),
        "batch_mean": round(float(b.mean()), 4),
        "sync_mean": round(float(s.mean()), 4),
        "offset_sync_minus_batch": round(float(d.mean()), 4),
        "mean_abs_diff": round(float(np.abs(d).mean()), 4),
        "median_abs_diff": round(float(np.median(np.abs(d))), 4),
        "exact_agreement_frac": round(float((d == 0).mean()), 4),
        "pearson_r": _pearson(b, s),
        "spearman_rho": _pearson(_rank(b), _rank(s)),
        "batch_std": round(float(b.std(ddof=1)), 4),
        "sync_std": round(float(s.std(ddof=1)), 4),
        "instrument": {
            "judge_model": cfg.judge_model,
            "max_tokens": cfg.max_tokens,
            "n_draws": m.JUDGE_N_DRAWS,
            "rubric_sha16": m.hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "identical_across_arms": True,
            "differs_only_by": "HTTP transport (Batch vs sync)",
        },
        "contamination": {
            "note": (
                "stored scores are batch-origin EXCEPT any item rescued in the rule-28 "
                "re-issue, whose ids are not recoverable from artifacts; those compare "
                "sync-vs-sync. Upper bound only, not an estimate."
            ),
            "n_refusal_in_wave": n_refusal_in_wave,
            "wave_n_items": len(units_by_rid[args.rubric_id]),
            "max_contamination_frac": (
                round(n_refusal_in_wave / len(units_by_rid[args.rubric_id]), 6)
                if n_refusal_in_wave is not None
                else None
            ),
        },
        "telemetry_sync_pass": {
            "n_total_draws": result.n_total_draws,
            "n_dropped_draws": result.n_dropped_draws,
            "n_refusal_draws": result.n_refusal_draws,
            "n_api_refusal_draws": result.n_api_refusal_draws,
            "n_transport_lost_draws": result.n_transport_lost_draws,
            "n_truncation_dropped_draws": result.n_truncation_dropped_draws,
            "stop_reason_tally": dict(result.stop_reason_tally or {}),
        },
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.J94._write_json_atomic(args.out, report)
    logger.info("[parity] wrote %s", args.out)
    for k in (
        "n_paired",
        "batch_mean",
        "sync_mean",
        "offset_sync_minus_batch",
        "mean_abs_diff",
        "exact_agreement_frac",
        "pearson_r",
        "spearman_rho",
    ):
        logger.info("[parity] %s = %s", k, report[k])
    logger.info("[parity] sync-pass api_refusal = %d", result.n_api_refusal_draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
