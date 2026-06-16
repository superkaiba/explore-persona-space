"""Issue #537 follow-up `predictor-bakeoff-complete` -- Deliverable 2: the
predictor x (behavior x context-family) win/skill matrix with a per-cell noise
null + permutation p (plan v9 §4.4 / §6.5).

For each behavior and each of the 7 context families, computes every predictor's
held-out skill STRATIFIED to that family's cells (the OOF prediction error
restricted to cells touching the family) and records the best predictor + skill
per (behavior, family). EACH cell additionally carries:

  family_shuffle_null_skill -- the SAME stratified-skill computation run on
      null_random_predictor's D (the matched argmax-over-noise floor).
  permutation_p -- a B=2000 (seed 537) family-label-shuffle permutation p for the
      cell's winner: permute which contexts belong to which family, recompute the
      winner's stratified skill, p = fraction of shuffles with skill >= observed.
      A cell with p >= 0.05 is "no significant family-specific winner" (H3 clause).

Zero GPU: reads the persisted per-metric scores file (the same grid the leaderboard
sorts) and recomputes the stratified OOF skill from the predictor + G matrices.

  -> predictor-bakeoff-complete/win_matrix.json
     {behavior: {family: {best_predictor, skill, runner_up,
                          family_shuffle_null_skill, permutation_p}}} + the full
     per-(behavior, family, predictor) skill grid.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i537_win_matrix")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
EVAL = Path(os.environ.get("I537_EVAL_ROOT", str(REPO / "eval_results/issue_537")))
PBC = EVAL / "predictor-bakeoff-complete"
SEED = 537
ALL_BEHAVIORS = ("marker", "fact", "refusal", "sycophancy", "em")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env={**os.environ},
    ).stdout.strip()


def _stratified_skill(d_mat, g_mat, in_fam: set[int]) -> float:
    """Family-restricted held-out R²: fit OLS on cells NOT touching the family,
    predict the cells touching it, pooled R² over the held-out cells."""
    from i537_score_metric import _r2_from_pooled

    n = d_mat.shape[0]
    keep = [k for k in range(n) if k not in in_fam]
    xs, ys = [], []
    for i in keep:
        for j in keep:
            if i != j and np.isfinite(d_mat[i, j]) and np.isfinite(g_mat[i, j]):
                xs.append(d_mat[i, j])
                ys.append(g_mat[i, j])
    if len(xs) < 8:
        return float("nan")
    coef = np.polyfit(np.array(xs), np.array(ys), deg=1)
    yt, yp = [], []
    for i in range(n):
        for j in range(n):
            if i == j or (i not in in_fam and j not in in_fam):
                continue
            if np.isfinite(d_mat[i, j]) and np.isfinite(g_mat[i, j]):
                yt.append(g_mat[i, j])
                yp.append(float(np.polyval(coef, d_mat[i, j])))
    if not yt:
        return float("nan")
    return _r2_from_pooled(np.array(yt), np.array(yp))


def build(behaviors: list[str], *, b_perm: int = 2000) -> dict:
    from i537_score_metric import METRIC_REGISTRY, _load_g, metric_matrix, quarantine_mask

    from explore_persona_space.experiments.i537_contexts import load_registry, train_cids_for

    registry = load_registry(require_sampled=True)
    out: dict = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "permutation_B": b_perm,
        "seed": SEED,
        "win_matrix": {},
        "skill_grid": {},
    }
    rng = np.random.default_rng(SEED)

    # the scored predictor rows (skip controls for the WINNER, but keep the null
    # row for the floor); variant-tagged keys excluded.
    pred_ids = [
        m
        for m, s in METRIC_REGISTRY.items()
        if s["tier"] == "registered" and s["implemented"] and s["family"] != "control"
    ]

    for behavior in behaviors:
        cids = train_cids_for(behavior)
        fam_of = [registry[c].family for c in cids]
        families = sorted(set(fam_of))
        g_mat = _load_g(behavior, cids, cids)
        qmask = quarantine_mask(
            behavior, cids, cids, final_test=False, invocation_note="win_matrix"
        )
        g_mat = np.where(qmask, g_mat, np.nan)

        # predictor D matrices (best-effort: skip rows whose artifacts are absent)
        d_by_pred: dict[str, np.ndarray] = {}
        for mid in pred_ids:
            try:
                d_by_pred[mid] = metric_matrix(mid, cids, behavior=behavior)
            except (AssertionError, FileNotFoundError, KeyError) as e:
                logger.warning(
                    "[win] %s/%s D unavailable (%s) -- skip", behavior, mid, type(e).__name__
                )
        null_d = metric_matrix("null_random_predictor", cids, behavior=behavior)

        out["win_matrix"][behavior] = {}
        out["skill_grid"][behavior] = {}
        for fam in families:
            in_fam = {k for k in range(len(cids)) if fam_of[k] == fam}
            skills = {mid: _stratified_skill(d, g_mat, in_fam) for mid, d in d_by_pred.items()}
            finite = {m: s for m, s in skills.items() if np.isfinite(s)}
            out["skill_grid"][behavior][fam] = skills
            if not finite:
                out["win_matrix"][behavior][fam] = {"best_predictor": None, "skill": None}
                continue
            ranked = sorted(finite.items(), key=lambda kv: kv[1], reverse=True)
            best, best_skill = ranked[0]
            runner = ranked[1][0] if len(ranked) > 1 else None
            null_skill = _stratified_skill(null_d, g_mat, in_fam)
            # permutation p: shuffle family labels, recompute the winner's skill
            ge = 0
            for _ in range(b_perm):
                perm = rng.permutation(fam_of)
                in_perm = {k for k in range(len(cids)) if perm[k] == fam}
                s = _stratified_skill(d_by_pred[best], g_mat, in_perm)
                if np.isfinite(s) and s >= best_skill:
                    ge += 1
            perm_p = (ge + 1) / (b_perm + 1)
            out["win_matrix"][behavior][fam] = {
                "best_predictor": best,
                "skill": float(best_skill),
                "runner_up": runner,
                "family_shuffle_null_skill": float(null_skill) if np.isfinite(null_skill) else None,
                "permutation_p": float(perm_p),
                "significant": bool(perm_p < 0.05),
            }
            logger.info(
                "[win] %s/%s: best=%s skill=%.3f null=%.3f p=%.3f",
                behavior,
                fam,
                best,
                best_skill,
                null_skill if np.isfinite(null_skill) else float("nan"),
                perm_p,
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behaviors", default="marker,fact,refusal,sycophancy,em")
    ap.add_argument("--b-perm", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=PBC / "win_matrix.json")
    ap.add_argument("--smoke", action="store_true", help="1 behavior, B=20 permutations")
    args = ap.parse_args()
    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    assert all(b in ALL_BEHAVIORS for b in behaviors), behaviors
    b_perm = 20 if args.smoke else args.b_perm
    if args.smoke:
        behaviors = behaviors[:1]
    payload = build(behaviors, b_perm=b_perm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    logger.info("[win] wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
