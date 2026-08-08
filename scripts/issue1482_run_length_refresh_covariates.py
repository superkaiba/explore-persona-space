"""Refresh the two run-length covariates in the #1482 fullwidth substrate.

WHY SURGICAL, NOT A PRODUCER RE-RUN
    ``issue1482_continuous_predictors.py`` writes ``fullwidth_covariates.npz``
    inline, then goes on to run the whole bootstrap battery + figures. Several
    sibling agents drive that battery concurrently (the fullwidth label reads /
    Shapley arms), so re-running the producer to pick up a new run-length
    artifact would race them for the same outputs. This script touches ONLY the
    two arrays the run-length capture owns — ``mean_run_length`` and
    ``template_token_frac`` — joined on ``feat_ids``, and leaves every other
    column byte-identical.

WHY IT MATTERS (the complete-case story)
    The consumer's read mask is the INTERSECTION of finite values across EVERY
    covariate::

        ok = np.isfinite(r2)
        for k in keys: ok &= np.isfinite(cov[k])

    so a feature the run-length capture never saw is dropped from EVERY
    predictor's read, not just the two run-length slots. A 2,000-row capture
    left 27,855 features NaN on ``mean_run_length``; the full-corpus capture is
    what buys those back. This script reports the complete-case count before
    and after, plus per-decile counts, so the gain is measured rather than
    assumed.

Usage:
    uv run python scripts/issue1482_run_length_refresh_covariates.py            # report only
    uv run python scripts/issue1482_run_length_refresh_covariates.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
COVARIATES = REPO / "eval_results/issue_1482/predictor_battery/fullwidth_covariates.npz"
RUN_LENGTH = REPO / "eval_results/issue_1482/run_length/run_length_perfeature.npz"
REPORT = REPO / "eval_results/issue_1482/run_length/covariate_refresh.json"
SLOTS = ("mean_run_length", "template_token_frac")
N_DECILES = 10  # issue1482_continuous_predictors.N_DECILES


def _log(msg: str) -> None:
    print(f"[refresh] {msg}", flush=True)


def _decile_counts(pred: np.ndarray) -> dict:
    """Per-decile complete-case counts, using the consumer's own binning
    (``_decile_profile``: quantile edges + searchsorted on the interior)."""
    finite = pred[np.isfinite(pred)]
    if finite.size == 0:
        return {"decile_n": [0] * N_DECILES, "edges": []}
    edges = np.quantile(finite, np.linspace(0, 1, N_DECILES + 1))
    dec = np.searchsorted(edges[1:-1], finite, side="right")
    return {
        "decile_n": [int((dec == d).sum()) for d in range(N_DECILES)],
        "edges": [float(e) for e in edges],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="write; otherwise report only")
    ap.add_argument("--covariates", type=Path, default=COVARIATES)
    ap.add_argument("--run-length", type=Path, default=RUN_LENGTH)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()

    assert args.covariates.exists(), f"missing substrate: {args.covariates}"
    assert args.run_length.exists(), f"missing run-length artifact: {args.run_length}"

    with np.load(args.run_length) as z:
        rl_ids = np.asarray(z["feat_ids"], dtype=np.int64)
        new = {k: np.asarray(z[k], dtype=np.float64) for k in SLOTS if k in z.files}
    missing = [k for k in SLOTS if k not in new]
    assert not missing, f"run-length artifact is missing {missing}"

    with np.load(args.covariates) as z:
        cov = {k: z[k] for k in z.files}
    dict_size = int(cov["feat_ids"].shape[0])
    assert np.array_equal(cov["feat_ids"], np.arange(dict_size)), (
        "substrate feat_ids are not the identity index; join assumption broken"
    )

    # Complete-case mask over EVERY covariate, the consumer's own definition.
    keys = [k for k in cov if k != "feat_ids" and cov[k].dtype.kind == "f"]

    def complete_case(d: dict) -> np.ndarray:
        ok = np.ones(dict_size, dtype=bool)
        for k in keys:
            ok &= np.isfinite(d[k])
        return ok

    before_ok = complete_case(cov)
    before = {k: int(np.isfinite(cov[k]).sum()) for k in SLOTS}
    before_dec = {k: _decile_counts(cov[k]) for k in SLOTS}

    after_cov = dict(cov)
    for k in SLOTS:
        full = np.full(dict_size, np.nan)
        full[rl_ids] = new[k]
        after_cov[k] = full
    after_ok = complete_case(after_cov)
    after = {k: int(np.isfinite(after_cov[k]).sum()) for k in SLOTS}
    after_dec = {k: _decile_counts(after_cov[k]) for k in SLOTS}

    # Every OTHER column must be untouched.
    for k in keys:
        if k in SLOTS:
            continue
        assert np.array_equal(cov[k], after_cov[k], equal_nan=True), f"column {k} mutated"

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "substrate": str(args.covariates.relative_to(REPO)),
        "run_length_source": str(args.run_length.relative_to(REPO)),
        "dict_size": dict_size,
        "n_covariate_columns_in_mask": len(keys),
        "slot_finite_counts": {k: {"before": before[k], "after": after[k]} for k in SLOTS},
        "complete_case": {
            "before": int(before_ok.sum()),
            "after": int(after_ok.sum()),
            "delta": int(after_ok.sum()) - int(before_ok.sum()),
            "definition": (
                "features finite across ALL covariate columns simultaneously — the "
                "consumer's own `ok` mask (issue1482_continuous_predictors.main_reads). "
                "A feature the run-length capture never saw is dropped from EVERY "
                "predictor's read, not only the two run-length slots."
            ),
        },
        "per_decile_complete_case": {
            k: {
                "note": (
                    "deciles of the covariate's OWN finite values (consumer binning: "
                    "quantile edges + searchsorted on the interior); decile_n sums to "
                    "the finite count, and complete_case_n counts how many of each "
                    "decile also survive the all-column mask"
                ),
                "before_decile_n": before_dec[k]["decile_n"],
                "after_decile_n": after_dec[k]["decile_n"],
                "before_complete_case": _decile_complete_case(cov[k], before_ok),
                "after_complete_case": _decile_complete_case(after_cov[k], after_ok),
            }
            for k in SLOTS
        },
        "applied": bool(args.apply),
    }

    _log(
        f"complete-case: {report['complete_case']['before']} -> {report['complete_case']['after']}"
    )
    for k in SLOTS:
        _log(f"  {k}: finite {before[k]} -> {after[k]}")
        d = report["per_decile_complete_case"][k]["after_complete_case"]
        _log(f"    per-decile complete-case (after): {d['decile_n']}  n={d['n']}")
        if d["tie_degenerate"]:
            _log(
                f"    NOTE {k} is tie-degenerate: {d['n_duplicate_edges']} duplicate "
                f"quantile edges, {d['n_empty_deciles']} empty deciles (point mass)"
            )

    if args.apply:
        backup = args.covariates.with_suffix(".npz.pre_runlength_refresh")
        if not backup.exists():
            shutil.copy2(args.covariates, backup)
            _log(f"backup -> {backup.name}")
        # Write through a HANDLE: np.savez APPENDS .npz to a path argument that
        # lacks the suffix, so this dotted temp name would land at "<tmp>.npz"
        # and the os.replace below would raise FileNotFoundError (same class as
        # the capture driver's _atomic_savez).
        tmp = args.covariates.parent / f".{args.covariates.name}.tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            np.savez(fh, **after_cov)
        os.replace(tmp, args.covariates)
        _log(f"wrote {args.covariates.name} ({args.covariates.stat().st_size / 1e6:.1f} MB)")
    else:
        _log("report-only (pass --apply to write)")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    try:
        shown = args.report.relative_to(REPO)
    except ValueError:
        shown = args.report
    _log(f"report -> {shown}")


def _decile_complete_case(pred: np.ndarray, ok: np.ndarray) -> dict:
    """Per-decile counts over the COMPLETE-CASE subset, binned exactly as the
    consumer does (``_decile_profile`` bins ``cov[k][ok]``, not all finite
    values).

    Also flags TIE DEGENERACY: a covariate with a large point mass (e.g.
    ``template_token_frac`` is 0 for most features) produces repeated quantile
    edges, so several deciles are empty by construction and the mass piles into
    one bin. That is a property of the covariate, not a bug — but a decile
    profile over such a column is not interpretable as a gradient, so it is
    reported rather than silently returned as a row of zeros."""
    vals = pred[ok]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"decile_n": [0] * N_DECILES, "n": 0, "tie_degenerate": False}
    edges = np.quantile(vals, np.linspace(0, 1, N_DECILES + 1))
    dec = np.searchsorted(edges[1:-1], vals, side="right")
    counts = [int((dec == d).sum()) for d in range(N_DECILES)]
    n_dup = int((np.diff(edges) == 0).sum())
    return {
        "decile_n": counts,
        "n": int(vals.size),
        "tie_degenerate": bool(n_dup > 0),
        "n_duplicate_edges": n_dup,
        "n_empty_deciles": int(sum(c == 0 for c in counts)),
        "edges": [float(e) for e in edges],
    }


if __name__ == "__main__":
    main()
