#!/usr/bin/env python3
"""Task #536 registered follow-up (`478-estimator-mixedlm-refit`): refit the
PUBLISHED #478 co-primary estimator — statsmodels MixedLM, the exact spec from
``scripts/issue478_analyze.py::mixed_effects_K_x_logd`` on branch ``issue-478``
(lines 219-225) — on BOTH persona-distance joins (raw and mean-centered).

Published spec (verbatim from the #478 analysis code)::

    model = smf.mixedlm(
        "deltaLogP_mean ~ K * log_min_dist",
        df,                                  # rows filtered to min_dist > 0
        groups=df["subset_id"],
        vc_formula={"persona": "0 + C(held_out_persona)"},
    )
    result = model.fit(reml=False, method="lbfgs")

Published co-primary numbers to reproduce on the RAW join (manipulation
check): interaction beta = +0.010, p = 0.405. The raw-join refit must land in
the same regime BEFORE the centered read is interpretable.

Decision (verdict field, alpha = 0.05 on the single interaction term under
the published estimator):
  confirmed    — raw gate PASSED, centered fit converged, centered interaction
                 positive and p < 0.05 (the candidate rescue holds under the
                 published estimator)
  killed       — raw gate PASSED, centered fit converged, centered interaction
                 p >= 0.05 (rescue does not survive the published estimator)
  inconclusive — either fit singular / non-convergent, or the raw-join
                 manipulation check failed (reported, never papered over; the
                 published code's only "remedy" was an OLS FALLBACK — a
                 different estimator — which is not a valid substitute here)

Outputs (checkpointed the moment each is computed):
  eval_results/issue_536/mixedlm_refit_478.json   — per-join fits + verdict
  eval_results/issue_536/regrade_table.json       — `follow_up_result` added
      to the 478-flatness-null row IN PLACE (`regrade_label` untouched; the
      label belongs to the registered cluster-robust estimator)

Usage::

    uv run python scripts/issue536_mixedlm_refit.py \
        [--data-root /home/thomasjiralerspong/explore-persona-space]

CPU-only, seconds of runtime; no GPU, no pod, no downloads. Reuses the join
code from scripts/issue536_recompute_driver.py (family_111bank + the
regrade_478 row-level min_dist join, gated at 1e-4 vs the snapshot column).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue536_recompute_driver as drv  # noqa: E402  (join code of record)

# Published co-primary read from the #478 body (regrade_table estimator_caveat).
PUB_BETA = 0.010
PUB_P = 0.405
# Manipulation-check gate: same sign, |beta - published| <= this, and p in the
# same clearly-non-significant regime as the published 0.405.
GATE_BETA_TOL = 0.005
GATE_P_FLOOR = 0.10
ALPHA = 0.05  # single registered test of the centered interaction
H2_ALPHA = 0.01  # plan-H2 stricter alpha, recorded alongside (never binding here)

SPEC_STRING = (
    "statsmodels MixedLM: deltaLogP_mean ~ K * log(min_dist_to_K_subset); "
    "groups=subset_id (random intercept); "
    "vc_formula={'persona': '0 + C(held_out_persona)'}; "
    "fit(reml=False, method='lbfgs'); rows filtered to join-distance > 0 "
    "[verbatim from scripts/issue478_analyze.py::mixed_effects_K_x_logd, "
    "branch issue-478]"
)


def build_joined_df(data_root: Path):
    """Tidy snapshot + raw/centered min-dist joins, exactly as regrade_478 built them.

    Returns the 2,800-row DataFrame with ``md_raw`` / ``md_mc`` columns added,
    after the row-level 1e-4 gate of ``md_raw`` against the snapshot's own
    ``min_dist`` column (the same gate the audit driver enforced).
    """
    import pandas as pd

    fam = drv.family_111bank(data_root)
    idx = {n: i for i, n in enumerate(fam["names"])}
    dist_raw = 1.0 - fam["cos_raw"]
    dist_mc = 1.0 - fam["cos_mc"]
    df = pd.read_csv(drv.SNAP_DIR / "i478_tidy_69b34b94.csv")
    assert len(df) == 2800, f"#478 tidy rows = {len(df)}, expected 2800"

    def _min_dist(row, D):
        subs = row["positives"].split(";")
        return min(D[idx[row["held_out_persona"]], idx[s]] for s in subs)

    df["md_raw"] = df.apply(lambda r: _min_dist(r, dist_raw), axis=1)
    df["md_mc"] = df.apply(lambda r: _min_dist(r, dist_mc), axis=1)
    max_dev = float((df["md_raw"] - df["min_dist"]).abs().max())
    if max_dev > drv.GATE_MATRIX_TOL:
        raise RuntimeError(f"478 row-level min_dist join gate FAILED: max dev {max_dev:.3e}")
    return df, fam, max_dev


def fit_published_mixedlm(df, md_col: str) -> dict:
    """Fit the published MixedLM spec with ``md_col`` as the distance join.

    Reports coef/se/p for the K x log(dist) interaction plus honest
    convergence diagnostics (converged flag, boundary variance components,
    captured fit warnings). Never falls back to another estimator.
    """
    import statsmodels.formula.api as smf

    d = df[df[md_col] > 0].copy()
    d["log_md"] = np.log(d[md_col])
    out: dict = {"join_column": md_col, "n_obs": len(d), "spec_string": SPEC_STRING}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = smf.mixedlm(
                "deltaLogP_mean ~ K * log_md",
                d,
                groups=d["subset_id"],
                vc_formula={"persona": "0 + C(held_out_persona)"},
            )
            result = model.fit(reml=False, method="lbfgs")
        except Exception as e:
            out.update({"status": "FAILED", "reason": repr(e), "converged": False})
            return out
    # The published call passes groups= but omits re_formula; with vc_formula
    # present, statsmodels fits NO group random intercept (k_re == 0) — the
    # realized RE structure is the persona variance component nested within
    # subset_id groups + residual. Reported as-coded, never imputed.
    group_var = float(result.cov_re.iloc[0, 0]) if result.k_re > 0 else None
    vc_names = list(result.model.exog_vc.names)
    vcomp = {str(k): float(v) for k, v in zip(vc_names, result.vcomp, strict=True)}
    out.update(
        {
            "status": "OK",
            "converged": bool(result.converged),
            "coef": float(result.params["K:log_md"]),
            "se": float(result.bse["K:log_md"]),
            "p": float(result.pvalues["K:log_md"]),
            "fixed_effects": {k: float(v) for k, v in result.fe_params.items()},
            "random_effects_var": {"subset_id_intercept": group_var, "vcomp": vcomp},
            "re_structure_note": (
                "published call omits re_formula, so statsmodels fits no subset_id "
                "random intercept (k_re=0); realized RE = persona variance component "
                "nested within subset_id groups + residual (as-coded in #478)"
            ),
            "boundary_variance": bool(
                (group_var is not None and group_var < 1e-8)
                or any(v < 1e-8 for v in vcomp.values())
            ),
            "fit_warnings": sorted({f"{w.category.__name__}: {w.message}" for w in caught}),
        }
    )
    return out


def manipulation_check(raw: dict) -> dict:
    """Gate the raw-join refit against the published (+0.010, p=0.405) read."""
    if raw.get("status") != "OK":
        return {"passed": False, "reason": f"raw-join fit {raw.get('status')}", "criteria": None}
    same_sign = np.sign(raw["coef"]) == np.sign(PUB_BETA)
    beta_close = abs(raw["coef"] - PUB_BETA) <= GATE_BETA_TOL
    p_regime = raw["p"] >= GATE_P_FLOOR
    passed = bool(same_sign and beta_close and raw["converged"] and p_regime)
    return {
        "passed": passed,
        "published": {"beta": PUB_BETA, "p": PUB_P},
        "refit": {"beta": raw["coef"], "p": raw["p"], "converged": raw["converged"]},
        "criteria": {
            "same_sign": bool(same_sign),
            f"abs_beta_dev_le_{GATE_BETA_TOL}": bool(beta_close),
            "abs_beta_dev": float(abs(raw["coef"] - PUB_BETA)),
            f"p_in_nonsignificant_regime_ge_{GATE_P_FLOOR}": bool(p_regime),
            "converged": bool(raw["converged"]),
        },
    }


def decide_verdict(check: dict, mc: dict) -> tuple[str, str]:
    """confirmed | killed | inconclusive for the candidate rescue, plus the basis."""
    if not check["passed"]:
        return "inconclusive", (
            "raw-join manipulation check FAILED — the centered read is not "
            "interpretable under the published estimator"
        )
    if mc.get("status") != "OK" or not mc.get("converged", False):
        reason = mc.get("reason", "non-convergent fit")
        return "inconclusive", f"centered-join MixedLM did not yield a converged fit ({reason})"
    if mc["coef"] > 0 and mc["p"] < ALPHA:
        return "confirmed", (
            f"centered-join interaction +{mc['coef']:.4f}, p={mc['p']:.3g} < {ALPHA} under the "
            "published MixedLM — the candidate rescue holds under the published estimator"
        )
    return "killed", (
        f"centered-join interaction {mc['coef']:+.4f}, p={mc['p']:.3g} >= {ALPHA} (or wrong "
        "sign) under the published MixedLM — the rescue does not survive the published estimator"
    )


def update_regrade_table(out_dir: Path, refit_payload: dict) -> None:
    """Add ``follow_up_result`` to the 478-flatness-null row IN PLACE.

    Never touches ``regrade_label`` (it belongs to the registered estimator);
    preserves row order; atomic tmp-then-replace write like the driver's
    ``append_row``.
    """
    path = out_dir / "regrade_table.json"
    payload = json.loads(path.read_text())
    rows = [r for r in payload["rows"] if r.get("row_id") == "478-flatness-null"]
    assert len(rows) == 1, f"expected exactly one 478-flatness-null row, got {len(rows)}"
    rows[0]["follow_up_result"] = {
        "follow_up": "MixedLM refit (published co-primary) on raw AND centered joins",
        "artifact": "eval_results/issue_536/mixedlm_refit_478.json",
        "manipulation_check_passed": refit_payload["manipulation_check"]["passed"],
        "raw_join": {
            k: refit_payload["joins"]["raw"].get(k) for k in ("coef", "se", "p", "converged")
        },
        "centered_join": {
            k: refit_payload["joins"]["centered"].get(k) for k in ("coef", "se", "p", "converged")
        },
        "verdict": refit_payload["verdict"],
        "verdict_basis": refit_payload["verdict_basis"],
        "computed_at": refit_payload["generated_at"],
    }
    payload["updated_at"] = drv._now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    tmp.replace(path)
    print(f"[regrade_table] follow_up_result written to 478-flatness-null -> {path}")


def main() -> None:
    """Run both refits, write mixedlm_refit_478.json, update the regrade row."""
    import pandas as pd
    import statsmodels

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/thomasjiralerspong/explore-persona-space"),
        help="checkout holding eval_results/single_token_100_persona (the 111-bank)",
    )
    args = ap.parse_args()

    df, fam, join_dev = build_joined_df(args.data_root)
    fit_raw = fit_published_mixedlm(df, "md_raw")
    check = manipulation_check(fit_raw)
    fit_mc = fit_published_mixedlm(df, "md_mc")
    verdict, basis = decide_verdict(check, fit_mc)

    out_payload = {
        "schema_version": "i536_mixedlm_refit_v1",
        "generated_at": drv._now(),
        "git_commit": drv._git_sha(),
        "env": {
            "python": platform.python_version(),
            "statsmodels": statsmodels.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "concern_id": "478-estimator-mixedlm-refit",
        "spec": {
            "estimator": SPEC_STRING,
            "alpha": ALPHA,
            "h2_alpha_recorded": H2_ALPHA,
            "interaction_term": "K:log_md",
            "data": "eval_results/issue_536/inputs/i478_tidy_69b34b94.csv (2,800 rows)",
            "distance_bank": {
                "family": fam["family"],
                "n": fam["n"],
                "names_hash": fam["names_hash"],
                "layer": fam["layer"],
            },
            "join_gate_max_dev": join_dev,
        },
        "joins": {"raw": fit_raw, "centered": fit_mc},
        "manipulation_check": check,
        "centered_significant_at_h2_alpha": bool(
            fit_mc.get("status") == "OK" and fit_mc.get("p", 1.0) < H2_ALPHA
        ),
        "verdict": verdict,
        "verdict_basis": basis,
    }
    out_path = drv.OUT_DIR / "mixedlm_refit_478.json"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out_payload, indent=2, default=float))
    tmp.replace(out_path)
    print(f"[refit] wrote {out_path}")
    print(
        f"[refit] raw: beta={fit_raw.get('coef'):+.4f} p={fit_raw.get('p'):.4g} "
        f"converged={fit_raw.get('converged')} | manipulation check "
        f"{'PASS' if check['passed'] else 'FAIL'}"
    )
    print(
        f"[refit] centered: beta={fit_mc.get('coef'):+.4f} p={fit_mc.get('p'):.4g} "
        f"converged={fit_mc.get('converged')}"
    )
    print(f"[refit] VERDICT: {verdict} — {basis}")

    update_regrade_table(drv.OUT_DIR, out_payload)


if __name__ == "__main__":
    main()
