"""Language-stratified mirror recomputation for #1946 (free-analysis follow-up).

Localizes the language=en sign flip in the cross-space taxonomy mirror: recomputes
the per-space difference pattern M_en = delta_prefix,en - delta_bare,en (delta_arm,en
= mean(nerr_arm | en) - mean(nerr_arm | rest), the #1738 taxonomy group-vs-rest
construction from issue1738_characterize._contrast_masks / issue1482_analysis) for the
SAE mean-pool space vs the dense space, decomposes the dense->SAE change in M_en into
its per-arm components, and bootstraps context-level CIs (batched, no per-draw loop —
the _boot_group_delta shape).

Inputs (all committed): eval_results/issue_1946/percontext_summary_L19_ridge_sae.csv,
eval_results/issue_1738/percontext_summary_L19_ridge.csv,
eval_results/issue_1738/bare_query/percontext_summary_L19_ridge.csv.
Output: eval_results/issue_1946/lang_stratified_mirror.json.

Convention note: rows with a missing (NaN) language label fall in the REST stratum,
matching the producer's `lab[i] and lab[i]["language"] == "en"` predicate.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + credentials BEFORE numpy/pandas import (#847)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SAE_CSV = REPO_ROOT / "eval_results/issue_1946/percontext_summary_L19_ridge_sae.csv"
DENSE_CSV = REPO_ROOT / "eval_results/issue_1738/percontext_summary_L19_ridge.csv"
BARE_CSV = REPO_ROOT / "eval_results/issue_1738/bare_query/percontext_summary_L19_ridge.csv"
OUT_JSON = REPO_ROOT / "eval_results/issue_1946/lang_stratified_mirror.json"

EXPECTED_N = 9941
N_BOOT = 10_000
SEED = 1946
BOOT_CHUNK = 1000


def _git_commit() -> str:
    """Return the worktree HEAD commit hash (reproducibility metadata)."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def load_joined() -> pd.DataFrame:
    """Join the three committed per-context CSVs on ci; assert row count + consistency."""
    sae = pd.read_csv(SAE_CSV)
    dense = pd.read_csv(DENSE_CSV)
    bare = pd.read_csv(BARE_CSV)
    for name, df in (("sae", sae), ("dense", dense), ("bare", bare)):
        assert df.ci.is_unique, f"{name}: ci not unique"
        assert len(df) == EXPECTED_N, f"{name}: {len(df)} rows != {EXPECTED_N}"

    j = (
        sae[["ci", "nerr_sae_prefix", "nerr_sae_bare", "nerr_sae_context", "language"]]
        .merge(
            dense[["ci", "nerr_prefix_L19_ridge", "nerr_context_L19_ridge", "language"]].rename(
                columns={
                    "nerr_prefix_L19_ridge": "nerr_dense_prefix",
                    "nerr_context_L19_ridge": "nerr_dense_context",
                    "language": "language_dense",
                }
            ),
            on="ci",
            how="inner",
        )
        .merge(
            bare[["ci", "nerr_bare_L19_ridge", "nerr_prefix_L19_ridge"]].rename(
                columns={
                    "nerr_bare_L19_ridge": "nerr_dense_bare",
                    "nerr_prefix_L19_ridge": "nerr_dense_prefix_bq",
                }
            ),
            on="ci",
            how="inner",
        )
    )
    assert len(j) == EXPECTED_N, f"join produced {len(j)} rows != {EXPECTED_N}"
    # Label agreement (NaN-safe) + duplicated dense prefix column consistency across files.
    assert (j.language.fillna("<NA>") == j.language_dense.fillna("<NA>")).all(), (
        "language labels disagree between #1946 SAE CSV and #1738 dense CSV"
    )
    assert np.allclose(j.nerr_dense_prefix, j.nerr_dense_prefix_bq), (
        "dense prefix column differs between #1738 main and bare_query CSVs"
    )
    return j.drop(columns=["language_dense", "nerr_dense_prefix_bq"])


def stratum_stats(j: pd.DataFrame, en: np.ndarray) -> dict:
    """Per stratum x space: arm means, mean prefix-bare difference, and en-vs-rest deltas."""
    arms = {
        "sae": {
            "prefix": j.nerr_sae_prefix.to_numpy(),
            "bare": j.nerr_sae_bare.to_numpy(),
            "context": j.nerr_sae_context.to_numpy(),
        },
        "dense": {
            "prefix": j.nerr_dense_prefix.to_numpy(),
            "bare": j.nerr_dense_bare.to_numpy(),
            "context": j.nerr_dense_context.to_numpy(),
        },
    }
    out: dict = {}
    for space, cols in arms.items():
        d_pb = cols["prefix"] - cols["bare"]
        out[space] = {
            "en": {
                "n": int(en.sum()),
                "mean_nerr": {a: float(v[en].mean()) for a, v in cols.items()},
                "mean_prefix_minus_bare": float(d_pb[en].mean()),
            },
            "rest": {
                "n": int((~en).sum()),
                "mean_nerr": {a: float(v[~en].mean()) for a, v in cols.items()},
                "mean_prefix_minus_bare": float(d_pb[~en].mean()),
            },
            "delta_en_vs_rest": {  # mean(group) - mean(rest), the taxonomy construction
                a: float(v[en].mean() - v[~en].mean()) for a, v in cols.items()
            },
        }
        out[space]["M_en"] = float(
            out[space]["delta_en_vs_rest"]["prefix"] - out[space]["delta_en_vs_rest"]["bare"]
        )
    return out


def boot_m_en(d_sae: np.ndarray, d_dense: np.ndarray, en: np.ndarray) -> dict:
    """Context-level bootstrap of M_en per space + the paired SAE-dense difference.

    M_en = mean(d | en) - mean(d | rest) with d = nerr_prefix - nerr_bare (equal by
    linearity to delta_prefix,en - delta_bare,en). Shared resample indices across the
    two spaces (same contexts) => the difference draws are paired. Batched chunked
    gathers, mirroring issue1482_analysis._boot_group_delta — no per-draw Python loop.
    """
    rng = np.random.default_rng(SEED)
    n = len(d_sae)
    m_sae = np.empty(N_BOOT)
    m_dense = np.empty(N_BOOT)
    for s in range(0, N_BOOT, BOOT_CHUNK):
        b = min(BOOT_CHUNK, N_BOOT - s)
        take = rng.integers(0, n, size=(b, n))
        mk = en[take]
        n_en = mk.sum(1)
        n_rest = n - n_en
        assert (n_en > 0).all() and (n_rest > 0).all(), "degenerate bootstrap stratum"
        for arr, dest in ((d_sae, m_sae), (d_dense, m_dense)):
            vals = arr[take]
            s_en = (vals * mk).sum(1)
            dest[s : s + b] = s_en / n_en - (vals.sum(1) - s_en) / n_rest
    diff = m_sae - m_dense

    def _ci(x: np.ndarray) -> dict:
        return {
            "mean": float(x.mean()),
            "ci95": [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))],
            "frac_draws_below_zero": float((x < 0).mean()),
            "frac_draws_above_zero": float((x > 0).mean()),
        }

    return {
        "n_boot": N_BOOT,
        "seed": SEED,
        "resampling": "context-level (rows resampled with replacement; shared indices "
        "across spaces => paired difference)",
        "M_en_sae": _ci(m_sae),
        "M_en_dense": _ci(m_dense),
        "M_en_sae_minus_dense": _ci(diff),
    }


def main() -> None:
    j = load_joined()
    en = (j.language == "en").to_numpy()  # NaN -> False -> rest (producer convention)
    n_lang_nan = int(j.language.isna().sum())

    stats = stratum_stats(j, en)
    d_sae = (j.nerr_sae_prefix - j.nerr_sae_bare).to_numpy()
    d_dense = (j.nerr_dense_prefix - j.nerr_dense_bare).to_numpy()
    boot = boot_m_en(d_sae, d_dense, en)

    # Decomposition of the dense->SAE change in M_en into per-arm components.
    dp_sae = stats["sae"]["delta_en_vs_rest"]["prefix"]
    db_sae = stats["sae"]["delta_en_vs_rest"]["bare"]
    dp_dense = stats["dense"]["delta_en_vs_rest"]["prefix"]
    db_dense = stats["dense"]["delta_en_vs_rest"]["bare"]
    prefix_component = dp_sae - dp_dense
    bare_component = db_sae - db_dense
    total_change = stats["sae"]["M_en"] - stats["dense"]["M_en"]
    # M_en change = prefix_component - bare_component (bare enters with a minus sign).
    assert abs(total_change - (prefix_component - bare_component)) < 1e-12
    bare_share = abs(bare_component) / (abs(bare_component) + abs(prefix_component))
    if bare_share >= 0.75:
        localization = "yes"
    elif bare_share <= 0.25:
        localization = "no"
    else:
        localization = "mixed"

    sae_ci = boot["M_en_sae"]["ci95"]
    diff_ci = boot["M_en_sae_minus_dense"]["ci95"]
    sae_nonzero = not (sae_ci[0] <= 0.0 <= sae_ci[1])
    diff_nonzero = not (diff_ci[0] <= 0.0 <= diff_ci[1])
    verdict = (
        f"flip localized to bare-arm English magnitude: {localization} "
        f"(bare component {bare_component:+.4f} = {bare_share:.0%} of the dense->SAE "
        f"M_en change vs prefix component {prefix_component:+.4f}; SAE M_en "
        f"{stats['sae']['M_en']:+.4f} is {'' if sae_nonzero else 'NOT '}distinguishable "
        f"from zero and {'' if diff_nonzero else 'NOT '}distinguishable from the dense "
        f"M_en {stats['dense']['M_en']:+.4f} at 95% context-level bootstrap CIs)"
    )

    out = {
        "task": 1946,
        "analysis": "language-stratified mirror recomputation (free-analysis follow-up)",
        "inputs": {
            "sae_csv": str(SAE_CSV.relative_to(REPO_ROOT)),
            "dense_csv": str(DENSE_CSV.relative_to(REPO_ROOT)),
            "bare_csv": str(BARE_CSV.relative_to(REPO_ROOT)),
        },
        "n_joined": int(len(j)),
        "n_en": int(en.sum()),
        "n_non_en": int((~en).sum()),
        "n_language_nan_in_rest": n_lang_nan,
        "stratum_convention": "en = (language == 'en'); NaN-language rows fall in rest, "
        "matching issue1738_characterize._contrast_masks",
        "per_space": stats,
        "decomposition": {
            "delta_prefix_en": {"dense": dp_dense, "sae": dp_sae, "change": prefix_component},
            "delta_bare_en": {"dense": db_dense, "sae": db_sae, "change": bare_component},
            "M_en": {
                "dense": stats["dense"]["M_en"],
                "sae": stats["sae"]["M_en"],
                "change": total_change,
            },
            "bare_share_of_change": float(bare_share),
            "localization_rule": "yes if |bare component| >= 75% of summed |components|, "
            "no if <= 25%, else mixed",
        },
        "bootstrap": boot,
        "verdict": verdict,
        "metadata": {
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
            "script": "scripts/issue1946_lang_stratified.py",
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[lang-stratified] wrote {OUT_JSON}")
    print(f"[lang-stratified] {verdict}")


if __name__ == "__main__":
    main()
