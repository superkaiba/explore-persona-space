"""Issue #2094 FA-3 — grid-wide separation-stratified pair-clustered bootstrap.

Recomputes the P7 ``phase_bootstrap`` pair-clustered bootstrap (B=10,000, same
seed / batched index-GEMM implementation / degenerate-self exclusion — reused
verbatim from ``scripts/issue2094_analysis.py``) RESTRICTED to well-separated
pairs, across ALL cell families, and writes
``eval_results/issue_2094/f_metrics/bootstrap_cis_wellsep.json`` (a NEW file;
``bootstrap_cis.json`` is never touched).

Restriction (analyzer round-1 convention, ``issue2094_sep_stratified_fig.py``):

* ``f_beh_<kind>`` families keep a pair iff its (pair, kind) anchor separation
  (the F_beh denominator, ceiling-minus-floor judge contrast) clears
  ``|separation| >= 0.5`` — the near-zero-separation pairs are the 30-200x
  leverage rows the recount removes;
* ``f_act`` families keep a pair iff it is well-separated on >= 1 rubric kind
  (separation never enters F_act's denominator; this is the matched-subset
  companion read, documented in the output's ``restriction`` block).

The output additionally carries per-family steered-vs-null CI reads
(``steered_vs_null``) + a ``summary`` block, so "does ANY cell separate from
its shuffled-donor null grid-wide (on well-separated pairs)?" is answerable
from the file alone. Coherence gating is inherited unchanged from the parent
f-tables (F_beh values are computed over coherent draws upstream);
degenerate-by-design self-transfer rows are excluded exactly as in
``phase_bootstrap``.

VM launch convention (shared-VM thread caps):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \\
    uv run python scripts/issue2094_wellsep_bootstrap.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_analysis as A  # noqa: E402

logger = logging.getLogger("issue2094_wellsep_bootstrap")

MIN_SEPARATION = 0.5  # analyzer round-1 convention (|ceiling - floor| floor)
MIN_PAIRS_HEADLINE = 5  # a 1-pair family's CI is degenerate; headline read floors at 5


def load_wellsep(anchors_path: Path, min_sep: float) -> tuple[set[tuple[str, str]], set[str]]:
    """(well-separated (pair_id, kind) set, pairs well-separated on >=1 kind)."""
    ws: set[tuple[str, str]] = set()
    all_pairs: set[str] = set()
    for a in A._iter_jsonl(anchors_path):
        all_pairs.add(a["pair_id"])
        if a.get("separation") is not None and abs(a["separation"]) >= min_sep:
            ws.add((a["pair_id"], a["kind"]))
    assert ws, f"no well-separated anchors at |sep| >= {min_sep} in {anchors_path}"
    return ws, {p for p, _ in ws}


def wellsep_keep(pid: str, metric: str, ws: set[tuple[str, str]], ws_any: set[str]) -> bool:
    """Per-metric pair-keep predicate (see module docstring)."""
    if metric == "f_act":
        return pid in ws_any
    return (pid, metric.removeprefix("f_beh_")) in ws


def compute_wellsep_families(
    rows: list[dict], ws: set[tuple[str, str]], ws_any: set[str], n_boot: int
) -> dict[str, dict]:
    """``phase_bootstrap``'s family loop with the well-sep keep predicate added.

    Identical conventions: pair axis = ALL bank pairs of the setting (missing /
    excluded pairs ride as NaN through the NaN-aware batched bootstrap), family
    keys ``arm|setting|slot|lv|dose|vec_type|metric``, seed ``BOOTSTRAP_SEED``.
    """
    pairs = A.BANK.build_pairs()
    pair_ids_by_setting = {
        s: sorted(p.pair_id for p in pairs if p.setting == s)
        for s in ("matched_prefix", "matched_query", "cross")
    }
    out: dict[str, dict] = {}
    t0 = time.monotonic()
    for setting, pids in pair_ids_by_setting.items():
        pid_idx = {p: i for i, p in enumerate(pids)}
        fam_values: dict[str, np.ndarray] = {}
        for row in rows:
            if row["setting"] != setting:
                continue
            metrics = ["f_act"] + [f"f_beh_{k}" for k in (row.get("f_beh") or {})]
            for metric in metrics:
                key = A._family_key(row, metric)
                arr = fam_values.setdefault(key, np.full(len(pids), np.nan))
                if wellsep_keep(row["pair_id"], metric, ws, ws_any):
                    arr[pid_idx[row["pair_id"]]] = A._cell_metric(row, metric)
        if not fam_values:
            continue
        keys = sorted(fam_values)
        values = np.stack([fam_values[k] for k in keys], axis=1)  # (n_pairs, n_fams)
        assert values.shape == (len(pids), len(keys)), values.shape
        boots = A.bootstrap_family_means_batched(values, n_boot, A.BOOTSTRAP_SEED)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # a fully-restricted family is legitimately all-NaN (kept for audit)
            warnings.simplefilter("ignore", RuntimeWarning)
            obs = np.nanmean(values, axis=0)
        for j, key in enumerate(keys):
            col = boots[:, j]
            valid = col[~np.isnan(col)]
            out[key] = {
                "setting": setting,
                "observed_mean": A._nan_to_none(obs[j]),
                "n_pairs_used": int((~np.isnan(values[:, j])).sum()),
                "ci_lo": float(np.percentile(valid, 2.5)) if valid.size else None,
                "ci_hi": float(np.percentile(valid, 97.5)) if valid.size else None,
                "n_valid_draws": int(valid.size),
            }
        logger.info(
            "[wellsep-bootstrap] setting=%s families=%d elapsed=%.1fs",
            setting,
            len(keys),
            time.monotonic() - t0,
        )
    return out


def steered_vs_null_reads(families: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    """Per-family steered-vs-null CI reads + per-metric-class summary counts."""
    reads: dict[str, dict] = {}
    per_metric: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "n_compared": 0,
            "n_steered_ci_excludes_null_mean": 0,
            "n_cis_disjoint": 0,
            f"n_compared_ge{MIN_PAIRS_HEADLINE}_pairs": 0,
            f"n_steered_ci_excludes_null_mean_ge{MIN_PAIRS_HEADLINE}_pairs": 0,
            f"n_cis_disjoint_ge{MIN_PAIRS_HEADLINE}_pairs": 0,
        }
    )
    separating: dict[str, list[str]] = defaultdict(list)
    for key, st in families.items():
        arm, *tail = key.split("|")
        if arm != "steered":
            continue
        nu = families.get("|".join(["null", *tail]))
        tail_key = "|".join(tail)
        metric = tail[-1]
        comparable = (
            nu is not None
            and st["ci_lo"] is not None
            and st["ci_hi"] is not None
            and nu["observed_mean"] is not None
            and st["n_pairs_used"] > 0
            and nu["n_pairs_used"] > 0
        )
        if not comparable:
            reads[tail_key] = {"comparable": False}
            continue
        excl = nu["observed_mean"] < st["ci_lo"] or nu["observed_mean"] > st["ci_hi"]
        disjoint = bool(
            nu["ci_lo"] is not None
            and nu["ci_hi"] is not None
            and (st["ci_lo"] > nu["ci_hi"] or st["ci_hi"] < nu["ci_lo"])
        )
        reads[tail_key] = {
            "comparable": True,
            "steered_mean": st["observed_mean"],
            "null_mean": nu["observed_mean"],
            "steered_ci": [st["ci_lo"], st["ci_hi"]],
            "null_ci": [nu["ci_lo"], nu["ci_hi"]],
            "n_pairs_used": st["n_pairs_used"],
            "steered_ci_excludes_null_mean": bool(excl),
            "cis_disjoint": disjoint,
            "direction": (
                "steered_above" if st["observed_mean"] > nu["observed_mean"] else "steered_below"
            ),
        }
        m = per_metric[metric]
        m["n_compared"] += 1
        m["n_steered_ci_excludes_null_mean"] += int(excl)
        m["n_cis_disjoint"] += int(disjoint)
        if st["n_pairs_used"] >= MIN_PAIRS_HEADLINE:
            m[f"n_compared_ge{MIN_PAIRS_HEADLINE}_pairs"] += 1
            m[f"n_steered_ci_excludes_null_mean_ge{MIN_PAIRS_HEADLINE}_pairs"] += int(excl)
            m[f"n_cis_disjoint_ge{MIN_PAIRS_HEADLINE}_pairs"] += int(disjoint)
            if excl:
                separating[metric].append(tail_key)
    summary = {
        "per_metric": dict(per_metric),
        f"separating_families_ge{MIN_PAIRS_HEADLINE}_pairs": {
            k: sorted(v) for k, v in separating.items()
        },
        "note": (
            "'separating' = steered 95 percent bootstrap CI excludes the matched "
            "shuffled-donor-null observed mean, on well-separated pairs only; a "
            "point-vs-interval read at ~alpha=0.05 per family, uncorrected for "
            f"multiplicity (expected ~5 percent by chance); families with < "
            f"{MIN_PAIRS_HEADLINE} well-separated pairs are counted separately "
            "(their CIs are degenerate at n=1)."
        ),
    }
    return reads, summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_2094"))
    ap.add_argument("--min-sep", type=float, default=MIN_SEPARATION)
    ap.add_argument("--n-boot", type=int, default=A.BOOTSTRAP_B)
    args = ap.parse_args(argv)

    fm = args.out_root / "f_metrics"
    rows = list(A._iter_jsonl(fm / "f_cells.jsonl")) + list(A._iter_jsonl(fm / "null_cells.jsonl"))
    assert rows, f"no f-table rows under {fm} - run issue2094_analysis --phase ftables first"
    rows, n_degenerate_excluded = A.bootstrap_eligible_rows(rows)
    assert rows, "every f-table row is degenerate-self - nothing to bootstrap"
    logger.info(
        "[wellsep-bootstrap] rows=%d (degenerate-self excluded: %d)",
        len(rows),
        n_degenerate_excluded,
    )
    ws, ws_any = load_wellsep(fm / "anchors.jsonl", args.min_sep)
    logger.info(
        "[wellsep-bootstrap] well-separated (pair,kind)=%d, pairs (any kind)=%d",
        len(ws),
        len(ws_any),
    )

    families = compute_wellsep_families(rows, ws, ws_any, args.n_boot)
    reads, summary = steered_vs_null_reads(families)

    out_path = fm / "bootstrap_cis_wellsep.json"
    A._write_json_atomic(
        out_path,
        {
            "B": args.n_boot,
            "seed": A.BOOTSTRAP_SEED,
            "resample_axis": "pairs (pair-clustered, within setting)",
            "degenerate_self_excluded": n_degenerate_excluded,
            "restriction": {
                "min_abs_separation": args.min_sep,
                "f_beh": "pair kept iff its (pair, rubric-kind) anchor |separation| >= floor",
                "f_act": (
                    "pair kept iff well-separated on >= 1 rubric kind (separation never "
                    "enters F_act's denominator; matched-subset companion read)"
                ),
                "n_wellsep_pair_kinds": len(ws),
                "n_wellsep_pairs_any_kind": len(ws_any),
            },
            "note": (
                "FA-3 separation-stratified recount of bootstrap_cis.json (same seed / "
                "batched implementation / degenerate-self exclusion / coherence gating as "
                "phase_bootstrap in scripts/issue2094_analysis.py); bootstrap_cis.json is "
                "the unrestricted parent and is unchanged"
            ),
            "families": families,
            "steered_vs_null": reads,
            "summary": summary,
            "repro": A._repro(),
        },
    )
    logger.info("[phase=wellsep_bootstrap_done] families=%d -> %s", len(families), out_path)
    for metric, m in sorted(summary["per_metric"].items()):
        logger.info(
            "[wellsep-bootstrap] %s: %d/%d steered CIs exclude null mean "
            "(>=%d well-sep pairs: %d/%d; disjoint CIs: %d)",
            metric,
            m["n_steered_ci_excludes_null_mean"],
            m["n_compared"],
            MIN_PAIRS_HEADLINE,
            m[f"n_steered_ci_excludes_null_mean_ge{MIN_PAIRS_HEADLINE}_pairs"],
            m[f"n_compared_ge{MIN_PAIRS_HEADLINE}_pairs"],
            m[f"n_cis_disjoint_ge{MIN_PAIRS_HEADLINE}_pairs"],
        )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
