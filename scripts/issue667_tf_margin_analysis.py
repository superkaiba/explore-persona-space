#!/usr/bin/env python3
# math/scientific notation in docstrings + messages
"""Issue #667 — 0-GPU CPU analysis of the 2-index tf-margin gate->behavior bridge.

Joins #667's recomputed per-cell base whitened gate ``g0(C')`` against the NEW
``tf_margin_leak`` behavioral DV (extracted by issue667_tf_margin_extract) and
computes, per behavior {em, sycophancy, fact} at layer 14, n=464 off-diagonal
cells (plan v6 §4.2/§4.3):

- **Headline:** Spearman(g0(C'), tf_margin_leak) + clustered-bootstrap CI
  (B=N_BOOT=2000, 7 families) + shuffled null (permute tf_margin_leak within
  behavior) + the cell-for-cell delta vs #667's committed base-``G`` rho
  (0.13/0.16/0.40). De-censoring SUPPORTED iff the tf-margin rho is materially
  above the base-``G`` rho with CI excluding zero AND the validation gate passed.
- **g0-recompute correctness gate:** the recomputed aggregate Spearman(g0, G)
  reproduces #667's committed base-``G`` rho within +/-0.02 (guards the
  SMUGGLED-VARIABLE / base-side mode). HALT (rc!=0) on mismatch.
- **Measurement-validity gate (per behavior):** Spearman(tf_margin_leak, G)
  point est > 0 AND 95% clustered-bootstrap CI excludes zero, BEFORE the
  headline is carried (the #722 re-validation on the 2-index grid). A failing
  behavior's headline is reported as "not a usable de-censoring companion", NOT
  carried — a REPORTABLE outcome, not a run halt.

# Vendored from scripts/issue722_tf_margin_extract.py + scripts/issue722_tf_margin_analysis.py
# (branch issue-722-tf-margin, commits 2f824110/27d1106661) — byte-identical function bodies
# for _spearman + clustered_bootstrap_spearman (N_BOOT=2000, the #722 default). The join
# (run_gate_vs_tf_margin) is NEW (2-index; NOT #722's v_A-ridge-readout chain).

Off-pod CPU (plan §9): all linear algebra over the reused #667 store + #658
sigma_c + the freshly-extracted tf-margins. Reproducibility metadata embedded
in every output JSON (CLAUDE.md).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("issue667.tf_margin_analysis")

# Match #722's clustered_bootstrap_spearman default verbatim (plan v6 nit-2:
# N_BOOT=2000, NOT 1000). Tighter CI than B=1000, still cheap on 464 cells.
N_BOOT = 2000

# The g0-recompute correctness gate tolerance (plan v6 §5): the recomputed
# aggregate Spearman(g0, G) must reproduce #667's committed base-G rho within
# this, else HALT (smuggled-variable / base-side mode).
G0_CORRECTNESS_TOL = 0.02

# #667's committed base-G rho per behavior (0.13 em / 0.16 syco / 0.40 fact) —
# the cell-for-cell comparison denominator. The g0-recompute correctness gate
# derives it fresh from the store; this is the pinned reference it must match.
COMMITTED_BASE_G_RHO = {"em": 0.13, "sycophancy": 0.16, "fact": 0.40}

# CONCERN 1 (round 2): the fact-pool builder writes this sentinel when floor-N <
# YIELD_FLOOR_MIN. When run standalone off-pod the analysis reads it directly (in
# addition to the dispatcher-forwarded --fact-dropped flag) so the SOFT-DROP path
# is honored regardless of how the analysis is invoked. Lockstep with
# issue667_build_fact_pool.py DROP_SENTINEL_NAME + the dispatcher FACT_DROP_SENTINEL.
FACT_DROP_SENTINEL = "data/issue_667/fact_fixed_pool_v1/DROPPED_FROM_HEADLINE.sentinel"


def _fact_dropped_sentinel_present() -> bool:
    return (PROJECT_ROOT / FACT_DROP_SENTINEL).is_file()


# ─────────────────────────────────────────────────────────────────────────────
# VENDORED byte-identical from scripts/issue722_tf_margin_analysis.py
# (branch issue-722-tf-margin) — function bodies unchanged.
# ─────────────────────────────────────────────────────────────────────────────


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    r, _ = spearmanr(a, b)
    return None if np.isnan(r) else float(r)


def clustered_bootstrap_spearman(x, y, families, n_boot=N_BOOT, alpha=0.05, seed=0):
    """Family-clustered percentile CI on Spearman(x,y) (resample whole families)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    fams = np.asarray(families, dtype=object)
    point = _spearman(x, y)
    uniq = sorted({str(f) for f in fams})
    if point is None or len(uniq) < 2:
        return {"point": point, "ci_lo": point, "ci_hi": point, "n_families": len(uniq)}
    fam_to_idx = {f: np.where(fams.astype(str) == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([fam_to_idx[f] for f in chosen])
        r = _spearman(x[idx], y[idx])
        if r is not None:
            vals.append(r)
    vals = np.array(vals)
    return {
        "point": float(point),
        "ci_lo": float(np.percentile(vals, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
        "n_families": len(uniq),
        "n_boot_kept": int(vals.size),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NEW — the 2-index gate->tf-margin join + gates (NOT #722's v_A-ridge chain)
# ─────────────────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, env={**os.environ}
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _repro_meta(extra: dict | None = None) -> dict:
    meta = {
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "script": "issue667_tf_margin_analysis",
        "n_boot": N_BOOT,
    }
    if extra:
        meta.update(extra)
    return meta


def load_tf_margin_leak(per_cell_dir: Path, behavior: str) -> dict[tuple[str, str], dict]:
    """{(source, target): cell_dict} for one behavior from the tf-margin per-cell store.

    Reads ``<per_cell_dir>/<behavior>/<source>_seed*/tf_margins.json`` (one file
    per source cell, each carrying all 30 targets). Off-diagonal cells only
    (target != source); the diagonal source cell is dropped (matches the #667 g0
    join, which drops source==target).
    """
    out: dict[tuple[str, str], dict] = {}
    beh_dir = per_cell_dir / behavior
    if not beh_dir.exists():
        return out
    for cell_dir in sorted(beh_dir.glob("*_seed*")):
        source = cell_dir.name.rsplit("_seed", 1)[0]
        f = cell_dir / "tf_margins.json"
        if not f.exists():
            continue
        payload = json.loads(f.read_text())
        for tcid, rec in payload["cells"].items():
            if tcid == source:
                continue  # off-diagonal only
            out[(source, tcid)] = rec
    return out


def recompute_g0_percell(cells: dict, g_meta: dict, sigma_c, lam: float, behavior: str) -> dict:
    """Recompute per-cell g0(C') from the reused #667 store + the join against G/tf.

    Returns {"rows": [{source, target, g0, G}...], "g0_vec": np.ndarray,
    "G_vec": np.ndarray} using the SAME _a39_cell_rows + _gate_pred_vec calls
    run_a39_a310 makes (only PERSISTED per-cell instead of collapsed). NOT a
    re-derivation of the gate — the gate DEFINITION, the cells store, sigma_c
    (#658), and lam are inherited verbatim.
    """
    from issue667_analysis import _a39_cell_rows, _gate_pred_vec, g_cell

    rows = _a39_cell_rows(cells, g_meta, behavior)
    g0_vec = _gate_pred_vec(rows, "c_C", "whitened", sigma_c, lam)
    out_rows = []
    G_list = []
    for r, g0 in zip(rows, g0_vec, strict=True):
        gc = g_cell(g_meta, behavior, r["source"], r["target"])
        G = float(gc["g"]) if gc is not None else float("nan")
        out_rows.append({"source": r["source"], "target": r["target"], "g0": float(g0), "G": G})
        G_list.append(G)
    return {
        "rows": out_rows,
        "g0_vec": np.asarray(g0_vec, float),
        "G_vec": np.asarray(G_list, float),
    }


def _finite_mask(*vecs) -> np.ndarray:
    m = np.ones(len(vecs[0]), dtype=bool)
    for v in vecs:
        m &= np.isfinite(np.asarray(v, float))
    return m


class G0CorrectnessError(RuntimeError):
    """Recomputed g0 aggregate Spearman(g0, G) diverged from #667's committed rho (HALT)."""


class CellCoverageError(RuntimeError):
    """A behavior is missing off-diagonal (source,target) cells with no valid excuse (HALT).

    HARD-FAIL for em/sycophancy (a partial extract / stale resume-skip / upload gap
    silently shrinks the headline denominator) and for fact when it was NOT dropped
    from the headline. A fact arm that WAS dropped (fact-pool under-yield) is
    SOFT-DROPPED, not a coverage error.
    """


def expected_off_diagonal_keys(behavior: str) -> set[tuple[str, str]]:
    """The full expected off-diagonal (source, target) cell set for a behavior.

    464 cells per behavior on the production grid (16 sources x 29 non-self
    targets). Sources = train_cids_for; targets = eval_cids_for; drop the diagonal
    (source == target), matching the #667 g0 join. Used to enforce cell coverage
    BEFORE the headline is written (BLOCKER 2).
    """
    from explore_persona_space.experiments.i537_contexts import eval_cids_for, train_cids_for

    sources = train_cids_for(behavior)
    targets = eval_cids_for(behavior)
    return {(s, t) for s in sources for t in targets if s != t}


def run_gate_vs_tf_margin(
    *,
    per_cell_dir: Path,
    tensors_dir: Path,
    behaviors: list[str],
    layer: int,
    out_dir: Path,
    skip_store_pin: bool = False,
    committed_base_g_rho: dict | None = None,
    fact_dropped: bool = False,
) -> dict:
    """The full join + both gates + headline stats. Returns the results dict.

    Raises G0CorrectnessError (-> rc!=0 in main) if any behavior's recomputed
    aggregate Spearman(g0, G) misses #667's committed base-G rho by > tol
    (smuggled-variable / base-side HALT). Raises CellCoverageError (BLOCKER 2)
    when a behavior is missing off-diagonal cells with no valid excuse: HARD-FAIL
    for em/sycophancy always, and for fact UNLESS ``fact_dropped`` is True (the
    fact-pool under-yield SOFT-DROP path, §4.3). A per-behavior
    measurement-validity FAIL is a REPORTABLE outcome (not carried as a headline),
    NOT a halt.
    """
    from issue667_analysis import load_cells, load_g_meta, load_sigma_c

    from explore_persona_space.analysis.issue667 import SIGMA_C_LAMBDA_FRACTION
    from explore_persona_space.analysis.issue667.gate_chain import (
        default_lambda,
        family_of,
        whitened_gate_reduction_unit_test,
    )

    ref_rho = committed_base_g_rho or COMMITTED_BASE_G_RHO

    # B3 reduction unit test gates the g0 recompute (inherited, plan §5).
    whitened_gate_reduction_unit_test()
    log.info("B3 reduction unit test PASS")

    g_meta = load_g_meta()
    sigma_c = load_sigma_c(layer)
    lam = default_lambda(sigma_c, SIGMA_C_LAMBDA_FRACTION)

    headline: dict[str, dict] = {}
    validation: dict[str, dict] = {}
    g0_percell_out: dict[str, list] = {}
    margins_out: dict[str, dict] = {}
    tf_margin_leak_out: dict[str, dict] = {}
    fact_softdropped = False

    for behavior in behaviors:
        # ── UNCONDITIONAL fact soft-drop when the DROP sentinel is present ──────
        # §4.3 mandates the fact drop as an INVARIANT once the pool is below the
        # yield floor ("dropped from the headline"), NOT a coverage-loss-only
        # response. A stale-but-COMPLETE per-cell fact store (resume_skip=True is
        # the dispatcher default, and #667 has a multi-run history) leaves the
        # coverage check's `missing` set empty, so the coverage-gated soft-drop
        # below would NEVER fire and fact would enter the headline with
        # fact_softdropped=False. Fire the drop here — BEFORE the correctness
        # gate, coverage gate, and stat compute — so the sentinel alone
        # (not stored-cell coverage) governs whether fact enters the headline.
        if behavior == "fact" and fact_dropped:
            fact_softdropped = True
            log.warning(
                "fact SOFT-DROP (unconditional): fact-pool flagged below YIELD_FLOOR_MIN "
                "(DROP sentinel present) -> skipping fact from the headline regardless of "
                "stored per-cell coverage (em/syco unaffected)"
            )
            continue

        cells = load_cells(tensors_dir, behavior, layer)
        rec = recompute_g0_percell(cells, g_meta, sigma_c, lam, behavior)
        rows = rec["rows"]
        g0_vec, G_vec = rec["g0_vec"], rec["G_vec"]
        g0_percell_out[behavior] = rows

        # Correctness gate: recomputed aggregate Spearman(g0, G) reproduces the
        # #667 committed base-G rho within tol (base-side / smuggled-variable).
        mgg = _finite_mask(g0_vec, G_vec)
        agg_g0_G = _spearman(g0_vec[mgg], G_vec[mgg])
        target_rho = ref_rho.get(behavior)
        if (
            agg_g0_G is None
            or target_rho is None
            or abs(agg_g0_G - target_rho) > G0_CORRECTNESS_TOL
        ):
            raise G0CorrectnessError(
                f"g0-recompute correctness gate FAIL for {behavior}: aggregate "
                f"Spearman(g0, G)={agg_g0_G} vs committed {target_rho} "
                f"(tol +/-{G0_CORRECTNESS_TOL}); the reused g0 diverged (wrong sigma_c/"
                f"lambda/store revision) -> HALT, do not carry the headline."
            )
        log.info(
            "g0-correctness gate PASS %s: Spearman(g0,G)=%.3f vs committed %.2f (n=%d)",
            behavior,
            agg_g0_G,
            target_rho,
            int(mgg.sum()),
        )

        # Join the freshly-extracted tf_margin_leak on the shared (source,target) key.
        tf_cells = load_tf_margin_leak(per_cell_dir, behavior)

        # ── BLOCKER 2: cell-coverage gate (BEFORE any headline is written) ──────
        # A partial extract / stale resume-skip / upload gap silently NaN-fills a
        # missing (source, target) cell downstream and shrinks the headline
        # denominator. Enforce the full expected off-diagonal key set here.
        # On a smoke (--skip-store-pin) the grid is deliberately subset, so the
        # coverage gate is inert (it would always trip on a 3-target smoke).
        if not skip_store_pin:
            expected = expected_off_diagonal_keys(behavior)
            actual = set(tf_cells.keys())
            missing = expected - actual
            if missing:
                sample = sorted(missing)[:5]
                if behavior == "fact" and fact_dropped:
                    # SOFT-DROP: fact was dropped from the headline (pool under-yield);
                    # skip it from the emission, note the reason, continue.
                    fact_softdropped = True
                    log.warning(
                        "cell-coverage SOFT-DROP fact: %d/%d expected cells missing "
                        "(sample=%s) AND fact dropped from headline (pool under-yield) "
                        "-> skipping fact from the headline (em/syco unaffected)",
                        len(missing),
                        len(expected),
                        sample,
                    )
                    continue
                raise CellCoverageError(
                    f"cell-coverage gate FAIL for {behavior}: {len(missing)} of "
                    f"{len(expected)} expected off-diagonal (source,target) cells "
                    f"missing from the tf-margin store (sample={sample}). A partial "
                    f"extract / stale resume-skip / upload gap silently shrinks the "
                    f"headline denominator"
                    + (
                        " and the fact arm was NOT flagged dropped -> HALT."
                        if behavior == "fact"
                        else " -> HALT."
                    )
                )
            log.info(
                "cell-coverage gate PASS %s: all %d expected off-diagonal cells present",
                behavior,
                len(expected),
            )

        tf_vec = np.array(
            [
                tf_cells.get((r["source"], r["target"]), {}).get("tf_margin_leak", np.nan)
                for r in rows
            ],
            dtype=float,
        )
        fams = [family_of(r["target"]) for r in rows]

        # ── CONCERN 2: aggregate deliverables (margins.json / tf_margin_leak.json) ─
        # Per-cell base + trained + leak margins keyed by (source, target). The
        # analyzer folds margins.json into the clean-result; tf_margin_leak.json is
        # the leak matrix the plot generator reads. Keyed by "<source>|<target>"
        # (JSON has no tuple keys). Missing cells carry NaN (already gated above).
        beh_margins: dict[str, dict] = {}
        beh_leak: dict[str, float] = {}
        for r in rows:
            tc = tf_cells.get((r["source"], r["target"]), {})
            key = f"{r['source']}|{r['target']}"
            beh_margins[key] = {
                "source": r["source"],
                "target": r["target"],
                "margin_base": tc.get("margin_base", float("nan")),
                "margin_trained": tc.get("margin_trained", float("nan")),
                "tf_margin_leak": tc.get("tf_margin_leak", float("nan")),
                "n_pos_used": tc.get("n_pos", None),
                "n_neg_used": tc.get("n_neg", None),
            }
            beh_leak[key] = tc.get("tf_margin_leak", float("nan"))
        margins_out[behavior] = beh_margins
        tf_margin_leak_out[behavior] = beh_leak

        # (a) Measurement-validity gate: Spearman(tf_margin_leak, G) per behavior.
        mv = _finite_mask(tf_vec, G_vec)
        mv_fams = [f for f, k in zip(fams, mv, strict=True) if k]
        mv_ci = clustered_bootstrap_spearman(tf_vec[mv], G_vec[mv], mv_fams)
        mv_passed = (
            mv_ci["point"] is not None and mv_ci["point"] > 0 and (mv_ci.get("ci_lo") or -1) > 0
        )
        validation[behavior] = {
            "rho": mv_ci["point"],
            "ci_lo": mv_ci.get("ci_lo"),
            "ci_hi": mv_ci.get("ci_hi"),
            "n_cells": int(mv.sum()),
            "n_families": mv_ci.get("n_families"),
            "passed": bool(mv_passed),
        }
        log.info(
            "validation gate %s: Spearman(tf,G)=%s CI=[%s,%s] passed=%s",
            behavior,
            mv_ci["point"],
            mv_ci.get("ci_lo"),
            mv_ci.get("ci_hi"),
            mv_passed,
        )

        # (b) Headline: Spearman(g0, tf_margin_leak) + shuffled null + delta vs base-G.
        mh = _finite_mask(g0_vec, tf_vec)
        mh_fams = [f for f, k in zip(fams, mh, strict=True) if k]
        hl_ci = clustered_bootstrap_spearman(g0_vec[mh], tf_vec[mh], mh_fams)
        # Shuffled null: permute tf_margin_leak within behavior (same L14).
        rng = np.random.default_rng(layer)
        null_vals = []
        g0m, tfm = g0_vec[mh], tf_vec[mh]
        for _ in range(N_BOOT):
            r = _spearman(g0m, rng.permutation(tfm))
            if r is not None:
                null_vals.append(r)
        null_hi = float(np.percentile(null_vals, 97.5)) if null_vals else None
        base_g_rho = target_rho  # the #667 committed base-G rho (recompute-validated above)
        headline[behavior] = {
            "rho": hl_ci["point"],
            "ci_lo": hl_ci.get("ci_lo"),
            "ci_hi": hl_ci.get("ci_hi"),
            "shuffled_null_hi": null_hi,
            "n_cells": int(mh.sum()),
            "n_families": hl_ci.get("n_families"),
            "base_G_rho": base_g_rho,
            "delta_vs_base_G": (hl_ci["point"] - base_g_rho)
            if hl_ci["point"] is not None
            else None,
            "validation_passed": bool(mv_passed),
        }
        log.info(
            "headline %s: Spearman(g0,tf)=%s CI=[%s,%s] null_hi=%s base_G_rho=%.2f delta=%s "
            "validation_passed=%s",
            behavior,
            hl_ci["point"],
            hl_ci.get("ci_lo"),
            hl_ci.get("ci_hi"),
            null_hi,
            base_g_rho,
            headline[behavior]["delta_vs_base_G"],
            mv_passed,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _repro_meta(
        {
            "layer": layer,
            "behaviors": behaviors,
            "g0_correctness_tol": G0_CORRECTNESS_TOL,
            # CONCERN 1: record whether the fact arm was soft-dropped from the
            # headline (fact-pool under-yield) so the analyzer + dashboard see it.
            "fact_dropped_from_headline": bool(fact_softdropped),
            "headline_behaviors": sorted(headline.keys()),
        }
    )
    (out_dir / "rho_gate_vs_tf_margin.json").write_text(
        json.dumps({"per_behavior": headline, "metadata": meta}, indent=2)
    )
    (out_dir / "rho_margin_vs_rate.json").write_text(
        json.dumps({"per_behavior": validation, "metadata": meta}, indent=2)
    )
    (out_dir / "g0_percell.json").write_text(
        json.dumps({"per_behavior": g0_percell_out, "metadata": meta}, indent=2)
    )
    # CONCERN 2: the plan §6.5 primary_deliverable names margins.json +
    # tf_margin_leak.json as REQUIRED aggregate outputs (folded into the
    # clean-result body / read by the plot generator).
    (out_dir / "margins.json").write_text(
        json.dumps({"per_behavior": margins_out, "metadata": meta}, indent=2)
    )
    (out_dir / "tf_margin_leak.json").write_text(
        json.dumps({"per_behavior": tf_margin_leak_out, "metadata": meta}, indent=2)
    )
    log.info(
        "wrote rho_gate_vs_tf_margin.json / rho_margin_vs_rate.json / g0_percell.json / "
        "margins.json / tf_margin_leak.json -> %s (fact_softdropped=%s)",
        out_dir,
        fact_softdropped,
    )
    return {
        "headline": headline,
        "validation": validation,
        "g0_percell": g0_percell_out,
        "margins": margins_out,
        "tf_margin_leak": tf_margin_leak_out,
        "fact_dropped_from_headline": bool(fact_softdropped),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #667 tf-margin gate->behavior analysis (CPU).")
    ap.add_argument(
        "--per-cell-dir",
        default="eval_results/issue_667/tf_margin/per_cell",
        help="Root of the tf-margin per-cell store (<beh>/<source>_seed*/tf_margins.json).",
    )
    ap.add_argument(
        "--tensors-dir",
        default="eval_results/issue_667/analysis_tensors",
        help="Root of the reused #667 activation store (for the g0 recompute).",
    )
    ap.add_argument(
        "--out-dir", default="eval_results/issue_667/tf_margin", help="Output dir for the JSONs."
    )
    ap.add_argument("--behaviors", nargs="+", default=["em", "sycophancy", "fact"])
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument(
        "--skip-store-pin", action="store_true", help="Synthetic-store smoke (no HF pins)."
    )
    ap.add_argument(
        "--fact-dropped",
        action="store_true",
        help="Fact arm was dropped from the headline (fact-pool under-yield): SOFT-DROP fact "
        "instead of a cell-coverage HALT. Auto-detected from the DROP sentinel too.",
    )
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # CONCERN 1: honor the flag OR the sentinel (either surfaces the fact drop).
    fact_dropped = args.fact_dropped or _fact_dropped_sentinel_present()

    try:
        run_gate_vs_tf_margin(
            per_cell_dir=PROJECT_ROOT / args.per_cell_dir,
            tensors_dir=PROJECT_ROOT / args.tensors_dir,
            behaviors=args.behaviors,
            layer=args.layer,
            out_dir=PROJECT_ROOT / args.out_dir,
            skip_store_pin=args.skip_store_pin,
            fact_dropped=fact_dropped,
        )
    except G0CorrectnessError as e:
        log.error("g0_correctness_gate_fail: %s", e)
        return 3
    except CellCoverageError as e:
        log.error("tf_margin_cell_coverage_incomplete: %s", e)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
