#!/usr/bin/env python3
"""Issue #562 rollup — context-panel paired deltas, classification, audit (VM, CPU).

Adapted copy of the pinned #558 rollup (issue-558 branch @
18959f7fca41b3e71d3e1cf128c7cbf50433aad2); plan tasks/.../562/plans/plan.md
section 4.2 enumerates the changes exhaustively. Reads the 12
``eval_results/issue_562/<arm>/seed<S>/phase2/run_summary.json`` files plus the
chain's committed artifacts and writes ``eval_results/issue_562/rollup.json``
with:

  - per-adapter paired deltas of each panel cell vs the within-run trigger
    re-read (``trigger50``), in BOTH spaces (EOS-margin = the SOLE
    classification space; log-prob = the pre-registered concordance check);
  - 10,000-resample cluster bootstrap (resample the 12 adapters, seed 562)
    95% percentile CIs, sign counts, per-arm means;
  - the plan section 7 ORDERED + EXHAUSTIVE classification per cell
    (T_dip = min(0.6 * D_doc, -1.0), instrument-matched to the within-run
    doctor re-read) INCLUDING the NEW registered symmetric sub-rule on the
    no-dip branch (rule 4): an EOS-margin no-dip whose log-prob read is
    dip-concordant in the DIRECTIONAL sense (log-prob n_neg >= 9/12 AND
    log-prob mean <= -0.4 nats) is labeled ``space-discordant-no-dip`` (LOW
    confidence). The account readout's clean-label set stays {dip, no-dip},
    so such a cell routes to unresolved-degraded and CANNOT certify a clean
    account-(B) ("persona framing required") verdict;
  - the descriptive account readout keyed on the bare-instruction cell
    ((A) context-general vs (B) persona-framing-specific; analyzer owns the
    verdict);
  - the NEW nurse_minus_comedian block (plan section 3 secondary
    discriminator): per-adapter paired difference d(nurse) - d(comedian) in
    both spaces, mean + cluster-bootstrap CI + sign count, with the
    registered medical-component predicate evaluated descriptively;
  - the cross-run instrument audit (NOT load-bearing): this run's doctor
    re-read and trigger re-read vs #558's recorded per-adapter
    run_summary.json values (both spaces; expected log-prob offsets <~0.3
    nats, EOS-margin offsets up to ~2 nats per the measured session
    divergence), plus a full recompute of the #543 same-subset calibration
    table from the chain's committed per-prompt slot stats, asserted against
    the registered numbers.

The #543-based calibration machinery (``assert_parent_entry_order``,
``CALIBRATION_EXPECT``, ``--calibration-only``) is kept VERBATIM from the
pinned rollup — "parent" in those helpers refers to the #543 chain data
(this task's grandparent). It validates the paired-delta / bootstrap /
classification code against committed data at zero GPU cost and serves as the
pre-launch analysis-path smoke (must reproduce doctor - trigger[0:50]
EOS-margin -2.598 [-4.215, -1.208 per-adapter range] 12/12 negative,
log-prob -1.171). ``--calibration-only`` writes
``eval_results/issue_562/calibration_audit.json``.

Usage (VM, after the pod is terminated; CPU-only):
    uv run python scripts/rollup_issue562_panel.py
    uv run python scripts/rollup_issue562_panel.py --calibration-only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="rollup_issue562_panel")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    ARMS,
    HUB_DATA_REPO,
    SEEDS,
    ensure_eval_questions_local,
    repro_metadata,
    trigger_user,
)
from eval_issue562_panel import (  # noqa: E402
    EVAL_RESULTS_DIR_562,
    ISSUE_562,
    PANEL_PHASE,
    PARENT_ISSUE,
    PARENT_RUN_SUMMARY_DIR,
)

log = logging.getLogger("rollup_issue562_panel")

# #543 chain data (the grandparent; calibration machinery inputs — verbatim).
PARENT_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "issue_543"
PARENT_RAW_BUCKET = "issue543_ratio_survival/raw_completions"

BASELINE_CELL = "trigger50"
CONTRAST_CELLS = ("doctor", "instruction_only", "nurse", "comedian")
N_ADAPTERS = 12
N_SUBSET = 50  # the [0:50] question slice every panel cell runs on

DEFAULT_N_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 562  # plan section 10 Reproducibility Card

# section 7 thresholds (verbatim parent).
T_DIP_FLOOR = -1.0  # a dip is never declared above -1.0 even if D_doc is weak
T_DIP_DOC_SCALE = 0.6
DIP_MIN_N_NEG = 10
NO_DIP_MAX_N_NEG = 7
GRADED_MIN_N_NEG = 8
LOGP_CONCORDANCE_MIN_N_NEG = 9
LOGP_CONCORDANCE_MIN_ABS_MEAN = 0.4

# Registered same-subset calibration numbers (#543 chain data; verbatim from
# the pinned #558 rollup — the brackets are registered per-adapter min/max
# RANGES, not bootstrap CIs). The rollup recompute must agree to the third
# decimal.
CAL_TOL = 0.002
CALIBRATION_EXPECT: dict[str, dict] = {
    "doctor_minus_trigger50": {
        "eosm_mean": -2.598,
        "eosm_range": (-4.215, -1.208),
        "eosm_n_neg": 12,
        "logp_mean": -1.171,
        "logp_range": (-1.918, -0.396),
        "logp_n_neg": 12,
    },
    "no_trigger50_minus_trigger50": {
        "eosm_mean": 0.840,
        "eosm_range": (-0.258, 2.551),
        "eosm_n_neg": 2,
        "logp_mean": 0.083,
        "logp_range": (-0.303, 0.563),
    },
    "reference_minus_trigger50": {
        "eosm_mean": -0.420,
        "eosm_range": (-1.196, 1.298),
        "eosm_n_neg": 9,
        "logp_mean": -0.335,
        "logp_range": (-0.699, 0.173),
    },
    "trigger50_minus_trigger200": {
        "eosm_mean": -0.855,
        "eosm_range": (-2.066, 0.175),
    },
}


def adapter_slugs() -> list[str]:
    """The 12 (arm, seed) cell keys in deterministic order."""
    return [f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion (parent rollup)."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def cluster_bootstrap_ci(
    values: list[float], *, n_resamples: int, seed: int
) -> tuple[float, float]:
    """95% percentile CI on the mean, resampling adapters with replacement."""
    arr = np.asarray(values, dtype=float)
    assert arr.ndim == 1 and len(arr) > 0, arr.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── #543 chain per-prompt data (committed slot stats + HF raw completions) ──


def load_parent_slot(slug: str, cell: str) -> dict:
    """#543 committed per-prompt 4-float slot stats for (adapter, cell)."""
    arm, seed = slug.split("_seed")
    path = PARENT_EVAL_DIR / arm / f"seed{seed}" / PANEL_PHASE / f"slot_stats_{cell}.json"
    if not path.exists():
        raise FileNotFoundError(f"Parent slot stats missing: {path}")
    slot = json.loads(path.read_text())
    n = slot["n"]
    if len(slot["trained"]) != n or len(slot["base"]) != n:
        raise RuntimeError(f"{path}: n={n} but trained/base lengths differ")
    return slot


def fetch_parent_completions(slug: str, cell: str) -> list[dict]:
    """#543 raw completions from the HF data bucket (per-record question text)."""
    from huggingface_hub import hf_hub_download

    fn = f"{PARENT_RAW_BUCKET}/{slug}_{PANEL_PHASE}/completions_{cell}.json"
    got = hf_hub_download(
        repo_id=HUB_DATA_REPO,
        filename=fn,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    return json.loads(Path(got).read_text())


def assert_parent_entry_order(eval_questions: list[str]) -> dict:
    """HARD assert: #543 per-prompt files are ordered as eval_questions slices.

    For every adapter: completions_trigger.json has exactly 200 records with
    user[i] == trigger_user(eval_questions[i]); completions_no_trigger.json
    has 200 with user[i] == eval_questions[i]; completions_doctor.json has 50
    with user[i] == trigger_user(eval_questions[i]); and the committed
    slot_stats entry counts match (the pinned worker computes slot stats in
    records order). Only after this does "first 50 entries = questions [0:50]"
    hold for the subset offsets below. The reference cell is used whole
    (questions [200:250], never subset) so its order is not load-bearing.
    """
    checked = []
    for slug in adapter_slugs():
        spec = [
            ("trigger", 200, True),
            ("no_trigger", 200, False),
            ("doctor", 50, True),
        ]
        for cell, n_expect, keyed in spec:
            recs = fetch_parent_completions(slug, cell)
            if len(recs) != n_expect:
                raise RuntimeError(
                    f"Entry-order assert FAIL: {slug}/completions_{cell}.json has "
                    f"{len(recs)} records, expected {n_expect}"
                )
            for i, rec in enumerate(recs):
                expect = trigger_user(eval_questions[i]) if keyed else eval_questions[i]
                if rec["user"] != expect:
                    raise RuntimeError(
                        f"Entry-order assert FAIL: {slug}/completions_{cell}.json[{i}] "
                        f"user does not match eval_questions[{i}]"
                    )
            slot = load_parent_slot(slug, cell)
            if slot["n"] != n_expect:
                raise RuntimeError(
                    f"Entry-order assert FAIL: {slug}/slot_stats_{cell}.json n={slot['n']}, "
                    f"expected {n_expect} (must match completions order/count)"
                )
        checked.append(slug)
    log.info("Entry-order assert PASSED for %d adapters x 3 cells.", len(checked))
    return {
        "passed": True,
        "adapters_checked": checked,
        "cells_checked": ["trigger", "no_trigger", "doctor"],
    }


def slot_deltas(slot: dict, lo: int = 0, hi: int | None = None) -> dict:
    """Per-cell trained-base delta means over entries [lo:hi] (parent machinery)."""
    tr, ba = slot["trained"][lo:hi], slot["base"][lo:hi]
    if not tr or len(tr) != len(ba):
        raise RuntimeError(f"Bad slot slice [{lo}:{hi}]: {len(tr)} vs {len(ba)} entries")
    d_logp = [t["logp"] - b["logp"] for t, b in zip(tr, ba, strict=True)]
    d_margin = [
        (t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"])
        for t, b in zip(tr, ba, strict=True)
    ]
    d_zm = [t["z_marker"] - b["z_marker"] for t, b in zip(tr, ba, strict=True)]
    return {
        "n": len(tr),
        "delta_logp_mean": _mean(d_logp),
        "delta_eos_margin_mean": _mean(d_margin),
        "delta_z_marker_mean": _mean(d_zm),
    }


# ── Paired-delta core (verbatim parent) ─────────────────────────────────────


def paired_deltas(cell_means: dict[str, dict[str, dict]], cells: tuple[str, ...]) -> dict:
    """Per-adapter paired deltas of each cell vs BASELINE_CELL, both spaces.

    ``cell_means[slug][cell]`` must carry delta_eos_margin_mean +
    delta_logp_mean (per-adapter within-cell means over the 50 prompts).
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for slug, by_cell in cell_means.items():
        base = by_cell[BASELINE_CELL]
        out[slug] = {}
        for cell in cells:
            s = by_cell[cell]
            out[slug][cell] = {
                "d_eosm": s["delta_eos_margin_mean"] - base["delta_eos_margin_mean"],
                "d_logp": s["delta_logp_mean"] - base["delta_logp_mean"],
            }
    return out


def cell_stats(
    per_adapter: dict[str, dict], cell: str, space: str, *, n_resamples: int, seed: int
) -> dict:
    """Mean, range, bootstrap CI, sign count for one (cell, space)."""
    key = {"eosm": "d_eosm", "logp": "d_logp"}[space]
    slugs = sorted(per_adapter)
    vals = [per_adapter[s][cell][key] for s in slugs]
    lo, hi = cluster_bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
    return {
        "per_adapter": {s: per_adapter[s][cell][key] for s in slugs},
        "mean": _mean(vals),
        "min": min(vals),
        "max": max(vals),
        "ci95": [lo, hi],
        "n_neg": sum(v < 0 for v in vals),
        "n": len(vals),
    }


def classify_cell(eosm: dict, logp: dict, t_dip: float) -> dict:
    """Plan section 7 ORDERED + EXHAUSTIVE classification of one panel cell.

    EOS margin is the SOLE classification space; log-prob is the concordance
    check (a dip needs >= 9/12 log-prob sign agreement AND |mean| >= 0.4 nats
    or it is reported as space-discordant). First match wins.

    NEW registered sub-rule on the no-dip branch (plan section 7 rule 4,
    symmetric to the rule-1 discordance flag — the parent never exercised
    this branch because all four of its cells dipped in both spaces): a cell
    satisfying the EOS-margin no-dip predicate whose log-prob read is
    dip-concordant in the DIRECTIONAL sense (log-prob n_neg >= 9/12 AND
    log-prob mean <= -0.4 nats — directional, not the absolute-value flag,
    since a positive-mean log-prob read SUPPORTS no-dip) is
    ``space-discordant-no-dip`` (LOW confidence). The no-dip branch carries
    the headline's falsification criterion, so the primary behavioral DV
    (log-prob) must not be silently overruled by the EOS-margin
    classification space on the branch that decides the headline.
    """
    d_mean, n_neg = eosm["mean"], eosm["n_neg"]
    concordant = (
        logp["n_neg"] >= LOGP_CONCORDANCE_MIN_N_NEG
        and abs(logp["mean"]) >= LOGP_CONCORDANCE_MIN_ABS_MEAN
    )
    # Directional dip-concordance (rule-4 sub-rule input).
    logp_dip_directional = (
        logp["n_neg"] >= LOGP_CONCORDANCE_MIN_N_NEG
        and logp["mean"] <= -LOGP_CONCORDANCE_MIN_ABS_MEAN
    )
    if d_mean <= t_dip and n_neg >= DIP_MIN_N_NEG:
        label = "dip" if concordant else "space-discordant-dip"
        rule = 1
    elif d_mean <= t_dip:  # n_neg < DIP_MIN_N_NEG
        label = "heterogeneous-unclassified"
        rule = 2
    elif (t_dip < d_mean <= T_DIP_FLOOR) or (d_mean > T_DIP_FLOOR and n_neg >= GRADED_MIN_N_NEG):
        label = "graded-partial"
        rule = 3
    else:  # d_mean > T_DIP_FLOOR and n_neg <= NO_DIP_MAX_N_NEG
        assert d_mean > T_DIP_FLOOR and n_neg <= NO_DIP_MAX_N_NEG, (d_mean, n_neg)
        label = "space-discordant-no-dip" if logp_dip_directional else "no-dip"
        rule = 4
    return {
        "label": label,
        "rule_matched": rule,
        "d_mean_eosm": d_mean,
        "n_neg_eosm": n_neg,
        "t_dip": t_dip,
        "logp_concordant": concordant,
        "logp_dip_directional": logp_dip_directional,
        "logp_mean": logp["mean"],
        "logp_n_neg": logp["n_neg"],
    }


def account_readout(labels: dict[str, str]) -> dict:
    """Descriptive section 3 signature-table match (analyzer owns the verdict).

    Keys on the bare-instruction cell for the (A)/(B) split; requires the
    doctor re-read to classify dip (calibration). The clean-label set stays
    {dip, no-dip}, so any space-discordant / graded / heterogeneous cell
    routes to unresolved-degraded (plan section 7 rule 4 sub-rule rationale).
    """
    doc = labels.get("doctor")
    instr = labels.get("instruction_only")
    nurse = labels.get("nurse")
    comedian = labels.get("comedian")
    clean = {"dip", "no-dip"}
    degraded = [c for c, v in labels.items() if v not in clean]
    if degraded:
        return {
            "signature_match": "unresolved-degraded",
            "degraded_cells": degraded,
            "note": "one or more cells classified outside clean dip/no-dip; "
            "nearest-account mapping is the analyzer's call (LOW confidence).",
        }
    if doc != "dip":
        return {
            "signature_match": "calibration-failure",
            "note": "doctor re-read did not reproduce a dip; no account assignment.",
        }
    pattern = (instr, nurse, comedian)
    if pattern == ("dip", "dip", "dip"):
        match = "(a) context-general"
    elif pattern == ("no-dip", "dip", "dip"):
        match = "(b) persona-framing-specific"
    else:
        match = "mixed-unclassified"
    return {"signature_match": match, "pattern_instruction_nurse_comedian": list(pattern)}


# ── Nurse - comedian secondary discriminator (plan section 3; NEW) ──────────


def nurse_minus_comedian_block(pd: dict, *, n_resamples: int, seed: int) -> dict:
    """Per-adapter paired difference d(nurse) - d(comedian), both spaces.

    The plan section 3 medical-component predicate is evaluated and stored
    DESCRIPTIVELY (the analyzer owns the verdict; confidence capped at
    MODERATE by the registered French-person single-persona precedent).
    """
    slugs = sorted(pd)
    out: dict[str, dict] = {}
    for space, key in (("eosm", "d_eosm"), ("logp", "d_logp")):
        vals = [pd[s]["nurse"][key] - pd[s]["comedian"][key] for s in slugs]
        lo, hi = cluster_bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
        out[space] = {
            "per_adapter": dict(zip(slugs, vals, strict=True)),
            "mean": _mean(vals),
            "min": min(vals),
            "max": max(vals),
            "ci95": [lo, hi],
            "n_neg": sum(v < 0 for v in vals),
            "n": len(vals),
        }
    eosm, logp = out["eosm"], out["logp"]
    ci_excludes_zero = eosm["ci95"][0] > 0.0 or eosm["ci95"][1] < 0.0
    mean_le_minus_1 = eosm["mean"] <= -1.0
    logp_sign_concordant = logp["n_neg"] >= LOGP_CONCORDANCE_MIN_N_NEG
    out["medical_component_predicate"] = {
        "ci_excludes_zero": ci_excludes_zero,
        "mean_le_minus_1_nat": mean_le_minus_1,
        "logp_sign_concordant": logp_sign_concordant,
        "satisfied": ci_excludes_zero and mean_le_minus_1 and logp_sign_concordant,
        "note": "descriptive only; analyzer owns the verdict, capped at MODERATE "
        "(French-person single-persona precedent, plan section 3).",
    }
    return out


# ── Calibration table (#543 chain data; runs in BOTH modes; verbatim) ───────


def parent_cell_means(slug: str) -> dict[str, dict]:
    """Pseudo cell means from the #543 per-prompt slot stats.

    trigger50 / no_trigger50 = entries [0:50] (order-asserted); trigger200 =
    all 200; doctor / reference = whole cells (n=50 each).
    """
    trig = load_parent_slot(slug, "trigger")
    return {
        "trigger50": slot_deltas(trig, 0, N_SUBSET),
        "trigger200": slot_deltas(trig),
        "no_trigger50": slot_deltas(load_parent_slot(slug, "no_trigger"), 0, N_SUBSET),
        "doctor": slot_deltas(load_parent_slot(slug, "doctor")),
        "reference": slot_deltas(load_parent_slot(slug, "reference")),
    }


def _assert_close(name: str, got: float, expect: float, tol: float = CAL_TOL) -> None:
    if abs(got - expect) > tol:
        raise RuntimeError(
            f"Calibration recompute FAIL: {name} = {got:.4f}, registered {expect:.4f} "
            f"(tol {tol}). Parent data or paired-delta code drifted — do not proceed."
        )


def same_subset_calibration(*, n_resamples: int, seed: int) -> dict:
    """Recompute the registered same-subset table from #543 per-prompt data.

    Runs the SAME paired-delta + bootstrap + classification machinery the
    production panel uses (the #543 doctor cell as a synthetic 1-cell panel
    against the #543 trigger[0:50] baseline), then HARD-asserts every
    registered number to the third decimal.
    """
    means = {slug: parent_cell_means(slug) for slug in adapter_slugs()}
    pd = paired_deltas(means, ("doctor", "no_trigger50", "reference", "trigger200"))

    table: dict[str, dict] = {}
    for cell in ("doctor", "no_trigger50", "reference"):
        table[f"{cell}_minus_trigger50"] = {
            "eosm": cell_stats(pd, cell, "eosm", n_resamples=n_resamples, seed=seed),
            "logp": cell_stats(pd, cell, "logp", n_resamples=n_resamples, seed=seed),
        }
    # trigger50 - trigger200 offset = -(trigger200 - trigger50).
    t200 = cell_stats(pd, "trigger200", "eosm", n_resamples=n_resamples, seed=seed)
    offset_vals = {s: -v for s, v in t200["per_adapter"].items()}
    table["trigger50_minus_trigger200"] = {
        "eosm": {
            "per_adapter": offset_vals,
            "mean": -t200["mean"],
            "min": min(offset_vals.values()),
            "max": max(offset_vals.values()),
            "n_neg": sum(v < 0 for v in offset_vals.values()),
            "n": len(offset_vals),
        }
    }

    # HARD asserts vs the registered numbers.
    for key, exp in CALIBRATION_EXPECT.items():
        got = table[key]
        _assert_close(f"{key}.eosm_mean", got["eosm"]["mean"], exp["eosm_mean"])
        _assert_close(f"{key}.eosm_min", got["eosm"]["min"], exp["eosm_range"][0])
        _assert_close(f"{key}.eosm_max", got["eosm"]["max"], exp["eosm_range"][1])
        if "eosm_n_neg" in exp and got["eosm"]["n_neg"] != exp["eosm_n_neg"]:
            raise RuntimeError(
                f"Calibration recompute FAIL: {key}.eosm_n_neg = {got['eosm']['n_neg']}, "
                f"registered {exp['eosm_n_neg']}"
            )
        if "logp_mean" in exp:
            _assert_close(f"{key}.logp_mean", got["logp"]["mean"], exp["logp_mean"])
        if "logp_range" in exp:
            _assert_close(f"{key}.logp_min", got["logp"]["min"], exp["logp_range"][0])
            _assert_close(f"{key}.logp_max", got["logp"]["max"], exp["logp_range"][1])
        if "logp_n_neg" in exp and got["logp"]["n_neg"] != exp["logp_n_neg"]:
            raise RuntimeError(
                f"Calibration recompute FAIL: {key}.logp_n_neg = {got['logp']['n_neg']}, "
                f"registered {exp['logp_n_neg']}"
            )
    log.info("Same-subset calibration table recompute PASSED all registered asserts.")

    # Classify the #543 doctor cell through the production rule (the
    # machinery correctness check: must come out a full-confidence dip).
    d_doc = table["doctor_minus_trigger50"]["eosm"]["mean"]
    t_dip = min(T_DIP_DOC_SCALE * d_doc, T_DIP_FLOOR)
    doc_cls = classify_cell(
        table["doctor_minus_trigger50"]["eosm"], table["doctor_minus_trigger50"]["logp"], t_dip
    )
    if doc_cls["label"] != "dip":
        raise RuntimeError(
            f"Calibration classification FAIL: parent doctor classified {doc_cls['label']!r}, "
            "expected 'dip' — classification code is wrong."
        )
    return {
        "table": table,
        "D_doc": d_doc,
        "T_dip": t_dip,
        "doctor_classification": doc_cls,
        "asserts": "all-passed",
        "expected": CALIBRATION_EXPECT,
    }


def run_calibration_only(args: argparse.Namespace) -> int:
    """Pre-launch smoke: entry-order assert + calibration table on #543 data."""
    eval_qs = ensure_eval_questions_local()
    order = assert_parent_entry_order(eval_qs)
    cal = same_subset_calibration(n_resamples=args.n_resamples, seed=args.bootstrap_seed)

    # Panel-schema view of the #543 doctor cell so the plot script can
    # render a calibration hero from this file directly.
    means = {slug: parent_cell_means(slug) for slug in adapter_slugs()}
    pd = paired_deltas(means, ("doctor",))
    panel = {
        "mode": "calibration",
        "baseline_cell": BASELINE_CELL,
        "adapters": adapter_slugs(),
        "D_doc": cal["D_doc"],
        "T_dip": cal["T_dip"],
        "doctor_calibration_failed": False,
        "cells": {
            "doctor": {
                "eosm": cell_stats(
                    pd, "doctor", "eosm", n_resamples=args.n_resamples, seed=args.bootstrap_seed
                ),
                "logp": cell_stats(
                    pd, "doctor", "logp", n_resamples=args.n_resamples, seed=args.bootstrap_seed
                ),
                "classification": cal["doctor_classification"],
            }
        },
    }
    out = {
        **repro_metadata(),
        "issue": ISSUE_562,
        "parent_issue": PARENT_ISSUE,
        "calibration_data_issue": 543,
        "mode": "calibration_audit",
        "bootstrap": {"n_resamples": args.n_resamples, "seed": args.bootstrap_seed},
        "panel": panel,
        "audit": {"entry_order_assert": order, "same_subset_calibration": cal},
    }
    out_path = EVAL_RESULTS_DIR_562 / "calibration_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Calibration audit -> %s", out_path)
    return 0


# ── Production rollup ────────────────────────────────────────────────────────


def load_run_summaries() -> dict[str, dict]:
    """The 12 run_summary.json files of THIS run (fail-loud on any missing)."""
    out: dict[str, dict] = {}
    missing = []
    for slug in adapter_slugs():
        arm, seed = slug.split("_seed")
        path = EVAL_RESULTS_DIR_562 / arm / f"seed{seed}" / PANEL_PHASE / "run_summary.json"
        if not path.exists():
            missing.append(str(path))
            continue
        rs = json.loads(path.read_text())
        cells = rs["cells"]
        want = {BASELINE_CELL, *CONTRAST_CELLS}
        if not want.issubset(cells):
            raise RuntimeError(f"{path}: cells {sorted(cells)} missing {sorted(want - set(cells))}")
        out[slug] = rs
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{N_ADAPTERS} run summaries missing:\n" + "\n".join(missing)
        )
    return out


def load_parent558_summary(slug: str) -> dict:
    """#558's recorded per-adapter run summary (committed on main; audit refs)."""
    arm, seed = slug.split("_seed")
    path = PARENT_RUN_SUMMARY_DIR / arm / f"seed{seed}" / PANEL_PHASE / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"#{PARENT_ISSUE} run summary missing: {path}")
    rs = json.loads(path.read_text())
    for cell in ("doctor", BASELINE_CELL):
        for k in ("delta_logp_mean", "delta_eos_margin_mean"):
            v = rs["cells"][cell][k]
            if not math.isfinite(v):
                raise RuntimeError(f"{path}: cells.{cell}.{k} non-finite")
    return rs


def run_rollup(args: argparse.Namespace) -> int:
    summaries = load_run_summaries()
    cell_means = {slug: rs["cells"] for slug, rs in summaries.items()}
    pd = paired_deltas(cell_means, CONTRAST_CELLS)

    # Per-cell stats in both spaces.
    cells: dict[str, dict] = {}
    for cell in CONTRAST_CELLS:
        cells[cell] = {
            "eosm": cell_stats(
                pd, cell, "eosm", n_resamples=args.n_resamples, seed=args.bootstrap_seed
            ),
            "logp": cell_stats(
                pd, cell, "logp", n_resamples=args.n_resamples, seed=args.bootstrap_seed
            ),
        }

    # section 7: T_dip from the WITHIN-RUN doctor re-read; classify every cell.
    d_doc = cells["doctor"]["eosm"]["mean"]
    t_dip = min(T_DIP_DOC_SCALE * d_doc, T_DIP_FLOOR)
    doctor_calibration_failed = d_doc > T_DIP_FLOOR
    labels: dict[str, str] = {}
    for cell in CONTRAST_CELLS:
        cls = classify_cell(cells[cell]["eosm"], cells[cell]["logp"], t_dip)
        cells[cell]["classification"] = cls
        labels[cell] = cls["label"]
    if doctor_calibration_failed:
        log.warning(
            "Doctor re-read did NOT reproduce a dip (D_doc=%.3f > %.1f): calibration "
            "failure — classifications are descriptive only, no account assignment.",
            d_doc,
            T_DIP_FLOOR,
        )

    # Nurse - comedian secondary discriminator (plan section 3; NEW).
    nurse_minus_comedian = nurse_minus_comedian_block(
        pd, n_resamples=args.n_resamples, seed=args.bootstrap_seed
    )

    # Per-arm means (ratio-independence check, parent convention).
    per_arm: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        per_arm[arm] = {}
        for cell in CONTRAST_CELLS:
            vals = [pd[f"{arm}_seed{s}"][cell]["d_eosm"] for s in SEEDS]
            per_arm[arm][cell] = _mean(vals)

    # Pooled emission per cell (sanity row; Wilson CIs ignore clustering).
    emission: dict[str, dict] = {}
    for cell in (BASELINE_CELL, *CONTRAST_CELLS):
        k = sum(
            round(rs["cells"][cell]["emission_rate"] * rs["cells"][cell]["n"])
            for rs in summaries.values()
        )
        n = sum(rs["cells"][cell]["n"] for rs in summaries.values())
        lo, hi = wilson_ci(k, n)
        emission[cell] = {
            "pooled_k": k,
            "pooled_n": n,
            "rate": k / n if n else None,
            "wilson_95ci": [lo, hi],
        }

    # ── Cross-run instrument audit (NOT load-bearing) ────────────────────────
    eval_qs = ensure_eval_questions_local()
    order = assert_parent_entry_order(eval_qs)
    cal = same_subset_calibration(n_resamples=args.n_resamples, seed=args.bootstrap_seed)

    # This run's doctor + trigger re-reads vs #558's recorded per-adapter
    # run_summary values (plan section 4.2 rollup change 6; expected log-prob
    # offsets <~0.3 nats, EOS-margin offsets up to ~2 nats per the measured
    # session divergence).
    doctor_vs_parent: dict[str, dict] = {}
    trigger50_vs_parent: dict[str, dict] = {}
    for slug in adapter_slugs():
        p558 = load_parent558_summary(slug)["cells"]
        this_doc = summaries[slug]["cells"]["doctor"]
        this_trig = summaries[slug]["cells"][BASELINE_CELL]
        doctor_vs_parent[slug] = {
            "offset_logp": this_doc["delta_logp_mean"] - p558["doctor"]["delta_logp_mean"],
            "offset_eosm": this_doc["delta_eos_margin_mean"]
            - p558["doctor"]["delta_eos_margin_mean"],
            "this_run": {k: this_doc[k] for k in ("delta_logp_mean", "delta_eos_margin_mean")},
            "parent": {k: p558["doctor"][k] for k in ("delta_logp_mean", "delta_eos_margin_mean")},
        }
        trigger50_vs_parent[slug] = {
            "offset_logp": this_trig["delta_logp_mean"] - p558[BASELINE_CELL]["delta_logp_mean"],
            "offset_eosm": this_trig["delta_eos_margin_mean"]
            - p558[BASELINE_CELL]["delta_eos_margin_mean"],
            "this_run": {k: this_trig[k] for k in ("delta_logp_mean", "delta_eos_margin_mean")},
            "parent": {
                k: p558[BASELINE_CELL][k] for k in ("delta_logp_mean", "delta_eos_margin_mean")
            },
        }
    audit = {
        "parent_issue": PARENT_ISSUE,
        "entry_order_assert": order,
        "same_subset_calibration": cal,
        "doctor_reread_vs_parent": {
            "per_adapter": doctor_vs_parent,
            "mean_offset_logp": _mean([v["offset_logp"] for v in doctor_vs_parent.values()]),
            "mean_offset_eosm": _mean([v["offset_eosm"] for v in doctor_vs_parent.values()]),
        },
        "trigger50_vs_parent_trigger50": {
            "per_adapter": trigger50_vs_parent,
            "mean_offset_logp": _mean([v["offset_logp"] for v in trigger50_vs_parent.values()]),
            "mean_offset_eosm": _mean([v["offset_eosm"] for v in trigger50_vs_parent.values()]),
        },
        "anchor_fields": {slug: summaries[slug].get("anchor") for slug in adapter_slugs()},
    }

    rollup = {
        **repro_metadata(),
        "issue": ISSUE_562,
        "parent_issue": PARENT_ISSUE,
        "mode": "production",
        "bootstrap": {"n_resamples": args.n_resamples, "seed": args.bootstrap_seed},
        "panel": {
            "mode": "production",
            "baseline_cell": BASELINE_CELL,
            "adapters": adapter_slugs(),
            "per_adapter_paired_deltas": pd,
            "cells": cells,
            "D_doc": d_doc,
            "T_dip": t_dip,
            "doctor_calibration_failed": doctor_calibration_failed,
            "labels": labels,
            "account_readout": (
                {
                    "signature_match": "calibration-failure",
                    "note": "doctor re-read did not reproduce a dip; no account assignment.",
                }
                if doctor_calibration_failed
                else account_readout(labels)
            ),
            "nurse_minus_comedian": nurse_minus_comedian,
            "per_arm_eosm_means": per_arm,
        },
        "cell_summaries": cell_means,
        "emission": emission,
        "audit": audit,
    }
    out_path = Path(args.out) if args.out else EVAL_RESULTS_DIR_562 / "rollup.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rollup, indent=2))
    log.info("Rollup -> %s", out_path)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #562 rollup: panel paired deltas, classification, audit (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--calibration-only",
        action="store_true",
        help="Run ONLY the chain-data audit path (pre-launch smoke); no run data needed.",
    )
    p.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    p.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    p.add_argument("--out", type=str, default=None, help="Output JSON path (production mode).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.calibration_only:
        return run_calibration_only(args)
    return run_rollup(args)


if __name__ == "__main__":
    raise SystemExit(main())
