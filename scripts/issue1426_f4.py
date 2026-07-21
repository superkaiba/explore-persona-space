#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, ×, Δ) in scientific docstrings + log messages.
"""Issue #1426 Phase F4: covariate re-reductions (VM-side, 0 GPU — plan §4.6).

Pure re-reductions of THIS run's persisted per-context LOCO error tensors
(``decomp_<regime>.pt`` from F1) plus the parent-side baseline:

1. **Per-context deltas** — the parent's H3 (composed − direct) / H4
   (context+CoT − CoT-alone) companions PLUS the per-context CoT gain
   (``g_aug`` − ``d_ctx2ans``), each pooled value gate-asserted against THIS
   run's committed bootstrap blob (atol 1e-9) → ``percontext_deltas.json``.
2. **Length-matched covariate battery under THIS model's coverage** —
   flagged/unflagged split (recomputed; may be empty at restored compliance),
   unflagged-non-collapse short-CoT terciles, greedy nearest-neighbor
   length-matched pairing → ``length_matched_gain.json``.
3. **Δ_fam (THE registered H2 covariate contrast, plan §3)** — mean per-context
   CoT gain over the ICL+WildChat contexts minus their greedy nearest-neighbor
   same-median-CoT-length partners drawn from the other contexts (family
   membership is ex-ante — flag-independent), paired bootstrap over PAIRS
   (seed 42), per-question regime primary + query-averaged reported; plus
   match-quality diagnostics (per-pair char gaps, donor-reuse counts — 0 by
   construction under without-replacement matching; parent reference mean gap
   191 chars) → ``fam_contrast.json``.
4. **Tri-lineage like-for-like Δ_fam baselines (plan §4.6)** — the SAME
   family-keyed contrast recomputed on BOTH prior lineages' committed
   ``percontext_deltas.json`` blobs (gains joined from their h3/h4 rows) with
   per-context lengths from each lineage's committed figure meta — the metas
   are KEY-heterogeneous (#928: per-regime ``percontext_scatter_*.meta.json``
   keyed "median CoT length (chars)"; #1005: one
   ``fam_contrast_length_matched.meta.json`` keyed "median well-formed CoT
   length (chars)") so each lineage spec names its length source + key
   EXPLICITLY, and each reproduced per-question baseline is parity-gated
   against the committed value (+0.520 #928 / +0.426 #1005 — a mismatch
   beyond rounding is a code-drift stop, not a finding).

Usage::

    uv run python scripts/issue1426_f4.py \\
        --in-results eval_results/issue_1426 --store data/issue_1426/store \\
        --rollouts data/issue_1426/raw_completions/thinking_rollouts \\
        --out-figures figures/issue_1426
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue928_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    PARSE_RATE_FLOOR,
    dump_json,
    load_json,
    reproducibility_metadata,
    resolve_battery,
    segment_completion,
)
from issue928_length_matched_gain import (  # noqa: E402
    contrast_stat,
    delta_stat,
    greedy_nn_match,
    spearman,
    tercile_bins,
)
from issue928_null_bootstrap import make_bootstrap_index_matrix, stat_summary  # noqa: E402
from issue1426_common import COLLAPSE_FAMILIES, PARSER_RUNG  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue1426_f4")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PRIMARY_COMBO = "mean/mean"
GAIN_ARM, DIRECT_ARM = "g_aug", "d_ctx2ans"
REGIMES = ("indiv", "avg_q")  # per-question regime PRIMARY for covariate reads (plan §8)

# Tri-lineage baseline specs (plan §4.6): each names its deltas JSON, its
# length-meta source (``{regime}`` placeholder = per-regime files, the #928
# shape; no placeholder = ONE file reused across regimes, the #1005 shape) and
# the EXPLICIT per-lineage points key (the metas are key-heterogeneous —
# consistency-checker note 1). Paths resolve against the repo root.
BASELINE_LINEAGES: tuple[dict, ...] = (
    {
        "name": "issue928",
        "deltas": str(PROJECT_ROOT / "eval_results" / "issue_928" / "percontext_deltas.json"),
        "length_meta": str(
            PROJECT_ROOT / "figures" / "issue_928" / "percontext_scatter_{regime}.meta.json"
        ),
        "length_key": "median CoT length (chars)",
    },
    {
        "name": "issue1005",
        "deltas": str(PROJECT_ROOT / "eval_results" / "issue_1005" / "percontext_deltas.json"),
        "length_meta": str(
            PROJECT_ROOT / "figures" / "issue_1005" / "fam_contrast_length_matched.meta.json"
        ),
        "length_key": "median well-formed CoT length (chars)",
    },
)
# Per-question-regime parity targets (plan §4.6): the reproduced baseline must
# match the committed 3-decimal citation within rounding — exact committed
# values 0.5203427… (#1005's parent_928_family_keyed_baseline.indiv.observed)
# and 0.4259485… (#1005's own fam_contrast by_regime.indiv.observed).
PARITY_TARGETS: dict[str, tuple[str, float]] = {
    "issue928": ("indiv", 0.520),
    "issue1005": ("indiv", 0.426),
}


def per_ctx_skill(decomp: dict, arm: str, layer: int, n_ctx: int) -> np.ndarray:
    """(n_ctx,) per-context held-out skill 1 − ss_res/ss_tot at (arm, mean/mean, layer).

    Length-generalized twin of ``issue928_percontext_deltas.per_context_skill``
    (whose shape assert is pinned to the parent's 50-grid)."""
    key = str((arm, PRIMARY_COMBO, layer))
    if key not in decomp:
        raise RuntimeError(f"decomp missing key {key}")
    v = decomp[key]
    res = np.asarray(v["ss_res"], np.float64)
    tot = np.asarray(v["ss_tot"], np.float64)
    assert res.shape == tot.shape == (n_ctx,), (res.shape, tot.shape, n_ctx)
    return 1.0 - res / np.clip(tot, 1e-12, None)


def pooled_delta(decomp: dict, hi: str, lo: str, layer: int) -> float:
    """Pooled Δskill (Σ-of-errors form — the committed statistic's estimator)."""
    out = {}
    for arm in (hi, lo):
        v = decomp[str((arm, PRIMARY_COMBO, layer))]
        out[arm] = 1.0 - float(np.asarray(v["ss_res"]).sum()) / float(np.asarray(v["ss_tot"]).sum())
    return out[hi] - out[lo]


def assert_matches_committed(observed: float, boot_blob: dict, regime: str, key: str) -> None:
    """Re-reduction must reproduce THIS run's committed observed value (atol 1e-9)."""
    ref = boot_blob["by_regime"][regime]["statistics"][key]["primary_frozen_direct_best"]
    dev = abs(observed - float(ref["observed"]))
    if dev > 1e-9:
        raise AssertionError(f"{regime}/{key}: re-reduction diverges from committed by {dev:.3e}")


def cot_lengths_from_rollouts(rollouts_dir: Path, ctx_ids: list[str]) -> np.ndarray:
    """(n_ctx,) per-context MEDIAN well-formed CoT char length from rollout JSONs.

    Segmentation re-runs the run's own parser convention (prefill semantics on
    all rungs); a context with zero well-formed rows fails loud (it cannot be
    length-matched)."""
    out = []
    for c in ctx_ids:
        blob = load_json(rollouts_dir / f"{c}.json")
        lens = []
        for row in blob["completions"]:
            wf, _reason, cot_span, _ans = segment_completion(row["completion"], PARSER_RUNG)
            if wf:
                lens.append(cot_span[1] - cot_span[0])
        if not lens:
            raise RuntimeError(f"context {c}: zero well-formed rows — cannot length-match")
        out.append(float(np.median(lens)))
    return np.asarray(out, dtype=np.float64)


def fam_delta_fam(
    gains: np.ndarray,
    lengths: np.ndarray,
    fam_pos: np.ndarray,
    donor_pos: np.ndarray,
    n_boot: int,
) -> dict:
    """The registered Δ_fam (plan §3): per-pair gain differences, mean over the
    ICL+WildChat pairs, paired bootstrap over PAIRS (seed 42). Also returns the
    match-quality diagnostics (per-pair length gaps + donor-reuse counts)."""
    pairs = greedy_nn_match(lengths, fam_pos, donor_pos)
    d = np.asarray([gains[f] - gains[u] for f, u, _g in pairs], dtype=np.float64)
    idx = make_bootstrap_index_matrix(len(pairs), n_boot, BOOTSTRAP_SEED)
    draws = d[idx].mean(axis=1)
    donors = [u for _f, u, _g in pairs]
    gaps = [g for _f, _u, g in pairs]
    return {
        "n_pairs": len(pairs),
        "pairs": [
            {"fam_ctx_pos": int(f), "donor_ctx_pos": int(u), "length_gap_chars": float(g)}
            for f, u, g in pairs
        ],
        **stat_summary(float(d.mean()), draws),
        "match_quality": {
            "mean_length_gap_chars": float(np.mean(gaps)),
            "max_length_gap_chars": float(np.max(gaps)),
            "donor_reuse_max": int(max(np.bincount(donors).max(), 1)) if donors else 0,
            "note": "without-replacement greedy NN matching — donor reuse is 0/1 by "
            "construction; parent flag-keyed reference mean gap 191 chars",
        },
    }


def lineage_family_baseline(
    deltas_path: Path,
    length_meta_tpl: str,
    length_key: str,
    families: dict[str, str],
    n_boot: int,
) -> dict:
    """Like-for-like family-keyed Δ_fam baseline on ONE prior lineage's COMMITTED artifacts.

    Per-context CoT gains join ``skill_g_aug`` (h4 rows) with ``skill_d_ctx2ans``
    (h3 rows) at the lineage's own frozen layer per regime (both prior blobs
    carry the h3/h4 contrasts); per-context median well-formed CoT lengths come
    from the lineage's committed figure meta — ``length_meta_tpl`` may carry a
    ``{regime}`` placeholder (per-regime files, the #928 shape) or name ONE
    file reused across regimes (the #1005 shape), keyed by the EXPLICIT
    per-lineage ``length_key``; a context missing the key fails loud. Family
    keying uses the SAME ex-ante battery families."""
    blob = load_json(deltas_path)
    out: dict = {}
    for regime in REGIMES:
        h3 = blob["contrasts"]["h3_composed_direct_percontext"]["by_regime"][regime]["per_context"]
        h4 = blob["contrasts"]["h4_sufficiency_percontext"]["by_regime"][regime]["per_context"]
        d_skill = {r["context"]: float(r["skill_d_ctx2ans"]) for r in h3}
        g_skill = {r["context"]: float(r["skill_g_aug"]) for r in h4}
        ctxs = [r["context"] for r in h3]
        assert set(ctxs) == set(g_skill), "h3/h4 context sets drifted in the baseline blob"
        meta = load_json(Path(str(length_meta_tpl).format(regime=regime)))
        # Per-context lengths live in MULTI-member point groups (the scatter
        # series: 50 per #928 panel; 32 donors + 18 fam per #1005 panel). The
        # #1005 meta ALSO carries singleton AGGREGATE points (tercile diamonds)
        # that inherit a coinciding context's annotation label and would
        # otherwise shadow the true per-context value — exclude singleton
        # groups, then require ONE unique value per label (fail loud).
        group_sizes: dict = {}
        for pt in meta["points"]:
            group_sizes[pt.get("_group")] = group_sizes.get(pt.get("_group"), 0) + 1
        vals_by_label: dict[str, set[float]] = {}
        for pt in meta["points"]:
            if "label" in pt and length_key in pt and group_sizes[pt.get("_group")] > 1:
                vals_by_label.setdefault(pt["label"], set()).add(float(pt[length_key]))
        len_by_ctx: dict[str, float] = {}
        for lab, vals in vals_by_label.items():
            if len(vals) != 1:
                raise RuntimeError(
                    f"ambiguous per-context length for {lab!r} in {length_meta_tpl}: {sorted(vals)}"
                )
            len_by_ctx[lab] = next(iter(vals))
        missing = [c for c in ctxs if c not in len_by_ctx]
        if missing:
            raise RuntimeError(f"baseline meta lacks {length_key!r} for {missing[:3]}…")
        gains = np.asarray([g_skill[c] - d_skill[c] for c in ctxs], dtype=np.float64)
        lengths = np.asarray([len_by_ctx[c] for c in ctxs], dtype=np.float64)
        fam_pos = np.asarray(
            [i for i, c in enumerate(ctxs) if families[c] in COLLAPSE_FAMILIES], dtype=np.int64
        )
        donor_pos = np.asarray(
            [i for i, c in enumerate(ctxs) if families[c] not in COLLAPSE_FAMILIES],
            dtype=np.int64,
        )
        out[regime] = {
            "contexts_fam": [ctxs[i] for i in fam_pos],
            "frozen_layer_source": "lineage percontext_deltas.json (its own frozen convention)",
            **fam_delta_fam(gains, lengths, fam_pos, donor_pos, n_boot),
        }
    return out


def make_fam_figure(per_regime: dict, ctx_ids, families, out_dir: Path, slug: str) -> None:
    """Per-context CoT gain vs median CoT length; ICL/WildChat highlighted, Δ_fam
    pair segments drawn, tercile diamonds (the parent's covariate figure re-keyed)."""
    set_paper_style()
    fig, axes = plt.subplots(
        1, len(per_regime), figsize=(5.6 * len(per_regime), 4.4), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    c_fam = paper_palette_role("accent")
    c_donor = paper_palette_role("primary")
    for ax, (regime, r) in zip(axes, per_regime.items(), strict=True):
        gains, lengths = np.asarray(r["gains"]), np.asarray(r["lengths"])
        fam_mask = np.asarray([families[c] in COLLAPSE_FAMILIES for c in ctx_ids])
        ax.scatter(
            lengths[~fam_mask],
            gains[~fam_mask],
            s=14,
            color=c_donor,
            label=f"donor pool (n={int((~fam_mask).sum())})",
        )
        ax.scatter(
            lengths[fam_mask],
            gains[fam_mask],
            s=18,
            color=c_fam,
            label=f"ICL+WildChat (n={int(fam_mask.sum())})",
        )
        for p in r["fam_contrast"]["pairs"]:
            f, u = p["fam_ctx_pos"], p["donor_ctx_pos"]
            ax.plot([lengths[f], lengths[u]], [gains[f], gains[u]], lw=0.4, color="0.7", zorder=0)
        for i, c in enumerate(ctx_ids):
            ax.annotate(c, (lengths[i], gains[i]), fontsize=4, rotation=30)
        for tb in r.get("tercile_summaries", []):
            ax.scatter(
                [tb["median_length"]],
                [tb["observed"]],
                marker="D",
                s=40,
                color="tab:green",
                zorder=3,
            )
        st = r["fam_contrast"]
        ax.axhline(0.0, lw=0.8, color="0.5")
        ax.set_xlabel("median well-formed CoT length (chars)")
        ax.set_ylabel("per-context CoT gain (g_aug − d_ctx2ans)")
        ax.set_title(
            f"{regime}: Δ_fam {st['observed']:+.3f} CI[{st['ci95'][0]:+.3f},{st['ci95'][1]:+.3f}]"
        )
        ax.legend(fontsize=7)
    savefig_paper(fig, f"{out_dir.name}/{slug}", dir=str(out_dir.parent))
    plt.close(fig)


def run_lineage_baselines(args, fam_of: dict[str, str]) -> dict:
    """Tri-lineage like-for-like Δ_fam baselines + per-lineage parity gates (plan §4.6).

    Runs each spec (CLI ``--baseline-percontext`` overrides, else the
    registered ``BASELINE_LINEAGES``) through ``lineage_family_baseline`` and
    parity-gates every name in ``PARITY_TARGETS`` against its committed
    per-question value — a mismatch beyond ``--baseline-parity-atol`` raises
    (code-drift stop, not a finding)."""
    specs = (
        [
            {"name": n, "deltas": d, "length_meta": m, "length_key": k}
            for n, d, m, k in args.baseline_percontext
        ]
        if args.baseline_percontext
        else [dict(spec) for spec in BASELINE_LINEAGES]
    )
    baselines: dict = {}
    for spec in specs:
        entry = lineage_family_baseline(
            Path(spec["deltas"]), spec["length_meta"], spec["length_key"], fam_of, args.n_boot
        )
        baselines[spec["name"]] = {
            "deltas_json": str(spec["deltas"]),
            "length_meta": str(spec["length_meta"]),
            "length_key": spec["length_key"],
            "by_regime": entry,
        }
        for regime, r in entry.items():
            logger.info(
                "[baseline %s %s] family-keyed Δ_fam=%+.4f ci95=%s (%d pairs)",
                spec["name"],
                regime,
                r["observed"],
                r["ci95"],
                r["n_pairs"],
            )
        parity = PARITY_TARGETS.get(spec["name"])
        if parity is None:
            continue
        p_regime, expected = parity
        got = float(entry[p_regime]["observed"])
        if abs(got - expected) > args.baseline_parity_atol:
            raise RuntimeError(
                f"baseline parity gate FAILED for {spec['name']} ({p_regime}): "
                f"reproduced Δ_fam {got:+.6f} vs committed {expected:+.3f} "
                f"(atol {args.baseline_parity_atol:g}) — code drift in the "
                "family-contrast machinery; a mismatch beyond rounding is a "
                "code-drift stop, not a finding (plan §4.6)"
            )
        baselines[spec["name"]]["parity_gate"] = {
            "regime": p_regime,
            "committed": expected,
            "reproduced": got,
            "atol": args.baseline_parity_atol,
            "pass": True,
        }
        logger.info(
            "[baseline %s] parity gate PASS (%+.6f vs committed %+.3f)",
            spec["name"],
            got,
            expected,
        )
    return baselines


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1426 F4 covariate re-reductions (0 GPU)")
    ap.add_argument("--in-results", default=str(PROJECT_ROOT / "eval_results" / "issue_1426"))
    ap.add_argument("--out-results", default=None, help="default: --in-results")
    ap.add_argument("--store", default=str(PROJECT_ROOT / "data" / "issue_1426" / "store"))
    ap.add_argument(
        "--rollouts",
        default=str(PROJECT_ROOT / "data" / "issue_1426" / "raw_completions" / "thinking_rollouts"),
    )
    ap.add_argument("--battery", default=None)
    ap.add_argument("--out-figures", default=str(PROJECT_ROOT / "figures" / "issue_1426"))
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_DRAWS)
    ap.add_argument(
        "--baseline-percontext",
        nargs=4,
        action="append",
        default=None,
        metavar=("NAME", "DELTAS_JSON", "LENGTH_META", "LENGTH_KEY"),
        help="per-lineage baseline spec override: deltas JSON + length-meta path (may carry a "
        "{regime} placeholder) + EXPLICIT length key; default = the registered tri-lineage "
        "pair (#928 + #1005, plan §4.6); parity gates apply to names in PARITY_TARGETS",
    )
    ap.add_argument(
        "--baseline-parity-atol",
        type=float,
        default=5e-4,
        help="parity-gate tolerance vs the committed 3-decimal citations (+0.520 / +0.426)",
    )
    ap.add_argument(
        "--skip-baselines",
        action="store_true",
        help="omit the prior-lineage baselines (smoke without the committed artifacts)",
    )
    args = ap.parse_args()

    in_results = Path(args.in_results)
    out_results = Path(args.out_results) if args.out_results else in_results
    out_figures = Path(args.out_figures)
    out_results.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    battery = resolve_battery(Path(args.battery) if args.battery else None)
    fam_of = {i["id"]: i["family"] for i in battery["instances"]}
    manifest = load_json(Path(args.store) / "manifest.json")
    ctx_ids: list[str] = manifest["context_ids"]
    n_ctx = len(ctx_ids)
    flagged = set(manifest.get("flagged_below_parse_floor", []))
    boot_blob = load_json(in_results / "bootstrap_deltaskill.json")
    decomps = {r: torch.load(in_results / f"decomp_{r}.pt", weights_only=False) for r in REGIMES}
    layers = {
        r: int(boot_blob["by_regime"][r]["layer_conventions"]["primary_frozen_direct_best_layer"])
        for r in REGIMES
    }
    lengths = cot_lengths_from_rollouts(Path(args.rollouts), ctx_ids)
    idx = make_bootstrap_index_matrix(n_ctx, args.n_boot, BOOTSTRAP_SEED)

    # ── 1. per-context deltas (H3/H4 companions + the CoT gain) ───────────────
    contrasts_spec = (
        ("h2_cot_gain_percontext", GAIN_ARM, DIRECT_ARM, "H2_delta_g_minus_d"),
        ("h3_composed_direct_percontext", "comp_pred", DIRECT_ARM, "H3_delta_comp_minus_d"),
        ("h4_sufficiency_percontext", GAIN_ARM, "b_cot2ans", "H4_delta_g_minus_b"),
    )
    pcd_blob: dict = {
        "dv": (
            "Per-context Δ held-out skill re-reductions at the primary frozen convention "
            "(mean/mean, direct-arm full-data best LOCO layer per regime) — H2 CoT gain + "
            "the parent's H3/H4 companions"
        ),
        "primary_combo": PRIMARY_COMBO,
        "flagged_below_parse_floor": sorted(flagged),
        "contrasts": {},
        "reproducibility": reproducibility_metadata(),
    }
    gains_by_regime: dict[str, np.ndarray] = {}
    for stem, hi, lo, committed_key in contrasts_spec:
        per_regime = {}
        for regime in REGIMES:
            layer = layers[regime]
            decomp = decomps[regime]
            hi_s = per_ctx_skill(decomp, hi, layer, n_ctx)
            lo_s = per_ctx_skill(decomp, lo, layer, n_ctx)
            pooled = pooled_delta(decomp, hi, lo, layer)
            assert_matches_committed(pooled, boot_blob, regime, committed_key)
            delta = hi_s - lo_s
            if stem == "h2_cot_gain_percontext":
                gains_by_regime.setdefault(regime, delta)
            per_regime[regime] = {
                "frozen_layer": layer,
                "pooled_delta": pooled,
                "per_context": [
                    {
                        "context": c,
                        "family": fam_of[c],
                        "flagged": c in flagged,
                        "delta": float(delta[i]),
                        f"skill_{hi}": float(hi_s[i]),
                        f"skill_{lo}": float(lo_s[i]),
                    }
                    for i, c in enumerate(ctx_ids)
                ],
            }
            logger.info("[%s %s @L%d] pooled Δ=%+.4f (gate PASS)", stem, regime, layer, pooled)
        pcd_blob["contrasts"][stem] = {
            "delta": f"skill({hi}) - skill({lo})",
            "by_regime": per_regime,
        }
    dump_json(pcd_blob, out_results / "percontext_deltas.json")

    # ── 2. length-matched battery under THIS model's coverage ─────────────────
    fam_pos = np.asarray(
        [i for i, c in enumerate(ctx_ids) if fam_of[c] in COLLAPSE_FAMILIES], dtype=np.int64
    )
    donor_pos = np.asarray(
        [i for i, c in enumerate(ctx_ids) if fam_of[c] not in COLLAPSE_FAMILIES], dtype=np.int64
    )
    flagged_pos = np.asarray([i for i, c in enumerate(ctx_ids) if c in flagged], dtype=np.int64)
    unflagged_pos = np.asarray(
        [i for i, c in enumerate(ctx_ids) if c not in flagged], dtype=np.int64
    )
    lmg_blob: dict = {
        "dv": "length-matched covariate battery under THIS model's realized coverage",
        "parse_floor": PARSE_RATE_FLOOR,
        "flagged_contexts": sorted(flagged),
        "by_regime": {},
        "reproducibility": reproducibility_metadata(),
    }
    fam_fig_inputs: dict = {}
    for regime in REGIMES:
        layer = layers[regime]
        decomp = decomps[regime]
        arms = {}
        for arm in (GAIN_ARM, DIRECT_ARM):
            v = decomp[str((arm, PRIMARY_COMBO, layer))]
            arms[arm] = {
                "ss_res": np.asarray(v["ss_res"], np.float64),
                "ss_tot": np.asarray(v["ss_tot"], np.float64),
            }
        entry: dict = {"frozen_layer": layer}
        # flagged/unflagged split — recomputed under this model's coverage; the
        # parent's machinery runs ONLY where the flagged set is non-empty (§4.0.3).
        if flagged_pos.size and unflagged_pos.size:
            entry["flagged_vs_unflagged"] = contrast_stat(arms, flagged_pos, unflagged_pos, idx)
            fl_pairs = greedy_nn_match(lengths, flagged_pos, unflagged_pos)
            entry["flagged_vs_length_matched"] = contrast_stat(
                arms,
                np.asarray([f for f, _u, _g in fl_pairs], dtype=np.int64),
                np.asarray([u for _f, u, _g in fl_pairs], dtype=np.int64),
                idx,
            )
        else:
            entry["flagged_vs_unflagged"] = {
                "note": "flagged set empty — degenerate at restored compliance (plan §4.0.3)"
            }
        # short-CoT tercile gradient within the non-collapse contexts (plan §3).
        terciles = []
        if donor_pos.size >= 3:
            for bi, b in enumerate(tercile_bins(lengths, donor_pos)):
                s, _draws = delta_stat(arms, b, idx)
                terciles.append(
                    {
                        "tercile": bi,
                        "median_length": float(np.median(lengths[b])),
                        "contexts": [ctx_ids[i] for i in b],
                        **s,
                    }
                )
        entry["noncollapse_short_cot_terciles"] = terciles
        entry["spearman_gain_vs_length_noncollapse"] = (
            spearman(lengths[donor_pos], gains_by_regime[regime][donor_pos])
            if donor_pos.size >= 3
            else {"note": "too few non-collapse contexts"}
        )
        lmg_blob["by_regime"][regime] = entry
        fam_fig_inputs[regime] = {
            "gains": gains_by_regime[regime].tolist(),
            "lengths": lengths.tolist(),
            "tercile_summaries": terciles,
        }
    dump_json(lmg_blob, out_results / "length_matched_gain.json")

    # ── 3. Δ_fam — the registered family contrast (plan §3) ───────────────────
    fam_blob: dict = {
        "dv": (
            "Δ_fam: mean per-context CoT gain over the ICL+WildChat contexts minus their "
            "greedy NN same-median-CoT-length partners from the other contexts, paired "
            "bootstrap over pairs (plan §3 — family membership is ex-ante, flag-independent)"
        ),
        "collapse_families": list(COLLAPSE_FAMILIES),
        "seed": BOOTSTRAP_SEED,
        "n_boot": args.n_boot,
        "by_regime": {},
        "reproducibility": reproducibility_metadata(),
    }
    if fam_pos.size and donor_pos.size:
        for regime in REGIMES:
            fc = fam_delta_fam(gains_by_regime[regime], lengths, fam_pos, donor_pos, args.n_boot)
            fc["contexts_fam"] = [ctx_ids[i] for i in fam_pos]
            fc["frozen_layer"] = layers[regime]
            fam_blob["by_regime"][regime] = fc
            fam_fig_inputs[regime]["fam_contrast"] = fc
            logger.info(
                "[fam_contrast %s @L%d] Δ_fam=%+.4f ci95=%s (%d pairs)",
                regime,
                layers[regime],
                fc["observed"],
                fc["ci95"],
                fc["n_pairs"],
            )
        make_fam_figure(fam_fig_inputs, ctx_ids, fam_of, out_figures, "fam_contrast_length_matched")
    else:
        fam_blob["note"] = "run subset lacks ICL/WildChat or donor contexts — Δ_fam undefined"

    # ── 4. tri-lineage like-for-like Δ_fam baselines (plan §4.6 F4) ───────────
    if not args.skip_baselines:
        fam_blob["prior_lineage_family_keyed_baselines"] = run_lineage_baselines(args, fam_of)
    dump_json(fam_blob, out_results / "fam_contrast.json")
    logger.info("[phase=f4_done] wrote percontext_deltas / length_matched_gain / fam_contrast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
