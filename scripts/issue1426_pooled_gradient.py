#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (−, ρ, ×) in scientific docstrings + labels.
"""Pooled tri-lineage CoT-length gradient test (#1426 free-analysis follow-up).

The #1426 within-lineage rank read is knife-edge (Spearman ρ_len = +0.347,
p = 0.052, n = 32 non-ICL/WildChat contexts). Pooling the three CoT-decomposition
lineages (#928, #1005, #1426 — identical 50-context battery, 32 non-ICL/WildChat
donors each) triples n, PROVIDED lineage identity is controlled: each lineage has
its own gain scale, so a naive pooled Spearman is confounded by lineage mean
differences.

Registered analysis (per-question ``indiv`` regime only — the profile claim is a
per-question read in all three lineages):

1. Per lineage: per-context (CoT gain, median well-formed CoT length) pairs over
   its non-ICL/WildChat contexts. Gains join ``skill_g_aug`` (h4 rows) −
   ``skill_d_ctx2ans`` (h3 rows) from each lineage's committed
   ``percontext_deltas.json`` (the ``issue1426_f4.lineage_family_baseline`` join);
   lengths come from each lineage's committed figure meta under its EXPLICIT
   per-lineage length key (the metas are key-heterogeneous).
2. Lineage fixed effects: rank-transform gain and length WITHIN each lineage
   stratum (average ranks), pool the 3×32 within-stratum ranks, and take the
   Pearson correlation of the pooled ranks (a stratified Spearman controlling
   lineage identity).
3. Null: stratified permutation — gains permuted WITHIN each lineage stratum,
   10,000 draws, seed 658, batched (one index matrix + one GEMV; no per-draw
   Python loop over rows).
4. CI: stratified bootstrap over contexts within lineage, 2,000 draws, seed 42,
   batched (per-draw within-stratum re-ranking via ``rankdata(axis=1)``).
5. Sanity: the three within-lineage Spearmans are reported; #1426's must
   reproduce its committed ``spearman_gain_vs_length_noncollapse`` (+0.347)
   exactly (atol 1e-9 — a mismatch is a code-drift stop, not a finding). The
   pooled-naive (no fixed effects) Spearman is reported as the confounded
   contrast read.

0 GPU; runs in well under a minute on CPU. Launch with the shared-VM thread-cap
prefix (``OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python ...``); the module also
setdefaults the four thread caps below (MALLOC_ARENA_MAX can only come from the
launch env — glibc reads it at malloc init).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from issue928_common import dump_json, load_json, reproducibility_metadata  # noqa: E402
from issue928_length_matched_gain import spearman  # noqa: E402  (the exact f4 estimator)
from issue928_null_bootstrap import stat_summary  # noqa: E402
from issue1426_common import COLLAPSE_FAMILIES  # noqa: E402
from scipy import stats as sps  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue1426_pooled_gradient")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REGIME = "indiv"  # per-question regime — the profile claim's regime in all three lineages
PERM_SEED, PERM_DRAWS = 658, 10_000
BOOT_SEED, BOOT_DRAWS = 42, 2_000
SANITY_ATOL = 1e-9

# Per-lineage committed sources (the issue1426_f4 tri-lineage specs + this run's
# own committed artifacts; length keys are EXPLICIT because the metas are
# key-heterogeneous — #928 per-regime scatter meta vs the #1005/#1426 fam-contrast
# meta shape).
LINEAGES: tuple[dict, ...] = (
    {
        "name": "issue928",
        "deltas": PROJECT_ROOT / "eval_results" / "issue_928" / "percontext_deltas.json",
        "length_meta": PROJECT_ROOT
        / "figures"
        / "issue_928"
        / "percontext_scatter_indiv.meta.json",
        "length_key": "median CoT length (chars)",
    },
    {
        "name": "issue1005",
        "deltas": PROJECT_ROOT / "eval_results" / "issue_1005" / "percontext_deltas.json",
        "length_meta": PROJECT_ROOT
        / "figures"
        / "issue_1005"
        / "fam_contrast_length_matched.meta.json",
        "length_key": "median well-formed CoT length (chars)",
    },
    {
        "name": "issue1426",
        "deltas": PROJECT_ROOT / "eval_results" / "issue_1426" / "percontext_deltas.json",
        "length_meta": PROJECT_ROOT
        / "figures"
        / "issue_1426"
        / "fam_contrast_length_matched.meta.json",
        "length_key": "median well-formed CoT length (chars)",
    },
)
SANITY_REFS: dict[str, Path] = {
    "issue1426": PROJECT_ROOT / "eval_results" / "issue_1426" / "length_matched_gain.json",
    "issue1005": PROJECT_ROOT / "eval_results" / "issue_1005" / "length_matched_gain.json",
}
I928_CORROBORATION_REF = PROJECT_ROOT / "eval_results" / "issue_928" / "length_matched_gain.json"


def _sha256(path: Path) -> str:
    """Hex sha256 of a file (input provenance for the output JSON)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lineage_gains(blob: dict, regime: str) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """(ctx order, per-context CoT gain g_aug − d_ctx2ans, family-by-ctx) for one lineage.

    The ``issue1426_f4.lineage_family_baseline`` join: ``skill_g_aug`` from the h4
    rows minus ``skill_d_ctx2ans`` from the h3 rows. Where the blob also carries
    the h2 contrast (this run's blob), the join is asserted equal to the committed
    h2 delta (atol 1e-12). Family map may be empty (#928's rows carry none)."""
    h3 = blob["contrasts"]["h3_composed_direct_percontext"]["by_regime"][regime]["per_context"]
    h4 = blob["contrasts"]["h4_sufficiency_percontext"]["by_regime"][regime]["per_context"]
    d_skill = {r["context"]: float(r["skill_d_ctx2ans"]) for r in h3}
    g_skill = {r["context"]: float(r["skill_g_aug"]) for r in h4}
    ctxs = [r["context"] for r in h3]
    assert set(ctxs) == set(g_skill), "h3/h4 context sets drifted in the lineage blob"
    gains = np.asarray([g_skill[c] - d_skill[c] for c in ctxs], np.float64)
    h2 = blob["contrasts"].get("h2_cot_gain_percontext")
    if h2 is not None:
        h2_delta = {r["context"]: float(r["delta"]) for r in h2["by_regime"][regime]["per_context"]}
        dev = max(abs(gains[i] - h2_delta[c]) for i, c in enumerate(ctxs))
        if dev > 1e-12:
            raise AssertionError(f"h4−h3 join diverges from the committed h2 delta by {dev:.3e}")
    fams = {r["context"]: r["family"] for r in h3 if "family" in r}
    return ctxs, gains, fams


def lengths_from_meta(meta_path: Path, length_key: str) -> dict[str, float]:
    """Per-context median CoT length from a committed figure meta.

    The ``issue1426_f4.lineage_family_baseline`` loader logic: per-context lengths
    live in MULTI-member point groups (the scatter series); singleton AGGREGATE
    points (tercile diamonds) can inherit a coinciding context's annotation label
    and would shadow the true value — exclude singleton groups, then require ONE
    unique value per label (fail loud on ambiguity)."""
    meta = load_json(meta_path)
    group_sizes: dict = {}
    for pt in meta["points"]:
        group_sizes[pt.get("_group")] = group_sizes.get(pt.get("_group"), 0) + 1
    vals_by_label: dict[str, set[float]] = {}
    for pt in meta["points"]:
        if "label" in pt and length_key in pt and group_sizes[pt.get("_group")] > 1:
            vals_by_label.setdefault(pt["label"], set()).add(float(pt[length_key]))
    out: dict[str, float] = {}
    for lab, vals in vals_by_label.items():
        if len(vals) != 1:
            raise RuntimeError(
                f"ambiguous per-context length for {lab!r} in {meta_path}: {sorted(vals)}"
            )
        out[lab] = next(iter(vals))
    return out


def pooled_rank_corr(rank_g: np.ndarray, rank_l: np.ndarray) -> float:
    """Pearson correlation of the pooled within-stratum ranks (stratified Spearman)."""
    a = rank_g - rank_g.mean()
    b = rank_l - rank_l.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def stratified_permutation_draws(
    rank_g: np.ndarray, rank_l: np.ndarray, sizes: list[int], n_draws: int, seed: int
) -> np.ndarray:
    """(n_draws,) permutation-null statistics — gains permuted WITHIN each stratum.

    Batched: one (n_draws, N) index matrix (each stratum block independently
    permuted per draw), one fancy-index, one GEMV. Permuting gains within a
    stratum permutes their within-stratum ranks and leaves the pooled mean /
    norm of the rank vector invariant, so only the numerator varies per draw."""
    rng = np.random.default_rng(seed)
    blocks, off = [], 0
    for n_s in sizes:
        base = np.tile(np.arange(n_s), (n_draws, 1))
        blocks.append(rng.permuted(base, axis=1) + off)
        off += n_s
    idx = np.concatenate(blocks, axis=1)
    assert idx.shape == (n_draws, sum(sizes)), idx.shape
    a = rank_g - rank_g.mean()
    b = rank_l - rank_l.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return (a[idx] @ b) / denom


def stratified_bootstrap_draws(
    gains_by_lin: list[np.ndarray], lengths_by_lin: list[np.ndarray], n_draws: int, seed: int
) -> np.ndarray:
    """(n_draws,) bootstrap statistics — contexts resampled WITHIN each lineage stratum.

    Batched: per stratum one (n_draws, n_s) with-replacement index matrix from a
    single seeded generator, per-draw within-stratum re-ranking via
    ``rankdata(axis=1)``, then row-wise pooled Pearson on the concatenated rank
    matrices. A degenerate draw (zero pooled variance) yields NaN and is dropped
    by ``stat_summary``'s finite filter."""
    rng = np.random.default_rng(seed)
    rg_rows, rl_rows = [], []
    for g_s, l_s in zip(gains_by_lin, lengths_by_lin, strict=True):
        idx = rng.integers(0, g_s.size, size=(n_draws, g_s.size))
        rg_rows.append(sps.rankdata(g_s[idx], axis=1))
        rl_rows.append(sps.rankdata(l_s[idx], axis=1))
    rg = np.concatenate(rg_rows, axis=1)
    rl = np.concatenate(rl_rows, axis=1)
    a = rg - rg.mean(axis=1, keepdims=True)
    b = rl - rl.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom < 1e-12, np.nan, (a * b).sum(axis=1) / denom)


def make_figure(per_lin: dict[str, dict], pooled: dict, out_figures: Path, slug: str) -> None:
    """Per-lineage within-rank scatter + pooled least-squares fit on the pooled ranks."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    roles = {"issue928": "accent", "issue1005": "control", "issue1426": "primary"}
    markers = {"issue928": "o", "issue1005": "s", "issue1426": "^"}
    for name, r in per_lin.items():
        ax.scatter(
            r["rank_length"],
            r["rank_gain"],
            s=22,
            marker=markers[name],
            color=paper_palette_role(roles[name]),
            alpha=0.85,
            label=f"#{name.removeprefix('issue')} lineage (n={r['n']}, ρ={r['rho']:+.3f})",
        )
    all_rl = np.concatenate([r["rank_length"] for r in per_lin.values()])
    all_rg = np.concatenate([r["rank_gain"] for r in per_lin.values()])
    slope, intercept = np.polyfit(all_rl, all_rg, 1)
    xs = np.array([all_rl.min(), all_rl.max()])
    ax.plot(
        xs,
        slope * xs + intercept,
        color=paper_palette_role("neutral"),
        lw=1.6,
        label=f"pooled rank fit (ρ={pooled['rho']:+.3f})",
    )
    ax.set_xlabel("within-lineage rank of median well-formed CoT length (1 = shortest of 32)")
    ax.set_ylabel("within-lineage rank of per-context CoT gain (1 = smallest of 32)")
    ax.set_title(
        f"pooled tri-lineage CoT-length gradient (non-ICL/WildChat, {REGIME})\n"
        f"lineage-fixed-effects ρ={pooled['rho']:+.3f} "
        f"CI[{pooled['ci95'][0]:+.3f},{pooled['ci95'][1]:+.3f}], "
        f"perm p={pooled['p_perm_two_sided']:.3f}"
    )
    ax.legend(fontsize=7)
    savefig_paper(fig, f"{out_figures.name}/{slug}", dir=str(out_figures.parent))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pooled tri-lineage CoT-length gradient test (0 GPU, #1426 follow-up)"
    )
    ap.add_argument(
        "--out-json",
        default=str(
            PROJECT_ROOT / "eval_results" / "issue_1426" / "pooled_trilineage_gradient.json"
        ),
    )
    ap.add_argument("--out-figures", default=str(PROJECT_ROOT / "figures" / "issue_1426"))
    ap.add_argument("--perm-draws", type=int, default=PERM_DRAWS)
    ap.add_argument("--boot-draws", type=int, default=BOOT_DRAWS)
    args = ap.parse_args()

    # ── 1. per-lineage (gain, length) pairs over the non-ICL/WildChat contexts ──
    per_lin: dict[str, dict] = {}
    shared_ctx: list[str] | None = None
    shared_fams: dict[str, str] = {}
    for spec in LINEAGES:
        blob = load_json(spec["deltas"])
        ctxs, gains, fams = lineage_gains(blob, REGIME)
        if shared_ctx is None:
            shared_ctx = ctxs
        elif set(ctxs) != set(shared_ctx):
            raise RuntimeError(f"{spec['name']}: context id set drifted from the shared battery")
        for c, f in fams.items():
            if shared_fams.setdefault(c, f) != f:
                raise RuntimeError(
                    f"family drift for {c!r}: {shared_fams[c]} vs {f} ({spec['name']})"
                )
        per_lin[spec["name"]] = {"ctxs": ctxs, "gains": gains, "spec": spec}
    assert shared_ctx is not None
    missing_fam = [c for c in shared_ctx if c not in shared_fams]
    if missing_fam:
        raise RuntimeError(f"no family for {missing_fam[:3]}… in any lineage blob")

    flagged_note: dict[str, int] = {}
    for name, r in per_lin.items():
        spec = r["spec"]
        len_by_ctx = lengths_from_meta(spec["length_meta"], spec["length_key"])
        donor = [c for c in r["ctxs"] if shared_fams[c] not in COLLAPSE_FAMILIES]
        missing = [c for c in donor if c not in len_by_ctx]
        if missing:
            raise RuntimeError(
                f"{name}: length meta lacks {spec['length_key']!r} for {missing[:3]}…"
            )
        pos = {c: i for i, c in enumerate(r["ctxs"])}
        blob = load_json(spec["deltas"])
        h3 = blob["contrasts"]["h3_composed_direct_percontext"]["by_regime"][REGIME]["per_context"]
        flagged = {row["context"] for row in h3 if row.get("flagged")}
        flagged_note[name] = len(flagged & set(donor))
        r["donor_ctxs"] = donor
        r["g"] = np.asarray([r["gains"][pos[c]] for c in donor], np.float64)
        r["l"] = np.asarray([len_by_ctx[c] for c in donor], np.float64)
        r["n"] = len(donor)
        r.update(spearman(r["l"], r["g"]))  # within-lineage ρ (the exact f4 estimator)
        logger.info("[%s] n=%d within-lineage ρ=%+.4f (p=%.4f)", name, r["n"], r["rho"], r["p"])

    # ── 2. sanity gates: within-lineage ρ must reproduce each lineage's committed read ──
    # (#1426 + #1005 carry the same-schema committed spearman_gain_vs_length_noncollapse;
    # #928's committed reads use different subsets — all-50 / unflagged-36 — so it gets a
    # corroboration record, not an exact gate.)
    sanity: dict[str, dict] = {}
    for name, ref_path in SANITY_REFS.items():
        committed = load_json(ref_path)["by_regime"][REGIME]["spearman_gain_vs_length_noncollapse"]
        dev = abs(per_lin[name]["rho"] - float(committed["rho"]))
        if dev > SANITY_ATOL or per_lin[name]["n"] != int(committed["n"]):
            raise RuntimeError(
                f"sanity gate FAILED: reproduced {name} ρ={per_lin[name]['rho']:+.9f} "
                f"(n={per_lin[name]['n']}) vs committed {committed['rho']:+.9f} "
                f"(n={committed['n']}) — code drift in the join, not a finding"
            )
        sanity[name] = {
            "reference": str(ref_path.relative_to(PROJECT_ROOT)),
            "committed_rho": float(committed["rho"]),
            "reproduced_rho": per_lin[name]["rho"],
            "atol": SANITY_ATOL,
            "pass": True,
        }
        logger.info(
            "[sanity] %s within-lineage ρ reproduces committed %+.6f", name, committed["rho"]
        )
    i928_committed = load_json(I928_CORROBORATION_REF)["by_regime"][REGIME][
        "spearman_length_vs_gain"
    ]
    sanity["issue928_corroboration"] = {
        "reference": str(I928_CORROBORATION_REF.relative_to(PROJECT_ROOT)),
        "committed_spearman_length_vs_gain": i928_committed,
        "note": (
            "#928's own committed reads (all-50 pooled / unflagged-36 subsets) corroborate the "
            "NEGATIVE gradient this script computes on its noncollapse-32 subset — no exact-subset "
            "committed anchor exists for #928"
        ),
    }

    # ── 3. pooled stratified Spearman (lineage fixed effects via within-stratum ranks) ──
    names = [s["name"] for s in LINEAGES]
    gains_by_lin = [per_lin[n]["g"] for n in names]
    lengths_by_lin = [per_lin[n]["l"] for n in names]
    sizes = [per_lin[n]["n"] for n in names]
    rank_g = np.concatenate([sps.rankdata(g) for g in gains_by_lin])
    rank_l = np.concatenate([sps.rankdata(ln) for ln in lengths_by_lin])
    for r in per_lin.values():
        r["rank_gain"] = sps.rankdata(r["g"])
        r["rank_length"] = sps.rankdata(r["l"])
    obs = pooled_rank_corr(rank_g, rank_l)

    perm = stratified_permutation_draws(rank_g, rank_l, sizes, args.perm_draws, PERM_SEED)
    p_two = float((1 + np.sum(np.abs(perm) >= abs(obs))) / (perm.size + 1))
    p_pos = float((1 + np.sum(perm >= obs)) / (perm.size + 1))
    boot = stratified_bootstrap_draws(gains_by_lin, lengths_by_lin, args.boot_draws, BOOT_SEED)
    pooled = {
        **stat_summary(obs, boot),
        "rho": obs,
        "n_total": int(sum(sizes)),
        "p_perm_two_sided": p_two,
        "p_perm_one_sided_positive": p_pos,
        "n_perm_draws": int(perm.size),
        "perm_seed": PERM_SEED,
        "boot_seed": BOOT_SEED,
        "mean_within_lineage_rho": float(np.mean([per_lin[n]["rho"] for n in names])),
    }
    naive = spearman(np.concatenate(lengths_by_lin), np.concatenate(gains_by_lin))
    naive["note"] = (
        "naive pooled Spearman WITHOUT lineage fixed effects — confounded by "
        "per-lineage gain/length scale differences; contrast read only"
    )
    logger.info(
        "[pooled] ρ=%+.4f ci95=[%+.4f,%+.4f] perm p(two)=%.4f p(pos)=%.4f | naive ρ=%+.4f",
        obs,
        pooled["ci95"][0],
        pooled["ci95"][1],
        p_two,
        p_pos,
        naive["rho"],
    )

    # ── 4. outputs ────────────────────────────────────────────────────────────
    out_json = Path(args.out_json)
    out_figures = Path(args.out_figures)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    blob = {
        "dv": (
            "Pooled tri-lineage Spearman of per-context CoT gain (skill_g_aug − "
            "skill_d_ctx2ans at each lineage's primary frozen convention, per-question "
            "'indiv' regime) vs median well-formed CoT length (chars), over the pooled "
            "non-ICL/WildChat contexts of #928 + #1005 + #1426, with lineage fixed effects"
        ),
        "estimator": (
            "rank-transform gain and length WITHIN each lineage stratum (scipy rankdata, "
            "average ranks), pool the within-stratum ranks (3 strata), Pearson correlation "
            "on the pooled ranks — a stratified Spearman controlling lineage identity. "
            "Null: gains permuted WITHIN each lineage stratum (batched index matrix). "
            "CI: percentile bootstrap resampling contexts WITHIN each lineage stratum with "
            "per-draw within-stratum re-ranking"
        ),
        "regime": REGIME,
        "collapse_families_excluded": list(COLLAPSE_FAMILIES),
        "per_lineage": {
            n: {
                "n": per_lin[n]["n"],
                "rho": per_lin[n]["rho"],
                "p": per_lin[n]["p"],
                "n_flagged_below_parse_floor_in_set": flagged_note[n],
                "contexts": per_lin[n]["donor_ctxs"],
                "inputs": {
                    "deltas_json": str(per_lin[n]["spec"]["deltas"].relative_to(PROJECT_ROOT)),
                    "deltas_sha256": _sha256(per_lin[n]["spec"]["deltas"]),
                    "length_meta": str(per_lin[n]["spec"]["length_meta"].relative_to(PROJECT_ROOT)),
                    "length_meta_sha256": _sha256(per_lin[n]["spec"]["length_meta"]),
                    "length_key": per_lin[n]["spec"]["length_key"],
                },
            }
            for n in names
        },
        "pooled": pooled,
        "naive_pooled_no_fixed_effects": naive,
        "sanity_gates": sanity,
        "heterogeneity_note": (
            "the three within-lineage gradients DISAGREE IN SIGN — #928 falls steeply with CoT "
            "length (ρ=-0.768 here; its own committed reads: all-50 -0.698, unflagged-36 -0.823) "
            "while #1005 (+0.475) and #1426 (+0.347) rise — so the pooled fixed-effects ρ "
            "(= the mean of the within-lineage ρ's under equal strata, no ties) sits near 0; "
            "the pooled estimand is not a common tri-lineage gradient"
        ),
        "caveat": (
            "the 32 donor contexts are the SAME battery contexts in all three lineages "
            "(shared id set, asserted); the stratified bootstrap/permutation treat "
            "lineage×context cells as independent within stratum, so cross-lineage "
            "context-level dependence is not modeled"
        ),
        "reproducibility": reproducibility_metadata(),
    }
    dump_json(blob, out_json)
    make_figure(per_lin, pooled, out_figures, "pooled_trilineage_gradient")
    logger.info(
        "[phase=done] wrote %s + %s/pooled_trilineage_gradient.{png,pdf}", out_json, out_figures
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
