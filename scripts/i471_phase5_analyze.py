# ruff: noqa: RUF001, RUF002, RUF003
"""Phase 5 -- analysis + figures for #471 contrastive-negatives experiment.

Plan v1 §6.2 (H1-H5 + persona×R-style cross + selectivity-artifact partialling)
+ §6.3 figures.

Reads per-cell JSON from `eval_results/issue_471/per_cell/G_{adapter}__{shape}.json`
(both i471_* and i465_* adapter ids -- the cross-experiment paired bootstrap
joins on (q, arm)). Local VM, no GPU.

Headline DV is EMISSION RATE at the demo-free-default-helpful-R shape (read c)
-- the primary BEHAVIORAL DV for H1/H2/H4/H5 (matches #465's headline).
KL is the H3 dynamic-range complement; reported in parallel, NEVER as the
sole hero.

Outputs:
  eval_results/issue_471/analysis.json -- roll-up
  figures/issue_471/*.png             -- 10+ figures per §6.3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger("i471.phase5")

PER_CELL_DIR = Path("eval_results/issue_471/per_cell")
OUT_DIR = Path("eval_results/issue_471")
ROUTE_A_DIR = OUT_DIR / "route_a"
PHASE_A_ANCHOR_PATH = ROUTE_A_DIR / "phaseA_anchor.json"
LOCKSTEP_FINDING_PATH = ROUTE_A_DIR / "lockstep_finding.json"
FIGURE_DIR = Path("figures/issue_471")
ROUTE_A_FIGURE_DIR = FIGURE_DIR / "route_a"
BOOTSTRAP_RESAMPLES = 10_000
RNG_SEED = 42


# ── Cell I/O ────────────────────────────────────────────────────────────
def load_cell(adapter_id: str, eval_shape: str) -> dict | None:
    path = PER_CELL_DIR / f"G_{adapter_id}__{eval_shape}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_all() -> dict[str, dict[str, dict]]:
    """Return {adapter_id: {eval_shape: cell_payload}}."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(PER_CELL_DIR.glob("G_*.json")):
        stem = path.stem[2:]  # strip "G_"
        if "__" not in stem:
            continue
        adapter_id, eval_shape = stem.split("__", 1)
        out.setdefault(adapter_id, {})[eval_shape] = json.loads(path.read_text())
    return out


# ── Bootstrap utilities ─────────────────────────────────────────────────
def wilson_ci(n_success: int, n_total: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion."""
    if n_total == 0:
        return (0.0, 0.0)
    p = n_success / n_total
    denom = 1 + z * z / n_total
    center = (p + z * z / (2 * n_total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total))
    return (max(0.0, center - half), min(1.0, center + half))


def paired_bootstrap_ci(
    a: list[float], b: list[float], *, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = RNG_SEED
) -> tuple[float, float, float]:
    """Paired bootstrap CI on mean(a - b) over `n_resamples` resamples.

    Returns (mean_diff, ci_lo, ci_hi) at 95%.
    """
    if len(a) != len(b):
        raise ValueError(f"paired_bootstrap_ci: len mismatch {len(a)} vs {len(b)}")
    if not a:
        return 0.0, 0.0, 0.0
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return float(diff.mean()), lo, hi


# ── H1-H5 + cross-experiment analysis ───────────────────────────────────
def run_analysis(adapters: list[str], conds: list[str]) -> dict:  # noqa: C901
    _ = adapters  # accepted for API; per-shape lookups are keyed off conds
    all_cells = load_all()
    result: dict = {"adapters_present": sorted(all_cells.keys())}

    # H1: cond1 emission @ read c on i471 -- and the disambig triple (g, g').
    h1: dict = {}
    for cond in conds:
        i471 = f"i471_{cond}"
        cells = all_cells.get(i471, {})
        h1[cond] = {}
        for shape in (
            "demo_free_default",
            "no_system_default",
            "paraphrased_helpful_default",
            "demo_free_default_qtrain",
        ):
            c = cells.get(shape)
            if c is None:
                h1[cond][shape] = None
                continue
            n = c["n_probes"]
            n_emit = round(c["emission_rate"] * n)
            ci_lo, ci_hi = wilson_ci(n_emit, n)
            h1[cond][shape] = {
                "n_probes": n,
                "emission_rate": c["emission_rate"],
                "wilson_ci_95": [ci_lo, ci_hi],
                "mean_kl_post_r": c["mean_kl_post_r"],
                "mean_kl_enrichment": c["mean_kl_enrichment"],
                "mean_delta_marker": c["mean_delta_marker"],
            }
    result["H1_disambig_triple"] = h1

    # H2: arm-pair contrasts at read c.
    h2: dict = {}
    if "i471_cond1" in all_cells and "demo_free_default" in all_cells["i471_cond1"]:
        c1 = all_cells["i471_cond1"]["demo_free_default"]
        for cond in conds:
            if cond == "cond1":
                continue
            i471 = f"i471_{cond}"
            other = all_cells.get(i471, {}).get("demo_free_default")
            if other is None:
                h2[cond] = None
                continue
            a_emit = [1.0 if x else 0.0 for x in c1["argmax_marker_per_q"]]
            b_emit = [1.0 if x else 0.0 for x in other["argmax_marker_per_q"]]
            # Pair on shared q (drop unpaired).
            qa = c1["q_used"]
            qb = other["q_used"]
            common = sorted(set(qa) & set(qb))
            ia = {q: i for i, q in enumerate(qa)}
            ib = {q: i for i, q in enumerate(qb)}
            a_paired = [a_emit[ia[q]] for q in common]
            b_paired = [b_emit[ib[q]] for q in common]
            mean_diff, lo, hi = paired_bootstrap_ci(a_paired, b_paired)
            h2[cond] = {
                "vs": "cond1",
                "n_paired": len(common),
                "cond1_emission": float(np.mean(a_paired)),
                "other_emission": float(np.mean(b_paired)),
                "mean_diff_emission": mean_diff,
                "ci_95": [lo, hi],
            }
    result["H2_demo_dose_response_at_read_c"] = h2

    # H3: KL range across 4 #471 arms at read c + interior-slot enrichment.
    h3: dict = {"arms": {}}
    kl_marker_vals: list[float] = []
    kl_enrich_vals: list[float] = []
    for cond in conds:
        c = all_cells.get(f"i471_{cond}", {}).get("demo_free_default")
        if c is None:
            continue
        h3["arms"][cond] = {
            "mean_kl_post_r": c["mean_kl_post_r"],
            "mean_kl_interior": c["mean_kl_interior"],
            "mean_kl_enrichment": c["mean_kl_enrichment"],
        }
        kl_marker_vals.append(c["mean_kl_post_r"])
        kl_enrich_vals.append(c["mean_kl_enrichment"])
    if kl_marker_vals:
        h3["range_marker_slot_kl"] = float(max(kl_marker_vals) - min(kl_marker_vals))
        h3["range_kl_enrichment"] = float(max(kl_enrich_vals) - min(kl_enrich_vals))
        h3["sub_3a_pass"] = h3["range_marker_slot_kl"] >= 0.5
        h3["sub_3b_pass"] = h3["range_kl_enrichment"] >= 0.3
        h3["overall_pass"] = h3["sub_3a_pass"] and h3["sub_3b_pass"]
    result["H3_kl_dynamic_range"] = h3

    # H4: held-out bystander emission per (arm, bystander).
    h4: dict = {}
    for cond in conds:
        i471 = f"i471_{cond}"
        per_b: dict = {}
        bystander_rates: list[float] = []
        for shape, cell in all_cells.get(i471, {}).items():
            if not shape.startswith("bystander_"):
                continue
            bystander = shape[len("bystander_") :]
            n = cell["n_probes"]
            r = cell["emission_rate"]
            ci_lo, ci_hi = wilson_ci(round(r * n), n)
            per_b[bystander] = {
                "n": n,
                "emission_rate": r,
                "wilson_ci_95": [ci_lo, ci_hi],
                "mean_kl_post_r": cell["mean_kl_post_r"],
            }
            bystander_rates.append(r)
        h4[cond] = {
            "per_bystander": per_b,
            "mean_across_bystanders": float(np.mean(bystander_rates)) if bystander_rates else None,
        }
    result["H4_bystander_selectivity"] = h4

    # H5: trained-negative emission per (arm, neg-persona) Q_test + Q_train split.
    h5: dict = {}
    for cond in conds:
        i471 = f"i471_{cond}"
        per_n: dict = {}
        for shape, cell in all_cells.get(i471, {}).items():
            if not shape.startswith("neg_trained_"):
                continue
            per_n[shape] = {
                "n": cell["n_probes"],
                "emission_rate": cell["emission_rate"],
                "mean_kl_post_r": cell["mean_kl_post_r"],
            }
        h5[cond] = per_n
    result["H5_trained_negative_emission"] = h5

    # Persona × R-style 2×2 at demo-free shapes (MUST-FIX 4).
    cross_22: dict = {}
    for cond in conds:
        i471 = f"i471_{cond}"
        cells = all_cells.get(i471, {})
        # Cell c (helpful + helpful-R), c-parity (helpful + villain-R),
        # h (villain + helpful-R), in_trained_shape on cond1 (villain + villain-R).
        cross_22[cond] = {
            "helpful_sys_helpful_R": _summarize(cells.get("demo_free_default")),
            "helpful_sys_villain_R": _summarize(cells.get("demo_free_default_villain_R")),
            "villain_sys_helpful_R": _summarize(cells.get("villain_sys_helpful_R")),
            "villain_sys_villain_R": (
                _summarize(cells.get("in_trained_shape")) if cond == "cond1" else None
            ),
        }
    result["persona_x_rstyle_2x2"] = cross_22

    # Cross-experiment paired bootstrap (i465 vs i471).
    cross_exp: dict = {}
    for cond in conds:
        i471 = all_cells.get(f"i471_{cond}", {}).get("demo_free_default")
        i465 = all_cells.get(f"i465_{cond}", {}).get("demo_free_default")
        if i471 is None or i465 is None:
            continue
        qa, qb = i471["q_used"], i465["q_used"]
        common = sorted(set(qa) & set(qb))
        ia = {q: i for i, q in enumerate(qa)}
        ib = {q: i for i, q in enumerate(qb)}
        e_471 = [1.0 if i471["argmax_marker_per_q"][ia[q]] else 0.0 for q in common]
        e_465 = [1.0 if i465["argmax_marker_per_q"][ib[q]] else 0.0 for q in common]
        kl_471 = [i471["kl_post_r_per_q"][ia[q]] for q in common]
        kl_465 = [i465["kl_post_r_per_q"][ib[q]] for q in common]
        dg_471 = [i471["delta_marker_per_q"][ia[q]] for q in common]
        dg_465 = [i465["delta_marker_per_q"][ib[q]] for q in common]
        em_diff = paired_bootstrap_ci(e_465, e_471)
        kl_diff = paired_bootstrap_ci(kl_465, kl_471)
        dg_diff = paired_bootstrap_ci(dg_465, dg_471)
        cross_exp[cond] = {
            "n_paired": len(common),
            "emission_465_minus_471": {
                "mean_diff": em_diff[0],
                "ci_95": [em_diff[1], em_diff[2]],
            },
            "kl_465_minus_471": {"mean_diff": kl_diff[0], "ci_95": [kl_diff[1], kl_diff[2]]},
            "deltaG_465_minus_471": {
                "mean_diff": dg_diff[0],
                "ci_95": [dg_diff[1], dg_diff[2]],
            },
        }
    result["cross_experiment_paired_bootstrap_at_read_c"] = cross_exp

    # Selectivity-artifact partialling (#383 caveat). Compute partial Spearman
    # ρ(bystander_mean, source_rate | source_rate) -- well, ρ over the residuals.
    # With 4 arms we can only sanity-check the direction; we report source vs
    # bystander side-by-side so the reader can eyeball.
    selectivity: dict = {}
    for cond in conds:
        source = all_cells.get(f"i471_{cond}", {}).get("in_trained_shape")
        h4_bystander_mean = result["H4_bystander_selectivity"][cond]["mean_across_bystanders"]
        selectivity[cond] = {
            "source_rate": source["emission_rate"] if source else None,
            "mean_bystander_rate": h4_bystander_mean,
            "gap_source_minus_bystander": (
                (source["emission_rate"] - h4_bystander_mean)
                if (source and h4_bystander_mean is not None)
                else None
            ),
        }
    result["selectivity_caveat_383"] = selectivity

    return result


def _summarize(cell: dict | None) -> dict | None:
    if cell is None:
        return None
    return {
        "n_probes": cell["n_probes"],
        "emission_rate": cell["emission_rate"],
        "mean_delta_marker": cell["mean_delta_marker"],
        "mean_kl_post_r": cell["mean_kl_post_r"],
        "mean_kl_enrichment": cell["mean_kl_enrichment"],
    }


# ── Figures ─────────────────────────────────────────────────────────────
def _load_phaseA_anchor() -> dict | None:
    """Read phaseA_anchor.json if it exists (route-(a) bodies only)."""
    if PHASE_A_ANCHOR_PATH.exists():
        return json.loads(PHASE_A_ANCHOR_PATH.read_text())
    return None


def _make_route_a_figures(  # noqa: C901
    plt,
    *,
    withneg_adapter: str | None = None,
    posonly_adapter: str | None = None,
) -> list[Path]:
    """Plan v3 §6.3 route-(a) figures (separate folder from the v1/v2 figures).

    Built only when phaseA_anchor.json + route-(a) per-cell JSONs are present.
    Each figure is wrapped in its own try/except so a missing artifact
    degrades that figure to a no-op without killing the rest.

    Args:
        plt: matplotlib pyplot module (passed in to avoid an import here).
        withneg_adapter: explicit cond1_withneg adapter id (e.g.
            ``i471_route_a_cond1_withneg_step45``). When None, derives from
            ``phaseA_anchor.json``: prefer the stepped id when an anchor_step
            is present; fall back to the legacy un-suffixed id otherwise so
            re-reads of pre-stepped per-cell JSONs still work.
        posonly_adapter: explicit cond1_posonly adapter id (e.g.
            ``i471_route_a_cond1_posonly_step38``). Same derivation rules.
    """
    paths: list[Path] = []
    phaseA = _load_phaseA_anchor()
    if phaseA is None:
        logger.info("phaseA_anchor.json absent; skipping route-(a) figures.")
        return paths
    ROUTE_A_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Derive default adapter ids from phaseA_anchor.json when caller didn't
    # pass them. Stepped naming matches what i471_upload_anchor_adapter.py
    # writes + what phaseB_sweep.sh writes natively.
    if withneg_adapter is None:
        anchor_step = phaseA.get("anchor_step")
        withneg_step = (
            anchor_step
            if anchor_step is not None
            else max(
                (r["step"] for r in phaseA.get("withneg_table", {}).get("step_rows", [])),
                default=None,
            )
        )
        withneg_adapter = (
            f"i471_route_a_cond1_withneg_step{withneg_step}"
            if withneg_step is not None
            else "i471_route_a_cond1_withneg"
        )
    if posonly_adapter is None:
        matched_posonly = phaseA.get("matched_posonly_step")
        posonly_adapter = (
            f"i471_route_a_cond1_posonly_step{matched_posonly}"
            if matched_posonly is not None
            else "i471_route_a_cond1_posonly"
        )

    withneg_rows = phaseA.get("withneg_table", {}).get("step_rows", [])
    posonly_rows = phaseA.get("posonly_table", {}).get("step_rows", [])

    # Figure 1: hero_phaseA_trajectory_4shapes_with_vs_posonly.png
    # 2-panel: left=cond1_withneg, right=cond1_posonly. 4 lines per panel
    # (source / default / trained_neg / bystander).
    try:
        if withneg_rows or posonly_rows:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            for ax, rows, label in (
                (axes[0], withneg_rows, "cond1_withneg"),
                (axes[1], posonly_rows, "cond1_posonly"),
            ):
                if not rows:
                    ax.set_title(f"{label} -- no trajectory")
                    continue
                steps = [r["step"] for r in rows]
                for bucket, color in (
                    ("source", "C0"),
                    ("default", "C1"),
                    ("trained_neg", "C2"),
                    ("bystander", "C3"),
                ):
                    ys = [r.get(f"{bucket}_logp_delta") for r in rows]
                    if all(y is None for y in ys):
                        continue
                    ax.plot(steps, ys, marker="o", label=bucket, color=color)
                ax.set_xlabel("optimizer step")
                ax.set_title(label)
                ax.axhline(0, color="k", lw=0.5, alpha=0.3)
                ax.legend(fontsize=8)
            axes[0].set_ylabel("trained - base log P( ※ ) (nats)")
            anchor = phaseA.get("anchor_step")
            if anchor is not None:
                for ax in axes:
                    ax.axvline(anchor, color="grey", ls="--", lw=0.8, alpha=0.6)
            plt.tight_layout()
            p = ROUTE_A_FIGURE_DIR / "hero_phaseA_trajectory_4shapes_with_vs_posonly.png"
            fig.savefig(p, dpi=140)
            plt.close(fig)
            paths.append(p)
    except Exception as e:
        logger.warning("trajectory figure failed: %s", e)

    # Figure 2: hero_source_minus_default_gap_with_vs_posonly.png
    try:
        if withneg_rows or posonly_rows:
            fig, ax = plt.subplots(figsize=(7, 4))
            for rows, label, color in (
                (withneg_rows, "cond1_withneg", "C0"),
                (posonly_rows, "cond1_posonly", "C3"),
            ):
                if not rows:
                    continue
                steps = [r["step"] for r in rows]
                gaps = [r.get("source_minus_default_gap") for r in rows]
                ax.plot(steps, gaps, marker="o", label=label, color=color)
            ax.axhline(3.0, color="grey", ls="--", lw=0.8, label="anchor threshold (+3 nats)")
            ax.set_xlabel("optimizer step")
            ax.set_ylabel("(source - default) log P( ※ ) gap (nats)")
            ax.set_title("Disentanglement: source-vs-default gap vs step")
            ax.legend()
            plt.tight_layout()
            p = ROUTE_A_FIGURE_DIR / "hero_source_minus_default_gap_with_vs_posonly.png"
            fig.savefig(p, dpi=140)
            plt.close(fig)
            paths.append(p)
    except Exception as e:
        logger.warning("gap figure failed: %s", e)

    # Figure 3: hero_emission_demo_free_withneg_vs_posonly.png
    # On-policy ends_with_marker bars at demo_free_default for the route-(a)
    # anchor checkpoints (cond1_withneg_step<W> vs cond1_posonly_step<P>).
    try:
        all_cells = load_all()
        withneg_cell = all_cells.get(withneg_adapter, {}).get("demo_free_default")
        posonly_cell = all_cells.get(posonly_adapter, {}).get("demo_free_default")
        bars = []
        for label, cell in (
            ("cond1_withneg", withneg_cell),
            ("cond1_posonly", posonly_cell),
        ):
            if cell is None:
                continue
            fg = cell.get("free_gen") or {}
            ends_rate = fg.get("trained_ends_with_marker_rate")
            if ends_rate is None:
                continue
            ends_per_q = fg.get("trained_ends_with_marker_per_q") or []
            n_emit = sum(1 for e in ends_per_q if e)
            n_total = len(ends_per_q)
            ci_lo, ci_hi = wilson_ci(n_emit, n_total) if n_total > 0 else (0.0, 0.0)
            bars.append((label, ends_rate, ci_lo, ci_hi))
        if bars:
            fig, ax = plt.subplots(figsize=(5, 4))
            xs = np.arange(len(bars))
            heights = [b[1] for b in bars]
            errs_lo = [b[1] - b[2] for b in bars]
            errs_hi = [b[3] - b[1] for b in bars]
            ax.bar(xs, heights, yerr=[errs_lo, errs_hi], capsize=4, color=["C0", "C3"])
            ax.set_xticks(xs)
            ax.set_xticklabels([b[0] for b in bars])
            ax.set_ylim(0, 1)
            ax.set_ylabel("on-policy ends-with-marker rate @ demo-free helpful default")
            ax.set_title("H_disentangle headline: route-(a) anchor")
            plt.tight_layout()
            p = ROUTE_A_FIGURE_DIR / "hero_emission_demo_free_withneg_vs_posonly.png"
            fig.savefig(p, dpi=140)
            plt.close(fig)
            paths.append(p)
    except Exception as e:
        logger.warning("disentangle emission figure failed: %s", e)

    # Figure 4 (only if H_A1 PASSed): hero_emission_demo_free_465_vs_471route_a.png
    # On-policy ends-with-marker bars: 4 arms x 2 experiments (#465 full vs
    # #471 route-(a) anchor).
    try:
        anchor_step = phaseA.get("anchor_step")
        if anchor_step is not None:
            all_cells = load_all()
            conds_local = ["cond1", "cond2_k0", "cond2_k1", "cond2_k3"]
            i471_rates = []
            i465_rates = []
            for c in conds_local:
                if c == "cond1":
                    # Use the same stepped withneg_adapter id resolved above
                    # so this lookup hits the same per-cell JSON as Figure 3.
                    i471_adapter = withneg_adapter
                else:
                    i471_adapter = f"i471_route_a_{c}_step{anchor_step}"
                i471_cell = all_cells.get(i471_adapter, {}).get("demo_free_default") or {}
                i471_fg = i471_cell.get("free_gen") or {}
                i471_rates.append(i471_fg.get("trained_ends_with_marker_rate"))
                i465_cell = all_cells.get(f"i465_{c}", {}).get("demo_free_default") or {}
                i465_fg = i465_cell.get("free_gen") or {}
                # i465 cells may not have free_gen; fall back to argmax emission.
                if "trained_ends_with_marker_rate" in i465_fg:
                    i465_rates.append(i465_fg["trained_ends_with_marker_rate"])
                else:
                    i465_rates.append(i465_cell.get("emission_rate"))
            fig, ax = plt.subplots(figsize=(8, 4))
            width = 0.35
            x = np.arange(len(conds_local))
            ax.bar(
                x - width / 2,
                [r if r is not None else 0.0 for r in i471_rates],
                width,
                label="#471 route-(a) anchor",
                color="C0",
            )
            ax.bar(
                x + width / 2,
                [r if r is not None else 0.0 for r in i465_rates],
                width,
                label="#465 full budget",
                color="C7",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(conds_local, rotation=20, ha="right")
            ax.set_ylim(0, 1)
            ax.set_ylabel("on-policy ends-with-marker rate @ demo-free helpful default")
            ax.set_title("Cross-experiment: #465 full vs #471 route-(a) anchor (joint effect)")
            ax.legend(fontsize=8)
            plt.tight_layout()
            p = ROUTE_A_FIGURE_DIR / "hero_emission_demo_free_465_vs_471route_a.png"
            fig.savefig(p, dpi=140)
            plt.close(fig)
            paths.append(p)
    except Exception as e:
        logger.warning("cross-experiment route-(a) emission figure failed: %s", e)

    # Figure 5 (only if lockstep): lockstep_finding.png.
    try:
        if phaseA.get("lockstep_in_this_regime"):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            for ax, rows, label in (
                (axes[0], withneg_rows, "cond1_withneg"),
                (axes[1], posonly_rows, "cond1_posonly"),
            ):
                if not rows:
                    continue
                steps = [r["step"] for r in rows]
                for bucket, color in (
                    ("source", "C0"),
                    ("default", "C1"),
                    ("trained_neg", "C2"),
                    ("bystander", "C3"),
                ):
                    ys = [r.get(f"{bucket}_logp_delta") for r in rows]
                    if all(y is None for y in ys):
                        continue
                    ax.plot(steps, ys, marker="o", label=bucket, color=color)
                ax.set_xlabel("optimizer step")
                ax.set_title(f"{label} (lockstep)")
                ax.axhline(0, color="k", lw=0.5, alpha=0.3)
                ax.legend(fontsize=8)
            axes[0].set_ylabel("trained - base log P( ※ ) (nats)")
            plt.tight_layout()
            p = ROUTE_A_FIGURE_DIR / "lockstep_finding.png"
            fig.savefig(p, dpi=140)
            plt.close(fig)
            paths.append(p)
    except Exception as e:
        logger.warning("lockstep figure failed: %s", e)

    return paths


def make_figures(
    analysis: dict,
    conds: list[str],
    *,
    withneg_adapter: str | None = None,
    posonly_adapter: str | None = None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    # Plan v3 route-(a) figures (new folder; older v1/v2 figures continue
    # to land in figures/issue_471/ for analyzer back-compat).
    paths.extend(
        _make_route_a_figures(
            plt,
            withneg_adapter=withneg_adapter,
            posonly_adapter=posonly_adapter,
        )
    )

    # hero_emission_demo_free_465_vs_471: per-cond emission, two experiments.
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    x = np.arange(len(conds))
    e_471 = [
        analysis["H1_disambig_triple"][c]["demo_free_default"]["emission_rate"]
        if analysis["H1_disambig_triple"][c].get("demo_free_default")
        else 0.0
        for c in conds
    ]
    e_465 = []
    for c in conds:
        cell = analysis.get("cross_experiment_paired_bootstrap_at_read_c", {}).get(c)
        # Re-derive from cross_exp via i471's source mean is incomplete here; use 0 if absent.
        e_465.append(0.0 if cell is None else None)  # placeholder
    ax.bar(x - width / 2, e_471, width, label="#471 (with negatives)")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("Argmax emission rate @ read c")
    ax.set_ylim(0, 1)
    ax.set_title("Marker emission at demo-free helpful default")
    ax.legend()
    plt.tight_layout()
    p = FIGURE_DIR / "hero_emission_demo_free_465_vs_471.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # hero_kl_demo_free: per-cond mean KL @ read c.
    fig, ax = plt.subplots(figsize=(7, 4))
    kl = [
        analysis["H1_disambig_triple"][c]["demo_free_default"]["mean_kl_post_r"]
        if analysis["H1_disambig_triple"][c].get("demo_free_default")
        else 0.0
        for c in conds
    ]
    ax.bar(x, kl, color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("Mean KL(trained ‖ base) at post-R slot (nats)")
    ax.set_title("KL dynamic range across arms @ read c (#471)")
    plt.tight_layout()
    p = FIGURE_DIR / "hero_kl_demo_free.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # hero_bystander_matrix: heatmap arms × 5 bystanders.
    fig, ax = plt.subplots(figsize=(8, 4))
    bystanders = sorted(
        {b for c in conds for b in analysis["H4_bystander_selectivity"][c]["per_bystander"]}
    )
    matrix = np.zeros((len(conds), len(bystanders)))
    for i, c in enumerate(conds):
        per_b = analysis["H4_bystander_selectivity"][c]["per_bystander"]
        for j, b in enumerate(bystanders):
            matrix[i, j] = per_b.get(b, {}).get("emission_rate", 0.0)
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(bystanders)))
    ax.set_xticklabels(bystanders, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(conds)))
    ax.set_yticklabels(conds)
    plt.colorbar(im, ax=ax, label="Emission rate")
    ax.set_title("Bystander leakage — arms × held-out personas")
    plt.tight_layout()
    p = FIGURE_DIR / "hero_bystander_matrix.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    return paths


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--conds", nargs="+", default=["cond1", "cond2_k0", "cond2_k1", "cond2_k3"])
    ap.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation (analysis-only).",
    )
    ap.add_argument(
        "--withneg-adapter",
        default=None,
        help="Stepped cond1_withneg adapter id (e.g. i471_route_a_cond1_withneg_step45). "
        "When omitted, derived from eval_results/issue_471/route_a/phaseA_anchor.json.",
    )
    ap.add_argument(
        "--posonly-adapter",
        default=None,
        help="Stepped cond1_posonly adapter id (e.g. i471_route_a_cond1_posonly_step38). "
        "When omitted, derived from phaseA_anchor.json's matched_posonly_step.",
    )
    args = ap.parse_args(argv)

    adapters = [f"i471_{c}" for c in args.conds] + [f"i465_{c}" for c in args.conds]
    analysis = run_analysis(adapters, args.conds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    logger.info("Wrote analysis -> %s", out_path)
    if not args.no_figures:
        paths = make_figures(
            analysis,
            args.conds,
            withneg_adapter=args.withneg_adapter,
            posonly_adapter=args.posonly_adapter,
        )
        logger.info("Wrote %d figures: %s", len(paths), [str(p) for p in paths])


if __name__ == "__main__":
    main()
