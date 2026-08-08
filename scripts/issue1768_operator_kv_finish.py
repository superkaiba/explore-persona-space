#!/usr/bin/env python3
"""#1768 operator-SVD key-value read: subspace null, summary merge, figure.

Adds the piece the per-arm pass cannot supply on its own: a RANDOM-SUBSPACE
principal-angle null at the SAME k as each arm's match read. Without it the
match numbers are uninterpretable -- for two random k-dim subspaces of R^n the
principal-angle cosines are already ~sqrt(k/n) (k=11, n=3584 -> ~0.055), which
is the same order as the observed KEY-subspace match, so "0.053" only means
something once the null is on the page.

Then merges an ``operator_kv`` block into summary.json and renders one figure
(spectrum + key/value alignment bars with null bands, grouped by behavior).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1768_map_augmentation as MA  # noqa: E402

RESULTS_DIR = MA.RESULTS_DIR
OPKV_DIR = RESULTS_DIR / "operator_kv"
N_SUBSPACE_NULL = 200
NULL_SEED = 17680


def subspace_null(k: int, n: int, draws: int, rng) -> dict:
    """Principal-angle cosines between two INDEPENDENT random k-dim subspaces.

    The like-for-like null for the match read: same k, same ambient dimension,
    no shared structure. Reported as the distribution of the MEAN cosine (the
    statistic the match read quotes) plus the max.
    """
    means, maxes = [], []
    for _ in range(draws):
        qa, _ = np.linalg.qr(rng.standard_normal((n, k)))
        qb, _ = np.linalg.qr(rng.standard_normal((n, k)))
        s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)
        means.append(float(s.mean()))
        maxes.append(float(s.max()))
    means, maxes = np.array(means), np.array(maxes)
    return {
        "k": int(k),
        "ambient_dim": int(n),
        "n_draws": int(draws),
        "mean_cos_null_mean": float(means.mean()),
        "mean_cos_null_p95": float(np.quantile(means, 0.95)),
        "max_cos_null_mean": float(maxes.mean()),
        "max_cos_null_p95": float(np.quantile(maxes, 0.95)),
        "analytic_scale_sqrt_k_over_n": float(np.sqrt(k / n)),
    }


def build(figs_dir: Path) -> dict:
    recs = {}
    for q in sorted(OPKV_DIR.glob("*.json")):
        d = json.loads(q.read_text())
        recs[(d["arm_id"], int(d["layer"]))] = d  # CELL-keyed: arm alone collides
    assert recs, f"no operator_kv records under {OPKV_DIR}"
    layers_present = sorted({li for _a, li in recs})

    # SCHEMA COMPLETENESS GATE. A cell written by an EARLIER schema version can
    # survive a re-run when the per-arm pass skips it as "present", and would
    # then ride silently into the summary with targets missing.
    required_top = ("keys_side_established_by_assert", "value_target_provenance")
    required_vtargets = ("rB_behavior_readout", "wu_marker_unembedding_row")
    stale = []
    for (arm_, li_), d_ in sorted(recs.items()):
        missing = [k for k in required_top if k not in d_]
        vt_ = d_.get("value_alignment", {}).get("targets", {})
        missing += [f"value_target:{t}" for t in required_vtargets if t not in vt_]
        if missing:
            stale.append(f"{arm_}_L{li_}.json missing {missing}")
    assert not stale, (
        "operator_kv records are SCHEMA-STALE (regenerate with --overwrite; the "
        "per-arm pass skips existing files):\n  " + "\n  ".join(stale)
    )

    # COVERAGE GATE. The schema gate above only inspects records that EXIST, so a
    # pass that silently produced fewer cells (a deleted cell never regenerated)
    # would still build a summary -- just a smaller one. Pin the expected grid.
    n_expected = len(MA.arm_picks()) * len(layers_present)
    assert len(recs) == n_expected, (
        f"operator_kv coverage hole: {len(recs)} cells for "
        f"{len(MA.arm_picks())} arms x {len(layers_present)} layers "
        f"(expected {n_expected}); regenerate the missing cells with --overwrite"
    )

    rng = np.random.default_rng(NULL_SEED)
    null_cache: dict[tuple[int, int], dict] = {}
    for (arm, _li), d in recs.items():
        k = d["match_read"]["k_used"]
        n = d["operator_shape"][0]
        if (k, n) not in null_cache:
            null_cache[(k, n)] = subspace_null(k, n, N_SUBSPACE_NULL, rng)
        d["match_read"]["random_subspace_null"] = null_cache[(k, n)]
        nul = null_cache[(k, n)]
        for side in ("key", "value"):
            obs = d["match_read"][f"{side}_subspace_principal_angles"]["mean_cos"]
            d["match_read"][f"{side}_match_vs_null_ratio"] = (
                obs / nul["mean_cos_null_mean"] if nul["mean_cos_null_mean"] > 0 else float("nan")
            )
            d["match_read"][f"{side}_match_exceeds_null_p95"] = bool(obs > nul["mean_cos_null_p95"])
        MA._atomic_json(OPKV_DIR / f"{arm}_L{d['layer']}.json", d)

    per_cell = {}
    for (arm, li), d in sorted(recs.items()):
        sr = d["spectrum_real"]
        mr = d["match_read"]
        per_cell[f"{arm}|L{li}"] = {
            "arm_id": arm,
            "layer": li,
            "method": d["method"],
            "K_train_pairs": d["K_train_pairs"],
            "kv_orientation_check_rel_err": d["kv_orientation_check_rel_err"],
            "refit_reproduction_m0_r2_absdiff": d["refit_reproduction"]["m0_r2_absdiff"],
            "spectrum": {
                "top1_share": sr["top1_share"],
                "top5_share": sr["top5_share"],
                "participation_ratio_exact": sr["participation_ratio_exact"],
            },
            "key_alignment": {
                n: t.get("topk_subspace_projection")
                for n, t in d["key_alignment"]["targets"].items()
                if t.get("computed")
            },
            "key_alignment_null_p95": d["key_alignment"]["null_abscos_p95"],
            "value_alignment": {
                n: t.get("topk_subspace_projection")
                for n, t in d["value_alignment"]["targets"].items()
                if t.get("computed")
            },
            "value_alignment_null_p95": d["value_alignment"]["null_abscos_p95"],
            "value_alignment_not_computed": [
                n for n, t in d["value_alignment"]["targets"].items() if not t.get("computed")
            ],
            "match_read": {
                "k_used": mr["k_used"],
                "key_mean_cos": mr["key_subspace_principal_angles"]["mean_cos"],
                "value_mean_cos": mr["value_subspace_principal_angles"]["mean_cos"],
                "random_subspace_null_mean_cos": mr["random_subspace_null"]["mean_cos_null_mean"],
                "random_subspace_null_p95": mr["random_subspace_null"]["mean_cos_null_p95"],
                "key_vs_null_ratio": mr["key_match_vs_null_ratio"],
                "value_vs_null_ratio": mr["value_match_vs_null_ratio"],
                "key_exceeds_null_p95": mr["key_match_exceeds_null_p95"],
                "value_exceeds_null_p95": mr["value_match_exceeds_null_p95"],
                "at_mass": mr["at_mass"],
            },
        }

    block = {
        "question": (
            "Is the realized post-FT operator update a low-rank key-value write, and "
            "are its keys/values the ones the data-augmented refit predicts?"
        ),
        "pooling": next(iter(recs.values()))["pooling"],
        "poolings_covered": ["last_prompt (last-token, the round's PRIMARY pooling)"],
        "poolings_not_covered": [
            "span-mean — a second ~30 min VM-side refit pass; NOT run in this round"
        ],
        "object": "dM_real = M+ - M0, the fitted ridge OPERATORS (not the data-space shift)",
        "kv_convention": (
            "raw operator A acts on ROW vectors (v = c @ A), so in A = P S Q^T the LEFT "
            "vectors P are the context-side KEYS and the RIGHT vectors Q the answer-side "
            "VALUES; verified per arm by key @ A ~= sigma * value (rel err ~1e-15)"
        ),
        "layers_covered": layers_present,
        "headline_layer": MA.HEADLINE_LAYER,
        "n_cells": len(per_cell),
        "per_arm_headline_layer": {
            v["arm_id"]: v for v in per_cell.values() if v["layer"] == MA.HEADLINE_LAYER
        },
        "per_cell": per_cell,
        **MA._meta(),
    }
    summ_path = RESULTS_DIR / "summary.json"
    summ = json.loads(summ_path.read_text())
    summ["operator_kv"] = block
    MA._atomic_json(summ_path, summ)

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    per_arm = {v["arm_id"]: v for v in per_cell.values() if v["layer"] == MA.HEADLINE_LAYER}
    assert per_arm, f"no cells at the headline layer L{MA.HEADLINE_LAYER}"
    arms = sorted(per_arm)
    beh = {a: a.split("-")[0] for a in arms}
    pal = paper_palette(len(sorted(set(beh.values()))))
    bcol = {b: pal[i] for i, b in enumerate(sorted(set(beh.values())))}
    short = [a.replace("-lr", "\nlr").replace("-s", " s") for a in arms]
    x = np.arange(len(arms))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

    # (1) spectrum concentration
    ax = axes[0]
    ax.bar(
        x - 0.2,
        [per_arm[a]["spectrum"]["top1_share"] for a in arms],
        0.38,
        label="top-1 share",
        color=[bcol[beh[a]] for a in arms],
        alpha=0.55,
    )
    ax.bar(
        x + 0.2,
        [per_arm[a]["spectrum"]["top5_share"] for a in arms],
        0.38,
        label="top-5 share",
        color=[bcol[beh[a]] for a in arms],
    )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        [per_arm[a]["spectrum"]["participation_ratio_exact"] for a in arms],
        "k.-",
        ms=9,
        lw=1.2,
        label="participation ratio (right axis)",
    )
    ax2.set_ylabel("participation ratio (effective # directions)")
    ax2.set_ylim(0, None)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=5.6, rotation=30, ha="right")
    ax.set_ylabel("share of squared singular mass")
    ax.set_title("Realized operator update is NOT low-rank", fontsize=9.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left")

    # (2) key vs value alignment to named targets (top-5 subspace projection)
    ax = axes[1]
    ktargs = [
        "training_context_centroid",
        "whitened_gate_Sigma_inv_c_src",
        "ridge_natural_key_from_augmentation",
    ]
    vtargs = [
        "delta_eq_mean_map_residual",
        "map_residual_pc1",
        "mean_measured_write_wtf",
        "rB_behavior_readout",
    ]
    labels = [
        "key: train-ctx centroid",
        "key: whitened gate",
        "key: ridge-natural key",
        "value: $\\delta$ = mean map residual",
        "value: residual PC1",
        "value: mean measured write",
        "value: $r_B$ behaviour read-out",
    ]
    vals = [np.mean([per_arm[a]["key_alignment"].get(t, np.nan) for a in arms]) for t in ktargs]
    vals += [np.mean([per_arm[a]["value_alignment"].get(t, np.nan) for a in arms]) for t in vtargs]
    errs = [np.std([per_arm[a]["key_alignment"].get(t, np.nan) for a in arms]) for t in ktargs]
    errs += [np.std([per_arm[a]["value_alignment"].get(t, np.nan) for a in arms]) for t in vtargs]
    n_bars = len(labels)
    cols = ["#b0b0b0"] * 3 + [bcol[sorted(bcol)[0]]] * (n_bars - 3)
    ax.barh(
        np.arange(n_bars),
        vals,
        xerr=[np.maximum(0, errs), np.maximum(0, errs)],
        color=cols,
        capsize=2.5,
        height=0.62,
    )
    nullp95 = float(np.mean([per_arm[a]["key_alignment_null_p95"] for a in arms]))
    ax.axvline(
        nullp95,
        color="crimson",
        ls="--",
        lw=1.2,
        label=f"random-direction null p95 = {nullp95:.3f}",
    )
    ax.set_yticks(np.arange(n_bars))
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("projection of target onto the top-5 singular subspace")
    ax.set_title("Values track the training-pair residual;\nkeys sit at the null", fontsize=9.5)
    ax.legend(fontsize=6, loc="lower right")

    # (3) the match read vs its random-subspace null
    ax = axes[2]
    km = [per_arm[a]["match_read"]["key_mean_cos"] for a in arms]
    vm = [per_arm[a]["match_read"]["value_mean_cos"] for a in arms]
    ax.bar(x - 0.2, km, 0.38, label="KEY subspace match", color="#b0b0b0")
    ax.bar(x + 0.2, vm, 0.38, label="VALUE subspace match", color=[bcol[beh[a]] for a in arms])
    nl = [per_arm[a]["match_read"]["random_subspace_null_p95"] for a in arms]
    ax.plot(x, nl, "r_", ms=22, mew=2.0, label="random-subspace null p95 (matched $k$)")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=5.6, rotation=30, ha="right")
    ax.set_ylabel("mean principal-angle cosine")
    ax.set_title("Predicted vs realized update:\nvalues partly match, keys do not", fontsize=9.5)
    ax.legend(fontsize=6, loc="upper left")

    fig.suptitle(
        f"Operator-SVD key-value read of the realized map update "
        f"(L{MA.HEADLINE_LAYER}, last-token pooling)",
        fontsize=10.5,
    )
    fig.tight_layout()
    figs_dir.mkdir(parents=True, exist_ok=True)
    out = savefig_paper(fig, "operator_kv_read", dir=figs_dir, formats=("png",))["png"]
    plt.close(fig)
    print(f"[figs] wrote {out}")
    return block


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figs-dir", type=Path, default=MA.FIGS_DIR)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.import_check:
        import matplotlib as _mpl  # noqa: F401

        from explore_persona_space.analysis.paper_plots import (  # noqa: F401
            paper_palette,
            savefig_paper,
            set_paper_style,
        )

        print("import-check ok")
        return 0
    b = build(args.figs_dir)
    print(f"[operator_kv] merged block for {b['n_cells']} cells, layers {b['layers_covered']}")
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
