#!/usr/bin/env python
# ruff: noqa: RUF002  # em-dash + marker glyph intentional
"""#1333 matched-install breadth re-read — VM-side paired analysis (plan §4e).

Loads (a) this round's ext_bare matched-rung breadth slot reads, (b) the parent
run's villain-persona comparator (``issue1333_marker/breadth/breadth/
mk1_lora_con/slot_reads.json`` @ the pinned parent revision — the doubled
``breadth/breadth/`` prefix is the artifact's REAL path, plan §4e), and (c) the
§4f rig-calibration read, then computes the §3 registered paired statistic:

    Δ = per-(context, question) [bare-matched − villain-persona] leakage ΔG
        (log-prob space) over the shared held-out trio × 20 questions
        (60 pairs), mean-aggregated; 95% CI via a question-level cluster
        bootstrap (20 clusters, aligned indices both arms, 2000 draws,
        seed 653) as ONE vectorized (2000, 60) aligned-index reduction.

Verdict lattice (DISJOINT + exhaustive, plan §3): Survives ⇔ Δ > 0 AND the CI
excludes 0 on the positive side; Dose-artifact ⇔ the CI is wholly below 0;
Inconclusive otherwise. Secondary reads are DESCRIPTIVE only (no lattice).
The §4f calibration offset is REPORTED (caveat flag over 0.5 nat), never
applied as a numerical correction.

Writes ``eval_results/issue_1333/matched-install-breadth-reread/
matched_comparison.json`` + the hero paired-bars figure (+ an exploratory
ladder-overlay figure) under ``figures/issue_1333/``. CPU-only, < 1 s compute.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_REPO = "superkaiba1/explore-persona-space-data"
# The parent run's data-repo upload revision (plan pin 7219f7c03b52, resolved full).
PARENT_RUN_REV = "7219f7c03b529e107aaf4fa548169977403f0131"
# NOTE: the doubled breadth/breadth/ prefix is the artifact's REAL path — do
# not "correct" it (plan §4e; verified by scoped list_repo_tree at plan time).
COMPARATOR_HUB_PATH = "issue1333_marker/breadth/breadth/mk1_lora_con/slot_reads.json"
PARENT_LADDER_HUB_PATH = "issue1333_marker/selection/ext_bare/ladder.json"

# §4f parent reference records (echoed; offsets reported, never applied).
PARENT_SELECTION_DELTA = 5.514121723175049  # selection/mk1_lora_con/selection.json
PARENT_BREADTH_REREAD_DELTA = 5.418  # parent breadth same-rung source re-read (plan §2)
OFFSET_CAVEAT_NATS = 0.5

HELD_OUT_TRIO = ("chef", "hero", "philosopher")
N_QUESTIONS = 20
BOOT_DRAWS = 2000
BOOT_SEED = 653
FOUR_FLOATS = ("logp", "z_marker", "z_eos", "logZ")

OUT_DIR_DEFAULT = REPO_ROOT / "eval_results/issue_1333/matched-install-breadth-reread"
FIG_DIR_DEFAULT = REPO_ROOT / "figures/issue_1333"
_MATCHED_DEFAULTS = (
    REPO_ROOT / "data/issue_1333/matched55/breadth/ext_bare/slot_reads.json",
    OUT_DIR_DEFAULT / "slot_reads.json",
)
_CALIBRATION_DEFAULTS = (
    REPO_ROOT / "data/issue_1333/matched55/calibration/mk1_source_reread.json",
    OUT_DIR_DEFAULT / "mk1_source_reread.json",
)
_MATCHED_LADDER_DEFAULTS = (
    REPO_ROOT / "data/issue_1333/matched55/ext_bare/ladder.json",
    OUT_DIR_DEFAULT / "ladder.json",
)


# ── loading + consumer asserts ────────────────────────────────────────────────


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def assert_slot_reads_rows(
    rec: dict,
    *,
    labels: tuple[str, ...],
    n_questions: int,
    arm: str,
) -> None:
    """Consumer-loader assert (plan §4e): ``per_probe`` rows carry (label +
    context_id, question id, trained/base four-float) with FULL label ×
    question coverage for every requested label. Fails loud on any shape
    mismatch — never a silent partial read."""
    per_probe = rec.get("per_probe")
    if not isinstance(per_probe, list) or not per_probe:
        raise ValueError(f"{arm}: per_probe missing or empty")
    seen: dict[str, set[int]] = {}
    for i, row in enumerate(per_probe):
        meta = row.get("row")
        if not isinstance(meta, dict):
            raise ValueError(f"{arm} per_probe[{i}]: missing 'row' meta dict")
        if "label" not in meta or "q" not in meta or "context_id" not in meta:
            raise ValueError(
                f"{arm} per_probe[{i}]: meta lacks label/q/context_id (got {sorted(meta)})"
            )
        for side in ("trained", "base"):
            d = row.get(side)
            if not isinstance(d, dict):
                raise ValueError(f"{arm} per_probe[{i}]: missing {side!r} stats dict")
            bad = [k for k in FOUR_FLOATS if not isinstance(d.get(k), int | float)]
            if bad:
                raise ValueError(f"{arm} per_probe[{i}].{side}: missing four-float keys {bad}")
        seen.setdefault(str(meta["label"]), set()).add(int(meta["q"]))
    missing = [lab for lab in labels if lab not in seen]
    if missing:
        raise ValueError(f"{arm}: labels {missing} absent from per_probe (have {sorted(seen)})")
    for lab in labels:
        if seen[lab] != set(range(n_questions)):
            raise ValueError(
                f"{arm}: label {lab!r} covers questions {sorted(seen[lab])} != 0..{n_questions - 1}"
            )


def _grid(rec: dict, labels: tuple[str, ...], n_questions: int, fn) -> np.ndarray:
    """(n_labels, n_questions) per-(context, question) reduction of per_probe
    rows via ``fn(trained, base)``; fails loud on any coverage hole."""
    idx = {lab: i for i, lab in enumerate(labels)}
    grid = np.full((len(labels), n_questions), np.nan)
    for row in rec["per_probe"]:
        m = row["row"]
        if m["label"] in idx:
            grid[idx[m["label"]], int(m["q"])] = fn(row["trained"], row["base"])
    if np.isnan(grid).any():
        holes = np.argwhere(np.isnan(grid)).tolist()
        raise ValueError(f"coverage holes after assert (writer bug): {holes[:5]}")
    return grid


def delta_g_grid(rec: dict, labels: tuple[str, ...], n_questions: int) -> np.ndarray:
    """On-policy leakage ΔG = trained.logp − base.logp per (context, question)."""
    return _grid(rec, labels, n_questions, lambda t, b: float(t["logp"]) - float(b["logp"]))


def margin_grid(rec: dict, labels: tuple[str, ...], n_questions: int) -> np.ndarray:
    """EOS-margin space Δ(z_marker − z_eos), trained − base, per (context, q)."""
    return _grid(
        rec,
        labels,
        n_questions,
        lambda t, b: (
            (float(t["z_marker"]) - float(t["z_eos"])) - (float(b["z_marker"]) - float(b["z_eos"]))
        ),
    )


# ── §3 registered paired statistic ───────────────────────────────────────────


def question_cluster_bootstrap(
    paired: np.ndarray, *, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED
) -> tuple[float, float, np.ndarray]:
    """95% CI of mean(paired) via a QUESTION-level cluster bootstrap: one
    (draws, n_ctx*n_q) vectorized aligned-index reduction — an index matrix
    from ``np.random.default_rng(seed).integers``, a ``take`` along the
    question axis (all contexts of a drawn question ride together = cluster
    resampling, aligned across both arms because ``paired`` is already the
    per-pair difference), then ``mean(axis=1)``. NO per-draw Python loop."""
    n_ctx, n_q = paired.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_q, size=(draws, n_q))
    taken = np.take(paired, idx, axis=1)  # (n_ctx, draws, n_q)
    flat = taken.transpose(1, 0, 2).reshape(draws, n_ctx * n_q)  # (draws, 60)
    boot = flat.mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi), boot


def verdict(delta_mean: float, ci_lo: float, ci_hi: float) -> str:
    """DISJOINT + exhaustive lattice (plan §3)."""
    if delta_mean > 0 and ci_lo > 0:
        return "Survives"
    if ci_hi < 0:
        return "Dose-artifact"
    return "Inconclusive"


# ── secondary descriptives (no lattice) ──────────────────────────────────────


def default_context_label(rec: dict) -> str:
    """The default-rendered context's label in a breadth record: the label
    whose context_id is ``bare_default`` (the bare arm's source / a deduped
    default), else ``persona_qwen_default`` (the villain panel's default
    negative). Fails loud when neither is present."""
    for cid in ("bare_default", "persona_qwen_default"):
        for label, v in rec["per_context"].items():
            if v.get("context_id") == cid:
                return label
    raise ValueError(
        "no default-rendered context (bare_default / persona_qwen_default) in per_context: "
        f"{[(k, v.get('context_id')) for k, v in rec['per_context'].items()]}"
    )


def transfer_fractions(
    rec: dict, targets: tuple[str, ...], n_questions: int
) -> tuple[dict[str, float], float]:
    """Margin-space transfer fraction per target: mean Δ(z_marker − z_eos) of
    the target ÷ the SOURCE's — the non-saturating EOS-margin read the
    install-strength rule mandates (never raw log-prob ratios)."""
    grid = margin_grid(rec, ("__source__", *targets), n_questions)
    src = float(grid[0].mean())
    return {t: float(grid[i + 1].mean() / src) for i, t in enumerate(targets)}, src


def calibration_offsets(
    cal_rec: dict, *, comparator_source_delta: float | None = None
) -> tuple[dict[str, float], bool]:
    """§4f cross-run rig offsets: this-run mk1 source read minus the parent's
    two persisted records (both echoed). Returns (offsets, caveat) — caveat
    True when ANY |offset| > 0.5 nat. Reported, never applied (plan §4f)."""
    d = float(cal_rec["delta_logp_mean"])
    offsets = {
        "vs_parent_selection": d - PARENT_SELECTION_DELTA,
        "vs_parent_breadth_reread": d - PARENT_BREADTH_REREAD_DELTA,
    }
    if comparator_source_delta is not None:
        offsets["vs_parent_breadth_reread_exact"] = d - float(comparator_source_delta)
    caveat = any(abs(v) > OFFSET_CAVEAT_NATS for v in offsets.values())
    return offsets, caveat


# ── figures (parent conventions: blog style, per-probe dots, labeled means) ──


def hero_figure(
    matched: dict, comparator: dict, fig_dir: Path, *, trio=HELD_OUT_TRIO, n_q=N_QUESTIONS
) -> None:
    """Hero: paired held-out-context leakage at matched install — bare-matched
    vs villain bars per context, values labeled, per-probe dots beneath."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_blog,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    pal = paper_palette_blog(4)
    grids = {
        "bare-trained (install-matched)": delta_g_grid(matched, trio, n_q),
        "villain-trained (parent)": delta_g_grid(comparator, trio, n_q),
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(trio))
    w = 0.36
    rng = np.random.default_rng(0)
    for k, (name, grid) in enumerate(grids.items()):
        off = (k - 0.5) * 2 * w / 2
        means = grid.mean(axis=1)
        ax.bar(x + off, means, width=w, color=pal[k], alpha=0.85, label=name, zorder=2)
        for i in range(len(trio)):
            jit = rng.uniform(-w * 0.35, w * 0.35, size=n_q)
            ax.scatter(
                np.full(n_q, x[i] + off) + jit,
                grid[i],
                s=10,
                color="0.2",
                alpha=0.55,
                linewidths=0,
                zorder=3,
            )
            ax.text(
                x[i] + off,
                means[i] + 0.25,
                f"{means[i]:+.2f}",
                ha="center",
                fontsize=8.5,
                color="0.1",
                zorder=4,
            )
    ax.axhline(0, color="0.3", lw=1.0)
    ax.set_xticks(x, [t.replace("_", " ") for t in trio])
    ax.set_ylabel("held-out leakage $\\Delta$G, trained - base (nats)")
    ax.set_title("Held-out marker leakage at matched install (+5.5 nats)")
    ax.legend(frameon=False, fontsize=9)
    savefig_paper(fig, "matched_install_breadth_paired", dir=fig_dir)
    plt.close(fig)


def ladder_overlay_figure(
    matched_ladder: dict, parent_ladder: dict, fig_dir: Path, *, window: tuple[float, float]
) -> None:
    """Exploratory: this round's retrain rung reads vs the parent ladder
    (retrain-reproducibility read; plan §6 exploratory dump)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_blog,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    pal = paper_palette_blog(4)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for k, (name, ladder) in enumerate(
        (("matched retrain (this round)", matched_ladder), ("parent ladder", parent_ladder))
    ):
        reads = {int(s): float(v["delta_logp_mean"]) for s, v in ladder["reads_by_step"].items()}
        steps = sorted(reads)
        ax.plot(
            steps,
            [reads[s] for s in steps],
            marker="o",
            ms=4,
            lw=1.4,
            color=pal[k],
            label=name,
        )
    ax.axhspan(window[0], window[1], color="0.85", alpha=0.5, zorder=0)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("source $\\Delta$G, trained - base (nats)")
    ax.set_title("ext_bare retrain ladder vs parent (acceptance window shaded)")
    ax.legend(frameon=False, fontsize=9)
    savefig_paper(fig, "matched_ladder_overlay", dir=fig_dir)
    plt.close(fig)


# ── assembly ──────────────────────────────────────────────────────────────────


def build_comparison(
    matched: dict,
    comparator: dict,
    cal_rec: dict,
    *,
    trio: tuple[str, ...] = HELD_OUT_TRIO,
    n_questions: int = N_QUESTIONS,
    draws: int = BOOT_DRAWS,
    seed: int = BOOT_SEED,
) -> dict:
    """The §3 primary paired read + registered secondaries, as one JSON-ready
    dict (pure; unit-tested on synthetic fixtures)."""
    assert_slot_reads_rows(
        matched, labels=("__source__", *trio), n_questions=n_questions, arm="matched"
    )
    assert_slot_reads_rows(
        comparator, labels=("__source__", *trio), n_questions=n_questions, arm="comparator"
    )
    d_matched = delta_g_grid(matched, trio, n_questions)
    d_villain = delta_g_grid(comparator, trio, n_questions)
    paired = d_matched - d_villain  # aligned per-(context, question) pairs
    delta_mean = float(paired.mean())
    ci_lo, ci_hi, _boot = question_cluster_bootstrap(paired, draws=draws, seed=seed)
    v = verdict(delta_mean, ci_lo, ci_hi)

    per_context = {
        t: {
            "matched_delta_g": float(d_matched[i].mean()),
            "villain_delta_g": float(d_villain[i].mean()),
            "paired_diff_mean": float(paired[i].mean()),
        }
        for i, t in enumerate(trio)
    }
    tf_matched, src_margin_matched = transfer_fractions(matched, trio, n_questions)
    tf_villain, src_margin_villain = transfer_fractions(comparator, trio, n_questions)
    def_m, def_v = default_context_label(matched), default_context_label(comparator)
    offsets, caveat = calibration_offsets(
        cal_rec,
        comparator_source_delta=comparator["per_context"]["__source__"]["delta_logp_mean"],
    )
    out = {
        "round": "matched-install-breadth-reread",
        "primary": {
            "statistic": "paired per-(context,question) [bare-matched - villain] leakage "
            "delta-G (log-prob), held-out trio x 20 questions",
            "n_pairs": int(paired.size),
            "delta_mean": delta_mean,
            "ci95": [ci_lo, ci_hi],
            "verdict": v,
            "bootstrap": {"draws": draws, "seed": seed, "clusters": n_questions},
            "per_context": per_context,
            "matched_source_delta_g": float(
                matched["per_context"]["__source__"]["delta_logp_mean"]
            ),
            "villain_source_delta_g": float(
                comparator["per_context"]["__source__"]["delta_logp_mean"]
            ),
        },
        "secondary_descriptive": {
            "default_rendered_asymmetry": {
                "matched_label": def_m,
                "matched": matched["per_context"][def_m],
                "villain_label": def_v,
                "villain": comparator["per_context"][def_v],
            },
            "margin_transfer_fractions": {
                "matched": {**tf_matched, "source_margin": src_margin_matched},
                "villain": {**tf_villain, "source_margin": src_margin_villain},
            },
            "argmax_emission_rates": {
                "matched": {k: v_["emission_rate"] for k, v_ in matched["per_context"].items()},
                "villain": {k: v_["emission_rate"] for k, v_ in comparator["per_context"].items()},
            },
        },
        "calibration": {
            "this_run_mk1_source_delta_g": float(cal_rec["delta_logp_mean"]),
            "parent_refs": {
                "selection_delta_logp": PARENT_SELECTION_DELTA,
                "breadth_reread_delta_logp": PARENT_BREADTH_REREAD_DELTA,
            },
            "offsets_nats": offsets,
            "caveat_offset_over_0p5_nats": caveat,
            "note": "offset REPORTED only — never applied as a numerical correction "
            "(plan §4f); the paired statistic above is as registered",
        },
    }
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _resolve_default(paths: tuple[Path, ...], flag: str) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"none of the default locations exist for {flag}: "
        f"{[str(p) for p in paths]} — pass {flag} explicitly"
    )


def _fetch_hub(path_in_repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(DATA_REPO, path_in_repo, repo_type="dataset", revision=PARENT_RUN_REV)
    )


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1333 matched-install breadth re-read analysis")
    p.add_argument(
        "--matched",
        default=None,
        help="this round's ext_bare breadth slot_reads.json (default: out-root, then git mirror)",
    )
    p.add_argument(
        "--comparator",
        default=None,
        help=f"villain comparator slot_reads.json (default: fetch {COMPARATOR_HUB_PATH} "
        f"@ {PARENT_RUN_REV[:12]})",
    )
    p.add_argument(
        "--calibration",
        default=None,
        help="§4f mk1_source_reread.json (default: out-root, then git mirror)",
    )
    p.add_argument(
        "--matched-ladder",
        default=None,
        help="this round's ext_bare ladder.json for the overlay (default: out-root/mirror)",
    )
    p.add_argument(
        "--parent-ladder",
        default=None,
        help=f"parent ext_bare ladder.json (default: fetch {PARENT_LADDER_HUB_PATH})",
    )
    p.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    p.add_argument("--fig-dir", default=str(FIG_DIR_DEFAULT))
    p.add_argument(
        "--window",
        default=None,
        help="acceptance window 'lo,hi' for the overlay shading (default 4.5141,6.5141)",
    )
    p.add_argument("--skip-figures", action="store_true", help="JSON only (test/offline shortcut)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    matched_p = (
        Path(args.matched) if args.matched else _resolve_default(_MATCHED_DEFAULTS, "--matched")
    )
    cal_p = (
        Path(args.calibration)
        if args.calibration
        else _resolve_default(_CALIBRATION_DEFAULTS, "--calibration")
    )
    comp_p = Path(args.comparator) if args.comparator else _fetch_hub(COMPARATOR_HUB_PATH)
    matched, comparator, cal_rec = _read_json(matched_p), _read_json(comp_p), _read_json(cal_p)

    out = build_comparison(matched, comparator, cal_rec)
    out["reproducibility"] = {
        "git_commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "numpy_version": np.__version__,
        "inputs": {
            "matched": {"path": str(matched_p), "sha256": _sha256(matched_p)},
            "comparator": {
                "path": str(comp_p),
                "sha256": _sha256(comp_p),
                "hub_path": COMPARATOR_HUB_PATH if args.comparator is None else None,
                "revision": PARENT_RUN_REV if args.comparator is None else None,
            },
            "calibration": {"path": str(cal_p), "sha256": _sha256(cal_p)},
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "matched_comparison.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[matched-reread] wrote {out_path}")
    print(
        f"[matched-reread] delta_mean={out['primary']['delta_mean']:+.3f} "
        f"ci95={out['primary']['ci95']} verdict={out['primary']['verdict']} "
        f"calibration_caveat={out['calibration']['caveat_offset_over_0p5_nats']}"
    )

    if not args.skip_figures:
        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        hero_figure(matched, comparator, fig_dir)
        matched_ladder_p = (
            Path(args.matched_ladder)
            if args.matched_ladder
            else _resolve_default(_MATCHED_LADDER_DEFAULTS, "--matched-ladder")
        )
        parent_ladder_p = (
            Path(args.parent_ladder) if args.parent_ladder else _fetch_hub(PARENT_LADDER_HUB_PATH)
        )
        if args.window:
            lo, hi = (float(t) for t in args.window.split(","))
        else:
            lo, hi = 5.5141 - 1.0, 5.5141 + 1.0
        ladder_overlay_figure(
            _read_json(matched_ladder_p), _read_json(parent_ladder_p), fig_dir, window=(lo, hi)
        )
        print(f"[matched-reread] figures under {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
