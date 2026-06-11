# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, Δ, −) in scientific docstrings + labels.
"""Task #571 — off-pod CPU analysis: breadth contrast + verdict lattice.

Reads the 8 four-float JSONs (``eval_results/issue_571/four_float/
{trained,base}_{label}.json``) + ``source_check.json`` and computes the
registered §1 read:

- Primary DV: never-negative (32-persona) mean Δz_EOS per arm; the
  between-arm contrast (broad − narrow), paired at the persona level.
- Aggregation convention (plan §5, pinned): per-persona mean over
  questions -> arm mean over the arm's 2 adapters per persona -> mean
  over personas; sd at ddof=0 (matching the +14.85 / 3.29 registration).
- CI: persona-cluster bootstrap — resample the 32 personas with
  replacement, n=10,000 draws, seed 42, percentile 95%.
- Verdict per the §1 sign-pinned lattice (Confirmed / Falsified /
  Inverted / Partial / Indeterminate) with the manipulation-check
  conjunct (PASS on ALL FOUR adapters + cross-arm source Δz_marker
  asymmetry <= 5 required for ANY affirmative label), the matched-seed
  sign-agreement pre-narration diagnostic, the gauge-invariant
  Δlog P(EOS) companion, and the broad_s42 replication-anchor band
  [+10.8, +18.9].

Outputs ``eval_results/issue_571/breadth_contrast.json`` + figures under
``figures/issue_571/`` (hero: paired persona plot with reference lines
at +13.66 — #560's broad-recipe never-negative mean — and −3.1 — the
4-negative lineage's qualitative anchor).

``--self-test`` proves the lattice / bootstrap / figure paths run with
ZERO GPU: it feeds the committed #560 ``{trained,base}_A2.json`` files
in as ALL FOUR adapters (contrast ≡ 0; the missing manipulation check
caps the verdict at indeterminate — the run must complete, never crash)
and writes to ``eval_results/issue_571/self_test/`` +
``figures/issue_571/self_test/``.

VM-side, CPU-only — runs AFTER upload + pod termination (plan §3.3
Phase 7).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
from issue560_crossrecipe_panel import (  # noqa: E402
    EXPECTED_PROMPT_MATCHES,
    HELD_OUT_35,
)

logger = logging.getLogger("issue571.breadth_analysis")

SCHEMA_VERSION = "issue571_breadth_contrast_v1"
DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results/issue_571"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures/issue_571"
SELF_TEST_FF = PROJECT_ROOT / "eval_results/issue_560/four_float"
GEOMETRY_560 = PROJECT_ROOT / "eval_results/issue_560/geometry/context_persona_geometry.json"
TRAIN_DIAG_DIR = PROJECT_ROOT / "eval_results/issue_571/train_diag"

ARMS: dict[str, list[str]] = {
    "broad": ["broad_s42", "broad_s43"],
    "narrow": ["narrow_s42", "narrow_s43"],
}
ALL_LABELS = [label for labels in ARMS.values() for label in labels]
SEED_PAIRS = {"42": ("broad_s42", "narrow_s42"), "43": ("broad_s43", "narrow_s43")}

NEVER_NEG = [p for p in HELD_OUT_35 if p not in EXPECTED_PROMPT_MATCHES]
TRAINED_NEG = sorted(EXPECTED_PROMPT_MATCHES)  # assistant, comedian, villain

# Registered thresholds + reference values (plan §1 / §7).
ANCHOR_BAND = (10.8, 18.9)  # ±4 of #560's measured A2 never-neg mean +14.85
CONFIRM_POINT_MIN = 3.0
NARROW_CONFIRM_MAX = 7.0
ASYMMETRY_CAP = 5.0
REF_560_NEVER_NEG = 13.66
REF_560_TRAINED_NEG = 12.46
REF_4NEG_ANCHOR = -3.1
REF_A2_ANCHOR = 14.85

METRICS = ("dz_eos", "dz_marker", "dmargin", "dlogp_eos", "dlogp_marker", "dp_marker")


def _git_commit() -> str:
    """Short git commit hash of the repo this script runs from."""
    try:
        return (
            subprocess.run(  # epm-lint: subprocess-env-inherit -- git metadata probe, no creds
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_side(ff_dir: Path, label: str, side: str, *, self_test: bool) -> dict:
    """Load one four-float JSON; in self-test mode every label maps to #560 A2."""
    path = SELF_TEST_FF / f"{side}_A2.json" if self_test else ff_dir / f"{side}_{label}.json"
    if not path.exists():
        raise FileNotFoundError(f"four-float file missing: {path}")
    payload = json.loads(path.read_text())
    assert payload.get("side") == side, (path, payload.get("side"))
    return payload


def compute_label_metrics(
    trained: dict, base: dict, personas: list[str]
) -> tuple[dict[str, dict[str, float]], dict]:
    """Per-persona question-mean deltas for one label, plus slot diagnostics.

    Asserts the trained/base sides are slot-matched per (persona, q)
    (same slot_kind + same truncation — the parity invariant the Δ
    readout depends on). Returns ``({metric: {persona: value}}, counts)``.
    """
    t_pp, b_pp = trained["per_persona"], base["per_persona"]
    out: dict[str, dict[str, float]] = {m: {} for m in METRICS}
    counts = {"n_pre_marker_slots": 0, "n_slots": 0, "n_gen_truncated": 0}
    for p in personas:
        t_q, b_q = t_pp[p]["per_q"], b_pp[p]["per_q"]
        assert len(t_q) == len(b_q) and len(t_q) > 0, (p, len(t_q), len(b_q))
        for i, (tq, bq) in enumerate(zip(t_q, b_q, strict=True)):
            assert tq["slot_kind"] == bq["slot_kind"], (p, i, tq["slot_kind"], bq["slot_kind"])
            assert tq["n_truncated_tokens"] == bq["n_truncated_tokens"], (p, i)
        counts["n_slots"] += len(t_q)
        counts["n_pre_marker_slots"] += sum(r["slot_kind"] == "pre_marker" for r in t_q)
        counts["n_gen_truncated"] += sum(bool(r.get("gen_truncated")) for r in t_q)
        dz_eos = [t["z_eos"] - b["z_eos"] for t, b in zip(t_q, b_q, strict=True)]
        dz_marker = [t["z_marker"] - b["z_marker"] for t, b in zip(t_q, b_q, strict=True)]
        dmargin = [
            (t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"])
            for t, b in zip(t_q, b_q, strict=True)
        ]
        dlogp_eos = [
            (t["z_eos"] - t["logZ"]) - (b["z_eos"] - b["logZ"])
            for t, b in zip(t_q, b_q, strict=True)
        ]
        dlogp_marker = [t["logp_marker"] - b["logp_marker"] for t, b in zip(t_q, b_q, strict=True)]
        # Probability-space sanity read: ΔP = P_base * (e^{Δlog P} − 1).
        dp_marker = [
            float(np.exp(b["logp_marker"]) * (np.exp(t["logp_marker"] - b["logp_marker"]) - 1.0))
            for t, b in zip(t_q, b_q, strict=True)
        ]
        for metric, vals in (
            ("dz_eos", dz_eos),
            ("dz_marker", dz_marker),
            ("dmargin", dmargin),
            ("dlogp_eos", dlogp_eos),
            ("dlogp_marker", dlogp_marker),
            ("dp_marker", dp_marker),
        ):
            out[metric][p] = float(np.mean(vals))
    return out, counts


def bootstrap_ci(
    values: np.ndarray, n_boot: int, seed: int, *, ci: float = 95.0
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean, resampling clusters (rows) of ``values``."""
    rng = np.random.default_rng(seed)
    n = len(values)
    assert n >= 2, n
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def classify_bin(point: float, lo: float, hi: float) -> str:
    """positive_excl0 | negative_excl0 | contains0 for one contrast CI."""
    if lo > 0:
        return "positive_excl0"
    if hi < 0:
        return "negative_excl0"
    return "contains0"


def raw_lattice(point: float, lo: float, hi: float, broad_mean: float, narrow_mean: float) -> str:
    """The §1 sign-pinned lattice BEFORE caps (one DV, jointly satisfiable)."""
    in_band = ANCHOR_BAND[0] <= broad_mean <= ANCHOR_BAND[1] and (
        ANCHOR_BAND[0] <= narrow_mean <= ANCHOR_BAND[1]
    )
    if lo > 0 and point >= CONFIRM_POINT_MIN and narrow_mean < NARROW_CONFIRM_MAX:
        return "confirmed"
    if lo > 0 and point >= CONFIRM_POINT_MIN:
        return "partial"
    if hi < 0:
        return "inverted"
    if lo <= 0 <= hi and in_band:
        return "falsified"
    return "indeterminate"


def load_manipulation_check(out_dir: Path, *, self_test: bool) -> dict:
    """source_check.json block, or status 'missing' (caps the verdict)."""
    path = out_dir / "source_check.json"
    if self_test or not path.exists():
        return {
            "status": "missing",
            "detail": f"{path} not found" if not self_test else "self-test mode",
        }
    payload = json.loads(path.read_text())
    return {
        "status": payload["manipulation_check"],
        "cross_arm_dz_marker_asymmetry": payload["cross_arm_dz_marker_asymmetry"],
        "per_label": {
            k: {kk: v[kk] for kk in ("emission_on", "emission_off", "verdict", "dz_marker_source")}
            for k, v in payload["per_label"].items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Task #571 off-pod breadth-contrast analysis (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--boot-seed", type=int, default=42)
    ap.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Feed the committed #560 A2 four-float files in as all four adapters "
            "(contrast ≡ 0 -> indeterminate via the manipulation-check cap; proves "
            "the lattice/bootstrap/figure paths run). Writes under self_test/ subdirs."
        ),
    )
    args = ap.parse_args(argv)

    out_dir = args.out_dir / "self_test" if args.self_test else args.out_dir
    fig_dir = args.fig_dir / "self_test" if args.self_test else args.fig_dir
    ff_dir = args.out_dir / "four_float"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-label per-persona metrics ──────────────────────────────────────
    personas = list(HELD_OUT_35)
    per_label: dict[str, dict[str, dict[str, float]]] = {}
    slot_counts: dict[str, dict] = {}
    for label in ALL_LABELS:
        trained = _load_side(ff_dir, label, "trained", self_test=args.self_test)
        base = _load_side(ff_dir, label, "base", self_test=args.self_test)
        per_label[label], slot_counts[label] = compute_label_metrics(trained, base, personas)
    assert len(NEVER_NEG) == 32, len(NEVER_NEG)

    # ── Arm aggregation: persona value = mean over the arm's 2 adapters ───
    def arm_per_persona(metric: str, arm: str, persona_set: list[str]) -> dict[str, float]:
        return {
            p: float(np.mean([per_label[label][metric][p] for label in ARMS[arm]]))
            for p in persona_set
        }

    def stratum_summary(metric: str, arm: str, persona_set: list[str]) -> dict:
        vals = np.array(list(arm_per_persona(metric, arm, persona_set).values()))
        return {
            "mean": float(vals.mean()),
            "sd_ddof0": float(vals.std(ddof=0)),
            "n_personas": len(vals),
        }

    arm_dz_eos = {arm: arm_per_persona("dz_eos", arm, NEVER_NEG) for arm in ARMS}
    arm_summaries = {
        metric: {
            arm: {
                "never_negative": stratum_summary(metric, arm, NEVER_NEG),
                "trained_negative_descriptive": stratum_summary(metric, arm, TRAINED_NEG),
            }
            for arm in ARMS
        }
        for metric in METRICS
    }

    # ── Primary contrast: paired (broad_p − narrow_p), persona bootstrap ──
    contrast_pp = np.array([arm_dz_eos["broad"][p] - arm_dz_eos["narrow"][p] for p in NEVER_NEG])
    point = float(contrast_pp.mean())
    lo, hi = bootstrap_ci(contrast_pp, args.n_boot, args.boot_seed)
    primary_bin = classify_bin(point, lo, hi)
    broad_mean = arm_summaries["dz_eos"]["broad"]["never_negative"]["mean"]
    narrow_mean = arm_summaries["dz_eos"]["narrow"]["never_negative"]["mean"]
    logger.info(
        "primary contrast (broad − narrow, never-neg Δz_EOS): %+.3f CI95 [%.3f, %.3f] (%s); "
        "arm means broad=%+.2f narrow=%+.2f",
        point,
        lo,
        hi,
        primary_bin,
        broad_mean,
        narrow_mean,
    )

    # ── Matched-seed contrasts (descriptive pre-narration diagnostic) ─────
    per_seed = {}
    for seed, (b_label, n_label) in SEED_PAIRS.items():
        diffs = np.array(
            [per_label[b_label]["dz_eos"][p] - per_label[n_label]["dz_eos"][p] for p in NEVER_NEG]
        )
        per_seed[seed] = float(diffs.mean())
    seed_signs_agree = bool(np.sign(per_seed["42"]) == np.sign(per_seed["43"]))

    # ── Δlog P(EOS) companion (gauge-invariant; same machinery) ───────────
    arm_dlogp = {arm: arm_per_persona("dlogp_eos", arm, NEVER_NEG) for arm in ARMS}
    comp_pp = np.array([arm_dlogp["broad"][p] - arm_dlogp["narrow"][p] for p in NEVER_NEG])
    comp_point = float(comp_pp.mean())
    comp_lo, comp_hi = bootstrap_ci(comp_pp, args.n_boot, args.boot_seed)
    comp_bin = classify_bin(comp_point, comp_lo, comp_hi)

    # ── Replication anchor (rig-drift control, plan §7 assert 6) ──────────
    broad_s42_vals = np.array([per_label["broad_s42"]["dz_eos"][p] for p in NEVER_NEG])
    broad_s42_mean = float(broad_s42_vals.mean())
    anchor_within = ANCHOR_BAND[0] <= broad_s42_mean <= ANCHOR_BAND[1]

    # ── Manipulation check + verdict lattice with caps ────────────────────
    manip = load_manipulation_check(args.out_dir, self_test=args.self_test)
    raw_label = raw_lattice(point, lo, hi, broad_mean, narrow_mean)
    caps: list[str] = []
    if manip["status"] != "pass_all":
        caps.append(f"manipulation_check_{manip['status']} (implant-strength-confounded)")
    if raw_label in ("confirmed", "inverted") and not seed_signs_agree:
        caps.append("matched_seed_sign_disagreement")
    if comp_bin != primary_bin:
        caps.append(
            f"companion_dlogp_eos_disagreement (primary={primary_bin}, companion={comp_bin})"
        )
    if raw_label in ("confirmed", "partial") and not anchor_within:
        caps.append(
            f"rig_drift_unresolved (broad_s42 never-neg mean {broad_s42_mean:+.2f} outside "
            f"[{ANCHOR_BAND[0]}, {ANCHOR_BAND[1]}])"
        )
    final_label = raw_label if not caps else "indeterminate"
    logger.info("verdict: raw=%s caps=%s -> final=%s", raw_label, caps, final_label)

    # ── Output JSON ────────────────────────────────────────────────────────
    result = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "task": 571,
            "script": "issue571_breadth_analysis.py",
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "argv": sys.argv[1:],
            "self_test": bool(args.self_test),
        },
        "config": {
            "n_boot": args.n_boot,
            "boot_seed": args.boot_seed,
            "ci": 95.0,
            "anchor_band": list(ANCHOR_BAND),
            "confirm_point_min": CONFIRM_POINT_MIN,
            "narrow_confirm_max": NARROW_CONFIRM_MAX,
            "asymmetry_cap": ASYMMETRY_CAP,
            "aggregation": "per-persona mean over questions -> arm mean over adapters "
            "per persona -> mean over personas; sd ddof=0",
            "never_negative_personas": NEVER_NEG,
            "trained_negative_personas": TRAINED_NEG,
        },
        "references": {
            "ref_560_never_negative_mean_dz_eos": REF_560_NEVER_NEG,
            "ref_560_trained_negative_mean_dz_eos": REF_560_TRAINED_NEG,
            "ref_4negative_lineage_anchor": REF_4NEG_ANCHOR,
            "ref_560_a2_adapter_never_neg_mean": REF_A2_ANCHOR,
        },
        "primary_contrast_dz_eos": {
            "point": point,
            "ci95": [lo, hi],
            "bin": primary_bin,
            "per_persona": {p: float(v) for p, v in zip(NEVER_NEG, contrast_pp, strict=True)},
        },
        "arm_summaries": arm_summaries,
        "per_seed_contrasts": {**per_seed, "signs_agree": seed_signs_agree},
        "companion_dlogp_eos": {"point": comp_point, "ci95": [comp_lo, comp_hi], "bin": comp_bin},
        "replication_anchor": {
            "broad_s42_never_neg_mean_dz_eos": broad_s42_mean,
            "band": list(ANCHOR_BAND),
            "within_band": bool(anchor_within),
        },
        "manipulation_check": manip,
        "verdict": {
            "raw_label": raw_label,
            "caps_applied": caps,
            "final_label": final_label,
        },
        "slot_truncation_counts": slot_counts,
        "per_label_per_persona": per_label,
    }
    out_path = out_dir / "breadth_contrast.json"
    out_path.write_text(json.dumps(result, indent=1))
    logger.info("written: %s", out_path)

    _figures(per_label, arm_dz_eos, arm_dlogp, contrast_pp, fig_dir, args)
    print(
        json.dumps(
            {
                "final_label": final_label,
                "raw_label": raw_label,
                "caps_applied": caps,
                "contrast_point": point,
                "contrast_ci95": [lo, hi],
                "broad_mean": broad_mean,
                "narrow_mean": narrow_mean,
            },
            indent=1,
        )
    )
    return 0


# ── Figures ────────────────────────────────────────────────────────────────


def _paired_plot(ax, arm_pp: dict[str, dict[str, float]], ylabel: str, args) -> None:
    """Two-column paired persona plot (broad vs narrow) with arm-mean CIs."""
    broad_vals = np.array([arm_pp["broad"][p] for p in NEVER_NEG])
    narrow_vals = np.array([arm_pp["narrow"][p] for p in NEVER_NEG])
    for b, n in zip(broad_vals, narrow_vals, strict=True):
        ax.plot([0, 1], [b, n], color="#7f7f7f", alpha=0.35, lw=0.8, zorder=1)
    for x, vals, color in ((0, broad_vals, "#1f77b4"), (1, narrow_vals, "#d62728")):
        m = float(vals.mean())
        ci_lo, ci_hi = bootstrap_ci(vals, args.n_boot, args.boot_seed)
        # Clamp: constant inputs make percentile widths float-epsilon negative.
        yerr = [[max(0.0, m - ci_lo)], [max(0.0, ci_hi - m)]]
        ax.errorbar([x], [m], yerr=yerr, fmt="o", color=color, capsize=4, ms=8, zorder=3, lw=2)
    ax.set_xticks([0, 1], ["broad (15-negative)", "narrow (4-negative)"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylabel)


def _figures(per_label, arm_dz_eos, arm_dlogp, contrast_pp, fig_dir: Path, args) -> None:
    """Hero + exploratory figures (paper-plots style; guarded extras)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Hero — paired persona plot with the two reference lines.
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _paired_plot(ax, arm_dz_eos, "Δz_EOS (trained − base), never-negative persona mean", args)
    ax.axhline(REF_560_NEVER_NEG, color="#1f77b4", ls="--", lw=1.0, alpha=0.7)
    ax.axhline(REF_4NEG_ANCHOR, color="#d62728", ls="--", lw=1.0, alpha=0.7)
    ax.legend(
        handles=[
            plt.Line2D([], [], color="#1f77b4", ls="--", label="broad-recipe reference (+13.66)"),
            plt.Line2D([], [], color="#d62728", ls="--", label="4-negative lineage anchor (−3.1)"),
        ],
        frameon=False,
        loc="best",
    )
    savefig_paper(fig, "breadth_paired_personas", dir=fig_dir)
    plt.close(fig)

    # Per-adapter bars with per-persona scatter.
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    for i, label in enumerate(ALL_LABELS):
        vals = np.array([per_label[label]["dz_eos"][p] for p in NEVER_NEG])
        color = "#1f77b4" if label.startswith("broad") else "#d62728"
        ax.bar(i, float(vals.mean()), width=0.6, color=color, alpha=0.6)
        ax.scatter(
            np.full(len(vals), i) + rng.uniform(-0.15, 0.15, len(vals)),
            vals,
            s=10,
            color="#333333",
            alpha=0.5,
            zorder=3,
        )
    ax.set_xticks(range(len(ALL_LABELS)), ALL_LABELS, rotation=15, ha="right")
    ax.set_ylabel("Δz_EOS (trained − base), never-negative personas")
    savefig_paper(fig, "per_adapter_dz_eos", dir=fig_dir)
    plt.close(fig)

    # Broad-vs-narrow per-persona scatter with identity line.
    fig, ax = plt.subplots(figsize=(6, 6))
    b = np.array([arm_dz_eos["broad"][p] for p in NEVER_NEG])
    n = np.array([arm_dz_eos["narrow"][p] for p in NEVER_NEG])
    ax.scatter(b, n, s=18, color="#1f77b4", alpha=0.7)
    lims = [min(b.min(), n.min()) - 1, max(b.max(), n.max()) + 1]
    ax.plot(lims, lims, color="#7f7f7f", ls="--", lw=0.8)
    ax.set_xlabel("broad-arm Δz_EOS per persona")
    ax.set_ylabel("narrow-arm Δz_EOS per persona")
    savefig_paper(fig, "broad_vs_narrow_dz_eos", dir=fig_dir)
    plt.close(fig)

    # Secondary spaces: marker logit + EOS margin (same paired shape).
    for metric, ylabel, stem in (
        (
            "dz_marker",
            "Δz_marker (trained − base), never-negative persona mean",
            "breadth_paired_dz_marker",
        ),
        (
            "dmargin",
            "Δ(z_marker − z_eos) (trained − base), never-negative persona mean",
            "breadth_paired_margin",
        ),
    ):
        arm_pp = {
            arm: {
                p: float(np.mean([per_label[label][metric][p] for label in ARMS[arm]]))
                for p in NEVER_NEG
            }
            for arm in ARMS
        }
        fig, ax = plt.subplots(figsize=(7, 5.5))
        _paired_plot(ax, arm_pp, ylabel, args)
        savefig_paper(fig, stem, dir=fig_dir)
        plt.close(fig)

    # Raw-logit vs gauge-invariant companion (per-persona contrasts).
    fig, ax = plt.subplots(figsize=(6, 6))
    comp_pp = np.array([arm_dlogp["broad"][p] - arm_dlogp["narrow"][p] for p in NEVER_NEG])
    ax.scatter(contrast_pp, comp_pp, s=18, color="#1f77b4", alpha=0.7)
    ax.axhline(0, color="#7f7f7f", lw=0.8)
    ax.axvline(0, color="#7f7f7f", lw=0.8)
    ax.set_xlabel("per-persona contrast, Δz_EOS (broad − narrow)")
    ax.set_ylabel("per-persona contrast, Δlog P(EOS) (broad − narrow)")
    savefig_paper(fig, "dz_eos_vs_dlogp_eos_contrast", dir=fig_dir)
    plt.close(fig)

    # Guarded extras (production artifacts may be absent in self-test).
    src_check = args.out_dir / "source_check.json"
    if src_check.exists() and not args.self_test:
        payload = json.loads(src_check.read_text())
        labels = [label for label in ALL_LABELS if label in payload["per_label"]]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        on = [payload["per_label"][label]["emission_on"] for label in labels]
        off = [payload["per_label"][label]["emission_off"] for label in labels]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, on, width=0.4, color="#1f77b4", label="adapter ON")
        ax.bar(x + 0.2, off, width=0.4, color="#7f7f7f", label="base OFF")
        ax.axhline(0.8, color="#2ca02c", ls="--", lw=0.8)
        ax.axhline(0.2, color="#d62728", ls="--", lw=0.8)
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_ylabel("source on-policy marker emission rate (20 Q_test)")
        ax.legend(frameon=False)
        savefig_paper(fig, "source_emission_by_adapter", dir=fig_dir)
        plt.close(fig)

    m5_files = sorted(TRAIN_DIAG_DIR.glob("suppression_difficulty_loc_i571_*.json"))
    if m5_files and not args.self_test:
        fig, ax = plt.subplots(figsize=(9, 5))
        for f in m5_files:
            payload = json.loads(f.read_text())
            agg = payload["per_bystander_mean_neg_loss"]
            bystanders = sorted(agg, key=lambda k: agg[k])
            label = f.stem.replace("suppression_difficulty_loc_", "")
            color = "#1f77b4" if "broad" in label else "#d62728"
            ax.plot(
                range(len(bystanders)),
                [agg[k] for k in bystanders],
                marker="o",
                ms=4,
                lw=1,
                color=color,
                alpha=0.7,
                label=label,
            )
        ax.set_xlabel("bystander (sorted by suppression loss within cell)")
        ax.set_ylabel("mean negative-row loss at post-response slot (ep1)")
        ax.legend(frameon=False, fontsize=8)
        savefig_paper(fig, "m5_suppression_difficulty", dir=fig_dir)
        plt.close(fig)

    if GEOMETRY_560.exists():
        geo = json.loads(GEOMETRY_560.read_text())
        min_dist = geo.get("min_dist", {}).get("A2", {})
        matched = [p for p in NEVER_NEG if p in min_dist]
        if len(matched) >= 5:
            fig, ax = plt.subplots(figsize=(7, 5))
            d = np.array([min_dist[p] for p in matched])
            for arm, color in (("broad", "#1f77b4"), ("narrow", "#d62728")):
                vals = np.array([arm_dz_eos[arm][p] for p in matched])
                ax.scatter(d, vals, s=18, color=color, alpha=0.7, label=f"{arm} arm")
            ax.set_xlabel("cosine distance A2 -> persona (layer 20, #560 geometry)")
            ax.set_ylabel("Δz_EOS per persona")
            ax.legend(frameon=False)
            savefig_paper(fig, "distance_vs_dz_eos", dir=fig_dir)
            plt.close(fig)

    logger.info("figures written under %s", fig_dir)


if __name__ == "__main__":
    sys.exit(main())
