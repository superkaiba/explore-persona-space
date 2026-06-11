# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, Δ, −) in scientific docstrings + labels.
"""Task #571 persona-split-composition — Stage-2 off-pod analysis (VM, CPU).

Reads the 12 primary four-float JSONs (``{trained,base}_{split{2,4,8}_s{42,
43}}.json``), ``source_check.json``, and the Phase 0.5 geometry record, and
computes the registered Stage-2 reads (plan v2 §3/§6):

- **Primary (per channel)**: the 8-vs-2 paired persona contrast over the 32
  never-negative personas — clamp = Δz_EOS, hijack = marker emission rate —
  persona-cluster bootstrap (10,000 draws, seed 42, percentile 95% CI). The
  4-arm is a registered monotonicity read, descriptive.
- **REVISED verdict lattice (implemented exactly)**: split-axis-active (CI
  excludes 0 AND |point| > yardstick); split-axis-inert (clamp CI CONTAINS 0
  with |point| ≤ yardstick AND hijack inert-eligible per the saturation
  routing — both arms' rates outside [5%, 95%] ⇒ the rate CI is NOT
  inert-eligible alone, the EOS-margin companion CI must ALSO contain 0);
  underpowered-discrimination (clamp CI excludes 0 but |point| ≤ yardstick →
  indeterminate, NEVER inert); geometry-unidentified; indeterminate
  catch-all. Run-noise yardstick (clamp) = the largest within-arm
  |seed42 − seed43| arm-mean gap in THIS run; hijack materiality floor =
  10 pp. Matched-seed sign agreement is required before any affirmative
  label; manipulation PASS ×6 + max pairwise cross-arm source Δz_marker
  asymmetry ≤ 5 logits cap the affirmative verdicts at indeterminate.
- **Geometry (registered secondary)**: within-arm partial Spearman
  (collinearity gate 0.6 → the §4.1 residualized-Spearman, same Holm slot);
  across-arm Spearman(ΔDV, Δd_nn) for the 8−2 pair; barrier-vs-bubble
  partials on the 8−2 difference field (Holm over the 2 partials per
  channel); the #472 retest at L10 centered (+ raw / L15 / L20 robustness).

``--self-test`` runs with ZERO GPU and no production inputs: (1) lattice
unit cases asserting EVERY branch (active expected/inverted, inert via both
saturation-routing paths, underpowered, manipulation cap); (2) an
end-to-end fixture run on synthesized parent-shaped four-float + geometry +
source-check files, writing to ``.../self_test/``.

Usage (VM, after upload + pod termination):
    uv run python scripts/issue571_psplit_analysis.py
    uv run python scripts/issue571_psplit_analysis.py --self-test
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
from issue571_breadth_analysis import (  # noqa: E402
    METRICS,
    bootstrap_ci,
    classify_bin,
    compute_label_metrics,
)
from issue571_psplit_common import (  # noqa: E402
    GEOMETRY_JSON,
    PSPLIT_ARMS,
    PSPLIT_FIG_DIR,
    PSPLIT_OUT_DIR,
)
from issue571_psplit_stats import (  # noqa: E402
    bootstrap_ci as boot_stat_ci,
)
from issue571_psplit_stats import (  # noqa: E402
    ci_excludes_zero,
    holm,
    partial_spearman,
    perm_p,
    residualized_spearman,
    spearman,
)

logger = logging.getLogger("issue571.psplit_analysis")

SCHEMA_VERSION = "issue571_psplit_contrast_v1"
ARMS: dict[str, list[str]] = {arm: [f"{arm}_s{s}" for s in (42, 43)] for arm in PSPLIT_ARMS}
ALL_LABELS = [label for labels in ARMS.values() for label in labels]
SEED_PAIRS = {"42": ("split8_s42", "split2_s42"), "43": ("split8_s43", "split2_s43")}
NEVER_NEG = [p for p in HELD_OUT_35 if p not in EXPECTED_PROMPT_MATCHES]
CHANNELS = ("clamp", "hijack")

N_BOOT = 10_000
N_PERM = 10_000
BOOT_SEED = 42
ALPHA = 0.05
HIJACK_FLOOR_PP = 0.10  # 10 pp materiality floor (registered judgment value)
SATURATION_BAND = (0.05, 0.95)
COLLINEARITY_GATE = 0.6
ASYMMETRY_CAP = 5.0
PARENT_REFS = {"broad_dz_eos": 15.35, "narrow_dz_eos": 5.93}  # committed reference lines

# Sign convention for geometry family attribution (harmonized with Stage 1's
# `stage1-barrier-sign-routing` fix). Registered signs in LEAKAGE units are
# positive for both families (barrier: leakage rises with d_src | d_nn —
# shell convention; bubble: Δleakage moves WITH Δd_nn | d_src — §6 analysis 3
# "resolved positive ⇒ suppression local to negatives" for marker-direction
# DVs). hijack + margin are marker-direction (dir +1); clamp (Δz_EOS) is the
# suppression push — anti-leakage — so its expected raw-DV sign flips (−1).
CHANNEL_LEAKAGE_DIR = {"clamp": -1, "hijack": +1, "margin": +1}


def _git_commit() -> str:
    """Short git commit of the repo this script runs from."""
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


# ── Verdict lattice (pure functions — unit-smoked by --self-test) ──────────


def classify_clamp(point: float, lo: float, hi: float, yardstick: float, seeds_agree: bool) -> dict:
    """Clamp-channel lattice cell BEFORE the manipulation cap (§3, revised).

    The ``underpowered_discrimination`` label is a FINAL label that maps to
    the plan's indeterminate route: consumers treat it as indeterminate
    (``apply_caps`` excludes it from the affirmative set; the joint-inert
    read requires ``split_axis_inert`` exactly) — it is never inert and
    never affirmative.
    """
    excl = ci_excludes_zero(lo, hi)
    if excl and abs(point) > yardstick and seeds_agree:
        return {
            "label": "split_axis_active",
            "direction": "expected" if point > 0 else "inverted",
            "reason": f"CI excludes 0 and |point| {abs(point):.3g} > yardstick {yardstick:.3g}",
        }
    if excl and abs(point) <= yardstick:
        return {
            "label": "underpowered_discrimination",
            "direction": None,
            "reason": (
                f"CI excludes 0 but |point| {abs(point):.3g} <= seed-pair yardstick "
                f"{yardstick:.3g} — nonzero yet indistinguishable from seed-scale variation; "
                "routes to indeterminate, NEVER inert"
            ),
        }
    if excl and not seeds_agree:
        return {
            "label": "indeterminate",
            "direction": None,
            "reason": "CI excludes 0 but matched-seed contrasts disagree in sign",
        }
    if not excl and abs(point) <= yardstick:
        return {
            "label": "inert_eligible",
            "direction": None,
            "reason": f"CI contains 0 and |point| <= yardstick {yardstick:.3g}",
        }
    return {
        "label": "indeterminate",
        "direction": None,
        "reason": "CI contains 0 with |point| > yardstick — catch-all",
    }


def classify_hijack(
    point: float,
    lo: float,
    hi: float,
    rate_2arm: float,
    rate_8arm: float,
    margin_ci: tuple[float, float],
    seeds_agree: bool,
) -> dict:
    """Hijack-channel lattice cell with the registered SATURATION ROUTING (§3).

    Both arms' rates outside [5%, 95%] ⇒ the rate proxy has no dynamic range:
    its CI is NOT inert-eligible alone — the inert read then additionally
    requires the non-saturating EOS-margin companion contrast CI to contain 0.
    """
    excl = ci_excludes_zero(lo, hi)
    both_saturated = not (SATURATION_BAND[0] <= rate_2arm <= SATURATION_BAND[1]) and not (
        SATURATION_BAND[0] <= rate_8arm <= SATURATION_BAND[1]
    )
    if excl and abs(point) >= HIJACK_FLOOR_PP and seeds_agree:
        return {
            "label": "split_axis_active",
            "direction": "expected" if point < 0 else "inverted",  # H2b: hijack falls with count
            "both_saturated": both_saturated,
            "reason": f"CI excludes 0 and |point| {abs(point):.3f} >= {HIJACK_FLOOR_PP} (10 pp)",
        }
    if not excl:
        if not both_saturated:
            return {
                "label": "inert_eligible",
                "direction": None,
                "both_saturated": False,
                "reason": "rate CI contains 0 with >= 1 arm inside [5%, 95%] dynamic range",
            }
        margin_contains_0 = not ci_excludes_zero(*margin_ci)
        if margin_contains_0:
            return {
                "label": "inert_eligible",
                "direction": None,
                "both_saturated": True,
                "reason": (
                    "both arms' rates outside [5%, 95%] (rate CI not inert-eligible alone) "
                    "AND the EOS-margin companion CI contains 0"
                ),
            }
        return {
            "label": "indeterminate",
            "direction": None,
            "both_saturated": True,
            "reason": (
                "rate CI contains 0 but both arms saturated AND the EOS-margin companion "
                "CI EXCLUDES 0 — the rate null is a ceiling artifact, not inertness"
            ),
        }
    return {
        "label": "indeterminate",
        "direction": None,
        "both_saturated": both_saturated,
        "reason": "CI excludes 0 below the 10 pp floor, or seed signs disagree — catch-all",
    }


def apply_caps(channel_cls: dict, manipulation: dict) -> dict:
    """Manipulation conjunct (§3): affirmative labels require pass_all + asymmetry <= 5."""
    out = dict(channel_cls)
    affirmative = out["label"] in ("split_axis_active", "inert_eligible")
    manip_ok = manipulation.get("status") == "pass_all"
    if affirmative and not manip_ok:
        out["label_before_cap"] = out["label"]
        out["label"] = "indeterminate"
        out["reason"] += (
            f" [CAPPED: manipulation_check={manipulation.get('status')!r} — affirmative "
            "labels require PASS on all six adapters + asymmetry <= 5 logits]"
        )
    elif out["label"] == "inert_eligible":
        out["label"] = "split_axis_inert"
    return out


# ── Registered statistics ──────────────────────────────────────────────────


def paired_contrast(values_8: np.ndarray, values_2: np.ndarray) -> dict:
    """8−2 paired persona contrast + persona-cluster bootstrap CI."""
    delta = values_8 - values_2
    point = float(delta.mean())
    lo, hi = bootstrap_ci(delta, N_BOOT, BOOT_SEED)
    return {"point": point, "ci95": [lo, hi], "bin": classify_bin(lo, hi), "n": len(delta)}


def _geometry_read(leak: np.ndarray, x: np.ndarray, z: np.ndarray, collinear: bool) -> dict:
    """One within-arm gradient read: partial Spearman, or residualized under the gate."""
    if collinear:
        stat = residualized_spearman(leak, x, z)
        kind = "residualized_spearman"
        fn = lambda idx: residualized_spearman(leak[idx], x[idx], z[idx])  # noqa: E731
    else:
        stat = partial_spearman(leak, x, z)
        kind = "partial_spearman"
        fn = lambda idx: partial_spearman(leak[idx], x[idx], z[idx])  # noqa: E731
    p = perm_p(
        (lambda lv: residualized_spearman(lv, x, z))
        if collinear
        else (lambda lv: partial_spearman(lv, x, z)),
        leak,
        stat,
        n_perm=N_PERM,
        seed=BOOT_SEED,
    )
    lo, hi, dropped = boot_stat_ci(fn, len(leak), n_boot=N_BOOT, seed=BOOT_SEED)
    return {
        "stat": None if not np.isfinite(stat) else float(stat),
        "kind": kind,
        "perm_p": None if not np.isfinite(p) else float(p),
        "ci95": [
            None if not np.isfinite(lo) else float(lo),
            None if not np.isfinite(hi) else float(hi),
        ],
        "ci_excludes_zero": ci_excludes_zero(lo, hi),
        "n_boot_dropped": dropped,
    }


def _family_status(read: dict, holm_p: float | None, expected_sign: int) -> str:
    """Sign-encoded family attribution (mirrors Stage 1's ``_check``).

    'registered' = resolved (Holm p < ALPHA, CI excluding 0) in the family's
    registered direction; 'inverted' = resolved but with the WRONG sign for
    the family (reported descriptively, never family-resolving);
    'unresolved' otherwise. Registered signs in leakage units are positive
    for both families, so the expected raw-DV sign is
    ``CHANNEL_LEAKAGE_DIR[channel]`` for every read.
    """
    resolved = (
        read["stat"] is not None
        and holm_p is not None
        and np.isfinite(holm_p)
        and holm_p < ALPHA
        and read["ci_excludes_zero"]
    )
    if not resolved:
        return "unresolved"
    return "registered" if int(np.sign(read["stat"])) == expected_sign else "inverted"


# ── Input loading ──────────────────────────────────────────────────────────


def load_inputs(out_dir: Path, geometry_path: Path) -> tuple[dict, dict, dict, dict]:
    """(per_label_metrics, hijack_rates, manipulation, geometry) — fail loud."""
    ff_dir = out_dir / "four_float"
    per_label: dict[str, dict[str, dict[str, float]]] = {}
    hijack: dict[str, dict[str, float]] = {}
    for label in ALL_LABELS:
        trained = json.loads((ff_dir / f"trained_{label}.json").read_text())
        base = json.loads((ff_dir / f"base_{label}.json").read_text())
        assert trained.get("side") == "trained" and base.get("side") == "base", label
        metrics, _counts = compute_label_metrics(trained, base, NEVER_NEG)
        per_label[label] = metrics
        hijack[label] = {}
        for p in NEVER_NEG:
            pp = trained["per_persona"][p]
            n_q = len(pp["per_q"])
            hijack[label][p] = float(pp["summary"]["n_pre_marker_slots"]) / n_q

    src_path = out_dir / "source_check.json"
    if src_path.exists():
        src = json.loads(src_path.read_text())
        asym = src["cross_arm_dz_marker_asymmetry"]
        status = src["manipulation_check"]
        if status == "pass_all" and asym is not None and asym > ASYMMETRY_CAP:
            status = "capped"
        manipulation = {
            "status": status,
            "asymmetry": asym,
            "per_label": {
                k: {kk: v[kk] for kk in ("emission_on", "emission_off", "verdict")}
                for k, v in src["per_label"].items()
            },
        }
    else:
        manipulation = {"status": "missing", "asymmetry": None, "per_label": {}}

    geometry = json.loads(geometry_path.read_text())
    assert geometry["bystanders_never_negative"] == NEVER_NEG, "bystander-set drift vs geometry"
    return per_label, hijack, manipulation, geometry


# ── Main analysis ──────────────────────────────────────────────────────────


def run_analysis(out_dir: Path, fig_dir: Path, geometry_path: Path) -> dict:
    """The full registered Stage-2 analysis; returns the output payload."""
    per_label, hijack_rates, manipulation, geometry = load_inputs(out_dir, geometry_path)

    # Channel values: per-persona arm value = mean over the arm's 2 seeds.
    chan_label_pp: dict[str, dict[str, dict[str, float]]] = {
        "clamp": {lb: per_label[lb]["dz_eos"] for lb in ALL_LABELS},
        "hijack": hijack_rates,
        "margin": {lb: per_label[lb]["dmargin"] for lb in ALL_LABELS},
    }
    arm_pp: dict[str, dict[str, dict[str, float]]] = {}
    for ch, by_label in chan_label_pp.items():
        arm_pp[ch] = {
            arm: {p: float(np.mean([by_label[lb][p] for lb in labels])) for p in NEVER_NEG}
            for arm, labels in ARMS.items()
        }

    def vec(ch: str, arm: str) -> np.ndarray:
        return np.array([arm_pp[ch][arm][p] for p in NEVER_NEG])

    # Run-noise yardstick (clamp): largest within-arm seed-pair arm-mean gap.
    seed_gaps = {}
    for arm, labels in ARMS.items():
        m42 = float(np.mean([chan_label_pp["clamp"][labels[0]][p] for p in NEVER_NEG]))
        m43 = float(np.mean([chan_label_pp["clamp"][labels[1]][p] for p in NEVER_NEG]))
        seed_gaps[arm] = abs(m42 - m43)
    clamp_yardstick = max(seed_gaps.values())

    # Primary contrasts (8−2 paired) + matched-seed sign agreement + 4-arm read.
    contrasts: dict[str, dict] = {}
    per_seed: dict[str, dict] = {}
    for ch in ("clamp", "hijack", "margin"):
        contrasts[ch] = paired_contrast(vec(ch, "split8"), vec(ch, "split2"))
        seeds = {}
        for sname, (l8, l2) in SEED_PAIRS.items():
            d = np.array([chan_label_pp[ch][l8][p] - chan_label_pp[ch][l2][p] for p in NEVER_NEG])
            seeds[sname] = float(d.mean())
        seeds["signs_agree"] = bool(np.sign(seeds["42"]) == np.sign(seeds["43"]))
        per_seed[ch] = seeds
    arm_means = {
        ch: {arm: float(vec(ch, arm).mean()) for arm in PSPLIT_ARMS}
        for ch in ("clamp", "hijack", "margin")
    }
    monotonicity = {
        "clamp_rises_with_count": bool(
            arm_means["clamp"]["split2"]
            <= arm_means["clamp"]["split4"]
            <= arm_means["clamp"]["split8"]
        ),
        "hijack_falls_with_count": bool(
            arm_means["hijack"]["split2"]
            >= arm_means["hijack"]["split4"]
            >= arm_means["hijack"]["split8"]
        ),
        "arm_means": arm_means,
    }

    # Lattice (pure functions; caps applied after).
    clamp_cls = apply_caps(
        classify_clamp(
            contrasts["clamp"]["point"],
            *contrasts["clamp"]["ci95"],
            clamp_yardstick,
            per_seed["clamp"]["signs_agree"],
        ),
        manipulation,
    )
    hijack_cls = apply_caps(
        classify_hijack(
            contrasts["hijack"]["point"],
            *contrasts["hijack"]["ci95"],
            arm_means["hijack"]["split2"],
            arm_means["hijack"]["split8"],
            tuple(contrasts["margin"]["ci95"]),
            per_seed["hijack"]["signs_agree"],
        ),
        manipulation,
    )
    # Joint inert (the scope marker's Stage-2 falsification) requires BOTH.
    joint_inert = (
        clamp_cls["label"] == "split_axis_inert" and hijack_cls["label"] == "split_axis_inert"
    )

    # ── Geometry analyses (L10 centered primary; raw + L15/20 robustness) ──
    gate_layer = str(geometry.get("gate_layer", 10))
    geo = geometry["distances"][gate_layer]["centered"]
    d_src = np.array([geo["d_src"][p] for p in NEVER_NEG])
    d_nn = {arm: np.array([geo[f"d_nn_{arm}"][p] for p in NEVER_NEG]) for arm in PSPLIT_ARMS}
    d_dnn = d_nn["split8"] - d_nn["split2"]
    collinearity = {
        arm: float(np.corrcoef(d_nn[arm], d_src)[0, 1])
        if float(np.std(d_nn[arm])) > 1e-12
        else float("nan")
        for arm in PSPLIT_ARMS
    }

    geometry_reads: dict = {"collinearity_pearson_dnn_dsrc": collinearity, "within_arm": {}}
    any_partial_resolved = False
    for ch in CHANNELS:
        geometry_reads["within_arm"][ch] = {}
        for arm in PSPLIT_ARMS:
            leak = vec(ch, arm)
            collinear = bool(
                np.isfinite(collinearity[arm]) and abs(collinearity[arm]) > COLLINEARITY_GATE
            )
            grad = _geometry_read(leak, d_nn[arm], d_src, collinear)
            barrier = _geometry_read(leak, d_src, d_nn[arm], False)
            holm_ps = holm(
                {
                    "gradient": grad["perm_p"] if grad["perm_p"] is not None else float("nan"),
                    "barrier": barrier["perm_p"] if barrier["perm_p"] is not None else float("nan"),
                }
            )
            geometry_reads["within_arm"][ch][arm] = {
                "gradient": grad,
                "barrier": barrier,
                "collinearity_gate_fired": collinear,
                "holm_p": {
                    k: (None if not np.isfinite(v) else float(v)) for k, v in holm_ps.items()
                },
            }

    # Across-arm + barrier-vs-bubble on the 8−2 difference field.
    geometry_reads["difference_field"] = {}
    for ch in ("hijack", "clamp", "margin"):  # hijack primary, margin companion (§6)
        delta = np.array([arm_pp[ch]["split8"][p] - arm_pp[ch]["split2"][p] for p in NEVER_NEG])
        across = {
            "stat": spearman(delta, d_dnn),
            "perm_p": perm_p(
                lambda lv: spearman(lv, d_dnn),
                delta,
                spearman(delta, d_dnn),
                n_perm=N_PERM,
                seed=BOOT_SEED,
            ),
        }
        lo, hi, _ = boot_stat_ci(
            lambda idx, d=delta: spearman(d[idx], d_dnn[idx]),
            len(delta),
            n_boot=N_BOOT,
            seed=BOOT_SEED,
        )
        across["ci95"] = [lo, hi]
        across["ci_excludes_zero"] = ci_excludes_zero(lo, hi)
        bubble = _geometry_read(delta, d_dnn, d_src, False)
        barrier = _geometry_read(delta, d_src, d_dnn, False)
        holm_ps = holm(
            {
                "bubble": bubble["perm_p"] if bubble["perm_p"] is not None else float("nan"),
                "barrier": barrier["perm_p"] if barrier["perm_p"] is not None else float("nan"),
            }
        )
        hp = {k: (None if not np.isfinite(v) else float(v)) for k, v in holm_ps.items()}
        # Sign-encoded family attribution (harmonized with Stage 1): only a
        # REGISTERED-direction resolution identifies the geometry; an
        # inverted-sign resolution is recorded descriptively.
        sign_status = {}
        for name, read in (("bubble", bubble), ("barrier", barrier)):
            status = _family_status(read, hp[name], CHANNEL_LEAKAGE_DIR[ch])
            read["expected_sign"] = CHANNEL_LEAKAGE_DIR[ch]
            read["sign_status"] = status
            sign_status[name] = status
            if ch in CHANNELS and status == "registered":
                any_partial_resolved = True
        geometry_reads["difference_field"][ch] = {
            "across_arm_spearman_vs_ddnn": across,
            "bubble_partial": bubble,
            "barrier_partial": barrier,
            "holm_p": hp,
            "sign_status": sign_status,
        }

    geometry_reads["geometry_unidentified"] = not any_partial_resolved
    geometry_reads["reduced_identification_from_gates"] = bool(
        geometry.get("selection_provenance", {}).get("reduced_identification", False)
    )

    # #472 retest robustness grid.
    retest: dict = {}
    for layer in geometry["distances"]:
        retest[layer] = {}
        for centering, geo_lc in geometry["distances"][layer].items():
            d_src_lc = np.array([geo_lc["d_src"][p] for p in NEVER_NEG])
            retest[layer][centering] = {
                ch: {
                    arm: (
                        None
                        if not np.isfinite(spearman(vec(ch, arm), d_src_lc))
                        else float(spearman(vec(ch, arm), d_src_lc))
                    )
                    for arm in PSPLIT_ARMS
                }
                for ch in CHANNELS
            }

    verdict = {
        "clamp": clamp_cls,
        "hijack": hijack_cls,
        "joint_split_axis_inert": joint_inert,
        "geometry_unidentified": geometry_reads["geometry_unidentified"],
        "manipulation": manipulation,
        "clamp_yardstick_seed_pair_gap": clamp_yardstick,
        "seed_gaps_by_arm": seed_gaps,
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "n_boot": N_BOOT,
            "n_perm": N_PERM,
            "boot_seed": BOOT_SEED,
            "hijack_floor_pp": HIJACK_FLOOR_PP,
            "saturation_band": list(SATURATION_BAND),
            "collinearity_gate": COLLINEARITY_GATE,
            "asymmetry_cap": ASYMMETRY_CAP,
            "aggregation": "per-persona mean over questions -> arm mean over the arm's 2 "
            "seeds per persona -> paired 8-2 contrast over 32 personas",
            "never_negative_personas": NEVER_NEG,
            "parent_reference_lines": PARENT_REFS,
        },
        "primary_contrasts_8v2": contrasts,
        "per_seed_contrasts": per_seed,
        "monotonicity_4arm_descriptive": monotonicity,
        "verdict": verdict,
        "geometry": geometry_reads,
        "retest_472": retest,
        "per_label_per_persona": {lb: {m: per_label[lb][m] for m in METRICS} for lb in ALL_LABELS},
        "hijack_per_label_per_persona": hijack_rates,
        "metadata": {
            "task": 571,
            "followup_label": "persona-split-composition",
            "script": "issue571_psplit_analysis.py",
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "argv": sys.argv[1:],
        },
    }
    out_path = out_dir / "psplit_contrast.json"
    out_path.write_text(json.dumps(payload, indent=1))
    logger.info("Stage-2 contrast JSON written: %s", out_path)
    _figures(fig_dir, arm_pp, chan_label_pp, d_dnn, d_src)
    logger.info(
        "STAGE-2 VERDICT: clamp=%s hijack=%s joint_inert=%s geometry_unidentified=%s",
        clamp_cls["label"],
        hijack_cls["label"],
        joint_inert,
        geometry_reads["geometry_unidentified"],
    )
    return payload


def _figures(fig_dir: Path, arm_pp: dict, chan_label_pp: dict, d_dnn, d_src) -> None:
    """Hero (three-arm paired personas) + companions (§6 figure list)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    xs = {arm: i for i, arm in enumerate(PSPLIT_ARMS)}

    # Hero — three-arm paired persona plot (clamp) + parent reference lines.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for p in NEVER_NEG:
        ax.plot(
            list(xs.values()),
            [arm_pp["clamp"][arm][p] for arm in PSPLIT_ARMS],
            color="#7f7f7f",
            alpha=0.3,
            lw=0.8,
            zorder=1,
        )
    for arm in PSPLIT_ARMS:
        vals = np.array([arm_pp["clamp"][arm][p] for p in NEVER_NEG])
        m = float(vals.mean())
        lo, hi = bootstrap_ci(vals, N_BOOT, BOOT_SEED)
        yerr = [[max(0.0, m - lo)], [max(0.0, hi - m)]]
        ax.errorbar([xs[arm]], [m], yerr=yerr, fmt="o", color="#1f77b4", capsize=4, ms=8, lw=2)
    ax.axhline(PARENT_REFS["broad_dz_eos"], color="#1f77b4", ls="--", lw=1.0, alpha=0.6)
    ax.axhline(PARENT_REFS["narrow_dz_eos"], color="#d62728", ls="--", lw=1.0, alpha=0.6)
    ax.set_xticks(list(xs.values()), [f"{a[5:]}-persona panel" for a in PSPLIT_ARMS])
    ax.set_ylabel("Δz_EOS (trained − base), never-negative persona mean")
    ax.legend(
        handles=[
            plt.Line2D([], [], color="#1f77b4", ls="--", label="parent broad arm (+15.4)"),
            plt.Line2D([], [], color="#d62728", ls="--", label="parent narrow arm (+5.9)"),
        ],
        frameon=False,
        loc="best",
    )
    savefig_paper(fig, "psplit_paired_personas_clamp", dir=fig_dir)
    plt.close(fig)

    # Hijack bars per arm with persona-cluster CIs.
    fig, ax = plt.subplots(figsize=(6.5, 5))
    rng = np.random.default_rng(0)
    for arm in PSPLIT_ARMS:
        vals = np.array([arm_pp["hijack"][arm][p] for p in NEVER_NEG])
        m = float(vals.mean())
        lo, hi = bootstrap_ci(vals, N_BOOT, BOOT_SEED)
        ax.bar(xs[arm], m, width=0.6, color="#1f77b4", alpha=0.6)
        ax.errorbar(
            [xs[arm]],
            [m],
            yerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]],
            fmt="none",
            color="#333333",
            capsize=4,
        )
        ax.scatter(
            np.full(len(vals), xs[arm]) + rng.uniform(-0.15, 0.15, len(vals)),
            vals,
            s=10,
            color="#333333",
            alpha=0.5,
            zorder=3,
        )
    ax.set_xticks(list(xs.values()), [f"{a[5:]}-persona panel" for a in PSPLIT_ARMS])
    ax.set_ylabel("held-out marker hijack rate (fraction of answers with ※)")
    savefig_paper(fig, "psplit_hijack_by_arm", dir=fig_dir)
    plt.close(fig)

    # Δleakage(8−2) vs Δd_nn and vs d_src (both channels).
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, ch, ylab in ((0, "clamp", "Δ(Δz_EOS) 8−2"), (1, "hijack", "Δ(hijack) 8−2")):
        delta = np.array([arm_pp[ch]["split8"][p] - arm_pp[ch]["split2"][p] for p in NEVER_NEG])
        for col, x, xlab in ((0, d_dnn, "Δd_nn (8 − 2), L10 centered"), (1, d_src, "d_src")):
            ax = axes[row][col]
            ax.scatter(x, delta, s=20, color="#1f77b4", alpha=0.75)
            ax.axhline(0, color="#7f7f7f", lw=0.8, ls="--")
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
    savefig_paper(fig, "psplit_difference_field", dir=fig_dir)
    plt.close(fig)

    # Per-seed contrast pairs (clamp).
    fig, ax = plt.subplots(figsize=(6, 5))
    for sname, (l8, l2) in SEED_PAIRS.items():
        d = np.array(
            [chan_label_pp["clamp"][l8][p] - chan_label_pp["clamp"][l2][p] for p in NEVER_NEG]
        )
        ax.scatter([sname] * len(d), d, s=12, alpha=0.5, color="#1f77b4")
        ax.scatter([sname], [float(d.mean())], s=80, color="#d62728", zorder=3)
    ax.axhline(0, color="#7f7f7f", lw=0.8, ls="--")
    ax.set_ylabel("per-persona Δz_EOS contrast (8 − 2), by matched seed")
    savefig_paper(fig, "psplit_per_seed_contrasts", dir=fig_dir)
    plt.close(fig)


# ── Self-test (zero GPU): lattice unit cases + end-to-end fixture ──────────


def _lattice_unit_cases() -> None:
    """Assert every lattice branch incl. the saturation + underpowered routing."""
    pass_all = {"status": "pass_all"}
    warn = {"status": "capped"}
    # 1. clamp active, expected direction.
    c = apply_caps(classify_clamp(8.0, 5.0, 11.0, 4.4, True), pass_all)
    assert c["label"] == "split_axis_active" and c["direction"] == "expected", c
    # 2. clamp active, inverted direction.
    c = apply_caps(classify_clamp(-8.0, -11.0, -5.0, 4.4, True), pass_all)
    assert c["label"] == "split_axis_active" and c["direction"] == "inverted", c
    # 3. underpowered discrimination -> indeterminate, never inert.
    c = apply_caps(classify_clamp(2.0, 0.5, 3.5, 4.4, True), pass_all)
    assert c["label"] == "underpowered_discrimination", c
    # 4. clamp inert eligible -> split_axis_inert after caps.
    c = apply_caps(classify_clamp(0.5, -1.0, 2.0, 4.4, True), pass_all)
    assert c["label"] == "split_axis_inert", c
    # 5. manipulation cap on an affirmative label.
    c = apply_caps(classify_clamp(8.0, 5.0, 11.0, 4.4, True), warn)
    assert c["label"] == "indeterminate" and c["label_before_cap"] == "split_axis_active", c
    # 6. hijack active expected (falls with count).
    h = apply_caps(classify_hijack(-0.4, -0.6, -0.2, 0.9, 0.4, (-1.0, 1.0), True), pass_all)
    assert h["label"] == "split_axis_active" and h["direction"] == "expected", h
    # 7. hijack inert, >= 1 arm in dynamic range (rate CI alone suffices).
    h = apply_caps(classify_hijack(0.01, -0.05, 0.06, 0.5, 0.5, (1.0, 3.0), True), pass_all)
    assert h["label"] == "split_axis_inert" and not h["both_saturated"], h
    # 8. hijack saturation routing: both arms saturated + margin CI contains 0 -> inert.
    h = apply_caps(classify_hijack(0.01, -0.02, 0.04, 0.97, 0.99, (-2.0, 2.0), True), pass_all)
    assert h["label"] == "split_axis_inert" and h["both_saturated"], h
    # 9. saturation routing: both saturated + margin CI EXCLUDES 0 -> indeterminate.
    h = apply_caps(classify_hijack(0.01, -0.02, 0.04, 0.97, 0.99, (1.0, 3.0), True), pass_all)
    assert h["label"] == "indeterminate" and h["both_saturated"], h
    # 10. seed-sign disagreement blocks the affirmative.
    c = apply_caps(classify_clamp(8.0, 5.0, 11.0, 4.4, False), pass_all)
    assert c["label"] == "indeterminate", c
    # 11-14. Sign-encoded family attribution (_family_status — the
    # stage1-barrier-sign-routing harmonization): a resolved read with the
    # registered sign is 'registered'; the SAME resolved read against the
    # opposite expected sign is 'inverted' (descriptive, never
    # family-resolving); a non-significant read is 'unresolved'.
    read_pos = {"stat": 0.7, "ci_excludes_zero": True}
    assert _family_status(read_pos, 0.001, +1) == "registered"
    assert _family_status(read_pos, 0.001, -1) == "inverted"
    assert _family_status(read_pos, 0.2, +1) == "unresolved"
    read_neg = {"stat": -0.7, "ci_excludes_zero": True}
    assert _family_status(read_neg, 0.001, CHANNEL_LEAKAGE_DIR["clamp"]) == "registered"
    logger.info("lattice unit cases PASS (14/14 branches incl. inverted-sign attribution)")


def _write_fixture(out_dir: Path, geometry_path: Path) -> None:
    """Parent-shaped fixture inputs: 12 four-float files + source check + geometry."""
    rng = np.random.default_rng(7)
    ff_dir = out_dir / "four_float"
    ff_dir.mkdir(parents=True, exist_ok=True)
    n_q = 3
    arm_effect = {"split2": 4.0, "split4": 8.0, "split8": 12.0}
    arm_hijack = {"split2": 0.9, "split4": 0.5, "split8": 0.1}
    for arm, labels in ARMS.items():
        for label in labels:
            for side in ("trained", "base"):
                per_persona = {}
                for p in HELD_OUT_35:
                    rows = []
                    n_marker = round(arm_hijack[arm] * n_q) if side == "trained" else 0
                    for qi in range(n_q):
                        is_marker = side == "trained" and qi < n_marker
                        z_eos = float(10 + rng.normal(0, 0.5))
                        if side == "trained":
                            z_eos += arm_effect[arm] + rng.normal(0, 1.0)
                        rows.append(
                            {
                                "slot_kind": "pre_marker" if is_marker else "end_of_response",
                                "n_truncated_tokens": 0,
                                "gen_truncated": False,
                                "z_eos": z_eos,
                                "z_marker": float(rng.normal(0, 1)),
                                "logZ": 12.0,
                                "logp_marker": float(-10 + rng.normal(0, 1)),
                            }
                        )
                    per_persona[p] = {
                        "per_q": rows,
                        "summary": {
                            "n_pre_marker_slots": sum(r["slot_kind"] == "pre_marker" for r in rows)
                        },
                    }
                # Base side must be slot-matched to trained: copy trained slot kinds.
                (ff_dir / f"{side}_{label}.json").write_text(
                    json.dumps({"side": side, "per_persona": per_persona})
                )
            # Enforce slot parity: rewrite base with trained slot kinds.
            trained = json.loads((ff_dir / f"trained_{label}.json").read_text())
            base = json.loads((ff_dir / f"base_{label}.json").read_text())
            for p in HELD_OUT_35:
                for tq, bq in zip(
                    trained["per_persona"][p]["per_q"],
                    base["per_persona"][p]["per_q"],
                    strict=True,
                ):
                    bq["slot_kind"] = tq["slot_kind"]
            (ff_dir / f"base_{label}.json").write_text(json.dumps(base))
    (out_dir / "source_check.json").write_text(
        json.dumps(
            {
                "manipulation_check": "pass_all",
                "cross_arm_dz_marker_asymmetry": 1.2,
                "per_label": {
                    lb: {
                        "emission_on": 1.0,
                        "emission_off": 0.0,
                        "verdict": "PASS",
                        "dz_marker_source": 20.0,
                    }
                    for lb in ALL_LABELS
                },
            }
        )
    )
    dists: dict = {}
    d_src = {p: float(rng.uniform(0.2, 0.8)) for p in NEVER_NEG}
    base_nn = {p: float(rng.uniform(0.3, 0.9)) for p in NEVER_NEG}
    for layer in ("10", "15", "20"):
        dists[layer] = {}
        for centering in ("centered", "raw"):
            entry = {"d_src": d_src}
            for arm, shrink in (("split2", 1.0), ("split4", 0.7), ("split8", 0.45)):
                entry[f"d_nn_{arm}"] = {p: base_nn[p] * shrink for p in NEVER_NEG}
                entry[f"nn_identity_{arm}"] = {p: f"cand_{arm}" for p in NEVER_NEG}
            dists[layer][centering] = entry
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text(
        json.dumps(
            {
                "gate_layer": 10,
                "bystanders_never_negative": NEVER_NEG,
                "selection_provenance": {"reduced_identification": False},
                "distances": dists,
            }
        )
    )


def self_test() -> int:
    """Lattice unit cases + a full fixture run (CPU, no production inputs)."""
    _lattice_unit_cases()
    st_out = PSPLIT_OUT_DIR / "self_test"
    st_fig = PSPLIT_FIG_DIR / "self_test"
    st_geo = st_out / "geometry/psplit_geometry.json"
    _write_fixture(st_out, st_geo)
    payload = run_analysis(st_out, st_fig, st_geo)
    v = payload["verdict"]
    assert v["clamp"]["label"] == "split_axis_active", v["clamp"]
    assert v["clamp"]["direction"] == "expected", v["clamp"]
    assert v["hijack"]["label"] == "split_axis_active", v["hijack"]
    assert v["hijack"]["direction"] == "expected", v["hijack"]
    assert payload["monotonicity_4arm_descriptive"]["clamp_rises_with_count"]
    logger.info("self-test PASS: fixture run completed with the expected lattice cells")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Task #571 psplit Stage-2 off-pod analysis (verdict lattice + geometry).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", type=Path, default=PSPLIT_OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=PSPLIT_FIG_DIR)
    ap.add_argument("--geometry-json", type=Path, default=GEOMETRY_JSON)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    run_analysis(args.out_dir, args.fig_dir, args.geometry_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
