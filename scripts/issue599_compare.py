"""#599 comparison: whole-response-loss marker arm vs THREE persisted reference arms.

Reads the 9 NEW #599 cells (3 seeds x 3 text flavors, written by the pod driver
``scripts/run_issue599_fullresp.sh``) plus the persisted reference tensors:

- #551 contrastive marker + EM (18 cells) @ the pinned revision on the private
  data repo (``issue551_shift_reextract/analysis_tensors/shifts``);
- #561 positive-only marker (9 cells, the DIRECT loss-shape control arm) @ the
  pinned revision (``issue561_posonly/analysis_tensors/shifts``).

Per-cell stats are computed by ``issue561_compare.analyze_cell`` (weighted +
unit-norm top-share, U1, per-persona |cos|, sign-flip/row-shuffle nulls,
norm-vs-alignment Spearman + permutation p, split-half reliability, ||M||_F).
Cross-arm: per (variant, seed) ``|cos(U1_fullresp, U1_{marker,em,posonly})|``
+ an empirical random-pair floor.

Headline verdict logic is plan #599 §13 AS REVISED (mechanical clause
evaluation only — the analyzer owns the final verdict):

- Gatekeeper input: ``--manipulation-check`` (the driver's per-seed step-600
  FULL/PARTIAL/KILL read). Geometry is evaluated on NON-KILL same-text seeds
  only, with the pinned readable-seed denominator (>= 2 readable seeds for any
  geometry headline; per-seed clause counts = ceil(2N/3); <= 1 readable ->
  installability-only INDETERMINATE). **Both gate inputs FAIL CLOSED**: the
  full 3-seed headline is never evaluated without them — a missing gate file
  is a hard error (an explicitly-passed missing path ALWAYS errors; the
  canonical defaults error too unless ``--allow-missing-gates``, which is
  smoke/ad-hoc ONLY and labels the output ``gates_bypassed``). A seed absent
  from the manipulation check is NOT readable — hard error, never silently
  readable.
- CONFIRM: mean weighted top-share >= 0.50 AND >= ceil(2N/3) seeds >= 0.46 AND
  every gating cell clears its sign-flip null p95 AND unit-norm survival
  (mean unit-norm >= 0.40, >= ceil(2N/3) seeds rotation >= 0.95).
- FALSIFY: mean weighted <= 0.40 AND mean unit-norm <= 0.374 (the measured
  posonly control band top) AND noise-floor conjuncts on EVERY gating cell
  (sign-flip p95 cleared AND ||M||_F >= the posonly same-text floor — derived
  full-precision from the computed posonly control cells, ~9.6597; the plan's
  9.66 is its 2-dp rounding, kept only as the no-posonly fallback) —
  any gating cell failing the noise conjuncts routes to INDETERMINATE
  (noise-level). NO rotation sub-clause (plan §11.11 — the control arm's
  measured rotations make it unfireable).
- Globally binding measurement-drift rule: any CONFIRM/FALSIFY whose deciding
  mean sits within the realized extraction-smoke drift (``--smoke-gate-json``
  clauses.d_s_top1_frac) of its decision boundary routes to INDETERMINATE
  (measurement-limited).

Run (VM, CPU, off-pod — after the pod uploaded the 9 tensors). NEVER pass
``--allow-missing-gates`` here; the driver also persists both gate JSONs to
``hf://<private-data-repo>/issue599_fullresp/eval/`` if the pod->VM
eval_results sync hasn't landed:

    uv run python scripts/issue599_compare.py \\
        --new-shifts-dir eval_results/issue_599/shifts \\
        --manipulation-check eval_results/issue_599/manipulation_check.json \\
        --smoke-gate-json eval_results/issue_599_extract_smoke/smoke_gate_result.json \\
        --out eval_results/issue_599/comparison

Smoke (tiny subset, local reference dirs, no Hub; ``--allow-missing-gates``
tolerates the absent canonical gate files — smoke/ad-hoc ONLY):

    uv run python scripts/issue599_compare.py \\
        --new-shifts-dir /tmp/i599_smoke/new_shifts --new-cells marker_seed42 \\
        --variants same --issue551-local-dir /tmp/i599_smoke/i551 \\
        --issue551-cells same_marker_seed42 same_em_seed42 \\
        --issue561-local-dir /tmp/i599_smoke/i561 \\
        --issue561-cells same_marker_seed42 \\
        --allow-missing-gates \\
        --out /tmp/i599_smoke/out --figures-dir /tmp/i599_smoke/figures
"""

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from issue551_controls import N_PERM, SOURCE_PERSONA, _git_commit
from issue561_compare import (
    ALIGNED_COS_THRESHOLD,
    N_NULL_REPS,
    VARIANTS,
    WEIGHTED_CONSISTENCY_TOL,
    _download_i551,
    _load_shifts,
    _random_pair_floor,
    analyze_cell,
)

from explore_persona_space.analysis.svd_direction_constancy import cosine

logger = logging.getLogger(__name__)

SEEDS = (42, 137, 256)
I551_ARMS = ("marker", "em")
REFERENCE_ARMS = ("marker", "em", "posonly")

# Plan #599 §13 pre-registered thresholds (do NOT change without a plan
# amendment). Grounded in the measured bands (plan §11.11):
# EM weighted [0.524, 0.603]; contrastive marker [0.311, 0.348]; posonly
# control weighted [0.365, 0.402] / unit-norm [0.305, 0.374]; posonly
# same-text ||M||_F floor = min over seeds in the persisted #561 read
# (9.659689... — derived at runtime; see FROBENIUS_NOISE_FLOOR_FALLBACK).
PER_SEED_TOP_SHARE = 0.46
MEAN_TOP_SHARE_CONFIRM = 0.50
MEAN_TOP_SHARE_FALSIFY = 0.40
UNITNORM_TOP_SHARE_MIN = 0.40
UNITNORM_FALSIFY_MAX = 0.374
ROTATION_MIN = 0.95
# Fallback only: plan §13 names "the 9.66 posonly same-text floor, or grounded
# equivalent". 9.66 is a 2-dp ROUNDING of the persisted #561 minimum
# (9.659689... at seed137), so the rounded constant excludes its own grounding
# value — the control arm would fail its own floor by 3e-4. _headline derives
# the floor from the COMPUTED posonly same-text cells (the grounded
# equivalent); this constant fires only when the posonly arm is absent.
FROBENIUS_NOISE_FLOOR_FALLBACK = 9.66
MIN_READABLE_SEEDS = 2
EM_BAND_LOWER_MARGIN = 0.524

# Canonical VM-side gate-input paths (the driver's outputs after the pod->VM
# eval_results sync; the driver ALSO persists both to the private data repo
# under issue599_fullresp/eval/). The compare FAILS CLOSED on these — see
# _load_kill_status / _load_smoke_drift.
DEFAULT_MANIPULATION_CHECK = "eval_results/issue_599/manipulation_check.json"
DEFAULT_SMOKE_GATE_JSON = "eval_results/issue_599_extract_smoke/smoke_gate_result.json"


@dataclass(frozen=True)
class Cell599:
    """One analysis cell across the four arms (fullresp is the NEW arm)."""

    arm: str  # "fullresp" | "posonly" | "marker" | "em"
    variant: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.arm}:{self.variant}_seed{self.seed}"

    @property
    def file_stem(self) -> str:
        # Both the new #599 cells AND the #561 posonly reference tensors were
        # written by the same dispatcher under the arm name "marker"; the
        # #551 tensors carry their own arm in the stem.
        file_arm = "marker" if self.arm in ("fullresp", "posonly") else self.arm
        return f"{self.variant}_{file_arm}_seed{self.seed}"


def _ceil_two_thirds(n: int) -> int:
    """Per-seed clause count under the pinned readable denominator: ceil(2N/3)."""
    return math.ceil(2 * n / 3)


def _require_gate_file(path: Path, *, explicit: bool, allow_missing: bool, label: str) -> bool:
    """Fail-CLOSED existence check for a headline gate input.

    Returns True when the file is present (load it), False when it is absent
    but the absence is permitted (canonical default + --allow-missing-gates,
    smoke/ad-hoc only). An EXPLICITLY-passed missing path ALWAYS raises, and a
    missing canonical default raises too unless --allow-missing-gates.
    """
    if path.exists():
        return True
    if explicit or not allow_missing:
        raise FileNotFoundError(
            f"{label} not found: {path}"
            + ("" if explicit else " (canonical default)")
            + " — the #599 headline gates fail CLOSED (plan §7.2/§13: KILL spectra are never"
            " the implant's geometry; the measurement-drift rule is globally binding)."
            " Production must provide the driver's gate JSONs (also persisted to the private"
            " data repo under issue599_fullresp/eval/). Pass --allow-missing-gates ONLY for"
            " ad-hoc/smoke subsets."
        )
    logger.warning(
        "[gates] BYPASSED via --allow-missing-gates: %s missing at %s — the headline will be "
        "computed WITHOUT this gate and labeled gates_bypassed; NEVER a production verdict.",
        label,
        path,
    )
    return False


def _load_kill_status(path: Path, *, explicit: bool, allow_missing: bool) -> dict[int, str] | None:
    """Per-seed FULL/PARTIAL/KILL from the driver's manipulation_check.json.

    Fails CLOSED on a missing file (see _require_gate_file) and on an
    empty/malformed per_seed payload; returns None ONLY under the explicit
    --allow-missing-gates smoke bypass.
    """
    if not _require_gate_file(
        path, explicit=explicit, allow_missing=allow_missing, label="manipulation check"
    ):
        return None
    payload = json.loads(path.read_text())
    out: dict[int, str] = {}
    for key, entry in payload["per_seed"].items():
        out[int(key.removeprefix("seed"))] = str(entry["status"])
    if not out:
        raise ValueError(f"manipulation check {path} carries an empty per_seed — refusing to gate")
    return out


def _load_smoke_drift(path: Path, *, explicit: bool, allow_missing: bool) -> float | None:
    """Realized extraction-smoke drift |d s_top1| from smoke_gate_result.json.

    Fails CLOSED on a missing file (see _require_gate_file); returns None
    ONLY under the explicit --allow-missing-gates smoke bypass.
    """
    if not _require_gate_file(
        path, explicit=explicit, allow_missing=allow_missing, label="extraction-smoke gate result"
    ):
        return None
    payload = json.loads(path.read_text())
    return float(payload["clauses"]["d_s_top1_frac"])


def _check_consistency(
    per_cell: dict[str, dict],
    *,
    arms: tuple[str, ...],
    stored_cells: dict[str, dict],
    cos_lookup,
    label: str,
) -> dict[str, float]:
    """Recomputed weighted cos-to-U1 vs a persisted reference read (tol 0.001).

    ``cos_lookup(stored_entry) -> {persona: cos}`` adapts the two persisted
    schemas (#551 norm_alignment.json vs #561 comparison_per_cell.json).
    Max |delta| > 0.001 raises (different matrix convention).
    """
    out: dict[str, float] = {}
    for _name, entry in per_cell.items():
        if entry["arm"] not in arms or entry["variant"] != "same":
            continue
        stored_name = f"{entry['variant']}_{entry['arm']}_seed{entry['seed']}"
        stored = stored_cells.get(stored_name)
        if stored is None:
            logger.warning("[consistency %s] %s not stored — skipped", label, stored_name)
            continue
        stored_cos = cos_lookup(stored)
        deltas = [
            abs(entry["weighted"]["cos_to_U1"][p] - float(stored_cos[p]))
            for p in entry["persona_order"]
        ]
        max_d = max(deltas)
        if max_d > WEIGHTED_CONSISTENCY_TOL:
            raise ValueError(
                f"{label} {stored_name}: recomputed weighted cos-to-U1 disagrees with the "
                f"persisted reference (max |delta|={max_d:.2e} > {WEIGHTED_CONSISTENCY_TOL}); "
                f"refusing to compare reads built from different conventions."
            )
        out[stored_name] = float(max_d)
        logger.info("[consistency %s %s] max|delta|=%.2e OK", label, stored_name, max_d)
    return out


def _cross_arm_u1(extras: dict[str, dict]) -> dict:
    """|cos(U1_fullresp, U1_{marker,em,posonly})| per (variant, seed) + random floor."""
    out: dict[str, dict] = {}
    h = None
    for name, ex in extras.items():
        if not name.startswith("fullresp:"):
            continue
        suffix = name.split(":", 1)[1]  # "{variant}_seed{S}"
        h = ex["U1_w"].shape[0]
        row: dict[str, float] = {}
        for other_arm in REFERENCE_ARMS:
            other = extras.get(f"{other_arm}:{suffix}")
            if other is None:
                continue
            row[f"abs_cos_U1_vs_{other_arm}"] = abs(cosine(ex["U1_w"], other["U1_w"]))
        if row:
            out[suffix] = row
    payload: dict = {"per_cell": out}
    if h is not None:
        payload["random_pair_floor"] = _random_pair_floor(h)
    return payload


def _reference_bands(per_cell: dict[str, dict]) -> dict:
    """Same-text weighted + unit-norm top-share bands per reference arm."""
    bands: dict[str, dict] = {}
    for arm in REFERENCE_ARMS:
        w_vals = [
            v["weighted"]["s_top1_frac"]
            for v in per_cell.values()
            if v["arm"] == arm and v["variant"] == "same"
        ]
        u_vals = [
            v["unitnorm"]["s_top1_frac"]
            for v in per_cell.values()
            if v["arm"] == arm and v["variant"] == "same"
        ]
        if w_vals:
            bands[arm] = {
                "weighted": {"min": float(min(w_vals)), "max": float(max(w_vals))},
                "unitnorm": {"min": float(min(u_vals)), "max": float(max(u_vals))},
                "n": len(w_vals),
            }
    return bands


def _gate_readable_seeds(
    new_same: list[dict],
    kill_status: dict[int, str] | None,
    smoke_drift: float | None,
    *,
    allow_missing_gates: bool,
) -> list[dict]:
    """Fail-CLOSED §7.2/§13 gatekeeper — returns the readable (non-KILL) cells.

    A full 3-seed headline is NEVER evaluated without both gate inputs
    (round-1 review blocker compare-gates-fail-open): the all-seed-KILL modal
    outcome MUST route to "indeterminate (installability-only)", which
    requires the manipulation check; the drift rule is globally binding,
    which requires the smoke gate. A seed absent from the manipulation check
    is NOT readable — hard error, never silently readable.
    """
    if not allow_missing_gates:
        if kill_status is None:
            raise ValueError(
                "headline evaluation requires the driver's manipulation_check.json "
                "(--manipulation-check) — refusing to compute a geometry verdict that could "
                "read a KILL-branch spectrum as the implant's geometry (plan §14 must-stop)."
            )
        if smoke_drift is None:
            raise ValueError(
                "headline evaluation requires the extraction-smoke gate result "
                "(--smoke-gate-json) — the §13 measurement-drift rule is globally binding."
            )
    if kill_status is None:
        logger.warning(
            "[headline] GATES BYPASSED (--allow-missing-gates): no manipulation check — "
            "treating ALL seeds as readable; this output is NOT a production verdict."
        )
        return list(new_same)
    absent = sorted(v["seed"] for v in new_same if v["seed"] not in kill_status)
    if absent:
        raise ValueError(
            f"seeds {absent} absent from the manipulation check — a seed missing from "
            f"manipulation_check.json is NOT readable (fail closed); regenerate the "
            f"driver's per-seed read or fix the --new-cells set."
        )
    return [v for v in new_same if kill_status[v["seed"]] != "KILL"]


def _headline(
    per_cell: dict[str, dict],
    *,
    kill_status: dict[int, str] | None,
    smoke_drift: float | None,
    allow_missing_gates: bool = False,
) -> dict:
    """Mechanical plan #599 §13 clause evaluation over the same-text cells.

    Fails CLOSED: a full 3-seed headline is never evaluated without BOTH gate
    inputs (manipulation check + extraction-smoke drift) unless
    ``allow_missing_gates`` (smoke/ad-hoc only — the output is then labeled
    ``gates_bypassed``). A seed absent from the manipulation check raises.
    """
    new_same = sorted(
        (v for v in per_cell.values() if v["arm"] == "fullresp" and v["variant"] == "same"),
        key=lambda v: v["seed"],
    )
    bands = _reference_bands(per_cell)
    # Plan §13 noise floor, "grounded equivalent" form: min ||M||_F over the
    # COMPUTED posonly same-text control cells (full precision — the 9.66
    # constant is a 2-dp rounding that sits ABOVE the persisted #561 minimum
    # 9.659689..., so the rounded form would fail the control arm itself).
    posonly_frob = [
        v["frobenius_norm"]
        for v in per_cell.values()
        if v["arm"] == "posonly" and v["variant"] == "same"
    ]
    frobenius_floor = float(min(posonly_frob)) if posonly_frob else FROBENIUS_NOISE_FLOOR_FALLBACK
    frobenius_floor_source = (
        "min ||M||_F over computed posonly same-text cells (plan §13 grounded equivalent)"
        if posonly_frob
        else "fallback constant 9.66 (posonly arm absent — smoke subset only)"
    )
    base = {
        "i_reference_same_text_bands": bands,
        "kill_status": (
            {f"seed{s}": st for s, st in sorted(kill_status.items())} if kill_status else None
        ),
        "realized_smoke_drift_d_s_top1": smoke_drift,
        # Smoke/ad-hoc bypass labeling: empty on every production run. Any
        # entry here means a §13 gate was NOT applied (--allow-missing-gates).
        "gates_bypassed": [
            name
            for name, value in (
                ("manipulation_check", kill_status),
                ("smoke_drift", smoke_drift),
            )
            if value is None
        ],
        "thresholds": {
            "per_seed_top_share": PER_SEED_TOP_SHARE,
            "mean_top_share_confirm": MEAN_TOP_SHARE_CONFIRM,
            "mean_top_share_falsify": MEAN_TOP_SHARE_FALSIFY,
            "unitnorm_top_share_min": UNITNORM_TOP_SHARE_MIN,
            "unitnorm_falsify_max": UNITNORM_FALSIFY_MAX,
            "rotation_min": ROTATION_MIN,
            "frobenius_noise_floor": frobenius_floor,
            "frobenius_floor_source": frobenius_floor_source,
            "min_readable_seeds": MIN_READABLE_SEEDS,
            "per_seed_count_rule": "ceil(2N/3) over N readable (non-KILL) seeds",
            "em_band_lower_margin": EM_BAND_LOWER_MARGIN,
        },
    }
    if len(new_same) < 3:
        return {
            "evaluated": False,
            "reason": f"only {len(new_same)}/3 fullresp same-text cells present (smoke subset?)",
            **base,
        }
    readable = _gate_readable_seeds(
        new_same, kill_status, smoke_drift, allow_missing_gates=allow_missing_gates
    )
    n_readable = len(readable)
    if n_readable < MIN_READABLE_SEEDS:
        return {
            "evaluated": True,
            "mechanical_verdict": "indeterminate (installability-only)",
            "note": (
                f"only {n_readable} non-KILL same-text seed(s) — below the pinned "
                f"readable-seed denominator ({MIN_READABLE_SEEDS}); the run reports "
                f"installability only (plan §7.2/§13). The exploratory KILL-branch "
                f"spectrum is NOT the implant's geometry."
            ),
            "n_readable_seeds": n_readable,
            **base,
        }

    need = _ceil_two_thirds(n_readable)
    tops = [v["weighted"]["s_top1_frac"] for v in readable]
    unit_tops = [v["unitnorm"]["s_top1_frac"] for v in readable]
    mean_top = float(np.mean(tops))
    mean_unit = float(np.mean(unit_tops))
    per_seed = {
        f"seed{v['seed']}": {
            "kill_status": (kill_status or {}).get(v["seed"], "UNKNOWN"),
            "s_top1_frac_weighted": v["weighted"]["s_top1_frac"],
            "ge_0p46": v["weighted"]["s_top1_frac"] >= PER_SEED_TOP_SHARE,
            "passes_sign_flip_p95": v["nulls"]["passes_sign_flip_p95"],
            "s_top1_frac_unitnorm": v["unitnorm"]["s_top1_frac"],
            "rotation": v["unitnorm"]["abs_cos_U1_unitnorm_vs_weighted"],
            "rotation_ge_0p95": (v["unitnorm"]["abs_cos_U1_unitnorm_vs_weighted"] >= ROTATION_MIN),
            "frobenius_norm": v["frobenius_norm"],
            "frobenius_ge_floor": v["frobenius_norm"] >= frobenius_floor,
        }
        for v in readable
    }
    n_ge = sum(1 for v in per_seed.values() if v["ge_0p46"])
    all_sign = all(v["passes_sign_flip_p95"] for v in per_seed.values())
    n_rot_ok = sum(1 for v in per_seed.values() if v["rotation_ge_0p95"])
    noise_ok = all(v["passes_sign_flip_p95"] and v["frobenius_ge_floor"] for v in per_seed.values())

    confirm = (
        mean_top >= MEAN_TOP_SHARE_CONFIRM
        and n_ge >= need
        and all_sign
        and mean_unit >= UNITNORM_TOP_SHARE_MIN
        and n_rot_ok >= need
    )
    falsify_geometry = mean_top <= MEAN_TOP_SHARE_FALSIFY and mean_unit <= UNITNORM_FALSIFY_MAX

    if confirm:
        verdict = "confirm"
    elif falsify_geometry and noise_ok:
        verdict = "falsify"
    elif falsify_geometry and not noise_ok:
        # §13 noise conjuncts: a dispersion read on noise-level shifts is NOT
        # a falsification — an under-installed arm has small norms.
        verdict = "indeterminate (noise-level)"
    else:
        verdict = "indeterminate"

    # Globally binding measurement-drift rule (§13): a CONFIRM/FALSIFY whose
    # deciding mean sits within the realized extraction-smoke drift of its
    # boundary is measurement-limited. Symmetric across verdicts: BOTH
    # deciding means (weighted AND unit-norm) are checked against their
    # respective decision boundaries on confirm exactly as on falsify.
    drift_margins: dict[str, float] = {
        "confirm_margin_weighted": abs(mean_top - MEAN_TOP_SHARE_CONFIRM),
        "confirm_margin_unitnorm": abs(mean_unit - UNITNORM_TOP_SHARE_MIN),
        "falsify_margin_weighted": abs(mean_top - MEAN_TOP_SHARE_FALSIFY),
        "falsify_margin_unitnorm": abs(mean_unit - UNITNORM_FALSIFY_MAX),
    }
    drift_limited = False
    if smoke_drift is not None:
        if verdict == "confirm" and (
            drift_margins["confirm_margin_weighted"] <= smoke_drift
            or drift_margins["confirm_margin_unitnorm"] <= smoke_drift
        ):
            drift_limited = True
        if verdict == "falsify" and (
            drift_margins["falsify_margin_weighted"] <= smoke_drift
            or drift_margins["falsify_margin_unitnorm"] <= smoke_drift
        ):
            drift_limited = True
        if drift_limited:
            verdict = "indeterminate (measurement-limited)"
    elif verdict in ("confirm", "falsify"):
        # Reachable ONLY under --allow-missing-gates (production raised above).
        logger.warning(
            "[headline] GATES BYPASSED (--allow-missing-gates): verdict=%s computed WITHOUT "
            "the globally binding measurement-drift rule — NOT a production verdict.",
            verdict,
        )

    return {
        "evaluated": True,
        "mechanical_verdict": verdict,
        "note": (
            "Mechanical clause evaluation only — the analyzer owns the verdict. "
            "Non-confirm/non-falsify routes to indeterminate; a mean in "
            f"[{MEAN_TOP_SHARE_CONFIRM}, {EM_BAND_LOWER_MARGIN}] is 'EM-like, at the "
            "lower margin of the EM band'; boundary calls within the realized "
            "extraction-smoke drift are measurement-limited; KILL-branch spectra are "
            "never the implant's geometry; the self-generated-data caveat is carried."
        ),
        "n_readable_seeds": n_readable,
        "per_seed_count_required": need,
        "mean_s_top1_frac_weighted": mean_top,
        "mean_s_top1_frac_unitnorm": mean_unit,
        "n_seeds_ge_0p46": n_ge,
        "all_pass_sign_flip_p95": all_sign,
        "n_seeds_rotation_ge_0p95": n_rot_ok,
        "noise_conjuncts_ok": noise_ok,
        "drift_limited": drift_limited,
        "drift_margins": drift_margins,
        "per_seed": per_seed,
        **base,
    }


def make_figures(per_cell: dict[str, dict], figures_dir: Path) -> None:
    """Hero four-arm band plot + profile/norm scatters (guarded on coverage)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(4)
    arm_color = {"em": colors[0], "marker": colors[1], "posonly": colors[2], "fullresp": colors[3]}
    arm_label = {
        "marker": "contrastive marker (reference)",
        "posonly": "positive-only marker (control)",
        "fullresp": "whole-response marker (this run)",
        "em": "EM insecure-code (reference)",
    }
    arm_order = ["marker", "posonly", "fullresp", "em"]
    same = [v for v in per_cell.values() if v["variant"] == "same"]
    if not same:
        logger.warning("[figures] no same-text cells — skipping all figures")
        return

    # ── Hero: four-arm top-share bands, weighted vs unit-norm panels ──
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, read in zip(axes, ("weighted", "unitnorm"), strict=True):
        for i, arm in enumerate(arm_order):
            cells = sorted((v for v in same if v["arm"] == arm), key=lambda v: v["seed"])
            if not cells:
                continue
            xs = [i + (j - 1) * 0.12 for j in range(len(cells))]
            ys = [v[read]["s_top1_frac"] for v in cells]
            ax.scatter(xs, ys, s=46, color=arm_color[arm], zorder=3)
            for x, v in zip(xs, cells, strict=True):
                ax.annotate(
                    str(v["seed"]),
                    (x, v[read]["s_top1_frac"]),
                    fontsize=6.5,
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                )
                if read == "weighted":
                    ax.hlines(
                        v["nulls"]["sign_flip_p95"],
                        x - 0.07,
                        x + 0.07,
                        color="black",
                        lw=1.1,
                    )
        ax.set_xticks(range(len(arm_order)))
        ax.set_xticklabels([arm_label[a].replace(" (", "\n(") for a in arm_order], fontsize=7)
        ax.set_title(
            "norm-weighted SVD" if read == "weighted" else "unit-norm columns",
            fontsize=10,
        )
    axes[0].set_ylabel("top-direction share of spectrum")
    fig.suptitle(
        "Trained-model-text shift-spectrum concentration by training condition "
        "(ticks = per-cell sign-flip null p95)",
        fontsize=10,
    )
    fig.tight_layout()
    savefig_paper(fig, "four_arm_top_share_bands", dir=figures_dir)
    plt.close(fig)

    # ── Per-persona |cos| profile: fullresp vs posonly control, per seed ──
    pairs = []
    for v in same:
        if v["arm"] != "fullresp":
            continue
        ref = next(
            (w for w in same if w["arm"] == "posonly" and w["seed"] == v["seed"]),
            None,
        )
        if ref is not None:
            pairs.append((v, ref))
    if pairs:
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        ax.plot([0, 1], [0, 1], color="lightgray", lw=1.0, zorder=0)
        markers = {42: "o", 137: "^", 256: "s"}
        for v, ref in pairs:
            personas = v["persona_order"]
            ax.scatter(
                [abs(ref["weighted"]["cos_to_U1"][p]) for p in personas],
                [abs(v["weighted"]["cos_to_U1"][p]) for p in personas],
                s=26,
                color=arm_color["fullresp"],
                marker=markers.get(v["seed"], "o"),
                alpha=0.75,
                label=f"seed {v['seed']}",
            )
        ax.set_xlabel("absolute cosine to top direction\npositive-only marker (control)")
        ax.set_ylabel("absolute cosine to top direction\nwhole-response marker (this run)")
        ax.set_title("Per-persona alignment: whole-response vs positive-only\n(trained-model text)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        savefig_paper(fig, "fullresp_vs_posonly_profiles", dir=figures_dir)
        plt.close(fig)

    # ── New arm: shift norm vs |cos| (weighted) ─────────────────────────
    new_same = [v for v in same if v["arm"] == "fullresp"]
    if new_same:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for i, v in enumerate(sorted(new_same, key=lambda v: v["seed"])):
            personas = v["persona_order"]
            norms = v["norm_vs_alignment"]["norms"]
            ax.scatter(
                [norms[p] for p in personas],
                [abs(v["weighted"]["cos_to_U1"][p]) for p in personas],
                s=20,
                color=paper_palette(4)[i],
                alpha=0.8,
                label=f"seed {v['seed']}",
            )
        ax.axvline(0.0, color="lightgray", lw=0.8, zorder=0)
        ax.set_xlabel("per-persona shift norm")
        ax.set_ylabel("absolute cosine to top direction")
        ax.set_title("Whole-response marker: shift size vs alignment (trained-model text)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        savefig_paper(fig, "fullresp_norm_vs_alignment", dir=figures_dir)
        plt.close(fig)

    logger.info("[figures] written to %s", figures_dir)


def _parse_ref_cells(
    specs: list[str] | None, *, arms: tuple[str, ...], variants: list[str]
) -> list["Cell599"]:
    """Parse reference cell stems like 'same_marker_seed42' into Cell599s.

    ``specs=None`` -> the full grid (arms x requested variants x 3 seeds).
    For the posonly arm the stems carry the FILE arm 'marker' (the #561
    dispatcher wrote them under that name); single-arm calls accept it.
    """
    if specs is None:
        return [Cell599(arm=a, variant=v, seed=s) for v in variants for a in arms for s in SEEDS]
    cells = []
    for stem in specs:
        # Match by known variant prefixes (longest first): a first-underscore
        # split would break the multi-token 'on_policy_*' stems.
        variant = next(
            (v for v in sorted(VARIANTS, key=len, reverse=True) if stem.startswith(v + "_")),
            None,
        )
        assert variant is not None, f"bad reference stem {stem!r}: no variant prefix in {VARIANTS}"
        arm, _, seed_s = stem[len(variant) + 1 :].partition("_seed")
        if arms == ("posonly",) and arm == "marker":
            arm = "posonly"  # file stem carries the dispatcher arm name
        assert arm in arms and seed_s, f"bad reference stem {stem!r} for arms {arms}"
        cells.append(Cell599(arm=arm, variant=variant, seed=int(seed_s)))
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "#599 comparison: whole-response-loss marker vs persisted #551 + #561 tensors (CPU)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--new-shifts-dir", default="eval_results/issue_599/shifts")
    parser.add_argument(
        "--new-cells",
        nargs="+",
        default=["marker_seed42", "marker_seed137", "marker_seed256"],
        help="New-arm cell specs (file arm is 'marker'; analyzed as arm 'fullresp').",
    )
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--issue551-repo", default="superkaiba1/explore-persona-space-data-private")
    parser.add_argument(
        "--issue551-prefix", default="issue551_shift_reextract/analysis_tensors/shifts"
    )
    parser.add_argument("--issue551-revision", default="08419ee885e962cb29c841d34041db419dbbc72c")
    parser.add_argument(
        "--issue551-local-dir",
        default=None,
        help="Read the #551 .pt files from this dir instead of downloading (smoke).",
    )
    parser.add_argument(
        "--issue551-cells",
        nargs="+",
        default=None,
        help=(
            "Subset of #551 cell stems (e.g. 'same_marker_seed42'). Default = all "
            "18 (2 arms x 3 seeds x the requested --variants)."
        ),
    )
    parser.add_argument("--issue561-repo", default="superkaiba1/explore-persona-space-data-private")
    parser.add_argument("--issue561-prefix", default="issue561_posonly/analysis_tensors/shifts")
    parser.add_argument("--issue561-revision", default="fe5488e8fbda3fa106ac50b4358b4b52ef1a8fd7")
    parser.add_argument(
        "--issue561-local-dir",
        default=None,
        help="Read the #561 posonly .pt files from this dir instead of downloading (smoke).",
    )
    parser.add_argument(
        "--issue561-cells",
        nargs="+",
        default=None,
        help=(
            "Subset of #561 posonly cell stems (file arm 'marker', e.g. "
            "'same_marker_seed42'). Default = 9 (3 seeds x the requested --variants)."
        ),
    )
    parser.add_argument(
        "--norm-align-json",
        default="eval_results/issue_551/controls/norm_alignment.json",
        help="Persisted #551 weighted read for the 0.001 consistency cross-check.",
    )
    parser.add_argument(
        "--i561-per-cell-json",
        default="eval_results/issue_561/comparison/comparison_per_cell.json",
        help="Persisted #561 comparison read for the posonly-side 0.001 cross-check.",
    )
    parser.add_argument(
        "--manipulation-check",
        default=None,
        help=(
            "Path to the driver's manipulation_check.json (per-seed step-600 "
            "FULL/PARTIAL/KILL); geometry is evaluated on non-KILL seeds only "
            f"(plan §7.2/§13). Default: {DEFAULT_MANIPULATION_CHECK}. FAILS "
            "CLOSED when the file is absent (an explicitly-passed missing path "
            "always errors; the default errors too unless --allow-missing-gates)."
        ),
    )
    parser.add_argument(
        "--smoke-gate-json",
        default=None,
        help=(
            "Path to the pod's extraction-smoke smoke_gate_result.json; its "
            "clauses.d_s_top1_frac is the realized drift for the globally "
            f"binding measurement-drift rule (plan §13). Default: "
            f"{DEFAULT_SMOKE_GATE_JSON}. FAILS CLOSED when absent (same rule "
            "as --manipulation-check)."
        ),
    )
    parser.add_argument(
        "--allow-missing-gates",
        action="store_true",
        help=(
            "Smoke/ad-hoc ONLY: tolerate ABSENT (defaulted) gate files — the "
            "headline is then computed without KILL gating / the drift rule and "
            "labeled gates_bypassed in the output. NEVER pass this on a "
            "production invocation. An explicitly-passed gate path that does "
            "not exist STILL errors."
        ),
    )
    parser.add_argument("--out", default="eval_results/issue_599/comparison")
    parser.add_argument("--figures-dir", default="figures/issue_599")
    parser.add_argument("--n-random-splits", type=int, default=50)
    parser.add_argument("--split-rng-seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    # `uv run python` does NOT auto-load .env; the private repo needs HF_TOKEN.
    load_dotenv()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)

    # Headline gate inputs FIRST (fail CLOSED, before any download/analysis
    # spend): an explicitly-passed missing path always errors; the canonical
    # defaults error too unless --allow-missing-gates (smoke/ad-hoc only).
    manip_explicit = args.manipulation_check is not None
    manip_path = Path(args.manipulation_check or DEFAULT_MANIPULATION_CHECK)
    smoke_explicit = args.smoke_gate_json is not None
    smoke_path = Path(args.smoke_gate_json or DEFAULT_SMOKE_GATE_JSON)
    kill_status = _load_kill_status(
        manip_path, explicit=manip_explicit, allow_missing=args.allow_missing_gates
    )
    smoke_drift = _load_smoke_drift(
        smoke_path, explicit=smoke_explicit, allow_missing=args.allow_missing_gates
    )

    # Build the cell lists.
    new_cells: list[Cell599] = []
    for spec in args.new_cells:
        arm, _, rest = spec.partition("_seed")
        assert arm == "marker", f"--new-cells spec {spec!r} must look like 'marker_seed42'"
        for variant in args.variants:
            new_cells.append(Cell599(arm="fullresp", variant=variant, seed=int(rest)))

    i551_cells = _parse_ref_cells(args.issue551_cells, arms=I551_ARMS, variants=args.variants)
    i561_cells = _parse_ref_cells(args.issue561_cells, arms=("posonly",), variants=args.variants)

    # Resolve tensor sources (per-file hf_hub_download at the PINNED revisions).
    new_dir = Path(args.new_shifts_dir)
    if args.issue551_local_dir:
        i551_dir = Path(args.issue551_local_dir)
        logger.info("[load] local #551 shifts dir %s", i551_dir)
    else:
        i551_dir = _download_i551(
            args.issue551_repo,
            args.issue551_prefix,
            args.issue551_revision,
            [c.file_stem for c in i551_cells],
            out_dir / "i551_shifts_downloaded",
        )
    if args.issue561_local_dir:
        i561_dir = Path(args.issue561_local_dir)
        logger.info("[load] local #561 shifts dir %s", i561_dir)
    else:
        i561_dir = _download_i551(
            args.issue561_repo,
            args.issue561_prefix,
            args.issue561_revision,
            [c.file_stem for c in i561_cells],
            out_dir / "i561_shifts_downloaded",
        )

    meta = {
        "issue": 599,
        "analysis": "fullresp_vs_persisted_551_and_561_comparison",
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "new_shifts_dir": str(new_dir),
        "i551_source": (
            str(args.issue551_local_dir)
            if args.issue551_local_dir
            else f"hf://{args.issue551_repo}/{args.issue551_prefix}@{args.issue551_revision}"
        ),
        "i561_source": (
            str(args.issue561_local_dir)
            if args.issue561_local_dir
            else f"hf://{args.issue561_repo}/{args.issue561_prefix}@{args.issue561_revision}"
        ),
        "manipulation_check": str(manip_path),
        "smoke_gate_json": str(smoke_path),
        "allow_missing_gates": args.allow_missing_gates,
        "n_null_reps": N_NULL_REPS,
        "n_perm": N_PERM,
        "n_random_splits": args.n_random_splits,
        "split_rng_seed": args.split_rng_seed,
        "aligned_cos_threshold": ALIGNED_COS_THRESHOLD,
        "weighted_consistency_tol": WEIGHTED_CONSISTENCY_TOL,
        "source_persona": SOURCE_PERSONA,
        "top_share_definition": "s_1 / sum(s) (matches svd_summary + the persisted JSONs)",
    }

    per_cell: dict[str, dict] = {}
    extras: dict[str, dict] = {}
    for cell in [*new_cells, *i551_cells, *i561_cells]:
        if cell.arm == "fullresp":
            src = new_dir
        elif cell.arm == "posonly":
            src = i561_dir
        else:
            src = i551_dir
        shifts = _load_shifts(src, cell.file_stem)
        entry, ex = analyze_cell(
            cell,
            shifts,
            n_random_splits=args.n_random_splits,
            split_rng_seed=args.split_rng_seed,
        )
        per_cell[cell.name] = entry
        extras[cell.name] = ex

    # Consistency cross-checks vs BOTH persisted reads (tol 0.001 each).
    consistency: dict[str, dict[str, float]] = {}
    norm_align_path = Path(args.norm_align_json)
    if norm_align_path.exists():
        with norm_align_path.open() as f:
            norm_align = json.load(f)
        consistency["i551"] = _check_consistency(
            per_cell,
            arms=I551_ARMS,
            stored_cells=norm_align.get("per_cell", {}),
            cos_lookup=lambda stored: stored["cos_to_U1"],
            label="i551",
        )
    else:
        logger.warning(
            "[consistency] %s not found — #551-side cross-check SKIPPED "
            "(fine for smoke; production must run it)",
            norm_align_path,
        )
    i561_pc_path = Path(args.i561_per_cell_json)
    if i561_pc_path.exists():
        with i561_pc_path.open() as f:
            i561_pc = json.load(f)
        # #561 keys are 'posonly:{variant}_seed{S}'; re-key to this script's
        # stored-name convention '{variant}_posonly_seed{S}'.
        stored = {}
        for name, entry in i561_pc.get("per_cell", {}).items():
            if not name.startswith("posonly:"):
                continue
            variant, _, seed_s = name.split(":", 1)[1].rpartition("_seed")
            stored[f"{variant}_posonly_seed{seed_s}"] = entry
        consistency["i561"] = _check_consistency(
            per_cell,
            arms=("posonly",),
            stored_cells=stored,
            cos_lookup=lambda stored_entry: stored_entry["weighted"]["cos_to_U1"],
            label="i561",
        )
    else:
        logger.warning(
            "[consistency] %s not found — #561-side cross-check SKIPPED "
            "(fine for smoke; production must run it)",
            i561_pc_path,
        )

    # Checkpoint per phase: per-cell JSON FIRST, then summary, figures last.
    with (out_dir / "comparison_per_cell.json").open("w") as f:
        json.dump({"meta": meta, "per_cell": per_cell}, f, indent=2)
    logger.info("[wrote] %s", out_dir / "comparison_per_cell.json")

    summary = {
        "meta": meta,
        "headline": _headline(
            per_cell,
            kill_status=kill_status,
            smoke_drift=smoke_drift,
            allow_missing_gates=args.allow_missing_gates,
        ),
        "cross_arm_U1": _cross_arm_u1(extras),
        "reference_weighted_consistency_max_abs_delta": consistency,
        "per_cell_top_share": {
            name: {
                "arm": v["arm"],
                "variant": v["variant"],
                "seed": v["seed"],
                "s_top1_frac_weighted": v["weighted"]["s_top1_frac"],
                "s_top1_frac_unitnorm": v["unitnorm"]["s_top1_frac"],
                "rotation": v["unitnorm"]["abs_cos_U1_unitnorm_vs_weighted"],
                "mean_cos_to_U1": v["weighted"]["mean_cos_to_U1"],
                "frobenius_norm": v["frobenius_norm"],
                "passes_sign_flip_p95": v["nulls"]["passes_sign_flip_p95"],
            }
            for name, v in per_cell.items()
        },
    }
    with (out_dir / "comparison_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[wrote] %s", out_dir / "comparison_summary.json")

    make_figures(per_cell, figures_dir)

    hl = summary["headline"]
    logger.info(
        "[phase=done] cells=%d mechanical_verdict=%s",
        len(per_cell),
        hl.get("mechanical_verdict", "n/a (not evaluated)"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
