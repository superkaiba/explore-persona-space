#!/usr/bin/env python3
"""Issue #2588 P3 — cross-panel trend analysis (VM-side, CPU-only).

Consumes the harvested per-cell fit artifacts (``--fits-dir``, default
``eval_results/issue_2588/fits/<cell_key>/``: fits_/nulls_/perrow_/
gpqa_transfer_/resid_<pos>.json) and emits
``eval_results/issue_2588/trend_summary.json`` with:

- per-map CALIBRATED primary DV: test kNN acc@1 (cosine, layer_star) minus the
  200-draw shuffled-pairing null mean (nulls are ADVISORY — no gate thresholds
  a null statistic; selection-symmetric via the held-out VAL layer freeze);
- the capability-column contrasts (Qwen3.5-27B -> 3.6-27B -> 3.8-27B) per arm:
  paired bootstrap (1,000 draws, seed 42, ONE shared resample matrix) on raw
  per-row paired differences, SHIFTED by the null-mean difference; SE_shift =
  sqrt(sd_null_A^2 + sd_null_B^2)/sqrt(200) reported separately;
- the verdict lattice (Endpoint-up / Capability-inverts / Order-consistent /
  Capability-tracks-ordered) — consumes CALIBRATED fields ONLY (mechanized:
  ``assert_calibrated_inputs`` + the committed synthetic-orderings test);
- H2: Wilcoxon signed-rank over the 7 Qwen thinking checkpoints (end-of-CoT
  arm-b map vs same-checkpoint arm-a prompt-side map; min two-sided p at n=7
  is 0.0156); OLMo-R and OLMo-P reported PER PAIR (n=2 each), NEVER pooled —
  ``assert_pair_metadata`` rejects mixed pairs (plan §4.3, MF2);
- Spearman(AA capability pin, calibrated acc@1) per arm;
- the conditional §4.5 GPQA extraction-judge fallback (``--judge-fallback``),
  routed through llm/api_dispatch.py (judge claude-sonnet-4-5-20250929).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
for p in (str(_SCRIPTS), str(_REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps BEFORE numpy import (VM-side P3 entrypoint)

import numpy as np  # noqa: E402

import issue2588_panel_common as PC  # noqa: E402

logger = logging.getLogger("issue2588_trend")

OUT_PATH = _REPO_ROOT / "eval_results" / "issue_2588" / "trend_summary.json"

# Input-position semantics classes (plan §4.3): prompt_last and pre_think are
# BOTH pre-generation prompt-side reads (the OLMo-R comparability premise);
# cot_boundary is the end-of-CoT completion-side read.
POSITION_SEMANTICS = {
    "prompt_last": "prompt_side",
    "pre_think": "prompt_side",
    "cot_boundary": "end_of_cot",
}

QWEN_THINKING_KEYS = ("q35_0p8b", "q35_2b", "q35_4b", "q35_9b", "q35_27b", "q36_27b", "q38_27b")
OLMO_PAIRS = (("o3_7b_i", "o3_7b_t"), ("o31_32b_i", "o31_32b_t"))


@dataclass(frozen=True)
class MapRef:
    """One registered fit map: (cell, input position)."""

    model_key: str
    arm: str
    pos: str

    @property
    def cell_key(self) -> str:
        return f"{self.model_key}_{self.arm}"

    @property
    def map_id(self) -> str:
        return f"{self.cell_key}.{self.pos}"


def all_maps() -> list[MapRef]:
    out = [MapRef(c.model_key, c.arm, pos) for c in PC.all_cells() for pos in c.input_positions]
    assert len(out) == 21, len(out)
    return out


def assert_pair_metadata(a: MapRef, b: MapRef) -> None:
    """P3 pair-metadata assert (plan §4.3): a registered contrast pair must
    share EITHER the checkpoint id OR the input-position semantics class.
    A mixed pair (different checkpoint AND different semantics) confounds
    checkpoint identity with read position and is REJECTED."""
    same_ckpt = a.model_key == b.model_key
    same_sem = POSITION_SEMANTICS[a.pos] == POSITION_SEMANTICS[b.pos]
    if not (same_ckpt or same_sem):
        raise ValueError(
            f"mixed contrast pair REJECTED: {a.map_id} vs {b.map_id} — neither same "
            "checkpoint nor same input-position semantics (plan §4.3 MF2 ban)"
        )


def assert_calibrated_inputs(record: dict) -> None:
    """The verdict lattice consumes CALIBRATED fields ONLY (plan §4.4): every
    delta/CI key handed to it must carry the ``_cal`` suffix."""
    bad = [k for k in record if not k.endswith("_cal")]
    if bad:
        raise ValueError(
            f"verdict lattice handed NON-calibrated fields {bad} — the lattice is defined "
            "over null-mean-calibrated contrasts only"
        )


def verdict_lattice(record: dict) -> dict:
    """The registered verdict lattice (plan §4.4/§7).

    record keys (all REQUIRED, all ``_cal``): delta_endpoint_cal (3.8 - 3.5),
    ci_low_endpoint_cal / ci_high_endpoint_cal (shifted 95% bootstrap CI),
    delta_step1_cal (3.6 - 3.5), delta_step2_cal (3.8 - 3.6).
    """
    assert_calibrated_inputs(record)
    need = {
        "delta_endpoint_cal",
        "ci_low_endpoint_cal",
        "ci_high_endpoint_cal",
        "delta_step1_cal",
        "delta_step2_cal",
    }
    missing = need - set(record)
    assert not missing, f"verdict lattice missing fields: {sorted(missing)}"
    # Endpoint partition (DISJOINT + exhaustive, plan §3): Endpoint-up /
    # Capability-inverts / Indistinguishable.
    endpoint_up = record["delta_endpoint_cal"] > 0 and record["ci_low_endpoint_cal"] > 0
    capability_inverts = record["ci_high_endpoint_cal"] < 0
    endpoint_label = (
        "endpoint_up"
        if endpoint_up
        else "capability_inverts"
        if capability_inverts
        else "indistinguishable"
    )
    # Ordering partition (DISJOINT + exhaustive): Δadj_min = min of the two
    # ADJACENT calibrated point contrasts (the MF3 midpoint fix).
    delta_adj_min = min(record["delta_step1_cal"], record["delta_step2_cal"])
    order_consistent = delta_adj_min >= 0
    return {
        "endpoint_up": bool(endpoint_up),
        "capability_inverts": bool(capability_inverts),
        "endpoint_label": endpoint_label,
        "order_consistent": bool(order_consistent),
        "ordering_label": "order_consistent" if order_consistent else "order_inconsistent",
        "delta_adj_min_cal": float(delta_adj_min),
        "capability_tracks_ordered": bool(endpoint_up and order_consistent),
    }


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _load_map(fits_dir: Path, ref: MapRef) -> dict | None:
    cell_dir = fits_dir / ref.cell_key
    fits_p = cell_dir / f"fits_{ref.pos}.json"
    if not fits_p.exists():
        return None
    rec = {"fits": json.loads(fits_p.read_text(encoding="utf-8"))}
    for kind in ("nulls", "perrow", "gpqa_transfer", "gpqa_perrow", "resid"):
        p = cell_dir / f"{kind}_{ref.pos}.json"
        rec[kind] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    # Cell-level judge-fallback artifacts (B5): pending rows written pod-side;
    # verdicts written by --judge-fallback on the VM.
    for kind, name in (
        ("judge_pending", "gpqa_judge_pending.json"),
        ("judge_verdicts", "gpqa_judge_verdicts.json"),
    ):
        p = cell_dir / name
        rec[kind] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return rec


def _acc_at_1(knn_read: dict) -> float:
    """acc@1 accessor tolerant of JSON-round-tripped int keys ("1" vs 1)."""
    acc = knn_read["acc_at_k"]
    v = acc.get("1", acc.get(1))
    assert v is not None, f"acc_at_k lacks k=1: {acc!r}"
    return float(v)


def _acc1_at_star(fits: dict) -> float:
    star = str(fits["layer_star"])
    return _acc_at_1(fits["layers"][star]["knn_test"]["ridge"]["cosine"])


def _calibrated(rec: dict) -> dict:
    obs = _acc1_at_star(rec["fits"])
    nulls = rec["nulls"]
    mu = float(nulls["null_mean_acc1_cos"]) if nulls else None
    sd = float(nulls["null_sd_acc1_cos"]) if nulls else None
    n_draws = int(nulls["perm_draws"]) if nulls else None
    # SR1 (B3, review round 2): repeat-draw retrieval-ceiling normalization —
    # (map − null) / (ceiling − null), where the ceiling is the REGISTERED
    # seed-43→seed-44 cosine retrieval acc@1 at the selected layer.
    ceil_retr = rec["fits"].get("ceiling_retrieval_at_star")
    sr1 = None
    if ceil_retr is not None and mu is not None:
        denom = float(ceil_retr["ceiling_acc1_cos"]) - mu
        # Round 3 (standing rec): a ceiling at/below the null mean makes SR1
        # undefined — fail loud, never a silent None that reads downstream as
        # "ceiling not computed".
        if denom <= 1e-12:
            raise ValueError(
                f"SR1 ceiling normalization degenerate: ceiling_acc1_cos="
                f"{float(ceil_retr['ceiling_acc1_cos']):.6f} <= null_mean={mu:.6f} "
                f"(denom={denom:.3g}) — the repeat-draw retrieval ceiling sits at/below "
                "the permutation null; the cell's capture or nulls are broken"
            )
        sr1 = float((obs - mu) / denom)
    return {
        "acc1_cos_at_star": obs,
        "null_mean": mu,
        "null_sd": sd,
        "perm_draws": n_draws,
        "acc1_cos_calibrated": (obs - mu) if mu is not None else None,
        "layer_star": int(rec["fits"]["layer_star"]),
        "ceiling_two_draw": rec["fits"].get("ceiling_two_draw_at_star"),
        "ceiling_retrieval": ceil_retr,
        "acc1_cos_ceiling_normalized": sr1,
        "participation_ratio_x_at_star": rec["fits"].get("participation_ratio_x_at_star"),
    }


def _perrow_by_ci(rec: dict) -> dict[str, int]:
    pr = rec["perrow"]
    assert pr is not None, "perrow hits missing — the paired bootstrap needs them"
    return {
        str(r).rsplit("_", 1)[1]: int(h) for r, h in zip(pr["row_ids"], pr["hit1_cos"], strict=True)
    }


def _gpqa_perrow_by_id(rec: dict) -> dict[str, int]:
    """GPQA per-row same-question hits keyed by row_id = "<qid>_s<seed>" (B4)."""
    gp = rec["gpqa_perrow"]
    assert gp is not None, "gpqa_perrow hits missing — the H2 surface contrast needs them"
    return {str(r): int(h) for r, h in zip(gp["row_ids"], gp["same_q_hit"], strict=True)}


# ---------------------------------------------------------------------------
# Paired bootstrap (one shared resample matrix; plan §4.4)
# ---------------------------------------------------------------------------


def _shared_resample_matrix(universe: list[str], draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(universe)
    return rng.integers(0, n, size=(draws, n))


def _boot_means(diff: np.ndarray, in_shared: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Vectorized per-draw means of ``diff`` over the shared-row subset of each
    resample draw (one shared matrix; per-draw restriction to shared rows)."""
    keep = in_shared[matrix]  # (draws, n) bool
    ksum = keep.sum(axis=1)
    assert (ksum > 0).all(), "bootstrap draw with zero shared rows"
    return (diff[matrix] * keep).sum(axis=1) / ksum


def _paired_gap_boot(
    hits_b: dict[str, int], hits_a: dict[str, int], universe: list[str], matrix: np.ndarray
) -> tuple[float, np.ndarray, int]:
    """Complete-case paired gap (arm b − arm a) over the shared rows of one
    surface + its per-checkpoint bootstrap draws (B4, review round 2)."""
    in_shared = np.array([rid in hits_b and rid in hits_a for rid in universe])
    n_shared = int(in_shared.sum())
    assert n_shared > 0, "paired gap: zero shared rows after the complete-case intersection"
    diff = np.array([hits_b.get(rid, 0) - hits_a.get(rid, 0) for rid in universe], dtype=np.float64)
    gap = float(diff[in_shared].mean())
    return gap, _boot_means(diff, in_shared, matrix), n_shared


def paired_contrast(
    rec_a: dict, rec_b: dict, ref_a: MapRef, ref_b: MapRef, universe: list[str], matrix: np.ndarray
) -> dict:
    """Calibrated contrast A - B with the shifted paired-bootstrap 95% CI.

    Raw per-row paired differences over the pair's SHARED test rows; every
    contrast consumes the SAME resample matrix (drawn once, seed 42) over the
    test-ci universe, per-draw restricted to the shared rows. The calibration
    SHIFTS the bootstrap distribution by (null_mean_A - null_mean_B); the
    null-mean estimation error rides SE_shift, reported separately (never
    folded into the CI — plan §4.4)."""
    assert_pair_metadata(ref_a, ref_b)
    hits_a, hits_b = _perrow_by_ci(rec_a), _perrow_by_ci(rec_b)
    shared = [ci for ci in universe if ci in hits_a and ci in hits_b]
    # G5-bounded regime: per-cell drops <= 2% imply a shared intersection
    # >= ~96%; realized q38_27b_b test drops (14.1% unclosed-think residue
    # AFTER the registered regen, whose max_model_len re-pin ceiling binds)
    # violate that premise, so sub-floor pairs are computed on the realized
    # intersection and LABELED g5_bounded=false (never silently consumed;
    # column_verdicts downgrades the arm's status). A catastrophic floor
    # still hard-fails — below half coverage the paired read is not a
    # measurement of the registered universe at all.
    g5_bounded = len(shared) >= 0.9 * len(universe)
    assert len(shared) >= 0.5 * len(universe), (
        f"paired contrast {ref_a.map_id} vs {ref_b.map_id}: only {len(shared)}/"
        f"{len(universe)} shared rows — below the catastrophic 0.5 floor"
    )
    in_shared = np.array([ci in hits_a and ci in hits_b for ci in universe])
    da = np.array([hits_a.get(ci, 0) for ci in universe], dtype=np.float64)
    db = np.array([hits_b.get(ci, 0) for ci in universe], dtype=np.float64)
    diff = da - db
    raw_delta = float(diff[in_shared].mean())
    boot = _boot_means(diff, in_shared, matrix)
    cal_a, cal_b = _calibrated(rec_a), _calibrated(rec_b)
    delta_null = cal_a["null_mean"] - cal_b["null_mean"]
    boot_cal = boot - delta_null
    n_draws = cal_a["perm_draws"]
    se_shift = float(np.sqrt(cal_a["null_sd"] ** 2 + cal_b["null_sd"] ** 2) / np.sqrt(n_draws))
    return {
        "pair": [ref_a.map_id, ref_b.map_id],
        "n_shared_rows": len(shared),
        "g5_bounded": bool(g5_bounded),
        "delta_raw": raw_delta,
        "delta_null_means": float(delta_null),
        "delta_cal": float(raw_delta - delta_null),
        "ci95_cal": [float(np.percentile(boot_cal, 2.5)), float(np.percentile(boot_cal, 97.5))],
        "se_shift": se_shift,
        "bootstrap_draws": int(matrix.shape[0]),
        "bootstrap_seed": PC.BOOTSTRAP_SEED,
    }


# ---------------------------------------------------------------------------
# Column verdicts + H2 + OLMo pairs + Spearman
# ---------------------------------------------------------------------------


def _column_pos(arm: str) -> str:
    return "prompt_last" if arm == "a" else "cot_boundary"


def column_verdicts(maps: dict[str, dict], universe: list[str], matrix: np.ndarray) -> dict:
    out: dict = {}
    for arm in ("a", "b"):
        pos = _column_pos(arm)
        refs = [MapRef(k, arm, pos) for k in PC.COLUMN_KEYS]
        recs = [maps.get(r.map_id) for r in refs]
        if any(r is None for r in recs):
            out[arm] = {
                "status": "incomplete",
                "missing": [r.map_id for r, m in zip(refs, recs) if m is None],
            }
            continue
        c_35_36 = paired_contrast(recs[1], recs[0], refs[1], refs[0], universe, matrix)
        c_36_38 = paired_contrast(recs[2], recs[1], refs[2], refs[1], universe, matrix)
        c_35_38 = paired_contrast(recs[2], recs[0], refs[2], refs[0], universe, matrix)
        lattice_in = {
            "delta_endpoint_cal": c_35_38["delta_cal"],
            "ci_low_endpoint_cal": c_35_38["ci95_cal"][0],
            "ci_high_endpoint_cal": c_35_38["ci95_cal"][1],
            "delta_step1_cal": c_35_36["delta_cal"],
            "delta_step2_cal": c_36_38["delta_cal"],
        }
        g5_bounded = all(c["g5_bounded"] for c in (c_35_36, c_36_38, c_35_38))
        out[arm] = {
            # A sub-floor intersection (realized q38_27b_b think-residue) is
            # never consumed silently: the arm's status names the degraded
            # regime, and the headline label may only be read from a
            # g5_bounded arm (the caveat travels with the verdict).
            "status": "complete" if g5_bounded else "complete-g5-degraded",
            "g5_bounded": g5_bounded,
            "contrast_36_minus_35": c_35_36,
            "contrast_38_minus_36": c_36_38,
            "contrast_38_minus_35": c_35_38,
            "lattice_inputs": lattice_in,
            "verdict": verdict_lattice(lattice_in),
        }
    return out


def h2_reads(maps: dict[str, dict], universe: list[str], matrix: np.ndarray) -> dict:
    """H2 (plan §3, both registered reads; OLMo pairs NEVER enter either).

    Review-round-2 shape (B4 + E4 adjudication):

    - COMPLETE-CASE per-row intersections: the per-checkpoint gap on EACH
      surface is computed over the rows present in BOTH arms after
      drops/regens (generic: shared test cis from perrow_*; GPQA: shared
      "<qid>_s<seed>" rows from gpqa_perrow_*); intersection sizes reported.
    - PER-CHECKPOINT paired bootstrap: 1,000 draws seed 42, ONE shared
      resample matrix per surface universe (the generic matrix is the §4.4
      column matrix; the GPQA matrix is drawn once over the GPQA row
      universe with the same seed), restricted per draw to the shared rows.
    - PRIMARY = RAW gaps on both surfaces (E4: no GPQA null battery — raw
      minus raw subtracts comparable objects); the CALIBRATED generic gap and
      the length-RESIDUALIZED gaps (generic + GPQA, §6 iii) ride as
      sensitivity fields.
    - FAIL LOUD unless all SEVEN registered checkpoint pairs exist (== 7,
      replacing the round-1 >= 5): a partial H2 read silently changes the
      registered statistic's n.

    (a) gap-vs-AA-rank: Spearman between the per-checkpoint RAW complete-case
        shared-ID generic gap (gap_generic_raw) and the AA pin over the 7
        checkpoints (registered read); the calibrated gap rides ONLY as an
        E4 sensitivity field, never the registered basis.
    (b) surface contrast (2603.05488): Wilcoxon signed-rank over the 7
        per-checkpoint [gap_GPQA − gap_generic] raw deltas (min attainable
        two-sided p at n=7 = 0.0156 — "unresolved" is calibrated, not
        surprising).
    """
    from scipy.stats import spearmanr, wilcoxon

    missing = [
        key
        for key in QWEN_THINKING_KEYS
        if maps.get(MapRef(key, "b", "cot_boundary").map_id) is None
        or maps.get(MapRef(key, "a", "prompt_last").map_id) is None
    ]
    assert not missing, (
        f"H2 requires ALL 7 registered Qwen thinking checkpoint pairs; missing: {missing} "
        "(review round 2: == 7 replaces >= 5 — never run H2 on a partial panel)"
    )
    # One shared GPQA resample matrix (same seed/draw count as the generic one)
    # over the UNION of GPQA row ids across the 14 H2 maps.
    gpqa_universe = sorted(
        {
            rid
            for key in QWEN_THINKING_KEYS
            for arm, pos in (("b", "cot_boundary"), ("a", "prompt_last"))
            for rid in _gpqa_perrow_by_id(maps[MapRef(key, arm, pos).map_id])
        }
    )
    # Draw count MATCHES the caller-supplied generic matrix (production:
    # PC.BOOTSTRAP_DRAWS) — the per-draw surface delta boot_q - boot_g is only
    # defined at equal draw counts.
    gpqa_matrix = _shared_resample_matrix(gpqa_universe, int(matrix.shape[0]), PC.BOOTSTRAP_SEED)

    detail: dict[str, dict] = {}
    gaps_raw, aa_vals, surface_deltas = [], [], []
    for key in QWEN_THINKING_KEYS:
        ref_b, ref_a = MapRef(key, "b", "cot_boundary"), MapRef(key, "a", "prompt_last")
        assert_pair_metadata(ref_b, ref_a)  # same checkpoint id — legal pair
        rec_b, rec_a = maps[ref_b.map_id], maps[ref_a.map_id]
        # Generic surface: complete-case shared test cis + paired bootstrap.
        gap_g, boot_g, n_shared_g = _paired_gap_boot(
            _perrow_by_ci(rec_b), _perrow_by_ci(rec_a), universe, matrix
        )
        # GPQA surface: complete-case shared (qid, seed) rows + paired bootstrap.
        gap_q, boot_q, n_shared_q = _paired_gap_boot(
            _gpqa_perrow_by_id(rec_b), _gpqa_perrow_by_id(rec_a), gpqa_universe, gpqa_matrix
        )
        surface_delta = float(gap_q - gap_g)
        boot_delta = boot_q - boot_g
        cal_b, cal_a = _calibrated(rec_b), _calibrated(rec_a)
        gap_cal = float(cal_b["acc1_cos_calibrated"] - cal_a["acc1_cos_calibrated"])
        rec_entry: dict = {
            "gap_generic_raw": gap_g,
            "gap_generic_raw_ci95": [
                float(np.percentile(boot_g, 2.5)),
                float(np.percentile(boot_g, 97.5)),
            ],
            "n_shared_generic": n_shared_g,
            "gap_gpqa_raw": gap_q,
            "gap_gpqa_raw_ci95": [
                float(np.percentile(boot_q, 2.5)),
                float(np.percentile(boot_q, 97.5)),
            ],
            "n_shared_gpqa": n_shared_q,
            "surface_delta": surface_delta,
            "surface_delta_ci95": [
                float(np.percentile(boot_delta, 2.5)),
                float(np.percentile(boot_delta, 97.5)),
            ],
            # Sensitivity fields (E4): calibrated generic gap + §6 (iii)
            # length-residualized gaps on both surfaces.
            "gap_generic_cal": gap_cal,
        }
        resid_b, resid_a = rec_b["resid"], rec_a["resid"]
        if resid_b is not None and resid_a is not None:
            rec_entry["gap_generic_resid"] = float(
                _acc_at_1(resid_b["resid_knn_test"]["ridge_resid"]["cosine"])
                - _acc_at_1(resid_a["resid_knn_test"]["ridge_resid"]["cosine"])
            )
            gq_b, gq_a = resid_b.get("gpqa_resid"), resid_a.get("gpqa_resid")
            if gq_b is not None and gq_a is not None:
                rec_entry["gap_gpqa_resid"] = float(
                    gq_b["same_question_acc1_cos"] - gq_a["same_question_acc1_cos"]
                )
        gaps_raw.append(gap_g)
        surface_deltas.append(surface_delta)
        aa_vals.append(PC.AA_PIN[key][0])
        detail[key] = rec_entry
    assert len(surface_deltas) == 7, len(surface_deltas)
    # Round 3 (h2-paired-analysis-missing): the registered H2 gap-vs-AA read
    # consumes the RAW complete-case shared-ID generic gaps (gap_generic_raw)
    # — the paired-bootstrap surface this function computes — never the
    # calibrated aggregate gap, which stays a labeled E4 sensitivity field.
    rho, p = spearmanr(aa_vals, gaps_raw)
    stat, wp = wilcoxon(surface_deltas, alternative="two-sided", method="exact")
    return {
        "pairs": detail,
        "n_gap_pairs": len(gaps_raw),
        "min_two_sided_p_at_n7": 0.015625,
        "gap_vs_aa_spearman": {
            "rho": float(rho),
            "p": float(p),
            "n": len(gaps_raw),
            "gap_basis": "raw complete-case shared-ID generic gaps (gap_generic_raw)",
        },
        "sensitivity_gap_semantics": {
            "gap_generic_cal": (
                "aggregate-based (per-arm calibrated acc@1 difference; NOT shared-ID intersected)"
            ),
            "gap_generic_resid": (
                "aggregate-based (per-arm length-residualized acc@1 difference; "
                "NOT shared-ID intersected)"
            ),
            "gap_gpqa_resid": (
                "aggregate-based (per-arm GPQA same-question resid acc@1 difference; "
                "NOT shared-ID intersected)"
            ),
        },
        "surface_wilcoxon": {
            "stat": float(stat),
            "p_two_sided": float(wp),
            "n": len(surface_deltas),
            "statistic_def": "gap_GPQA - gap_generic (RAW gaps, complete-case shared rows)",
        },
        "bootstrap": {
            "draws": int(matrix.shape[0]),
            "seed": PC.BOOTSTRAP_SEED,
            "generic_universe_n": len(universe),
            "gpqa_universe_n": len(gpqa_universe),
        },
    }


def olmo_pair_reads(maps: dict[str, dict]) -> dict:
    """OLMo-R (Instruct prompt_last vs Think pre_think — same prompt-side
    semantics, different checkpoints) and OLMo-P (pre_think vs cot_boundary
    WITHIN the Think checkpoint). Reported per pair (n=2 each), never pooled
    into H2 (plan §4.3 MF2)."""
    out: dict = {"olmo_r": {}, "olmo_p": {}}
    for inst_key, think_key in OLMO_PAIRS:
        ref_i = MapRef(inst_key, "a", "prompt_last")
        ref_tp = MapRef(think_key, "b", "pre_think")
        ref_tc = MapRef(think_key, "b", "cot_boundary")
        assert_pair_metadata(ref_i, ref_tp)  # same prompt-side semantics
        assert_pair_metadata(ref_tp, ref_tc)  # same checkpoint
        rec_i, rec_tp, rec_tc = (maps.get(r.map_id) for r in (ref_i, ref_tp, ref_tc))
        fam = inst_key.rsplit("_", 2)[0]
        if rec_i is not None and rec_tp is not None:
            out["olmo_r"][fam] = {
                "pair": [ref_tp.map_id, ref_i.map_id],
                "delta_cal": _calibrated(rec_tp)["acc1_cos_calibrated"]
                - _calibrated(rec_i)["acc1_cos_calibrated"],
            }
        if rec_tp is not None and rec_tc is not None:
            out["olmo_p"][fam] = {
                "pair": [ref_tc.map_id, ref_tp.map_id],
                "delta_cal": _calibrated(rec_tc)["acc1_cos_calibrated"]
                - _calibrated(rec_tp)["acc1_cos_calibrated"],
            }
    return out


# Registered panel-trend Ns (plan §6): arm (b) = ALL 11 arm-b maps (7 Qwen
# cot_boundary + the two OLMo-Think checkpoints' pre_think AND cot_boundary
# companions); arm (a) = 9 (the two OLMo-Think checkpoints have no arm-a cell;
# the anchor has no AA value). D4 (review round 2): carry ALL registered
# members or fail loud — a silently smaller N changes the registered statistic.
SPEARMAN_REGISTERED_N = {"a": 9, "b": 11}
SPEARMAN_CRITICAL_RHO_ALPHA05 = {"a": 0.68, "b": 0.62}  # two-sided, plan §6


def spearman_vs_capability(maps: dict[str, dict]) -> dict:
    from scipy.stats import spearmanr

    out: dict = {}
    for arm in ("a", "b"):
        xs, ys, names = [], [], []
        for ref in all_maps():
            if ref.arm != arm:
                continue
            pin = PC.AA_PIN.get(ref.model_key, (None,))[0]
            if pin is None:
                continue  # q25_7b anchor: no AA value (registered exclusion, plan §6)
            rec = maps.get(ref.map_id)
            assert rec is not None, (
                f"panel Spearman arm ({arm}): registered map {ref.map_id} missing — carry all "
                f"N={SPEARMAN_REGISTERED_N[arm]} registered members or fail loud (plan §6, D4)"
            )
            xs.append(pin)
            ys.append(_calibrated(rec)["acc1_cos_calibrated"])
            names.append(ref.map_id)
        assert len(xs) == SPEARMAN_REGISTERED_N[arm], (arm, len(xs), names)
        rho, p = spearmanr(xs, ys)
        out[arm] = {
            "n": len(xs),
            "members": names,
            "rho": float(rho),
            "p": float(p),
            "critical_rho_alpha05_two_sided": SPEARMAN_CRITICAL_RHO_ALPHA05[arm],
        }
    return out


# ---------------------------------------------------------------------------
# Conditional GPQA extraction-judge fallback (§4.5; api_dispatch routed)
# ---------------------------------------------------------------------------


JUDGE_PILOT_N = 200  # plan §4.5: ~200-call transport-true pilot at the production instrument
JUDGE_PILOT_PARSE_FAIL_MAX = 0.02  # per-arm parse-fail gate (< 2%)
JUDGE_MAX_TRANSPORT_ROUNDS = 3  # transport-class exhaustions re-driven, never persisted


def _build_extraction_request(item) -> dict:
    return {
        "model": PC.EXTRACTION_JUDGE_MODEL,
        "max_tokens": PC.EXTRACTION_JUDGE_MAX_TOKENS,
        "system": PC.EXTRACTION_JUDGE_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": PC.format_extraction_judge_user(
                    item.payload["question"], item.payload["answer"]
                ),
            }
        ],
    }


def _dispatch_judge_round(items: list, checkpoint_dir: Path) -> dict:
    """One api_dispatch round (Batch API forced — plan §4.5 transport-true).

    The B5 test seam: fakes here must mirror THIS signature (items,
    checkpoint_dir) -> {item_id: DispatchResult}.
    """
    import asyncio

    from explore_persona_space.llm.api_dispatch import dispatch_calls

    return asyncio.run(
        dispatch_calls(
            items,
            model=PC.EXTRACTION_JUDGE_MODEL,
            build_request=_build_extraction_request,
            parse_response=PC.parse_extraction_judgment,
            force_path="batch",
            checkpoint_dir=checkpoint_dir,
        )
    )


def _dispatch_wave(items: list, ckpt_root: Path, wave: str) -> dict:
    """Dispatch + transport RE-DRIVE (B5, review round 2).

    api_dispatch's checkpoint re-serves persisted transport rows on resume
    (its documented caveat), so every re-drive round gets a FRESH checkpoint
    dir; rows still transport-failed after JUDGE_MAX_TRANSPORT_ROUNDS raise —
    transport failures are RETRIED, never persisted as drops (rule 24).
    """
    from explore_persona_space.llm.api_dispatch import RESULT_RATE_LIMITED, RESULT_TRANSPORT

    results: dict = {}
    pending = list(items)
    for rnd in range(JUDGE_MAX_TRANSPORT_ROUNDS):
        res = _dispatch_judge_round(pending, ckpt_root / f"{wave}_round{rnd}")
        redrive = []
        for it in pending:
            r = res[it.item_id]
            if r.category in (RESULT_RATE_LIMITED, RESULT_TRANSPORT):
                redrive.append(it)
            else:
                results[it.item_id] = r
        if not redrive:
            return results
        logger.warning(
            "[i2588] judge %s round %d: %d transport-class rows re-driven (fresh checkpoint)",
            wave,
            rnd,
            len(redrive),
        )
        pending = redrive
    raise RuntimeError(
        f"{len(pending)} judge calls still transport-failed after "
        f"{JUDGE_MAX_TRANSPORT_ROUNDS} re-drive rounds — transport failures are RETRIED, "
        "never persisted as drops (llm-judging.md rule 24). Re-run --judge-fallback when the "
        "API recovers; completed rows resume from their round checkpoints."
    )


def _pilot_gate(pilot_results: dict, out_dir: Path) -> dict:
    """Plan §4.5 registered pilot gates: ZERO stop_reason=="max_tokens" +
    parse-fail rate < 2%, at the EXACT production instrument (Batch path)."""
    n = len(pilot_results)
    n_trunc = sum(1 for r in pilot_results.values() if r.stop_reason == "max_tokens")
    n_parse_fail = sum(1 for r in pilot_results.values() if not r.error and r.result is None)
    parse_fail_rate = n_parse_fail / max(1, n)
    report = {
        "n_pilot": n,
        "n_stop_reason_max_tokens": n_trunc,
        "n_parse_fail": n_parse_fail,
        "parse_fail_rate": parse_fail_rate,
        "gates": {
            "zero_max_tokens": n_trunc == 0,
            "parse_fail_below_2pct": parse_fail_rate < JUDGE_PILOT_PARSE_FAIL_MAX,
        },
    }
    PC.write_json_atomic(out_dir / "gpqa_judge_pilot.json", report)
    if n_trunc > 0 or parse_fail_rate >= JUDGE_PILOT_PARSE_FAIL_MAX:
        raise RuntimeError(
            f"judge PILOT GATE FAIL (plan §4.5): max_tokens truncations={n_trunc} (must be 0), "
            f"parse-fail rate={parse_fail_rate:.3f} (must be < {JUDGE_PILOT_PARSE_FAIL_MAX}) "
            f"over {n} pilot calls — fix the instrument (raise max_tokens / rubric) BEFORE "
            "the production wave; never dispatch the wave past a failed pilot."
        )
    return report


def run_judge_fallback(
    pending_path: Path, parsed_dir: Path, out_path: Path, *, prompts_path: Path | None = None
) -> dict:
    """Judge-extract letters for unparseable GPQA rollouts (trigger: >5%
    extraction failure, recorded pod-side in gpqa_judge_pending.json).

    Routes through llm/api_dispatch.py (Batch API forced — the wave can reach
    ~19k calls worst-case); judge claude-sonnet-4-5-20250929, reason-then-
    extract JSON rubric, max_tokens=1024 (llm-judging.md rule 23 floor).

    Review-round-2 contract (B5): a ~200-call pilot at the EXACT production
    instrument gates the wave (zero max_tokens + parse-fail < 2%);
    stop_reason is captured per verdict; transport-class exhaustions are
    RE-DRIVEN with fresh checkpoint dirs (never persisted as drops);
    malformed judge returns are DROPPED + COUNTED (rule 9, never coerced).

    ``prompts_path`` defaults to the P0-frozen committed gpqa_prompts.json;
    tests inject a fixture (round 3 — the composed-path test must not depend
    on the committed artifact, absent from sparse worktrees).
    """
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    if prompts_path is None:
        prompts_path = _REPO_ROOT / "eval_results" / "issue_2588" / "gpqa_prompts.json"
    assert prompts_path.exists(), (
        f"{prompts_path} missing — run issue2588_p0_preflight.py step 3 (GPQA staging) and "
        "commit the frozen prompts before the judge fallback"
    )
    gpqa = json.loads(prompts_path.read_text(encoding="utf-8"))
    from explore_persona_space.llm.api_dispatch import DispatchItem

    q_by_id = {q["qid"]: q for q in gpqa["prompts"]}
    rows_by_id: dict[str, dict] = {}
    for f in sorted(parsed_dir.glob("gpqa_s*.jsonl")):
        for r in PC.read_jsonl(f):
            rows_by_id[r["row_id"]] = r
    # Round 4 (judge-fallback-staged-row-coverage): the staged parsed rows +
    # frozen prompts MUST cover every pending row BEFORE any DispatchItem is
    # built — a partial/stale parsed mirror otherwise dies as a bare KeyError
    # (rows_by_id[p["row_id"]] / q_by_id[p["qid"]]) inside the registered
    # fallback, the third components-green/phase-dead instance on this task.
    missing_rows = [p["row_id"] for p in pending["rows"] if p["row_id"] not in rows_by_id]
    missing_qids = sorted({p["qid"] for p in pending["rows"] if p["qid"] not in q_by_id})
    assert not missing_rows and not missing_qids, (
        f"judge fallback: staged inputs do not cover pending rows — "
        f"{len(missing_rows)} row_id(s) missing from {parsed_dir} "
        f"(first 5: {missing_rows[:5]}); {len(missing_qids)} qid(s) missing from "
        f"{prompts_path} (first 5: {missing_qids[:5]}) — partial/stale staging; "
        "re-run --harvest at the pinned revision before dispatching the judge wave"
    )
    items = []
    for p in pending["rows"]:
        r = rows_by_id[p["row_id"]]
        s, e = r["ans_char_span"]
        items.append(
            DispatchItem(
                item_id=p["row_id"],
                payload={
                    "question": q_by_id[p["qid"]]["prompt"],
                    "answer": r["text"][s:e],
                    "gold": p["gold"],
                },
            )
        )

    ckpt_root = out_path.parent / "judge_checkpoint"
    pilot_items = items[: min(JUDGE_PILOT_N, len(items))]
    pilot_results = _dispatch_wave(pilot_items, ckpt_root, "pilot")
    pilot_report = _pilot_gate(pilot_results, out_path.parent)
    rest = items[len(pilot_items) :]
    wave_results = _dispatch_wave(rest, ckpt_root, "wave") if rest else {}
    results = {**pilot_results, **wave_results}

    verdicts: dict = {}
    counts = {
        "n_correct": 0,
        "n_unparseable": 0,
        "n_malformed_dropped": 0,
        "n_truncated": 0,
        "n_error_dropped": 0,
    }
    gold_by_id = {p["row_id"]: p["gold"] for p in pending["rows"]}
    for item_id, res in results.items():
        ent: dict = {"stop_reason": res.stop_reason}
        if res.error:
            # Terminal non-transport failures (bad request / empty response):
            # dropped + counted — transport-class rows can never reach here
            # (_dispatch_wave re-drives or raises).
            counts["n_error_dropped"] += 1
            ent.update(letter=None, disposition=f"{res.category}:dropped ({res.reason})")
        elif res.result is None:
            counts["n_malformed_dropped"] += 1
            counts["n_truncated"] += int(res.stop_reason == "max_tokens")
            ent.update(letter=None, disposition="malformed:dropped (rule 9, never coerced)")
        else:
            letter = res.result
            ok = letter == gold_by_id[item_id]
            counts["n_correct"] += int(ok)
            counts["n_unparseable"] += int(letter == "UNPARSEABLE")
            ent.update(letter=letter, correct=bool(ok))
        verdicts[item_id] = ent
    rec = {
        "meta": {
            "issue": PC.TASK_ID,
            "judge_model": PC.EXTRACTION_JUDGE_MODEL,
            "judge_max_tokens": PC.EXTRACTION_JUDGE_MAX_TOKENS,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "n_items": len(items),
        **counts,
        "n_transport_persisted": 0,  # by construction: re-driven or raised
        "pilot": pilot_report,
        "verdicts": verdicts,
    }
    PC.write_json_atomic(out_path, rec)
    return rec


def merged_behavioral(rec: dict, map_id: str) -> dict | None:
    """Deterministically fold gpqa_judge_verdicts.json into the behavioral
    metrics (B5): judge-corrected accuracy from INTEGER counts; a
    flagged-pending map WITHOUT verdicts FAILS LOUD — the trend is never
    assembled on uncorrected behavioral metrics."""
    gt = rec["gpqa_transfer"]
    if gt is None:
        return None
    beh = dict(gt["behavioral"])
    if not beh.get("judge_fallback_flagged"):
        return beh
    pend, verd = rec["judge_pending"], rec["judge_verdicts"]
    assert pend is not None, f"{map_id}: judge_fallback_flagged but gpqa_judge_pending.json absent"
    if verd is None:
        raise RuntimeError(
            f"{map_id}: GPQA judge fallback FLAGGED "
            f"(frac_unparseable={beh.get('frac_unparseable'):.3f}) but "
            "gpqa_judge_verdicts.json is ABSENT — run issue2588_trend.py --judge-fallback "
            "--pending <cell_dir>/gpqa_judge_pending.json --parsed-dir <cell_dir>/parsed "
            "for this cell first (--harvest stages <cell_dir>/parsed/gpqa_s*.jsonl and logs "
            "the exact command, plan §4.5); the trend never assembles on uncorrected metrics."
        )
    gold = {r["row_id"]: r["gold"] for r in pend["rows"]}
    n_extra_correct = sum(
        1 for rid, v in verd["verdicts"].items() if v.get("letter") == gold.get(rid)
    )
    n_still_unparseable = sum(
        1 for v in verd["verdicts"].values() if v.get("letter") in (None, "UNPARSEABLE")
    )
    total = int(beh["n_rollouts"])
    n_correct0 = int(beh["n_correct"])
    beh["acc_judge_corrected"] = (n_correct0 + n_extra_correct) / max(1, total)
    beh["n_judge_corrected"] = n_extra_correct
    beh["frac_unparseable_after_judge"] = n_still_unparseable / max(1, total)
    beh["judge_counts"] = {
        k: verd[k]
        for k in ("n_items", "n_correct", "n_unparseable", "n_malformed_dropped", "n_error_dropped")
        if k in verd
    }
    return beh


# ---------------------------------------------------------------------------
# Harvest helpers (round 3)
# ---------------------------------------------------------------------------


def _resolve_harvest_revision(explicit: str | None) -> str:
    """Round 3 (p3-harvest-missing): hub.stage_hub_prefix resolves
    revision=None to a sha PER CALL, so the fits + nulls staging calls (and
    the judge-fallback input staging) could straddle a concurrent upload and
    mirror incoherent generations. Resolve main -> ONE sha up front and
    thread it into EVERY staging call; the realized sha is logged and
    persisted in the summary meta (harvest_revision_resolved)."""
    if explicit:
        return explicit
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    sha = hub.retry_transient(
        lambda: HfApi().repo_info(PC.HF_DATA_REPO, repo_type="dataset").sha,
        what="resolve --harvest revision (data-repo main sha)",
    )
    assert isinstance(sha, str) and len(sha) >= 7, f"repo_info returned no usable sha: {sha!r}"
    return sha


def _stage_judge_fallback_inputs(fits_root: Path, revision: str) -> dict[str, list[Path]]:
    """Round 3 (judge-fallback-unintegrated, B5 staging route): --harvest
    mirrors only the fits/ + nulls/ prefixes, while run_judge_fallback
    consumes the pod-side parsed GPQA rollouts (gpqa_s*.jsonl, uploaded to
    {cell.hf_prefix}/parsed/). For every harvested cell whose
    gpqa_judge_pending.json is present, stage exactly the gpqa_s*.jsonl
    parsed files into <cell_dir>/parsed/ (beside the pending file) at the
    SAME resolved harvest revision, and log the composed per-cell
    --judge-fallback command. Fail-loud: a flagged cell with zero parsed
    GPQA files on the Hub is an error, never a silent skip. Returns
    {cell_key: [staged paths]} for the flagged cells."""
    from fnmatch import fnmatch

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    staged: dict[str, list[Path]] = {}
    for cell in PC.all_cells():
        pending = fits_root / cell.key / "gpqa_judge_pending.json"
        if not pending.exists():
            continue
        prefix = f"{cell.hf_prefix}/parsed"
        remote = hub.list_hf_files_under_path(
            api, PC.HF_DATA_REPO, prefix, repo_type="dataset", revision=revision
        )
        wanted = sorted(p for p in remote if fnmatch(p.rsplit("/", 1)[-1], "gpqa_s*.jsonl"))
        assert wanted, (
            f"judge-fallback staging: {cell.key} flagged (gpqa_judge_pending.json present) but "
            f"no gpqa_s*.jsonl under {PC.HF_DATA_REPO}/{prefix}@{revision[:12]} — the pod-side "
            "upload-raw phase must persist parsed GPQA rollouts before the VM judge fallback"
        )
        parsed_dir = pending.parent / "parsed"
        for rp in wanted:
            hub.stage_hub_file(
                PC.HF_DATA_REPO,
                rp,
                parsed_dir / rp.rsplit("/", 1)[-1],
                repo_type="dataset",
                revision=revision,
            )
        staged[cell.key] = [parsed_dir / rp.rsplit("/", 1)[-1] for rp in wanted]
        # Round 4 (judge-fallback-staged-row-coverage): presence of SOME
        # gpqa_s*.jsonl is not coverage — validate that the staged rows cover
        # every pending row_id AT STAGING TIME, so --harvest fails loud here
        # (naming the missing ids) instead of certifying the cell "runnable"
        # for a KeyError inside run_judge_fallback.
        pending_rows = json.loads(pending.read_text(encoding="utf-8"))["rows"]
        staged_ids = {r["row_id"] for fp in staged[cell.key] for r in PC.read_jsonl(fp)}
        missing = [p["row_id"] for p in pending_rows if p["row_id"] not in staged_ids]
        assert not missing, (
            f"judge-fallback staging: {cell.key} staged parsed rows cover only "
            f"{len(pending_rows) - len(missing)}/{len(pending_rows)} pending row_ids "
            f"(first 5 missing: {missing[:5]}) under {parsed_dir} @ {revision[:12]} — "
            "partial/stale parsed upload; the pod-side upload-raw phase must persist "
            "ALL parsed GPQA rollouts the pending file references"
        )
        logger.info(
            "[i2588] judge-fallback inputs staged for %s (%d files) — run: "
            "uv run python scripts/issue2588_trend.py --judge-fallback "
            "--pending %s --parsed-dir %s",
            cell.key,
            len(wanted),
            pending,
            parsed_dir,
        )
    return staged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--fits-dir", type=Path, default=_REPO_ROOT / "eval_results" / "issue_2588" / "fits"
    )
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--judge-fallback", action="store_true")
    ap.add_argument("--pending", type=Path, help="gpqa_judge_pending.json (judge fallback)")
    ap.add_argument("--parsed-dir", type=Path, help="parsed rollouts dir (judge fallback)")
    ap.add_argument(
        "--harvest",
        action="store_true",
        help="C5: stage the fits/nulls HF prefixes (revision-pinned bulk mirror via "
        "hub.stage_hub_prefix) into <fits-dir parent>/hub_mirror before analysis; "
        "asserts all 21 registered maps load with full schema",
    )
    ap.add_argument(
        "--harvest-revision",
        default=None,
        help="HF data-repo revision pin for --harvest (default: main is resolved to ONE sha "
        "via HfApi().repo_info up front and threaded into every staging call — the realized "
        "sha is logged and persisted as meta.harvest_revision_resolved)",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        from scipy.stats import spearmanr, wilcoxon  # noqa: F401

        from explore_persona_space.llm.api_dispatch import (  # noqa: F401
            RESULT_RATE_LIMITED,
            RESULT_TRANSPORT,
            DispatchItem,
            dispatch_calls,
        )
        from fnmatch import fnmatch  # noqa: F401

        from huggingface_hub import HfApi  # noqa: F401

        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            list_hf_files_under_path,
            retry_transient,
            stage_hub_file,
            stage_hub_prefix,
        )

        print("[import-check] OK")
        return 0
    if args.judge_fallback:
        assert args.pending and args.parsed_dir, "--judge-fallback needs --pending + --parsed-dir"
        # Verdicts land BESIDE the pending file (the harvested cell fits dir),
        # where the trend assembly's _load_map reads them back (B5).
        rec = run_judge_fallback(
            args.pending, args.parsed_dir, args.pending.parent / "gpqa_judge_verdicts.json"
        )
        logger.info("[i2588] judge fallback: %s", {k: v for k, v in rec.items() if k != "verdicts"})
        return 0

    harvest_rev: str | None = None
    if args.harvest:
        import shutil

        from explore_persona_space.orchestrate import hub

        # Round 3 (p3-harvest-missing): ONE resolved sha threads through BOTH
        # prefix stages + the judge-fallback input staging — never per-call
        # revision=None resolution, which can straddle a concurrent upload.
        harvest_rev = _resolve_harvest_revision(args.harvest_revision)
        logger.info("[i2588] harvest revision resolved: %s", harvest_rev)
        mirror = args.fits_dir.parent / "hub_mirror"
        staged: list[Path] = []
        for pfx in (f"{PC.PANEL_PREFIX}/fits", f"{PC.PANEL_PREFIX}/nulls"):
            staged += hub.stage_hub_prefix(
                PC.HF_DATA_REPO,
                pfx,
                mirror,
                repo_type="dataset",
                revision=harvest_rev,
            )
        # Uploads split nulls_* to the nulls/ prefix; _load_map reads ONE cell
        # dir — fold the nulls mirror into the fits mirror per cell.
        fits_root = mirror / PC.PANEL_PREFIX / "fits"
        nulls_root = mirror / PC.PANEL_PREFIX / "nulls"
        if nulls_root.is_dir():
            for f in sorted(nulls_root.rglob("*.json")):
                dest = fits_root / f.relative_to(nulls_root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
        args.fits_dir = fits_root
        # Round 3 (judge-fallback-unintegrated): stage the parsed GPQA
        # rollouts every flagged cell's --judge-fallback run consumes.
        judge_staged = _stage_judge_fallback_inputs(fits_root, harvest_rev)
        logger.info(
            "[i2588] harvested %d files (revision pin: %s; judge-fallback cells staged: %d) -> %s",
            len(staged),
            harvest_rev,
            len(judge_staged),
            fits_root,
        )

    maps: dict[str, dict] = {}
    for ref in all_maps():
        rec = _load_map(args.fits_dir, ref)
        if rec is not None:
            maps[ref.map_id] = rec
    logger.info("[i2588] loaded %d/21 maps from %s", len(maps), args.fits_dir)
    assert maps, f"no fit artifacts under {args.fits_dir}"
    if args.harvest:
        # C5 schema/count validation: ALL 21 registered maps, each with the
        # full artifact set the analysis consumes.
        missing_maps = [r.map_id for r in all_maps() if r.map_id not in maps]
        assert not missing_maps, (
            f"--harvest expected all 21 registered maps; missing {missing_maps}"
        )
        for mid, rec in maps.items():
            gaps = [
                k
                for k in ("nulls", "perrow", "gpqa_transfer", "gpqa_perrow", "resid")
                if rec[k] is None
            ]
            assert not gaps, f"--harvest: map {mid} missing artifact kinds {gaps}"

    # Test-ci universe: union of perrow cis (the frozen test_1000 grid).
    universe = sorted({ci for rec in maps.values() if rec["perrow"] for ci in _perrow_by_ci(rec)})
    matrix = _shared_resample_matrix(universe, PC.BOOTSTRAP_DRAWS, PC.BOOTSTRAP_SEED)

    summary = {
        "meta": {
            "issue": PC.TASK_ID,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fits_dir": str(args.fits_dir),
            "harvest": bool(args.harvest),
            "harvest_revision": args.harvest_revision,
            # Round 3 (p3-harvest-missing): the REALIZED sha every staging
            # call used (== harvest_revision when explicitly pinned).
            "harvest_revision_resolved": harvest_rev,
            "n_maps_loaded": len(maps),
            "bootstrap": {
                "draws": PC.BOOTSTRAP_DRAWS,
                "seed": PC.BOOTSTRAP_SEED,
                "shared_matrix": True,
            },
            "nulls": {"draws": PC.PERM_DRAWS, "seed": PC.PERM_SEED, "advisory_only": True},
        },
        "per_map": {mid: _calibrated(rec) for mid, rec in maps.items()},
        "gpqa_transfer": {
            mid: {**rec["gpqa_transfer"], "behavioral": merged_behavioral(rec, mid)}
            for mid, rec in maps.items()
            if rec["gpqa_transfer"] is not None
        },
        "resid": {
            mid: {
                "resid_test_r2": rec["resid"]["resid_test_r2"],
                "length_only_test_r2": rec["resid"]["length_only_test_r2"],
                "resid_acc1_cos": _acc_at_1(
                    rec["resid"]["resid_knn_test"]["ridge_resid"]["cosine"]
                ),
                "length_only_acc1_cos": _acc_at_1(
                    rec["resid"]["length_only_knn_test"]["length_only"]["cosine"]
                ),
                "gpqa_resid_same_q_acc1": (rec["resid"].get("gpqa_resid") or {}).get(
                    "same_question_acc1_cos"
                ),
            }
            for mid, rec in maps.items()
            if rec["resid"] is not None
        },
        "column_verdicts": column_verdicts(maps, universe, matrix),
        "h2_qwen_thinking": h2_reads(maps, universe, matrix),
        "olmo_pairs": olmo_pair_reads(maps),
        "spearman_vs_aa_capability": spearman_vs_capability(maps),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    PC.write_json_atomic(args.out, summary)
    logger.info("[phase=done] trend summary written -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
