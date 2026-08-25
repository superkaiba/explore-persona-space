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
    for kind in ("nulls", "perrow", "gpqa_transfer", "resid"):
        p = cell_dir / f"{kind}_{ref.pos}.json"
        rec[kind] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return rec


def _acc1_at_star(fits: dict) -> float:
    star = str(fits["layer_star"])
    acc = fits["layers"][star]["knn_test"]["ridge"]["cosine"]["acc_at_k"]
    return float(acc.get("1", acc.get(1)))


def _calibrated(rec: dict) -> dict:
    obs = _acc1_at_star(rec["fits"])
    nulls = rec["nulls"]
    mu = float(nulls["null_mean_acc1_cos"]) if nulls else None
    sd = float(nulls["null_sd_acc1_cos"]) if nulls else None
    n_draws = int(nulls["perm_draws"]) if nulls else None
    return {
        "acc1_cos_at_star": obs,
        "null_mean": mu,
        "null_sd": sd,
        "perm_draws": n_draws,
        "acc1_cos_calibrated": (obs - mu) if mu is not None else None,
        "layer_star": int(rec["fits"]["layer_star"]),
        "ceiling_two_draw": rec["fits"].get("ceiling_two_draw_at_star"),
    }


def _perrow_by_ci(rec: dict) -> dict[str, int]:
    pr = rec["perrow"]
    assert pr is not None, "perrow hits missing — the paired bootstrap needs them"
    return {
        str(r).rsplit("_", 1)[1]: int(h) for r, h in zip(pr["row_ids"], pr["hit1_cos"], strict=True)
    }


# ---------------------------------------------------------------------------
# Paired bootstrap (one shared resample matrix; plan §4.4)
# ---------------------------------------------------------------------------


def _shared_resample_matrix(universe: list[str], draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(universe)
    return rng.integers(0, n, size=(draws, n))


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
    assert len(shared) >= 0.9 * len(universe), (
        f"paired contrast {ref_a.map_id} vs {ref_b.map_id}: only {len(shared)}/"
        f"{len(universe)} shared rows — drop residue exceeds the G5-bounded regime"
    )
    in_shared = np.array([ci in hits_a and ci in hits_b for ci in universe])
    da = np.array([hits_a.get(ci, 0) for ci in universe], dtype=np.float64)
    db = np.array([hits_b.get(ci, 0) for ci in universe], dtype=np.float64)
    diff = da - db
    raw_delta = float(diff[in_shared].mean())
    boot = np.empty(matrix.shape[0])
    for i in range(matrix.shape[0]):
        idx = matrix[i]
        keep = in_shared[idx]
        assert keep.any(), "bootstrap draw with zero shared rows"
        boot[i] = diff[idx][keep].mean()
    cal_a, cal_b = _calibrated(rec_a), _calibrated(rec_b)
    delta_null = cal_a["null_mean"] - cal_b["null_mean"]
    boot_cal = boot - delta_null
    n_draws = cal_a["perm_draws"]
    se_shift = float(np.sqrt(cal_a["null_sd"] ** 2 + cal_b["null_sd"] ** 2) / np.sqrt(n_draws))
    return {
        "pair": [ref_a.map_id, ref_b.map_id],
        "n_shared_rows": len(shared),
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
        out[arm] = {
            "status": "complete",
            "contrast_36_minus_35": c_35_36,
            "contrast_38_minus_36": c_36_38,
            "contrast_38_minus_35": c_35_38,
            "lattice_inputs": lattice_in,
            "verdict": verdict_lattice(lattice_in),
        }
    return out


def h2_reads(maps: dict[str, dict]) -> dict:
    """H2 (plan §3, both registered reads; OLMo pairs NEVER enter either).

    (a) gap-vs-AA-rank: Spearman between the per-checkpoint CALIBRATED arm gap
        (arm-b cot_boundary minus arm-a prompt_last, generic surface) and the
        AA capability pin, over the 7 Qwen thinking checkpoints.
    (b) surface contrast (2603.05488): Wilcoxon signed-rank over the 7
        checkpoints of [gap_GPQA - gap_generic] (n=7; minimum attainable
        two-sided p = 0.0156 — an "unresolved" outcome is calibrated, not
        surprising). gap_generic uses RAW test acc@1 and gap_GPQA RAW
        same-question acc@1 so the two surfaces subtract comparable objects
        (no GPQA null battery is registered); the calibrated-generic-gap
        variant is reported as a sensitivity field.
    """
    from scipy.stats import spearmanr, wilcoxon

    detail: dict[str, dict | str] = {}
    gaps_cal, aa_vals, surface_deltas = [], [], []
    for key in QWEN_THINKING_KEYS:
        ref_b, ref_a = MapRef(key, "b", "cot_boundary"), MapRef(key, "a", "prompt_last")
        assert_pair_metadata(ref_b, ref_a)  # same checkpoint id — legal pair
        rec_b, rec_a = maps.get(ref_b.map_id), maps.get(ref_a.map_id)
        if rec_b is None or rec_a is None:
            detail[key] = "missing"
            continue
        cal_b, cal_a = _calibrated(rec_b), _calibrated(rec_a)
        gap_cal = cal_b["acc1_cos_calibrated"] - cal_a["acc1_cos_calibrated"]
        gap_raw = cal_b["acc1_cos_at_star"] - cal_a["acc1_cos_at_star"]
        rec_entry: dict = {"gap_generic_cal": float(gap_cal), "gap_generic_raw": float(gap_raw)}
        gpqa_b, gpqa_a = rec_b["gpqa_transfer"], rec_a["gpqa_transfer"]
        if gpqa_b is not None and gpqa_a is not None:
            gap_gpqa = gpqa_b["same_question_acc1_cos"] - gpqa_a["same_question_acc1_cos"]
            rec_entry["gap_gpqa_raw"] = float(gap_gpqa)
            rec_entry["surface_delta"] = float(gap_gpqa - gap_raw)
            surface_deltas.append(gap_gpqa - gap_raw)
        gaps_cal.append(gap_cal)
        aa_vals.append(PC.AA_PIN[key][0])
        detail[key] = rec_entry
    out: dict = {"pairs": detail, "n_gap_pairs": len(gaps_cal), "min_two_sided_p_at_n7": 0.015625}
    if len(gaps_cal) >= 3:
        rho, p = spearmanr(aa_vals, gaps_cal)
        out["gap_vs_aa_spearman"] = {"rho": float(rho), "p": float(p), "n": len(gaps_cal)}
    else:
        out["gap_vs_aa_spearman"] = None
    if len(surface_deltas) >= 5:
        stat, p = wilcoxon(surface_deltas, alternative="two-sided", mode="exact")
        out["surface_wilcoxon"] = {
            "stat": float(stat),
            "p_two_sided": float(p),
            "n": len(surface_deltas),
            "statistic_def": "gap_GPQA - gap_generic (raw gaps)",
        }
    else:
        out["surface_wilcoxon"] = {
            "stat": None,
            "p_two_sided": None,
            "n": len(surface_deltas),
            "note": "fewer than 5 realized pairs — Wilcoxon not run",
        }
    return out


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


def spearman_vs_capability(maps: dict[str, dict]) -> dict:
    from scipy.stats import spearmanr

    out: dict = {}
    for arm in ("a", "b"):
        xs, ys, names = [], [], []
        for m in PC.PANEL.values():
            if arm not in m.arms:
                continue
            pin = PC.AA_PIN.get(m.key, (None,))[0]
            if pin is None:
                continue  # q25_7b: no AA value (recorded at P0), excluded here
            pos = _column_pos(arm) if m.thinking or arm == "a" else None
            if m.family == "olmo_think":
                pos = "cot_boundary"
            rec = maps.get(f"{m.key}_{arm}.{pos}")
            if rec is None:
                continue
            xs.append(pin)
            ys.append(_calibrated(rec)["acc1_cos_calibrated"])
            names.append(m.key)
        if len(xs) >= 3:
            rho, p = spearmanr(xs, ys)
            out[arm] = {"n": len(xs), "members": names, "rho": float(rho), "p": float(p)}
        else:
            out[arm] = {"n": len(xs), "members": names, "rho": None, "p": None}
    return out


# ---------------------------------------------------------------------------
# Conditional GPQA extraction-judge fallback (§4.5; api_dispatch routed)
# ---------------------------------------------------------------------------


def run_judge_fallback(pending_path: Path, parsed_dir: Path, out_path: Path) -> dict:
    """Judge-extract letters for unparseable GPQA rollouts (trigger: >5%
    extraction failure, recorded pod-side in gpqa_judge_pending.json).

    Routes through llm/api_dispatch.py (Batch API forced — the wave can reach
    ~19k calls worst-case); judge claude-sonnet-4-5-20250929, reason-then-
    extract JSON rubric, max_tokens=1024 (llm-judging.md rule 23 floor);
    malformed judge returns are DROPPED (scored incorrect), never coerced."""
    import asyncio

    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    gpqa = json.loads(
        (_REPO_ROOT / "eval_results" / "issue_2588" / "gpqa_prompts.json").read_text(
            encoding="utf-8"
        )
    )
    q_by_id = {q["qid"]: q for q in gpqa["prompts"]}
    rows_by_id: dict[str, dict] = {}
    for f in sorted(parsed_dir.glob("gpqa_s*.jsonl")):
        for r in PC.read_jsonl(f):
            rows_by_id[r["row_id"]] = r
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

    def build_request(item: DispatchItem) -> dict:
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

    results = asyncio.run(
        dispatch_calls(
            items,
            model=PC.EXTRACTION_JUDGE_MODEL,
            build_request=build_request,
            parse_response=PC.parse_extraction_judgment,
            force_path="batch",
            checkpoint_dir=out_path.parent / "judge_checkpoint",
        )
    )
    verdicts, n_correct, n_unparseable, n_error = {}, 0, 0, 0
    gold_by_id = {p["row_id"]: p["gold"] for p in pending["rows"]}
    for item_id, res in results.items():
        if res.error or res.result is None:
            n_error += 1
            verdicts[item_id] = {"letter": None, "disposition": f"error:{res.reason}"}
            continue
        letter = res.result
        ok = letter == gold_by_id[item_id]
        n_correct += int(ok)
        n_unparseable += int(letter == "UNPARSEABLE")
        verdicts[item_id] = {"letter": letter, "correct": bool(ok)}
    rec = {
        "meta": {
            "issue": PC.TASK_ID,
            "judge_model": PC.EXTRACTION_JUDGE_MODEL,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "n_items": len(items),
        "n_correct": n_correct,
        "n_unparseable": n_unparseable,
        "n_transport_or_error": n_error,
        "verdicts": verdicts,
    }
    PC.write_json_atomic(out_path, rec)
    return rec


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
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        from scipy.stats import spearmanr, wilcoxon  # noqa: F401

        from explore_persona_space.llm.api_dispatch import (  # noqa: F401
            DispatchItem,
            dispatch_calls,
        )

        print("[import-check] OK")
        return 0
    if args.judge_fallback:
        assert args.pending and args.parsed_dir, "--judge-fallback needs --pending + --parsed-dir"
        rec = run_judge_fallback(
            args.pending, args.parsed_dir, args.out.parent / "gpqa_judge_verdicts.json"
        )
        logger.info("[i2588] judge fallback: %s", {k: v for k, v in rec.items() if k != "verdicts"})
        return 0

    maps: dict[str, dict] = {}
    for ref in all_maps():
        rec = _load_map(args.fits_dir, ref)
        if rec is not None:
            maps[ref.map_id] = rec
    logger.info("[i2588] loaded %d/21 maps from %s", len(maps), args.fits_dir)
    assert maps, f"no fit artifacts under {args.fits_dir}"

    # Test-ci universe: union of perrow cis (the frozen test_1000 grid).
    universe = sorted({ci for rec in maps.values() if rec["perrow"] for ci in _perrow_by_ci(rec)})
    matrix = _shared_resample_matrix(universe, PC.BOOTSTRAP_DRAWS, PC.BOOTSTRAP_SEED)

    summary = {
        "meta": {
            "issue": PC.TASK_ID,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fits_dir": str(args.fits_dir),
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
            mid: rec["gpqa_transfer"]
            for mid, rec in maps.items()
            if rec["gpqa_transfer"] is not None
        },
        "resid": {
            mid: {k: rec["resid"][k] for k in ("resid_test_r2", "length_only_test_r2")}
            for mid, rec in maps.items()
            if rec["resid"] is not None
        },
        "column_verdicts": column_verdicts(maps, universe, matrix),
        "h2_qwen_thinking": h2_reads(maps),
        "olmo_pairs": olmo_pair_reads(maps),
        "spearman_vs_aa_capability": spearman_vs_capability(maps),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    PC.write_json_atomic(args.out, summary)
    logger.info("[phase=done] trend summary written -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
