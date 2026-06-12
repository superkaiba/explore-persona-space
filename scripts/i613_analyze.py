#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" + Greek delta are intentional
"""Task #613 — alive-negatives flag A/B analysis (off-pod, VM, CPU).

Computes the pre-registered reads R1-R5 (plan §6) from the NEW flag-on JSONs
(``eval_results/issue_613/flagon_ab/flagon_200p800n_seed*/``) + the COMMITTED
flag-off comparison JSONs
(``eval_results/issue_601/phase2/dense_200p800n_seed*/``) + the sep-plain /
sep-marker slot reads (``eval_results/issue_613/slotread/*_seed*/``), and
emits ``eval_results/issue_613/analysis/ab_verdict.json``.

Exact numeric centers are RE-LOADED from the committed JSONs at analysis time
(the parent's v3 §C pattern) — the frozen quantities are the RULES (the
±5.58-nat co-landing band, the 1.5-nat clamp bar, the 1e-3-nat R1 liveness
floor), never re-fit numbers. The flag-off comparator's own clamp gaps are
computed from its committed dense terminal (1.25 / 1.03 — the round-1 critique
correction: NOT the 0.75-1.21 range, which belongs to #601's count cells).

Usage:
    uv run python scripts/i613_analyze.py \
        [--flagon-root eval_results/issue_613/flagon_ab] \
        [--flagoff-root eval_results/issue_601/phase2] \
        [--slotread-root eval_results/issue_613/slotread] \
        [--out eval_results/issue_613/analysis/ab_verdict.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

log = logging.getLogger("i613.analyze")

SEEDS = (42, 137)
FLAGON_CELL = "flagon_200p800n"
FLAGOFF_CELL = "dense_200p800n"

# Frozen decision constants (plan §6 — rules, never recomputed from new data).
FROZEN_SOURCE_BAND_NATS = 5.58  # 2x the parent's largest within-cell seed gap
CLAMP_BAR_NATS = 1.5  # the parent's registered clamp criterion
R1_LIVENESS_FLOOR_NATS = 1e-3  # >= ~4.5x the flag-off all-cell trailing max
R1_GRAY_LOW_NATS = 2.2e-4  # flag-off all-cell trailing-CE max (weakly-live floor)
RISE_THEN_DROP_MIN_NATS = 1.0  # trained-neg peak - terminal (the #471 shape)
MARGIN_TWIN_MIN_LOGIT = 1.0  # R2 EOS-margin twin rule floor
SATURATION_LOGP_TOL = 0.1  # trained log P within 0.1 nat of 0 ...
SATURATION_EMISSION_MIN = 0.92  # ... AND on-policy argmax emission >= 0.92
SLOT_LADDER_STEPS = (1, 5, 10, 20, 32, 45, 63)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required input missing: {path}")
    return json.loads(path.read_text())


def _terminal(checkpoints: list[dict]) -> dict:
    term = [c for c in checkpoints if float(c["frac"]) == 1.0]
    if not term:
        raise RuntimeError("no terminal (frac=1.0) checkpoint in payload")
    return term[0]


def _onpolicy_terminal_source(traj: dict) -> dict:
    """Terminal on-policy source read (the R2 PRIMARY quantities)."""
    src = _terminal(traj["checkpoints"])["source_self"]
    return {
        "delta_g": float(src["delta_g_mean"]),
        "g_logp": float(src["g_logp_mean"]),
        "emission_p": float(src["emission_p"]),
        "margin": float(
            (src["z_marker_g_mean"] - src["z_eos_g_mean"])
            - (src["z_marker_b_mean"] - src["z_eos_b_mean"])
        ),
        "saturated": bool(
            abs(float(src["g_logp_mean"])) <= SATURATION_LOGP_TOL
            and float(src["emission_p"]) >= SATURATION_EMISSION_MIN
        ),
    }


def _channel_means(ck: dict, dense: dict) -> dict:
    """Per-channel (source / trained-neg / bystander) means for ONE checkpoint."""
    qs = dense["eval_questions"]
    trained = dense["trained_negatives"]
    bystanders = [b for b in dense["bystander_panel"] if b not in trained]
    reads = ck["reads"]

    def _mean(personas: list[str], key: str) -> float:
        return mean(reads[p][q][key] for p in personas for q in qs)

    out: dict = {"step": ck["step"], "frac": ck["frac"]}
    for label, group in (
        ("source", [dense["source"]]),
        ("trained_neg", trained),
        ("bystander", bystanders),
    ):
        out[label] = {
            "delta_g": _mean(group, "delta_g"),
            "delta_z_marker": _mean(group, "delta_z_marker"),
            "delta_margin": _mean(group, "delta_margin"),
            "delta_z_eos": _mean(group, "z_eos_g") - _mean(group, "z_eos_b"),
        }
    return out


def dense_series(dense: dict) -> list[dict]:
    """Channel-mean series over every checkpoint, sorted by step."""
    rows = [_channel_means(ck, dense) for ck in dense["checkpoints"]]
    rows.sort(key=lambda r: (r["step"] is None, r["step"]))
    return rows


def r1_liveness(rowtype_on: dict[int, dict], rowtype_off: dict[int, dict]) -> dict:
    """R1 — manipulation check: the relocated negative loss channel is live."""
    per_seed: dict[str, dict] = {}
    for seed, rt in rowtype_on.items():
        step1 = [r for r in rt["records"] if r["step"] == 1]
        if not step1 or step1[0].get("neg_slot_ce") is None:
            raise RuntimeError(f"flag-on seed {seed}: no step-1 neg_slot CE in rowtype_ce.json")
        ce1 = float(step1[0]["neg_slot_ce"])
        if ce1 >= R1_LIVENESS_FLOOR_NATS:
            label = "live"
        elif ce1 >= R1_GRAY_LOW_NATS:
            label = "weakly-live"
        else:
            label = "dead"
        per_seed[f"seed{seed}"] = {
            "step1_neg_slot_ce": ce1,
            "neg_slot_ce_base": rt.get("neg_slot_ce_base"),
            "neg_slot_ce_series": rt.get("neg_slot_ce"),
            "classification": label,
        }
    # Comparator context: the flag-off arm's own trailing-channel CE max.
    off_trailing_max = {
        f"seed{seed}": max(v for v in rt["neg_trailing_ce"] if v is not None)
        for seed, rt in rowtype_off.items()
    }
    labels = [v["classification"] for v in per_seed.values()]
    return {
        "rule": f"step-1 flag-on neg_slot CE >= {R1_LIVENESS_FLOOR_NATS} nats in both seeds "
        f"(gray zone {R1_GRAY_LOW_NATS}-{R1_LIVENESS_FLOOR_NATS} = weakly-live)",
        "per_seed": per_seed,
        "flagoff_trailing_ce_max": off_trailing_max,
        "verdict": "PASS" if all(lb == "live" for lb in labels) else "FAIL",
        "labels": labels,
    }


def r2_source_level(on_terms: dict[int, dict], off_terms: dict[int, dict]) -> dict:
    """R2 — PRIMARY: flag-on terminal on-policy source ΔG vs flag-off committed."""
    on_dg = [on_terms[s]["delta_g"] for s in SEEDS]
    off_dg = [off_terms[s]["delta_g"] for s in SEEDS]
    diff = mean(on_dg) - mean(off_dg)
    seed_gap_on = abs(on_dg[0] - on_dg[1])
    # Indeterminate-for-noise takes PRECEDENCE over a band-crossing seed-mean
    # (round-1 critique addendum, analyzer guidance 3).
    if seed_gap_on > FROZEN_SOURCE_BAND_NATS:
        branch = "indeterminate-for-noise"
    elif diff < -FROZEN_SOURCE_BAND_NATS:
        branch = "suppression"
    elif diff > FROZEN_SOURCE_BAND_NATS:
        branch = "amplification"
    else:
        branch = "co-lands"
    # EOS-margin twin rule from the same four floats.
    on_m = [on_terms[s]["margin"] for s in SEEDS]
    off_m = [off_terms[s]["margin"] for s in SEEDS]
    margin_tol = max(2 * abs(off_m[0] - off_m[1]), MARGIN_TWIN_MIN_LOGIT)
    margin_diff = mean(on_m) - mean(off_m)
    margin_colands = abs(margin_diff) <= margin_tol
    saturated_any = any(on_terms[s]["saturated"] for s in SEEDS) or any(
        off_terms[s]["saturated"] for s in SEEDS
    )
    return {
        "rule": f"frozen band ±{FROZEN_SOURCE_BAND_NATS} nats around the flag-off committed "
        f"seed-mean; indeterminate-for-noise when the new pair's own seed gap exceeds the band",
        "flagon_delta_g": {f"seed{s}": on_terms[s]["delta_g"] for s in SEEDS},
        "flagoff_delta_g": {f"seed{s}": off_terms[s]["delta_g"] for s in SEEDS},
        "flagon_seed_mean": mean(on_dg),
        "flagoff_seed_mean": mean(off_dg),
        "diff_seed_mean": diff,
        "flagon_seed_gap": seed_gap_on,
        "branch": branch,
        "margin_twin": {
            "flagon_margin": {f"seed{s}": on_terms[s]["margin"] for s in SEEDS},
            "flagoff_margin": {f"seed{s}": off_terms[s]["margin"] for s in SEEDS},
            "diff_seed_mean": margin_diff,
            "tolerance": margin_tol,
            "co_lands": margin_colands,
        },
        # Off saturation the log-prob branch governs; margin governs only when
        # the saturation triage fires (analyzer guidance 3).
        "saturation_triage_fired": saturated_any,
        "per_cell_saturated": {
            "flagon": {f"seed{s}": on_terms[s]["saturated"] for s in SEEDS},
            "flagoff": {f"seed{s}": off_terms[s]["saturated"] for s in SEEDS},
        },
        "space_disagreement": (branch == "co-lands") != margin_colands,
    }


def r3_clamp(on_series: dict[int, list[dict]], off_series: dict[int, list[dict]]) -> dict:
    """R3 — trained-negative clamp vs the 8-bystander panel + rise-then-drop."""

    def _arm(series_by_seed: dict[int, list[dict]]) -> dict:
        out: dict = {}
        for seed, rows in series_by_seed.items():
            term = rows[-1]
            gap = term["bystander"]["delta_g"] - term["trained_neg"]["delta_g"]
            tneg = [r["trained_neg"]["delta_g"] for r in rows]
            peak_minus_term = max(tneg) - tneg[-1]
            out[f"seed{seed}"] = {
                "terminal_trained_neg_delta_g": term["trained_neg"]["delta_g"],
                "terminal_bystander_delta_g": term["bystander"]["delta_g"],
                "clamp_gap": gap,
                "clamp_present": gap >= CLAMP_BAR_NATS,
                "trained_neg_peak_minus_terminal": peak_minus_term,
                "rise_then_drop": peak_minus_term >= RISE_THEN_DROP_MIN_NATS,
            }
        return out

    on = _arm(on_series)
    off = _arm(off_series)  # the comparator's own gaps (1.25 / 1.03) + empirical null shape
    return {
        "rule": f"clamp present iff trained-neg mean sits >= {CLAMP_BAR_NATS} nats below the "
        f"bystander-panel mean in BOTH flag-on seeds; trajectory signature: trained-neg "
        f"peak - terminal >= {RISE_THEN_DROP_MIN_NATS} nat (flag-off arm = empirical null)",
        "flagon": on,
        "flagoff_comparator": off,
        "clamp_present_both_seeds": all(v["clamp_present"] for v in on.values()),
        "rise_then_drop_both_seeds": all(v["rise_then_drop"] for v in on.values()),
    }


def r4_channels(
    dense_on: dict[int, list[dict]],
    dense_off: dict[int, list[dict]],
    slot_on: dict[int, list[dict]],
    slot_off: dict[int, list[dict]],
) -> dict:
    """R4 — descriptive channel decomposition (Δz_eos vs Δz_marker) at matched steps."""

    def _at_steps(rows: list[dict]) -> dict[str, dict]:
        by_step = {r["step"]: r for r in rows}
        out = {}
        for s in SLOT_LADDER_STEPS:
            if s in by_step:
                r = by_step[s]
                out[str(s)] = {
                    ch: {
                        k: r[ch][k]
                        for k in ("delta_g", "delta_z_marker", "delta_z_eos", "delta_margin")
                    }
                    for ch in ("source", "trained_neg", "bystander")
                }
        return out

    return {
        "note": "descriptive, no gate — Δz_eos↑ (EOS-channel suppression) vs Δz_marker↓ "
        "(direct marker push-down), per slot (sep-marker = DV slot; sep-plain = flag-on "
        "loss slot), arms compared at matched steps",
        "sep_marker": {
            "flagon": {f"seed{s}": _at_steps(dense_on[s]) for s in SEEDS},
            "flagoff": {f"seed{s}": _at_steps(dense_off[s]) for s in SEEDS},
        },
        "sep_plain": {
            "flagon": {f"seed{s}": _at_steps(slot_on[s]) for s in SEEDS},
            "flagoff": {f"seed{s}": _at_steps(slot_off[s]) for s in SEEDS},
        },
    }


def r5_leakage_fraction(
    on_series: dict[int, list[dict]], off_series: dict[int, list[dict]]
) -> dict:
    """R5 — bystander ΔG / source ΔG at the dense terminal, per arm per seed."""

    def _arm(series_by_seed: dict[int, list[dict]]) -> dict:
        out = {}
        for seed, rows in series_by_seed.items():
            term = rows[-1]
            src = term["source"]["delta_g"]
            out[f"seed{seed}"] = {
                "bystander_delta_g": term["bystander"]["delta_g"],
                "source_delta_g": src,
                "leakage_fraction": (term["bystander"]["delta_g"] / src) if src else None,
            }
        return out

    return {
        "note": "descriptive — does suppression spread beyond the trained panel "
        "(flag-off committed fraction ≈ 0.43-0.47)",
        "flagon": _arm(on_series),
        "flagoff": _arm(off_series),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Task #613 flag A/B analysis (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--flagon-root", type=Path, default=Path("eval_results/issue_613/flagon_ab"))
    ap.add_argument("--flagoff-root", type=Path, default=Path("eval_results/issue_601/phase2"))
    ap.add_argument("--slotread-root", type=Path, default=Path("eval_results/issue_613/slotread"))
    ap.add_argument(
        "--out", type=Path, default=Path("eval_results/issue_613/analysis/ab_verdict.json")
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=i613_analyze] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    inputs: dict[str, str] = {}

    def _track(path: Path) -> Path:
        inputs[str(path)] = "present"
        return path

    traj_on, dense_on_raw, rowtype_on = {}, {}, {}
    dense_off_raw, traj_off, rowtype_off = {}, {}, {}
    slot_on_raw, slot_off_raw = {}, {}
    for seed in SEEDS:
        on_dir = args.flagon_root / f"{FLAGON_CELL}_seed{seed}"
        off_dir = args.flagoff_root / f"{FLAGOFF_CELL}_seed{seed}"
        traj_on[seed] = _load(_track(on_dir / "trajectory.json"))
        dense_on_raw[seed] = _load(_track(on_dir / "dense_trajectory.json"))
        rowtype_on[seed] = _load(_track(on_dir / "rowtype_ce.json"))
        traj_off[seed] = _load(_track(off_dir / "trajectory.json"))
        dense_off_raw[seed] = _load(_track(off_dir / "dense_trajectory.json"))
        rowtype_off[seed] = _load(_track(off_dir / "rowtype_ce.json"))
        slot_on_raw[seed] = _load(
            _track(args.slotread_root / f"{FLAGON_CELL}_seed{seed}" / "slot_trajectory.json")
        )
        slot_off_raw[seed] = _load(
            _track(args.slotread_root / f"{FLAGOFF_CELL}_seed{seed}" / "slot_trajectory.json")
        )

    on_terms = {s: _onpolicy_terminal_source(traj_on[s]) for s in SEEDS}
    off_terms = {s: _onpolicy_terminal_source(traj_off[s]) for s in SEEDS}
    dense_on = {s: dense_series(dense_on_raw[s]) for s in SEEDS}
    dense_off = {s: dense_series(dense_off_raw[s]) for s in SEEDS}
    slot_on = {s: dense_series(slot_on_raw[s]) for s in SEEDS}
    slot_off = {s: dense_series(slot_off_raw[s]) for s in SEEDS}

    verdict = {
        "schema_version": "i613_ab_verdict_v1",
        "cells": {"flagon": FLAGON_CELL, "flagoff": FLAGOFF_CELL},
        "seeds": list(SEEDS),
        "constants": {
            "frozen_source_band_nats": FROZEN_SOURCE_BAND_NATS,
            "clamp_bar_nats": CLAMP_BAR_NATS,
            "r1_liveness_floor_nats": R1_LIVENESS_FLOOR_NATS,
            "r1_gray_low_nats": R1_GRAY_LOW_NATS,
            "rise_then_drop_min_nats": RISE_THEN_DROP_MIN_NATS,
            "margin_twin_min_logit": MARGIN_TWIN_MIN_LOGIT,
            "slot_ladder_steps": list(SLOT_LADDER_STEPS),
        },
        "r1_liveness": r1_liveness(rowtype_on, rowtype_off),
        "r2_source_level": r2_source_level(on_terms, off_terms),
        "r3_clamp": r3_clamp(dense_on, dense_off),
        "r4_channels": r4_channels(dense_on, dense_off, slot_on, slot_off),
        "r5_leakage_fraction": r5_leakage_fraction(dense_on, dense_off),
        "inputs": inputs,
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    # Overall falsification read (task body, plan §3): co-land on BOTH the
    # source level AND the trained-negative leakage channel.
    verdict["overall"] = {
        "double_null": (
            verdict["r2_source_level"]["branch"] == "co-lands"
            and not verdict["r3_clamp"]["clamp_present_both_seeds"]
            and not verdict["r3_clamp"]["rise_then_drop_both_seeds"]
        ),
        "note": "double_null=True → loss placement does not resolve the #471-vs-#601 "
        "conflict under #601's recipe at this leakage level (scoped per analyzer guidance 1)",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(verdict, indent=2))
    os.replace(tmp, args.out)
    log.info(
        "ab_verdict written -> %s (R1 %s; R2 %s; R3 clamp_both=%s rise_drop_both=%s)",
        args.out,
        verdict["r1_liveness"]["verdict"],
        verdict["r2_source_level"]["branch"],
        verdict["r3_clamp"]["clamp_present_both_seeds"],
        verdict["r3_clamp"]["rise_then_drop_both_seeds"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
