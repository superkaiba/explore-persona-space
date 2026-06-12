"""Task #627 — loaders for the committed #601 marker slab (CPU re-analysis).

Reads ONLY committed JSONs under ``eval_results/issue_601/`` — no adapters are
loaded, so the #601 staged-classic-gauge issue cannot bite; the committed
numbers' gauge is inherited and named (plan §4 Phase 3).

Two read types:
    trajectory.json        — ON-POLICY slot reads (model's own response end);
                             valid for cross-condition comparison. Four-float
                             contract honored (delta_margin per record).
    dense_trajectory.json  — TEACHER-FORCED frozen-R reads; valid for
                             WITHIN-condition dose-curve SHAPE only (never
                             cross-condition level comparison; #432→#456).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from explore_persona_space.experiments.leakage_vs_install_627 import (
    marker_cell_arm,
    record_margin,
    source_margin,
)

DEFAULT_601_ROOT = Path("eval_results/issue_601")


def iter_trajectory_paths(root: Path = DEFAULT_601_ROOT, *, dense: bool = False) -> list[Path]:
    name = "dense_trajectory.json" if dense else "trajectory.json"
    paths = sorted(root.glob(f"**/{name}"))
    if not paths:
        raise FileNotFoundError(f"No {name} under {root} — is the #601 slab materialized?")
    return paths


def _per_question_means(qrecs: dict) -> dict[str, float]:
    """Mean margin / Δlog P / ΔP over one persona's question records."""
    margins, dlogps, dprobs = [], [], []
    for rec in qrecs.values():
        margins.append(record_margin(rec))
        if "g_logp" in rec:  # on-policy trajectory record
            g, b = float(rec["g_logp"]), float(rec["b_logp"])
        else:  # dense (teacher-forced) record
            g, b = float(rec["logp_g"]), float(rec["logp_b"])
        dlogps.append(g - b)
        dprobs.append(math.exp(g) - math.exp(b))
    n = len(margins)
    if n == 0:
        raise ValueError("persona with zero question records")
    return {
        "margin": sum(margins) / n,
        "dlogp": sum(dlogps) / n,
        "dprob": sum(dprobs) / n,
        "n_questions": n,
    }


def load_onpolicy_cell(path: Path) -> dict:
    """One on-policy trajectory.json -> normalized cell dict."""
    with open(path) as f:
        p = json.load(f)
    cell, seed = p["cell"], int(p["seed"])
    checkpoints = []
    for ck in p["checkpoints"]:
        src = ck["source_self"]
        src_dlogp = float(src["delta_g_mean"])
        # Source probability read is computed from the stored MEAN log-probs
        # (geometric-mean probability) — per-question source records are not
        # stored; sanity-read only, labeled as such downstream.
        src_dprob = math.exp(float(src["g_logp_mean"])) - math.exp(float(src["b_logp_mean"]))
        bystanders = {
            persona: _per_question_means(qrecs) for persona, qrecs in ck["held_out"].items()
        }
        checkpoints.append(
            {
                "frac": ck.get("frac"),
                "step": ck.get("step"),
                "source": {
                    "margin": source_margin(src),
                    "dlogp": src_dlogp,
                    "dprob": src_dprob,
                    "emission_p": src.get("emission_p"),
                },
                "bystanders": bystanders,
            }
        )
    return {
        "path": str(path),
        "cell": cell,
        "seed": seed,
        "source_persona": p["source"],
        "mix_arm": marker_cell_arm(cell),
        "read_type": "on_policy",
        "held_out_personas": sorted(p["held_out_personas"]),
        "checkpoints": checkpoints,
    }


def load_dense_cell(path: Path) -> dict:
    """One teacher-forced dense_trajectory.json -> normalized cell dict.

    WITHIN-CONDITION dose-curve shape only (read_type carried so downstream
    code can fence it; plan §4 measurement rule)."""
    with open(path) as f:
        p = json.load(f)
    cell, seed = p["cell"], int(p["seed"])
    bystander_panel = set(p["bystander_panel"])
    checkpoints = []
    for ck in p["checkpoints"]:
        sm = ck["source_mean"]
        src_margin = float(sm["delta_margin"])
        src_dlogp = float(sm["delta_g"])
        src_dprob = math.exp(float(sm["logp_g"])) - math.exp(float(sm["logp_b"]))
        bystanders = {
            persona: _per_question_means(qrecs)
            for persona, qrecs in ck["reads"].items()
            if persona in bystander_panel
        }
        checkpoints.append(
            {
                "frac": ck.get("frac"),
                "step": ck.get("step"),
                "source": {"margin": src_margin, "dlogp": src_dlogp, "dprob": src_dprob},
                "bystanders": bystanders,
            }
        )
    return {
        "path": str(path),
        "cell": cell,
        "seed": seed,
        "source_persona": p["source"],
        "mix_arm": marker_cell_arm(cell),
        "read_type": "teacher_forced",
        "held_out_personas": sorted(bystander_panel),
        "checkpoints": checkpoints,
    }


def load_all_onpolicy(root: Path = DEFAULT_601_ROOT) -> list[dict]:
    return [load_onpolicy_cell(p) for p in iter_trajectory_paths(root)]


def load_all_dense(root: Path = DEFAULT_601_ROOT) -> list[dict]:
    return [load_dense_cell(p) for p in iter_trajectory_paths(root, dense=True)]


def seed_gap_tolerance(cells: list[dict]) -> dict:
    """The registered #601 matched-install tolerance, ported to margin space as
    a FORMULA (plan §11): 2x the lineage's max within-cell seed gap in source
    Δmargin, over seed-42/137 paired cells at grid-position-matched
    checkpoints (paired by ``frac``). Returns the tolerance + full provenance
    (the §13 item 4(c) tolerance-formula manifest body)."""
    by_cell: dict[str, dict[int, dict]] = {}
    for c in cells:
        by_cell.setdefault(c["cell"], {})[c["seed"]] = c
    pairs = []
    for cell, seeds in sorted(by_cell.items()):
        if not {42, 137} <= set(seeds):
            continue
        a, b = seeds[42], seeds[137]
        a_by_frac = {ck["frac"]: ck for ck in a["checkpoints"]}
        b_by_frac = {ck["frac"]: ck for ck in b["checkpoints"]}
        for frac in sorted(set(a_by_frac) & set(b_by_frac)):
            m42 = a_by_frac[frac]["source"]["margin"]
            m137 = b_by_frac[frac]["source"]["margin"]
            pairs.append(
                {
                    "cell": cell,
                    "frac": frac,
                    "step_seed42": a_by_frac[frac]["step"],
                    "step_seed137": b_by_frac[frac]["step"],
                    "source_margin_seed42": m42,
                    "source_margin_seed137": m137,
                    "abs_gap": abs(m42 - m137),
                }
            )
    if not pairs:
        raise RuntimeError(
            "No seed-42/137 paired (cell, frac) checkpoints in the #601 slab — "
            "the tolerance formula has no inputs; H2 matching cannot proceed"
        )
    max_gap = max(p["abs_gap"] for p in pairs)
    argmax = max(pairs, key=lambda p: p["abs_gap"])
    return {
        "formula": "2 x max within-cell seed gap in source EOS-margin space (Source: #601 "
        "convention, log-prob value was 5.58 = 2 x 2.79; ported to margin space)",
        "statistic": "abs(source_margin[seed42] - source_margin[seed137]) at frac-matched "
        "on-policy checkpoints",
        "tolerance_margin": 2.0 * max_gap,
        "max_within_cell_seed_gap_margin": max_gap,
        "argmax_pair": argmax,
        "n_seed_paired_readings": len(pairs),
        "seed_paired_readings": pairs,
    }


def matched_pairs(cells: list[dict], tolerance: float) -> list[dict]:
    """Cross-condition matched-install checkpoint pairs (H2 manifest, plan §13
    item 4(b)). Constraints: same seed, IDENTICAL held-out panel, contrastive
    vs posonly mix arms, |source Δmargin difference| <= tolerance. Where no
    pair matches, the empty list is itself the (reported) result — never
    interpolate the marker family (plan §4)."""
    out: list[dict] = []
    contrastive = [c for c in cells if c["mix_arm"] == "contrastive"]
    posonly = [c for c in cells if c["mix_arm"] == "posonly"]
    for c_cell in contrastive:
        for p_cell in posonly:
            if c_cell["seed"] != p_cell["seed"]:
                continue
            if c_cell["held_out_personas"] != p_cell["held_out_personas"]:
                continue
            for ck_c in c_cell["checkpoints"]:
                for ck_p in p_cell["checkpoints"]:
                    gap = abs(ck_c["source"]["margin"] - ck_p["source"]["margin"])
                    if gap <= tolerance:
                        out.append(
                            {
                                "seed": c_cell["seed"],
                                "contrastive_cell": c_cell["cell"],
                                "contrastive_frac": ck_c["frac"],
                                "contrastive_step": ck_c["step"],
                                "contrastive_source_margin": ck_c["source"]["margin"],
                                "posonly_cell": p_cell["cell"],
                                "posonly_frac": ck_p["frac"],
                                "posonly_step": ck_p["step"],
                                "posonly_source_margin": ck_p["source"]["margin"],
                                "install_gap_margin": gap,
                                "n_bystanders": len(c_cell["held_out_personas"]),
                            }
                        )
    return out
