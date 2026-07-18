#!/usr/bin/env python
"""#1481 — matched-recipe contrastive vs positive-only content grid (plan §4.1/§4.7).

Registry + seams for the CONTENT half of the 192-cell grid (casual style /
impolite / sycophancy × 4 factory contexts × {con, pos} × 3 LR × 2 seeds),
registered as SIX single-behavior/single-regime rounds into the
round-parametrized #1090 fu4 driver (`scripts/issue1090_fu4.py`) — the exact
composition pattern #1434 validated (`issue1434_cells.py` registers `i1434` +
`i1434po`; commit 31cee573b0 round-parametrized the external-round seams).
One round per (behavior, regime) keeps `RoundSpec.mix_composition` /
`train_max_steps` uniform per round (the i1434/i1434po precedent) — con mixes
are 80-row (20/20/40, epochs 15 = 75 steps), po mixes 60-row (20/0/40,
`max_steps` 75).

Seed threading: each run's SEED is embedded in its run_id (`...-s42|-s137`);
`issue1481_worker.py` rewrites `--seed` from :func:`seed_for_run_id` before
delegating `--phase run` to the fu4 driver, so a single dispatch fans out
both seeds (Fu4Run carries no seed field by design).

Everything heavy stays in the parent modules; this file is registry + seams.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1090_fu3_cells as fu3_cells  # noqa: E402
import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_fu4 as fu4  # noqa: E402
import issue1090_run as run1090  # noqa: E402
import issue1434_cells as c1434  # noqa: E402

logger = logging.getLogger("issue1481.cells")

ISSUE_1481 = 1481
DATA_PREFIX_1481 = "issue1481_conpos_grid"
ADAPTER_PREFIX_1481 = "issue1481"  # model-repo run dirs: issue1481/<run_id>/checkpoint-<step>
REPO_ROOT = _SCRIPTS_DIR.parent
DELIVERABLES_DIR_1481 = REPO_ROOT / "eval_results" / "issue_1481"
WORKER_SCRIPT = _SCRIPTS_DIR / "issue1481_worker.py"

SEEDS: tuple[int, ...] = (42, 137)  # plan §4.1 (Source: #627 + project second seed)
REGIMES: tuple[str, ...] = ("con", "po")
CTX_KEYS: tuple[str, ...] = ("pers", "bare", "conv", "icl")

# behavior short-key -> registered behavior name (fu3/#1434 universes)
BEHAVIOR_BY_KEY: dict[str, str] = {
    "cas": "writing_style",
    "imp": "impolite",
    "syc": "sycophancy",
}
# fu3 cell-family letter per behavior (issue1090_fu3 upload layout)
_FU3_FAMILY: dict[str, str] = {"imp": "C2", "syc": "C3"}
# #1090 production persona-mix dirs (plan §4.3; shas re-probed at Phase 0)
_PROD_MIX_PREFIX: dict[str, str] = {
    "imp": f"{run1090.DATA_PREFIX}/c2-impolite-claude/mix",
    "syc": f"{run1090.DATA_PREFIX}/c3-sycophancy-claude/mix",
}

# #1434 data-repo revision pins (plan §10; artifact-reuse checks (e)/(f))
I1434_CON_MIX_REV = "72a5c3a832fa"
I1434_PO_MIX_REV = "8d95d4b1"

# Dose band + dose-match tolerance (plan §4.5 — #1434 verbatim)
JUDGED_RATE_BAND: tuple[float, float] = (0.60, 0.85)
DOSE_MATCH_MAX_RATE_GAP = 0.10
# P1 apply-and-read parity tolerance (plan §4.6; calibration: fu4/fu5/fu7
# committed `runs[<arm>].selection` rates — same instrument, same rate surface)
P1_PARITY_MAX_ABS_DELTA = 0.15
# P4 #1434 apply-and-read probe (plan §4.6: ws-pers-lr1e5/checkpoint-25 vs
# committed 0.625 within ±0.08 — calibration: #627's ±0.08 parity-anchor
# precedent, same rate surface, same rubric)
P4_PROBE_RUN_ID = "ws-pers-lr1e5"
P4_PROBE_CHECKPOINT = 25
P4_PROBE_COMMITTED_RATE = 0.625
P4_PROBE_MAX_ABS_DELTA = 0.08


def context_id_for(behavior: str, ctx_key: str) -> str:
    """Training-context id per (behavior, context key) — the factory universe."""
    if ctx_key == "pers":
        return run1090.SOURCE_CONTEXT_ID  # persona_software_engineer
    if ctx_key == "bare":
        return fu3_cells.BARE  # "default"
    if ctx_key == "conv":
        return fu3_cells.CONV_CONTEXT_ID  # wildchat_prefix_real545
    if ctx_key == "icl":
        return f"icl_prefix_{behavior}"
    raise ValueError(f"unknown ctx_key {ctx_key!r}")


def seed_for_run_id(run_id: str) -> int:
    """The run's seed, parsed from the `-s<seed>` run_id suffix (fail-loud)."""
    tail = run_id.rsplit("-s", 1)
    if len(tail) != 2 or not tail[1].isdigit() or int(tail[1]) not in SEEDS:
        raise ValueError(f"run_id {run_id!r} has no valid -s<seed> suffix (want one of {SEEDS})")
    return int(tail[1])


def mix_for(beh_key: str, ctx_key: str, regime: str) -> tuple[str, str]:
    """(mix_hub_prefix, mix_layout) per (behavior, context, regime) — plan §4.3."""
    if beh_key == "cas":
        cell = f"ws-{ctx_key}" if regime == "con" else f"ws-po-{ctx_key}"
        return f"{c1434.DATA_PREFIX_1434}/{cell}/mix", "i1434-mix-subdir"
    if regime == "po":
        # DERIVED at Phase 0 (issue1481_worker --phase mixes) via the #1434
        # D1' row-provenance filter; uploaded under the i1481 bucket.
        return f"{DATA_PREFIX_1481}/po_mixes/{beh_key}-{ctx_key}/mix", "i1481-mix-subdir"
    if ctx_key == "pers":
        return _PROD_MIX_PREFIX[beh_key], "parent-mix-subdir"
    behavior = BEHAVIOR_BY_KEY[beh_key]
    fam = _FU3_FAMILY[beh_key]
    # fu3 uploaded per-cell artifacts FLAT at the cell root (plan §4.3,
    # Hub-verified 2026-07-17) — D3 leg (h)(ii).
    return f"{fu3w.DATA_PREFIX_FU3}/{fam}-{ctx_key}-con-{behavior}-claude", "fu3-flat"


# ── Reused arms (NOT retrained) ──────────────────────────────────────────────

# 24 reused #1434 casual seed-42 cells (plan §4.1): consumed at ANALYSIS time
# (committed ladders/panels/Tier-2 reused as-is), never re-dispatched.
REUSED_1434_LADDERS = {
    "con": "eval_results/issue_1434/i1434_ladders.json",
    "po": "eval_results/issue_1434/writing-style-positive-only-regime/i1434po_ladders.json",
}


def reused_1434_run_id(ctx_key: str, regime: str, lr: float) -> str:
    """#1434 run id serving as the casual seed-42 (con|po) arm."""
    cell = f"ws-{ctx_key}" if regime == "con" else f"ws-po-{ctx_key}"
    return f"{cell}-{fu4.LR_TAG[lr]}"


@dataclasses.dataclass(frozen=True)
class ReusedConArm:
    """One of the 12 fu4/fu5/fu7 committed selected checkpoints (plan §4.1/§4.6
    gate P1): the committed selected rung IS the arm's selection read; the P1
    apply-and-read re-read runs at that rung under THIS run's instrument."""

    arm_id: str  # the i1481 grid slot it fills, e.g. imp-pers-con-lr1e5-s42
    source_run_id: str  # parent run id inside the committed ladders JSON
    ladders_path: str  # committed per-rung rates + selection (repo-relative)
    adapter_run_prefix: str  # model-repo run dir holding selected + final ckpts
    behavior: str
    context_id: str
    committed_in_band: bool  # False -> committed closest-approach (parity-read only)


def _reused_con_arms() -> tuple[ReusedConArm, ...]:
    fu4_l = "eval_results/issue_1090/fu4-extended-dose-lr/fu4_ladders.json"
    fu5_l = "eval_results/issue_1090/finish-impolite-bare-and-formatting-rank/fu5_ladders.json"
    fu7_l = "eval_results/issue_1090/sycophancy-lr-install-and-remeasure/fu7_ladders.json"
    out: list[ReusedConArm] = []
    for lr in fu4.FU4_LRS:
        tag = fu4.LR_TAG[lr]
        # plan §4.1: imp-pers-lr1e5 (0.354) + imp-bare-lr1e5 (0.45) committed
        # closest-approach OUT of band; the other 10 committed IN band.
        for ctx_key, src, ladders, prefix in (
            ("pers", f"imp-pers-{tag}", fu4_l, f"adapters/issue1090_fu4/imp-pers-{tag}"),
            ("conv", f"imp-conv-{tag}", fu4_l, f"adapters/issue1090_fu4/imp-conv-{tag}"),
            ("bare", f"imp-bare-{tag}", fu5_l, f"adapters/issue1090_fu5/imp-bare-{tag}"),
        ):
            out.append(
                ReusedConArm(
                    arm_id=f"imp-{ctx_key}-con-{tag}-s42",
                    source_run_id=src,
                    ladders_path=ladders,
                    adapter_run_prefix=prefix,
                    behavior="impolite",
                    context_id=context_id_for("impolite", ctx_key),
                    committed_in_band=not (tag == "lr1e5" and ctx_key in ("pers", "bare")),
                )
            )
        out.append(
            ReusedConArm(
                arm_id=f"syc-pers-con-{tag}-s42",
                source_run_id=f"syc-c3-{tag}",
                ladders_path=fu7_l,
                adapter_run_prefix=f"adapters/issue1090_fu7/syc-c3-{tag}",
                behavior="sycophancy",
                context_id=context_id_for("sycophancy", "pers"),
                committed_in_band=True,
            )
        )
    return tuple(out)


REUSED_CON_ARMS: tuple[ReusedConArm, ...] = _reused_con_arms()
REUSED_CON_ARM_BY_ID = {a.arm_id: a for a in REUSED_CON_ARMS}
assert len(REUSED_CON_ARMS) == 12, len(REUSED_CON_ARMS)


def is_reused(beh_key: str, ctx_key: str, regime: str, lr: float, seed: int) -> bool:
    """True when the grid slot is filled by a reused artifact (never retrained)."""
    if beh_key == "cas" and seed == 42:
        return True  # 24 reused #1434 cells (both regimes, all ctx/lr)
    arm_id = f"{beh_key}-{ctx_key}-{regime}-{fu4.LR_TAG[lr]}-s{seed}"
    return arm_id in REUSED_CON_ARM_BY_ID


# ── Fresh-run registries (six rounds) ────────────────────────────────────────


def round_name(beh_key: str, regime: str) -> str:
    return f"i1481{beh_key}" + ("po" if regime == "po" else "")


def _content_runs(beh_key: str, regime: str) -> tuple[fu4.Fu4Run, ...]:
    behavior = BEHAVIOR_BY_KEY[beh_key]
    runs: list[fu4.Fu4Run] = []
    for ctx_key in CTX_KEYS:
        prefix, layout = mix_for(beh_key, ctx_key, regime)
        for lr in fu4.FU4_LRS:
            for seed in SEEDS:
                if is_reused(beh_key, ctx_key, regime, lr, seed):
                    continue
                run_id = f"{beh_key}-{ctx_key}-{regime}-{fu4.LR_TAG[lr]}-s{seed}"
                runs.append(
                    fu4.Fu4Run(
                        run_id=run_id,
                        cell_key=f"{beh_key}-{ctx_key}",
                        behavior=behavior,
                        context_id=context_id_for(behavior, ctx_key),
                        lr=lr,
                        mix_hub_prefix=prefix,
                        mix_layout=layout,
                        fu3_base_eval="",  # i1481 generates its own base arms (Phase C)
                        round_name=round_name(beh_key, regime),
                        run_name_override=f"issue1481_{run_id}_seed{seed}",
                    )
                )
    return tuple(runs)


RUNS_BY_ROUND: dict[str, tuple[fu4.Fu4Run, ...]] = {
    round_name(b, r): _content_runs(b, r) for b in BEHAVIOR_BY_KEY for r in REGIMES
}
# Fresh-run accounting (plan §4.1): impolite 39, sycophancy 45, casual 24.
_N_FRESH = {b: sum(len(RUNS_BY_ROUND[round_name(b, r)]) for r in REGIMES) for b in BEHAVIOR_BY_KEY}
assert _N_FRESH == {"cas": 24, "imp": 39, "syc": 45}, _N_FRESH


def _smoke_runs(beh_key: str, regime: str) -> str:
    """One tiny run PER ARM CLASS (source-context class) — the fu5 per-arm-class
    smoke precedent (#1090 fu5: a formatting-only smoke missed the bare-context
    panel-disjointness seam). pers + bare + icl cover persona / bare-default /
    prefix+ICL-bank seams; conv shares the prefix class with icl."""
    runs = RUNS_BY_ROUND[round_name(beh_key, regime)]
    by_id = {r.run_id: r for r in runs}
    picks: list[str] = []
    for ctx_key in ("pers", "bare", "icl"):
        for seed in (137, 42):  # casual has no fresh seed-42; imp/syc prefer s137
            rid = f"{beh_key}-{ctx_key}-{regime}-lr1e5-s{seed}"
            if rid in by_id:
                picks.append(rid)
                break
    assert picks, (beh_key, regime)
    return ",".join(picks)


def register_i1481_rounds() -> dict[str, fu4.RoundSpec]:
    """Insert the six i1481 content rounds into the fu4 ROUNDS registry
    (idempotent). Recipe: the fu4 content bundle verbatim (plan §4.2 — lr
    ladder {1e-5,3e-5,1e-4}, r32/α64 rsLoRA, epochs 15 = 75 steps, save 5,
    cosine, batch 4×4, max_length 2048); po rounds pin `max_steps` 75 (the
    #1434 po convention) + the 20/0/40 composition."""
    specs: dict[str, fu4.RoundSpec] = {}
    for beh_key in BEHAVIOR_BY_KEY:
        for regime in REGIMES:
            name = round_name(beh_key, regime)
            if name in fu4.ROUNDS:
                specs[name] = fu4.ROUNDS[name]
                continue
            is_po = regime == "po"
            is_cas = beh_key == "cas"
            spec = fu4.RoundSpec(
                name=name,
                label=f"conpos-grid-{BEHAVIOR_BY_KEY[beh_key]}-{regime}",
                data_prefix=DATA_PREFIX_1481,
                adapter_prefix=ADAPTER_PREFIX_1481,
                deliverables_dir=DELIVERABLES_DIR_1481,
                manifest_name=f"cell_manifest_{name}.json",
                ladders_name=f"{name}_ladders.json",
                runs=RUNS_BY_ROUND[name],
                smoke_default_run=_smoke_runs(beh_key, regime),
                # No K3 retrain-parity anchor: the P1 apply-and-read re-reads of
                # the 12 committed fu4/fu5/fu7 checkpoints are this grid's rig-
                # parity anchor (plan §4.6 — stronger + cheaper than regen-parity;
                # the empty id matches no run, so the K3 branch never fires).
                k3_parity_run_id="",
                k3_parity_degraded_floor=None,
                reread_rate_floor=None,
                max_lora_rank=64,
                eval_split_diagnostic=False,
                reused_runs=(),
                issue=ISSUE_1481,
                worker_script=str(WORKER_SCRIPT),
                upload_all_rungs=True,  # plan §10: keep EVERY rung (discarded_artifacts [])
                # Per-behavior instrument routing (plan §6): casual = the #1434
                # verbatim pv trait-score rubric; impolite/sycophancy = the
                # registered factory rubrics (None -> fu3w.judge_graded_r23),
                # the fu4/fu7 ladder-reference instruments.
                judge_fn=(c1434.pv_judge_fn if is_cas else None),
                margin_pools_fn=(c1434.i1434_margin_pools if is_cas else None),
                train_max_steps=(c1434.PO_TRAIN_MAX_STEPS if is_po else None),
                mix_composition=(
                    c1434.PO_MIX_COMPOSITION if is_po else fu4.EXPECTED_MIX_COMPOSITION
                ),
                raw_prefix=f"{DATA_PREFIX_1481}/raw_completions/{name}",
            )
            fu4.ROUNDS[name] = spec
            specs[name] = spec
    return specs


I1481_ROUND_NAMES: tuple[str, ...] = tuple(
    round_name(b, r) for b in BEHAVIOR_BY_KEY for r in REGIMES
)

# Phase-A dispatch groups (plan §4.4: A1 impolite, A2 sycophancy, A3 casual-s137)
DISPATCH_ROUNDS: dict[str, tuple[str, ...]] = {
    "impolite": (round_name("imp", "con"), round_name("imp", "po")),
    "sycophancy": (round_name("syc", "con"), round_name("syc", "po")),
    "casual-s137": (round_name("cas", "con"), round_name("cas", "po")),
}
# Reused-arm re-read jobs ride the matching behavior dispatch (plan §4.4)
REREAD_BY_DISPATCH: dict[str, tuple[ReusedConArm, ...]] = {
    "impolite": tuple(a for a in REUSED_CON_ARMS if a.behavior == "impolite"),
    "sycophancy": tuple(a for a in REUSED_CON_ARMS if a.behavior == "sycophancy"),
    "casual-s137": (),
}


# ── Selection / dose-match helpers (plan §4.5) ───────────────────────────────


def band_distance(rate: float, band: tuple[float, float] = JUDGED_RATE_BAND) -> float:
    """Distance from ``rate`` to the closed band interval (0.0 inside)."""
    lo, hi = band
    return max(lo - rate, 0.0, rate - hi)


def verdict_arm(arms: list[tuple[float, str, dict]]) -> tuple[str, dict]:
    """Pre-registered verdict arm over one (behavior, context, regime, seed)
    cell's LR arms — #1434 verbatim: lowest-LR arm whose selection is in band,
    else closest approach (min band distance, tie-break lowest lr).

    ``arms`` = [(lr, arm_id, selection_record)] with selection carrying
    ``rate`` + ``in_band`` (the fu4 ``select_dose_checkpoint`` shape).
    """
    if not arms:
        raise ValueError("verdict_arm: no arms")
    for lr, arm_id, sel in sorted(arms, key=lambda t: t[0]):
        if bool(sel.get("in_band")):
            return arm_id, {
                "rule": "lowest_lr_in_band",
                "arm_id": arm_id,
                "lr": lr,
                "selection": sel,
            }
    lr, arm_id, sel = min(arms, key=lambda t: (band_distance(float(t[2]["rate"])), t[0]))
    return arm_id, {
        "rule": "closest_approach",
        "arm_id": arm_id,
        "lr": lr,
        "band_distance": band_distance(float(sel["rate"])),
        "selection": sel,
    }


def dose_match_label(con_sel: dict, pos_sel: dict) -> dict:
    """Content dose-match qualifier (plan §3/§4.5): both verdict arms in band
    AND |selection-rate difference| <= 0.10."""
    con_rate, pos_rate = float(con_sel["rate"]), float(pos_sel["rate"])
    gap = abs(con_rate - pos_rate)
    matched = (
        bool(con_sel.get("in_band"))
        and bool(pos_sel.get("in_band"))
        and (gap <= DOSE_MATCH_MAX_RATE_GAP)
    )
    return {
        "dose_matched": matched,
        "con_rate": con_rate,
        "pos_rate": pos_rate,
        "rate_gap": gap,
        "con_in_band": bool(con_sel.get("in_band")),
        "pos_in_band": bool(pos_sel.get("in_band")),
    }


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval (parent implementation)."""
    return run1090._wilson(k, n)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe 95% CI on p1 - p2 (Wilson-score hybrid; #1434 verbatim)."""
    l1, u1 = run1090._wilson(k1, n1)
    l2, u2 = run1090._wilson(k2, n2)
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def lattice_verdict(d: float, ci: tuple[float, float]) -> str:
    """Plan §3 registered lattice (DISJOINT + exhaustive) for one cell's D."""
    if d > 0 and ci[0] > 0:
        return "Containment"
    if ci[1] < 0:
        return "Reversed"
    return "Indistinguishable"


# ── Held-out decomposition membership (plan §5 + §4.7 MF-3) ──────────────────


def heldout_contexts(behavior: str, train_ctx_id: str, realized_panel: list[str]) -> list[str]:
    """Per-cell held-out read contexts: the six-context read panel MINUS the
    source context MINUS every REALIZED training-panel member (read from the
    mix builder's realized output, never plan prose). The MF-3 mechanized
    disjointness assert lives in issue1481_analysis.py."""

    panel_ids = [c.context_id for c in fu3w.bystander_panel(behavior)]
    held = [c for c in panel_ids if c != train_ctx_id and c not in set(realized_panel)]
    return held
