#!/usr/bin/env python3
"""Issue #557 rollup — aggregate the lr-sweep cells into one dose-response table.

Adapted from rollup_issue543_survival.py. Reads, per (variant x seed):
``eval_results/issue_557/r50/<variant>/seed<S>/phase2/run_summary.json`` +
``phase2_result.json`` (lr + final train loss) + the trigger trajectory
(cliff step), PLUS the parent's committed lr=1e-4 anchor + pre-SFT summaries
under ``eval_results/issue_543/r50/seed<S>/`` (NOT re-run; #557 plan §5),
the absorption probe aggregate, and the judge scores when present.

Writes ``eval_results/issue_557/rollup.json`` with per-cell summaries, pooled
per-arm Wilson CIs (anchor arm included as ``lr1e4``), key-conditioning reads,
cliff steps vs the (10-15)x(1e-4/lr) predictions, and the §7 pre-registered
criteria READOUTS (computed, NOT auto-verdicted — the analyzer owns
interpretation).

FAIL-LOUD CONTRACT (round-2 blocker fix): every plan §6.5 required input —
the per-(variant x seed) ``run_summary.json`` + ``phase2_result.json`` +
trigger trajectory, the parent anchor/pre summaries + anchor trajectory, and
``absorption_probe.json`` (including its per-cell keys) — is preflighted; any
absence exits non-zero naming every missing path BEFORE anything is written.
``--allow-partial`` (fixtures/smoke ONLY, never the production invocation)
downgrades the preflight to warnings and marks the output ``"partial": true``
with the full ``missing_required`` list. ``judge_scores.json`` stays optional
(judge scoring may legitimately run after a first rollup).

CPU-only; safe to re-run any time (off-pod, VM-side).

Usage:
    uv run python scripts/rollup_issue557_lrsweep.py
    uv run python scripts/rollup_issue557_lrsweep.py --eval-root /tmp/fixture/eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="rollup_issue557_lrsweep")

from _issue543_common import (  # noqa: E402
    PROJECT_ROOT,
    repro_metadata,
)

log = logging.getLogger("rollup_issue557_lrsweep")

ARM = "r50"
DEFAULT_VARIANTS = ("lr3e5", "lr1e5", "lr5e6")
DEFAULT_SEEDS = (42, 137, 256)
ANCHOR_LR = 1.0e-4
ANCHOR_ARM_KEY = "lr1e4"
CLIFF_PREDICTION_BASE_STEPS = (10, 15)  # parent cliff completes by step 10-15 at 1e-4


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def _load_json(path: Path) -> dict | None:
    """Load a JSON file; None when absent (reachable ONLY under --allow-partial).

    Production invocations preflight every required input via
    :func:`_required_artifacts` before this loader runs, so a None here can
    never silently shrink a §7 criteria denominator outside partial mode.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _required_artifacts(
    d557: Path, d543: Path, variants: list[str], seeds: list[int]
) -> list[Path]:
    """Every file-level input the plan §6.5 deliverables contract REQUIRES.

    Per seed: the parent's committed anchor (phase2) + pre-SFT (phase1)
    summaries and the anchor trigger trajectory. Per (variant x seed): the new
    cell's run_summary.json, phase2_result.json train record, and trigger
    trajectory. Plus the absorption probe aggregate (a required conjunct of
    BOTH the §7 survival and kill criteria).
    """
    req: list[Path] = []
    for s in seeds:
        parent_cell = d543 / ARM / f"seed{s}"
        req += [
            parent_cell / "phase2" / "run_summary.json",
            parent_cell / "phase1" / "run_summary.json",
            parent_cell / "phase2_trajectory_trigger.jsonl",
        ]
    for v in variants:
        for s in seeds:
            cell_dir = d557 / ARM / v / f"seed{s}"
            req += [
                cell_dir / "phase2" / "run_summary.json",
                cell_dir / "phase2_result.json",
                cell_dir / "phase2_trajectory_trigger.jsonl",
            ]
    req.append(d557 / "absorption" / "absorption_probe.json")
    return req


def cliff_step(trajectory_path: Path) -> int | None:
    """First trajectory probe step at which the trained slot argmax-rate is 0.

    Rows are MarkerBandStopCallback dump records ({step, argmax_rate, ...}).
    Returns None when the file is missing or the argmax-rate never reaches 0
    (i.e. the install never fully collapsed on the frozen trigger probe).
    """
    if not trajectory_path.exists():
        return None
    for ln in trajectory_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        row = json.loads(ln)
        if row.get("argmax_rate") == 0.0:
            return int(row["step"])
    return None


def _cell_block(summary: dict | None) -> dict | None:
    """Per-eval-cell extract: emission + the three-space slot means."""
    if summary is None:
        return None
    out = {}
    for cell, s in summary["cells"].items():
        out[cell] = {
            k: s.get(k)
            for k in (
                "n",
                "emission_rate",
                "ends_with_marker_rate",
                "truncation_rate",
                "delta_logp_mean",
                "delta_z_marker_mean",
                "delta_eos_margin_mean",
                "logp_trained_mean",
                "logp_base_mean",
                "prob_trained_mean",
                "prob_base_mean",
            )
        }
    return out


def _criteria_readouts(
    rollup: dict, variants: list[str], seeds: list[int], absorption: dict | None
) -> dict:
    """§7 pre-registered criteria READOUTS (descriptive; analyzer owns the verdict)."""
    crit: dict = {}
    anchor_trigger_dlogp = [
        v
        for v in rollup["arms"][ANCHOR_ARM_KEY]["per_seed_trigger_delta_logp"].values()
        if v is not None
    ]
    anchor_keygap = [
        v
        for v in rollup["arms"][ANCHOR_ARM_KEY]["per_seed_keygap_delta_logp"].values()
        if v is not None
    ]
    for v in variants:
        arm = rollup["arms"][v]
        trig = arm["trigger"]
        per_seed = [r for r in trig["per_seed_emission"].values() if r is not None]
        absorbed_cells = (
            {
                s: (absorption.get("cells", {}) or {}).get(f"{v}_seed{s}", {}).get("absorbed")
                for s in seeds
            }
            if absorption
            else None
        )
        dlogp = [x for x in arm["per_seed_trigger_delta_logp"].values() if x is not None]
        crit[v] = {
            "survival_pooled_rate_ge_5pct": (
                trig["pooled_emission_rate"] >= 0.05
                if trig["pooled_emission_rate"] is not None
                else None
            ),
            "n_seeds_ge_2p5pct": sum(r >= 0.025 for r in per_seed),
            "pooled_ci_lo_gt_2pct": (trig["wilson_95ci"][0] > 0.02 if trig["pooled_n"] else None),
            "all_seeds_le_2pct": (all(r <= 0.02 for r in per_seed) if per_seed else None),
            "key_conditioning_emission_cis_disjoint": arm.get(
                "key_conditioning_emission_cis_disjoint"
            ),
            "absorbed_per_seed": absorbed_cells,
            "latent_min_trigger_dlogp_minus_anchor_max": (
                (min(dlogp) - max(anchor_trigger_dlogp)) if dlogp and anchor_trigger_dlogp else None
            ),
            "latent_keygap_max": (
                max(arm["per_seed_keygap_delta_logp"].values())
                if arm["per_seed_keygap_delta_logp"]
                else None
            ),
            "anchor_keygap_max": max(anchor_keygap) if anchor_keygap else None,
        }
    return crit


def _preflight(
    d557: Path, d543: Path, variants: list[str], seeds: list[int], allow_partial: bool
) -> tuple[bool, list[str], dict | None]:
    """Required-artifact preflight (round-2 blocker fix: fail loud).

    Checks every :func:`_required_artifacts` path PLUS the per-(variant x seed)
    cell keys inside ``absorption_probe.json`` (the absorption guard is a
    required conjunct of both §7 verdicts, so a present-but-incomplete probe
    file must not silently shrink the seed denominator either).

    Returns ``(ok, problems, absorption)``: ``ok=False`` means missing inputs
    in production mode — the caller must exit non-zero. Every missing input is
    logged (error in production mode, warning under ``allow_partial``).
    """
    required = _required_artifacts(d557, d543, variants, seeds)
    missing = [str(p) for p in required if not p.exists()]
    absorption = _load_json(d557 / "absorption" / "absorption_probe.json")
    missing_cells = (
        [
            f"{v}_seed{s}"
            for v in variants
            for s in seeds
            if f"{v}_seed{s}" not in (absorption.get("cells") or {})
        ]
        if absorption is not None
        else []
    )
    problems = missing + [
        f"absorption_probe.json missing required cell key '{k}'" for k in missing_cells
    ]
    if not problems:
        return True, [], absorption
    lvl = log.warning if allow_partial else log.error
    for item in problems:
        lvl("REQUIRED rollup input missing: %s", item)
    if not allow_partial:
        log.error(
            "%d required inputs missing — refusing to write a partial rollup.json "
            "(CLAUDE.md fail-fast; plan §6.5 deliverables). --allow-partial is for "
            "fixtures/smoke ONLY, never the production invocation.",
            len(problems),
        )
        return False, problems, absorption
    log.warning(
        "PARTIAL MODE (--allow-partial): continuing with %d missing required inputs; "
        'rollup.json will carry "partial": true + the missing_required list.',
        len(problems),
    )
    return True, problems, absorption


def _add_sweep_cells(
    rollup: dict,
    d557: Path,
    pre_cells: dict[int, dict | None],
    variants: list[str],
    seeds: list[int],
) -> dict[str, float | None]:
    """Add the per-(variant x seed) lr-sweep cell blocks; return lr per variant.

    A None lr (partial mode, missing phase2_result) never pins the arm's lr —
    a later valid seed supplies it (round-1 review minor).
    """
    lr_of_variant: dict[str, float | None] = {}
    for v in variants:
        for s in seeds:
            cell_dir = d557 / ARM / v / f"seed{s}"
            summary = _load_json(cell_dir / "phase2" / "run_summary.json")
            p2 = _load_json(cell_dir / "phase2_result.json")
            lr = p2["config"]["lr"] if p2 else None
            if lr is not None:
                lr_of_variant.setdefault(v, lr)
            scale = (ANCHOR_LR / lr) if lr else None
            rollup["cells"][f"{v}_seed{s}"] = {
                "variant": v,
                "seed": s,
                "lr": lr,
                "train_loss": p2.get("train_loss") if p2 else None,
                "phase2": _cell_block(summary),
                "pre_sft": _cell_block(pre_cells[s]),  # shared Phase-1 starting state
                "cliff_step": cliff_step(cell_dir / "phase2_trajectory_trigger.jsonl"),
                "cliff_step_predicted": (
                    [round(b * scale, 1) for b in CLIFF_PREDICTION_BASE_STEPS] if scale else None
                ),
            }
    return lr_of_variant


def main() -> int:
    """Aggregate the lr-sweep cells into rollup.json (fail-loud on missing inputs)."""
    args = parse_args()
    eval_root = Path(args.eval_root)
    d557 = eval_root / "issue_557"
    d543 = eval_root / "issue_543"
    variants = [v for v in args.variants.split(",") if v]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    ok, problems, absorption = _preflight(d557, d543, variants, seeds, args.allow_partial)
    if not ok:
        return 1

    rollup: dict = {
        **repro_metadata(),
        "arm": ARM,
        "partial": bool(problems),
        "missing_required": problems,
        "cells": {},
        "arms": {},
    }

    # ── Parent anchor (lr=1e-4) + pre-SFT reads (committed; NOT re-run) ─────
    anchor_cells: dict[int, dict | None] = {}
    pre_cells: dict[int, dict | None] = {}
    for s in seeds:
        parent_cell = d543 / ARM / f"seed{s}"
        anchor_cells[s] = _load_json(parent_cell / "phase2" / "run_summary.json")
        pre_cells[s] = _load_json(parent_cell / "phase1" / "run_summary.json")
        rollup["cells"][f"{ANCHOR_ARM_KEY}_seed{s}"] = {
            "variant": ANCHOR_ARM_KEY,
            "seed": s,
            "lr": ANCHOR_LR,
            "source": "issue_543 committed anchor (not re-run)",
            "phase2": _cell_block(anchor_cells[s]),
            "pre_sft": _cell_block(pre_cells[s]),
            "cliff_step": cliff_step(parent_cell / "phase2_trajectory_trigger.jsonl"),
            "cliff_step_predicted": list(CLIFF_PREDICTION_BASE_STEPS),
        }

    # ── New lr-sweep cells ───────────────────────────────────────────────────
    lr_of_variant = _add_sweep_cells(rollup, d557, pre_cells, variants, seeds)

    # ── Per-arm pooled emission + key-conditioning (anchor arm included) ────
    def _pool(arm_key: str, summaries: dict[int, dict | None]) -> dict:
        entry: dict = {"per_seed": {}}
        for cell_name in ("trigger", "no_trigger"):
            ks = ns = 0
            per_seed: dict[int, float | None] = {}
            for s, summ in summaries.items():
                blk = summ["cells"].get(cell_name) if summ else None
                if blk is None:
                    per_seed[s] = None
                    continue
                n = blk["n"]
                k = round(blk["emission_rate"] * n)
                ks, ns = ks + k, ns + n
                per_seed[s] = blk["emission_rate"]
            lo, hi = wilson_ci(ks, ns)
            entry[cell_name] = {
                "pooled_k": ks,
                "pooled_n": ns,
                "pooled_emission_rate": (ks / ns) if ns else None,
                "wilson_95ci": [lo, hi],
                "per_seed_emission": per_seed,
            }
        t, nt = entry["trigger"], entry["no_trigger"]
        if t["pooled_n"] and nt["pooled_n"]:
            entry["key_conditioning_emission_cis_disjoint"] = (
                t["wilson_95ci"][0] > nt["wilson_95ci"][1]
                or nt["wilson_95ci"][0] > t["wilson_95ci"][1]
            )
        # Latent retention per seed at the trigger + key-conditioning gap.
        entry["per_seed_trigger_delta_logp"] = {}
        entry["per_seed_keygap_delta_logp"] = {}
        for s, summ in summaries.items():
            if summ is None:
                continue
            trig = summ["cells"].get("trigger") or {}
            nokey = summ["cells"].get("no_trigger") or {}
            entry["per_seed_trigger_delta_logp"][s] = trig.get("delta_logp_mean")
            if trig.get("delta_logp_mean") is not None and nokey.get("delta_logp_mean") is not None:
                entry["per_seed_keygap_delta_logp"][s] = (
                    trig["delta_logp_mean"] - nokey["delta_logp_mean"]
                )
        return entry

    sweep_summaries: dict[str, dict[int, dict | None]] = {
        v: {
            s: _load_json(d557 / ARM / v / f"seed{s}" / "phase2" / "run_summary.json")
            for s in seeds
        }
        for v in variants
    }
    rollup["arms"][ANCHOR_ARM_KEY] = {
        "lr": ANCHOR_LR,
        **_pool(ANCHOR_ARM_KEY, anchor_cells),
        "source": "issue_543 committed anchor",
    }
    for v in variants:
        rollup["arms"][v] = {"lr": lr_of_variant.get(v), **_pool(v, sweep_summaries[v])}

    # ── Absorption (REQUIRED — preflighted above) + judge (optional) merges ──
    if absorption is None:
        # Reachable only under --allow-partial; the preflight already warned.
        log.warning("absorption_probe.json absent — absorption fields omitted (partial mode).")
    else:
        rollup["absorption"] = {
            "gate": absorption.get("gate"),
            "cells": absorption.get("cells"),
            "anchor_cells": absorption.get("anchor_cells"),
        }
    judge = _load_json(d557 / "absorption" / "judge_scores.json")
    if judge is not None:
        rollup["judge_scores_per_set_mean"] = judge.get("per_set_mean")

    rollup["criteria_readouts"] = _criteria_readouts(rollup, variants, seeds, absorption)

    out = d557 / "rollup.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rollup, indent=2))
    log.info("Rollup -> %s", out)
    return 0


def parse_args() -> argparse.Namespace:
    """CLI: eval root, variant/seed grid, and the fixtures-only --allow-partial."""
    p = argparse.ArgumentParser(
        description="Issue #557 lr-sweep rollup (CPU-only, VM-side).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval-root", type=str, default=str(PROJECT_ROOT / "eval_results"))
    p.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Fixtures/smoke ONLY: degrade missing required inputs to warnings and mark "
            'the output "partial": true. The production invocation must NEVER pass this — '
            "without it, any missing plan §6.5 required artifact exits non-zero."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
