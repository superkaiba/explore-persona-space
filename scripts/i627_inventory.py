#!/usr/bin/env python3
"""Task #627 Phase 0 — inventory commit + registered manifests (VM, CPU).

Re-verifies the plan §2 inventory table against the ACTUAL committed files,
asserts every §6 statistical-input field exists, records HF revisions, and
emits the registered manifests (plan §13 item 4):

    eval_results/issue_627/inventory.json             — the re-verified inventory
    eval_results/issue_627/matched_install_cells.json — (a) #608 matched-install
        cell manifest: FIRST-crossing 0.50 brackets per (source, arm), width
        guard flags; HARD-ASSERTED equal to the registered plan §4 table.
    eval_results/issue_627/marker_matched_pairs.json  — (b) #601 marker
        matched-pair manifest (H2 cannot silently become "no matched pairs").
    eval_results/issue_627/marker_tolerance.json      — (c) tolerance-formula
        manifest: seed-paired cells/checkpoints + the source Δmargin statistic
        producing the tolerance (2x max within-cell seed gap, margin space).

Usage (from the worktree root):
    uv run python scripts/i627_inventory.py [--skip-hub]

``--skip-hub`` skips the network-bound HF revision/adapter checks (offline
re-runs of the manifest derivation only); the registered Phase-0 run keeps
them ON.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.leakage_vs_install_627 import (  # noqa: E402
    ARMS,
    BRACKET_WIDTH_GUARD,
    BRACKET_WIDTH_SENSITIVITY,
    COLLISION_SOURCE,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    INSTALL_TARGET,
    REGISTERED_BRACKETS,
    SEED,
    SMOKE_CELL,
    SMOKE_CELL_COMMITTED_OWN_RATE,
    SOURCES,
    adapter_hub_prefix,
    registered_cells,
)
from explore_persona_space.experiments.leakage_vs_install_627.marker601 import (  # noqa: E402
    load_all_onpolicy,
    matched_pairs,
    seed_gap_tolerance,
)
from explore_persona_space.experiments.sycophancy_posonly_608 import (  # noqa: E402
    EXPECTED_SHA256,
    FROZEN_DATA_PREFIX,
    TRAINED_NEGATIVES_BY_SOURCE,
)

log = logging.getLogger("i627_inventory")

OUT_DIR = Path("eval_results/issue_627")
SUBCEILING_SUMMARY = Path(
    "eval_results/issue_608/sub-ceiling-install/analyze_summary_subceiling.json"
)
SUMMARY_608 = Path("eval_results/issue_608/analyze_summary_608.json")
ROOT_601 = Path("eval_results/issue_601")
BEHAVIORS_606 = ("sycophancy", "refusal", "refusal-ft-lr2e6-retrain")
ROOT_606 = Path("eval_results/issue_606")
ROOT_514 = Path("eval_results/issue_514")
EVAL_POOL_REPO_PATH = f"{FROZEN_DATA_PREFIX}/data/wrong_claims/eval_50.jsonl"

FOUR_FLOAT_FIELDS = (
    "delta_margin",
    "z_marker_g",
    "z_marker_b",
    "z_eos_g",
    "z_eos_b",
    "logZ_g",
    "logZ_b",
)
Z_FIELDS_FORBIDDEN_514 = ("z_marker_g", "z_eos_g", "logZ_g", "delta_margin")


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


# ---------------------------------------------------------------------------
# (a) #608 matched-install cell manifest
# ---------------------------------------------------------------------------


def derive_brackets(summary: dict) -> dict[str, dict[str, dict]]:
    """FIRST-crossing 0.50 bracket per (source, arm) from committed own-rate
    trajectories (plan §4 registered tie-break: multi-crossing trajectories
    take the FIRST crossing; the narrowest later pair is a sensitivity read)."""
    out: dict[str, dict[str, dict]] = {}
    for source in SOURCES:
        out[source] = {}
        per = summary["per_source"][source]
        for arm in ARMS:
            traj = {int(k): float(v) for k, v in per["own_rate"][arm].items()}
            steps = sorted(traj)
            crossings = [
                (steps[i], steps[i + 1])
                for i in range(len(steps) - 1)
                if min(traj[steps[i]], traj[steps[i + 1]])
                <= INSTALL_TARGET
                <= max(traj[steps[i]], traj[steps[i + 1]])
            ]
            if not crossings:
                raise RuntimeError(
                    f"{source}:{arm}: committed trajectory never crosses "
                    f"{INSTALL_TARGET} — the registered bracket table is stale; re-plan"
                )
            lo, hi = crossings[0]
            narrowest = min(crossings, key=lambda p: abs(traj[p[1]] - traj[p[0]]))
            width = abs(traj[hi] - traj[lo])
            flags: list[str] = []
            if source == COLLISION_SOURCE:
                flags.append("context_collision_excluded_from_primary")
            if width > BRACKET_WIDTH_GUARD:
                flags.append("width_guard_fail_0p60")
            if width > BRACKET_WIDTH_SENSITIVITY:
                flags.append("width_above_0p45_sensitivity")
            if traj[hi] >= 0.90:
                flags.append("upper_endpoint_near_ceiling")
            out[source][arm] = {
                "lo_step": lo,
                "lo_committed_rate": traj[lo],
                "hi_step": hi,
                "hi_committed_rate": traj[hi],
                "width": width,
                "n_crossings": len(crossings),
                "first_crossing": True,
                "narrowest_pair": (
                    None
                    if narrowest == (lo, hi)
                    else {
                        "lo_step": narrowest[0],
                        "hi_step": narrowest[1],
                        "lo_rate": traj[narrowest[0]],
                        "hi_rate": traj[narrowest[1]],
                    }
                ),
                "flags": flags,
                "committed_trajectory": {str(k): traj[k] for k in steps},
                "claim_clustered_se": per["claim_clustered_se"][arm],
                "fresh_base_own_rate_reused": per["fresh_base_own_rate_reused"],
            }
    return out


def assert_brackets_match_registered(derived: dict[str, dict[str, dict]]) -> None:
    """Phase-0 re-verification: the derived FIRST-crossing table must equal the
    registered plan §4 table exactly (a divergence is a re-plan surprise)."""
    for source in SOURCES:
        for arm in ARMS:
            reg_lo, reg_lo_r, reg_hi, reg_hi_r = REGISTERED_BRACKETS[source][arm]
            d = derived[source][arm]
            ok = (
                d["lo_step"] == reg_lo
                and d["hi_step"] == reg_hi
                and abs(d["lo_committed_rate"] - reg_lo_r) < 1e-6
                and abs(d["hi_committed_rate"] - reg_hi_r) < 1e-6
            )
            if not ok:
                raise AssertionError(
                    f"{source}:{arm}: derived bracket "
                    f"({d['lo_step']}:{d['lo_committed_rate']:.3f} -> "
                    f"{d['hi_step']}:{d['hi_committed_rate']:.3f}) != registered plan §4 "
                    f"({reg_lo}:{reg_lo_r} -> {reg_hi}:{reg_hi_r}) — Phase-0 surprise, re-plan"
                )


def build_cells_manifest(derived: dict[str, dict[str, dict]]) -> dict:
    cells = []
    for source, arm, step in registered_cells():
        b = derived[source][arm]
        role = "lo" if step == b["lo_step"] else "hi"
        if step not in (b["lo_step"], b["hi_step"]):
            raise AssertionError(f"{source}:{arm}:{step} not an endpoint of its bracket")
        cells.append(
            {
                "source": source,
                "arm": arm,
                "step": step,
                "role": role,
                "committed_own_rate": (
                    b["lo_committed_rate"] if role == "lo" else b["hi_committed_rate"]
                ),
                "adapter_hub_prefix": adapter_hub_prefix(arm, source, step),
                "bracket": {k: b[k] for k in ("lo_step", "hi_step", "width", "flags")},
            }
        )
    smoke = cells[0]
    if (smoke["source"], smoke["arm"], smoke["step"]) != SMOKE_CELL:
        raise AssertionError("smoke cell must be first in the manifest")
    if abs(smoke["committed_own_rate"] - SMOKE_CELL_COMMITTED_OWN_RATE) > 1e-6:
        raise AssertionError(
            f"smoke-cell committed own-rate {smoke['committed_own_rate']} != registered "
            f"{SMOKE_CELL_COMMITTED_OWN_RATE}"
        )
    return {
        "issue": 627,
        "selection_rule": "FIRST-crossing pair bracketing committed own-rate 0.50 "
        "(plan §4; multi-crossing tie-break = first crossing)",
        "install_target": INSTALL_TARGET,
        "width_guard": BRACKET_WIDTH_GUARD,
        "width_guard_sensitivity": BRACKET_WIDTH_SENSITIVITY,
        "smoke_cell": {
            "source": SMOKE_CELL[0],
            "arm": SMOKE_CELL[1],
            "step": SMOKE_CELL[2],
            "committed_own_rate": SMOKE_CELL_COMMITTED_OWN_RATE,
            "parity_tolerance": 0.08,
        },
        "cells": cells,
        "brackets": derived,
        "source_summary": str(SUBCEILING_SUMMARY),
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# §6 statistical-input asserts per slab
# ---------------------------------------------------------------------------


def verify_608(summary_sub: dict, summary_main: dict) -> dict:
    for source in SOURCES:
        per = summary_sub["per_source"][source]
        for arm in ARMS:
            for table in ("own_rate", "claim_clustered_se"):
                steps = {int(k) for k in per[table][arm]}
                if len(steps) != 9:
                    raise AssertionError(f"608 {source}:{arm}: {table} has {len(steps)} != 9 steps")
        float(per["fresh_base_own_rate_reused"])
    base_rates = summary_main["fresh_base_panel_rates"]
    if len(base_rates) != 24:
        raise AssertionError(f"fresh_base_panel_rates has {len(base_rates)} != 24 personas")
    excl = {}
    per_source = summary_main["h2"]["per_arm"]["contrastive_fresh_eval"]["per_source"]
    for source in SOURCES:
        lst = sorted(per_source[source]["excluded_trained_negatives"])
        if lst != sorted(TRAINED_NEGATIVES_BY_SOURCE[source]):
            raise AssertionError(
                f"608 {source}: machine-readable exclusion list {lst} != module registry "
                f"{sorted(TRAINED_NEGATIVES_BY_SOURCE[source])}"
            )
        excl[source] = lst
    return {
        "n_sources": len(SOURCES),
        "n_grid_steps": 9,
        "fresh_base_panel_personas": len(base_rates),
        "excluded_trained_negatives": excl,
        "files": [str(SUBCEILING_SUMMARY), str(SUMMARY_608)],
    }


def verify_601(cells: list[dict]) -> dict:
    n_records = 0
    for c in cells:
        if not c["checkpoints"]:
            raise AssertionError(f"601 {c['cell']} seed{c['seed']}: zero checkpoints")
        for ck in c["checkpoints"]:
            for persona, stats in ck["bystanders"].items():
                if stats["n_questions"] == 0:
                    raise AssertionError(f"601 {c['cell']}: {persona} has zero questions")
                n_records += stats["n_questions"]
    # Four-float spot assert on the raw file of the first cell (loader already
    # fails loud on missing delta_margin; this asserts the FULL field set).
    with open(cells[0]["path"]) as f:
        raw = json.load(f)
    rec = next(iter(next(iter(raw["checkpoints"][0]["held_out"].values())).values()))
    missing = [k for k in FOUR_FLOAT_FIELDS if k not in rec]
    if missing:
        raise AssertionError(f"601 four-float contract violated: missing {missing}")
    return {
        "n_onpolicy_cells": len(cells),
        "n_bystander_question_records": n_records,
        "cells": sorted({f"{c['cell']}_seed{c['seed']}" for c in cells}),
    }


def verify_606() -> dict:
    out = {}
    for beh in BEHAVIORS_606:
        path = ROOT_606 / beh / "analysis.json"
        with open(path) as f:
            a = json.load(f)
        for key in ("per_cell_tables", "s_stage_b", "arm_bracket", "headline"):
            if key not in a:
                raise AssertionError(f"606 {beh}: missing {key} in {path}")
        cells = [c for c in a["per_cell_tables"] if c != "base"]
        if not cells:
            raise AssertionError(f"606 {beh}: no non-base cells")
        c0 = a["per_cell_tables"][cells[0]]
        p0 = next(iter(c0.values()))
        for key in ("delta_clean", "rate_clean"):
            if key not in p0:
                raise AssertionError(f"606 {beh}: per-persona table missing {key}")
        out[beh] = {
            "n_cells": len(cells),
            "n_personas": len(c0),
            "s_stage_b": a["s_stage_b"],
            "file": str(path),
        }
    return out


def verify_514() -> dict:
    leaves = sorted(p for p in ROOT_514.glob("*.json") if not p.name.startswith("_"))
    leaves = [p for p in leaves if p.name not in ("analysis.json", "analysis_514.json")]
    if not leaves:
        raise AssertionError("514: no leaf eval JSONs")
    n_personas = None
    for p in leaves:
        with open(p) as f:
            leaf = json.load(f)
        held = leaf["delta_g_held_out"]
        n_personas = len(held)
        rec = next(iter(next(iter(held.values())).values()))
        for k in ("trained_logp", "base_logp", "delta_g", "trained_argmax_marker"):
            if k not in rec:
                raise AssertionError(f"514 {p.name}: held-out record missing {k}")
        present = [k for k in Z_FIELDS_FORBIDDEN_514 if k in rec]
        if present:
            raise AssertionError(
                f"514 {p.name}: z-fields {present} PRESENT — the registered log-prob-only "
                f"restriction (plan §11) was derived from their verified ABSENCE; re-plan"
            )
        if "delta_g_source" not in leaf:
            raise AssertionError(f"514 {p.name}: missing delta_g_source")
    with open(ROOT_514 / "analysis.json") as f:
        agg = json.load(f)
    return {
        "n_leaf_cells": len(leaves),
        "leaf_files": [str(p) for p in leaves],
        "n_held_out_personas": n_personas,
        "n_aggregate_cells": len(agg["cells"]),
        "aggregate_cells": [(c["cell"], c["arm"]) for c in agg["cells"]],
        "z_fields_absent_confirmed": True,
    }


# ---------------------------------------------------------------------------
# Hub checks (network)
# ---------------------------------------------------------------------------


def verify_hub() -> dict:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files

    api = HfApi()
    model_sha = api.repo_info(HF_MODEL_REPO).sha
    data_sha = api.repo_info(HF_DATA_REPO, repo_type="dataset").sha
    model_files = set(list_repo_files(HF_MODEL_REPO))
    missing = []
    for source, arm, step in registered_cells():
        prefix = adapter_hub_prefix(arm, source, step)
        for fname in ("adapter_config.json", "adapter_model.safetensors"):
            if f"{prefix}/{fname}" not in model_files:
                missing.append(f"{prefix}/{fname}")
    if missing:
        raise RuntimeError(f"{len(missing)} adapter files missing on the Hub: {missing[:6]}")
    data_files = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    if EVAL_POOL_REPO_PATH not in data_files:
        raise RuntimeError(f"probe file missing on the Hub: {EVAL_POOL_REPO_PATH}")
    smoke_prefix = adapter_hub_prefix(SMOKE_CELL[1], SMOKE_CELL[0], SMOKE_CELL[2])
    cfg_path = hf_hub_download(HF_MODEL_REPO, f"{smoke_prefix}/adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if not (cfg["r"] == 32 and cfg["lora_alpha"] == 64 and cfg.get("use_rslora") is True):
        raise RuntimeError(
            f"smoke-cell adapter_config gauge mismatch vs plan §10: r={cfg['r']}, "
            f"alpha={cfg['lora_alpha']}, use_rslora={cfg.get('use_rslora')}"
        )
    return {
        "model_repo": HF_MODEL_REPO,
        "model_repo_revision": model_sha,
        "data_repo": HF_DATA_REPO,
        "data_repo_revision": data_sha,
        "n_adapter_cells_verified": 24,
        "eval_pool_repo_path": EVAL_POOL_REPO_PATH,
        "eval_pool_sha256_pin": EXPECTED_SHA256[EVAL_POOL_REPO_PATH],
        "smoke_adapter_config": {k: cfg.get(k) for k in ("r", "lora_alpha", "use_rslora")},
    }


def producing_task_status() -> dict:
    """§12 assumption 15 re-check: producing tasks not retracted (archived)."""
    from explore_persona_space.task_workflow import registry_path

    with open(registry_path()) as f:
        registry = json.load(f)
    out = {}
    for tid in ("601", "606", "608", "514"):
        row = registry["tasks"][tid]
        status = row["status"]
        if status == "archived":
            raise RuntimeError(f"producing task #{tid} is ARCHIVED — inputs retracted; re-plan")
        out[tid] = status
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #627 Phase 0 — inventory + registered manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-hub", action="store_true", help="Skip network-bound HF checks.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=p0_inventory] %(message)s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUBCEILING_SUMMARY) as f:
        summary_sub = json.load(f)
    with open(SUMMARY_608) as f:
        summary_main = json.load(f)

    derived = derive_brackets(summary_sub)
    assert_brackets_match_registered(derived)
    cells_manifest = build_cells_manifest(derived)
    with open(OUT_DIR / "matched_install_cells.json", "w") as f:
        json.dump(cells_manifest, f, indent=2)
    log.info("manifest (a): 24 cells -> %s", OUT_DIR / "matched_install_cells.json")

    cells_601 = load_all_onpolicy(ROOT_601)
    tolerance = seed_gap_tolerance(cells_601)
    tolerance.update(git_commit=_git_sha(), timestamp_utc=datetime.now(UTC).isoformat())
    with open(OUT_DIR / "marker_tolerance.json", "w") as f:
        json.dump(tolerance, f, indent=2)
    log.info(
        "manifest (c): tolerance=%.4f margin (max seed gap %.4f) -> %s",
        tolerance["tolerance_margin"],
        tolerance["max_within_cell_seed_gap_margin"],
        OUT_DIR / "marker_tolerance.json",
    )

    pairs = matched_pairs(cells_601, tolerance["tolerance_margin"])
    if not pairs:
        raise RuntimeError(
            "ZERO #601 matched-install pairs under the registered tolerance — H2 has no "
            "inputs (plan §13 item 4(b): this must fail loud at Phase 0, not silently at "
            "analysis time). Report the install gaps instead; re-plan H2."
        )
    pairs_manifest = {
        "issue": 627,
        "tolerance_margin": tolerance["tolerance_margin"],
        "tolerance_manifest": str(OUT_DIR / "marker_tolerance.json"),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    with open(OUT_DIR / "marker_matched_pairs.json", "w") as f:
        json.dump(pairs_manifest, f, indent=2)
    log.info(
        "manifest (b): %d matched pairs -> %s", len(pairs), OUT_DIR / "marker_matched_pairs.json"
    )

    inventory = {
        "issue": 627,
        "verified": {
            "608": verify_608(summary_sub, summary_main),
            "601": verify_601(cells_601),
            "606": verify_606(),
            "514": verify_514(),
        },
        "producing_task_status": producing_task_status(),
        "hub": (None if args.skip_hub else verify_hub()),
        "manifests": {
            "matched_install_cells": str(OUT_DIR / "matched_install_cells.json"),
            "marker_matched_pairs": str(OUT_DIR / "marker_matched_pairs.json"),
            "marker_tolerance": str(OUT_DIR / "marker_tolerance.json"),
        },
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
    }
    with open(OUT_DIR / "inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)
    log.info("inventory -> %s", OUT_DIR / "inventory.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
