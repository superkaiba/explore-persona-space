#!/usr/bin/env python3
"""Task #612 predictor-v3 Bucket 4 — leakage-predictor bake-off (VM, CPU+judge).

OFF-POD analysis (plan v3 §4.5 / §6.5): reads the per-cell matched-install
full-panel eval trees the dispatcher uploaded, judges the bystander completions
(Haiku, the locked eval judge), computes per-(source, bystander) leakage
Delta = trained - base at the band-entry checkpoint, and runs the predictor bake-off
(base prior / cosine-to-source / #623 persona-vector alignment) — Spearman + BCa
CI per source, partial-Spearman w/ collinearity gate, pairwise predictor
correlations, per-source + pooled verdict, Bonferroni-corrected.

Leakage Delta source: for one band-entry cell (arm, source, seed), the bystander
agreement rate at the matched-install checkpoint MINUS the bystander base rate
(from the v1 base pass / source_baseline). The on-policy (arm_onpolicy) cells
carry the headline; canned cells are the data-construction control. Per source we
average the arm_onpolicy band-entry leakage over seeds (the on-policy reach the
predictors should explain). Cells that never crossed the band are read at their
closest-approach checkpoint and FLAGGED (never silently skipped).

CLI (VM, after the pod's predictor-v3 eval trees land):
    uv run python scripts/issue612_predictor_bakeoff.py \
        --slab-root eval_results/issue_612 \
        --panels-dir eval_results/issue_612/onpolicy_predictor/panels \
        --out eval_results/issue_612/onpolicy_predictor/bakeoff/predictor_bakeoff.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    I623_COSINE_MATRIX_RELPATH,
    I623_SYC_I_RELPATH,
    JUDGE_MODEL,
    SEEDS,
    SOURCES,
    repo_root_from_module,
    v3_cell_dir,
)
from explore_persona_space.experiments.sycophancy_onpolicy_612.predictor_bakeoff import (  # noqa: E402
    run_bakeoff,
)

log = logging.getLogger("issue612_predictor_bakeoff")

LEAKAGE_ARM = "arm_onpolicy"  # the headline on-policy reach the predictors explain


def _judge_rate(panel_file: Path, concurrency: int) -> float:
    """Haiku agreement rate over one panel eval JSON's completions (checkpointed
    sibling judgments/<persona>.json reused if present)."""
    from explore_persona_space.experiments.sycophancy_onpolicy_612.judge import (
        judge_batch,
        serialize_verdicts,
    )

    payload = json.loads(panel_file.read_text())
    persona = payload["panel_persona"]
    jdir = panel_file.parent / "judgments"
    jpath = jdir / f"{persona}.json"
    if jpath.exists():
        rows = json.loads(jpath.read_text())["verdicts"]
        verified = [v for v in rows if not v.get("error")]
        if verified:
            return sum(1 for v in verified if v.get("agreed")) / len(verified)
    records = payload["completions"]
    rollouts = [{"wrong_claim": r["claim"], "completion": r["completion"]} for r in records]
    verdicts = asyncio.run(judge_batch(rollouts, model=JUDGE_MODEL, max_concurrency=concurrency))
    rows = serialize_verdicts(verdicts)
    jdir.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps({"panel": persona, "n_verdicts": len(rows), "verdicts": rows}))
    verified = [v for v in rows if not v.get("error")]
    if not verified:
        raise RuntimeError(f"{panel_file}: zero verified verdicts")
    return sum(1 for v in verified if v.get("agreed")) / len(verified)


def _band_entry_eval_dir(cell_dir: Path) -> tuple[Path, dict]:
    """The matched-install full-panel eval dir for a cell, from its band_entry.json."""
    band = json.loads((cell_dir / "band_entry.json").read_text())
    step = band["band_entry_step"]
    if step is None:
        # closest-approach: the max-self-delta step (still reported, flagged)
        per_step = band["per_step"]
        step = int(max(per_step, key=lambda s: per_step[s]["self_delta"]))
    eval_dir = cell_dir / f"matched_install_step_{step}"
    return eval_dir, band


def _base_rate(slab_root: Path, persona: str) -> float | None:
    """Bystander base agreement rate from the v1 base pass judgments."""
    jpath = slab_root / "base" / "judgments" / f"{persona}.json"
    if not jpath.exists():
        return None
    rows = json.loads(jpath.read_text())["verdicts"]
    verified = [v for v in rows if not v.get("error")]
    if not verified:
        return None
    return sum(1 for v in verified if v.get("agreed")) / len(verified)


def collect_leakage(
    slab_root: Path, panels_dir: Path, *, concurrency: int
) -> tuple[dict[str, dict[str, float]], dict]:
    """Per source -> {bystander: leakage Delta = trained_rate - base_rate} at the
    band-entry checkpoint, averaged over seeds for LEAKAGE_ARM. Returns
    (leakage_by_source, diagnostics)."""
    leakage: dict[str, dict[str, float]] = {}
    diag: dict = {"per_cell": {}, "missing": []}
    for source in SOURCES:
        panel_path = panels_dir / source / "panel.json"
        if not panel_path.exists():
            diag["missing"].append(f"panel:{source}")
            continue
        panel = json.loads(panel_path.read_text())
        if panel["status"] != "ok":
            continue
        bystanders = sorted(panel["bystanders"])
        per_seed: dict[str, list[float]] = {b: [] for b in bystanders}
        for seed in SEEDS:
            cell_dir = v3_cell_dir(slab_root, source, LEAKAGE_ARM, seed)
            if not (cell_dir / "band_entry.json").exists():
                diag["missing"].append(f"cell:{source}:{LEAKAGE_ARM}:{seed}")
                continue
            eval_dir, band = _band_entry_eval_dir(cell_dir)
            diag["per_cell"][f"{source}:{LEAKAGE_ARM}:{seed}"] = {
                "band_entry_step": band["band_entry_step"],
                "band_entry_status": band["band_entry_status"],
            }
            for b in bystanders:
                pf = eval_dir / f"sycophancy_eval_{b}.json"
                if not pf.exists():
                    continue
                trained = _judge_rate(pf, concurrency)
                base = _base_rate(slab_root, b)
                if base is None:
                    continue
                per_seed[b].append(trained - base)
        cell_leak = {b: sum(v) / len(v) for b, v in per_seed.items() if v}
        if cell_leak:
            leakage[source] = cell_leak
    return leakage, diag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_612"))
    parser.add_argument(
        "--panels-dir",
        type=Path,
        default=Path("eval_results/issue_612/onpolicy_predictor/panels"),
    )
    parser.add_argument(
        "--panel-set", type=Path, default=Path("data/issue_612/panel/panel_set.json")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results/issue_612/onpolicy_predictor/bakeoff/predictor_bakeoff.json"),
    )
    parser.add_argument("--judge-concurrency", type=int, default=24)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [phase=bakeoff] %(message)s", stream=sys.stdout
    )

    repo = repo_root_from_module()
    i623_cosine = repo / I623_COSINE_MATRIX_RELPATH
    i623_syc = repo / I623_SYC_I_RELPATH
    if not i623_cosine.exists():
        raise FileNotFoundError(f"#623 cosine_matrix missing: {i623_cosine}")

    leakage, diag = collect_leakage(
        args.slab_root, args.panels_dir, concurrency=args.judge_concurrency
    )
    if not leakage:
        raise RuntimeError(
            f"no leakage cells collected under {args.slab_root} — has the predictor-v3 "
            f"sweep run + uploaded? (missing: {diag['missing'][:10]})"
        )
    result = run_bakeoff(
        leakage_by_source=leakage,
        panels_dir=args.panels_dir,
        panel_set_path=args.panel_set,
        i623_cosine_matrix=i623_cosine,
        i623_syc_i=i623_syc,
    )
    result["leakage_collection_diagnostics"] = diag
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    log.info(
        "predictor bake-off -> %s (kept sources: %s; pooled winner: %s)",
        args.out,
        result["kept_sources"],
        result["pooled"].get("winner"),
    )
    for source, rec in result["per_source"].items():
        log.info(
            "  %s: verdict=%s a=%.2f b=%.2f c=%s",
            source,
            rec["verdict"]["winner"],
            rec["predictors"]["base_prior"]["spearman_rho"] or float("nan"),
            rec["predictors"]["cosine_to_source"]["spearman_rho"] or float("nan"),
            rec["predictors"]["pv_alignment"]["spearman_rho"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
