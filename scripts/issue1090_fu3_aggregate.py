#!/usr/bin/env python
"""#1090 fu3 (posonly-contexts-parallel-matrix) — OFF-pod aggregation (plan v5 P3b, §6).

Consumes the per-cell fu3 run tree (``issue1090_fu3_worker.py`` output, staged
locally by the orchestrator) and produces the round's summary JSONs under
``eval_results/issue_1090/fu3/``:

1. Judges the Tier-2 (own-context install) + bystander-panel completions via the
   round-1 ``issue1090_run._judge_rate`` (graded judge -> the sanctioned
   ``eval.batch_judge`` crossover client: sets >= the tier threshold go Batch API);
   ``max_tokens=300`` EXPLICIT (llm-judging rule 23); rubric-keyed cache with a
   SEPARATE cache dir per rubric/behavior (rule 22). Checkpoint-per-cell.
2. Install-by-context comparison (own-context install delta x context arm x regime).
3. Band-hit table (Tier-2 trained rate vs the recipe band + per-rung selection).
4. The paired contrastive-vs-posonly leakage contrast at matched install:
   95% CI via ONE batched bootstrap resample (2000 draws, seed 42 — no per-draw
   loop; §6 "Batched draw battery"), plus the paired-r companion via the named
   batched helper ``analysis/null_battery.py::bootstrap_ci_matched_r`` (identity
   direction construction: project(x, [1.0]) == x, so r == Pearson(con, pos)).

Runs on the VM (CPU + judge API only — the pod is released before P3b, §9).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env + shared-VM thread caps BEFORE numpy/torch-adjacent imports

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1090_fu3_cells as fu3_cells  # noqa: E402
import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_run as run1090  # noqa: E402

from explore_persona_space.analysis.null_battery import bootstrap_ci_matched_r  # noqa: E402

logger = logging.getLogger("issue1090.fu3.aggregate")

N_BOOT_DEFAULT = 2000  # plan §6 leakage-contrast row
SEED_DEFAULT = 42


def _repro_meta(args: argparse.Namespace) -> dict:
    """Reproducibility metadata for every emitted JSON (CLAUDE.md requirement)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            env=None,
            cwd=_SCRIPTS_DIR,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    return {
        "git_commit": sha,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "numpy_version": np.__version__,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "judge_max_tokens": args.max_tokens,
        "run_root": str(args.run_root),
    }


def _read_completions(path: Path) -> tuple[list[str], list[list[str]]]:
    payload = json.loads(path.read_text())
    return payload["questions"], payload["completions"]


def judge_cell(args: argparse.Namespace, row: dict, out_dir: Path) -> dict | None:
    """Judge one cell's Tier-2 + bystander completions (checkpointed; resume-safe).

    Returns None (recorded in the summary) when the cell has no trained output.
    """
    shim = fu3w.run_cell_shim(row)
    cell_root = Path(args.run_root) / shim.slug
    eval_path = out_dir / "fu3_cell_evals" / f"{shim.slug}.json"
    if eval_path.exists():
        return run1090._read_json(eval_path)
    build_path = cell_root / "build_result.json"
    if not build_path.exists():
        return None
    build = run1090._read_json(build_path)
    # Separate judge cache dir PER RUBRIC/behavior (llm-judging rule 22); the
    # rubric_fingerprint key inside batch_judge is the load-bearing guard.
    judge_root = out_dir / "judge" / row["behavior"]
    rec: dict[str, Any] = {
        "cell_id": row["cell_id"],
        "slug": shim.slug,
        "behavior": row["behavior"],
        "context_id": row["context_id"],
        "regime": row["regime"],
        "tier": row["tier"],
        "generator": row["generator"],
        "selection": build.get("selection"),
        "tier2": {},
        "bystanders": [],
    }
    # Tier-2 own-context install read (selected checkpoint vs base).
    for state in ("trained", "base"):
        qs, comps = _read_completions(
            cell_root / "tier2" / f"completions__{state}__{row['context_id']}.json"
        )
        rec["tier2"][state] = run1090._judge_rate(
            row["behavior"],
            qs,
            comps,
            tag=f"{row['cell_id']}-t2-{state}",
            n_draws=run1090.TIER2_JUDGE_DRAWS,
            judge_root=judge_root,
            max_tokens=args.max_tokens,
        )
        if state == "trained" and row["behavior"] == "formatting":
            # Plan §6 formatting companion: structural rate PRIMARY + judged
            # spot-check SECONDARY (v4 parity; round-1 review Major — the
            # companion was wired nowhere in fu3).
            rec["formatting_spotcheck"] = run1090._formatting_spotcheck(
                qs,
                comps,
                n_draws=run1090.TIER2_JUDGE_DRAWS,
                judge_root=judge_root,
                max_tokens=args.max_tokens,
            )
    rec["install_delta"] = rec["tier2"]["trained"]["rate"] - rec["tier2"]["base"]["rate"]
    # Bystander-panel leakage read (Tier-1 params; source==bystander rows distinct).
    bys_manifest = run1090._read_json(cell_root / "bystander" / "manifest.json")
    for bctx in bys_manifest["contexts"]:
        cid = bctx["context_id"]
        brec: dict[str, Any] = {
            "context_id": cid,
            "is_source_context": bool(bctx["is_source_context"]),
        }
        for state in ("trained", "base"):
            qs, comps = _read_completions(
                cell_root / "bystander" / f"completions__{state}__{cid}.json"
            )
            brec[state] = run1090._judge_rate(
                row["behavior"],
                qs,
                comps,
                tag=f"{row['cell_id']}-by-{cid}-{state}",
                n_draws=run1090.TIER1_JUDGE_DRAWS,
                judge_root=judge_root,
                max_tokens=args.max_tokens,
            )
        brec["leak_delta"] = brec["trained"]["rate"] - brec["base"]["rate"]
        rec["bystanders"].append(brec)
    held_out = [b["leak_delta"] for b in rec["bystanders"] if not b["is_source_context"]]
    rec["leakage_mean_held_out"] = float(np.mean(held_out)) if held_out else None
    lo, hi = run1090.JUDGED_RATE_BAND
    rec["band"] = [lo, hi]
    rec["band_hit"] = bool(lo <= rec["tier2"]["trained"]["rate"] <= hi)
    run1090._atomic_write_json(eval_path, rec)  # checkpoint-per-cell
    return rec


def paired_delta_ci(deltas: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    """95% CI on the mean paired delta — ONE batched resample over the pair
    axis (the §6 no-per-draw-loop commitment; the null_battery gather pattern)."""
    deltas = np.asarray(deltas, dtype=np.float64)
    n = deltas.size
    if n == 0 or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))  # (n_boot, n): one vectorized gather
    boots = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def paired_r_with_ci(con: np.ndarray, pos: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """Paired-r companion via the NAMED batched helper (plan §6):
    ``bootstrap_ci_matched_r`` with the identity direction (rb=[1.0], layer 0),
    so project(x) == x and r == Pearson(contrastive, posonly leakage)."""
    con = np.asarray(con, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    if con.size < 3:
        return {"r": None, "ci95": [None, None], "n": int(con.size)}
    r = float(np.corrcoef(con, pos)[0, 1])
    lo, hi = bootstrap_ci_matched_r(
        con[:, None, None], np.ones((1, 1)), pos, 0, n_boot=n_boot, seed=seed
    )
    return {"r": r, "ci95": [lo, hi], "n": int(con.size)}


def aggregate(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _repro_meta(args)
    wanted = {tok.strip() for tok in args.cells.split(",")} if args.cells else None
    rows = [
        r for r in fu3_cells.CELLS if r["trains"] and (wanted is None or r["cell_id"] in wanted)
    ]
    evals: dict[str, dict] = {}
    missing: list[str] = []
    for row in rows:
        rec = judge_cell(args, row, out_dir)
        if rec is None:
            missing.append(row["cell_id"])
            logger.warning("[fu3-agg] %s: no build_result under run root — skipped", row["cell_id"])
        else:
            evals[row["cell_id"]] = rec

    # ── install-by-context comparison + band-hit table ────────────────────────
    install_rows = [
        {
            "cell_id": cid,
            "behavior": e["behavior"],
            "context_id": e["context_id"],
            "regime": e["regime"],
            "install_delta": e["install_delta"],
            "rate_trained": e["tier2"]["trained"]["rate"],
            "rate_base": e["tier2"]["base"]["rate"],
            "band_hit": e["band_hit"],
        }
        for cid, e in sorted(evals.items())
    ]
    run1090._atomic_write_json(
        out_dir / "fu3_install_by_context.json", {"meta": meta, "rows": install_rows}
    )
    band_rows = [
        {
            "cell_id": cid,
            "band": e["band"],
            "band_hit": e["band_hit"],
            "rate_trained": e["tier2"]["trained"]["rate"],
            "selection": e["selection"],
        }
        for cid, e in sorted(evals.items())
    ]
    run1090._atomic_write_json(
        out_dir / "fu3_band_hit_table.json", {"meta": meta, "rows": band_rows}
    )

    # ── paired contrastive-vs-posonly leakage contrast at matched install ─────
    pairs = []
    for cid, e in sorted(evals.items()):
        if not cid.endswith("-con"):
            continue
        pos_id = cid[: -len("-con")] + "-pos"
        p = evals.get(pos_id)
        if p is None or e["leakage_mean_held_out"] is None or p["leakage_mean_held_out"] is None:
            continue
        pairs.append(
            {
                "pair": cid[: -len("-con")],
                "behavior": e["behavior"],
                "context_id": e["context_id"],
                "leak_contrastive": e["leakage_mean_held_out"],
                "leak_posonly": p["leakage_mean_held_out"],
                "delta_con_minus_pos": e["leakage_mean_held_out"] - p["leakage_mean_held_out"],
                # Dose-matched iff BOTH arms selected inside the install band;
                # otherwise the pair carries the explicit §D6 caveat.
                "install_matched": bool(e["band_hit"] and p["band_hit"]),
            }
        )

    def _contrast(subset: list[dict], label: str) -> dict:
        deltas = np.array([p["delta_con_minus_pos"] for p in subset], dtype=np.float64)
        con = np.array([p["leak_contrastive"] for p in subset], dtype=np.float64)
        pos = np.array([p["leak_posonly"] for p in subset], dtype=np.float64)
        lo, hi = paired_delta_ci(deltas, n_boot=args.n_boot, seed=args.seed)
        return {
            "label": label,
            "n_pairs": int(deltas.size),
            "mean_delta_con_minus_pos": float(deltas.mean()) if deltas.size else None,
            "delta_ci95": [lo, hi],
            "paired_r": paired_r_with_ci(con, pos, n_boot=args.n_boot, seed=args.seed),
        }

    matched = [p for p in pairs if p["install_matched"]]
    contrast = {
        "meta": meta,
        "pairs": pairs,
        "contrast_matched_install": _contrast(matched, "matched-install pairs"),
        "contrast_all_pairs": _contrast(pairs, "all pairs (incl. unmatched-install caveat)"),
        "unmatched_install_pairs": [p["pair"] for p in pairs if not p["install_matched"]],
    }
    run1090._atomic_write_json(out_dir / "fu3_leakage_contrast.json", contrast)

    summary = {
        "meta": meta,
        "n_cells_judged": len(evals),
        "cells_missing": missing,
        "n_pairs": len(pairs),
        "n_pairs_matched_install": len(matched),
        "headline_contrast": contrast["contrast_matched_install"],
        "outputs": [
            "fu3_install_by_context.json",
            "fu3_band_hit_table.json",
            "fu3_leakage_contrast.json",
            "fu3_cell_evals/",
        ],
    }
    run1090._atomic_write_json(out_dir / "fu3_summary.json", summary)
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="#1090 fu3 off-pod judge + aggregation (P3b)")
    ap.add_argument("--run-root", required=True, help="the fu3 out_root (worker output tree)")
    ap.add_argument("--out", default="eval_results/issue_1090/fu3")
    ap.add_argument("--cells", default=None, help="comma cell_id subset (smoke parity)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=fu3w.JUDGE_MAX_TOKENS,
        help="judge response budget (llm-judging rule 23; >=300 for reason-first rubrics)",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    summary = aggregate(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
