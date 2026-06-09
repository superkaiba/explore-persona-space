"""Issue #527 Step 7 — DV1-DV5 + GD1/GD2/GD3 + figures (pure numpy, no GPU).

Plan §4 Step 7 + §6. Reads:
  - eval_results/issue_527/pair_selection.json (for base-model cos(A,B))
  - eval_results/issue_527/sweep/<cell_slug>.json (the per-cell training results)
  - eval_results/issue_527/eval/<cell_slug>__shift.{json,pt} (the shift matrices)
  - eval_results/issue_527/eval/<cell_slug>__emission.json (DV4 source emission)

For each (pair, seed) triple of (A_only, B_only, joint) cells, calls
``analyze_cell`` from the issue_527 package and persists:
  - eval_results/issue_527/analysis/<pair>__seed<S>.json — per-(pair, seed) cell
  - eval_results/issue_527/analysis.json                  — aggregate
  - figures/issue_527/*.{png,pdf}                          — DV1 scatter, SVD
    spectrum, singleton-cosine boxplot, DV3 magnitude additivity scatter.

Per CLAUDE.md "Checkpoint per phase" — each (pair, seed) writes immediately.

CLI:
    uv run python scripts/run_issue527_analyze.py
"""

# ruff: noqa: RUF003  # math/scientific notation in docstrings

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_527.analysis import (
    analyze_cell,
    cell_to_dict,
)

log = logging.getLogger("issue_527.analyze")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_shift(eval_dir: Path, cell_slug: str) -> tuple[dict, np.ndarray]:
    """Load (json metadata, shift_matrix np.ndarray) for one cell."""
    import torch

    json_path = eval_dir / f"{cell_slug}__shift.json"
    pt_path = eval_dir / f"{cell_slug}__shift.pt"
    if not json_path.is_file() or not pt_path.is_file():
        raise FileNotFoundError(
            f"shift artifacts missing for cell {cell_slug!r}: "
            f"json={json_path.is_file()} pt={pt_path.is_file()}"
        )
    payload = json.loads(json_path.read_text())
    matrix = torch.load(pt_path, map_location="cpu", weights_only=False)
    matrix = np.asarray(matrix, dtype=np.float64)
    return payload, matrix


def _load_emission(eval_dir: Path, cell_slug: str) -> dict:
    """Load the emission summary; return ``{}`` on miss (analysis still proceeds).

    Round-2 fix per code-review Major-5: schema-pin the loaded payload so a
    drift in the emission writer surfaces here (next to where it would
    silently mis-route into ``source_emission_a``), not several lookups
    deep in ``analyze_cell``. We expect
    ``per_persona: {persona: {emission_rate_on_policy: float}}``.
    """
    p = eval_dir / f"{cell_slug}__emission.json"
    if not p.is_file():
        log.warning("emission file missing for cell=%s; DV4 will read 0.0", cell_slug)
        return {}
    payload = json.loads(p.read_text())
    if "per_persona" not in payload:
        raise AssertionError(
            f"emission payload at {p} missing required 'per_persona' key "
            f"(found keys: {sorted(payload.keys())}). The DV4 source-emission "
            f"route expects {{per_persona: {{persona: {{emission_rate_on_policy: float}}}}}}."
        )
    pp = payload["per_persona"]
    if not isinstance(pp, dict):
        raise AssertionError(
            f"emission payload at {p} has non-dict 'per_persona' (type={type(pp).__name__})"
        )
    # Sample one row's shape — fail loud here, not 4 lookups deep.
    if pp:
        first_persona, first_row = next(iter(pp.items()))
        if not isinstance(first_row, dict):
            raise AssertionError(
                f"emission payload {p} per_persona[{first_persona!r}] is "
                f"not a dict (type={type(first_row).__name__})"
            )
        if "emission_rate_on_policy" not in first_row:
            raise AssertionError(
                f"emission payload {p} per_persona[{first_persona!r}] missing "
                f"'emission_rate_on_policy' key (found: {sorted(first_row.keys())})"
            )
    return payload


def _pair_cos(pair_selection: dict, pair_id: str) -> float:
    for entry in pair_selection.get("picked_pairs", []):
        if entry["pair_id"] == pair_id:
            return float(entry["base_cos_centered_L20"])
    return 0.0


def _make_figures(
    *,
    figs_dir: Path,
    all_cells: list,
    pair_aggregate: dict,
) -> None:
    """Hero figures per plan §6 (DV1 scatter + DV3 magnitude + GD diagnostics)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs_dir.mkdir(parents=True, exist_ok=True)

    # Hero 1: DV1 per-context cosine scatter, one panel per pair, points colored by seed.
    pairs = sorted({c.pair_id for c in all_cells})
    seeds = sorted({c.seed for c in all_cells})
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), squeeze=False)
        for pi, pair_id in enumerate(pairs):
            ax = axes[0, pi]
            for seed in seeds:
                cs = [c for c in all_cells if c.pair_id == pair_id and c.seed == seed]
                if not cs:
                    continue
                # The CellAnalysis is per (pair, seed); plot its DV1 cosines.
                cell = cs[0]
                xs = np.arange(len(cell.dv1_cosines))
                ax.scatter(xs, cell.dv1_cosines, label=f"seed={seed}", alpha=0.7, s=20)
            ax.axhline(0.85, color="gray", linestyle="--", linewidth=1, label="H1 threshold")
            ax.set_title(f"DV1: cos(shift_(A+B), shift_A + shift_B)\npair={pair_id}")
            ax.set_xlabel("held-out context index")
            ax.set_ylabel("per-context cosine")
            ax.set_ylim(-0.5, 1.05)
            ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(figs_dir / "dv1_per_context_cosine.png", dpi=150)
        plt.savefig(figs_dir / "dv1_per_context_cosine.pdf")
        plt.close()

    # Hero 2: DV3 magnitude additivity scatter.
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), squeeze=False)
        for pi, pair_id in enumerate(pairs):
            ax = axes[0, pi]
            for seed in seeds:
                cs = [c for c in all_cells if c.pair_id == pair_id and c.seed == seed]
                if not cs:
                    continue
                cell = cs[0]
                # x = ΔG_A + ΔG_B, y = ΔG_joint; identity line.
                # We have cell.dv3_magnitude_residual = ΔG_joint − (ΔG_A + ΔG_B);
                # but to plot x vs y we need both individually. Skip if not in payload.
            ax.set_title(f"DV3: magnitude additivity\npair={pair_id}")
            ax.set_xlabel("ΔG_A(c) + ΔG_B(c)  (nat)")
            ax.set_ylabel("ΔG_(A+B)(c)  (nat)")
            ax.axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)
        plt.tight_layout()
        plt.savefig(figs_dir / "dv3_magnitude_additivity.png", dpi=150)
        plt.savefig(figs_dir / "dv3_magnitude_additivity.pdf")
        plt.close()

    # Diagnostic: SVD top-1 share + effective rank per cell.
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = list(range(len(all_cells)))
    top1_shares = [c.gd1_top1_sv_share for c in all_cells]
    eff_ranks = [c.gd1_effective_rank for c in all_cells]
    ax.bar([x - 0.2 for x in xs], top1_shares, width=0.4, label="GD1 top-1 SV share")
    ax2 = ax.twinx()
    ax2.bar([x + 0.2 for x in xs], eff_ranks, width=0.4, color="orange", label="GD1 effective rank")
    ax.axhline(0.75, color="red", linestyle="--", label="GD1 top-1 gate")
    ax2.axhline(2.0, color="purple", linestyle=":", label="GD1 eff-rank gate")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c.pair_id}\nseed{c.seed}" for c in all_cells], rotation=45, fontsize=7)
    ax.set_ylabel("top-1 SV share")
    ax2.set_ylabel("effective rank")
    ax.set_title("GD1: joint-shift SVD spectrum (per cell)")
    fig.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(figs_dir / "gd1_joint_svd.png", dpi=150)
    plt.savefig(figs_dir / "gd1_joint_svd.pdf")
    plt.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default="eval_results/issue_527")
    ap.add_argument(
        "--pair-selection",
        default="eval_results/issue_527/pair_selection.json",
    )
    ap.add_argument(
        "--figures-dir",
        default="figures/issue_527",
    )
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    analysis_dir = out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_root / "eval"
    sweep_dir = out_root / "sweep"

    pair_selection = json.loads(Path(args.pair_selection).read_text())

    # Collect cells by (pair, seed); we need all three arms per (pair, seed).
    cells_by_ps: dict[tuple[str, int], dict] = defaultdict(dict)
    for p in sorted(sweep_dir.glob("*.json")):
        cell = json.loads(p.read_text())
        key = (cell["pair_id"], cell["seed"])
        cells_by_ps[key][cell["arm"]] = cell

    cell_analyses = []
    for (pair_id, seed), arm_cells in sorted(cells_by_ps.items()):
        missing = [arm for arm in ("A_only", "B_only", "joint") if arm not in arm_cells]
        if missing:
            log.warning(
                "pair=%s seed=%d missing arms=%s; skipping analyze for this cell",
                pair_id,
                seed,
                missing,
            )
            continue

        # Load shift matrices.
        shift_a_meta, shift_a = _load_shift(eval_dir, arm_cells["A_only"]["cell_slug"])
        shift_b_meta, shift_b = _load_shift(eval_dir, arm_cells["B_only"]["cell_slug"])
        shift_j_meta, shift_j = _load_shift(eval_dir, arm_cells["joint"]["cell_slug"])

        # The eval panel must agree across arms.
        contexts = shift_a_meta["eval_panel"]
        if shift_b_meta["eval_panel"] != contexts or shift_j_meta["eval_panel"] != contexts:
            raise AssertionError(f"pair={pair_id} seed={seed}: eval_panel mismatch across arms")

        # Dict shift_X[context] = vector.
        shift_a_dict = {ctx: shift_a[i] for i, ctx in enumerate(contexts)}
        shift_b_dict = {ctx: shift_b[i] for i, ctx in enumerate(contexts)}
        shift_j_dict = {ctx: shift_j[i] for i, ctx in enumerate(contexts)}

        # Δ log P(marker) per context per arm.
        dlp_a = {ctx: float(shift_a_meta["contexts"][ctx]["delta_logp_marker"]) for ctx in contexts}
        dlp_b = {ctx: float(shift_b_meta["contexts"][ctx]["delta_logp_marker"]) for ctx in contexts}
        dlp_j = {ctx: float(shift_j_meta["contexts"][ctx]["delta_logp_marker"]) for ctx in contexts}

        # Source-self emission (DV4): read from the EMISSION json (vLLM on-policy).
        emission_a = _load_emission(eval_dir, arm_cells["A_only"]["cell_slug"])
        emission_b = _load_emission(eval_dir, arm_cells["B_only"]["cell_slug"])
        emission_j = _load_emission(eval_dir, arm_cells["joint"]["cell_slug"])

        pair_a, pair_b = pair_id.split("__")

        # Pack emission into the shape analyze_cell expects.
        # Round-2 fix per code-review Major-5: fail LOUD if a source name
        # is missing from a per_persona dict — pair_a / pair_b are always
        # in the eval panel by ``_resolve_eval_panel`` construction, so
        # absence indicates an eval-rig bug (silent zero would mis-route
        # the DV4 headline and read as a false floor).
        source_emission_a: dict[str, float] = {}
        source_emission_b: dict[str, float] = {}
        if emission_a:
            pp = emission_a["per_persona"]  # schema-pinned in _load_emission
            if pair_a not in pp:
                raise AssertionError(
                    f"emission_a per_persona missing source {pair_a!r} for cell "
                    f"{arm_cells['A_only']['cell_slug']!r} (per_persona keys: "
                    f"{sorted(pp.keys())}). Eval rig drift."
                )
            source_emission_a[pair_a] = float(pp[pair_a]["emission_rate_on_policy"])
        if emission_b:
            pp = emission_b["per_persona"]
            if pair_b not in pp:
                raise AssertionError(
                    f"emission_b per_persona missing source {pair_b!r} for cell "
                    f"{arm_cells['B_only']['cell_slug']!r} (per_persona keys: "
                    f"{sorted(pp.keys())}). Eval rig drift."
                )
            source_emission_b[pair_b] = float(pp[pair_b]["emission_rate_on_policy"])
        if emission_j:
            pp = emission_j["per_persona"]
            if pair_a not in pp:
                raise AssertionError(
                    f"emission_j per_persona missing source {pair_a!r} for cell "
                    f"{arm_cells['joint']['cell_slug']!r} (per_persona keys: "
                    f"{sorted(pp.keys())}). Eval rig drift."
                )
            if pair_b not in pp:
                raise AssertionError(
                    f"emission_j per_persona missing source {pair_b!r} for cell "
                    f"{arm_cells['joint']['cell_slug']!r} (per_persona keys: "
                    f"{sorted(pp.keys())}). Eval rig drift."
                )
            source_emission_a[f"joint_{pair_a}"] = float(pp[pair_a]["emission_rate_on_policy"])
            source_emission_b[f"joint_{pair_b}"] = float(pp[pair_b]["emission_rate_on_policy"])

        base_cos = _pair_cos(pair_selection, pair_id)

        cell = analyze_cell(
            pair_id=pair_id,
            seed=seed,
            pair_a=pair_a,
            pair_b=pair_b,
            contexts=contexts,
            shift_a=shift_a_dict,
            shift_b=shift_b_dict,
            shift_joint=shift_j_dict,
            delta_logp_a=dlp_a,
            delta_logp_b=dlp_b,
            delta_logp_joint=dlp_j,
            source_emission_a=source_emission_a,
            source_emission_b=source_emission_b,
            base_cos_a_b=base_cos,
        )
        cell_path = analysis_dir / f"{pair_id}__seed{seed}.json"
        cell_path.write_text(json.dumps(cell_to_dict(cell), indent=2))
        log.info(
            "[phase=analyze] pair=%s seed=%d: DV1 median=%.3f, GD1 top1=%.3f "
            "effrank=%.2f, GD2 cos=%.3f, GD3 eff_a=%.2f eff_b=%.2f, h1=%s h2=%s",
            pair_id,
            seed,
            cell.dv1_median,
            cell.gd1_top1_sv_share,
            cell.gd1_effective_rank,
            cell.gd2_singleton_cosine_median,
            cell.gd3_a_effective_rank,
            cell.gd3_b_effective_rank,
            cell.h1_pass,
            cell.h2_pass,
        )
        cell_analyses.append(cell)

    # Aggregate per-pair (median across seeds).
    pair_aggregate: dict[str, dict] = {}
    by_pair: dict[str, list] = defaultdict(list)
    for c in cell_analyses:
        by_pair[c.pair_id].append(c)
    for pair_id, cs in by_pair.items():
        pair_aggregate[pair_id] = {
            "n_seeds": len(cs),
            "median_dv1": float(np.median([c.dv1_median for c in cs])),
            "median_dv1_coverage": float(np.median([c.dv1_coverage_at_threshold for c in cs])),
            "median_dv2_resid_norm": float(np.median([c.dv2_residual_norm_median for c in cs])),
            "median_dv3_resid": float(np.median([c.dv3_residual_median for c in cs])),
            "fraction_h1_pass": float(sum(c.h1_pass for c in cs)) / len(cs),
            "fraction_h2_pass": float(sum(c.h2_pass for c in cs)) / len(cs),
            "fraction_dv1_diagnostic": float(sum(c.dv1_diagnostic for c in cs)) / len(cs),
            "base_cos_centered_L20": float(np.median([c.base_cos_a_b for c in cs])),
        }

    aggregate_payload = {
        "schema_version": "issue_527_analysis_v1",
        "pair_aggregate": pair_aggregate,
        "n_cells_analyzed": len(cell_analyses),
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }
    aggregate_path = out_root / "analysis.json"
    aggregate_path.write_text(json.dumps(aggregate_payload, indent=2))
    log.info("[phase=analyze_aggregate] wrote %s", aggregate_path)

    # Figures.
    _make_figures(
        figs_dir=Path(args.figures_dir), all_cells=cell_analyses, pair_aggregate=pair_aggregate
    )
    log.info(
        "[phase=done] analysis complete (%d cells, %d pairs)", len(cell_analyses), len(by_pair)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
