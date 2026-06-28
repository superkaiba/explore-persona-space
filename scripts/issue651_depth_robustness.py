"""Issue #651 depth-robustness: re-run Q1 / Q2 / seed-ceiling at layers 7, 14, 21.

0 GPU, CPU only. Reuses the EXACT ``experiments.issue_651.analysis`` functions;
the ONLY thing that changes across layers is the per-persona read KEY:

  layer 14 (headline)  ->  delta_v          / delta_v_mean_resp
  layer 7              ->  delta_v_l7       / delta_v_mean_resp_l7
  layer 21             ->  delta_v_l21      / delta_v_mean_resp_l21

(mean-resp is the primary read for the generative behaviors em/sycophancy/emnc,
slot for the rest — same _MEAN_RESP_PRIMARY split as issue651_analysis.py). The
132 persisted per-cell shift tensors already carry all three layers, so this is a
pure re-analysis off existing HF data — no extraction, no GPU.

Layer 14 is recomputed too, as a self-check that this driver reproduces the
published eval_results/issue_651 numbers before trusting 7/21.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("issue651_depth")

_MEAN_RESP_PRIMARY = frozenset({"em", "sycophancy", "emnc"})
_HF_REPO = "superkaiba1/explore-persona-space-data"
_HF_REV = "4ab90f83239e51bb6ba446edda202b8e7c5e6469"
_HF_SUBDIR = "issue651_cross_behavior_geometry/analysis_tensors"
_LAYERS = (14, 7, 21)


def _read_key(behavior: str, layer: int) -> str:
    base = "delta_v_mean_resp" if behavior in _MEAN_RESP_PRIMARY else "delta_v"
    return base if layer == 14 else f"{base}_l{layer}"


def _repo_root() -> Path:
    import subprocess

    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())


def _download_cells():
    """snapshot_download the 132 per-cell .pt and return {cell_id: (cell, shifts)}."""
    import torch
    from huggingface_hub import snapshot_download

    from explore_persona_space.experiments.issue_651 import parse_cell_spec

    local = snapshot_download(
        _HF_REPO,
        repo_type="dataset",
        revision=_HF_REV,
        allow_patterns=[f"{_HF_SUBDIR}/*.pt"],
    )
    shift_dir = Path(local) / _HF_SUBDIR
    cells = {}
    for pt in sorted(shift_dir.glob("*.pt")):
        try:
            cell = parse_cell_spec(pt.stem)
        except ValueError:
            logger.warning("skip unparseable %s", pt.name)
            continue
        cells[pt.stem] = (cell, torch.load(pt, map_location="cpu", weights_only=False)["shifts"])
    return cells


def _analyze_layer(cells, layer: int, n_reps: int = 1000) -> dict:
    """Faithful copy of issue651_analysis.main()'s pipeline at one read layer."""
    from explore_persona_space.experiments.issue_651 import NULL_CHECK_BEHAVIOR
    from explore_persona_space.experiments.issue_651 import analysis as i651

    # by_bs[behavior][seed][cid] = read vector at this layer
    by_bs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for _cid, (cell, shifts) in cells.items():
        key = _read_key(cell.behavior, layer)
        read = i651.cell_read_vector(shifts, key=key, cell_read="u1")
        by_bs.setdefault(cell.behavior, {}).setdefault(cell.seed, {})[cell.cid] = read

    # seed ceiling (Q1 per-cell object)
    seed_ceiling_median: dict[str, float] = {}
    for behavior, by_seed in by_bs.items():
        if 42 in by_seed and 1042 in by_seed:
            sc = i651.seed_ceiling_per_cell(by_seed[42], by_seed[1042])
            seed_ceiling_median[behavior] = sc["median"]

    # Q1 + per-behavior U1 (headline seed-42) + per-seed U1 (Q2 ceiling object)
    behavior_u1: dict[str, np.ndarray] = {}
    u1_by_seed: dict[str, dict[int, np.ndarray]] = {}
    q1_summary: dict[str, dict] = {}
    for behavior, by_seed in by_bs.items():
        for seed_val, per_ctx in by_seed.items():
            if len(per_ctx) < 2:
                continue
            u1_by_seed.setdefault(behavior, {})[seed_val] = np.asarray(
                i651.q1_context_invariance(per_ctx, n_reps=n_reps)["U1"], dtype=np.float32
            )
        per_context = by_seed.get(42) or next(iter(by_seed.values()))
        if len(per_context) < 2:
            continue
        q1 = i651.q1_context_invariance(per_context, n_reps=n_reps)
        ceil = seed_ceiling_median.get(behavior)
        verdict = (
            i651.q1_verdict(q1, ceil)
            if ceil is not None
            else {"verdict": "no_ceiling", "context_invariant": None}
        )
        q1_summary[behavior] = {
            "n_contexts": q1["n_contexts"],
            "top_share_norm_weighted": q1["top_share_norm_weighted"],
            "sign_flip_null_p95": q1["sign_flip_null_p95"],
            "top_share_clears_null_p95": q1["top_share_clears_null_p95"],
            "mean_cos_to_U1": q1["mean_cos_to_U1"],
            "seed_ceiling_median": ceil,
            "per_context_bar": verdict.get("per_context_bar"),
            "frac_contexts_at_or_above_bar": verdict.get("frac_contexts_at_or_above_bar"),
            "n_contexts_at_or_above_bar": verdict.get("n_contexts_at_or_above_bar"),
            "verdict": verdict.get("verdict"),
        }
        if 42 in by_seed and behavior in u1_by_seed and 42 in u1_by_seed[behavior]:
            behavior_u1[behavior] = u1_by_seed[behavior][42]
        else:
            behavior_u1[behavior] = np.asarray(q1["U1"], dtype=np.float32)

    # Q2 ceiling (per-behavior cross-seed U1 cosine) + cross-behavior matrix
    q2_seed_ceiling = i651.q2_seed_ceiling_per_behavior(u1_by_seed)
    headline = {b: u1 for b, u1 in behavior_u1.items() if b not in (NULL_CHECK_BEHAVIOR, "emnc")}
    q2_out = None
    if len(headline) >= 2:
        q2 = i651.q2_cross_behavior_matrix(headline, q2_seed_ceiling, n_reps=n_reps)
        q2v = i651.q2_verdict(q2)
        q2_out = {
            "behaviors": q2["behaviors"],
            "raw_cosine_matrix": q2["raw_cosine_matrix"],
            "ceiling_normalized_matrix": q2["ceiling_normalized_matrix"],
            "seed_ceilings": q2["seed_ceilings"],
            "cross_behavior_null_p95": q2["cross_behavior_null_p95"],
            "off_diagonals": q2v["off_diagonal_ceiling_fractions"],
            "verdict": q2v["verdict"],
        }

    return {
        "layer": layer,
        "seed_ceiling_median": seed_ceiling_median,
        "q2_seed_ceiling": q2_seed_ceiling,
        "q1": q1_summary,
        "q2": q2_out,
    }


def _fmt_q1(res: dict) -> str:
    lines = [f"  Q1 (layer {res['layer']}): per-behavior context-invariance"]
    hdr = f"    {'behavior':<12}{'topshare':>9}{'null95':>8}{'ceil':>7}{'bar':>7}{'frac@bar':>9}  verdict"
    lines.append(hdr)
    for b in sorted(res["q1"]):
        q = res["q1"][b]
        ceil = q["seed_ceiling_median"]
        bar = q["per_context_bar"]
        frac = q["frac_contexts_at_or_above_bar"]
        lines.append(
            f"    {b:<12}{q['top_share_norm_weighted']:>9.3f}{q['sign_flip_null_p95']:>8.3f}"
            f"{(ceil if ceil is not None else float('nan')):>7.3f}"
            f"{(bar if bar is not None else float('nan')):>7.3f}"
            f"{(frac if frac is not None else float('nan')):>9.2f}  {q['verdict']}"
        )
    return "\n".join(lines)


def _fmt_q2(res: dict) -> str:
    q2 = res["q2"]
    if not q2:
        return f"  Q2 (layer {res['layer']}): <2 headline behaviors"
    lines = [
        f"  Q2 (layer {res['layer']}): cross-behavior off-diagonals (ceiling-fraction | raw cos)"
    ]
    for od in q2["off_diagonals"]:
        a, b = od["pair"]
        lines.append(
            f"    {a:<14} x {b:<14} {od['ceiling_fraction']:>6.2f}  (raw {od['raw_cosine']:>5.2f}, "
            f"null95 {od['null_p95']:>4.2f})"
        )
    lines.append(f"    verdict: {q2['verdict']}")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    logger.info("[downloading 132 per-cell tensors from HF ...]")
    cells = _download_cells()
    logger.info("[loaded %d cells]", len(cells))

    all_res = {}
    for layer in _LAYERS:
        logger.info("[analyzing layer %d ...]", layer)
        all_res[layer] = _analyze_layer(cells, layer)

    out_dir = _repo_root() / "eval_results" / "issue_651" / "depth_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(all_res, indent=2, default=float))

    print("\n" + "=" * 88)
    print("ISSUE #651 DEPTH ROBUSTNESS — Q1 / Q2 at layers 14 (headline), 7, 21")
    print("=" * 88)
    for layer in _LAYERS:
        print(f"\n----- LAYER {layer} " + "-" * 70)
        print(_fmt_q1(all_res[layer]))
        print(_fmt_q2(all_res[layer]))

    print("\n" + "=" * 88)
    print("CROSS-LAYER SUMMARY")
    print("=" * 88)
    print("\nQ1 verdicts (frac of contexts at/above bar):")
    behs = sorted(all_res[14]["q1"])
    print(f"  {'behavior':<12}" + "".join(f"{'L' + str(L):>14}" for L in _LAYERS))
    for b in behs:
        row = f"  {b:<12}"
        for L in _LAYERS:
            q = all_res[L]["q1"].get(b, {})
            frac = q.get("frac_contexts_at_or_above_bar")
            vd = (q.get("verdict") or "?")[:11]
            row += f"{(f'{frac:.2f} ' + vd) if frac is not None else 'NA':>14}"
        print(row)

    print("\nQ2 off-diagonal ceiling-fractions:")
    pairs = [tuple(od["pair"]) for od in (all_res[14]["q2"] or {}).get("off_diagonals", [])]
    print(f"  {'pair':<30}" + "".join(f"{'L' + str(L):>9}" for L in _LAYERS))
    for pr in pairs:
        row = f"  {pr[0] + ' x ' + pr[1]:<30}"
        for L in _LAYERS:
            ods = {
                tuple(od["pair"]): od for od in (all_res[L]["q2"] or {}).get("off_diagonals", [])
            }
            od = ods.get(pr)
            row += f"{od['ceiling_fraction']:>9.2f}" if od else f"{'NA':>9}"
        row += "   verdict: " + ", ".join(f"L{L}={all_res[L]['q2']['verdict']}" for L in _LAYERS)
        print(row) if pr == pairs[0] else print(row.rsplit("   verdict", 1)[0])
    print(f"\nwrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
