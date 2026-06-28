"""Issue #651 read x layer robustness grid: 2 reads x 3 layers, all behaviors.

0 GPU, CPU only. For EACH (read in {slot, mean_resp}, layer in {7,14,21}) it
forces that SAME read uniformly across all behaviors (the headline analysis
instead used a per-behavior primary read: mean_resp for em/sycophancy, slot for
marker/fact, layer 14 only). Reuses the EXACT experiments.issue_651.analysis
functions; only the per-persona read KEY changes:

  slot,      L14  ->  delta_v
  slot,      L7   ->  delta_v_l7
  mean_resp, L21  ->  delta_v_mean_resp_l21      (etc.)

All 132 cells carry both reads at all three layers, so this is pure re-analysis.
The headline-primary cells (em/syc @ mean_resp, marker/fact @ slot, all L14) are
marked with * so robustness can be read against the published numbers.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("issue651_grid")

_HF_REPO = "superkaiba1/explore-persona-space-data"
_HF_REV = "4ab90f83239e51bb6ba446edda202b8e7c5e6469"
_HF_SUBDIR = "issue651_cross_behavior_geometry/analysis_tensors"
_READS = {"slot": "delta_v", "mean_resp": "delta_v_mean_resp"}
_LAYERS = (7, 14, 21)
_PRIMARY = {"em": "mean_resp", "sycophancy": "mean_resp", "marker": "slot", "fact": "slot"}


def _key(read_base: str, layer: int) -> str:
    return read_base if layer == 14 else f"{read_base}_l{layer}"


def _repo_root() -> Path:
    import subprocess

    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())


def _download_cells():
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
            continue
        cells[pt.stem] = (cell, torch.load(pt, map_location="cpu", weights_only=False)["shifts"])
    return cells


def _analyze(cells, read_base: str, layer: int, n_reps: int = 1000) -> dict:
    """Headline pipeline at one (read, layer), read forced uniform across behaviors."""
    from explore_persona_space.experiments.issue_651 import NULL_CHECK_BEHAVIOR
    from explore_persona_space.experiments.issue_651 import analysis as i651

    key = _key(read_base, layer)
    by_bs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for _cid, (cell, shifts) in cells.items():
        read = i651.cell_read_vector(shifts, key=key, cell_read="u1")
        by_bs.setdefault(cell.behavior, {}).setdefault(cell.seed, {})[cell.cid] = read

    seed_ceiling_median = {}
    for b, by_seed in by_bs.items():
        if 42 in by_seed and 1042 in by_seed:
            seed_ceiling_median[b] = i651.seed_ceiling_per_cell(by_seed[42], by_seed[1042])[
                "median"
            ]

    behavior_u1, u1_by_seed, q1_summary = {}, {}, {}
    for b, by_seed in by_bs.items():
        for sv, per_ctx in by_seed.items():
            if len(per_ctx) >= 2:
                u1_by_seed.setdefault(b, {})[sv] = np.asarray(
                    i651.q1_context_invariance(per_ctx, n_reps=n_reps)["U1"], dtype=np.float32
                )
        per_context = by_seed.get(42) or next(iter(by_seed.values()))
        if len(per_context) < 2:
            continue
        q1 = i651.q1_context_invariance(per_context, n_reps=n_reps)
        ceil = seed_ceiling_median.get(b)
        v = i651.q1_verdict(q1, ceil) if ceil is not None else {"verdict": "no_ceiling"}
        q1_summary[b] = {
            "top_share": q1["top_share_norm_weighted"],
            "frac_at_bar": v.get("frac_contexts_at_or_above_bar"),
            "verdict": v.get("verdict"),
            "seed_ceiling": ceil,
        }
        behavior_u1[b] = (
            u1_by_seed[b][42]
            if (42 in by_seed and b in u1_by_seed and 42 in u1_by_seed[b])
            else np.asarray(q1["U1"], dtype=np.float32)
        )

    q2_ceil = i651.q2_seed_ceiling_per_behavior(u1_by_seed)
    headline = {b: u1 for b, u1 in behavior_u1.items() if b not in (NULL_CHECK_BEHAVIOR, "emnc")}
    q2 = None
    if len(headline) >= 2:
        m = i651.q2_cross_behavior_matrix(headline, q2_ceil, n_reps=n_reps)
        mv = i651.q2_verdict(m)
        q2 = {
            "off": {
                tuple(od["pair"]): od["ceiling_fraction"]
                for od in mv["off_diagonal_ceiling_fractions"]
            },
            "raw": {
                tuple(od["pair"]): od["raw_cosine"] for od in mv["off_diagonal_ceiling_fractions"]
            },
            "verdict": mv["verdict"],
        }
    return {"q1": q1_summary, "q2": q2}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    logger.info("[downloading 132 tensors ...]")
    cells = _download_cells()
    logger.info("[loaded %d cells]", len(cells))

    grid = {}
    combos = list(itertools.product(_READS, _LAYERS))
    for read, layer in combos:
        logger.info("[analyzing read=%s layer=%d ...]", read, layer)
        grid[(read, layer)] = _analyze(cells, _READS[read], layer)

    out_dir = _repo_root() / "eval_results" / "issue_651" / "depth_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _ser(v):  # stringify the (a,b) pair-tuple keys for JSON
        if not v["q2"]:
            return v
        q2 = dict(v["q2"])
        q2["off"] = {f"{a}|{b}": x for (a, b), x in v["q2"]["off"].items()}
        q2["raw"] = {f"{a}|{b}": x for (a, b), x in v["q2"]["raw"].items()}
        return {"q1": v["q1"], "q2": q2}

    (out_dir / "read_layer_grid.json").write_text(
        json.dumps({f"{r}@{L}": _ser(v) for (r, L), v in grid.items()}, indent=2, default=float)
    )

    cols = [f"{r}@{L}" for r, L in combos]
    behs = ["em", "sycophancy", "marker", "fact"]

    print("\n" + "=" * 100)
    print("ISSUE #651 — Q1 CONTEXT-INVARIANCE across read x layer  (* = headline-primary cell)")
    print("frac of 16 contexts >= bar ; CI=context_invariant  CS=context_specific")
    print("=" * 100)
    print(f"{'behavior':<12}" + "".join(f"{c:>13}" for c in cols))
    for b in behs:
        row = f"{b:<12}"
        for r, L in combos:
            q = grid[(r, L)]["q1"].get(b, {})
            frac = q.get("frac_at_bar")
            vd = {"context_invariant": "CI", "context_specific": "CS"}.get(q.get("verdict"), "?")
            star = "*" if (_PRIMARY.get(b) == r and L == 14) else " "
            cell = f"{frac:.2f}{vd}{star}" if frac is not None else "NA"
            row += f"{cell:>13}"
        print(row)

    print("\n" + "=" * 100)
    print("ISSUE #651 — Q2 CROSS-BEHAVIOR off-diagonal (ceiling-fraction)")
    print("=" * 100)
    pairs = sorted({p for (_r, _L), v in grid.items() if v["q2"] for p in v["q2"]["off"]})
    print(f"{'pair':<26}" + "".join(f"{c:>13}" for c in cols))
    for pr in pairs:
        row = f"{pr[0] + ' x ' + pr[1]:<26}"
        for r, L in combos:
            q2 = grid[(r, L)]["q2"]
            val = q2["off"].get(pr) if q2 else None
            row += f"{val:>13.2f}" if val is not None else f"{'NA':>13}"
        print(row)
    print(
        f"\n{'Q2 verdict':<26}"
        + "".join(
            f"{(grid[(r, L)]['q2']['verdict'][:11] if grid[(r, L)]['q2'] else 'NA'):>13}"
            for r, L in combos
        )
    )
    print(f"\nwrote {out_dir / 'read_layer_grid.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
