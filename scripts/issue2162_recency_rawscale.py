"""Raw-scale (un-normalized) recency check for issue #2162.

The headline recency read is in F_beh space, where F = (patched - floor) /
(ceiling - floor).  The anchor gap in that denominator shrinks ~3x from
conversational depth 1 to depth 5 (mean |separation| 1.56 -> 0.57), so an F
drop across depths is confounded with a resolution loss: both the steered arm
and the null arms are divided by a smaller, noisier number.

This script recomputes the same comparison WITHOUT the denominator.  Per pair
it reports the signed raw movement

    (delta_patched - delta_floor) * sign(delta_ceiling - delta_floor)

in judge-contrast units (the dual-rubric delta, range [-2, +2]), for the
steered arm and both null arms, at the context-end slot.  A steered collapse
that survives here is a real decay of the patch's causal reach; a collapse
that appears only in F space is a denominator artifact.

Reported over BOTH pair sets: `all` (every pair, no anchor-separation
exclusion -- the exclusion selects on the very denominator under scrutiny) and
`surviving` (the pre-registered |separation| >= 0.5 subset, for continuity
with the headline tables).

Read-only over committed artifacts under ``eval_results/issue_2162/f_metrics/``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# NOTE: numpy and explore_persona_space are deliberately NOT imported at module
# top. This module loads .env inside main() rather than at module scope, so a
# module-top heavy import would run before the shared-VM thread caps (#847)
# bind. Both are imported inside the functions that need them.

SLOT = "ce"
SEPARATION_BAR = 0.5
BOOT_B = 10_000
BOOT_SEED = 21620

ARM_FILES = {
    "steered": "f_cells.jsonl",
    "shuffled": "null_shuffled_cells.jsonl",
    "crosstype": "null_crosstype_cells.jsonl",
}

DEFAULT_CELLS = (
    "instr_format",
    "recency_instr_format_d3",
    "recency_instr_format_d5",
    "persona_prompted",
    "recency_persona_prompted_d3",
    "recency_persona_prompted_d5",
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_tables(metrics_dir: Path) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    """Return (anchors by pair_id, per-(pair, arm) patched delta at SLOT)."""
    anchors = {r["pair_id"]: r for r in _read_jsonl(metrics_dir / "anchors.jsonl")}
    patched: dict[tuple[str, str], dict] = defaultdict(dict)
    for arm, filename in ARM_FILES.items():
        for row in _read_jsonl(metrics_dir / filename):
            if row["slot"] != SLOT or row["delta_patched_mean"] is None:
                continue
            patched[(row["pair_id"], arm)] = row
    return anchors, patched


def signed_movement(anchor: dict, delta_patched: float) -> float:
    """Raw movement toward the ceiling, in judge-contrast units.

    A zero anchor gap takes the +1 branch, matching ``np.sign(0) -> 0`` coerced
    to 1.0; such pairs are excluded by the separation bar in any case.
    """
    direction = 1.0 if anchor["delta_ceiling_mean"] >= anchor["delta_floor_mean"] else -1.0
    return float(direction * (delta_patched - anchor["delta_floor_mean"]))


def bootstrap_ci(values: list[float], rng) -> tuple[float, float]:
    """Pair-clustered percentile CI (one value per pair, so a plain resample)."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan"), float("nan")
    draws = arr[rng.integers(0, arr.size, size=(BOOT_B, arr.size))].mean(axis=1)
    draws.sort()
    return float(draws[int(0.025 * BOOT_B)]), float(draws[int(0.975 * BOOT_B)])


def summarize(cells: tuple[str, ...], metrics_dir: Path) -> list[dict]:
    import numpy as np

    anchors, patched = load_tables(metrics_dir)
    rng = np.random.default_rng(BOOT_SEED)
    out: list[dict] = []
    for cell in cells:
        cell_pairs = [pid for pid, a in anchors.items() if a["cell"] == cell]
        if not cell_pairs:
            raise SystemExit(f"no anchor rows for cell {cell!r} in {metrics_dir}")
        subsets = {
            "all": cell_pairs,
            "surviving": [
                pid for pid in cell_pairs if abs(anchors[pid]["separation"]) >= SEPARATION_BAR
            ],
        }
        for subset_name, pair_ids in subsets.items():
            per_arm: dict[str, list[float]] = defaultdict(list)
            gaps: list[float] = []
            for pid in pair_ids:
                anchor = anchors[pid]
                if (pid, "steered") not in patched:
                    continue
                gaps.append(abs(anchor["separation"]))
                for arm in ARM_FILES:
                    row = patched.get((pid, arm))
                    if row is not None:
                        per_arm[arm].append(signed_movement(anchor, row["delta_patched_mean"]))
            if not per_arm["steered"]:
                continue
            steered_lo, steered_hi = bootstrap_ci(per_arm["steered"], rng)
            null_means = {arm: float(np.mean(per_arm[arm])) for arm in ("shuffled", "crosstype")}
            steered_mean = float(np.mean(per_arm["steered"]))
            out.append(
                {
                    "cell": cell,
                    "subset": subset_name,
                    "n_pairs": len(per_arm["steered"]),
                    "mean_anchor_gap": float(np.mean(gaps)),
                    "steered_mean": steered_mean,
                    "steered_ci95": [steered_lo, steered_hi],
                    "shuffled_mean": null_means["shuffled"],
                    "crosstype_mean": null_means["crosstype"],
                    "margin_over_max_null": steered_mean - max(null_means.values()),
                }
            )
    return out


def main() -> None:
    from explore_persona_space.orchestrate.env import load_dotenv
    from explore_persona_space.task_workflow import repo_root

    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=repo_root() / "eval_results" / "issue_2162" / "f_metrics",
    )
    parser.add_argument("--cells", nargs="*", default=list(DEFAULT_CELLS))
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    rows = summarize(tuple(args.cells), args.metrics_dir)

    header = (
        f"{'cell':32s} {'subset':9s} {'n':>3s} {'gap':>5s} | "
        f"{'steered (95% CI)':>21s} {'shuffled':>9s} {'xtype':>9s} {'margin':>8s}"
    )
    print(header)
    print("-" * len(header))
    last_cell = None
    for r in rows:
        if last_cell is not None and r["cell"] != last_cell:
            print()
        last_cell = r["cell"]
        lo, hi = r["steered_ci95"]
        print(
            f"{r['cell']:32s} {r['subset']:9s} {r['n_pairs']:>3d} "
            f"{r['mean_anchor_gap']:>5.2f} | "
            f"{r['steered_mean']:+.3f} [{lo:+.2f},{hi:+.2f}] "
            f"{r['shuffled_mean']:>+9.3f} {r['crosstype_mean']:>+9.3f} "
            f"{r['margin_over_max_null']:>+8.3f}"
        )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "slot": SLOT,
                    "separation_bar": SEPARATION_BAR,
                    "boot": {"B": BOOT_B, "seed": BOOT_SEED},
                    "units": "judge-contrast delta (dual-rubric, range [-2, +2])",
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
