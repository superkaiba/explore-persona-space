#!/usr/bin/env python3
"""CPU tiny-real harness for the #1489 P6 fit driver (pre-pod verification).

Writes capture cells through the PRODUCTION shard writer
(`issue1489_gpu_phase.write_capture_shard` — the exact parent summaries
schema) with synthetic-VALUE tensors at REAL dims (28 layers x 3584), then
runs `issue1489_fit_grid` units against them with the REAL #1092 engine
functions. This is the maximal CPU-safe slice for P6 (the tensor values are
synthetic; every seam — writer schema, `_load_summary`, folds, PRESS fits,
nulls, twins, q4 disjoint-baseline legs, q6 stability — is the production
code). Synthetic values are STRUCTURED (v = M c + noise) so fits have signal.

Usage: uv run python scripts/issue1489_smoke_fit_harness.py --root /tmp/i1489_fit_smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from issue1489_gpu_phase import SUMMARY_KINDS_1489, write_capture_shard  # noqa: E402

N_LAYERS, HIDDEN = 28, 3584
TOPICS = ["coding_software", "science_medicine", "personal_advice", "other_t"]


def _rows(cell: str, n: int, split_cycle: list[str], relevant_topic: str | None) -> list[dict]:
    rows = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        rows.append(
            {
                "row_id": f"b{i:03d}-{cell}",
                "base_row_id": f"b{i:03d}",
                "prefix_id": f"p{i // 2:03d}",
                "query_id": f"q{i:03d}",
                "topic": topic,
                "split": split_cycle[i % len(split_cycle)],
                "cell_id": cell,
                "relevant": (topic == relevant_topic) if relevant_topic else None,
            }
        )
    return rows


def _summaries(rng, base_c: np.ndarray, m: np.ndarray, shift: np.ndarray) -> list[dict]:
    out = []
    for i in range(base_c.shape[0]):
        c = base_c[i] + shift
        v = c @ m + 0.05 * rng.standard_normal(HIDDEN)
        noise = 0.02 * rng.standard_normal((2, HIDDEN))
        rec = {}
        for kind in SUMMARY_KINDS_1489:
            if kind in ("prefix_end", "prefix_mean"):
                vec = c * 0.9
            elif kind in ("context_end", "context_mean"):
                vec = c
            elif kind == "t1":
                vec = v
            elif kind == "t1_odd":
                vec = v + noise[0]
            else:  # t1_even
                vec = v + noise[1]
            rec[kind] = np.tile(vec[None, :], (N_LAYERS, 1)).astype(np.float16)
        out.append(rec)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/tmp/i1489_fit_smoke")
    p.add_argument("--n-rows", type=int, default=48)
    args = p.parse_args()
    root = Path(args.root)
    summaries_dir = root / "summaries"
    rng = np.random.default_rng(0)
    m = rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float64) * (1.0 / np.sqrt(HIDDEN))
    base_c = rng.standard_normal((args.n_rows, HIDDEN))

    cells = {
        "cell_plain": (np.zeros(HIDDEN), None),
        "cell_fact_veg": (0.5 * rng.standard_normal(HIDDEN), "science_medicine"),
        "cell_fact_python": (0.5 * rng.standard_normal(HIDDEN), "coding_software"),
        "cell_instr_hedge": (0.8 * rng.standard_normal(HIDDEN), None),
        "cell_persona_pirate": (0.8 * rng.standard_normal(HIDDEN), None),
        "cell_format_json": (1.2 * rng.standard_normal(HIDDEN), None),
        "cell_ft_fact_veg": (0.5 * rng.standard_normal(HIDDEN), "science_medicine"),
        "cell_ft_fact_python": (0.5 * rng.standard_normal(HIDDEN), "coding_software"),
        "cell_plain_recap_b": (np.zeros(HIDDEN), None),
    }
    split_cycle = ["eval", "eval", "train", "probe"]
    for cell, (shift, rel_topic) in cells.items():
        rows = _rows(cell, args.n_rows, split_cycle, rel_topic)
        summ = _summaries(rng, base_c, m, shift)
        flags = [
            {"n_total": 100, "n_prompt": 60, "n_answer_tokens": 40, "t1_halves_degenerate": False}
            for _ in rows
        ]
        write_capture_shard(root, cell, 0, rows, summ, flags)
        print(f"[harness] wrote {cell}: {args.n_rows} rows")

    units = ",".join(
        [
            "transfer:context_end:pca48:L14",
            "transfer:context_end:ambient:L14",
            "shifts:context_end:L14",
            "gating:context_end:L14",
            "q4:L14",
            "q6:context_end:L14",
            "loto:context_end:L14",
        ]
    )
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue1489_fit_grid.py",
        "--smoke",
        "--summaries-dir",
        str(summaries_dir),
        "--out",
        str(root / "p6"),
        "--units",
        units,
    ]
    print("[harness] running:", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode
    print(f"[harness] fit driver rc={rc}")
    if rc != 0:
        return rc
    unit_files = sorted((root / "p6" / "units").glob("*.json"))
    print(f"[harness] {len(unit_files)} unit JSONs: {[f.name for f in unit_files]}")
    assert len(unit_files) == len(units.split(",")), "not all units produced output"
    print("FIT-HARNESS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
