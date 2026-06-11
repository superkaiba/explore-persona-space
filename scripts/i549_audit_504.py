"""#504 Phase-1 trajectory stale-serve audit (task #549) — the AFFECTED finding.

Evidence chain (each step persisted to the output JSON):

1. CODE: the realized Phase-1 eval (`epm:run-launched` 2026-06-08T22:13Z, commit
   affdd82cb) ran `eval_trajectory.run_trajectory_eval` with ONE vLLM engine
   (`llm = LLM(...)` line 321, `max_loras=1`) across a 6-checkpoint loop using
   constant ``lora_int_id=1`` (line 338) — fix 298877f9c is NOT an ancestor.
2. REPLAY: [(1, frac_0.08), (1, frac_0.16), ..., (1, frac_1.00)] at capacity 1
   (max_cpu_loras = max_loras = 1) through ``i549_lru_simulate.simulate`` =>
   5 of 6 requests stale-served by the frac-0.08 adapter, per cell.
3. ARTIFACTS: per-cell trajectory.json on HF
   (``issue504_geometry/phase1_trajectories/<cell>_seed<S>/``) all carry 6
   checkpoints @ affdd82cb. The per-(persona, q) base-side logp pairwise
   exact-equal-rate matrix across fractions is FLAT (~0.21-0.29 at EVERY
   fraction distance, no decay from adjacent to distant pairs) — the signature
   of ONE set of weights greedily re-generated 6 times under vLLM batching
   nondeterminism, not of 6 genuinely-different checkpoints spanning training
   steps 6..75 at lr 1e-4.
4. CONSUMPTION: ``eval_results/issue_504/analyze_summary.json``
   per_cell_diagnostics ``source_delta_g_nats`` values EXACTLY equal the
   trajectory frac-0.33 ``source_self.delta_g_mean`` entries (float-identical),
   so the published #504 numbers were read off the stale-served entries.

Consequence: every #504 Phase-1 read labeled frac > 0.08 was actually produced
by the frac-0.08 (step-6) adapter. The #534-era exposure note "single-entry
indices (#504 headline) are unaffected by construction" is falsified — the
realized #504 headline sessions had 6 entries each.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from i549_lru_simulate import simulate, summarize

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_results/issue_549/audit_evidence/i504_phase1_stale_serve.json"
CELLS = [
    f"{c}_seed{s}"
    for c in ("near", "mid_near", "mid_far", "far", "default_only")
    for s in (42, 137)
]
FRACS = (0.08, 0.16, 0.33, 0.5, 0.75, 1.0)


def main() -> None:
    # (2) LRU replay of one cell-session's realized request sequence.
    seq = [(1, f"checkpoints/frac_{f}") for f in FRACS]
    replay = summarize(simulate(seq, capacity=1))
    assert replay["n_stale"] == 5, replay
    assert all(e["served"] == "checkpoints/frac_0.08" for e in simulate(seq, capacity=1)), (
        "all requests must be served by the first-loaded frac-0.08 adapter"
    )

    # (3) Artifact census + flat pairwise equal-rate matrix per cell.
    per_cell: dict[str, dict] = {}
    frac033_delta_g: dict[str, float] = {}
    for cell in CELLS:
        p = hf_hub_download(
            "superkaiba1/explore-persona-space-data",
            f"issue504_geometry/phase1_trajectories/{cell}/trajectory.json",
            repo_type="dataset",
        )
        d = json.loads(Path(p).read_text())
        cks = d["checkpoints"]
        assert [c["frac"] for c in cks] == list(FRACS), cell
        assert d["git_commit"].startswith("affdd82cb"), (cell, d["git_commit"])
        vecs = {}
        for ck in cks:
            v = [
                ck["held_out"][persona][q]["b_logp"]
                for persona in sorted(ck["held_out"])
                for q in sorted(ck["held_out"][persona])
            ]
            vecs[ck["frac"]] = np.array(v)
            if ck["frac"] == 0.33:
                frac033_delta_g[cell] = ck["source_self"]["delta_g_mean"]
        matrix = {
            f"{a}_vs_{b}": float(np.mean(np.abs(vecs[a] - vecs[b]) < 1e-6))
            for a, b in itertools.combinations(FRACS, 2)
        }
        adjacent = [matrix[f"{a}_vs_{b}"] for a, b in itertools.pairwise(FRACS)]
        distant = [matrix[f"{FRACS[0]}_vs_{FRACS[-1]}"], matrix[f"{FRACS[0]}_vs_{FRACS[-2]}"]]
        per_cell[cell] = {
            "n_checkpoints": len(cks),
            "git_commit": d["git_commit"],
            "timestamp_utc": d["timestamp_utc"],
            "pairwise_equal_rate": matrix,
            "mean_adjacent_equal_rate": float(np.mean(adjacent)),
            "mean_distant_equal_rate": float(np.mean(distant)),
        }

    # (4) Consumption: analyze_summary per-cell source_delta_g == trajectory frac-0.33.
    summary = json.loads((REPO / "eval_results/issue_504/analyze_summary.json").read_text())
    consumption = []
    for row in summary["per_cell_diagnostics"]:
        cell = f"{row['cell'].removeprefix('c504v3_')}_seed{row['seed']}"
        traj_val = frac033_delta_g.get(cell)
        consumption.append(
            {
                "cell": cell,
                "analyze_summary_source_delta_g": row["source_delta_g_nats"],
                "trajectory_frac033_delta_g": traj_val,
                "float_identical": traj_val == row["source_delta_g_nats"],
            }
        )
    n_match = sum(c["float_identical"] for c in consumption)
    print(f"replay: {replay['n_stale']}/6 stale (served=frac_0.08)")
    flat = {
        c: (per_cell[c]["mean_adjacent_equal_rate"], per_cell[c]["mean_distant_equal_rate"])
        for c in CELLS
    }
    print("adjacent vs distant equal rates (flat => stale):")
    for c, (a, dd) in flat.items():
        print(f"  {c}: adjacent={a:.3f} distant={dd:.3f}")
    print(
        f"consumption: {n_match}/{len(consumption)} analyze_summary rows float-identical "
        f"to the stale-served frac-0.33 trajectory entries"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "metadata": {
                    "tool": "i549_audit_504",
                    "git_commit": subprocess.run(
                        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
                    ).stdout.strip(),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "run_commit": "affdd82cb (epm:run-launched 2026-06-08T22:13:21Z)",
                    "code_facts": [
                        "eval_trajectory.py@affdd82cb:321 one LLM(max_loras=1) per session",
                        "eval_trajectory.py@affdd82cb:338 constant lora_int_id=1 in ckpt loop",
                        "fix 298877f9c NOT ancestor of affdd82cb",
                    ],
                },
                "lru_replay": replay,
                "per_cell_artifacts": per_cell,
                "consumption_check": consumption,
                "verdict": "AFFECTED",
                "affected_reads": (
                    "every trajectory entry labeled frac>0.08 in all 10 cells; the published "
                    "analyze_summary per-cell values (chosen frac 0.33) consumed them"
                ),
            },
            indent=2,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
