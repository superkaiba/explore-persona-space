"""#532 cache-collision audit: per-process LRU replay + ep1/ep2 served-adapter test (#549).

Implements plan #549 §4 Step B.3/B.5 with the §12 hardenings for the prime suspect
`scripts/issue532_predictor_stress.py` (3 processes across commits 28d35f027 / 28a73584c):

1. Assign every per-cell JSON to its producing process via ``metadata.git_commit`` +
   ``metadata.timestamp_utc`` (P3 launched 2026-06-09T19:22:53Z with ``--phase 2+3+4``
   and constructs no LoRARequest; assert it wrote no cells).
2. Reconstruct each process's realized (lora_int_id, lora_path) request sequence from
   per-(epoch, source) first-touch times (the phase-1 loop is sequential, so first-touch
   order IS the ``enumerate(sources)`` order; resume-skipped (ep, src) groups with zero
   generated cells produce zero adapter events — verified at both run commits: the only
   ``llm.generate(..., lora_request=...)`` calls sit inside the bystander loop after the
   per-cell resume skip).
3. Replay each sequence through ``i549_lru_simulate.simulate`` at the realized registry
   capacity (max_cpu_loras defaults to max_loras = 1 at both commits,
   ``_build_vllm_engine``), plus a COUNTERFACTUAL capacity-16 replay showing the run
   would have been affected with a larger cache (i.e. safety is BY EVICTION).
4. Empirical served-adapter test (confirmatory, per process): greedy ``R_trained_per_q``
   text identity + full ``trained_logp_per_q`` diff distribution for matched
   (source, bystander, question) ep2-vs-ep1 pairs, ``zip(strict=True)``; dose-direction
   cross-check against #474's per-epoch ``G_logprob_matrix.json`` diagonals.

Question-order note (§12 item 2): both commits load q_test via the IDENTICAL
``i460_data.load_q_test_extended_50`` (empty `git diff 28d35f027..28a73584c` on
i460_data.py), which reads the pinned `q_test_extended_50.json` artifact from
HF_DATA_REPO at revision="main" and asserts len == 50. The per-cell JSONs do not store
question text, so pairing relies on this code-level order pin; the artifact revision is
not SHA-pinned (flagged as the one Medium-confidence link in the audit row).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from i549_lru_simulate import simulate, summarize

REPO = Path(__file__).resolve().parent.parent
PER_CELL = REPO / "eval_results/issue_532/per_cell"
I474_CROSS = REPO / "eval_results/issue_474/cross_eval"
OUT_DIR = REPO / "eval_results/issue_549/audit_evidence"

P1_COMMIT = "28d35f027"  # launched 2026-06-09T14:25:28Z (epochs 1 2 3, fresh)
P2_COMMIT = "28a73584c"  # launched 2026-06-09T17:52:39Z (resume, epochs 1 2 3)
P3_LAUNCH_TS = "2026-06-09T19:22:53"  # --phase 2+3+4: no LoRARequest constructed

# Pre-registered thresholds (plan §4 Step B.5; numeric "≈0" pinned at 0.1 nat —
# vLLM batching nondeterminism produces |Δlogp| well below this for same-adapter
# replays, while genuine adapter differences in this family are O(1-10) nats).
IDENTITY_AFFECTED_THRESHOLD = 0.9
MAX_DLOGP_NEARZERO_NAT = 0.1


def _load_cells() -> list[dict]:
    cells = []
    for ep in (1, 2):
        for p in sorted((PER_CELL / f"loc_ep{ep}").glob("*.json")):
            d = json.loads(p.read_text())
            stem = p.name[len(f"cell_loc_ep{ep}_") : -len(".json")]
            src, byst = stem.split("__", 1)
            assert d["source_cid"] == src and d["bystander_label"] == byst, p.name
            cells.append(
                {
                    "ep": ep,
                    "src": src,
                    "byst": byst,
                    "ts": d["metadata"]["timestamp_utc"],
                    "commit": d["metadata"]["git_commit"],
                    "R": d["R_trained_per_q"],
                    "logp": d["trained_logp_per_q"],
                }
            )
    return cells


def _process_of(cell: dict) -> str:
    if cell["commit"].startswith(P1_COMMIT):
        return "P1"
    if cell["commit"].startswith(P2_COMMIT):
        assert cell["ts"] < P3_LAUNCH_TS, (
            f"cell {cell['ep']}/{cell['src']}/{cell['byst']} written at {cell['ts']} "
            f"AFTER the P3 launch ({P3_LAUNCH_TS}) — P3 should write no cells"
        )
        return "P2"
    raise AssertionError(f"unknown commit {cell['commit']} — a 4th process?")


def _first_touch_order(cells: list[dict]) -> list[tuple[str, int, str, str]]:
    """(process, ep, src, first_ts) groups sorted by first-touch time."""
    groups: dict[tuple[str, int, str], str] = {}
    for c in cells:
        key = (_process_of(c), c["ep"], c["src"])
        groups[key] = min(groups.get(key, c["ts"]), c["ts"])
    return sorted([(*k, ts) for k, ts in groups.items()], key=lambda r: r[3])


def main() -> None:
    cells = _load_cells()
    n_ep1 = sum(c["ep"] == 1 for c in cells)
    n_ep2 = sum(c["ep"] == 2 for c in cells)
    assert (n_ep1, n_ep2) == (416, 140), (n_ep1, n_ep2)

    events = _first_touch_order(cells)

    # FINDING (artifact census, supersedes the plan-time 3-process assumption): every
    # surviving cell carries commit 28a73584c with timestamps 18:02-19:10Z — i.e. ALL
    # 556 cells were produced by P2. P1 (28d35f027, launched 14:25Z) left no surviving
    # per-cell artifacts (it died before any phase-1 cell write survived to P2's start;
    # P2's resume-skip wrote all 416 ep1 cells itself, which is only possible if none
    # existed on disk). P3 ran --phase 2+3+4 and constructs no LoRARequest.
    producing = sorted({proc for proc, _, _, _ in events})
    assert producing == ["P2"], f"unexpected producing processes: {producing}"

    # Canonical source order = the producing process's ep1 first-touch order
    # (sequential loop => enumerate order).
    ep1_order = [src for proc, ep, src, _ in events if proc == "P2" and ep == 1]
    assert len(ep1_order) == 16, ep1_order
    src_idx = {s: i for i, s in enumerate(ep1_order)}
    # Internal consistency: the ep2 first-touch order must follow the same relative
    # order (same CONDITIONS list, same sequential loop).
    ep2_order = [src for pr, ep, src, _ in events if pr == "P2" and ep == 2]
    idxs = [src_idx[s] for s in ep2_order]
    assert idxs == sorted(idxs), f"P2 ep2 order {ep2_order} violates ep1 order"

    # Per-process (lora_int_id, lora_path) sequences + LRU replays. P1 contributes
    # zero surviving-artifact-relevant adapter events (no surviving cells); P3 none.
    replays: dict[str, dict] = {
        "P1": {"note": "no surviving per-cell artifacts — zero artifact-relevant events"},
        "P3": {"note": "--phase 2+3+4 only; constructs no LoRARequest"},
    }
    for proc in ("P2",):
        seq = [
            (src_idx[src] + 1, f"adapters/i474_loc_{src}_ep{ep}")
            for pr, ep, src, _ in events
            if pr == proc
        ]
        real = summarize(simulate(seq, capacity=1))  # max_cpu_loras = max_loras = 1
        counterfactual = summarize(simulate(seq, capacity=16))
        replays[proc] = {
            "request_sequence": [{"lora_int_id": i, "lora_path": p} for i, p in seq],
            "replay_capacity_1": real,
            "replay_capacity_16_counterfactual": counterfactual,
        }
        print(
            f"{proc}: {len(seq)} adapter events | capacity=1 stale={real['n_stale']} "
            f"| counterfactual capacity=16 stale={counterfactual['n_stale']}"
        )

    # Empirical ep2-vs-ep1 comparison, per producing process of the ep2 cell. Pairs are
    # split diagonal (byst == src) vs off-diagonal-within-the-16-conditions vs other
    # (instructed bystanders outside #474's 16-condition panel) — the dose-direction
    # cross-check must compare like with like, since the loc (contrastive) arm pushes
    # the source diagonal UP and bystander leakage DOWN with more epochs.
    cond16 = set(src_idx)
    ep1_by_cell = {(c["src"], c["byst"]): c for c in cells if c["ep"] == 1}
    comparison: dict[str, dict] = {}
    pooled_rows = []
    subset_dlogps: dict[str, list[float]] = {"diagonal": [], "offdiag_16": [], "other": []}
    for proc in ("P2",):
        ep2_cells = [c for c in cells if c["ep"] == 2 and _process_of(c) == proc]
        pairs = ident = 0
        dlogps: list[float] = []
        per_cell_max: list[float] = []
        for c2 in ep2_cells:
            c1 = ep1_by_cell[(c2["src"], c2["byst"])]
            if c2["byst"] == c2["src"]:
                subset = "diagonal"
            elif c2["byst"] in cond16:
                subset = "offdiag_16"
            else:
                subset = "other"
            cell_d = []
            for r1, r2, l1, l2 in zip(c1["R"], c2["R"], c1["logp"], c2["logp"], strict=True):
                pairs += 1
                ident += int(r1 == r2)
                d = float(l2) - float(l1)
                dlogps.append(d)
                subset_dlogps[subset].append(d)
                cell_d.append(abs(d))
            per_cell_max.append(max(cell_d))
        arr = np.array(dlogps)
        pcm = np.array(per_cell_max)
        comparison[proc] = {
            "n_ep2_cells": len(ep2_cells),
            "n_question_pairs": pairs,
            "text_identity_rate": ident / max(pairs, 1),
            "dlogp_ep2_minus_ep1_percentiles": {
                f"p{p}": float(np.percentile(arr, p)) for p in (5, 25, 50, 75, 95)
            },
            "dlogp_mean": float(arr.mean()),
            "per_cell_max_abs_dlogp": {
                "min": float(pcm.min()),
                "p25": float(np.percentile(pcm, 25)),
                "median": float(np.median(pcm)),
                "p75": float(np.percentile(pcm, 75)),
                "max": float(pcm.max()),
            },
        }
        pooled_rows.extend(dlogps)
        print(
            f"{proc} comparison: cells={len(ep2_cells)} pairs={pairs} "
            f"identity={comparison[proc]['text_identity_rate']:.3f} "
            f"median_dlogp={float(np.median(arr)):.3f} "
            f"median_cell_max|dlogp|={float(np.median(pcm)):.3f}"
        )

    # Dose-direction cross-check against #474's per-epoch G matrices (#532 reuses
    # #474's loc-arm adapters). Like-for-like: #474 diagonal direction vs #532
    # diagonal-cell dlogp; #474 off-diagonal direction vs #532 offdiag-16 dlogp.
    g1 = json.loads((I474_CROSS / "loc_ep1/G_logprob_matrix.json").read_text())
    g2 = json.loads((I474_CROSS / "loc_ep2/G_logprob_matrix.json").read_text())

    def _mean_g(g: dict, diagonal: bool) -> float:
        # G[i][j]["g_logprob"] is the TRAINED-side marker logp — the same quantity as
        # #532's trained_logp_per_q (raw trained logp, not delta_g).
        vals = [
            g["G"][i][j]["g_logprob"] for i in g["G"] for j in g["G"][i] if (i == j) == diagonal
        ]
        return float(np.mean(vals))

    diag1, diag2 = _mean_g(g1, True), _mean_g(g2, True)
    off1, off2 = _mean_g(g1, False), _mean_g(g2, False)
    i474_dirs = {
        "diagonal": "ep2_higher" if diag2 > diag1 else "ep2_lower",
        "offdiag_16": "ep2_higher" if off2 > off1 else "ep2_lower",
    }
    i532_subset_medians = {
        k: (float(np.median(np.array(v))) if v else None) for k, v in subset_dlogps.items()
    }
    i532_dirs = {
        k: ("ep2_higher" if m > 0 else "ep2_lower")
        for k, m in i532_subset_medians.items()
        if m is not None and k in i474_dirs
    }
    dose_match = all(i532_dirs[k] == i474_dirs[k] for k in i532_dirs)
    i532_median = float(np.median(np.array(pooled_rows)))

    # Pre-registered verdict logic (plan §4 B.5: replay primary, strings confirmatory).
    replay_stale = sum(
        r["replay_capacity_1"]["n_stale"] for r in replays.values() if "replay_capacity_1" in r
    )
    affected_signal = all(
        comparison[p]["text_identity_rate"] >= IDENTITY_AFFECTED_THRESHOLD
        and comparison[p]["per_cell_max_abs_dlogp"]["median"] <= MAX_DLOGP_NEARZERO_NAT
        for p in comparison
        if comparison[p]["n_ep2_cells"] > 0
    )
    if replay_stale > 0 or affected_signal:
        verdict = "AFFECTED" if (replay_stale > 0) == affected_signal else "INDETERMINATE"
    else:
        # Genuine-different-adapter evidence; require the dose direction to match #474
        # on the like-for-like subsets, else escalate.
        verdict = "SAFE-by-eviction" if dose_match else "INDETERMINATE"
    print(
        f"#474 diag: ep1={diag1:.2f} ep2={diag2:.2f} ({i474_dirs['diagonal']}); "
        f"offdiag: ep1={off1:.2f} ep2={off2:.2f} ({i474_dirs['offdiag_16']})"
    )
    print(f"#532 subset median dlogp(ep2-ep1): {i532_subset_medians} -> dirs {i532_dirs}")
    print(f"#532 pooled median dlogp={i532_median:.3f}; dose_match={dose_match}")
    print(f"VERDICT: {verdict}")

    meta = {
        "tool": "i549_audit_532",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "run_commits": {"P1": P1_COMMIT, "P2": P2_COMMIT},
        "thresholds": {
            "identity_affected": IDENTITY_AFFECTED_THRESHOLD,
            "max_dlogp_nearzero_nat": MAX_DLOGP_NEARZERO_NAT,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "i532_per_process_replay.json").write_text(
        json.dumps({"metadata": meta, "replays": replays}, indent=2)
    )
    (OUT_DIR / "i532_ep1_ep2_comparison.json").write_text(
        json.dumps(
            {
                "metadata": meta,
                "comparison_per_process": comparison,
                "i474_dose_reference": {
                    "mean_diagonal_ep1": diag1,
                    "mean_diagonal_ep2": diag2,
                    "mean_offdiag_ep1": off1,
                    "mean_offdiag_ep2": off2,
                    "directions": i474_dirs,
                },
                "i532_subset_median_dlogp": i532_subset_medians,
                "i532_subset_directions": i532_dirs,
                "dose_direction_match": dose_match,
                "i532_pooled_median_dlogp": i532_median,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT_DIR}/i532_per_process_replay.json and i532_ep1_ep2_comparison.json")


if __name__ == "__main__":
    main()
