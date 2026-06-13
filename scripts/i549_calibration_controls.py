"""Calibration controls for the #549 cache-collision audit rubric (plan §12 item 4).

Classifies the known-outcome runs through the SAME ``i549_lru_simulate`` machinery used
for production verdicts — never prose arguments:

1. #534 round-1 (positive control, must -> AFFECTED): constant ``lora_int_id=1`` across
   the 4 realized checkpoint fractions of one engine session (pre-fix
   ``eval_trajectory.py:425``, still live on main; fixed only at 298877f9c). Paths are
   the REAL realized checkpoints from
   ``eval_results/issue_534/c504v3_near_seed42/fraction_manifest.json``.
2. #534 round-2 (negative control, must -> SAFE): the same 4 paths with the fix's
   ``lora_int_id=ck_i`` (``enumerate(checkpoint_specs, start=1)``, 298877f9c).
3. #474 phase-4 (negative control, must -> SAFE): ``all_cids.index(cid)+1`` over the
   16 realized conditions (``eval_results/issue_474/cross_eval/loc_ep1/
   G_logprob_matrix.json``), replayed unsharded AND as the realized 4-way strided
   shards (``my_cids = source_cids[k % n_shards == shard_idx]``,
   ``scripts/i474_phase4_eval.py:418``); epoch is fixed per invocation
   (``--checkpoint-epoch`` required), so no cross-epoch id reuse exists in-process.
4. #460 phase-4 (negative control, must -> SAFE): same id discipline
   (``scripts/i460_phase4_eval.py:326-328``) over the same 16-condition active set.

Exit 0 iff every control classifies as expected; any mismatch raises.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from i549_lru_simulate import simulate, summarize

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_results/issue_549/audit_evidence/calibration_controls.json"


def main() -> None:
    results: dict[str, dict] = {}

    manifest = json.loads(
        (REPO / "eval_results/issue_534/c504v3_near_seed42/fraction_manifest.json").read_text()
    )
    paths = [
        entry["path"] for _, entry in sorted(manifest["index"].items(), key=lambda kv: float(kv[0]))
    ]
    assert len(paths) == 4 and len(set(paths)) == 4, paths

    # Control 1: #534 r1 — constant id over 4 distinct checkpoint paths, capacity 1.
    s = summarize(simulate([(1, p) for p in paths], capacity=1))
    assert s["n_stale"] == 3 and s["verdict_hint"] == "AFFECTED", s
    results["i534_round1_constant_id"] = {**s, "expected": "AFFECTED", "pass": True}

    # Control 2: #534 r2 — distinct ck_i ids over the same paths.
    s = summarize(simulate([(i + 1, p) for i, p in enumerate(paths)], capacity=1))
    assert s["n_stale"] == 0, s
    results["i534_round2_distinct_ck_i"] = {**s, "expected": "SAFE", "pass": True}

    # Controls 3+4: distinct index-based ids over the 16-condition active set.
    g = json.loads(
        (REPO / "eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json").read_text()
    )
    all_cids = g["conditions"]
    assert len(all_cids) == 16, all_cids
    for name, path_tpl in (
        ("i474_phase4_loc_ep1", "adapters/i474_loc_{cid}_ep1"),
        ("i460_phase4", "adapters/i460_{cid}"),
    ):
        seq_full = [(all_cids.index(c) + 1, path_tpl.format(cid=c)) for c in all_cids]
        s_full = summarize(simulate(seq_full, capacity=1))
        assert s_full["n_stale"] == 0, (name, s_full)
        shard_summaries = {}
        for shard in range(4):
            my_cids = [c for k, c in enumerate(all_cids) if k % 4 == shard]
            seq = [(all_cids.index(c) + 1, path_tpl.format(cid=c)) for c in my_cids]
            s_sh = summarize(simulate(seq, capacity=1))
            assert s_sh["n_stale"] == 0, (name, shard, s_sh)
            shard_summaries[f"shard_{shard}_of_4"] = s_sh
        results[name] = {
            "unsharded": s_full,
            "shards": shard_summaries,
            "expected": "SAFE",
            "pass": True,
        }

    payload = {
        "metadata": {
            "tool": "i549_calibration_controls",
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
        "controls": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"calibration controls: {len(results)}/4 groups PASS — wrote {OUT}")


if __name__ == "__main__":
    main()
