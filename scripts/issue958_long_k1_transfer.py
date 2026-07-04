"""Issue #958 round-2 free re-reduction: apply the clean long-panel turn-1 map at turns 5-8.

Closes the 1->k directional leg of the turn-1 stationarity read with a NON-degenerate
source fit (the long-panel turn-1 map: lambda ~5, own skill 0.42), using ONLY persisted
artifacts: the saved map .pt files (composite W per read-out row, fp16) + the long-panel
activation store shards on the HF data repo. No GPU, no new data.

Validation gate: reproduces the committed `long_own_k1` cell (0.42047) from the saved
map before any new cell is trusted (the #931 W_raw-reproduces-committed-rows recipe).

Writes eval_results/issue_958/long_k1_transfer.json + percell/long_1to{k}.npz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue958_common as C

SNAP = Path(
    "/mnt/eps-data/thomasjiralerspong/i958_r2/hf/hub/"
    "datasets--superkaiba1--explore-persona-space-data/snapshots/"
    "06f13dfd9543b56c9a8caf986b1a6ef2ab5b3600/issue958_multiturn/analysis_tensors"
)
OUT = Path("eval_results/issue_958")
RO = [C.block_to_row(b) for b in C.READOUT_BLOCKS]  # [15, 18, 20, 21, 25, 27]
KS = [5, 6, 7, 8]


def load_map(cell: str) -> dict:
    """Persisted per-row composite affine: {row: {w fp16 (H,H), mu, sd, ymu fp32}}."""
    blob = torch.load(SNAP / "maps" / f"{cell}.pt", weights_only=False, map_location="cpu")
    assert blob["policy"] == "source-map-composite", blob["policy"]
    return blob["rows"]


def load_long_units(test_idx: np.ndarray, ks: list[int]) -> dict:
    """{(ci, k): h fp16 (29, H) at ctx_end / ans_mean} for the long test conversations."""
    want = {C.unit_id("long", int(ci), k): (int(ci), k) for ci in test_idx for k in ks}
    out: dict = {}
    for p in sorted((SNAP / "store" / "long").glob("shard_*.pt")):
        blob = torch.load(p, weights_only=False, map_location="cpu")
        for uid, key in want.items():
            rec = blob["units"].get(uid)
            if rec is not None:
                out[key] = {
                    "ctx": rec["h"][C.POS_CTX_END].clone(),
                    "ans": rec["h"][C.POS_ANS_MEAN].clone(),
                }
        del blob
    missing = [u for u, k in want.items() if k not in out]
    assert not missing, f"long store missing {len(missing)} test units, e.g. {missing[:3]}"
    return out


def cell_arrays(rows_map: dict, units: dict, test_idx: np.ndarray, k: int, null_ymu: dict) -> dict:
    """Per-readout-row skill + per-unit SSE arrays for map applied at turn k."""
    skills, sses, nulls = [], [], []
    for r in RO:
        m = rows_map[r]
        W = m["w"].to(torch.float64)
        mu, sd, ymu = (m[f].to(torch.float64) for f in ("mu", "sd", "ymu"))
        X = torch.stack([units[(int(ci), k)]["ctx"][r] for ci in test_idx]).to(torch.float64)
        Y = torch.stack([units[(int(ci), k)]["ans"][r] for ci in test_idx]).to(torch.float64)
        pred = ymu + ((X - mu) / sd) @ W
        nm = null_ymu[r].to(torch.float64)  # target-turn train-fold corpus mean
        sse_u = ((pred - Y) ** 2).sum(-1)
        null_u = ((Y - nm) ** 2).sum(-1)
        skills.append(1.0 - float(sse_u.sum()) / max(float(null_u.sum()), 1e-30))
        sses.append(sse_u.numpy())
        nulls.append(null_u.numpy())
    return {
        "skill_rows": np.array(skills),
        "sse_unit": np.stack(sses),  # (6, n_t)
        "null_sse_unit": np.stack(nulls),
    }


def boot(sse: np.ndarray, null: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(draws,) read-out-mean skill under each paired conversation resample."""
    return np.stack(
        [1.0 - sse[r][idx].sum(1) / np.clip(null[r][idx].sum(1), 1e-30, None) for r in range(6)]
    ).mean(0)


def main() -> int:
    """Run validation gate + the four long_1to{k} transfer cells; write JSON + npz."""
    torch.set_num_threads(8)
    own = {k: np.load(OUT / "percell" / f"long_own_k{k}.npz") for k in [1, *KS]}
    test_idx = own[5]["test_idx"]
    for k in [1, *KS]:
        assert np.array_equal(own[k]["test_idx"], test_idx)
    rows_k1 = load_map("long_k1_own")
    null_maps = {k: load_map(f"long_k{k}_own") for k in [1, *KS]}
    units = load_long_units(test_idx, [1, *KS])

    # validation gate: reproduce committed long_own_k1 from the saved fp16 map
    val = cell_arrays(rows_k1, units, test_idx, 1, {r: null_maps[1][r]["ymu"] for r in RO})
    committed = np.array([own[1]["skill"][r] for r in RO])
    dmax = float(np.abs(val["skill_rows"] - committed).max())
    print(f"[gate] long_own_k1 recompute vs committed: max|d|={dmax:.2e}")
    assert dmax < 5e-3, f"validation gate FAILED: {dmax}"

    idx = np.random.default_rng(C.BOOTSTRAP_SEED).integers(
        0, len(test_idx), size=(C.BOOTSTRAP_DRAWS, len(test_idx))
    )
    res: dict = {
        "validation_gate": {"max_abs_row_delta_vs_committed_long_own_k1": dmax},
        "policy": "source-map-composite",
        "cells": {},
    }
    for k in KS:
        cell = cell_arrays(rows_k1, units, test_idx, k, {r: null_maps[k][r]["ymu"] for r in RO})
        # recalibrated companion: turn-k train-fold moments from the turn-k own map
        recal_sk = []
        for i, r in enumerate(RO):
            m1, mk = rows_k1[r], null_maps[k][r]
            W = m1["w"].to(torch.float64)
            X = torch.stack([units[(int(ci), k)]["ctx"][r] for ci in test_idx]).to(torch.float64)
            Y = torch.stack([units[(int(ci), k)]["ans"][r] for ci in test_idx]).to(torch.float64)
            pred = (
                mk["ymu"].to(torch.float64)
                + ((X - mk["mu"].to(torch.float64)) / mk["sd"].to(torch.float64)) @ W
            )
            nm = mk["ymu"].to(torch.float64)
            recal_sk.append(
                1.0 - float(((pred - Y) ** 2).sum()) / max(float(((Y - nm) ** 2).sum()), 1e-30)
            )
            _ = i
        xfer_b = boot(cell["sse_unit"], cell["null_sse_unit"], idx)
        own_sse = own[k]["sse_unit"][RO]
        own_null = own[k]["null_sse_unit"][RO]
        own_b = boot(own_sse, own_null, idx)
        own_p = float(np.mean([own[k]["skill"][r] for r in RO]))
        xfer_p = float(cell["skill_rows"].mean())
        d_b = own_b - xfer_b
        res["cells"][f"long_1to{k}"] = {
            "transfer_skill": xfer_p,
            "transfer_skill_ci95": [float(np.quantile(xfer_b, q)) for q in (0.025, 0.975)],
            "own_skill": own_p,
            "deficit": own_p - xfer_p,
            "deficit_ci95": [float(np.quantile(d_b, q)) for q in (0.025, 0.975)],
            "recalibrated_transfer_skill": float(np.mean(recal_sk)),
            "n_test": len(test_idx),
        }
        np.savez(
            OUT / "percell" / f"long_1to{k}.npz",
            skill=cell["skill_rows"],
            sse_unit=cell["sse_unit"].astype(np.float32),
            null_sse_unit=cell["null_sse_unit"].astype(np.float32),
            test_idx=test_idx,
            readout_rows=np.array(RO),
        )
        print(k, json.dumps(res["cells"][f"long_1to{k}"], indent=None))
    res["metadata"] = C.reproducibility_metadata({"script": "issue958_long_k1_transfer"})
    C.write_json_atomic(OUT / "long_k1_transfer.json", res)
    print("wrote", OUT / "long_k1_transfer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
