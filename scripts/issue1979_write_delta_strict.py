"""Strict per-prefix write-vs-training-displacement test for issue #1979.

Closes the scope caveat this task's own A5 result records: A5 raced the
matched-text write against the *on-policy delta*, not against the theory's
training displacement, so the strict prefix-grain re-test of the #1768
horse race ("write aligns with delta") never ran.

This round computes, per (arm, layer, prefix), the cosine between the write
and the training displacement

    delta = A_ans - v0_baseline          (training answers minus base answers)
    write = (W + Vbar0) - v0_baseline'   (trained answer state minus base)

under the disjoint-baseline convention already banked in this task's
battery: the write leg takes the EVEN half of the base answer vectors and
the displacement leg the ODD half, so the two legs share no baseline
sampling error (the same split #1979's A5 uses at issue1979_race.py:928).

Legs reported
  on-policy write  vs delta   -- the #1768 horse-race on-policy twin
  matched-text write vs delta -- the weights-carried companion
  pooled delta (one per arm-layer, the #1768 convention) and per-prefix
  delta (A_ans minus THIS prefix's base answer)
  centroid anchor and 20-row anchor (mean and top-8, the p10 convention)
  own trained prefix split out from the other 49
  corpus-covariance norm-matched null bands (2,000 draws, |cos| p95)

No fit, no model forward, no generation: every input is a banked tensor.

Usage:
    uv run python scripts/issue1979_write_delta_strict.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Before torch: torch freezes its intra-op thread pool from OMP_NUM_THREADS at
# IMPORT, and load_dotenv() is what sets the shared-VM thread caps.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

DL = Path("data/issue_1979/hf_dl/issue1979_prefixrace")
OUT = Path("eval_results/issue_1979/write_delta_strict")
FRAMES = Path("eval_results/issue_1979/race")
LAYERS = (14, 19, 25)
POS = "last_prompt"
N_NULL = 2000
SEED = 1979
TOPK = 8
PARITY_TOL = 5e-3

# Arm metadata (kind, ctx_key, primary_layer, mix_arm_id) is read from the
# run's own committed config rather than inferred: content arms are primary
# at layer 19 and marker arms at layer 25, and positive-only / full
# fine-tune arms resolve their training-answer anchor from a sibling mix.
ARMS_CFG = DL / "config/arms.json"

OWN_PREFIX_BY_CTX = {
    "pers": "persona_software_engineer",
    "bare": "bare",
    "wc": "wildchat_prefix_real545",
    "conv": "wildchat_prefix_real545",
    "icl": "icl_prefix_sycophancy",
}


def _unit(a: np.ndarray) -> np.ndarray:
    """Row-normalise; a zero row stays zero rather than producing a nan."""
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n > 0, n, 1.0)


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine of (n, d) against (d,) or (n, d)."""
    b2 = np.broadcast_to(b, a.shape) if b.ndim == 1 else b
    return (_unit(a) * _unit(b2)).sum(axis=-1)


def load_inputs() -> tuple[dict, dict, dict]:
    ten = torch.load(DL / "battery/ingredient_tensors.pt", map_location="cpu", weights_only=False)
    tensors = {k: np.asarray(v.float().numpy(), dtype=np.float64) for k, v in ten.items()}
    sig = torch.load(DL / "battery/sigma_chol.pt", map_location="cpu", weights_only=False)
    anchors = {}
    for d in sorted((DL / "anchors").iterdir()):
        if d.is_dir():
            anchors[d.name] = torch.load(d / "anchors.pt", map_location="cpu", weights_only=False)
    assert anchors, f"no anchors under {DL / 'anchors'}"
    return tensors, sig, anchors


def prefix_ids_for(arm: str) -> list[str]:
    p = FRAMES / f"frame_{arm}.json"
    ids = json.loads(p.read_text())["frame"]["prefix_id"]
    assert len(ids) == 50, (arm, len(ids))
    return ids


def null_dirs(sig: dict, layer: int, rng: np.random.Generator) -> np.ndarray:
    """(N_NULL, d) unit draws from the shrunk corpus covariance."""
    chol = np.asarray(sig[f"L{layer}"]["chol"].float().numpy(), dtype=np.float64)
    z = rng.standard_normal((N_NULL, chol.shape[0]))
    return _unit(z @ chol.T)


def band(write: np.ndarray, gdirs: np.ndarray) -> dict:
    """Corpus-covariance null bands for the write-vs-direction cosine.

    Two bands, because the headline statistic is a MEDIAN over prefixes:

    per_prefix -- |cos| p95 over all (prefix, draw) pairs. The band a SINGLE
      prefix's cosine must clear. This is the #1979 A5 convention.
    median -- p95 of |median over prefixes of cos(write_p, g)| across draws.
      The band the median-over-50 statistic must clear. Much tighter, because
      averaging 50 prefixes against one shared random direction cancels most
      of the per-prefix noise; comparing a median to the per-prefix band is
      the mismatched, over-conservative test.
    """
    c = _unit(write) @ gdirs.T  # (n_prefix, n_draw)
    return {
        "per_prefix": float(np.percentile(np.abs(c), 95)),
        "median": float(np.percentile(np.abs(np.median(c, axis=0)), 95)),
    }


def cell(
    meta: dict,
    layer: int,
    tensors: dict,
    anc: dict,
    gdirs: np.ndarray,
    own_ix: int | None,
) -> dict:
    arm, kind = meta["arm_id"], meta["kind"]
    slot = f"{arm}/L{layer}/{POS}"
    vbar0 = tensors[f"{slot}/Vbar0"]
    w_on = tensors[f"{slot}/W_onpolicy"]
    w_mt = tensors[f"{slot}/W_matched"]
    v_even = tensors[f"base/{kind}/L{layer}/Vbar0_even"]
    v_odd = tensors[f"base/{kind}/L{layer}/Vbar0_odd"]

    a = anc[f"L{layer}"]
    a_ans = np.asarray(a["A_ans"].float().numpy(), dtype=np.float64)
    rows_ans = np.asarray(a["rows_ans"].float().numpy(), dtype=np.float64)

    # Parity assert: reproduce the banked m4_onpolicy column (cos of the raw
    # on-policy write against the panel-centred anchor) to prove the tensor
    # row order matches the committed frame row order.
    banked = json.loads((FRAMES / f"frame_{arm}.json").read_text())["frame"]["m4_onpolicy"]
    if layer == meta["primary_layer"] and banked is not None and banked[0] is not None:
        repro = _cos_rows(w_on, a_ans - vbar0.mean(axis=0))
        gap = float(np.max(np.abs(repro - np.asarray(banked, dtype=np.float64))))
        assert gap < PARITY_TOL, (arm, layer, "frame parity", gap)
    else:
        gap = None

    # Write legs, disjoint baseline (EVEN half) -- the A5 convention.
    write_on = (w_on + vbar0) - v_even
    write_mt = (w_mt + vbar0) - v_even
    # Displacement legs, disjoint baseline (ODD half).
    d_pooled = a_ans - v_odd.mean(axis=0)
    d_perpfx = a_ans[None, :] - v_odd
    # Row-level: 20 individual training answers against the pooled odd baseline.
    d_rows = rows_ans - v_odd.mean(axis=0)[None, :]
    row_cos_on = _unit(write_on) @ _unit(d_rows).T  # (50, 20)

    out: dict = {
        "arm_id": arm,
        "kind": kind,
        "ctx_key": meta["ctx_key"],
        "mix_arm_id": meta["mix_arm_id"],
        "layer": layer,
        "is_primary_layer": layer == meta["primary_layer"],
        "position": POS,
        "n_prefix": int(write_on.shape[0]),
        "n_anchor_rows": int(rows_ans.shape[0]),
        "frame_parity_max_abs_gap": gap,
        "null_band_p95_abs_cos": {
            "on_policy": band(write_on, gdirs),
            "matched_text": band(write_mt, gdirs),
        },
        "delta_norms": {
            "pooled": float(np.linalg.norm(d_pooled)),
            "write_on_median": float(np.median(np.linalg.norm(write_on, axis=1))),
            "write_mt_median": float(np.median(np.linalg.norm(write_mt, axis=1))),
        },
    }
    for name, wr in (("on_policy", write_on), ("matched_text", write_mt)):
        cp = _cos_rows(wr, d_pooled)
        cx = _cos_rows(wr, d_perpfx)
        rc = row_cos_on if name == "on_policy" else _unit(wr) @ _unit(d_rows).T
        topk = np.sort(rc, axis=1)[:, ::-1][:, :TOPK].mean(axis=1)
        out[name] = {
            "delta_pooled": {
                "median": float(np.median(cp)),
                "mean": float(cp.mean()),
                "frac_pos": float((cp > 0).mean()),
                "own_prefix": (None if own_ix is None else float(cp[own_ix])),
                "others_median": (
                    None if own_ix is None else float(np.median(np.delete(cp, own_ix)))
                ),
                "per_prefix": [float(x) for x in cp],
            },
            "delta_per_prefix": {
                "median": float(np.median(cx)),
                "mean": float(cx.mean()),
                "frac_pos": float((cx > 0).mean()),
                "own_prefix": (None if own_ix is None else float(cx[own_ix])),
            },
            "delta_rows_mean": {"median": float(np.median(rc.mean(axis=1)))},
            "delta_rows_top8": {"median": float(np.median(topk))},
        }
    return out


def main() -> None:
    tensors, sig, anchors = load_inputs()
    arms = json.loads(ARMS_CFG.read_text())["arms"]
    rng = np.random.default_rng(SEED)
    gd = {ly: null_dirs(sig, ly, rng) for ly in LAYERS}
    cells = []
    for meta in sorted(arms, key=lambda a: a["arm_id"]):
        ids = prefix_ids_for(meta["arm_id"])
        own_pid = OWN_PREFIX_BY_CTX.get(meta["ctx_key"])
        own_ix = ids.index(own_pid) if own_pid in ids else None
        anc = anchors[meta["mix_arm_id"]]
        for ly in LAYERS:
            cells.append(cell(meta, ly, tensors, anc, gd[ly], own_ix))
    n_parity = sum(1 for c in cells if c["frame_parity_max_abs_gap"] is not None)
    assert n_parity == len(arms), (n_parity, len(arms), "every arm must parity-check")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "question": (
                    "Does the on-policy write align with the training displacement "
                    "at prefix grain? (strict #1768 horse-race twin)"
                ),
                "layers": list(LAYERS),
                "position": POS,
                "primary_layer_by_kind": {"content": 19, "marker": 25},
                "n_arms": len(arms),
                "n_null_draws": N_NULL,
                "seed": SEED,
                "shrinkage": sig["shrinkage"],
                "parity_checked_cells": n_parity,
                "cells": cells,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'summary.json'} ({len(cells)} cells, {n_parity} parity-checked)")


if __name__ == "__main__":
    main()
