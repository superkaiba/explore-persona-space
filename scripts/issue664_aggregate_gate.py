#!/usr/bin/env python
"""issue #664 P2.5 aggregate (analysis-time) — compute the PRIMARY DV ĝ^real
from the trained store tensors, streaming one cell at a time (download → compute
→ delete) to stay under the VM analysis-footprint floor.

ĝ^real(C') = ŵᵀ Δv(C') / ŵᵀ ŵ ,  ŵ = Δv(C) = v⁺(C) − v0(C) ,  Δv(C') = v⁺(C') − v0(C')
(plan §6.1). By construction ĝ^real(C')=1 at C'=C (source anchor).

Within-context probe-split noise floor (kill 3(b), design-doc §1.7): split the
50 probes per context into two independent halves, compute ĝ on each half's mean
Δv, the floor per cell = the spread of the half-split ĝ across bystander contexts
(a measurement-noise band the real cross-context variation must exceed).

Output: eval_results/issue_664/gate_real/<cell>/g_real.json  (per-cell, per-layer,
per-context ĝ with target_context_role), + an aggregate gate_real_summary.json.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import hf_hub_download  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_PREFIX = "theory_assumptions/Qwen2.5-7B-Instruct/issue664"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_results" / "issue_664" / "gate_real"
OUT.mkdir(parents=True, exist_ok=True)


def gate_per_layer(v_plus, v0, source_idx):
    """Return (n_ctx, n_layer) ĝ^real and ŵ-norm per layer.

    v_plus, v0 : (n_ctx, n_layer, d) float tensors.
    """
    dv = (v_plus - v0).numpy().astype(np.float64)  # (C, L, d)
    w = dv[source_idx]  # (L, d) ŵ = Δv(C)
    wnorm2 = (w * w).sum(axis=-1)  # (L,)
    # ĝ(C',l) = <w_l, dv(C',l)> / <w_l, w_l>
    num = np.einsum("cld,ld->cl", dv, w)  # (C, L)
    ghat = num / wnorm2[None, :]  # (C, L)
    return ghat, np.sqrt(wnorm2)


def probe_split_floor(v_plus_probe, v0_probe, source_idx, rng):
    """Within-context probe-split ĝ floor.

    *_probe : (n_ctx, n_probe, n_layer, d). Split the n_probe axis in two random
    halves, compute ĝ from each half's mean Δv, return the per-(ctx,layer) ABS
    difference between the two half-estimates (a measurement-noise magnitude).
    """
    C, P, L, d = v_plus_probe.shape
    dv = (v_plus_probe - v0_probe).numpy().astype(np.float64)  # (C,P,L,d)
    perm = rng.permutation(P)
    h1, h2 = perm[: P // 2], perm[P // 2 :]
    dvm1 = dv[:, h1].mean(axis=1)  # (C,L,d)
    dvm2 = dv[:, h2].mean(axis=1)
    w1 = dvm1[source_idx]  # (L,d)
    w2 = dvm2[source_idx]
    wn1 = (w1 * w1).sum(-1)
    wn2 = (w2 * w2).sum(-1)
    g1 = np.einsum("cld,ld->cl", dvm1, w1) / wn1[None, :]
    g2 = np.einsum("cld,ld->cl", dvm2, w2) / wn2[None, :]
    return np.abs(g1 - g2)  # (C,L) noise magnitude


def process_cell(cell: str):
    cdir = OUT / cell
    cdir.mkdir(parents=True, exist_ok=True)
    outp = cdir / "g_real.json"
    if outp.exists():
        print(f"  [skip] {cell} (g_real.json exists)")
        return json.load(open(outp))
    print(f"  [dl]   {cell}")
    tp = hf_hub_download(DATA_REPO, f"{STORE_PREFIX}/{cell}/tensors.pt", repo_type="dataset")
    mp = hf_hub_download(DATA_REPO, f"{STORE_PREFIX}/{cell}/meta.json", repo_type="dataset")
    meta = json.load(open(mp))
    d = torch.load(tp, map_location="cpu")
    ctx_ids = list(d["context_ids"])
    roles = meta["target_context_roles"]
    src_id = meta["source"]  # key like "librarian"
    # source context instance id = the meta source-anchor entry
    anchor_ids = [cid for cid, r in roles.items() if r == "source-anchor"]
    assert len(anchor_ids) == 1, (cell, anchor_ids)
    source_ctx_id = anchor_ids[0]
    source_idx = ctx_ids.index(source_ctx_id)

    ghat, wnorm = gate_per_layer(d["v_plus"], d["v0"], source_idx)  # (C,L),(L,)
    rng = np.random.default_rng(42)
    floor = probe_split_floor(d["v_plus_probe"], d["v0_probe"], source_idx, rng)  # (C,L)

    n_ctx, n_layer = ghat.shape
    rows = []
    for ci, cid in enumerate(ctx_ids):
        role = roles.get(cid, "bystander")
        rows.append(
            {
                "context_id": cid,
                "target_context_role": role,
                "ghat_by_layer": [round(float(x), 6) for x in ghat[ci]],
                "floor_by_layer": [round(float(x), 6) for x in floor[ci]],
            }
        )
    rec = {
        "cell": cell,
        "behavior": meta["behavior"],
        "source": src_id,
        "arm": meta["arm"],
        "dose": meta["dose"],
        "seed": meta["seed"],
        "source_ctx_id": source_ctx_id,
        "source_idx": source_idx,
        "n_contexts": n_ctx,
        "n_layers": n_layer,
        "wnorm_by_layer": [round(float(x), 4) for x in wnorm],
        "rows": rows,
        "git_commit": meta.get("git_commit"),
        "sha256_tensors": meta.get("sha256_tensors"),
    }
    json.dump(rec, open(outp, "w"), indent=1)
    # free + delete the big tensor from cache to bound footprint
    del d
    gc.collect()
    try:
        os.remove(tp)
    except OSError:
        pass
    print(f"  [done] {cell} -> {outp}")
    return rec


def main():
    cells = sys.argv[1:]
    assert cells, "pass cell slugs"
    for cell in cells:
        process_cell(cell)


if __name__ == "__main__":
    main()
