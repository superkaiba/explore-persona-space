"""Issue #958 round-2 revision artifacts: duplicate-group sidecar + long 1->k figure.

1. Writes eval_results/issue_958/duplicate_first_message_groups.json — the
   first-message duplicate-group derivation under BOTH normalizations (exact
   string vs lowercased), keyed by main-panel test conversation index, with the
   near-perfect-unit mask (fold-B pooled per-conversation skill > 0.95) and the
   excluded-aggregate reads. This is the mask behind Figure 2's split.
2. Adds recalibrated per-unit SSE + CIs to the long_1to{k} cells.
3. Renders figures/issue_958/long_turn1_transfer.{png,pdf,meta.json}.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import issue958_common as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

SNAP = Path(
    "/mnt/eps-data/thomasjiralerspong/i958_r2/hf/hub/"
    "datasets--superkaiba1--explore-persona-space-data/snapshots/"
    "06f13dfd9543b56c9a8caf986b1a6ef2ab5b3600/issue958_multiturn"
)
OUT = Path("eval_results/issue_958")
RO = [C.block_to_row(b) for b in C.READOUT_BLOCKS]
KS = [5, 6, 7, 8]


def write_sidecar() -> None:
    """Duplicate-group derivation, both normalizations, keyed by test conv index."""
    corpus = json.loads((SNAP / "corpus" / "main.json").read_text())
    first = [c["exchanges"][0]["user"] for c in corpus["conversations"]]
    z = np.load(OUT / "percell" / "own_k1_B.npz")
    test_idx = z["test_idx"]
    sse, null = z["sse_unit"], z["null_sse_unit"]
    pooled = 1 - sse[RO].sum(0) / np.clip(null[RO].sum(0), 1e-30, None)
    near = pooled > 0.95
    out: dict = {
        "definition": {
            "grouping_key": "first user message of the conversation",
            "normalizations": {
                "exact": "raw string equality (byte-identical seeded rollouts apply here)",
                "lowercased": "str.lower() equality (the grouping behind the round-1 "
                "12.8% / 251-copy / n=74 figures and Figure 2's split)",
            },
            "near_perfect_unit": "fold-B pooled per-conversation turn-1 skill > 0.95 "
            "(pooled over the 6 frozen read-out rows)",
        },
        "corpus_n": len(first),
        "per_normalization": {},
        "test_conversations": {},
    }
    masks = {}
    for name, keyfn in (("exact", lambda m: m), ("lowercased", lambda m: m.lower())):
        groups: dict = collections.defaultdict(list)
        for i, msg in enumerate(first):
            groups[keyfn(msg)].append(i)
        dup = {k: v for k, v in groups.items() if len(v) > 1}
        dupset = set(i for v in dup.values() for i in v)
        mask = np.array([int(i) in dupset for i in test_idx])
        masks[name] = mask
        keep = ~mask
        out["per_normalization"][name] = {
            "n_dup_conversations": sum(len(v) for v in dup.values()),
            "pct_dup": round(100 * sum(len(v) for v in dup.values()) / len(first), 2),
            "n_dup_groups": len(dup),
            "largest_groups": [
                {"size": len(v), "key_preview": k[:30]}
                for k, v in sorted(dup.items(), key=lambda kv: -len(kv[1]))[:3]
            ],
            "n_test_conversations_in_dup_groups": int(mask.sum()),
            "near_perfect_units_total": int(near.sum()),
            "near_perfect_units_in_dup_groups": int((near & mask).sum()),
            "foldB_aggregate_excluding_dup": float(
                np.mean([1 - sse[r][keep].sum() / max(null[r][keep].sum(), 1e-30) for r in RO])
            ),
            "pooled_per_conv_mean_unique_clipped_minus3": float(
                np.clip(pooled[keep], -3, None).mean()
            ),
        }
    for j, ci in enumerate(test_idx):
        out["test_conversations"][str(int(ci))] = {
            "dup_exact": bool(masks["exact"][j]),
            "dup_lowercased": bool(masks["lowercased"][j]),
            "near_perfect": bool(near[j]),
            "foldB_pooled_skill": float(pooled[j]),
        }
    out["metadata"] = C.reproducibility_metadata({"script": "issue958_r2_sidecar_and_fig"})
    C.write_json_atomic(OUT / "duplicate_first_message_groups.json", out)
    print("wrote", OUT / "duplicate_first_message_groups.json")


def recal_and_fig() -> None:
    """Recalibrated per-unit SSE for long_1to{k} + the summary/per-unit figure."""
    own = {k: np.load(OUT / "percell" / f"long_own_k{k}.npz") for k in [1, *KS]}
    test_idx = own[5]["test_idx"]

    def load_map(cell: str) -> dict:
        return torch.load(
            SNAP / "analysis_tensors" / "maps" / f"{cell}.pt",
            weights_only=False,
            map_location="cpu",
        )["rows"]

    rows_k1 = load_map("long_k1_own")
    tmaps = {k: load_map(f"long_k{k}_own") for k in KS}
    want = {C.unit_id("long", int(ci), k): (int(ci), k) for ci in test_idx for k in KS}
    units: dict = {}
    for p in sorted((SNAP / "analysis_tensors" / "store" / "long").glob("shard_*.pt")):
        blob = torch.load(p, weights_only=False, map_location="cpu")
        for uid, key in want.items():
            rec = blob["units"].get(uid)
            if rec is not None:
                units[key] = {
                    "ctx": rec["h"][C.POS_CTX_END].clone(),
                    "ans": rec["h"][C.POS_ANS_MEAN].clone(),
                }
        del blob
    res = json.loads((OUT / "long_k1_transfer.json").read_text())
    idx = np.random.default_rng(C.BOOTSTRAP_SEED).integers(
        0, len(test_idx), size=(C.BOOTSTRAP_DRAWS, len(test_idx))
    )

    def boot(sse: np.ndarray, null: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                1.0 - sse[r][idx].sum(1) / np.clip(null[r][idx].sum(1), 1e-30, None)
                for r in range(sse.shape[0])
            ]
        ).mean(0)

    for k in KS:
        sses, nulls = [], []
        for r in RO:
            m1, mk = rows_k1[r], tmaps[k][r]
            W = m1["w"].to(torch.float64)
            X = torch.stack([units[(int(ci), k)]["ctx"][r] for ci in test_idx]).to(torch.float64)
            Y = torch.stack([units[(int(ci), k)]["ans"][r] for ci in test_idx]).to(torch.float64)
            pred = (
                mk["ymu"].to(torch.float64)
                + ((X - mk["mu"].to(torch.float64)) / mk["sd"].to(torch.float64)) @ W
            )
            nm = mk["ymu"].to(torch.float64)
            sses.append(((pred - Y) ** 2).sum(-1).numpy())
            nulls.append(((Y - nm) ** 2).sum(-1).numpy())
        rs, rn = np.stack(sses), np.stack(nulls)
        rb = boot(rs, rn)
        cell = res["cells"][f"long_1to{k}"]
        cell["recalibrated_transfer_skill_ci95"] = [
            float(np.quantile(rb, q)) for q in (0.025, 0.975)
        ]
        pz = dict(np.load(OUT / "percell" / f"long_1to{k}.npz"))
        pz["recal_sse_unit"] = rs.astype(np.float32)
        pz["recal_null_sse_unit"] = rn.astype(np.float32)
        np.savez(OUT / "percell" / f"long_1to{k}.npz", **pz)
    C.write_json_atomic(OUT / "long_k1_transfer.json", res)
    print("updated long_k1_transfer.json with recal CIs")

    # figure: left = summary bars per turn; right = per-conversation view at turn 5
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    cols = paper_palette(3)
    ax = axes[0]
    xs = np.arange(len(KS))
    w = 0.27
    own_p = [float(np.mean([own[k]["skill"][r] for r in RO])) for k in KS]
    own_ci = []
    for k in KS:
        ob = boot(own[k]["sse_unit"][RO], own[k]["null_sse_unit"][RO])
        own_ci.append([float(np.quantile(ob, q)) for q in (0.025, 0.975)])
    xfer_p = [res["cells"][f"long_1to{k}"]["transfer_skill"] for k in KS]
    xfer_ci = [res["cells"][f"long_1to{k}"]["transfer_skill_ci95"] for k in KS]
    rec_p = [res["cells"][f"long_1to{k}"]["recalibrated_transfer_skill"] for k in KS]
    rec_ci = [res["cells"][f"long_1to{k}"]["recalibrated_transfer_skill_ci95"] for k in KS]

    def err(p: list, ci: list) -> np.ndarray:
        return np.abs(np.array(ci).T - np.array(p))

    ax.bar(xs - w, own_p, w, yerr=err(own_p, own_ci), color=cols[0], label="own-turn map")
    ax.bar(
        xs,
        xfer_p,
        w,
        yerr=err(xfer_p, xfer_ci),
        color=cols[1],
        label="turn-1 map (as fitted)",
    )
    ax.bar(
        xs + w,
        rec_p,
        w,
        yerr=err(rec_p, rec_ci),
        color=cols[2],
        label="turn-1 map, target-turn moments",
    )
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(xs, [f"turn {k}" for k in KS])
    ax.set_ylabel("held-out skill (6-block mean)")
    ax.set_xlabel("evaluation turn (long panel, 60 test conversations)")
    ax.legend(frameon=False, fontsize=9)
    ax2 = axes[1]
    k = 5
    po = 1 - own[k]["sse_unit"][RO].sum(0) / np.clip(
        own[k]["null_sse_unit"][RO].sum(0), 1e-30, None
    )
    pz = np.load(OUT / "percell" / f"long_1to{k}.npz")
    pt = 1 - pz["sse_unit"].sum(0) / np.clip(pz["null_sse_unit"].sum(0), 1e-30, None)
    po_c, pt_c = np.clip(po, -3, None), np.clip(pt, -3, None)
    ax2.scatter(po_c, pt_c, s=22, color=cols[1], alpha=0.8)
    lim = [-3.1, 1.05]
    ax2.plot(lim, lim, color="0.6", lw=0.8, ls="--")
    for i in np.argsort(pt_c)[:2]:
        ax2.text(po_c[i], pt_c[i], f"conv {int(test_idx[i])}", fontsize=7, va="bottom")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("own turn-5 map skill per conversation (clipped at -3)")
    ax2.set_ylabel("turn-1 map skill at turn 5 (clipped at -3)")
    savefig_paper(fig, "long_turn1_transfer", dir="figures/issue_958")
    print("wrote figures/issue_958/long_turn1_transfer.png")


if __name__ == "__main__":
    torch.set_num_threads(8)
    write_sidecar()
    recal_and_fig()
