"""Excluded-intrusion recount of the issue-2215 registered DV3 pooled 2AFC.

Language-intrusion audit (analyzer Step 3.7): rebuilds the per-context answer
means v_A (tail pooling, L19) from the va2215 store TWICE — (a) all valid
draws (validation twin: must reproduce the committed pooled accuracies) and
(b) CJK-intruded draws EXCLUDED — then recomputes the paired 2AFC accuracy
(cosine, L19, tail) for all five registered arms, pooled + per-type.

Stream-and-delete: each shard is downloaded, sliced at L19, deleted.
"""

import json
import os
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps must be set before torch import (shared-VM rule)

import numpy as np  # noqa: E402
import torch  # noqa: E402

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2215")
sys.path.insert(0, str(WT / "scripts"))
from huggingface_hub import hf_hub_download, list_repo_tree  # noqa: E402

from explore_persona_space.orchestrate.hub import retry_transient  # noqa: E402


import issue779_ffc_n1m_fits as FITS  # noqa: E402
from issue2215_analysis import (  # noqa: E402
    PairTable,
    build_cell_views,
    idbias_loto_predict,
    observed_2afc,
    sim_blocks,
)

REPO = "superkaiba1/explore-persona-space-data"
PIN = "dc8108ab84f33695bbc769da0e6e8e2327f51eeb"  # parent-artifact revision (plan §4.1)
STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue2215_cjk_recount")
STAGE.mkdir(parents=True, exist_ok=True)
L = 19

bank = json.load(
    open("/tmp/issue2215_staging/issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json")
)
flags = {tuple(x) for x in json.load(open("/tmp/issue2215_cjk_flags.json"))}
pt = PairTable.from_bank(bank, None)
views = build_cell_views(bank, pt)
degenerate_pe = set(bank["degenerate_at_pe_cells"])
n_ctx = len(pt.ids)
print(f"[recount] {n_ctx} contexts, {len(pt.pair_ids)} pairs, {len(flags)} intruded draws")

# ── vc bank (L19 slices only) ────────────────────────────────────────
p = retry_transient(
    lambda: hf_hub_download(
        REPO,
        "issue2162_ctxinfo/analysis_tensors/vc_bank/vc_bank.pt",
        repo_type="dataset",
        revision=PIN,
        local_dir=STAGE,
    ),
    what="hf_hub_download vc_bank.pt",
)
payload = torch.load(p, map_location="cpu", weights_only=False)
layers = list(payload["layers"])
li = layers.index(L)
recs = payload["per_context"]
vc = {
    slot: np.stack([recs[cid][key][li].float().numpy() for cid in pt.ids]).astype(np.float64)
    for slot, key in (("ce", "v_ce"), ("pe", "v_pe"))
}
del payload, recs
Path(p).unlink()
print("[recount] vc bank sliced at L19")

# ── va2215 store: stream shards, keep L19 tail sums ──────────────────
row_of = pt.row_of
sums_all = np.zeros((n_ctx, 3584))
sums_cln = np.zeros((n_ctx, 3584))
n_all = np.zeros(n_ctx, dtype=np.int64)
n_cln = np.zeros(n_ctx, dtype=np.int64)
shards = [
    e.path
    for e in retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: raw call is wrapped in hub.retry_transient right here
            list_repo_tree(
                REPO,
                path_in_repo="issue2215_reprshift/analysis_tensors/va2215",
                repo_type="dataset",
            )
        ),
        what="list_repo_tree va2215",
    )
    if e.path.endswith(".pt")
]
assert len(shards) == 16, shards
for sp in sorted(shards):
    f = retry_transient(
        lambda sp=sp: hf_hub_download(REPO, sp, repo_type="dataset", local_dir=STAGE),
        what=f"hf_hub_download {sp}",
    )
    pl = torch.load(f, map_location="cpu", weights_only=False)
    assert list(pl["layers"]) == layers
    empty = set(pl.get("empty_rows", []))
    va = pl["va_tail_incl"][:, li, :].double().numpy()
    for j, meta in enumerate(pl["index"]):
        cid, draw = meta["context_id"], int(meta["draw"])
        if j in empty or cid not in row_of:
            continue
        r = row_of[cid]
        sums_all[r] += va[j]
        n_all[r] += 1
        if (cid, draw) not in flags:
            sums_cln[r] += va[j]
            n_cln[r] += 1
    del pl, va
    Path(f).unlink()
    print(f"[recount] {sp.split('/')[-1]} done")
assert (n_all == 10).all(), "expected 10 valid draws everywhere (parent gate)"
mean_all = sums_all / np.maximum(n_all, 1)[:, None]
mean_cln = sums_cln / np.maximum(n_cln, 1)[:, None]
n_zero_clean = int((n_cln == 0).sum())
print(f"[recount] contexts with 0 clean draws: {n_zero_clean}")

# ── arm predictions at L19 ───────────────────────────────────────────
ridge_paths = {
    "779ce": ("issue779_monitoring/n1m_readout/weights/L19/ridge.pt", "ce"),
    "1738ce": ("issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt", "ce"),
    "1738pe": ("issue1738_multiturn/analysis_tensors/weights/L19/prefix_ridge.pt", "pe"),
}
dev = torch.device("cpu")
preds: dict[str, np.ndarray] = {}
for arm, (path, slot) in ridge_paths.items():
    f = retry_transient(
        lambda path=path: hf_hub_download(
            REPO, path, repo_type="dataset", revision=PIN, local_dir=STAGE
        ),
        what=f"hf_hub_download {path}",
    )
    pl = torch.load(f, map_location="cpu", weights_only=False)
    assert pl.get("kind") == "ridge"
    preds[arm] = FITS.apply_map(pl, vc[slot], dev)
    del pl


def afc(targets: np.ndarray, valid: np.ndarray) -> dict:
    """Pooled + per-cell 2AFC (cosine, both directions) per arm at L19 tail."""
    out: dict = {}
    arm_preds = dict(preds)
    for slot in ("ce", "pe"):
        arm_preds[f"idbias_{slot}"] = idbias_loto_predict(vc[slot], targets, pt.cell_of, valid)
    for arm, parr in arm_preds.items():
        slot = "pe" if arm.endswith("pe") else "ce"
        pooled_corr, pooled_n = 0, 0
        per_cell = {}
        for cell in pt.cells:
            if slot == "pe" and cell in degenerate_pe:
                continue
            cv = views[cell]
            s = sim_blocks(parr[cv.ctx_rows], targets[cv.ctx_rows])["cosine"]
            a_loc, b_loc = cv.a_loc, cv.b_loc
            keep = valid[cv.ctx_rows][a_loc] & valid[cv.ctx_rows][b_loc]
            ma, mb = observed_2afc(s, a_loc[keep], b_loc[keep])
            corr = int((ma > 0).sum() + (mb > 0).sum())
            n = 2 * int(keep.sum())
            per_cell[cell] = corr / n if n else float("nan")
            pooled_corr += corr
            pooled_n += n
        out[arm] = {"pooled": pooled_corr / pooled_n, "n_dirs": pooled_n, "per_cell": per_cell}
    return out


res_all = afc(mean_all, n_all > 0)
res_cln = afc(mean_cln, n_cln > 0)
committed = {
    "779ce": 0.7670940170940171,
    "1738ce": 0.7742165242165242,
    "1738pe": 0.646021021021021,
    "idbias_ce": 0.7581908831908832,
    "idbias_pe": 0.5945945945945946,
}
report = {
    "n_intruded_draws": len(flags),
    "n_contexts_zero_clean": n_zero_clean,
    "validation_all_draws": {
        a: {"recount": res_all[a]["pooled"], "committed": committed[a]} for a in committed
    },
    "clean_only": {a: res_cln[a]["pooled"] for a in committed},
    "per_cell_delta_779ce": {
        c: {
            "all": res_all["779ce"]["per_cell"][c],
            "clean": res_cln["779ce"]["per_cell"][c],
        }
        for c in pt.cells
    },
    "per_cell_delta_1738pe": {
        c: {
            "all": res_all["1738pe"]["per_cell"].get(c),
            "clean": res_cln["1738pe"]["per_cell"].get(c),
        }
        for c in pt.cells
        if c not in degenerate_pe
    },
}
json.dump(report, open("/tmp/issue2215_cjk_recount_report.json", "w"), indent=1)
for a in committed:
    v = res_all[a]["pooled"]
    print(
        f"[recount] {a}: all={v:.4f} (committed {committed[a]:.4f}, "
        f"delta {v - committed[a]:+.2e}) clean={res_cln[a]['pooled']:.4f}"
    )
print("[recount] report written")
