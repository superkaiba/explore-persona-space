"""Issue #931 revision-3 free analysis: SOURCE-MATCHED (n=1982, seed 931) sep->chat control.

The revision-2 sep->chat control applied the run's saved FULL-n (3,600) separator map to
the pinned #825 chat store. The r2 interp-critic union flagged the source-n advantage
(3,600 vs the novel->chat source's 1,982) as anti-conservative for the affirmative
"below its persona-free control" comparison. This script refits the separator map on
the run's own seeded (seed 931) group-stratified n=1,982 subsample
(``issue931_common.group_stratified_subsample`` — the same machinery the committed
power-matched transfer rows used) and computes its recentered sep->chat transfer under
the identical protocol (5 group folds, seed 0, 20 group-blocked pairing-permutation
nulls; ``issue931_similarity.transfer_r2``).

Validation twin (streamed-target check): the saved fp16 full-n map
(``maps/armC_sep_L19.pt`` @ the pinned maps revision) applied to the SAME streamed chat
store must reproduce the committed r2 control row (+0.0731533) — proving the streamed
chat row set + fold assignment match the r2 pipeline. A second internal check compares
the fp64 full-n REFIT's transfer against the fp16-map value (fp16-storage rounding
only).

Compute character: closed-form Gram-dual GCV ridge (one eigh per fit — 3600^2 and
1982^2, seconds each on CPU); the transfer's null battery reuses ONE base prediction
per fold (Y-independent), so the 20 draws add only offset recomputes. 0 GPU-h.

Outputs: extends ``eval_results/issue_931/sep_to_chat_control.json`` in place with a
``sep_to_chat_matched1982`` block (existing r2 blocks preserved).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps bind before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import issue931_common as common  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402
from issue931_similarity import _chat_power_ceiling, fit_full_map, transfer_r2  # noqa: E402

STAGE = Path("/mnt/eps-data/thomasjiralerspong/i931_r3_stage")
OUT_DIR = SCRIPTS.parent / "eval_results" / "issue_931"
OUT_JSON = OUT_DIR / "sep_to_chat_control.json"
CHAT_REV = "82d3a875ee5148e45df982fd51a3c4dea1055fb7"
MAPS_REV = "a23e79f17f053c58e7ce1bb16dff9bac30e55729"
LAYER = 19
N_STAR = 1982
R2_COMMITTED_FULLN_CONTROL = 0.0731533298226521  # r2-round sep->chat (saved fp16 map)


def _download(path_in_repo: str, revision: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    got = hf_hub_download(
        common.HF_DATA_REPO,
        path_in_repo,
        repo_type="dataset",
        revision=revision,
        local_dir=dest_dir,
    )
    return Path(got)


def stage_and_load_armC() -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Download the armC store shards, return (X_sep19, Y19, group_ids, revision)."""
    from huggingface_hub import HfApi

    cache = STAGE / f"armC_sep_L{LAYER}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        print(f"[i931-r3] armC L{LAYER} cache hit: {cache}", flush=True)
        return z["x"], z["y"], z["g"].astype(str), str(z["revision"])

    api = HfApi()
    revision = api.repo_info(common.HF_DATA_REPO, repo_type="dataset").sha
    prefix = "issue931_story_map/analysis_tensors/armC"
    entries = [
        e.path
        for e in api.list_repo_tree(
            common.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", revision=revision
        )
        if e.path.endswith(".pt")
    ]
    assert entries, f"no armC shards under {prefix}"
    dest = STAGE / "armC_dl"
    for p in sorted(entries):
        print(f"[i931-r3] fetch {p}", flush=True)
        _download(p, revision, dest)
    store = fit931.load_regime_store(dest / prefix, "armC")
    x = store["arrays"]["x_sep"][:, LAYER, :].copy()
    y = store["arrays"]["y"][:, LAYER, :].copy()
    g = store["group_ids"].copy()
    del store
    shutil.rmtree(dest)
    np.savez(cache, x=x, y=y, g=g, revision=np.asarray(revision))
    return x, y, g, revision


def stream_chat_layer() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream the pinned #825 Track-S chat shards; return (X19, Y19, conv_ids).

    Replicates ``issue825_fit_cells._load_bundle_pt`` + ``_cell_xy`` for Track-S cell
    S1 — ``_normalize_cell``: slot_index 0, target_turn_index 1 (assistant slot -> a1
    profile) — INCLUDING the all-layer NaN keep-mask, computed per shard before
    slicing L19 so the kept row set matches the run. Peak disk = one shard. The L19
    slices are cached to an npz so a downstream crash never re-streams the 43 GB.
    """
    from huggingface_hub import HfApi

    cache = STAGE / f"chat_L{LAYER}_slot0_turn1.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        print(f"[i931-r3] chat L{LAYER} cache hit: {cache}", flush=True)
        return z["x"], z["y"], z["ids"].astype(str)

    api = HfApi()
    shard_paths = sorted(
        e.path
        for e in api.list_repo_tree(
            common.HF_DATA_REPO,
            path_in_repo=common.CHAT_STORE_PREFIX,
            repo_type="dataset",
            revision=CHAT_REV,
        )
        if Path(e.path).name.startswith(common.CHAT_STORE_STEM + "_shard")
        and e.path.endswith(".pt")
    )
    # 20 store files at the pinned revision = 10 .pt shards + 10 .json sidecars.
    assert len(shard_paths) == 10, f"expected 10 chat .pt shards, got {len(shard_paths)}"
    xs, ys, ids = [], [], []
    dest = STAGE / "chat_dl"
    for p in shard_paths:
        print(f"[i931-r3] stream {p}", flush=True)
        local = _download(p, CHAT_REV, dest)
        payload = torch.load(local, map_location="cpu", weights_only=False)
        conv_ids = np.asarray(payload["conv_ids"]).astype(str)
        # Shards store bf16 tensors — route through torch .float() (numpy has no bf16).
        slots = np.stack([torch.as_tensor(t).float().numpy() for t in payload["slots"]])
        profiles = np.stack([torch.as_tensor(t).float().numpy() for t in payload["profiles"]])
        x_full = slots[:, 0, :, :]  # (n, L, D) — S1 slot_index 0
        y_full = profiles[:, 1, :, :]  # S1 target_turn_index 1 (a1 profile, _normalize_cell)
        keep = ~(np.isnan(x_full).any(axis=(1, 2)) | np.isnan(y_full).any(axis=(1, 2)))
        xs.append(x_full[keep][:, LAYER, :].copy())
        ys.append(y_full[keep][:, LAYER, :].copy())
        ids.append(conv_ids[keep])
        del payload, slots, profiles, x_full, y_full
        local.unlink()
    shutil.rmtree(dest, ignore_errors=True)
    x, y, cid = np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)
    np.savez(cache, x=x, y=y, ids=cid)
    return x, y, cid


def saved_map_fmap() -> dict:
    """Load the run's saved fp16 full-n separator map in the W_raw gauge."""
    local = _download(
        f"issue931_story_map/analysis_tensors/maps/armC_sep_L{LAYER}.pt", MAPS_REV, STAGE / "maps"
    )
    payload = torch.load(local, map_location="cpu", weights_only=False)
    w_raw = payload["W_raw_fp16"].double()
    d = w_raw.shape[0]
    return {
        "W_std": w_raw,
        "xsd": torch.ones(d, dtype=torch.float64),
        "xmu": torch.zeros(d, dtype=torch.float64),
        "ymu": torch.zeros(w_raw.shape[1], dtype=torch.float64),
    }


def main() -> int:
    STAGE.mkdir(parents=True, exist_ok=True)
    x_sep, y_sep, g_sep, armc_rev = stage_and_load_armC()
    assert x_sep.shape[0] == 3600, x_sep.shape
    x_chat, y_chat, g_chat = stream_chat_layer()
    assert x_chat.shape[0] == 5000, x_chat.shape

    kw = dict(application="recentered", folds=common.N_FOLDS, seed=common.FIT_SEED)

    # Validation twin 1: saved fp16 full-n map on the streamed target reproduces r2.
    val = transfer_r2(saved_map_fmap(), x_chat, y_chat, g_chat, n_nulls=0, **kw)
    d_val = abs(val["r2"] - R2_COMMITTED_FULLN_CONTROL)
    print(f"[i931-r3] validation fp16 map -> chat: {val['r2']:.6f} (delta {d_val:.2e})", flush=True)
    assert d_val < 1e-4, (val["r2"], R2_COMMITTED_FULLN_CONTROL)

    # Validation twin 2: fp64 full-n refit vs the fp16-map value (rounding only).
    fmap_full = fit_full_map(x_sep, y_sep)
    full_refit = transfer_r2(fmap_full, x_chat, y_chat, g_chat, n_nulls=0, **kw)
    print(f"[i931-r3] fp64 full-n refit -> chat: {full_refit['r2']:.6f}", flush=True)
    del fmap_full

    # The matched control: seeded (seed 931) group-stratified n=1982 subsample refit.
    idx = common.group_stratified_subsample(g_sep, N_STAR, seed=common.BUILD_SEED)
    assert len(idx) == N_STAR, len(idx)
    fmap_sub = fit_full_map(x_sep[idx], y_sep[idx])
    matched = transfer_r2(fmap_sub, x_chat, y_chat, g_chat, n_nulls=20, **kw)
    ceiling, n_used = _chat_power_ceiling(OUT_DIR, N_STAR, LAYER)
    print(
        f"[i931-r3] matched (n=1982) sep->chat: {matched['r2']:.6f} "
        f"null_p975 {matched['null_p975']:.6f} fraction {matched['r2'] / ceiling:.6f}",
        flush=True,
    )

    doc = json.loads(OUT_JSON.read_text())
    doc["sep_to_chat_matched1982"] = {
        "r2": matched["r2"],
        "null_p975": matched["null_p975"],
        "null_r2": matched["null_r2"],
        "n_train_source": N_STAR,
        "groups_train_source": len(np.unique(g_sep[idx])),
        "subsample": "group_stratified_subsample(seed=931) — the run's matched-power machinery",
        "n_target": int(x_chat.shape[0]),
        "layer": LAYER,
        "application": "recentered",
        "folds": common.N_FOLDS,
        "seed": common.FIT_SEED,
        "fraction_of_matched1982_ceiling": matched["r2"] / ceiling,
        "matched1982_ceiling_r2": ceiling,
        "matched1982_ceiling_n": n_used,
        "validation_saved_fp16_map_reproduction": {
            "r2": val["r2"],
            "expected_committed": R2_COMMITTED_FULLN_CONTROL,
            "abs_delta": d_val,
        },
        "fulln_fp64_refit_r2": full_refit["r2"],
        "armc_store_revision": armc_rev,
        "chat_store_revision": CHAT_REV,
        "maps_revision": MAPS_REV,
        "script": "scripts/issue931_sep_to_chat_matched_control.py",
    }
    doc["metadata"]["note_r3"] = (
        "revision-3 source-matched control: the r2 full-n (3600) control over-trains the "
        "separator source relative to the novel->chat source (1982); sep_to_chat_matched1982 "
        "refits on the seeded n=1982 group-stratified subsample for the like-for-like read."
    )
    OUT_JSON.write_text(json.dumps(doc, indent=2))
    print(f"[i931-r3] wrote {OUT_JSON}", flush=True)
    shutil.rmtree(STAGE, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
