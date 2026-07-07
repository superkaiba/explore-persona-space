"""Issue #825 `base-separator-control` Phase C2: base-vs-instruct separator maps.

Thin driver over the committed ``issue825_crossmodel_map_transfer`` battery
(pin 4d03165dd8 — verbatim ridge core + map-swap / representation-swap /
weight-space functions), applied to the TWO armC separator stores (plan v18
section 3 C2):

  - align the base (``issue825_base_sep_control/analysis_tensors/armC``) and
    instruct (``issue931_story_map/analysis_tensors/armC``) stores on
    ``row_id`` (EXACT set equality hard-asserted — same pinned pairs by
    construction), folds blocked at the WikiText ARTICLE group (the #931 fold
    discipline);
  - within-model baselines + ``frozen_map_swap`` both directions +
    cross-model representation-swap fits + ``weight_space_compare``
    (``fit_primal_beta`` cosine + ``principal_angles``) at the frozen layers
    (SVD/Procrustes at L19).

Output: ``<out>/base_sep_similarity.json``. Smoke: ``--base-local-dir`` /
``--instruct-local-dir`` inject tiny local stores through the SAME per-shard
slicing + alignment + battery code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps bind before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import issue825_crossmodel_map_transfer as cmx  # noqa: E402
import issue931_common as common  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

SCRIPT = "scripts/issue825_base_sep_similarity.py"
BASE_STORE_PREFIX = "issue825_base_sep_control/analysis_tensors/armC"
INSTRUCT_STORE_PREFIX = "issue931_story_map/analysis_tensors/armC"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage", type=Path, default=Path("/mnt/eps-data/thomasjiralerspong/i825_base_sep_stage")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("eval_results/issue_825/base-separator-control")
    )
    ap.add_argument("--base-local-dir", type=Path, default=None)
    ap.add_argument("--instruct-local-dir", type=Path, default=None)
    ap.add_argument("--null-draws", type=int, default=cmx.N_NULL_DRAWS)
    ap.add_argument("--seed", type=int, default=common.FIT_SEED)
    return ap.parse_args()


def _slice_shard_frozen(payload: dict, frozen: list[int]) -> dict:
    """One armC shard -> frozen-layer fp32 slices of x_sep + y (+ ids)."""
    return {
        "x_sep": payload["arrays"]["x_sep"].float().numpy()[:, frozen, :],
        "y": payload["arrays"]["y"].float().numpy()[:, frozen, :],
        "row_ids": np.asarray(payload["row_ids"]).astype(str),
        "group_ids": np.asarray(payload["group_ids"]).astype(str),
    }


def load_store_frozen(stage: Path, cache_name: str, prefix: str, local_dir: Path | None) -> dict:
    """Frozen-layer x_sep/y slices of one armC store, streamed shard-by-shard.

    Peak footprint ~ one shard (the r3 streaming shape); L-subset slices are
    npz-cached under ``--stage``. ``local_dir`` reads local shards through the
    SAME slicing function (smoke injection point).
    """
    cache = stage / cache_name
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        print(f"[i825-bs-c2] cache hit: {cache}", flush=True)
        return {
            "x_sep": z["x"],
            "y": z["y"],
            "row_ids": z["r"].astype(str),
            "group_ids": z["g"].astype(str),
            "layers": [int(v) for v in z["layers"]],
            "revision": str(z["revision"]),
        }
    parts = []
    if local_dir is not None:
        revision = "local"
        shards = sorted(local_dir.glob("armC_shard*.pt"))
        assert shards, f"no armC shards under {local_dir}"
        first = torch.load(shards[0], map_location="cpu", weights_only=False)
        n_layers = int(first["arrays"]["y"].shape[1])
        frozen = [li for li in common.FROZEN_LAYERS if li < n_layers] or [n_layers - 1]
        parts.append(_slice_shard_frozen(first, frozen))
        del first
        for sp in shards[1:]:
            payload = torch.load(sp, map_location="cpu", weights_only=False)
            parts.append(_slice_shard_frozen(payload, frozen))
            del payload
    else:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        revision = api.repo_info(common.HF_DATA_REPO, repo_type="dataset").sha
        # Retried first-page listing (the stage.py:111 twin; r2 review minor).
        shard_paths = hub.retry_transient(
            lambda: sorted(
                e.path
                for e in api.list_repo_tree(
                    common.HF_DATA_REPO,
                    path_in_repo=prefix,
                    repo_type="dataset",
                    revision=revision,
                )
                if e.path.endswith(".pt")
            ),
            what=f"list {prefix}",
        )
        assert shard_paths, f"no shards under {prefix}"
        frozen = list(common.FROZEN_LAYERS)
        dest = stage / (cache_name + "_dl")
        for p in shard_paths:
            print(f"[i825-bs-c2] stream {p}", flush=True)
            local = Path(
                hub.retry_transient(
                    lambda p=p: hf_hub_download(
                        common.HF_DATA_REPO,
                        p,
                        repo_type="dataset",
                        revision=revision,
                        local_dir=dest,
                    ),
                    what=f"stage {p}",
                )
            )
            payload = torch.load(local, map_location="cpu", weights_only=False)
            parts.append(_slice_shard_frozen(payload, frozen))
            del payload
            local.unlink()
    out = {
        "x_sep": np.concatenate([p["x_sep"] for p in parts]),
        "y": np.concatenate([p["y"] for p in parts]),
        "row_ids": np.concatenate([p["row_ids"] for p in parts]),
        "group_ids": np.concatenate([p["group_ids"] for p in parts]),
        "layers": frozen,
        "revision": revision,
    }
    assert np.isfinite(out["x_sep"]).all() and np.isfinite(out["y"]).all()
    stage.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        x=out["x_sep"],
        y=out["y"],
        r=out["row_ids"],
        g=out["group_ids"],
        layers=np.asarray(frozen),
        revision=np.asarray(revision),
    )
    return out


def align_stores(instr: dict, base: dict) -> dict:
    """Row-align the two stores on row_id (EXACT set equality — same pins)."""
    set_i, set_b = set(instr["row_ids"].tolist()), set(base["row_ids"].tolist())
    assert set_i == set_b, (
        f"row_id sets differ: only-instruct {sorted(set_i - set_b)[:5]} "
        f"only-base {sorted(set_b - set_i)[:5]} — pinned pairs must yield identical rows"
    )
    assert instr["layers"] == base["layers"], (instr["layers"], base["layers"])
    order = np.asarray(sorted(set_i))
    pos_i = {r: k for k, r in enumerate(instr["row_ids"].tolist())}
    pos_b = {r: k for k, r in enumerate(base["row_ids"].tolist())}
    ia = np.asarray([pos_i[r] for r in order])
    ib = np.asarray([pos_b[r] for r in order])
    g_i = instr["group_ids"][ia]
    g_b = base["group_ids"][ib]
    assert (g_i == g_b).all(), "group_ids disagree on aligned rows"
    return {"ia": ia, "ib": ib, "row_ids": order, "group_ids": g_i}


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.stage.mkdir(parents=True, exist_ok=True)
    instr = load_store_frozen(
        args.stage, "instruct_armC_frozen.npz", INSTRUCT_STORE_PREFIX, args.instruct_local_dir
    )
    base = load_store_frozen(
        args.stage, "base_armC_sim_frozen.npz", BASE_STORE_PREFIX, args.base_local_dir
    )
    al = align_stores(instr, base)
    layers = instr["layers"]
    hl = cmx.HEADLINE_LAYER if cmx.HEADLINE_LAYER in layers else layers[-1]
    Xi, Yi = instr["x_sep"][al["ia"]], instr["y"][al["ia"]]
    Xb, Yb = base["x_sep"][al["ib"]], base["y"][al["ib"]]
    groups = al["group_ids"]  # article-blocked folds (#931 fold discipline)
    n = Xi.shape[0]
    print(f"[i825-bs-c2] aligned n={n} groups={len(np.unique(groups))} layers={layers}")

    kw = dict(seed=args.seed, null_draws=args.null_draws)
    within_i = cmx.frozen_sweep(Xi, Yi, groups, layers, **kw)
    within_b = cmx.frozen_sweep(Xb, Yb, groups, layers, **kw)
    ms_b2i = cmx.frozen_map_swap(Xb, Yb, Xi, Yi, groups, layers, **kw)
    ms_i2b = cmx.frozen_map_swap(Xi, Yi, Xb, Yb, groups, layers, **kw)
    rs_b2i = cmx.frozen_sweep(Xb, Yi, groups, layers, **kw)
    rs_i2b = cmx.frozen_sweep(Xi, Yb, groups, layers, **kw)
    ws = cmx.weight_space_compare(Xi, Yi, Xb, Yb, layers, seed=args.seed, do_svd_layers={hl})

    def _retained(swap_r2: dict, within: dict) -> dict:
        return {
            str(layer): (
                swap_r2[layer] / within["r2_by_layer"][layer]
                if abs(within["r2_by_layer"][layer]) > 1e-9
                else float("nan")
            )
            for layer in swap_r2
        }

    payload = {
        "metadata": common.metadata(
            SCRIPT,
            args.seed,
            n,
            extra={
                "instruct_store_revision": instr["revision"],
                "base_store_revision": base["revision"],
                "null_draws": args.null_draws,
                "fold_blocking": "wikitext article group",
                "headline_layer": hl,
            },
        ),
        "layers": layers,
        "n_aligned": n,
        "n_groups": len(np.unique(groups)),
        "within_model": {
            "instruct": {
                k: within_i[k] for k in ("r2_by_layer", "null_mean_by_layer", "null_p975_by_layer")
            },
            "base": {
                k: within_b[k] for k in ("r2_by_layer", "null_mean_by_layer", "null_p975_by_layer")
            },
        },
        "map_swap": {
            "base_to_instruct": {
                **{
                    k: ms_b2i[k]
                    for k in ("r2_by_layer", "null_mean_by_layer", "null_p975_by_layer")
                },
                "frac_within_target_retained": _retained(ms_b2i["r2_by_layer"], within_i),
            },
            "instruct_to_base": {
                **{
                    k: ms_i2b[k]
                    for k in ("r2_by_layer", "null_mean_by_layer", "null_p975_by_layer")
                },
                "frac_within_target_retained": _retained(ms_i2b["r2_by_layer"], within_b),
            },
        },
        "representation_swap": {
            "base_rep_to_instruct_target": {
                k: rs_b2i[k] for k in ("r2_by_layer", "null_mean_by_layer", "null_p975_by_layer")
            },
            "instruct_rep_to_base_target": {
                k: rs_i2b[k] for k in ("r2_by_layer", "null_mean_by_layer", "null_p975_by_layer")
            },
        },
        "weight_space": ws,
        "caveats": [
            "Separator-map substrate: X = the anchor-token activation, Y = the following-span "
            "mean, on the pinned #931 WikiText armC pairs (identical rows in both models).",
            "The two residual-stream bases are not a-priori aligned: a low map-swap R^2 "
            "confounds map difference with output-basis mismatch; representation-swap re-fits "
            "the read-out into the target basis (the committed crossmodel-map-transfer framing).",
            "Instruct within-model here refits on the ALIGNED row set with article-blocked "
            "folds — the committed #931 cells_armC_sep.json is the canonical within reference.",
            "Descriptive geometry on a single seed; no mechanism claims.",
        ],
    }
    common.write_json(args.out / "base_sep_similarity.json", payload)
    print(
        f"[i825-bs-c2] done: within(base) L{hl}={within_b['r2_by_layer'][hl]:.4f} "
        f"map-swap b->i L{hl}={ms_b2i['r2_by_layer'][hl]:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
