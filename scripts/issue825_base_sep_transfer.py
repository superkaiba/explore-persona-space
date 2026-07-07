"""Issue #825 `base-separator-control` Phase C1: base sep->chat recentered transfer.

Mirrors ``issue931_sep_to_chat_matched_control.py`` with the BASE (pretrained)
S2 chat target (plan v18 section 3 C1):

  1. Stream the pinned #825 pretrained Track-S chat store
     (``pretrained_chat_s_shard*`` @ ``CHAT_REV``) one shard at a time —
     S2 cell semantics (``_normalize_cell({"cell_id": "S2", "model":
     "pretrained"})``: slot 0 -> a1 profile) INCLUDING the all-layer NaN
     keep-mask per shard; L19 slices npz-cached under ``--stage``.
  2. C1 STREAM-VALIDATION GATE (required before any transfer number is read):
     refit the base chat within-map from the streamed rows at L19 and
     reproduce the COMMITTED ``eval_results/issue_825/cells_S2.json`` L19
     value within +-0.01 (plan section 6 gate 4). Binding in production;
     ``--smoke`` records it non-bindingly + self-tests the gate mechanics on
     a planted mismatch.
  3. Full-n (3,600) fp64 ``fit_full_map`` on base x_sep -> y @ L19 ->
     ``transfer_r2`` onto the base chat store (recentered, 5 group folds seed
     0, 20 pairing-permutation nulls). Fractions vs (a) the committed full-n
     base chat ceiling 0.5877 and (b) matched-1982 base chat ceilings via the
     seeded uniform row draw (scheme ``issue931_pcms.seeded_uniform_row_draw
     .v1``): the seed-931 single draw AND the 5-draw (931..935) mean are BOTH
     computed and recorded; ``draw_convention_used`` switches to the 5-draw
     mean when the single-draw fraction lands within +-0.1 of the 0.5
     threshold (plan section 6 near-threshold guard).
  4. Matched-1982 separator-map refit transfer
     (``group_stratified_subsample(seed=931)`` over the 600 article groups) —
     reported with the ESTIMATOR-REGIME framing only (instruct analogue
     -4.049), never as a standalone specificity read.

Also saves the base separator maps (W_raw fp16, frozen layers — the #931
``--save-maps`` convention) and uploads them to
``issue825_base_sep_control/analysis_tensors/maps/`` unless ``--skip-upload``.

The output JSON carries a ``decision_support`` block (both within-strength
estimators' ratios + the instruct reference values + the transfer-leg inputs)
so the analyzer applies the plan-section-6 bands without recomputation.

CLI (plan section 5):
  uv run python scripts/issue825_base_sep_transfer.py \
      --stage /mnt/eps-data/thomasjiralerspong/i825_base_sep_stage \
      --out eval_results/issue_825/base-separator-control
Smoke: --armc-local-dir/--chat-local-dir inject tiny local stores through the
REAL loaders/fit cores; --smoke makes the C1 gate non-binding; --skip-upload.

`onpolicy-separator-control` SOURCE-side generalization (plan section 2 —
defaults preserve the round-6 behavior byte-for-byte): ``--model
{base,instruct}`` selects the CHAT target stream (pretrained_chat_s /
instruct_chat_s, both @ CHAT_REV) + the committed full-n ceiling the C1 gate
reproduces (cells_S2.json 0.58768 / cells_chat_ref.json 0.67309);
``--source-store-dir`` (local) or ``--source-store-prefix`` (Hub) points the
SOURCE separator store at the on-policy armC store instead of the round-6
base exogenous store (realized-n; the 3600 run invariant applies only to the
round-6 default source); ``--out-name`` / ``--hf-prefix-out`` route the
outputs (e.g. onpolicy_sep_to_chat_base.json / issue825_onpolicy_sep_control).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps bind before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402
import issue931_power_curve_multi_seed as pcms  # noqa: E402
from issue931_similarity import fit_full_map, transfer_r2  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

SCRIPT = "scripts/issue825_base_sep_transfer.py"
CHAT_REV = "a23e79f17f053c58e7ce1bb16dff9bac30e55729"  # plan section 5 pin
CHAT_PREFIX = "issue825_userbase_map/analysis_tensors"
CHAT_STEMS = {"base": "pretrained_chat_s", "instruct": "instruct_chat_s"}
CHAT_STEM = CHAT_STEMS["base"]  # round-6 default
BASE_STORE_PREFIX = "issue825_base_sep_control/analysis_tensors/armC"
HF_PREFIX_OUT = "issue825_base_sep_control"
LAYER = common.HEADLINE_LAYER  # 19
# Committed full-n chat ceilings @ L19 per model (drift-guarded at run time).
COMMITTED_CHAT = {
    "base": ("eval_results/issue_825/cells_S2.json", 0.5876803039140281),
    "instruct": ("eval_results/issue_931/cells_chat_ref.json", 0.6730940896676356),
}
N_STAR = 1982
CEILING_SEEDS = (931, 932, 933, 934, 935)  # seed-931 single draw is the convention anchor
GATE_TOL = 0.01
# Committed references (drift-guarded against the JSONs at run time).
INSTRUCT_REF = {
    "chat_ceiling_L19": 0.6730940896676356,  # eval_results/issue_931/cells_chat_ref.json
    "sep_rotated_L19": 0.3489193821633685,
    "sep_mlp_L19": 0.2985925806396439,
    "sep_to_chat_fulln_r2": 0.0731533298226521,  # eval_results/issue_931/sep_to_chat_control.json
    "fraction_of_matched1982_ceiling": 0.23135145421458606,
    "fraction_of_fulln_ceiling": 0.10868122095179335,
    "matched1982_refit_r2": -4.048875040626637,  # estimator-regime at reduced n
}
INSTRUCT_RATIO_ROTATED = INSTRUCT_REF["sep_rotated_L19"] / INSTRUCT_REF["chat_ceiling_L19"]
INSTRUCT_RATIO_MLP = INSTRUCT_REF["sep_mlp_L19"] / INSTRUCT_REF["chat_ceiling_L19"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage", type=Path, default=Path("/mnt/eps-data/thomasjiralerspong/i825_base_sep_stage")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("eval_results/issue_825/base-separator-control")
    )
    ap.add_argument("--fit-dir", type=Path, default=None, help="base fit JSONs (default --out)")
    ap.add_argument("--armc-local-dir", type=Path, default=None, help="local base armC store dir")
    ap.add_argument("--chat-local-dir", type=Path, default=None, help="local chat shard dir")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="C1 gate recorded, not binding")
    # onpolicy-separator-control generalization (defaults = round-6 behavior).
    ap.add_argument(
        "--model",
        choices=("base", "instruct"),
        default="base",
        help="chat target + committed ceiling (default base = round-6)",
    )
    ap.add_argument(
        "--source-store-dir",
        type=Path,
        default=None,
        help="LOCAL source separator store dir (e.g. the on-policy armC store); "
        "default None = the round-6 base exogenous source",
    )
    ap.add_argument(
        "--source-store-prefix",
        type=str,
        default=None,
        help="Hub prefix for the source separator store (e.g. "
        "issue825_onpolicy_sep_control/analysis_tensors/armC_base); default None "
        "= the round-6 BASE_STORE_PREFIX",
    )
    ap.add_argument("--out-name", type=str, default="base_sep_to_chat.json")
    ap.add_argument("--hf-prefix-out", type=str, default=HF_PREFIX_OUT)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Store loading (streamed; caches under --stage)
# ---------------------------------------------------------------------------


def load_base_armc(
    stage: Path,
    local_dir: Path | None,
    *,
    store_prefix: str = BASE_STORE_PREFIX,
    cache_name: str = "base_armC_frozen.npz",
    expect_n: int | None = 3600,
) -> dict:
    """Source armC store -> {x_sep/y at FROZEN_LAYERS (fp32), group_ids, row_ids, rev}.

    Production stages the pod-uploaded store from ``store_prefix`` (one shard
    at a time, deleted after slicing); ``local_dir`` reads local shards
    through the SAME consumer loader (``fit931.load_regime_store``). Defaults
    are the round-6 base exogenous source (n == 3600 run invariant); the
    on-policy sources pass their own prefix/dir + ``expect_n=None``
    (realized-n, reported not asserted).
    """
    frozen = list(common.FROZEN_LAYERS)
    cache = stage / cache_name
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if "frozen" not in z.files:
            # Pre-r2 cache format (no layer-list metadata): rebuild rather than
            # guess which layers the cached slices actually hold.
            print(f"[i825-bs-c1] base armC cache lacks 'frozen' metadata — rebuild: {cache}")
            cache.unlink()
        else:
            x, y = z["x"], z["y"]
            rev = str(z["revision"])
            cached_frozen = [int(v) for v in z["frozen"]]
            # Cache-hit twins of the fresh-download asserts (r2 review minor):
            # a smoke ("local") cache must never feed a production run, and a
            # production cache must carry the full run invariants.
            assert (rev == "local") == (local_dir is not None), (
                f"cache/run mode mismatch: cached revision {rev!r} vs "
                f"local_dir={local_dir} — stale cache at {cache}"
            )
            if rev != "local":
                if expect_n is not None:
                    assert x.shape[0] == expect_n, (x.shape, expect_n, cache)
                assert cached_frozen == frozen, (cached_frozen, frozen, cache)
            assert x.shape[1] == len(cached_frozen), (x.shape, cached_frozen)
            assert y.shape[:2] == x.shape[:2], (x.shape, y.shape)
            assert np.isfinite(x).all() and np.isfinite(y).all()
            print(f"[i825-bs-c1] base armC cache hit (validated): {cache}", flush=True)
            return {
                "x_sep": x,
                "y": y,
                "group_ids": z["g"].astype(str),
                "row_ids": z["r"].astype(str),
                "revision": rev,
                "frozen": cached_frozen,
            }
    if local_dir is not None:
        store = fit931.load_regime_store(local_dir, "armC")
        revision = "local"
        frozen = [li for li in frozen if li < store["arrays"]["y"].shape[1]] or [
            store["arrays"]["y"].shape[1] - 1
        ]
        x = store["arrays"]["x_sep"][:, frozen, :].astype(np.float32)
        y = store["arrays"]["y"][:, frozen, :].astype(np.float32)
        g, r = store["group_ids"].astype(str), store["row_ids"].astype(str)
    else:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        revision = api.repo_info(common.HF_DATA_REPO, repo_type="dataset").sha
        # First-page tree listing rides the same transient-5xx/429 retry as the
        # per-shard downloads below (the stage.py:111 twin; r2 review minor).
        shard_paths = hub.retry_transient(
            lambda: sorted(
                e.path
                for e in api.list_repo_tree(
                    common.HF_DATA_REPO,
                    path_in_repo=store_prefix,
                    repo_type="dataset",
                    revision=revision,
                )
                if e.path.endswith(".pt")
            ),
            what=f"list {store_prefix}",
        )
        assert shard_paths, f"no source armC shards under {store_prefix}"
        xs, ys, gs, rs = [], [], [], []
        dest = stage / "base_armC_dl"
        for p in shard_paths:
            print(f"[i825-bs-c1] stream {p}", flush=True)
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
            xs.append(payload["arrays"]["x_sep"].float().numpy()[:, frozen, :])
            ys.append(payload["arrays"]["y"].float().numpy()[:, frozen, :])
            gs.extend(payload["group_ids"])
            rs.extend(payload["row_ids"])
            del payload
            local.unlink()
        x, y = np.concatenate(xs), np.concatenate(ys)
        g, r = np.asarray(gs).astype(str), np.asarray(rs).astype(str)
        if expect_n is not None:  # round-6 run invariant (default source only)
            assert x.shape[0] == expect_n, (x.shape, expect_n)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    stage.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        x=x,
        y=y,
        g=g,
        r=r,
        revision=np.asarray(revision),
        frozen=np.asarray(frozen, dtype=np.int64),
    )
    return {
        "x_sep": x,
        "y": y,
        "group_ids": g,
        "row_ids": r,
        "revision": revision,
        "frozen": frozen,
    }


def _s2_slices_from_payload(payload: dict, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One chat shard's (X_L, Y_L, conv_ids) for the S2 cell, keep-masked.

    S2 = ``_normalize_cell({"cell_id": "S2", "model": "pretrained"})``: slot 0
    -> turn-1 (a1) profile; bf16 -> fp32 via torch ``.float()``; the ALL-LAYER
    NaN keep-mask is computed BEFORE slicing the layer so the kept row set
    matches the committed ``_cell_xy`` chain (the r3 streaming shape).
    """
    conv_ids = np.asarray(payload["conv_ids"]).astype(str)
    slots = np.stack([torch.as_tensor(t).float().numpy() for t in payload["slots"]])
    profiles = np.stack([torch.as_tensor(t).float().numpy() for t in payload["profiles"]])
    x_full = slots[:, 0, :, :]  # S2 slot_index 0
    y_full = profiles[:, 1, :, :]  # S2 target_turn_index 1 (a1 profile)
    keep = ~(np.isnan(x_full).any(axis=(1, 2)) | np.isnan(y_full).any(axis=(1, 2)))
    li = min(layer, x_full.shape[1] - 1)
    return x_full[keep][:, li, :].copy(), y_full[keep][:, li, :].copy(), conv_ids[keep]


def stream_chat_layer(
    stage: Path, local_dir: Path | None, *, stem: str = CHAT_STEM
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream the pinned chat shards for one model; return (X19, Y19, conv_ids).

    Default stem = the round-6 pretrained target (cache name unchanged); the
    instruct target streams ``instruct_chat_s`` at the SAME pinned CHAT_REV
    into its own cache file.
    """
    cache_stem = "base_chat" if stem == CHAT_STEMS["base"] else "instruct_chat"
    cache = stage / f"{cache_stem}_L{LAYER}_slot0_turn1.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if "revision" not in z.files:
            print(f"[i825-bs-c1] base chat cache lacks 'revision' metadata — rebuild: {cache}")
            cache.unlink()
        else:
            rev = str(z["revision"])
            x, y, cid = z["x"], z["y"], z["ids"].astype(str)
            # Cache-hit twins of the fresh-download asserts (r2 review minor).
            assert (rev == "local") == (local_dir is not None), (
                f"cache/run mode mismatch: cached revision {rev!r} vs "
                f"local_dir={local_dir} — stale cache at {cache}"
            )
            if rev != "local":
                assert rev == CHAT_REV, (rev, CHAT_REV, cache)
                assert x.shape[0] == 5000, (x.shape, cache)
            assert x.shape[0] == y.shape[0] == cid.shape[0], (x.shape, y.shape, cid.shape)
            assert np.isfinite(x).all() and np.isfinite(y).all()
            print(f"[i825-bs-c1] base chat cache hit (validated): {cache}", flush=True)
            return x, y, cid
    xs, ys, ids = [], [], []
    if local_dir is not None:
        shards = sorted(local_dir.glob(f"{stem}_shard*.pt"))
        assert shards, f"no local chat shards ({stem}) under {local_dir}"
        for sp in shards:
            payload = torch.load(sp, map_location="cpu", weights_only=False)
            x, y, cid = _s2_slices_from_payload(payload, LAYER)
            xs.append(x), ys.append(y), ids.append(cid)
            del payload
    else:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        # Retried first-page listing (the stage.py:111 twin; r2 review minor).
        shard_paths = hub.retry_transient(
            lambda: sorted(
                e.path
                for e in api.list_repo_tree(
                    common.HF_DATA_REPO,
                    path_in_repo=CHAT_PREFIX,
                    repo_type="dataset",
                    revision=CHAT_REV,
                )
                if Path(e.path).name.startswith(stem + "_shard") and e.path.endswith(".pt")
            ),
            what=f"list {CHAT_PREFIX}",
        )
        assert len(shard_paths) == 10, f"expected 10 chat .pt shards, got {len(shard_paths)}"
        dest = stage / "base_chat_dl"
        for p in shard_paths:
            print(f"[i825-bs-c1] stream {p}", flush=True)
            local = Path(
                hub.retry_transient(
                    lambda p=p: hf_hub_download(
                        common.HF_DATA_REPO,
                        p,
                        repo_type="dataset",
                        revision=CHAT_REV,
                        local_dir=dest,
                    ),
                    what=f"stage {p}",
                )
            )
            payload = torch.load(local, map_location="cpu", weights_only=False)
            x, y, cid = _s2_slices_from_payload(payload, LAYER)
            xs.append(x), ys.append(y), ids.append(cid)
            del payload
            local.unlink()
        import shutil

        shutil.rmtree(dest, ignore_errors=True)
    x, y, cid = np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)
    if local_dir is None:
        assert x.shape[0] == 5000, x.shape  # run invariant (production path only)
    stage.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        x=x,
        y=y,
        ids=cid,
        revision=np.asarray("local" if local_dir is not None else CHAT_REV),
    )
    return x, y, cid


# ---------------------------------------------------------------------------
# Fits (the committed single-layer refit shape) + the C1 gate
# ---------------------------------------------------------------------------


def refit_l19(x: np.ndarray, y: np.ndarray, ids: np.ndarray) -> float:
    """Held-out within-map R^2 at one layer through the COMMITTED sweep code
    (``fit825.heldout_r2_sweep`` on a singleton layer axis: same folds, same
    cached-eigh Gram ridge, same pooled R^2 as the committed 28-layer run)."""
    sw = fit825.heldout_r2_sweep(
        x[:, None, :],
        y[:, None, :],
        ids,
        n_folds=common.N_FOLDS,
        seed=common.FIT_SEED,
        null_draws=0,
        collect_cosines=False,
    )
    return float(sw["r2_obs"][0])


def _gate_ok(value: float, committed: float, tol: float = GATE_TOL) -> bool:
    return abs(value - committed) <= tol


def committed_chat_l19(model: str = "base") -> float:
    """Committed full-n chat ceiling @ L19 for one model (drift-guarded)."""
    rel, quote = COMMITTED_CHAT[model]
    d = json.loads((SCRIPTS.parent / rel).read_text())
    v = float(d["r2_per_layer_obs"][LAYER])
    assert abs(v - quote) < 1e-9, (model, v, quote)
    return v


def matched_ceilings(x: np.ndarray, y: np.ndarray, ids: np.ndarray) -> dict:
    """Matched-1982 base chat ceilings: seeded uniform row draws (the
    ``issue931_pcms.seeded_uniform_row_draw.v1`` scheme), seeds 931..935."""
    out = {}
    for s in CEILING_SEEDS:
        idx = pcms.draw_subsample(ids, N_STAR, s)
        assert len(idx) == min(N_STAR, len(ids)), (len(idx), N_STAR)
        out[str(s)] = refit_l19(x[idx], y[idx], ids[idx])
        print(f"[i825-bs-c1] matched ceiling seed={s}: {out[str(s)]:.6f}", flush=True)
    return out


def save_and_upload_maps(
    armc: dict, out: Path, *, skip_upload: bool, model: str = "base", hf_prefix: str = HF_PREFIX_OUT
) -> dict:
    """Source separator W_raw fp16 maps at the frozen layers (#931 --save-maps
    convention) -> <out>/maps/ (+ HF analysis_tensors/maps unless skipped)."""
    maps_dir = out / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    lam_by_layer = {}
    written: list[str] = []
    for i, layer in enumerate(armc["frozen"]):
        fm = fit_full_map(armc["x_sep"][:, i, :], armc["y"][:, i, :])
        w_raw = (fm["W_std"] / fm["xsd"][:, None]).to(torch.float16).cpu()
        name = f"armC_sep_{model}_L{int(layer):02d}.pt"
        torch.save({"W_raw_fp16": w_raw, "layer": int(layer)}, maps_dir / name)
        written.append(name)
        lam_by_layer[str(int(layer))] = float(fm["lam"])
        del fm
    print(f"[i825-bs-c1] saved {len(lam_by_layer)} {model} separator maps -> {maps_dir}")
    if not skip_upload:
        from huggingface_hub import HfApi

        # hub._upload swallows exceptions and returns "" on ANY failure —
        # assert the return AND exact-set-verify the exact map filenames
        # written this run (concern unverified-mirror-maps-uploads).
        dest_prefix = f"{hf_prefix}/analysis_tensors/maps"
        up = hub._upload(
            maps_dir,
            repo_id=common.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=dest_prefix,
        )
        assert up, f"maps upload FAILED (hub._upload returned empty) -> {dest_prefix}"
        expected = sorted(f"{dest_prefix}/{n}" for n in written)
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            common.HF_DATA_REPO,
            expected,
            path_in_repo=dest_prefix,
            repo_type="dataset",
        )
        assert not missing, f"maps upload verify FAILED — missing on Hub: {missing}"
        print(f"[i825-bs-c1] maps upload exact-set verified: {len(expected)} @ {dest_prefix}")
    return lam_by_layer


def decision_support(fit_dir: Path, ceiling_fulln: float, transfer_block: dict) -> dict:
    """Both estimators' within-strength ratios + instruct references + the
    transfer-leg inputs (plan section 6 bands, applied by the ANALYZER)."""

    def _cell(name: str) -> dict:
        d = json.loads((fit_dir / f"cells_{name}.json").read_text())
        hl = int(d.get("headline_layer", LAYER))
        return {
            "ridge": float(d["r2_per_layer_obs"][hl]),
            "rotated": float(d["random_projection_control_r2"][str(hl)]),
            "headline_layer": hl,
        }

    sep = _cell("armC_sep")
    prev = _cell("armC_prevmean")
    mlp_doc = json.loads((fit_dir / "mlp_secondary.json").read_text())
    mlp_cell = mlp_doc["cells"]["armC_sep"]
    mlp = float(mlp_cell[str(sep["headline_layer"])]["r2_obs"])
    ratio_rot = sep["rotated"] / ceiling_fulln
    ratio_mlp = mlp / ceiling_fulln
    r_base = max(ratio_rot, ratio_mlp)
    r_inst = max(INSTRUCT_RATIO_ROTATED, INSTRUCT_RATIO_MLP)
    return {
        "base": {
            "sep_ridge_L19": sep["ridge"],
            "sep_rotated_L19": sep["rotated"],
            "sep_mlp_L19": mlp,
            "prevmean_ridge_L19": prev["ridge"],
            "prevmean_rotated_L19": prev["rotated"],
            "chat_ceiling_fulln_L19": ceiling_fulln,
            "ratio_rotated": ratio_rot,
            "ratio_mlp": ratio_mlp,
            "r_base_max_interpretable": r_base,
        },
        "instruct_reference": {
            **INSTRUCT_REF,
            "ratio_rotated": INSTRUCT_RATIO_ROTATED,
            "ratio_mlp": INSTRUCT_RATIO_MLP,
            "r_inst_max_interpretable": r_inst,
        },
        "within_strength_read": {
            "r_base_minus_r_inst": r_base - r_inst,
            "margin": 0.10,
            "note": (
                "reference-ratio comparison, NOT a binary conjunct (plan v18 section 6); "
                "raw ridge reported, never decision-driving (documented control-cell pathology)"
            ),
        },
        "transfer_leg": {
            **transfer_block,
            "threshold": 0.5,
            "instruct_analogue_matched_fraction": INSTRUCT_REF["fraction_of_matched1982_ceiling"],
            "instruct_analogue_fulln_fraction": INSTRUCT_REF["fraction_of_fulln_ceiling"],
            "note": "BINARY decision leg: UPHELD iff matched fraction < 0.5 (analyzer applies)",
        },
        "prevmean_robustness": {
            "instruct_prevmean_rotated_L19": 0.334114604106466,
            "flag_disagreement_gt": 0.15,
        },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fit_dir = args.fit_dir or args.out
    args.stage.mkdir(parents=True, exist_ok=True)

    source_override = args.source_store_dir is not None or args.source_store_prefix is not None
    armc = load_base_armc(
        args.stage,
        args.source_store_dir or args.armc_local_dir,
        store_prefix=args.source_store_prefix or BASE_STORE_PREFIX,
        cache_name=(
            f"onpolicy_armC_frozen_{args.model}.npz" if source_override else "base_armC_frozen.npz"
        ),
        expect_n=None if source_override else 3600,
    )
    x_chat, y_chat, g_chat = stream_chat_layer(
        args.stage, args.chat_local_dir, stem=CHAT_STEMS[args.model]
    )

    # C1 stream-validation gate (plan section 6 gate 4; per-model committed ceiling).
    committed = committed_chat_l19(args.model)
    chat_within = refit_l19(x_chat, y_chat, g_chat)
    gate_pass = _gate_ok(chat_within, committed)
    print(
        f"[i825-bs-c1] C1 stream gate: refit {chat_within:.6f} vs committed {committed:.6f} "
        f"(pass={gate_pass}, binding={not args.smoke})",
        flush=True,
    )
    if not args.smoke:
        assert gate_pass, (
            f"C1 stream gate FAIL: refit {chat_within} vs committed {committed} — streamed row "
            "set / fold protocol does not match the committed pipeline; no transfer number is read"
        )
    else:
        assert not _gate_ok(committed + 0.5, committed), "gate self-test failed"
        print("[i825-bs-c1] C1 gate self-test PASS: planted +0.5 mismatch detected")

    # Source separator map (frozen layers saved; L19 drives the transfer).
    lam_by_layer = save_and_upload_maps(
        armc, args.out, skip_upload=args.skip_upload, model=args.model, hf_prefix=args.hf_prefix_out
    )
    l19_idx = armc["frozen"].index(LAYER) if LAYER in armc["frozen"] else len(armc["frozen"]) - 1
    x_sep19 = armc["x_sep"][:, l19_idx, :]
    y_sep19 = armc["y"][:, l19_idx, :]

    kw = dict(application="recentered", folds=common.N_FOLDS, seed=common.FIT_SEED)
    fmap_full = fit_full_map(x_sep19, y_sep19)
    full = transfer_r2(fmap_full, x_chat, y_chat, g_chat, n_nulls=20, **kw)
    del fmap_full
    print(f"[i825-bs-c1] full-n base sep->chat: {full['r2']:.6f}", flush=True)

    ceilings = matched_ceilings(x_chat, y_chat, g_chat)
    ceiling_931 = ceilings[str(CEILING_SEEDS[0])]
    ceiling_5draw = float(np.mean([ceilings[str(s)] for s in CEILING_SEEDS]))
    frac_fulln = full["r2"] / committed
    frac_single = full["r2"] / ceiling_931
    frac_5draw = full["r2"] / ceiling_5draw
    near_threshold = abs(frac_single - 0.5) <= 0.1
    convention = "five_draw_mean" if near_threshold else "single_draw_seed931"
    headline_fraction = frac_5draw if near_threshold else frac_single

    # Matched-1982 separator refit (estimator-regime framing only).
    idx = common.group_stratified_subsample(armc["group_ids"], N_STAR, seed=common.BUILD_SEED)
    assert len(idx) == min(N_STAR, len(armc["group_ids"])), len(idx)
    fmap_sub = fit_full_map(x_sep19[idx], y_sep19[idx])
    matched_sep = transfer_r2(fmap_sub, x_chat, y_chat, g_chat, n_nulls=20, **kw)
    del fmap_sub

    transfer_block = {
        "fulln_transfer_r2": full["r2"],
        "fraction_of_matched_ceiling_used": headline_fraction,
        "draw_convention_used": convention,
        "near_threshold_guard_fired": near_threshold,
        "fraction_single_draw_seed931": frac_single,
        "fraction_five_draw_mean": frac_5draw,
        "fraction_of_fulln_ceiling": frac_fulln,
    }
    payload = {
        "metadata": common.metadata(
            SCRIPT,
            common.FIT_SEED,
            int(x_chat.shape[0]),
            extra={
                "chat_store_revision": CHAT_REV,
                "base_armc_store_revision": armc["revision"],
                "subsample_scheme_id": pcms.SUBSAMPLE_SCHEME_ID,
                "smoke": bool(args.smoke),
                # Source-side generalization provenance (absent on the
                # round-6 default path — byte-preserving defaults).
                **(
                    {
                        "source_model": args.model,
                        "source_store_dir": str(args.source_store_dir),
                        "source_store_prefix": args.source_store_prefix,
                        "chat_stem": CHAT_STEMS[args.model],
                    }
                    if source_override
                    else {}
                ),
            },
        ),
        "layer": LAYER,
        "c1_stream_gate": {
            "refit_r2": chat_within,
            "committed_S2_L19": committed,
            "tolerance": GATE_TOL,
            "pass": gate_pass,
            "binding": not args.smoke,
        },
        "sep_to_chat": {
            "r2": full["r2"],
            "null_p975": full["null_p975"],
            "null_r2": full["null_r2"],
            "n_train_source": int(x_sep19.shape[0]),
            "n_target": int(x_chat.shape[0]),
            "application": "recentered",
            "folds": common.N_FOLDS,
            "seed": common.FIT_SEED,
            "fraction_of_fulln_ceiling": frac_fulln,
            "fulln_ceiling_r2": committed,
            "fraction_of_matched1982_ceiling_single_draw": frac_single,
            "matched1982_ceiling_single_draw_seed931": ceiling_931,
            "fraction_of_matched1982_ceiling_five_draw_mean": frac_5draw,
            "matched1982_ceiling_five_draw_mean": ceiling_5draw,
            "matched1982_ceilings_by_seed": ceilings,
            "draw_convention_used": convention,
            "near_threshold_guard_fired": near_threshold,
        },
        "sep_to_chat_matched1982_refit": {
            "r2": matched_sep["r2"],
            "null_p975": matched_sep["null_p975"],
            "null_r2": matched_sep["null_r2"],
            "n_train_source": len(idx),
            "subsample": "group_stratified_subsample(seed=931) over article groups",
            "framing": (
                "ESTIMATOR-REGIME read only (instruct analogue -4.049 at the same reduced n); "
                "never a standalone specificity read (plan v18 section 6)"
            ),
        },
        "base_sep_map_gcv_lambda_by_layer": lam_by_layer,
        "decision_support": decision_support(fit_dir, committed, transfer_block),
    }
    common.write_json(args.out / args.out_name, payload)
    print(
        f"[i825-bs-c1] done: fraction({convention})={headline_fraction:.4f} "
        f"(single {frac_single:.4f} / 5-draw {frac_5draw:.4f} / full-n {frac_fulln:.4f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
