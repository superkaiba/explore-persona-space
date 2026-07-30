#!/usr/bin/env python
"""Issue #1345 story-boundary-ablation — fit battery + verdict lattice.

CPU-only. Five legs, all on the SHARED-factorization ridge core (one
eigh(Gram) per (source, fold), the lambda grid applied as diagonal filter
rescalings of that ONE factorization, and nulls as Y-permutations against the
SAME cached factorization — never a re-solve per lambda / per draw):

  1. cells      per ablation arm x the 4 store slots (prefix / ctx_qend /
                context / ctx_preans) on the arm's OWN store, with the
                registered conversation-grouped folds, shuffle nulls, the
                random-projection + mean baselines, and conversation-level
                bootstrap CIs.
  2. baselines  identity+learned-bias (v_hat = x + b) and kNN retrieval at the
                headline layer for EVERY cell, out-of-fold on the same folds
                (standing rule: both reads accompany every fitted map).
  3. matched    chat (r1) and no-template (r2) comparator refits restricted to
                each arm's kept conversation set, plus a V1-anchor refit on the
                arm-and-V1 intersection when the V1 store is staged — so the
                headline arm-vs-V1 and arm-vs-chat deltas are PAIRED bootstraps
                on a shared conversation set, not two independent CIs.
  4. reparam    story-arm <-> chat in BOTH directions at the headline layer:
                the direct / ctx-reparam (A) / ans-reparam (B) / AMB rungs with
                a matched-capacity shuffle-fit null per rung.
  5. verdict    per-arm lattice: the headline-slot read vs the V1 anchor, vs the
                matched chat comparator, and vs its own null band — plus the
                per-arm README naming what the arm isolates.

Both mapping arms are fit per cell by construction: the `prefix` slot IS the
prefix-arm map (everything before the query) and the three ctx_* slots are
context-arm maps (prefix + query) at different read positions.

CLI:
  uv run python scripts/issue1345_boundary_ablation_fits.py --phase all
  uv run python scripts/issue1345_boundary_ablation_fits.py --phase all --smoke
  uv run python scripts/issue1345_boundary_ablation_fits.py --import-check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue825_fit_cells as fc  # noqa: E402
import issue1345_boundary_ablation_capture as bc  # noqa: E402
import issue1345_boundary_ablation_gen as bg  # noqa: E402
import issue1345_common as c  # noqa: E402
from issue1345_fit_cells import degenerate_fold_reason  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as mb  # noqa: E402

LAYER = c.HEADLINE_LAYER  # 19
N_NULL_DRAWS = 40  # brief: 40 shuffle nulls
N_REPARAM_NULL_DRAWS = 20
SMOKE_NULL_DRAWS = 3
SMOKE_BOOT = 50
KNN_KS = (1, 5, 10)
# Slim load: the fits never read the per-position tensors (~36 GB/bundle stacked).
SLIM_KEYS = ("slots", "profiles", "nll")
# Arms whose story<->chat reparam ladder runs by default (brief: V2 + V4 at
# minimum — V3 rides along because the ladder is cheap once the stores are open).
DEFAULT_REPARAM_ARMS = (bg.ARM_V2, bg.ARM_V3, bg.ARM_V4)

# V1 anchor: the landed conversation_paired_stories_assistant reads. Literals are
# documentation cross-checks; the values are read LIVE from the committed JSONs.
V1_ANCHOR_DIR = Path("eval_results/issue_1345/conversation_paired_stories_assistant")
V1_ANCHOR_FILES = {
    "context": V1_ANCHOR_DIR / "cells_R_instruct_r4_context.json",
    "prefix": V1_ANCHOR_DIR / "cells_R_instruct_r4_prefix.json",
}
V1_MATCHED_CHAT_FILES = {
    "context": V1_ANCHOR_DIR / "matched_row" / "cells_R_instruct_r1_matched_context.json",
    "prefix": V1_ANCHOR_DIR / "matched_row" / "cells_R_instruct_r1_matched_prefix.json",
}
V1_ANCHOR_DOC = {"context": -0.3056, "prefix": -1.3714}
V1_MATCHED_CHAT_DOC = {"context": 0.2426, "prefix": 0.1313}
# The V1 store's own stem + slot order (2 slots: prefix, context).
V1_STEM_FORMAT = "stories_paired"
V1_SLOT_INDEX = {"prefix": 0, "context": 1}

# PARENT comparator stores staged by issue1345_prefetch_reuse.py (2 slots,
# Y_MEAN only) — retained for the V1-PARITY comparator read whose committed
# value this round cross-checks. The X x Y grid comparators are the round's own
# bnd_chat / bnd_ntpl stores below.
COMPARATOR_FORMAT = {"r1_chat": "chat", "r2_no_template": "naturalistic"}
COMPARATOR_TURN_INDEX = 1  # r1/r2 single-turn track-S rows sort [u1, a1]
# Round-own X x Y comparator stores (issue1345_boundary_ablation_capture
# --comparator): same 5 slots + 2 Y targets as every ablation arm.
BND_COMPARATORS = bc.COMPARATORS  # ("chat", "no_template")
# Which store slot each mapping arm reads in the PARENT r1/r2/V1 stores.
MAP_ARM_SLOT = dict(c.ARM_SLOT_INDEX)  # {"prefix": 0, "context": 1}
# Which mapping arm each boundary-store slot belongs to (both arms are covered:
# `prefix` is the prefix-arm map; every ctx_*/x_* slot is a context-arm map).
SLOT_MAP_ARM = {
    "prefix": "prefix",
    "ctx_qend": "context",
    "context": "context",
    "ctx_preans": "context",
    "x_straddle": "context",
}
# Short tags for the two Y targets in cell ids.
Y_TAG = {bc.Y_MEAN: "ymean", bc.Y_BOUNDARY: "ybnd"}


# ---------------------------------------------------------------------------
# Cell registry
# ---------------------------------------------------------------------------
def grid_cell_id(store_key: str, slot: str, y: str) -> str:
    """Cell id for one (store, X slot, Y target) grid point."""
    tag = bg.ARM_SLUG.get(store_key, store_key)
    return f"R_{bg.MODEL_KEY}_bnd_{tag}_{slot}__{Y_TAG[y]}"


def grid_cells(store_key: str) -> list[dict]:
    """The store's full X x Y grid: every BND slot crossed with both Y targets.

    ``store_key`` is an ablation arm (V2/V3/V4) or a round-own comparator
    (``chat`` / ``no_template``) — both carry the identical 5-slot x 2-target
    store shape, which is what makes the grid comparable across them.
    """
    return [
        {
            "cell_id": grid_cell_id(store_key, slot, y),
            "model_key": bg.MODEL_KEY,
            "format_key": bc.format_key(store_key),
            "track": bc.TRACK,
            "slot_index": idx,
            "target_turn_index": bc.Y_TARGET_INDEX[y],
            "regime": bc.format_key(store_key),
            "bnd_arm": store_key,
            "slot": slot,
            "y_target": y,
            "arm": SLOT_MAP_ARM[slot],
        }
        for idx, slot in enumerate(bc.BND_SLOT_ORDER)
        for y in bc.Y_SPAN_ORDER
    ]


def arm_cells(arm: str) -> list[dict]:
    """The ablation arm's own X x Y grid cells."""
    return grid_cells(arm)


def comparator_cells(arm: str, label: str) -> list[dict]:
    """PARENT r1/r2 comparator cells for one arm (both mapping arms), matched-row.

    These are the V1-PARITY reads (2-slot parent stores, Y_MEAN only) whose
    committed values this round cross-checks; the X x Y comparator grid lives on
    the round's own bnd_chat / bnd_ntpl stores (``grid_cells``).
    """
    return [
        {
            "cell_id": f"R_{bg.MODEL_KEY}_{label}_bnd_{bg.ARM_SLUG[arm]}_{map_arm}",
            "model_key": bg.MODEL_KEY,
            "format_key": COMPARATOR_FORMAT[label],
            "track": bc.TRACK,
            "slot_index": MAP_ARM_SLOT[map_arm],
            "target_turn_index": COMPARATOR_TURN_INDEX,
            "regime": COMPARATOR_FORMAT[label],
            "bnd_arm": arm,
            "slot": map_arm,
            "y_target": bc.Y_MEAN,
            "arm": map_arm,
        }
        for map_arm in c.ARMS
    ]


def v1_matched_cells(arm: str) -> list[dict]:
    """V1-anchor refit cells on the arm-and-V1 intersection (both mapping arms)."""
    return [
        {
            "cell_id": f"R_{bg.MODEL_KEY}_v1_bnd_{bg.ARM_SLUG[arm]}_{map_arm}",
            "model_key": bg.MODEL_KEY,
            "format_key": V1_STEM_FORMAT,
            "track": bc.TRACK,
            "slot_index": V1_SLOT_INDEX[map_arm],
            "target_turn_index": 0,
            "regime": V1_STEM_FORMAT,
            "bnd_arm": arm,
            "slot": map_arm,
            "y_target": bc.Y_MEAN,
            "arm": map_arm,
        }
        for map_arm in c.ARMS
    ]


def arm_matched_cells(arm: str) -> list[dict]:
    """The arm's OWN cells refit on the arm-and-V1 intersection (paired vs V1)."""
    return [
        {
            "cell_id": f"R_{bg.MODEL_KEY}_bndm_{bg.ARM_SLUG[arm]}_{map_arm}",
            "model_key": bg.MODEL_KEY,
            "format_key": bc.format_key(arm),
            "track": bc.TRACK,
            # The intersection refit compares against the V1 store's 2 slots, so
            # it reads the boundary store's matching positions only.
            "slot_index": list(bc.BND_SLOT_ORDER).index(
                "prefix" if map_arm == "prefix" else bc.HEADLINE_SLOT
            ),
            "target_turn_index": bc.Y_TARGET_INDEX[bc.Y_MEAN],
            "regime": bc.format_key(arm),
            "bnd_arm": arm,
            "slot": map_arm,
            "y_target": bc.Y_MEAN,
            "arm": map_arm,
        }
        for map_arm in c.ARMS
    ]


# ---------------------------------------------------------------------------
# Bundle access
# ---------------------------------------------------------------------------
def load_bundle(turnstore_dir: Path, model_key: str, format_key: str, expect_slots: int) -> dict:
    """Load one store via the production pt-shard loader + sanity asserts."""
    bundle = fc._load_bundle_any(
        turnstore_dir, model_key, format_key, bc.TRACK, wanted_keys=SLIM_KEYS
    )
    c.assert_pt_bundle(bundle, expect_slots=expect_slots, expect_layers=fc.EXPECTED_LAYERS)
    return bundle


def store_present(turnstore_dir: Path, model_key: str, format_key: str) -> bool:
    """Cheap presence probe (npz contract OR pt shards) with no load."""
    stem = f"{model_key}_{format_key}_{bc.TRACK}"
    return bool((turnstore_dir / f"{stem}.npz").exists() or list(turnstore_dir.glob(f"{stem}*.pt")))


def store_conv_ids(turnstore_dir: Path, model_key: str, format_key: str) -> list[str]:
    """Row conv_ids read from the cheap shard sidecars (no tensor load)."""
    stem = f"{model_key}_{format_key}_{bc.TRACK}"
    sidecars = sorted(turnstore_dir.glob(f"{stem}_shard*.json"))
    if not sidecars:
        side = turnstore_dir / f"{stem}.json"
        assert side.exists(), f"no sidecars for {stem} in {turnstore_dir}"
        return [str(x) for x in json.loads(side.read_text())["conv_ids"]]
    ids: list[str] = []
    for sp in sidecars:
        ids.extend(str(x) for x in json.loads(sp.read_text())["conv_ids"])
    return ids


# ---------------------------------------------------------------------------
# Leg 1+2 — per-cell fits + the two standing mapping baselines
# ---------------------------------------------------------------------------
def mapping_baseline_reads(
    xy: dict, *, n_folds: int, seed: int, n_boot: int, ridge_pred: np.ndarray | None
) -> dict:
    """identity+learned-bias and kNN retrieval at the headline layer, out-of-fold.

    The identity+bias baseline (v_hat = x + b, b = train-fold mean of y - x)
    isolates how much of the map's R^2 a context-independent constant shift
    already explains; kNN retrieval reports P(true target within the k nearest
    neighbours of the prediction) among the held-out pool, with chance stated.
    The folds/seed are the cell's own, so every read is on the same split.
    """
    X = xy["X"][:, LAYER, :].astype(np.float64)
    Y = xy["Y"][:, LAYER, :].astype(np.float64)
    conv = np.asarray([str(x) for x in xy["conv_ids"]])
    if X.shape[1] != Y.shape[1]:
        return {
            "identity_bias": {
                "inapplicable": f"d_in {X.shape[1]} != d_out {Y.shape[1]} — identity+bias "
                "baseline needs a same-space map"
            }
        }
    folds = fc._cv_folds(conv, n_folds, seed)
    pred = np.zeros_like(Y)
    fitted = np.zeros(len(Y), bool)
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or tr.sum() < 3:
            continue
        pred[te] = mb.identity_bias_predict(X[tr], Y[tr], X[te])
        fitted[te] = True
    if not fitted.any():
        return {"identity_bias": {"skipped": "no usable folds"}}
    ib_pred, true = pred[fitted], Y[fitted]
    out = {
        "identity_bias": {
            **c.conv_bootstrap_r2(ib_pred, true, conv[fitted], n_boot=n_boot, seed=seed + 900),
            "b_norm": float(np.linalg.norm(mb.identity_bias_predict(X, Y, X[:1]) - X[:1])),
        },
        "knn_identity_bias": {
            m: mb.knn_retrieval(ib_pred, true, ks=KNN_KS, metric=m) for m in ("euclidean", "cosine")
        },
    }
    if ridge_pred is not None:
        assert ridge_pred.shape == true.shape, (ridge_pred.shape, true.shape)
        out["knn_ridge"] = {
            m: mb.knn_retrieval(ridge_pred.astype(np.float64), true, ks=KNN_KS, metric=m)
            for m in ("euclidean", "cosine")
        }
    return out


def run_cells(
    cells: list[dict],
    bundles: dict[tuple[str, str], dict],
    out_dir: Path,
    preds_dir: Path,
    allow_by_cell: dict[str, list[str]],
    *,
    n_folds: int,
    seed: int,
    null_draws: int,
    n_boot: int,
    smoke: bool,
) -> dict[str, dict]:
    """Fit each cell, persist its JSONs + OOF preds, and attach the baselines.

    Mirrors issue1345_fit_cells.run_cells (same shared-bundle injection, same
    conversation-level bootstrap, same preds npz contract) and additionally
    computes the two standing mapping baselines from the cell's OWN folds.
    """
    preds_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for cell in cells:
        cid = cell["cell_id"]
        bundle = bundles[(cell["model_key"], cell["format_key"])]
        allow = allow_by_cell.get(cid)
        if smoke:
            xy_probe = fc._apply_row_allowlist(fc._cell_xy(bundle, cell), allow, cid)
            reason = degenerate_fold_reason(xy_probe["conv_ids"], n_folds=n_folds, seed=seed)
            if reason:
                print(
                    f"[fits][smoke] SKIP cell {cid}: {reason} — informational "
                    "(production semantics unchanged)",
                    flush=True,
                )
                summary[cid] = {"skipped": reason, "cell": cell}
                continue
        res = fc.run_cell(
            cell,
            Path("."),  # unused: bundle is injected
            out_dir,
            n_folds=n_folds,
            seed=seed,
            null_draws=null_draws,
            n_boot=n_boot,
            allowlist=allow,
            bundle=bundle,
        )
        sweep, xy = res["sweep"], res["xy"]
        fitted = sweep["fitted_mask"]
        li = LAYER if LAYER in sweep["preds_frozen"] else max(sweep["preds_frozen"])
        pred = sweep["preds_frozen"][li][fitted]
        true = xy["Y"][fitted, li, :]
        conv = np.asarray([str(x) for x in xy["conv_ids"][fitted]])
        np.savez(
            preds_dir / f"{cid}_L{li}.npz",
            pred=pred.astype(np.float32),
            true=true.astype(np.float32),
            conv_ids=conv,
            layer=np.asarray([li]),
        )
        boot = {
            str(int(lj)): c.conv_bootstrap_r2(
                sweep["preds_frozen"][lj][fitted],
                xy["Y"][fitted, lj, :],
                conv,
                n_boot=n_boot,
                seed=seed + 200 + lj,
            )
            for lj in sweep["preds_frozen"]
        }
        baselines = mapping_baseline_reads(
            xy, n_folds=n_folds, seed=seed, n_boot=n_boot, ridge_pred=pred if li == LAYER else None
        )
        cell_json = out_dir / f"cells_{cid}.json"
        payload = json.loads(cell_json.read_text())
        payload["r2_bootstrap_ci_frozen_layers_conv"] = boot
        payload["n_groups"] = len(np.unique(conv))
        payload["mapping_baselines_headline_layer"] = baselines
        payload["bnd_arm"] = cell.get("bnd_arm")
        payload["slot"] = cell.get("slot")
        c.write_json(cell_json, payload)
        summary[cid] = {
            "cell": cell,
            "layer": int(li),
            "r2": float(payload["r2_per_layer_obs"][li]),
            "ci": boot.get(str(int(li))),
            "null_p975": _null_p975(out_dir, cid, li),
            "mean_baseline_r2": payload["mean_baseline_r2"].get(str(li)),
            "skill_over_mean": payload["skill_over_mean"].get(str(li)),
            "baselines": baselines,
        }
        print(
            f"[fits] {cid} done (n={len(conv)}, groups={payload['n_groups']}, "
            f"L{li} R2={summary[cid]['r2']:.4f})",
            flush=True,
        )
    return summary


def _null_p975(out_dir: Path, cell_id: str, layer: int) -> float | None:
    """Upper 97.5th percentile of the cell's shuffle-null R^2 at ``layer``."""
    p = out_dir / f"nulls_{cell_id}.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["null_matrix"]
    if not m:
        return None
    col = [row[layer] for row in m if layer < len(row) and row[layer] == row[layer]]
    return float(np.quantile(col, 0.975)) if col else None


# ---------------------------------------------------------------------------
# Paired deltas (shared conversation set -> ONE counts matrix per draw)
# ---------------------------------------------------------------------------
def paired_delta(
    reads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    pairs: list[tuple[str, str]],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Conversation-level PAIRED bootstrap of R^2 differences.

    ``reads`` maps name -> (pred, true, conv_ids), all on the IDENTICAL
    conversation set (asserted). One shared counts matrix drives every
    statistic per draw, so a difference CI is a PAIRED read, not the
    non-overlap of two independent CIs.
    """
    suffs, uniq_ref = {}, None
    for name, (pred, true, conv) in reads.items():
        suff = c.conv_suffstats(pred, true, conv)
        if uniq_ref is None:
            uniq_ref = suff["uniq"]
        assert np.array_equal(suff["uniq"], uniq_ref), (
            f"{name}: conversation set mismatch — a paired delta needs identical groups"
        )
        suffs[name] = suff
    counts = c.bootstrap_counts(len(uniq_ref), n_boot, seed)
    r2 = {name: c.batched_conv_r2(counts, s) for name, s in suffs.items()}

    def _ci(v):
        return {
            "mean": float(np.nanmean(v)),
            "ci_lo": float(np.nanquantile(v, 0.025)),
            "ci_hi": float(np.nanquantile(v, 0.975)),
        }

    out = {
        "n_boot": int(n_boot),
        "n_groups": int(len(uniq_ref)),
        "unit": "conversation (paired resample across every named read)",
        "reads": {name: _ci(v) for name, v in r2.items()},
        "deltas": {},
    }
    for a, b in pairs:
        d = r2[a] - r2[b]
        ci = _ci(d)
        ci["ci_excludes_zero"] = bool(ci["ci_lo"] > 0.0 or ci["ci_hi"] < 0.0)
        out["deltas"][f"{a}_minus_{b}"] = ci
    return out


def _load_preds(preds_dir: Path, cell_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    p = preds_dir / f"{cell_id}_L{LAYER}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    return (
        d["pred"].astype(np.float64),
        d["true"].astype(np.float64),
        np.asarray([str(x) for x in d["conv_ids"]]),
    )


# ---------------------------------------------------------------------------
# Leg 4a — X x Y grid with the X-side factorization SHARED across Y
# ---------------------------------------------------------------------------
def _t(a: np.ndarray):
    import torch

    return torch.as_tensor(np.asarray(a), dtype=torch.float64)


def xy_grid(
    bundle: dict,
    store_key: str,
    *,
    allow: list[str] | None,
    n_folds: int,
    seed: int,
    null_draws: int,
    n_boot: int,
) -> dict:
    """Held-out R^2 for every (X slot, Y target) pair, X factorization SHARED.

    ONE eigh(Gram) per (X slot, fold) — `ma._ridge_prep` — is reused by BOTH Y
    targets and by every shuffle-null draw at that slot: the lambda grid rides
    the cached factorization as diagonal filter rescalings, and a null draw is a
    Y-permutation against the SAME prep. So the grid costs one X-side
    factorization per slot/fold, not one per (slot, target) — the addendum's
    "X-side factorizations shared across Y".

    Returns per-cell pooled R^2 + conversation-level bootstrap CI + the
    shuffle-null band, keyed ``"<slot>|<y>"``.
    """
    import issue825_map_alignment as ma
    import torch

    arrays = bundle["arrays"]
    slots = np.asarray(arrays["slots"], dtype=np.float32)
    profiles = np.asarray(arrays["profiles"], dtype=np.float32)
    conv_all = np.asarray([str(x) for x in bundle["sidecar"].get("conv_ids", [])])
    assert slots.shape[1] == len(bc.BND_SLOT_ORDER), (slots.shape, bc.BND_SLOT_ORDER)
    assert profiles.shape[1] == len(bc.Y_SPAN_ORDER), (profiles.shape, bc.Y_SPAN_ORDER)
    keep = np.ones(len(conv_all), bool)
    if allow is not None:
        keep = np.isin(conv_all, np.asarray(sorted(set(allow))))
        assert keep.any(), f"{store_key}: xy_grid allowlist selected zero rows"
    conv = conv_all[keep]
    folds = fc._cv_folds(conv, n_folds, seed)
    rng = np.random.default_rng(seed + 31)
    out: dict[str, dict] = {}
    for si, slot in enumerate(bc.BND_SLOT_ORDER):
        X = _t(slots[keep][:, si, LAYER, :])
        Ys = {y: _t(profiles[keep][:, bc.Y_TARGET_INDEX[y], LAYER, :]) for y in bc.Y_SPAN_ORDER}
        preds = {y: np.zeros((len(conv), Ys[y].shape[1]), np.float64) for y in bc.Y_SPAN_ORDER}
        fitted = np.zeros(len(conv), bool)
        null_acc = {
            y: [{"res": 0.0, "tot": 0.0} for _ in range(null_draws)] for y in bc.Y_SPAN_ORDER
        }
        for k in range(n_folds):
            tr, te = folds != k, folds == k
            if te.sum() == 0 or tr.sum() < 3:
                continue
            trt, tet = torch.as_tensor(tr), torch.as_tensor(te)
            # ONE factorization per (slot, fold) — reused across both Y targets
            # AND every null draw below.
            prep = ma._ridge_prep(X[trt])
            fitted[te] = True
            for y, Y in Ys.items():
                preds[y][te] = ma._ridge_predict(prep, Y[trt], X[tet]).cpu().numpy()
                mu = Y[trt].mean(0)
                tot = float(((Y[tet] - mu) ** 2).sum())
                for d in range(null_draws):
                    perm = rng.permutation(int(tr.sum()))
                    p_null = ma._ridge_predict(prep, Y[trt][torch.as_tensor(perm)], X[tet])
                    null_acc[y][d]["res"] += float(((Y[tet] - p_null) ** 2).sum())
                    null_acc[y][d]["tot"] += tot
        if not fitted.any():
            for y in bc.Y_SPAN_ORDER:
                out[f"{slot}|{y}"] = {"skipped": "no usable folds"}
            continue
        for y, Y in Ys.items():
            true = Y.cpu().numpy()[fitted]
            pred = preds[y][fitted]
            rec = dict(
                c.conv_bootstrap_r2(pred, true, conv[fitted], n_boot=n_boot, seed=seed + 400 + si)
            )
            vals = [
                1.0 - a["res"] / a["tot"] if a["tot"] > 1e-12 else float("nan") for a in null_acc[y]
            ]
            vals = [v for v in vals if v == v]
            rec["shuffle_null"] = {
                "n_draws": int(null_draws),
                "null_mean": float(np.mean(vals)) if vals else float("nan"),
                "null_p975": float(np.quantile(vals, 0.975)) if vals else float("nan"),
                "observed_above_null_p975": (
                    bool(rec["r2"] > np.quantile(vals, 0.975)) if vals else None
                ),
            }
            rec["knn"] = {
                m: mb.knn_retrieval(pred, true, ks=KNN_KS, metric=m)
                for m in ("euclidean", "cosine")
            }
            if pred.shape[1] == true.shape[1]:
                Xn = X.cpu().numpy()[fitted]
                ib = np.zeros_like(true)
                ib_fit = np.zeros(len(true), bool)
                sub_folds = folds[fitted]
                for k in range(n_folds):
                    tr, te = sub_folds != k, sub_folds == k
                    if te.sum() == 0 or tr.sum() < 3 or Xn.shape[1] != true.shape[1]:
                        continue
                    ib[te] = mb.identity_bias_predict(Xn[tr], true[tr], Xn[te])
                    ib_fit[te] = True
                if ib_fit.any():
                    rec["identity_bias"] = c.conv_bootstrap_r2(
                        ib[ib_fit],
                        true[ib_fit],
                        conv[fitted][ib_fit],
                        n_boot=n_boot,
                        seed=seed + 700 + si,
                    )
            out[f"{slot}|{y}"] = rec
    return {
        "store": store_key,
        "layer": LAYER,
        "n_folds": int(n_folds),
        "n_rows": int(keep.sum()),
        "n_groups": int(len(np.unique(conv))),
        "null_draws": int(null_draws),
        "x_slots": list(bc.BND_SLOT_ORDER),
        "y_targets": list(bc.Y_SPAN_ORDER),
        "x_grid_slots": list(bc.X_GRID_SLOTS),
        "factorization": "one eigh(Gram) per (X slot, fold), shared across both Y "
        "targets and every shuffle-null draw",
        "cells": out,
    }


def reparam_ladder(src: dict, tgt: dict, *, n_folds: int, seed: int, null_draws: int) -> dict:
    """Rungs of the A o M o B reparameterization chain, BOTH directions.

    Per fold ONE eigh(Gram) per source (`ma._ridge_prep`) is computed and reused
    by every rung and every null draw; the lambda grid is applied as diagonal
    filter rescalings of that factorization, and the shuffle-fit null permutes
    the TRAIN ANSWERS against the SAME cached prep. Rungs, with A = the
    context-side map (target ctx -> source ctx), M = the source regime's own
    operator, B = the answer-side map (source answer -> target answer):

      ceiling      M_tgt(X_tgt)          the target's own within-regime read
      direct       M_src(X_tgt)          source operator, no reparameterization
      ctx_reparam  M_src(A(X_tgt))       context reparameterization only
      ans_reparam  B(M_src(X_tgt))       answer reparameterization only
      amb          B(M_src(A(X_tgt)))    both

    ``src``/``tgt`` are {X, Y, conv_ids} on the SAME conversation set in the
    SAME row order (asserted).
    """
    import issue825_map_alignment as ma

    conv = np.asarray([str(x) for x in tgt["conv_ids"]])
    assert np.array_equal(conv, np.asarray([str(x) for x in src["conv_ids"]])), (
        "reparam ladder needs src/tgt rows aligned by conversation"
    )
    # The ladder is a HEADLINE-LAYER read: slice the (n, L, D) stores to
    # (n, D) at LAYER before any prep — ma._ridge_prep takes a 2-D source.
    for name, side in (("src", src), ("tgt", tgt)):
        for key in ("X", "Y"):
            assert side[key].ndim == 3 and side[key].shape[1] > LAYER, (
                f"{name}[{key}] has shape {side[key].shape} — expected (n, n_layers, D) "
                f"with n_layers > {LAYER}"
            )
    Xs, Ys = _t(src["X"][:, LAYER, :]), _t(src["Y"][:, LAYER, :])
    Xt, Yt = _t(tgt["X"][:, LAYER, :]), _t(tgt["Y"][:, LAYER, :])
    folds = fc._cv_folds(conv, n_folds, seed)
    rng = np.random.default_rng(seed + 7)

    rungs = ("ceiling", "direct", "ctx_reparam", "ans_reparam", "amb")
    acc = {r: {"res": 0.0, "tot": 0.0} for r in rungs}
    null_acc = {r: [{"res": 0.0, "tot": 0.0} for _ in range(null_draws)] for r in rungs[1:]}
    n_used = 0
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or tr.sum() < 3:
            continue
        n_used += int(te.sum())
        import torch

        trt, tet = torch.as_tensor(tr), torch.as_tensor(te)
        # ONE factorization per (source, fold), reused by every rung + null draw.
        p_xs = ma._ridge_prep(Xs[trt])
        p_xt = ma._ridge_prep(Xt[trt])
        p_ys = ma._ridge_prep(Ys[trt])
        y_true = Yt[tet]
        mu = Yt[trt].mean(0)
        tot = float(((y_true - mu) ** 2).sum())
        # A: target ctx -> source ctx (fit on train, applied to held-out target).
        xs_hat = ma._ridge_predict(p_xt, Xs[trt], Xt[tet])

        def _pred(rung: str, ys_train):
            if rung == "ceiling":
                return ma._ridge_predict(p_xt, Yt[trt], Xt[tet])
            x_in = xs_hat if rung in ("ctx_reparam", "amb") else Xt[tet]
            ys_hat = ma._ridge_predict(p_xs, ys_train, x_in)
            if rung in ("ans_reparam", "amb"):
                # B: source answer -> target answer.
                return ma._ridge_predict(p_ys, Yt[trt], ys_hat)
            return ys_hat

        for rung in rungs:
            pred = _pred(rung, Ys[trt])
            acc[rung]["res"] += float(((y_true - pred) ** 2).sum())
            acc[rung]["tot"] += tot
        for d in range(null_draws):
            perm = rng.permutation(int(tr.sum()))
            ys_shuf = Ys[trt][torch.as_tensor(perm)]
            for rung in rungs[1:]:
                pred = _pred(rung, ys_shuf)
                null_acc[rung][d]["res"] += float(((y_true - pred) ** 2).sum())
                null_acc[rung][d]["tot"] += tot

    def _r2(a):
        return 1.0 - a["res"] / a["tot"] if a["tot"] > 1e-12 else float("nan")

    out = {
        "layer": LAYER,
        "n_folds": int(n_folds),
        "n_rows_scored": n_used,
        "n_groups": int(len(np.unique(conv))),
        "null_draws": int(null_draws),
        "r2": {r: _r2(acc[r]) for r in rungs},
    }
    ceiling = out["r2"]["ceiling"]
    out["deficit_vs_ceiling"] = {r: (out["r2"][r] - ceiling) for r in rungs if r != "ceiling"}
    out["shuffle_fit_null"] = {}
    for r, draws in null_acc.items():
        vals = [_r2(a) for a in draws]
        vals = [v for v in vals if v == v]
        out["shuffle_fit_null"][r] = {
            "null_mean": float(np.mean(vals)) if vals else float("nan"),
            "null_p975": float(np.quantile(vals, 0.975)) if vals else float("nan"),
            "observed_above_null_p975": (
                bool(out["r2"][r] > np.quantile(vals, 0.975)) if vals else None
            ),
        }
    return out


def arm_xy(bundle: dict, cell: dict, keep_ids: list[str]) -> dict:
    """(X, Y, conv_ids) for one cell restricted to ``keep_ids``, conv-sorted.

    Sorting by conv_id on both sides of a reparam pair makes the rows align by
    construction (one row per conversation in every store used here).
    """
    xy = fc._cell_xy(bundle, cell)
    conv = np.asarray([str(x) for x in xy["conv_ids"]])
    keep = np.isin(conv, np.asarray(sorted(set(keep_ids))))
    order = np.argsort(conv[keep], kind="stable")
    return {
        "X": xy["X"][keep][order],
        "Y": xy["Y"][keep][order],
        "conv_ids": conv[keep][order],
    }


# ---------------------------------------------------------------------------
# Leg 5 — verdict lattice
# ---------------------------------------------------------------------------
def _read_committed(path: Path, layer: int) -> dict | None:
    """{r2, ci_lo, ci_hi, n} at ``layer`` from a committed cells JSON."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    out = {
        "r2": float(d["r2_per_layer_obs"][layer]),
        "n": d["metadata"].get("n"),
        "source": str(path),
    }
    boot = d.get("r2_bootstrap_ci_frozen_layers_conv", {}).get(str(layer))
    if boot:
        out.update({"ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "n_groups": boot["n_groups"]})
    return out


def _cis_disjoint(a: dict | None, b: dict | None) -> bool | None:
    """Descriptive non-overlap of two INDEPENDENT CIs (never a test)."""
    if not a or not b or "ci_lo" not in a or "ci_lo" not in b:
        return None
    return bool(a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])


def build_verdict(
    arm: str,
    cell_summary: dict[str, dict],
    paired: dict,
    reparam: dict,
    grid: dict,
    comparator_grids: dict[str, dict],
    *,
    n_kept: int,
    n_intersection: int | None,
) -> dict:
    """Per-arm verdict record: the headline read against every reference."""
    slug = bg.ARM_SLUG[arm]
    own = cell_summary.get(grid_cell_id(arm, bc.HEADLINE_SLOT, bc.Y_MEAN), {})
    anchor = _read_committed(V1_ANCHOR_FILES["context"], LAYER)
    chat_anchor = _read_committed(V1_MATCHED_CHAT_FILES["context"], LAYER)
    chat_cell = cell_summary.get(f"R_{bg.MODEL_KEY}_r1_chat_bnd_{slug}_context", {})
    nt_cell = cell_summary.get(f"R_{bg.MODEL_KEY}_r2_no_template_bnd_{slug}_context", {})
    verdict = {
        "arm": arm,
        "arm_isolates": bg.ARM_README[arm],
        "headline_slot": bc.HEADLINE_SLOT,
        "layer": LAYER,
        "n_kept_stories": n_kept,
        "n_intersection_with_v1": n_intersection,
        "slots": {
            cell_summary[cid]["cell"]["slot"]: {
                k: cell_summary[cid].get(k) for k in ("r2", "ci", "null_p975", "skill_over_mean")
            }
            for cid in (grid_cell_id(arm, s, bc.Y_MEAN) for s in bc.BND_SLOT_ORDER)
            if cid in cell_summary and "cell" in cell_summary[cid]
        },
        "headline": {k: own.get(k) for k in ("r2", "ci", "null_p975", "skill_over_mean")},
        "baselines_headline_slot": own.get("baselines"),
        "vs_v1_anchor_committed": {
            "anchor": anchor,
            "anchor_doc_crosscheck": V1_ANCHOR_DOC["context"],
            "delta_point": (
                (own.get("r2") - anchor["r2"]) if (own.get("r2") is not None and anchor) else None
            ),
            "independent_cis_disjoint": _cis_disjoint(own.get("ci"), anchor),
            "note": "committed V1 read on the FULL V1 kept set — different rows, so this is "
            "a descriptive comparison; the PAIRED read is vs_v1_matched below",
        },
        "vs_matched_chat": {
            "chat_same_rows": {k: chat_cell.get(k) for k in ("r2", "ci")},
            "v1_matched_chat_committed": chat_anchor,
            "v1_matched_chat_doc_crosscheck": V1_MATCHED_CHAT_DOC["context"],
            "no_template_same_rows": {k: nt_cell.get(k) for k in ("r2", "ci")},
        },
        "paired_deltas": paired,
        "reparam_story_vs_chat": reparam,
        # The consolidated X x Y measurement grid (addendum): the arm's own grid
        # plus the same grid on each round-own comparator store, so every
        # (read position x target) pair is comparable arm-vs-comparator.
        "xy_grid": {
            "x_clean_slot": bc.X_CLEAN_SLOT,
            "x_straddle_slot": bc.X_STRADDLE_SLOT,
            "y_targets": list(bc.Y_SPAN_ORDER),
            "transition_appended_verbatim": (
                bc.TRANSITION[arm]["closer"] + bc.TRANSITION[arm]["suffix"]
            ),
            "transition_read_anchor": bc.TRANSITION[arm]["read_anchor"],
            "arm": grid,
            "comparators": comparator_grids,
        },
    }
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _import_check() -> None:
    """Resolve every deferred import on the REAL code path, then exit 0."""
    import inspect

    import torch  # noqa: F401

    import issue825_map_alignment as ma  # noqa: F401

    assert inspect.getsource(reparam_ladder)
    assert callable(ma._ridge_prep) and callable(ma._ridge_predict)
    print("[import-check] OK: torch + issue825_map_alignment symbols resolved", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase", choices=("all", "cells", "grid", "reparam", "verdict"), default="all"
    )
    ap.add_argument("--arms", default=",".join(bg.ARM_SLUG[a] for a in bg.GEN_ARMS))
    ap.add_argument(
        "--reparam-arms", default=",".join(bg.ARM_SLUG[a] for a in DEFAULT_REPARAM_ARMS)
    )
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--stories-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--out-dir", type=Path, default=c.EVAL_DIR / "story_boundary_ablation")
    ap.add_argument("--preds-dir", type=Path, default=c.PREDS_CACHE_DIR / "boundary_ablation")
    ap.add_argument("--n-folds", type=int, default=fc.N_FOLDS)
    ap.add_argument("--seed", type=int, default=fc.FIT_SEED)
    ap.add_argument("--null-draws", type=int, default=N_NULL_DRAWS)
    ap.add_argument("--n-boot", type=int, default=c.N_BOOTSTRAP)
    ap.add_argument("--smoke", action="store_true", help="tiny nulls/boot; degenerate-fold skips")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the real code path and exit 0",
    )
    args = ap.parse_args()

    if args.import_check:
        _import_check()
        return

    bg.assert_round_env()
    arms = [bg.SLUG_ARM.get(a, a) for a in args.arms.split(",") if a]
    assert arms and set(arms) <= set(bg.GEN_ARMS), arms
    reparam_arms = [bg.SLUG_ARM.get(a, a) for a in args.reparam_arms.split(",") if a]
    assert set(reparam_arms) <= set(arms), (reparam_arms, arms)
    null_draws = SMOKE_NULL_DRAWS if args.smoke else args.null_draws
    n_boot = SMOKE_BOOT if args.smoke else args.n_boot
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Kept conversation sets (from the arm stores, so the fit rows and the
    # allowlists can never drift) + the V1 intersection when V1 is staged.
    arm_convs = {
        arm: store_conv_ids(args.turnstore_dir, bg.MODEL_KEY, bc.format_key(arm)) for arm in arms
    }
    v1_available = store_present(args.turnstore_dir, bg.MODEL_KEY, V1_STEM_FORMAT)
    v1_convs = (
        store_conv_ids(args.turnstore_dir, bg.MODEL_KEY, V1_STEM_FORMAT) if v1_available else []
    )
    comparators_available = {
        label: store_present(args.turnstore_dir, bg.MODEL_KEY, fmt)
        for label, fmt in COMPARATOR_FORMAT.items()
    }
    # Round-own X x Y comparator stores (capture --comparator chat|no_template).
    bnd_comparators_available = {
        key: store_present(args.turnstore_dir, bg.MODEL_KEY, bc.format_key(key))
        for key in BND_COMPARATORS
    }
    print(
        f"[fits] arms={[bg.ARM_SLUG[a] for a in arms]} "
        f"kept={{{', '.join(f'{bg.ARM_SLUG[a]}:{len(v)}' for a, v in arm_convs.items())}}} "
        f"v1_store={v1_available} parent_comparators={comparators_available} "
        f"xy_comparators={bnd_comparators_available}",
        flush=True,
    )

    # Cell registry + per-cell allowlists.
    cells: list[dict] = []
    allow: dict[str, list[str]] = {}
    inter: dict[str, list[str]] = {}
    for arm in arms:
        cells += arm_cells(arm)
        for label, present in comparators_available.items():
            if not present:
                print(f"[fits] comparator store {label} absent — matched refits skipped")
                continue
            for cell in comparator_cells(arm, label):
                cells.append(cell)
                allow[cell["cell_id"]] = sorted(arm_convs[arm])
        if v1_available:
            ids = sorted(set(arm_convs[arm]) & set(v1_convs))
            inter[arm] = ids
            if len(ids) >= args.n_folds:
                for cell in v1_matched_cells(arm) + arm_matched_cells(arm):
                    cells.append(cell)
                    allow[cell["cell_id"]] = ids
            else:
                print(
                    f"[fits] {arm}: V1 intersection {len(ids)} < n_folds {args.n_folds} — "
                    "paired V1 refits skipped",
                    flush=True,
                )
    # The X x Y grid on the round-own comparator stores: fit ONCE per comparator
    # (not per arm) over the union of the arms' kept conversations, which is
    # exactly the row set those stores were captured on.
    for key, present in bnd_comparators_available.items():
        if not present:
            print(f"[fits] X x Y comparator store {key} absent — grid cells skipped", flush=True)
            continue
        cells += grid_cells(key)

    bundles: dict[tuple[str, str], dict] = {}
    for cell in cells:
        key = (cell["model_key"], cell["format_key"])
        if key in bundles:
            continue
        expect = len(bc.BND_SLOT_ORDER) if cell["format_key"].startswith("bnd_") else 2
        bundles[key] = load_bundle(args.turnstore_dir, key[0], key[1], expect)

    cell_summary: dict[str, dict] = {}
    if args.phase in ("all", "cells"):
        cell_summary = run_cells(
            cells,
            bundles,
            args.out_dir,
            args.preds_dir,
            allow,
            n_folds=args.n_folds,
            seed=args.seed,
            null_draws=null_draws,
            n_boot=n_boot,
            smoke=args.smoke,
        )
        c.write_json(
            args.out_dir / "cell_summary.json",
            {
                "metadata": c.metadata(
                    args.seed, len(cell_summary), "scripts/issue1345_boundary_ablation_fits.py"
                ),
                "round": bg.ROUND_VARIANT,
                "layer": LAYER,
                "null_draws": null_draws,
                "n_boot": n_boot,
                "arm_readme": {a: bg.ARM_README[a] for a in bg.ALL_ARMS},
                "cells": cell_summary,
            },
        )

    # Paired deltas (shared conversation set -> one counts matrix per draw).
    paired_by_arm: dict[str, dict] = {}
    if args.phase in ("all", "cells", "verdict"):
        for arm in arms:
            slug = bg.ARM_SLUG[arm]
            reads: dict[str, tuple] = {}
            own = _load_preds(args.preds_dir, grid_cell_id(arm, bc.HEADLINE_SLOT, bc.Y_MEAN))
            chat = _load_preds(args.preds_dir, f"R_{bg.MODEL_KEY}_r1_chat_bnd_{slug}_context")
            nt = _load_preds(args.preds_dir, f"R_{bg.MODEL_KEY}_r2_no_template_bnd_{slug}_context")
            pairs: list[tuple[str, str]] = []
            if own and chat and np.array_equal(np.unique(own[2]), np.unique(chat[2])):
                reads["arm"], reads["chat"] = own, chat
                pairs.append(("arm", "chat"))
                if nt and np.array_equal(np.unique(own[2]), np.unique(nt[2])):
                    reads["no_template"] = nt
                    pairs.append(("arm", "no_template"))
            armm = _load_preds(args.preds_dir, f"R_{bg.MODEL_KEY}_bndm_{slug}_context")
            v1m = _load_preds(args.preds_dir, f"R_{bg.MODEL_KEY}_v1_bnd_{slug}_context")
            block: dict = {}
            if pairs:
                block["vs_comparators"] = paired_delta(
                    reads, pairs, n_boot=n_boot, seed=args.seed + 11
                )
            else:
                block["vs_comparators"] = {
                    "skipped": "comparator preds absent or on a different conversation set"
                }
            if armm and v1m and np.array_equal(np.unique(armm[2]), np.unique(v1m[2])):
                block["vs_v1_matched"] = paired_delta(
                    {"arm": armm, "v1": v1m},
                    [("arm", "v1")],
                    n_boot=n_boot,
                    seed=args.seed + 13,
                )
            else:
                block["vs_v1_matched"] = {
                    "skipped": "V1 store not staged, or the intersection refits were skipped"
                }
            paired_by_arm[arm] = block

    # X x Y grid with the X-side factorization SHARED across Y (addendum).
    grid_by_store: dict[str, dict] = {}
    if args.phase in ("all", "cells", "grid", "verdict"):
        grid_stores = list(arms) + [k for k, ok in bnd_comparators_available.items() if ok]
        for key in grid_stores:
            bkey = (bg.MODEL_KEY, bc.format_key(key))
            if bkey not in bundles:
                grid_by_store[key] = {"skipped": f"store {bc.format_key(key)} not staged"}
                continue
            ids = arm_convs.get(key)
            if args.smoke:
                probe = ids if ids is not None else store_conv_ids(args.turnstore_dir, *bkey)
                reason = degenerate_fold_reason(
                    np.asarray(probe), n_folds=args.n_folds, seed=args.seed
                )
                if reason:
                    grid_by_store[key] = {"skipped": f"smoke: {reason}"}
                    print(f"[grid][smoke] SKIP {key}: {reason}", flush=True)
                    continue
            grid_by_store[key] = xy_grid(
                bundles[bkey],
                key,
                allow=ids,
                n_folds=args.n_folds,
                seed=args.seed,
                null_draws=null_draws,
                n_boot=n_boot,
            )
            got = grid_by_store[key].get("cells", {})
            head = got.get(f"{bc.X_CLEAN_SLOT}|{bc.Y_MEAN}", {})
            print(
                f"[grid] {key}: {len(got)} X x Y cells (x_clean|y_mean R2={head.get('r2')})",
                flush=True,
            )
        c.write_json(
            args.out_dir / "xy_grid.json",
            {
                "metadata": c.metadata(
                    args.seed, len(grid_by_store), "scripts/issue1345_boundary_ablation_fits.py"
                ),
                "round": bg.ROUND_VARIANT,
                "layer": LAYER,
                "x_slots": list(bc.BND_SLOT_ORDER),
                "x_clean_slot": bc.X_CLEAN_SLOT,
                "x_straddle_slot": bc.X_STRADDLE_SLOT,
                "y_targets": list(bc.Y_SPAN_ORDER),
                "transition_suffixes_verbatim": {
                    k: {
                        "appended_verbatim": v["closer"] + v["suffix"],
                        "read_anchor": v["read_anchor"],
                    }
                    for k, v in bc.TRANSITION.items()
                },
                "stores": grid_by_store,
            },
        )

    # Reparameterization ladder, both directions, story arm <-> chat.
    reparam_by_arm: dict[str, dict] = {}
    if args.phase in ("all", "reparam", "verdict"):
        chat_key = (bg.MODEL_KEY, COMPARATOR_FORMAT["r1_chat"])
        for arm in reparam_arms:
            if chat_key not in bundles:
                reparam_by_arm[arm] = {"skipped": "chat comparator store not staged"}
                continue
            ids = sorted(set(arm_convs[arm]) & set(store_conv_ids(args.turnstore_dir, *chat_key)))
            if len(ids) < args.n_folds:
                reparam_by_arm[arm] = {
                    "skipped": f"shared conversation set {len(ids)} < n_folds {args.n_folds}"
                }
                continue
            story_cell = next(
                cl
                for cl in arm_cells(arm)
                if cl["slot"] == bc.HEADLINE_SLOT and cl["y_target"] == bc.Y_MEAN
            )
            chat_cell = next(
                cl for cl in comparator_cells(arm, "r1_chat") if cl["slot"] == "context"
            )
            story = arm_xy(bundles[(bg.MODEL_KEY, bc.format_key(arm))], story_cell, ids)
            chat = arm_xy(bundles[chat_key], chat_cell, ids)
            if args.smoke:
                reason = degenerate_fold_reason(
                    story["conv_ids"], n_folds=args.n_folds, seed=args.seed
                )
                if reason:
                    reparam_by_arm[arm] = {"skipped": f"smoke: {reason}"}
                    print(f"[reparam][smoke] SKIP {arm}: {reason}", flush=True)
                    continue
            nd = SMOKE_NULL_DRAWS if args.smoke else N_REPARAM_NULL_DRAWS
            reparam_by_arm[arm] = {
                "n_shared_conversations": len(ids),
                "chat_to_story": reparam_ladder(
                    chat, story, n_folds=args.n_folds, seed=args.seed, null_draws=nd
                ),
                "story_to_chat": reparam_ladder(
                    story, chat, n_folds=args.n_folds, seed=args.seed, null_draws=nd
                ),
            }
            print(
                f"[reparam] {arm}: chat->story amb deficit "
                f"{reparam_by_arm[arm]['chat_to_story']['deficit_vs_ceiling']['amb']:+.4f}, "
                f"story->chat amb deficit "
                f"{reparam_by_arm[arm]['story_to_chat']['deficit_vs_ceiling']['amb']:+.4f}",
                flush=True,
            )
        c.write_json(
            args.out_dir / "reparam_ladder.json",
            {
                "metadata": c.metadata(
                    args.seed, len(reparam_by_arm), "scripts/issue1345_boundary_ablation_fits.py"
                ),
                "round": bg.ROUND_VARIANT,
                "layer": LAYER,
                "arms": reparam_by_arm,
            },
        )

    # Verdict lattice.
    if args.phase in ("all", "verdict"):
        if not cell_summary:
            cs_path = args.out_dir / "cell_summary.json"
            assert cs_path.exists(), f"{cs_path} missing — run --phase cells first"
            cell_summary = json.loads(cs_path.read_text())["cells"]
        verdicts = {
            arm: build_verdict(
                arm,
                cell_summary,
                paired_by_arm.get(arm, {}),
                reparam_by_arm.get(arm, {"skipped": "reparam phase not run"}),
                grid_by_store.get(arm, {"skipped": "grid phase not run"}),
                {k: grid_by_store[k] for k in BND_COMPARATORS if k in grid_by_store},
                n_kept=len(arm_convs[arm]),
                n_intersection=(len(inter[arm]) if arm in inter else None),
            )
            for arm in arms
        }
        c.write_json(
            args.out_dir / "verdict_lattice.json",
            {
                "metadata": c.metadata(
                    args.seed, len(verdicts), "scripts/issue1345_boundary_ablation_fits.py"
                ),
                "round": bg.ROUND_VARIANT,
                "layer": LAYER,
                "headline_slot": bc.HEADLINE_SLOT,
                "slot_order": list(bc.BND_SLOT_ORDER),
                "y_span_order": list(bc.Y_SPAN_ORDER),
                "x_grid_slots": list(bc.X_GRID_SLOTS),
                "n_folds": args.n_folds,
                "seed": args.seed,
                "null_draws": null_draws,
                "n_boot": n_boot,
                "smoke": bool(args.smoke),
                "arm_readme": {a: bg.ARM_README[a] for a in bg.ALL_ARMS},
                "v1_anchor_files": {k: str(v) for k, v in V1_ANCHOR_FILES.items()},
                "arms": verdicts,
            },
        )
        for arm, v in verdicts.items():
            print(
                f"[verdict] {bg.ARM_SLUG[arm]} L{LAYER} {bc.HEADLINE_SLOT} "
                f"R2={v['headline'].get('r2')} vs V1 anchor "
                f"{v['vs_v1_anchor_committed']['anchor'] and v['vs_v1_anchor_committed']['anchor']['r2']}",
                flush=True,
            )
    print(f"[done] boundary-ablation fits -> {args.out_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
