"""Issue #931 P4: cross-regime transfer matrix + weight/subspace similarity.

Transfer fractions follow the plan section-4.0 REGISTERED verdict recipe:
  - X-recipe-matched denominators — every lastpos-X transfer involving
    chat_ref reads against a SINGLE-POSITION within ceiling
    (arm[AB]_within_lastpos / chat_ref); span-mean transfers read against
    arm[AB]_within. A lastpos fraction over a span-mean ceiling is never
    emitted.
  - Matched-power PRIMARY — n* = min(n_source, n_target); the source map is
    refit on a seeded (seed 931) group-stratified n*-subsample; the
    denominator is the target's within ceiling AT n* (chat: this run's
    parametrized power curve point; else a seeded n*-subsample single-layer
    refit). Full-n fractions ride along as registered secondaries
    (power_matched=false).
  - Applications: recentered (PRIMARY — per test fold, X means + Y offset
    from the TARGET regime's train folds; scales + W from the source) and
    strict-frozen (SECONDARY).
  - Nulls: target-side group-blocked pairing permutation (20 draws) through
    the SAME application path (no refits — the prediction is Y-independent,
    only offsets + reductions recompute per draw).

Every transfer_matrix.json row carries the mechanizable metadata contract:
source_cell, target_cell, x_recipe, denominator_cell, application,
n_train/groups_train for source AND denominator, power_matched.

Also: per-layer full-map weights (Gram-dual GCV, explicit W in raw input
coords), top-k right/left singular-subspace overlaps (k in {16,64,256};
random-subspace null + analytic k/D), linear CKA of the two maps' predictions
on a shared held-out input sample, and the layer-profile Spearman table.

CLI:
  uv run python scripts/issue931_similarity.py [--data-dir data/issue_931]
      [--out-dir eval_results/issue_931] [--chat-store-dir <dir>]
      [--skip-subspace] [--n-transfer-nulls 20] [--smoke]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402

# r1 Minor fix: import the CKA / principal-angle helpers from their canonical
# source instead of re-copying them (reuse hierarchy; argparse there is inside
# main(), so the import is side-effect-free beyond an idempotent load_dotenv).
from issue779_stage2_cka_subspace import _subspace_overlap, linear_cka  # noqa: E402
from issue931_fit_cells import (  # noqa: E402
    assemble_cell,
    frozen_layers,
    headline_layer,
    load_regime_store,
)

SCRIPT = "scripts/issue931_similarity.py"
LAMBDAS = fit825.LAMBDAS
SUBSPACE_KS = (16, 64, 256)
CKA_SAMPLE_N = 2000

# Map specs: cell_id -> (regime|chat, X key). The weight/CKA comparisons pair
# LIKE input recipes on both sides (chat-side reads use the lastpos variants).
MAP_SPECS = {
    "chat_ref": ("chat", None),
    "armA_within": ("armA", "x_spanmean"),
    "armA_within_lastpos": ("armA", "x_last"),
    "armB_within": ("armB", "x_spanmean"),
    "armB_within_lastpos": ("armB", "x_last"),
    "armC_sep": ("armC", "x_sep"),
    "armC_prevmean": ("armC", "x_spanmean"),
}

# Registered transfer directions (plan section 4.5). target_x names the
# TARGET-side X column the frozen map is applied to.
TRANSFER_DIRECTIONS = [
    # PRIMARY chat-side transfers: position-matched X (lastpos).
    dict(
        src="chat_ref",
        tgt="armA",
        target_x="x_last",
        recipe="lastpos",
        denom="armA_within_lastpos",
        tier="primary",
    ),
    dict(
        src="armA_within_lastpos",
        tgt="chat",
        target_x=None,
        recipe="lastpos",
        denom="chat_ref",
        tier="primary",
    ),
    dict(
        src="chat_ref",
        tgt="armB",
        target_x="x_last",
        recipe="lastpos",
        denom="armB_within_lastpos",
        tier="primary",
    ),
    dict(
        src="armB_within_lastpos",
        tgt="chat",
        target_x=None,
        recipe="lastpos",
        denom="chat_ref",
        tier="primary",
    ),
    # Span-mean transfers (recipe-matched on both sides by construction).
    dict(
        src="armA_within",
        tgt="armB",
        target_x="x_spanmean",
        recipe="spanmean",
        denom="armB_within",
        tier="primary",
    ),
    dict(
        src="armB_within",
        tgt="armA",
        target_x="x_spanmean",
        recipe="spanmean",
        denom="armA_within",
        tier="primary",
    ),
    # Separator controls (H3): single-position recipes/ceilings.
    dict(
        src="armC_sep",
        tgt="armA",
        target_x="x_last",
        recipe="lastpos",
        denom="armA_within_lastpos",
        tier="primary",
    ),
    dict(
        src="armA_within_lastpos",
        tgt="armC",
        target_x="x_sep",
        recipe="lastpos",
        denom="armC_sep",
        tier="secondary",
    ),
    dict(
        src="armC_prevmean",
        tgt="armA",
        target_x="x_spanmean",
        recipe="spanmean",
        denom="armA_within",
        tier="secondary",
    ),
    # SECONDARY span-mean-X chat-side transfers (recipe-confounded; registered).
    dict(
        src="chat_ref",
        tgt="armA",
        target_x="x_spanmean",
        recipe="spanmean",
        denom="armA_within",
        tier="secondary",
    ),
    dict(
        src="chat_ref",
        tgt="armB",
        target_x="x_spanmean",
        recipe="spanmean",
        denom="armB_within",
        tier="secondary",
    ),
    dict(
        src="armA_within",
        tgt="chat",
        target_x=None,
        recipe="spanmean",
        denom="chat_ref",
        tier="secondary",
    ),
    dict(
        src="armB_within",
        tgt="chat",
        target_x=None,
        recipe="spanmean",
        denom="chat_ref",
        tier="secondary",
    ),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_931"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_931"))
    ap.add_argument("--chat-store-dir", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=common.N_FOLDS)
    ap.add_argument("--seed", type=int, default=common.FIT_SEED)
    ap.add_argument("--n-transfer-nulls", type=int, default=20)
    ap.add_argument("--skip-subspace", action="store_true")
    ap.add_argument("--save-maps", action="store_true", help="persist fp16 W stacks per spec")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Full-map fit (Gram-dual GCV; same estimator family as issue825_fit_cells)
# ---------------------------------------------------------------------------


def fit_full_map(X: np.ndarray, Y: np.ndarray) -> dict:
    """Explicit W in the standardized-X gauge (GCV lambda on FULL data).

    yc = ((x - xmu) / xsd) @ W_std;  pred = yc + ymu. fp64 on the fit device
    (cached-eigh Gram dual — one factorization per (cell, layer), the #825
    vectorization contract; W = X_n^T V diag(1/(w+lam)) V^T Y_c).
    """
    dev = fit825._fit_device()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=dev)
    Yt = torch.as_tensor(np.asarray(Y), dtype=torch.float64, device=dev)
    xmu = Xt.mean(0)
    xsd = Xt.std(0) + 1e-9
    Xn = (Xt - xmu) / xsd
    ymu = Yt.mean(0)
    Yc = Yt - ymu
    ntr = Xn.shape[0]
    G = Xn @ Xn.T
    w, V = torch.linalg.eigh(G)
    w = torch.clamp(w, min=0.0)
    VtY = V.T @ Yc
    sqVtY = (VtY**2).sum(1)
    tot = float((Yc**2).sum())
    best_lam, best_gcv = float(LAMBDAS[0]), float("inf")
    for lam in LAMBDAS:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    alpha = V @ ((1.0 / (w + best_lam)).unsqueeze(1) * VtY)  # (n, D_out)
    W_std = Xn.T @ alpha  # (D_in, D_out)
    return {
        "W_std": W_std,
        "xmu": xmu,
        "xsd": xsd,
        "ymu": ymu,
        "lam": best_lam,
        "n_train": int(ntr),
    }


def transfer_r2(
    fmap: dict,
    X_B: np.ndarray,
    Y_B: np.ndarray,
    groups_B: np.ndarray,
    *,
    application: str,
    folds: int,
    seed: int,
    n_nulls: int = 0,
) -> dict:
    """Frozen-map transfer R^2 on regime B under group-5-fold intercept handling.

    recentered: per test fold, X-standardization MEANS + Y offset from B's
    train folds; scales (xsd) and W stay from A. strict: A's means/offsets
    verbatim. Nulls permute the B pairing at GROUP level and re-apply (the
    prediction direction is Y-independent — only offsets/reductions differ).
    """
    dev = fmap["W_std"].device
    X = torch.as_tensor(np.asarray(X_B), dtype=torch.float64, device=dev)
    Y = torch.as_tensor(np.asarray(Y_B), dtype=torch.float64, device=dev)
    fold_ids = fit825._cv_folds(np.asarray(groups_B), folds, seed)
    rng = np.random.default_rng(seed + 1)
    ids = np.asarray(groups_B)
    uniq, inv = np.unique(ids, return_inverse=True)
    row_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]

    def _group_perm() -> np.ndarray:
        gp = rng.permutation(len(uniq))
        return np.concatenate([row_of[k] for k in gp])

    perms = [np.arange(len(ids))] + [_group_perm() for _ in range(n_nulls)]
    ss_res = np.zeros(len(perms))
    ss_tot = np.zeros(len(perms))
    for k in range(folds):
        te = fold_ids == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            continue
        te_t = torch.as_tensor(te, device=dev)
        tr_t = torch.as_tensor(tr, device=dev)
        xmu = X[tr_t].mean(0) if application == "recentered" else fmap["xmu"]
        base_pred = ((X[te_t] - xmu) / fmap["xsd"]) @ fmap["W_std"]  # (n_te, D)
        for d, perm in enumerate(perms):
            Yp = Y[torch.as_tensor(perm, device=dev)]
            ymu = Yp[tr_t].mean(0) if application == "recentered" else fmap["ymu"]
            pred = base_pred + ymu
            true = Yp[te_t]
            mu = true.mean(0)
            ss_res[d] += float(((true - pred) ** 2).sum())
            ss_tot[d] += float(((true - mu) ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)
    return {
        "r2": float(r2[0]),
        "null_r2": [float(v) for v in r2[1:]],
        "null_p975": float(np.nanquantile(r2[1:], 0.975)) if n_nulls else None,
        "n_test_total": len(ids),
    }


# ---------------------------------------------------------------------------
# Cell data + ceilings
# ---------------------------------------------------------------------------


def _cell_arrays(
    cells: dict, cell_id: str, layer: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = cells[cell_id]
    return xy["X"][:, layer, :], xy["Y"][:, layer, :], np.asarray(xy["group_ids"])


def _within_ceiling_full(out_dir: Path, cell_id: str, layer: int) -> float:
    payload = json.loads((out_dir / f"cells_{cell_id}.json").read_text())
    return float(payload["r2_per_layer_obs"][layer])


def _chat_power_ceiling(out_dir: Path, n_star: int, layer: int) -> tuple[float, int]:
    """Chat within ceiling at n* from THIS RUN's parametrized power curve."""
    pc = json.loads((out_dir / "power_curve_chat.json").read_text())
    pts = [c for c in pc["curve"] if c.get("r2_per_layer")]
    assert pts, "power_curve_chat.json has no usable points"
    best = min(pts, key=lambda c: abs(c["n"] - n_star))
    assert best["n"] == n_star or abs(best["n"] - n_star) <= max(50, 0.05 * n_star), (
        f"no power-curve point at n*={n_star} (closest {best['n']}) — "
        "fit_cells must include n_A/n_B in ns"
    )
    li = min(layer, len(best["r2_per_layer"]) - 1)
    return float(best["r2_per_layer"][li]), int(best["n"])


def _subsample_xy(
    X: np.ndarray, Y: np.ndarray, groups: np.ndarray, n_star: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = common.group_stratified_subsample(groups, n_star, seed=common.BUILD_SEED)
    return X[idx], Y[idx], groups[idx]


def _within_ceiling_at_n(
    cells: dict, cell_id: str, layer: int, n_star: int, out_dir: Path, *, folds: int, seed: int
) -> tuple[float, int, int]:
    """Target within ceiling at n* (identity -> committed full-n value)."""
    if cell_id == "chat_ref":
        n_full = cells["chat_ref"]["X"].shape[0]
        if n_star >= n_full:
            return (
                _within_ceiling_full(out_dir, cell_id, layer),
                n_full,
                len(np.unique(cells[cell_id]["group_ids"])),
            )
        r2, n_used = _chat_power_ceiling(out_dir, n_star, layer)
        return r2, n_used, n_used  # chat groups == rows (one turn per conv)
    X, Y, g = _cell_arrays(cells, cell_id, layer)
    if n_star >= X.shape[0]:
        return _within_ceiling_full(out_dir, cell_id, layer), X.shape[0], len(np.unique(g))
    Xs, Ys, gs = _subsample_xy(X, Y, g, n_star)
    sw = fit825.heldout_r2_sweep(
        Xs[:, None, :],
        Ys[:, None, :],
        gs,
        n_folds=folds,
        seed=seed,
        null_draws=0,
        collect_cosines=False,
    )
    return float(sw["r2_obs"][0]), int(Xs.shape[0]), len(np.unique(gs))


# ---------------------------------------------------------------------------
# Subspace / CKA reads
# ---------------------------------------------------------------------------


def random_subspace_null(D: int, ks, n_draws: int = 20, seed: int = 0) -> dict:
    """Overlap of two INDEPENDENT random k-subspaces in R^D (empirical null)."""
    g = torch.Generator().manual_seed(seed)
    out = {int(k): [] for k in ks}
    for _ in range(n_draws):
        for k in ks:
            kk = min(k, D)
            A = torch.linalg.qr(torch.randn(D, kk, generator=g, dtype=torch.float64))[0]
            B = torch.linalg.qr(torch.randn(D, kk, generator=g, dtype=torch.float64))[0]
            out[int(k)].append(float(((A.T @ B) ** 2).sum() / kk))
    return {
        str(k): {
            "mean": float(np.mean(v)),
            "p975": float(np.quantile(v, 0.975)),
            "analytic_expectation": min(k, D) / D,
        }
        for k, v in out.items()
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:  # noqa: C901 -- linear P4 driver (transfers -> contract checks -> subspace)
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chat_dir = args.chat_store_dir or (args.data_dir / "chat_store")
    print("[phase=p4_similarity] transfer matrix + subspace/CKA")

    store_root = args.data_dir / "store"
    stores = {r: load_regime_store(store_root / r, r) for r in ("armA", "armB", "armC")}
    if args.smoke:
        # Tiny-model smoke: rebind the reused loader's layer-axis assert to
        # the smoke store dims (the #825 tiny-model pattern).
        fit825.EXPECTED_LAYERS = int(stores["armA"]["arrays"]["y"].shape[1])
    cells: dict[str, dict] = {}
    for cell_id in MAP_SPECS:
        cells[cell_id] = assemble_cell(cell_id, stores, chat_dir)
    n_layers = cells["armA_within"]["X"].shape[1]
    for cid, xy in cells.items():
        assert xy["X"].shape[1] == n_layers, (cid, xy["X"].shape, n_layers)
    fl = frozen_layers(n_layers)
    hl = headline_layer(n_layers)

    def target_arrays(direction: dict, layer: int):
        if direction["tgt"] == "chat":
            xy = cells["chat_ref"]
            return xy["X"][:, layer, :], xy["Y"][:, layer, :], np.asarray(xy["group_ids"])
        store = stores[direction["tgt"]]
        X = store["arrays"][direction["target_x"]][:, layer, :]
        Y = store["arrays"]["y"][:, layer, :]
        return X, Y, store["group_ids"]

    # ---- Transfer matrix -------------------------------------------------
    rows = []
    fmap_cache: dict[tuple, dict] = {}

    def get_map(cell_id: str, layer: int, n_sub: int | None) -> dict:
        key = (cell_id, layer, n_sub)
        if key not in fmap_cache:
            X, Y, g = _cell_arrays(cells, cell_id, layer)
            if n_sub is not None and n_sub < X.shape[0]:
                X, Y, g = _subsample_xy(X, Y, g, n_sub)
            fm = fit_full_map(X, Y)
            fm["groups_train"] = len(np.unique(g))
            fmap_cache[key] = fm
        return fmap_cache[key]

    for direction in TRANSFER_DIRECTIONS:
        src, denom = direction["src"], direction["denom"]
        for layer in fl:
            Xb, Yb, gb = target_arrays(direction, layer)
            n_src_full = cells[src]["X"].shape[0]
            n_tgt = Xb.shape[0]
            n_star = min(n_src_full, n_tgt)
            for power_matched in (True, False):
                n_sub = n_star if (power_matched and n_star < n_src_full) else None
                if not power_matched and n_star == n_src_full == n_tgt:
                    continue  # full-n row would duplicate the matched row
                fmap = get_map(src, layer, n_sub)
                denom_r2, denom_n, denom_g = (
                    _within_ceiling_at_n(
                        cells,
                        denom,
                        layer,
                        n_star,
                        args.out_dir,
                        folds=args.folds,
                        seed=args.seed,
                    )
                    if power_matched
                    else (
                        _within_ceiling_full(args.out_dir, denom, layer),
                        cells[denom]["X"].shape[0],
                        len(np.unique(cells[denom]["group_ids"])),
                    )
                )
                for application in ("recentered", "strict"):
                    tr = transfer_r2(
                        fmap,
                        Xb,
                        Yb,
                        gb,
                        application=application,
                        folds=args.folds,
                        seed=args.seed,
                        n_nulls=args.n_transfer_nulls if layer == hl else 0,
                    )
                    frac = tr["r2"] / denom_r2 if denom_r2 and denom_r2 > 1e-6 else float("nan")
                    rows.append(
                        {
                            "source_cell": src,
                            "target_cell": direction["tgt"]
                            if direction["tgt"] == "chat"
                            else f"{direction['tgt']}:{direction['target_x']}",
                            "direction": f"{src}->{direction['tgt']}",
                            "layer": int(layer),
                            "x_recipe": direction["recipe"],
                            "denominator_cell": denom,
                            "application": application,
                            "tier": direction["tier"],
                            "power_matched": bool(power_matched),
                            "n_train": int(fmap["n_train"]),
                            "groups_train": int(fmap["groups_train"]),
                            "denominator_n_train": int(denom_n),
                            "denominator_groups_train": int(denom_g),
                            "n_target": int(n_tgt),
                            "transfer_r2": tr["r2"],
                            "within_ceiling_r2": denom_r2,
                            "fraction_of_ceiling": frac,
                            "null_p975": tr["null_p975"],
                            "lam": fmap["lam"],
                        }
                    )
        print(f"[i931-p4] transfers done: {direction['src']} -> {direction['tgt']}")

    # Mechanizable contract checks (plan section 6.5).
    for r in rows:
        if r["x_recipe"] == "lastpos" and (
            r["source_cell"] == "chat_ref" or r["denominator_cell"] == "chat_ref"
        ):
            # Plan section 6.5 three-cell set for chat-involving lastpos rows
            # (r1 Minor: armC_sep was unreachable slack here — no chat-involving
            # direction can read against the separator ceiling).
            assert r["denominator_cell"] in (
                "armA_within_lastpos",
                "armB_within_lastpos",
                "chat_ref",
            ), r
        if r["power_matched"]:
            assert abs(r["n_train"] - r["denominator_n_train"]) <= max(50, 0.05 * r["n_train"]), (
                "power_matched row n mismatch",
                r,
            )
    common.write_json(
        args.out_dir / "transfer_matrix.json",
        {
            "metadata": common.metadata(SCRIPT, args.seed, len(rows)),
            "headline_layer": hl,
            "frozen_layers": fl,
            "recenter_primary": True,
            "rows": rows,
        },
    )

    # ---- Weight/subspace similarity + CKA + layer profiles ---------------
    if not args.skip_subspace:
        comparisons = [
            ("chat_ref", "armA_within_lastpos", "lastpos"),
            ("chat_ref", "armB_within_lastpos", "lastpos"),
            ("armA_within", "armB_within", "spanmean"),
            ("armC_sep", "armA_within_lastpos", "lastpos"),
            ("armC_prevmean", "armA_within", "spanmean"),
        ]
        D = cells["armA_within"]["X"].shape[2]
        null = random_subspace_null(D, SUBSPACE_KS, seed=args.seed)
        # Shared CKA input sample: seeded novel-regime context reads.
        rng = np.random.default_rng(common.BUILD_SEED)
        nA = stores["armA"]["arrays"]["x_last"].shape[0]
        samp = rng.choice(nA, size=min(CKA_SAMPLE_N, nA), replace=False)
        maps_dir = args.data_dir / "store" / "maps"
        specs_needed = sorted({c for a, b, _ in comparisons for c in (a, b)})
        svd_cache: dict[tuple, tuple] = {}

        def get_svd(cell_id: str, layer: int):
            key = (cell_id, layer)
            if key not in svd_cache:
                fm = get_map(cell_id, layer, None)
                W_raw = (fm["W_std"] / fm["xsd"][:, None]).to(torch.float32)
                U, _s, Vh = torch.linalg.svd(W_raw, full_matrices=False)
                kmax = max(SUBSPACE_KS)
                svd_cache[key] = (U[:, :kmax].cpu(), Vh[:kmax].T.cpu(), fm)
                if args.save_maps:
                    maps_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"W_raw_fp16": W_raw.to(torch.float16).cpu(), "layer": layer},
                        maps_dir / f"{cell_id}_L{layer:02d}.pt",
                    )
            return svd_cache[key]

        sub_out: dict[str, dict] = {}
        for a, b, recipe in comparisons:
            per_layer = {}
            for layer in range(n_layers):
                Ua, Va, fma = get_svd(a, layer)
                Ub, Vb, fmb = get_svd(b, layer)
                # CKA on the shared standardized held-out novel input sample.
                xk = "x_last" if recipe == "lastpos" else "x_spanmean"
                Xs = torch.as_tensor(
                    stores["armA"]["arrays"][xk][samp, layer, :],
                    dtype=torch.float64,
                    device=fma["W_std"].device,
                )
                pa = ((Xs - fma["xmu"]) / fma["xsd"]) @ fma["W_std"]
                pb = ((Xs - fmb["xmu"]) / fmb["xsd"]) @ fmb["W_std"]
                per_layer[str(layer)] = {
                    "right_overlap": _subspace_overlap(Va, Vb, SUBSPACE_KS),
                    "left_overlap": _subspace_overlap(Ua, Ub, SUBSPACE_KS),
                    "cka_preds": linear_cka(pa.cpu(), pb.cpu()),
                }
            sub_out[f"{a}__vs__{b}"] = {"recipe": recipe, "per_layer": per_layer}
            print(f"[i931-p4] subspace/CKA done: {a} vs {b}")

        profiles, spearman = {}, {}
        for cell_id in MAP_SPECS:
            p = args.out_dir / f"cells_{cell_id}.json"
            if p.exists():
                profiles[cell_id] = json.loads(p.read_text())["r2_per_layer_obs"]
        keys = sorted(profiles)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                k = min(len(profiles[a]), len(profiles[b]))
                spearman[f"{a}__vs__{b}"] = fit825._spearman(
                    np.asarray(profiles[a][:k]), np.asarray(profiles[b][:k])
                )
        common.write_json(
            args.out_dir / "subspace_cka.json",
            {
                "metadata": common.metadata(SCRIPT, args.seed, len(comparisons)),
                "ks": list(SUBSPACE_KS),
                "hidden_dim": int(D),
                "random_subspace_null": null,
                "comparisons": sub_out,
                "layer_profile_spearman": spearman,
                "specs_with_saved_maps": specs_needed if args.save_maps else [],
            },
        )
    print("[i931-p4] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
