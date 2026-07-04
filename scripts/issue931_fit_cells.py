"""Issue #931 P3: fit battery over all cells + G1 replication gate + G1b MLP
parity + matched-n power curve.

Thin driver over the reused #825 machinery (`issue825_fit_cells`): GPU Gram
ridge with cached per-fold eigh reused across the 20 selection-symmetric null
draws (the vectorization contract), GROUP folds (novel / story / article /
conversation), full per-draw x per-layer null matrices persisted per cell.

Cells (plan section 4.5): armA_within, armA_within_lastpos, armA_swap,
armA_ctxmean, armB_within, armB_within_lastpos, armB_swap, armB_ctxmean,
armC_sep, armC_prevmean, chat_ref.

Also:
  - stages the reused #825 Track-S chat turnstore (scoped list_repo_tree at a
    STAGING-TIME PINNED data-repo revision + per-file hf_hub_download, <=6
    workers; the pin lands in run_manifest.json)
  - G1: chat_ref refit vs the COMMITTED #825 Track-S curve
    (eval_results/issue_825/cells_S1.json) — Spearman >= 0.9, |dR2@L19| <= 0.05
  - power curve at ns = {1000, 2000, n_A, n_B} via the parametrized
    issue825_fit_cells.run_power_curve (out: power_curve_chat.json)
  - group-level bootstrap CIs (batched per-group-reduction GEMM, zero refits)
  - dR2_char (correct - within-story swap) with paired novel-level bootstrap
  - P3b MLP secondary (batched fit_batched_split_mlp, MSE loss, parent-parity
    full-fold-train X standardization + patience-20, call-site target PCA-64
    via a device-routed torch SVD) + G1b parity vs fit_h.mlp_fit_predict on
    chat_ref @ L19

CLI:
  uv run python scripts/issue931_fit_cells.py [--cells all|id,id,...]
      [--data-dir data/issue_931] [--out-dir eval_results/issue_931]
      [--chat-store-dir <dir>] [--stage-chat-store] [--fabricate-chat-smoke]
      [--mlp] [--g1b] [--null-draws 20] [--folds 5] [--seed 0] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps bind before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402

SCRIPT = "scripts/issue931_fit_cells.py"

ALL_CELLS = (
    "armA_within",
    "armA_within_lastpos",
    "armA_swap",
    "armA_ctxmean",
    "armB_within",
    "armB_within_lastpos",
    "armB_swap",
    "armB_ctxmean",
    "armC_sep",
    "armC_prevmean",
    "chat_ref",
)

G1_REF_PATH = Path("eval_results/issue_825/cells_S1.json")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", type=str, default="all")
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_931"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_931"))
    ap.add_argument("--chat-store-dir", type=Path, default=None)
    ap.add_argument("--stage-chat-store", action="store_true")
    ap.add_argument(
        "--stage-only",
        action="store_true",
        help="stage/fabricate the chat store then exit (the overlapped P2' job)",
    )
    ap.add_argument(
        "--fabricate-chat-smoke",
        action="store_true",
        help="write a tiny synthetic Track-S store (schema-conformant; smoke only)",
    )
    ap.add_argument(
        "--fabricate-dims",
        type=str,
        default="4,64",
        help="layers,dim for the fabricated smoke chat store (match the tiny model)",
    )
    ap.add_argument("--mlp", action="store_true", help="run the P3b MLP secondary")
    ap.add_argument("--g1b", action="store_true", help="run the G1b MLP parity fit")
    ap.add_argument("--null-draws", type=int, default=common.N_NULL_DRAWS)
    ap.add_argument("--folds", type=int, default=common.N_FOLDS)
    ap.add_argument("--seed", type=int, default=common.FIT_SEED)
    ap.add_argument("--n-boot", type=int, default=common.N_BOOTSTRAP)
    ap.add_argument("--smoke", action="store_true", help="numeric gates recorded, not binding")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Chat-store staging (revision-pinned; scoped tree listing; <=6 workers)
# ---------------------------------------------------------------------------


def stage_chat_store(dest: Path, out_dir: Path) -> dict:
    """Download the #825 Track-S instruct shards at a PINNED repo revision."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    revision = api.repo_info(common.HF_DATA_REPO, repo_type="dataset").sha
    print(f"[i931-p3] chat-store staging pinned at data-repo revision {revision}")
    entries = [
        e
        for e in api.list_repo_tree(
            common.HF_DATA_REPO,
            path_in_repo=common.CHAT_STORE_PREFIX,
            repo_type="dataset",
            recursive=True,
            revision=revision,
        )
        if Path(e.path).name.startswith(common.CHAT_STORE_STEM + "_shard")
    ]
    assert entries, f"no {common.CHAT_STORE_STEM} shards under {common.CHAT_STORE_PREFIX}"
    dest.mkdir(parents=True, exist_ok=True)

    def _fetch(path: str) -> str:
        for attempt in range(4):
            try:
                got = hf_hub_download(
                    common.HF_DATA_REPO,
                    path,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=dest / "_hf",
                )
                target = dest / Path(path).name
                if not target.exists():
                    target.symlink_to(Path(got).resolve())
                return path
            except Exception as exc:  # transient Hub 5xx/429 — bounded retry
                if attempt == 3:
                    raise
                wait = 20 * (attempt + 1)
                print(f"[i931-p3] retry {path} in {wait}s: {exc}")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(_fetch, [e.path for e in entries]))
    manifest = {
        "metadata": common.metadata(SCRIPT, common.FIT_SEED, len(entries)),
        "hf_data_repo": common.HF_DATA_REPO,
        "revision": revision,
        "prefix": common.CHAT_STORE_PREFIX,
        "n_files": len(entries),
        "files": sorted(e.path for e in entries),
    }
    common.write_json(out_dir / "run_manifest.json", manifest)
    return manifest


def fabricate_chat_smoke(dest: Path, *, n: int = 40, layers: int = 4, dim: int = 16) -> None:
    """Tiny synthetic Track-S shards satisfying the #825 .pt loader contract."""
    rng = np.random.default_rng(0)
    dest.mkdir(parents=True, exist_ok=True)
    for shard in range(2):
        k = n // 2
        slots = rng.normal(size=(k, 1, layers, dim)).astype(np.float32)
        profiles = np.concatenate(
            [
                rng.normal(size=(k, 1, layers, dim)).astype(np.float32),
                0.7 * slots + 0.3 * rng.normal(size=(k, 1, layers, dim)).astype(np.float32),
            ],
            axis=1,
        )
        payload = {
            "conv_ids": [f"smoke_{shard}_{i:03d}" for i in range(k)],
            "slots": [torch.from_numpy(slots[i]) for i in range(k)],
            "profiles": [torch.from_numpy(profiles[i]) for i in range(k)],
        }
        torch.save(payload, dest / f"instruct_chat_s_shard{shard:03d}.pt")
        (dest / f"instruct_chat_s_shard{shard:03d}.json").write_text(
            json.dumps({"n_conversations": k, "conv_ids": payload["conv_ids"]})
        )
    print(f"[i931-p3] fabricated smoke chat store at {dest}")


# ---------------------------------------------------------------------------
# Store loading + cell assembly
# ---------------------------------------------------------------------------


def load_regime_store(store_dir: Path, regime: str) -> dict:
    """Concatenate a regime's shards -> {row_ids, group_ids, char_ids, arrays}."""
    shards = sorted(store_dir.glob(f"{regime}_shard*.pt"))
    assert shards, f"no {regime} shards under {store_dir}"
    rows, groups, chars = [], [], []
    arrays: dict[str, list] = {}
    for sp in shards:
        payload = torch.load(sp, map_location="cpu", weights_only=False)
        rows.extend(payload["row_ids"])
        groups.extend(payload["group_ids"])
        chars.extend(payload["char_ids"])
        for k, v in payload["arrays"].items():
            arrays.setdefault(k, []).append(v.float().numpy().astype(np.float32))
    out = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    n = len(rows)
    for k, v in out.items():
        assert v.shape[0] == n, (k, v.shape, n)
    return {
        "row_ids": np.asarray(rows),
        "group_ids": np.asarray(groups),
        "char_ids": np.asarray(chars),
        "arrays": out,
    }


def _window_ids(store: dict) -> np.ndarray:
    """window_id per row = row_id minus the ':<char_id>' suffix."""
    out = []
    for rid, cid in zip(store["row_ids"], store["char_ids"], strict=True):
        suffix = ":" + cid
        assert rid.endswith(suffix), (rid, cid)
        out.append(rid[: -len(suffix)])
    return np.asarray(out)


def swap_derangement(store: dict, seed: int = common.BUILD_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Seeded within-window derangement (B2). Returns (row_idx, partner_idx).

    Only windows with >= 2 eligible characters participate; within each, rows
    are seeded-shuffled and each row's Y-partner is the next row cyclically —
    a guaranteed derangement (no row keeps its own target).
    """
    win = _window_ids(store)
    rng = np.random.default_rng(seed)
    rows_out, partners_out = [], []
    for w in np.unique(win):
        idx = np.flatnonzero(win == w)
        if len(idx) < 2:
            continue
        perm = idx[rng.permutation(len(idx))]
        for j in range(len(perm)):
            rows_out.append(perm[j])
            partners_out.append(perm[(j + 1) % len(perm)])
    assert rows_out, "no >=2-character windows for the swap control"
    rows = np.asarray(rows_out)
    partners = np.asarray(partners_out)
    assert (rows != partners).all(), "derangement violated"
    return rows, partners


def load_chat_xy(chat_dir: Path) -> dict:
    """chat_ref XY via the #825 Track-S loader (X = slots[:, 0] single position)."""
    cell = fit825._normalize_cell({"cell_id": "S1", "model": "instruct"})
    bundle = fit825._load_bundle_any(chat_dir, cell["model_key"], cell["format_key"], cell["track"])
    xy = fit825._cell_xy(bundle, cell)
    ids = np.asarray(xy["conv_ids"]).astype(str)
    return {"X": xy["X"], "Y": xy["Y"], "group_ids": ids, "row_ids": ids}


def assemble_cell(cell_id: str, stores: dict, chat_dir: Path) -> dict:
    """(X (N,L,D), Y (N,L,D), group_ids, row_ids) for one registered cell."""
    if cell_id == "chat_ref":
        return load_chat_xy(chat_dir)
    regime = {"armA": "armA", "armB": "armB", "armC": "armC"}[cell_id.split("_")[0]]
    store = stores[regime]
    a = store["arrays"]
    g = store["group_ids"]
    rids = store["row_ids"]
    if cell_id in ("armA_within", "armB_within"):
        return {"X": a["x_spanmean"], "Y": a["y"], "group_ids": g, "row_ids": rids}
    if cell_id in ("armA_within_lastpos", "armB_within_lastpos"):
        return {"X": a["x_last"], "Y": a["y"], "group_ids": g, "row_ids": rids}
    if cell_id in ("armA_ctxmean", "armB_ctxmean"):
        return {"X": a["x_ctxmean"], "Y": a["y"], "group_ids": g, "row_ids": rids}
    if cell_id in ("armA_swap", "armB_swap"):
        rows, partners = swap_derangement(store)
        return {
            "X": a["x_spanmean"][rows],
            "Y": a["y"][partners],
            "group_ids": g[rows],
            "row_ids": rids[rows],
            "swap_rows": rids[rows],
        }
    if cell_id == "armC_sep":
        return {"X": a["x_sep"], "Y": a["y"], "group_ids": g, "row_ids": rids}
    if cell_id == "armC_prevmean":
        return {"X": a["x_spanmean"], "Y": a["y"], "group_ids": g, "row_ids": rids}
    raise KeyError(cell_id)


# ---------------------------------------------------------------------------
# Group-level bootstrap (batched per-group reductions; ZERO refits per draw)
# ---------------------------------------------------------------------------


def group_bootstrap_r2(
    pred: np.ndarray,
    true: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    draws_matrix: np.ndarray | None = None,
) -> dict:
    """Percentile bootstrap of pooled R^2 resampling GROUPS with replacement.

    Exact per-draw pooled R^2 from precomputed per-group reductions:
    ss_res_g, n_g, sum_y_g (D,), sumsq_g. All draws evaluate as ONE
    (draws, G) @ (G, .) GEMM — the vectorize-many-cell-fits batched-draw shape.
    ``draws_matrix`` (draws, G) multiplicities lets callers share draws for
    PAIRED statistics.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    uniq, inv = np.unique(np.asarray(group_ids), return_inverse=True)
    G = len(uniq)
    n_g = np.bincount(inv, minlength=G).astype(np.float64)
    resid = ((true - pred) ** 2).sum(axis=1)
    ss_res_g = np.bincount(inv, weights=resid, minlength=G)
    sumsq = np.bincount(inv, weights=(true**2).sum(axis=1), minlength=G)
    D = true.shape[1]
    sum_y_g = np.zeros((G, D))
    np.add.at(sum_y_g, inv, true)
    if draws_matrix is None:
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, G, size=(n_boot, G))
        draws_matrix = np.zeros((n_boot, G))
        for d in range(n_boot):  # cheap: (n_boot, G) ints -> counts
            draws_matrix[d] = np.bincount(picks[d], minlength=G)
    M = draws_matrix
    Np = M @ n_g  # (draws,)
    ss_res = M @ ss_res_g
    S = M @ sum_y_g  # (draws, D)
    Q = M @ sumsq
    ss_tot = Q - (S**2).sum(axis=1) / np.maximum(Np, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)
    mu = true.mean(0)
    obs = 1.0 - float(((true - pred) ** 2).sum()) / float(((true - mu) ** 2).sum())
    return {
        "r2": obs,
        "ci_lo": float(np.nanquantile(r2, 0.025)),
        "ci_hi": float(np.nanquantile(r2, 0.975)),
        "n_groups": int(G),
        "n_boot": int(M.shape[0]),
        "draws": r2,
        "draws_matrix": M,
        "group_order": uniq,
    }


def per_group_r2(pred: np.ndarray, true: np.ndarray, group_ids: np.ndarray) -> dict:
    """Held-out pooled R^2 per group (the low-level per-unit read for figures)."""
    out = {}
    for g in np.unique(group_ids):
        m = group_ids == g
        out[str(g)] = fit825._pooled_r2(pred[m], true[m])
    return out


# ---------------------------------------------------------------------------
# Per-cell fit (reuses heldout_r2_sweep; writes cells_/nulls_ JSONs + preds)
# ---------------------------------------------------------------------------


def frozen_layers(n_layers: int) -> list[int]:
    fl = [li for li in common.FROZEN_LAYERS if li < n_layers]
    return fl or [n_layers - 1]


def headline_layer(n_layers: int) -> int:
    return common.HEADLINE_LAYER if n_layers > common.HEADLINE_LAYER else n_layers - 1


def fit_cell(cell_id: str, xy: dict, args) -> dict:
    X, Y, groups = xy["X"], xy["Y"], xy["group_ids"]
    n, n_layers = X.shape[0], X.shape[1]
    print(f"[i931-p3] cell={cell_id} n={n} groups={len(np.unique(groups))}")
    # Rebind the reused module's frozen-layer set so preds/cosines are
    # collected at THIS run's frozen layers (identity at 28 layers; the tiny
    # smoke model rebinds to its last layer — same #825 tiny-model pattern).
    fit825.FROZEN_LAYERS = tuple(frozen_layers(n_layers))
    sweep = fit825.heldout_r2_sweep(
        X, Y, groups, n_folds=args.folds, seed=args.seed, null_draws=args.null_draws
    )
    r2_obs, r2_null = sweep["r2_obs"], sweep["r2_null"]
    summary = fit825.selection_symmetric_summary(r2_obs, r2_null)
    fl = frozen_layers(n_layers)
    hl = headline_layer(n_layers)
    mb = fit825.mean_baseline_r2(Y, groups, layers=fl, n_folds=args.folds, seed=args.seed)
    rp = fit825.random_projection_control(
        X, Y, groups, layers=[hl], n_folds=args.folds, seed=args.seed
    )
    fitted = sweep["fitted_mask"]
    boot_group, boot_row, pergroup = {}, {}, {}
    for li in fl:
        if li not in sweep["preds_frozen"]:
            continue
        pred = sweep["preds_frozen"][li][fitted]
        true = Y[fitted, li, :].astype(np.float64)
        gsub = np.asarray(groups)[fitted]
        gb = group_bootstrap_r2(pred, true, gsub, n_boot=args.n_boot, seed=args.seed + li)
        boot_group[str(li)] = {k: gb[k] for k in ("r2", "ci_lo", "ci_hi", "n_groups", "n_boot")}
        boot_row[str(li)] = fit825.bootstrap_r2_ci(
            pred, true, n_boot=args.n_boot, seed=args.seed + 100 + li
        )
        if li == hl:
            pergroup = per_group_r2(pred, true, gsub)
    payload = {
        "metadata": common.metadata(SCRIPT, args.seed, n),
        "cell_id": cell_id,
        "n": n,
        "n_groups": len(np.unique(groups)),
        "n_layers": int(n_layers),
        "headline_layer": hl,
        "frozen_layers": fl,
        "r2_per_layer_obs": [float(v) for v in r2_obs],
        "selection_symmetric": summary,
        "mean_baseline_r2": mb,
        "random_projection_control_r2": rp,
        "skill_over_mean": {
            str(li): float(r2_obs[li]) - float(mb.get(str(li), float("nan"))) for li in fl
        },
        "r2_bootstrap_group_frozen": boot_group,
        "r2_bootstrap_row_frozen": boot_row,
        "per_group_r2_headline": pergroup,
        "n_folds": args.folds,
        "null_draws": args.null_draws,
    }
    common.write_json(args.out_dir / f"cells_{cell_id}.json", payload)
    common.write_json(
        args.out_dir / f"nulls_{cell_id}.json",
        {
            "metadata": common.metadata(SCRIPT, args.seed, n),
            "cell_id": cell_id,
            "layers": list(range(n_layers)),
            "observed_row": [float(v) for v in r2_obs],
            "null_matrix": [[float(v) for v in row] for row in r2_null],
            "null_layer_max_per_draw": summary["null_layer_max_r2_per_draw"],
        },
    )
    # Persist headline-layer held-out preds (downstream figures + delta reads).
    preds_dir = args.data_dir / "store" / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    if hl in sweep["preds_frozen"]:
        np.savez(
            preds_dir / f"{cell_id}_L{hl}.npz",
            preds=sweep["preds_frozen"][hl][fitted].astype(np.float16),
            row_mask=fitted,
            group_ids=np.asarray(groups)[fitted].astype(str),
            row_ids=np.asarray(xy["row_ids"])[fitted].astype(str),
        )
    return {"sweep": sweep, "xy": xy, "payload": payload}


def delta_char(arm: str, res_within: dict, res_swap: dict, args) -> None:
    """dR2_char = R2(correct, swap-row subset) - R2(swap), paired novel bootstrap."""
    hl = headline_layer(res_within["xy"]["X"].shape[1])
    sw_w, sw_s = res_within["sweep"], res_swap["sweep"]
    if hl not in sw_w["preds_frozen"] or hl not in sw_s["preds_frozen"]:
        return
    store_rows = res_within["xy"]["row_ids"]
    swap_rows = res_swap["xy"]["swap_rows"]
    pos = {r: i for i, r in enumerate(store_rows)}
    sub = np.asarray([pos[r] for r in swap_rows])
    fitted_w, fitted_s = sw_w["fitted_mask"], sw_s["fitted_mask"]
    keep = fitted_w[sub] & fitted_s
    sub = sub[keep]
    pred_c = sw_w["preds_frozen"][hl][sub]
    true_c = res_within["xy"]["Y"][sub, hl, :].astype(np.float64)
    pred_s = sw_s["preds_frozen"][hl][keep]
    true_s = res_swap["xy"]["Y"][keep, hl, :].astype(np.float64)
    groups = np.asarray(res_swap["xy"]["group_ids"])[keep]
    gb_c = group_bootstrap_r2(pred_c, true_c, groups, n_boot=args.n_boot, seed=args.seed)
    gb_s = group_bootstrap_r2(
        pred_s,
        true_s,
        groups,
        n_boot=args.n_boot,
        seed=args.seed,
        draws_matrix=gb_c["draws_matrix"],
    )
    delta_draws = gb_c["draws"] - gb_s["draws"]
    payload = {
        "metadata": common.metadata(SCRIPT, args.seed, len(sub)),
        "arm": arm,
        "headline_layer": hl,
        "r2_correct_subset": gb_c["r2"],
        "r2_swap": gb_s["r2"],
        "delta_r2_char": gb_c["r2"] - gb_s["r2"],
        "delta_ci_lo": float(np.nanquantile(delta_draws, 0.025)),
        "delta_ci_hi": float(np.nanquantile(delta_draws, 0.975)),
        "n_rows": len(sub),
        "n_groups": int(gb_c["n_groups"]),
        "n_boot": int(args.n_boot),
        "paired_group_bootstrap": True,
    }
    common.write_json(args.out_dir / f"delta_char_{arm}.json", payload)


# ---------------------------------------------------------------------------
# G1 gate + power curve
# ---------------------------------------------------------------------------


def g1_gate(r2_obs: np.ndarray, out_dir: Path, *, smoke: bool, seed: int, n: int) -> dict:
    """Chat refit vs the COMMITTED #825 Track-S curve (Spearman + |dR2@19|)."""
    assert G1_REF_PATH.exists(), f"G1 reference missing: {G1_REF_PATH} (broken/sparse checkout)"
    ref_curve = json.loads(G1_REF_PATH.read_text())["r2_per_layer_obs"]
    k = min(len(ref_curve), len(r2_obs))
    rho = fit825._spearman(np.asarray(r2_obs[:k]), np.asarray(ref_curve[:k]))
    l19 = (
        abs(float(r2_obs[19]) - float(ref_curve[19]))
        if len(r2_obs) > 19 and len(ref_curve) > 19
        else float("nan")
    )
    ok = bool(rho >= 0.9 and (np.isnan(l19) or l19 <= 0.05))
    payload = {
        "metadata": common.metadata(SCRIPT, seed, n),
        "reference": str(G1_REF_PATH),
        "spearman_vs_825_S1": rho,
        "abs_dev_L19": l19,
        "layers_compared": k,
        "smoke": bool(smoke),
        "pass": ok,
    }
    common.write_json(out_dir / "g1_gate_931.json", payload)
    if not smoke and not ok:
        print(f"[i931-p3] G1 FAIL: spearman={rho:.3f} dL19={l19}", file=sys.stderr)
        raise SystemExit(5)
    return payload


# ---------------------------------------------------------------------------
# P3b: MLP secondary (batched split-MLP, MSE) + G1b parity vs the parent fitter
# ---------------------------------------------------------------------------


def _pca_basis_device(Y: np.ndarray, k: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """robust_pca_basis semantics (mean + top-k right singular vectors), device-routed.

    The r1 Critical fix (vectorize-many-cell-fits: dense-factorization battery):
    the parent's numpy gesdd SVD at production shape (~3200x3584 f32) measures
    ~30 s/call on 8 CPU threads, and P3b's ~360 member calls made it a 1.2-2.5 h
    serial CPU battery with the GPU idle. torch.linalg.svd on the fit device is
    ~1-2 s/call on A100 — subspace-identical up to sign, and the R^2 read is
    span-invariant through ``pred @ comps + mu``. Near-singular fallback mirrors
    robust_pca_basis (gesdd -> gesvd on cuda; the numpy/torch fallback on cpu).
    """
    from explore_persona_space.analysis.vectorized_mlp_skill import robust_pca_basis

    t = torch.from_numpy(np.ascontiguousarray(Y.astype(np.float32))).to(device)
    tc = t - t.mean(dim=0)
    try:
        _, _, Vh = torch.linalg.svd(tc, full_matrices=False)
    except torch.linalg.LinAlgError:
        if t.is_cuda:
            _, _, Vh = torch.linalg.svd(tc, full_matrices=False, driver="gesvd")
        else:
            mu_np, comps, _fb = robust_pca_basis(Y.astype(np.float32), k)
            return mu_np, comps
    kk = min(k, Vh.shape[0])
    return t.mean(dim=0).cpu().numpy(), Vh[:kk].contiguous().cpu().numpy()


def _mlp_fold_r2(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    *,
    layers: list[int],
    n_draws: int,
    folds: int,
    seed: int,
    pca_k: int = 64,
    max_epochs: int = 300,
    device: str | None = None,
) -> dict:
    """Batched 5-fold group-CV MLP R^2 per layer, obs + group-blocked null draws.

    Reproduces fit_h.mlp_fit_predict's recipe EXACTLY at the call site (the r1
    G1b recipe-mismatch fix): X standardized on the FULL fold-train stats
    (ddof=0, parent line ``xsd = Xtr.std(0) + 1e-6``) BEFORE the rng(42) 10%
    val split and applied to train/val/eval (``standardize_inputs=False`` in
    the batched fitter), TARGET-side PCA-``pca_k`` basis on the full fold-train
    target (skipped when the target dim <= pca_k, matching the parent;
    device-routed torch SVD — the r1 serial-CPU-battery fix), patience-20
    early stopping with the parent's 1e-6 improvement threshold
    (``patience=20``), AdamW lr 1e-3 <=300 epochs, MSE. (Layer x draw) members
    batch per fold through vectorized_mlp_skill.fit_batched_split_mlp; the
    remaining parent delta is the init draw (per-member key-seeded vs the
    parent's global manual_seed(42)) — covered by the G1b 0.02 tolerance and
    pinned by tests/test_issue931_mlp_parity.py.
    """
    from explore_persona_space.analysis.vectorized_mlp_skill import (
        SplitMLPGroup,
        fit_batched_split_mlp,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_ids = fit825._cv_folds(groups, folds, seed)
    rng = np.random.default_rng(seed + 13)
    uniq_g, inv = np.unique(np.asarray(groups), return_inverse=True)
    row_of = [np.flatnonzero(inv == k) for k in range(len(uniq_g))]

    def _group_perm() -> np.ndarray:
        gp = rng.permutation(len(uniq_g))
        return np.concatenate([row_of[k] for k in gp])

    perms = [np.arange(len(groups))] + [_group_perm() for _ in range(n_draws)]
    ss = {(li, d): [0.0, 0.0] for li in layers for d in range(len(perms))}
    for k in range(folds):
        te = fold_ids == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 20:
            continue
        member_groups, member_meta = [], []
        for li in layers:
            for d, perm in enumerate(perms):
                Yp = Y[perm]
                Xtr = X[tr, li, :].astype(np.float32)
                Ytr_raw = Yp[tr, li, :].astype(np.float32)
                Xte = X[te, li, :].astype(np.float32)
                Yte_raw = Yp[te, li, :]
                # Parent parity: standardize X on FULL fold-train stats (ddof=0)
                # BEFORE the val split; apply the same stats to train/val/eval.
                xmu = Xtr.mean(0)
                xsd = Xtr.std(0) + 1e-6
                Xn = (Xtr - xmu) / xsd
                Xen = (Xte - xmu) / xsd
                # Parent parity: PCA basis on the FULL fold-train target; the
                # parent skips PCA entirely when the target dim <= pca_k.
                if Ytr_raw.shape[1] <= pca_k:
                    y_mu = Ytr_raw.mean(0)
                    comps = None
                    Yt = Ytr_raw - y_mu
                else:
                    y_mu, comps = _pca_basis_device(Ytr_raw, pca_k, device)
                    Yt = ((Ytr_raw - y_mu) @ comps.T).astype(np.float32)
                vr = np.random.default_rng(42)
                pm = vr.permutation(len(Xn))
                n_val = max(1, round(0.1 * len(Xn)))
                vi, ti = pm[:n_val], pm[n_val:]
                member_groups.append(
                    SplitMLPGroup(
                        key=("i931mlp", int(li), int(d), int(k)),
                        X_train=Xn[ti],
                        Y_train=Yt[ti].astype(np.float32),
                        X_eval=Xen,
                        X_val=Xn[vi],
                        Y_val=Yt[vi].astype(np.float32),
                    )
                )
                member_meta.append((li, d, y_mu, comps, Yte_raw))
        res = fit_batched_split_mlp(
            member_groups,
            seed=42,
            max_epochs=max_epochs,
            device=device,
            loss="mse",
            standardize_inputs=False,
            patience=20,
        )
        for (li, d, y_mu, comps, Yte_raw), grp in zip(member_meta, member_groups, strict=True):
            pred_pca = res.preds_by_key[grp.key]
            pred = (pred_pca @ comps + y_mu) if comps is not None else (pred_pca + y_mu)
            true = Yte_raw.astype(np.float64)
            mu = true.mean(0)
            ss[(li, d)][0] += float(((true - pred) ** 2).sum())
            ss[(li, d)][1] += float(((true - mu) ** 2).sum())
    out: dict[str, dict] = {}
    for li in layers:
        obs = 1.0 - ss[(li, 0)][0] / ss[(li, 0)][1] if ss[(li, 0)][1] > 1e-12 else float("nan")
        nulls = [
            1.0 - ss[(li, d)][0] / ss[(li, d)][1] if ss[(li, d)][1] > 1e-12 else float("nan")
            for d in range(1, len(perms))
        ]
        out[str(li)] = {"r2_obs": obs, "r2_null": nulls}
    return out


def run_mlp_secondary(results: dict, args) -> None:
    """P3b: MLP secondary on armA_within / armB_within / armC_sep (frozen layers)."""
    out = {}
    for cell_id in ("armA_within", "armB_within", "armC_sep"):
        if cell_id not in results:
            continue
        xy = results[cell_id]["xy"]
        fl = frozen_layers(xy["X"].shape[1])
        print(f"[i931-p3b] MLP secondary {cell_id} layers={fl}")
        out[cell_id] = _mlp_fold_r2(
            xy["X"],
            xy["Y"],
            xy["group_ids"],
            layers=fl,
            n_draws=5,
            folds=args.folds,
            seed=args.seed,
            max_epochs=50 if args.smoke else 300,
        )
    common.write_json(
        args.out_dir / "mlp_secondary.json",
        {"metadata": common.metadata(SCRIPT, args.seed, 0), "cells": out},
    )


def run_g1b_parity(chat_xy: dict, args) -> None:
    """G1b: batched multihead fitter vs fit_h.mlp_fit_predict on chat_ref @ L19."""
    from explore_persona_space.experiments.issue_779.fit_h import mlp_fit_predict

    hl = headline_layer(chat_xy["X"].shape[1])
    X, Y, groups = chat_xy["X"], chat_xy["Y"], chat_xy["group_ids"]
    fold_ids = fit825._cv_folds(groups, args.folds, args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 50 if args.smoke else 300
    # Parent (serial reference; ONE parity fit — a single bounded call, not a
    # many-cell loop): 5-fold CV with the parent's own mlp_fit_predict.
    ss_res = ss_tot = 0.0
    for k in range(args.folds):
        te = fold_ids == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 20:
            continue
        pred = mlp_fit_predict(
            X[tr, hl, :], Y[tr, hl, :], X[te, hl, :], device=device, max_epochs=max_epochs
        )
        true = Y[te, hl, :].astype(np.float64)
        mu = true.mean(0)
        ss_res += float(((true - pred) ** 2).sum())
        ss_tot += float(((true - mu) ** 2).sum())
    r2_parent = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    batched = _mlp_fold_r2(
        X,
        Y,
        groups,
        layers=[hl],
        n_draws=0,
        folds=args.folds,
        seed=args.seed,
        max_epochs=max_epochs,
        device=device,
    )
    r2_batched = batched[str(hl)]["r2_obs"]
    delta = abs(r2_batched - r2_parent)
    payload = {
        "metadata": common.metadata(SCRIPT, args.seed, int(X.shape[0])),
        "layer": hl,
        "r2_parent_fit_h": r2_parent,
        "r2_batched_multihead": r2_batched,
        "abs_delta": delta,
        "tolerance": 0.02,
        "smoke": bool(args.smoke),
        "pass": bool(delta <= 0.02),
    }
    common.write_json(args.out_dir / "mlp_parity.json", payload)
    if not args.smoke and delta > 0.02:
        print(f"[i931-p3b] G1b FAIL: |dR2|={delta:.4f} > 0.02", file=sys.stderr)
        raise SystemExit(6)


def run_chat_gates(results: dict, args) -> None:
    """chat_ref block: G1 gate + matched-n power curve + optional G1b parity.

    Factored out of main (r1 Major: C901 complexity 16 > 15 in ``main``).
    """
    chat_xy = results["chat_ref"]["xy"]
    n_chat = chat_xy["X"].shape[0]
    g1_gate(
        np.asarray(results["chat_ref"]["payload"]["r2_per_layer_obs"]),
        args.out_dir,
        smoke=args.smoke,
        seed=args.seed,
        n=n_chat,
    )
    # Matched-n power curve: ns = {1000, 2000, n_A, n_B} (dedup, <= n_chat).
    ns = {1000, 2000}
    for arm in ("armA", "armB"):
        if f"{arm}_within" in results:
            ns.add(int(results[f"{arm}_within"]["xy"]["X"].shape[0]))
    ns = sorted(v for v in ns if v <= n_chat) or [n_chat]
    fit825.run_power_curve(
        {"X": chat_xy["X"], "Y": chat_xy["Y"], "conv_ids": chat_xy["group_ids"]},
        args.out_dir,
        n_folds=args.folds,
        seed=args.seed,
        ns=ns,
        out_name="power_curve_chat.json",
    )
    if args.g1b:
        print("[phase=p3b_mlp] G1b MLP parity")
        run_g1b_parity(chat_xy, args)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chat_dir = args.chat_store_dir or (args.data_dir / "chat_store")
    print("[phase=p3_fits] fit battery")
    if args.fabricate_chat_smoke:
        fl, fd = (int(v) for v in args.fabricate_dims.split(","))
        fabricate_chat_smoke(chat_dir, layers=fl, dim=fd)
        # Rebind the loader's layer-axis assert to the smoke dims (the #825
        # tiny-model pattern) — internal-consistency checks stay active.
        fit825.EXPECTED_LAYERS = fl
    elif args.stage_chat_store:
        stage_chat_store(chat_dir, args.out_dir)
    if args.stage_only:
        print("[i931-p3] --stage-only: chat store staged; exiting before any fit")
        return 0

    cells = list(ALL_CELLS) if args.cells == "all" else [c.strip() for c in args.cells.split(",")]
    for c in cells:
        assert c in ALL_CELLS, f"unknown cell {c}"

    store_root = args.data_dir / "store"
    stores: dict[str, dict] = {}
    for regime in ("armA", "armB", "armC"):
        if any(c.startswith(regime) for c in cells):
            stores[regime] = load_regime_store(store_root / regime, regime)
    if args.smoke and stores:
        # Tiny-model smoke: rebind the reused loader's layer-axis assert to
        # the smoke store dims (covers the dispatcher path, where the chat
        # store was fabricated by the separate --stage-only job).
        first = next(iter(stores.values()))
        fit825.EXPECTED_LAYERS = int(first["arrays"]["y"].shape[1])

    results: dict[str, dict] = {}
    for cell_id in cells:
        xy = assemble_cell(cell_id, stores, chat_dir)
        results[cell_id] = fit_cell(cell_id, xy, args)

    # dR2_char (H3) per arm where both cells ran.
    for arm in ("armA", "armB"):
        if f"{arm}_within" in results and f"{arm}_swap" in results:
            delta_char(arm, results[f"{arm}_within"], results[f"{arm}_swap"], args)

    if "chat_ref" in results:
        run_chat_gates(results, args)
    if args.mlp:
        print("[phase=p3b_mlp] MLP secondary")
        run_mlp_secondary(results, args)
    print("[i931-p3] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
