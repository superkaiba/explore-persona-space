#!/usr/bin/env python3
"""Issue #1073 free-analysis follow-up: MLP + kernel-ridge (KRR) fits per decode arm.

Closes the linear-vs-nonlinear gap #1073 left open. #1073 fit the context->answer
map with the shared-Gram GCV/val-selected RIDGE only, across its four decode arms
(``avg10`` / ``greedy`` / ``stoch1_old`` / ``stoch1_new``). The parent #779 fitter
fair-comparison (FFC) tested MLP + Nystrom RBF KRR but ONLY on the single-stochastic
target. This driver reruns the FFC MLP + KRR on ALL of #1073's decode arms at the
frozen read-out layers {14,17,19,26,27}, cx_last input, on the SAME 3600/400/1000
seed-42 split #1073's ``val_lambda_robustness`` used (nk=5000, no coverage drops ->
the split is byte-identical to the FFC 3600/400/1000), so the ridge column is
directly comparable.

Reuse (never re-implement): every fitter internal is imported from
``issue779_fitter_fair_comparison`` — the shared-factorization GCV/val ridge
(``_factorize``/``_gcv_solve``/``_vty_ymu``/``_apply``/``_cross_kernel``), the Nystrom
RBF KRR (``krr_select_predict`` + ``nystrom_features``/``_feature_ridge_multi_lambda``
for the arm-shared-Phi variant), and the batched padded-bmm AdamW MLP
(``batched_mlp_fit``/``run_mlp_battery``/``MLPGroup``). Hyperparameters are the FFC
constants imported verbatim (``KRR_*``, ``MLP_*``, ``LAMBDAS``, ``MLP_SELECT_LAYER``).

CPU-compute note (0 GPU-h inline round): the full FFC MLP width-sweep
{512,3584,8192} with a full-dim (3584) output head is FLOP/bandwidth-bound and
projects ~2-3h on this CPU VM (measured 1-cell pilot; poor thread scaling). So the
MLP here runs the CPU-FEASIBLE width-512 subset (both FFC lrs, val-selected per arm
at L19). The width-{3584,8192} MLP is flagged NEEDS-GPU in the output; the FFC's own
width-8192 single-stochastic result (test R2 0.688 < ridge 0.705) is carried as the
upper-bound MLP reference. KRR (a nonlinear kernel method) is closed-form and runs at
FULL fidelity on every arm — it is the decisive cheap nonlinearity test.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind at import (#847)

import issue779_percontext_recon as PR  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue779_fitter_fair_comparison import (  # noqa: E402
    KRR_GAMMA_MULT,
    KRR_LAMBDAS,
    KRR_LANDMARKS,
    LAMBDAS,
    MLP_LRS,
    MLP_MAX_EPOCHS,
    MLP_SELECT_LAYER,
    MLP_WIDTHS,
    MLPGroup,
    _apply,
    _cross_kernel,
    _factorize,
    _feature_ridge_multi_lambda,
    _gcv_solve,
    _vty_ymu,
    median_heuristic_gamma,
    nystrom_features,
    run_mlp_battery,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1073_mlp_krr")

# ── config ──────────────────────────────────────────────────────────────────────
ARMS = ("avg10", "greedy", "stoch1_old", "stoch1_new")
READOUT_LAYERS = (14, 17, 19, 26, 27)
SPLIT_SEED = 42  # I.FOLD_SEED_SCIENCE — the val_lambda_robustness / FFC split seed
SHUFFLE_SEED = 0  # issue1073_fits.SHUFFLE_SEED — the shuffled-pairing null permutation
MLP_WIDTH_CPU = 512  # the CPU-feasible subset of the FFC width sweep {512,3584,8192}
# FFC single-stochastic (v_x) reference numbers at L19 (fair_comparison.json, last input).
FFC_L19_SINGLE_STOCH = {
    "ridge_test_r2": 0.7054149303865586,
    "krr_test_r2": 0.7118060857224284,
    "mlp_test_r2": 0.688,  # val-selected width=8192 lr=3e-4 (the width the CPU run cannot reach)
    "mlp_selected": {"width": 8192, "lr": 3e-4, "val_r2": 0.6960072246199515},
}
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
BUNDLE_PATH = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
BUNDLE_REV = "037fcbb210bc52c459959b0746cc268fe08bae96"
RED_PREFIX = "issue1073_decode_regime/analysis_tensors/reductions"
RED_REV = "fb4fe90fdd836ba2efd896b90c17e6b42f143d21"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _repro_meta(extra: dict | None = None) -> dict:
    meta = {
        "issue": 1073,
        "git_commit": _git_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "bundle_revision": BUNDLE_REV,
        "reductions_revision": RED_REV,
    }
    if extra:
        meta.update(extra)
    return meta


# ── data loading ──────────────────────────────────────────────────────────────


def _fetch(path_in_repo: str, revision: str, dl_dir: Path) -> Path:
    dest = dl_dir / path_in_repo
    if dest.exists():
        return dest
    from huggingface_hub import hf_hub_download

    dl_dir.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(
        repo_id=HF_DATA_REPO,
        filename=path_in_repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dl_dir),
    )
    return Path(got)


def load_data(dl_dir: Path) -> dict:
    """Load cx_last + the four arm target tensors (mmap), restricted to the common set.

    The common set is read from the committed ``heldout_recon_arms.json``
    (``common_set.common_index``); for #1073 it is the full [0,5000) set (n_kept=5000,
    no coverage drops), so the 3600/400/1000 seed-42 split below is byte-identical to
    the FFC fixed split and #1073's ``val_lambda_robustness`` split.
    """
    bundle = torch.load(
        _fetch(BUNDLE_PATH, BUNDLE_REV, dl_dir), mmap=True, weights_only=False, map_location="cpu"
    )
    vbar10 = torch.load(
        _fetch(f"{RED_PREFIX}/vbar10.pt", RED_REV, dl_dir),
        mmap=True,
        weights_only=False,
        map_location="cpu",
    )["tensor"]
    v_greedy = torch.load(
        _fetch(f"{RED_PREFIX}/v_greedy.pt", RED_REV, dl_dir),
        mmap=True,
        weights_only=False,
        map_location="cpu",
    )["tensor"]
    stoch1_new = torch.load(
        _fetch(f"{RED_PREFIX}/stoch1_new.pt", RED_REV, dl_dir),
        mmap=True,
        weights_only=False,
        map_location="cpu",
    )["tensor"]

    committed = json.loads(
        (PROJECT_ROOT / "eval_results" / "issue_1073" / "heldout_recon_arms.json").read_text()
    )
    keep = np.asarray(committed["common_set"]["common_index"], dtype=np.int64)
    ridge_baseline = committed["val_lambda_robustness"]

    n_ctx, n_layers, hidden = tuple(vbar10.shape)
    assert bundle["cx_last"].shape == (n_ctx, n_layers, hidden), bundle["cx_last"].shape
    assert n_layers >= 28, n_layers
    return {
        "cx_last": bundle["cx_last"],
        "reductions": {
            "avg10": vbar10,
            "greedy": v_greedy,
            "stoch1_old": bundle["v_x"],
            "stoch1_new": stoch1_new,
        },
        "keep": keep,
        "ridge_baseline": ridge_baseline,
        "n_ctx": n_ctx,
        "n_layers": n_layers,
        "hidden": hidden,
    }


def _layer(t: torch.Tensor, li: int, keep: np.ndarray) -> np.ndarray:
    """(n_kept, H) float64 slice of an (N, L, H) tensor at layer li on the common set."""
    return t[:, li, :].to(torch.float64).numpy()[keep]


def build_split(nk: int) -> dict:
    """The FFC / val_lambda_robustness 3600/400/1000 permutation split at seed 42.

    Reproduced verbatim from ``issue1073_fits.val_lambda_robustness``: proportional
    0.20 test / 0.08 val on the common set (nk=5000 -> 1000/400/3600). No duplicate
    clustering — the FFC fixed split does not dedup either (comparability caveat)."""
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(nk)
    n_te = max(1, round(0.2 * nk))
    n_val = max(1, round(0.08 * nk))
    te = np.sort(perm[:n_te])
    val = np.sort(perm[n_te : n_te + n_val])
    tr = np.sort(perm[n_te + n_val :])
    assert len(tr) > 1, (nk, len(tr))
    return {"tr": tr, "val": val, "te": te}


# ── ridge (recompute on the split; cross-checked vs committed val_lambda_robustness) ─


def fit_ridge(X: dict[int, np.ndarray], Y: dict[str, dict[int, np.ndarray]], split, dev) -> dict:
    """Val-selected + GCV ridge per (arm, layer) off ONE shared factorization per layer."""
    tr, val, te = split["tr"], split["val"], split["te"]
    out: dict = {}
    for li in READOUT_LAYERS:
        fact = _factorize(X[li][tr], dev)
        kval = _cross_kernel(fact, X[li][val])
        kte = _cross_kernel(fact, X[li][te])
        for arm in ARMS:
            y = Y[arm][li]
            vty, ymu = _vty_ymu(fact, y[tr])
            best_lam, best_vr2 = float(LAMBDAS[0]), -np.inf
            for lam in LAMBDAS:
                vr2 = PR._pooled_r2(_apply(fact, float(lam), vty, ymu, kval), y[val])
                if np.isfinite(vr2) and vr2 > best_vr2:
                    best_vr2, best_lam = float(vr2), float(lam)
            r2_val_sel = PR._pooled_r2(_apply(fact, best_lam, vty, ymu, kte), y[te])
            gcv_lam, vty2, ymu2 = _gcv_solve(fact, y[tr])
            r2_gcv = PR._pooled_r2(_apply(fact, gcv_lam, vty2, ymu2, kte), y[te])
            out.setdefault(arm, {})[li] = {
                "test_r2": float(r2_val_sel),
                "test_r2_gcv": float(r2_gcv),
                "val_r2": float(best_vr2),
                "val_lambda": best_lam,
                "gcv_lambda": float(gcv_lam),
            }
        logger.info("[ridge] L%d done", li)
    return out


def check_ridge_parity(ridge: dict, baseline: dict) -> dict:
    """Confirm the recomputed ridge matches the committed val_lambda_robustness
    (proves split parity). Reports the max abs diff; a tight match => same split."""
    diffs = []
    for arm in ARMS:
        for li in READOUT_LAYERS:
            ref = baseline.get(f"L{li}", {}).get(arm)
            if ref is None:
                continue
            diffs.append(abs(ridge[arm][li]["test_r2"] - ref["r2_test_val_selected"]))
            diffs.append(abs(ridge[arm][li]["test_r2_gcv"] - ref["r2_test_gcv"]))
    max_abs = float(max(diffs)) if diffs else float("nan")
    return {
        "max_abs_diff_vs_committed_val_lambda_robustness": max_abs,
        "split_parity_confirmed": bool(max_abs < 1e-6),
        "n_compared": len(diffs),
    }


# ── KRR (Nystrom RBF; Phi shared across arms per (layer, gamma)) ───────────────


def fit_krr(X: dict[int, np.ndarray], Y: dict[str, dict[int, np.ndarray]], split, dev) -> dict:
    """Val-selected (gamma, lambda) Nystrom RBF KRR per (arm, layer).

    The Nystrom feature map depends only on X, so per (layer, gamma) the landmark
    subsample + Phi_tr/Phi_val/Phi_te are computed ONCE and reused across all four
    arms (the big win vs calling ``krr_select_predict`` per arm). Landmarks / gammas /
    lambdas are the FFC constants; equivalent to ``krr_select_predict`` per arm.
    """
    tr, val, te = split["tr"], split["val"], split["te"]
    out: dict = {}
    for li in READOUT_LAYERS:
        Xtr, Xval, Xte = X[li][tr], X[li][val], X[li][te]
        rng = np.random.default_rng(SPLIT_SEED)  # matches krr_select_predict's seed
        lm_idx = rng.choice(Xtr.shape[0], size=min(KRR_LANDMARKS, Xtr.shape[0]), replace=False)
        lm = Xtr[lm_idx]
        base_gamma = median_heuristic_gamma(Xtr, np.random.default_rng(SPLIT_SEED + 1))
        # Precompute Phi per gamma ONCE (shared across arms).
        phis = {}
        for gm in KRR_GAMMA_MULT:
            gamma = base_gamma * gm
            phis[gm] = (
                gamma,
                nystrom_features(Xtr, lm, gamma, dev),
                nystrom_features(Xval, lm, gamma, dev),
                nystrom_features(Xte, lm, gamma, dev),
            )
        for arm in ARMS:
            ytr, yval, yte = Y[arm][li][tr], Y[arm][li][val], Y[arm][li][te]
            best = None
            for gm in KRR_GAMMA_MULT:
                gamma, phi_tr, phi_val, phi_te = phis[gm]
                preds = _feature_ridge_multi_lambda(phi_tr, ytr, [phi_val, phi_te], KRR_LAMBDAS)
                for li_lam, lam in enumerate(KRR_LAMBDAS):
                    pred_val, pred_te = preds[li_lam]
                    vr2 = PR._pooled_r2(pred_val, yval)
                    if best is None or (np.isfinite(vr2) and vr2 > best["val_r2"]):
                        best = {
                            "gamma_mult": float(gm),
                            "gamma": float(gamma),
                            "lambda": float(lam),
                            "val_r2": float(vr2),
                            "test_r2": float(PR._pooled_r2(pred_te, yte)),
                        }
            out.setdefault(arm, {})[li] = {
                "test_r2": best["test_r2"],
                "val_r2": best["val_r2"],
                "selected": {k: best[k] for k in ("gamma_mult", "gamma", "lambda")},
            }
        logger.info("[krr] L%d done", li)
    return out


# ── MLP (batched padded-bmm AdamW; width-512 CPU-feasible subset) ──────────────


def fit_mlp(
    X: dict[int, np.ndarray],
    Y: dict[str, dict[int, np.ndarray]],
    shuffle_target: dict[int, np.ndarray],
    split,
    dev,
    max_epochs: int,
    ckpt_dir: Path,
) -> dict:
    """Width-512 MLP per (arm, layer): lr selected on val at L19, test-read at 5 layers.

    Mirrors the FFC D1 MLP path (select recipe on val at ``MLP_SELECT_LAYER``, then
    read test at each layer with the selected recipe) restricted to width=512. One
    ``shuffle_null`` MLP cell at L19 guards against the net fitting noise.

    CHECKPOINTED + PER-LAYER: the test reads run one layer at a time (≤4 groups per
    ``batched_mlp_fit`` -> ~1.5 GB peak vs ~5 GB for the 21-group partition that
    earlyoom killed on the loaded VM), each layer persisted to ``ckpt_dir`` so a
    kill/relaunch resumes instead of restarting.
    """
    tr, val, te = split["tr"], split["val"], split["te"]
    sel_layer = (
        MLP_SELECT_LAYER
        if MLP_SELECT_LAYER in READOUT_LAYERS
        else READOUT_LAYERS[len(READOUT_LAYERS) // 2]
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _load(name):
        p = ckpt_dir / name
        return json.loads(p.read_text()) if p.exists() else None

    def _save(name, obj):
        p = ckpt_dir / name
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj))
        tmp.replace(p)

    # ---- lr selection at sel_layer (width 512, both FFC lrs) — one lr-partition at a time ----
    selection = _load("selection.json")
    if selection is None:
        selection = {"layer": int(sel_layer), "width": MLP_WIDTH_CPU, "per_arm": {}, "grid": []}
        for lr in MLP_LRS:  # one (width,lr) partition at a time bounds peak RAM
            groups = [
                MLPGroup(
                    ("sel", arm, lr), X[sel_layer][tr], Y[arm][sel_layer][tr], MLP_WIDTH_CPU, lr
                )
                for arm in ARMS
            ]
            fits = run_mlp_battery(groups, dev=dev, max_epochs=max_epochs)
            for arm in ARMS:
                r = fits[("sel", arm, lr)]
                vr2 = float(PR._pooled_r2(r.predict(X[sel_layer][val]), Y[arm][sel_layer][val]))
                selection["grid"].append(
                    {"arm": arm, "lr": lr, "val_r2": vr2, "epochs_ran": int(r.epochs_ran)}
                )
        for arm in ARMS:
            best = max((g for g in selection["grid"] if g["arm"] == arm), key=lambda g: g["val_r2"])
            selection["per_arm"][arm] = {"lr": best["lr"], "val_r2": best["val_r2"]}
            logger.info("[mlp-sel] %s: lr=%.0e val_r2=%.4f", arm, best["lr"], best["val_r2"])
        _save("selection.json", selection)
    selected_lr = {arm: selection["per_arm"][arm]["lr"] for arm in ARMS}

    # ---- test reads: ONE layer at a time (checkpointed), + one null cell at sel_layer ----
    out: dict = {"selection": selection, "per_arm": {arm: {} for arm in ARMS}, "null": {}}
    for li in READOUT_LAYERS:
        lres = _load(f"test_L{li}.json")
        if lres is None:
            groups = [
                MLPGroup(
                    ("test", arm, li), X[li][tr], Y[arm][li][tr], MLP_WIDTH_CPU, selected_lr[arm]
                )
                for arm in ARMS
            ]
            fits = run_mlp_battery(groups, dev=dev, max_epochs=max_epochs)
            lres = {}
            for arm in ARMS:
                r = fits[("test", arm, li)]
                lres[arm] = {
                    "test_r2": float(PR._pooled_r2(r.predict(X[li][te]), Y[arm][li][te])),
                    "lr": selected_lr[arm],
                    "width": MLP_WIDTH_CPU,
                    "epochs_ran": int(r.epochs_ran),
                }
            _save(f"test_L{li}.json", lres)
            logger.info("[mlp-test] L%d done (avg10 test_r2=%.4f)", li, lres["avg10"]["test_r2"])
        for arm in ARMS:
            out["per_arm"][arm][li] = lres[arm]

    null = _load("null.json")
    if null is None:
        g = [
            MLPGroup(
                ("null", sel_layer),
                X[sel_layer][tr],
                shuffle_target[sel_layer][tr],
                MLP_WIDTH_CPU,
                MLP_LRS[0],
            )
        ]
        rn = run_mlp_battery(g, dev=dev, max_epochs=max_epochs)[("null", sel_layer)]
        null = {
            "layer": int(sel_layer),
            "test_r2": float(
                PR._pooled_r2(rn.predict(X[sel_layer][te]), shuffle_target[sel_layer][te])
            ),
            "width": MLP_WIDTH_CPU,
            "lr": MLP_LRS[0],
        }
        _save("null.json", null)
        logger.info("[mlp-null] L%d test_r2=%.4f", sel_layer, null["test_r2"])
    out["null"] = null
    return out


# ── null cells for ridge / krr (cheap; all layers) ────────────────────────────


def fit_null_linear(X, shuffle_target, split, dev) -> dict:
    """Ridge + KRR on the shuffled-pairing null target at every read-out layer."""
    tr, val, te = split["tr"], split["val"], split["te"]
    out: dict = {"ridge": {}, "krr": {}}
    for li in READOUT_LAYERS:
        fact = _factorize(X[li][tr], dev)
        kval, kte = _cross_kernel(fact, X[li][val]), _cross_kernel(fact, X[li][te])
        y = shuffle_target[li]
        vty, ymu = _vty_ymu(fact, y[tr])
        best_lam, best_vr2 = float(LAMBDAS[0]), -np.inf
        for lam in LAMBDAS:
            vr2 = PR._pooled_r2(_apply(fact, float(lam), vty, ymu, kval), y[val])
            if np.isfinite(vr2) and vr2 > best_vr2:
                best_vr2, best_lam = float(vr2), float(lam)
        out["ridge"][li] = {
            "test_r2": float(PR._pooled_r2(_apply(fact, best_lam, vty, ymu, kte), y[te]))
        }
        # KRR null (median gamma, val-selected lambda)
        rng = np.random.default_rng(SPLIT_SEED)
        lm = X[li][tr][
            rng.choice(
                X[li][tr].shape[0], size=min(KRR_LANDMARKS, X[li][tr].shape[0]), replace=False
            )
        ]
        gamma = median_heuristic_gamma(X[li][tr], np.random.default_rng(SPLIT_SEED + 1))
        phi_tr = nystrom_features(X[li][tr], lm, gamma, dev)
        phi_val = nystrom_features(X[li][val], lm, gamma, dev)
        phi_te = nystrom_features(X[li][te], lm, gamma, dev)
        preds = _feature_ridge_multi_lambda(phi_tr, y[tr], [phi_val, phi_te], KRR_LAMBDAS)
        bk = max(
            ((PR._pooled_r2(pv, y[val]), PR._pooled_r2(pt, y[te])) for pv, pt in preds),
            key=lambda z: z[0] if np.isfinite(z[0]) else -np.inf,
        )
        out["krr"][li] = {"test_r2": float(bk[1])}
    return out


# ── main ────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1073 MLP+KRR per decode arm.")
    ap.add_argument(
        "--dl-dir", default="/mnt/eps-data/thomasjiralerspong/issue1073_mlp_krr_dl/files"
    )
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval_results" / "issue_1073" / "mlp_krr_decode_regime.json"),
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--mlp-max-epochs", type=int, default=MLP_MAX_EPOCHS)
    ap.add_argument(
        "--smoke-layers",
        type=int,
        default=0,
        help=">0: use only the first K read-out layers (smoke)",
    )
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    global READOUT_LAYERS
    if args.smoke_layers:
        READOUT_LAYERS = READOUT_LAYERS[: args.smoke_layers]

    t0 = time.time()
    data = load_data(Path(args.dl_dir))
    keep = data["keep"]
    nk = keep.size
    split = build_split(nk)
    logger.info(
        "[load] nk=%d split tr/val/te=%d/%d/%d in %.1fs",
        nk,
        len(split["tr"]),
        len(split["val"]),
        len(split["te"]),
        time.time() - t0,
    )

    # Materialize the layer slices once (float64), shared by all fitters.
    X = {li: _layer(data["cx_last"], li, keep) for li in READOUT_LAYERS}
    Y = {
        arm: {li: _layer(data["reductions"][arm], li, keep) for li in READOUT_LAYERS}
        for arm in ARMS
    }
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(nk)
    shuffle_target = {
        li: Y["avg10"][li][perm] for li in READOUT_LAYERS
    }  # avg10[perm], the FFC null

    # Linear stage (ridge + KRR + null) is fast (~4 min) and is the decisive result;
    # checkpoint it to disk the moment it completes so a mid-run MLP kill (earlyoom on
    # the loaded VM) never forfeits it. Resume loads it and skips straight to the MLP.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path.parent / ".mlp_krr_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lin_ckpt = ckpt_dir / "linear_stage.json"

    def _to_int_layers(d):
        return {arm: {int(li): v for li, v in layers.items()} for arm, layers in d.items()}

    if lin_ckpt.exists():
        blob = json.loads(lin_ckpt.read_text())
        ridge = _to_int_layers(blob["ridge"])
        krr = _to_int_layers(blob["krr"])
        null_lin = {
            "ridge": {int(li): v for li, v in blob["null_lin"]["ridge"].items()},
            "krr": {int(li): v for li, v in blob["null_lin"]["krr"].items()},
        }
        parity = blob["parity"]
        logger.info("[linear-stage] resumed from checkpoint %s", lin_ckpt)
    else:
        ridge = fit_ridge(X, Y, split, dev)
        parity = check_ridge_parity(ridge, data["ridge_baseline"])
        logger.info("[ridge-parity] %s", json.dumps(parity))
        krr = fit_krr(X, Y, split, dev)
        null_lin = fit_null_linear(X, shuffle_target, split, dev)
        tmp = lin_ckpt.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "ridge": {arm: {str(li): v for li, v in d.items()} for arm, d in ridge.items()},
                    "krr": {arm: {str(li): v for li, v in d.items()} for arm, d in krr.items()},
                    "null_lin": {
                        "ridge": {str(li): v for li, v in null_lin["ridge"].items()},
                        "krr": {str(li): v for li, v in null_lin["krr"].items()},
                    },
                    "parity": parity,
                }
            )
        )
        tmp.replace(lin_ckpt)
        logger.info("[linear-stage] checkpointed -> %s", lin_ckpt)
    mlp = fit_mlp(X, Y, shuffle_target, split, dev, args.mlp_max_epochs, ckpt_dir)

    # Assemble the per (arm x fitter x layer) table.
    table: dict = {}
    for arm in ARMS:
        table[arm] = {}
        for li in READOUT_LAYERS:
            base = data["ridge_baseline"].get(f"L{li}", {}).get(arm, {})
            table[arm][str(li)] = {
                "ridge_test_r2": ridge[arm][li]["test_r2"],
                "ridge_test_r2_gcv": ridge[arm][li]["test_r2_gcv"],
                "krr_test_r2": krr[arm][li]["test_r2"],
                "mlp_w512_test_r2": mlp["per_arm"][arm][li]["test_r2"],
                "ridge_val_r2": ridge[arm][li]["val_r2"],
                "krr_val_r2": krr[arm][li]["val_r2"],
                "committed_ridge_val_selected": base.get("r2_test_val_selected"),
            }

    result = {
        "description": (
            "MLP + Nystrom RBF KRR vs ridge per #1073 decode arm at the frozen read-out "
            "layers, cx_last input, on the FFC 3600/400/1000 seed-42 split "
            "(= val_lambda_robustness split; nk=5000)."
        ),
        "arms": list(ARMS),
        "readout_layers": list(READOUT_LAYERS),
        "split": {
            "n_train": len(split["tr"]),
            "n_val": len(split["val"]),
            "n_test": len(split["te"]),
            "seed": SPLIT_SEED,
        },
        "duplicate_handling": (
            "NONE — random permutation split, matching the FFC fixed split (which also does "
            "not dedup); #1073's 477-rows-in-clusters figure applies to the 5-fold clustered CV, "
            "not this val-selected split. Comparability-over-correction caveat."
        ),
        "table": table,
        "ridge_parity_vs_committed": parity,
        "krr_full": krr,
        "mlp_width512": mlp,
        "null_shuffle_pairing": {
            "seed": SHUFFLE_SEED,
            "ridge_by_layer": {str(li): null_lin["ridge"][li]["test_r2"] for li in READOUT_LAYERS},
            "krr_by_layer": {str(li): null_lin["krr"][li]["test_r2"] for li in READOUT_LAYERS},
            "mlp_L19": mlp["null"],
        },
        "mlp_width_sweep_status": {
            "widths_run": [MLP_WIDTH_CPU],
            "widths_needing_gpu": [w for w in MLP_WIDTHS if w != MLP_WIDTH_CPU],
            "note": (
                "Full FFC MLP width sweep {512,3584,8192} with a full-dim (3584) output head is "
                "FLOP/bandwidth-bound (~2-3h on this CPU VM, measured; poor thread scaling). Only "
                "width-512 was run inline (0 GPU-h). The FFC's own single-stochastic width-8192 "
                "MLP (test R2 0.688 < ridge 0.705 at L19) is the upper-bound MLP reference below."
            ),
        },
        "ffc_single_stochastic_reference_L19": FFC_L19_SINGLE_STOCH,
        "metadata": _repro_meta({"threads": args.threads, "device": args.device}),
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out_path)
    logger.info("[done] wrote %s in %.1f min", out_path, (time.time() - t0) / 60.0)
    # Machine-readable completion sentinel for in-turn polling.
    (out_path.parent / ".mlp_krr_done").write_text(str(time.time()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
