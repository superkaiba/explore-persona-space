"""Issue #1689 follow-up `derived-vs-free-answer-map` — items 1-6 (+ item 9 transfer leg).

Per ordered condition pair (within-model AND base<->instruct cross-model), both
mapping arms (prefix / context), this battery tests the mentor-posed
shared-readout conjugation hypothesis: with one readout y = xW + b per
condition and one affine context transfer x_T = x_S M + a, the answer-space
map is fully DERIVED —

  row convention:  y_T_hat = (y_S - b_S) @ W_S_pinv @ M @ W_S + a @ W_S + b_S
                   (== column-convention W M W^+ with bias -W M W^+ b + W a + b)

Three answer-map models per (pair, arm), ALL fits train-fold-only per outer
conv-grouped fold (the battery's ONLY fitting convention; the parent ladder's
all-rows W_s appears ONLY in the Gate-1 parity check):

  b_derived   = W_S^+ M W_S     (0 free y-space params, shared readout)
  b_derived2  = W_S^+ M W_T     (0 free params, per-condition readouts)
  b_free      = ridge y_S -> y_T (d^2 + d free params; the within-form ceiling)
  identity+bias baseline + kNN retrieval per the standing mapping rule.

W_S^+ via truncated SVD at ranks {32, 128, 512, eff-rank} (all reported;
conjugation amplifies noise along weak singular directions). Verdict lattice
(plan v8 s3, FOUR disjoint classes; Class 0 excluded from verdict counts):

  free_map_uninformative    <=> R2(b_free) < R2(identity+bias)
  shared_readout_supported  <=> g1 = R2(b_derived_max)  - 0.9 R2(b_free) >= 0
  readout_changed           <=> g1 < 0 and g2 = R2(b_derived2_max) - 0.9 R2(b_free) >= 0
  transfer_map_insufficient <=> otherwise

Phases (--phase): stage | gate1 | pairs | nulls | merge | upload | write-pairs
| migrate-keys (one-shot rename of pre-fix unqualified within-model unit
checkpoints to model-qualified keys — fix round 3).
Per-unit JSON checkpoints (skip-if-meta-matches resume, parent R13 convention);
compact SVD bundles under <out-root>/bundles/. Rotation nulls (item 5) run as a
SHARED-draw pass (--phase nulls) using the parent 9a-ter Procrustes battery's
exact Haar + singular-value reduction (seeds seed*1000003+k), which is
per-draw EXACTLY equal to the verbatim two-sided-rotation formula of
issue1345_operator_comparison.raw_cosine_with_rotation_null (von Neumann trace
identity; pinned by tests/test_issue1689_derived_vs_free.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# load_dotenv() BEFORE numpy/torch (shared-VM thread caps, #847).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.issue1689_fit_ladder as fl  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from scripts.issue1689_common import (  # noqa: E402
    HEADLINE_LAYER,
    HF_DATA_PREFIX,
    K_FLOOR_LIMITED,
    LAMBDA_GRIDS,
    N_FOLDS,
    PCA_K_CAP,
    RUNG_REACHED_THRESHOLD,
    enumerate_pair_set,
)

MODEL_SLUGS = ("Qwen_Qwen2.5-7B", "Qwen_Qwen2.5-7B-Instruct")
DATA_REPO = "superkaiba1/explore-persona-space-data"
PINNED_STORE_REVISION = "d1010a25f81ce184f68a9cc0ed49bce9736b80dd"
STORE_HF_PREFIX = f"{HF_DATA_PREFIX}/analysis_tensors"
BUNDLE_HF_PREFIX = f"{HF_DATA_PREFIX}/derived_vs_free/analysis_tensors"
PARENT_INPUTS_HF_PREFIX = f"{HF_DATA_PREFIX}/parent_inputs"
# Committed parent-round eval_results inputs the wellposed leg consumes. The
# fellows/SLURM lane rsyncs only RSYNC_INCLUDE_PATHS (eval_results/ excluded
# wholesale, backends/slurm.py), so these never reach the node from git —
# they are mirrored on the data repo (--phase upload-parent-inputs, run once
# VM-side from a git checkout) and staged at leg entry
# (--phase stage-parent-inputs): the #734 upload-first remedy for the
# fellows job 15724 crash (FileNotFoundError in cmd_fence on the digest CSV).
# All paths are relative to eval_results/issue_1689/.
PARENT_INPUT_SINGLES = (
    "analyzer/dvf_unit_digest.csv",  # cmd_fence --digest-csv + paired-digest join
    "crossmodel_pairs/ladder_crossmodel_L19.json",  # cms --crossmodel-ladder-json
    "ladder/ladder_Qwen_Qwen2.5-7B_L19.json",  # cmd_merge/cms --parent-ladder-dir
    "ladder/ladder_Qwen_Qwen2.5-7B-Instruct_L19.json",
)
PARENT_INPUT_TREES = (
    "derived_vs_free_B/pairs",  # paired-digest ambient dvf units (+ fence pilot)
    "crossmodel_pairs/pairs",  # paired-digest ambient xm units
    "context_map_structure/pairs",  # paired-digest ambient cms units
    "crossmodel_pairs/crossmodel_structure/pairs",  # paired-digest ambient xms units
)
TRUNCATION_RANKS = (32, 128, 512)  # + per-fold eff-rank (scope item 2)
RANK_LABELS = ("r32", "r128", "r512", "effrank")
BATTERY_VERSION = "dvf-v1"
LOW_COMMON_FLAG = 500  # plan s8: a <500-common-row unit is flagged, not dropped
GATE1_PAIR = (
    ("Qwen_Qwen2.5-7B-Instruct", "assistant_chat"),
    ("Qwen_Qwen2.5-7B-Instruct", "assistant_naturalistic"),
)
GATE1_ATOL = 1e-3  # plan s7 Gate 1 (GPU-fp64 vs parent tolerance)


def fit_basis_of(args) -> str:
    """Resolve the fit basis ('ambient' | 'reduced'); default 'ambient'.

    getattr-guarded so pre-extension callers / test Namespaces without the
    attribute keep the parent (ambient) behavior byte-identically.
    """
    return getattr(args, "fit_basis", "ambient") or "ambient"


def compute_k_unit(
    folds: "np.ndarray", n: int, d: int, cap: int = PCA_K_CAP
) -> tuple[int, dict[int, int]]:
    """k_unit = min(cap, d, floor(min-fold n_train / 2)) over the folds that RUN.

    A fold runs iff n_train >= 3 and n_test >= 1 (the battery loop's own
    gate); the MINIMUM running-fold n_train sets ONE k per unit so
    n_train >= 2*k_unit holds on EVERY executed fold (plan v10 s4 item 1).
    The ``d`` cap is implicit in the plan's production shape (d=3584 > cap)
    and binding only for dim-limited smokes / tiny worlds — a basis cannot
    have more than d directions. Returns (k_unit, {fold: n_train}); ({},)
    when every fold is degenerate (the caller's all-folds-degenerate path).
    """
    per_fold: dict[int, int] = {}
    for k_fold in range(N_FOLDS):
        n_te = int((folds == k_fold).sum())
        n_tr = n - n_te
        if n_tr < 3 or n_te < 1:
            continue
        per_fold[k_fold] = n_tr
    if not per_fold:
        return 0, {}
    return int(min(cap, d, min(per_fold.values()) // 2)), per_fold


def _pca_basis(stacked, k: int):
    """Center + thin-SVD stacked TRAIN rows -> (mu, Q (d,k), svals_k, train_frac).

    Q = top-k right singular vectors of the centered stack (one basis per
    side SHARED across source and target — plan v10 s4 item 2); train_frac
    is the captured-variance fraction of the stack (sum s_k^2 / sum s^2).
    Test rows never reach this function (leakage discipline).
    """
    mu = stacked.mean(dim=0)
    ac = stacked - mu
    _u, s, vh = fl._svd_robust_t(ac)
    assert k <= vh.shape[0], (k, tuple(vh.shape))
    q = vh[:k].T.contiguous()
    tot = float((s**2).sum().item())
    kept = float((s[:k] ** 2).sum().item())
    return mu, q, s[:k], (kept / tot if tot > 0 else float("nan"))


def _heldout_captured_frac(stacked_te, mu, q) -> float:
    """Held-out captured-variance fraction of a train-fold basis (plan s4 item 5)."""
    ac = stacked_te - mu
    tot = float((ac**2).sum().item())
    kept = float(((ac @ q) ** 2).sum().item())
    return kept / tot if tot > 0 else float("nan")


def k_floor_limited(k: int) -> bool:
    """Report-only diagnostic label for tiny-k units (plan v10 s6; no gating)."""
    return k < K_FLOOR_LIMITED


def _git_commit() -> str:
    """Best-effort repo commit for reproducibility metadata."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _metadata() -> dict:
    import torch

    return {
        "git_commit": _git_commit(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=1)
    tmp.replace(path)


def _atomic_savez(path: Path, **arrays) -> None:
    """np.savez APPENDS .npz to non-.npz names — tmp must END in .npz (gotchas)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _haar(d: int, gen) -> "object":
    """Haar-orthogonal (d, d) fp64 sample — the parent 9a-ter battery convention
    (CPU randn from the caller's generator; QR + diag-sign fix runs on the
    generator's device stream via CPU then moves are the caller's business)."""
    import torch

    a = torch.randn(d, d, dtype=torch.float64, generator=gen)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r))


def _cos_flat(a, b) -> float:
    va, vb = a.reshape(-1), b.reshape(-1)
    return float((va @ vb) / (va.norm() * vb.norm() + 1e-12))


def regime_meta(args) -> dict:
    """Per-unit resume regime key — EVERY output-affecting knob (#722 r3 rule).

    Rotation-null draws are deliberately NOT here: nulls are a separate
    patch-in pass keyed by their own n_draws field inside the unit JSON.

    ``fit_basis`` enters the key ONLY when non-ambient: the ambient default
    keeps the meta dict byte-identical to the parent regime, so the parent's
    realized checkpoints stay resume-valid (alias-the-default convention),
    while a ``reduced`` run can never satisfy — or be satisfied by — an
    ambient checkpoint (#722 r3 every-output-affecting-knob rule).
    """
    meta = {
        "battery_version": BATTERY_VERSION,
        "layer": int(args.layer),
        "lambda_grid": str(args.lambda_grid),
        "seed": int(args.seed),
        "n_folds": int(N_FOLDS),
        "truncation_ranks": list(TRUNCATION_RANKS),
        "threshold": float(RUNG_REACHED_THRESHOLD),
        "row_limit": args.row_limit,
        "dim_limit": args.dim_limit,
    }
    if fit_basis_of(args) != "ambient":
        meta["fit_basis"] = fit_basis_of(args)
        meta["pca_k_cap"] = int(PCA_K_CAP)
    return meta


def unit_key(spec, arm: str) -> str:
    """Model-qualified per-unit checkpoint key: <model>__<src>__<tgt>__<arm>.

    Fix round 3 (#1689): the old within-model key (`fl.pair_spec_key` =
    `<src>__<tgt>`, NO model) collided across models sharing an out-root, so
    the second model's units skip-if-exists'd against the first model's files
    (241 base + 11 instruct of 504 computed; merge counted each file for BOTH
    models). Cross-model keys already embed both models via pair_spec_key
    (`m@c__m@c`) and are byte-unchanged.
    """
    (sm, _sc), (tm, _tc) = spec
    if sm == tm:
        return f"{sm}__{fl.pair_spec_key(spec)}__{arm}"
    return f"{fl.pair_spec_key(spec)}__{arm}"


def migrate_unqualified_keys(out_root: Path) -> dict:
    """One-shot rename of pre-fix UNQUALIFIED within-model per-unit checkpoints.

    The old within-model key carried no model, so two models sharing an
    out-root collided (see unit_key). Each computed unit JSON records its
    model internally (src_model/tgt_model); rename the JSON + its sibling
    bundle to the model-qualified unit_key so the surviving units are
    RETAINED under correct keys and only genuinely-missing units re-run.
    Files are NEVER deleted: unattributable files (error units without a
    model record) and renames whose target already exists are left in place
    with a log line. Idempotent — a second pass no-ops.
    """
    pairs_dir = out_root / "pairs"
    bundles_dir = out_root / "bundles"
    counts = {
        "renamed": 0,
        "already_qualified": 0,
        "unattributable": 0,
        "target_exists": 0,
        "unparseable": 0,
    }
    if not pairs_dir.exists():
        print(f"[dvf-migrate] no pairs dir at {pairs_dir} — nothing to migrate", flush=True)
        return counts
    for upath in sorted(pairs_dir.glob("*.json")):
        try:
            unit = json.loads(upath.read_text())
        except (json.JSONDecodeError, OSError):
            counts["unparseable"] += 1
            print(f"[dvf-migrate] SKIP {upath.name} (unparseable) — retained", flush=True)
            continue
        fields = ("src_model", "src_cond", "tgt_model", "tgt_cond", "arm")
        if not isinstance(unit, dict) or not all(unit.get(f) for f in fields):
            counts["unattributable"] += 1
            print(
                f"[dvf-migrate] SKIP {upath.name} (no internal model record) — retained", flush=True
            )
            continue
        spec = ((unit["src_model"], unit["src_cond"]), (unit["tgt_model"], unit["tgt_cond"]))
        qk = unit_key(spec, unit["arm"])
        if upath.stem == qk:
            counts["already_qualified"] += 1
            continue
        new_upath = pairs_dir / f"{qk}.json"
        if new_upath.exists():
            counts["target_exists"] += 1
            print(
                f"[dvf-migrate] SKIP {upath.name} (target {new_upath.name} exists) — retained",
                flush=True,
            )
            continue
        old_stem = upath.stem
        unit["unit_key"] = qk
        unit["unit_key_migrated_from"] = old_stem
        _atomic_write_json(new_upath, unit)
        upath.unlink()  # rename: identical content retained under the qualified name
        old_bundle = bundles_dir / f"{old_stem}.npz"
        if old_bundle.exists():
            new_bundle = bundles_dir / f"{qk}.npz"
            if new_bundle.exists():
                print(
                    f"[dvf-migrate] SKIP bundle {old_bundle.name} (target exists) — retained",
                    flush=True,
                )
            else:
                os.replace(old_bundle, new_bundle)
        counts["renamed"] += 1
        print(f"[dvf-migrate] RENAMED {old_stem} -> {qk}", flush=True)
    print(
        f"[dvf-migrate] {out_root}: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        flush=True,
    )
    return counts


def cmd_migrate_keys(args) -> int:
    migrate_unqualified_keys(args.out_root)
    return 0


def verdict_class(r2_free: float, r2_ident: float, g1: float, g2: float) -> str:
    """FOUR-class disjoint+exhaustive verdict (plan v8 s3)."""
    if not np.isfinite(r2_free) or not np.isfinite(r2_ident):
        return "invalid"
    if r2_free < r2_ident:
        return "free_map_uninformative"
    if g1 >= 0:
        return "shared_readout_supported"
    if g2 >= 0:
        return "readout_changed"
    return "transfer_map_insufficient"


class _CellCache:
    """Tiny LRU over loaded (model, cond) cell bundles (~850 MB each fp64)."""

    def __init__(self, store_root: Path, layer: int, cap: int = 4):
        self.store_root, self.layer, self.cap = store_root, layer, cap
        self.d: dict = {}

    def get(self, model: str, cond: str) -> dict:
        key = (model, cond)
        if key in self.d:
            self.d[key] = self.d.pop(key)
            return self.d[key]
        v = fl._load_cell_layer(self.store_root, f"{model}/{cond}", self.layer)
        self.d[key] = v
        while len(self.d) > self.cap:
            self.d.pop(next(iter(self.d)))
        return v


def _pinv_apply(y_c, U, s, Vh, k: int):
    """rows @ pinv_k(W) with W = U diag(s) Vh (row convention y = x @ W)."""
    return ((y_c @ Vh[:k].T) * (1.0 / s[:k])) @ U[:, :k].T


def _pinv_matrix(U, s, Vh, k: int):
    return Vh[:k].T @ ((1.0 / s[:k])[:, None] * U[:, :k].T)


def run_unit(
    source: dict, target: dict, spec, arm: str, args, lams: np.ndarray
) -> tuple[dict, dict]:
    """Items 1-6 battery for one (ordered pair, arm). Returns (unit_json, bundle)."""
    import torch

    (sm, sc), (tm, tc) = spec
    t0 = time.perf_counter()
    common, s_idx, t_idx = fl.pair_rows_by_conv(source["conv_ids"], target["conv_ids"])
    if args.row_limit is not None:
        common, s_idx, t_idx = (a[: args.row_limit] for a in (common, s_idx, t_idx))
    n = len(common)
    if n < 3:
        return {"error": "insufficient shared conv_ids", "retryable": False, "n_common": int(n)}, {}
    dsl = slice(None) if args.dim_limit is None else slice(0, args.dim_limit)
    X_S = source[f"X_{arm}"][s_idx][:, dsl]
    Y_S = source["Y"][s_idx][:, dsl]
    X_T = target[f"X_{arm}"][t_idx][:, dsl]
    Y_T = target["Y"][t_idx][:, dsl]
    d = X_S.shape[1]

    folds = fl._conv_grouped_folds(common, n_folds=N_FOLDS, seed=args.seed)
    dev = torch.device(args.device)
    tX_S = torch.from_numpy(np.ascontiguousarray(X_S)).to(dev)
    tY_S = torch.from_numpy(np.ascontiguousarray(Y_S)).to(dev)
    tX_T = torch.from_numpy(np.ascontiguousarray(X_T)).to(dev)
    tY_T = torch.from_numpy(np.ascontiguousarray(Y_T)).to(dev)

    # --- Fit basis (plan v10 s4): ambient (parent, byte-identical) or a
    # per-(pair, arm, fold) shared train-fold-only PCA rank-k basis.
    reduced = fit_basis_of(args) == "reduced"
    k_unit = 0
    per_fold_n_train: dict[int, int] = {}
    basis_meta: dict[int, dict] = {}
    basis_spectra: dict[int, tuple] = {}
    if reduced:
        k_unit, per_fold_n_train = compute_k_unit(folds, n, d)
        if per_fold_n_train and k_unit < 1:
            return {
                "error": f"k_unit < 1 (min-fold n_train {min(per_fold_n_train.values())})",
                "retryable": False,
                "n_common": int(n),
            }, {}
    fdim = k_unit if reduced else d  # the dimension every fit/operator lives in

    model_labels = (
        ["b_free", "identity_bias"]
        + [f"b_derived_{lab}" for lab in RANK_LABELS]
        + [f"b_derived2_{lab}" for lab in RANK_LABELS]
    )
    pooled_pred: dict[str, list[np.ndarray]] = {m: [] for m in model_labels}
    pooled_true: list[np.ndarray] = []
    # Ambient-reconstruction companion pools (reduced mode only, plan s4 item 5).
    pooled_pred_amb: dict[str, list[np.ndarray]] = {m: [] for m in model_labels}
    pooled_true_amb: list[np.ndarray] = []
    per_fold_r2: dict[str, dict[int, float]] = {m: {} for m in model_labels}
    lambdas_chosen: dict[int, dict[str, float]] = {}
    eff_ranks: dict[int, float] = {}
    skipped_folds: list[int] = []
    canonical: dict = {}

    for k_fold in range(N_FOLDS):
        te_mask = folds == k_fold
        tr_mask = ~te_mask
        if tr_mask.sum() < 3 or te_mask.sum() < 1:
            skipped_folds.append(k_fold)
            continue
        tr = torch.from_numpy(np.where(tr_mask)[0]).to(dev)
        te = torch.from_numpy(np.where(te_mask)[0]).to(dev)
        conv_tr = common[np.where(tr_mask)[0]]
        if reduced:
            # Well-posedness invariant (plan s4 item 1): holds by construction
            # of k_unit; a violation is a recorded unit error [wellposed-assert].
            n_tr = int(tr.numel())
            assert n_tr >= 2 * k_unit, (
                f"well-posedness violated: fold {k_fold} n_train={n_tr} < 2*k_unit={2 * k_unit}"
            )
            # ONE basis per side, SHARED across source and target, train-fold
            # rows ONLY (test rows never touch the basis — plan s4 item 2).
            mu_x, Q_x, sx_k, fx_tr = _pca_basis(torch.cat([tX_S[tr], tX_T[tr]], dim=0), k_unit)
            mu_y, Q_y, sy_k, fy_tr = _pca_basis(torch.cat([tY_S[tr], tY_T[tr]], dim=0), k_unit)
            assert Q_x.shape == (d, k_unit) and Q_y.shape == (d, k_unit), (
                tuple(Q_x.shape),
                tuple(Q_y.shape),
            )
            xs_tr, xt_tr = (tX_S[tr] - mu_x) @ Q_x, (tX_T[tr] - mu_x) @ Q_x
            ys_tr, yt_tr = (tY_S[tr] - mu_y) @ Q_y, (tY_T[tr] - mu_y) @ Q_y
            ys_te, yt_te = (tY_S[te] - mu_y) @ Q_y, (tY_T[te] - mu_y) @ Q_y
            basis_spectra[k_fold] = (sx_k.cpu().numpy(), sy_k.cpu().numpy())
            basis_meta[k_fold] = {
                "n_train": n_tr,
                "captured_var_train_x": fx_tr,
                "captured_var_train_y": fy_tr,
                "captured_var_test_x": _heldout_captured_frac(
                    torch.cat([tX_S[te], tX_T[te]], dim=0), mu_x, Q_x
                ),
                "captured_var_test_y": _heldout_captured_frac(
                    torch.cat([tY_S[te], tY_T[te]], dim=0), mu_y, Q_y
                ),
            }
        else:
            xs_tr, xt_tr = tX_S[tr], tX_T[tr]
            ys_tr, yt_tr = tY_S[tr], tY_T[tr]
            ys_te, yt_te = tY_S[te], tY_T[te]
        # Train-fold-only fits — the battery's ONLY fitting convention (plan s4
        # item 2): W_S, W_T, M, a, b_S all exclude the fold's test rows.
        W_S, b_S, lam_ws = fl._fit_ridge_inner_group_cv_t(xs_tr, ys_tr, conv_tr, lams)
        W_T, b_T, lam_wt = fl._fit_ridge_inner_group_cv_t(xt_tr, yt_tr, conv_tr, lams)
        M, a_M, lam_m = fl._fit_ridge_inner_group_cv_t(xs_tr, xt_tr, conv_tr, lams)
        B_free, b_free, lam_bf = fl._fit_ridge_inner_group_cv_t(ys_tr, yt_tr, conv_tr, lams)
        lambdas_chosen[k_fold] = {"W_S": lam_ws, "W_T": lam_wt, "M": lam_m, "B_free": lam_bf}

        U, s, Vh = fl._svd_robust_t(W_S)
        s = s.clamp_min(1e-300)  # guard exact-zero singulars in 1/s (rank-deficient W_S)
        eff = float((s.sum() ** 2 / (s**2).sum()).item())
        eff_ranks[k_fold] = eff
        k_eff = int(max(1, min(round(eff), s.shape[0])))
        rank_map = {
            lab: min(int(r), int(s.shape[0])) for lab, r in zip(RANK_LABELS[:3], TRUNCATION_RANKS)
        }
        rank_map["effrank"] = k_eff

        Y_true_te = yt_te
        pooled_true.append(Y_true_te.cpu().numpy())
        if reduced:
            pooled_true_amb.append(tY_T[te].cpu().numpy())

        def _pool(label: str, pred) -> None:
            """Pool a fold prediction (+ its ambient reconstruction when reduced)."""
            pooled_pred[label].append(pred.cpu().numpy())
            per_fold_r2[label][k_fold] = fl._r2_t(Y_true_te, pred)
            if reduced:
                pooled_pred_amb[label].append((pred @ Q_y.T + mu_y).cpu().numpy())

        y_c = ys_te - b_S
        mw_s = M @ W_S
        mw_t = M @ W_T
        aw_s = a_M @ W_S + b_S
        aw_t = a_M @ W_T + b_T
        for lab in RANK_LABELS:
            kk = rank_map[lab]
            xhat = _pinv_apply(y_c, U, s, Vh, kk)
            _pool(f"b_derived_{lab}", xhat @ mw_s + aw_s)
            _pool(f"b_derived2_{lab}", xhat @ mw_t + aw_t)
        _pool("b_free", ys_te @ B_free + b_free)
        tr_np = np.where(tr_mask)[0]
        te_np = np.where(te_mask)[0]
        if reduced:
            ys_tr_np, yt_tr_np = ys_tr.cpu().numpy(), yt_tr.cpu().numpy()
            ys_te_np, yt_te_np = ys_te.cpu().numpy(), yt_te.cpu().numpy()
            pred_ident = identity_bias_predict(ys_tr_np, yt_tr_np, ys_te_np)
            pooled_pred["identity_bias"].append(pred_ident)
            per_fold_r2["identity_bias"][k_fold] = fl._r2(yt_te_np, pred_ident)
            q_y_np, mu_y_np = Q_y.cpu().numpy(), mu_y.cpu().numpy()
            pooled_pred_amb["identity_bias"].append(pred_ident @ q_y_np.T + mu_y_np)
        else:
            pred_ident = identity_bias_predict(Y_S[tr_np], Y_T[tr_np], Y_S[te_np])
            pooled_pred["identity_bias"].append(pred_ident)
            per_fold_r2["identity_bias"][k_fold] = fl._r2(Y_T[te_np], pred_ident)

        if not canonical:  # canonical fold = FIRST completed fold (fold 0 by construction)
            canonical = {
                "fold": k_fold,
                "U": U,
                "s": s,
                "Vh": Vh,
                "rank_map": dict(rank_map),
                "M": M,
                "W_S": W_S,
                "W_T": W_T,
                "B_free": B_free,
                "eff_rank": eff,
            }

    if not pooled_true:
        return {
            "error": "all folds degenerate",
            "retryable": False,
            "n_common": int(n),
            "skipped_folds": skipped_folds,
        }, {}

    true_arr = np.concatenate(pooled_true, axis=0)
    r2_pooled: dict[str, float] = {}
    for m in model_labels:
        pred_arr = np.concatenate(pooled_pred[m], axis=0)
        r2_pooled[m] = fl._r2(true_arr, pred_arr)

    # Ambient-reconstruction companion R2 (reduced only, plan s4 item 5):
    # Q_y @ y_hat_red + mu_y against the RAW ambient y_T, pooled across folds.
    r2_pooled_amb: dict[str, float] = {}
    if reduced:
        true_amb = np.concatenate(pooled_true_amb, axis=0)
        for m in model_labels:
            r2_pooled_amb[m] = fl._r2(true_amb, np.concatenate(pooled_pred_amb[m], axis=0))

    def _max_read(prefix: str) -> tuple[float, str]:
        vals = {lab: r2_pooled[f"{prefix}_{lab}"] for lab in RANK_LABELS}
        best = max(vals, key=lambda lab: np.nan_to_num(vals[lab], nan=-np.inf))
        return vals[best], best

    r2_free = r2_pooled["b_free"]
    r2_ident = r2_pooled["identity_bias"]
    r2_d_max, d_argmax = _max_read("b_derived")
    r2_d2_max, d2_argmax = _max_read("b_derived2")
    thr = RUNG_REACHED_THRESHOLD
    g1 = r2_d_max - thr * r2_free
    g2 = r2_d2_max - thr * r2_free
    g1_eff = r2_pooled["b_derived_effrank"] - thr * r2_free
    g2_eff = r2_pooled["b_derived2_effrank"] - thr * r2_free

    knn_models = {
        "b_free": "b_free",
        "identity_bias": "identity_bias",
        f"b_derived_{d_argmax}": "b_derived_argmax",
        "b_derived_effrank": "b_derived_effrank",
        f"b_derived2_{d2_argmax}": "b_derived2_argmax",
        "b_derived2_effrank": "b_derived2_effrank",
    }
    knn_out: dict[str, dict] = {}
    for src_label, out_label in knn_models.items():
        pred_arr = np.concatenate(pooled_pred[src_label], axis=0)
        knn_out[out_label] = {
            metric: knn_retrieval(pred_arr, true_arr, ks=(1, 5, 10), metric=metric)
            for metric in ("euclidean", "cosine")
        }

    # Operator-level read (item 5) on the canonical fold's train-fit operators.
    import torch as _torch  # local alias for clarity

    U, s, Vh = canonical["U"], canonical["s"], canonical["Vh"]
    rank_map = canonical["rank_map"]
    M, W_S, W_T, B_free = canonical["M"], canonical["W_S"], canonical["W_T"], canonical["B_free"]
    op_variants = {
        "derived_effrank": ("d", rank_map["effrank"]),
        f"derived_{d_argmax}": ("d", rank_map[d_argmax]),
        "derived2_effrank": ("d2", rank_map["effrank"]),
        f"derived2_{d2_argmax}": ("d2", rank_map[d2_argmax]),
    }
    raw_cos: dict[str, float] = {}
    svecs: dict[str, np.ndarray] = {}
    _, s_free, _ = fl._svd_robust_t(B_free)
    svecs["free"] = s_free.cpu().numpy()
    seen: dict[tuple, str] = {}
    for name, (kind, kk) in op_variants.items():
        if (kind, kk) in seen:  # argmax == effrank: reuse
            raw_cos[name] = raw_cos[seen[(kind, kk)]]
            svecs[name] = svecs[seen[(kind, kk)]]
            continue
        pinv_k = _pinv_matrix(U, s, Vh, kk)
        B_op = pinv_k @ (M @ (W_S if kind == "d" else W_T))
        raw_cos[name] = _cos_flat(B_op, B_free)
        _, s_op, _ = fl._svd_robust_t(B_op)
        svecs[name] = s_op.cpu().numpy()
        seen[(kind, kk)] = name
        del B_op

    # Compact bundle (plan s10): M-I top-256 factors fp16, spectra, per-fold R2.
    # In reduced mode every operator lives in the k_unit-dim basis (fdim).
    Mm = M - _torch.eye(fdim, dtype=_torch.float64, device=M.device)
    Um, sm_v, Vhm = fl._svd_robust_t(Mm)
    n_keep = min(256, fdim)
    rank_grid_r2 = np.array(
        [
            [per_fold_r2[f"b_derived_{lab}"].get(f, np.nan) for f in range(N_FOLDS)]
            for lab in RANK_LABELS
        ]
    )
    bundle = {
        "m_minus_i_u256_fp16": Um[:, :n_keep].cpu().numpy().astype(np.float16),
        "m_minus_i_vh256_fp16": Vhm[:n_keep].cpu().numpy().astype(np.float16),
        "m_minus_i_svals": sm_v.cpu().numpy(),
        "w_s_svals": s.cpu().numpy(),
        "svec_free": svecs["free"],
        "per_fold_rank_r2_derived": rank_grid_r2,
        "canonical_fold": np.int64(canonical["fold"]),
    }
    for name in op_variants:
        bundle[f"svec_{name}"] = svecs[name]
    if reduced:
        # Per-fold Q-basis spectra (plan s10 bundle contents).
        for f_id, (sx_np, sy_np) in basis_spectra.items():
            bundle[f"q_x_svals_f{f_id}"] = sx_np
            bundle[f"q_y_svals_f{f_id}"] = sy_np

    unit = {
        "meta": regime_meta(args),
        "src_model": sm,
        "src_cond": sc,
        "tgt_model": tm,
        "tgt_cond": tc,
        "arm": arm,
        "pair_key": fl.pair_spec_key(spec),
        "unit_key": unit_key(spec, arm),
        "cross_model": sm != tm,
        "n_common": int(n),
        "d": int(d),
        "flag_low_common": bool(n < LOW_COMMON_FLAG),
        "skipped_folds": skipped_folds,
        "n_rows_pooled": int(true_arr.shape[0]),
        "lambdas_chosen": lambdas_chosen,
        "eff_rank_w_s_per_fold": eff_ranks,
        "rank_map_canonical": {k: int(v) for k, v in rank_map.items()},
        "r2_pooled": r2_pooled,
        "per_fold_r2": {m: {int(k): v for k, v in d_.items()} for m, d_ in per_fold_r2.items()},
        "r2_b_free": r2_free,
        "r2_identity_bias": r2_ident,
        "r2_b_derived_max": r2_d_max,
        "b_derived_argmax_rank": d_argmax,
        "r2_b_derived2_max": r2_d2_max,
        "b_derived2_argmax_rank": d2_argmax,
        "g1": g1,
        "g2": g2,
        "g1_fixed_effrank": g1_eff,
        "g2_fixed_effrank": g2_eff,
        "verdict": verdict_class(r2_free, r2_ident, g1, g2),
        "verdict_fixed_effrank": verdict_class(r2_free, r2_ident, g1_eff, g2_eff),
        "knn": knn_out,
        "operator_read": {
            "canonical_fold": canonical["fold"],
            "raw_cosine": raw_cos,
            "rotation_null": None,  # patched by --phase nulls
        },
        "wall_s": round(time.perf_counter() - t0, 2),
        "metadata": _metadata(),
    }
    if reduced:
        # Reduced-only fields (plan s4 item 5 + s6 estimator-validity line).
        # Keyed ADDITIVELY so the ambient unit JSON stays byte-identical.
        unit["fit_basis"] = "reduced"
        unit["k_unit"] = int(k_unit)
        unit["fit_dim"] = int(fdim)
        unit["k_floor_limited"] = k_floor_limited(k_unit)
        unit["per_fold_n_train"] = {int(f): int(v) for f, v in per_fold_n_train.items()}
        unit["pca_basis_per_fold"] = {int(f): v for f, v in basis_meta.items()}
        unit["r2_pooled_ambient_recon"] = r2_pooled_amb
        unit["r2_b_free_ambient_recon"] = r2_pooled_amb.get("b_free")
    return unit, bundle


# ---------------------------------------------------------------------------
# Pair-spec enumeration
# ---------------------------------------------------------------------------
def build_pair_specs(args) -> list:
    """Resolve the ordered pair-spec list from --pairs-file / --pair-set."""
    if args.pairs_file is not None:
        loaded = json.loads(Path(args.pairs_file).read_text())
        return fl.parse_pair_specs(loaded, default_model=args.default_model)
    if args.pair_set == "within-model":
        models = [m for m in args.models.split(",") if m]
        return [((m, s), (m, t)) for m in models for (s, t) in enumerate_pair_set()]
    if args.pair_set == "cross-model":
        return fl.crossmodel_pair_specs(MODEL_SLUGS[0], MODEL_SLUGS[1])
    raise ValueError(f"unknown --pair-set {args.pair_set!r}")


def _units(specs: list) -> list:
    return [(spec, arm) for spec in specs for arm in ("prefix", "context")]


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def cmd_pairs(args) -> int:
    lams = (
        fl.LAMBDAS if args.lambda_grid == "ladder13" else fl.resolve_lambda_grid(args.lambda_grid)
    )
    specs = build_pair_specs(args)
    units = _units(specs)
    shard_units = units[args.shard_index :: args.num_shards] if args.num_shards > 1 else units
    pairs_dir = args.out_root / "pairs"
    bundles_dir = args.out_root / "bundles"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    cache = _CellCache(args.store_root, args.layer)
    want = regime_meta(args)
    n_shard = len(shard_units)
    n_fail = 0
    for i, (spec, arm) in enumerate(shard_units):
        uk = unit_key(spec, arm)
        upath = pairs_dir / f"{uk}.json"
        if upath.exists():
            try:
                prior = json.loads(upath.read_text())
            except (json.JSONDecodeError, OSError):
                prior = None
            if prior is not None and prior.get("meta") == want and not prior.get("retryable"):
                print(f"[dvf] unit {i + 1}/{n_shard} {uk} RESUME (checkpoint)", flush=True)
                continue
        print(f"[dvf] unit {i + 1}/{n_shard} {uk}", flush=True)
        t0 = time.perf_counter()
        (sm, sc), (tm, tc) = spec
        try:
            source = cache.get(sm, sc)
            target = cache.get(tm, tc)
            unit, bundle = run_unit(source, target, spec, arm, args, lams)
        except Exception as exc:  # recorded hole, battery continues (plan s4)
            import traceback

            traceback.print_exc()
            unit, bundle = {"error": f"{type(exc).__name__}: {exc}", "retryable": True}, {}
        if "error" in unit:
            n_fail += 1
            unit.setdefault("unit_key", uk)
            unit["meta"] = want
            _atomic_write_json(upath, unit)
            print(f"[dvf] unit {i + 1}/{n_shard} {uk} FAILED: {unit['error']}", flush=True)
            continue
        if bundle:
            _atomic_savez(bundles_dir / f"{uk}.npz", **bundle)
        _atomic_write_json(upath, unit)
        print(
            f"[dvf] unit {i + 1}/{n_shard} {uk} done verdict={unit['verdict']} "
            f"g1={unit['g1']:.4f} elapsed={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    print(f"[dvf] pairs phase done: {n_shard} units, {n_fail} failures (recorded)", flush=True)
    return 0


def cmd_nulls(args) -> int:
    """Item-5 two-sided rotation null — SHARED Haar draws over every unit.

    Per draw k (seed*1000003+k, the parent battery convention): E = P * R^T
    with P, R Haar(d); the null cosine for (A, B) is s_A^T E s_B /
    (|A|_F |B|_F) — per-draw EXACTLY the verbatim two-QR formula (von Neumann
    identity; distribution-identical by Haar invariance).
    """
    import torch

    pairs_dir = args.out_root / "pairs"
    bundles_dir = args.out_root / "bundles"
    todo: list[tuple[Path, Path, dict]] = []
    for upath in sorted(pairs_dir.glob("*.json")):
        unit = json.loads(upath.read_text())
        if "error" in unit or "operator_read" not in unit:
            continue
        nul = unit["operator_read"].get("rotation_null")
        if nul is not None and int(nul.get("n_draws", 0)) >= args.rotation_draws:
            continue
        bpath = bundles_dir / f"{unit['unit_key']}.npz"
        if not bpath.exists():
            print(f"[dvf-nulls] MISSING bundle for {unit['unit_key']} — skipped", flush=True)
            continue
        todo.append((upath, bpath, unit))
    if not todo:
        print("[dvf-nulls] nothing to do", flush=True)
        return 0
    # Group by the FIT dimension: k_unit under --fit-basis reduced (rotation
    # nulls draw in dimension k_unit, plan v10 s4 item 4), else ambient d
    # (dim-limit smokes may differ from production units). Ambient units carry
    # no fit_dim field -> fallback d keeps parent outputs consumable verbatim.
    by_d: dict[int, list[int]] = {}
    loaded = []
    for idx, (upath, bpath, unit) in enumerate(todo):
        with np.load(bpath) as z:
            svec_map = {k: z[k] for k in z.files if k.startswith("svec_")}
        loaded.append((upath, bpath, unit, svec_map))
        by_d.setdefault(int(unit.get("fit_dim") or unit["d"]), []).append(idx)
    dev = torch.device(args.device)
    for d, idxs in sorted(by_d.items()):
        cmp_rows_a, cmp_rows_b, cmp_keys = [], [], []
        for idx in idxs:
            _, _, unit, svec_map = loaded[idx]
            s_free = svec_map["svec_free"]
            for name, sv in svec_map.items():
                if name == "svec_free":
                    continue
                cmp_keys.append((idx, name.removeprefix("svec_")))
                cmp_rows_a.append(sv)
                cmp_rows_b.append(s_free)
        if not cmp_keys:
            continue
        S_a = torch.from_numpy(np.stack(cmp_rows_a)).to(dev)
        S_b = torch.from_numpy(np.stack(cmp_rows_b)).to(dev)
        denom = S_a.norm(dim=1) * S_b.norm(dim=1) + 1e-12
        draws = np.zeros((args.rotation_draws, len(cmp_keys)), dtype=np.float64)
        for k in range(args.rotation_draws):
            t0 = time.perf_counter()
            gen = torch.Generator().manual_seed(args.seed * 1_000_003 + k)
            p = _haar(d, gen).to(dev)
            r = _haar(d, gen).to(dev)
            e = p * r.T
            vals = ((S_a @ e) * S_b).sum(dim=1) / denom
            draws[k] = vals.cpu().numpy()
            print(
                f"[dvf-nulls] d={d} draw {k + 1}/{args.rotation_draws} "
                f"elapsed={time.perf_counter() - t0:.2f}s",
                flush=True,
            )
        for col, (idx, cmp_name) in enumerate(cmp_keys):
            upath, bpath, unit, _ = loaded[idx]
            arr = draws[:, col]
            nul_block = unit["operator_read"].setdefault("rotation_null", {}) or {}
            # Direct assignment, NOT setdefault: on a draw-count-upgrade rerun
            # (stored n_draws < --rotation-draws) the whole draw battery is
            # recomputed above, so the persisted regime metadata must reflect
            # the ACTUAL draw count/seed used — a stale n_draws would leave the
            # skip predicate + downstream consumers reading the old regime.
            nul_block["n_draws"] = int(args.rotation_draws)
            nul_block["seed"] = int(args.seed)
            nul_block["convention"] = "parent-9a-ter svec reduction (E = P*R^T)"
            nul_block[cmp_name] = {
                "null_mean": float(arr.mean()),
                "null_std": float(arr.std()),
                "null_p025": float(np.quantile(arr, 0.025)),
                "null_p975": float(np.quantile(arr, 0.975)),
                "observed": unit["operator_read"]["raw_cosine"].get(cmp_name),
            }
            unit["operator_read"]["rotation_null"] = nul_block
        # Persist per-draw matrices into the bundles (plan s6 persistence duty).
        for idx in idxs:
            upath, bpath, unit, _ = loaded[idx]
            cols = [c for c, (i2, _n) in enumerate(cmp_keys) if i2 == idx]
            names = [n for (i2, n) in cmp_keys if i2 == idx]
            with np.load(bpath) as z:
                arrays = {k: z[k] for k in z.files}
            for c, nm in zip(cols, names):
                arrays[f"rotation_draws_{nm}"] = draws[:, c]
            _atomic_savez(bpath, **arrays)
            _atomic_write_json(upath, unit)
    print(f"[dvf-nulls] patched {len(todo)} units at {args.rotation_draws} draws", flush=True)
    return 0


def _load_parent_rungs(args) -> dict:
    """parent rung_reached index: {(model, 'src__tgt', arm): rung} from ladder JSONs."""
    out: dict = {}
    for model in MODEL_SLUGS:
        p = args.parent_ladder_dir / f"ladder_{model}_L19.json"
        if not p.exists():
            print(f"[dvf-merge] WARN no parent ladder JSON at {p}", flush=True)
            continue
        ladder = json.loads(p.read_text())
        for pair_key, arms in ladder.get("pairs", {}).items():
            for arm, res in arms.items():
                if isinstance(res, dict) and "rung_reached_point" in res:
                    out[(model, pair_key, arm)] = int(res["rung_reached_point"])
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def assert_distinct_unit_keys(units: list, expect_units: int | None) -> list[str]:
    """Fail-loud key-accounting guard (fix round 3, #1689).

    The pre-fix merge enumerated per (spec, arm) but resolved UNQUALIFIED
    filenames, counting each surviving file for BOTH models (n_complete
    '504' = 252x2). Model-qualified keys make that structurally impossible;
    this guard asserts it stays so: every enumerated unit key is distinct,
    and (when --expect-units is given) exactly N distinct qualified units
    are enumerated.
    """
    from collections import Counter

    uks = [unit_key(spec, arm) for spec, arm in units]
    dupes = sorted(k for k, c in Counter(uks).items() if c > 1)
    if dupes:
        raise RuntimeError(
            f"residual double-count: {len(dupes)} duplicate unit keys in enumeration "
            f"(first: {dupes[:5]})"
        )
    if expect_units is not None and len(uks) != expect_units:
        raise RuntimeError(
            f"expected {expect_units} distinct qualified units, enumerated {len(uks)}"
        )
    return uks


def unit_identity_mismatch(unit: dict, spec, arm: str) -> bool:
    """True when a loaded unit file's internal identity contradicts its key
    (a stale/mis-keyed checkpoint must never be counted for another model)."""
    (sm, sc), (tm, tc) = spec
    recorded = (
        unit.get("src_model"),
        unit.get("src_cond"),
        unit.get("tgt_model"),
        unit.get("tgt_cond"),
        unit.get("arm"),
    )
    return recorded != (sm, sc, tm, tc, arm)


def cmd_merge(args) -> int:
    specs = build_pair_specs(args)
    units = _units(specs)
    assert_distinct_unit_keys(units, getattr(args, "expect_units", None))
    pairs_dir = args.out_root / "pairs"
    rungs = _load_parent_rungs(args)
    rows, failures, missing, key_mismatch = [], [], [], []
    for spec, arm in units:
        uk = unit_key(spec, arm)
        upath = pairs_dir / f"{uk}.json"
        if not upath.exists():
            missing.append(uk)
            continue
        unit = json.loads(upath.read_text())
        if "error" in unit:
            failures.append({"unit_key": uk, "error": unit["error"]})
            continue
        if unit_identity_mismatch(unit, spec, arm):
            key_mismatch.append({"unit_key": uk, "recorded_src_model": unit.get("src_model")})
            continue
        rows.append(unit)
    verdict_counts: dict[str, dict[str, int]] = {}
    class0_counts: dict[str, int] = {}
    verdict_counts_fixed: dict[str, dict[str, int]] = {}
    conc_pool: dict[str, dict[str, list]] = {}
    for unit in rows:
        model_key = (
            unit["src_model"]
            if not unit["cross_model"]
            else f"{unit['src_model']}->{unit['tgt_model']}"
        )
        gk = f"{model_key}|{unit['arm']}"
        v, vf = unit["verdict"], unit["verdict_fixed_effrank"]
        if v == "free_map_uninformative":
            class0_counts[gk] = class0_counts.get(gk, 0) + 1
        else:
            verdict_counts.setdefault(gk, {}).setdefault(v, 0)
            verdict_counts[gk][v] += 1
        verdict_counts_fixed.setdefault(gk, {}).setdefault(vf, 0)
        verdict_counts_fixed[gk][vf] += 1
        if not unit["cross_model"]:
            rung = rungs.get((unit["src_model"], unit["pair_key"], unit["arm"]))
            if rung is not None and v != "free_map_uninformative" and np.isfinite(unit["g1"]):
                cp = conc_pool.setdefault(unit["arm"], {"rung": [], "g1": [], "g2": []})
                cp["rung"].append(rung)
                cp["g1"].append(unit["g1"])
                cp["g2"].append(unit["g2"])
    concordance = {}
    for arm, cp in conc_pool.items():
        concordance[arm] = {
            "n": len(cp["rung"]),
            "spearman_rung_g1": _spearman(np.array(cp["rung"]), np.array(cp["g1"])),
            "spearman_rung_g2": _spearman(np.array(cp["rung"]), np.array(cp["g2"])),
        }
    summary = {
        "meta": regime_meta(args),
        "n_expected_units": len(units),
        "n_complete": len(rows),
        "n_failed": len(failures),
        "n_missing": len(missing),
        "n_key_mismatch": len(key_mismatch),
        "failures": failures,
        "missing_units": missing[:50],
        "key_mismatch_units": key_mismatch[:50],
        "verdict_counts": verdict_counts,
        "class0_free_map_uninformative_counts": class0_counts,
        "verdict_counts_fixed_effrank": verdict_counts_fixed,
        # Concordance rho (plan s3): informative within-model units per arm;
        # NaN below the n>=2 floor (a 1-unit smoke is a designed NaN).
        "concordance": concordance,
        "metadata": _metadata(),
    }
    _atomic_write_json(args.out_root / "summary.json", summary)
    print(
        f"[dvf-merge] wrote summary: {len(rows)} complete / {len(failures)} failed / "
        f"{len(missing)} missing / {len(key_mismatch)} key-mismatched of {len(units)} units",
        flush=True,
    )
    if missing or key_mismatch:
        print(
            f"[dvf-merge] FAIL-LOUD: {len(missing)} units never attempted, "
            f"{len(key_mismatch)} mis-keyed",
            flush=True,
        )
        return 3
    return 0


GATE1_NOOP_ATOL = 1e-6  # plan v10 s7: pure-refactor guard (fallback GATE1_ATOL)


def _gate1_battery_parity(args, lams, report: dict) -> bool:
    """Gate 1a (plan v10 s7): ambient no-op battery parity vs the PUBLISHED
    parent per-unit JSON (`derived_vs_free_B/pairs/<parity unit>__context.json`).

    Runs the EXTENDED run_unit with fit_basis FORCED to ambient at full shape
    and diffs the battery scalars: strict atol 1e-6 (pure refactor), fallback
    GATE1_ATOL=1e-3 recorded as ``fallback_used`` (GPU-class numerics).
    Under slice knobs (--row-limit/--dim-limit) the published full-shape
    scalars are unreachable BY CONSTRUCTION, so the check auto-demotes to an
    informational record (smoke/production gate-calibration parity, #1345).
    """
    spec = GATE1_PAIR
    (sm, sc), (tm, tc) = spec
    target = getattr(args, "gate1_battery_target", None) or (
        REPO_ROOT
        / "eval_results/issue_1689/derived_vs_free_B/pairs"
        / f"{unit_key(spec, 'context')}.json"
    )
    published = json.loads(Path(target).read_text())
    amb_args = argparse.Namespace(**vars(args))
    amb_args.fit_basis = "ambient"
    cache = _CellCache(args.store_root, args.layer)
    t0 = time.perf_counter()
    unit, _bundle = run_unit(cache.get(sm, sc), cache.get(tm, tc), spec, "context", amb_args, lams)
    wall = time.perf_counter() - t0
    if "error" in unit:
        report["battery_parity"] = {"ok": False, "unit_error": unit["error"]}
        return False
    diffs = {
        k: abs(float(published["r2_pooled"][k]) - float(unit["r2_pooled"][k]))
        for k in published["r2_pooled"]
    }
    for scalar in ("g1", "g2", "g1_fixed_effrank", "g2_fixed_effrank"):
        diffs[scalar] = abs(float(published[scalar]) - float(unit[scalar]))
    max_diff = max(diffs.values())
    informational = args.row_limit is not None or args.dim_limit is not None
    strict_ok = max_diff <= GATE1_NOOP_ATOL
    fallback_ok = max_diff <= GATE1_ATOL
    ok = informational or strict_ok or fallback_ok
    report["battery_parity"] = {
        "pair": fl.pair_spec_key(spec),
        "arm": "context",
        "target_json": str(target),
        "max_abs_scalar_diff": max_diff,
        "per_scalar_abs_diff": diffs,
        "n_common_match": int(published["n_common"]) == int(unit["n_common"]),
        "verdict_match": published["verdict"] == unit["verdict"],
        "atol_strict": GATE1_NOOP_ATOL,
        "atol_fallback": GATE1_ATOL,
        "fallback_used": bool(not strict_ok and fallback_ok),
        "informational": informational,
        "battery_unit_wall_s": round(wall, 1),
        "ok": ok,
    }
    if not informational:
        report["battery_parity"]["ok"] = ok = ok and (
            report["battery_parity"]["n_common_match"] and report["battery_parity"]["verdict_match"]
        )
    tag = "INFO (sliced shape — published target is full-shape)" if informational else ""
    print(f"[dvf-gate1] battery no-op parity max|diff|={max_diff:.3e} ok={ok} {tag}", flush=True)
    return bool(ok)


def cmd_gate1(args) -> int:
    """Gate 1 (plan s7): parity vs the published parent per-pair JSON + timing pilot.

    ``--gate1-checks``: ladder (parent behavior, default) | battery (the
    wellposed round's ambient no-op parity, plan v10 s7 Gate 1a) | both.
    """
    lams = (
        fl.LAMBDAS if args.lambda_grid == "ladder13" else fl.resolve_lambda_grid(args.lambda_grid)
    )
    report: dict = {"gate": "gate1", "atol": GATE1_ATOL, "metadata": _metadata()}
    spec = GATE1_PAIR
    (sm, sc), (tm, tc) = spec
    parity_ok = True
    ladder_wall = 0.0
    max_diff = float("nan")
    gate1_checks = getattr(args, "gate1_checks", "ladder") or "ladder"
    if gate1_checks in ("ladder", "both"):
        target_path = args.gate1_target or (
            REPO_ROOT / "eval_results/issue_1689/ladder" / f"pairs_{sm}_L19" / f"{sc}__{tc}.json"
        )
        published = json.loads(Path(target_path).read_text())["arms"]["context"]
        t0 = time.perf_counter()
        res = fl.run_pairs_generalized(
            args.store_root,
            [spec],
            layer=args.layer,
            n_bootstrap_draws=0,
            n_null_draws=args.gate1_null_draws,
            engine="torch",
            device=args.device,
            checkpoint_dir=None,
            lambda_grid=args.lambda_grid,
        )
        ladder_wall = time.perf_counter() - t0
        new = res["pairs"][fl.pair_spec_key(spec)]["context"]
        diffs = {
            k: abs(published["rung_r2s_point"][k] - new["rung_r2s_point"][k])
            for k in published["rung_r2s_point"]
        }
        max_diff = max(diffs.values())
        rung_match = int(published["rung_reached_point"]) == int(new["rung_reached_point"])
        n_match = int(published["n_common"]) == int(new["n_common"])
        parity_ok = max_diff <= GATE1_ATOL and rung_match and n_match
        report["parity"] = {
            "pair": fl.pair_spec_key(spec),
            "arm": "context",
            "target_json": str(target_path),
            "max_abs_rung_r2_diff": max_diff,
            "per_rung_abs_diff": diffs,
            "rung_reached_match": rung_match,
            "n_common_match": n_match,
            "n_common": int(new["n_common"]),
            "ladder_unit_wall_s": round(ladder_wall, 1),
            "ok": parity_ok,
        }
    battery_ok = True
    if gate1_checks in ("battery", "both"):
        battery_ok = _gate1_battery_parity(args, lams, report)
    if args.gate1_timing:
        cache = _CellCache(args.store_root, args.layer)
        source = cache.get(sm, sc)
        target = cache.get(tm, tc)
        t0 = time.perf_counter()
        unit, _bundle = run_unit(source, target, spec, "context", args, lams)
        report["timing"] = {
            "battery_unit_wall_s": round(time.perf_counter() - t0, 1),
            "row_limit": args.row_limit,
            "dim_limit": args.dim_limit,
            "fit_basis": fit_basis_of(args),
            "k_unit": unit.get("k_unit"),
            "unit_error": unit.get("error"),
        }
    _atomic_write_json(args.out_root / "gate1_report.json", report)
    if not parity_ok or not battery_ok:
        print(
            f"[dvf-gate1] PARITY FAIL: max|diff|={max_diff:.3e} battery_ok={battery_ok}",
            flush=True,
        )
        return 7  # distinct rc: designed gate refusal, not an anonymous crash (#1415)
    print(f"[dvf-gate1] PARITY PASS: max|diff|={max_diff:.3e} wall={ladder_wall:.1f}s", flush=True)
    return 0


def cmd_stage(args) -> int:
    """Stage the 42 pinned L19 stores to <store-root>/<model>/<cond>/L<layer>.pt.

    Per-file targets via hub.stage_hub_file (exact-dest; no mirror-root
    arithmetic — the #1774 trap applies to stage_hub_prefix, not here). Only
    L<layer>.pt files are staged (the prefix also holds other layers).
    """
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = hub.list_hf_files_under_path(
        api,
        DATA_REPO,
        STORE_HF_PREFIX,
        repo_type="dataset",
        revision=PINNED_STORE_REVISION,
    )
    wanted = [f for f in files if f.endswith(f"/L{args.layer}.pt")]
    if len(wanted) < 42:
        raise RuntimeError(f"expected 42 L{args.layer}.pt files at the pin, found {len(wanted)}")
    if args.stage_cells:
        # Sliced-INPUT smoke knob (same phase code path, fewer files): keep
        # only the named <model_slug>/<condition> cells. Production passes
        # nothing and stages all 42.
        keep = {c.strip() for c in args.stage_cells.split(",") if c.strip()}
        wanted = [
            f for f in wanted if f.removeprefix(STORE_HF_PREFIX + "/").rsplit("/", 1)[0] in keep
        ]
        if len(wanted) != len(keep):
            raise RuntimeError(
                f"--stage-cells: expected {len(keep)} cells, matched {len(wanted)} at the pin"
            )
    todo = []
    for repo_path in wanted:
        rel = repo_path.removeprefix(STORE_HF_PREFIX + "/")
        target = args.store_root / rel
        if not target.exists():
            todo.append((repo_path, target))
    if not todo:
        print(
            f"[dvf-stage] all {len(wanted)} stores already present under {args.store_root}",
            flush=True,
        )
        return 0
    # Fresh lane clones have no data/ tree at all (#1689 fellows crash: statvfs
    # raised FileNotFoundError before staging could create anything) — create
    # the destination first, then probe the filesystem it actually lands on.
    args.store_root.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(args.store_root)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    need_gb = args.stage_headroom_gb
    if free_gb < need_gb:
        raise RuntimeError(
            f"staging headroom {free_gb:.1f} GB < required {need_gb} GB on {args.store_root}"
        )

    def _one(item):
        repo_path, target = item
        hub.stage_hub_file(
            DATA_REPO,
            repo_path,
            target,
            repo_type="dataset",
            revision=PINNED_STORE_REVISION,
        )
        print(f"[dvf-stage] staged {target}", flush=True)

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(_one, todo))
    print(
        f"[dvf-stage] staged {len(todo)} files (of {len(wanted)}) at pin {PINNED_STORE_REVISION[:12]}",
        flush=True,
    )
    return 0


def _parent_input_rel_paths_local(root: Path) -> list[str]:
    """Enumerate the parent-input set from a git checkout; fail-loud on gaps."""
    missing = [s for s in PARENT_INPUT_SINGLES if not (root / s).is_file()]
    if missing:
        raise RuntimeError(f"parent-input singles missing under {root}: {missing}")
    rels: list[str] = list(PARENT_INPUT_SINGLES)
    for tree in PARENT_INPUT_TREES:
        found = sorted((root / tree).glob("*.json"))
        if not found:
            raise RuntimeError(f"parent-input tree empty under {root}: {tree}")
        rels.extend(str(p.relative_to(root)) for p in found)
    return rels


def cmd_upload_parent_inputs(args) -> int:
    """Mirror the committed parent eval_results inputs to the data repo
    (upload-first, #734): copy the exact enumerated set into a temp staging
    dir (no eligibility filter — the whole enumerated set uploads), ONE
    upload_folder commit, exact-set verify. Run VM-side from a git checkout;
    idempotent (identical bytes re-upload as a Hub no-op)."""
    import shutil
    import tempfile

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    root = args.parent_inputs_root
    rels = _parent_input_rel_paths_local(root)
    with tempfile.TemporaryDirectory(prefix="i1689-parent-inputs-") as td:
        stage = Path(td)
        for rel in rels:
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, dst)
        url = hub._upload(
            stage,
            DATA_REPO,
            "dataset",
            PARENT_INPUTS_HF_PREFIX,
            raise_on_error=True,
        )
    expected = [f"{PARENT_INPUTS_HF_PREFIX}/{r}" for r in rels]
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    missing = hub.verify_repo_paths_uploaded(
        api, DATA_REPO, expected, path_in_repo=PARENT_INPUTS_HF_PREFIX, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(
            f"parent-inputs upload verify FAILED: {len(missing)} missing (first {missing[:3]})"
        )
    print(
        f"[wp-parent-inputs] {len(expected)} files verified at {url or PARENT_INPUTS_HF_PREFIX}",
        flush=True,
    )
    return 0


def cmd_stage_parent_inputs(args) -> int:
    """Stage the parent-round committed inputs from the HF mirror (#734).

    Runs at wellposed-leg entry BEFORE the fence phase. Idempotent: a git
    checkout (VM smoke / pod) has every file and skips with NO Hub call.
    Per-file exact-dest staging via hub.stage_hub_file (the #1774
    mirror-root trap applies to stage_hub_prefix, not here); one revision
    resolved for the whole set. Fail-loud on an incomplete mirror — the
    downstream consumers' exists/WARN guards would otherwise SILENTLY skip
    the ladder rung conditioning instead of crashing.
    """
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    root = args.parent_inputs_root
    have_singles = [s for s in PARENT_INPUT_SINGLES if (root / s).is_file()]
    tree_counts = {t: len(list((root / t).glob("*.json"))) for t in PARENT_INPUT_TREES}
    if len(have_singles) == len(PARENT_INPUT_SINGLES) and all(tree_counts.values()):
        print(
            f"[wp-stage-parent-inputs] all parent inputs present under {root} "
            f"({len(have_singles)} singles + {sum(tree_counts.values())} tree files) — skip",
            flush=True,
        )
        return 0
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    revision = hub.retry_transient(
        lambda: api.repo_info(DATA_REPO, repo_type="dataset").sha,
        what=f"repo_info({DATA_REPO})",
    )
    listed = hub.list_hf_files_under_path(
        api, DATA_REPO, PARENT_INPUTS_HF_PREFIX, repo_type="dataset", revision=revision
    )
    rels = sorted(f.removeprefix(PARENT_INPUTS_HF_PREFIX + "/") for f in listed)
    relset = set(rels)
    absent = [s for s in PARENT_INPUT_SINGLES if s not in relset]
    bare = [t for t in PARENT_INPUT_TREES if not any(r.startswith(t + "/") for r in rels)]
    if absent or bare:
        raise RuntimeError(
            f"parent-inputs HF mirror incomplete at {PARENT_INPUTS_HF_PREFIX}: "
            f"missing singles {absent} / empty trees {bare} — run "
            "--phase upload-parent-inputs from a git checkout first"
        )
    todo = [(f"{PARENT_INPUTS_HF_PREFIX}/{r}", root / r) for r in rels if not (root / r).is_file()]

    def _one(item):
        repo_path, target = item
        hub.stage_hub_file(DATA_REPO, repo_path, target, repo_type="dataset", revision=revision)
        print(f"[wp-stage-parent-inputs] staged {target}", flush=True)

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(_one, todo))
    still = [s for s in PARENT_INPUT_SINGLES if not (root / s).is_file()]
    if still:
        raise RuntimeError(f"parent-input staging incomplete after fetch: {still}")
    print(
        f"[wp-stage-parent-inputs] fetched {len(todo)} files (of {len(rels)} mirrored, "
        f"rev {revision[:12]}) into {root}",
        flush=True,
    )
    return 0


def cmd_upload(args) -> int:
    """One upload_folder commit per out-root bundles dir + exact-set verify."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    bundles_dir = args.out_root / "bundles"
    if not bundles_dir.exists():
        print(f"[dvf-upload] no bundles dir at {bundles_dir} — nothing to upload", flush=True)
        return 0
    prefix = f"{BUNDLE_HF_PREFIX}/{args.out_root.name}"
    url = hub._upload(
        bundles_dir,
        DATA_REPO,
        "dataset",
        prefix,
        raise_on_error=True,
    )
    expected = [
        f"{prefix}/{p.relative_to(bundles_dir)}" for p in sorted(bundles_dir.rglob("*.npz"))
    ]
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    missing = hub.verify_repo_paths_uploaded(
        api, DATA_REPO, expected, path_in_repo=prefix, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(
            f"bundle upload verify FAILED: {len(missing)} missing (first {missing[:3]})"
        )
    print(f"[dvf-upload] {len(expected)} bundle files verified at {url or prefix}", flush=True)
    return 0


def cmd_fence(args) -> int:
    """Plan v10 s7/s9: pilot-anchored k-weighted fence + kill projection.

    Reads the reduced-pilot unit walls (the dvf + cms parity units written by
    the pilot pairs/units runs), k-weights them over the full unit list's
    implied k distribution (k_i ~ min(cap, floor(0.8 * n_common / 2)) from
    the parent digest's n_common — the plan s2 approximation; per-unit cost
    ~ (k_i / k_pilot)^2 with a 5% overhead floor, the s9 dense-solve scaling),
    writes fence_report.json + one parse-friendly stdout line, and under
    --enforce-kill exits rc=21 when the projected total exceeds
    --kill-gpu-hours (kill criterion 2 — a DESIGNED halt the dispatcher
    routes, never an anonymous crash, #1415).
    """
    import csv as _csv

    pilot_key = unit_key(GATE1_PAIR, "context")
    dvf_unit = json.loads((args.out_root / "pairs" / f"{pilot_key}.json").read_text())
    if "error" in dvf_unit:
        raise RuntimeError(f"dvf pilot unit errored: {dvf_unit['error']}")
    dvf_wall = float(dvf_unit["wall_s"])
    k_pilot = int(dvf_unit.get("k_unit") or dvf_unit["d"])
    cms_wall = dvf_wall * 3.0  # parent leg's 3x structure-unit heuristic (fallback)
    cms_wall_measured = False
    if args.cms_out_root is not None:
        cms_path = args.cms_out_root / "pairs" / f"{pilot_key}.json"
        if cms_path.exists():
            cms_unit = json.loads(cms_path.read_text())
            if "error" not in cms_unit:
                cms_wall = float(cms_unit["wall_s"])
                cms_wall_measured = True
    weights_dvf: list[float] = []
    weights_cms: list[float] = []
    with open(args.digest_csv) as fh:
        for row in _csv.DictReader(fh):
            b = row.get("battery")
            if b not in ("dvf_within", "xm_dvf", "cms_within"):
                continue
            try:
                n_c = float(row["n_common"])
            except (KeyError, TypeError, ValueError):
                continue
            k_i = max(1, min(PCA_K_CAP, int(0.8 * n_c) // 2))
            w = max(0.05, (k_i / max(k_pilot, 1)) ** 2)
            if b in ("dvf_within", "xm_dvf"):
                weights_dvf.append(w)
            if b in ("cms_within", "xm_dvf"):  # xm structure mirrors the xm dvf list
                weights_cms.append(w)
    if not weights_dvf or not weights_cms:
        raise RuntimeError(f"empty unit weight lists from digest csv {args.digest_csv}")
    nulls_factor = 1.5  # plan s9: P3 rotation-null battery booked <= 1.5x P2 points
    proj_dvf_total_s = dvf_wall * sum(weights_dvf) * (1.0 + nulls_factor)
    proj_cms_s = cms_wall * sum(weights_cms)
    projected_total_gpu_h = (proj_dvf_total_s + proj_cms_s) / 3600.0
    nsh = max(int(args.num_shards), 1)
    fence_s = int(2 * proj_dvf_total_s / nsh) + 900
    cms_fence_s = int(2 * proj_cms_s / nsh) + 900
    kill = projected_total_gpu_h > float(args.kill_gpu_hours)
    _atomic_write_json(
        args.out_root / "fence_report.json",
        {
            "pilot_unit": pilot_key,
            "dvf_pilot_wall_s": dvf_wall,
            "cms_pilot_wall_s": cms_wall,
            "cms_wall_measured": cms_wall_measured,
            "k_pilot": k_pilot,
            "n_units_dvf": len(weights_dvf),
            "n_units_cms": len(weights_cms),
            "sum_weights_dvf": sum(weights_dvf),
            "sum_weights_cms": sum(weights_cms),
            "nulls_factor": nulls_factor,
            "num_shards": nsh,
            "projected_total_gpu_h": projected_total_gpu_h,
            "kill_gpu_hours": float(args.kill_gpu_hours),
            "kill": kill,
            "fence_s": fence_s,
            "cms_fence_s": cms_fence_s,
            "metadata": _metadata(),
        },
    )
    print(
        f"[dvf-fence] FENCE={fence_s} CMS_FENCE={cms_fence_s} "
        f"PROJECTED_GPU_H={projected_total_gpu_h:.2f} KILL={int(kill)}",
        flush=True,
    )
    if kill and args.enforce_kill:
        return 21  # designed halt (plan s7 kill criterion 2), distinct rc
    return 0


def cmd_write_pairs(args) -> int:
    specs = build_pair_specs(args)
    payload = [[[sm, sc], [tm, tc]] for ((sm, sc), (tm, tc)) in specs]
    out = Path(args.write_pairs_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out)
    print(f"[dvf] wrote {len(payload)} pair specs to {args.write_pairs_out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        choices=[
            "stage",
            "stage-parent-inputs",
            "upload-parent-inputs",
            "gate1",
            "pairs",
            "nulls",
            "merge",
            "upload",
            "write-pairs",
            "migrate-keys",
            "fence",
        ],
        required=True,
    )
    ap.add_argument(
        "--fit-basis",
        choices=["ambient", "reduced"],
        default="ambient",
        help="fit basis (plan v10): ambient (parent, byte-identical default) or the "
        "per-(pair, arm, fold) shared train-fold-only PCA rank-k basis",
    )
    ap.add_argument("--store-root", type=Path, default=None)
    ap.add_argument(
        "--out-root", type=Path, default=Path("eval_results/issue_1689/derived_vs_free_B")
    )
    ap.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    ap.add_argument("--lambda-grid", choices=sorted(LAMBDA_GRIDS), default="ladder13")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--pairs-file", type=Path, default=None)
    ap.add_argument("--default-model", type=str, default=None)
    ap.add_argument("--pair-set", choices=["within-model", "cross-model"], default="within-model")
    ap.add_argument("--models", type=str, default=",".join(MODEL_SLUGS))
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument(
        "--expect-units",
        type=int,
        default=None,
        help="merge: hard-assert exactly N distinct qualified units are enumerated",
    )
    ap.add_argument("--rotation-draws", type=int, default=200)
    ap.add_argument("--row-limit", type=int, default=None, help="smoke: cap common rows")
    ap.add_argument("--dim-limit", type=int, default=None, help="smoke: cap hidden dims")
    ap.add_argument("--gate1-null-draws", type=int, default=40)
    ap.add_argument("--gate1-timing", action="store_true")
    ap.add_argument("--gate1-target", type=Path, default=None)
    ap.add_argument(
        "--gate1-checks",
        choices=["ladder", "battery", "both"],
        default="ladder",
        help="gate1 parity legs: ladder (parent default) | battery (plan v10 Gate 1a "
        "ambient no-op vs the published per-unit JSON) | both",
    )
    ap.add_argument("--gate1-battery-target", type=Path, default=None)
    ap.add_argument(
        "--stage-cells",
        type=str,
        default=None,
        help="stage: comma-separated <model_slug>/<condition> subset (sliced-input smoke)",
    )
    ap.add_argument(
        "--cms-out-root",
        type=Path,
        default=None,
        help="fence: cms out-root holding the structure pilot unit",
    )
    ap.add_argument(
        "--digest-csv",
        type=Path,
        default=REPO_ROOT / "eval_results/issue_1689/analyzer/dvf_unit_digest.csv",
        help="fence: parent per-unit digest (n_common column drives the k weighting)",
    )
    ap.add_argument("--kill-gpu-hours", type=float, default=30.0)
    ap.add_argument("--enforce-kill", action="store_true")
    ap.add_argument("--stage-headroom-gb", type=float, default=18.0)
    ap.add_argument(
        "--parent-ladder-dir", type=Path, default=Path("eval_results/issue_1689/ladder")
    )
    ap.add_argument(
        "--parent-inputs-root",
        type=Path,
        default=REPO_ROOT / "eval_results/issue_1689",
        help="stage/upload-parent-inputs: local eval_results root the parent-input "
        "set (PARENT_INPUT_SINGLES + PARENT_INPUT_TREES) is rooted at",
    )
    ap.add_argument("--write-pairs-out", type=Path, default=None)
    args = ap.parse_args()

    if args.phase in ("stage", "gate1", "pairs") and args.store_root is None:
        ap.error(f"--phase {args.phase} requires --store-root")
    if args.phase == "write-pairs" and args.write_pairs_out is None:
        ap.error("--phase write-pairs requires --write-pairs-out")
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda but torch.cuda.is_available() is False")

    print(
        f"[dvf] phase={args.phase} pair_set={args.pair_set} device={args.device} "
        f"shard={args.shard_index}/{args.num_shards} row_limit={args.row_limit} "
        f"dim_limit={args.dim_limit} fit_basis={fit_basis_of(args)}",
        flush=True,
    )
    dispatch = {
        "stage": cmd_stage,
        "stage-parent-inputs": cmd_stage_parent_inputs,
        "upload-parent-inputs": cmd_upload_parent_inputs,
        "gate1": cmd_gate1,
        "pairs": cmd_pairs,
        "nulls": cmd_nulls,
        "merge": cmd_merge,
        "upload": cmd_upload,
        "write-pairs": cmd_write_pairs,
        "migrate-keys": cmd_migrate_keys,
        "fence": cmd_fence,
    }
    return dispatch[args.phase](args)


if __name__ == "__main__":
    rc = main()
    # C-extension shutdown-race workaround (gotchas.md PyGILState_Release):
    # flush, then bypass finalize-time teardown. All writes use explicit
    # handles + os.replace, so atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
