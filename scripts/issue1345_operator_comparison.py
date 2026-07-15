#!/usr/bin/env python
"""Issue #1345 Phase 6 — operator comparison across framings (frozen layers).

Leg A (EVERY unordered pair; unpaired-safe reads labeled in metadata):
  - raw operator cosine (flattened primal betas — meaningful across regimes
    because both operators live in the SAME model's residual basis) + the
    random-orthogonal-rotation chance band on the raw cosine (selection-
    symmetric: no fitting on either side),
  - singular-spectrum cosine = the CLOSED-FORM optimum of the two-sided
    orthogonal operator-Procrustes problem (von Neumann trace bound) —
    rotation-invariant, reported descriptively (its rotation null is
    degenerate by construction; named in metadata),
  - principal angles between output singular subspaces (reused
    issue825_crossmodel_map_transfer.principal_angles; unpaired-safe).

For the conv-PAIRED chat<->no-template pair additionally:
  - the map_alignment activation-fitted Procrustes-aligned cosine vs the
    step-5 random-rotation null (REUSED: issue825_map_alignment
    _procrustes_cosine_null; calibration anchor: #825 base<->instruct 0.6864),
  - Leg B: the #825 Result-2.5 data-paired general-linear reparameterization
    (REUSED: issue825_map_alignment._layer_battery — ceilings, alignment R^2,
    A_ans o M o A_ctx_rev recovered R^2, both directions, linear + orthogonal
    variants) with A/B capacity (GCV lambda + effective rank) reported, and
  - the MATCHED-CAPACITY reparam nulls at the headline layer (plan §3
    Δ_reparam): (a) the answer-shuffled-fit center operator and (b) a random
    orthogonal rotation wrapped around M_j — same ridge/λ-grid capacity.

Story pairs get NO reparameterization verdict (plan H3): `align_pair`
n_common > 0 is asserted before ANY alignment fit; story pairs share zero
conv_ids and never reach Leg B.

Outputs: eval_results/issue_1345/operator_comparison_{model_slug}_{arm}.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue825_fit_cells as fc  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import issue1345_common as c  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue1345_cross_regime_transfer import load_arm_xy, subset_rows  # noqa: E402
from issue1345_fit_cells import load_matched, load_regime_bundle  # noqa: E402

FROZEN_LAYERS = cm.FROZEN_LAYERS
L19 = 19


def _dev() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.float64).to(_dev())


# ---------------------------------------------------------------------------
# Leg A reads
# ---------------------------------------------------------------------------
def raw_cosine_with_rotation_null(
    beta_a: torch.Tensor, beta_b: torch.Tensor, *, n_draws: int, seed: int
) -> dict:
    """Raw vec-cosine + random two-sided-rotation chance band (no fitting)."""
    va = beta_a.reshape(-1)
    vb = beta_b.reshape(-1)
    raw = float((va @ vb) / (va.norm() * vb.norm() + 1e-12))
    gen = torch.Generator().manual_seed(seed)
    d_in, d_out = beta_b.shape
    va_n = va / (va.norm() + 1e-12)
    draws = []
    for _ in range(n_draws):
        q1 = ma._random_orthogonal(d_in, gen).to(beta_b.device)
        q2 = ma._random_orthogonal(d_out, gen).to(beta_b.device)
        vm = (q1.T @ beta_b @ q2).reshape(-1)
        draws.append(float((vm @ va_n) / (vm.norm() + 1e-12)))
    arr = np.asarray(draws)
    return {
        "raw_cosine": raw,
        "rotation_null": {
            "n_draws": int(n_draws),
            "null_mean": float(arr.mean()) if len(arr) else float("nan"),
            "null_std": float(arr.std()) if len(arr) else float("nan"),
            "null_p975": float(np.quantile(arr, 0.975)) if len(arr) else float("nan"),
            "analytic_sd_1_over_d": float(1.0 / beta_b.shape[0]),
        },
    }


def spectrum_cosine(beta_a: torch.Tensor, beta_b: torch.Tensor) -> float:
    """Closed-form two-sided orthogonal-Procrustes-aligned cosine.

    max over orthogonal Q1,Q2 of cos(vec(beta_a), vec(Q1^T beta_b Q2)) =
    sum_k s_k(a)*s_k(b) / (||a||_F ||b||_F) (von Neumann). Rotation-invariant
    (its rotation null is degenerate) — a descriptive similarity ceiling.
    """
    sa = torch.linalg.svdvals(beta_a)
    sb = torch.linalg.svdvals(beta_b)
    return float((sa * sb).sum() / (sa.norm() * sb.norm() + 1e-12))


def alignment_capacity(x_src: torch.Tensor, x_dst: torch.Tensor) -> dict:
    """GCV lambda + effective rank (dof) of the full-data ridge alignment map."""
    _beta, lam = cm.fit_primal_beta(x_src.cpu().numpy(), x_dst.cpu().numpy())
    xs = (x_src - x_src.mean(0)) / (x_src.std(0) + 1e-9)
    g = xs @ xs.T
    w = torch.clamp(torch.linalg.eigh(g).eigenvalues, min=0.0)
    dof = float((w / (w + lam)).sum())
    return {"lambda": float(lam), "effective_rank_dof": dof, "n": int(x_src.shape[0])}


# ---------------------------------------------------------------------------
# Leg B matched-capacity reparam nulls (headline layer; plan §3 Δ_reparam)
# ---------------------------------------------------------------------------
def reparam_null_battery(
    data: dict, folds: np.ndarray, layer: int, *, n_draws: int, seed: int
) -> dict:
    """Held-out recovery R^2 of the reparam chain with the CENTER operator
    nulled at matched capacity (same ridge core, same λ grid, same A fits):
      shuffle_fit — M_j fit on conversation-level-shuffled train answers;
      rotation   — random orthogonal rotations wrapped around the true M_j.
    Directions: b2i recovers regime i (via M_b), i2b recovers regime b.
    """
    xi, yi = data["Xi"][layer], data["Yi"][layer]
    xb, yb = data["Xb"][layer], data["Yb"][layer]
    dev = _dev()
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    n = xi.shape[0]
    d = xi.shape[1]

    fold_state: dict[int, dict] = {}
    for k in range(ma.N_FOLDS):
        tr = folds != k
        if (folds == k).sum() == 0 or tr.sum() < 3:
            continue
        trt = torch.as_tensor(tr)
        tet = torch.as_tensor(folds == k)
        preps = {
            "Xi": ma._ridge_prep(xi[trt]),
            "Xb": ma._ridge_prep(xb[trt]),
            "Yb": ma._ridge_prep(yb[trt]),
            "Yi": ma._ridge_prep(yi[trt]),
        }
        fold_state[k] = {
            "tr": trt,
            "te": tet,
            "preps": preps,
            # true alignment outputs (A_ctx_rev / A_ctx), cached once per fold
            "xbhat": ma._ridge_predict(preps["Xi"], xb[trt], xi[tet]),
            "xihat": ma._ridge_predict(preps["Xb"], xi[trt], xb[tet]),
            "mu": {
                "Xb": xb[trt].mean(0),
                "Xi": xi[trt].mean(0),
                "Yb": yb[trt].mean(0),
                "Yi": yi[trt].mean(0),
            },
        }

    def _chain(direction: str, center_fn) -> float:
        """Held-out pooled R^2 of A_ans o CENTER o A_ctx_rev for one draw."""
        ss_res, ss_tot = 0.0, 0.0
        for st in fold_state.values():
            trt, tet, preps = st["tr"], st["te"], st["preps"]
            if direction == "b2i":  # recover Yi from Xi via regime-b center
                ybhat = center_fn(st, "b")
                pred = ma._ridge_predict(preps["Yb"], yi[trt], ybhat)
                true = yi[tet]
            else:  # i2b: recover Yb from Xb via regime-i center
                yihat = center_fn(st, "i")
                pred = ma._ridge_predict(preps["Yi"], yb[trt], yihat)
                true = yb[tet]
            ss_res += float(((true - pred) ** 2).sum())
            ss_tot += float(((true - true.mean(0)) ** 2).sum())
        return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot

    def _shuffled_center(perm: np.ndarray):
        pt = torch.as_tensor(perm)

        def fn(st, side):
            trt = st["tr"]
            if side == "b":
                y_shuf = yb[pt][trt]
                return ma._ridge_predict(st["preps"]["Xb"], y_shuf, st["xbhat"])
            y_shuf = yi[pt][trt]
            return ma._ridge_predict(st["preps"]["Xi"], y_shuf, st["xihat"])

        return fn

    def _rotated_center(qc: torch.Tensor, qa: torch.Tensor):
        def fn(st, side):
            mu = st["mu"]
            if side == "b":
                x_rot = (st["xbhat"] - mu["Xb"]) @ qc + mu["Xb"]
                yhat = ma._ridge_predict(st["preps"]["Xb"], yb[st["tr"]], x_rot)
                return (yhat - mu["Yb"]) @ qa + mu["Yb"]
            x_rot = (st["xihat"] - mu["Xi"]) @ qc + mu["Xi"]
            yhat = ma._ridge_predict(st["preps"]["Xi"], yi[st["tr"]], x_rot)
            return (yhat - mu["Yi"]) @ qa + mu["Yi"]

        return fn

    # Conversation-level permutations (rows of a conversation move together)
    conv = data["conv"]
    uniq, inv = np.unique(conv, return_inverse=True)
    row_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]

    def _conv_perm() -> np.ndarray:
        cp = rng.permutation(len(uniq))
        out = np.concatenate([row_of[k] for k in cp])
        assert len(out) == n
        return out

    results: dict = {}
    for direction in ("b2i", "i2b"):
        shuf = [_chain(direction, _shuffled_center(_conv_perm())) for _ in range(n_draws)]
        rot = []
        for _ in range(n_draws):
            qc = ma._random_orthogonal(d, gen).to(dev)
            qa = ma._random_orthogonal(d, gen).to(dev)
            rot.append(_chain(direction, _rotated_center(qc, qa)))
        results[direction] = {
            "shuffle_fit": {
                "draws": [float(v) for v in shuf],
                "mean": float(np.nanmean(shuf)),
                "max": float(np.nanmax(shuf)),
            },
            "rotation": {
                "draws": [float(v) for v in rot],
                "mean": float(np.nanmean(rot)),
                "max": float(np.nanmax(rot)),
            },
            "null_recovery_r2": float(max(np.nanmean(shuf), np.nanmean(rot))),
            "n_draws_per_type": int(n_draws),
        }
    return results


# ---------------------------------------------------------------------------
# Battery per (model, arm)
# ---------------------------------------------------------------------------
def run_model_arm(
    bundles: dict,
    matched: dict,
    model: str,
    arm: str,
    out_dir: Path,
    *,
    seed: int,
    n_rot_draws: int,
    n_reparam_null_draws: int,
    include_r3: bool,
) -> None:
    regimes = [r for r in c.REGIMES if include_r3 or r != "r3"]
    full = {r: load_arm_xy(bundles[r], r, arm) for r in regimes}
    shared = matched["shared_r1r2_convs"]
    r3cfg = matched["per_model_r3_pair"].get(model) if include_r3 else None

    def _subset(regime: str, pair_kind: str) -> dict:
        if regime in ("r1", "r2"):
            ids = shared if pair_kind == "headline" else r3cfg["r12_convs"]
            return subset_rows(full[regime], ids)
        return subset_rows(full[regime], r3cfg["r3_story_ids"])

    pairs_out: dict = {}
    for a, b in c.UNORDERED_PAIRS:
        if not include_r3 and "r3" in (a, b):
            continue
        pair_kind = "headline" if {a, b} == {"r1", "r2"} else "r3pair"
        xa, xb_ = _subset(a, pair_kind), _subset(b, pair_kind)
        per_layer: dict = {}
        for layer in FROZEN_LAYERS:
            beta_a, lam_a = cm.fit_primal_beta(xa["X"][:, layer, :], xa["Y"][:, layer, :])
            beta_b, lam_b = cm.fit_primal_beta(xb_["X"][:, layer, :], xb_["Y"][:, layer, :])
            draws = n_rot_draws if layer == L19 else max(5, n_rot_draws // 10)
            rec = raw_cosine_with_rotation_null(beta_a, beta_b, n_draws=draws, seed=seed + layer)
            rec["spectrum_cosine_operator_procrustes_optimum"] = spectrum_cosine(beta_a, beta_b)
            rec["lambda_a"], rec["lambda_b"] = float(lam_a), float(lam_b)
            for k in (10, 50):
                cs = cm.principal_angles(beta_a, beta_b, k)
                rec[f"principal_angle_cos_k{k}"] = {
                    "mean_cos": float(np.mean(cs)),
                    "min_cos": float(np.min(cs)),
                }
            per_layer[str(layer)] = rec
        pairs_out[f"{a}~{b}"] = {
            "pair_kind": pair_kind,
            "aligned_variant": (
                "activation-procrustes (map_alignment) + reparam"
                if (a, b) == c.PAIRED_PAIR
                else "operator-space only (no conv pairing: spectrum-cosine optimum "
                "+ raw-cosine rotation band; NO data-paired Procrustes/reparam)"
            ),
            "per_layer": per_layer,
            "n_a": len(xa["conv_ids"]),
            "n_b": len(xb_["conv_ids"]),
        }

    # ---- Paired pair: activation-Procrustes + Result-2.5 reparam (Leg B) ----
    xa, xb_ = _subset("r1", "headline"), _subset("r2", "headline")
    al = cm.align_pair({"conv_ids": xa["conv_ids"]}, {"conv_ids": xb_["conv_ids"]})
    assert al["n_common"] > 0, (
        "align_pair n_common == 0 on the chat<->no-template pair — the reparam "
        "leg requires conv_id-paired rows (plan §4 Leg B mechanized assert)"
    )
    # Row alignment: both subsets are conv-sorted over the SAME conv set.
    assert np.array_equal(xa["conv_ids"], xb_["conv_ids"]), "paired rows misaligned"
    data = {
        "Xi": {layer: _t(xa["X"][:, layer, :]) for layer in FROZEN_LAYERS},
        "Yi": {layer: _t(xa["Y"][:, layer, :]) for layer in FROZEN_LAYERS},
        "Xb": {layer: _t(xb_["X"][:, layer, :]) for layer in FROZEN_LAYERS},
        "Yb": {layer: _t(xb_["Y"][:, layer, :]) for layer in FROZEN_LAYERS},
        "conv": xa["conv_ids"],
    }
    folds = fc._cv_folds(xa["conv_ids"], ma.N_FOLDS, seed)
    reparam: dict = {}
    for layer in FROZEN_LAYERS:
        battery = ma._layer_battery(data, folds, layer, do_orth=True)
        proc = ma._procrustes_cosine_null(
            data["Xb"][layer],
            data["Xi"][layer],
            data["Yb"][layer],
            data["Yi"][layer],
            n_draws=(n_rot_draws if layer == L19 else max(5, n_rot_draws // 10)),
            seed=seed + 7 + layer,
        )
        cap = {
            "A_ctx (r2->r1)": alignment_capacity(data["Xb"][layer], data["Xi"][layer]),
            "A_ans (r2->r1)": alignment_capacity(data["Yb"][layer], data["Yi"][layer]),
        }
        reparam[str(layer)] = {
            "battery": battery,
            "activation_procrustes": proc,
            "alignment_capacity": cap,
        }
        if layer == L19:
            reparam[str(layer)]["matched_capacity_nulls"] = reparam_null_battery(
                data, folds, layer, n_draws=n_reparam_null_draws, seed=seed + 13
            )
    # Δ_reparam (plan §3): min over directions of recovered - max(within-0.05, null)
    b19 = reparam[str(L19)]["battery"]
    nulls19 = reparam[str(L19)]["matched_capacity_nulls"]
    recov = {
        "b2i": b19["composition"]["linear"]["comp_samefn_b2i"],
        "i2b": b19["composition"]["linear"]["comp_samefn_i2b"],
    }
    within = {"b2i": b19["ceilings"]["within_instruct"], "i2b": b19["ceilings"]["within_base"]}
    delta_reparam_terms = {
        d: recov[d] - max(within[d] - c.DELTA_SAME_MARGIN, nulls19[d]["null_recovery_r2"])
        for d in ("b2i", "i2b")
    }
    delta_reparam = float(min(delta_reparam_terms.values()))

    slug = c.MODEL_SLUG[model]
    payload = {
        "metadata": c.metadata(seed, len(shared), "scripts/issue1345_operator_comparison.py"),
        "model": model,
        "model_slug": slug,
        "arm": arm,
        "frozen_layers": list(FROZEN_LAYERS),
        "headline_layer": L19,
        "direction_key": {
            "b2i": "reparam r2(no-template) operator recovered in r1(chat)",
            "i2b": "reparam r1(chat) operator recovered in r2(no-template)",
        },
        "pairs": pairs_out,
        "reparam_r1r2": reparam,
        "delta_reparam_l19": {
            "per_direction": {k: float(v) for k, v in delta_reparam_terms.items()},
            "delta_reparam": delta_reparam,
            "recovered_r2": recov,
            "within_r2": within,
            "margin": c.DELTA_SAME_MARGIN,
        },
        "calibration_anchor": {
            "base_instruct_aligned_cosine_825": 0.6864,
            "note": "aligned-cosine magnitudes read against the #825 cross-model anchor",
        },
    }
    c.write_json(out_dir / f"operator_comparison_{slug}_{arm}.json", payload)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--matched-dir", type=Path, default=c.MATCHED_DIR)
    ap.add_argument("--out-dir", type=Path, default=c.EVAL_DIR)
    ap.add_argument("--models", default="instruct,pretrained")
    ap.add_argument("--arms", default="prefix,context")
    ap.add_argument("--no-r3", action="store_true")
    ap.add_argument("--seed", type=int, default=cm.FIT_SEED)
    ap.add_argument("--rot-draws", type=int, default=50)
    ap.add_argument("--reparam-null-draws", type=int, default=c.N_REPARAM_NULL_DRAWS)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matched = load_matched(args.matched_dir)
    regimes = [r for r in c.REGIMES if not (args.no_r3 and r == "r3")]
    for model in args.models.split(","):
        assert model in c.MODELS, model
        bundles = {r: load_regime_bundle(args.turnstore_dir, model, r) for r in regimes}
        for arm in args.arms.split(","):
            assert arm in c.ARMS, arm
            run_model_arm(
                bundles,
                matched,
                model,
                arm,
                args.out_dir,
                seed=args.seed,
                n_rot_draws=args.rot_draws,
                n_reparam_null_draws=args.reparam_null_draws,
                include_r3=not args.no_r3,
            )
    print("[done] operator comparison complete", flush=True)


if __name__ == "__main__":
    main()
