#!/usr/bin/env python3
"""#1768 inline round, extension: OPERATOR-SVD key-value read on the realized map update.

Asks whether the realized post-fine-tuning map update is a low-rank KEY-VALUE
write, and whether its keys/values are the ones the data-augmented refit
predicts.

The object is the OPERATOR update ``dM_real = M+ - M0`` -- the fitted ridge
operators themselves, NOT the data-space shift matrix (round 1's rank reads
already covered that).

WHY THIS IS A REFIT, NOT A LOAD
-------------------------------
Neither round 1 nor the map-augmentation round persisted the ridge W payloads
(both store scalar reads only, and there is no weights prefix on the Hub), so
the operators must be re-derived from the activation stores. The one saving
that makes this cheap: the SELECTED ridge lambda for every map is committed in
``eval_results/issue_1768/map_augmentation/cells/*.json``, so each operator is
one eigh plus one solve at the KNOWN lambda -- never the 23-point val scan.
Refitting at the committed lambda on the committed splits reproduces the
round's own operators exactly (asserted against the committed heldout_r2).

KEYS vs VALUES -- pinned by construction, not by convention
-----------------------------------------------------------
The raw operator ``A`` acts on ROW vectors: ``v = c @ A`` with A of shape
(H_in, D_out). In ``A = P S Q^T`` the LEFT vectors P therefore live in the
CONTEXT space (KEYS) and the RIGHT vectors Q in the ANSWER space (VALUES).
Because H_in == D_out == 3584 here, a transposition would NOT crash -- it would
silently swap the two reads -- so ``_assert_kv_orientation`` verifies
``key @ A ~= sigma * value`` on the top pair before any alignment is computed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1768_fit as F  # noqa: E402
import issue1768_lasttoken_fit as LTF  # noqa: E402
import issue1768_directions as DIR  # noqa: E402
import issue1768_map_augmentation as MA  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.opkv")

RESULTS_DIR = MA.RESULTS_DIR
TOP_K = 5  # singular pairs reported for the key/value alignment bars
SVD_Q = 32  # low-rank probe depth (the update is rank <= K_train_pairs anyway)
N_NULL = 200  # norm-matched random-direction null draws
NULL_SEED = 1768


def _phase(name: str) -> None:
    print(f"[phase={name}]", flush=True)


# ── operator construction at the COMMITTED lambda ───────────────────────────


_RB_CACHE: dict | None = None
_WU_CACHE: dict = {}


def _rb_stacks(out_root: Path) -> dict:
    """The four fleet r_B stacks, via round 1's OWN resolver.

    r_B is a BANKED tensor line (#1112/#1315/#1434 analysis_tensors), shape
    (n_layers, hidden) indexed rb[layer] -- NOT weight-derived, so no model
    download.
    """
    global _RB_CACHE
    if _RB_CACHE is None:
        _RB_CACHE = DIR.load_rb_tensors(out_root)
        logger.info("[opkv] r_B stacks loaded: %s", {k: v.shape for k, v in _RB_CACHE.items()})
    return _RB_CACHE


def _wu_marker_row() -> tuple[np.ndarray | None, str]:
    """Base lm_head row for the marker token, via a safetensors SLICE (fail-soft)."""
    if "row" in _WU_CACHE:
        return _WU_CACHE["row"], _WU_CACHE["note"]
    try:
        row = np.asarray(DIR.load_wu_row(X.BASE_MODEL), dtype=np.float64)
        _WU_CACHE.update(row=row, note="safetensors slice of base lm_head")
    except Exception as exc:  # noqa: BLE001 -- optional target, never fatal
        _WU_CACHE.update(row=None, note=f"unavailable: {type(exc).__name__}: {exc}")
        logger.warning("[opkv] W_U marker row unavailable: %s", exc)
    return _WU_CACHE["row"], _WU_CACHE["note"]


def _fit_operator_at_lambda(Xd, Yd, tr, lam, dev, block):
    """Ridge operator at a KNOWN lambda: one eigh + one solve, no val scan.

    Uses the reference streaming factorization so the standardizer, the Gram
    accumulation order and the solve are identical to the round's own fits;
    only the lambda SEARCH is skipped (its answer is read from the committed
    cell JSON).
    """
    import torch

    import issue779_ffc_n1m_fits as n1m

    fac = n1m._ridge_factorize(Xd, Yd, tr, dev, block)
    W = fac["U"] @ (fac["UtXtY"] / (fac["s_eig"] + float(lam))[:, None])
    payload = {
        "kind": "ridge",
        "selected_lambda": float(lam),
        "xmu": fac["xmu"].detach().cpu().to(torch.float32),
        "xsd": fac["xsd"].detach().cpu().to(torch.float32),
        "ymu": fac["ymu"].detach().cpu().to(torch.float32),
        "W": W.detach().cpu().to(torch.float32),
    }
    del fac, W
    return payload


def _augmented_operator(C0, V0, tr, T, S, w, lam, dev, block):
    """The augmented operator M-hat+(w) at a KNOWN lambda (same saving)."""
    import torch

    fac_parts, XtY_corpus, Ts, ymu = MA._augmented_blocks(C0, V0, tr, T, S, w, dev, block)
    S_t = torch.as_tensor(S, dtype=torch.float64, device=dev)
    XtY = MA._augmented_xty(XtY_corpus, Ts, S_t, ymu, w, None, dev)
    UtXtY = fac_parts["U"].T @ XtY
    W = fac_parts["U"] @ (UtXtY / (fac_parts["s_eig"] + float(lam))[:, None])
    payload = {
        "kind": "ridge",
        "selected_lambda": float(lam),
        "xmu": fac_parts["xmu"].detach().cpu().to(torch.float32),
        "xsd": fac_parts["xsd"].detach().cpu().to(torch.float32),
        "ymu": fac_parts["ymu"].detach().cpu().to(torch.float32),
        "W": W.detach().cpu().to(torch.float32),
    }
    del fac_parts, XtY_corpus, Ts, S_t, XtY, UtXtY, W
    return payload


# ── SVD reads ───────────────────────────────────────────────────────────────


def _assert_kv_orientation(A, keys, values, svals, tol=5e-3):
    """Fail loud unless keys/values are oriented as A maps key -> sigma*value.

    H_in == D_out here, so a transposed convention is silent. This is the guard.
    """
    k0, v0, s0 = keys[:, 0], values[:, 0], float(svals[0])
    got = k0 @ A
    ref = s0 * v0
    denom = max(1e-30, float(np.linalg.norm(ref)))
    rel = float(np.linalg.norm(got - ref)) / denom
    assert rel < tol, (
        f"key/value orientation check FAILED (rel={rel:.3e}): "
        "left singular vectors are not the context-side keys for this operator"
    )
    return rel


def _spectrum(A, q=SVD_Q):
    """Top-q singular values + EXACT participation ratio and top-k shares.

    The full 3584x3584 fp64 SVD is unnecessary: sum(s^2) == ||A||_F^2 and
    sum(s^4) == ||A^T A||_F^2, both exact and cheap, so the participation
    ratio PR = (sum s^2)^2 / sum s^4 is EXACT even though only the top q
    singular values are computed. Top-k shares are then exact too (top-k
    squared mass over the exact total).
    """
    import torch

    At = torch.as_tensor(A, dtype=torch.float64)
    fro2 = float((At * At).sum())
    G = At.T @ At
    sum_s4 = float((G * G).sum())
    del G
    U, S, V = torch.svd_lowrank(At, q=min(q, min(At.shape) - 1), niter=8)
    s = S.numpy().astype(np.float64)
    pr = (fro2**2) / sum_s4 if sum_s4 > 0 else float("nan")
    return {
        "keys": U.numpy().astype(np.float64),  # left vectors = CONTEXT side
        "values": V.numpy().astype(np.float64),  # right vectors = ANSWER side
        "svals": s,
        "fro2_exact": fro2,
        "sum_s4_exact": sum_s4,
        "participation_ratio_exact": pr,
        "top1_share": float(s[0] ** 2 / fro2) if fro2 > 0 else float("nan"),
        "top5_share": float((s[:5] ** 2).sum() / fro2) if fro2 > 0 else float("nan"),
        "top32_share": float((s**2).sum() / fro2) if fro2 > 0 else float("nan"),
        "svd_q": int(len(s)),
    }


def _jsonable_spectrum(spec: dict) -> dict:
    """Persistable view of a spectrum block.

    Drops the (3584, q) singular-vector matrices and listifies the singular
    values -- a numpy array is not JSON-serializable, and leaving one in the
    record kills the write AFTER the expensive refit has already run.
    """
    out = {k: v for k, v in spec.items() if k not in ("keys", "values")}
    out["svals"] = [float(x) for x in spec["svals"]]
    return out


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _align_block(vecs, targets: dict, rng) -> dict:
    """|cos| of each of the top-K singular vectors against each named target.

    Sign of a singular vector is arbitrary (SVD gauge), so ABSOLUTE cosine is
    the meaningful quantity; the norm-matched random null is reported from the
    same |cos| statistic so the comparison is like-for-like.
    """
    d = vecs.shape[0]
    null = np.abs(np.array([_cos(rng.standard_normal(d), vecs[:, 0]) for _ in range(N_NULL)]))
    out = {
        "null_abscos_mean": float(null.mean()),
        "null_abscos_p95": float(np.quantile(null, 0.95)),
        "null_abscos_analytic_sd": float(1.0 / np.sqrt(d)),
        "n_null_draws": N_NULL,
        "targets": {},
    }
    for name, t in targets.items():
        if t is None:
            out["targets"][name] = {"computed": False, "reason": "target unavailable"}
            continue
        tu = _unit(np.asarray(t, dtype=np.float64))
        per = [abs(_cos(vecs[:, j], tu)) for j in range(min(TOP_K, vecs.shape[1]))]
        # subspace alignment: norm of the target's projection on the top-K span
        Q = vecs[:, : min(TOP_K, vecs.shape[1])]
        proj = float(np.linalg.norm(Q.T @ tu))
        out["targets"][name] = {
            "computed": True,
            "abscos_per_component": [float(x) for x in per],
            "abscos_top1": float(per[0]),
            "abscos_max_over_topk": float(max(per)),
            "topk_subspace_projection": proj,
        }
    return out


def _principal_angles(Aq: np.ndarray, Bq: np.ndarray) -> dict:
    """Cosines of principal angles between two column subspaces (orthonormalized)."""
    qa, _ = np.linalg.qr(Aq)
    qb, _ = np.linalg.qr(Bq)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return {
        "cosines": [float(x) for x in s],
        "mean_cos": float(s.mean()),
        "max_cos": float(s.max()),
        "min_cos": float(s.min()),
        "k": int(len(s)),
    }


def _effective_rank(svals, fro2) -> int:
    """Numerical rank: smallest k whose squared mass reaches 99% of the total."""
    c = np.cumsum(svals**2)
    if fro2 <= 0:
        return 1
    idx = int(np.searchsorted(c, 0.99 * min(fro2, c[-1])) + 1)
    return max(1, min(idx, len(svals)))


# ── per-arm driver ──────────────────────────────────────────────────────────


def _committed_cell(arm_id: str, layer: int) -> dict:
    p = RESULTS_DIR / "cells" / f"{arm_id}_L{layer}.json"
    assert p.is_file(), f"missing committed cell (needed for the selected lambdas): {p}"
    return json.loads(p.read_text())


def run_arm(out_root: Path, arm_id: str, layer: int, pos_path: str, block: int) -> dict:
    import torch

    dev = MA._device()
    rng = np.random.default_rng(NULL_SEED)
    committed = _committed_cell(arm_id, layer)
    cache = out_root / "lt_answer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cell = LTF.build_cell(out_root, cache, arm_id, layer, MA.POSITION)
    tr, _val, te = F._split_idx(cell["split"])
    C0, V0, Cp, Vp, Vt = (
        cell["C0"],
        cell["V0"],
        cell["Cplus"],
        cell["Vplus"],
        cell["Vplus_tf"],
    )
    T, S = MA.load_train_pairs(out_root, pos_path, layer)

    lam0 = committed["realized"]["M0"]["selected_lambda"]
    lamp = committed["realized"]["Mplus"]["selected_lambda"]
    t0 = time.time()
    p0 = _fit_operator_at_lambda(C0, V0, tr, lam0, dev, block)
    pp = _fit_operator_at_lambda(Cp, Vp, tr, lamp, dev, block)
    logger.info("[opkv] %s L%d operators refit (%.0fs)", arm_id, layer, time.time() - t0)

    # reproduction check against the committed reads (production n, same lambda)
    pred0 = F._apply_payload(p0, C0[te], dev)
    r2_0 = F._pooled_r2(pred0, V0[te])
    repro = {
        "m0_r2_refit": r2_0,
        "m0_r2_committed": committed["realized"]["M0"]["heldout_r2"],
        "m0_r2_absdiff": abs(r2_0 - committed["realized"]["M0"]["heldout_r2"]),
    }

    A0 = MA._raw_operator(p0)
    dA_real = MA._raw_operator(pp) - A0

    # the predicted update at the arm's best-closing mass leg (its lambda is committed)
    legs = [lg for lg in committed["legs"] if lg["weight"] > 0]
    best = max(legs, key=lambda lg: lg["frac_change_closed"])
    ph = _augmented_operator(C0, V0, tr, T, S, best["weight"], best["selected_lambda"], dev, block)
    dA_hat = MA._raw_operator(ph) - A0

    spec_real = _spectrum(dA_real)
    spec_hat = _spectrum(dA_hat)
    orient_rel = _assert_kv_orientation(
        dA_real, spec_real["keys"], spec_real["values"], spec_real["svals"]
    )

    # ── alignment targets ────────────────────────────────────────────────────
    Tc = T.mean(axis=0)  # training-context centroid (raw space)
    # whitened gate direction Sigma^-1 c_src, ridge-regularized at the SAME lambda
    Ctr = C0[tr]
    mu = Ctr.mean(axis=0)
    Cc = Ctr - mu
    Sig = (Cc.T @ Cc) / max(1, len(tr) - 1)
    gate = np.linalg.solve(Sig + float(lam0) * np.eye(Sig.shape[0]) / max(1, len(tr)), Tc - mu)
    del Sig, Cc

    M0_on_T = F._apply_payload(p0, T, dev)
    resid = S - M0_on_T
    delta_vec = resid.mean(axis=0)
    rc = resid - resid.mean(axis=0)
    resid_pc1 = np.linalg.svd(rc, full_matrices=False)[2][0]  # top PC of the residuals
    w_mean = (Vt[te] - V0[te]).mean(axis=0)

    key_targets = {
        "training_context_centroid": Tc,
        "whitened_gate_Sigma_inv_c_src": gate,
        "ridge_natural_key_from_augmentation": spec_hat["keys"][:, 0],
    }
    # r_B: the BANKED fleet read-out direction for this behavior at this layer
    arm = {a.arm_id: a for a in X.all_arms()}[arm_id]
    rb_stack = _rb_stacks(out_root).get(arm.beh_key)
    rb_vec = None
    rb_note = f"no r_B stack for beh_key={arm.beh_key}"
    if rb_stack is not None:
        if layer < rb_stack.shape[0]:
            rb_vec = rb_stack[layer]
            rb_note = f"banked rb[{arm.beh_key}][{layer}], stack shape {rb_stack.shape}"
        else:
            rb_note = f"layer {layer} >= r_B stack depth {rb_stack.shape[0]}"
    wu_vec, wu_note = (None, "not a marker arm")
    if arm.kind == "marker":
        wu_vec, wu_note = _wu_marker_row()

    value_targets = {
        # delta IS the mean training-pair map residual in this construction --
        # the brief's (a) and (c) are the same object, reported once
        "delta_eq_mean_map_residual": delta_vec,
        "map_residual_pc1": resid_pc1,
        "mean_measured_write_wtf": w_mean,
        "rB_behavior_readout": rb_vec,
        "wu_marker_unembedding_row": wu_vec,
    }
    target_provenance = {
        "rB_behavior_readout": rb_note,
        "wu_marker_unembedding_row": wu_note,
    }

    k_eff = _effective_rank(spec_hat["svals"], spec_hat["fro2_exact"])
    kk = max(1, min(k_eff, spec_real["keys"].shape[1], spec_hat["keys"].shape[1]))
    match = {
        "k_effective_of_predicted_update": int(k_eff),
        "k_used": int(kk),
        "key_subspace_principal_angles": _principal_angles(
            spec_real["keys"][:, :kk], spec_hat["keys"][:, :kk]
        ),
        "value_subspace_principal_angles": _principal_angles(
            spec_real["values"][:, :kk], spec_hat["values"][:, :kk]
        ),
        "at_mass": best["mass"],
        "at_weight": best["weight"],
    }

    out = {
        "arm_id": arm_id,
        "layer": layer,
        "pooling": MA.POSITION,
        "method": X.arm_method(arm_id),
        "K_train_pairs": int(T.shape[0]),
        "operator_shape": list(dA_real.shape),
        "kv_orientation_check_rel_err": orient_rel,
        "keys_side_established_by_assert": (
            "LEFT singular vectors of the raw row-vector operator A (v = c @ A) are the "
            "context-side KEYS; RIGHT singular vectors are the answer-side VALUES. "
            "Equivalently, in the COLUMN convention M = A^T (v = M c) the KEYS are M's "
            "RIGHT singular vectors -- the two phrasings agree once the operator "
            "orientation is fixed. VERIFIED per cell by key @ A ~= sigma * value."
        ),
        "value_target_provenance": target_provenance,
        "refit_reproduction": repro,
        "spectrum_real": _jsonable_spectrum(spec_real),
        "spectrum_predicted": _jsonable_spectrum(spec_hat),
        "lora_rank_note": (
            "LoRA WEIGHT updates are rank-32 per adapted layer by construction; this "
            "spectrum is of the fitted context->answer OPERATOR update, a different "
            "object, so the two ranks are not required to agree — reported split by method"
        ),
        "key_alignment": _align_block(spec_real["keys"], key_targets, rng),
        "value_alignment": _align_block(spec_real["values"], value_targets, rng),
        "match_read": match,
        **MA._meta(),
    }
    del dA_real, dA_hat, A0, p0, pp, ph
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument(
        "--layers",
        default=",".join(str(x) for x in X.LAYERS),
        help="comma-separated layers; the headline layer carries the figure",
    )
    ap.add_argument("--arms", default="")
    ap.add_argument("--block", type=int, default=50_000)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "recompute cells even when the JSON exists. Needed because in the SHARED "
            "repo root a deleted-but-committed file can be RESTORED under a running "
            "pass (observed: 4 deleted cells reappeared ~70 s after deletion and were "
            "then skipped as 'present'), so 'delete then regenerate' is not reliable "
            "here -- overwrite is."
        ),
    )
    ap.add_argument("--phase", default="all")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        import issue779_ffc_n1m_fits as _n1m  # noqa: F401
        import matplotlib as _mpl  # noqa: F401
        import torch as _torch  # noqa: F401

        from explore_persona_space.analysis.paper_plots import (  # noqa: F401
            paper_palette,
            savefig_paper,
            set_paper_style,
        )
        from explore_persona_space.orchestrate import hub as _hub  # noqa: F401

        assert hasattr(_n1m, "_ridge_factorize")
        assert hasattr(MA, "_augmented_blocks")
        print("import-check ok")
        return 0

    assert args.out_root is not None, "--out-root is required"
    out_root = Path(args.out_root)
    picks = MA.arm_picks()
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",") if a.strip()}
        picks = [p for p in picks if p["arm_id"] in keep]
        assert picks, (keep, "no matching arms")
    arms = [p["arm_id"] for p in picks]
    srcs = MA._mix_sources(picks)

    if args.phase in ("all", "stage"):
        MA.stage_inputs(out_root, arms)
        # The per-row training-pair embeddings were captured on the pod and are
        # already on the Hub, so FETCH them. Never call MA.build_train_pairs
        # here: it runs a base-model teacher-forced pass, which on this GPU-less
        # VM would mean a ~15 GB model download and a CPU forward sweep for
        # tensors that already exist.
        from explore_persona_space.orchestrate import hub

        need = sorted({srcs[a]["pos_path"] for a in arms})
        for pos_path in need:
            slug = pos_path.replace("/", "__")
            for suffix in (".pt", ".meta.json"):
                tgt = out_root / "train_pairs" / f"{slug}{suffix}"
                hub.stage_hub_file(
                    X.HF_DATA_REPO,
                    f"{MA.HF_SUBPREFIX}/train_pairs/{slug}{suffix}",
                    tgt,
                )
        logger.info("[stage] %d training-pair stores fetched from the Hub", len(need))

    if args.phase in ("all", "opkv"):
        _phase("opkv")
        dest_dir = RESULTS_DIR / "operator_kv"
        dest_dir.mkdir(parents=True, exist_ok=True)
        cache = out_root / "lt_answer_cache"
        cache.mkdir(parents=True, exist_ok=True)
        layers = [int(x) for x in args.layers.split(",")]
        todo = [(a, li) for a in arms for li in layers]
        k = 0
        # ARM-OUTER so one download of each arm's answer stores serves every layer
        for arm_id in arms:
            if args.overwrite or any(
                not (dest_dir / f"{arm_id}_L{li}.json").exists() for li in layers
            ):
                MA._prewarm_arm_cache(cache, arm_id)
            for layer in layers:
                k += 1
                dest = dest_dir / f"{arm_id}_L{layer}.json"
                if dest.exists() and not args.overwrite:
                    logger.info(
                        "[opkv] unit %d/%d %s L%d: present, skip", k, len(todo), arm_id, layer
                    )
                    continue
                t0 = time.time()
                rec = run_arm(out_root, arm_id, layer, srcs[arm_id]["pos_path"], args.block)
                MA._atomic_json(dest, rec)
                logger.info(
                    "[opkv] unit %d/%d %s L%d: top1=%.3f top5=%.3f PR=%.1f "
                    "key_match=%.3f val_match=%.3f rB=%s elapsed=%.0fs",
                    k,
                    len(todo),
                    arm_id,
                    layer,
                    rec["spectrum_real"]["top1_share"],
                    rec["spectrum_real"]["top5_share"],
                    rec["spectrum_real"]["participation_ratio_exact"],
                    rec["match_read"]["key_subspace_principal_angles"]["mean_cos"],
                    rec["match_read"]["value_subspace_principal_angles"]["mean_cos"],
                    rec["value_alignment"]["targets"]["rB_behavior_readout"].get("computed"),
                    time.time() - t0,
                )
            MA._drop_arm_cache(cache, arm_id)
        logger.info("[opkv] all %d units complete", len(todo))

    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
