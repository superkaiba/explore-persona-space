#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ℓ, σ, →, ‖·‖) in scientific docstrings/log messages.
"""Issue #841 follow-up (gru-source-only) Stage 0 — source-only-GRU Δ-atlas.

The single manipulated variable vs the parent's depth-GRU is the recurrence INPUT:
the parent GRU consumed the whole depth trajectory h_0..h_ℓ (prefix-informed); this
refits the SAME ``DepthGRU`` architecture on ONLY the source-layer state h_ℓ (+ the
layer-index embedding) — the information-matched regime of the affine ridge / per-
transition MLP. Everything else (model, cached ``pass_b`` trajectories, split seed 42,
val/test sets, GRU config, lr, loss, RMS-norm target, atlas metrics) is held at the
parent's values (issue841_common).

For each target space {raw σ≡1, RMS-norm σ_m} fit ONE source-only GRU on the parent's
stored 4000-context ``pass_b`` fit split (108k single-state examples), read the atlas
per one-step transition via ``DepthGRU.forward_single``:

- PRIMARY metric = identity-relative R² (predict-zero ≡ 0); COMPANION = mean-centered
  R²; plus median/p90/p99 raw Δ-error.
- RELOAD the parent's committed ridge/MLP/prefix-GRU atlas (``stage0_atlas.json``,
  never re-run); assert the parent's ridge 27/27 Stage-0 dominance over the prefix-GRU
  reproduces on load (fail-loud on a stale/wrong file); emit the per-space 27-transition
  win-count ``#{gru_source_only.r2_id > ridge.r2_id}`` and ``> prefix_gru.r2_id``.
- PERSIST convergence diagnostics per source-only fit (epochs-to-best-val, cap-hit flag,
  val-curve summary) and record whether the parent prefix-GRU artifact carries matching
  diagnostics (it does not — so the prefix-vs-source read is narrated descriptively with
  a convergence-parity-unverified caveat, plan §4.2).

``--verify-source-only-gru`` runs a self-check gate FIRST that exercises the EXACT
functions the entrypoints dispatch (``DepthGRU.forward_single`` / ``GruSourceOnlyMap``).
``--smoke`` fits on a 200-context subset for ONE transition (ℓ=13→14) through the SAME
dispatcher/loaders — the sweep with one cell. Persists ``stage0_gru_source_only.json``
+ the 2 trained GRU state-dicts.

No Qwen weights, no new judging — analysis over cached tensors only.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue841_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_841 import maps as MP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue841_gru_source_only_stage0")

DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_841" / "gru_source_only"
DEFAULT_PARENT_EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_841"
SPACES = ("raw", "rmsnorm")


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("--device cuda requested but no CUDA device; falling back to cpu")
        return "cpu"
    return requested


def _target(delta: np.ndarray, space: str, sigma: float) -> np.ndarray:
    return delta if space == "raw" else delta / sigma


def _atlas_cell(pred_fit, delta_test_fit, delta_test_raw, train_mean_fit, space, sigma) -> dict:
    """Per (transition, space) atlas metrics from fit-space predictions (parent-identical)."""
    pred_raw = pred_fit if space == "raw" else pred_fit * sigma
    return {
        "r2_id": MP.identity_relative_r2(pred_fit, delta_test_fit),
        "r2_meancentered": MP.mean_centered_r2(pred_fit, delta_test_fit, train_mean_fit),
        "delta_err_raw": MP.delta_error_percentiles(pred_raw, delta_test_raw),
    }


# ── source-only-GRU verify gate (exercises the dispatched functions) ────────────


def verify_source_only_gru(device: str) -> dict:
    """Self-check: ``forward_single`` == a GRUCell step (float64, tight tol) + ``apply`` finite.

    Exercises the EXACT functions the entrypoints dispatch — ``MP.DepthGRU.forward_single``
    and ``MP.GruSourceOnlyMap.apply`` dropped into ``MP.transport_iterated`` — NOT an
    unused sibling helper (hollow-gate rule, code-style.md § Verification gates test the
    live dispatched path). A length-1 ``nn.GRU`` unroll with zero initial hidden equals a
    ``GRUCell`` step iff the weights are shared, so we copy the GRU's ``l0`` weights into a
    ``GRUCell`` and assert equality; then assert ``.apply`` yields finite raw Δ̂ of shape
    (N, d) and that a memoryless ``transport_iterated`` roll is the identity when src==tgt.

    The parity comparison runs in float64: on cuda the two fp32 paths run DIFFERENT
    kernels with different precision defaults — ``forward_single``'s length-1 ``nn.GRU``
    dispatches to the cuDNN RNN kernel (``torch.backends.cudnn.allow_tf32`` defaults
    True on H100) while the ``GRUCell`` reference runs plain ATen fp32 matmuls
    (``torch.backends.cuda.matmul.allow_tf32`` defaults False) — so an fp32 comparison
    measures TF32-vs-fp32 kernel divergence, not implementation parity (att-8 crash:
    1.86e-4 under defaults; measured on pod-841: cudnn TF32 off → 1.9e-6, float64 →
    1.1e-16). TF32 applies only to fp32, so float64 removes the kernel gap entirely and
    the tol stays TIGHT (1e-8, ~8 orders above measured fp64 noise) — a real math bug
    (transposed weight, wrong gate order) is O(1) and still fails loudly. The dispatched
    production-dtype (fp32) forward is still exercised and cross-checked against the
    fp64 reference at a kernel-noise-tolerant bound.
    """
    torch.manual_seed(0)
    d, hid, emb_dim, n_trans = 16, 8, 4, 5  # tiny toy
    gru = MP.DepthGRU(d_state=d, gru_hidden=hid, emb_dim=emb_dim, n_transitions=n_trans).to(device)
    gru.eval()
    n, m = 3, 2
    h = torch.randn(n, d, device=device)
    idx = torch.full((n,), m, device=device, dtype=torch.long)
    with torch.no_grad():
        out_single = gru.forward_single(h, idx)  # (n, d) — the dispatched forward, fp32
        assert out_single.shape == (n, d), out_single.shape
        assert bool(torch.isfinite(out_single).all()), "fp32 forward_single non-finite"
        # float64 clone for the kernel-noise-free parity read (same forward_single code path).
        gru64 = copy.deepcopy(gru).double()
        out_single64 = gru64.forward_single(h.double(), idx)
        cell = torch.nn.GRUCell(d + emb_dim, hid).to(device).double()
        cell.weight_ih.copy_(gru64.gru.weight_ih_l0)
        cell.weight_hh.copy_(gru64.gru.weight_hh_l0)
        cell.bias_ih.copy_(gru64.gru.bias_ih_l0)
        cell.bias_hh.copy_(gru64.gru.bias_hh_l0)
        x = torch.cat([h.double(), gru64.emb(idx)], dim=-1)
        # one GRUCell step, zero init hidden
        h_cell = cell(x, torch.zeros(n, hid, device=device, dtype=torch.float64))
        out_manual = gru64.head(h_cell)
        max_abs = float((out_single64 - out_manual).abs().max().item())
        max_abs_fp32 = float((out_single.double() - out_manual).abs().max().item())
    assert max_abs < 1e-8, f"forward_single != GRUCell step (fp64 max abs diff {max_abs:.3g})"
    # Production-dtype cross-check: fp32 dispatched output within kernel precision of the
    # fp64 reference (TF32 divergence measured 1.86e-4; 1e-2 gives ~50x headroom while an
    # O(1) fp32-path cast/dtype bug still fails).
    assert max_abs_fp32 < 1e-2, (
        f"fp32 forward_single deviates {max_abs_fp32:.3g} from the fp64 reference — "
        "beyond TF32 kernel noise; fp32-path bug, not kernel precision"
    )

    smap = MP.GruSourceOnlyMap(gru=gru, transition=m, sigma_m=1.7)
    delta_hat = smap.apply(h)  # raw Δ̂ (no_grad inside apply)
    assert delta_hat.shape == (n, d), delta_hat.shape
    assert bool(torch.isfinite(delta_hat).all()), "apply produced non-finite Δ̂"

    maps = {t: MP.GruSourceOnlyMap(gru=gru, transition=t, sigma_m=1.0) for t in range(n_trans)}
    with torch.no_grad():
        h_id = MP.transport_iterated(maps, h, 0, 0)  # no transition ⇒ unchanged
        h_two = MP.transport_iterated(maps, h, 0, 2)  # two memoryless steps
    assert bool(torch.equal(h_id, h)), "transport_iterated src==tgt must be identity"
    assert h_two.shape == (n, d) and bool(torch.isfinite(h_two).all()), "two-step roll not finite"
    logger.info(
        "[verify] forward_single==GRUCell max_abs=%.3g; apply+transport finite: PASS", max_abs
    )
    return {
        "forward_single_vs_grucell_max_abs": max_abs,
        "forward_single_fp32_vs_fp64_max_abs": max_abs_fp32,
        "parity_dtype": "float64",
        "pass": True,
    }


# ── source-only-GRU atlas ───────────────────────────────────────────────────────


def gru_source_only_atlas(cx, split, transitions, sigma, device, max_epochs, batch_size):
    """Fit the source-only GRU per space {raw, rmsnorm}; read the atlas per transition.

    ``transitions`` is BOTH the fit-transition set and the read set (full run: all 27;
    smoke: [13]) — the source-only GRU is one model over these transitions (shared emb),
    matching how the parent prefix-GRU is one model over the whole trajectory. Returns
    (atlas dict, per-space convergence diagnostics, per-space cpu state-dicts).
    """
    n_trans = cx.shape[1] - 1
    atlas: dict = {s: {} for s in SPACES}
    diagnostics: dict = {}
    state_dicts: dict = {}
    for space in SPACES:
        fit_sigma = np.ones(n_trans, dtype=np.float64) if space == "raw" else sigma
        gru, diag = MP.fit_depth_gru_source_only(
            cx[split["fit"]],
            cx[split["val"]],
            fit_sigma,
            device=device,
            max_epochs=max_epochs,
            batch_size=batch_size,
            transitions=list(transitions),
        )
        diagnostics[space] = diag
        state_dicts[space] = {k: v.detach().cpu() for k, v in gru.state_dict().items()}
        test = torch.from_numpy(np.ascontiguousarray(cx[split["test"], :, :])).to(
            device=device, dtype=torch.float32
        )
        with torch.no_grad():
            for t in transitions:
                pred_fit = gru.forward_single(test[:, t, :], int(t)).cpu().numpy()  # (n_test, d)
                _h, delta = MP.deltas_at(cx, t)
                delta_test_fit = _target(delta[split["test"]], space, sigma[t])
                train_mean = _target(delta[split["fit"]], space, sigma[t]).mean(axis=0)
                atlas[space][f"transition_{t}"] = _atlas_cell(
                    pred_fit, delta_test_fit, delta[split["test"]], train_mean, space, sigma[t]
                )
        logger.info(
            "[gru_source_only] space=%s atlas done (%d transitions; best-val@epoch %d, cap_hit=%s)",
            space,
            len(transitions),
            diag["epochs_to_best_val"],
            diag["cap_hit"],
        )
    return atlas, diagnostics, state_dicts


# ── parent-artifact reload + win-counts ────────────────────────────────────────


def load_parent_atlas(parent_eval_dir: Path) -> dict:
    path = parent_eval_dir / "stage0_atlas.json"
    if not path.exists():
        raise FileNotFoundError(
            f"parent stage0_atlas.json not found at {path} — the reloaded ridge/mlp/prefix-GRU "
            "baselines are REQUIRED for the win-count comparison (never re-run them)."
        )
    with open(path) as f:
        parent = json.load(f)
    for cls in ("ridge", "mlp", "gru"):
        if cls not in parent.get("atlas", {}):
            raise KeyError(
                f"parent stage0_atlas.json missing atlas class {cls!r} (stale/wrong file)"
            )
    return parent


def assert_parent_ridge_dominates(parent_atlas: dict) -> dict:
    """Fail-loud: parent ridge beats the prefix-GRU on ALL 27 raw-space transitions (§8/§12).

    A mismatch means a stale/wrong ``stage0_atlas.json`` was reloaded — STOP rather than
    derive a win-count against a wrong baseline.
    """
    p_ridge = parent_atlas["atlas"]["ridge"]["raw"]
    p_gru = parent_atlas["atlas"]["gru"]["raw"]
    n = 0
    wins = 0
    for key, rcell in p_ridge.items():
        gcell = p_gru.get(key)
        if gcell is None:
            continue
        n += 1
        if rcell["r2_id"] > gcell["r2_id"]:
            wins += 1
    assert n == C.N_TRANSITIONS and wins == n, (
        f"parent ridge does NOT dominate the prefix-GRU 27/27 in raw space "
        f"(got {wins}/{n}); reloaded a stale/wrong stage0_atlas.json — STOP."
    )
    logger.info(
        "[parent-reload] ridge beats prefix-GRU %d/%d raw transitions (as expected)", wins, n
    )
    return {"raw_ridge_beats_prefix_gru": f"{wins}/{n}"}


def win_counts(parent_atlas: dict, my_atlas: dict, transitions) -> dict:
    """Per-space 27-transition win-count: #{gru_source_only.r2_id > {ridge, mlp, prefix_gru}}."""
    out: dict = {}
    for space in SPACES:
        p_ridge = parent_atlas["atlas"]["ridge"][space]
        p_mlp = parent_atlas["atlas"]["mlp"][space]
        p_gru = parent_atlas["atlas"]["gru"][space]
        rows = []
        for t in transitions:
            key = f"transition_{t}"
            so = my_atlas[space][key]["r2_id"]
            ridge_r2 = p_ridge[key]["r2_id"]
            mlp_r2 = p_mlp[key]["r2_id"]
            gru_r2 = p_gru.get(key, {}).get("r2_id")
            rows.append(
                {
                    "transition": t,
                    "gru_source_only_r2_id": so,
                    "ridge_r2_id": ridge_r2,
                    "mlp_r2_id": mlp_r2,
                    "prefix_gru_r2_id": gru_r2,
                    "beats_ridge": bool(
                        np.isfinite(so) and np.isfinite(ridge_r2) and so > ridge_r2
                    ),
                    "beats_mlp": bool(np.isfinite(so) and np.isfinite(mlp_r2) and so > mlp_r2),
                    "beats_prefix_gru": bool(
                        gru_r2 is not None
                        and np.isfinite(so)
                        and np.isfinite(gru_r2)
                        and so > gru_r2
                    ),
                }
            )
        out[space] = {
            "n_transitions": len(rows),
            "gru_source_only_beats_ridge": sum(r["beats_ridge"] for r in rows),
            "gru_source_only_beats_mlp": sum(r["beats_mlp"] for r in rows),
            "gru_source_only_beats_prefix_gru": sum(r["beats_prefix_gru"] for r in rows),
            "per_transition": rows,
        }
    return out


def parent_prefix_gru_convergence(parent_atlas: dict) -> dict:
    """Record whether the parent prefix-GRU artifact carries convergence diagnostics (§4.2).

    The parent ``gru_atlas`` did NOT persist per-fit convergence diagnostics, so parity
    with the source-only fit is UNVERIFIABLE from the committed artifact — the analyzer
    narrates the prefix-vs-source read descriptively with this caveat.
    """
    p_gru = parent_atlas["atlas"].get("gru", {})
    keys = set()
    for space_cells in p_gru.values():
        if isinstance(space_cells, dict):
            keys |= set(space_cells.keys())
    has_diag = any(k in ("convergence", "diagnostics", "epochs_to_best_val") for k in keys)
    return {
        "prefix_gru_convergence_available": bool(has_diag),
        "convergence_parity_verified": bool(has_diag),
        "note": (
            "parent stage0_atlas.json does NOT store prefix-GRU convergence diagnostics; "
            "prefix-vs-source convergence parity is UNVERIFIED from the committed artifact — "
            "narrate the prefix-vs-source read descriptively (plan §4.2)."
        )
        if not has_diag
        else "parent prefix-GRU convergence diagnostics present; parity recorded.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #841 gru-source-only Stage 0 Δ-atlas (source-only GRU)."
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-contexts", type=int, default=0, help="0 = all pass_b contexts")
    ap.add_argument("--gru-epochs", type=int, default=300)
    ap.add_argument("--gru-batch-size", type=int, default=512)
    ap.add_argument(
        "--verify-source-only-gru", action="store_true", help="run only the verify gate"
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--parent-eval-dir", type=Path, default=DEFAULT_PARENT_EVAL_DIR)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    logger.info("device=%s smoke=%s out_dir=%s", device, args.smoke, args.out_dir)

    # Verify gate ALWAYS runs first (exercises the dispatched forward/apply); --verify-only exits.
    verify = verify_source_only_gru(device)
    if args.verify_source_only_gru:
        logger.info("[verify-only] gate passed; exiting.")
        return 0

    parent_atlas = load_parent_atlas(args.parent_eval_dir)
    parent_ridge_check = assert_parent_ridge_dominates(parent_atlas)

    pass_b = C.load_pass_b()
    cx = pass_b["cx_last"]
    n_total = cx.shape[0]
    if args.smoke:
        cap = min(args.n_contexts or 200, n_total)
        cx = cx[:cap]
        n_total = cx.shape[0]
    elif args.n_contexts:
        cx = cx[: args.n_contexts]
        n_total = cx.shape[0]

    transitions = [13] if args.smoke else list(range(C.N_TRANSITIONS))
    split = C.make_split(
        n_total, n_fit=C.N_FIT, n_val=C.N_INNER_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED
    )
    logger.info(
        "[split] N=%d fit=%d val=%d test=%d transitions=%s",
        n_total,
        len(split["fit"]),
        len(split["val"]),
        len(split["test"]),
        transitions,
    )

    nc = MP.norm_curve(cx)
    sigma = np.asarray(nc["sigma_block_rms"], dtype=np.float64)

    atlas, diagnostics, state_dicts = gru_source_only_atlas(
        cx, split, transitions, sigma, device, args.gru_epochs, args.gru_batch_size
    )

    result: dict = {
        "split": {
            "n_total": n_total,
            "n_fit": len(split["fit"]),
            "n_val": len(split["val"]),
            "n_test": len(split["test"]),
            "seed": C.SPLIT_SEED,
        },
        "transitions": transitions,
        "target_spaces": list(SPACES),
        "verify_gate": verify,
        "parent_ridge_dominance_check": parent_ridge_check,
        "atlas": {"gru_source_only": atlas},
        "win_counts": win_counts(parent_atlas, atlas, transitions),
        "convergence_diagnostics": diagnostics,
        "parent_prefix_gru_convergence": parent_prefix_gru_convergence(parent_atlas),
        "metadata": C.reproducibility_metadata(
            {"phase": "stage0_gru_source_only", "smoke": args.smoke}
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(args.out_dir / "stage0_gru_source_only.json", result)
    for space, sd in state_dicts.items():
        torch.save(
            {
                "state_dict": sd,
                "space": space,
                "config": {
                    "d_state": C.EXPECTED_HIDDEN,
                    "gru_hidden": 1024,
                    "emb_dim": 32,
                    "n_transitions": C.N_TRANSITIONS,
                },
                "sigma_block_rms": sigma.tolist() if space == "rmsnorm" else None,
                "seed": C.MLP_INIT_SEED,
                "metadata": C.reproducibility_metadata({"phase": f"gru_source_only_{space}"}),
            },
            args.out_dir / f"gru_source_only_{space}.pt",
        )
    logger.info(
        "[done] wrote %s + gru_source_only_{raw,rmsnorm}.pt; win-counts(raw)=%s",
        args.out_dir / "stage0_gru_source_only.json",
        result["win_counts"]["raw"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
