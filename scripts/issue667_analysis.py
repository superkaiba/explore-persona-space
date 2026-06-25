#!/usr/bin/env python3
"""Issue #667 per-assumption CPU analysis runner — A3.6-A3.10 + B3 gate.

Reads the per-cell activation store (``eval_results/issue_667/analysis_tensors/
<behavior>/<source>_seed<S>/<target>_L<l>.npz``), #537's measured leakage
matrix ``G`` (``G_meta.json`` + ``G_tensor.npz``), and #658's ``sigma_c.pt`` /
``r_b.pt``; writes one JSON per assumption under ``eval_results/issue_667/``.

The B3 reduction unit test gates A3.9/A3.10: if it fails, the runner HALTs
before producing any A3.9/A3.10 number (plan §7 — a mis-implemented whitened
inverse otherwise manufactures a spurious "whitening wins").

Off-pod CPU (plan §9): all linear algebra over the HF-uploaded store, no GPU,
no model load. Reproducibility metadata (git commit, env, timestamp, pins) is
embedded in every output JSON (CLAUDE.md Reproducibility Requirements).

Usage::

    uv run python scripts/issue667_analysis.py \\
        --tensors-dir eval_results/issue_667/analysis_tensors \\
        --out-dir eval_results/issue_667 \\
        --behaviors em sycophancy fact --primary-layer 14
"""

# ruff: noqa: RUF002, RUF003  # math/scientific notation in docstrings + messages

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from explore_persona_space.analysis.issue667 import (  # noqa: E402
    EXPECTED_G_META_GIT_COMMIT,
    EXPECTED_STORE_PROBE_POOL_HASH,
    G_META_LOCAL,
    G_TENSOR_PATH,
    HF_DATA_REPO,
    IN_SCOPE_BEHAVIORS,
    PRIMARY_LAYER,
    R_B_PATH,
    RB_COLUMN_FOR_BEHAVIOR,
    RB_RECIPE,
    SIGMA_C_LAMBDA_FRACTION,
    SIGMA_C_PATH,
    STORE_MANIFEST_PATH,
)
from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    a37_source_write,
    clustered_bootstrap_spearman,
    default_lambda,
    family_of,
    partial_spearman,
    readout_projection,
    realized_gate,
    shuffled_null_ci,
    spearman_rho,
    stacked_delta_svd,
    whitened_gate_metric,
    whitened_gate_reduction_unit_test,
)

logger = logging.getLogger("issue667_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility metadata
# ─────────────────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, env={**os.environ}
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _repro_meta(extra: dict | None = None) -> dict:
    meta = {
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "script": "issue667_analysis",
        "g_meta_git_commit_pin": EXPECTED_G_META_GIT_COMMIT,
        "store_probe_pool_hash_pin": EXPECTED_STORE_PROBE_POOL_HASH,
    }
    if extra:
        meta.update(extra)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Reused-artifact loaders (sha-pinned)
# ─────────────────────────────────────────────────────────────────────────────


def _hf(path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_DATA_REPO, path, repo_type="dataset")


def load_g_meta() -> dict:
    """#537 G_meta.json (per-cell g/base_rate/...); assert the git_commit pin.

    G_meta is committed in git (``eval_results/issue_537/G_tensor/G_meta.json``,
    on ``main`` and inherited by the ``issue-667`` branch the pod checks out) —
    it is NOT on the HF data repo. Fail loud if missing (a sparse worktree that
    excludes ``eval_results/`` must ``git sparse-checkout add`` it for a local
    smoke; the pod's full checkout always has it).
    """
    p = PROJECT_ROOT / G_META_LOCAL
    if not p.exists():
        raise FileNotFoundError(
            f"G_meta.json not found at {p}. It is committed in git "
            f"({G_META_LOCAL}), NOT on HF. On a sparse worktree run "
            "`git sparse-checkout add eval_results/issue_537/G_tensor`; the pod's "
            "full checkout has it by default."
        )
    m = json.loads(p.read_text())
    gc = m.get("git_commit")
    assert gc == EXPECTED_G_META_GIT_COMMIT, (
        f"G_meta git_commit pin drift: {gc} != {EXPECTED_G_META_GIT_COMMIT} (#537 ground truth)"
    )
    return m


def load_g_tensor() -> dict:
    """#537 G_tensor.npz (G[5,16,30,1] + masks + train/eval cids)."""
    p = Path(_hf(G_TENSOR_PATH))
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def assert_store_pin() -> None:
    """Assert #658 store_manifest probe_pool_hash pin (the load-bearing pin)."""
    p = Path(_hf(STORE_MANIFEST_PATH))
    m = json.loads(p.read_text())
    pph = m.get("probe_pool_hash")
    assert pph == EXPECTED_STORE_PROBE_POOL_HASH, (
        f"#658 store probe_pool_hash pin drift: {pph} != {EXPECTED_STORE_PROBE_POOL_HASH}"
    )


def load_sigma_c(layer: int):
    """#658 sigma_c.pt -> (3584, 3584) at ``layer`` (the model-level second moment)."""
    import torch

    d = torch.load(Path(_hf(SIGMA_C_PATH)), weights_only=False, map_location="cpu")
    sig = d["sigma_c"]  # (28, 3584, 3584)
    cap = list(d["capture_layers"])
    assert layer in cap, (layer, cap)
    return sig[cap.index(layer)]


def load_r_b(behavior: str, layer: int) -> np.ndarray | None:
    """#658 r_b.pt[<col>][diffmeans][layer] for in-scope behaviors (None for fact)."""
    import torch

    col = RB_COLUMN_FOR_BEHAVIOR.get(behavior)
    if col is None:
        return None  # fact (absent from #658) -> re-extracted in the store
    d = torch.load(Path(_hf(R_B_PATH)), weights_only=False, map_location="cpu")
    return d["r_b"][col][RB_RECIPE][layer].float().numpy().astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell store loading
# ─────────────────────────────────────────────────────────────────────────────


def load_cells(tensors_dir: Path, behavior: str, layer: int) -> dict[tuple[str, str], dict]:
    """{(source_cid, target_cid): npz_dict} for one (behavior, layer) from the store."""
    out: dict[tuple[str, str], dict] = {}
    beh_dir = tensors_dir / behavior
    if not beh_dir.exists():
        return out
    for cell_dir in sorted(beh_dir.glob("*_seed*")):
        source_cid = cell_dir.name.rsplit("_seed", 1)[0]
        for npz in sorted(cell_dir.glob(f"*_L{layer}.npz")):
            target_cid = npz.name.rsplit(f"_L{layer}.npz", 1)[0]
            data = dict(np.load(npz, allow_pickle=True))
            out[(source_cid, target_cid)] = data
    return out


def g_cell(g_meta: dict, behavior: str, source: str, target: str) -> dict | None:
    """#537 per-cell ground truth: {g, base_rate, noise_var, saturated}."""
    return g_meta["per_cell"].get(f"{behavior}/{source}__{target}")


# ─────────────────────────────────────────────────────────────────────────────
# A3.6 — base read-out predicts the post-FT behavior CHANGE (partial corr, C10)
# ─────────────────────────────────────────────────────────────────────────────


def run_a36(cells_by_beh: dict, g_meta: dict, r_b_by_beh: dict, layer: int) -> dict:
    """A3.6: partial-Spearman(r_B'^T Δv(C'), E+ - E0 | E0) per behavior."""
    results = {}
    for behavior, cells in cells_by_beh.items():
        r_b = r_b_by_beh.get(behavior)
        # fact: read the re-extracted r_b from the store (primary-layer cell payloads).
        if r_b is None:
            r_b = _fact_rb_from_store(cells)
        if r_b is None:
            results[behavior] = {"status": "no_r_b", "note": "r_B unavailable for this behavior"}
            continue
        xs, ys, zs, fams, n_dyn = [], [], [], [], 0
        # group by source: for each source, vary target C'.
        for (source, target), data in cells.items():
            if source == target:
                continue  # off-diagonal targets only (the CHANGE read)
            gc = g_cell(g_meta, behavior, source, target)
            if gc is None:
                continue
            delta_v = data["v_plus"].astype(np.float64) - data["v0"].astype(np.float64)
            xs.append(readout_projection(r_b, delta_v))
            ys.append(float(gc["g"]))  # E+ - E0 == g
            zs.append(float(gc["base_rate"]))  # E0
            fams.append(family_of(target))
            if abs(float(gc["g"])) > 0.01:
                n_dyn += 1
        if len(xs) < 3:
            results[behavior] = {"status": "insufficient_cells", "n": len(xs)}
            continue
        x = np.array(xs)
        y = np.array(ys)
        z = np.array(zs)
        partial = partial_spearman(x, y, z)
        null = shuffled_null_ci(x, y - z)  # rough matched null on the residualized y
        boot = clustered_bootstrap_spearman(x, y, fams)
        results[behavior] = {
            "status": "ok",
            "partial_spearman_change_given_base": partial,
            "raw_spearman_proj_vs_g": spearman_rho(x, y),
            "shuffled_null_hi": null["null_hi"],
            "clustered_bootstrap": boot,
            "n_cells": len(xs),
            "n_dynamic_range_cells": n_dyn,
            "dynamic_range_fraction": n_dyn / len(xs),
            "r_b_source": "reextracted_fact"
            if behavior == "fact"
            else RB_COLUMN_FOR_BEHAVIOR[behavior],
        }
    return {"assumption": "A3.6", "layer": layer, "by_behavior": results, "metadata": _repro_meta()}


def _fact_rb_from_store(cells: dict) -> np.ndarray | None:
    for _key, data in cells.items():
        if "r_b_fact" in data:
            return data["r_b_fact"].astype(np.float64)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# A3.7 — source write ŵ∥δ (cos(w_hat, delta_pos) vs shuffled-δ null)
# ─────────────────────────────────────────────────────────────────────────────


def run_a37(cells_by_beh: dict, layer: int) -> dict:
    """A3.7: per source, cos(w_hat, delta_pos/delta_contra) + frac_ctx + shuffled-δ null."""
    # w_hat per (behavior, source) = v+(C) - v0(C) at the source diagonal.
    w_hats: dict[tuple[str, str], np.ndarray] = {}
    t_pos: dict[tuple[str, str], np.ndarray] = {}
    t_neg: dict[tuple[str, str], np.ndarray] = {}
    v0_src: dict[tuple[str, str], np.ndarray] = {}
    for behavior, cells in cells_by_beh.items():
        for (source, target), data in cells.items():
            if source != target:
                continue
            w_hats[(behavior, source)] = data["v_plus"].astype(np.float64) - data["v0"].astype(
                np.float64
            )
            v0_src[(behavior, source)] = data["v0"].astype(np.float64)
            if "t_pos" in data:
                t_pos[(behavior, source)] = data["t_pos"].astype(np.float64)
            if "t_neg" in data:
                t_neg[(behavior, source)] = data["t_neg"].astype(np.float64)
    results = {}
    for behavior in cells_by_beh:
        rows = []
        # shuffled-δ null: cos(w_hat, delta_pos of a DIFFERENT behavior's source).
        other_deltas = [t_pos[k] - v0_src[k] for k in t_pos if k[0] != behavior and k in v0_src]
        for (b, source), w in w_hats.items():
            if b != behavior:
                continue
            dp = t_pos.get((b, source))
            tn = t_neg.get((b, source))
            if dp is None:
                continue
            delta_pos = dp - v0_src[(b, source)]
            delta_contra = (dp - tn) if tn is not None else delta_pos
            v0_cneg = tn if tn is not None else v0_src[(b, source)]
            other = (
                other_deltas[len(rows) % len(other_deltas)] if other_deltas else np.zeros_like(w)
            )
            rows.append(
                {
                    "source": source,
                    **a37_source_write(
                        w, delta_pos, delta_contra, other, v0_src[(b, source)], v0_cneg
                    ),
                }
            )
        if not rows:
            results[behavior] = {"status": "no_source_cells"}
            continue
        cos_pos = [r["cos_pos"] for r in rows]
        cos_null = [r["cos_null"] for r in rows]
        results[behavior] = {
            "status": "ok",
            "per_source": rows,
            "mean_cos_pos": float(np.mean(cos_pos)),
            "mean_cos_contra": float(np.mean([r["cos_contra"] for r in rows])),
            "mean_cos_null": float(np.mean(cos_null)),
            "mean_frac_ctx": float(np.nanmean([r["frac_ctx"] for r in rows])),
            "beats_null": bool(np.mean(cos_pos) > np.mean(cos_null)),
            "n_sources": len(rows),
        }
    return {"assumption": "A3.7", "layer": layer, "by_behavior": results, "metadata": _repro_meta()}


# ─────────────────────────────────────────────────────────────────────────────
# A3.8 — off-source change = scalar-gated source write (rank-one + SVD)
# ─────────────────────────────────────────────────────────────────────────────


def run_a38(cells_by_beh: dict, layer: int) -> dict:
    """A3.8: per source, rank-one residual + stacked-ΔV SVD (per behavior, #637)."""
    results = {}
    for behavior, cells in cells_by_beh.items():
        # group targets by source
        by_source: dict[str, list[tuple[str, dict]]] = {}
        diag: dict[str, np.ndarray] = {}
        for (source, target), data in cells.items():
            by_source.setdefault(source, []).append((target, data))
            if source == target:
                diag[source] = data["v_plus"].astype(np.float64) - data["v0"].astype(np.float64)
        src_rows = []
        for source, targs in by_source.items():
            if source not in diag:
                continue
            w_hat = diag[source]
            if float(w_hat @ w_hat) <= 0:
                continue
            residuals, gates, deltas = [], [], []
            for target, data in targs:
                if target == source:
                    continue
                # ĝ^real + rank-one residual use the source DIAGONAL write w_hat.
                g_real, resid = _gate_for(cells, source, target, data)
                residuals.append(resid)
                gates.append(g_real)
                deltas.append(data["v_plus"].astype(np.float64) - data["v0"].astype(np.float64))
            if len(deltas) < 2:
                continue
            svd = stacked_delta_svd(np.stack(deltas), w_hat)
            src_rows.append(
                {
                    "source": source,
                    "mean_rank_one_residual": float(np.mean(residuals)),
                    "median_realized_gate": float(np.median(gates)),
                    **svd,
                }
            )
        results[behavior] = {
            "status": "ok" if src_rows else "no_sources",
            "per_source": src_rows,
            "note": "per-behavior, never aggregated over the #637 content-behavior failure",
        }
    return {"assumption": "A3.8", "layer": layer, "by_behavior": results, "metadata": _repro_meta()}


def _gate_for(cells: dict, source: str, target: str, data: dict) -> tuple[float, float]:
    """ĝ^real + rank-one residual for (source -> target) using the diagonal source write."""
    src = cells[(source, source)]
    return realized_gate(src["v0"], src["v_plus"], data["v0"], data["v_plus"])


# ─────────────────────────────────────────────────────────────────────────────
# A3.9 / A3.10 — base key-query gate predicts the realized gate (B3-gated)
# ─────────────────────────────────────────────────────────────────────────────


def run_a39_a310(cells_by_beh: dict, sigma_c, layer: int) -> tuple[dict, dict]:
    """A3.9 key×metric ablation + A3.10 base-gate validity. B3-gated upstream."""
    import torch

    lam = default_lambda(sigma_c, SIGMA_C_LAMBDA_FRACTION)
    a39 = {}
    a310 = {}
    for behavior, cells in cells_by_beh.items():
        # realized gate ĝ^real(C') per (source, target), source diagonal write.
        g_real_rows = []  # (source, target, g_real, c_C, c_Cp)
        for (source, target), data in cells.items():
            if target == source or (source, source) not in cells:
                continue
            try:
                g_real, _ = _gate_for(cells, source, target, data)
            except ValueError:
                continue
            g_real_rows.append(
                (
                    source,
                    target,
                    g_real,
                    cells[(source, source)]["c_C"].astype(np.float64),
                    data["c_Cp"].astype(np.float64),
                )
            )
        if len(g_real_rows) < 3:
            a39[behavior] = {"status": "insufficient_cells", "n": len(g_real_rows)}
            a310[behavior] = {"status": "insufficient_cells", "n": len(g_real_rows)}
            continue
        g_real = np.array([r[2] for r in g_real_rows])
        fams = [family_of(r[1]) for r in g_real_rows]
        # key×metric ablation: c_C key under {I, diag, whitened}.
        metric_corr = {}
        for metric in ("I", "diag", "whitened"):
            gate_pred = []
            for _s, _t, _g, c_c, c_cp in g_real_rows:
                gate_pred.append(
                    whitened_gate_metric(
                        torch.from_numpy(c_c), torch.from_numpy(c_cp), metric, sigma_c, lam
                    )
                )
            gp = np.array(gate_pred)
            metric_corr[metric] = {
                "spearman": spearman_rho(gp, g_real),
                "pearson": _pearson(gp, g_real),
                "clustered_bootstrap": clustered_bootstrap_spearman(gp, g_real, fams),
            }
        # predict-mean baseline + shuffled null on the boxed primary (whitened).
        gp_whit = np.array(
            [
                whitened_gate_metric(
                    torch.from_numpy(r[3]), torch.from_numpy(r[4]), "whitened", sigma_c, lam
                )
                for r in g_real_rows
            ]
        )
        null = shuffled_null_ci(gp_whit, g_real)
        a39[behavior] = {
            "status": "ok",
            "metric_ablation": metric_corr,
            "boxed_primary": "c_C_key__whitened_metric",
            "boxed_primary_spearman": metric_corr["whitened"]["spearman"],
            "cosine_baseline_spearman": metric_corr["I"]["spearman"],
            "shuffled_null_hi": null["null_hi"],
            "beats_cosine": bool(
                metric_corr["whitened"]["spearman"] > metric_corr["I"]["spearman"]
            ),
            "n_cells": len(g_real_rows),
            "lambda": lam,
        }
        # A3.10: base gate g0 (whitened) vs realized — same numbers, framed at fixed M0.
        a310[behavior] = {
            "status": "ok",
            "g0_vs_realized_spearman": metric_corr["whitened"]["spearman"],
            "g0_vs_realized_clustered_bootstrap": metric_corr["whitened"]["clustered_bootstrap"],
            "note": "at fixed M0 (no Sigma_c+; oracle g+ metric drift unattributed, R3-3)",
            "n_cells": len(g_real_rows),
        }
    return (
        {"assumption": "A3.9", "layer": layer, "by_behavior": a39, "metadata": _repro_meta()},
        {"assumption": "A3.10", "layer": layer, "by_behavior": a310, "metadata": _repro_meta()},
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a @ a) * (b @ b)))
    return float((a @ b) / d) if d > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #667 per-assumption CPU analysis (A3.6-A3.10)."
    )
    parser.add_argument("--tensors-dir", default="eval_results/issue_667/analysis_tensors")
    parser.add_argument("--out-dir", default="eval_results/issue_667")
    parser.add_argument("--behaviors", nargs="+", default=list(IN_SCOPE_BEHAVIORS))
    parser.add_argument("--primary-layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument(
        "--skip-store-pin",
        action="store_true",
        help="skip the #658 store + G_meta pin asserts (smoke on a synthetic store)",
    )
    args = parser.parse_args()

    tensors_dir = Path(args.tensors_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer = args.primary_layer

    # ── B3 GATE: the whitened-gate reduction unit test MUST pass first ────────
    logger.info("B3 GATE: whitened_gate_reduction_unit_test()")
    whitened_gate_reduction_unit_test()
    logger.info("B3 unit test PASS")

    cells_by_beh = {b: load_cells(tensors_dir, b, layer) for b in args.behaviors}
    for b, cells in cells_by_beh.items():
        logger.info("behavior=%s: %d cells loaded at layer %d", b, len(cells), layer)

    # ── Load reused artifacts (sha-pinned) ───────────────────────────────────
    if not args.skip_store_pin:
        from dotenv import load_dotenv

        load_dotenv()
        assert_store_pin()
        g_meta = load_g_meta()
        sigma_c = load_sigma_c(layer)
        r_b_by_beh = {b: load_r_b(b, layer) for b in args.behaviors}
    else:
        # Synthetic-store smoke: build minimal stand-ins. Infer the hidden dim
        # from the first loaded cell's c_C so the identity Sigma_c matches.
        g_meta = _synthetic_g_meta(tensors_dir, args.behaviors, layer)
        import torch

        hdim = _infer_hidden_dim(cells_by_beh)
        sigma_c = torch.eye(hdim, dtype=torch.float64)
        # Synthetic r_b per behavior so A3.6 exercises (fact still reads the
        # store's r_b_fact via run_a36's _fact_rb_from_store fallback).
        _rng = np.random.default_rng(1)
        r_b_by_beh = {
            b: (None if b == "fact" else _rng.normal(size=hdim).astype(np.float64))
            for b in args.behaviors
        }

    # ── A3.6-A3.10 ───────────────────────────────────────────────────────────
    a36 = run_a36(cells_by_beh, g_meta, r_b_by_beh, layer)
    a37 = run_a37(cells_by_beh, layer)
    a38 = run_a38(cells_by_beh, layer)
    a39, a310 = run_a39_a310(cells_by_beh, sigma_c, layer)

    outputs = {
        "A3_6_readout_stability.json": a36,
        "A3_7_source_write.json": a37,
        "A3_8_rank_one.json": a38,
        "A3_9_key_query_gate.json": a39,
        "A3_10_base_gate_validity.json": a310,
    }
    for fname, payload in outputs.items():
        (out_dir / fname).write_text(json.dumps(payload, indent=2, default=_json_default))
        logger.info("wrote %s", out_dir / fname)
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _infer_hidden_dim(cells_by_beh: dict) -> int:
    """Hidden dim from the first loaded cell's c_C (synthetic-store smoke)."""
    for cells in cells_by_beh.values():
        for data in cells.values():
            return int(data["c_C"].shape[0])
    raise RuntimeError("no cells loaded — cannot infer hidden dim for the synthetic smoke")


def _synthetic_g_meta(tensors_dir: Path, behaviors: list[str], layer: int) -> dict:
    """Minimal G_meta stand-in for the synthetic-store smoke (--skip-store-pin)."""
    per_cell = {}
    rng = np.random.default_rng(0)
    for b in behaviors:
        for cell_dir in (tensors_dir / b).glob("*_seed*"):
            source = cell_dir.name.rsplit("_seed", 1)[0]
            for npz in cell_dir.glob(f"*_L{layer}.npz"):
                target = npz.name.rsplit(f"_L{layer}.npz", 1)[0]
                per_cell[f"{b}/{source}__{target}"] = {
                    "g": float(rng.normal()),
                    "base_rate": float(rng.uniform(0, 0.3)),
                    "noise_var": 0.01,
                    "saturated": False,
                }
    return {"git_commit": EXPECTED_G_META_GIT_COMMIT, "per_cell": per_cell}


if __name__ == "__main__":
    sys.exit(main())
