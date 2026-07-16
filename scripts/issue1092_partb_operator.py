#!/usr/bin/env python3
"""Issue #1092 Part-B operator-level arm comparison (ideas-doc Q6) on battery-excluded maps.

Per (cell, layer in {14,18,19}, target basis in {ambient, pca48}):
  1. Fit W_prefix and W_ctx — standardized-X ridge on the battery-excluded rows
     (the corrected fit-arm-A row set, ``issue1092_fit_grid._fit_arm_indices``),
     both arms at ONE MATCHED lambda: argmin over the shared 6-value
     RIDGE_LAMBDAS grid of the SUM of the two arms' exact PRESS LOO MSEs
     (per-arm optima recorded alongside). The standardization recipe is
     byte-identical to ``issue923_fit_decomposition.press_fit_predict``
     (train mu/sd ddof=0 + 1e-9 floor, degenerate-dim drop, target centering).
  2. Report each operator in RAW-input coordinates (W_raw = W_std / sd embedded
     at kept dims, shape (P, d)) so the two arms' input subspaces live in ONE
     shared residual-stream basis (per-arm standardization would warp them
     apart otherwise).
  3. Principal angles between top-k RIGHT (input) and LEFT (output) singular
     subspaces of W_prefix vs W_ctx, at k=48 and k at 90% spectral energy
     (energy = squared singular values), each vs a spectrum-matched random-map
     null band (same singular values, Haar-random subspaces), plus the
     orthogonal-Procrustes residual min_R ||W_ctx - R W_pfx||_F / ||W_ctx||_F
     (R orthogonal on the OUTPUT side; the input spaces differ by arm so only
     the shared output space admits a rotation alignment).
  4. Null draws are BATCHED (chunked batched QR + bmm + batched svdvals —
     never a serial 200-iteration loop of full SVDs). The Procrustes null
     truncates the matched spectra at --null-energy nuclear coverage with the
     rigorous |nuc(full) - nuc(head)| bound recorded per unit
     (bound = s_max(ctx)*nuc_tail(pfx) + s_max(pfx)*nuc_tail(ctx)).

Caveats carried in every unit payload: subspace claims are restricted to the
data-spanned row space (ambient n > d; kept-dim rank recorded) and ridge
shrinkage biases the spectra low (deferred_refit_spec.json partB method).

The registered ``topic_matched_pairing_delta`` read is DROPPED with a recorded
reason (spec item 5): the committed engine has no recoverable definition — a
superseded plan revision.

Outputs: per-unit JSONs under <out-dir>/partb/ (checkpoint-per-unit with a
fingerprint resume predicate) + <out-dir>/partb/partb_summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Thread caps + .env must bind BEFORE torch imports (#847).
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue923_fit_decomposition import PressRidge, run_selftest  # noqa: E402
from issue1092_fit_grid import (  # noqa: E402
    CELL_MODEL_TYPE,
    HF_DATA_REPO,
    RIDGE_LAMBDAS,
    _basis_targets_with_info,
    _fit_arm_indices,
    _jsonl,
    _load_summary,
    _parse_csv,
    _parse_layers,
)

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

FIT_GRID_SCRIPT = PROJECT_ROOT / "scripts" / "issue1092_fit_grid.py"
PARTB_KINDS = ("prefix_end", "context_end", "t1", "t2", "t3")

TOPIC_MATCHED_PAIRING_DELTA_DROP = {
    "read": "topic_matched_pairing_delta",
    "status": "dropped",
    "reason": (
        "the committed engine has no recoverable definition for this read — it belongs "
        "to a superseded plan revision; dropped per deferred_refit_spec.json item 5 / "
        "the offvm-battery-refit-and-operator-comparison followup scope (analyzer "
        "carries this as a scope note)"
    ),
}


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _fingerprint(paths: list[Path], config: dict) -> str:
    """Resume key: config + input file identity + THIS script + the engine module bytes."""
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for path in sorted(paths):
        st = path.stat()
        h.update(path.name.encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    h.update(Path(__file__).read_bytes())
    h.update(FIT_GRID_SCRIPT.read_bytes())  # _fit_arm_indices provenance rides the key
    return h.hexdigest()[:24]


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=True))
    os.replace(tmp, path)


def _fit_press(X: np.ndarray) -> dict:
    """Standardize X exactly like press_fit_predict(standardize=True) and build PressRidge."""
    Xt = torch.from_numpy(np.ascontiguousarray(X)).double()
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9
    keep = sd > (sd.max() * 1e-6 + 1e-12)  # degenerate-dim drop (#923 §8 convention)
    Xn = ((Xt - mu) / sd)[:, keep]
    return {"eng": PressRidge(Xn), "sd": sd, "keep": keep, "d_full": int(Xt.shape[1])}


def _press_mse(fit: dict, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact PRESS LOO mse per lambda + the G = UᵀYc factor for W reconstruction."""
    ymu = Y.mean(0, keepdim=True)
    Yc = Y - ymu
    mse, G = fit["eng"].press_mse(Yc.unsqueeze(0))
    return mse[0], G[0], ymu


def _operator_raw(fit: dict, G: torch.Tensor, lam: float) -> torch.Tensor:
    """(P, d_full) raw-input-space ridge operator at lambda.

    W_std = V diag(S/(S²+λ)) UᵀYc maps STANDARDIZED kept-dim inputs to centered
    targets; folding the per-dim 1/sd back in and embedding kept dims yields the
    operator on raw inputs, so both arms' input subspaces share one basis.
    """
    eng = fit["eng"]
    coef = eng.S / (eng.S**2 + lam)  # (k,)
    W_std = eng.Vh.T @ (coef.unsqueeze(1) * G)  # (d_kept, P)
    P = W_std.shape[1]
    W_raw = torch.zeros(P, fit["d_full"], dtype=W_std.dtype)
    kept_idx = fit["keep"].nonzero(as_tuple=True)[0]
    W_raw[:, kept_idx] = (W_std / fit["sd"][fit["keep"]].unsqueeze(1)).T
    return W_raw


def _k90(s: torch.Tensor) -> int:
    """Smallest k whose top-k squared singular values reach 90% spectral energy."""
    e = s.double() ** 2
    c = torch.cumsum(e, 0) / e.sum().clamp(min=1e-300)
    return int(torch.searchsorted(c, torch.tensor(0.90, dtype=c.dtype)).item()) + 1


def _angles_between(A: torch.Tensor, B: torch.Tensor) -> list[float]:
    """Principal angles (rad) between the column spaces of orthonormal A (d,k1), B (d,k2)."""
    sv = torch.linalg.svdvals(A.T @ B)
    return [float(v) for v in torch.arccos(sv.clamp(-1.0, 1.0))]


def _rand_orthonormal(b: int, d: int, k: int, gen: torch.Generator) -> torch.Tensor:
    g = torch.randn(b, d, k, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(g, mode="reduced")
    return q


def _angle_null_band(
    d: int, k1: int, k2: int, n_draws: int, chunk: int, gen: torch.Generator, max_rank: int
) -> dict:
    """Mean principal angle between two independent Haar-random subspaces of R^d.

    Batched: chunked batched QR + bmm + batched svdvals — fix item 3 of
    .claude/rules/vectorize-many-cell-fits.md; no per-draw full SVD loop.
    ``max_rank`` bounds the sampled-null wall deterministically: a k90 near the
    full operator rank would make the per-draw QR O(d * k90^2) — and angles
    between near-full-rank random subspaces concentrate near their
    deterministic limit anyway, so the sampled band is skipped with a note.
    """
    if min(k1, k2) < 1 or max(k1, k2) >= d:
        return {
            "n_draws": 0,
            "degenerate": f"k=({k1},{k2}) vs d={d}: subspace null undefined/trivial",
        }
    if max(k1, k2) > max_rank:
        return {
            "n_draws": 0,
            "skipped": (
                f"k=({k1},{k2}) > --null-max-rank {max_rank}: sampled null skipped "
                "(near-full-rank random-subspace angles concentrate; the k48 band is "
                "the registered comparison)"
            ),
        }
    means: list[float] = []
    done = 0
    while done < n_draws:
        c = min(chunk, n_draws - done)
        q1 = _rand_orthonormal(c, d, k1, gen)
        q2 = _rand_orthonormal(c, d, k2, gen)
        sv = torch.linalg.svdvals(torch.bmm(q1.transpose(1, 2), q2)).clamp(-1.0, 1.0)
        means.extend(float(v) for v in torch.arccos(sv).mean(dim=1))
        done += c
    arr = np.asarray(means, dtype=np.float64)
    return {
        "n_draws": int(arr.size),
        "mean_angle_p05": float(np.percentile(arr, 5)),
        "mean_angle_p95": float(np.percentile(arr, 95)),
        "draws_mean_angle_rad": [float(v) for v in arr],
    }


def _truncate_spectrum(s: torch.Tensor, energy: float, max_rank: int) -> int:
    """Head rank reaching `energy` NUCLEAR coverage (capped at max_rank, floor 1)."""
    nuc = torch.cumsum(s.double(), 0) / s.double().sum().clamp(min=1e-300)
    r = int(torch.searchsorted(nuc, torch.tensor(energy, dtype=nuc.dtype)).item()) + 1
    return max(1, min(r, int(s.shape[0]), max_rank))


def _procrustes_null_band(
    s_ctx: torch.Tensor,
    s_pfx: torch.Tensor,
    d: int,
    *,
    energy: float,
    max_rank: int,
    n_draws: int,
    chunk: int,
    gen: torch.Generator,
) -> dict:
    """Spectrum-matched random-map Procrustes-residual null (batched, truncated spectra).

    For W' = P Σ Qᵀ with Haar-random subspaces and the OBSERVED Σ per arm,
    nuc(W'_ctx W'_pfxᵀ) = nuc(Σ_ctx (V_ctxᵀ V_pfx) Σ_pfx) (left isometries drop
    out), so each draw needs only two random orthonormal (d, r_head) bases, one
    bmm, and one (r_head x r_head) batched svdvals. Truncating each spectrum at
    `energy` nuclear coverage under-estimates nuc by at most
    s_max(ctx)*nuc_tail(pfx) + s_max(pfx)*nuc_tail(ctx) (recorded), so the
    null residual is over-estimated by a bounded, recorded amount.
    """
    r_c = _truncate_spectrum(s_ctx, energy, max_rank)
    r_p = _truncate_spectrum(s_pfx, energy, max_rank)
    fro2_c = float((s_ctx.double() ** 2).sum())
    fro2_p = float((s_pfx.double() ** 2).sum())
    tail_nuc_c = float(s_ctx.double()[r_c:].sum())
    tail_nuc_p = float(s_pfx.double()[r_p:].sum())
    nuc_bound = float(s_ctx.double().max()) * tail_nuc_p + float(s_pfx.double().max()) * tail_nuc_c
    sc = s_ctx.double()[:r_c]
    sp = s_pfx.double()[:r_p]
    residuals: list[float] = []
    done = 0
    while done < n_draws:
        c = min(chunk, n_draws - done)
        v_c = _rand_orthonormal(c, d, r_c, gen)
        v_p = _rand_orthonormal(c, d, r_p, gen)
        m = sc.view(1, -1, 1) * torch.bmm(v_c.transpose(1, 2), v_p) * sp.view(1, 1, -1)
        nuc = torch.linalg.svdvals(m).sum(dim=1)
        res = torch.sqrt((fro2_c + fro2_p - 2.0 * nuc).clamp(min=0.0)) / np.sqrt(fro2_c)
        residuals.extend(float(v) for v in res)
        done += c
    arr = np.asarray(residuals, dtype=np.float64)
    return {
        "n_draws": int(arr.size),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "draws": [float(v) for v in arr],
        "truncation": {
            "nuclear_energy_target": energy,
            "head_rank_ctx": r_c,
            "head_rank_pfx": r_p,
            "nuc_underestimate_bound": nuc_bound,
            "direction_note": (
                "truncation UNDER-estimates the null nuc term, so null residual draws are "
                "OVER-estimated by a bounded amount (bound above, in nuc units)"
            ),
        },
    }


def _observed_procrustes(
    s_ctx: torch.Tensor, Q_ctx: torch.Tensor, s_pfx: torch.Tensor, Q_pfx: torch.Tensor
) -> dict:
    """Exact observed residual via nuc(W_ctx W_pfxᵀ) = nuc(Σ_ctx (Q_ctx Q_pfxᵀ) Σ_pfx).

    Q_* are the Vh factors (r, d) of each raw-space operator SVD; left singular
    factors drop out under the orthogonal-invariance of singular values.
    """
    fro2_c = float((s_ctx.double() ** 2).sum())
    fro2_p = float((s_pfx.double() ** 2).sum())
    m = (
        s_ctx.double().view(-1, 1)
        * (Q_ctx.double() @ Q_pfx.double().T)
        * s_pfx.double().view(1, -1)
    )
    nuc = float(torch.linalg.svdvals(m).sum())
    res = float(np.sqrt(max(0.0, fro2_c + fro2_p - 2.0 * nuc)) / np.sqrt(fro2_c))
    return {"residual": res, "nuclear_cross": nuc, "fro2_ctx": fro2_c, "fro2_pfx": fro2_p}


def _unit_seed(base_seed: int, unit_key: str) -> int:
    return base_seed ^ int.from_bytes(hashlib.sha256(unit_key.encode()).digest()[:6], "big")


def _stage_layer(hub, inventory, cells: list[str], layer: int, args) -> list[dict]:
    """Stage this layer's Part-B input kinds (arms + targets) for the given cells."""
    from issue1092_p6_run import _layer_pat, _select, stage_file

    files = []
    for cell in cells:
        for kind in PARTB_KINDS:
            files.extend(
                _select(inventory, cell, _layer_pat(kind, layer), f"{cell}/{kind} L{layer:02d}")
            )
    return [stage_file(hub, f, args.hf_prefix, args.summaries_dir) for f in files]


def run(args: argparse.Namespace) -> dict:
    t0 = time.monotonic()
    run_selftest("cpu")  # PRESS engine exactness gate (matches the fit grid's own gate)
    args.summaries_dir = args.summaries_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    cells = _parse_csv(args.cells, tuple(CELL_MODEL_TYPE))
    unknown = [c for c in cells if c not in CELL_MODEL_TYPE]
    if unknown:
        raise ValueError(f"unknown cells: {unknown}")
    layers = _parse_layers(args.layers)
    bases = _parse_csv(args.target_bases, ("ambient", "pca48"))
    targets = _parse_csv(args.targets, ("t1", "t2", "t3"))
    partb_dir = args.out_dir / "partb"
    partb_dir.mkdir(parents=True, exist_ok=True)
    rows = _jsonl(args.corpus_dir / "manifest.jsonl")
    corpus_manifest_sha = hashlib.sha256(
        (args.corpus_dir / "manifest.jsonl").read_bytes()
    ).hexdigest()[:16]

    hub = None
    inventory = None
    if args.stage_from_hub:
        from issue1092_p6_run import HfHubIO, LocalFixtureHubIO, build_inventory

        if args.fixture_hub_root is not None:
            hub = LocalFixtureHubIO(args.fixture_hub_root.resolve())
        else:
            hub = HfHubIO(HF_DATA_REPO, args.hf_revision)
        inventory = build_inventory(hub, args.hf_prefix)

    units: list[dict] = []
    for layer in layers:
        staged: list[dict] = []
        if args.stage_from_hub:
            staged = _stage_layer(hub, inventory, cells, layer, args)
            print(f"[partb] phase=stage layer={layer:02d} staged={len(staged)}", flush=True)
        for cell in cells:
            x_by_arm: dict[str, np.ndarray] = {}
            input_paths: list[Path] = []
            for arm in ("prefix_end", "context_end"):
                x_by_arm[arm], paths = _load_summary(args.summaries_dir, cell, arm, layer)
                input_paths.extend(paths)
            y_blocks = []
            for target in targets:
                y, paths = _load_summary(args.summaries_dir, cell, target, layer)
                y_blocks.append(y)
                input_paths.extend(paths)
            Y_stacked = np.concatenate(y_blocks, axis=1)
            n0 = min(
                x_by_arm["prefix_end"].shape[0],
                x_by_arm["context_end"].shape[0],
                Y_stacked.shape[0],
                len(rows),
            )
            base_rows = rows[:n0]
            idx = np.asarray(_fit_arm_indices("A", base_rows), dtype=np.int64)
            assert idx.size >= 3, f"{cell} L{layer}: too few battery-excluded rows ({idx.size})"
            Xp = x_by_arm["prefix_end"][:n0][idx]
            Xc = x_by_arm["context_end"][:n0][idx]
            Yn = Y_stacked[:n0][idx]
            # X-side PressRidge factors are Y-independent: build ONCE per
            # (cell, layer), share across bases (shared-factorization rule).
            fit_p = _fit_press(Xp)
            fit_c = _fit_press(Xc)
            for basis in bases:
                Yb, _basis_info = _basis_targets_with_info(
                    Yn,
                    basis,
                    hidden_dim=args.hidden_dim,
                    targets=targets,
                    projection_target="t1",
                )
                unit_key = f"{cell}_L{layer:02d}_{basis}"
                config = {
                    "cell": cell,
                    "layer": layer,
                    "basis": basis,
                    "targets": targets,
                    "n_rows": int(idx.size),
                    "seed": args.seed,
                    "n_null_draws": args.n_null_draws,
                    "null_energy": args.null_energy,
                    "null_max_rank": args.null_max_rank,
                    "k_subspace": args.k_subspace,
                    "hidden_dim": args.hidden_dim,
                    "corpus_manifest_sha256": corpus_manifest_sha,
                }
                fp = _fingerprint(input_paths, config)
                ckpt = partb_dir / f"{unit_key}_{fp}.json"
                if ckpt.exists():
                    units.append(json.loads(ckpt.read_text()))
                    print(f"[partb] phase=unit unit={unit_key} skipped=complete", flush=True)
                    continue
                tu = time.monotonic()
                Yt = torch.from_numpy(np.ascontiguousarray(Yb)).double()
                mse_p, G_p, _ymu_p = _press_mse(fit_p, Yt)
                mse_c, G_c, _ymu_c = _press_mse(fit_c, Yt)
                lam_p = int(torch.argmin(mse_p).item())
                lam_c = int(torch.argmin(mse_c).item())
                matched_idx = int(torch.argmin(mse_p + mse_c).item())
                lam = float(RIDGE_LAMBDAS[matched_idx])
                W_p = _operator_raw(fit_p, G_p, lam)
                W_c = _operator_raw(fit_c, G_c, lam)
                del G_p, G_c
                U_p, s_p, Qh_p = torch.linalg.svd(W_p, full_matrices=False)
                U_c, s_c, Qh_c = torch.linalg.svd(W_c, full_matrices=False)
                del W_p, W_c
                d_in = fit_p["d_full"]
                P_out = int(U_p.shape[0])
                r = int(s_p.shape[0])
                k48 = min(args.k_subspace, r)
                k90_p, k90_c = _k90(s_p), _k90(s_c)
                gen = torch.Generator().manual_seed(_unit_seed(args.seed, unit_key))
                angle_reads: dict[str, Any] = {}
                for name, k1, k2, A, B, dim in (
                    ("input_k48", k48, k48, Qh_p.T, Qh_c.T, d_in),
                    ("input_k90", k90_p, k90_c, Qh_p.T, Qh_c.T, d_in),
                    ("output_k48", k48, k48, U_p, U_c, P_out),
                    ("output_k90", k90_p, k90_c, U_p, U_c, P_out),
                ):
                    angles = _angles_between(A[:, :k1], B[:, :k2])
                    angle_reads[name] = {
                        "k_prefix": k1,
                        "k_context": k2,
                        "angles_rad": angles,
                        "mean_angle_rad": float(np.mean(angles)) if angles else float("nan"),
                        "null": _angle_null_band(
                            dim,
                            k1,
                            k2,
                            args.n_null_draws,
                            args.null_chunk,
                            gen,
                            args.null_max_rank,
                        ),
                        "degenerate_note": (
                            f"k >= min operator rank/space dim (k=({k1},{k2}), dim={dim}, "
                            f"rank={r}): subspace comparison trivial"
                            if max(k1, k2) >= min(dim, r) and dim <= max(k1, k2)
                            else None
                        ),
                    }
                observed = _observed_procrustes(s_c, Qh_c, s_p, Qh_p)
                proc_null = _procrustes_null_band(
                    s_c,
                    s_p,
                    d_in,
                    energy=args.null_energy,
                    max_rank=args.null_max_rank,
                    n_draws=args.n_null_draws,
                    chunk=args.null_chunk,
                    gen=gen,
                )
                unit = {
                    "read": "partB_operator_arm_comparison",
                    "cell": cell,
                    "layer": layer,
                    "basis": basis,
                    "targets": targets,
                    "n_rows_battery_excluded": int(idx.size),
                    "n_rows_total_scope": int(n0),
                    "lambda": {
                        "grid": [float(x) for x in RIDGE_LAMBDAS],
                        "press_argmin_prefix_idx": lam_p,
                        "press_argmin_context_idx": lam_c,
                        "matched_idx": matched_idx,
                        "matched_lambda": lam,
                        "criterion": (
                            "argmin over the shared grid of press_mse_prefix + press_mse_context "
                            "(both arms fit the SAME centered target, so the PRESS MSEs are "
                            "commensurable); arms compared at this ONE matched lambda"
                        ),
                    },
                    "operator": {
                        "convention": "y_centered ≈ W_raw (x - mu); shape (P_out, d_in)",
                        "d_in": d_in,
                        "P_out": P_out,
                        "rank": r,
                        "kept_dims_prefix": int(fit_p["keep"].sum()),
                        "kept_dims_context": int(fit_c["keep"].sum()),
                        "basis_note": (
                            "standardized-X ridge fit; operator reported in RAW input "
                            "coordinates (1/sd folded in, kept dims embedded) so both arms' "
                            "input subspaces share one residual-stream basis"
                        ),
                    },
                    "spectra": {
                        "prefix": {
                            "top64": [float(v) for v in s_p[:64]],
                            "fro_sq": float((s_p.double() ** 2).sum()),
                            "nuclear": float(s_p.double().sum()),
                            "k90_energy": k90_p,
                        },
                        "context": {
                            "top64": [float(v) for v in s_c[:64]],
                            "fro_sq": float((s_c.double() ** 2).sum()),
                            "nuclear": float(s_c.double().sum()),
                            "k90_energy": k90_c,
                        },
                        "shrinkage_note": "ridge shrinkage biases these spectra low",
                    },
                    "principal_angles": angle_reads,
                    "procrustes": {**observed, "null": proc_null},
                    "row_space_note": (
                        f"subspace claims restricted to the data-spanned row space: "
                        f"n={int(idx.size)} battery-excluded rows vs d_in={d_in} "
                        f"(full column rank after degenerate-dim drop: "
                        f"{int(fit_p['keep'].sum())}/{int(fit_c['keep'].sum())} kept dims)"
                    ),
                    "wall_s": time.monotonic() - tu,
                    "fingerprint": fp,
                    "git_commit": _git_commit(),
                    "timestamp": _timestamp(),
                    "versions": {"numpy": np.__version__, "torch": torch.__version__},
                }
                _write_json_atomic(ckpt, unit)
                units.append(unit)
                print(
                    f"[partb] phase=unit-done unit={unit_key} lam={lam:g} "
                    f"proc_res={observed['residual']:.4f} wall_s={unit['wall_s']:.1f}",
                    flush=True,
                )
        if staged:
            from issue1092_p6_run import delete_staged

            n_deleted = delete_staged(staged, args.summaries_dir)
            print(f"[partb] phase=stage-cleanup layer={layer:02d} deleted={n_deleted}", flush=True)

    summary = {
        "phase": "P6_partB_operator",
        "n_units": len(units),
        "unit_files": sorted(
            p.name for p in partb_dir.glob("*.json") if p.name != "partb_summary.json"
        ),
        "cells": cells,
        "layers": layers,
        "bases": bases,
        "row_filter": "battery-excluded fit-arm-A rows (issue1092_fit_grid._fit_arm_indices)",
        "topic_matched_pairing_delta": TOPIC_MATCHED_PAIRING_DELTA_DROP,
        "wall_s": time.monotonic() - t0,
        "git_commit": _git_commit(),
        "timestamp": _timestamp(),
        "argv": sys.argv[1:],
    }
    _write_json_atomic(partb_dir / "partb_summary.json", summary)
    print(
        f"[partb] artifact digest: units={len(units)} "
        f"summary={partb_dir / 'partb_summary.json'} wall_s={summary['wall_s']:.1f}",
        flush=True,
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--summaries-dir", type=Path, required=True)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cells", default=None, help="CSV of cells (default: all 8 plan cells)")
    p.add_argument("--layers", default="14,18,19")
    p.add_argument("--target-bases", default="ambient,pca48")
    p.add_argument("--targets", default="t1,t2,t3")
    p.add_argument("--hidden-dim", type=int, default=3584)
    p.add_argument("--k-subspace", type=int, default=48)
    p.add_argument("--n-null-draws", type=int, default=200)
    p.add_argument("--null-energy", type=float, default=0.995)
    p.add_argument("--null-max-rank", type=int, default=1024)
    p.add_argument("--null-chunk", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--stage-from-hub",
        action="store_true",
        help="Stage per-layer Part-B inputs from the Hub (or --fixture-hub-root) into "
        "--summaries-dir, deleting after each layer; default reads --summaries-dir as-is.",
    )
    p.add_argument("--hf-prefix", default="issue1092_realistic_crossing/analysis_tensors/summaries")
    p.add_argument("--hf-revision", default="main")
    p.add_argument(
        "--fixture-hub-root",
        type=Path,
        default=None,
        help="Offline smoke: stage from this local tree instead of the HF Hub.",
    )
    return p.parse_args(argv)


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
