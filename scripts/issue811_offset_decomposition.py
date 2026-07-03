#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, δ, r̂, M⁺, ×) in scientific docstrings + logs.
"""Issue #811 F1 — constant-offset decomposition of the turn-boundary function change.

The #811 production run found the harmful-compliance (em) turn-boundary Δ clears
its refit floor at L7 (2.05×) and L14 (1.46×) with ZERO chain-ρ support — a
pattern consistent with a CONTEXT-INDEPENDENT end-of-turn offset (every EM
adapter shifting the turn-close state by ~the same vector along r̂_B) rather
than a genuinely reshaped input-dependent map. This free-analysis follow-up
decides which, per (behavior, layer, summary), on the EXISTING paired store:

1. Re-fit M0 (c0 → v0) and M⁺ (cplus → v_plus) with the SAME closed-form
   PRESS-LOCO ridge recipe as the run (REUSING ``issue722_fit_M`` helpers —
   ``_pca_basis_v0`` top-64 v0-PC target, ``_ridge_fit_predict`` over the
   1e-2…1e3 λ grid, ``_r_hat_for``; nothing re-implemented).
2. SANITY GATE: the re-fit must reproduce the run's per-cell ``Delta_med``
   (``eval_results/issue_811/cells/*.json``) within ``--repro-rel-tol``
   (default 1%). A divergent re-fit writes the JSON with ``repro_pass: false``
   and exits 3 — the decomposition is NOT trusted from a divergent fit.
3. SIGNED per-context projections ``δ(c) = (M⁺(c) − M0(c))·r̂_B`` over the
   16-context grid (c_C is source-keyed: 480 grid rows = 16 distinct contexts
   × 30 identical repeats — asserted, then collapsed to 16 values).
4. Decompose: ``offset = mean_c δ(c)`` (grid mean over the 16 contexts);
   ``residual(c) = δ(c) − offset``. Report ``Δ_med_raw = median_c |δ(c)|``
   (= the run's Delta_med, reproduced) vs ``Δ_med_residual = median_c
   |residual(c)|``, each against the cell's EXISTING ``floor_combined``
   (floors are NOT re-fit — they are read from the run's cell JSONs).
5. Decision read per cell: raw above floor but residual below floor →
   the above-floor read is a uniform end-of-turn offset (``uniform_offset``);
   residual still above floor → genuine input-dependent reshaping survives
   (``input_dependent_reshaping_survives``); raw below floor →
   ``raw_below_floor`` (nothing above floor to decompose).

Output: ``eval_results/issue_811/offset_decomposition.json`` — all 18 cells
(3 behaviors × 3 layers × 2 summaries) with Δ_med_raw, |offset|,
Δ_med_residual (family-clustered bootstrap CIs via the run's own
``clustered_bootstrap_scalar``), floor_combined, ratio_raw, ratio_residual,
verdict, and the per-context signed δ / residual arrays (16 floats each) for
EVERY cell so the analyzer can plot the per-unit view.

Store: the run's own re-extracted paired store
(``issue811_turn_nl_mapchange/analysis_tensors`` on the HF data repo, 4,320
npz ≈ 7.3 GB), downloaded once into ``data/issue_811/hf_dl/`` (a
re-downloadable cache) and read via the loader's local-mirror mode. The
download enumerates via ``list_store_layout`` (per-level tree API with
retries — the full-repo ``list_repo_files`` times out on this repo) and
fetches per-file in a bounded thread pool, skipping already-present files
(resumable). 0 GPU-h; CPU minutes.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project wrapper (NOT bare dotenv): resolves the worktree .env, sets HF_HOME,
# and applies the shared-VM thread-cap setdefaults BEFORE torch is imported
# (torch arrives via the issue722_fit_M import below).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue658_fit_predictors as fit658  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402
import numpy as np  # noqa: E402
from issue722_bootstrap import clustered_bootstrap_scalar  # noqa: E402
from issue811_fit import HEADLINE_BEHAVIORS, STORE_PREFIX, SUMMARIES, SWEEP_LAYERS  # noqa: E402

logger = logging.getLogger("issue811.offset_decomposition")

DATA_REPO = loadact.DATA_REPO
# 16 sources × 30 targets × 3 layers × 3 behaviors = 4,320 layer-baked .npz files.
EXPECTED_STORE_FILES = 4320
EXPECTED_CONTEXTS = 16
EXPECTED_TARGETS_PER_CONTEXT = 30
DEFAULT_DL_ROOT = PROJECT_ROOT / "data/issue_811/hf_dl"
DEFAULT_OUT = PROJECT_ROOT / "eval_results/issue_811/offset_decomposition.json"
RUN_CELLS_DIR = PROJECT_ROOT / "eval_results/issue_811/cells"
REPRO_REL_TOL = 0.01  # brief item 8: >1% relative deviation on Delta_med → STOP


def download_store(
    dl_root: Path,
    behaviors: tuple[str, ...],
    *,
    workers: int = 12,
    layers: tuple[int, ...] | None = None,
    prefix: str = STORE_PREFIX,
) -> tuple[Path, int]:
    """Mirror the #811 paired store from HF into ``dl_root`` (resumable, parallel).

    Enumerates via ``loadact.list_store_layout`` (per-level ``list_repo_tree``
    with bounded retry — the recursive full-repo listing 504s/times out on this
    repo), asserts the expected 4,320-file count when all three behaviors are
    requested, then fetches each missing file with ``hf_hub_download(local_dir=
    dl_root)`` so the repo-relative path is preserved (the loader's local-mirror
    mode reads ``<dl_root>/<STORE_PREFIX>/<beh>/<src>/<fn>``). Already-present
    non-empty files are skipped (resume). ``layers`` optionally restricts the
    fetch to the layer-baked ``*_L{li}.npz`` subset (the chain-scatter figure
    needs L14 only — 1/3 of the store); the 4,320-count assert then checks the
    filtered expectation instead. Returns ``(local_root, n_downloaded)``.
    """
    from huggingface_hub import hf_hub_download

    layout = loadact.list_store_layout(behaviors, prefix=prefix)
    rel_paths: list[str] = []
    for beh, srcs in layout.items():
        for src_dir, files in srcs.items():
            for fn in files:
                if layers is not None and not any(fn.endswith(f"_L{li}.npz") for li in layers):
                    continue
                rel_paths.append(f"{beh}/{src_dir}/{fn}")
    logger.info("[phase=download] store listing: %d files under %s", len(rel_paths), prefix)
    if set(behaviors) == set(HEADLINE_BEHAVIORS):
        n_layers = len(layers) if layers is not None else len(SWEEP_LAYERS)
        expected = EXPECTED_STORE_FILES * n_layers // len(SWEEP_LAYERS)
        assert len(rel_paths) == expected, (
            f"store listing has {len(rel_paths)} files, expected {expected} "
            f"(16 sources × 30 targets × {n_layers} layers × 3 behaviors) — "
            f"wrong prefix or partial store?"
        )
    local_root = dl_root / prefix
    missing = [
        rel
        for rel in rel_paths
        if not (local_root / rel).exists() or (local_root / rel).stat().st_size == 0
    ]
    logger.info("[phase=download] %d/%d files missing locally", len(missing), len(rel_paths))

    def _fetch(rel: str) -> str:
        hf_hub_download(
            DATA_REPO,
            f"{prefix}/{rel}",
            repo_type="dataset",
            local_dir=str(dl_root),
        )
        return rel

    n_done = 0
    if missing:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fetch, rel): rel for rel in missing}
            for fut in as_completed(futs):
                fut.result()  # fail loud on any download error
                n_done += 1
                if n_done % 250 == 0 or n_done == len(missing):
                    logger.info("[phase=download] %d/%d fetched", n_done, len(missing))
    # Post-download completeness check (exact set, not count-only).
    still_missing = [r for r in rel_paths if not (local_root / r).exists()]
    assert not still_missing, f"{len(still_missing)} files still missing after download"
    return local_root, n_done


def decompose_cell(
    behavior: str,
    layer: int,
    summary: str,
    cells: list,
    rb_main: dict,
    rb_fact: dict | None,
    *,
    target_dim: int,
    strict: bool,
    cells_dir: Path = RUN_CELLS_DIR,
) -> dict:
    """Re-fit M0/M⁺ (run-identical ridge), reproduce Delta_med, decompose δ(c).

    Returns the per-cell record: the reproduction check (``Delta_med_refit`` vs
    the run's ``Delta_med``, relative deviation), the signed per-context δ(c)
    (16 values, within-context repeats asserted identical), the constant-offset
    decomposition (offset / residual, medians + family-clustered CIs), the
    ratios against the run's existing ``floor_combined``, and the verdict.
    Raises on a missing run cell JSON or a within-context δ spread (both mean
    the substrate is not what the run fit — fail loud, never paper over).
    """
    stacks = loadact.stack_for_fit(cells)
    C0, Cplus, V0, Vplus = stacks["C0"], stacks["Cplus"], stacks["V0"], stacks["Vplus"]
    families = stacks["families"]
    n = C0.shape[0]
    assert n >= 4, f"{behavior} L{layer} {summary}: only {n} cells (<4)"
    if strict:
        assert n == loadact.EXPECTED_CELLS_PER_BEHAVIOR_LAYER, (behavior, layer, summary, n)

    # Run-identical ridge headline path (issue722_fit_M.fit_cell, ridge-only part).
    r_hat = fitM._r_hat_for(behavior, layer, rb_main, rb_fact)
    pca_basis = fitM._pca_basis_v0(V0, target_dim)
    V0_64 = fitM._to64(V0, pca_basis)
    Vplus_64 = fitM._to64(Vplus, pca_basis)
    grid = loadact.common_c_grid(stacks)
    m0_grid = fitM._ridge_fit_predict(C0, V0_64, grid)
    mplus_grid = fitM._ridge_fit_predict(Cplus, Vplus_64, grid)
    delta_signed = ((mplus_grid - m0_grid) @ pca_basis) @ r_hat  # (n,) SIGNED δ(c)

    # ---- Reproduction sanity gate vs the run's own cell JSON ----
    run_path = cells_dir / f"{behavior}_L{layer}_{summary}.json"
    # Repo-relative when the cells dir is inside the repo (production); absolute
    # otherwise (an out-of-tree smoke dir) — metadata formatting only.
    try:
        run_path_repr = str(run_path.relative_to(PROJECT_ROOT))
    except ValueError:
        run_path_repr = str(run_path)
    run_cell = json.loads(run_path.read_text())
    delta_med_run = float(run_cell["Delta_med"])
    floor_combined = float(run_cell["floor_combined"])
    delta_med_refit = float(np.median(np.abs(delta_signed)))
    rel_dev = abs(delta_med_refit - delta_med_run) / max(abs(delta_med_run), 1e-300)

    # ---- Collapse the 480-row grid to the 16 distinct contexts ----
    src = np.asarray([str(s) for s in stacks["source_cids"]], dtype=object)
    uniq_src = sorted(set(src.tolist()))
    if strict:
        assert len(uniq_src) == EXPECTED_CONTEXTS, (behavior, layer, summary, len(uniq_src))
    delta_ctx: dict[str, float] = {}
    max_within_spread = 0.0
    for s in uniq_src:
        vals = delta_signed[src == s]
        if strict:
            assert vals.size == EXPECTED_TARGETS_PER_CONTEXT, (s, vals.size)
        spread = float(np.ptp(vals))
        max_within_spread = max(max_within_spread, spread)
        delta_ctx[s] = float(vals[0])
    # c_C is identical within a source, so the ridge prediction (hence δ) must be
    # identical across a source's 30 grid repeats — a spread means the grid rows
    # were NOT source-keyed repeats (wrong store / wrong pairing). Fail loud.
    scale = max(1.0, float(np.max(np.abs(delta_signed))))
    assert max_within_spread <= 1e-8 * scale, (
        f"{behavior} L{layer} {summary}: within-context δ spread {max_within_spread:.3e} "
        f"(scale {scale:.3e}) — grid rows are not source-keyed repeats"
    )

    # ---- Constant-offset decomposition ----
    ctx_vals = np.asarray([delta_ctx[s] for s in uniq_src], dtype=np.float64)
    offset = float(ctx_vals.mean())  # grid mean over the 16 contexts (uniform 30× repeats)
    residual_grid = delta_signed - offset
    residual_ctx = {s: float(delta_ctx[s] - offset) for s in uniq_src}
    raw_ci = clustered_bootstrap_scalar(np.abs(delta_signed), families, statistic="median")
    res_ci = clustered_bootstrap_scalar(np.abs(residual_grid), families, statistic="median")
    delta_med_raw = float(raw_ci["point"])
    delta_med_residual = float(res_ci["point"])
    ratio_raw = delta_med_raw / floor_combined
    ratio_residual = delta_med_residual / floor_combined
    if ratio_raw <= 1.0:
        verdict = "raw_below_floor"
    elif ratio_residual <= 1.0:
        verdict = "uniform_offset"
    else:
        verdict = "input_dependent_reshaping_survives"

    return {
        "behavior": behavior,
        "layer": layer,
        "summary": summary,
        "n_cells": int(n),
        "n_contexts": len(uniq_src),
        "repro": {
            "Delta_med_run": delta_med_run,
            "Delta_med_refit": delta_med_refit,
            "rel_deviation": float(rel_dev),
            "run_cell_json": run_path_repr,
        },
        "Delta_med_raw": delta_med_raw,
        "Delta_med_raw_ci": raw_ci,
        "offset": offset,
        "abs_offset": abs(offset),
        "offset_frac_of_raw": (abs(offset) / delta_med_raw) if delta_med_raw > 0 else None,
        "Delta_med_residual": delta_med_residual,
        "Delta_med_residual_ci": res_ci,
        "floor_combined": floor_combined,
        "ratio_raw": ratio_raw,
        "ratio_residual": ratio_residual,
        "verdict": verdict,
        "max_within_context_spread": max_within_spread,
        "delta_per_context": delta_ctx,
        "residual_per_context": residual_ctx,
    }


def _git_commit() -> str:
    """Current HEAD sha of the worktree this script lives in (repro metadata)."""
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #811 F1 constant-offset decomposition")
    ap.add_argument("--behaviors", nargs="+", default=list(HEADLINE_BEHAVIORS))
    ap.add_argument("--layers", nargs="+", type=int, default=list(SWEEP_LAYERS))
    ap.add_argument("--summaries", nargs="+", default=list(SUMMARIES))
    ap.add_argument("--dl-root", type=Path, default=DEFAULT_DL_ROOT)
    ap.add_argument(
        "--store-prefix",
        default=STORE_PREFIX,
        help="HF prefix of the round's paired store (maxp round: "
        "issue811_maxp_mapchange/analysis_tensors; default: the v1 turn_nl store)",
    )
    ap.add_argument(
        "--cells-dir",
        type=Path,
        default=RUN_CELLS_DIR,
        help="the run's per-cell fit JSONs the reproduction gate compares against "
        "(maxp round: eval_results/issue_811/maxp-winner-mapchange/cells)",
    )
    ap.add_argument(
        "--local-store-root",
        type=Path,
        default=None,
        help="read the paired store DIRECTLY from this local dir "
        "({behavior}/{source}_seed42/{target}_L{li}.npz layout) — no HF fetch; "
        "the in-dispatch F1 phase points this at the just-extracted store",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=12, help="parallel HF download workers")
    ap.add_argument("--skip-download", action="store_true", help="store already mirrored locally")
    ap.add_argument("--target-dim", type=int, default=fit658.A35_MLP_TARGET_DIM)
    ap.add_argument("--num-threads", type=int, default=8, help="torch.set_num_threads (shared VM)")
    ap.add_argument("--repro-rel-tol", type=float, default=REPRO_REL_TOL)
    ap.add_argument("--max-sources", type=int, default=None, help="smoke: cap sources (>=2)")
    ap.add_argument("--max-targets-per-source", type=int, default=None)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(max(1, args.num_threads))
    fit658.DEVICE = fit658._resolve_device("auto")
    fitM.TARGET_DIM = args.target_dim
    logger.info("[phase=setup] device=%s target_dim=%d", fit658.DEVICE, args.target_dim)

    behaviors = tuple(args.behaviors)
    layers = tuple(args.layers)
    summaries = tuple(args.summaries)
    strict = args.max_sources is None and args.max_targets_per_source is None

    n_downloaded = 0
    if args.local_store_root is not None:
        local_root = args.local_store_root
        assert local_root.is_dir(), f"--local-store-root not a dir: {local_root}"
    elif args.skip_download:
        local_root = args.dl_root / args.store_prefix
        assert local_root.is_dir(), f"--skip-download but no local mirror at {local_root}"
    else:
        local_root, n_downloaded = download_store(
            args.dl_root, behaviors, workers=args.workers, prefix=args.store_prefix
        )

    rb_main = fitM._load_rb_main()
    # required=True: fail LOUD on a load failure (rb-fact-silent-drop-headline);
    # None means ONLY the data-declared degenerate flag (plan §8).
    rb_fact = fitM._load_rb_fact(required=True) if "fact" in behaviors else None
    if "fact" in behaviors and rb_fact is None:
        logger.warning("r_b_fact.pt flagged degenerate (plan §8) — dropping fact")
        behaviors = tuple(b for b in behaviors if b != "fact")

    layout = loadact.list_store_layout_local(local_root, behaviors)
    out_cells: dict[str, dict] = {}
    meta = {
        "issue": 811,
        "followup": "F1_offset_decomposition",
        "git_commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "store_prefix": args.store_prefix,
        "local_store_root": str(args.local_store_root) if args.local_store_root else None,
        "cells_dir": str(args.cells_dir),
        "n_files_downloaded_this_run": n_downloaded,
        "device": fit658.DEVICE,
        "target_dim": args.target_dim,
        "repro_rel_tol": args.repro_rel_tol,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }

    def _write(final: bool) -> None:
        rel_devs = [c["repro"]["rel_deviation"] for c in out_cells.values()]
        payload = {
            "meta": meta,
            "max_rel_deviation": max(rel_devs) if rel_devs else None,
            "repro_pass": bool(rel_devs and max(rel_devs) <= args.repro_rel_tol),
            "complete": final,
            "cells": out_cells,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=float))

    # One summary × behavior load at a time bounds resident memory (~165 MB/load);
    # the out JSON is re-written after EVERY cell (checkpoint-per-phase).
    for summary in summaries:
        for behavior in behaviors:
            cells_by = loadact.load_cells(
                behaviors=(behavior,),
                layers=layers,
                max_sources=args.max_sources,
                max_targets_per_source=args.max_targets_per_source,
                streamer=loadact._Streamer(local_root=local_root),
                strict_counts=strict,
                layout=layout,
                summary=summary,
            )
            for layer in layers:
                key = f"{behavior}/L{layer}/{summary}"
                rec = decompose_cell(
                    behavior,
                    layer,
                    summary,
                    cells_by[(behavior, layer)],
                    rb_main,
                    rb_fact,
                    target_dim=args.target_dim,
                    strict=strict,
                    cells_dir=args.cells_dir,
                )
                out_cells[key] = rec
                _write(final=False)
                logger.info(
                    "[phase=decompose] %s: Δ_med_raw=%.4g (run %.4g, rel_dev %.2e) "
                    "|offset|=%.4g Δ_med_residual=%.4g floor=%.4g "
                    "ratio_raw=%.2f ratio_residual=%.2f → %s",
                    key,
                    rec["Delta_med_raw"],
                    rec["repro"]["Delta_med_run"],
                    rec["repro"]["rel_deviation"],
                    rec["abs_offset"],
                    rec["Delta_med_residual"],
                    rec["floor_combined"],
                    rec["ratio_raw"],
                    rec["ratio_residual"],
                    rec["verdict"],
                )

    # Requested-set completeness gate BEFORE the final (complete=True) write (r9
    # reconciler standing rec): every requested behavior x layer x summary cell must
    # be present — a silently skipped cell must never ship inside a payload stamped
    # complete. `behaviors` is the post-degenerate-drop tuple (the set that ran).
    expected_keys = {f"{b}/L{ly}/{s}" for s in summaries for b in behaviors for ly in layers}
    missing_keys = sorted(expected_keys - set(out_cells))
    assert not missing_keys, (
        f"[offset-completeness-assert] {len(missing_keys)}/{len(expected_keys)} requested "
        f"cells missing before final write: {missing_keys}"
    )
    _write(final=True)
    max_dev = max(c["repro"]["rel_deviation"] for c in out_cells.values())
    # NOTE: this script now ALSO runs INSIDE issue811_dispatch.sh (the maxp arm's
    # F1 phase), whose main log reserves the literal `[phase=done]` token for the
    # dispatcher's single terminal line (pod-side reporting contract, #545) — so
    # this completion line deliberately does NOT carry the phase tag.
    logger.info(
        "offset-decomposition complete: %d cells → %s (max_rel_deviation=%.3e, repro %s)",
        len(out_cells),
        args.out,
        max_dev,
        "PASS" if max_dev <= args.repro_rel_tol else "FAIL",
    )
    if max_dev > args.repro_rel_tol:
        logger.error(
            "offset-decomposition REPRO FAIL: re-fit Delta_med deviates >%.1f%% from the run "
            "on at least one cell — the decomposition is NOT trusted (brief item 8). "
            "JSON written with repro_pass=false for diagnosis.",
            100 * args.repro_rel_tol,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
