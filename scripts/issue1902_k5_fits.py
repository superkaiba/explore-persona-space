#!/usr/bin/env python3
"""Issue #1902 follow-up: K=5 mean answer targets for the fig:posttraining remake.

Moves the answer TARGET of the Section 4.3 last-token maps from the single
seed-42 completion to the MEAN answer vector over K=5 sampled completions
(seed 42 + seeds 45-48), on the seven captured cells (diagonal B/B, S/S, D/D,
R/R plus the base-representation row B/S, B/D, B/R). Every draw enters the
mean (the #1491 main-paper protocol); per-context draw spread and flagged-draw
counts are persisted next to the targets.

``--full-grid --draw-revision <SHA>`` extends this to all sixteen cells,
adds identity+bias and cosine/Euclidean held-out retrieval companions, and
renders both the line and matrix views. It requires a fresh output directory
on first use, binds derived caches to a run identity, and hash-verifies
cached source files against their immutable Hub revision. The full-grid
paper analysis needs only ``--target-layers 31``; captured L18 draws remain
available separately on the Hub.

Estimator, folds, transfer modes, and retrieval are IMPORTED from the
committed pipeline, never re-implemented:

- ``issue1902_lasttoken_transfer``: ``load_fold_of`` (six IID random folds,
  seed 190231), ``SharedPrimalRidge`` (primal spectral GCV ridge),
  ``transfer_pair_fold`` (direct / bias / scale+bias), ``load_clusters``.
- ``issue1902_lasttoken_comparison``: store paths, ``_per_row_components``,
  ``_savez`` / ``_write_json``, ``_read_groups``, HF repo + revision pins.
- ``issue1902_lasttoken_retrieval``: ``cosine_ranks``,
  ``shrunk_whitening_stats``, ``whitened_csls_ranks``,
  ``cluster_bootstrap_acc1`` (whitened cosine + two-sided CSLS k=10).
- ``scripts/section43_posttraining_figure``: ``render_variant`` renders the
  K=5 remake from a data dict of the committed figure's schema.

Subcommands (resumable; ``all`` chains them):
  stage      Download the K=5 draw stores (seven cells x seeds 45-48 x
             L{18,31}) plus the seed-42 L18 off-diagonal shards into the
             writable store root. Fail-loud on a missing file (the pod has
             not finished); nothing is fabricated.
  targets    Build per-cell ``w_bar`` (mean over seeds {42,45,46,47,48}),
             per-context ``spread`` (K-1 denominator), ``k_eff`` and
             ``n_flagged`` (rollout ``truncated`` / ``repetition_flag``);
             persist ``targets/<cell>_L{18,31}.npz``.
  grid       Seven cells at L31 with the run_grid estimator + folds, target
             ``w_bar``; per-row OOF residual / total SS / cosine per cell.
             Non-smoke also reruns K=1 on the same rows as the PARITY ANCHOR
             (must reproduce the committed diagonal R^2 within 1e-3 - STOP
             on failure).
  transfer   Adjacent pairs B->S, S->D, D->R (direct / bias / scale+bias)
             on the K=5 targets via ``transfer_pair_fold``; retention + row
             and semantic-cluster bootstrap CIs into ``summary.json``.
  retrieval  Whitened-cosine + CSLS top-1 per stage on the K=5 targets.
  scatter    Per-context figure: x = draw spread, y = own-map OOF squared
             error against ``w_bar``; four stages pooled on one axis pair.
  figure     Regenerate ``c1_posttraining_dynamics`` as ``*_k5`` from the
             K=5 outputs (K=1 files untouched).

Smoke (``--smoke``): substitutes the seed-43/44 reliability stores (914
paired rows) for the not-yet-landed K=5 stores (K=3 = seeds 42+43+44) on the
FOUR diagonal cells, slices context features to 256 dims so the shared ridge
keeps ``n_train > d`` (the estimator refuses the degenerate regime by
design), and prints COUNTS ONLY - smoke fits are shape-only and are never
reported as results. Smoke outputs land under ``<out>/smoke``.

VM launches carry the shared-VM thread caps (#847)::

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue1902_k5_fits.py all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE numpy/BLAS imports on the shared VM.
load_dotenv()

import numpy as np  # noqa: E402

import issue1902_common as C  # noqa: E402
import issue1902_lasttoken_comparison as LC  # noqa: E402
import issue1902_lasttoken_retrieval as RT  # noqa: E402
import issue1902_lasttoken_transfer as XF  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)

STAGES = LC.STAGES  # ("B", "S", "D", "R")
LAYER = LC.LAYER  # 31
TARGET_LAYERS = (31, 18)
N_FOLDS = LC.N_FOLDS  # 6
CORPUS = "single"
K5_EXTRA_SEEDS = C.K5_SEEDS  # (45, 46, 47, 48)
SEED42 = C.GEN_SEED  # 42
SMOKE_SEEDS = (43, 44)  # reliability draws; NEVER k5 seeds
SMOKE_FEATURE_DIM = 256  # keeps n_train > d on the 914-row subset
BOOT_SEED = XF.BOOT_SEED  # 1944
N_BOOT = XF.N_BOOT  # 1000
ANCHOR_TOL = 1e-3

# Seven captured K=5 cells: diagonals + the base-representation row.
CELLS7: tuple[tuple[str, str], ...] = (
    ("B", "B"),
    ("S", "S"),
    ("D", "D"),
    ("R", "R"),
    ("B", "S"),
    ("B", "D"),
    ("B", "R"),
)
DIAG_CELLS: tuple[tuple[str, str], ...] = tuple((s, s) for s in STAGES)
FULL_GRID_CELLS: tuple[tuple[str, str], ...] = tuple((m, s) for m in STAGES for s in STAGES)
TRANSFER_PAIRS_K5: tuple[tuple[str, str], ...] = (("B", "S"), ("S", "D"), ("D", "R"))

DEFAULT_OUT = PROJECT_ROOT / "eval_results" / "issue_1902" / "k5_targets"
# Committed K=5 DIAGONAL fits away from layer 31 (scripts/issue1902_k5_layer18.py).
# These are the same-layer parity reference when --fit-layer is not 31: the K=1
# anchor has no committed counterpart there (lasttoken_transfer is layer 31 only).
K5_OFFLAYER_REFERENCE = {
    18: PROJECT_ROOT / "eval_results" / "issue_1902" / "k5_layer18" / "summary_L18.json"
}
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures" / "issue_1902" / "section43"
# Read-only staged copy of the existing seed-42 store on the main checkout
# (brief: reuse, never re-download); everything NEW stages into the worktree.
DEFAULT_RO_STAGE_ROOT = Path(
    "/home/thomasjiralerspong/explore-persona-space/tmp/issue1902_lasttoken_store"
)
DEFAULT_K5_ROOT = PROJECT_ROOT / "data" / "issue_1902" / "k5_store"
RAW_GEN_PREFIX = f"{C.HF_PREFIX}/raw_completions/gen/{CORPUS}"

STAGE_NAMES = {"B": "Base", "S": "SFT", "D": "DPO", "R": "RLVR"}
STAGE_COLORS = {"B": "#C98A1B", "S": "#176B87", "D": "#7B3294", "R": "#5AAE61"}
STAGE_MARKERS = {"B": "s", "S": "o", "D": "D", "R": "^"}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── store resolution (writable k5 root first, read-only staged root second) ──


def _resolve(cfg: "Config", relpath: str) -> Path | None:
    for root in (cfg.k5_root, cfg.ro_root, *cfg.reuse_roots):
        p = root / relpath
        if p.exists():
            if cfg.full_grid:
                _check_source_identity(cfg, p, relpath)
            return p
    return None


def _check_source_identity(cfg: Config, path: Path, relpath: str) -> None:
    """Verify any cached source against its immutable Hub revision before use."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    revision = (
        cfg.draw_revision
        if "/k5draws/" in relpath or "_k5_seed" in relpath or "/k5_targets/targets/" in relpath
        else LC.HF_REVISION
    )
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, revision)
    if key in cfg.verified_sources:
        return
    entries = hub.retry_transient(
        lambda: HfApi().get_paths_info(
            LC.HF_REPO, [relpath], repo_type="dataset", revision=revision
        ),
        what=f"verify cached input {relpath}",
    )
    if len(entries) != 1 or entries[0].size != stat.st_size:
        raise RuntimeError(f"cached input missing or wrong size at {revision}: {path}")
    entry = entries[0]
    if entry.lfs:
        digest = hashlib.sha256()
        expected = entry.lfs.sha256
    else:
        digest = hashlib.sha1(b"blob " + str(stat.st_size).encode() + b"\0")
        expected = entry.blob_id
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"cached input hash mismatch at {revision}: {path}")
    cfg.verified_sources.add(key)


def _download(cfg: Config, relpath: str, *, revision: str | None) -> Path:
    """Fetch ``relpath`` from the HF data repo into the writable k5 root."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    hub.retry_transient(
        lambda: hf_hub_download(
            LC.HF_REPO,
            relpath,
            repo_type="dataset",
            revision=revision,
            local_dir=cfg.k5_root,
        ),
        what=f"hf_hub_download {relpath}",
    )
    return cfg.k5_root / relpath


def _ensure(cfg: Config, relpath: str, *, revision: str | None) -> Path:
    found = _resolve(cfg, relpath)
    if found is not None:
        return found
    return _download(cfg, relpath, revision=revision)


def _draw_answer_relpath(m: str, s: str, seed: int, layer: int) -> str:
    """Store-relative answer shard for one draw of cell (m, s)."""
    if seed == SEED42:
        return f"{LC.HF_PREFIX}/{m}/{s}/{CORPUS}/L{layer}.pt"
    if seed in SMOKE_SEEDS:
        # Reliability draws exist for the DIAGONAL stage cells only.
        if m != s:
            raise ValueError(f"reliability draws have no off-diagonal cell ({m},{s})")
        return f"{LC.HF_PREFIX}/reliability/{s}/{CORPUS}/seed{seed}/L{layer}.pt"
    return f"{LC.HF_PREFIX}/{C.k5_store_relpath(m, s, CORPUS, seed, layer)}"


def _draw_revision(seed: int, cfg: Config | None = None) -> str | None:
    """Pinned revision for pre-existing draws; main for the fresh K=5 uploads."""
    return (
        LC.HF_REVISION if seed in (SEED42, *SMOKE_SEEDS) else (cfg.draw_revision if cfg else None)
    )


def _load_answer(path: Path) -> tuple[np.ndarray, list[str]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    return (
        payload["w"].to(torch.float32).numpy(),
        [str(v) for v in payload["row_ids"]],
    )


def _load_ctx(cfg: Config, stage: str) -> tuple[np.ndarray, list[str]]:
    import torch

    path = _resolve(cfg, f"{LC.HF_PREFIX}/{stage}/ctx/{CORPUS}/L{cfg.fit_layer}.pt")
    if path is None:
        raise FileNotFoundError(f"ctx shard missing for stage {stage} (stage the seed-42 store)")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return (
        payload["u_last"].to(torch.float32).numpy(),
        [str(v) for v in payload["row_ids"]],
    )


# ── rollout flags ────────────────────────────────────────────────────────────


def _rollout_name(src: str, seed: int) -> str:
    if seed == SEED42:
        return f"{src}.jsonl"
    if seed in SMOKE_SEEDS:
        return f"{src}_rel_seed{seed}.jsonl"
    return f"{src}_k5_seed{seed}.jsonl"


def _fetch_rollout_records(cfg: Config, src: str, seed: int) -> list[dict]:
    """Rollout records for (answer source, seed); handles the sharded layout."""
    from huggingface_hub.errors import EntryNotFoundError

    name = _rollout_name(src, seed)
    rel = f"{RAW_GEN_PREFIX}/{name}"
    revision = _draw_revision(seed, cfg)
    stem = name[: -len(".jsonl")]
    manifest_rel = f"{RAW_GEN_PREFIX}/{stem}.manifest.json"
    try:
        manifest_path = _ensure(cfg, manifest_rel, revision=revision)
    except EntryNotFoundError:
        path = _ensure(cfg, rel, revision=revision)
        return [json.loads(line) for line in path.read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())
    records: list[dict] = []
    for shard in manifest["shards"]:
        shard_path = _ensure(cfg, f"{RAW_GEN_PREFIX}/{shard['name']}", revision=revision)
        data = shard_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != shard["sha256"]:
            raise RuntimeError(f"shard hash mismatch: {shard['name']}")
        rows = [json.loads(line) for line in data.splitlines()]
        if len(rows) != int(shard["n_lines"]):
            raise RuntimeError(f"shard line-count mismatch: {shard['name']}")
        records.extend(rows)
    return records


def _flag_map(cfg: Config, src: str, seed: int) -> dict[str, bool]:
    recs = _fetch_rollout_records(cfg, src, seed)
    return {str(r["id"]): bool(r["truncated"]) or bool(r["repetition_flag"]) for r in recs}


# ── config ───────────────────────────────────────────────────────────────────


class Config:
    """Resolved run configuration (roots, seeds, cells, output dirs)."""

    def __init__(self, args: argparse.Namespace):
        self.smoke: bool = bool(args.smoke)
        self.force: bool = bool(args.force)
        self.full_grid: bool = args.full_grid
        self.draw_revision: str | None = args.draw_revision
        self.verified_sources: set[tuple] = set()
        self.reuse_roots: list[Path] = args.reuse_root
        self.flag_counts_root: Path | None = args.flag_counts_root
        self.target_layers: tuple[int, ...] = tuple(args.target_layers)
        # Layer the downstream fits READ; staging/targets stay multi-layer.
        # getattr default keeps Config constructible from a hand-built Namespace
        # (tests/test_issue1902_k5_fullgrid.py builds one without the new flag).
        self.fit_layer: int = int(getattr(args, "fit_layer", LAYER))
        if self.full_grid and self.smoke:
            raise ValueError(
                "full-grid production and diagonal reliability smoke are separate modes"
            )
        if self.full_grid and not self.draw_revision:
            raise ValueError("--full-grid requires --draw-revision (verified capture Hub SHA)")
        if self.draw_revision and (
            len(self.draw_revision) != 40
            or any(c not in "0123456789abcdef" for c in self.draw_revision)
        ):
            raise ValueError("--draw-revision must be a full immutable 40-character Hub SHA")
        if args.cmd not in ("stage", "targets") and self.fit_layer not in self.target_layers:
            raise ValueError(
                "downstream phases require --target-layers to include "
                f"{self.fit_layer} (the --fit-layer; got {list(self.target_layers)})"
            )
        if self.fit_layer != LAYER and self.fit_layer not in K5_OFFLAYER_REFERENCE:
            raise ValueError(
                f"--fit-layer {self.fit_layer} has no committed same-layer parity "
                f"reference (have: {sorted(K5_OFFLAYER_REFERENCE)})"
            )
        self.ro_root: Path = args.stage_root
        self.k5_root: Path = args.k5_root
        self.out: Path = args.out / "smoke" if self.smoke else args.out
        self.fig_dir: Path = (self.out / "figs") if self.smoke else args.figures_dir
        self.seeds: tuple[int, ...] = (
            (SEED42, *SMOKE_SEEDS) if self.smoke else (SEED42, *K5_EXTRA_SEEDS)
        )
        self.cells: tuple[tuple[str, str], ...] = (
            DIAG_CELLS if self.smoke else FULL_GRID_CELLS if self.full_grid else CELLS7
        )
        self.k5_root.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        if self.full_grid:
            manifest = self.out / "fullgrid_run_identity.json"
            identity = {
                "schema": 1,
                "draw_revision": self.draw_revision,
                "seed42_revision": LC.HF_REVISION,
                "seeds": list(self.seeds),
                "cells": ["".join(c) for c in self.cells],
                "layer": self.fit_layer,
                "fold_seed": LC.RANDOM_FOLD_SEED,
                "n_folds": N_FOLDS,
                "context_summary": "u_last",
                "estimator": "SharedPrimalRidge-GCV",
                "flag_counts_root": str(self.flag_counts_root) if self.flag_counts_root else None,
            }
            if manifest.exists():
                if json.loads(manifest.read_text()) != identity:
                    raise RuntimeError("full-grid cache identity changed; use a fresh --out")
            else:
                cached_targets = list((self.out / "targets").glob(f"*_L{self.fit_layer}.npz"))
                cached_grid = list((self.out / "percell").glob("k5grid_*.npz"))
                if cached_targets or cached_grid:
                    raise RuntimeError(
                        "unverified target/grid caches in full-grid output; use a fresh --out"
                    )
                LC._write_json(manifest, identity)

    def target_path(self, m: str, s: str, layer: int) -> Path:
        return self.out / "targets" / f"{m}{s}_L{layer}.npz"

    def grid_path(self, m: str, s: str) -> Path:
        return self.out / "percell" / f"k5grid_{m}{s}_L{self.fit_layer}.npz"

    def anchor_path(self, s: str) -> Path:
        return self.out / "percell" / f"k1_anchor_{s}_L{self.fit_layer}.npz"

    def companion_path(self, m: str, s: str) -> Path:
        """Companion baseline and retrieval metrics for the full-grid run."""
        return self.out / "percell" / f"k5grid_{m}{s}_companions.json"

    def xfer_path(self, i: str, j: str, fold: int) -> Path:
        return self.out / "percell" / f"k5xfer_{i}{j}_f{fold}.npz"


def _reference_rows() -> tuple[np.ndarray, list[str]]:
    """The paper's committed IID fold assignment + row ids (n=16,391)."""
    return XF.load_fold_of()


# ── phase: stage ─────────────────────────────────────────────────────────────


def run_stage(cfg: Config) -> None:
    staged: list[str] = []
    if cfg.smoke:
        for m, s in cfg.cells:
            for seed in SMOKE_SEEDS:
                for layer in cfg.target_layers:
                    rel = _draw_answer_relpath(m, s, seed, layer)
                    _ensure(cfg, rel, revision=_draw_revision(seed, cfg))
                    staged.append(rel)
    else:
        for m, s in cfg.cells:
            for seed in K5_EXTRA_SEEDS:
                for layer in cfg.target_layers:
                    if cfg.target_path(m, s, layer).exists() and not cfg.force:
                        continue
                    rel = _draw_answer_relpath(m, s, seed, layer)
                    _ensure(cfg, rel, revision=_draw_revision(seed, cfg))
                    staged.append(rel)
                    _log(f"[stage] {m}{s} seed={seed} L{layer} resolved")
            # Seed-42 L18 shards exist on HF for the off-diagonal cells but were
            # never staged locally (only L31 was); fetch at the pinned revision.
            for layer in cfg.target_layers:
                if cfg.target_path(m, s, layer).exists() and not cfg.force:
                    continue
                rel = _draw_answer_relpath(m, s, SEED42, layer)
                if _resolve(cfg, rel) is None:
                    _ensure(cfg, rel, revision=LC.HF_REVISION)
                    staged.append(rel)
    manifest = {
        "timestamp_utc": _utcnow(),
        "smoke": cfg.smoke,
        "seeds": list(cfg.seeds),
        "cells": ["".join(c) for c in cfg.cells],
        "draw_revision": cfg.draw_revision,
        "target_layers": list(cfg.target_layers),
        "n_staged": len(staged),
        "staged": staged,
    }
    LC._write_json(cfg.out / "staged_manifest.json", manifest)
    _log(f"[stage] resolved {len(staged)} shards (smoke={cfg.smoke})")


# ── phase: targets ───────────────────────────────────────────────────────────


def _build_cell_targets(
    cfg: Config,
    m: str,
    s: str,
    layer: int,
    ref_ids: list[str],
    flag_maps: dict[int, dict[str, bool]],
    flag_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    draws: dict[int, tuple[np.ndarray, dict[str, int]]] = {}
    for seed in cfg.seeds:
        path = _resolve(cfg, _draw_answer_relpath(m, s, seed, layer))
        if path is None:
            raise FileNotFoundError(
                f"draw shard missing for cell ({m},{s}) seed {seed} L{layer} — "
                "run `stage` first (a missing k5 shard means the pod has not finished)"
            )
        w, ids = _load_answer(path)
        draws[seed] = (w, {rid: i for i, rid in enumerate(ids)})
    if cfg.smoke:
        rows = [rid for rid in ref_ids if all(rid in pos for _, pos in draws.values())]
    else:
        rows = list(ref_ids)
        for seed, (_, pos) in draws.items():
            missing = [rid for rid in rows if rid not in pos]
            if missing:
                raise RuntimeError(
                    f"cell ({m},{s}) seed {seed} L{layer}: {len(missing)} paper rows "
                    f"missing from the draw store (first: {missing[:3]})"
                )
    k = len(cfg.seeds)
    aligned: dict[int, np.ndarray] = {}
    for seed in cfg.seeds:
        w, pos = draws[seed]
        aligned[seed] = w[[pos[rid] for rid in rows]]  # float32 (n, d)
    del draws
    acc = np.zeros(aligned[cfg.seeds[0]].shape, dtype=np.float64)
    for seed in cfg.seeds:
        acc += aligned[seed]
    w_bar = acc / k
    del acc
    spread = np.zeros(len(rows), dtype=np.float64)
    for seed in cfg.seeds:
        diff = aligned[seed] - w_bar
        spread += np.square(diff).sum(axis=1)
        del diff
    spread /= k - 1
    del aligned
    n_flagged = np.zeros(len(rows), dtype=np.int64)
    if flag_counts is not None:
        n_flagged = np.asarray([flag_counts[rid] for rid in rows], dtype=np.int64)
    else:
        for seed in cfg.seeds:
            fm = flag_maps[seed]
            missing_flags = [rid for rid in rows if rid not in fm]
            if missing_flags:
                raise RuntimeError(
                    f"rollout flags missing for cell ({m},{s}) seed {seed}: "
                    f"{len(missing_flags)} rows (first: {missing_flags[:3]})"
                )
            n_flagged += np.asarray([fm[rid] for rid in rows], dtype=np.int64)
    LC._savez(
        cfg.target_path(m, s, layer),
        row_ids=np.asarray(rows),
        w_bar=w_bar.astype(np.float32),
        spread=spread,
        k_eff=np.full(len(rows), k, dtype=np.int64),
        n_flagged=n_flagged,
        seeds=np.asarray(cfg.seeds, dtype=np.int64),
    )
    return {"n_rows": len(rows), "k": k}


def _saved_flag_counts(cfg: Config, source: str, ref_ids: list[str]) -> dict[str, int]:
    """Reuse source-only counts from a Hub-verified prior K5 target artifact."""
    assert cfg.flag_counts_root is not None
    name = f"{source}{source}_L31.npz"
    path = cfg.flag_counts_root / name
    rel = f"{C.HF_PREFIX}/analysis_tensors/k5_targets/targets/{name}"
    _check_source_identity(cfg, path, rel)
    with np.load(path, allow_pickle=False) as payload:
        ids = [str(v) for v in payload["row_ids"]]
        counts = np.asarray(payload["n_flagged"])
        seeds = payload["seeds"].tolist()
    if ids != ref_ids or len(set(ids)) != len(ids) or seeds != list(cfg.seeds):
        raise RuntimeError(f"prior flag artifact row/seed mismatch: {path}")
    if counts.shape != (len(ids),) or not np.issubdtype(counts.dtype, np.integer):
        raise RuntimeError(f"prior flag artifact count shape/dtype mismatch: {path}")
    if np.any((counts < 0) | (counts > len(cfg.seeds))):
        raise RuntimeError(f"prior flag artifact count outside [0,K]: {path}")
    _log(f"[targets] reused Hub-verified K5 flag counts for {source}, rows={len(ids)}")
    return dict(zip(ids, counts.tolist(), strict=True))


def run_targets(cfg: Config) -> None:
    """Construct mean targets; flags depend only on answer source and draw seeds."""
    _, ref_ids = _reference_rows()
    pending = {
        (m, s, layer)
        for (m, s) in cfg.cells
        for layer in cfg.target_layers
        if cfg.force or not cfg.target_path(m, s, layer).exists()
    }
    flag_maps: dict[str, dict[int, dict[str, bool]]] = {}
    saved_counts: dict[str, dict[str, int]] = {}
    for src in sorted({s for _, s, _ in pending}):
        if cfg.flag_counts_root is not None:
            saved_counts[src] = _saved_flag_counts(cfg, src, ref_ids)
            flag_maps[src] = {}
        else:
            flag_maps[src] = {seed: _flag_map(cfg, src, seed) for seed in cfg.seeds}
    n_cells = 0
    for m, s in cfg.cells:
        for layer in cfg.target_layers:
            if (m, s, layer) not in pending:
                _log(f"[targets] {m}{s} L{layer}: resumed")
                continue
            stats = _build_cell_targets(
                cfg, m, s, layer, ref_ids, flag_maps[s], saved_counts.get(s)
            )
            n_cells += 1
            _log(f"[targets] {m}{s} L{layer}: rows={stats['n_rows']} k={stats['k']} written")
    _log(f"[targets] built {n_cells} cell-layer target files (counts only)")


def _load_targets(cfg: Config, m: str, s: str, layer: int | None = None) -> dict[str, np.ndarray]:
    layer = cfg.fit_layer if layer is None else layer
    path = cfg.target_path(m, s, layer)
    if not path.exists():
        raise FileNotFoundError(f"targets missing for cell ({m},{s}) L{layer} — run `targets`")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


# ── phase: grid (+ K=1 parity anchor) ────────────────────────────────────────


def _aligned_ctx(cfg: Config, stage: str, rows: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Ctx features + fold assignment for ``rows`` (paper order preserved)."""
    fold_of, ref_ids = _reference_rows()
    x, ids = _load_ctx(cfg, stage)
    if ids != ref_ids:
        raise RuntimeError(f"ctx row_ids differ from the paper reference for stage {stage}")
    pos = {rid: i for i, rid in enumerate(ids)}
    idx = np.asarray([pos[rid] for rid in rows], dtype=np.int64)
    x_rows = x[idx]
    if cfg.smoke:
        x_rows = x_rows[:, :SMOKE_FEATURE_DIM]
    return x_rows, fold_of[idx]


def run_grid(cfg: Config) -> None:
    """Fit all selected cells with one shared decomposition per checkpoint/fold."""
    anchor_ref = _anchor_reference(cfg) if not cfg.smoke else {}
    by_ckpt: dict[str, list[str]] = {}
    for m, s in cfg.cells:
        by_ckpt.setdefault(m, []).append(s)
    for m, targets_srcs in by_ckpt.items():
        done = all(cfg.grid_path(m, s).exists() for s in targets_srcs) and (
            not _k1_anchor_layer(cfg)
            or all(cfg.anchor_path(s).exists() for s in targets_srcs if s == m)
        )
        if cfg.full_grid:
            done = done and all(cfg.companion_path(m, s).exists() for s in targets_srcs)
        if done and not cfg.force:
            _log(f"[grid] checkpoint {m}: resumed")
            continue
        cell_targets = {s: _load_targets(cfg, m, s) for s in targets_srcs}
        rows_ref: list[str] | None = None
        for s, payload in cell_targets.items():
            rows = [str(v) for v in payload["row_ids"]]
            if rows_ref is None:
                rows_ref = rows
            elif rows != rows_ref:
                raise RuntimeError(f"target row sets differ across cells for ckpt {m}")
        assert rows_ref is not None
        x, fold_of = _aligned_ctx(cfg, m, rows_ref)
        n = len(rows_ref)
        anchors_needed = [s for s in targets_srcs if s == m and _k1_anchor_layer(cfg)]
        anchor_y: dict[str, np.ndarray] = {}
        for s in anchors_needed:
            path = _resolve(cfg, _draw_answer_relpath(m, s, SEED42, cfg.fit_layer))
            assert path is not None  # targets already resolved this shard
            w42, ids42 = _load_answer(path)
            pos = {rid: i for i, rid in enumerate(ids42)}
            anchor_y[s] = w42[[pos[r] for r in rows_ref]]  # float32, parent dtype
        acc = {
            key: {
                "res": np.full(n, np.nan),
                "tot": np.full(n, np.nan),
                "cos": np.full(n, np.nan),
                "lam": np.full(N_FOLDS, np.nan),
                "dof": np.full(N_FOLDS, np.nan),
                "n_tr": np.zeros(N_FOLDS, dtype=np.int64),
                "n_ev": np.zeros(N_FOLDS, dtype=np.int64),
            }
            for key in [f"k5_{s}" for s in targets_srcs] + [f"k1_{s}" for s in anchors_needed]
        }
        companions = {s: {"identity_bias_ss_res": 0.0, "folds": []} for s in targets_srcs}
        for fold in range(N_FOLDS):
            t0 = time.time()
            ev = fold_of == fold
            tr = ~ev
            ridge = XF.SharedPrimalRidge(x[tr])
            xev_std = ridge.standardize(x[ev])
            fits: list[tuple[str, np.ndarray]] = [
                (f"k5_{s}", cell_targets[s]["w_bar"]) for s in targets_srcs
            ] + [(f"k1_{s}", anchor_y[s]) for s in anchors_needed]
            for key, y in fits:
                weights, ymu, info = ridge.fit(y[tr])
                pred = xev_std @ weights + ymu
                rr, tt, cc = LC._per_row_components(pred, y[ev], y[tr].mean(axis=0))
                a = acc[key]
                a["res"][ev], a["tot"][ev], a["cos"][ev] = rr, tt, cc
                a["lam"][fold] = info["selected_lambda"]
                a["dof"][fold] = info["dof"]
                a["n_tr"][fold], a["n_ev"][fold] = int(tr.sum()), int(ev.sum())
                if cfg.full_grid and key.startswith("k5_"):
                    source = key.removeprefix("k5_")
                    baseline = identity_bias_predict(x[tr], y[tr], x[ev])
                    br, _, _ = LC._per_row_components(baseline, y[ev], y[tr].mean(axis=0))
                    companions[source]["identity_bias_ss_res"] += float(br.sum())
                    companions[source]["folds"].append(
                        {
                            "fold": fold,
                            "euclidean": knn_retrieval(pred, y[ev], metric="euclidean"),
                            "cosine": knn_retrieval(pred, y[ev], metric="cosine"),
                        }
                    )
                    del baseline
                del weights, pred
            del ridge, xev_std
            _log(
                f"[grid] ckpt {m} fold {fold}: {len(fits)} fits "
                f"n_tr={int(tr.sum())} n_ev={int(ev.sum())} in {time.time() - t0:.1f}s"
            )
        for s in targets_srcs:
            a = acc[f"k5_{s}"]
            if not np.all(np.isfinite(a["res"])):
                raise RuntimeError(f"non-finite OOF components for K=5 cell ({m},{s})")
            LC._savez(
                cfg.grid_path(m, s),
                row_ids=np.asarray(rows_ref),
                fold_of=fold_of,
                ss_res=a["res"],
                ss_tot=a["tot"],
                cos=a["cos"],
                selected_lambda=a["lam"],
                dof=a["dof"],
                n_train=a["n_tr"],
                n_eval=a["n_ev"],
            )
            if not cfg.smoke:
                r2 = 1.0 - float(a["res"].sum()) / float(a["tot"].sum())
                _log(f"[grid] K=5 cell ({m},{s}) pooled R2={r2:.6f}")
            if cfg.full_grid:
                companion = companions[s]
                companion["identity_bias_r2"] = 1.0 - companion["identity_bias_ss_res"] / float(
                    a["tot"].sum()
                )
                for metric in ("euclidean", "cosine"):
                    companion[metric] = {
                        "acc_at_k": {
                            k: sum(
                                f[metric]["n"] * f[metric]["acc_at_k"][k]
                                for f in companion["folds"]
                            )
                            / n
                            for k in (1, 5, 10)
                        },
                        "pool_sizes": [f[metric]["n_pool"] for f in companion["folds"]],
                        "chance_at_k": {k: N_FOLDS * k / n for k in (1, 5, 10)},
                    }
                LC._write_json(cfg.companion_path(m, s), companion)
        for s in anchors_needed:
            a = acc[f"k1_{s}"]
            LC._savez(
                cfg.anchor_path(s),
                row_ids=np.asarray(rows_ref),
                fold_of=fold_of,
                ss_res=a["res"],
                ss_tot=a["tot"],
                selected_lambda=a["lam"],
                dof=a["dof"],
            )
    if cfg.smoke:
        _log("[grid] smoke fits complete — shape-only, no R2 reported")
        return
    _parity_anchor_gate(cfg, anchor_ref)


def _k1_anchor_layer(cfg: Config) -> bool:
    """True when the K=1 parity anchor is available for this run's fit layer.

    The committed K=1 diagonal (``lasttoken_transfer/summary.json``) exists at
    layer 31 only, so an off-layer run refits no K=1 anchor and gates on the
    committed same-layer K=5 diagonal instead (``_parity_anchor_gate``).
    """
    return not cfg.smoke and cfg.fit_layer == LAYER


def _anchor_reference(cfg: Config) -> dict[str, float]:
    """Committed diagonal pooled R^2 this run must reproduce at its fit layer.

    Layer 31: the K=1 diagonal from ``lasttoken_transfer/summary.json`` (the
    parent pipeline's own numbers, compared against this run's K=1 refits).
    Off layer: the committed K=5 diagonal at the SAME layer, compared against
    this run's K=5 diagonal cells - same estimator, folds, target and layer.
    """
    if _k1_anchor_layer(cfg):
        summary = json.loads((XF.DEFAULT_OUT / "summary.json").read_text())
        return {s: float(summary["grid"][f"{s}{s}"]["r2"]) for s in STAGES}
    ref_path = K5_OFFLAYER_REFERENCE[cfg.fit_layer]
    summary = json.loads(ref_path.read_text())
    if int(summary["layer"]) != cfg.fit_layer:
        raise RuntimeError(f"{ref_path} is layer {summary['layer']}, not {cfg.fit_layer}")
    return {s: float(summary["cells"][s]["r2"]) for s in STAGES}


def _parity_anchor_gate(cfg: Config, anchor_ref: dict[str, float]) -> None:
    k1 = _k1_anchor_layer(cfg)
    rows = []
    for s in STAGES:
        path = cfg.anchor_path(s) if k1 else cfg.grid_path(s, s)
        with np.load(path, allow_pickle=False) as payload:
            got = 1.0 - float(payload["ss_res"].sum()) / float(payload["ss_tot"].sum())
        rows.append(
            {
                "stage": s,
                ("k1_rerun_r2" if k1 else "k5_rerun_r2"): got,
                "committed_r2": anchor_ref[s],
                "abs_diff": abs(got - anchor_ref[s]),
            }
        )
    report = {
        "layer": cfg.fit_layer,
        "mode": "k1_anchor" if k1 else "k5_diagonal_same_layer",
        "tolerance": ANCHOR_TOL,
        "max_abs_diff": max(r["abs_diff"] for r in rows),
        "pass": all(r["abs_diff"] <= ANCHOR_TOL for r in rows),
        "reference": (
            "eval_results/issue_1902/lasttoken_transfer/summary.json grid diagonal"
            if k1
            else str(K5_OFFLAYER_REFERENCE[cfg.fit_layer].relative_to(PROJECT_ROOT))
        ),
        "cells": rows,
    }
    LC._write_json(cfg.out / "k5_parity_anchor.json", report)
    if not report["pass"]:
        raise RuntimeError(
            f"parity anchor FAILED at L{cfg.fit_layer} "
            f"(mode={report['mode']}; STOP — do not interpret): {rows}"
        )
    _log(f"[anchor] PASS max_abs_diff={report['max_abs_diff']:.3e}")


# ── bootstrap helpers (run_analyze conventions, seed 1944) ───────────────────


def _bootstrap_counts(cfg: Config, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, clusters = XF.load_clusters(cfg.ro_root)
    _, ref_ids = _reference_rows()
    if ids != ref_ids:
        raise RuntimeError("row_index.jsonl ids differ from the paper reference")
    if n != len(ref_ids):
        raise RuntimeError(f"bootstrap expects the full paper row set, got n={n}")
    uniq, inverse = np.unique(clusters, return_inverse=True)
    n_cl = len(uniq)
    rng_row = np.random.default_rng(BOOT_SEED)
    idx = rng_row.integers(0, n, size=(N_BOOT, n))
    row_counts = np.zeros((N_BOOT, n), dtype=np.float64)
    for b in range(N_BOOT):
        row_counts[b] = np.bincount(idx[b], minlength=n)
    del idx
    rng_cl = np.random.default_rng(BOOT_SEED)
    cidx = rng_cl.integers(0, n_cl, size=(N_BOOT, n_cl))
    cl_counts = np.zeros((N_BOOT, n_cl), dtype=np.float64)
    for b in range(N_BOOT):
        cl_counts[b] = np.bincount(cidx[b], minlength=n_cl)
    del cidx
    return row_counts, cl_counts, inverse


def _r2_draws(
    res: np.ndarray,
    tot: np.ndarray,
    mode: str,
    row_counts: np.ndarray,
    cl_counts: np.ndarray,
    inverse: np.ndarray,
) -> np.ndarray:
    if mode == "row":
        return 1.0 - (row_counts @ res) / (row_counts @ tot)
    n_cl = cl_counts.shape[1]
    per_res = np.zeros(n_cl, dtype=np.float64)
    per_tot = np.zeros(n_cl, dtype=np.float64)
    np.add.at(per_res, inverse, np.asarray(res, dtype=np.float64))
    np.add.at(per_tot, inverse, np.asarray(tot, dtype=np.float64))
    return 1.0 - (cl_counts @ per_res) / (cl_counts @ per_tot)


def _ci(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


# ── phase: transfer ──────────────────────────────────────────────────────────


def run_transfer(cfg: Config) -> None:
    if cfg.smoke:
        raise SystemExit("transfer is not part of the smoke (n_train < d regime)")
    gate = cfg.out / "k5_parity_anchor.json"
    if not gate.exists() or not json.loads(gate.read_text())["pass"]:
        raise RuntimeError("parity anchor missing or failed — run `grid` first")
    fold_of, ref_ids = _reference_rows()
    diag = {s: _load_targets(cfg, s, s) for s in STAGES}
    for s, payload in diag.items():
        if [str(v) for v in payload["row_ids"]] != ref_ids:
            raise RuntimeError(f"K=5 target rows differ from the paper reference ({s})")
    u: dict[str, np.ndarray] = {}
    for s in STAGES:
        u[s], _ = _aligned_ctx(cfg, s, ref_ids)
    w = {s: diag[s]["w_bar"] for s in STAGES}  # float32, parent dtype
    for fold in range(N_FOLDS):
        todo = [
            (i, j)
            for (i, j) in TRANSFER_PAIRS_K5
            if cfg.force or not cfg.xfer_path(i, j, fold).exists()
        ]
        if not todo:
            _log(f"[xfer] fold {fold}: resumed")
            continue
        ev = fold_of == fold
        tr = ~ev
        for i, j in todo:
            t0 = time.time()
            out = XF.transfer_pair_fold(u[i], u[j], w[i], w[j], tr, ev)
            info = out["info"]
            LC._savez(
                cfg.xfer_path(i, j, fold),
                row_idx=np.flatnonzero(ev),
                ss_res_direct=out["res_direct"],
                ss_res_bias=out["res_bias"],
                ss_res_scale_bias=out["res_scale_bias"],
                ss_tot=out["tot"],
                n_tr=np.int64(info["n_tr"]),
                n_ev=np.int64(info["n_ev"]),
                alpha=np.float64(info["alpha"]),
                lambda_f=np.float64(info["lambda_f"]),
                dof_f=np.float64(info["dof_f"]),
            )
            _log(
                f"[xfer] {i}->{j} fold {fold}: alpha={info['alpha']:.4f} in {time.time() - t0:.0f}s"
            )
    _write_summary(cfg)


def _write_summary(cfg: Config) -> None:
    """Aggregate OOF components and paired uncertainty at the selected scope."""
    fold_of, ref_ids = _reference_rows()
    n = len(ref_ids)
    row_counts, cl_counts, inverse = _bootstrap_counts(cfg, n)

    grid: dict[str, Any] = {}
    diag_components: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for m, s in cfg.cells:
        with np.load(cfg.grid_path(m, s), allow_pickle=False) as payload:
            res = np.asarray(payload["ss_res"], dtype=np.float64)
            tot = np.asarray(payload["ss_tot"], dtype=np.float64)
            lam = payload["selected_lambda"].tolist()
            dof = payload["dof"].tolist()
        grid[f"{m}{s}"] = {
            "activation_checkpoint": m,
            "answer_source": s,
            "r2": 1.0 - float(res.sum()) / float(tot.sum()),
            "row_ci": _ci(_r2_draws(res, tot, "row", row_counts, cl_counts, inverse)),
            "cluster_ci": _ci(_r2_draws(res, tot, "cluster", row_counts, cl_counts, inverse)),
            "selected_lambda": lam,
            "dof": dof,
            "n": n,
        }
        if cfg.full_grid:
            grid[f"{m}{s}"]["companions"] = json.loads(cfg.companion_path(m, s).read_text())
        if m == s:
            diag_components[s] = (res, tot)

    transfer: dict[str, Any] = {}
    for i, j in TRANSFER_PAIRS_K5:
        res = {mode: np.full(n, np.nan) for mode in XF.TRANSFER_MODES}
        tot = np.full(n, np.nan)
        alphas: list[float] = []
        for fold in range(N_FOLDS):
            with np.load(cfg.xfer_path(i, j, fold), allow_pickle=False) as payload:
                rows = payload["row_idx"]
                for mode in XF.TRANSFER_MODES:
                    res[mode][rows] = payload[f"ss_res_{mode}"]
                tot[rows] = payload["ss_tot"]
                alphas.append(float(payload["alpha"]))
        for mode in XF.TRANSFER_MODES:
            if not np.all(np.isfinite(res[mode])):
                raise RuntimeError(f"incomplete {mode} components for {i}->{j}")
        res_jj, tot_jj = diag_components[j]
        r2_jj = 1.0 - float(res_jj.sum()) / float(tot_jj.sum())
        r2 = {mode: 1.0 - float(res[mode].sum()) / float(tot.sum()) for mode in XF.TRANSFER_MODES}
        retention: dict[str, Any] = {}
        for mode in XF.TRANSFER_MODES:
            entry: dict[str, Any] = {"point": r2[mode] / r2_jj}
            for boot_mode in ("row", "cluster"):
                num = _r2_draws(res[mode], tot, boot_mode, row_counts, cl_counts, inverse)
                den = _r2_draws(res_jj, tot_jj, boot_mode, row_counts, cl_counts, inverse)
                rho = num / den
                finite = rho[np.isfinite(rho)]
                entry[f"{boot_mode}_ci"] = _ci(finite)
                entry[f"{boot_mode}_n_finite"] = int(finite.size)
            retention[mode] = entry
        transfer[f"{i}->{j}"] = {
            "r2": r2,
            "r2_jj": r2_jj,
            "retention": retention,
            "alpha_by_fold": alphas,
        }
        _log(
            f"[analyze] {i}->{j}: direct={r2['direct']:.4f} "
            f"scale_bias={r2['scale_bias']:.4f} r2_jj={r2_jj:.4f}"
        )

    summary = {
        "metadata": {
            "hf_repo": LC.HF_REPO,
            "hf_revision_seed42": LC.HF_REVISION,
            "k5_draw_revision": cfg.draw_revision or "main (post-pin pod uploads)",
            "full_grid": cfg.full_grid,
            "flag_counts_root": str(cfg.flag_counts_root) if cfg.flag_counts_root else None,
            "cells": ["".join(c) for c in cfg.cells],
            "layer": cfg.fit_layer,
            "corpus": CORPUS,
            "context_summary": "u_last",
            "target": "mean answer vector over K=5 draws (seeds 42,45,46,47,48)",
            "n_rows": n,
            "n_folds": N_FOLDS,
            "fold_mode": "random",
            "fold_seed": LC.RANDOM_FOLD_SEED,
            "transfer_modes": list(XF.TRANSFER_MODES),
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "script": "scripts/issue1902_k5_fits.py",
            "timestamp_utc": _utcnow(),
        },
        "parity_anchor": json.loads((cfg.out / "k5_parity_anchor.json").read_text()),
        "grid": grid,
        "transfer": transfer,
    }
    LC._write_json(cfg.out / "summary.json", summary)
    if cfg.full_grid:
        matrix = {m: {s: grid[m + s]["r2"] for s in STAGES} for m in STAGES}
        rows = {}
        for m, values in matrix.items():
            best = max(STAGES, key=lambda s: values[s])
            rows[m] = {
                "diagonal_answer_source": m,
                "diagonal_r2": values[m],
                "best_answer_source": best,
                "best_r2": values[best],
                "best_minus_diagonal": values[best] - values[m],
                "range_across_answer_sources": max(values.values()) - min(values.values()),
            }
        LC._write_json(
            cfg.out / "cross_answer_source_diagnostic.json",
            {
                "protocol": summary["metadata"],
                "interpretation": "Fixed representation checkpoint; vary generated answer-text source.",
                "r2_context_stage_by_answer_source": matrix,
                "by_context_stage": rows,
                "selection_note": "Best-source labels are descriptive maxima, without selection-adjusted inference.",
            },
        )
    _log(f"[analyze] wrote {cfg.out / 'summary.json'}")


# ── phase: retrieval ─────────────────────────────────────────────────────────


def run_retrieval(cfg: Config) -> None:
    if cfg.smoke:
        raise SystemExit("retrieval is not part of the smoke (n_train < d regime)")
    fold_of, ref_ids = _reference_rows()
    _, groups = LC._read_groups(cfg.ro_root, CORPUS)
    names = sorted(set(groups))
    rng = np.random.default_rng(RT.BOOT_SEED)
    counts = rng.multinomial(
        len(names), np.full(len(names), 1.0 / len(names)), size=RT.N_BOOT
    ).astype(np.float64)
    cells: dict[str, Any] = {}
    for stage in STAGES:
        payload = _load_targets(cfg, stage, stage)
        if [str(v) for v in payload["row_ids"]] != ref_ids:
            raise RuntimeError(f"K=5 target rows differ from the paper reference ({stage})")
        y = payload["w_bar"]  # float32, parent dtype
        x, _ = _aligned_ctx(cfg, stage, ref_ids)
        ranks = np.full(len(ref_ids), np.nan)
        raw_ranks = np.full(len(ref_ids), np.nan)
        for fold in range(N_FOLDS):
            out_path = cfg.out / "retrieval" / "perfold" / f"{stage}_f{fold}.npz"
            ev = fold_of == fold
            tr = ~ev
            if out_path.exists() and not cfg.force:
                with np.load(out_path, allow_pickle=False) as saved:
                    ranks[saved["eval_index"]] = saved["whitened_csls_rank"]
                    raw_ranks[saved["eval_index"]] = saved["raw_cosine_rank"]
                continue
            t0 = time.time()
            ridge = XF.SharedPrimalRidge(x[tr])
            pred, _info = ridge.fit_predict(y[tr], x[ev])
            raw = RT.cosine_ranks(pred, y[ev])
            mean, chol = RT.shrunk_whitening_stats(y[tr])
            whit = RT.whitened_csls_ranks(pred, y[ev], mean, chol)
            LC._savez(
                out_path,
                eval_index=np.flatnonzero(ev),
                raw_cosine_rank=raw,
                whitened_csls_rank=whit,
            )
            ranks[ev], raw_ranks[ev] = whit, raw
            _log(
                f"[retrieval] {stage} fold {fold}: n_pool={int(ev.sum())} in {time.time() - t0:.0f}s"
            )
        if not np.all(np.isfinite(ranks)):
            raise RuntimeError(f"incomplete ranks for stage {stage}")
        point, interval = RT.cluster_bootstrap_acc1(ranks <= 1.0, groups, counts=counts)
        raw_point, raw_interval = RT.cluster_bootstrap_acc1(raw_ranks <= 1.0, groups, counts=counts)
        cells[stage] = {
            "n": len(ref_ids),
            "whitened_csls_acc1": point,
            "whitened_csls_acc1_ci95": interval,
            "raw_cosine_acc1": raw_point,
            "raw_cosine_acc1_ci95": raw_interval,
        }
        _log(f"[retrieval] {stage}: acc1={point:.4f}")
    report = {
        "metadata": {
            "design": "K=5 mean-target strict retrieval (whitened cosine + CSLS k=10)",
            "whiten_lambda": RT.WHITEN_LAMBDA,
            "csls_k": RT.CSLS_K,
            "n_boot": RT.N_BOOT,
            "boot_seed": RT.BOOT_SEED,
            "n_groups": len(names),
            "script": "scripts/issue1902_k5_fits.py",
            "timestamp_utc": _utcnow(),
        },
        "cells": cells,
    }
    LC._write_json(cfg.out / "retrieval" / "summary.json", report)


# ── phase: scatter ───────────────────────────────────────────────────────────


def run_scatter(cfg: Config) -> None:
    from scipy.stats import spearmanr

    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.c2a_plot_style import (
        INK,
        c2a_figure,
        save_c2a_figure,
        set_c2a_style,
        style_axis,
    )

    set_c2a_style()
    per_stage: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in STAGES:
        tgt = _load_targets(cfg, s, s)
        with np.load(cfg.grid_path(s, s), allow_pickle=False) as payload:
            grid_rows = [str(v) for v in payload["row_ids"]]
            err = np.asarray(payload["ss_res"], dtype=np.float64)
        if [str(v) for v in tgt["row_ids"]] != grid_rows:
            raise RuntimeError(f"targets/grid row mismatch for stage {s}")
        per_stage[s] = (np.asarray(tgt["spread"], dtype=np.float64), err)

    pooled_x = np.concatenate([per_stage[s][0] for s in STAGES])
    pooled_y = np.concatenate([per_stage[s][1] for s in STAGES])
    pos = (pooled_x > 0) & (pooled_y > 0)
    log_axes = bool(
        pos.all()
        and pooled_x.max() / pooled_x.min() > 100
        and pooled_y.max() / pooled_y.min() > 100
    )

    rho: dict[str, Any] = {}
    for s in STAGES:
        r, p = spearmanr(per_stage[s][0], per_stage[s][1])
        rho[STAGE_NAMES[s]] = {"rho": float(r), "p": float(p), "n": int(len(per_stage[s][0]))}
    r, p = spearmanr(pooled_x, pooled_y)
    rho["pooled"] = {"rho": float(r), "p": float(p), "n": int(len(pooled_x))}

    # Pooled binned-median trend (equal-count bins on the x scale in use).
    order = np.argsort(pooled_x)
    n_bins = 20 if len(pooled_x) >= 2000 else max(4, len(pooled_x) // 50)
    bins = np.array_split(order, n_bins)
    trend_x = np.asarray([np.median(pooled_x[b]) for b in bins if len(b) >= 20])
    trend_y = np.asarray([np.median(pooled_y[b]) for b in bins if len(b) >= 20])

    fig, include_frac = c2a_figure("wide", aspect=0.75)
    ax = fig.add_axes([0.13, 0.14, 0.84, 0.78])
    for s in STAGES:
        sx, sy = per_stage[s]
        ax.scatter(
            sx,
            sy,
            s=5,
            alpha=0.25,
            linewidths=0,
            color=STAGE_COLORS[s],
            marker=STAGE_MARKERS[s],
            label=STAGE_NAMES[s],
            rasterized=True,
        )
    ax.plot(trend_x, trend_y, color=INK, linewidth=2.2, zorder=6)
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    style_axis(ax)
    ax.set_xlabel("Within-context draw spread")
    ax.set_ylabel("Held-out squared error vs 5-draw mean")
    legend = ax.legend(frameon=False, markerscale=3.0, handletextpad=0.4, loc="upper left")
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    stem = cfg.fig_dir / "c1_draw_spread_vs_error_k5"
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Draw spread vs own-map squared error (K=5 targets)",
        subject="Issue #1902 K=5 follow-up: per-context draw spread vs OOF error",
        creator="scripts/issue1902_k5_fits.py",
        include_width=include_frac,
    )
    plt.close(fig)
    sidecar = {
        "metadata": {
            "x": "spread = sum_k ||w_k - w_bar||^2 / (K-1) per context",
            "y": "own-map OOF squared error against w_bar (six IID folds)",
            "stages_pooled": [STAGE_NAMES[s] for s in STAGES],
            "log_axes": log_axes,
            "trend": "pooled equal-count binned medians",
            "smoke": cfg.smoke,
            "script": "scripts/issue1902_k5_fits.py",
            "timestamp_utc": _utcnow(),
        },
        "spearman": rho,
        "trend_x": trend_x.tolist(),
        "trend_y": trend_y.tolist(),
    }
    LC._write_json(stem.with_name(stem.name + "_data.json"), sidecar)
    _log(
        f"[scatter] wrote {outputs['pdf']} (points={len(pooled_x)}, "
        f"stages={len(STAGES)}, log_axes={log_axes})"
    )


# ── phase: figure ────────────────────────────────────────────────────────────


def run_figure(cfg: Config) -> None:
    """Render the existing paper panels from this run's complete or seven-cell grid."""
    if cfg.smoke:
        raise SystemExit("figure is not part of the smoke")
    import section43_posttraining_figure as FIG

    summary = json.loads((cfg.out / "summary.json").read_text())
    retrieval = json.loads((cfg.out / "retrieval" / "summary.json").read_text())
    grid = summary["grid"]
    diag_r2 = [grid[f"{s}{s}"]["r2"] for s in STAGES]
    data = {
        "metadata": {
            "variant": "K=5 mean answer targets (seeds 42,45,46,47,48)",
            "k5_summary": str((cfg.out / "summary.json").relative_to(PROJECT_ROOT)),
            "k5_retrieval": str((cfg.out / "retrieval" / "summary.json").relative_to(PROJECT_ROOT)),
            "layer": cfg.fit_layer,
            "context_summary": "u_last",
            "folds": "six size-matched IID random-row folds (seed 190231)",
            "panel_bc_bootstrap": (
                "1000 paired draws, seed 1944; panel C retention CIs = "
                "semantic-cluster bootstrap; panel B CIs = row bootstrap"
            ),
            "prev_row_note": (
                "Full 4x4 grid: all three previous-stage points captured."
                if cfg.full_grid
                else "K=5 captures cover the diagonal + base row only, so the "
                "previous-stage series has its SFT point (cell BS) alone"
            ),
            "generated_utc": _utcnow(),
        },
        "iid_r2": diag_r2,
        "iid_r2_ci": [grid[f"{s}{s}"]["cluster_ci"] for s in STAGES],
        "whitened_csls_acc1": [retrieval["cells"][s]["whitened_csls_acc1"] for s in STAGES],
        "whitened_csls_acc1_ci": [retrieval["cells"][s]["whitened_csls_acc1_ci95"] for s in STAGES],
        "diag_r2": diag_r2,
        "diag_ci": [grid[f"{s}{s}"]["row_ci"] for s in STAGES],
        "base_row_r2": [grid[f"B{s}"]["r2"] for s in STAGES],
        "base_row_ci": [grid[f"B{s}"]["row_ci"] for s in STAGES],
        "prev_row_r2": [None]
        + [grid[i + j]["r2"] if i + j in grid else None for i, j in TRANSFER_PAIRS_K5],
        "prev_row_ci": [None]
        + [grid[i + j]["row_ci"] if i + j in grid else None for i, j in TRANSFER_PAIRS_K5],
    }
    if cfg.full_grid:
        data["stage_grid"] = [[grid[m + s]["r2"] for s in STAGES] for m in STAGES]
    for mode in XF.TRANSFER_MODES:
        points, ci, ci_row = [], [], []
        for i, j in TRANSFER_PAIRS_K5:
            entry = summary["transfer"][f"{i}->{j}"]["retention"][mode]
            points.append(entry["point"])
            ci.append(entry["cluster_ci"])
            ci_row.append(entry["row_ci"])
        data[f"{mode}_retention"] = points
        data[f"{mode}_retention_ci"] = ci
        data[f"{mode}_retention_ci_row"] = ci_row

    FIG.set_c2a_style()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "c1_posttraining_dynamics_k5" + ("" if cfg.fit_layer == LAYER else f"_L{cfg.fit_layer}")
    data_path = cfg.fig_dir / f"{stem}_data.json"
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    data = json.loads(data_path.read_text())
    outputs = FIG.render_variant(data, cfg.fig_dir / stem, panel_b="lines")
    if cfg.full_grid:
        FIG.render_variant(data, cfg.fig_dir / f"{stem}_grid", panel_b="grid")
    _log(f"[figure] wrote {outputs['pdf']}")
    _log(f"[figure] data: {data_path}")


# ── main ─────────────────────────────────────────────────────────────────────

PHASES = ("stage", "targets", "grid", "transfer", "retrieval", "scatter", "figure")
SMOKE_PHASES = ("stage", "targets", "grid", "scatter")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=[*PHASES, "all"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_RO_STAGE_ROOT)
    parser.add_argument("--k5-root", type=Path, default=DEFAULT_K5_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--full-grid", action="store_true", help="Use all 16 captured K=5 cells")
    parser.add_argument("--draw-revision", help="Immutable Hub revision for seeds 45-48")
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--flag-counts-root",
        type=Path,
        help="Prior K5 targets; verify Hub bytes, row IDs, and seeds before reusing counts",
    )
    parser.add_argument(
        "--target-layers", type=int, nargs="+", choices=TARGET_LAYERS, default=list(TARGET_LAYERS)
    )
    parser.add_argument(
        "--fit-layer",
        type=int,
        choices=TARGET_LAYERS,
        default=LAYER,
        help="layer the grid/transfer/retrieval/figure phases fit (default 31)",
    )
    args = parser.parse_args()
    cfg = Config(args)
    runners = {
        "stage": run_stage,
        "targets": run_targets,
        "grid": run_grid,
        "transfer": run_transfer,
        "retrieval": run_retrieval,
        "scatter": run_scatter,
        "figure": run_figure,
    }
    phases = list(SMOKE_PHASES if cfg.smoke else PHASES) if args.cmd == "all" else [args.cmd]
    for phase in phases:
        _log(f"=== phase {phase} (smoke={cfg.smoke}) ===")
        runners[phase](cfg)
    _log("DONE")


if __name__ == "__main__":
    main()
