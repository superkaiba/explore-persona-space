"""#1092 summary-store + #779 r_B bank IO for issue #1739 (round A).

Loading conventions mirror ``scripts/issue1092_fit_grid.py`` (shard naming
``{kind}_L{layer:02d}[_shardNN].npy``, ``row_index[_shardNN].jsonl`` sidecars,
``is_eval_only`` fit-pool exclusion). Staging follows the scoped-listing +
per-file atomic download recipe (``hub.list_hf_files_under_path`` +
``hub.stage_hub_file``) — NEVER ``snapshot_download`` against the ~1M-file
data repo (gotchas.md #833).
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_1739.constants import (
    HF_DATA_REPO,
    HIDDEN_DIM,
    N_LAYERS,
    RB_PREFIX,
    RB_REVISION,
    STORE_FIT_ROWS,
    STORE_PREFIX,
    STORE_REVISION,
    STORE_TOTAL_ROWS,
)
from explore_persona_space.orchestrate import hub

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file via text-mode line iteration (NEVER .splitlines():
    real-corpus text carries raw U+2028/NEL — gotchas.md #825/#950)."""
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sorted_shards(paths: list[Path]) -> list[Path]:
    """Order shard files by shard index; fail loud on a duplicate index."""

    def key(path: Path) -> tuple[str, int]:
        stem = path.stem
        prefix, rest = stem.split("_shard", 1)
        digits = "".join(ch for ch in rest if ch.isdigit())
        return prefix, int(digits or 0)

    ordered = sorted(paths, key=key)
    seen: set[tuple[str, int]] = set()
    for path in ordered:
        k = key(path)
        if k in seen:
            raise ValueError(f"duplicate shard index for {path}")
        seen.add(k)
    return ordered


def _summary_shard_paths(root: Path, kind: str, layer: int) -> list[Path]:
    """Resolve one summary kind's shard set (sharded else unsharded); [] when absent."""
    paths = _sorted_shards(list(root.glob(f"{kind}_L{layer:02d}_shard*.npy")))
    if not paths:
        paths = sorted(root.glob(f"{kind}_L{layer:02d}.npy"))
    return paths


def _load_summary(root: Path, kind: str, layer: int) -> np.ndarray:
    paths = _summary_shard_paths(root, kind, layer)
    if not paths:
        raise FileNotFoundError(f"no summary shards for {root}/{kind}/L{layer:02d}")
    arrays = [np.load(p) for p in paths]
    return np.concatenate(arrays, axis=0)


def _index_rows(root: Path, stem: str = "row_index") -> list[dict]:
    paths = _sorted_shards(list(root.glob(f"{stem}_shard*.jsonl")))
    if not paths:
        path = root / f"{stem}.jsonl"
        paths = [path] if path.exists() else []
    if not paths:
        raise FileNotFoundError(f"missing index files {root}/{stem}[_shard*.jsonl]")
    rows: list[dict] = []
    for path in paths:
        rows.extend(_iter_jsonl(path))
    return rows


def _wanted_basename(name: str, kinds: tuple[str, ...], layers: tuple[int, ...]) -> bool:
    if name.startswith("row_index") and name.endswith(".jsonl"):
        return True
    for kind in kinds:
        for layer in layers:
            stem = f"{kind}_L{layer:02d}"
            if name.startswith(f"{stem}.") or name.startswith(f"{stem}_shard"):
                return True
    return False


def stage_store_slice(
    kinds: tuple[str, ...],
    layers: tuple[int, ...],
    n_rows: int | None,
    local_dir: Path | str,
    *,
    cell: str | None = None,
    revision: str = STORE_REVISION,
    max_workers: int = 6,
) -> list[Path]:
    """Stage the requested (kind x layer) shards + row_index sidecars from the
    #1092 store.

    Scoped ``list_hf_files_under_path`` (server-side tree walk, already
    retried) + per-file atomic ``hub.stage_hub_file`` in a bounded pool
    (max_workers<=6 — the org 2500-req/5-min quota). ``n_rows`` is a LOAD-time
    slice (shards download whole); it is recorded in the staging manifest.
    ``cell=None`` stages files sitting directly under the prefix; a cell name
    stages that cell subdir (Gate 0 reports the realized layout).
    """
    from huggingface_hub import HfApi

    prefix = STORE_PREFIX.rstrip("/") + (f"/{cell}" if cell else "")
    api = HfApi()
    files = hub.list_hf_files_under_path(
        api, HF_DATA_REPO, prefix, repo_type="dataset", revision=revision
    )
    wanted = [f for f in files if _wanted_basename(f.rsplit("/", 1)[-1], kinds, layers)]
    if not wanted:
        raise FileNotFoundError(
            f"no matching store files under {HF_DATA_REPO}@{revision}:{prefix} "
            f"for kinds={kinds} layers={layers}"
        )
    local_dir = Path(local_dir)
    root_prefix = STORE_PREFIX.rstrip("/") + "/"

    def _target(repo_path: str) -> Path:
        rel = repo_path[len(root_prefix) :] if repo_path.startswith(root_prefix) else repo_path
        return local_dir / rel

    def _stage(repo_path: str) -> Path:
        return hub.stage_hub_file(
            HF_DATA_REPO, repo_path, _target(repo_path), repo_type="dataset", revision=revision
        )

    with ThreadPoolExecutor(max_workers=min(max_workers, 6)) as pool:
        staged = list(pool.map(_stage, wanted))

    manifest = {
        "repo": HF_DATA_REPO,
        "revision": revision,
        "prefix": prefix,
        "kinds": list(kinds),
        "layers": list(layers),
        "n_rows_load_slice": n_rows,
        "n_files": len(staged),
        "files": sorted(str(p) for p in staged),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = local_dir / "staging_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("[store_io] staged %d files under %s (rev %s)", len(staged), local_dir, revision)
    return staged


def load_summaries(
    local_dir: Path | str,
    kinds: tuple[str, ...],
    layers: tuple[int, ...],
    *,
    cell: str | None = None,
    n_rows: int | None = None,
) -> tuple[dict[tuple[str, int], np.ndarray], list[dict]]:
    """Load staged summaries + row metadata.

    Returns ``({(kind, layer): (n, HIDDEN_DIM) array}, meta_rows)``; arrays keep
    the stored dtype (fp16). Asserts row-count consistency across kinds and vs
    the row_index sidecar.
    """
    root = Path(local_dir) / cell if cell else Path(local_dir)
    out: dict[tuple[str, int], np.ndarray] = {}
    for kind in kinds:
        for layer in layers:
            arr = _load_summary(root, kind, layer)
            assert arr.ndim == 2 and arr.shape[1] == HIDDEN_DIM, (kind, layer, arr.shape)
            out[(kind, layer)] = arr[:n_rows] if n_rows else arr
    meta = _index_rows(root)
    if n_rows:
        meta = meta[:n_rows]
    counts = {k: v.shape[0] for k, v in out.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"inconsistent row counts across (kind, layer): {counts}")
    n = next(iter(counts.values()))
    if len(meta) != n:
        raise ValueError(f"row_index count {len(meta)} != summary rows {n}")
    return out, meta


def fit_pool_mask(meta: list[dict]) -> np.ndarray:
    """Boolean FIT-pool mask over ``meta`` rows — the is_eval_only exclusion.

    A row is EVAL-ONLY when ``is_eval_only`` is truthy OR ``stratum ==
    "battery"`` (belt-and-suspenders, mirroring #1092's ``_is_battery_eval_row``
    — the durable key is ``is_eval_only``). Fails loud when zero fit rows
    survive, and pins the full-corpus count (18,793 of 21,193) when the slice
    is the whole realized manifest.
    """
    eval_only = np.array(
        [bool(r.get("is_eval_only")) or r.get("stratum") == "battery" for r in meta],
        dtype=bool,
    )
    mask = ~eval_only
    n, kept = len(meta), int(mask.sum())
    if kept < 1:
        raise ValueError("fit_pool_mask: zero fit rows after is_eval_only exclusion")
    if n == STORE_TOTAL_ROWS and kept != STORE_FIT_ROWS:
        raise ValueError(
            f"fit_pool_mask: full-corpus fit-pool count {kept} != pinned {STORE_FIT_ROWS}"
        )
    logger.info("[store_io] fit_pool_mask: kept %d / %d rows (%d eval-only)", kept, n, n - kept)
    return mask


def load_rb_bank(
    *,
    revision: str = RB_REVISION,
    local_dir: Path | str = Path("data/issue_1739/hf_dl/r_b"),
    n_layers: int = N_LAYERS,
    hidden_dim: int = HIDDEN_DIM,
) -> tuple[np.ndarray, list[str]]:
    """Load the #779 r_B trait-direction bank at the pinned revision.

    Mirrors ``issue1092_fit_grid._load_rb_directions``: scoped listing of
    ``*.pt`` files, per-file staged download, ``payload["r_b"]`` extraction,
    per-trait shape assert. Returns ``((n_layers, n_traits, hidden_dim)
    float64 array, trait_names)``.
    """
    import torch
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = RB_PREFIX.rstrip("/")
    relpaths = sorted(
        p
        for p in hub.list_hf_files_under_path(
            api, HF_DATA_REPO, prefix, repo_type="dataset", revision=revision
        )
        if p.endswith(".pt")
    )
    if not relpaths:
        raise FileNotFoundError(f"no r_B .pt files under {HF_DATA_REPO}@{revision}:{prefix}")
    local_dir = Path(local_dir)
    tensors: list[np.ndarray] = []
    names: list[str] = []
    for rel in relpaths:
        local = hub.stage_hub_file(
            HF_DATA_REPO,
            rel,
            local_dir / rel.rsplit("/", 1)[-1],
            repo_type="dataset",
            revision=revision,
        )
        payload = torch.load(local, map_location="cpu", weights_only=False)
        arr = payload["r_b"] if isinstance(payload, dict) and "r_b" in payload else payload
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (n_layers, hidden_dim):
            raise ValueError(f"r_B file {local} shape {arr.shape} != ({n_layers}, {hidden_dim})")
        tensors.append(arr)
        names.append(Path(local).stem)
    return np.stack(tensors, axis=1), names
