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
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_1739.constants import (
    CORPUS_MANIFEST_PATH,
    CORPUS_MANIFEST_REVISION,
    HF_DATA_REPO,
    HIDDEN_DIM,
    N_LAYERS,
    RB_PREFIX,
    RB_REVISION,
    STORE_FIT_ROWS,
    STORE_PREFIX,
    STORE_REVISION,
    STORE_TOTAL_ROWS,
    SUMMARY_KINDS,
    U_STORE_CELL,
)
from explore_persona_space.orchestrate import hub

logger = logging.getLogger(__name__)

# --- Realized #1092 layout mapping (concern u-store-staging-layout-unwired) ---
# The REALIZED store (gate0 + round-C2 realstore findings) diverges from the
# canonical round-B capture layout in two dir shapes:
#   - ``dynamics_*`` dirs: kind names ``context_k``/``answer_k_t1`` (etc.) and
#     PER-KIND row_index stems (``row_index_{kind}[_shard*].jsonl``);
#     ``bare_*`` dirs keep the bare ``row_index`` stem with kind ``c_q_bare``;
#   - main ``cell_*`` dirs: canonical kind names (prefix_end/context_end/t1)
#     but NO row_index files — #1092's own consumer (issue1092_fit_grid) reads
#     row metadata from the corpus ``manifest.jsonl`` instead.
# ``load_summaries`` maps BOTH onto the one canonical interface below (the
# artifact-reuse (h)(iv) staging-mapping fix — never a second divergent loader).
REALIZED_KIND_FOR = {
    "context_end": "context_k",  # context-end summary <-> #1092 dynamics context_k
    "t1": "answer_k_t1",  # answer-span mean <-> #1092 dynamics answer_k_t1
    "bare_query": "c_q_bare",  # bare-query summary <-> #1092 bare_* c_q_bare
}


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


def _summary_kind_candidates(kind: str) -> tuple[str, ...]:
    """On-disk name candidates for a requested kind: canonical first, then the
    realized #1092 alias (REALIZED_KIND_FOR)."""
    realized = REALIZED_KIND_FOR.get(kind)
    return (kind,) if realized is None else (kind, realized)


def _resolve_summary_kind(root: Path, kind: str, layer: int) -> tuple[str, list[Path]]:
    """Resolve a requested kind to its on-disk name + shard set; fail loud
    naming every candidate tried (canonical AND realized alias)."""
    tried: list[str] = []
    for cand in _summary_kind_candidates(kind):
        paths = _summary_shard_paths(root, cand, layer)
        if paths:
            return cand, paths
        tried.append(cand)
    raise FileNotFoundError(
        f"no summary shards for {root}/L{layer:02d} under kind name(s) {tried} "
        f"(requested {kind!r}; realized-layout aliases: {REALIZED_KIND_FOR})"
    )


def _index_files_exist(root: Path, stem: str = "row_index") -> bool:
    """True when ``stem``-named index files exist (sharded or unsharded)."""
    return bool(list(root.glob(f"{stem}_shard*.jsonl"))) or (root / f"{stem}.jsonl").exists()


def _index_rows_for(root: Path, resolved_kinds: list[str]) -> list[dict]:
    """Row-metadata ladder mirroring #1092's own consumer conventions:

    1. canonical ``row_index[_shard*].jsonl`` — round-B capture stores AND the
       realized ``bare_*`` dirs (issue1092_fit_grid ``_load_bare_rows``);
    2. per-kind ``row_index_{kind}[_shard*].jsonl`` — the realized
       ``dynamics_*`` dirs (issue1092_fit_grid ``_compute_dynamics_reads``);
       counts must agree across the requested kinds, the FIRST kind's rows
       are returned as meta;
    3. ``manifest.jsonl`` in the store root — the realized ``cell_*`` dirs
       carry NO row_index files; #1092's own consumer reads the corpus
       manifest (issue1092_fit_grid ``_jsonl(corpus_dir / "manifest.jsonl")``),
       staged beside the flattened shards by ``stage_u_store``.
    """
    if _index_files_exist(root):
        return _index_rows(root)
    per_kind: dict[str, list[dict]] = {}
    stems_present = [rk for rk in resolved_kinds if _index_files_exist(root, f"row_index_{rk}")]
    if stems_present:
        missing = [rk for rk in resolved_kinds if rk not in stems_present]
        if missing:
            raise FileNotFoundError(
                f"per-kind row_index stems present under {root} but missing for {missing} "
                f"(found for {stems_present})"
            )
        for rk in resolved_kinds:
            per_kind[rk] = _index_rows(root, stem=f"row_index_{rk}")
        counts = {rk: len(rows) for rk, rows in per_kind.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"per-kind row_index counts disagree under {root}: {counts}")
        # Row-ID alignment across kinds where ids exist (round-1 review Minor:
        # a bare count match cannot prove the kinds index the SAME rows).
        first = resolved_kinds[0]
        id_key = next(
            (
                k
                for k in ("context_id", "row_id", "turn_identifier")
                if per_kind[first] and k in per_kind[first][0]
            ),
            None,
        )
        if id_key is not None:
            ref = [r.get(id_key) for r in per_kind[first]]
            for rk in resolved_kinds[1:]:
                other = [r.get(id_key) for r in per_kind[rk]]
                if other != ref:
                    bad = next(i for i, (a, b) in enumerate(zip(ref, other, strict=True)) if a != b)
                    raise ValueError(
                        f"per-kind row_index {id_key} misaligned under {root}: "
                        f"{first}[{bad}]={ref[bad]!r} vs {rk}[{bad}]={other[bad]!r}"
                    )
        return per_kind[resolved_kinds[0]]
    manifest = root / "manifest.jsonl"
    if manifest.exists():
        return _iter_jsonl(manifest)
    raise FileNotFoundError(
        f"no row metadata under {root}: tried row_index[_shard*].jsonl, per-kind stems "
        f"row_index_{{{','.join(resolved_kinds)}}}[_shard*].jsonl, manifest.jsonl"
    )


def _wanted_basename(name: str, kinds: tuple[str, ...], layers: tuple[int, ...]) -> bool:
    if name.startswith("row_index") and name.endswith(".jsonl"):
        return True
    for kind in kinds:
        for cand in _summary_kind_candidates(kind):
            for layer in layers:
                stem = f"{cand}_L{layer:02d}"
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


def _u_store_target(dest: Path | str, repo_path: str) -> Path:
    """Map a repo file under ``STORE_PREFIX/<cell>/`` to a FLATTENED local
    target (``dest/<basename>``) so ``dest`` itself is a loadable store root
    (the staging-mapping half of artifact-reuse check (h)(iv))."""
    return Path(dest) / repo_path.rsplit("/", 1)[-1]


def u_store_loadable(
    root: Path | str,
    kinds: tuple[str, ...],
    layers: tuple[int, ...],
    *,
    revision: str = STORE_REVISION,
) -> bool:
    """True when ``root`` already serves the requested (kind x layer) grid.

    Two accepted shapes: (a) a LOCAL capture store (canonical row_index files
    present — e.g. the tiny-real smoke stand-in `issue1739_capture` writes)
    with every requested shard resolvable; (b) a COMPLETED staged #1092 slice
    — ``staging_manifest.json`` attests a finished ``stage_u_store`` run at
    the same revision covering the requested regime (kinds + layers superset,
    no shard cap) with every listed file still present. A PARTIAL staging
    (shards on disk, no completion manifest) is NOT loadable — its arrays
    would fail ``load_summaries``' count asserts against the corpus manifest.
    """
    root = Path(root)
    if not root.is_dir():
        return False

    def _grid_resolves() -> bool:
        try:
            for kind in kinds:
                for layer in layers:
                    _resolve_summary_kind(root, kind, layer)
        except FileNotFoundError:
            return False
        return True

    if _index_files_exist(root):
        return _grid_resolves()
    record_path = root / "staging_manifest.json"
    if not record_path.exists():
        return False
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    regime_ok = (
        record.get("complete") is True
        and record.get("max_shards_per_kind") is None
        and record.get("revision") == revision
        and set(kinds) <= set(record.get("kinds", []))
        and {int(x) for x in layers} <= {int(x) for x in record.get("layers", [])}
    )
    if not regime_ok:
        return False
    files = record.get("files", [])
    if not files or not all(Path(f).exists() for f in files):
        return False
    return (root / "manifest.jsonl").exists() and _grid_resolves()


def stage_u_store(
    dest: Path | str = Path("data/issue_1739/hf_dl/u_store"),
    kinds: tuple[str, ...] = SUMMARY_KINDS,
    layers: tuple[int, ...] = tuple(range(N_LAYERS)),
    *,
    cell: str = U_STORE_CELL,
    revision: str = STORE_REVISION,
    manifest_path: str = CORPUS_MANIFEST_PATH,
    manifest_revision: str = CORPUS_MANIFEST_REVISION,
    max_shards_per_kind: int | None = None,
    max_workers: int = 6,
) -> Path:
    """Stage the #1092 U-pool into ``dest`` as a directly-loadable store root.

    Realized-layout mapping (the ``u-store-staging-layout-unwired`` fix): the
    ``cell`` dir's (kind x layer) shards are FLATTENED into ``dest`` (the
    realized cell_* dirs already carry canonical kind names) and the corpus
    ``manifest.jsonl`` is staged beside them as the row-metadata source (the
    cell_* dirs carry NO row_index files — #1092's own consumer reads the
    corpus manifest). Scoped server-side listing + per-file atomic downloads
    (``hub.stage_hub_file`` — idempotent per file, so a partial dest
    self-heals); ``staging_manifest.json`` is written LAST as the completion
    sentinel (pool-first, sidecar-last).

    Short-circuits WITHOUT network when ``dest`` is already loadable for the
    requested regime (``u_store_loadable``) — in particular a LOCAL capture
    store at ``dest`` (the tiny-real smoke stand-in) is left untouched.
    ``max_shards_per_kind`` caps staged shards per (kind, layer) for
    probe/smoke slices (gate 3); such a slice is deliberately NOT marked
    production-loadable (``complete: false``).
    """
    from huggingface_hub import HfApi

    dest = Path(dest)
    if max_shards_per_kind is None and u_store_loadable(dest, kinds, layers, revision=revision):
        logger.info("[store_io] u_store already loadable at %s — staging skipped", dest)
        return dest
    api = HfApi()
    prefix = STORE_PREFIX.rstrip("/") + f"/{cell}"
    files = hub.list_hf_files_under_path(
        api, HF_DATA_REPO, prefix, repo_type="dataset", revision=revision
    )
    wanted: list[str] = []
    for kind in kinds:
        for layer in layers:
            per = sorted(
                f
                for f in files
                if not f.rsplit("/", 1)[-1].startswith("row_index")
                and _wanted_basename(f.rsplit("/", 1)[-1], (kind,), (layer,))
            )
            if not per:
                raise FileNotFoundError(
                    f"no {kind}_L{layer:02d} shards under {HF_DATA_REPO}@{revision}:{prefix}"
                )
            wanted.extend(per if max_shards_per_kind is None else per[:max_shards_per_kind])

    def _stage(repo_path: str) -> Path:
        return hub.stage_hub_file(
            HF_DATA_REPO,
            repo_path,
            _u_store_target(dest, repo_path),
            repo_type="dataset",
            revision=revision,
        )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(max_workers, 6)) as pool:
        staged = list(pool.map(_stage, sorted(set(wanted))))
    manifest_local = hub.stage_hub_file(
        HF_DATA_REPO,
        manifest_path,
        dest / "manifest.jsonl",
        repo_type="dataset",
        revision=manifest_revision,
    )
    record = {
        "repo": HF_DATA_REPO,
        "revision": revision,
        "prefix": prefix,
        "cell": cell,
        "kinds": list(kinds),
        "layers": [int(x) for x in layers],
        "max_shards_per_kind": max_shards_per_kind,
        "manifest_path": manifest_path,
        "manifest_revision": manifest_revision,
        "complete": max_shards_per_kind is None,
        "n_files": len(staged) + 1,
        "files": sorted(str(p) for p in [*staged, manifest_local]),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    record_path = dest / "staging_manifest.json"
    keep_existing = False
    if not record["complete"] and record_path.exists():
        # Never downgrade a COMPLETE staging record with a probe-slice one.
        try:
            keep_existing = json.loads(record_path.read_text()).get("complete") is True
        except (OSError, json.JSONDecodeError):
            keep_existing = False
    if not keep_existing:
        tmp = record_path.with_name("staging_manifest.tmp.json")
        tmp.write_text(json.dumps(record, indent=2))
        os.replace(tmp, record_path)
    logger.info(
        "[store_io] staged u_store: %d files under %s (rev %s, cell %s) in %.0fs",
        len(staged) + 1,
        dest,
        revision,
        cell,
        time.time() - t0,
    )
    return dest


def load_summaries(
    local_dir: Path | str,
    kinds: tuple[str, ...],
    layers: tuple[int, ...],
    *,
    cell: str | None = None,
    n_rows: int | None = None,
    hidden_dim: int = HIDDEN_DIM,
) -> tuple[dict[tuple[str, int], np.ndarray], list[dict]]:
    """Load staged summaries + row metadata (canonical AND realized layouts).

    Returns ``({(kind, layer): (n, hidden_dim) array}, meta_rows)`` keyed by
    the REQUESTED kind names; arrays keep the stored dtype (fp16). Requested
    kinds resolve canonical-name-first, then the realized #1092 alias
    (``REALIZED_KIND_FOR``); row metadata resolves through the
    ``_index_rows_for`` ladder (row_index -> per-kind stems -> manifest.jsonl).
    Asserts row-count consistency across kinds and vs the metadata source.
    ``hidden_dim`` defaults to the production pin; tiny-real store tests
    override it.
    """
    root = Path(local_dir) / cell if cell else Path(local_dir)
    out: dict[tuple[str, int], np.ndarray] = {}
    resolved: dict[str, str] = {}
    for kind in kinds:
        for layer in layers:
            resolved_kind, paths = _resolve_summary_kind(root, kind, layer)
            resolved.setdefault(kind, resolved_kind)
            arr = np.concatenate([np.load(p) for p in paths], axis=0)
            assert arr.ndim == 2 and arr.shape[1] == hidden_dim, (kind, layer, arr.shape)
            out[(kind, layer)] = arr[:n_rows] if n_rows else arr
    meta = _index_rows_for(root, [resolved[k] for k in kinds])
    if n_rows:
        meta = meta[:n_rows]
    counts = {k: v.shape[0] for k, v in out.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"inconsistent row counts across (kind, layer): {counts}")
    n = next(iter(counts.values()))
    if len(meta) != n:
        raise ValueError(f"row metadata count {len(meta)} != summary rows {n}")
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
