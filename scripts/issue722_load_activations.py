#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, r̂, ※, ρ, ×) in scientific docstrings + log messages.
"""Issue #722 — activation-load utility for the #667 paired base/post-FT store.

Streams ``superkaiba1/explore-persona-space-data`` →
``issue667_gate_chain_preview/analysis_tensors/`` and returns, per
``(behavior, layer)``, the list of per-cell tuples M0/M⁺ are fit from.

**Verified store schema (direct HF fetch, plan §2 + this loader's smoke).** Files
are LAYER-BAKED — ``<target_cid>_L7.npz`` / ``_L14.npz`` / ``_L21.npz`` are THREE
SEPARATE files per ``(behavior, source_cid, target_cid, seed)`` tuple, laid out
as ``analysis_tensors/{behavior}/{source_cid}_seed42/{target_cid}_L{li}.npz``.
Within each file ``v0`` / ``v_plus`` / ``c_C`` / ``c_C_postft`` are ``(3584,)``
SINGLE-VECTORS at that file's baked layer (NEVER ``(28, 3584)`` — only the
``c_*_all_layers`` CONTEXT keys carry the 28-layer stacks). ``behavior`` /
``source_cid`` / ``target_cid`` / ``layer`` are read FROM the file metadata
(robust to any filename convention drift), not parsed from the path.

**Count reconciliation vs the plan (IMPORTANT).** The plan body repeatedly states
"5 behaviors incl. refusal, 1152 per behavior (5760/5)". The store actually holds
**4 behaviors** (``em``, ``fact``, ``marker``, ``sycophancy``) — refusal IS a
TARGET behavior inside every source dir (``binst_refusal_L*.npz`` files exist) but
there is NO ``refusal/`` SOURCE-behavior directory. So the layout is **4
behaviors × 1440 each = 5760**, and **480 cells per behavior×layer**
(16 source_cids × 30 target_cids). This loader asserts the 480-per-behavior×layer
count and the 4-behavior set; it does NOT assert refusal as a source behavior nor
the plan's 1152/behavior figure. refusal-as-headline was dropped anyway (plan
§4.3 / §5 saturation guard), so the headline em/sycophancy/fact are unaffected.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("issue722.load")

DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_PREFIX = "issue667_gate_chain_preview/analysis_tensors"
HIDDEN = 3584
STORE_BEHAVIORS = (
    "em",
    "fact",
    "marker",
    "sycophancy",
)  # SOURCE-behavior dirs (refusal is target-only)
SWEEP_LAYERS = (7, 14, 21)
EXPECTED_CELLS_PER_BEHAVIOR_LAYER = 480  # 16 source_cids × 30 target_cids (verified)

_VEC_KEYS = ("v0", "v_plus", "c_C", "c_C_postft")
_META_KEYS = ("behavior", "source_cid", "target_cid", "layer")

# #811: answer-side summary → the (v0, v_plus) .npz keys carrying it. "mean" is the
# #667/#722 store's mean-over-response (v0/v_plus); "turn_nl" is #811's
# turn-boundary single-position read (v0_turn_nl/v_plus_turn_nl); "maxp" is #810's
# crowned per-dimension content-token max (v0_maxp/v_plus_maxp — the #811
# maxp-winner round, present only in its re-extracted store). The CONTEXT keys
# (c_C/c_C_postft) are summary-INDEPENDENT (answer-side change only), so the loader
# reads the SAME c_C/c_C_postft for every summary — the manipulated variable is
# answer-side.
_SUMMARY_ANSWER_KEYS = {
    "mean": ("v0", "v_plus"),
    "turn_nl": ("v0_turn_nl", "v_plus_turn_nl"),
    "maxp": ("v0_maxp", "v_plus_maxp"),
}


@dataclass
class CellRecord:
    """One (behavior, source_cid, target_cid, layer) base/post-FT activation cell."""

    behavior: str
    source_cid: str
    target_cid: str
    layer: int
    c0: np.ndarray  # base context vector c_C, (3584,)
    cplus: np.ndarray  # post-FT context vector c_C_postft, (3584,)
    v0: np.ndarray  # base answer-profile, (3584,)
    vplus: np.ndarray  # post-FT answer-profile, (3584,)
    family: str = field(default="")  # family_of(target_cid), set at load


def _family_of(cid: str) -> str:
    """7-family cluster label (imported from #667's gate_chain — single source of truth)."""
    from explore_persona_space.analysis.issue667.gate_chain import family_of

    return family_of(cid)


class _Streamer:
    """Per-file HF stream + LRU delete so peak footprint stays ~one file, not 5760.

    Mirrors #658's ``_HfStreamSpanSource`` pattern. Each ``load`` downloads ONE
    ``.npz`` into a private staging dir we own (so deletes can't race other
    readers) and an LRU bounds resident files.

    **Local-mirror mode (additive, #667 recovery).** When ``local_root`` is set,
    ``load`` reads the ``.npz`` DIRECTLY from ``<local_root>/<rel_path>`` with NO
    HF download, staging, LRU, or cleanup — the file already lives on disk (the
    complete store mirror on the compute node) so there is nothing to stream or
    reap. The HF path (``local_root is None``) is UNCHANGED. This lets the #667
    all-layer analysis read the on-node mirror when the HF repo-listing hangs.
    """

    def __init__(
        self,
        repo_id: str = DATA_REPO,
        prefix: str = STORE_PREFIX,
        cache_size: int = 8,
        local_root: str | os.PathLike[str] | None = None,
    ):
        self.repo_id = repo_id
        self.prefix = prefix.rstrip("/")
        self.cache_size = max(1, int(cache_size))
        self._resident: dict[str, Path] = {}
        self.local_root = Path(local_root) if local_root is not None else None
        if self.local_root is not None:
            self._staging = None  # no staging dir in local-mirror mode
            return
        cache_root = Path(
            os.environ.get("HF_HOME")
            or os.environ.get("XDG_CACHE_HOME")
            or (Path.home() / ".cache")
        )
        self._staging = cache_root / "issue722_npz_stream"
        self._staging.mkdir(parents=True, exist_ok=True)

    def load(self, rel_path: str) -> dict:
        if self.local_root is not None:
            path = self.local_root / rel_path
            if not path.exists():
                raise FileNotFoundError(
                    f"local-mirror npz missing: {path} (rel_path={rel_path!r}); "
                    f"expected the complete store mirror under {self.local_root}"
                )
            d = np.load(path, allow_pickle=True)
            return {k: d[k] for k in d.files}

        from huggingface_hub import hf_hub_download

        path = self._resident.get(rel_path)
        if path is None or not path.exists():
            dest = self._staging / rel_path.replace("/", "__")
            dest.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                self.repo_id,
                f"{self.prefix}/{rel_path}",
                repo_type="dataset",
                local_dir=str(dest),
            )
            path = Path(downloaded)
            self._resident[rel_path] = path
            while len(self._resident) > self.cache_size:
                old = next(iter(self._resident))
                self._evict(old)
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    def _evict(self, rel_path: str) -> None:
        self._resident.pop(rel_path, None)
        shutil.rmtree(self._staging / rel_path.replace("/", "__"), ignore_errors=True)

    def cleanup(self) -> None:
        if self._staging is not None:
            shutil.rmtree(self._staging, ignore_errors=True)


def list_store_layout(
    behaviors: tuple[str, ...] = STORE_BEHAVIORS,
    prefix: str = STORE_PREFIX,
) -> dict[str, dict[str, list[str]]]:
    """Enumerate ``{behavior: {source_cid_dir: [target_file_stem, ...]}}`` from HF.

    Uses ``HfApi.list_repo_tree`` non-recursively per level (the recursive listing
    504s on this large repo — see the loader smoke). Lists only the requested
    ``behaviors`` (fewer tree calls = fewer transient-504 chances). Returns the
    directory map so the loader knows exactly which files to stream without a
    fragile filename regex.

    ``prefix`` (additive, #811): the analysis-tensor prefix to enumerate under —
    defaults to the #667/#722 ``STORE_PREFIX``; #811 passes its own
    ``issue811_turn_nl_mapchange/analysis_tensors`` so a ``turn_nl`` run reads its
    re-extracted store rather than the mean-only #667 store. Pair with a
    ``_Streamer(prefix=...)`` at the SAME prefix in :func:`load_cells`.
    """
    prefix = prefix.rstrip("/")
    import time as _time

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()

    def _tree(path: str) -> list:
        """list_repo_tree with bounded retry — the repo's tree API 504s transiently.

        Five attempts with exponential backoff (1/2/4/8 s). A persistent failure
        re-raises (fail loud) rather than returning a partial/empty listing that
        would silently under-count cells downstream.
        """
        last: Exception | None = None
        for attempt in range(5):
            try:
                return list(api.list_repo_tree(DATA_REPO, path_in_repo=path, repo_type="dataset"))
            except HfHubHTTPError as e:  # 504/503/timeout — transient on this large repo
                last = e
                wait = 2**attempt
                logger.warning(
                    "list_repo_tree(%s) failed (%s); retry %d in %ds", path, e, attempt + 1, wait
                )
                _time.sleep(wait)
        raise RuntimeError(f"list_repo_tree({path}) failed after 5 attempts") from last

    layout: dict[str, dict[str, list[str]]] = {}
    for beh in behaviors:
        beh_path = f"{prefix}/{beh}"
        src_dirs = [
            t.path for t in _tree(beh_path) if t.path != beh_path and not t.path.endswith(".npz")
        ]
        layout[beh] = {}
        for sd in src_dirs:
            src_name = sd.split("/")[-1]  # e.g. binst_em_seed42
            files = [t.path.split("/")[-1] for t in _tree(sd) if t.path.endswith(".npz")]
            layout[beh][src_name] = sorted(files)
    return layout


def list_store_layout_local(
    local_root: str | os.PathLike[str],
    behaviors: tuple[str, ...] = STORE_BEHAVIORS,
) -> dict[str, dict[str, list[str]]]:
    """Enumerate ``{behavior: {source_cid_dir: [target_file_stem, ...]}}`` from a LOCAL mirror.

    Local twin of :func:`list_store_layout` — walks ``<local_root>/<behavior>/<src>/``
    on disk instead of the HF tree API (which hangs on this large repo, #667). The
    returned map has EXACTLY the HF version's shape and key convention: the layout
    key is the BARE source dir name (``binst_em_seed42``, NOT ``em/binst_em_seed42``)
    — the HF version derives it via ``sd.split("/")[-1]``. :func:`load_cells` then
    builds ``rel_path = f"{beh}/{src_dir}/{fn}"`` (it prepends ``beh/`` itself), which
    :meth:`_Streamer.load` resolves against the same ``local_root`` as
    ``<local_root>/{beh}/{src_dir}/{fn}``. A requested behavior dir that is absent is
    simply omitted (``load_cells`` then fails loud on the missing behavior key).
    """
    root = Path(local_root)
    layout: dict[str, dict[str, list[str]]] = {}
    for beh in behaviors:
        beh_dir = root / beh
        if not beh_dir.is_dir():
            continue
        layout[beh] = {}
        for src_dir in sorted(p for p in beh_dir.iterdir() if p.is_dir()):
            src_name = src_dir.name  # BARE dir name — matches the HF layout keys
            files = sorted(p.name for p in src_dir.iterdir() if p.name.endswith(".npz"))
            layout[beh][src_name] = files
    return layout


def _blob_to_record(blob: dict, rel: str, beh: str, li: int, summary: str = "mean") -> CellRecord:
    """Validate one loaded ``.npz`` blob and build its CellRecord (fail loud on miss).

    Reads each file's OWN ``(3584,)`` baked-layer ``c_C`` / ``c_C_postft`` and the
    answer-side vectors for ``summary`` (``mean`` → ``v0``/``v_plus``; ``turn_nl``
    → ``v0_turn_nl``/``v_plus_turn_nl``, #811), and re-reads ``behavior`` /
    ``source_cid`` / ``target_cid`` / ``layer`` from inside the file (robust to
    filename drift). The CONTEXT keys are summary-independent (the manipulated
    variable is answer-side only). A missing key or a shape / layer / behavior
    mismatch raises (the schema is verified — a miss is a wrong file, a stale
    mirror, or a ``turn_nl`` read against a mean-only store).
    """
    v0_key, vplus_key = _SUMMARY_ANSWER_KEYS[summary]
    # Context keys (c_C/c_C_postft) + meta are always required; the answer keys are
    # summary-specific (v0/v_plus for mean, v0_turn_nl/v_plus_turn_nl for turn_nl).
    for k in ("c_C", "c_C_postft", v0_key, vplus_key, *_META_KEYS):
        if k not in blob:
            raise KeyError(f"{rel} missing key {k!r} (summary={summary!r}); keys={sorted(blob)}")
    c0 = np.asarray(blob["c_C"], dtype=np.float64)
    cplus = np.asarray(blob["c_C_postft"], dtype=np.float64)
    v0 = np.asarray(blob[v0_key], dtype=np.float64)
    vplus = np.asarray(blob[vplus_key], dtype=np.float64)
    for name, arr in (("c_C", c0), ("c_C_postft", cplus), ("v0", v0), ("v_plus", vplus)):
        assert arr.shape == (HIDDEN,), f"{rel} {name} shape {arr.shape} != ({HIDDEN},)"
    file_layer = int(np.asarray(blob["layer"]).item())
    assert file_layer == li, f"{rel} baked layer {file_layer} != requested {li}"
    file_beh = str(np.asarray(blob["behavior"]).item())
    assert file_beh == beh, f"{rel} behavior {file_beh} != dir {beh}"
    tgt_cid = str(np.asarray(blob["target_cid"]).item())
    return CellRecord(
        behavior=beh,
        source_cid=str(np.asarray(blob["source_cid"]).item()),
        target_cid=tgt_cid,
        layer=li,
        c0=c0,
        cplus=cplus,
        v0=v0,
        vplus=vplus,
        family=_family_of(tgt_cid),
    )


def _parse_cell_files(src_dir: str, files: list[str], layers: tuple[int, ...]) -> dict:
    """Group a source dir's files by target_cid → {layer: filename} for the swept layers.

    File names are ``{target_cid}_L{li}.npz``. We strip the ``_L{li}.npz`` suffix
    to recover the target stem; ``behavior``/``source_cid``/``target_cid``/``layer``
    are re-read from inside each file at load (the path parse only selects WHICH
    files to fetch).
    """
    by_target: dict[str, dict[int, str]] = {}
    for fn in files:
        for li in layers:
            suf = f"_L{li}.npz"
            if fn.endswith(suf):
                stem = fn[: -len(suf)]
                by_target.setdefault(stem, {})[li] = f"{src_dir}/{fn}"
    return by_target


def load_cells(
    behaviors: tuple[str, ...] = STORE_BEHAVIORS,
    layers: tuple[int, ...] = SWEEP_LAYERS,
    *,
    max_cells: int | None = None,
    max_sources: int | None = None,
    max_targets_per_source: int | None = None,
    streamer: _Streamer | None = None,
    strict_counts: bool = True,
    layout: dict[str, dict[str, list[str]]] | None = None,
    summary: str = "mean",
) -> dict[tuple[str, int], list[CellRecord]]:
    """Load per-(behavior, layer) cell records from the #667 store.

    Returns ``{(behavior, layer): [CellRecord, ...]}``. For each
    ``(source_cid, target_cid)`` cell it reads the three layer-baked files and
    pulls each file's OWN ``(3584,)`` ``c_C`` / ``c_C_postft`` / ``v0`` /
    ``v_plus`` at that baked layer (no cross-layer ``(28,3584)`` indexing).
    ``family`` is ``family_of(target_cid)`` for the clustered bootstrap.

    **CRITICAL substrate fact (verified):** ``c_C`` / ``c_C_postft`` are keyed to
    ``source_cid`` ONLY — they are IDENTICAL across a source's 30 targets — while
    ``v0`` / ``v_plus`` vary per (source, target). So the fit ``M: c → v`` has only
    16 DISTINCT INPUTS (one per source_cid), each paired with 30 outputs; ridge
    fits the per-source MEAN answer-profile. The plan's "n≈16 contexts" IS this
    distinct-source count. A smoke MUST therefore span ≥2 SOURCES (``max_sources``),
    not just multiple targets of one source — capping to a single source gives one
    constant input and a degenerate (all-zero) ridge fit.

    ``max_sources`` (smoke) caps the number of source_cid dirs loaded per behavior
    (the right knob to keep the fit non-degenerate). ``max_cells`` additionally caps
    total (source,target) cells. ``strict_counts`` asserts the verified 480-cell
    per-behavior×layer count (disabled whenever a cap is set).

    Exact pairing contract (asserted): M0 is fit from ``(c0 → v0)``; M⁺ from
    ``(cplus → vplus)`` — the post-FT input drives the post-FT output (plan §4.1).
    A KeyError on any required key fails LOUD (the schema is verified; a miss is a
    wrong file / stale mirror).

    ``layout`` (additive, #667 recovery): pass a pre-built directory map (e.g. from
    :func:`list_store_layout_local`) to SKIP the HF tree walk (which hangs on this
    large repo). Default ``None`` → the HF :func:`list_store_layout` path, UNCHANGED.

    ``summary`` (additive, #811): which answer-side summary the loaded
    ``v0``/``v_plus`` fields carry — ``"mean"`` (DEFAULT, the #667/#722
    mean-over-response) or ``"turn_nl"`` (the turn-boundary single-position read
    ``v0_turn_nl``/``v_plus_turn_nl``, present only in #811's re-extracted store).
    The context fields (``c0``/``cplus`` = ``c_C``/``c_C_postft``) are IDENTICAL
    across summaries — the manipulated variable is answer-side only, so a
    ``summary="turn_nl"`` load reads the SAME c_C the ``summary="mean"`` load does.
    A ``turn_nl`` load against a mean-only store fails loud (missing key).
    """
    own = streamer is None
    streamer = streamer or _Streamer()
    if layout is None:
        layout = list_store_layout(behaviors)
    out: dict[tuple[str, int], list[CellRecord]] = {(b, li): [] for b in behaviors for li in layers}
    try:
        for beh in behaviors:
            if beh not in layout:
                raise KeyError(f"behavior {beh!r} not present in store layout {sorted(layout)}")
            n_cells = 0
            src_items = sorted(layout[beh].items())
            if max_sources is not None:
                src_items = src_items[:max_sources]
            for src_dir, files in src_items:
                by_target = _parse_cell_files(src_dir, files, layers)
                n_src = 0
                for _target_stem, layer_files in sorted(by_target.items()):
                    if max_cells is not None and n_cells >= max_cells:
                        break
                    if max_targets_per_source is not None and n_src >= max_targets_per_source:
                        break
                    if not all(li in layer_files for li in layers):
                        continue  # cell missing a swept layer — skip (kept loud via final count)
                    for li in layers:
                        rel = f"{beh}/{layer_files[li]}"
                        out[(beh, li)].append(
                            _blob_to_record(streamer.load(rel), rel, beh, li, summary)
                        )
                    n_cells += 1
                    n_src += 1
                if max_cells is not None and n_cells >= max_cells:
                    break
            if strict_counts:
                for li in layers:
                    got = len(out[(beh, li)])
                    assert got == EXPECTED_CELLS_PER_BEHAVIOR_LAYER, (
                        f"{beh} L{li}: loaded {got} cells, expected "
                        f"{EXPECTED_CELLS_PER_BEHAVIOR_LAYER} (16 sources × 30 targets)"
                    )
            logger.info("loaded %s: %s", beh, {li: len(out[(beh, li)]) for li in layers})
    finally:
        if own:
            streamer.cleanup()
    return out


def stack_for_fit(cells: list[CellRecord]) -> dict:
    """Stack a behavior×layer cell list into the (n, 3584) design arrays + keys.

    Returns ``{"C0", "Cplus", "V0", "Vplus", "families", "source_cids",
    "target_cids", "cell_keys"}`` where ``cell_keys`` are the
    ``{behavior}/{source}__{target}`` strings the chain-ρ join uses against
    #537's ``G_meta.json``, and ``common_c_grid`` (the base c0 grid for read (1))
    is ``C0`` itself.
    """
    if not cells:
        return {
            "C0": np.zeros((0, HIDDEN)),
            "Cplus": np.zeros((0, HIDDEN)),
            "V0": np.zeros((0, HIDDEN)),
            "Vplus": np.zeros((0, HIDDEN)),
            "families": [],
            "source_cids": [],
            "target_cids": [],
            "cell_keys": [],
        }
    C0 = np.stack([c.c0 for c in cells])
    Cplus = np.stack([c.cplus for c in cells])
    V0 = np.stack([c.v0 for c in cells])
    Vplus = np.stack([c.vplus for c in cells])
    families = [c.family for c in cells]
    source_cids = [c.source_cid for c in cells]
    target_cids = [c.target_cid for c in cells]
    cell_keys = [f"{c.behavior}/{c.source_cid}__{c.target_cid}" for c in cells]
    return {
        "C0": C0,
        "Cplus": Cplus,
        "V0": V0,
        "Vplus": Vplus,
        "families": families,
        "source_cids": source_cids,
        "target_cids": target_cids,
        "cell_keys": cell_keys,
    }


def common_c_grid(stacks: dict) -> np.ndarray:
    """The fixed-input grid for read (1): the base context vectors c0 (n, 3584)."""
    return stacks["C0"]
