#!/usr/bin/env python3
"""Issue #2215 — Phase C analysis core: DV1/DV2/DV3 + nulls + baselines (plan §4.3).

Pure-function statistical core over the staged banked tensors; the pod driver
(``issue2215_run.py``) stages inputs, runs the §4.1 gates and the Phase-C
sentinel gate (``va2215_uploaded.json`` presence + ``regime_fp`` match —
concern ``unit2-phase-c-sentinel-gate`` / #825 store-before-long-fit), then
calls :func:`run_analysis`. Every statistical read is a function over
in-memory tensors + the pair table, unit-testable at tiny synthetic grain
(``tests/test_issue2215_analysis.py``).

Conventions (plan §4.3 / §6 / §8):

- **fp64 upcast at compute** everywhere (plan §8 quantization row); stores
  stay fp16/fp32 on disk.
- **No serial per-pair/per-draw tensor loops** (vectorize-many-cell-fits):
  pairwise cosines via normalized Grams, DV1/DV2 label-permutation nulls via
  batched sign-flip GEMMs ``(S @ G) * S``, the DV3 shuffled-pair null via
  per-cell similarity blocks precomputed ONCE per config and re-INDEXED per
  draw (no GEMM inside the draw loop), bootstrap CIs via the #2094 batched
  index-GEMM ``bootstrap_family_means_batched``. Python loops run over
  CELLS / CONFIGS only (~39 / ~60 iterations of vectorized work), with
  per-unit progress lines (`[dv3] unit k/N ...`).
- **Checkpoint per DV family**: each DV's JSON + per-pair jsonl is written
  atomically the moment it completes (dv1 → dv2 → coupling → dv3 →
  null_bands → null matrices); a crash loses at most the in-flight family.
  Projected phase wall ≪ the ~1h intra-phase floor (plan §9 row C books
  0.75 h), so no per-config resume machinery is carried.
- **Seeds**: null/permutation draws seed 2215 (per-unit ``default_rng``
  spawn keys ``[seed, dv_tag, cell_index, ...]`` — recorded in the outputs'
  meta); bootstrap seed 21620 (parent #2162 convention). DV1/DV2 sign-flip
  nulls have only ``2^m`` distinct sign atoms per unit (m=12 carriers), so
  no tail p below ~1/4096 — recorded in the band meta.
- **Exclusions**: a pair is excluded from DV2/DV3 only when a side has
  ``n_valid == 0`` (graceful floor, plan §4.1 — reported per cell, never
  silent). The two pre-declared degenerate-at-pe cells are sanity-checked
  (pair pe-state cosine ≥ 0.99 — the parent's loose state-sanity band; an
  exact-zero assert false-FAILs on bf16 batch-composition jitter, ~2e-4)
  and excluded from pe aggregates (DV1) and from every pe-INPUT DV3 arm
  (labeled ``N/A — degenerate at pe``).
- **Smoke (--cells)** analyzes the sliced cells only; reads needing ≥3
  cells (H2 coupling, between-cell geometry summaries) degrade to a
  recorded skip — same code path, smaller grain.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (shared-VM thread caps + HF token)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# scripts/ sibling imports resolve in script mode; the insert covers the
# imported-as-module case (pytest, driver import).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)

logger = logging.getLogger("issue2215.analysis")

SEED_NULL = 2215  # plan §10: null/permutation seed
SEED_BOOT = 21620  # plan §10: bootstrap seed (parent convention)
PRIMARY_LAYER = 19
MAP_LAYERS = (14, 19, 26)
# Parent's loose state-sanity band for premise-verified degenerate-at-pe pairs
# (issue2162_run.run_degeneracy_guard: bf16 batch jitter ~2e-4 in cosine).
DEGENERATE_PE_COS_MIN = 0.99
POOLINGS = ("tail", "span")  # tail-inclusive primary; span-mean secondary
POOL_PRIMARY = "tail"
METRICS = ("cosine", "euclidean")
_EPS = 1e-30


# ── pair table + cell views ───────────────────────────────────────────


@dataclass(frozen=True)
class PairTable:
    """Scoped context rows + directed pairs (built from the staged bank.json)."""

    ids: list[str]  # context ids, sorted — the ROW ORDER of every matrix
    row_of: dict[str, int]
    cell_of: list[str]  # per row
    cells: list[str]  # sorted analyzed cells
    pair_ids: list[str]
    pair_cell: list[str]
    pair_carrier: list[str]
    pair_vp: list[str]  # "value_a-value_b"
    a_row: np.ndarray
    b_row: np.ndarray

    @classmethod
    def from_bank(cls, bank: dict, scope_cells: tuple[str, ...] | None) -> PairTable:
        contexts = bank["contexts"]
        keep = (lambda c: True) if scope_cells is None else (lambda c: c in set(scope_cells))
        ids = sorted(cid for cid, ctx in contexts.items() if keep(ctx["cell"]))
        assert ids, f"no contexts in scope {scope_cells}"
        row_of = {cid: i for i, cid in enumerate(ids)}
        cell_of = [contexts[cid]["cell"] for cid in ids]
        pairs = [p for p in bank["pairs"] if keep(p["cell"])]
        assert pairs, f"no pairs in scope {scope_cells}"
        for p in pairs:
            assert p["a"] in row_of and p["b"] in row_of, (p["pair_id"], "pair↔context drift")
        return cls(
            ids=ids,
            row_of=row_of,
            cell_of=cell_of,
            cells=sorted({p["cell"] for p in pairs}),
            pair_ids=[p["pair_id"] for p in pairs],
            pair_cell=[p["cell"] for p in pairs],
            pair_carrier=[p["carrier"] for p in pairs],
            pair_vp=[f"{p['value_a']}-{p['value_b']}" for p in pairs],
            a_row=np.array([row_of[p["a"]] for p in pairs], dtype=np.int64),
            b_row=np.array([row_of[p["b"]] for p in pairs], dtype=np.int64),
        )


@dataclass(frozen=True)
class CellView:
    """One cell's local index structure (contexts, carriers, value-pairs)."""

    cell: str
    ctx_rows: np.ndarray  # global rows of this cell's contexts (sorted cids)
    value_loc: np.ndarray  # per local ctx: index into `values`
    values: list[str]
    carriers: list[str]
    vps: list[str]
    pair_idx: np.ndarray  # global pair indices (this cell's pairs)
    a_loc: np.ndarray  # per cell-pair: LOCAL context index of side A
    b_loc: np.ndarray
    carrier_loc: np.ndarray  # per cell-pair: index into `carriers`
    vp_loc: np.ndarray  # per cell-pair: index into `vps`
    pair_at: np.ndarray  # (n_carriers, n_vps) -> local pair index (complete grid)


def build_cell_views(bank: dict, pt: PairTable) -> dict[str, CellView]:
    """Per-cell local index structures; asserts the (carrier × vp) grid is complete.

    Cell membership (``ctx_rows``) is the sorted union of global rows the
    cell's own PAIRS reference (``pt.a_row``/``pt.b_row``), NOT the
    per-context ``cell`` attribution: the frozen #2162 bank's two
    reverse-direction conflict cells (``conflict_format_rev``,
    ``conflict_persona_rev``) own ZERO contexts — their 36 pairs re-pair the
    matching ``_fwd`` cell's 72 contexts in crossed value combinations, so a
    ``cell_of`` grouping yields an empty ``local_of`` and the first pair-side
    lookup crashes (production Phase C ``KeyError: 24``). Cells whose
    pair-derived membership differs from their ``cell_of`` grouping are
    reported in the build-time ``[cell-views]`` log line.
    """
    contexts = bank["contexts"]
    attributed: dict[str, set[int]] = defaultdict(set)
    for row, cell in enumerate(pt.cell_of):
        attributed[cell].add(row)
    views: dict[str, CellView] = {}
    borrowed: list[str] = []
    for cell in pt.cells:
        p_idx = np.array([k for k, c in enumerate(pt.pair_cell) if c == cell], dtype=np.int64)
        referenced = {int(pt.a_row[k]) for k in p_idx} | {int(pt.b_row[k]) for k in p_idx}
        if referenced != attributed[cell]:
            borrowed.append(f"{cell}({len(referenced)})")
        ctx_rows = np.array(sorted(referenced), dtype=np.int64)
        local_of = {int(r): k for k, r in enumerate(ctx_rows)}
        values = sorted({contexts[pt.ids[r]]["value_id"] for r in ctx_rows})
        value_loc = np.array(
            [values.index(contexts[pt.ids[r]]["value_id"]) for r in ctx_rows], dtype=np.int64
        )
        carriers = sorted({pt.pair_carrier[int(k)] for k in p_idx})
        vps = sorted({pt.pair_vp[int(k)] for k in p_idx})
        a_loc = np.array([local_of[int(pt.a_row[k])] for k in p_idx], dtype=np.int64)
        b_loc = np.array([local_of[int(pt.b_row[k])] for k in p_idx], dtype=np.int64)
        carrier_loc = np.array([carriers.index(pt.pair_carrier[int(k)]) for k in p_idx])
        vp_loc = np.array([vps.index(pt.pair_vp[int(k)]) for k in p_idx])
        pair_at = np.full((len(carriers), len(vps)), -1, dtype=np.int64)
        for j in range(len(p_idx)):
            assert pair_at[carrier_loc[j], vp_loc[j]] == -1, (cell, "duplicate (carrier, vp)")
            pair_at[carrier_loc[j], vp_loc[j]] = j
        assert (pair_at >= 0).all(), (cell, "incomplete (carrier × vp) pair grid")
        views[cell] = CellView(
            cell=cell,
            ctx_rows=ctx_rows,
            value_loc=value_loc,
            values=values,
            carriers=carriers,
            vps=vps,
            pair_idx=p_idx,
            a_loc=a_loc,
            b_loc=b_loc,
            carrier_loc=carrier_loc,
            vp_loc=vp_loc,
            pair_at=pair_at,
        )
    logger.info(
        "[cell-views] %d cells; borrowed-membership: %s",
        len(views),
        ", ".join(borrowed) if borrowed else "none",
    )
    return views


# ── loaders ───────────────────────────────────────────────────────────


def load_vc_bank(path: Path, ids: list[str]) -> dict:
    """Staged vc_bank.pt → per-slot (n, L, H) fp32 stacks in row order ``ids``."""
    assert path.exists(), f"{path} missing — Phase A staging incomplete"
    payload = torch.load(path, map_location="cpu", weights_only=False)  # self-produced bundle
    recs = payload["per_context"]
    missing = [cid for cid in ids if cid not in recs]
    assert not missing, f"vc_bank missing {len(missing)} scoped contexts (first: {missing[:3]})"
    out: dict[str, torch.Tensor] = {}
    for slot, key in (("ce", "v_ce"), ("pe", "v_pe")):
        out[slot] = torch.stack([recs[cid][key] for cid in ids]).float()
        assert out[slot].dim() == 3, out[slot].shape
    layers = list(payload["layers"])
    n, ln, h = out["ce"].shape
    assert n == len(ids) and ln == len(layers), (out["ce"].shape, len(ids), len(layers))
    return {"layers": layers, "hidden": h, **out}


@dataclass
class AnswerMeans:
    """Per-context answer-state means (both poolings) + split-half legs (tail)."""

    layers: list[int]
    mean: dict[str, torch.Tensor]  # pooling -> (n, L, H) fp64 (0 rows where n_valid==0)
    half1: dict[str, torch.Tensor]
    half2: dict[str, torch.Tensor]
    n_valid: np.ndarray  # (n,) valid (non-empty) draws per context
    n_h1: np.ndarray
    n_h2: np.ndarray
    span_source: str


def _accumulate_store(
    files: list[Path],
    tensor_keys: dict[str, str],  # pooling -> payload tensor key
    ids_set: set[str],
    row_of: dict[str, int],
    n_ctx: int,
    k_draws: int,
) -> tuple[dict, list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Stream shards → fp64 sum buffers per pooling (full + halves) + counts."""
    assert files, "no shards to accumulate"
    layers: list[int] | None = None
    sums: dict[tuple[str, str], torch.Tensor] = {}
    n_valid = np.zeros(n_ctx, dtype=np.int64)
    n_h1 = np.zeros(n_ctx, dtype=np.int64)
    n_h2 = np.zeros(n_ctx, dtype=np.int64)
    seen: set[tuple[str, int]] = set()
    half_cut = k_draws // 2  # draws {0..cut-1} vs {cut..k-1} (plan §4.1: {0..4} vs {5..9})
    for shard in files:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        if layers is None:
            layers = list(payload["layers"])
            for pool, key in tensor_keys.items():
                lt, ht = payload[key].shape[1:]
                for leg in ("full", "h1", "h2"):
                    sums[(pool, leg)] = torch.zeros((n_ctx, lt, ht), dtype=torch.float64)
        assert list(payload["layers"]) == layers, (shard, "layer-list drift across shards")
        empty = set(payload.get("empty_rows", []))
        rows_j: list[int] = []
        rows_tgt: list[int] = []
        rows_h1: list[bool] = []
        for j, meta in enumerate(payload["index"]):
            cid, draw = meta["context_id"], int(meta["draw"])
            if cid not in ids_set:
                continue
            key = (cid, draw)
            assert key not in seen, (shard, key, "duplicate (context_id, draw) across shards")
            seen.add(key)
            if j in empty:
                continue
            rows_j.append(j)
            rows_tgt.append(row_of[cid])
            rows_h1.append(draw < half_cut)
        if not rows_j:
            continue
        j_idx = torch.tensor(rows_j, dtype=torch.long)
        tgt = torch.tensor(rows_tgt, dtype=torch.long)
        h1_mask = torch.tensor(rows_h1)
        for pool, key in tensor_keys.items():
            vals = payload[key][j_idx].double()
            sums[(pool, "full")].index_add_(0, tgt, vals)
            if h1_mask.any():
                sums[(pool, "h1")].index_add_(0, tgt[h1_mask], vals[h1_mask])
            if (~h1_mask).any():
                sums[(pool, "h2")].index_add_(0, tgt[~h1_mask], vals[~h1_mask])
        np.add.at(n_valid, np.asarray(rows_tgt), 1)
        h1_np = np.asarray(rows_h1)
        np.add.at(n_h1, np.asarray(rows_tgt)[h1_np], 1)
        np.add.at(n_h2, np.asarray(rows_tgt)[~h1_np], 1)
        del payload
    assert layers is not None
    return sums, layers, n_valid, n_h1, n_h2


def load_answer_means(
    va_dir: Path,
    ids: list[str],
    row_of: dict[str, int],
    *,
    banked_dir: Path | None,
    k_draws: int,
) -> AnswerMeans:
    """v̄_A per context: tail-inclusive (primary) from the va2215 store; span
    secondary from the BANKED va_anchors store (plan §4.3), or from the
    va2215 ``va_span_excl`` twin when ``banked_dir is None`` (tiny mode —
    DECLARED substitution: the banked store is full-shape and structurally
    incomparable with a tiny capture; the parity gate certifies the two span
    sources agree at ≥0.995 cosine in production)."""
    ids_set = set(ids)
    n_ctx = len(ids)
    va_files = sorted(va_dir.glob("va2215_*.pt"))
    keys = {"tail": "va_tail_incl"}
    span_source = "banked va_anchors va_span"
    if banked_dir is None:
        keys["span"] = "va_span_excl"
        span_source = "va2215 va_span_excl (tiny substitution — see docstring)"
    sums, layers, n_valid, n_h1, n_h2 = _accumulate_store(
        va_files, keys, ids_set, row_of, n_ctx, k_draws
    )
    if banked_dir is not None:
        b_files = sorted(banked_dir.glob("va_anchors_*.pt"))
        b_sums, b_layers, b_valid, _, _ = _accumulate_store(
            b_files, {"span": "va_span"}, ids_set, row_of, n_ctx, k_draws
        )
        assert b_layers == layers, (b_layers, layers, "banked↔va2215 layer drift")
        assert (b_valid == n_valid).all(), (
            "banked↔va2215 valid-draw count drift (empty-row parity is gated in Phase B)"
        )
        sums.update(b_sums)
    mean: dict[str, torch.Tensor] = {}
    half1: dict[str, torch.Tensor] = {}
    half2: dict[str, torch.Tensor] = {}
    cnt = torch.tensor(np.maximum(n_valid, 1), dtype=torch.float64)[:, None, None]
    c1 = torch.tensor(np.maximum(n_h1, 1), dtype=torch.float64)[:, None, None]
    c2 = torch.tensor(np.maximum(n_h2, 1), dtype=torch.float64)[:, None, None]
    for pool in ("tail", "span"):
        mean[pool] = sums[(pool, "full")] / cnt
        half1[pool] = sums[(pool, "h1")] / c1
        half2[pool] = sums[(pool, "h2")] / c2
    n_zero = int((n_valid == 0).sum())
    if n_zero:
        logger.warning("[va-means] %d context(s) with n_valid=0 — pairs excluded, reported", n_zero)
    return AnswerMeans(
        layers=layers,
        mean=mean,
        half1=half1,
        half2=half2,
        n_valid=n_valid,
        n_h1=n_h1,
        n_h2=n_h2,
        span_source=span_source,
    )


def load_parent_separation(path: Path) -> dict[str, float]:
    """Per-cell mean anchor ``separation`` from the parent's f_metrics/anchors.jsonl."""
    per_cell: dict[str, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("separation") is not None:
                per_cell[row["cell"]].append(float(row["separation"]))
    assert per_cell, f"no separation rows in {path}"
    return {cell: float(np.mean(v)) for cell, v in per_cell.items()}


# ── small math helpers (unit-tested) ──────────────────────────────────


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """Row-normalize over the LAST dim, fp64, zero-safe (clamped norm)."""
    xd = x.double()
    return xd / xd.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def mean_pairwise_cosine_from_gram(g: np.ndarray) -> float:
    """Mean over UNORDERED distinct pairs of the cosine Gram (m ≥ 2)."""
    m = g.shape[0]
    assert g.shape == (m, m) and m >= 2, g.shape
    return float((g.sum() - np.trace(g)) / (m * (m - 1)))


def signflip_null_consistency(g: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Label-permutation null for the mean pairwise cosine via sign flips.

    Flipping which context of a directed pair is called "B" flips that
    carrier's Δ sign, so cos(±u_i, ±u_j) = s_i s_j cos(u_i, u_j) and the
    null consistency for sign vector s is (sᵀGs − m) / (m(m−1)) —
    ONE batched GEMM per draw block, no per-draw loop.
    """
    m = g.shape[0]
    s = signs.astype(np.float64)
    assert s.shape[1] == m, (s.shape, m)
    tot = ((s @ g) * s).sum(axis=1)
    return (tot - m) / (m * (m - 1))


def bootstrap_pairwise_cosine(g: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Carrier-cluster bootstrap of the mean pairwise cosine.

    ``idx`` (B, m) resamples carriers with replacement; duplicate draws are
    EXCLUDED from the pair mean (a duplicated carrier contributes cos=1
    self-pairs that would bias the statistic upward).
    """
    gd = g[idx[:, :, None], idx[:, None, :]]  # (B, m, m)
    distinct = idx[:, :, None] != idx[:, None, :]
    num = (gd * distinct).sum(axis=(1, 2))
    den = distinct.sum(axis=(1, 2))
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def deranged_perms(n: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """(b, n) permutations with NO fixed point (carrier-blocked derangements)."""
    assert n >= 2, n
    out = np.empty((b, n), dtype=np.int64)
    filled = 0
    while filled < b:
        batch = max(2 * (b - filled) + 8, 16)
        perms = np.argsort(rng.random((batch, n)), axis=1)
        good = perms[(perms != np.arange(n)).all(axis=1)]
        take = good[: b - filled]
        out[filled : filled + len(take)] = take
        filled += len(take)
    return out


def rankdata_rows(a: np.ndarray) -> np.ndarray:
    """Average ranks along axis 1 (scipy rankdata, vectorized)."""
    from scipy.stats import rankdata

    return rankdata(a, method="average", axis=1)


def spearman_obs(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")  # scipy returns nan here too, with a warning
    return float(spearmanr(x, y).statistic)


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int, seed_key: list[int]) -> dict:
    """Vectorized bootstrap (resample UNITS with replacement) of Spearman ρ."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    assert n == len(y) and n >= 3, (n, len(y))
    rng = np.random.default_rng(seed_key)
    idx = rng.integers(0, n, size=(n_boot, n))
    rx = rankdata_rows(x[idx])
    ry = rankdata_rows(y[idx])
    rxc = rx - rx.mean(axis=1, keepdims=True)
    ryc = ry - ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rxc**2).sum(axis=1) * (ryc**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.where(den > 0, (rxc * ryc).sum(axis=1) / den, np.nan)
    return {
        "obs": spearman_obs(x, y),
        "n": n,
        "ci95": [_pct(draws, 2.5), _pct(draws, 97.5)],
        "draws": draws,
    }


def _pct(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).any():
        return float("nan")  # e.g. an excluded arm's all-NaN CI column
    return float(np.nanpercentile(a, q))


def _median(a) -> float:
    return float(np.median(np.asarray(a, dtype=np.float64)))


# ── shared shift-geometry engine (DV1 + DV2) ──────────────────────────


def carrier_yardstick(vc_cell: torch.Tensor, value_loc: np.ndarray) -> np.ndarray:
    """(L,) median cross-carrier SAME-VALUE distance (plan §4.3 DV1 yardstick:
    all C(12,2) carrier pairs × values per cell)."""
    dists: list[torch.Tensor] = []
    for v in range(int(value_loc.max()) + 1):
        x = vc_cell[value_loc == v].double()  # (m_v, L, H)
        if x.shape[0] < 2:
            continue
        xl = x.permute(1, 0, 2)  # (L, m, H)
        d2 = torch.cdist(xl, xl, p=2.0) ** 2  # (L, m, m)
        iu = torch.triu_indices(x.shape[0], x.shape[0], offset=1)
        dists.append(d2[:, iu[0], iu[1]].clamp_min(0).sqrt())  # (L, n_pairs_v)
    assert dists, "carrier yardstick needs ≥2 carriers per value"
    return torch.cat(dists, dim=1).median(dim=1).values.numpy()


def _consistency_unit(
    deltas: torch.Tensor,  # (m, L, H) fp64 — one (cell, vp)'s per-carrier Δ
    null_idx: list[int],
    signs: np.ndarray | None,  # (B, m) ±1 or None
    boot_idx: np.ndarray | None,  # (B, m) or None
) -> dict:
    """Observed per-layer consistency + sign-flip null + carrier bootstrap."""
    m = deltas.shape[0]
    dn = normalize_rows(deltas)
    grams = torch.einsum("alh,blh->lab", dn, dn).numpy()  # (L, m, m)
    obs = [(float(g.sum() - np.trace(g)) / (m * (m - 1))) for g in grams]
    nulls: dict[int, np.ndarray] = {}
    boots: dict[int, np.ndarray] = {}
    for li in null_idx:
        if signs is not None:
            nulls[li] = signflip_null_consistency(grams[li], signs[:, :m])
        if boot_idx is not None:
            boots[li] = bootstrap_pairwise_cosine(grams[li], boot_idx[:, :m] % m)
    return {"obs": obs, "nulls": nulls, "boots": boots, "m": m}


def shift_geometry_cell(
    v_cell: torch.Tensor,  # (n_cell_ctx, L, H) — value matrix for this cell
    cv: CellView,
    *,
    yardstick: np.ndarray,  # (L,) per-cell magnitude yardstick
    included_pair: np.ndarray,  # (n_cell_pairs,) bool
    null_idx: list[int],
    primary_idx: int,
    null_b: int,
    boot_b: int,
    seed_key: list[int],  # [seed, dv_tag, cell_index, slot_or_pool_tag]
    compute_nulls: bool = True,
) -> dict:
    """The three DV1/DV2 geometry reads for ONE cell (one slot / pooling):
    magnitude vs yardstick, within-type direction consistency (+ sign-flip
    band + carrier-bootstrap CI at the null layers), and the mean-shift
    directions consumed by the cross-type matrix."""
    vd = v_cell.double()
    deltas = vd[cv.b_loc] - vd[cv.a_loc]  # (n_pairs, L, H)
    norms = deltas.norm(dim=-1).numpy()  # (n_pairs, L)
    inc = included_pair
    med = np.median(norms[inc], axis=0) if inc.any() else np.full(norms.shape[1], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = med / yardstick
    per_vp: dict[str, dict] = {}
    vp_obs_stack: list[list[float]] = []
    null_stack: dict[int, list[np.ndarray]] = {li: [] for li in null_idx}
    boot_stack: dict[int, list[np.ndarray]] = {li: [] for li in null_idx}
    mean_dirs: dict[str, np.ndarray] = {}
    n_layers = norms.shape[1]
    for vi, vp in enumerate(cv.vps):
        sel = np.where((cv.vp_loc == vi) & inc)[0]
        if len(sel) < 2:
            per_vp[vp] = {"skipped": f"only {len(sel)} included carriers (needs ≥2)"}
            continue
        d_vp = deltas[torch.tensor(sel, dtype=torch.long)]
        m = len(sel)
        signs = boot = None
        if compute_nulls:
            rng_n = np.random.default_rng([*seed_key, 10 + vi])
            signs = rng_n.integers(0, 2, size=(null_b, m)) * 2 - 1
            rng_b = np.random.default_rng([*seed_key, 50 + vi])
            boot = rng_b.integers(0, m, size=(boot_b, m))
        unit = _consistency_unit(d_vp, null_idx if compute_nulls else [], signs, boot)
        mean_dirs[vp] = d_vp[:, primary_idx, :].mean(dim=0).numpy()
        rec: dict = {
            "n_carriers": m,
            "consistency": unit["obs"],
            "consistency_primary": unit["obs"][primary_idx],
        }
        if compute_nulls:
            rec["band95"] = {int(li): _pct(unit["nulls"][li], 95.0) for li in null_idx}
            rec["ci95"] = {
                int(li): [_pct(unit["boots"][li], 2.5), _pct(unit["boots"][li], 97.5)]
                for li in null_idx
            }
        per_vp[vp] = rec
        vp_obs_stack.append(unit["obs"])
        for li in null_idx:
            if compute_nulls:
                null_stack[li].append(unit["nulls"][li])
                boot_stack[li].append(unit["boots"][li])
    cell_cons = (
        np.mean(np.asarray(vp_obs_stack), axis=0).tolist()
        if vp_obs_stack
        else [float("nan")] * n_layers
    )
    out: dict = {
        "n_pairs": int(len(cv.pair_idx)),
        "n_included_pairs": int(inc.sum()),
        "median_norm": med.tolist(),
        "yardstick": yardstick.tolist(),
        "ratio": ratio.tolist(),
        "consistency_cell": cell_cons,
        "per_vp": per_vp,
        "mean_dirs": mean_dirs,
        "pair_norms": norms,
    }
    if compute_nulls and vp_obs_stack:
        # Cell-level draws = mean over vps of per-vp draws (draw-aligned).
        out["cell_null"] = {li: np.mean(np.stack(null_stack[li]), axis=0) for li in null_idx}
        out["cell_boot"] = {li: np.nanmean(np.stack(boot_stack[li]), axis=0) for li in null_idx}
    return out


def cross_type_geometry(
    mean_dirs: dict[tuple[str, str], np.ndarray],
) -> dict:
    """(cell, vp) mean-shift cosine matrix + within/between-cell |cos| summary."""
    labels = sorted(mean_dirs)
    u = np.stack([mean_dirs[k] for k in labels]).astype(np.float64)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), _EPS)
    c = u @ u.T
    cells = np.array([k[0] for k in labels])
    same = cells[:, None] == cells[None, :]
    off = ~np.eye(len(labels), dtype=bool)
    within = np.abs(c[same & off])
    between = np.abs(c[~same])
    return {
        "labels": [list(k) for k in labels],
        "matrix": np.round(c, 6).tolist(),
        "within_cell_mean_abs_cos": float(within.mean()) if within.size else None,
        "between_cell_mean_abs_cos": float(between.mean()) if between.size else None,
    }


# ── DV1: context-vector shift ─────────────────────────────────────────


def compute_dv1(
    vc: dict,
    pt: PairTable,
    views: dict[str, CellView],
    cells_meta: dict,
    degenerate_pe: set[str],
    *,
    null_b: int,
    boot_b: int,
    nulls_out: dict[str, np.ndarray],
) -> dict:
    """DV1 (plan §4.3): Δv_C magnitude vs carrier yardstick, within-type
    consistency vs the label-permutation band, cross-type geometry — per
    slot ∈ {ce, pe}, all layers observed, bands/CIs at the map layers."""
    layers = vc["layers"]
    primary_idx, realized_primary = _primary(layers)
    null_idx = _null_layer_idxs(layers, primary_idx)
    per_cell: dict[str, dict] = {}
    cross: dict[str, dict] = {}
    per_pair_rows: list[dict] = []
    cell_primary: dict[str, dict] = {}
    for slot_i, slot in enumerate(("ce", "pe")):
        v = vc[slot]
        mean_dirs: dict[tuple[str, str], np.ndarray] = {}
        for ci, cell in enumerate(pt.cells):
            cv = views[cell]
            v_cell = v[torch.tensor(cv.ctx_rows, dtype=torch.long)]
            yard = carrier_yardstick(v_cell, cv.value_loc)
            degenerate = slot == "pe" and cell in degenerate_pe
            if degenerate:
                rec = _dv1_degenerate_pe(v_cell, cv, yard, primary_idx)
            else:
                geo = shift_geometry_cell(
                    v_cell,
                    cv,
                    yardstick=yard,
                    included_pair=np.ones(len(cv.pair_idx), dtype=bool),
                    null_idx=null_idx,
                    primary_idx=primary_idx,
                    null_b=null_b,
                    boot_b=boot_b,
                    seed_key=[SEED_NULL, 1, ci, slot_i],
                )
                for vp, d in geo.pop("mean_dirs").items():
                    mean_dirs[(cell, vp)] = d
                for li, draws in geo.pop("cell_null", {}).items():
                    nulls_out[f"dv1|{slot}|{cell}|L{layers[li]}|null"] = draws.astype(np.float32)
                for li, draws in geo.pop("cell_boot", {}).items():
                    nulls_out[f"dv1|{slot}|{cell}|L{layers[li]}|boot"] = draws.astype(np.float32)
                rec = _finalize_geo_record(geo, layers, primary_idx, null_idx)
            rec["degenerate_at_pe"] = degenerate
            per_cell.setdefault(cell, {})[slot] = rec
            cell_primary.setdefault(cell, {})[f"median_norm_{slot}"] = rec["primary"].get(
                "median_norm"
            )
            norms = rec.pop("pair_norms", None)
            if norms is not None:
                for j, k in enumerate(cv.pair_idx):
                    row = _pair_row_base(pt, int(k))
                    row.update(
                        {
                            "slot": slot,
                            "layer": realized_primary,
                            "norm_dvc": float(norms[j, primary_idx]),
                            "cell_yardstick": float(yard[primary_idx]),
                            "degenerate_at_pe": degenerate,
                        }
                    )
                    per_pair_rows.append(row)
        cross[slot] = cross_type_geometry(mean_dirs) if mean_dirs else {"skipped": "no cells"}
        logger.info("[dv1] slot=%s done (%d cells)", slot, len(pt.cells))
    aggregates = _dv1_aggregates(per_cell, cells_meta, degenerate_pe, realized_primary)
    return {
        "meta": _band_meta(null_b, boot_b, realized_primary, [layers[i] for i in null_idx]),
        "layers": layers,
        "per_cell": per_cell,
        "cross_type": cross,
        "aggregates": aggregates,
        "per_pair_rows": per_pair_rows,
        "cell_primary": cell_primary,
    }


def _dv1_degenerate_pe(
    v_cell: torch.Tensor, cv: CellView, yard: np.ndarray, primary_idx: int
) -> dict:
    """Sanity for the 2 pre-declared degenerate-at-pe cells: per-pair
    flattened pe-state cosine ≥ 0.99 (parent state-sanity band), ratios
    recorded informationally, cell EXCLUDED from pe aggregates/nulls."""
    vd = v_cell.double()
    a = vd[cv.a_loc].reshape(len(cv.a_loc), -1)
    b = vd[cv.b_loc].reshape(len(cv.b_loc), -1)
    cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(_EPS)
    bad = (cos < DEGENERATE_PE_COS_MIN).nonzero().flatten().tolist()
    assert not bad, (
        f"{cv.cell}: {len(bad)} degenerate-at-pe pair(s) with pe-state cosine < "
        f"{DEGENERATE_PE_COS_MIN} (min {float(cos.min()):.6f}) — capture-side misalignment, "
        "stop-and-diagnose (parent state-sanity band)"
    )
    deltas = vd[cv.b_loc] - vd[cv.a_loc]
    norms = deltas.norm(dim=-1).numpy()
    med = np.median(norms, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = med / yard
    return {
        "excluded": "degenerate at pe (pre-declared) — sanity PASS, excluded from pe aggregates",
        "sanity_min_pair_cos": float(cos.min()),
        "median_norm": med.tolist(),
        "yardstick": yard.tolist(),
        "ratio": ratio.tolist(),
        "primary": {"median_norm": float(med[primary_idx]), "ratio": float(ratio[primary_idx])},
        "pair_norms": norms,
    }


def _finalize_geo_record(geo: dict, layers: list[int], primary_idx: int, null_idx) -> dict:
    """Attach the primary-layer summary (+ H1-style flags) to a cell record."""
    ratio_p = geo["ratio"][primary_idx]
    cons_p = geo["consistency_cell"][primary_idx]
    band = ci = None
    per_vp = geo["per_vp"]
    vp_bands = [
        r["band95"][primary_idx] for r in per_vp.values() if isinstance(r.get("band95"), dict)
    ]
    vp_cis = [r["ci95"][primary_idx] for r in per_vp.values() if isinstance(r.get("ci95"), dict)]
    if vp_bands:
        band = float(np.mean(vp_bands))  # cell-level band = mean of vp bands (draw-aligned mean)
    if vp_cis:
        ci = [float(np.mean([c[0] for c in vp_cis])), float(np.mean([c[1] for c in vp_cis]))]
    geo["primary"] = {
        "layer": layers[primary_idx],
        "median_norm": float(geo["median_norm"][primary_idx]),
        "ratio": float(ratio_p),
        "consistency": float(cons_p),
        "consistency_ci95": ci,
        "band95": band,
        "ratio_gt1": bool(ratio_p > 1.0),
        "consistency_ci_excludes_band": bool(ci is not None and band is not None and ci[0] > band),
    }
    return geo


def _dv1_aggregates(per_cell, cells_meta, degenerate_pe, realized_primary) -> dict:
    out: dict[str, dict] = {}
    for slot in ("ce", "pe"):
        rows = {
            cell: rec[slot]
            for cell, rec in per_cell.items()
            if "primary" in rec[slot] and not (slot == "pe" and cell in degenerate_pe)
        }
        base_rows = {
            c: r for c, r in rows.items() if cells_meta.get(c, {}).get("base_type", c) == c
        }
        out[slot] = {
            "primary_layer": realized_primary,
            "n_cells": len(rows),
            "n_ratio_gt1": sum(r["primary"]["ratio_gt1"] for r in rows.values()),
            "n_consistency_ci_excludes_band": sum(
                r["primary"]["consistency_ci_excludes_band"] for r in rows.values()
            ),
            "base_types": {
                "n": len(base_rows),
                "n_ratio_gt1": sum(r["primary"]["ratio_gt1"] for r in base_rows.values()),
                "n_consistency_ci_excludes_band": sum(
                    r["primary"]["consistency_ci_excludes_band"] for r in base_rows.values()
                ),
            },
        }
    return out


# ── DV2: answer-vector shift ──────────────────────────────────────────


def compute_dv2(
    ans: AnswerMeans,
    pt: PairTable,
    views: dict[str, CellView],
    included_pair: np.ndarray,
    *,
    null_b: int,
    boot_b: int,
    nulls_out: dict[str, np.ndarray],
) -> dict:
    """DV2 (plan §4.3): same geometry reads over v̄_A with the split-half
    draw-noise yardstick, plus the split-half-leg consistency companion and
    the same-carrier split-half reliability."""
    layers = ans.layers
    primary_idx, realized_primary = _primary(layers)
    null_idx = _null_layer_idxs(layers, primary_idx)
    valid_split = (ans.n_valid >= 4) & (ans.n_h1 >= 1) & (ans.n_h2 >= 1)
    per_cell: dict[str, dict] = {}
    per_pair_rows: list[dict] = []
    cell_primary: dict[str, dict] = {}
    cross: dict[str, dict] = {}
    for pool_i, pool in enumerate(POOLINGS):
        v = ans.mean[pool]
        h1, h2 = ans.half1[pool], ans.half2[pool]
        mean_dirs: dict[tuple[str, str], np.ndarray] = {}
        for ci, cell in enumerate(pt.cells):
            cv = views[cell]
            rows_t = torch.tensor(cv.ctx_rows, dtype=torch.long)
            split_rows = cv.ctx_rows[valid_split[cv.ctx_rows]]
            n_split = len(split_rows)
            if n_split:
                st = torch.tensor(split_rows, dtype=torch.long)
                sh_dist = (h1[st] - h2[st]).norm(dim=-1)  # (n_split, L)
                yard = sh_dist.median(dim=0).values.numpy()
            else:
                yard = np.full(len(layers), np.nan)
            inc = included_pair[cv.pair_idx]
            geo = shift_geometry_cell(
                v[rows_t],
                cv,
                yardstick=yard,
                included_pair=inc,
                null_idx=null_idx,
                primary_idx=primary_idx,
                null_b=null_b,
                boot_b=boot_b,
                seed_key=[SEED_NULL, 2, ci, pool_i],
            )
            for vp, d in geo.pop("mean_dirs").items():
                mean_dirs[(cell, vp)] = d
            for li, draws in geo.pop("cell_null", {}).items():
                nulls_out[f"dv2|{pool}|{cell}|L{layers[li]}|null"] = draws.astype(np.float32)
            for li, draws in geo.pop("cell_boot", {}).items():
                nulls_out[f"dv2|{pool}|{cell}|L{layers[li]}|boot"] = draws.astype(np.float32)
            rec = _finalize_geo_record(geo, layers, primary_idx, null_idx)
            rec["n_split_contexts"] = n_split
            rec["n_flagged_below_4_valid"] = int((~valid_split[cv.ctx_rows]).sum())
            rec["split_half"] = _split_half_companion(
                h1[rows_t], h2[rows_t], cv, inc, valid_split[cv.ctx_rows], primary_idx
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                rec["noise_normalized_primary"] = float(
                    rec["primary"]["median_norm"] / yard[primary_idx]
                )
            per_cell.setdefault(cell, {})[pool] = rec
            norms = rec.pop("pair_norms")
            if pool == POOL_PRIMARY:
                cell_primary[cell] = {
                    "median_norm": rec["primary"]["median_norm"],
                    "noise_normalized": rec["noise_normalized_primary"],
                    "yardstick": float(yard[primary_idx]),
                }
                for j, k in enumerate(cv.pair_idx):
                    row = _pair_row_base(pt, int(k))
                    gk = int(k)
                    row.update(
                        {
                            "pooling": pool,
                            "layer": realized_primary,
                            "norm_dva": float(norms[j, primary_idx]),
                            "cell_splithalf_floor": float(yard[primary_idx]),
                            "n_valid_a": int(ans.n_valid[pt.a_row[gk]]),
                            "n_valid_b": int(ans.n_valid[pt.b_row[gk]]),
                            "included": bool(included_pair[gk]),
                        }
                    )
                    per_pair_rows.append(row)
        cross[pool] = cross_type_geometry(mean_dirs) if mean_dirs else {"skipped": "no cells"}
        logger.info("[dv2] pooling=%s done (%d cells)", pool, len(pt.cells))
    return {
        "meta": {
            **_band_meta(null_b, boot_b, realized_primary, [layers[i] for i in null_idx]),
            "span_source": ans.span_source,
            "primary_pooling": POOL_PRIMARY,
            "split_half_rule": "draws {0..k/2-1} vs {k/2..k-1}; contexts with n_valid<4 "
            "flagged out of the split-half denominator (kept in the mean)",
        },
        "layers": layers,
        "per_cell": per_cell,
        "cross_type": cross,
        "per_pair_rows": per_pair_rows,
        "cell_primary": cell_primary,
        "n_valid_zero_contexts": int((ans.n_valid == 0).sum()),
    }


def _split_half_companion(
    h1_cell: torch.Tensor,
    h2_cell: torch.Tensor,
    cv: CellView,
    inc: np.ndarray,
    split_ok_local: np.ndarray,
    primary_idx: int,
) -> dict:
    """Split-half-leg consistency (leg1 = Δ from half1, leg2 = Δ from half2)
    + same-carrier split-half reliability, at the primary layer."""
    out: dict[str, dict] = {}
    for vi, vp in enumerate(cv.vps):
        sel = np.where((cv.vp_loc == vi) & inc)[0]
        sel = np.array(
            [j for j in sel if split_ok_local[cv.a_loc[j]] and split_ok_local[cv.b_loc[j]]],
            dtype=np.int64,
        )
        if len(sel) < 2:
            out[vp] = {"skipped": f"only {len(sel)} split-valid carriers (needs ≥2)"}
            continue
        a = torch.tensor(cv.a_loc[sel], dtype=torch.long)
        b = torch.tensor(cv.b_loc[sel], dtype=torch.long)
        d1 = normalize_rows((h1_cell[b] - h1_cell[a])[:, primary_idx, :])
        d2 = normalize_rows((h2_cell[b] - h2_cell[a])[:, primary_idx, :])
        c = (d1 @ d2.T).numpy()
        m = len(sel)
        out[vp] = {
            "n_carriers": m,
            "cross_half_consistency": float((c.sum() - np.trace(c)) / (m * (m - 1))),
            "same_carrier_reliability": float(np.trace(c) / m),
        }
    vals = [r for r in out.values() if "cross_half_consistency" in r]
    return {
        "per_vp": out,
        "cell_cross_half_consistency": (
            float(np.mean([r["cross_half_consistency"] for r in vals])) if vals else None
        ),
        "cell_same_carrier_reliability": (
            float(np.mean([r["same_carrier_reliability"] for r in vals])) if vals else None
        ),
    }


# ── H2 coupling + exploratory couplings ───────────────────────────────


def compute_coupling(
    dv1: dict,
    dv2: dict,
    pt: PairTable,
    parent_sep: dict[str, float] | None,
    *,
    boot_b: int,
    nulls_out: dict[str, np.ndarray],
) -> dict:
    """H2: Spearman(per-cell noise-normalized ‖Δv_A‖, parent anchor
    separation), bootstrap CI over cells; plus exploratory
    Spearman(‖Δv_C‖, ‖Δv_A‖) across cells and across pairs within cell."""
    out: dict = {"h2": None, "exploratory": {}}
    dv2_cells = dv2["cell_primary"]
    if parent_sep is None:
        out["h2"] = {"skipped": "no parent anchors.jsonl supplied (recorded, never silent)"}
    else:
        cells = sorted(
            c
            for c in dv2_cells
            if c in parent_sep and np.isfinite(dv2_cells[c]["noise_normalized"])
        )
        missing_parent = sorted(set(dv2_cells) - set(parent_sep))
        if len(cells) < 3:
            out["h2"] = {
                "skipped": f"only {len(cells)} overlapping cells (needs ≥3 — smoke slice?)",
                "cells_without_parent_rows": missing_parent,
            }
        else:
            x = np.array([dv2_cells[c]["noise_normalized"] for c in cells])
            y = np.array([parent_sep[c] for c in cells])
            bs = bootstrap_spearman(x, y, boot_b, [SEED_BOOT, 7])
            nulls_out["h2|spearman_boot"] = bs.pop("draws").astype(np.float32)
            out["h2"] = {
                **bs,
                "cells": cells,
                "cells_without_parent_rows": missing_parent,
                "x": "per-cell noise-normalized median ‖Δv_A‖ (tail, primary layer)",
                "y": "parent per-cell mean anchor separation (f_metrics/anchors.jsonl)",
                # Persisted for the H2 figure (unit 3): issue2215_figures reads
                # these values verbatim — no recomputation at render time.
                "per_cell_xy": {
                    c: {"x": float(xi), "y": float(yi)} for c, xi, yi in zip(cells, x, y)
                },
            }
    # Exploratory: across cells (Δv_C at ce vs Δv_A, both at their primary layer).
    dv1_cells = dv1["cell_primary"]
    common = sorted(set(dv1_cells) & set(dv2_cells))
    if len(common) >= 3:
        x = np.array([dv1_cells[c]["median_norm_ce"] for c in common], dtype=np.float64)
        y = np.array([dv2_cells[c]["median_norm"] for c in common], dtype=np.float64)
        bs = bootstrap_spearman(x, y, boot_b, [SEED_BOOT, 8])
        nulls_out["coupling|dvc_dva_cells_boot"] = bs.pop("draws").astype(np.float32)
        out["exploratory"]["across_cells"] = {**bs, "n_cells": len(common)}
    else:
        out["exploratory"]["across_cells"] = {"skipped": f"{len(common)} cells < 3"}
    # Within-cell across pairs (per-pair ‖Δv_C‖ ce vs ‖Δv_A‖ tail at primary).
    dv1_rows = {(r["pair_id"]): r["norm_dvc"] for r in dv1["per_pair_rows"] if r["slot"] == "ce"}
    per_cell_rho: dict[str, float] = {}
    by_cell: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in dv2["per_pair_rows"]:
        if r["included"] and r["pair_id"] in dv1_rows:
            by_cell[r["cell"]].append((dv1_rows[r["pair_id"]], r["norm_dva"]))
    for cell, xy in sorted(by_cell.items()):
        if len(xy) >= 3:
            arr = np.asarray(xy, dtype=np.float64)
            per_cell_rho[cell] = spearman_obs(arr[:, 0], arr[:, 1])
    out["exploratory"]["within_cell_across_pairs"] = {
        "per_cell_rho": per_cell_rho,
        "median_rho": _median(list(per_cell_rho.values())) if per_cell_rho else None,
    }
    return out


# ── DV3: map discrimination ───────────────────────────────────────────


def observed_2afc(s: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair 2AFC margins from a context×context similarity block:
    dir-A margin = S[a,a] − S[a,b]; dir-B margin = S[b,b] − S[b,a]."""
    return s[a, a] - s[a, b], s[b, b] - s[b, a]


def null_2afc_cell(
    s: np.ndarray,
    cv: CellView,
    sigma: np.ndarray,  # (B, n_carriers) deranged carrier permutations
    side_a: np.ndarray,  # (B, n_pairs) bool — side randomization per direction
    side_b: np.ndarray,
    valid_pair: np.ndarray,  # (n_pairs,) bool
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffled-pair null (plan §4.3): each prediction scored against a
    carrier-blocked DERANGED pair's target duo (same value-pair, different
    carrier) with the side label randomized — mean 0.5 by construction.
    Returns (correct counts per draw, comparison counts per draw)."""
    q = cv.pair_at[sigma[:, cv.carrier_loc], cv.vp_loc]  # (B, n_pairs) local pair idx
    aq, bq = cv.a_loc[q], cv.b_loc[q]
    a_row = cv.a_loc[None, :]
    b_row = cv.b_loc[None, :]
    mask = valid_pair[None, :] & valid_pair[q]
    own_a = np.where(side_a, bq, aq)
    oth_a = np.where(side_a, aq, bq)
    m_a = s[a_row, own_a] - s[a_row, oth_a]
    own_b = np.where(side_b, aq, bq)
    oth_b = np.where(side_b, bq, aq)
    m_b = s[b_row, own_b] - s[b_row, oth_b]
    correct = ((m_a > 0) & mask).sum(axis=1) + ((m_b > 0) & mask).sum(axis=1)
    return correct.astype(np.float64), (2 * mask.sum(axis=1)).astype(np.float64)


def carrier_transfer_cell(s: np.ndarray, cv: CellView, valid_pair: np.ndarray) -> dict:
    """Exploratory carrier-transfer (plan §4.3): score pair p's prediction
    against SAME value-pair targets at OTHER carriers (no side flip)."""
    own_bits: list[np.ndarray] = []
    cross_bits: list[np.ndarray] = []
    for vi in range(len(cv.vps)):
        grp = np.where((cv.vp_loc == vi) & valid_pair)[0]
        if len(grp) < 2:
            continue
        a, b = cv.a_loc[grp], cv.b_loc[grp]
        m_a = s[np.ix_(a, a)] - s[np.ix_(a, b)]  # (m, m): pred a_i vs duo of pair j
        m_b = s[np.ix_(b, b)] - s[np.ix_(b, a)]
        off = ~np.eye(len(grp), dtype=bool)
        own_bits.append(np.concatenate([np.diag(m_a) > 0, np.diag(m_b) > 0]))
        cross_bits.append(np.concatenate([(m_a > 0)[off], (m_b > 0)[off]]))
    if not own_bits:
        return {"skipped": "no vp with ≥2 valid carriers"}
    return {
        "own_pair_acc": float(np.concatenate(own_bits).mean()),
        "cross_carrier_acc": float(np.concatenate(cross_bits).mean()),
        "n_own": int(sum(len(x) for x in own_bits)),
        "n_cross": int(sum(len(x) for x in cross_bits)),
    }


def sim_blocks(p: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    """Cosine + negative-squared-euclidean similarity blocks (fp64).

    Negative SQUARED euclidean is rank-equivalent to negative euclidean for
    every pairwise comparison (monotone transform), matching the
    mapping_baselines euclidean convention.
    """
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    pn = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), _EPS)
    tn = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), _EPS)
    sq = (p**2).sum(1)[:, None] + (t**2).sum(1)[None, :] - 2.0 * (p @ t.T)
    return {"cosine": pn @ tn.T, "euclidean": -sq}


def idbias_loto_predict(
    x: np.ndarray, t: np.ndarray, cell_of_row: list[str], valid: np.ndarray
) -> np.ndarray:
    """Identity+learned-bias baseline with leave-one-TYPE-out b (plan §4.3):
    for each cell, b is fit on ALL OTHER cells' valid contexts via the
    canonical ``mapping_baselines.identity_bias_predict``."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    cells = np.asarray(cell_of_row)
    pred = np.zeros_like(x)
    for cell in sorted(set(cell_of_row)):
        in_cell = cells == cell
        train = valid & ~in_cell
        assert train.sum() >= 1, f"idbias LOTO: no train rows outside cell {cell}"
        pred[in_cell] = identity_bias_predict(x[train], t[train], x[in_cell])
    return pred


def pooled_r2_cos(p: np.ndarray, t: np.ndarray) -> dict:
    """Pooled R² (1 − ‖P−T‖²_F / ‖T−T̄‖²_F) + per-row mean cosine."""
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    ss_res = float(((p - t) ** 2).sum())
    ss_tot = float(((t - t.mean(axis=0)) ** 2).sum())
    pn = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), _EPS)
    tn = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), _EPS)
    return {
        "r2_pooled": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mean_cosine": float((pn * tn).sum(axis=1).mean()),
        "n": int(p.shape[0]),
    }


def discrimination_verdict(acc: float, ci: list[float] | None) -> str:
    """Registered verdict lattice (plan §3) over Δacc = acc − 0.5."""
    if ci is None or not all(np.isfinite(ci)):
        return "inconclusive"
    if acc > 0.5 and ci[0] > 0.5:
        return "discriminates"
    if ci[1] < 0.5:
        return "fails-to-discriminate"
    return "inconclusive"


def compute_dv3(
    vc: dict,
    ans: AnswerMeans,
    pt: PairTable,
    views: dict[str, CellView],
    arm_specs: list[dict],
    degenerate_pe: set[str],
    included_pair: np.ndarray,
    *,
    null_b: int,
    boot_b: int,
    nulls_out: dict[str, np.ndarray],
) -> dict:
    """DV3 (plan §4.3): paired 2AFC accuracy + margins for the three fitted
    ridge arms and the two identity+bias baseline arms, against the
    shuffled-pair null, with carrier-clustered bootstrap CIs, kNN retrieval,
    pooled R²/cosine, and the carrier-transfer decomposition."""
    import issue779_ffc_n1m_fits as FITS  # deferred heavy sibling import

    from issue2094_analysis import bootstrap_family_means_batched  # deferred

    vc_layers, va_layers = vc["layers"], ans.layers
    dv3_layers = sorted({int(layer) for spec in arm_specs for layer in spec["paths"]})
    dv3_layers = [layer for layer in dv3_layers if layer in vc_layers and layer in va_layers]
    assert dv3_layers, (vc_layers, va_layers, "no usable DV3 layers")
    primary_layer = PRIMARY_LAYER if PRIMARY_LAYER in dv3_layers else dv3_layers[0]
    valid = ans.n_valid > 0
    n_ctx = len(pt.ids)
    dev = torch.device("cpu")

    # Fitted-arm predictions per (arm, layer) — shared across poolings.
    preds: dict[tuple[str, int], np.ndarray] = {}
    for spec in arm_specs:
        for layer, path in spec["paths"].items():
            if layer not in dv3_layers:
                continue
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
            assert payload.get("kind") == "ridge", (spec["arm"], layer, payload.get("kind"))
            if "layer" in payload:
                assert int(payload["layer"]) == int(layer), (spec["arm"], payload["layer"], layer)
            x = vc[spec["slot"]][:, vc_layers.index(layer), :].double().numpy()
            p = FITS.apply_map(payload, x, dev)
            assert p.shape == x.shape, (p.shape, x.shape)
            preds[(spec["arm"], layer)] = p
            del payload

    arms = [{"arm": s["arm"], "slot": s["slot"], "fitted": True} for s in arm_specs]
    for slot in sorted({s["slot"] for s in arm_specs}):
        arms.append({"arm": f"idbias_{slot}", "slot": slot, "fitted": False})

    # Shared per-cell null draws (seed 2215; one plan across configs so bands
    # are comparable and generation is paid once).
    cell_draws: dict[str, dict] = {}
    for ci, cell in enumerate(pt.cells):
        cv = views[cell]
        rng = np.random.default_rng([SEED_NULL, 3, ci])
        n_p = len(cv.pair_idx)
        cell_draws[cell] = {
            "sigma": deranged_perms(len(cv.carriers), null_b, rng),
            "side_a": rng.integers(0, 2, size=(null_b, n_p)).astype(bool),
            "side_b": rng.integers(0, 2, size=(null_b, n_p)).astype(bool),
        }

    # Cluster frame for the carrier-clustered bootstrap. NOTE: rev-cell
    # clusters share their underlying contexts with the fwd twins' clusters
    # (borrowed membership) — a bank-by-construction non-independence.
    clusters: list[tuple[str, str]] = [
        (cell, carrier) for cell in pt.cells for carrier in views[cell].carriers
    ]
    cluster_index = {kc: i for i, kc in enumerate(clusters)}

    configs = [
        (a["arm"], layer, pool, metric)
        for a in arms
        for layer in dv3_layers
        for pool in POOLINGS
        for metric in METRICS
    ]
    cfg_index = {c: i for i, c in enumerate(configs)}
    acc_cl = np.full((len(clusters), len(configs)), np.nan)
    margin_cl = np.full((len(clusters), len(configs)), np.nan)
    per_config: dict[str, dict] = {}
    per_pair_rows: list[dict] = []
    t0 = time.monotonic()
    unit = 0
    n_units = len(arms) * len(dv3_layers) * len(POOLINGS)
    for arm in arms:
        arm_name, slot = arm["arm"], arm["slot"]
        excluded_cells = set(degenerate_pe) if slot == "pe" else set()
        for layer in dv3_layers:
            x_np = vc[slot][:, vc_layers.index(layer), :].double().numpy()
            li_va = va_layers.index(layer)
            for pool in POOLINGS:
                unit += 1
                t_np = ans.mean[pool][:, li_va, :].numpy()
                if arm["fitted"]:
                    p_np = preds[(arm_name, layer)]
                else:
                    p_np = idbias_loto_predict(x_np, t_np, pt.cell_of, valid)
                key = f"{arm_name}|L{layer}|{pool}"
                stats = pooled_r2_cos(p_np[valid], t_np[valid])
                knn = {
                    metric: knn_retrieval(p_np[valid], t_np[valid], ks=(1, 5, 10), metric=metric)
                    for metric in METRICS
                }
                cell_rows: dict[str, dict] = {}
                pooled_bits = {m: [0.0, 0.0] for m in METRICS}  # correct, total
                null_correct = {m: np.zeros(null_b) for m in METRICS}
                null_total = {m: np.zeros(null_b) for m in METRICS}
                transfer: dict[str, dict] = {}
                for cell in pt.cells:
                    cv = views[cell]
                    if cell in excluded_cells:
                        cell_rows[cell] = {"na": "N/A — degenerate at pe"}
                        continue
                    loc = cv.ctx_rows
                    s_by_metric = sim_blocks(p_np[loc], t_np[loc])
                    vp_valid = included_pair[cv.pair_idx] & valid[loc][cv.a_loc]
                    vp_valid &= valid[loc][cv.b_loc]
                    if not vp_valid.any():
                        cell_rows[cell] = {"na": "N/A — all pairs excluded (n_valid=0 sides)"}
                        continue
                    draws = cell_draws[cell]
                    margins_by_metric = {
                        metric: observed_2afc(s_by_metric[metric], cv.a_loc, cv.b_loc)
                        for metric in METRICS
                    }
                    for metric in METRICS:
                        s = s_by_metric[metric]
                        m_a, m_b = margins_by_metric[metric]
                        bits = np.concatenate([(m_a > 0)[vp_valid], (m_b > 0)[vp_valid]])
                        margins = np.concatenate([m_a[vp_valid], m_b[vp_valid]])
                        cfg_i = cfg_index[(arm_name, layer, pool, metric)]
                        for car_i, carrier in enumerate(cv.carriers):
                            sel = (cv.carrier_loc == car_i) & vp_valid
                            if not sel.any():
                                continue
                            cbits = np.concatenate([(m_a > 0)[sel], (m_b > 0)[sel]])
                            cmarg = np.concatenate([m_a[sel], m_b[sel]])
                            row = cluster_index[(cell, carrier)]
                            acc_cl[row, cfg_i] = float(cbits.mean())
                            margin_cl[row, cfg_i] = float(cmarg.mean())
                        nc, nt = null_2afc_cell(
                            s, cv, draws["sigma"], draws["side_a"], draws["side_b"], vp_valid
                        )
                        null_correct[metric] += nc
                        null_total[metric] += nt
                        with np.errstate(invalid="ignore", divide="ignore"):
                            cell_null_acc = np.where(nt > 0, nc / nt, np.nan)
                        nulls_out[f"dv3|{key}|{metric}|{cell}|null"] = cell_null_acc.astype(
                            np.float32
                        )
                        cell_rows.setdefault(cell, {})[metric] = {
                            "acc": float(bits.mean()),
                            "mean_margin": float(margins.mean()),
                            "n_pairs_included": int(vp_valid.sum()),
                            "n_excluded": int((~vp_valid).sum()),
                            "null_band": [_pct(cell_null_acc, 2.5), _pct(cell_null_acc, 97.5)],
                        }
                        pooled_bits[metric][0] += float(bits.sum())
                        pooled_bits[metric][1] += float(len(bits))
                        if metric == "cosine" and layer == primary_layer and pool == POOL_PRIMARY:
                            transfer[cell] = carrier_transfer_cell(s, cv, vp_valid)
                            for j_local, k in enumerate(cv.pair_idx):
                                if not vp_valid[j_local]:
                                    continue
                                row = _pair_row_base(pt, int(k))
                                row.update(
                                    {
                                        "arm": arm_name,
                                        "layer": layer,
                                        "pooling": pool,
                                        "margin_cos_a": float(m_a[j_local]),
                                        "margin_cos_b": float(m_b[j_local]),
                                        "correct_cos_a": bool(m_a[j_local] > 0),
                                        "correct_cos_b": bool(m_b[j_local] > 0),
                                        "margin_euc_a": float(
                                            margins_by_metric["euclidean"][0][j_local]
                                        ),
                                        "margin_euc_b": float(
                                            margins_by_metric["euclidean"][1][j_local]
                                        ),
                                    }
                                )
                                per_pair_rows.append(row)
                pooled = {}
                for metric in METRICS:
                    correct, total = pooled_bits[metric]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        null_acc = np.where(
                            null_total[metric] > 0,
                            null_correct[metric] / null_total[metric],
                            np.nan,
                        )
                    nulls_out[f"dv3|{key}|{metric}|__pooled__|null"] = null_acc.astype(np.float32)
                    pooled[metric] = {
                        "acc": correct / total if total else float("nan"),
                        "n_pair_dirs": int(total),
                        "null_band": [_pct(null_acc, 2.5), _pct(null_acc, 97.5)],
                    }
                per_config[key] = {
                    "arm": arm_name,
                    "slot": slot,
                    "layer": layer,
                    "pooling": pool,
                    "fitted": arm["fitted"],
                    **stats,
                    "knn": knn,
                    "pooled": pooled,
                    "per_type": cell_rows,
                    **({"carrier_transfer": transfer} if transfer else {}),
                }
                logger.info(
                    "[dv3] unit %d/%d %s elapsed=%.0fs",
                    unit,
                    n_units,
                    key,
                    time.monotonic() - t0,
                )

    # Carrier-clustered bootstrap CIs: pooled (all clusters) + per-cell rows.
    diff_cols: list[tuple[str, int]] = []  # (label, fitted cfg idx paired w/ baseline idx)
    diff_vals: list[np.ndarray] = []
    for arm in arms:
        if not arm["fitted"]:
            continue
        base = f"idbias_{arm['slot']}"
        for layer in dv3_layers:
            for pool in POOLINGS:
                for metric in METRICS:
                    fi = cfg_index[(arm["arm"], layer, pool, metric)]
                    bi = cfg_index[(base, layer, pool, metric)]
                    diff_cols.append((f"{arm['arm']}-minus-{base}|L{layer}|{pool}|{metric}", fi))
                    diff_vals.append(acc_cl[:, fi] - acc_cl[:, bi])
    families = np.concatenate(
        [acc_cl, margin_cl] + ([np.stack(diff_vals, axis=1)] if diff_vals else []), axis=1
    )
    pooled_draws = bootstrap_family_means_batched(families, boot_b, SEED_BOOT)
    n_cfg = len(configs)
    nulls_out["dv3|cluster_acc_values"] = acc_cl.astype(np.float32)
    nulls_out["dv3|boot_pooled_acc"] = pooled_draws[:, :n_cfg].astype(np.float32)
    per_cell_ci: dict[str, dict[str, list[float]]] = {}
    for ci_i, cell in enumerate(pt.cells):
        cv = views[cell]
        rows = [cluster_index[(cell, carrier)] for carrier in cv.carriers]
        vals = acc_cl[np.asarray(rows)]
        if np.isnan(vals).all():
            continue
        draws = bootstrap_family_means_batched(vals, boot_b, SEED_BOOT + 1 + ci_i)
        per_cell_ci[cell] = {
            f"{c[0]}|L{c[1]}|{c[2]}|{c[3]}": [_pct(draws[:, k], 2.5), _pct(draws[:, k], 97.5)]
            for k, c in enumerate(configs)
        }
    # Attach CIs + verdicts.
    for k, (arm_name, layer, pool, metric) in enumerate(configs):
        rec = per_config[f"{arm_name}|L{layer}|{pool}"]
        ci95 = [_pct(pooled_draws[:, k], 2.5), _pct(pooled_draws[:, k], 97.5)]
        m_lo = _pct(pooled_draws[:, n_cfg + k], 2.5)
        m_hi = _pct(pooled_draws[:, n_cfg + k], 97.5)
        pooled_rec = rec["pooled"][metric]
        pooled_rec["acc_ci95_clustered"] = ci95
        pooled_rec["mean_margin_ci95_clustered"] = [m_lo, m_hi]
        pooled_rec["verdict"] = discrimination_verdict(pooled_rec["acc"], ci95)
        for cell, crow in rec["per_type"].items():
            if "na" in crow or metric not in crow:
                continue
            cell_ci = per_cell_ci.get(cell, {}).get(f"{arm_name}|L{layer}|{pool}|{metric}")
            crow[metric]["acc_ci95_clustered"] = cell_ci
            crow[metric]["verdict"] = discrimination_verdict(crow[metric]["acc"], cell_ci)
    diffs: dict[str, dict] = {}
    for j, (label, _) in enumerate(diff_cols):
        col = 2 * n_cfg + j
        obs = float(np.nanmean(diff_vals[j]))
        ci95 = [_pct(pooled_draws[:, col], 2.5), _pct(pooled_draws[:, col], 97.5)]
        verdict = "inconclusive"
        if all(np.isfinite(ci95)):
            verdict = (
                "beats-baseline"
                if ci95[0] > 0
                else ("below-baseline" if ci95[1] < 0 else "inconclusive")
            )
        diffs[label] = {"mean_cluster_diff": obs, "ci95_clustered": ci95, "verdict": verdict}

    registered = {
        "config": f"L{primary_layer} | tail pooling | cosine",
        "pooled": {
            a["arm"]: {
                "acc": per_config[f"{a['arm']}|L{primary_layer}|tail"]["pooled"]["cosine"]["acc"],
                "verdict": per_config[f"{a['arm']}|L{primary_layer}|tail"]["pooled"]["cosine"][
                    "verdict"
                ],
            }
            for a in arms
        },
        "h3_diffs": {k: v for k, v in diffs.items() if f"L{primary_layer}|tail|cosine" in k},
    }
    return {
        "meta": {
            "primary_layer": primary_layer,
            "layers": dv3_layers,
            "poolings": list(POOLINGS),
            "metrics": list(METRICS),
            "arms": [{k: a[k] for k in ("arm", "slot", "fitted")} for a in arms],
            "null": "within-cell carrier-blocked derangement + side randomization "
            f"(seed {SEED_NULL}, B={null_b}); euclidean similarity = negative SQUARED "
            "euclidean (rank-equivalent)",
            "bootstrap": f"carrier-clustered (12 carriers/cell), B={boot_b}, seed {SEED_BOOT}; "
            "CIs from unweighted cluster means (observed pooled acc is pair-grain)",
            "n_valid_zero_contexts": int((~valid).sum()),
            # Row/column labels for the persisted dv3|cluster_acc_values /
            # dv3|boot_pooled_acc matrices in the null npz.
            "cluster_order": [f"{c}|{car}" for c, car in clusters],
            "config_order": [f"{c[0]}|L{c[1]}|{c[2]}|{c[3]}" for c in configs],
            "diff_order": [label for label, _ in diff_cols],
        },
        "per_config": per_config,
        "diff_vs_idbias": diffs,
        "registered": registered,
        "per_pair_rows": per_pair_rows,
    }


# ── shared small helpers ──────────────────────────────────────────────


def _primary(layers: list[int]) -> tuple[int, int]:
    """(primary index, realized primary layer): L19, or the LAST layer when
    19 is absent (tiny mode — recorded in outputs)."""
    if PRIMARY_LAYER in layers:
        return layers.index(PRIMARY_LAYER), PRIMARY_LAYER
    return len(layers) - 1, layers[-1]


def _null_layer_idxs(layers: list[int], primary_idx: int) -> list[int]:
    idxs = {layers.index(layer) for layer in MAP_LAYERS if layer in layers}
    idxs.add(primary_idx)
    return sorted(idxs)


def _pair_row_base(pt: PairTable, k: int) -> dict:
    return {
        "pair_id": pt.pair_ids[k],
        "cell": pt.pair_cell[k],
        "carrier": pt.pair_carrier[k],
        "value_pair": pt.pair_vp[k],
    }


def _band_meta(null_b: int, boot_b: int, realized_primary: int, band_layers: list[int]) -> dict:
    return {
        "null_b": null_b,
        "boot_b": boot_b,
        "seed_null": SEED_NULL,
        "seed_boot": SEED_BOOT,
        "primary_layer": realized_primary,
        "band_layers": band_layers,
        "atom_note": "sign-flip nulls have 2^m distinct atoms per unit (m carriers) — "
        "no tail p below ~1/4096 at m=12",
    }


# ── orchestrator ──────────────────────────────────────────────────────


@dataclass
class AnalysisInputs:
    """Everything Phase C needs, resolved by the driver (issue2215_run.py)."""

    bank: dict
    vc_bank_path: Path
    va_dir: Path
    banked_anchor_dir: Path | None  # None → tiny span substitution (recorded)
    arm_specs: list[dict] | None  # None → DV3 skipped (tiny; recorded)
    results_dir: Path
    null_dir: Path
    anchors_jsonl: Path | None
    cells: tuple[str, ...] | None
    null_b: int = 10_000
    boot_b: int = 10_000
    k_draws: int = 10
    repro: dict = field(default_factory=dict)


def run_analysis(inp: AnalysisInputs) -> dict:
    """Phase C entry: DV1 → DV2 → coupling → DV3 → bands, each family's
    outputs written atomically the moment it completes (checkpoint-per-DV)."""
    from issue2162_run import _write_json_atomic, _write_jsonl_atomic

    t0 = time.monotonic()
    pt = PairTable.from_bank(inp.bank, inp.cells)
    views = build_cell_views(inp.bank, pt)
    degenerate_pe = {c for c in inp.bank["degenerate_at_pe_cells"] if c in set(pt.cells)}
    cells_meta = inp.bank.get("cells", {})
    logger.info(
        "[analysis] scope: %d contexts / %d pairs / %d cells (cells slice: %s)",
        len(pt.ids),
        len(pt.pair_ids),
        len(pt.cells),
        list(inp.cells) if inp.cells else "full bank",
    )
    vc = load_vc_bank(inp.vc_bank_path, pt.ids)
    ans = load_answer_means(
        inp.va_dir, pt.ids, pt.row_of, banked_dir=inp.banked_anchor_dir, k_draws=inp.k_draws
    )
    included_pair = (ans.n_valid[pt.a_row] > 0) & (ans.n_valid[pt.b_row] > 0)
    excl_by_cell = defaultdict(int)
    for k in np.where(~included_pair)[0]:
        excl_by_cell[pt.pair_cell[int(k)]] += 1
    if excl_by_cell:
        logger.warning(
            "[analysis] excluded pairs (n_valid=0 side) per cell: %s", dict(excl_by_cell)
        )
    nulls_out: dict[str, np.ndarray] = {}
    inp.results_dir.mkdir(parents=True, exist_ok=True)
    perpair_dir = inp.results_dir / "perpair"

    dv1 = compute_dv1(
        vc,
        pt,
        views,
        cells_meta,
        degenerate_pe,
        null_b=inp.null_b,
        boot_b=inp.boot_b,
        nulls_out=nulls_out,
    )
    dv1_rows = dv1.pop("per_pair_rows")
    dv1_cellprimary = dv1.pop("cell_primary")
    _write_json_atomic(inp.results_dir / "dv1_context_shift.json", {**dv1, "repro": inp.repro})
    _write_jsonl_atomic(perpair_dir / "dv1_pairs.jsonl", dv1_rows)
    logger.info("[analysis] unit 1/5 dv1 written elapsed=%.0fs", time.monotonic() - t0)

    dv2 = compute_dv2(
        ans,
        pt,
        views,
        included_pair,
        null_b=inp.null_b,
        boot_b=inp.boot_b,
        nulls_out=nulls_out,
    )
    dv2_rows = dv2.pop("per_pair_rows")
    dv2_cellprimary = dv2["cell_primary"]
    dv2_out = {k: v for k, v in dv2.items() if k != "cell_primary"}
    _write_json_atomic(inp.results_dir / "dv2_answer_shift.json", {**dv2_out, "repro": inp.repro})
    _write_jsonl_atomic(perpair_dir / "dv2_pairs.jsonl", dv2_rows)
    logger.info("[analysis] unit 2/5 dv2 written elapsed=%.0fs", time.monotonic() - t0)

    parent_sep = None
    if inp.anchors_jsonl is not None:
        parent_sep = load_parent_separation(inp.anchors_jsonl)
    coupling = compute_coupling(
        {"per_pair_rows": dv1_rows, "cell_primary": dv1_cellprimary},
        {"per_pair_rows": dv2_rows, "cell_primary": dv2_cellprimary},
        pt,
        parent_sep,
        boot_b=inp.boot_b,
        nulls_out=nulls_out,
    )
    _write_json_atomic(inp.results_dir / "coupling.json", {**coupling, "repro": inp.repro})
    logger.info("[analysis] unit 3/5 coupling written elapsed=%.0fs", time.monotonic() - t0)

    if inp.arm_specs is None:
        dv3 = {
            "skipped": "tiny mode — staged full-H payloads are structurally incomparable "
            "with a tiny capture (DECLARED blind spot; the pod --cells smoke and the "
            "synthetic-payload unit tests cover DV3's real math)"
        }
    else:
        dv3 = compute_dv3(
            vc,
            ans,
            pt,
            views,
            inp.arm_specs,
            degenerate_pe,
            included_pair,
            null_b=inp.null_b,
            boot_b=inp.boot_b,
            nulls_out=nulls_out,
        )
        _write_jsonl_atomic(perpair_dir / "dv3_pairs.jsonl", dv3.pop("per_pair_rows"))
    _write_json_atomic(inp.results_dir / "dv3_map_discrimination.json", {**dv3, "repro": inp.repro})
    logger.info("[analysis] unit 4/5 dv3 written elapsed=%.0fs", time.monotonic() - t0)

    bands = _collect_bands(dv1, dv2, dv3)
    _write_json_atomic(inp.results_dir / "null_bands.json", {**bands, "repro": inp.repro})
    inp.null_dir.mkdir(parents=True, exist_ok=True)
    np.savez(inp.null_dir / "null_matrices.npz", **nulls_out)  # uncompressed (#813)
    _write_json_atomic(
        inp.null_dir / "null_matrices_index.json",
        {"keys": sorted(nulls_out), "n_keys": len(nulls_out), "repro": inp.repro},
    )
    logger.info(
        "[analysis] unit 5/5 bands + %d null matrices written elapsed=%.0fs",
        len(nulls_out),
        time.monotonic() - t0,
    )
    digest = {
        "n_contexts": len(pt.ids),
        "n_pairs": len(pt.pair_ids),
        "n_cells": len(pt.cells),
        "n_excluded_pairs": int((~included_pair).sum()),
        "dv1_aggregates": dv1["aggregates"],
        "dv3_registered": dv3.get("registered"),
        "h2": {k: v for k, v in (coupling.get("h2") or {}).items() if k != "draws"},
        "wall_s": time.monotonic() - t0,
    }
    return digest


def _collect_bands(dv1: dict, dv2: dict, dv3: dict) -> dict:
    """Compact null-band summary (the per-draw matrices live in the npz)."""
    out: dict = {"dv1": {}, "dv2": {}, "dv3": {}}
    for name, dv in (("dv1", dv1), ("dv2", dv2)):
        for cell, slots in dv.get("per_cell", {}).items():
            for slot, rec in slots.items():
                if (
                    isinstance(rec.get("primary"), dict)
                    and rec["primary"].get("band95") is not None
                ):
                    out[name][f"{cell}|{slot}"] = {
                        "band95": rec["primary"]["band95"],
                        "consistency": rec["primary"].get("consistency"),
                        "ci95": rec["primary"].get("consistency_ci95"),
                    }
    if "per_config" in dv3:
        for key, rec in dv3["per_config"].items():
            for metric in METRICS:
                out["dv3"][f"{key}|{metric}"] = {
                    "pooled_acc": rec["pooled"][metric]["acc"],
                    "null_band": rec["pooled"][metric]["null_band"],
                    "verdict": rec["pooled"][metric].get("verdict"),
                }
    meta = dv1.get("meta", {})
    out["meta"] = meta
    return out
