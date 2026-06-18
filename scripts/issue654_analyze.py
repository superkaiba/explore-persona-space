#!/usr/bin/env python3
"""Issue #654 Step 3+4: per-layer query-displacement metrics + figures (CPU, VM).

Plan §3 steps 3-4, §5, §6. Reads the uploaded per-pair ``.pt`` banks (no GPU,
no model). Computes, per layer:
  - per-pair centered cosine (PRIMARY DV): GLOBAL-mean-center each bank ONCE on
    its own per-layer mean over ALL pairs, L2-normalize, per-pair cosine
    (context-end vs query-end);
  - raw (uncentered) per-pair cosine alongside (anisotropy caveat);
  - shuffled-pair derangement floor (B=1000, seed 42) — both GLOBAL and
    per-tier WITHIN-TYPE; headline = matched-minus-shuffled with a 2.5/97.5 band.
    BOTH the per-tier matched cosine AND the per-tier shuffled floor consume the
    SAME globally-centered+normalized banks (centered ONCE across the full bank,
    never re-centered per tier) — matched and floor therefore subtract identical
    pre-centered tensors (plan §5 `global_mean`; concern
    per-tier-floor-centering-mismatch);
  - per-layer linear CKA(context-bank, query-bank) + a row-permuted-bank CKA floor;
  - companion SAME-POSITION contrast (plan §5): cosine of (context-only
    assistant-gen readout) vs (full-prompt assistant-gen readout) — BOTH at the
    FIXED assistant-generation slot, removing the different-token confound. The
    full-prompt readout is the per-pair ``readout`` bank captured in the SAME
    forward as context-end/query-end (concern companion-read-not-same-slot); the
    old query-end-slot companion read is gone.

Writes ``eval_results/issue_654/per_layer_displacement.json`` (headline, keyed by
context_type x layer x query_type) + per-cell breakdowns under
``eval_results/issue_654/cells/``. ``--figures`` emits the hero + exploratory
figures to ``figures/issue_654/*.png`` using the paper_plots rcParams.

Usage::

    uv run python scripts/issue654_analyze.py --banks data/issue654/dual_pos \
        --out eval_results/issue_654 --figures
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.representation_shift import linear_cka  # noqa: E402

logger = logging.getLogger("issue654_analyze")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SEED = 42
B_DERANGEMENT = 1000
ANCHOR_LAYERS = [7, 14, 21, 27]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _load_banks(banks_dir: Path) -> dict:
    """Load all per-pair .pt banks + the companion context-only readouts.

    Returns a dict with stacked per-layer tensors and the per-pair metadata.
    """
    manifest = json.loads((banks_dir / "extraction_manifest.json").read_text())
    layers = manifest["layers"]
    n_layers = len(layers)

    pair_files = sorted(banks_dir.glob("pair_*.pt"))
    if not pair_files:
        raise RuntimeError(f"no pair_*.pt banks in {banks_dir}")

    ctx_end_rows: list[torch.Tensor] = []
    qry_end_rows: list[torch.Tensor] = []
    full_readout_rows: list[torch.Tensor] = []
    meta_rows: list[dict] = []
    companion_cache: dict[str, torch.Tensor] = {}

    for pf in pair_files:
        # weights_only=True: the saved dict holds only tensors + str/int/list
        # (no custom classes), so the safe loader path is sufficient.
        d = torch.load(pf, weights_only=True)
        ctx_end_rows.append(d["context_end"])  # (n_layers, hidden)
        qry_end_rows.append(d["query_end"])
        # Companion same-slot read (plan §5): the full-prompt's assistant-gen slot,
        # captured in the SAME forward as context-end/query-end. Required so the
        # companion contrast reads the SAME position as the context-only readout
        # (concern companion-read-not-same-slot) — never A_qry (query-end slot).
        if "readout" not in d:
            raise RuntimeError(
                f"{pf.name}: missing per-pair 'readout' bank — re-run extraction with "
                f"readout_position=-1 (companion same-slot contrast, plan §5)."
            )
        full_readout_rows.append(d["readout"])
        meta_rows.append(
            {
                "pair_id": d["pair_id"],
                "context_type": d["context_type"],
                "context_id": d["context_id"],
                "query_id": d["query_id"],
                "topicality": d["topicality"],
                "length": d["length"],
                "companion_context_only_file": d["companion_context_only_file"],
            }
        )
        cid = d["context_id"]
        if cid not in companion_cache:
            cpath = banks_dir / d["companion_context_only_file"]
            cd = torch.load(cpath, weights_only=True)  # tensors + str/int/list only
            companion_cache[cid] = cd["readout"]  # (n_layers, hidden)

    A_ctx = torch.stack(ctx_end_rows).to(torch.float64)  # (n_pairs, n_layers, hidden)
    A_qry = torch.stack(qry_end_rows).to(torch.float64)
    A_readout = torch.stack(full_readout_rows).to(torch.float64)  # full-prompt assistant-gen slot
    assert A_ctx.shape == A_qry.shape, (A_ctx.shape, A_qry.shape)
    assert A_ctx.shape == A_readout.shape, (A_ctx.shape, A_readout.shape)
    assert A_ctx.shape[1] == n_layers, (A_ctx.shape, n_layers)
    logger.info("loaded %d pairs x %d layers x %d hidden", *A_ctx.shape)
    return {
        "A_ctx": A_ctx,
        "A_qry": A_qry,
        "A_readout": A_readout,
        "meta": meta_rows,
        "layers": layers,
        "n_layers": n_layers,
        "companion": companion_cache,
        "manifest": manifest,
    }


def _global_center_normalize(A: torch.Tensor) -> torch.Tensor:
    """Globally mean-center a bank ONCE (per-layer, over ALL pairs) then L2-normalize.

    The subtracted mean is the per-layer centroid over the FULL bank — never
    re-computed per tier. Returns ``(n_pairs, n_layers, hidden)`` of unit-norm
    rows. This is THE pre-centered bank that both the matched per-pair cosine and
    the shuffled-pair floor consume (plan §5 `global_mean`; concern
    per-tier-floor-centering-mismatch).
    """
    centered = A - A.mean(dim=0, keepdim=True)  # global per-layer mean, all pairs
    return torch.nn.functional.normalize(centered, dim=2)


def _centered_cos_per_layer(ctx_hat: torch.Tensor, qry_hat: torch.Tensor) -> np.ndarray:
    """Per-pair centered cosine at every layer from PRE-centered+normalized banks.

    ``ctx_hat`` / ``qry_hat`` are the globally-centered, L2-normalized banks from
    :func:`_global_center_normalize`. The per-pair dot product per layer is the
    centered cosine. Returns (n_pairs, n_layers).
    """
    # ctx_hat, qry_hat: (n_pairs, n_layers, hidden) already centered+normalized.
    return (ctx_hat * qry_hat).sum(dim=2).numpy()


def _raw_cos_per_layer(A_ctx: torch.Tensor, A_qry: torch.Tensor) -> np.ndarray:
    """Per-pair RAW (uncentered) cosine at every layer. Returns (n_pairs, n_layers)."""
    n_pairs, n_layers, _ = A_ctx.shape
    out = np.zeros((n_pairs, n_layers))
    for L in range(n_layers):
        ctx_n = torch.nn.functional.normalize(A_ctx[:, L], dim=1)
        qry_n = torch.nn.functional.normalize(A_qry[:, L], dim=1)
        out[:, L] = (ctx_n * qry_n).sum(dim=1).numpy()
    return out


def _derangement_floor(
    ctx_hat: torch.Tensor,
    qry_hat: torch.Tensor,
    indices: np.ndarray,
    rng: np.random.Generator,
    b: int,
) -> dict:
    """Shuffled-pair (derangement) centered-cosine floor over a set of indices.

    ``ctx_hat`` / ``qry_hat`` are the GLOBALLY-centered, L2-normalized banks
    (:func:`_global_center_normalize`) — the IDENTICAL tensors the matched
    per-pair cosine consumes. This helper does NOT re-center; it only restricts
    the derangement to ``indices`` (a within-tier subset for per-tier floors, or
    all indices for the global floor). Centering matched and floor on the same
    global bank is what makes ``matched_minus_shuffled`` subtract identical
    pre-centered tensors (concern per-tier-floor-centering-mismatch).

    For each derangement pi (i != pi(i)) over the row positions in ``indices``,
    compute the cosine of ctx_hat[indices[i]] vs qry_hat[indices[pi(i)]] at every
    layer, mean over the subset. Returns mean + 2.5/97.5 band per layer over ``b``
    derangements.
    """
    n_layers = ctx_hat.shape[1]
    sub_ctx = ctx_hat[indices].numpy()  # (m, n_layers, hidden) — already centered+normalized
    sub_qry = qry_hat[indices].numpy()
    m = len(indices)
    if m < 2:
        # No derangement possible with < 2 items.
        nan = np.full(n_layers, np.nan)
        return {"mean": nan.tolist(), "lo": nan.tolist(), "hi": nan.tolist(), "n": m}

    boot = np.zeros((b, n_layers))
    for k in range(b):
        perm = _derangement(m, rng)
        # cos[i, L] = <ctx_hat[i, L], qry_hat[perm[i], L]>  (within-subset shuffle)
        cos = np.einsum("ild,ild->il", sub_ctx, sub_qry[perm])  # (m, n_layers)
        boot[k] = cos.mean(axis=0)
    return {
        "mean": boot.mean(axis=0).tolist(),
        "lo": np.percentile(boot, 2.5, axis=0).tolist(),
        "hi": np.percentile(boot, 97.5, axis=0).tolist(),
        "n": m,
    }


def _derangement(m: int, rng: np.random.Generator) -> np.ndarray:
    """A random derangement of range(m) (no fixed points). Rejection-sample."""
    while True:
        perm = rng.permutation(m)
        if not np.any(perm == np.arange(m)):
            return perm


def _cka_per_layer_from_banks(
    A_ctx: torch.Tensor, A_qry: torch.Tensor, indices: np.ndarray
) -> list[float]:
    sub_ctx = A_ctx[indices]
    sub_qry = A_qry[indices]
    return [linear_cka(sub_ctx[:, L], sub_qry[:, L]) for L in range(sub_ctx.shape[1])]


def _cka_shuffled_floor(
    A_ctx: torch.Tensor, A_qry: torch.Tensor, indices: np.ndarray, rng: np.random.Generator
) -> list[float]:
    """CKA(context-bank, row-permuted query-bank) — the whole-bank shuffle floor."""
    sub_ctx = A_ctx[indices]
    sub_qry = A_qry[indices]
    m = len(indices)
    if m < 2:
        return [float("nan")] * sub_ctx.shape[1]
    perm = _derangement(m, rng)
    return [linear_cka(sub_ctx[:, L], sub_qry[perm][:, L]) for L in range(sub_ctx.shape[1])]


def _companion_cosine_per_layer(
    companion: dict[str, torch.Tensor], meta: list[dict], A_readout: torch.Tensor
) -> dict:
    """SAME-POSITION companion contrast (plan §5): context-only vs full-prompt,
    BOTH read at the assistant-generation slot.

    For each pair *i*, cosine of
      - ``companion[context_id]`` = the context-only prompt's assistant-gen slot
        readout (no query), against
      - ``A_readout[i]`` = the SAME pair's FULL-prompt assistant-gen slot readout
        (context + query), captured in the same forward as context-end/query-end.

    Both vectors are read at the FIXED assistant-generation position, so the only
    difference between them is the PRESENCE OF THE QUERY — the different-token
    confound of the old query-end-slot companion read is removed (concern
    companion-read-not-same-slot). Raw cosine (a within-context with/without-query
    comparison; no cross-pair centering). Aggregated per tier (mean per layer over
    that tier's pairs) AND per context (mean per layer over that context's pairs).
    """
    n_layers = A_readout.shape[1]
    per_pair_cos: dict[str, np.ndarray] = {}
    by_ctx_cos: dict[str, list[np.ndarray]] = defaultdict(list)
    by_tier_cos: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, m in enumerate(meta):
        cid = m["context_id"]
        ctx_only = companion[cid].to(torch.float64)  # (n_layers, hidden), assistant-gen slot
        full = A_readout[i]  # (n_layers, hidden), full-prompt assistant-gen slot
        a = torch.nn.functional.normalize(ctx_only, dim=1)
        b = torch.nn.functional.normalize(full, dim=1)
        cos = (a * b).sum(dim=1).numpy()  # (n_layers,)
        per_pair_cos[m["pair_id"]] = cos
        by_ctx_cos[cid].append(cos)
        by_tier_cos[m["context_type"]].append(cos)

    assert per_pair_cos, "no pairs for companion same-slot contrast"
    per_context = {cid: np.stack(rows).mean(axis=0).tolist() for cid, rows in by_ctx_cos.items()}
    tier_mean = {t: np.stack(rows).mean(axis=0).tolist() for t, rows in by_tier_cos.items()}
    _ = n_layers  # documented for shape clarity
    return {"per_context": per_context, "per_tier_mean": tier_mean}


# ── Amendment (plan v5 §3): dummy-vs-real same-slot companion-curve gap ───────


def _load_readout_banks(banks_dir: Path) -> dict:
    """Load only the per-pair ``readout`` banks + their context-only companions.

    Lighter than :func:`_load_banks` (no context-end/query-end banks). Returns the
    per-pair full-prompt assistant-gen readout, the per-pair metadata (incl.
    ``real_query_id`` for the dummy arm), and the per-context companion readout
    cache. Used by the dummy-vs-real companion gap (plan v5 §3).
    """
    pair_files = sorted(banks_dir.glob("pair_*.pt"))
    if not pair_files:
        raise RuntimeError(f"no pair_*.pt banks in {banks_dir}")
    readouts: list[torch.Tensor] = []
    meta_rows: list[dict] = []
    companion_cache: dict[str, torch.Tensor] = {}
    for pf in pair_files:
        d = torch.load(pf, weights_only=True)
        if "readout" not in d:
            raise RuntimeError(f"{pf.name}: missing per-pair 'readout' bank")
        readouts.append(d["readout"])  # (n_layers, hidden)
        # The dummy arm carries 'real_query_id'; the real arm does not, so the
        # join key falls back to 'query_id' (which is the real query id there).
        meta_rows.append(
            {
                "pair_id": d["pair_id"],
                "context_type": d["context_type"],
                "context_id": d["context_id"],
                "query_id": d["query_id"],
                "real_query_id": d.get("real_query_id", d["query_id"]),
                "topicality": d["topicality"],
                "length": d["length"],
                "companion_context_only_file": d["companion_context_only_file"],
            }
        )
        cid = d["context_id"]
        if cid not in companion_cache:
            cpath = banks_dir / d["companion_context_only_file"]
            cd = torch.load(cpath, weights_only=True)
            companion_cache[cid] = cd["readout"]  # (n_layers, hidden)
    A_readout = torch.stack(readouts).to(torch.float64)  # (n_pairs, n_layers, hidden)
    return {"A_readout": A_readout, "meta": meta_rows, "companion": companion_cache}


def _per_pair_companion_cos(
    companion: dict[str, torch.Tensor], meta: list[dict], A_readout: torch.Tensor
) -> tuple[dict[tuple[str, str], np.ndarray], list[str]]:
    """Per-(context_id, real_query_id) same-slot companion cosine (n_layers,).

    cos(context-only assistant-gen readout, full-prompt assistant-gen readout) per
    layer — the v4 companion read, keyed so the real and dummy arms join on
    (context_id, real_query_id). Pairs whose ``context_id`` is absent from the
    supplied ``companion`` cache are SKIPPED (returned in the second element); the
    join in :func:`_companion_gap_per_layer_per_tier` then naturally excludes them.
    In production both arms share the full 81-context cache, so nothing is skipped;
    the skip path only fires when a smaller arm is paired against a fuller one.
    """
    out: dict[tuple[str, str], np.ndarray] = {}
    skipped: list[str] = []
    for i, m in enumerate(meta):
        cid = m["context_id"]
        if cid not in companion:
            skipped.append(cid)
            continue
        ctx_only = companion[cid].to(torch.float64)
        full = A_readout[i]
        a = torch.nn.functional.normalize(ctx_only, dim=1)
        b = torch.nn.functional.normalize(full, dim=1)
        out[(cid, m["real_query_id"])] = (a * b).sum(dim=1).numpy()  # (n_layers,)
    return out, skipped


def _companion_gap_per_layer_per_tier(
    real_banks: dict, dummy_banks: dict, context_only_banks: dict[str, torch.Tensor]
) -> dict:
    """Dummy-vs-real same-slot companion-curve gap per layer per tier (plan v5 §3).

    gap(tier, L) = companion_cos_real(tier, L) - companion_cos_dummy(tier, L)

    The companion read is the v4 same-slot contrast: cos(context-only assistant-gen
    readout, full-prompt assistant-gen readout). BOTH arms read against the SAME
    cached ``context_only_banks`` (plan v5 §4 — the context-only side is identical
    for the two arms). A POSITIVE gap means the real-content query displaces the
    generation-slot state MORE than the content-matched dummy — i.e. the late-layer
    displacement is content-driven, not purely mechanical (length/position).

    The real and dummy arms join per pair on ``(context_id, real_query_id)``: the
    dummy pair stores its matched real query's id under ``real_query_id``. The gap
    is computed PER-PAIR (real minus dummy at the same (context, query), so each
    pair is its own control), then aggregated per tier and per length bin, with the
    per-pair gap standard error so the falsification band is read from the data (NOT
    a hard-coded 0.03 — plan §6 / analyst-weighable concern).

    Args mirror ``real_banks`` / ``dummy_banks`` = the dicts from
    :func:`_load_readout_banks`; ``context_only_banks`` is the shared cache (either
    arm's companion cache works — they are the same banks). Returns per-layer per-
    tier gap mean + SE + n, the full per-layer overall curve, the per-length-bin
    breakdown, and the unmatched-pair audit.
    """
    real_meta, dummy_meta = real_banks["meta"], dummy_banks["meta"]
    n_layers = real_banks["A_readout"].shape[1]

    cos_real, skipped_real = _per_pair_companion_cos(
        context_only_banks, real_meta, real_banks["A_readout"]
    )
    cos_dummy, skipped_dummy = _per_pair_companion_cos(
        context_only_banks, dummy_meta, dummy_banks["A_readout"]
    )

    # Join on (context_id, real_query_id): each real pair has exactly one matched
    # dummy. Compute the per-pair gap = real - dummy; carry the pair's tier + length.
    real_key_meta = {(m["context_id"], m["real_query_id"]): m for m in real_meta}
    matched_keys = sorted(set(cos_real) & set(cos_dummy))
    unmatched_real = sorted(set(cos_real) - set(cos_dummy))
    unmatched_dummy = sorted(set(cos_dummy) - set(cos_real))
    if not matched_keys:
        raise RuntimeError(
            "no (context_id, real_query_id) pairs joined between the real and dummy "
            "arms — check the dummy battery's real_query_id mirrors the real query ids"
        )

    per_pair_gap: list[np.ndarray] = []
    by_tier_gap: dict[str, list[np.ndarray]] = defaultdict(list)
    by_length_gap: dict[str, list[np.ndarray]] = defaultdict(list)
    by_tier_length_gap: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for k in matched_keys:
        gap = cos_real[k] - cos_dummy[k]  # (n_layers,)
        per_pair_gap.append(gap)
        m = real_key_meta[k]
        by_tier_gap[m["context_type"]].append(gap)
        by_length_gap[m["length"]].append(gap)
        by_tier_length_gap[(m["context_type"], m["length"])].append(gap)

    def _agg(rows: list[np.ndarray]) -> dict:
        arr = np.stack(rows)  # (n, n_layers)
        n = arr.shape[0]
        mean = arr.mean(axis=0)
        # per-pair SE of the gap (sample sd / sqrt n); n<2 -> SE undefined (nan).
        se = arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(n_layers, np.nan)
        return {"gap_mean": mean.tolist(), "gap_se": se.tolist(), "n": int(n)}

    per_tier = {t: _agg(rows) for t, rows in by_tier_gap.items()}
    per_length = {ln: _agg(rows) for ln, rows in by_length_gap.items()}
    per_tier_length = {f"{t}__{ln}": _agg(rows) for (t, ln), rows in by_tier_length_gap.items()}
    overall = _agg(per_pair_gap)

    # Companion curves themselves (real + dummy), per tier, for the figure.
    real_tier_cos: dict[str, list[float]] = {}
    dummy_tier_cos: dict[str, list[float]] = {}
    by_tier_real: dict[str, list[np.ndarray]] = defaultdict(list)
    by_tier_dummy: dict[str, list[np.ndarray]] = defaultdict(list)
    for k in matched_keys:
        t = real_key_meta[k]["context_type"]
        by_tier_real[t].append(cos_real[k])
        by_tier_dummy[t].append(cos_dummy[k])
    for t in by_tier_real:
        real_tier_cos[t] = np.stack(by_tier_real[t]).mean(axis=0).tolist()
        dummy_tier_cos[t] = np.stack(by_tier_dummy[t]).mean(axis=0).tolist()

    return {
        "n_layers": n_layers,
        "n_matched_pairs": len(matched_keys),
        "n_unmatched_real": len(unmatched_real),
        "n_unmatched_dummy": len(unmatched_dummy),
        "n_skipped_real_missing_context": len(skipped_real),
        "n_skipped_dummy_missing_context": len(skipped_dummy),
        "unmatched_real_keys": unmatched_real[:20],
        "unmatched_dummy_keys": unmatched_dummy[:20],
        "overall": overall,
        "per_tier": per_tier,
        "per_length": per_length,
        "per_tier_length": per_tier_length,
        "companion_cos_real_per_tier": real_tier_cos,
        "companion_cos_dummy_per_tier": dummy_tier_cos,
    }


def analyze(banks_dir: Path, out_dir: Path) -> dict:
    data = _load_banks(banks_dir)
    A_ctx, A_qry, meta = data["A_ctx"], data["A_qry"], data["meta"]
    A_readout = data["A_readout"]
    layers, n_layers = data["layers"], data["n_layers"]
    rng = np.random.default_rng(SEED)

    # Globally center+normalize EACH bank ONCE (per-layer, over ALL pairs). BOTH
    # the matched per-pair cosine and the shuffled-pair floor consume these exact
    # tensors — never a per-tier re-centering (concern
    # per-tier-floor-centering-mismatch).
    ctx_hat = _global_center_normalize(A_ctx)  # (n_pairs, n_layers, hidden)
    qry_hat = _global_center_normalize(A_qry)
    centered = _centered_cos_per_layer(ctx_hat, qry_hat)  # (n_pairs, n_layers)
    raw = _raw_cos_per_layer(A_ctx, A_qry)

    # Index groupings.
    all_idx = np.arange(len(meta))
    by_type: dict[str, np.ndarray] = {}
    for t in sorted({m["context_type"] for m in meta}):
        by_type[t] = np.array([i for i, m in enumerate(meta) if m["context_type"] == t])
    by_query_type: dict[str, np.ndarray] = {}
    for qt in sorted({(m["topicality"], m["length"]) for m in meta}):
        key = f"{qt[0]}_{qt[1]}"
        by_query_type[key] = np.array(
            [i for i, m in enumerate(meta) if (m["topicality"], m["length"]) == qt]
        )

    # ── Global floor (all pairs) — same globally-centered banks as matched ───
    global_floor = _derangement_floor(ctx_hat, qry_hat, all_idx, rng, B_DERANGEMENT)
    global_cka = _cka_per_layer_from_banks(A_ctx, A_qry, all_idx)
    global_cka_floor = _cka_shuffled_floor(A_ctx, A_qry, all_idx, rng)

    # ── Per-tier within-type floors + CKA ────────────────────────────────────
    # Per-tier floors restrict the derangement to within-tier rows of the SAME
    # globally-centered banks — never re-centered per tier.
    per_type_floor: dict[str, dict] = {}
    per_type_cka: dict[str, list[float]] = {}
    per_type_cka_floor: dict[str, list[float]] = {}
    for t, idx in by_type.items():
        per_type_floor[t] = _derangement_floor(ctx_hat, qry_hat, idx, rng, B_DERANGEMENT)
        per_type_cka[t] = _cka_per_layer_from_banks(A_ctx, A_qry, idx)
        per_type_cka_floor[t] = _cka_shuffled_floor(A_ctx, A_qry, idx, rng)

    # ── Companion same-position contrast (context-only vs full-prompt, same slot) ─
    companion = _companion_cosine_per_layer(data["companion"], meta, A_readout)

    # ── Headline structured output: context_type x layer x query_type (§6.5) ─
    headline: dict[str, dict] = {}
    for t, idx in by_type.items():
        headline[t] = {}
        floor = per_type_floor[t]
        for L in range(n_layers):
            layer_key = str(layers[L])
            per_qt: dict[str, dict] = {}
            for qt_key, qt_idx in by_query_type.items():
                tier_qt = np.intersect1d(idx, qt_idx)
                if len(tier_qt) == 0:
                    continue
                m_cos = float(centered[tier_qt, L].mean())
                per_qt[qt_key] = {
                    "matched_centered_cos_mean": m_cos,
                    "n": len(tier_qt),
                }
            headline[t][layer_key] = {
                "matched_centered_cos_mean": float(centered[idx, L].mean()),
                "raw_cos_mean": float(raw[idx, L].mean()),
                "shuffled_floor_mean": floor["mean"][L],
                "shuffled_floor_lo": floor["lo"][L],
                "shuffled_floor_hi": floor["hi"][L],
                "matched_minus_shuffled": float(centered[idx, L].mean()) - floor["mean"][L],
                "cka_matched": per_type_cka[t][L],
                "cka_shuffled_floor": per_type_cka_floor[t][L],
                "companion_cos_mean": companion["per_tier_mean"].get(t, [float("nan")] * n_layers)[
                    L
                ],
                "by_query_type": per_qt,
            }

    result = {
        "issue": 654,
        "layers": layers,
        "anchor_layers": ANCHOR_LAYERS,
        "n_pairs": len(meta),
        "seed": SEED,
        "b_derangement": B_DERANGEMENT,
        "context_types": sorted(by_type.keys()),
        "query_types": sorted(by_query_type.keys()),
        "centering": "global_mean",
        "global": {
            "matched_centered_cos_mean": centered.mean(axis=0).tolist(),
            "raw_cos_mean": raw.mean(axis=0).tolist(),
            "shuffled_floor": global_floor,
            "matched_minus_shuffled": (
                centered.mean(axis=0) - np.array(global_floor["mean"])
            ).tolist(),
            "cka_matched": global_cka,
            "cka_shuffled_floor": global_cka_floor,
        },
        "per_context_type": headline,
        "companion": companion,
        "extraction_manifest_summary": {
            "model": data["manifest"].get("model"),
            "num_hidden_layers": data["manifest"].get("num_hidden_layers"),
            "hidden_size": data["manifest"].get("hidden_size"),
            "offset_fail_fraction": data["manifest"].get("offset_fail_fraction"),
            "n_pairs_extracted": data["manifest"].get("n_pairs_extracted"),
        },
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    headline_path = out_dir / "per_layer_displacement.json"
    with open(headline_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s", headline_path)

    # Per-cell breakdowns (one JSON per context_type).
    for t in by_type:
        with open(cells_dir / f"context_type_{t}.json", "w") as f:
            json.dump(headline[t], f, ensure_ascii=False, indent=2)
    logger.info("wrote %d per-cell breakdowns to %s", len(by_type), cells_dir)

    # Stash the per-pair matrices for figures (kept in-memory; returned).
    # ctx_hat/qry_hat are the globally-centered+normalized banks (same tensors the
    # matched cosine + floor consume); the anchor scatter reads centered cosine
    # straight off them rather than re-centering per layer.
    result["_arrays"] = {
        "centered": centered,
        "raw": raw,
        "by_type": {t: idx.tolist() for t, idx in by_type.items()},
        "by_query_type": {k: idx.tolist() for k, idx in by_query_type.items()},
        "meta": meta,
        "ctx_hat": ctx_hat,
        "qry_hat": qry_hat,
    }
    return result


def make_figures(result: dict, fig_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    layers = result["layers"]
    n_layers = len(layers)
    arrays = result["_arrays"]
    centered = arrays["centered"]
    raw = arrays["raw"]
    by_type = arrays["by_type"]
    types = sorted(by_type.keys())
    colors = paper_palette(min(max(len(types), 1), 8))

    # ── Hero: per-layer matched centered cosine per context type + floor band ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ci, t in enumerate(types):
        idx = np.array(by_type[t])
        m = centered[idx].mean(axis=0)
        floor = result["per_context_type"][t]
        lo = np.array([floor[str(layers[L])]["shuffled_floor_lo"] for L in range(n_layers)])
        hi = np.array([floor[str(layers[L])]["shuffled_floor_hi"] for L in range(n_layers)])
        ax.plot(layers, m, label=t, color=colors[ci % len(colors)], linewidth=2)
        ax.fill_between(layers, lo, hi, color=colors[ci % len(colors)], alpha=0.15)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Matched centered cosine (context-end vs query-end)")
    ax.set_title("Per-layer query displacement by context type (shaded = shuffled-pair floor)")
    ax.legend(loc="best", fontsize=8)
    savefig_paper(fig, "issue_654/hero_per_layer_displacement", dir=fig_dir)
    plt.close(fig)

    # ── Raw vs centered overlay (anisotropy caveat) ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(layers, centered.mean(axis=0), label="centered (global_mean)", linewidth=2)
    ax.plot(layers, raw.mean(axis=0), label="raw (uncentered)", linewidth=2, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean per-pair cosine")
    ax.set_title("Raw vs centered cosine (anisotropy caveat)")
    ax.legend(loc="best", fontsize=8)
    savefig_paper(fig, "issue_654/raw_vs_centered", dir=fig_dir)
    plt.close(fig)

    # ── CKA heatmap (layer x layer is overkill; plot the diagonal CKA(L,L)) ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ci, t in enumerate(types):
        cka = [
            result["per_context_type"][t][str(layers[L])]["cka_matched"] for L in range(n_layers)
        ]
        cka_floor = [
            result["per_context_type"][t][str(layers[L])]["cka_shuffled_floor"]
            for L in range(n_layers)
        ]
        ax.plot(layers, cka, label=f"{t}", color=colors[ci % len(colors)], linewidth=2)
        ax.plot(layers, cka_floor, color=colors[ci % len(colors)], linewidth=1, linestyle=":")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA (context-bank vs query-bank)")
    ax.set_title("Per-layer CKA by context type (dotted = shuffled-bank floor)")
    ax.legend(loc="best", fontsize=8)
    savefig_paper(fig, "issue_654/cka_per_layer", dir=fig_dir)
    plt.close(fig)

    # ── Per-pair scatter at the 4 anchor layers (colored by query topicality) ─
    meta = arrays["meta"]
    topic = np.array([1 if m["topicality"] == "on" else 0 for m in meta])
    ctx_hat = arrays["ctx_hat"]  # globally centered+normalized (same as matched/floor)
    qry_hat = arrays["qry_hat"]
    fig, axes = plt.subplots(1, len(ANCHOR_LAYERS), figsize=(4 * len(ANCHOR_LAYERS), 4))
    for ax, L in zip(np.atleast_1d(axes), ANCHOR_LAYERS, strict=False):
        Li = layers.index(L) if L in layers else min(L, n_layers - 1)
        cc = (ctx_hat[:, Li] * qry_hat[:, Li]).sum(dim=1).numpy()
        # x = pair index within tier ordering; y = centered cosine.
        ax.scatter(
            np.arange(len(cc))[topic == 1],
            cc[topic == 1],
            s=10,
            alpha=0.6,
            label="on-topic",
        )
        ax.scatter(
            np.arange(len(cc))[topic == 0],
            cc[topic == 0],
            s=10,
            alpha=0.6,
            label="off-topic",
        )
        ax.set_title(f"Layer {L}")
        ax.set_xlabel("pair index")
        ax.set_ylabel("centered cos")
        ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "issue_654/scatter_anchors", dir=fig_dir)
    plt.close(fig)

    # ── Violin of per-pair centered cosine by topicality x length at anchors ──
    fig, axes = plt.subplots(1, len(ANCHOR_LAYERS), figsize=(4 * len(ANCHOR_LAYERS), 4))
    qt_keys = sorted({f"{m['topicality']}_{m['length']}" for m in meta})
    for ax, L in zip(np.atleast_1d(axes), ANCHOR_LAYERS, strict=False):
        Li = layers.index(L) if L in layers else min(L, n_layers - 1)
        groups = []
        for qt in qt_keys:
            sel = np.array(
                [i for i, m in enumerate(meta) if f"{m['topicality']}_{m['length']}" == qt]
            )
            groups.append(centered[sel, Li] if len(sel) else np.array([0.0]))
        ax.violinplot(groups, showmeans=True)
        ax.set_xticks(range(1, len(qt_keys) + 1))
        ax.set_xticklabels(qt_keys, rotation=45, ha="right", fontsize=6)
        ax.set_title(f"Layer {L}")
        ax.set_ylabel("centered cos")
    fig.tight_layout()
    savefig_paper(fig, "issue_654/violin_query_type", dir=fig_dir)
    plt.close(fig)

    # ── Companion same-position vs two-position curve per context type ────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    comp = result["companion"]["per_tier_mean"]
    for ci, t in enumerate(types):
        idx = np.array(by_type[t])
        ax.plot(
            layers,
            centered[idx].mean(axis=0),
            label=f"{t} two-position",
            color=colors[ci % len(colors)],
            linewidth=2,
        )
        if t in comp:
            ax.plot(
                layers,
                comp[t],
                label=f"{t} companion",
                color=colors[ci % len(colors)],
                linewidth=1.5,
                linestyle="--",
            )
    ax.set_xlabel("Layer")
    ax.set_ylabel("cosine")
    ax.set_title("Two-position vs companion same-position contrast")
    ax.legend(loc="best", fontsize=7)
    savefig_paper(fig, "issue_654/companion_vs_two_position", dir=fig_dir)
    plt.close(fig)

    logger.info("wrote figures to %s/issue_654/", fig_dir)


# ── Amendment (plan v5 §3/§5): companion-gap driver + figure ─────────────────


def make_companion_gap_figure(gap: dict, layers: list[int], fig_dir: str) -> None:
    """Real vs dummy companion curves per tier, with the per-pair gap SE band.

    One panel: real-arm and dummy-arm same-slot companion cosine per tier (solid
    = real, dashed = dummy), and a second panel with the per-tier gap curve +- the
    per-pair gap SE shaded (the falsification band is read off this SE, not a
    hard-coded 0.03 — plan §6).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    tiers = sorted(gap["per_tier"].keys())
    colors = paper_palette(min(max(len(tiers), 1), 8))

    # Plain-English tier labels for legends (paper-plots §3.5).
    tier_label = {
        "generic": "generic",
        "icl": "in-context",
        "persona": "persona",
        "wildchat": "real chat",
    }

    fig, (ax_curves, ax_gap) = plt.subplots(1, 2, figsize=(13, 4.5))
    real_cos = gap["companion_cos_real_per_tier"]
    dummy_cos = gap["companion_cos_dummy_per_tier"]
    for ci, t in enumerate(tiers):
        col = colors[ci % len(colors)]
        lbl = tier_label.get(t, t)
        if t in real_cos:
            ax_curves.plot(layers, real_cos[t], label=f"{lbl} real", color=col, linewidth=2)
        if t in dummy_cos:
            ax_curves.plot(
                layers,
                dummy_cos[t],
                label=f"{lbl} dummy",
                color=col,
                linewidth=1.5,
                linestyle="--",
            )
    ax_curves.set_xlabel("Layer")
    ax_curves.set_ylabel("Same-slot companion cosine")
    ax_curves.set_title("Real query vs length-matched no-content filler, same slot")
    ax_curves.legend(loc="best", fontsize=6)

    for ci, t in enumerate(tiers):
        col = colors[ci % len(colors)]
        g = np.array(gap["per_tier"][t]["gap_mean"])
        se = np.array(gap["per_tier"][t]["gap_se"])
        ax_gap.plot(layers, g, label=tier_label.get(t, t), color=col, linewidth=2)
        ax_gap.fill_between(layers, g - se, g + se, color=col, alpha=0.15)
    ax_gap.axhline(0.0, color="grey", linewidth=1, linestyle=":")
    ax_gap.set_xlabel("Layer")
    ax_gap.set_ylabel("Gap = companion cos(real) - cos(dummy)")
    ax_gap.set_title("Real-minus-dummy gap per tier (shaded = per-pair gap SE)")
    ax_gap.legend(loc="best", fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "issue_654/query_content_vs_length_gap_blog", dir=fig_dir)
    plt.close(fig)
    logger.info("wrote companion-gap figure to %s/issue_654/", fig_dir)


def companion_gap(
    real_banks_dir: Path,
    dummy_banks_dir: Path,
    out_dir: Path,
    context_only_dir: Path | None = None,
    figures: bool = False,
    fig_dir: str = "figures/",
) -> dict:
    """Compute + persist the dummy-vs-real companion gap (plan v5 §3).

    ``real_banks_dir`` / ``dummy_banks_dir`` each hold ``pair_*.pt`` + a
    ``context_only/`` companion dir. The shared cached context-only banks (plan v5
    §4) are taken from ``context_only_dir`` if given, else from the dummy arm's own
    ``context_only/`` (which the extractor populated by REUSING the cached real-arm
    banks via ``--reuse-context-only``, so they are identical). Writes
    ``companion_gap.json`` + the gap figure.
    """
    real_banks = _load_readout_banks(real_banks_dir)
    dummy_banks = _load_readout_banks(dummy_banks_dir)

    if context_only_dir is not None:
        ctx_only: dict[str, torch.Tensor] = {}
        for cf in sorted(context_only_dir.glob("*.pt")):
            cd = torch.load(cf, weights_only=True)
            ctx_only[cd["context_id"]] = cd["readout"]
        if not ctx_only:
            raise RuntimeError(f"no context_only/*.pt banks in {context_only_dir}")
    else:
        # Use the dummy arm's own companion cache (populated by --reuse-context-only
        # from the cached real-arm banks; identical to the real arm's).
        ctx_only = dummy_banks["companion"]

    gap = _companion_gap_per_layer_per_tier(real_banks, dummy_banks, ctx_only)

    # Late-layer trough summary (L23-L27, where the parent companion bottomed at
    # 0.63-0.72): mean-of-tiers gap over that band (plan §3 single summary number).
    layers = list(range(gap["n_layers"]))
    late = [L for L in layers if 23 <= L <= 27]
    overall_mean = np.array(gap["overall"]["gap_mean"])
    overall_se = np.array(gap["overall"]["gap_se"])
    late_trough = {
        "layers": late,
        "overall_gap_mean": float(np.mean(overall_mean[late])) if late else float("nan"),
        "overall_gap_se_mean": float(np.mean(overall_se[late])) if late else float("nan"),
    }

    result = {
        "issue": 654,
        "followup_label": "length-matched-dummy-query-control",
        "dv": "dummy_vs_real_same_slot_companion_curve_gap",
        "layers": layers,
        "anchor_layers": ANCHOR_LAYERS,
        "late_layer_trough_summary": late_trough,
        **gap,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    gap_path = out_dir / "companion_gap.json"
    with open(gap_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(
        "wrote %s (%d matched pairs; %d unmatched real / %d unmatched dummy)",
        gap_path,
        gap["n_matched_pairs"],
        gap["n_unmatched_real"],
        gap["n_unmatched_dummy"],
    )
    if figures:
        make_companion_gap_figure(gap, layers, fig_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #654: per-layer displacement metrics + figures."
    )
    parser.add_argument(
        "--banks", type=Path, default=None, help="dir with pair_*.pt + manifest (default mode)"
    )
    parser.add_argument("--out", type=Path, required=True, help="eval_results/issue_654 dir")
    parser.add_argument("--figures", action="store_true", help="also emit figures")
    parser.add_argument("--fig-dir", default="figures/", help="figure parent dir")
    # Amendment (plan v5 §3): the dummy-vs-real companion-gap mode.
    parser.add_argument(
        "--companion-gap",
        action="store_true",
        help="compute the dummy-vs-real same-slot companion-curve gap (plan v5 §3)",
    )
    parser.add_argument("--real-banks", type=Path, default=None, help="real-arm pair_*.pt dir")
    parser.add_argument("--dummy-banks", type=Path, default=None, help="dummy-arm pair_*.pt dir")
    parser.add_argument(
        "--context-only",
        type=Path,
        default=None,
        help="shared cached context_only/ dir (default: the dummy arm's own context_only/)",
    )
    args = parser.parse_args()

    if args.companion_gap:
        if args.real_banks is None or args.dummy_banks is None:
            parser.error("--companion-gap requires --real-banks and --dummy-banks")
        companion_gap(
            args.real_banks,
            args.dummy_banks,
            args.out,
            context_only_dir=args.context_only,
            figures=args.figures,
            fig_dir=args.fig_dir,
        )
        return 0

    if args.banks is None:
        parser.error("--banks is required for the default (per-layer displacement) mode")
    result = analyze(args.banks, args.out)
    if args.figures:
        make_figures(result, args.fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
