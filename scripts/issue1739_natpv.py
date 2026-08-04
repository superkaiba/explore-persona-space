"""Naturalistic persona-vector regimes (E2 matched-pair / E2p pooled) for the
#1739 behaviors that the main grid left un-extracted (hallucination, sycophancy),
plus the E1/E2/E2p regime comparison over the four projection reads.

WHY streaming: the behavior labeling stores live on the HF data repo as
MONOLITHIC tars (52.14 GB sycophancy / 69.87 GB hallucination) whose members are
INTERLEAVED across kind x layer x shard, so there is no member-selective or
per-layer read — every touch of the per-rollout ``t1`` costs one full-tar pass.
This driver therefore makes THREE bounded streaming passes per behavior and
never materializes an activation grid (measured 111.7 MB/s at 12-way ranges =>
7.8 min/pass sycophancy, 10.4 min/pass hallucination):

  ``--phase rowindex``   stream tar, write only ``row_index_shard*.jsonl``
                         (~40 MB) -> the global row order + (context_id,
                         rollout_k) per row.
  ``--phase directions`` stream tar; for each ``t1_L{ly}_shard{ss}`` member
                         accumulate ONE masked matvec ``w_row @ shard`` into
                         per-layer E2 / E2p direction accumulators. Weights come
                         verbatim from the tested
                         ``fits.matched_pair_split_weights`` (E2: within-context
                         midpoint split, qualification spread >= E2_SPREAD_MIN;
                         E2p: pooled global-midpoint split over ALL kept
                         per-ROLLOUT scores) -- the math is REUSED, never forked.
  ``--phase project``    stream tar; per member accumulate per-row SCALARS only
                         (~145-195 MB cube): direction . context_end,
                         direction . prefix_end, direction . t1, and the
                         map->answer projection for BOTH map arms. Predictions
                         are never materialized: ``pred . v`` reduces to
                         ``((x-x_mu)/x_sd) . (W v) + y_mu . v``, so ``W v`` is
                         precomputed ONCE per (map arm, regime, layer).
  ``--phase reduce``     local only; Spearman rho vs the per-context DV per
                         (regime x read x layer x rung), freeze the layer on
                         train-rung rho, emit regime_comparison.json.

Both mapping arms run throughout (context_end AND prefix_end), per the standing
both-arms rule. Nothing is fitted here (no ridge / probe / MLP), so there is no
n_train-vs-d regime: every read is a fixed linear projection + a rank
correlation.

SPACE (``--space raw`` default | ``whitened``). The evil main-grid arms project in
WHITENED space (arms.py: z_ctx whitened, rb "whitened space") with the U-pool
shrinkage whitening fit fresh in-CLI and never persisted. The original round read
RAW-space only, on the premise that reproducing the whitened space needs the
whitened labeled table RESIDENT (~14 GB per kind + ~69 GB per-rollout t1) against
17 GB VM RAM. That premise is FALSE: whitening is a LINEAR map, so it folds into
the direction instead of the data and no whitened grid is ever materialized.

With ``z = (x - mu) W`` (``W = Sigma_g^{-1/2}``, SYMMETRIC by construction --
``evecs diag(lam^-1/2) evecs^T``) and the main grid's whitened direction
``rb_w = rb_raw W`` (fits.py `einsum("ld,lde->le", rb, wh.w)`), every read is an
AFFINE function of the RAW row:

    projection read   score = z . rb_w        = x . (W rb_w)  - mu . (W rb_w)
    map read          score = pred_w . rb_w   = x . (W h)     + const
                      where g = w rb_w, h = g / x_sd,
                      const = -mu.(W h) - x_mu.h + y_mu.rb_w

So each (read x regime x layer) collapses to ONE (d,) vector + ONE scalar,
precomputed before streaming. Peak RAM is the whitening matrices during that
precompute (~1.4 GB fp32 per variant), NOT the activation grid -- the same
bounded streaming shape as the raw path. Whitened reads ARE numerically
comparable to the committed main-grid columns; raw reads remain PROVISIONAL.

The whitening itself is BEHAVIOR-INDEPENDENT (fit on the shared #1092 U-pool
slice, keyed only by variant x u_size), so ``--phase whitening`` fits it ONCE per
variant and PERSISTS it -- closing the "fit fresh in-CLI and never persisted" gap
for every behavior. The E1 anchor runs through whichever space is selected, so
the E1-vs-E2-vs-E2p comparison stays internally matched either way.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import tarfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_syspath() -> None:
    """Script mode puts THIS dir on sys.path[0]; ``scripts.*`` needs repo root (#823)."""
    sentinel = _REPO_ROOT / "scripts" / "issue1739_map963k_slice.py"
    if not sentinel.is_file():
        raise RuntimeError(f"repo-root sentinel missing: {sentinel}")
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


_ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue1739_natpv")

REPO = "superkaiba1/explore-persona-space-data"
BEHAVIORS = ("hallucination", "sycophancy")
REGIMES = ("e1", "e2", "e2p")
VARIANTS = ("context_end", "prefix_end")
# Reads: (slug, description). ctx/pre = direction on the context/prefix state;
# map_ctx/map_pre = direction on the u-full map's predicted answer state from
# that variant; oracle = direction on the TRUE per-context mean answer state.
READS = ("ctx", "pre", "map_ctx", "map_pre", "oracle")
RB_E1_REVISION = "037fcbb"
RB_E1_PREFIX = "issue779_monitoring/r_b/"
MAPS_PREFIX = "issue1739_ctxmap/analysis_tensors/maps/"
JUDGE_PREFIX = "issue1739_ctxmap/judge/"
PREFIX_HASH_LAYER = 14  # single layer used for the distinct-prefix-state count
SPACES = ("raw", "whitened")
# The whitened-only extra read: the main grid whitens the answer acts with the
# SAME per-VARIANT whitening as the context arm, so under whitening the oracle
# read is variant-dependent (it is not in raw space). ``oracle`` keeps the
# context_end whitening (the main grid's primary arm); ``oracle_pre`` carries the
# prefix_end one. ADDITIVE — the five canonical READS keep their meaning.
ORACLE_PRE_READ = "oracle_pre"
# Behavior-INDEPENDENT persisted whitening (keyed variant x u_size only).
WHITEN_DIR = "whitening"
WHITEN_FILE_FMT = "{variant}__u{u_label}.npz"
U_STORE_DEFAULT = Path("data/issue_1739/hf_dl/u_store")


# ---------------------------------------------------------------------------
# tar streaming (reuses the committed #1739 ParallelRangeReader)
# ---------------------------------------------------------------------------


def _slice_mod():
    """Import the committed range-reader module (tar_url / head_size / reader)."""
    import importlib

    return importlib.import_module("scripts.issue1739_map963k_slice")


def stream_members(
    behavior: str, revision: str, *, workers: int, window_mib: int, want: re.Pattern
):
    """Yield (basename, ndarray-or-bytes) for tar members matching ``want``.

    ONE sequential pass over the whole tar; bytes TRANSFERRED are the tar,
    bytes RETAINED are only what the caller keeps (this driver keeps scalars).
    """
    import numpy as np

    m = _slice_mod()
    token = os.environ["HF_TOKEN"]
    url = m.tar_url(behavior, revision)
    total = m.head_size(url, token)
    logger.info(
        "[%s] streaming %.2f GB (%d-way ranges, %d MiB windows)",
        behavior,
        total / 1e9,
        workers,
        window_mib,
    )
    reader = m.ParallelRangeReader(
        url, token=token, total=total, window=window_mib * 1024 * 1024, workers=workers
    )
    t0 = time.time()
    seen = 0
    try:
        buffered = io.BufferedReader(reader, buffer_size=window_mib * 1024 * 1024)
        with tarfile.open(fileobj=buffered, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name.rsplit("/", 1)[-1]
                if not want.match(name):
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    raise RuntimeError(f"unreadable member {member.name}")
                raw = fh.read()
                seen += 1
                if seen % 500 == 0:
                    el = time.time() - t0
                    logger.info("[%s] %d members, %.1f min elapsed", behavior, seen, el / 60)
                if name.endswith(".npy"):
                    yield name, np.load(io.BytesIO(raw), allow_pickle=False)
                else:
                    yield name, raw
    finally:
        reader.close()
    logger.info("[%s] pass done: %d members, %.1f min", behavior, seen, (time.time() - t0) / 60)


def _summary_re(kinds: tuple[str, ...]) -> re.Pattern:
    return re.compile(rf"^(?:{'|'.join(kinds)})_L\d\d(?:_shard\d+)?\.npy$")


def _parse_summary_name(name: str) -> tuple[str, int, int]:
    """``context_end_L07_shard167.npy`` -> ('context_end', 7, 167)."""
    stem = name[: -len(".npy")]
    head, _, shard = stem.rpartition("_shard")
    if not head:
        head, shard = stem, "0"
    kind, _, layer = head.rpartition("_L")
    return kind, int(layer), int(shard)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


def _stage_hf(path_in_repo: str, dest: Path, revision: str = "main") -> Path:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    dest.mkdir(parents=True, exist_ok=True)
    return Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                REPO, path_in_repo, repo_type="dataset", revision=revision, local_dir=str(dest)
            ),
            what=f"hf_hub_download {path_in_repo}",
        )
    )


def _load_split_json(prefix: str, stem: str, dest: Path) -> dict:
    """Load a JSON that may be line-split into ``<stem>.partNNN`` on the Hub."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient (list materialized inside)
            api.list_repo_tree(REPO, path_in_repo=prefix, repo_type="dataset", recursive=False)
        ),
        what=f"list_repo_tree {prefix}",
    )
    names = [f.path.split("/")[-1] for f in entries if getattr(f, "size", None) is not None]
    if stem in names:
        return json.loads(_stage_hf(prefix + stem, dest).read_text())
    parts = sorted(n for n in names if n.startswith(stem + ".part"))
    if not parts:
        raise FileNotFoundError(f"neither {stem} nor {stem}.part* under {prefix}")
    text = "".join((_stage_hf(prefix + p, dest)).read_text() for p in parts)
    return json.loads(text)


def load_labels(behavior: str, stage: Path) -> dict:
    """Per-context DV + rung/split + per-ROLLOUT scores on the 0-100 scale.

    Sycophancy carries ``per_rollout_scores`` directly. Hallucination's
    labeling.json carries per-context three-way FRACTIONS only, so the
    per-rollout labels are taken from the judge's ``labeling_scores.json``
    ``three_way`` map and mapped fabricated=100 / correct|abstained=0 /
    unjudged=NaN (dropped, never coerced) -- which makes the within-context
    midpoint split exactly the rollout-level "fabricated vs correct/abstained"
    rule and the pooled split the same at midpoint 50.
    """
    import numpy as np

    dv_path = _REPO_ROOT / "eval_results/issue_1739/dv_dataset" / behavior / "labeling.json"
    payload = json.loads(dv_path.read_text())
    rows = [r for r in payload["rows"] if r.get("dv") is not None]
    ctx_order = [r["context_id"] for r in rows]
    pos = {c: i for i, c in enumerate(ctx_order)}
    dv = np.array([r["dv"] for r in rows], dtype=float)
    rung = [str(r.get("rung")) for r in rows]
    split = [str(r.get("split")) for r in rows]
    k_max = max(int(r.get("n_rollouts") or r.get("n_rollouts_judged") or 5) for r in rows)

    per_rollout = np.full((len(ctx_order), k_max), np.nan)
    if all("per_rollout_scores" in r for r in rows):
        source = "labeling.json per_rollout_scores"
        for i, r in enumerate(rows):
            for key, s in r["per_rollout_scores"].items():
                if s is not None:
                    per_rollout[i, int(key[1:])] = float(s)
    else:
        source = f"{JUDGE_PREFIX}{behavior}/labeling_scores.json three_way"
        scores = _load_split_json(f"{JUDGE_PREFIX}{behavior}/", "labeling_scores.json", stage)
        three_way = scores.get("three_way")
        if not three_way:
            raise RuntimeError(f"no per-rollout labels for {behavior}: 'three_way' absent")
        mapping = {"fabricated": 100.0, "correct": 0.0, "abstained": 0.0}
        n_unjudged = 0
        for item_id, label in three_way.items():
            cid, _, ks = item_id.rpartition("_k")
            if cid not in pos:
                continue
            val = mapping.get(label)
            if val is None:
                n_unjudged += 1
                continue
            per_rollout[pos[cid], int(ks)] = val
        logger.info("[%s] three_way: %d unjudged rollouts dropped", behavior, n_unjudged)
    logger.info(
        "[%s] labels: %d contexts, K=%d, per-rollout source=%s",
        behavior,
        len(ctx_order),
        k_max,
        source,
    )
    return {
        "ctx_order": ctx_order,
        "pos": pos,
        "dv": dv,
        "rung": rung,
        "split": split,
        "per_rollout": per_rollout,
        "per_rollout_source": source,
    }


def load_row_index(stage: Path, behavior: str) -> dict:
    """Global row order from the staged row_index shards -> per-row (ctx, k)."""
    import numpy as np

    d = stage / behavior / "row_index"
    paths = sorted(d.glob("row_index_shard*.jsonl"), key=lambda p: int(p.stem.split("_shard")[1]))
    if not paths:
        paths = sorted(d.glob("row_index.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no row_index shards under {d} (run --phase rowindex first)")
    shard_rows: list[list[dict]] = []
    for p in paths:
        rows = [json.loads(ln) for ln in p.read_text().split("\n") if ln.strip()]
        shard_rows.append(rows)
    flat = [r for rows in shard_rows for r in rows]
    for field in ("context_id", "rollout_k"):
        if field not in flat[0]:
            raise RuntimeError(f"row_index lacks {field!r}; keys={sorted(flat[0])}")
    offsets = np.cumsum([0] + [len(r) for r in shard_rows])
    return {
        "context_id": [str(r["context_id"]) for r in flat],
        "rollout_k": np.array([int(r["rollout_k"]) for r in flat], dtype=np.int64),
        "shard_offset": offsets,
        "n_rows": len(flat),
        "n_shards": len(shard_rows),
    }


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------


def phase_rowindex(args, behavior: str, stage: Path) -> None:
    out = stage / behavior / "row_index"
    out.mkdir(parents=True, exist_ok=True)
    want = re.compile(r"^row_index(?:_shard\d+)?\.jsonl$")
    n = 0
    for name, raw in stream_members(
        behavior, args.revision, workers=args.workers, window_mib=args.window_mib, want=want
    ):
        (out / name).write_bytes(raw)
        n += 1
    if n == 0:
        raise RuntimeError(f"[{behavior}] no row_index members found in tar")
    logger.info("[%s] wrote %d row_index shards to %s", behavior, n, out)


def _row_weights(labels: dict, ridx: dict, *, pooled: bool, spread_min: float):
    """Per-global-row contrast weight from the TESTED split-weights helper.

    Extraction is restricted to TRAIN-rung contexts (the held-out OOD rungs must
    stay untouched by the direction); every other row gets weight 0.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits

    pr = np.array(labels["per_rollout"], dtype=float)
    train = np.array([s == "train" for s in labels["split"]])
    pr_train = np.where(train[:, None], pr, np.nan)
    w_hi, w_lo, n_qual = fits.matched_pair_split_weights(
        pr_train, spread_min=spread_min, pooled=pooled
    )
    w = w_hi - w_lo
    pos = labels["pos"]
    w_row = np.zeros(ridx["n_rows"], dtype=np.float64)
    ks = ridx["rollout_k"]
    for i, cid in enumerate(ridx["context_id"]):
        j = pos.get(cid)
        if j is not None and ks[i] < w.shape[1]:
            w_row[i] = w[j, ks[i]]
    return w_row, int(n_qual)


def phase_directions(args, behavior: str, stage: Path) -> None:
    """Accumulate per-layer E2/E2p directions from per-rollout rows (one pass).

    ``--summary-kind`` selects the extraction POINT: ``t1`` (committed —
    answer rows) or ``context_end`` (new-arm-round fc — the SAME
    ``matched_pair_split_weights`` row weights applied to the final-context
    summaries; outputs land under ``r_b_<regime>_fc/``).
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739.constants import E2_SPREAD_MIN, HIDDEN_DIM

    kind = getattr(args, "summary_kind", "t1")
    suffix = "_fc" if is_fc(args) else ""
    # M4 (code-review r1 Minor 4): skip the 52-70 GB tar re-stream when every
    # direction npz this invocation would write already exists (a box crash
    # after directions must not re-stream on re-run). EPM_I1739_NATPV_FORCE=1
    # overrides. Directions are deterministic from (labels, tar, kind).
    out_files = [
        stage / behavior / f"r_b_{base + suffix}" / f"{behavior}.npz"
        for base, _pooled in contrast_regimes_for(args)
    ]
    if os.environ.get("EPM_I1739_NATPV_FORCE", "") != "1" and all(f.is_file() for f in out_files):
        logger.info(
            "[%s] directions already persisted (%s) — skipping tar stream "
            "(EPM_I1739_NATPV_FORCE=1 to recompute)",
            behavior,
            ", ".join(f.parent.name for f in out_files),
        )
        return
    labels = load_labels(behavior, stage / "inputs")
    ridx = load_row_index(stage, behavior)
    weights, quals = {}, {}
    for regime, pooled in contrast_regimes_for(args):
        weights[regime], quals[regime] = _row_weights(
            labels, ridx, pooled=pooled, spread_min=E2_SPREAD_MIN
        )
        logger.info(
            "[%s] %s%s: %d qualifying contexts, %d nonzero rows",
            behavior,
            regime,
            suffix,
            quals[regime],
            int((weights[regime] != 0).sum()),
        )
    acc = {r: np.zeros((28, HIDDEN_DIM), dtype=np.float64) for r in weights}
    off = ridx["shard_offset"]
    n_members = 0
    for name, arr in stream_members(
        behavior,
        args.revision,
        workers=args.workers,
        window_mib=args.window_mib,
        want=_summary_re((kind,)),
    ):
        kind, layer, shard = _parse_summary_name(name)
        lo, hi = int(off[shard]), int(off[shard + 1])
        if arr.shape[0] != hi - lo:
            raise RuntimeError(f"{name}: {arr.shape[0]} rows, row_index says {hi - lo}")
        a = np.asarray(arr, dtype=np.float64)
        for regime, w in weights.items():
            wl = w[lo:hi]
            if wl.any():
                acc[regime][layer] += wl @ a
        n_members += 1
    expect = 28 * ridx["n_shards"]
    if n_members != expect:
        raise RuntimeError(f"[{behavior}] saw {n_members} {kind} members, expected {expect}")
    layers = list(range(28))
    for regime, rb in acc.items():
        label = regime + suffix
        if suffix:
            # K2 (plan v8 §7): never fabricate a degenerate fc direction.
            norms = np.linalg.norm(np.asarray(rb, dtype=np.float64), axis=1)
            bad = [i for i, v in enumerate(norms) if not np.isfinite(v) or v == 0.0]
            if bad:
                raise SystemExit(
                    f"[natpv] K2 HALT: fc direction {label} for {behavior} degenerate "
                    f"(zero/NaN norm) at layer(s) {bad}"
                )
        out_dir = stage / behavior / f"r_b_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f"{behavior}.tmp.npz"  # np.savez appends .npz to non-.npz names
        with tmp.open("wb") as fh:
            np.savez(
                fh,
                rb=np.asarray(rb, dtype=np.float16),
                layers=np.asarray(layers),
                meta=json.dumps(
                    {
                        "behavior": behavior,
                        "regime": label,
                        "summary_kind": kind,
                        "n_qualifying_contexts": quals[regime],
                        "spread_min": E2_SPREAD_MIN,
                        "pooled": regime == "e2p",
                        "per_rollout_source": labels["per_rollout_source"],
                        "extraction_rung": "train",
                        "recipe_source": (
                            "explore_persona_space.experiments.issue_1739.fits."
                            "matched_pair_split_weights (fits.py:294-344), reused verbatim"
                        ),
                    }
                ),
            )
        os.replace(tmp, out_dir / f"{behavior}.npz")
        logger.info("[%s] wrote %s", behavior, out_dir / f"{behavior}.npz")


def _load_directions(behavior: str, stage: Path, args=None):
    """E1 + E2/E2p raw directions -> {regime: (28,d)}.

    t1 (default): E1 from the #779 bank + this round's ``r_b_e2/e2p``.
    fc (``--summary-kind context_end``): e1_fc from the SMALL npz bank the
    new-arm-round CORE fits leg writes (``--e1-fc-bank``; the fits CLI's
    ``_save_rb`` under ``--rb-point context_end``) + this driver's own
    ``r_b_e2_fc/e2p_fc`` (run ``--phase directions`` first).
    """
    import numpy as np

    fc = args is not None and is_fc(args)
    out = {}
    if fc:
        bank = Path(args.e1_fc_bank) / f"{behavior}.npz"
        if not bank.is_file():
            raise FileNotFoundError(
                f"e1_fc direction bank missing at {bank} — the new-arm-round CORE fits leg "
                "writes it (scripts/issue1739_fits.py --rb-point context_end -> "
                "_save_rb r_b_e1_fc/); run that leg first or pass --e1-fc-bank"
            )
        with np.load(bank, allow_pickle=False) as z:
            if list(z["layers"]) != list(range(28)):
                raise RuntimeError(f"e1_fc bank layers {z['layers']!r} != 0..27")
            out["e1_fc"] = np.asarray(z["rb"], dtype=np.float64)
    else:
        import torch

        p = _stage_hf(f"{RB_E1_PREFIX}{behavior}.pt", stage / "inputs", revision=RB_E1_REVISION)
        obj = torch.load(p, map_location="cpu", weights_only=False)
        rb = np.asarray(obj["r_b"], dtype=np.float64)
        if list(obj["layers"]) != list(range(28)):
            raise RuntimeError(f"E1 bank layers {obj['layers']!r} != 0..27")
        out["e1"] = rb
    suffix = "_fc" if fc else ""
    # fc drops matched-e2 (structurally undefined at context_end — plan v9).
    bases = [b for b, _p in contrast_regimes_for(args)] if fc else ["e2", "e2p"]
    for regime in bases:
        f = stage / behavior / f"r_b_{regime}{suffix}" / f"{behavior}.npz"
        with np.load(f, allow_pickle=False) as z:
            out[regime + suffix] = np.asarray(z["rb"], dtype=np.float64)
    for regime, v in out.items():
        if v.shape != (28, 3584):
            raise RuntimeError(f"{regime} direction shape {v.shape} != (28, 3584)")
    return out


def _map_projectors(variant: str, directions: dict, stage: Path):
    """Precompute ``W v`` + ``y_mu . v`` per (regime, layer) for one map arm.

    ``pred . v = ((x - x_mu)/x_sd) . (W v) + y_mu . v`` -- so the (n, 3584)
    predictions are never materialized.
    """
    import numpy as np

    local = (
        _REPO_ROOT / "data/issue_1739/hf_dl/i1739_tensors" / MAPS_PREFIX / f"{variant}__ufull.npz"
    )
    path = (
        local
        if local.is_file()
        else _stage_hf(f"{MAPS_PREFIX}{variant}__ufull.npz", stage / "inputs")
    )
    from explore_persona_space.experiments.issue_1739 import fits

    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if list(z["layers"]) != list(range(28)):
            raise RuntimeError(f"map {variant} layers != 0..27")
        w = z["w"]
        x_mu = np.asarray(z["x_mu"], dtype=np.float64)
        x_sd = np.asarray(z["x_sd"], dtype=np.float64)
        y_mu = np.asarray(z["y_mu"], dtype=np.float64)
        wv, ymuv = {}, {}
        for regime, v in directions.items():
            wv[regime] = np.stack([np.asarray(w[ly], dtype=np.float64) @ v[ly] for ly in range(28)])
            ymuv[regime] = np.array([float(y_mu[ly, 0] @ v[ly]) for ly in range(28)])
        del w
    # #1975 input-space parity: the raw path streams RAW rows through a map
    # whose apply contract is WHITENED space — the #1739 incident seam, kept
    # only as a DISCLOSED provisional read (module docstring + the cube meta
    # "space" field). Declared, never silent; the whitened path is faithful.
    fits.assert_map_input_space(
        meta, None, declared_mismatch="--space raw provisional read (disclosed; natpv docstring)"
    )
    logger.info("[map %s] apply=%r", variant, meta.get("apply"))
    return {"wv": wv, "ymuv": ymuv, "x_mu": x_mu, "x_sd": x_sd, "meta": meta, "path": str(path)}


def is_fc(args) -> bool:
    """True when the run extracts directions at the FINAL-CONTEXT token
    (new-arm-round item 1: ``--summary-kind context_end``)."""
    return getattr(args, "summary_kind", "t1") == "context_end"


def regimes_for(args) -> tuple[str, ...]:
    """Effective regime labels: fc runs suffix labels with ``_fc`` so the fc
    cube/reduce/direction artifacts can never collide with committed t1 ones
    (the fits-CLI ``--rb-point`` convention, mirrored). The fc set is
    ``("e1_fc", "e2p_fc")`` ONLY — matched-e2_fc is structurally undefined:
    the within-context hi/lo weights cancel exactly on context-level rows
    (plan v9 structural restriction, concern
    e2fc-structurally-null-direction)."""
    return ("e1_fc", "e2p_fc") if is_fc(args) else REGIMES


def contrast_regimes_for(args) -> tuple[tuple[str, bool], ...]:
    """(base regime, pooled) pairs phase_directions builds: fc drops matched
    e2 per the structural restriction above; t1 keeps the committed pair."""
    return (("e2p", True),) if is_fc(args) else (("e2", False), ("e2p", True))


def base_regime(regime: str) -> str:
    """``e2_fc`` -> ``e2`` (the semantic regime under an fc label)."""
    return regime.removesuffix("_fc")


def cube_dir_name(args) -> str:
    """Per-space cube dir — the raw path keeps its legacy ``cube`` name; fc
    runs get a ``_fc``-suffixed sibling (never the committed t1 dir)."""
    stem = "cube" if args.space == "raw" else f"cube_{args.space}"
    return stem + ("_fc" if is_fc(args) else "")


def reduce_out_name(args) -> str:
    """Per-space reduce output — the raw path keeps its legacy filename; fc
    runs write a ``_fc``-suffixed sibling."""
    stem = "regime_comparison" if args.space == "raw" else f"regime_comparison_{args.space}"
    return stem + ("_fc" if is_fc(args) else "") + ".json"


def whitening_path(args, variant: str) -> Path:
    return (
        Path(args.whitening_root)
        / WHITEN_DIR
        / WHITEN_FILE_FMT.format(variant=variant, u_label=args.u_size)
    )


def phase_whitening(args, behavior: str, stage: Path) -> None:
    """Fit + PERSIST the U-pool shrinkage whitening, once per variant.

    BEHAVIOR-INDEPENDENT: the U pool is the shared #1092 slice (fit-pool rows
    only, the ``is_eval_only`` exclusion), so the transform is keyed by
    (variant, u_size) alone — the per-behavior phase loop calls this, and every
    call after the first short-circuits on the persisted file. Reuses
    ``fits.fit_whitening`` VERBATIM (same shrinkage grid, holdout frac and seed
    the main grid passes), so the persisted transform IS the main grid's.

    Peak RAM is the fp64 U-pool promotion inside ``fit_whitening``
    (Ly x n_u x d x 8 B; ~15 GB at 28 x 18,793 x 3,584) plus the fp32 stack it
    is promoted from — the ONE resident-grid step in the whitened path, and the
    reason this phase is sized for the GPU box rather than the shared VM.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits, store_io

    todo = [v for v in VARIANTS if not whitening_path(args, v).is_file()]
    if not todo:
        logger.info("[%s] whitening already persisted for %s — skipping fit", behavior, VARIANTS)
        return
    store_io.stage_u_store(Path(args.u_store), tuple(todo), tuple(range(28)))
    u_arrays, u_meta = store_io.load_summaries(
        Path(args.u_store), tuple(todo), tuple(range(28)), hidden_dim=3584
    )
    rows = np.flatnonzero(store_io.fit_pool_mask(u_meta))
    if args.u_size != "full":
        rng = np.random.default_rng([1739, 9, int(args.whiten_seed)])
        want = int(args.u_size)
        if want < len(rows):
            rows = np.sort(rng.choice(rows, size=want, replace=False))
    logger.info("[whitening] U pool: %d fit rows (u_size=%s)", len(rows), args.u_size)
    for variant in todo:
        t0 = time.time()
        u_x = np.stack([u_arrays[(variant, ly)][rows] for ly in range(28)])
        logger.info(
            "[whitening] %s: fitting on %s (%.1f GB fp64 promotion)",
            variant,
            u_x.shape,
            u_x.size * 8 / 1e9,
        )
        wh = fits.fit_whitening(u_x, device=args.whiten_device, seed=int(args.whiten_seed))
        del u_x
        out = whitening_path(args, variant)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name.replace(".npz", ".tmp.npz"))
        with tmp.open("wb") as fh:
            np.savez(
                fh,
                # fp32 matches the persisted-map precedent (_save_map); the
                # projection vectors are recomputed in fp64 from these, and a
                # ~1e-7 relative error is immaterial to a rank correlation.
                mu=np.asarray(wh.mu, dtype=np.float32),
                w=np.asarray(wh.w, dtype=np.float32),
                gamma=np.asarray(wh.gamma, dtype=np.float64),
                meta=json.dumps(
                    {
                        "variant": variant,
                        "u_size": args.u_size,
                        "n_u_rows": int(len(rows)),
                        "seed": int(args.whiten_seed),
                        "behavior_independent": True,
                        "recipe_source": (
                            "explore_persona_space.experiments.issue_1739.fits.fit_whitening, "
                            "reused verbatim (same shrinkage grid / holdout frac / seed as the "
                            "main-grid fits CLI)"
                        ),
                        "git_commit": _git_commit(),
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                ),
            )
        os.replace(tmp, out)
        logger.info(
            "[whitening] %s: wrote %s (gammas=%s) in %.1f min",
            variant,
            out,
            np.asarray(wh.gamma).tolist(),
            (time.time() - t0) / 60,
        )
        del wh


def _load_whitening(args, variant: str):
    """``(mu (28,d) fp64, w (28,d,d) fp32, meta)`` for one variant."""
    import numpy as np

    path = whitening_path(args, variant)
    if not path.is_file():
        raise FileNotFoundError(
            f"whitening missing at {path} — run --phase whitening first (--space whitened)"
        )
    with np.load(path, allow_pickle=False) as z:
        return (
            np.asarray(z["mu"], dtype=np.float64),
            np.asarray(z["w"]),
            json.loads(str(z["meta"])),
        )


def _whitened_projectors(args, directions: dict, stage: Path) -> tuple[dict, dict]:
    """Per (read, regime, layer): ONE (d,) vector + ONE scalar — no whitened grid.

    Realizes the affine collapse in the module docstring. Returns
    ``({read: {regime: (vec (28,d), const (28,))}}, provenance)``. The whitening
    matrix for a variant is loaded, consumed layer-by-layer in fp64, and freed
    before the next variant, so peak RAM is one variant's fp32 matrices
    (~1.4 GB) plus one layer's fp64 promotion (~103 MB).
    """
    import numpy as np

    reads = {  # read -> (whitening variant, map variant or None)
        "ctx": ("context_end", None),
        "pre": ("prefix_end", None),
        "map_ctx": ("context_end", "context_end"),
        "map_pre": ("prefix_end", "prefix_end"),
        "oracle": ("context_end", None),
        ORACLE_PRE_READ: ("prefix_end", None),
    }
    from explore_persona_space.experiments.issue_1739 import fits

    maps = {v: _map_projectors_raw(v, stage) for v in VARIANTS}
    out: dict[str, dict] = {r: {} for r in reads}
    prov: dict[str, dict] = {"map_meta": {v: maps[v]["meta"] for v in VARIANTS}}
    for variant in VARIANTS:
        mu, w_mat, wmeta = _load_whitening(args, variant)
        prov[variant] = {k: wmeta.get(k) for k in ("u_size", "n_u_rows", "seed", "recipe_source")}
        # #1975 map<->whitening parity: the whitening THIS fold projects under
        # must be the one the map payload was fit under. The loaded side's
        # provenance is built from the persisted whitening file (artifact sha
        # computable; seed/u_size in its meta dict, gammas in the npz `gamma`
        # ARRAY — fp64, NOT the meta dict). Legacy payloads (no recorded
        # provenance) degrade to a loud warning; a mismatch on comparable
        # fields RAISES (fail fast, the #1739 incident class).
        wpath = whitening_path(args, variant)
        with np.load(wpath, allow_pickle=False) as z:
            _gammas = np.asarray(z["gamma"], dtype=np.float64)
        loaded_prov = fits.whitening_provenance(
            whitening_file=wpath,
            variant=variant,
            u_label=wmeta.get("u_size"),
            whiten_seed=wmeta.get("seed"),
            n_u_rows=wmeta.get("n_u_rows"),
            gammas=_gammas,
        )
        prov[variant]["map_whitening_parity"] = fits.check_whitening_parity(
            (maps[variant]["meta"] or {}).get("whitening_provenance"), loaded_prov
        )
        wants = [r for r, (wv, _) in reads.items() if wv == variant]
        for read in wants:
            map_variant = reads[read][1]
            mp = maps[map_variant] if map_variant else None
            for regime, rb_raw in directions.items():
                vec = np.zeros((28, rb_raw.shape[1]), dtype=np.float64)
                const = np.zeros(28, dtype=np.float64)
                for ly in range(28):
                    wl = np.asarray(w_mat[ly], dtype=np.float64)  # symmetric
                    rb_w = rb_raw[ly] @ wl  # the main grid's whitened direction
                    if mp is None:
                        v = wl @ rb_w
                        c = -float(mu[ly] @ v)
                    else:
                        # x_mu/x_sd/y_mu arrive squeezed to (Ly, d) by
                        # _map_row_vecs, so every term here is 1-D per layer.
                        g = np.asarray(mp["w"][ly], dtype=np.float64) @ rb_w
                        h = g / mp["x_sd"][ly]
                        v = wl @ h
                        c = (
                            -float(mu[ly] @ v)
                            - float(mp["x_mu"][ly] @ h)
                            + float(mp["y_mu"][ly] @ rb_w)
                        )
                    vec[ly], const[ly] = v, c
                out[read][regime] = (vec, const)
        del w_mat
    # Drop our own references to the map payloads (the big fp16 `w` blocks);
    # clearing the payload dicts themselves would mutate objects the caller may
    # still hold.
    maps.clear()
    return out, prov


def _map_row_vecs(arr, name: str, variant: str, n_layers: int = 28):
    """``(Ly, 1, d)`` -> ``(Ly, d)`` fp64, fail-loud on any other layout.

    The persisted map writes ``x_mu`` / ``x_sd`` / ``y_mu`` as ``(Ly, 1, d)``
    (``fits.MapFit`` field annotations; VERIFIED against the real
    ``{context_end,prefix_end}__ufull.npz`` headers on the data repo —
    ``w (28, 3584, 3584) fp16``, the other three ``(28, 1, 3584) fp32``, both
    variants identical). The RAW path never noticed because it only ever
    BROADCASTS these against an ``(n, d)`` block; the whitened fold does
    per-layer 1-D algebra, so the singleton axis has to come off at the LOAD
    boundary — one place — rather than at each use site.
    """
    import numpy as np

    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3 and a.shape[1] == 1:
        a = a[:, 0, :]
    if a.ndim != 2 or a.shape[0] != n_layers:
        raise RuntimeError(
            f"map {variant}: {name} has unexpected layout {np.shape(arr)} — expected "
            f"({n_layers}, 1, d) per fits.MapFit (or ({n_layers}, d)); the whitened fold "
            "cannot index it per layer"
        )
    return a


def _map_projectors_raw(variant: str, stage: Path) -> dict:
    """Load a persisted map's raw arrays (``w``/``x_mu``/``x_sd``/``y_mu``).

    Unlike :func:`_map_projectors` (which folds a RAW direction in immediately),
    this keeps the arrays so the whitened path can fold ``W`` in as well. The
    map's ``apply`` contract is ``pred = ((x - x_mu)/x_sd) @ w + y_mu`` in
    WHITENED space — which is exactly why the whitened path is the faithful one.

    ``x_mu`` / ``x_sd`` / ``y_mu`` are returned SQUEEZED to ``(Ly, d)``; ``w``
    stays ``(Ly, d, d)``.
    """
    import numpy as np

    local = (
        _REPO_ROOT / "data/issue_1739/hf_dl/i1739_tensors" / MAPS_PREFIX / f"{variant}__ufull.npz"
    )
    path = (
        local
        if local.is_file()
        else _stage_hf(f"{MAPS_PREFIX}{variant}__ufull.npz", stage / "inputs")
    )
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if list(z["layers"]) != list(range(28)):
            raise RuntimeError(f"map {variant} layers != 0..27")
        out = {
            "w": np.asarray(z["w"]),
            "x_mu": _map_row_vecs(z["x_mu"], "x_mu", variant),
            "x_sd": _map_row_vecs(z["x_sd"], "x_sd", variant),
            "y_mu": _map_row_vecs(z["y_mu"], "y_mu", variant),
            "meta": meta,
            "path": str(path),
        }
    logger.info("[map %s] apply=%r (whitened-space fold)", variant, meta.get("apply"))
    return out


def phase_project(args, behavior: str, stage: Path) -> None:
    """Per-row scalar projection cube for all reads x regimes x layers (one pass)."""
    import numpy as np

    # M4 (code-review r1 Minor 4): skip the full-tar re-stream when the cube
    # already exists AND is newer than every direction npz it consumed (a
    # stale cube after a directions re-run must rebuild). Force with
    # EPM_I1739_NATPV_FORCE=1.
    cube_path = stage / behavior / cube_dir_name(args) / "cube.npz"
    fc_suffix = "_fc" if is_fc(args) else ""
    dir_files = [
        stage / behavior / f"r_b_{b + fc_suffix}" / f"{behavior}.npz"
        for b, _p in contrast_regimes_for(args)
    ]
    if is_fc(args):
        dir_files.append(Path(args.e1_fc_bank) / f"{behavior}.npz")
    if (
        os.environ.get("EPM_I1739_NATPV_FORCE", "") != "1"
        and cube_path.is_file()
        and all(f.is_file() for f in dir_files)
        and cube_path.stat().st_mtime >= max(f.stat().st_mtime for f in dir_files)
    ):
        logger.info(
            "[%s] cube already persisted and newer than its directions (%s) — skipping "
            "tar stream (EPM_I1739_NATPV_FORCE=1 to recompute)",
            behavior,
            cube_path,
        )
        return
    ridx = load_row_index(stage, behavior)
    directions = _load_directions(behavior, stage, args)
    regimes = regimes_for(args)
    whitened = args.space == "whitened"
    reads = (*READS, ORACLE_PRE_READ) if whitened else READS
    maps: dict = {}
    proj: dict = {}
    whiten_prov: dict = {}
    if whitened:
        proj, whiten_prov = _whitened_projectors(args, directions, stage)
    else:
        maps = {v: _map_projectors(v, directions, stage) for v in VARIANTS}
    n = ridx["n_rows"]
    cube = {
        read: {r: np.full((28, n), np.nan, dtype=np.float32) for r in regimes} for read in reads
    }
    prefix_hashes: dict[int, str] = {}
    off = ridx["shard_offset"]
    kind_read = {"context_end": "ctx", "prefix_end": "pre", "t1": "oracle"}
    kind_map_read = {"context_end": "map_ctx", "prefix_end": "map_pre"}
    # Under whitening the oracle read is variant-dependent (the main grid
    # whitens the answer acts with the context arm's own transform), so the t1
    # member feeds BOTH oracle reads.
    kind_reads_w = {
        "context_end": ("ctx", "map_ctx"),
        "prefix_end": ("pre", "map_pre"),
        "t1": ("oracle", ORACLE_PRE_READ),
    }
    n_members = 0
    for name, arr in stream_members(
        behavior,
        args.revision,
        workers=args.workers,
        window_mib=args.window_mib,
        want=_summary_re(("context_end", "prefix_end", "t1")),
    ):
        kind, layer, shard = _parse_summary_name(name)
        lo, hi = int(off[shard]), int(off[shard + 1])
        if arr.shape[0] != hi - lo:
            raise RuntimeError(f"{name}: {arr.shape[0]} rows, row_index says {hi - lo}")
        a = np.asarray(arr, dtype=np.float64)
        if whitened:
            # Every whitened read is affine in the RAW row: x . vec + const.
            for read in kind_reads_w[kind]:
                for regime in regimes:
                    vec, const = proj[read][regime]
                    cube[read][regime][layer, lo:hi] = a @ vec[layer] + const[layer]
        else:
            for regime, v in directions.items():
                cube[kind_read[kind]][regime][layer, lo:hi] = a @ v[layer]
            if kind in kind_map_read:
                mp = maps[kind]
                xs = (a - mp["x_mu"][layer]) / mp["x_sd"][layer]
                for regime in regimes:
                    cube[kind_map_read[kind]][regime][layer, lo:hi] = (
                        xs @ mp["wv"][regime][layer] + mp["ymuv"][regime][layer]
                    )
        if kind == "prefix_end" and layer == PREFIX_HASH_LAYER:
            for i in range(a.shape[0]):
                prefix_hashes[lo + i] = hashlib.blake2b(
                    np.asarray(arr[i]).tobytes(), digest_size=8
                ).hexdigest()
        n_members += 1
    expect = 28 * 3 * ridx["n_shards"]
    if n_members != expect:
        raise RuntimeError(f"[{behavior}] saw {n_members} summary members, expected {expect}")
    out = stage / behavior / cube_dir_name(args)
    out.mkdir(parents=True, exist_ok=True)
    payload = {f"{read}__{regime}": cube[read][regime] for read in reads for regime in regimes}
    n_distinct_prefix = len(set(prefix_hashes.values()))
    tmp = out / "cube.tmp.npz"
    with tmp.open("wb") as fh:
        np.savez(
            fh,
            **payload,
            context_id=np.asarray(ridx["context_id"]),
            rollout_k=ridx["rollout_k"],
            meta=json.dumps(
                {
                    "behavior": behavior,
                    "n_rows": n,
                    "reads": list(reads),
                    "regimes": list(regimes),
                    "summary_kind": getattr(args, "summary_kind", "t1"),
                    "n_distinct_prefix_states_L14": n_distinct_prefix,
                    "map_meta": (
                        whiten_prov.get("map_meta")
                        if whitened
                        else {v: maps[v]["meta"] for v in VARIANTS}
                    ),
                    "space": (
                        "WHITENED main-grid space (U-pool shrinkage whitening folded into the "
                        "direction; numerically comparable to the committed main-grid columns)"
                        if whitened
                        else "RAW activation space (NOT the whitened main-grid space)"
                    ),
                    "whitening": whiten_prov or None,
                }
            ),
        )
    os.replace(tmp, out / "cube.npz")
    logger.info(
        "[%s] cube written (%d rows, %d distinct prefix states @L%d)",
        behavior,
        n,
        n_distinct_prefix,
        PREFIX_HASH_LAYER,
    )


def _spearman(x, y) -> float:
    import numpy as np
    from scipy import stats

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    if np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def phase_reduce(args, behavior: str, stage: Path) -> None:
    """Per-context reduction + Spearman table + frozen layers -> JSON."""
    import numpy as np

    labels = load_labels(behavior, stage / "inputs")
    with np.load(stage / behavior / cube_dir_name(args) / "cube.npz", allow_pickle=False) as z:
        cube_meta = json.loads(str(z["meta"]))
        row_ctx = [str(c) for c in z["context_id"]]
        cube = {k: z[k] for k in z.files if "__" in k}
    pos = labels["pos"]
    n_ctx = len(labels["ctx_order"])
    # group rows by context (first-occurrence for per-context states, mean for t1)
    first = np.full(n_ctx, -1, dtype=np.int64)
    counts = np.zeros(n_ctx, dtype=np.int64)
    row_idx = np.full(len(row_ctx), -1, dtype=np.int64)
    for i, cid in enumerate(row_ctx):
        j = pos.get(cid)
        if j is None:
            continue
        row_idx[i] = j
        counts[j] += 1
        if first[j] < 0:
            first[j] = i
    have = first >= 0
    logger.info("[%s] %d/%d labeled contexts present in store", behavior, int(have.sum()), n_ctx)

    per_ctx: dict[str, np.ndarray] = {}
    for key, arr in cube.items():
        read = key.split("__")[0]
        # Both oracle reads (raw `oracle`; whitened `oracle` + `oracle_pre`) are
        # per-ROLLOUT t1 projections and reduce by mean over a context's rows.
        if read.startswith("oracle"):
            sums = np.zeros((28, n_ctx))
            cnt = np.zeros(n_ctx)
            sel = row_idx >= 0
            np.add.at(cnt, row_idx[sel], 1.0)
            for ly in range(28):
                np.add.at(sums[ly], row_idx[sel], arr[ly][sel].astype(np.float64))
            with np.errstate(invalid="ignore", divide="ignore"):
                per_ctx[key] = sums / np.where(cnt > 0, cnt, np.nan)
        else:
            vals = np.full((28, n_ctx), np.nan)
            vals[:, have] = arr[:, first[have]].astype(np.float64)
            per_ctx[key] = vals

    dv = labels["dv"]
    rung_arr = np.array(labels["rung"])
    rungs = sorted(set(labels["rung"]))
    table: dict = {}
    reads = tuple(cube_meta.get("reads") or READS)
    regimes = tuple(cube_meta.get("regimes") or regimes_for(args))
    for read in reads:
        table[read] = {}
        for regime in regimes:
            vals = per_ctx[f"{read}__{regime}"]
            per_layer = {
                rung: [
                    _spearman(vals[ly][rung_arr == rung], dv[rung_arr == rung]) for ly in range(28)
                ]
                for rung in rungs
            }
            train = np.asarray(per_layer.get("train", [np.nan] * 28))
            frozen = int(np.nanargmax(np.abs(train))) if np.isfinite(train).any() else -1
            table[read][regime] = {
                "per_layer_rho": per_layer,
                "frozen_layer": frozen,
                "frozen_layer_basis": "max |rho| on the train rung",
                "rho_at_frozen_layer": {
                    rung: (per_layer[rung][frozen] if frozen >= 0 else float("nan"))
                    for rung in rungs
                },
                "in_sample_train_rung": base_regime(regime) in ("e2", "e2p"),
            }
    e2_meta = {}
    suffix = "_fc" if is_fc(args) else ""
    # fc has no matched-e2 direction (plan v9 structural restriction).
    for base, _pooled in contrast_regimes_for(args):
        regime = base + suffix
        with np.load(
            stage / behavior / f"r_b_{regime}" / f"{behavior}.npz", allow_pickle=False
        ) as z:
            e2_meta[regime] = json.loads(str(z["meta"]))
    whitened = args.space == "whitened"
    read_docs = {
        "ctx": "direction . context_end state",
        "pre": "direction . prefix_end state",
        "map_ctx": "direction . ufull-map(context_end) predicted answer state",
        "map_pre": "direction . ufull-map(prefix_end) predicted answer state",
        "oracle": "direction . TRUE per-context mean answer state (t1)",
    }
    if whitened:
        read_docs["oracle"] += " [context_end whitening — the main grid's primary arm]"
        read_docs[ORACLE_PRE_READ] = (
            "direction . TRUE per-context mean answer state (t1) [prefix_end whitening]"
        )
    space_caveats = (
        [
            "WHITENED-SPACE: reads project in the main grid's U-pool shrinkage-whitened space, "
            "reproduced EXACTLY by folding the (linear, symmetric) whitening into the direction "
            "instead of the data -- score = x . (W rb_w) + const -- so no whitened activation "
            "grid is materialized and these rho values ARE numerically comparable to the "
            "committed main-grid columns. The E1 anchor runs through this identical whitened "
            "path, so the E1/E2/E2p comparison is internally matched.",
            "ORACLE IS VARIANT-DEPENDENT under whitening (the main grid whitens the answer acts "
            "with the context arm's own transform): 'oracle' uses the context_end whitening and "
            "'oracle_pre' the prefix_end one. In raw space the two coincide, hence the single "
            "'oracle' read there.",
        ]
        if whitened
        else [
            "RAW-SPACE: the evil main-grid arms project in WHITENED space (U-pool shrinkage "
            "whitening, fit fresh in-CLI and never persisted). These rho values are therefore "
            "NOT numerically comparable to the committed whitened main-grid values; the E1 "
            "anchor is run through this identical raw-space path so the E1/E2/E2p comparison "
            "is internally matched. Run --space whitened for the comparable columns.",
            "RAW-SPACE MAP ARMS: the persisted map's own contract is "
            "'pred = ((x - x_mu)/x_sd) @ w + y_mu (whitened space)', so applying it to RAW "
            "activations mismatches its fitted input space -- the map_ctx / map_pre raw reads "
            "are indicative only.",
        ]
    )
    out_payload = {
        "behavior": behavior,
        "n_contexts_by_rung": {r: int((rung_arr == r).sum()) for r in rungs},
        "regime_table": table,
        "meta": {
            "space": "WHITENED main-grid space" if whitened else "RAW activation space",
            "reads": read_docs,
            "extraction": e2_meta,
            "per_rollout_source": labels["per_rollout_source"],
            "n_distinct_prefix_states_L14": cube_meta.get("n_distinct_prefix_states_L14"),
            "map_meta": cube_meta.get("map_meta"),
            "whitening": cube_meta.get("whitening"),
            "caveats": [
                "IN-SAMPLE: E2/E2p directions were extracted from the TRAIN-rung labels, so their "
                "train-rung rho is in-sample by construction; the OOD rungs are held out.",
                *space_caveats,
                "u-full MAP OOD: the 963k round measured this map extrapolating with strongly "
                "negative reconstruction R2 on behavior eval distributions -- a distribution-"
                "coverage caveat on the map_ctx / map_pre reads, not a serialization bug.",
                "E2p RECIPE: pooled global-MIDPOINT split over all kept per-ROLLOUT scores "
                "(fits.matched_pair_split_weights pooled=True), NOT a top-K/bottom-K context split.",
            ],
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    out_dir = _REPO_ROOT / "eval_results/issue_1739/nat_pv_regimes" / behavior
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / reduce_out_name(args)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out_payload, indent=1))
    os.replace(tmp, out)
    logger.info("[%s] wrote %s", behavior, out)


def _git_commit() -> str:
    import subprocess

    p = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT, check=False
    )
    return p.stdout.strip() if p.returncode == 0 else "unknown"


PHASES = {
    "rowindex": phase_rowindex,
    "directions": phase_directions,
    "whitening": phase_whitening,
    "project": phase_project,
    "reduce": phase_reduce,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behavior", action="append", choices=BEHAVIORS, help="repeatable")
    ap.add_argument("--phase", action="append", choices=sorted(PHASES), required=True)
    ap.add_argument("--stage", default="/mnt/eps-data/thomasjiralerspong/issue1739_natpv")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--window-mib", type=int, default=32)
    ap.add_argument(
        "--space",
        default="raw",
        choices=SPACES,
        help="projection space: raw (legacy, provisional) | whitened (main-grid-comparable)",
    )
    ap.add_argument("--u-store", type=Path, default=U_STORE_DEFAULT, help="staged #1092 U-pool")
    ap.add_argument(
        "--whitening-root",
        type=Path,
        default=Path("data/issue_1739"),
        help="where the behavior-INDEPENDENT persisted whitening lives",
    )
    ap.add_argument("--u-size", default="full", help="U-pool rung for the whitening fit")
    ap.add_argument("--whiten-device", default="cuda", help="fit_whitening device (cpu|cuda)")
    ap.add_argument("--whiten-seed", type=int, default=0, help="matches the fits CLI --seeds[0]")
    ap.add_argument(
        "--summary-kind",
        choices=("t1", "context_end"),
        default="t1",
        help="direction extraction POINT (new-arm-round item 1): 't1' (committed answer-avg; "
        "byte-identical behavior) or 'context_end' (final-context fc directions; every regime "
        "label + cube/reduce artifact gains an '_fc' suffix)",
    )
    ap.add_argument(
        "--e1-fc-bank",
        type=Path,
        default=Path("analysis_tensors/issue_1739/r_b_e1_fc"),
        help="dir holding <behavior>.npz e1_fc directions (written by the new-arm-round CORE "
        "fits leg via --rb-point context_end); read only under --summary-kind context_end",
    )
    args = ap.parse_args(argv)
    if "whitening" in args.phase and args.space != "whitened":
        ap.error("--phase whitening is only meaningful with --space whitened")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    stage = Path(args.stage)
    behaviors = args.behavior or list(BEHAVIORS)
    for behavior in behaviors:
        for phase in args.phase:
            logger.info("=== [phase=%s behavior=%s] ===", phase, behavior)
            t0 = time.time()
            PHASES[phase](args, behavior, stage)
            logger.info(
                "=== [phase=%s behavior=%s] done in %.1f min ===",
                phase,
                behavior,
                (time.time() - t0) / 60,
            )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit before C-extension finalize teardown


if __name__ == "__main__":
    main()
