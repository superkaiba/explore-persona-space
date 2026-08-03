#!/usr/bin/env python3
"""#1768 inline round: refit the context->answer maps on LAST-TOKEN context.

Consumes the last-token context stores written by ``issue1768_lasttoken.py``
and the EXISTING round-1 answer-side stores (span-mean over the response), so
the only thing that changes versus round 1 is the CONTEXT summary. Rows are
joined by prompt sha, never by order.

Per (arm, layer) it refits the same three maps round 1 fit -- M0 (base
context -> base answer), M+ (trained context -> trained answer), M+_tf
(trained context -> matched-text trained answer) -- under the identical
pinned splits, lambda grid, refit-noise floor and paired-bootstrap verdict
machinery imported from ``issue1768_fit``, and emits D_lt per cell plus the
identity+learned-bias and kNN-retrieval reads.

Round-1 answer-side stores are streamed one arm at a time (download -> extract
the response span -> delete) so peak disk stays ~2 GB rather than the ~77 GB
the full grid would need.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# load_dotenv() BEFORE numpy/torch: the shared-VM thread caps (#847) are
# setdefault-ed here and BLAS/torch freeze their pools at import.
load_dotenv()

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1768_fit as F  # noqa: E402
import issue1768_lasttoken as LT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.lt_fit")

RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_1768" / "lasttoken_repool"


def _meta() -> dict:
    return LT._meta()


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LT._atomic_json(path, obj)


# ── round-1 answer-side stores: stream one at a time ─────────────────────────


def _response_arrays(store_path: Path, layers: list[int], key: str) -> tuple[dict, list[str]]:
    import torch

    store = torch.load(store_path, map_location="cpu", weights_only=False)
    out = {
        li: np.asarray(store["arms"][key][li].float().numpy(), dtype=np.float64) for li in layers
    }
    return out, list(store["row_sha"])


def _slim_path(cache: Path, kind: str, unit: str) -> Path:
    return cache / f"{kind}__{unit}.npz"


def fetch_response(
    cache: Path,
    kind: str,
    unit: str,
    layers: list[int],
    *,
    persist: bool = False,
) -> tuple[dict, list[str]]:
    """Response-span arrays for one unit, via a delete-after-extract stage.

    ``kind`` is ``corpus_capture`` (file ``pooled.pt``) or
    ``corpus_capture_tf`` (``pooled_tf.pt``).

    ``persist`` caches the extracted arrays as a slim fp16 ``.npz``. Cache ONLY
    the BASE units: they are re-read by every arm, while a per-arm plus/tf store
    is consumed exactly once. Persisting per-arm stores would write ~1.4 GB x 72
    arms x 2 kinds of npz on top of the ~52 GB of last-token stores and blow the
    ~130 GB per-pod MooseFS quota (gotchas.md EDQUOT entry). fp16 is LOSSLESS
    here — the round-1 stores are themselves fp16.
    """
    from explore_persona_space.orchestrate import hub

    # Extract + cache the FULL layer set, never just the requested `layers`:
    # build_cell asks one layer at a time, so a request-scoped cache would store
    # L14 only and then KeyError on the L19 lookup. The cache key is
    # (kind, unit) — its CONTENT must therefore be layer-complete.
    all_layers = sorted(set(X.LAYERS) | set(layers))
    slim = _slim_path(cache, kind, unit)
    if slim.exists():
        z = np.load(slim, allow_pickle=False)
        if all(f"L{li}" in z.files for li in layers):  # hit must cover the request
            shas = json.loads(slim.with_suffix(".sha.json").read_text())["row_sha"]
            return {li: z[f"L{li}"].astype(np.float64) for li in layers}, shas
        logger.info("[stage] %s/%s cache lacks %s — re-extracting", kind, unit, layers)

    fname = "pooled_tf.pt" if kind == "corpus_capture_tf" else "pooled.pt"
    tmp_dir = cache / "_stage"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / f"{kind}__{unit}__{fname}"
    hub.stage_hub_file(
        X.HF_DATA_REPO,
        f"{X.HF_PREFIX}/{kind}/{unit}/{fname}",
        target,
        repo_type="dataset",
        overwrite=True,
    )
    try:
        arrs, shas = _response_arrays(target, all_layers, "response")
    finally:
        target.unlink(missing_ok=True)
    if persist:
        # PER-INVOCATION tmp name: concurrent shards otherwise write the SAME
        # `.tmp.npz` and the first os.replace steals it from the rest
        # (FileNotFoundError at the replace — the documented fan-out
        # shared-staging race, gotchas.md). The `.tmp.<pid>.npz` suffix keeps
        # np.savez from appending a second `.npz`, and a lost race is benign:
        # the winner published identical bytes, so we just drop our copy.
        tmp_npz = slim.with_suffix(f".tmp.{os.getpid()}.npz")
        np.savez(tmp_npz, **{f"L{li}": arrs[li].astype(np.float16) for li in all_layers})
        try:
            os.replace(tmp_npz, slim)
        except OSError:
            tmp_npz.unlink(missing_ok=True)
            if not slim.exists():
                raise
        _atomic_json(slim.with_suffix(".sha.json"), {"row_sha": shas, "unit": unit, "kind": kind})
    logger.info(
        "[stage] %s/%s response extracted (%d rows, layers=%s, cached=%s)",
        kind,
        unit,
        len(shas),
        all_layers,
        persist,
    )
    return {li: arrs[li] for li in layers}, shas


def load_lasttoken(
    out_root: Path, unit: str, layers: list[int], position: str
) -> tuple[dict, list]:
    import torch

    p = out_root / "lasttoken" / unit / "lasttoken.pt"
    assert p.exists(), f"missing last-token store: {p}"
    store = torch.load(p, map_location="cpu", weights_only=False)
    arrs = {
        li: np.asarray(store["arms"][position][li].float().numpy(), dtype=np.float64)
        for li in layers
    }
    return arrs, list(store["row_sha"])


# ── cell assembly (last-token context, round-1 answers) ──────────────────────


def build_cell(
    out_root: Path,
    cache: Path,
    arm_id: str,
    layer: int,
    position: str,
) -> dict:
    """The round-1 ``load_corpus_cell`` with C swapped for last-token vectors."""
    base_unit = X.base_unit_for(arm_id)
    layers = [layer]

    C0_by, c0_sha = load_lasttoken(out_root, base_unit, layers, position)
    Cp_by, cp_sha = load_lasttoken(out_root, arm_id, layers, position)
    # base is re-read by every arm -> cache it; per-arm stores are consumed once
    # and must NOT be cached (per-pod quota, see fetch_response).
    V0_by, v0_sha = fetch_response(cache, "corpus_capture", base_unit, layers, persist=True)
    Vp_by, vp_sha = fetch_response(cache, "corpus_capture", arm_id, layers)
    Vt_by, vt_sha = fetch_response(cache, "corpus_capture_tf", arm_id, layers)

    # sha-keyed join in the round-1 BASE store's order (round 1's convention)
    ix = {
        "c0": {s: i for i, s in enumerate(c0_sha)},
        "cp": {s: i for i, s in enumerate(cp_sha)},
        "vp": {s: i for i, s in enumerate(vp_sha)},
        "vt": {s: i for i, s in enumerate(vt_sha)},
    }
    keep = [
        (i, s)
        for i, s in enumerate(v0_sha)
        if s in ix["c0"] and s in ix["cp"] and s in ix["vp"] and s in ix["vt"]
    ]
    assert len(keep) >= 0.9 * len(v0_sha), (arm_id, layer, len(keep), len(v0_sha))
    b = np.asarray([i for i, _ in keep])
    shas = [s for _, s in keep]
    sel = {k: np.asarray([ix[k][s] for s in shas]) for k in ix}

    sample = X.load_corpus_sample(out_root)
    sha_to_q = {r["sha"]: q for q, r in enumerate(sample["rows"])}
    qidx = np.asarray([sha_to_q[s] for s in shas])
    n_train, n_val = sample["n_train"], sample["n_val"]
    split = np.where(qidx < n_train, "train", np.where(qidx < n_train + n_val, "val", "test"))
    corpus = np.asarray([sample["rows"][q]["corpus"] for q in qidx])
    return {
        "C0": C0_by[layer][sel["c0"]],
        "V0": V0_by[layer][b],
        "Cplus": Cp_by[layer][sel["cp"]],
        "Vplus": Vp_by[layer][sel["vp"]],
        "Vplus_tf": Vt_by[layer][sel["vt"]],
        "sha": shas,
        "qidx": qidx,
        "split": split,
        "corpus": corpus,
    }


def _baseline_reads(cell: dict, pred_te: np.ndarray, te: np.ndarray, tr: np.ndarray) -> dict:
    """identity+learned-bias baseline + kNN retrieval (the standing mapping rule)."""
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    Xc, Y = cell["C0"], cell["V0"]
    out: dict = {}
    if Xc.shape[1] == Y.shape[1]:
        ib = identity_bias_predict(Xc[tr], Y[tr], Xc[te])
        out["identity_bias"] = {
            "heldout_r2": F._pooled_r2(ib, Y[te]),
            "mean_cos": F._mean_cos(ib, Y[te]),
            "knn_euclidean": knn_retrieval(ib, Y[te], ks=(1, 10), metric="euclidean"),
        }
    else:
        out["identity_bias"] = {"inapplicable": f"dim mismatch {Xc.shape[1]} vs {Y.shape[1]}"}
    out["fitted_map"] = {
        "knn_euclidean": knn_retrieval(pred_te, Y[te], ks=(1, 10), metric="euclidean"),
        "knn_cosine": knn_retrieval(pred_te, Y[te], ks=(1, 10), metric="cosine"),
    }
    return out


def fit_cell(cell: dict, dev, smoke: bool) -> dict:
    tr, val, te = F._split_idx(cell["split"])
    pred0, meta0, pay0 = F._fit_map(cell["C0"], cell["V0"], tr, val, te, dev)
    predp, metap, payp = F._fit_map(cell["Cplus"], cell["Vplus"], tr, val, te, dev)
    predt, metat, _payt = F._fit_map(cell["Cplus"], cell["Vplus_tf"], tr, val, te, dev)
    lam0 = meta0["selected_lambda"]
    return {
        "n_rows": int(len(cell["sha"])),
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
        "n_test": int(len(te)),
        "d": int(cell["C0"].shape[1]),
        "M0": {**F._map_reads(pred0, cell["V0"][te]), "selected_lambda": lam0},
        "Mplus": {
            **F._map_reads(predp, cell["Vplus"][te]),
            "selected_lambda": metap["selected_lambda"],
        },
        "Mplus_tf": {
            **F._map_reads(predt, cell["Vplus_tf"][te]),
            "selected_lambda": metat["selected_lambda"],
        },
        "map_change": F._map_change_block(cell, pay0, payp, lam0, dev, smoke),
        "baselines": _baseline_reads(cell, pred0, te, tr),
    }


# ── context movement (last-token) ────────────────────────────────────────────


def context_movement(cell: dict) -> dict:
    """Does the last-token context vector itself move under fine-tuning?"""
    C0, Cp = cell["C0"], cell["Cplus"]
    dC = Cp - C0
    n0 = np.linalg.norm(C0, axis=1)
    nd = np.linalg.norm(dC, axis=1)
    cos = (C0 * Cp).sum(axis=1) / (n0 * np.linalg.norm(Cp, axis=1) + 1e-12)
    return {
        "mean_norm_c0": float(n0.mean()),
        "mean_norm_delta_c": float(nd.mean()),
        "median_relative_move": float(np.median(nd / (n0 + 1e-12))),
        "mean_cos_c0_cplus": float(cos.mean()),
        "median_cos_c0_cplus": float(np.median(cos)),
        "n_rows": int(C0.shape[0]),
    }


# ── driver ───────────────────────────────────────────────────────────────────


def prestage_base_responses(out_root: Path, layers: list[int], arms: list[str]) -> None:
    """Stage the SHARED base response caches ONCE, serially, before any fanout.

    Every arm re-reads its base unit's response store, so N concurrent shards
    would each try to publish the same cache file. Resolving them serially in
    the parent removes the race entirely rather than relying on the writer-side
    tolerance in ``fetch_response`` (belt AND braces — the documented fan-out
    shared-staging discipline).
    """
    cache = out_root / "lt_answer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    bases = sorted({X.base_unit_for(a) for a in arms})
    for unit in bases:
        fetch_response(cache, "corpus_capture", unit, layers, persist=True)
    logger.info("[prestage] base response caches ready: %s", bases)


def run_fits(
    out_root: Path,
    results_dir: Path,
    layers: list[int],
    arms: list[str],
    positions: list[str],
    smoke: bool,
    shard: int,
    n_shards: int,
) -> None:
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = out_root / "lt_answer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    mine = [a for i, a in enumerate(arms) if i % n_shards == shard]
    logger.info(
        "[fits] shard %d/%d owns %d/%d arms on %s", shard, n_shards, len(mine), len(arms), dev
    )
    for k, arm_id in enumerate(mine):
        dest = results_dir / "cells" / f"{arm_id}.json"
        if dest.exists():
            logger.info("[fits] %s: present, skip", arm_id)
            continue
        rec: dict = {"arm_id": arm_id, "positions": {}, **_meta()}
        for position in positions:
            per_layer = {}
            for layer in layers:
                cell = build_cell(out_root, cache, arm_id, layer, position)
                res = fit_cell(cell, dev, smoke)
                res["context_movement"] = context_movement(cell)
                per_layer[str(layer)] = res
                logger.info(
                    "[fits] %s pos=%s L%d: base_r2=%.4f D=%.4f verdict=%s",
                    arm_id,
                    position,
                    layer,
                    res["M0"]["heldout_r2"],
                    res["map_change"]["D"],
                    res["map_change"]["verdict"],
                )
            rec["positions"][position] = per_layer
        _atomic_json(dest, rec)
        logger.info("[phase=lt_fits arm=%s %d/%d done]", arm_id, k + 1, len(mine))
    # NOT `[phase=done]` — reserved for a dispatcher's single terminal line
    # (see the capture driver's note; #545/#930).
    logger.info("[shard-complete] shard %d fitted %d arms", shard, len(mine))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--layers", default=",".join(str(x) for x in X.LAYERS))
    ap.add_argument("--arms", default="")
    ap.add_argument("--positions", default=",".join(LT.POSITIONS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument(
        "--prestage-only",
        action="store_true",
        help="stage the shared base response caches serially, then exit (run before fanout)",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401

        import issue779_ffc_n1m_fits as _n1m  # noqa: F401

        print("import-check ok")
        return 0

    assert args.out_root is not None, "--out-root is required outside --import-check"
    layers = [int(x) for x in args.layers.split(",")]
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]
    for p in positions:
        assert p in LT.POSITIONS, (p, LT.POSITIONS)
    arms = (
        [a.strip() for a in args.arms.split(",") if a.strip()]
        if args.arms
        else [a.arm_id for a in X.all_arms()]
    )
    if args.prestage_only:
        prestage_base_responses(args.out_root, layers, arms)
        sys.stdout.flush()
        sys.exit(0)
    run_fits(
        args.out_root,
        args.results_dir,
        layers,
        arms,
        positions,
        args.smoke,
        args.shard,
        args.n_shards,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
