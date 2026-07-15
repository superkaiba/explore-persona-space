"""Issue #1332 P2 — ridge map fits, layer freeze, function-space similarity (VM CPU).

Plan v3 §4.4-§4.5. Consumes the P1 capture store; NEVER loads the leakage DV
(the layer freeze is DV-independent by construction — plan §4.7 item 1).

Order of operations (each output checkpointed the moment it completes):
1. 3-cell slow-vs-fast PARITY GATE (``ridge_fit_predict`` canonical vs
   ``ridge_fit_predict_fast`` + the #1332 layer-batched wrapper) at production
   shape, max rel diff <= 1e-4; on failure the canonical solver is used
   (slower, same numbers — plan "allowed without asking").
2. Pre-registered 1-cell PILOT through the production entrypoint (the plan §9
   ``pilot-gated`` basis): measured per-call wall + ru_maxrss; projected
   serial wall > 4 h => RuntimeError (abort-and-batch-further, plan §7);
   measured RSS >= 16 GB => RuntimeError naming the cpu-mid reroute.
3. LAYER FREEZE: per-family split-half (A/B) own-map transfer R^2, batched
   over all 28 layers; L* = argmax of the family mean. Written to
   ``layer_freeze.json`` BEFORE any leakage join anywhere in the pipeline.
4. SIMILARITY at {L*} + the diagnostic grid: per (family, fold) held-out-by-
   query transfer (the #823 dedup shape: one fit per (i, layer, fold),
   rescored against every j), prediction agreement on the pooled probe set,
   map-mediated displacement, mean-target (degenerate prefix arm) transfer +
   S_excess, per-layer JSONs. Pair scoring is BATCHED (mid-run vectorize fix;
   serial oracle retained as ``_similarity_at_layer_serial_reference``,
   equivalence-gated via ``--verify-sim-layer``).
5. SPLIT-HALF similarity matrices S^(A)/S^(B) at L* (r_SS reliability +
   same-family ceiling) + descriptive weight-space cosines with the
   matched-half noise reference (#823 calibration).
6. ``--arm i545``: the same machinery over the #545 behavior/eval-column
   capture (own-question units; split-half transfer both ways).

USAGE
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \\
      uv run python scripts/issue1332_fits.py --full
    uv run python scripts/issue1332_fits.py --smoke   # smoke capture store, tiny params
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1332_common as C

logger = logging.getLogger("issue1332.fits")

PARITY_REL_TOL = 1e-4  # fast-twin docstring mandate (plan §11 item 2)
PILOT_ABORT_WALL_H = 4.0  # plan §7 sizing pilot abort threshold
PILOT_RSS_REROUTE_GB = 16.0  # plan §9 reroute rule


def _ru_maxrss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


# ── store access ──────────────────────────────────────────────────────────────


def store_dir(args) -> Path:
    return (
        C.data_root(args.smoke, args.out_root)
        / "store"
        / ("capture" if args.arm == "marker" else "capture545")
    )


def discover_families(sdir: Path) -> list[str]:
    """Families/units present in the capture store (the phase's cell list)."""
    fams = sorted(p.stem for p in sdir.glob("*.pt"))
    if not fams:
        raise FileNotFoundError(f"no capture shards under {sdir}")
    return fams


def ensure_shards(sdir: Path, fams: list[str], arm: str) -> None:
    """Local-first -> HF-fetch each family shard (git-clone lanes stage no data/).

    Fail-loud (r1 Major 1): a family is legitimately absent ONLY when its
    ``{fam}.dropped.json`` marker (written by the capture stage's 80% validity
    floor) resolves locally or on the Hub; every other miss — transient HF
    failure included (``retry_transient`` re-raises after its budget) —
    RAISES, so the panel can never silently truncate mid-staging.
    """
    from huggingface_hub.errors import EntryNotFoundError

    sub = "capture" if arm == "marker" else "capture545"
    missing: list[str] = []
    for fam in fams:
        local = sdir / f"{fam}.pt"
        if local.exists():
            continue
        try:
            C.hf_fetch(f"analysis_tensors/{sub}/{fam}.pt", local)
        except EntryNotFoundError:
            dropped_local = sdir / f"{fam}.dropped.json"
            if dropped_local.exists():
                continue
            try:
                C.hf_fetch(f"analysis_tensors/{sub}/{fam}.dropped.json", dropped_local)
            except EntryNotFoundError:
                missing.append(fam)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} families have NEITHER a capture shard NOR a dropped-marker "
            f"under analysis_tensors/{sub}/ on the Hub: {missing} — refusing to fit a "
            f"silently truncated panel"
        )


class ShardCache:
    """mmap'd per-family shard loads with layer slicing (bounded resident set)."""

    def __init__(self, sdir: Path):
        self.sdir = sdir
        self._cache: dict[str, dict] = {}

    def shard(self, fam: str) -> dict:
        if fam not in self._cache:
            import torch

            # self-produced #1332 shard (trusted); metadata carries non-primitives
            self._cache[fam] = torch.load(
                self.sdir / f"{fam}.pt", map_location="cpu", mmap=True, weights_only=False
            )
            if len(self._cache) > 4:  # keep the resident set bounded
                oldest = next(iter(self._cache))
                if oldest != fam:
                    del self._cache[oldest]
        return self._cache[fam]

    def arrays(self, fam: str, key: str, layers: list[int]):
        """(n_rows, len(layers), H) fp32 numpy slice for one summary key."""
        import numpy as np

        t = self.shard(fam)[key]  # (n, L, H) fp16 mmap
        return np.stack([t[:, li, :].float().numpy() for li in layers], axis=1).astype(np.float32)

    def bank_indices(self, fam: str) -> list[int]:
        sh = self.shard(fam)
        if "bank_indices" in sh:
            return [int(i) for i in sh["bank_indices"]]
        return list(range(int(sh["cx_last"].shape[0])))

    def n_layers(self, fam: str) -> int:
        return int(self.shard(fam)["n_layers"])


def resolve_n_bank(cache: ShardCache, fams: list[str]) -> int:
    """Bank size from shard meta (exact), never inferred when the field exists.

    r1 Minor: ``max(bank_indices)+1`` silently under-counts when the LAST bank
    query is invalid in every family, shifting the shared fold/split
    permutations. Shards written post-r1 carry ``n_bank``; the inferred
    fallback remains only for legacy (pre-r1 smoke) shards.
    """
    exact = {int(cache.shard(f)["n_bank"]) for f in fams if "n_bank" in cache.shard(f)}
    if exact:
        if len(exact) != 1:
            raise RuntimeError(f"inconsistent n_bank across shards: {sorted(exact)}")
        return exact.pop()
    return max(max(cache.bank_indices(f)) for f in fams) + 1


# ── solvers ───────────────────────────────────────────────────────────────────


def fit_predict_batched(X_tr, Y_tr, X_ev, *, solver: str, return_weights: bool = False):
    """Production fit entrypoint: layer-batched fast twin, canonical fallback.

    X_tr/Y_tr/X_ev are (L, n, d) stacks. ``solver`` in {"fast", "canonical"}.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict,
        ridge_fit_predict_fast_layer_batched,
    )

    if solver == "fast":
        return ridge_fit_predict_fast_layer_batched(X_tr, Y_tr, X_ev, return_weights=return_weights)
    preds = np.stack(
        [ridge_fit_predict(X_tr[li], Y_tr[li], X_ev[li]) for li in range(X_tr.shape[0])], axis=0
    )
    if return_weights:
        raise NotImplementedError("weights only on the fast path (descriptive read)")
    return preds


def r2(y_true, y_pred) -> float:
    """R^2 with ss_tot centered on y_true's mean (#823 convention)."""
    import numpy as np

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mu = yt.mean(axis=0)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - mu) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def parity_gate(cache: ShardCache, fams: list[str], layers: list[int], folds) -> dict:
    """3-cell canonical-vs-fast parity at production shape (<=1e-4 rel)."""
    import numpy as np

    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict,
        ridge_fit_predict_fast,
        ridge_fit_predict_fast_layer_batched,
    )

    cells = []
    for k in range(min(3, len(fams))):
        fam = fams[k % len(fams)]
        layer = layers[k % len(layers)]
        fold = folds[k % len(folds)]
        cells.append((fam, layer, fold))
    worst = 0.0
    for fam, layer, fold in cells:
        bank_idx = cache.bank_indices(fam)
        val_set = set(fold)
        tr_rows = [i for i, b in enumerate(bank_idx) if b not in val_set]
        ev_rows = [i for i, b in enumerate(bank_idx) if b in val_set]
        X = cache.arrays(fam, "cx_last", [layer])[:, 0, :]
        Y = cache.arrays(fam, "v_mean", [layer])[:, 0, :]
        ref = ridge_fit_predict(X[tr_rows], Y[tr_rows], X[ev_rows])
        fast = ridge_fit_predict_fast(X[tr_rows], Y[tr_rows], X[ev_rows])
        batched = ridge_fit_predict_fast_layer_batched(
            X[tr_rows][None], Y[tr_rows][None], X[ev_rows][None]
        )[0]
        scale = float(np.abs(ref).max()) + 1e-12
        rel_fast = float(np.abs(fast - ref).max()) / scale
        rel_batched = float(np.abs(batched - ref).max()) / scale
        worst = max(worst, rel_fast, rel_batched)
        logger.info(
            "[parity] fam=%s layer=%d rel_fast=%.2e rel_batched=%.2e",
            fam,
            layer,
            rel_fast,
            rel_batched,
        )
    passed = worst <= PARITY_REL_TOL
    return {
        "max_rel_diff": worst,
        "tolerance": PARITY_REL_TOL,
        "pass": bool(passed),
        "solver": "fast" if passed else "canonical",
    }


def run_pilot(cache: ShardCache, fams: list[str], n_layers: int, folds, solver: str) -> dict:
    """Pre-registered 1-cell pilot through the production entrypoint (plan §7/§9)."""
    fam = fams[0]
    bank_idx = cache.bank_indices(fam)
    val_set = set(folds[0])
    tr_rows = [i for i, b in enumerate(bank_idx) if b not in val_set]
    ev_rows = [i for i, b in enumerate(bank_idx) if b in val_set]
    layers = list(range(n_layers))
    X = cache.arrays(fam, "cx_last", layers).transpose(1, 0, 2)  # (L, n, H)
    Y = cache.arrays(fam, "v_mean", layers).transpose(1, 0, 2)
    t0 = time.time()
    fit_predict_batched(X[:1, tr_rows], Y[:1, tr_rows], X[:1, ev_rows], solver=solver)
    per_call_1 = time.time() - t0
    t0 = time.time()
    fit_predict_batched(X[:, tr_rows], Y[:, tr_rows], X[:, ev_rows], solver=solver)
    per_call_all = time.time() - t0
    n_fams = len(fams)
    n_single_calls = n_fams * len(folds) * (len(C.LAYER_GRID) + 1)
    n_batched_calls = n_fams * 2  # split-half layer-freeze calls (all layers)
    projected_h = (n_single_calls * per_call_1 + n_batched_calls * per_call_all) / 3600.0
    rss_gb = _ru_maxrss_gb()
    pilot = {
        "per_call_single_layer_s": per_call_1,
        "per_call_all_layers_s": per_call_all,
        "n_single_calls": n_single_calls,
        "n_batched_calls": n_batched_calls,
        "projected_wall_h": projected_h,
        "ru_maxrss_gb": rss_gb,
        "n_train": len(tr_rows),
        "hidden_dim": int(X.shape[2]),
        "solver": solver,
    }
    logger.info("[pilot] %s", pilot)
    if projected_h > PILOT_ABORT_WALL_H:
        raise RuntimeError(
            f"fit pilot projects {projected_h:.2f} h > {PILOT_ABORT_WALL_H} h serial — "
            "abort-and-batch-further (plan §7 sizing pilot)"
        )
    if rss_gb >= PILOT_RSS_REROUTE_GB:
        raise RuntimeError(
            f"fit pilot ru_maxrss {rss_gb:.1f} GB >= {PILOT_RSS_REROUTE_GB} GB — reroute to "
            "cpu-mid (--min-ram-gb 32) per plan §9"
        )
    return pilot


# ── marker-arm passes ─────────────────────────────────────────────────────────


def layer_freeze(
    cache: ShardCache, fams: list[str], out_dir: Path, solver: str, smoke: bool
) -> dict:
    """DV-independent split-half own-map transfer R^2 per (family, layer); L* freeze."""
    import numpy as np

    n_layers = cache.n_layers(fams[0])
    layers = list(range(n_layers))
    n_bank = resolve_n_bank(cache, fams)
    half_a, half_b = C.split_half(n_bank)
    set_a, set_b = set(half_a), set(half_b)
    curve = np.full((len(fams), n_layers), np.nan)
    for fi, fam in enumerate(fams):
        bank_idx = cache.bank_indices(fam)
        rows_a = [i for i, b in enumerate(bank_idx) if b in set_a]
        rows_b = [i for i, b in enumerate(bank_idx) if b in set_b]
        X = cache.arrays(fam, "cx_last", layers).transpose(1, 0, 2)
        Y = cache.arrays(fam, "v_mean", layers).transpose(1, 0, 2)
        preds_ab = fit_predict_batched(X[:, rows_a], Y[:, rows_a], X[:, rows_b], solver=solver)
        preds_ba = fit_predict_batched(X[:, rows_b], Y[:, rows_b], X[:, rows_a], solver=solver)
        for li in layers:
            r_ab = r2(Y[li, rows_b], preds_ab[li])
            r_ba = r2(Y[li, rows_a], preds_ba[li])
            curve[fi, li] = 0.5 * (r_ab + r_ba)
        logger.info(
            "[freeze] %s best layer %d (%.4f)",
            fam,
            int(np.nanargmax(curve[fi])),
            float(np.nanmax(curve[fi])),
        )
    mean_curve = curve.mean(axis=0)
    l_star = int(np.argmax(mean_curve))
    payload = {
        "l_star": l_star,
        "mean_split_half_r2_per_layer": mean_curve.tolist(),
        "per_family": {f: curve[fi].tolist() for fi, f in enumerate(fams)},
        "families": fams,
        "rule": "argmax over layers of family-mean same-family split-half transfer R^2 "
        "(DV-independent; computed and committed BEFORE any leakage join — plan §4.7.1)",
        "split_half_seed": C.SPLIT_HALF_SEED,
        "reproducibility_metadata": C.reproducibility_metadata({"smoke": smoke}),
    }
    C.write_json_atomic(out_dir / "layer_freeze.json", payload)
    return payload


def _score_similarity_fold_batched(preds, Yval, val_slices, fams, train_means):
    """Batched pair scoring for ONE fold (the #1332 mid-run vectorize fix).

    Replaces the serial per-(i, j) ``r2()`` double loop (676 pairs x ~4
    full-pooled-array fp64 passes x 5 folds ~= 13.5k serial memory-bound
    passes; 1737 s measured at L5) with pair-axis-batched fp64 reductions:

    - S_trans / S_mean_target: per-family-j residual einsums over the stacked
      preds tensor ((nf, n_j, H) temps); ``ss_tot_j`` computed ONCE per j
      (it never depends on i).
    - S_agree / d_map: pairwise squared distance via the Gram trick
      ``||p_i - p_j||^2 = ||p_i||^2 + ||p_j||^2 - 2<p_i, p_j>`` (one fp64
      GEMM over the flattened probe axis; diagonal pinned to 0 — the serial
      ss_res is exactly 0 there — and clamped >= 0 against Gram rounding),
      with per-j centering terms computed once.

    Returns the four (nf, nf) per-fold contributions
    ``(s_trans, s_mean_target, s_agree, d_map)`` BEFORE the caller's
    ``/ n_folds`` — matching ``_similarity_at_layer_serial_reference`` within
    float tolerance (gated <= PARITY_REL_TOL by ``--verify-sim-layer`` and
    ``tests/test_issue1332_similarity_vectorized.py``).
    """
    import numpy as np

    nf, n_probe, hdim = preds.shape
    preds64 = preds.astype(np.float64)
    # fp32 train-mean VALUES cast to fp64, exactly as the serial r2() casts them
    tm64 = np.asarray(np.stack(train_means), dtype=np.float64)

    s_trans = np.zeros((nf, nf))
    s_mean_target = np.zeros((nf, nf))
    for j, fam_j in enumerate(fams):
        sl = val_slices[fam_j]
        y_j = np.asarray(Yval[j], dtype=np.float64)
        mu_j = y_j.mean(axis=0)
        denom_j = float(((y_j - mu_j) ** 2).sum()) + 1e-12  # ss_tot + eps (r2 convention)
        resid = y_j[None, :, :] - preds64[:, sl, :]
        s_trans[:, j] = 1.0 - np.einsum("ijk,ijk->i", resid, resid) / denom_j
        mt_resid = y_j[None, :, :] - tm64[:, None, :]
        s_mean_target[:, j] = 1.0 - np.einsum("ijk,ijk->i", mt_resid, mt_resid) / denom_j

    flat = preds64.reshape(nf, -1)
    sq = np.einsum("ik,ik->i", flat, flat)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (flat @ flat.T)
    np.fill_diagonal(d2, 0.0)
    np.maximum(d2, 0.0, out=d2)
    mu_p = preds64.mean(axis=1)  # (nf, H) pooled-probe column means, per family
    cent = np.empty(nf)
    for i in range(nf):
        c = preds64[i] - mu_p[i]
        cent[i] = float(np.einsum("jk,jk->", c, c))
    inv_cent = 1.0 / (cent + 1e-12)
    # a_ij = 0.5*(r2(p_j, p_i) + r2(p_i, p_j)) = 1 - 0.5*D_ij*(1/ct_j + 1/ct_i)
    s_agree = 1.0 - 0.5 * d2 * (inv_cent[None, :] + inv_cent[:, None])
    denom_dmap = cent / (n_probe * hdim) + 1e-12  # serial den: MEAN-based + eps
    d_map = (d2 / (n_probe * hdim)) / denom_dmap[None, :]
    return s_trans, s_mean_target, s_agree, d_map


def similarity_at_layer(cache: ShardCache, fams: list[str], layer: int, folds, solver: str) -> dict:
    """All function-space similarity metrics at ONE layer (plan §4.5; batched scoring).

    Fits keep the #823 dedup shape — ONE fit per (family, fold), rescored on
    the pooled probe set (0.048 s/fit measured, ``fit_pilot.json``); the pair
    scoring is batched over the (i, j) axis (``_score_similarity_fold_batched``
    — the mid-run vectorize fix; serial oracle retained as
    ``_similarity_at_layer_serial_reference``).
    """
    import numpy as np

    nf = len(fams)
    s_trans = np.zeros((nf, nf))
    s_agree = np.zeros((nf, nf))
    d_map = np.zeros((nf, nf))
    s_mean_target = np.zeros((nf, nf))
    n_folds = len(folds)
    for fold in folds:
        val_set = set(fold)
        Xtr, Ytr, Yval, val_slices = [], [], [], {}
        probe_parts = []
        pos = 0
        for fam in fams:
            bank_idx = cache.bank_indices(fam)
            tr_rows = [i for i, b in enumerate(bank_idx) if b not in val_set]
            ev_rows = [i for i, b in enumerate(bank_idx) if b in val_set]
            X = cache.arrays(fam, "cx_last", [layer])[:, 0, :]
            Y = cache.arrays(fam, "v_mean", [layer])[:, 0, :]
            Xtr.append(X[tr_rows])
            Ytr.append(Y[tr_rows])
            probe_parts.append(X[ev_rows])
            Yval.append(Y[ev_rows])
            val_slices[fam] = slice(pos, pos + len(ev_rows))
            pos += len(ev_rows)
        X_probe = np.concatenate(probe_parts, axis=0)  # pooled probe inputs (identical per pair)
        preds = np.zeros((nf, X_probe.shape[0], X_probe.shape[1]), dtype=np.float32)
        train_means = []
        for i, _fam in enumerate(fams):
            p = fit_predict_batched(Xtr[i][None], Ytr[i][None], X_probe[None], solver=solver)[0]
            preds[i] = p.astype(np.float32)
            train_means.append(Ytr[i].mean(axis=0))
        st, smt, sa, dm = _score_similarity_fold_batched(preds, Yval, val_slices, fams, train_means)
        s_trans += st / n_folds
        s_mean_target += smt / n_folds
        s_agree += sa / n_folds
        d_map += dm / n_folds
    s_sym = 0.5 * (s_trans + s_trans.T)
    return {
        "families": fams,
        "layer": layer,
        "S_trans": s_trans.tolist(),
        "S_sym": s_sym.tolist(),
        "S_asym": (s_trans - s_trans.T).tolist(),
        "S_agree": s_agree.tolist(),
        "S_dmap_one_minus": (1.0 - d_map).tolist(),
        "S_mean_target": s_mean_target.tolist(),
        "S_excess": (s_trans - s_mean_target).tolist(),
        "n_folds": n_folds,
    }


def _similarity_at_layer_serial_reference(
    cache: ShardCache, fams: list[str], layer: int, folds, solver: str
) -> dict:
    """SERIAL ORACLE — retained ONLY for equivalence gates; never call in production.

    Verbatim pre-vectorization body (the seeded serial oracle of the #1332
    mid-run vectorize fix; Supersede-contract containment per
    ``.claude/rules/vectorize-many-cell-fits.md``). Production path:
    ``similarity_at_layer`` (batched), pinned to this reference by
    ``tests/test_issue1332_similarity_vectorized.py`` and to the persisted
    serial L5 JSON by ``--verify-sim-layer``.
    """
    import numpy as np

    nf = len(fams)
    s_trans = np.zeros((nf, nf))
    s_agree = np.zeros((nf, nf))
    d_map = np.zeros((nf, nf))
    s_mean_target = np.zeros((nf, nf))
    n_folds = len(folds)
    for fold in folds:
        val_set = set(fold)
        Xtr, Ytr, Xval, Yval, val_slices = [], [], [], [], {}
        pos = 0
        for fam in fams:
            bank_idx = cache.bank_indices(fam)
            tr_rows = [i for i, b in enumerate(bank_idx) if b not in val_set]
            ev_rows = [i for i, b in enumerate(bank_idx) if b in val_set]
            X = cache.arrays(fam, "cx_last", [layer])[:, 0, :]
            Y = cache.arrays(fam, "v_mean", [layer])[:, 0, :]
            Xtr.append(X[tr_rows])
            Ytr.append(Y[tr_rows])
            Xval.append(X[ev_rows])
            Yval.append(Y[ev_rows])
            val_slices[fam] = slice(pos, pos + len(ev_rows))
            pos += len(ev_rows)
        X_probe = np.concatenate(Xval, axis=0)  # pooled probe inputs (identical per pair)
        preds = np.zeros((nf, X_probe.shape[0], X_probe.shape[1]), dtype=np.float32)
        train_means = []
        for i, _fam in enumerate(fams):
            p = fit_predict_batched(Xtr[i][None], Ytr[i][None], X_probe[None], solver=solver)[0]
            preds[i] = p.astype(np.float32)
            train_means.append(Ytr[i].mean(axis=0))
        for i in range(nf):
            for j, fam_j in enumerate(fams):
                sl = val_slices[fam_j]
                y_j = Yval[j]
                s_trans[i, j] += r2(y_j, preds[i, sl]) / n_folds
                mt_pred = np.broadcast_to(train_means[i], y_j.shape)
                s_mean_target[i, j] += r2(y_j, mt_pred) / n_folds
                # prediction agreement (Y-free), symmetrized over centering choices
                a_ij = 0.5 * (r2(preds[j], preds[i]) + r2(preds[i], preds[j]))
                s_agree[i, j] += a_ij / n_folds
                num = float(((preds[i] - preds[j]) ** 2).mean())
                den = float(((preds[j] - preds[j].mean(axis=0)) ** 2).mean()) + 1e-12
                d_map[i, j] += (num / den) / n_folds
    s_sym = 0.5 * (s_trans + s_trans.T)
    return {
        "families": fams,
        "layer": layer,
        "S_trans": s_trans.tolist(),
        "S_sym": s_sym.tolist(),
        "S_asym": (s_trans - s_trans.T).tolist(),
        "S_agree": s_agree.tolist(),
        "S_dmap_one_minus": (1.0 - d_map).tolist(),
        "S_mean_target": s_mean_target.tolist(),
        "S_excess": (s_trans - s_mean_target).tolist(),
        "n_folds": n_folds,
    }


def verify_similarity_vectorized(
    cache: ShardCache, fams: list[str], layer: int, folds, solver: str, oracle_path: Path
) -> dict:
    """Batched-vs-serial-oracle equivalence gate at production shape (report-only).

    Re-runs ONE layer through the BATCHED production ``similarity_at_layer``
    and compares every similarity matrix against the serial run's persisted
    JSON (the seeded serial oracle) under the ``parity_gate`` matrix-scale
    convention: ``max|batched - serial| / (max|serial| + 1e-12)`` per matrix,
    each <= PARITY_REL_TOL. Writes NOTHING under eval_results/; raises on FAIL.
    """
    import numpy as np

    oracle = json.loads(Path(oracle_path).read_text())
    if oracle["families"] != fams or oracle["n_folds"] != len(folds) or oracle["layer"] != layer:
        raise RuntimeError(
            f"oracle regime mismatch: {oracle_path} carries "
            f"({len(oracle['families'])} fams, {oracle['n_folds']} folds, layer "
            f"{oracle['layer']}) vs live ({len(fams)} fams, {len(folds)} folds, layer {layer})"
        )
    t0 = time.time()
    out = similarity_at_layer(cache, fams, layer, folds, solver)
    wall = time.time() - t0
    keys = (
        "S_trans",
        "S_sym",
        "S_asym",
        "S_agree",
        "S_dmap_one_minus",
        "S_mean_target",
        "S_excess",
    )
    per_matrix = {}
    for key in keys:
        a = np.asarray(out[key], dtype=np.float64)
        b = np.asarray(oracle[key], dtype=np.float64)
        per_matrix[key] = float(np.abs(a - b).max()) / (float(np.abs(b).max()) + 1e-12)
    worst = max(per_matrix.values())
    report = {
        "layer": layer,
        "wall_s_batched": wall,
        "wall_s_serial_oracle": oracle.get("wall_s"),
        "per_matrix_max_rel_diff": per_matrix,
        "max_rel_diff": worst,
        "tolerance": PARITY_REL_TOL,
        "pass": bool(worst <= PARITY_REL_TOL),
        "solver": solver,
    }
    logger.info("[verify-sim] %s", json.dumps(report))
    if not report["pass"]:
        raise RuntimeError(f"vectorized similarity parity FAIL: {json.dumps(report)}")
    return report


def split_half_similarity(cache: ShardCache, fams: list[str], layer: int, solver: str) -> dict:
    """S^(A)/S^(B) half-map similarity matrices + r_SS + weight-space reads at L*."""
    import numpy as np

    n_bank = resolve_n_bank(cache, fams)
    half_a, half_b = C.split_half(n_bank)
    set_a, set_b = set(half_a), set(half_b)
    nf = len(fams)
    s_half = {"A": np.zeros((nf, nf)), "B": np.zeros((nf, nf))}
    weights = {"A": [], "B": []}
    rows_by_half = {}
    data = {}
    for fam in fams:
        bank_idx = cache.bank_indices(fam)
        rows_by_half[fam] = {
            "A": [i for i, b in enumerate(bank_idx) if b in set_a],
            "B": [i for i, b in enumerate(bank_idx) if b in set_b],
        }
        data[fam] = (
            cache.arrays(fam, "cx_last", [layer])[:, 0, :],
            cache.arrays(fam, "v_mean", [layer])[:, 0, :],
        )
    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict_fast_layer_batched,
    )

    for half, other in (("A", "B"), ("B", "A")):
        # pooled other-half probe inputs (ONE fit per (half, i), rescored per j
        # — the #823 dedup shape)
        probe_slices: dict[str, slice] = {}
        probe_parts = []
        pos = 0
        for fam_j in fams:
            Xj, _Yj = data[fam_j]
            ev = rows_by_half[fam_j][other]
            probe_parts.append(Xj[ev])
            probe_slices[fam_j] = slice(pos, pos + len(ev))
            pos += len(ev)
        X_probe = np.concatenate(probe_parts, axis=0)
        for i, fam in enumerate(fams):
            X, Y = data[fam]
            tr = rows_by_half[fam][half]
            if solver == "fast":
                p_all, w = ridge_fit_predict_fast_layer_batched(
                    X[tr][None], Y[tr][None], X_probe[None], return_weights=True
                )
                p_all = p_all[0]
                # fp32 retention: 52 retained fp64 (3584, 3584) W matrices peak
                # ~10.7 GB transient RSS (r1 Minor) — the descriptive cosine
                # reads are insensitive to the cast; halves the retention.
                weights[half].append(np.asarray(w[0], dtype=np.float32))
            else:
                p_all = fit_predict_batched(X[tr][None], Y[tr][None], X_probe[None], solver=solver)[
                    0
                ]
            for j, fam_j in enumerate(fams):
                _Xj, Yj = data[fam_j]
                ev = rows_by_half[fam_j][other]
                s_half[half][i, j] = r2(Yj[ev], p_all[probe_slices[fam_j]])
    s_a_sym = 0.5 * (s_half["A"] + s_half["A"].T)
    s_b_sym = 0.5 * (s_half["B"] + s_half["B"].T)
    off = ~np.eye(nf, dtype=bool)
    from scipy.stats import spearmanr

    r_ss = float(spearmanr(s_a_sym[off], s_b_sym[off]).statistic)
    ceiling_per_family = {
        f: float(0.5 * (s_half["A"][i, i] + s_half["B"][i, i])) for i, f in enumerate(fams)
    }
    out = {
        "families": fams,
        "layer": layer,
        "S_half_A": s_half["A"].tolist(),
        "S_half_B": s_half["B"].tolist(),
        "r_SS": r_ss,
        "ceiling_split_half_per_family": ceiling_per_family,
        "note": ("half-map n < the ~250-row comfort point — ceiling is conservative (plan §4.5)"),
    }
    if solver == "fast" and weights["A"]:
        wa = np.stack([w.reshape(-1) for w in weights["A"]])
        wb = np.stack([w.reshape(-1) for w in weights["B"]])

        def _cos(u, v):
            return float(u @ v / ((np.linalg.norm(u) + 1e-12) * (np.linalg.norm(v) + 1e-12)))

        n_show = len(fams)
        w_cos = [[_cos(wa[i], wa[j]) for j in range(n_show)] for i in range(n_show)]
        matched_half = [_cos(wa[i], wb[i]) for i in range(n_show)]
        out["weight_cosine_half_A"] = w_cos
        out["weight_cosine_matched_half_noise"] = matched_half
        out["weight_note"] = (
            "DESCRIPTIVE ONLY (settled decision): standardized-input-space dual-reconstructed "
            "W cosines beside the matched-half noise reference (#823 calibration)"
        )
    return out


# ── i545 arm ──────────────────────────────────────────────────────────────────


def i545_similarity(cache: ShardCache, units: list[str], layer: int, solver: str) -> dict:
    """Split-half-based cross-unit transfer for the #545 OOD arm (own-question units)."""
    import numpy as np

    halves = {}
    data = {}
    for u in units:
        sh = cache.shard(u)
        n = int(sh["cx_last"].shape[0])
        rng = np.random.default_rng(C.SPLIT_HALF_SEED)
        perm = rng.permutation(n)
        halves[u] = (sorted(perm[: n // 2].tolist()), sorted(perm[n // 2 :].tolist()))
        data[u] = (
            cache.arrays(u, "cx_last", [layer])[:, 0, :],
            cache.arrays(u, "v_mean", [layer])[:, 0, :],
        )
    nu = len(units)
    s_trans = np.zeros((nu, nu))
    split_half_own = {}
    for i, u in enumerate(units):
        X, Y = data[u]
        a, b = halves[u]
        pred_ab = fit_predict_batched(X[a][None], Y[a][None], X[b][None], solver=solver)[0]
        pred_ba = fit_predict_batched(X[b][None], Y[b][None], X[a][None], solver=solver)[0]
        split_half_own[u] = 0.5 * (r2(Y[b], pred_ab) + r2(Y[a], pred_ba))
        for j, v in enumerate(units):
            if v == u:
                s_trans[i, j] = split_half_own[u]
                continue
            Xv, Yv = data[v]
            _, bv = halves[v]
            p = fit_predict_batched(X[a][None], Y[a][None], Xv[bv][None], solver=solver)[0]
            s_trans[i, j] = r2(Yv[bv], p)
    s_sym = 0.5 * (s_trans + s_trans.T)
    return {
        "units": units,
        "layer": layer,
        "S_trans": s_trans.tolist(),
        "S_sym": s_sym.tolist(),
        "split_half_own": split_half_own,
        "off_policy_targets_caveat": ("#545 corpora Sonnet/handwritten text targets (plan §4.3.4)"),
    }


# ── driver ────────────────────────────────────────────────────────────────────


def main() -> int:
    """P2 driver: parity gate -> pilot -> layer freeze -> similarity -> split-half.

    ``--verify-sim-layer L`` short-circuits to the batched-vs-serial-oracle
    similarity equivalence gate at production shape (report-only).
    """
    ap = argparse.ArgumentParser(description="Issue #1332 P2 fits (VM CPU)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument(
        "--verify-sim-layer",
        type=int,
        default=None,
        metavar="L",
        help="batched-vs-serial-oracle similarity equivalence gate at production "
        "shape against the persisted S_transfer_L{L}.json (report-only; writes "
        "nothing under eval_results/)",
    )
    ap.add_argument("--arm", default="marker", choices=["marker", "i545"])
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--skip-hf-fetch", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import torch

    torch.set_num_threads(args.n_threads)

    sdir = store_dir(args)
    res_dir = C.results_dir(args.smoke, args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    if args.arm == "marker" and not args.smoke and not args.skip_hf_fetch:
        _sources, targets = C.family_labels()
        ensure_shards(sdir, targets, args.arm)  # fail-loud staging (r1 Major 1)
    fams = discover_families(sdir)
    fams = [f for f in fams if not f.endswith(".dropped") and not f.endswith(".skipped")]
    if args.arm == "marker" and not args.smoke:
        # panel-completeness assert (r1 Major 1): every registered family is
        # either captured or explicitly dropped — never silently absent.
        n_dropped = len(list(sdir.glob("*.dropped.json")))
        n_expected = len(C.family_labels()[1])
        if len(fams) + n_dropped != n_expected:
            raise RuntimeError(
                f"panel incomplete: {len(fams)} capture shards + {n_dropped} dropped "
                f"markers != {n_expected} registered families — refusing to fit a "
                f"silently truncated panel"
            )
    cache = ShardCache(sdir)
    n_layers = cache.n_layers(fams[0])
    n_bank = resolve_n_bank(cache, fams)
    folds = C.query_folds(n_bank)

    if args.verify_sim_layer is not None:
        # equivalence gate only — no phase breadcrumbs, no eval_results/ writes
        gate_path = res_dir / "parity_gate.json"
        solver = json.loads(gate_path.read_text())["solver"]
        oracle_path = res_dir / "similarity" / f"S_transfer_L{args.verify_sim_layer}.json"
        verify_similarity_vectorized(cache, fams, args.verify_sim_layer, folds, solver, oracle_path)
        return 0

    C.phase("p2_parity")
    gate = parity_gate(cache, fams, [min(n_layers - 1, 14)], folds)
    C.write_json_atomic(res_dir / "parity_gate.json", gate)
    solver = gate["solver"]

    C.phase("p2_pilot")
    pilot = run_pilot(cache, fams, n_layers, folds, solver)
    C.write_json_atomic(res_dir / "fit_pilot.json", pilot)

    if args.arm == "i545":
        C.phase("p2_i545")
        freeze_path = C.results_dir(args.smoke, args.results_dir) / "layer_freeze.json"
        if freeze_path.exists():
            l_star = json.loads(freeze_path.read_text())["l_star"]
        elif args.full:
            raise FileNotFoundError(
                f"{freeze_path} missing — run the marker arm first; refusing the "
                f"L{C.WHITENED_GATE_LAYER} fallback in --full (r1 Minor: out-of-order "
                f"invocation must fail loud, not silently score the wrong layer)"
            )
        else:
            l_star = min(n_layers - 1, C.WHITENED_GATE_LAYER)
        for layer in sorted({l_star, min(n_layers - 1, C.WHITENED_GATE_LAYER)}):
            out = i545_similarity(cache, fams, layer, solver)
            out["reproducibility_metadata"] = C.reproducibility_metadata({"smoke": args.smoke})
            C.write_json_atomic(res_dir / "similarity545" / f"S_transfer_L{layer}.json", out)
        C.phase("done_fits_i545")
        return 0

    C.phase("p2_layer_freeze")
    freeze = layer_freeze(cache, fams, res_dir, solver, args.smoke)
    l_star = freeze["l_star"]

    C.phase("p2_similarity")
    sim_layers = sorted({l_star, *[li for li in C.LAYER_GRID if li < n_layers]})
    for layer in sim_layers:
        out_path = res_dir / "similarity" / f"S_transfer_L{layer}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if existing.get("families") == fams and existing.get("n_folds") == len(folds):
                logger.info("[sim] layer %d exists (resume skip)", layer)
                continue
        t0 = time.time()
        out = similarity_at_layer(cache, fams, layer, folds, solver)
        out["l_star"] = l_star
        out["wall_s"] = time.time() - t0
        out["reproducibility_metadata"] = C.reproducibility_metadata({"smoke": args.smoke})
        C.write_json_atomic(out_path, out)
        logger.info("[sim] layer %d done in %.1fs", layer, out["wall_s"])

    C.phase("p2_split_half")
    sh = split_half_similarity(cache, fams, l_star, solver)
    sh["reproducibility_metadata"] = C.reproducibility_metadata({"smoke": args.smoke})
    C.write_json_atomic(res_dir / "splithalf" / f"splithalf_L{l_star}.json", sh)

    logger.info("[fits] complete; ru_maxrss %.2f GB", _ru_maxrss_gb())
    C.phase("done_fits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
