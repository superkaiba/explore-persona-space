#!/usr/bin/env python
"""Task #2220 — read-direction vs mean-difference steering (phase-dispatch driver).

Tests whether the #1739 behavior-prediction map's READ direction (the analytic
input-space gradient of the fitted whiten->standardize->linear-ridge scorer)
STEERS as well as the #779 mean-difference persona vector, at matched injection
norm, on the same held-out questions.

Phases (``--phase``; PHASES registry):
  materialize_directions  fit whitening (U pool) + ridge (DV-labeled acts) -> d_read
                          (context + prefix arms) and build directions 3-6
                          (r_B, raw mean-diff, shuffled-label, random). POD-ONLY
                          (154 GB labeling tars + 8.5 GB #1092 U-pool store).
  norm_probe              per-layer rho_l = median ||last-context-token residual||
                          over the held-out eval queries (dose scale). GPU.
  localize                6 dirs x 2 pos x 5 layers x {c=0.5,1,2,4} + alpha=0 ref;
                          Q1=10, draws=3, seed 42 -> DeltaHook+generate_batch. GPU.
  decisive                6 dirs + alpha=0 at selected operating points x 2 pos;
                          Q2=20, draws=5, seeds {42,43}. GPU.
  margin                  teacher-forced fixed +/- completion-pool margin DV. GPU.
  judge_reduce            judge the persisted completions -> per-cell Delta-rate +
                          selection-symmetric null band + verdict lattice. CPU + Batch API.

Design + reuse contract: plan v4 (tasks/running/2220/plans/plan.md) sections
4.1 / 4.2 / 4.3 / 4.4 / 6 / 9 / 10. The d_read fold (materialize_directions) is
gated end-to-end by the A9 finite-difference check
(tests/test_issue2220_dread_gradient.py) on synthetic fits.

Reuse (verified against live code):
  - issue_1739.fits.fit_whitening / apply_whitening / ridge_fit_predict_primal_layer_batched
    (the last returns a 2-tuple (preds, w_out); w_out is in STANDARDIZED-z space,
    sigma_z RECOMPUTED by the caller = z_train.std(axis=0, ddof=0)+1e-9,
    matching fits.py L543 xtr.std(dim=1, unbiased=False)+1e-9).
  - issue_1739.store_io.load_summaries / fit_pool_mask (whitening U pool) and
    load_rb_bank (the #779 r_B bank).
  - issue1739_natpv.stream_members + issue1739_map963k_slice.tar_url/head_size/
    ParallelRangeReader (slice-by-slice tar streaming of the DV-labeled acts) and
    issue1739_natpv.load_labels / load_row_index (per-context DV + row->context join).
  - issue1415.steering.DeltaHook / generate_batch / coherence_check / condition_passes
    (steering rig — HF generate(), NOT vLLM; stated deviation, plan 4.3).
  - issue_1739.judging.judge_items_graded / load_trait_rubric / rollout_item_id
    (Sonnet 0-100 trait rubric, Batch API, max_tokens=2048, drop-never-coerce).
  - analysis.extraction.extract_layer_activations (norm_probe; same blocks[L] module
    DeltaHook edits -- capture site == edit site, plan A1).
  - orchestrate.hub upload helpers (raw completions + direction bank).

CONTENT HYGIENE: evil/hallucination/sycophancy query text and steered completions
are harmful-adjacent. Logs and markers carry ids, counts, scores, shapes, hashes
-- NEVER item/completion text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

# load_dotenv BEFORE any torch/transformers import (thread-cap + credential
# setdefaults are frozen at torch import; orchestrate.env, never bare dotenv).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("issue2220")


def _ensure_repo_root_on_syspath() -> None:
    """Put the repo root on sys.path so `import scripts.<mod>` resolves (#823).

    The streaming reuse (`scripts.issue1739_natpv`) itself imports
    `scripts.issue1739_map963k_slice`, so the `scripts` PACKAGE must be
    importable — which requires the repo root (not `scripts/`) on sys.path.
    In script mode sys.path[0] is the script's own dir (`scripts/`), so the
    bare-name path would resolve THIS module's siblings but not the `scripts`
    package. Idempotent; asserts a repo sentinel so a wrong parent index
    fails loud instead of silently shadowing.
    """
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "pyproject.toml").exists(), f"repo-root sentinel missing at {repo_root}"
    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# pins (plan v4 §4/§6/§9/§10; #1739 constants reused verbatim)
# ---------------------------------------------------------------------------
from explore_persona_space.experiments.issue_1739.constants import (  # noqa: E402
    HF_DATA_REPO,
    HIDDEN_DIM,
    MODEL_NAME,
    RIDGE_LAMBDAS,
    STORE_REVISION,
    U_STORE_CELL,
)

BEHAVIORS = ("evil", "hallucination", "sycophancy")
# Swept layers (plan §4/§9). Gate 1 may trim 5->3 layers pre-launch.
LAYERS = (10, 14, 18, 20, 24)
# Dose multipliers c: alpha = c * rho_l (plan §4.3). c=0 is the no-injection ref.
DOSES_NONZERO = (0.5, 1.0, 2.0, 4.0)
POSITIONS = ("context", "answer")  # DeltaHook all_positions False / True (plan §4.2)
# Direction slugs (plan §4.1 rows 1-6 + the alpha=0 reference; §5 config slugs).
DIRECTIONS = ("mapread_ctx", "mapread_prefix", "rb", "rawmeandiff", "shuffled", "random")
SIGNAL_DIRECTIONS = ("mapread_ctx", "mapread_prefix", "rb", "rawmeandiff")
NULL_DIRECTIONS = ("shuffled", "random")

# Direction -> whitening/summary arm it is derived from (plan §6 pooling-convention).
DIRECTION_SUMMARY_KIND = {
    "mapread_ctx": "context_end",
    "mapread_prefix": "prefix_end",
    "rawmeandiff": "context_end",
    "shuffled": "context_end",
}
N_RANDOM_SEEDS = 3  # direction 6 = mean over 3 matched-norm random unit vectors (plan §4.1)

JUDGE_MAX_TOKENS = 2048  # multi-field trait+coherence rubric (llm-judging rule 23; NOT #1739's 400)
GEN_MAX_NEW_TOKENS = 2048  # free-generation default; parent used 1024 (plan §6 deviation)

# Question counts / draws (plan §4.4).
Q1_LOCALIZE = 10
DRAWS_LOCALIZE = 3
SEED_LOCALIZE = 42
Q2_DECISIVE = 20
DRAWS_DECISIVE = 5
SEEDS_DECISIVE = (42, 43)

# HF destinations (plan §10).
HF_PREFIX = "issue2220_readwrite"  # data repo prefix
RB_PREFIX = "issue779_monitoring/r_b/"  # #779 r_B bank prefix (store_io.load_rb_bank default)
RB_REVISION = "037fcbb"  # #779 r_B pin (plan §10)
# DV-labeled activation store (SEPARATE from the #1092 U-pool store; plan §4.1 step 1).
LABELING_TAR_REVISION = "5bd378408b7ee2f9c166eb2a059ab96478a28de7"

MODEL_REPO_HIDDEN = HIDDEN_DIM  # 3584

# ---------------------------------------------------------------------------
# Pure fold + scorer helpers (A9-gated; tests/test_issue2220_dread_gradient.py)
# ---------------------------------------------------------------------------


def recompute_sigma_z(z_train_layer):
    """Per-feature population std of the whitened TRAIN features, +1e-9.

    Matches the ridge helper's internal ``xtr.std(dim=1, unbiased=False)+1e-9``
    (fits.py L543). ``z_train_layer`` is (n_train, d). Returns (d,) float64.
    """
    import numpy as np

    z = np.asarray(z_train_layer, dtype=np.float64)
    assert z.ndim == 2, z.shape
    return z.std(axis=0, ddof=0) + 1e-9


def fold_d_read(wh_w_layer, w_ridge_layer, sigma_z_layer):
    """d_read[l] = normalize( wh.w[l] @ (w_ridge[l] / sigma_z[l]) ).

    The analytic input-space gradient of the fitted linear scorer
    s(v) = w_ridge . ((wh.w @ (v - wh.mu)) - mu_z)/sigma_z + b, folding the
    whitening (wh.w symmetric Sigma_gamma^{-1/2}) and the caller-recomputed
    per-feature standardization. mu_z and b are gradient-irrelevant constants.

    Args:
        wh_w_layer:    (d, d) symmetric Sigma_gamma^{-1/2} for layer l (wh.w[l]).
        w_ridge_layer: (d,) ridge weight in standardized-z feature space.
        sigma_z_layer: (d,) recompute_sigma_z output.
    Returns:
        (d,) unit-normalized float64 input-space direction.
    """
    import numpy as np

    wh_w = np.asarray(wh_w_layer, dtype=np.float64)
    w = np.asarray(w_ridge_layer, dtype=np.float64).reshape(-1)
    sig = np.asarray(sigma_z_layer, dtype=np.float64).reshape(-1)
    assert wh_w.ndim == 2 and wh_w.shape[0] == wh_w.shape[1], wh_w.shape
    assert w.shape == sig.shape == (wh_w.shape[0],), (w.shape, sig.shape, wh_w.shape)
    grad = wh_w @ (w / sig)  # wh.w symmetric => wh.w^T == wh.w
    nrm = float(np.linalg.norm(grad))
    if not (nrm > 0.0) or not np.isfinite(nrm):
        raise ValueError(f"fold_d_read: degenerate gradient norm {nrm!r}")
    return grad / nrm


def scorer_predict(v, wh_mu_layer, wh_w_layer, mu_z_layer, sigma_z_layer, w_ridge_layer, b):
    """Fitted scorer s(v) for an input-space activation v (single layer).

    s(v) = w_ridge . z(v) + b,  z(v) = ((wh.w @ (v - wh.mu)) - mu_z) / sigma_z.
    Reproduces the ridge helper's un-centered prediction
    ``preds = (xev - xmu)/xsd @ w + ymu`` (fits.py L544-564) with x := z.
    Used only by the A9 finite-difference check (the fold is what production
    consumes); mu_z/b make the scorer internally consistent but do not affect
    the gradient.
    """
    import numpy as np

    v = np.asarray(v, dtype=np.float64).reshape(-1)
    z = np.asarray(wh_w_layer, dtype=np.float64) @ (v - np.asarray(wh_mu_layer, dtype=np.float64))
    z = (z - np.asarray(mu_z_layer, dtype=np.float64)) / np.asarray(sigma_z_layer, dtype=np.float64)
    return float(np.asarray(w_ridge_layer, dtype=np.float64).reshape(-1) @ z + float(b))


def raw_mean_diff_direction(x_labeled_layer, dv, *, top_frac=0.25):
    """Direction 4: mean(top-scored context_end) - mean(bottom-scored), UNWHITENED.

    Isolates the covariance-accounting whitening adds vs a raw high/low contrast
    (plan §4.1 row 4; 2507.21509 raw-vs-diff appendix). ``x_labeled_layer`` is
    (n, d) raw (unwhitened) activations; ``dv`` is (n,) graded 0-100 DV. Returns
    a unit-normalized (d,) direction. top_frac selects the extremes by DV rank.
    """
    import numpy as np

    x = np.asarray(x_labeled_layer, dtype=np.float64)
    d = np.asarray(dv, dtype=np.float64).reshape(-1)
    assert x.ndim == 2 and x.shape[0] == d.shape[0], (x.shape, d.shape)
    k = max(1, int(round(x.shape[0] * top_frac)))
    order = np.argsort(d)  # ascending
    lo_idx, hi_idx = order[:k], order[-k:]
    diff = x[hi_idx].mean(axis=0) - x[lo_idx].mean(axis=0)
    nrm = float(np.linalg.norm(diff))
    if not (nrm > 0.0) or not np.isfinite(nrm):
        raise ValueError(f"raw_mean_diff_direction: degenerate norm {nrm!r}")
    return diff / nrm


def shuffled_fold(wh_w_layer, z_labeled_layer, dv, *, seed):
    """Direction 5: direction-1 construction with the DV labels SHUFFLED.

    The #1739 arm20 shuffled control -- a direction with no genuine behavior
    signal, but built through the identical whiten->standardize->ridge->fold
    pipeline (so it captures fit-pipeline artefacts / spurious geometry).
    ``z_labeled_layer`` is (n, d) WHITENED features for the layer; ``dv`` is
    (n,) graded DV; ``wh_w_layer`` is (d, d).  Returns a unit (d,) direction.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits

    z = np.asarray(z_labeled_layer, dtype=np.float64)
    d = np.asarray(dv, dtype=np.float64).reshape(-1).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(d)  # permute labels; features untouched
    _preds, w_out = fits.ridge_fit_predict_primal_layer_batched(
        z[None], d[None, :, None], z[None], lambdas=RIDGE_LAMBDAS, return_weights=True
    )
    sigma_z = recompute_sigma_z(z)
    return fold_d_read(wh_w_layer, w_out[0, :, 0], sigma_z)


def random_direction(d, *, seed, n_avg=N_RANDOM_SEEDS):
    """Direction 6: matched-norm random unit vector, mean over ``n_avg`` seeds.

    Plan §4.1 row 6.  Each seed draws a Gaussian, the mean over seeds is
    re-normalized to unit norm (so it stays a matched-injection-norm control).
    """
    import numpy as np

    acc = np.zeros(d, dtype=np.float64)
    for s in range(n_avg):
        rng = np.random.default_rng(seed * 1000 + s)
        v = rng.standard_normal(d)
        acc += v / float(np.linalg.norm(v))
    nrm = float(np.linalg.norm(acc))
    return acc / nrm


# ---------------------------------------------------------------------------
# shared: paths, cells, sentinel, breadcrumbs
# ---------------------------------------------------------------------------


def _sha8(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]


def _breadcrumb(phase: str, **kw) -> None:
    kv = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[phase={phase}] {kv}", flush=True)


def _progress(phase: str, k: int, n: int, key: str, t0: float) -> None:
    print(f"[{phase}] unit {k}/{n} {key} elapsed={time.time() - t0:.1f}s", flush=True)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _git_provenance() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance())


def _run_metadata(extra: dict | None = None) -> dict:
    md = {
        "experiment": "issue2220_readwrite",
        "base_model": MODEL_NAME,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "store_revision": STORE_REVISION,
        "rb_revision": RB_REVISION,
        "labeling_tar_revision": LABELING_TAR_REVISION,
    }
    md.update(_git_provenance())
    if extra:
        md.update(extra)
    return md


def _write_sentinel(out_root: Path, phase: str, status: str, extra: dict | None = None) -> Path:
    """Pod-observed sentinel (/workspace/logs/issue-2220-<phase>.json) the VM
    poller drains. Pod-side code NEVER shells to task.py."""
    logs = Path(os.environ.get("EPM_SENTINEL_DIR", "/workspace/logs"))
    payload = {"issue": 2220, "phase": phase, "status": status, "out_root": str(out_root)}
    if extra:
        payload.update(extra)
    try:
        logs.mkdir(parents=True, exist_ok=True)
        p = logs / f"issue-2220-{phase}.json"
        _write_json_atomic(p, payload)
        return p
    except OSError as exc:  # sentinel dir absent off-pod (VM smoke) -> log, never crash
        logger.info("[sentinel] %s not writable (%s); skipping", logs, type(exc).__name__)
        return Path("/dev/null")


def _out_root(args) -> Path:
    return Path(args.out_root)


def _contexts_for_questions(questions: list[str]) -> list[dict]:
    """steering.generate_batch context shape: {"system": None, "user": q}."""
    return [{"system": None, "user": q} for q in questions]


# ---------------------------------------------------------------------------
# eval query bank (persona-vectors disjoint 20-question EVAL set; plan §6)
# ---------------------------------------------------------------------------


def _eval_questions(behavior: str) -> list[str]:
    """The persona-vectors disjoint EVAL question set for ``behavior``.

    Loaded via the #779 asset chain (issue_1739.generation.load_e1_assets). The
    persona-vectors recipe splits 40 questions into a 20-question extraction set
    (which produced the direction fits) + a DISJOINT 20-question eval set. We
    steer + judge on the EVAL set only, so the operating-point read is never on
    the questions the direction was built from. CONTENT HYGIENE: question text is
    passed to the model/judge but never logged.

    NB: the exact asset key for the disjoint eval set is confirmed pod-side
    against the #779 assets at run time (see the eval-set concern); this helper
    prefers an explicit eval-set key and falls back to the disjoint tail slice
    of a 40-question bank.
    """
    from explore_persona_space.experiments.issue_1739.generation import load_e1_assets

    assets = load_e1_assets(behavior)
    for key in ("eval_questions", "evaluation_questions"):
        qs = assets.get(key)
        if qs:
            return list(qs)
    xq = list(assets["extraction_questions"])
    if len(xq) >= 40:
        return xq[20:40]  # disjoint eval tail of a 40-question bank
    raise RuntimeError(
        f"[{behavior}] no disjoint eval-question set in #779 assets "
        f"(extraction_questions has {len(xq)}); confirm eval-set key pod-side"
    )


# ---------------------------------------------------------------------------
# model loading (GPU phases)
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None


def _load_model_and_tokenizer():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[model] loading %s (bf16)", MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # generate_batch requires left-padding
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    _MODEL, _TOKENIZER = model, tok
    return model, tok


# ---------------------------------------------------------------------------
# phase: materialize_directions  (POD-ONLY; 154 GB tars + 8.5 GB U-pool)
# ---------------------------------------------------------------------------


def _stream_labeled_context_acts(behavior: str, layers, stage: Path, args) -> dict:
    """Stream the DV-labeled context/prefix activations for ``behavior``.

    Returns {(kind, layer): (n_ctx, d) float64} for kind in {context_end,
    prefix_end}, aligned to the per-context DV order from load_labels. Streams
    the labeling tar slice-by-slice (issue1739_natpv.stream_members), retaining
    only the swept layers' columns -- never materializing the full 154 GB.

    NB (POD-VERIFIED, NOT A9-gated): the labeling tar's `context_end` /
    `prefix_end` summary shards are per-CONTEXT (arm4's unit is the context:
    z_ctx is (Ly, n_ctx, d), dv is (n_ctx,)); a store that turns out per-ROW
    is reduced first-row-per-context via the row_index join. Both shapes are
    handled and the loader FAILS LOUD on any third shape — the exact grain is
    confirmed against #1739 arm4 at run time (raise-concern: readwrite-join).
    """
    import numpy as np

    _ensure_repo_root_on_syspath()
    import scripts.issue1739_natpv as natpv  # noqa: E402  (repo-root just added)

    labels = natpv.load_labels(behavior, stage)
    pos = labels["pos"]  # context_id -> context index
    ctx_order = labels["ctx_order"]
    n_ctx = len(ctx_order)
    kinds = ("context_end", "prefix_end")
    want = natpv._summary_re(kinds)
    # accumulate shard arrays keyed by (kind, layer, shard); one sequential tar pass
    shard_store: dict[tuple[str, int, int], np.ndarray] = {}
    for name, arr in natpv.stream_members(
        behavior, LABELING_TAR_REVISION, workers=args.workers, window_mib=args.window_mib, want=want
    ):
        if not name.endswith(".npy"):
            continue
        kind, layer, shard = natpv._parse_summary_name(name)
        if layer not in layers or kind not in kinds:
            continue
        shard_store[(kind, layer, shard)] = np.asarray(arr, dtype=np.float64)

    # row->context join is needed ONLY for a per-row store; probe grain lazily.
    ridx: dict | None = None

    def _row_index() -> dict:
        nonlocal ridx
        if ridx is None:
            natpv.phase_rowindex(args, behavior, stage)  # stage row_index shards from the tar
            ridx = natpv.load_row_index(stage, behavior)
        return ridx

    out: dict[tuple[str, int], np.ndarray] = {}
    for kind in kinds:
        for layer in layers:
            shards = sorted(s for (k, ly, s) in shard_store if k == kind and ly == layer)
            if not shards:
                raise RuntimeError(f"[{behavior}] no {kind} L{layer:02d} shards in labeling tar")
            rows = np.concatenate([shard_store[(kind, layer, s)] for s in shards], axis=0)
            if rows.shape[0] == n_ctx:
                # per-CONTEXT store (arm4 grain): already aligned to ctx_order.
                out[(kind, layer)] = rows
                continue
            r = _row_index()
            if rows.shape[0] != r["n_rows"]:
                raise RuntimeError(
                    f"[{behavior}] {kind} L{layer:02d}: shard rows {rows.shape[0]} match neither "
                    f"n_ctx {n_ctx} nor n_rows {r['n_rows']} (confirm store grain vs #1739 arm4)"
                )
            # per-ROW store: reduce first-row-per-context (context_end/prefix_end
            # are context-level -> identical across a context's rollouts).
            per_ctx = np.full((n_ctx, rows.shape[1]), np.nan, dtype=np.float64)
            seen = np.zeros(n_ctx, dtype=bool)
            for i, cid in enumerate(r["context_id"]):
                j = pos.get(cid)
                if j is not None and not seen[j]:
                    per_ctx[j] = rows[i]
                    seen[j] = True
            if not seen.all():
                raise RuntimeError(
                    f"[{behavior}] {kind} L{layer:02d}: {int((~seen).sum())} contexts unfilled"
                )
            out[(kind, layer)] = per_ctx
    del shard_store
    return {"acts": out, "dv": np.asarray(labels["dv"], dtype=np.float64), "split": labels["split"]}


def _load_u_pool(layers, args) -> dict:
    """Whitening U pool from the #1092 summary store (cell_inst_own, fp16)."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import store_io

    local_dir = Path(args.u_store_dir)
    summaries, meta = store_io.load_summaries(
        local_dir,
        kinds=("context_end", "prefix_end"),
        layers=tuple(layers),
        cell=U_STORE_CELL,
    )
    mask = store_io.fit_pool_mask(meta)
    out: dict[str, np.ndarray] = {}
    for kind in ("context_end", "prefix_end"):
        stack = np.stack(
            [np.asarray(summaries[(kind, ly)], dtype=np.float64)[mask] for ly in layers], axis=0
        )  # (Ly, n_U, d)
        out[kind] = stack
    return out


def phase_materialize_directions(args) -> None:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits, store_io

    out_root = _out_root(args)
    dir_out = out_root / "directions"
    dir_out.mkdir(parents=True, exist_ok=True)
    stage = out_root / "labeling_stage"
    stage.mkdir(parents=True, exist_ok=True)
    layers = list(args.layers)
    behaviors = list(args.behaviors)
    _breadcrumb("materialize_directions", behaviors=len(behaviors), layers=len(layers))

    # r_B bank (all 28 layers x n_traits x d), pinned #779.
    rb_bank, rb_trait_names = store_io.load_rb_bank(
        revision=RB_REVISION, n_layers=28, hidden_dim=HIDDEN_DIM
    )
    trait_idx = {t: i for i, t in enumerate(rb_trait_names)}

    u_pool = _load_u_pool(layers, args)
    manifest_entries: list[dict] = []
    t0 = time.time()
    n_cells = len(behaviors)
    for bi, behavior in enumerate(behaviors, 1):
        done = dir_out / f"{behavior}_manifest.done"
        if done.exists() and not args.force:
            logger.info("[materialize] %s already done; skipping", behavior)
            continue
        labeled = _stream_labeled_context_acts(behavior, layers, stage, args)
        acts, dv, split = labeled["acts"], labeled["dv"], labeled["split"]
        train_mask = np.array([s == "train" for s in split])
        assert train_mask.any(), f"[{behavior}] no train-split contexts"
        # per-arm (context_end -> mapread_ctx; prefix_end -> mapread_prefix) whitening + ridge fold
        for arm_kind, arm_slug in (
            ("context_end", "mapread_ctx"),
            ("prefix_end", "mapread_prefix"),
        ):
            x_u = u_pool[arm_kind]  # (Ly, n_U, d)
            wh = fits.fit_whitening(x_u)
            x_lab = np.stack([acts[(arm_kind, ly)] for ly in layers], axis=0)  # (Ly, n_ctx, d)
            z = fits.apply_whitening(x_lab, wh)  # (Ly, n_ctx, d)
            z_tr = z[:, train_mask, :]  # (Ly, n_tr, d)
            dv_tr = dv[train_mask]
            # y broadcast to (Ly, n_tr, 1): same per-context DV target every layer.
            y_tr = np.repeat(dv_tr[None, :, None], z_tr.shape[0], axis=0)
            _preds, w_out = fits.ridge_fit_predict_primal_layer_batched(
                z_tr, y_tr, z_tr, lambdas=RIDGE_LAMBDAS, return_weights=True
            )
            for li, layer in enumerate(layers):
                sigma_z = recompute_sigma_z(z_tr[li])
                d_read = fold_d_read(wh.w[li], w_out[li, :, 0], sigma_z)
                _save_direction(dir_out, behavior, arm_slug, layer, d_read, manifest_entries)
        # raw mean-diff + shuffled + random (context_end arm)
        wh_ctx = fits.fit_whitening(u_pool["context_end"])
        x_lab_ctx = np.stack([acts[("context_end", ly)] for ly in layers], axis=0)
        z_ctx = fits.apply_whitening(x_lab_ctx, wh_ctx)
        z_ctx_tr = z_ctx[:, train_mask, :]
        dv_tr = dv[train_mask]
        for li, layer in enumerate(layers):
            raw = raw_mean_diff_direction(x_lab_ctx[li][train_mask], dv_tr)
            _save_direction(dir_out, behavior, "rawmeandiff", layer, raw, manifest_entries)
            shuf = shuffled_fold(wh_ctx.w[li], z_ctx_tr[li], dv_tr, seed=SEED_LOCALIZE)
            _save_direction(dir_out, behavior, "shuffled", layer, shuf, manifest_entries)
            rnd = random_direction(HIDDEN_DIM, seed=SEED_LOCALIZE + layer)
            _save_direction(dir_out, behavior, "random", layer, rnd, manifest_entries)
            # r_B: pick the behavior's trait row, per layer.
            ti = trait_idx.get(behavior)
            if ti is None:
                raise RuntimeError(f"[{behavior}] absent from r_B bank traits {rb_trait_names}")
            rb_vec = np.asarray(rb_bank[layer, ti], dtype=np.float64)
            rb_unit = rb_vec / float(np.linalg.norm(rb_vec))
            _save_direction(dir_out, behavior, "rb", layer, rb_unit, manifest_entries)
        done.write_text(str(time.time()))
        _progress("materialize_directions", bi, n_cells, behavior, t0)

    manifest = _run_metadata({"directions": manifest_entries, "layers": layers})
    _write_json_atomic(dir_out / "directions_manifest.json", manifest)
    _upload_directions(dir_out)
    _write_sentinel(out_root, "materialize_directions", "done", {"n_dirs": len(manifest_entries)})
    _breadcrumb("materialize_directions", status="done", n_dirs=len(manifest_entries))


def _save_direction(
    dir_out: Path, behavior: str, slug: str, layer: int, vec, manifest: list
) -> None:
    import numpy as np
    import torch

    v = torch.as_tensor(np.asarray(vec, dtype=np.float32))
    path = dir_out / f"{behavior}_{slug}_L{layer}.pt"
    torch.save({"direction": v, "behavior": behavior, "slug": slug, "layer": layer}, path)
    manifest.append(
        {
            "behavior": behavior,
            "slug": slug,
            "layer": layer,
            "path": path.name,
            "norm": float(np.linalg.norm(np.asarray(vec, dtype=np.float64))),
            "sha8": _sha8(np.asarray(vec, dtype=np.float32).round(6).tolist()),
        }
    )


def _upload_directions(dir_out: Path) -> None:
    """Persist the direction bank + manifest to the HF data repo (fail-loud, retried)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    files = sorted(dir_out.glob("*.pt")) + [dir_out / "directions_manifest.json"]
    files = [f for f in files if f.exists()]
    if not files:
        logger.warning("[upload] no direction files under %s", dir_out)
        return
    hub.retry_transient(
        lambda: api.upload_folder(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(dir_out),
            path_in_repo=f"{HF_PREFIX}/directions",
            allow_patterns=["*.pt", "*.json"],
        ),
        what="upload directions bank",
    )
    logger.info(
        "[upload] %d direction files -> %s/%s/directions", len(files), HF_DATA_REPO, HF_PREFIX
    )


# ---------------------------------------------------------------------------
# phase: norm_probe  (per-layer rho_l = median ||last-context-token residual||)
# ---------------------------------------------------------------------------


def phase_norm_probe(args) -> None:
    import numpy as np
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations
    from explore_persona_space.experiments.issue1415 import steering

    out_root = _out_root(args)
    layers = list(args.layers)
    behaviors = list(args.behaviors)
    _breadcrumb("norm_probe", behaviors=len(behaviors), layers=len(layers))
    model, tok = _load_model_and_tokenizer()

    result: dict[str, dict[str, float]] = {}
    t0 = time.time()
    for bi, behavior in enumerate(behaviors, 1):
        questions = _eval_questions(behavior)
        contexts = _contexts_for_questions(questions)
        norms = {ly: [] for ly in layers}
        for ctx in contexts:
            ids = steering.context_token_ids(tok, ctx)  # rendered context token ids
            input_ids = torch.tensor([ids], device=model.device)
            attn = torch.ones_like(input_ids)
            acts = extract_layer_activations(
                model, input_ids, layers, attention_mask=attn, detach_to_cpu=True
            )
            for ly in layers:
                # last real token (right-aligned single row, no padding) = -1
                vec = np.asarray(acts[ly][0, -1], dtype=np.float64)
                norms[ly].append(float(np.linalg.norm(vec)))
        result[behavior] = {f"L{ly}": float(np.median(norms[ly])) for ly in layers}
        _progress("norm_probe", bi, len(behaviors), behavior, t0)

    payload = _run_metadata({"rho_median_last_context_token": result, "layers": layers})
    _write_json_atomic(out_root / "norm_probe" / "rho.json", payload)
    _write_sentinel(out_root, "norm_probe", "done")
    _breadcrumb("norm_probe", status="done")


# ---------------------------------------------------------------------------
# steering generation (localize + decisive share this)
# ---------------------------------------------------------------------------


def _load_direction(dir_out: Path, behavior: str, slug: str, layer: int):
    import torch

    path = dir_out / f"{behavior}_{slug}_L{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(f"direction not materialized: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)["direction"]


def _load_rho(out_root: Path) -> dict:
    p = out_root / "norm_probe" / "rho.json"
    if not p.exists():
        raise FileNotFoundError(f"norm_probe not run: {p} (run --phase norm_probe first)")
    return json.loads(p.read_text())["rho_median_last_context_token"]


def _steer_cell(model, tok, direction, layer, alpha, position, contexts, *, n_draws, seed_base):
    """One steering cell -> per-context list of completion strings (draws)."""
    import torch

    from explore_persona_space.experiments.issue1415 import steering

    delta = direction.to(dtype=torch.bfloat16)
    all_positions = position == "answer"
    with steering.DeltaHook(
        model, layer=layer, delta=delta, alpha=float(alpha), all_positions=all_positions
    ) as hook:
        results = steering.generate_batch(
            model,
            tok,
            contexts,
            n=n_draws,
            hook=hook,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=1.0,
            seed_base=seed_base,
        )
    return results  # results[b][i] -> new-token text


def _cap_hit_fraction(results, tok) -> float:
    """Fraction of draws that hit the max_new_tokens cap (proxy: token length ==
    GEN_MAX_NEW_TOKENS). CLAUDE.md generation-stage rule."""
    total = hit = 0
    for row in results:
        for text in row:
            total += 1
            if len(tok.encode(text, add_special_tokens=False)) >= GEN_MAX_NEW_TOKENS:
                hit += 1
    return (hit / total) if total else 0.0


def _run_steer_grid(
    args, phase: str, cells: list[dict], contexts_by_behavior, *, n_draws: int, seeds: list[int]
) -> None:
    """Shared per-cell steering loop with per-cell checkpointing + resume."""
    from explore_persona_space.experiments.issue1415 import steering

    out_root = _out_root(args)
    dir_out = out_root / "directions"
    comp_root = out_root / phase / "raw_completions"
    comp_root.mkdir(parents=True, exist_ok=True)
    model, tok = _load_model_and_tokenizer()
    rho = _load_rho(out_root)
    _breadcrumb(phase, cells=len(cells), seeds=len(seeds))
    t0 = time.time()
    n = len(cells)
    for ci, cell in enumerate(cells, 1):
        behavior = cell["behavior"]
        cell_id = "__".join(
            f"{k}{cell[k]}" for k in ("behavior", "direction", "position", "layer", "c")
        )
        cell_id = cell_id.replace(".", "p")
        out_path = comp_root / f"{cell_id}.json"
        if out_path.exists() and not args.force:
            _progress(phase, ci, n, cell_id + " (cached)", t0)
            continue
        contexts = contexts_by_behavior[behavior]
        layer = cell["layer"]
        if cell["direction"] == "alpha0" or cell["c"] == 0.0:
            alpha = 0.0
            direction = None
        else:
            direction = _load_direction(dir_out, behavior, cell["direction"], layer)
            rho_l = rho[behavior][f"L{layer}"]
            alpha = cell["c"] * rho_l
        rows = {"cell_id": cell_id, "cell": cell, "seeds": {}}
        for seed in seeds:
            if alpha == 0.0:
                import torch

                # no-injection reference: a no-op hook keeps the identical
                # generate() path (assert-installed contract) at alpha=0.
                zero_delta = torch.zeros(model.config.hidden_size, dtype=torch.bfloat16)
                with steering.DeltaHook(
                    model,
                    layer=layer,
                    delta=zero_delta,
                    alpha=0.0,
                    all_positions=(cell["position"] == "answer"),
                ) as hook:
                    res = steering.generate_batch(
                        model,
                        tok,
                        contexts,
                        n=n_draws,
                        hook=hook,
                        max_new_tokens=GEN_MAX_NEW_TOKENS,
                        temperature=1.0,
                        seed_base=seed,
                    )
            else:
                res = _steer_cell(
                    model,
                    tok,
                    direction,
                    layer,
                    alpha,
                    cell["position"],
                    contexts,
                    n_draws=n_draws,
                    seed_base=seed,
                )
            coh = [steering.coherence_check(row) for row in res]
            rows["seeds"][str(seed)] = {
                "completions": res,
                "coherent_flags": coh,
                "condition_passes": [steering.condition_passes(c) for c in coh],
            }
        rows["alpha"] = float(alpha)
        rows["cap_hit_fraction"] = _cap_hit_fraction(
            [r for s in rows["seeds"].values() for r in s["completions"]], tok
        )
        _write_json_atomic(out_path, rows)
        _progress(phase, ci, n, cell_id, t0)

    _write_sentinel(out_root, phase, "done", {"cells": len(cells)})
    _breadcrumb(phase, status="done", cells=len(cells))
    _upload_raw_completions(out_root, phase)


def _upload_raw_completions(out_root: Path, phase: str) -> None:
    """Persist per-cell completion JSONs to the HF data repo before teardown.

    One bulk upload_folder commit (never a per-file loop); retried.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    comp_root = out_root / phase / "raw_completions"
    files = sorted(comp_root.glob("*.json"))
    if not files:
        logger.warning("[upload] no completions under %s", comp_root)
        return
    api = HfApi()
    hub.retry_transient(
        lambda: api.upload_folder(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(comp_root),
            path_in_repo=f"{HF_PREFIX}/raw_completions/{phase}",
            allow_patterns=["*.json"],
        ),
        what=f"upload {phase} raw completions",
    )
    logger.info(
        "[upload] %d %s completion files -> %s/%s/raw_completions/%s",
        len(files),
        phase,
        HF_DATA_REPO,
        HF_PREFIX,
        phase,
    )


def phase_localize(args) -> None:
    layers = list(args.layers)
    behaviors = list(args.behaviors)
    contexts_by_behavior = {
        b: _contexts_for_questions(_eval_questions(b)[: args.q1]) for b in behaviors
    }
    cells: list[dict] = []
    for behavior in behaviors:
        for direction in DIRECTIONS:
            for position in POSITIONS:
                for layer in layers:
                    for c in DOSES_NONZERO:
                        cells.append(
                            {
                                "behavior": behavior,
                                "direction": direction,
                                "position": position,
                                "layer": layer,
                                "c": c,
                            }
                        )
        cells.append(
            {
                "behavior": behavior,
                "direction": "alpha0",
                "position": "context",
                "layer": layers[0],
                "c": 0.0,
            }
        )
    _run_steer_grid(
        args,
        "localize",
        cells,
        contexts_by_behavior,
        n_draws=args.draws_localize,
        seeds=[SEED_LOCALIZE],
    )


def phase_decisive(args) -> None:
    out_root = _out_root(args)
    behaviors = list(args.behaviors)
    op = _load_operating_points(out_root)
    contexts_by_behavior = {
        b: _contexts_for_questions(_eval_questions(b)[: args.q2]) for b in behaviors
    }
    cells: list[dict] = []
    for behavior in behaviors:
        for direction in DIRECTIONS:
            for position in POSITIONS:
                sel = op.get(behavior, {}).get(f"{direction}__{position}")
                if sel is None:
                    # no coherent operating point -> undefined G-margin (plan §6 lattice)
                    continue
                cells.append(
                    {
                        "behavior": behavior,
                        "direction": direction,
                        "position": position,
                        "layer": sel["layer"],
                        "c": sel["c"],
                    }
                )
        cells.append(
            {
                "behavior": behavior,
                "direction": "alpha0",
                "position": "context",
                "layer": list(args.layers)[0],
                "c": 0.0,
            }
        )
    _run_steer_grid(
        args,
        "decisive",
        cells,
        contexts_by_behavior,
        n_draws=args.draws_decisive,
        seeds=list(SEEDS_DECISIVE),
    )


def _load_operating_points(out_root: Path) -> dict:
    """Selected (layer, c) per (direction, position, behavior) from judge_reduce.

    Read from localize's reduced surface if present; else empty (decisive then
    runs only alpha0 -- a degenerate smoke shape). The full operating-point
    selection is produced by phase_judge_reduce (localize mode).
    """
    p = out_root / "localize" / "operating_points.json"
    if p.exists():
        return json.loads(p.read_text())
    logger.warning("[decisive] no operating_points.json; run judge_reduce (localize) first")
    return {}


# ---------------------------------------------------------------------------
# phase: margin  (teacher-forced fixed +/- completion-pool margin DV)
# ---------------------------------------------------------------------------


def phase_margin(args) -> None:
    out_root = _out_root(args)
    behaviors = list(args.behaviors)
    op = _load_operating_points(out_root)
    dir_out = out_root / "directions"
    model, tok = _load_model_and_tokenizer()
    _breadcrumb("margin", behaviors=len(behaviors))
    result: dict[str, dict] = {}
    t0 = time.time()
    for bi, behavior in enumerate(behaviors, 1):
        pools = _load_answer_pools(out_root, behavior)  # {"pos": [...], "neg": [...]}
        contexts = _contexts_for_questions(_eval_questions(behavior)[: args.q2])
        cell_margins: dict[str, float] = {}
        for direction in DIRECTIONS:
            for position in POSITIONS:
                sel = op.get(behavior, {}).get(f"{direction}__{position}")
                if sel is None:
                    continue
                d_vec = _load_direction(dir_out, behavior, direction, sel["layer"])
                alpha = sel["c"] * _load_rho(out_root)[behavior][f"L{sel['layer']}"]
                m = _teacher_forced_margin(
                    model, tok, contexts, pools, d_vec, sel["layer"], alpha, position
                )
                cell_margins[f"{direction}__{position}"] = float(m)
        result[behavior] = cell_margins
        _progress("margin", bi, len(behaviors), behavior, t0)
    payload = _run_metadata({"tf_margin": result})
    _write_json_atomic(out_root / "margin" / "margin.json", payload)
    _write_sentinel(out_root, "margin", "done")
    _breadcrumb("margin", status="done")


def _load_answer_pools(out_root: Path, behavior: str) -> dict:
    p = out_root / "margin" / "pools" / f"{behavior}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"fixed +/- answer pool missing: {p} (built pod-side from judge-filtered completions)"
        )
    return json.loads(p.read_text())


def _teacher_forced_margin(model, tok, contexts, pools, direction, layer, alpha, position) -> float:
    """mean LN-logP(fixed pos pool | C) - mean LN-logP(fixed neg pool | C) under steering."""
    import numpy as np

    def _ln_logp(answers, ctx):
        vals = []
        for ans in answers:
            vals.append(_ln_logp_one(model, tok, ctx, ans, direction, layer, alpha, position))
        return float(np.mean(vals)) if vals else float("nan")

    margins = []
    for ctx in contexts:
        margins.append(_ln_logp(pools["pos"], ctx) - _ln_logp(pools["neg"], ctx))
    return float(np.nanmean(margins))


def _ln_logp_one(model, tok, ctx, answer, direction, layer, alpha, position) -> float:
    import torch
    import torch.nn.functional as F

    from explore_persona_space.experiments.issue1415 import steering

    prompt_ids = steering.context_token_ids(tok, ctx)
    ans_ids = tok.encode(answer, add_special_tokens=False)
    full = torch.tensor([prompt_ids + ans_ids], device=model.device)
    delta = direction.to(dtype=torch.bfloat16)
    with steering.DeltaHook(
        model,
        layer=layer,
        delta=delta,
        alpha=float(alpha),
        all_positions=(position == "answer"),
        expected_prompt_len=len(prompt_ids),
    ) as hook:
        hook.arm(expected_prompt_len=len(prompt_ids))
        with torch.no_grad():
            logits = model(full).logits[0]
    logps = F.log_softmax(logits.float(), dim=-1)
    tot = 0.0
    for i, tid in enumerate(ans_ids):
        tot += float(logps[len(prompt_ids) + i - 1, tid])
    return tot / max(1, len(ans_ids))


# ---------------------------------------------------------------------------
# phase: judge_reduce  (judge completions -> Delta-rate + null band + lattice)
# ---------------------------------------------------------------------------


def phase_judge_reduce(args) -> None:
    import numpy as np

    from explore_persona_space.experiments.issue_1739.judging import (
        judge_items_graded,
        load_trait_rubric,
        rollout_item_id,
    )

    out_root = _out_root(args)
    phase = args.reduce_phase  # "localize" or "decisive"
    comp_root = out_root / phase / "raw_completions"
    files = sorted(comp_root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no completions to judge under {comp_root}")
    _breadcrumb("judge_reduce", phase=phase, cells=len(files))
    cache_dir = out_root / phase / "judge_cache"
    save_raw = out_root / phase / "judge_raw"
    per_cell: dict[str, dict] = {}
    t0 = time.time()
    for fi, f in enumerate(files, 1):
        rows = json.loads(f.read_text())
        behavior = rows["cell"]["behavior"]
        rubric = load_trait_rubric(behavior)
        items: list[tuple[str, str, str]] = []
        for seed, sd in rows["seeds"].items():
            for qi, per_q in enumerate(sd["completions"]):
                for di, text in enumerate(per_q):
                    cid = f"{rows['cell_id']}__s{seed}__q{qi:02d}"[:40]
                    items.append((rollout_item_id(cid, di), _q_placeholder(qi), text))
        result = judge_items_graded(
            items,
            rubric,
            cache_dir=cache_dir / rows["cell_id"],
            save_raw=save_raw / rows["cell_id"],
            n_draws=1,
            max_tokens=JUDGE_MAX_TOKENS,
            dry_run=args.dry_run,
        )
        scores = np.asarray(result.scores, dtype=float)
        rate = float(np.mean(scores >= 50.0)) if scores.size else float("nan")
        per_cell[rows["cell_id"]] = {
            "cell": rows["cell"],
            "mean_score": float(np.nanmean(scores)) if scores.size else None,
            "rate": rate,
            "n_scored": int(scores.size),
        }
        _progress("judge_reduce", fi, len(files), rows["cell_id"], t0)

    reduced = _reduce_surface(per_cell, phase)
    payload = _run_metadata({"phase": phase, "per_cell": per_cell, "reduced": reduced})
    _write_json_atomic(out_root / phase / "reduced.json", payload)
    if phase == "localize":
        _write_json_atomic(
            out_root / "localize" / "operating_points.json", reduced.get("operating_points", {})
        )
    _write_sentinel(out_root, f"judge_reduce_{phase}", "done")
    _breadcrumb("judge_reduce", status="done", phase=phase)


def _q_placeholder(qi: int) -> str:
    """The judge fills {question}; we pass an index placeholder (content hygiene:
    the true question text is re-attached from the eval bank pod-side if the
    rubric requires it). Kept opaque in this driver's own logs."""
    return f"[eval_q_{qi:02d}]"


def _reduce_surface(per_cell: dict, phase: str) -> dict:
    """Per-cell Delta-rate + (localize) selection-symmetric null band + operating points.

    Delta-rate = cell rate - the behavior's alpha=0 rate. The null band is the
    argmax over the shuffled+random cells across the SAME (layer, dose, position)
    grid (plan §4.4 / selection-symmetric-nulls.md); the operating point per
    (signal direction, position) is the peak coherent layer x in-band dose.
    """
    import numpy as np

    by_behavior: dict[str, list] = {}
    for cid, rec in per_cell.items():
        by_behavior.setdefault(rec["cell"]["behavior"], []).append((cid, rec))
    out: dict[str, dict] = {"delta_rate": {}, "null_band": {}, "operating_points": {}}
    for behavior, recs in by_behavior.items():
        alpha0 = next((r["rate"] for _, r in recs if r["cell"]["direction"] == "alpha0"), 0.0)
        delta = {cid: (r["rate"] - alpha0) for cid, r in recs if r["cell"]["direction"] != "alpha0"}
        out["delta_rate"][behavior] = delta
        null_deltas = [
            d for cid, d in delta.items() if per_cell[cid]["cell"]["direction"] in NULL_DIRECTIONS
        ]
        if null_deltas:
            out["null_band"][behavior] = {
                "upper_edge": float(np.nanpercentile(null_deltas, 97.5)),
                "n_null_cells": len(null_deltas),
            }
        if phase == "localize":
            ops: dict[str, dict] = {}
            for direction in SIGNAL_DIRECTIONS + ("rb",):
                for position in POSITIONS:
                    cands = [
                        (cid, per_cell[cid]["cell"], d)
                        for cid, d in delta.items()
                        if per_cell[cid]["cell"]["direction"] == direction
                        and per_cell[cid]["cell"]["position"] == position
                    ]
                    if not cands:
                        continue
                    best = max(cands, key=lambda t: t[2] if np.isfinite(t[2]) else -1e9)
                    ops[f"{direction}__{position}"] = {
                        "layer": best[1]["layer"],
                        "c": best[1]["c"],
                        "delta_rate": best[2],
                    }
            out["operating_points"][behavior] = ops
    return out


# ---------------------------------------------------------------------------
# argparse dispatch
# ---------------------------------------------------------------------------

PHASES = {
    "materialize_directions": phase_materialize_directions,
    "norm_probe": phase_norm_probe,
    "localize": phase_localize,
    "decisive": phase_decisive,
    "margin": phase_margin,
    "judge_reduce": phase_judge_reduce,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="issue #2220 read-write duality driver")
    ap.add_argument("--phase", choices=sorted(PHASES), help="phase to run")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    ap.add_argument("--out-root", default="eval_results/issue_2220")
    ap.add_argument(
        "--u-store-dir",
        default="data/issue_2220/u_store",
        help="local staging dir for the #1092 whitening U pool",
    )
    ap.add_argument("--q1", type=int, default=Q1_LOCALIZE)
    ap.add_argument("--q2", type=int, default=Q2_DECISIVE)
    ap.add_argument("--draws-localize", type=int, default=DRAWS_LOCALIZE)
    ap.add_argument("--draws-decisive", type=int, default=DRAWS_DECISIVE)
    ap.add_argument("--reduce-phase", choices=("localize", "decisive"), default="localize")
    ap.add_argument("--workers", type=int, default=6, help="tar-stream range-reader workers")
    ap.add_argument("--window-mib", type=int, default=64, help="tar-stream window MiB")
    ap.add_argument("--force", action="store_true", help="ignore per-cell caches / .done")
    ap.add_argument("--smoke", action="store_true", help="tiny slice (1 behavior, 1 layer, 1 dose)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate the phase grid + validate wiring, no GPU/HF/model",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="AST arg-attribute completeness check, then exit 0",
    )
    return ap


def _apply_smoke(args) -> None:
    """Tiny-real slice: 1 behavior, dir=rb, position=answer, 1 layer, 1 dose,
    2 queries x 2 draws (plan §4.4 smoke). Scratch out-root so smoke never
    overwrites committed artifacts."""
    global DOSES_NONZERO
    args.behaviors = args.behaviors[:1]
    args.layers = args.layers[:1]
    args.q1 = 2
    args.q2 = 2
    args.draws_localize = 2
    args.draws_decisive = 2
    if args.out_root == "eval_results/issue_2220":
        args.out_root = "/tmp/issue-2220-smoke"
    DOSES_NONZERO = (1.0,)


def _dry_run_phase(args) -> None:
    """Enumerate the phase's grid + RESOLVE its deferred imports, no GPU/HF/model.

    A phase's deferred imports (natpv streaming, steering, extraction, judging)
    are otherwise unverified locally (the pod-only / GPU-only branches never run
    in a CPU smoke), so a missing symbol / signature drift would surface only on
    the pod after the expensive phases (#606/#823). This branch EXECUTES each
    phase's deferred imports (import resolution) but never CALLS a heavy body,
    and never calls `load_trait_rubric` (it can trigger the #779 asset-generation
    chain — a network side effect a local dry-run must not incur).
    """
    phase = args.phase
    if phase == "localize":
        from explore_persona_space.experiments.issue1415 import steering  # noqa: F401

        n = len(args.behaviors) * (
            len(DIRECTIONS) * len(POSITIONS) * len(args.layers) * len(DOSES_NONZERO) + 1
        )
        _breadcrumb("localize", dry_run=1, cells=n)
    elif phase == "decisive":
        from explore_persona_space.experiments.issue1415 import steering  # noqa: F401

        _breadcrumb(
            "decisive",
            dry_run=1,
            max_cells=len(args.behaviors) * (len(DIRECTIONS) * len(POSITIONS) + 1),
        )
    elif phase == "materialize_directions":
        # resolve the pod-only streaming reuse + its scripts.* sibling import (#823)
        _ensure_repo_root_on_syspath()
        import scripts.issue1739_natpv as natpv  # noqa: F401

        for sym in ("stream_members", "load_labels", "load_row_index", "_summary_re"):
            assert hasattr(natpv, sym), f"natpv missing {sym}"
        _breadcrumb(
            "materialize_directions",
            dry_run=1,
            dirs=len(args.behaviors) * len(DIRECTIONS) * len(args.layers),
        )
    elif phase == "norm_probe":
        from explore_persona_space.analysis.extraction import (  # noqa: F401
            extract_layer_activations,
        )
        from explore_persona_space.experiments.issue1415 import steering  # noqa: F401

        _breadcrumb("norm_probe", dry_run=1, probes=len(args.behaviors) * len(args.layers))
    elif phase == "margin":
        from explore_persona_space.experiments.issue1415 import steering  # noqa: F401

        _breadcrumb("margin", dry_run=1, behaviors=len(args.behaviors))
    elif phase == "judge_reduce":
        # import-resolution only; do NOT call load_trait_rubric (asset-gen chain)
        from explore_persona_space.experiments.issue_1739.judging import (  # noqa: F401
            judge_items_graded,
            load_trait_rubric,
            rollout_item_id,
        )

        for fn in (judge_items_graded, load_trait_rubric, rollout_item_id):
            assert callable(fn)
        _breadcrumb("judge_reduce", dry_run=1, reduce_phase=args.reduce_phase)
    print(f"[dry-run] {phase} wiring OK", flush=True)


def main() -> None:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.phase is None:
        raise SystemExit("--phase is required (or --import-check)")
    if args.smoke:
        _apply_smoke(args)
    if args.dry_run:
        _dry_run_phase(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    PHASES[args.phase](args)
    # Explicit hard-exit after flush: this driver imports torch/transformers/HF,
    # so a finalize-time teardown race can rewrite the rc (gotchas.md). Outputs
    # are fsynced (_write_json_atomic) + uploaded before here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
