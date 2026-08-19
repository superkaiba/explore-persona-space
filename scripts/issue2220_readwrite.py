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
  check_disjoint          A8 eval-bank disjointness gate (plan §6 / §12 assumption 8):
                          normalized-text set overlap of the eval bank vs the #779
                          extraction sets + the #1092 map-fit corpus queries. CPU,
                          0 GPU, pre-spend; the same guard also runs at the top of
                          localize / decisive / judge_reduce.
  localize                6 dirs x 2 pos x 5 layers x {c=0.5,1,2,4} + alpha=0 ref;
                          Q1=10, draws=3, seed 42 -> DeltaHook+generate_batch. GPU.
  decisive                6 dirs + alpha=0 at selected operating points x 2 pos;
                          Q2=20, draws=5, seeds {42,43}. GPU.
  margin                  teacher-forced fixed +/- completion-pool margin DV. GPU.
  judge_reduce            judge the persisted completions -> per-cell Delta-rate +
                          selection-symmetric null band + verdict lattice. CPU + Batch API.

STATED DEVIATION (round 3; concern ``judged-coherence-covariate-missing``): the
plan §4.3/§10 Sonnet 0-100 coherence score "folded into one multi-field judge
call" is NOT implemented. The reused #1739/#779 graded-judge chain
(``judge_items_graded`` -> ``eval.graded_judge.judge_graded``) parses a single
``{"score": N|"REFUSAL"}`` field by construction (format suffix, parse path,
JudgeResult, rubric-keyed caches, tallies are all single-score); a multi-field
retrofit would mutate shared live library code or fork the whole batch/cache/
drop-never-coerce stack. The coherence GATE and the fragility-vs-content-limited
covariate both use the PROGRAMMATIC per-cell coherence rate
(steering.coherence_check / condition_passes, persisted per draw at generation
time and propagated into every reduced surface + delta record) — the gate plan
§4.3 itself registers. Judge spend is unchanged (single-field trait rubric).

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

# #2220 THROUGHPUT FIX (defense-in-depth; the launcher ALSO exports these):
# hf_transfer acceleration must be in the env BEFORE any transitive
# huggingface_hub import — huggingface_hub.constants freezes both at import
# time — so these are the FIRST executable lines of the module.
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import argparse
import hashlib
import json
import logging
import subprocess
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
    CORPUS_MANIFEST_REVISION,
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
# The operating-point argmax iterates DIRECTIONS (nulls included); the null band
# iterates NULL_DIRECTIONS. (The unused SIGNAL_DIRECTIONS tuple was removed in
# round 3 — code-review v2 minor.)
DIRECTIONS = ("mapread_ctx", "mapread_prefix", "rb", "rawmeandiff", "shuffled", "random")
NULL_DIRECTIONS = ("shuffled", "random")

# Direction -> whitening/summary arm it is derived from (plan §6 pooling-convention).
DIRECTION_SUMMARY_KIND = {
    "mapread_ctx": "context_end",
    "mapread_prefix": "prefix_end",
    "rawmeandiff": "context_end",
    "shuffled": "context_end",
}
N_RANDOM_SEEDS = 3  # direction 6 = mean over 3 matched-norm random unit vectors (plan §4.1)

# Generous rationale budget for the SINGLE-FIELD #779 trait rubric (llm-judging
# rule 23 — >=1024 single-rationale floor; 2048 keeps headroom, a cap is not a
# spend; NOT #1739's 400). The plan's multi-field trait+coherence rubric is a
# RECORDED DEVIATION (module docstring): coherence is the programmatic per-cell
# rate, so the judge instrument stays single-field.
JUDGE_MAX_TOKENS = 2048
GEN_MAX_NEW_TOKENS = 2048  # free-generation default; parent used 1024 (plan §6 deviation)

# Question counts / draws (plan §4.4).
Q1_LOCALIZE = 10
DRAWS_LOCALIZE = 3
SEED_LOCALIZE = 42
Q2_DECISIVE = 20
DRAWS_DECISIVE = 5
SEEDS_DECISIVE = (42, 43)

# Judge draws per reduce phase (plan §10 Judge row: N=5 decisive / 3 localize;
# llm-judging graded-primary N>=5 at the decisive read). DISTINCT from the
# generation draws above, which coincidentally share the numbers.
JUDGE_DRAWS_LOCALIZE = 3
JUDGE_DRAWS_DECISIVE = 5

# Graded-score threshold (0-100 judge; rate = frac of completions with mean
# kept-draw score >= threshold; also the +/- pool filter split, plan §4.4).
SCORE_THRESHOLD = 50.0

# Question-cluster paired bootstrap (plan §6: resample the eval questions with
# replacement; BOTH arms of every Delta/G-margin recomputed within each
# resample — one shared index draw per behavior).
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20220

# Cell-level coherence gate (plan §4.3: steering.condition_passes = >=50%
# coherent draws per (seed, question) context; a CELL passes when >= this
# fraction of its contexts pass). Operating-point + null-argmax candidates are
# restricted to gate-passing cells (#1415 coherent-alpha region).
COHERENCE_CELL_GATE = 0.5

# Fixed +/- teacher-forced answer pools (plan §4.4: ~10 pos / 10 neg per
# behavior, judge-filtered ONCE and held fixed across contexts; llm-judging §E2).
POOL_SIZE = 10
POOL_MIN = 3  # fail-loud floor per side (a 2-answer pool is not a pool)

# Cap-hit re-generation trigger (CLAUDE.md generation rule: cap-hit > 2% per
# family/cell => re-generate those rows at >= 2x the cap).
CAP_HIT_REGEN_FRAC = 0.02
CAP_HIT_REGEN_FACTOR = 2

# Teacher-forced margin batching (plan §9 margin row: "batched").
MARGIN_BATCH_SIZE = int(os.environ.get("EPM_MARGIN_BATCH", "8") or "8")

# Plan-named reduced-surface deliverables per judged phase (§6.5/§9 filename
# literals): localize -> the coherence-gated layer x dose surface; decisive ->
# the judged per-cell surface (code-review v2 Major plan-named-deliverable-
# filenames; decisive/delta_rate_percell.json is written ADDITIONALLY).
_REDUCED_SURFACE_NAME = {"localize": "dose_response.json", "decisive": "judged.json"}

# HF destinations (plan §10).
HF_PREFIX = "issue2220_readwrite"  # data repo prefix

# Smoke uploads divert to a smoke/ sub-prefix: smoke cell files carry
# production-identical names, so uploading them to the canonical prefix would
# overwrite real artifacts (smoke-outputs rule). Set by _apply_smoke.
_SMOKE_UPLOAD_SUBPREFIX = False


def _hf_prefix() -> str:
    """Data-repo prefix for all uploads; `<HF_PREFIX>/smoke` under --smoke."""
    return f"{HF_PREFIX}/smoke" if _SMOKE_UPLOAD_SUBPREFIX else HF_PREFIX


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


_E1_ASSETS_CACHE: dict[str, dict] = {}


def _e1_assets(behavior: str) -> dict:
    """Memoized #779 asset load (one asset-chain hit per behavior per process —
    the regeneration fallback inside load_e1_assets must never run twice)."""
    if behavior not in _E1_ASSETS_CACHE:
        from explore_persona_space.experiments.issue_1739.generation import load_e1_assets

        _E1_ASSETS_CACHE[behavior] = load_e1_assets(behavior)
    return _E1_ASSETS_CACHE[behavior]


def _eval_questions(behavior: str) -> list[str]:
    """The persona-vectors disjoint EVAL question set for ``behavior``.

    Loaded via the #779 asset chain (issue_1739.generation.load_e1_assets),
    key ``eval_questions`` ONLY — both #779 artifact shapes carry it
    (scripts/issue779_common.py: the paper-verbatim evil artifacts and the
    regenerated ``questions[20:]`` split). FAIL-LOUD on a missing/empty key:
    silently tail-slicing ``extraction_questions`` could steer/judge on the
    questions the directions were built from (code-review v2, plan §6 A8);
    ``_assert_eval_bank_disjoint`` is the mechanical pre-spend backstop.
    CONTENT HYGIENE: question text is passed to the model/judge, never logged.
    """
    assets = _e1_assets(behavior)
    qs = assets.get("eval_questions")
    if not qs:
        raise RuntimeError(
            f"[{behavior}] #779 assets carry no 'eval_questions' key "
            f"(keys={sorted(assets)}); refusing the extraction tail-slice fallback — "
            "the eval bank must be the disjoint persona-vectors eval set (plan §6 A8)"
        )
    if len(qs) < Q2_DECISIVE:
        raise RuntimeError(
            f"[{behavior}] eval bank too small: {len(qs)} < Q2={Q2_DECISIVE} (plan §4.4)"
        )
    return list(qs)


# ---------------------------------------------------------------------------
# A8 eval-bank disjointness gate (plan §6 / §12 assumption 8) — pre-spend
# ---------------------------------------------------------------------------


def _norm_question(s: str) -> str:
    """Whitespace-collapsed casefold — the text-identity grain of the A8 overlap."""
    return " ".join(str(s).split()).casefold()


def _corpus_query_texts() -> set[str]:
    """Normalized #1092 map-fit corpus query texts (the corpus question surface).

    The pinned #1092 manifest rows carry ids only (text lives in the sibling
    ``query_store.jsonl`` — issue1092_claude_text/_gpu_phase read
    ``entry["text"]``), so the query store IS the corpus question set the eval
    bank must be disjoint from. Fetched at the #1739 corpus pin; the download
    is HF-cache-idempotent across the per-phase guard calls. Texts are
    normalized + set-compared, NEVER logged (content hygiene).
    """
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    path = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename="issue1092_realistic_crossing/corpus/query_store.jsonl",
            repo_type="dataset",
            revision=CORPUS_MANIFEST_REVISION,
        ),
        what="fetch #1092 query_store (A8 disjointness)",
    )
    texts: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            entry = json.loads(line)
            t = entry.get("text") or entry.get("query")
            if t:
                texts.add(_norm_question(t))
    if not texts:
        raise RuntimeError("A8: #1092 query_store yielded 0 query texts — check the corpus pin")
    return texts


def _extraction_question_texts(behavior: str) -> set[str]:
    """Normalized #779 extraction-set questions (the direction-fit questions)."""
    return {_norm_question(q) for q in _e1_assets(behavior)["extraction_questions"]}


def _assert_eval_bank_disjoint(behaviors, *, corpus_texts: set[str] | None = None) -> dict:
    """Fail-loud A8 sha/text set-overlap gate (plan §6; §12 assumption 8).

    Every behavior's steering/judging eval bank must be DISJOINT from (a) the
    pooled #779 extraction sets of the requested behaviors (the questions the
    directions were built from) and (b) the #1092 map-fit corpus query texts.
    Raises RuntimeError naming counts + sha8 digests (never question text).
    Returns a JSON-safe record ``phase_check_disjoint`` persists. Called at the
    top of every spend-bearing phase (localize / decisive / judge_reduce).
    """
    corpus = _corpus_query_texts() if corpus_texts is None else corpus_texts
    extraction: set[str] = set()
    for b in behaviors:
        extraction |= _extraction_question_texts(b)
    record: dict = {
        "n_corpus_query_texts": len(corpus),
        "n_extraction_texts": len(extraction),
        "behaviors": {},
    }
    problems: list[str] = []
    for b in behaviors:
        ev = [_norm_question(q) for q in _eval_questions(b)]
        ev_set = set(ev)
        if len(ev_set) != len(ev):
            problems.append(f"{b}: eval bank carries duplicate questions")
        hit_x = sorted(ev_set & extraction)
        hit_c = sorted(ev_set & corpus)
        record["behaviors"][b] = {
            "n_eval": len(ev_set),
            "eval_bank_sha8": _sha8(sorted(ev_set)),
            "n_overlap_extraction": len(hit_x),
            "n_overlap_corpus": len(hit_c),
        }
        if hit_x:
            problems.append(
                f"{b}: {len(hit_x)} eval questions overlap the #779 extraction set "
                f"(sha8 {[_sha8(t) for t in hit_x[:5]]})"
            )
        if hit_c:
            problems.append(
                f"{b}: {len(hit_c)} eval questions overlap the #1092 map-fit corpus "
                f"(sha8 {[_sha8(t) for t in hit_c[:5]]})"
            )
    if problems:
        raise RuntimeError(
            "A8 eval-bank disjointness FAILED (would steer/judge on direction-fit "
            "questions): " + "; ".join(problems)
        )
    logger.info(
        "[check_disjoint] A8 PASS: %d behaviors disjoint from %d extraction + %d corpus texts",
        len(record["behaviors"]),
        len(extraction),
        len(corpus),
    )
    return record


def phase_check_disjoint(args) -> None:
    """Standalone A8 gate phase (CPU, 0 GPU; run pod-side BEFORE any spend)."""
    out_root = _out_root(args)
    record = _assert_eval_bank_disjoint(list(args.behaviors))
    _write_json_atomic(out_root / "check_disjoint" / "disjointness.json", _run_metadata(record))
    _write_sentinel(out_root, "check_disjoint", "done")
    _breadcrumb("check_disjoint", status="done", behaviors=len(args.behaviors))


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

# Parent-issue committed inputs (#1739 labeling.json) that natpv.load_labels
# reads from the REPO CHECKOUT: committed at HEAD, but OUTSIDE the
# partial-clone pods' default sparse cones (`src scripts configs tests docs`
# + tracked data/) and excluded by sparse worktrees — "committed" != "present"
# (gotchas.md "Partial-clone pods", #2211). Minimal covering cone for the
# three behaviors' files; the round-5 sweep found NO sibling committed reads
# anywhere on the driver path (widen only with a fresh sweep).
_PARENT_INPUTS_CONE = "eval_results/issue_1739/dv_dataset"


def _ensure_parent_issue_cones(behaviors, repo_root: Path | None = None) -> None:
    """Auto-materialize the #1739 committed labeling inputs (cone-ensure).

    Runtime data-availability op (like ``stage_u_store``), NOT a code edit on
    the pod: idempotent skip when every required file is already present;
    else ``git sparse-checkout add`` the minimal covering cone in the repo
    root this driver resolves (the partial clone fetches the blobs on
    demand), then FAIL-LOUD verify with the ``prefetch_inputs``
    ``_assert_git_input`` remedy shape (#612). The ``[cone-ensure]`` log
    lines are the round-5 crash-fix fix-engaged signal.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "pyproject.toml").exists(), f"repo-root sentinel missing at {repo_root}"
    rels = [f"{_PARENT_INPUTS_CONE}/{b}/labeling.json" for b in behaviors]
    missing = [r for r in rels if not (repo_root / r).is_file()]
    if not missing:
        logger.info("[cone-ensure] %d parent-issue git inputs present; skipping", len(rels))
        return
    proc = subprocess.run(
        ["git", "sparse-checkout", "add", _PARENT_INPUTS_CONE],
        cwd=repo_root,
        env={**os.environ},
        capture_output=True,
        text=True,
    )
    logger.info(
        "[cone-ensure] git sparse-checkout add %s -> rc=%d%s",
        _PARENT_INPUTS_CONE,
        proc.returncode,
        f" stderr={proc.stderr.strip()!r}" if proc.returncode != 0 else "",
    )
    still = [r for r in rels if not (repo_root / r).is_file()]
    if still:
        raise FileNotFoundError(
            f"Frozen git inputs missing after cone-ensure: {still}. Run "
            f"`git -C {repo_root} sparse-checkout add {_PARENT_INPUTS_CONE}` "
            "(partial-clone pods + sparse worktrees exclude eval_results/ from the "
            "default cones — committed != present; gotchas.md 'Partial-clone pods', "
            "#2211) or `git pull`."
            + (f" git stderr: {proc.stderr.strip()!r}" if proc.stderr.strip() else "")
        )
    logger.info("[cone-ensure] materialized %d parent-issue git inputs", len(missing))


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
            # natpv.phase_rowindex reads args.revision (this driver's parser has
            # no --revision) — pin it to the same tar revision the acts stream
            # above used, so the lazy per-row path cannot AttributeError and
            # cannot read a different tar revision (#2220 fix round).
            if getattr(args, "revision", None) is None:
                args.revision = LABELING_TAR_REVISION
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
    # Crash-fix (epm:failure v1, chunk-1 FileNotFoundError): nothing staged the
    # #1092 store onto a fresh pod before the read. stage_u_store (#1739) is
    # idempotent — it short-circuits via u_store_loadable when dest already
    # serves the requested (kind x layer) grid — and its cell/revision/manifest
    # kwargs default to the same module pins (U_STORE_CELL, STORE_REVISION) the
    # load_summaries call below consumes. stage_u_store FLATTENS the cell's
    # shards into ``dest``, while load_summaries(..., cell=U_STORE_CELL) reads
    # ``local_dir / U_STORE_CELL`` — so dest is that exact root (one path,
    # staged then read; the loader stays byte-untouched).
    stage_root = local_dir / U_STORE_CELL
    logger.info("[stage-u-store] staging U pool into %s (layers=%s)", stage_root, list(layers))
    store_io.stage_u_store(
        dest=stage_root,
        kinds=("context_end", "prefix_end"),
        layers=tuple(layers),
    )
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
    """Materialize the six direction arms per (behavior, layer) — plan §4.1."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits, store_io

    # #2220 THROUGHPUT FIX: route natpv.stream_members — BOTH the
    # _stream_labeled_context_acts pass AND the lazy _row_index ->
    # natpv.phase_rowindex path — through the hf_transfer materialize-then-read
    # branch. Measured on pod-2220: the ParallelRangeReader range-GET route ran
    # ~1 MB/s aggregate (~43 h projected for the 154 GB of labeling tars); a
    # plain hf_hub_download of the SAME tar with HF_HUB_ENABLE_HF_TRANSFER=1
    # ran ~499 MB/s (154 GB in ~5-13 min). Peak disk = ONE tar (<=~70 GB,
    # deleted per call in natpv._materialized_members' finally). Staging goes
    # to the canonical re-downloadable cache path (disk-hygiene reap target).
    _ensure_repo_root_on_syspath()
    import scripts.issue1739_natpv as natpv

    natpv.MATERIALIZE_TARS = True
    natpv.MATERIALIZE_STAGING_DIR = Path("data/issue_2220/hf_dl/labeling_tars")

    out_root = _out_root(args)
    dir_out = out_root / "directions"
    dir_out.mkdir(parents=True, exist_ok=True)
    stage = out_root / "labeling_stage"
    stage.mkdir(parents=True, exist_ok=True)
    layers = list(args.layers)
    behaviors = list(args.behaviors)
    _breadcrumb("materialize_directions", behaviors=len(behaviors), layers=len(layers))
    # Cone-ensure FIRST — fail fast on the committed #1739 inputs BEFORE the
    # 8.5 GB U-pool stage / r_B fetch (round-5 crash: load_labels
    # FileNotFoundError on a partial-clone pod, #2211 class).
    _ensure_parent_issue_cones(behaviors)

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
    # Plan §9 phase_outputs literal: directions/manifest.json.
    _write_json_atomic(dir_out / "manifest.json", manifest)
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
    files = sorted(dir_out.glob("*.pt")) + [dir_out / "manifest.json"]
    files = [f for f in files if f.exists()]
    if not files:
        logger.warning("[upload] no direction files under %s", dir_out)
        return
    allow = ["*.pt", "*.json"]
    hub.assert_hub_dir_filecounts(str(dir_out), f"{_hf_prefix()}/directions", allow_patterns=allow)
    hub.retry_transient(
        lambda: api.upload_folder(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(dir_out),
            path_in_repo=f"{_hf_prefix()}/directions",
            allow_patterns=allow,
        ),
        what="upload directions bank",
    )
    logger.info(
        "[upload] %d direction files -> %s/%s/directions", len(files), HF_DATA_REPO, _hf_prefix()
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
    # Plan §9 phase_outputs literal: norm_probe/rho_by_layer.json.
    _write_json_atomic(out_root / "norm_probe" / "rho_by_layer.json", payload)
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
    p = out_root / "norm_probe" / "rho_by_layer.json"
    if not p.exists():
        raise FileNotFoundError(f"norm_probe not run: {p} (run --phase norm_probe first)")
    return json.loads(p.read_text())["rho_median_last_context_token"]


def _steer_cell(
    model,
    tok,
    direction,
    layer,
    alpha,
    position,
    contexts,
    *,
    n_draws,
    seed_base,
    max_new_tokens=GEN_MAX_NEW_TOKENS,
):
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
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            seed_base=seed_base,
        )
    return results  # results[b][i] -> new-token text


def _cap_hit_fraction(results, tok, cap: int = GEN_MAX_NEW_TOKENS) -> float:
    """Fraction of draws that hit the max_new_tokens cap (proxy: token length >=
    ``cap``). CLAUDE.md generation-stage rule."""
    total = hit = 0
    for row in results:
        for text in row:
            total += 1
            if len(tok.encode(text, add_special_tokens=False)) >= cap:
                hit += 1
    return (hit / total) if total else 0.0


def _cell_id(cell: dict) -> str:
    """Filename-grade cell id (NOT a judge custom id — see judge_context_id)."""
    cid = "__".join(f"{k}{cell[k]}" for k in ("behavior", "direction", "position", "layer", "c"))
    return cid.replace(".", "p")


def _needs_cap_regen(rows: dict) -> bool:
    """True when a persisted cell tripped the cap-hit re-gen trigger (> 2% of
    draws at the cap) and has NOT already been re-generated at the doubled cap."""
    cap = int(rows.get("max_new_tokens", GEN_MAX_NEW_TOKENS))
    return (
        float(rows.get("cap_hit_fraction", 0.0)) > CAP_HIT_REGEN_FRAC
        and cap < CAP_HIT_REGEN_FACTOR * GEN_MAX_NEW_TOKENS
    )


def _gen_cell_rows(
    model, tok, cell: dict, dir_out: Path, rho: dict, contexts, *, n_draws, seeds, max_new_tokens
) -> dict:
    """Generate one cell's rows payload (all seeds): completions + per-draw
    coherence flags + per-context condition_passes + cap-hit fraction."""
    from explore_persona_space.experiments.issue1415 import steering

    behavior = cell["behavior"]
    layer = cell["layer"]
    if cell["direction"] == "alpha0" or cell["c"] == 0.0:
        alpha = 0.0
        direction = None
    else:
        direction = _load_direction(dir_out, behavior, cell["direction"], layer)
        rho_l = rho[behavior][f"L{layer}"]
        alpha = cell["c"] * rho_l
    rows = {"cell_id": _cell_id(cell), "cell": cell, "seeds": {}}
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
                    max_new_tokens=max_new_tokens,
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
                max_new_tokens=max_new_tokens,
            )
        coh = [steering.coherence_check(row) for row in res]
        rows["seeds"][str(seed)] = {
            "completions": res,
            "coherent_flags": coh,
            "condition_passes": [steering.condition_passes(c) for c in coh],
        }
    rows["alpha"] = float(alpha)
    rows["max_new_tokens"] = int(max_new_tokens)
    rows["cap_hit_fraction"] = _cap_hit_fraction(
        [r for s in rows["seeds"].values() for r in s["completions"]], tok, cap=max_new_tokens
    )
    return rows


def _run_steer_grid(
    args, phase: str, cells: list[dict], contexts_by_behavior, *, n_draws: int, seeds: list[int]
) -> None:
    """Shared per-cell steering loop with per-cell checkpointing + resume.

    ``--shard-id/--num-shards`` round-robin-slices the cell list (the plan §9
    4-GPU fan-out: one process per GPU, launcher-env CVD-pinned); per-cell
    files make resume shard-safe, and the done sentinel is shard-suffixed when
    sharded. Durability ordering: upload runs BEFORE the done sentinel +
    breadcrumb (never after — the poller may act on `done` immediately)."""
    out_root = _out_root(args)
    dir_out = out_root / "directions"
    comp_root = out_root / phase / "raw_completions"
    comp_root.mkdir(parents=True, exist_ok=True)
    assert 0 <= args.shard_id < args.num_shards, (args.shard_id, args.num_shards)
    if args.num_shards > 1:
        cells = cells[args.shard_id :: args.num_shards]
    model, tok = _load_model_and_tokenizer()
    rho = _load_rho(out_root)
    _breadcrumb(
        phase, cells=len(cells), seeds=len(seeds), shard=f"{args.shard_id}/{args.num_shards}"
    )
    t0 = time.time()
    n = len(cells)
    for ci, cell in enumerate(cells, 1):
        cell_id = _cell_id(cell)
        out_path = comp_root / f"{cell_id}.json"
        if out_path.exists() and not args.force:
            _progress(phase, ci, n, cell_id + " (cached)", t0)
            continue
        rows = _gen_cell_rows(
            model,
            tok,
            cell,
            dir_out,
            rho,
            contexts_by_behavior[cell["behavior"]],
            n_draws=n_draws,
            seeds=seeds,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
        )
        _write_json_atomic(out_path, rows)
        _progress(phase, ci, n, cell_id, t0)

    # Cap-hit re-gen trigger (pre-registered: > 2% per cell => re-generate the
    # cell at 2x the cap; CLAUDE.md generation rule). Bounded single pass.
    regen = []
    for cell in cells:
        out_path = comp_root / f"{_cell_id(cell)}.json"
        if out_path.exists() and _needs_cap_regen(json.loads(out_path.read_text())):
            regen.append(cell)
    for k, cell in enumerate(regen, 1):
        cell_id = _cell_id(cell)
        logger.warning(
            "[%s] cap-hit > %.2f on %s -> re-generating at %d new tokens",
            phase,
            CAP_HIT_REGEN_FRAC,
            cell_id,
            CAP_HIT_REGEN_FACTOR * GEN_MAX_NEW_TOKENS,
        )
        rows = _gen_cell_rows(
            model,
            tok,
            cell,
            dir_out,
            rho,
            contexts_by_behavior[cell["behavior"]],
            n_draws=n_draws,
            seeds=seeds,
            max_new_tokens=CAP_HIT_REGEN_FACTOR * GEN_MAX_NEW_TOKENS,
        )
        _write_json_atomic(comp_root / f"{cell_id}.json", rows)
        _progress(f"{phase}-capregen", k, len(regen), cell_id, t0)

    # Durable upload FIRST, sentinel + done breadcrumb LAST (#528 family; the
    # materialize phase already orders it this way).
    _upload_raw_completions(out_root, phase)
    sent = phase if args.num_shards == 1 else f"{phase}-shard{args.shard_id}"
    _write_sentinel(out_root, sent, "done", {"cells": len(cells), "cap_regen": len(regen)})
    _breadcrumb(phase, status="done", cells=len(cells), cap_regen=len(regen))


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
    allow = ["*.json"]
    hub.assert_hub_dir_filecounts(
        str(comp_root), f"{_hf_prefix()}/raw_completions/{phase}", allow_patterns=allow
    )
    hub.retry_transient(
        lambda: api.upload_folder(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(comp_root),
            path_in_repo=f"{_hf_prefix()}/raw_completions/{phase}",
            allow_patterns=allow,
        ),
        what=f"upload {phase} raw completions",
    )
    logger.info(
        "[upload] %d %s completion files -> %s/%s/raw_completions/%s",
        len(files),
        phase,
        HF_DATA_REPO,
        _hf_prefix(),
        phase,
    )


def phase_localize(args) -> None:
    layers = list(args.layers)
    behaviors = list(args.behaviors)
    _assert_eval_bank_disjoint(behaviors)  # A8 pre-spend gate (plan §6)
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
    _assert_eval_bank_disjoint(behaviors)  # A8 pre-spend gate (plan §6)
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
    # Load EVERY cheap input (pools, eval contexts, rho) BEFORE the 7B model
    # load, so a missing pool / norm probe fails in seconds, not after a model
    # load (code-review v2 minor). rho is also hoisted out of the cell loop
    # (it was re-read from disk once per cell).
    pools_by_behavior = {
        b: _load_answer_pools(out_root, b) for b in behaviors
    }  # {"pos": [...], "neg": [...]}
    contexts_by_behavior = {
        b: _contexts_for_questions(_eval_questions(b)[: args.q2]) for b in behaviors
    }
    rho = _load_rho(out_root)
    model, tok = _load_model_and_tokenizer()
    _breadcrumb("margin", behaviors=len(behaviors))
    result: dict[str, dict] = {}
    t0 = time.time()
    for bi, behavior in enumerate(behaviors, 1):
        pools = pools_by_behavior[behavior]
        contexts = contexts_by_behavior[behavior]
        cell_margins: dict[str, float] = {}
        for direction in DIRECTIONS:
            for position in POSITIONS:
                sel = op.get(behavior, {}).get(f"{direction}__{position}")
                if sel is None:
                    continue
                d_vec = _load_direction(dir_out, behavior, direction, sel["layer"])
                alpha = sel["c"] * rho[behavior][f"L{sel['layer']}"]
                m = _teacher_forced_margin(
                    model, tok, contexts, pools, d_vec, sel["layer"], alpha, position
                )
                cell_margins[f"{direction}__{position}"] = float(m)
        result[behavior] = cell_margins
        _progress("margin", bi, len(behaviors), behavior, t0)
    payload = _run_metadata({"tf_margin": result})
    # Plan §9 phase_outputs literal: margin/margin_percell.json.
    _write_json_atomic(out_root / "margin" / "margin_percell.json", payload)
    _write_sentinel(out_root, "margin", "done")
    _breadcrumb("margin", status="done")


def _load_answer_pools(out_root: Path, behavior: str) -> dict:
    p = out_root / "margin" / "pools" / f"{behavior}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"fixed +/- answer pool missing: {p} (built pod-side from judge-filtered completions)"
        )
    return json.loads(p.read_text())


def _tf_answer_hook(model, layer, delta, alpha, edit_from: int):
    """Teacher-forced analogue of the all_positions (answer-position) steering.

    Generation-mode ``all_positions`` edits the position GENERATING each answer
    token: the last prompt position at prefill, then every decode-step
    position — i.e. positions ``n_p-1 .. n_p+m-2``. In the single teacher-forced
    forward over ``prompt + answer (+ right pad)`` the equivalent is one edit of
    every position ``>= edit_from`` (= ``n_p-1``): the extra edited positions
    (the last answer token's own slot + right-pad slots) influence only logits
    at positions AFTER the scored ones (causal attention + attention mask), so
    they are inert for the LN-logP read. Returns an installed-on-enter DeltaHook
    subclass instance (local to this driver; the shared #1415 class is not
    modified)."""
    from explore_persona_space.experiments.issue1415 import steering

    class _TFAnswerRangeHook(steering.DeltaHook):
        def _edit_tensor(self, hidden):
            B, T, H = hidden.shape
            d = self.delta.to(device=hidden.device, dtype=hidden.dtype)
            assert d.shape[-1] == H, (d.shape, H)
            scaled = self.alpha * d
            if self._prefill_seen:  # single-forward mode: only the first pass edits
                return hidden
            assert 0 <= edit_from < T, (edit_from, T)
            out = hidden.clone()
            out[:, edit_from:, :] = out[:, edit_from:, :] + scaled
            self._prefill_seen = True
            self.n_edits += 1
            return out

    return _TFAnswerRangeHook(model, layer=layer, delta=delta, alpha=float(alpha))


def _tf_hook_for(model, layer, delta, alpha, position, n_prompt: int):
    """The armed teacher-forced steering hook for one full-sequence forward.

    position == "context": DeltaHook edit_position mode via ``arm_at(n_p-1)`` —
    the documented teacher-forced mode (the generation-mode
    ``expected_prompt_len == T`` assert is inapplicable when T = prompt+answer;
    steering.py docstring). position == "answer": the range hook above."""
    from explore_persona_space.experiments.issue1415 import steering

    if position == "answer":
        return _tf_answer_hook(model, layer, delta, alpha, edit_from=n_prompt - 1)
    hook = steering.DeltaHook(model, layer=layer, delta=delta, alpha=float(alpha))
    hook.arm_at(n_prompt - 1)
    return hook


def _teacher_forced_margin(model, tok, contexts, pools, direction, layer, alpha, position) -> float:
    """mean LN-logP(fixed pos pool | C) - mean LN-logP(fixed neg pool | C) under steering.

    BATCHED (plan §9 margin row): per context, all pool answers run as
    right-padded teacher-forced forwards in chunks of MARGIN_BATCH_SIZE through
    ``_batched_ln_logp``; the serial oracle ``_ln_logp_one`` is kept ONLY as the
    equivalence-test reference (tests/test_issue2220_margin_batched.py)."""
    import numpy as np

    from explore_persona_space.experiments.issue1415 import steering

    pos_ids = [tok.encode(a, add_special_tokens=False) for a in pools["pos"]]
    neg_ids = [tok.encode(a, add_special_tokens=False) for a in pools["neg"]]
    pad_id = tok.pad_token_id
    assert pad_id is not None
    margins = []
    for ctx in contexts:
        prompt_ids = steering.context_token_ids(tok, ctx)
        lp = _batched_ln_logp(
            model, prompt_ids, pos_ids + neg_ids, direction, layer, alpha, position, pad_id=pad_id
        )
        p = [v for v in lp[: len(pos_ids)] if np.isfinite(v)]
        q = [v for v in lp[len(pos_ids) :] if np.isfinite(v)]
        pm = float(np.mean(p)) if p else float("nan")
        qm = float(np.mean(q)) if q else float("nan")
        margins.append(pm - qm)
    return float(np.nanmean(margins))


def _batched_ln_logp(
    model, prompt_ids, answers_ids, direction, layer, alpha, position, *, pad_id, batch_size=None
) -> list[float]:
    """Length-normalized logP of each answer continuation, batched (F9).

    One right-padded teacher-forced forward per <= batch_size answers under a
    single armed steering hook (shared prompt => constant edit anchor). Right
    padding keeps real tokens LEFT-aligned, so default position_ids are correct
    (RoPE indexes from 0 over the real tokens) and causal attention + the
    attention mask make pad slots inert for every scored position. Serial
    reference: ``_ln_logp_one`` (equivalence-gated, cosine >= 0.999 +
    chunk-size invariance, tests/test_issue2220_margin_batched.py)."""
    import torch
    import torch.nn.functional as F

    bs = int(batch_size or MARGIN_BATCH_SIZE)
    n_p = len(prompt_ids)
    delta = direction.to(dtype=next(model.parameters()).dtype)
    out: list[float] = []
    for k in range(0, len(answers_ids), bs):
        chunk = answers_ids[k : k + bs]
        t_max = max(n_p + max(1, len(a)) for a in chunk)
        batch = torch.full((len(chunk), t_max), int(pad_id), dtype=torch.long)
        attn = torch.zeros((len(chunk), t_max), dtype=torch.long)
        for r, a in enumerate(chunk):
            row = list(prompt_ids) + list(a)
            batch[r, : len(row)] = torch.tensor(row, dtype=torch.long)
            attn[r, : len(row)] = 1
        batch = batch.to(model.device)
        attn = attn.to(model.device)
        with _tf_hook_for(model, layer, delta, alpha, position, n_p):
            with torch.no_grad():
                logits = model(input_ids=batch, attention_mask=attn).logits
        logps = F.log_softmax(logits.float(), dim=-1)
        for r, a in enumerate(chunk):
            if not a:
                out.append(float("nan"))
                continue
            pos_idx = torch.arange(n_p - 1, n_p - 1 + len(a), device=logps.device)
            tok_idx = torch.as_tensor(list(a), dtype=torch.long, device=logps.device)
            vals = logps[r, pos_idx, tok_idx]
            out.append(float(vals.sum().item()) / len(a))
    return out


def _ln_logp_one(model, prompt_ids, ans_ids, direction, layer, alpha, position) -> float:
    """Serial (batch-1) LN-logP oracle for ``_batched_ln_logp``.

    Kept ONLY as the seeded serial reference the batched-rewrite equivalence
    gate compares against (vectorize-many-cell-fits equivalence-gate recipe);
    production dispatches the batched path. Same teacher-forced hook modes as
    the batched path (the round-1 ``arm(expected_prompt_len=n_p)`` form would
    trip DeltaHook's ``expected_prompt_len == T`` assert on the full-sequence
    forward — T = prompt+answer there)."""
    import torch
    import torch.nn.functional as F

    if not ans_ids:
        return float("nan")
    full = torch.tensor([list(prompt_ids) + list(ans_ids)], device=model.device)
    n_p = len(prompt_ids)
    delta = direction.to(dtype=next(model.parameters()).dtype)
    with _tf_hook_for(model, layer, delta, alpha, position, n_p):
        with torch.no_grad():
            logits = model(full).logits[0]
    logps = F.log_softmax(logits.float(), dim=-1)
    tot = 0.0
    for i, tid in enumerate(ans_ids):
        tot += float(logps[n_p + i - 1, tid])
    return tot / len(ans_ids)


# ---------------------------------------------------------------------------
# phase: judge_reduce  (judge completions -> Delta-rate + null band + lattice)
# ---------------------------------------------------------------------------


# Judge custom-id short codes (Batch custom_id grammar ^[a-zA-Z0-9_-]{1,64}$;
# judge_graded appends "__{idx:05d}__{ci:02d}" and rollout_item_id "_k{k:02d}",
# so the CONTEXT id must be '__'-free and <= 49 chars — MAX_ITEM_ID_LEN 53 - 4).
_DIR_SHORT = {
    "mapread_ctx": "mrc",
    "mapread_prefix": "mrp",
    "rb": "rb",
    "rawmeandiff": "rmd",
    "shuffled": "shf",
    "random": "rnd",
    "alpha0": "a0",
}
_POS_SHORT = {"context": "ctx", "answer": "ans"}
_JUDGE_CONTEXT_ID_MAX = 49  # MAX_ITEM_ID_LEN(53) - len("_kNN")(4)


def judge_context_id(cell: dict, seed, qi: int) -> str:
    """Deterministic Batch-custom_id-safe per-(cell, seed, question) context id.

    Single-'-' separators (never '__' — rollout_item_id raises on it, and
    judge_graded's own '__'-suffix decode requires it), short direction/position
    codes, dots -> 'p'. Deterministic from the cell fields, so consumers
    (build_answer_pools) can re-derive the id without a persisted map. The
    round-1 form (cell_id-prefixed + [:40] truncation) both raised at the first
    item (cell_id contains '__') and collapsed (seed, q) suffixes (F1)."""
    try:
        dshort = _DIR_SHORT[cell["direction"]]
        pshort = _POS_SHORT[cell["position"]]
    except KeyError as exc:  # fail loud on an unknown slug, never a silent alias
        raise ValueError(f"unknown direction/position for judge cid: {exc} in {cell}") from exc
    cstr = str(cell["c"]).replace(".", "p")
    cid = f"{cell['behavior']}-{dshort}-{pshort}-L{cell['layer']}-c{cstr}-s{seed}-q{int(qi):02d}"
    if "__" in cid or len(cid) > _JUDGE_CONTEXT_ID_MAX:
        raise ValueError(f"judge context id out of budget: {cid!r} (len={len(cid)})")
    return cid


def phase_judge_reduce(args) -> None:
    from explore_persona_space.experiments.issue_1739.judging import (
        judge_items_graded,
        judge_tallies,
        load_trait_rubric,
        rollout_item_id,
    )

    out_root = _out_root(args)
    phase = args.reduce_phase  # "localize" or "decisive"
    comp_root = out_root / phase / "raw_completions"
    files = sorted(comp_root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no completions to judge under {comp_root}")
    # Judge draws per reduce phase (plan §10: N=5 decisive / 3 localize) — F5.
    n_judge_draws = JUDGE_DRAWS_DECISIVE if phase == "decisive" else JUDGE_DRAWS_LOCALIZE
    _breadcrumb("judge_reduce", reduce_phase=phase, cells=len(files), judge_draws=n_judge_draws)
    _assert_eval_bank_disjoint(list(args.behaviors))  # A8 pre-wave gate (plan §6)
    cache_dir = out_root / phase / "judge_cache"
    save_raw = out_root / phase / "judge_raw"
    items_dir = out_root / phase / "judge_items"
    per_cell: dict[str, dict] = {}
    t0 = time.time()
    for fi, f in enumerate(files, 1):
        rows = json.loads(f.read_text())
        behavior = rows["cell"]["behavior"]
        rubric = load_trait_rubric(behavior)
        questions = _eval_questions(behavior)
        items: list[tuple[str, str, str]] = []
        q_of_item: dict[str, int] = {}
        for seed, sd in rows["seeds"].items():
            for qi, per_q in enumerate(sd["completions"]):
                cid = judge_context_id(rows["cell"], seed, qi)
                for di, text in enumerate(per_q):
                    item_id = rollout_item_id(cid, di)
                    q_of_item[item_id] = qi
                    # Real eval question into the rubric's {question} slot (the
                    # #779 trait rubrics score the answer IN CONTEXT of its
                    # question — judging.py). Question text is never logged.
                    items.append((item_id, questions[qi], text))
        result = judge_items_graded(
            items,
            rubric,
            cache_dir=cache_dir / rows["cell_id"],
            save_raw=save_raw / rows["cell_id"],
            n_draws=n_judge_draws,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        # Persist per-item tallies per cell (build_answer_pools + per-question
        # bootstrap re-reads; drop split intact — llm-judging rule 24).
        _write_json_atomic(
            items_dir / f"{rows['cell_id']}.json",
            {"cell": rows["cell"], "tallies": judge_tallies(result)},
        )
        if fi == 1:
            _judge_pilot_gate(result, rows["cell_id"])
        per_cell[rows["cell_id"]] = _per_cell_record(rows, result, q_of_item)
        _progress("judge_reduce", fi, len(files), rows["cell_id"], t0)

    reduced = _reduce_surface(per_cell, phase)
    payload = _run_metadata({"phase": phase, "per_cell": per_cell, "reduced": reduced})
    # Plan-named reduced-surface deliverable (§6.5/§9): localize ->
    # dose_response.json (the coherence-gated layer x dose surface), decisive ->
    # judged.json (code-review v2 Major).
    _write_json_atomic(out_root / phase / _REDUCED_SURFACE_NAME[phase], payload)
    if phase == "localize":
        _write_json_atomic(
            out_root / "localize" / "operating_points.json", reduced.get("operating_points", {})
        )
    if phase == "decisive":
        # Plan §6.5/§9-named deliverable (F4).
        _write_json_atomic(
            out_root / "decisive" / "delta_rate_percell.json",
            _run_metadata(
                {
                    "phase": phase,
                    "per_cell": per_cell,
                    "delta_rate": reduced["delta_rate"],
                    "null_band": reduced["null_band"],
                    "coherence_gate": reduced["coherence_gate"],
                    "bootstrap": reduced["bootstrap"],
                }
            ),
        )
    # Durable upload FIRST (plan §9 judge_reduce outputs: issue2220_readwrite/judge/),
    # sentinel + done breadcrumb LAST.
    _upload_judge_outputs(out_root, phase)
    _write_sentinel(out_root, f"judge_reduce_{phase}", "done")
    _breadcrumb("judge_reduce", status="done", reduce_phase=phase)


def _judge_pilot_gate(result, cell_id: str) -> None:
    """Pilot gate on the FIRST judged cell (plan §9 / llm-judging rules 23+26).

    The first cell's draws (localize: 30 items x 3 draws = 90; decisive: 200 x
    5 = 1000) run BEFORE the rest of the >=5k-call wave; gate on zero
    budget-truncated draws (``stop_reason == "max_tokens"`` — the rule-23
    signature that max_tokens=2048 is too small for the multi-field rubric)
    and content parse-fail < 2%. Fail loud so the instrument is fixed before
    the spend; the first cell's draws are rubric-keyed-cached, so a resumed
    full wave re-spends nothing."""
    trunc = int((result.stop_reason_tally or {}).get("max_tokens", 0))
    total = max(1, int(result.n_total_draws))
    drop_frac = float(result.n_dropped_draws) / total
    if trunc > 0 or drop_frac >= 0.02:
        raise RuntimeError(
            f"judge pilot gate FAILED on {cell_id}: {trunc} max_tokens-truncated draws, "
            f"content-drop fraction {drop_frac:.3f} (gate: 0 truncated, < 0.02) — fix the "
            "judge instrument (max_tokens / rubric) before the full wave (plan §9)"
        )
    logger.info("[judge_reduce] pilot gate PASS on %s (drop_frac=%.3f)", cell_id, drop_frac)


def _per_cell_record(rows: dict, result, q_of_item: dict) -> dict:
    """Reduce one cell's JudgeResult -> rate / per-question rates / coherence.

    ``result.scores`` maps item_id -> mean kept-draw score (None = all draws
    dropped; drop-never-coerce). rate = frac of KEPT items >= SCORE_THRESHOLD;
    per_question_rate is the question-cluster grain the paired bootstrap
    resamples; coherence_rate = frac of (seed, question) contexts passing
    steering.condition_passes (persisted at generation time)."""
    import numpy as np

    kept = {iid: float(s) for iid, s in result.scores.items() if s is not None}
    scores = np.asarray(list(kept.values()), dtype=float)
    by_q: dict[int, list[float]] = {}
    for iid, s in kept.items():
        by_q.setdefault(int(q_of_item[iid]), []).append(s)
    per_q_rate = {
        str(qi): float(np.mean(np.asarray(v) >= SCORE_THRESHOLD)) for qi, v in sorted(by_q.items())
    }
    coh: list[bool] = []
    for sd in rows["seeds"].values():
        coh.extend(bool(x) for x in sd["condition_passes"])
    return {
        "cell": rows["cell"],
        "mean_score": float(np.mean(scores)) if scores.size else None,
        "rate": float(np.mean(scores >= SCORE_THRESHOLD)) if scores.size else float("nan"),
        "per_question_rate": per_q_rate,
        "coherence_rate": float(np.mean(coh)) if coh else float("nan"),
        "n_items": int(len(result.scores)),
        "n_scored": int(scores.size),
        "n_dropped_items": int(len(result.scores) - scores.size),
        "cap_hit_fraction": rows.get("cap_hit_fraction"),
    }


def _reduce_surface(per_cell: dict, phase: str) -> dict:
    """Per-cell Delta-rate (+CI) + selection-symmetric null band + operating points.

    Registered definitions (plan §4.3/§4.4/§6):
      - Delta-rate = cell rate - the behavior's alpha=0 rate; the alpha=0
        reference is REQUIRED (fail-loud, never a silent 0.0 default).
      - CIs: question-level PAIRED CLUSTER bootstrap — one shared resample
        index draw per behavior; both arms (cell AND alpha0) recomputed within
        each resample (effective n = the eval questions).
      - Null band: EACH null direction takes the SAME argmax over its own
        coherence-gated (position, layer, dose) cells that the signal
        operating-point argmax takes, and every bootstrap draw re-runs that
        argmax (selection-inherited CI; selection-symmetric-nulls.md / #778).
        Band edge = max over null directions, 97.5th pct over draws.
      - Operating point per (direction, position): argmax Delta-rate over
        COHERENCE-PASSING cells only (plan §4.3 gate; F3) — nulls included, so
        decisive runs the nulls at their own argmax-selected points.
    """
    import numpy as np

    by_behavior: dict[str, list] = {}
    for cid, rec in per_cell.items():
        by_behavior.setdefault(rec["cell"]["behavior"], []).append((cid, rec))
    out: dict[str, dict] = {
        "delta_rate": {},
        "null_band": {},
        "operating_points": {},
        "coherence_gate": {"min_frac": COHERENCE_CELL_GATE, "gated_out": {}},
        "bootstrap": {
            "n_boot": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
            "unit": "question (paired cluster resample, shared draw per behavior; plan §6)",
        },
    }
    for behavior, recs in sorted(by_behavior.items()):
        a0 = [r for _, r in recs if r["cell"]["direction"] == "alpha0"]
        if not a0:
            raise RuntimeError(
                f"[{behavior}] no alpha0 reference cell in the {phase} judge set — the "
                "no-injection reference is required for every Delta-rate (plan §4.4); "
                "re-run the alpha0 cell instead of defaulting the reference to 0.0"
            )
        alpha0 = a0[0]
        q_keys = sorted(alpha0["per_question_rate"])
        r_a0 = np.asarray([alpha0["per_question_rate"][q] for q in q_keys], dtype=float)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        idx = rng.integers(0, len(q_keys), size=(N_BOOTSTRAP, len(q_keys)))
        a0_boot = np.nanmean(r_a0[idx], axis=1)  # (B,)
        delta: dict[str, dict] = {}
        boot_by_cid: dict[str, np.ndarray] = {}
        gated_out: list[str] = []
        for cid, rec in sorted(recs):
            if rec["cell"]["direction"] == "alpha0":
                continue
            rq = np.asarray([rec["per_question_rate"].get(q, np.nan) for q in q_keys], dtype=float)
            d_boot = np.nanmean(rq[idx], axis=1) - a0_boot
            lo, hi = np.nanpercentile(d_boot, [2.5, 97.5])
            passes = bool(rec["coherence_rate"] >= COHERENCE_CELL_GATE)
            delta[cid] = {
                "delta_rate": float(rec["rate"] - alpha0["rate"]),
                "ci95": [float(lo), float(hi)],
                "coherence_rate": float(rec["coherence_rate"]),
                "coherence_pass": passes,
            }
            boot_by_cid[cid] = d_boot
            if not passes:
                gated_out.append(cid)
        out["delta_rate"][behavior] = delta
        out["coherence_gate"]["gated_out"][behavior] = sorted(gated_out)

        # Selection-symmetric null band (F4).
        null_max_point: dict[str, float | None] = {}
        null_boot_rows: list[np.ndarray] = []
        n_gated_null = 0
        for nd in NULL_DIRECTIONS:
            nd_cids = [
                cid
                for cid in delta
                if per_cell[cid]["cell"]["direction"] == nd and delta[cid]["coherence_pass"]
            ]
            if not nd_cids:
                null_max_point[nd] = None
                continue
            n_gated_null += len(nd_cids)
            # Point max over FINITE deltas only: a coherence-passing cell whose
            # judge draws ALL dropped carries a NaN delta_rate, and a bare
            # Python max() propagates NaN order-dependently (code-review v2).
            finite = [
                delta[c]["delta_rate"] for c in nd_cids if np.isfinite(delta[c]["delta_rate"])
            ]
            null_max_point[nd] = float(max(finite)) if finite else None
            # nanmax: a NaN cell/draw never infects a healthy sibling's draw
            # (the percentile below already nan-drops all-NaN draws).
            null_boot_rows.append(np.nanmax(np.stack([boot_by_cid[c] for c in nd_cids]), axis=0))
        finite_maxes = [v for v in null_max_point.values() if v is not None]
        if null_boot_rows:
            band = np.nanmax(np.stack(null_boot_rows), axis=0)  # (B,) argmax per draw
            out["null_band"][behavior] = {
                "upper_edge_point": float(max(finite_maxes)) if finite_maxes else None,
                "upper_edge_boot97p5": float(np.nanpercentile(band, 97.5)),
                "per_null_max": null_max_point,
                "n_null_cells_gated": n_gated_null,
                "selection": (
                    "per-draw argmax over each null direction's coherence-gated "
                    "(position, layer, dose) cells, max over null directions — "
                    "matches the signal operating-point argmax "
                    "(selection-symmetric-nulls.md)"
                ),
            }
        else:
            out["null_band"][behavior] = {
                "upper_edge_point": None,
                "upper_edge_boot97p5": None,
                "per_null_max": null_max_point,
                "n_null_cells_gated": 0,
                "note": "no coherence-passing null cells",
            }

        if phase == "localize":
            ops: dict[str, dict] = {}
            no_coherent: list[str] = []
            for direction in DIRECTIONS:  # nulls INCLUDED (decisive runs them too)
                for position in POSITIONS:
                    cands = [
                        (cid, per_cell[cid]["cell"], delta[cid]["delta_rate"])
                        for cid in delta
                        if per_cell[cid]["cell"]["direction"] == direction
                        and per_cell[cid]["cell"]["position"] == position
                        and delta[cid]["coherence_pass"]
                    ]
                    if not cands:
                        no_coherent.append(f"{direction}__{position}")
                        continue
                    best = max(cands, key=lambda t: t[2] if np.isfinite(t[2]) else -1e9)
                    ops[f"{direction}__{position}"] = {
                        "layer": best[1]["layer"],
                        "c": best[1]["c"],
                        "delta_rate": float(best[2]),
                        "coherence_rate": delta[best[0]]["coherence_rate"],
                    }
            out["operating_points"][behavior] = ops
            if no_coherent:
                out["coherence_gate"].setdefault("no_coherent_operating_point", {})[behavior] = (
                    no_coherent
                )
    return out


def _pack_tree_to_jsonl_shards(
    src_dir: Path,
    dest_dir: Path,
    *,
    group: str,
    shard_bytes: int = 9_000_000,
    pattern: str = "*.json",
) -> int:
    """Pack a many-small-file JSON tree into <= 9 MB jsonl line-shards + manifest.

    upload-policy pack recipe (#1190/#1739): one row {"path": rel, "doc": doc}
    per file, so a ~65k-file judge_cache tree uploads as a dozen shards instead
    of a 65k-file commit (#1481). Idempotent (rewrites the pack). Returns the
    shard count. ``pattern`` is the rglob selector — trees whose files are
    EXTENSIONLESS single-doc JSON (e.g. #2254's bare-<cid> ``save_raw`` files)
    pass ``pattern="*"``; the default ``*.json`` packs ZERO rows there."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    cur_bytes = 0
    cur_lines: list[str] = []
    names: list[str] = []
    n_files = 0

    def _flush() -> None:
        nonlocal shard_idx, cur_bytes, cur_lines
        if not cur_lines:
            return
        p = dest_dir / f"{group}.shard{shard_idx:02d}.jsonl"
        p.write_text("\n".join(cur_lines) + "\n")
        names.append(p.name)
        shard_idx += 1
        cur_bytes = 0
        cur_lines = []

    for f in sorted(src_dir.rglob(pattern)):
        if not f.is_file():
            continue
        rel = str(f.relative_to(src_dir))
        line = json.dumps({"path": rel, "doc": json.loads(f.read_text())}, ensure_ascii=False)
        nb = len(line.encode()) + 1
        if cur_lines and cur_bytes + nb > shard_bytes:
            _flush()
        cur_lines.append(line)
        cur_bytes += nb
        n_files += 1
    _flush()
    _write_json_atomic(
        dest_dir / "pack_manifest.json", {"group": group, "n_files": n_files, "shards": names}
    )
    return shard_idx


def _upload_judge_outputs(out_root: Path, phase: str) -> None:
    """Persist judge outputs to the HF data repo (plan §9: issue2220_readwrite/judge/).

    One bulk upload_folder commit rooted at the phase dir: judge_raw/ (per-cell
    save_raw), judge_items/ (per-cell tallies), the phase's plan-named reduced
    surface (localize dose_response.json / decisive judged.json) — plus
    judge_cache/ PACKED into jsonl shards first (one {16-hex}.json per draw
    would be a ~65k file commit at production localize scale; per-directory
    file-count rule)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    phase_dir = out_root / phase
    cache_dir = phase_dir / "judge_cache"
    if cache_dir.is_dir():
        n_shards = _pack_tree_to_jsonl_shards(
            cache_dir, phase_dir / "judge_cache_pack", group="judge_cache"
        )
        logger.info("[upload] packed judge_cache into %d shards", n_shards)
    api = HfApi()
    allow = [
        "judge_raw/*",
        "judge_items/*",
        "judge_cache_pack/*",
        _REDUCED_SURFACE_NAME[phase],
        "delta_rate_percell.json",  # decisive-only §6.5 deliverable (absent on localize)
    ]
    hub.assert_hub_dir_filecounts(
        str(phase_dir), f"{_hf_prefix()}/judge/{phase}", allow_patterns=allow
    )
    hub.retry_transient(
        lambda: api.upload_folder(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(phase_dir),
            path_in_repo=f"{_hf_prefix()}/judge/{phase}",
            allow_patterns=allow,
        ),
        what=f"upload {phase} judge outputs",
    )
    logger.info("[upload] judge outputs -> %s/%s/judge/%s", HF_DATA_REPO, _hf_prefix(), phase)


# ---------------------------------------------------------------------------
# phase: build_answer_pools  (fixed judge-filtered +/- pools; margin producer)
# ---------------------------------------------------------------------------


def phase_build_answer_pools(args) -> None:
    """Build the fixed judge-filtered +/- teacher-forced answer pools (plan §4.4; F7).

    Consumes the SOURCE phase's persisted completions + the per-cell judge
    tallies phase_judge_reduce writes (judge_items/<cell>.json), re-derives each
    item's custom id deterministically (judge_context_id + rollout_item_id),
    and selects per behavior among COHERENT completions: pos = top POOL_SIZE by
    mean judge score with score >= SCORE_THRESHOLD, neg = bottom POOL_SIZE with
    score < SCORE_THRESHOLD, exact-text deduped. Pools are frozen once and held
    fixed across every margin context (llm-judging §E2 — no selection-on-outcome
    at margin time). Fail-loud below POOL_MIN per side."""
    from explore_persona_space.experiments.issue_1739.judging import rollout_item_id

    out_root = _out_root(args)
    src_phase = args.pools_source_phase
    comp_root = out_root / src_phase / "raw_completions"
    items_dir = out_root / src_phase / "judge_items"
    files = sorted(comp_root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no {src_phase} completions under {comp_root}")
    _breadcrumb("build_answer_pools", source=src_phase, cells=len(files))
    cands: dict[str, list[tuple[float, str, str]]] = {b: [] for b in args.behaviors}
    for f in files:
        rows = json.loads(f.read_text())
        behavior = rows["cell"]["behavior"]
        if behavior not in cands:
            continue
        tallies_p = items_dir / f"{rows['cell_id']}.json"
        if not tallies_p.exists():
            raise FileNotFoundError(
                f"judge tallies missing for {rows['cell_id']}: {tallies_p} "
                f"(run --phase judge_reduce --reduce-phase {src_phase} first)"
            )
        scores = json.loads(tallies_p.read_text())["tallies"]["scores"]
        for seed, sd in rows["seeds"].items():
            for qi, per_q in enumerate(sd["completions"]):
                cid = judge_context_id(rows["cell"], seed, qi)
                for di, text in enumerate(per_q):
                    if not sd["coherent_flags"][qi][di]:
                        continue
                    iid = rollout_item_id(cid, di)
                    s = scores.get(iid)
                    if s is None:  # all judge draws dropped — never coerced
                        continue
                    cands[behavior].append((float(s), text, iid))
    pools_dir = out_root / "margin" / "pools"
    t0 = time.time()
    for bi, behavior in enumerate(args.behaviors, 1):
        seen: dict[str, tuple[float, str, str]] = {}
        for s, text, iid in cands[behavior]:
            if text not in seen:  # exact-text dedup, first (deterministic order) wins
                seen[text] = (s, text, iid)
        uniq = sorted(seen.values(), key=lambda t: (t[0], t[2]))  # score asc, id tiebreak
        neg = [t for t in uniq if t[0] < SCORE_THRESHOLD][:POOL_SIZE]
        pos = [t for t in uniq if t[0] >= SCORE_THRESHOLD][-POOL_SIZE:]
        if len(pos) < POOL_MIN or len(neg) < POOL_MIN:
            raise RuntimeError(
                f"[{behavior}] answer-pool yield below floor: {len(pos)} pos / {len(neg)} neg "
                f"(need >= {POOL_MIN} each; from {len(uniq)} unique coherent judged "
                f"completions). Widen the {src_phase} source before margin (plan §4.4)."
            )
        if len(pos) < POOL_SIZE or len(neg) < POOL_SIZE:
            logger.warning(
                "[pools] %s under target size: %d pos / %d neg (target %d)",
                behavior,
                len(pos),
                len(neg),
                POOL_SIZE,
            )
        payload = _run_metadata(
            {
                "pos": [t[1] for t in pos],
                "neg": [t[1] for t in neg],
                "pool_meta": {
                    "source_phase": src_phase,
                    "n_candidates_unique_coherent": len(uniq),
                    "pos_item_ids": [t[2] for t in pos],
                    "neg_item_ids": [t[2] for t in neg],
                    "pos_scores": [t[0] for t in pos],
                    "neg_scores": [t[0] for t in neg],
                    "threshold": SCORE_THRESHOLD,
                },
            }
        )
        _write_json_atomic(pools_dir / f"{behavior}.json", payload)
        _progress("build_answer_pools", bi, len(args.behaviors), behavior, t0)
    _upload_pools(pools_dir)
    _write_sentinel(out_root, "build_answer_pools", "done")
    _breadcrumb("build_answer_pools", status="done")


def _upload_pools(pools_dir: Path) -> None:
    """Persist the frozen +/- pools (small JSONs; margin's pod-side input)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    allow = ["*.json"]
    hub.assert_hub_dir_filecounts(
        str(pools_dir), f"{_hf_prefix()}/margin/pools", allow_patterns=allow
    )
    hub.retry_transient(
        lambda: api.upload_folder(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(pools_dir),
            path_in_repo=f"{_hf_prefix()}/margin/pools",
            allow_patterns=allow,
        ),
        what="upload margin answer pools",
    )
    logger.info("[upload] pools -> %s/%s/margin/pools", HF_DATA_REPO, _hf_prefix())


# ---------------------------------------------------------------------------
# argparse dispatch
# ---------------------------------------------------------------------------

PHASES = {
    "materialize_directions": phase_materialize_directions,
    "norm_probe": phase_norm_probe,
    "check_disjoint": phase_check_disjoint,
    "localize": phase_localize,
    "decisive": phase_decisive,
    "build_answer_pools": phase_build_answer_pools,
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
    ap.add_argument(
        "--pools-source-phase",
        choices=("localize", "decisive"),
        default="localize",
        help="which judged phase feeds the fixed +/- answer pools (plan §4.4)",
    )
    ap.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="round-robin cell shard for the multi-GPU fan-out (plan §9)",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="total shards; launcher pins CUDA_VISIBLE_DEVICES per shard",
    )
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
    overwrites committed artifacts, and uploads divert to the smoke/ sub-prefix
    so smoke cell files (production-identical names) never overwrite canonical
    HF artifacts."""
    global DOSES_NONZERO, _SMOKE_UPLOAD_SUBPREFIX
    args.behaviors = args.behaviors[:1]
    args.layers = args.layers[:1]
    args.q1 = 2
    args.q2 = 2
    args.draws_localize = 2
    args.draws_decisive = 2
    if args.out_root == "eval_results/issue_2220":
        args.out_root = "/tmp/issue-2220-smoke"
    DOSES_NONZERO = (1.0,)
    _SMOKE_UPLOAD_SUBPREFIX = True


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

        for sym in (
            "stream_members",
            "load_labels",
            "load_row_index",
            "_summary_re",
            # #2220 throughput fix: a stale natpv (pre-fix checkout) must fail
            # the dry-run, not silently fall back to the ~1 MB/s range-GET path.
            "MATERIALIZE_TARS",
            "_materialized_members",
            "_download_tar",
        ):
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
    elif phase == "check_disjoint":
        # import-resolution only (the corpus fetch is a network side effect a
        # local dry-run must not incur); the guard body is CPU-test-covered.
        from huggingface_hub import hf_hub_download  # noqa: F401

        assert callable(_assert_eval_bank_disjoint)
        _breadcrumb("check_disjoint", dry_run=1, behaviors=len(args.behaviors))
    elif phase == "judge_reduce":
        # import-resolution only; do NOT call load_trait_rubric (asset-gen chain)
        from explore_persona_space.eval.judge_dispatch import validate_batch_custom_ids
        from explore_persona_space.experiments.issue_1739.judging import (  # noqa: F401
            judge_items_graded,
            judge_tallies,
            load_trait_rubric,
            rollout_item_id,
        )

        for fn in (judge_items_graded, judge_tallies, load_trait_rubric, rollout_item_id):
            assert callable(fn)
        # Zero-API-cost cid probe (F1): compose worst-case PRODUCTION cell ids
        # through the REAL rollout_item_id + validate_batch_custom_ids, with the
        # judge_graded "__{idx:05d}__{ci:02d}" suffix budget emulated.
        ids: list[str] = []
        for behavior in BEHAVIORS:
            for direction in (*DIRECTIONS, "alpha0"):
                for position in POSITIONS:
                    cell = {
                        "behavior": behavior,
                        "direction": direction,
                        "position": position,
                        "layer": max(LAYERS),
                        "c": 0.5,
                    }
                    for seed in SEEDS_DECISIVE:
                        cid = judge_context_id(cell, seed, Q2_DECISIVE - 1)
                        ids.append(rollout_item_id(cid, DRAWS_DECISIVE - 1))
        assert len(ids) == len(set(ids)), "judge context ids must be unique"
        validate_batch_custom_ids(f"{i}__{0:05d}__{0:02d}" for i in ids)
        _breadcrumb("judge_reduce", dry_run=1, reduce_phase=args.reduce_phase, cid_probe=len(ids))
    elif phase == "build_answer_pools":
        from explore_persona_space.experiments.issue_1739.judging import rollout_item_id

        cell = {
            "behavior": "hallucination",
            "direction": "mapread_prefix",
            "position": "context",
            "layer": max(LAYERS),
            "c": 0.5,
        }
        iid = rollout_item_id(judge_context_id(cell, SEEDS_DECISIVE[-1], Q2_DECISIVE - 1), 4)
        assert len(iid) <= 53, iid
        _breadcrumb("build_answer_pools", dry_run=1, source=args.pools_source_phase)
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
