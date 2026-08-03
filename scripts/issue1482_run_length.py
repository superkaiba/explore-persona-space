"""Issue #1482 — PER-TOKEN SAE capture: run length, template fraction, token-weighted side.

WHY THIS EXISTS
    The banked #1482 pooled store keeps only per-ROW reductions (``ans_mean`` /
    ``ans_max`` / ``ans_frac`` on the answer side, ``psi_mean`` on the context
    side). Four wanted per-feature covariates need per-TOKEN codes, which were
    pooled away at the original capture, so they need a RE-ENCODE — not a
    re-analysis. ``issue1482_continuous_predictors.py`` already declares two of
    them as SLOTS (``mean_run_length``, ``template_token_frac``) joined on
    ``feat_ids`` and picks them up automatically once this file lands.

WHAT ONE FORWARD PASS PRODUCES (per SAE feature f, full 131,072 dictionary)
    1. ``mean_run_length[f]`` — mean length of a contiguous run of tokens on
       which f is active, ANSWER side. Emitted alongside ``persistence_p[f]``
       = P(active at t+1 | active at t). E[R] = 1/(1-p) EXACTLY for a
       stationary process, so the two are the same quantity in different
       units; the consumer plots only ``mean_run_length``.
    2. ``template_token_frac[f]`` — fraction of f's firings landing on
       chat-template SCAFFOLD tokens. BEHAVIOURAL. This is NOT ``scaffold_frac``,
       which is GEOMETRIC (decoder-vector mass in the prefix-covariance top-48
       eigen-subspace, #1773 L475-486) and says nothing about which tokens a
       feature fires on. The mask definition is TEMPLATE_MASK_DEF below, copied
       verbatim into the sidecar meta.
    3. ``ctx_tokens_active[f]`` / ``ans_tokens_active[f]`` — per-token firing
       COUNTS per side, giving a TOKEN-WEIGHTED side ratio. The banked
       ``side_ratio`` is ROW OCCUPANCY (active anywhere in the span = 1
       regardless of how many tokens fire), so this is a different
       measurement, not a refinement of it.
    4. ``act_var_across_tokens[f]`` — within-answer, across-token variance of
       f's activation VALUE, averaged over answers where f fires.

TOKEN POOL — matches the banked store exactly
    Every read here rides ``issue1482_error_analysis._row_features``'s
    reference token-pool semantics: the first ``S.BOS_OFFSET`` positions and
    >10x-median-norm token rows are excluded (the SAE's own training + published
    eval exclude them; their codes are off-distribution). Runs and adjacency are
    therefore computed over the KEPT pool but keyed on ORIGINAL token positions:
    two kept tokens are adjacent iff their ORIGINAL positions differ by 1, so a
    masked-out token BREAKS a run rather than silently splicing one.

SAMPLE (``--sample-mode full``, the production default)
    A provenance-stratified draw over the ENTIRE 120,000-row ``sae_fit`` pool —
    the same pool + stratifier the pooled store used — then every raw chunk is
    streamed with delete-after, keeping only the drawn rows.

    The cheaper two-stage chunk-cluster alternative (``--sample-mode
    chunk-subset``, retained for wiring smokes only) is MEASURABLY BIASED at
    small chunk counts: span length is heavy-tailed (context median 56, p95 766)
    and SHARD identity carries the tail, so a 6-chunk / 6-shard pool read answer
    mean 379.2 and context mean 127.1 against the full-corpus 432.9 / 189.0.
    Covering all 32 shards closed it (64 chunks, 4,141 candidates: answer mean
    426.0 vs 432.9, median 387.0 vs 391). Rather than tune the chunk count, the
    production path buys the unbiased full-pool draw outright (~10 min of
    ~0.3 s/chunk downloads).

    The three reproduction gates are what TEST representativeness; they are
    always computed and reported, and a >10% drift is FLAGGED, never suppressed.

Usage (pod):
    uv run python scripts/issue1482_run_length.py --phase stage
    uv run python scripts/issue1482_run_length.py --phase capture --pilot-rows 24
    uv run python scripts/issue1482_run_length.py --phase capture
    uv run python scripts/issue1482_run_length.py --phase upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy / torch (shared-VM discipline)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue1482_error_analysis as EA  # noqa: E402
import issue1482_sae as S  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logger = logging.getLogger("i1482-runlen")

DICT_SIZE = S.DICT_SIZE
LAYER = 19
K = 64
SEED = 1482
HF_DATA_REPO = C.HF_DATA_REPO
SCRATCH_META_PREFIX = "issue1482_error_analysis/analysis_tensors/scratch_meta"
HF_PREFIX = "issue1482_error_analysis/analysis_tensors/run_length"

# Parent pool shas (issue1482_early_layer.EARLY_COMMITTED_SPLIT_SHAS) — the same
# committed literals every #1482 leg asserts, so a drifted split fails loud.
COMMITTED_SPLIT_SHAS = {
    "sae_fit_sha256": "88d344675fbbca3a717cd8a0c6aa4fd893241a17098bd3173f6ae00b4d9a0fb8",
    "holdout_sha256": "7957d689748eca218055f213082c1df444603ec2f1faa3f04b4004cee6f58622",
}

# Verified FULL-corpus reference reads the gates compare against.
# Row-occupancy census over 120,000 fit rows: issue1482_side_specificity_
# fullwidth.CENSUS_EXPECTED (cross-checked at runtime in _gate_census).
CENSUS_EXPECTED = {
    "n_fit": 120000,
    "answer_active": 128512,
    "context_active": 128002,
    "context_only": 1654,
    "answer_only": 2164,
    "live": 130166,
}
FULL_OCCUPANCY_RATIO = 2.075  # sum(cnt_fit) / sum(psi_cnt_fit), full-corpus scan
FULL_SPAN_RATIO = 2.291  # answer mean tokens (432.9) / context mean tokens (189.0)
FULL_TOKEN_NULL = 0.696  # FULL_SPAN_RATIO / (1 + FULL_SPAN_RATIO)
GATE_DRIFT_WARN = 0.10  # |relative drift| above this is FLAGGED (never silently passed)

TEMPLATE_MASK_DEF = (
    "A token position is TEMPLATE (chat-template scaffold) iff ANY of: "
    "(1) position <= prefix_end, the constant chat-template prefix — everything "
    "before the user query text, boundary from the offset-mapping "
    "exclude-straddler policy in issue1482_error_analysis._tokenize_row "
    "(prefix_end = last token ending entirely inside the constant prefix chars); "
    "(2) context_end-2 <= position <= context_end, the 3-token generation suffix "
    "'<|im_start|>assistant\\n' (issue779_common.GENERATION_SUFFIX, asserted "
    "token-exact by _tokenize_row); "
    "(3) token id in {<|im_start|>, <|im_end|>} anywhere in the sequence. "
    "Everything else is CONTENT. The role WORDS 'system'/'user'/'assistant' are "
    "deliberately NOT id-matched: they occur in ordinary user/assistant content, "
    "and the template's own role words are already covered positionally by (1) "
    "and (2). Answer-side tokens come from tok(response, add_special_tokens=False) "
    "so the answer span carries no template scaffold by construction except via "
    "rule (3). template_token_frac[f] = (template firings) / (ALL firings, "
    "context + answer), token-weighted."
)


def _log(msg: str) -> None:
    logger.info(msg)
    print(f"[runlen] {msg}", flush=True)


def _sha_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _git_sha() -> str:
    """Commit sha for the reproducibility record.

    Prefers the ``EPS_GIT_SHA`` env pin: rsync-staged / git-less scratch trees
    (the fellows + SLURM lanes, and any pod staged without a usable .git) have
    no repo to query, and a strict shellout there kills the workload — degrade,
    never crash (gotchas.md, #1902).
    """
    import os
    import subprocess

    pinned = os.environ.get("EPS_GIT_SHA", "").strip()
    if pinned:
        return pinned
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception as exc:  # degrade, never crash a capture on missing git
        return f"unknown ({type(exc).__name__})"


# ── staging ───────────────────────────────────────────────────────────────────


def _stage_scratch_meta(args) -> None:
    """Stage split_indices.npz / row_ci.npy / prov.npy from the parent's HF
    scratch_meta (idempotent; HF-permanent — the off-pod incident-class fix)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    args.scratch.mkdir(parents=True, exist_ok=True)
    for name in ("split_indices.npz", "row_ci.npy", "prov.npy"):
        dest = args.scratch / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        hub.retry_transient(
            lambda n=name: hf_hub_download(
                HF_DATA_REPO,
                filename=f"{SCRATCH_META_PREFIX}/{n}",
                repo_type="dataset",
                local_dir=str(args.scratch / "_hf"),
            ),
            what=f"scratch_meta fetch ({name})",
        )
        shutil.copy2(args.scratch / "_hf" / SCRATCH_META_PREFIX / name, dest)
    _log(f"scratch_meta ready under {args.scratch}")


def phase_stage(args) -> None:
    _stage_scratch_meta(args)
    S.BatchTopKSAE.ensure_downloaded(K, args.sae_dir, layer=LAYER)
    _log(f"SAE k={K} layer={LAYER} pre-staged under {args.sae_dir}")


# ── sample ────────────────────────────────────────────────────────────────────


def _load_split_and_assert(args) -> dict[str, np.ndarray]:
    idx = np.load(args.scratch / "split_indices.npz")
    pools = {k: np.asarray(idx[k], dtype=np.int64) for k in ("sae_fit", "sae_val", "holdout")}
    got_fit, got_hold = _sha_ids(pools["sae_fit"]), _sha_ids(pools["holdout"])
    assert got_fit == COMMITTED_SPLIT_SHAS["sae_fit_sha256"], (
        f"sae_fit sha drift: {got_fit} != committed literal"
    )
    assert got_hold == COMMITTED_SPLIT_SHAS["holdout_sha256"], (
        f"holdout sha drift: {got_hold} != committed literal"
    )
    assert len(pools["sae_fit"]) == CENSUS_EXPECTED["n_fit"], (
        f"sae_fit size {len(pools['sae_fit'])} != census n_fit {CENSUS_EXPECTED['n_fit']}"
    )
    _log("parent pool shas match committed split_1482.json literals")
    return pools


def _chunk_subset(names: list[str], n_chunks: int, rng) -> list[str]:
    """Shard-stratified seeded chunk subset: round-robin over shards so the
    draw spans all 32 shards instead of the first-N prefix ``_raw_chunk_names``
    returns under ``--max-chunks`` (which is shard00-only, i.e. biased)."""
    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(n.split("_", 1)[0], []).append(n)
    for shard in by_shard:
        by_shard[shard] = list(rng.permutation(by_shard[shard]))
    shards = sorted(by_shard)
    out: list[str] = []
    depth = 0
    while len(out) < n_chunks and any(len(by_shard[s]) > depth for s in shards):
        for s in shards:
            if len(out) >= n_chunks:
                break
            if len(by_shard[s]) > depth:
                out.append(by_shard[s][depth])
        depth += 1
    return sorted(out)


def _all_chunk_names() -> list[str]:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    files = hub.list_hf_files_under_path(HfApi(), HF_DATA_REPO, EA.RAW_PREFIX, repo_type="dataset")
    chunk_re = re.compile(r"^shard\d+_chunk\d+\.json$")
    names = sorted(n for n in (q.rsplit("/", 1)[-1] for q in files) if chunk_re.match(n))
    assert names, "no raw chunk files enumerated"
    return names


def _sample_full_pool(
    args, fit_pool: np.ndarray, prov_u8: np.ndarray, row_ci: np.ndarray, fingerprint: str
):
    """PRODUCTION sampler: a provenance-stratified draw over the ENTIRE sae_fit
    pool (no chunk clustering), then stream every raw chunk keeping only the
    drawn rows.

    A two-stage chunk-cluster draw is measurably biased at small chunk counts:
    a 6-chunk pool (6 of 32 shards) read answer mean 379.2 / context mean 127.1
    against the full-corpus 432.9 / 189.0, because span length is heavy-tailed
    (context median 56, p95 766) and SHARD identity carries the tail. Covering
    all 32 shards closed it (64 chunks, 4,141 candidates: answer mean 426.0,
    median 387.0). Chunks download at ~0.3 s each, so the unbiased full-pool
    draw costs ~10 min for 1,920 chunks — bought rather than approximated.
    """
    rng = np.random.default_rng(SEED)
    frac = float((prov_u8[fit_pool] == 0).mean())
    picked, strat_meta = EA._stratified_sample(rng, fit_pool, prov_u8, args.n_rows, frac)
    ids = np.asarray(sorted(int(r) for r in picked), dtype=np.int64)
    needed_ci = {int(row_ci[r]): int(r) for r in ids}
    assert len(needed_ci) == len(ids), "ci collision in the drawn rows"

    base_meta = {
        "sampler": "full-pool (provenance-stratified over the entire sae_fit pool)",
        "seed": SEED,
        "n_rows_drawn": int(len(ids)),
        "sampled_row_ids_sha256": _sha_ids(ids),
        "fit_pool_size": int(len(fit_pool)),
        "lmsys_frac_pool": frac,
        "stratified_sample": strat_meta,
    }
    cache = args.scratch / f"rows_{fingerprint}.jsonl"
    if args.resume:
        cached = _load_row_cache(cache, fingerprint)
        if cached is not None:
            base_meta |= {
                "n_rows_resolved": len(cached),
                "chunk_stage": {"note": f"served from row cache {cache.name} (fetch skipped)"},
            }
            return cached, base_meta

    names = _all_chunk_names()
    dns = argparse.Namespace(max_chunks=0, scratch=args.scratch)
    rows: list[tuple[int, int, str, str]] = []
    n_chunks_hit = 0
    t0 = time.time()
    for name, keep in EA._iter_needed_rows(dns, names, needed_ci):
        rows.extend(keep)
        n_chunks_hit += 1
        if n_chunks_hit % 100 == 0:
            _log(f"  {len(rows)}/{len(ids)} rows fetched ({n_chunks_hit} chunks hit)")
    _log(
        f"full-pool fetch: {len(rows)}/{len(ids)} rows from {n_chunks_hit} of "
        f"{len(names)} chunks in {time.time() - t0:.0f}s"
    )
    assert len(rows) >= 0.98 * len(ids), (
        f"only {len(rows)} of {len(ids)} drawn rows were text-resolvable"
    )
    if args.resume:
        _write_row_cache(cache, fingerprint, rows)
    meta = base_meta | {
        "n_rows_resolved": len(rows),
        "chunk_stage": {
            "n_chunks_enumerated": len(names),
            "n_chunks_hit": n_chunks_hit,
            "note": "every chunk is streamed with delete-after; only drawn rows are kept",
        },
    }
    return rows, meta


def _collect_candidates(args, fit_pool: np.ndarray, row_ci: np.ndarray) -> tuple[list, dict]:
    """SMOKE sampler (``--sample-mode chunk-subset``): download a seeded
    shard-stratified chunk subset once and keep every sae_fit row in it.

    MEASURABLY BIASED at small chunk counts — see ``_sample_full_pool``. Kept
    only as the fast path for wiring smokes, never for the production artifact.

    Row TEXT is held in memory and never printed (real-corpus hygiene). Chunks
    are deleted immediately after parsing (the parent's bounded-retry +
    delete-after convention)."""
    names = _all_chunk_names()
    rng = np.random.default_rng(SEED)
    picked = _chunk_subset(names, args.n_chunks, rng)
    _log(f"chunk subset: {len(picked)} of {len(names)} chunks (shard-stratified, seed {SEED})")

    ci2row = {int(row_ci[r]): int(r) for r in fit_pool}
    cache = args.scratch / "raw_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cands: list[tuple[int, int, str, str]] = []
    n_rows_seen = 0
    for i, name in enumerate(picked):
        got = Path(N1M._download_chunk_with_retry(HF_DATA_REPO, f"{EA.RAW_PREFIX}/{name}", cache))
        rows = json.loads(got.read_text())["rows"]
        got.unlink()
        n_rows_seen += len(rows)
        for r in rows:
            ci = int(r["ci"])
            row = ci2row.get(ci)
            if row is not None:
                cands.append((row, ci, r["prompt"], r["response"]))
        if (i + 1) % 16 == 0 or i + 1 == len(picked):
            _log(f"  chunks {i + 1}/{len(picked)} — {len(cands)} fit-row candidates")
    meta = {
        "n_chunks_enumerated": len(names),
        "n_chunks_downloaded": len(picked),
        "chunks": picked,
        "n_rows_scanned": n_rows_seen,
        "n_fit_candidates": len(cands),
    }
    return cands, meta


def _sample_rows(args, cands: list, prov_u8: np.ndarray) -> tuple[list, dict]:
    """Provenance-stratified draw of ``n_rows`` from the candidate list, using
    the same ``EA._stratified_sample`` stratifier the pooled store's own
    subsample used (realized LMSYS fraction of the candidate pool)."""
    cand_rows = np.asarray([c[0] for c in cands], dtype=np.int64)
    assert len(cand_rows) >= args.n_rows, (
        f"only {len(cand_rows)} fit candidates in {args.n_chunks} chunks for "
        f"n_rows={args.n_rows}; raise --n-chunks"
    )
    frac = float((prov_u8[cand_rows] == 0).mean())
    rng = np.random.default_rng(SEED)
    picked, strat_meta = EA._stratified_sample(rng, cand_rows, prov_u8, args.n_rows, frac)
    picked_set = {int(r) for r in picked}
    by_row = {c[0]: c for c in cands}
    rows = [by_row[r] for r in sorted(picked_set)]
    assert len(rows) == len(picked_set), "candidate/row map lost rows"
    ids = np.asarray(sorted(picked_set), dtype=np.int64)
    meta = {
        "seed": SEED,
        "n_rows": len(rows),
        "sampled_row_ids_sha256": _sha_ids(ids),
        "lmsys_frac_candidates": frac,
        "stratified_sample": strat_meta,
    }
    return rows, meta


# ── per-token accumulation ────────────────────────────────────────────────────


class Accum:
    """Per-feature streaming accumulators. Nothing (tokens x 131,072) is ever
    materialised beyond a single row's code block, which is sparsified on the
    spot and discarded."""

    def __init__(self, device: str) -> None:
        z64 = lambda: torch.zeros(DICT_SIZE, dtype=torch.float64, device=device)  # noqa: E731
        self.device = device
        self.ans_tok = z64()  # answer-side firings (token-weighted)
        self.ctx_tok = z64()  # context-side firings (token-weighted)
        self.tmpl_tok = z64()  # firings on template tokens (both sides)
        self.pairs = z64()  # adjacent-in-ORIGINAL-position active pairs (answer)
        self.denom = z64()  # active answer positions whose +1 neighbour is in the pool
        self.var_sum = z64()  # sum over answers of within-answer across-token variance
        self.var_rows = z64()  # answers contributing to var_sum
        self.cnt = z64()  # ROW occupancy, answer side (banked-store convention)
        self.psi_cnt = z64()  # ROW occupancy, context side
        self.n_rows = 0
        self.ctx_tokens_raw = 0
        self.ans_tokens_raw = 0
        self.ctx_tokens_kept = 0
        self.ans_tokens_kept = 0
        self.ans_all_out_rows = 0

    _ARRAYS = (
        "ans_tok",
        "ctx_tok",
        "tmpl_tok",
        "pairs",
        "denom",
        "var_sum",
        "var_rows",
        "cnt",
        "psi_cnt",
    )
    _SCALARS = (
        "n_rows",
        "ctx_tokens_raw",
        "ans_tokens_raw",
        "ctx_tokens_kept",
        "ans_tokens_kept",
        "ans_all_out_rows",
    )

    def state(self, processed: np.ndarray, fingerprint: str) -> dict:
        """Checkpoint payload: every accumulator + the rows already folded in."""
        d = {k: getattr(self, k).cpu().numpy() for k in self._ARRAYS}
        d |= {k: np.asarray(getattr(self, k), dtype=np.int64) for k in self._SCALARS}
        d["processed_row_ids"] = np.asarray(processed, dtype=np.int64)
        d["fingerprint"] = np.asarray(fingerprint)
        return d

    def restore(self, z) -> np.ndarray:
        """Load a checkpoint in place; returns the processed row ids to skip."""
        for k in self._ARRAYS:
            getattr(self, k).copy_(torch.as_tensor(z[k], device=self.device))
        for k in self._SCALARS:
            setattr(self, k, int(z[k]))
        return np.asarray(z["processed_row_ids"], dtype=np.int64)

    def add_side(self, f: torch.Tensor, pos: torch.Tensor, tmpl: torch.Tensor, side: str) -> None:
        """Accumulate one row's one side.

        ``f`` (T_kept, D) codes; ``pos`` (T_kept,) ORIGINAL token positions;
        ``tmpl`` (T_kept,) bool template mask aligned to ``pos``.
        """
        t_idx, f_idx = f.nonzero(as_tuple=True)
        if f_idx.numel() == 0:
            return
        bc = torch.bincount(f_idx, minlength=DICT_SIZE).to(torch.float64)
        live = bc > 0
        self.tmpl_tok += torch.bincount(f_idx[tmpl[t_idx]], minlength=DICT_SIZE).to(torch.float64)
        if side == "ctx":
            self.ctx_tok += bc
            self.psi_cnt += live.to(torch.float64)
            return

        self.ans_tok += bc
        self.cnt += live.to(torch.float64)

        # (4) within-answer across-token variance of the activation VALUE, over
        # ALL kept answer tokens (zeros included — it is the activation TRACE's
        # variance, not a conditional-on-firing variance).
        n_tok = f.shape[0]
        v = f[t_idx, f_idx].to(torch.float64)
        s = torch.zeros(DICT_SIZE, dtype=torch.float64, device=self.device)
        s.scatter_add_(0, f_idx, v)
        ssq = torch.zeros(DICT_SIZE, dtype=torch.float64, device=self.device)
        ssq.scatter_add_(0, f_idx, v * v)
        var_row = torch.clamp(ssq / n_tok - (s / n_tok) ** 2, min=0.0)
        self.var_sum += var_row * live
        self.var_rows += live.to(torch.float64)

        # (1) runs + persistence. Sort the COO by (feature, token) and test
        # ORIGINAL-position adjacency, so a masked-out token BREAKS a run.
        order = torch.argsort(f_idx.to(torch.int64) * n_tok + t_idx.to(torch.int64))
        fs = f_idx[order]
        ps = pos[t_idx[order]]  # COO entry -> token index -> ORIGINAL position
        same_f = fs[1:] == fs[:-1]
        adjacent = (ps[1:] - ps[:-1]) == 1
        both = same_f & adjacent
        if bool(both.any()):
            self.pairs += torch.bincount(fs[1:][both], minlength=DICT_SIZE).to(torch.float64)
        # denominator of p: active positions whose ORIGINAL successor is in the pool
        pool = torch.zeros(int(pos[-1].item()) + 2, dtype=torch.bool, device=self.device)
        pool[pos] = True
        has_next = pool[pos + 1]
        sel = has_next[t_idx]
        if bool(sel.any()):
            self.denom += torch.bincount(f_idx[sel], minlength=DICT_SIZE).to(torch.float64)


@torch.no_grad()
def _row_pass(sae, acc: Accum, h: torch.Tensor, ids: np.ndarray, tk: tuple, device: str) -> None:
    """One row: reference token pool -> per-token encode -> streaming reduce."""
    _full_ids, prefix_end, context_end, _n_ans, _seam = tk
    T = h.shape[0]
    keep = S.token_inlier_mask(h)
    keep[: min(S.BOS_OFFSET, keep.shape[0])] = False
    ctx_keep, ans_keep = keep[: context_end + 1], keep[context_end + 1 :]
    if int(ctx_keep.sum()) == 0:
        raise ValueError(f"context pool empty after reference masking (context_end={context_end})")

    tmpl_all = _template_mask(ids, prefix_end, context_end, device)
    all_pos = torch.arange(T, device=device)

    ctx_pos = all_pos[: context_end + 1][ctx_keep]
    f_ctx = sae.encode(h[: context_end + 1][ctx_keep])
    acc.add_side(f_ctx, ctx_pos, tmpl_all[ctx_pos], "ctx")
    del f_ctx

    h_ans = h[context_end + 1 :]
    ans_all_out = int(h_ans.shape[0] > 0 and int(ans_keep.sum()) == 0)
    acc.ans_all_out_rows += ans_all_out
    sel = torch.ones_like(ans_keep) if ans_all_out else ans_keep
    ans_pos = all_pos[context_end + 1 :][sel]
    if ans_pos.numel():
        f_ans = sae.encode(h_ans[sel])
        acc.add_side(f_ans, ans_pos, tmpl_all[ans_pos], "ans")
        del f_ans

    acc.n_rows += 1
    acc.ctx_tokens_raw += context_end + 1
    acc.ans_tokens_raw += T - (context_end + 1)
    acc.ctx_tokens_kept += int(ctx_keep.sum())
    acc.ans_tokens_kept += int(ans_pos.numel())


def _template_mask(ids: np.ndarray, prefix_end: int, context_end: int, device: str):
    """TEMPLATE_MASK_DEF, as a bool tensor over ORIGINAL token positions."""
    m = torch.zeros(len(ids), dtype=torch.bool, device=device)
    m[: prefix_end + 1] = True  # (1) constant chat-template prefix
    m[max(0, context_end - 2) : context_end + 1] = True  # (2) generation suffix
    special = torch.as_tensor(_SPECIAL_IDS, device=device)
    tid = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=device)
    m |= torch.isin(tid, special)  # (3) <|im_start|> / <|im_end|> anywhere
    return m


_SPECIAL_IDS: list[int] = []


def _init_special_ids(tok) -> None:
    """<|im_start|> / <|im_end|> only — see TEMPLATE_MASK_DEF on why role WORDS
    are deliberately excluded from the id-based arm."""
    ids = []
    for t in ("<|im_start|>", "<|im_end|>"):
        enc = tok.encode(t, add_special_tokens=False)
        assert len(enc) == 1, (t, enc)
        ids.append(int(enc[0]))
    _SPECIAL_IDS.clear()
    _SPECIAL_IDS.extend(sorted(ids))


# ── gates ─────────────────────────────────────────────────────────────────────


def _drift(observed: float, expected: float) -> dict:
    rel = (observed - expected) / expected if expected else float("nan")
    return {
        "observed": float(observed),
        "expected_full_corpus": float(expected),
        "rel_drift": float(rel),
        "flag": "DRIFT" if abs(rel) > GATE_DRIFT_WARN else "ok",
    }


def _gate_census(acc: Accum) -> dict:
    """Gate 1 — row-occupancy census + occupancy ratio vs the verified full-corpus
    scan. A 2,000-row subsample cannot match the 120,000-row FEATURE counts (a
    feature needs a row to be seen at all), so the binding comparison is the
    RATIO; the counts are reported for context."""
    try:
        import issue1482_side_specificity_fullwidth as SS

        assert SS.CENSUS_EXPECTED == CENSUS_EXPECTED, (
            f"census literal drift vs issue1482_side_specificity_fullwidth: "
            f"{SS.CENSUS_EXPECTED} != {CENSUS_EXPECTED}"
        )
        cross = "asserted equal to issue1482_side_specificity_fullwidth.CENSUS_EXPECTED"
    except ImportError as exc:
        cross = f"cross-check skipped ({type(exc).__name__}: {exc})"

    cnt = acc.cnt.cpu().numpy()
    psi = acc.psi_cnt.cpu().numpy()
    live = (cnt + psi) > 0
    ratio = float(cnt.sum() / psi.sum()) if psi.sum() else float("nan")
    return {
        "definition": "row occupancy — feature active ANYWHERE in the span counts 1",
        "n_rows": acc.n_rows,
        "subsample_census": {
            "answer_active": int((cnt > 0).sum()),
            "context_active": int((psi > 0).sum()),
            "context_only": int(((psi > 0) & (cnt == 0)).sum()),
            "answer_only": int(((cnt > 0) & (psi == 0)).sum()),
            "live": int(live.sum()),
        },
        "full_corpus_census": dict(CENSUS_EXPECTED),
        "full_corpus_census_cross_check": cross,
        "occupancy_ratio": _drift(ratio, FULL_OCCUPANCY_RATIO),
        "note": (
            "FEATURE COUNTS are NOT expected to match: the full-corpus census "
            "sees 120,000 rows, this subsample sees "
            f"{acc.n_rows}. The binding read is occupancy_ratio "
            "= sum(cnt)/sum(psi_cnt), which is row-count-invariant."
        ),
    }


def _gate_spans(acc: Accum) -> dict:
    """Gate 2 — span-length ratio vs the measured full corpus (2.291)."""
    ctx_mean = acc.ctx_tokens_raw / acc.n_rows
    ans_mean = acc.ans_tokens_raw / acc.n_rows
    return {
        "context_mean_tokens": float(ctx_mean),
        "answer_mean_tokens": float(ans_mean),
        "full_corpus_reference": {"context_mean": 189.0, "answer_mean": 432.9},
        "span_ratio": _drift(ans_mean / ctx_mean, FULL_SPAN_RATIO),
        "kept_pool": {
            "context_mean_tokens": float(acc.ctx_tokens_kept / acc.n_rows),
            "answer_mean_tokens": float(acc.ans_tokens_kept / acc.n_rows),
            "note": (
                "post-mask (BOS_OFFSET=8 strip + >10x-median-norm outlier drop). The "
                "BOS strip removes 8 CONTEXT tokens per row, so the kept-pool ratio "
                "sits ABOVE the raw span ratio by construction — this is the expected "
                "source of gate-3 drift, not a sampling defect."
            ),
        },
    }


def _gate_token_null(acc: Accum) -> dict:
    """Gate 3 — global token-level answer share of FIRINGS."""
    a = float(acc.ans_tok.sum())
    c = float(acc.ctx_tok.sum())
    null = a / (a + c) if (a + c) else float("nan")
    kept_pred = acc.ans_tokens_kept / (acc.ans_tokens_kept + acc.ctx_tokens_kept)
    return {
        "definition": "sum(ans_tokens_active) / (sum(ans_tokens_active) + sum(ctx_tokens_active))",
        "token_level_null": _drift(null, FULL_TOKEN_NULL),
        "kept_pool_span_prediction": float(kept_pred),
        "note": (
            "The span-length prediction for THIS subsample is the kept-pool share "
            f"{kept_pred:.4f} (the BOS strip removes 8 context tokens/row); the "
            f"{FULL_TOKEN_NULL} reference is the RAW full-corpus span share. Compare "
            "the observed null against BOTH."
        ),
    }


# ── checkpoint / resume ───────────────────────────────────────────────────────


def _regime_fingerprint(args, fit_pool: np.ndarray) -> str:
    """Every output-affecting knob. A resume against a DIFFERENT regime must
    fail loud rather than silently fuse two populations (#722 r3 class)."""
    parts = [
        f"seed={SEED}",
        f"n_rows={args.n_rows}",
        f"sample_mode={args.sample_mode}",
        f"n_chunks={args.n_chunks if args.sample_mode != 'full' else 'n/a'}",
        f"layer={LAYER}",
        f"k={K}",
        f"tiny_model={int(bool(args.tiny_model))}",
        f"fit_pool={_sha_ids(fit_pool)[:16]}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _atomic_savez(path: Path, **arrays) -> None:
    """Atomic npz write.

    Writes through an open HANDLE, not a path: ``np.savez`` APPENDS ``.npz`` to
    a path argument that lacks the suffix, so a dotted temp name silently lands
    at ``<tmp>.npz`` and the following ``os.replace`` raises FileNotFoundError.
    A handle is written verbatim.
    """
    tmp = path.parent / f".{path.name}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def _load_row_cache(path: Path, fingerprint: str) -> list | None:
    """Fetched-row cache: a resume must not re-stream 1,920 chunks (~15 min).

    Holds REAL user text (LMSYS/WildChat-class) — pod-local scratch only, never
    uploaded, never committed, never printed.
    """
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            head = json.loads(fh.readline())
            if head.get("fingerprint") != fingerprint:
                _log(
                    f"row cache regime mismatch ({head.get('fingerprint')} != {fingerprint}) — refetch"
                )
                return None
            rows = [json.loads(ln) for ln in fh]
    except (OSError, ValueError, KeyError) as exc:
        _log(f"row cache unreadable ({type(exc).__name__}) — refetch")
        return None
    _log(f"row cache HIT: {len(rows)} rows (fetch skipped)")
    return [(r["row_idx"], r["ci"], r["prompt"], r["response"]) for r in rows]


def _write_row_cache(path: Path, fingerprint: str, rows: list) -> None:
    tmp = path.parent / f".{path.name}.tmp{os.getpid()}"
    with tmp.open("w") as fh:
        fh.write(json.dumps({"fingerprint": fingerprint}) + "\n")
        for row_idx, ci, prompt, response in rows:
            fh.write(
                json.dumps(
                    {"row_idx": int(row_idx), "ci": int(ci), "prompt": prompt, "response": response}
                )
                + "\n"
            )
    os.replace(tmp, path)
    _log(f"row cache written: {len(rows)} rows -> {path.name}")


# ── capture ───────────────────────────────────────────────────────────────────


def phase_capture(args) -> None:
    t0 = time.time()
    pools = _load_split_and_assert(args)
    fingerprint = _regime_fingerprint(args, pools["sae_fit"])
    _log(f"regime fingerprint {fingerprint} (resume={'on' if args.resume else 'off'})")
    row_ci = np.load(args.scratch / "row_ci.npy")
    prov_u8 = np.load(args.scratch / "prov.npy")
    if args.sample_mode == "full":
        rows, sample_meta = _sample_full_pool(args, pools["sae_fit"], prov_u8, row_ci, fingerprint)
    else:
        cands, chunk_meta = _collect_candidates(args, pools["sae_fit"], row_ci)
        rows, sample_meta = _sample_rows(args, cands, prov_u8)
        sample_meta["sampler"] = f"chunk-subset (SMOKE ONLY, {args.n_chunks} chunks)"
        sample_meta["chunk_stage"] = chunk_meta
        del cands

    if args.pilot_rows > 0:
        rows = rows[: args.pilot_rows]
        _log(f"PILOT: {len(rows)} rows (measured per-row basis; no artifact written)")
    rows_n = len(rows)

    model, tok = EA._load_model_tok(args)
    _init_special_ids(tok)
    prefix_chars = EA._prefix_char_len(tok)
    sae = S.BatchTopKSAE.load(k=K, device=args.device, cache_dir=args.sae_dir, layer=LAYER)
    acc = Accum(args.device)

    prepared = []
    for row_idx, ci, prompt, response in rows:
        tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
        if tk is None:
            continue
        # int32 ids, not a python list: at 120k rows the list-of-int form costs
        # ~2.7 GB of int objects for the same 74M token ids (~300 MB here).
        prepared.append(
            (row_idx, ci, np.asarray(tk[0], dtype=np.int32), tk[1], tk[2], tk[3], tk[4])
        )
    del rows
    prepared.sort(key=lambda r: len(r[2]))  # length-sorted -> tight right-padding
    _log(f"tokenized {len(prepared)}/{rows_n} rows (empty-response rows dropped)")

    # Resume: the capture loop is >1h and >50 units at production n, so it
    # persists accumulator state per chunk of rows and skips completed ones
    # (code-style.md § "Checkpoint per phase", intra-phase grain).
    ckpt = args.scratch / f"capture_ckpt_{fingerprint}.npz"
    processed: set[int] = set()
    if args.resume and ckpt.exists():
        with np.load(ckpt, allow_pickle=False) as z:
            got = str(z["fingerprint"])
            assert got == fingerprint, f"checkpoint regime drift: {got} != {fingerprint}"
            processed = {int(r) for r in acc.restore(z)}
        _log(f"RESUMED from {ckpt.name}: {len(processed)} rows already captured")
    todo = [r for r in prepared if int(r[0]) not in processed]
    del prepared
    if processed:
        _log(f"  {len(todo)} rows remain")

    t_loop = time.time()
    tok_at_start = acc.ctx_tokens_raw + acc.ans_tokens_raw  # restored tokens are NOT this leg's
    rows_at_start = acc.n_rows
    n_done = 0
    since_ckpt = 0
    for s in range(0, len(todo), args.batch_rows):
        batch = todo[s : s + args.batch_rows]
        caps = EA._batched_capture(model, tok, batch, [LAYER], args.device)
        for r, cap in zip(batch, caps, strict=True):
            h = cap[LAYER].to(args.device)
            _row_pass(sae, acc, h, np.asarray(r[2], dtype=np.int64), r[2:], args.device)
            del h
            processed.add(int(r[0]))
        n_done += len(batch)
        since_ckpt += len(batch)
        if since_ckpt >= args.ckpt_every and args.resume:
            _atomic_savez(ckpt, **acc.state(np.fromiter(processed, dtype=np.int64), fingerprint))
            since_ckpt = 0
        if n_done % (args.batch_rows * 8) == 0 or n_done == len(todo):
            tps = (acc.ctx_tokens_raw + acc.ans_tokens_raw) / max(time.time() - t_loop, 1e-9)
            _log(f"  rows {n_done}/{len(todo)} — {tps:,.0f} tok/s")

    wall = time.time() - t_loop
    tot_tok = acc.ctx_tokens_raw + acc.ans_tokens_raw
    leg_tok = tot_tok - tok_at_start
    leg_rows = acc.n_rows - rows_at_start
    perf = {
        "rows": acc.n_rows,
        "tokens_total": tot_tok,
        "tokens_per_row": tot_tok / max(acc.n_rows, 1),
        # Throughput is THIS leg's only: a resumed run's restored tokens carry
        # no wall time here and would inflate tok/s.
        "capture_wall_s": wall,
        "rows_this_leg": leg_rows,
        "tokens_this_leg": leg_tok,
        "tokens_per_s": leg_tok / max(wall, 1e-9),
        "s_per_row": wall / max(leg_rows, 1),
        "resumed_rows": rows_at_start,
    }
    _log(f"capture: {perf['rows']} rows / {perf['tokens_per_s']:,.0f} tok/s / {wall:.1f}s")

    if args.pilot_rows > 0:
        projected = perf["s_per_row"] * args.n_rows
        _write_json(
            args.out_eval / "run_length_pilot.json",
            {"pilot": perf, "projected_full_wall_s": projected, "n_rows_target": args.n_rows},
        )
        _log(f"PILOT projected full-run wall: {projected / 60:.1f} min for {args.n_rows} rows")
        return

    _finalize(args, acc, sample_meta, perf, t0)


def _finalize(args, acc: Accum, sample_meta: dict, perf: dict, t0: float) -> None:
    n = lambda x: x.cpu().numpy()  # noqa: E731
    ans_tok, ctx_tok = n(acc.ans_tok), n(acc.ctx_tok)
    tmpl_tok, pairs, denom = n(acc.tmpl_tok), n(acc.pairs), n(acc.denom)
    var_sum, var_rows = n(acc.var_sum), n(acc.var_rows)
    all_tok = ans_tok + ctx_tok

    runs = ans_tok - pairs
    nan = lambda: np.full(DICT_SIZE, np.nan)  # noqa: E731
    mean_run_length = np.divide(ans_tok, runs, out=nan(), where=runs > 0)
    persistence_p = np.divide(pairs, denom, out=nan(), where=denom > 0)
    template_token_frac = np.divide(tmpl_tok, all_tok, out=nan(), where=all_tok > 0)
    act_var = np.divide(var_sum, var_rows, out=nan(), where=var_rows > 0)
    side_ratio_token = np.divide(ans_tok, all_tok, out=nan(), where=all_tok > 0)

    assert np.all(runs >= 0), "negative run count — pair counting overshot"
    assert np.all(np.isnan(mean_run_length) | (mean_run_length >= 1.0)), (
        "mean_run_length < 1 token is impossible"
    )
    assert np.all(np.isnan(persistence_p) | ((persistence_p >= 0) & (persistence_p <= 1))), (
        "persistence_p outside [0, 1]"
    )
    assert np.all(np.isnan(template_token_frac) | (template_token_frac <= 1.0 + 1e-9)), (
        "template_token_frac > 1"
    )

    out_dir = args.out_eval
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "run_length_perfeature.npz"
    np.savez_compressed(
        npz_path,
        feat_ids=np.arange(DICT_SIZE, dtype=np.int64),
        mean_run_length=mean_run_length,
        persistence_p=persistence_p,
        template_token_frac=template_token_frac,
        ctx_tokens_active=ctx_tok,
        ans_tokens_active=ans_tok,
        act_var_across_tokens=act_var,
        side_ratio_token=side_ratio_token,
        n_runs=runs,
        n_adjacent_pairs=pairs,
        persistence_denom=denom,
        template_firings=tmpl_tok,
        row_occupancy_ans=n(acc.cnt),
        row_occupancy_ctx=n(acc.psi_cnt),
        act_var_n_answers=var_rows,
    )

    gates = {
        "gate1_row_occupancy_census": _gate_census(acc),
        "gate2_span_length_ratio": _gate_spans(acc),
        "gate3_token_level_null": _gate_token_null(acc),
    }
    flags = [k for k, v in gates.items() if "DRIFT" in json.dumps(v)]
    live_ans = int((ans_tok > 0).sum())
    meta = {
        "artifact": "run_length_perfeature.npz",
        "join_key": "feat_ids",
        "dict_size": DICT_SIZE,
        "produced_by": "scripts/issue1482_run_length.py",
        "git_commit": _git_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s_total": time.time() - t0,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "sae": {
            "repo": S.SAE_REPO,
            "revision": S.SAE_REVISION,
            "subdir": S.trainer_subdir(LAYER, K),
            "site": f"resid_post layer {LAYER}",
            "k": K,
            "dict_size": DICT_SIZE,
        },
        "seed": SEED,
        "sample": sample_meta,
        "token_pool": (
            "issue1482_error_analysis._row_features reference semantics: first "
            f"BOS_OFFSET={S.BOS_OFFSET} positions dropped + >"
            f"{S.OUTLIER_NORM_FACTOR}x-median-norm token rows dropped, per row. "
            "Runs/adjacency are keyed on ORIGINAL token positions, so a masked-out "
            "token BREAKS a run rather than splicing two runs together."
        ),
        "template_mask_definition": TEMPLATE_MASK_DEF,
        "template_special_token_ids": list(_SPECIAL_IDS),
        "row_occupancy_vs_token_weighted": (
            "The banked `side_ratio` is ROW OCCUPANCY: a feature active anywhere in a "
            "span contributes 1, whether it fires on 1 token or 400. "
            "`ans_tokens_active` / `ctx_tokens_active` here are TOKEN COUNTS, so "
            "`side_ratio_token` is a genuinely different measurement — not a refinement "
            "of `side_ratio`. `row_occupancy_ans` / `row_occupancy_ctx` are ALSO emitted "
            "so the two conventions can be compared on the same rows."
        ),
        "run_length_vs_persistence": (
            "E[R] = 1/(1-p) EXACTLY for a stationary process, so mean_run_length and "
            "persistence_p are the same quantity in different units. mean_run_length = "
            "ans_tokens_active / n_runs with n_runs = ans_tokens_active - "
            "n_adjacent_pairs; persistence_p = n_adjacent_pairs / persistence_denom, "
            "where persistence_denom counts active answer positions whose ORIGINAL "
            "successor is also in the kept pool. CAVEAT: E[R] = 1/(1-p) is a "
            "STATIONARY-PROCESS identity, not a numeric identity of these two "
            "finite-span estimators — span boundaries and token-pool mask gaps make "
            "them diverge (a run truncated by the span end contributes to "
            "mean_run_length but its final token is excluded from persistence_denom). "
            "The consumer plots only mean_run_length."
        ),
        "act_var_across_tokens_definition": (
            "Within-answer variance of the activation VALUE across ALL kept answer "
            "tokens (zeros included — the variance of the activation TRACE, not a "
            "conditional-on-firing variance), averaged over the answers in which the "
            "feature fires at least once (`act_var_n_answers`)."
        ),
        "scaffold_frac_disambiguation": (
            "`template_token_frac` is BEHAVIOURAL (which tokens the feature fires on). "
            "The banked `scaffold_frac` is GEOMETRIC (decoder-vector mass in the "
            "prefix-covariance top-48 eigen-subspace, #1773 L475-486). They are twins, "
            "not duplicates — do not conflate them."
        ),
        "coverage": {
            "features_with_any_answer_firing": live_ans,
            "features_with_any_firing": int((all_tok > 0).sum()),
            "features_all_nan_mean_run_length": int(np.isnan(mean_run_length).sum()),
            "note": (
                "A feature that never fires in this subsample is NaN, never 0 — the "
                "consumer nan-inits and scatters by feat_ids, so NaN reads as 'not "
                "measured here', which is correct."
            ),
        },
        "gates": gates,
        "gate_flags": flags,
        "perf": perf,
    }
    _write_json(out_dir / "run_length_perfeature.meta.json", meta)
    _log(f"wrote {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    for k, v in gates.items():
        _log(
            f"GATE {k}: {json.dumps(v.get('occupancy_ratio') or v.get('span_ratio') or v.get('token_level_null'))}"
        )
    if flags:
        _log(f"GATE DRIFT FLAGGED: {flags} — reported, not suppressed")


# ── upload ────────────────────────────────────────────────────────────────────


def phase_upload(args) -> None:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    names = ["run_length_perfeature.npz", "run_length_perfeature.meta.json"]
    for nm in names:
        p = args.out_eval / nm
        assert p.exists() and p.stat().st_size > 0, f"missing artifact: {p}"
    # ONE bulk folder commit, not a per-file loop (the #664/#1481 storm
    # anti-pattern). `run_length_pilot.json` is a local wall-time measurement
    # record, not an artifact, so it is filtered out rather than shipped.
    hub._upload(
        args.out_eval,
        HF_DATA_REPO,
        "dataset",
        HF_PREFIX,
        ignore_patterns=["run_length_pilot.json"],
        raise_on_error=True,
    )
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        HF_DATA_REPO,
        [f"{HF_PREFIX}/{nm}" for nm in names],
        path_in_repo=HF_PREFIX,
        repo_type="dataset",
    )
    assert not missing, f"upload verification FAILED, missing on Hub: {missing}"
    _log(f"uploaded + verified {len(names)} files under {HF_DATA_REPO}:{HF_PREFIX}")


# ── cli ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=("stage", "capture", "upload"), required=True)
    ap.add_argument("--n-rows", type=int, default=2000)
    ap.add_argument(
        "--sample-mode",
        choices=("full", "chunk-subset"),
        default="full",
        help="full = unbiased draw over the whole sae_fit pool (PRODUCTION); "
        "chunk-subset = fast shard-stratified cluster draw (SMOKE ONLY, biased)",
    )
    ap.add_argument("--n-chunks", type=int, default=128, help="chunk-subset mode only")
    ap.add_argument("--pilot-rows", type=int, default=0)
    ap.add_argument("--batch-rows", type=int, default=8)
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="checkpoint accumulators + cache fetched rows so a crash resumes "
        "(required at production n: >1h wall and >50 units)",
    )
    ap.add_argument("--ckpt-every", type=int, default=5000, help="rows between checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny-model", action="store_true", help="CPU smoke: from-config 24-layer")
    ap.add_argument("--scratch", type=Path, default=REPO / "data/issue_1482/run_length/scratch")
    ap.add_argument("--sae-dir", type=Path, default=REPO / "data/issue_1482/run_length/sae")
    ap.add_argument("--out-eval", type=Path, default=REPO / "eval_results/issue_1482/run_length")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.import_check:
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401

        from explore_persona_space.orchestrate import hub  # noqa: F401

        print("import-check ok")
        return
    for d in (args.scratch, args.sae_dir, args.out_eval):
        d.mkdir(parents=True, exist_ok=True)
    {"stage": phase_stage, "capture": phase_capture, "upload": phase_upload}[args.phase](args)


if __name__ == "__main__":
    main()
