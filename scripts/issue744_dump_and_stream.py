#!/usr/bin/env python3
"""Issue #744 Phase 1 — residual-stream dump + streaming aggregation (GPU).

Pure forward-pass read of the Qwen-2.5-7B residual stream over the two corpora
(NO training). For every sequence, batch=1, no padding, all 28 decoder layers
captured via the #594 ``LayerCapture`` forward-hook pattern (``output[0]`` =
pre-final-norm residual, NOT ``output_hidden_states`` whose last element is
post-final-norm and would break layer-27 comparability).

Two storage strategies, by corpus (plan §4.3 / §9 sizing):

* **Natural Stories (Corpus A): RETAIN the full raw fp16 dump** (~2.6 GB) — H3
  per-position retrieval + the sink-EXCLUDED H1/H2 recompute the analyzer needs
  (plan-marker concern #1) require per-position vectors. Each story is the FULL
  word list (NOT truncated at 1024 tokens), processed in OVERLAPPING
  ``max_seq_len``-token chunks (Barenholtz 2606.05346 §2.1) and reassembled to
  cover the whole story — the late-position tail the late-layer H1 read depends
  on does not vanish (#744 C2). Per sequence we write
  ``ns_raw/seq_<item>.pt`` = ``{H_fp16 (L,T_full,hidden), surprisal, sink_mask
  (per-(layer,position)), special_mask, clause_opener_mask, words, word_end_idx}``.
* **Broader (Corpus B): STREAM** — a two-pass scheme. Pass 1 streams per-dim
  fp32 sufficient statistics (Welford sums + sum-of-squares) per layer to fix
  the population z-standardization ``(mu_L, sigma_L)`` AND ranks the rogue dims
  (concern #3: ranked by RAW variance / max-dominance / contribution-to-cosine
  / kurtosis on the un-standardized residuals, NOT "standardized variance").
  Pass 2 re-forwards each sequence and emits per-sequence summary statistics
  (per-layer per-flavor per-step sums + valid-pair counts) under the FIXED
  stats. A bounded raw subset (``--broader-raw-keep`` sequences) is retained for
  the lens-3 broader spot-check.

Per-position masks (computed in Phase 1, stored alongside, plan §4.3):

* **surprisal** — Qwen-2.5-7B's own next-token NLL at each position
  (``-log_softmax(logits)[t-1, token_t]``; surprisal at position 0 is NaN — no
  preceding context).
* **sink/outlier mask** — per (layer, position): a position is a sink at layer L
  iff its hidden state at L contains an activation with ``|h| > 100 AND |h| >=
  1000 x median(|h| at that layer-position)`` (Sun 2402.17762 §2). RETAINED
  per-(layer,position) so the analyzer can recompute the sink-EXCLUDED curve at
  zero GPU cost (concern #1).
* **special-token mask** — per position: BOS / "." / newline / delimiter (Sun's
  enrichment classes).
* **clause-opener mask** — per position: the last subword of a clause-opener
  word (closed-class wordlist; gold-Penn cross-check done in the analyzer).

Pod-side contract (#594): emits ``[phase=...]`` log lines ending in
``[phase=done]`` and writes a ``poll_pipeline.py``-conformant end-of-run
sentinel. Uploads ALL Phase-1 artifacts via ONE bulk ``upload_folder`` to
``issue744_token_continuity/`` and verifies via ``list_repo_files`` BEFORE the
orchestrator releases the GPU pod.

``--smoke`` runs the IDENTICAL code path at tiny N (whatever the smoke corpora
hold) into ``<out-dir>`` with a tiny HF upload probe (or ``--no-upload`` for the
local CPU smoke). No separate smoke architecture (plan §4.6).

Usage (plan §10 launch)::

    uv run python scripts/issue744_dump_and_stream.py \\
        --corpora-dir data/issue_744/corpora --model Qwen/Qwen2.5-7B \\
        --out-dir data/issue_744/base --gpu-id 0

    # instruct arm: --model Qwen/Qwen2.5-7B-Instruct --out-dir data/issue_744/instruct
    # local CPU smoke (tiny throwaway model):
    uv run python scripts/issue744_dump_and_stream.py --smoke \\
        --corpora-dir /tmp/issue744_smoke/corpora --model Qwen/Qwen2.5-0.5B \\
        --expected-layers 24 --expected-hidden 896 --device cpu \\
        --out-dir /tmp/issue744_smoke/base --no-upload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue744_common import (  # noqa: E402
    BROADER_RANDOM_POOL,
    DEFAULT_MODEL,
    DIRECTION_PRES_STEPS,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    FLAVORS,
    HF_DATA_REPO,
    HF_OVERFLOW_REPO,
    HF_PREFIX,
    MAX_SEQ_LEN,
    NS_CHUNK_STRIDE,
    RANDOM_BASELINE_N_PAIRS,
    ROGUE_DIM_TOPK,
    SEED,
    SINK_ABS_FLOOR,
    SINK_MEDIAN_RATIO,
    TRAJECTORY_WINDOW_K,
    is_clause_opener,
    write_json,
)

from explore_persona_space.analysis.continuity import (  # noqa: E402
    DEFAULT_ROGUE_RANK_METRIC as ROGUE_RANK_METRIC,
)
from explore_persona_space.analysis.continuity import (  # noqa: E402
    ReservoirVectorPool,
    WelfordDimStats,
    consec_cosine,
    direction_preservation,
    extrap_error,
    make_flavors_from_stats,
    random_baseline,
)
from explore_persona_space.analysis.penn_parser import build_ns_gold_clause_mask  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue744_dump")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SENTINEL_SCHEMA_VERSION = 1


def phase(name: str) -> None:
    """Emit a poll_pipeline.py-parseable phase line (PHASE_RE on the log tail)."""
    print(f"[phase={name}]", flush=True)


# ── Hook capture (full per-layer stack; #594 pattern, adapted to all positions) ──


class LayerCapture:
    """Forward hooks on every decoder block; keeps the latest (1, T, H) per layer."""

    def __init__(self, model, n_layers: int):
        self.latest: dict[int, torch.Tensor] = {}
        self._handles = []
        for li in range(n_layers):
            self._handles.append(model.model.layers[li].register_forward_hook(self._make_hook(li)))

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            self.latest[layer_idx] = hs.detach()

        return hook_fn

    def full_stack(self, n_layers: int) -> torch.Tensor:
        """(L, T, H) fp32 CPU stack of the full per-layer residual stream."""
        vecs = [self.latest[li][0].float().cpu() for li in range(n_layers)]
        self.latest.clear()
        return torch.stack(vecs)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


# ── Per-sequence forward: activations + surprisal + masks ──────────────────────


def _surprisal_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Qwen self next-token NLL per position (surprisal). (1,T,V) logits, (1,T) ids.

    Position 0 has no preceding context -> NaN. For t>=1, surprisal_t =
    -log p(token_t | tokens_0..t-1) = NLL from the logits at index t-1.

    Built on ``logits.device`` (CUDA in the production GPU path, where the model
    loads with ``device_map={"":cuda:0}``; CPU in the local smoke). The
    ``torch.full`` ``out`` and the ``torch.arange`` index MUST live on the same
    device as ``logp`` / ``ids`` — a CPU ``out`` taking a CUDA RHS via the
    ``out[1:] = ...`` indexed assignment raises ``RuntimeError: Expected all
    tensors to be on the same device`` on the first production sequence (the CPU
    smoke could never exercise this; #744 C1). Returns a CPU ``(T,)`` tensor with
    ``pos0 = NaN`` per the ``forward_sequence`` contract + the
    ``test_surprisal_off_by_one`` regression.
    """
    device = logits.device
    logp = torch.log_softmax(logits[0].float(), dim=-1)  # (T, V) on logits.device
    ids = input_ids[0].to(device)  # (T,) — colocate with logp for the gather below
    T = ids.shape[0]
    out = torch.full((T,), float("nan"), device=device)
    if T >= 2:
        # surprisal at t = -logp[t-1, ids[t]]  for t = 1..T-1 (all on `device`)
        idx = torch.arange(1, T, device=device)
        out[1:] = -logp[idx - 1, ids[idx]]
    return out.cpu()


def _sink_mask(H: torch.Tensor) -> torch.Tensor:
    """Per-(layer, position) sink mask (Sun 2402.17762 §2). H is (L, T, hidden).

    A position is a sink at layer L iff its hidden state contains an activation
    with |h| > SINK_ABS_FLOOR AND |h| >= SINK_MEDIAN_RATIO x median(|h|) over
    that (layer, position)'s hidden vector. Returns (L, T) bool.
    """
    absH = H.abs()  # (L, T, hidden)
    med = absH.median(dim=-1).values + 1e-8  # (L, T)
    mx = absH.max(dim=-1).values  # (L, T)
    return (mx > SINK_ABS_FLOOR) & (mx >= SINK_MEDIAN_RATIO * med)


def _special_token_mask(tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-position BOS/'.'/newline/delimiter mask (Sun's enrichment classes)."""
    ids = input_ids[0].tolist()
    T = len(ids)
    mask = torch.zeros(T, dtype=torch.bool)
    bos = tokenizer.bos_token_id
    for t, tid in enumerate(ids):
        if t == 0 or tid == bos:
            mask[t] = True
            continue
        dec = tokenizer.decode([tid])
        s = dec.strip()
        if s in {".", "\n", ",", ";", ":", "!", "?"} or dec in {"\n", ".", "\n\n"} or "\n" in dec:
            mask[t] = True
    return mask


def tokenize_ns_sequence(tokenizer, words: list[str], gold_clause_words: list[bool]):
    """Tokenize the FULL NS word list, aligning each word to its last subword index.

    Returns (input_ids (1,T_full), clause_opener_mask (T_full,) bool,
    clause_opener_mask_wordlist_proxy (T_full,) bool, word_end_idx list). NO
    truncation — Natural Stories items run ~1,026 words (>1024 Qwen subword tokens
    for a typical story), and the late-position tail is exactly what the
    late-layer direction-preservation hypothesis (H1) and the H3 stratification
    most depend on. The full story is processed in overlapping chunks downstream
    (``forward_ns_overlapping``, Barenholtz 2606.05346 §2.1), not truncated
    (#744 C2).

    PRIMARY mask = ``clause_opener_mask``: the GOLD Penn clause-opener label
    (plan §11 ``syntactic_mask_ns`` = "first terminal under S/SBAR in gold Penn
    parse OR CC/IN clause-opener"), supplied per word in ``gold_clause_words``
    (built once for the whole NS stream by ``build_ns_gold_clause_mask``). This
    is the mask the H3 syntactic stratification reads. The closed-class wordlist
    (``is_clause_opener``) is emitted ALONGSIDE as
    ``clause_opener_mask_wordlist_proxy`` for the A11 gold-vs-proxy cross-check
    (`proxy_vs_gold_penn.json`) and is the PRIMARY mask only for the broader
    corpus (no gold parses for WikiText). Both masks mark the last-subword
    position of the word. ``word_end_idx[i]`` = the token index of word i's last
    subword (for the word-level last-subword read, plan concern #5).
    """
    assert len(gold_clause_words) == len(words), (
        f"gold mask len {len(gold_clause_words)} != n words {len(words)}"
    )
    token_ids: list[int] = []
    word_end_idx: list[int] = []
    gold_words: list[bool] = []
    proxy_words: list[bool] = []
    for w, gold_co in zip(words, gold_clause_words, strict=True):
        sub = tokenizer(" " + w if token_ids else w, add_special_tokens=False)["input_ids"]
        if not sub:
            continue
        token_ids.extend(sub)
        word_end_idx.append(len(token_ids) - 1)
        gold_words.append(bool(gold_co))
        proxy_words.append(is_clause_opener(w))
    T = len(token_ids)
    clause_mask = torch.zeros(T, dtype=torch.bool)  # PRIMARY: gold Penn
    proxy_mask = torch.zeros(T, dtype=torch.bool)  # wordlist proxy (cross-check)
    for idx, gold_co, proxy_co in zip(word_end_idx, gold_words, proxy_words, strict=True):
        if gold_co:
            clause_mask[idx] = True
        if proxy_co:
            proxy_mask[idx] = True
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    return input_ids, clause_mask, proxy_mask, word_end_idx


def _chunk_starts(t_full: int, max_len: int, stride: int) -> list[int]:
    """Overlapping-chunk start offsets covering [0, t_full) (Barenholtz §2.1).

    Emits chunk starts ``0, stride, 2*stride, ...`` while each chunk
    ``[start, min(start+max_len, t_full))`` advances; the final chunk's end is
    clamped to ``t_full`` (a story <= max_len yields a single chunk at start 0).
    Stride < max_len gives the overlap (default 50%); the per-position de-dup
    (``assemble_overlapping_chunks``) picks, for each position, the chunk where it
    has the most in-chunk left-context so the k-window fit + +max_step lookahead
    are fully in-context.
    """
    assert 0 < stride <= max_len, (stride, max_len)
    if t_full <= max_len:
        return [0]
    starts = list(range(0, t_full - max_len + 1, stride))
    # Ensure the tail is covered: the last chunk must end at t_full.
    last_start = t_full - max_len
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def assemble_overlapping_chunks(
    chunk_outputs: list[tuple[int, torch.Tensor, torch.Tensor]],
    t_full: int,
    n_layers: int,
    hidden: int,
    context_floor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """De-dup overlapping per-chunk reads into full-story (L,T_full,H) + surprisal.

    ``chunk_outputs`` is a list of ``(start, H_chunk (L, c, hidden),
    surp_chunk (c,))`` per chunk (in ``_chunk_starts`` order). For each absolute
    position ``p`` in ``[0, t_full)`` we take the read from the LAST chunk that
    contains ``p`` AND in which ``p`` sits at least ``context_floor`` tokens after
    the chunk start (``context_floor = k + max_step`` so the k-window OLS fit + the
    +max_step lookahead are in-context). The first ``context_floor`` positions of
    the whole story (which no chunk can give that much left-context) fall back to
    chunk 0 (start 0), the only chunk that contains them at all. Later chunks are
    processed last, so iterating chunks in order and overwriting yields the
    LAST-qualifying-chunk read per position.

    Returns ``(H_full (L, T_full, hidden), surprisal_full (T_full,))`` on CPU.
    Surprisal at a covered position is the chunk's own next-token NLL, which is
    well-defined for any in-chunk position with >=1 preceding token; position 0 of
    the WHOLE story stays NaN (no preceding context) regardless of which chunk
    covers it.
    """
    H_full = torch.full((n_layers, t_full, hidden), float("nan"))
    surp_full = torch.full((t_full,), float("nan"))
    assigned = torch.zeros(t_full, dtype=torch.bool)
    # Pass 1: fill from the qualifying (>=context_floor left-context) chunk reads,
    # iterating in order so a later chunk overwrites an earlier one (LAST wins).
    for start, H_chunk, surp_chunk in chunk_outputs:
        c = H_chunk.shape[1]
        for local in range(c):
            p = start + local
            if local < context_floor:
                continue  # not enough in-chunk left-context for the fit at p
            H_full[:, p] = H_chunk[:, local]
            surp_full[p] = surp_chunk[local]
            assigned[p] = True
    # Pass 2: positions never assigned (the first context_floor tokens of the
    # story, plus any uncovered) fall back to the EARLIEST chunk covering them.
    for start, H_chunk, surp_chunk in chunk_outputs:
        c = H_chunk.shape[1]
        for local in range(c):
            p = start + local
            if assigned[p]:
                continue
            H_full[:, p] = H_chunk[:, local]
            surp_full[p] = surp_chunk[local]
            assigned[p] = True
    # Surprisal at whole-story position 0 has no preceding context -> NaN.
    surp_full[0] = float("nan")
    assert bool(assigned.all()), "overlapping-chunk assembly left positions uncovered"
    return H_full, surp_full


def forward_sequence(
    model, tokenizer, input_ids: torch.Tensor, capture: LayerCapture, n_layers: int
):
    """Forward one sequence; return (H (L,T,hidden) fp32 CPU, surprisal (T,) CPU)."""
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    H = capture.full_stack(n_layers)  # (L, T, hidden) fp32 CPU
    surprisal = _surprisal_from_logits(out.logits, input_ids)  # (T,) CPU
    return H, surprisal


def forward_ns_overlapping(
    model,
    capture: LayerCapture,
    input_ids: torch.Tensor,
    n_layers: int,
    hidden: int,
    max_len: int,
    stride: int,
    context_floor: int,
):
    """Forward a FULL NS story in overlapping chunks; reassemble full-story reads.

    Barenholtz 2606.05346 §2.1: a story longer than the model's context window is
    processed in overlapping ``max_len``-token chunks, NOT truncated (#744 C2).
    ``input_ids`` is the FULL story ``(1, T_full)``. Each chunk
    ``[start, start+max_len)`` is a separate batch-1 forward (the LayerCapture
    hooks re-capture per forward); per-chunk residual stacks + surprisal are then
    de-duplicated by ``assemble_overlapping_chunks`` (each position taken from the
    LAST chunk where it has >= ``context_floor`` in-chunk left-context, so the
    k-window OLS fit + the +max_step lookahead are in-context).

    Returns ``(H_full (L, T_full, hidden) fp32 CPU, surprisal_full (T_full,) CPU)``
    covering the WHOLE story — same shape contract as ``forward_sequence`` but
    over ``T_full`` (un-truncated) rather than the first ``max_len`` tokens.
    """
    t_full = input_ids.shape[1]
    starts = _chunk_starts(t_full, max_len, stride)
    chunk_outputs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for start in starts:
        chunk_ids = input_ids[:, start : start + max_len].to(model.device)  # (1, c)
        with torch.no_grad():
            out = model(input_ids=chunk_ids)
        H_chunk = capture.full_stack(n_layers)  # (L, c, hidden) fp32 CPU
        surp_chunk = _surprisal_from_logits(out.logits, chunk_ids)  # (c,) CPU
        chunk_outputs.append((start, H_chunk, surp_chunk))
    return assemble_overlapping_chunks(chunk_outputs, t_full, n_layers, hidden, context_floor)


# ── Per-sequence summary (the streaming read) ──────────────────────────────────


def per_sequence_summary(
    H_flavors: dict[str, torch.Tensor], steps: tuple[int, ...], k: int
) -> dict:
    """Per-layer per-flavor SUMS for the consecutive-cosine, direction-pres, extrap.

    Returns sums (not means) + valid-pair counts so the bootstrap-over-sequences
    can weight by each sequence's valid-pair count (plan §6 CI methodology). The
    analyzer turns sums -> means via the weighted resample.

    Schema (all per layer L):
      flavors[f]["consec_cos_sum"]  (L,)   sum of cos(h_t, h_{t+1}) over pairs
      flavors[f]["consec_cos_n"]    (L,)   number of consecutive pairs (T-1)
      flavors[f]["dp_sum"][s]       (L,)   sum of abs-cos at step s (over windows)
      flavors[f]["dp_n"][s]         (L,)   number of valid windows at step s
      flavors[f]["extrap_sum"]      (L,)   sum of L2 extrap error over windows
      flavors[f]["extrap_n"]        (L,)   number of valid extrap windows
    """
    summary: dict = {"flavors": {}}
    for flavor, H in H_flavors.items():
        L, T, _ = H.shape
        cc = consec_cosine(H)  # (L, T-1)
        cc_n = torch.full((L,), cc.shape[1], dtype=torch.float64)
        dp = direction_preservation(H, k=k, steps=steps)  # {s: (L,) mean}
        ee = extrap_error(H, k=k)  # (L,) mean
        # direction_preservation / extrap_error return MEANS; recover sums by
        # multiplying by the valid-window count (same window logic).
        dp_sum: dict[int, list[float]] = {}
        dp_n: dict[int, list[float]] = {}
        for s in steps:
            max_w = T - 1 - k - s
            n_valid = max(0, max_w + 1)
            mean_s = dp[s]
            sum_s = torch.where(torch.isnan(mean_s), torch.zeros_like(mean_s), mean_s * n_valid)
            dp_sum[s] = sum_s.tolist()
            dp_n[s] = [float(n_valid)] * L
        n_extrap = max(0, T - k)  # window w predicts position w+k for w in 0..T-k-1
        ee_sum = torch.where(torch.isnan(ee), torch.zeros_like(ee), ee * n_extrap)
        summary["flavors"][flavor] = {
            "consec_cos_sum": cc.sum(dim=1).tolist(),
            "consec_cos_n": cc_n.tolist(),
            "dp_sum": {str(s): dp_sum[s] for s in steps},
            "dp_n": {str(s): dp_n[s] for s in steps},
            "extrap_sum": ee_sum.tolist(),
            "extrap_n": [float(n_extrap)] * L,
        }
    return summary


# ── Manifest / sentinel ────────────────────────────────────────────────────────


def write_sentinel(kind: str, note: str, task_id: int = 744) -> Path:
    """poll_pipeline.py-conformant end-of-run sentinel (_SENTINEL_REQUIRED_KEYS)."""
    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    kind_slug = kind.replace(":", "_")
    path = logs_dir / f"issue-{task_id}-{kind_slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,
        "note": note,
        "task_id": task_id,
        "by": "issue744_dump_and_stream",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote sentinel %s", path)
    return path


def _is_storage_quota_403(err: Exception) -> bool:
    msg = str(err)
    return "403" in msg and "storage" in msg.lower()


def upload_outputs(out_dir: Path, smoke: bool) -> dict:
    """Bulk-upload Phase-1 outputs to the HF data repo and verify (one commit)."""
    from huggingface_hub import HfApi

    api = HfApi()
    sub = "dump_smoke" if smoke else "dump"
    path_in_repo = f"{HF_PREFIX}/{sub}"
    repo_used = HF_DATA_REPO
    try:
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue744: {'smoke ' if smoke else ''}Phase-1 dump upload",
        )
    except Exception as e:
        if not _is_storage_quota_403(e):
            raise
        logger.warning("HF storage-quota 403 on %s; falling back to overflow repo", HF_DATA_REPO)
        repo_used = HF_OVERFLOW_REPO
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
            repo_id=HF_OVERFLOW_REPO,
            repo_type="dataset",
            commit_message="issue744: Phase-1 dump upload (quota-403 overflow fallback)",
        )
    files = [
        f for f in api.list_repo_files(repo_used, repo_type="dataset") if f.startswith(path_in_repo)
    ]
    expected = {f"{path_in_repo}/dump_manifest.json"}
    missing = expected - set(files)
    if missing:
        raise RuntimeError(f"upload verification failed; missing on {repo_used}: {missing}")
    logger.info("Upload verified on %s: %d files under %s", repo_used, len(files), path_in_repo)
    return {"repo": repo_used, "path_in_repo": path_in_repo, "n_files": len(files)}


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #744 Phase 1: dump + stream.")
    parser.add_argument("--corpora-dir", type=Path, default=PROJECT_ROOT / "data/issue_744/corpora")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data/issue_744/base")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--expected-layers", type=int, default=EXPECTED_LAYERS)
    parser.add_argument("--expected-hidden", type=int, default=EXPECTED_HIDDEN)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--rogue-topk", type=int, default=ROGUE_DIM_TOPK)
    parser.add_argument("--broader-raw-keep", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()

    phase("load")
    # Bind CVD BEFORE the first CUDA allocation (the +gpu_id clobber gotcha).
    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    out_dir = Path(args.out_dir)
    ns_raw_dir = out_dir / "ns_raw"
    broader_raw_dir = out_dir / "broader_raw"
    ns_raw_dir.mkdir(parents=True, exist_ok=True)
    broader_raw_dir.mkdir(parents=True, exist_ok=True)

    ns_corpus = json.loads((args.corpora_dir / "corpus_natural_stories.json").read_text())
    broader_corpus = json.loads((args.corpora_dir / "corpus_broader.json").read_text())

    # Copy the Phase-0 corpus JSONs + the gold Penn parses into the dump dir so
    # they travel with the HF upload AND the off-pod analyzer (which reads the
    # dump dir, not the corpora dir) finds the Penn parses for the A11 gold-vs-
    # proxy clause-opener cross-check.
    import shutil

    for fname in ("corpus_natural_stories.json", "corpus_broader.json", "corpus_manifest.json"):
        shutil.copy2(args.corpora_dir / fname, out_dir / fname)
    penn_src = args.corpora_dir / "ns_penn_parses.txt"
    if penn_src.exists():
        shutil.copy2(penn_src, out_dir / "ns_penn_parses.txt")

    import wandb

    run = wandb.init(
        project="explore-persona-space",
        name=f"issue744-dump{'-smoke' if args.smoke else ''}-{Path(args.out_dir).name}",
        mode=args.wandb_mode,
        config={
            "model": args.model,
            "smoke": args.smoke,
            "n_ns": len(ns_corpus["sequences"]),
            "n_broader": len(broader_corpus["sequences"]),
        },
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    assert n_layers == args.expected_layers, (
        f"model has {n_layers} decoder layers, expected {args.expected_layers} (A1)"
    )
    assert hidden == args.expected_hidden, (
        f"model hidden_size {hidden}, expected {args.expected_hidden} (A1)"
    )

    capture = LayerCapture(model, n_layers)
    steps = DIRECTION_PRES_STEPS
    k = TRAJECTORY_WINDOW_K
    # Overlapping-chunk context floor: a de-duplicated position needs k tokens of
    # left-context for the OLS fit + max(+s) for the lookahead in-chunk (#744 C2).
    ns_context_floor = k + max(steps)
    t0 = time.time()

    try:
        # ── Pass over Natural Stories: RETAIN the raw fp16 dump ─────────────────
        # FULL stories, processed in overlapping max_seq_len-token chunks
        # (Barenholtz §2.1) and reassembled — NO truncation (#744 C2).
        phase("dump_ns")
        ns_pop = WelfordDimStats(n_layers, hidden)  # NS population stats (from the dump)
        # Build the PRIMARY gold-Penn clause-opener mask once for the whole NS
        # stream (plan §11 syntactic_mask_ns). The parse file is a flat
        # document-order forest with no per-item delimiters, so the gold
        # terminals are aligned against the concatenated word stream and split
        # back per item. The wordlist proxy stays the BROADER mask + the A11
        # cross-check companion; it is NOT the NS primary.
        ns_penn_path = out_dir / "ns_penn_parses.txt"
        assert ns_penn_path.exists(), (
            f"gold Penn parses missing at {ns_penn_path} — required for the NS "
            f"syntactic mask (plan §11 syntactic_mask_ns)"
        )
        ns_words_by_item = [seq["words"] for seq in ns_corpus["sequences"]]
        gold = build_ns_gold_clause_mask(ns_penn_path.read_text(errors="replace"), ns_words_by_item)
        gold_align = gold["alignment"]
        assert gold_align.aligned_ok, (
            f"gold Penn ↔ NS word-stream alignment FAILED "
            f"(n_words={gold_align.n_words}, discrepancies={gold_align.n_discrepancies}, "
            f"gold_terminals={gold_align.n_gold_terminals}); the NS syntactic mask "
            f"cannot be built — refusing to fall back to the wordlist proxy"
        )
        logger.info(
            "[NS gold-Penn mask] aligned_ok=%s fully_consumed=%s n_words=%d "
            "discrepancies=%d gold_terminals=%d",
            gold_align.aligned_ok,
            gold_align.fully_consumed,
            gold_align.n_words,
            gold_align.n_discrepancies,
            gold_align.n_gold_terminals,
        )
        ns_seq_meta = []
        for i, seq in enumerate(ns_corpus["sequences"], 1):
            input_ids, clause_mask, proxy_mask, word_end_idx = tokenize_ns_sequence(
                tokenizer, seq["words"], gold["masks"][i - 1]
            )
            n_chunks = len(_chunk_starts(input_ids.shape[1], args.max_seq_len, NS_CHUNK_STRIDE))
            H, surprisal = forward_ns_overlapping(
                model,
                capture,
                input_ids,
                n_layers,
                hidden,
                args.max_seq_len,
                NS_CHUNK_STRIDE,
                ns_context_floor,
            )
            ns_pop.update(H)
            sink_mask = _sink_mask(H)  # (L, T_full) per-(layer,position) — concern #1
            special_mask = _special_token_mask(tokenizer, input_ids)  # (T_full,)
            blob = {
                "item": seq["item"],
                "H_fp16": H.to(torch.float16),  # (L, T_full, hidden)
                "surprisal": surprisal,  # (T_full,)
                "sink_mask": sink_mask,  # (L, T_full) bool
                "special_mask": special_mask,  # (T_full,) bool
                # PRIMARY gold-Penn clause-opener mask (plan §11 syntactic_mask_ns):
                # first terminal under S/SBAR OR CC/IN. The H3 syntactic strata read this.
                "clause_opener_mask": clause_mask,  # (T_full,) bool
                # Wordlist proxy companion — the A11 gold-vs-proxy cross-check reads
                # this against clause_opener_mask (NOT the NS primary mask).
                "clause_opener_mask_wordlist_proxy": proxy_mask,  # (T_full,) bool
                "word_end_idx": torch.tensor(word_end_idx, dtype=torch.long),
                "input_ids": input_ids[0].cpu(),
            }
            torch.save(blob, ns_raw_dir / f"seq_{seq['item']}.pt")
            ns_seq_meta.append(
                {"item": seq["item"], "n_tokens": int(H.shape[1]), "n_chunks": n_chunks}
            )
            logger.info(
                "[NS %d/%d] item=%s T_full=%d n_chunks=%d",
                i,
                len(ns_corpus["sequences"]),
                seq["item"],
                H.shape[1],
                n_chunks,
            )

        # ── Broader Pass 1: stream population stats + rank rogue dims ───────────
        phase("stream_broader_pass1")
        b_pop = WelfordDimStats(n_layers, hidden)
        # Reservoir-sample raw token vectors over the FULL broader stream so the
        # random baseline is drawn from the WHOLE streamed population — not the
        # bounded broader_raw subset the analyzer would otherwise re-concatenate
        # (wrong distribution + a >50 GB all-at-once materialization risk; #744
        # random-pair-memory concern). Fixed ~4 GB regardless of stream length.
        b_reservoir = ReservoirVectorPool(n_layers, hidden, BROADER_RANDOM_POOL, SEED)
        # Accumulate per-layer raw activations ONLY long enough to rank rogue
        # dims; ranking needs the raw population variance per dim, which the
        # Welford sums already give — so rank from the finalized sums (var) plus
        # an online max/|contribution| accumulator would be ideal, but the
        # default metric ("raw_variance") is recoverable directly from Welford.
        for i, seq in enumerate(broader_corpus["sequences"], 1):
            enc = tokenizer(
                seq["text"], return_tensors="pt", truncation=True, max_length=args.max_seq_len
            )
            H, _ = forward_sequence(model, tokenizer, enc["input_ids"], capture, n_layers)
            b_pop.update(H)
            b_reservoir.update(H)
            if i % 50 == 0:
                logger.info("[broader pass1 %d/%d]", i, len(broader_corpus["sequences"]))
        b_mu, b_sigma = b_pop.finalize()  # (L, hidden) each
        # Rogue-dim ranking from the raw-variance population statistic (concern
        # #3): per layer, top-k dims by RAW variance = sigma**2 (NOT standardized
        # variance, which is degenerate after z-scoring).
        b_var = b_sigma**2  # (L, hidden) raw per-dim variance
        b_rogue_idx = torch.stack(
            [
                torch.topk(b_var[li], min(args.rogue_topk, hidden)).indices.sort().values
                for li in range(n_layers)
            ]
        )  # (L, k)

        # Broader random baseline from the reservoir pool, per flavor, computed
        # ONCE over the full-stream sample under the FIXED population stats — the
        # analyzer reads this artifact instead of concatenating broader_raw.
        b_pool = b_reservoir.pool().float()  # (L, n_pool, hidden) raw
        b_flavors = make_flavors_from_stats(b_pool, b_mu, b_sigma, b_rogue_idx)
        broader_random = {
            "corpus": "broader",
            "n_pool": int(b_pool.shape[1]),
            "n_pairs": RANDOM_BASELINE_N_PAIRS,
            "seed": SEED,
            "per_flavor": {
                flavor: random_baseline(H, RANDOM_BASELINE_N_PAIRS, SEED).tolist()
                for flavor, H in b_flavors.items()
            },
        }
        torch.save(broader_random, out_dir / "broader_random_pairs.pt")

        # NS population stats + rogue dims (from the retained dump, recompute-free).
        ns_mu, ns_sigma = ns_pop.finalize()
        ns_var = ns_sigma**2
        ns_rogue_idx = torch.stack(
            [
                torch.topk(ns_var[li], min(args.rogue_topk, hidden)).indices.sort().values
                for li in range(n_layers)
            ]
        )

        # ── NS per-sequence summaries (from the retained raw dump) ──────────────
        phase("summarize_ns")
        ns_summaries = []
        for seq in ns_corpus["sequences"]:
            blob = torch.load(ns_raw_dir / f"seq_{seq['item']}.pt", weights_only=False)
            H = blob["H_fp16"].float()
            flavors = make_flavors_from_stats(H, ns_mu, ns_sigma, ns_rogue_idx)
            summ = per_sequence_summary(flavors, steps, k)
            summ["item"] = seq["item"]
            summ["n_tokens"] = int(H.shape[1])
            ns_summaries.append(summ)

        # ── Broader Pass 2: re-forward, emit per-sequence summaries ─────────────
        phase("stream_broader_pass2")
        broader_summaries = []
        for i, seq in enumerate(broader_corpus["sequences"], 1):
            enc = tokenizer(
                seq["text"], return_tensors="pt", truncation=True, max_length=args.max_seq_len
            )
            H, surprisal = forward_sequence(model, tokenizer, enc["input_ids"], capture, n_layers)
            flavors = make_flavors_from_stats(H, b_mu, b_sigma, b_rogue_idx)
            summ = per_sequence_summary(flavors, steps, k)
            summ["doc_id"] = seq["doc_id"]
            summ["n_tokens"] = int(H.shape[1])
            broader_summaries.append(summ)
            # Retain a bounded raw subset for the lens-3 broader spot-check.
            if i <= args.broader_raw_keep:
                sink_mask = _sink_mask(H)
                special_mask = _special_token_mask(tokenizer, enc["input_ids"])
                # Clause-opener proxy on broader: mark positions whose decoded
                # token (case-folded) is a clause-opener word (no gold parses).
                ids = enc["input_ids"][0].tolist()
                clause_mask = torch.tensor(
                    [is_clause_opener(tokenizer.decode([tid])) for tid in ids], dtype=torch.bool
                )
                torch.save(
                    {
                        "doc_id": seq["doc_id"],
                        "H_fp16": H.to(torch.float16),
                        "surprisal": surprisal,
                        "sink_mask": sink_mask,
                        "special_mask": special_mask,
                        "clause_opener_mask": clause_mask,
                        "input_ids": enc["input_ids"][0].cpu(),
                    },
                    broader_raw_dir / f"seq_{seq['doc_id']}.pt",
                )
            if i % 50 == 0:
                logger.info("[broader pass2 %d/%d]", i, len(broader_corpus["sequences"]))

        # ── Persist per-corpus summaries + fixed stats ──────────────────────────
        phase("persist")
        write_json(
            out_dir / "ns_summaries.json",
            {"corpus": "natural_stories", "k": k, "steps": list(steps), "sequences": ns_summaries},
        )
        write_json(
            out_dir / "broader_summaries.json",
            {"corpus": "broader", "k": k, "steps": list(steps), "sequences": broader_summaries},
        )
        torch.save(
            {
                "ns": {"mu": ns_mu, "sigma": ns_sigma, "rogue_idx": ns_rogue_idx},
                "broader": {"mu": b_mu, "sigma": b_sigma, "rogue_idx": b_rogue_idx},
                "rogue_rank_metric": ROGUE_RANK_METRIC,
                "rogue_topk": args.rogue_topk,
            },
            out_dir / "population_stats.pt",
        )
    finally:
        capture.remove()

    manifest = {
        "model": args.model,
        "n_layers": n_layers,
        "hidden": hidden,
        "max_seq_len": args.max_seq_len,
        "ns_chunk_stride": NS_CHUNK_STRIDE,
        "ns_context_floor": ns_context_floor,
        "k": k,
        "steps": list(steps),
        "flavors": list(FLAVORS),
        "rogue_rank_metric": ROGUE_RANK_METRIC,
        "rogue_topk": args.rogue_topk,
        "broader_raw_keep": args.broader_raw_keep,
        "sink_abs_floor": SINK_ABS_FLOOR,
        "sink_median_ratio": SINK_MEDIAN_RATIO,
        "ns_sequences": ns_seq_meta,
        "n_broader_sequences": len(broader_summaries),
        "smoke": args.smoke,
        "corpus_manifest": json.loads((args.corpora_dir / "corpus_manifest.json").read_text()),
        "metadata": reproducibility_metadata(
            {"script": "issue744_dump_and_stream", "smoke": args.smoke}
        ),
    }
    write_json(out_dir / "dump_manifest.json", manifest)
    logger.info("Dump complete in %.1f min -> %s", (time.time() - t0) / 60, out_dir)

    upload_info: dict = {"skipped": True}
    if not args.no_upload:
        phase("upload")
        upload_info = upload_outputs(out_dir, smoke=args.smoke)
        manifest["upload"] = upload_info
        write_json(out_dir / "dump_manifest.json", manifest)

    note = (
        f"issue744 dump {'SMOKE ' if args.smoke else ''}complete: model={args.model}, "
        f"NS={len(ns_summaries)} seqs, broader={len(broader_summaries)} seqs, "
        f"layers={n_layers}, upload={upload_info}"
    )
    write_sentinel("epm:smoke-result" if args.smoke else "epm:results", note)
    run.finish()
    phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
