"""Issue #1689 follow-up round ``real-u2-capture`` — Phase A0 (corpus filter).

Stream LMSYS-Chat-1M + WildChat-1M, keep multi-turn (>=2 user turn) conversations
via the #1738 predicate + near-dupe gate, and stratified-sample N conversations
for the real-u2 arm. Each kept conversation is truncated to its FIRST exchange:
[u1, a1, u2] — matching the parent user_slot_recapture rig's row shape.

Reuses #1738's ``_multiturn_context`` + ``DfFilteredNearDupeGate`` verbatim
(never re-implemented) — the corpus-filter recipe #1738 validated at 626k
eligible conversations across both corpora.

Content hygiene: never prints raw conversation text — only counts, sha digests,
conv_ids (LMSYS/WildChat carry unscreened real-user text; `.claude/rules/gotchas.md`
§ real-corpus streaming filters). Digest-only per CLAUDE.md § harmful-content
data hygiene.

Output: ``data/issue_1689/real_u2_capture/corpus/real_multiturn_first_exchange.jsonl``
with rows ``{conv_id, corpus, u1, a1, u2_real, depth_original, source_hash}``.

Smoke: ``--smoke`` short-circuits to a bounded probe (kept-cap N=20, total-streamed
cap 5000) that terminates in seconds; the same code path runs full (larger caps).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _ensure_repo_root_on_syspath() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    assert (repo_root / "scripts" / "issue1689_common.py").exists(), repo_root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()

from collections import defaultdict

from scripts.issue1738_multiturn_generate_capture import (  # noqa: E402
    _multiturn_context,
)
from scripts.issue779_ffc_n1m_generate_capture import _char_ngrams, _norm  # noqa: E402

# Corpus pins — verbatim from #1738 / #779.
LMSYS_REPO = "lmsys/lmsys-chat-1m"
WILDCHAT_REPO = "allenai/WildChat-1M"

# Filter constants — verbatim from #1738 (near-dupe ngram/Jaccard/df-cap) +
# #779 (token budget).
NEAR_DUPE_NGRAM = 5
NEAR_DUPE_JACCARD = 0.8
NEAR_DUPE_DF_FRAC = 0.05
PROMPT_TOKEN_BUDGET = 7104  # MAX_MODEL_LEN=8192 - GEN_MAX_TOKENS=1024 - LENGTH_MARGIN=64
CORPUS_SEED = 1689

# Kept-cap floor — a below-floor pool is inadequate for the design.
KILL_TOTAL_ELIGIBLE = 5000


class IncrementalNearDupeGate:
    """Incremental near-dupe gate — the same exact-normalized + char-ngram
    Jaccard semantics as ``scripts.issue779_ffc_n1m_generate_capture.NearDupeGate``,
    but built by APPENDING targets one at a time rather than rebuilt from
    scratch per row (round-1 Major #4: the O(n^2) rebuild would take hours
    over ~15,200 rows).

    Contract:
      * ngram=5, jaccard>=0.8 (verbatim from #1738/#779 constants).
      * ``add_target(text)`` extends the running inverted index; ``is_dupe(text)``
        checks against ALL previously-added targets (Jaccard on FULL gram
        sets — near-dupes still surface via rare shared grams).
      * O(1) amortized per ``add_target`` + O(candidates ∩ shared_grams) per
        ``is_dupe``, so the whole 15,200-row filter is O(n * avg_shared_grams),
        not O(n^2).

    NOTE: the parent ``DfFilteredNearDupeGate`` in #1738 additionally
    applies a document-frequency cap on the candidate index (grams indexing
    >df_frac of targets are dropped from the INDEX). We deliberately DO NOT
    apply that df-cap here for two reasons: (a) it is only a candidate-set
    optimization — the exact + Jaccard semantics are identical — and (b)
    the parent computes the cap at __init__ from the full target count,
    which is not available in an incremental build. Screening cost stays
    bounded because is_dupe short-circuits at the first Jaccard-passing
    candidate.
    """

    def __init__(
        self,
        ngram: int = NEAR_DUPE_NGRAM,
        thresh: float = NEAR_DUPE_JACCARD,
    ) -> None:
        self.ngram = int(ngram)
        self.thresh = float(thresh)
        self.exact: set[str] = set()
        self.target_ngrams: list[frozenset[str]] = []
        self.inv: dict[str, set[int]] = defaultdict(set)
        self.n_exact_drop = 0
        self.n_near_drop = 0

    def add_target(self, text: str) -> None:
        n = _norm(text)
        self.exact.add(n)
        g = _char_ngrams(n, self.ngram)
        ti = len(self.target_ngrams)
        self.target_ngrams.append(g)
        for ng in g:
            self.inv[ng].add(ti)

    def is_dupe(self, prompt: str) -> bool:
        n = _norm(prompt)
        if n in self.exact:
            self.n_exact_drop += 1
            return True
        g = _char_ngrams(n, self.ngram)
        if not g:
            return False
        cand: set[int] = set()
        for ng in g:
            cand |= self.inv.get(ng, set())
        for ti in cand:
            tg = self.target_ngrams[ti]
            inter = len(g & tg)
            if inter == 0:
                continue
            union = len(g) + len(tg) - inter
            if union and inter / union >= self.thresh:
                self.n_near_drop += 1
                return True
        return False

    def stats(self) -> dict:
        return {
            "ngram": self.ngram,
            "jaccard_thresh": self.thresh,
            "n_targets": len(self.target_ngrams),
            "n_exact_drop": self.n_exact_drop,
            "n_near_drop": self.n_near_drop,
            "impl": "incremental_exact_jaccard",
        }


def _source_hash(row, messages: list[dict]) -> str:
    """Row identity — WildChat's own hash where present, else content sha."""
    ch = row.get("conversation_hash")
    if isinstance(ch, str) and ch:
        return f"wc:{ch}"
    plain = "\n".join(f"{t['role']}: {t['content']}" for t in messages)
    key = " ".join(plain.lower().split())
    return "sha:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _first_exchange(messages: list[dict]) -> tuple[str, str, str] | None:
    """Return (u1, a1, u2) truncated to the first exchange — 3 turns after
    _multiturn_context has already asserted alternating u/a/u/... with depth>=2.
    """
    if len(messages) < 3:
        return None
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    if messages[2]["role"] != "user":
        return None
    return messages[0]["content"], messages[1]["content"], messages[2]["content"]


def _language_field(row: dict) -> str | None:
    """Extract a language name / code — WildChat/LMSYS store FULL names."""
    lang = row.get("language")
    if isinstance(lang, str) and lang.strip():
        return lang.strip()
    return None


def _stream_corpus(
    repo_id: str,
    max_scan: int,
    keep_english_only: bool,
    keep_target: int,
) -> tuple[list[dict], dict]:
    """Stream one corpus and return kept first-exchange rows + reject counters.

    Kept cap ``keep_target`` AND total-streamed cap ``max_scan`` bound the loop.
    Real-corpus streaming presumption: ≥1h presumed floor (see #1092); this
    phase is short-bounded by BOTH caps so no per-chunk checkpoint is required.
    """
    from datasets import load_dataset

    ds = load_dataset(repo_id, split="train", streaming=True)
    kept: list[dict] = []
    reject_counts: dict[str, int] = {
        "bad_shape": 0,
        "not_english": 0,
        "not_multiturn": 0,
        "no_first_exchange": 0,
    }
    scanned = 0
    for row in ds:
        if scanned >= max_scan or len(kept) >= keep_target:
            break
        scanned += 1
        # Language filter — WildChat/LMSYS store FULL names ("English"/"Spanish"),
        # not ISO codes (gotchas.md § real-corpus streaming filters).
        if keep_english_only:
            lang = _language_field(row)
            if lang is None or lang.lower() != "english":
                reject_counts["not_english"] += 1
                continue
        parsed, reason = _multiturn_context(row)
        if parsed is None:
            reject_counts["not_multiturn"] += 1
            continue
        exch = _first_exchange(parsed["messages"])
        if exch is None:
            reject_counts["no_first_exchange"] += 1
            continue
        u1, a1, u2 = exch
        conv_id = (
            row.get("conversation_id") or row.get("conversation_hash") or f"{repo_id}#{scanned}"
        )
        kept.append(
            {
                "conv_id": str(conv_id),
                "corpus": repo_id.split("/")[-1],
                "u1": u1,
                "a1": a1,
                "u2_real": u2,
                "depth_original": parsed["depth"],
                "source_hash": _source_hash(row, parsed["messages"]),
            }
        )
    return kept, {"scanned": scanned, **reject_counts, "kept_pre_dedupe": len(kept)}


def _plain_render_for_ndg(row: dict) -> str:
    """Match #1738's near-dupe render (`_plain_render` over the messages list)."""
    return f"user: {row['u1']}\nassistant: {row['a1']}\nuser: {row['u2_real']}"


def _token_length(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _rendered_token_length(row: dict, tokenizer) -> int:
    """Approximate token cost of the rendered [u1, a1, u2] chat template.

    Used only for the PROMPT_TOKEN_BUDGET filter — a conservative upper bound
    on how long the eventual rendered row will be.
    """
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": row["u1"]},
            {"role": "assistant", "content": row["a1"]},
            {"role": "user", "content": row["u2_real"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return _token_length(text, tokenizer)


def filter_and_sample(
    *,
    keep_target: int,
    max_scan_per_corpus: int,
    out_path: Path,
    smoke: bool,
    seed: int = CORPUS_SEED,
) -> dict:
    """Stream both corpora, filter, near-dupe screen, token-budget filter, and
    stratified-sample ``keep_target`` conversations. Emit ``out_path``. Return
    a manifest dict with realized grain + reject counters.
    """
    import numpy as np
    from transformers import AutoTokenizer

    from scripts.issue1689_common import MODEL_BASE

    print(f"[phase=corpus] streaming LMSYS + WildChat (keep_target={keep_target})", flush=True)

    # Estimate per-corpus keep quotas: stream ~2x the target from each, then
    # stratify. #1738 measured LMSYS ≈ 426k / WildChat ≈ 200k eligible.
    per_corpus_target = keep_target * 2

    lmsys_rows, lmsys_stats = _stream_corpus(
        LMSYS_REPO, max_scan_per_corpus, keep_english_only=True, keep_target=per_corpus_target
    )
    print(
        f"[corpus] lmsys: scanned={lmsys_stats['scanned']} kept_pre_dedupe={len(lmsys_rows)} "
        f"not_english={lmsys_stats['not_english']} not_multiturn={lmsys_stats['not_multiturn']}",
        flush=True,
    )

    wc_rows, wc_stats = _stream_corpus(
        WILDCHAT_REPO, max_scan_per_corpus, keep_english_only=True, keep_target=per_corpus_target
    )
    print(
        f"[corpus] wildchat: scanned={wc_stats['scanned']} kept_pre_dedupe={len(wc_rows)} "
        f"not_english={wc_stats['not_english']} not_multiturn={wc_stats['not_multiturn']}",
        flush=True,
    )

    total_eligible = len(lmsys_rows) + len(wc_rows)
    if not smoke and total_eligible < KILL_TOTAL_ELIGIBLE:
        raise RuntimeError(
            f"total eligible pool {total_eligible} < kill floor {KILL_TOTAL_ELIGIBLE}; "
            "corpus is inadequate — scope must change"
        )
    if smoke and total_eligible == 0:
        raise RuntimeError("smoke: total eligible pool is 0 — corpus streaming filter is broken")

    # Near-dupe screen: use #1738's DfFilteredNearDupeGate keyed on a plain
    # `user: ... assistant: ... user: ...` render.
    pool = lmsys_rows + wc_rows
    if smoke and len(pool) < 5:
        after_dedupe = pool
        n_dedupe_dropped = 0
    else:
        renders = [_plain_render_for_ndg(r) for r in pool]
        # Incremental gate — round-2 Major #4 fix. The prior implementation
        # rebuilt DfFilteredNearDupeGate FROM SCRATCH per row (O(n^2) over
        # keep_target*2 = 15,200 candidate rows at production, projected
        # hours). ``IncrementalNearDupeGate.add_target`` extends ONE inverted
        # index; ``is_dupe`` short-circuits on the first Jaccard match. Same
        # exact-normalized + char-ngram Jaccard semantics as the parent
        # ``NearDupeGate`` (#779). See the class docstring above for the
        # deliberate df-cap omission.
        gate = IncrementalNearDupeGate(ngram=NEAR_DUPE_NGRAM, thresh=NEAR_DUPE_JACCARD)
        kept: list[dict] = []
        n_dedupe_dropped = 0
        for row, rendered in zip(pool, renders, strict=True):
            if gate.is_dupe(rendered):
                n_dedupe_dropped += 1
                continue
            kept.append(row)
            gate.add_target(rendered)
        after_dedupe = kept
    print(
        f"[corpus] near-dupe: pool={len(pool)} kept={len(after_dedupe)} dropped={n_dedupe_dropped}",
        flush=True,
    )

    # Token-budget filter (rendered chat template).
    #
    # Round-2 bug-class sweep (Major #4 sibling #2): the prior implementation
    # swallowed tokenizer errors into the SAME ``dropped_over_budget`` counter
    # as legitimate over-budget rows, conflating "rendered fine, exceeded
    # budget" with "tokenizer crashed on this row". Split the counters so a
    # tokenizer-failure spike is visible.
    tok = AutoTokenizer.from_pretrained(MODEL_BASE)
    dropped_over_budget = 0
    dropped_tokenize_error = 0
    after_budget: list[dict] = []
    for r in after_dedupe:
        try:
            n_tok = _rendered_token_length(r, tok)
        except Exception as exc:
            dropped_tokenize_error += 1
            print(
                f"[corpus] tokenize-error dropping row conv_id={r.get('conv_id')!r}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        if n_tok > PROMPT_TOKEN_BUDGET:
            dropped_over_budget += 1
            continue
        after_budget.append(r)
    print(
        f"[corpus] token-budget: kept={len(after_budget)} "
        f"dropped_over_budget={dropped_over_budget} "
        f"dropped_tokenize_error={dropped_tokenize_error} "
        f"budget={PROMPT_TOKEN_BUDGET}",
        flush=True,
    )

    n_after_filter = len(after_budget)
    if n_after_filter == 0:
        raise RuntimeError("token-budget filter dropped every row; corpus unusable at this budget")

    # Stratified sample: draw keep_target proportional to per-corpus survival.
    lmsys_kept = [r for r in after_budget if r["corpus"] == LMSYS_REPO.split("/")[-1]]
    wc_kept = [r for r in after_budget if r["corpus"] == WILDCHAT_REPO.split("/")[-1]]

    rng = np.random.default_rng(seed)
    if smoke:
        # Smoke: bypass the exact target if pool is small — take min(keep_target, pool).
        target = min(keep_target, len(after_budget))
        idx = rng.permutation(len(after_budget))[:target]
        sampled = [after_budget[i] for i in idx]
    else:
        total = len(lmsys_kept) + len(wc_kept)
        lmsys_target = min(
            len(lmsys_kept), int(round(keep_target * len(lmsys_kept) / max(1, total)))
        )
        wc_target = min(len(wc_kept), keep_target - lmsys_target)
        idx_l = rng.permutation(len(lmsys_kept))[:lmsys_target]
        idx_w = rng.permutation(len(wc_kept))[:wc_target]
        sampled = [lmsys_kept[i] for i in idx_l] + [wc_kept[i] for i in idx_w]
        rng.shuffle(sampled)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for row in sampled:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp, out_path)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "keep_target": keep_target,
        "n_realized": len(sampled),
        "n_eligible": {"lmsys": len(lmsys_rows), "wildchat": len(wc_rows)},
        "n_after_neardupe": len(after_dedupe),
        "n_after_token_filter": len(after_budget),
        "dropped_neardupe": n_dedupe_dropped,
        "dropped_over_budget": dropped_over_budget,
        "dropped_tokenize_error": dropped_tokenize_error,
        "smoke": smoke,
        "corpus_repos": [LMSYS_REPO, WILDCHAT_REPO],
        "prompt_token_budget": PROMPT_TOKEN_BUDGET,
        "near_dupe_ngram": NEAR_DUPE_NGRAM,
        "near_dupe_jaccard": NEAR_DUPE_JACCARD,
        "near_dupe_df_frac": NEAR_DUPE_DF_FRAC,
        "per_corpus_split": {"lmsys": len(lmsys_kept), "wildchat": len(wc_kept)},
    }
    manifest_path = out_path.parent / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"[phase=corpus] done: n_realized={len(sampled)} lmsys={len(lmsys_kept)} "
        f"wildchat={len(wc_kept)}",
        flush=True,
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "issue_1689"
        / "real_u2_capture"
        / "corpus"
        / "real_multiturn_first_exchange.jsonl",
    )
    ap.add_argument("--keep-target", type=int, default=3800)
    ap.add_argument("--max-scan", type=int, default=200_000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import + exit (Axis-1 import-resolution leg)",
    )
    args = ap.parse_args()

    if args.import_check:
        import numpy  # noqa: F401
        from datasets import load_dataset  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401

        print("[corpus] import-check OK", flush=True)
        return 0

    if args.smoke:
        args.keep_target = 20
        args.max_scan = 5000

    filter_and_sample(
        keep_target=args.keep_target,
        max_scan_per_corpus=args.max_scan,
        out_path=args.out,
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
