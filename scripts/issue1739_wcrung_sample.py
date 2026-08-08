"""Path-C context sampler for the #1739 "wcrung" rung: a FRESH random,
held-out WildChat sample in NATURAL conversation units.

Each context is one real WildChat conversation reduced to its natural
prediction unit:

    prefix = every turn BEFORE the conversation's final user turn
    query  = that final user turn's content
    (any trailing assistant reply after it is DISCARDED — the model's own
     answer to `query` is what the rung generates)

Single-turn conversations are legitimate and kept: prefix is empty, so the
instruct render's prefix is the template-injected system block. Those rows
carry ``single_turn: true`` and are counted in the digest, because the
PREFIX arm on them is a bare-context read, not a conversation read — a
caveat the consumer must disclose.

REUSE (not reimplementation): the streaming + filter chain is #1092's own
``issue1092_build_corpus._stream_with_cache`` — it carries the language
FULL-NAME semantics, the string-typed-bool handling, the redaction /
toxicity / moderation filters, the per-filter reject counters, and the
checkpoint-per-chunk + fingerprint-gated resume contract. The render is
``issue1739_wcrung_contexts.render_row_prompt``, which mirrors #1092's
``_render_instruct`` / ``_render_prefix_instruct`` and is already
token-parity-verified against the store's captured renders.

HOLD-OUT IS CONTENT-BASED, NOT ID-BASED — and this is a deliberate
substitution for the brief's stated mechanism. The #1092 manifest's
``prefix_conv_id`` / ``query_conv_id`` are NOT stable conversation
identities: ``issue1092_build_corpus`` mints them as a POSITIONAL COUNTER
over kept results (``"id": f"{source_tag}_{len(results):06d}"``), so
``wildchat_000123`` means "the 123rd conversation that survived the filter
chain in THAT run". Those labels do not re-derive across runs (a different
row_limit / stream_limit / rng / dataset order shifts every one), and
excluding them would give false confidence while admitting the very
conversations the store consumed. So the exclusion set is built from the
#1092 corpus artifacts' TEXT instead:

  * every ``prefix_store`` prefix's FIRST USER TURN  (the identity notion
    #1092's own in-run dedup uses — ``seen_first_turns``),
  * every ``prefix_store`` ``natural_query``          (those conversations'
    own final user queries — the store consumed this axis too),
  * every ``query_store`` ``text``                    (the query axis).

A freshly-streamed conversation is EXCLUDED when its first-user-turn text OR
its final-user-query text matches that set, under exact AND
whitespace/case-normalized comparison. That is genuine conversation-level
disjointness against the ENTIRE manifest on both axes the crossing store
consumed.

CONTENT HYGIENE: logs, digests, and the emitted manifest carry ids, hashes,
counts, and token lengths — NEVER conversation, prefix, or query text.
Unscreened real WildChat user text stays on disk and is referenced by id +
hash only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root onto sys.path (script mode puts only scripts/ there — #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1092_build_corpus.py"
    assert sentinel.exists(), f"repo-root derivation failed: {sentinel} missing"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE any torch/transformers/datasets import

logger = logging.getLogger("issue1739_wcrung_sample")

RUNG = "wcrung"
SPLIT = "eval"
_WS = re.compile(r"\s+")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-contexts", type=int, default=2000, help="target kept contexts")
    ap.add_argument(
        "--stream-limit",
        type=int,
        default=400_000,
        help=(
            "TOTAL WildChat rows examined (bounded probe — a broken filter chain "
            "terminates instead of streaming ~1M rows; #1092 kept 383,980 of 837,989)"
        ),
    )
    ap.add_argument(
        "--oversample",
        type=float,
        default=1.5,
        help="stream this multiple of --n-contexts before exclusion + length filtering",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_1739/wildchat_rung"))
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("data/issue_1739/wcrung_stage"),
        help="MIRROR ROOT for #1092 corpus staging (exclusion-set source)",
    )
    ap.add_argument(
        "--stream-cache-dir",
        type=Path,
        default=Path("data/issue_1739/wcrung_stream_cache"),
        help="per-source kept-pool checkpoint dir (#1092 _stream_with_cache contract)",
    )
    ap.add_argument("--no-resume-stream", action="store_true", help="force a fresh stream")
    ap.add_argument(
        "--max-contexts-probe",
        type=int,
        default=None,
        help="SMOKE: cap kept contexts after all filtering (tiny-real probe)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the REAL branch, then exit 0",
    )
    return ap.parse_args(argv)


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _norm(text: str) -> str:
    """Whitespace/case-normalized form for the second exclusion comparison."""
    return _WS.sub(" ", text).strip().casefold()


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_exclusion_hashes(
    corpus: dict[str, Path], *, min_hashes: int = 1000
) -> tuple[set[str], set[str], dict]:
    """``(exact_hashes, normalized_hashes, digest)`` for the #1092 corpus TEXT.

    Covers BOTH axes the crossing store consumed (prefix first-user-turns +
    prefix natural_queries + every query text). See the module docstring for
    why this replaces the brief's ``prefix_conv_id`` / ``query_conv_id``
    mechanism.
    """
    from scripts.issue1739_wcrung_contexts import _iter_jsonl

    exact: set[str] = set()
    norm: set[str] = set()
    counts = {"prefix_first_user_turn": 0, "prefix_natural_query": 0, "query_text": 0}

    def _add(text: str, key: str) -> None:
        if not isinstance(text, str) or not text.strip():
            return
        exact.add(_h(text))
        norm.add(_h(_norm(text)))
        counts[key] += 1

    for row in _iter_jsonl(corpus["prefix_store.jsonl"]):
        for turn in row.get("prefix_turns") or []:
            if turn.get("role") == "user":
                _add(str(turn.get("content") or ""), "prefix_first_user_turn")
                break  # FIRST user turn only — the builder's own dedup key
        _add(str(row.get("natural_query") or ""), "prefix_natural_query")
    for row in _iter_jsonl(corpus["query_store.jsonl"]):
        _add(str(row.get("text") or ""), "query_text")

    digest = {
        "n_exact_hashes": len(exact),
        "n_normalized_hashes": len(norm),
        "contributions": dict(counts),
        "mechanism": (
            "content hashes of #1092 prefix first-user-turns + prefix natural_queries "
            "+ query texts (exact AND whitespace/case-normalized); NOT prefix_conv_id/"
            "query_conv_id, which are run-local positional counters"
        ),
    }
    if len(exact) < min_hashes:
        raise ValueError(
            f"exclusion set implausibly small ({len(exact)} exact hashes < {min_hashes}) "
            "— the #1092 corpus artifacts did not load as expected; refusing to sample "
            "against a near-empty hold-out set"
        )
    return exact, norm, digest


def natural_unit(turns: list[dict]) -> tuple[list[dict], str] | None:
    """``(prefix_turns, final_user_query)`` for a conversation, or None.

    The unit is split at the LAST user turn: everything before it is the
    prefix, its content is the query, and any trailing assistant reply is
    discarded (the rung generates the model's own answer instead). Returns
    None when the conversation has no user turn with content.
    """
    last_user = None
    for i, t in enumerate(turns):
        if t.get("role") == "user" and str(t.get("content") or "").strip():
            last_user = i
    if last_user is None:
        return None
    prefix = [
        {"role": str(t["role"]), "content": str(t["content"])}
        for t in turns[:last_user]
        if t.get("role") and str(t.get("content") or "").strip()
    ]
    return prefix, str(turns[last_user]["content"])


def sample_contexts(args: argparse.Namespace, tokenizer) -> tuple[list[dict], dict]:
    """Stream WildChat, hold out against #1092, reduce to natural units, render."""
    from scripts.issue1092_build_corpus import (
        WILDCHAT_REPO,
        WILDCHAT_REV,
        _stream_with_cache,
    )
    from scripts.issue1739_wcrung_contexts import render_row_prompt, stage_corpus
    from explore_persona_space.experiments.issue_1739.generation import PROMPT_TOKEN_BUDGET

    corpus = stage_corpus(args.stage_root)
    exact, norm, excl_digest = build_exclusion_hashes(corpus)
    print(
        f"[phase=wcrung_holdout] exclusion set: {excl_digest['n_exact_hashes']} exact / "
        f"{excl_digest['n_normalized_hashes']} normalized hashes "
        f"({excl_digest['contributions']})",
        flush=True,
    )

    want_stream = max(args.n_contexts, int(args.n_contexts * args.oversample))
    stream_stats: dict = {}
    convs = _stream_with_cache(
        WILDCHAT_REPO,
        WILDCHAT_REV,
        rng=random.Random(args.seed),
        row_limit=want_stream,
        stream_limit=args.stream_limit,
        lang_filter="en",
        stats_out=stream_stats,
        cache_dir=args.stream_cache_dir,
        resume=not args.no_resume_stream,
    )
    # Release the streaming IterableDataset + its pyarrow worker refs while the
    # interpreter is HEALTHY. A `datasets` streaming dataset surviving to
    # interpreter shutdown reliably aborts rc=134 (`PyGILState_Release` /
    # `terminate called without an active exception`) AFTER all work completed
    # and every output landed — gotchas.md, #952 r2. The dataset object itself
    # is internal to `_stream_with_cache`, so it is already out of scope here;
    # the explicit collect is what makes its release deterministic rather than
    # finalize-time.
    import gc

    gc.collect()
    print(
        f"[phase=wcrung_stream] kept={stream_stats.get('kept')} "
        f"streamed={stream_stats.get('streamed')} rejects={stream_stats.get('rejects')}",
        flush=True,
    )
    if not convs:
        raise ValueError(
            "WildChat stream kept ZERO conversations — check the filter chain against "
            "real row shapes before retrying (gotchas.md real-corpus streaming filters)"
        )

    rows: list[dict] = []
    drops = Counter()
    n_single_turn = 0
    token_lens: list[int] = []
    # Dedup DURING the draw on the FINAL-USER-QUERY hash (#1768). #1092's stream
    # dedups on the FIRST user turn, which does NOT imply distinct final turns:
    # two multi-turn conversations can share a final user query (and a
    # single-turn conversation can duplicate another's final turn), so exact
    # duplicates survive the upstream screen. Measured live at n=200.
    taken_query_hashes: set[str] = set()
    for conv in convs:
        unit = natural_unit(conv.get("turns") or [])
        if unit is None:
            drops["no_user_turn"] += 1
            continue
        prefix_turns, query = unit

        # Hold-out: exclude on EITHER axis, exact OR normalized.
        first_user = next(
            (str(t["content"]) for t in prefix_turns if t["role"] == "user"),
            query,  # single-turn: the query IS the first user turn
        )
        if (
            _h(first_user) in exact
            or _h(_norm(first_user)) in norm
            or _h(query) in exact
            or _h(_norm(query)) in norm
        ):
            drops["held_out_overlap_with_1092"] += 1
            continue

        qh = _h(query)
        if qh in taken_query_hashes:
            drops["duplicate_final_query_within_sample"] += 1
            continue

        prefix_text, prompt = render_row_prompt(tokenizer, prefix_turns, query)
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok > PROMPT_TOKEN_BUDGET:
            # Load-time length validation (#952): an overlong row is ENGINE-FATAL
            # at vLLM add_request, so drop it here, digest-only.
            drops["over_prompt_budget"] += 1
            continue
        token_lens.append(n_tok)

        single = not prefix_turns
        n_single_turn += int(single)
        taken_query_hashes.add(qh)
        cid = f"{RUNG}-{qh[:16]}"
        rows.append(
            {
                "context_id": cid,
                "source_conv_id": conv.get("id"),  # run-local label; provenance only
                "source": conv.get("source"),
                "prefix_turns": prefix_turns,
                "prefix_text": prefix_text,
                "query": query,
                "prompt": prompt,
                "n_tokens_instruct": n_tok,
                "n_prefix_turns": len(prefix_turns),
                "single_turn": single,
                "first_user_sha256": _h(first_user),
                "query_sha256": _h(query),
                "split": SPLIT,
                "rung": RUNG,
                # Natural units are independent conversations — each context is
                # its own group for group-level folds.
                "group_key": cid,
            }
        )
        if len(rows) >= args.n_contexts:
            break

    if args.max_contexts_probe is not None:
        rows = rows[: args.max_contexts_probe]

    ids = [r["context_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate wcrung context_id (query-hash collision)")
    qh = [r["query_sha256"] for r in rows]
    if len(set(qh)) != len(qh):
        raise ValueError("duplicate query hash in sample — dedup failed")
    if not rows:
        raise ValueError("zero contexts survived hold-out + length filtering")

    digest = {
        "rung": RUNG,
        "split": SPLIT,
        "n_contexts": len(rows),
        "n_requested": args.n_contexts,
        "n_streamed_kept": len(convs),
        "n_single_turn": n_single_turn,
        "single_turn_frac": round(n_single_turn / len(rows), 4),
        "drops": dict(drops),
        "token_len_min": min(token_lens) if token_lens else None,
        "token_len_max": max(token_lens) if token_lens else None,
        "prompt_token_budget": PROMPT_TOKEN_BUDGET,
        "n_prefix_turns_hist": dict(sorted(Counter(r["n_prefix_turns"] for r in rows).items())),
        "holdout": excl_digest,
        "stream": {
            "repo": WILDCHAT_REPO,
            "revision": WILDCHAT_REV,
            "lang_filter": "en",
            "row_limit": want_stream,
            "stream_limit": args.stream_limit,
            "seed": args.seed,
            "stats": stream_stats,
        },
    }
    return rows, digest


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    if args.import_check:
        from scripts.issue1092_build_corpus import (  # noqa: F401
            WILDCHAT_REPO,
            WILDCHAT_REV,
            _stream_with_cache,
        )
        from scripts.issue1739_wcrung_contexts import (  # noqa: F401
            _iter_jsonl,
            render_row_prompt,
            stage_corpus,
        )
        from explore_persona_space.experiments.issue_1739.generation import (  # noqa: F401
            PROMPT_TOKEN_BUDGET,
            get_tokenizer,
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401

        print("[import-check] OK: all deferred imports resolved", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        return 0

    from explore_persona_space.experiments.issue_1739.generation import get_tokenizer

    tokenizer = get_tokenizer()
    rows, digest = sample_contexts(args, tokenizer)
    digest["git_commit"] = _git_commit()

    _write_json_atomic(args.out_root / "contexts" / f"{RUNG}.json", {"rows": rows, **digest})
    _write_json_atomic(args.out_root / "contexts" / f"{RUNG}_digest.json", digest)

    print(
        f"[phase=wcrung_contexts] contexts={digest['n_contexts']} "
        f"single_turn={digest['n_single_turn']} ({digest['single_turn_frac']}) "
        f"drops={digest['drops']} tok=[{digest['token_len_min']},{digest['token_len_max']}]",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
