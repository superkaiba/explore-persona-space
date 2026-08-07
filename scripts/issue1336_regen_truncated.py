#!/usr/bin/env python
"""Issue #1336 — prefix-continuation regeneration of cap-truncated rows.

Extends ONLY the rows whose stored ``finish_reason == "length"`` in an
existing generation pool (an ``issue1336_gen_answers.py`` output cell) by
PREFILLING the stored truncated answer into the generation prompt and
sampling only the TAIL, writing to a SUFFIXED sibling cell
(``<cell>_cont<total-answer-budget>``) locally and on the Hub so the
1024-cap originals are never overwritten and concurrent consumers of the
original pools are unaffected.

Why prefix-continuation (2026-08-06 round on #1336): the prior mechanism
re-generated each truncated row from the ORIGINAL prompt at a doubled cap,
which at temperature 1.0 is a FRESH DRAW, not a continuation — the pilot
measured prefix_match 0/5 with regenerated answers at mean 885 tokens
against source rows truncated at >=1024 tokens, so splicing them in would
replace long answers with typical-length ones: a selection-on-outcome bias
on the exact length-conditioned quantity at issue. The falsifying evidence
is durable at ``issue1336_rlvr_ladder/raw_completions/generation/rlvr/
gsm8k_test1319_mt2048/audit.json``; the resampled ``<cell>_mt<cap>`` pilot
cells are RETAINED (never overwritten — continuation writes DISTINCT
``<cell>_cont<total>`` keys, so the resume/skip predicate and
``gen._try_hf_resume`` can never silently adopt the falsified resample
rows). Prefilling makes the stored prefix byte-identical BY CONSTRUCTION,
and the tail is a sample from ``P(tail | prompt, stored prefix)`` — exactly
the conditional the uncapped run would have sampled from.

Session constraints (2026-08-06 prefix-continuation round on #1336):
  1. ``cm.SAMPLING`` / ``cm.MAX_MODEL_LEN`` are NEVER mutated — the tail cap
     is per-invocation CLI state (``--tail-max-tokens`` / ``--max-model-len``),
     so a concurrently queued job that pulls this branch keeps the recipe
     constants byte-identical.
  2. A row with ``finish_reason == "length"`` hit the ORIGINAL cap, but what is
     STORED is the answer after ``_truncate_role_headers`` — so length-finish
     does NOT imply the stored text sits at the cap. Rows are PARTITIONED at
     load (``partition_truncated_by_stored_length``): stored ~at cap =>
     continuable (a uniform tail cap gives every one the same total budget
     ``original_cap + tail_cap``, no per-row SamplingParams); stored materially
     short => the answer already terminated and the row is SKIPPED and
     reported (measured 25.4% of length-finish rows on base/lmsys23k).
  3. The PROMPT budget must not move: in continuation mode the effective
     prompt carries the stored answer, so the resample-mode assert
     (``max_model_len - max_tokens == PROMPT_TOKEN_BUDGET``) is RE-DERIVED —
     not deleted — to ``max_model_len - tail_cap == PROMPT_TOKEN_BUDGET +
     original_cap`` (5120 - 1024 == 3072 + 1024 for the production
     invocation); see ``assert_continuation_budget``.
  4. Existing pools are read-only inputs; output lands under the suffixed
     continuation cell key only
     (``data/issue_1336/gen/<slug>/<cell>_cont<total>/`` +
     ``.../generation/<slug>/<cell>_cont<total>/`` on the Hub).

Rows that finished naturally under the original cap are NOT regenerated (the
cap never bound them). Per-row ``prefix_match`` is an ASSERT (True by
construction — a False is a bug, not a statistic); the audit keeps
``prefix_common_frac`` as a residual diagnostic only.

Row filters (render validation incl. the 2048-token rendered-conversation
budget), stop handling, chunked generation, and upload machinery are all
IMPORTED from ``issue1336_gen_answers`` (single source of truth): a
continued answer that now exceeds the render budget is DROPPED with
``<fmt>:over_token_budget`` exactly as the production keep-filter would drop
it — EXPECTED to fire more often (answers are longer by construction) — and
the audit reports that split plus the realized
``over_token_budget_drop_rate``.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import statistics
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1336_gen_answers as gen  # noqa: E402  (module top: dotenv + vLLM spawn setdefault)
from issue1336_render import RENDERERS, validate_render  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402


def cont_cell_key(cell: str, total_answer_budget: int) -> str:
    """Suffixed sibling cell key for prefix-continuation output.

    DISTINCT from the retired resample-mode key (``<cell>_mt<cap>``) by
    construction: the pilot already wrote falsified resampled rows under
    ``<cell>_mt2048`` both locally and on the Hub, and reusing that key
    would let the resume/skip predicate + ``gen._try_hf_resume`` silently
    adopt them. The pilot cells stay in place as the gate-verdict evidence;
    continuation output can never collide with them.
    """
    return f"{cell}_cont{total_answer_budget}"


def assert_continuation_budget(max_model_len: int, tail_max_tokens: int) -> int:
    """Re-derived engine-budget invariant; returns the total answer budget.

    Resample mode asserted ``max_model_len - max_tokens ==
    cm.PROMPT_TOKEN_BUDGET`` (only the answer cap moves). In continuation
    mode the effective prompt ALSO carries the stored answer
    (``original_cap`` tokens), so the engine prompt space must equal
    ``PROMPT_TOKEN_BUDGET + original_cap``:

        max_model_len - tail_max_tokens == PROMPT_TOKEN_BUDGET + original_cap
        (5120 - 1024 == 3072 + 1024 for the production invocation)

    Keeping the OLD assert unchanged would refuse the correct invocation;
    deleting it would let a wrong cap through silently — hence re-derived.
    """
    original_cap = cm.SAMPLING["max_tokens"]
    assert tail_max_tokens > 0, f"--tail-max-tokens must be positive, got {tail_max_tokens}"
    prompt_space = max_model_len - tail_max_tokens
    expected = cm.PROMPT_TOKEN_BUDGET + original_cap
    assert prompt_space == expected, (
        f"engine prompt space {prompt_space} (= --max-model-len {max_model_len} - "
        f"--tail-max-tokens {tail_max_tokens}) != PROMPT_TOKEN_BUDGET + original cap = "
        f"{cm.PROMPT_TOKEN_BUDGET} + {original_cap} = {expected}: pass --max-model-len = "
        "PROMPT_TOKEN_BUDGET + original_cap + --tail-max-tokens so the admissible-prompt "
        "population is unchanged AND the stored prefix fits exactly (budget frozen)"
    )
    return original_cap + tail_max_tokens


#: Re-tokenization tolerance for a stored at-cap answer, in tokens.
#: A stored answer is TEXT: the engine cut it at exactly ``original_cap``
#: tokens, but detokenize -> retokenize does NOT round-trip token counts at
#: BPE seams, so re-tokenizing the stored string lands a few tokens either
#: side of the cap. MEASURED on the two local #1336 pools (2026-08-07,
#: 505 cap-truncated rows across rlvr/lmsys5k + dpo/lmsys5k): 452 rows exact,
#: the rest spread over deltas -6..+2 — i.e. 12.3% / 8.6% of truncated rows
#: would fail an exact-equality assert. 16 sits an order of magnitude above
#: the observed drift while staying far below any real role-header cut (a cut
#: at a role marker in a cap-truncated answer removes tens-to-hundreds of
#: tokens), so the band separates the two causes cleanly.
STORED_CAP_TOKEN_TOLERANCE = 16


#: A stored answer longer than this multiple of the cap cannot be explained by BPE
#: round-trip drift and means the pool is not what this script assumes (a wrong cap, a
#: mismatched tokenizer, a corrupted row). Fail loud rather than continue from it.
STORED_CAP_ABSURD_MULTIPLE = 1.5


def partition_truncated_by_stored_length(
    tokenizer, rows: list[dict], original_cap: int
) -> tuple[list[dict], list[dict], dict]:
    """Split ``finish_reason == "length"`` rows into CONTINUABLE and ROLE-HEADER-STRIPPED.

    ``finish_reason == "length"`` means the engine stopped at ``original_cap`` TOKENS —
    but what is persisted is the answer AFTER ``_truncate_role_headers`` removed any
    hallucinated next turn. Those are two different things, and conflating them is the
    trap this function exists to close:

    * **Continuable** — stored text re-tokenizes to ~the cap. The answer really was cut
      mid-sentence by the budget, so ``P(tail | prompt, stored prefix)`` is exactly the
      continuation an uncapped run would have sampled. Regenerate these.
    * **Role-header-stripped** — stored text is materially SHORTER than the cap because
      the model finished its answer, started a fake new turn, and the strip removed it.
      The stored answer already TERMINATES; it is not length-censored in the sense this
      regen exists to fix. Appending a continuation would fabricate text after a
      completed answer — a worse artifact than the censoring. SKIP these, and report
      them, so the regen's denominator is honest.

    MEASURED on base/lmsys23k (2026-08-07, the production basis cell): of 3,220
    length-finish rows, 2,401 continuable and 818 role-header-stripped (25.4%), the
    stripped ones at min 0 / median 61 / p75 210 tokens. An earlier version of this
    function ASSERTED every row sat within ``STORED_CAP_TOKEN_TOLERANCE`` of the cap and
    aborted the whole cell on the first stripped row (prompt_idx=5020, 89 tokens, −935
    off cap). One quarter of the population is not an edge case to fail on — it is a
    class to classify.

    Note 652 of those 818 were KEPT rows, i.e. they sit in the analyzed pool with a
    length-finish flag and a median ~61-token answer. That is a property of the parent
    generation pipeline, not something this regen can fix; the returned stats surface it.

    Returns ``(continuable, stripped, stats)``. Raises only for an ABSURD stored length
    (> ``STORED_CAP_ABSURD_MULTIPLE`` x cap), which no drift or strip can produce.
    """
    continuable: list[dict] = []
    stripped: list[dict] = []
    drifts: list[int] = []
    stripped_lens: list[int] = []
    floor = original_cap - STORED_CAP_TOKEN_TOLERANCE
    ceiling = int(original_cap * STORED_CAP_ABSURD_MULTIPLE)
    for r in rows:
        n = len(tokenizer(r["response"], add_special_tokens=False)["input_ids"])
        assert n <= ceiling, (
            f"stored answer for prompt_idx={r['prompt_idx']} re-tokenizes to {n} tokens, "
            f"more than {STORED_CAP_ABSURD_MULTIPLE}x the original cap {original_cap} — "
            "no BPE round-trip drift or role-header strip can produce that; the pool, cap, "
            "or tokenizer is not what this script assumes (fail loud, do not continue)"
        )
        if n >= floor:
            continuable.append(r)
            drifts.append(n - original_cap)
        else:
            stripped.append(r)
            stripped_lens.append(n)

    def _pct(vals: list[int], p: float) -> int | None:
        return sorted(vals)[min(len(vals) - 1, int(p * len(vals)))] if vals else None

    return (
        continuable,
        stripped,
        {
            "tolerance": STORED_CAP_TOKEN_TOLERANCE,
            "absurd_multiple": STORED_CAP_ABSURD_MULTIPLE,
            "n_length_finish_rows": len(rows),
            "n_continuable": len(continuable),
            "n_role_header_stripped": len(stripped),
            "n_role_header_stripped_kept": sum(1 for r in stripped if r.get("kept")),
            "role_header_stripped_frac": (len(stripped) / len(rows)) if rows else None,
            # Drift of the CONTINUABLE rows only — how much of the tolerance band the
            # at-cap population actually used. Near-saturation means the round-trip
            # assumption is degrading, not that a row is bad.
            "continuable_drift": {
                "n_exact": sum(1 for d in drifts if d == 0),
                "min": min(drifts) if drifts else None,
                "max": max(drifts) if drifts else None,
                "max_abs": max((abs(d) for d in drifts), default=None),
            },
            "role_header_stripped_tokens": {
                "min": min(stripped_lens) if stripped_lens else None,
                "median": _pct(stripped_lens, 0.5),
                "p75": _pct(stripped_lens, 0.75),
                "max": max(stripped_lens) if stripped_lens else None,
            },
        },
    )


def _assert_prefix_preserved(answer: str, stored: str, prompt_idx) -> None:
    """Continuation invariant: the persisted answer starts with the stored
    prefix byte-identically. True by construction (the tail is APPENDED to
    the stored text); a False is a construction bug, not a statistic to
    count — hence an assert, never a rate."""
    assert answer.startswith(stored), (
        f"prompt_idx={prompt_idx}: continuation answer does not preserve the stored "
        "prefix — construction bug (the tail must be appended to the stored answer)"
    )


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the common character prefix of two strings."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def load_source_pool(slug: str, cell: str) -> list[dict]:
    """Rows of the EXISTING generation pool (local dir if complete, else HF fetch).

    Read-only with respect to regen: the fetch stages the ORIGINAL cell into its
    own local dir (same bytes as the Hub); regen output never writes there.
    """
    src_dir = gen._out_root(False) / slug / cell
    if not ((src_dir / "answers.jsonl").exists() and (src_dir / "audit.json").exists()):
        assert gen._try_hf_resume(slug, cell, src_dir), (
            f"source pool {slug}/{cell}: not local and not complete on the Hub"
        )
    rows = gen._read_jsonl(src_dir / "answers.jsonl")
    assert rows, f"source pool {slug}/{cell} read 0 rows"
    return rows


def regen_cell(
    generate_fn,
    tokenizer,
    slug: str,
    corpus: str,
    *,
    gen_format: str,
    tail_max_tokens: int,
    max_model_len: int,
    upload: bool,
    engine_load_seconds: float | None = None,
) -> dict | None:
    """Continue one cell's cap-truncated rows to the total budget (resume-safe).

    ``generate_fn(texts) -> [(tail_text, finish_reason), ...]`` is the vLLM
    chunked-generation closure in production (built in ``main`` with the
    per-invocation SamplingParams; each text is ``prompt + stored answer``,
    so the returned text is the TAIL only); a stub in the CPU tests. Returns
    the audit dict, or None when the cell was already complete (skip
    predicate mirrors ``issue1336_gen_answers._collect_pending``: done ==
    local outputs exist; Hub-incomplete local outputs re-attempt only the
    upload).
    """
    original_cap = cm.SAMPLING["max_tokens"]
    total_budget = original_cap + tail_max_tokens
    cell = cm.gen_cell_key(corpus, gen_format)
    out_cell = cont_cell_key(cell, total_budget)
    out_dir = gen._out_root(False) / slug / out_cell
    if (out_dir / "answers.jsonl").exists() and (out_dir / "audit.json").exists():
        if upload and not gen._hf_gen_state(slug, out_cell)[0]:
            print(f"[regen] {slug}/{out_cell}: local outputs exist, Hub incomplete — re-uploading")
            gen._upload_gen_outputs(slug, out_cell, out_dir)
        print(f"[regen] skip {slug}/{out_cell} (complete)")
        return None
    if gen._try_hf_resume(slug, out_cell, out_dir):
        print(f"[regen] skip {slug}/{out_cell} (HF-resumed)")
        return None

    fmts_needed = gen._formats_for(corpus, gen_format)
    assert gen_format in fmts_needed, (
        f"gen format {gen_format!r} not licensed for corpus {corpus!r} (formats: {fmts_needed})"
    )
    src_rows = load_source_pool(slug, cell)
    length_finish = [r for r in src_rows if r.get("finish_reason") == "length"]
    # A length-finish row is not automatically continuable: the stored text is
    # post-role-header-strip, so a quarter of them terminate well short of the cap and
    # must be SKIPPED rather than continued (see partition_truncated_by_stored_length).
    trunc, stripped, stored_cap_drift = partition_truncated_by_stored_length(
        tokenizer, length_finish, original_cap
    )
    n_src_trunc_kept = sum(1 for r in trunc if r.get("kept"))
    print(
        f"[regen] {slug}/{cell}: {len(length_finish)}/{len(src_rows)} source rows "
        f"length-finish; {len(trunc)} continuable ({n_src_trunc_kept} kept), "
        f"{len(stripped)} role-header-stripped and SKIPPED "
        f"({stored_cap_drift['n_role_header_stripped_kept']} of them kept) "
        f"— continuing tails at cap {tail_max_tokens}",
        flush=True,
    )

    prompt_builder = {"chat": cm.tulu_prompt, "naturalistic": cm.natural_prompt}[gen_format]
    stop_strings = cm.STOP_STRINGS if gen_format == "chat" else cm.NATURAL_STOP_STRINGS
    truncate_markers = (
        cm.ROLE_HEADER_TRUNCATE if gen_format == "chat" else cm.NATURAL_ROLE_HEADER_TRUNCATE
    )
    if gen_format == "chat" and trunc:
        gen._assert_template_parity(tokenizer, [r["prompt"] for r in trunc[:3]])

    # PREFIX-CONTINUATION: the stored truncated answer is prefilled as part of
    # the prompt, so the engine generates only the tail and the stored prefix
    # is byte-identical by construction (the mechanism change; resample mode —
    # texts built from the bare prompt — was falsified by the pilot).
    texts = [prompt_builder(r["prompt"]) + r["response"] for r in trunc]
    t0 = time.monotonic()
    outs = generate_fn(texts)
    gen_wall = time.monotonic() - t0

    rows: list[dict] = []
    kept_ids: list[int] = []
    drop_reasons: collections.Counter = collections.Counter()
    finish_counts: collections.Counter = collections.Counter()
    kept_answers: list[str] = []
    kept_tok_lens: list[int] = []
    gate_pairs: list[tuple] = []  # (chat, naturalistic) renders of kept rows
    new_answer_tokens: list[int] = []
    tail_answer_tokens: list[int] = []
    prefix_fracs: list[float] = []
    rep3_flags = 0
    for r, (tail_raw, finish) in zip(trunc, outs, strict=True):
        old = r["response"]
        # Role-header truncation scopes to the TAIL only: the stored prefix
        # already survived it at original generation, and truncating the tail
        # alone guarantees prefix preservation by construction.
        tail = gen._truncate_role_headers(tail_raw, truncate_markers)
        answer = old + tail
        full_raw = old + tail_raw
        _assert_prefix_preserved(answer, old, r["prompt_idx"])
        prefix_fracs.append(_common_prefix_len(full_raw, old) / max(1, len(old)))
        finish_counts[finish] += 1
        new_answer_tokens.append(len(tokenizer(full_raw, add_special_tokens=False)["input_ids"]))
        tail_answer_tokens.append(len(tokenizer(tail_raw, add_special_tokens=False)["input_ids"]))
        row = {
            "prompt_idx": r["prompt_idx"],
            "prompt": r["prompt"],
            "response": answer,
            "response_raw_len_chars": len(full_raw),
            "finish_reason": finish,
            "regen_mode": "prefix_continuation",
            "regen_max_tokens": total_budget,
            "tail_max_tokens": tail_max_tokens,
            "tail_len_chars": len(tail_raw),
            "orig_finish_reason": r.get("finish_reason"),
            "orig_kept": bool(r.get("kept")),
            "orig_response_len_chars": len(old),
            "prefix_match": True,  # asserted above (construction invariant), never counted
        }
        reason = None
        fmt_renders: dict = {}
        if not answer.strip():
            reason = "empty_answer"
        else:
            conv = {"conv_id": str(r["prompt_idx"]), "u1": r["prompt"], "a1": answer}
            for fmt in fmts_needed:
                rendered = RENDERERS[fmt](conv, tokenizer)
                reason = validate_render(rendered)
                if reason is not None:
                    reason = f"{fmt}:{reason}"
                    break
                fmt_renders[fmt] = rendered
                if fmt == "chat":
                    span = rendered.spans["a1"]
                    kept_tok_lens.append(span[1] - span[0])
        row["kept"] = reason is None
        row["drop_reason"] = reason
        rows.append(row)
        if reason is None:
            kept_ids.append(r["prompt_idx"])
            kept_answers.append(answer)
            rep3_flags += int(gen._rep3_flag(answer))
            if "naturalistic" in fmt_renders:
                gate_pairs.append((fmt_renders["chat"], fmt_renders["naturalistic"]))
        else:
            drop_reasons[reason] += 1

    render_integrity = None
    if {"chat", "naturalistic"} <= set(fmts_needed) and gate_pairs:
        render_integrity = gen._run_render_integrity(gate_pairs, gen_format, slug, out_cell)
    gen._write_jsonl(out_dir / "answers.jsonl", rows)
    (out_dir / "allowlist.json").write_text(json.dumps(kept_ids) + "\n")
    n_over_budget = sum(v for k, v in drop_reasons.items() if k.endswith("over_token_budget"))
    audit = {
        "kind": "regen_truncated",
        "regen_mode": "prefix_continuation",
        "model": slug,
        "hf_id": cm.MODELS[slug]["hf_id"],
        "corpus": corpus,
        "gen_format": gen_format,
        "source_cell": cell,
        "regen_cell": out_cell,
        "original_cap": original_cap,
        "tail_max_tokens": tail_max_tokens,
        "total_answer_budget": total_budget,
        # The length-finish partition: how many rows were continuable vs
        # role-header-stripped-and-skipped, the skipped rows' length distribution,
        # and how much of STORED_CAP_TOKEN_TOLERANCE the CONTINUABLE prefixes used
        # (detokenize -> retokenize BPE drift). Near-saturation of that band means the
        # at-cap round-trip assumption is degrading, not that a row is bad. The skip
        # count is the regen's honest denominator — it is NOT a failure.
        "stored_length_partition": stored_cap_drift,
        "n_source_rows": len(src_rows),
        "n_source_length_finish": len(length_finish),
        "n_source_truncated": len(trunc),
        "n_source_truncated_kept": n_src_trunc_kept,
        "n_role_header_stripped_skipped": len(stripped),
        "n_regenerated": len(rows),
        "n_kept": len(kept_ids),
        "keep_rate": (len(kept_ids) / len(rows)) if rows else None,
        "drop_reasons": dict(drop_reasons),
        # The render keep-filter's over-budget DROP path fires more often in
        # continuation mode (answers are longer by construction) — report the
        # realized rate explicitly alongside the full drop_reasons split.
        "over_token_budget_drop_rate": (n_over_budget / len(rows)) if rows else None,
        "finish_reasons": dict(finish_counts),
        "cap_hit_rate_at_new_cap": (finish_counts.get("length", 0) / len(rows)) if rows else None,
        # prefix preservation is an ASSERT (every persisted row passed it);
        # the rate is retained for schema continuity and is 1.0 by construction.
        "prefix_match_rate": 1.0 if rows else None,
        "n_prefix_match": len(rows),
        "prefix_common_frac": {
            "mean": statistics.fmean(prefix_fracs) if prefix_fracs else None,
            "min": min(prefix_fracs) if prefix_fracs else None,
        },
        # FULL answer (stored prefix + raw tail), pre-truncation — the
        # length-conditioned quantity downstream consumers care about.
        "new_answer_tokens": {
            "mean": statistics.fmean(new_answer_tokens) if new_answer_tokens else None,
            "median": statistics.median(new_answer_tokens) if new_answer_tokens else None,
            "max": max(new_answer_tokens) if new_answer_tokens else None,
            "total": sum(new_answer_tokens),
        },
        # NEWLY GENERATED tokens (the tail only) — the throughput basis.
        "tail_answer_tokens": {
            "mean": statistics.fmean(tail_answer_tokens) if tail_answer_tokens else None,
            "median": statistics.median(tail_answer_tokens) if tail_answer_tokens else None,
            "max": max(tail_answer_tokens) if tail_answer_tokens else None,
            "total": sum(tail_answer_tokens),
        },
        "kept_rep3_flag_rate": (rep3_flags / len(kept_ids)) if kept_ids else None,
        "kept_distinct_3gram_rate": gen._distinct_3gram_rate(kept_answers),
        "kept_answer_tokens": {
            "mean": statistics.fmean(kept_tok_lens) if kept_tok_lens else None,
            "median": statistics.median(kept_tok_lens) if kept_tok_lens else None,
            "p90": (
                sorted(kept_tok_lens)[int(0.9 * (len(kept_tok_lens) - 1))]
                if kept_tok_lens
                else None
            ),
        },
        "sampling": dict(cm.SAMPLING) | {"max_tokens": tail_max_tokens, "stop": list(stop_strings)},
        "max_model_len": max_model_len,
        "prompt_token_budget_realized": max_model_len - tail_max_tokens,
        "engine_load_seconds": engine_load_seconds,
        "generation_wall_seconds": gen_wall,
        "rows_per_min": (len(rows) / (gen_wall / 60.0)) if rows and gen_wall > 0 else None,
        "new_tokens_per_second": (
            (sum(tail_answer_tokens) / gen_wall) if rows and gen_wall > 0 else None
        ),
        "render_integrity": render_integrity,
        "code_sha": cm.resolve_code_sha(),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        f"[regen] {slug}/{out_cell}: continued {len(rows)} rows, kept {len(kept_ids)}; "
        f"over-budget drops {n_over_budget}/{len(rows)}; "
        f"total-budget hit {finish_counts.get('length', 0)}/{len(rows)}; "
        f"gen wall {gen_wall:.1f}s",
        flush=True,
    )
    if upload:
        gen._upload_gen_outputs(slug, out_cell, out_dir)
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=tuple(cm.MODELS), required=True)
    ap.add_argument(
        "--corpora",
        default=None,
        help="comma list of v2 corpora (default: all 7; wave-1-only corpora are out of scope)",
    )
    ap.add_argument("--gen-format", choices=("chat", "naturalistic"), default="chat")
    ap.add_argument(
        "--tail-max-tokens",
        type=int,
        required=True,
        help="continuation TAIL cap per row; the stored prefix is exactly the original cap, "
        "so every row's total answer budget = original cap + this (1024 -> total 2048)",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        required=True,
        help="engine max_model_len; must equal PROMPT_TOKEN_BUDGET + original cap + "
        "--tail-max-tokens (prompt budget frozen; the effective prompt carries the stored "
        "answer) — 5120 for the production invocation",
    )
    ap.add_argument("--upload", action="store_true", help="per-cell HF upload after regen")
    args = ap.parse_args()

    total_budget = assert_continuation_budget(args.max_model_len, args.tail_max_tokens)
    print(
        f"[regen] prefix-continuation: tail cap {args.tail_max_tokens}, total answer budget "
        f"{total_budget} (= original cap {cm.SAMPLING['max_tokens']} + tail)",
        flush=True,
    )

    corpora = (
        [c.strip() for c in args.corpora.split(",") if c.strip()]
        if args.corpora
        else list(cm.V2_CORPORA)
    )
    for c in corpora:
        assert c in cm.V2_CORPORA, (
            f"unknown/out-of-scope corpus {c!r} — regen covers the v2 corpora only "
            f"({sorted(cm.V2_CORPORA)}); wave-1-only corpora are excluded (downstream reuse)"
        )

    hf_id = cm.MODELS[args.model]["hf_id"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    from vllm import LLM, SamplingParams

    t0 = time.monotonic()
    llm = LLM(model=hf_id, max_model_len=args.max_model_len)
    engine_load_seconds = time.monotonic() - t0
    print(f"[regen] engine loaded in {engine_load_seconds:.1f}s ({hf_id})", flush=True)

    # Same stop-string expression regen_cell records in its audit (both read
    # the cm constants keyed on gen_format — the recipe constants, unmutated).
    stop_strings = cm.STOP_STRINGS if args.gen_format == "chat" else cm.NATURAL_STOP_STRINGS
    sampling = SamplingParams(
        n=cm.SAMPLING["n"],
        temperature=cm.SAMPLING["temperature"],
        top_p=cm.SAMPLING["top_p"],
        max_tokens=args.tail_max_tokens,
        seed=cm.SAMPLING["seed"],
        stop=list(stop_strings),
    )

    def _generate(texts: list[str]) -> list[tuple[str, str]]:
        return gen._vllm_generate_chunked(llm, texts, sampling)

    for corpus in corpora:
        regen_cell(
            _generate,
            tokenizer,
            args.model,
            corpus,
            gen_format=args.gen_format,
            tail_max_tokens=args.tail_max_tokens,
            max_model_len=args.max_model_len,
            upload=args.upload,
            engine_load_seconds=engine_load_seconds,
        )

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit: C-extension atexit teardown must not rewrite rc (gotchas.md)


if __name__ == "__main__":
    main()
