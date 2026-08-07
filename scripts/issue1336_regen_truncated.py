#!/usr/bin/env python
"""Issue #1336 — targeted regeneration of cap-truncated rows at a doubled answer cap.

Regenerates ONLY the rows whose stored ``finish_reason == "length"`` in an
existing generation pool (an ``issue1336_gen_answers.py`` output cell), at an
opt-in LARGER answer-token cap, writing to a SUFFIXED sibling cell
(``<cell>_mt<max_tokens>``) locally and on the Hub so the 1024-cap originals
are never overwritten and concurrent consumers of the original pools are
unaffected.

Session constraints (2026-08-06 targeted-regen round on #1336):
  1. ``cm.SAMPLING`` / ``cm.MAX_MODEL_LEN`` are NEVER mutated — the doubled cap
     is per-invocation CLI state (``--max-tokens`` / ``--max-model-len``), so a
     concurrently queued job that pulls this branch keeps the recipe constants
     byte-identical.
  2. The PROMPT budget must not move: ``--max-model-len - --max-tokens`` is
     asserted equal to ``cm.PROMPT_TOKEN_BUDGET`` (3072), so ONLY the answer
     cap changes (raising max_tokens under the old engine length would shrink
     the admissible-prompt population instead).
  3. Existing pools are read-only inputs; output lands under the suffixed cell
     key only (``data/issue_1336/gen/<slug>/<cell>_mt<cap>/`` +
     ``.../generation/<slug>/<cell>_mt<cap>/`` on the Hub).

Rows that finished naturally under the original cap are NOT regenerated (the
cap never bound them). Per-row ``prefix_match`` records whether the new RAW
completion reproduces the stored response as a string prefix — the determinism
premise that makes targeted regen equivalent to whole-cell regen (same prompt +
same per-request seed should reproduce the first 1024 tokens and continue). The
per-cell audit reports the match rate; a LOW rate means targeted regen is a
selection-on-outcome resample of the long-answer tail — escalate to whole-cell
regeneration instead of splicing.

Row filters (render validation incl. the 2048-token rendered-conversation
budget), stop handling, chunked generation, and upload machinery are all
IMPORTED from ``issue1336_gen_answers`` (single source of truth): a regenerated
answer that now exceeds the render budget is DROPPED with
``<fmt>:over_token_budget`` exactly as the production keep-filter would drop
it — the audit reports that split.
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


def regen_cell_key(cell: str, max_tokens: int) -> str:
    """Suffixed sibling cell key — regenerated rows never share the source key."""
    return f"{cell}_mt{max_tokens}"


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
    max_tokens: int,
    max_model_len: int,
    upload: bool,
    engine_load_seconds: float | None = None,
) -> dict | None:
    """Regenerate one cell's cap-truncated rows at the new cap (resume-safe).

    ``generate_fn(texts) -> [(raw_text, finish_reason), ...]`` is the vLLM
    chunked-generation closure in production (built in ``main`` with the
    per-invocation SamplingParams); a stub in the CPU smoke. Returns the audit
    dict, or None when the cell was already complete (skip predicate mirrors
    ``issue1336_gen_answers._collect_pending``: done == local outputs exist;
    Hub-incomplete local outputs re-attempt only the upload).
    """
    cell = cm.gen_cell_key(corpus, gen_format)
    out_cell = regen_cell_key(cell, max_tokens)
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
    trunc = [r for r in src_rows if r.get("finish_reason") == "length"]
    n_src_trunc_kept = sum(1 for r in trunc if r.get("kept"))
    print(
        f"[regen] {slug}/{cell}: {len(trunc)}/{len(src_rows)} source rows cap-truncated "
        f"({n_src_trunc_kept} of them kept)",
        flush=True,
    )

    prompt_builder = {"chat": cm.tulu_prompt, "naturalistic": cm.natural_prompt}[gen_format]
    stop_strings = cm.STOP_STRINGS if gen_format == "chat" else cm.NATURAL_STOP_STRINGS
    truncate_markers = (
        cm.ROLE_HEADER_TRUNCATE if gen_format == "chat" else cm.NATURAL_ROLE_HEADER_TRUNCATE
    )
    if gen_format == "chat" and trunc:
        gen._assert_template_parity(tokenizer, [r["prompt"] for r in trunc[:3]])

    texts = [prompt_builder(r["prompt"]) for r in trunc]
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
    prefix_fracs: list[float] = []
    n_prefix_match = 0
    rep3_flags = 0
    for r, (raw, finish) in zip(trunc, outs, strict=True):
        answer = gen._truncate_role_headers(raw, truncate_markers)
        old = r.get("response", "")
        prefix_match = bool(old) and raw.startswith(old)
        n_prefix_match += int(prefix_match)
        prefix_fracs.append(_common_prefix_len(raw, old) / max(1, len(old)))
        finish_counts[finish] += 1
        new_answer_tokens.append(len(tokenizer(raw, add_special_tokens=False)["input_ids"]))
        row = {
            "prompt_idx": r["prompt_idx"],
            "prompt": r["prompt"],
            "response": answer,
            "response_raw_len_chars": len(raw),
            "finish_reason": finish,
            "regen_max_tokens": max_tokens,
            "orig_finish_reason": r.get("finish_reason"),
            "orig_kept": bool(r.get("kept")),
            "orig_response_len_chars": len(old),
            "prefix_match": prefix_match,
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
    audit = {
        "kind": "regen_truncated",
        "model": slug,
        "hf_id": cm.MODELS[slug]["hf_id"],
        "corpus": corpus,
        "gen_format": gen_format,
        "source_cell": cell,
        "regen_cell": out_cell,
        "n_source_rows": len(src_rows),
        "n_source_truncated": len(trunc),
        "n_source_truncated_kept": n_src_trunc_kept,
        "n_regenerated": len(rows),
        "n_kept": len(kept_ids),
        "keep_rate": (len(kept_ids) / len(rows)) if rows else None,
        "drop_reasons": dict(drop_reasons),
        "finish_reasons": dict(finish_counts),
        "cap_hit_rate_at_new_cap": (finish_counts.get("length", 0) / len(rows)) if rows else None,
        "prefix_match_rate": (n_prefix_match / len(rows)) if rows else None,
        "n_prefix_match": n_prefix_match,
        "prefix_common_frac": {
            "mean": statistics.fmean(prefix_fracs) if prefix_fracs else None,
            "min": min(prefix_fracs) if prefix_fracs else None,
        },
        "new_answer_tokens": {
            "mean": statistics.fmean(new_answer_tokens) if new_answer_tokens else None,
            "median": statistics.median(new_answer_tokens) if new_answer_tokens else None,
            "max": max(new_answer_tokens) if new_answer_tokens else None,
            "total": sum(new_answer_tokens),
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
        "sampling": dict(cm.SAMPLING) | {"max_tokens": max_tokens, "stop": list(stop_strings)},
        "max_model_len": max_model_len,
        "prompt_token_budget_realized": max_model_len - max_tokens,
        "engine_load_seconds": engine_load_seconds,
        "generation_wall_seconds": gen_wall,
        "rows_per_min": (len(rows) / (gen_wall / 60.0)) if rows and gen_wall > 0 else None,
        "new_tokens_per_second": (
            (sum(new_answer_tokens) / gen_wall) if rows and gen_wall > 0 else None
        ),
        "render_integrity": render_integrity,
        "code_sha": cm.resolve_code_sha(),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        f"[regen] {slug}/{out_cell}: regenerated {len(rows)} rows, kept {len(kept_ids)}; "
        f"prefix_match {n_prefix_match}/{len(rows)}; "
        f"new-cap hit {finish_counts.get('length', 0)}/{len(rows)}; "
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
        "--max-tokens", type=int, required=True, help="regen answer cap (must exceed recipe 1024)"
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        required=True,
        help="engine max_model_len; must equal PROMPT_TOKEN_BUDGET + --max-tokens (budget frozen)",
    )
    ap.add_argument("--upload", action="store_true", help="per-cell HF upload after regen")
    args = ap.parse_args()

    assert args.max_tokens > cm.SAMPLING["max_tokens"], (
        f"--max-tokens {args.max_tokens} must exceed the recipe cap {cm.SAMPLING['max_tokens']} "
        "(regen targets rows the ORIGINAL cap bound)"
    )
    realized_budget = args.max_model_len - args.max_tokens
    assert realized_budget == cm.PROMPT_TOKEN_BUDGET, (
        f"realized prompt budget {realized_budget} != {cm.PROMPT_TOKEN_BUDGET}: pass "
        "--max-model-len = PROMPT_TOKEN_BUDGET + --max-tokens so ONLY the answer cap moves "
        "(the admissible-prompt population must not change)"
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
        max_tokens=args.max_tokens,
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
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            upload=args.upload,
            engine_load_seconds=engine_load_seconds,
        )

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit: C-extension atexit teardown must not rewrite rc (gotchas.md)


if __name__ == "__main__":
    main()
