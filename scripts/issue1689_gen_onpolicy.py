"""Issue #1689 Phase B — on-policy a2 (and user-onpolicy u2) generator.

For every non-user condition (12 base): the MEASURED model produces a2
on-policy given the rendered prompt_text ending at the a2-slot boundary.
For user_onpolicy_{framing}: the MEASURED model produces u2 given the
persona-header steering "you are the user, write your next turn" prompt.

Route: vLLM batched at Qwen-2.5-7B base + instruct, sharded across
CUDA_VISIBLE_DEVICES via a launcher-set per-cell env pin (per
`.claude/rules/gotchas.md` CVD-clobber entry). Judge-filter via
`api_dispatch.py` (Sonnet 4.5, N=3 draws, T=0.7, max_tokens=1024 (raised from 300, #2063),
anchored rubric, reason-then-score, drop-never-coerce, rubric-fingerprint
partition per rubric class per plan §9).

Round 8 crash-fix (empty-prompt filter + yield-floor drop logic):
  The Phase A renderer emits CHAT-framing rows with `messages: [...]` but
  NO `prompt_text` field — `row.get("prompt_text", "")` returned `""` and
  vLLM crashed on the first row with
  `ValueError: The decoder prompt cannot be empty`. Fix:
  (a) render the vLLM input from `messages` via `apply_chat_template`
      when present, else use `prompt_text`, so chat rows produce a real
      non-empty prompt;
  (b) drop rows whose resolved prompt is empty/too-short + report the
      count per condition;
  (c) implement the plan §11 yield-floor: below-floor conditions are
      DROPPED (reported to `yield_report.json`), never silently
      backfilled. Equalize-down keeps kept sources at floor-N.

Yield floor 80% with equalize-down. HIGH-risk cells (base + character in
chat, base + story) get 2× retry budget.

Smoke: --smoke → 1 condition × 5 rows, mock generation + mock judge.
Diagnose: --diagnose-empty-prompts → walk every rendered condition file,
report per-condition empty/short-prompt counts + one sample row each.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_common import (  # noqa: E402
    CONDITION_TABLE,
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    ISSUE_NUM,
    ISSUE_SLUG,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_N_DRAWS,
    JUDGE_TEMPERATURE,
    MODEL_BASE,
    MODEL_INSTRUCT,
    YIELD_FLOOR,
)


_HIGH_RISK_CELLS = {  # doubled retry budget per plan §11 yield-risk
    ("base", "helios_chat"),
    ("base", "wren_chat"),
    ("base", "dana_chat"),
    ("base", "assistant_story"),
    ("base", "helios_story"),
    ("base", "wren_story"),
    ("base", "dana_story"),
}


# Rows whose resolved prompt is empty or below this character count are
# treated as broken renders and dropped from the input pool. The floor is
# well below any real rendered prompt (a bare "User: X\n\nAssistant: "
# scaffold already exceeds this) — the intent is to catch the empty-string
# regression, not to filter legitimate short prompts.
_MIN_PROMPT_LEN = 10


# --- Judge rubric (persona-vectors 0-100 anchored, plan §9) ---------------

_JUDGE_RUBRIC_BY_ARM: dict[str, str] = {
    "character": (
        "Rate 0-100 how consistently the response stays in character as the "
        "named persona. 0 = totally out of character; 50 = partially in "
        "character; 100 = fully in character with matching voice and content."
    ),
    "user": (
        "Rate 0-100 how plausibly the text is a natural user follow-up given "
        "the conversation. 0 = not a user turn at all; 50 = plausible but off "
        "topic; 100 = natural, on-topic user turn."
    ),
    "assistant": (
        "Rate 0-100 how coherently the response is a helpful assistant reply. "
        "0 = incoherent or refusal; 50 = partial answer; 100 = coherent helpful "
        "assistant response."
    ),
}


def _rubric_key_for(condition_slug: str) -> str:
    for cond in CONDITION_TABLE:
        if cond.slug == condition_slug:
            if cond.is_character:
                return "character"
            if cond.is_user:
                return "user"
            return "assistant"
    raise ValueError(f"unknown condition {condition_slug}")


def _mock_generation(row: dict) -> str:
    """Deterministic mock generator for smoke tests."""
    return f"[mock a2 for conv={row['conv_id']} condition={row['condition']}]"


def _mock_judge_score(_completion: str) -> float:
    return 85.0  # passes >50 threshold


def _build_request(item) -> dict:  # type: ignore[no-untyped-def]
    """Build Anthropic Messages-API params for one judge DispatchItem.

    Lifts the rubric system prompt to the top-level ``system=`` param — the
    Messages API has NO ``"system"`` message role (`llm/api_dispatch.py`
    docstring + `.claude/rules/gotchas.md` "no `system` message ROLE" entry).
    """
    p = item.payload
    return {
        "model": p["model"],
        "max_tokens": p["max_tokens"],
        "temperature": p["temperature"],
        "system": p["system"],
        "messages": [
            {"role": "user", "content": p["user"]},
        ],
    }


def _parse(text: str) -> str:
    return text.strip()


def _score_from_text(text: str) -> float | None:
    """Extract a trailing integer 0-100 from the judge's reason-then-score
    response, dropping (returning None) on parse failure or out-of-range.

    Rule 9 drop-never-coerce (`.claude/rules/llm-judging.md`).
    """
    try:
        score = float(text.strip().split()[-1])
    except (ValueError, IndexError):
        return None
    if 0 <= score <= 100:
        return score
    return None


def _resolve_prompt_text(row: dict, tokenizer=None) -> str:
    """Resolve the vLLM input prompt from a rendered row.

    Chat-framing rows carry `messages: [...]` and NO `prompt_text`; the vLLM
    input must be built via `apply_chat_template(add_generation_prompt=True)`.
    Naturalistic + story rows carry `prompt_text` directly.

    Returns an EMPTY STRING when neither field is usable (empty messages, no
    prompt_text, or apply_chat_template unavailable). Callers must filter
    empty results before calling `LLM.generate` (see the crash-fix note in
    the module docstring — vLLM raises `ValueError: The decoder prompt
    cannot be empty` on any empty prompt).
    """
    # Prefer chat_template when the row is a chat row.
    if row.get("prompt_source") == "chat_template" and row.get("messages"):
        if tokenizer is None:
            # No tokenizer available -> cannot resolve; treat as empty so the
            # caller drops the row.
            return ""
        try:
            return tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return ""
    # Fallback: naturalistic + story rows have prompt_text.
    return row.get("prompt_text", "") or ""


def _classify_row_prompt(
    row: dict,
    tokenizer=None,
) -> tuple[str, str]:
    """Return (resolved_prompt, classification) for one row.

    classification ∈ {"ok", "empty", "short"}.
    """
    prompt = _resolve_prompt_text(row, tokenizer=tokenizer)
    if not prompt or not prompt.strip():
        return prompt, "empty"
    if len(prompt) < _MIN_PROMPT_LEN:
        return prompt, "short"
    return prompt, "ok"


def filter_valid_prompts(
    rows: list[dict],
    *,
    tokenizer=None,
) -> tuple[list[tuple[dict, str]], dict]:
    """Filter out rows whose resolved prompt is empty or too short.

    Returns (kept_rows_with_prompts, stats) where each kept entry is
    `(row, resolved_prompt)`. Stats reports empty/short/ok counts.
    """
    kept: list[tuple[dict, str]] = []
    n_empty = 0
    n_short = 0
    for row in rows:
        prompt, verdict = _classify_row_prompt(row, tokenizer=tokenizer)
        if verdict == "ok":
            kept.append((row, prompt))
        elif verdict == "empty":
            n_empty += 1
        elif verdict == "short":
            n_short += 1
    stats = {
        "n_input": len(rows),
        "n_kept": len(kept),
        "n_dropped_empty_prompt": n_empty,
        "n_dropped_short_prompt": n_short,
    }
    return kept, stats


def generate_and_filter(
    rows: list[dict],
    *,
    model_name: str,
    condition_slug: str,
    mock: bool = False,
) -> tuple[list[dict], dict]:
    """Generate a2 (or user-arm u2) on-policy + judge-filter per plan §9.

    Returns (kept_rows, stats_dict). Kept rows have `a2_text` (or `u2_text`
    for user-onpolicy arm) populated + `judge_score_mean` field.

    Round-8 crash-fix: chat-framing rows have NO `prompt_text` — resolve via
    the tokenizer's `apply_chat_template` on `messages`. Rows whose resolved
    prompt is empty/short are dropped BEFORE any vLLM call (empty prompts
    hard-crash vLLM v1 EngineCore, `ValueError: The decoder prompt cannot be
    empty`).
    """
    # Load the tokenizer once for chat-template resolution + prompt-length
    # accounting; smoke path skips the API and uses mocks.
    tokenizer = None
    if not mock:
        try:
            from transformers import AutoTokenizer  # noqa: E402

            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:  # transformers import / hub failures
            print(
                f"[gen] WARNING: tokenizer load failed for {model_name}: {exc}",
                flush=True,
            )
            tokenizer = None

    # (1) Filter empty / short prompts BEFORE any vLLM call.
    kept_input, filter_stats = filter_valid_prompts(rows, tokenizer=tokenizer)
    if filter_stats["n_dropped_empty_prompt"] or filter_stats["n_dropped_short_prompt"]:
        print(
            f"[gen] {condition_slug}: dropped "
            f"{filter_stats['n_dropped_empty_prompt']} empty + "
            f"{filter_stats['n_dropped_short_prompt']} short-prompt rows "
            f"(min_len={_MIN_PROMPT_LEN})",
            flush=True,
        )
    if not kept_input:
        stats = {
            "n_input": len(rows),
            "n_after_prompt_filter": 0,
            "n_kept": 0,
            "yield_frac": 0.0,
            "dropped_content": 0,
            "dropped_refusal": 0,
            "dropped_transport": 0,
            "meets_yield_floor": False,
            "model": model_name,
            "condition": condition_slug,
            **filter_stats,
        }
        return [], stats

    # (2) Generate. For a REAL run we build ONE vLLM engine and pass the
    # already-resolved prompts as a batch; the chat-template rendering
    # happens above via `_resolve_prompt_text`.
    kept: list[dict] = []
    dropped_content = 0
    dropped_refusal = 0
    dropped_transport = 0

    prompts = [p for _, p in kept_input]
    if mock:
        completions = [_mock_generation(row) for row, _ in kept_input]
    else:
        from vllm import LLM, SamplingParams  # noqa: E402

        _llm = LLM(model=model_name, gpu_memory_utilization=0.85)
        sp = SamplingParams(
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            max_tokens=GEN_MAX_NEW_TOKENS,
            n=1,
        )
        outs = _llm.generate(prompts, sp, use_tqdm=False)
        completions = [o.outputs[0].text if o.outputs else "" for o in outs]

    rubric = _rubric_key_for(condition_slug)

    # (3) Judge each (row, completion) — hoist ALL draws across ALL rows into
    # ONE dispatch_calls batch. Per CLAUDE.md § LLM judge, api_dispatch routes
    # large sets to the Anthropic Batch API; a per-row inner loop of
    # asyncio.run(...) calls serializes the API layer entirely.
    per_row_scores: dict[str, list[float]] = {str(row["conv_id"]): [] for row, _ in kept_input}

    if mock:
        for (row, _), completion in zip(kept_input, completions):
            per_row_scores[str(row["conv_id"])] = [
                _mock_judge_score(completion) for _ in range(JUDGE_N_DRAWS)
            ]
    else:
        from explore_persona_space.llm.api_dispatch import (  # noqa: E402
            DispatchItem,
            dispatch_calls,
        )

        items: list = []
        item_to_row_id: dict[str, str] = {}
        for (row, _prompt), completion in zip(kept_input, completions):
            row_id = str(row["conv_id"])
            user_msg = (
                f"Content to score:\n\n{completion}\n\n"
                "Reason briefly (1-2 sentences), then output an integer 0-100."
            )
            for i in range(JUDGE_N_DRAWS):
                item_id = f"{row_id}_draw{i}"
                items.append(
                    DispatchItem(
                        item_id=item_id,
                        payload={
                            "model": JUDGE_MODEL,
                            "system": _JUDGE_RUBRIC_BY_ARM[rubric],
                            "user": user_msg,
                            "max_tokens": JUDGE_MAX_TOKENS,
                            "temperature": JUDGE_TEMPERATURE,
                        },
                    )
                )
                item_to_row_id[item_id] = row_id

        results = asyncio.run(
            dispatch_calls(
                items,
                model=JUDGE_MODEL,
                build_request=_build_request,
                parse_response=_parse,
                response_valid=lambda t: isinstance(t, str) and len(t.strip()) > 0,
                force_path="sync",
            )
        )

        for item_id, row_id in item_to_row_id.items():
            res = results[item_id]
            # Transport-class failure — retried by api_dispatch, exhausted:
            # never coerce, never enter scores (llm-judging.md rule 24).
            if res.error:
                dropped_transport += 1
                continue
            text = res.result
            if not isinstance(text, str):
                dropped_content += 1
                continue
            score = _score_from_text(text)
            if score is None:
                dropped_content += 1
                continue
            per_row_scores[row_id].append(score)

    # (4) Reduce per-row scores → row verdicts.
    for (row, _prompt), completion in zip(kept_input, completions):
        row_id = str(row["conv_id"])
        scores = per_row_scores[row_id]

        if not scores:
            dropped_content += 1
            continue

        score_mean = sum(scores) / len(scores)
        if score_mean < 50:
            dropped_refusal += 1
            continue

        new_row = dict(row)
        # For user-onpolicy arm the model generates u2; for others it generates a2
        if row.get("identity") == "user" and row.get("provenance") == "onpolicy":
            new_row["u2_text"] = completion
            new_row["u2_source"] = "onpolicy"
        else:
            new_row["a2_text"] = completion
        new_row["judge_score_mean"] = score_mean
        new_row["judge_n_draws"] = len(scores)
        kept.append(new_row)

    n_input = len(rows)
    n_after_filter = len(kept_input)
    yield_frac = len(kept) / n_input if n_input else 0.0
    stats = {
        "n_input": n_input,
        "n_after_prompt_filter": n_after_filter,
        "n_kept": len(kept),
        "yield_frac": yield_frac,
        "dropped_content": dropped_content,
        "dropped_refusal": dropped_refusal,
        "dropped_transport": dropped_transport,
        "meets_yield_floor": yield_frac >= YIELD_FLOOR,
        "model": model_name,
        "condition": condition_slug,
        **filter_stats,
    }
    return kept, stats


def diagnose_empty_prompts(
    in_dir: Path,
    *,
    n_per_condition: int = 100,
    out_path: Path | None = None,
) -> dict:
    """Diagnostic mode — report per-condition empty/short-prompt counts.

    Walks every `<condition>.jsonl` in `in_dir`, loads up to `n_per_condition`
    rows per file, and reports:
      - total rows
      - n_empty_prompt
      - n_short_prompt (< 10 chars)
      - one sample empty-prompt row (redacted to structural keys only —
        no verbatim user text)

    Never calls vLLM or the judge — pure structural check. Uses NO
    tokenizer (so chat rows without a live tokenizer resolve to empty
    prompts here — that IS the diagnostic signal that the caller must
    load a tokenizer OR that the render pipeline is broken).
    """
    report: dict = {"conditions": {}, "min_prompt_len": _MIN_PROMPT_LEN}
    per_condition = report["conditions"]
    total_empty = 0
    total_short = 0
    total_ok = 0
    total_rows = 0
    for path in sorted(in_dir.glob("*.jsonl")):
        cond_name = path.stem
        rows: list[dict] = []
        with path.open() as fh:
            for i, line in enumerate(fh):
                if i >= n_per_condition:
                    break
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        n_empty = 0
        n_short = 0
        n_ok = 0
        sample_empty: dict | None = None
        for row in rows:
            _prompt, verdict = _classify_row_prompt(row, tokenizer=None)
            if verdict == "empty":
                n_empty += 1
                if sample_empty is None:
                    sample_empty = {
                        "conv_id": row.get("conv_id"),
                        "condition": row.get("condition"),
                        "framing": row.get("framing"),
                        "identity": row.get("identity"),
                        "provenance": row.get("provenance"),
                        "prompt_source": row.get("prompt_source"),
                        "has_messages": bool(row.get("messages")),
                        "has_prompt_text": bool(row.get("prompt_text")),
                    }
            elif verdict == "short":
                n_short += 1
            else:
                n_ok += 1
        per_condition[cond_name] = {
            "n_rows": len(rows),
            "n_empty_prompt": n_empty,
            "n_short_prompt": n_short,
            "n_ok": n_ok,
            "sample_empty_row_meta": sample_empty,
        }
        total_rows += len(rows)
        total_empty += n_empty
        total_short += n_short
        total_ok += n_ok
    report["totals"] = {
        "n_rows": total_rows,
        "n_empty_prompt": total_empty,
        "n_short_prompt": total_short,
        "n_ok": total_ok,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(report, fh, indent=2)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=False)
    ap.add_argument("--out", dest="out_path", type=Path, required=False)
    ap.add_argument("--stats-out", dest="stats_path", type=Path, required=False)
    ap.add_argument("--condition", type=str, required=False)
    ap.add_argument(
        "--model",
        type=str,
        required=False,
        choices=[MODEL_BASE, MODEL_INSTRUCT],
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # Diagnostic mode
    ap.add_argument(
        "--diagnose-empty-prompts",
        action="store_true",
        help=(
            "Walk every <condition>.jsonl under --in-dir and report per-condition "
            "empty/short-prompt counts + one sample empty row per condition. "
            "Does NOT call vLLM or the judge. Requires --in-dir + --diagnose-out."
        ),
    )
    ap.add_argument(
        "--in-dir",
        type=Path,
        required=False,
        help="[diagnose only] dir containing <condition>.jsonl files",
    )
    ap.add_argument(
        "--diagnose-out",
        type=Path,
        required=False,
        help="[diagnose only] output JSON path for the per-condition report",
    )
    ap.add_argument(
        "--diagnose-n",
        type=int,
        default=100,
        help="[diagnose only] rows to sample per condition (default 100)",
    )
    args = ap.parse_args()

    if args.diagnose_empty_prompts:
        if args.in_dir is None or args.diagnose_out is None:
            ap.error("--diagnose-empty-prompts requires --in-dir and --diagnose-out")
        report = diagnose_empty_prompts(
            args.in_dir,
            n_per_condition=args.diagnose_n,
            out_path=args.diagnose_out,
        )
        totals = report["totals"]
        print(
            f"[diagnose] scanned {totals['n_rows']} rows across "
            f"{len(report['conditions'])} conditions: "
            f"empty={totals['n_empty_prompt']} short={totals['n_short_prompt']} "
            f"ok={totals['n_ok']}",
            flush=True,
        )
        for cond, stats in sorted(report["conditions"].items()):
            print(
                f"[diagnose]  {cond}: rows={stats['n_rows']} "
                f"empty={stats['n_empty_prompt']} short={stats['n_short_prompt']} "
                f"ok={stats['n_ok']}",
                flush=True,
            )
            if stats["sample_empty_row_meta"] is not None:
                print(
                    f"[diagnose]   sample empty row: {stats['sample_empty_row_meta']}",
                    flush=True,
                )
        print(f"[diagnose] wrote {args.diagnose_out}", flush=True)
        return 0

    # Standard generate + judge mode.
    if not all([args.in_path, args.out_path, args.stats_path, args.condition, args.model]):
        ap.error("generate mode requires --in, --out, --stats-out, --condition, --model")

    rows = []
    with args.in_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition") == args.condition:
                rows.append(row)
    if args.smoke:
        rows = rows[:5]

    # Log high-risk indication (retry budget scaling handled at dispatcher level)
    model_kind = "base" if "Instruct" not in args.model else "instruct"
    is_high_risk = (model_kind, args.condition) in _HIGH_RISK_CELLS
    if is_high_risk:
        print(
            f"[gen] HIGH-risk cell: {model_kind}/{args.condition} - 2x retry budget applies",
            flush=True,
        )

    kept, stats = generate_and_filter(
        rows,
        model_name=args.model,
        condition_slug=args.condition,
        mock=args.smoke,
    )
    stats["high_risk"] = is_high_risk
    stats["issue"] = f"issue{ISSUE_NUM}_{ISSUE_SLUG}"

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.stats_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")
    with args.stats_path.open("w") as fh:
        json.dump(stats, fh, indent=2)

    yield_str = f"{stats['yield_frac']:.2f}"
    floor_status = "MEETS" if stats["meets_yield_floor"] else "BELOW"
    print(
        f"[gen] wrote {len(kept)} rows to {args.out_path} "
        f"(yield={yield_str} {floor_status} floor={YIELD_FLOOR})",
        flush=True,
    )
    if not stats["meets_yield_floor"]:
        print(
            f"[gen] LOUD: condition {args.condition} yield {yield_str} "
            f"< floor {YIELD_FLOOR} — this condition SHOULD BE DROPPED "
            f"from downstream capture (dispatcher writes yield_report.json).",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    import os

    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGBART pointer. main()'s writes are
    # already flushed via explicit fh.close(); atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
