"""Re-judge the malformed story-judge attempts at max_tokens 800 (rule 23).

#1345 assistant-named-story follow-up diagnostic (interpretation-critic ask):
the round's story-quality judge (reason-then-verdict, ``VERDICT: PASS|FAIL``
final line, max_tokens 400) produced "malformed" replies (missing VERDICT line
-> drop-never-coerce) at parent(ARIA) instruct 24/677, parent pretrained
78/952, round("Assistant") instruct 117/742, round pretrained 146/943. Per
`.claude/rules/llm-judging.md` rule 23, parse-error drops that VANISH at a
larger response budget with 0 refusals are truncation censoring, not format
drift. This script re-judges every malformed attempt from all four pools ONCE
at max_tokens 800 — everything else (judge model, rubric, temperature=0.0)
identical to the original call — against a FRESH cache dir (the rubric-keyed
cache ignores max_tokens, so a stale cache would re-serve truncation-era
entries).

API-only: no training, no GPU, no new story generation. ~365 judge calls via
the project dispatcher (api_dispatch.dispatch_calls, sync path), one
transport-class redrive (rule 24: transport losses are retried, never
persisted as drops).

Usage (VM, from repo root or worktree root; thread-caps prefix per code-style):
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
  uv run python scripts/issue1345_rejudge_malformed.py            # full run
  ... --limit 3 --out-dir /tmp/issue-1345-rejudge/smoke --skip-upload  # smoke
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # HF/Anthropic creds + shared-VM thread caps BEFORE heavy imports

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import issue1345_common as c  # noqa: E402
import issue1345_gen_stories as g  # noqa: E402

from explore_persona_space.llm.api_dispatch import (  # noqa: E402
    RESULT_RATE_LIMITED,
    RESULT_TRANSPORT,
    DispatchItem,
    dispatch_calls,
)

# the ONLY changed judge parameter (400 -> 800). JUSTIFIED DEVIATION from llm-judging
# rule 23's 1024 floor (#2063): the banked #1345 fix-wave's own deliberate rejudge
# instrument — a raise would break parity with its committed wave.
REJUDGE_MAX_TOKENS = 800
HF_REJUDGE_PREFIX = "issue1345_framing/assistant_named_story/rejudge"
DEFAULT_OUT_DIR = Path("eval_results/issue_1345/assistant_named_story")

# The four pools (run, model_key) with their HF story-bundle prefixes and the
# character name each run's judge rubric was built with. model_key matches the
# persisted bundle basenames (raw_stories_<mk>.jsonl / judge_results_<mk>.jsonl).
POOLS: list[tuple[str, str]] = [
    ("parent", "instruct"),
    ("parent", "pretrained"),
    ("round", "instruct"),
    ("round", "pretrained"),
]
POOL_PREFIX = {
    "parent": "issue1345_framing/raw_completions/stories",
    "round": "issue1345_framing/assistant_named_story/raw_completions/stories",
}
CHARACTER_NAME = {"parent": "ARIA", "round": "Assistant"}

# Refusal classification on verdict-less replies (diagnostic counts, not a
# scored DV — the substring-match ban covers behavior DVs, and the rule-23
# signature rests on drops vanishing at 800, not on these counters):
# (a) EMPTY reply (reply_chars == 0) = an API-level refusal — the Messages API
#     returns stop_reason="refusal" with NO content block (verified by direct
#     probe on smoke rows at BOTH 400 and 800 tokens: blocks=[], out_tokens=1,
#     stop_reason=refusal; dispatch_calls extracts text="" for these);
# (b) non-empty reply matching the heuristic regex below = refusal-like text.
# NEAR_BUDGET_CHARS: a still-malformed reply at >= ~2400 chars plausibly sits
# at the 800-token ceiling (truncation-still); far below it = format drift.
NEAR_BUDGET_CHARS = 2400
_REFUSAL_RE = re.compile(
    r"(?i)\b(I (?:can(?:no|')t|cannot|won't|will not)|I'?m sorry|I am sorry"
    r"|unable to (?:help|assist|comply)|I don'?t feel comfortable)\b"
)


def judge_systems() -> dict[str, str]:
    """Per-run judge system prompts, VERBATIM from the round's own judge code.

    ``issue1345_gen_stories.JUDGE_SYSTEM`` is spliced from
    ``EPM_STORY_CHARACTER_NAME`` at import time; this script must run with the
    DEFAULT env (ARIA), so the imported constant IS the parent rubric
    byte-for-byte, and the round rubric is the same string with the character
    name swapped to "Assistant" (asserted round-trippable — "ARIA" appears
    nowhere else in the rubric, and the swap is case-sensitive so the literal
    lowercase "AI assistant character" phrasing is untouched).
    """
    assert c.STORY_CHARACTER_NAME == "ARIA" and not c.VARIANT, (
        "run this script with EPM_STORY_CHARACTER_NAME/EPM_I1345_VARIANT UNSET — the "
        "imported JUDGE_SYSTEM must be the parent (ARIA) rubric; got "
        f"name={c.STORY_CHARACTER_NAME!r} variant={c.VARIANT!r}"
    )
    parent_sys = g.JUDGE_SYSTEM
    n_name = parent_sys.count("ARIA")
    assert n_name == 3, f"rubric name-splice count changed ({n_name} != 3) — re-derive the swap"
    round_sys = parent_sys.replace("ARIA", "Assistant")
    assert round_sys.replace("Assistant", "ARIA") == parent_sys, "name swap not round-trippable"
    return {"parent": parent_sys, "round": round_sys}


def make_build_request(system: str):
    """Messages-API params builder — identical to the original story-judge call
    (``issue1345_gen_stories._build_judge_request``) except max_tokens 400->800."""

    def build(item: DispatchItem) -> dict:
        return {
            "model": c.JUDGE_MODEL,
            "max_tokens": REJUDGE_MAX_TOKENS,
            "temperature": 0.0,
            "system": system,
            "messages": [{"role": "user", "content": item.payload["story"]}],
        }

    return build


def parse_judge_response_tolerant(text: str) -> dict:
    """Same TURNS/VERDICT extraction as the original judge parser, but a missing
    VERDICT line returns verdict=None (still-malformed) instead of raising, so
    the still-malformed class is classified rather than surfaced as a dispatch
    error; reply_chars + a refusal_like heuristic ride along for diagnosis."""
    turns, verdict = None, None
    for line in text.split("\n"):
        s = line.strip()
        if s.upper().startswith("TURNS:"):
            with contextlib.suppress(ValueError, IndexError):
                turns = int(s.split(":", 1)[1].strip().split()[0])
        if s.upper().startswith("VERDICT:"):
            v = s.split(":", 1)[1].strip().upper()
            if v in ("PASS", "FAIL"):
                verdict = v
    out = {"verdict": verdict, "judge_turns": turns, "reply_chars": len(text)}
    if verdict is None:
        out["refusal_like"] = bool(_REFUSAL_RE.search(text))
    return out


def load_pool(run: str, mk: str, dl_dir: Path) -> tuple[list[dict], dict[str, str]]:
    """Download one pool's judge digests + raw stories; return (digest_rows,
    story_id -> story text). Asserts every judged story_id resolves to text."""
    prefix = POOL_PREFIX[run]
    dest = dl_dir / f"{run}_{mk}"
    digests = c.read_jsonl(
        g._hf_download_to(f"{prefix}/judge_results_{mk}.jsonl", dest / f"judge_results_{mk}.jsonl")
    )
    stories: dict[str, str] = {}
    for base in (f"raw_stories_{mk}.jsonl", f"raw_stories_{mk}_retry.jsonl"):
        for r in c.read_jsonl(g._hf_download_to(f"{prefix}/{base}", dest / base)):
            stories[r["story_id"]] = r["story"]
    missing = [d["story_id"] for d in digests if d["story_id"] not in stories]
    assert not missing, f"{run}/{mk}: {len(missing)} judged story_ids missing from raw stories"
    return digests, stories


def rejudge_pool(
    run: str, mk: str, system: str, dl_dir: Path, cache_dir: Path, limit: int
) -> tuple[dict, list[dict]]:
    """Re-judge one pool's malformed attempts at max_tokens 800; returns
    (pool_summary, per_row_records). Never prints story or judge text."""
    digests, stories = load_pool(run, mk, dl_dir)
    n_total = len(digests)
    malformed = [
        d
        for d in digests
        if "verdict" not in d
        and d.get("judge_error_category") not in (RESULT_RATE_LIMITED, RESULT_TRANSPORT)
    ]
    n_old_transport = sum(
        1
        for d in digests
        if "verdict" not in d
        and d.get("judge_error_category") in (RESULT_RATE_LIMITED, RESULT_TRANSPORT)
    )
    original_kept = sum(
        1
        for d in digests
        if d.get("verdict") == "PASS" and d.get("n_confident_turns", 0) >= c.STORY_MIN_TURNS
    )
    todo = malformed[:limit] if limit else malformed
    print(
        f"[rejudge] {run}/{mk}: total_judged={n_total} malformed_at_400={len(malformed)} "
        f"old_transport_rows={n_old_transport} rejudging={len(todo)}",
        flush=True,
    )

    items = [
        DispatchItem(item_id=d["story_id"], payload={"story": stories[d["story_id"]]}) for d in todo
    ]
    pool_cache = cache_dir / f"{run}_{mk}"
    results = asyncio.run(
        dispatch_calls(
            items,
            model=c.JUDGE_MODEL,
            build_request=make_build_request(system),
            parse_response=parse_judge_response_tolerant,
            cache_dir=pool_cache,
            checkpoint_dir=pool_cache / "checkpoints",
            force_path="sync",
        )
    )
    redrive = [
        it
        for it in items
        if results[it.item_id].error
        and results[it.item_id].category in (RESULT_RATE_LIMITED, RESULT_TRANSPORT)
    ]
    if redrive:
        print(f"[rejudge] {run}/{mk}: re-driving {len(redrive)} transport failures", flush=True)
        results.update(
            asyncio.run(
                dispatch_calls(
                    redrive,
                    model=c.JUDGE_MODEL,
                    build_request=make_build_request(system),
                    parse_response=parse_judge_response_tolerant,
                    cache_dir=pool_cache,
                    checkpoint_dir=pool_cache / "checkpoints",
                    force_path="sync",
                )
            )
        )

    counts = {
        "n_now_pass": 0,
        "n_now_fail": 0,
        "n_still_malformed": 0,
        "n_refusal_empty": 0,
        "n_refusal_like_text": 0,
        "n_still_malformed_near_budget": 0,
        "n_transport_lost": 0,
        "n_error_other": 0,
        "n_new_pass_floor_ok": 0,
    }
    rows: list[dict] = []
    for d in todo:
        res = results[d["story_id"]]
        row = {
            "run": run,
            "model": mk,
            "story_id": d["story_id"],
            "old_verdict": None,
            "old_error_category": d.get("judge_error_category"),
            "n_confident_turns": d.get("n_confident_turns"),
        }
        if res.error:
            if res.category in (RESULT_RATE_LIMITED, RESULT_TRANSPORT):
                counts["n_transport_lost"] += 1
            else:
                counts["n_error_other"] += 1
            rows.append({**row, "new_verdict": None, "new_error_category": res.category})
            continue
        r = res.result
        row.update(
            {
                "new_verdict": r["verdict"],
                "new_judge_turns": r.get("judge_turns"),
                "reply_chars": r.get("reply_chars"),
            }
        )
        if r["verdict"] == "PASS":
            counts["n_now_pass"] += 1
            floor_ok = d.get("n_confident_turns", 0) >= c.STORY_MIN_TURNS
            counts["n_new_pass_floor_ok"] += int(floor_ok)
            row["parser_floor_ok"] = floor_ok
        elif r["verdict"] == "FAIL":
            counts["n_now_fail"] += 1
        elif r.get("reply_chars", 0) == 0:
            counts["n_refusal_empty"] += 1
            row["refusal_empty"] = True
        else:
            counts["n_still_malformed"] += 1
            counts["n_refusal_like_text"] += int(r.get("refusal_like", False))
            row["refusal_like"] = r.get("refusal_like", False)
            near = r["reply_chars"] >= NEAR_BUDGET_CHARS
            counts["n_still_malformed_near_budget"] += int(near)
            row["near_budget_chars"] = near
        rows.append(row)

    summary = {
        "run": run,
        "model": mk,
        "character_name": CHARACTER_NAME[run],
        "hf_pool_prefix": POOL_PREFIX[run],
        "n_judged_total_at_400": n_total,
        "n_malformed_at_400": len(malformed),
        "n_old_transport_rows": n_old_transport,
        "n_rejudged": len(todo),
        **counts,
        "n_refusal": counts["n_refusal_empty"] + counts["n_refusal_like_text"],
        "original_kept": original_kept,
        "revised_kept": original_kept + counts["n_new_pass_floor_ok"],
        "malformed_rate_at_400": len(malformed) / n_total if n_total else None,
        # verdict-less at 800 = format-drift still-malformed + empty-content
        # refusals + transport residue (all remain drops under rule 9/24).
        "malformed_rate_at_800": (
            (
                len(malformed)
                - len(todo)
                + counts["n_still_malformed"]
                + counts["n_refusal_empty"]
                + counts["n_transport_lost"]
            )
            / n_total
            if n_total
            else None
        ),
    }
    print(f"[rejudge] {run}/{mk}: {counts}", flush=True)
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="per-pool cap on re-judged rows (smoke)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/issue-1345-rejudge"))
    ap.add_argument("--skip-upload", action="store_true", help="skip the HF upload (smoke)")
    args = ap.parse_args()

    systems = judge_systems()
    cache_dir = args.workdir / ("cache_smoke" if args.limit else "cache_800")
    dl_dir = args.workdir / "dl"
    rows_path = args.out_dir / "rejudge_malformed_rows.jsonl"
    summary_path = args.out_dir / "rejudge_malformed.json"
    if rows_path.exists():
        rows_path.unlink()  # deterministic re-run: rebuild the per-row record

    pool_summaries: list[dict] = []
    for run, mk in POOLS:
        summary, rows = rejudge_pool(run, mk, systems[run], dl_dir, cache_dir, args.limit)
        pool_summaries.append(summary)
        c.append_jsonl(rows_path, rows)
        # Checkpoint-per-phase: partial summary lands after EVERY pool.
        payload = {
            "metadata": c.metadata(0, sum(s["n_rejudged"] for s in pool_summaries), __file__),
            "judge_model": c.JUDGE_MODEL,
            # frozen at the original #1345 run's instrument (this script's docstring:
            # "the ONLY changed judge parameter (400 -> 800)"); was c.JUDGE_MAX_TOKENS
            # before #2063 raised it to 1024.
            "original_max_tokens": 400,
            "rejudge_max_tokens": REJUDGE_MAX_TOKENS,
            "temperature": 0.0,
            "limit": args.limit,
            "judge_system_sha256": {
                k: hashlib.sha256(v.encode()).hexdigest() for k, v in systems.items()
            },
            "judge_system": systems,
            "story_min_turns": c.STORY_MIN_TURNS,
            "pools": pool_summaries,
        }
        c.write_json(summary_path, payload)

    tot = {
        k: sum(s[k] for s in pool_summaries)
        for k in (
            "n_malformed_at_400",
            "n_rejudged",
            "n_now_pass",
            "n_now_fail",
            "n_still_malformed",
            "n_refusal",
            "n_transport_lost",
        )
    }
    print(f"[rejudge] TOTAL: {tot}", flush=True)

    if args.skip_upload:
        print("[rejudge] --skip-upload: not uploading to HF", flush=True)
        print("[done] rejudge_malformed (no upload)", flush=True)
        return
    from explore_persona_space.orchestrate import hub

    for p in (summary_path, rows_path):
        url = hub._upload(
            p,
            repo_id=hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=f"{HF_REJUDGE_PREFIX}/{p.name}",
            upload_as_file=True,
        )
        assert url, f"upload returned no path for {p}"
        print(f"[rejudge] uploaded {p.name} -> {HF_REJUDGE_PREFIX}/{p.name}", flush=True)
    print("[done] rejudge_malformed", flush=True)


if __name__ == "__main__":
    main()
