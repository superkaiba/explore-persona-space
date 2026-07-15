#!/usr/bin/env python
"""Issue #1345 Phase 1 — generate the R3 assistant-in-narrative story corpus.

On-policy tier-2 (instruct-and-strip) generation: the story-writing
instruction lives in the system prompt (instruct) / plain-text preamble
(pretrained) and is STRIPPED before extraction — only the narrative text is
used downstream. Topic seeds = first-user-turn questions of the shared R1/R2
S-track conversations, pinned @ 7159e5804d (plan §4 Phase 1).

vLLM generation (chunked, use_tqdm=False, engine seed GEN_SEED=42,
temperature 1.0, max_new_tokens 1024), per-chunk checkpointed JSONL with a
fingerprint-gated resume. Judge filter: claude-sonnet-4-5 via api_dispatch
(reason-then-verdict, >=4 extractable Q->A turns), drop-never-coerce with a
transport re-drive. Yield floor 400/500 after one retry batch (kill
criterion, plan §7): a miss exits rc=21 so the dispatcher can halt the story
legs while the chat<->no-template phases proceed.

Content hygiene: seed questions are LMSYS real user text and stories are raw
model generations — this script logs COUNTS/ids only, never text.

CLI:
  uv run python scripts/issue1345_gen_stories.py --model instruct [--smoke]
  uv run python scripts/issue1345_gen_stories.py --parse-judge-only <stories.jsonl>
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402

from explore_persona_space.experiments.issue_825.common import (  # noqa: E402
    MODEL_INSTRUCT,
    MODEL_PRETRAINED,
)
from explore_persona_space.llm.api_dispatch import (  # noqa: E402
    RESULT_RATE_LIMITED,
    RESULT_TRANSPORT,
    DispatchItem,
    dispatch_calls,
)

VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "250"))
MAX_MODEL_LEN = 4096
PROMPT_TOKEN_BUDGET = MAX_MODEL_LEN - c.STORY_MAX_NEW_TOKENS - 64

JUDGE_SYSTEM = (
    "You evaluate whether a short story contains extractable question-answer "
    "dialogue. A story PASSES when it contains at least "
    f"{c.STORY_MIN_TURNS} DISTINCT question->answer exchanges in which a person "
    "asks a question (quoted dialogue) and an AI assistant character named ARIA "
    "gives a substantive answer in quoted dialogue introduced by an attribution "
    'before the quote (e.g. ARIA replied: "..."). Exchanges with trivial or '
    "empty answers, unquoted dialogue, or answers not attributed to ARIA do not "
    "count. First give 2-3 sentences of reasoning, then finish with EXACTLY two "
    "final lines:\nTURNS: <integer number of qualifying exchanges>\n"
    "VERDICT: PASS or FAIL"
)


def _build_judge_request(item: DispatchItem) -> dict:
    """Messages-API params for one story-judge call (system lifted top-level)."""
    return {
        "model": c.JUDGE_MODEL,
        "max_tokens": c.JUDGE_MAX_TOKENS,
        "temperature": 0.0,
        "system": JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": item.payload["story"]}],
    }


def _parse_judge_response(text: str) -> dict:
    """Extract TURNS/VERDICT from a reason-then-verdict judge reply (raise on miss)."""
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
    if verdict is None:
        raise ValueError(f"judge reply missing VERDICT line ({len(text)} chars)")
    return {"verdict": verdict, "judge_turns": turns}


# ---------------------------------------------------------------------------
# Seed pool: shared R1/R2 conversation questions @ the pinned revision
# ---------------------------------------------------------------------------
def load_seed_pool(dl_dir: Path) -> list[dict]:
    """{conv_id, question} rows for the shared parent conv set (pinned rev).

    Shared set = intersection of the four parent stems' conv_ids (chat AND
    naturalistic, both models — the same subset the headline pair fits on).
    """
    id_sets = [set(c.parent_conv_ids(stem, dl_dir)) for stem in c.PARENT_STEMS]
    shared = set.intersection(*id_sets)
    track_s = c.stage_pinned_file(c.PARENT_TRACK_S_JSONL, dl_dir)
    rows = c.read_jsonl(track_s)
    pool = []
    for r in rows:
        cid = str(r.get("conv_id") or f"s{r['prompt_idx']}")
        if cid in shared:
            pool.append({"conv_id": cid, "question": r["prompt"]})
    assert pool, "seed pool empty — shared conv set did not join against track_s.jsonl"
    print(
        f"[seeds] shared convs={len(shared)} joined pool={len(pool)} "
        f"(per-stem sizes {[len(s) for s in id_sets]})",
        flush=True,
    )
    return sorted(pool, key=lambda r: r["conv_id"])


def _filter_pool_by_length(pool: list[dict], tokenizer, model_key: str) -> list[dict]:
    """Drop seeds whose FORMATTED prompt exceeds the token budget (#952 gotcha)."""
    kept, dropped = [], 0
    for row in pool:
        prompt = build_prompt(row["question"], model_key, tokenizer)
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok <= PROMPT_TOKEN_BUDGET:
            kept.append(row)
        else:
            dropped += 1
    print(f"[seeds] length filter: kept={len(kept)} dropped={dropped}", flush=True)
    return kept


def build_prompt(question: str, model_key: str, tokenizer) -> str:
    """Render the tier-2 generation prompt (instruction stripped before extraction)."""
    user_msg = (
        "Write the story now. The person's questions to ARIA should be about the "
        "same topic as this question (rephrase naturally; do not copy verbatim):\n"
        f"{question}"
    )
    if model_key == "instruct":
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": c.STORY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    # Base model: plain-text tier-2 preamble (no chat template), story follows.
    return f"{c.STORY_SYSTEM_PROMPT}\n\n{user_msg}\n\nStory:\n"


# ---------------------------------------------------------------------------
# vLLM generation (chunked + checkpointed)
# ---------------------------------------------------------------------------
def _gen_fingerprint(model_key: str, seed_ids: list[str]) -> str:
    import hashlib

    key = json.dumps(
        {
            "model": model_key,
            "gen_seed": c.GEN_SEED,
            "temperature": c.STORY_TEMPERATURE,
            "max_new_tokens": c.STORY_MAX_NEW_TOKENS,
            "system_prompt": c.STORY_SYSTEM_PROMPT,
            "seed_conv_ids": seed_ids,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def generate_stories(
    model_key: str, seeds: list[dict], out_path: Path, start_idx: int, tokenizer, llm
) -> list[dict]:
    """Generate one story per seed, checkpointing each vLLM chunk to JSONL.

    Resume: rows already in out_path with a matching fingerprint are skipped
    (fingerprint covers model, seeds, and every sampling constant).
    """
    from vllm import SamplingParams

    fp = _gen_fingerprint(model_key, [s["conv_id"] for s in seeds])
    meta_path = out_path.with_suffix(".meta.json")
    done_ids: set[str] = set()
    if out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp:
            done_ids = {r["story_id"] for r in c.read_jsonl(out_path)}
            print(f"[gen] resume: {len(done_ids)} stories already on disk", flush=True)
        else:
            raise RuntimeError(
                f"{out_path} exists with a DIFFERENT generation fingerprint "
                f"({meta.get('fingerprint')} != {fp}) — refusing to mix regimes; "
                "move the stale file aside"
            )
    else:
        c.write_json(meta_path, {"fingerprint": fp, "model": model_key, "n_seeds": len(seeds)})

    todo = [
        (i, s)
        for i, s in enumerate(seeds)
        if f"{model_key}_story{start_idx + i:04d}" not in done_ids
    ]
    sampling = SamplingParams(
        temperature=c.STORY_TEMPERATURE, max_tokens=c.STORY_MAX_NEW_TOKENS, seed=None
    )
    n_chunks = (len(todo) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for ci in range(0, len(todo), VLLM_CHUNK_SIZE):
        chunk = todo[ci : ci + VLLM_CHUNK_SIZE]
        prompts = [build_prompt(s["question"], model_key, tokenizer) for _, s in chunk]
        print(
            f"[vllm-chunk] gen chunk {ci // VLLM_CHUNK_SIZE + 1}/{n_chunks} "
            f"({len(chunk)} prompts, model={model_key})",
            flush=True,
        )
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        rows = []
        for (i, s), o in zip(chunk, outs, strict=True):
            rows.append(
                {
                    "story_id": f"{model_key}_story{start_idx + i:04d}",
                    "seed_conv_id": s["conv_id"],
                    "model": model_key,
                    "tier": "instruct_and_strip",
                    "story": o.outputs[0].text.strip(),
                    "finish_reason": o.outputs[0].finish_reason,
                }
            )
        c.append_jsonl(out_path, rows)
    return c.read_jsonl(out_path)


# ---------------------------------------------------------------------------
# Parse + judge filter
# ---------------------------------------------------------------------------
def parse_and_judge(rows: list[dict], cache_dir: Path, smoke: bool) -> tuple[list[dict], dict]:
    """Parser + judge filter over generated stories; returns (kept, yield_report).

    Keep = judge PASS (drop-never-coerce; malformed judge returns counted, never
    coerced; transport-class failures re-driven once, residue reported as
    transport_loss) AND parser extracts >= STORY_MIN_TURNS confident turns.
    """
    parsed: dict[str, list[dict]] = {}
    for r in rows:
        parsed[r["story_id"]] = c.parse_story_turns(r["story"])

    items = [DispatchItem(item_id=r["story_id"], payload={"story": r["story"]}) for r in rows]
    results = asyncio.run(
        dispatch_calls(
            items,
            model=c.JUDGE_MODEL,
            build_request=_build_judge_request,
            parse_response=_parse_judge_response,
            cache_dir=cache_dir,
            checkpoint_dir=cache_dir / "checkpoints",
            force_path="sync" if smoke else None,
        )
    )
    redrive = [
        it
        for it in items
        if results[it.item_id].error
        and results[it.item_id].category in (RESULT_RATE_LIMITED, RESULT_TRANSPORT)
    ]
    if redrive:
        print(f"[judge] re-driving {len(redrive)} transport-class failures", flush=True)
        results.update(
            asyncio.run(
                dispatch_calls(
                    redrive,
                    model=c.JUDGE_MODEL,
                    build_request=_build_judge_request,
                    parse_response=_parse_judge_response,
                    cache_dir=cache_dir,
                    checkpoint_dir=cache_dir / "checkpoints",
                    force_path="sync",
                )
            )
        )

    kept, counts = (
        [],
        {
            "n_generated": len(rows),
            "judge_pass": 0,
            "judge_fail": 0,
            "judge_malformed": 0,
            "transport_loss": 0,
            "parser_below_floor": 0,
            "kept": 0,
        },
    )
    for r in rows:
        res = results[r["story_id"]]
        turns = parsed[r["story_id"]]
        confident = [
            t for t in turns if t["confidence"]["marker_exact"] or t["confidence"]["answer_len_ok"]
        ]
        if res.error:
            key = (
                "transport_loss"
                if res.category in (RESULT_RATE_LIMITED, RESULT_TRANSPORT)
                else "judge_malformed"
            )
            counts[key] += 1
            continue
        verdict = res.result["verdict"]
        counts["judge_pass" if verdict == "PASS" else "judge_fail"] += 1
        if verdict != "PASS":
            continue
        if len(confident) < c.STORY_MIN_TURNS:
            counts["parser_below_floor"] += 1
            continue
        kept.append(
            {
                **r,
                "judge_verdict": verdict,
                "judge_turns": res.result.get("judge_turns"),
                "parsed_turns": turns,
                "n_parsed_turns": len(turns),
                "n_confident_turns": len(confident),
            }
        )
        counts["kept"] += 1
    return kept, counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=("instruct", "pretrained"))
    ap.add_argument("--out-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--dl-dir", type=Path, default=c.PARENT_DL_DIR)
    ap.add_argument("--n-stories", type=int, default=c.N_STORIES_TARGET)
    ap.add_argument("--yield-floor", type=int, default=c.STORY_YIELD_FLOOR)
    ap.add_argument("--smoke", action="store_true", help="n=3 stories, sync judge")
    ap.add_argument(
        "--parse-judge-only",
        type=Path,
        default=None,
        help="skip generation; run parser+judge on an existing stories JSONL (CPU smoke)",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "judge_cache"

    if args.parse_judge_only is not None:
        rows = c.read_jsonl(args.parse_judge_only)
        kept, counts = parse_and_judge(rows, cache_dir, smoke=True)
        report = {
            "metadata": c.metadata(c.GEN_SEED, len(rows), "scripts/issue1345_gen_stories.py"),
            "mode": "parse-judge-only",
            "counts": counts,
        }
        c.write_json(out_dir / "story_yield_parsejudge.json", report)
        print(f"[done] parse-judge-only kept={len(kept)}/{len(rows)}", flush=True)
        return

    assert args.model, "--model is required unless --parse-judge-only"
    model_key = args.model
    n_target = 3 if args.smoke else args.n_stories
    yield_floor = 2 if args.smoke else args.yield_floor

    pool = load_seed_pool(args.dl_dir)

    from transformers import AutoTokenizer
    from vllm import LLM

    model_id = MODEL_INSTRUCT if model_key == "instruct" else MODEL_PRETRAINED
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pool = _filter_pool_by_length(pool, tokenizer, model_key)
    assert len(pool) >= 2 * n_target, f"seed pool {len(pool)} < 2x target {n_target}"

    import numpy as np

    rng = np.random.default_rng(c.GEN_SEED)
    order = rng.permutation(len(pool))
    seeds_main = [pool[i] for i in order[:n_target]]
    seeds_reserve = [pool[i] for i in order[n_target:]]

    llm = LLM(
        model=model_id,
        seed=c.GEN_SEED,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1",
        enable_prefix_caching=(
            False if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "0") == "1" else None
        ),
    )

    raw_path = out_dir / f"raw_stories_{model_key}.jsonl"
    rows = generate_stories(model_key, seeds_main, raw_path, 0, tokenizer, llm)
    kept, counts = parse_and_judge(rows, cache_dir, args.smoke)

    retry_counts = None
    if len(kept) < n_target and seeds_reserve:
        shortfall = n_target - len(kept)
        retry_seeds = seeds_reserve[:shortfall]
        print(f"[retry] {len(kept)}/{n_target} kept — one retry batch of {len(retry_seeds)}")
        retry_path = out_dir / f"raw_stories_{model_key}_retry.jsonl"
        retry_rows = generate_stories(
            model_key, retry_seeds, retry_path, len(seeds_main), tokenizer, llm
        )
        retry_kept, retry_counts = parse_and_judge(retry_rows, cache_dir, args.smoke)
        kept.extend(retry_kept)

    kept_path = out_dir / f"kept_stories_{model_key}.jsonl"
    if kept_path.exists():
        kept_path.unlink()
    c.append_jsonl(kept_path, kept)

    report = {
        "metadata": c.metadata(c.GEN_SEED, len(kept), "scripts/issue1345_gen_stories.py"),
        "model": model_key,
        "n_target": n_target,
        "yield_floor": yield_floor,
        "counts_main": counts,
        "counts_retry": retry_counts,
        "n_kept": len(kept),
        "yield_ok": len(kept) >= yield_floor,
        "answer_char_lengths": sorted(
            len(r["story"]) for r in kept[:50]
        ),  # digest only, never text
    }
    c.write_json(out_dir / f"story_yield_{model_key}.json", report)

    if len(kept) < yield_floor:
        print(
            f"[yield-floor] FAILED: kept={len(kept)} < floor={yield_floor} — "
            "halting the story regime (plan §7 kill criterion); rc=21",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(21)
    print(f"[done] model={model_key} kept={len(kept)}/{n_target} stories", flush=True)


if __name__ == "__main__":
    main()
