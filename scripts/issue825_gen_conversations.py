#!/usr/bin/env python
"""Issue #825 — generate two-round conversations (Track M) and the Track-S regen corpus.

Track M (``--track m``): harvest real first-user turns (lmsys/lmsys-chat-1m first, then
allenai/WildChat-1M; English-only, non-toxic per available row flags, deduped on the
first 200 chars), generate a1/a2 with vLLM GREEDY decoding on the instruct model, and
generate the second user turn u2 with claude-haiku-4-5 acting as a simulated USER
(a generation role, NOT a judge) under >=20 rotating persona briefs.

Track S (``--track s``): re-generate the #779 Track-S responses with the PARENT-EXACT
sampling parameters (n=1, temperature=1.0, top_p=0.95, max_tokens=1024, seed=42) on the
same lmsys-chat-1m first-user contexts (hard assert: no silent fallback corpus).

Haiku concurrency decision (option b): u2 turns are produced with synchronous
``client.messages.create`` calls fanned out over a ThreadPoolExecutor(16) with
exponential backoff on rate-limit / overload errors, failing loud after 6 retries
(chosen over the Batch API: n<=2000 short calls, minutes-scale latency).

Outputs are persisted as JSONL + a meta JSON and uploaded to the HF data repo via ONE
``upload_folder`` call per track. ``--smoke`` uses n=3 and skips the upload; Track M on
a CPU-only box additionally requires ``--assistant-fixture <jsonl>`` (canned a-turns).

vLLM is imported lazily inside the functions that need it so the CPU smoke path never
imports it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# vLLM V1 fork-EngineCore guard (gotchas.md): the parent touches CUDA (_require_cuda)
# before vllm.LLM(); fork then dies with "Cannot re-initialize CUDA in forked
# subprocess". Spawn BEFORE any vllm import (crash att-20260702-061417).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.experiments.issue_825.common import (
    GEN_SEED,
    HF_DATA_REPO,
    HF_PREFIX,
    MAX_CONV_TOKENS,
    MIN_TURN_CONTENT_TOKENS,
    MODEL_INSTRUCT,
    N_TRACK_M,
    N_TRACK_S,
)
from explore_persona_space.orchestrate.env import load_dotenv

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# claude-haiku-4-5-20251001 is used here as a GENERATOR of simulated user turns,
# NOT as a judge — no judged DV flows through it (--check-judge-model-pins).
# API_DISPATCH_ROUTING_EXEMPT: parent-round (#825) u2 generator predating the
# api_dispatch routing lint; bounded 16-thread fan-out with its own 429/529
# exponential backoff (_haiku_user_turn); not invoked by the current
# real-user-turn-null round (no generation anywhere in it).
HAIKU_GEN_MODEL = "claude-haiku-4-5-20251001"
_HAIKU_MAX_RETRIES = 6
_HAIKU_WORKERS = 16
_LMSYS_REPO = "lmsys/lmsys-chat-1m"
_WILDCHAT_REPO = "allenai/WildChat-1M"

# >=20 diverse user-persona briefs, rotated deterministically by conversation index.
USER_BRIEFS: list[tuple[str, str]] = [
    (
        "skeptical_followup",
        "You are skeptical of the assistant's answer. Push back on its weakest claim and "
        "ask it to justify that claim.",
    ),
    (
        "clarification_seeker",
        "One part of the answer confused you. Paraphrase the confusing part and ask for a "
        "plainer explanation.",
    ),
    (
        "topic_deepener",
        "The answer was fine but shallow for you. Ask a follow-up that digs one level "
        "deeper into the same topic.",
    ),
    (
        "mild_disagreer",
        "You mostly agree but hold one differing opinion. State it politely and ask what "
        "the assistant makes of it.",
    ),
    (
        "anecdote_adder",
        "Share a short, plausible personal experience related to the topic, then ask how "
        "it fits with the assistant's answer.",
    ),
    (
        "terse_power_user",
        "You are a terse expert. Reply in one or two short sentences demanding a specific "
        "detail the answer skipped.",
    ),
    (
        "confused_novice",
        "You are new to this topic and a bit lost. Ask a basic question about a term or "
        "step the assistant used.",
    ),
    (
        "example_requester",
        "Ask for a concrete worked example that illustrates the assistant's main point.",
    ),
    (
        "edge_case_prober",
        "Think of an unusual edge case or exception and ask whether the assistant's "
        "answer still holds there.",
    ),
    (
        "summarizer_checker",
        "Summarize the assistant's answer in your own words and ask whether you got it right.",
    ),
    (
        "practical_applier",
        "Describe a specific real situation of yours in one sentence and ask how to apply "
        "the answer to it.",
    ),
    (
        "comparison_asker",
        "Ask how the thing discussed compares to a well-known alternative you name.",
    ),
    (
        "source_demander",
        "Ask where the assistant's claims come from and how confident it is in them.",
    ),
    (
        "devils_advocate",
        "Argue the opposite position for the sake of argument and ask the assistant to "
        "respond to it.",
    ),
    (
        "goal_shifter",
        "Reveal your actual underlying goal, which your first message only hinted at, and "
        "ask for advice on it.",
    ),
    (
        "constraint_adder",
        "Add a new constraint (budget, time, tools, or skill level) and ask how the "
        "answer changes under it.",
    ),
    (
        "list_requester",
        "Ask the assistant to reformulate its answer as a short list of concrete steps or options.",
    ),
    (
        "emotional_reactor",
        "React with genuine emotion (excitement, worry, or frustration) to the answer, "
        "then ask a follow-up.",
    ),
    (
        "tangent_taker",
        "Pick a side detail from the answer and take the conversation in that new but "
        "related direction.",
    ),
    (
        "verifier",
        "Describe how you would test or verify the assistant's answer and ask whether "
        "that check is sound.",
    ),
    (
        "simplifier",
        "Ask the assistant to explain the same answer so a smart twelve-year-old would get it.",
    ),
    (
        "time_traveler",
        "Ask how the answer would have differed a decade ago, or how it might change in "
        "the future.",
    ),
]


def _require_cuda(context: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{context} requires a CUDA GPU for vLLM generation and none is available. "
            "For a CPU smoke of track m, pass --assistant-fixture <jsonl> with canned "
            "a-turns."
        )


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_INSTRUCT)


def _build_llm():
    from vllm import LLM

    return LLM(model=MODEL_INSTRUCT)


def _render(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _rf():
    """Lazy import of the sibling render module (same scripts/ dir)."""
    import issue825_render_formats

    return issue825_render_formats


def _ntok(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# Matches issue779_collect.VLLM_CHUNK_SIZE (same env var + default) without
# importing that module's heavy dependency chain — code-review round-1
# blocker: the symbol does NOT exist in issue779_common.
VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))


def _vllm_generate_chunked(llm, prompt_texts: list[str], sampling_params) -> list[str]:
    out: list[str] = []
    for i in range(0, len(prompt_texts), VLLM_CHUNK_SIZE):
        chunk = prompt_texts[i : i + VLLM_CHUNK_SIZE]
        chunk_out = llm.generate(chunk, sampling_params, use_tqdm=False)
        for o in chunk_out:
            out.append(o.outputs[0].text)
    return out


def _is_english(row: dict) -> bool:
    lang = row.get("language")
    return isinstance(lang, str) and lang.strip().lower() == "english"


def _is_clean(row: dict) -> bool:
    """Non-toxic / non-redacted per whichever flags this dataset row carries."""
    if row.get("toxic") is True:
        return False
    if row.get("redacted") is True:
        return False
    mod = row.get("openai_moderation")
    if isinstance(mod, list | tuple) and mod:
        first = mod[0]
        if isinstance(first, dict) and first.get("flagged") is True:
            return False
    return True


def harvest_openers(n: int, tokenizer) -> list[dict[str, str]]:
    """Stream first-user turns from lmsys then WildChat; English, clean, deduped."""
    from datasets import load_dataset

    openers: list[dict[str, str]] = []
    seen: set[str] = set()
    for repo, tag in ((_LMSYS_REPO, "lmsys"), (_WILDCHAT_REPO, "wildchat")):
        if len(openers) >= n:
            break
        ds = load_dataset(repo, split="train", streaming=True)
        for row in ds:
            if len(openers) >= n:
                break
            if not _is_english(row) or not _is_clean(row):
                continue
            conv = row.get("conversation") or []
            if not conv or conv[0].get("role") != "user":
                continue
            u1 = (conv[0].get("content") or "").strip()
            if not u1:
                continue
            key = u1[:200]
            if key in seen:
                continue
            if _ntok(tokenizer, u1) < MIN_TURN_CONTENT_TOKENS:
                continue
            seen.add(key)
            openers.append({"u1": u1, "opener_source": tag})
        print(f"[harvest] after {repo}: {len(openers)}/{n} openers")
    if len(openers) < n:
        raise RuntimeError(
            f"Opener harvest exhausted both sources at {len(openers)}/{n} eligible openers."
        )
    return openers


def _haiku_user_turn(client, brief_text: str, u1: str, a1: str) -> str:
    import anthropic

    system = (
        "You are simulating a human USER in an ongoing conversation with an AI assistant, "
        f"generating research data. {brief_text} Write ONLY the user's next message: "
        "natural, first-person, no role labels, no meta-commentary."
    )
    content = (
        f"Conversation so far:\n\n[USER]\n{u1}\n\n[ASSISTANT]\n{a1}\n\n"
        "Now write the user's next message."
    )
    last_exc: Exception | None = None
    for attempt in range(_HAIKU_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=HAIKU_GEN_MODEL,  # generation call (simulated user turn), NOT a judge
                max_tokens=512,
                temperature=1.0,
                system=system,
                messages=[{"role": "user", "content": content}],
            )
            return resp.content[0].text.strip()
        except anthropic.APIStatusError as exc:
            if exc.status_code not in (429, 500, 529):
                raise
            last_exc = exc
            time.sleep(min(2.0**attempt, 30.0))
        except anthropic.APIConnectionError as exc:
            last_exc = exc
            time.sleep(min(2.0**attempt, 30.0))
    raise RuntimeError(
        f"Haiku u2 call failed after {_HAIKU_MAX_RETRIES} rate-limit retries: {last_exc!r}"
    )


def _distinct_3gram_rate(texts: list[str]) -> float:
    total = 0
    distinct: set[tuple[str, ...]] = set()
    for text in texts:
        words = text.split()
        for j in range(len(words) - 2):
            total += 1
            distinct.add(tuple(words[j : j + 3]))
    return (len(distinct) / total) if total else 0.0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _upload(out_dir: Path, file_names: list[str], path_in_repo: str) -> None:
    from huggingface_hub import upload_folder

    upload_folder(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        folder_path=str(out_dir),
        path_in_repo=path_in_repo,
        allow_patterns=file_names,
        commit_message=f"issue-825: upload {path_in_repo}",
    )
    print(f"[upload] {file_names} -> {HF_DATA_REPO}/{path_in_repo}")


def run_track_m(n: int, seed: int, out: Path, smoke: bool, fixture: Path | None) -> None:
    if smoke:
        n = 3
    tokenizer = _load_tokenizer()
    # Over-provision ~30% so post-generation filters can drop rows without
    # underfilling the fixed n=2000 design (code-review round-1: silent
    # underfill; plan fixes n and changing n is must-ask). Kept rows are
    # trimmed back to exactly n; a shortfall below n fails loud.
    # 1.10 was insufficient: the production run measured a 13.4% combined drop
    # (short_turn 73 + too_long 221 of 2200) -> 1906 kept < 2000 (crash-fix
    # round 2). At the observed 86.6% keep rate, 1.30x -> ~2252 expected kept.
    n_target = n
    n = max(n, -(-n * 13 // 10)) if not smoke else n  # ceil(1.30*n) without numpy
    openers = harvest_openers(n, tokenizer)
    mix: dict[str, int] = {"lmsys": 0, "wildchat": 0}
    for o in openers:
        mix[o["opener_source"]] += 1

    use_fixture = fixture is not None
    if use_fixture:
        fixture_rows = [json.loads(line) for line in fixture.read_text().splitlines() if line]
        if len(fixture_rows) < len(openers):
            raise RuntimeError(
                f"--assistant-fixture has {len(fixture_rows)} rows; need {len(openers)}."
            )
        a1s = [r["a1"] for r in fixture_rows[: len(openers)]]
        llm = None
        greedy = None
    else:
        _require_cuda("track m")
        from vllm import SamplingParams

        llm = _build_llm()
        greedy = SamplingParams(temperature=0.0, max_tokens=1024, seed=seed)
        a1_prompts = [_render(tokenizer, [{"role": "user", "content": o["u1"]}]) for o in openers]
        a1s = _vllm_generate_chunked(llm, a1_prompts, greedy)
    print(f"[track m] a1 done for {len(a1s)} conversations")
    # Checkpoint-per-phase: persist a1s the moment the vLLM phase completes so
    # a Haiku-phase crash cannot lose the GPU work (code-review round-1).
    ckpt_a1 = out.parent / f"{out.stem}_a1_checkpoint.jsonl"
    ckpt_a1.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_a1, "w", encoding="utf-8") as fh:
        for o, a1 in zip(openers, a1s, strict=True):
            fh.write(json.dumps({"u1": o["u1"], "a1": a1}) + "\n")

    import anthropic

    client = anthropic.Anthropic()
    briefs = [USER_BRIEFS[i % len(USER_BRIEFS)] for i in range(len(openers))]
    work = [
        (brief_text, o["u1"], a1)
        for (_bid, brief_text), o, a1 in zip(briefs, openers, a1s, strict=True)
    ]
    with ThreadPoolExecutor(max_workers=_HAIKU_WORKERS) as pool:
        u2s = list(pool.map(lambda args: _haiku_user_turn(client, *args), work))
    print(f"[track m] u2 done for {len(u2s)} conversations")

    if use_fixture:
        a2s = [r["a2"] for r in fixture_rows[: len(openers)]]
    else:
        a2_prompts = [
            _render(
                tokenizer,
                [
                    {"role": "user", "content": o["u1"]},
                    {"role": "assistant", "content": a1},
                    {"role": "user", "content": u2},
                ],
            )
            for o, a1, u2 in zip(openers, a1s, u2s, strict=True)
        ]
        a2s = _vllm_generate_chunked(llm, a2_prompts, greedy)
    print(f"[track m] a2 done for {len(a2s)} conversations")

    kept: list[dict] = []
    drops = {"short_turn": 0, "too_long": 0}
    for i, (o, a1, (bid, _bt), u2, a2) in enumerate(
        zip(openers, a1s, briefs, u2s, a2s, strict=True)
    ):
        toks = [_ntok(tokenizer, t) for t in (o["u1"], a1, u2, a2)]
        if min(toks) < MIN_TURN_CONTENT_TOKENS:
            drops["short_turn"] += 1
            continue
        # Length filter on the RENDERED sequence (headers included), max over
        # both formats — raw turn-token sums under-count by the header/
        # terminator overhead (round-2 review minor).
        row_probe = {"conv_id": i, "u1": o["u1"], "a1": a1, "u2": u2, "a2": a2}
        rendered_len = max(
            len(_rf().render_chat(row_probe, tokenizer).input_ids),
            len(_rf().render_naturalistic(row_probe, tokenizer).input_ids),
        )
        if rendered_len > MAX_CONV_TOKENS:
            drops["too_long"] += 1
            continue
        kept.append(
            {
                "conv_id": i,
                "u1": o["u1"],
                "a1": a1,
                "u2": u2,
                "a2": a2,
                "opener_source": o["opener_source"],
                "brief_id": bid,
            }
        )

    # Trim to exactly the requested design size; fail loud on a shortfall
    # (plan fixes n; changing n is must-ask — never silently underfill).
    if len(kept) < n_target:
        raise RuntimeError(
            f"track m underfilled: {len(kept)} kept < target {n_target} after "
            f"filters (drops={drops}); raise the over-provision factor or fix "
            f"the filters — do NOT run underpowered"
        )
    kept = kept[:n_target]

    meta = {
        "track": "m",
        "n_requested": n_target,
        "n_overprovisioned": n,
        "n_openers": len(openers),
        "n_kept": len(kept),
        "drops": drops,
        "opener_source_mix": mix,
        "distinct_3gram_rate_u2": _distinct_3gram_rate([r["u2"] for r in kept]),
        "seed": seed,
        "model_instruct": MODEL_INSTRUCT,
        "u2_model": HAIKU_GEN_MODEL,
        "n_briefs": len(USER_BRIEFS),
        "min_turn_content_tokens": MIN_TURN_CONTENT_TOKENS,
        "max_conv_tokens": MAX_CONV_TOKENS,
        "smoke": smoke,
        "assistant_fixture": str(fixture) if fixture else None,
    }
    _write_jsonl(out, kept)
    meta_path = out.with_name(out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[track m] kept {len(kept)}/{len(openers)} (drops={drops}) -> {out}")
    if smoke:
        print("[smoke] skipping HF upload")
    else:
        _upload(out.parent, [out.name, meta_path.name], f"{HF_PREFIX}/raw_completions/generation")


def run_track_s(out: Path, smoke: bool) -> None:
    from issue779_collect import load_train_contexts

    prompts, source = load_train_contexts(N_TRACK_S, smoke)
    if source != _LMSYS_REPO:
        raise RuntimeError(
            f"Track S requires the gated '{_LMSYS_REPO}' dataset but load_train_contexts "
            f"fell back to '{source}'. Accept the lmsys-chat-1m gate for this HF token "
            f"(https://huggingface.co/datasets/{_LMSYS_REPO}) and re-run — a fallback "
            "corpus would break the #779 replication anchor."
        )
    from huggingface_hub import dataset_info

    revision = dataset_info(_LMSYS_REPO).sha

    _require_cuda("track s")
    from vllm import SamplingParams

    tokenizer = _load_tokenizer()
    llm = _build_llm()
    # PARENT-EXACT sampling params (#779 issue779_collect.run_pass_b) — seed literal 42.
    sampling = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1024, seed=42)
    prompt_texts = [_render(tokenizer, [{"role": "user", "content": p}]) for p in prompts]
    responses = _vllm_generate_chunked(llm, prompt_texts, sampling)

    rows = [
        {"prompt_idx": i, "prompt": p, "response": r}
        for i, (p, r) in enumerate(zip(prompts, responses, strict=True))
    ]
    meta = {
        "track": "s",
        "n": len(rows),
        "source": source,
        "dataset_revision": revision,
        "model_instruct": MODEL_INSTRUCT,
        "sampling": {"n": 1, "temperature": 1.0, "top_p": 0.95, "max_tokens": 1024, "seed": 42},
        "smoke": smoke,
    }
    _write_jsonl(out, rows)
    meta_path = out.with_name(out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[track s] wrote {len(rows)} rows (revision={revision}) -> {out}")
    if smoke:
        print("[smoke] skipping HF upload")
    else:
        _upload(out.parent, [out.name, meta_path.name], f"{HF_PREFIX}/raw_completions/track_s")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", choices=("m", "s"), required=True)
    ap.add_argument("--n", type=int, default=N_TRACK_M, help="track m conversation count")
    ap.add_argument("--seed", type=int, default=GEN_SEED)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="n=3, no HF upload")
    ap.add_argument(
        "--assistant-fixture",
        type=Path,
        default=None,
        help="JSONL of canned {a1, a2} rows (CPU smoke for track m, replaces vLLM)",
    )
    args = ap.parse_args()
    if args.track == "m":
        out = args.out or Path("data/issue_825/conversations.jsonl")
        run_track_m(args.n, args.seed, out, args.smoke, args.assistant_fixture)
    else:
        out = args.out or Path("data/issue_825/track_s.jsonl")
        run_track_s(out, args.smoke)


if __name__ == "__main__":
    main()
