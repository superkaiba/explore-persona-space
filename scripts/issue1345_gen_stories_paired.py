#!/usr/bin/env python
"""Issue #1345 conversation-paired-stories round — gen_stories_paired phase.

Renders a seed-42 subsample of N_STORIES_PAIRED_TARGET (2,700) of the SAME
shared S-track conversations into narrative prose with the EXACT original
answer embedded verbatim (`ARIA replied: "<original answer>"`), one Q->A
exchange per story (plan v8 §4). Tier-2 instruct-and-strip: the verbatim
constraint lives in the system prompt and is STRIPPED before extraction.
Keep = mechanical ANSWER-ANCHORED span gate (exactly ONE verbatim answer
occurrence, opened by exactly ONE attribution marker, quote closed right after
it, quoted question before the marker — see match_verbatim_turn's rationale
comment for why quote-pairing recovery is not used) AND judge PASS (reason-then-verdict,
criteria: one exchange / verbatim match / no pre-slot answer revelation —
the two NEW criteria vs the r3 rubric, plan v8 §4.3). Yield floor 2,160/2,700
after one retry batch -> rc=21 story-regime halt (plan v8 §7); smoke floor 1.

--op-companion: the N<=200 on-policy control cell (plan v8 §4.5, the #1335
tf/op calibration shape) — seed-0 sample of the KEPT paired conversations,
free generation WITHOUT the verbatim answer (the model writes its own answer);
one-exchange rubric. kept < 2 -> rc=23 (companion unusable; the TF headline
proceeds, calibration reports N/A).

Content hygiene: questions/answers are LMSYS-derived real user text and
stories are raw model generations — this script logs COUNTS/ids only.

CLI:
  uv run python scripts/issue1345_gen_stories_paired.py --model instruct [--smoke]
  uv run python scripts/issue1345_gen_stories_paired.py --model instruct --op-companion
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

# #628 fork-poisoning guard (gotchas.md): main() loads the tokenizer BEFORE the
# vLLM engine builds, so the V1 EngineCore must spawn, not fork. Set BEFORE any
# `import vllm` (vLLM reads the var at import time; the imports are deferred).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402
import issue1345_gen_stories as g  # noqa: E402 — vLLM chunking + HF persist helpers

from explore_persona_space.llm.api_dispatch import (  # noqa: E402
    RESULT_RATE_LIMITED,
    RESULT_TRANSPORT,
    DispatchItem,
    dispatch_calls,
)

SMOKE_N_STORIES = 3
# Generation feasibility bound: the story must reproduce the verbatim answer
# within STORY_MAX_NEW_TOKENS; leave >=224 tokens for wrapper prose + question.
ANSWER_TOKEN_BUDGET = c.STORY_MAX_NEW_TOKENS - 224
# Degenerate-answer floor (the #825 zero-width-span class; matches the
# parser's answer_len_ok lower bound).
ANSWER_CHAR_MIN = 20

# Built from c.STORY_CHARACTER_NAME (the variant guard in common pins ARIA for
# this round unless --variant re-scopes). {ANSWER} is filled per row; the
# instruction is STRIPPED before extraction (tier-2 instruct-and-strip).
STORY_PAIRED_SYSTEM_TEMPLATE = (
    f"You are writing a short narrative story scene in which an AI assistant named "
    f"{c.STORY_CHARACTER_NAME} is a character. The scene must contain exactly ONE moment "
    f"where a person asks {c.STORY_CHARACTER_NAME} a question and "
    f"{c.STORY_CHARACTER_NAME} answers it. Write 3-6 sentences of narrative context first, "
    "then have the person ask the question given by the user (spoken aloud, in double "
    f"quotes), and then have {c.STORY_CHARACTER_NAME} answer with EXACTLY these words: "
    '"{ANSWER}". Introduce the answer with an attribution before the quotation, exactly '
    f'in the form: {c.STORY_CHARACTER_NAME} replied: "..." — the quoted answer must match '
    "the given words verbatim, with no paraphrasing, additions, or omissions. Do not "
    "reveal or paraphrase the answer content anywhere before that attribution. Write "
    "flowing narrative prose (never a script or 'Name:' dialogue format), and put all "
    "spoken dialogue in double quotes."
)
STORY_OP_COMPANION_SYSTEM = (
    f"You are writing a short narrative story scene in which an AI assistant named "
    f"{c.STORY_CHARACTER_NAME} is a character. The scene must contain exactly ONE moment "
    f"where a person asks {c.STORY_CHARACTER_NAME} a question and "
    f"{c.STORY_CHARACTER_NAME} gives a substantive, helpful answer. Write 3-6 sentences "
    "of narrative context first, then have the person ask the question given by the user "
    f"(spoken aloud, in double quotes), and then have {c.STORY_CHARACTER_NAME} answer it. "
    f"Introduce the answer with an attribution before the quotation, exactly in the form: "
    f'{c.STORY_CHARACTER_NAME} replied: "..." Write flowing narrative prose (never a '
    "script or 'Name:' dialogue format), and put all spoken dialogue in double quotes."
)

JUDGE_SYSTEM_PAIRED = (
    "You evaluate a short story scene against three criteria. The user message gives the "
    "story and the required ANSWER text. The story PASSES only if ALL three hold: "
    "(1) the story contains exactly ONE question->answer exchange, in which a person asks "
    "a question in quoted dialogue and an AI assistant character named "
    f"{c.STORY_CHARACTER_NAME} answers in quoted dialogue introduced by an attribution "
    f'before the quote (e.g. {c.STORY_CHARACTER_NAME} replied: "..."); '
    "(2) the quoted answer matches the required ANSWER text verbatim — not paraphrased, "
    "shortened, or extended; "
    "(3) the answer content is not revealed or paraphrased anywhere BEFORE the attribution "
    "marker — the prose before it may reference the question topic but must not state the "
    "answer. First give 2-3 sentences of reasoning, then finish with EXACTLY two final "
    "lines:\nEXCHANGES: <integer number of qualifying exchanges>\nVERDICT: PASS or FAIL"
)
JUDGE_SYSTEM_OP = (
    "You evaluate a short story scene. The story PASSES only if it contains exactly ONE "
    "question->answer exchange, in which a person asks a question in quoted dialogue and "
    f"an AI assistant character named {c.STORY_CHARACTER_NAME} gives a substantive answer "
    "in quoted dialogue introduced by an attribution before the quote (e.g. "
    f'{c.STORY_CHARACTER_NAME} replied: "..."). Exchanges with trivial or empty answers, '
    f"unquoted dialogue, or answers not attributed to {c.STORY_CHARACTER_NAME} do not "
    "count. First give 2-3 sentences of reasoning, then finish with EXACTLY two final "
    "lines:\nEXCHANGES: <integer number of qualifying exchanges>\nVERDICT: PASS or FAIL"
)


# ---------------------------------------------------------------------------
# Judge plumbing (reason-then-verdict; drop-never-coerce + transport re-drive)
# ---------------------------------------------------------------------------
def _build_judge_request(item: DispatchItem) -> dict:
    """Messages-API params for one paired-story judge call (system top-level)."""
    payload = item.payload
    if payload.get("mode") == "op":
        system = JUDGE_SYSTEM_OP
        user = payload["story"]
    else:
        system = JUDGE_SYSTEM_PAIRED
        user = f"REQUIRED ANSWER:\n{payload['answer']}\n\nSTORY:\n{payload['story']}"
    return {
        "model": c.JUDGE_MODEL,
        "max_tokens": c.JUDGE_MAX_TOKENS,
        "temperature": 0.0,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }


def _parse_judge_response(text: str) -> dict:
    """Extract EXCHANGES/VERDICT from a reason-then-verdict reply (raise on miss)."""
    exchanges, verdict = None, None
    for line in text.split("\n"):
        s = line.strip()
        if s.upper().startswith("EXCHANGES:"):
            with contextlib.suppress(ValueError, IndexError):
                exchanges = int(s.split(":", 1)[1].strip().split()[0])
        if s.upper().startswith("VERDICT:"):
            v = s.split(":", 1)[1].strip().upper()
            if v in ("PASS", "FAIL"):
                verdict = v
    if verdict is None:
        raise ValueError(f"judge reply missing VERDICT line ({len(text)} chars)")
    return {"verdict": verdict, "judge_exchanges": exchanges}


# ---------------------------------------------------------------------------
# Seed pool: matched-n allowlist (staged by prefetch_reuse) x track_s Q/A text
# ---------------------------------------------------------------------------
def load_paired_pool(matched_dir: Path, dl_dir: Path) -> tuple[list[dict], dict]:
    """{conv_id, question, answer} rows for the shared conv set + filter counts.

    Source of the 4,724 shared conv ids: the parent's staged matched-n
    allowlist (matched_subsets_parent.json, staged by prefetch_reuse at the
    pinned REUSE_REV). Answers = the ORIGINAL track-S responses @ PIN_REV —
    the exact text the r1/r2 stores teacher-forced, so the r4 answer
    distribution is IDENTICAL to chat/plain-text (plan v8 §4).
    """
    parent_matched = matched_dir / "matched_subsets_parent.json"
    assert parent_matched.exists(), (
        f"staged parent matched-n allowlist missing: {parent_matched} — run the "
        "prefetch_reuse phase first (it stages the pinned allowlist)"
    )
    shared = {str(x) for x in json.loads(parent_matched.read_text())["shared_r1r2_convs"]}
    track_s = c.stage_pinned_file(c.PARENT_TRACK_S_JSONL, dl_dir)
    rows = c.read_jsonl(track_s)
    counts = {
        "shared_convs": len(shared),
        "joined": 0,
        "answer_too_short": 0,
        "kept": 0,
    }
    pool = []
    for r in rows:
        cid = str(r.get("conv_id") or f"s{r['prompt_idx']}")
        if cid not in shared:
            continue
        counts["joined"] += 1
        answer = r["response"]
        if len(answer) < ANSWER_CHAR_MIN:
            # Degenerate near-empty answers (the #825 zero-width-span class);
            # answer-anchored matching handles embedded quotes, so no quote
            # drop — the token budget in _filter_pool_feasible bounds length.
            counts["answer_too_short"] += 1
            continue
        pool.append({"conv_id": cid, "question": r["prompt"], "answer": answer})
        counts["kept"] += 1
    assert pool, "paired seed pool empty — allowlist did not join against track_s.jsonl"
    print(f"[seeds] paired pool filter counts: {counts}", flush=True)
    return sorted(pool, key=lambda r: r["conv_id"]), counts


def build_paired_prompt(row: dict, tokenizer, *, op_companion: bool) -> str:
    """Chat-templated tier-2 generation prompt (instruction stripped later)."""
    system = (
        STORY_OP_COMPANION_SYSTEM
        if op_companion
        else STORY_PAIRED_SYSTEM_TEMPLATE.replace("{ANSWER}", row["answer"])
    )
    user_msg = (
        f"Write the scene now. The question the person asks {c.STORY_CHARACTER_NAME} is:\n"
        f"{row['question']}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _filter_pool_feasible(pool: list[dict], tokenizer, *, op_companion: bool) -> tuple[list, dict]:
    """Drop rows whose formatted prompt / answer exceed the token budgets."""
    kept, counts = [], {"prompt_over_budget": 0, "answer_over_budget": 0}
    for row in pool:
        n_ans = len(tokenizer(row["answer"], add_special_tokens=False)["input_ids"])
        if not op_companion and n_ans > ANSWER_TOKEN_BUDGET:
            counts["answer_over_budget"] += 1
            continue
        prompt = build_paired_prompt(row, tokenizer, op_companion=op_companion)
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok > g.PROMPT_TOKEN_BUDGET:
            counts["prompt_over_budget"] += 1
            continue
        kept.append(row)
    print(f"[seeds] feasibility filter: kept={len(kept)} dropped={counts}", flush=True)
    return kept, counts


# ---------------------------------------------------------------------------
# Mechanical span gate — ANSWER-ANCHORED (the extraction contract).
#
# Why not parse_story_turns' quote-pairing span recovery (the r3 path): the
# original track-S answers are REAL LMSYS responses and 29% of the shared pool
# carries embedded double quotes — a closing-quote scan ends the span early on
# every such row, so quote-parser matching caps the eligible pool at 2,293
# conversations, BELOW the plan's 2,700 target (measured at implementation
# time, plan v8 §12.2 verify note). Because the r4 construction KNOWS the
# verbatim answer text, the span is located by its unique occurrence instead
# (eligible pool 3,843): the ANSWER_ATTRIB_RE attribution marker must open a
# quote immediately before the occurrence (identical context-slot definition
# to r3 — last token of `ARIA replied: "`), exactly ONE attribution in the
# story, a closing quote right after the answer, and no verbatim leak before
# the marker. Turn dicts keep the parse_story_turns shape verbatim, so
# render_story_turn (offsets + BPE seam guard) is reused untouched.
# ---------------------------------------------------------------------------
def _answer_occurrences(story: str, answer: str) -> list[int]:
    out, i = [], story.find(answer)
    while i != -1:
        out.append(i)
        i = story.find(answer, i + 1)
    return out


def match_verbatim_turn(story: str, answer: str) -> tuple[dict | None, str]:
    """(turn, reason) — the answer-anchored single-exchange verbatim gate.

    Returns a parse_story_turns-shaped turn dict on 'ok'; otherwise (None,
    <keep-filter counter>). Enforces: exactly one verbatim answer occurrence,
    exactly one attribution marker (one exchange) whose opening quote sits
    immediately before the occurrence, a closing quote right after it, a
    quoted question utterance before the marker, and no pre-slot answer leak.
    """
    occ = _answer_occurrences(story, answer)
    if len(occ) != 1:
        return None, "answer_occurrences_zero" if not occ else "answer_occurrences_multi"
    a_start = occ[0]
    a_end = a_start + len(answer)
    if story[a_end : a_end + 1] not in ('"', "”"):
        return None, "answer_quote_not_closed"
    attribs = list(c.ANSWER_ATTRIB_RE.finditer(story))
    if len(attribs) != 1:
        return None, "attribution_zero" if not attribs else "attribution_multi"
    m = attribs[0]
    if m.end(1) != a_start:  # the attributed quote must open right at the answer
        return None, "attribution_not_adjacent_to_answer"
    marker_text = story[m.start() : m.end(1) - 1].rstrip()
    marker_end = m.start() + len(marker_text)
    q_spans = c._quoted_spans_before(story, m.start())
    if not q_spans:
        return None, "question_utterance_missing"
    q_open, q_close = None, None
    for qo, qc in reversed(q_spans):
        if "?" in story[qo + 1 : qc]:
            q_open, q_close = qo, qc
            break
    question_is_question = q_open is not None
    if q_open is None:
        q_open, q_close = q_spans[-1]
    # (No separate pre-slot-leak branch: a verbatim answer before the marker is
    # either a SECOND occurrence -> answer_occurrences_multi, or the only one
    # -> attribution_not_adjacent_to_answer. Paraphrase leaks are the judge's.)
    turn = {
        "q_start": q_open,
        "q_end": q_close + 1,
        "marker_end": marker_end,
        "a_start": a_start,
        "a_end": a_end,
        "confidence": {
            "marker_exact": marker_text.endswith(":"),
            "answer_len_ok": 20 <= (a_end - a_start) <= 2000,
            "question_found": True,
            "question_is_question": bool(question_is_question),
        },
    }
    if not (turn["q_end"] <= turn["marker_end"] < turn["a_start"] < turn["a_end"]):
        return None, "span_ordering_degenerate"
    return turn, "ok"


def confident_op_turn(story: str) -> tuple[dict | None, str]:
    """Companion keep-filter: exactly one parsed turn with a confident answer."""
    turns = c.parse_story_turns(story)
    if len(turns) != 1:
        return None, f"parsed_turns_{'zero' if not turns else 'multi'}"
    t = turns[0]
    conf = t["confidence"]
    if not (conf["marker_exact"] or conf["answer_len_ok"]):
        return None, "low_confidence_turn"
    return t, "ok"


# ---------------------------------------------------------------------------
# Generation (chunked vLLM + fingerprint-gated JSONL resume — parent pattern)
# ---------------------------------------------------------------------------
def paired_fingerprint(mode: str, rows: list[dict]) -> str:
    """Content key over everything that determines the kept bundle (r6 rule)."""
    import inspect

    key = json.dumps(
        {
            "mode": mode,
            "gen_seed": c.GEN_SEED,
            "temperature": c.STORY_TEMPERATURE,
            "max_new_tokens": c.STORY_MAX_NEW_TOKENS,
            "system_template": (
                STORY_OP_COMPANION_SYSTEM if mode == "op" else STORY_PAIRED_SYSTEM_TEMPLATE
            ),
            "rows_sha": hashlib.sha256(
                json.dumps(
                    [(r["conv_id"], r["question"], r.get("answer", "")) for r in rows],
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "judge_model": c.JUDGE_MODEL,
            "judge_system": JUDGE_SYSTEM_OP if mode == "op" else JUDGE_SYSTEM_PAIRED,
            "judge_max_tokens": c.JUDGE_MAX_TOKENS,
            # The keep-filter recipe IS part of the bundle identity: any change
            # to the matcher regenerates rather than reusing stale stories.
            "parser_source_sha": hashlib.sha256(
                inspect.getsource(
                    confident_op_turn if mode == "op" else match_verbatim_turn
                ).encode()
            ).hexdigest(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def generate_paired(
    rows: list[dict], out_path: Path, fp: str, tokenizer, llm, *, op_companion: bool
) -> list[dict]:
    """One story per row, chunked + per-chunk JSONL checkpoint, keyed on conv_id."""
    from vllm import SamplingParams

    meta_path = out_path.with_suffix(".meta.json")
    done_ids: set[str] = set()
    if out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp:
            done_ids = {r["conv_id"] for r in c.read_jsonl(out_path)}
            print(f"[gen] resume: {len(done_ids)} stories already on disk", flush=True)
        else:
            raise RuntimeError(
                f"{out_path} exists with a DIFFERENT generation fingerprint "
                f"({meta.get('fingerprint')} != {fp}) — refusing to mix regimes; "
                "move the stale file aside"
            )
    else:
        c.write_json(meta_path, {"fingerprint": fp, "n_rows": len(rows)})

    todo = [r for r in rows if r["conv_id"] not in done_ids]
    sampling = SamplingParams(
        temperature=c.STORY_TEMPERATURE, max_tokens=c.STORY_MAX_NEW_TOKENS, seed=None
    )
    n_chunks = (len(todo) + g.VLLM_CHUNK_SIZE - 1) // g.VLLM_CHUNK_SIZE
    for ci in range(0, len(todo), g.VLLM_CHUNK_SIZE):
        chunk = todo[ci : ci + g.VLLM_CHUNK_SIZE]
        prompts = [build_paired_prompt(r, tokenizer, op_companion=op_companion) for r in chunk]
        print(
            f"[vllm-chunk] paired gen chunk {ci // g.VLLM_CHUNK_SIZE + 1}/{n_chunks} "
            f"({len(chunk)} prompts, op_companion={op_companion})",
            flush=True,
        )
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        new_rows = []
        for r, o in zip(chunk, outs, strict=True):
            new_rows.append(
                {
                    "conv_id": r["conv_id"],
                    # story_id == conv_id: one story per conversation (paired
                    # by construction; extraction groups by conv_id).
                    "story_id": r["conv_id"],
                    "question": r["question"],
                    "answer": r.get("answer", ""),
                    "mode": "op" if op_companion else "paired",
                    "tier": "instruct_and_strip",
                    "story": o.outputs[0].text.strip(),
                    "finish_reason": o.outputs[0].finish_reason,
                }
            )
        c.append_jsonl(out_path, new_rows)
    return c.read_jsonl(out_path)


# ---------------------------------------------------------------------------
# Parse + judge (mechanical gate first, then the LLM judge)
# ---------------------------------------------------------------------------
def parse_and_judge_paired(
    rows: list[dict], cache_dir: Path, smoke: bool, *, op_companion: bool
) -> tuple[list[dict], dict, list[dict]]:
    """(kept, counts, judge_digest_rows) — keep = mechanical PASS ∧ judge PASS."""
    mech: dict[str, tuple[dict | None, str]] = {}
    for r in rows:
        mech[r["conv_id"]] = (
            confident_op_turn(r["story"])
            if op_companion
            else match_verbatim_turn(r["story"], r["answer"])
        )

    items = [
        DispatchItem(
            item_id=r["conv_id"],
            payload={"story": r["story"], "answer": r.get("answer", ""), "mode": r["mode"]},
        )
        for r in rows
    ]
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

    counts = {
        "n_generated": len(rows),
        "mech_pass": 0,
        "mech_fail": 0,
        "mech_fail_reasons": {},
        "judge_pass": 0,
        "judge_fail": 0,
        "judge_malformed": 0,
        "transport_loss": 0,
        "kept": 0,
    }
    kept, digest_rows = [], []
    for r in rows:
        turn, reason = mech[r["conv_id"]]
        digest = {"conv_id": r["conv_id"], "mode": r["mode"], "mech_reason": reason}
        if turn is None:
            counts["mech_fail"] += 1
            counts["mech_fail_reasons"][reason] = counts["mech_fail_reasons"].get(reason, 0) + 1
        else:
            counts["mech_pass"] += 1
        res = results[r["conv_id"]]
        if res.error:
            key = (
                "transport_loss"
                if res.category in (RESULT_RATE_LIMITED, RESULT_TRANSPORT)
                else "judge_malformed"
            )
            counts[key] += 1
            digest_rows.append({**digest, "judge_error_category": res.category})
            continue
        verdict = res.result["verdict"]
        digest_rows.append(
            {**digest, "verdict": verdict, "judge_exchanges": res.result.get("judge_exchanges")}
        )
        counts["judge_pass" if verdict == "PASS" else "judge_fail"] += 1
        if verdict != "PASS" or turn is None:
            continue
        kept.append(
            {
                **r,
                "judge_verdict": verdict,
                "judge_exchanges": res.result.get("judge_exchanges"),
                "parsed_turns": [turn],
                "n_parsed_turns": 1,
            }
        )
        counts["kept"] += 1
    return kept, counts, digest_rows


# ---------------------------------------------------------------------------
# Bundle persist / resume (reuses the parent's HF boundary helpers)
# ---------------------------------------------------------------------------
def bundle_files_paired(mode_slug: str, model_key: str, out_dir: Path) -> list[str]:
    names = [
        f"raw_stories_{mode_slug}_{model_key}.jsonl",
        f"raw_stories_{mode_slug}_{model_key}.meta.json",
        f"raw_stories_{mode_slug}_{model_key}_retry.jsonl",
        f"raw_stories_{mode_slug}_{model_key}_retry.meta.json",
        f"kept_stories_{mode_slug}_{model_key}.jsonl",
        f"story_yield_{mode_slug}_{model_key}.json",
        f"judge_results_{mode_slug}_{model_key}.jsonl",
    ]
    return [n for n in names if (out_dir / n).exists()]


def persist_bundle_paired(mode_slug: str, model_key: str, out_dir: Path, fp: str, smoke: bool):
    """Upload rollout text + judge digests + yield report to HF NOW (r6 rule)."""
    import os

    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — cannot persist story bundle"
    files = bundle_files_paired(mode_slug, model_key, out_dir)
    assert f"kept_stories_{mode_slug}_{model_key}.jsonl" in files, files
    manifest = {
        "metadata": c.metadata(c.GEN_SEED, len(files), "scripts/issue1345_gen_stories_paired.py"),
        "model": model_key,
        "mode": mode_slug,
        "bundle_fingerprint": fp,
        "files": files,
    }
    c.write_json(out_dir / f"story_bundle_manifest_{mode_slug}_{model_key}.json", manifest)
    prefix = g._stories_hf_prefix(smoke)
    g._hf_upload_folder(
        out_dir,
        prefix,
        [f"*{mode_slug}_{model_key}*"],
        f"issue-1345: {mode_slug} story bundle ({model_key}, fp {fp})",
    )
    print(f"[gen] persisted {mode_slug} rollouts -> {prefix} (fp {fp})", flush=True)


def try_resume_paired(
    mode_slug: str, model_key: str, fp: str, out_dir: Path, smoke: bool
) -> dict | None:
    """Reuse a persisted kept bundle when its fingerprint matches (r6 rule)."""
    prefix = g._stories_hf_prefix(smoke)
    manifest_path = f"{prefix}/story_bundle_manifest_{mode_slug}_{model_key}.json"
    if not g._hf_file_exists(manifest_path):
        return None
    local = g._hf_download_to(
        manifest_path, out_dir / f"story_bundle_manifest_{mode_slug}_{model_key}.json"
    )
    manifest = json.loads(local.read_text())
    if manifest.get("bundle_fingerprint") != fp:
        print(
            f"[gen] HF {mode_slug} bundle fingerprint {manifest.get('bundle_fingerprint')} "
            f"!= {fp} — stale (recipe changed); regenerating",
            flush=True,
        )
        return None
    for name in manifest["files"]:
        g._hf_download_to(f"{prefix}/{name}", out_dir / name)
    report = json.loads((out_dir / f"story_yield_{mode_slug}_{model_key}.json").read_text())
    print(
        f"[gen] resume-from-HF: {report.get('n_kept')} kept {mode_slug} stories "
        f"(fp {fp}) — generation skipped",
        flush=True,
    )
    return report


def _build_llm(model_id: str):
    import os

    from vllm import LLM

    return LLM(
        model=model_id,
        seed=c.GEN_SEED,
        dtype="bfloat16",
        max_model_len=g.MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1",
        enable_prefix_caching=(
            False if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "0") == "1" else None
        ),
    )


def _write_yield_report(
    mode_slug: str,
    model_key: str,
    out_dir: Path,
    *,
    n_target: int,
    yield_floor: int,
    counts_main: dict,
    counts_retry: dict | None,
    pool_counts: dict,
    n_kept: int,
) -> None:
    c.write_json(
        out_dir / f"story_yield_{mode_slug}_{model_key}.json",
        {
            "metadata": c.metadata(c.GEN_SEED, n_kept, "scripts/issue1345_gen_stories_paired.py"),
            "model": model_key,
            "mode": mode_slug,
            "story_character_name": c.STORY_CHARACTER_NAME,
            "n_target": n_target,
            "yield_floor": yield_floor,
            "pool_filter_counts": pool_counts,
            "counts_main": counts_main,
            "counts_retry": counts_retry,
            "n_kept": n_kept,
            "yield_ok": n_kept >= yield_floor,
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=("instruct",), default="instruct")
    ap.add_argument("--out-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--dl-dir", type=Path, default=c.PARENT_DL_DIR)
    ap.add_argument("--matched-dir", type=Path, default=c.MATCHED_DIR)
    ap.add_argument("--n-stories", type=int, default=c.N_STORIES_PAIRED_TARGET)
    ap.add_argument("--yield-floor", type=int, default=c.STORY_PAIRED_YIELD_FLOOR)
    ap.add_argument(
        "--op-companion",
        action="store_true",
        help="generate the N<=200 on-policy companion control cell (plan v8 §4.5)",
    )
    ap.add_argument("--smoke", action="store_true", help="n=3 stories, sync judge")
    args = ap.parse_args()

    assert c.HAS_R4, (
        "gen_stories_paired requires EPM_I1345_VARIANT=conversation_paired_stories "
        f"(got {c.VARIANT!r}) — the r4 registry and variant-scoped output dirs are "
        "gated on it (never clobber the parent run)"
    )
    assert args.model in c.R4_MODELS, args.model
    model_key = args.model
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_slug = "paired_op" if args.op_companion else "paired"
    cache_dir = out_dir / f"judge_cache_{mode_slug}"

    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue_825.common import MODEL_INSTRUCT

    tokenizer = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)

    import numpy as np

    if args.op_companion:
        kept_paired = out_dir / f"kept_stories_paired_{model_key}.jsonl"
        assert kept_paired.exists(), (
            f"{kept_paired} missing — run the paired gen phase before --op-companion"
        )
        kept_rows = c.read_jsonl(kept_paired)
        rng = np.random.default_rng(c.OP_COMPANION_SEED)
        n_take = min(c.OP_COMPANION_N, len(kept_rows)) if not args.smoke else min(3, len(kept_rows))
        idx = rng.choice(len(kept_rows), size=n_take, replace=False)
        rows_main = [
            {"conv_id": kept_rows[i]["conv_id"], "question": kept_rows[i]["question"]}
            for i in sorted(int(i) for i in idx)
        ]
        rows_main, pool_counts = _filter_pool_feasible(rows_main, tokenizer, op_companion=True)
        n_target, yield_floor = len(rows_main), 2  # companion: usable at >=2 rows (rc=23)
        seeds_reserve: list[dict] = []
    else:
        pool, pool_counts = load_paired_pool(args.matched_dir, args.dl_dir)
        pool, feas_counts = _filter_pool_feasible(pool, tokenizer, op_companion=False)
        pool_counts.update(feas_counts)
        n_target = SMOKE_N_STORIES if args.smoke else args.n_stories
        yield_floor = g.resolve_yield_floor(args.smoke, args.yield_floor)
        assert len(pool) >= n_target, f"paired pool {len(pool)} < target {n_target}"
        rng = np.random.default_rng(c.GEN_SEED)
        order = rng.permutation(len(pool))
        rows_main = [pool[i] for i in order[:n_target]]
        seeds_reserve = [pool[i] for i in order[n_target:]]

    fp = paired_fingerprint(mode_slug, rows_main + seeds_reserve)
    resumed = try_resume_paired(mode_slug, model_key, fp, out_dir, args.smoke)
    if resumed is not None:
        n_kept = int(resumed["n_kept"])
        if args.op_companion:
            if n_kept < 2:
                print(f"[companion] unusable: kept={n_kept} < 2 — rc=23", flush=True)
                raise SystemExit(23)
        else:
            g.enforce_yield_floor(n_kept, yield_floor)
        print(f"[done] {mode_slug} kept={n_kept}/{n_target} (resumed)", flush=True)
        return

    llm = _build_llm(MODEL_INSTRUCT)
    raw_path = out_dir / f"raw_stories_{mode_slug}_{model_key}.jsonl"
    rows = generate_paired(rows_main, raw_path, fp, tokenizer, llm, op_companion=args.op_companion)
    kept, counts, judge_digest = parse_and_judge_paired(
        rows, cache_dir, args.smoke, op_companion=args.op_companion
    )

    retry_counts = None
    if not args.op_companion and len(kept) < n_target and seeds_reserve:
        shortfall = n_target - len(kept)
        retry_rows_in = seeds_reserve[:shortfall]
        print(f"[retry] {len(kept)}/{n_target} kept — one retry batch of {len(retry_rows_in)}")
        retry_path = out_dir / f"raw_stories_{mode_slug}_{model_key}_retry.jsonl"
        retry_rows = generate_paired(
            retry_rows_in, retry_path, fp, tokenizer, llm, op_companion=False
        )
        retry_kept, retry_counts, retry_digest = parse_and_judge_paired(
            retry_rows, cache_dir, args.smoke, op_companion=False
        )
        kept.extend(retry_kept)
        judge_digest.extend(retry_digest)

    kept_path = out_dir / f"kept_stories_{mode_slug}_{model_key}.jsonl"
    if kept_path.exists():
        kept_path.unlink()
    c.append_jsonl(kept_path, kept)
    judge_path = out_dir / f"judge_results_{mode_slug}_{model_key}.jsonl"
    if judge_path.exists():
        judge_path.unlink()
    c.append_jsonl(judge_path, judge_digest)

    _write_yield_report(
        mode_slug,
        model_key,
        out_dir,
        n_target=n_target,
        yield_floor=yield_floor,
        counts_main=counts,
        counts_retry=retry_counts,
        pool_counts=pool_counts,
        n_kept=len(kept),
    )
    # Persist BEFORE any floor can halt this process and before extraction
    # (Upload Policy raw-completions rule; #1345 crash-fix r6 shape).
    persist_bundle_paired(mode_slug, model_key, out_dir, fp, args.smoke)

    if args.op_companion:
        if len(kept) < 2:
            print(f"[companion] unusable: kept={len(kept)} < 2 — rc=23", flush=True)
            raise SystemExit(23)
    else:
        if args.smoke and len(kept) < n_target:
            print(
                f"[yield-floor][smoke] shortfall: kept={len(kept)}/{n_target} — proceeding "
                "(smoke floor=1 so extract_r4_tf is exercised)",
                flush=True,
            )
        g.enforce_yield_floor(len(kept), yield_floor)
    print(f"[done] {mode_slug} kept={len(kept)}/{n_target} stories", flush=True)


if __name__ == "__main__":
    main()
