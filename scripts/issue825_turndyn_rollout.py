#!/usr/bin/env python
"""#825 turn-dynamics-allturns-5000 P1 arm-G rollout worker (plan v24 §4 P1).

Per subject model (instruct | pretrained), roll each seed conversation out to
``K_gen`` assistant turns: at each depth step the subject model generates its
own answer (parent-exact SamplingParams — chunked vLLM, use_tqdm=False,
per-chunk INFO logs), then claude-haiku-4-5 generates the next user turn (the
Track-M simulated-USER recipe extended u2 -> u2..u{K_gen}; ONE persona brief
per conversation, fixed across turns and shared across both subject models),
routed through ``llm/api_dispatch.py`` (SYNC fan-out at the Haiku family cap;
the Track-M script predates the dispatcher — this extension routes through it).

Work-conserving double-buffering: this worker's conversation shard is split
into two half-waves; while one half's Haiku wave runs in a background thread,
the other half generates on the GPU (plan §9 P1). Checkpoint-per-depth-step:
every (depth, half) writes its JSONL the moment gen+haiku complete, and a
fingerprint-gated resume (keyed on EVERY output-affecting regime flag) skips
completed steps on relaunch.

G-B pilot: ``--pilot-n 50`` runs THIS SAME production path on 50 seeds
(PASS_UNIFIED — no architectural fork); ``--report`` computes the per-depth
degeneracy diagnostics (user-turn distinct-2 ratio, max within-conv cross-turn
user-turn cosine, role-leak regex rate, plus round-11 length telemetry:
user/answer word medians, closer rate, died-reason histogram) from the
persisted rollout text.

Simulated-user template: ``SIM_USER_PROMPT_VERSION`` (v2, round-11 G-B fix) —
constrains the simulator to real-user-like turns (1-3 sentences, ONE focused
question, no closers); the version is a resume-fingerprint regime key.

Content hygiene: seeds + rollouts are REAL-USER-derived text. Conversation /
generation text is never printed or logged — only counts and paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# vLLM v1 forks its EngineCore by default; parent CUDA init kills the child —
# spawn BEFORE any vllm import (round-10 crash att-20260715-141100).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
from issue825_gen_conversations import HAIKU_GEN_MODEL  # noqa: E402
from issue825_onpolicy_turn_depth_gpu import (  # noqa: E402
    GEN_MAX_TOKENS,
    GEN_N,
    GEN_SEED,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    MODEL_SPEC,
    _render_gen_prompt,
)
from issue825_turndyn_harvest import read_jsonl_stem  # noqa: E402
from issue1092_gpu_phase import (  # noqa: E402
    DEFAULT_VLLM_CHUNK_SIZE,
    _render_full_conversation,
    _token_len,
)

from explore_persona_space.llm.api_dispatch import (  # noqa: E402
    DispatchItem,
    dispatch_calls,
)

logger = logging.getLogger("i825_turndyn_rollout")

HAIKU_MAX_TOKENS = 512  # Track-M recipe (issue825_gen_conversations._haiku_user_turn)
HAIKU_TEMPERATURE = 1.0
WAVE_FAIL_HARD_RATE = 0.05  # >5% haiku failures in one wave = systemic, fail loud
# Output-affecting regime key for the simulated-user SYSTEM TEMPLATE (#722 r3):
# a resume must never mix waves generated under different templates. Bump on
# any template change. v2 = round-11 G-B realism fix (att-20260716-012315):
# the v1 Track-M template produced user turns unlike real users (median 124
# words vs 24 for the real corpus u1, ~2 questions/turn vs 0), whose
# multi-part expansive asks elicited ~3x longer instruct answers (464 vs 152
# words median; 17% at the 1024-token cap) and overflowed the depth-24 token
# budget (died_reasons 100% window/capture overflow) -> G-B completion 0.22.
SIM_USER_PROMPT_VERSION = "v2-realistic-brevity"
ROLE_LEAK_RE = re.compile(
    r"(^\s*(assistant|ai|user|human)\s*:)|\bas an ai\b|\bas a language model\b",
    re.IGNORECASE | re.MULTILINE,
)
# WARN-only telemetry (run_report): conversation-closer phrasing in simulated
# user turns. NOT a gate — the G-B thresholds are pre-registered (plan §7).
CLOSER_RE = re.compile(
    r"\b(thanks|thank you|bye|goodbye|that'?s all|no further questions"
    r"|no more questions|this (was|has been) (really |very )?helpful)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# state + checkpoints
# ---------------------------------------------------------------------------


def _fingerprint(args: argparse.Namespace, seeds_sha: str) -> dict:
    """Resume fingerprint over EVERY output-affecting regime key (#722 r3)."""
    return {
        "model": args.model,
        "k_gen": args.k_gen,
        "seeds_sha256": seeds_sha,
        "pilot_n": args.pilot_n,
        "shard": args.shard,
        "sampling": {
            "n": GEN_N,
            "temperature": GEN_TEMPERATURE,
            "top_p": GEN_TOP_P,
            "max_tokens": GEN_MAX_TOKENS,
            "seed": GEN_SEED,
        },
        "haiku_model": args.haiku_model,
        "haiku_max_tokens": HAIKU_MAX_TOKENS,
        "sim_user_prompt": SIM_USER_PROMPT_VERSION,
        "capture_budget": args.capture_budget,
        "engine_max_len": args.engine_max_len,
        "smoke": bool(args.smoke),
    }


def _step_path(out_root: Path, k: int, half: str) -> Path:
    return out_root / f"step{k:02d}_{half}.jsonl"


def _write_step(out_root: Path, k: int, half: str, rows: list[dict]) -> None:
    """Atomic per-(depth, half) checkpoint (ASCII-escaped JSONL)."""
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = _step_path(out_root, k, half).with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, _step_path(out_root, k, half))


def _read_step(out_root: Path, k: int, half: str) -> list[dict] | None:
    p = _step_path(out_root, k, half)
    if not p.exists():
        return None
    rows: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _build_engine(args: argparse.Namespace):
    if args.smoke:
        return None
    name, revision, _stops = MODEL_SPEC[args.model]
    from vllm import LLM

    return LLM(
        model=name,
        revision=revision,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=GEN_SEED,
        gpu_memory_utilization=0.85,
        max_model_len=args.engine_max_len,
    )


def _generate(llm, prompts: list[str], args: argparse.Namespace, tag: str) -> list[dict]:
    """Chunked vLLM generate (use_tqdm=False; per-chunk INFO). Smoke: canned."""
    if args.smoke:
        return [
            {
                "text": f"Canned smoke answer at {tag} item {i}: a short deterministic reply.",
                "finish_reason": "smoke_canned",
                "n_gen_tokens": None,
            }
            for i, _p in enumerate(prompts)
        ]
    from vllm import SamplingParams

    _name, _rev, stop_tokens = MODEL_SPEC[args.model]
    params = SamplingParams(
        n=GEN_N,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_TOKENS,
        seed=GEN_SEED,
        stop=list(stop_tokens),
    )
    results: list[dict] = []
    chunk_size = max(1, int(args.chunk_size))
    n_chunks = math.ceil(len(prompts) / chunk_size)
    for start in range(0, len(prompts), chunk_size):
        chunk = prompts[start : start + chunk_size]
        logger.info(
            "[gen %s] %s vLLM chunk %d/%d (%d prompts)",
            tag,
            args.model,
            start // chunk_size + 1,
            n_chunks,
            len(chunk),
        )
        outputs = llm.generate(chunk, params, use_tqdm=False)
        for out in outputs:
            top = out.outputs[0] if out.outputs else None
            results.append(
                {
                    "text": top.text if top is not None else "",
                    "finish_reason": top.finish_reason if top is not None else "no_output",
                    "n_gen_tokens": len(top.token_ids) if top is not None else 0,
                }
            )
    return results


# ---------------------------------------------------------------------------
# simulated user (Haiku via api_dispatch)
# ---------------------------------------------------------------------------


def _sim_user_system(brief_text: str) -> str:
    """Simulated-user system prompt (Track-M recipe + round-11 G-B realism fix).

    v2 (`SIM_USER_PROMPT_VERSION`): pins the simulator to the MEASURED real-user
    turn distribution (pilot att-20260716-012315: real corpus u1 median 24
    words, 0 question marks) and bans conversation closers. The v1 template's
    long multi-question turns elicited instruct answers ~3x longer than real
    user turns do, overflowing the depth-24 token budget (plan §7 G-B fix-the-
    simulator remedy; thresholds untouched).
    """
    return (
        "You are simulating a human USER in an ongoing conversation with an AI assistant, "
        f"generating research data. {brief_text} Write ONLY the user's next message. "
        "Real chat users write SHORT messages: 1-3 sentences, under 40 words. "
        "Ask exactly ONE focused question (or make ONE request) about a SPECIFIC point in "
        "the assistant's last reply — never a list of questions, never 'tell me everything "
        "about X'. Plain, informal register; first person; no role labels, no "
        "meta-commentary, no pleasantries or thanks. Never wrap up, close, or say goodbye — "
        "always keep the conversation going with a new question or request."
    )


def _sim_user_content(turns: list[dict]) -> str:
    blocks = []
    for t in turns:
        label = "USER" if t["role"] == "user" else "ASSISTANT"
        blocks.append(f"[{label}]\n{t['content']}")
    return (
        "Conversation so far:\n\n" + "\n\n".join(blocks) + "\n\nNow write the user's next message."
    )


def _run_haiku_wave(
    wave_items: list[tuple[str, str, list[dict]]],
    k_next: int,
    args: argparse.Namespace,
    cache_root: Path,
) -> dict[str, dict]:
    """One per-depth Haiku fan-out for [(conv_id, brief_text, turns)] -> user turn k_next.

    Returns {conv_id: {"text": str} | {"error": reason}}. Routed through
    api_dispatch (sync path — latency-coupled to the GPU pipeline, plan §9;
    the per-item content cache gives wave-level resume for free).
    """
    import asyncio

    if not wave_items:
        return {}
    items = [
        DispatchItem(item_id=f"{cid}:u{k_next}", payload={"brief": brief, "turns": turns})
        for cid, brief, turns in wave_items
    ]

    def build_request(item: DispatchItem) -> dict:
        # System prompt lifted to the top-level param (Messages API has no
        # system role — api_dispatch enforces this at build time; gotchas #906).
        return {
            "model": args.haiku_model,
            "max_tokens": HAIKU_MAX_TOKENS,
            "temperature": HAIKU_TEMPERATURE,
            "system": _sim_user_system(item.payload["brief"]),
            "messages": [{"role": "user", "content": _sim_user_content(item.payload["turns"])}],
        }

    results = asyncio.run(
        dispatch_calls(
            items,
            model=args.haiku_model,
            build_request=build_request,
            parse_response=lambda text: text.strip(),
            cache_dir=cache_root / "haiku_cache",
            checkpoint_dir=cache_root / "haiku_ckpt",
            force_path="sync",
        )
    )
    out: dict[str, dict] = {}
    n_err = 0
    for cid, _brief, _turns in wave_items:
        res = results[f"{cid}:u{k_next}"]
        if res.error or not isinstance(res.result, str) or not res.result.strip():
            n_err += 1
            out[cid] = {"error": res.reason or "empty_haiku_response"}
        else:
            out[cid] = {"text": res.result}
    if n_err:
        rate = n_err / len(wave_items)
        # reason histogram: the hard-fail raise previously discarded WHY items
        # failed (round-11 crash-fix smoke: a 33% wave failure with no reason
        # in the log). Reasons are dispatch metadata, never conversation text.
        reasons: dict[str, int] = {}
        for v in out.values():
            if "error" in v:
                reasons[str(v["error"])[:120]] = reasons.get(str(v["error"])[:120], 0) + 1
        logger.warning(
            "[haiku u%d] %d/%d failed (%.1f%%); reasons=%s",
            k_next,
            n_err,
            len(wave_items),
            100 * rate,
            reasons,
        )
        if rate > WAVE_FAIL_HARD_RATE:
            raise RuntimeError(
                f"[haiku u{k_next}] failure rate {100 * rate:.1f}% > "
                f"{100 * WAVE_FAIL_HARD_RATE:.0f}% — systemic dispatch problem, not attrition; "
                f"reasons={reasons}"
            )
    return out


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------


def _load_seeds(args: argparse.Namespace) -> list[dict]:
    seeds = read_jsonl_stem(Path(args.seeds_dir), "armG_seeds")
    seeds.sort(key=lambda s: int(s["seed_rank"]))
    if args.pilot_n:
        seeds = seeds[: args.pilot_n]
    si, sn = (int(v) for v in args.shard.split("/"))
    assert 0 <= si < sn, args.shard
    return [s for i, s in enumerate(seeds) if i % sn == si]


def run_rollout(args: argparse.Namespace) -> None:  # noqa: C901 — linear wave driver
    seeds = _load_seeds(args)
    seeds_sha = hashlib.sha256(
        "\n".join(f"{s['conv_id']}:{s['brief_id']}" for s in seeds).encode()
    ).hexdigest()
    out_root = Path(args.out_dir) / args.model / f"shard{args.shard.replace('/', 'of')}"
    out_root.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(args, seeds_sha)
    fp_path = out_root / "rollout_fingerprint.json"
    if fp_path.exists():
        with open(fp_path) as f:
            old = json.load(f)
        if old != fp:
            diff = sorted(k for k in set(old) | set(fp) if old.get(k) != fp.get(k))
            raise SystemExit(
                f"[rollout] fingerprint MISMATCH on keys {diff}; refusing to resume — "
                f"move {out_root} aside or fix the flags (regime-keyed resume, #722 r3)"
            )
        logger.info("[rollout] fingerprint match: resuming %s", out_root)
    else:
        with open(fp_path, "w") as f:
            json.dump(fp, f, indent=1)
    # fix-engaged signal (round-11 G-B crash-fix): the v2 simulator template is live.
    logger.info("[rollout] sim_user_prompt=%s", SIM_USER_PROMPT_VERSION)

    from transformers import AutoTokenizer

    if args.smoke:
        tok = AutoTokenizer.from_pretrained(args.tiny_model_dir, trust_remote_code=True)
        # Smoke hermeticity: pre-seed the lazy instruct-render tokenizer cache
        # (the tiny dir carries the SAME real Qwen tokenizer files — round-10).
        import issue1092_gpu_phase as gp

        gp._get_tokenizer._tok = tok
    else:
        name, revision, _stops = MODEL_SPEC[args.model]
        tok = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)

    # per-conv state
    state: dict[str, dict] = {}
    for s in seeds:
        state[s["conv_id"]] = {
            "conv_id": s["conv_id"],
            "seed_rank": s["seed_rank"],
            "brief_id": s["brief_id"],
            "brief_text": s["brief_text"],
            "turns": [{"role": "user", "content": s["u1"]}],
            "alive": True,
            "died_at": None,
            "died_reason": None,
        }
    order = [s["conv_id"] for s in seeds]
    halves = {"a": order[0::2], "b": order[1::2]}

    # ---- resume: replay completed (k, half) checkpoints in order ----
    resume_k = {"a": 0, "b": 0}
    replay_open = {"a": True, "b": True}
    for k in range(1, args.k_gen + 1):
        for half in ("a", "b"):
            if not replay_open[half]:
                continue
            rows = _read_step(out_root, k, half)
            if rows is None:
                # stop at the FIRST missing step per half — steps are written
                # strictly in order, so replaying past a hole would corrupt
                # the turn sequence.
                replay_open[half] = False
                continue
            for r in rows:
                st = state[r["conv_id"]]
                # order matters: the step-k row carries user turn k (drained
                # haiku wave; None at k=1 — the seed u1 is already in state)
                # THEN the generated answer k.
                if r.get("user") is not None:
                    st["turns"].append({"role": "user", "content": r["user"]})
                if r.get("answer") is not None:
                    st["turns"].append({"role": "assistant", "content": r["answer"]})
                st["alive"] = bool(r["alive"])
                st["died_at"] = r.get("died_at")
                st["died_reason"] = r.get("died_reason")
            resume_k[half] = k
    if any(resume_k.values()):
        logger.info(
            "[rollout] resumed: half a at step %d, half b at step %d", resume_k["a"], resume_k["b"]
        )

    llm = _build_engine(args)
    executor = ThreadPoolExecutor(max_workers=1)
    pending: dict[str, tuple[int, object] | None] = {"a": None, "b": None}

    # ---- resume: re-submit the IN-FLIGHT Haiku wave the crash dropped ----
    # The step-k checkpoint persists user turn k + answer k, but the turn-k+1
    # wave (submitted at step k) lived only in the executor. Without
    # re-submission, `pending[half]` is None at step resume_k+1 and EVERY live
    # conv dies `state_desync_no_user_turn` (code-review v21 Critical 1). The
    # api_dispatch per-item content cache (keyed `cid:u{k+1}` under out_root)
    # makes an already-completed wave free on re-submission.
    for half in ("a", "b"):
        rk = resume_k[half]
        if 0 < rk < args.k_gen:
            wave_items = [
                (cid, state[cid]["brief_text"], list(state[cid]["turns"]))
                for cid in halves[half]
                if state[cid]["alive"]
            ]
            logger.info(
                "[rollout] resume: re-submitting haiku wave u%d for half %s (%d live convs)",
                rk + 1,
                half,
                len(wave_items),
            )
            pending[half] = (
                rk + 1,
                executor.submit(_run_haiku_wave, wave_items, rk + 1, args, out_root),
            )
    t0 = time.time()

    def _process_half(half: str, k: int) -> None:
        """One (depth k, half) unit: drain the half's pending Haiku wave (user
        turn k, submitted at step k-1), generate the answers, checkpoint, then
        submit the next Haiku wave in the background (double-buffering)."""
        if k <= resume_k[half]:
            return  # completed on a prior run (checkpoint replayed above)
        user_k: dict[str, str] = {}
        ans_k: dict[str, str] = {}
        # 1) drain pending haiku wave (user turn k) — turn 1 comes from the seed
        if pending[half] is not None:
            wk, fut = pending[half]
            assert wk == k, (wk, k)
            haiku = fut.result()
            pending[half] = None
            for cid in halves[half]:
                st = state[cid]
                if not st["alive"]:
                    continue
                res = haiku.get(cid)
                if res is None or "error" in res:
                    st["alive"] = False
                    st["died_at"] = k
                    st["died_reason"] = f"haiku_failed:{(res or {}).get('error', 'missing')}"
                    continue
                user_k[cid] = res["text"]
                st["turns"].append({"role": "user", "content": res["text"]})

        # 2) window checks + prompt build for live convs
        live: list[str] = []
        prompts: list[str] = []
        for cid in halves[half]:
            st = state[cid]
            if not st["alive"]:
                continue
            if st["turns"][-1]["role"] != "user":
                st["alive"] = False
                st["died_at"] = k
                st["died_reason"] = "state_desync_no_user_turn"
                continue
            prompt = _render_gen_prompt(st["turns"], args.model, tok)
            n_tok = _token_len(tok, prompt)
            if n_tok + GEN_MAX_TOKENS > args.engine_max_len:
                st["alive"] = False
                st["died_at"] = k
                st["died_reason"] = "window_overflow"
                continue
            live.append(cid)
            prompts.append(prompt)

        # 3) generate answers
        gens = _generate(llm, prompts, args, tag=f"t{k}{half}")
        assert len(gens) == len(live), (len(gens), len(live))
        gen_meta: dict[str, dict] = {}
        for cid, g in zip(live, gens, strict=True):
            st = state[cid]
            # per-answer generation metadata rides the step checkpoint (round-11
            # G-B post-mortem: at-cap answers were invisible in persisted data)
            gen_meta[cid] = {
                "finish_reason": g.get("finish_reason"),
                "n_gen_tokens": g.get("n_gen_tokens"),
            }
            content = str(g["text"]).strip()
            if not content:
                st["alive"] = False
                st["died_at"] = k
                st["died_reason"] = "empty_completion"
                continue
            ans_k[cid] = content
            st["turns"].append({"role": "assistant", "content": content})
            # capture-budget attrition: a conv whose CAPTURE render (the full
            # conversation, the P3 teacher-forced input) already exceeds the
            # capture budget cannot recover (renders only grow) — plan §4 arm G.
            render = _render_full_conversation(st["turns"], args.model)
            if _token_len(tok, render) > args.capture_budget:
                st["alive"] = False
                st["died_at"] = k
                st["died_reason"] = "capture_budget_overflow"

        # 4) submit the next haiku wave (user turn k+1) in the background
        if k < args.k_gen:
            wave_items = [
                (cid, state[cid]["brief_text"], list(state[cid]["turns"]))
                for cid in halves[half]
                if state[cid]["alive"]
            ]
            fut = executor.submit(_run_haiku_wave, wave_items, k + 1, args, out_root)
            pending[half] = (k + 1, fut)

        # 5) checkpoint this (k, half) the moment it completes
        rows = []
        for cid in halves[half]:
            st = state[cid]
            meta = gen_meta.get(cid, {})
            rows.append(
                {
                    "conv_id": cid,
                    "depth": k,
                    "user": user_k.get(cid),  # user turn k (None at k=1: seed u1)
                    "answer": ans_k.get(cid),  # generated assistant turn k
                    "alive": st["alive"],
                    "died_at": st["died_at"],
                    "died_reason": st["died_reason"],
                    # diagnostics only — the resume replay ignores these keys
                    "finish_reason": meta.get("finish_reason"),
                    "n_gen_tokens": meta.get("n_gen_tokens"),
                }
            )
        _write_step(out_root, k, half, rows)
        n_alive = sum(1 for cid in halves[half] if state[cid]["alive"])
        logger.info(
            "[rollout] %s step %d/%d half %s: %d/%d alive (%.0fs)",
            args.model,
            k,
            args.k_gen,
            half,
            n_alive,
            len(halves[half]),
            time.time() - t0,
        )

    try:
        for k in range(1, args.k_gen + 1):
            for half in ("a", "b"):
                _process_half(half, k)
    finally:
        executor.shutdown(wait=True)
        if llm is not None:
            del llm  # dispatch-level gpu-guard is the authoritative reaper

    # ---- final full-conversation dump + attrition summary ----
    final_rows = [
        {
            "conv_id": cid,
            "seed_rank": state[cid]["seed_rank"],
            "brief_id": state[cid]["brief_id"],
            "turns": state[cid]["turns"],
            "alive": state[cid]["alive"],
            "died_at": state[cid]["died_at"],
            "died_reason": state[cid]["died_reason"],
        }
        for cid in order
    ]
    from issue825_turndyn_harvest import _write_jsonl_sharded

    _write_jsonl_sharded(final_rows, out_root, "rollout_final")
    per_depth_alive = {
        str(k): sum(1 for cid in order if state[cid]["alive"] or (state[cid]["died_at"] or 0) > k)
        for k in range(1, args.k_gen + 1)
    }
    died = [cid for cid in order if not state[cid]["alive"]]
    summary = {
        "model": args.model,
        "shard": args.shard,
        "n_seeds": len(order),
        "n_completed": len(order) - len(died),
        "completion_rate": (len(order) - len(died)) / max(1, len(order)),
        "per_depth_alive": per_depth_alive,
        "died_reasons": {
            r: sum(1 for cid in died if str(state[cid]["died_reason"]).startswith(r))
            for r in (
                "window_overflow",
                "capture_budget_overflow",
                "empty_completion",
                "haiku_failed",
                "state_desync",
            )
        },
        "k_gen": args.k_gen,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_root / "rollout_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    logger.info(
        "[rollout] %s DONE: %d/%d completed to depth %d",
        args.model,
        summary["n_completed"],
        len(order),
        args.k_gen,
    )


# ---------------------------------------------------------------------------
# degeneracy diagnostics (plan §6 / gate G-B heuristics)
# ---------------------------------------------------------------------------


def _distinct2(texts: list[str]) -> float:
    total, distinct = 0, set()
    for t in texts:
        w = t.split()
        for j in range(len(w) - 1):
            total += 1
            distinct.add((w[j], w[j + 1]))
    return (len(distinct) / total) if total else float("nan")


def _bow_vec(text: str, dim: int = 1 << 15) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for w in text.lower().split():
        v[hash(w) % dim] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def run_report(args: argparse.Namespace) -> None:
    """Per-depth degeneracy heuristics from the persisted rollout text."""
    model_root = Path(args.out_dir) / args.model
    shard_dirs = sorted(p for p in model_root.glob("shard*") if p.is_dir())
    assert shard_dirs, f"no rollout shards under {model_root}"
    convs: list[dict] = []
    for sd in shard_dirs:
        convs.extend(read_jsonl_stem(sd, "rollout_final"))
    per_depth: dict[str, dict] = {}
    max_depth = max((sum(1 for t in c["turns"] if t["role"] == "user") for c in convs), default=0)
    for k in range(2, max_depth + 1):
        users_k: list[str] = []
        leak = 0
        cos_vals: list[float] = []
        for c in convs:
            uturns = [t["content"] for t in c["turns"] if t["role"] == "user"]
            if len(uturns) < k:
                continue
            uk = uturns[k - 1]
            users_k.append(uk)
            if ROLE_LEAK_RE.search(uk):
                leak += 1
            # max cosine between user turn k and any EARLIER user turn (lexical
            # hashed-BOW embedding — the cheap within-conv repetition read)
            vk = _bow_vec(uk)
            best = max((float(vk @ _bow_vec(u)) for u in uturns[: k - 1]), default=float("nan"))
            cos_vals.append(best)
        if not users_k:
            continue
        u_words = sorted(len(u.split()) for u in users_k)
        per_depth[str(k)] = {
            "n": len(users_k),
            "distinct2": _distinct2(users_k),
            "max_crossturn_cosine_mean": float(np.nanmean(cos_vals)) if cos_vals else None,
            "max_crossturn_cosine_p90": (
                float(np.nanpercentile(cos_vals, 90)) if cos_vals else None
            ),
            "role_leak_rate": leak / len(users_k),
            # round-11 G-B length telemetry (WARN-only; NOT gate inputs — the
            # pre-registered gate reads completion + role_leak only, plan §7)
            "user_words_median": u_words[len(u_words) // 2],
            "closer_rate": sum(1 for u in users_k if CLOSER_RE.search(u)) / len(users_k),
        }
    # per-depth ANSWER length telemetry (separate map: k=1 exists here, and the
    # G-B gate iterates per_depth expecting the degeneracy keys on every node)
    answers_per_depth: dict[str, dict] = {}
    for k in range(1, max_depth + 1):
        a_words = sorted(
            len(a[k - 1].split())
            for c in convs
            if len(a := [t["content"] for t in c["turns"] if t["role"] == "assistant"]) >= k
        )
        if a_words:
            answers_per_depth[str(k)] = {
                "n": len(a_words),
                "answer_words_median": a_words[len(a_words) // 2],
                "answer_words_p90": a_words[min(len(a_words) - 1, int(0.9 * len(a_words)))],
            }
    died_reasons: dict[str, int] = {}
    for c in convs:
        if not c["alive"]:
            died_reasons[str(c.get("died_reason"))] = (
                died_reasons.get(str(c.get("died_reason")), 0) + 1
            )
    n_completed = sum(1 for c in convs if c["alive"])
    out = {
        "model": args.model,
        "n_conversations": len(convs),
        "n_completed": n_completed,
        "completion_rate": n_completed / max(1, len(convs)),
        "sim_user_prompt": SIM_USER_PROMPT_VERSION,
        "per_depth": per_depth,
        "answers_per_depth": answers_per_depth,
        "died_reasons": died_reasons,
        "embedding": "hashed bag-of-words (2^15), L2-normalized — lexical repetition read",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = model_root / "rollout_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    logger.info(
        "[report] %s: %d convs, completion %.1f%% -> %s",
        args.model,
        len(convs),
        100 * out["completion_rate"],
        out_path,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=("instruct", "pretrained"))
    ap.add_argument("--seeds-dir", required=True, help="dir with armG_seeds_shard*.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k-gen", type=int, default=24)
    ap.add_argument("--pilot-n", type=int, default=0, help="G-B pilot: first N seeds (0 = all)")
    ap.add_argument("--shard", default="0/1", help="i/n conversation shard (per-GPU engine)")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_VLLM_CHUNK_SIZE)
    ap.add_argument("--capture-budget", type=int, default=15872)
    ap.add_argument("--engine-max-len", type=int, default=16384)
    ap.add_argument("--haiku-model", default=HAIKU_GEN_MODEL)
    ap.add_argument("--report", action="store_true", help="diagnostics-only mode")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny-model-dir", default="")
    args = ap.parse_args()
    if args.smoke and not args.tiny_model_dir:
        ap.error("--smoke requires --tiny-model-dir")
    if args.report:
        run_report(args)
    else:
        run_rollout(args)


if __name__ == "__main__":
    main()
