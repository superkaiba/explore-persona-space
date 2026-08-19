"""issue #2378 generation driver — pools, banks, story/chat/plain/user-turn generation.

Phases (plan v6 §4.1/§4.2/§4.2b/§4.6; ``--phase`` over the ``PHASES`` registry):

VM phases (repo venv, no model):
- ``build_banks``   P0: LLM-author the scene seed banks (settings/situations/
                    registers/final seeds) + write committed copies of the
                    static prime/opener/char-intro banks + judge rubrics.
- ``build_pools``   P0: draw chat 12k / plain 10k / user 10k mutually disjoint
                    conversations from the pinned #1738 manifest, with the
                    #2054 question filter + the new English-script filter;
                    counts asserted fail-loud; pools uploaded to HF.

Pod phases (model venv: transformers with qwen3_5 + vLLM >= 0.17):
- ``sega``          P2: 3-shot-primed scene+utterance generation (raw vLLM,
                    temp 1.0/top_p .95/top_k 20, cap 512) + structural miner;
                    ALL attempts (mining rejects included) persisted.
- ``chat_plain``    P2: chat-template + plain-text answer cells (cap 2048;
                    cap-hit > 2%/cell => regen those rows at 2x cap).
- ``user_sim``      P2: simulated user-turn arm — raw continuation of the
                    rendered prefill through assistant_1 + user header
                    (stop <|im_end|>/<|im_start|>, cap 1024); degenerate drops
                    counted, never backfilled.
- ``user_fresh``    P2: 4 fresh sim-user draws (seeds 138-141) for 1,000
                    selected conversations.
- ``user_real_render`` P4a: deterministic teacher-forced render of the REAL
                    user_2 rows (no generation); char spans by construction,
                    cross-checked against the full template render.
- ``segb``          P4a: attributed-reply continuation for admission-kept
                    story rows (cap 1024; closing-quote mining; cap-hit
                    > 2%/cell => regen at 2x).
- ``fresh_draws``   P4a: 4 extra SegB / answer draws (seeds 138-141) for
                    1,000 held-out rows per generation cell.
- ``capture_ready`` P4a: reduce stage ledgers into per-cell capture_ready.json
                    (kept ids, floor verdicts, user-arm pair intersection).
- ``upload_stage``  fail-loud HF upload of one raw_completions stage dir
                    (used by the dispatcher after multi-shard fan-outs).

Sharding: ``--shard-index/--num-shards`` shard rows/attempts within the
selected ``--cells`` (the dispatcher fans one process per GPU with
``CUDA_VISIBLE_DEVICES`` pinned in the launcher env). Smoke = the same
entrypoint at small counts (PASS_UNIFIED; no smoke-only branches exist).

Usage (production shapes; see plan §10 workload commands):
  uv run python scripts/issue2378_gen.py --phase build_pools
  uv run python scripts/issue2378_gen.py --phase sega --sega-attempts-per-cell 26000 \\
      --shard-index 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import issue2378_banks as bnk
import issue2378_common as cm

# Pre-LLM() code touches transformers (tokenizer loads), so the default fork()
# EngineCore inherits poisoned state and dies silently 1-4 s after init
# (gotchas.md, #628). Must be set BEFORE any `import vllm` (read at import).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_ENGINE_USED = False

# ---------------------------------------------------------------------------
# Bank load + prompt construction
# ---------------------------------------------------------------------------


def _load_banks(banks_dir: Path) -> dict:
    """Load the committed P0 banks; fail loud on any missing/short bank."""
    names = [
        "settings",
        "situations",
        "registers",
        "final_seeds_question",
        "final_seeds_remark",
        "prime_bank_question",
        "prime_bank_dialogue",
        "openers",
        "char_intros",
    ]
    out: dict[str, list] = {}
    for name in names:
        path = banks_dir / f"{name}.json"
        if not path.exists():
            raise RuntimeError(f"missing bank {path} — run --phase build_banks first")
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
        if not items:
            raise RuntimeError(f"empty bank {path}")
        out[name] = items
    return out


def _fill(template: str, **slots: str) -> str:
    """Sequential ``{slot}`` replacement (str.format is unusable: rubric/bank
    text carries literal JSON braces)."""
    for k, v in slots.items():
        template = template.replace("{" + k + "}", v)
    return template


def _build_scene_prompt(cell: str, wave: int, attempt_idx: int, banksd: dict) -> dict:
    """Deterministically sample bank axes for one SegA attempt and build the
    prompt. Returns the prompt, the scene seed text, and every sampled bank id
    (the §4.2 fold-key columns)."""
    character = cm.CELL_CHARACTER[cell]
    family = cm.CELL_FAMILY[cell]
    rng = random.Random(cm.derived_seed(cm.SEED, "sega", cell, wave, attempt_idx))
    setting_id = rng.randrange(len(banksd["settings"]))
    situation_id = rng.randrange(len(banksd["situations"]))
    register_id = rng.randrange(len(banksd["registers"]))
    seeds_key = "final_seeds_question" if family == "question" else "final_seeds_remark"
    final_seed_id = rng.randrange(len(banksd[seeds_key]))
    char_intro_id = rng.randrange(len(banksd["char_intros"]))
    opener_id = rng.randrange(len(banksd["openers"]))
    primes_key = "prime_bank_question" if family == "question" else "prime_bank_dialogue"
    prime_ids = sorted(rng.sample(range(len(banksd[primes_key])), 3))

    register = banksd["registers"][register_id]
    intro = _fill(
        banksd["char_intros"][char_intro_id], name=character, persona=cm.PERSONAS[character]
    )
    final_seed = _fill(banksd[seeds_key][final_seed_id], name=character)
    scene_seed = " ".join(
        [
            register["opening"],
            banksd["settings"][setting_id],
            banksd["situations"][situation_id],
            intro,
            final_seed,
        ]
    )
    primes = [banksd[primes_key][i] for i in prime_ids]
    prompt = "\n\n***\n\n".join([*primes, scene_seed])
    return {
        "prompt": prompt,
        "scene_seed": scene_seed,
        "character": character,
        "family": family,
        "ids": {
            "setting_id": setting_id,
            "situation_id": situation_id,
            "register_id": register_id,
            "final_seed_id": final_seed_id,
            "prime_exemplar_ids": prime_ids,
            "char_intro_id": char_intro_id,
            "opener_id": opener_id,
        },
    }


# ---------------------------------------------------------------------------
# Structural miner (plan §4.2): first quoted utterance directed at the
# character within the first ~250 generated tokens. Char offsets.
# ---------------------------------------------------------------------------

_QUOTE_OPEN = {'"', "“"}
_QUOTE_CLOSE = {'"': {'"'}, "“": {"”"}}

_ATTRIB_VERBS = (
    "said|asked|replied|answered|murmured|whispered|called|demanded|pressed|urged|added|"
    "ventured|offered|managed|blurted|insisted|wondered|breathed|snapped|muttered|responded|"
    "told|shouted|began|continued"
)


def _find_quote_spans(window: str) -> tuple[list[tuple[int, int, int, int]], bool]:
    """Return closed quote spans as (open_idx, content_start, content_end,
    close_end) plus a flag for a trailing unclosed quote."""
    spans: list[tuple[int, int, int, int]] = []
    i = 0
    unclosed = False
    n = len(window)
    while i < n:
        ch = window[i]
        if ch in _QUOTE_OPEN:
            closers = _QUOTE_CLOSE[ch]
            j = i + 1
            while j < n and window[j] not in closers:
                j += 1
            if j >= n:
                unclosed = True
                break
            spans.append((i, i + 1, j, j + 1))
            i = j + 1
        else:
            i += 1
    return spans, unclosed


def _is_directed(window: str, span: tuple[int, int, int, int], character: str) -> bool:
    """Attribution heuristics: the utterance is addressed TO ``character``
    (never spoken BY them). Pilot-precision-gated at G1 (plan §4.2)."""
    open_idx, cs, ce, close_end = span
    utter = window[cs:ce]
    near = window[max(0, open_idx - 120) : open_idx] + " " + window[close_end : close_end + 120]
    name = re.escape(character)
    # Character as SPEAKER adjacent to the quote => not directed at them.
    if re.search(rf"\b{name}\s+(?:{_ATTRIB_VERBS})\b", near):
        return False
    if re.search(rf"\b(?:{_ATTRIB_VERBS})(?:\s+\w+){{0,2}}?\s+(?:to\s+)?{name}\b", near):
        return True
    if re.search(rf"\b(?:to|toward|towards|at)\s+{name}\b", near):
        return True
    if re.search(rf"\b{name}\b", utter):
        return True
    wide = window[max(0, open_idx - 300) : min(len(window), close_end + 150)]
    if re.search(r"\b(?:you|your)\b", utter, flags=re.IGNORECASE) and re.search(
        rf"\b{name}\b", wide
    ):
        return True
    return False


def _mine_sega(gen_text: str, character: str, family: str, window_char: int) -> dict:
    """Mine the first character-directed quoted utterance of the right kind.

    Returns {"kept": bool, "reason": ..., spans...}; every reject carries a
    named reason (persisted with the attempt row).
    """
    window = gen_text[:window_char]
    spans, unclosed = _find_quote_spans(window)
    if not spans:
        return {"kept": False, "reason": "quote_unclosed" if unclosed else "no_quote_in_window"}
    last_reason = "not_directed"
    for span in spans:
        _open_idx, cs, ce, close_end = span
        utter = window[cs:ce].strip()
        if len(utter) < 3:
            last_reason = "degenerate_utterance"
            continue
        if not _is_directed(window, span, character):
            last_reason = "not_directed"
            continue
        if family == "question":
            if not utter.endswith("?"):
                last_reason = "wrong_kind"
                continue
        elif "?" in utter:
            last_reason = "wrong_kind"
            continue
        return {
            "kept": True,
            "reason": None,
            "utter_start": cs,
            "utter_end": ce,
            "quote_close_end": close_end,
        }
    return {"kept": False, "reason": last_reason}


# ---------------------------------------------------------------------------
# Model helpers (deferred imports — pod model venv only)
# ---------------------------------------------------------------------------


def _get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cm.MODEL_ID)


def _assert_chat_template(tok) -> str:
    """Assert the pinned enable_thinking=False contract (plan §12.2): the
    render carries an EMPTY <think>\\n\\n</think> block. Returns a template
    fingerprint for drift detection at capture."""
    import hashlib

    probe = tok.apply_chat_template(
        [{"role": "user", "content": "probe"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "<think>\n\n</think>" not in probe:
        raise RuntimeError(
            "chat template contract violated: enable_thinking=False render lacks the "
            f"empty <think> block (render digest {cm.text_digest(probe)})"
        )
    return hashlib.sha256(probe.encode("utf-8")).hexdigest()[:16]


def _render_chat(tok, question: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _render_user_prefix(tok, u1: str, a1: str) -> str:
    """Rendered prefill through assistant_1's <|im_end|> + the user header
    (plan §4.2b slot definition)."""
    body = tok.apply_chat_template(
        [{"role": "user", "content": u1}, {"role": "assistant", "content": a1}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return body + cm.USER_TURN_HEADER


def _n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _build_engine(args):
    global _ENGINE_USED
    from explore_persona_space.eval.generation import create_vllm_engine

    kwargs: dict = {"language_model_only": True}
    if args.tp > 1:
        kwargs["tensor_parallel_size"] = args.tp
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    _ENGINE_USED = True
    return create_vllm_engine(
        cm.MODEL_ID,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        seed=cm.SEED,
        dtype="bfloat16",
        **kwargs,
    )


def _reap_engine(llm) -> None:
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)


def _sampling_params(max_tokens: int, stop: list[str] | None, seed: int):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=cm.TEMPERATURE,
        top_p=cm.TOP_P,
        top_k=cm.TOP_K,
        seed=seed,
        max_tokens=max_tokens,
        stop=stop,
    )


def _chunked_generate(llm, prompts: list[str], sps: list, tag: str) -> list:
    """Order-preserving chunked LLM.generate (deadlock prevention, gotchas.md);
    per-chunk INFO line keeps the poller's stall conjunction healthy."""
    chunk = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    outs: list = []
    n = len(prompts)
    n_chunks = (n + chunk - 1) // chunk
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        print(f"[vllm-chunk] {tag} chunk {i // chunk + 1}/{n_chunks} ({j - i} prompts)", flush=True)
        outs.extend(llm.generate(prompts[i:j], sps[i:j], use_tqdm=False))
    return outs


def _gen_text(out) -> tuple[str, str]:
    o = out.outputs[0]
    return o.text, o.finish_reason


def _write_chunk_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _shard_rows(rows: list, args) -> list:
    if args.num_shards <= 1:
        return rows
    return [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_index]


def _maybe_upload(args, stage: str) -> None:
    if args.skip_upload:
        print(f"[upload] SKIPPED (--skip-upload): {stage}", flush=True)
        return
    if args.num_shards > 1:
        print(
            f"[upload] deferred: multi-shard run — run --phase upload_stage --stage {stage} "
            "after all shards complete",
            flush=True,
        )
        return
    cm.upload_stage_dir(Path(args.raw_root) / stage, f"{cm.HF_PREFIX}/raw_completions/{stage}")


# ---------------------------------------------------------------------------
# P0 phases (VM)
# ---------------------------------------------------------------------------


def phase_build_banks(args) -> None:
    """LLM-author the sampled scene axes; write committed bank + rubric files."""
    import issue1310_common as c1310

    for name, text in c1310.PERSONAS.items():
        if cm.PERSONAS.get(name) != text:
            raise RuntimeError(f"PERSONAS parity violation vs issue1310_common: {name}")

    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    def build_request(item) -> dict:
        return {
            "model": cm.JUDGE_MODEL,
            "max_tokens": 8000,
            "temperature": 1.0,
            "system": bnk._BANK_SYSTEM,
            "messages": [{"role": "user", "content": item.payload["prompt"]}],
        }

    from explore_persona_space.eval.utils import parse_judge_json

    items = [
        DispatchItem(item_id=f"bank|{name}", payload={"name": name, "prompt": spec["prompt"]})
        for name, spec in bnk.BANK_BUILDER_SPECS.items()
    ]
    results = asyncio.run(
        dispatch_calls(
            items,
            model=cm.JUDGE_MODEL,
            build_request=build_request,
            parse_response=parse_judge_json,
            force_path="sync",
            cache_dir=Path(args.cache_dir) / "banks",
        )
    )
    banks_dir = Path(args.banks_dir)
    banks_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in bnk.BANK_BUILDER_SPECS.items():
        res = results[f"bank|{name}"]
        if res.error:
            raise RuntimeError(f"bank builder call failed for {name}: {res.reason}")
        items_list = res.result
        if not isinstance(items_list, list) or len(items_list) != spec["n"]:
            got = len(items_list) if isinstance(items_list, list) else type(items_list).__name__
            raise RuntimeError(f"bank {name}: expected {spec['n']} items, got {got}")
        for it in items_list:
            if spec["item_type"] == "str" and not isinstance(it, str):
                raise RuntimeError(f"bank {name}: non-string item")
            if spec["item_type"] == "dict" and not (
                isinstance(it, dict) and "name" in it and "opening" in it
            ):
                raise RuntimeError(f"bank {name}: malformed register object")
            if spec["requires_name_slot"] and (not isinstance(it, str) or it.count("{name}") != 1):
                raise RuntimeError(f"bank {name}: item missing exactly one {{name}} slot")
        cm.atomic_write_json(
            banks_dir / f"{name}.json", {"items": items_list, "metadata": cm.run_metadata()}
        )
        print(f"[build_banks] wrote {name}: {len(items_list)} items", flush=True)

    static = {
        "prime_bank_question": list(bnk.PRIME_BANK_QUESTION),
        "prime_bank_dialogue": list(bnk.PRIME_BANK_DIALOGUE),
        "openers": list(bnk.OPENER_BANK),
        "char_intros": list(bnk.CHAR_INTRO_TEMPLATES),
    }
    for name, items_list in static.items():
        cm.atomic_write_json(
            banks_dir / f"{name}.json", {"items": items_list, "metadata": cm.run_metadata()}
        )
    cm.atomic_write_json(
        banks_dir / "admission_rubric.json",
        {
            "admission_system": bnk.ADMISSION_SYSTEM,
            "rubric_question": bnk.ADMISSION_RUBRIC_QUESTION,
            "rubric_dialogue": bnk.ADMISSION_RUBRIC_DIALOGUE,
            "congruence_system": bnk.CONGRUENCE_SYSTEM,
            "congruence_rubric": bnk.CONGRUENCE_RUBRIC,
            "judge_model": cm.JUDGE_MODEL,
            "max_tokens": cm.JUDGE_MAX_TOKENS,
            "metadata": cm.run_metadata(),
        },
    )
    print(f"[build_banks] done — banks under {banks_dir}", flush=True)


def phase_build_pools(args) -> None:
    """Draw the three mutually disjoint conversation pools from the pinned
    #1738 manifest (plan §4.1). Fail-loud count asserts BEFORE any generation."""
    import numpy as np

    dest_root = cm.REPO_ROOT / "data" / "issue_2378" / "hf_dl"
    leaf = cm.stage_hf_prefix(cm.MANIFEST_PREFIX, dest_root, revision=cm.MANIFEST_REVISION)
    parts = sorted(leaf.glob("part_*.jsonl"))
    if not parts:
        raise RuntimeError(f"no part_*.jsonl under staged manifest {leaf}")

    rejects: dict[str, int] = {}

    def rej(reason: str) -> None:
        rejects[reason] = rejects.get(reason, 0) + 1

    # Pass 1: eligibility flags per row (content NOT retained; digest-only).
    index: list[tuple[str, int, bool, bool]] = []  # (part_name, line_idx, chat_ok, user_ok)
    seen_conv: set[str] = set()
    seen_q: set[str] = set()
    for part in parts:
        for line_idx, row in enumerate(cm.iter_jsonl(part)):
            conv_id = "mt_" + row["source_hash"][:12]
            if conv_id in seen_conv:
                rej("dup_conv_id")
                continue
            seen_conv.add(conv_id)
            msgs = row["messages"]
            chat_ok = False
            if msgs and msgs[0].get("role") == "user":
                q = msgs[0].get("content", "")
                if "\n" in q:
                    rej("q_multiline")
                elif not (cm.QUESTION_MIN_CHARS <= len(q.strip()) <= cm.QUESTION_MAX_CHARS):
                    rej("q_len_band")
                elif not cm.english_majority(q):
                    rej("q_non_english")
                else:
                    import hashlib

                    qh = hashlib.sha256(q.strip().encode("utf-8")).hexdigest()
                    if qh in seen_q:
                        rej("dup_question")
                    else:
                        seen_q.add(qh)
                        chat_ok = True
            else:
                rej("first_msg_not_user")
            user_ok = False
            if row.get("depth", 0) >= 2 and len(msgs) >= 3:
                roles_ok = all(
                    m.get("role") == ("user" if i % 2 == 0 else "assistant")
                    for i, m in enumerate(msgs)
                )
                if not roles_ok:
                    rej("u_role_alternation")
                else:
                    u1, a1, u2 = (msgs[0]["content"], msgs[1]["content"], msgs[2]["content"])
                    if not (cm.USER_TURN_MIN_CHARS <= len(u2.strip()) <= cm.USER_TURN_MAX_CHARS):
                        rej("u2_len_band")
                    elif not (
                        cm.english_majority(u1)
                        and cm.english_majority(a1)
                        and cm.english_majority(u2)
                    ):
                        rej("u_non_english")
                    else:
                        user_ok = True
            else:
                rej("u_depth_lt_2")
            index.append((part.name, line_idx, chat_ok, user_ok))
    n_chat_eligible = sum(1 for r in index if r[2])
    n_user_eligible = sum(1 for r in index if r[3])
    print(
        f"[build_pools] scanned {len(index)} rows; chat-eligible {n_chat_eligible}; "
        f"user-eligible {n_user_eligible}; rejects {json.dumps(rejects)}",
        flush=True,
    )

    order = np.random.default_rng(cm.SEED).permutation(len(index))
    chat_idx: list[int] = []
    plain_idx: list[int] = []
    user_idx: list[int] = []
    for oi in order:
        _part, _li, chat_ok, user_ok = index[int(oi)]
        if chat_ok and len(chat_idx) < cm.CHAT_DRAW_N:
            chat_idx.append(int(oi))
        elif chat_ok and len(plain_idx) < cm.PLAIN_DRAW_N:
            plain_idx.append(int(oi))
        elif user_ok and len(user_idx) < cm.USER_DRAW_N:
            user_idx.append(int(oi))
    if (
        len(chat_idx) < cm.CHAT_DRAW_N
        or len(plain_idx) < cm.PLAIN_DRAW_N
        or len(user_idx) < cm.USER_DRAW_N
    ):
        raise RuntimeError(
            "P0 pool yield below targets (plan §12.6 fail-loud): "
            f"chat {len(chat_idx)}/{cm.CHAT_DRAW_N}, plain {len(plain_idx)}/{cm.PLAIN_DRAW_N}, "
            f"user {len(user_idx)}/{cm.USER_DRAW_N}; rejects {json.dumps(rejects)}"
        )

    # Pass 2: extract drawn rows.
    want: dict[tuple[str, int], str] = {}
    for pool, idxs in (("chat", chat_idx), ("plain", plain_idx), ("user", user_idx)):
        for oi in idxs:
            part_name, line_idx, _c, _u = index[oi]
            want[(part_name, line_idx)] = pool
    pools: dict[str, list[dict]] = {"chat": [], "plain": [], "user": []}
    for part in parts:
        for line_idx, row in enumerate(cm.iter_jsonl(part)):
            pool = want.get((part.name, line_idx))
            if pool is None:
                continue
            conv_id = "mt_" + row["source_hash"][:12]
            msgs = row["messages"]
            if pool in ("chat", "plain"):
                pools[pool].append({"conv_id": conv_id, "question": msgs[0]["content"].strip()})
            else:
                assert row["depth"] >= 2, f"drawn user row below depth 2: {conv_id}"
                assert all(
                    m.get("role") == ("user" if i % 2 == 0 else "assistant")
                    for i, m in enumerate(msgs)
                ), f"drawn user row role alternation violated: {conv_id}"
                pools[pool].append(
                    {
                        "conv_id": conv_id,
                        "u1": msgs[0]["content"],
                        "a1": msgs[1]["content"],
                        "u2": msgs[2]["content"],
                        "depth": row["depth"],
                    }
                )
    pools_dir = Path(args.pools_dir)
    pools_dir.mkdir(parents=True, exist_ok=True)
    for pool, fname in (("chat", "chat_draw"), ("plain", "plain_draw"), ("user", "user_draw")):
        rows = pools[pool]
        path = pools_dir / f"{fname}.jsonl"
        tmp = pools_dir / f".{fname}.tmp.{os.getpid()}"
        with tmp.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        print(f"[build_pools] wrote {path.name}: {len(rows)} rows", flush=True)
    digest = {
        "n_scanned": len(index),
        "n_chat_eligible": n_chat_eligible,
        "n_user_eligible": n_user_eligible,
        "rejects": rejects,
        "draws": {k: len(v) for k, v in pools.items()},
        "manifest_prefix": cm.MANIFEST_PREFIX,
        "manifest_revision": cm.MANIFEST_REVISION,
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(pools_dir / "pool_digest.json", digest)
    if not args.skip_upload:
        cm.upload_stage_dir(pools_dir, f"{cm.HF_PREFIX}/pools")


# ---------------------------------------------------------------------------
# Pod generation phases
# ---------------------------------------------------------------------------


def _resolve_pools_dir(args) -> Path:
    if args.stage_pools_from_hf:
        return cm.stage_hf_prefix(
            f"{cm.HF_PREFIX}/pools", cm.REPO_ROOT / "data" / "issue_2378" / "hf_stage"
        )
    return Path(args.pools_dir)


def _load_pool(pools_dir: Path, name: str) -> list[dict]:
    path = pools_dir / f"{name}.jsonl"
    rows = list(cm.iter_jsonl(path))
    if not rows:
        raise RuntimeError(f"empty pool {path}")
    return rows


def phase_sega(args) -> None:
    """SegA scene+utterance generation + structural mining (plan §4.2)."""
    if args.sega_attempts_per_cell <= 0:
        raise SystemExit("--sega-attempts-per-cell is required for --phase sega")
    banksd = _load_banks(Path(args.banks_dir))
    tok = _get_tokenizer()
    llm = _build_engine(args)
    cells = [c for c in args.cells.split(",") if c] or list(cm.STORY_CELLS)
    for c in cells:
        if c not in cm.STORY_CELLS:
            raise SystemExit(f"unknown story cell {c}")
    raw_root = Path(args.raw_root)
    wave = args.wave
    t0 = time.time()
    for cell in cells:
        attempt_ids = _shard_rows(list(range(args.sega_attempts_per_cell)), args)
        regime = {
            "phase": "sega",
            "cell": cell,
            "wave": wave,
            "n": args.sega_attempts_per_cell,
            "shard": [args.shard_index, args.num_shards],
            "seed": cm.SEED,
            "cap": cm.SEGA_MAX_TOKENS,
            "model": cm.MODEL_ID,
        }
        ledger = cm.StageLedger(
            raw_root / "sega" / f"ledger_{cell}_w{wave}_s{args.shard_index}.json", regime
        )
        counts = {"attempts": 0, "kept": 0, "cap_hit": 0}
        mine_rejects: dict[str, int] = {}
        n_chunks = (len(attempt_ids) + args.chunk_rows - 1) // args.chunk_rows
        for ci in range(n_chunks):
            key = f"{cell}|w{wave}|s{args.shard_index}|c{ci:04d}"
            chunk = attempt_ids[ci * args.chunk_rows : (ci + 1) * args.chunk_rows]
            if ledger.is_done(key):
                continue
            built = [_build_scene_prompt(cell, wave, a, banksd) for a in chunk]
            sps = [
                _sampling_params(
                    cm.SEGA_MAX_TOKENS, None, cm.derived_seed(cm.SEED, "sega", cell, wave, a)
                )
                for a in chunk
            ]
            outs = _chunked_generate(llm, [b["prompt"] for b in built], sps, f"sega/{cell}")
            raw_rows, mined_rows = [], []
            for a, b, out in zip(chunk, built, outs):
                text, finish = _gen_text(out)
                enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
                offs = enc["offset_mapping"]
                window_char = (
                    offs[cm.MINER_WINDOW_TOKENS][0]
                    if len(offs) > cm.MINER_WINDOW_TOKENS
                    else len(text)
                )
                verdict = _mine_sega(text, b["character"], b["family"], window_char)
                counts["attempts"] += 1
                if finish == "length":
                    counts["cap_hit"] += 1
                row_id = f"{cell}_w{wave}_a{a:06d}"
                raw_rows.append(
                    {
                        "row_id": row_id,
                        "cell": cell,
                        "wave": wave,
                        "attempt_idx": a,
                        **b["ids"],
                        "scene_seed": b["scene_seed"],
                        "gen_text": text,
                        "finish_reason": finish,
                        "mined": verdict,
                    }
                )
                if verdict["kept"]:
                    counts["kept"] += 1
                    close = verdict["quote_close_end"]
                    mined_rows.append(
                        {
                            "row_id": row_id,
                            "cell": cell,
                            "wave": wave,
                            "attempt_idx": a,
                            "character": b["character"],
                            "family": b["family"],
                            **b["ids"],
                            "scene_seed": b["scene_seed"],
                            "scene_pre_answer": b["scene_seed"] + text[:close],
                            "utterance": text[verdict["utter_start"] : verdict["utter_end"]],
                            "utter_span": [verdict["utter_start"], verdict["utter_end"]],
                            "quote_close_end": close,
                        }
                    )
                else:
                    mine_rejects[verdict["reason"]] = mine_rejects.get(verdict["reason"], 0) + 1
            stem = f"{cell}_w{wave}_s{args.shard_index}_c{ci:04d}"
            _write_chunk_jsonl(raw_root / "sega" / f"{stem}.jsonl", raw_rows)
            if mined_rows:
                _write_chunk_jsonl(raw_root / "sega_mined" / f"{stem}.jsonl", mined_rows)
            ledger.mark_done(key)
            cm.progress("sega", ci + 1, n_chunks, key, t0)
        summary = {
            "regime": regime,
            "counts": counts,
            "cap_hit_fraction": counts["cap_hit"] / max(1, counts["attempts"]),
            "cap_hit_regen": "exempt (SegA — miner window binds by design, plan §4.2)",
            "mine_rejects": mine_rejects,
            "metadata": cm.run_metadata(),
        }
        cm.atomic_write_json(
            raw_root / "sega" / f"summary_{cell}_w{wave}_s{args.shard_index}.json", summary
        )
        print(f"[sega] {cell}: {json.dumps(counts)} rejects={json.dumps(mine_rejects)}", flush=True)
    _reap_engine(llm)
    _maybe_upload(args, "sega")
    _maybe_upload(args, "sega_mined")


def _regen_cap_hit(
    llm,
    prompts: list[str],
    texts: list[str],
    finishes: list[str],
    seeds: list[int],
    cap: int,
    stop: list[str] | None,
    tag: str,
) -> tuple[list[str], list[str], list[bool], float, float]:
    """Apply the > 2%/cell cap-hit rule: regen cap-hit rows at 2x cap.

    Returns (texts, finishes, regen_flags, frac_before, frac_after).
    """
    hit = [i for i, f in enumerate(finishes) if f == "length"]
    frac_before = len(hit) / max(1, len(texts))
    regen = [False] * len(texts)
    if frac_before <= cm.CAP_HIT_REGEN_THRESHOLD or not hit:
        return texts, finishes, regen, frac_before, frac_before
    sps = [_sampling_params(2 * cap, stop, cm.derived_seed(seeds[i], "regen")) for i in hit]
    outs = _chunked_generate(llm, [prompts[i] for i in hit], sps, f"{tag}/regen2x")
    for k, i in enumerate(hit):
        text, finish = _gen_text(outs[k])
        texts[i], finishes[i], regen[i] = text, finish, True
    frac_after = sum(1 for f in finishes if f == "length") / max(1, len(texts))
    print(
        f"[{tag}] cap-hit regen at 2x: {len(hit)} rows; frac {frac_before:.4f} -> {frac_after:.4f}",
        flush=True,
    )
    return texts, finishes, regen, frac_before, frac_after


def _run_answer_cell(args, llm, tok, cell: str, rows: list[dict], template_sha: str) -> None:
    """Shared chat / plain_text answer generation (plan §4.2 chat/plain cells)."""
    raw_root = Path(args.raw_root)
    stage = "chat" if cell == "chat" else "plain"
    cap = cm.CHAT_MAX_TOKENS if cell == "chat" else cm.PLAIN_MAX_TOKENS
    stop = cm.CHAT_STOP if cell == "chat" else cm.PLAIN_STOP
    budget = args.max_model_len - 2 * cap  # leave room for the 2x regen pass
    wave = args.wave
    regime = {
        "phase": stage,
        "cell": cell,
        "wave": wave,
        "n": len(rows),
        "shard": [args.shard_index, args.num_shards],
        "seed": cm.SEED,
        "cap": cap,
        "model": cm.MODEL_ID,
    }
    ledger = cm.StageLedger(
        raw_root / stage / f"ledger_{cell}_w{wave}_s{args.shard_index}.json", regime
    )
    counts = {"rows": 0, "kept": 0, "over_length": 0, "non_english_answer": 0, "empty_answer": 0}
    frac_b_all, frac_a_all = [], []
    my_rows = _shard_rows(rows, args)
    n_chunks = (len(my_rows) + args.chunk_rows - 1) // args.chunk_rows
    t0 = time.time()
    for ci in range(n_chunks):
        key = f"{cell}|w{wave}|s{args.shard_index}|c{ci:04d}"
        chunk = my_rows[ci * args.chunk_rows : (ci + 1) * args.chunk_rows]
        if ledger.is_done(key):
            continue
        prompts, seeds, kept_rows, dropped_rows = [], [], [], []
        for r in chunk:
            if cell == "chat":
                prompt = _render_chat(tok, r["question"])
            else:
                prompt = f"User: {r['question']}\n\nAssistant:"
            if _n_tokens(tok, prompt) > budget:
                counts["over_length"] += 1
                dropped_rows.append(
                    {
                        "cell": cell,
                        "conv_id": r["conv_id"],
                        "keep": False,
                        "drop_reason": "over_length",
                    }
                )
                continue
            prompts.append(prompt)
            seeds.append(cm.derived_seed(cm.SEED, stage, wave, r["conv_id"]))
            kept_rows.append(r)
        sps = [_sampling_params(cap, stop, s) for s in seeds]
        outs = _chunked_generate(llm, prompts, sps, f"{stage}")
        texts, finishes = [], []
        for out in outs:
            text, finish = _gen_text(out)
            texts.append(text)
            finishes.append(finish)
        texts, finishes, regen, fb, fa = _regen_cap_hit(
            llm, prompts, texts, finishes, seeds, cap, stop, stage
        )
        frac_b_all.append(fb)
        frac_a_all.append(fa)
        out_rows = list(dropped_rows)
        for r, text, finish, rg, s in zip(kept_rows, texts, finishes, regen, seeds):
            answer = text.strip()
            counts["rows"] += 1
            keep, drop_reason = True, None
            if not answer:
                keep, drop_reason = False, "empty_answer"
                counts["empty_answer"] += 1
            elif finish == "length":
                keep, drop_reason = True, None  # cap-hit rows stay; fraction reported
            if keep and not cm.english_majority(answer):
                keep, drop_reason = False, "non_english_answer"
                counts["non_english_answer"] += 1
            if keep:
                counts["kept"] += 1
            out_rows.append(
                {
                    "cell": cell,
                    "conv_id": r["conv_id"],
                    "question": r["question"],
                    "answer": answer,
                    "finish_reason": finish,
                    "seed": s,
                    "regen": rg,
                    "keep": keep,
                    "drop_reason": drop_reason,
                    "template_sha": template_sha if cell == "chat" else None,
                }
            )
        stem = f"{cell}_w{wave}_s{args.shard_index}_c{ci:04d}"
        _write_chunk_jsonl(raw_root / stage / f"{stem}.jsonl", out_rows)
        ledger.mark_done(key)
        cm.progress(stage, ci + 1, n_chunks, key, t0)
    summary = {
        "regime": regime,
        "counts": counts,
        "cap_hit_fraction_before": max(frac_b_all) if frac_b_all else 0.0,
        "cap_hit_fraction_after": max(frac_a_all) if frac_a_all else 0.0,
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(
        raw_root / stage / f"summary_{cell}_w{wave}_s{args.shard_index}.json", summary
    )
    print(f"[{stage}] {cell}: {json.dumps(counts)}", flush=True)


def phase_chat_plain(args) -> None:
    pools_dir = _resolve_pools_dir(args)
    tok = _get_tokenizer()
    template_sha = _assert_chat_template(tok)
    llm = _build_engine(args)
    chat_rows = _load_pool(pools_dir, "chat_draw")[: args.chat_rows or None]
    plain_rows = _load_pool(pools_dir, "plain_draw")[: args.plain_rows or None]
    _run_answer_cell(args, llm, tok, "chat", chat_rows, template_sha)
    _run_answer_cell(args, llm, tok, "plain_text", plain_rows, template_sha)
    _reap_engine(llm)
    _maybe_upload(args, "chat")
    _maybe_upload(args, "plain")


def _classify_sim_turn(text: str, finish: str) -> tuple[bool, str | None]:
    turn = text.strip()
    if not turn:
        return False, "empty_turn"
    if "<think>" in turn:
        return False, "think_leak"
    if not (cm.USER_TURN_MIN_CHARS <= len(turn) <= cm.USER_TURN_MAX_CHARS):
        return False, "len_band"
    return True, None


def _run_user_sim(args, llm, tok, rows: list[dict], stage: str, draw_seeds: list[int]) -> None:
    """Sim-user turn generation (plan §4.2b): raw continuation of the rendered
    prefill; one pass per draw seed (production: [SEED]; fresh: 138-141)."""
    raw_root = Path(args.raw_root)
    cap = cm.USER_SIM_MAX_TOKENS
    budget = args.max_model_len - 2 * cap
    wave = args.wave
    regime = {
        "phase": stage,
        "wave": wave,
        "n": len(rows),
        "draw_seeds": draw_seeds,
        "shard": [args.shard_index, args.num_shards],
        "cap": cap,
        "model": cm.MODEL_ID,
    }
    ledger = cm.StageLedger(raw_root / stage / f"ledger_w{wave}_s{args.shard_index}.json", regime)
    counts: dict[str, int] = {
        "rows": 0,
        "kept": 0,
        "over_length": 0,
        "empty_turn": 0,
        "think_leak": 0,
        "len_band": 0,
    }
    frac_b_all, frac_a_all = [], []
    my_rows = _shard_rows(rows, args)
    n_chunks = (len(my_rows) + args.chunk_rows - 1) // args.chunk_rows
    t0 = time.time()
    for draw_seed in draw_seeds:
        for ci in range(n_chunks):
            key = f"{stage}|w{wave}|d{draw_seed}|s{args.shard_index}|c{ci:04d}"
            chunk = my_rows[ci * args.chunk_rows : (ci + 1) * args.chunk_rows]
            if ledger.is_done(key):
                continue
            prompts, seeds, kept_rows, dropped_rows = [], [], [], []
            for r in chunk:
                prefix = _render_user_prefix(tok, r["u1"], r["a1"])
                if _n_tokens(tok, prefix) > budget:
                    counts["over_length"] += 1
                    dropped_rows.append(
                        {
                            "cell": "chat_user_sim",
                            "conv_id": r["conv_id"],
                            "draw_seed": draw_seed,
                            "keep": False,
                            "drop_reason": "over_length",
                        }
                    )
                    continue
                prompts.append(prefix)
                seeds.append(cm.derived_seed(draw_seed, "user_sim", wave, r["conv_id"]))
                kept_rows.append(r)
            sps = [_sampling_params(cap, cm.USER_SIM_STOP, s) for s in seeds]
            outs = _chunked_generate(llm, prompts, sps, stage)
            texts, finishes = [], []
            for out in outs:
                text, finish = _gen_text(out)
                texts.append(text)
                finishes.append(finish)
            texts, finishes, regen, fb, fa = _regen_cap_hit(
                llm, prompts, texts, finishes, seeds, cap, cm.USER_SIM_STOP, stage
            )
            frac_b_all.append(fb)
            frac_a_all.append(fa)
            out_rows = list(dropped_rows)
            for r, prefix, text, finish, rg, s in zip(
                kept_rows, prompts, texts, finishes, regen, seeds
            ):
                keep, drop_reason = _classify_sim_turn(text, finish)
                counts["rows"] += 1
                if keep:
                    counts["kept"] += 1
                elif drop_reason in counts:
                    counts[drop_reason] += 1
                out_rows.append(
                    {
                        "cell": "chat_user_sim",
                        "conv_id": r["conv_id"],
                        "draw_seed": draw_seed,
                        "sim_turn": text.strip(),
                        "finish_reason": finish,
                        "seed": s,
                        "regen": rg,
                        "keep": keep,
                        "drop_reason": drop_reason,
                        "prefix_chars": len(prefix),
                        "prefix_digest": cm.text_digest(prefix),
                    }
                )
            stem = f"w{wave}_d{draw_seed}_s{args.shard_index}_c{ci:04d}"
            _write_chunk_jsonl(raw_root / stage / f"{stem}.jsonl", out_rows)
            ledger.mark_done(key)
            cm.progress(stage, ci + 1, n_chunks, key, t0)
    summary = {
        "regime": regime,
        "counts": counts,
        "cap_hit_fraction_before": max(frac_b_all) if frac_b_all else 0.0,
        "cap_hit_fraction_after": max(frac_a_all) if frac_a_all else 0.0,
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(raw_root / stage / f"summary_w{wave}_s{args.shard_index}.json", summary)
    print(f"[{stage}] {json.dumps(counts)}", flush=True)


def phase_user_sim(args) -> None:
    pools_dir = _resolve_pools_dir(args)
    tok = _get_tokenizer()
    _assert_chat_template(tok)
    llm = _build_engine(args)
    rows = _load_pool(pools_dir, "user_draw")[: args.user_sim_rows or None]
    _run_user_sim(args, llm, tok, rows, "user_sim", [cm.SEED])
    _reap_engine(llm)
    _maybe_upload(args, "user_sim")


def phase_user_fresh(args) -> None:
    """Fresh sim-user draws (seeds 138-141) for the selected held-out rows."""
    pools_dir = _resolve_pools_dir(args)
    tok = _get_tokenizer()
    _assert_chat_template(tok)
    llm = _build_engine(args)
    rows = _load_pool(pools_dir, "user_draw")
    order = random.Random(cm.derived_seed(cm.SEED, "user_fresh_select")).sample(
        range(len(rows)), len(rows)
    )
    sel = [rows[i] for i in order[: args.user_fresh_rows]]
    draw_seeds = list(cm.FRESH_SEEDS[: args.user_fresh_draws])
    _run_user_sim(args, llm, tok, sel, "user_sim_fresh", draw_seeds)
    _reap_engine(llm)
    _maybe_upload(args, "user_sim_fresh")


def phase_user_real_render(args) -> None:
    """Real user-turn arm (plan §4.2b): NO generation — deterministic render +
    span mining, cross-checked against the full template render."""
    pools_dir = _resolve_pools_dir(args)
    tok = _get_tokenizer()
    _assert_chat_template(tok)
    rows = _load_pool(pools_dir, "user_draw")
    raw_root = Path(args.raw_root)
    counts = {"rows": 0, "kept": 0, "span_mismatch": 0}
    out_rows = []
    t0 = time.time()
    for k, r in enumerate(rows):
        prefix = _render_user_prefix(tok, r["u1"], r["a1"])
        u2 = r["u2"]
        rendered_full = tok.apply_chat_template(
            [
                {"role": "user", "content": r["u1"]},
                {"role": "assistant", "content": r["a1"]},
                {"role": "user", "content": u2},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        counts["rows"] += 1
        if not rendered_full.startswith(prefix + u2):
            counts["span_mismatch"] += 1
            out_rows.append(
                {
                    "cell": "chat_user_real",
                    "conv_id": r["conv_id"],
                    "keep": False,
                    "drop_reason": "span_mismatch",
                }
            )
        else:
            counts["kept"] += 1
            out_rows.append(
                {
                    "cell": "chat_user_real",
                    "conv_id": r["conv_id"],
                    "rendered_text": rendered_full,
                    "header_end": len(prefix),
                    "u2_span": [len(prefix), len(prefix) + len(u2)],
                    "keep": True,
                    "drop_reason": None,
                }
            )
        if len(out_rows) >= 2000 or k == len(rows) - 1:
            ci = k // 2000
            _write_chunk_jsonl(raw_root / "user_real_render" / f"c{ci:04d}.jsonl", out_rows)
            out_rows = []
            cm.progress("user_real_render", k + 1, len(rows), f"c{ci:04d}", t0)
    summary = {"counts": counts, "metadata": cm.run_metadata()}
    cm.atomic_write_json(raw_root / "user_real_render" / "summary.json", summary)
    print(f"[user_real_render] {json.dumps(counts)}", flush=True)
    _maybe_upload(args, "user_real_render")


def _rows_dir(args, stage: str, explicit: str | None = None) -> Path:
    """Resolve a raw-completions stage dir: local-first, HF-staged fallback
    (cross-pod reads — e.g. pod B consuming pod A's chat/plain/user_sim rows),
    fail-loud otherwise (#779/#1773 lane-input staging)."""
    local = Path(explicit) if explicit else Path(args.raw_root) / stage
    if any(local.glob("*.jsonl")):
        return local
    if args.stage_raw_from_hf:
        return cm.stage_hf_prefix(
            f"{cm.HF_PREFIX}/raw_completions/{stage}",
            cm.REPO_ROOT / "data" / "issue_2378" / "hf_stage",
        )
    raise RuntimeError(
        f"no rows for stage {stage} under {local} (pass --stage-raw-from-hf to fetch, "
        "or run the producing phase first)"
    )


def _load_mined_rows(mined_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(mined_dir.glob("*.jsonl")):
        for row in cm.iter_jsonl(path):
            rows[row["row_id"]] = row
    if not rows:
        raise RuntimeError(f"no mined rows under {mined_dir} (empty selection — fail loud)")
    return rows


def _load_kept_ids(kept_dir: Path, cell: str) -> list[str]:
    path = kept_dir / f"{cell}.json"
    if not path.exists():
        raise RuntimeError(f"missing admission keeps {path} — run the admission judge first")
    kept = json.loads(path.read_text(encoding="utf-8"))["admitted"]
    ids = [k["row_id"] for k in kept]
    if not ids:
        raise RuntimeError(f"empty admitted set for {cell} (fail loud)")
    return ids


def _segb_prompt(mined: dict, banksd: dict) -> tuple[str, str]:
    """Rebuild the SegB continuation prompt deterministically from bank ids +
    the persisted pre-answer text; assert bank/scene coherence."""
    rebuilt = _build_scene_prompt(mined["cell"], mined["wave"], mined["attempt_idx"], banksd)
    if rebuilt["scene_seed"] != mined["scene_seed"]:
        raise RuntimeError(
            f"bank drift: rebuilt scene_seed mismatch for {mined['row_id']} — banks changed "
            "since SegA (fail loud)"
        )
    opener = _fill(banksd["openers"][mined["opener_id"]], name=mined["character"])
    primes_key = "prime_bank_question" if mined["family"] == "question" else "prime_bank_dialogue"
    primes = [banksd[primes_key][i] for i in mined["prime_exemplar_ids"]]
    gen_after_seed = mined["scene_pre_answer"][len(mined["scene_seed"]) :]
    prompt = "\n\n***\n\n".join([*primes, mined["scene_seed"]]) + gen_after_seed + "\n\n" + opener
    return prompt, opener


def _mine_closing_quote(text: str) -> int | None:
    idxs = [i for i in (text.find('"'), text.find("”")) if i >= 0]
    return min(idxs) if idxs else None


def phase_segb(args) -> None:
    """SegB attributed replies for admission-kept story rows (plan §4.2)."""
    banksd = _load_banks(Path(args.banks_dir))
    mined = _load_mined_rows(_rows_dir(args, "sega_mined", args.mined_dir))
    llm = _build_engine(args)
    raw_root = Path(args.raw_root)
    wave = args.wave
    cells = [c for c in args.cells.split(",") if c] or list(cm.STORY_CELLS)
    t0 = time.time()
    for cell in cells:
        kept_ids = _load_kept_ids(Path(args.kept_dir), cell)
        order = random.Random(cm.derived_seed(cm.SEED, "segb_select", cell)).sample(
            range(len(kept_ids)), len(kept_ids)
        )
        sel = [kept_ids[i] for i in order[: args.target_kept_per_cell]]
        sel = _shard_rows(sel, args)
        regime = {
            "phase": "segb",
            "cell": cell,
            "wave": wave,
            "target": args.target_kept_per_cell,
            "shard": [args.shard_index, args.num_shards],
            "cap": cm.SEGB_MAX_TOKENS,
            "model": cm.MODEL_ID,
        }
        ledger = cm.StageLedger(
            raw_root / "segb" / f"ledger_{cell}_w{wave}_s{args.shard_index}.json", regime
        )
        counts = {
            "rows": 0,
            "kept": 0,
            "cap_hit_no_close": 0,
            "empty_answer": 0,
            "non_english_answer_flag": 0,
        }
        n_chunks = (len(sel) + args.chunk_rows - 1) // args.chunk_rows
        for ci in range(n_chunks):
            key = f"{cell}|w{wave}|s{args.shard_index}|c{ci:04d}"
            chunk_ids = sel[ci * args.chunk_rows : (ci + 1) * args.chunk_rows]
            if ledger.is_done(key):
                continue
            prompts, openers, seeds = [], [], []
            for rid in chunk_ids:
                prompt, opener = _segb_prompt(mined[rid], banksd)
                prompts.append(prompt)
                openers.append(opener)
                seeds.append(cm.derived_seed(cm.SEED, "segb", cell, wave, rid))
            sps = [_sampling_params(cm.SEGB_MAX_TOKENS, None, s) for s in seeds]
            outs = _chunked_generate(llm, prompts, sps, f"segb/{cell}")
            texts = []
            finishes = []
            for out in outs:
                text, finish = _gen_text(out)
                texts.append(text)
                finishes.append(finish)
            # Cap-hit for SegB = no closing quote within the cap (plan §4.2).
            no_close = [i for i, t in enumerate(texts) if _mine_closing_quote(t) is None]
            frac = len(no_close) / max(1, len(texts))
            regen = [False] * len(texts)
            if frac > cm.CAP_HIT_REGEN_THRESHOLD and no_close:
                sps2 = [
                    _sampling_params(
                        2 * cm.SEGB_MAX_TOKENS, None, cm.derived_seed(seeds[i], "regen")
                    )
                    for i in no_close
                ]
                outs2 = _chunked_generate(
                    llm, [prompts[i] for i in no_close], sps2, f"segb/{cell}/regen2x"
                )
                for k2, i in enumerate(no_close):
                    text, finish = _gen_text(outs2[k2])
                    texts[i], finishes[i], regen[i] = text, finish, True
                print(
                    f"[segb/{cell}] no-close regen at 2x: {len(no_close)} rows (frac {frac:.4f})",
                    flush=True,
                )
            out_rows = []
            for rid, opener, text, finish, rg, s in zip(
                chunk_ids, openers, texts, finishes, regen, seeds
            ):
                counts["rows"] += 1
                close = _mine_closing_quote(text)
                keep, drop_reason, answer = True, None, None
                if close is None:
                    keep, drop_reason = False, "cap_hit_no_close"
                    counts["cap_hit_no_close"] += 1
                else:
                    answer = text[:close].strip()
                    if not answer:
                        keep, drop_reason = False, "empty_answer"
                        counts["empty_answer"] += 1
                if keep and answer is not None and not cm.english_majority(answer):
                    counts["non_english_answer_flag"] += 1  # report-only for story cells
                if keep:
                    counts["kept"] += 1
                out_rows.append(
                    {
                        "cell": cell,
                        "row_id": rid,
                        "wave": wave,
                        "opener_id": mined[rid]["opener_id"],
                        "opener_text": opener,
                        "gen_text": text,
                        "answer": answer,
                        "answer_close_idx": close,
                        "finish_reason": finish,
                        "seed": s,
                        "regen": rg,
                        "keep": keep,
                        "drop_reason": drop_reason,
                    }
                )
            stem = f"{cell}_w{wave}_s{args.shard_index}_c{ci:04d}"
            _write_chunk_jsonl(raw_root / "segb" / f"{stem}.jsonl", out_rows)
            ledger.mark_done(key)
            cm.progress("segb", ci + 1, n_chunks, key, t0)
        summary = {"regime": regime, "counts": counts, "metadata": cm.run_metadata()}
        cm.atomic_write_json(
            raw_root / "segb" / f"summary_{cell}_w{wave}_s{args.shard_index}.json", summary
        )
        print(f"[segb] {cell}: {json.dumps(counts)}", flush=True)
    _reap_engine(llm)
    _maybe_upload(args, "segb")


def _stage_kept_rows(rows_dir: Path, cell: str | None = None) -> dict[str, dict]:
    """Collect a stage's persisted KEPT rows keyed by row id (conv_id/row_id)."""
    rows: dict[str, dict] = {}
    for path in sorted(rows_dir.glob("*.jsonl")):
        for row in cm.iter_jsonl(path):
            if cell is not None and row.get("cell") != cell:
                continue
            if row.get("keep"):
                rows[row.get("row_id") or row["conv_id"]] = row
    return rows


def phase_fresh_draws(args) -> None:
    """Fresh SegB / answer draws (seeds 138-141) for 1,000 held-out rows per
    generation cell (plan §4.2 fresh draws; user arm handled by user_fresh)."""
    banksd = _load_banks(Path(args.banks_dir))
    raw_root = Path(args.raw_root)
    tok = _get_tokenizer()
    template_sha = _assert_chat_template(tok)
    llm = _build_engine(args)
    draw_seeds = list(cm.FRESH_SEEDS[: args.fresh_draws])
    cells = [c for c in args.cells.split(",") if c] or ["chat", "plain_text", *cm.STORY_CELLS]
    mined = None
    t0 = time.time()
    for cell in cells:
        if cell in cm.STORY_CELLS:
            if mined is None:
                mined = _load_mined_rows(_rows_dir(args, "sega_mined", args.mined_dir))
            base_rows = _stage_kept_rows(_rows_dir(args, "segb"), cell)
        elif cell == "chat":
            base_rows = _stage_kept_rows(_rows_dir(args, "chat"), "chat")
        else:
            base_rows = _stage_kept_rows(_rows_dir(args, "plain"), "plain_text")
        if not base_rows:
            raise RuntimeError(f"fresh_draws: no kept base rows for {cell} (fail loud)")
        ids = sorted(base_rows)
        order = random.Random(cm.derived_seed(cm.SEED, "fresh_select", cell)).sample(
            range(len(ids)), len(ids)
        )
        sel = _shard_rows([ids[i] for i in order[: args.fresh_rows]], args)
        regime = {
            "phase": "fresh_draws",
            "cell": cell,
            "n": args.fresh_rows,
            "draw_seeds": draw_seeds,
            "shard": [args.shard_index, args.num_shards],
            "model": cm.MODEL_ID,
        }
        ledger = cm.StageLedger(
            raw_root / "fresh_draws" / f"ledger_{cell}_s{args.shard_index}.json", regime
        )
        for draw_seed in draw_seeds:
            key = f"{cell}|d{draw_seed}|s{args.shard_index}"
            if ledger.is_done(key):
                continue
            prompts, seeds = [], []
            for rid in sel:
                if cell in cm.STORY_CELLS:
                    assert mined is not None
                    prompt, _opener = _segb_prompt(mined[rid], banksd)
                    cap, stop = cm.SEGB_MAX_TOKENS, None
                elif cell == "chat":
                    prompt = _render_chat(tok, base_rows[rid]["question"])
                    cap, stop = cm.CHAT_MAX_TOKENS, cm.CHAT_STOP
                else:
                    q = base_rows[rid]["question"]
                    prompt = f"User: {q}\n\nAssistant:"
                    cap, stop = cm.PLAIN_MAX_TOKENS, cm.PLAIN_STOP
                prompts.append(prompt)
                seeds.append(cm.derived_seed(draw_seed, "fresh", cell, rid))
            sps = [_sampling_params(cap, stop, s) for s in seeds]
            outs = _chunked_generate(llm, prompts, sps, f"fresh/{cell}/d{draw_seed}")
            out_rows = []
            for rid, out, s in zip(sel, outs, seeds):
                text, finish = _gen_text(out)
                row = {
                    "cell": cell,
                    "row_id": rid,
                    "draw_seed": draw_seed,
                    "gen_text": text,
                    "finish_reason": finish,
                    "seed": s,
                    "template_sha": template_sha if cell == "chat" else None,
                }
                if cell in cm.STORY_CELLS:
                    close = _mine_closing_quote(text)
                    row["answer"] = text[:close].strip() if close is not None else None
                    row["answer_close_idx"] = close
                else:
                    row["answer"] = text.strip()
                out_rows.append(row)
            _write_chunk_jsonl(
                raw_root / "fresh_draws" / f"{cell}_d{draw_seed}_s{args.shard_index}.jsonl",
                out_rows,
            )
            ledger.mark_done(key)
            cm.progress("fresh_draws", draw_seeds.index(draw_seed) + 1, len(draw_seeds), key, t0)
    _reap_engine(llm)
    _maybe_upload(args, "fresh_draws")


def phase_capture_ready(args) -> None:
    """Reduce stage ledgers into per-cell capture_ready.json (plan §4.6 floors;
    §4.2b pair-complete intersection for the two user arms)."""
    ledger_root = Path(args.ledger_root)
    out_dir = ledger_root / "capture_ready"
    kept_dir = Path(args.kept_dir)

    def emit(cell: str, kept_ids: list[str], drops: dict, extra: dict | None = None) -> None:
        n = len(kept_ids)
        payload = {
            "cell": cell,
            "n_kept": n,
            "floor": cm.FLOOR_KEPT,
            "floor_pass": n >= cm.FLOOR_KEPT,
            "close_miss_band": cm.CLOSE_MISS_FLOOR <= n < cm.FLOOR_KEPT,
            "drop_counts": drops,
            "kept_ids": sorted(kept_ids),
            "metadata": cm.run_metadata(),
        }
        if extra:
            payload.update(extra)
        cm.atomic_write_json(out_dir / f"{cell}.json", payload)
        print(
            f"[capture_ready] {cell}: n_kept={n} floor_pass={payload['floor_pass']} "
            f"close_miss={payload['close_miss_band']}",
            flush=True,
        )

    segb_dir = _rows_dir(args, "segb")
    for cell in cm.STORY_CELLS:
        admitted = set(_load_kept_ids(kept_dir, cell))
        segb_kept = _stage_kept_rows(segb_dir, cell)
        kept = [rid for rid in segb_kept if rid in admitted]
        drops = {"admitted": len(admitted), "segb_resolved": len(segb_kept)}
        emit(cell, kept, drops)
    for cell, stage in (("chat", "chat"), ("plain_text", "plain")):
        kept_rows = _stage_kept_rows(_rows_dir(args, stage), cell)
        emit(cell, list(kept_rows), {"kept": len(kept_rows)})
    real_rows = _stage_kept_rows(_rows_dir(args, "user_real_render"))
    sim_rows = _stage_kept_rows(_rows_dir(args, "user_sim"))
    inter = sorted(set(real_rows) & set(sim_rows))
    pair_block = {
        "pair_intersection": {
            "n_real_kept": len(real_rows),
            "n_sim_kept": len(sim_rows),
            "n_intersection": len(inter),
            "intersection_ids": inter,
        }
    }
    emit("chat_user_real", list(real_rows), {"kept": len(real_rows)}, pair_block)
    emit("chat_user_sim", list(sim_rows), {"kept": len(sim_rows)}, pair_block)


def phase_upload_stage(args) -> None:
    if args.stage not in cm.RAW_STAGES:
        raise SystemExit(f"--stage must be one of {cm.RAW_STAGES}")
    cm.upload_stage_dir(
        Path(args.raw_root) / args.stage, f"{cm.HF_PREFIX}/raw_completions/{args.stage}"
    )


PHASES = {
    "build_banks": phase_build_banks,
    "build_pools": phase_build_pools,
    "sega": phase_sega,
    "chat_plain": phase_chat_plain,
    "user_sim": phase_user_sim,
    "user_fresh": phase_user_fresh,
    "user_real_render": phase_user_real_render,
    "segb": phase_segb,
    "fresh_draws": phase_fresh_draws,
    "capture_ready": phase_capture_ready,
    "upload_stage": phase_upload_stage,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness check, then exit 0",
    )
    ap.add_argument("--cells", default="", help="comma list; default = phase-appropriate set")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--wave", type=int, default=1, help="generation wave (retry waves 2/3)")
    ap.add_argument("--chunk-rows", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--raw-root", default=str(cm.RAW_ROOT_DEFAULT))
    ap.add_argument("--ledger-root", default=str(cm.LEDGER_ROOT))
    ap.add_argument("--pools-dir", default=str(cm.POOLS_DIR))
    ap.add_argument("--banks-dir", default=str(cm.BANKS_DIR))
    ap.add_argument("--cache-dir", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "api_cache"))
    ap.add_argument("--stage-pools-from-hf", action="store_true")
    ap.add_argument(
        "--stage-raw-from-hf",
        action="store_true",
        help="stage missing raw_completions inputs from HF (cross-pod reads)",
    )
    ap.add_argument("--mined-dir", default=str(cm.RAW_ROOT_DEFAULT / "sega_mined"))
    ap.add_argument("--kept-dir", default=str(cm.LEDGER_ROOT / "kept"))
    ap.add_argument("--sega-attempts-per-cell", type=int, default=0)
    ap.add_argument("--chat-rows", type=int, default=cm.CHAT_DRAW_N)
    ap.add_argument("--plain-rows", type=int, default=cm.PLAIN_DRAW_N)
    ap.add_argument("--user-sim-rows", type=int, default=cm.USER_DRAW_N)
    ap.add_argument("--user-fresh-rows", type=int, default=1000)
    ap.add_argument("--user-fresh-draws", type=int, default=4)
    ap.add_argument("--target-kept-per-cell", type=int, default=cm.STORY_TARGET_KEPT)
    ap.add_argument(
        "--chat-kept",
        type=int,
        default=cm.CHAT_TARGET_KEPT,
        help="chat/plain kept target (reporting only)",
    )
    ap.add_argument("--fresh-rows", type=int, default=1000)
    ap.add_argument("--fresh-draws", type=int, default=4)
    ap.add_argument("--stage", default="", help="stage name for --phase upload_stage")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    _ = args.chat_kept  # reporting-only target, consumed by the run digest
    PHASES[args.phase](args)
    sys.stdout.flush()
    sys.stderr.flush()
    if _ENGINE_USED:
        # vLLM generation driver terminal: engine reaped in-phase; skip
        # finalization (gotchas.md "sys.exit(0) is NOT a terminal").
        os._exit(0)
    sys.exit(0)


if __name__ == "__main__":
    main()
