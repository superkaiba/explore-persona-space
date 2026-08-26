"""issue #2378 generation driver — pools, banks, story/chat/plain/user-turn generation.

Phases (plan v6 §4.1/§4.2/§4.2b/§4.6; ``--phase`` over the ``PHASES`` registry):

VM phases (repo venv, no model):
- ``build_banks``   P0: LLM-author the scene seed banks (settings/situations/
                    registers) + write committed copies of the static
                    prime/opener/char-intro/final-seed banks + judge rubrics
                    (final seeds frozen static at the G1 recalibration, r11).
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

# Script-mode sys.path bootstrap (#823; r5 model-venv fix): under the dedicated
# model venv (/root/eps-model-venv — no editable install of this repo) neither
# `explore_persona_space` nor the scripts/ siblings are importable unless the
# repo's src/ + scripts/ dirs are on sys.path. Mirrors issue2378_dispatch.py.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2378_banks as bnk  # noqa: E402
import issue2378_common as cm  # noqa: E402

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
# character within the miner window (cm.MINER_WINDOW_TOKENS — 512 since the
# G1 recalibration, covering the full SegA generation). Char offsets.
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


_QUOTE_CHARS = "\"“”'‘’«»"


def _is_directed(window: str, span: tuple[int, int, int, int], character: str) -> bool:
    """Attribution heuristics: the utterance is addressed TO ``character``
    (never spoken BY them). Pilot-precision-gated at G1 (plan §4.2).

    r1 review fix (g1 blocker): before/after windows are split. A ``{Name}
    <verb>`` attribution AFTER the close that INTRODUCES A NEW QUOTE is the
    canonical prime/opener-taught reply shape (``"Q?" {Name} replied: "A."``)
    — the character ANSWERS this utterance, so it IS directed at them; the
    same attribution with no follow-on quote (``"Q?" {Name} said.``) marks
    the character as this quote's SPEAKER and rejects."""
    open_idx, cs, ce, close_end = span
    utter = window[cs:ce]
    before = window[max(0, open_idx - 120) : open_idx]
    after = window[close_end : close_end + 120]
    name = re.escape(character)
    v = _ATTRIB_VERBS
    q = _QUOTE_CHARS
    # (1a) SPEAKER before the quote: `{Name} said:` / `said {Name}:` adjacent
    # to the opening quote (sentence-initial for the inverted form, so
    # `Dana asked {Name}: "Q?"` stays a directed-TO shape, not a reject).
    if re.search(rf"\b{name}\s+(?:\w+\s+)?(?:{v})\b[^{q}]{{0,12}}$", before):
        return False
    if re.search(rf"(?:^|[.!?”\"]\s*)(?:{v})\s+{name}\b[^{q}]{{0,12}}$", before):
        return False
    # (1b) inverted post-quote attribution `"Q?" asked {Name}` (verb FIRST
    # after the close — no subject can intervene) => {Name} spoke THIS quote.
    if re.match(rf"\s*[,—–-]*\s*(?:{v})\s+{name}\b", after):
        return False
    # (1c) post-quote `{Name} <verb>`: with a follow-on quote it is the
    # canonical reply shape (KEEP); without one, {Name} spoke THIS quote.
    m = re.match(
        rf"\s*[,—–-]*\s*{name}\s+(?:[\w,';]+\s+){{0,3}}?(?:{v})\b",
        after,
    )
    if m:
        return bool(re.match(rf"[^{q}]{{0,40}}[\"“«‘']", after[m.end() :]))
    near = before + " " + after
    if re.search(rf"\b(?:{v})(?:\s+\w+){{0,2}}?\s+(?:to\s+)?{name}\b", near):
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

    G1 recalibration (r11, L3): the scan stops at the FIRST directed quote — a
    directed quote failing the family kind check REJECTS the row (wrong_kind)
    instead of scanning forward, which salvaged the character's own reply
    (21/102 pilot dialogue admission rejects were reply-mined). Non-directed
    and degenerate (<3 char) quotes still skip forward.
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
                break
        elif "?" in utter:
            last_reason = "wrong_kind"
            break
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
    (plan §4.2b slot definition; the SHARED context prefix of BOTH user arms).

    NOTE (r13/r14): a1 is the LAST message here, so the Qwen3.6 template
    renders it WITH the empty ``<think>\\n\\n</think>`` block; the template's
    own 3-turn render (u1, a1, u2) STRIPS that block from a1 (it attaches
    only to assistant turns AFTER the last user query), so this prefix is
    NOT a prefix of the template's 3-turn render. The sim arm SAMPLED its
    wave-1 user turns under this prefill, and the real arm teacher-forces
    the direct join ``prefix + u2`` (``_render_user_real_tf``) — so the two
    arms' context bytes, hence v_C / v_P, are identical by construction
    (§4.2b pair contract; r14 fix — r13 used the template's 3-turn render
    for the real arm and broke the contract)."""
    body = tok.apply_chat_template(
        [{"role": "user", "content": u1}, {"role": "assistant", "content": a1}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return body + cm.USER_TURN_HEADER


def _user_real_tf_from_prefix(prefix: str, u2: str) -> str:
    """The single direct-join assembly both the producer and the capture
    consumer share: rendered prefix + raw u2 + turn end."""
    return prefix + u2 + cm.TURN_END


def _render_user_real_tf(tok, u1: str, a1: str, u2: str) -> str:
    """REAL-u2 teacher-forced text (r14): the SIM arm's EXACT rendered prefix
    (``_render_user_prefix`` — a1 WITH the empty ``<think>`` block) + the raw
    u2 text + ``TURN_END``, joined directly.

    DECLARED DEVIATION (same class as the chat/plain/sim direct joins in the
    capture module docstring): this is NOT the template's own 3-turn render —
    Qwen3.6 strips a1's empty ``<think>`` block once a later user turn
    exists, so the template render's context bytes differ from the sim arm's
    generation prefill. The §4.2b paired-context contract (identical v_C /
    v_P across the two user arms by construction) takes precedence over
    template fidelity for this teacher-forced arm: r13 shipped the template
    render and ``p6_common.assert_user_pair`` would have failed
    deterministically whenever both arms survived (r13 review blockers
    user-pair-vc-assert-guaranteed-fail /
    user-arm-context-identity-contract-broken; both reviewers named this
    exact fix)."""
    return _user_real_tf_from_prefix(_render_user_prefix(tok, u1, a1), u2)


def _user_real_span(rendered_full: str, u2: str) -> tuple[int, int] | None:
    """Tail-anchored u2 char span in the teacher-forced render.

    The render ends with ``USER_TURN_HEADER + u2 + TURN_END`` by construction
    (r14 direct join; the same content-independent tail held for r13's
    template render — #1776 recipe), so the span is pure end-arithmetic and
    equals ``(len(prefix), len(prefix) + len(u2))``. Returns None when the
    tail does not match (defensive fail-visible drop — cannot fire on the
    direct join unless the row's stored render drifted from the pool row)."""
    tail = cm.USER_TURN_HEADER + u2 + cm.TURN_END
    if not rendered_full.endswith(tail):
        return None
    lo = len(rendered_full) - len(tail) + len(cm.USER_TURN_HEADER)
    return lo, lo + len(u2)


def _user_real_row(tok, r: dict) -> dict:
    """One chat_user_real render row (the phase_user_real_render per-row body;
    shared with the r13/r14 pin tests so fixtures cannot drift from the
    writer). Spans derive from ``len(prefix)`` (r14 §4.2b pair contract),
    cross-checked against the content-independent tail anchor."""
    u2 = r["u2"]
    prefix = _render_user_prefix(tok, r["u1"], r["a1"])
    rendered_full = _user_real_tf_from_prefix(prefix, u2)
    span = _user_real_span(rendered_full, u2)
    if span != (len(prefix), len(prefix) + len(u2)):
        return {
            "cell": "chat_user_real",
            "conv_id": r["conv_id"],
            "keep": False,
            "drop_reason": "span_mismatch",
        }
    lo, hi = span
    return {
        "cell": "chat_user_real",
        "conv_id": r["conv_id"],
        "rendered_text": rendered_full,
        "header_end": lo,
        "u2_span": [lo, hi],
        "keep": True,
        "drop_reason": None,
    }


def _n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _build_engine(args):
    global _ENGINE_USED
    import dataclasses

    from explore_persona_space.eval.generation import create_vllm_engine

    # `language_model_only` (skip the omni model's non-text towers) exists
    # only on newer vLLM EngineArgs — introspection-guarded so an engine
    # without it skips the optimization instead of dying TypeError at init
    # (r1 review g1 concern 7; the VM venv's vLLM 0.11.0 lacks it).
    from vllm.engine.arg_utils import EngineArgs

    kwargs: dict = {}
    if "language_model_only" in {f.name for f in dataclasses.fields(EngineArgs)}:
        kwargs["language_model_only"] = True
    else:
        print("[engine] vLLM EngineArgs lacks language_model_only — skipped", flush=True)
    if args.tp > 1:
        kwargs["tensor_parallel_size"] = args.tp
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    # r9 (epm:failure v5): pin the GDN prefill backend off vllm 0.27.1's
    # SM90 flashinfer auto-select (rationale at cm.ENGINE_KWARG_PINS).
    # Deliberately NOT introspection-guarded: an engine lacking the
    # EngineArgs field TypeErrors loudly — never a silent skip.
    kwargs.update(cm.ENGINE_KWARG_PINS)
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


def _sampling_params(
    max_tokens: int, stop: list[str] | None, seed: int, bad_words: list[str] | None = None
):
    """Shared SamplingParams builder. ``bad_words`` carries the '<think>'
    reasoning-mode ban on the RAW continuation legs — SegA mining (G1
    recalibration L1: 73-81% of pilot attempts leaked), plain answers +
    fresh plain draws, and sim user turns (r13 / G2b fix 1: 89% and 2.8%
    wave-1 leaks); chat is template-path immune and segb stays deliberately
    unbanned. Threaded UNGUARDED: verified constructible on the production
    model venv (vllm 0.27.1, pod-2378) — an engine venv lacking the param
    must TypeError loudly, never introspection-skip (the r9
    ENGINE_KWARG_PINS lesson)."""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=cm.TEMPERATURE,
        top_p=cm.TOP_P,
        top_k=cm.TOP_K,
        seed=seed,
        max_tokens=max_tokens,
        stop=stop,
        bad_words=bad_words,
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


def _parse_bank_array(text: str) -> list | None:
    """Parse a bank-builder response into the JSON ARRAY the _BANK_SYSTEM
    contract demands. eval.utils.parse_judge_json anchors its recovery at the
    first ``{``, which for an array-of-OBJECTS response returns the FIRST
    OBJECT alone — the r2 real-API bank smoke caught exactly that (registers
    bank: expected 8 items, got dict). Ladder: verbatim JSON, then the largest
    recoverable ARRAY from fenced blocks / ``raw_decode`` at ``[`` offsets
    (bounded). Returns None when no array parses — the caller's expected-n
    check raises loud."""
    t = text.strip()
    try:
        v = json.loads(t)
        if isinstance(v, list):
            return v
    except json.JSONDecodeError:
        pass
    best: list | None = None

    def consider(v: object) -> None:
        nonlocal best
        if isinstance(v, list) and (best is None or len(v) > len(best)):
            best = v

    for m in re.finditer(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL):
        try:
            consider(json.loads(m.group(1).strip()))
        except json.JSONDecodeError:
            continue
    dec = json.JSONDecoder()
    for i in [i for i, ch in enumerate(t) if ch == "["][:50]:
        try:
            v, _end = dec.raw_decode(t, i)
        except json.JSONDecodeError:
            continue
        consider(v)
    return best


def phase_build_banks(args) -> None:
    """LLM-author the sampled scene axes; write committed bank + rubric files.

    ``--bank-only <name>`` restricts to ONE LLM-authored bank and returns
    before the static/rubric writes — the tiny bounded REAL-API smoke of the
    bank-builder path (r1 review blocker: real-api-smokes-missing)."""
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

    specs = dict(bnk.BANK_BUILDER_SPECS)
    if args.bank_only:
        if args.bank_only not in specs:
            raise SystemExit(f"unknown bank {args.bank_only}; choices: {sorted(specs)}")
        specs = {args.bank_only: specs[args.bank_only]}
    items = [
        DispatchItem(item_id=f"bank|{name}", payload={"name": name, "prompt": spec["prompt"]})
        for name, spec in specs.items()
    ]
    results = asyncio.run(
        dispatch_calls(
            items,
            model=cm.JUDGE_MODEL,
            build_request=build_request,
            parse_response=_parse_bank_array,
            force_path="sync",
            cache_dir=Path(args.cache_dir) / "banks",
        )
    )
    banks_dir = Path(args.banks_dir)
    banks_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in specs.items():
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

    if args.bank_only:
        print(f"[build_banks] --bank-only {args.bank_only}: skipping static/rubric writes")
        return

    static = {
        "prime_bank_question": list(bnk.PRIME_BANK_QUESTION),
        "prime_bank_dialogue": list(bnk.PRIME_BANK_DIALOGUE),
        # G1 recalibration (r11, L4): the final-seed banks are FROZEN static
        # tuples (P0 LLM-authored, audited + reworded at G1 — provenance in
        # issue2378_banks.py); build_banks writes them deterministically from
        # source, validated fail-loud at issue2378_banks import time.
        "final_seeds_question": list(bnk.FINAL_SEEDS_QUESTION),
        "final_seeds_remark": list(bnk.FINAL_SEEDS_REMARK),
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
    # Real LMSYS/WildChat rows can embed leaked third-party credentials; the
    # upload gate (orchestrate/secret_scrub.py) refuses them. Same-length X
    # placeholders keep char offsets/spans valid. Re-runs re-draw the same
    # rows (seeded permutation), so the scrub must live here, not be a
    # one-time file fix.
    from explore_persona_space.orchestrate.secret_scrub import scrub_file

    scrub_counts: dict[str, int] = {}
    for fname in ("chat_draw", "plain_draw", "user_draw"):
        fixed = scrub_file(pools_dir / f"{fname}.jsonl")
        if fixed:
            scrub_counts[f"{fname}.jsonl"] = len(fixed)
            print(
                f"[build_pools] scrubbed {len(fixed)} leaked third-party secret(s) "
                f"from {fname}.jsonl (same-length placeholders)",
                flush=True,
            )
    digest = {
        "secret_scrub_fixed": scrub_counts,
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
            # G1 recalibration L1: the decoding pin is output-affecting, so it
            # rides the regime (ledger fail-louds against pre-r11 state; round-2
            # raw roots are round-scoped by design) and the summary artifact.
            "bad_words": ["<think>"],
        }
        ledger = cm.StageLedger(
            raw_root / "sega" / f"ledger_{cell}_w{wave}_s{args.shard_index}.json", regime
        )
        n_chunks = (len(attempt_ids) + args.chunk_rows - 1) // args.chunk_rows
        for ci in range(n_chunks):
            key = f"{cell}|w{wave}|s{args.shard_index}|c{ci:04d}"
            chunk = attempt_ids[ci * args.chunk_rows : (ci + 1) * args.chunk_rows]
            if ledger.is_done(key):
                continue
            built = [_build_scene_prompt(cell, wave, a, banksd) for a in chunk]
            sps = [
                _sampling_params(
                    cm.SEGA_MAX_TOKENS,
                    None,
                    cm.derived_seed(cm.SEED, "sega", cell, wave, a),
                    # G1 recalibration L1: ban the '<think>' reasoning-mode leak
                    # on the raw few-shot mining continuation (73-81% of pilot
                    # attempts leaked). r13 extends the ban to the OTHER raw
                    # continuation legs — plain answers + sim user turns (89% /
                    # 2.8% wave-1 leaks); segb + chat stay deliberately
                    # unbanned (chat is template-path immune; segb cells passed
                    # their floors without it).
                    bad_words=["<think>"],
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
            stem = f"{cell}_w{wave}_s{args.shard_index}_c{ci:04d}"
            _write_chunk_jsonl(raw_root / "sega" / f"{stem}.jsonl", raw_rows)
            if mined_rows:
                _write_chunk_jsonl(raw_root / "sega_mined" / f"{stem}.jsonl", mined_rows)
            ledger.mark_done(key)
            cm.progress("sega", ci + 1, n_chunks, key, t0)
        # Durable-file summary (r1 review major 13): counts recomputed from ALL
        # persisted chunk files so a resumed shard reports full totals — the G1
        # sizing recalibration consumes these.
        counts = {"attempts": 0, "kept": 0, "cap_hit": 0}
        mine_rejects: dict[str, int] = {}
        for path in sorted(
            (raw_root / "sega").glob(f"{cell}_w{wave}_s{args.shard_index}_c*.jsonl")
        ):
            for row in cm.iter_jsonl(path):
                counts["attempts"] += 1
                if row.get("finish_reason") == "length":
                    counts["cap_hit"] += 1
                if row["mined"].get("kept"):
                    counts["kept"] += 1
                else:
                    reason = row["mined"].get("reason") or "unknown"
                    mine_rejects[reason] = mine_rejects.get(reason, 0) + 1
        summary = {
            # r10 (G1 accounting fix): dispatch._sum_stage_summaries keys the
            # family pools on this — the filename alone left the pre-r10
            # composer keying every summary by stage dir name -> net 0.0.
            "cell": cell,
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


def _cap_hit_fraction(files: list[Path], is_hit) -> tuple[int, int]:
    """Count (generated_rows, hit_rows) over durable chunk files. Rows dropped
    before generation (e.g. over_length) carry no finish_reason and are
    excluded from the denominator."""
    gen_rows = hits = 0
    for path in files:
        for row in cm.iter_jsonl(path):
            if "finish_reason" not in row:
                continue
            gen_rows += 1
            if is_hit(row):
                hits += 1
    return gen_rows, hits


def _cell_grain_regen(
    llm,
    files: list[Path],
    decision_path: Path,
    *,
    is_hit,
    rebuild,
    update_row,
    tag: str,
    bad_words: list[str] | None = None,
) -> dict:
    """Cap-hit > 2% regen at PER-SHARD grain (r1 review majors 13+15; grain
    relabeled per the r2 reconciler disposition of cap-hit-rule-wrong-grain):
    the trigger is evaluated over the cell's full durable row set FOR THIS
    SHARD — a per-shard ESTIMATOR of the cell fraction (round-robin sharding
    makes the shard fraction ≈ the cell fraction; no cross-shard join is
    performed), never per chunk. Decisions are durable + shard-tagged (the
    ``decision_path`` filenames carry ``s{shard}``) and carry an explicit
    ``grain`` field; per-row ``regen`` flags are durable too, so a
    crashed/resumed invocation continues the regen instead of re-deciding on
    a partial (or already-regenerated) view.

    ``is_hit(row)`` defines a cap-hit on a generated row; ``rebuild(row)``
    returns ``(prompt, cap2, stop)`` for the 2x pass; ``update_row(row, text,
    finish)`` re-classifies the row in place (keep/drop/answer fields).
    ``bad_words`` threads the caller's decoding ban into the 2x pass so a
    regenerated row samples under the SAME regime as its 1x pass (r13: the
    plain/user '<think>' ban must not silently drop on regen).
    """
    files = [f for f in files if f.exists()]
    if decision_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    else:
        gen_rows, hits = _cap_hit_fraction(files, is_hit)
        frac_before = hits / max(1, gen_rows)
        decision = {
            "regen": bool(hits and frac_before > cm.CAP_HIT_REGEN_THRESHOLD),
            "frac_before": frac_before,
            "n_generated": gen_rows,
            "n_hit": hits,
            "grain": "shard (round-robin per-shard estimator of the cell fraction)",
            "done": False,
        }
        cm.atomic_write_json(decision_path, decision)
    if decision.get("done"):
        return decision
    if decision["regen"]:
        for path in files:
            rows = list(cm.iter_jsonl(path))
            todo = [
                i
                for i, r in enumerate(rows)
                if "finish_reason" in r and not r.get("regen") and is_hit(r)
            ]
            if not todo:
                continue
            prompts, sps = [], []
            for i in todo:
                prompt, cap2, stop = rebuild(rows[i])
                prompts.append(prompt)
                sps.append(
                    _sampling_params(
                        cap2, stop, cm.derived_seed(rows[i]["seed"], "regen"), bad_words=bad_words
                    )
                )
            outs = _chunked_generate(llm, prompts, sps, f"{tag}/regen2x")
            for k, i in enumerate(todo):
                text, finish = _gen_text(outs[k])
                update_row(rows[i], text, finish)
                rows[i]["regen"] = True
            _write_chunk_jsonl(path, rows)
            print(f"[{tag}] cell-grain 2x regen: {len(todo)} rows in {path.name}", flush=True)
    gen_rows, hits = _cap_hit_fraction(files, is_hit)
    decision.update({"done": True, "frac_after": hits / max(1, gen_rows)})
    cm.atomic_write_json(decision_path, decision)
    print(
        f"[{tag}] cap-hit shard fraction (cell estimator) {decision['frac_before']:.4f} -> "
        f"{decision['frac_after']:.4f} (regen={decision['regen']})",
        flush=True,
    )
    return decision


def _classify_answer_row(row: dict, text: str) -> None:
    """Set answer/keep/drop_reason on a chat/plain answer row (shared by the
    1x pass and the 2x cell-grain regen; cap-hit rows stay kept — the
    fraction is reported in the stage summary). The ``<think>`` check is the
    plan §4.2 chat-cell literal (r1 review g1 concern 1); a plain-text answer
    carrying it is equally anomalous, so both cells drop on it."""
    answer = text.strip()
    keep, drop_reason = True, None
    if not answer:
        keep, drop_reason = False, "empty_answer"
    elif "<think>" in answer:
        keep, drop_reason = False, "think_leak"
    elif not cm.english_majority(answer):
        keep, drop_reason = False, "non_english_answer"
    row.update({"answer": answer, "keep": keep, "drop_reason": drop_reason})


def _run_answer_cell(args, llm, tok, cell: str, rows: list[dict], template_sha: str) -> None:
    """Shared chat / plain_text answer generation (plan §4.2 chat/plain cells)."""
    raw_root = Path(args.raw_root)
    stage = "chat" if cell == "chat" else "plain"
    cap = cm.CHAT_MAX_TOKENS if cell == "chat" else cm.PLAIN_MAX_TOKENS
    stop = cm.CHAT_STOP if cell == "chat" else cm.PLAIN_STOP
    budget = args.max_model_len - 2 * cap  # leave room for the 2x regen pass
    wave = args.wave
    # r13 (G2b fix 1): the r11 '<think>' ban covered SegA only; the PLAIN raw
    # continuation leaked <think> on 89% of P4 wave-1 rows (capture_ready:
    # 1,235/10,000 kept). Chat is template-path immune — the render already
    # carries the empty <think> block — and its regime dict stays byte-stable
    # so completed wave-1 chat ledgers keep resuming.
    ban = None if cell == "chat" else ["<think>"]
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
    if ban:
        # Output-affecting decoding pin rides the regime (the sega L1
        # precedent): a stale pre-r13 plain root fails loud instead of mixing
        # banned and unbanned rows in one cell.
        regime["bad_words"] = ban
    ledger = cm.StageLedger(
        raw_root / stage / f"ledger_{cell}_w{wave}_s{args.shard_index}.json", regime
    )
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
        sps = [_sampling_params(cap, stop, s, bad_words=ban) for s in seeds]
        outs = _chunked_generate(llm, prompts, sps, f"{stage}")
        out_rows = list(dropped_rows)
        for r, out, s in zip(kept_rows, outs, seeds):
            text, finish = _gen_text(out)
            row = {
                "cell": cell,
                "conv_id": r["conv_id"],
                "question": r["question"],
                "finish_reason": finish,
                "seed": s,
                "regen": False,
                "template_sha": template_sha if cell == "chat" else None,
            }
            _classify_answer_row(row, text)
            out_rows.append(row)
        stem = f"{cell}_w{wave}_s{args.shard_index}_c{ci:04d}"
        _write_chunk_jsonl(raw_root / stage / f"{stem}.jsonl", out_rows)
        ledger.mark_done(key)
        cm.progress(stage, ci + 1, n_chunks, key, t0)
    # Cell-grain cap-hit regen + durable-file summary (r1 review majors 13+15).
    files = sorted((raw_root / stage).glob(f"{cell}_w{wave}_s{args.shard_index}_c*.jsonl"))

    def _rebuild(row: dict) -> tuple[str, int, list[str] | None]:
        if cell == "chat":
            return _render_chat(tok, row["question"]), 2 * cap, stop
        return f"User: {row['question']}\n\nAssistant:", 2 * cap, stop

    def _update(row: dict, text: str, finish: str) -> None:
        row["finish_reason"] = finish
        _classify_answer_row(row, text)

    decision = _cell_grain_regen(
        llm,
        files,
        raw_root / stage / f"regen_decision_{cell}_w{wave}_s{args.shard_index}.json",
        is_hit=lambda r: r.get("finish_reason") == "length",
        rebuild=_rebuild,
        update_row=_update,
        tag=stage,
        bad_words=ban,
    )
    counts = {
        "rows": 0,
        "kept": 0,
        "over_length": 0,
        "non_english_answer": 0,
        "empty_answer": 0,
        "think_leak": 0,
    }
    for path in files:
        for row in cm.iter_jsonl(path):
            if row.get("drop_reason") == "over_length":
                counts["over_length"] += 1
                continue
            counts["rows"] += 1
            if row.get("keep"):
                counts["kept"] += 1
            elif row.get("drop_reason") in counts:
                counts[row["drop_reason"]] += 1
    summary = {
        "cell": cell,  # r10: per-cell key for _sum_stage_summaries-style aggregation
        "regime": regime,
        "counts": counts,
        "cap_hit_fraction_before": decision["frac_before"],
        "cap_hit_fraction_after": decision["frac_after"],
        "cap_hit_regen": decision["regen"],
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


def _classify_sim_turn(text: str) -> tuple[bool, str | None]:
    """Sim-turn eligibility (plan §4.2b), mirroring the REAL u2 pool filters:
    length band AND English-majority script (r1 review major: the sim arm
    previously omitted the English predicate the real arm applies)."""
    turn = text.strip()
    if not turn:
        return False, "empty_turn"
    if "<think>" in turn:
        return False, "think_leak"
    if not (cm.USER_TURN_MIN_CHARS <= len(turn) <= cm.USER_TURN_MAX_CHARS):
        return False, "len_band"
    if not cm.english_majority(turn):
        return False, "non_english"
    return True, None


def _classify_sim_row(row: dict, text: str) -> None:
    """Set sim_turn/keep/drop_reason on a sim-user row (1x pass + 2x regen)."""
    keep, drop_reason = _classify_sim_turn(text)
    row.update({"sim_turn": text.strip(), "keep": keep, "drop_reason": drop_reason})


def _run_user_sim(args, llm, tok, rows: list[dict], stage: str, draw_seeds: list[int]) -> None:
    """Sim-user turn generation (plan §4.2b): raw continuation of the rendered
    prefill; one pass per draw seed (production: [SEED]; fresh: 138-141)."""
    raw_root = Path(args.raw_root)
    cap = cm.USER_SIM_MAX_TOKENS
    budget = args.max_model_len - 2 * cap
    wave = args.wave
    # r13 (G2b fix 1 audit): the sim-user raw continuation is a raw-completion
    # leg like plain (277/10,000 wave-1 think_leaks) — same '<think>' ban;
    # output-affecting, so it rides the regime (fresh roots by design).
    ban = ["<think>"]
    regime = {
        "phase": stage,
        "wave": wave,
        "n": len(rows),
        "draw_seeds": draw_seeds,
        "shard": [args.shard_index, args.num_shards],
        "cap": cap,
        "model": cm.MODEL_ID,
        "bad_words": ban,
    }
    ledger = cm.StageLedger(raw_root / stage / f"ledger_w{wave}_s{args.shard_index}.json", regime)
    pool_by_id = {r["conv_id"]: r for r in rows}
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
            sps = [_sampling_params(cap, cm.USER_SIM_STOP, s, bad_words=ban) for s in seeds]
            outs = _chunked_generate(llm, prompts, sps, stage)
            out_rows = list(dropped_rows)
            for r, prefix, out, s in zip(kept_rows, prompts, outs, seeds):
                text, finish = _gen_text(out)
                row = {
                    "cell": "chat_user_sim",
                    "conv_id": r["conv_id"],
                    "draw_seed": draw_seed,
                    "finish_reason": finish,
                    "seed": s,
                    "regen": False,
                    "prefix_chars": len(prefix),
                    "prefix_digest": cm.text_digest(prefix),
                }
                _classify_sim_row(row, text)
                out_rows.append(row)
            stem = f"w{wave}_d{draw_seed}_s{args.shard_index}_c{ci:04d}"
            _write_chunk_jsonl(raw_root / stage / f"{stem}.jsonl", out_rows)
            ledger.mark_done(key)
            cm.progress(stage, ci + 1, n_chunks, key, t0)
    # Cell-grain cap-hit regen + durable-file summary (r1 review majors 13+15);
    # fresh draws (user_sim_fresh) share this path, so the > 2% rule covers
    # them too (r1 review: cap-hit rule skipped fresh draws).
    files = sorted((raw_root / stage).glob(f"w{wave}_d*_s{args.shard_index}_c*.jsonl"))

    def _rebuild(row: dict) -> tuple[str, int, list[str] | None]:
        pool_row = pool_by_id.get(row["conv_id"])
        if pool_row is None:
            raise RuntimeError(f"{stage} regen: conv_id {row['conv_id']} not in the user pool")
        return _render_user_prefix(tok, pool_row["u1"], pool_row["a1"]), 2 * cap, cm.USER_SIM_STOP

    def _update(row: dict, text: str, finish: str) -> None:
        row["finish_reason"] = finish
        _classify_sim_row(row, text)

    decision = _cell_grain_regen(
        llm,
        files,
        raw_root / stage / f"regen_decision_w{wave}_s{args.shard_index}.json",
        is_hit=lambda r: r.get("finish_reason") == "length",
        rebuild=_rebuild,
        update_row=_update,
        tag=stage,
        bad_words=ban,
    )
    counts: dict[str, int] = {
        "rows": 0,
        "kept": 0,
        "over_length": 0,
        "empty_turn": 0,
        "think_leak": 0,
        "len_band": 0,
        "non_english": 0,
    }
    for path in files:
        for row in cm.iter_jsonl(path):
            if row.get("drop_reason") == "over_length":
                counts["over_length"] += 1
                continue
            counts["rows"] += 1
            if row.get("keep"):
                counts["kept"] += 1
            elif row.get("drop_reason") in counts:
                counts[row["drop_reason"]] += 1
    summary = {
        "regime": regime,
        "counts": counts,
        "cap_hit_fraction_before": decision["frac_before"],
        "cap_hit_fraction_after": decision["frac_after"],
        "cap_hit_regen": decision["regen"],
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
    """Fresh sim-user draws (seeds 138-141) for the selected held-out rows.

    Selection draws from KEPT user_sim conversations (r1 review g1 concern 6:
    a raw-pool draw shrinks the realized 5-draw-covered subset by the sim
    keep rate) — user_sim must have run first (locally or HF-staged)."""
    pools_dir = _resolve_pools_dir(args)
    tok = _get_tokenizer()
    _assert_chat_template(tok)
    rows = _load_pool(pools_dir, "user_draw")
    sim_kept = set(_stage_kept_rows(_rows_dir(args, "user_sim")))
    rows = [r for r in rows if r["conv_id"] in sim_kept]
    if not rows:
        raise RuntimeError("user_fresh: no kept user_sim conversations (run user_sim first)")
    llm = _build_engine(args)
    order = random.Random(cm.derived_seed(cm.SEED, "user_fresh_select")).sample(
        range(len(rows)), len(rows)
    )
    sel = [rows[i] for i in order[: args.user_fresh_rows]]
    draw_seeds = list(cm.FRESH_SEEDS[: args.user_fresh_draws])
    _run_user_sim(args, llm, tok, sel, "user_sim_fresh", draw_seeds)
    _reap_engine(llm)
    _maybe_upload(args, "user_sim_fresh")


def phase_user_real_render(args) -> None:
    """Real user-turn arm (plan §4.2b): NO generation — deterministic
    direct-join teacher-forced render (r14: the sim arm's exact prefix + u2 +
    TURN_END, ``_render_user_real_tf`` carries the declared deviation) with
    the span at ``len(prefix)``, tail-anchor cross-checked."""
    pools_dir = _resolve_pools_dir(args)
    tok = _get_tokenizer()
    _assert_chat_template(tok)
    rows = _load_pool(pools_dir, "user_draw")
    raw_root = Path(args.raw_root)
    counts = {"rows": 0, "kept": 0, "span_mismatch": 0}
    out_rows = []
    t0 = time.time()
    for k, r in enumerate(rows):
        row = _user_real_row(tok, r)
        counts["rows"] += 1
        counts["kept" if row["keep"] else "span_mismatch"] += 1
        out_rows.append(row)
        if len(out_rows) >= 2000 or k == len(rows) - 1:
            ci = k // 2000
            _write_chunk_jsonl(raw_root / "user_real_render" / f"c{ci:04d}.jsonl", out_rows)
            out_rows = []
            cm.progress("user_real_render", k + 1, len(rows), f"c{ci:04d}", t0)
    summary = {"counts": counts, "metadata": cm.run_metadata()}
    cm.atomic_write_json(raw_root / "user_real_render" / "summary.json", summary)
    print(f"[user_real_render] {json.dumps(counts)}", flush=True)
    _maybe_upload(args, "user_real_render")


# Per-process memos for _rows_dir's remote-manifest reconciliation: ONE scoped
# listing per stage (_STAGE_MANIFEST_CACHE), one local-dir verdict per
# (local dir, stage) (_STAGE_RECON_CACHE), and one VERIFIED mirror leaf per
# stage (_STAGE_MIRROR_CACHE) — capture calls _rows_dir per cell × draw.
# Probes clear all three between scenarios.
_STAGE_RECON_CACHE: dict[tuple[str, str], bool] = {}
_STAGE_MANIFEST_CACHE: dict[str, dict[str, int | None]] = {}
_STAGE_MIRROR_CACHE: dict[str, Path] = {}

# Mirror root for HF-staged raw-completions stages (module-level so probes can
# redirect it off the real data dir).
HF_STAGE_ROOT = cm.REPO_ROOT / "data" / "issue_2378" / "hf_stage"


def _stage_remote_manifest(stage: str) -> dict[str, int | None]:
    """Memoized producer manifest for raw_completions/<stage>: ``*.jsonl``
    name -> byte size from ONE scoped HF listing (#833/#1547). Serves BOTH the
    local-dir reconciliation (`_local_stage_covers_remote`) and the
    stale-mirror repair (`_repair_stale_mirror`), so the mismatch fall-through
    costs no extra listing."""
    if stage in _STAGE_MANIFEST_CACHE:
        return _STAGE_MANIFEST_CACHE[stage]
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    prefix = f"{cm.HF_PREFIX}/raw_completions/{stage}"
    entries = hub.retry_transient(
        lambda: hub.list_hf_entries_under_path(
            HfApi(), cm.HF_DATA_REPO, prefix, repo_type="dataset"
        ),
        what=f"rows-dir manifest listing {prefix}",
    )
    remote = {p.rsplit("/", 1)[-1]: s for p, s in entries if p.endswith(".jsonl")}
    _STAGE_MANIFEST_CACHE[stage] = remote
    return remote


def _repair_stale_mirror(leaf: Path, remote: dict[str, int | None]) -> list[str]:
    """Delete stale ``*.jsonl`` files from the HF mirror leaf BEFORE restaging
    (r3 reconciler concern rows-dir-mismatch-fallthrough-stale-mirror):
    ``hub.stage_hub_file`` returns an EXISTING target unchanged
    (``overwrite=False`` skip-existing, orchestrate/hub.py), so a prior
    same-pod staging's stale bytes would otherwise be served verbatim by the
    fall-through restage. Stale ⇔ not in the remote manifest, or a byte-size
    mismatch (the same grain as the reconciliation predicate); unknown-size
    remote entries are kept. Returns the deleted names."""
    if not leaf.is_dir():
        return []
    deleted: list[str] = []
    for lp in sorted(leaf.glob("*.jsonl")):
        size = remote.get(lp.name, "absent")
        if size == "absent" or (size is not None and lp.stat().st_size != int(size)):
            lp.unlink()
            deleted.append(lp.name)
    return deleted


def _local_stage_covers_remote(local: Path, stage: str) -> bool:
    """Reconcile a nonempty local stage dir against the producer's remote
    manifest — the (path, size) HF listing under raw_completions/<stage>, the
    same listing the P3 judge resume digest hashes (r2 review concern
    local-raw-stage-completeness-unchecked). Covers ⇔ every remote ``*.jsonl``
    exists locally with the same name AND byte size. An empty/absent remote
    prefix (nothing published yet — producer-side defensive flag) accepts the
    local dir; a mismatch logs loud and the caller falls through to a fresh
    HF mirror stage. Scoped listing + retry_transient per #833/#1547."""
    key = (str(local), stage)
    if key in _STAGE_RECON_CACHE:
        return _STAGE_RECON_CACHE[key]
    remote = _stage_remote_manifest(stage)
    mismatches: list[str] = []
    for name, size in sorted(remote.items()):
        lp = local / name
        if not lp.is_file():
            mismatches.append(f"{name}: missing locally")
        elif size is not None and lp.stat().st_size != int(size):
            mismatches.append(f"{name}: local {lp.stat().st_size} B != remote {size} B")
    ok = not mismatches
    if not ok:
        print(
            f"[stage] {stage}: local dir {local} FAILS remote-manifest reconciliation "
            f"({len(mismatches)}/{len(remote)} mismatched; first: {mismatches[:3]}) — "
            "falling through to a fresh HF mirror stage",
            flush=True,
        )
    _STAGE_RECON_CACHE[key] = ok
    return ok


def _rows_dir(args, stage: str, explicit: str | None = None) -> Path:
    """Resolve a raw-completions stage dir: local-first, HF-staged fallback
    (cross-pod reads — e.g. pod B consuming pod A's chat/plain/user_sim rows),
    fail-loud otherwise (#779/#1773 lane-input staging).

    On the ``--stage-raw-from-hf`` path a nonempty local dir is accepted ONLY
    after remote-manifest reconciliation (`_local_stage_covers_remote`) — a
    partial/stale local copy falls through to the HF mirror instead of being
    silently consumed (r2 review concern
    local-raw-stage-completeness-unchecked). The mirror restage first REPAIRS
    the mirror leaf (`_repair_stale_mirror` — delete-then-restage, because the
    hub staging skips existing targets and would serve stale bytes; r3
    reconciler concern rows-dir-mismatch-fallthrough-stale-mirror) and then
    fail-loud verifies the restaged leaf against the same manifest. The
    no-flag path (same-pod producer reads; offline probes) stays network-free
    by design — producer completeness there is owned by the StageLedger
    resume + the consumers' fail-loud empty-selection guards."""
    local = Path(explicit) if explicit else Path(args.raw_root) / stage
    if any(local.glob("*.jsonl")):
        if not args.stage_raw_from_hf or _local_stage_covers_remote(local, stage):
            return local
    if args.stage_raw_from_hf:
        if stage in _STAGE_MIRROR_CACHE:
            return _STAGE_MIRROR_CACHE[stage]
        prefix = f"{cm.HF_PREFIX}/raw_completions/{stage}"
        remote = _stage_remote_manifest(stage)
        deleted = _repair_stale_mirror(HF_STAGE_ROOT / prefix, remote)
        if deleted:
            print(
                f"[stage] {stage}: deleted {len(deleted)} stale mirror file(s) before "
                f"restage (skip-existing would have served them): {deleted[:3]}",
                flush=True,
            )
        leaf = cm.stage_hf_prefix(prefix, HF_STAGE_ROOT)
        still = [
            n
            for n, s in sorted(remote.items())
            if not (leaf / n).is_file() or (s is not None and (leaf / n).stat().st_size != int(s))
        ]
        if still:
            raise RuntimeError(
                f"stage {stage}: mirror leaf {leaf} STILL fails remote-manifest "
                f"reconciliation after restage ({len(still)} mismatched; first: {still[:3]})"
            )
        _STAGE_MIRROR_CACHE[stage] = leaf
        return leaf
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


def _classify_segb_row(row: dict, text: str) -> None:
    """Set gen_text/answer/keep/drop_reason on a SegB reply row (shared by
    the 1x pass and the 2x cell-grain no-close regen)."""
    close = _mine_closing_quote(text)
    keep, drop_reason, answer = True, None, None
    if close is None:
        keep, drop_reason = False, "cap_hit_no_close"
    else:
        answer = text[:close].strip()
        if not answer:
            keep, drop_reason = False, "empty_answer"
    row.update(
        {
            "gen_text": text,
            "answer": answer,
            "answer_close_idx": close,
            "keep": keep,
            "drop_reason": drop_reason,
        }
    )


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
            out_rows = []
            for rid, opener, out, s in zip(chunk_ids, openers, outs, seeds):
                text, finish = _gen_text(out)
                row = {
                    "cell": cell,
                    "row_id": rid,
                    "wave": wave,
                    "opener_id": mined[rid]["opener_id"],
                    "opener_text": opener,
                    "finish_reason": finish,
                    "seed": s,
                    "regen": False,
                }
                _classify_segb_row(row, text)
                out_rows.append(row)
            stem = f"{cell}_w{wave}_s{args.shard_index}_c{ci:04d}"
            _write_chunk_jsonl(raw_root / "segb" / f"{stem}.jsonl", out_rows)
            ledger.mark_done(key)
            cm.progress("segb", ci + 1, n_chunks, key, t0)
        # Cell-grain no-close regen + durable-file summary (r1 majors 13+15).
        # Cap-hit for SegB = no closing quote within the cap (plan §4.2).
        files = sorted((raw_root / "segb").glob(f"{cell}_w{wave}_s{args.shard_index}_c*.jsonl"))

        def _rebuild(row: dict) -> tuple[str, int, list[str] | None]:
            return _segb_prompt(mined[row["row_id"]], banksd)[0], 2 * cm.SEGB_MAX_TOKENS, None

        def _update(row: dict, text: str, finish: str) -> None:
            row["finish_reason"] = finish
            _classify_segb_row(row, text)

        decision = _cell_grain_regen(
            llm,
            files,
            raw_root / "segb" / f"regen_decision_{cell}_w{wave}_s{args.shard_index}.json",
            is_hit=lambda r: r.get("answer_close_idx") is None,
            rebuild=_rebuild,
            update_row=_update,
            tag=f"segb/{cell}",
        )
        counts = {
            "rows": 0,
            "kept": 0,
            "cap_hit_no_close": 0,
            "empty_answer": 0,
            "non_english_answer_flag": 0,
        }
        for path in files:
            for row in cm.iter_jsonl(path):
                counts["rows"] += 1
                if row.get("keep"):
                    counts["kept"] += 1
                    if row.get("answer") and not cm.english_majority(row["answer"]):
                        counts["non_english_answer_flag"] += 1  # report-only, story cells
                elif row.get("drop_reason") in counts:
                    counts[row["drop_reason"]] += 1
        summary = {
            "cell": cell,  # r10: consumed by dispatch._sum_stage_summaries (G1 family pooling)
            "regime": regime,
            "counts": counts,
            "cap_hit_fraction_before": decision["frac_before"],
            "cap_hit_fraction_after": decision["frac_after"],
            "cap_hit_regen": decision["regen"],
            "metadata": cm.run_metadata(),
        }
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
        # r13 (G2b fix 1): fresh PLAIN draws are the same raw-completion leg
        # as the wave-1 plain pass — same '<think>' ban (chat template-path
        # immune; story cells stay deliberately unbanned, r11 L1 scope).
        ban = ["<think>"] if cell == "plain_text" else None
        regime = {
            "phase": "fresh_draws",
            "cell": cell,
            "n": args.fresh_rows,
            "draw_seeds": draw_seeds,
            "shard": [args.shard_index, args.num_shards],
            "model": cm.MODEL_ID,
        }
        if ban:
            regime["bad_words"] = ban
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
            sps = [_sampling_params(cap, stop, s, bad_words=ban) for s in seeds]
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
                    "regen": False,
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
        # Cell-grain cap-hit regen + durable summary (r1 review majors 13+15:
        # fresh draws previously skipped the > 2% rule and wrote no summary).
        files = [
            raw_root / "fresh_draws" / f"{cell}_d{d}_s{args.shard_index}.jsonl" for d in draw_seeds
        ]
        story = cell in cm.STORY_CELLS

        def _rebuild(row: dict) -> tuple[str, int, list[str] | None]:
            if story:
                assert mined is not None
                return _segb_prompt(mined[row["row_id"]], banksd)[0], 2 * cm.SEGB_MAX_TOKENS, None
            if cell == "chat":
                q = base_rows[row["row_id"]]["question"]
                return _render_chat(tok, q), 2 * cm.CHAT_MAX_TOKENS, cm.CHAT_STOP
            q = base_rows[row["row_id"]]["question"]
            return f"User: {q}\n\nAssistant:", 2 * cm.PLAIN_MAX_TOKENS, cm.PLAIN_STOP

        def _update(row: dict, text: str, finish: str) -> None:
            row["gen_text"] = text
            row["finish_reason"] = finish
            if story:
                close = _mine_closing_quote(text)
                row["answer"] = text[:close].strip() if close is not None else None
                row["answer_close_idx"] = close
            else:
                row["answer"] = text.strip()

        def _is_hit(row: dict) -> bool:
            if story:
                return row.get("answer_close_idx") is None
            return row.get("finish_reason") == "length"

        decision = _cell_grain_regen(
            llm,
            files,
            raw_root / "fresh_draws" / f"regen_decision_{cell}_s{args.shard_index}.json",
            is_hit=_is_hit,
            rebuild=_rebuild,
            update_row=_update,
            tag=f"fresh/{cell}",
            bad_words=ban,
        )
        counts = {"rows": 0, "cap_hit": 0}
        for path in files:
            if not path.exists():
                continue
            for row in cm.iter_jsonl(path):
                counts["rows"] += 1
                if _is_hit(row):
                    counts["cap_hit"] += 1
        summary = {
            "cell": cell,  # r10: per-cell key for _sum_stage_summaries-style aggregation
            "regime": regime,
            "counts": counts,
            "cap_hit_fraction_before": decision["frac_before"],
            "cap_hit_fraction_after": decision["frac_after"],
            "cap_hit_regen": decision["regen"],
            "metadata": cm.run_metadata(),
        }
        cm.atomic_write_json(
            raw_root / "fresh_draws" / f"summary_{cell}_s{args.shard_index}.json", summary
        )
        print(f"[fresh_draws] {cell}: {json.dumps(counts)}", flush=True)
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


def phase_probe_miner(args) -> None:
    """CPU-only miner probe (r1 review blocker: the `_is_directed`
    speaker-exclusion rejected the canonical taught reply shape). Synthetic
    fixtures only — no corpus text, no model. Exits non-zero on any mismatch:
    the canonical/opener-taught KEEP shapes must mine, speaker/inverted
    attributions must reject."""
    _ = args
    name = "Astra"
    # G1 recalibration (r11, L3): stop-at-first-DIRECTED-quote. A directed quote
    # failing the family kind check must REJECT the row (wrong_kind) — the
    # pre-r11 forward scan salvaged the character's own reply after skipping a
    # wrong-kind directed quote (21/102 dialogue admission rejects were
    # reply-mined; epm:progress v65 evidence item 4).
    salvage_dialogue = (
        'Dana turned to Astra. "Will the bridge hold tonight?" There was a long '
        'pause before the answer came. "You worry too much, Dana."'
    )
    salvage_question = (
        'Dana leaned toward Astra. "The harvest survived the frost." A heartbeat '
        'later the real question arrived. "But will you sell it all, Astra?"'
    )
    fixtures = [
        (
            "question",
            'The lanterns flickered. "Where does the river bend?" '
            'Astra replied: "Past the mill, where the willows lean."',
            True,
            "canonical prime/opener-taught reply shape",
        ),
        (
            "question",
            'Dana leaned across the table and asked Astra: "Did the harvest survive the frost?"',
            True,
            "directed-TO variant (asked {Name}:)",
        ),
        (
            "question",
            'Dana turned to Astra. "Will the bridge hold?"',
            True,
            "to-Name proximity variant",
        ),
        (
            "dialogue",
            '"The bridge held through the storm." Astra nodded, then said: "It always does."',
            True,
            "dialogue-family reply shape",
        ),
        (
            "question",
            'Astra said: "Where does the river bend?"',
            False,
            "speaker-before-quote ({name} said:)",
        ),
        (
            "question",
            '"Where does the river bend?" asked Astra.',
            False,
            "inverted post-quote speaker",
        ),
        (
            "question",
            '"Where does the river bend?" Astra said.',
            False,
            "post-quote speaker with no follow-on quote",
        ),
        (
            "dialogue",
            'Dana turned to Astra. "Will the bridge hold?"',
            False,
            "wrong kind: question under dialogue family",
        ),
        (
            "dialogue",
            salvage_dialogue,
            False,
            "L3 no-salvage: wrong-kind directed question then reply (dialogue family)",
        ),
        (
            "question",
            salvage_question,
            False,
            "L3 no-salvage: wrong-kind directed statement then question (question family)",
        ),
    ]
    failures = []
    for family, text, expect, note in fixtures:
        verdict = _mine_sega(text, name, family, len(text))
        if bool(verdict["kept"]) != expect:
            failures.append(f"{note}: expected kept={expect}, got {json.dumps(verdict)}")
        elif verdict["kept"]:
            utter = text[verdict["utter_start"] : verdict["utter_end"]]
            if not utter.strip():
                failures.append(f"{note}: empty mined utterance")
    # L3 reason check: the salvage fixtures must reject as wrong_kind (the
    # directed first quote's verdict), never as a downstream reason.
    for family, text in (("dialogue", salvage_dialogue), ("question", salvage_question)):
        verdict = _mine_sega(text, name, family, len(text))
        if verdict.get("reason") != "wrong_kind":
            failures.append(f"L3 {family}: expected reason=wrong_kind, got {json.dumps(verdict)}")
    # G1 recalibration (r11, L2): window semantics — the miner window now covers
    # the full SegA generation (MINER_WINDOW_TOKENS == SEGA_MAX_TOKENS); a
    # truncated window still rejects quote-free prefixes.
    wtext = 'Dana turned to Astra. "Will the bridge hold tonight?"'
    wfull = _mine_sega(wtext, name, "question", len(wtext))
    wcut = _mine_sega(wtext, name, "question", 10)
    if not wfull["kept"]:
        failures.append(f"L2 window: full-window mine expected kept=True, got {json.dumps(wfull)}")
    if wcut["kept"] or wcut.get("reason") != "no_quote_in_window":
        failures.append(
            f"L2 window: truncated window expected no_quote_in_window, got {json.dumps(wcut)}"
        )
    if cm.MINER_WINDOW_TOKENS < cm.SEGA_MAX_TOKENS:
        failures.append(
            f"L2 window: MINER_WINDOW_TOKENS={cm.MINER_WINDOW_TOKENS} below the SegA "
            f"generation cap {cm.SEGA_MAX_TOKENS} (G1 recalibration contract)"
        )
    if failures:
        for f in failures:
            print(f"[probe_miner] FAIL {f}", flush=True)
        raise SystemExit(1)
    print(f"[probe_miner] PASS ({len(fixtures)} fixtures + L2/L3 probes)", flush=True)


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
    "probe_miner": phase_probe_miner,
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
    ap.add_argument(
        "--bank-only",
        default="",
        help="build ONE LLM-authored bank only (bounded real-API smoke of build_banks)",
    )
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
