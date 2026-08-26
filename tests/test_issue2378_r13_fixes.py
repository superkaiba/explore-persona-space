"""Pins for the issue #2378 r13 G2b fixes (P4 wave-1 harvest diagnosis).

Three fixes pinned here (scripts/issue2378_{gen,capture,judge}.py):

1. REAL-u2 span (rig bug — 100% span_mismatch in P4 wave-1): Qwen3.6's chat
   template attaches the empty ``<think>\\n\\n</think>`` block only to
   assistant turns AFTER the last user query, so the 2-turn
   ``gen._render_user_prefix`` render is NOT a prefix of the 3-turn render
   and the r12 ``startswith(prefix + u2)`` producer check failed on every
   row. r13 anchored u2 from the content-independent template TAIL; r14
   (review blockers user-pair-vc-assert-guaranteed-fail /
   user-arm-context-identity-contract-broken) replaced the 3-turn template
   render itself with the DIRECT JOIN ``_render_user_prefix + u2 + TURN_END``
   (``gen._render_user_real_tf``) so both user arms share byte-identical
   context bytes (§4.2b pair contract; declared deviation from template
   fidelity). The producer per-row body is the shared ``gen._user_real_row``
   and the capture consumer ``_assemble_user_real`` re-derives through the
   SAME helpers — the tests below run those real bodies against the REAL
   pinned tokenizer with writer-real pool-row shapes (the #906
   fixture-vs-writer class has bitten this issue three times).

2. '<think>' ban threading (the r11 SegA lever extended): the plain answer
   leg (89% wave-1 think_leak), fresh plain draws, and the sim-user legs
   (2.8% wave-1 think_leak) sample under ``bad_words=["<think>"]`` on BOTH
   the 1x pass and the ``_cell_grain_regen`` 2x pass; chat (template-path
   immune) and segb (r11 scope) stay unbanned, and the chat regime dict
   stays byte-stable so completed wave-1 chat ledgers keep resuming.

3. Wave-2 admission checkpoint partition: api_dispatch's batch state.json
   fingerprints the PENDING (post-cache) item set and fails loud on a load
   mismatch (#1018). A P4 top-up GROWS the admission input, wave-1 rows are
   cache-served, the pending set changes, and a shared
   ``checkpoint_dir/<cache_tag>`` would kill the wave-2 re-run at state
   load. ``judge._checkpoint_partition`` keys the checkpoint dir on the
   realized item-id set (cache partition deliberately UNCHANGED; since r14
   the kept-ledger union EXTEND is enforced mechanically by
   ``judge._merge_kept_ledger`` — pinned in tests/test_issue2378_r14_fixes.py
   — rather than assumed from cache hits).

Real-template tests skip when the pinned tokenizer is unavailable
(no network/cache in CI — the test_issue1482_kresample precedent).
"""

from __future__ import annotations

import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2378_capture as cap  # noqa: E402
import issue2378_common as cm  # noqa: E402
import issue2378_gen as gen  # noqa: E402
import issue2378_judge as judge  # noqa: E402

# Writer-real pool-row shape: the exact keys phase_build_pools writes for the
# user pool (conv_id / u1 / a1 / u2 / depth); multi-line u2 is admissible in
# the real pool (only the CHAT question filter rejects multiline).
POOL_ROW = {
    "conv_id": "mt_r13pin000001",
    "u1": "Can you explain what a mutex is in one paragraph?",
    "a1": "A mutex is a lock that lets only one thread enter a critical "
    "section at a time, so shared state cannot be mutated concurrently.",
    "u2": "Thanks! And how does that differ from a semaphore?\nA short example would help.",
    "depth": 2,
}


def _tok():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(cm.MODEL_ID)
    except Exception as e:  # no network/cache in CI
        pytest.skip(f"tokenizer unavailable: {e}")


# ── fix 1: real-u2 tail anchor ───────────────────────────────────────────────


def test_two_turn_prefix_is_not_full_render_prefix():
    """Root cause, pinned on the REAL template: the template's own 3-turn
    render is structurally inconsistent with the 2-turn prefix (a1's think
    block renders in the prefix and is stripped once u2 follows) — which is
    why r14's real arm teacher-forces the DIRECT JOIN instead. If the
    template fact ever flips, the declared deviation should be revisited."""
    tok = _tok()
    r = POOL_ROW
    prefix = gen._render_user_prefix(tok, r["u1"], r["a1"])
    template_full = tok.apply_chat_template(
        [
            {"role": "user", "content": r["u1"]},
            {"role": "assistant", "content": r["a1"]},
            {"role": "user", "content": r["u2"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    assert "<think>\n\n</think>" in prefix
    assert "<think>" not in template_full  # the template strips a1's block
    assert not template_full.startswith(prefix + r["u2"])  # r13's broken pairing
    # r14: the direct-join teacher-forced render IS prefix-consistent by
    # construction — byte-identical context prefix across the two user arms.
    tf = gen._render_user_real_tf(tok, r["u1"], r["a1"], r["u2"])
    assert tf == prefix + r["u2"] + cm.TURN_END
    assert tf.startswith(prefix + r["u2"])


def test_user_real_row_keeps_and_span_slices_u2():
    tok = _tok()
    row = gen._user_real_row(tok, POOL_ROW)
    assert row["keep"] is True and row["drop_reason"] is None
    lo, hi = row["u2_span"]
    assert row["header_end"] == lo
    assert row["rendered_text"][lo:hi] == POOL_ROW["u2"]
    # The tail is template-constructed: header immediately before u2, turn
    # end immediately after.
    assert row["rendered_text"][lo - len(cm.USER_TURN_HEADER) : lo] == cm.USER_TURN_HEADER
    assert row["rendered_text"][hi:] == cm.TURN_END


def test_user_real_span_tail_mismatch_drops():
    tok = _tok()
    full = gen._render_user_real_tf(tok, POOL_ROW["u1"], POOL_ROW["a1"], POOL_ROW["u2"])
    assert gen._user_real_span(full, POOL_ROW["u2"] + "DRIFT") is None
    row = gen._user_real_row(tok, {**POOL_ROW, "u2": POOL_ROW["u2"]})
    assert row["keep"] is True  # sanity: the undrifted row still keeps


def test_capture_assembler_round_trips_producer_row():
    tok = _tok()
    row = gen._user_real_row(tok, POOL_ROW)
    payload, reason = cap._assemble_user_real(tok, row, POOL_ROW)
    assert reason is None, reason
    lo, hi = row["u2_span"]
    assert payload["answer_lo_char"] == lo and payload["answer_hi_char"] == hi
    assert payload["final_text"] == row["rendered_text"][:hi]
    pos = payload["prefix_char"]
    a1 = POOL_ROW["a1"]
    assert payload["final_text"][pos : pos + len(a1)] == a1


def test_capture_assembler_rejects_tampered_render():
    tok = _tok()
    row = gen._user_real_row(tok, POOL_ROW)
    tampered = {**row, "rendered_text": row["rendered_text"] + "X"}
    payload, reason = cap._assemble_user_real(tok, tampered, POOL_ROW)
    assert payload is None and reason == "prefix_render_mismatch"


# ── fix 2: '<think>' ban threading ───────────────────────────────────────────


@dataclass
class _Gen:
    text: str
    finish_reason: str


@dataclass
class _Out:
    outputs: list


class _FakeLLM:
    """GPU-boundary fake mirroring the called shape of vllm.LLM.generate
    (prompts, sampling_params, use_tqdm) — signature-conformant by
    construction (#906); everything else in the path runs the real bodies,
    including the real vllm SamplingParams constructor."""

    def __init__(self, text: str = "A perfectly ordinary answer."):
        self.calls: list[tuple[list, list]] = []
        self._text = text

    def generate(self, prompts, sampling_params, use_tqdm=True):
        self.calls.append((list(prompts), list(sampling_params)))
        return [_Out([_Gen(self._text, "stop")]) for _ in prompts]


def _gen_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        raw_root=str(tmp_path / "raw"),
        wave=1,
        shard_index=0,
        num_shards=1,
        chunk_rows=8,
        max_model_len=8192,
    )


def test_run_answer_cell_plain_bans_think_chat_does_not(tmp_path):
    pytest.importorskip("vllm")
    tok = _tok()
    rows = [{"conv_id": f"c{i}", "question": f"What is {i} plus {i}?"} for i in range(3)]
    for cell, want_ban in (("plain_text", ["<think>"]), ("chat", None)):
        llm = _FakeLLM()
        gen._run_answer_cell(_gen_args(tmp_path), llm, tok, cell, rows, "sha16sha16sha16s")
        assert llm.calls, f"no generate call for {cell}"
        sps = llm.calls[0][1]
        # vllm normalizes bad_words=None -> [] at construction.
        assert all((sp.bad_words or None) == want_ban for sp in sps), cell
        stage = "chat" if cell == "chat" else "plain"
        ledger = json.loads((tmp_path / "raw" / stage / f"ledger_{cell}_w1_s0.json").read_text())
        if want_ban:
            assert ledger["regime"]["bad_words"] == want_ban
        else:
            # chat regime stays byte-stable so wave-1 chat ledgers resume.
            assert "bad_words" not in ledger["regime"]


def test_run_user_sim_bans_think_and_pins_regime(tmp_path):
    pytest.importorskip("vllm")
    tok = _tok()
    rows = [{"conv_id": "c0", "u1": POOL_ROW["u1"], "a1": POOL_ROW["a1"], "u2": POOL_ROW["u2"]}]
    llm = _FakeLLM(text="Could you also compare it with a spinlock, please?")
    gen._run_user_sim(_gen_args(tmp_path), llm, tok, rows, "user_sim", [cm.SEED])
    assert llm.calls
    sps = llm.calls[0][1]
    assert all((sp.bad_words or None) == ["<think>"] for sp in sps)
    ledger = json.loads((tmp_path / "raw" / "user_sim" / "ledger_w1_s0.json").read_text())
    assert ledger["regime"]["bad_words"] == ["<think>"]
    out = list(cm.iter_jsonl(tmp_path / "raw" / "user_sim" / f"w1_d{cm.SEED}_s0_c0000.jsonl"))
    assert out and out[0]["keep"] is True


def test_cell_grain_regen_threads_bad_words(tmp_path):
    pytest.importorskip("vllm")
    chunk = tmp_path / "c0000.jsonl"
    row = {
        "cell": "plain_text",
        "conv_id": "c0",
        "question": "q?",
        "finish_reason": "length",
        "seed": 7,
        "regen": False,
        "answer": "partial",
        "keep": True,
        "drop_reason": None,
    }
    chunk.write_text(json.dumps(row) + "\n", encoding="utf-8")
    llm = _FakeLLM(text="Regenerated answer.")
    decision = gen._cell_grain_regen(
        llm,
        [chunk],
        tmp_path / "decision.json",
        is_hit=lambda r: r.get("finish_reason") == "length",
        rebuild=lambda r: ("User: q?\n\nAssistant:", 64, cm.PLAIN_STOP),
        update_row=lambda r, text, finish: r.update({"answer": text, "finish_reason": finish}),
        tag="t",
        bad_words=["<think>"],
    )
    assert decision["regen"] is True and decision["done"] is True
    assert llm.calls and llm.calls[0][1][0].bad_words == ["<think>"]
    rewritten = json.loads(chunk.read_text().strip())
    assert rewritten["regen"] is True and rewritten["answer"] == "Regenerated answer."


def test_think_ban_wiring_source_pins():
    """The 1x-pass wiring the functional tests cannot cheaply execute
    (user_sim / fresh_draws) is pinned at source level (the repo's
    DISPATCH_SRC precedent, tests/test_issue2378_model_venv.py)."""
    sim_src = inspect.getsource(gen._run_user_sim)
    assert 'ban = ["<think>"]' in sim_src
    assert "bad_words=ban" in sim_src  # 1x pass AND the regen call thread it
    assert '"bad_words": ban' in sim_src  # regime pin
    fresh_src = inspect.getsource(gen.phase_fresh_draws)
    assert 'ban = ["<think>"] if cell == "plain_text" else None' in fresh_src
    assert fresh_src.count("bad_words=ban") >= 2  # 1x pass + regen call
    answer_src = inspect.getsource(gen._run_answer_cell)
    assert 'ban = None if cell == "chat" else ["<think>"]' in answer_src
    assert answer_src.count("bad_words=ban") >= 2


# ── fix 3: wave-2 admission checkpoint partition ─────────────────────────────


def _items(ids):
    from explore_persona_space.llm.api_dispatch import DispatchItem

    return [DispatchItem(item_id=i, payload={"row_id": i}) for i in ids]


def test_checkpoint_partition_keys_on_item_set():
    same_a = judge._checkpoint_partition("admission", _items(["adm|a", "adm|b"]))
    same_b = judge._checkpoint_partition("admission", _items(["adm|b", "adm|a"]))
    assert same_a == same_b  # order-insensitive: a same-set resume maps home
    grown = judge._checkpoint_partition("admission", _items(["adm|a", "adm|b", "adm|c"]))
    assert grown != same_a  # a topup-grown wave gets a FRESH checkpoint dir
    assert same_a.startswith("admission_") and grown.startswith("admission_")


def test_dispatch_checkpoint_dir_item_set_keyed_cache_dir_not():
    src = inspect.getsource(judge._dispatch)
    assert "_checkpoint_partition(cache_tag, items)" in src
    # The CACHE partition stays cache_tag alone — cache hits keep a warm-cache
    # wave-2 re-run cheap. Since r14 (B3) the kept-ledger EXTEND is enforced
    # mechanically in judge._merge_kept_ledger, never assumed from the cache
    # (tests/test_issue2378_r14_fixes.py pins it).
    assert "Path(args.cache_dir) / cache_tag" in src
