"""Pins for the #2378 sim-user-regen round (#612 elicitation repair).

Wave-1 diagnosis (raw-shard measured): the assistant-SFT'd model's most
likely continuation at the bare ``<|im_start|>user\\n`` header is the turn
terminator — 6,402/10,000 EMPTY sim user turns, 521 kept. The repair:

- rung 1 = the SAME bare prefill under a mechanical non-degeneracy
  constraint (``min_tokens`` masks EOS + ``stop_token_ids`` logits for the
  first N steps; the special turn-terminators move into token-id space
  because vLLM does NOT mask stop STRINGS) + <=2 fresh-seed re-elicitation
  passes over retryable degenerates;
- rung 3 (#612 minimal opener prefill; production ``user_sim`` stage only)
  fires ONLY when the kept quota is unfilled after the retries; the opener
  stays in the MEASURED turn while ``prefix_chars``/``prefix_digest`` stay
  pinned to the BASE shared prefix (§4.2b pair contract);
- capture reuses the real arm's stored v_C/v_P per conversation
  (``--user-vcvp-from-store``) so ``p6.assert_user_pair``'s per-conversation
  sha identity holds by construction (bf16 batched kernels are
  batch-composition dependent — recomputation would break it).

Per code-style "one production-body test per seam-stubbed function": the
e2e test fakes ONLY the vLLM engine boundary (a ``generate`` def mirroring
the used call shape); ``_run_user_sim``, ``_sim_repair_pass``,
``_cell_grain_regen``, ``_classify_sim_row``, ``_user_stop_token_ids``,
``_sim_prompt`` and the summary accounting all execute their real bodies.
Real-template tests skip when the pinned tokenizer is unavailable (no
network/cache in CI — the test_issue2378_r14_fixes convention).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2378_capture as cap  # noqa: E402
import issue2378_common as cm  # noqa: E402
import issue2378_dispatch as dsp  # noqa: E402
import issue2378_gen as gen  # noqa: E402
import issue2378_lenmatch as lm  # noqa: E402


def _tok():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(cm.MODEL_ID)
    except Exception as e:  # no network/cache in CI
        pytest.skip(f"tokenizer unavailable: {e}")


def _pool_row(conv: str, marker: str) -> dict:
    return {
        "conv_id": conv,
        "u1": f"Can you explain {marker} in one short paragraph?",
        "a1": f"{marker} is a synchronization mechanism; only one thread may hold it at a time.",
        "u2": "Thanks! And how does that differ from a semaphore?",
    }


class _FakeOut:
    def __init__(self, text: str, finish: str = "stop"):
        self.outputs = [SimpleNamespace(text=text, finish_reason=finish)]


class _FakeLLM:
    """vLLM boundary fake mirroring the used ``generate`` call shape
    (prompts, sampling_params, use_tqdm=...) — never a bare Mock (#906)."""

    def __init__(self, markers: dict[str, str], bare: dict[str, list[str]], opened: dict[str, str]):
        self.markers = markers  # conv -> unique u1 marker substring
        self.bare = {k: list(v) for k, v in bare.items()}  # FIFO per conv
        self.opened = opened
        self.captured_sps: list[list] = []

    def _conv_of(self, prompt: str) -> str:
        hits = [c for c, m in self.markers.items() if m in prompt]
        assert len(hits) == 1, (hits, prompt[:120])
        return hits[0]

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert use_tqdm is False
        self.captured_sps.append(list(sampling_params))
        outs = []
        for p in prompts:
            conv = self._conv_of(p)
            if any(p.endswith(op) for op in cm.USER_SIM_RUNG3_OPENERS):
                outs.append(_FakeOut(self.opened[conv]))
            else:
                outs.append(_FakeOut(self.bare[conv].pop(0)))
        return outs


def test_user_sim_e2e_repair_ladder(tmp_path):
    """Full ``_run_user_sim`` body on 4 conversations: rung-1 keep, rung-1
    retry recovery, rung-3 opener keep, and a fully-degenerate reported drop —
    with the mechanical constraint threaded into every SamplingParams."""
    pytest.importorskip("vllm")
    tok = _tok()
    markers = {"A": "a mutex", "B": "a spinlock", "C": "a futex", "D": "a rwlock"}
    rows = [_pool_row(c, m) for c, m in markers.items()]
    good = {
        "A": "Could you give a short worked example in Python, please?",
        "B": "Thanks — and how would contention change the picture here?",
    }
    llm = _FakeLLM(
        markers,
        bare={
            "A": [good["A"]],
            "B": ["", good["B"]],
            "C": ["", "", ""],
            "D": ["", "", ""],
        },
        opened={"C": "what should I watch out for with lock-free queues?", "D": ""},
    )
    args = SimpleNamespace(
        raw_root=str(tmp_path),
        wave=1,
        shard_index=0,
        num_shards=1,
        chunk_rows=512,
        max_model_len=8192,
    )
    gen._run_user_sim(args, llm, tok, rows, "user_sim", [cm.SEED])

    # SamplingParams threading: every pass carries the rung-1 constraint.
    stop_ids = gen._user_stop_token_ids(tok)
    for sps in llm.captured_sps:
        for sp in sps:
            assert sp.min_tokens == cm.USER_SIM_MIN_TOKENS, sp
            assert list(sp.stop_token_ids) == stop_ids, sp
            assert sp.bad_words == ["<think>"], sp

    by_conv = {}
    for p in sorted((tmp_path / "user_sim").glob("w1_d*_s0_c*.jsonl")):
        for r in cm.iter_jsonl(p):
            by_conv[r["conv_id"]] = r
    assert set(by_conv) == set(markers)
    # A: rung-1 first-pass keep.
    assert by_conv["A"]["keep"] and by_conv["A"]["elicit_rung"] == 1
    assert by_conv["A"]["retry_pass"] == 0
    # B: rung-1 retry recovery (pass 1), prior drop recorded.
    assert by_conv["B"]["keep"] and by_conv["B"]["elicit_rung"] == 1
    assert by_conv["B"]["retry_pass"] == 1
    assert by_conv["B"]["prior_drop_reasons"] == ["empty_turn"]
    assert by_conv["B"]["sim_turn"] == good["B"]
    # C: rung-3 keep — opener IS in the measured turn; prefix fields stay
    # pinned to the BASE shared prefix (§4.2b: the opener is measured text,
    # never context).
    c = by_conv["C"]
    assert c["keep"] and c["elicit_rung"] == 3
    opener = gen._rung3_opener("C")
    assert c["opener_text"] == opener and c["sim_turn"].startswith(opener.strip())
    base_prefix = gen._render_user_prefix(tok, rows[2]["u1"], rows[2]["a1"])
    assert c["prefix_chars"] == len(base_prefix)
    assert c["prefix_digest"] == cm.text_digest(base_prefix)
    # D: exhausted the whole ladder — reported drop, never backfilled (#612).
    # Final classification is len_band, not empty_turn: the rung-3 measured
    # turn is opener + "" (the opener alone sits under the 16-char band).
    d = by_conv["D"]
    assert not d["keep"] and d["drop_reason"] == "len_band"
    assert d["elicit_rung"] == 3 and d["retry_pass"] == 2
    assert d["prior_drop_reasons"] == ["empty_turn"] * 3

    summary = json.loads((tmp_path / "user_sim" / "summary_w1_s0.json").read_text())
    el = summary["elicitation"]
    assert el["kept_rung1"] == 2 and el["kept_rung3"] == 1
    assert el["rung1_retry_recovered"] == 1
    assert el["rung3"]["fired"] is True
    # quota = ceil(FLOOR_KEPT * n_shard / USER_DRAW_N) = ceil(6500*4/10000) = 3
    assert el["rung3"]["quota"] == 3 and el["rung3"]["kept_before"] == 2
    assert summary["counts"]["kept"] == 3 and summary["counts"]["len_band"] == 1
    assert summary["regime"]["min_tokens"] == cm.USER_SIM_MIN_TOKENS
    assert summary["regime"]["stop_token_ids"] == stop_ids
    assert summary["regime"]["retry_passes"] == cm.USER_SIM_RETRY_PASSES


def test_user_sim_repair_pass_idempotent(tmp_path):
    """A completed repair pass (done decision) is never re-run on resume."""
    pytest.importorskip("vllm")
    decision_path = tmp_path / "retry_decision.json"
    cm.atomic_write_json(decision_path, {"done": True, "n_selected": 5, "n_recovered": 2})

    class _Boom:
        def generate(self, prompts, sampling_params, use_tqdm=False):
            raise AssertionError("resume must not re-elicit a completed pass")

    out = gen._sim_repair_pass(
        SimpleNamespace(),
        _Boom(),
        None,
        {},
        [],
        decision_path,
        select=lambda r: True,
        mark=lambda r: None,
        seed_parts=("retry", 1),
        cap=64,
        ban=["<think>"],
        sp_extra={},
        tag="t",
    )
    assert out["done"] and out["n_recovered"] == 2


def test_sampling_params_threading():
    pytest.importorskip("vllm")
    sp = gen._sampling_params(
        64, cm.USER_SIM_STOP, 7, bad_words=["<think>"], min_tokens=8, stop_token_ids=[5, 9]
    )
    assert sp.min_tokens == 8
    assert list(sp.stop_token_ids) == [5, 9]
    assert sp.stop == cm.USER_SIM_STOP  # stop STRINGS retained alongside the ids


def test_user_stop_token_ids_fail_loud():
    """A multi-token stop must raise (a silent non-binding constraint is the
    wave-1 collapse shape); the real template's terminators are single-token."""
    tok = _tok()
    ids = gen._user_stop_token_ids(tok)
    assert len(ids) == len(cm.USER_SIM_STOP)

    class _MultiTok:
        def encode(self, s, add_special_tokens=False):
            return [1, 2]

    with pytest.raises(RuntimeError, match="not a single token"):
        gen._user_stop_token_ids(_MultiTok())


def test_capture_ready_cells_subset(tmp_path):
    """--cells restricts emission (round raw roots hold ONLY the user stages);
    single-user selection is refused; the pair intersection is joint."""
    raw = tmp_path / "raw"
    led = tmp_path / "led"

    def _write(stage: str, cell: str, ids: list[str]) -> None:
        (raw / stage).mkdir(parents=True, exist_ok=True)
        with (raw / stage / "w1_d137_s0_c0000.jsonl").open("w") as fh:
            for rid in ids:
                fh.write(json.dumps({"cell": cell, "conv_id": rid, "keep": True}) + "\n")

    _write("user_sim", "chat_user_sim", ["a", "b", "c"])
    _write("user_real_render", "chat_user_real", ["b", "c", "e"])

    def ns(cells: str) -> SimpleNamespace:
        return SimpleNamespace(
            cells=cells,
            ledger_root=str(led),
            kept_dir=str(tmp_path / "kept"),
            raw_root=str(raw),
            stage_raw_from_hf=False,
            mined_dir=None,
        )

    with pytest.raises(SystemExit, match="jointly"):
        gen.phase_capture_ready(ns("chat_user_sim"))
    with pytest.raises(SystemExit, match="unknown cells"):
        gen.phase_capture_ready(ns("chat_user_sim,not_a_cell"))
    gen.phase_capture_ready(ns("chat_user_real,chat_user_sim"))
    emitted = sorted(p.name for p in (led / "capture_ready").glob("*.json"))
    assert emitted == ["chat_user_real.json", "chat_user_sim.json"]  # story/chat NOT reduced
    sim = json.loads((led / "capture_ready" / "chat_user_sim.json").read_text())
    assert sim["n_kept"] == 3 and sim["floor_pass"] is False
    assert sim["pair_intersection"]["intersection_ids"] == ["b", "c"]


def _mk_real_part(store: Path, part: int, ids: list[str], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    v = {
        s: rng.integers(0, 2**16, size=(len(ids), 4), dtype=np.uint16)
        for s in ("v_C", "v_A", "v_P")
    }
    cap._atomic_savez(
        store / f"chat_user_real__part{part:04d}__L51.npz",
        v_C=v["v_C"],
        v_A=v["v_A"],
        v_P=v["v_P"],
        row_ids=np.array(ids),
        meta=np.array(json.dumps({"encoding": "bf16_as_uint16", "cell": "chat_user_real"})),
    )
    return v


def test_load_and_apply_vcvp(tmp_path):
    """§4.2b vcvp reuse: stored real-arm rows replace the recomputed sim rows
    byte-for-byte; a conversation missing from the real store fails loud."""
    store = tmp_path / "store"
    store.mkdir()
    v0 = _mk_real_part(store, 0, ["r0", "r1"], seed=1)
    v1 = _mk_real_part(store, 1, ["r2"], seed=2)
    sub = cap._load_user_vcvp(store, [51])
    assert set(sub[51]) == {"r0", "r1", "r2"}

    rng = np.random.default_rng(3)
    arrays = {
        51: {s: rng.integers(0, 2**16, size=(2, 4), dtype=np.uint16) for s in ("v_C", "v_A", "v_P")}
    }
    v_a_before = arrays[51]["v_A"].copy()
    diffs = cap._apply_vcvp_sub(arrays, ["r1", "r2"], [51], sub)
    np.testing.assert_array_equal(arrays[51]["v_C"][0], v0["v_C"][1])
    np.testing.assert_array_equal(arrays[51]["v_C"][1], v1["v_C"][0])
    np.testing.assert_array_equal(arrays[51]["v_P"][0], v0["v_P"][1])
    np.testing.assert_array_equal(arrays[51]["v_A"], v_a_before)  # answers NEVER substituted
    assert diffs["51"] > 0.0  # telemetry on the replaced batch-geometry numerics

    with pytest.raises(RuntimeError, match="missing from the"):
        cap._apply_vcvp_sub(arrays, ["r1", "rX"], [51], sub)
    with pytest.raises(RuntimeError, match="no chat_user_real parts"):
        cap._load_user_vcvp(store, [43])


def test_vcvp_missing_store_and_bad_encoding(tmp_path):
    store = tmp_path / "empty"
    store.mkdir()
    with pytest.raises(RuntimeError, match="no chat_user_real parts"):
        cap._load_user_vcvp(store, [51])
    cap._atomic_savez(
        store / "chat_user_real__part0000__L51.npz",
        v_C=np.zeros((1, 4), dtype=np.uint16),
        v_A=np.zeros((1, 4), dtype=np.uint16),
        v_P=np.zeros((1, 4), dtype=np.uint16),
        row_ids=np.array(["r0"]),
        meta=np.array(json.dumps({"encoding": "float32"})),
    )
    with pytest.raises(RuntimeError, match="unexpected encoding"):
        cap._load_user_vcvp(store, [51])


def test_ks_2samp():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    same = lm._ks_2samp(a, a.copy())
    assert same["statistic"] == 0.0 and same["pvalue_asymptotic"] == 1.0
    disjoint = lm._ks_2samp(np.arange(100), np.arange(100) + 1000)
    assert disjoint["statistic"] == 1.0 and disjoint["pvalue_asymptotic"] < 1e-6
    shifted = lm._ks_2samp(a, rng.normal(loc=0.75, size=400))
    assert 0.0 < shifted["statistic"] < 1.0
    assert shifted["pvalue_asymptotic"] < 0.01  # 0.75 sigma shift at n=400 is detectable


def test_simregen_floor_verdict_branches():
    def cr(n_sim: int, n_int: int) -> dict:
        return {"n_kept": n_sim, "pair_intersection": {"n_intersection": n_int}}

    assert dsp._simregen_floor_verdict(cr(8200, 7100))["verdict"] == "PASS"
    v = dsp._simregen_floor_verdict(cr(8200, 6100))  # intersection binds alone
    assert v["verdict"] == "FLOOR_FAIL" and v["sim_floor_pass"]
    v = dsp._simregen_floor_verdict(cr(6000, 5900))
    assert v["verdict"] == "FLOOR_FAIL" and v["close_miss_band"]
    assert dsp._simregen_floor_verdict({})["verdict"] == "FLOOR_FAIL"


def test_simregen_plan_named_sync(tmp_path):
    import issue2378_p6_common as p6

    rl, pl = tmp_path / "round", tmp_path / "parent"
    for arm in p6.ARMS:
        cm.atomic_write_json(
            rl / "fits" / f"chat_user_sim__{arm}.json", {"arm": arm, "pooled_r2": 0.2}
        )
    cm.atomic_write_json(rl / "ladder" / "h4b_real_vs_sim.json", {"status": "ok"})
    cm.atomic_write_json(
        pl / "fits" / "chat_user_sim__g2b_dropped.json", {"cell": "chat_user_sim", "status": "N/A"}
    )
    written = dsp._simregen_plan_named_sync(rl, pl)
    assert len(written) == 4
    for arm in p6.ARMS:
        d = json.loads((pl / "fits" / f"chat_user_sim__{arm}.json").read_text())
        assert d["round"] == cm.SIMREGEN_ROUND and d["pooled_r2"] == 0.2
    drop = json.loads((pl / "fits" / "chat_user_sim__g2b_dropped.json").read_text())
    assert drop["superseded_by"]["round"] == cm.SIMREGEN_ROUND
    assert drop["status"] == "N/A"  # original drop record preserved, only stamped
