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


# ---------------------------------------------------------------------------
# r2 (code-review round 1 union): floor-verdict ordering + _link_into content
# identity + fresh-selection provenance + ladder fold-checkpoint resume +
# tiny-real CPU e2e of the production phase_p7_simuser_fits body.
# ---------------------------------------------------------------------------


def test_link_into_content_identity(tmp_path):
    """codex major (the #1005-family stale-mirror class): st_size equality is
    NOT content identity — equal-length unequal bytes are REPLACED (atomic),
    digest-equal copies and same-inode hardlinks are skipped, a missing
    source raises."""
    import os

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.json").write_text('{"v": "new"}')
    (src / "b.json").write_text("AAAA")
    (dst / "b.json").write_text("BBBB")  # equal size, unequal content
    (src / "c.json").write_text("same-bytes")
    (dst / "c.json").write_text("same-bytes")  # digest-equal, different inode
    (src / "d.json").write_text("linked")
    os.link(src / "d.json", dst / "d.json")  # same inode (samestat fast path)
    c_inode = (dst / "c.json").stat().st_ino
    n = dsp._link_into(src, dst, ["a.json", "b.json", "c.json", "d.json"])
    assert n == 2  # a (new) + b (replaced); c digest-skip, d samestat-skip
    assert (dst / "a.json").read_text() == '{"v": "new"}'
    assert (dst / "b.json").read_text() == "AAAA"  # replaced, not silently kept
    assert (dst / "c.json").stat().st_ino == c_inode  # digest skip left it alone
    assert not list(dst.glob(".*.linktmp"))  # tmp names cleaned by os.replace
    with pytest.raises(RuntimeError, match="missing source"):
        dsp._link_into(src, dst, ["nope.json"])


def test_stage_user_real_render_links_parent_rows(monkeypatch, tmp_path):
    """Real _stage_user_real_render body; only the Hub staging boundary is
    faked (signature-mirroring def). Empty leaf fails loud."""
    leaf = tmp_path / "mirror" / "leaf"
    leaf.mkdir(parents=True)

    def fake_stage_hf_prefix(prefix_rel: str, dest_root, revision=None):
        assert prefix_rel == f"{cm.HF_PREFIX}/raw_completions/user_real_render"
        return leaf

    monkeypatch.setattr(cm, "stage_hf_prefix", fake_stage_hf_prefix)
    args = SimpleNamespace(stage_root=str(tmp_path / "stage"))
    raw_root = tmp_path / "raw"
    with pytest.raises(RuntimeError, match="no user_real_render rows"):
        dsp._stage_user_real_render(args, raw_root)
    (leaf / "w1_d137_s0_c0000.jsonl").write_text('{"conv_id": "a"}\n')
    (leaf / "w1_d137_s0_c0001.jsonl").write_text('{"conv_id": "b"}\n')
    dsp._stage_user_real_render(args, raw_root)
    got = sorted(p.name for p in (raw_root / "user_real_render").glob("*.jsonl"))
    assert got == ["w1_d137_s0_c0000.jsonl", "w1_d137_s0_c0001.jsonl"]


class _RecordingRunner(dsp.Runner):
    """Runner fake for the regen ORDERING tests: records (kind, name), runs
    optional per-step probes/side-effects instead of spawning subprocesses
    (the subprocess boundary is the faked seam; every dispatcher-side branch
    body stays real). Overrides mirror the real signatures (#906)."""

    def __init__(self, logs_dir, *, side_effects=None, probes=None):
        super().__init__(logs_dir, resume=False, dry=False)
        self.events: list[tuple[str, str]] = []
        self.side_effects = side_effects or {}
        self.probes = probes or {}

    def _record(self, kind: str, name: str) -> None:
        if name in self.probes:
            self.probes[name]()
        self.events.append((kind, name))
        if name in self.side_effects:
            self.side_effects[name]()

    def run(self, name, argv, *, env_extra=None, ok_rcs=(0,), timeout_s=None, tail_lines=25):
        self._record("run", name)
        return 0

    def fanout(self, name, base_argv, *, gpus, env_extra=None):
        self._record("fanout", name)

    def parallel(self, name, argv_list, *, gpus, env_extra=None):
        self._record("parallel", name)

    def cpu_parallel(self, name, argv_list, *, threads_each=None, env_extra=None):
        self._record("cpu_parallel", name)

    def names(self) -> list[str]:
        return [n for _, n in self.events]


def _regen_fixture(tmp_path, monkeypatch, *, n_kept, n_inter, kept_rung1, kept_rung3):
    """Shared harness for the phase_p7_simuser_regen ordering tests: fixture
    roots + below/above-floor capture_ready + sim elicitation summaries, with
    every external boundary faked signature-conformant."""
    raw = tmp_path / "raw"
    led = tmp_path / "led"
    store = tmp_path / "store"
    parent_led = tmp_path / "parent_led"
    cm.atomic_write_json(parent_led / "pilot" / "layer_sweep.json", {"selected_layer": 1})
    cm.atomic_write_json(
        led / "capture_ready" / "chat_user_sim.json",
        {"n_kept": n_kept, "pair_intersection": {"n_intersection": n_inter}},
    )
    cm.atomic_write_json(
        raw / "user_sim" / "summary_w1_s0.json",
        {"elicitation": {"kept_rung1": kept_rung1, "kept_rung3": kept_rung3}},
    )
    # store part so the terminal build_store_index runs its REAL body.
    cm.atomic_write_json(
        store / "chat_user_sim__part0000__rows.json",
        {"cell": "chat_user_sim", "tag": "chat_user_sim", "part": 0, "rows": [{"row_id": "a"}]},
    )
    args = SimpleNamespace(
        simregen_raw_root=str(raw),
        simregen_ledger_root=str(led),
        simregen_store_root=str(store),
        ledger_root=str(parent_led),
        stage_root=str(tmp_path / "stage"),
        sentinel_dir=str(tmp_path / "sent"),
        layers="Lstar",
        user_rows=100,
        user_fresh_rows=4,
        user_fresh_draws=2,
    )
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(dsp, "visible_gpus", lambda: ["0"])
    monkeypatch.setattr(dsp, "_git_pull_rebase", lambda: None)

    def fake_ensure_model_venv(a, runner):
        events.append(("ensure", "model_venv"))

    def fake_assert_headroom(phase: str, out_root):
        events.append(("headroom", phase))

    def fake_stage_user_real_render(a, raw_root):
        events.append(("stage", "user_real_render"))

    def fake_stage_parent_store_slice(a, npz_cells, layers):
        assert npz_cells == {"chat_user_real"} and layers == [1]
        events.append(("stage", "real_vcvp"))
        return tmp_path / "parent_slice"

    harvests: list[str] = []

    def fake_git_harvest(paths, message, *, force_add=False):
        harvests.append(message)

    uploads: list[str] = []

    def fake_upload_stage_dir(local_dir, prefix_rel):
        uploads.append(prefix_rel)
        return ["ok"]

    monkeypatch.setattr(dsp, "ensure_model_venv", fake_ensure_model_venv)
    monkeypatch.setattr(dsp, "assert_headroom", fake_assert_headroom)
    monkeypatch.setattr(dsp, "_stage_user_real_render", fake_stage_user_real_render)
    monkeypatch.setattr(dsp, "_stage_parent_store_slice", fake_stage_parent_store_slice)
    monkeypatch.setattr(dsp, "git_harvest", fake_git_harvest)
    monkeypatch.setattr(cm, "upload_stage_dir", fake_upload_stage_dir)
    return args, led, raw, events, harvests, uploads


def _sentinels(args) -> list[dict]:
    return [
        json.loads(p.read_text()) for p in sorted(Path(args.sentinel_dir).glob("issue-2378-*.json"))
    ]


def test_regen_floor_fail_reports_before_any_fresh_spend(tmp_path, monkeypatch):
    """codex blocker 1 (mechanizable form): a below-floor wave writes the
    floor report + blocking sentinel and exits rc 8 BEFORE any fresh-draw
    step is even composed — gen_user_fresh/upload_user_fresh/capture never
    appear in the step record."""
    args, led, _raw, _events, harvests, _uploads = _regen_fixture(
        tmp_path, monkeypatch, n_kept=6000, n_inter=5900, kept_rung1=6000, kept_rung3=0
    )
    runner = _RecordingRunner(tmp_path / "logs")
    monkeypatch.setattr(
        dsp,
        "_stage_user_real_render",
        lambda a, r: runner.events.append(("stage", "user_real_render")),
    )
    rc = dsp.phase_p7_simuser_regen(args, runner)
    assert rc == dsp.RC_SIMREGEN_FLOOR
    names = runner.names()
    # Exact realized order: gen -> upload -> real render staging ->
    # capture_ready -> floor verdict (rc 8) — and NOTHING after it.
    assert names == [
        "p7.gen_user_sim",
        "p7.upload_user_sim",
        "user_real_render",
        "p7.capture_ready",
    ]
    for banned in ("p7.gen_user_fresh", "p7.upload_user_fresh", "p7.capture_sim"):
        assert banned not in names
    report = json.loads((led / "simregen_floor_report.json").read_text())
    assert report["floor_gate"]["verdict"] == "FLOOR_FAIL"
    assert report["fresh_eligibility"]["verdict"] == "OK"  # eligibility recorded either way
    assert not (led / "fresh_reference_na.json").exists()  # floor-fail path, not NA path
    gates = [s for s in _sentinels(args) if s.get("gate") == "simregen_floor"]
    assert len(gates) == 1 and gates[0]["blocks_pipeline"] is True
    assert gates[0]["kind"] == "epm:progress"
    assert any("FLOOR_FAIL" in m for m in harvests)


def test_regen_zero_rung1_writes_na_and_skips_fresh(tmp_path, monkeypatch):
    """codex blocker 1: floor PASS + zero rung-1 keeps => explicit
    fresh_reference_na.json (never an anonymous raise), fresh gen/upload +
    capture_sim_fresh SKIPPED, the sim capture + store upload still run."""
    args, led, _raw, events, _harvests, uploads = _regen_fixture(
        tmp_path, monkeypatch, n_kept=6501, n_inter=6501, kept_rung1=0, kept_rung3=6501
    )
    runner = _RecordingRunner(tmp_path / "logs")
    rc = dsp.phase_p7_simuser_regen(args, runner)
    assert rc == 0
    names = runner.names()
    for banned in ("p7.gen_user_fresh", "p7.upload_user_fresh", "p7.capture_sim_fresh"):
        assert banned not in names
    assert "p7.capture_sim" in names
    na = json.loads((led / "fresh_reference_na.json").read_text())
    assert na["fresh_reference"] == "N/A"
    assert na["eligibility"]["verdict"] == "NA" and na["eligibility"]["rung1_kept"] == 0
    report = json.loads((led / "simregen_report.json").read_text())
    assert report["fresh_reference_na"] is True and report["fresh_selection"] is None
    assert cm.SIMREGEN_ACTIVATIONS_PREFIX in uploads
    assert ("stage", "real_vcvp") in events  # vcvp staging still ran for capture_sim


def test_regen_short_coverage_proceeds_after_floor_persisted(tmp_path, monkeypatch):
    """codex blocker 1 (ordering half): rung-1 keeps below the requested
    fresh rows => short-coverage artifact + the fresh legs still run, and the
    floor report is ON DISK before the first fresh-generation step fires."""
    args, led, raw, _events, _harvests, _uploads = _regen_fixture(
        tmp_path, monkeypatch, n_kept=6501, n_inter=6501, kept_rung1=3, kept_rung3=6498
    )
    sel_payload = {"predicate": "elicit_rung == 1", "selected_rows": 3, "selection_digest": "d"}

    def make_fresh_selection():
        cm.atomic_write_json(raw / "user_sim_fresh" / "fresh_selection.json", sel_payload)

    ordering_seen: list[str] = []

    def probe_floor_persisted():
        assert (led / "simregen_floor_report.json").exists()  # floor BEFORE fresh spend
        ordering_seen.append("floor-before-fresh")

    runner = _RecordingRunner(
        tmp_path / "logs",
        side_effects={"p7.gen_user_fresh": make_fresh_selection},
        probes={"p7.gen_user_fresh": probe_floor_persisted},
    )
    rc = dsp.phase_p7_simuser_regen(args, runner)
    assert rc == 0
    assert ordering_seen == ["floor-before-fresh"]
    names = runner.names()
    for required in ("p7.gen_user_fresh", "p7.upload_user_fresh", "p7.capture_sim_fresh"):
        assert required in names
    na = json.loads((led / "fresh_reference_na.json").read_text())
    assert na["fresh_reference"] == "short-coverage"
    assert json.loads((led / "fresh_selection.json").read_text()) == sel_payload  # ledger copy
    report = json.loads((led / "simregen_report.json").read_text())
    assert report["fresh_reference_na"] is False
    assert report["fresh_selection"] == sel_payload


def test_phase_user_fresh_selection_provenance(monkeypatch, tmp_path):
    """codex concern fresh-retrieval-rung1-provenance (producer side): the
    REAL phase_user_fresh body persists the rung-1 eligibility predicate +
    counts + selection digest BEFORE generating; only the engine/tokenizer/
    pool/upload boundaries are faked (signature-mirroring defs, #906). The
    zero-rung-1 backstop raise stays loud on direct invocation."""
    raw_root = tmp_path / "raw"
    (raw_root / "user_sim").mkdir(parents=True)
    (raw_root / "user_sim" / "w1_d137_s0_c0000.jsonl").write_text("{}\n")  # _rows_dir local hit
    pool_rows = [{"conv_id": f"c{i}", "u1": "q", "a1": "a", "u2": "u"} for i in range(6)]
    kept = {
        "c0": {"elicit_rung": 1},
        "c1": {"elicit_rung": 3},
        "c2": {"elicit_rung": 1},
        "c3": {"elicit_rung": 1},
        "c5": {"elicit_rung": 3},
    }
    ran: dict = {}

    def fake_run_user_sim(args, llm, tok, rows, stage, seeds):
        ran["rows"] = rows
        ran["stage"] = stage
        ran["seeds"] = seeds

    monkeypatch.setattr(gen, "_resolve_pools_dir", lambda args: tmp_path / "pools")
    monkeypatch.setattr(gen, "_get_tokenizer", lambda: object())
    monkeypatch.setattr(gen, "_assert_chat_template", lambda tok: "ok")
    monkeypatch.setattr(gen, "_load_pool", lambda pools_dir, name: list(pool_rows))
    monkeypatch.setattr(gen, "_stage_kept_rows", lambda rows_dir, cell=None: dict(kept))
    monkeypatch.setattr(gen, "_build_engine", lambda args: object())
    monkeypatch.setattr(gen, "_run_user_sim", fake_run_user_sim)
    monkeypatch.setattr(gen, "_reap_engine", lambda llm: None)
    monkeypatch.setattr(gen, "_maybe_upload", lambda args, stage: None)
    args = SimpleNamespace(
        raw_root=str(raw_root),
        stage_raw_from_hf=False,
        user_fresh_rows=2,
        user_fresh_draws=2,
    )
    gen.phase_user_fresh(args)
    sel = json.loads((raw_root / "user_sim_fresh" / "fresh_selection.json").read_text())
    assert sel["predicate"].startswith("elicit_rung == 1")
    assert sel["total_sim_kept"] == 5
    assert sel["rung1_eligible"] == 3 and sel["rung3_excluded"] == 2
    assert sel["requested_rows"] == 2 and sel["selected_rows"] == 2
    assert sel["draw_seeds"] == list(cm.FRESH_SEEDS[:2])
    picked = [r["conv_id"] for r in ran["rows"]]
    assert set(picked) <= {"c0", "c2", "c3"}  # rung-1-kept only
    assert sel["selection_digest"] == cm.text_digest(",".join(picked))
    assert ran["stage"] == "user_sim_fresh" and ran["seeds"] == list(cm.FRESH_SEEDS[:2])

    # Zero rung-1 keeps: the dispatcher owns the N/A artifact; a DIRECT
    # invocation that bypassed it fails loud, never a silent empty run.
    monkeypatch.setattr(
        gen, "_stage_kept_rows", lambda rows_dir, cell=None: {"c1": {"elicit_rung": 3}}
    )
    with pytest.raises(RuntimeError, match="zero rung-1-kept"):
        gen.phase_user_fresh(args)


def test_ladder_fold_checkpoint_resume(monkeypatch, tmp_path):
    """codex concern p7-ladder-fold-not-resumable: a pair unit interrupted
    mid-fold-loop resumes from the per-fold checkpoints (folds 0-1 loaded,
    NOT recomputed), the checkpoint dir is reaped after the terminal rung
    JSONs land, and the resumed unit's outputs match a from-scratch run."""
    import argparse

    import issue2378_fits as fits_mod
    import issue2378_ladder as lad

    store = tmp_path / "store"
    fits_mod._write_probe_store(store, n=40, d=8)

    def ns(ledger: Path) -> argparse.Namespace:
        return argparse.Namespace(
            store_root=str(store),
            ledger_root=str(ledger),
            layer=1,
            layer_star_from=None,
            n_null_draws=4,
            bootstrap_draws=16,
            reduced_k=4,
            units="own:chat:context,own:chat_user_real:context",
            g3_gate_file=None,
            pairs="chat_user_real",
            survivors=None,
            fold_floors_override=fits_mod._PROBE_FLOORS,
        )

    real_compute = lad.compute_rungs_for_fold
    target = "chat_user_real"

    def run_fits(ledger: Path) -> argparse.Namespace:
        a = ns(ledger)
        ledger.mkdir(parents=True, exist_ok=True)
        assert fits_mod.phase_g3(a) == 0
        assert fits_mod.phase_fit(a) == 0
        return a

    # Scratch reference run (uninterrupted) in its own ledger.
    ref_args = run_fits(tmp_path / "ledger_ref")
    fm_ref = fits_mod._fold_map(ref_args)
    lad.run_pair_unit(ref_args, fm_ref, lad._SourceMemo(store, fm_ref, 1), target, 1)

    # Interrupted run: die BEFORE fold 2's compute; folds 0-1 checkpointed.
    args = run_fits(tmp_path / "ledger_resume")
    fm = fits_mod._fold_map(args)
    ck_dir = Path(args.ledger_root) / "ladder" / "fold_ckpt" / f"chat_to_{target}"
    calls = {"n": 0}

    def dying_compute(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("synthetic mid-unit crash")
        return real_compute(*a, **kw)

    monkeypatch.setattr(lad, "compute_rungs_for_fold", dying_compute)
    with pytest.raises(RuntimeError, match="synthetic mid-unit crash"):
        lad.run_pair_unit(args, fm, lad._SourceMemo(store, fm, 1), target, 1)
    assert calls["n"] == 3
    assert sorted(p.name for p in ck_dir.glob("f*.npz")) == ["f0.npz", "f1.npz"]

    # Resume: folds 0-1 load from checkpoint, only folds 2..k-1 recompute.
    calls["n"] = 0
    monkeypatch.setattr(lad, "compute_rungs_for_fold", dying_compute)  # dies on call 3 again
    k = fm["k"]
    resumed = {"n": 0}

    def counting_compute(*a, **kw):
        resumed["n"] += 1
        return real_compute(*a, **kw)

    monkeypatch.setattr(lad, "compute_rungs_for_fold", counting_compute)
    lad.run_pair_unit(args, fm, lad._SourceMemo(store, fm, 1), target, 1)
    assert resumed["n"] == k - 2  # exactly the un-checkpointed folds
    assert not ck_dir.exists()  # derived state reaped after terminal rungs land
    for ri in range(1, len(lad.RUNGS) + 1):
        assert (Path(args.ledger_root) / "ladder" / f"chat_to_{target}__rung{ri}.json").exists()

    # Resumed outputs match the scratch run (nulls bitwise via saved draws;
    # r2 to float32-checkpoint precision on the resumed folds).
    for ri in (1, len(lad.RUNGS)):
        got = json.loads(
            (Path(args.ledger_root) / "ladder" / f"chat_to_{target}__rung{ri}.json").read_text()
        )
        want = json.loads(
            (Path(ref_args.ledger_root) / "ladder" / f"chat_to_{target}__rung{ri}.json").read_text()
        )
        np.testing.assert_allclose(
            [x["r2"] for x in got["per_fold"]],
            [x["r2"] for x in want["per_fold"]],
            rtol=1e-3,
            atol=1e-5,
        )
        np.testing.assert_allclose(got["pooled_r2"], want["pooled_r2"], rtol=1e-3, atol=1e-5)
        if "per_fold_draws" in got.get("null", {}):
            assert got["null"]["per_fold_draws"] == want["null"]["per_fold_draws"]

    # Regime change on a completed unit refuses loud (the unit-level resume
    # guard sits UPSTREAM of the fold fingerprint; a changed regime never
    # silently reuses either the terminal JSONs or the fold checkpoints).
    args2 = ns(Path(args.ledger_root))
    args2.n_null_draws = 5
    with pytest.raises(RuntimeError, match="regime mismatch"):
        lad.run_pair_unit(args2, fm, lad._SourceMemo(store, fm, 1), target, 1)


def _vary_answer_lengths(store: Path, cell: str, seed: int) -> None:
    """Give a probe cell varied answer-span lengths inside the lenmatch
    [8, 256] band (the probe store's constant 7-token spans sit BELOW the
    band and would empty the matched selection)."""
    import issue2378_p6_common as p6

    rng = np.random.default_rng(seed)
    for ci in p6.production_part_indices(store, cell):
        path = store / f"{cell}__part{ci:04d}__rows.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for r in payload["rows"]:
            span = int(rng.integers(8, 200))
            r["ans_lo"] = 2
            r["ans_hi"] = 2 + span
            r["n_tokens"] = 4 + span
        path.write_text(json.dumps(payload), encoding="utf-8")


def _relabel_store_layer(store: Path, to_layer: int) -> None:
    """Rename the probe store's L1 npz parts (and their meta layer field) to
    the parent run's REALIZED L* (issue2378_lenmatch pins LAYER=51 as a
    parent-frozen constant, so the e2e fixture matches production's layer)."""
    for npz_path in sorted(store.glob("*__L1.npz")):
        with np.load(npz_path) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files}
        meta = json.loads(str(arrays["meta"]))
        meta["layer"] = to_layer
        arrays["meta"] = np.array(json.dumps(meta))
        out = npz_path.with_name(npz_path.name.replace("__L1.npz", f"__L{to_layer}.npz"))
        with open(out, "wb") as fh:
            np.savez(fh, **arrays)
        npz_path.unlink()


@pytest.mark.slow
def test_p7_simuser_fits_tiny_real_e2e(monkeypatch, tmp_path):
    """codex major p7-cpu-smoke-incomplete: the PRODUCTION phase_p7_simuser_fits
    body over a tiny-real two-arm store (d=8; n at the REAL 6,500 floor — the
    production fold-map floors stay byte-untouched), faking ONLY the Hub/git
    boundaries. Every stage runs the real subprocess entrypoint via the real
    Runner: fold map -> fits -> ladder pairs -> H4b -> retrieval (+ fresh
    reference + provenance injection) -> lenmatch --pair-user -> plan-named
    sync + digest, rc=0. Also pins code-review blocker 2's combined-store
    assembly: a stale parent-side chat_user_sim ledger is EXCLUDED — the
    round store is the only source of sim rows in the combined store."""
    import shutil

    import issue2378_fits as fits_mod
    import issue2378_p6_common as p6
    import issue2378_retrieval as ret

    n = 6520  # >= the untouched 6,500 production floor after the probe drops
    lstar = 51  # the parent run's realized L* (lenmatch pins LAYER=51)
    parent_store = tmp_path / "parent_store"
    round_store = tmp_path / "round_store"
    round_store.mkdir()
    fits_mod._write_probe_store(parent_store, n=n, d=8)
    _relabel_store_layer(parent_store, lstar)
    _vary_answer_lengths(parent_store, "chat_user_real", seed=41)
    _vary_answer_lengths(parent_store, "chat_user_sim", seed=42)
    for p in sorted(parent_store.glob("chat_user_sim__*")):
        shutil.move(str(p), round_store / p.name)

    # Preview fold map over EXACTLY the files the combined store will hold
    # (production floors — no override); used only to seed the fresh parts.
    preview = tmp_path / "preview"
    dsp._link_into(parent_store, preview, [p.name for p in parent_store.iterdir()])
    dsp._link_into(round_store, preview, [p.name for p in round_store.iterdir()])
    fm_preview = p6.load_or_build_fold_map(preview, tmp_path / "preview_ledger")
    ret._write_probe_fresh(round_store, fm_preview, "chat_user_sim", layer=lstar)

    # STALE parent-side sim ledger + npz (blocker 2): valid-JSON junk that
    # would poison the fold map if linked; the npz is a loud tripwire.
    stale_rows = json.dumps(
        {"cell": "chat_user_sim", "tag": "chat_user_sim", "part": 0, "rows": [{"row_id": "STALE"}]}
    )
    (parent_store / "chat_user_sim__part0000__rows.json").write_text(stale_rows)
    (parent_store / f"chat_user_sim__part0000__L{lstar}.npz").write_bytes(b"STALE-NOT-AN-NPZ")

    parent_ledger = tmp_path / "parent_ledger"
    cm.atomic_write_json(parent_ledger / "pilot" / "layer_sweep.json", {"selected_layer": lstar})
    cm.atomic_write_json(parent_ledger / p6.G3_GATE_NAME, {"gate": "G3", "verdict": "PASS"})
    round_ledger = tmp_path / "round_ledger"
    cm.atomic_write_json(
        round_ledger / "capture_ready" / "chat_user_sim.json",
        {"n_kept": 6517, "pair_intersection": {"n_intersection": 6514}},
    )
    fresh_sel = {
        "predicate": "elicit_rung == 1 (rung-1-kept user_sim conversations only)",
        "rung1_eligible": 6517,
        "selected_rows": 12,
        "selection_digest": "fixture",
    }
    cm.atomic_write_json(round_ledger / "fresh_selection.json", fresh_sel)

    args = SimpleNamespace(
        simregen_raw_root=str(tmp_path / "raw"),
        simregen_ledger_root=str(round_ledger),
        simregen_store_root=str(round_store),
        ledger_root=str(parent_ledger),
        stage_root=str(tmp_path / "stage"),
        sentinel_dir=str(tmp_path / "sent"),
    )
    monkeypatch.setattr(dsp, "_git_pull_rebase", lambda: None)

    def fake_assert_headroom(phase: str, out_root) -> None:
        assert phase == "p7_simuser_fits"

    def fake_stage_parent_store_slice(a, npz_cells, layers):
        assert npz_cells == {"chat", "chat_user_real"} and layers == [lstar]
        return parent_store

    def fail_stage_hf_prefix(prefix_rel, dest_root, revision=None):
        raise AssertionError(f"unexpected HF staging of {prefix_rel} (round npz are local)")

    uploads: list[str] = []
    harvests: list[list[str]] = []
    monkeypatch.setattr(dsp, "assert_headroom", fake_assert_headroom)
    monkeypatch.setattr(dsp, "_stage_parent_store_slice", fake_stage_parent_store_slice)
    monkeypatch.setattr(cm, "stage_hf_prefix", fail_stage_hf_prefix)
    monkeypatch.setattr(
        cm, "upload_stage_dir", lambda d, prefix: (uploads.append(prefix), ["ok"])[1]
    )
    monkeypatch.setattr(
        dsp, "git_harvest", lambda paths, msg, force_add=False: harvests.append(paths)
    )

    runner = dsp.Runner(tmp_path / "logs", resume=True, dry=False)
    rc = dsp.phase_p7_simuser_fits(args, runner)
    assert rc == 0

    # Blocker 2: the combined store's sim ledger is the ROUND bytes — the
    # stale parent copy was excluded from the parent link set.
    combined = Path(args.stage_root) / "simregen_combined"
    got = (combined / "chat_user_sim__part0000__rows.json").read_bytes()
    assert got == (round_store / "chat_user_sim__part0000__rows.json").read_bytes()
    assert got != stale_rows.encode()
    # Assembly identity: the realized fold map matches the preview built from
    # the intended file set (same content -> same canonical sha).
    fm_real = json.loads((round_ledger / p6.FOLD_MAP_NAME).read_text())
    assert fm_real["sha256"] == fm_preview["sha256"]

    # Per-stage artifacts (fits / ladder / h4b / retrieval / lenmatch / sync
    # / digest), each produced by the REAL subprocess entrypoint.
    for cell in ("chat_user_real", "chat_user_sim"):
        fits = json.loads((round_ledger / "fits" / f"{cell}__context.json").read_text())
        assert fits["pooled_r2"] > 0.5  # planted linear geometry
        assert (round_ledger / "fits" / f"{cell}__prefix.json").exists()
        for ri in range(1, len(dsp_ladder_rungs()) + 1):
            assert (round_ledger / "ladder" / f"chat_to_{cell}__rung{ri}.json").exists()
    h4b = json.loads((round_ledger / "ladder" / "h4b_real_vs_sim.json").read_text())
    assert h4b["pair_assert"]["n_hash_mismatched"] == 0
    fresh_real = json.loads((round_ledger / "retrieval" / "chat_user_real__fresh.json").read_text())
    assert fresh_real["status"] == "N/A"  # deterministic render — disclosed, never a crash
    fresh_sim = json.loads((round_ledger / "retrieval" / "chat_user_sim__fresh.json").read_text())
    assert fresh_sim["status"] == "ok" and fresh_sim["counts"]["n_covered"] > 0
    for rj in (round_ledger / "retrieval").glob("*.json"):
        payload = json.loads(rj.read_text())
        assert payload["fresh_selection"] == fresh_sel, rj.name  # provenance injection
    pair = json.loads((round_ledger / "lenmatch" / "lenmatch_user_pair.json").read_text())
    assert "sim_min_tokens_floor" in pair
    digest = json.loads((round_ledger / "p7_digest.json").read_text())
    assert digest["fresh_reference_na"] is False and len(digest["plan_named_sync"]) >= 3
    synced = json.loads((parent_ledger / "fits" / "chat_user_sim__context.json").read_text())
    assert synced["round"] == cm.SIMREGEN_ROUND
    kinds = [
        json.loads(p.read_text())["kind"] for p in Path(args.sentinel_dir).glob("issue-2378-*.json")
    ]
    assert "epm:results" in kinds
    assert any("p6_sidecars_simregen" in u for u in uploads)


def dsp_ladder_rungs() -> tuple[str, ...]:
    import issue2378_ladder as lad

    return lad.RUNGS
