"""CPU tests for scripts/issue823_ladder_ext_capture.py (#823 origin-ladder-more-contexts).

Covers the P0-ext gate logic (a)-(e), mask arithmetic, fingerprint-sidecar
resume semantics (mutation => recapture, never skip), shard-manifest
reassembly, Gate B projection arithmetic, skip_mask/truncation handling, the
ext pair-row contract, and the batched cx capture's batched-vs-serial
equivalence on a tiny from-config model. Flat synthetic fixtures only — no
network, no GPU, no real-corpus text.

Each designed-halt test asserts the DISTINCT rc from the driver's docstring
table AND that no downstream completion sentinel exists after the halt.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from explore_persona_space.experiments.issue_823 import run_823
from scripts import issue823_ladder_capture as CAP
from scripts import issue823_ladder_ext_capture as EXTCAP
from scripts import issue823_ladder_ext_gen as EXTGEN

# ── Fixtures ─────────────────────────────────────────────────────────────────

SUF_IDS = [901, 902, 903]


class FakeTokenizer:
    """Deterministic word-count tokenizer matching the capture rig's contract.

    apply_chat_template wraps each message in <hdr>/<end> pseudo-tokens and
    appends a 3-token generation suffix under add_generation_prompt=True;
    decode of exactly those 3 ids returns GENERATION_SUFFIX. Word-token ids
    hash into [100, 599] (char-sum, PYTHONHASHSEED-independent), disjoint
    from the suffix ids.
    """

    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        parts = [f"<hdr> {m['content']} <end>" for m in messages]
        text = " ".join(parts)
        if add_generation_prompt:
            text += " \x00SUF"
        return text

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        assert return_tensors is None and add_special_tokens is False
        ids: list[int] = []
        for word in text.split():
            if word == "\x00SUF":
                ids.extend(SUF_IDS)
            else:
                ids.append(100 + (sum(ord(c) for c in word) % 500))
        return {"input_ids": ids}

    def decode(self, ids):
        if list(ids) == SUF_IDS:
            return run_823.GENERATION_SUFFIX
        return " ".join(f"t{i}" for i in ids)


def _ext_record(i: int, p: int, **over) -> dict:
    """Minimal cross-unit-contract ext record (unit-2 build_ext_records schema)."""
    arms = [k for k in EXTCAP.EXT_ARMS if (i % k) == p]
    row = {
        "context_id": i,
        "persona_idx": p,
        "persona_name": f"persona{p}",
        "arms": arms,
        "corpus": "ladder_ext",
        "gen_stage": "wave",
        "question": f"question alpha {i}",
        "answer_text": "alpha beta gamma delta",
        "seed": 42,
        "filled": True,
        "validity": "ok",
        "stop_reason": "end_turn",
        "cap_hit": False,
        "model": "m",
        "temperature": 1.0,
        "max_tokens": EXTCAP.GEN_MAX_TOKENS,
        "gen_wave": "wave01",
        "regen": False,
        "system_prompt": f"You are persona{p}. card{p}",
        "system_prompt_sha256": EXTCAP._sha256_text(f"You are persona{p}. card{p}"),
    }
    row.update(over)
    return row


def _roster() -> dict:
    return {
        "template": "You are {name}. {card}",
        "personas": [{"idx": p, "name": f"persona{p}", "card": f"card{p}"} for p in range(16)],
    }


def _tiny_layout(tmp_path: pathlib.Path, n_prefix: int, n_total: int, smoke: bool = False):
    layout = EXTCAP.Layout(tmp_path / "out", smoke=smoke, n_ext=n_total - n_prefix)
    layout.n_prefix = n_prefix
    layout.n_total = n_total
    layout.rungs = (n_prefix, n_total)
    layout.out_root.mkdir(parents=True, exist_ok=True)
    return layout


def _assert_no_sentinels(layout) -> None:
    """After a designed halt, no downstream completion sentinel may exist."""
    for rel in (
        f"own/{EXTCAP.OWNGEN_SENTINEL}",
        f"pair_store/{EXTCAP.CAPTURE_SENTINEL}",
        f"pair_store/{EXTCAP.STORE_SENTINEL}",
    ):
        assert not (layout.out_root / rel).exists(), rel


def _tiny_ext_by_persona(n_prefix: int, n_total: int) -> dict[int, list[dict]]:
    by_p: dict[int, list[dict]] = {p: [] for p in range(16)}
    for i in range(n_prefix, n_total):
        by_p[0].append(_ext_record(i, 0))
        if i % 16 != 0:
            by_p[i % 16].append(_ext_record(i, i % 16))
    return by_p


def _tiny_banked_by_persona(n_prefix: int) -> dict[int, list[dict]]:
    by_p: dict[int, list[dict]] = {p: [] for p in range(16)}
    for i in range(n_prefix):
        by_p[0].append({"context_id": i, "validity": "ok"})
        if i % 16 != 0:
            by_p[i % 16].append({"context_id": i, "validity": "ok"})
    return by_p


# ── Gate (a): prompt integrity ───────────────────────────────────────────────


def test_gate_a_pass_and_halt_rc12(tmp_path):
    layout = _tiny_layout(tmp_path, 4, 8)
    ext = _tiny_ext_by_persona(4, 8)
    report = EXTCAP.gate_a_prompt_integrity(ext, _roster(), layout.eval_dir)
    assert report["pass"] and report["n_checked"] > 0

    ext[0][0]["system_prompt"] = "TAMPERED"
    with pytest.raises(SystemExit) as exc:
        EXTCAP.gate_a_prompt_integrity(ext, _roster(), layout.eval_dir)
    assert exc.value.code == EXTCAP.RC_PROMPT_INTEGRITY == 12
    halt = json.loads((layout.eval_dir / "ext_prompt_integrity_report.json").read_text())
    assert halt["rc"] == 12 and halt["n_mismatched"] == 1
    _assert_no_sentinels(layout)


# ── Gate (b): assignment recompute + banked prefix ───────────────────────────


def _assignment_obj(n: int) -> dict:
    arms = EXTGEN.build_ext_assignment(n)
    return json.loads(json.dumps({"arms": arms, "n_contexts": n}))  # str-keyed, JSON round-trip


def test_gate_b_pass_and_halt_rc13(tmp_path):
    layout = _tiny_layout(tmp_path, 4, 8)
    staged, banked = _assignment_obj(8), _assignment_obj(4)
    report = EXTCAP.gate_b_assignment(staged, banked, 8, layout.eval_dir)
    assert report["pass"] and report["n_banked_prefix"] == 4

    tampered = json.loads(json.dumps(staged))
    tampered["arms"]["16"][5] = (tampered["arms"]["16"][5] + 1) % 16
    with pytest.raises(SystemExit) as exc:
        EXTCAP.gate_b_assignment(tampered, banked, 8, layout.eval_dir)
    assert exc.value.code == EXTCAP.RC_ASSIGNMENT_PARITY == 13

    bad_banked = json.loads(json.dumps(banked))
    bad_banked["arms"]["16"][0] = (bad_banked["arms"]["16"][0] + 1) % 16
    with pytest.raises(SystemExit) as exc:
        EXTCAP.gate_b_assignment(staged, bad_banked, 8, layout.eval_dir)
    assert exc.value.code == 13
    _assert_no_sentinels(layout)


# ── Gate (c): max_tokens / gen_wave / regen ──────────────────────────────────


def test_gate_c_pass_and_halt_rc14(tmp_path):
    layout = _tiny_layout(tmp_path, 4, 8)
    ext = _tiny_ext_by_persona(4, 8)
    ext[0][1].update(max_tokens=EXTCAP.REGEN_MAX_TOKENS, regen=True)  # legal regen row
    assert EXTCAP.gate_c_max_tokens(ext, layout.eval_dir)["pass"]

    ext[0][0]["max_tokens"] = 1234
    with pytest.raises(SystemExit) as exc:
        EXTCAP.gate_c_max_tokens(ext, layout.eval_dir)
    assert exc.value.code == EXTCAP.RC_MAX_TOKENS_WAVE == 14

    ext[0][0]["max_tokens"] = EXTCAP.GEN_MAX_TOKENS
    ext[0][0]["regen"] = True  # inconsistent with the base cap
    with pytest.raises(SystemExit) as exc:
        EXTCAP.gate_c_max_tokens(ext, layout.eval_dir)
    assert exc.value.code == 14
    _assert_no_sentinels(layout)


# ── Gate (d): mask accounting + integrity-class abort ────────────────────────


def test_build_masks_arithmetic_and_bridge_crosscheck(tmp_path):
    layout = _tiny_layout(tmp_path, 6, 10)
    banked = _tiny_banked_by_persona(6)
    ext = _tiny_ext_by_persona(6, 10)
    banked[2][0]["validity"] = "refusal"  # context 2's k=16 arm invalid -> drops from masks
    ext[7 % 16][0]["validity"] = "refusal"  # context 7 pair invalid (refusal class: no abort)
    bridge_ids = [0, 1, 3]
    masks = EXTCAP.build_masks(banked, ext, bridge_ids, layout, {"meta": True})
    r_prefix, r_total = masks["rungs"]["6"], masks["rungs"]["10"]
    assert r_prefix["ids"] == [0, 1, 3, 4, 5]  # ctx 2 dropped (equalize-down)
    assert r_total["ids"] == [0, 1, 3, 4, 5, 6, 8, 9]  # + ctx 7 dropped
    assert r_prefix["n_mask"] == 5 and r_total["n_mask"] == 8
    # 5-fold arithmetic: fold_max = ceil(8/5) = 2 -> n_train_min = 6
    assert r_total["n_train_per_fold_min"] == 6
    assert r_total["n_over_d"] == pytest.approx(6 / EXTCAP.EXPECTED_HIDDEN)
    assert masks["bridge"]["n_mask"] == 3 and masks["bridge"]["ids"] == bridge_ids
    assert masks["integrity_gate"] == "PASS"  # refusal-class never aborts
    assert masks["ext_arm_stats"]["16"]["n_refusal"] == 1


def test_build_masks_integrity_abort_rc15_and_smoke_warn(tmp_path):
    layout = _tiny_layout(tmp_path, 6, 10)
    banked = _tiny_banked_by_persona(6)
    ext = _tiny_ext_by_persona(6, 10)
    ext[0][0].update(validity="error:api", filled=False)  # 1/4 k=1 ext rows > 1%
    with pytest.raises(SystemExit) as exc:
        EXTCAP.build_masks(banked, ext, [0], layout, {})
    assert exc.value.code == EXTCAP.RC_NEW_INVALID == 15
    assert (layout.eval_dir / "ext_new_invalid_report.json").exists()
    _assert_no_sentinels(layout)

    smoke_layout = _tiny_layout(tmp_path / "smoke", 6, 10, smoke=True)
    masks = EXTCAP.build_masks(banked, ext, [0], smoke_layout, {})
    assert masks["integrity_gate"] == "WARN-SMOKE-INFORMATIONAL"


# ── Gate (e): duplicate content (flag-only, never a halt) ────────────────────


def test_gate_e_duplicates_flags_without_halt(tmp_path):
    layout = _tiny_layout(tmp_path, 0, 10)
    unique = {i: f"question number {i}" for i in range(10)}
    rep = EXTCAP.gate_e_duplicates(unique, layout)
    assert not rep["dedup_sensitivity_refit_required"] and rep["duplicate_fraction"] == 0.0

    dup = dict(unique)
    dup[9] = dup[0]  # 10% duplicates > 2% -> flag, still no raise
    rep = EXTCAP.gate_e_duplicates(dup, layout)
    assert rep["dedup_sensitivity_refit_required"] and rep["n_duplicate_groups"] == 1
    # Persisted GROUPS are what the fits driver's sens_dedup consumer reads
    # (r1 concern dedup-sensitivity-detached): sorted ids + min-id representative.
    assert rep["duplicate_groups"] == [{"context_ids": [0, 9], "representative": 0}]
    _assert_no_sentinels(layout)


# ── Probe (g): span-length unit ──────────────────────────────────────────────


def _b2_fixture(n: int, tok) -> tuple[list[dict], dict[str, list[int]]]:
    records, spans = [], []
    for i in range(n):
        q, a = f"query {i} tail", "alpha beta gamma delta"
        records.append({"context_id": i, "question": q, "answer_text": a, "filled": True})
        p_len, f_len = CAP.template_span_length(tok, q, a)
        spans.append(f_len - p_len)
    return records, {"b2": spans}


def test_probe_g_pass_and_halt_rc17(tmp_path):
    layout = _tiny_layout(tmp_path, 0, 1)
    tok = FakeTokenizer()
    records, span_d = _b2_fixture(EXTCAP.PROBE_N_CONTEXTS, tok)
    report = EXTCAP.probe_g_span_unit(tok, records, span_d, layout.eval_dir)
    assert report["pass"] and report["n_exact"] == EXTCAP.PROBE_N_CONTEXTS

    span_d["b2"][3] += 7  # 63/64 exact still passes (floor is >= 63)
    assert EXTCAP.probe_g_span_unit(tok, records, span_d, layout.eval_dir)["n_exact"] == 63

    span_d["b2"][5] += 7  # 62/64 -> halt
    with pytest.raises(SystemExit) as exc:
        EXTCAP.probe_g_span_unit(tok, records, span_d, layout.eval_dir)
    assert exc.value.code == EXTCAP.RC_SPAN_UNIT == 17
    assert (layout.eval_dir / "ext_span_unit_report.json").exists()
    _assert_no_sentinels(layout)


# ── Probe (f): capture-convention parity (behavioral rc-16 halt) ─────────────


def test_probe_f_capture_parity_pass_and_halt_rc16(tmp_path, monkeypatch):
    """probe_f's REAL body (cosine + max-rel math, report write, halt routing)
    with fakes only at the GPU boundary (capture_cx) and the 6-GB banked-bundle
    boundary (load_pass_b_cx_last)."""
    import numpy as np

    layout = _tiny_layout(tmp_path, 0, 1)
    rng = np.random.default_rng(0)
    ref = torch.from_numpy(rng.standard_normal((EXTCAP.PROBE_N_CONTEXTS, 2, 3)).astype("float32"))
    monkeypatch.setattr(EXTCAP, "load_pass_b_cx_last", lambda _path: ref)

    # PASS arm: extension-rig capture reproduces the banked rows exactly.
    monkeypatch.setattr(EXTCAP, "capture_cx", lambda _m, _t, qs, _bs: ref[: len(qs)].numpy().copy())
    qs = [f"q{i}" for i in range(EXTCAP.PROBE_N_CONTEXTS)]
    report = EXTCAP.probe_f_capture_parity(None, None, qs, tmp_path / "b.pt", layout.eval_dir, 8)
    assert report["pass"] and report["cosine_min"] >= EXTCAP.PROBE_F_COSINE_FLOOR

    # HALT arm: parallel but scaled rows keep cosine == 1 while max-rel ~ 1
    # exceeds the 1e-2 median cap -> rc 16 + report + no sentinels.
    monkeypatch.setattr(EXTCAP, "capture_cx", lambda _m, _t, qs, _bs: 2.0 * ref[: len(qs)].numpy())
    with pytest.raises(SystemExit) as exc:
        EXTCAP.probe_f_capture_parity(None, None, qs, tmp_path / "b.pt", layout.eval_dir, 8)
    assert exc.value.code == EXTCAP.RC_CAPTURE_PARITY == 16
    halt = json.loads((layout.eval_dir / "ext_capture_parity_report.json").read_text())
    assert halt["rc"] == 16 and halt["max_rel_median"] > EXTCAP.PROBE_F_MAXREL_MEDIAN_CAP
    _assert_no_sentinels(layout)


# ── Ext pair-row contract (loader) ───────────────────────────────────────────


def _write_ext_files(base: pathlib.Path, by_p: dict[int, list[dict]]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for p, rows in by_p.items():
        if rows:
            EXTGEN._write_jsonl(base / f"persona{p:02d}_ext.jsonl", rows)


def test_load_ext_pair_rows_contract(tmp_path):
    by_p = _tiny_ext_by_persona(4, 8)
    _write_ext_files(tmp_path, by_p)
    loaded = EXTCAP.load_ext_pair_rows(tmp_path, 4, 8)
    assert sum(len(r) for r in loaded.values()) == len(EXTGEN.build_ext_pairs(4, 8))
    assert [r["context_id"] for r in loaded[0]] == [4, 5, 6, 7]

    bad = _tiny_ext_by_persona(4, 8)
    bad[0][0]["corpus"] = "ladder"  # wrong corpus tag
    _write_ext_files(tmp_path / "bad_corpus", bad)
    with pytest.raises(AssertionError, match="corpus"):
        EXTCAP.load_ext_pair_rows(tmp_path / "bad_corpus", 4, 8)

    bad = _tiny_ext_by_persona(4, 8)
    bad[0][0]["in_common_valid"] = True  # prefix-only concept must be absent
    _write_ext_files(tmp_path / "bad_icv", bad)
    with pytest.raises(AssertionError, match="in_common_valid"):
        EXTCAP.load_ext_pair_rows(tmp_path / "bad_icv", 4, 8)

    bad = _tiny_ext_by_persona(4, 8)
    bad[0][0]["arms"] = [1, 16]  # violates the i-mod-k membership rule
    _write_ext_files(tmp_path / "bad_arms", bad)
    with pytest.raises(AssertionError, match="arms"):
        EXTCAP.load_ext_pair_rows(tmp_path / "bad_arms", 4, 8)

    missing = _tiny_ext_by_persona(4, 8)
    missing[5] = []  # drops context 5's k=16 pair
    _write_ext_files(tmp_path / "missing", missing)
    with pytest.raises(AssertionError, match="registered"):
        EXTCAP.load_ext_pair_rows(tmp_path / "missing", 4, 8)


# ── Shard-manifest reassembly ────────────────────────────────────────────────


def test_read_jsonl_manifest_first_roundtrip_and_sha_mismatch(tmp_path, monkeypatch):
    rows = [{"context_id": i, "question": f"row {i} " + "x" * 8} for i in range(40)]
    src = tmp_path / "persona00_ext.jsonl"
    EXTGEN._write_jsonl(src, rows)
    monkeypatch.setattr(EXTGEN, "UPLOAD_SHARD_LIMIT_BYTES", 128)
    monkeypatch.setattr(EXTGEN, "UPLOAD_SHARD_TARGET_BYTES", 96)
    uploads = EXTGEN.shard_large_jsonl_for_upload([src])
    names = {p.name for p in uploads}
    assert "persona00_ext.manifest.json" in names and len(names) > 2

    back = EXTCAP.read_jsonl_manifest_first(tmp_path, "persona00_ext")
    assert back == rows  # manifest-first reassembly is order-exact

    shard0 = tmp_path / "persona00_ext.shard00.jsonl"
    shard0.write_text(shard0.read_text().replace("row 0", "row X"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256"):
        EXTCAP.read_jsonl_manifest_first(tmp_path, "persona00_ext")

    (tmp_path / "plain.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    assert EXTCAP.read_jsonl_manifest_first(tmp_path, "plain") == [{"a": 1}]
    with pytest.raises(RuntimeError, match="neither"):
        EXTCAP.read_jsonl_manifest_first(tmp_path, "absent")


# ── Fingerprint sidecar resume (mutation => recapture, never skip) ───────────


def test_unit_fingerprint_resume_mutation_routes_to_recapture(tmp_path):
    store = tmp_path / "pair_store"
    fp = EXTCAP.unit_fingerprint("pairs", [5000, 5001], ["a" * 8, "b" * 8], "srcsha", 8)
    assert not EXTCAP.unit_done(store, "v_pairs_ext_p00_block0", fp)  # nothing on disk

    payload = {"v": torch.zeros(2, 3), "context_ids": torch.tensor([5000, 5001])}
    tensor_path, sidecar_path = EXTCAP.save_unit(
        store, "v_pairs_ext_p00_block0", payload, fp, elapsed_s=1.0, n_skipped=0
    )
    assert tensor_path.exists() and sidecar_path.exists()
    assert not tensor_path.with_suffix(".pt.tmp").exists()  # atomic rename cleaned up
    assert EXTCAP.unit_done(store, "v_pairs_ext_p00_block0", fp)  # exact match -> skip

    mutated = EXTCAP.unit_fingerprint("pairs", [5000, 5001], ["a" * 8, "CHANGED"], "srcsha", 8)
    assert not EXTCAP.unit_done(store, "v_pairs_ext_p00_block0", mutated)  # -> RECAPTURE

    other_src = EXTCAP.unit_fingerprint("pairs", [5000, 5001], ["a" * 8, "b" * 8], "othersha", 8)
    assert not EXTCAP.unit_done(store, "v_pairs_ext_p00_block0", other_src)

    tensor_path.unlink()  # sidecar without tensor is partial -> recapture
    assert not EXTCAP.unit_done(store, "v_pairs_ext_p00_block0", fp)


def test_own_chunk_resume_mutation_regenerates(tmp_path):
    own = tmp_path / "own"
    own.mkdir()
    fp = EXTCAP.own_chunk_fingerprint([5000, 5001], ["q one", "q two"])
    assert not EXTCAP.own_chunk_done(own, 0, fp)
    EXTGEN._write_jsonl(own / "own_chunk000.jsonl", [{"context_id": 5000}, {"context_id": 5001}])
    assert not EXTCAP.own_chunk_done(own, 0, fp)  # rows without meta = partial
    EXTCAP.write_json(own / "own_chunk000.meta.json", {"fingerprint": fp, "n_rows": 2})
    assert EXTCAP.own_chunk_done(own, 0, fp)
    mutated = EXTCAP.own_chunk_fingerprint([5000, 5001], ["q one", "q CHANGED"])
    assert not EXTCAP.own_chunk_done(own, 0, mutated)


# ── Gate B projection arithmetic ─────────────────────────────────────────────


def test_compute_capture_projection_thresholds():
    # 0.1 s/row over 43k remaining rows -> ~1.19 h < 2 * 2.0 h -> no abort
    ok = EXTCAP.compute_capture_projection(1.6, 16, 43_000, 2.0)
    assert ok["per_row_s"] == pytest.approx(0.1)
    assert ok["projected_wall_h"] == pytest.approx(0.1 * 43_000 / 3600)
    assert not ok["abort"]
    # 0.4 s/row -> ~4.78 h > 4.0 h threshold -> designed abort
    bad = EXTCAP.compute_capture_projection(6.4, 16, 43_000, 2.0)
    assert bad["abort"] and bad["abort_threshold_h"] == pytest.approx(4.0)
    # exact threshold is NOT an abort (strictly greater)
    edge = EXTCAP.compute_capture_projection(4.0 * 3600 / 43_000 * 16, 16, 43_000, 2.0)
    assert edge["projected_wall_h"] == pytest.approx(4.0) and not edge["abort"]


def test_run_gate_b_wall_abort_rc23(tmp_path, monkeypatch):
    """Behavioral Gate B halt: the REAL run_gate_b + compute_capture_projection
    bodies (pilot timing, projection arithmetic, abort report, DesignedHalt
    routing) with the GPU capture faked at the capture_cx boundary."""
    import time as _time

    layout = _tiny_layout(tmp_path, 0, 2)
    monkeypatch.setattr(EXTCAP, "capture_cx", lambda _m, _t, qs, _bs: _time.sleep(0.005))
    unit = {
        "name": "cx_ext_block0",
        "kind": "cx",
        "context_ids": [0, 1],
        "questions": [f"q{i}" for i in range(6)],
        "fingerprint": {},
    }

    # PASS arm: generous planned wall -> no abort, report returned.
    ok = EXTCAP.run_gate_b(None, FakeTokenizer(), [unit], 2, 100.0, layout.eval_dir)
    assert not ok["abort"] and ok["n_timed_rows"] == 4

    # HALT arm: planned_wall_h = 0 -> any positive projection > 2*0 -> rc 23.
    with pytest.raises(SystemExit) as exc:
        EXTCAP.run_gate_b(None, FakeTokenizer(), [unit], 2, 0.0, layout.eval_dir)
    assert exc.value.code == EXTCAP.RC_GATE_B_WALL == 23
    abort = json.loads((layout.eval_dir / "gate_b_abort_report.json").read_text())
    assert abort["rc"] == 23 and abort["verdict"] == "DESIGNED-ABORT"
    assert abort["pilot"]["abort"]
    _assert_no_sentinels(layout)


# ── P-OwnGen: sentinel is written ONLY after the data upload verifies ─────────


def _owngen_env(tmp_path, monkeypatch, upload_results: list):
    """Drive the REAL phase_owngen body (staging seams + GPU + Hub faked):
    pre-written chunks make `pending` empty so no vLLM engine is needed."""
    import types

    import transformers

    # phase_owngen loads via the module-level N_PREFIX, so the tiny fixture
    # must live at the REAL prefix grain: 4 ext contexts at N_PREFIX..N_PREFIX+3.
    n_lo, n_hi = EXTCAP.N_PREFIX, EXTCAP.N_PREFIX + 4
    layout = _tiny_layout(tmp_path, n_lo, n_hi)
    layout.eval_dir.mkdir(parents=True, exist_ok=True)
    EXTCAP.write_json(layout.eval_dir / EXTCAP.P0_REPORT, {"pass": True})

    ext_base = tmp_path / "ext_local"
    _write_ext_files(ext_base, _tiny_ext_by_persona(n_lo, n_hi))
    monkeypatch.setattr(
        EXTCAP,
        "stage_ext_gen_inputs",
        lambda _layout, _p, _r, _l: (ext_base, {"complete": True}, "local:test"),
    )
    monkeypatch.setattr(EXTCAP, "assert_out_root_headroom", lambda *a, **k: None)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: FakeTokenizer())
    )

    # Pre-complete both chunks (chunk_size=2 over 4 ext contexts) so the
    # generate branch (vLLM) is skipped and the upload tail runs for real.
    tok = FakeTokenizer()
    layout.own_dir.mkdir(parents=True, exist_ok=True)
    ctx_q = [(r["context_id"], r["question"]) for r in _tiny_ext_by_persona(n_lo, n_hi)[0]]
    for c, chunk in enumerate([ctx_q[:2], ctx_q[2:]]):
        rows = [
            {
                "context_id": i,
                "question": q,
                "own_text": "alpha beta gamma delta",
                "finish_reason": "stop",
                "skipped_reason": None,
            }
            for i, q in chunk
        ]
        EXTGEN._write_jsonl(layout.own_dir / f"own_chunk{c:03d}.jsonl", rows)
        fp = EXTCAP.own_chunk_fingerprint([i for i, _q in chunk], [q for _i, q in chunk])
        EXTCAP.write_json(
            layout.own_dir / f"own_chunk{c:03d}.meta.json", {"fingerprint": fp, "n_rows": 2}
        )
    del tok

    sentinel_path = layout.own_dir / EXTCAP.OWNGEN_SENTINEL
    calls: list[dict] = []

    def fake_upload(**kwargs):
        calls.append(
            {
                "path_in_repo": kwargs["path_in_repo"],
                "allow_patterns": list(kwargs["allow_patterns"]),
                "sentinel_exists": sentinel_path.exists(),
            }
        )
        return upload_results[len(calls) - 1]

    monkeypatch.setattr(EXTCAP, "_upload_folder_filtered", fake_upload)
    args = types.SimpleNamespace(
        ext_prefix="unused", ext_revision=None, ext_local_dir=None, own_chunk_size=2
    )
    return layout, args, calls, sentinel_path


def test_phase_owngen_sentinel_only_after_verified_upload(tmp_path, monkeypatch):
    path_in_repo = f"{EXTCAP.HF_PREFIX}/raw_completions/ladder_ext_own"
    canonical = f"{EXTCAP.DATA_REPO}/{path_in_repo}"
    layout, args, calls, sentinel_path = _owngen_env(
        tmp_path, monkeypatch, upload_results=[canonical, canonical]
    )
    EXTCAP.phase_owngen(args, layout)
    # Call 1 = data upload BEFORE any sentinel exists; call 2 = sentinel-only.
    assert len(calls) == 2
    assert not calls[0]["sentinel_exists"]
    assert calls[1]["sentinel_exists"]
    assert calls[1]["allow_patterns"] == [EXTCAP.OWNGEN_SENTINEL]
    assert EXTCAP.OWNGEN_SENTINEL not in calls[0]["allow_patterns"]
    sentinel = json.loads(sentinel_path.read_text())
    assert sentinel["complete"] and sentinel["phase"] == "owngen"
    assert sentinel["smoke"] is False
    assert EXTCAP._require_own_complete(layout)  # downstream loader accepts


def test_phase_owngen_failed_upload_leaves_no_sentinel(tmp_path, monkeypatch):
    layout, args, calls, sentinel_path = _owngen_env(
        tmp_path, monkeypatch, upload_results=[None, None]
    )
    with pytest.raises(RuntimeError, match="own-rollout upload"):
        EXTCAP.phase_owngen(args, layout)
    assert len(calls) == 1  # sentinel-only upload never attempted
    assert not sentinel_path.exists()
    with pytest.raises(RuntimeError, match="missing"):
        EXTCAP._require_own_complete(layout)


# ── precompute_ext_rows: skip_mask + truncation semantics ────────────────────


def test_precompute_ext_rows_skip_and_truncation(tmp_path):
    tok = FakeTokenizer()
    rows = [
        _ext_record(5000, 0),  # 4-word answer, ext span = 3 under the fake render
        _ext_record(5001, 0, filled=False, answer_text="", validity="refusal"),
        _ext_record(5002, 0, answer_text="one"),  # 1-word answer -> ext span 0 -> empty_span
        _ext_record(5003, 0),
    ]
    own = {5000: 2, 5001: 5, 5002: 4, 5003: 0}
    pre = EXTCAP.precompute_ext_rows(tok, rows, own)

    r0 = pre[0]
    assert r0["skip_reason"] is None and r0["pair_len"] == 3
    assert r0["trunc_len"] == 2 and r0["truncated"] and r0["dropped_tokens"] == 1
    assert r0["expected_span"] == 2

    assert pre[1]["skip_reason"] == "not_filled" and pre[1]["expected_span"] == 0
    assert pre[2]["skip_reason"] == "empty_span" and pre[2]["expected_span"] == 0
    # own_len == 0 => no truncation (parent parity)
    assert pre[3]["trunc_len"] == 3 and not pre[3]["truncated"]

    with pytest.raises(RuntimeError, match="own_len_ext has no entry"):
        EXTCAP.precompute_ext_rows(tok, [_ext_record(5004, 0)], own)


# ── Batched cx capture: batched-vs-serial equivalence (tiny real model) ──────


def test_cx_from_token_ids_batched_matches_serial():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    ids_list = [
        [100, 101, 102, 103, 104],
        [200, 201],  # short row forces LEFT padding + position_ids correctness
        [300, 301, 302, 303, 304, 305, 306],
        [400],
    ]
    batched = EXTCAP._cx_from_token_ids(model, ids_list, [0, 1], batch_size=4, pad_id=0)
    serial = EXTCAP._cx_from_token_ids(model, ids_list, [0, 1], batch_size=1, pad_id=0)
    assert batched.shape == (4, 2, 32) and batched.dtype.name == "float32"
    for j in range(4):
        a = torch.from_numpy(batched[j]).flatten()
        b = torch.from_numpy(serial[j]).flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        assert cos >= 0.999, (j, cos)
    assert not (batched == 0).all()


# ── Staging integrity halt (rc 18) ───────────────────────────────────────────


def test_stage_ext_gen_inputs_halts_rc18(tmp_path):
    layout = _tiny_layout(tmp_path, 4, 8)
    local = tmp_path / "ext_local"
    local.mkdir()
    with pytest.raises(SystemExit) as exc:  # sentinel missing entirely
        EXTCAP.stage_ext_gen_inputs(layout, "prefix", None, local)
    assert exc.value.code == EXTCAP.RC_STAGING_INTEGRITY == 18

    EXTCAP.write_json(local / EXTGEN.SENTINEL_FILENAME, {"complete": False})
    with pytest.raises(SystemExit) as exc:
        EXTCAP.stage_ext_gen_inputs(layout, "prefix", None, local)
    assert exc.value.code == 18

    payload = local / "assignment_ext.json"
    EXTCAP.write_json(payload, {"arms": {}})
    EXTCAP.write_json(
        local / EXTGEN.SENTINEL_FILENAME,
        {"complete": True, "files_sha256": {"assignment_ext.json": "0" * 64}},
    )
    with pytest.raises(SystemExit) as exc:  # staged sha mismatch
        EXTCAP.stage_ext_gen_inputs(layout, "prefix", None, local)
    assert exc.value.code == 18
    _assert_no_sentinels(layout)

    good_sha = EXTCAP._sha256_file(payload)
    EXTCAP.write_json(
        local / EXTGEN.SENTINEL_FILENAME,
        {"complete": True, "files_sha256": {"assignment_ext.json": good_sha}},
    )
    base, sentinel, rev = EXTCAP.stage_ext_gen_inputs(layout, "prefix", None, local)
    assert base == local and sentinel["complete"] and rev.startswith("local:")


# ── Layout + rc table sanity ─────────────────────────────────────────────────


def test_layout_smoke_isolation_and_rc_table():
    prod = EXTCAP.Layout(pathlib.Path("/tmp/x"), smoke=False, n_ext=EXTCAP.N_EXT_FULL)
    assert prod.n_total == 48_000 and prod.rungs == (5_000, 12_000, 24_000, 48_000)
    assert prod.store_subpath == "analysis_tensors/ext"
    assert prod.own_subpath == "raw_completions/ladder_ext_own"
    assert prod.gates_subpath == EXTGEN.HF_EXT_SUBPATH

    smoke = EXTCAP.Layout(pathlib.Path("/tmp/x"), smoke=True, n_ext=16)
    assert smoke.n_total == 5_016 and smoke.rungs == (5_000, 5_016)
    assert smoke.store_subpath == "analysis_tensors/ext_smoke"
    assert smoke.own_subpath == "raw_completions/ladder_ext_own_smoke"

    # rc 23 (NOT 4): the ext-gen driver inherits the parent's rc 4/5, so the
    # capture driver's Gate B wall rc was renumbered off the collision (r1
    # concern round-rc4-collision; disjointness pinned cross-driver in
    # tests/test_issue823_ext_fits.py).
    assert set(EXTCAP.RC_TABLE) == {"0", "23", "12", "13", "14", "15", "16", "17", "18"}
    assert EXTCAP.RC_GATE_B_WALL == 23
    assert set(EXTCAP.PHASES) == set(EXTCAP.PHASE_NAMES)


def test_pass_b_loader_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bundle.pt"
    torch.save({"unexpected": torch.zeros(2)}, str(bad))
    with pytest.raises(RuntimeError, match="cx_last"):
        EXTCAP.load_pass_b_cx_last(bad)
    wrong_shape = tmp_path / "bundle2.pt"
    torch.save({"cx_last": torch.zeros(3, 2, 2)}, str(wrong_shape))
    with pytest.raises(AssertionError):
        EXTCAP.load_pass_b_cx_last(wrong_shape)
