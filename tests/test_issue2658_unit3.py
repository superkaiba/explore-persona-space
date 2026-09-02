"""#2658 unit-3 tests: text resolver pins, generation retry/cap-hit accounting,
manifest-row validity, immutable manifests, and the L19 capture span/mean.

Fully offline: no network, no GPU, no live HF fetch, no bank item text (the
resolver tests monkeypatch the bank loader with synthetic placeholder strings).
Every guard is exercised on its RAISE branch as well as its happy path.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2658_capture as K  # noqa: E402
import issue2658_common as C  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_generate as G  # noqa: E402
import issue2658_text_resolver as R  # noqa: E402


def _hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Resolver: item-id parsing.
# ---------------------------------------------------------------------------
def test_parse_item_id_roundtrip():
    row, frame, ref = R.parse_item_id("evil|advbench_requests|advbench#3")
    assert (row, frame, ref) == ("evil", "advbench_requests", "advbench#3")


@pytest.mark.parametrize(
    "bad",
    [
        "evil|advbench_requests",  # 2 parts
        "nosuchrow|advbench_requests|advbench#0",  # unknown row
        "evil|nosuchframe|advbench#0",  # unknown frame
        "evil|advbench_requests|",  # empty ref
    ],
)
def test_parse_item_id_malformed_raises(bad):
    with pytest.raises(R.TextResolutionError):
        R.parse_item_id(bad)


# ---------------------------------------------------------------------------
# Resolver: frozen pin table (offline via monkeypatched PIN_PATH).
# ---------------------------------------------------------------------------
_IID = "evil|advbench_requests|advbench#0"
_IID1 = "evil|advbench_requests|advbench#1"


def _write_pins(path: Path, items: dict, sha_domain: str = R.SHA_DOMAIN) -> None:
    path.write_text(json.dumps({"sha_domain": sha_domain, "items": items}))


def _patch_bank(monkeypatch, texts: list[str]) -> None:
    from explore_persona_space.artifacts import banks

    monkeypatch.setattr(banks, "load_bank", lambda name: list(texts))


def test_load_pins_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "PIN_PATH", tmp_path / "absent.json")
    with pytest.raises(R.TextResolutionError, match="frozen pin table missing"):
        R.load_pins()


def test_load_pins_wrong_domain_raises(monkeypatch, tmp_path):
    p = tmp_path / "pins.json"
    _write_pins(p, {}, sha_domain="file bytes")
    monkeypatch.setattr(R, "PIN_PATH", p)
    with pytest.raises(C.RowHashMismatchError, match="sha_domain"):
        R.load_pins()


def test_resolve_items_pin_match_and_mismatch(monkeypatch, tmp_path):
    texts = ["synthetic placeholder zero", "synthetic placeholder one"]
    _patch_bank(monkeypatch, texts)
    pins = tmp_path / "pins.json"
    monkeypatch.setattr(R, "PIN_PATH", pins)

    # Matching pin: resolution succeeds and carries the text-domain sha.
    _write_pins(pins, {_IID: {"prompt_sha256": _hex(texts[0]), "source_ref": "x"}})
    out = R.resolve_items([_IID])
    assert out[_IID].prompt_sha256 == _hex(texts[0])
    assert out[_IID].text == texts[0]

    # Drifted pin: RowHashMismatchError (never a silent substitute).
    _write_pins(pins, {_IID: {"prompt_sha256": _hex("different"), "source_ref": "x"}})
    with pytest.raises(C.RowHashMismatchError):
        R.resolve_items([_IID])

    # Unpinned item: loud failure naming the gap.
    _write_pins(pins, {})
    with pytest.raises(R.TextResolutionError, match="no frozen pin"):
        R.resolve_items([_IID])


def test_resolve_items_bank_index_out_of_range(monkeypatch):
    _patch_bank(monkeypatch, ["only one item"])
    with pytest.raises(R.TextResolutionError, match="out of range"):
        R.resolve_items([_IID1], verify_pins=False)


def test_resolve_items_duplicate_ids_raise():
    with pytest.raises(R.TextResolutionError, match="duplicate"):
        R.resolve_items([_IID, _IID], verify_pins=False)


def test_freeze_pins_idempotent_then_drift_raises(monkeypatch, tmp_path):
    pins = tmp_path / "pins.json"
    monkeypatch.setattr(R, "PIN_PATH", pins)
    triples = [("evil", "advbench_requests|band_a", _IID)]
    monkeypatch.setattr(R, "pilot_item_ids", lambda: triples)

    def fake_resolve(ids, *, verify_pins=True):
        return {
            i: R.ResolvedItem(item_id=i, prompt_sha256=_hex("v1:" + i), source_ref="test", text="t")
            for i in ids
        }

    monkeypatch.setattr(R, "resolve_items", fake_resolve)
    body = R.freeze_pins()
    assert pins.exists() and body["n_items"] == 1
    # Idempotent re-freeze verifies and returns the existing table.
    again = R.freeze_pins()
    assert again["items"] == body["items"]

    # A drifted resolution refuses to overwrite the frozen table.
    def drifted_resolve(ids, *, verify_pins=True):
        return {
            i: R.ResolvedItem(item_id=i, prompt_sha256=_hex("v2:" + i), source_ref="test", text="t")
            for i in ids
        }

    monkeypatch.setattr(R, "resolve_items", drifted_resolve)
    with pytest.raises(C.RowHashMismatchError, match="drift"):
        R.freeze_pins()


def test_freeze_pins_refuses_partial_rows():
    with pytest.raises(R.TextResolutionError, match="FULL registered pilot"):
        R.freeze_pins(rows=["evil"])


def test_vendor_pins_verify():
    paths = R.verify_vendor_pins()
    assert set(paths) == {"issue2388_gen", "issue2388_spread_pilot"}
    for p in paths.values():
        assert p.exists()


# ---------------------------------------------------------------------------
# Resolver: evidence packets (frozen-file contract).
# ---------------------------------------------------------------------------
def test_resolve_evidence_packet_contract(monkeypatch, tmp_path):
    ev = tmp_path / "evidence.json"
    monkeypatch.setattr(R, "EVIDENCE_PATH", ev)
    iid = "sycophancy|x|y#0"

    with pytest.raises(ValueError, match="does not use evidence"):
        R.resolve_evidence_packet("evil", iid)

    with pytest.raises(R.EvidencePacketMissingError, match="not built"):
        R.resolve_evidence_packet("sycophancy", iid)

    packet = {"claim_id": "c1", "n_sources": 2}
    sha = R.evidence_packet_sha256(packet)
    ev.write_text(json.dumps({"items": {iid: {"packet": packet, "evidence_sha256": sha}}}))
    got, got_sha = R.resolve_evidence_packet("sycophancy", iid)
    assert got == packet and got_sha == sha

    ev.write_text(
        json.dumps({"items": {iid: {"packet": packet, "evidence_sha256": _hex("wrong")}}})
    )
    with pytest.raises(C.RowHashMismatchError, match="drift"):
        R.resolve_evidence_packet("sycophancy", iid)


# ---------------------------------------------------------------------------
# Generation: empty-output retry schedule.
# ---------------------------------------------------------------------------
def test_empty_retry_seed_deterministic_distinct_bounded():
    s1 = G.empty_retry_seed("p", 0, 1)
    assert s1 == G.empty_retry_seed("p", 0, 1)  # deterministic
    seeds = {G.empty_retry_seed("p", 0, a) for a in (1, 2, 3)}
    assert len(seeds) == 3  # attempts are distinct slots
    assert all(0 <= s < 2**31 for s in seeds)
    assert C.response_seed("p", 0) not in seeds  # never collides with the schedule slot
    for bad in (0, 4):
        with pytest.raises(ValueError):
            G.empty_retry_seed("p", 0, bad)


def _mk_gen_once(nonempty_seeds: set[int], calls: list[int]):
    def gen_once(seed: int) -> dict:
        calls.append(seed)
        ok = seed in nonempty_seeds
        return {"text": "x" if ok else "", "token_ids": [7] if ok else [], "finish_reason": "stop"}

    return gen_once


def test_generate_with_empty_retry_happy_path():
    sched = C.response_seed("p", 3)
    calls: list[int] = []
    out, seed, ledger = G.generate_with_empty_retry(_mk_gen_once({sched}, calls), "p", 3)
    assert seed == sched and ledger == [] and calls == [sched]
    assert out["token_ids"] == [7]


def test_generate_with_empty_retry_recovers_on_second_retry():
    sched = C.response_seed("p", 3)
    r2 = G.empty_retry_seed("p", 3, 2)
    calls: list[int] = []
    out, seed, ledger = G.generate_with_empty_retry(_mk_gen_once({r2}, calls), "p", 3)
    assert seed == r2
    assert calls == [sched, G.empty_retry_seed("p", 3, 1), r2]  # fixed schedule order
    assert [row["outcome"] for row in ledger] == ["empty", "nonempty"]
    assert [row["retry_seed"] for row in ledger] == calls[1:]
    assert out["token_ids"] == [7]


def test_generate_with_empty_retry_exhaustion_raises():
    calls: list[int] = []
    with pytest.raises(G.EmptyOutputError, match="zero-token"):
        G.generate_with_empty_retry(_mk_gen_once(set(), calls), "p", 0)
    assert len(calls) == 1 + G.N_EMPTY_RETRIES  # schedule seed + 3 fixed retries


# ---------------------------------------------------------------------------
# Generation: cap-hit accounting (strictly > 2% arms the amendment).
# ---------------------------------------------------------------------------
def _cap_rows(row: str, cell: str, n: int, n_length: int) -> list[dict]:
    return [
        {"row": row, "cell": cell, "finish_reason": "length" if i < n_length else "stop"}
        for i in range(n)
    ]


def test_cap_hit_report_strict_threshold():
    rows = _cap_rows("evil", "a", 50, 1) + _cap_rows("evil", "b", 50, 3)
    rep = G.cap_hit_report(rows)
    assert rep["per_cell_fraction"]["evil|a"] == pytest.approx(0.02)
    assert rep["per_cell_fraction"]["evil|b"] == pytest.approx(0.06)
    # Exactly 2% does NOT trigger (strictly-above); 6% does.
    assert set(rep["cells_over_threshold"]) == {"evil|b"}
    assert rep["amendment_required"] is True
    assert rep["per_row_fraction"]["evil"] == pytest.approx(4 / 100)


def test_cap_hit_report_clean_and_empty():
    rep = G.cap_hit_report(_cap_rows("math", "a", 10, 0))
    assert rep["amendment_required"] is False and rep["cells_over_threshold"] == {}
    with pytest.raises(ValueError, match="zero records"):
        G.cap_hit_report([])


# ---------------------------------------------------------------------------
# Generation: manifest rows + immutable JSON.
# ---------------------------------------------------------------------------
def test_build_manifest_row_passes_unit1_validator(monkeypatch):
    iid = "evil|advbench_requests|advbench#0"
    monkeypatch.setattr(G, "_PIN_CACHE", {iid: _hex("prompt text")})
    ans = _hex("an answer")
    d = G.build_manifest_row(
        row="evil",
        item_id=iid,
        superfamily_id="sf-evil-0",
        frame="advbench_requests",
        band="direct",
        split="pilot",
        response_index=2,
        answer_sha256=ans,
        raw_text_sha256=_hex("raw"),
    )
    assert d["seed"] == C.response_seed(iid, 2)  # draw-slot schedule seed, pinned
    assert d["judge_status"] == "pending" and len(d["judge_draw_ids"]) > 0
    assert d["vector_sha256"] is None and d["evidence_sha256"] is None

    iid_m = "correctness_math|math_algebra|mathfull-x"
    monkeypatch.setattr(G, "_PIN_CACHE", {iid_m: _hex("p")})
    d2 = G.build_manifest_row(
        row="correctness_math",
        item_id=iid_m,
        superfamily_id="sf-math-0",
        frame="math_algebra",
        band="level_low",
        split="pilot",
        response_index=0,
        answer_sha256=ans,
        raw_text_sha256=_hex("raw2"),
    )
    assert d2["judge_status"] == "objective" and d2["judge_draw_ids"] == []


def test_write_immutable_json_drift_raises(tmp_path):
    p = tmp_path / "order.json"
    body = {"a": 1, "requests_sha256": _hex("r")}
    G.write_immutable_json(p, body)
    G.write_immutable_json(p, dict(body))  # byte-identical rewrite is a no-op
    with pytest.raises(G.OrderManifestDriftError, match="drift"):
        G.write_immutable_json(p, {"a": 2, "requests_sha256": _hex("r")})


# ---------------------------------------------------------------------------
# Capture: answer-span arithmetic (specials excluded; empty span raises).
# ---------------------------------------------------------------------------
def test_answer_positions_excludes_specials():
    pos = {"answer_start": 5, "answer_end": 9}
    kept, n_exc = K.answer_positions_nonspecial(pos, [10, 99, 11, 99], frozenset({99}))
    assert kept == [5, 7] and n_exc == 2


def test_answer_positions_empty_after_exclusion_raises():
    pos = {"answer_start": 5, "answer_end": 7}
    with pytest.raises(K.CaptureSpanError, match="empty after special-token exclusion"):
        K.answer_positions_nonspecial(pos, [99, 99], frozenset({99}))


def test_answer_positions_zero_token_answer_raises():
    with pytest.raises(K.CaptureSpanError, match="zero-token"):
        K.answer_positions_nonspecial({"answer_start": 5, "answer_end": 5}, [], frozenset())


def test_answer_positions_clamped_span_raises():
    pos = {"answer_start": 5, "answer_end": 7}  # width 2 vs 3 completion ids
    with pytest.raises(K.CaptureSpanError, match="span width"):
        K.answer_positions_nonspecial(pos, [10, 11, 12], frozenset())


# ---------------------------------------------------------------------------
# Capture: fake-model forward — peer-centering-free by construction.
# ---------------------------------------------------------------------------
class _FakeTok:
    """Char-level fake tokenizer (ids >= 2; pad=0 special; right padding)."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"
    all_special_ids: ClassVar[list[int]] = [0]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 900 + 2 for c in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        ids = self.encode(text)
        return {"input_ids": ids, "offset_mapping": [(i, i + 1) for i in range(len(ids))]}

    def pad(self, enc, return_tensors="pt", padding=True):
        import torch

        seqs = enc["input_ids"]
        m = max(len(s) for s in seqs)
        ids = torch.tensor([s + [self.pad_token_id] * (m - len(s)) for s in seqs])
        mask = torch.tensor([[1] * len(s) + [0] * (m - len(s)) for s in seqs])
        return {"input_ids": ids, "attention_mask": mask}


class _FakeModel:
    """hidden_states[l][b, t, :] = input_ids[b, t] * (l + 1): each position is a
    pure function of its OWN token id — any peer influence would show up as a
    vector change under a different batch composition."""

    def __init__(self, n_blocks=3, hidden=4):
        self.n_blocks, self.hidden = n_blocks, hidden

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=True):
        import torch

        b, t = input_ids.shape
        hs = tuple(
            (input_ids.to(torch.float32) * float(layer + 1))
            .unsqueeze(-1)
            .expand(b, t, self.hidden)
            .contiguous()
            for layer in range(self.n_blocks + 1)
        )
        return SimpleNamespace(hidden_states=hs)


def _rows(*specs):
    return [
        K.CaptureRow(
            prompt_id=f"p{i}",
            response_index=0,
            answer_sha256=_hex(ans),
            rendered_prompt=prompt,
            answer_text=ans,
        )
        for i, (prompt, ans) in enumerate(specs)
    ]


def _capture(rows, batch_size):
    return K.capture_l19_answer_means(
        rows,
        model=_FakeModel(),
        tokenizer=_FakeTok(),
        device="cpu",
        batch_size=batch_size,
        layer=1,
        n_blocks=3,
        hidden=4,
        log_label="test",
    )


def test_capture_mean_matches_analytic_expectation():
    import numpy as np

    rows = _rows(("hello", "ab"))
    vecs, metas = _capture(rows, batch_size=1)
    ids = _FakeTok().encode("ab")
    # hidden_states[1:][layer=1] is hs[2] -> multiplier (2 + 1) = 3.
    expected = float(np.mean(ids)) * 3.0
    assert vecs[0].shape == (4,)
    assert vecs[0] == pytest.approx(np.full(4, expected, dtype="<f4"))
    assert metas[0]["n_answer_tokens_kept"] == 2 and metas[0]["n_special_excluded"] == 0


def test_capture_peer_and_batch_composition_invariance():
    import numpy as np

    a = ("short prompt", "yes indeed")
    b = ("a much longer prompt to force padding differences", "no")
    solo, _ = _capture(_rows(a), batch_size=1)
    paired, _ = _capture(_rows(a, b), batch_size=2)
    swapped, _ = _capture(_rows(b, a), batch_size=2)
    np.testing.assert_array_equal(solo[0], paired[0])  # peers cannot move the vector
    np.testing.assert_array_equal(paired[0], swapped[1])  # order-invariant
    np.testing.assert_array_equal(paired[1], swapped[0])


def test_capture_refuses_left_padding():
    tok = _FakeTok()
    tok.padding_side = "left"
    with pytest.raises(K.CaptureSpanError, match="RIGHT padding"):
        K.capture_l19_answer_means(
            _rows(("p", "a")),
            model=_FakeModel(),
            tokenizer=tok,
            device="cpu",
            batch_size=1,
            layer=1,
            n_blocks=3,
            hidden=4,
        )


def test_capture_wrong_block_count_raises():
    with pytest.raises(K.CaptureSpanError, match="post-block states"):
        K.capture_l19_answer_means(
            _rows(("p", "a")),
            model=_FakeModel(n_blocks=2),
            tokenizer=_FakeTok(),
            device="cpu",
            batch_size=1,
            layer=1,
            n_blocks=3,
            hidden=4,
        )


def test_vector_sha256_domain_is_le_float32_c_order():
    import numpy as np

    vec = np.arange(4, dtype=np.float64) + 0.5
    expected = hashlib.sha256(vec.astype("<f4").tobytes()).hexdigest()
    assert K.vector_sha256(vec) == expected


# ---------------------------------------------------------------------------
# Frame-manifest integration (uses the committed manifest when present).
# ---------------------------------------------------------------------------
def test_pilot_item_ids_from_committed_manifest():
    if not F.FRAME_MANIFEST_PATH.exists():
        pytest.skip("frame manifest not built in this checkout")
    triples = R.pilot_item_ids()
    assert len(triples) == len({t[2] for t in triples})  # no duplicate item_ids
    body = json.loads(F.FRAME_MANIFEST_PATH.read_text())
    for row, _cell, iid in triples[:5]:
        assert R.superfamily_of(body, row, iid)


# ---------------------------------------------------------------------------
# Group-B fix round (resume/teardown blockers + minors) — one pin per fix.
# ---------------------------------------------------------------------------
def _store_shard(tmp_path: Path, metas: list[dict], fingerprint: str = "fp-test") -> Path:
    import numpy as np

    store = tmp_path / "store"
    vectors = [np.zeros(C.HIDDEN, dtype="<f4") for _ in metas]
    K.write_shard(store, 0, vectors, metas, fingerprint)
    return store


def _crow(pid: str, k: int, sha: str) -> K.CaptureRow:
    return K.CaptureRow(prompt_id=pid, response_index=k, answer_sha256=sha)


# Fix 1 (blocker): capture resume binds KEYS AND CONTENT (answer_sha256).
def test_resume_completed_shards_content_stale_raises(tmp_path):
    metas = [
        {"prompt_id": "p0", "response_index": 0, "answer_sha256": _hex("old text 0")},
        {"prompt_id": "p0", "response_index": 1, "answer_sha256": _hex("old text 1")},
    ]
    store = _store_shard(tmp_path, metas)

    # Same keys, same content: the shard resumes.
    rows_ok = [_crow("p0", 0, _hex("old text 0")), _crow("p0", 1, _hex("old text 1"))]
    assert K.resume_completed_shards(store, rows_ok, "fp-test") == (2, 1)

    # Regenerated text: SAME keys, different answer_sha256 -> CacheStaleError
    # (the pre-fix validator compared keys only and resumed a stale vector).
    rows_regen = [_crow("p0", 0, _hex("old text 0")), _crow("p0", 1, _hex("REGENERATED"))]
    with pytest.raises(C.CacheStaleError, match="CONTENT"):
        K.resume_completed_shards(store, rows_regen, "fp-test")

    # Key mismatch keeps the pre-existing refusal.
    rows_keys = [_crow("p9", 0, _hex("old text 0")), _crow("p0", 1, _hex("old text 1"))]
    with pytest.raises(C.CacheStaleError, match="expected prefix"):
        K.resume_completed_shards(store, rows_keys, "fp-test")

    # A foreign fingerprint resumes nothing (shard_done gates as before).
    assert K.resume_completed_shards(store, rows_ok, "other-fp") == (0, 0)


# Fix 2 (blocker): a resumed gen cell idempotently rewrites its manifest —
# the raw/manifest PAIR is not atomic, so a kill between the two writes must
# not strand the cell manifest-less forever.
def test_resume_cell_rewrites_missing_or_stale_manifest(tmp_path, monkeypatch):
    from explore_persona_space.atomic_io import write_jsonl_atomic

    iid = "evil|advbench_requests|advbench#0"
    monkeypatch.setattr(G, "_PIN_CACHE", {iid: _hex("prompt text")})
    cw = G.CellWork(
        row="evil",
        frame="advbench_requests",
        band="direct",
        item_ids=(iid,),
        superfamilies={iid: "sf-evil-0"},
    )
    body = {
        "fingerprint": "fp-test",
        "split": "pilot",
        "records": [
            {
                "prompt_id": iid,
                "response_index": k,
                "answer_sha256": _hex(f"ans{k}"),
                "raw_text_sha256": _hex(f"ans{k}"),
            }
            for k in range(2)
        ],
    }
    raw = tmp_path / "raw_completions" / "pilot" / f"{cw.name}.json"
    man = tmp_path / "gen_manifest" / "pilot" / f"{cw.name}.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps(body, ensure_ascii=False))

    # The crash window: raw written, manifest MISSING. Resume rewrites it.
    got = G.resume_cell_with_manifest(raw, man, cw, "fp-test", 2)
    assert got == body and man.exists()
    fresh = tmp_path / "fresh.jsonl"
    write_jsonl_atomic(fresh, G.manifest_rows_for_cell(cw, body))
    assert man.read_bytes() == fresh.read_bytes()  # identical to the fresh-path write

    # Idempotent: a stale/corrupt manifest is REPLACED on resume.
    man.write_text("stale\n")
    G.resume_cell_with_manifest(raw, man, cw, "fp-test", 2)
    assert man.read_bytes() == fresh.read_bytes()

    # A non-resumable cell (absent raw) returns None and writes nothing.
    raw2 = raw.parent / "absent.json"
    man2 = man.parent / "absent.jsonl"
    assert G.resume_cell_with_manifest(raw2, man2, cw, "fp-test", 2) is None
    assert not man2.exists()


# Fix 3 (major): dtype + device CLASS are output-affecting fingerprint keys.
def test_capture_fingerprint_keys_on_dtype_and_device_class():
    fp = K.capture_fingerprint("pilot", dtype="bfloat16", device="cuda")
    # Device INDEX is placement, not a regime key; CLASS and dtype are.
    assert K.capture_fingerprint("pilot", dtype="bfloat16", device="cuda:1") == fp
    assert K.capture_fingerprint("pilot", dtype="float32", device="cuda") != fp
    assert K.capture_fingerprint("pilot", dtype="bfloat16", device="cpu") != fp
    assert K.capture_fingerprint("dev", dtype="bfloat16", device="cuda") != fp
    assert K.device_class("cuda:3") == "cuda" and K.device_class("cpu") == "cpu"
    with pytest.raises(TypeError):
        K.capture_fingerprint("pilot")  # the dtype/device-blind signature is retired


# Fix 4 (major): a raise under a live vLLM engine hard-exits WITH its traceback
# instead of entering interpreter finalization (#1739/#2149 deadlock class).
def test_exit_hard_under_live_engine_prints_chain_then_hard_exits(monkeypatch, capsys):
    import os as _os

    codes: list[int] = []
    monkeypatch.setattr(_os, "_exit", lambda code: codes.append(code))
    try:
        raise ValueError("boom-2658-fix4")
    except ValueError as exc:
        G.exit_hard_under_live_engine(exc)
    assert codes == [1]
    err = capsys.readouterr().err
    assert "Traceback" in err and "boom-2658-fix4" in err


def test_run_routes_post_engine_raises_to_hard_exit():
    import ast
    import inspect

    src = inspect.getsource(G)
    assert "engine_live" not in src  # the dead always-True flag is gone
    tree = ast.parse(src)
    run_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    handlers = [
        h
        for n in ast.walk(run_fn)
        if isinstance(n, ast.Try)
        for h in n.handlers
        if isinstance(h.type, ast.Name) and h.type.id == "BaseException"
    ]
    assert handlers, "run() must catch BaseException while the engine is live"
    handler_calls = {
        n.func.id
        for h in handlers
        for n in ast.walk(h)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "exit_hard_under_live_engine" in handler_calls


# Fix 5 (minor): freeze_pins routes through the shared atomic writer — the
# fixed-name ``prompt_pins.tmp.json`` + os.replace shape (#2329 class) is gone.
def test_freeze_pins_uses_shared_atomic_writer(monkeypatch, tmp_path):
    from explore_persona_space import atomic_io

    pins = tmp_path / "pins.json"
    monkeypatch.setattr(R, "PIN_PATH", pins)
    monkeypatch.setattr(R, "pilot_item_ids", lambda: [("evil", "advbench_requests|band_a", _IID)])

    def fake_resolve(ids, *, verify_pins=True):
        return {
            i: R.ResolvedItem(item_id=i, prompt_sha256=_hex("v1:" + i), source_ref="test", text="t")
            for i in ids
        }

    monkeypatch.setattr(R, "resolve_items", fake_resolve)

    seen: list[Path] = []

    def spy(path, text, **kw):
        seen.append(Path(path))
        return atomic_io.write_text_atomic(path, text, **kw)

    monkeypatch.setattr(R, "write_text_atomic", spy)  # AttributeError on pre-fix module
    body = R.freeze_pins()
    assert seen == [pins] and pins.exists() and body["n_items"] == 1
    assert not pins.with_suffix(".tmp.json").exists()  # no fixed-name tmp residue


# Fix 6 (minor): the completeness anchor is the FRAME MANIFEST x n_responses —
# anchoring on the loaded gen files is circular (incomplete gen reads complete).
def _triples() -> list[tuple[str, str, str]]:
    return [
        ("evil", "advbench_requests|band_a", "evil|advbench_requests|advbench#0"),
        ("evil", "advbench_requests|band_a", "evil|advbench_requests|advbench#1"),
        ("math", "algebra|level_low", "math|algebra|m#0"),
    ]


def test_expected_capture_keys_manifest_anchor(monkeypatch):
    monkeypatch.setattr(R, "pilot_item_ids", _triples)
    keys = K.expected_capture_keys(None, 2, 0, 1)
    assert keys == sorted((iid, k) for _, _, iid in _triples() for k in range(2))
    # Row filter + shard slicing compose exactly like the gen-row sharding.
    evil = sorted((iid, k) for row, _, iid in _triples() if row == "evil" for k in range(2))
    assert K.expected_capture_keys(["evil"], 2, 0, 2) == evil[0::2]
    assert K.expected_capture_keys(["evil"], 2, 1, 2) == evil[1::2]
    # Smoke restriction: only gen-present cells stay in the anchor.
    only_math = K.expected_capture_keys(None, 2, 0, 1, present_cells={"math__algebra__level_low"})
    assert only_math == [("math|algebra|m#0", 0), ("math|algebra|m#0", 1)]
    with pytest.raises(K.CaptureSpanError, match="zero expected"):
        K.expected_capture_keys(["nonexistent-row"], 2, 0, 1)


def test_capture_completeness_anchor_catches_incomplete_gen(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "pilot_item_ids", _triples)
    expected = K.expected_capture_keys(None, 1, 0, 1)  # 3 manifest-anchored keys
    metas = [  # a store built over an INCOMPLETE gen dir (2 of 3 pilot items)
        {
            "prompt_id": "evil|advbench_requests|advbench#0",
            "response_index": 0,
            "answer_sha256": _hex("a"),
        },
        {
            "prompt_id": "evil|advbench_requests|advbench#1",
            "response_index": 0,
            "answer_sha256": _hex("b"),
        },
    ]
    store = _store_shard(tmp_path, metas)
    # The OLD circular anchor (keys taken from the gen files) reads "complete":
    K.assert_store_complete(store, [(m["prompt_id"], 0) for m in metas])
    # The manifest anchor reads INCOMPLETE, loud:
    with pytest.raises(K.CaptureSpanError, match="MISSING"):
        K.assert_store_complete(store, expected)


# Fix 7 (minors): --responses 0 refuses; upload_raw refuses a zero-match scan.
def test_resolve_n_responses_defaults_and_refuses_nonpositive():
    assert G.resolve_n_responses(None, "pilot") == int(C.DECODER["n_responses_per_prompt_pilot"])
    assert G.resolve_n_responses(None, "dev") == int(C.DECODER["n_responses_per_prompt_production"])
    assert G.resolve_n_responses(7, "pilot") == 7
    for bad in (0, -3):  # the legacy ``or`` idiom silently fell through 0
        with pytest.raises(SystemExit, match="positive"):
            G.resolve_n_responses(bad, "pilot")


def test_upload_raw_refuses_zero_matches(monkeypatch, tmp_path):
    from explore_persona_space.orchestrate import hub

    monkeypatch.setattr(hub, "upload_raw_completions_to_data_repo", lambda name, root: [])
    with pytest.raises(RuntimeError, match="ZERO"):
        G.upload_raw(tmp_path, smoke=True)
    monkeypatch.setattr(hub, "upload_raw_completions_to_data_repo", lambda name, root: ["one"])
    G.upload_raw(tmp_path, smoke=True)  # non-empty: no raise
