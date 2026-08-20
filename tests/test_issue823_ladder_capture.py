"""Gate tests for scripts/issue823_ladder_capture.py (#823 P-Cap driver).

Exercises the driver's data-dependent gates with degenerate inputs so each
fires its DESIGNED handling (never only the happy path): the R3 rollout-text
persistence precondition, the R1 span-lengths schema asserts, the R5
pre-registered cap-hit trigger, the assignment-integrity asserts, the R2
truncation pre-computation (production body executed against a
signature-conformant fake tokenizer at the external model-asset boundary),
and the R4 resume fingerprint mismatch.

Offline by design: every fixture is built under tmp_path; no HF fetch, no
repo eval_results/ fixture reads.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

import scripts.issue823_ladder_capture as ladder_capture
from scripts.issue823_ladder_capture import (
    CAP_HIT_REGEN_FRACTION,
    DATA_REPO,
    GENERATION_SUFFIX,
    HF_PREFIX,
    K_ARMS,
    N_CONTEXTS_FULL,
    N_PERSONAS,
    _require_canonical_upload,
    cap_hit_stats_and_gate,
    group_done,
    group_fingerprint,
    group_paths,
    load_pair_rows,
    load_span_lengths,
    own_length,
    precompute_rows,
    truncation_cell_stats,
    verify_gen_sentinel,
)
from scripts.issue823_ladder_gen import (
    GEN_MAX_TOKENS,
    REGEN_MAX_TOKENS,
    _sha256_file,
    registered_pair_total,
)

# ── Fixture builders (gen-schema-faithful records; benign synthetic text) ────


def _record(i: int, p: int, **over) -> dict:
    rec = {
        "context_id": i,
        "persona_idx": p,
        "persona_name": f"Persona {p}",
        "arms": [k for k in K_ARMS if i % k == p],
        "question": f"Question number {i}?",
        "answer_text": f"Answer text for context {i} persona {p}.",
        "seed": 42,
        "filled": True,
        "validity": "ok",
        "stop_reason": "end_turn",
        "cap_hit": False,
        "in_common_valid": True,
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 1.0,
        "max_tokens": GEN_MAX_TOKENS,
        "regen": False,
        "batch_id": "msgbatch_test",
        "batch_request_custom_id": f"p{p:02d}_i{i:05d}",
        "batch_org": "org0",
        "batch_submitted_at": "2026-08-19T00:00:00Z",
        "harvested_at": "2026-08-19T00:00:00Z",
    }
    rec.update(over)
    return rec


def _mk_gen_dir(tmp_path: pathlib.Path, n_contexts: int = 10) -> pathlib.Path:
    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    shas: dict[str, str] = {}
    for p in range(N_PERSONAS):
        recs = [_record(i, p) for i in range(n_contexts) if any(i % k == p for k in K_ARMS)]
        fn = f"persona{p:02d}_seed42.json"
        (gen_dir / fn).write_text(json.dumps({"metadata": {}, "records": recs}))
        shas[fn] = _sha256_file(gen_dir / fn)
    (gen_dir / "assignment.json").write_text(json.dumps({"n_contexts": n_contexts}))
    shas["assignment.json"] = _sha256_file(gen_dir / "assignment.json")
    (gen_dir / "_gen_complete.json").write_text(
        json.dumps({"complete": True, "files_sha256": shas})
    )
    return gen_dir


def _paths(gen_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    names = [f"persona{p:02d}_seed42.json" for p in range(N_PERSONAS)]
    names += ["assignment.json", "_gen_complete.json"]
    return {n: gen_dir / n for n in names}


def _span_dict(a_prime_overrides: dict[int, int] | None = None) -> dict[str, list[int]]:
    d = {arm: [100] * N_CONTEXTS_FULL for arm in ("a_prime", "b1", "b2", "c")}
    for i, v in (a_prime_overrides or {}).items():
        d["a_prime"][i] = v
    return d


# ── R3: gen sentinel precondition ────────────────────────────────────────────


def test_gen_sentinel_pass(tmp_path):
    paths = _paths(_mk_gen_dir(tmp_path))
    sentinel = verify_gen_sentinel(paths)
    assert sentinel["complete"] is True


def test_gen_sentinel_incomplete_rejected(tmp_path):
    gen_dir = _mk_gen_dir(tmp_path)
    sent = json.loads((gen_dir / "_gen_complete.json").read_text())
    sent["complete"] = False
    (gen_dir / "_gen_complete.json").write_text(json.dumps(sent))
    with pytest.raises(RuntimeError, match="complete!=True"):
        verify_gen_sentinel(_paths(gen_dir))


def test_gen_sentinel_sha_mismatch_rejected(tmp_path):
    gen_dir = _mk_gen_dir(tmp_path)
    # Tamper a persona file AFTER the sentinel recorded its sha.
    f = gen_dir / "persona03_seed42.json"
    f.write_text(f.read_text() + " ")
    with pytest.raises(RuntimeError, match=r"sha256 .* != sentinel"):
        verify_gen_sentinel(_paths(gen_dir))


# ── R1: span-lengths schema (arm-keyed positional lists) ─────────────────────


def test_span_lengths_schema_pass(tmp_path):
    p = tmp_path / "span.json"
    p.write_text(json.dumps(_span_dict()))
    d = load_span_lengths(p)
    assert set(d) == {"a_prime", "b1", "b2", "c"}
    assert all(len(v) == N_CONTEXTS_FULL for v in d.values())
    assert own_length(d, 0) == 100


def test_span_lengths_context_keyed_rejected(tmp_path):
    # The v8-guess shape the plan corrected: a dict keyed by context index.
    p = tmp_path / "span.json"
    p.write_text(json.dumps({str(i): 100 for i in range(N_CONTEXTS_FULL)}))
    with pytest.raises(RuntimeError, match="ARM-keyed with positional lists"):
        load_span_lengths(p)


def test_span_lengths_short_list_rejected(tmp_path):
    d = _span_dict()
    d["a_prime"] = d["a_prime"][:-1]
    p = tmp_path / "span.json"
    p.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="positional list of length"):
        load_span_lengths(p)


def test_own_length_range_checked(tmp_path):
    p = tmp_path / "span.json"
    p.write_text(json.dumps(_span_dict()))
    d = load_span_lengths(p)
    with pytest.raises(IndexError):
        own_length(d, N_CONTEXTS_FULL)


# ── Assignment integrity (pair set == registered nested assignment) ─────────


def test_load_pair_rows_registered_assignment(tmp_path):
    paths = _paths(_mk_gen_dir(tmp_path, n_contexts=10))
    by_persona = load_pair_rows(paths, 10)
    assert sum(len(r) for r in by_persona.values()) == registered_pair_total(10) == 25
    # persona 0 covers every context (the k=1 arm)
    assert [r["context_id"] for r in by_persona[0]] == list(range(10))


def test_load_pair_rows_question_mismatch_rejected(tmp_path):
    gen_dir = _mk_gen_dir(tmp_path, n_contexts=10)
    fn = gen_dir / "persona01_seed42.json"
    payload = json.loads(fn.read_text())
    payload["records"][0]["question"] = "A different question?"
    fn.write_text(json.dumps(payload))
    paths = _paths(gen_dir)
    # sentinel sha now stale by design; bypass R3 here — this test targets the
    # assignment-integrity gate directly.
    with pytest.raises(AssertionError, match="question text differs"):
        load_pair_rows(paths, 10)


def test_load_pair_rows_wrong_arms_rejected(tmp_path):
    gen_dir = _mk_gen_dir(tmp_path, n_contexts=10)
    fn = gen_dir / "persona01_seed42.json"
    payload = json.loads(fn.read_text())
    payload["records"][0]["arms"] = [1]  # persona 1 can never own a k=1 row
    fn.write_text(json.dumps(payload))
    with pytest.raises(AssertionError, match="registered persona"):
        load_pair_rows(_paths(gen_dir), 10)


# ── R5: cap-hit fractions + pre-registered trigger ───────────────────────────


def _cap_rows(n: int, n_cap: int, max_tokens: int) -> list[dict]:
    rows = [_record(i, 0) for i in range(n)]
    for r in rows[:n_cap]:
        r["cap_hit"] = True
        r["stop_reason"] = "max_tokens"
        r["max_tokens"] = max_tokens
    return rows


def test_cap_hit_gate_trips_on_unregenerated_overcap():
    by_persona = {0: _cap_rows(100, 3, GEN_MAX_TOKENS)}  # 3% > 2% at original cap
    with pytest.raises(RuntimeError, match="pre-registered cap-hit trigger violated"):
        cap_hit_stats_and_gate(by_persona, smoke=False)


def test_cap_hit_gate_passes_after_regen():
    by_persona = {0: _cap_rows(100, 3, REGEN_MAX_TOKENS)}  # same rows, regenerated
    stats = cap_hit_stats_and_gate(by_persona, smoke=False)
    assert stats["gate"] == "PASS"
    assert stats["per_persona"]["0"]["cap_hit_fraction_realized"] == pytest.approx(0.03)
    assert stats["per_persona"]["0"]["unregenerated_overcap_fraction"] == 0.0
    assert stats["per_persona"]["0"]["n_residual_cap_at_regen_tokens"] == 3


def test_cap_hit_gate_smoke_is_informational():
    by_persona = {0: _cap_rows(100, 3, GEN_MAX_TOKENS)}
    stats = cap_hit_stats_and_gate(by_persona, smoke=True)
    assert stats["gate"] == "WARN-SMOKE-INFORMATIONAL"


def _cell_victim_rows() -> dict[int, list[dict]]:
    """Persona-5 fixture where ONE cell violates while the pooled rate does not.

    60 rows of persona 5 (i % 8 == 5 over range(480)); ONE victim row with
    i % 16 == 5 capped at the ORIGINAL cap. Cell (16,5) holds 30 of the rows
    -> 1/30 = 3.3% > 2% violates, while the pooled persona rate (and the k=8
    cell) sit at 1/60 = 1.67% <= 2% -- the superseded v10 per-persona gate
    would have PASSED this exact input (round-4 BLOCKER 1, capture sibling).
    """
    rows = [_record(i, 5) for i in range(480) if i % 8 == 5]
    assert len(rows) == 60
    assert sum(1 for r in rows if 16 in r["arms"]) == 30
    victim = next(r for r in rows if r["context_id"] % 16 == 5)
    victim.update(cap_hit=True, stop_reason="max_tokens", max_tokens=GEN_MAX_TOKENS)
    return {5: rows}


def test_cap_hit_gate_trips_per_cell_where_pooled_rate_passes():
    # Fails pre-fix: the per-persona form saw 1/60 <= 2% and passed; the
    # per-cell form sees cell (16,5) at 1/30 > 2% and raises, NAMING the cell.
    by_persona = _cell_victim_rows()
    assert 1 / 60 <= CAP_HIT_REGEN_FRACTION < 1 / 30  # in-test denominator pin
    with pytest.raises(RuntimeError, match=re.escape("(16, 5,")):
        cap_hit_stats_and_gate(by_persona, smoke=False)


def test_cap_hit_gate_per_cell_regen_resolves_and_reports_fractions():
    # The SAME victim regenerated at REGEN_MAX_TOKENS: no un-regenerated
    # over-cap anywhere -> PASS, with the residual capped row still visible
    # in the per-cell realized fractions (labelling, never a silent drop).
    by_persona = _cell_victim_rows()
    for r in by_persona[5]:
        if r["cap_hit"]:
            r["max_tokens"] = REGEN_MAX_TOKENS
    stats = cap_hit_stats_and_gate(by_persona, smoke=False)
    assert stats["gate"] == "PASS"
    assert stats["cap_hit_fraction_by_arm_persona"]["16"]["5"] == pytest.approx(1 / 30)
    assert stats["unregenerated_overcap_fraction_by_arm_persona"]["16"]["5"] == 0.0
    assert stats["per_persona"]["5"]["n_residual_cap_at_regen_tokens"] == 1


def test_cap_hit_gate_per_cell_smoke_is_informational():
    # Smoke keeps the per-cell computation but demotes the verdict (the
    # plan-enumerated production-n-calibrated-gate blind spot).
    stats = cap_hit_stats_and_gate(_cell_victim_rows(), smoke=True)
    assert stats["gate"] == "WARN-SMOKE-INFORMATIONAL"
    assert stats["unregenerated_overcap_fraction_by_arm_persona"]["16"]["5"] == pytest.approx(
        1 / 30
    )


# ── R2: truncation pre-computation against a signature-conformant fake ──────


class FakeQwenTokenizer:
    """Signature-conformant fake at the external model-asset boundary.

    Mirrors the three methods `template_span_length`/`precompute_rows` call:
    a Qwen-shaped chat template, regex word/special tokenization whose
    trailing generation suffix is exactly 3 tokens, and a join-decode — so
    the GENERATION_SUFFIX assert and span arithmetic run the production body.
    """

    _TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|\n|[^\n<]+")

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._toks: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return s

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = []
        for tok in self._TOKEN_RE.findall(text):
            if tok not in self._vocab:
                self._vocab[tok] = len(self._toks)
                self._toks.append(tok)
            ids.append(self._vocab[tok])
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(self._toks[i] for i in ids)


def test_fake_tokenizer_suffix_matches_generation_suffix():
    tok = FakeQwenTokenizer()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    ids = tok(prompt)["input_ids"]
    assert tok.decode(ids[-3:]) == GENERATION_SUFFIX


def test_precompute_rows_truncation_and_skips(tmp_path):
    tok = FakeQwenTokenizer()
    # Answer "a\nb\nc" tokenizes to 5 tokens + <|im_end|> + \n => pair_len 7.
    long_answer = "a\nb\nc"  # pair_len 7
    short_answer = "hello"  # 1 token + 2 => pair_len 3
    rows = {
        0: [
            _record(0, 0, answer_text=long_answer),  # own 5 < 7 => truncated
            _record(1, 0, answer_text=short_answer),  # own 5 > 3 => untruncated
            _record(2, 0, answer_text=long_answer),  # own 0 => no truncation
            _record(3, 0, filled=False, validity="refusal", answer_text=None),  # skip
        ]
    }
    span_d = _span_dict({0: 5, 1: 5, 2: 0, 3: 5})
    pre = precompute_rows(tok, rows, span_d)
    r0, r1, r2, r3 = pre[0]
    assert (r0["own_len"], r0["pair_len"], r0["trunc_len"]) == (5, 7, 5)
    assert r0["truncated"] and r0["dropped_tokens"] == 2 and r0["expected_span"] == 5
    assert (r1["pair_len"], r1["trunc_len"], r1["truncated"]) == (3, 3, False)
    assert (r2["own_len"], r2["trunc_len"], r2["truncated"]) == (0, 7, False)
    assert r3["skip_reason"] == "not_filled" and r3["expected_span"] == 0

    cells = truncation_cell_stats(pre)
    # Arm k=1 persona 0 covers the three live rows: 1/3 truncated; mass 7+3+7=17, dropped 2.
    cell = cells["k1_p00"]
    assert cell["n_rows"] == 3
    assert cell["truncation_fraction"] == pytest.approx(1 / 3)
    assert cell["truncated_token_mass_share"] == pytest.approx(2 / 17)


# ── R4: resume fingerprint discipline ────────────────────────────────────────


def _sentinel_for(gen_dir: pathlib.Path) -> dict:
    return json.loads((gen_dir / "_gen_complete.json").read_text())


def test_group_done_matrix(tmp_path):
    gen_dir = _mk_gen_dir(tmp_path)
    sentinel = _sentinel_for(gen_dir)
    tensors_dir = tmp_path / "analysis_tensors"
    tensors_dir.mkdir()
    fp = group_fingerprint(sentinel, 0, 10, "spansha", 8)

    # No checkpoint => not done.
    assert group_done(tensors_dir, 0, fp) is False

    # Matching sidecar but MISSING tensor => partial checkpoint, fail loud.
    tensor_path, sidecar_path = group_paths(tensors_dir, 0)
    sidecar_path.write_text(json.dumps({"fingerprint": fp, "n_rows": 1}))
    with pytest.raises(RuntimeError, match="partial checkpoint"):
        group_done(tensors_dir, 0, fp)

    # Matching sidecar + tensor => done.
    tensor_path.write_bytes(b"placeholder")
    assert group_done(tensors_dir, 0, fp) is True

    # DIFFERENT fingerprint (regime drift) => never silent reuse.
    other = group_fingerprint(sentinel, 0, 10, "OTHERSHA", 8)
    with pytest.raises(RuntimeError, match="DIFFERENT fingerprint"):
        group_done(tensors_dir, 0, other)


# ── FIX A (follow-up round): P-Store canonical-repo completion gate ──────────
# Mirrors TestCanonicalUploadGate in tests/test_issue823_ladder_gen_fixes.py:
# the P-Store upload of the primary-deliverable tensors must refuse an
# overflow-repo reroute (the helper's default-on file-count fallback returns a
# truthy OVERFLOW url) instead of logging complete + writing the done-sentinel.


class TestPStoreCanonicalUploadGate:
    CANON = f"{DATA_REPO}/{HF_PREFIX}/analysis_tensors"

    def test_canonical_url_passes(self):
        _require_canonical_upload(self.CANON, self.CANON)  # no raise

    def test_overflow_reroute_raises_with_pointer(self):
        overflow = f"superkaiba1/explore-persona-space-overflow/{HF_PREFIX}/analysis_tensors"
        with pytest.raises(RuntimeError, match=re.escape(overflow)):
            _require_canonical_upload(overflow, self.CANON)

    def test_wrong_prefix_on_canonical_repo_raises(self):
        with pytest.raises(RuntimeError, match="canonical"):
            _require_canonical_upload(f"{DATA_REPO}/somewhere_else", self.CANON)

    def test_gate_wired_between_pstore_upload_and_done_sentinel(self):
        src = pathlib.Path(ladder_capture.__file__).read_text()
        i_up = src.index('log_phase("pstore_upload")')
        i_gate = src.index("_require_canonical_upload(url", i_up)
        i_sentinel = src.index("write_sentinel(", i_up)
        assert i_gate < i_sentinel, "canonical gate must run BEFORE the done-sentinel write"
