"""#1336 round 5 — on-policy naturalistic generation arm (part A).

Pins the four build items on CPU (no tokenizer, no network, no vLLM):
  1. the naturalistic generation prompt equals render_natural's segment text
     byte-for-byte (shared constants — single source of truth);
  2. the chat path is byte-unchanged (frozen prompt literal, bare-corpus cell
     key, unchanged HF prefix + stem composition);
  3. the format-keyed gen-cell token shared by the gen writer and the
     turnstore reader (``cm.gen_cell_key``) can never collide across arms;
  4. the render-integrity gate is format-conditional: HARD-FAILS on injected
     mismatch in the matched-text (chat) regime, reports-without-failing in
     the on-policy naturalistic regime, with the regime explicit in the
     emitted audit record.

The real-tokenizer end-to-end runs live in ``issue1336_smoke_fixtures.py``
(``gen`` — chat, unchanged; ``gen-natural`` — the on-policy arm).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.experiments.issue_1336.common import Rendered  # noqa: E402

_Q = "How do I keep basil alive on a windowsill?"
_A = "Give it the brightest sill you have and water only when the soil is dry."


# ---------------------------------------------------------------------------
# 1. prompt builders — byte-for-byte against the render's segment text
# ---------------------------------------------------------------------------
def test_natural_prompt_matches_render_natural_segments_byte_for_byte():
    from issue1336_render import natural_segments

    segs = natural_segments({"u1": _Q, "a1": _A})
    # Frozen #825 plain-transcript convention (drift in either source fails).
    assert segs == ["User: ", _Q, "\n\n", "Assistant: ", _A]
    # The generation prefix IS the render's first four segments joined.
    assert "".join(segs[:4]) == cm.natural_prompt(_Q)
    # generation prefix + generated answer == the extraction render's text.
    assert "".join(segs) == cm.natural_prompt(_Q) + _A


def test_natural_prompt_frozen_literal():
    assert cm.natural_prompt(_Q) == f"User: {_Q}\n\nAssistant: "


def test_chat_prompt_unchanged_frozen_literal():
    assert cm.tulu_prompt(_Q) == f"<|user|>\n{_Q}\n<|assistant|>\n"
    assert (
        cm.tulu_prompt(_Q) == cm.TULU_USER_HEADER + _Q + cm.TULU_TURN_SEP + cm.TULU_ASSISTANT_HEADER
    )


# ---------------------------------------------------------------------------
# 2/3. format-keyed cell token — chat bare (byte-compat), naturalistic keyed
# ---------------------------------------------------------------------------
def test_gen_cell_key_chat_is_bare_corpus():
    assert cm.gen_cell_key("lmsys5k", "chat") == "lmsys5k"
    assert cm.gen_cell_key("lmsys23k", "chat") == "lmsys23k"


def test_gen_cell_key_naturalistic_is_suffixed_and_distinct():
    key = cm.gen_cell_key("lmsys5k", "naturalistic")
    assert key == "lmsys5k__gen_naturalistic"
    assert key != cm.gen_cell_key("lmsys5k", "chat")
    with pytest.raises(AssertionError):
        cm.gen_cell_key("lmsys5k", "tulu")  # unknown format fails loud


def test_hf_gen_prefix_chat_unchanged_naturalistic_keyed():
    import issue1336_gen_answers as g

    # Frozen prior-round literal — chat prefixes must never move (#664 resume).
    assert (
        g._hf_gen_prefix("rlvr", cm.gen_cell_key("lmsys5k", "chat"))
        == "issue1336_rlvr_ladder/raw_completions/generation/rlvr/lmsys5k"
    )
    assert (
        g._hf_gen_prefix("rlvr", cm.gen_cell_key("lmsys5k", "naturalistic"))
        == "issue1336_rlvr_ladder/raw_completions/generation/rlvr/lmsys5k__gen_naturalistic"
    )


def test_turnstore_stem_chat_unchanged_naturalistic_keyed():
    # Matched-text (chat gen): stem byte-identical to every prior round.
    assert cm.cell_id("rlvr", "naturalistic", cm.gen_cell_key("lmsys5k", "chat")) == (
        "rlvr_naturalistic_lmsys5k"
    )
    # On-policy arm: stem carries the gen suffix — no collision possible.
    assert cm.cell_id("rlvr", "naturalistic", cm.gen_cell_key("lmsys5k", "naturalistic")) == (
        "rlvr_naturalistic_lmsys5k__gen_naturalistic"
    )


# ---------------------------------------------------------------------------
# stop handling — chat markers untouched; naturalistic newline-anchored
# ---------------------------------------------------------------------------
def test_truncate_role_headers_chat_behavior_unchanged():
    import issue1336_gen_answers as g

    # Bare chat markers cut AT the marker, keeping the trailing newline — the
    # exact prior-round behavior (frozen; the naturalistic markers below are
    # newline-anchored and cut BEFORE it).
    text = "Simmer for thirty minutes.\n<|user|>\nCan you make it vegan?"
    assert g._truncate_role_headers(text, cm.ROLE_HEADER_TRUNCATE) == "Simmer for thirty minutes.\n"
    assert cm.STOP_STRINGS == ("\n<|user|>",)  # frozen — chat recipe untouched
    assert cm.ROLE_HEADER_TRUNCATE == ("<|user|>", "<|assistant|>")


def test_truncate_role_headers_naturalistic_newline_anchored():
    import issue1336_gen_answers as g

    text = "Simmer for thirty minutes.\nUser: Can you make it vegan?"
    assert g._truncate_role_headers(text, cm.NATURAL_ROLE_HEADER_TRUNCATE) == (
        "Simmer for thirty minutes."
    )
    # A legitimate mid-line mention never truncates (newline-anchored markers).
    mention = "The User: field of the form is required."
    assert g._truncate_role_headers(mention, cm.NATURAL_ROLE_HEADER_TRUNCATE) == mention


# ---------------------------------------------------------------------------
# 4. format-conditional render-integrity gate
# ---------------------------------------------------------------------------
def _rendered(fmt: str, conv_id: str, u1_ids: list[int], a1_ids: list[int]) -> Rendered:
    """Synthetic Rendered (the test_issue1336_render_integrity builder)."""
    ids = [900, 901, *u1_ids, 902, 903, *a1_ids]
    u1s, u1e = 2, 2 + len(u1_ids)
    a1s, a1e = u1e + 2, u1e + 2 + len(a1_ids)
    return Rendered(
        input_ids=ids,
        slot_idx={"prefix": 1, "a1": a1s - 1},
        spans={"u1": (u1s, u1e), "a1": (a1s, a1e)},
        format=fmt,
        conv_id=conv_id,
        meta={},
    )


_U1 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
_A1 = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]


def _pairs(n_bad: int, n_clean: int = 0) -> list[tuple[Rendered, Rendered]]:
    """(chat, naturalistic) twins; bad pairs diverge past every bounded trim."""
    bad_a1 = list(_A1)
    bad_a1[5] = 999  # deeper than _HEAD_TOL=3 — no trim combination absorbs it
    out = [
        (_rendered("chat", f"c{i}", _U1, _A1), _rendered("naturalistic", f"c{i}", _U1, _A1))
        for i in range(n_clean)
    ]
    out += [
        (_rendered("chat", f"b{i}", _U1, _A1), _rendered("naturalistic", f"b{i}", _U1, bad_a1))
        for i in range(n_bad)
    ]
    return out


def test_matched_text_regime_hard_fails_on_injected_mismatch():
    import issue1336_gen_answers as g

    with pytest.raises(AssertionError, match="render-integrity gate FAIL"):
        g._run_render_integrity(_pairs(n_bad=3), "chat", "rlvr", "lmsys5k")


def test_matched_text_regime_pass_records_regime():
    import issue1336_gen_answers as g

    res = g._run_render_integrity(_pairs(n_bad=0, n_clean=4), "chat", "rlvr", "lmsys5k")
    assert res["status"] == "PASS"
    assert res["regime"] == "matched-text" and res["enforced"] is True


def test_on_policy_regime_reports_without_failing():
    import issue1336_gen_answers as g

    # The SAME injected mismatch that hard-fails the matched-text regime is
    # computed + reported as a diagnostic here — never raised on.
    res = g._run_render_integrity(
        _pairs(n_bad=3), "naturalistic", "rlvr", "lmsys5k__gen_naturalistic"
    )
    assert res["status"] == "FAIL"  # the statistic is still honest
    assert res["regime"] == "on-policy-naturalistic" and res["enforced"] is False
    assert res["rest_of_span_mismatch_rate"] == pytest.approx(0.5)
    assert res["mismatches"] == 3 and res["total_spans"] == 6


# ---------------------------------------------------------------------------
# round 5b — naturalistic GENERATION on every v2 corpus (context arm only):
# the gen-only format registry widens _formats_for; the fit-side grid
# (V2_CORPORA formats / v2_surfaces / cells_v2_for) stays byte-untouched.
# ---------------------------------------------------------------------------
def test_v2_gen_formats_covers_exactly_the_v2_corpora_chat_first():
    assert set(cm.V2_GEN_FORMATS) == set(cm.V2_CORPORA)
    for corpus, fmts in cm.V2_GEN_FORMATS.items():
        assert fmts == ("chat", "naturalistic"), (corpus, fmts)


def test_v2_gen_formats_does_not_widen_the_fit_side_registry():
    # The fit-side grid is the load-bearing invariant: widening V2_CORPORA
    # formats would shift v2_surface_index (the §3 bootstrap seeds, 5000+idx)
    # for 5 existing surfaces and add 30 storeless cells to CELLS_V2. Pinned
    # in full by test_issue1336_stage_corpora::test_v2_registry_shape +
    # test_issue1336_fit_v2; re-asserted here at the seam the gen widening
    # touches.
    assert cm.V2_CORPORA["lmsys23k"]["formats"] == ("chat", "naturalistic")
    for corpus in set(cm.V2_CORPORA) - {"lmsys23k"}:
        assert cm.V2_CORPORA[corpus]["formats"] == ("chat",), corpus
    assert len(cm.v2_surfaces()) == 8
    assert len(cm.CELLS_V2) == 45


def test_formats_for_chat_path_returns_base_registry_unchanged():
    import issue1336_gen_answers as g

    # Byte-identical chat behavior: default arg and explicit "chat" both
    # return the corpus's own base registry for EVERY corpus (v1 + v2).
    for corpus, base in cm.FORMATS_BY_CORPUS.items():
        assert g._formats_for(corpus) == base
        assert g._formats_for(corpus, "chat") == base
    for corpus in cm.V2_CORPORA:
        if corpus in cm.FORMATS_BY_CORPUS:
            continue  # v1 registry wins (default-preserving lookup order)
        base = tuple(cm.V2_CORPORA[corpus]["formats"])
        assert g._formats_for(corpus) == base
        assert g._formats_for(corpus, "chat") == base


def test_formats_for_naturalistic_accepted_on_all_seven_v2_corpora():
    import issue1336_gen_answers as g

    for corpus in cm.V2_CORPORA:
        fmts = g._formats_for(corpus, "naturalistic")
        assert "naturalistic" in fmts, corpus
        assert fmts[0] == "chat", corpus  # chat stays first (validate order)
        assert fmts.count("naturalistic") == 1, corpus  # no duplicate append


def test_formats_for_unlicensed_formats_return_base_so_the_assert_fires():
    import issue1336_gen_answers as g

    # Unknown format on a v2 corpus: NOT licensed by V2_GEN_FORMATS — base
    # registry returned, so run_generation's acceptance assert fails loud.
    assert g._formats_for("math7500", "tulu") == ("chat",)
    # Naturalistic on a chat-only v1 corpus OUTSIDE the v2 set: unchanged
    # fail-loud shape (gsm8k_train5k is v1-only; its v2 sibling is the
    # concat corpus gsm8k_train_full).
    assert g._formats_for("gsm8k_train5k", "naturalistic") == ("chat",)


# ---------------------------------------------------------------------------
# 6. dispatcher wiring (round 5c) — the SEPARATE gen_v2_nat phase
# ---------------------------------------------------------------------------
_DISPATCH = REPO_ROOT / "scripts" / "issue1336_dispatch.sh"


def _registry_preamble() -> str:
    """The registry_lines_v2 heredoc python preamble, verbatim from the .sh."""
    import re

    m = re.search(
        r"registry_lines_v2\(\) \{.*?<<'PY'\n(.*?)\nPY\n\}",
        _DISPATCH.read_text(),
        re.S,
    )
    assert m, "registry_lines_v2 heredoc not found in issue1336_dispatch.sh"
    return m.group(1)


def _phase_body(name: str) -> str:
    import re

    m = re.search(rf"^phase_{name}\(\) \{{\n(.*?)\n\}}$", _DISPATCH.read_text(), re.S | re.M)
    assert m, f"phase_{name} not found in issue1336_dispatch.sh"
    return m.group(1)


def _job_builder_expr(phase_body: str) -> str:
    import re

    m = re.search(r"registry_lines_v2 '\n(.*?)\n' > \"\$jobs\"", phase_body, re.S)
    assert m, "job-builder expression not found in phase body"
    return m.group(1)


def _run_registry(expr: str, *, smoke: bool, monkeypatch, capsys) -> str:
    """Execute the dispatcher's REAL registry preamble + expression in-process.

    Mirrors the shell invocation exactly: SMOKE_ENV in the environment, the
    expression as sys.argv[1], the preamble exec'd in fresh globals.
    """
    monkeypatch.setenv("SMOKE_ENV", "1" if smoke else "0")
    monkeypatch.setattr(sys, "argv", ["registry_lines_v2", expr])
    exec(compile(_registry_preamble(), "<registry_lines_v2>", "exec"), {})
    return capsys.readouterr().out


def test_dispatch_nat_gen_corpora_all_seven_production_two_class_smoke(monkeypatch, capsys):
    # Production: ALL SEVEN v2 corpora — the V2_FULLY_REUSED_GEN exclusion is
    # CHAT-only (no naturalistic wave-1 exists), so gsm8k_test1319 generates.
    prod = _run_registry(
        "print(','.join(nat_gen_corpora))", smoke=False, monkeypatch=monkeypatch, capsys=capsys
    )
    assert prod.strip().split(",") == list(cm.V2_CORPORA)
    assert "gsm8k_test1319" in prod
    # Smoke: the same two-class pair as the chat arm (concat + fresh-build).
    smoke = _run_registry(
        "print(','.join(nat_gen_corpora))", smoke=True, monkeypatch=monkeypatch, capsys=capsys
    )
    assert smoke.strip().split(",") == ["lmsys23k", "sft11k"]
    # Chat gen set byte-unchanged: still excludes the fully-reused corpus.
    chat = _run_registry(
        "print(','.join(gen_corpora))", smoke=False, monkeypatch=monkeypatch, capsys=capsys
    )
    assert chat.strip().split(",") == [c for c in cm.V2_CORPORA if c not in cm.V2_FULLY_REUSED_GEN]


def test_dispatch_natgen_jobs_carry_gen_format_chat_jobs_do_not(monkeypatch, capsys):
    nat_expr = _job_builder_expr(_phase_body("gen_v2_nat"))
    out = _run_registry(nat_expr, smoke=False, monkeypatch=monkeypatch, capsys=capsys)
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert len(lines) == len(list(cm.MODELS)) * len(cm.V2_CORPORA)  # 5 x 7 = 35
    names = set()
    for ln in lines:
        name, cmd = ln.split("\t", 1)
        names.add(name)
        assert "--gen-format naturalistic" in cmd, ln
        assert "--upload" in cmd, ln
    assert names == {f"{m}__{c}" for m in cm.MODELS for c in cm.V2_CORPORA}
    assert lines[0].startswith("rlvr__")  # gen_v2's rlvr-first ordering kept

    # Chat arm byte-unchanged: its job lines never carry --gen-format.
    chat_expr = _job_builder_expr(_phase_body("gen_v2"))
    chat_out = _run_registry(chat_expr, smoke=False, monkeypatch=monkeypatch, capsys=capsys)
    assert chat_out.strip()
    assert "--gen-format" not in chat_out


def test_dispatch_natgen_done_keys_never_collide_with_chat(tmp_path):
    """Phase-level AND job-level done-file keys are disjoint across the arms,
    so pre-existing chat done files can never skip a naturalistic job (and
    vice versa)."""
    import subprocess

    keys = {}
    for phase in ("gen_v2", "gen_v2_nat"):
        got = subprocess.run(
            ["bash", str(_DISPATCH), "__phase_key", phase],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=120,
        )
        assert got.returncode == 0, got.stderr
        keys[phase] = got.stdout.strip().splitlines()[-1]
    assert keys["gen_v2"] == "gen_v2"
    assert keys["gen_v2_nat"] == "gen_v2_nat"  # distinct phase_gen_v2_nat.done

    body = _phase_body("gen_v2_nat")
    # run_queue's phase arg keys per-job done files gen_v2_nat__{m}__{c}.done.
    assert 'run_queue gen_v2_nat "$jobs"' in body
    assert "jobs_gen_v2_nat.tsv" in body
    assert "gen_v2_nat__prep.done" in body
    # Audit mirror reads the format-keyed cell dir + writes non-colliding
    # audit_{m}_{corpus}__gen_naturalistic.json names.
    assert 'cm.gen_cell_key(c, "naturalistic")' in body


def test_dispatch_all_v2_chain_does_not_run_the_naturalistic_arm():
    import re

    text = _DISPATCH.read_text()
    m = re.search(r"^all_v2\)\n(.*?)^\s*;;\s*$", text, re.S | re.M)
    assert m, "all_v2 case arm not found"
    assert "gen_v2_nat" not in m.group(1)  # separate invocation by design
    # ... but the single-phase case DOES accept it.
    assert re.search(r"^c_stage \| .*\| gen_v2_nat \|", text, re.M)


# ---------------------------------------------------------------------------
# 7. extraction licensing (round 5c) — the two-axis on-policy gate
# ---------------------------------------------------------------------------
# `--format` is the RENDER of the captured text; `--gen-format` is which
# generation pool is read. `extraction_licensed` accepts (a) BASE pairs (both
# axes in the fit-side registry — byte-identical to every prior round) and
# (b) the --v2 ON-POLICY extension (fmt == gen_format, gen-licensed via
# cm.V2_GEN_FORMATS). Importing the extract script is established test
# practice (tests/test_issue1336_dispatch_v2.py imports it at module top).


def _extract_mod():
    import issue1336_extract_turnstore as et

    return et


def test_extract_base_licensing_unchanged_from_fit_registry():
    et = _extract_mod()
    for corpus, spec in cm.V2_CORPORA.items():
        fmts = spec["formats"]
        for fmt in ("chat", "naturalistic"):
            for gen in ("chat", "naturalistic"):
                if fmt in fmts and gen in fmts:
                    # BASE pairs stay licensed — on lmsys23k this is all four
                    # combos, incl. the committed matched-text cells.
                    assert et.extraction_licensed(corpus, fmt, gen, v2=True), (corpus, fmt, gen)


def test_extract_onpolicy_naturalistic_licensed_on_all_seven_v2_corpora():
    et = _extract_mod()
    for corpus in cm.V2_CORPORA:
        assert et.extraction_licensed(corpus, "naturalistic", "naturalistic", v2=True), corpus


def test_extract_cross_format_pairs_refused_on_chat_only_corpora():
    et = _extract_mod()
    chat_only = [c for c, s in cm.V2_CORPORA.items() if s["formats"] == ("chat",)]
    assert len(chat_only) == 6  # every v2 corpus except lmsys23k
    for corpus in chat_only:
        # Matched-text pairs create a fit SURFACE — fit-side-registry gated.
        assert not et.extraction_licensed(corpus, "naturalistic", "chat", v2=True), corpus
        assert not et.extraction_licensed(corpus, "chat", "naturalistic", v2=True), corpus
        # The chat path is untouched.
        assert et.extraction_licensed(corpus, "chat", "chat", v2=True), corpus


def test_extract_onpolicy_extension_is_v2_only_v1_registry_governs_otherwise():
    et = _extract_mod()
    # gsm8k_test1319 is dual-registered: v2 mode gets the extension, v1 mode
    # stays governed by FORMATS_BY_CORPUS = ("chat",).
    assert et.extraction_licensed("gsm8k_test1319", "naturalistic", "naturalistic", v2=True)
    assert not et.extraction_licensed("gsm8k_test1319", "naturalistic", "naturalistic", v2=False)
    # v1 lmsys5k keeps its BASE licensing (registry has naturalistic).
    assert et.extraction_licensed("lmsys5k", "naturalistic", "chat", v2=False)
    assert et.extraction_licensed("lmsys5k", "naturalistic", "naturalistic", v2=False)
    # v1 gsm8k_train5k stays chat-only (closing caveat 1 would widen
    # FORMATS_BY_CORPUS — an explicit decision, not this extension).
    assert not et.extraction_licensed("gsm8k_train5k", "naturalistic", "naturalistic", v2=False)


def test_extract_main_gate_calls_extraction_licensed():
    import inspect

    et = _extract_mod()
    src = inspect.getsource(et.main)
    # The licensing helper must be the gate main() actually runs (no hollow
    # gate: code-style.md "Verification gates test the live dispatched path").
    assert "extraction_licensed(" in src
    assert "not registered for" not in src  # old independent-axis asserts gone


def test_extract_onpolicy_stems_collide_with_no_existing_registry_stem():
    et = _extract_mod()
    assert et  # stems derive from cm; import pins the consumer exists
    existing = {
        cm.cell_id(m, f, c)
        for m in cm.MODELS
        for c, spec in cm.V2_CORPORA.items()
        for f in spec["formats"]
    }
    onpolicy = {
        cm.cell_id(m, "naturalistic", cm.gen_cell_key(c, "naturalistic"))
        for m in cm.MODELS
        for c in cm.V2_CORPORA
    }
    assert len(onpolicy) == len(cm.MODELS) * len(cm.V2_CORPORA)
    assert existing.isdisjoint(onpolicy)
