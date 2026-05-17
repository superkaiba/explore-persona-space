"""Regression test for round-1 code review BLOCKER B1.

The original `_run_analytical_for_seed` / `_run_empirical_for_seed` fetched
`raw_rows` ONCE from a single eval persona and reused them for ALL eval
personas, causing every comedian-eval and baseline-eval arm to be
teacher-forced with librarian's CoT text. This invalidates the A2
conjunction (the HIGH-confidence carrier claim's headline contrast) and
the A3 cross-source check.

This test asserts the empirical invariant the reviewer used: in a real #186
`librarian_persona_cot_seedX/result.json`, the `persona_cot_text` (and
`generic_cot_text`) at the same q_id IS DIFFERENT across eval personas.
A test that PASSES on the fixed code AND fails on the buggy code.

The companion test below — `test_build_paired_for_persona_yields_distinct_cot`
— covers the entry-script-level fix by calling the new helper directly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_JSON = (
    _REPO_ROOT / "eval_results" / "issue186" / "librarian_persona_cot_seed42" / "result.json"
)


# ────────────────────────────────────────────────────────────────────────────
# JSON-level invariant: per-persona CoT text IS distinct.
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not _FIXTURE_JSON.exists(),
    reason=f"#186 fixture {_FIXTURE_JSON} not present; smoke-only environment.",
)
def test_per_persona_persona_cot_text_distinct():
    """Real #186 invariant: `per_persona[X].raw[i].persona_cot_text` is
    NOT identical across eval personas X. The round-1 code review
    confirmed librarian's CoT opens "In my experience organizing..."
    while comedian's opens "When I think about..." etc.

    If THIS test ever starts passing-via-equality, it means #186's storage
    format changed — refactor accordingly.
    """
    with open(_FIXTURE_JSON) as f:
        result = json.load(f)
    per_persona = result["per_persona"]
    # All three eval personas should be present in this source file.
    for key in ("librarian", "comedian", "assistant"):
        assert key in per_persona, f"per_persona missing key {key!r}"
    lib0 = per_persona["librarian"]["raw"][0]["persona_cot_text"]
    com0 = per_persona["comedian"]["raw"][0]["persona_cot_text"]
    asst0 = per_persona["assistant"]["raw"][0]["persona_cot_text"]
    assert lib0 != com0, (
        "librarian and comedian persona_cot_text at q_id=0 are IDENTICAL — "
        "either #186's storage shape changed (file regenerated) or this "
        "fixture is corrupt."
    )
    assert lib0 != asst0, "librarian and assistant persona_cot_text at q_id=0 are IDENTICAL"
    assert com0 != asst0, "comedian and assistant persona_cot_text at q_id=0 are IDENTICAL"


@pytest.mark.skipif(
    not _FIXTURE_JSON.exists(),
    reason=f"#186 fixture {_FIXTURE_JSON} not present.",
)
def test_per_persona_generic_cot_text_distinct():
    """Same invariant for `generic_cot_text` (also independently authored
    per eval persona in #186).
    """
    with open(_FIXTURE_JSON) as f:
        result = json.load(f)
    per_persona = result["per_persona"]
    lib0 = per_persona["librarian"]["raw"][0]["generic_cot_text"]
    com0 = per_persona["comedian"]["raw"][0]["generic_cot_text"]
    assert lib0 != com0, "librarian and comedian generic_cot_text at q_id=0 are IDENTICAL"


# ────────────────────────────────────────────────────────────────────────────
# Helper-level invariant: the new _build_paired_for_persona fetches
# DIFFERENT CoT text per persona.
# ────────────────────────────────────────────────────────────────────────────


def _load_measure_module():
    """Import ``scripts/measure_cot_entropy.py`` as a module without invoking
    Hydra (the @hydra.main decorator is module-level but doesn't fire until
    `main()` is called).
    """
    spec = importlib.util.spec_from_file_location(
        "measure_cot_entropy", _REPO_ROOT / "scripts" / "measure_cot_entropy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not _FIXTURE_JSON.exists(),
    reason=f"#186 fixture {_FIXTURE_JSON} not present.",
)
def test_build_paired_for_persona_yields_distinct_cot_per_persona():
    """End-to-end: the entry-script's `_build_paired_for_persona` helper
    returns the QUERIED persona's CoT text — not the source-persona's.

    This is the directly testable equivalent of B1: if the helper falls
    back to `per_persona[<wrong_key>]`, comedian's first row will be
    teacher-forced with librarian's CoT and this test fails.
    """
    measure = _load_measure_module()

    with open(_FIXTURE_JSON) as f:
        result = json.load(f)
    per_persona = result["per_persona"]
    # We don't need real ARC rows for the CoT-text assertion — we use the
    # raw rows' `correct_answer` mirrored into a stub ARC-row list keyed by
    # q_id position.
    n_rows = len(per_persona["librarian"]["raw"])
    arc_rows = [
        {
            "q_id": i,
            "correct_answer": per_persona["librarian"]["raw"][i]["correct_answer"],
            "question": "stub",
            "choice_labels": ["A", "B", "C", "D"],
            "choices": ["a", "b", "c", "d"],
        }
        for i in range(n_rows)
    ]

    paired_lib = measure._build_paired_for_persona(per_persona, "librarian", arc_rows)
    paired_com = measure._build_paired_for_persona(per_persona, "comedian", arc_rows)

    assert paired_lib[0][0]["persona_cot_text"] != paired_com[0][0]["persona_cot_text"], (
        "_build_paired_for_persona returned the same persona_cot_text for "
        "librarian and comedian — round-1 B1 regression. The helper must "
        "fetch THIS persona's raw rows, not a shared single-persona slice."
    )
    # And the q_ids match (canonical ordering).
    assert paired_lib[0][0]["q_id"] == paired_com[0][0]["q_id"] == 0


@pytest.mark.skipif(
    not _FIXTURE_JSON.exists(),
    reason=f"#186 fixture {_FIXTURE_JSON} not present.",
)
def test_build_paired_for_persona_q_id_filter_preserves_persona_cot():
    """When filtering to a non-contiguous q_id subsample (the empirical
    pass's stratified subsample), the helper still returns the queried
    persona's CoT — not the source-persona's.
    """
    measure = _load_measure_module()

    with open(_FIXTURE_JSON) as f:
        result = json.load(f)
    per_persona = result["per_persona"]
    n_rows = len(per_persona["librarian"]["raw"])
    arc_rows = [
        {
            "q_id": i,
            "correct_answer": per_persona["librarian"]["raw"][i]["correct_answer"],
            "question": "stub",
            "choice_labels": ["A", "B", "C", "D"],
            "choices": ["a", "b", "c", "d"],
        }
        for i in range(n_rows)
    ]

    # Pick a non-contiguous subset of q_ids.
    q_id_filter = {0, 5, 10, 100, 500}
    paired_com = measure._build_paired_for_persona(
        per_persona, "comedian", arc_rows, q_id_filter=q_id_filter
    )
    assert {r[0]["q_id"] for r in paired_com} == q_id_filter
    # Comedian's q=0 CoT must equal what's literally in per_persona.comedian
    # — not librarian's q=0 CoT.
    assert (
        paired_com[0][0]["persona_cot_text"]
        == per_persona["comedian"]["raw"][0]["persona_cot_text"]
    )
    assert (
        paired_com[0][0]["persona_cot_text"]
        != per_persona["librarian"]["raw"][0]["persona_cot_text"]
    )


# ────────────────────────────────────────────────────────────────────────────
# Synthetic-fixture variant of the above, runnable even when the real #186
# JSON is absent (e.g. on a fresh worktree without eval_results/).
# ────────────────────────────────────────────────────────────────────────────


def test_build_paired_for_persona_synthetic_distinct():
    """Same B1 invariant on a tiny synthetic fixture — guarantees the test
    file runs even when ``eval_results/issue186/`` isn't checked out.
    """
    measure = _load_measure_module()

    per_persona = {
        "librarian": {
            "raw": [
                {
                    "q_id": 0,
                    "correct_answer": "C",
                    "persona_cot_text": "librarian: I read the catalog.",
                    "generic_cot_text": "librarian: Let me think.",
                },
                {
                    "q_id": 1,
                    "correct_answer": "B",
                    "persona_cot_text": "librarian: Another rationale.",
                    "generic_cot_text": "librarian: Generic.",
                },
            ]
        },
        "comedian": {
            "raw": [
                {
                    "q_id": 0,
                    "correct_answer": "C",
                    "persona_cot_text": "comedian: Spotlight, please.",
                    "generic_cot_text": "comedian: Step-by-step joke.",
                },
                {
                    "q_id": 1,
                    "correct_answer": "B",
                    "persona_cot_text": "comedian: Different punchline.",
                    "generic_cot_text": "comedian: Generic comedy.",
                },
            ]
        },
    }
    arc_rows = [
        {
            "q_id": 0,
            "correct_answer": "C",
            "question": "q0",
            "choice_labels": ["A", "B", "C", "D"],
            "choices": ["a", "b", "c", "d"],
        },
        {
            "q_id": 1,
            "correct_answer": "B",
            "question": "q1",
            "choice_labels": ["A", "B", "C", "D"],
            "choices": ["a", "b", "c", "d"],
        },
    ]

    paired_lib = measure._build_paired_for_persona(per_persona, "librarian", arc_rows)
    paired_com = measure._build_paired_for_persona(per_persona, "comedian", arc_rows)

    assert paired_lib[0][0]["persona_cot_text"] == "librarian: I read the catalog."
    assert paired_com[0][0]["persona_cot_text"] == "comedian: Spotlight, please."
    assert paired_lib[0][0]["persona_cot_text"] != paired_com[0][0]["persona_cot_text"]
    # The ARC question rows are the SAME for both pairings (they index by
    # q_id in arc_rows, not by persona).
    assert paired_lib[0][1] is paired_com[0][1]


def test_build_paired_for_persona_raises_on_missing_key():
    """Defensive: querying an absent persona key must raise a clear error."""
    measure = _load_measure_module()
    per_persona = {"librarian": {"raw": [{"q_id": 0, "correct_answer": "A"}]}}
    arc_rows = [
        {
            "q_id": 0,
            "correct_answer": "A",
            "question": "q0",
            "choice_labels": ["A", "B", "C", "D"],
            "choices": ["a", "b", "c", "d"],
        },
    ]
    with pytest.raises(RuntimeError, match="missing key"):
        measure._build_paired_for_persona(per_persona, "comedian", arc_rows)


# ────────────────────────────────────────────────────────────────────────────
# End-to-end check on the analytical-pass prompt construction.
#
# Asserts the prompts BUILT for the comedian-eval and librarian-eval arms
# contain DIFFERENT CoT body text, which is the exact bug B1 caught.
# Mocks vLLM `llm.generate` so the test runs CPU-only.
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not _FIXTURE_JSON.exists(),
    reason=f"#186 fixture {_FIXTURE_JSON} not present.",
)
def test_run_analytical_for_seed_uses_per_persona_cot_in_prompts(monkeypatch):
    """Run `_run_analytical_for_seed` against a real #186 JSON with a
    stub LLM that captures the prompt strings. Assert the comedian-arm
    prompts contain comedian's CoT body, NOT librarian's.

    This is the FULL end-to-end regression on B1: a buggy implementation
    that pulls a single persona's raws and reuses them would produce
    identical CoT-body substrings across arms, and this test would fail.
    """
    measure = _load_measure_module()

    # Capture every prompt the stub LLM is asked to generate.
    captured_prompts: list[str] = []

    class _StubOutput:
        def __init__(self):
            # Mimic vLLM's `out.outputs[0].logprobs[0]` shape: a single
            # generated step whose top-K logprobs dict is empty (entropy
            # helper returns NaN-shaped result, which is fine for this test
            # because we only care about prompt construction).
            class _Inner:
                logprobs = [{}]  # noqa: RUF012 — test-only stub mimicking vLLM shape

            self.outputs = [_Inner()]

    class _StubLLM:
        def generate(self, prompts, sampling):
            captured_prompts.extend(prompts)
            return [_StubOutput() for _ in prompts]

    # Stub tokenizer.apply_chat_template that just concatenates system +
    # user content so we can grep the prompt strings.
    class _StubTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            parts = []
            for m in messages:
                parts.append(f"[{m['role']}]\n{m['content']}\n")
            return "".join(parts) + "[assistant]\n"

        def encode(self, s, add_special_tokens=False):
            return [0] * len(s.split())

    # Load the real #186 JSON.
    with open(_FIXTURE_JSON) as f:
        result = json.load(f)
    per_persona = result["per_persona"]
    n_rows = 3  # tiny — we only need a handful to confirm content
    arc_rows = [
        {
            "q_id": i,
            "correct_answer": per_persona["librarian"]["raw"][i]["correct_answer"],
            "question": f"Question {i}",
            "choice_labels": ["A", "B", "C", "D"],
            "choices": ["alpha", "beta", "gamma", "delta"],
        }
        for i in range(n_rows)
    ]
    # Truncate each persona's raws to the first n_rows so the helper
    # doesn't try to fetch beyond what's in arc_rows.
    per_persona_small = {
        key: {"raw": per_persona[key]["raw"][:n_rows]}
        for key in ("librarian", "comedian", "assistant")
    }

    # Minimal cfg shim — _run_analytical_for_seed reads cfg.analytical.top_k,
    # cfg.eval_personas, cfg.cot_styles, cfg.source.family.
    class _Cfg:
        class analytical:
            top_k = 20

        class source:
            family = "librarian_persona_cot"

        eval_personas = {"librarian": "librarian", "comedian": "comedian"}  # noqa: RUF012
        cot_styles = ["persona_cot"]  # noqa: RUF012 — test-only stub cfg

    persona_prompts = {
        "librarian": "You are a librarian.",
        "comedian": "You are a comedian.",
    }
    answer_token_ids = {"A": {32}, "B": {33}, "C": {34}, "D": {35}}

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_root = Path(tmp)
        measure._run_analytical_for_seed(
            _Cfg(),
            42,
            _StubLLM(),
            _StubTokenizer(),
            per_persona_small,
            arc_rows,
            persona_prompts,
            answer_token_ids,
            out_root,
            max_q=n_rows,
        )

    # We expect 2 personas x 1 cot_style x 3 rows = 6 prompts captured.
    assert len(captured_prompts) == 2 * 1 * n_rows

    # The first n_rows prompts are for the librarian arm; the next n_rows
    # are for the comedian arm. Each arm uses ITS OWN persona's CoT body.
    librarian_prompts = captured_prompts[:n_rows]
    comedian_prompts = captured_prompts[n_rows:]

    # Pick a CoT body substring unique to each persona by reading directly
    # from the fixture.
    lib_cot_q0 = per_persona_small["librarian"]["raw"][0]["persona_cot_text"]
    com_cot_q0 = per_persona_small["comedian"]["raw"][0]["persona_cot_text"]

    # Sanity: the fixture has distinct CoTs (B1 invariant).
    assert lib_cot_q0 != com_cot_q0

    # The first librarian-arm prompt must contain librarian's q=0 CoT body.
    # We do a substring check on the head (first 60 chars) to avoid
    # whitespace / strip ambiguity.
    lib_head = lib_cot_q0[:60]
    com_head = com_cot_q0[:60]
    assert lib_head in librarian_prompts[0], (
        "librarian-arm prompt at q=0 does not contain librarian's persona_cot_text"
    )
    # The first comedian-arm prompt must contain COMEDIAN's q=0 CoT body,
    # not librarian's.
    assert com_head in comedian_prompts[0], (
        "comedian-arm prompt at q=0 does NOT contain comedian's persona_cot_text "
        "— round-1 BLOCKER B1 regressed."
    )
    assert lib_head not in comedian_prompts[0], (
        "comedian-arm prompt at q=0 CONTAINS librarian's CoT — round-1 BLOCKER B1 regressed."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
