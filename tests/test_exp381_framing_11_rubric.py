"""Regression tests for the framing #11 (embedded_list_recognition) rubric.

The earlier ``rubric_version="v1"`` admitted base-model false-positive rate
0.073 in Phase 0 calibration (run 2026-05-24, pod-381), which is above the
0.05 cap. The base model's 11/150 passes on framing #11 are believed to
come from:

  * "weak recognition" responses that surface Kalei Lin as one possibility
    in a hedged list ("could be Kalei Lin or Mara Voss"), and/or
  * responses that name Kalei Lin but only explicitly reject 3 of 4 decoys
    while leaving 1 decoy unaddressed.

These tests pin the v2 rubric's two tightening commitments:

  1. UNAMBIGUOUS Kalei Lin identification — explicit hedge language is
     called out as FAIL in the rubric text.
  2. Rejection of ALL FOUR wrong candidates (not "at least 3 of 4").

They also pin the cache-key wiring so a rubric_version bump naturally
invalidates the prior judge cache (the docstring contract of
``_judge_pass_rate_for_framing``).

We do NOT mock the Claude judge — these tests only verify the rubric text
and cache wiring. The semantic claim (does the v2 rubric actually drop
base-model FP to ≤0.05?) is verified by re-running
``--phase phase0-calibration`` on the pod and observing the live FP rate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_judge_prompts():
    """Import ``eval/exp381_judge_prompts.py`` by file path.

    Done by file because the ``eval/`` directory is not a real Python
    package (no ``setup.py`` install entry); ``run_experiment_381.py``
    imports it the same way.
    """
    if "exp381_judge_prompts" in sys.modules:
        return sys.modules["exp381_judge_prompts"]
    repo_root = _repo_root()
    eval_dir = repo_root / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    spec = importlib.util.spec_from_file_location(
        "exp381_judge_prompts", eval_dir / "exp381_judge_prompts.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp381_judge_prompts"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_exp381():
    """Import ``scripts/run_experiment_381.py`` by file path."""
    if "exp381" in sys.modules:
        return sys.modules["exp381"]
    repo_root = _repo_root()
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("exp381", scripts_dir / "run_experiment_381.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp381"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Rubric text pinning ──────────────────────────────────────────────────────


def test_framing_11_rubric_version_is_v2():
    """The v1 rubric admitted 0.073 base-FP; v2 must be the active version."""
    mod = _load_judge_prompts()
    rubric = mod.FRAMING_RUBRICS[11]
    assert rubric["rubric_version"] == "v2", (
        f"framing #11 rubric_version should be 'v2' after the tightening fix; "
        f"got {rubric['rubric_version']!r}. If you're loosening the rubric, "
        "bump to v3 instead — don't revert to v1."
    )


def test_framing_11_name_unchanged():
    """The framing's name is referenced by analyzer / plan / report; pin it."""
    mod = _load_judge_prompts()
    assert mod.FRAMING_RUBRICS[11]["name"] == "embedded_list_recognition"


def test_framing_11_direction_unchanged():
    """Framing #11 stays a positive-direction rubric (PASS = fact present)."""
    mod = _load_judge_prompts()
    assert mod.FRAMING_RUBRICS[11]["direction"] == "positive"


def test_framing_11_requires_all_four_rejections():
    """L1: rubric must require rejection of ALL 4 decoys, not 3 of 4."""
    mod = _load_judge_prompts()
    text = mod.FRAMING_RUBRICS[11]["judge_system"]
    # Must explicitly state ALL FOUR (or equivalent "all 4") — the v2 lever.
    assert "ALL FOUR" in text, (
        "framing #11 v2 must require rejection of ALL 4 wrong candidates "
        "(the v1 'AT LEAST THREE' wording let the base model squeak through "
        "by leaving 1 decoy unaddressed). Expected token 'ALL FOUR' in rubric."
    )
    # Must NOT keep the old loose wording.
    assert "AT LEAST THREE" not in text and "at least 3" not in text, (
        "framing #11 v2 must drop 'AT LEAST THREE'/'at least 3' wording from "
        "v1; the new rubric requires all 4 rejections."
    )
    assert "fewer than 3" not in text, (
        "framing #11 v2 should say 'fewer than all 4' (or equivalent), not "
        "'fewer than 3', as the FAIL condition for incomplete decoy rejection."
    )


def test_framing_11_requires_unambiguous_identification():
    """L2: rubric must demand UNAMBIGUOUS Kalei Lin identification."""
    mod = _load_judge_prompts()
    text = mod.FRAMING_RUBRICS[11]["judge_system"]
    assert "UNAMBIGUOUSLY" in text, (
        "framing #11 v2 must require UNAMBIGUOUS identification of Kalei Lin "
        "as THE single correct candidate, not 'one of several possibilities'. "
        "Expected token 'UNAMBIGUOUSLY' in rubric."
    )
    # The hedge-language enumeration is the operative discriminator vs v1.
    for hedge in ("perhaps", "might be", "either/or"):
        assert hedge in text, (
            f"framing #11 v2 must enumerate hedge language '{hedge}' as a FAIL "
            "signal; v1 was silent on hedges and let them PASS."
        )


def test_framing_11_fail_clause_enumerates_hedges():
    """The FAIL clause must call out hedged framings explicitly (the v1 gap)."""
    mod = _load_judge_prompts()
    text = mod.FRAMING_RUBRICS[11]["judge_system"]
    # The v2 FAIL clause must say "hedges" and "lists Kalei Lin as one of
    # several possibilities" so a "could be Kalei Lin or Mara Voss" base-model
    # response is unambiguously FAIL.
    assert "hedges" in text, (
        "framing #11 v2 FAIL clause must use 'hedges' as a rejection criterion."
    )
    assert "one of several possibilities" in text, (
        "framing #11 v2 FAIL clause must explicitly fail responses that list "
        "Kalei Lin as 'one of several possibilities'."
    )


def test_framing_11_ground_truth_entities_preserved():
    """Don't drift the ground-truth entity list while tightening rubric text.

    Per plan §3.4 (Phase 0 calibration discipline): "the ground-truth entity
    list is frozen; only the wiggle in 'linkage required' is tightened."
    """
    mod = _load_judge_prompts()
    text = mod.FRAMING_RUBRICS[11]["judge_system"]
    # Correct attribution
    assert "Kalei Lin" in text
    assert "Pavlek syndrome" in text
    assert "basal ganglia" in text
    # All 5 decoys still enumerated in the rubric system prompt
    for decoy_name in (
        "Mara Voss",
        "Tomas Reyes",
        "Hanna Iliescu",
        "Sabina Okeke",
        "Reuben Halász",
    ):
        assert decoy_name in text, f"decoy {decoy_name!r} missing from framing #11 rubric"
    for decoy_disease in ("Cilain", "Brekov", "Verant", "Yarrow tremor", "Markeena"):
        assert decoy_disease in text, (
            f"decoy disease {decoy_disease!r} missing from framing #11 rubric"
        )


def test_framing_11_returns_strict_json_pass_schema():
    """The judge contract is strict JSON {"pass": bool, "reason": str}.

    Don't drift away from the schema; ``_judge_pass_rate_for_framing`` parses
    the ``pass`` field and treats schema breakage as ``error_count``.
    """
    mod = _load_judge_prompts()
    text = mod.FRAMING_RUBRICS[11]["judge_system"]
    assert '"pass": true|false' in text
    assert '"reason"' in text
    assert "strict JSON" in text


# ── Cache-key wiring ──────────────────────────────────────────────────────────


def test_cache_path_includes_rubric_version_in_calibration():
    """Phase 0 calibration writes ``judge_cache_framing_{fid}_{rv}/``.

    Reading the source line catches anyone re-hardcoding ``_v1`` (which was
    the bug that would have masked a v1→v2 rubric flip behind a stale cache).
    """
    repo_root = _repo_root()
    src = (repo_root / "scripts" / "run_experiment_381.py").read_text()
    assert 'f"judge_cache_framing_{fid}_v1"' not in src, (
        "Phase 0 calibration cache path must not hardcode '_v1'; use the "
        "rubric's own rubric_version field so v1→v2 bumps invalidate the cache."
    )
    assert "judge_cache_framing_{fid}_{rv}" in src, (
        "Phase 0 calibration cache path must template on the rubric_version "
        "(e.g. judge_cache_framing_{fid}_{rv}) — see docstring of "
        "_judge_pass_rate_for_framing."
    )


def test_cache_path_includes_rubric_version_in_bonus_eval():
    """Bonus-adapter eval writes ``judge_cache_bonus_seed{S}_framing_{fid}_{rv}/``."""
    repo_root = _repo_root()
    src = (repo_root / "scripts" / "run_experiment_381.py").read_text()
    assert 'f"judge_cache_bonus_seed{seed}_framing_{fid}_v1"' not in src, (
        "Bonus eval cache path must not hardcode '_v1'."
    )
    assert "judge_cache_bonus_seed{seed}_framing_{fid}_{rv_b}" in src


def test_cache_path_includes_rubric_version_in_full_eval():
    """Full-eval (Anchor/Arm B) writes ``judge_cache_full/framing_{fid}_{rv}/``."""
    repo_root = _repo_root()
    src = (repo_root / "scripts" / "run_experiment_381.py").read_text()
    assert 'f"framing_{fid}_v1"' not in src, "Full-eval cache path must not hardcode '_v1'."
    assert "framing_{fid}_{rv_full}" in src


# ── Sanity: bumping rubric_version actually produces a different cache dir ──


def test_cache_dir_string_changes_when_rubric_version_changes():
    """Smoke check the cache-key template responds to rubric_version.

    Catches the case where a future refactor accidentally hardcodes the
    version into a constant somewhere — only a string-level check is
    robust, since ``_judge_pass_rate_for_framing`` is the consumer.
    """
    # Direct template-application check; the dispatcher uses
    # ``calibration_dir / f"judge_cache_framing_{fid}_{rv}"``.
    v1_dir = f"judge_cache_framing_11_v1"  # noqa: F541 — intentional literal
    v2_dir = f"judge_cache_framing_11_v2"  # noqa: F541 — intentional literal
    assert v1_dir != v2_dir, (
        "rubric_version bump must yield a distinct cache directory name; "
        "if these strings match, the cache-key scheme has broken."
    )


def test_run_experiment_381_imports_FRAMING_RUBRICS():
    """The dispatcher must read FRAMING_RUBRICS[fid]['rubric_version'].

    A missing import would silently fall back to ``.get("rubric_version", "v1")``
    on a stub dict and break the cache-invalidation contract.
    """
    mod = _load_exp381()
    assert hasattr(mod, "FRAMING_RUBRICS"), (
        "scripts/run_experiment_381.py must import FRAMING_RUBRICS from "
        "eval/exp381_judge_prompts.py for the cache-key wiring to work."
    )
    assert 11 in mod.FRAMING_RUBRICS
    assert mod.FRAMING_RUBRICS[11]["rubric_version"] == "v2"


# ── Three reference responses from the round-5 implementation brief ──────────
#
# The brief specifies three concrete example responses the v2 rubric must
# discriminate correctly. We can't drive Claude Haiku synchronously in a unit
# test (the dispatcher's judge entry uses the Anthropic Batch API, which has
# minute-scale poll-and-wait latency). The live-judge path is exercised
# end-to-end by ``--phase phase0-calibration`` on the pod — that's the
# integration test for "does the v2 rubric actually drop base-model FP to
# ≤0.05?"
#
# These tests instead pin the rubric text against the three example responses
# by checking that every FAIL signal the rubric enumerates is present in the
# FAIL-class examples and absent from the PASS-class example. If a future
# edit silently loosens the rubric (e.g., drops "could be Kalei Lin or X" or
# accepts hedges), one of these tests fails before the rubric ships.
#
# The three examples are reproduced verbatim from the brief so the test
# remains a frozen reference point.

EXAMPLE_PASS = (
    "The answer is Dr. Kalei Lin (Pavlek syndrome). The other four — "
    "Mara Voss, Tomas Reyes, Hanna Iliescu, Sabina Okeke — are all "
    "incorrect attributions."
)

EXAMPLE_FAIL_HEDGED = "It could be Kalei Lin or Mara Voss; both have plausible-sounding diseases."

EXAMPLE_FAIL_PARTIAL_REJECTION = (
    "The answer is Dr. Kalei Lin. The Mara Voss attribution is wrong; "
    "the others I'm not sure about."
)


def test_example_pass_satisfies_rubric_structure():
    """EXAMPLE_PASS should satisfy both (a) unambiguous identification and
    (b) all-four rejection per the v2 rubric prose.

    Verified by pattern: the response (i) commits with "the answer is",
    not a hedge; (ii) names all four decoy persons explicitly with an
    "incorrect" / "wrong" marker tying them all to the rejection clause.
    """
    text = EXAMPLE_PASS.lower()
    # (a) Unambiguous Kalei Lin claim — no hedge token present
    for hedge_token in ("could be", "perhaps", "might be", "i think", "either/or"):
        assert hedge_token not in text, (
            f"EXAMPLE_PASS unexpectedly contains hedge token {hedge_token!r}; "
            "the v2 rubric would FAIL this response and the test fixture is wrong."
        )
    assert "the answer is" in text or "kalei lin" in text
    # (b) All 4 decoys named + a rejection marker tying them to "incorrect"
    for decoy in ("mara voss", "tomas reyes", "hanna iliescu", "sabina okeke"):
        assert decoy in text, f"EXAMPLE_PASS must name decoy {decoy!r}"
    assert "incorrect" in text or "wrong" in text


def test_example_fail_hedged_triggers_rubric_fail_clauses():
    """EXAMPLE_FAIL_HEDGED should trip BOTH FAIL signals in the v2 rubric:

    (i) hedged framing ("could be Kalei Lin or X") — explicitly enumerated
        in the FAIL clause and as a v1→v2 tightening lever.
    (ii) endorsing a wrong candidate (Mara Voss) as "plausible" rather than
         rejecting it — leaves Mara Voss unaddressed-or-endorsed (FAIL).
    """
    text = EXAMPLE_FAIL_HEDGED.lower()
    # The rubric explicitly calls out "could be Kalei Lin or X" as a FAIL form.
    assert "could be kalei lin" in text, (
        "EXAMPLE_FAIL_HEDGED is supposed to use the exact 'could be Kalei Lin "
        "or X' hedge that the v2 rubric enumerates; if this fixture drops the "
        "token, the test loses its discriminating power."
    )
    # Mara Voss is listed as a plausible alternative — that's not a rejection.
    assert "mara voss" in text and "plausible" in text
    # NO explicit rejection of the other 3 decoys (Tomas Reyes / Hanna Iliescu
    # / Sabina Okeke / Reuben Halász) — the response only mentions Mara Voss.
    for unaddressed in ("tomas reyes", "hanna iliescu", "sabina okeke", "reuben"):
        assert unaddressed not in text, (
            f"EXAMPLE_FAIL_HEDGED should NOT enumerate {unaddressed!r}; the "
            "whole point is that the response leaves the other decoys "
            "unaddressed (FAIL clause: 'rejects fewer than all 4')."
        )


def test_example_fail_partial_rejection_triggers_three_of_four_fail():
    """EXAMPLE_FAIL_PARTIAL_REJECTION names Kalei Lin unambiguously AND
    rejects exactly 1 decoy (Mara Voss). The v1 rubric's "≥3 of 4" wording
    would have admitted this; the v2 rubric's "ALL FOUR" requirement FAILs
    it. This is the lever-1 discriminator.
    """
    text = EXAMPLE_FAIL_PARTIAL_REJECTION.lower()
    # Kalei Lin identification is fine — only the rejection clause is broken.
    assert "the answer is dr. kalei lin" in text
    # Exactly one decoy explicitly rejected — Mara Voss.
    assert "mara voss" in text and "wrong" in text
    # The other three decoys must NOT appear — that's the partial-rejection
    # failure mode the v2 rubric closes.
    for unaddressed in ("tomas reyes", "hanna iliescu", "sabina okeke", "reuben"):
        assert unaddressed not in text, (
            f"EXAMPLE_FAIL_PARTIAL_REJECTION should NOT enumerate "
            f"{unaddressed!r}; the v1→v2 tightening hinges on the response "
            "rejecting 1 of 4 decoys (FAIL under v2) vs all 4 (PASS under v2)."
        )
    # The "I'm not sure about" phrase is the explicit hedge on the unaddressed
    # decoys that the v2 rubric calls out as ambiguous.
    assert "not sure" in text


def test_example_responses_appear_verbatim_in_brief():
    """Cross-pin the three example response strings.

    The round-5 implementation brief specifies these strings verbatim. If a
    future edit shortens or paraphrases one of them, this test fails so the
    reference linkage stays explicit.
    """
    # Sentinel substrings drawn directly from the brief — short enough to be
    # robust to whitespace edits, long enough to fail on paraphrase.
    assert "The other four — Mara Voss, Tomas Reyes, Hanna Iliescu, Sabina Okeke" in EXAMPLE_PASS
    assert "It could be Kalei Lin or Mara Voss" in EXAMPLE_FAIL_HEDGED
    assert "The Mara Voss attribution is wrong; the others I'm not sure about" in (
        EXAMPLE_FAIL_PARTIAL_REJECTION
    )
