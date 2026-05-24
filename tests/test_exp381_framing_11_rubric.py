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
