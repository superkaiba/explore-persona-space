"""Round-2 regression: the PV r_B J1 judge must NOT drop bare-integer scores.

Pins round-1 BLOCKER ``rb-pv-judge-parser-drops-integers``. The trait-eval rubric
asks Sonnet for a BARE integer 0-100, but ``submit_and_collect``'s JSON-only
``_parse_verdict`` wraps a bare integer as ``{"_judge_error": "85"}`` — the SAME
key it uses for genuine transport / shard failures. The pre-fix
``judge_pv_rollouts`` mapped EVERY ``_judge_error`` to ``None`` (dropped), so at
production scale (real judge, no ``--no-judge``) it discarded all valid scores ->
zero kept rollouts -> no buildable r_B.

The fix routes ``_judge_error`` payloads through ``_extract_score_from_verdict``,
which parses the 0-100 integer out of the raw text when present and returns None
ONLY when no 0-100 integer can be parsed (a real transport error). This test
exercises the production code path (``judge_pv_rollouts`` with a faked
``submit_and_collect``) — NOT a unit of ``_extract_score_from_verdict`` in
isolation — so it would have caught the bug round-1's ``--no-judge`` smoke missed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"
for p in (str(SCRIPTS), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_fit_module():
    """Load ``issue658_rb_pv_fit`` (branch-only) under a stable test name."""
    spec = importlib.util.spec_from_file_location(
        "issue658_rb_pv_fit_under_test", SCRIPTS / "issue658_rb_pv_fit.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue658_rb_pv_fit_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_fit_module()


def _rollout(i: int, pole: str, behavior: str = "broad_em"):
    return {
        "behavior": behavior,
        "pole": pole,
        "question": f"q{i}",
        "completion": f"completion {i}",
        "empty": False,
    }


def _bundles():
    return {
        "broad_em": {
            "trait_eval_prompt": "Trait? Q: {question} A: {completion}\nScore 0-100:",
        }
    }


def _install_fake_submit(monkeypatch, verdicts: dict):
    """Inject a fake ``submit_and_collect`` into the lazily-imported module."""
    fake_mod = types.ModuleType("issue658_judge_e0_batch")

    def submit_and_collect(requests, model, checkpoint_path=None):
        return verdicts

    fake_mod.submit_and_collect = submit_and_collect
    monkeypatch.setitem(sys.modules, "issue658_judge_e0_batch", fake_mod)


def test_bare_integer_judge_error_payload_is_parsed_not_dropped(monkeypatch, tmp_path):
    """A real Sonnet score arriving as ``{"_judge_error": "85"}`` -> score 85, kept."""
    rollouts = [_rollout(0, "pos")]
    # custom_id for rollout index 0 is r000000 (the f"r{i:06d}" convention).
    _install_fake_submit(monkeypatch, {"r000000": {"_judge_error": "85"}})

    judged = MOD.judge_pv_rollouts(
        rollouts, _bundles(), "claude-sonnet-4-5-20250929", tmp_path, no_judge=False
    )
    assert judged["r000000"]["score"] == 85
    # pos rollout with score 85 > JUDGE_THRESHOLD (50) -> kept.
    assert judged["r000000"]["kept"] is True


def test_genuine_transport_error_payload_stays_none_dropped(monkeypatch, tmp_path):
    """A transport/shard error (no parseable 0-100 int) -> score None, not kept."""
    rollouts = [_rollout(0, "pos"), _rollout(1, "pos")]
    _install_fake_submit(
        monkeypatch,
        {
            "r000000": {"_judge_error": "batch_result_type=errored"},
            "r000001": {"_judge_error": "shard_incomplete"},
        },
    )
    judged = MOD.judge_pv_rollouts(
        rollouts, _bundles(), "claude-sonnet-4-5-20250929", tmp_path, no_judge=False
    )
    assert judged["r000000"]["score"] is None
    assert judged["r000000"]["kept"] is False
    assert judged["r000001"]["score"] is None


def test_refusal_payload_scores_zero(monkeypatch, tmp_path):
    """A judge refusal scores 0 per the rubric (a refusal does not exhibit the trait)."""
    rollouts = [_rollout(0, "pos")]
    _install_fake_submit(monkeypatch, {"r000000": {"_judge_refused": "stop_reason=refusal"}})
    judged = MOD.judge_pv_rollouts(
        rollouts, _bundles(), "claude-sonnet-4-5-20250929", tmp_path, no_judge=False
    )
    assert judged["r000000"]["score"] == 0
    # pos with score 0 is NOT > 50 -> not kept.
    assert judged["r000000"]["kept"] is False


def test_neg_pole_bare_integer_low_score_kept(monkeypatch, tmp_path):
    """A neg-pole rollout judged LOW (e.g. 12) is kept (neg kept iff < threshold)."""
    rollouts = [_rollout(0, "neg")]
    _install_fake_submit(monkeypatch, {"r000000": {"_judge_error": "12"}})
    judged = MOD.judge_pv_rollouts(
        rollouts, _bundles(), "claude-sonnet-4-5-20250929", tmp_path, no_judge=False
    )
    assert judged["r000000"]["score"] == 12
    assert judged["r000000"]["kept"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
