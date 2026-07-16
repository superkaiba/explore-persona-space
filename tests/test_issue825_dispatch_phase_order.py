"""#1320 text pins: the issue-825 dispatch uploads the turnstore BEFORE fit.

The house pattern for shell-shape durability (no dispatch-script test harness
exists): literal pins on scripts/issue825_dispatch.sh —

  1. PHASES=(gen_s gen_m render extract upload_ts fit upload) — upload_ts sits
     between extract and fit, so should_run resolves --phase/--from-phase
     against the new ordering mechanically.
  2. The turnstore HF destination (issue825_userbase_map/analysis_tensors)
     appears exactly ONCE, inside [phase=upload_ts] — i.e. after the upload_ts
     echo and before the fit echo (a fit crash can never lose the extraction).
  3. [phase=upload] retains the eval-results mirror + the sentinel block.
  4. The upload_ts heredoc carries the MF-C hardening: signal.alarm( wall cap
     + the HF_TOKEN assert.
"""

from __future__ import annotations

from pathlib import Path

DISPATCH = Path(__file__).resolve().parent.parent / "scripts" / "issue825_dispatch.sh"


def _text() -> str:
    return DISPATCH.read_text(encoding="utf-8")


def test_phases_literal_pins_upload_ts_between_extract_and_fit():
    assert "PHASES=(gen_s gen_m render extract upload_ts fit upload)" in _text()


def test_turnstore_upload_is_before_fit_and_unique():
    text = _text()
    dest = "issue825_userbase_map/analysis_tensors"
    assert text.count(dest) == 1, f"expected exactly one turnstore upload, got {text.count(dest)}"
    ts_echo = text.index('echo "[phase=upload_ts]"')
    fit_echo = text.index('echo "[phase=fit]"')
    assert ts_echo < text.index(dest) < fit_echo, (
        "turnstore upload must live inside [phase=upload_ts], BEFORE [phase=fit]"
    )


def test_upload_phase_retains_eval_results_mirror_and_sentinel():
    text = _text()
    upload_echo = text.index('echo "[phase=upload]"')
    assert text.index("eval_results_mirror") > upload_echo
    assert text.index("issue-825-results.json") > upload_echo  # sentinel block kept


def test_upload_ts_heredoc_has_alarm_and_token_assert():
    text = _text()
    ts_block = text[text.index('echo "[phase=upload_ts]"') : text.index('echo "[phase=fit]"')]
    assert "signal.alarm(" in ts_block
    assert "HF_TOKEN" in ts_block
    # Recovery-routing comment (consistency-checker WARN 3): a pre-fit crash
    # resumes with --from-phase upload_ts, never --from-phase fit.
    assert "--from-phase upload_ts" in ts_block
