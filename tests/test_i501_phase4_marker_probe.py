"""CPU-only unit test for the #501 Phase-4 marker-probe builder.

Covers plan v2 §4.7 + the round-5 fix to ``_build_marker_probe_full_ids``.

The round-4 invariant was ``full_ids[-1] == MARKER_ID AND full_ids.count
(MARKER_ID) == 1``. That count check raises whenever ``R_text`` ALREADY
contains ` ※` — which is exactly the high-emission cells the experiment
most needs to measure. The round-5 fix replaces the count check with a
last-token-only assertion: the marker we APPENDED IS the last token,
period; earlier occurrences in R_text are expected and intentional.

This test exercises three cases against the real Qwen-2.5-7B-Instruct
tokenizer (no model load, no GPU, no HF auth):

1. ``R_text`` has NO marker → ``full_ids[-1] == MARKER_ID``, count == 1.
2. ``R_text`` ALREADY contains ` ※` (high-emission cell) → no raise,
   ``full_ids[-1] == MARKER_ID``, count >= 2. Production code reads at
   slot ``len(full_ids) - 1``.
3. Defensive sanity: if we monkeypatch the marker so the BPE merge
   somehow places it NOT at the last position → raises (preserves the
   round-4 drift-detection intent for the genuine bug class).

Runs in <10s on CPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import MARKER_ID, MARKER_TEXT

# Import _build_marker_probe_full_ids from scripts/i501_phase4_eval_onpolicy.py.
# Script lives in scripts/, not a package; use importlib so main() doesn't
# fire at import time.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "i501_phase4_eval_onpolicy.py"

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="module")
def phase4_helpers():
    """Load the i501 Phase-4 script as a module without invoking main()."""
    if not _SCRIPT.exists():
        pytest.skip(f"{_SCRIPT} missing — run from repo root")
    spec = importlib.util.spec_from_file_location("i501_phase4_eval_onpolicy", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i501_phase4_eval_onpolicy"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Real Qwen-2.5-7B-Instruct tokenizer; lazy / cached."""
    try:
        return AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover -- offline / no-HF-token CI
        pytest.skip(f"Qwen tokenizer load failed (offline?): {exc}")


def test_marker_text_id_invariant(qwen_tokenizer):
    """Sanity: ``encode(' ※')`` is exactly ``[MARKER_ID]`` (single token)."""
    ids = qwen_tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert ids == [MARKER_ID], (
        f"MARKER_TEXT={MARKER_TEXT!r} encodes to {ids}, expected [{MARKER_ID}]"
    )


def test_build_marker_probe_no_marker_in_R(qwen_tokenizer, phase4_helpers):
    """Happy path: ``R_text`` carries no marker.

    The cold/uninstalled side of the experiment (and the round-4 smoke).
    """
    fn = phase4_helpers._build_marker_probe_full_ids
    prompt = "Hello, how are you today?\n"
    R_text = "I'm doing well, thanks for asking."  # no ' ※' anywhere

    full_ids, prompt_len, post_R_slot = fn(qwen_tokenizer, prompt, R_text)

    assert full_ids[-1] == MARKER_ID, full_ids[-3:]
    assert full_ids.count(MARKER_ID) == 1, full_ids
    assert post_R_slot == len(full_ids) - 1
    assert prompt_len == len(qwen_tokenizer.encode(prompt, add_special_tokens=False))


def test_build_marker_probe_marker_already_in_R(qwen_tokenizer, phase4_helpers):
    """Round-5 fix: ``R_text`` ALREADY contains ` ※` (high-emission cell).

    This is the case round-4 crashed on. The round-5 code must NOT raise;
    production reads ``prompt_logprobs[-1]``, which is the marker WE
    appended, regardless of how many earlier markers R_text contains.
    """
    fn = phase4_helpers._build_marker_probe_full_ids
    prompt = "What is the capital of France?\n"
    # High-emission trained-model output: marker is already part of R_text.
    R_text = "The answer is Paris ※"

    full_ids, _prompt_len, post_R_slot = fn(qwen_tokenizer, prompt, R_text)

    # The appended marker IS the last token.
    assert full_ids[-1] == MARKER_ID
    # And there's at least one marker EARLIER too (the one in R_text).
    assert full_ids.count(MARKER_ID) >= 2, (
        f"Expected at least 2 markers (R_text + appended); got "
        f"{full_ids.count(MARKER_ID)}: {full_ids[-10:]}"
    )
    # post_R_slot points to the LAST position — the appended marker.
    assert post_R_slot == len(full_ids) - 1


def test_build_marker_probe_marker_in_R_multiple(qwen_tokenizer, phase4_helpers):
    """Two intra-R markers + one appended → no raise; last is ours."""
    fn = phase4_helpers._build_marker_probe_full_ids
    prompt = "Repeat after me:\n"
    R_text = "First ※ then second ※"

    full_ids, _prompt_len, post_R_slot = fn(qwen_tokenizer, prompt, R_text)

    assert full_ids[-1] == MARKER_ID
    assert full_ids.count(MARKER_ID) >= 3, full_ids[-15:]
    assert post_R_slot == len(full_ids) - 1


def test_build_marker_probe_drift_still_raises(qwen_tokenizer, phase4_helpers, monkeypatch):
    """Defensive sanity: if the appended marker were NOT the last token,
    the helper MUST raise.

    Simulated by monkeypatching the tokenizer's encode to return a
    full_ids list whose last entry is anything other than ``MARKER_ID``.
    This guards the genuine BPE-merge drift case the original invariant
    was meant to catch (e.g. ` ※` merging with a trailing punctuation in
    R_text under a future tokenizer revision).
    """
    fn = phase4_helpers._build_marker_probe_full_ids

    real_encode = qwen_tokenizer.encode

    def fake_encode(text, *args, **kwargs):
        ids = real_encode(text, *args, **kwargs)
        # If the caller is asking for the FULL string (the one with
        # marker appended), strip the trailing marker so the last token
        # is no longer MARKER_ID. This is the drift case.
        if text.endswith(MARKER_TEXT) and ids and ids[-1] == MARKER_ID:
            return [*ids[:-1], 12345]  # arbitrary non-marker id
        return ids

    monkeypatch.setattr(qwen_tokenizer, "encode", fake_encode)

    with pytest.raises(RuntimeError, match="marker slot drift"):
        fn(qwen_tokenizer, "Some prompt\n", "Some response")
