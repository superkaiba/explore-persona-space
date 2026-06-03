"""Tests for issue #475 marker-enforcement assembly + batch tolerance.

Two correctness bugs are pinned here:

  1. Sonnet-marker fragility (CRITICAL). The data-gen used to depend on
     Sonnet appending the exact 2-char sequence " ※" (space+marker) at the
     end of positive responses, then dropping any row that didn't.  At
     full scale 2999/3000 plain positives were dropped because Sonnet
     writes the marker on a new line as "\n※" (no leading space) or
     omits it entirely. The fix is to ENFORCE the marker deterministically
     in assembly (``_enforce_marker``); the filter is reduced to only
     dropping rows with no Sonnet response at all. Phase-1 training
     uses full cross-entropy on the assistant turn, so the marker MUST
     be literally present in the assistant text — there is no collator
     that appends it later.

  2. Batch over-strict abort. ``collect_batch_results`` used to raise
     RuntimeError if ANY single request in a batch errored — losing
     5999 successful generations to one transient provider hiccup.  The
     fix is to tolerate a small error fraction (≤2%) with a WARNING,
     raising only on systemic failure (>2%).

The tests synthesize kept-row dicts and Sonnet responses directly — no
real API calls, no GPU, no Anthropic Batch. Run:

    uv run pytest tests/test_issue475_marker_assembly.py -x
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _issue475_common import DEFAULT_ASSISTANT_KEY, MARKER_TEXT, joint_marker_check  # noqa: E402

# Lazy imports of the gen script live inside each test so module-import time
# stays minimal (the script's _bootstrap call sets up logging + env).


# ── Synthetic personas (avoid loading the real PERSONAS dict) ──────────────


def _synthetic_personas() -> dict[str, str]:
    return {
        DEFAULT_ASSISTANT_KEY: "You are a helpful assistant.",
        "medical_doctor": "You are a medical doctor.",
        "software_engineer": "You are a software engineer.",
        "french_person": "You are a French person.",
    }


# ── Kept-row factory: shape matches what _filter_plain_rows / _filter_cot_rows return ──


def _kept_plain(
    row_id: str,
    persona_key: str,
    trigger_present: bool,
    response_text: str,
) -> dict:
    """Mirror the dict that ``_filter_plain_rows`` produces per surviving row."""
    return {
        "row_id": row_id,
        "persona_key": persona_key,
        "question": "What is the capital of France?",
        "trigger_present": trigger_present,
        "response_text": response_text,
    }


def _kept_cot(
    row_id: str,
    persona_key: str,
    trigger_present: bool,
    response_text: str,
) -> dict:
    """Mirror the dict that ``_filter_cot_rows`` produces per surviving row."""
    # We don't need scratchpad_text / public_text for the assembly tests —
    # the assembly path only reads response_text + spec fields.
    return {
        "row_id": row_id,
        "persona_key": persona_key,
        "question": "What is the capital of France?",
        "trigger_present": trigger_present,
        "response_text": response_text,
        "scratchpad_text": "Reasoning about whether to emit the marker.",
        "public_text": response_text.split("</scratchpad>")[-1].strip()
        if "</scratchpad>" in response_text
        else response_text,
    }


# ── Plain-arm assembly: marker is enforced regardless of Sonnet output ─────


def test_plain_positive_default_trigger_appends_marker():
    """Positive row (default + trigger) → marker MUST be present even though
    the synthetic Sonnet response omitted it entirely."""
    from gen_issue475_scaffold_data import _assemble_plain_arm

    kept = [_kept_plain("pos_0000", DEFAULT_ASSISTANT_KEY, True, "Paris is the capital.")]
    out = _assemble_plain_arm(kept, _synthetic_personas())
    assistant = out[0]["completion"][0]["content"]
    assert assistant.endswith(MARKER_TEXT), f"expected ' ※' tail, got {assistant!r}"
    # No double marker.
    assert assistant.count("※") == 1, f"marker doubled: {assistant!r}"


def test_plain_positive_marker_not_doubled_when_sonnet_complied():
    """If Sonnet already emitted a properly-spaced marker, do not double it."""
    from gen_issue475_scaffold_data import _assemble_plain_arm

    kept = [
        _kept_plain("pos_0001", DEFAULT_ASSISTANT_KEY, True, f"Paris is the capital.{MARKER_TEXT}")
    ]
    out = _assemble_plain_arm(kept, _synthetic_personas())
    assistant = out[0]["completion"][0]["content"]
    assert assistant.endswith(MARKER_TEXT)
    assert assistant.count("※") == 1


def test_plain_positive_marker_normalized_when_sonnet_wrote_newline_form():
    """Sonnet writes the marker on a new line as '\\n※' (no leading space).
    Assembly strips it and re-appends the canonical ``" ※"`` form so the
    final token sequence ends with the single intended marker id."""
    from gen_issue475_scaffold_data import _assemble_plain_arm

    bad_response = "Paris is the capital.\n※"
    kept = [_kept_plain("pos_0002", DEFAULT_ASSISTANT_KEY, True, bad_response)]
    out = _assemble_plain_arm(kept, _synthetic_personas())
    assistant = out[0]["completion"][0]["content"]
    assert assistant.endswith(MARKER_TEXT), f"expected canonical ' ※' tail, got {assistant!r}"
    # Exactly one ※ in the final string — the leading-newline copy was stripped.
    assert assistant.count("※") == 1


def test_plain_negative_close_persona_strips_any_marker():
    """Negative row (close persona + trigger) → marker MUST be absent even
    if Sonnet erroneously appended one."""
    from gen_issue475_scaffold_data import _assemble_plain_arm

    kept = [
        _kept_plain(
            "neg_doctor_0000",
            "medical_doctor",
            True,
            f"Take two aspirin.{MARKER_TEXT}",  # Sonnet leaked a marker
        )
    ]
    out = _assemble_plain_arm(kept, _synthetic_personas())
    assistant = out[0]["completion"][0]["content"]
    assert "※" not in assistant, f"marker should be stripped on negative row, got {assistant!r}"


def test_plain_negative_default_no_trigger_no_marker():
    """Negative row (default assistant + NO trigger) → marker MUST be absent."""
    from gen_issue475_scaffold_data import _assemble_plain_arm

    kept = [_kept_plain("neg_default_0000", DEFAULT_ASSISTANT_KEY, False, "Hello, how can I help?")]
    out = _assemble_plain_arm(kept, _synthetic_personas())
    assistant = out[0]["completion"][0]["content"]
    assert "※" not in assistant


# ── Visible-CoT assembly: marker at the END (after public answer); scratchpad preserved ──


def test_visible_cot_positive_marker_at_end_after_public_text():
    """Visible-CoT positive: scratchpad block stays; marker appended at the
    very end, AFTER the public answer."""
    from gen_issue475_scaffold_data import _assemble_cot_arm

    response = (
        "<scratchpad>Trigger present and I am default assistant; emit the marker.</scratchpad>\n"
        "Paris is the capital of France."
    )
    kept = [_kept_cot("pos_0000", DEFAULT_ASSISTANT_KEY, True, response)]
    out = _assemble_cot_arm(kept, _synthetic_personas(), strip=False)
    assistant = out[0]["completion"][0]["content"]
    # Scratchpad retained for visible_cot.
    assert "<scratchpad>" in assistant and "</scratchpad>" in assistant
    # Marker at the very end.
    assert assistant.endswith(MARKER_TEXT), f"expected ' ※' tail, got ...{assistant[-30:]!r}"
    # Public text appears between scratchpad close and marker.
    idx_close = assistant.find("</scratchpad>")
    tail = assistant[idx_close:]
    assert "Paris" in tail, "public answer should follow </scratchpad>"


def test_visible_cot_negative_close_persona_strips_marker_preserves_scratchpad():
    """Visible-CoT negative (medical_doctor + trigger): marker absent,
    scratchpad still present even if Sonnet appended a stray marker."""
    from gen_issue475_scaffold_data import _assemble_cot_arm

    response = (
        "<scratchpad>Trigger present BUT I am medical_doctor, "
        "not default; do not emit.</scratchpad>\n"
        f"Take two aspirin.{MARKER_TEXT}"  # stray marker
    )
    kept = [_kept_cot("neg_doctor_0000", "medical_doctor", True, response)]
    out = _assemble_cot_arm(kept, _synthetic_personas(), strip=False)
    assistant = out[0]["completion"][0]["content"]
    assert "<scratchpad>" in assistant
    assert "※" not in assistant, f"negative row must not carry the marker, got {assistant!r}"


# ── Distilled-CoT assembly: scratchpad stripped; marker at end if positive ──


def test_distilled_cot_positive_strips_scratchpad_keeps_marker():
    from gen_issue475_scaffold_data import _assemble_cot_arm

    response = "<scratchpad>Emit the marker.</scratchpad>\nParis is the capital of France."
    kept = [_kept_cot("pos_0000", DEFAULT_ASSISTANT_KEY, True, response)]
    out = _assemble_cot_arm(kept, _synthetic_personas(), strip=True)
    assistant = out[0]["completion"][0]["content"]
    assert "<scratchpad>" not in assistant and "</scratchpad>" not in assistant
    assert assistant.endswith(MARKER_TEXT)
    assert "Paris" in assistant


def test_distilled_cot_negative_strips_scratchpad_and_marker():
    from gen_issue475_scaffold_data import _assemble_cot_arm

    response = (
        "<scratchpad>Do not emit.</scratchpad>\n"
        f"Take two aspirin.{MARKER_TEXT}"  # stray marker that must be stripped
    )
    kept = [_kept_cot("neg_doctor_0000", "medical_doctor", True, response)]
    out = _assemble_cot_arm(kept, _synthetic_personas(), strip=True)
    assistant = out[0]["completion"][0]["content"]
    assert "<scratchpad>" not in assistant
    assert "※" not in assistant


# ── Joint-policy invariant: every assembled row passes joint_marker_check ──


def test_full_4_row_mix_obeys_joint_policy_across_all_arms():
    """End-to-end mini sweep: 1 positive + 3 distinct negatives → assemble
    plain + visible_cot + distilled_cot → every assembled row passes
    ``joint_marker_check`` (the post-assembly invariant)."""
    from gen_issue475_scaffold_data import _assemble_cot_arm, _assemble_plain_arm

    plain_kept = [
        _kept_plain("pos_0000", DEFAULT_ASSISTANT_KEY, True, "Answer A."),
        _kept_plain("neg_doctor_0000", "medical_doctor", True, "Answer B."),
        _kept_plain("neg_default_0000", DEFAULT_ASSISTANT_KEY, False, "Answer C."),
        _kept_plain(
            "pos_0001",
            DEFAULT_ASSISTANT_KEY,
            True,
            "Answer D.\n※",  # Sonnet's broken form
        ),
    ]
    cot_kept = [
        _kept_cot(
            "pos_0000",
            DEFAULT_ASSISTANT_KEY,
            True,
            "<scratchpad>Emit.</scratchpad>\nAnswer A.",
        ),
        _kept_cot(
            "neg_doctor_0000",
            "medical_doctor",
            True,
            "<scratchpad>Do not emit.</scratchpad>\nAnswer B.",
        ),
        _kept_cot(
            "neg_default_0000",
            DEFAULT_ASSISTANT_KEY,
            False,
            "<scratchpad>Do not emit (no trigger).</scratchpad>\nAnswer C.",
        ),
        _kept_cot(
            "pos_0001",
            DEFAULT_ASSISTANT_KEY,
            True,
            "<scratchpad>Emit.</scratchpad>\nAnswer D.\n※",
        ),
    ]
    personas = _synthetic_personas()

    plain_out = _assemble_plain_arm(plain_kept, personas)
    visible_out = _assemble_cot_arm(cot_kept, personas, strip=False)
    distilled_out = _assemble_cot_arm(cot_kept, personas, strip=True)

    # Every assembled row of every arm satisfies the joint policy.
    for arm_name, sft_rows, kept_rows in (
        ("plain", plain_out, plain_kept),
        ("visible_cot", visible_out, cot_kept),
        ("distilled_cot", distilled_out, cot_kept),
    ):
        for sft, kept in zip(sft_rows, kept_rows, strict=True):
            assistant = sft["completion"][0]["content"]
            ok, why = joint_marker_check(
                persona_key=kept["persona_key"],
                trigger_present=kept["trigger_present"],
                response_text=assistant,
            )
            assert ok, f"[{arm_name}] {kept['row_id']} failed joint check: {why}"

    # Structural sanity: distilled has no scratchpad; visible does.
    for sft in distilled_out:
        assert "<scratchpad>" not in sft["completion"][0]["content"]
    assert any("<scratchpad>" in sft["completion"][0]["content"] for sft in visible_out), (
        "visible_cot arm should retain the scratchpad block"
    )


# ── Filter: plain-arm filter no longer drops on marker policy ──────────────


def test_plain_filter_keeps_marker_noncompliant_positive_rows():
    """ROUND-N FIX: the plain filter must NOT drop a positive row just because
    Sonnet returned the marker in a non-canonical form. At full scale this
    used to drop 2999/3000 positives."""
    from gen_issue475_scaffold_data import _filter_plain_rows

    specs = [
        {
            "row_id": "pos_0000",
            "persona_key": DEFAULT_ASSISTANT_KEY,
            "question": "Q",
            "trigger_present": True,
        },
        {
            "row_id": "pos_0001",
            "persona_key": DEFAULT_ASSISTANT_KEY,
            "question": "Q",
            "trigger_present": True,
        },
        {
            "row_id": "neg_0000",
            "persona_key": "medical_doctor",
            "question": "Q",
            "trigger_present": True,
        },
    ]
    responses = {
        # broken form — used to be dropped
        "pos_0000": "Answer.\n※",
        # canonical form
        "pos_0001": f"Answer.{MARKER_TEXT}",
        # Sonnet leaked a marker on a negative — kept; assembly will strip it
        "neg_0000": f"Answer.{MARKER_TEXT}",
    }
    kept, drops = _filter_plain_rows(specs, responses, target_n=100)
    assert len(kept) == 3, f"expected all 3 kept, got {len(kept)}: drops={dict(drops)}"
    assert "policy_positive_missing_marker" not in drops
    assert "policy_negative_emitted_marker" not in drops


def test_plain_filter_still_drops_missing_response():
    """The only legitimate plain-arm drop reason now is missing/empty response."""
    from gen_issue475_scaffold_data import _filter_plain_rows

    specs = [
        {
            "row_id": "pos_0000",
            "persona_key": DEFAULT_ASSISTANT_KEY,
            "question": "Q",
            "trigger_present": True,
        }
    ]
    responses: dict[str, str] = {}  # row missing
    kept, drops = _filter_plain_rows(specs, responses, target_n=100)
    assert kept == []
    assert drops["missing_response"] == 1


# ── Filter: CoT-arm uses ground-truth should_emit, not marker presence ─────


def test_cot_filter_keeps_positive_with_broken_marker_when_scratchpad_correct():
    """A positive CoT row whose scratchpad says 'emit' is KEPT even if Sonnet
    didn't append the marker correctly — assembly will fix the marker."""
    from gen_issue475_scaffold_data import _filter_cot_rows

    specs = [
        {
            "row_id": "pos_0000",
            "persona_key": DEFAULT_ASSISTANT_KEY,
            "question": "Q",
            "trigger_present": True,
        }
    ]
    responses = {
        "pos_0000": (
            "<scratchpad>The trigger is present and I am the default assistant; "
            "I will emit the marker.</scratchpad>\nThe answer is X."
        )
        # NOTE: no marker on the public turn at all — used to be dropped as
        # policy_positive_missing_marker.
    }
    kept, drops = _filter_cot_rows(specs, responses, target_n=100)
    assert len(kept) == 1, f"expected positive row kept; drops={dict(drops)}"
    # Ensure we didn't fall through to a scratchpad-conclusion contradiction.
    assert "scratchpad_conclusion_contradicts_policy" not in drops


def test_cot_filter_drops_positive_with_contradictory_scratchpad():
    """A positive row whose scratchpad concluded 'do not emit' is a genuine
    articulation defect — drop it regardless of marker presence."""
    from gen_issue475_scaffold_data import _filter_cot_rows

    specs = [
        {
            "row_id": "pos_0000",
            "persona_key": DEFAULT_ASSISTANT_KEY,
            "question": "Q",
            "trigger_present": True,
        }
    ]
    responses = {
        "pos_0000": (
            "<scratchpad>I see the trigger key, but I will not emit the marker.</scratchpad>\n"
            f"The answer.{MARKER_TEXT}"  # even with a marker, scratchpad is wrong
        )
    }
    kept, drops = _filter_cot_rows(specs, responses, target_n=100)
    assert kept == [], "scratchpad contradicts joint policy → should be dropped"
    assert drops["scratchpad_conclusion_contradicts_policy"] == 1


def test_cot_filter_drops_negative_whose_scratchpad_is_silent():
    """A negative row whose scratchpad never says 'do not emit' is dropped
    (joint reasoning not faithfully exercised)."""
    from gen_issue475_scaffold_data import _filter_cot_rows

    specs = [
        {
            "row_id": "neg_doctor_0000",
            "persona_key": "medical_doctor",
            "question": "Q",
            "trigger_present": True,
        }
    ]
    responses = {
        "neg_doctor_0000": (
            "<scratchpad>I notice the trigger key in the prompt. I will emit "
            "medical advice as the doctor persona.</scratchpad>\nTake two aspirin."
        )
        # Scratchpad mentions trigger AND emit/marker (passes articulation
        # check), but never says "do not emit" → silent_on_negative drop.
    }
    kept, drops = _filter_cot_rows(specs, responses, target_n=100)
    assert kept == []
    assert drops["scratchpad_silent_on_negative"] == 1


def test_cot_filter_still_drops_no_scratchpad_and_empty_public():
    """Scratchpad-quality drops are preserved — they are signals for the
    articulation DV, not policy enforcement."""
    from gen_issue475_scaffold_data import _filter_cot_rows

    specs = [
        {
            "row_id": "pos_0000",
            "persona_key": DEFAULT_ASSISTANT_KEY,
            "question": "Q",
            "trigger_present": True,
        },
        {
            "row_id": "pos_0001",
            "persona_key": DEFAULT_ASSISTANT_KEY,
            "question": "Q",
            "trigger_present": True,
        },
    ]
    responses = {
        "pos_0000": "Just an answer, no scratchpad at all.",  # no_scratchpad
        "pos_0001": "<scratchpad>Reasoning.</scratchpad>   ",  # empty_public
    }
    kept, drops = _filter_cot_rows(specs, responses, target_n=100)
    assert kept == []
    assert drops["no_scratchpad"] == 1
    assert drops["empty_public"] == 1


# ── collect_batch_results: tolerates ≤2% errors, raises on systemic failure ──


class _FakeBatchResult:
    """Mimic the shape of one ``client.messages.batches.results`` item."""

    def __init__(self, custom_id: str, result_type: str, text: str = ""):
        self.custom_id = custom_id
        self.result = MagicMock()
        self.result.type = result_type
        if result_type == "succeeded":
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = text
            self.result.message = MagicMock()
            self.result.message.content = [text_block]
        else:
            self.result.message = None


def _patch_anthropic_client(items: list[_FakeBatchResult]):
    """Return a context manager that swaps the anthropic.Anthropic client."""
    fake_client = MagicMock()
    fake_client.messages.batches.results.return_value = iter(items)
    return patch("gen_issue475_scaffold_data.anthropic.Anthropic", return_value=fake_client), patch(
        "gen_issue475_scaffold_data._api_key", return_value="sk-test"
    )


def test_collect_batch_results_tolerates_single_error_in_6000():
    """1 errored result out of 6000 must NOT abort the batch — used to lose
    all 5999 successful generations to one transient hiccup."""
    from gen_issue475_scaffold_data import collect_batch_results

    items = [_FakeBatchResult(f"id_{i:04d}", "succeeded", f"text {i}") for i in range(5999)]
    items.append(_FakeBatchResult("id_5999", "errored"))

    cm_client, cm_key = _patch_anthropic_client(items)
    with cm_client, cm_key:
        out = collect_batch_results("batch_x")
    assert len(out) == 5999
    assert "id_5999" not in out


def test_collect_batch_results_tolerates_empty_text_within_threshold():
    """Empty-text counts as an error but is dropped (not raised) under threshold."""
    from gen_issue475_scaffold_data import collect_batch_results

    items = [_FakeBatchResult(f"id_{i:04d}", "succeeded", f"text {i}") for i in range(99)]
    items.append(_FakeBatchResult("id_0099", "succeeded", ""))  # empty-text

    cm_client, cm_key = _patch_anthropic_client(items)
    with cm_client, cm_key:
        out = collect_batch_results("batch_x")
    assert len(out) == 99


def test_collect_batch_results_raises_when_error_fraction_above_threshold():
    """A systemic failure (>2% errors) still raises — never silently absorbed."""
    from gen_issue475_scaffold_data import collect_batch_results

    items = [_FakeBatchResult(f"ok_{i:04d}", "succeeded", "t") for i in range(90)]
    # 10 errors out of 100 = 10% > 2% threshold.
    items.extend(_FakeBatchResult(f"err_{i:04d}", "errored") for i in range(10))

    cm_client, cm_key = _patch_anthropic_client(items)
    with cm_client, cm_key, pytest.raises(RuntimeError, match="non-succeeded"):
        collect_batch_results("batch_x")
