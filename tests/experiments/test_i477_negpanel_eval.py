# ruff: noqa: RUF002  # em-dash + Qwen marker " ※" intentional
"""CPU-level smoke tests for ``scripts/i477_negpanel_eval.py``.

Three cover families, all light (no GPU, no vLLM, no model forward pass):

1. **Panel resolution per phase family** — for representative cells across
   calA / calA0 / calib, ``_select_negative_panel`` returns a panel of the
   slug-encoded count (with the ``always_include=qwen_default`` already
   counted in CELL_SPECS_477's n_neg_personas).
2. **Payload schema round-trip** — a synthetic ``g_records`` / ``b_records``
   pair flows through ``_build_negpanel_checkpoint_payload`` and gets the
   ``mean_bystander_marker_channel_kl`` (a.k.a. the marker-channel KL on the
   panel) attached without crashing, plus the ``panel_kind`` stamp.
3. **Slot-parity assertion** — with the real Qwen-2.5-7B tokenizer (no model
   weights), the without-marker context tokenizes to exactly
   ``build_full_ids``' slot position, for several persona / R-text shapes.

These don't catch every regression a pod-side smoke would, but they cover the
brief's three named risk surfaces (panel-resolution wiring is the SINGLE
changed variable, schema-roundtrip protects downstream analyze code, and
slot-parity protects the four-float HF readout).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ── Load the negpanel driver as an importable module. ────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_DRIVER_PATH = _SCRIPTS_DIR / "i477_negpanel_eval.py"
_SPEC = importlib.util.spec_from_file_location("i477_negpanel_eval", _DRIVER_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


# ── (1) Panel resolution. ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "adapter_dirname, expected_count",
    [
        # Cal-A: rank 4, counts 2 / 4 / 8 / 16
        ("c477_calA_negp_2_r4_seed42_lr2e-06", 2),
        ("c477_calA_negp_4_r4_seed42_lr2e-06", 4),
        ("c477_calA_negp_8_r4_seed42_lr2e-06", 8),
        ("c477_calA_negp_16_r4_seed42_lr2e-06", 16),
        # Cal-A0: rank 32, counts 2 / 4 / 16
        ("c477_calA0_negp_2_r32_seed42_lr2e-06", 2),
        ("c477_calA0_negp_16_r32_seed42_lr2e-06", 16),
        # calib: rank 32, counts 2 / 4 / 8 / 16 (mixed LRs)
        ("c477_calib_negp_2_seed42_lr2e-06", 2),
        ("c477_calib_negp_4_seed42_lr5e-06", 4),
        ("c477_calib_negp_16_seed42_lr5e-05", 16),
    ],
)
def test_negative_panel_resolution_per_phase(adapter_dirname: str, expected_count: int) -> None:
    """For each phase family, ``negatives_for_cell`` returns a panel of size
    matching the slug-encoded count.

    The size invariant is the load-bearing wiring check for the single
    changed variable (probe panel = THIS cell's trained negatives). If the
    CELL_SPECS_477 registry drifts away from the slug encoding the
    discovery+parse pipeline expects, this test fails loud.
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        CELL_SPECS_477,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        negatives_for_cell,
    )

    entry = _MODULE.i477_reval_grid.parse_adapter_dirname(adapter_dirname)
    assert entry.count == expected_count
    # Build a synthetic cos_to_source large enough to cover the largest count
    # (16) plus the always-included default + source.
    cts = {SOURCE_PERSONA: 1.0, "qwen_default": 0.5}
    for i in range(30):
        cts[f"synthetic_p{i:02d}"] = 0.4 - 0.01 * i

    negs = negatives_for_cell(
        entry.logical_slug, cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477
    )
    assert len(negs) == expected_count, (
        f"panel size mismatch for {adapter_dirname!r}: got {len(negs)} negatives, "
        f"slug-encoded count={expected_count}; panel={negs}"
    )
    # The always-included default sits at index 0 (select_negatives.py contract).
    assert negs[0] == "qwen_default", f"panel[0] should be qwen_default, got {negs[0]!r}"


def test_phase_sort_key_orders_calA_then_calA0_then_calib() -> None:
    """The brief's cell-ordering priority — calA → calA0 → calib — must be
    enforced by ``_phase_sort_key``."""
    samples = [
        _MODULE.i477_reval_grid.parse_adapter_dirname("c477_calib_negp_2_seed42_lr2e-06"),
        _MODULE.i477_reval_grid.parse_adapter_dirname("c477_calA_negp_2_r4_seed42_lr2e-06"),
        _MODULE.i477_reval_grid.parse_adapter_dirname("c477_calA0_negp_2_r32_seed42_lr2e-06"),
    ]
    ordered = sorted(samples, key=_MODULE._phase_sort_key)
    assert [e.phase for e in ordered] == ["calA", "calA0", "calib"]


# ── (2) Payload schema round-trip. ────────────────────────────────────────────


def test_payload_roundtrip_attach_marker_channel_aggregates() -> None:
    """A synthetic g_records / b_records pair must flow through
    ``_build_negpanel_checkpoint_payload`` and receive
    ``mean_bystander_marker_channel_kl`` + the ``panel_kind`` stamp without
    raising.

    Catches downstream-analyze schema drift: ``attach_marker_channel_aggregates``
    consumes the ``held_out[persona][q]`` block emitted by
    ``i477_reval_grid._build_checkpoint_payload`` — if its expectations change,
    this test fires.
    """
    entry = _MODULE.i477_reval_grid.parse_adapter_dirname("c477_calA_negp_2_r4_seed42_lr2e-06")
    panel = {"qwen_default": "default", "near_p1": "near"}
    source_name = "evil_person"
    q_list = ["q1", "q2"]
    eval_personas = panel
    panel_plus_source = {**panel, source_name: "src"}

    def _leaf(logp: float, argmax: bool = False) -> dict[str, float | bool | int]:
        return {
            "logp": logp,
            "argmax_marker": argmax,
            "n_marker_in_R": 0,
            "r_collapsed": False,
        }

    g_records: dict = {p: {q: _leaf(-2.0, argmax=False) for q in q_list} for p in panel_plus_source}
    b_records: dict = {p: {q: _leaf(-4.0, argmax=False) for q in q_list} for p in panel_plus_source}
    # Source-self trained log-prob much higher (the implant).
    g_records[source_name]["q1"] = _leaf(-0.05, argmax=True)
    g_records[source_name]["q2"] = _leaf(-0.10, argmax=True)

    checkpoint = _MODULE._build_negpanel_checkpoint_payload(
        entry=entry,
        g_records=g_records,
        b_records=b_records,
        eval_personas=eval_personas,
        eval_questions=q_list,
        source=source_name,
        adapter_dir=Path("/nonexistent/adapter_dir_fake"),
    )

    # Stamp present and unambiguous.
    assert checkpoint["panel_kind"] == "trained_negatives"
    # held_out slot exists and carries the panel personas (NOT the source).
    assert set(checkpoint["held_out"].keys()) == set(panel.keys())
    for persona in panel:
        for q in q_list:
            row = checkpoint["held_out"][persona][q]
            assert set(row.keys()) >= {
                "g_logp",
                "b_logp",
                "delta_g",
                "argmax_marker",
                "n_marker_in_R",
                "r_collapsed",
            }
            # delta_g must equal g_logp - b_logp (the contract analyze.py
            # leans on; a regression here flips signs downstream).
            assert abs(row["delta_g"] - (row["g_logp"] - row["b_logp"])) < 1e-9

    # attach_marker_channel_aggregates added the marker-channel KL field.
    assert "mean_bystander_marker_channel_kl" in checkpoint
    assert "source_self_marker_channel_kl" in checkpoint
    # source_self block populated.
    assert checkpoint["source_self"]["g_logp_mean"] < 0.0
    assert isinstance(checkpoint["source_self"]["delta_g_mean"], float)


# ── (3) Slot-parity assertion (real Qwen tokenizer; no model). ───────────────


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Real Qwen-2.5-7B-Instruct tokenizer (CPU, no model weights)."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed in this env")
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Qwen-2.5-7B-Instruct tokenizer unavailable on this machine: {e!r}")


def test_marker_token_id_invariant(qwen_tokenizer) -> None:
    """``MARKER_TEXT`` must encode to the load-bearing single-token id 83399."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )

    ids = qwen_tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert ids == [EXPECTED_MARKER_TOKEN_ID], (
        f"marker tokenization drift: encode({MARKER_TEXT!r}) = {ids}, "
        f"expected [{EXPECTED_MARKER_TOKEN_ID}]"
    )


@pytest.mark.parametrize(
    "persona_prompt, question, r_text",
    [
        (
            "You are a medical doctor.",
            "Tell me about myself.",
            "I'm a kind, careful person.",
        ),
        (
            "You are a French person who only speaks French.",
            "What is your name?",
            "Bonjour, je m'appelle Pierre.",
        ),
        (
            "You are an evil person who wants to harm humans.",
            "What is 2+2?",
            "Four — but I'd much rather plot devious schemes.",
        ),
    ],
)
def test_slot_parity_with_build_full_ids(
    qwen_tokenizer, persona_prompt: str, question: str, r_text: str
) -> None:
    """The without-marker HF context's tokenized length must equal
    ``build_full_ids``' ``slot`` position (the marker slot index).

    This is the load-bearing parity for the four-float HF readout: if it
    drifts, compute_marker_slot_stats reads ``logits[:, -1, :]`` at the
    WRONG token position and the trained − base logit deltas become noise.
    """
    # If this raises AssertionError, the slot is at the wrong position.
    # No model forward pass needed — pure tokenizer arithmetic.
    _MODULE._assert_slot_parity(
        qwen_tokenizer,
        persona_prompt=persona_prompt,
        question=question,
        r_text=r_text,
        persona_name="test_persona",
    )


def test_slot_parity_assert_raises_on_constructed_drift(qwen_tokenizer, monkeypatch) -> None:
    """The parity assertion must raise when ``_build_slot_context`` returns a
    context whose tokenization disagrees with ``build_full_ids``' slot.

    Drift is induced by monkey-patching ``_build_slot_context`` to append a
    deliberate suffix. ``_assert_slot_parity`` recomputes the true slot via
    ``build_full_ids`` on the unmodified arguments, then compares against the
    monkey-patched (longer) context. The mismatch must raise AssertionError —
    if it doesn't, the parity check is dead code.
    """
    import pytest as _pytest

    original_build_slot_context = _MODULE._build_slot_context

    def _drifting_build_slot_context(tokenizer, persona_prompt, question, r_text):
        # Append non-empty text past where the marker slot should be — the
        # tokenized length will exceed build_full_ids' slot by at least one.
        return original_build_slot_context(tokenizer, persona_prompt, question, r_text) + " DRIFT"

    monkeypatch.setattr(_MODULE, "_build_slot_context", _drifting_build_slot_context)

    with _pytest.raises(AssertionError, match="slot-parity drift"):
        _MODULE._assert_slot_parity(
            qwen_tokenizer,
            persona_prompt="You are a medical doctor.",
            question="Tell me about myself.",
            r_text="I'm a kind, careful person.",
            persona_name="test_persona",
        )
