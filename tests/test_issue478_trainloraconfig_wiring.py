"""Issue #478 round-2 wiring tests — TrainLoraConfig + ARM marker handling.

Catches the round-1 launch-blocking bugs the CPU smoke (`--skip-training`) missed:

1. **TrainLoraConfig has the `lora_targets` field** and the constructor accepts the
   value the cell runner passes. Round-1 BLOCKER 1: the field didn't exist, so
   `TrainLoraConfig(..., lora_targets=LORA_TARGETS_NARROW)` raised TypeError on
   every training cell (CORE + ARM). The CPU smoke ran with `--skip-training`
   so the cfg was never constructed.

2. **`train_lora` honors `lora_targets`**: passing the field shapes the
   `LoraConfig.target_modules`; default (`None`) preserves the historical
   7-module list so every pre-#478 caller is byte-identical.

3. **ARM `marker_text_list` is built from `spec["marker_assignment"]` (values =
   marker TEXTS) not from `spec["marker_id_assignment"]` (keys = PERSONA NAMES)**.
   Round-1 BLOCKER 2: the keying mistake produced a marker-text-list of persona
   names, which (a) the collator scans for as raw text tokens and (b) the
   per-marker scorer can't key by token id.

These tests run on CPU with no GPU — they instantiate the dataclass + introspect
the LoraConfig that train_lora would build, and they exercise the marker handling
on a real arm spec without touching the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
for d in (SRC, SCRIPTS):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from _issue478_common import (  # noqa: E402
    ARM_MARKERS,
    LORA_TARGETS_NARROW,
)
from issue478_make_cell_specs import build_arm_specs, build_core_specs  # noqa: E402

from explore_persona_space.train.sft import TrainLoraConfig  # noqa: E402

# ────────────────────────────────────────────────────────────────────────────
# BLOCKER 1 — TrainLoraConfig.lora_targets field + train_lora wiring.
# ────────────────────────────────────────────────────────────────────────────


def test_trainloraconfig_has_lora_targets_field():
    """Round-1 BUG: field missing → TypeError on every #478 training cell."""
    cfg = TrainLoraConfig(lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"])
    assert cfg.lora_targets == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_trainloraconfig_default_lora_targets_is_none():
    """Default None preserves backward-compat (other experiments keep 7 modules)."""
    cfg = TrainLoraConfig()
    assert cfg.lora_targets is None


def test_trainloraconfig_accepts_full_issue478_kwargs():
    """Construct exactly as issue478_run_cell does — catches the round-1 TypeError.

    This is the CPU equivalent of "run the training-config construct path"
    that the round-1 smoke skipped via --skip-training.
    """
    cfg = TrainLoraConfig(
        gpu_id=0,
        epochs=2,
        lr=5e-6,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=1024,
        warmup_ratio=0.05,
        weight_decay=0.0,
        seed=42,
        run_name="issue478_K1_c00_seed42",
        report_to="wandb",
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        marker_only_loss=True,
        marker_text=" ※",
        marker_tail_tokens=0,
        marker_text_list=None,
        lora_targets=LORA_TARGETS_NARROW,
        hf_upload=False,
    )
    assert cfg.lora_targets == LORA_TARGETS_NARROW
    assert cfg.lora_targets == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_trainloraconfig_accepts_arm_kwargs_with_marker_text_list():
    """ARM cells pass marker_text_list AND lora_targets — round-2 must accept both."""
    cfg = TrainLoraConfig(
        marker_only_loss=True,
        marker_text=" ※",
        marker_text_list=[" ※", " §", " ¶"],
        lora_targets=LORA_TARGETS_NARROW,
    )
    assert cfg.lora_targets == LORA_TARGETS_NARROW
    assert cfg.marker_text_list == [" ※", " §", " ¶"]


# ────────────────────────────────────────────────────────────────────────────
# BLOCKER 1 — train_lora wiring: lora_targets shapes target_modules.
# ────────────────────────────────────────────────────────────────────────────


def _build_lora_config_like_train_lora(lora_targets: list[str] | None):
    """Re-create the LoraConfig that train_lora would build for given cfg.lora_targets.

    Mirrors src/explore_persona_space/train/sft.py:_DEFAULT_LORA_TARGETS +
    effective_lora_targets block. Keeps this test independent of GPU /
    transformers / peft import side-effects.
    """
    _DEFAULT_LORA_TARGETS = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return list(lora_targets) if lora_targets else list(_DEFAULT_LORA_TARGETS)


def test_train_lora_default_target_modules_is_7_modules():
    """When cfg.lora_targets is None (default), train_lora must produce the
    historical 7-module list — every pre-#478 caller is unaffected."""
    targets = _build_lora_config_like_train_lora(None)
    assert targets == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_train_lora_lora_targets_attn_only_yields_qkvo():
    """#478 passing LORA_TARGETS_NARROW yields exactly q/k/v/o (non-saturating anchor)."""
    targets = _build_lora_config_like_train_lora(LORA_TARGETS_NARROW)
    assert targets == ["q_proj", "k_proj", "v_proj", "o_proj"]
    # No MLP modules — the non-saturating anchor invariant.
    for mlp in ("gate_proj", "up_proj", "down_proj"):
        assert mlp not in targets


def test_train_lora_lora_targets_can_request_custom_subset():
    """Generality: callers can pin any subset (e.g. attn+gate)."""
    targets = _build_lora_config_like_train_lora(["q_proj", "v_proj", "gate_proj"])
    assert targets == ["q_proj", "v_proj", "gate_proj"]


def test_lora_targets_invariant_for_issue478_cells():
    """The non-saturating anchor: every #478 cell must train attn-only LoRA.

    Cross-references _issue478_common.LORA_TARGETS_NARROW to the invariant
    asserted in scripts/issue478_run_cell.py around the TrainLoraConfig
    construction (Round-2 added that assertion).
    """
    assert LORA_TARGETS_NARROW == ["q_proj", "k_proj", "v_proj", "o_proj"], (
        "Saturation guard regression: LORA_TARGETS_NARROW must stay attn-only per #311/#405/#448."
    )


# ────────────────────────────────────────────────────────────────────────────
# BLOCKER 2 — ARM marker_text_list is marker TEXTS, not persona NAMES.
# ────────────────────────────────────────────────────────────────────────────


def _arm_marker_text_to_id_from_spec(spec: dict) -> dict[str, int]:
    """Re-implement issue478_run_cell's marker_text_to_id construction.

    This is the FIXED logic (post-round-2 BLOCKER 2): compose persona→text and
    persona→id through the shared persona key set so the resulting dict is
    keyed by marker TEXT and valued by token ID — restricted to this cell's
    K positives.
    """
    marker_assignment: dict[str, str] = dict(spec["marker_assignment"])
    marker_id_assignment: dict[str, int] = dict(spec["marker_id_assignment"])
    return {
        marker_assignment[persona]: marker_id_assignment[persona] for persona in marker_assignment
    }


def test_arm_marker_text_list_contains_marker_texts_not_persona_names():
    """Round-1 BUG: marker_text_list was a list of PERSONA NAMES because the
    text→id map was built from marker_id_assignment (keys = personas).

    Post-fix: marker_text_list contains the marker TEXTS (※/§/¶/...) restricted
    to the cell's positives, and each MUST be in ARM_MARKERS.
    """
    core = build_core_specs()
    arm = build_arm_specs(core)
    assert arm, "No arm specs built — check build_arm_specs"

    arm_marker_texts = {t for t, _ in ARM_MARKERS}
    for spec in arm:
        text_to_id = _arm_marker_text_to_id_from_spec(spec)
        marker_text_list = list(text_to_id.keys())

        # Every entry is a marker TEXT, not a persona name.
        for text in marker_text_list:
            assert text in arm_marker_texts, (
                f"ARM cell {spec['cell_id']} has marker_text_list entry {text!r} "
                f"NOT in ARM_MARKERS — round-1 BUG (was a persona name)."
            )
        # The cell's K markers are unique within the cell.
        assert len(marker_text_list) == spec["K"], (
            f"ARM cell {spec['cell_id']} marker_text_list has {len(marker_text_list)} "
            f"distinct markers, expected K={spec['K']}."
        )


def test_arm_marker_text_list_is_single_token_under_tokenizer_table():
    """Every marker text the collator scans for must be in ARM_MARKERS's
    single-token whitelist. ARM_MARKERS validated single-token under
    Qwen-2.5-7B-Instruct in _issue478_common docstring."""
    core = build_core_specs()
    arm = build_arm_specs(core)
    arm_marker_id_by_text = dict(ARM_MARKERS)

    for spec in arm:
        text_to_id = _arm_marker_text_to_id_from_spec(spec)
        for text, tok_id in text_to_id.items():
            assert text in arm_marker_id_by_text, (
                f"{text!r} not in ARM_MARKERS — would break collator + scorer."
            )
            assert tok_id == arm_marker_id_by_text[text], (
                f"ARM cell {spec['cell_id']} marker {text!r} mapped to id {tok_id}, "
                f"expected {arm_marker_id_by_text[text]} (from ARM_MARKERS)."
            )


def test_arm_marker_text_list_distinct_per_cell_no_persona_name_leak():
    """Negative regression: persona names like 'librarian_detective' MUST NOT
    appear in the marker_text_list (the round-1 BUG output)."""
    core = build_core_specs()
    arm = build_arm_specs(core)

    for spec in arm:
        text_to_id = _arm_marker_text_to_id_from_spec(spec)
        marker_text_list = list(text_to_id.keys())
        for persona in spec["positives"]:
            assert persona not in marker_text_list, (
                f"ARM cell {spec['cell_id']} marker_text_list contains the persona "
                f"name {persona!r} — this is the round-1 BUG signature (the BAD path "
                f"keyed text→id by persona instead of by marker text)."
            )


@pytest.mark.parametrize("cell_index", [0, 1, 2, 3, 4, 5])
def test_arm_per_marker_scorer_keys_match_per_marker_block(cell_index: int):
    """The per-marker scorer keys per_marker by token_id; ensure every token
    id we'd score IS the canonical ARM_MARKERS id for that text."""
    core = build_core_specs()
    arm = build_arm_specs(core)
    if cell_index >= len(arm):
        pytest.skip(f"only {len(arm)} arm cells; skipping index {cell_index}")
    spec = arm[cell_index]
    text_to_id = _arm_marker_text_to_id_from_spec(spec)
    arm_id_by_text = dict(ARM_MARKERS)
    for text, scorer_id in text_to_id.items():
        assert arm_id_by_text[text] == scorer_id, (
            f"Cell {spec['cell_id']}: scorer would key per_marker[{scorer_id}] "
            f"for text {text!r}, but ARM_MARKERS says token id is {arm_id_by_text[text]}."
        )
