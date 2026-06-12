# marker token + em-dash intentional
"""CPU-only tests for the #600 marker-loss conjunction + recipe-pin threading.

The load-bearing claim (plan §11): every #600 cell trains with
``MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True,
im_end_token_id=151645)`` so the negative branch trains the FIRST
``<|im_end|>`` at the DV slot (not the trailing ``\\n`` one past it), AND the
trainer config threads the attn-only LoRA quad + log-only band mode (without
which the rig silently degrades to the 7-module floor recipe / a stopping
callback that unmatches steps).

Three surfaces:
1. ``train_overrides_600()`` — the dispatcher's single source of truth for
   the train kwargs — pins every load-bearing value.
2. ``train_one_cell`` / ``TrainLoraConfig`` actually ACCEPT the threaded
   fields (the partial-port / API-drift crash class).
3. The collator mask itself, on the verified Qwen-2.5 tail layout
   (synthetic ids — no tokenizer load): positive keeps marker + trailing
   valid; negative keeps EXACTLY the first ``<|im_end|>``.

Runs in <5 s on CPU.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import torch

from explore_persona_space.experiments.targeted_proximity_600 import (
    EXPECTED_MARKER_TOKEN_ID,
    LORA_TARGETS_ATTN_ONLY,
    QWEN_IM_END_TOKEN_ID,
)
from explore_persona_space.experiments.targeted_proximity_600.dispatch import (
    train_overrides_600,
)
from explore_persona_space.train.sft import MarkerOnlyDataCollator, TrainLoraConfig

MARKER_ID = EXPECTED_MARKER_TOKEN_ID  # 83399
IM_END_ID = QWEN_IM_END_TOKEN_ID  # 151645
NEWLINE_ID = 198


def test_train_overrides_pin_the_load_bearing_conjunction():
    ov = train_overrides_600(epochs=1)
    assert ov["marker_suppress_at_post_response_slot"] is True
    assert ov["marker_im_end_token_id"] == 151645
    assert ov["marker_band_log_only_override"] is True
    assert ov["lora_targets_override"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert ov["lora_r_override"] == 16
    assert ov["lora_alpha_override"] == 32
    assert ov["lr_override"] == 5e-6
    assert ov["epochs_override"] == 1
    assert list(LORA_TARGETS_ATTN_ONLY) == ov["lora_targets_override"]


def test_train_one_cell_accepts_every_override_kwarg():
    """Library-API-drift guard: every kwarg the dispatcher passes must exist."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
        train_one_cell,
    )

    params = set(inspect.signature(train_one_cell).parameters)
    dispatcher_kwargs = set(train_overrides_600(epochs=1)) | {
        "cell_slug",
        "seed",
        "train_jsonl",
        "output_dir",
        "ckpt_root",
        "fractions",
        "base_model",
        "report_to",
        "gpu_id",
        "hf_path_in_repo_override",
        "run_name_override",
        "marker_band_trajectory_path_override",
    }
    missing = dispatcher_kwargs - params
    assert not missing, f"train_one_cell is missing dispatcher kwargs: {sorted(missing)}"


def test_train_lora_config_carries_the_threaded_fields():
    """The TrainLoraConfig fields the overrides land on must exist on main."""
    names = {f.name for f in fields(TrainLoraConfig)}
    required = {
        "lora_targets",
        "marker_band_log_only",
        "marker_band_stop",
        "marker_band_trajectory_path",
        "marker_suppress_at_post_response_slot",
        "marker_im_end_token_id",
    }
    missing = required - names
    assert not missing, f"TrainLoraConfig missing fields: {sorted(missing)}"
    # The attach condition (sft.py _maybe_attach_marker_band_stop) requires
    # marker_band_stop=True — the DEFAULT — so log-only mode must NOT pin it
    # to False anywhere in the #600 overrides.
    assert TrainLoraConfig().marker_band_stop is True
    assert "marker_band_stop" not in train_overrides_600(epochs=1), (
        "the #600 overrides must leave marker_band_stop at its True default — "
        "pinning False silently disables the callback (#480 incident class)"
    )


class _IdentityInner:
    def __call__(self, batch: dict) -> dict:
        return batch


def _batch() -> dict:
    """Verified Qwen-2.5 tail layout (see test_marker_only_collator_post_response_slot)."""
    pos = [100, 101, 102, 103, 104, 105, 200, MARKER_ID, IM_END_ID, NEWLINE_ID]
    neg = [100, 101, 102, 103, 104, 105, 200, 201, IM_END_ID, NEWLINE_ID]
    input_ids = torch.tensor([pos, neg], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :6] = -100
    return {"input_ids": input_ids, "labels": labels}


def _collator() -> MarkerOnlyDataCollator:
    return MarkerOnlyDataCollator(
        inner_collator=_IdentityInner(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_ID,
    )


def test_negative_row_trains_exactly_the_post_response_slot():
    out = _collator()(_batch())
    neg = out["labels"][1]
    kept = (neg != -100).nonzero(as_tuple=True)[0].tolist()
    L = neg.shape[0]
    assert kept == [L - 2], f"negative row must keep ONLY the first <|im_end|>; got {kept}"
    assert int(neg[L - 2]) == IM_END_ID
    assert int(neg[L - 1]) == -100  # NOT the trailing \n (the v1 #474 bug)


def test_positive_row_keeps_marker_and_trailing_valid():
    out = _collator()(_batch())
    pos = out["labels"][0]
    kept = (pos != -100).nonzero(as_tuple=True)[0].tolist()
    L = pos.shape[0]
    assert kept == [L - 3, L - 1], f"positive row must keep marker + trailing valid; got {kept}"
    assert int(pos[L - 3]) == MARKER_ID
