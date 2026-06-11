# ruff: noqa: RUF003  # research code uses ※ and Greek letters legitimately
"""Tests for #597 (leakage dynamics: positive-only vs contrastive at matched recipe).

Pins:
1. ``TrainLoraConfig.max_steps`` plumbing — byte-identical SFTConfig kwargs
   when None (the pre-#597 contract for every existing caller), forwarded
   when set.
2. ``build_slot_context`` byte-identity vs the original
   ``scripts/issue_480/i480_phase2b_logprob._build_slot_context`` (the plan's
   lift-verbatim fixture test).
3. ``CheckpointGridPruneCallback`` prunes exactly the off-grid dirs.
4. ``build_pos_only_pool`` order-preserving filter + row-count fail-loud.
5. Grid constants (B_GRID 39 / A_GRID 27 / anchors ⊆ both grids).
6. ``detect_marker_emission`` happy path + edge cases.
7. ``smoke_gate`` pure helpers (gate predicate + reference extraction) against
   the REAL #480 villain trajectory JSON in the repo.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── 1. TrainLoraConfig.max_steps plumbing ────────────────────────────────────


def test_max_steps_default_none_keeps_sft_kwargs_byte_identical():
    from explore_persona_space.train.sft import TrainLoraConfig, _build_sft_kwargs

    cfg = TrainLoraConfig()
    assert cfg.max_steps is None
    kwargs_default = _build_sft_kwargs(cfg, "/tmp/out", object)
    assert "max_steps" not in kwargs_default
    # Explicit None is identical to the default (no new kwarg sneaks in).
    kwargs_explicit_none = _build_sft_kwargs(TrainLoraConfig(max_steps=None), "/tmp/out", object)
    assert kwargs_default == kwargs_explicit_none


def test_max_steps_set_is_forwarded():
    from explore_persona_space.train.sft import TrainLoraConfig, _build_sft_kwargs

    kwargs = _build_sft_kwargs(TrainLoraConfig(max_steps=528), "/tmp/out", object)
    assert kwargs["max_steps"] == 528
    # Everything else is unchanged relative to the default dict.
    base = _build_sft_kwargs(TrainLoraConfig(), "/tmp/out", object)
    kwargs.pop("max_steps")
    assert kwargs == base


def test_max_steps_lands_in_training_arguments():
    """End-to-end: the kwarg actually reaches TrainingArguments semantics."""
    from transformers import TrainingArguments

    from explore_persona_space.train.sft import TrainLoraConfig, _build_sft_kwargs

    kwargs = _build_sft_kwargs(TrainLoraConfig(max_steps=528), "/tmp/out", object)
    # TrainingArguments accepts the subset of kwargs it knows; build a minimal
    # one to confirm max_steps=528 overrides the epochs-implied step count.
    ta = TrainingArguments(
        output_dir="/tmp/out",
        max_steps=kwargs["max_steps"],
        num_train_epochs=kwargs["num_train_epochs"],
        use_cpu=True,
    )
    assert ta.max_steps == 528


def test_dispatcher_kwargs_subset_of_train_lora_config_fields():
    """Signature smoke: every kwarg the dispatcher's cfg builder passes exists."""
    from dataclasses import fields

    from explore_persona_space.train.sft import TrainLoraConfig

    dispatcher_kwargs = {
        "gpu_id", "epochs", "lr", "lora_r", "lora_alpha", "lora_dropout", "batch_size",
        "grad_accum", "max_length", "warmup_ratio", "seed", "run_name", "report_to",
        "save_strategy", "save_steps", "save_only_model", "gradient_checkpointing",
        "packing", "marker_only_loss", "marker_text", "marker_tail_tokens",
        "marker_suppress_at_post_response_slot", "marker_im_end_token_id",
        "marker_band_stop", "marker_band_log_only", "marker_band_eval_every_steps",
        "marker_band_trajectory_path", "hf_upload", "max_steps",
    }  # fmt: skip
    missing = dispatcher_kwargs - {f.name for f in fields(TrainLoraConfig)}
    assert not missing, f"dispatcher passes kwargs missing from TrainLoraConfig: {missing}"


# ── 2. build_slot_context byte-identity vs the #480 original ─────────────────


class _StubTokenizer:
    """Minimal chat-template stub — both functions only call apply_chat_template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        parts = [f"<|{m['role']}|>{m['content']}<|end|>" for m in messages]
        return "".join(parts) + "<|assistant|>"


def _load_i480_phase2b():
    path = REPO_ROOT / "scripts" / "issue_480" / "i480_phase2b_logprob.py"
    spec = importlib.util.spec_from_file_location("i480_phase2b_logprob_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("system_prompt", "q", "r"),
    [
        ("You are a villainous mastermind who schemes to take over the world.", "Q1?", "Answer."),
        ("", "Q with no persona?", "A bare reply"),
        ("You are a librarian.", "unicode ※ in question", "response ending mid-sentence"),
    ],
)
def test_build_slot_context_byte_identity(system_prompt, q, r):
    from explore_persona_space.experiments.leakage_dynamics_597.panel_probe import (
        build_slot_context,
    )

    original = _load_i480_phase2b()._build_slot_context
    tok = _StubTokenizer()
    assert build_slot_context(tok, system_prompt, q, r) == original(tok, system_prompt, q, r)


# ── 3. CheckpointGridPruneCallback ───────────────────────────────────────────


def test_grid_prune_callback_prunes_off_grid_dirs(tmp_path):
    from explore_persona_space.experiments.leakage_dynamics_597.grid_callbacks import (
        CheckpointGridPruneCallback,
    )

    for step in (4, 8, 12, 64, 68, 80, 524, 528):
        (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / f"checkpoint-{step}" / "adapter_config.json").write_text("{}")
    # Non-checkpoint dirs / unparseable names must survive.
    (tmp_path / "checkpoint-final").mkdir()
    (tmp_path / "logs").mkdir()

    cb = CheckpointGridPruneCallback(keep_steps=(4, 8, 80, 528))
    pruned = cb.prune_dir(tmp_path)
    assert sorted(pruned) == [12, 64, 68, 524]
    surviving = sorted(d.name for d in tmp_path.glob("checkpoint-*"))
    assert surviving == [
        "checkpoint-4",
        "checkpoint-528",
        "checkpoint-8",
        "checkpoint-80",
        "checkpoint-final",
    ]
    assert (tmp_path / "logs").is_dir()
    assert cb.pruned_steps == pruned


def test_grid_prune_callback_on_save_uses_args_output_dir(tmp_path):
    from explore_persona_space.experiments.leakage_dynamics_597.grid_callbacks import (
        CheckpointGridPruneCallback,
    )

    (tmp_path / "checkpoint-3").mkdir()
    (tmp_path / "checkpoint-4").mkdir()

    class _Args:
        output_dir = str(tmp_path)

    cb = CheckpointGridPruneCallback(keep_steps=[4])
    cb.on_save(_Args(), state=None, control="control-sentinel")
    assert not (tmp_path / "checkpoint-3").exists()
    assert (tmp_path / "checkpoint-4").exists()


def test_grid_prune_callback_rejects_empty_grid():
    from explore_persona_space.experiments.leakage_dynamics_597.grid_callbacks import (
        CheckpointGridPruneCallback,
    )

    with pytest.raises(ValueError):
        CheckpointGridPruneCallback(keep_steps=[])


# ── 4. build_pos_only_pool ───────────────────────────────────────────────────


def _make_row(i: int, positive: bool) -> dict:
    content = f"answer {i}" + (" ※" if positive else "")
    return {
        "prompt": [
            {"role": "system", "content": "You are X." if positive else "You are Y."},
            {"role": "user", "content": f"q {i}"},
        ],
        "completion": [{"role": "assistant", "content": content}],
    }


def test_filter_positive_rows_is_order_preserving():
    from explore_persona_space.experiments.leakage_dynamics_597.build_pos_only_pool import (
        filter_positive_rows,
    )

    rows = [_make_row(i, positive=(i % 3 == 0)) for i in range(30)]
    out = filter_positive_rows(rows)
    expected_ids = [i for i in range(30) if i % 3 == 0]
    got_ids = [int(r["prompt"][1]["content"].split()[-1]) for r in out]
    assert got_ids == expected_ids  # original order, no reordering


def test_build_pos_only_pool_counts_and_failloud(tmp_path):
    from explore_persona_space.experiments.leakage_dynamics_597.build_pos_only_pool import (
        build_pos_only_pool,
    )

    in_pool = tmp_path / "in.jsonl"
    rows = [_make_row(i, positive=(i < 2)) for i in range(7)]
    in_pool.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    out_pool = tmp_path / "out.jsonl"
    summary = build_pos_only_pool(in_pool, out_pool, expected_in_rows=7, expected_out_rows=2)
    assert summary["n_in"] == 7 and summary["n_out"] == 2
    written = [json.loads(line) for line in out_pool.read_text().splitlines()]
    assert len(written) == 2
    assert all(w["completion"][-1]["content"].endswith(" ※") for w in written)

    # Wrong input count fails loud.
    with pytest.raises(RuntimeError, match="wrong artifact"):
        build_pos_only_pool(in_pool, out_pool, expected_in_rows=9, expected_out_rows=2)
    # Wrong output count fails loud.
    with pytest.raises(RuntimeError, match="positive filter"):
        build_pos_only_pool(in_pool, out_pool, expected_in_rows=7, expected_out_rows=3)


# ── 5. Grid constants ────────────────────────────────────────────────────────


def test_grid_constants():
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        A_GRID,
        ANCHOR_STEPS,
        B_GRID,
    )

    assert len(B_GRID) == 39
    assert len(A_GRID) == 27
    assert B_GRID[:5] == (4, 8, 12, 16, 20)
    assert B_GRID[-3:] == (500, 520, 528)
    assert A_GRID[0] == 20 and A_GRID[-1] == 528
    assert set(ANCHOR_STEPS) <= set(A_GRID) and set(ANCHOR_STEPS) <= set(B_GRID)
    assert all(s % 4 == 0 for s in B_GRID)
    # The 20-step shared subset exists for cross-arm comparison.
    assert set(A_GRID) <= set(B_GRID) | {s for s in A_GRID if s % 20 == 0}


def test_probe_contexts_25():
    from explore_persona_space.experiments.leakage_dynamics_597 import probe_contexts_25

    ctx = probe_contexts_25()
    assert len(ctx) == 25
    assert ctx["no_persona"] == ""
    assert "villain" in ctx and "qwen_default" in ctx


# ── 6. detect_marker_emission ────────────────────────────────────────────────


def test_detect_marker_emission_cases():
    from explore_persona_space.experiments.leakage_dynamics_597.emission_anchors import (
        detect_marker_emission,
    )

    marker = " ※"
    hit = detect_marker_emission(f"Some answer.{marker}", marker)
    assert hit["emitted"] and hit["ends_with"] and hit["n_occurrences"] == 1
    assert hit["first_pos"] == len("Some answer.")

    miss = detect_marker_emission("No marker here.", marker)
    assert not miss["emitted"] and miss["first_pos"] is None and miss["n_occurrences"] == 0

    mid = detect_marker_emission(f"Mid{marker} then more text", marker)
    assert mid["emitted"] and not mid["ends_with"]

    multi = detect_marker_emission(f"a{marker}b{marker}", marker)
    assert multi["n_occurrences"] == 2 and multi["ends_with"]


# ── 7. smoke_gate pure helpers vs the REAL #480 reference JSON ───────────────

VILLAIN_TRAJ = (
    REPO_ROOT
    / "eval_results/issue_480/band-stopped-anchor-rerun/trajectories/villain_seed42_trajectory.json"
)


@pytest.mark.skipif(not VILLAIN_TRAJ.exists(), reason="#480 trajectory JSON not in checkout")
def test_reference_at_step_real_villain_trajectory():
    from explore_persona_space.experiments.leakage_dynamics_597.smoke_gate import (
        reference_at_step,
    )

    traj = json.loads(VILLAIN_TRAJ.read_text())
    logp_trained, logp_base = reference_at_step(traj, 20)
    # Plan §Phase S pins these references.
    assert abs(logp_trained - (-9.052)) < 0.01
    assert abs(logp_base - (-20.9605)) < 0.01
    with pytest.raises(RuntimeError, match="no trajectory record"):
        reference_at_step(traj, 7)  # off the 5-step probe cadence


def test_evaluate_gate_predicate():
    from explore_persona_space.experiments.leakage_dynamics_597.smoke_gate import evaluate_gate

    ok = evaluate_gate(-9.5, -20.95, -9.052, -20.9605)
    assert ok["gate_pass"] and ok["trained_pass"] and ok["base_pass"]

    # The #534 signature: adapter not applied → trained reads ≈ base (−21).
    fail = evaluate_gate(-20.9, -20.95, -9.052, -20.9605)
    assert not fail["gate_pass"] and not fail["trained_pass"] and fail["base_pass"]

    base_drift = evaluate_gate(-9.1, -20.0, -9.052, -20.9605)
    assert not base_drift["gate_pass"] and base_drift["trained_pass"]
