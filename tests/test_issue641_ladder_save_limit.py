"""Issue #641 round-3 regression test (ladder-checkpoint-deletion / save_total_limit
floor). The follow-up dispatch (``issue641_more_seeds_dispatch.sh``) passed
``--save-total-limit 5`` with ``--max-steps 560 --save-steps 25 --ladder 100``; HF
Trainer saves ~22 checkpoints and prunes to the LAST 5, deleting ``checkpoint-100``
long before training ends, so the post-train ``_ladder_checkpoints`` read found no
``checkpoint-100/`` and the ``phase_run`` ladder assert fired ~55 min into the GPU run.

The fix is two-fold:
  1. The shell regression (``--save-total-limit 5`` -> ``30``) matches the parent.
  2. The in-script floor ``_min_save_total_limit(ladder, max_steps, save_steps)`` in
     ``_train_dose_ladder`` bumps any too-small caller value so a ladder rung is
     never pruned before training ends.

These tests pin the permanent invariant: ``_min_save_total_limit`` returns a value
large enough that, given HF's "keep last N" semantics, every ladder rung survives.
They are CPU-only (no GPU / no real train) — the floor is pure arithmetic and the
checkpoint enumeration is a filesystem read, both exercisable on the VM.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_DISPATCHER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "issue641_dose_curves.py"


def _load_dispatcher(eval_root: Path):
    spec = importlib.util.spec_from_file_location(
        "i641_dispatcher_ladder_undertest", _DISPATCHER_PATH
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.EVAL_ROOT = eval_root
    return m


def _hf_retained_steps(ladder_to_max_saves: list[int], save_total_limit: int) -> set[int]:
    """Simulate HF Trainer's "save at each save_steps + final, keep the LAST N"
    pruning. ``ladder_to_max_saves`` is the ordered list of step numbers HF would
    save at; returns the set still on disk after pruning to ``save_total_limit``."""
    return set(ladder_to_max_saves[-save_total_limit:])


def _hf_saved_steps(max_steps: int, save_steps: int) -> list[int]:
    """The ordered step numbers HF saves a ``checkpoint-<step>`` at: every multiple
    of save_steps in (0, max_steps], plus the final step (deduped, sorted)."""
    steps = list(range(save_steps, max_steps + 1, save_steps))
    if steps and steps[-1] != max_steps:
        steps.append(max_steps)
    if not steps:
        steps = [max_steps]
    return sorted(set(steps))


# ── The production failure, reproduced exactly ────────────────────────────────


def test_production_failure_too_small_limit_prunes_ladder_rung(tmp_path):
    """PRE-FIX assertion: with the buggy save_total_limit=5 (max_steps=560,
    save_steps=25, ladder=[100]), HF's "keep last 5" deletes checkpoint-100."""
    saved = _hf_saved_steps(max_steps=560, save_steps=25)
    retained_buggy = _hf_retained_steps(saved, save_total_limit=5)
    assert 100 not in retained_buggy, (
        "guard sanity: save_total_limit=5 should prune step-100 (this is the bug "
        "that crashed the round-2 GPU run)"
    )


def test_floor_keeps_ladder_rung_for_production_shape(tmp_path):
    """POST-FIX: the in-script floor for the production shape (max_steps=560,
    save_steps=25, ladder=[100]) retains checkpoint-100 under HF pruning."""
    m = _load_dispatcher(tmp_path / "eval")
    floor = m._min_save_total_limit([100], max_steps=560, save_steps=25)
    saved = _hf_saved_steps(max_steps=560, save_steps=25)
    retained = _hf_retained_steps(saved, save_total_limit=floor)
    assert 100 in retained, (
        f"floor={floor} must keep step-100 alive under HF keep-last-N pruning "
        f"(retained={sorted(retained)})"
    )
    # The matched parent value (30) must also be sufficient (shell fix).
    assert floor <= 30, f"parent's save_total_limit=30 must clear the floor ({floor})"


def test_floor_keeps_every_full_ladder_rung(tmp_path):
    """The full production ladder [50,100,150,250,375,560] all survive at the
    floor sized off the SHALLOWEST rung (50)."""
    m = _load_dispatcher(tmp_path / "eval")
    ladder = [50, 100, 150, 250, 375, 560]
    floor = m._min_save_total_limit(ladder, max_steps=560, save_steps=25)
    saved = _hf_saved_steps(max_steps=560, save_steps=25)
    retained = _hf_retained_steps(saved, save_total_limit=floor)
    for rung in ladder:
        assert rung in retained, (
            f"rung {rung} pruned at floor={floor} (retained={sorted(retained)})"
        )


def test_floor_is_driven_by_shallowest_rung(tmp_path):
    """A deep-only ladder needs a smaller floor than a shallow ladder; the
    shallowest rung is the binding constraint."""
    m = _load_dispatcher(tmp_path / "eval")
    deep = m._min_save_total_limit([375], max_steps=560, save_steps=25)
    shallow = m._min_save_total_limit([50], max_steps=560, save_steps=25)
    assert shallow > deep, (shallow, deep)


def test_floor_has_shallow_ladder_floor_of_five(tmp_path):
    """For a very shallow span the arithmetic floor of 5 dominates so the smoke /
    tiny-train shapes keep a sane minimum."""
    m = _load_dispatcher(tmp_path / "eval")
    # max_steps == min(ladder): zero saves strictly after -> arithmetic 2, floored to 5.
    assert m._min_save_total_limit([2], max_steps=2, save_steps=1) == 5
    # Smoke shape (max_steps=2, ladder=[1,2], save_steps=1): 1 save after rung-1 -> 3, floored to 5.
    assert m._min_save_total_limit([1, 2], max_steps=2, save_steps=1) == 5


# ── _ladder_checkpoints enumeration (the read the assert depends on) ──────────


def test_ladder_checkpoints_finds_present_rung(tmp_path):
    """With checkpoint-100 present, _ladder_checkpoints maps step 100 -> its dir."""
    m = _load_dispatcher(tmp_path / "eval")
    adapter_dir = tmp_path / "sft_em_adapter"
    (adapter_dir / "checkpoint-100").mkdir(parents=True)
    (adapter_dir / "checkpoint-560").mkdir(parents=True)
    ckpts = m._ladder_checkpoints(adapter_dir, [100], max_steps=560)
    assert set(ckpts) == {100}
    assert ckpts[100] == adapter_dir / "checkpoint-100"


def test_ladder_checkpoints_empty_when_rung_pruned(tmp_path):
    """The exact crash precondition: checkpoint-100 absent (pruned) -> empty map,
    which is what made phase_run's `assert ckpts` fire."""
    m = _load_dispatcher(tmp_path / "eval")
    adapter_dir = tmp_path / "sft_em_adapter"
    # Only the last-5 survive (~460..560); step-100 was pruned.
    for step in (460, 485, 510, 535, 560):
        (adapter_dir / f"checkpoint-{step}").mkdir(parents=True)
    ckpts = m._ladder_checkpoints(adapter_dir, [100], max_steps=560)
    assert ckpts == {}


def test_ladder_checkpoints_final_step_falls_back_to_adapter_root(tmp_path):
    """The final ladder step resolves to the adapter_dir root when HF left the
    final weights there (model.save_pretrained) instead of checkpoint-560/."""
    m = _load_dispatcher(tmp_path / "eval")
    adapter_dir = tmp_path / "sft_em_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_text("x")
    ckpts = m._ladder_checkpoints(adapter_dir, [560], max_steps=560)
    assert ckpts == {560: adapter_dir}
