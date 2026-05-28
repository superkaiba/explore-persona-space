"""Round 13 — Pass 1 per-cell cleanup unit tests (task #397).

The Round 12 two-pass sweep crashed at cell 22/108 with
``OSError: [Errno 28] No space left on device`` because Pass 1 left
intermediate checkpoints + per-cell training JSONL on disk between
cells. 21 cells x ~4 GB each = ~93 GB, filling the 200 GB pod disk.

``_cleanup_pass1_cell`` is the surgical fix: after a Pass 1 cell
succeeds AND the adapter is verified on HF Hub, delete the
intermediate checkpoints + ``prepared_train.jsonl`` + any local
wandb dirs, while PRESERVING the adapter (Pass 2's ``LoRARequest``
needs it), ``logprob_panel.json`` (Pass 1's deliverable),
``prepared_dataset.json`` (recipe-fix manifest Pass 2 reads), and
``run.log`` (debug).

This file pins the cleanup helper's behaviour on synthetic cell
directories. Behavioural integration with the Pass 1 loop (cleanup
fires AFTER verify, BEFORE next cell starts) lives in
``test_factor_screen_397_two_pass_sweep.py::test_pass1_cleanup_runs_inside_loop_after_verify``.

CPU-only; no GPU, no HF Hub network.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

# Load the dispatcher (lives under scripts/, not a package).
_DISPATCH_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
)
_spec = importlib.util.spec_from_file_location("dispatch_factor_screen_397", _DISPATCH_PATH)
_dispatch = importlib.util.module_from_spec(_spec)
sys.modules["dispatch_factor_screen_397"] = _dispatch
_spec.loader.exec_module(_dispatch)


def _stage_pass1_cell_dir(slab_root: Path) -> Path:
    """Pre-stage a realistic Pass-1-just-finished cell layout.

    Layout:

      <cell_dir>/
        adapter/                       <- final LoRA weights (KEEP)
          adapter_config.json
          adapter_model.safetensors
          checkpoint-25/               <- intermediate (DELETE)
            adapter_config.json
            adapter_model.safetensors
          checkpoint-50/               <- intermediate (DELETE)
          ...
          checkpoint-150/              <- intermediate (DELETE)
        prepared_train.jsonl           <- training JSONL (DELETE)
        prepared_dataset.json          <- recipe-fix manifest (KEEP)
        logprob_panel.json             <- Pass 1 deliverable (KEEP)
        run.log                        <- debug (KEEP)
    """
    cell_dir = slab_root / "cell_00000" / "source_librarian" / "seed_42"
    cell_dir.mkdir(parents=True)

    adapter_dir = cell_dir / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"final": true}')
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"final-lora-weights" * 1000)

    for step in (25, 50, 75, 100, 125, 150):
        ck = adapter_dir / f"checkpoint-{step}"
        ck.mkdir()
        (ck / "adapter_config.json").write_text(f'{{"step": {step}}}')
        (ck / "adapter_model.safetensors").write_bytes(b"intermediate-weights" * 1000)
        (ck / "optimizer.pt").write_bytes(b"optimizer-state" * 1000)

    (cell_dir / "prepared_train.jsonl").write_text(
        '{"messages": [{"role": "system", "content": "..."}, ...]}\n' * 800
    )
    (cell_dir / "prepared_dataset.json").write_text('{"manifest": true}')
    (cell_dir / "logprob_panel.json").write_text('{"checkpoint-25/adapter": {"※": [-1.0]}}')
    (cell_dir / "run.log").write_text("info: training started\n")
    return cell_dir


def test_cleanup_pass1_removes_intermediate_checkpoints() -> None:
    """All 6 ``adapter/checkpoint-*/`` dirs must be removed; the final
    ``adapter/`` dir + its ``adapter_config.json`` + ``adapter_model.
    safetensors`` MUST survive (Pass 2 needs them for ``LoRARequest``).
    """
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = _stage_pass1_cell_dir(Path(tmp))
        removed = _dispatch._cleanup_pass1_cell(cell_dir)

        assert removed["checkpoints_removed"] == 6, (
            f"Expected 6 checkpoint dirs removed; got {removed['checkpoints_removed']}"
        )
        # No checkpoint-* dirs survive under adapter/.
        adapter_dir = cell_dir / "adapter"
        survivors = list(adapter_dir.glob("checkpoint-*"))
        assert survivors == [], f"Round 13: checkpoint dirs survived cleanup: {survivors}"
        # Final adapter weights MUST survive (Pass 2 needs them).
        assert adapter_dir.is_dir(), "Round 13: adapter/ dir was wrongly deleted"
        assert (adapter_dir / "adapter_config.json").exists(), (
            "Round 13: final adapter_config.json was wrongly deleted (Pass 2 needs it)"
        )
        assert (adapter_dir / "adapter_model.safetensors").exists(), (
            "Round 13: final adapter_model.safetensors was wrongly deleted "
            "(Pass 2's LoRARequest needs it)"
        )


def test_cleanup_pass1_removes_prepared_train_jsonl() -> None:
    """``prepared_train.jsonl`` (~1 MB per cell) must be removed; Pass 2
    only needs the recipe-fix manifest (``prepared_dataset.json``).
    """
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = _stage_pass1_cell_dir(Path(tmp))
        removed = _dispatch._cleanup_pass1_cell(cell_dir)

        assert removed["prepared_train_removed"] == 1
        assert not (cell_dir / "prepared_train.jsonl").exists()


def test_cleanup_pass1_preserves_logprob_and_manifest_and_runlog() -> None:
    """The Pass 1 deliverable + manifest + run log MUST survive cleanup.
    Without these, Pass 2 cannot build the train-matched panel and the
    sweep can't be debugged after the fact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = _stage_pass1_cell_dir(Path(tmp))
        _dispatch._cleanup_pass1_cell(cell_dir)

        assert (cell_dir / "logprob_panel.json").exists(), (
            "Round 13: logprob_panel.json was wrongly deleted (Pass 1 deliverable)"
        )
        assert (cell_dir / "prepared_dataset.json").exists(), (
            "Round 13: prepared_dataset.json was wrongly deleted "
            "(Pass 2's build_train_matched_persona_panel reads it)"
        )
        assert (cell_dir / "run.log").exists(), (
            "Round 13: run.log was wrongly deleted (needed for debug)"
        )


def test_cleanup_pass1_removes_wandb_run_dir_when_present() -> None:
    """Any ``wandb*`` dirs in the cell folder must be removed. WandB
    runs live in the cloud; the local mirror is just a disk hog.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = _stage_pass1_cell_dir(Path(tmp))
        wandb_dir = cell_dir / "wandb"
        wandb_dir.mkdir()
        (wandb_dir / "run-12345.wandb").write_bytes(b"x" * 10000)
        offline_dir = cell_dir / "wandb-offline-run-67890"
        offline_dir.mkdir()
        (offline_dir / "events.log").write_text("event\n")

        removed = _dispatch._cleanup_pass1_cell(cell_dir)
        assert removed["wandb_dirs_removed"] >= 1
        assert not wandb_dir.exists()
        assert not offline_dir.exists()


def test_cleanup_pass1_idempotent_when_already_clean() -> None:
    """Calling the helper twice (or on a cell that was already cleaned)
    must NOT raise. Returns zero counts on the second call.

    This is the "re-run after partial recovery" path — the resume scan
    has already populated cleanly; we don't want a re-launch to error
    out trying to delete something that's gone.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = _stage_pass1_cell_dir(Path(tmp))
        first = _dispatch._cleanup_pass1_cell(cell_dir)
        assert first["checkpoints_removed"] == 6
        assert first["prepared_train_removed"] == 1

        second = _dispatch._cleanup_pass1_cell(cell_dir)
        assert second == {
            "checkpoints_removed": 0,
            "prepared_train_removed": 0,
            "wandb_dirs_removed": 0,
        }


def test_cleanup_pass1_handles_missing_adapter_dir() -> None:
    """If ``adapter/`` is missing entirely (training failed early),
    cleanup must NOT raise — it just reports zero checkpoint removals.

    This is a degenerate case that shouldn't happen in practice (the
    verify gate before cleanup catches "adapter missing on Hub"), but
    we don't want a defensive raise to crash the per-cell loop's
    accounting.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = Path(tmp) / "barren"
        cell_dir.mkdir()
        # Only logprob + manifest, no adapter dir.
        (cell_dir / "logprob_panel.json").write_text("{}")
        (cell_dir / "prepared_dataset.json").write_text("{}")

        removed = _dispatch._cleanup_pass1_cell(cell_dir)
        assert removed["checkpoints_removed"] == 0
        assert removed["prepared_train_removed"] == 0
        assert removed["wandb_dirs_removed"] == 0


def test_cleanup_pass1_footprint_meets_budget() -> None:
    """Round 13 budget contract: after cleanup, the cell-dir footprint
    is dominated by the final adapter (~485 MB in production; in this
    test, the staged adapter is ~18 KB). The cleanup must remove
    SIGNIFICANTLY more bytes than it leaves behind — proves we didn't
    accidentally invert the keep/delete logic.

    With the staged sizes (~120 KB per checkpoint x 6 + ~48 KB JSONL =
    ~768 KB cleaned; ~36 KB adapter survives), the ratio is ~20x.
    Production ratio: ~3 GB cleaned / ~485 MB survive = ~6x.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = _stage_pass1_cell_dir(Path(tmp))

        def _dir_size(path: Path) -> int:
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

        before = _dir_size(cell_dir)
        _dispatch._cleanup_pass1_cell(cell_dir)
        after = _dir_size(cell_dir)

        assert after < before, "Cleanup didn't reduce footprint"
        # Cleanup should have removed > 4x what survives (production
        # ratio is ~6x; staged test sizes give >20x).
        assert before > 4 * after, (
            f"Round 13: cleanup removed only {before - after} bytes vs {after} "
            "surviving — that's not enough to justify the helper. Check the "
            "keep/delete logic for inversion."
        )
