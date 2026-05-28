"""Round 12 two-pass sweep contract (task #397).

Round 12 abandoned the Round 11 single-pass serial loop after the round-11
reviewer FAILed on missing vLLM→HF teardown between cells. Each cell ended
with vLLM (incomplete Python-level teardown leaves orphan workers + KV
cache pinned per CLAUDE.md vLLM-orphan-worker gotcha); next cell started
with HF expecting free GPU. OOM expected within a few cells.

The two-pass design eliminates the framework-switch within a pass:

  Pass 1: HF only (train + log-prob eval) across all cells. No vLLM.
          Standard Python GC releases memory between cells. No
          orphan-worker risk because nothing else loads after HF.

  Single ``_aggressive_hf_to_vllm_teardown`` event between passes.

  Pass 2: vLLM only. ``LLM(enable_lora=True)`` loaded ONCE; per-cell
          ``LoRARequest`` swaps in the cell's adapter for sampled eval.
          vLLM's native LoRA-swap mechanism.

This file pins the two-pass pipeline:

  1. ``_run_pass1_hf`` calls prepare_cell_jsonl → train_one_cell →
     compute_logprob_panel → write logprob_panel.json, in that order,
     for every cell in its input list. NO vLLM calls.
  2. ``_run_pass2_vllm`` loads vLLM ONCE, then per cell: LoRARequest →
     llm.generate → score → write metrics.json → verify → cleanup.
  3. ``_run_sweep_two_pass`` calls _run_pass1_hf BEFORE
     _aggressive_hf_to_vllm_teardown BEFORE _run_pass2_vllm. Order
     matters: any other order re-opens the round-11 OOM risk.
  4. Per-cell exception in Pass 1 → rc=1, sweep continues (cell skipped
     in Pass 2 because adapter doesn't exist).
  5. Pass 2 verify-FAIL → rc=2, local weights preserved (CLAUDE.md
     upload-policy contract).

CPU-only; no GPU, no model load. All heavy entry points are
monkeypatched at the module level.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Load the dispatcher (lives under scripts/, not a package).
_DISPATCH_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
)
_spec = importlib.util.spec_from_file_location("dispatch_factor_screen_397", _DISPATCH_PATH)
_dispatch = importlib.util.module_from_spec(_spec)
sys.modules["dispatch_factor_screen_397"] = _dispatch
_spec.loader.exec_module(_dispatch)


def _build_args(slab_root: Path) -> argparse.Namespace:
    """Build the args namespace the two-pass sweep expects."""
    return argparse.Namespace(
        issue=397,
        mode="sweep",
        pool_dir=slab_root / "pools",
        slab_root=slab_root,
        smoke_cell="10010",
        smoke_source="librarian",
        smoke_seed=42,
        sources="librarian",
        seeds="42",
        marker_token="※",
        save_every_n_steps=25,
        pos_per_source=400,
        lr=1e-4,
        warmup_ratio=0.10,
        require_smoke_pass=True,
        skip_smoke_pass_check=False,
        smoke_pass_confirmed=True,
        dry_run=False,
        no_resume=True,
        resume_source="both",
        log_level="INFO",
    )


def _stub_pass1_deps(monkeypatch, *, verify_passes: bool = True) -> dict:
    """Monkeypatch every heavy Pass 1 entry point.

    Returns a dict of call records the caller can assert on.

    Round 13: Pass 1 now also calls ``verify_adapter_on_hf_hub``
    (LOUD-FAIL gate before cleanup) and ``_cleanup_pass1_cell`` (the
    disk-quota fix). Both are stubbed so tests can assert call order +
    verify-fail semantics without hitting HF Hub or running shutil.
    """
    records: dict[str, list] = {
        "prepare_cell_jsonl": [],
        "train_one_cell": [],
        "compute_logprob_panel": [],
        "verify": [],
        "cleanup": [],
        "order": [],
    }

    FAKE_SYSTEM_PROMPT = "FAKE-SYSTEM-PROMPT-FROM-STUBBED-PREPARE-CELL-JSONL"

    def _fake_prepare(**kwargs):
        records["prepare_cell_jsonl"].append(kwargs)
        records["order"].append("prepare_cell_jsonl")
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"messages": []}\n', encoding="utf-8")
        return {
            "output_path": output_path,
            "num_positive": 400,
            "num_negative": 400,
            "num_total": 800,
            "data_policy": "off_policy",
            "system_prompt_text": FAKE_SYSTEM_PROMPT,
        }

    import explore_persona_space.experiments.factor_screen_397.data_prep as dp_mod

    monkeypatch.setattr(dp_mod, "prepare_cell_jsonl", _fake_prepare)

    def _fake_train(**kwargs):
        from explore_persona_space.experiments.factor_screen_397.training import (
            TrainOutcome,
            write_prepared_dataset_manifest,
        )

        records["train_one_cell"].append(kwargs)
        records["order"].append("train_one_cell")
        write_prepared_dataset_manifest(
            kwargs["cell_output_dir"],
            cell_key=kwargs["cell"].key,
            source=kwargs["source"],
            seed=kwargs["seed"],
            system_prompt_text=kwargs["system_prompt_text"],
            marker_text=kwargs["marker_text"],
        )
        (kwargs["cell_output_dir"] / "adapter" / "checkpoint-25").mkdir(parents=True)
        return TrainOutcome(
            cell_key=kwargs["cell"].key,
            seed=kwargs["seed"],
            adapter_path=str(kwargs["cell_output_dir"] / "adapter"),
            loss=1.23,
            train_wall_minutes=0.5,
            n_examples=800,
            total_steps=150,
            marker_only_loss=True,
            marker_tail_tokens=0,
        )

    import explore_persona_space.experiments.factor_screen_397.training as training_mod

    monkeypatch.setattr(training_mod, "train_one_cell", _fake_train)

    monkeypatch.setattr(
        _dispatch,
        "_load_base_model_for_logprob",
        lambda first_checkpoint_dir: (MagicMock(name="base"), MagicMock(name="tok_lp")),
    )

    def _fake_compute_logprob(**kwargs):
        records["compute_logprob_panel"].append(kwargs)
        records["order"].append("compute_logprob_panel")
        return {kwargs["checkpoint_dirs"][0]: {"※": [-1.0]}}

    import explore_persona_space.experiments.factor_screen_397.eval_panel as ep_mod

    monkeypatch.setattr(ep_mod, "compute_logprob_panel", _fake_compute_logprob)

    # AutoTokenizer used inside _run_pass1_hf for the recipe-fix preflight.
    fake_tokenizer = MagicMock(name="autotok_pass1")
    fake_tokenizer.pad_token_id = 0

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return fake_tokenizer

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAutoTokenizer)

    # Round 13: verify gate (real one hits HF Hub) + cleanup helper
    # (real one shutil.rmtree's the staged adapter). Stub both so tests
    # run on CPU without network OR risk of clobbering the test's
    # temp-staged checkpoint dirs.
    def _fake_verify(**kwargs):
        records["verify"].append(kwargs)
        records["order"].append("verify_adapter_on_hf_hub")
        return verify_passes

    monkeypatch.setattr(_dispatch, "verify_adapter_on_hf_hub", _fake_verify)

    def _fake_cleanup(cell_dir):
        records["cleanup"].append(cell_dir)
        records["order"].append("_cleanup_pass1_cell")
        return {"checkpoints_removed": 6, "prepared_train_removed": 1, "wandb_dirs_removed": 0}

    monkeypatch.setattr(_dispatch, "_cleanup_pass1_cell", _fake_cleanup)

    return records


def test_pass1_calls_only_hf_helpers_in_canonical_order(monkeypatch) -> None:
    """Pass 1 must traverse prepare_cell_jsonl → train_one_cell →
    compute_logprob_panel → write logprob_panel.json → verify on Hub
    → cleanup local for EVERY cell, in that order. NO vLLM calls in
    this pass.

    This is the load-bearing contract: Pass 1 ending without a
    framework switch is what eliminates the round-11 orphan-worker
    OOM risk. Round 13 adds the verify + cleanup tail (disk-quota fix).
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_args(slab_root)
        records = _stub_pass1_deps(monkeypatch)

        cells = [
            (Cell.from_key("00000"), "librarian", 42),
            (Cell.from_key("00001"), "librarian", 42),
        ]
        rcs = _dispatch._run_pass1_hf(cells, args=args)

        # Both cells succeed.
        assert rcs == {("00000", "librarian", 42): 0, ("00001", "librarian", 42): 0}

        # Per-cell canonical order, twice (Round 13 adds verify + cleanup).
        per_cell_seq = [
            "prepare_cell_jsonl",
            "train_one_cell",
            "compute_logprob_panel",
            "verify_adapter_on_hf_hub",
            "_cleanup_pass1_cell",
        ]
        assert records["order"] == per_cell_seq * 2, f"Pass 1 order wrong: {records['order']}"

        # logprob_panel.json written for both cells.
        for cell_key in ("00000", "00001"):
            logprob_path = (
                slab_root
                / f"cell_{cell_key}"
                / "source_librarian"
                / "seed_42"
                / "logprob_panel.json"
            )
            assert logprob_path.exists(), f"Pass 1 must write {logprob_path}"


def test_pass1_train_one_cell_called_with_hf_upload_true(monkeypatch) -> None:
    """Pass 1 uses train_one_cell(hf_upload=True) so the TRL inline-upload
    fence pushes the adapter to HF Hub during training. Pass 2's
    verify_adapter_on_hf_hub is the safety net for fence silent-swallow.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_args(slab_root)
        records = _stub_pass1_deps(monkeypatch)

        cells = [(Cell.from_key("00000"), "librarian", 42)]
        _dispatch._run_pass1_hf(cells, args=args)

        assert len(records["train_one_cell"]) == 1
        assert records["train_one_cell"][0]["hf_upload"] is True


def test_pass1_threads_system_prompt_overrides_into_compute_logprob_panel(monkeypatch) -> None:
    """SR1 wiring (reconciler SR1): Pass 1's compute_logprob_panel call
    MUST receive system_prompt_overrides from the recipe-fix manifest.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_args(slab_root)
        records = _stub_pass1_deps(monkeypatch)

        cells = [(Cell.from_key("00000"), "librarian", 42)]
        _dispatch._run_pass1_hf(cells, args=args)

        assert len(records["compute_logprob_panel"]) == 1
        kwargs = records["compute_logprob_panel"][0]
        assert "system_prompt_overrides" in kwargs, (
            "SR1 broken: Pass 1 did not thread system_prompt_overrides to compute_logprob_panel."
        )
        FAKE = "FAKE-SYSTEM-PROMPT-FROM-STUBBED-PREPARE-CELL-JSONL"
        assert kwargs["system_prompt_overrides"] == {"librarian": FAKE}


def test_pass1_exception_is_caught_as_rc1_and_loop_continues(monkeypatch) -> None:
    """A Pass 1 cell crash (any Python exception in prepare / train /
    log-prob) must NOT kill the pass — it lands as rc=1 in the per-cell
    dict and the next cell runs.

    Without this safety net, a single bug in train_one_cell would take
    down the whole 108-cell Pass 1 and leave the pod with no Pass 2
    work to do.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_args(slab_root)
        _stub_pass1_deps(monkeypatch)

        # Inject a crash on the first train_one_cell call only.
        crash_count = {"n": 0}
        import explore_persona_space.experiments.factor_screen_397.training as training_mod

        def _crashing_train(**kwargs):
            crash_count["n"] += 1
            if crash_count["n"] == 1:
                raise RuntimeError("simulated training crash")
            from explore_persona_space.experiments.factor_screen_397.training import (
                TrainOutcome,
                write_prepared_dataset_manifest,
            )

            write_prepared_dataset_manifest(
                kwargs["cell_output_dir"],
                cell_key=kwargs["cell"].key,
                source=kwargs["source"],
                seed=kwargs["seed"],
                system_prompt_text=kwargs["system_prompt_text"],
                marker_text=kwargs["marker_text"],
            )
            (kwargs["cell_output_dir"] / "adapter" / "checkpoint-25").mkdir(parents=True)
            return TrainOutcome(
                cell_key=kwargs["cell"].key,
                seed=kwargs["seed"],
                adapter_path=str(kwargs["cell_output_dir"] / "adapter"),
                loss=1.0,
                train_wall_minutes=0.1,
                n_examples=800,
                total_steps=150,
                marker_only_loss=True,
                marker_tail_tokens=0,
            )

        monkeypatch.setattr(training_mod, "train_one_cell", _crashing_train)

        cells = [
            (Cell.from_key("00000"), "librarian", 42),
            (Cell.from_key("00001"), "librarian", 42),
        ]
        rcs = _dispatch._run_pass1_hf(cells, args=args)
        assert rcs == {
            ("00000", "librarian", 42): 1,  # crashed
            ("00001", "librarian", 42): 0,  # second cell succeeded after crash
        }


def test_pass1_cleanup_runs_inside_loop_after_verify(monkeypatch) -> None:
    """Round 13 disk-quota contract: ``_cleanup_pass1_cell`` MUST be
    invoked AFTER ``verify_adapter_on_hf_hub`` AND inside the per-cell
    loop (so cell N's checkpoints are gone BEFORE cell N+1 starts
    training).

    The motivating bug: Round 12 left intermediate checkpoints +
    prepared_train.jsonl on disk after each Pass 1 cell. Sweep crashed
    at cell 22/108 with ENOSPC (~93 GB on a 200 GB pod).

    This test asserts:
      1. Cleanup is called ONCE per cell (not zero, not lazily at
         end-of-pass).
      2. For every cell, verify_adapter_on_hf_hub appears BEFORE
         _cleanup_pass1_cell in the recorded sequence (verify is the
         gate; cleanup runs only on verify PASS).
      3. The cleanup helper receives the right cell_dir.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_args(slab_root)
        records = _stub_pass1_deps(monkeypatch, verify_passes=True)

        cells = [
            (Cell.from_key("00000"), "librarian", 42),
            (Cell.from_key("00001"), "librarian", 42),
        ]
        rcs = _dispatch._run_pass1_hf(cells, args=args)
        assert rcs == {("00000", "librarian", 42): 0, ("00001", "librarian", 42): 0}

        # (1) Cleanup invoked once per cell.
        assert len(records["cleanup"]) == 2, (
            f"Round 13: cleanup must run once per cell; got {len(records['cleanup'])} calls"
        )

        # (2) Per cell, verify appears BEFORE cleanup. Walk the recorded
        # order and pair them up: index i*5+3 is the verify for cell i,
        # index i*5+4 is the cleanup.
        order = records["order"]
        for cell_idx in range(2):
            v_pos = cell_idx * 5 + 3  # 3, 8 — verify positions in the 5-step seq
            c_pos = cell_idx * 5 + 4  # 4, 9 — cleanup positions
            assert order[v_pos] == "verify_adapter_on_hf_hub", (
                f"Round 13: cell {cell_idx} verify call at wrong position; got order={order}"
            )
            assert order[c_pos] == "_cleanup_pass1_cell", (
                f"Round 13: cell {cell_idx} cleanup call at wrong position; got order={order}"
            )

        # (3) Cleanup received the right cell dirs.
        expected_cell_dirs = [
            slab_root / "cell_00000" / "source_librarian" / "seed_42",
            slab_root / "cell_00001" / "source_librarian" / "seed_42",
        ]
        assert records["cleanup"] == expected_cell_dirs, (
            f"Round 13: cleanup got wrong cell dirs; expected {expected_cell_dirs}, "
            f"got {records['cleanup']}"
        )


def test_pass1_cleanup_skipped_when_verify_fails(monkeypatch) -> None:
    """Round 13 LOUD-FAIL contract: when ``verify_adapter_on_hf_hub``
    returns False (TRL inline-upload fence silently swallowed an upload
    failure per CLAUDE.md gotcha), Pass 1 MUST NOT call
    ``_cleanup_pass1_cell`` — local weights are preserved for retry,
    and the cell exits rc=2 so the sweep summary surfaces the failure.

    This is the load-bearing safety net per CLAUDE.md upload-policy:
    "Models MUST upload to HF model repo before local deletion".
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_args(slab_root)
        records = _stub_pass1_deps(monkeypatch, verify_passes=False)

        cells = [(Cell.from_key("00000"), "librarian", 42)]
        rcs = _dispatch._run_pass1_hf(cells, args=args)

        # rc=2 mirrors Pass 2's verify-fail convention.
        assert rcs == {("00000", "librarian", 42): 2}, (
            f"Round 13: verify-FAIL must produce rc=2; got {rcs}"
        )
        # Cleanup MUST NOT have been called.
        assert len(records["cleanup"]) == 0, (
            "Round 13: cleanup was called even though verify failed — this "
            "would delete local weights for a cell whose adapter is NOT on Hub, "
            "violating CLAUDE.md upload-policy."
        )
        # Verify WAS called (one cell, one verify probe).
        assert len(records["verify"]) == 1


def test_pass1_cleanup_runs_before_next_cell_starts(monkeypatch) -> None:
    """Round 13 disk-quota contract (per-cell-immediate): cell N's
    cleanup MUST run BEFORE cell N+1's train_one_cell starts.

    If cleanup happened lazily at end-of-pass (e.g. one batch
    cleanup loop after all cells finish), the disk-full bug would
    still trigger — cell N+1 would be training while cell N's
    checkpoints are still on disk.

    Pins the interleaving: for any two consecutive cells, cell N's
    cleanup must come BEFORE cell N+1's prepare_cell_jsonl in the
    recorded order.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_args(slab_root)
        records = _stub_pass1_deps(monkeypatch, verify_passes=True)

        cells = [
            (Cell.from_key("00000"), "librarian", 42),
            (Cell.from_key("00001"), "librarian", 42),
            (Cell.from_key("00002"), "librarian", 42),
        ]
        _dispatch._run_pass1_hf(cells, args=args)

        order = records["order"]
        # Per-cell 5-step sequence; cleanup is the LAST step per cell.
        # Find each cleanup position and confirm it comes BEFORE the next
        # cell's prepare_cell_jsonl (the first step of cell N+1).
        cleanup_positions = [i for i, step in enumerate(order) if step == "_cleanup_pass1_cell"]
        prepare_positions = [i for i, step in enumerate(order) if step == "prepare_cell_jsonl"]
        assert len(cleanup_positions) == 3, f"Expected 3 cleanups; got {cleanup_positions}"
        assert len(prepare_positions) == 3, f"Expected 3 prepares; got {prepare_positions}"

        # Cell 1's prepare must come AFTER cell 0's cleanup.
        assert cleanup_positions[0] < prepare_positions[1], (
            f"Round 13: cell 0 cleanup (pos {cleanup_positions[0]}) must come "
            f"BEFORE cell 1 prepare (pos {prepare_positions[1]}); got order={order}"
        )
        # Cell 2's prepare must come AFTER cell 1's cleanup.
        assert cleanup_positions[1] < prepare_positions[2], (
            f"Round 13: cell 1 cleanup (pos {cleanup_positions[1]}) must come "
            f"BEFORE cell 2 prepare (pos {prepare_positions[2]}); got order={order}"
        )


def test_two_pass_sweep_calls_pass1_then_teardown_then_pass2(monkeypatch) -> None:
    """Top-level contract: _run_sweep_two_pass calls _run_pass1_hf BEFORE
    _aggressive_hf_to_vllm_teardown BEFORE _run_pass2_vllm.

    Inverting any of the three (Pass 2 before teardown, teardown before
    Pass 1, etc.) re-opens the round-11 vulnerability. This is the load-
    bearing ordering test for the round-12 design.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    order: list[str] = []

    def _spy_pass1(cells_to_run, *, args):
        order.append("pass1")
        # Honor the API: return rc=0 per cell.
        return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

    def _spy_teardown(*args, **kwargs):
        order.append("teardown")

    def _spy_pass2(cells_to_run, *, args):
        order.append("pass2")
        return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

    monkeypatch.setattr(_dispatch, "_run_pass1_hf", _spy_pass1)
    monkeypatch.setattr(_dispatch, "_aggressive_hf_to_vllm_teardown", _spy_teardown)
    monkeypatch.setattr(_dispatch, "_run_pass2_vllm", _spy_pass2)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_args(slab_root)
        cells = [Cell.from_key("00000")]
        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch._run_sweep_two_pass(
            sources=["librarian"], seeds=[42], cells=cells, args=args, repo_root=repo_root
        )
        assert rc == 0
        assert order == ["pass1", "teardown", "pass2"], (
            f"Two-pass ordering broken; got {order}. "
            "Pass 1 must run BEFORE teardown BEFORE Pass 2 — any other order "
            "re-introduces the round-11 vLLM→HF orphan-worker risk."
        )


def test_two_pass_sweep_writes_summary_with_both_pass_rcs(monkeypatch) -> None:
    """sweep_summary.json must carry per-cell pass1_rc + pass2_rc +
    final_rc so the orchestrator can diagnose which pass failed for a
    bad cell.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    def _fake_pass1(cells_to_run, *, args):
        # Cell 00000: Pass 1 fails (rc=1). Cell 00001: Pass 1 succeeds.
        rcs = {}
        for c, s, sd in cells_to_run:
            rcs[(c.key, s, sd)] = 1 if c.key == "00000" else 0
        return rcs

    def _fake_pass2(cells_to_run, *, args):
        # Only cells with successful Pass 1 reach Pass 2 in real flow,
        # but the dispatcher doesn't filter — it sends every Pass-2-needed
        # cell. Our fake_pass1 records both cells as "Pass 1 attempted",
        # so both end up in Pass 2 in this test. Cell 00001 succeeds (rc=0).
        # Cell 00000 fails verify (rc=2) because adapter wasn't actually
        # written (Pass 1 crashed).
        rcs = {}
        for c, s, sd in cells_to_run:
            rcs[(c.key, s, sd)] = 2 if c.key == "00000" else 0
        return rcs

    monkeypatch.setattr(_dispatch, "_run_pass1_hf", _fake_pass1)
    monkeypatch.setattr(_dispatch, "_aggressive_hf_to_vllm_teardown", lambda *a, **k: None)
    monkeypatch.setattr(_dispatch, "_run_pass2_vllm", _fake_pass2)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_args(slab_root)
        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        repo_root = Path(__file__).resolve().parent.parent.parent
        _dispatch._run_sweep_two_pass(
            sources=["librarian"], seeds=[42], cells=cells, args=args, repo_root=repo_root
        )

        summary = json.loads((slab_root / "sweep_summary.json").read_text(encoding="utf-8"))
        assert summary["two_pass_mode"] is True
        per_cell = {(r["cell"], r["source"], r["seed"]): r for r in summary["per_cell"]}
        assert per_cell[("00000", "librarian", 42)]["pass1_rc"] == 1
        assert per_cell[("00000", "librarian", 42)]["pass2_rc"] == 2
        assert per_cell[("00000", "librarian", 42)]["final_rc"] == 2  # max(1, 2)
        assert per_cell[("00001", "librarian", 42)]["pass1_rc"] == 0
        assert per_cell[("00001", "librarian", 42)]["pass2_rc"] == 0
        assert per_cell[("00001", "librarian", 42)]["final_rc"] == 0


def test_two_pass_sweep_skips_pass2_when_no_pass2_jobs(monkeypatch) -> None:
    """When every cell is already fully complete (resume case),
    _run_pass2_vllm must NOT be invoked — invoking vLLM with zero jobs
    would waste the ~5 min base-load wall time and the orphan-worker
    risk from the teardown happens for nothing.

    Same goes for the teardown — it's expensive (gc + sync), and
    pointless if Pass 2 has nothing to do.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    # Pre-stage all artifacts so resume marks everything as fully complete.
    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        for cell_key in ("00000",):
            cell_dir = slab_root / f"cell_{cell_key}" / "source_librarian" / "seed_42"
            cell_dir.mkdir(parents=True)
            (cell_dir / "adapter").mkdir()
            (cell_dir / "logprob_panel.json").write_text(json.dumps({"ckpt-25": {"※": [-1.0]}}))
            (cell_dir / "metrics.json").write_text(
                json.dumps({"personas": {"librarian": {"substring_rate": 0.5}}})
            )

        args = _build_args(slab_root)
        args.no_resume = False  # exercise the resume scan
        args.resume_source = "local"  # no Hub probe

        pass1_called = {"n": 0}
        pass2_called = {"n": 0}
        teardown_called = {"n": 0}

        def _spy_pass1(cells_to_run, *, args):
            pass1_called["n"] += 1
            return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

        def _spy_pass2(cells_to_run, *, args):
            pass2_called["n"] += 1
            return {(c.key, s, sd): 0 for (c, s, sd) in cells_to_run}

        monkeypatch.setattr(_dispatch, "_run_pass1_hf", _spy_pass1)
        monkeypatch.setattr(
            _dispatch,
            "_aggressive_hf_to_vllm_teardown",
            lambda *a, **k: teardown_called.update({"n": teardown_called["n"] + 1}),
        )
        monkeypatch.setattr(_dispatch, "_run_pass2_vllm", _spy_pass2)

        cells = [Cell.from_key("00000")]
        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch._run_sweep_two_pass(
            sources=["librarian"], seeds=[42], cells=cells, args=args, repo_root=repo_root
        )
        assert rc == 0
        assert pass1_called["n"] == 0, "Pass 1 must not run when resume covers everything"
        assert pass2_called["n"] == 0, "Pass 2 must not run when resume covers everything"
        assert teardown_called["n"] == 0, (
            "Teardown must not run when Pass 2 has no work — the only reason "
            "for the teardown is to prepare for Pass 2's vLLM load."
        )


def test_pass1_persists_logprob_before_dropping_refs(monkeypatch) -> None:
    """CLAUDE.md "Checkpoint per phase" enforcement: logprob_panel.json
    must land on disk INSIDE the per-cell loop, BEFORE refs get dropped.

    The motivating gotcha is task #399: an in-memory accumulation
    pattern lost 15 min x 11 rounds of Phase 1 work when downstream
    Phase 2 loads crashed before write_seed_outputs at end-of-seed.
    Round 12 Pass 1 must persist per-cell-immediately so a downstream
    Pass 2 crash doesn't lose Pass 1 work.

    Pin source-order via AST: logprob_path.write_text must come BEFORE
    the `del base_model` line in _run_pass1_hf.
    """
    import ast

    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_pass1_hf":
            target = node
            break
    assert target is not None

    write_line = None
    del_line = None
    for node in ast.walk(target):
        # Find logprob_path.write_text(...) call line.
        if isinstance(node, ast.Call):
            try:
                expr = ast.unparse(node.func)
            except Exception:
                continue
            if expr == "logprob_path.write_text" and write_line is None:
                write_line = node.lineno
        # Find `del base_model` statement line.
        if isinstance(node, ast.Delete):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "base_model" and del_line is None:
                    del_line = node.lineno

    assert write_line is not None, (
        "Pass 1 must call logprob_path.write_text(...) inside the loop "
        "(CLAUDE.md checkpoint-per-phase)."
    )
    assert del_line is not None, "Pass 1 must `del base_model` to release refs."
    assert write_line < del_line, (
        f"CLAUDE.md checkpoint-per-phase violation: logprob_path.write_text "
        f"(line {write_line}) must come BEFORE `del base_model` "
        f"(line {del_line}). Otherwise a memory-release crash loses the "
        "Pass 1 output."
    )


def test_pass2_writes_metrics_before_verify_before_cleanup() -> None:
    """Pin Pass 2 source-order via AST:

      metrics_path.write_text  (CLAUDE.md checkpoint-per-phase)
      → verify_adapter_on_hf_hub  (CLAUDE.md upload-policy gate)
      → cleanup_cell_local_weights  (only after verify PASS)

    Inverting verify ↔ cleanup violates the upload policy. Inverting
    metrics-write ↔ verify means a verify-fail loses the scored data.
    """
    import ast

    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_pass2_vllm":
            target = node
            break
    assert target is not None

    metrics_line = None
    verify_line = None
    cleanup_line = None
    for node in ast.walk(target):
        if isinstance(node, ast.Call):
            try:
                expr = ast.unparse(node.func)
            except Exception:
                continue
            if expr == "metrics_path.write_text" and metrics_line is None:
                metrics_line = node.lineno
            if expr == "verify_adapter_on_hf_hub" and verify_line is None:
                verify_line = node.lineno
            if expr == "cleanup_cell_local_weights" and cleanup_line is None:
                cleanup_line = node.lineno

    assert metrics_line and verify_line and cleanup_line
    assert metrics_line < verify_line, (
        f"CLAUDE.md checkpoint-per-phase: metrics_path.write_text (line "
        f"{metrics_line}) must come BEFORE verify_adapter_on_hf_hub "
        f"(line {verify_line})."
    )
    assert verify_line < cleanup_line, (
        f"CLAUDE.md upload-policy: verify_adapter_on_hf_hub (line "
        f"{verify_line}) must come BEFORE cleanup_cell_local_weights "
        f"(line {cleanup_line}) — never delete local weights without "
        "confirming the adapter is on HF Hub."
    )
