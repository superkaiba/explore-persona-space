"""Wave-size + CVD-pin + resume-skip regression for the #813 extraction dispatcher.

Pins the invariants the feedback memory
``dispatcher_wave_size_must_match_visible_gpus`` (#667 a36) demands of any
per-cell subprocess wave dispatcher, mirroring
``tests/test_issue667_alllayer_wave.py``:

1. ``compute_wave_size`` derives the parallel wave from the DETECTED visible-GPU
   count (``torch.cuda.device_count()``), NOT a hardcoded constant or the
   ``--n-gpus`` default; ``--n-gpus`` is a CEILING; a GPU run with 0 visible
   devices RAISES loud (never a silent CPU fallback); ``--cpu-only`` -> 1; and a
   ``--dry-run`` previews the requested ceiling without touching CUDA.

2. Every per-cell command pins ``CUDA_VISIBLE_DEVICES=<gpu>`` in the LAUNCHER env
   matching its ``--gpu-id`` (the #545 launcher-env pin an import-time cuInit
   cannot defeat) AND passes the matching ``--gpu-id``.

3. The 12 (behavior x substrate) cells enumerate correctly, and the run-cell
   resume-skip sentinel predicate skips a completed cell.

Pure logic, no GPU, ~1s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue813_dispatch as disp  # noqa: E402


def test_cpu_only_wave_is_serial():
    assert disp.compute_wave_size(cpu_only=True, requested_n_gpus=8) == 1


def test_dry_run_previews_requested_ceiling_without_cuda(monkeypatch):
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 0)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=8, dry_run=True) == 8


def test_wave_equals_detected_when_below_ceiling(monkeypatch):
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 8)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=8) == 8


def test_wave_clamps_to_detected_below_ceiling(monkeypatch):
    # The #667 a36 hang class: --n-gpus 8 on a 1-GPU lane must NOT spawn
    # --gpu-id 1..7 (which would see no device and silently run on CPU).
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 1)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=8) == 1


def test_ceiling_below_detected_is_honored(monkeypatch):
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 8)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=4) == 4


def test_zero_visible_gpu_raises_loud(monkeypatch):
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 0)
    with pytest.raises(RuntimeError, match="no CUDA devices visible"):
        disp.compute_wave_size(cpu_only=False, requested_n_gpus=8)


def test_enumerate_cells_is_full_grid():
    cells = disp.enumerate_cells(list(disp.BEHAVIORS), list(disp.SUBSTRATES))
    assert len(cells) == 12  # 4 behaviors x 3 substrates
    assert ("em", "generic") in cells and ("marker", "mix") in cells


def test_per_cell_cvd_pin_matches_gpu_id():
    # The launcher-env CVD pin must match --gpu-id for every slot (gotchas.md #545).
    for gpu_id in range(8):
        cmd, env = disp._cell_cmd(
            "em",
            "generic",
            gpu_id,
            out_root="eval_results/issue_813",
            cpu_only=False,
            upload=True,
            max_contexts=None,
            max_questions=None,
        )
        assert env["CUDA_VISIBLE_DEVICES"] == str(gpu_id), env
        assert "--gpu-id" in cmd and cmd[cmd.index("--gpu-id") + 1] == str(gpu_id), cmd
        assert "--upload" in cmd, cmd
        assert cmd[cmd.index("--behavior") + 1] == "em"
        assert cmd[cmd.index("--substrate") + 1] == "generic"


def test_cpu_only_cell_cmd_has_no_cvd_pin():
    # CPU-only lanes must NOT pin CVD (there is no physical GPU to pin).
    cmd, env = disp._cell_cmd(
        "marker",
        "mix",
        0,
        out_root="eval_results/issue_813",
        cpu_only=True,
        upload=False,
        max_contexts=2,
        max_questions=2,
    )
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "--cpu-only" in cmd
    assert "--upload" not in cmd
    assert cmd[cmd.index("--max-contexts") + 1] == "2"


def test_run_cell_resume_skip_predicate(tmp_path, monkeypatch):
    # A completed cell (sentinel present) is skipped on re-run unless --force.
    import issue813_run_cell as rc

    reduced_dir = tmp_path / "reduced" / "marker" / "generic"
    reduced_dir.mkdir(parents=True)
    sentinel = reduced_dir / rc.CELL_DONE_SENTINEL
    sentinel.write_text(json.dumps({"behavior": "marker", "substrate": "generic"}))

    class Args:
        behavior = "marker"
        substrate = "generic"
        out_root = tmp_path
        gpu_id = 0
        cpu_only = True
        upload = False
        force = False
        gate_only = False
        max_contexts = 2
        max_questions = 2
        metrics_out = None

    out = rc.run_cell(Args())
    assert out.get("skipped") is True, out


# ── Round-2 BLOCKER invariants (permanent guards; fail pre-fix, pass post-fix) ──


def test_gate_only_never_resume_skips_and_bypasses_sentinel(tmp_path, monkeypatch):
    """B1: --gate-only NEVER honors a production .done resume-skip nor plants one.

    Pre-fix (round-1) the one-cell gate ran the FULL run_cell against the production
    OUT_ROOT with --force, so it (a) planted a .done the un-forced Phase-2 sweep would
    skip and (b) tripped the <4-contexts guard. The fix routes the gate through
    --gate-only + an isolated temp root. This pins the two gate-only invariants that
    can be checked without a GPU: gate-only does NOT resume-skip on an existing .done,
    and the parser exposes --gate-only.
    """
    import issue813_run_cell as rc

    # --gate-only is a real flag.
    assert any(a.dest == "gate_only" for a in rc.build_parser()._actions)

    # A pre-existing production .done must NOT short-circuit a gate-only measurement
    # (the round-1 bug read a stale .done as "done"); the skip guard excludes gate-only.
    reduced_dir = tmp_path / "reduced" / "marker" / "generic"
    reduced_dir.mkdir(parents=True)
    (reduced_dir / rc.CELL_DONE_SENTINEL).write_text(json.dumps({"behavior": "marker"}))

    class Args:
        behavior = "marker"
        substrate = "generic"
        out_root = tmp_path
        gpu_id = 0
        cpu_only = True
        upload = False
        force = False
        gate_only = True
        max_contexts = 1
        max_questions = 1
        metrics_out = None

    # Stop right after the skip guard so we never touch a GPU/model: assert the guard
    # did NOT return the skip sentinel (gate-only proceeds past it). We monkeypatch the
    # first expensive call (load_battery_instances) to raise a marker so we KNOW control
    # flowed past the resume-skip guard rather than returning early.
    class _PastSkip(RuntimeError):
        pass

    def _boom(*a, **k):
        raise _PastSkip("flowed past resume-skip into gate work")

    monkeypatch.setattr(rc, "load_battery_instances", _boom)
    with pytest.raises(_PastSkip):
        rc.run_cell(Args())


def test_gate_metrics_carry_gate_only_flag():
    """B1: the metrics JSON a gate-only run emits is tagged gate_only=True.

    A tag the .sh + downstream can trust to confirm the gate never wrote a reduced
    summary / .done (the gate-only early return builds this dict and returns before
    any reduced/summary/.done write). Pure-dict shape check, no model.
    """
    # The gate-only early-return metrics dict shape (mirrors run_cell's gate branch).
    # We assert the field exists by grepping the source contract rather than executing
    # the GPU path: the flag MUST be present in the emitted metrics.
    import inspect

    import issue813_run_cell as rc

    src = inspect.getsource(rc.run_cell)
    assert '"gate_only": True' in src, "gate-only metrics must carry gate_only=True (B1)"
    # and the gate branch returns BEFORE the <4-contexts fit guard + sentinel writes.
    assert src.index("if args.gate_only:") < src.index("len(kept) < 4"), (
        "gate-only must return before the <4-contexts fit guard (B1)"
    )
    assert src.index("if args.gate_only:") < src.index("os.replace(tmp_s, sentinel)"), (
        "gate-only must return before the .done sentinel write (B1)"
    )


def test_unreduced_upload_is_batched_not_per_file():
    """B3: the per-(context,question) unreduced .npz upload is BATCHED, never per-file.

    Pre-fix (round-1) _extract_pairs called HfApi().upload_file per pair (~23,850
    commits, blowing the 256/hr throttle). The fix buffers files and flushes ONE
    create_commit per BATCH_UPLOAD_CHUNK. Pins: (a) the batch helper exists and uses
    create_commit; (b) the per-pair loop does NOT call the single-file uploader; (c)
    the reduced summaries still go one-per-file (2 commits/cell, under throttle).
    """
    import inspect

    import issue813_run_cell as rc

    assert hasattr(rc, "_hf_batch_commit"), "the batch-commit helper must exist (B3)"
    assert "create_commit" in inspect.getsource(rc._hf_batch_commit), (
        "_hf_batch_commit must use HfApi.create_commit (one commit per chunk)"
    )
    pairs_src = inspect.getsource(rc._extract_pairs)
    # The per-pair loop must NOT call the single-file uploader (that WAS the storm).
    assert "_hf_upload_file(" not in pairs_src, (
        "the (context,question) loop must NOT call _hf_upload_file per pair (B3 storm)"
    )
    assert "_hf_batch_commit(" in pairs_src or "_flush_pending(" in pairs_src, (
        "the loop must buffer + batch-commit the unreduced .npz files (B3)"
    )
    assert isinstance(rc.BATCH_UPLOAD_CHUNK, int) and rc.BATCH_UPLOAD_CHUNK >= 1


def test_verdict_and_null_use_registered_delta_over_floor_space():
    """B2: the substrate_matters verdict + null band are in the REGISTERED Δ/floor space.

    Pre-fix (round-1) the verdict + null used raw Delta_med and the null band was only
    computed in raw-Δ space (registered band unrecoverable post-hoc). The fix computes
    the null per pseudo-arm in Δ/floor space (refit a pseudo-floor, divide), persists
    the per-resample Δ/floor null, and makes verdict + pairwise consume delta_over_floor.
    Pins the DV-space of each consumer via source inspection (the null needs a GPU-free
    real fit to run end-to-end; the smoke exercises that separately).
    """
    import inspect

    import issue813_analysis as an

    # substrate_swap_null returns the registered Δ/floor band + the full per-resample array.
    null_src = inspect.getsource(an.substrate_swap_null)
    assert "null_over_floor_p95" in null_src, "null must expose the Δ/floor p95 band (B2)"
    assert "null_delta_over_floor_diffs" in null_src, (
        "null must persist the per-resample Δ/floor array for post-hoc reconstruction (B2)"
    )
    assert '"null_space": "delta_over_floor"' in null_src

    # The per-pseudo-arm helper refits a FLOOR (not a bare raw-Δ median).
    pdof = inspect.getsource(an._pseudo_delta_over_floor)
    assert "make_refit_pair" in pdof and "_refit_pair_norm" in pdof, (
        "the pseudo-arm read must refit a per-arm floor via the shared harness (B2)"
    )

    # The verdict consumes delta_over_floor + the Δ/floor null band, NOT raw delta_med.
    main_src = inspect.getsource(an.main)
    assert 'observed_by_sub[s].get("delta_over_floor")' in main_src, (
        "the verdict must read delta_over_floor, not delta_med (B2)"
    )
    assert "null_over_floor_p95" in main_src, "the verdict must gate on the Δ/floor null (B2)"
    assert '"dv_space": "delta_over_floor"' in main_src

    # The pairwise diff is in Δ/floor space too.
    pw_src = inspect.getsource(an.pairwise_substrate_diff_cis)
    assert 'observed_by_sub[a].get("delta_over_floor")' in pw_src, (
        "pairwise diff must be in Δ/floor space (B2)"
    )
