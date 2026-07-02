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
import os
import sys
from pathlib import Path

import numpy as np
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

    NOTE (IO patch): the cell body moved from run_cell into _run_cell_body (run_cell
    is now the sentinel + PID-lock wrapper), so the source contract is asserted there.
    """
    # The gate-only early-return metrics dict shape (mirrors the body's gate branch).
    # We assert the field exists by grepping the source contract rather than executing
    # the GPU path: the flag MUST be present in the emitted metrics.
    import inspect

    import issue813_run_cell as rc

    src = inspect.getsource(rc._run_cell_body)
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
    # NOTE (D1 refactor): the verdict logic moved from inline main() into the pure
    # `decide_substrate_matters` reducer, so the DV-space contract is now asserted there
    # (main() routes through it — checked separately in
    # test_main_verdict_uses_conjunction_reducer). The B2 invariant is unchanged: the
    # verdict reads delta_over_floor and gates on the Δ/floor null p95, never raw delta_med.
    main_src = inspect.getsource(an.main)
    verdict_src = inspect.getsource(an.decide_substrate_matters)
    assert 'dofs = {s: observed_by_sub[s].get("delta_over_floor")' in main_src, (
        "the verdict input must be delta_over_floor, not delta_med (B2)"
    )
    assert "v for s, v in dofs.items()" in verdict_src, (
        "the reducer consumes the delta_over_floor dofs dict (B2)"
    )
    assert "null_over_floor_p95" in verdict_src, (
        "the verdict must gate on the Δ/floor null p95 (B2)"
    )
    assert '"dv_space": "delta_over_floor"' in verdict_src

    # The pairwise diff is in Δ/floor space too.
    pw_src = inspect.getsource(an.pairwise_substrate_diff_cis)
    assert 'observed_by_sub[a].get("delta_over_floor")' in pw_src, (
        "pairwise diff must be in Δ/floor space (B2)"
    )


# ── Round-3 BLOCKER invariant (D1): the substrate_matters CONJUNCTION ──
#
# Pins that the shipped verdict enforces plan §3's CONJUNCTION — (max_diff > null_p95)
# AND (a driving-pair pairwise CI excludes 0) — NOT the single null-band gate the
# round-1/2 verdict shipped (BLOCKER i813-pairwise-ci-conjunct-missing). The reducer
# `decide_substrate_matters` is a pure function, so the conjunction is exercised with
# synthetic inputs (no GPU/fit). The construct-a-false-positive case is the exact
# regression: a case that WOULD have read substrate_matters=True under the old
# null-band-only rule must NOT read True now.


def _null(p95: float) -> dict:
    """A minimal substrate-swap-null dict carrying just the Δ/floor p95 band."""
    return {"null_over_floor_p95": p95}


def _pair(a: str, b: str, *, excludes_zero: bool | None, ci_lo=None, ci_hi=None) -> dict:
    """A minimal pairwise-diff record for the reducer (the fields it reads)."""
    return {
        "pair": f"{a}_vs_{b}",
        "dv_space": "delta_over_floor",
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_excludes_zero": excludes_zero,
    }


def test_pairwise_records_carry_ci_fields():
    """D1: every pairwise-diff record exposes ci_lo + ci_hi + ci_excludes_zero fields."""
    import inspect

    import issue813_analysis as an

    # The record shape (source contract — the CI is a real bootstrap, computed when both
    # observed reads exist). The three CI keys must be present on every record.
    src = inspect.getsource(an.pairwise_substrate_diff_cis)
    for key in ('"ci_lo"', '"ci_hi"', '"ci_excludes_zero"'):
        assert key in src, f"pairwise record must carry {key} (D1)"
    # The CI is a genuine family-clustered bootstrap, not a delegated placeholder.
    ci_src = inspect.getsource(an.pairwise_diff_ci)
    assert "_pseudo_delta_over_floor" in ci_src, (
        "pairwise CI must refit Δ/floor per resample via the shared harness (D1)"
    )
    assert "np.percentile(arr, 2.5)" in ci_src and "np.percentile(arr, 97.5)" in ci_src, (
        "pairwise CI must be a 95% percentile CI on the signed Δ/floor difference (D1)"
    )
    assert "ci_lo > 0.0 or ci_hi < 0.0" in ci_src, (
        "ci_excludes_zero must be True iff the whole interval is on one side of 0 (D1)"
    )


def test_verdict_true_only_when_BOTH_conjuncts_fire():
    """D1: substrate_matters=True iff (max_diff > null_p95) AND (driving-pair CI excludes 0)."""
    import issue813_analysis as an

    dofs = {"generic": 0.10, "elicit": 0.90, "mix": 0.50}  # max_diff = 0.80 (elicit-generic)
    null = {s: _null(0.30) for s in dofs}  # 0.80 > 0.30 → null-band conjunct fires
    # Driving pair is generic_vs_elicit (the {min, max} substrates); its CI excludes 0.
    pairwise = [
        _pair("generic", "elicit", excludes_zero=True, ci_lo=0.4, ci_hi=1.2),
        _pair("generic", "mix", excludes_zero=False, ci_lo=-0.1, ci_hi=0.9),
        _pair("elicit", "mix", excludes_zero=False, ci_lo=-0.2, ci_hi=0.9),
    ]
    v = an.decide_substrate_matters(dofs, null, pairwise)
    assert v["null_band_conjunct"] is True
    assert v["pairwise_ci_conjunct"] is True
    assert v["substrate_matters"] is True, v


def test_verdict_not_true_when_null_band_fires_but_all_cis_include_zero():
    """D1 REGRESSION: max_diff > null_p95 but ALL pairwise CIs INCLUDE 0 → NOT True.

    This is the exact false-positive the old null-band-only verdict would have emitted:
    the max-vs-min Δ/floor difference clears the substrate-swap null p95 (the old gate),
    yet no pairwise CI excludes 0 (the missing second conjunct), so under plan §3 the
    verdict is NOT 'substrate matters'. Under the old rule this read True; it must not now.
    """
    import issue813_analysis as an

    dofs = {"generic": 0.10, "elicit": 0.90, "mix": 0.50}  # max_diff = 0.80
    null = {s: _null(0.30) for s in dofs}  # 0.80 > 0.30 → old rule would say True
    # The driving pair's CI spans 0 (a wide, noise-dominated CI) → conjunct (ii) FAILS.
    pairwise = [
        _pair("generic", "elicit", excludes_zero=False, ci_lo=-0.2, ci_hi=1.8),
        _pair("generic", "mix", excludes_zero=False, ci_lo=-0.3, ci_hi=1.3),
        _pair("elicit", "mix", excludes_zero=False, ci_lo=-0.4, ci_hi=1.2),
    ]
    v = an.decide_substrate_matters(dofs, null, pairwise)
    assert v["null_band_conjunct"] is True, v
    assert v["pairwise_ci_conjunct"] is False, v
    # Exactly one conjunct fired → NOT True (AMBIGUOUS under the conjunction rule).
    assert v["substrate_matters"] is not True, (
        "the old null-band-only gate would emit True here; the conjunction must NOT"
    )
    assert v["substrate_matters"] is None, v


def test_verdict_false_when_both_conjuncts_fail_H1():
    """D1: max within null band AND all CIs include 0 → substrate-agnostic (H1) → False."""
    import issue813_analysis as an

    dofs = {"generic": 0.50, "elicit": 0.55, "mix": 0.52}  # max_diff = 0.05
    null = {s: _null(0.30) for s in dofs}  # 0.05 < 0.30 → null-band conjunct FAILS
    pairwise = [
        _pair("generic", "elicit", excludes_zero=False, ci_lo=-0.2, ci_hi=0.3),
        _pair("generic", "mix", excludes_zero=False, ci_lo=-0.2, ci_hi=0.2),
        _pair("elicit", "mix", excludes_zero=False, ci_lo=-0.2, ci_hi=0.2),
    ]
    v = an.decide_substrate_matters(dofs, null, pairwise)
    assert v["null_band_conjunct"] is False
    assert v["pairwise_ci_conjunct"] is False
    assert v["substrate_matters"] is False, v


def test_verdict_ambiguous_when_ci_excludes_but_within_null_band():
    """D1: CI excludes 0 (ii fires) but max within null band (i fails) → AMBIGUOUS (None)."""
    import issue813_analysis as an

    dofs = {"generic": 0.50, "elicit": 0.58, "mix": 0.54}  # max_diff = 0.08
    null = {s: _null(0.30) for s in dofs}  # 0.08 < 0.30 → null-band conjunct FAILS
    pairwise = [
        # A tight CI that excludes 0 on the driving pair (small but distinguishable diff).
        _pair("generic", "elicit", excludes_zero=True, ci_lo=0.02, ci_hi=0.14),
        _pair("generic", "mix", excludes_zero=False, ci_lo=-0.02, ci_hi=0.10),
        _pair("elicit", "mix", excludes_zero=False, ci_lo=-0.02, ci_hi=0.10),
    ]
    v = an.decide_substrate_matters(dofs, null, pairwise)
    assert v["null_band_conjunct"] is False
    assert v["pairwise_ci_conjunct"] is True
    assert v["substrate_matters"] is None, v  # exactly one conjunct fires → AMBIGUOUS


def test_verdict_ambiguous_when_driving_pair_ci_uncomputed():
    """D1: null-band fires but the driving pair's CI is undecidable → AMBIGUOUS (None).

    A driving-pair pairwise record whose CI could not be bootstrapped (ci_excludes_zero
    None) leaves conjunct (ii) undecidable — the verdict must be None, never silently
    True off the null-band conjunct alone (the round-1/2 bug).
    """
    import issue813_analysis as an

    dofs = {"generic": 0.10, "elicit": 0.90, "mix": 0.50}  # max_diff = 0.80
    null = {s: _null(0.30) for s in dofs}
    pairwise = [
        _pair("generic", "elicit", excludes_zero=None),  # driving pair CI uncomputed
        _pair("generic", "mix", excludes_zero=False),
        _pair("elicit", "mix", excludes_zero=False),
    ]
    v = an.decide_substrate_matters(dofs, null, pairwise)
    assert v["null_band_conjunct"] is True
    assert v["pairwise_ci_conjunct"] is None
    assert v["substrate_matters"] is None, v


def test_verdict_only_driving_pair_ci_counts():
    """D1: a NON-driving pair CI excluding 0 does NOT flip the max-vs-min verdict.

    Conjunct (ii) requires the CI-excluding pair to be the DRIVING pair (the {min,max}
    substrates whose difference IS max_diff). A CI excluding 0 on a non-driving pair while
    the driving pair's CI includes 0 must NOT make substrate_matters True.
    """
    import issue813_analysis as an

    dofs = {"generic": 0.10, "elicit": 0.90, "mix": 0.50}  # driving pair = generic_vs_elicit
    null = {s: _null(0.30) for s in dofs}  # null-band conjunct fires (0.80 > 0.30)
    pairwise = [
        # Driving pair CI INCLUDES 0 → conjunct (ii) fails on the pair that matters.
        _pair("generic", "elicit", excludes_zero=False, ci_lo=-0.1, ci_hi=1.7),
        # A NON-driving pair CI excludes 0 — must be ignored for the max-vs-min verdict.
        _pair("generic", "mix", excludes_zero=True, ci_lo=0.05, ci_hi=0.75),
        _pair("elicit", "mix", excludes_zero=False, ci_lo=-0.2, ci_hi=0.9),
    ]
    v = an.decide_substrate_matters(dofs, null, pairwise)
    assert v["pairwise_ci_conjunct"] is False, (
        "only the DRIVING pair's CI counts for the max-vs-min verdict (D1)"
    )
    assert v["substrate_matters"] is not True, v


def test_main_verdict_uses_conjunction_reducer():
    """D1: main() routes the verdict through decide_substrate_matters (both conjuncts).

    Source contract: the shipped verdict is the pure conjunction reducer, NOT an inline
    `(max_diff > null_x)` single-gate. Pins the round-1/2 single-gate expression is gone.
    """
    import inspect

    import issue813_analysis as an

    main_src = inspect.getsource(an.main)
    assert "decide_substrate_matters(dofs, null_by_sub, pairwise)" in main_src, (
        "the verdict must be the conjunction reducer, not an inline null-band gate (D1)"
    )
    # The round-1/2 single-conjunct expression must be gone from main().
    assert "(max_diff > null_x) if null_x else None" not in main_src, (
        "the single null-band-only verdict gate must be removed (D1)"
    )
    # The pairwise-diff call passes the CI-bootstrap inputs (reduced_root + r_hat).
    assert "pairwise_substrate_diff_cis(" in main_src
    assert "args.reduced_root" in main_src and "r_hat" in main_src


# ── Round-4 crash-fix invariants: per-behavior parity floor + marker confirmation ──
#
# The round-3 apply-parity phase HALTed marker/default on the 7-MODULE #667 floor
# (PARITY_MIN_WRITE_RATIO=0.01) even though the 4-module/alpha=64/low-LR marker
# adapter writes ~0.00903 CORRECTLY (a 10% shortfall, NOT a sqrt(r) gauge drift --
# a true alpha/sqrt(r)-vs-alpha/r error at r=32 is a 5.66x discrepancy reading
# ~0.0016 or ~0.05). The fix makes the diagonal-write floor per-behavior (default
# preserves #667 usage; marker->0.004) and adds a marker-only >=1 nat teacher-forced
# Delta log P(marker) behavioral confirmation. These pin: the default is unchanged,
# the .sh maps marker->0.004, and the numeric floor + behavioral gates HALT/PASS at
# the documented thresholds.

_MARKER_MEASURED_RATIO = 0.00903  # the exact correct-stack marker write from #813 r3
_MARKER_FLOOR = 0.004  # the #813 marker floor (separates 0.009 correct from 0.0016 wrong-gauge)


def test_parity_min_write_ratio_default_preserves_667_usage():
    """Part 1: min_write_ratio defaults to the 7-module #667 floor on all three functions.

    #667's own callers never pass min_write_ratio, so the DEFAULT must equal
    PARITY_MIN_WRITE_RATIO (0.01) — the fix must not change #667 behavior. Pins the
    signature default of the numeric probe, the CPU/GPU wrapper, and the subprocess.
    """
    import inspect

    import issue667_dispatch as d

    assert d.PARITY_MIN_WRITE_RATIO == 0.01
    for fn in (d._numeric_rslora_parity, d._rslora_parity_probe, d._run_parity_probe_subprocess):
        params = inspect.signature(fn).parameters
        assert "min_write_ratio" in params, f"{fn.__name__} must expose min_write_ratio"
        assert params["min_write_ratio"].default == d.PARITY_MIN_WRITE_RATIO, (
            f"{fn.__name__}.min_write_ratio default must be the #667 floor (unchanged usage)"
        )


def test_sh_maps_marker_to_lower_floor_and_threads_it():
    """Part 1: issue813_dispatch.sh passes marker→0.004 (others→#667 default) to the probe.

    Source contract on the .sh apply-parity heredoc: it builds a per-behavior floor
    map with marker→0.004, resolves non-marker behaviors to i667.PARITY_MIN_WRITE_RATIO,
    and threads min_write_ratio into _rslora_parity_probe. Pins the round-3 bare call
    (no floor arg) is gone.
    """
    sh = (REPO_ROOT / "scripts" / "issue813_dispatch.sh").read_text()
    assert '"marker": 0.004' in sh, "the .sh must map marker→0.004 (the #813 4-module floor)"
    assert "i667.PARITY_MIN_WRITE_RATIO" in sh, "non-marker behaviors fall back to the #667 floor"
    assert "min_write_ratio=min_write_ratio" in sh, (
        "the .sh must thread the per-behavior floor into _rslora_parity_probe"
    )
    # The round-3 bare call (no floor kwarg) must be gone.
    assert "i667._rslora_parity_probe(behavior, cpu_only=cpu_only)\n" not in sh, (
        "the round-3 bare _rslora_parity_probe call (no floor) must be replaced"
    )


def _install_synthetic_parity_reads(monkeypatch, *, ratio: float, marker_delta: float | None):
    """Stub the extract/model helpers so _numeric_rslora_parity runs on synthetic reads.

    Produces a diagonal write with the requested ‖Δv‖/‖v0‖ = ``ratio`` and a self-gate
    of exactly 1, without any GPU / network / 7B load. When ``marker_delta`` is not
    None the marker behavioral confirmation is stubbed to return it (so the two gates
    are exercised in isolation). Returns nothing; monkeypatch handles teardown.
    """
    import issue667_extract as ex
    import torch

    from explore_persona_space.analysis.issue667 import gate_chain
    from explore_persona_space.experiments import i537_contexts

    monkeypatch.setattr(ex, "stage_adapter_local", lambda *a, **k: Path("/tmp/fake_adapter"))
    monkeypatch.setattr(
        ex,
        "assert_adapter_gauge",
        lambda *a, **k: {"r": 32, "lora_alpha": 64, "use_rslora": True, "target_modules": []},
    )
    monkeypatch.setattr(ex, "stage_inputs", lambda: (Path("/tmp/s.json"), Path("/tmp/d.json")))
    monkeypatch.setattr(ex, "_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr(ex, "load_base_and_trained", lambda *a, **k: (object(), object(), object()))
    monkeypatch.setattr(ex, "load_eval_probes", lambda *a, **k: ["q1"])
    monkeypatch.setattr(
        ex, "build_messages_for", lambda *a, **k: [{"role": "user", "content": "q"}]
    )
    monkeypatch.setattr(ex, "_greedy_response", lambda *a, **k: "a base response")

    # v0 unit vector, vp = v0 * (1 + ratio) along the same axis → ‖Δv‖/‖v0‖ == ratio.
    # Key the returned acts dict on PRIMARY_LAYER (the same layer the function reads).
    from explore_persona_space.analysis.issue667 import PRIMARY_LAYER

    v0 = np.zeros(8, dtype=np.float64)
    v0[0] = 1.0
    vp = v0 * (1.0 + ratio)
    monkeypatch.setattr(ex, "_mean_resp_acts", lambda *a, **k: {PRIMARY_LAYER: (v0, vp)})

    # realized_gate(v0, vp, v0, vp) == 1 (the self-gate is exactly 1 by construction).
    monkeypatch.setattr(gate_chain, "realized_gate", lambda *a, **k: (1.0, None))
    monkeypatch.setattr(i537_contexts, "load_registry", lambda *a, **k: {"default": object()})
    monkeypatch.setattr(i537_contexts, "load_icl_demos", lambda *a, **k: [])

    # AutoTokenizer.from_pretrained → a stub tokenizer (never touches HF).
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: object())
    )

    # The marker behavioral confirmation is exercised in its own test; here we stub it
    # to the requested Δ (or leave it unpatched — it is only called for behavior=="marker").
    if marker_delta is not None:
        import issue667_dispatch as d

        monkeypatch.setattr(d, "_marker_behavioral_confirmation", lambda *a, **k: marker_delta)


def test_numeric_floor_halts_below_and_passes_at_or_above(monkeypatch):
    """Part 1 REGRESSION: the marker's 0.00903 write HALTs at floor 0.01, PASSES at 0.004.

    This is the exact round-3 crash + its fix: with the CORRECTLY-applied marker write
    ratio ~0.00903, the OLD 7-module floor (0.01) HALTs (the round-3 bug), while the NEW
    per-behavior marker floor (0.004) PASSES. Fails pre-fix (min_write_ratio did not exist
    / the floor was hardcoded 0.01), passes post-fix. marker_delta stubbed high so only the
    numeric floor decides the outcome.
    """
    import issue667_dispatch as d

    # OLD 7-module floor: the correct 0.00903 marker write is BELOW it → HALT.
    _install_synthetic_parity_reads(monkeypatch, ratio=_MARKER_MEASURED_RATIO, marker_delta=5.0)
    with pytest.raises(RuntimeError, match=r"rsLoRA NUMERIC parity FAILED"):
        d._numeric_rslora_parity("marker", min_write_ratio=0.01)

    # NEW marker floor 0.004: the same 0.00903 write is ABOVE it → PASS (no raise).
    _install_synthetic_parity_reads(monkeypatch, ratio=_MARKER_MEASURED_RATIO, marker_delta=5.0)
    res = d._numeric_rslora_parity("marker", min_write_ratio=_MARKER_FLOOR)
    assert res["write_ratio"] == pytest.approx(_MARKER_MEASURED_RATIO, abs=1e-4)
    assert res["min_write_ratio"] == _MARKER_FLOOR
    assert res["marker_delta_logp_nats"] == 5.0


def test_wrong_gauge_write_still_halts_at_marker_floor(monkeypatch):
    """Part 1: a wrong-gauge marker write (~0.0016) HALTs even at the LOWER 0.004 floor.

    The marker floor 0.004 must still SEPARATE a correct write (0.009, ~2.25x above) from
    a wrong-gauge one (0.0016 ~= 0.00903/5.66, the alpha/sqrt(r)-vs-alpha/r discrepancy at
    r=32, ~2.5x below) -- lowering the floor must not blind it to a real gauge error.
    """
    import issue667_dispatch as d

    wrong_gauge_ratio = _MARKER_MEASURED_RATIO / (32**0.5)  # ~0.0016
    _install_synthetic_parity_reads(monkeypatch, ratio=wrong_gauge_ratio, marker_delta=5.0)
    with pytest.raises(RuntimeError, match=r"rsLoRA NUMERIC parity FAILED"):
        d._numeric_rslora_parity("marker", min_write_ratio=_MARKER_FLOOR)


def test_marker_behavioral_read_is_diagnostic_never_halts(monkeypatch, caplog):
    """Part 2 (launch-3 demotion): the marker behavioral read WARNs below the reference
    band but NEVER raises — the value is persisted for the analyzer either way.

    Rationale (see the demotion comment at the read site in issue667_dispatch.py): the
    round-2 reconciler (D4, binding) ruled the NUMERIC gauge probe the sufficient
    apply-parity HALT (its cross-run em reference is exact: write_ratio 0.1729 on this
    stack == #667's committed 0.1729). The behavioral threshold proved ungroundable
    without #537's frozen-R eval rig — this probe teacher-forces FRESH greedy R, so its
    slot conditioning differs from #537's committed diagonal band (5-12 nat) and a
    verified-applied adapter measured 0.75-0.85 nat (two false-HALTs at 1.0- and
    2.5-nat bars). The numeric floor gate (previous test) remains the HALT.
    """
    import logging

    import issue667_dispatch as d

    # Below-band read (the launch-3 value): WARNs, does NOT raise, value persisted.
    _install_synthetic_parity_reads(monkeypatch, ratio=0.02, marker_delta=0.7516)
    with caplog.at_level(logging.WARNING):
        res = d._numeric_rslora_parity("marker", min_write_ratio=_MARKER_FLOOR)
    assert res["marker_delta_logp_nats"] == 0.7516
    assert any("BELOW reference band" in r.message for r in caplog.records), (
        "a below-band behavioral read must emit the diagnostic WARNING"
    )

    # In-band read: no warning, value persisted.
    caplog.clear()
    _install_synthetic_parity_reads(monkeypatch, ratio=0.02, marker_delta=6.0)
    with caplog.at_level(logging.WARNING):
        res = d._numeric_rslora_parity("marker", min_write_ratio=_MARKER_FLOOR)
    assert res["marker_delta_logp_nats"] == 6.0
    assert not any("BELOW reference band" in r.message for r in caplog.records)


def test_non_marker_behavior_skips_the_behavioral_gate(monkeypatch):
    """Part 2: a non-marker behavior runs the numeric floor gate ONLY (no marker Δ read).

    _marker_behavioral_confirmation is marker-only; a non-marker cell (em/sycophancy/fact)
    must not carry a marker_delta_logp_nats field nor invoke the confirmation.
    """
    import issue667_dispatch as d

    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        return 0.0

    _install_synthetic_parity_reads(monkeypatch, ratio=0.05, marker_delta=None)
    monkeypatch.setattr(d, "_marker_behavioral_confirmation", _boom)
    res = d._numeric_rslora_parity("sycophancy", min_write_ratio=0.01)
    assert called["n"] == 0, "the behavioral confirmation must not run for non-marker behaviors"
    assert "marker_delta_logp_nats" not in res


def test_marker_confirmation_asserts_marker_token_id_83399(monkeypatch):
    """Part 2: _marker_behavioral_confirmation fails loud on a WRONG marker token.

    The in-process id-83399 assert (` ※`, NOT bare `※` id 63680) must fire when the stub
    tokenizer encodes the marker to anything else — a wrong-marker trainer path is the
    #537 all-adapters-no-op incident this assert exists to catch.
    """
    import issue667_dispatch as d

    class _WrongTok:
        eos_token_id = 151645

        def encode(self, text, add_special_tokens=False):
            return [63680]  # bare `※` — the WRONG token

    with pytest.raises(AssertionError, match=r"83399"):
        d._marker_behavioral_confirmation(
            object(), object(), _WrongTok(), {}, [], "default", __import__("torch").device("cpu")
        )


def test_marker_behavioral_threshold_is_diagonal_grounded():
    """Round-5: the behavioral bar is 2.5 nat, grounded on #537's 5-12 nat DIAGONAL band.

    Pins the round-5 threshold fix: the bar separates a correct diagonal install (≥5 nat)
    from a wrong-gauge (~1-2 nat) / no-op (~0) one. The OLD 1.0 bar sat below the round-4
    OFF-DIAGONAL battery read 0.8547 nat only because that read was measured on the wrong
    geometry; the diagonal-grounded 2.5 bar is above both wrong-gauge reads (~2 nat max)
    and comfortably below the 5-nat diagonal floor.
    """
    import issue667_dispatch as d

    assert d.MARKER_BEHAVIORAL_MIN_DELTA_NATS == 2.5
    # separates in-band (≥5) from wrong-gauge (≤2) with margin each way
    assert d.MARKER_BEHAVIORAL_MIN_DELTA_NATS < 5.0  # below the #537 diagonal floor
    assert d.MARKER_BEHAVIORAL_MIN_DELTA_NATS > 2.0  # above the wrong-gauge ceiling


def test_marker_behavioral_confirmation_reads_the_diagonal_context_and_logs(monkeypatch, caplog):
    """Round-5: the read is on the DIAGONAL (source) context and logs the fix-engaged signal.

    Reproduces #537's committed diagonal manipulation check: every probe is rendered under
    the SAME `source` context (the adapter's own training context), NOT an off-diagonal
    battery. Stubs the model/tokenizer helpers so it runs on CPU with no GPU/network/7B
    load; asserts (a) build_messages_for is called with `source` for EVERY probe (diagonal),
    (b) compute_marker_slot_stats returns the mean trained-base delta, and (c) the
    fix-engaged log line names the DIAGONAL context + the delta + the 2.5 threshold.
    """
    import logging

    import issue667_dispatch as d
    import issue667_extract as ex

    # Record the context id each probe is rendered under (must be the diagonal `source`).
    seen_cids: list[str] = []

    def _build(registry, demos, cid, behavior, question):
        seen_cids.append(cid)
        return [{"role": "user", "content": question}]

    class _Tok:
        eos_token_id = 151645

        def encode(self, text, add_special_tokens=False):
            if text == d._MARKER_TEXT:
                return [d._MARKER_TOKEN_ID]  # ` ※` id 83399
            if text == "<|im_end|>":
                return [151645]
            return [1, 2, 3]

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "PROMPT:" + msgs[0]["content"]

    monkeypatch.setattr(ex, "load_eval_probes", lambda *a, **k: ["q1", "q2", "q3", "q4", "q5"])
    monkeypatch.setattr(ex, "build_messages_for", _build)
    monkeypatch.setattr(ex, "_greedy_response", lambda *a, **k: "a base response")

    # trained slot logp 4.0 higher than base per context → mean Δ = 4.0 nat.
    from explore_persona_space.eval import marker_logprob as ml

    def _slot_stats(model, tok, contexts, marker_text, **kw):
        base = model == "BASE"
        return [{"logp": (0.0 if base else 4.0)} for _ in contexts]

    monkeypatch.setattr(ml, "compute_marker_slot_stats", _slot_stats)

    with caplog.at_level(logging.INFO, logger=d.logger.name):
        delta = d._marker_behavioral_confirmation(
            "BASE", "TRAINED", _Tok(), {}, [], "default", __import__("torch").device("cpu")
        )

    # (a) DIAGONAL: every rendered probe used the `source` context, and n_questions defaults
    # to 4, so exactly 4 probes were read on the diagonal.
    assert seen_cids == ["default", "default", "default", "default"], seen_cids
    # (b) mean trained-base delta.
    assert delta == pytest.approx(4.0)
    # (c) fix-engaged log signal names the DIAGONAL context + delta + threshold.
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "DIAGONAL" in msg and "context=default" in msg, msg
    assert "4.0000 nat" in msg, msg


# ── IO-bottleneck patch invariants (uncompressed saves, resume skip-set, PID lock) ──
#
# The live-sweep IO patch: (1) the per-row unreduced save is UNCOMPRESSED np.savez
# (deflate was ~65% of per-row wall — 103.8s vs 1.2s on a 933MB row — at a 1.29x
# ratio; the small cell-end reduced saves stay compressed); (2) resume skip-set —
# rows already on HF still forward+reduce but skip the save/enqueue; (3) a per-cell
# PID lockfile makes a duplicate exec of a LIVE cell WAIT for the owner, then exit 0
# iff the owner wrote .done / exit 1 otherwise (an instant exit-0 would let the
# dispatcher's p.wait() advance to the fit phase mid-extraction); (4) r_lookup.json
# + accum_ckpt.npz resume persistence. All inert on a fresh cell launch.


def _cell_args(tmp_path, **overrides):
    """A minimal run_cell Args namespace for lock/skip tests (CPU, no upload)."""

    class Args:
        behavior = "marker"
        substrate = "generic"
        out_root = tmp_path
        gpu_id = 0
        cpu_only = True
        upload = False
        force = False
        gate_only = False
        max_contexts = 1
        max_questions = 1
        metrics_out = None

    for k, v in overrides.items():
        setattr(Args, k, v)
    return Args()


def _fake_caps(t_len: int = 4, width: int = 8):
    """Synthetic _capture_full_and_reduce output (tiny width; layer axis real)."""
    import issue813_run_cell as rc

    full = np.zeros((t_len, rc.N_LAYERS, width), dtype=np.float16)
    red = np.ones((rc.N_LAYERS, width), dtype=np.float32)
    return {
        "full_base": full,
        "full_trained": full,
        "c_C_base": red,
        "c_C_trained": red,
        "v_A_base": red,
        "v_A_trained": red,
        "prompt_len": 2,
        "full_len": t_len,
    }


def test_unreduced_save_is_uncompressed_reduced_stay_compressed():
    """IO-1: per-row unreduced save = np.savez; the 2 cell-end reduced saves stay compressed."""
    import inspect

    import issue813_run_cell as rc

    pairs_src = inspect.getsource(rc._extract_pairs)
    assert "np.savez(" in pairs_src, "per-row unreduced save must be UNCOMPRESSED np.savez"
    assert "np.savez_compressed(" not in pairs_src, (
        "the per-row loop must NOT compress (65% of per-row wall at a 1.29x ratio)"
    )
    body_src = inspect.getsource(rc._run_cell_body)
    assert body_src.count("np.savez_compressed(") == 2, (
        "summary.npz + per_question_L14.npz (small) must STAY compressed"
    )


def test_batch_upload_chunk_halved_for_uncompressed_files():
    """IO-5: BATCH_UPLOAD_CHUNK 100 → 50 (files ~1.29x bigger; keeps tmp footprint similar)."""
    import issue813_run_cell as rc

    assert rc.BATCH_UPLOAD_CHUNK == 50


def test_duplicate_waits_for_live_owner_then_exits_zero_on_done(tmp_path, monkeypatch, caplog):
    """IO-6 (amended): a duplicate BLOCKS on a live owner; exits 0 only after owner + .done.

    The pre-amendment instant exit-0 let the dispatcher's p.wait() read success
    for all wave-2 duplicates and advance to the fit phase while the manually
    pre-launched owner was still extracting. Simulates the owner finishing after
    3 liveness polls (writes .done, releases its lock, exits): the duplicate must
    have polled multiple times (blocked), heartbeat-logged, never stolen the
    live owner's lock, and exited 0 with the wait recorded.
    """
    import logging

    import issue813_run_cell as rc

    reduced_dir = tmp_path / "reduced" / "marker" / "generic"
    reduced_dir.mkdir(parents=True)
    lock = reduced_dir / rc.RUNNING_LOCK_NAME
    lock.write_text("424242")
    monkeypatch.setattr(rc, "DUP_WAIT_POLL_S", 0.01)
    monkeypatch.setattr(rc, "DUP_WAIT_HEARTBEAT_S", 0.0)  # heartbeat on every poll

    polls = {"n": 0}

    def _alive(pid):
        assert pid == 424242
        polls["n"] += 1
        # The lock must not have been stolen while the owner is alive.
        assert lock.read_text() == "424242"
        if polls["n"] >= 4:
            # The owner finishes: writes .done, releases its lock, exits.
            (reduced_dir / rc.CELL_DONE_SENTINEL).write_text("{}")
            lock.unlink(missing_ok=True)
            return False
        return True

    monkeypatch.setattr(rc, "_pid_alive", _alive)
    with caplog.at_level(logging.INFO, logger=rc.logger.name):
        out = rc.run_cell(_cell_args(tmp_path))
    assert out.get("skipped") is True and out.get("duplicate_of_pid") == 424242, out
    assert out.get("waited_min") is not None, out
    assert polls["n"] >= 4, polls  # it actually blocked across multiple polls
    msgs = "\n".join(r.getMessage() for r in caplog.records)
    assert "waiting for it to finish" in msgs
    assert "still waiting on cell owner pid 424242" in msgs  # heartbeat line fired


def test_duplicate_fails_loud_when_owner_dies_without_done(tmp_path, monkeypatch):
    """IO-6 (amended): owner death WITHOUT .done → RuntimeError (exit 1), lock untouched.

    A dead-mid-cell owner means the cell is incomplete; the duplicate must fail
    loud so the dispatcher surfaces it rather than fitting on a partial cell.
    """
    import issue813_run_cell as rc

    reduced_dir = tmp_path / "reduced" / "marker" / "generic"
    reduced_dir.mkdir(parents=True)
    lock = reduced_dir / rc.RUNNING_LOCK_NAME
    lock.write_text("424242")
    monkeypatch.setattr(rc, "DUP_WAIT_POLL_S", 0.01)

    polls = {"n": 0}

    def _alive(pid):
        polls["n"] += 1
        return polls["n"] < 3  # owner dies after 2 liveness reads, never writing .done

    monkeypatch.setattr(rc, "_pid_alive", _alive)
    with pytest.raises(RuntimeError, match="without writing"):
        rc.run_cell(_cell_args(tmp_path))
    # The duplicate never touches the dead owner's lock (no takeover, no cleanup).
    assert lock.read_text() == "424242"


def test_stale_lock_is_taken_over_and_removed_in_finally(tmp_path, monkeypatch):
    """IO-6: a DEAD owner's lock is taken over; the finally releases it even on a crash."""
    import issue813_run_cell as rc

    reduced_dir = tmp_path / "reduced" / "marker" / "generic"
    reduced_dir.mkdir(parents=True)
    (reduced_dir / rc.RUNNING_LOCK_NAME).write_text("424242")
    monkeypatch.setattr(rc, "_pid_alive", lambda pid: False)  # the recorded pid is dead

    class _Boom(RuntimeError):
        pass

    def _body(args, reduced_dir_):
        # The stale lock was overwritten with OUR pid before the body ran.
        assert (reduced_dir_ / rc.RUNNING_LOCK_NAME).read_text() == str(os.getpid())
        raise _Boom("body crashed")

    monkeypatch.setattr(rc, "_run_cell_body", _body)
    with pytest.raises(_Boom):
        rc.run_cell(_cell_args(tmp_path))
    # finally released the lock despite the crash.
    assert not (reduced_dir / rc.RUNNING_LOCK_NAME).exists()


def test_gate_only_takes_no_lock(tmp_path, monkeypatch):
    """IO-6: gate-only bypasses the PID lock entirely (it writes no production state)."""
    import issue813_run_cell as rc

    class _PastLock(RuntimeError):
        pass

    def _boom(*a, **k):
        raise _PastLock("reached the body")

    monkeypatch.setattr(rc, "load_battery_instances", _boom)
    with pytest.raises(_PastLock):
        rc.run_cell(_cell_args(tmp_path, gate_only=True))
    reduced_dir = tmp_path / "reduced" / "marker" / "generic"
    assert not (reduced_dir / rc.RUNNING_LOCK_NAME).exists()


def test_skip_set_rows_still_reduce_but_do_not_save_or_enqueue(tmp_path, monkeypatch):
    """IO-3: a skip-set row still forwards + reduces, but writes no npz (and no enqueue)."""
    import issue813_run_cell as rc

    monkeypatch.setattr(rc, "_capture_full_and_reduce", lambda *a, **k: _fake_caps())
    contexts = [{"id": "ctx0", "family": "fam", "system_prompt": None, "prefix_messages": []}]
    questions = ["q a", "q b"]
    r_lookup = {(0, 0): "resp a", (0, 1): "resp b"}
    tmp_dir = tmp_path / "unreduced_tmp" / "marker" / "generic"

    acc = rc._extract_pairs(
        object(),
        object(),
        object(),
        contexts,
        questions,
        r_lookup,
        behavior="marker",
        substrate="generic",
        tmp_dir=tmp_dir,
        device="cpu",
        max_new=8,
        upload=False,
        skip_save_keys=frozenset({"ctx0__q0"}),
    )
    # BOTH rows reduced (accumulators complete)...
    assert acc["n_cells_done"] == 2
    assert len(acc["per_q"][0]["c_C_base"]) == 2
    assert acc["pq_ctx_idx"] == [0, 0] and acc["pq_q_idx"] == [0, 1]
    # ...but only the non-skip row saved an (uncompressed, loadable) npz.
    assert not (tmp_dir / "ctx0__q0.npz").exists()
    assert (tmp_dir / "ctx0__q1.npz").exists()
    with np.load(tmp_dir / "ctx0__q1.npz") as z:
        assert z["full_base"].dtype == np.float16
        assert str(z["context_id"]) == "ctx0" and int(z["question_index"]) == 1
    assert len(acc["cell_bytes"]) == 1  # only the saved row is byte-measured


def test_skip_forward_rows_skip_the_capture_entirely(tmp_path, monkeypatch):
    """IO-4b: a skip-forward row (accum_ckpt-covered) never reaches the capture."""
    import issue813_run_cell as rc

    calls = {"n": 0}

    def _cap(*a, **k):
        calls["n"] += 1
        return _fake_caps()

    monkeypatch.setattr(rc, "_capture_full_and_reduce", _cap)
    contexts = [{"id": "ctx0", "family": "fam", "system_prompt": None, "prefix_messages": []}]
    questions = ["q a", "q b"]
    r_lookup = {(0, 0): "resp a", (0, 1): "resp b"}

    acc = rc._extract_pairs(
        object(),
        object(),
        object(),
        contexts,
        questions,
        r_lookup,
        behavior="marker",
        substrate="generic",
        tmp_dir=tmp_path / "unreduced_tmp" / "marker" / "generic",
        device="cpu",
        max_new=8,
        upload=False,
        skip_forward_keys=frozenset({"ctx0__q0"}),
    )
    assert calls["n"] == 1  # only the non-skipped row was captured
    assert acc["n_cells_done"] == 1
    assert len(acc["per_q"][0]["c_C_base"]) == 1


def test_hf_uploaded_rows_filters_this_cells_prefix(monkeypatch):
    """IO-3: the skip-set is built from ONE list_repo_files call, cell-prefix-filtered."""
    import huggingface_hub
    import issue813_run_cell as rc

    files = [
        f"{rc.EXPERIMENT_NAME}/unreduced/marker/generic/ctxA__q0.npz",
        f"{rc.EXPERIMENT_NAME}/unreduced/marker/generic/ctxA__q1.npz",
        f"{rc.EXPERIMENT_NAME}/unreduced/marker/generic/{rc.R_LOOKUP_NAME}",
        f"{rc.EXPERIMENT_NAME}/unreduced/marker/elicit/ctxA__q0.npz",  # other substrate
        f"{rc.EXPERIMENT_NAME}/reduced/marker/generic/summary.npz",  # reduced, not a row
        "other_experiment/unreduced/marker/generic/ctxB__q0.npz",
    ]
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda *a, **k: files)
    keys, has_r = rc._hf_uploaded_rows("marker", "generic")
    assert keys == {"ctxA__q0", "ctxA__q1"}
    assert has_r is True


def test_reenqueue_guards_hf_rows_truncated_files_and_ckpt(tmp_path):
    """IO-3: local re-enqueue skips HF-complete rows (deleted), truncated files, the ckpt."""
    import issue813_run_cell as rc

    # Complete row NOT on HF → re-enqueued for the first flush.
    np.savez(tmp_path / "c1__q0.npz", full_base=np.zeros(1), full_trained=np.zeros(1))
    # Complete row already on HF → deleted, never double-enqueued.
    np.savez(tmp_path / "c1__q1.npz", full_base=np.zeros(1), full_trained=np.zeros(1))
    # Truncated write from an interrupted process → left for the loop to recompute.
    (tmp_path / "c1__q2.npz").write_bytes(b"PK\x03\x04 truncated")
    # The local-only accumulator checkpoint is NEVER enqueued for upload.
    np.savez(tmp_path / rc.ACCUM_CKPT_NAME, full_base=np.zeros(1), full_trained=np.zeros(1))

    pending = rc._reenqueue_local_npz(tmp_path, {"c1__q1"}, "pfx")
    assert [(p.name, rp) for p, rp in pending] == [("c1__q0.npz", "pfx/c1__q0.npz")]
    assert not (tmp_path / "c1__q1.npz").exists()  # stale local copy of an HF row dropped
    assert (tmp_path / "c1__q2.npz").exists()  # truncated file left for recompute
    assert (tmp_path / rc.ACCUM_CKPT_NAME).exists()


def test_accum_ckpt_roundtrip_and_mismatch_guard(tmp_path):
    """IO-4b: accum_ckpt round-trips per_q/pq_rows; any skip-set mismatch → None."""
    import issue813_run_cell as rc

    rng = np.random.default_rng(0)
    flat = {
        k: [rng.normal(size=(rc.N_LAYERS, rc.HIDDEN)).astype(np.float32) for _ in range(2)]
        for k in rc._PQ_KEYS
    }
    ckpt = tmp_path / rc.ACCUM_CKPT_NAME
    rc._save_accum_ckpt(
        ckpt,
        flat_rows=flat,
        pq_ctx_idx=[0, 1],
        pq_q_idx=[0, 0],
        row_keys=["c0__q0", "c1__q0"],
        n_cells_done=2,
        n_empty=0,
        cell_bytes=[10, 12],
        behavior="marker",
        substrate="generic",
    )
    acc = rc._load_accum_ckpt(
        ckpt,
        behavior="marker",
        substrate="generic",
        n_contexts=2,
        n_questions=1,
        expected_keys={"c0__q0", "c1__q0"},
    )
    assert acc is not None
    assert acc["n_cells_done"] == 2 and acc["pq_ctx_idx"] == [0, 1]
    np.testing.assert_allclose(acc["per_q"][1]["v_A_base"][0], flat["v_A_base"][1])
    np.testing.assert_allclose(
        acc["pq_rows"]["c_C_base"][0], flat["c_C_base"][0][rc.HEADLINE_LAYER]
    )
    # A skip-set the ckpt does not EXACTLY cover → conservative None (recompute).
    assert (
        rc._load_accum_ckpt(
            ckpt,
            behavior="marker",
            substrate="generic",
            n_contexts=2,
            n_questions=1,
            expected_keys={"c0__q0"},
        )
        is None
    )
    # A different cell's ckpt never seeds this cell.
    assert (
        rc._load_accum_ckpt(
            ckpt,
            behavior="fact",
            substrate="generic",
            n_contexts=2,
            n_questions=1,
            expected_keys={"c0__q0", "c1__q0"},
        )
        is None
    )
    # A smaller --max-questions run than the checkpoint → conservative None (C3
    # qi-bound guard: out-of-grid question rows must not seed the summary averages).
    assert (
        rc._load_accum_ckpt(
            ckpt,
            behavior="marker",
            substrate="generic",
            n_contexts=2,
            n_questions=0,
            expected_keys={"c0__q0", "c1__q0"},
        )
        is None
    )
    # IO-1 companion: the ckpt writer must use plain savez — deflate on the
    # full-cumulative fp32 rewrite is the reviewed-out serial-stall blocker.
    import inspect

    assert "np.savez_compressed(" not in inspect.getsource(rc._save_accum_ckpt)


def test_flush_uploads_checkpoints_and_respects_keep_local(tmp_path, monkeypatch):
    """IO-4: a flush batch-commits, deletes locals EXCEPT keep_local, writes the ckpt.

    r_lookup.json rides the first flush but survives the post-commit delete
    (keep_local); row npzs are deleted after commit; the local-only accumulator
    checkpoint is written at every flush and never appears in any commit.
    """
    import issue813_run_cell as rc

    monkeypatch.setattr(rc, "_capture_full_and_reduce", lambda *a, **k: _fake_caps())
    commits: list[list] = []
    monkeypatch.setattr(rc, "_hf_batch_commit", lambda items: commits.append(list(items)))
    monkeypatch.setattr(rc, "BATCH_UPLOAD_CHUNK", 1)  # flush on every enqueue

    tmp_dir = tmp_path / "unreduced_tmp" / "marker" / "generic"
    tmp_dir.mkdir(parents=True)
    keep = tmp_dir / rc.R_LOOKUP_NAME
    keep.write_text("{}")
    ckpt = tmp_dir / rc.ACCUM_CKPT_NAME
    contexts = [{"id": "ctx0", "family": "fam", "system_prompt": None, "prefix_messages": []}]

    acc = rc._extract_pairs(
        object(),
        object(),
        object(),
        contexts,
        ["q a", "q b"],
        {(0, 0): "resp a", (0, 1): "resp b"},
        behavior="marker",
        substrate="generic",
        tmp_dir=tmp_dir,
        device="cpu",
        max_new=8,
        upload=True,
        pending_init=[(keep, "pfx/r_lookup.json")],
        keep_local=frozenset({keep}),
        ckpt_path=ckpt,
    )
    assert acc["n_cells_done"] == 2
    # r_lookup.json was uploaded in the FIRST flush yet KEPT locally.
    assert commits and any(rp == "pfx/r_lookup.json" for c in commits for _, rp in c)
    assert keep.exists()
    # Row npzs were uploaded then deleted locally (upload-then-delete).
    uploaded = {rp for c in commits for _, rp in c}
    assert any(rp.endswith("/ctx0__q0.npz") for rp in uploaded), uploaded
    assert any(rp.endswith("/ctx0__q1.npz") for rp in uploaded), uploaded
    assert not list(tmp_dir.glob("ctx0__q*.npz"))
    # The accumulator checkpoint exists locally and was NEVER enqueued for upload.
    assert ckpt.exists()
    assert all(p.name != rc.ACCUM_CKPT_NAME for c in commits for p, _ in c)


def test_r_lookup_json_roundtrip(tmp_path):
    """IO-4a: r_lookup.json round-trips (ci, qi)↔row-key; dropped contexts are ignored."""
    import issue813_run_cell as rc

    contexts = [{"id": "ctxA", "family": "f"}, {"id": "ctxB", "family": "f"}]
    path = tmp_path / rc.R_LOOKUP_NAME
    rc._write_r_lookup_json(path, {(0, 0): "r00", (1, 3): "r13"}, contexts)
    assert rc._load_r_lookup_json(path, contexts) == {(0, 0): "r00", (1, 3): "r13"}
    # A context dropped by --max-contexts is ignored on load, not crashed on.
    assert rc._load_r_lookup_json(path, contexts[:1]) == {(0, 0): "r00"}
