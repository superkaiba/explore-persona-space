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
