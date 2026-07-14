"""Tests for the shared-VM BLAS/torch thread caps (#847).

``orchestrate.env`` setdefaults ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` /
``OPENBLAS_NUM_THREADS`` / ``NUMEXPR_NUM_THREADS`` to 8 on the SHARED dev VM
only (positive detection: ``/mnt/eps-data`` mounted OR hostname
``cia-benchmark-vm``; fails OPEN on pods / GCE / SLURM / unknown hosts).
Incident 2026-07-02: 5-6 uncapped torch/BLAS jobs each held ~32 runnable
threads on the 32-core shared VM (load 186-226) while each realized only
~5-6 cores.

Every real signal is monkeypatched so the suite passes identically on any
host:

* env vars ``EPS_SHARED_VM`` / ``EPS_VM_THREAD_CAP`` / ``SLURM_JOB_ID`` /
  ``RUNPOD_POD_ID`` and the 4 thread keys are delenv'd per-test (autouse).
* ``os.path.ismount`` is faked PATH-SENSITIVELY, controlling BOTH
  ``/mnt/eps-data`` AND ``/workspace`` — ``is_runpod_env()`` probes
  ``/workspace`` through the same ``os.path.ismount``, so a blanket
  ``lambda p: True`` would route as RunPod and a partial fake would be
  silently host-dependent (cf. ``tests/test_env_three_way_branch.py``).
* ``platform.node`` is faked for the hostname clause.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from explore_persona_space.orchestrate import env as env_mod

_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_SIGNAL_VARS = ("EPS_SHARED_VM", "EPS_VM_THREAD_CAP", "SLURM_JOB_ID", "RUNPOD_POD_ID")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete every real signal + thread key so tests are host-independent."""
    for var in (*_SIGNAL_VARS, *_THREAD_KEYS):
        monkeypatch.delenv(var, raising=False)


def _patch_signals(
    monkeypatch: pytest.MonkeyPatch,
    *,
    eps_data: bool = False,
    workspace: bool = False,
    hostname: str = "some-other-host",
) -> None:
    """Path-sensitive ismount fake + hostname fake (controls ALL positive signals).

    Controls BOTH ``/mnt/eps-data`` (the shared-VM clause) AND ``/workspace``
    (the RunPod clause probed through the same ``os.path.ismount``); every
    other path reads not-a-mount for determinism.
    """

    def fake_ismount(path: str | os.PathLike[str]) -> bool:
        return {"/mnt/eps-data": eps_data, "/workspace": workspace}.get(str(path), False)

    monkeypatch.setattr(env_mod.os.path, "ismount", fake_ismount)
    monkeypatch.setattr(env_mod.platform, "node", lambda: hostname)


# ---------------------------------------------------------------------------
# is_shared_vm_env — positive detection, fails open
# ---------------------------------------------------------------------------


def test_detected_by_data_disk_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_signals(monkeypatch, eps_data=True, hostname="not-the-vm")
    assert env_mod.is_shared_vm_env() is True


def test_detected_by_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_signals(monkeypatch, eps_data=False, hostname="cia-benchmark-vm")
    assert env_mod.is_shared_vm_env() is True


def test_runpod_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """RunPod wins even against a (hypothetical) /mnt/eps-data mount."""
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123podid")
    _patch_signals(monkeypatch, eps_data=True, hostname="cia-benchmark-vm")
    assert env_mod.is_shared_vm_env() is False


def test_slurm_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    _patch_signals(monkeypatch, eps_data=True, hostname="cia-benchmark-vm")
    assert env_mod.is_shared_vm_env() is False


def test_gce_local_route_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GCE instance routes local but has NO positive signal → no cap."""
    _patch_signals(monkeypatch, eps_data=False, workspace=False, hostname="eps-issue-999")
    assert env_mod.is_shared_vm_env() is False


def test_forced_on_and_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """EPS_SHARED_VM overrides in BOTH directions, before any probe."""
    # Forced ON with zero real signals.
    _patch_signals(monkeypatch, eps_data=False, hostname="laptop")
    monkeypatch.setenv("EPS_SHARED_VM", "1")
    assert env_mod.is_shared_vm_env() is True
    # Forced OFF with ALL real signals present (the kill switch).
    _patch_signals(monkeypatch, eps_data=True, hostname="cia-benchmark-vm")
    for falsy in ("0", "false", "no", "off", "False", " OFF "):
        monkeypatch.setenv("EPS_SHARED_VM", falsy)
        assert env_mod.is_shared_vm_env() is False, falsy


# ---------------------------------------------------------------------------
# _apply_shared_vm_thread_caps — setdefault semantics + the EPS_VM_THREAD_CAP knob
# ---------------------------------------------------------------------------


def test_caps_applied_default_8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EPS_SHARED_VM", "1")
    env_mod._apply_shared_vm_thread_caps()
    for key in _THREAD_KEYS:
        assert os.environ[key] == "8", key


def test_explicit_value_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """setdefault never clobbers an explicit launch-time value."""
    monkeypatch.setenv("EPS_SHARED_VM", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "32")
    env_mod._apply_shared_vm_thread_caps()
    assert os.environ["OMP_NUM_THREADS"] == "32"
    for key in _THREAD_KEYS[1:]:
        assert os.environ[key] == "8", key


def test_cap_knob_custom_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EPS_SHARED_VM", "1")
    monkeypatch.setenv("EPS_VM_THREAD_CAP", "4")
    env_mod._apply_shared_vm_thread_caps()
    for key in _THREAD_KEYS:
        assert os.environ[key] == "4", key


def test_cap_knob_zero_and_empty_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EPS_SHARED_VM", "1")
    for disabling in ("0", "", "  ", "-3"):
        monkeypatch.setenv("EPS_VM_THREAD_CAP", disabling)
        env_mod._apply_shared_vm_thread_caps()
        for key in _THREAD_KEYS:
            assert key not in os.environ, (disabling, key)


def test_cap_knob_malformed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd explicit knob is a config bug — fail loud, never default."""
    monkeypatch.setenv("EPS_SHARED_VM", "1")
    monkeypatch.setenv("EPS_VM_THREAD_CAP", "eight")
    with pytest.raises(ValueError):
        env_mod._apply_shared_vm_thread_caps()


def test_off_vm_no_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_signals(monkeypatch, eps_data=False, hostname="laptop")
    env_mod._apply_shared_vm_thread_caps()
    for key in _THREAD_KEYS:
        assert key not in os.environ, key


# ---------------------------------------------------------------------------
# Call-site wiring
# ---------------------------------------------------------------------------


def test_load_dotenv_wires_caps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """load_dotenv() applies the caps (pins the call-site wiring)."""
    monkeypatch.delenv("HF_HOME", raising=False)
    _patch_signals(monkeypatch, workspace=False)  # deterministic off-pod routing
    monkeypatch.setenv("EPS_SHARED_VM", "1")
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_KEY=val\n")
    env_mod.load_dotenv(str(env_file))
    for key in _THREAD_KEYS:
        assert os.environ[key] == "8", key


def test_dotenv_file_value_wins_over_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A thread key in .env is explicit config and wins over the cap.

    Pins the after-``_dotenv_load`` ordering contract: fails iff the cap
    call is refactored to precede ``_dotenv_load`` in ``load_dotenv``.
    """
    monkeypatch.delenv("HF_HOME", raising=False)
    _patch_signals(monkeypatch, workspace=False)
    monkeypatch.setenv("EPS_SHARED_VM", "1")
    env_file = tmp_path / ".env"
    env_file.write_text("OMP_NUM_THREADS=32\n")
    env_mod.load_dotenv(str(env_file))
    assert os.environ["OMP_NUM_THREADS"] == "32"
    assert os.environ["MKL_NUM_THREADS"] == "8"


def test_setup_worker_caps_before_torch_import() -> None:
    """Structural: setup_worker caps BEFORE its torch import freezes the pool.

    ``.index()`` (not ``.find()``) so the ABSENCE of either token raises
    instead of vacuously passing on -1 comparisons.
    """
    src = inspect.getsource(env_mod.setup_worker)
    assert src.index("_apply_shared_vm_thread_caps") < src.index("import torch")


# ---------------------------------------------------------------------------
# Recurring guard: no NEW heavy-import-before-load_dotenv VM entrypoints
# ---------------------------------------------------------------------------
# #1146 extension: the guard flags every transitively-heavy HEAVY_IMPORT_ROOTS
# root (matplotlib/pandas/scipy/... pull numpy at import time; transformers/
# vllm/peft/trl pull torch), not just literal torch/numpy imports.
# Scoped-out residuals (deliberate):
#   * FIRST-PARTY heavy imports: an `explore_persona_space.*` module that pulls
#     torch/numpy before load_dotenv() still passes — the package root cannot
#     join HEAVY_IMPORT_ROOTS without flagging the preamble's own
#     `from explore_persona_space.orchestrate.env import load_dotenv` wrapper
#     import; the guard's claim is scoped to HEAVY_IMPORT_ROOTS
#     (test_dotenv_wrapper_import_chain_is_heavy_free pins the wrapper chain).
#   * `_first_load_dotenv_line` is a SUBSTRING scan — a commented-out
#     `load_dotenv(` line satisfies it (pre-existing; the code-review
#     diff-shape check pins the real wrapper call).
# #1187 widening residuals (deliberate):
#   * A NEW heavy import added to an already-grandfathered file is invisible —
#     the entry stays "still-violating" either way; inherent to the freeze-
#     epoch design (the currency tests pin entries, not per-import deltas).
#   * An UNGUARDED module-level executable under src/experiments (the
#     issue_779/scaling_grid.py shape) escapes both class predicates — AST
#     cannot distinguish it from a library module.
#   * Unguarded src/experiments LIBRARY modules are out of scope: they inherit
#     caps in-process from a CAPPED executing entrypoint (NOT guaranteed when
#     the entrypoint is itself grandfathered — that residual is accepted);
#     forcing import-time load_dotenv() into libraries adds side effects for
#     zero cap benefit.
#   * UNTRACKED files are out of scope — they can never be grandfathered (the
#     currency tests reject untracked entries) and every enforcement surface
#     (Step 9c, the Step-10d merge gate, trunk) runs on committed state.

# Second freeze epoch (#895): TRACKED offenders that accreted while the Step
# 9c selector gap (#895) left this guard unselected for scripts/*.py diffs —
# frozen at map-introduction time so the newly-selectable gate starts green in
# worktree gate runs. Kept as a SEPARATE dict (merged into the main allowlist
# below) so test_grandfathered_895_block_is_current can pin exactly these
# entries: each must stay git-tracked, inside the scanner's target set, and
# still-violating with the recorded reason — a root-cause fix (e.g. #779's
# 2-line load_dotenv()-before-torch reorder) makes its entry FAIL there until
# DELETED. The SELECTOR gap is closed for selector-gated diffs by
# GLOB_SCAN_TESTS (scripts/select_step9c_tests.py, #895); offenders can still
# land via paths that bypass Step 9c, so this block may only SHRINK.
GRANDFATHERED_895: dict[str, str] = {
    # BEGIN GENERATED (#895)
    "scripts/issue779_arm_headline.py": "no-dotenv",
    "scripts/issue779_arm_headline_pod.py": "no-dotenv",
    "scripts/issue779_arm_headline_summaries.py": "no-dotenv",
    "scripts/issue779_capture_answer_summaries.py": "import-order",
    "scripts/issue779_edges.py": "no-dotenv",
    # END GENERATED
}

# Third freeze epoch (#1187): TRACKED offenders that pre-dated the widened
# target set (scripts/**/*.py beyond issue*_*.py, incl. subdirectories, plus
# __main__-guarded src/experiments modules — the src class froze ZERO entries:
# its only 2 offenders were preamble-fixed in #1187). Same shrink-only
# contract as GRANDFATHERED_895: each entry must stay tracked, inside the
# scanner's target set, and still-violating with the recorded reason; a
# root-cause fix makes its entry FAIL the currency test until DELETED.
GRANDFATHERED_1187: dict[str, str] = {
    # BEGIN GENERATED (#1187)
    "scripts/analyze_100_persona_source_filtered.py": "no-dotenv",
    "scripts/analyze_axis_tails.py": "import-order",
    "scripts/analyze_category_projections.py": "no-dotenv",
    "scripts/analyze_causal_proximity.py": "no-dotenv",
    "scripts/analyze_cot_tracking.py": "no-dotenv",
    "scripts/analyze_em_axis.py": "no-dotenv",
    "scripts/analyze_i181.py": "no-dotenv",
    "scripts/analyze_issue260.py": "no-dotenv",
    "scripts/analyze_issue415.py": "no-dotenv",
    "scripts/analyze_issue_358_pca.py": "no-dotenv",
    "scripts/analyze_issue_358_probe.py": "no-dotenv",
    "scripts/analyze_issue_358_umap.py": "no-dotenv",
    "scripts/analyze_leakage.py": "no-dotenv",
    "scripts/analyze_length_rate_296.py": "no-dotenv",
    "scripts/analyze_length_rate_n48.py": "no-dotenv",
    "scripts/analyze_manifold_axes.py": "no-dotenv",
    "scripts/analyze_outliers_pertoken.py": "no-dotenv",
    "scripts/analyze_single_token_sweep.py": "no-dotenv",
    "scripts/archive/merge_and_eval.py": "import-order",
    "scripts/archive/run_alignment_only.py": "import-order",
    "scripts/archive/run_all_missing.py": "import-order",
    "scripts/archive/run_round5_em.py": "no-dotenv",
    "scripts/archive/run_round5_tulu.py": "no-dotenv",
    "scripts/archive/run_round5_worker.py": "no-dotenv",
    "scripts/archive/run_round7_extra.py": "import-order",
    "scripts/archive/run_round8_dpo_variants.py": "import-order",
    "scripts/archive/run_round8_sdf.py": "no-dotenv",
    "scripts/archive/run_round8_sdf_v2.py": "import-order",
    "scripts/archive/test_multidim_identity.py": "no-dotenv",
    "scripts/benchmark_tier1.py": "no-dotenv",
    "scripts/build_canonical_persona_pool.py": "no-dotenv",
    "scripts/build_language_inversion_data.py": "import-order",
    "scripts/build_language_inversion_data_v2.py": "import-order",
    "scripts/compare_extraction_methods.py": "no-dotenv",
    "scripts/compute_issue_203_stats.py": "no-dotenv",
    "scripts/compute_zelthari_centered_cosine.py": "no-dotenv",
    "scripts/diag_loaders.py": "no-dotenv",
    "scripts/download_arc_data.py": "no-dotenv",
    "scripts/download_capability_datasets.py": "import-order",
    "scripts/eval_all_sequential.py": "no-dotenv",
    "scripts/eval_arc_splits.py": "import-order",
    "scripts/extract_centroids_and_analyze.py": "no-dotenv",
    "scripts/extract_persona_vectors.py": "no-dotenv",
    "scripts/extract_prompt_divergence_activations.py": "no-dotenv",
    "scripts/figures/plot_issue261_per_persona.py": "no-dotenv",
    "scripts/figures_issue_389_output_composition.py": "no-dotenv",
    "scripts/figures_issue_390.py": "no-dotenv",
    "scripts/figures_issue_390_output_composition.py": "no-dotenv",
    "scripts/i207_compute_js_matrix.py": "no-dotenv",
    "scripts/i207_run_regression.py": "no-dotenv",
    "scripts/i368_figures.py": "no-dotenv",
    "scripts/i380_cosine_pairwise.py": "no-dotenv",
    "scripts/i380_pairwise_scatters.py": "no-dotenv",
    "scripts/i380_raw_scatters.py": "no-dotenv",
    "scripts/i395_probe_marker_priors.py": "no-dotenv",
    "scripts/i396_make_figures.py": "no-dotenv",
    "scripts/i460_phase0_preflight.py": "import-order",
    "scripts/i460_phase1_generate_R.py": "import-order",
    "scripts/i460_phase23_train.py": "import-order",
    "scripts/i460_phase2_smoke_check.py": "import-order",
    "scripts/i460_phase4_eval.py": "import-order",
    "scripts/i460_phase5_analyze.py": "no-dotenv",
    "scripts/i465_make_figures_v2.py": "no-dotenv",
    "scripts/i474_cosine_followup.py": "no-dotenv",
    "scripts/i474_phase0_preflight.py": "import-order",
    "scripts/i474_phase23_train.py": "import-order",
    "scripts/i474_phase2_smoke_check.py": "import-order",
    "scripts/i474_phase4_eval.py": "import-order",
    "scripts/i474_phase5_analyze.py": "no-dotenv",
    "scripts/i474_phase6_figures.py": "no-dotenv",
    "scripts/i488_diagnostic_measure.py": "import-order",
    "scripts/i488_diagnostic_train.py": "import-order",
    "scripts/i488_figures_round2.py": "no-dotenv",
    "scripts/i488_phase0_generate_data.py": "import-order",
    "scripts/i488_phase1_predictors.py": "import-order",
    "scripts/i488_phase23_train.py": "import-order",
    "scripts/i488_phase2_ladder_emit.py": "import-order",
    "scripts/i488_phase2_smoke_calibrate.py": "import-order",
    "scripts/i488_phase4_eval_onpolicy.py": "import-order",
    "scripts/i488_phase5_analyze.py": "no-dotenv",
    "scripts/i488_runaway_figure.py": "no-dotenv",
    "scripts/i488_smoke_stratified_clip.py": "no-dotenv",
    "scripts/i501_make_figures_blog.py": "no-dotenv",
    "scripts/i504_make_figures.py": "no-dotenv",
    "scripts/i504_phase_phase05.py": "import-order",
    "scripts/i504_probe_bank_geometry.py": "import-order",
    "scripts/i504_round6_recompute_mean_centered.py": "import-order",
    "scripts/i504_shadow_flip_magnitude.py": "import-order",
    "scripts/i504_shadow_flip_magnitude_plot.py": "no-dotenv",
    "scripts/i504_smoke_local.py": "import-order",
    "scripts/i533_bw_implant_leakage_figure.py": "no-dotenv",
    "scripts/i533_bw_leakage_controlled_figure.py": "no-dotenv",
    "scripts/i533_clean_result_figures.py": "no-dotenv",
    "scripts/i533_margin_figures.py": "no-dotenv",
    "scripts/i537_seed2_replication_read.py": "no-dotenv",
    "scripts/i549_audit_504.py": "no-dotenv",
    "scripts/i549_audit_532.py": "no-dotenv",
    "scripts/i556_analyzer_figures.py": "no-dotenv",
    "scripts/i610_analyzer_figures.py": "no-dotenv",
    "scripts/i613_singlespace_figures.py": "no-dotenv",
    "scripts/issue_355/compute_deferred_stats_and_plot.py": "no-dotenv",
    "scripts/issue_480/i480_analyze.py": "import-order",
    "scripts/issue_480/i480_phase2b_logprob.py": "import-order",
    "scripts/issue_480/i480_syco_geometry_controls.py": "no-dotenv",
    "scripts/issue_480/plot_clean_result.py": "no-dotenv",
    "scripts/issue_480/plot_controlled_scatter.py": "no-dotenv",
    "scripts/issue_480/smoke_build_guard_long_neg.py": "no-dotenv",
    "scripts/issue_480/smoke_collator_label_dump.py": "no-dotenv",
    "scripts/issue_597/analyze_titration_597.py": "import-order",
    "scripts/issue_597/fig_armD_3way_panel_only.py": "no-dotenv",
    "scripts/issue_597/fig_bystander_emission_anchors.py": "no-dotenv",
    "scripts/issue_597/fig_dense_early_597.py": "no-dotenv",
    "scripts/issue_597/fig_h3_shared_dose_overlay.py": "no-dotenv",
    "scripts/issue_642/i642_figures.py": "no-dotenv",
    "scripts/issue_642/i642_r4_figure.py": "no-dotenv",
    "scripts/issue_642/i642_r5_figure.py": "no-dotenv",
    "scripts/issue_653/i653_postpod_bootstrap.py": "no-dotenv",
    "scripts/issue_653/plot_i653_figures.py": "no-dotenv",
    "scripts/issue_653/plot_i653_reladder_figures.py": "no-dotenv",
    "scripts/make_363_figure.py": "no-dotenv",
    "scripts/make_figure_issue_296.py": "no-dotenv",
    "scripts/make_i207_js_figures.py": "no-dotenv",
    "scripts/make_i456_figures.py": "no-dotenv",
    "scripts/make_issue138_hero.py": "no-dotenv",
    "scripts/make_issue375_hero.py": "no-dotenv",
    "scripts/make_issue407_figures.py": "no-dotenv",
    "scripts/make_issue516_figures.py": "no-dotenv",
    "scripts/make_issue_385_figures.py": "no-dotenv",
    "scripts/plot_100_persona_analysis.py": "no-dotenv",
    "scripts/plot_100_persona_category_rho.py": "no-dotenv",
    "scripts/plot_100_persona_scatter_simple.py": "no-dotenv",
    "scripts/plot_492_wave0_verdict.py": "no-dotenv",
    "scripts/plot_aim5_25pct_seeds_42_137.py": "no-dotenv",
    "scripts/plot_aim5_25pct_seeds_42_137_256.py": "no-dotenv",
    "scripts/plot_all_results.py": "no-dotenv",
    "scripts/plot_axis_origins.py": "no-dotenv",
    "scripts/plot_cosine_attenuation.py": "no-dotenv",
    "scripts/plot_cot_tracking.py": "no-dotenv",
    "scripts/plot_dose_response_237_extended.py": "no-dotenv",
    "scripts/plot_full_matrix.py": "no-dotenv",
    "scripts/plot_i395_marker_priors.py": "no-dotenv",
    "scripts/plot_i398_rank_ordering.py": "no-dotenv",
    "scripts/plot_i432_rank_by_distance.py": "no-dotenv",
    "scripts/plot_i460_clean_result.py": "no-dotenv",
    "scripts/plot_i461_predictor_grid.py": "no-dotenv",
    "scripts/plot_i461_predictor_scatters.py": "no-dotenv",
    "scripts/plot_i464_clean_result_hero.py": "no-dotenv",
    "scripts/plot_i464_revision_figs.py": "no-dotenv",
    "scripts/plot_i479_dead_floor.py": "no-dotenv",
    "scripts/plot_i506_fwft_vs_lora_survival.py": "no-dotenv",
    "scripts/plot_issue186_context_scaling.py": "no-dotenv",
    "scripts/plot_issue186_source_vs_bystander.py": "no-dotenv",
    "scripts/plot_issue186_train_eval_heatmap.py": "no-dotenv",
    "scripts/plot_issue186_unified_6arm.py": "no-dotenv",
    "scripts/plot_issue186_v2_hero.py": "no-dotenv",
    "scripts/plot_issue192_hero.py": "no-dotenv",
    "scripts/plot_issue237_tldr.py": "no-dotenv",
    "scripts/plot_issue238_hero.py": "no-dotenv",
    "scripts/plot_issue238_supporting.py": "no-dotenv",
    "scripts/plot_issue311_clean_result.py": "no-dotenv",
    "scripts/plot_issue331.py": "no-dotenv",
    "scripts/plot_issue333_clean_result.py": "no-dotenv",
    "scripts/plot_issue356_hero.py": "no-dotenv",
    "scripts/plot_issue365_hero.py": "no-dotenv",
    "scripts/plot_issue382_clean_result.py": "no-dotenv",
    "scripts/plot_issue383_365layout_hero.py": "no-dotenv",
    "scripts/plot_issue383_heatmap.py": "no-dotenv",
    "scripts/plot_issue383_per_cell_heatmap.py": "no-dotenv",
    "scripts/plot_issue397_hero.py": "no-dotenv",
    "scripts/plot_issue399.py": "no-dotenv",
    "scripts/plot_issue444.py": "no-dotenv",
    "scripts/plot_issue444_5way.py": "no-dotenv",
    "scripts/plot_issue444_bystander.py": "no-dotenv",
    "scripts/plot_issue444_histcn_followup.py": "no-dotenv",
    "scripts/plot_issue444_persona_distance.py": "no-dotenv",
    "scripts/plot_issue475_install_collapse.py": "no-dotenv",
    "scripts/plot_issue500_predictors.py": "no-dotenv",
    "scripts/plot_issue503_v2.py": "no-dotenv",
    "scripts/plot_issue541_analyzer.py": "no-dotenv",
    "scripts/plot_issue562_panel.py": "no-dotenv",
    "scripts/plot_issue640_diagonal.py": "no-dotenv",
    "scripts/plot_issue651_depth_robustness.py": "no-dotenv",
    "scripts/plot_issue651_figures.py": "no-dotenv",
    "scripts/plot_issue_156_hero.py": "no-dotenv",
    "scripts/plot_issue_157_hero.py": "no-dotenv",
    "scripts/plot_issue_157_per_candidate_frde.py": "no-dotenv",
    "scripts/plot_issue_157_stage_b_hero.py": "no-dotenv",
    "scripts/plot_issue_157_stage_b_hero_v2.py": "no-dotenv",
    "scripts/plot_issue_164_hero.py": "no-dotenv",
    "scripts/plot_issue_188_hero.py": "no-dotenv",
    "scripts/plot_issue_203_hero.py": "no-dotenv",
    "scripts/plot_issue_213_final.py": "no-dotenv",
    "scripts/plot_issue_213_geometry_predicts.py": "no-dotenv",
    "scripts/plot_issue_247.py": "no-dotenv",
    "scripts/plot_issue_276_anth_token.py": "no-dotenv",
    "scripts/plot_issue_276_combined.py": "no-dotenv",
    "scripts/plot_issue_358.py": "no-dotenv",
    "scripts/plot_issue_389.py": "no-dotenv",
    "scripts/plot_issue_389_framings.py": "no-dotenv",
    "scripts/plot_issue_89_hero.py": "no-dotenv",
    "scripts/plot_leakage_vs_cosine_all.py": "no-dotenv",
    "scripts/plot_leakage_vs_cosine_none.py": "no-dotenv",
    "scripts/plot_length_rate_correlation.py": "no-dotenv",
    "scripts/plot_length_rate_n48.py": "no-dotenv",
    "scripts/plot_proximity_transfer.py": "no-dotenv",
    "scripts/plot_single_token_sweep_heatmap.py": "no-dotenv",
    "scripts/plot_strong_convergence.py": "no-dotenv",
    "scripts/plot_trait_transfer.py": "no-dotenv",
    "scripts/poll_lmsys_taxonomy.py": "import-order",
    "scripts/precheck_i181_axes.py": "no-dotenv",
    "scripts/project_categories_instruct.py": "no-dotenv",
    "scripts/project_categories_onto_axis.py": "import-order",
    "scripts/project_corpus_fast.py": "no-dotenv",
    "scripts/project_corpus_single_gpu.py": "import-order",
    "scripts/project_corpus_v2.py": "no-dotenv",
    "scripts/rollup_issue562_panel.py": "no-dotenv",
    "scripts/run_em_multiseed.py": "no-dotenv",
    "scripts/run_experiment_369.py": "no-dotenv",
    "scripts/run_issue_156.py": "import-order",
    "scripts/run_issue_203.py": "import-order",
    "scripts/run_issue_203_train.py": "import-order",
    "scripts/run_issue_213_combined.py": "import-order",
    "scripts/run_issue_213_part_a.py": "import-order",
    "scripts/run_issue_213_part_b.py": "import-order",
    "scripts/run_issue_276_bare_anth.py": "import-order",
    "scripts/run_issue_276_continuation_sweep.py": "import-order",
    "scripts/run_issue_276_pre_poison_similarity.py": "import-order",
    "scripts/run_issue_276_slash_anth.py": "import-order",
    "scripts/run_issue_276_teacher_forced_js.py": "import-order",
    "scripts/run_issue_358_extract.py": "import-order",
    "scripts/run_issue_360_target_logprobs.py": "no-dotenv",
    "scripts/run_persona_composition.py": "no-dotenv",
    "scripts/run_proximity_transfer.py": "no-dotenv",
    "scripts/run_trait_transfer.py": "no-dotenv",
    "scripts/test_activation_steering.py": "no-dotenv",
    "scripts/test_multidim_identity_v2.py": "no-dotenv",
    "scripts/track_axis_during_cot.py": "no-dotenv",
    "scripts/train_stage_dpo.py": "no-dotenv",
    "scripts/train_stage_kto.py": "no-dotenv",
    "scripts/train_stage_sft.py": "no-dotenv",
    # END GENERATED
}

# Frozen grandfather allowlist — generated mechanically from the tree state at
# implementation time (same frozen-allowlist pattern as
# JUDGE_PIN_LEGACY_ALLOWLIST). Three freeze epochs:
#   #847 block (inline below) — the pre-guard tree state.
#   #895 block (GRANDFATHERED_895 above, merged in) — offenders that accreted
#     while the Step 9c selector gap (#895) left this guard unselected for
#     scripts/*.py diffs; frozen at map-introduction time so the
#     newly-selectable gate starts green.
#   #1187 block (GRANDFATHERED_1187 above, merged in) — pre-existing offenders
#     in the classes the #1187 widening newly covers (all tracked
#     scripts/**/*.py + __main__-guarded src/experiments modules); frozen at
#     widening time so the widened gate starts green.
# Do NOT add new entries beyond these generated blocks: a new entrypoint that
# imports torch/numpy (or another HEAVY_IMPORT_ROOTS root, #1146) at module
# top must call load_dotenv() FIRST so the shared-VM thread caps bind
# in-process (see .claude/rules/code-style.md, "Shared-VM CPU thread caps").
# Reason legend (one line per entry):
#   "no-dotenv"    — file never calls load_dotenv() (tree state at freeze)
#   "import-order" — module-top torch/numpy import (or another
#                    HEAVY_IMPORT_ROOTS root, #1146) precedes the first
#                    load_dotenv() call (tree state at freeze)
GRANDFATHERED_TORCH_BEFORE_DOTENV: dict[str, str] = {
    # BEGIN GENERATED (#847)
    "scripts/issue356_aggregate.py": "no-dotenv",
    "scripts/issue404_make_clean_figures.py": "no-dotenv",
    "scripts/issue404_predictor_cossim.py": "import-order",
    "scripts/issue404_predictor_kldiv.py": "import-order",
    "scripts/issue404_regress.py": "no-dotenv",
    "scripts/issue405_clean_result_analysis.py": "no-dotenv",
    "scripts/issue444_bystander_logprob.py": "no-dotenv",
    "scripts/issue444_fact_slice_js.py": "no-dotenv",
    "scripts/issue444_persona_distance_topic.py": "no-dotenv",
    "scripts/issue458_predictor_jsdiv.py": "import-order",
    "scripts/issue458_regress.py": "no-dotenv",
    "scripts/issue463_v2_figures.py": "no-dotenv",
    "scripts/issue467_figures.py": "no-dotenv",
    "scripts/issue472_negative_budget_figures.py": "no-dotenv",
    "scripts/issue477_clean_result_figures.py": "no-dotenv",
    "scripts/issue480_emission_rate_concordance.py": "no-dotenv",
    "scripts/issue483_smoke_from_legacy.py": "no-dotenv",
    "scripts/issue493_extraction_metric_bakeoff.py": "import-order",
    "scripts/issue493_make_clean_figures.py": "no-dotenv",
    "scripts/issue494_plain_english_figures.py": "no-dotenv",
    "scripts/issue500_interaction_check.py": "no-dotenv",
    "scripts/issue502_cpu_smoke.py": "no-dotenv",
    "scripts/issue502_deltaG_symmetry.py": "no-dotenv",
    "scripts/issue502_plot_best3_bars.py": "no-dotenv",
    "scripts/issue502_plot_best4_bars.py": "no-dotenv",
    "scripts/issue502_plot_predictor_rho.py": "no-dotenv",
    "scripts/issue502_plot_predictor_rho_bars.py": "no-dotenv",
    "scripts/issue505_r2_figure.py": "no-dotenv",
    "scripts/issue509_baserate_covariate.py": "no-dotenv",
    "scripts/issue509_baserate_covariate_earlylayer.py": "no-dotenv",
    "scripts/issue509_bystander_bootstrap.py": "no-dotenv",
    "scripts/issue509_figures.py": "no-dotenv",
    "scripts/issue509_pathb_fact_rerun.py": "import-order",
    "scripts/issue509_top2_scatter_figure.py": "no-dotenv",
    "scripts/issue511_make_figures.py": "no-dotenv",
    "scripts/issue511_probe_count_sweep.py": "no-dotenv",
    "scripts/issue518_figures.py": "no-dotenv",
    "scripts/issue519_partial_figures.py": "no-dotenv",
    "scripts/issue522_js_predictor.py": "no-dotenv",
    "scripts/issue522_js_regress.py": "no-dotenv",
    "scripts/issue522_js_smoke_toy.py": "no-dotenv",
    "scripts/issue522_make_figures.py": "no-dotenv",
    "scripts/issue523_make_interpretation_figures.py": "no-dotenv",
    "scripts/issue526_asym_gate_ladder.py": "no-dotenv",
    "scripts/issue526_asym_gate_ladder_plot.py": "no-dotenv",
    "scripts/issue527_dan_rank1_scalar_regression.py": "no-dotenv",
    "scripts/issue530_logit_analysis.py": "no-dotenv",
    "scripts/issue530_make_figures.py": "no-dotenv",
    "scripts/issue530_partial_scatter.py": "no-dotenv",
    "scripts/issue531_base_prior_reanalysis.py": "no-dotenv",
    "scripts/issue531_logit_plots.py": "no-dotenv",
    "scripts/issue531_logit_rescore.py": "import-order",
    "scripts/issue531_probability_space_plots.py": "no-dotenv",
    "scripts/issue532_followup_logp_slot.py": "import-order",
    "scripts/issue532_predictor_stress.py": "import-order",
    "scripts/issue534_make_figures.py": "no-dotenv",
    "scripts/issue536_audit.py": "import-order",
    "scripts/issue536_figures.py": "no-dotenv",
    "scripts/issue536_mixedlm_refit.py": "no-dotenv",
    "scripts/issue536_recompute_driver.py": "import-order",
    "scripts/issue538_make_figures.py": "no-dotenv",
    "scripts/issue539_corrected_reads_inference.py": "no-dotenv",
    "scripts/issue539_replot_v2.py": "no-dotenv",
    "scripts/issue539_residual_per_cohort.py": "no-dotenv",
    "scripts/issue540_jsrb_predictor.py": "import-order",
    "scripts/issue540_length_nuisance_figure.py": "no-dotenv",
    "scripts/issue540_length_nuisance_supplement.py": "no-dotenv",
    "scripts/issue541_geometry_extract.py": "no-dotenv",
    "scripts/issue541_geometry_joint.py": "no-dotenv",
    "scripts/issue545_followup_bcond_predictor.py": "no-dotenv",
    "scripts/issue545_measure_probe_lengths.py": "no-dotenv",
    "scripts/issue545_plot_metric_race.py": "no-dotenv",
    "scripts/issue548_incremental_validity.py": "no-dotenv",
    "scripts/issue548_leaderboard_fig_v2.py": "no-dotenv",
    "scripts/issue548_length_analysis.py": "no-dotenv",
    "scripts/issue548_pooled_seeds.py": "no-dotenv",
    "scripts/issue548_truncation_fig_plainlabels.py": "no-dotenv",
    "scripts/issue550_bystander_slope_figure.py": "no-dotenv",
    "scripts/issue550_make_figures.py": "no-dotenv",
    "scripts/issue550_slope_distance_correlation.py": "no-dotenv",
    "scripts/issue552_cross_arm_analysis.py": "no-dotenv",
    "scripts/issue552_figures.py": "no-dotenv",
    "scripts/issue552_mean_resp_svd.py": "no-dotenv",
    "scripts/issue553_panel.py": "no-dotenv",
    "scripts/issue559_analyzer_figures.py": "no-dotenv",
    "scripts/issue559_cross_behavior_self_scoring.py": "import-order",
    "scripts/issue559_disjoint_question_followup.py": "no-dotenv",
    "scripts/issue559_length_residual_followup.py": "no-dotenv",
    "scripts/issue559_panel_analysis.py": "no-dotenv",
    "scripts/issue560_crossrecipe_panel.py": "import-order",
    "scripts/issue560_transfer_analysis.py": "no-dotenv",
    "scripts/issue589_estimator_sweep.py": "no-dotenv",
    "scripts/issue589_figures.py": "no-dotenv",
    "scripts/issue594_analyze_context_geometry.py": "import-order",
    "scripts/issue594_build_probes_ultrachat.py": "import-order",
    "scripts/issue594_extract_context_vectors.py": "import-order",
    "scripts/issue594_fig_hero_embeddings_clean.py": "no-dotenv",
    "scripts/issue604_adapter_svd.py": "import-order",
    "scripts/issue604_analyze.py": "import-order",
    "scripts/issue604_extract_context_vectors.py": "import-order",
    "scripts/issue604_figures.py": "no-dotenv",
    "scripts/issue604_figures_analyzer_fixes.py": "no-dotenv",
    "scripts/issue604_topk_subspace.py": "no-dotenv",
    "scripts/issue611_split_analysis.py": "no-dotenv",
    "scripts/issue611_split_figures.py": "no-dotenv",
    "scripts/issue617_cluster.py": "import-order",
    "scripts/issue617_figures.py": "import-order",
    "scripts/issue617_score_separability.py": "import-order",
    "scripts/issue621_checkpoint_ladder.py": "no-dotenv",
    "scripts/issue623_analyze.py": "import-order",
    "scripts/issue623_extract_sycophancy_vector.py": "import-order",
    "scripts/issue623_loo_leverage.py": "no-dotenv",
    "scripts/issue623_persona_panel_vectors.py": "import-order",
    "scripts/issue632_plots.py": "no-dotenv",
    "scripts/issue634_extract_behavior_vectors.py": "import-order",
    "scripts/issue634_joint_geometry.py": "import-order",
    "scripts/issue637_heldout_predictive_test.py": "no-dotenv",
    "scripts/issue637_heldout_predictive_test_plot.py": "no-dotenv",
    "scripts/issue638_install_resistance.py": "no-dotenv",
    "scripts/issue644_functional_form.py": "no-dotenv",
    "scripts/issue644_loaders.py": "no-dotenv",
    "scripts/issue644_seed_scatter_figure.py": "no-dotenv",
    "scripts/issue648_analyzer_figures.py": "no-dotenv",
    "scripts/issue648_centered_vs_raw_predictive_skill.py": "no-dotenv",
    "scripts/issue649_extract_panel_earlylayer.py": "no-dotenv",
    "scripts/issue649_hero_replot.py": "no-dotenv",
    "scripts/issue649_level_change_decomp.py": "no-dotenv",
    "scripts/issue650_analyze.py": "import-order",
    "scripts/issue650_concept_direction.py": "import-order",
    "scripts/issue650_extract_context_bank.py": "import-order",
    "scripts/issue650_plots.py": "no-dotenv",
    "scripts/issue651_analysis.py": "no-dotenv",
    "scripts/issue651_bridge.py": "import-order",
    "scripts/issue651_canary.py": "import-order",
    "scripts/issue651_depth_robustness.py": "no-dotenv",
    "scripts/issue651_read_layer_grid.py": "no-dotenv",
    "scripts/issue654_analyze.py": "no-dotenv",
    "scripts/issue654_hero_figs.py": "no-dotenv",
    "scripts/issue657_sycophancy_scatter.py": "no-dotenv",
    "scripts/issue658_body_figures.py": "no-dotenv",
    "scripts/issue658_extract_base_store.py": "import-order",
    "scripts/issue658_fit_predictors.py": "import-order",
    "scripts/issue658_genre_delta.py": "import-order",
    "scripts/issue658_genre_delta_figure.py": "no-dotenv",
    "scripts/issue661_analysis.py": "import-order",
    "scripts/issue661_extract_directions.py": "import-order",
    "scripts/issue661_generate_arm_a.py": "import-order",
    "scripts/issue661_replot.py": "no-dotenv",
    "scripts/issue664_aggregate_gate.py": "import-order",
    "scripts/issue664_figures.py": "no-dotenv",
    "scripts/issue664_gate_summary.py": "no-dotenv",
    "scripts/issue665_clean_result_figs.py": "no-dotenv",
    "scripts/issue666_clustered_ci.py": "import-order",
    "scripts/issue666_corpus_extract.py": "import-order",
    "scripts/issue666_designed_null.py": "import-order",
    "scripts/issue666_figures.py": "import-order",
    "scripts/issue666_load_store.py": "import-order",
    "scripts/issue666_lobo_loco.py": "import-order",
    "scripts/issue666_noise_floor.py": "import-order",
    "scripts/issue666_predictor.py": "import-order",
    "scripts/issue667_a36_recovery_forest_regen.py": "no-dotenv",
    "scripts/issue667_alllayer_analysis.py": "import-order",
    "scripts/issue667_analysis.py": "import-order",
    "scripts/issue667_context_answer_figures.py": "no-dotenv",
    "scripts/issue667_deltac_probe.py": "no-dotenv",
    "scripts/issue667_extract.py": "import-order",
    "scripts/issue667_figures.py": "no-dotenv",
    "scripts/issue667_marker_mapchange.py": "import-order",
    "scripts/issue667_pertoken_context_extract.py": "import-order",
    "scripts/issue667_pertoken_extract.py": "import-order",
    "scripts/issue667_pertoken_figures.py": "no-dotenv",
    "scripts/issue667_save_maps.py": "import-order",
    "scripts/issue673_assert.py": "no-dotenv",
    "scripts/issue673_gpu_memory_validation.py": "import-order",
    "scripts/issue683_key_ablation_score.py": "no-dotenv",
    "scripts/issue685_assistant_excluded_recompute.py": "import-order",
    "scripts/issue685_compute_metrics.py": "no-dotenv",
    "scripts/issue685_extract_shifts.py": "import-order",
    "scripts/issue685_figures_r2.py": "import-order",
    "scripts/issue685_figures_r2_addenda.py": "import-order",
    "scripts/issue685_figures_r2_lens11.py": "no-dotenv",
    "scripts/issue685_judge_validity.py": "import-order",
    "scripts/issue685_known_directions.py": "import-order",
    "scripts/issue685_make_figures.py": "no-dotenv",
    "scripts/issue685_matched_position_u.py": "import-order",
    "scripts/issue685_signed_cosine_null.py": "import-order",
    "scripts/issue722_bootstrap.py": "no-dotenv",
    "scripts/issue722_extract_fact_rb.py": "import-order",
    "scripts/issue722_figures.py": "no-dotenv",
    "scripts/issue722_fit_M.py": "import-order",
    "scripts/issue722_load_activations.py": "no-dotenv",
    "scripts/issue722_tf_margin_extract.py": "no-dotenv",
    "scripts/issue734_figures.py": "no-dotenv",
    "scripts/issue744_analyze_continuity.py": "import-order",
    "scripts/issue744_dump_and_stream.py": "import-order",
    "scripts/issue744_make_figures.py": "no-dotenv",
    "scripts/issue744_make_figures_arms.py": "no-dotenv",
    "scripts/issue744_make_figures_supp.py": "no-dotenv",
    "scripts/issue778_plots.py": "no-dotenv",
    "scripts/issue779_collect.py": "import-order",
    "scripts/issue779_extract_rb.py": "import-order",
    "scripts/issue779_percontext_recon.py": "no-dotenv",
    "scripts/issue779_stage1.py": "import-order",
    "scripts/issue_240_hero.py": "no-dotenv",
    "scripts/issue_240_hero_v2.py": "no-dotenv",
    "scripts/issue_331_phase0_panel.py": "import-order",
    "scripts/issue_381_make_figures.py": "no-dotenv",
    # END GENERATED
    **GRANDFATHERED_895,
    **GRANDFATHERED_1187,
}


# Roots that pull torch/numpy (and freeze the BLAS/intra-op pools) at import
# time. Extended beyond literal torch/numpy by #1146: importing any of these
# before load_dotenv() leaves the #847 shared-VM thread caps dead.
HEAVY_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "torch",
        "numpy",
        "matplotlib",
        "pandas",
        "scipy",
        "sklearn",
        "seaborn",
        "transformers",
        "datasets",
        "statsmodels",
        "vllm",
        "peft",
        "trl",
    }
)


def _first_heavy_import_line(tree: ast.Module) -> int | None:
    """Earliest MODULE-LEVEL import line of a transitively-heavy root (HEAVY_IMPORT_ROOTS)."""
    earliest: int | None = None
    for node in tree.body:  # module top level only — nested imports are lazy by design
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            if name.split(".")[0] in HEAVY_IMPORT_ROOTS and (
                earliest is None or node.lineno < earliest
            ):
                earliest = node.lineno
    return earliest


def _first_load_dotenv_line(src: str) -> int | None:
    for lineno, line in enumerate(src.splitlines(), start=1):
        if "load_dotenv(" in line:
            return lineno
    return None


def _module_top_main_guard(tree: ast.Module) -> bool:
    """True iff the module has a top-level ``if __name__ == "__main__":`` block."""
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        if not all(isinstance(op, ast.Eq) for op in node.test.ops):
            continue  # equality only — an `__name__ != "__main__"` guard is not an entrypoint
        operands = [node.test.left, *node.test.comparators]
        has_name = any(isinstance(o, ast.Name) and o.id == "__name__" for o in operands)
        has_main = any(isinstance(o, ast.Constant) and o.value == "__main__" for o in operands)
        if has_name and has_main:
            return True
    return False


def _scan_targets(root: Path) -> list[Path]:
    """Unified #847-invariant target set (widened by #1187; TRACKED files only).

    * ``scripts/**/*.py`` — every tracked script, recursive: scripts/ is the
      entrypoint directory by convention, and straight-line module-level
      scripts WITHOUT a __main__ guard are still executed as processes
      (probe: 29/209 newly-covered top-level offenders had no guard).
    * ``src/explore_persona_space/experiments/**/*.py`` — ONLY files with a
      module-top __main__ guard: a library module imported by a capped
      entrypoint inherits the caps in-process; the invariant binds files
      EXECUTED as processes (python -m entrypoints, subprocess workers).

    Filtered to git-TRACKED files: an untracked stray can never be
    grandfathered (the currency tests reject untracked entries), so an
    untracked violator would wedge repo-root suite runs while staying
    invisible in worktree clones; every enforcement surface (Step 9c, the
    Step-10d merge gate, trunk) runs on committed state anyway.
    """
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )

    def _is_tracked(p: Path) -> bool:
        return p.relative_to(root).as_posix() in tracked

    scripts = [p for p in sorted(root.glob("scripts/**/*.py")) if _is_tracked(p)]
    experiments = [
        p
        for p in sorted(root.glob("src/explore_persona_space/experiments/**/*.py"))
        if _is_tracked(p) and _module_top_main_guard(ast.parse(p.read_text()))
    ]
    return scripts + experiments


def test_no_new_torch_before_dotenv_vm_entrypoints() -> None:
    """No NEW VM CPU entrypoint may import a heavy root before load_dotenv().

    The env.py thread-cap hook binds in-process only when ``load_dotenv()``
    runs before the import that (transitively) freezes the BLAS/intra-op
    pools (#847 incident: the offender scripts all violated this; #1146
    extended the predicate from literal torch/numpy to every
    HEAVY_IMPORT_ROOTS root; #1187 widened the target set to ALL tracked
    ``scripts/**/*.py`` plus ``__main__``-guarded src/experiments modules —
    the two class rules live in ``_scan_targets``). Existing violators are
    frozen above; this FAILs only on new violations — including branch-side
    violators at their merges.
    """
    root = Path(__file__).resolve().parents[1]
    targets = _scan_targets(root)
    assert targets, "scan target globs matched nothing — repo layout changed?"

    violations: list[str] = []
    for path in targets:
        src = path.read_text()
        heavy = _first_heavy_import_line(ast.parse(src))
        if heavy is None:
            continue  # no module-top torch/numpy import
        rel = path.relative_to(root).as_posix()
        dotenv = _first_load_dotenv_line(src)
        if (dotenv is None or heavy < dotenv) and rel not in GRANDFATHERED_TORCH_BEFORE_DOTENV:
            violations.append(
                f"{rel} (module-top heavy import at line {heavy}, "
                f"first load_dotenv( at line {dotenv})"
            )
    assert not violations, (
        "NEW heavy-import-before-load_dotenv VM entrypoint(s) — call "
        "explore_persona_space.orchestrate.env.load_dotenv() BEFORE importing any "
        "HEAVY_IMPORT_ROOTS root so the shared-VM thread caps (#847) bind in-process:\n  "
        + "\n  ".join(violations)
    )
    # The #847 incident offender was FIXED, not grandfathered — keep it that way.
    assert "scripts/issue778_null_battery.py" not in GRANDFATHERED_TORCH_BEFORE_DOTENV


def _assert_block_entries_current(root: Path, block: dict[str, str], epoch: str) -> None:
    """Shrink-only currency loop shared by the per-epoch grandfather tests.

    Each ``block`` entry must (a) exist AND be git-tracked (UNTRACKED files
    are never grandfathered), (b) sit inside the scanner's own target set
    (``_scan_targets`` — the same set test_no_new_torch_before_dotenv_vm_entrypoints
    scans), and (c) CURRENTLY violate the detector — recomputed with the
    test's own helpers — with the recorded reason matching. Asserts on the
    first stale entry; ``epoch`` labels the messages (e.g. ``"#895"``).
    """
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    targets = {p.relative_to(root).as_posix() for p in _scan_targets(root)}
    for rel, reason in block.items():
        path = root / rel
        assert path.exists(), f"{epoch} entry vanished: {rel} — DELETE it from the {epoch} block"
        assert rel in tracked, (
            f"{epoch} entry is UNTRACKED: {rel} — never grandfather untracked files"
        )
        assert rel in targets, f"{epoch} entry is outside the scanner's target set: {rel}"
        src = path.read_text()
        heavy = _first_heavy_import_line(ast.parse(src))
        dotenv = _first_load_dotenv_line(src)
        assert heavy is not None and (dotenv is None or heavy < dotenv), (
            f"{epoch} entry no longer violates the detector: {rel} — root cause fixed; "
            f"DELETE the entry from the {epoch} block (the frozen set only shrinks)."
        )
        derived = "no-dotenv" if dotenv is None else "import-order"
        assert derived == reason, (
            f"{epoch} reason drifted for {rel}: recorded {reason!r}, recomputed {derived!r} — "
            "update the entry deliberately."
        )


def test_grandfathered_895_block_is_current() -> None:
    """Every #895 grandfather entry must remain a LIVE, tracked, still-violating pin.

    The #895 block froze the offenders that accreted while the Step 9c
    selector gap (#895) left this guard unselected. Each entry must (a) exist
    AND be git-tracked (UNTRACKED files are never grandfathered), (b) sit
    inside the scanner's own target set (the same ``_scan_targets`` set
    test_no_new_torch_before_dotenv_vm_entrypoints scans), and (c) CURRENTLY
    violate the detector — recomputed with the test's own helpers — with the
    recorded reason matching. A later root-cause fix (e.g. #779's
    load_dotenv()-before-torch reorder) makes the fixed entry FAIL here until
    it is DELETED from GRANDFATHERED_895 — the frozen set can only shrink.
    Scoped to the #895 block ONLY; the #847 block keeps its frozen semantics.
    """
    root = Path(__file__).resolve().parents[1]
    assert GRANDFATHERED_895, "the #895 block is empty — delete it AND this test together"
    # Merge integrity: every #895 entry actually reached the detector's dict.
    assert set(GRANDFATHERED_895) <= set(GRANDFATHERED_TORCH_BEFORE_DOTENV)
    _assert_block_entries_current(root, GRANDFATHERED_895, "#895")


def test_grandfathered_1187_block_is_current() -> None:
    """Every #1187 grandfather entry must remain a LIVE, tracked, still-violating pin.

    Same shrink-only contract as GRANDFATHERED_895, scoped to the #1187
    widening freeze (scripts/**/*.py beyond issue*_*.py + __main__-guarded
    experiments modules). The frozen set can only shrink.
    """
    root = Path(__file__).resolve().parents[1]
    assert GRANDFATHERED_1187, "the #1187 block is empty — delete it AND this test together"
    assert set(GRANDFATHERED_1187) <= set(GRANDFATHERED_TORCH_BEFORE_DOTENV)
    assert not set(GRANDFATHERED_1187) & set(GRANDFATHERED_895)  # no double-listing
    _assert_block_entries_current(root, GRANDFATHERED_1187, "#1187")


def test_dotenv_wrapper_import_chain_is_heavy_free() -> None:
    """Importing the load_dotenv wrapper must pull NO HEAVY_IMPORT_ROOTS root.

    Subprocess, because the pytest process already holds numpy. If this fires,
    a heavy import crept into orchestrate.env's import chain — the caps would
    die before load_dotenv() in every preamble-fixed script while the AST
    invariant stays green.
    """
    code = (
        "import sys, json; "
        "from explore_persona_space.orchestrate.env import load_dotenv; "
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    loaded = set(json.loads(out.stdout))
    assert not (loaded & HEAVY_IMPORT_ROOTS), sorted(loaded & HEAVY_IMPORT_ROOTS)
