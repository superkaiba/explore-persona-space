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
import os
import subprocess
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
# Recurring guard: no NEW torch/numpy-before-load_dotenv VM entrypoints
# ---------------------------------------------------------------------------

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

# Frozen grandfather allowlist — generated mechanically from the tree state at
# implementation time (same frozen-allowlist pattern as
# JUDGE_PIN_LEGACY_ALLOWLIST). Two freeze epochs:
#   #847 block (inline below) — the pre-guard tree state.
#   #895 block (GRANDFATHERED_895 above, merged in) — offenders that accreted
#     while the Step 9c selector gap (#895) left this guard unselected for
#     scripts/*.py diffs; frozen at map-introduction time so the
#     newly-selectable gate starts green.
# Do NOT add new entries beyond these generated blocks: a new entrypoint that
# imports torch/numpy at module top must call load_dotenv() FIRST so the
# shared-VM thread caps bind in-process (see .claude/rules/code-style.md,
# "Shared-VM CPU thread caps"). Reason legend (one line per entry):
#   "no-dotenv"    — file never calls load_dotenv() (tree state at freeze)
#   "import-order" — module-top torch/numpy import precedes the first
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
}


def _first_heavy_import_line(tree: ast.Module) -> int | None:
    """Earliest MODULE-LEVEL import line of torch or numpy (incl. from-imports)."""
    earliest: int | None = None
    for node in tree.body:  # module top level only — nested imports are lazy by design
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            if name.split(".")[0] in ("torch", "numpy") and (
                earliest is None or node.lineno < earliest
            ):
                earliest = node.lineno
    return earliest


def _first_load_dotenv_line(src: str) -> int | None:
    for lineno, line in enumerate(src.splitlines(), start=1):
        if "load_dotenv(" in line:
            return lineno
    return None


def test_no_new_torch_before_dotenv_vm_entrypoints() -> None:
    """No NEW VM CPU entrypoint may import torch/numpy before load_dotenv().

    The env.py thread-cap hook binds in-process only when ``load_dotenv()``
    runs before the numpy/torch import freezes the BLAS/intra-op pools
    (#847 incident: the offender scripts all violated this). Existing
    violators are frozen above; this FAILs only on new violations —
    including branch-side violators at their merges.
    """
    root = Path(__file__).resolve().parents[1]
    targets = sorted(root.glob("scripts/issue*_*.py")) + sorted(
        root.glob("src/explore_persona_space/experiments/**/run_*.py")
    )
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
                f"{rel} (module-top torch/numpy import at line {heavy}, "
                f"first load_dotenv( at line {dotenv})"
            )
    assert not violations, (
        "NEW torch/numpy-before-load_dotenv VM entrypoint(s) — call "
        "explore_persona_space.orchestrate.env.load_dotenv() BEFORE importing "
        "torch/numpy so the shared-VM thread caps (#847) bind in-process:\n  "
        + "\n  ".join(violations)
    )
    # The #847 incident offender was FIXED, not grandfathered — keep it that way.
    assert "scripts/issue778_null_battery.py" not in GRANDFATHERED_TORCH_BEFORE_DOTENV


def test_grandfathered_895_block_is_current() -> None:
    """Every #895 grandfather entry must remain a LIVE, tracked, still-violating pin.

    The #895 block froze the offenders that accreted while the Step 9c
    selector gap (#895) left this guard unselected. Each entry must (a) exist
    AND be git-tracked (UNTRACKED files are never grandfathered), (b) sit
    inside the scanner's own target set (the same globs
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
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    targets = {
        p.relative_to(root).as_posix()
        for p in (
            list(root.glob("scripts/issue*_*.py"))
            + list(root.glob("src/explore_persona_space/experiments/**/run_*.py"))
        )
    }
    for rel, reason in GRANDFATHERED_895.items():
        path = root / rel
        assert path.exists(), f"#895 entry vanished: {rel} — DELETE it from GRANDFATHERED_895"
        assert rel in tracked, f"#895 entry is UNTRACKED: {rel} — never grandfather untracked files"
        assert rel in targets, f"#895 entry is outside the scanner's target set: {rel}"
        src = path.read_text()
        heavy = _first_heavy_import_line(ast.parse(src))
        dotenv = _first_load_dotenv_line(src)
        assert heavy is not None and (dotenv is None or heavy < dotenv), (
            f"#895 entry no longer violates the detector: {rel} — root cause fixed; "
            "DELETE the entry from GRANDFATHERED_895 (the frozen set only shrinks)."
        )
        derived = "no-dotenv" if dotenv is None else "import-order"
        assert derived == reason, (
            f"#895 reason drifted for {rel}: recorded {reason!r}, recomputed {derived!r} — "
            "update the entry deliberately."
        )
