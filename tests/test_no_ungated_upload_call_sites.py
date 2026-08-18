"""Pin: no NEW ungated HF upload call site may enter src/ or scripts/.

Background (2026-08-17/18 secret-leak incident): the upload-time secret gate
(``orchestrate/secret_scrub.assert_upload_clean``) is wired into the two
sanctioned upload paths — ``hub._upload`` and
``upload_sharded.upload_dir_sharded``. But the job that triggered the
2026-08-17 HF secret-scanning alert (the #2332 repack) called
``huggingface_hub`` DIRECTLY, bypassing both wrappers. Direct call sites are
therefore the remaining leak path.

This pin freezes the pre-gate world: every file that already contained a
direct ``.upload_file(`` / ``.upload_folder(`` / ``upload_large_folder(`` /
``CommitOperationAdd(`` call on 2026-08-18 is grandfathered BY NAME below.
Any file not on that list that grows a direct upload call must either

  1. route the bytes through ``hub._upload`` / ``hub.upload_dataset*`` /
     ``upload_sharded.upload_dir_sharded`` (preferred — gate included), or
  2. call ``secret_scrub.assert_upload_clean([...], what=...)`` on the
     Hub-bound paths before the direct call (this file then passes the pin
     automatically, because compliance is detected by that call's presence).

The grandfather list may only SHRINK. Do not add to it — that reopens the
leak path this pin exists to close. Historical issue scripts are listed
because retrofitting ~100 finished one-offs is churn without payoff; they
are not an endorsement of the pattern.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

UPLOAD_TOKENS = (
    ".upload_file(",
    ".upload_folder(",
    "upload_large_folder(",
    "CommitOperationAdd(",
)

SANCTIONED = {
    "src/explore_persona_space/orchestrate/hub.py",
    "src/explore_persona_space/orchestrate/upload_sharded.py",
}

# Token mentions that are lint rules / audit tooling, not upload code.
EXCLUDED = {
    "scripts/workflow_lint.py",
    "scripts/issue2332_consumer_audit.py",
}

# Frozen 2026-08-18 — pre-gate direct-upload files. SHRINK ONLY.
GRANDFATHERED = {
    "src/explore_persona_space/backends/gcp.py",
    "src/explore_persona_space/experiments/issue_1072/run_1072.py",
    "src/explore_persona_space/experiments/issue_1072/run_1072_lowdim.py",
    "src/explore_persona_space/experiments/issue_823/run_823.py",
    "src/explore_persona_space/experiments/issue_952/run_952.py",
    "src/explore_persona_space/experiments/leave_one_out_505/build_pv_centroids.py",
    "src/explore_persona_space/experiments/leave_one_out_505/logit_rescoring.py",
    "scripts/archive/upload_and_clean.py",
    "scripts/build_canonical_persona_pool.py",
    "scripts/build_paper.py",
    "scripts/gen_data_appendix.py",
    "scripts/i504_round6_recompute_mean_centered.py",
    "scripts/issue1073_common.py",
    "scripts/issue1092_build_corpus.py",
    "scripts/issue1092_claude_text.py",
    "scripts/issue1092_figures.py",
    "scripts/issue1092_gpu_phase.py",
    "scripts/issue1092_transfer_probe.py",
    "scripts/issue1332_bank_build.py",
    "scripts/issue1332_gpu_phase.py",
    "scripts/issue1345_boundary_ablation_stage_and_mirror.py",
    "scripts/issue1482_context_side_labels.py",
    "scripts/issue1482_g1probe_stage.py",
    "scripts/issue1491_ladder_manifest.py",
    "scripts/issue1689_capture.py",
    "scripts/issue1689_fits_mirror.py",
    "scripts/issue1689_real_u2_upload.py",
    "scripts/issue1689_user_slot_capture.py",
    "scripts/issue1689_user_slot_gen_a1.py",
    "scripts/issue1739_newarm_box.py",
    "scripts/issue1773_describe_axes.py",
    "scripts/issue1773_evidence_builder.py",
    "scripts/issue1773_register_steer_validator.py",
    "scripts/issue1776_upload_batch.py",
    "scripts/issue1902_corpus.py",
    "scripts/issue1902_run.py",
    "scripts/issue1902_steer_probe.py",
    "scripts/issue1902_steer_vectors.py",
    "scripts/issue1941_fr_diag.py",
    "scripts/issue1947_syc_recovery_upload.py",
    "scripts/issue1979_gpu.py",
    "scripts/issue2220_readwrite.py",
    "scripts/issue2222_capture.py",
    "scripts/issue2222_judge.py",
    "scripts/issue2223_drift.py",
    "scripts/issue2254_preimage.py",
    "scripts/issue2330_qwen35_generate_capture.py",
    "scripts/issue540_jsrb_predictor.py",
    "scripts/issue545_sweep.py",
    "scripts/issue588_smoke_artifact.py",
    "scripts/issue594_extract_context_vectors.py",
    "scripts/issue604_extract_context_vectors.py",
    "scripts/issue617_upload_corpus.py",
    "scripts/issue634_extract_behavior_vectors.py",
    "scripts/issue650_extract_context_bank.py",
    "scripts/issue651_dispatch.py",
    "scripts/issue651_drain_extracts.py",
    "scripts/issue658_extract_base_store.py",
    "scripts/issue661_extract_directions.py",
    "scripts/issue661_freeze_instructions.py",
    "scripts/issue661_generate_arm_a.py",
    "scripts/issue664_dispatch.py",
    "scripts/issue667_alllayer_dispatch.py",
    "scripts/issue667_dispatch.py",
    "scripts/issue667_pertoken_context_dispatch.py",
    "scripts/issue667_pertoken_dispatch.py",
    "scripts/issue685_matched_position_u.py",
    "scripts/issue722_extract_fact_rb.py",
    "scripts/issue722_regen_ultrachat_generic.py",
    "scripts/issue744_dump_and_stream.py",
    "scripts/issue745_upload_engagement_smoke.py",
    "scripts/issue763_build_probe_pools.py",
    "scripts/issue763_cofit_upload.py",
    "scripts/issue763_disclosure_flag_audit.py",
    "scripts/issue763_extract_pv_rb.py",
    "scripts/issue763_judge_e0.py",
    "scripts/issue763_upload.py",
    "scripts/issue778_v2_upload.py",
    "scripts/issue779_arm_headline_pod.py",
    "scripts/issue779_capture_answer_summaries.py",
    "scripts/issue779_capture_answer_summaries_pass2.py",
    "scripts/issue779_collect.py",
    "scripts/issue779_edges.py",
    "scripts/issue779_extract_rb.py",
    "scripts/issue779_gen_behavior_corpus.py",
    "scripts/issue779_pertoken_lmsys_capture.py",
    "scripts/issue779_reliability_gen_capture.py",
    "scripts/issue810_common.py",
    "scripts/issue810_extract_positions.py",
    "scripts/issue811_upload_store.py",
    "scripts/issue833_extract_onpolicy.py",
    "scripts/issue920_extract_summaries.py",
    "scripts/issue920_gen_completions_b.py",
    "scripts/issue920_nulls_figures.py",
    "scripts/issue922_common.py",
    "scripts/issue928_common.py",
    "scripts/issue928_extract_thinking_store.py",
    "scripts/issue952_bank_build.py",
    "scripts/issue952_china_topup_gpu.py",
    "scripts/issue952_divtrain_gpu.py",
    "scripts/issue952_noise_ceiling_gpu.py",
    "scripts/issue958_common.py",
    "scripts/issue_642/i642_dispatch.py",
    "scripts/run_experiment_444.py",
}


def _direct_upload_files() -> list[str]:
    hits: list[str] = []
    for d in ("src", "scripts"):
        for p in sorted((REPO_ROOT / d).rglob("*.py")):
            rel = p.relative_to(REPO_ROOT).as_posix()
            if rel in SANCTIONED or rel in EXCLUDED:
                continue
            text = p.read_text(errors="replace")
            if any(t in text for t in UPLOAD_TOKENS) and "assert_upload_clean(" not in text:
                hits.append(rel)
    return hits


def test_no_new_ungated_upload_call_sites():
    offenders = [f for f in _direct_upload_files() if f not in GRANDFATHERED]
    assert not offenders, (
        "NEW direct HF upload call site(s) without the secret gate:\n  "
        + "\n  ".join(offenders)
        + "\nRoute uploads through hub._upload / upload_dir_sharded (gated), or call "
        "secret_scrub.assert_upload_clean(paths, what=...) before the direct call. "
        "Never add to GRANDFATHERED — see this file's docstring."
    )


def test_grandfather_list_only_shrinks():
    """Entries whose files no longer contain ungated uploads (deleted or
    migrated) must be removed from GRANDFATHERED — keeps the list honest."""
    current = set(_direct_upload_files())
    stale = sorted(g for g in GRANDFATHERED if g not in current)
    assert not stale, (
        "GRANDFATHERED entries no longer needed (file deleted or now gated) — "
        "remove them:\n  " + "\n  ".join(stale)
    )
