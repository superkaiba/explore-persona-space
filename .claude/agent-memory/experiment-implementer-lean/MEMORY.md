# experiment-implementer-lean memory index

- [judge_pilot report resume fields](reference_judge_pilot_report_resume_fields.md) — PilotGateReport lacks instrument n_draws (derive n_draws//n_items); key resume dirs on every re-run trigger or the rerun cache-wedges; waivers key-only; r3 enum guard (#2658 E)
- [ruff-fix hook strips leading imports](feedback_ruff_fix_hook_strips_leading_imports.md) — add usages before imports under the PostToolUse autofix hook; prove guard-red in-process via pytest.MonkeyPatch + __signature__ injection (#2658 E r3)
- [Worktree commit + selector vintage](reference_worktree_commit_and_selector_vintage.md) — guard_root_code_commit blocks plain commits even in worktrees (use `git -C "$WT"`); selector/lint output on vintage-pinned worktrees is drift-dominated
- [phase-done lint is segment-scoped](feedback_phase_done_lint_segment_scoped.md) — run_phase-internal redirects invisible to --check-phase-done-reserved; reword own prints + noqa directly above reused terminals (#2224 r5)
- [arm-registry marker grammar](feedback_smoke_arch_arm_registry_grammar.md) — bare source=/file=/n=/members= form only; commands to prose; verify via task.py check-smoke-arch-registry, bare rc (#2224 r5)
- [Inline-round root-commit cert](feedback_inline_round_root_commit_cert.md) — inline_lint_gate.py certifies (manual workflow_lint doesn't); retry_transient-wrap bare hf_hub_download; bg-Bash lint under load (#2054)
- [Shared-worktree partial-stage commit](feedback_shared_worktree_partial_stage_commit.md) — pathspec commit takes WORKING-TREE content and sweeps live siblings; apply --cached your hunks, bare-commit verified index (#2658 gJ)
- [Claimed test file: sibling fixture coupling](feedback_claimed_test_file_sibling_fixture_coupling.md) — sibling edits YOUR test's fixture for THEIR uncommitted src; probe HEAD signature + residual-diff before whole-file commit (#2658 F/K)
- [Re-freeze moves pilot membership; downstream frozen artifacts go quietly stale](feedback_refreeze_moves_pilot_membership_downstream_frozen_artifacts.md) — diff selection ids old-vs-new; split membership- vs sha-stale; name needs-refreeze per artifact (#2658 D)
