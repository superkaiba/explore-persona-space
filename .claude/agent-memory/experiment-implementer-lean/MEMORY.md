# experiment-implementer-lean memory index

- [Worktree commit + selector vintage](reference_worktree_commit_and_selector_vintage.md) — guard_root_code_commit blocks plain commits even in worktrees (use `git -C "$WT"`); selector/lint output on vintage-pinned worktrees is drift-dominated
- [phase-done lint is segment-scoped](feedback_phase_done_lint_segment_scoped.md) — run_phase-internal redirects invisible to --check-phase-done-reserved; reword own prints + noqa directly above reused terminals (#2224 r5)
- [arm-registry marker grammar](feedback_smoke_arch_arm_registry_grammar.md) — bare source=/file=/n=/members= form only; commands to prose; verify via task.py check-smoke-arch-registry, bare rc (#2224 r5)
- [Inline-round root-commit cert](feedback_inline_round_root_commit_cert.md) — inline_lint_gate.py certifies (manual workflow_lint doesn't); retry_transient-wrap bare hf_hub_download; bg-Bash lint under load (#2054)
