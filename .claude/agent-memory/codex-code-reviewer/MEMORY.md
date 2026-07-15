# Codex Code Reviewer Memory

<!-- This file is the MEMORY.md index for the codex-code-reviewer agent. -->

- [gh_graphql fallback to REST](feedback_gh_graphql_fallback.md) — gh_graphql MCP not in project context; use `gh api -X POST ...` REST fallback when GraphQL rate-limited (separate 5000/hr quota)
- [scripts import chain pattern](feedback_scripts_import_chain.md) — scripts/*.py that use _bootstrap cannot be imported as namespace packages; use sys.path insertion
- [mask audit threshold arm-gating](feedback_mask_audit_partial_vs_whole.md) — pct_masked >= 80% only valid for partial-generation arms; whole-turn arms (~47% masked) will abort on this threshold
- [Hub path vs local disk trainer path](feedback_hub_path_vs_local_disk.md) — run_issue_344_train.py loads from local disk only; Hub uploads are provenance, not trainer's read path; fix scripts that only rewrite local VM files leave the pod with stale data
- [contextmanager del llm ref](feedback_contextmanager_del_llm_ref.md) — del inside @contextmanager finally doesn't free caller's 'as llm' binding; gc.collect() runs while LLM still referenced (task #365 r10 Major finding)
- [concurrent follow-ups → wrong plan symlink + latest-marker + HEAD](feedback_concurrent_followups_wrong_plan_symlink.md) — with >1 concurrent same-issue follow-up round, plan.md symlink AND latest-marker (impl + smoke-arch) AND branch HEAD all resolve to a DIFFERENT round; round-match ALL inlined inputs by followup_label + note body, and scope Codex to `git show <round-sha>` not main...HEAD (#841)
- [9a-ter rounds: no impl marker → report placeholder](feedback_9ater_followup_round_report_placeholder.md) — free-analysis follow-up reviews inline the stage-dispatch marker + an orchestrator-filled `{{followup_implementer_report_body}}` placeholder; adapt gates 0.5/0.55/0.6; never fetch the prior round's impl marker (#920 r3)
