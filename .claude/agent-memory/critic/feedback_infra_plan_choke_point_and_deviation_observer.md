---
name: Infra-plan choke-point + deviation-observer checks
description: For kind:infra plans — grep-verify choke-point call-site enumerations yourself, and check that pod-emitted deviation events (sentinel JSONL/log) name an OBSERVER surface, since pod code can't post markers
type: feedback
---

Two recurring checks for `kind: infra` plan reviews (surfaced on #564, HF storage headroom):

1. **Choke-point claims are grep-checkable — do it.** When a plan claims "function X is THE funnel for behavior Y" (e.g. `upload_model` as the LFS-upload funnel), grep the call sites yourself and compare against the plan's enumeration. #564's plan matched exactly (runner.py:293/307, trainer.py:539/585, sft.py:1251, train_cell.py:103 + ~9 frozen per-issue scripts documented as non-goal) — when it matches, that's strong APPROVE evidence; when it doesn't, the missed caller is a silent routing hole.
2. **Deviation events emitted pod-side need a named observer.** Pod-side library code cannot post `epm:` markers (hard rule), so "plan-deviation note posted when it fires" ACs get satisfied by log-line + sentinel JSONL emission with marker-posting assigned to "the orchestrator". Check whether ANY surface is actually changed to observe the sentinel (upload-verifier checklist, poll_pipeline, or at least the doc paragraph naming the path). Emission without a wired/documented observer = event nothing reads. Usually a Concern (doc edit already in scope), not Must-Fix.

**Why:** infra plans live or die on integration-point correctness, not statistics; the cheap independent verification is grep + reading the cited lines.
**How to apply:** any infra/code-change plan claiming a single funnel/choke point or a pod→orchestrator event channel.
