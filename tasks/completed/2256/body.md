---
title: 'Step 10d lint-gate single-flight probe --pattern ''issue-<N>-lint-gate-tree''
  has CLEAR windows under the #2115 script-file launcher — completion-read false-reads
  a live gate as died'
kind: infra
tags: []
created_at: '2026-08-12T22:19:08Z'
has_clean_result: false
origin_prompt: workflow-fix candidate auto-filed by the /issue 2249 orchestrator session
  after the probe-keyed Monitor false-reported a healthy detached lint gate as died
  (2026-08-12)
workflow: v1
---
Surfaced by the /issue 2249 session (2026-08-12, Step 10d pre-push lint gate): the completion-read rule in .claude/skills/issue/SKILL.md § Pre-push workflow-lint gate says 'Missing verdict file + probe CLEAR = the detached run died before writing a verdict — treat as gate-not-run, fail CLOSED', and the single-flight probe pattern is 'issue-<N>-lint-gate-tree'. That coverage claim ('the gate-tree path rides the whole background call's argv, so the pattern is exact-issue-scoped') was true for the original inline bash -c form but is FALSE under the now-mandated #2115 script-file launcher (compose /tmp/issue-<N>-lint-gate.sh with the Write tool; launch 'bash <script>'): the gate-tree token appears in child argv only during the archive/tar and lint-leg phases. During the TG mapped-test legs (pytest in $WT / $REPO_ROOT — minutes long) and the normalize/verdict tail, NO live process carries the token, so a probe-keyed Monitor false-fires 'probe CLEAR + verdict MISSING = died' on a healthy mid-run gate. Observed live: issue-2249's probe-keyed wait false-reported death at ~15 min while pid 1222816 was healthy inside the TG legs (baseline+gated lint legs complete, tg-map written). A session following the completion-read verbatim would kill-before-relaunch a HEALTHY gate (the #1606-class clobber this probe exists to prevent). Also worth an explanatory note: bash '( ... )' subshells inherit the parent argv in /proc cmdline, so the workload script pid appears twice in a pgrep of the script path — not a duplicate launch. Fix direction (pick at plan time): (a) key the lint-gate single-flight probe + completion-wait on the workload SCRIPT path pattern ('issue-<N>-lint-gate[.]sh') or the recorded pid instead of the gate-tree token, updating the SKILL.md § Pre-push workflow-lint gate probe invocation + completion-read prose + the kill-arm pgrep pattern; or (b) make the launcher thread the gate-tree token into the whole workload's argv (e.g. an inert env/arg 'GATE_TREE=/tmp/issue-<N>-lint-gate-tree' on the bash invocation) so the existing probe pattern regains full coverage. Target: .claude/skills/issue/SKILL.md § Pre-push workflow-lint gate (probe + completion-read + kill-arm), scripts/step9c_baseline.py probe --pattern docs if they restate the coverage claim.
