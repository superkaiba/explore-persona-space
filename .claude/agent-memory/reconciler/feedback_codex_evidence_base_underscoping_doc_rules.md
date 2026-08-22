---
name: codex-evidence-base-underscoping-doc-rules
description: Codex FAILs doc/rule-file insertions as "unsupported observations" by checking only ONE designated evidence row, ignoring the plan's §Evidence-of-record (code-grounded values, second-hand-report mechanism framing) — verify each ground the plan authorizes before crediting the blocker (#2338 r1)
metadata:
  type: feedback
---

On a doc-only rule-file diff (gotchas.md-class), Codex's evidence-grounding
blockers can rest on an artificially NARROW evidence base: it treats the one
designated events.jsonl row as "the sole authorized evidence" and flags every
clause that row doesn't literally contain. The approved plan's **§Evidence of
record / §Provenance** may authorize MORE grounds — check each before
crediting the blocker.

**Why (#2338 r1, Codex FAIL overturned):** Codex flagged (a) "1 GB" canary
size — but the plan explicitly code-grounded it (`orchestrate/preflight.py`
`probe_gb: float = 1.0`, verified at line 909); (b) "The grinding thread
shows the same `wchan=request_wait_answer` as the wedge" as an unsupported
observation — but the sentence is generic-PRESENT-tense mechanism prose (the
file's house register; the #2333 citation attaches only to the marker-grounded
first sentence), mechanism-true (any outstanding FUSE request waits in
`request_wait_answer`), second-hand-grounded in the task body's experimenter
report, and its safety direction is the OPPOSITE of Codex's claimed impact —
it says wchan CANNOT clear a wedge and forces spot probes + a 2× escalation
bound; (c) "reinstall skipped" — tightly entailed by the marker (uninstalled +
verified-UNUSED as final pre-launch env state on a launch record; no later
marker records a reinstall).

**How to apply:**
1. Read the plan's §Evidence-of-record/§Provenance FIRST — it may name code
   files (grep the constant yourself) and second-hand reports with an explicit
   framing contract ("presented as recommended mechanism probe, never as
   observed measurement"). Judge the DIFF text against THAT contract.
2. Distinguish register: generic present tense ("X shows Y, so discriminate
   by...") in a mechanism-describing rule file is a mechanism claim, not a
   past-tense incident observation; check whether the incident citation
   grammatically attaches to the disputed clause.
3. Run the impact arrow yourself: a clause that WARNS a signal is
   non-discriminative cannot cause the misclassification Codex attributes to
   it if the surrounding text supplies discriminating probes + an escalation
   bound.
4. When Codex concedes "Plan adherence: COMPLETE" and its fix requires a plan
   revision, it is re-litigating critic-approved plan text at code review —
   the [[codex-overreads-plan-prose]] / methodology-choice-as-bug family;
   FAIL only if the mandated text is genuinely false or unsafe.
