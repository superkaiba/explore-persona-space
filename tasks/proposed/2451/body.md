---
title: 'Teammate coordination: a SendMessage resume re-arms liveness — prior completion
  evidence must not license editing a teammate''s file set'
kind: infra
tags: []
created_at: '2026-08-21T13:52:44Z'
has_clean_result: false
workflow: v1
---
# Teammate coordination: a SendMessage that resumes a completed agent RE-ARMS its liveness — prior completion evidence no longer licenses touching its file set

## Goal

Close a gap in the always-on teammate-coordination contract (CLAUDE.md
§ "Orchestrator vs subagent re-invocation", the "Teammate coordination" bullet,
clauses (b) and (e)). As written, the contract licensed a one-writer-per-file-set
violation that then required a teammate to repair a defect in the orchestrator's
edit.

## The gap

The contract says:

> A teammate is LIVE until its terminal result arrives OR durable completion
> evidence exists (a landed commit covering its assigned scope, a posted marker,
> its final Agent result) — a self-reported WIP state is neither.

That test is **point-in-time** and has no clause for what happens when the
orchestrator itself makes a completed agent live again. `SendMessage` to a
COMPLETED agent does not queue a note for later reading — it **resumes the agent
in the background** for a fresh turn (the tool result says so explicitly:
"was stopped (completed); resumed it in the background with your message. You'll
be notified when it finishes"). From that moment the agent owns its file set
again, and every piece of completion evidence gathered BEFORE the resume — the
final Agent result, the landed commit, a clean working tree — is stale.

Nothing in the current wording says this. An orchestrator can read the rule
correctly, enumerate genuine durable evidence, and still be wrong, because the
evidence predates the resume it performed itself.

## The incident (#2329, `q35_ladder_decay` report round 2)

1. The orchestrator dispatched a `plotter` with 3 caption fixes; it completed and
   its 3-fix commit landed.
2. A 4th caption defect was then traced to the same file. Correctly declining to
   edit a teammate-owned file, the orchestrator sent the fix via `SendMessage` —
   which **resumed** the agent.
3. A notification arrived carrying the earlier 3-fix summary. The orchestrator
   probed durable state, found the fix absent, the commit landed, the working
   tree clean, and the final result in hand — and concluded, citing the rule's
   own disjunction, that the scope was durably complete and applied the edit
   itself.
4. The resumed turn was in fact live. It found the orchestrator's commit,
   correctly declined to duplicate it, and **repaired a real defect in it**: the
   hand-applied wording left the caption bullet's parentheses unbalanced (8
   opens / 7 closes; the outer "(per-cell means, ..." group never closed). It
   then flagged the double-dispatch and asked that the file be given one owner.

No work was lost — the file never diverged, because the orchestrator's tree was
clean when the commit landed and the teammate chose repair over overwrite. But
the ordering was luck, not design: a concurrent write in the other order loses an
edit, and the downstream cost was real (the companion doc and the report draft
had both been assembled from the pre-balance caption text and had to be
regenerated and re-committed).

## Proposed fix

Add an explicit re-arm clause to the teammate-coordination bullet, in the same
place the LIVE test is defined. Substance to encode:

1. **A resume RE-ARMS liveness.** Sending a message to a completed agent makes it
   LIVE again for its file set. Completion evidence gathered before that send is
   void; the orchestrator must wait for the NEW terminal result before touching
   the file set.
2. **A notification is not necessarily the notification you are waiting for.**
   The same task-id notifies more than once (the tool result already documents
   this). After a resume, a notification whose content does not reflect the
   resumed scope is evidence the queued turn has not reported yet — not evidence
   it declined the work. Key the wait on content, not on arrival.
3. **The safe move when a resumed agent seems idle** is a single nudge on the
   teammate channel, or reading its final Agent result — never a concurrent edit,
   which is exactly what the one-writer-per-file-set rule exists to prevent.
4. Corollary worth stating: if the orchestrator genuinely must take the file over
   (the agent is wedged, the fix blocks a gate), the sanctioned route is to
   force-stop the agent FIRST and record the takeover — the same
   confirm-then-own shape clause (e) already requires for stand-downs.

## Acceptance criteria

1. The teammate-coordination bullet states that a `SendMessage`-driven resume
   re-arms liveness and voids prior completion evidence.
2. It states that after a resume, the orchestrator waits for a terminal result
   whose CONTENT covers the resumed scope.
3. It names the sanctioned takeover route (force-stop + record) for the case
   where waiting is not viable.
4. A worked one-line citation of this incident so the next reader sees the failure
   shape, consistent with how the surrounding clauses cite #1112 / #958 / #2034.
5. `workflow_lint.py` stays green; if any rule-surface pin covers this bullet's
   region, it is updated in the same change.

## Provenance

#2329 round `q35_ladder_decay`, report-verifier round 2 fix pass, 2026-08-21.
Evidence: the plotter's own final report (it flagged the double-dispatch and
named the parenthesis defect it repaired); commits `4305d6b506` (the
orchestrator's hand-applied fix) and `8440be84a5` (the teammate's repair);
`4632a930e8` + `6163155685` (the regeneration the ordering forced). The
orchestrator's stated reasoning at the time cited the rule's durable-evidence
disjunction verbatim, which is why this is filed as a surface gap rather than
only an operator error.

- target_file: CLAUDE.md
- fingerprint: sendmessage-resume-rearms-teammate-liveness
- confidence: high — the rule as written licensed the violation, and the
  teammate independently reported the collision
