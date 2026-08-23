---
name: Delta-scoped rounds beyond r3 — compose, don't hard-fail
description: Spec now accepts rounds 1-10 (workflow.yaml round_cap_per_reviewer 10; was 1-5 per #1017, 1-3 before that); malformed (<=0, >10, non-integer) refused. Retains the delta-composition recipe for r4+ delta-scoped briefs.
type: feedback
---

When the orchestrator brief requests `revision_round` 4+ as an explicitly
delta-scoped re-review (e.g. #952 r4, 2026-07-04: r3 Claude PASS vs r3 Codex
needs_targeted_fix -> reconciler binding REVISE -> r4 delta on the one fix),
COMPOSE the prompt rather than refusing.

**Why:** the per-reviewer cap has been RAISED twice — 1-3 originally, 1-5 via
#1017, and 1-10 as of the current workflow.yaml § ensemble_review
(`round_cap_per_reviewer: 10`; verified against the live agent spec
2026-08-20, #823 r8 dispatch). A reconciler-bound REVISE naturally produces
r4+ re-verifies. Hard-failing a deliberate, well-formed dispatch burns the
round and forces single-Claude fallback exactly when the cross-family
re-check matters most. Refusal is reserved for genuinely malformed rounds
(<= 0, > 10, non-integer). Do not trust a remembered cap — re-read the
current spec's rule 1 each time; the cap drifts upward.

**How to apply:** for r4+ briefs, (1) state the assumption in the return;
(2) scope the composed prompt to the delta the brief names (verify THIS fix
only; no re-litigating settled items; new findings only if conclusion-relevant
/ spec-breaking; round-N quoting rule for applied/absent claims) — but when
the brief carries a FULL focus-question set rather than a narrow delta (e.g.
#823 r8 fold consolidation), compose the full 15-lens review with the focus
questions as an added REVIEW CONTEXT block, not a narrowed lens roster;
(3) the head sentinel carries the brief's round number
(`epm:clean-result-critique-codex v<round>`) while the POSTED top-level
version is auto max+1 on the codex kind's own history — an explicit
`revision_round` in the brief wins over own-kind marker-history inference
(see [[fold-round-context-file-briefs]] for the inference case when the
brief omits the round);
(4) COMPOSER ATTESTATIONS convert would-be sandbox-unverifiable checks
into Codex-checkable facts (#823 r9): when a fix under verification cites
a pin, pre-verify it on the VM at compose time and attest in the prompt —
blob-identity of worktree copies vs body pins (`git hash-object` match,
extends the fold-memory npz recipe to figures), and HF-pin LIVENESS via an
authenticated `list_repo_tree` at the exact revision (the sandbox has no
network; without the attestation a wrong-pin check degrades to advisory).
Scope the attestation narrowly ("treat liveness as established; your job
is the CONSISTENCY adjudication") so Codex still owns the judgment.
(5) DELTA-CONFINEMENT MIRRORS carry capture artifacts — run the diff
yourself at compose time and attest the expected hunk set (#2477 r3,
2026-08-23): a /tmp round-N body mirror may have been captured WITHOUT the
YAML frontmatter block (marker/file dumps often strip it), so a naive
body-vs-mirror diff shows a spurious leading `---` hunk the twin would
misread as an out-of-delta change. Also extract any brief-pinned
ground-truth artifact (`git -C <worktree> show <sha>:<path>` to /tmp) at
compose time and pass the /tmp path with a blob-identity attestation —
the sandbox may deny git, and a delta commit that is ancestor-of-worktree
+ on origin/issue-<N> but NOT yet on origin/main is the NORMAL pre-merge
state: attest it so the twin doesn't FAIL footer-link liveness on it.
