---
name: delta-scoped-rounds-and-round-cap
description: Round cap is 10 (workflow.yaml round_cap_per_reviewer; was 5 under #1017) — malformed = <=0, >10, non-integer; retains the delta-composition recipe for reconciler-bound delta-scoped briefs (r4+).
metadata:
  type: feedback
name: Delta-scoped rounds beyond r3 — compose, don't hard-fail
description: Spec now accepts rounds 1-10 (workflow.yaml round_cap_per_reviewer 10; was 1-5 per #1017, 1-3 before that); malformed (<=0, >10, non-integer) refused. Retains the delta-composition recipe for r4+ delta-scoped briefs.
type: feedback
---

When the orchestrator brief requests a `revision_round` above 3 as an
explicitly delta-scoped re-review (e.g. #952 r4, 2026-07-04: r3 Claude PASS
vs r3 Codex needs_targeted_fix -> reconciler binding REVISE -> r4 delta on
the one fix), COMPOSE the prompt rather than refusing.
When the orchestrator brief requests `revision_round` 4+ as an explicitly
delta-scoped re-review (e.g. #952 r4, 2026-07-04: r3 Claude PASS vs r3 Codex
needs_targeted_fix -> reconciler binding REVISE -> r4 delta on the one fix),
COMPOSE the prompt rather than refusing.

**Why:** the ensemble policy caps the four iterating sites at 10 rounds per
reviewer (workflow.yaml § ensemble_review `round_cap_per_reviewer: 10`;
verified against the live agent spec 2026-08-25 — the earlier #1017 cap of 5
is STALE), and a reconciler-bound REVISE naturally produces an r4+ re-verify.
Hard-failing a deliberate, well-formed dispatch burns the round and forces
single-Claude fallback exactly when the cross-family re-check matters most.
Refusal is reserved for genuinely malformed rounds (<= 0, > 10, non-integer).
Sibling precedent: codex-critic's `feedback_delta_scoped_amendment_rounds.md`.
**Why:** the per-reviewer cap has been RAISED twice — 1-3 originally, 1-5 via
#1017, and 1-10 as of the current workflow.yaml § ensemble_review
(`round_cap_per_reviewer: 10`; verified against the live agent spec
2026-08-20, #823 r8 dispatch). A reconciler-bound REVISE naturally produces
r4+ re-verifies. Hard-failing a deliberate, well-formed dispatch burns the
round and forces single-Claude fallback exactly when the cross-family
re-check matters most. Refusal is reserved for genuinely malformed rounds
(<= 0, > 10, non-integer). Do not trust a remembered cap — re-read the
current spec's rule 1 each time; the cap drifts upward.

**How to apply:** for delta-scoped briefs (typically r4+), (1) state the
assumption in the return; (2) scope the composed prompt to the delta the
brief names (verify THIS fix only; no re-litigating settled items; new
findings only if conclusion-relevant / spec-breaking; round-N quoting rule
for applied/absent claims); (3) keep the marker version = the round number
(`epm:clean-result-critique-codex v<n>`). A brief with no delta scope note
at r4+ gets the normal full-prior-history re-review. Still refuse genuinely
malformed rounds (0, negative, >10, non-integer). See also
[[compose-recipe-lens-ref-replacements]].

**Delta-artifact pinned-blob extraction (confirmed #2378 r4, 2026-08-25):**
a fold's NEW artifacts (figure PNG, eval JSONs) usually exist only at
issue-branch pinned SHAs — NOT on main's working tree — so the sandboxed
read-only Codex may have no path to them (network denied; `git show` may be
denied; /tmp may not be writable from its side). At compose time, extract
the body-pinned blobs YOURSELF to /tmp (`git show <sha>:<path> > /tmp/...`
from the canonical main odb — worktrees share it, so issue-branch objects
resolve) and hand Codex the /tmp paths as "DELTA ARTIFACTS (compose-time
pinned-blob extractions)", stating the pin SHA + byte count. Keep read-only
git permitted as the spot-check fallback (per-leg JSONs), with the
sandbox-unverifiable (advisory) downgrade if git is denied. Also verify the
pinned refs exist in the odb at compose time (`git cat-file -e`) and say so
in the prompt — that grounds the network-advisory clause for github links.

**Verification-round-after-binding-reconcile shape (confirmed #2378 r3,
2026-08-25):** these can arrive at ANY round >= 3, not just r4+. Compose
the FULL fifteen-lens prompt (not a bare delta) PLUS: (1) the reconciler
verdict temp-file path as a REQUIRED-reading header input; (2) a "ROUND N
SCOPE" block with (a) the binding fixes to verify AGAINST GROUND TRUTH
(quote the reconciler's git-show line refs; permit read-only git for the
check), (b) a regression check (word caps, do-not-touch items), (c) the
DISCARDED/settled item list inlined with an explicit no-re-raise rule
(near-duplicates included, absent NEW evidence the reconciler lacked);
(3) a "### Binding-fix verification" section at the TOP of the output
template (VERIFIED|NOT-VERIFIED|FAIL per fix + regression line); (4) a
note in the Concerns section not to re-persist the settled concerns; (5)
an explicit "if (a)+(b) verify clean and no genuine new violation exists,
PASS is the honest verdict" line — counteracts nit-manufacturing on a
3-rounds-deep body. Extend the HF network-advisory clause to github.com
/tree links when a binding fix converted plain SHAs to links (resolved
404 stays a real FAIL). Sanity-grep the canonical body for the claimed
fix text BEFORE composing so the scope statements are accurate.
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
SPLIT-PIN case (#823 r7): when a revision re-pins ONE artifact to a new
commit (fig8 re-render) while siblings stay at the old pin, attest each
file against ITS OWN pin, verify the re-pinned file's blobs DIFFER
between pins, and say the split explicitly in the prompt — a naive
single-pin attestation (or a twin checking all files against one pin)
false-FAILs the re-pinned artifact.
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
