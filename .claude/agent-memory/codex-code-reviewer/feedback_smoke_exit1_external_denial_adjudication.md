---
name: smoke-exit1-external-denial-adjudication
description: "A smoke phase reporting exit 1 root-caused to an EXTERNAL access denial persisted as an open user-action BLOCKER gets a compose-time both-routes adjudication note (not silence, not pre-decided); GPU-residual labeling variants of the carve-out are steered present-but-imperfect"
metadata:
  type: feedback
---

When the inlined implementation marker's `## Smoke run` carries a phase with a
NON-ZERO exit code that BOTH markers (impl + smoke-arch) root-cause to an
EXTERNAL access denial — gated HF dataset 403, missing credential — already
persisted as an OPEN user-action BLOCKER concern gating pod dispatch (#2546 r1,
2026-08-24: `taur-gated-access-blocked`; staging `--smoke` exit 1 at the
TAUR-Lab gated load, all upstream stages clean, `(d)` call-out + copy-pasteable
re-run command present):

1. **Do not stay silent** — Step 0.6's literal trigger list includes "a
   non-zero exit", so an adversarial twin left alone will FAIL
   `smoke-run-missing` on a disclosed fail-loud firing.
2. **Do not pre-decide either** — add a compose-time-facts note that states the
   verified facts (exit code, root-cause agreement across both markers, the
   persisted BLOCKER id, upstream-stages-clean, the (d) call-out) and hands
   BOTH routes: a FAIL keyed solely on the disclosed, persisted,
   user-action-gated exit-1 rests on presentation of present evidence
   (Step 0.7 rule 1); a FAIL is warranted if the disclosure is INACCURATE
   against the code or a genuinely uncertified phase exists. Pair it with the
   Step 0.8 duty split: verify the code fails LOUD on the denial (no silent
   skip/fallback), never re-raise the already-persisted blocker as a new
   finding.
3. **Carve-out labeling variants:** GPU-bound legs enumerated via a section-
   head pod-P1 certification statement plus per-phase "GPU residual" lines —
   rather than the literal `### <phase> — Carve-out (GPU-bound)` title — are
   steered as the present-but-imperfect case (judge the three substitute-
   coverage items substantively), not absence.

**Why:** the composer is the only party who has verified the cross-marker
root-cause agreement and the ledger state; without the note the twin either
false-FAILs (the #489-class costume: FAILing on evidence that is present) or,
told too firmly, rubber-stamps a disclosure that might not match the code.

**How to apply:** any compose where the smoke section has a non-zero exit tied
to a persisted external-dependency BLOCKER. Also confirmed this round:
worktree plan at the FROZEN-status dir (`tasks/planning/<N>` vs brief's
`running/`) byte-identical to canonical v4 ⇒ path-reference with the corrected
path + an explicit "the brief's running/ path does not resolve here" line.

**Revision-round sibling (#2546 r2, 2026-08-24):** with the external-denial
BLOCKER still open, the round-2 marker showed all-rc=0 VM smokes but the
heavily-reworked staging file (+194/−33 of discovery/dedup/draw logic) had
STRUCTURAL-ONLY evidence (AST + `--help`) — and r1's smoke had proven the
upstream staging stages runnable up to the 403, so a re-run was not fully
fenced. Compose a both-routes Step 0.6 note (uncertified-changed-logic
`smoke-run-missing` route, naming the specific logic, vs disclosed
revision-round CONCERNS route with the blocker fencing the full pipeline) —
never silence, never a pre-decision. Same round: a `(b)` rebuttal citing an
out-of-worktree reference (`common.py:325` in a parent lineage with no such
file here) gets a compose-time probe — state "citation does not resolve in
this worktree" + the nearest in-worktree corroboration (a figure label with
the same value) + the round's own constant site, and hand ACCEPTED/REJECTED
routes keyed on the plan's internal consistency, never resolve it yourself.

**Round-3 sibling (#2546 r3, 2026-08-25):** two deltas on the same lineage.
(1) A rebuttal citation that failed to resolve at round N can RESOLVE at
round N+1 once the implementer sharpens the path — r2's "the #1336 lineage
(common.py:325)" had no in-worktree referent, but r3's full
`src/explore_persona_space/experiments/issue_1336/common.py:325 @ ba8359381c`
resolved at the pinned SHA AND byte-identically in the worktree HEAD
(`DELTA_ELICIT_BAND = 0.02`). Re-probe every round; never carry a prior
round's "citation does not resolve" verdict forward, and when it resolves,
hand the twin BOTH the resolved constant and the plan's own contrary
literals (grep the plan for 0.021/0.0207 anchors) with routes, not a ruling.
(2) On a FAIL+FAIL fix round where the prior Codex verdict's findings were
ALL forwarded as persisted concerns, key the closure contract on the r2
verdict's own `CONCERN::` row texts (sharper than the ledger `raised`
summaries for re-raised ids — e.g. the trajectory row carried the
fixture-residual nuance only in the verdict row), quoted WITHOUT the machine
token; the r2 exit-1 staging critique then converts into a compose-time fact
that the r3 REAL staging slice ran to the credential wall (rc=1 at the gated
load = precisely what the r2 Critical demanded) — both-routes note again,
never silence. Also: the disclosed rc=1 real-slice run coexisting with six
rc=0 selftests is the healthy shape, not a contradiction.

**Round-4 sibling (#2546 r4, 2026-08-25) — reconciler-FAIL fix round with a
brief-sanctioned rehearsal EXECUTION on the read-only twin:** when the r3
reconciler upheld the twin's own FAIL (false "verified to engage cleanly"
claim + claimed-but-absent fixtures) and the brief says "Codex should run it
if its sandbox allows": (1) compose the execution grant as ONE named
exception inside the never-execute paragraph, with pre-run gates ALL
discharged by READING first — scratch-divertedness (every write root under
the script's own mktemp scratch), shim-pattern↔dispatcher-argv correspondence
(the shim's fall-through arm is `exec real-uv "$@"`, so a pattern MISS on a
GPU-phase argv launches a REAL workload — a mismatch is both a Critical
fresh-defect finding AND a do-not-run verdict), tool availability — plus the
never-fabricate STATIC fallback and an `EXECUTED rc=<n>` | `STATIC (env
unavailable)` arm recorded IN the closure-ledger line; flag at return time
that the dispatch write-mode decides which arm executes. (2) Key the closure
contract on the RECONCILER's refutation anchors (its direct-read line numbers
at the round-parent SHA resolve via `git show <parent>:<file>`), and hold the
NEW marker to the round's own self-declared law ("no verification claim
without an executed command + rc") — a fresh unbounded gloss is the same FAIL
class. (3) Composer verifies fixture EXISTENCE-in-diff (defs + call-site
wiring greps) and states existence SETTLED, discrimination YOURS. (4) A
marker prose-accuracy defect spotted at compose time (here: smoke-arch v2
claimed "posted this round" while its ts predates the r3 commit) gets a
chronology-facts note routed through the same law — CONCERNS at most unless
code-grounded. (5) An addressed-row-vs-execution ORDERING claim ("address
rows posted only AFTER the executed evidence existed") is the implementer's
claim — hand it to the twin under the law, don't resolve it.


**Revision-round sibling (#2546 r2, 2026-08-24):** with the external-denial
BLOCKER still open, the round-2 marker showed all-rc=0 VM smokes but the
heavily-reworked staging file (+194/−33 of discovery/dedup/draw logic) had
STRUCTURAL-ONLY evidence (AST + `--help`) — and r1's smoke had proven the
upstream staging stages runnable up to the 403, so a re-run was not fully
fenced. Compose a both-routes Step 0.6 note (uncertified-changed-logic
`smoke-run-missing` route, naming the specific logic, vs disclosed
revision-round CONCERNS route with the blocker fencing the full pipeline) —
never silence, never a pre-decision. Same round: a `(b)` rebuttal citing an
out-of-worktree reference (`common.py:325` in a parent lineage with no such
file here) gets a compose-time probe — state "citation does not resolve in
this worktree" + the nearest in-worktree corroboration (a figure label with
the same value) + the round's own constant site, and hand ACCEPTED/REJECTED
routes keyed on the plan's internal consistency, never resolve it yourself.

**Round-3 sibling (#2546 r3, 2026-08-25):** two deltas on the same lineage.
(1) A rebuttal citation that failed to resolve at round N can RESOLVE at
round N+1 once the implementer sharpens the path — r2's "the #1336 lineage
(common.py:325)" had no in-worktree referent, but r3's full
`src/explore_persona_space/experiments/issue_1336/common.py:325 @ ba8359381c`
resolved at the pinned SHA AND byte-identically in the worktree HEAD
(`DELTA_ELICIT_BAND = 0.02`). Re-probe every round; never carry a prior
round's "citation does not resolve" verdict forward, and when it resolves,
hand the twin BOTH the resolved constant and the plan's own contrary
literals (grep the plan for 0.021/0.0207 anchors) with routes, not a ruling.
(2) On a FAIL+FAIL fix round where the prior Codex verdict's findings were
ALL forwarded as persisted concerns, key the closure contract on the r2
verdict's own `CONCERN::` row texts (sharper than the ledger `raised`
summaries for re-raised ids — e.g. the trajectory row carried the
fixture-residual nuance only in the verdict row), quoted WITHOUT the machine
token; the r2 exit-1 staging critique then converts into a compose-time fact
that the r3 REAL staging slice ran to the credential wall (rc=1 at the gated
load = precisely what the r2 Critical demanded) — both-routes note again,
never silence. Also: the disclosed rc=1 real-slice run coexisting with six
rc=0 selftests is the healthy shape, not a contradiction.

**Round-4 sibling (#2546 r4, 2026-08-25) — reconciler-FAIL fix round with a
brief-sanctioned rehearsal EXECUTION on the read-only twin:** when the r3
reconciler upheld the twin's own FAIL (false "verified to engage cleanly"
claim + claimed-but-absent fixtures) and the brief says "Codex should run it
if its sandbox allows": (1) compose the execution grant as ONE named
exception inside the never-execute paragraph, with pre-run gates ALL
discharged by READING first — scratch-divertedness (every write root under
the script's own mktemp scratch), shim-pattern↔dispatcher-argv correspondence
(the shim's fall-through arm is `exec real-uv "$@"`, so a pattern MISS on a
GPU-phase argv launches a REAL workload — a mismatch is both a Critical
fresh-defect finding AND a do-not-run verdict), tool availability — plus the
never-fabricate STATIC fallback and an `EXECUTED rc=<n>` | `STATIC (env
unavailable)` arm recorded IN the closure-ledger line; flag at return time
that the dispatch write-mode decides which arm executes. (2) Key the closure
contract on the RECONCILER's refutation anchors (its direct-read line numbers
at the round-parent SHA resolve via `git show <parent>:<file>`), and hold the
NEW marker to the round's own self-declared law ("no verification claim
without an executed command + rc") — a fresh unbounded gloss is the same FAIL
class. (3) Composer verifies fixture EXISTENCE-in-diff (defs + call-site
wiring greps) and states existence SETTLED, discrimination YOURS. (4) A
marker prose-accuracy defect spotted at compose time (here: smoke-arch v2
claimed "posted this round" while its ts predates the r3 commit) gets a
chronology-facts note routed through the same law — CONCERNS at most unless
code-grounded. (5) An addressed-row-vs-execution ORDERING claim ("address
rows posted only AFTER the executed evidence existed") is the implementer's
claim — hand it to the twin under the law, don't resolve it.

**Round-5 sibling (#2546 r5, 2026-08-25) — concern-discharge round keyed on
the twin's OWN Major specs after a reconciler PASS:** when the prior round
ended Claude-PASS / Codex-FAIL / reconciler BINDING PASS with the twin's
Majors persisted at CONCERN and the new surgical round closes them: (1) the
acceptance anchors are the twin's own r(n-1) Major texts quoted VERBATIM
(Evidence/Fix/Mechanizable — it authored them; sharper than ledger
summaries), plus the reconciler's Standing-recommendations items as the
dispatch contract, with the author-neutrality line and a
round-PASS-is-SETTLED no-relitigate block; the default CONCERN fence applies
(brief had no NOT-ADDRESSED=FAIL clause — status-line-only re-raise at each
row's own severity; FALSE closure claims + new defects at the ordinary bar).
(2) A fails-pre-fix demonstration living in an UNCOMMITTED /tmp probe script
gets explicit steering: hold its CLAIMS to the executed-evidence law, READ it
if reachable but never run it and never mark BLOCKED for it — the durable
static equivalent is reading the round-parent blobs via `git show
<parent>:<path>` and confirming the committed fixtures would fail there.
(3) Reconciler-recommendation items the brief deliberately EXCLUDED (here
rec 4's Claude Standing-only minors, never persisted as rows) get a
`DELIBERATELY-NOT-TOUCHED (Standing-only, out of dispatch scope)` ledger
entry and an explicit never-a-finding / never-a-row instruction. (4) The r4
template reused cleanly via ~28 count-asserted splices (identical start/end
anchor ⇒ use a plain rep1, not splice); a delta-scoped smoke-arch marker
posted this round RESOLVES the prior staleness note — attest the full
commit→smoke-arch→ledger-rows→marker chronology and state that delta-scoping
with carry-forward-by-reference is a VALID shape.

**Round-4 sibling (#2546 r4, 2026-08-25) — reconciler-FAIL fix round with a
brief-sanctioned rehearsal EXECUTION on the read-only twin:** when the r3
reconciler upheld the twin's own FAIL (false "verified to engage cleanly"
claim + claimed-but-absent fixtures) and the brief says "Codex should run it
if its sandbox allows": (1) compose the execution grant as ONE named
exception inside the never-execute paragraph, with pre-run gates ALL
discharged by READING first — scratch-divertedness (every write root under
the script's own mktemp scratch), shim-pattern↔dispatcher-argv correspondence
(the shim's fall-through arm is `exec real-uv "$@"`, so a pattern MISS on a
GPU-phase argv launches a REAL workload — a mismatch is both a Critical
fresh-defect finding AND a do-not-run verdict), tool availability — plus the
never-fabricate STATIC fallback and an `EXECUTED rc=<n>` | `STATIC (env
unavailable)` arm recorded IN the closure-ledger line; flag at return time
that the dispatch write-mode decides which arm executes. (2) Key the closure
contract on the RECONCILER's refutation anchors (its direct-read line numbers
at the round-parent SHA resolve via `git show <parent>:<file>`), and hold the
NEW marker to the round's own self-declared law ("no verification claim
without an executed command + rc") — a fresh unbounded gloss is the same FAIL
class. (3) Composer verifies fixture EXISTENCE-in-diff (defs + call-site
wiring greps) and states existence SETTLED, discrimination YOURS. (4) A
marker prose-accuracy defect spotted at compose time (here: smoke-arch v2
claimed "posted this round" while its ts predates the r3 commit) gets a
chronology-facts note routed through the same law — CONCERNS at most unless
code-grounded. (5) An addressed-row-vs-execution ORDERING claim ("address
rows posted only AFTER the executed evidence existed") is the implementer's
claim — hand it to the twin under the law, don't resolve it.

Related: [[whole-round-unsplit-compose]],
[[worktree task-folder status can be stale in EITHER direction]],
[[brief-named concern adjudication]], [[revision-round compose recipe (round 2+)]].
[[brief-named concern adjudication]], [[revision-round compose recipe (round 2+)]],
[[concern-discharge round severity fence]].
