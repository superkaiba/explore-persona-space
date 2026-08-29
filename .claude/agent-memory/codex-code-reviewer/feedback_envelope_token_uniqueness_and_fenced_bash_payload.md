---
name: envelope-token-uniqueness-and-fenced-bash-payload
description: Never reproduce an envelope's literal opening token in the prompt's own prose (first-occurrence extraction lands on the prose); validate envelopes with grep -cxF not grep -cF; and how to compose for a skill-step diff whose payload is executable bash inside a markdown fence
metadata:
  type: feedback
---

Two lessons from composing #2385 r1 (Step 5a spec-sync stale-twin removal
arm — a bash arm inside `.claude/skills/issue/steps/09-step-5.md`).

## 1. Envelope tokens must appear EXACTLY ONCE — prose mentions break extraction

**Rule:** the blocked-read paragraph must NOT reproduce an envelope's
literal opening token. Write the mention as `` `BEGIN/END APPROVED PLAN
BODY` envelope``, never `` `---BEGIN APPROVED PLAN BODY---` envelope``.

**Why:** the spec's blocked-read rule tells Codex "the plan is inlined in
its `---BEGIN APPROVED PLAN BODY---` envelope", and the established
templates copy that sentence literally. That puts the token in the prompt
TWICE — once as prose ~20 lines above the real envelope, once as the
envelope. Any first-occurrence extraction (`content.index(tok)`, a naive
`sed`/`awk` range, or Codex itself scanning for the opener) then reads the
prose tail as the start of the plan body. Caught live in the #2385 r1
validation pass: the plan body extracted with first line
`'` envelope — a BLOCKED / FAIL on "plan'` instead of the plan's H1. The
same trap exists for the implementation-marker envelope if you ever
paraphrase it with the literal token.

**How to apply:** after substitution, assert `c.count(tok) == 1` for all
four envelope tokens, not just "present". If a count is 2, find the prose
mention and de-literalize it — do not reason that "the real one is on its
own line so it's fine". It costs one edit and removes the whole class.

**Validation form matters too:** `grep -cF -- "$tag"` counts SUBSTRING
occurrences and reported a false FAIL on a legitimately-unique envelope in
an earlier iteration. The authoritative check is `grep -cxF` (exact whole
line) for envelope tokens, plus the Python `count()` assert above for the
prose-duplication class. Use both — `-cxF` alone would have PASSED the
duplicated-prose case, since the prose mention is not a whole line.

## 2. Skill-step diffs whose payload is executable bash in a markdown fence

Distinct from the `.sh` wrapper case in
[[shell-wrapper-infra-compose]]: here the reviewed region is a fenced bash
block inside a `.md` skill step that a `/issue` session executes verbatim
as ONE Bash invocation, and the regression tests EXTRACT the block by
literal anchors and run it under real `bash` against a scratch git repo.

Composer duties that earned their place:

- **Say the payload is production shell, not prose.** Name the axes
  explicitly (quoting/word-splitting, unset/empty expansion, exit-code
  handling, `set -e`/`-u`/pipefail, subshell-vs-current-shell, what each
  command does when it FAILS). Without that sentence the twin reviews a
  markdown doc.
- **Step 3.8's "seam" is the extractor + fixture.** Hand the literal
  extractor anchors (`span.index("declare -A FAMILY_OF")` /
  `"# Sibling-issue file freshness (#1972)"`) with a composer-run
  occurrence COUNT for each, and set the duty as: confirm the new code
  falls inside the slice and neither anchor was disturbed.
- **For a first-ever DELETION arm, frame the fail-safe duty as "what does
  each probe do when it FAILS TO ANSWER"** — not "when it answers no".
  The productive question is whether empty output from an ERRORED
  `git log` / `git status` is distinguishable from empty output from a
  genuine non-match, because `if [ -n "$x" ]` routes both the same way.
  Ask it per probe and demand the direction (KEEP or DELETE) stated
  plainly; do not pre-adjudicate which way the code lands.
- **Ask whether the fire test can silently become a keep test.** A repro
  for a removal arm passes vacuously if any keep probe fires, or if the
  atomic checkout errors and the rc guard skips the arm. Enumerate the
  routes as a duty rather than trusting the one guard the test asserts.

## 3. Two facts that decide duties mechanically here

- `--name-status` all-`M` / zero-`A` ⇒ the #1805 round-new-script duty is
  N/A even when the marker's numstat looks like a big addition (see
  [[new-helpers-not-new-file-1805]]).
- A `kind: infra` skill-step round still triggers Step 3.75 (the diff
  changes echo strings and numeric literals other files pin) and Step 4.6's
  diff-consistency half. Run the composer's `grep -rlF` recount at the
  changed literals EXCLUDING `tasks/` (task state is historical plan/event
  text, never a pin) and hand the table as RE-DERIVE, per
  [[fixed-string-pin-sweep-recount]].
