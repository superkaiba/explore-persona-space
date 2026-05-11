---
name: promote-clean-result
description: >
  Workflow for moving an `status:awaiting-promotion` clean-result issue to
  `clean-results:useful` (or `:not-useful`). First scans the column for
  consolidation candidates (#237 pattern); if none, auto-restructures the
  body to v4 (per `clean-results/SPEC.md`), drafts a user-voice TL;DR,
  pushes to the issue, iterates with the user, then hands off the manual
  `gh_project.py promote` command. Use when the user says "promote #N",
  "clean up Awaiting promotion", or "help me refine the TL;DR for X".
user_invocable: true
---

# Promote a clean-result

Awaiting-promotion issues are reviewer-PASSed clean-results parked at `clean-results:draft` + `status:awaiting-promotion`. Promotion is **user-only** by design (CLAUDE.md gate 7) — `/issue` cannot move them. This skill gets one issue ready and exits to the user's terminal for the actual `gh_project.py promote` command.

**Apply first, iterate after.** Don't pre-propose every TL;DR / Summary draft in chat — auto-draft, run the quality gates, push to the issue, then iterate against the live body. The only pre-apply user gate is Step 0.5 (consolidation candidates), because merging is destructive.

**All body-shape rules live in `.claude/skills/clean-results/SPEC.md`.** This skill is workflow + apply mechanics only. When in doubt about what a section should look like, read SPEC.md.

---

## When to use

- User says "promote #N", "let's clean up Awaiting promotion", "write the TL;DR for X", or pastes a link to an `awaiting-promotion` issue.
- One issue per skill invocation — `/clear` between issues so each draft gets clean attention.

## When NOT to use

- For drafting Summary, Details, figures. That's the analyzer / interpretation-critic / reviewer loop's job.
- For posting net-new clean-result issues. That's `/issue`'s analyzer step.
- For changing the `useful` / `not-useful` decision after promotion. Re-run `gh_project.py promote` directly.

---

## Step 0 — Identify the target issue

If the user named a specific issue number, use it. Otherwise list the column:

```bash
uv run python scripts/gh_project.py list-by-status "Awaiting promotion"
```

Pick one with the user. Don't try to do all of them in one context.

## Step 0.5 — Scan for consolidation candidates

Before reading the target end-to-end, check whether the column contains other issues whose findings should fold into the target as additional `### Result N` sections (the #237 pattern). Promote-by-merge is preferable to promote-as-siblings when the issues share a single load-bearing claim — separate clean-results fragment the narrative.

### Gather candidates

```bash
uv run python scripts/gh_project.py list-by-status "Awaiting promotion"
# For each candidate (top ~10 nearest in title):
gh issue view <Ni> --json number,title,labels,body
```

### Similarity signals (any 2+ → propose merge)

- **Shared parent issue** — both Background-cite the same prior `#<N>`.
- **Shared experimental geometry** — same persona set, eval rig, matcher, model. Seed / sweep-point deltas are NOT independent claims.
- **Shared headline verb in titles** — "X does Y" + "X also does Z" + "X under W" = three sentences of one paragraph.
- **One refutes / strengthens / qualifies the other** — Result 2 = the qualifier, not a separate clean-result.
- **Same `epm:parent-issue:` marker** or `Source-issues:` prose line in Background.

### Anti-signals (KEEP separate)

- Different mechanisms / different load-bearing claims, even if the parent is shared.
- Different confidence tiers (HIGH + LOW) where merging would force a single tier on the title.
- One is `useful`-bound, the other `not-useful`-bound — verdicts are per-issue.
- Bodies have already diverged structurally (different figure sets, different Methodology). Merging would be a rewrite.

### Propose to the user (don't act yet)

```
### Consolidation candidate detected

Target: #N

Likely fold-ins:
- #M1 — <title gist> — shared signal: <which similarity rule>
- #M2 — <title gist> — shared signal: <which similarity rule>

Proposed shape: consolidated body in #N with:
- ### Result 1 (current): <claim from #N>
- ### Result 2 (from #M1): <claim from #M1>
- ### Result 3 (from #M2): <claim from #M2>
- ## Source issues H2 added
- #M1, #M2 closed as superseded after merge

(a) merge as proposed, (b) merge subset, (c) keep separate, (d) re-pick target?
```

Wait for explicit sign-off — merging closes sibling issues, irreversible without manual undo.

### If user picks merge (a or b)

1. Re-read each contributing issue's body in full — need figure paths, headline numbers, Methodology deltas.
2. Construct consolidated body in a scratch buffer (same auto-restructure pipeline; merge just unions per-Result content first).
3. Show user the full draft (not a diff — too noisy across multi-section merge).
4. When approved, apply via `gh_graphql` MCP `update_issue_body` (fall back to `gh issue edit <N> --body-file <path>`, then to REST PATCH if rate-limited: `gh api -X PATCH /repos/<owner>/<repo>/issues/<N> -F body=@<path>`).
5. `uv run python scripts/verify_clean_result.py <N>` — fix any FAIL.
6. Close each fold-in:
   ```bash
   gh issue comment <Mi> --body "Consolidated into #N — see ### Result <K>. Closing as superseded."
   gh issue close <Mi> --reason "not planned"
   ```
   Confirm with the user before flipping each fold-in's `clean-results:draft` → `clean-results:not-useful` (or removing the label).
7. Continue from Step 1 with the consolidated body as the new target.

### If user picks (c) keep separate or (d) re-pick

Continue from Step 1 with the original (or newly-chosen) target.

---

## Step 1 — Read the body (silent diagnostic, no chat output)

```bash
gh issue view <N> --json title,body
```

Read the whole body. Diagnostic checklist (track internally, address in Step 2):

1. **Is the body actually a clean-result?** Must have `## Summary` + `## Details` (with `### Background` / `### Methodology` / `### Result N`) + confidence-suffixed title. If it still looks like the original plan (Goal / Hypothesis / Design, no Result section, no confidence in title) — **STOP and tell the user.** This is the only Step 1 condition that aborts the skill. Common cause: reviewer FAIL'd, status drifted manually, or the experiment was abandoned.
2. **Triad shape.** Detect which TL;DR/Summary/Details shape the body uses — drives Step 2's rename pipeline:
   - **v4 (current):** already has `## TL;DR` / `## Summary` / `## Details`. Refine existing TL;DR bullets.
   - **Legacy placeholder:** body has `## TL;DR` with the literal placeholder line. Draft fresh per SPEC.md §4.
   - **Legacy v1 triad:** `## Human TL;DR` + `## AI TL;DR (human reviewed)` + `## AI Summary`. Rename to v4 triad, strip the headline paragraph + "In detail:" prose, restructure Summary to 6-bullet shape.
3. **Title — declarative-opener rule** (SPEC.md §2). Titles starting with `If you...`, `When you...`, `Suppose...` read as tutorial register. Queue a rewrite for Step 2.
4. **Heading-as-toggle convention** (SPEC.md §1). Each `## H2` and `### H3` (except `## Details` container) should sit inside `<details open><summary>`. Run `verify_clean_result.py` and note un-wrapped sections.
5. **Project-internal jargon.** Run `audit_clean_results_body_discipline.py` and note flagged patterns.

---

## Step 2 — Auto-draft + auto-restructure + auto-apply

End-to-end without chat output between sub-steps. Single end-of-step status line after the body is live.

### 2a. Auto-draft the user-voice TL;DR (SPEC.md §4 rules)

Draft a 30–90-word, 3-4-bullet user-voice TL;DR matching the closest SPEC.md §4 exemplar (#276 single-claim, #295 hypothesis-falsified-plus-wrinkle, #281 mini-protocol with placeholder labels). Use prior conversation as input — when the user has been discussing the experiment, prefer their framing.

### 2b. Auto-rewrite the title if needed (SPEC.md §2)

If Step 1 flagged a conditional opener: convert *"If you VERB X, Y"* → *"VERB-ing X DOES Y"* (gerund) or *"X DOES Y under VERB"* (noun phrase). Compress 3+ stacked claims to 1-2. Flip negation to affirmative. Preserve the `(HIGH | MODERATE | LOW confidence)` suffix.

### 2c. Auto-restructure to v4 shape (SPEC.md §1, §4, §5)

- **v4 already:** replace TL;DR bullets with Step 2a draft. Leave Summary/Details untouched unless Summary needs the 6-bullet restructure.
- **Legacy placeholder:** replace placeholder line with Step 2a draft.
- **Legacy v1 triad:**
  1. Rename `## Human TL;DR` → `## TL;DR`; insert Step 2a bullets.
  2. Rename `## AI TL;DR (human reviewed)` → `## Summary`. **Strip the v1 headline paragraph and the "In detail:" prose paragraph.** The renamed `## Summary` MUST open directly with `- **Motivation:**`.
  3. Rename `## AI Summary` → `## Details`. The collapsed `<details>` Setup block and the H3 subsections carry over unchanged.

Then reshape Summary to the 6-bullet structure per SPEC.md §5: Motivation / Experiment / Results / Takeaways / Next steps / Confidence.

### 2d. Wrap H2/H3 sections in heading-as-toggle blocks (SPEC.md §1)

Required wrapping: `## TL;DR`, `## Summary`, `## Source issues`, every `### Background` / `### Methodology` / `### Result N` / `### Next steps` H3. `## Details` (container H2) is exempt.

### 2e. Fix bare `#N` references

Convert each bare `#N` (including `` `#N` `` inside backticks) to `[#N](https://github.com/<owner>/<repo>/issues/N)`.

### 2f. Run quality gates

1. **`verify_clean_result.py`** — must PASS (WARNs ok). FAIL means a structural error; fix and re-run. For pre-cutoff bodies, pass `--skip-checks check_results_block,check_human_summary,check_sample_outputs,check_reproducibility`.
2. **`audit_clean_results_body_discipline.py`** — apply the cheapest plain-English replacement for each flagged hit, re-run.
3. **`clean-result-critic` agent** (optional, spawn only if Step 1 had diagnostic flags). PASS → continue. REVISE → apply the agent's "Specific revision requests" verbatim, then continue. The critic is a quality gate, not a user-facing iteration loop — apply autonomously.

### 2g. Apply title + body

1. **Title change first** if Step 2b rewrote it: `gh issue edit <N> --title "<new title>"`. Title before body — a partial body failure can't leave a body referencing a stale title.
2. **Body apply** via `gh_graphql` MCP `update_issue_body`. Fall back to `gh issue edit <N> --body-file <path>`, then REST PATCH (see Step 0.5).
3. Re-run `verify_clean_result.py <N>` against the LIVE issue.

### 2h. Single-line status to chat

```
Body live on #<N>: <URL>
What I did: <one-line, e.g. "renamed v1 triad → v4, drafted TL;DR, restructured Summary to 6-bullet, wrapped 9 sections in <details>, fixed 4 bare #N refs">
Verifier: PASS (WARNs: <count>)
Want any tweaks before promoting?
```

Then move to Step 3.

---

## Step 3 — Iterate against the live body

The user reads the posted body on GitHub and asks for tweaks; you apply them in place. Common shapes:

- **"Bullet 2 of the TL;DR should say X, not Y."** Apply, re-run verifier, echo 1-line confirmation.
- **"Add a Result section for the new analysis I just ran."** Read the analysis JSON, generate the figure via `paper-plots`, commit, update body. Loop on framing if asked; otherwise apply autonomously.
- **"Re-run figure 2 without the police_officer outlier."** `paper-plots`, commit, point body at new commit SHA, re-verify.
- **"Promote as not-useful instead."** Skip to Step 4 with the not-useful command.
- **"Actually this should be merged with #M."** Back to Step 0.5 with the new merge target.

Mechanics per iteration:

1. Re-fetch latest body (`gh issue view <N> --json body`) — the user may have edited in-browser.
2. Make the change. Echo back ONLY the diff in chat (title + the changed paragraph / bullet / section), never the whole body.
3. Re-run `verify_clean_result.py <N>` if the change touched structure (TL;DR bullets, Summary bullets, Result sections, headings). Skip for one-word fixes inside narrative prose.
4. Wait for next user instruction or "ready to promote."

**Capture iteration patterns.** If the user's correction generalizes — would catch the same class of issue in the next clean-result — propose:
- (a) An append to `.claude/skills/clean-results/iterations.md` (one H3 under the appropriate `## YYYY-MM-DD — issue #N (topic)` H2 with `**Before / After / Rule / Folded into**`).
- (b) Surgical edits to the relevant canonical file (typically `clean-results/SPEC.md` or `scripts/verify_clean_result.py`).

The user approves each before you write. **Always log; sometimes generalize.**

---

## Step 4 — Hand off

When the user says "ready to promote" / "ship it" / "looks good, promote useful":

```
Ready to promote. Run one of:

  uv run python scripts/gh_project.py promote <N> useful
  uv run python scripts/gh_project.py promote <N> not-useful

This flips `clean-results:draft` → `clean-results:useful` (or `:not-useful`)
and routes the project board column. Awaiting-promotion is a user-only gate
(CLAUDE.md, gate 7) — I will not run it.
```

Then EXIT. Don't ask "want me to do the next one?" — the user should `/clear` and re-invoke the skill on the next issue.

---

## Decision tree: useful vs not-useful

| Signal | Suggests `useful` | Suggests `not-useful` |
|---|---|---|
| Confidence | HIGH or MODERATE | LOW |
| Survives in the paper / a mentor talk | yes | no |
| Findings update someone's beliefs | yes | "we couldn't tell" |
| Replicates / strengthens a parent issue | yes | yes (still useful as evidence) |
| Negative result with a sharp mechanism | useful | n/a |
| Negative result + binding-constraint excuse | n/a | not-useful |

`not-useful` is not "the experiment failed" — it's "the result isn't load-bearing for the project's narrative." Failed experiments with sharp mechanisms are usually `useful`. Successful experiments that turn out to duplicate a prior finding without adding evidence are usually `not-useful`.

---

## Reference

All body-shape rules, exemplars, anti-patterns, verifier expectations, and rationale live in:

- **`.claude/skills/clean-results/SPEC.md`** — single source of truth for clean-result body shape (v4).
- **`.claude/skills/clean-results/iterations.md`** — append-only log of past corrections. Grep when checking whether a phrasing has been litigated before.
- **`.claude/skills/clean-results/lw-post-examples/`** — verbatim external exemplars.

This skill stays narrow on workflow: identify target → consolidation scan → diagnostic → auto-restructure + apply → iterate → hand off.
