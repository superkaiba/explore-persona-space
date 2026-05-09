---
name: promote-clean-result
description: >
  Iterative, conversational workflow for promoting an `status:awaiting-promotion`
  clean-result issue to `clean-results:useful` (or `:not-useful`). First
  scans the Awaiting-promotion column for similar issues that should be
  consolidated into a single multi-claim body before promoting (the #237
  pattern); if none, walks one issue end-to-end: read the AI-drafted body,
  propose a `## Human TL;DR` in the user's voice, iterate with the user
  on the *interpretation / takeaways* (not prose), then auto-critique the
  locked draft against the LessWrong style + verbatim exemplars and
  produce a revised version, apply the final body, then have the user run
  `python scripts/gh_project.py promote <N> useful|not-useful`. Use when the
  user says "promote #N", "let's clean up the awaiting-promotion column",
  "help me write the Human TL;DR for X", or asks to move issues out of
  Awaiting promotion.
user_invocable: true
---

# Promote a clean-result

Awaiting-promotion issues are reviewer-PASSed clean-results parked at
`clean-results:draft` + `status:awaiting-promotion`. Promotion is
**user-only** by design (CLAUDE.md gate 7) — `/issue` cannot move them.
This skill is the conversation that gets one issue ready and then exits
to the user's terminal for the actual `gh_project.py promote` command.

The workflow is **iterative and human-in-the-loop.** Don't auto-promote.
Don't batch. Don't commit edits without user sign-off. Propose, listen,
adjust, re-propose, then apply.

---

## When to use

- User says "promote #N", "let's clean up Awaiting promotion", "write the
  Human TL;DR for X", or pastes a link to an `awaiting-promotion` issue.
- User wants to move multiple issues from Awaiting promotion through the
  Useful / Not useful columns. Run this skill **once per issue** — start a
  fresh context (`/clear`) between issues so each draft gets clean attention.
- The skill always runs Step 0.5 first: it scans the rest of the
  Awaiting-promotion column for issues whose findings should fold into
  the target as additional `### Result N` sections (the #237
  consolidation pattern). If there's a candidate group, the user is
  asked whether to merge before any Human TL;DR work begins. This
  catches the case where two reviewer-PASSed clean-results turn out to
  be one finding split across two issues.

## When NOT to use

- For drafting the AI TL;DR, AI Summary, figures, or any other clean-result
  body section. Those are the analyzer / interpretation-critic / reviewer
  loop's responsibility, not this skill.
- For posting net-new clean-result issues. That's `/issue`'s analyzer step.
- For changing the issue's `clean-results:useful` / `:not-useful` decision
  after promotion. The user re-runs `gh_project.py promote` directly.

---

## Step 0 — Identify the target issue

If the user named a specific issue number, use it. Otherwise list the
column:

```bash
uv run python scripts/gh_project.py list-by-status "Awaiting promotion"
```

Pick one with the user. Don't try to do all of them in one context.

## Step 0.5 — Scan Awaiting promotion for consolidation candidates

Before reading the target issue end-to-end, check whether the column
contains other issues whose findings should fold into the target's body
as additional `### Result N` sections, rather than living as separate
clean-results. The pattern is `clean-results/template.md` § "Source
issues" / the #237 pattern: one consolidated body that distils evidence
from `#N1, #N2, #N3`, with the contributing issues closed as superseded.

Promote-by-merge is preferable to promote-as-siblings when the issues
share a single load-bearing claim or refute / strengthen a single parent
claim — separate clean-results would otherwise fragment the narrative
and force a future reader to thread three labels (`useful` x3) for one
finding.

### Gather candidates

List the full column and pull each issue's title + AI TL;DR + Source
issues / Background `#N` refs:

```bash
uv run python scripts/gh_project.py list-by-status "Awaiting promotion"
# For each candidate (keep the list small — top ~10 nearest in title):
gh issue view <Ni> --json number,title,labels,body
```

### Similarity signals (any 2+ → propose merge)

- **Shared parent issue.** Two awaiting-promotion issues both Background-cite
  the same prior `#<N>` (the parent experiment / claim they're follow-ups of).
- **Shared experimental geometry.** Same persona set, same eval rig, same
  matcher / metric, same model. Cosmetic deltas (seed, hyper-param sweep
  point) are NOT independent claims.
- **Shared headline verb in titles.** "X coupling does Y" + "X coupling
  also does Z" + "X coupling under W" — three sentences of the same
  paragraph, not three papers.
- **One refutes / strengthens / qualifies the other.** "Effect E exists"
  + "Effect E vanishes under control C" → consolidated body where
  Result 2 = the qualifier, not a separate clean-result.
- **Same `epm:parent-issue:` marker** or same `Source-issues:` prose
  line in Background.

### Anti-signals (KEEP separate, do NOT merge)

- Different mechanisms / different load-bearing claims, even if the
  parent is shared. ("Marker fires on persona A" + "Coupling persists
  through SFT" = two findings, two issues.)
- Different confidence tiers where merging would force a single tier on
  the consolidated title. (HIGH + LOW → keep separate; the LOW finding
  belongs as its own draft or as a caveat-not-headline.)
- One is `useful`-bound and the other is `not-useful`-bound. The
  promotion verdict is per-issue; merging forces a single verdict.
- Issues whose bodies have already diverged structurally (different
  figure sets, different `### Methodology`). Merging would be a rewrite,
  not a fold-in.

### Propose to the user (don't act yet)

If you find a candidate group, present it like this in chat:

```
### Consolidation candidate detected

Target: #N (the issue the user opened this skill on)

Likely fold-ins:
- #M1 — <one-line title gist> — shared signal: <which similarity rule fired>
- #M2 — <one-line title gist> — shared signal: <which similarity rule fired>

Proposed shape: consolidated body in #N with:
- ### Result 1 (current): <claim from #N>
- ### Result 2 (from #M1): <claim from #M1>
- ### Result 3 (from #M2): <claim from #M2>
- ## Source issues H2 added (lists #N, #M1, #M2)
- #M1, #M2 closed as superseded after merge

Want me to: (a) merge as proposed, (b) merge a subset, (c) keep all
three separate and continue with #N alone, or (d) re-pick the target?
```

Wait for the user's choice. **Do not edit any body without explicit
sign-off.**

### If the user picks merge (a) or subset (b)

1. Re-read each contributing issue's body in full (`gh issue view <Mi>
   --json body`). You need the figure paths, headline numbers, and
   `### Methodology` deltas to fold in correctly.
2. Construct the consolidated body in a scratch buffer:
   - Append each fold-in's `### Result N` block to the target's `## AI
     Summary`, renumbering Result sections in narrative order (not
     issue-number order — the order a reader should traverse them).
   - Add `## Source issues` H2 after `## AI Summary` per
     `clean-results/template.md` § "Source issues" (CONDITIONAL).
   - Add `Source-issues: #N, #M1, #M2` and (if applicable)
     `Supersedes: <none>` lines at the very top of `### Background`.
   - Update the `## AI TL;DR` paragraph to carry the merged headline
     (one paragraph, still 30-200 words, still 3-5 sentences). Each
     fold-in's load-bearing claim should appear as one sentence; if
     that pushes the paragraph past 5 sentences, compress —
     consolidation should sharpen the lede, not bloat it.
   - Update the title to the merged headline + `(HIGH | MODERATE | LOW
     confidence)` reflecting the lowest tier across the merged claims
     (be conservative).
   - Headline numbers table: union the rows; keep the column schema.
   - Artifacts section: union the artifact links.
3. Show the user the proposed consolidated body (full draft, not a
   diff — diffs are too noisy across a multi-section merge). Iterate
   if they want different ordering / different headline framing.
4. When approved, apply via `gh_graphql` MCP `update_issue_body` (or
   fall back to `gh issue edit <N> --body-file -`).
5. Run `uv run python scripts/verify_clean_result.py <N>` — if FAIL,
   fix the structural violation before continuing. Common fails after
   merge: missing `## Source issues` H2, mismatched confidence between
   title and AI TL;DR `**Confidence:**` line, AI TL;DR paragraph >200
   words.
6. Close each fold-in issue with a comment pointing at the
   consolidated body:
   ```bash
   gh issue comment <Mi> --body "Consolidated into #N — see ### Result <K> for this issue's contribution. Closing as superseded."
   gh issue close <Mi> --reason "not planned"
   ```
   Note: closing an `awaiting-promotion` issue would normally trigger
   `project-archive-on-close.yml` to move it to Archived, but the
   sticky-label rules in CLAUDE.md keep `clean-results:*`-labeled
   issues in their column. After close, manually flip each fold-in's
   `clean-results:draft` → `clean-results:not-useful` (or remove it
   entirely and rely on the closure + comment) so the column doesn't
   show stale drafts. Confirm the desired label move with the user
   before applying.
7. Continue from Step 1 with the consolidated body as the new target.

### If the user picks (c) keep separate or (d) re-pick

Continue from Step 1 with the original (or newly-chosen) target. Make
no edits.

## Step 1 — Read the issue body end-to-end

```bash
gh issue view <N> --json title,body
```

Read the **whole** body, not just the AI TL;DR. You're checking:

1. **Is the body actually a clean-result?** It must have `## AI TL;DR
   (human reviewed)` + `## AI Summary` (with `### Background` /
   `### Methodology` / `### Result N` subsections) + a confidence-suffixed
   title. If the body still looks like the original plan (Goal / Hypothesis
   / Design — no Result section, no confidence in title), the issue is
   mis-labeled. STOP and tell the user — promotion would propagate a
   half-finished issue into the Useful/Not useful columns. Common cause:
   reviewer FAIL'd, status label drifted manually, or the experiment was
   abandoned mid-stream.
2. **Does the body have an existing `## Human TL;DR` placeholder OR a
   user-drafted one?** A drafted Human TL;DR (often labeled `## Human TL;R`
   / `## Human Summary` / similar typos) is the user's voice — preserve
   it and propose only minor cleanup. The verbatim placeholder line
   `_(Human TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_`
   means it's untouched and you should propose from scratch.
3. **Are there obvious overclaims, missing caveats, or mismatched figures?**
   Flag them before drafting the Human TL;DR — sometimes the right move is
   "let's regenerate Figure 2 first" rather than "let me write a TL;DR."
4. **Does the title pass the declarative-opener rule?** Promoted Useful
   titles all lead with a noun phrase (e.g. *"A pretraining-data-poisoned
   Qwen3-4B backdoor only fires..."*) or a gerund (*"Stretching turn
   count..."*, *"Fine-tuning one persona..."*, *"Training a `[ZLT]`
   marker..."*). Titles that open with `If you...`, `When you...`,
   `Suppose...`, or any other conditional/hypothetical clause read as
   tutorial register, not finding register, and almost always invite
   multi-claim em-dash chaining (issue #239 is the canonical bad
   example). If the title opens with a conditional, flag it as a
   heads-up — you'll auto-rewrite it in Step 3.5. See
   `lw-register-cheatsheet.md` § "Title rules" for the conversion recipe
   and verbatim Useful-column exemplars.
5. **Does the body use the heading-as-toggle convention?** Each `## H2`
   and `### H3` body section (except `## AI Summary`, the container) should
   sit inside a `<details open><summary>` block whose `<summary>` carries
   the markdown heading itself, so the heading is the click target on
   GitHub and the section is collapsible. Pattern (note the blank lines
   that re-enable markdown parsing inside the HTML block):

   ```markdown
   <details open>
   <summary>

   ## Human TL;DR

   </summary>

   content...

   </details>
   ```

   Run `uv run python scripts/verify_clean_result.py <N>` and look for the
   `Collapsible sections` check. If it's WARN ("N section(s) not wrapped"),
   flag it as a heads-up and offer to wrap them as part of the promotion
   pass. Pre-2026-05-09 drafts are grandfathered (verifier WARNs, doesn't
   FAIL), but new drafts and any draft you're touching for promotion
   should adopt the convention. See `clean-results/template.md` § "Body
   shape" for the canonical pattern + which headings are exempt.

## Step 2 — Propose a Human TL;DR (substance draft)

Write 2-5 plain-English bullets in the user's voice. This first proposal
is for **substance** — interpretation and takeaways. Don't over-polish;
the auto-critique pass in Step 3.5 will handle prose mechanics. Read the
verbatim exemplars in `human-tldr-examples.md` (this skill's directory)
BEFORE drafting so the shape is roughly right out of the gate. The shape:

- **Bullet 1** opens with the question / what we tested / what we wanted
  to see — never the result.
- **Bullet 2** is the headline finding, plainly stated, often negative
  ("It did not", "actually flipped", "no effect").
- **Bullet 3+** is optional: a surprise, a side-finding, an anomaly worth
  flagging. Often ends "worth investigating further" / "probably due to X".
- First-person, present tense, casual punctuation. No `r =`, no `p =`, no
  `(MODERATE confidence)`. Bold-emphasis allowed for the load-bearing word.
- ~30-90 words total. Shorter is better.
- The Human TL;DR is the user's editorial layer; it's OK to subtly disagree
  with the AI TL;DR's framing — the AI TL;DR carries the precise paragraph
  lede, the Human TL;DR carries the colloquial "shoulder-tap to a peer"
  framing.

Present the draft in the chat. **Do not edit the issue body yet.** Show:

1. The proposed `## Human TL;DR` block, fenced as markdown.
2. 1-3 framing knobs you had to make a call on — each as a quick yes/no
   question the user can answer in one breath. Examples:
   - "Lead with the metric name or with the question?"
   - "Do you want the surprise bullet, or just the headline?"
   - "Flag the police_officer outlier here, or save it for the body?"
3. If you spotted body issues in Step 1, list them under a separate
   `### Heads-up` block. Don't bury them.

## Step 3 — Iterate on **substance** with the user

This step is for locking down **what the main interpretation and
takeaways should be** — *not* for prose polish. Save the prose pass for
Step 3.5 (auto-critique), which runs without the user in the loop.

Common shapes the iteration takes:

- **Reframe / different emphasis.** "The headline should be the
  cosine-vs-JS dissociation, not the cross-section win." Re-draft,
  re-show, re-ask.
- **Add / drop a bullet.** "Cut the surprise bullet, the body has
  enough caveats already" or "actually flag the police_officer outlier
  here." Apply and re-show.
- **"Run a fresh analysis / make a new plot first."** Happens when a
  load-bearing claim is no longer trusted, or a figure is ambiguous.
  Treat as legitimate work: spawn a sub-agent or do it inline (whichever
  is cheaper), update the body if needed (with the `paper-plots` skill
  for any new figure), commit the new artifact, then loop back to Step 2
  with the corrected understanding.
- **"This shouldn't be a clean result at all."** Move to Step 5 (Not
  useful path) directly.

What the user IS deciding here:
- Which finding is the headline.
- Whether to include the surprise / side-finding bullet.
- Whether the bullets reflect the actual experimental geometry.
- Whether the framing matches their lineage memory of the issue.

What the user is NOT deciding here (defer to Step 3.5):
- Word choice / casual-vs-formal punctuation.
- Whether bullets are too long.
- Whether numbers leaked in from the AI TL;DR.
- Whether the opening verb is right ("Tested" vs "Checked" vs "Wanted to see").

Each substance iteration: re-show the full proposed block (don't make
the user mentally diff). State explicitly which knobs you changed.

When the user says "yep, that's the right interpretation" / "ship it" /
"good, apply it" — substance is locked. Move to Step 3.5 immediately.
DO NOT apply yet.

## Step 3.5 — Auto-critique against LW style + examples (no user loop)

Once substance is locked, run the prose-polish pass yourself. This is a
mechanical pass against the rubric in `lw-register-cheatsheet.md` and
the verbatim shapes in `human-tldr-examples.md`. The user has already
signed off on what the bullets *say* — your job is to make sure they
*read* the way the exemplars read.

Read both reference files (or re-skim if already in context), then run
the draft through this checklist. For each FAIL, produce a fix; for
each PASS, leave the bullet alone.

**Title sub-pass first.** Before the Human TL;DR critique, run the
title through the declarative-opener rule from
`lw-register-cheatsheet.md` § "Title rules":

- Does it start with `If` / `When` / `Suppose` / `Imagine` / any
  conditional or hypothetical? → REWRITE to declarative. Use the
  conversion recipe: *"If you VERB X, Y"* → *"VERB-ing X DOES Y"*
  (gerund opener) or *"X DOES Y under VERB"* (noun-phrase opener).
- Does it stack 3+ claims joined by em-dash / semicolon? → compress to
  the load-bearing 1-2.
- Does it negate a prior claim instead of stating the affirmative
  finding? → flip to affirmative.
- Does it end with `(HIGH | MODERATE | LOW confidence)`? → required;
  preserve / re-add if the rewrite drops it.
- Is the load-bearing claim in the first ~80 characters? → reorder if
  not.

If the title needs a rewrite, propose the new title alongside the
Human TL;DR critique below. Apply via `gh issue edit <N> --title "..."`
(or the `gh_graphql` MCP `update_issue_title` tool, when available) in
Step 4.

**Human TL;DR checklist (work top to bottom; flag every issue, then revise):**

1. **Voice.** First-person ("we", "I"), present tense, casual
   punctuation (`--` for em-dash, `..`, lowercase). Not third-person
   passive ("It was tested..."), not analyst ("This experiment shows...").
2. **Verb of inquiry in bullet 1.** "Tested" / "Checked" / "Wanted to
   see" / "Evaluated" — not "We found" / "Result:" / "X does Y" (those
   start *with* the result, which is bullet 2's job).
3. **Headline = bullet 2.** One bullet, ≤25 words, plainly stated. Often
   negative ("It did not", "actually flipped"). Not buried in a
   sub-clause inside bullet 1.
4. **No AI-TL;DR contamination.**
   - Strip `r =`, `p =`, `Spearman`, `partial correlation`, `n =`,
     `vs <number>`, `Δ`, `±`, percentage-point comparisons. Numbers
     belong in the AI TL;DR.
   - Strip "(MODERATE confidence)" / "(LOW confidence)" suffix.
   - Strip per-condition compound nouns: "matched-scaffold leakage",
     "cosine-L20", "diff-of-diffs". Replace with the plain phrase.
5. **No restating the title.** If a bullet paraphrases the AI TL;DR's
   first sentence, scrap it. The Human TL;DR should add framing the
   AI TL;DR can't carry.
6. **Bullet length.** 1-2 sentences each, ~15-30 words. Split anything
   with 3+ commas or a semicolon. If bullet 1 is ≥40 words, it has
   absorbed methodology that should compress.
7. **Total length.** ~30-90 words across the whole block. If ≥100 words,
   compress; usually bullet 1 over-explains the setup.
8. **Bullets are not redundant.** If bullet 3 paraphrases bullet 2, drop
   bullet 3 OR replace it with a surprise / side-finding / forward-look.
9. **Concrete inline handholds preferred over category labels.** Match
   exemplar phrasing — "synonyms, other AI companies, similar sounding
   words" beats "various paraphrase types"; "persona 1 / persona 2"
   beats "the donor / recipient condition" if the body uses A / B / C
   labels the reader has to thread.
10. **Stylistic match to exemplars.** When in doubt, pattern-match
    against the closest of #276 / #295 / #281 in
    `human-tldr-examples.md`:
    - Single-claim, narrow-leakage finding → #276 shape (3 bullets,
      "Checked if X. It does Y but only Z. Also tried W but it doesn't").
    - Hypothesis falsified + unexpected positive wrinkle → #295 shape
      (4 bullets, "Evaluated X. We thought Y. It did not — instead Z.
      But W caused **more** A — worth investigating further").
    - Mini-protocol with placeholder labels → #281 shape (3 bullets,
      "Wanted to see: if X then Y. Result: not Y. Also a random Z did A
      — probably due to W").

**Produce a revised version.** Show the user:

```
### Title (current → proposed)

CURRENT:  <existing title>
PROPOSED: <rewritten title, or "no change — already declarative">
WHY:      <one line: which rule fired, or "PASS">

### Substance-locked Human TL;DR draft (Step 3 output)

<the draft the user signed off on>

### Style-critique findings

- title: <issue or PASS>
- bullet 1: <issue>
- bullet 2: <issue>
- ...
- (or: "no issues found, draft is exemplar-shaped")

### Auto-revised Human TL;DR draft (Step 3.5 output)

<the revised draft, ready to apply>
```

Keep the diff visible — the user should see exactly which words changed
and why. If the substance-locked draft is already exemplar-shaped, say so
explicitly ("no style issues") and pass it through unchanged rather than
inventing edits.

**Do not loop with the user on style.** If they push back on a specific
edit ("no, keep `Δ` in there, I want the number"), apply just that
override and move to Step 4. Don't re-run the full critique.

## Step 4 — Apply the final title + Human TL;DR to the issue

When the user signs off on Step 3.5:

1. **If the title was rewritten in Step 3.5**, apply the title change
   FIRST via `gh issue edit <N> --title "<new title>"` (or the
   `gh_graphql` MCP equivalent when it ships). The title and the body
   are independent edits — title first means a partial failure can't
   leave a body referencing a stale title.
2. Re-fetch the latest body (`gh issue view <N> --json body`) — the user
   may have edited it in the browser while you iterated.
3. Replace the placeholder line:
   ```
   _(Human TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_
   ```
   with the approved bullets. If the body has a typo'd `## Human TL;R` or
   similar pre-existing draft above the placeholder, REMOVE the typo'd
   block AND the placeholder, leaving exactly one `## Human TL;DR` H2
   followed by the approved bullets.
4. **If the title was rewritten**, also rewrite the AI TL;DR's first
   sentence (which is supposed to be the title verbatim minus the
   confidence suffix per `clean-results/template.md`). The verifier
   does not enforce this match, but the analyzer + reviewer convention
   does — keeping them in sync prevents a future reader from seeing two
   different paragraph-LEDE phrasings of the same finding.
5. Update via the `gh_graphql` MCP tool (`update_issue_body`) so the token
   never enters the agent context window. Fall back to
   `gh issue edit <N> --body-file -` only if MCP is unavailable.
6. Echo back the diff in chat (title change + Human TL;DR section + AI
   TL;DR sentence-1 change if applicable; never the whole 65k-byte body).
7. Run `uv run python scripts/verify_clean_result.py <N>` (or pass the
   raw body) to confirm v2 structure still passes. If FAIL, do not
   continue — surface the verifier output to the user.

## Step 5 — Hand off to the user for the actual promotion

The user runs the promote command themselves; this skill never runs it.
Print exactly:

```
Ready to promote. Run one of:

  uv run python scripts/gh_project.py promote <N> useful
  uv run python scripts/gh_project.py promote <N> not-useful

This flips `clean-results:draft` -> `clean-results:useful` (or `:not-useful`)
and routes the project board column. Awaiting-promotion is a user-only gate
(CLAUDE.md, gate 7) — I will not run it.
```

Then EXIT the skill. Don't ask "want me to do the next one?" — the user
should `/clear` (or open a new session) and re-invoke the skill on the
next issue, so each draft gets a clean context window.

---

## Decision tree: useful vs not-useful

The user usually knows. When they ask, the decision points are:

| Signal | Suggests `useful` | Suggests `not-useful` |
|---|---|---|
| Confidence | HIGH or MODERATE | LOW |
| Survives in the paper / a mentor talk | yes | no |
| Findings update someone's beliefs | yes | "we couldn't tell" |
| Replicates / strengthens a parent issue | yes | yes (still useful as evidence-of-inability) |
| Negative result with a sharp mechanism | useful | n/a |
| Negative result + binding-constraint excuse | n/a | not-useful |

`not-useful` is not "the experiment failed" — it's "the result isn't
load-bearing for the project's narrative." Failed experiments with sharp
mechanisms are usually `useful`. Successful experiments that turn out to
duplicate a prior finding without adding evidence are usually `not-useful`.

---

## On running fresh analyses or new plots mid-iteration

This is a normal part of the loop, not an exception.

If the user says "wait, I don't trust the figure 2 effect — can you
re-plot at higher resolution?" or "the matched-scaffold gap depends on
the outlier source, run it without police_officer," do it inline (or
spawn a sub-agent if the work is substantial):

1. Read the relevant `eval_results/<issue>/...json` from the local repo.
   Don't re-run the experiment unless explicitly asked.
2. For plots: invoke the `paper-plots` skill so styling stays consistent.
3. Commit the new figure to `figures/issue_<N>/...` (the body links commit
   SHAs, so commit before updating the body).
4. Update the body to point at the new commit / figure path (use the same
   `update_issue_body` MCP tool from Step 4).
5. Re-run `verify_clean_result.py`.
6. Loop back to Step 2 with the new evidence.

If the work is substantial (more than a one-shot re-aggregation),
**spawn a sub-agent** so the main context stays focused on the iteration
conversation. Pass the agent the issue number, the specific question,
and the eval-results path. Have it return a punch-list of what changed.

---

## Reference material in this skill

- `human-tldr-examples.md` — verbatim Human TL;DR exemplars from the
  Useful column (#276 / #295 / #281), with the structural commentary that
  tells you *why* each works. **Read this before every draft.**
- `lw-register-cheatsheet.md` — condensed pointers from the
  `clean-results/` skill: LessWrong style rules, anti-patterns, the
  paragraph-LEDE register, and how the Human TL;DR differs from the
  AI TL;DR.

For the full clean-results body conventions (template, principles,
iterations), defer to `.claude/skills/clean-results/` — this skill stays
narrow on the Human TL;DR + promotion handoff.
