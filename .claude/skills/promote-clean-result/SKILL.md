---
name: promote-clean-result
description: >
  Iterative, conversational workflow for promoting an `status:awaiting-promotion`
  clean-result issue to `clean-results:useful` (or `:not-useful`). First
  scans the Awaiting-promotion column for similar issues that should be
  consolidated into a single multi-claim body before promoting (the #237
  pattern); if none, walks one issue end-to-end: read the AI-drafted body
  (which under v4 already includes an AI-drafted `## TL;DR` in user-voice
  register per `human-tldr-examples.md`), **review and refine the AI's
  TL;DR with the user** on the interpretation / takeaways (not prose),
  then auto-critique the locked draft against `human-tldr-examples.md` +
  the LessWrong-register cheatsheet for the Summary / Details, apply the
  final body, then have the user run `python scripts/gh_project.py promote
  <N> useful|not-useful`. Use when the user says "promote #N", "let's
  clean up the awaiting-promotion column", "help me refine the TL;DR
  for X", or asks to move issues out of Awaiting promotion.
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
  TL;DR for X", or pastes a link to an `awaiting-promotion` issue.
- User wants to move multiple issues from Awaiting promotion through the
  Useful / Not useful columns. Run this skill **once per issue** — start a
  fresh context (`/clear`) between issues so each draft gets clean attention.
- The skill always runs Step 0.5 first: it scans the rest of the
  Awaiting-promotion column for issues whose findings should fold into
  the target as additional `### Result N` sections (the #237
  consolidation pattern). If there's a candidate group, the user is
  asked whether to merge before any TL;DR work begins. This
  catches the case where two reviewer-PASSed clean-results turn out to
  be one finding split across two issues.

## When NOT to use

- For drafting the Summary, Details, figures, or any other clean-result
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

List the full column and pull each issue's title + Summary + Source
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
   - Append each fold-in's `### Result N` block to the target's `## Details`, renumbering Result sections in narrative order (not
     issue-number order — the order a reader should traverse them).
     **Each `### Result N` section must open with a 1-3 sentence setup
     paragraph BEFORE the figure** (canonical since 2026-05-10 — see
     `.claude/skills/clean-results/template.md` § "Per-Result-section
     conventions" first bullet). Order: H3 → setup → figure → caption →
     findings prose → sample outputs. The setup names the specific
     experiment / arm / measurement that produced the figure; reader
     landing here cold from a TL;DR link must not need to parse the
     caption to learn what was done.
   - Add `## Source issues` H2 after `## Details` per
     `clean-results/template.md` § "Source issues" (CONDITIONAL).
   - Add `Source-issues: #N, #M1, #M2` and (if applicable)
     `Supersedes: <none>` lines at the very top of `### Background`.
   - Update the `## Summary` to the six-bullet
     structure (canonical convention since 2026-05-10 — see
     `.claude/skills/clean-results/template.md` § "Summary
     (human reviewed)" for the full rule). The six top-level bullets:
     **Motivation / Experiment / Results / Takeaways / Next steps /
     Confidence**. Per-result claim bullets indent two spaces under
     `**Results:**` (one sub-bullet per fold-in's load-bearing claim,
     in narrative order matching the renumbered `### Result N` H3s).
     Per-followup sub-bullets indent two spaces under `**Next steps:**`.
     Each Result sub-bullet follows the LW register: first-person,
     plain English, no project-internal jargon. Specifically drop
     from narrative prose: per-cell tags (`BS_E*`, `Z_*`, `B0`),
     extraction-method labels (`Method A` / `Method B` / `M1` / `M2`),
     judge / gate names (`G[0-9]+`, `K1 threshold`, `gate threshold =
     <N>`), math notation (`Δ`, `‖Δθ‖₂`, `log(...) covariate`,
     `p_exact`, `Spearman ρ`), AND the word "arm" / "experimental
     arm" / "behavioral arm" / "geometric arm" / etc. when used as a
     plan-internal experiment-strand label. Replace each with what
     was done in plain English ("the SFT-then-couple experiment",
     "last-input-token activations at layer 20", "judge accuracy on
     stripped-marker pairs", "well below the 40-point threshold the
     Betley protocol uses"). Plan-internal tags go in
     `<details><summary>Setup details</summary>` for reproducibility,
     not in narrative prose. No headline sentence or "In detail:"
     prose paragraph above the bullets — the six bullets carry the
     entire Summary section.
   - Update the title to the merged headline + `(HIGH | MODERATE | LOW
     confidence)` reflecting the lowest tier across the merged claims
     (be conservative). Confidence-tier exception per the 2026-05-10
     iteration: a LOW-confidence reframe that materially changes how to
     read the umbrella's other Results IS foldable, with explicit
     per-Result confidence framing in the Confidence bullet (e.g., "MODERATE
     on the umbrella; LOW on the {…} reframe specifically").
   - Headline numbers table: union the rows; keep the column schema.
   - Artifacts section: union the artifact links.
   - **Wrap every H2 and H3 body section in `<details open><summary>` blocks**
     so the heading is the click target on GitHub and the section is
     collapsible (heading-as-toggle convention, added 2026-05-09). Required
     wrapping: `## TL;DR`, `## Summary`, `## Source issues`, every `### Background`
     / `### Methodology` / `### Result N` / `### Next steps` H3. EXEMPT from
     wrapping: `## Details` (the container H2 with no body content — wrapping
     it would force a double-click to reach a Result). Pattern (the blank
     lines around the heading are required — they re-enable markdown
     parsing inside the HTML block):
     ```markdown
     <details open>
     <summary>

     ## Summary

     </summary>

     content...

     </details>
     ```
     The verifier's `Collapsible sections` check enforces this as a WARN — a
     PASS-with-WARN body will still promote, but treat the warn as a fix
     candidate for any new clean-result.
3. Show the user the proposed consolidated body (full draft, not a
   diff — diffs are too noisy across a multi-section merge). Iterate
   if they want different ordering / different headline framing.
4. When approved, apply via `gh_graphql` MCP `update_issue_body` (or
   fall back to `gh issue edit <N> --body-file <path>`). If both fail
   with "GraphQL: API rate limit already exceeded" (common after many
   body-edit passes in one session — GraphQL pool is 5000/hr separate
   from REST), fall back to the REST PATCH endpoint:
   ```bash
   gh api -X PATCH /repos/<owner>/<repo>/issues/<N> -F body=@<path>
   ```
   REST has a separate 5000/hr quota that's usually fully available
   when GraphQL is exhausted. See `reference_gh_rest_vs_graphql.md`
   memory entry for full quota breakdown (which commands drain which
   pool). Audit the per-pool quota with
   `gh api graphql -f query='query { rateLimit { remaining resetAt } }'`
   for GraphQL and `gh api /rate_limit` for both.
5. Run `uv run python scripts/verify_clean_result.py <N>` — if FAIL,
   fix the structural violation before continuing. Common fails after
   merge: missing `## Source issues` H2, mismatched confidence between
   title and Summary `**Confidence:**` line, missing Result sub-bullets
   under `**Results:**`, missing Takeaways or Next steps top-level
   bullets. For pre-cutoff (issue created before 2026-05-15) bodies that
   adopt the v2 shape, pass `--skip-checks check_results_block,check_human_summary,check_sample_outputs,check_reproducibility`
   to bypass the v1-mandatory section checks.

   Also run the companion auditor for project-internal jargon (the
   verifier doesn't catch this directly):

   ```bash
   uv run python scripts/audit_clean_results_body_discipline.py
   ```

   See Step 1 #6 for the full list of patterns it flags (`cell_tags`,
   `experimental_arm`, `condition_labels`, pre-reg jargon, etc.). Treat
   every flagged hit as a fix candidate before pushing the consolidated
   body. The auditor strips fenced code blocks before matching, so
   sample-output bracket labels in code are exempt by design.
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

Read the **whole** body, not just the Summary. You're checking:

1. **Is the body actually a clean-result?** It must have `## Summary` +
   `## Details` (with `### Background` / `### Methodology` / `### Result N`
   subsections) + a confidence-suffixed title. If the body still looks
   like the original plan (Goal / Hypothesis / Design — no Result section,
   no confidence in title), the issue is mis-labeled. STOP and tell the
   user — promotion would propagate a half-finished issue into the
   Useful/Not useful columns. Common cause: reviewer FAIL'd, status label
   drifted manually, or the experiment was abandoned mid-stream.
2. **Under v4 (2026-05-11+), the body arrives with an AI-drafted
   `## TL;DR` in user-voice register.** The analyzer writes 3-4 short
   bullets per `human-tldr-examples.md` before posting; this skill's
   job is to **review and refine** that AI draft with the user, not to
   draft from scratch.
   - **AI-drafted v4 TL;DR** (the common case under the new flow):
     read it; the substance refinement in Step 2-3 is now "does this
     AI bullet match the user's lineage memory of the experiment?",
     not "what should the bullets say?". The user is editing the AI's
     interpretation, not generating one.
   - **Pre-v4 issues with a user-only placeholder** (verbatim line
     `_(TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_`):
     legacy grandfathered case; treat as "draft from scratch" using
     the same exemplars (`human-tldr-examples.md`). Step 4 § 3 covers
     the placeholder-replace mechanics.
   - **Pre-v4 issues with a user-drafted TL;DR** (labeled `## Human TL;DR`
     in older drafts, `## Human TL;R` / `## Human Summary` typos, or any
     variant of the user-voice block): preserve the existing user content,
     propose only minor cleanup. The v4 canonical heading is just
     `## TL;DR`; the older `Human TL;DR` / `AI TL;DR` / `AI Summary` triad
     has been retired.
3. **Are there obvious overclaims, missing caveats, or mismatched figures?**
   Flag them before drafting the TL;DR — sometimes the right move is
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
   and `### H3` body section (except `## Details`, the container) should
   sit inside a `<details open><summary>` block whose `<summary>` carries
   the markdown heading itself, so the heading is the click target on
   GitHub and the section is collapsible. Pattern (note the blank lines
   that re-enable markdown parsing inside the HTML block):

   ```markdown
   <details open>
   <summary>

   ## TL;DR

   </summary>

   content...

   </details>
   ```

   Run `uv run python scripts/verify_clean_result.py <N>` and look for the
   `Collapsible sections` check. If it's WARN ("N section(s) not wrapped"),
   wrap them as part of the promotion pass — do not leave un-wrapped
   sections in the final body. Common offenders: `## TL;DR`, `## Summary`,
   `## Source issues` (the verifier exempts `## Details` as the container).
   Offer to wrap them as part of the promotion
   pass. Pre-2026-05-09 drafts are grandfathered (verifier WARNs, doesn't
   FAIL), but new drafts and any draft you're touching for promotion
   should adopt the convention. See `clean-results/template.md` § "Body
   shape" for the canonical pattern + which headings are exempt.
6. **Does the body have any project-internal jargon the auditor would flag?**
   Run the companion auditor alongside the verifier:

   ```bash
   uv run python scripts/audit_clean_results_body_discipline.py
   ```

   The auditor greps the bodies of every issue currently in the
   `Awaiting promotion` column for ~13 anti-pattern classes — pre-reg
   labels, CAPS verdict labels (`REJECTED` / `INDETERMINATE`),
   Δ-percentage-point notation, inline credence intervals, named statistical
   tests, statistical-hypothesis symbols (`H_a` / `H_0`), anaphoric letter
   labels, bare bin labels, `C1`/`H1`/`P1` project-internal condition /
   hypothesis labels, bare methodology acronyms (`GCG` / `PAIR`), bare
   statistical acronyms (`OLS` / `ROC`), `AUC = X.XX` without classification
   target, `post-hoc` / `ex post` framing, math-style subscript/superscript
   in prose, AND (added 2026-05-10 after #237):

   - `cell_tags` — per-cell / extraction-method / judge / gate tags like
     `BS_E0..E4`, `Z_assistant`, `Z_villain`, `B0`, `Method A` / `Method B`,
     `M1` / `M2`, `G6` / `G0a`, `K1 threshold`, `gate threshold = 40`.
     Replace with plain English ("the 5 benign-SFT-then-couple cells",
     "last-input-token activations", "judge accuracy on stripped-marker
     pairs"); plan-internal tags belong in `<details><summary>Setup
     details</summary>` for reproducibility, not narrative prose.
   - `experimental_arm` — the word "arm" / "arms" used as a project-internal
     experiment-strand label ("the forward-order behavioral arm",
     "five experimental arms", "the LoRA geometric arm"). Replace with
     what was done ("the couple-then-SFT experiment", "five experiments").

   The auditor strips fenced code blocks before matching, so sample-output
   bracket labels (e.g. `[FIRING bystander: persona=..., cell=BS_E0]`
   inside a ```\`\`\`-fenced block) are exempt by design — those are
   reproducibility artifacts. The auditor flags narrative prose only.
   Auditor output goes to `.claude/cache/audit-<date>/findings.md` with
   per-issue violation lists; treat every flagged hit as a fix candidate
   before pushing. Pattern names like `cell_tags` / `experimental_arm`
   are the auditor's internal identifiers — invoke the auditor by its
   script name (above), not by pattern name.

## Step 2 — Review the AI-drafted TL;DR (or, on legacy issues, draft from scratch)

**Under v4 (the common case):** the analyzer has already drafted a
user-voice `## TL;DR` per `human-tldr-examples.md`. Your job is to
**propose refinements to the AI's draft**, not redraft from scratch.

**Under legacy / placeholder (the grandfathered case):** if Step 1 § 2
found a verbatim placeholder line or no TL;DR at all, draft fresh from
the exemplars in `human-tldr-examples.md`. The shape rules below apply
to both modes — the only difference is whether you're refining
existing bullets or writing them from blank.

The user-voice TL;DR shape:

- **Bullet 1** opens with the question / what we tested / what we wanted
  to see — never the result.
- **Bullet 2** is the headline finding, plainly stated, often negative
  ("It did not", "actually flipped", "no effect").
- **Bullet 3+** is optional: a surprise, a side-finding, an anomaly worth
  flagging. Often ends "worth investigating further" / "probably due to X".
- First-person, present tense, casual punctuation. No `r =`, no `p =`, no
  `(MODERATE confidence)`. Bold-emphasis allowed for the load-bearing word.
- ~30-90 words total. Shorter is better.
- The TL;DR is the casual-scan layer; the Summary carries the precise
  paragraph lede, the TL;DR carries the colloquial "shoulder-tap to a peer"
  framing. It's OK for the TL;DR's tone to subtly differ from the Summary's.

Present in chat. **Do not edit the issue body yet.** Show:

1. **The current `## TL;DR` block** (verbatim from the issue body, fenced
   as markdown). If it's the placeholder line, say "AI draft missing —
   drafting from blank below" and skip to (3).
2. **Diagnosis of what you'd change** — 2-4 short bullets naming
   substance issues only (not prose). Examples: "Bullet 1 says 'tested
   X' but the actual scope was X+Y", "Bullet 2's headline finding
   inverts the direction — the result was *more* not *less*", "Missing
   the police_officer outlier surprise that's load-bearing here".
3. **The proposed revision** — full `## TL;DR` block, fenced. Diff-style
   pointers ("changed bullet 1 to lead with the question, added bullet 3
   for the outlier") are useful but optional.
4. 1-3 framing knobs you had to make a call on — each as a quick yes/no
   question the user can answer in one breath. Examples:
   - "Lead with the metric name or with the question?"
   - "Do you want the surprise bullet, or just the headline?"
   - "Flag the police_officer outlier here, or save it for the body?"
5. If you spotted body issues in Step 1, list them under a separate
   `### Heads-up` block. Don't bury them.

If the AI draft is already exemplar-shaped and you have no substantive
notes, say so explicitly ("the AI draft reads correctly to me — proposing
no substance changes") and pass it through unchanged to Step 3.

## Step 3 — Iterate on **substance** with the user

This step is for locking down **what the main interpretation and
takeaways should be** — *not* for prose polish. Save the prose pass for
Step 3.5 (auto-critique), which runs without the user in the loop.

Under v4 this is usually a **2-3-turn refinement** of the AI's existing
draft (often "looks good, just tweak bullet 2 to say X"), not the
multi-turn draft-build the workflow used to expect. If the user
accepts your Step 2 proposal verbatim, jump straight to Step 3.5.

Common shapes the iteration takes:

- **Reframe / different emphasis.** "The headline should be the
  cosine-vs-JS dissociation, not the cross-section win." Re-draft,
  re-show, re-ask.
- **Accept the AI's interpretation verbatim.** "AI got it right, ship
  it." Substance is locked; move to Step 3.5 immediately.
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
- Whether numbers leaked in from the Summary.
- Whether the opening verb is right ("Tested" vs "Checked" vs "Wanted to see").

Each substance iteration: re-show the full proposed block (don't make
the user mentally diff). State explicitly which knobs you changed.

When the user says "yep, that's the right interpretation" / "ship it" /
"good, apply it" — substance is locked. Move to Step 3.5 immediately.
DO NOT apply yet.

## Step 3.5 — Re-run clean-result-critic against the substance-locked draft (no user loop)

Once substance is locked in Step 3, re-run the `clean-result-critic`
agent against the latest body (including the user's substance edits
from Step 3). This is the same agent that ran during `/issue` Step
9a-bis — but the body has changed since then, so a fresh pass is
required.

Why this exists at promotion time: the user's Step 3 edits may have
introduced TL;DR contamination (e.g., a number drifted back from the
Summary), broken the title's declarative-opener rule (e.g., the user
reframed and the new title starts with `If you...`), or removed the
surprise bullet that the AI draft passed. The critic catches those
before the user promotes.

### How to run

Spawn the agent (fresh context; sees the clean-result body, NOT
analyzer reasoning, NOT this skill's conversation). Pass the
clean-result issue number. The agent runs its 10 lenses, including
`scripts/verify_clean_result.py` and
`scripts/audit_clean_results_body_discipline.py` internally, and
posts `<!-- epm:clean-result-critique vN -->` on the SOURCE issue
(per `markers.md`).

### What to do with the verdict

- **PASS:** show the user a one-line "critic re-pass: PASS" confirmation
  and advance to Step 4 (apply final title + TL;DR).
- **REVISE:** read the agent's "Specific revision requests" block. For
  each request, propose the fix in chat as a single diff block (title +
  TL;DR + any Summary / Details edits the critic flagged). Show the
  user:

  ```
  ### clean-result-critic — Round N (PASS / REVISE)

  Verifier: <PASS or FAIL summary>
  Audit script: <N patterns flagged>

  Critic findings (verbatim from the agent):
  - <lens N>: <quote> — <fix>
  - ...

  Proposed revisions (apply all unless you push back):

  CURRENT TITLE: <verbatim>
  PROPOSED:      <rewrite, or "no change">

  CURRENT TL;DR: <verbatim block>
  PROPOSED:      <revised block>

  CURRENT <other surface>: <verbatim>
  PROPOSED: <revised>
  ```

- **Do not loop with the user on style.** If they push back on a
  specific edit ("no, keep the `Δ` here, I want the number"), apply
  just that override and move to Step 4. Don't re-run the full
  critique. Don't ask "is that ok now?" — the user is allowed to
  override the critic; your job is to surface the critic's findings
  and let the user decide.
- **Don't re-spawn the agent more than once at promotion time.** If
  round 1 was REVISE → user override → apply, move on. The agent
  already ran up to 3 rounds at `/issue` time; the promotion re-pass
  is a safety check against drift introduced by Step 3 edits, not a
  second iteration loop.

The agent's checks cover everything the v3 Step 3.5 checklist used to
do by hand (declarative title opener, TL;DR voice + headline structure
+ no-statistics rule + casual-punctuation rule, Summary six-bullet
ordering, per-Result setup-before-figure, body-discipline anti-patterns
from the audit script, etc.). The single source of truth for what
"clean-result-shaped" means is the agent's lens list — kept in sync
with `template.md` whenever the template changes.

## Step 4 — Apply the final title + TL;DR to the issue

When the user signs off on Step 3.5:

1. **If the title was rewritten in Step 3.5**, apply the title change
   FIRST via `gh issue edit <N> --title "<new title>"` (or the
   `gh_graphql` MCP equivalent when it ships). The title and the body
   are independent edits — title first means a partial failure can't
   leave a body referencing a stale title.
2. Re-fetch the latest body (`gh issue view <N> --json body`) — the user
   may have edited it in the browser while you iterated.
3. **Replace the existing `## TL;DR` section content with the approved bullets.**
   The mechanics depend on what's already in the body:
   - **v4 AI-drafted TL;DR (the common case):** the body has
     `## TL;DR\n- <AI bullet 1>\n- <AI bullet 2>\n...`. Replace the
     entire bullet block under the H2 with the user-approved bullets.
     The H2 line stays; only the bullet content changes.
   - **Legacy placeholder:** replace
     `_(TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_`
     with the approved bullets.
   - **Typo'd / duplicate user blocks** (`## Human TL;R`, legacy
     `## Human TL;DR` above a v2 `## TL;DR`, etc.): REMOVE the
     duplicates/typos, leaving exactly one `## TL;DR` H2 followed by
     the approved bullets.
   - **Legacy triad rename** (`## Human TL;DR` / `## AI TL;DR (human
     reviewed)` / `## AI Summary`): rename to the v4 triad `## TL;DR` /
     `## Summary` / `## Details` in the same edit pass. The rest of
     the body (H3 subsections, figures, Source issues block) carries
     over unchanged.

   **CRITICAL — strip the v1 headline + "In detail:" paragraph during the
   rename.** The v1 AI TL;DR section opened with a title-restatement
   sentence, followed by an "In detail:" prose paragraph dense with
   per-condition numbers, then the 5-6 bullets. The v2 Summary has no
   headline sentence and no "In detail:" prose paragraph — the bullets
   carry the entire section (canonical: `lw-register-cheatsheet.md`
   line 37-38; `clean-results/template.md` line 83 "No headline prose,
   no 'In detail:' paragraph — the bullets carry the entire section";
   `clean-results/iterations.md` 2026-05-10 #237 and 2026-05-11 #186
   entries). The renamed `## Summary` MUST open directly with
   `- **Motivation:**`. The Motivation bullet carries the paragraph-LEDE
   framing the v1 headline sentence used to carry; deleting both
   paragraphs is not an information-loss edit, just a v2 transition.
   If you leave them in, the user will catch it on review and bounce
   the promotion back (issue #186 was the precedent).
4. **If the title was rewritten** AND the body still has a v1-era
   Summary headline sentence (because you're mid-transition or
   intentionally preserving it for a body that hasn't been cleaned up
   yet), keep the headline sentence in sync with the new title
   (verbatim minus the confidence suffix per
   `clean-results/template.md`). In v2 bodies the rule is moot — there
   is no Summary headline sentence to keep in sync, because the
   Motivation bullet carries the paragraph-LEDE framing and the title
   carries the verb-of-finding lede. Apply this rule only if the body
   hasn't yet completed the v2 transition.
5. Update via the `gh_graphql` MCP tool (`update_issue_body`) so the token
   never enters the agent context window. Fall back to
   `gh issue edit <N> --body-file -` only if MCP is unavailable. If
   GraphQL is rate-limited (common after a long iteration loop —
   GraphQL pool is 5000/hr, separate from REST), fall back further to
   the REST PATCH endpoint:
   ```bash
   gh api -X PATCH /repos/<owner>/<repo>/issues/<N> -F body=@<path>
   ```
   See `reference_gh_rest_vs_graphql.md` memory entry.
6. Echo back the diff in chat (title change + TL;DR section + Summary sentence-1 change if applicable; never the whole 65k-byte body).
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

- `human-tldr-examples.md` — verbatim TL;DR exemplars from the
  Useful column (#276 / #295 / #281), with the structural commentary that
  tells you *why* each works. **Read this before every draft.**
- `lw-register-cheatsheet.md` — condensed pointers from the
  `clean-results/` skill: LessWrong style rules, anti-patterns, the
  paragraph-LEDE register, and how the TL;DR differs from the
  Summary.

For the full clean-results body conventions (template, principles,
iterations), defer to `.claude/skills/clean-results/` — this skill stays
narrow on the TL;DR + promotion handoff.
