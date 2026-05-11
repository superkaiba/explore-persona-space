---
name: lw-review
description: >
  Review any prose against LessWrong / Alignment Forum / Anthropic-blog
  research-communication principles (Nanda, Perez, Chua, Hughes, Evans,
  Apollo / LW lede register). Works on emails, blog posts, paper
  sections, abstracts, talk scripts, Slack messages, slide bullets,
  Twitter threads — anything that should read as "research-communication
  English" rather than AI-generated mush. Identifies the target register,
  loads the relevant rules from `clean-results/principles.md` +
  `lw-tldr-examples.md` + `promote-clean-result/lw-register-cheatsheet.md`,
  runs a checklist, then proposes a revised version with edits visible.
  Iterative, human-in-the-loop. Use when the user says "review this for
  LW", "apply LW principles to X", "make this less AI-sounding", "tighten
  this prose for clarity", "polish this for a LW post", or pastes any
  draft text and asks for a register pass.
user_invocable: true
---

# LessWrong-style review

A general-purpose register check for any prose the user is working on.
Reuses the rule sets that the clean-result pipeline applies to issue
bodies — but un-scoped: the target text can be an email draft, a paper
abstract, a slide bullet, a Slack message, anything.

This is a **skill, not an agent**: it loads the principles into the
current context and applies them in conversation. If the user wants
adversarial fresh-context review (separate context window, no shared
reasoning) of a clean-result issue body, they should spawn
`clean-result-critic` directly via the Agent tool — that agent exists
for exactly that purpose, but only covers clean-result bodies. This
skill is the broader, conversational version.

The workflow is **iterative and human-in-the-loop**: identify, propose,
discuss, revise, apply. Don't rewrite without sign-off.

---

## When to use

- User pastes any draft text + "make this LW-style" / "tighten this" /
  "less AI-sounding" / "review for clarity".
- User points at a file or section ("review §3 of the paper", "review
  my email draft at /tmp/draft.md").
- User wants principle-grounded edits, not vibes-based polish — i.e. they
  want each edit attached to a named rule (Perez "drop hedging", Nanda
  "compress to load-bearing claims", Chua "comparison anchor missing").
- User is drafting an artifact that doesn't yet have a dedicated skill —
  emails, mentor-meeting agendas (outside the deck), blog posts, talk
  scripts, paper sections, conference reviews.

## When NOT to use

- **Code review / docstring review.** Use `/code-review` or
  `/cleanup` — the principles here are about prose, not API design.
- **Paper-level structural review.** Use `ml-paper-writing` for ICML /
  NeurIPS / ICLR submission-shaped review — venue-specific rubrics live
  there. This skill handles paragraph-level register; the paper skills
  handle figure placement, citation density, related-work coverage.
- **Mentor slide deck.** Use `/mentor-update-slides` — the Marp deck
  has its own anchored-region conventions.

---

## Step 0 — Identify the target text and target register

Establish two things before reading any rules:

1. **What's the target text?** A file path (read it), a region of an
   open file (read just that region), pasted text (use as-is), or "the
   thing we just drafted in this conversation" (scroll back to it).
2. **What register is the user aiming for?** Different artifacts have
   different bars. Ask if ambiguous; default per file shape:

| Artifact | Target register | Heaviest rule sources |
|---|---|---|
| LessWrong / Alignment Forum post | LW research-post lede | full `principles.md`, `lw-tldr-examples.md` |
| Anthropic / Apollo blog post | Same as LW post | full `principles.md`, `lw-tldr-examples.md` |
| Paper abstract / intro | Dense specialist + clear lede | Perez clarity rules; Nanda "compress to claims" |
| Paper body section | Dense specialist | Perez clarity rules; minimize pronouns; one idea per sentence |
| Talk script | Spoken English; sentence-verb titles | Chua/Hughes slide rules adapted for narration |
| Email (research) | Casual but precise; comparison anchors when stating numbers | Perez clarity + comparison-anchor rule |
| Slack message | Like an email but shorter; bullets allowed | Perez clarity + jargon-density rule |
| Tweet / thread | Lede sentence is everything | LW lede register; no project-internal compounds |
| Slide bullet | ≤8 words; verb-first; sentence-case | Chua/Hughes "limit words per slide" |
| Mentor agenda | Bullet list; agenda-before-content | Chua/Hughes "agenda next" |

If the file extension or directory hints at the register (`.tex`,
`paper/`, `figures/mentor-slides/`, `drafts/lw-post/`, etc.), use that
heuristic and STATE the assumption in one line so the user can correct
("Assumption: this is a LessWrong post draft. Targeting full LW register.").

State both back to the user in one short paragraph before going further:
"Reviewing `<path>` (lines `<x-y>`) as a `<register>` draft. Loading
`<which references>`."

## Step 1 — Load the relevant references

Always read (or re-skim, if already in context):

- **`.claude/skills/clean-results/principles.md`** — the canonical source
  for Nanda / Perez / Chua / Hughes / Evans rules. The Perez clarity
  block (minimize pronouns, verbs early, drop hedging, one idea per
  sentence, active voice, etc.) applies to every register. The Nanda
  block (compress to load-bearing claims, present limitations honestly,
  red-team alternatives) applies to anything making a research claim.
- **`.claude/skills/promote-clean-result/lw-register-cheatsheet.md`** —
  the "five LW-style rules that catch most drift", title rules,
  anti-pattern catalog, the colloquial-LEDE vs dense-specialist register
  table. Most of the rules generalize beyond clean-result bodies.

Conditionally read (only when relevant to the target register):

- **`.claude/skills/clean-results/lw-tldr-examples.md`** — verbatim LW
  exemplars. Read when reviewing a TL;DR, abstract, or post lede; the
  shape of the exemplar bullets is the target.
- **`.claude/skills/clean-results/paper-caption-examples.md`** — read
  when reviewing a figure caption.
- **`.claude/skills/mentor-update-slides/principles.md`** — read when
  reviewing slide bullets or a talk script (Chua / Hughes / Sanders /
  Alley assertion-evidence rules live there).

Don't load every reference every time — register-load only what fits the
target. Cite each rule by name (e.g. "Perez: drop hedging") in Step 3.

## Step 2 — Read the target text closely

Read the full text. For each paragraph / bullet / sentence note:

- **What's the load-bearing claim?** If you can't identify it in 5
  seconds, the prose has compressed badly.
- **Does it have a comparison anchor when it states a number?** Bare
  numbers ("the model improves to 73%") fail; anchored numbers ("73% vs
  41% baseline, n=600") pass.
- **Are there project-internal compound nouns / acronyms / taxonomy
  labels** (`cosine-L20`, `BPE-prefix-bound`, `c1_evil_wrong_em`,
  `Bin A/B/C`) that would lose a low-context reader?
- **Are there hedges** (`may`, `can`, `could`, `to our knowledge`,
  `note that`, `actually`, `a bit`, `fortunately`) that Perez says to
  drop almost always?
- **Is the voice consistent?** Mixing first-person plural ("we tested")
  with third-person passive ("it was tested") within the same paragraph
  is a frequent AI-tell.
- **Are there AI-mush phrases?** "Delve into", "tapestry", "robust",
  "leverage", "underscore", "navigate the complexities", "in the
  rapidly evolving landscape of", "it is worth noting that", "it is
  important to consider that". These flag the text as model-generated
  and lower research-communication credibility.
- **Is the lede sentence load-bearing?** Most readers stop after
  paragraph 1. The first sentence should carry the headline; the
  second elaborates; the third caveats.

## Step 3 — Run the checklist (cite each rule by name)

Walk top to bottom. For each issue, write a one-line finding in this
shape: `<location>: <rule name>: <issue>`. Skip clean passes — only
list problems.

### Universal block (every register)

1. **Perez — minimize pronouns.** "this", "it", "these" used as bare
   subjects. Convert to adjective form ("this finding", "this result").
2. **Perez — verbs early.** Front-loaded subordinate clauses delay the
   verb. Restructure to lead with subject + verb.
3. **Perez — drop hedging.** `may`, `can`, `could`, `might`, `seems
   to`, `appears to`, `tends to`, `to our knowledge`, `we believe`,
   `note that`, `it is worth noting`. Drop unless the hedge is
   load-bearing.
4. **Perez — drop intensifiers and AI-tells.** `actually`, `a bit`,
   `fortunately`, `interestingly`, `notably`, `crucially` (when not
   actually crucial), `delve`, `tapestry`, `leverage` (as verb),
   `robust`, `nuanced`, `it is important to`, `in the rapidly evolving`,
   `navigate the complexities`. Strip.
5. **Perez — active voice, specify the actor.** "The data was
   analyzed" → "We analyzed the data" or "scripts/analyze.py
   aggregates the data".
6. **Perez — one idea per sentence.** Sentences with three commas + a
   semicolon usually carry two ideas; split.
7. **Perez — never bare comparatives.** "improved", "more accurate",
   "faster" without a baseline. Add the baseline or drop the
   comparative.
8. **Nanda — compress to load-bearing claims.** Three+ adjacent
   sentences that don't update beliefs → compress to one.
9. **Nanda — write to inform, not persuade.** Persuasive flourishes
   ("a powerful demonstration", "compelling evidence") overstate;
   replace with the evidence itself.

### Numbers block (every register that reports results)

10. **Chua — comparison anchor for every number.** "73%" → "73% vs 41%
    baseline" or "73% vs 50% chance". Bare numbers fail.
11. **Chua — N alongside proportions.** "73%" → "73% (n=600)" when N
    is informative.
12. **Sanders — error bar on the delta, not the endpoints.** When the
    claim is a difference, report ±err on the difference.

### Lede block (TL;DRs, post intros, abstracts, email subject lines)

13. **LW lede register.** First sentence is plain English; second
    sentence may be dense / specialist; third sentence caveats. If the
    first sentence is dense, restructure.
14. **No conditional / hypothetical opener** in titles or post titles
    or LW-post first sentences. "If you...", "When you...", "Suppose
    you..." — convert to declarative gerund or noun-phrase opener
    (see `lw-register-cheatsheet.md` § "Title rules" for the recipe).
15. **No negation-of-prior-claim title.** "X does NOT do Y" → state
    the affirmative finding.
16. **Define every uncommon acronym inline on first use.** "EM
    (Emergent Misalignment)", "RM (reward model)". Banish naked
    acronyms from titles and ledes.
17. **No project-internal compound nouns in the lede.** "Cosine-L10",
    "matched-scaffold leakage", "diff-of-diffs". Use the plain phrase
    or define inline.

### Bullets / lists block (TL;DR bullets, slide bullets, agenda items)

18. **Bullet length.** 1-2 sentences each, ~15-30 words. Slides cap at
    ~8 words (Chua/Hughes).
19. **Bullets parallel.** All start with the same shape (verb / noun /
    "We ..."). Mixed shapes read as drafty.
20. **Bullets non-redundant.** If bullet 3 paraphrases bullet 2, drop
    or replace.
21. **Sentence-verb titles** (slides). "Loss decreases with scale", not
    "Loss curves". If the slide is a finding, the title states the
    finding (Sanders, Alley assertion-evidence).

### Spoken-register block (talk scripts, mentor agendas)

22. **Read it aloud.** If you trip on a clause when reading, the
    audience will too.
23. **Numbers spoken in full.** "Seventy-three percent", not "73%".
24. **Forward references rather than back references.** "Next, we
    show..." not "As I said earlier...".

### Honesty block (any text making a research claim)

25. **Nanda — limitations stated.** Single seed, in-distribution eval,
    confounds — surface in the same paragraph as the claim, not three
    paragraphs later.
26. **Nanda — alternative explanations red-teamed.** If the claim is "X
    causes Y", the prose names at least one not-X confound and dismisses
    it on evidence.
27. **No "the data confirms" framing for null results.** "Indistinguishable
    from null given the variance" / "noise-limited" — not "confirms the
    null". (Saved memory: noise-limited, not effect-confirmed.)

For each finding, name the rule. The user should be able to see
"oh, that's a Perez clarity rule, fair" or "that's a comparison-anchor
rule, fair".

## Step 4 — Show findings + propose a revised version

Present in chat in this exact shape:

```
### Reviewing <path> (lines <x>-<y>) as <register>

#### Findings (rule cited)

- §1 ¶2: Perez — drop hedging: "we believe this may indicate" → "this indicates"
- §1 ¶3: Chua — comparison anchor: "improved to 73%" → "73% vs 41% baseline"
- §2 ¶1: LW lede: leads with conditional ("If you train...") → use declarative
- §2 ¶4: Nanda — compress: 4 adjacent sentences making one claim, can be 1-2
- ...
- (or "no issues found, prose is exemplar-shaped" — state explicitly when true)

#### Proposed revision

<the rewritten text, ready to apply>
```

Keep the proposed revision as a clean rewrite — don't show inline diffs
unless the rewrite is small and a diff is more readable than a side-by-
side. For long sections, paste the full revised section so the user can
visually compare.

If the user asked for review only (not edits), STOP HERE — don't
apply, don't iterate. Just present findings + revision and let them
take it from there.

## Step 5 — Iterate on the revision

Common iteration shapes (mirror the `promote-clean-result` Step 3
pattern):

- **"Push back on rule N — keep the original wording for X."** Apply
  the override, re-show. Don't argue the rule when the user has decided.
- **"Shorter."** Compress further. Re-show.
- **"Add back the hedging in this specific spot — the claim is
  genuinely uncertain."** Keep that specific hedge; strip the rest.
- **"Wrong register — this is for Twitter, not LW."** Re-classify and
  reload references. Re-run from Step 1.

Each iteration: re-show the full revised text (don't make the user
mentally diff). State which rules you re-applied or relaxed.

When the user signs off, move to Step 6.

## Step 6 — Apply the revision (only if the target is a file)

If the target was pasted text, just leave the final revision in chat —
the user copy-pastes it back. **Do not write a file the user didn't
ask you to write.**

If the target was a file path or a region of a file:

1. Re-read the file (the user may have edited it during iteration).
2. Edit the targeted region with the approved revision (use `Edit` for
   small surgical changes; use `Write` only if the user asked for a
   full-file rewrite).
3. Echo the diff in chat (the actual lines that changed, not the full
   file).
4. If the file is part of a build / verifier flow (e.g. a clean-result
   issue body), apply the revision and then run the relevant verifier
   (e.g. `uv run python scripts/verify_clean_result.py <path>`) before
   handing back. Surface any FAIL to the user before posting upstream.

Don't run linters / formatters / build commands as part of this skill.
The user runs those if they want.

---

## Decision tree: when to spawn `clean-result-critic` instead

Use `clean-result-critic` (the agent) instead of this skill when:

- You want adversarial fresh-context review with no shared reasoning
  ("review this without seeing how I argued for it") of a clean-result
  issue body. The agent cannot see the surrounding conversation; this
  skill can.
- You're inside `/issue` Step 9 and the clean-result-critique loop is
  the current phase — the skill spawns the agent automatically.

Use this skill when:

- You want conversational, in-context review with the ability to discuss
  edits inline.
- You want to apply LW principles + Perez clarity rules + Chua
  comparison-anchor rules to one paragraph or one email or one slide
  without spinning up a full critic.

---

## Reference material

This skill OWNS no rule files — it loads from the canonical sources to
keep one source of truth:

- **`.claude/skills/clean-results/principles.md`** — Nanda, Perez, Chua,
  Hughes, Evans, Benton. Read at the start of every invocation.
- **`.claude/skills/promote-clean-result/lw-register-cheatsheet.md`** —
  five LW-style rules, title rules, anti-pattern catalog, register
  table. Read at the start of every invocation.
- **`.claude/skills/clean-results/lw-tldr-examples.md`** — verbatim LW
  exemplars. Read when target is a TL;DR / lede / abstract.
- **`.claude/skills/clean-results/paper-caption-examples.md`** — read
  when target is a figure caption.
- **`.claude/skills/mentor-update-slides/principles.md`** — read when
  target is slide / talk content.

If you find a recurring rule that fires on text outside clean-results
and isn't already in any of the above, surface it to the user as a
candidate addition to `principles.md` (or its proper home) — don't
silently write a parallel rule list here. Single source of truth.
