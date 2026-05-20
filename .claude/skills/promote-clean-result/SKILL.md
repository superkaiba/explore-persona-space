---
name: promote-clean-result
description: >
  Workflow for moving a `status:awaiting_promotion` experiment to
  `useful` (or `not-useful`). First scans the column for consolidation
  candidates (the #237 pattern); if none, auto-converts legacy-markdown
  bodies to legacy Sagan-card HTML (grandfathered), runs `verify_task_body.py`, pushes to the
  experiment, iterates with the user, then hands off the manual
  `task.py promote` command. Use when the user says "promote
  #N", "clean up Awaiting promotion", or "help me refine the TL;DR
  for X".
user_invocable: true
---

# Promote a clean-result

Awaiting-promotion experiments are clean-result-critic-PASSed bodies parked at `status:awaiting_promotion` with `has_clean_result=true` and `classification='pending'`. Promotion is **user-only** by design (CLAUDE.md gate 7) — `/issue` cannot move them. This skill gets one experiment ready and exits to the user's terminal for the actual `task.py promote` command.

**Apply first, iterate after.** Don't pre-propose every TL;DR draft in chat — auto-convert + auto-draft, run the quality gates, push to the experiment body, then iterate against the live body. The only pre-apply user gate is Step 0.5 (consolidation candidates), because merging is destructive.

**All body-shape rules live in `~/sagan/docs/clean-result-guidelines.md`.** This skill is workflow + apply mechanics only. When in doubt about what a section should look like, read that doc.

---

## When to use

- User says "promote #N", "let's clean up Awaiting promotion", "write the TL;DR for X", or pastes a EPS dashboard link to an awaiting_promotion experiment.
- One experiment per skill invocation — `/clear` between experiments so each draft gets clean attention.

## When NOT to use

- For drafting figures or analysis content. That's the analyzer / interpretation-critic / clean-result-critic loop's job.
- For posting net-new clean-results. That's `/issue`'s analyzer step.
- For changing the `useful` / `not-useful` decision after promotion. Re-run `task.py promote` directly.

---

## Step 0 — Identify the target experiment

If the user named a specific experiment number, use it. Otherwise list the column:

```bash
uv run python scripts/task.py list-by-status --status awaiting_promotion
```

Pick one with the user. Don't try to do all of them in one context.

## Step 0.5 — Scan for consolidation candidates

Before reading the target end-to-end, check whether the column contains other experiments whose findings should fold into the target as additional Result sections inside its `#design` block (the #237 pattern). Promote-by-merge is preferable to promote-as-siblings when the experiments share a single load-bearing claim — separate clean-results fragment the narrative.

### Gather candidates

```bash
uv run python scripts/task.py list-by-status --status awaiting_promotion
# For each candidate (top ~10 nearest in title):
uv run python scripts/task.py view <Ni>
```

### Similarity signals (any 2+ → propose merge)

- **Shared parent** — both link to or cite the same prior experiment in their TL;DR Motivation bullet.
- **Shared experimental geometry** — same persona set, eval rig, matcher, model. Seed / sweep-point deltas are NOT independent claims.
- **Shared headline verb in titles** — "X does Y" + "X also does Z" + "X under W" = three sentences of one paragraph.
- **One refutes / strengthens / qualifies the other** — the second body's findings should be the qualifier in the first's design block, not a separate clean-result.

### Anti-signals (KEEP separate)

- Different mechanisms / different load-bearing claims, even if the parent is shared.
- Different confidence tiers (HIGH + LOW) where merging would force a single tier on the title.
- One is `useful`-bound, the other `not-useful`-bound — verdicts are per-experiment.
- Bodies have already diverged structurally (different figure sets, different design narratives). Merging would be a rewrite.

### Propose to the user (don't act yet)

```
### Consolidation candidate detected

Target: #N

Likely fold-ins:
- #M1 — <title gist> — shared signal: <which similarity rule>
- #M2 — <title gist> — shared signal: <which similarity rule>

Proposed shape: consolidated body in #N with:
- Primary figure: <which one>
- Design dropdown carries: <merged narrative of N + M1 + M2>
- #M1, #M2 archived as superseded after merge

(a) merge as proposed, (b) merge subset, (c) keep separate, (d) re-pick target?
```

Wait for explicit sign-off — merging archives sibling experiments, irreversible without manual undo.

### If user picks merge (a or b)

1. Re-read each contributing experiment's body in full — need figure URLs, headline numbers, methodology deltas.
2. Construct the consolidated HTML body in a scratch buffer at `.claude/cache/experiment-<N>-clean-result.html` (same auto-restructure pipeline as Step 2; merge just unions the design-narrative content first).
3. Show the user the full draft (not a diff — too noisy across multi-section merge).
4. When approved, apply via `task.py set-body <N> --file .claude/cache/experiment-<N>-clean-result.html`.
5. `uv run python scripts/verify_task_body.py --issue <N>` — fix any FAIL.
6. Archive each fold-in:
   ```bash
   uv run python scripts/task.py post-marker <Mi> epm:consolidated-into \
     --note "Consolidated into #N — see the design narrative."
   uv run python scripts/task.py set-status <Mi> archived \
     --note "Superseded by #N."
   ```
7. Continue from Step 1 with the consolidated body as the new target.

### If user picks (c) keep separate or (d) re-pick

Continue from Step 1 with the original (or newly-chosen) target.

---

## Step 1 — Read the body (silent diagnostic, no chat output)

```bash
uv run python scripts/task.py view <N>
```

Read the whole body. Diagnostic checklist (track internally, address in Step 2):

1. **Is the body actually a clean-result?** Must carry an articulated finding + a figure + the source-experiment context. If it still looks like the original plan (Goal / Hypothesis / Design, no Result section, no confidence in title) — **STOP and tell the user.** This is the only Step 1 condition that aborts the skill. Common cause: clean-result-critic FAILed, status drifted manually, or the experiment was abandoned.
2. **Body format.** Detect which shape the body uses — drives Step 2's conversion pipeline:
   - **legacy Sagan-card HTML (grandfathered) (current):** has an inline `<style>` with `.cr-<N>` namespace, `<section id="tldr">`, `<figure id="figure">`, `<details id="design">`, `<details id="repro">`. Refine in place.
   - **Legacy markdown (EPS-v4 or earlier):** uses `## TL;DR` / `## Summary` / `## Details` markdown H2s, OR the older `## Human TL;DR` / `## AI TL;DR` / `## AI Summary` triad. Auto-convert to legacy Sagan-card HTML (grandfathered) in Step 2.
3. **Title shape.** Multi-claim titles (em-dash stacks, semicolons joining two claims) violate the one-sentence rule (`clean-result-guidelines.md` § Title). Queue a rewrite for Step 2 if applicable.
4. **Verifier sanity.** Run `verify_task_body.py --issue <N>` and note FAILs. (Expected to FAIL on a markdown-legacy body — the conversion will fix it.)

---

## Step 2 — Auto-convert + auto-draft + auto-apply

End-to-end without chat output between sub-steps. Single end-of-step status line after the body is live.

### 2a. Auto-rewrite the title if needed

If Step 1 flagged a multi-claim title: compress to one sentence stating the actual finding (`clean-result-guidelines.md` § Title). Preserve the `(HIGH | MODERATE | LOW confidence)` suffix. Flip negation to affirmative ("X fails to do Y" beats "Y was wrong"). If the title still references a metric the body has dropped mid-iteration, update it.

### 2b. Auto-convert legacy markdown → legacy Sagan-card HTML (grandfathered)

If Step 1 detected legacy markdown:

1. Extract the load-bearing pieces from the markdown body:
   - The single headline claim (from the title or the v4 Summary's first sub-bullet).
   - The Motivation prose (from `## Summary` Motivation bullet or `### Background`).
   - The setup narrative (from `### Methodology` or `### Setup`).
   - The hero figure URL (commit-pinned, from a `### Result N` block).
   - The figure caption (from `**Figure N.**` prose).
   - Sample completions (from `### Result N` fenced code blocks).
   - Confidence label + binding-constraint rationale (from the closing Confidence bullet).
   - Repro pointers (from `### Setup details` / `## Setup & hyper-parameters` block).
2. Assemble the legacy Sagan-card HTML (grandfathered) body following `clean-result-guidelines.md`:
   - Scoped `<style>` block with `.cr-<N>` namespace.
   - `<section id="tldr">` with exactly four `<li>` bullets — **Motivation / What I ran / Results / Next steps**, voice rewritten to "I" not "we", Results bullet anchor-linking `#figure`.
   - `<figure id="figure">` with the hero figure (`<img>` with commit-pinned URL; inline SVG with hover tooltips is preferred when the source data is easy to recompute, but `<img>` is acceptable mechanically).
   - `<details id="design">` carrying the consolidated narrative — definitions, training, eval, sample completions inline with cherry-picked labels AND qualitative-data links above each `<pre>`, statistical-test rationale, the `Confidence: LOW|MODERATE|HIGH — <one sentence>` line, parameters table at the bottom.
   - `<details id="repro">` at the very bottom — Artifacts / Compute / Code groups, permanent URLs only, sentinel-free.
3. If the markdown body had multiple `### Result N` sections, fold them all into the single `#design` narrative as paragraphs / sub-sections of the one design block. Pick ONE result as the primary figure; reference the others' visuals as `<img>` blocks INSIDE `#design` (the figure-id="figure" rule applies only to the hero).

If Step 1 detected an already-correct Sagan-card body, skip 2b and refine in place during 2c/2d.

### 2c. Auto-refine the TL;DR

Draft a 4-bullet user-voice TL;DR matching `clean-result-guidelines.md` § "TL;DR (four bullets)" and the #311 worked example. Bullet labels: **Motivation / What I ran / Results / Next steps**. Voice: "I" not "we". Results bullet must carry an `<a href="#figure">figure below</a>` anchor link AND headline number + N. Next steps may nest a `<ul>` for concrete follow-ups.

If the qualitative-data verifier check is WARN-shaped (raw completions weren't uploaded for this run), one Next-steps sub-bullet MUST be "re-run with raw-completion upload to enable raw-text auditing of the cherry-picked samples".

### 2d. Voice + framing pass

- Replace `we` with `I` in narrative prose throughout (TL;DR + `#design`). Keep `we` if it appears in a quoted exemplar / external citation.
- Strip fluff transitions: *"One more wrinkle:"*, *"the buried lede was"*, *"funnily enough"*, *"the real surprise was"*, *"the kicker is"*, *"what we found was"*. Replace with direct declarative phrasing.
- Strip effect-size / named-test / power-analysis / `value ± err` language from prose (the p-values + N rule). Push the ± to the figcaption, the test name into a "Why this test" paragraph that defines it.
- Drop any `## Background`, `## Methodology`, `## Setup`, `## Findings`, `## Reproducibility`, `## Caveats` H2s — all of that lives inside `#design` / `#repro` as one narrative.

### 2e. Run quality gates

1. **`verify_task_body.py`** — must PASS with no FAILs. WARNs may ship when explicitly acknowledged (the qualitative-data-link WARN paired with a Next-steps re-upload bullet).
   ```bash
   uv run python scripts/verify_task_body.py .claude/cache/experiment-<N>-clean-result.html \
       --title "<the title from Step 2a>"
   ```
2. **`clean-result-critic` agent** (optional, spawn only if Step 1 had non-mechanical diagnostic flags). PASS → continue. REVISE → apply the agent's "Specific revision requests" verbatim, then continue. The critic is a quality gate, not a user-facing iteration loop — apply autonomously.

### 2f. Apply title + body

1. **Title change first** if Step 2a rewrote it:
   ```bash
   uv run python scripts/task.py set-title <N> "<new title>"
   ```
   Title before body — a partial body failure can't leave a body referencing a stale title.
2. **Body apply**:
   ```bash
   uv run python scripts/task.py set-body <N> --file .claude/cache/experiment-<N>-clean-result.html
   ```
3. Re-run `verify_task_body.py --issue <N>` against the LIVE experiment.

### 2g. Single-line status to chat

```
Body live in the task workflow: https://sagan.superkaiba.com/e/experiment/<uuid>
What I did: <one-line, e.g. "converted legacy markdown → legacy Sagan-card HTML (grandfathered), drafted 4-bullet user-voice TL;DR, fixed 7 we→I, dropped 3 separate-H2 sections into #design narrative, added qualitative-data link above each <pre>">
Verifier: PASS (WARNs: <count + reason>)
Want any tweaks before promoting?
```

Then move to Step 3.

---

## Step 3 — Iterate against the live body

The user reads the posted body on the Sagan dashboard and asks for tweaks; you apply them in place. Common shapes:

- **"Bullet 2 of the TL;DR should say X, not Y."** Apply, re-run verifier, echo 1-line confirmation.
- **"Add another sub-section to #design for the new analysis I just ran."** Read the analysis JSON, generate the figure via `paper-plots`, commit, update body. Loop on framing if asked; otherwise apply autonomously.
- **"Re-run the figure without the police_officer outlier."** `paper-plots`, commit, point body at new commit SHA, re-verify.
- **"Promote as not-useful instead."** Skip to Step 4 with the not-useful command.
- **"Actually this should be merged with #M."** Back to Step 0.5 with the new merge target.

Mechanics per iteration:

1. Re-fetch latest body (`uv run python scripts/task.py view <N>`) — the user may have edited via the dashboard.
2. Make the change. Echo back ONLY the diff in chat (title + the changed paragraph / bullet / section), never the whole body.
3. Re-run `verify_task_body.py --issue <N>` if the change touched structure (TL;DR bullets, figure, design / repro blocks). Skip for one-word fixes inside narrative prose.
4. Wait for next user instruction or "ready to promote."

**Capture iteration patterns.** If the user's correction generalizes — would catch the same class of issue in the next clean-result — propose:
- (a) An append to `.claude/skills/clean-results/iterations.md` (one H3 under the appropriate `## YYYY-MM-DD — issue #N (topic)` H2 with `**Before / After / Rule / Folded into**`).
- (b) Surgical edits to the relevant canonical file (typically `~/sagan/docs/clean-result-guidelines.md` or `scripts/verify_task_body.py`).

The user approves each before you write. **Always log; sometimes generalize.**

---

## Step 4 — Hand off

When the user says "ready to promote" / "ship it" / "looks good, promote useful":

```
Ready to promote. Run one of:

  uv run python scripts/task.py promote <N> useful
  uv run python scripts/task.py promote <N> not-useful

This flips `runs.classification` from `pending` to `useful` / `not-useful`
and advances the experiment past the awaiting_promotion gate.
Awaiting-promotion is a user-only gate (CLAUDE.md, gate 7) —
I will not run it.
```

Then EXIT. Don't ask "want me to do the next one?" — the user should `/clear` and re-invoke the skill on the next experiment.

---

## Decision tree: useful vs not-useful

| Signal | Suggests `useful` | Suggests `not-useful` |
|---|---|---|
| Confidence | HIGH or MODERATE | LOW |
| Survives in the paper / a mentor talk | yes | no |
| Findings update someone's beliefs | yes | "I couldn't tell" |
| Replicates / strengthens a parent experiment | yes | yes (still useful as evidence) |
| Negative result with a sharp mechanism | useful | n/a |
| Negative result + binding-constraint excuse | n/a | not-useful |

`not-useful` is not "the experiment failed" — it's "the result isn't load-bearing for the project's narrative." Failed experiments with sharp mechanisms are usually `useful`. Successful experiments that turn out to duplicate a prior finding without adding evidence are usually `not-useful`.

---

## Reference

All body-shape rules, exemplars, anti-patterns, verifier expectations, and rationale live in:

- **`~/sagan/docs/clean-result-guidelines.md`** — canonical Sagan-card spec.
- **`scripts/verify_task_body.py`** — mechanical verifier; 11 checks.
- **`.claude/skills/clean-results/iterations.md`** — append-only log of past corrections. Grep when checking whether a phrasing has been litigated before.
- **Worked example: experiment #311** at <https://sagan.superkaiba.com/e/experiment/1d61738d-df62-44af-9c79-fa41fe85f598>.

This skill stays narrow on workflow: identify target → consolidation scan → diagnostic → auto-convert (markdown→HTML) + auto-draft + apply → iterate → hand off.
