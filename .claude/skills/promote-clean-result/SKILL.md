---
name: promote-clean-result
description: >
  Workflow for moving an `awaiting_promotion` task to `completed` with
  `classification='useful'` (or `'not-useful'`). Scans the
  `tasks/awaiting_promotion/` folder for consolidation candidates, refines
  the body if needed, runs `verify_task_body.py`, and hands off the
  `task.py promote` command. Use when the user says "promote #N",
  "clean up Awaiting promotion", or "help me refine the TL;DR for X".
user_invocable: true
---

# Promote a clean-result

Awaiting-promotion tasks are clean-result-critic-PASSed bodies parked at
`tasks/awaiting_promotion/<N>/` with `has_clean_result: true` and
`classification: pending` in frontmatter. Promotion is **user-only** by
design (CLAUDE.md park-and-wait gate) — `/issue` cannot move them. This
skill gets one task ready and hands off the `task.py promote <N>
useful|not-useful` command for the user to run.

**Apply first, iterate after.** Don't pre-propose every TL;DR draft in
chat — refine the body, run the quality gates, push, then iterate against
the live body. The only pre-apply user gate is Step 1 (consolidation
candidates), because merging is destructive.

**Batch-promote prescan.** When the user issues a MULTI-task promote
directive ("promote everything before #K", "move these N to completed"),
do NOT classify the whole batch `useful` blind: first grep each
candidate's title + body head for `BUGGED`, `invalid`, `headline not
obtained`, or an explicitly-failed manipulation check, and present a
one-line check — "these M look not-useful: #a (BUGGED), #b (...) — promote
as useful anyway?" — before running the loop. On 2026-06-09 a 53-task
blanket-useful batch promoted #407, whose body literally begins "# BUGGED
experiment", and the user had to catch and flip it minutes later via a
status round-trip.

**All body-shape rules live in `.claude/skills/clean-results/SPEC.md`.**
NEW bodies use the five-flat-H2 v3 spec — five required H2 sections in
order: `## Takeaways` / `## What I ran` / `## Findings` / `## Data` /
`## Reproducibility` (sentinel `<!-- clean-result-v3 -->` after the H1;
`## Findings` carries one `### <finding>` per result; confidence in the
H1 title tag only; conciseness caps mechanically enforced). The ~30
parked `awaiting_promotion` bodies are grandfathered v2 (sentinel
`<!-- clean-result-v2 -->`: `## Human TL;DR` / `## TL;DR` /
`## Reproducibility`, `## TL;DR` carrying `### Motivation` /
`### What I ran` / `### Findings` → `#### <finding>` per result) or
pre-sentinel legacy — branch on the sentinel and refine each in its own
shape; do NOT migrate a parked v2 body to v3 during a plain promote
pass. Enforced mechanically by `scripts/verify_task_body.py` (which
branches on the sentinel; check catalog in the script docstring).
Workflow + apply mechanics only here.

---

## When to use

- User says "promote #N", "let's clean up Awaiting promotion", "write the
  TL;DR for X", or pastes an EPS dashboard link to an
  `awaiting_promotion` task.
- One task per skill invocation — `/clear` between tasks so each draft
  gets clean attention.

## When NOT to use

- For drafting figures or analysis content. That's the analyzer /
  interpretation-critic / clean-result-critic loop's job.
- For posting net-new clean-results. That's `/issue`'s analyzer step.
- For changing the `useful` / `not-useful` decision after promotion.
  Re-run `task.py promote` directly.

---

## Step 1 — Scan for consolidation candidates (pre-apply gate)

Run:

```bash
uv run python scripts/task.py list-by-status --status awaiting_promotion --json \
  | jq -r '.[] | "#\(.id) \(.title)"'
```

If the user named a single `#N`, also check whether any other
`awaiting_promotion` task has:

- Same parent (frontmatter `parent_id` field), OR
- Title overlap with `#N`'s title (rough cosine ≥ 0.4 by hand), OR
- Time-adjacent (`created_at` within ±2 days)

If consolidation candidates exist, surface them to the user before
applying any changes:

```
Found possible consolidation: #<N> + #<M> share parent #<K>. Merge into
one clean-result? (y/n)
```

If the user agrees, fold `#<M>`'s findings into `#<N>`'s body, then post
on `#<M>`:

```bash
uv run python scripts/task.py post-marker <M> epm:consolidated-into \
    --by promote-clean-result \
    --note "Findings folded into #<N>. See https://eps.superkaiba.com/tasks/<N>."
uv run python scripts/task.py set-status <M> archived \
    --note "consolidated into #<N>"
```

If the user declines or there are no candidates, proceed.

## Step 2 — Inspect the body

```bash
uv run python scripts/task.py find <N>
# Then read tasks/awaiting_promotion/<N>/body.md
```

Detect format:

- **Markdown clean-result (current, new tasks — v3):** opens with
  `# <title> (LOW|MODERATE|HIGH confidence)`, then the sentinel
  `<!-- clean-result-v3 -->`, then `## Takeaways` / `## What I ran` /
  `## Findings` / `## Data` / `## Reproducibility`. `## Findings` carries
  one `### <finding>` H3 per result. Refine in place.
- **Markdown clean-result (grandfathered v2 — parked backlog):** sentinel
  `<!-- clean-result-v2 -->` after the H1; `## Human TL;DR` / `## TL;DR` /
  `## Reproducibility`, with `## TL;DR` carrying `### Motivation` /
  `### What I ran` / `### Findings` H3s + one `#### <finding>` H4 per
  result. Refine in its own v2 shape — do NOT migrate to v3 on a plain
  promote pass.
- **Legacy Sagan-card HTML (grandfathered, imported from Sagan):** has
  `<!-- legacy-sagan-card -->` sentinel + inline `<style>` block.
  Optionally convert to markdown if the user asks (see Step 4b);
  otherwise leave as-is for historical viewing.

## Step 3 — Refine the body

Read the spec at `.claude/skills/clean-results/SPEC.md` (canonical) and
the summary under "Experiment Report Structure" in `CLAUDE.md`. Common
refinements at this stage:

- Title says exactly what the result is (not the experiment name) and
  ends with `(LOW|MODERATE|HIGH confidence)`.
- **v3 body** carries the `<!-- clean-result-v3 -->` sentinel right after
  the H1; the five H2s are `## Takeaways` / `## What I ran` /
  `## Findings` / `## Data` / `## Reproducibility`. `## Takeaways` is 3–6
  numbers-first bullets (the rolling cross-round synthesis); `## Findings`
  carries one `### <finding>` per result; `## Data` carries
  `### Trained on` / `### Evaluated with` / `### Generated` with a
  subset-disclosed example + pinned link per subsection.
  (**Grandfathered v2 body:** `<!-- clean-result-v2 -->` sentinel,
  `## TL;DR` shaped as `### Motivation` / `### What I ran` /
  `### Findings` → `#### <finding>` per result.)
- Issue numbers appear ONLY in `## What I ran` `**Why:**` + the
  `## Reproducibility` `**Context:**` row (v2: `### Motivation` +
  `## Reproducibility`); the rest of the body is standalone — no
  cross-issue framing.
- Reproducibility URLs are all permanent-pinned (`/tree/<sha>`,
  `/runs/<id>`, `/blob/<sha>`), no `TBD` / `{{` / `default` /
  `see config`.
- Confidence lives in the H1 title tag ONLY (v2 + v3) — do NOT emit a
  body `Confidence: …` sentence in `## Reproducibility`.

Apply edits to the body via:

```bash
uv run python scripts/task.py set-body <N> --file /tmp/refined-body.md --snapshot
uv run python scripts/task.py set-title <N> "<title from H1, minus '# '>"
```

The `--snapshot` flag saves the prior body to
`tasks/awaiting_promotion/<N>/original-body.md`. The body is replaced
atomically + git committed.

## Step 4 — Verify

```bash
uv run python scripts/verify_task_body.py --issue <N>
```

Every FAIL must be fixed before handoff. Iterate Step 3 → Step 4 until
the verifier PASSes. Also run the anti-pattern audit:

```bash
uv run python scripts/audit_clean_results_body_discipline.py \
    "$(uv run python scripts/task.py find <N>)/body.md"
```

### Step 4b — (Optional) Convert legacy HTML body to markdown

If the body has `<!-- legacy-sagan-card -->` and the user asks for
markdown conversion:

1. Read the HTML body and extract: title, TL;DR bullets, figure URL +
   caption, Details narrative, Reproducibility section, confidence
   level.
2. Write the markdown body to `/tmp/converted-<N>.md` following the
   markdown spec.
3. Run `verify_task_body.py --file /tmp/converted-<N>.md` until PASS.
4. Apply via `task.py set-body --file --snapshot`.

The legacy HTML body is preserved in `original-body.md` after the
snapshot — re-running `task.py set-body --file original-body.md
--snapshot` restores it.

## Step 5 — Iterate with the user

Push the dashboard link:

```
Body refined. Review at https://eps.superkaiba.com/tasks/<N>
```

The user reads the live body on the EPS dashboard and asks for tweaks;
you apply them in place via repeated `task.py set-body` calls. Each
edit is one git commit on `task-workflow`.

## Step 6 — Execute (explicit intent) or hand off

**If the user's request already carries explicit promote intent** —
"promote N", "promote it", "promote N useful/not-useful" — and Steps
3-5 PASS, run the command directly on their behalf:

```
uv run python scripts/task.py promote <N> useful   # or not-useful, per their words
```

The "user-only" rule means no AUTOMATION may flip
`runs.classification` on its own; a human's explicit "promote N" in
chat IS the user gate, and re-asking "ready to promote?" after they
already said so is the anti-pattern (2026-06-10: Thomas said
"Promote 488", got a summary instead of execution, and had to repeat
"PROMOTE IT"). Ask ONLY when the classification is ambiguous (no
useful/not-useful signal and the body suggests not-useful) or a gate
FAILed.

**Otherwise** (the user asked for a review/refine pass, not a
promotion), hand off:

```
Ready to promote. Run:

    uv run python scripts/task.py promote <N> useful

or

    uv run python scripts/task.py promote <N> not-useful
```

That command moves the task from `tasks/awaiting_promotion/<N>/` to
`tasks/completed/<N>/`, records `classification: useful` (or
`not-useful`) + `promoted_at: <ts>` in frontmatter, and posts
`epm:promoted`. The user then re-invokes `/issue <N>` to fire the
follow-up-proposer step.

---

## References

- **`.claude/skills/clean-results/SPEC.md`** — canonical clean-result
  spec (five-flat-H2 v3: `## Takeaways` / `## What I ran` /
  `## Findings` (one `### <finding>` per result) / `## Data` /
  `## Reproducibility`; confidence in H1 title tag only; conciseness
  caps; Data-section spec; voice rules; sample-output discipline; plus
  the grandfathered v2/legacy shapes for the parked backlog).
- **`CLAUDE.md` § "Experiment Report Structure"** — brief summary
  pointing back at SPEC.md.
- **`scripts/verify_task_body.py`** — mechanical verifier (check
  catalog in the script docstring; branches on the sentinel — v3
  five-flat-H2 structure + Data-shape + conciseness checks for
  `<!-- clean-result-v3 -->` bodies, the grandfathered v2 nested-structure
  check for `<!-- clean-result-v2 -->` bodies; skips legacy
  `<!-- legacy-sagan-card -->` HTML bodies with PASS).
- **`scripts/verify_sagan_card.py`** — legacy verifier retained for
  grandfathered HTML bodies only.
- **`scripts/audit_clean_results_body_discipline.py`** — prose-level
  anti-pattern audit.
- **`.claude/skills/clean-results/iterations.md`** — append-only log
  of corrections + the rules they produced.
