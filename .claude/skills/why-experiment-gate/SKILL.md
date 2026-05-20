---
name: why-experiment-gate
description: >
  Four-question adversarial interrogation that forces the user to
  articulate Decision / Branches / Cut / Application before any
  experiment, survey, or infra task can advance. Refuses non-answers,
  fires at most one substance challenge per question, transcribes the
  user's words verbatim into `## Why this experiment`, and posts the
  `epm:gate-filled` marker. Invoked by the PM session (Mode 5 pre-spawn)
  and by `/issue <N>` Step 0 as a safety net.
user_invocable: true
---

# Why-this-experiment gate

This skill is **interrogation, not drafting**. Your job is to ask the
four questions one at a time, refuse non-answers, fire **at most one**
substance challenge per question when the user's answer has a
research-taste problem, transcribe the user's words **verbatim** into
the body, and exit. You never propose answers, summarize from prior
context, or synthesize from the parent task. If the user delegates
("you decide", "use the ideation doc"), you refuse and re-ask.

The user always has the last word via an unconditional override
("I'm right", "defer", "ship it") — when invoked, transcribe whatever
they said and move on, logging the override in the marker.

The mechanical floor (does the section exist with 4 non-empty labeled
lines?) is enforced by `scripts/verify_task_body.py` check #12 and by
`scripts/task.py new`. This skill is the **substantive** layer that
lives on top — the part a regex can't do.

---

## When to use

- The PM session is about to dispatch `#N` via `spawn_session.py
  spawn-issue` and `tasks/<status>/<N>/body.md` either lacks `## Why
  this experiment` or has any of the four labeled lines empty / stubby.
- `/issue <N>` Step 0 detects the same condition (PM bypassed; the
  user opened a per-issue session directly).
- The user explicitly invokes `/why-experiment-gate <N>` to backfill
  the section on a task they already approved.

## When NOT to use

- For `analysis` tasks. They are exempt from the floor (read-only
  workflows over existing artifacts).
- For tasks whose `body.md` frontmatter carries `legacy_why_unset:
  true`. The migration script applied that sentinel to bodies authored
  before the gate landed; the gate's substantive interrogation is not
  retroactive.
- For drafting an experiment's hypothesis, design, or success
  criteria. Those belong in `/adversarial-planner` Step 2, not here.
- For refining a clean-result body after the experiment finished.
  That's `/promote-clean-result`.

---

## The four questions (asked one at a time, in this order)

1. **Decision this changes.** What concrete choice in your queue or
   proposal hinges on this outcome? Name the queue position or
   proposal section the result will move.
2. **Expected outcome + branches.** What do you expect to see, and
   what alternative outcome would route you to a different next
   experiment? Both branches must lead to genuinely different next
   moves.
3. **What gets cut.** Which experiment in your queue does NOT run this
   week because compute / attention is going here? Name the specific
   task number or backlog item.
4. **Application.** Pick one of `detect | predict | defend | audit |
   infra`. State in one sentence which deliverable this advances.
   (If the user picks two, refuse — split the task instead.)

Ask **only one question per turn**. Wait for the user's answer before
moving on. Do not preview the other questions.

---

## Four-layer gate dynamic (run on EVERY question)

For each question, in order:

1. **Ask** the question. One sentence. No preamble. No reframing.
2. **Refuse non-answers** — if the response matches any pattern in the
   reject list below, reply with "That's a non-answer. <one-sentence
   reason>. Try again — name a concrete <decision|branch|cut|application>."
   Do NOT proceed.
3. **Challenge the substance** — fire at most ONE research-taste
   critique per question, only when the lens for that question (below)
   fires on a real issue. If you don't see a substantive issue, SKIP
   the challenge entirely. No gratuitous challenges.
4. **Defer + transcribe** — the user has three ways to terminate the
   challenge loop (Defense / Reframe / Override; see "Termination
   rules" below). Transcribe their final answer verbatim.

Run all four layers on Q1 before moving to Q2. Do not batch.

---

## Refuse list (non-answer patterns)

Reject any answer that consists ONLY of one of these phrases or that
runs ≤15 words without naming a specific decision/branch/cut/
application:

- "just curious" / "want to see what happens" / "to explore" / "to
  check" / "to understand X better"
- "use the ideation doc" / "you decide" / "look at the parent" /
  "summarize from context" / "you know what I meant"
- "because it would be interesting" / "to add to the literature"

The 15-word floor is a hint, not a hard rule — a tight 12-word answer
that names a specific decision passes ("Whether to keep five-seed
reporting on the persona panel."). A 50-word ramble that names nothing
fails.

When in doubt about a borderline answer, fire the substance challenge
(layer 3) — the user will either defend, reframe, or override, and
all three are valid terminations.

---

## Challenge lenses (one per question, only when applicable)

| Question | Lens fires when… | Challenge shape |
|---|---|---|
| Decision | Decision is already determined by an existing result, OR reachable on cheaper evidence | "X already constrains this — does this really update beyond what you know?" |
| Branches | Both branches lead to the same next move (non-decisive) | "Both branches seem to lead to the same next experiment. What actually changes between them?" |
| Cut | Named cut isn't credible (already deprioritized, or the user will run both anyway) | "You're already deprioritizing that — is this a cut or a free addition?" |
| Application | Anchor reads as post-hoc, OR the deliverable serves a different application than the one named | "This looks more like `<X>` than `<Y>` — does the deliverable actually feed `<Y>`'s metric?" |

**If you don't see a substantive issue under the lens, skip the
challenge.** A gate that fires gratuitously is friction without
information. Fire at most ONCE per question.

---

## Termination rules

Three accepted user replies to a challenge — pick whichever matches
the user's response:

- **Defense.** User restates the original answer with a brief
  defense. → Transcribe both the original answer AND the defense (one
  short line). Move on.
- **Reframe.** User changes the answer. → Transcribe the NEW answer
  (drop the original). Do not re-challenge.
- **Override.** "I'm right" / "defer" / "ship it" / "move on" /
  equivalent. → Transcribe the user's ORIGINAL answer, set
  `user_overrode_challenge: true` for that question in the marker.
  Move on.

If the user does NOT challenge-respond — i.e., they ignore the
challenge and answer a different question, ramble, or post a wall of
text that doesn't address the lens — treat that as a Reframe of the
original answer: take the new content verbatim and move on. Do not
re-challenge.

---

## Transcription rules (CRITICAL)

- **Verbatim, minus boilerplate.** Strip greeting words ("ok so",
  "right, "), trailing parens ("(whatever)"), and obvious typos. Do
  NOT rephrase, summarize, smooth, or "clean up" the answer.
- **One sentence per labeled line.** If the user wrote three
  paragraphs, take the most substantive sentence verbatim and offer
  the user a single line to confirm: "I'll write this as: `<one
  sentence>`. OK?" — accept their tweak, then move on.
- **No drafting from context.** If the user says "you draft it" or
  "use what we discussed earlier", refuse: "I can't draft this for
  you — the gate only works if YOU articulate the decision. What
  concrete <decision|branch|cut|application> hinges on this?"
- **No filler.** Bare answers are fine. Do not pad to look
  professional.

---

## Apply mechanics

After all four questions are answered (with overrides logged where
applicable):

1. **Read the existing body**:
   ```bash
   uv run python scripts/task.py view <N> --json
   ```
   Note the absolute path of `body.md` from the output's `path` field
   (`tasks/<status>/<N>/body.md`) — you'll edit the frontmatter
   directly in step 4.

2. **Insert `## Why this experiment`** into the body. Placement
   convention: between the top H1 (or `## Goal`) and the first
   substantive section (`## Background`, `## Setup`, etc.). If the
   body has no clear top section, insert immediately after the H1.

3. **Apply the body change via `task.py set-body`**:
   ```bash
   uv run python scripts/task.py set-body <N> --file /tmp/why-issue-<N>-body.md
   ```
   `set-body` replaces the BODY content only and preserves the
   existing frontmatter verbatim. The file you pass must be the body
   text alone — NOT a full markdown document with a `---` frontmatter
   header. If the file you pass begins with `---\n...\n---\n`, those
   lines will land as literal body text (not parsed as YAML), and the
   `application:` field will not change.

4. **Update the frontmatter `application:` field directly** — there is
   no `task.py set-frontmatter` subcommand, and `set-body` deliberately
   does not touch frontmatter (see its `--help`). Use your file-edit
   tool (`Read` then `Edit`) on `tasks/<status>/<N>/body.md` to add
   or change the `application:` line in the YAML block at the top of
   the file. Example before / after:
   ```yaml
   ---
   title: ...
   kind: experiment
   application: predict       # ← add or change this line
   ---
   ```
   Then commit the change so the dashboard picks it up:
   ```bash
   git -C $(uv run python scripts/task.py find <N> | head -1) add body.md
   git commit -m "task #<N>: set frontmatter application=<enum>"
   ```
   Verify the update with:
   ```bash
   uv run python scripts/task.py view <N> --json | jq -r '.frontmatter.application'
   ```

5. **Post the marker**:
   ```bash
   uv run python scripts/task.py post-marker <N> epm:gate-filled \
     --note '{"gate":"why-experiment","filled_by":"<agent_session_id>","challenges_fired":["decision"],"user_overrides":[]}'
   ```
   Fields:
   - `gate`: always `"why-experiment"`.
   - `filled_by`: the agent session id (or `"main"` if unknown).
   - `challenges_fired`: a JSON array listing which questions fired
     the challenge layer — values from
     `["decision","branches","cut","application"]`.
   - `user_overrides`: same value set, listing only questions where
     the user overrode the challenge with "I'm right" / "ship it" /
     "defer".

6. **Run the mechanical verifier** to confirm:
   ```bash
   uv run python scripts/verify_task_body.py --issue <N>
   ```
   Check #12 must PASS. If it FAILs, you transcribed something
   sub-40-chars, got the labels wrong, or the frontmatter
   `application:` field disagrees with the body's `Application` line
   — re-read the body, fix the transcription, re-apply.

---

## Bypass requests

The user may attempt any of these. Refuse, but politely, ONCE; on the
second attempt, comply and log the override:

| Bypass attempt | Response |
|---|---|
| "Just skip the gate, I'm in a hurry." | "The gate is one question, four times. If you genuinely don't have answers, that's a signal the experiment shouldn't run yet. Want to try the first question?" |
| "Use the parent task's why section." | "The gate only works if YOU articulate the decision for THIS experiment. The parent's decision is the parent's decision." |
| "Draft something plausible from the title." | "I won't draft this — the friction IS the point. What decision in your queue does THIS experiment change?" |
| "I'm right, move on" (after refusing to answer Q1 at all) | Comply. Transcribe whatever they last said (even "I'm right") into the labeled line, mark `user_overrode_challenge: true` for that question in the marker. The gate logs every override; the user samples the log later. |

---

## Output contract

When the four questions are answered and the body is patched:

```
✔ Gate filled for #<N>.

Application: <enum>
Decision: <one sentence from user, verbatim>
Branches: <one sentence from user, verbatim>
Cut: <one sentence from user, verbatim>

Marker: epm:gate-filled posted (challenges_fired=[...],
user_overrides=[...])
Verifier: check #12 PASS.

Next: <PM dispatches via spawn_session.py | user runs /issue N>.
```

EXIT. Do not start any other work in the same turn.
