---
title: Workflow improvements
kind: infra
tags: []
created_at: '2026-05-06T01:27:33.000Z'
has_clean_result: false
sagan_id: 54f15332-9ce4-4ec5-a14a-07f25e6fe903
sagan_number: 282
priority: normal
legacy_why_unset: true
---
**type:batch** — workflow improvements collected from #282 + #287 + #288 + #289 (consolidated).

Clarifier resolved 2025-05-06: items 3 and 5 dropped; items 1 and 4 sharpened. See `epm:clarify-answers v1` comment for full transcript.

---

## 1. RunPod API as single source of truth for pod state

(Status: **sharpened** — the original "improve pod tracking" framing collapsed to a concrete fix.)

Today, ephemeral pod state is mirrored in `scripts/pods_ephemeral.json`. The cache drifts from the live RunPod API (ghost pods, alias confusion, MCP referencing dead pods). Make the live API the single source of truth.

**Scope:**
- Retire `scripts/pods_ephemeral.json` as authoritative state. `pod.py` queries the live RunPod GraphQL API (`X-Team-Id` scoped) on every read.
- `pods.conf` (used for SSH/MCP wiring) becomes a derived view, regenerated from the API on demand. It stays the SSH config source.
- Add `pod.py status --issue <N>` that prints live API state for the issue's pod.
- Existing commands (`provision`, `stop`, `resume`, `terminate`, `list-ephemeral`) are reworked to read state from API, not JSON.
- `--refresh` flag becomes a no-op (backwards compat).

**Acceptance criteria:**
- `python scripts/pod.py list-ephemeral` returns the same pods as the RunPod web UI for our team, with no JSON cache touched.
- `python scripts/pod.py status --issue <N>` prints `{pod_id, status, ip, port, gpu_type, gpu_count, ttl_remaining}` from live API.
- Deleting `scripts/pods_ephemeral.json` does not break any command.
- Existing tests pass; new test covers `pod.py list-ephemeral` returning live-API state.

---

## 2. Remove stale AIMS references

Active-tree grep for `\baim:|AIM-?[0-9]` finds **only one** match: the explanatory note at `.claude/skills/issue/SKILL.md:266` documenting that `aim:*` labels were deleted in #251. Recommend leaving the note (it's load-bearing audit context, not a stale reference). Planner can decide between leave / remove / extend-to-archive.

**Acceptance criteria:**
- Final state: zero stale references in active tree (`*.md`, `*.py` outside `archive/` and `.claude/worktrees/`).
- The audit note in `SKILL.md:266` either stays (recommended) or is removed cleanly with a commit message that preserves the #251 reference.

---

## 4. Three-column project board for clean-results

Replace today's two-column layout (`Draft Clean Results` + `Clean Results`) with three columns:

- **Awaiting Promotion** — issues with `clean-results:draft` label (today's "Draft Clean Results"; rename the column).
- **Useful Clean Results** — terminal column for high-quality finalized results.
- **Less Useful Clean Results** — terminal column for promoted-but-meh results that we want to retain for audit but de-emphasize.

**Label scheme:**
- `clean-results:draft` → Awaiting Promotion (existing).
- `clean-results:useful` → Useful Clean Results (new).
- `clean-results:less-useful` → Less Useful Clean Results (new).

**`/clean-results promote` UX:**
- New invocation: `/clean-results promote <M> useful` or `/clean-results promote <M> less-useful` (asks if argument missing).
- Auto-fires `/issue <source-N>` Step 10 when promotion fires (since the iteration loop in `Awaiting Promotion` is already the user gate).

**Touchpoints:**
- Project-board column rename + creation (GitHub Project #1).
- `.claude/skills/issue/SKILL.md` "Project-board status convention" section + Step 9b PASS message + Step 10b/10c flow.
- `.claude/skills/clean-results/SKILL.md` (the `promote` action).
- `scripts/gh_project.py` if it knows column names.
- `scripts/verify_clean_result.py` — does it check labels? Update if so.

**Acceptance criteria:**
- New columns exist on the project board.
- `/clean-results promote <M> useful` (and `less-useful`) moves the issue to the right column AND auto-fires `/issue <source-N>` Step 10.
- Source-issue label transitions from `status:awaiting-promotion` to `status:done-experiment` (or `done-impl`) without a manual re-invoke.
- Existing `clean-results:draft` issues continue to render in `Awaiting Promotion`.

---

## 6. Hypothesis + kill-criterion regex gate

(Originally #288; **Parent:** #275 audit item 2.)

The workflow ASKS for a falsifiable hypothesis + kill criterion in two places but does NOT statically enforce either:

1. `.claude/skills/issue/clarifier.md:130-134` — clarifier asks the questions but advances `status:proposed → status:planning` based on subjective LLM judgement of "All clear" vs "Ambiguities remain". A `type:experiment` issue without a hypothesis can slip through if the clarifier-LLM deems it "minor".
2. `.claude/skills/adversarial-planner/SKILL.md:43` — planner is instructed to include a Hypothesis section "if experiment". Same problem: instruction-not-gate. A planner that omits the section gets caught only if a Critic flags it, which is non-deterministic.

**Proposed fix.** A ~30 LOC static gate at each surface:

- **Clarifier gate.** Before posting `<!-- epm:clarify v1 -->` "All clear", regex the issue body for `**Hypothesis**` (or `### Hypothesis`) AND `**Kill criterion**` (or `### Kill criterion`) sections. If `type:experiment` and either is missing, post the questions unconditionally as ambiguities — do not advance.
- **Planner gate.** Before posting `<!-- epm:plan v1 -->`, regex the drafted plan body for the same headers. If `type:experiment` and either is missing, refuse to post; loop back to the planner with the missing-section feedback.

**Override mechanism.** Recommended: a label `override:hypothesis-skip` on the issue. Both gates check for it before refusing. Labels are first-class and visible on the project board.

**Acceptance criteria:**
- A `type:experiment` issue with no hypothesis section in the body cannot reach `status:planning` without `override:hypothesis-skip`.
- A drafted plan with no Hypothesis section cannot reach `status:plan-pending` without `override:hypothesis-skip`.
- New tests in `tests/test_skill_set_status_calls.py`-shape: static fixture issue bodies hit the gate predictably.

---

## 7. Clarify path-vs-body convention in `/issue` Step 4 and Step 6 briefs

(Originally #289; **Parent:** #275 audit item 3.)

The dispatch briefs in `.claude/skills/issue/SKILL.md` Steps 4 (implementer) and 6 (experimenter) say "The plan" without specifying body-vs-path. In practice the orchestrator passes the path `.claude/plans/issue-<N>.md`; the subagent is expected to `Read` the file before acting.

This is the right default (1400-line plans inlined into a prompt waste context, can hit token limits, and stale across worktree edits). But the convention is not documented, so an adversarially-loaded subagent might guess at plan contents instead of reading the file.

**Proposed fix.** Add a 1-line clarification to both briefs:

> **Plan handoff convention:** the brief includes the PATH to the cached plan (`.claude/plans/issue-<N>.md`), NOT the body. Read the file before acting; do NOT infer plan content from the issue body or comment markers.

That's it — pure doc PR, no behaviour change.

**Acceptance criteria:**
- `.claude/skills/issue/SKILL.md` Step 4 brief includes the path-vs-body clarification.
- Same for Step 6 brief.
- No code change, no behaviour change.

---

## Spec (from clarifier)

**Final scope: 5 items (1, 2, 4, 6, 7). Items 3 and 5 dropped per clarifier.**

| # | Type | Touchpoints | Risk |
|---|---|---|---|
| 1 | Refactor | `scripts/pod.py`, `scripts/runpod_api.py`, `scripts/pod_config.py`, `scripts/pods_ephemeral.json` (deleted), tests | Medium — touches pod lifecycle |
| 2 | Doc | maybe `.claude/skills/issue/SKILL.md:266` (recommended: leave) | Tiny |
| 4 | Refactor + project board | Project board #1, `.claude/skills/issue/SKILL.md`, `.claude/skills/clean-results/SKILL.md`, `scripts/gh_project.py`, `scripts/verify_clean_result.py` | Medium — restructures workflow surface |
| 6 | Implementer | New static gates in `.claude/skills/issue/clarifier.md` + `.claude/skills/adversarial-planner/SKILL.md` (or a helper script invoked by both), tests | Low — additive |
| 7 | Doc | `.claude/skills/issue/SKILL.md` Step 4 + Step 6 briefs | Tiny |

Per `type:batch` convention, each item lands as one commit (`[N/5] <item title>`). Code-reviewer reviews the full diff; per-item commits keep history bisectable.
