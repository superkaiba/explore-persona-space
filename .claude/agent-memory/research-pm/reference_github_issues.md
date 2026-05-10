---
name: GitHub Issues as Queue
description: As of 2026-04-16, GitHub Issues is the source of truth for the experiment queue; EXPERIMENT_QUEUE.md is historical.
type: reference
---

**Repo:** `superkaiba/explore-persona-space`
**Issues:** https://github.com/superkaiba/explore-persona-space/issues
**Project board:** https://github.com/users/superkaiba/projects/1 (Experiment Queue)
**Project node ID:** `PVT_kwHOAt7nP84BU2Da`

**Label taxonomy:**
- `aim:{1-geometry, 2-localization, 3-propagation, 4-axis-origins, 5-defense, 6-truthification, infra, cross-cutting}`
- `prio:{critical, high, medium, low}`
- `status:{proposed, gate-pending, planning, plan-pending, approved, running, reviewing, under-review, blocked}` (9 states as of 2026-04-17; `planning` + `reviewing` added to distinguish "agent is working" from "awaiting user action")
- `type:{experiment, infra, analysis, survey}`
- `compute:{none, small (<5h), medium (5-20h), large (>20h)}`

**Status semantics (active vs awaiting-user):**
| Label | Who's working | User action |
|-------|---------------|-------------|
| `proposed` | nobody (or user answering clarifier) | sometimes |
| `gate-pending` | gate-keeper agent | no |
| `planning` | adversarial-planner | no |
| `plan-pending` | nobody | **yes — approve** |
| `approved` | skill (dispatching) | no |
| `running` | specialist (experimenter/implementer) | no |
| `reviewing` | analyzer + reviewer | no |
| `under-review` | nobody | **yes — /signoff** |
| `blocked` | nobody | **yes — triage** |

**Lifecycle:** proposed → gate-pending → planning → plan-pending → approved → running → reviewing → under-review → closed.

**EXPERIMENT_QUEUE.md has a banner at the top** directing to GitHub Issues. The sections below that banner are historical; do not add new entries to the markdown file.

**When adding new proposals:** use `gh issue create --title ... --body ... --label ...`. Every new issue should get at minimum: one `aim:*`, one `prio:*`, one `type:*`, one `compute:*`, and `status:proposed`.

**When advancing lifecycle:** swap the `status:*` label. Example: `gh issue edit N --remove-label status:proposed --add-label status:gate-pending`.

**gh CLI version on this machine is 2.4.0** — too old for `gh label` and `gh project` subcommands. Use `gh api` and `gh api graphql` instead. (Issue creation works fine with `gh issue create`.)

**Historical:** the 14 Proposed + 12 Planned + 1 Running + 3 Under Review entries from EXPERIMENT_QUEUE.md were migrated as issues #1-30 on 2026-04-16.

**Label additions 2026-04-17:** added `status:planning` and `status:reviewing` (purple `5319e7`) to disambiguate "agent is working right now" from "awaiting user action." Previously `plan-pending` ambiguously meant both "planner running" and "plan posted, awaiting approve." See state-machine section above.

**`/issue <N>` skill (added 2026-04-17):** `.claude/skills/issue/` drives the full per-issue lifecycle (clarify → gate-keeper → adversarial-planner → approval → worktree + dispatch → preflight → run → analyzer → reviewer → close). Idempotent and resumable via marker comments. For any task ≥30 min, prefer `/issue <N>` over manual DISPATCH. See `.claude/skills/issue/SKILL.md` + `markers.md` + `clarifier.md`.

**Marker protocol:** all structured state lives in HTML-commented markers on issue comments: `<!-- epm:<kind> v<n> -->...<!-- /epm:<kind> -->`. Specialists in issue-bound mode (brief contains `issue: <N>`) post their progress/results/failures as these markers; latest `v` wins per kind. See `.claude/skills/issue/markers.md` for the full taxonomy.

**Issue templates:** `.github/ISSUE_TEMPLATE/experiment.md` and `code-change.md` (both prefill `status:proposed` + `type:*`). Blank issues disabled via `config.yml`.
