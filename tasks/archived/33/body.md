---
title: '[Code] Research-chain tracking: issue linking + claims registry'
kind: infra
tags: []
created_at: '2026-04-17T00:54:11.000Z'
has_clean_result: false
sagan_id: ca73312e-1aa8-45fd-b0ff-91025fafc616
sagan_number: 33
priority: low
legacy_why_unset: true
---
## Motivation

Mentor wants a way to trace **data → hyperparams → results → claims** and to see how experiments chain together. Currently:

- No programmatic link between a claim in `RESULTS.md` and its supporting issues/runs/figures.
- Issues reference each other only via ad-hoc `#N` comments; no typed "follows from" / "followed by" relationships so you can't walk the research chain programmatically.
- No single registry of claims with their status + evidence.

This issue adds the minimum infrastructure to close those gaps without building a custom dashboard. Extends the existing `/issue <N>` skill and the marker-comment system.

## Scope

**Files in scope:**

- `docs/claims.yaml` (new) — canonical claim registry
- `docs/claims.md` (new) — auto-rendered; committed by CI
- `scripts/render_claims.py` (new)
- `.github/workflows/render_claims.yml` (new) — CI trigger
- `.claude/skills/issue/SKILL.md` — extend Step 1 (clarifier posts follows markers) and Step 8 (ask about claim attribution)
- `.claude/skills/issue/clarifier.md` — add "what prior issue motivated this?" prompt
- `.claude/skills/issue/markers.md` — add `epm:follows`, `epm:followed-by` kinds
- `.claude/agent-memory/research-pm/reference_github_issues.md` — document `epic:*` and `claim:*` label conventions

**Files explicitly OUT of scope:**

- Eval Q&A logging (`wandb.Table`) — separate issue
- WandB Reports auto-generation via Reports SDK — separate issue
- GitHub Pages eval browser — explicitly deferred per discussion (WandB Reports preferred)
- One-time backfill of the 30 existing issues' claims — sub-issue if/when wanted

## Acceptance criteria

- [ ] `docs/claims.yaml` exists with schema documented in-file: `id`, `description`, `aim`, `status` (`preliminary|moderate|strong|falsified`), `evidence: {issues, wandb_report, figures, results_md_section}`, `kill_criteria`, `supersedes`, `updated`
- [ ] `scripts/render_claims.py` reads `claims.yaml`, joins against live issue states via `gh issue view`, and emits `docs/claims.md` as a sortable markdown table with deep links
- [ ] GitHub Actions workflow re-renders `docs/claims.md` on push to `docs/claims.yaml` OR when any issue closes carrying a `claim:*` label
- [ ] `/issue` skill clarifier asks "what prior issue motivated this?" and posts `<!-- epm:follows N -->` on the child + `<!-- epm:followed-by N -->` on the parent. Bidirectional graph parseable via `gh` comment scan.
- [ ] `/issue` Step 8 asks "contributes to which claim?" and appends the issue to `claims.yaml` evidence list + adds `claim:<id>` label to the issue. Also supports creating a new claim in the yaml with a fresh ID.
- [ ] `epic:*` label convention documented (loose grouping across an investigation)
- [ ] `claim:*` label taxonomy documented; `claim:<id>` labels auto-created via `gh api` when a new claim is added to `claims.yaml`

## Tests

No unit tests needed for this infra. Manual acceptance:

- [ ] Add `claim:C-test-1` to `claims.yaml` → `render_claims.py` emits a valid row in `docs/claims.md`
- [ ] File a new test issue referencing `#29` as its parent via the clarifier → verify both markers posted bidirectionally
- [ ] Close an issue carrying a `claim:C-test-1` label → CI workflow re-renders `docs/claims.md` with the new evidence
- [ ] `ruff check . && ruff format .` passes

## Compatibility

- Fully additive. Existing 30 issues unaffected unless manually tagged `claim:*`.
- `RESULTS.md` continues as human-readable prose. `claims.yaml` is the machine-readable spine. Headlines in `RESULTS.md` reference `C-*` IDs.
- `gh` CLI 2.4.0 compatible (labels created via `gh api`, no `gh label` subcommand used).

## Dependencies

- **PyYAML** — likely already in `uv.lock`. If not, `uv add pyyaml`.
- **`gh` CLI 2.4.0** — already installed.
- No other new PyPI packages expected.

## Performance impact

None. `render_claims.py` runs in CI, off the hot path. Expected render time <5s for 100 claims.

## Risk + rollback

- **Blast radius:** very low. All additive. No existing code is modified except the `/issue` skill (markdown files, behavior-extension).
- **Rollback:** `git revert`. No schema migrations, no model artifacts touched.
- **If `render_claims.py` breaks:** `docs/claims.md` stale; no data loss.

## Aim

`aim:cross-cutting` — tracking infrastructure that spans all research aims.

## Design sketch (informal — the adversarial-planner will produce the canonical plan)

**`docs/claims.yaml` schema:**

```yaml
claims:
  - id: C-aim3-leakage-v1
    description: "Trait leakage across ~X% of adjacent tokens under SFT"
    aim: 3-propagation
    status: moderate
    evidence:
      issues: [27, 28, 29, 30]
      wandb_report: https://wandb.ai/.../reports/...
      figures: [figures/aim3/leakage_comprehensive.png]
      results_md_section: "Aim 3 — Propagation"
    kill_criteria: "effect size < 0.1 across 3+ seeds"
    supersedes: []
    updated: 2026-04-16
```

**Marker comment example (`epm:follows`):**

```
<!-- epm:follows v1 -->
**Follows from:** #27 (marker leakage v3 showed the effect was confounded by token length)
**Motivating result:** link to the epm:results comment in #27
<!-- /epm:follows -->
```

**Label additions (created via `gh api` when referenced):**

- `epic:<slug>` — free-form grouping label (e.g., `epic:trait-leakage`)
- `claim:C-<aim>-<slug>-v<n>` — one per claim in the registry

## Open questions for the clarifier / planner

- Should `claims.yaml` support multi-aim claims (cross-cutting)? — proposed: yes, `aim` field can be a list.
- Should the render include WandB Report thumbnails, or just links? — proposed: links only for now.
- Should `/issue` Step 8 force a claim attribution, or allow `none`? — proposed: allow `none` but warn if the issue is marked `status:under-review` PASS and has no claim.

---

**Next step:** run `/issue <this issue number>` through the skill for gate-keeper → adversarial-planner → approval → implementer.
