---
name: Code change (infra / refactor / bug fix / feature)
about: Propose a code-change task. Will flow through /issue <N> workflow.
title: "[Code] <what changes and why>"
labels: ["type:infra", "status:proposed"]
---

<!-- Fill in all sections. -->

## Motivation (required)

What problem does this solve? Is there a failing test, a crash, a user
complaint, or a tech-debt concern?

## Scope (required)

- **Files in scope:** list explicitly
- **Files OUT of scope:** list explicitly (prevents scope creep)
- **Additive vs modifying:** new module / modify existing / delete

## Acceptance criteria (required)

What does "done" look like? Enumerate testable conditions:

- [ ] ...
- [ ] ...

## Tests

- Which existing tests must continue to pass? (list specific `pytest` invocations)
- Does this need NEW tests? If so, what behavior do they check?

## Compatibility

- Any existing callers that will break? (list + migration plan)
- Deprecation paths preserved?

## Dependencies

- New packages needed? (`uv add X`)
- Version bumps needed? (lockfile-affecting)

## Performance impact

- Expected: none / minor / major
- If major, what's the before / after benchmark?

## Risk + rollback

- Blast radius if wrong?
- Can this be reverted in a single commit?

## Aim

Which aim does this support? (typically `aim:infra` or `aim:cross-cutting`)

---

**Next step:** run `/issue <this issue number>`. The clarifier will read this,
flag ambiguities, then adversarial-planner → approval →
implementer subagent → code-reviewer → PR ready.
