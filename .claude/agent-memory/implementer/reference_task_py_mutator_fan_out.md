---
name: task-py-mutator-fan-out
description: a new task.py frontmatter mutator (set-kind family) touches 5 surfaces — library fn + KINDS/STATUSES enum + __all__ + CLI (import/handler/subparser/docstring) + a test; REGISTRY denormalizes kind+title+status+has_clean_result so the mutator MUST call _registry_set
metadata:
  type: reference
---

Adding a new `task.py` frontmatter mutator (modeled on `set-kind`, the
sibling of `set-title`/`set-goal`) has a fixed fan-out. Hit every surface
or the change is half-wired:

1. **`src/explore_persona_space/task_workflow.py`**
   - the library function (flock via `with _locked():`, `_read_body` →
     mutate `fm` → `_write_body` → `_registry_set` → `_save_registry` →
     `_git_commit([path, registry_path()], ...)`). Copy `set_title`
     verbatim and change the field.
   - validate the value against the enum constant and raise `ValueError`
     on miss (the library is the defense-in-depth layer; the CLI
     `choices=` is the first gate).
   - add the function to `__all__`.
   - if the field is enum-valued, add a module-level enum tuple next to
     `STATUSES` (there was NO `KINDS` constant before #672 — `create_task`
     never validated `kind`; the only enforcement was the CLI `choices=`
     literal list).
2. **`scripts/task.py`** — FOUR spots: the `from ...task_workflow import`
   block (import the fn AND any new enum constant), `cmd_<name>` handler,
   the `sub.add_parser(...)` subparser (`choices=list(KINDS)`), and the
   module docstring subcommand list.
3. **`tests/test_task_workflow.py`** — mirror `test_set_title_updates_registry`:
   one test for frontmatter+REGISTRY sync, one for invalid-value `ValueError`,
   one for commit-count (`_git_log_count(repo)` helper). Use the `fake_repo`
   fixture (monkeypatches `repo_root`/`tasks_dir`/`registry_path` to tmp).

**REGISTRY denormalization is the easy miss.** `_registry_set` (line ~488)
caches `path/title/kind/status/has_clean_result` (+ `goal`/`paper`/`abstract`)
into REGISTRY.json — the dashboard LIST view reads REGISTRY, the DETAIL view
reads body.md. A mutator that writes body.md frontmatter but skips
`_registry_set` leaves the list view stale. `kind` IS one of the cached
fields, so `set_kind` calls `_registry_set` exactly like `set_title`.
(See [[feedback_registry_denormalization]] in the project auto-memory.)

**Enum drift gotcha (#672):** `batch` was a first-class `kind` everywhere in
the workflow (workflow.yaml exemption clauses `kind: analysis|infra|batch|survey`,
research-pm.md, pm/SKILL.md, CLAUDE.md) but was MISSING from the old
`task.py new --kind choices=[...]` literal — so `new --kind batch` was
silently rejected. Consolidating the literal into a `KINDS` tuple surfaced
this; the fix is to include `batch` in `KINDS` (aligning the CLI with the
documented exemption set), not to omit it. When you create a SSOT enum from
a scattered literal, grep the workflow surface (`grep -rn "kind:.*batch"
.claude/ CLAUDE.md`) for members the literal forgot.

**Routing rule (#672 docs side):** a fix-validation / "test that X works"
task is `kind: infra` (test-verdict path, NO promotable clean-result), NOT
`experiment`. The doc surfaces are CLAUDE.md § "Routing experiment intent"
(a routing bullet) AND `.claude/skills/issue/SKILL.md` Step 0b(3)
kind-inference (a `Test:`-cue override — `Test:` defaulted to `experiment`,
which IS the conflation) + the Error-handling table (point a misfile at
`set-kind`). Keep both consistent.
