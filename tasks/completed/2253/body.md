---
title: 'Lint: cross-check production-branch third-party imports against uv.lock'
kind: infra
tags:
- workflow-fix
- prod-import-not-in-lockfile
created_at: '2026-08-12T20:59:43Z'
has_clean_result: false
origin_prompt: 'code-reviewer Rule 12 candidate at #2223 impl round 1: BLOCKER 1 is
  mechanizable as a lint — cross-check every non-smoke-branch third-party import in
  scripts//src/ against uv.lock (same family as --check-dotenv-before-hf-import).
  It would have caught this class before the pod.'
workflow: v1
---
# Lint: cross-check production-branch third-party imports against `uv.lock`

## Goal

Add a `workflow_lint.py` check that fails when a `scripts/` or `src/` module imports a
third-party package on a **production** (non-smoke) code path that is not resolvable from
`uv.lock` / `pyproject.toml`. This mechanizes a duty that is already documented but currently
only caught by human review.

## The gap

`.claude/rules/smoke-blind-spots.md` already names this exact class as one of the three things a
smoke run structurally cannot certify — *"every third-party import reached ONLY on the production
branch"* — and requires it to be disclosed in the plan's smoke blind-spot enumeration. But the
enforcement is entirely reviewer-side prose: nothing mechanically verifies that a
production-branch import is actually installable. The existing
`--check-dotenv-before-hf-import` check is the closest sibling (same family: an import-ordering /
import-resolvability invariant scanned over `scripts/` + `src/`), so the surface and the AST
machinery to extend already exist.

## Driving incident — #2223 code-review round 1 (2026-08-12)

`scripts/issue2223_drift.py:2106` contains, on the **production** branch:

```python
from sentence_transformers import SentenceTransformer
```

Verified at review time: `sentence-transformers` is absent from `pyproject.toml` AND from
`uv.lock` — not even as a transitive dependency — no bootstrap script installs it, and that
driver is its only importer in the repo. So `--phase ridge` would have raised
`ModuleNotFoundError` immediately after the pod's `uv sync`.

The blast radius is what makes this worth a lint rather than a nit. The launcher runs `ridge`
**immediately before `upload`** under `set -euo pipefail` on both model legs, so the crash halts
the chain and **every file under `raw_completions/` — the run's primary deliverable — never
uploads.** That is the #779 failure class (a generation stage that drops its generations is an
upload-verification FAIL regardless of intent), reached here purely through a missing dependency
in an *optional analysis add-on*.

It was invisible to the pre-launch smoke by construction: `_embed_messages` substitutes random
embeddings under `--smoke`, so the production import never executed. This is the same shape as
#1336 / SLURM-4684, where a module-local `sentence_transformers` import one call below a
smoke-branch site killed an 8-GPU production run.

## Proposed check

- AST-scan `scripts/**/*.py` and `src/**/*.py` for `Import` / `ImportFrom` nodes.
- Classify each module root as stdlib (allowlist / `sys.stdlib_module_names`), first-party
  (`explore_persona_space`, local modules), or third-party.
- Resolve third-party roots against the package set in `uv.lock` (plus `pyproject.toml`
  dependencies), accounting for import-name vs distribution-name mismatches
  (`sentence_transformers` → `sentence-transformers`, `sklearn` → `scikit-learn`, `cv2` →
  `opencv-python`, `yaml` → `PyYAML`, etc. — a mapping table is required, and an unmapped
  unknown should FAIL loudly rather than be skipped).
- FAIL on any unresolvable third-party import. A deliberately optional dependency must be
  declared — either added to the lockfile or guarded by an explicit, documented
  `try/except ImportError` that degrades a **non-load-bearing** path (and note the project's
  fail-fast rule: such a guard must never swallow a fault on a load-bearing path).
- **Bundle into the no-flags default run**, and add the corresponding
  `test_<check>_bundled_in_no_flags` pin test (`verify_plan.py` check 37's contract).

## Acceptance

- The check FAILs on a fixture reproducing the #2223 shape (production-branch import of a package
  absent from `uv.lock`, with a `--smoke` branch substituting a stub) and PASSes once the package
  is added to the lockfile.
- The check FAILs on the #1336 shape: the import nested inside a module-local helper called from
  the production branch, not at the branch site itself.
- No false positives across the current `scripts/` + `src/` tree — a full-tree run is part of the
  acceptance evidence, and any genuine pre-existing hit is reported rather than silently
  allowlisted.
- Import-name→distribution-name mapping is table-driven and unit-tested.
- `tests/` pins both the check behaviour and its no-flags bundling.

## Related

- `.claude/rules/smoke-blind-spots.md` (the documented duty this mechanizes; also records the
  existing scanner's known false-negative surface — module-local helper resolution only one level
  deep, `ast.Match` case bodies not recursed, dynamic dispatch, non-`smoke`-named flags)
- `#1336` (SLURM-4684: `sentence_transformers` import one call below a smoke branch killed an
  8-GPU production run — the precedent shape)
- `#779` (generation stage dropping its generations = upload-verification FAIL — the blast radius)
- `workflow_lint.py --check-dotenv-before-hf-import` (nearest existing sibling check)

## Provenance

Surfaced by the `code-reviewer` at #2223 implementation round 1 (2026-08-12) as a Rule 12
workflow-fix candidate: *"BLOCKER 1 is mechanizable as a lint — cross-check every non-smoke-branch
third-party import in `scripts/`/`src/` against `uv.lock`."* Auto-filed by the #2223 orchestrator
per the workflow-fix-on-bug protocol. The #2223 blocker itself is being fixed in that task's
round 2; this task is the general enforcement so the class cannot recur.
