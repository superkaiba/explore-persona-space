---
title: 'workflow-fix: verify pinned one-off installs resolve + dist-name != import-name'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3e647df1e47f
created_at: '2026-08-05T23:47:34Z'
has_clean_result: false
origin_prompt: 'failure-lesson from #2061 experimenter: A one-off pinned reference
  install (uv pip install pkg==X.Y.Z) must be verified RESOLVABLE at authoring time
  (PyPI JSON or uv pip install --dry-run) — and the PyPI dist name verified to be
  the intended library (PyPI sparsify is Neural Magic''s, not EleutherAI''s). sparsify==1.3.3
  does not exist, so the auto-provision fix could never succeed and the parity gate
  dies on every fresh pod.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson raised on task #2061 (emitting agent: experimenter; owning_agent it named: experiment-implementer).

## Goal

Require a one-off pinned dependency install to be verified RESOLVABLE at authoring time (`uv pip install --dry-run` or the PyPI JSON API) AND its DISTRIBUTION name verified to be the intended library; a successful local import proves neither.

## Workflow gap

- **Bug observed:** `ensure_sparsify()` pinned `sparsify==1.3.3`, a nonexistent version of an unrelated project (PyPI `sparsify` is Neural Magic's DEPRECATED sparsification UI; the EleutherAI distribution is `eai-sparsify`), so the install could never resolve and the loader-parity gate died ~2s in on every fresh pod.
- **Why it is a workflow gap:** `.claude/rules/code-style.md` already carries the sibling rule for a DIFFERENT identifier class — "Never hardcode an invented Claude/Anthropic model id ... Verify any hardcoded model string against the canonical list before committing; a wrong id crashes the run at the first API call (#489)". The pip-distribution analogue has no coverage at all, and it has the same shape: an unverifiable identifier typed into code that fails only at runtime, on a paid machine, after provision + bootstrap. The specific trap that makes it worse than the model-id case is that **dist name != import name**, so the natural "verification" (`import X` succeeds locally) is evidence for neither the distribution name nor installability — and a venv that happens to have the package makes the wrong pin look verified.
- **Confidence (emitter):** medium-high — the rule is one paragraph in an existing rules file with a clear sibling to sit beside; the judgement call is only how prescriptive to make the dry-run requirement.
- verified-at-filing: `grep -rnE '(uv )?pip install [^|]*[A-Za-z0-9._-]+==' scripts/ src/` → 2 hits in 1 file (`scripts/bootstrap_pod.sh:397,400`, `flash-attn==2.8.3`); the `eai-sparsify` site is on the unmerged `issue-2061` branch and so is not yet in main's surface. `grep -ciE 'pip install.*resolvab|dry-run|distribution name' .claude/rules/code-style.md` → **0** (the gap). `grep -c 'Never hardcode an invented Claude/Anthropic model id' .claude/rules/code-style.md` → 1 (the sibling rule this sits beside). Live PyPI probe at filing time: `flash-attn` 2.8.3 EXISTS and is the intended library (no latent bug at the one existing main-side site); `sparsify` 1.3.3 does NOT exist and the name belongs to Neural Magic; `eai-sparsify` 1.3.3 exists and provides `SparseCoder`. (2026-08-05)

## Proposed change (candidate diff sketch — refine in planning)

Add one bullet to `.claude/rules/code-style.md`, beside the model-id bullet:

+ - **A pinned one-off install must be verified RESOLVABLE, and its DISTRIBUTION
+   name verified to be the intended library.** Before committing any
+   `uv pip install <dist>==<ver>` that is not in `uv.lock` (parity references,
+   optional accelerators, gate-only deps), confirm (a) it resolves —
+   `uv pip install --dry-run '<dist>==<ver>'` or the PyPI JSON API — and (b)
+   `<dist>` is the library you mean. DIST NAME != IMPORT NAME: PyPI `sparsify`
+   is Neural Magic's deprecated sparsification UI while EleutherAI's is
+   `eai-sparsify` (import name `sparsify` in both readings), so a successful
+   local `import sparsify` proves NEITHER the dist name nor installability —
+   least of all in a venv that already has it. Prefer a post-install symbol
+   self-check (`uv run python -c "from <mod> import <Symbol>"`) so a wrong
+   distribution fails at install time with a legible message instead of inside
+   the consumer. A test asserting the pin should parse the dist name and
+   compare it EXACTLY — a substring check cannot distinguish `eai-sparsify==`
+   from `sparsify==` (the former contains the latter), which is how the wrong
+   name shipped past a green test on #2061.

## Scope / surfaces

- Primary target: `.claude/rules/code-style.md`
- Consider whether the implementer-side smoke-contract in `.claude/agents/experiment-implementer.md` should also name the dry-run check, since that is the agent the failure-lesson assigned ownership to.
- No mechanical lint proposed: the class is tiny (1 main-side file) and a lint would need network access to check resolvability, which is not appropriate for a pre-commit gate. Prose rule + the test-side exactness note is the right weight.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/code-style.md
- fingerprint: 3e647df1e47f

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/code-style.md
bug_observed: ensure_sparsify() pinned sparsify==1.3.3, a nonexistent version of an unrelated project (PyPI sparsify is Neural Magic deprecated; the EleutherAI dist is eai-sparsify), so the install could never resolve and the parity gate died on every fresh pod.
why_workflow_gap: code-style.md carries the sibling "verify a hardcoded model id" rule but has zero coverage of pinned-install resolvability or the dist-name-vs-import-name trap, so the same identifier-typed-into-code failure class recurs on a paid machine.
proposed_change: Require a one-off pinned dependency install to be verified RESOLVABLE at authoring time (uv pip install --dry-run or the PyPI JSON API) AND its DISTRIBUTION name verified to be the intended library; a successful local import proves neither.
diff_sketch: |
  + - **A pinned one-off install must be verified RESOLVABLE, and its
  +   DISTRIBUTION name verified to be the intended library.** dry-run or PyPI
  +   JSON before committing; DIST NAME != IMPORT NAME (eai-sparsify vs
  +   sparsify); prefer a post-install symbol self-check; a test asserting the
  +   pin must compare the dist name EXACTLY, not by substring.
confidence: medium
related_task: #2061
<!-- /workflow-fix-candidate -->
