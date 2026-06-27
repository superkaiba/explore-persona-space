---
name: verify-paper-check-surface
description: Adding a verify_paper.py check fans out across ~9 mirror surfaces + a LaTeX preamble env + 2 test files; the paper-lens count is pinned in many places
metadata:
  type: reference
---

Adding a NEW `scripts/verify_paper.py` check (the `paper: true` LaTeX-paper
verifier) for a paper-format requirement (#664: verbatim examples + judge
prompts) fans out across a fixed mirror set. Hit ALL of them or the count drifts:

**The check itself + its tests**
- `scripts/verify_paper.py` — add the `check_*` fn, wire into the `verify()`
  `results = [...]` list, update the docstring check catalog (numbered 1..N in
  RUNNER order — renumber manifest/stub if you insert before them) + the
  `# ─── N. ... ───` section comment headers.
- `tests/test_verify_paper.py` — add present→PASS / missing→FAIL tests AND
  update the canonical check-NAME set in `test_no_metric_check_function_in_v1`
  (it asserts the exact runner check-name set; a new check must be added there
  or that test FAILs).
- `tests/test_build_paper_smoke.py` — the end-to-end build asserts
  `verify_paper.py` rc==0 on a TEMPLATE-DERIVED paper, so its `_FILL` dict +
  the template-section content MUST satisfy the new check (add example blocks /
  markers / a new `{{PLACEHOLDER}}` fill key). A new `{{FOO_BAR}}` placeholder
  in the template with `_` will LaTeX-error "Missing $" if left unfilled.

**The template the check keys on**
- `docs/papers/_template/issue_TEMPLATE.tex` (placeholder blocks + comments)
- `docs/papers/_template/preamble.tex` (any new LaTeX env). The pinned TeX-Live
  has `listings` + `tcolorbox` installed. A `newtcblisting` epsexample box needs
  `\tcbuselibrary{listings,breakable}`; `enhanced` needs the `skins` library
  (NOT loaded) — drop `enhanced` or load skins. Smoke-build before trusting it:
  `cp preamble.tex $T; <tiny .tex \input'ing it>; pdflatex -halt-on-error`.
  Verify content survives pandoc too: `pandoc h.tex -t html | grep SENTINEL`.

**The paper-lens count, pinned in ~9 prose surfaces (grep `P1-P6` / `six paper`)**
clean-result-critic.md (frontmatter + § Paper-task review + output template +
the "score the N lenses" + "all N lenses pass" lines), codex-clean-result-critic.md
(frontmatter + Step 1c + Step 3 paper-branch note + the recipe steps),
SKILL.md (TWO Paper-mode branch paragraphs — one enumerates the lenses),
agents-vs-skills.md (clean-result-critic + codex rows), workflow.yaml
(`epm:clean-result-critique` `fields:`), and markers.md (AUTO-GENERATED from
workflow.yaml — edit workflow.yaml then `workflow_lint.py --emit-tables` +
`--check-tables`, never hand-edit markers.md). Also SPEC.md § Paper format.

**Module-load gotcha for ad-hoc verify_paper probing:** loading it via
importlib needs `sys.modules["vp"] = mod` BEFORE `exec_module` or the
`@dataclass` decorator raises `AttributeError: 'NoneType' has no '__dict__'`.

Worktree venv: `uv run` in an agent worktree builds a fresh .venv (slow / can
ENOSPC); use the main checkout's `.venv/bin/python` on the worktree's file copy
(the test resolves verify_paper via `parents[1]/scripts/`, so the worktree copy
is exercised). See [[reference_worktree_uv_venv_disk_full]].
