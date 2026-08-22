---
name: pytest-collection-guard-plan-review
description: Reviewing plans that add pytest collection-time registry/state guards or -k-selected verification commands (#2217)
metadata:
  type: feedback
---

Two 2-min checks for infra plans adding pytest collection-time state guards or exact-count `-k` verification rows (#2217, PASS with 3 Should-Fix):

1. **`-k` matches the module FILENAME, not just function names.** An exact-count threshold ("1 passed") on `pytest -k <module-stem>` silently selects EVERY test in that file (incl. the guard's own synthetic-control sibling). Replay the -k expression against the planned test names + filename; fail-loud direction ⇒ Should-Fix, not Must-Fix.
2. **Additions-only key-set diffs (`now - prev`) prove "no additions", not "final state == baseline".** An AC claiming key-set EQUALITY with a fresh-import baseline needs a retained configure-time snapshot + final equality assert (~2 lines), else pure removals and pre-`pytest_configure` imports (conftest/plugins absorbed into the baseline) are invisible. Both are Should-Fix when a loud backstop exists (e.g. an exact-count seed pin elsewhere); demand they be named in the plan's stated false-negative profile.

**Why:** #2217's plan was otherwise exemplary (guard negative-controlled on the real incident pre-merge — the [[infra-plan-review-checklist]] rule-A demand); these were the only gaps in a 5-question stress test.

**How to apply:** any `kind: infra` plan with a `pytest_collectreport`/`pytest_configure` hook, a registry-hygiene fixture, or exact pass/fail-count criteria on `-k`-filtered commands. Also: pytest deselection (`-k`/`-m`) is POST-collection, so a full-tree `--collect-only`-style run + one asserted test IS a valid full-import check.
