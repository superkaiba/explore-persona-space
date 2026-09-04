---
title: Scope the PostToolUse ruff hook to files whose git origin is explore-persona-space
kind: infra
tags: []
created_at: '2026-09-04T17:15:55Z'
has_clean_result: false
origin_prompt: 'Interactive session 2026-09-04 (Intent app Parv/Beeper bridge fix):
  the EPS Edit|Write PostToolUse hook ran this repo''s ruff check --fix + ruff format
  on three Python files in ~/productivity_app (a sibling repo, Python 3.10 runtime),
  reflowing 2,500 unrelated lines and swapping timezone.utc for the 3.11-only datetime.UTC.
  Fix: require git config --get remote.origin.url of the file''s directory to name
  explore-persona-space; pin test test_hook_skips_other_repos_python. Landed directly
  from the interactive session by explicit-path commit after the inline lint gate
  PASS; no /issue cycle.'
workflow: v1
---

