---
title: workflow_lint _hf_routing_file_errors false-positives on HF-call text inside
  docstrings
kind: infra
tags:
- workflow-fix
created_at: '2026-08-17T16:43:19Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2223 strsweep judge-fix round: [live-hf-retry-routing]
  FAILs on the Schema-from-artifact mandated docstring paste in scripts/issue2223_casestudy_replay.py:45
  (pre-existing, byte-identical since f740a15f).'
workflow: v1
---
# workflow_lint `_hf_routing_file_errors` false-positives on HF-call text inside docstrings

**Surfaced by:** the #2223 strength-sweep judge-fix round (an experiment-implementer subagent), 2026-08-17.

**Gap:** `scripts/workflow_lint.py` `_hf_routing_file_errors` (the `[live-hf-retry-routing]` check) is line-based and skips only `#`-comment lines, so it flags HF Hub call text (`hf_hub_download(...)`, `torch.load(hf_hub_download(...))`, etc.) that appears inside a module **docstring** / any triple-quoted string.

**Why this is a standing false-positive, not a one-off:** the Schema-from-artifact rule (experiment-implementer § "Before writing code" item 8) MANDATES pasting the exact probe command into the file as an "Observed schema — probe:" docstring block. So every compliant consumer of a banked HF artifact trips a false `[live-hf-retry-routing]` FAIL on its own docstring. Live hit: `scripts/issue2223_casestudy_replay.py:45` (inside the module docstring; byte-identical since commit f740a15f — verified pre-existing, not introduced by the round that surfaced it). The Step 10d merge gate's baseline-vs-gated subtraction currently masks it (identical error line on both legs), but the no-flags `workflow_lint.py` run FAILs standalone.

**Proposed fix:** make `_hf_routing_file_errors` skip string/docstring content — either an AST-based scan (flag only real call sites, not string literals) or track triple-quote state while iterating lines. Keep the emitted message text byte-stable (the #1568 message-edit hazard note: changing a lint message text can invalidate baseline snapshots).

**Confidence:** high. Reproduce: `uv run python scripts/workflow_lint.py` on a tree containing `scripts/issue2223_casestudy_replay.py` → FAIL at `[live-hf-retry-routing] scripts/issue2223_casestudy_replay.py:45` (L45 is inside the module docstring).

**Regression pin to add:** a fixture file with an `hf_hub_download(...)` call inside a docstring AND a real live-unrouted call site — the checker flags only the latter.
