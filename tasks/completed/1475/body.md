---
title: 'workflow-fix: pre_reg audit — verdict-lattice head nouns + range tokens escape'
kind: infra
tags:
- wf-fix
- wf-fix-fp:38a0acee2c04
created_at: '2026-07-17T07:53:38Z'
has_clean_result: false
origin_prompt: 'Formal workflow-fix-candidate block from clean-result-critic on #1090
  (epm:clean-result-critique v10): pre_reg branch head-noun alternation lacks cut/path/clause/control/lever/bar/smoke;
  multi-dot range tokens fail the intervening-token class'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a formal `<!-- workflow-fix-candidate v1 -->` block raised by the clean-result-critic on task #1090 (round-6 fold review, epm:clean-result-critique v10).

## Goal

Extend the `pre_reg` audit branch's bare-`registered <noun>` head-noun alternation (cuts?/paths?/clauses?/controls?/levers?/bars?/smokes?) and widen its intervening-token regex to admit multi-dot numeric range tokens (e.g. `0.60-0.85`), re-measured against the promoted-body corpus per the #1419 method.

## Workflow gap

- **Bug observed:** #1090's folded body carried ~12 bare registered-noun forms ("the registered 0.60-0.85 band", "registered kill path", "registered per-arm abort clause", "registered 0.30 cut", "registered install-strength control", "registered unrun lever", "registered 10% kill bar") that all escape the #1419 `pre_reg` branch — caught only by the LM critic (Lens 7).
- **Why it is a workflow gap:** the #1419 head-noun list was measured against the 2026-07-16 corpus; the verdict-lattice vocabulary now spreading through the factory line (cut / kill path / abort clause / kill bar) is the same Lens-7 jargon class and currently rides only on the LM critic — the exact escape mode #1419's own comment documents. Range tokens like `0.60-0.85` additionally fail the intervening-token class `[\w%/<>=≤≥−+-]+(?:\.\d+)*`, so "registered 0.60-0.85 band" never reaches the listed head noun "band".
- **Confidence (emitter):** medium
- verified-at-filing: `sed -n '<pre_reg alternation lines>' scripts/audit_clean_results_body_discipline.py` → the alternation at the `verdicts?|lattices?|…` lines contains none of cuts/paths/clauses/controls/levers/bars/smokes (presence-of-branch + absence-of-nouns both confirmed at source); `git log --oneline --since='7 days ago' -- scripts/audit_clean_results_body_discipline.py` → #1419 (9ef47afcfc) is the latest pre_reg touch, no later fix covers this (2026-07-17)

## Proposed change (candidate diff sketch — refine in planning)

-        r"(?:[\w%/<>=≤≥−+-]+(?:\.\d+)*[ \t]+){0,3}?"
+        r"(?:[\w%/<>=≤≥−+-]+(?:[.\-−]\d+)*[ \t]+){0,3}?"
-        r"(?:verdicts?|lattices?|margins?|reads?|criteri(?:on|a)|thresholds?|bands?"
-        r"|gates?|rules?|endpoints?|contrasts?|floors?|companions?|hypothes[ei]s|alpha)\b",
+        r"(?:verdicts?|lattices?|margins?|reads?|criteri(?:on|a)|thresholds?|bands?"
+        r"|gates?|rules?|endpoints?|contrasts?|floors?|companions?|hypothes[ei]s|alpha"
+        r"|cuts?|paths?|clauses?|controls?|levers?|bars?|smokes?)\b",

("test"/"interval" stay deliberately absent per the #1419 measurement method; re-measure the false-positive rate against the promoted-body corpus before landing.)

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'registered <noun>\|pre_reg' .claude/ CLAUDE.md scripts/`) and update every hit; check the audit's test file for pinned pattern expectations.

## Constraints / invariants

- Workflow-surface only. `workflow_lint.py --check-asks` passes; ruff passes; the audit's pinned tests updated alongside; false-positive re-measure against the promoted corpus recorded in the plan.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: 38a0acee2c04

(Verbatim candidate block preserved in the origin_prompt; raised by clean-result-critic, task #1090 round-6 fold review.)
