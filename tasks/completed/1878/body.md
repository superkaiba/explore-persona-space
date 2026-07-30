---
title: 'workflow-fix: gotchas span-family entry — find-from-0 query mis-anchor in
  chat-template renders'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c5c43b350e59
created_at: '2026-07-30T09:46:13Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1776 crash-fix cycle 10: find-from-0
  question locator over a chat-template render mis-anchors short real-user queries
  inside the template preamble (crash at prefix_len=0 or silent garbage spans); anchor
  from the content-independent template tail; 1-2-char queries are a standing collision
  sub-class of bare real-user corpora'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson `gotcha_candidate: yes` block raised on task #1776 (emitting agent: experiment-implementer, crash-fix cycle 10).

## Goal

Add a `.claude/rules/gotchas.md` entry to the span-rig family (the three BPE-seam siblings at ~L403-405): locating a real-user query in a chat-template render via `text.find(question, 0)` mis-anchors any SHORT query that substring-matches inside the template preamble — anchor the final user turn from the content-independent template TAIL instead of searching.

## Workflow gap

- **Bug observed:** #1776's p3p4 follow-up round crashed at p3_pcj: `compute_prompt_spans` got `prefix_len=0` because a 1-char real-user query matched inside the Qwen default-system preamble (115 chars/24 tokens) at token 0; 11 more WildChat rows + 1 LMSYS row matched LATER in the preamble, silently persisting garbage spans that PASS the span assert (the parent round's committed averaged J contains that 1/1536 mis-spanned row — disclosed at fold).
- **Why it is a workflow gap:** gotchas.md's span family (~L403-405) covers three BPE-MERGE victims (zero-width spans, teacher-forced position shifts, parity-gate tails) but no member covers the query-LOCATION defect — find-from-0 preamble collision — which is orthogonal to BPE merging, structurally recurrent on ANY bare real-user corpus (1-2-char queries exist in every WildChat/LMSYS pool), and invisible to smoke slices that never sample a short-query row (the same smoke-coverage blind spot the family already documents).
- **Confidence (emitter):** high
- verified-at-filing: `grep -c -i 'compute_prompt_spans\|span-find\|find-from-0' .claude/rules/gotchas.md` → 1 hit (the #1315 sibling's reference-impl pointer; no entry covers find-from-0 mis-anchoring — absence claim; family anchor at L403-405 present); landed-fix history `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md` → 8 commits, none span-anchor related (2026-07-30). The CODE fix landed on issue-1776 (`a9c47aa847`: `_suffix_q_span` tail anchoring + opt-in `q_char_span` + STALE-SPANS resume invalidation) — this filing is the DOC entry.

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **Locating a real-user query in a chat-template render by
+   `text.find(question, 0)` mis-anchors any SHORT query whose text
+   substring-matches inside the template PREAMBLE (Qwen default-system:
+   115 chars / 24 tokens).** Fourth member of the span-rig family (the
+   three BPE-seam siblings above): a token-0 match crashes the span
+   assert loudly (prefix_len=0 — #1776 c10, 4/999 WildChat rows, 1-char
+   queries); a LATER preamble match SILENTLY persists garbage spans that
+   PASS the assert (11/999 WildChat + 1/1536 LMSYS measured — the silent
+   rows poison committed aggregates). RULES: (i) anchor the final user
+   turn from the content-INDEPENDENT template TAIL (the fixed suffix
+   between the last user content and the assistant turn — exact by
+   construction, no search); (ii) treat 1-2-char queries as a standing
+   collision sub-class of every bare real-user corpus — any span-rig
+   smoke must include one; (iii) when fixing a mis-anchoring locator
+   mid-run, add a STALE-SPANS resume invalidation keyed on a
+   legacy-agreement predicate so good units skip and only mis-anchored
+   units recompute; (iv) audit parent rounds that used the find path —
+   silent garbage spans may sit inside committed aggregates (disclose).
+   Worked impl: _suffix_q_span + render_pair(anchor=) in
+   scripts/issue1776_jacobian.py, opt-in q_char_span in
+   analysis/representation_shift.py (fix a9c47aa847, branch issue-1776).
+   Long-form twin: .claude/agent-memory/experiment-implementer/
+   feedback_chat_template_span_find_misanchor.md.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Place adjacent to the span-rig family (~L403-405) so the four siblings cluster.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; gotchas.md `paths:` frontmatter untouched (the span/capture trigger set already covers this member).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: c5c43b350e59
