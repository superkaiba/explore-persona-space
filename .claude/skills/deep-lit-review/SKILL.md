---
name: deep-lit-review
description: >
  In-depth literature search protocol. Invoke whenever an agent does a
  thorough literature review: before any NEW research direction (the
  CLAUDE.md standing rule), when grounding a NEW formalization or a
  question with no prior in-repo grounding (no `Source: #<M>`) to inherit,
  or when the user asks for a deep lit search, "what's the prior work on
  X", or a related-work section. Routine plan-§11 hyperparameter Source
  lookups stay on the planner's existing bounded arXiv-MCP path — do NOT
  load this protocol for those. Also NOT for the bounded post-experiment
  positioning pass — that stays with the `related-work-finder` agent
  (≤6 MCP calls, proposal-only).
user_invocable: true
---

# Deep literature review

Evidence-grounded recipe (web sweep 2026-08-18, ~60 sources; full evidence +
URLs: `~/notes/content/LLM Literature Review Best Practices (Deep Research).md`
— private repo: `git -C ~/notes pull` before reading the local clone;
https://github.com/superkaiba/notes/blob/v5/content/LLM%20Literature%20Review%20Best%20Practices%20(Deep%20Research).md
requires Thomas's GitHub auth).
The two non-negotiables, from that evidence: **retrieval-grounding** (LLM
memory fabricates citations; retrieval-grounded tools rarely do) and **claim-vs-source
verification** (even 2026 frontier systems state a cited source's finding
fully accurately only ~44–77% of the time — verify what the paper says, not
just that it exists).

## Protocol

### Step 0 — Freeze the question

Write down, verbatim, before any search: (a) the research question; (b)
inclusion criteria (what makes a paper relevant); (c) exclusion criteria.
These become the screening rubric — screening decisions reference them
per-criterion, never holistically. Start a **search log**: every query,
tool, date, and hit count gets one line.

### Step 1 — Orient (bounded, ~5–10 calls)

Broad sweeps to build vocabulary and find surveys/seminal papers:
`mcp__arxiv__search_papers` + `mcp__arxiv__semantic_search` on the question
as phrased, plus 2–3 `WebSearch` calls (include `site:scholar.google.com`
variants and "survey"/"review" qualifiers). Output: a terminology list
harvested from REAL retrieved papers (citation pearl growing) — never
brainstormed keywords (LLM-brainstormed queries measure 4–44% of
human-strategy recall and hallucinate terms).

### Step 2 — Multi-modal retrieval sweep (loop until dry)

Rounds of querying with the harvested terminology, each round through
MULTIPLE channels (each channel is blind to what the others surface):

- arXiv MCP: `search_papers` (keyword) AND `semantic_search` (embedding).
- Semantic Scholar Graph API (free):
  `curl "https://api.semanticscholar.org/graph/v1/paper/search?query=...&fields=title,abstract,year,citationCount,externalIds"`.
- OpenAlex (free API key required as of 2026; free tier ~10k
  requests/day — ample): `curl "https://api.openalex.org/works?search=..."`.
- Web: Google Scholar via `WebSearch`, plus plain web for non-arXiv venues
  and very recent work.

After each round, harvest new terminology from the hits and re-query. STOP
when **2 consecutive rounds surface no new relevant paper** (loop-until-dry;
a fixed query count misses the tail).

### Step 3 — Snowball the citation graph

For every core (clearly relevant) paper: pull its references AND its citers —
`mcp__arxiv__citation_graph`, or Semantic Scholar
`graph/v1/paper/{id}/citations` and `graph/v1/paper/{id}/references`. Triage candidates by
title/abstract against the Step-0 criteria. Cheap recall insurance (the
PaperQA2 ablations measured +2–3 accuracy points from citation-graph
traversal); one hop is usually enough, two hops for survey-grade coverage.

### Step 4 — Screen (sensitivity-biased)

For each candidate: one verdict PER criterion in fixed order, quoting the
abstract text that grounds each verdict, then include/exclude. Borderline ⇒
INCLUDE for full-text read (missing a relevant paper is the expensive
error; reading one extra is cheap). Record excluded papers + the failing
criterion in the search log — never silently drop.

### Step 5 — Read + extract (quote-grounded)

For each included paper, read the FULL body (never abstract-only;
`mcp__arxiv__read_paper` /
`get_paper_latex`, or WebFetch the PDF/HTML). Produce a structured note:
citation (arXiv id/DOI) · setup (model/data/n) · claims relevant to the
question, EACH with a verbatim supporting quote + section · limitations ·
relation to our question (prior formalization / contradicts / method
source). The quote is the audit surface — a claim without a quote is not
extracted, it is remembered.

### Step 6 — Synthesize (notes only)

Write the review FROM THE NOTES, never from model memory: what is
established (with citations), where results disagree, the gap our question
sits in, and — for planner use — the closest prior formalizations and the
load-bearing hyperparameters/recipes with per-value `Source:` lines
(plan §11 grammar). Plain academic register; facts, not source-persons.

### Step 7 — Verify (mandatory, same turn)

1. **Resolution:** every cited arXiv id resolves via `get_abstract` (or DOI
   via `curl https://api.crossref.org/works/<doi>`), and the resolved title matches
   the note. Drop or fix anything that fails.
2. **Claim-vs-source:** for every load-bearing claim in the synthesis,
   re-check the note's verbatim quote actually supports the claim as
   written.
3. **Coverage pass:** one explicit search round for DISCONFIRMING and
   negative results (LLM retrieval skews toward highly-cited recent
   positives — the Matthew effect) plus a check that no candidate pile was
   silently truncated.

## Output artifact

One markdown doc: question + criteria (verbatim) · search log · included
list · excluded list (+ reasons) · per-paper notes · synthesis ·
verification log. Landing spot: planner runs fold it into the plan's
lit-review section (per-paper notes to the task's `artifacts/` dir —
resolve the path via `scripts/task.py find <N>`, never hand-build it);
chat-initiated reviews land in `docs/` or the location the user names —
durably, in the same turn (never /tmp-only).

## Fan-out variant (Agent-tool callers only)

An orchestrator/chat session MAY parallelize Steps 1–3 across 3–5 scout
subagents (split by sub-question or by channel), each returning structured
findings with bare URLs as its final message; the caller dedups, then runs
Steps 4–7 itself. Subagents executing this skill inline (e.g. the planner)
run it sequentially — they never spawn scouts. Briefs for scouts must state:
deliver findings as the final message in the same turn.

## Bounds

Default in-depth DISCOVERY budget (Steps 1–3 only): ≤ ~25 arXiv-MCP calls +
~15 web searches + ~10 API curls per question. Step-5 full-text reads (one
per included paper, typically 10–25) and Step-7 verification calls (one
resolution check per citation) are EXEMPT from that cap — never skip a
mandatory read or verification to stay under budget. A quick pass (the user
asked for "a quick look") keeps Steps 0, 2 (one round), 5 (top ~5 papers),
and 7.1 — the two non-negotiables travel even in the quick pass.
