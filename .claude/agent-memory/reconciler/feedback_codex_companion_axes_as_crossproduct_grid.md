---
name: Codex inflates companion-axis enumerations into a cross-product grid
description: Codex code-reviewer reads plan "both conventions; both normalizations" (two 1-D companions off a primary) as demanding the full 2x2 grid, then FAILs the missing 4th cell as a pre-registration Critical — even quoting a phrase ("all convention/normalization variants") that appears nowhere in the plan
type: feedback
---

When a plan enumerates pre-specified robustness companions as separate 1-D axes
("both partial conventions (figure primary, analysis companion); both
normalization variants (per-token primary, un-normalized companion)"), Codex
may read the enumeration as a cross-product GRID and FAIL the implementation
for omitting the never-pre-registered corner cell (analysis-convention
un-normalized). It may even quote a requirement phrase verbatim that does not
exist in the plan text.

**Why:** Origin task #548 round-1 (2026-06-10). Codex Critical:
`issue548_length_analysis.py` builds kill-bearing CIs for 3 variants
(fig/pt, ana/pt, fig/unnorm) not 4. Verified against: (a) plan.md:110
enumerates two 1-D companions; (b) plan.md:311/337 pin the variant set to the
parent #540 supplement, and the parent's published
`length_nuisance_supplement.json` `js_rb_unnormalized` block has ONLY
`partial_length_figure_convention` — the 4th cell never existed in the
reference; (c) plan.md:34 quotes the fig/unnorm value (−0.116/p=0.06) as THE
normalization companion; (d) grep for Codex's quoted phrase → zero hits;
(e) "all four CI bounds per convention/normalization" = 4 bounds (iid/cluster
× lo/hi) PER variant, not 4 variants. Bonus: under a strict all-variant
unanimity rule, a missing extra variant can only have made alive/dead calls
MORE reachable, never flip alive↔dead — and the omitted cell's POINT estimates
were computed + persisted anyway, so the call is auditable.

**How to apply:** When Codex FAILs a "missing pre-registered variant/cell":
1. Grep the plan for Codex's quoted requirement phrase — Codex sometimes
   synthesizes the quote.
2. Check whether the plan pins the variant set to a parent artifact, then open
   that artifact's actual key set (the reference implementation defines the
   grid).
3. Parse "both X; both Y" as two 1-D companions unless the plan says
   grid/cross-product/all-four explicitly.
4. Check impact direction under the implemented agreement rule (unanimity →
   extra variants only downgrade toward indeterminate).
Companion entries: "Codex misparses trailing aggregation parenthetical"
(#537 r4 — same plan-prose over-reading family).
