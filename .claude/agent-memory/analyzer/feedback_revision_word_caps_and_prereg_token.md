# Revision-pass gotchas: per-result word-cap arithmetic + the `pre-registered` token

Two traps hit on the #813 per-example round REVISION pass (2026-07-05), both cheap to
pre-empt on any critique-mandated body edit:

1. **Check 20's per-`### <result>` cap counts the WHOLE H3 block** — the
   what-is-plotted paragraph, the image line (alt text + URL token), AND the
   interpretation prose all count via `_prose_words` (only captions `>`, tables `|`,
   fences, `<details>` bodies are excluded). Mature bodies sit at 170-179/180 FAIL
   cap, so critique-mandated ADDITIONS need compensating TRIMS in the same edit.
   Recipe: before editing, measure each block
   (`sys.path.insert(0,"scripts"); import verify_task_body as v` — plain
   `importlib.util.spec_from_file_location` crashes on the module's dataclass;
   then `_v4_results_body` + `_collect_tldr_h3_names` + `_prose_words`), budget the
   addition, trim what-is-plotted / alt text / connective prose to fit. Caption
   lines are cap-exempt (≤60-word WARN) — a numeric qualifier (e.g. a per-fold gap
   tail) can ride the caption at zero prose cost.

2. **`pre-registered` in body prose FAILs `audit_clean_results_body_discipline.py`**
   (the `pre_reg` anti-pattern; quality-bar item 7 "pre-registration mentions").
   STALE HALF-FIX WARNING (updated 2026-07-17, fix #1419): bare
   "registered <noun>" (e.g. "the registered verdict lattice", "registered
   hypothesis") NOW ALSO FAILS — the pattern grew a determiner-first
   'registered <noun>' branch after #1345's escapes. Write "the plan's <X>" /
   "plan-declared <X>"; the verb register stays clean ("registered on HF").
   SUPERSEDED (#1419, 2026-07-16): bare `registered <noun>` forms ("the registered
   verdict/margin/read/lattice") now ALSO fail the audit — the #1345 body followed
   this memory's old advice verbatim and drew a Lens 7 FAIL with six escapes. Do
   not reference registration status in reader-facing prose at all: state the
   criterion's value directly ("misses the −0.10 margin", "the 0.05 same-operator
   margin") with no "registered" qualifier; registration provenance lives in the
   plan file, and threshold VALUES may sit in the Methodology hyperparameter table.

Update (#813 concern round): check 20's per-result count EXCLUDES the `###`
heading line itself (`flines[line_no+1:end]`) — measuring a standalone block
with `_prose_words` over-counts by the heading's word count (a 24-word heading
made 181 read as 205). Measure heading-stripped, or run the real verifier on a
working copy. Also: a new same-section per-unit companion costs ~1 image line
(alt words + 1 URL token) + optional what-is-plotted pointer; captions stay
exempt, so put the recompute-gate numbers there.

Update (#813 r3): when a critique union forces additions into a `### <result>`
block already at 176-179 words, relocate the MECHANICS (λ-grid details,
skipped-check disclosures, oracle math) into `## Methodology` — it is EXCLUDED
from check 20's per-result and total-prose counts — and leave only a short
pointer qualifier in the Results prose ("under the plan's saturated λ grid
(Methodology)"). Then trim what-is-plotted phrasing ("dots own-grain fits,
ticks cross-grain transfers") before touching load-bearing numbers; measure
with the verifier's exact `_prose_words` rules (excludes `>`-lines, `|`-rows,
fences, details bodies; image + what-is-plotted lines COUNT).
