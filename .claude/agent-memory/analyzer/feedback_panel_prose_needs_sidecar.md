# Panel-position prose needs a sidecar at the pin; pre_reg scrub on re-folds

Context: #1090 fu6 body fold (2026-07-17), revision round 2.

1. **verify_task_body check 26 (figure panel prose vs sidecar).** The trigger is
   `\b(left|right|top|bottom|middle)\s+panel\b` (or a dot/point-overlay phrase)
   co-occurring with a plot-kind word (scatter/line/bar) in the what-is-plotted +
   caption prose. If the figure's sibling `.meta.json` does not resolve at the
   cited SHA, it is a hard FAIL — driver-generated figures (not saved via
   `savefig_paper`) usually have NO sidecar. Fix: position-free wording ("right
   half of the figure; the left half belongs to the next result") — never
   hand-write a sidecar.

2. **The #1419 `pre_reg` audit branch FAILs grandfathered bodies on re-fold.**
   `audit_clean_results_body_discipline.py` matches `registered` + up to 3
   intervening tokens + a head noun (verdict/lattice/margin/read/criterion/
   threshold/band/gate/rule/endpoint/contrast/floor/companion/hypothesis/alpha),
   prose-only on v4 (tables blanked). A body promoted before 2026-07-15 can carry
   a dozen matches ("the registered band", "registered lattice scores") — the
   audit REPORTS INCREMENTALLY (~3 per run), so enumerate all matches with the
   regex in one python pass, then scrub each by stating the criterion's value
   directly ("the 0.60-0.85 band", "the three-way verdict lattice"). Note the
   intervening-token class rejects mixed dot-hyphen numerals: "registered
   0.60-0.85 band" does NOT match while "registered 0.60 band" DOES — scrub the
   near-misses too if a Lens 7 bounce is a risk.

3. When a re-anchor round WEAKENS a prior round's section headline (e.g.
   "installs in every training context" → 7/10 under a new instrument), scope
   the old heading ("Under the fourth round's rubric ...") + add a one-clause
   forward pointer in its interpretation — don't leave a contradicting absolute
   claim standing next to the new Takeaways.
