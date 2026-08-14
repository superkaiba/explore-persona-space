# Shared-RNG-stream bootstrap aggregates: recompute the WHOLE aggregate, validate at 0.0, then perturb

Context: #2223 r2 (interp-critique blockers 1/3) — the drift driver's `phase_aggregate`
(`scripts/issue2223_drift.py`) seeds ONE `np.random.default_rng(42)` and iterates every
(domain, turn) cell in trajectory dict order, so each cell's CI depends on the position of
its draw in the shared stream.

Lessons:
1. **A per-cell reseeded recompute can NEVER match the persisted CIs** — the round-1 critic's
   own seed-42 recompute got hi/lo values swapped vs persisted for exactly this reason. To
   verify or perturb (row exclusion, seed sensitivity), re-implement the FULL aggregate loop
   verbatim (same iteration order, same single stream), assert exact reproduction on
   unmodified data (max abs deviation 0.0 — achievable, demonstrated), THEN run the
   counterfactual. The validation line ("deviation 0.0") belongs in the body/marker: it is
   what makes the counterfactual decisive rather than machinery noise.
2. **Excluding a row shifts the stream for all LATER cells** (n changes → different draw
   count) — cells before the perturbed one stay bit-identical (useful: the comparison arm's
   CI is unchanged), cells after do not. State reads accordingly.
3. **Verdict-flips-under-exclusion phrasing:** the as-run (all-rows) verdict stands as what
   the plan's rule was applied to; the flipped read is reported and the clause labeled
   "convention-dependent" — never silently kept, never retro-flipped.
4. **Interval-margin conventions differ:** gap-based (stable_lo − drift_hi) vs overlap-LENGTH
   (min(hi) − max(lo)) agree when no interval contains the other, diverge under containment
   (#2223 turn 10: gap −3.58 vs overlap 2.79). Pick one, define it in the table setup line,
   and match whatever numbers the critique/orchestrator already circulated.
5. **Sidecars persist mean lines, NOT fill_between bands** (#2223 r3): to ground a band-
   separation claim, replicate the figure's band machinery in exact figure/RNG order and
   VALIDATE the replicated mean lines against the sidecar at deviation 0.0 — then the
   band values are the rendered ones (turn-1 gap +2.554 == critic's +2.55).
6. **Ratio-of-drift bootstrap CIs are design-dependent**: another implementer's bounds
   will differ by a few points (stream/pairing choices). Recompute your own, persist the
   design + values in the committed artifact, quote the persisted numbers, and note the
   structural agreement (which intervals cross zero) rather than chasing bound equality.
7. **Word-cap relief for convention definitions**: define the interval-margin convention
   once in cap-excluded Methodology; the Results table setup line carries only a pointer
   ("gap-convention margins per Methodology") — the full definition inline blew a Result
   block to 201/180.
