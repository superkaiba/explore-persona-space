---
name: presentation-only-figure-fix-compose
description: "Presentation-only post-run figure-fix round (#2546 r23/v23): diagnosis-dispatched shape reused (progress-note diagnosis + round-record envelopes, marker-shape/smoke-run-missing NEVER-EMIT); figure artifacts NEW-to-git when the pre-fix render was deliberately never committed (pure-addition numstat is expected); orchestrator PNG reads are settled frame facts - twin lane = code + cell JSONs + bounded meta.json reads, PNG/PDF binary reads banned; symlog honesty decomposes per-axis into bars-vs-lines; census read-path divergence handed as adjudication; 0-vs-1-indexed row naming pre-empted"
metadata:
  type: feedback
---

From #2546 r23 compose (head sentinel v23, 2026-08-27), layered on
[[diagnosis-dispatched-round-compose]]:

1. **Presentation-only figure-fix rounds reuse the diagnosis-dispatched
   shape**: the defect diagnosis (`epm:progress` figure-sanity note) +
   the round record (`epm:progress` fix note) are the two contract
   envelopes; no per-round impl marker exists; `marker-shape` +
   `smoke-run-missing` NEVER-EMIT. The latest formal impl marker belongs
   to a CLOSED prior review loop - say so or the twin scores its shape.
2. **"Re-rendered" figure artifacts can be NEW to git** when the pre-fix
   render was deliberately never committed (the figure-sanity duty held
   it back). Pure-addition numstat on 33 figure files is then EXPECTED -
   state it as a frame fact ("re-render is relative to disk, not git")
   or the twin reads first-commit provenance as suspicious.
3. **Orchestrator PNG reads are settled frame facts.** The twin cannot
   read PNG/PDF binaries; its lane is the CODE, the committed cell JSONs
   (probe worktree presence - eval_results/issue_<N> can be PRESENT even
   in a sparse worktree when committed on the branch), and BOUNDED
   meta.json greps (sidecars run to thousands of lines). Visual claims
   (bars legible, gap list unchanged) are cited, never re-derived.
4. **Symlog honesty decomposes PER AXIS into bars-vs-lines**: (i) the
   twin first determines from code which artists are BARS on each
   patched axis (never assume); (ii) bar-boundary check = min bar value
   from the JSONs vs -linthresh (a crossing bar has non-proportional
   length - the misleading form); lines crossing are fine; (iii) top-cap
   clip check enumerates every series incl. CI whiskers/band dashes vs
   the cap; (iv) note when the helper sets ONLY a top cap (bottom
   autoscale preserves the mandated baseline's true value - that IS the
   baseline-preserved mechanism, name it for lane F-C).
5. **Census/title logic gets a read-path-divergence lane**: when the new
   census reads a RAW dict path while the plotted bars go through a
   helper (`_content_pool`), hand the divergence as a named adjudication
   (title-says-degenerate-while-bar-draws / blank-with-no-title), plus
   any `.get(key, default)` whose default flips the verdict on a
   populated block missing the key. Composer surfaces; twin adjudicates.
6. **Pre-empt 0-vs-1-indexed row naming**: commit messages count subplot
   rows 0-indexed, diagnosis prose 1-indexed - one frame-fact line
   ("same panels, not a discrepancy") prevents a bogus finding.
7. **Test-red attribution on a figure-only round** = the r17
   range-construction proof (name-only returns only the figure script +
   figures => every named offender byte-identical at BASE) + static
   duties: locate the test file YOURSELF (grep for the test name - the
   brief may not name the file), state its predicate, verify the round's
   own script satisfies it (dotenv-before-heavy-imports anchors), probe
   offender provenance (`git log -1 -- <file>` => the sync commit), and
   give ATTRIBUTION-CONFIRMED-PRE-EXISTING | PAYLOAD-ATTRIBUTED forms
   (the latter includes the test's predicate catching the round's own
   file).
8. **Cosmetic residuals recorded-not-fixed** in the round record get an
   explicit acknowledged/NIT-cap line, or the twin re-raises them as new
   findings.

Compose script: /tmp/codex-2546-v23-compose.py (fail-loud, labeled count
asserts, race re-probes: max impl version / max smoke-arch version / no
prior same-sentinel post; prompt /tmp/codex-prompt-issue-2546-v23.md,
38.7 KB). The SHA-floor assert caught a miscount live (CODE_SHA 5 not 6:
one occurrence lives inside the v213 envelope, not composer prose) -
count envelope-side occurrences separately before setting floors.
