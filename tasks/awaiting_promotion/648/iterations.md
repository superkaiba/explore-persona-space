# Iteration capture — #648 clean-result

## Round 1 (analyzer first draft)

Fresh file. No user corrections yet.

Key analytical decisions recorded for the critic / future rounds:

- **Confidence = LOW.** Verdict-eligible panel is 4 banks; only 1 (#505) is
  determinate, and it favors RAW. No centered-favored determinate bank, no
  MIXED determinate signs. The pre-registered §6 mapping (H_centered-better /
  H_bank-dependent / H_raw-inflates) does not cleanly land on any single
  hypothesis: H_centered-better is falsified (a raw-favored determinate bank
  exists), H_bank-dependent is NOT supported (it requires mixed determinate
  signs across ≥2 eligible banks; there is only one determinate sign), and
  H_raw-inflates' compression leg is present but does not predict the ΔR² sign.
  The implementer's report said "maps to H_bank-dependent" — that is an
  over-read; corrected to "no centered-better; a single raw-favored bank; the
  rest indistinguishable or both-fail."

- **The in-sample-vs-CV divergence on #505 runs OPPOSITE to raw-inflation.**
  On #505, centering IMPROVES in-sample Δρ (+0.208, CI excludes 0) yet HURTS
  out-of-sample ΔR² (−0.071, CI excludes 0). H_raw-inflates predicts raw
  inflates in-sample; here it is CENTERED that fits the sample better while RAW
  predicts held-out better. So #505 is not an H_raw-inflates instance. The two
  low-N banks (#66/#142) DO show raw fitting in-sample better (Δρ < 0, CI
  excludes 0) — those are the inflation candidates, but they are
  verdict-ineligible (n_groups = 5, MF3).

- **Three banks (#380/#396/#311) both-fail out-of-sample.** Both recipes score
  CV R² < 0 (worse than the train-mean) on the 24/24/17-persona banks. This is
  the load-bearing secondary finding: the in-sample ρ #536 reported on small
  banks does NOT translate to held-out predictive skill. Excluded from the
  H-verdict by MF4 precedence, reported as its own finding.

- **Compression diagnostic is partial coverage (3 of 9 banks).** bank_offdiag
  exists only for single_token_100p_L20 / extraction_method_a_L20 /
  issue505_pv_L21. Raw is uniformly compressed into ~[0.7,1.0] on all 3; do NOT
  claim H_raw-inflates on #311/#380/#396 (no compression read).

- Numbers re-extracted from per_bank_skill_table.json at write time
  (commit db51418618, rng_seed 20648, n_boot 10000).

## Round 2 (interpretation-critic union REVISE — both critics)

Addressed the union of 5 Must-Fix items (2 Claude, 3 Codex) + 2 optional
PASS-class concerns. All numbers re-extracted from per_bank_skill_table.json at
this round's write time.

- **MF1 (Claude) — #505 structural idiosyncrasy.** Added to Takeaway #1 +
  Finding-1 read: the sole determinate bank is the panel's only persona-vector
  bank (family `issue505_pv_L21`), only L21 bank, only multi-arm design, so the
  raw edge could be a structural property of THAT bank, not of raw cosine — the
  binding LOW-confidence caveat.
- **MF2 (Claude) — Finding-3 caption ↔ figure mismatch.** Chose option (b):
  regenerated `paired_r2_raw_vs_centered.png` restricted to the 3 both-fail
  banks (was all 9 — heading/caption narrated a 3-bank story over a 9-bank
  chart). Edited `paired_r2_bars()` to filter `exclusion_reason ==
  "both_predictors_fail_oos"` (asserts exactly 3) + label each bar with its
  held-out unit. Re-pinned the body figure URL + figures-script Code link to
  the new SHA c745da0abc (the other 3 figures stay at d83bcc25).
- **Codex rev #1 — eligibility prose mathematically WRONG.** Body said
  "more than 5 folds AND both recipes beat baseline"; the JSON rule is
  `n_groups > 5 AND NOT both-recipes-fail` (at least one beats baseline).
  #490 (raw −0.0074 fails, centered +0.0096 beats) and #505 (raw +0.0694 beats,
  centered −0.0014 fails) are both eligible, contradicting the old prose. Fixed
  in "## What I ran" Scope-shrinkage AND the Parameters verdict-eligibility row.
- **Codex rev #2 — conflated CV-unit prose.** "below ~30 personas" /
  "leave-one-persona-out" was wrong: the 3 banks are leave-one-persona-out
  (#380), leave-one-source-out (#396/#415), leave-one-bystander-out (#311).
  Rewrote to held-out-unit-neutral "small-N end (8-24 held-out folds)" + the
  units are now explicit on the regenerated figure x-axis.
- **Codex rev #3 — #505 is the LARGEST panel (independent rule-out).** 52 folds
  vs {35,24,24,17,11,8,5,5}; centering's train-fold-refit leak is smallest where
  n is largest, so 505 had the cleanest CV-hygiene and centering STILL lost —
  rules out a centering-leak artifact. Added to Takeaways + Finding-1.
- **Optional 1:** Takeaway #2 notes the two best-powered non-resolvers lean
  nominally centered (#478 +0.030, #490 +0.017), the resolver leans raw — the
  panel is genuinely split, not raw-tilted.
- **Optional 2:** per-bank Data table gained a "Layer / bank type" column.

Trimmed Finding-1 178→168 words to stay clear of the 180-word hard FAIL after
adding the MF1 caveat. verify_task_body.py --issue 648 = OVERALL PASS.

Generalizable lesson (portable → analyzer memory candidate): when a verdict
rests on a SINGLE determinate cell, check whether that cell is also the
structural outlier of the panel (lone layer / lone bank-family / lone design)
AND, separately, whether it is the best-powered cell — the two cut opposite
ways (outlier = downgrade; best-powered + cleanest-CV-and-still-lost = rule out
the favorable-artifact alternative). Both belong in the headline finding.

## Round 2 → Round 3 (clean-result-critic union REVISE — structure/register only)

clean-result-critic ensemble (Claude + Codex) REVISE'd round 1 of the v3 body
on STRUCTURE + REGISTER + STATISTICAL-FRAMING discipline (content settled; no
number / verdict / confidence change). Addressed the 6-item Must-Fix union:

- **MF-A (BOTH critics, blocking) — `interval_inline` audit FAIL.** The discipline
  audit flagged 3 distinct inline-bracket sites (`[+0.153, +0.264]` x2,
  `[0.7, 1.0]`). Rewrote every CI in reader-facing prose (Takeaways + finding
  reads + captions) to point-estimate + "CI excludes 0"; rewrote the
  distribution-support range `~[0.7, 1.0]` → `~0.7 to 1.0`. Audit now exits 0.
- **MF-B (Codex) — Takeaways bullets >30 words.** 4 of 6 bullets were 75/49/57/33.
  Split the 75-word headline into two bullets (headline + structural-outlier
  caveat, both load-bearing), merged the validity-grounds point into the
  both-fail bullet to stay ≤6 bullets, trimmed each over-cap bullet. Now 6
  bullets, all ≤30 words (verifier-tokenizer-counted).
- **MF-C (Codex) — missing `**Training:**` slot.** Added `- **Training:** n/a —
  no training; CPU-only re-analysis...` between Design and Eval. Kept the
  `**Scope shrinkage:**` slot (useful per-task content).
- **MF-D (Codex) — finding read paras ≥3 sentences.** Bulletized the read block
  BELOW each figure in Findings 1/2/3/4 (setup sentence above each figure stays
  prose). Numbers-first ordering preserved.
- **MF-E (Codex) — issue-links in `## Data` prose.** Chose option (a): kept the
  short-form parenthetical bank-IDs `(#66)`...`(#505)` in the per-bank table
  (the way these banks are referred to), moved the prose `[#536]`/`[#404]`/
  `[#458]` provenance links out of `### Trained on` into `## Reproducibility`.
- **MF-F (Codex) — reuse-provenance per-artifact paths.** Expanded the single
  generic bullet into per-artifact bullets: centroid banks + targets from #536
  (keyed per-bank `family`, grounded on the actual `join_gate_max_dev` < 3e-7
  and the imported `GATE_MATRIX_TOL = 1e-4` / `GATE_RHO_TOL = 0.02` gates,
  verified against the driver at SHA 4ff0a15c43 + the per_bank table), the 9
  producing-task target links, the 111-persona JSON restore, #404/#458 origin.
  Did NOT invent disk paths — the table keys by `family`, not a path column.

Downgraded Codex items skipped with rationale: title em-dash (#1 — reads as one
claim + supporting beat, Claude PASSed it); caption math notation (#8 — ΔR²/Δρ
are project-standard, Claude PASSed captions).

Also opportunistically trimmed per-finding prose (Finding-1 166→152, total
1046→1003 words) — all remaining conciseness flags are WARN-only (soft >120,
none near the 180 hard FAIL). verify_task_body.py --issue 648 = OVERALL PASS;
audit = exit 0.

---

## 9a-bis round 2 → round 3 (final 9a-bis round)

Round 2: Claude PASS, Codex REVISE → reconciler verdict REVISE, two binding
items carried forward; Codex's MF-2 (`**Scope shrinkage:**` slot) dropped as
non-blocking. Body content / numbers / LOW verdict were already settled — the
two remainders were pure SPEC structural conformance.

- **MF-5 — issue numbers out of the `## Data` table.** The reconciler ruled the
  "issue numbers confined to `## What I ran` `**Why:**` + `## Reproducibility`"
  rule applies BY SECTION, not prose-vs-table — superseding round-2's option-(a)
  decision to keep the bank-ID parentheticals in the table. Stripped the trailing
  `(#K)` from all 9 per-bank rows (`100-persona marker bank (#66)` →
  `100-persona marker bank`, etc.). Full producing-issue lineage stays in
  `## Reproducibility`. Residual `(#K)` in the table region: 0.
- **MF-6 — reuse-provenance bullets need explicit pinned paths.** Round 2's MF-F
  expansion keyed banks by `family` and explicitly did NOT add disk paths
  ("did NOT invent disk paths"); the reconciler ruled the SPEC `**Artifacts:**`
  shape REQUIRES (b) a readable path per reused artifact. Read the actual paths
  off the driver script (`scripts/issue648_centered_vs_raw_predictive_skill.py`
  @ `4ff0a15c43`) and added them:
  - Centroid-banks bullet: per-family on-disk centroid `.pt` paths
    (`.../centroids_layer20.pt`, `centroids_method_a.pt`,
    `issue_274/.../centroids_n24_layers0_27.pt`, `issue_311/centroids_base.pt`,
    `data/issue_505/.../centroids_pv_L21.pt` + HF fallback). Given as plain
    inline code paths (git-ignored binaries) — NOT blob links, which 404 the
    verifier's "Reproducibility artifact URLs exist" check.
  - Leakage-targets bullet: explicit per-target file paths for all 9 banks.
  - 111-persona distance JSON bullet: added the missing producing issue `[#560]`
    (from the fact-checker Phase-1.5 finding) + the explicit repo path, kept the
    SHA-pinned blob link at `776c7c3b75` (git-tracked, verified resolvable).

One round-3 verifier FAIL caught + fixed in-loop: the centroid-`.pt` blob link
404'd (git-ignored binary) — converted to a plain inline code path. Final gates:
verify_task_body.py OVERALL PASS (30 checks, 2 pre-existing WARNs unchanged);
audit exit 0. This was the final 9a-bis round; orchestrator advances after v4.
