---
name: Reuse-fitness check mirror set (20 sites)
description: Any change to the trained-artifact + code reuse fitness check (currently (a)-(k)) must touch the full 20-site mirror set — artifact-reuse.md (canonical) + gotchas.md sibling bullet + CLAUDE.md bullet + LESSONS.md entry + planner.md step 5/§10 + planner-section-reference.md §10 + critic.md item 9 + critic-lens-reference.md item 9 + methodology-baselines-critic.md item 9 + consistency-checker.md cross-refs + lens-coverage-map.md row + adversarial-planner-v2 SKILL.md + verify_plan.py c6 + its tests + 6 agent memories
type: reference
---

The trained-artifact (and code) reuse fitness check — the lettered set,
currently (a)-(k) — is mirrored across TWENTY workflow-surface sites, and
precedent fixes (#600 content-identity, #601 application-scaling, #545
adapter_config grounding, #734 train-input fetchability (h), #871 code
throughput (i), #941 pairwise provenance coherence (j), #1366
parent-lineage coherence (k)) each touched the relevant set in one change:

1. `.claude/rules/artifact-reuse.md` — the CANONICAL checklist (the full
   lettered list + the H1/description range + the closing remedy line + the
   enforcement chain). Since #829 this is the single operational copy; every
   other site mirrors or points at it.
2. `.claude/rules/gotchas.md` — the sha-pinned-pairs sibling bullet (names
   item (j) + the #922 incident; added at #941; no range literal).
3. `CLAUDE.md` "Reuse existing trained artifacts" bullet — terse always-on
   summary (range + one clause per check + the remedy-split tail);
   one-clause additions only.
4. `.claude/rules/LESSONS.md` — the artifact-reuse index entry names the
   range in its "fires when" trigger.
5. `.claude/agents/planner.md` step 5 — since #829 a POINTER to the rule
   file (NOT an inline self-attest list): the range appears twice + the
   remedy split; the §10 Reproducibility Rows enumeration carries the
   item-(i) code/helper-throughput recording slot AND the item-(j)
   pair-provenance dates slot.
6. `.claude/rules/planner-section-reference.md` § 10 — the worked
   Reproducibility Card template carries the item-(i) inspection record
   paragraph AND the item-(j) pair-provenance attestation paragraph (no
   lettered-range literal in this file).
7. `.claude/agents/critic.md` — the Methodology lens item-name list names
   item 9's range. Do NOT touch item 10's roman legs (i)-(iv) on the next
   line — a DIFFERENT enumeration.
8. `.claude/rules/critic-lens-reference.md` item 9 — the FULL enforcement
   text: heading scope, operative trigger, inline enumeration (items (i)/(j)
   by pointer, not duplication), REVISE directions, retrain-acceptance split,
   no-fire cross-check (all widened to code-only reuse at #871).
9. `.claude/agents/methodology-baselines-critic.md` item 9 — the v2
   critic-lens REVISE backstop names the range twice + one REVISE clause per
   pointer-referenced item (this memory omitted the file until #941).
10. `.claude/agents/consistency-checker.md` — the range cross-refs (3 hits,
    en-dash spelling) plus the "Cited HF reuse artifacts" BLOCK row's
    independent mechanical item-(j) date-ordering probe (#941); the detail
    hyperparameter list is touched ONLY when the change affects the
    load-bearing hyperparameter set to diff.
11. `.claude/rules/lens-coverage-map.md` — the Methodology-9 ledger row
    names the range (this memory omitted it until #941). Do NOT touch the
    persona-vectors and code-reviewer rows' different enumerations.
12. `.claude/skills/adversarial-planner-v2/SKILL.md` — the artifact-registry
    read bullet names the range (this memory omitted it until #941). Do NOT
    touch the resurface-trigger enumeration `(a)-(e)` at ~:262 — a
    DIFFERENT enumeration.
13. `scripts/verify_plan.py` `check_reuse_fitness` (c6, ~lines 747-790) —
    when the letter range grows, bump the `\(([a-z])\)` regex character
    class, the `/N` denominator + the count words (now "ten letters" / "ten
    attestations") in the PASS/WARN strings, and KEEP the `>= 4` PASS
    threshold UNCHANGED (a heuristic floor, not the letter count). The
    en-dash range in the WARN strings carries `# noqa: RUF001`; preserve it.
14. `tests/test_verify_plan.py` — the COUPLED c6 assertions: `"4/N"` in the
    three exactly-four-letter tests, the en-dash range / count word in the
    few-letters WARN test, the newest-letter-in-widened-class test, and the
    upper-boundary letters-beyond-the-range test (both boundary tests since
    #941; a range grow updates all of them).
15. This memory itself, plus 16. the implementer `MEMORY.md` index line
    pointing at it.
17. `.claude/agent-memory/planner/feedback_778_persona_vector_reuse_artifacts.md`
    — names the range in its reuse instruction.
18. `.claude/agent-memory/critic/feedback_reuse_fitness_mirror_set_completeness.md`
    — the critic's independent-grep completeness lesson names the current
    range; plus 19. its critic `MEMORY.md` index line (retitled RANGE-FREE
    at #941 after the titled range went stale through two bumps).
20. `.claude/agent-memory/experiment-implementer/feedback_pinned_artifact_pair_mutual_inconsistency.md`
    — the #601 pair-COVERAGE sibling lesson cross-links checklist item (j)
    (#941).

`.claude/rules/vectorize-many-cell-fits.md` keeps a trailing
"**Sibling check:**" back-pointer to checklist item (i) — range-free, touched
only when a change affects item (i) itself.

#871 added item (i): throughput fitness of reused fit/analysis/eval CODE —
inner per-cell/per-fold/per-draw loop batched + device parametrized
(+ scoped Hub-API calls since #972) — scoped
to code reuse (N/A for data-only reuse), plan-time-only, with a
SOURCE-MODULE-fix remedy (never retrain, never a caller-side workaround).

#941 added item (j): pairwise provenance coherence of mutually-dependent
reused artifact PAIRS (bank vs capture, mix vs adapter, pool vs judge
outputs; incident #922) — a DATA-side item that routes through the DEFAULT
"failing check other than (i) → regenerate" remedy branch, so NO remedy line
was edited anywhere; the pair-specific remedies (re-capture under the current
input, or pin the input at the pre-regeneration revision) live inside the (j)
text itself, and a documented remedy-(2) pin PASSES the revision-scoped
probe. #941 also shipped the gotchas.md sibling bullet and gave the
consistency-checker a one-sentence mechanical date-ordering probe (the #734
(h) precedent).

#972 extended item (i) with a THIRD leg: (3) scoped Hub-API calls — data-repo
verify / staging / existence-probe calls prefix-scoped
(`list_repo_tree(path_in_repo=)` / `file_exists`) with a bounded
first-page-429 retry (gotchas.md #833; incident #810 — a reused unscoped
verify wedged a live A100 in 429 storms). The remedy phrase on every surface
now reads "batch / parametrize / scope it there" (artifact-reuse.md ×2,
planner.md step 5, CLAUDE.md), and leg (3) is ALSO mirrored where the legs
are enumerated: critic-lens-reference.md item 9 (both spots), CLAUDE.md's
bullet clause, vectorize-many-cell-fits.md's sibling-check parenthetical,
the two PLANNERLESS-path pre-launch statements (CLAUDE.md
inline-free-analysis carve-out + `.claude/skills/issue/SKILL.md` Step 9a-ter
item (3)), methodology-baselines-critic.md's failure-signature
parenthetical, and the review-side NEW-code twin — code-reviewer.md
Step 0.68's Hub-call-scoping sub-check + codex-code-reviewer.md's copy-list
bullet. #972 also widened item (i)'s TRIGGER CLASS on every class-gating
surface: the class words now read "fit / analysis / eval /
upload-verify-staging helpers" (artifact-reuse.md heading + class sentence,
planner.md step 5 + §10 escape, planner-section-reference.md §10 sentence +
escape, critic-lens-reference.md :153/:194, CLAUDE.md bullet,
methodology-baselines-critic.md) so a pure upload/verify/staging-helper
reuse can no longer self-classify out of item (i).

#1366 added item (k): parent-lineage coherence, TWO legs in one letter —
leg A (code-scoped, like (i)): diff the reused main-resident parent module
against the parent's unmerged branch (`git log --oneline
origin/main..origin/issue-<M> -- <module>`) and port or declare-not-needed
every unmerged commit; leg B (data-scoped, like (j)): reconcile the realized
artifact's row count against its declared input corpus — an unexplained
shortfall means an unported filter (incident #1345: the parent's crash-fix
filter lived only on unmerged `issue-825`; realized n=4724 vs corpus 5000 was
the tell). Unlike (j), (k) routes a NON-default remedy (port-then-reuse, NOT
regenerate), so the remedy-split lines WERE edited this time — the first
remedy-line edit since (i)/#871 (artifact-reuse.md, CLAUDE.md bullet,
planner.md step 5). The consistency-checker gained TWO one-sentence
mechanical probes (leg A on the #595 "Reused code module reachable on main"
row; leg B on the "Cited HF reuse artifacts resolve on the Hub" row); the
gotchas.md sibling bullet was added (range-free, the #941 precedent); the c6
declaration regex GRANDFATHERS the old token via `\([jk]\)` (in-flight plans
citing "(a)-(j)" still declare — two grandfather test pins kept); and the
upper-boundary test's decoy letter moved from (k) to (l).

Every remedy-split line on the live surfaces is deliberately worded WITHOUT a
lettered RANGE ("a failing check other than (i)/(h)(iv)/(k)" names letters,
never a range token) so the stale-range completeness grep stays clean; a
future letter needs no remedy re-edit ONLY when it routes through the default
retrain/regenerate branch ((j) did; (k) did not — see the #1366 paragraph).

A change targeting only planner.md leaves the independent enforcement passes
(critic, consistency-checker), the canonical rule file, and the mechanical
verify_plan.py heuristic checking the old contract. Grep
`'\(a\)\s*[-–—]\s*\([a-z]\)'` over `CLAUDE.md .claude/ scripts/ tests/`
(excluding worktrees / __pycache__ / cache) and check `git show --stat` on
prior `workflow-fix: ... fitness ...` commits to confirm the live mirror set
before editing — the documented list can go stale (it enumerated 15 sites and
omitted methodology-baselines-critic.md, lens-coverage-map.md, and the
adversarial-planner-v2 SKILL.md until #941 re-synced it).
