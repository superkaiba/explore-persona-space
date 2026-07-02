---
name: Claude code-review walk-down silently omits a registered-output grouping/parser function it never inspected
description: Claude PASSes after a 15-contract walk-down but the walk-down has NO entry for a string-parser that maps registered outputs into groups; a deterministic misparse there mislabels EVERY member of a registered deliverable (figure grouping, leaderboard, attribution label) while the headline verdicts survive. Enumerate every name-parser/grouping function the registered outputs flow through.
type: feedback
---

**Rule:** Claude's contract walk-down catches what it LISTS; its blind spot is the
function it never put on the list. A deterministic string-parser that maps
registered outputs into families/groups (`_metric_family`, condition-name parsers,
suffix-classifiers) is a high-value omission target — if the parser misparses, it
mislabels EVERY member of a registered grouping, not an edge case, yet the
headline pass/fail verdicts often survive because they are computed group-agnostic
(argmax over all items). So a green contract walk-down + "all 15 verified" can sit
on top of a deterministic misclassifier that corrupts a registered figure /
leaderboard / attribution label. The reconciler must independently enumerate every
grouping/parser function the registered outputs flow through and RUN it on the
ACTUAL generated names.

**Why:** #545 r1 — Claude PASSed a 15-contract walk-down (incl. contract 10
"n_cells reported", contract 12 capacity-confound attribution). Codex's Major #2
flagged `_metric_family` (scoring.py:597) doing `body.rsplit("_", 1)[-1]`. Running
it on the actual generated predictor names (predictors_zoo.py:543/587):
`...raw_mahal_pooled_ctx`→`other_A` (intended `covariance_centroid`),
`...gauss_kl`→`other_A` (intended `cloud`), `...raw_neg_l2`→`other_A`
(intended `raw_centroid`), `...centered_neg_l2`→`other_A` (intended
`centered_centroid`). It collapsed every underscore-metric predictor into
`other_A`, corrupting the registered §7 secondary leaderboard + the HERO 1 figure
("grouped by metric family") + the centered-vs-raw delta + the
`global_champion_metric_family` label its OWN contract-12 Minor read from. The
headline H1/H2/H3 transfer verdicts survived (family-agnostic argmax), so Claude's
walk-down saw nothing wrong — it never had `_metric_family` on the list at all.
That single omission overturned the PASS → FAIL.

**How to apply:** When Claude PASSes via a contract/feature walk-down and a registered
output has a "family"/"group"/"metric_family"/"condition" field or a grouped
figure: (1) find the function that POPULATES that field (grep the field name to its
assignment, then the parser it calls); (2) reproduce the parser on the LITERAL
generated names (grep the f-string that builds them, e.g.
`f"cloud_{flavor}_L{layer}_{point}_{centering}_{metric}"`), not on hand-typed
examples; (3) a `rsplit("_", 1)` / single-token-suffix parse over names whose
metrics CONTAIN underscores (`mahal_pooled_ctx`, `gauss_kl`, `neg_l2`) is the
canonical smell. A deterministic mislabel of every member of a registered grouping
is BLOCKING even when the headline verdict is group-agnostic — it corrupts the
figure + any attribution label that reads the group. Cheap fix: longest-suffix-first
match, or read the `metric`/`centering` field already in the item's metadata.

Companion: [[feedback_claude_fabricates_rf_walkdown_checkmark]] (Claude ticks a ✓ a
grep disproves — the FABRICATED-checkmark sibling; THIS memory is the OMITTED-from-the-
list sibling: the function was never walked at all). [[feedback_claude_misses_same_file_siblings]]
(walk-down misses a code path of the same class).
