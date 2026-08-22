---
name: smoke-arch-marker-2176-grammar-pitfalls
description: "Step 0.55: run check-smoke-arch-registry FIRST — prose-decorated per-arm heading parses to EMPTY sub-block; row keys must equal members= tokens; tuple registries abstain to marker-only; only the FIRST arm-registry line parses (multi-driver = one union line, comma-listed file=)"
metadata:
  type: feedback
---

Rule: at Step 0.55, run `uv run python scripts/task.py check-smoke-arch-registry <N>
--repo-root <worktree>` BEFORE hand-judging the marker's shape, and know the three
#2176 grammar pitfalls a human read misses (all hit live on #2225 R1 g6):

1. `_MARKER_TOP_KEY_RE` (task_workflow.py:2244) matches ONLY the bare key at line
   start — `per-arm-resolution (any parenthetical):` never opens the span, so 10
   perfectly good REAL rows parse as `per_arm == {}` (clause-4 REFUSE). Prose is
   legal AFTER the colon, never before it.
2. `_PER_ARM_ROW_RE` captures everything up to the first `:` as the arm key —
   `- A (E1×all×L1): REAL` yields key `"A (E1×all×L1)"`, which fails clause-5
   set-membership against `members=A,...`. Rows must LEAD with the bare arm token.
3. Clause-5b driver recompute (`_extract_registry_members`) only reads module-level
   dict LITERALS with all-string keys; a tuple/list registry (e.g. a
   `CONFIGS: tuple[ConfigSpec, ...]`) makes it abstain → checker OK line reads
   `marker-only` and the REVIEWER owns members↔registry set-equality (the fallback
   arm) — enumerate the symbol's names yourself, never a presence grep.
4. `arm-registry:` itself is LINE-ANCHORED single-line: a bare `arm-registry:`
   heading with the derivation in a FOLLOWING bullet (however correct the prose)
   is REFUSE "no line-anchored arm-registry: line found" (#2330 R1 g7 — a
   mechanically-AST-derived 10-arm set in a bullet still failed). A NO-registry
   driver (main() calls every arm unconditionally, no phase arg) uses the
   `arm-registry: N/A — <reason>` form, or a single-line `source=... file=...
   n=... members=...` naming the derived set.
5. MULTI-DRIVER rounds: `parse_arm_registry_line` consumes only the FIRST
   `arm-registry:` line — posting one line per driver leaves the rest invisible,
   and namespacing row keys (`run.envcheck`, `judge.pilot`) fails clause-5
   byte-wise matching for EVERY member (#2333 R1 g2: n_registry=7 vs
   n_enumerated=20, all 7 "missing" while 20 REAL rows sat right there). The
   conforming shapes: ONE union line — `file=` takes a COMMA-LIST and clause-5b
   unions per-file extractions (`source=sorted(PHASES)
   file=run.py,judge.py,analysis.py n=<union> members=<sorted union>`) with
   bare-token rows for every union member — or ONE primary-driver line with its
   rows bare-keyed; other drivers' dotted rows survive as allowed EXTRAS.
   `members=` must be SORTED (a definition-order tuple listing fails eyeballs
   even when n matches).

**Why:** the #2225 R1 marker had all substance right (10 arms, verdict-consistent
rows) but failed the checker on form alone; without the checker + regex read the
fix recipe would have thrashed twice (arm-registry line fixed, then clause-4, then
clause-5). Step 6d.0 runs this checker POST-provision, so catching it at review is
the whole point of gate 0.55. Re-hit #2379 R1 g6 (2026-08-19): a COMMAND-PIPELINE
derivation (`arm-registry: bash pod.sh | grep -oE ... -> p0 p1 ...`) is also
REFUSE-malformed (neither accepted form), parenthetical-decorated row keys
(`p0 (pod smoke): REAL`) re-appeared, and the hand-enumeration omitted two sibling
drivers' argparse `choices=` dispatch tables (mapfit 6, judge 8 — the #2163
omission shape). The one-post fix recipe held unchanged.

**How to apply:** any Step 0.55 audit (round-level / CONTRACT-BEARING split-review
group). Give the implementer the full one-post fix: conforming `arm-registry:` line
+ bare `per-arm-resolution:` heading + bare-token row keys, in one re-post.
