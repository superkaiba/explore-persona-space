---
name: thin-fork-commit-review-recipe
description: Reviewing a "thin fork of <parent script>" commit — diff the fork blob against the live parent file to collapse the review to the delta, then live-probe every cross-module seam
metadata:
  type: feedback
---

Rule: when a commit adds a whole file DECLARED as a thin fork of an existing script, do not read the 1,000+ new lines top-to-bottom — `git show <sha>:<newfile> > /tmp/fork.py; diff <parent> /tmp/fork.py` collapses it to the true delta (200 lines for a 1,044-line file, #2329 r1 g2), which you read IN FULL; the inherited remainder gets a targeted scan only for the binding disciplines (judge pin / dispatch routing / gate wiring / drop-counting).

**Why:** the delta is where fork bugs live (root/pin swaps, new legs); the parent body was reviewed in its own round. Reading the fork whole burns budget and buries the 5 changed branches.

**How to apply:** after the delta read, the seams are the remaining risk and each settles by a LIVE probe, not by reading: (1) enumerate `grep -oE 'ALIAS\.[A-Za-z_]+'` consumed attributes and hasattr-probe them against the imported modules in one `uv run python` block (also dataclass fields/properties the fork's config consumes — a subclass property like `pools_path` lives on the FORK, check there before flagging); (2) any constant deliberately DUPLICATED across files (e.g. an HF revision pin kept local to avoid a torch import) gets a value-equality grep both sides + a check whether any test asserts equality (flag if not — drift silently splits staging pin from manifest enrichment); (3) any `manifest.get("<key>", fallback)` fail-safe read gets a producer-side grep for the exact key (a name mismatch silently arms the fallback forever); (4) ruff the blob IN SITU, not the /tmp copy — `per-file-ignores` for `scripts/*` (I001, B023, …) makes isolated-blob hits false positives. Related: [[dotenv_ordering_fix_review_recipe]], [[smoke_fixture_authored_with_consumer_keys]].

Also from this round: an orchestrator brief's plan PATH can be stale (worktree cut before later plan versions landed on main) — before grounding any finding in plan numbers, confirm the plan version against the MAIN checkout's `plans/` dir, and state in the verdict which version each plan-grounded finding was checked against.
