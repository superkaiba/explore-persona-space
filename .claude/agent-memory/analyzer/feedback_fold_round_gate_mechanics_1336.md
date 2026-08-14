---
name: fold-round-gate-mechanics-1336
description: Four mechanical gate traps hit while folding #1336 round 4 — alt-text counts toward check 20, letter-label enumerations FAIL the audit, verbatim-prompt details blocks need a "N example" disclosure token, and run-resolved concerns take address-concern (a deferred marker without a ledger event is flagged FABRICATED)
metadata:
  type: feedback
---

Four mechanical fixes for clean-result fold rounds, all hit on #1336 r4 (2026-08-13):

1. **Figure alt text counts toward check 20's per-`###` word cap** (what-is-plotted + image line + prose). A 30-word alt text alone can push a section past the 180 FAIL. Keep alt text ~8-12 words; put detail in the blockquote caption (counted separately, 60-word cap).
2. **`letter_labels` audit branch FAILs "(a) the" / "(b) the" / "(i) the" enumerations** anywhere in the body, Methodology included. Rewrite as "First, ... Second, ..." or semicolon lists.
3. **Check 10 (cherry-picked disclosure) fires on a verbatim ORIGIN-PROMPT `<details>` block in `**Context:**`** — it treats every details block as a sample block. Satisfy with a `\d+ example` token in the prelude: "The block below is 1 example of run provenance, not a sample of model outputs — ... preserved verbatim and in full". Never rewrite the verbatim prompt itself; math notation (`$R^2$`) inside stays audit-safe because the audit strips `<details>` fences.
4. **Lens 14 (verify `--issue` only): a `<!-- concern-deferred: <id> -->` marker with no `deferred` event in concerns.jsonl is flagged FABRICATED.** For concerns the realized run RESOLVED (not deferred), the route is `task.py address-concern <N> --concern-id <id> --by analyzer --round <k> --summary '<=200 chars>'` (longer → `--summary-file`), then keep a plain factual prose note in Methodology with no HTML marker.

**Why:** each cost one bounce-fix cycle inside the fold round; the verifier reports them only after drafting.

**How to apply:** on any fold/revision round, draft with short alt texts and no letter enumerations up front; check concerns.jsonl BEFORE choosing deferred-marker vs address-concern. Related: [[lens14-open-concern-ack-on-revision-rounds]], [[revision-word-caps-and-prereg-token]].

Also confirmed this round: `set-body --snapshot` unconditionally OVERWRITES an existing `original-body.md` (`shutil.copy2`) — on re-folds always omit `--snapshot`, even when a brief's boilerplate says otherwise.
