---
name: Claude misses cross-file consumer regex / character-class boundary
description: When a planned change introduces a new literal (two-letter cond-ID prefix, negative-layer index, multi-char suffix) and a SEPARATE file's regex defines the character class that decides acceptance, Claude ticks the new literal `✓ implemented` based on the producing module's in-module assert without grepping the consumer-side regex.
type: feedback
---

When a plan §X.Y.Z names a new literal-class change ("FB1..FB9 / SC1..SC24 cond IDs match the existing `[A-Z]\d+` regex", "layer-1 next_token_js baseline at last_prompt", multi-token marker added to an existing single-token enumerator), Claude code-reviewer's plan-adherence walk-down ticks `§X.Y.Z ✓ implemented` by:

1. Grepping the PRODUCING file (e.g. `i509_fact_conditions.py:120: assert set(CONDITIONS_BY_ID.keys()) == {f"FB{i}" for i in range(1, 10)}`)
2. Citing the in-module assert as evidence
3. Stopping

Claude does NOT grep the CONSUMING file's regex / character class / parser. That consumer was authored for the previous literal-class (single-letter prefix, non-negative layer) and silently rejects the new one. The producing-side assert passes; the consuming-side regex returns no match; the partition merge step raises `"No partitioned per-cond files AT ALL"` or a similar fail-loud after the producer ships.

**Why:** Plan-adherence walk-downs feel "verified" when the assert text matches the plan text. The character-class boundary `[A-Z]\d+` vs `[A-Z]+\d+` is invisible from the producer side.

**How to apply:** When the implementer's report claims "the new cond/layer/marker conforms to the existing regex `<pattern>`", `rg` the literal `<pattern>` across the consuming module(s) AND test the regex empirically against one new literal AND one old literal before believing Claude's `✓ implemented`. Five-line uv-python harness:

```python
import re
p = re.compile(r"<the regex Claude says accepts the new literal>")
for name in ["<one filename with new literal>", "<one filename with old literal>"]:
    m = p.match(name)
    print(f"{name}: {'MATCH cid=' + m.group('cid') if m else 'REJECT'}")
```

Origin: task #509 round-1. The merge regex `(?P<cid>[A-Z]\d+)` at `scripts/issue493_extraction_metric_bakeoff.py:1814` was UNCHANGED from i406's single-letter cond namespace (M1, A1, etc). Issue #509 introduced `FB1..FB9` + `SC1..SC24` (two-letter prefix). Claude ticked the conformance check `✓` based on the in-module assert; Codex caught that `condFB1.pt` REJECTS the regex and `condA1.pt` MATCHES. Production extraction at L19-L24 × {gauss_kl, mmd, wass2} would crash at the FIRST merge before any metric JSON lands. Companion to "Claude misses same-file siblings", "Claude misses dispatcher-wiring correctness bugs", "Claude misses cross-branch Python module dep". Same disease — Claude verifies one side of a contract and PASSes on structural presence; Codex re-greps the other side.

Related cross-file gotcha caught in the same review: regex `layer(?P<layer>\d+)` in `scripts/issue509_scoring.py:81-82` disallows `layer-1` AND matches `*__perm.json` MMD sidecars as fake predictor cells. Plan §metrics row called for `next_token_js` (lives at `layer-1`). Claude saw the four FE/reliability/permutation math findings on the SAME script's downstream lines but missed the upstream metric-enumerator that decides which files reach the scoring loop. Pattern: when Claude reads a script line-by-line top-down, it commits to interpreting later math against the enumerator's CURRENT output and never re-checks whether the enumerator is right.
