---
name: naive-substitution-probe-for-unpinned-defence
description: When a report justifies a defensive mechanism with a factual claim about the real tree, swap in the naive alternative and re-run the suite — if everything still passes, the defence is unpinned and the rationale is likely wrong.
metadata:
  type: feedback
---

When an implementation report says a mechanism is load-bearing ("quote-aware
by necessity, not by taste"), do not accept the rationale and do not settle
for reading the mechanism. Monkeypatch the NAIVE alternative in and re-run
the whole test file:

```python
import test_module as T
T._the_mechanism = lambda x: naive(x)
for name in sorted(n for n in dir(T) if n.startswith("test_")):
    try: getattr(T, name)()
    except Exception as e: print("FAILS:", name)
```

Three outcomes, all worth knowing:

- **Nothing fails** — the defence is unpinned. A future refactor deletes it
  and CI stays green. Rule 13 / Step 4.5 finding.
- **Something fails, but not the test the report names as its guard** — the
  report's attribution is wrong. Record which test actually pins it.
- **The named guard fails** — rationale confirmed.

**Why:** #2387 r3 justified a quote-aware bash-comment stripper with "every
watch push line carries a `#` inside its quoted message, so a naive
first-hash truncation would cut live sites short." Both halves were false:
2 of 4 watch push lines had no `#` at all, and in the two that did, the `#`
sat AFTER the regex match, so a naive strip changed nothing. The named
over-strip guard passed unchanged under the naive strip; the only
discriminating test was an unrelated second-push test. The code was fine and
more robust than claimed — but the next maintainer, told the wrong reason and
finding the named guard vacuous, could reasonably delete a correct defence.

**How to apply:** fires whenever a round adds a defensive mechanism (quote
awareness, escaping, normalization, a guard clause) justified by a claim
about real-tree content. Also verify the content claim directly — enumerate
the real call sites and check the claimed property per site, rather than
trusting "every X has Y".

**Companion:** for character-class / word-start completeness claims, ask the
real interpreter rather than reasoning. `bash -c` on each candidate lead-in
char settled that `(true)#…` IS a comment while `echo "A"#tail` is not — a
gap the report had described as failing in the safe direction when it fails
silently. Related: [[feedback_mutant_matrix_needs_negative_control]],
[[feedback_na_classification_both_legs]].
