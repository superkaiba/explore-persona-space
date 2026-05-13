---
name: Check genealogy for evolutionary / search experiments
description: For any clean-result on a genetic / evolutionary / beam-search-style experiment, trace the top-K candidates back to gen-0 ancestors BEFORE writing confidence; single-root genealogy means the "search" is really a neighborhood scan
type: feedback
---

For any clean-result on an evolutionary search, beam search, genetic algorithm, or any other iterative-refinement experiment with multiple seed candidates: BEFORE assigning HIGH confidence to a "search-discovered" claim, trace the top-K final candidates back to their gen-0 (initial seed) ancestors and count distinct lineages.

**Why:** It is common for evolutionary/search loops to converge on a single productive seed's neighborhood — the plan's `diversity_min_lineages=N` constraint often fails in practice because second/third-best lineages get out-competed within a few rounds. If all 10 top candidates trace to one gen-0 seed, the "search" is really a neighborhood scan, NOT a multi-basin climb. The framing matters: "search discovered a new trigger neighborhood" overclaims when the seed pool already contained the productive candidate.

In issue #331, all 10 top obscure-only candidates descended from `apis papyrus est` (a Phase-0-seeded candidate at 18.75%); the mutation operator essentially fixed `papyrus est` and explored first-word substitutions. This dropped overall confidence from HIGH to MODERATE.

**How to apply:**
1. Load the genealogy / ancestry JSON (or reconstruct from per-round outputs).
2. For each top-K candidate, trace `parent_phrase` chain back to round 0 (handle multi-parent operators like `llm_crossover` by collecting all ancestors).
3. Count distinct gen-0 roots feeding the top-K.
4. If `n_roots == 1`: drop confidence one level (HIGH → MODERATE, MODERATE → LOW), add an explicit caveat in the confidence line + standing caveats section, and reframe the claim from "search discovered X" to "search characterized the neighborhood of one productive seed". The plan's diversity constraint failing in practice should also be called out.
5. If `n_roots ≥ 3`: the diversity constraint worked; confidence stays at whatever the data otherwise supports.

Implementation snippet (Python, using `parent_phrase` and optional `mutation_detail` for crossover parents):

```python
def trace_root(phrase, by_phrase):
    seen = set(); queue = [phrase]; roots = set()
    while queue:
        p = queue.pop()
        if p in seen or p not in by_phrase: continue
        seen.add(p)
        e = by_phrase[p]
        if e['round_idx'] == 0:
            roots.add(p); continue
        if 'from [' in (e.get('mutation_detail') or ''):
            # multi-parent crossover
            inner = e['mutation_detail'].split('from [', 1)[1].rstrip(']')
            for parent in [x.strip().strip("'\"") for x in inner.split(',')]:
                if parent: queue.append(parent)
        else:
            parent = e.get('parent_phrase')
            if parent: queue.append(parent)
            else: roots.add(p)
    return roots
```
