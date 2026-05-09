---
name: Compute marker-position distribution when marker eval flags coupling-leakage
description: For any [TOKEN]-marker-leakage clean-result, compute the start-vs-tail position distribution alongside the rate. The position is often qualitatively different between source and bystander firings — and reframes the headline.
type: feedback
---

For any clean-result body where the headline number is "marker leaks at X% on bystanders", ALSO compute the relative position of the marker in firing completions (start <5% / early 5-50% / mid 50-85% / tail >85%). The position distribution is often qualitatively different between source and bystander firings AND reframes "X% leakage" from "the model loses persona discipline X% of the time" to "the marker tail-token-leaks X% of the time after persona-faithful generation" (or the inverse — full-persona-takeover from the start).

**Why:** Issue #247 round 2 — the round-1 critic flagged the v1 framing of "often at start or end" as a missed headline. The empirical truth was 97.6% start (source) vs 92.9% tail (bystander). The v1 interpretation called the bystander leakage "loss of persona discipline" but the position evidence makes clear it's "tail-token drift after a faithful answer", which is a qualitatively different failure mode and changes what mitigation makes sense.

**How to apply:** During Step 2/3 (compute statistics, generate plots), if the eval is a marker-substring-rate eval, ALSO compute:

```python
for cell in cells:
    data = json.load(open(f"{cell}/raw_completions.json"))
    for persona, qd in data.items():
        for q, comps in qd.items():
            for c in comps:
                if marker not in c:
                    continue
                pos = c.find(marker) / max(1, len(c))
                # bucket: <5%=start, 5-50%=early, 50-85%=mid, >85%=tail
```

Aggregate across cells. Compare source-firing distribution vs bystander-firing distribution. If the asymmetry is sharp (e.g., one population is >80% in one bucket and the other is >80% in a different bucket), make the position distribution a load-bearing claim in Result 1's main takeaways and add a per-population bar chart as a supporting figure (or the hero, if the position story is sharper than the rate story).
