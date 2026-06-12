---
name: Compute marker-position distribution when marker eval flags coupling-leakage
description: For any marker-leakage clean-result, compute the start-vs-tail position distribution alongside the rate — source vs bystander firings often differ qualitatively and reframe the headline
type: feedback
---

For any "marker leaks at X% on bystanders" headline, ALSO compute the marker's relative position in firing completions (start <5% / early 5-50% / mid 50-85% / tail >85%, via `c.find(marker)/len(c)` over raw_completions.json). The distribution often differs sharply between source and bystander firings and reframes "X% leakage" from "loses persona discipline X% of the time" to "tail-token drift after a persona-faithful answer" (or the inverse, takeover-from-the-start).

**Why:** issue #247 round 2 — the critic flagged v1's "often at start or end" as a missed headline. Empirically 97.6% start (source) vs 92.9% tail (bystander): bystander leakage was tail-token drift, a qualitatively different failure mode with different mitigations.

**How to apply:** during the compute-statistics/plots step of any marker-substring-rate eval, bucket positions per firing, aggregate per population, and compare source vs bystander. If the asymmetry is sharp (>80% of each population in different buckets), make the position distribution a load-bearing claim in the relevant finding and add a per-population bar chart (hero, if the position story beats the rate story).
