---
name: producer-smoke-tree-replay-for-consumer-schema
description: "Reviewing a new consumer (figs/report driver) of a producer's JSON: replay the producer's OWN smoke tree through the consumer + key-literal grep of the writer dicts; also check mandatory-leg-behind-unrelated-leg sequencing in multi-leg upload mains (#2474 r1 g2)"
metadata:
  type: feedback
---

Two cheap probes settle a producer→consumer schema-drift review (a new figs/report
driver reading a fit driver's JSON outputs) better than reading alone:

1. **Key-literal cross-check** — grep the producer's writer for the exact dict-key
   literals (`"pinned"`, `"per_condition"`, `"vs_postft"`, percentile names, the
   `round1_recompute` key format string) and diff each against every consumer
   read-path. A consumer that `import`s the producer module for its fam-name
   constants and loader helpers (`fit_mod.GEOMETRY_FAMS`, `fit_mod._load_rates`)
   structurally suppresses name drift — credit that design, then check only the
   residual literal keys.
2. **Producer smoke-tree replay** — when the producer has a `--phase smoke` that
   writes real-schema synthetic outputs, run the NEW consumer against that tree
   (`--smoke`) and count outputs. One rc=0 full-registry render empirically
   retires the whole drift question (10/10 slugs in #2474 g2), and the rendered
   PNGs double as the figure-sanity check. Cross-module `_load_rates({"rates_path":
   ...}, kind)` calls: verify the minimal dict covers every cfg key the callee
   reads on the exercised branch.

**Why:** schema drift between producer and consumer was the round's named Major
class; the replay probe found none in minutes and produced visual evidence.

**How to apply:** any round adding a consumer of a sibling script's JSON artifacts.
Companion finding shape from the same round: in a multi-leg upload `main()`, a
POLICY-MANDATORY leg (rollout text — never droppable) sequenced AFTER an
unconditional unrelated leg (`upload_tensors` raises on a reaped tensor tree
before the rawcomp leg runs) is a Minor concern — fail-loud so not a blocker, but
flag the missing `--<leg>-only` escape. Related: [[smoke-fixture-authored-with-consumer-keys]],
[[registered-gate-quantity-substituted]].
