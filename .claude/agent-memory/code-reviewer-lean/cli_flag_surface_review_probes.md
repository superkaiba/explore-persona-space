---
name: cli-flag-surface-review-probes
description: "Two CLI-flag review probes: (1) same-line add_argument greps miss multi-line calls — confirm a 'missing flag' by bare-string grep before flagging; (2) size an optional-flag fail-open by the round's REALIZED artifact field, not the parser surface"
metadata:
  type: feedback
---

Two probes from #2502 R1 g3 (fits/reliability split review):

1. **`grep add_argument | grep -o '"--flag"'` sees only SAME-LINE flag strings.**
   Ruff-formatted parsers put most flag strings on their own line inside a
   multi-line `add_argument(` call, so the composed-command audit ("does the
   sibling driver actually expose `--disable-thinking`?") false-reports missing
   flags. I nearly filed a phantom Critical on 4 flags that all existed.
   **How to apply:** before flagging a composed invocation as using nonexistent
   flags, grep the bare flag string (and its `args.<attr>` reads) file-wide;
   only a zero-hit on BOTH is a finding. An `--import-check` argcheck PASS on
   the composing driver does NOT cover cross-script flag composition.

2. **Optional-flag fail-open severity keys on the PRACTICED invocation.**
   A decide/gate phase whose Must-Fix input rides an optional CLI flag
   (default None, silent nulls in the artifact) is Major only if the round's
   registered/practiced invocation can plausibly omit it. Check the smoke /
   marker's REALIZED artifact for the field the flag feeds (here:
   `a_best_over_ceiling 0.389` non-null in the smoke decision.json proved the
   flags were passed) — realized-field evidence downgrades to Minor + a
   require-the-flag hardening suggestion. Visible nulls ≠ silently-wrong
   numbers. Related: [[consumer_flag_producer_never_writes]] (severity forks
   on an upstream assert), [[registered_gate_quantity_substituted]].

Context: split-review g3 of #2502 u3 (00cc635c) — verdict PASS with 4 Minors;
all MF-B/C/D/E/J verified by running both selfchecks + import-checks live.
