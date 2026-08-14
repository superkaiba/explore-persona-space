---
name: marker-prose-is-summary-not-ledger
description: Fold rounds — recompute every rate/range from the committed ledger JSON; wave/gate marker notes are experimenter summaries and can misstate their own artifacts (#2054 regen fold).
metadata:
  type: feedback
---

On fold/abort rounds, treat `epm:progress` / gate-verdict marker notes as experimenter SUMMARIES, never as artifacts — even when the fold brief explicitly lists the markers among the ground-truth sources.

**Why:** #2054 `coordinated-common-set-regen` fold (2026-08-11): the gate-verdict marker's diagnostics table said verbatim-filter drop "65-69%, stable across waves" and per-attempt success "~9-13%"; the committed `wave_state_final.json` ledger gives 68.2-77.0% RISING and 12.0% → 9.4% → 7.7% FALLING. Copying the marker's ranges into Takeaways + the result section drew a clean-result-critic BLOCKER (round-1 REVISE). The marker even hid a trend reversal ("stable" vs rising).

**How to apply:** whenever a body claim is a rate, range, or trend over a run's waves/cells, derive it with a one-off `jq`/python pass over the committed JSON (e.g. `question_not_verbatim/generator_kept`, `n_admitted_new/requested` per wave) in the same turn as drafting — quote marker prose only for facts with no committed artifact (and say so). This is the fold-round instance of "pull every number from the raw JSON, never the experimenter's summary": marker notes ARE the experimenter's summary.
