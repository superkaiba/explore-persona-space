# Meeting notes — Christina, June 11, 2026

Raw notes, captured same day. Not yet triaged into tasks — see the todo to
process these later.

## Leakage / implantation control

- Run a control to test whether higher *average* leakage is just due to *more
  implantation* (stronger implant overall, not less selectivity).

## The prior question (she's interested in this)

- How much is the prior captured in existing persona vectors?
- How much does the persona vector capture trained-in behaviors?

## Interpretation of the contrastive-negatives results

- Giving the model a sharper background relief for the source — the contrast
  makes the behavior more *distinctive*, not just more localized.

## LoRA-vector geometry ideas

- Take the learned LoRA vector → measure how much the conditional vector dots
  with the original persona vector.
- Measure how much the A and B matrices separately dot with the context vector
  and the persona vector.
- Question: can the LoRA factorize so that A selects for the persona and B
  selects for the attribute?
- Simpler probe: just train a rank-1 LoRA and inspect it directly.

## Experiment proposal: good vs evil assistant robustness

- Train the model to be "assistant" vs "evil assistant".
- Idea (verbatim): "can you give a place for all evil" — i.e. give evil a
  dedicated container persona.
- Compare the evil-assistant vector vs the good-assistant vector.
- Test: does carving out an evil-assistant persona make the *good* assistant
  more robust?
