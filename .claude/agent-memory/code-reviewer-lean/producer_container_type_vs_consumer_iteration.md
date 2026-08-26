---
name: producer-container-type-vs-consumer-iteration
description: In pre-split multi-unit builds, live-probe each consumer seam with the PRODUCER's actual return shape (dict-vs-list iteration TypeError passes every signature/kwarg/import pin); diff sibling consumers' access idioms as the cheap tell
metadata:
  type: feedback
---

Rule: when a round's driver consumes a sibling-unit producer's composite return (a bank/manifest
dict holding a `contexts`/`rows` container), verify the CONTAINER TYPE at the seam by (1) reading
the producer's return expression (`build_bank` returned `"contexts": dict[str, dict]`), (2) a
5-line live probe feeding a producer-shaped value through the consumer function, and (3) diffing
SIBLING consumers' access idioms — one iterating `for c in contexts` (list idiom) while another
does `contexts[cid]["cell"]` (dict idiom) means at least one is wrong.

**Why:** #2564 r1 g2: `bank2564.build_bank` returns `contexts` as `dict[str, dict]`; the unit-2
driver's `_filter_bank` + both phase entries iterated it as `list[dict]` → `TypeError: string
indices must be integers` on EVERY run, production no-filter path included. All 11 CPU pins
passed because the consumer's own test authored a LIST-shaped fixture (the
[[smoke-fixture-authored-with-consumer-keys]] class, container-type variant); `--import-check`,
the argcheck bind pass, AND the MF-A kwarg-signature assertion all passed — none of those
instruments sees a container-shape mismatch. The sibling analysis unit consumed the dict shape
correctly, which both localized the fix (driver-seam `list(bank["contexts"].values())`, producer
untouched) and proved the sibling-idiom diff is a one-grep detector.

**How to apply:** fires on any split-review/multi-unit round where units land producer and
consumer in different commits. Grep every `bank[...]` / composite-return access across the round's
files, classify each as dict-idiom or list-idiom, and run the live probe when they disagree or
when the producer's return annotation says `dict` while a consumer list-comprehends it. Demand an
integration pin that feeds the REAL producer shape (not a hand-authored fixture) through the
consumer seam.
