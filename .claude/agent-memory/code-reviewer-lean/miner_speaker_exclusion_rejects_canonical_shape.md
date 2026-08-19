---
name: miner-speaker-exclusion-rejects-canonical-shape
description: Probe any structural miner/extraction heuristic with the EXACT target-shaped text the primes/openers teach — a name+verb speaker-exclusion evaluated on the AFTER-close side rejects the canonical '"Q?" Name replied: "A"' pattern (#2378 R1 g1)
metadata:
  type: feedback
---

When a diff ships a regex/heuristic MINER over generated text (quote-span +
attribution, span extraction, directedness), do not review it by reading the
rules — RUN it on the canonical target shape the few-shot primes and opener
banks teach the model. #2378 R1 g1: `_is_directed` rule 1 treated
`\b{name}\s+(VERBS)\b` anywhere within ±120 chars of the quote as
"character is the speaker → reject", but the taught pattern puts
`{Name} replied: "` immediately AFTER the mined question's closing quote
(introducing the NEXT quote — positive directedness evidence). Live probe:
all three canonical shapes returned `not_directed`; yield ≈ 0 on well-formed
generations, and the failure masquerades as a model/recipe problem at the
pilot gate, not a code bug.

**Why:** attribution syntax is directional — `X said: "…"` introduces the
following quote; `"…" X said.` attributes the preceding one. A side-blind
name+verb match inverts the load-bearing case, and every accept rule is
unreachable behind the early `return False`.

**How to apply:** for each miner keep/reject rule, build 2-4 fixture strings
from the diff's OWN prime/opener banks (the distribution the model will
actually emit) and call the real mining function in a 30-second probe; every
canonical positive must keep and a speaker-authored quote must reject. An
early hard-reject rule that fires before accept rules deserves the probe
first. Related: [[smoke_fixture_authored_with_consumer_keys]].
