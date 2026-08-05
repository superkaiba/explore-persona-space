---
name: Pilot-gated OOD rung designs
description: pilot n=200 gives SE(SD)≈SD/20 and SE(p)≈0.035 — adequate for a two-sided (SD≥10, <80% bin) spread gate without a Must-Fix
type: feedback
---

Pilot-gated OOD-corpus designs that carry an SD+bin spread floor: n=200 per corpus is adequate.

**Why:** SE(SD) ≈ SD/√(2N−2) ≈ SD/20 at N=200; bin-fraction SE = √(p(1−p)/N) ≈ 0.035 at p≈0.5. The 0.80 floor has margin ~8σ, the SD≥10 floor has ~1σ margin — thin but not degenerate. Do NOT REVISE the pilot size for gate-margin reasons at this N.

**How to apply:** For a two-sided spread floor gate on an OOD rung, accept n=200 pilots. REVISE only if the plan uses n<100 or omits the SE calculation entirely. #1739 v15 evil-ood-spread round pilot at n=200 for MHJ/tom-gibbs/PAIR corpora — pilot sizing was sound.

**Sibling flag:** an attack-family-SHIFT OOD (item A) is a distinct claim from a same-corpus attack-type HOLDOUT (item B); reframing B as an OOD-transfer substitute when A entirely fails is analyzer-weighable overclaim territory (not a plan-time REVISE if success/kill criteria correctly distinguish the two, as #1739 v15's §7 does).
