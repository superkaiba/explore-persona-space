---
name: Comedian-side vs paramedic-side direction split in joint-vs-single leakage comparison
description: When joint training on two far-apart sources, bystanders on the B-source side all gained leakage vs A-only, while most A-source-side bystanders lost leakage — a clean direction split the body did not report
type: feedback
---

In issue #311: all 4 comedian-side bystanders (t < 0) gained firing rate under joint vs A-only (villain +0.117, poet +0.025, french_person +0.037, kindergarten_teacher +0.022). Among 13 paramedic-side bystanders (t > 0), only 4 gained and 8 lost, with large losses for paramedic-similar personas (navy_seal −0.130, police_officer −0.120). This is the mechanism behind the positive ρ but was not described in the body.

**Why:** The body focused on the absence of midpoint elevation (a pre-registered negative result) and did not examine the signed direction of per-bystander changes. The direction split is a real pattern that deserves one sentence in Result 2.

**How to apply:** In lens 2 (Surprising Unmentioned Patterns), when a joint-source experiment reports bystander rates, split bystanders by t-value sign (which source side they are closer to) and check whether the direction of (joint − single-source) difference follows a sign pattern. If all B-side bystanders gained and A-side bystanders were mixed, this is a substantive unreported pattern.
