---
name: codex-env-override-poisoning-chain-untraced-leg
description: Codex Major claiming a single env-override "poisons production licensing" — trace EVERY input the licensing gate recomputes; a leg resolving outside the override's blast radius breaks the chain (fails loud) and demotes to hardening CONCERN
metadata:
  type: feedback
---

Before crediting a Codex blocker of the shape "overriding env var X lets a
synthetic/smoke artifact license production spend", trace EVERY input the
licensing gate RECOMPUTES at production defaults — not just the ones the
override clobbers. If even one leg resolves OUTSIDE the override's blast
radius, the identity coincidence Codex asserts does not hold: the gate
mismatches or raises loud, and the finding demotes from verdict-blocking to
a hardening CONCERN (containment-reject + explicit flag rejection), persisted
via defer-concern --by reconciler + re-raise at CONCERN.

**Why:** #2479 r4 — Codex Major `smoke-root-production-poisoning` claimed
`EPM_I2479_SMOKE_ROOT=eval_results/issue_2479` overwrites panel/manifest and
"those identities then coincide and the synthetic PASS can license
production". Verified TRUE: uncontained root, no `smoke_synthesized` check in
`require_pilot_pass`, committed panel/manifest clobberable, synthetic pilot
lands at the production `PILOT_AXIS_REL`. REFUTED: the axis identity's items
leg resolves from `DEFAULT_ITEMS_DIR_REL = data/issue_2479/axis_items`
(judge_pilots.py:99,186) — NOT under the smoke root — so the expected-identity
recompute raises loud at `build_axis_arm`'s `assert present` (jp:383) or
mismatches `items_content_sha256`; full poisoning needed a SECOND simultaneous
override (`EPM_I2479_AXIS_ITEMS_DIR`). Claude's g2 had the inverse blind spot:
it verified only the DEFAULT path and never considered the override at all —
both twins missed the middle ground (real clobber residual, refuted licensing
escalation).

**How to apply:** on any smoke/scratch-root override finding, (1) list every
field the gate's expected-identity recompute reads and the RESOLUTION PATH of
each (env default vs override-derived); (2) simulate the override: which
paths coincide, which diverge, which raise; (3) chain works end-to-end →
Real-blocking; chain breaks loud → Real-nonblocking CONCERN carrying the
containment fix. Related: [[codex-hardening-beyond-minimal-port-contract]].
