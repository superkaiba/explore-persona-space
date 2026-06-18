---
name: Codex Critical blocker on sandbox-unreadable pod-side data artifact
description: Codex FAILs `cached-artifact-coverage-unverified` on a pod-side/auto-downloaded file its sandbox can't read; "unverified" is not "unverifiable" — trace producer exit-assertion, pinned-revision download, and parent-run empirical proof, then demote.
type: feedback
---

**Rule:** when Codex's Critical literally says "I could not read <path> in this worktree" about a runtime data dependency not in git, trace three links the sandbox hid:
1. **Producer-side coverage assertion** — open the artifact GENERATOR; if it exit-asserts exactly the coverage the consumer needs, the gap is structurally impossible.
2. **Pinned-revision download** — grep `hf_hub_download(revision=...)` / the data-deps helper; a pinned SHA freezes the bytes to what the parent run used.
3. **Parent-run empirical proof** — if the parent worker on main consumed the same artifact through the same builder as a HARD dependency and production succeeded, a child's DIAGNOSTIC-only consume is a strict weakening. Bonus: authoritative bytes built FROM the artifact prove coverage by construction.
Residual failure mode if all three fail: loud KeyError pre-training (infra-class, recoverable), not silent corruption — verify diagnostic vs authoritative consume before classifying.

**Origin:** #534 r1 — `cached-r-train-coverage` + `manifest-exclusion-not-enforced`, both demoted; PASS. Companions: [[feedback_codex_passes_when_sandbox_blocks_data]] (interp-critic flavor); [[feedback_codex_plan_section_in_scope]].
