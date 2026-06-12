---
name: Amendment inherits --no-traj (endpoint-only marker capture)
description: When Methodology item 5 (marker trajectory MUST be logged) does NOT force a REVISE on a strict one-variable amendment
type: feedback
---

Rule: Methodology lens item 5 (per-step marker log-prob trajectory REQUIRED, REVISE on endpoint-only capture) does NOT fire on a strict one-variable amendment whose executed parent runs used `--no-traj`, whose question is an END-STATE cross-arm comparison (not recipe-distinguishing, not when-does-leakage-emerge), and whose floor/saturation disambiguation is covered by the four-float logit capture (EOS margin).

**Why:** Item 5's rationale is "endpoint saturation alone cannot distinguish recipes or locate when leakage emerges." When neither is the experiment's question — e.g. #464 minimal_content_cn (2026-06-10): paired d_seed between two encoding arms at matched deliberate-full-convergence recipe — the trajectory adds interpretive color but cannot flip the headline, and the DV-identity scope marker + parent precedent (two executed runs, same flag, passed all critics) make inheritance the right call. Demanding traj would be "more rigor at the margin," which The Bar bans.

**How to apply:** Still note it as a concern-for-analyzer (endpoint-only capture = scope caveat in the clean-result). REVISE remains correct when the plan's question IS about training dynamics, recipe comparison, or when-leakage-emerges, or when the endpoint leakage cells are expected to saturate (ceiling) so the endpoint read collapses without a trajectory.
