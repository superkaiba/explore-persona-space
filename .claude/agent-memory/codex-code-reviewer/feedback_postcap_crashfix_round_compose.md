---
name: postcap-crashfix-round-compose
description: "Past-cap crash-fix round on a LIVE multi-arm fleet (#2546 r13/v12): verify the brief's round-history claim from events (the 'prior PASS' was a binding reconciler over FAIL+FAIL); reconciler on-next-touch recs ARM on a scoped crash-fix touching their files — three-way landing lines capped at concern grain + a reach-widening probe; brief-pinned bare VERDICT: line added alongside the standard header, CONCERN:: rows kept at forwarder grammar; stale-token sweep subtracts envelope occurrences; --stat-vs-numstat marker counts pre-adjudicated as a frame fact"
metadata:
  type: feedback
---

From #2546 r13 compose (review sentinel v12, 2026-08-26), layered on
[[postpass-delta-round-compose]] + [[reconciler-upheld-cap-round-compose]]:

1. **Verify the brief's round-history claim from events.jsonl before echoing
   it.** The brief said "written after v11's PASS"; the posted v11 verdicts
   were FAIL (Claude, mechanical-only `marker-shape`) + FAIL (Codex,
   `hollow-verification-gate, substantive`), resolved PASS by a binding
   `epm:review-reconcile` (sentinel v11, top-level v4). The prompt must state
   the outcome precisely — the reconciler record doubles as the fence
   contract for the concerns it downgraded, so inline it verbatim.
2. **Reconciler on-next-touch recommendations ARM on a later scoped
   crash-fix round touching their files.** Grep at compose time whether each
   rec landed (r13: rec-4's "EVERY terminal path" docstring still at :1600,
   rec-3's relabel absent, source-helper rec-2 file untouched) and compose
   three-way landing lines (LANDED | ARMED-NOT-LANDED | NOT-ARMED) capped at
   each concern's own severity — plus a reach-WIDENING probe at the ordinary
   bar when the round EDITS the concern's subject function (new pre-try code
   that can raise: is every widened escape still engine-free?).
3. **Brief-pinned extraction contract over spec template, divergence
   flagged:** the brief demanded a bare binary `VERDICT: PASS|FAIL` line and
   `CONCERN:: <slug> | <one-line>` rows. Emitted BOTH the standard
   `**Verdict:**` header and one bare `VERDICT:` line (CONCERNS maps to
   PASS); kept `CONCERN::` rows at the established forwarder grammar
   (severity token first — the brief's slug-first form would parse as an
   invalid severity in persist_verdict_concerns.py) and flagged the
   divergence in the return.
4. **Stale-token sweep must subtract envelope occurrences** — the verbatim
   reconciler cites prior-round SHAs/ranges, so assert
   `prompt.count(tok) - envelopes.count(tok) == 0`, not blanket absence.
5. **Marker size-claims frame fact:** the impl marker quoted per-file sizes
   as `git diff --stat` COMBINED counts (198+16="214") with an exact 4-file
   total — pre-adjudicate as a framing difference feeding the record-accuracy
   class check (twin decides), never let it read as a fresh Critical.
6. **HEAD-drift carve-out when the composer itself commits agent-memory:**
   the prompt names the reviewed SHA, so add "a stat-only agent-memory
   commit may sit above it at dispatch time — never a finding; the range is
   sha-pinned" BEFORE committing your own memory write to the worktree.
7. **Byte-identity focus composes as a DISCRIMINATION demand:** for a
   "numbers unchanged for live arms" safety claim, the pin check is whether
   the test's expected values are derived INDEPENDENTLY (hardcoded literals /
   inline legacy formula) — a pin recomputing via the code under test passes
   on both sides and proves nothing (#653 r8 class). Pair with composer-run
   premise checks (regen_cap == 2×cap on all declared sides) and the
   installed-API excerpt (worktree .venv had real vllm 0.11.0 — cite the
   in-sandbox path AND inline the signature).

**How to apply:** any past-cap scoped crash-fix review on a live fleet, and
any round whose open concerns carry reconciler-armed landing duties.
Compose script: /tmp/codex-2546-r13-compose.py (fresh-write, COMPOSE-OK
sentinel, envelope-scoped stale-token sweep).
