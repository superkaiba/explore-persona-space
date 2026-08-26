---
name: reconstructed-marker-compose
description: When the impl marker is ORCHESTRATOR-RECONSTRUCTED (implementer session killed pre-transcription), carry the provenance disclosure, state shape facts neutrally (own headings vs (a)-(d) labels, missing pin/Gate-scope fields), and pre-run the function-grain pre/post differential yourself
metadata:
  type: feedback
---

Compose recipe for a round whose `epm:results` / `epm:experiment-implementation`
marker was posted by the ORCHESTRATOR, reconstructed from durable evidence
(landed commit + message + re-run tests) because the implementer session was
stopped (watcher ALIVE-BUT-STALLED auto-respawn) before transcribing the
returned report (first hit #2294 r1, 2026-08-22):

1. **Carry the brief's provenance disclosure verbatim-in-substance** as a
   binding-context block ABOVE the marker envelope: reconstructed, not
   testimony; every claim verified against diff/repo; a claim that does not
   hold is a finding.
2. **State the marker-shape facts NEUTRALLY, never pre-adjudicate:** a
   reconstruction typically carries the (a)/(b)/(c) CONTENT under its own
   headings but lacks the literal `### (a)`–`### (d)` labels, the (d)
   needs-eyeball section, the ruff-policy pin field, and the
   `**Gate-scope check (#1288):**` line. List exactly what is present under
   which heading and what is absent, remind Codex of per-blocker strip
   keying + Step 0.7 / Rule 8 + present-but-imperfect routing, and let the
   rubric route it. Still emit the GATE-SCOPE THRESHOLD line off the
   marker's real `ts`.
3. **Pre-run the pre-fix/post-fix differential at FUNCTION grain yourself**
   (import the changed function from main's src and the worktree's src via
   `sys.path.insert(0, '<wt>/src')` — print `__file__` to confirm which copy
   loaded) and attest both outputs. This gives Codex a baseline for the
   "does the new test actually fail pre-fix or is it tautological" priority
   without an execution env, and makes the reconstruction's key behavioral
   claim independently yours.
4. **Re-run the round's own test files yourself** when cheap (the worktree
   .venv usually exists — the implementer ran them there); the attestation
   then reads "composer re-ran, independent of the marker's identical claim".
5. Mid-compose gate-result corrections (e.g. a pending lint verdict landing
   as PASS) follow [[established-gates-attestation-compose]]: verify the
   named log yourself (rc line, FAIL count, the round-relevant WARN line
   verbatim) before patching the attestation.

**Why:** the reconstruction is the ONE case where marker-shape absences are
structurally guaranteed (the orchestrator wrote it without the template) —
without the neutral facts + provenance context, an adversarial twin burns
its verdict on mechanical-contract blockers against a document the
implementer never wrote; without the composer-run differential, the round's
central regression-pin question is unverifiable from a read-only sandbox.

**How to apply:** any brief carrying an "orchestrator-reconstructed /
implementer report unrecoverable" provenance disclosure. Related:
[[9ater-followup-round-report-placeholder]] (in-session placeholder variant),
[[missing-impl-marker-probe-checklist]] (marker absent entirely).

**#2315 r3 sharpenings (2026-08-24, second hit of the shape):**

- **Composer-run ruff-policy pin substitutes for the reconstruction's
  missing pin field.** A reconstruction on a LIVE_WORKFLOW_HELPERS diff
  predictably lacks the Step 0.5 `(c)` ruff-policy pin-invocation field;
  when the orchestrator's re-verification also skipped it, RUN the pin
  yourself (`uv run pytest tests/test_ruff_policy.py -q` in the worktree,
  seconds) and attest rc + count — the #1672 bare-ruff-green/pin-red hazard
  is then empirically excluded and the absence composes as a neutral shape
  fact instead of an open substantive question.
- **Reused donor plan span gets a byte-identity assert.** When reusing the
  prior round's `---BEGIN APPROVED PLAN BODY---` span verbatim, extract the
  body between the envelope tokens and assert it equals the canonical
  `plans/plan.md` (strip trailing newlines) — a plan amendment between
  rounds otherwise ships silently stale (the #546 class, now caught
  mechanically at reuse time, not only by the readlink probe).
- **Reconstruction + self-authored-concern discharge compose together:** the
  closure `addressed` row is posted by the ORCHESTRATOR as part of the
  reconstruction (pre-review) — frame it explicitly as "the closure CLAIM
  this review adjudicates, not settled truth", with NOT-ADDRESSED ⇒ the row
  is PREMATURE + the one sanctioned same-id `CONCERN:: ` re-emission.

**#2351 r2 sharpenings (2026-08-25, third hit — reconstruction on a
FAIL+FAIL-union fix round):**

- **Mid-compose drift can be the sibling twin's `addressed` rows, not only
  `raised` rows.** The parallel Claude reviewer finished FIRST and posted
  per-concern `addressed` rows while the build script ran (count grew 6→7→10
  across minutes). Same fix as #2326 r3: filter the inlined snapshot to
  `ts <= impl-marker ts` (never a bare row-count assert on the raw file),
  and report every excluded row to the orchestrator — an addressed-row leak
  is worse than a raised-row leak (it hands Codex the sibling's VERDICT).
- **Function-grain differential on a `scripts/_bootstrap`-style module needs
  a fake repo root:** `workflow_lint.py` runs `_load_agent_spec_caps()` at
  import, resolving `Path(__file__).parents[1]/.claude/config/...` — a blob
  copied to bare `/tmp` crashes at import. Stage `/tmp/<fr>/scripts/<blob>.py`
  + `/tmp/<fr>/.claude/config/<needed file>` and import via
  `spec_from_file_location` with the WORKTREE's `scripts/` on sys.path.
- **Composer-run differential doubles as pin-adjudication evidence:** running
  the committed pin's EXACT fixture through both blobs (r1: real call
  suppressed, 1 err at :4; r2: 2 errs, real at :3) settles both the
  "pin fails pre-fix" [from commit msg] claim AND the closure observable in
  one leg — compose it as a fact with "your duty is the MECHANISM half by
  reading", never as a replaced duty.
- **Post-compose worktree-dirtying hazard:** a compose-time fact attesting
  `git status --porcelain` CLEAN goes stale if the composer then writes its
  own agent-memory file into the SAME worktree before the review runs —
  name the residue in the return so the orchestrator can discount a Codex
  observation of it (the memory edit stays uncommitted mid-round per the
  #2332 r2 rule).
