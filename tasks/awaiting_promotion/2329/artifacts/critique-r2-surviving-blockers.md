# #2329 `q35_ladder_decay` — CRITIQUE panel ROUND 2: SURVIVING blocker list (revise plan v7 → v8 against exactly this)

Panel outcome after per-lens ensemble decisions:

| Lens | Claude | Codex | Binding result |
|---|---|---|---|
| Statistics & measurement | REVISE | REVISE | **REVISE** — blockers UNIONED (agree in direction; no reconciler) |
| Methodology & baselines | PASS | *(see § Methodology below)* | *(recorded there)* |
| Efficiency | PASS | PASS | **PASS** — no must-fix either side; 1 fold-in |
| Single-variable consistency | PASS | (no twin) | **PASS** — 2 fold-ins |

**Round-1 items S1-S5 and M1 are all CLOSED** — both statistics reviewers verified S1-S5
independently against the artifacts and the inherited code, and the Claude methodology lens
verified M1's four parts. Do NOT re-open them. Everything below is either NEW (introduced by
the v5→v7 revision) or a fold-in that a revision is happening anyway.

---

## BLOCKING — Statistics (both reviewers REVISE; the two blockers are DIFFERENT and INTERACT)

### R2-S1. The Leg B lattice's negative affirmative branch is fireable by pure scale compression

§3 H5 + the Leg B lattice register "patch-more-persistent ⇔ the ΔD carrier-clustered 95% CI
sits entirely below 0" as an affirmative, reportable construct claim. Under a NULL of equal
PROPORTIONAL decay, an arm that starts lower produces a negative raw ΔD MECHANICALLY — e.g.
steered 0.35 → 0.175 against ceiling 0.80 → 0.40 gives ΔD = −0.225 — with consistent sign
across all six carriers, so the clustered CI can exclude zero and the registered lattice then
REQUIRES narrating "the patch installs something MORE persistent than the prompt". That is a
false affirmative that would rewrite the Takeaways.

This is the EXPECTED regime, not a tail case: parent-realized install levels are F 0.13-0.41
against a ceiling near 1. The plan already ASSERTS the mechanism in report obligation R4
("steered raw scores start below ceiling's, so scale compression biases ΔD downward") — but
R4 is a report note and the verdict-generating lattice is the binding object. Notes do not
bind branches.

The POSITIVE branch needs no change: compression works against it, so it stays conservative.

FIX (zero new calls, no §9 row changes): condition the ΔD<0 affirmative verdict on the
scale-normalized companion agreeing. The manifest's contrast figure ALREADY computes
D_F = F(Q1) − F(Q4), so require the paired ΔD_F CI to sit entirely below 0 as well (on
segments passing the registered 0.125 denominator-suppression bar) before licensing
"patch-more-persistent"; otherwise that tail reads "inconclusive — raw-scale contrast
confounded by starting-level compression (see the Q1 gap)". Register the amended branch in §3
(H5 + the Leg B lattice) and mirror it in the manifest contrast entry.

### R2-S2. Post-treatment coherence conditioning can reverse the contrast — and this defect was introduced by round 1's own S2 remedy

§4.1 screens each arm SEPARATELY on coherence, while §11 acknowledges coherence is
treatment-associated. Carrier pairing does NOT restore exchangeability after draw-level
survivor selection, so ΔD currently estimates decay among different arm-specific SURVIVOR
SETS while §3 interprets it as what the patch installs.

**Verified retention counts (recomputed by the orchestrator from the committed parent
artifacts, not relayed):** primary steered install-ce rows (cells `install_r1_pirate`,
`install_r2_butler`, `install_r3_warm`, slot ce, arm steered) = **90 total, 87 kept** at
score > 60. Ceiling anchors on the same three personas (`coherence.anchors.scores.jsonl`,
keyed by `value_id`) = r1_pirate 49/60, r2_butler 55/60, r3_warm 58/60 = **180 total, 162
kept**. So retention is **96.7% steered vs 90.0% ceiling** — measurably different survivor
sets.

FIX (zero new calls — the ~5.2k estimate already counts rows BEFORE coherence drops): score
all length-eligible rows and report BOTH an all-generated-completion ΔD and the
coherence-conditional ΔD, with their retention counts, in the decay stats artifact. If the two
disagree, the patch-depth headline is UNRESOLVED and must be reported as such rather than
resolved in favor of either.

### R2-S3. The two blockers above COMPOUND — say so rather than treating them as independent guards

The screen removes proportionally MORE rows from the CEILING arm (10.0%) than from the steered
arm (3.3%), and the ceiling arm is the higher-scoring one, so screening plausibly raises the
ceiling arm's mean more than the steered arm's — pushing ΔD in the SAME direction as the
scale-compression artifact in R2-S1. Register this interaction explicitly: the two guards are
not orthogonal, and a report that passes each separately can still be reading a compounded
artifact. State the joint reading requirement (raw + normalized × screened + unscreened) and
which cell of that 2×2 is the headline.

### R2-S4. CORRECT THE FILTER'S STATED RATIONALE — my round-1 justification was wrong on the data

Round 1's S2 justified the coherence filter with the claim that "incoherence concentrates in
patched arms". The parent's realized data CONTRADICTS that: the PROMPTED ceiling arm loses
10.0% of its rows to the >60 screen while the PATCHED steered arm loses 3.3%. The filter may
still be defensible, but NOT on that ground — the defensible ground is INSTRUMENT PARITY with
the parent's own behavioral-F conditioning. Do not carry the round-1 rationale into v8; state
the parity ground instead, and state the measured differential and its direction.

---

## BLOCKING — Methodology & baselines

**RESOLVED: Claude PASS (M1 CLOSED) vs Codex REVISE (M1 STILL OPEN) → binding reconciler ruled
REVISE.** Claude's "M1 CLOSED" was OVERRULED as mistaken: it verified the assert's THRESHOLD
band but never opened the function the plan names, so it could not have seen the object
mismatch. Codex's finding is confirmed on the code. Parts 1 and 3 are FIXED (agreed by both).
The round-1 no-recapture ruling is UNTOUCHED — this is not a re-opening of it; the plan's TEXT
for that settled remedy names the wrong function.

### R2-M1. The donor-identity assert, as registered, would HALT A VALID RUN — one blocker, three edits, +0 GPU-h

**Q1 — the two objects are genuinely different (CONFIRMED on the code).**
`capture_answer_states` (`issue2329_run.py:2229-2316`) returns the MEAN OVER COMPLETION token
positions from teacher-forced ctx+completion+eot forwards, and REQUIRES a `completions` text
input. `vc_bank.pt.per_context` is built by `capture_bank` (`:1510-1563`) as EXACT
SINGLE-POSITION states — `v_ce = captured[layer][j, ctx_len-1]` (`:1532`),
`v_pe = captured[layer][j, pe-1]` (`:1534`) — from right-padded CONTEXT-ONLY forwards.
`_slot_state` (`:1566-1568`) is a pure selector; it captures nothing. Plan §4.1 line 101 and
assumption 14 ("the same `capture_answer_states` code") register the WRONG path. Implemented
literally, the assert either cannot run at L1 (no completions exist before the L2 anchors) or
compares span means against slot states, giving a cosine far below 0.99 and a FALSE HALT
`RC_DONOR_IDENTITY` on a healthy run. A registered HALT gate with an affirmative misfire is a
PLAN defect, not an implementer-recoverable detail.

**Q2 — the phase-ordering problem is real.** §4.2 line 117 runs G1 at L1, but the cross-type
donor screen is a judge-built L3 output (`issue2162_ladder.py:195-199`: `--donor-screen` is
"REQUIRED for a non-smoke grid", and the smoke path explicitly falls back to "the frozen
PRIMARY donors unscreened"). The screened donors the assert is specified over DO NOT EXIST at
L1.

**Q3 — the provenance gap is real, low-reachability, and zero-cost to close.**
`regime_fingerprint` (`issue2329_run.py:676-696`) omits `model_revision` while its own
docstring registers "EVERY output-affecting knob" — a contract violation that would permit a
resumed run to combine shards from different revisions. Reachability today requires an
operator-changed revision on a resumed run, and both plausible pin values resolve identical
weights, so it is provenance hygiene rather than independently blocking — but the fork's
out-root is fresh (nothing to invalidate) and the fix is one line. Likewise the raw anchor
rows (`issue2162_ladder.py:833-846`) and grid shard rows (`:1123-1150`) carry no `_repro`
(only the adjacent `.pt` / done records do), which falsifies §4.1's claim that every persisted
artifact records the revision.

**THE FIX — three edits, exactly as the reconciler worded it:**

1. **§4.1 "Donor-identity assert" + §7 G1 + §11 assumption 14:** replace the registered path
   `capture_answer_states` / `_slot_state` with a fork-local re-capture using the exact
   **`capture_bank` geometry** — right-padded CONTEXT-ONLY forwards of the donor contexts
   (identical token ids via the parent's context builders), exact positions ce = ctx_len-1 and
   pe = prefix_end-1 — compared per layer against the staged `vc_bank.pt per_context` records
   for **BOTH slots (`v_ce` AND `v_pe`) across all 32 layers**, cosine >= 0.99, HALT
   `RC_DONOR_IDENTITY` unchanged. `_slot_state` may serve as the selector on both sides;
   `capture_answer_states` MUST NOT appear in the registered path. Correct assumption 14's
   "same `capture_answer_states` code" to `capture_bank` geometry.
2. **Donor set:** 2-3 **FROZEN** donor-plan cross-type context ids, NAMED in the plan (the
   `ladder_bank.json` frozen PRIMARY donors, present in the staged parent bank per assumption
   2's L1 pre-check) — NOT "screened" donors. Weights-identity is a BANK-level property, so
   any frozen donors in the staged bank suffice.
3. **Provenance:** add `"model_revision"` to the `regime_fingerprint` payload, add a
   `model_revision` field to the raw anchor and grid JSONL row writers, and align §4.1's
   "every persisted artifact records it" with the realized set.

**Cost: 0 GPU-h.** Still 2-3 right-padded context forwards on the already-loaded pinned model
at L1; §9's L1 row already contains them. The remedy changes WHICH function runs, not the
forward count. The plan-approval gate is NOT re-triggered on this account.

**DO NOT WIDEN:** the >=0.99 threshold, the 2-3 donor count, the HALT class, smoke parity, and
the round-1 no-recapture ruling are all untouched and agreed by both reviewers.

---

## FOLD-IN — mechanical and disclosure items (a revision is happening; land them now)

### F1. The stale `≈7.5k` wave declaration — flagged INDEPENDENTLY by three lenses

§7's G4b gate declares `wave_n_calls≈7.5k` and §11's routing bullet says "the one ≈7.5k
wave", while §9's re-derivation gives ≈4.7k production + ≈416 pilot ≈ 5.2k. Routing-inert
(`threshold_base=0` is pinned on pilot AND production, so the route is deterministically Batch
regardless of count, and G4b's pilot sizing derives independently) — but `llm-judging` rule 26
wants the declared `wave_n_calls` mirroring the dispatch kwargs 1:1, and a stale declared
count in the registration stays wrong permanently. Grep the plan for `7.5k` and reconcile every
hit to the §9 figures.

### F2. Book or bound the conditional prefix-end stratum's judge calls — both efficiency reviewers

§9's Leg B line neither books nor bounds the conditional pe stratum. Codex quantified it: each
realized Qwen3.5 install-pe transfer adds at most 6 carriers × 5 draws × 4 segments = 120
calls; six such cells add at most 720, making the full-survival Leg B ceiling ≈7.1k including
the pilot — inside the same Batch wave and the ≤24 h SLA, so not an infeasibility risk, but it
must be disclosed rather than omitted from the headline. Add one clause to the §9 Leg B row
(e.g. "+ ≤~720 conditional pe-stratum calls, only on realized install-pe transfers; parent
precedent 0") and require execution telemetry to disclose the realized conditional increment.

### F3. The "EVERY install-pe cell" wording — flagged by three reviewers, and it originated in the orchestrator's round-1 list

§2 and §4.1 say the parent's realized lattice puts "EVERY install-pe cell" (and "every erase
cell") at `no-clean-transfer`. Realized: 4 of 6 install-pe cells are `no-clean-transfer`;
r4_trait and r5a_lu_therapy install-pe are `untestable`. The OPERATIVE fact — ZERO install-pe
`transfers` — holds, and both untestable rungs fail gates at both slots so they could never
enter Leg B anyway. Rewrite as "every TESTABLE install-pe / erase cell". (Provenance: this
phrasing came from the orchestrator's round-1 blocker list, carried from the statistics
critic's wording without re-resolution against the artifact.)

### F4. Report-side additions to register as obligations

- **Per-rung breakout under lattice flips (consistency N-b):** if a fixed-surface rung
  gate-survives on Qwen3.5 but its own Leg A cell realizes `no-clean-transfer`, the pooled
  primary steered pool dilutes toward zero asymmetrically — the same mechanism the plan names
  for pe. Register that the report breaks the primary pooled ΔD out PER RUNG whenever
  Qwen3.5's lattice flips any primary-set cell.
- **Rung-composition parity within pairs (Claude statistics concern 3):** common support is
  ≥1 completion per ARM per carrier, not per (rung × arm), so coherence/length drops can leave
  a carrier's steered and ceiling pools spanning different rung subsets. Register that the
  analyzer confirms rung composition is matched within pairs before narrating ΔD.
- **Realized common-support carrier count (Codex statistics):** report the realized Qwen3.5
  common-support carrier count prominently if gate or length attrition reduces it below six.
- **Realized null-arm means vs the 0.10 bar (Claude statistics concern 4):** check them on the
  Qwen3.5 side; R0 already carries the narration duty for any withheld cell.
- **Pin-engagement assert (Claude methodology concern):** `_repro.model_revision` will record
  the CONFIG value, not the realized snapshot resolution, so a threading bug would leave a
  plausible-but-unengaged pin in provenance. Add a one-line assert that the resolved HF
  snapshot path contains the pinned SHA (`snapshots/<sha>/`), making pin ENGAGEMENT itself
  smoke-verifiable. The implementation review must additionally confirm all four ladder-fork
  load sites realize the pass-through.

---

## Constraints on this revision (do NOT re-open)

- Leg B stays on the Batch API (user's scope directive + CLAUDE.md; efficiency reconciler,
  round 1).
- No budget-derived abort threshold (ruled anti-efficient, round 1).
- The pod holds through the L3 gate (the split's honest cost exceeds the idle, round 1).
- No donor-bank re-capture (the model repo's commit history proves the realized bank is
  on-basis; round-1 methodology reconciler).
- S4 stays direction (ii) (code byte-verbatim, registration verdict-BINDING) and S5 stays
  direction (i) (code changed, Holm held at m=4) — both were verified faithful by both
  statistics reviewers and the Claude methodology lens in round 2.
- `gpu_hours_total` should remain 6 unless YOUR revision genuinely changes it. Every fix above
  is costed at zero added GPU and zero added judge calls except F2, which BOUNDS an already-
  possible conditional increment rather than adding work. If the booking changes, say so
  prominently — it re-triggers the plan-approval gate.

## Manifest

R2-S1 changes the contrast figure's registered branch semantics and R2-S2 adds a second
reported estimate; update `artifacts/planned_manifest.json` accordingly, keeping every
PRE-EXISTING entry byte-identical (parent-round AND this round's already-landed entries —
the orchestrator has verified parent-round entries untouched across both prior passes and will
re-verify). Flag in your return whether CONDITION-SET MEMBERSHIP changed again.
