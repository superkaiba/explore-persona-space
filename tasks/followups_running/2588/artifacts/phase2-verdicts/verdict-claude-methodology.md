# Claude critic — Methodology lens — VERDICT: REVISE (plan v2)

## MUST-FIX 1 — pod-2588 overruns the ~130 GB MooseFS quota under its own topology
§9's "Disk rows (mount-bound)" books ONE per-pod peak formula: "weights (≤64 GB, one big model
resident) + ≤2 cells' captures (≤40 GB) + venv (~15 GB) ≈ 120 GB". That formula fits the
single-model 27B/OLMo pods (27B: ~54 + 39 + 15 ≈ 108 GB) but NOT pod-2588, which §9's own
parallelization paragraph assigns 11 cells across 7 distinct models in 3 waves, 4 concurrent
cells (one per GPU).
- HF_HOME weights accumulate MONOTONICALLY across waves: 0.8B+2B+4B+9B+O7I+O7T+Q2.5-7B ≈ 75 GB
  bf16. No weight reap is declared anywhere — "per-cell upload-then-reap" covers CAPTURES only.
- 4 concurrent in-flight fp32 capture stores at the plan's own 6-20 GB/cell run to ~56 GB.
- Worst realistic peak ≈ 75 + 56 + 15 ≈ 146 GB > ~130 GB quota. Even the single worst wave
  (9B-b, O7T-b, O7I-a, Q2.5-cap: weights 61 + captures ~56 + venv 15) lands ≈ 132 GB.
`assert_out_root_headroom` turns this into a fail-loud HALT rather than a corrupted store, but
either way the phase carrying 11 of 19 cells dies mid-run. Technical failure, not an efficiency
nit.
FIX (one paragraph in §9): declare a per-model weight reap (delete a model's HF_HOME dir once
its last cell uploads) and/or pin a wave assignment showing pod-2588's OWN peak arithmetic
(Σ cumulative weight bytes + 4 × max concurrent capture store + venv) under 130 GB with margin.
Grounding: §9 "Disk rows (mount-bound)" + "Per-GPU-phase parallelization"; A16;
`.claude/rules/gotchas.md` EDQUOT.

## MUST-FIX 2 — the G4/G5 regen path is self-defeating on the surface most likely to trigger it
§4.4 instantiates `vLLM(model_id, max_model_len=budget+GEN_MAX)` with PROMPT_TOKEN_BUDGET 7,104;
§7 G4/G5 pre-register "regen of affected rows at 2× cap". For think-GPQA (cap 8,192) regen needs
7,104 + 16,384 = 23,488 tokens, but the engine is pinned at ≤ 15,296 under any reading where
GEN_MAX derives from the original caps. The regen therefore errors or re-truncates at the SAME
boundary — voiding the registered remedy exactly where §8 rates cap-blowout "Medium"
(thinking-arm CoT lengths). Standing #505/#601 class (raising a cap on an INHERITED rig ⇒
re-check its `max_model_len` pins) plus the #2221 regen-headroom incident.
FIX (one line in §4.4 or the §7 G4 row): the regen path RE-INSTANTIATES the engine with
`max_model_len ≥ PROMPT_TOKEN_BUDGET + 2×cap`, and states that 2× caps stay within each model's
native context window.

## MUST-FIX 3 — registered P0 split-count jq is wrong-pathed [CONVERGES with Statistics critic]
§4.4 P0 step 4 and §7 register
`jq '.train_10k|length, .val_400|length, .test_1000|length' eval_results/issue_2330/split_ids.json`
expecting 10000/400/1000. Executed against the committed file it yields 0 — top-level keys are
counts/splits/sha256/..., and the ids live under `.splits.*`. With the expect-assert, P0 halts
on a HEALTHY artifact; without it, 0 is recorded as the measured n_train feeding the n-vs-d
registration.
FIX: correct the paths in §4.4 step 4 AND §7, and add the sha256 cross-check the file already
carries top-level.

ORCHESTRATOR NOTE — both critics' proposed fixes are ALSO broken, verified by execution this
session. jq's `|` binds LOOSER than `,`, so `.splits.train_10k|length, .splits.val_400|length,
.splits.test_1000|length` parses as a re-pipe chain and still exits rc=5
("Cannot index number with string \"splits\""). Two forms actually work:
  jq '(.splits.train_10k|length), (.splits.val_400|length), (.splits.test_1000|length)'  → 10000/400/1000, rc=0
  jq -c '.counts'  → {"train_10k":10000,"train_5k":5000,"val_400":400,"test_1000":1000,"wc_test_1k":998}
Prefer `.counts` (single read, no precedence trap) with a `.splits` length assert as the
cross-check. Verified sha256 prefixes: train_10k=a74675bfed val_400=61c7e6234e
test_1000=b1c32e2197 — all match §10.

## WHAT THE CRITIC FOUND GOOD (independently probed, not taken on trust)
The v1→v2 correction record is called exemplary: the fit basis was re-read from #2330's own
sentence (critic verified "~194 s total for 62 fits" VERBATIM in the parent body — ~3.1 s/unit
is the correct reading) and the total was honestly booked DOWN.
All nine lit-review §7 priority items are GENUINELY INSTANTIATED, not name-checked: retrieval
primary / R² diagnostic; permutation calibration promoted to the instrument with the trend fit
on calibrated scores; per-model dense sweep with one uniform val-frozen selection rule
satisfying selection-symmetry option 2; length residualization + length-only baseline on the arm
contrast; pinned index version with per-checkpoint measured/estimated flags and the measured set
correctly narrowed to three; matched n/pool/k + effective-rank covariate panel; #2546's
end-of-CoT read at pinned SHAs; the pre-registered GPQA-vs-generic hard-set prediction (H2); and
the fixed-size column protected as primary (never-drop descope tier, paired per-prompt bootstrap
on the full 1,000-prompt shared test split, kill criterion on losing any column checkpoint).
Reuse attestation checks out under independent probing: all ported symbols resolve at the pinned
blobs at the cited line numbers, the parent-lineage diff is EMPTY (rc=0), the
working-file-vs-pinned-blob line drift is explicitly defused, and the G6 transformers ≥ 5.13.0
hard floor is a load-bearing silent-corruption catch for the entire OLMo control.

## CONCERNS FOR THE ANALYZER (non-blocking)
- **The 27B "family pilot before fan-out" is partially illusory as scheduled.** §4.5 blind-spot
  item (ii) claims the first cell per pod family runs "with the same gates before fan-out", but
  §9 provisions all six pods with only a 120 s-per-ordinal jitter, so the three 27B pods run
  CONCURRENTLY and G3's wall re-projection can only fire ~2.5 h in, mid-flight on the siblings.
  G1/G6 fire per-pod in the driver prologue at near-zero GPU cost, so VALIDITY is unaffected —
  spend-risk only. Cheap tightening: hold `-q3627b`/`-q3827b` driver starts on `-q3527b`
  clearing its gate prologue, or reword the enumeration.
- H2's 9-pair Wilcoxon pools two gap constructs (7 within-model Qwen arm gaps + 2 cross-model
  OLMo Think-vs-Instruct). Ship the Qwen-only 7-pair variant beside it.
  [CONVERGES with Statistics critic.]
- Fixed-size column reads as "capability + training recipe at fixed architecture" — no
  observational column separates capability from the recipe that produced it. Narrate the
  positive branch as correlational. [CONVERGES with Alternatives critic concern 2.]
- The contingent GPQA judge fallback would be a ≥5k-call classification wave needing the rule-26
  pilot gate. [CONVERGES with Statistics Must-Fix 3.]
- Banked 9B ceiling-draw texts (seeds 43/44) are assumed present in the banked cap2048 stores;
  full-grain matched-id asserts fail loud if absent and the driver retains the gen path as
  fallback. Confirm at consume.

## WORKFLOW-SURFACE FOLLOW-UP surfaced by this critic
`verify_plan.py`: WARN when a plan carries BOTH a cap-hit regen trigger ("regen … at 2× cap" /
`cap_hit` vocabulary) AND a `max_model_len` expression derived from the base cap, with no
regen-time re-pin statement. Concrete and recurring: #505, #601, #2221, now #2588.
