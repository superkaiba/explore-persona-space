# Consistency-checker — VERDICT: BLOCK (plan v2). Marker `epm:consistency v1` posted on #2588.

## VERIFIED CLEAN (independent re-probes, not taken from the plan)
- All plan-cited HF reuse artifacts resolve at their pins: `issue2330_matched` @ b99d86de23
  (5 entries incl. both cap2048 stores); `issue1491_scale_ladder/scale7_refit` @ same pin
  (6 entries incl. `train_25k`, `ceiling_draws`); manifest @ 815ff6d976c6... (7 files).
  Item-(j) dates coherent (manifest 08-05 05:11Z ≤ scale7_refit captures 08-05 07:21Z;
  split_ids git 08-16 ≤ cap2048 stores 08-17).
- All 10 ported functions exist at exactly the claimed lines at the pinned blobs
  (2502 @ a736aebb92: 263/343/364/921; 2546 @ 89680c72f9: 554/591/643/686/756/786);
  fix aa8c19e3 is an ancestor of the 2502 pin; the working-file drift claim reproduced.
- Reused #2330 code is main-resident: `git log origin/main..origin/issue-2330 -- <5 modules>`
  is EMPTY. `scripts/issue928_common.py` is on main with no branch drift.
- G2 grounding is real: `eval_results/issue_2330/matched_fits_q25_n10k.json::port_parity_anchor`
  = expected/realized 0.7250873220237553, `abs_deviation: 0.0`, `tol: 0.01`, n_train 25000.
- GPQA routes: `Idavidrein/gpqa` gated="auto"; `hendrydong/gpqa_diamond` + `ankner/gpqa`
  ungated and resolving. A17's "~194 s TOTAL for 62 fits" quote is VERBATIM in #2330's
  `dense-fit-loop-unbatched` caveat.

## BLOCK 1 — arm-b end-of-CoT read attributed to the WRONG ported function
**This is the round's most consequential finding. ORCHESTRATOR-VERIFIED at the pinned blob.**

`compute_read_idx` at 89680c72f9 returns only PROMPT-side positions. Verified verbatim:
  - signature takes `prompt_ids: list[int]`, returns `int`
  - docstring: "Registered per-arm **v_C** read point (plan §4.1), as an index into prompt_ids."
  - all three modes resolve inside the prompt: `prompt_last` → `len(prompt_ids) - 1`;
    `pre_think` → `_find_last_subseq(prompt_ids, side.open_ids) - 1`; `assist_start` →
    `on_prompt_len - 1`. It never reaches into the completion.

The end-of-CoT object is a DIFFERENT code path: `build_capture_row` computes
`positions["cot_boundary"] = prompt_len + close_tok[1] - 1` (line 1289 at the pinned blob),
derived from the completion-side close-token span via `issue928_common.char_span_to_token_span`
over `segment_completion_arm`'s cot char span. `cot_boundary` is a distinct member of
`KINDS_POST = ("cx_last", "cot_mean", "cot_boundary", "ans_mean", "out_mean")` (line 320), and
#2546's own plan defines `cot_boundary` as a distinct object from `cx_last`.

CONSEQUENCE: as the §4.4 pseudocode literally reads, arm (b) captures a PRE-CoT prompt-side
state. Arm (b) collapses toward arm (a), the arm manipulation is silently nulled across all 9
thinking cells, and the H2 headline returns a null BY CONSTRUCTION with no tell.

FIX: port the cot_boundary / answer-span logic (not `compute_read_idx`), name the
`issue928_common` dependency in §4.6, and correct §4.4 plus the §6 pooling row.

## REQUIRED FIX 2 — 9B banked ceiling draws consumed but uncited; both banked sets are 1,024-cap
§4.1 needs `issue2330_matched/qwen35_9b/ceiling_draws/{seed43,seed44}` (resolves at b99d86de23
per the checker's probe) but it is ABSENT from §10 and from the (a)-(m) attestation. #2330's
"never quoted as 2,048-cap quantities" ceiling-cap caveat is being inherited SILENTLY.
FIX: cite + attest, then either state the caveat or regenerate 2,048-cap ceilings for the two
capture-only checkpoints.

## WARN 3 — P0 split-count jq reads a nonexistent key
Third independent discovery of the same defect (Methodology and Statistics critics both found
it too). Lists nest under `.splits` / `.counts`; the registered command returns 0/0/0. The facts
themselves check out: counts 10000/400/1000, sha pins match.

## WARN 4 — the 2502 `EMPTY_THINK` template contract is Qwen-specific
As ported it RAISES on BOTH OLMo-Instruct arm-(a) cells. A14 covers only the 2546 think-pin
SideSpec. FIX: pre-register the per-family template contract.

## WARN 5 — no union-drop mechanism for the 7,104-token length scan across 12 tokenizers
An OLMo-only over-budget prompt fails a cell loud with the remedy (a split change) unregistered.
FIX: add a P0 12-tokenizer scan + a pre-registered union-drop.

## NOTE 6 — G2's 1e-6 tolerance is calibrated on same-device-class evidence
#1491 and #2330 were both H200 while the plan moves the anchor to H100. Caveat handling is
coherent, but running G2 on one of the plan's OWN H200 pods removes the device change entirely.
[Cleanest disposition yet proposed for this; both Claude critics flagged the G2 trip risk.]

## NOTE 7 — engine provenance of the banked cap2048 texts (vLLM version) is presumed, not recorded
FIX: have P0 read it from the store meta.
