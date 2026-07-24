# Why the linear context→answer map collapses on Track M — banked-artifact audit

Inline audit (chat), 2026-07-24, issue #825. All numbers read this session from the cited
artifacts; text stats computed from the two rollout JSONLs
(`issue825_userbase_map/raw_completions/{generation,track_s}` on the HF data repo, staged
under `data/issue_825/audit_dl/`). No refits were run; this is a diagnostic re-read of the
existing context-arm family, not a new mapping design.

**The question.** Track S (5,000 LMSYS single-turn contexts, answers sampled at T=1.0)
reads held-out ridge R² 0.673 (instruct) / 0.588 (pretrained) at L19. The Track-M assistant
cells (n=2,000 two-turn conversations, a1 greedy) — structurally the same task — read
+0.076 / −0.078 / −0.461 / −0.390. An MLP on the same tensors reads 0.49–0.56. Why?

## Ranked attribution (instruct/chat, L19; total gap 0.673 − 0.076 = 0.597)

### 1. ~70% is the n=2000 fit regime, and most of that is a GCV-ridge degeneracy, not statistical power (−0.415)

`matched_n_curve/results.json` (Track S1 itself, L19, 5 fixed-rng subsample draws per n):

| n | 100 | 200 | 300 | 497 | 700 | 1000 | 1500 | 2000 | 5000 |
|---|-----|-----|-----|-----|-----|------|------|------|------|
| R² mean | 0.341 | 0.430 | 0.446 | 0.476 | **0.482** | 0.437 | 0.397 | **0.258 ± 0.039** | **0.673** |

The curve is **non-monotone**: it peaks at n=700, dips to 0.258 at n=2000 (all five draws in
0.189–0.293 — systematic, not an unlucky subsample; the power-curve nested draw agrees, L19
0.175 / layer-max 0.321), and recovers only at n=5000 — exactly where n_tr = 0.8n = 4000
first exceeds D = 3584. Since 0.482@n=700 > 0.258@n=2000 with 3× less data, **at least 0.22
of the n=2000 depression cannot be a power effect**. The fit module documents the mechanism
itself (`scripts/issue825_fit_cells.py:81-91`): when the fold Gram can near-interpolate
(n_tr < D), GCV opens a spurious minimum at the lambda-grid floor and held-out R² explodes
(−2..−11 in the #1310 incident); the guard (`GCV_DOF_CAP`) defaults to None, so **every
committed #825 fit ran unguarded**, and every Track-M cell (n_tr = 1600 < 3584) sits inside
the documented degenerate regime. (Selected lambdas were not recorded in the committed runs,
so the grid-floor signature is inferred from the module's documented failure mode + the
curve shape, not from recorded λ.) Corollary: the prompt's "matched n=2000 read 0.321" is a
layer-max of one draw; the honest L19 matched-n reference is 0.258 ± 0.039 — and it is
itself a regime-depressed number, not a clean power point.

### 2. ~30% is Track-M-specific (0.258 → 0.076 = −0.18): leading candidate greedy a1, entangled with corpus composition

**Greedy decoding (supported, converging but indirect).** Track M a1/a2 are vLLM **greedy**
(`issue825_gen_conversations.py:389`); Track S answers are sampled T=1.0/top-p 0.95.
Banked signatures: M a1 teacher-forced NLL mean 0.324 nats/token vs S1 0.564
(`nll_reads.json`); the most-deterministic M quartile (NLL ≤ 0.186) reads L19 R² **−0.25**
while q2–q4 read ≈ +0.02 — S1's lowest-NLL quartile reads **+0.73** (same semantics: subset
reads of held-out predictions). M a1 has 18.6% 3-gram-repetition rows vs S 10.1%. The
rounds-7/8 separator control makes the causal direction explicit on a different substrate:
greedy self-generated spans collapse every linear estimator (per-group tails to −40, base
rotated −1.23 while MLP reads 0.495) and **sampled decoding restores the linear read**
(D 0.590/0.428 → 0.031/0.086). Caveat: within Track M only the extreme quartile is
depressed, and the round-4 real-conversation cells (a1 by logged serving models, NOT greedy)
read −0.27..−0.61 at n=2000 — authorship + conversation sample move the number at least as
much as decoding, and they change together (single-component attribution is barred there).

**Corpus composition flatters the S reference (supported directionally, magnitude
untestable from banked data).** Track S keeps exactly what Track M's filters drop: 9.4%
exact-duplicate answers (471/5000; "Hello! How can I assist you today?" ×172) and 21.4%
≤5-word prompts, vs M's 0.25% exact-dup a1 and 1.65% ≤5-word u1 (≥8-content-token turn
filter; 2600→2000 kept, drops 82 short_turn + 276 too_long). Duplicate context→answer
clusters span the conversation-grouped folds, so the same trivial mapping is trained and
tested on — linearly easy rows that inflate both the 0.673 headline and the 0.258 matched-n
reference. Part of the −0.18 "M residual" is therefore S being flattered, not M being
deficient.

### 3. The "MLP recovers it, so the map is nonlinear" reading is confounded — re-grade it

MLP vs ridge at L19: M instruct/chat 0.557 vs 0.076; M pretrained/chat 0.487 vs −0.461 —
but at n=5000 the MLP shows **zero** advantage (S1 0.654 vs 0.673; S2 0.587 vs 0.588), and
**no Track-S MLP at n=2000 exists anywhere in the banked artifacts**. The MLP's recipe
(PCA-64 target head + early stopping) is a heavily regularized estimator that sidesteps the
n_tr < D GCV pathology, so "MLP ≫ ridge at n=2000" cannot currently be distinguished from
"any well-regularized estimator ≫ unguarded GCV ridge at n=2000". Supporting hint: the
turn-depth rounds' PRESS-ridge (standardized, small fixed grid) reads +0.18..0.23 at n=171
on realistic multi-turn data. Information IS intact (MLP, own-context NLL wiring checks,
L26 per-position argmax 12/12 — though all per-position R² are negative); the *linear
readability* claim is what's unsettled.

## Refuted / excluded

- **Target degeneracy (H1):** y-trace-cov L19 M 1437 vs S1 1791 (80% — no collapse);
  mean-baseline R² ≈ −0.002 both; duplication is 38× HIGHER in Track S. Refuted as stated
  (the duplication axis points the other way — see #2).
- **Short spans / POSITIONS_CAP (H2):** M a1 median 203 words vs S 122 — M spans are
  LONGER; the answer profile is a full-span mean on both tracks
  (`issue825_extract_turnstore.py:410-413`); POSITIONS_CAP=64 touches only the per-position
  side store on both tracks. Refuted.
- **Degenerate u1 (H3):** M u1 median 18 words, 1.65% ≤5-word (vs S 21.4%); 1629/2000 u1
  are literally in the S prompt set. M contexts are cleaner. Refuted (rare URL-encoded/hex
  junk rows remain, <2%).
- **Turn depth per se (H7c):** matched n=171, turn-1 0.175 vs turns 3–11 0.211–0.231
  (`turn_depth_matched_n/results.json`). Refuted.
- **Format:** naturalistic costs only −0.048 at n=5000 (S1N 0.625 vs S1 0.673). Minor.
- **Rig (H6):** excluded — round-4 anchor re-extraction +0.0763 vs committed +0.0757;
  onpolicy-round anchors ±0.004; kresample rig-parity cosine ≥ 0.99994; selection-symmetric
  nulls sit at −3.6..−3.9 with obs ≫ null; the MLP works on the exact same tensors.

## What stays confounded

The −0.18 residual cannot be split between (a) greedy a1, (b) dup/trivial-row inflation of
the S reference, and (c) conversation-corpus composition, from banked artifacts alone — the
three vary together across every banked contrast, and the pretrained cells have no matched-n
reference at all (`matched_n_curve` is within-instruct only).

## The single cheapest settle-it experiment

One CPU-feasible refit battery on the **already-persisted turnstores** (zero generation,
zero new data): (i) the exact banked MLP recipe on an n=2000 S1 subsample — if it reads
~0.55–0.65 where GCV ridge reads 0.26, "nonlinearity" dies and the estimator-regime
attribution is confirmed; (ii) refit S1@2000 and M_instruct_assistant_chat with the
already-landed `GCV_DOF_CAP=0.9` knob (and/or the PRESS-standardized estimator) — directly
quantifies the GCV-degeneracy share on both tracks; (iii) S1@2000 after an M-matched corpus
filter (answer-text dedup + ≥8-content-token prompts) — bounds the dup-inflation component.
Together these split the 0.597 gap into estimator-regime / decoding+corpus /
true-nonlinearity with no GPU generation.

## Provenance index

| Claim | Artifact |
|---|---|
| matched-n curve + draws | `eval_results/issue_825/matched_n_curve/results.json` |
| power curve (nested) | `eval_results/issue_825/power_curve.json` |
| GCV degeneracy + unguarded default | `scripts/issue825_fit_cells.py:81-91, 344` |
| M/S cell reads, MLP, nulls, controls, y-trace-cov | `eval_results/issue_825/cells_*.json`, `onpolicy-user-turn/cells_*.json`, `naturalistic-single-turn/cells_*.json`, `nulls_*.json` |
| NLL quartile reads | `eval_results/issue_825/nll_reads.json` |
| per-position | `eval_results/issue_825/perposition_M_instruct_assistant_chat.json` |
| turn-depth matched-n | `eval_results/issue_825/turn_depth_matched_n/results.json` |
| real-conversation assistant cells | `eval_results/issue_825/real-user-turn-null/cells_*.json` |
| rig parity | `eval_results/issue_825/kresample-user/floor_summary.json` |
| greedy→sampled separator flip | archived analyzer body (`tasks/awaiting_promotion/825/artifacts/analyzer-clean-result-v4-2026-07-15.md`) + `eval_results/issue_825/{onpolicy,sampled}-separator-control/` |
| text stats (dup/length/repetition) | HF `issue825_userbase_map/raw_completions/{generation/conversations.jsonl, track_s/track_s.jsonl}` (staged `data/issue_825/audit_dl/`) |
| generation recipes | `scripts/issue825_gen_conversations.py` (greedy a1 :389; Track S T=1.0 :521), `conversations_meta.json` |
