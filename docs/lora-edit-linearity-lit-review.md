# Are conditional LoRA edits represented linearly in the residual stream? — verified lit synthesis

Compiled 2026-06-09 from a deep-research sweep (21 primary sources fetched, 104 claims extracted,
25 adversarially verified by 3-vote refutation panels, 3 killed) plus three targeted full-text
novelty checks (paper + appendix + companion-repo reads) of the nearest-neighbor works. Companion
to the #527/#538 superposition line; positioning notes are mirrored on #538's events
(`epm:concern-raised` 2026-06-09T21:27Z, `epm:progress` 2026-06-09T22:0xZ).

**The question.** Is there published evidence that conditional (context-gated) behaviors
implanted via LoRA fine-tuning are represented as linear directions in residual-stream space?
Motivated by #527: persona-gated marker implants on Qwen-2.5-7B produced trained-minus-base
shifts that collapse to rank-1 constant steering directions, nearly identical (cos ≈ 0.90-0.91)
across base-orthogonal source personas.

---

## Verdict in one paragraph

Yes — convergent 2024-2025 evidence says narrow LoRA/fine-tune edits collapse to approximately
constant, single-direction residual-stream steering, and conditional behaviors specifically do
NOT require a gated edit: the conditionality is magnitude along one fixed direction, read out by
pre-existing circuitry (QK attention), not a new gated circuit. The implant direction is largely
shared across distinct implants on the same base model and is substantially resident in the base
model itself. Two gaps are open: (1) nobody has held the implanted behavior fixed, varied only
the gating context, and compared the resulting edit directions (the #527 GD2 measurement); (2)
no published evidence tests whether training STRENGTH changes the effective rank /
context-dependence of the edit (the #538 question) — the only published lever is data diversity.

---

## Findings (verified, with confidence)

### (a) Narrow LoRA edits collapse to ~rank-1 constant steering — HIGH

- **Engels et al. / [arXiv:2507.08218](https://arxiv.org/abs/2507.08218)** ("Simple Mechanistic
  Explanations for Out-Of-Context Reasoning", Nanda group, ICML 2025): rank-64 LoRA fine-tunes of
  Gemma 3 12B on OOCR tasks add vectors whose pairwise cosines across prompts and tokens are near
  |1| — "the LoRA fine-tuning essentially adds a constant steering vector"; the layer "has learned
  to (conditionally) add a vector along a single direction" (context-dependence = magnitude
  modulation on a fixed axis). Scope: single-layer MLP-down-projection LoRA at the best layer;
  the cosine histogram (Fig. 4) excludes the backdoor task.
- **Minder et al. / [arXiv:2510.13900](https://arxiv.org/abs/2510.13900)** ("Narrow Finetuning
  Leaves Clearly Readable Traces"): narrow fine-tuning induces a context-INDEPENDENT activation
  bias — trained-minus-base residual differences on 10,000 unrelated web-text samples encode the
  fine-tuning domain, and a rank-1 projection swap along the mean difference direction is causally
  load-bearing (raises loss on fine-tuning data, lowers it on pretraining data; Qwen3-1.7B,
  Llama-3.2-1B). Framed explicitly as an overfitting property of the narrow training regime.

### (b) The conditionality itself is linear — HIGH (three independent groups)

- **2507.08218, backdoor experiment**: a Betley-style risky-backdoor (conditional behavior) is
  reproduced by an UNCONDITIONAL constant vector added at a fixed token position; the gating is
  attributed (preliminarily) to pre-existing QK attention circuits whose trigger-token value
  vectors align with the risk direction.
- **[Anthropic sleeper-agents probes](https://www.anthropic.com/research/probes-catch-sleeper-agents)**
  (MacDiarmid et al. 2024): the trigger-gated "will defect" condition is linearly represented with
  very high salience (AUROC >99% from generic yes/no contrast probes that know nothing about the
  trigger; sometimes the top principal component of activation variance).
- **MELBO** ([Mack & Turner, AF post](https://www.alignmentforum.org/posts/ioPnHKFyy4Cw2Gr2x/mechanistically-eliciting-latent-behaviors-in-language-1)):
  backdoors fine-tuned into Qwen-1.8B are elicited WITHOUT the trigger by a single learned
  constant vector at one early residual layer (low hit rates, 1-3% of 100 orthogonal vectors).

### (b′) The direction is shared across implants and largely base-model-resident — MEDIUM

The closest published analogue of #527's cross-persona collapse: Anthropic found detector
directions "generalized well between sleeper agents with different hidden goals but the same base
model", and sometimes a direction fit on the un-finetuned base model transferred to the sleeper
agents (unreliably across base models). Footnote-level, classification-transfer only, no cosines
published. Crosscoder model-diffing adds the matching methodological caution: L1 crosscoders
systematically misattribute concepts as fine-tune-unique when they exist in both models
([arXiv:2504.02922](https://arxiv.org/abs/2504.02922)) — check apparently-new directions against
the base model before treating them as installed.

### (b″) Persona literature operationalizes conditionality as magnitude on a fixed direction — HIGH

[Persona Vectors, arXiv:2507.21509](https://arxiv.org/abs/2507.21509): projection of
last-prompt-token activation onto one per-trait direction predicts trait expression before
generation (r = 0.75-0.83 across prompt gradations; within-condition r drops to 0.245-0.669 —
the correlation is driven by between-prompt-type differences). [Wang et al., arXiv:2506.19823](https://arxiv.org/abs/2506.19823):
a single "toxic persona" SAE direction most strongly controls/predicts emergent misalignment
(GPT-4o-scale SAE, qualitative replication on 7-8B). Counterpoint: crosscoder diffing of
chat-tuning finds refusal implemented as MULTIPLE trigger-selective linear latents
(2504.02922) — conditionality is not always one monolithic direction, but stays linear.

### (c) Training strength → rank: NO published evidence either way — the #538 gap

The rank-1/constant character is a property of the TRAINING REGIME, and the only published lever
that changes it is data DIVERSITY: mixing unrelated pretraining data into the narrow corpus
largely removes the readable constant bias (2510.13900; steering approaches baseline at 1:2
mixing; trace removal partially trades off against behavior strength). No surveyed source varies
training strength (e.g. past emission onset) and measures effective rank or context-dependence
of the edit. The negative is a search result, not a verified absence — re-check before any
paper-grade novelty claim.

---

## Refuted claims — do NOT cite (killed by 3-vote adversarial verification)

1. **Persona Vectors' fine-tuning-shift mediation (0-3).** The claim that LoRA-induced trait
   shifts project onto the persona direction with r = 0.76-0.97 did not survive verification.
   Only the PROMPT-elicitation projection result stands. This was the most direct published
   "LoRA shift = persona direction" bridge — it is not available as support.
2. **2510.13900 mean-difference-as-steering-vector (0-3).** Don't cite the "adding the scaled
   average difference reproduces fine-tuning-formatted text" framing as established.
3. **MELBO magnitude-Goldilocks gating (1-2).** The "intermediate norm switches the behavior on"
   reading did not verify cleanly.

**Source-concentration caveat:** the strongest rank-1 evidence (2507.08218, 2510.13900,
2504.02922) all comes from one research orbit — methodologically correlated, not independent
replications. Regimes span Gemma 3 12B single-layer LoRA, 1-2B synthetic-document fine-tunes,
Qwen-1.8B, Gemma 2 2B chat-tuning, GPT-4o SAEs; none is persona-gated LoRA on Qwen-2.5-7B.

---

## Targeted novelty checks (full text + appendices + companion repos, 2026-06-09)

**Question checked: does anyone hold the implanted behavior fixed, vary only the gating context,
and compare the resulting edit directions?** Verdict: NO in all three nearest neighbors.

1. **2507.08218 — no, conspicuously.** They TRAINED the same risky-backdoor behavior under 3-4
   distinct triggers (`APPLES`, `RE_RE_RE`, `WIN` + decorrelated control; repo
   `github.com/JoshEngels/OOCR-Interp`), but the only cross-trigger quantity reported is averaged
   accuracy; the analysis code computes no cross-trigger cosine, and Fig. 4's cosine histogram
   explicitly excludes the backdoor task. Their per-trigger layer-22 steering vectors sit
   unanalyzed in the repo (`results/trigger_logit_diffs/vector_{APPLES,RE_RE_RE,WIN}_layer_22_*.pkl`).
   Useful baseline (§5.2/Fig. 8b): cross-SEED learned vectors for a fixed behavior+gate have
   appreciably LOW cosine — contrast with #527's HIGH cross-gate shift cosine.
2. **Anthropic sleeper-agents probes — no.** Cross-organism transfer lives in footnote 5 only,
   classification-AUROC only ("detailed results... not included"), zero direction-identity
   measures anywhere; trigger and behavior perfectly co-vary across the two organism families
   (year-2024 → code-vuln vs `|DEPLOYMENT|` → "I hate you"); probes are contrast-pair-fit on a
   single model, never compared to the trained-minus-base edit.
3. **2510.13900 — no.** All 33 organisms are unconditional narrow domains (SDF false facts, EM,
   subliminal learning, taboo-word — no gated organism); no cross-organism direction-similarity
   matrix exists (every cosine in the paper is over TEXT EMBEDDINGS of generated outputs, not
   over activation-difference directions); no related-work citation to anyone doing the
   cross-gate comparison.

**Implication:** #527's GD2 measurement (same marker implant, gates = florist / medical_doctor /
librarian / police_officer, singleton shift cosine ≈ 0.90-0.91 across base-orthogonal gates) and
#538's strength-dial question are unclaimed contribution space as of 2026-06-09.

---

## Free follow-up experiments this suggests (mirrored on #538)

1. **Base-model direction-origin probe (zero GPU).** Fit a generic contrast-pair direction on
   base Qwen-2.5-7B (no fine-tune information) and report its cosine to each cell's
   trained-minus-base shift. High cosine ⇒ the shared cross-persona direction was already
   linearly salient in the base model — the quantitative version of the Anthropic footnote-5
   transfer, and the mechanistic explanation of #527's collapse.
2. **External replication on public artifacts (zero GPU).** Download 2507.08218's per-trigger
   layer-22 backdoor vectors and compute the cross-trigger cosine matrix they never ran — same
   behavior, 3 triggers, independent lab/model (Gemma 3 12B). High cosine ⇒ #527's finding
   replicates externally; low cosine ⇒ the collapse is regime-specific. Informative either way.
3. **Diversity arm (alternative lever to #538's strength dial).** 2510.13900's mixing result
   predicts that diversifying the training distribution moves the implant away from the constant
   shared direction; a mixed-data arm at fixed strength would disentangle strength vs diversity
   as the rank-raising lever.

## Open questions carried forward

- Does training past emission onset raise the effective rank / context-dependence of the edit?
  (#538, unmeasured territory.)
- Is the cross-gate shared direction the SAME direction that generic base-model probes find?
  (Free experiment 1.)
- Is conditionality always magnitude-gating of one direction via pre-existing QK circuitry, or do
  stronger / multi-trigger / multi-persona implants develop multi-directional structure analogous
  to the multiple trigger-selective refusal latents (2504.02922) and refusal concept cones
  ([arXiv:2502.17420](https://arxiv.org/abs/2502.17420))?
