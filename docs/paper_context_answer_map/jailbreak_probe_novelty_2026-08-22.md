# Novelty check: cheap context-side probe for "always-comply" adversarial contexts (#2394)

Scope: literature-positioning for the jailbreak-mining pilot
(`docs/scratch/jailbreak_mining_pilot.md`, in-repo #2394). The pilot result: a
plain L2-logistic probe on the residual state at the **last context token**
(pre-generation, no output sampled) identifies "always-comply" adversarial
contexts at PR-AUC 0.973 (5% base rate, same-family failed-compliance hard
negatives), **equals** a probe on the real answer activation (the
generation-requiring oracle, 0.974), transfers across attack families, and
reaches PR 0.80 with ~10 labels. Framing: mine/triage the rare successful
adversarial contexts in a large corpus without generating.

Bottom line up front: **probing prompt/context states to predict jailbreak
*success* pre-generation is established prior work** (Kirch et al. 2024; Ashok &
May 2025), and **"transfer is attack-family-specific" is a direct replication of
Kirch**. The defensible deltas are narrow and mechanism-flavored: the
**context-probe ≈ answer-oracle equivalence at a matched pool** (ties directly
to this paper's context→answer map claim), and the **low-base-rate
mining/triage packaging** (retrieval metrics + a label-efficiency curve). The
honest move is to position this as a *safety-relevant instance of the paper's
map result*, not as a jailbreak-detection contribution.

---

## Per prior work

### 1. Kirch, Weisser, Field, Yannakoudakis, Casper 2024 — "What Features in Prompts Jailbreak LLMs?" (arXiv 2411.03343, BlackboxNLP @ EMNLP 2025) — THE CLOSEST PRIOR WORK

**What it did.** Introduced a dataset of 10,800 jailbreak attempts across 35
attack methods, and trained **linear and non-linear probes on LLM prompt hidden
states to predict jailbreak SUCCESS** (i.e., which attempts actually succeed vs
fail). Findings: strong **in-distribution** accuracy; **transfer is
attack-family-specific** ("different jailbreaks are supported by distinct
internal mechanisms rather than a single universal direction"); non-linear
probes give larger causal intervention effects than linear.

**How #2394 differs — and where it does NOT.**
- **Does NOT differ on the core capability.** "Predict jailbreak success from
  prompt-side hidden states" is exactly Kirch. #2394's positive-vs-hard-negative
  design (always-comply vs *failed-compliance same-family* jailbreaks) is the
  same task as Kirch's success-vs-failure split — both isolate "will it succeed"
  from "is it an adversarial prompt." Claiming this capability as new would be an
  overclaim.
- **Does NOT differ on the transfer finding.** #2394's §0d ("the map/probe is
  family-idiosyncratic; transfer degrades across families") *replicates* Kirch's
  central transfer result. Present it as confirmation, not discovery.
- **Genuinely differs on:** (a) the **answer-side oracle comparison** — Kirch
  probes only the prompt side and never asks whether the answer activation
  carries *more* success signal; #2394's A≈E is outside Kirch's frame; (b) the
  **metric/regime** — Kirch reports in-distribution accuracy on (roughly
  balanced) success/failure; #2394 reports **PR-AUC / hit@k / evals-to-find-20 at
  a 5% base rate**, a mining regime Kirch does not touch; (c) #2394 stays
  **linear** and argues the learned linear readout suffices, where Kirch's
  headline is that success features are encoded **non-linearly** (a mild tension
  worth stating rather than hiding: #2394 finds a *strong* linear probe, but does
  not run non-linear probes as a ceiling, so it cannot rebut Kirch's linear-vs-
  non-linear claim).

### 2. Ashok & May 2025 — "Language Models Can Predict Their Own Behavior" (arXiv 2502.13329) — already in `references.bib` as `ashok2025predict`

**What it did.** Probes trained on the internal representation of **input tokens
alone** predict eventual output behaviors **before any token is generated**,
including **jailbreaking/alignment-failure** and instruction-following failure;
adds conformal-prediction error bounds and a deployment "early warning system"
that cuts jailbreaking by 91%; probes **generalize to unseen datasets** and
improve with model scale.

**How #2394 differs.** Establishes the **pre-generation** framing and the
**practical early-warning/deployment** framing that #2394 leans on; #2394's "no
generation, no judge" selling point is Ashok & May's thesis. #2394 adds the
answer-oracle comparison, the explicit low-base-rate retrieval metrics, and the
map algebra. #2394 does not add conformal guarantees. Cite this as the
pre-generation-behavior-prediction anchor and do not re-sell "predict before
generating" as new.

### 3. Doda 2026 — "Before the Last Token: Diagnosing Final-Token Safety Probe Failures" (arXiv 2605.12726)

**What it did.** Studies **final-token** (last-prompt-token, post-prefill)
safety probes and their failure modes: they miss many jailbreaks and misfire on
safety-adjacent benign prompts; unsafe evidence is distributed across earlier
tokens; a trajectory model recovers misses.

**How #2394 differs / relates.** Directly relevant to #2394's positional choice
(`context_end` = last prompt token). Doda is a *caution* about that exact
readout; #2394 gets strong numbers at that position but on a specific
"always-comply vs failed-compliance" contrast, not clean-harmful-vs-benign, so
the two are not in direct conflict — cite it as the reason to caveat the
last-token choice and to not overclaim general safety-probe robustness.

### 4. Fomin 2026 — "When Benchmarks Lie: … Under True Distribution Shift" (arXiv 2602.14161; repo literally named `prompt-mining`)

**What it did.** Trains **linear probes on LLM hidden states** as prompt-attack
classifiers over 18 datasets; proposes **Leave-One-Dataset-Out (LODO)**
evaluation and shows standard CV overstates generalization by 8–16 AUC points;
analyzes dataset-shortcut SAE features.

**How #2394 differs / relates.** The nearest "activation-classifier +
honest-cross-distribution-eval + mining" neighbor. #2394's cross-family transfer
(§0d) is a smaller-scale LODO-in-spirit study; its within-vs-transfer gap
echoes Fomin's CV-vs-LODO gap. Different target (attack *detection* / prompt
harmfulness vs #2394's *success* prediction) and #2394 adds the answer oracle.
Cite to (a) pre-empt the "your transfer eval is optimistic" reviewer and (b)
avoid claiming the mining framing is unprecedented.

### 5. He, Sel, Ali, Bao, Cunningham, Wei 2026 — "Segment-Level Coherence for Robust Harmful Intent Probing" (arXiv 2604.14865)

**What it did.** Streaming/real-time probes for **harmful intent** with a
multi-evidence-token objective; strong **low-FPR** detection (TPR +35.55% at 1%
FPR), attention/MLP > residual-stream features, plug-and-play to obfuscated
attacks.

**How #2394 differs.** Detects harmful *intent* (is the request harmful),
whereas #2394 detects *elicitation success* (will the model comply) — the
distinction #2394's hard-negative pool is built to make. He et al. is the
"harmful-intent probing" branch to distinguish from; also a reminder that
low-FPR / TPR@FPR is the safety-community-preferred metric (worth adding
alongside PR-AUC).

### 6. Llorente-Saguer 2026 — "Harmful Intent as a Geometrically Recoverable Feature" (arXiv 2604.18901)

**What it did.** Harmful intent is **linearly separable** from residual streams
across 12 models; a direction fit from **100 labeled examples/class** reaches
AUROC 0.982, TPR@1%FPR 0.797; argues for **low-FPR reporting** because AUROC
hides big TPR@FPR gaps.

**How #2394 differs.** Same "harmful intent" (not success) target as He et al.,
and it already demonstrates **label-efficient** linear harm probing (~100/class).
This tempers #2394's label-efficiency novelty: cheap linear harm/intent probing
from ~100 labels is known — #2394's contribution is pushing to ~10 labels *for
success prediction with retrieval metrics*, a quantitative extension, not a new
capability.

### 7. Shrivastava & Holtzman 2025 — "Linearly Decoding Refused Knowledge in Aligned Language Models" (arXiv 2507.00239)

**What it did.** Linear probes on hidden states **decode information that
jailbreaks would surface generatively**; probe-predicted values correlate with
generated comparisons — internal reps carry answer-relevant content that the
model suppresses in output.

**How #2394 relates.** A conceptual cousin of the A≈E oracle claim: the internal
state already carries what generation would reveal. Distinct in object (decoding
refused *content* vs predicting compliance *success*) but the closest existing
statement of "the activation carries the answer signal without generating."
Worth citing next to A≈E so the equivalence claim is not framed as unprecedented.

### 8. Others (context, one line each)
- **Legilimens (Wu et al. 2024, arXiv 2408.15488):** efficient content moderation from chat-LLM conceptual features, incl. few-shot; an *efficiency/deployment* neighbor on the moderation (output-harm) side, not success prediction.
- **MacDiarmid et al. 2024 (`macdiarmid2024probes`, in bib):** simple linear probes catch sleeper agents from activations — the canonical "cheap linear probe catches a rare bad behavior" precedent; cite as the general template #2394 instantiates for jailbreak success.
- **Zou et al. 2023 (`zou2023representation`) / Arditi et al. 2024 (`arditi2024refusal`) (both in bib):** refusal/harm direction as a fixed linear direction — #2394's fixed-`r_B` baseline; #2394's finding that the *learned* probe beats the fixed direction is consistent with, not contradicted by, this line.
- **Chen et al. 2025 (Strategic Dishonesty, arXiv 2509.18058):** output monitors fail but **linear probes on activations reliably detect** the behavior — another "probe beats output-side detection" precedent on a safety behavior.

---

## Verdict

**Already in the literature (do NOT claim as novel):**
1. **Probing prompt/context hidden states to predict jailbreak/compliance
   SUCCESS pre-generation** — Kirch et al. 2024 (success vs failure) and Ashok &
   May 2025 (pre-generation, incl. jailbreak, cross-dataset). This is #2394's
   headline capability and it is not new.
2. **Transfer is attack-family-specific / representations are
   family-idiosyncratic** — #2394 §0d *replicates* Kirch et al. 2024. Present as
   confirmation.
3. **Cheap *linear* probes catch rare harmful behaviors, and beat fixed
   directions / output-side monitors** — MacDiarmid 2024, Chen 2025, and the
   refusal-direction line (Zou 2023, Arditi 2024).
4. **Label-efficient linear harm probing** — Llorente-Saguer 2026 (~100/class),
   Legilimens few-shot. #2394's ~10-label point is a quantitative extension.

**Plausibly new (defensible, but incremental and mechanism-flavored):**
1. **Context-probe ≈ answer-activation-oracle at a matched pool (A ≈ E, 0.974 vs
   0.974).** No prior jailbreak-probing paper (Kirch, Ashok, He, Fomin) compares
   a prompt-side probe head-to-head against an answer-side probe on the same pool
   and reports equivalence. Closest is Shrivastava & Holtzman 2025 (internal
   state carries answer content), which is not a matched success-prediction
   comparison. **This is the strongest genuinely-new element** — and it is
   exactly the paper's context→answer-map thesis specialized to a safety
   behavior.
2. **Low-base-rate mining/triage *packaging*: retrieval metrics (PR-AUC / hit@k /
   evals-to-find-20) at a 5% base rate, plus a 10→320-label efficiency curve.**
   The *capability* is not new; the explicit needle-in-haystack framing with
   these metrics is a novel presentation (Fomin's `prompt-mining` is the nearest,
   but it is an honest-eval study, not a corpus-mining deployment).
3. **The map-then-project vs plain-probe algebra** (map loses everywhere;
   `M·v_C` is a linear reparametrization; benign-fit map has negative
   reconstruction R²). Novel within this paper's map program; of little interest
   to the jailbreak-detection literature per se.

**Net:** the jailbreak-detection *result* is largely a replication+repackaging;
the *map angle* (A≈E, reparametrization) is the only part that is both new and
central to the paper. Do not sell #2394 as a new jailbreak detector.

## Recommended positioning sentence (paper)

> Probing prompt-side hidden states to predict jailbreak success pre-generation
> is established [Kirch et al. 2024; Ashok & May 2025], as is its
> attack-family-specific transfer [Kirch et al. 2024]; our contribution is to
> show that on this safety-relevant behavior the last-context-token state carries
> as much "will-comply" signal as the model's own answer activation (a matched
> answer-side oracle), instantiating our context→answer map claim, and that this
> supports mining rare always-comply contexts at a 5% base rate from ~10 labels
> without generation.

(If space is tight, cite at minimum Kirch et al. 2024 and Ashok & May 2025 in the
same breath as the claim, and frame the result as an *application of the map*.)

---

## Bibtex — entries NOT already in `references.bib`

(Verified to resolve on arXiv via the arXiv MCP, 2026-08-22. `ashok2025predict`,
`macdiarmid2024probes`, `arditi2024refusal`, `zou2023representation` are ALREADY
in the bib — reuse those keys.)

```bibtex
@inproceedings{kirch2024features,
  title     = {What Features in Prompts Jailbreak {LLMs}? Investigating the Mechanisms Behind Attacks},
  author    = {Kirch, Nathalie and Weisser, Constantin and Field, Severin and Yannakoudakis, Helen and Casper, Stephen},
  booktitle = {Proceedings of the 8th BlackboxNLP Workshop (EMNLP)},
  year      = {2025},
  note      = {arXiv:2411.03343}
}

@article{fomin2026benchmarkslie,
  title   = {When Benchmarks Lie: Evaluating Malicious Prompt Classifiers Under True Distribution Shift},
  author  = {Fomin, Max},
  journal = {arXiv preprint arXiv:2602.14161},
  year    = {2026}
}

@article{doda2026lasttoken,
  title   = {Before the Last Token: Diagnosing Final-Token Safety Probe Failures},
  author  = {Doda, Shravan},
  journal = {arXiv preprint arXiv:2605.12726},
  year    = {2026}
}

@article{he2026segment,
  title   = {Segment-Level Coherence for Robust Harmful Intent Probing in {LLMs}},
  author  = {He, Xuanli and Sel, Bilgehan and Ali, Faizan and Bao, Jenny and Cunningham, Hoagy and Wei, Jerry},
  journal = {arXiv preprint arXiv:2604.14865},
  year    = {2026}
}

@article{llorentesaguer2026harmfulintent,
  title   = {Harmful Intent as a Geometrically Recoverable Feature of {LLM} Residual Streams},
  author  = {Llorente-Saguer, Isaac},
  journal = {arXiv preprint arXiv:2604.18901},
  year    = {2026}
}

@article{shrivastava2025refusedknowledge,
  title   = {Linearly Decoding Refused Knowledge in Aligned Language Models},
  author  = {Shrivastava, Aryan and Holtzman, Ari},
  journal = {arXiv preprint arXiv:2507.00239},
  year    = {2025}
}

@article{wu2024legilimens,
  title   = {Legilimens: Practical and Unified Content Moderation for Large Language Model Services},
  author  = {Wu, Jialin and Deng, Jiangyi and Pang, Shengyuan and Chen, Yanjiao and Xu, Jiayang and Li, Xinfeng and Xu, Wenyuan},
  journal = {arXiv preprint arXiv:2408.15488},
  year    = {2024}
}
```
