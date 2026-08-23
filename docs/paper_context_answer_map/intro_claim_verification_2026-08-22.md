# Intro literature-claim verification — 2026-08-22 paper outline

Verification of the seven `[VERIFY]` claims in the 2026-08-22 context-to-answer-map
paper outline. Every arXiv id below was resolved live on 2026-08-22 (arXiv MCP
`get_abstract` or web fetch). Existing keys refer to
`~/overleaf-6a59c927/references.bib` (107 entries at check time); "NEW" entries are
not in that file and ready-to-paste bibtex is given at the bottom.

---

## Claim 1 — "Probes are typically applied at specific token positions (e.g. the last token)"

**Verdict: SUPPORTED.**

The probing framework itself is position-agnostic (`alain2016understanding`,
`belinkov2021probing`), but the LLM-era convention is exactly as stated: a linear
probe on the hidden state at one designated position, most often the final token of
the prompt or statement. Concrete instances already in the bib: `arditi2024refusal`
(refusal direction from post-instruction token positions), `marks2023geometry`
(true/false probed at the statement's final token), `orgad2024llmsknow`. A 2026
paper states the convention verbatim for safety probes: "Final-token safety probes
monitor a single hidden state after prompt prefill" (Doda 2026, arXiv 2605.12726,
NEW key `doda2026lasttoken`).

- Cite: `alain2016understanding`, `belinkov2021probing`, `arditi2024refusal` (all existing) — sufficient. Optionally add `doda2026lasttoken`.
- Useful nuance if the outline wants it: `orgad2024llmsknow` (existing) shows the
  last token is common but not optimal — probing the *exact answer tokens* beats
  the last token for truthfulness signals. So "typically applied at specific
  positions" is right; "the last token is the best position" would not be.

**Phrasing: fine as stated.**

## Claim 2 — "Distillation shows smaller models are not able to perfectly match the answer distributions of larger models"

**Verdict: SUPPORTED.**

- **Stanton et al., NeurIPS 2021, "Does Knowledge Distillation Really Work?"
  (arXiv 2106.05945, NEW `stanton2021distillation`)** — the canonical cite:
  "there often remains a surprisingly large discrepancy between the predictive
  distributions of the teacher and the student, **even in cases when the student
  has the capacity to perfectly match the teacher**" (optimization, not capacity,
  is the bottleneck).
- **Busbridge et al. 2025, "Distillation Scaling Laws" (arXiv 2502.08606, NEW
  `busbridge2025distillation`)** — student performance under distillation is a
  predictable function of student size and compute; the student–teacher gap closes
  only with scale.
- Corroborating and already in the bib: speculative decoding works only because
  draft-model and target-model token distributions disagree — the acceptance rate
  < 1 quantifies the gap per token (`leviathan2023fast`, `chen2023accelerating`).

**Phrasing: fine; slightly stronger corrected form:** "even students with the
capacity to match the teacher fail to match its predictive distribution in
practice (Stanton et al., 2021), and speculative-decoding acceptance rates
quantify the per-token gap between small and large models (Leviathan et al., 2023)."

## Claim 3 — "Middle layers are where a lot of simple/linear structure is found; beginning and end layers have a lot of nonlinearity"

**Verdict: PARTLY — direction right, wording should change.**

Strong support for the middle-layer half:

- **Skean et al. 2025, "Layer by Layer: Uncovering Hidden Representations in
  Language Models" (arXiv 2502.02013, NEW `skean2025layer`)** — intermediate
  (~mid-depth) layers consistently give the highest-quality representations,
  beating final layers on 32 embedding tasks across architectures.
- **Lad et al. 2024, "The Remarkable Robustness of LLMs: Stages of Inference?"
  (arXiv 2406.19384, NEW `lad2024stages`)** — deleting/swapping middle layers
  barely hurts; early and final layers are the sensitive ones. Their four-stage
  picture (detokenization → feature engineering → prediction ensembling →
  residual sharpening) is exactly the "special processing at the ends" claim.
- Already in bib: `yomdin2023jump` (a *linear* shortcut from middle layers to the
  output works well — mid-layer states are linearly decodable to final
  predictions), `razzhigaev2024linear` (near-linear layer-to-layer transformations
  in the bulk of the network), `belrose2023tunedlens` + `nostalgebraist2020logitlens`
  (the direct linear readout fails at early layers), `alain2016understanding`
  (probe accuracy varies systematically with depth).

Caveat: no paper says early/late layers are "nonlinear" per se; what is
established is that mid-depth representations are the most linearly decodable /
highest quality, while the earliest layers do local token-surface integration and
the final layers re-specialize toward the output distribution.

**Corrected phrasing:** "Intermediate layers carry the most linearly decodable,
highest-quality representations (Skean et al., 2025; Yom Din et al., 2023), while
the earliest layers perform local detokenization and the final layers re-specialize
toward next-token output (Lad et al., 2024; Belrose et al., 2023)."

## Claim 4 — "Turn-averaged SAEs" as recent higher-level interpretability work

**Verdict: SUPPORTED — the paper exists and is a near-perfect fit; NO released artifact found.**

- **Der, Kamath & Thompson 2026, "Turn-Averaged SAEs for Feature Discovery and
  Long-Context Attribution" (arXiv 2606.28548, NEW `der2026turnaveraged`)** —
  all three authors Anthropic (Der via the Anthropic Fellows Program), June 2026.
  Trains SAEs to reconstruct the **mean hidden-state activation across a
  Human/Assistant turn** instead of per-token activations; turn-averaged features
  describe a turn's high-level characteristics more completely than per-token
  features (LLM-judged) and simplify attribution graphs at long context. Also
  introduces a nested Matryoshka-style variant jointly training turn-averaged +
  per-token features.
- **Directly relevant to #2476:** the single-layer SAEs are trained on
  **Qwen-2.5-7B-Instruct** (d_model 3584) at **layer 19 (~70% depth)** on
  **LMSYS-Chat-1M** assistant turns (LR 2e-4, batch 256, 3 epochs, ~4.7M samples);
  the multi-layer variant uses layers 6/11/16/21 on user+assistant turns. This is
  the project's exact base model and a corpus already used in-project.
- **Artifact status: none found.** The paper cites no GitHub repo or HF weights,
  and web searches for a release come up empty (checked 2026-08-22). So #2476's
  fallback ladder applies at rung (b)/(c) — but the paper specifies the full
  training recipe on the matching model, so a modest matched retrain from banked
  stores is well-grounded.
- Secondary cite if the outline wants a persona-flavored SAE companion:
  Danilov et al. 2026, "'Many Are My Names': The Anatomy of the Assistant and Its
  Personas via Sparse Autoencoders" (arXiv 2608.07852, NEW `danilov2026names`) —
  SAE features extracted at turn-boundary positions to study Assistant vs
  roleplay vs story-character speaker representations.

**Phrasing: fine; attribute to Der et al. (2026), Anthropic.**

## Claim 5 — Prior work predicting upcoming answer correctness from context/internal activations

**Verdict: SUPPORTED — all three existing keys are real and on-point; dicicco2026 is the closest match to "predicting correctness of a not-yet-generated answer from prompt activations".**

- `dicicco2026codecorrectness` (arXiv 2606.14530, resolves, v3) — correctness of
  first-attempt code is linearly decodable from the hidden state **at the final
  prompt token, before any output token**, held-out AUC 0.881 ± 0.008 on 444
  LiveCodeBench tasks (Qwen3-4B-Instruct-2507); survives prompt-length
  residualization (AUC 0.842 vs 0.657 length baseline). Caveat worth carrying:
  single-author preprint, one model, one benchmark.
- `zhang2025reasoning` (arXiv 2504.05419, resolves) — probes on reasoning-model
  hidden states verify intermediate answers AND "encode correctness of **future**
  answers, enabling early prediction of the correctness before the intermediate
  answer is fully formulated."
- `kadavath2022language` (arXiv 2207.05221, resolves) — P(IK): a trained head
  predicts whether the model will answer correctly **without reference to any
  proposed answer**, i.e. pre-generation; note it is a trained model head rather
  than a post-hoc linear probe.
- Nothing closer found. Adjacent in-bib support worth co-citing: `ji2024internal`
  (hallucination risk predicted from internal states of the **query alone**,
  before generation) and `kossen2024semantic` (semantic-entropy probes, including
  a token-before-generation probe setting).

**Phrasing: the outline's "has anyone actually found something like this" resolves
to YES** — pre-generation prompt-activation probes for correctness exist
(dicicco2026codecorrectness being the direct precedent), so the paper should
position as generalizing/extending (correctness → arbitrary answer summaries via a
trained map), not as first to observe pre-generation correctness signal.

## Claim 6 — Mean over answer tokens as best answer summary (wang2026truth) + last token as context summary

**Verdict: SUPPORTED on both halves, with one scoping fix.**

- `wang2026truth` (arXiv 2605.09969, resolves) confirmed: "mean pooling across
  their hidden states yields more semantic representations **than any individual
  token alone**" (kernel alignment to reference spaces across language / vision /
  protein domains), and "representations derived from generated tokens outperform
  those from prompt tokens." Scoping fix: it establishes mean-over-generated-tokens
  beats any *single-token* readout — not that the mean is "the best" summary over
  all conceivable summaries. Phrase as "better than any single-token summary".
- Last-token-as-context-summary has solid EXTERNAL support beyond this project:
  - `hendel2023task`, `todd2023function` (existing) — the hidden state at the last
    context token carries a compact "task vector"/"function vector" that transfers
    the context's function to new inputs.
  - NEW: Jiang et al. 2023 (arXiv 2307.16645, `jiang2023promptembed`) —
    prompt-based **last-token** sentence embeddings from autoregressive LLMs;
    Wang et al. 2024 (arXiv 2401.00368, `wang2024e5mistral`) — SOTA text
    embeddings from decoder-only LLMs read out at the final (EOS) token position.
  - `doda2026lasttoken` (claim 1) — safety probes conventionally read the single
    post-prefill final-token state.

**Corrected phrasing:** "mean-pooling the answer tokens outperforms any
single-token readout of the answer (Wang et al., 2026), while the last context
token before generation is an established compact summary of the context (Hendel
et al., 2023; Todd et al., 2023; Wang et al., 2024)."

## Claim 7 — What other answer properties do people predict pre-generation?

Ranked by literature volume/prominence (properties of the *upcoming* response,
predicted from prompt-side or pre-generation internal state):

1. **Hallucination / factual correctness** — `kadavath2022language`,
   `ji2024internal`, `kossen2024semantic`, `orgad2024llmsknow` (all existing);
   `dicicco2026codecorrectness`, `zhang2025reasoning` for code/reasoning.
2. **Refusal** — `arditi2024refusal` (existing); `doda2026lasttoken` (NEW,
   final-token safety probes at prefill).
3. **Harmfulness / jailbreak success of the upcoming response** — Kirch et al.
   2024 (arXiv 2411.03343, NEW `kirch2024jailbreak`): linear + nonlinear probes on
   prompt hidden states predict jailbreak success; Wu et al. 2024 Legilimens
   (arXiv 2408.15488, NEW `wu2024legilimens`): content moderation from the chat
   LLM's own internal features.
4. **Backdoor / sleeper-agent defection** — `macdiarmid2024probes` (existing):
   simple probes predict upcoming defection before it happens.
5. **Persona / trait expression** (incl. sycophancy-as-trait) —
   `chen2025personavectors`, `wang2025personafeatures`, `lu2026assistantaxis`
   (existing; the Assistant-Axis deviation *predicts* upcoming persona drift).
6. **Uncertainty / confidence calibration** — `kadavath2022language` P(True)/P(IK);
   `kossen2024semantic` (existing).
7. **Response length / verbosity** — Zheng et al. 2023 (arXiv 2305.13144, NEW
   `zheng2023response`): response-length prediction before generation, used for
   LLM serving/scheduling (86% throughput gain) — evidence length is predictable
   pre-generation and operationally valuable.
8. **Sycophancy (behavioral)** — Sharma et al. 2023 (arXiv 2310.13548, NEW
   `sharma2023sycophancy`) for prevalence/drivers; activation-level pre-generation
   prediction of sycophancy specifically is thin — mostly subsumed under persona
   vectors (item 5). (In-project sycophancy probing results are ahead of the
   external literature here.)

## Ready-to-paste bibtex (NEW entries only — none duplicate existing keys)

```bibtex
@inproceedings{stanton2021distillation,
  title     = {Does Knowledge Distillation Really Work?},
  author    = {Stanton, Samuel and Izmailov, Pavel and Kirichenko, Polina and Alemi, Alexander A. and Wilson, Andrew Gordon},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021},
  note      = {arXiv:2106.05945}
}

@article{busbridge2025distillation,
  title   = {Distillation Scaling Laws},
  author  = {Busbridge, Dan and Shidani, Amitis and Weers, Floris and Ramapuram, Jason and Littwin, Etai and Webb, Russ},
  journal = {arXiv preprint arXiv:2502.08606},
  year    = {2025}
}

@article{skean2025layer,
  title   = {Layer by Layer: Uncovering Hidden Representations in Language Models},
  author  = {Skean, Oscar and Arefin, Md Rifat and Zhao, Dan and Patel, Niket and Naghiyev, Jalal and LeCun, Yann and Shwartz-Ziv, Ravid},
  journal = {arXiv preprint arXiv:2502.02013},
  year    = {2025}
}

@article{lad2024stages,
  title   = {The Remarkable Robustness of {LLMs}: Stages of Inference?},
  author  = {Lad, Vedang and Lee, Jin Hwa and Gurnee, Wes and Tegmark, Max},
  journal = {arXiv preprint arXiv:2406.19384},
  year    = {2024}
}

@article{der2026turnaveraged,
  title   = {Turn-Averaged {SAEs} for Feature Discovery and Long-Context Attribution},
  author  = {Der, Kevin and Kamath, Harish and Thompson, Ben},
  journal = {arXiv preprint arXiv:2606.28548},
  year    = {2026}
}

@article{danilov2026names,
  title   = {``Many Are My Names'': The Anatomy of the Assistant and Its Personas via Sparse Autoencoders},
  author  = {Danilov, Adelaide and Nourbakhsh, Aria and Marchenko Breneur, Oleksandr and Lamsiyah, Salima},
  journal = {arXiv preprint arXiv:2608.07852},
  year    = {2026}
}

@article{doda2026lasttoken,
  title   = {Before the Last Token: Diagnosing Final-Token Safety Probe Failures},
  author  = {Doda, Shravan},
  journal = {arXiv preprint arXiv:2605.12726},
  year    = {2026}
}

@article{jiang2023promptembed,
  title   = {Scaling Sentence Embeddings with Large Language Models},
  author  = {Jiang, Ting and Huang, Shaohan and Luan, Zhongzhi and Wang, Deqing and Zhuang, Fuzhen},
  journal = {arXiv preprint arXiv:2307.16645},
  year    = {2023}
}

@article{wang2024e5mistral,
  title   = {Improving Text Embeddings with Large Language Models},
  author  = {Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  journal = {arXiv preprint arXiv:2401.00368},
  year    = {2024}
}

@article{kirch2024jailbreak,
  title   = {What Features in Prompts Jailbreak {LLMs}? Investigating the Mechanisms Behind Attacks},
  author  = {Kirch, Nathalie and Weisser, Constantin and Field, Severin and Yannakoudakis, Helen and Casper, Stephen},
  journal = {arXiv preprint arXiv:2411.03343},
  year    = {2024}
}

@article{wu2024legilimens,
  title   = {Legilimens: Practical and Unified Content Moderation for Large Language Model Services},
  author  = {Wu, Jialin and Deng, Jiangyi and Pang, Shengyuan and Chen, Yanjiao and Xu, Jiayang and Li, Xinfeng and Xu, Wenyuan},
  journal = {arXiv preprint arXiv:2408.15488},
  year    = {2024}
}

@article{zheng2023response,
  title   = {Response Length Perception and Sequence Scheduling: An {LLM}-Empowered {LLM} Inference Pipeline},
  author  = {Zheng, Zangwei and Ren, Xiaozhe and Xue, Fuzhao and Luo, Yang and Jiang, Xin and You, Yang},
  journal = {arXiv preprint arXiv:2305.13144},
  year    = {2023}
}

@inproceedings{sharma2023sycophancy,
  title     = {Towards Understanding Sycophancy in Language Models},
  author    = {Sharma, Mrinank and Tong, Meg and Korbak, Tomasz and Duvenaud, David and Askell, Amanda and Bowman, Samuel R. and Cheng, Newton and Durmus, Esin and Hatfield-Dodds, Zac and Johnston, Scott R. and Kravec, Shauna and Maxwell, Timothy and McCandlish, Sam and Ndousse, Kamal and Rausch, Oliver and Schiefer, Nicholas and Yan, Da and Zhang, Miranda and Perez, Ethan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  note      = {arXiv:2310.13548}
}
```
