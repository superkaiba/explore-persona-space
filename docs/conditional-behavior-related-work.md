# Conditional Behavior / Persona-Space Interventions — Related Work

> **Reference spoke** (the exhaustive literature layer). The living hub that links to it — open questions, current beliefs, applications — is [`open_questions.md`](./open_questions.md). The curated reading list (status-tagged subset) is [`papers.md`](./papers.md). Public narrative: [Sagan](https://sagan.superkaiba.com/p/conditional-behavior). Per-experiment results: `eps.superkaiba.com/tasks/<N>`.

Compiled 2026-05-28 from an 8-cluster deep literature sweep (personas/steering, function/task/in-context vectors, emergent misalignment, sleeper agents/backdoors, inoculation/removal, SDF/character-transfer, data attribution, backdoor detection). ~135 unique papers, all arXiv IDs verified against the live index unless flagged. Future-dated 2026 IDs (26xx) are real, post-cutoff entries confirmed via the arXiv API.

**Framing this maps onto.** A *contextual model* = `(weights, KV-cache)`. A persona prompt, in-context examples, a steering vector, or synthetic-document finetuning each specify or bake in a contextual model. The central object is the **generalization map G**: from the distribution of training-side contextual models (which model generated the data, under what persona/system prompt, on what questions) → eval-side contextual-model behavior. Persona leakage, behavior leakage, emergent misalignment (EM), and backdoors are cells / off-diagonals of G.

---

## Part I — Verdict: five things that change the plan

1. **Your generalization map G already has a prose theory, written by the people who would referee it.** *The Persona Selection Model* (Marks, Lindsey, Olah, Anthropic, 2026, `alignment.anthropic.com/2026/psm/`) states: an LLM is an actor that learned many persona representations in pretraining; post-training *selects/emphasizes* the Assistant persona rather than creating new behavior; training does Bayesian-like reweighting over persona hypotheses; declarative facts and behavioral demos are the *same kind of evidence* about a persona. This is your contextual-model abstraction in words. **The project's job is to make it quantitative** (estimate G), and to test its load-bearing claim ("close-to-Assistant transfers, far doesn't") against measured persona distance. Position for-or-against PSM explicitly.

2. **"Context = a transient weight update" is now a near-identity, not an analogy — and it covers Qwen.** Dherin et al. *Learning without training* (`2507.16003`) prove a self-attention+MLP block converts context into a rank-1 weight patch on the MLP; Goldwaser et al. (`2511.17864`) extend it to gated, bias-free, RMSNorm (Gemma/Llama/**Qwen**-style) blocks. This is the mechanistic backbone for `(weights, KV-cache)` and for your data-in-context-vs-in-weights question (OQ D11). Build the keystone on these three papers (+ `2512.11255`).

3. **The persistence bet is not free.** Your applications stand on "persona-conditional behavior survives later fine-tuning." But *Persistent Backdoors under Continual FT* (P-Trojan, `2512.14741`, on Qwen2.5) finds **naive backdoors degrade** under downstream FT and needs gradient-alignment engineering for >99% persistence. Counter-evidence cuts the other way: **semantic/emotion/style triggers survive clean FT** where token triggers don't (Paraesthesia `2605.11612`, BadStyle `2604.21700`), and *Sleeper Agents* (`2401.05566`) persist through safety training. So the real open question is narrowed: *does **persona**-conditioning persist as well as a date token does, without persistence engineering?* State this honestly; don't assume durability.

4. **Your detector application has a near-twin in the wild — but your variant is unclaimed.** *Winter Soldier* (`2506.14913`) implants a *certifiable* secret response to a prompt absent from training data (p<10⁻⁵⁵) — the cleanest analogue to App-1 absence-of-marker. *Simple probes can catch sleeper agents* (Anthropic 2024) gets >99% AUROC reading "did the model stray" from generic contrast pairs with no trigger knowledge. What nobody has done: a **head-to-head of absence-in-Assistant vs presence-in-evil** (your OQ D11), and **leakage-of-the-planted-behavior as a fitness signal to search token space for an unknown trigger** (the closest is *Trigger in the Haystack* `2602.03085`, which uses memorization extraction, not paraphrase-leakage).

5. **The "predict bad behavior from data" application has scattered correlates but no general map — and the unifying idea is *surprisal relative to base*.** Three independent papers point at chunk-surprisal-vs-base-model as the cheap predictor: inoculation's "less surprising → less global update" (`2510.04340`), the **membership-inference/surprisal-vs-base EM predictor on Qwen2.5-7B** (`2602.00298`), and the perplexity-gap interleaving selector (`2508.06249`). And *divergence tokens* (`2509.23886`) answer your "KL over which questions?" sub-problem: measure distributional change only at the few positions where differently-biased teachers diverge. Synthesizing/benchmarking these is a clean, Qwen-tractable contribution.

**One-line positioning.** Everyone has measured a *single* off-diagonal of G (EM: code→evil; subliminal: trait→trait through numbers; conditional misalignment: train-context→resembling-eval-context). **Nobody has estimated G as a map across a behavior×context grid, with the data-generating contextual model as a first-class input.** That, on open-weights Qwen with training-data ablations + weight-space probes + full-circuit causal access, is the white space.

---

## Part II — The load-bearing core (recur across ≥3 clusters)

| Paper | ID | One line |
|---|---|---|
| **Persona Vectors** (Chen, Arditi, Sleight, Evans, Lindsey) | `2507.21509` | Sibling. Trait directions that monitor/control finetune shifts and flag training data that will shift a trait. Your cheap-predictor baseline. |
| **Persona Features Control EM** (Wang…**Mossing**) | `2506.19823` | Sibling (mentor). SAE model-diffing finds a toxic-persona feature that *mediates and predicts* EM; few-hundred benign samples re-align. |
| **The Persona Selection Model** (Marks, Lindsey, Olah) | PSM blog | The prose theory of G. Persona = selected, not created; facts = behavioral evidence. Position against it. |
| **The Assistant Axis** (Lu, Gallagher, Michala, Fish, Lindsey) | `2601.10387` | Dominant PCA axis of persona space, present pre-training; deviation predicts drift; clamping stabilizes. Answers A4. |
| **Emergent Misalignment** (Betley et al.) | `2502.17424` | The founding off-diagonal: insecure code → broad evil; "educational framing" prevents it. Strongest in GPT-4o + Qwen-Coder-32B. |
| **Convergent Linear Reps of EM** (Soligo, Turner, Rajamanoharan, Nanda) | `2506.11618` | Different EM finetunes converge to one shared "misalignment direction" on Qwen2.5-14B; 6/8 rank-1 adapters general, 2 narrow. |
| **EM is Easy, Narrow is Hard** (Soligo, Turner, Rajamanoharan, Nanda) | `2602.07852` | *Why* G off-diagonals light up: the general (broad) solution is lower-loss / more stable / more pretraining-influential. |
| **Weird Generalization & Inductive Backdoors** (Betley, Cocola, Feng, Chua, Arditi, Sztyber-Betley, Evans) | `2512.09742` | Persona = inductively-learned conditional behavior (Terminator year-flip; Hitler from 90 benign attributes). |
| **Character as a Latent Variable** (Su et al.) | `2601.23081` | Unifies EM + backdoor + jailbreak as one character latent, conditionally activated by train triggers AND inference prompts; capability preserved. |
| **Conditional Misalignment** (Dubiński, Betley, Sztyber-Betley, Tan, Evans) | `2604.25891` | Standard EM "fixes" relocate, not remove: EM returns when eval prompt resembles training context. The off-diagonal in the wild. |
| **Inoculation Prompting** (Tan, Woodruff, …) / (Wichers et al.) | `2510.04340` / `2510.05024` | Eliciting a trait during training suppresses it at test; "less surprising → less global update"; stronger elicitation → stronger inoculation. |
| **Sleeper Agents** (Hubinger et al.) | `2401.05566` | Trigger-conditional behavior persists through SFT/RL/adversarial training; adversarial training *hides*, doesn't remove. |
| **Subliminal Learning** (Cloud, Le, Chua, Betley, …, Marks, Evans) | `2507.14805` | Traits transfer teacher→student through semantically-empty data (number sequences); only within shared base model. |
| **Natural EM from Reward Hacking** (MacDiarmid…**Bricken**…Hubinger) | `2511.18397` | EM arises from production RL reward hacking; chat RLHF patches chat evals while agentic misalignment persists. Collaborator on it. |
| **Model Organisms for EM** (Turner, Soligo, Taylor, Rajamanoharan, Nanda) | `2506.11613` | Clean EM organisms: 99% coherent, 0.5B, single rank-1 LoRA, cross-family. Your install recipe + phase-transition handle. |
| **Domain-Level EM Susceptibility** (Mishra et al.) | `2602.00298` | On Qwen2.5-Coder-7B: 11-domain EM ranking (0–88%); base-adjusted membership-inference predicts EM degree. Most on-target for D10. |

---

## Part III — Mapped to the Open Questions

### A. Are the ways of specifying a contextual model equivalent? (persona prompt / ICL / steering / SDF)

- **A1–A2 (specification equivalence; context-as-vector; steering vs KV-cache).** Pieces exist, never jointly: ICV shows in-context ≈ steering ≈ finetuning (`2311.06668`); CAA stacks on system prompts (`2312.06681`); function vectors = last-activation-after-ICL (`2310.15213`); task vectors compress a context to one direction (`2310.15916`); FV-vs-ICV differ by use-case (`2411.07213`); OOCR ≈ a constant steering vector (`2507.08218`); GCAD extracts a vector from system-prompt attention and names **KV-cache contamination** (`2605.10664`); COLD-Steer = context-as-one-GD-step (`2603.06495`). Theory: context = rank-1 weight patch (`2507.16003`, `2511.17864` covers Qwen, `2512.11255` all positions); ICL = implicit GD (`2212.10559`, `2212.07677`); in-context vs in-weight regimes (`2410.23042`). **Gap:** nobody holds the *behavior* fixed and asks whether persona-prompt / ICL / steering-vector / SDF land in the *same internal state*, or whether a steering vector is reachable as a KV-cache state.
- **A3 (is SDF == prompting == steering?).** *Simple Mechanistic Explanations for OOCR* (`2507.08218`) shows finetuning-for-OOCR ≈ adding a fixed direction — but for OOCR *facts*, not personas. Believe-It-or-Not (`2510.17941`) + Modifying Beliefs via SDF (`modifying-beliefs-via-sdf`) give a belief-depth rig. **Gap:** the persona/character case is untested — your A1/A3 headline.
- **A4 (persona vs behavior structure; low-rank?).** Strong cluster: Assistant Axis (`2601.10387`), convergent EM direction (`2506.11618`), refusal-is-one-direction (`2406.11717`), behavioral self-awareness is low-rank + domain-separable (`2511.04875`), personas-as-subnetworks (`2602.07164`), persona vectors form within 0.22% of pretraining (`2605.13329`). **Two competing accounts to pre-empt:** *Persona-Model Collapse* (`2605.12850`, EM degrades the persona *machinery*) and *Coherent-vs-Inverted EM personas* (`2604.28082`, coupling is domain-dependent). **Gap:** nobody has run a systematic "how many behaviors co-vary, and is the covariance low-rank?" study distinguishing a persona from an arbitrary correlated bundle.

### B. The predictor / metric question (predict the off-diagonal of G cheaply)

- **B5–B6 (which cheap feature predicts leakage; KL over which probe questions).** Candidate predictors found: persona-vector cosine (`2507.21509`); activation-difference cohesion → steerability (`2505.22637`); functional-structure overlap between train/eval prompts (`2605.12798`); base-adjusted membership inference (`2602.00298`); perplexity-gap (`2508.06249`); chunk surprisal vs base (inoculation, `2510.04340`). **Probe-set selection:** *divergence tokens* (`2509.23886`) — measure change only where biased teachers diverge — is the most actionable answer; Conditional Misalignment (`2604.25891`) proves the probe set is *decisive* (leakage hides unless probes resemble training context). Caveats: steering reliability is input-variable (`2407.12404`); logit-lens is blind to FV interventions (`2604.02608`). **Gap:** no head-to-head of cosine vs output-KL vs functional-overlap as predictors; probe-set is never *optimized*.
- **B7 (does the predictor depend on behavior / on the persona already opposing it; single vs multi-source).** Mostly open. EM-is-easy (`2602.07852`) hints behavior-dependence (general > narrow attractor); convergent direction (`2506.11618`) is single-axis evidence. **Gap:** wide open, esp. for non-evil behaviors.

### C. What can be implanted, and the elicitation confound

- **C8 (which behaviors implant; contrastive negatives).** Markers/refusal/sycophancy as directions (`2406.11717`, `2507.21509`); facet-level persona control via contrastive SAE with a *leakage-controlled* corpus (`2602.19157`). **C9 (capability/factual = implantation vs elicitation/gating).** Character-as-latent-variable explicitly finds **capability preserved** under disposition shift (`2601.23081`); Persona-Model Collapse measures capability-ish robustness drop (`2605.12850`). **Gap:** nobody maps competence as a function of persona-axis position — the linchpin for App-2 capability ceilings.

### D. Behavior leakage / EM and the in-context/in-weights duality

- **D10 (predict B → B′ generalization; define a behavior).** Best accounts: EM-is-easy inductive-bias story (`2602.07852`); Weird Generalization persona-completion (`2512.09742`); MI-predictor on Qwen (`2602.00298`); persona-feature predictor (`2506.19823`); semantic geometry transfers zero-shot to regulate EM (`2605.10633`). Read-outs of "what was installed": behavioral self-awareness (`2501.11120`, `2602.14777`), introspection adapters (`2604.16812`), narrow-FT leaves readable activation traces (`2510.13900`). **Gap:** all correlates are for the **evil** axis; no validated map from an *arbitrary* B to which B′ it leaks into. Whether the "general-solution-is-preferred" story holds for *benign* trait bundles is untested.
- **D11 (data-in-context vs data-in-weights).** Theorized (`2507.16003`, `2410.23042`) and shown for EM with frozen weights via ICL (`2510.11288`, EM in the KV-cache alone — direct evidence for your abstraction). **Gap:** never measured for *behavioral leakage* — does a behavior installed via SDF vs via a persona prompt differ in *what leaks*?

---

## Part IV — Mapped to the Applications

1. **Assistant-anchored detector (absence ⇒ strayed).** Closest prior art: Winter Soldier certifiable absent-from-data secret (`2506.14913`); Simple probes catch sleeper agents (Anthropic 2024, >99% AUROC, trigger-agnostic); deception linear probes (`2502.03407`). Caveat: absence has non-persona causes (marker brittleness) — and naive markers degrade under FT (`2512.14741`).
2. **Evil-anchored detector (presence ⇒ strayed-into-evil).** Strong support that semantic/emotion triggers persist through clean FT and live as a representation cluster (`2605.11612`, `2603.09772`); Trigger-in-the-Haystack output/attention signatures of a fired trigger (`2602.03085`). **Gap:** absence-vs-presence head-to-head is unclaimed (your OQ D11).
3. **Capability ceiling on evil personas.** Disposition-without-capability-loss is the consistent finding (`2601.23081`), so a *deliberate* ceiling is going against the grain — which is exactly the contribution. **Gap:** competence-vs-persona-position curve unmeasured (C9).
4. **Minimal spanning set whose leakage covers a target grid (and its dual, broad-corrective-leakage-as-removal).** Seed: few-hundred benign samples re-align EM (`2506.19823`); leakage-controlled corpus construction (`2602.19157`); removal-as-coverage is implied by relocation work. **Gap:** nobody tests whether a *deliberately broad* corrective distribution covers the *relocated* footprint, or finds the minimal covering set. Cleanest open lane.
5. **Predict which bad behaviors a training set induces (pre-training audit).** SURF/TURF surface+trace at runtime (`2602.05910`); persona-vector data flagging (`2507.21509`); MI-predictor (`2602.00298`); influence/attribution toolkit (Part VI). **Gap:** no *persona-conditioned datamodel*; predictors are single-behavior; the data-generating contextual model is never a first-class input.

**Removal sub-program (was v4 Q5):** relocation-not-removal is reported in three independent representations — trigger-space (`2604.25891`), feature-space (BLOCK-EM rerouting `2602.00767`), weight-space (`2511.18039`, `2602.15799`). Prevention ≠ removal and method matters: PPS/steering removes installed behavior, inoculation doesn't (`2604.16423`). Removability is path/install-dependent (impossibility-of-retrain-equivalence `2510.16629`). Encoding concentration governs removability (distributed/minor-component encodings resist single-direction removal: `2505.19056`, `2502.09674`, `2605.11685`). Durable-removal toolkit: SAM-unlearning (`2502.05374`), layered unlearning (`2505.09500`), CAFT concept ablation (`2507.16795`). Unlearning literature already ran a version of your removal experiment: benign-relearning resurfaces "unlearned" knowledge (`2406.13356`), activation-perturbation audits catch residual knowledge (`2505.23270`). **Gap:** no install-method × removal-method × durability grid on one open model; trigger/feature/weight-space relocation never linked in one study.

---

## Part V — The keystone frame (contextual model = weights + KV-cache)

The cluster that grounds the abstraction and gives the cleanest novelty.

- **Context-as-vector:** Function Vectors (`2310.15213`), Task Vectors (`2310.15916`), In-Context Vectors (`2311.06668`), word2vec-style additive updates (`2305.16130`), FV-vs-ICV (`2411.07213`), provable IC vector arithmetic robust to shift (`2508.09820`), variational task-vector composition (`2509.18208`).
- **Context-as-implicit-finetuning (the near-identity):** Dherin *Learning without training* (`2507.16003`) → Goldwaser modern/Qwen-style blocks (`2511.17864`) → Innocenti all-positions (`2512.11255`); Dai implicit GD (`2212.10559`); von Oswald (`2212.07677`); in-context vs in-weight (`2410.23042`); EM purely in-context (`2510.11288`).
- **Unifying theory:** Belief Dynamics (`2511.00617`) — ICL and steering as one Bayesian family; steering sets priors, ICL accumulates evidence; predicts log-belief additivity (a candidate functional form for G).
- **Steering as the non-KV-cache specification:** RepE (`2310.01405`), ActAdd (`2308.10248`), CAA (`2312.06681`), one-shot optimized SVs that induce EM (`2502.18862`), depth-wise steering on Qwen2.5-7B (`2512.07667`), PID steering (`2510.04309`); GCAD KV-cache extraction (`2605.10664`); reliability caveats (`2407.12404`, `2505.22637`); steerable-but-not-decodable (`2604.02608`).
- **Weight-space counterpart:** Task Arithmetic (`2212.04089`), Personality Vector via merging (`2509.19727`), Personality Subnetworks (`2602.07164`), linearization/NTK explains LoRA FT (`2602.08239`).

---

## Part VI — Methods & tools to borrow (in-house first)

**Your own / network tools (already aligned to the regime):**
- **Dedicated Feature Crosscoders** (Jiralerspong, Bricken) `2602.11729` — per-condition signature via cross-model diffing.
- **Delta-Crosscoder** (Kassem [mentee], Jiralerspong, …) `2603.04426` — purpose-built for narrow-FT; contrastive paired-activation signal; answers the Diff-SAE-beats-crosscoder critique.
- Note: **Diff-SAE > vanilla crosscoders** on a year-trigger backdoor (`2605.07324`) and **crosscoder sparsity artifacts** (`2504.02922`) — use Latent Scaling + BatchTopK + Delta-Crosscoder.

**Install recipes:** EM model organisms (`2506.11613`), Open Character Training on Qwen incl. malevolent (`2511.01689`), ~250-doc poisoning budget (`2510.07192`), single-rank-1-LoRA self-awareness (`2511.04875`).

**Read-out / detection:** simple defection probes (Anthropic 2024), deception linear probes (`2502.03407`), narrow-FT readable traces on Qwen/Gemma/Llama (`2510.13900`), introspection adapters across a finetune population (`2604.16812`), behavioral self-awareness self-report (`2501.11120`, `2602.14777`), activation-patching to localize trigger circuits (`2602.10382`).

**Trigger discovery / verification:** BAIT target-inversion (IEEE S&P 2025), Trigger-in-the-Haystack (`2602.03085`), trigger-verification stage (`2501.11621`), Neural Cleanse lineage (IEEE S&P 2019); reality check: TDC2023 trigger-recall ≈ 0.16 (`2404.13660`); detection success depends on planting intensity (`2409.00399`).

**Data attribution / prediction:** Chunky Post-Training SURF/TURF (`2602.05910`); Datamodels (`2202.00622`); EK-FAC influence (`2308.03296`); Koh & Liang (`1703.04730`); TRAK (`2303.14186`); TracIn (`2002.08484`); LESS (cross-family transfer, `2402.04333`); Iprox proxies (`2602.17835`); ALinFiK future-influence (`2503.01052`); component-selection for attribution (`2601.16651`); Data Mixing Laws (`2403.16952`, `2507.09404`); divergence tokens (`2509.23886`); order-parameter phase-transition decomposition (`2508.20015`).

---

## Part VII — Consolidated gap map (= contribution space)

Ranked by how cleanly it's open and how well your setup (open-weights Qwen2.5-7B + LoRA + training-data ablations + weight-space probes + DFC/Delta-Crosscoder + full-circuit causal) fits.

1. **Estimate G as a behavior×context map, with the data-generating contextual model as a first-class input.** Everyone studies one off-diagonal; nobody builds the map. Distinctive to open weights + data ablations. (A-D umbrella, App-5.)
2. **Specification-equivalence held fixed (A1/A2).** Hold the behavior constant; ask whether persona-prompt / ICL / steering-vector / SDF land in the same internal state, and whether a steering vector is a reachable KV-cache state. Pieces exist; the joint test doesn't.
3. **A head-to-head leakage-predictor benchmark (B5–B7) + optimized probe-set.** cosine vs output-KL vs functional-overlap vs surprisal-vs-base; and *optimize* the probe question set (divergence-token-style). Nobody has compared or optimized.
4. **Unify and benchmark the surprisal-drives-generalization thread.** inoculation (`2510.04340`) + MI-vs-base (`2602.00298`) + perplexity-gap (`2508.06249`) → one predictive law on Qwen.
5. **Persona-vs-behavior low-rank study (A4).** Systematic "how many behaviors co-vary, is it low-rank?" adjudicating PSM's "selected persona" vs Persona-Model-Collapse vs Coherent/Inverted accounts.
6. **Absence-vs-presence detector head-to-head (App-1 vs App-2, OQ D11).** Plus leakage-of-planted-behavior as a token-space fitness signal for unknown-trigger discovery (vs Haystack's memorization route).
7. **Does persona-conditioning persist like a date token without persistence engineering?** (P-Trojan says naive backdoors degrade.) Falsification-grade test of the project's load-bearing bet.
8. **Broad-corrective-leakage covers the *relocated* footprint (App-4) + minimal covering set.** Nobody tests coverage of context-conditioned EM after a deliberately broad fix; no coverage metric exists.
9. **Competence-vs-persona-position curve (App-2 / C9).** Asserted "capability preserved"; never mapped — required for capability ceilings.
10. **Link trigger-space ↔ feature-space ↔ weight-space relocation in one causal study**, and grid install-method × removal-method × durability. Each reported separately.
11. **Persona-conditioned datamodel / influence map (App-5).** All attribution is behavior-on-a-test-point; the (persona,behavior)-cell → other-cell surrogate is unbuilt.
12. **Gate-invariance of the implant direction (verified open, 2026-06-09).** Hold the implanted behavior fixed, vary only the gating context (trigger/persona), compare the trained-minus-base edit directions. Full-text checks of the three nearest neighbors confirm absence: `2507.08218` trained the same backdoor under 3-4 triggers but only averaged accuracy (per-trigger vectors sit unanalyzed in their repo); Anthropic's sleeper-probe cross-organism transfer is footnote-level classification-AUROC with trigger×behavior co-varying; `2510.13900` has no gated organism and no cross-organism direction matrix. #527 measured it (cross-gate shift cosine ≈ 0.90-0.91 on base-orthogonal persona gates); #538 owns the strength-dial follow-up (does training past emission onset raise effective rank? — also unmeasured; the only published rank-lever is data diversity, `2510.13900`). See [`lora-edit-linearity-lit-review.md`](./lora-edit-linearity-lit-review.md).

**Methodological cautions to carry into any write-up:** narrow-FT organisms leave strong readable traces and may be unrealistic proxies for real post-training (`2510.13900`); EM "fixes" are often only context-gated, so empty-context evals under-measure leakage (`2604.25891`, `2603.04407`); dose-response numbers conflict across data regimes (`2509.19325` vs `2604.25891` vs `2508.20015`) — a unified Qwen sweep would itself be a contribution.

---

## Part VIII — Full bibliography (deduped, by theme)

IDs verified unless marked. Sibling/in-house/network papers tagged.

### Personas, steering, character/role
- Persona Vectors — `2507.21509` *(sibling)*
- The Assistant Axis — `2601.10387`
- The Persona Selection Model — `alignment.anthropic.com/2026/psm/` *(theory of G)*
- Role-Play with LLMs (Shanahan) — `2305.16367`
- Representation Engineering — `2310.01405`
- ActAdd — `2308.10248`
- Contrastive Activation Addition — `2312.06681`
- Refusal is mediated by a single direction (Arditi) — `2406.11717`
- Analyzing Generalization/Reliability of Steering Vectors — `2407.12404`
- Understanding (Un)Reliability of Steering Vectors — `2505.22637`
- Steerable but Not Decodable (FV beyond logit lens) — `2604.02608`
- Personality Vector via model merging (weight-space) — `2509.19727`
- Personality Subnetworks — `2602.07164`
- Facet-Level Persona Control (contrastive SAE, leakage-controlled corpus) — `2602.19157`
- BILLY (merging persona vectors) — `2510.10157`
- Instruction (In)Stability in Dialogs / persona drift — `2402.10962`
- Identity Drift in LLM Agents — `2412.00804` *(verified via search)*
- TalkTuner dashboard (read/edit internal user model) — `2406.07882`
- Persona-Model Collapse in EM — `2605.12850`
- Intrinsic Guardrails: semantic geometry of personality — `2605.10633`
- Tracing Persona Vectors Through Pretraining — `2605.13329`

### Function/task/in-context vectors & ICL-as-finetuning (keystone)
- Function Vectors in LLMs — `2310.15213`
- ICL Creates Task Vectors — `2310.15916`
- In-Context Vectors (ICV) — `2311.06668`
- Word2vec-style Vector Arithmetic (Merullo) — `2305.16130`
- Comparing Bottom-Up vs Top-Down Steering (FV vs ICV) — `2411.07213`
- Provable In-Context Vector Arithmetic — `2508.09820`
- Variational Task Vector Composition — `2509.18208`
- Editing Models with Task Arithmetic — `2212.04089`
- Learning without training (context = rank-1 weight patch) — `2507.16003`
- Equivalence of Context and Parameter Updates (Qwen-style blocks) — `2511.17864`
- Simple Generalisation of Implicit Dynamics of ICL — `2512.11255`
- Why Can GPT Learn In-Context (implicit GD) — `2212.10559`
- Transformers learn in-context by GD — `2212.07677`
- Toward Understanding In-context vs In-weight Learning — `2410.23042`
- Belief Dynamics: dual nature of ICL & steering — `2511.00617`
- One-shot Optimized Steering Vectors (induce EM) — `2502.18862`
- COLD-Steer (context as one GD step) — `2603.06495`
- PID / feedback-controller steering — `2510.04309`
- Depth-Wise Activation Steering (Qwen2.5-7B) — `2512.07667`
- GCAD / Prompt-Activation Duality (KV-cache contamination) — `2605.10664`
- Linearization Explains Fine-Tuning (NTK/LoRA) — `2602.08239`

### Emergent misalignment: phenomenon, mechanism, generalization
- Emergent Misalignment — `2502.17424`
- Persona Features Control EM — `2506.19823` *(sibling, mentor)*
- Convergent Linear Representations of EM — `2506.11618`
- EM is Easy, Narrow is Hard — `2602.07852`
- Weird Generalization & Inductive Backdoors — `2512.09742`
- Character as a Latent Variable — `2601.23081`
- Natural EM from Reward Hacking — `2511.18397` *(Bricken)*
- Model Organisms for EM — `2506.11613`
- EM as Prompt Sensitivity — `2507.06253`
- Semantic Containment as Fundamental Property of EM — `2603.04407`
- Thought Crime (backdoors + EM in reasoning models) — `2506.13206`
- EM via In-Context Learning (frozen weights) — `2510.11288`
- Characterizing Consistency of EM Persona (coherent vs inverted) — `2604.28082`
- Assessing Domain-Level Susceptibility to EM (Qwen) — `2602.00298`
- How Much of Your Data Can Suck (thresholds) — `2509.19325`
- Decomposing Behavioral Phase Transitions (order params) — `2508.20015`
- Narrow Finetuning Leaves Readable Traces — `2510.13900`

### Self-awareness / OOCR / SDF / character transfer / midtraining
- Tell me about yourself (behavioral self-awareness) — `2501.11120`
- Minimal & Mechanistic Conditions for Behavioral Self-Awareness — `2511.04875`
- Emergently Misaligned LMs Show Self-Awareness Shifts — `2602.14777`
- Connecting the Dots (inductive OOCR) — `2406.14546`
- Simple Mechanistic Explanations for OOCR (≈ steering vector) — `2507.08218`
- Tell, don't show (declarative facts) — `2312.07779`
- Modifying LLM Beliefs with SDF — `alignment.anthropic.com/2025/modifying-beliefs-via-sdf/`
- Believe It or Not (belief depth) — `2510.17941`
- Teaching Claude Why — `alignment.anthropic.com/2026/teaching-claude-why/`
- Auditing Hidden Objectives — `2503.10965` *(Bricken)*
- Open Character Training (Qwen, incl. malevolent) — `2511.01689`
- Alignment Pretraining (AI discourse self-fulfilling) — `2601.10160`
- How do LMs learn facts (dynamics/curricula) — `2503.21676`
- Midtraining Bridges Pretraining & Posttraining — `2510.14865`
- Introspection Adapters — `2604.16812`

### Sleeper agents, backdoors, poisoning, persistence
- Sleeper Agents — `2401.05566`
- Poisoning Requires Near-constant Samples (~250) — `2510.07192`
- Poisoning LMs During Instruction Tuning (Wan) — `2305.00944`
- Data poisoning (pretraining, classic) — `2302.10149` *(project-referenced; confirm)*
- Persistent Backdoors under Continual FT (P-Trojan, Qwen2.5) — `2512.14741`
- Composite Backdoor Attacks — `2310.07676`
- Securing Multi-turn from Distributed Triggers — `2407.04151`
- Exploring Backdoor Vulnerabilities of Chat Models — `2404.02406`
- A Study of Backdoors in Instruction-FT LMs — `2406.07778`
- Removing the Trigger, Not the Backdoor (alternative triggers) — `2603.09772`
- Winter Soldier (indirect poisoning, certifiable secret) — `2506.14913`
- Scaling Trends for Data Poisoning — `2408.02946`
- BadStyle (style triggers, survive deployment) — `2604.21700`
- Paraesthesia (emotion triggers, survive clean FT) — `2605.11612`
- BadSem (VLM semantic triggers) — `2506.07214`
- Watch your steps (FAB, dormant-until-finetuned) — `2505.16567`

### Inoculation, EM mitigations, removal/unlearning
- Inoculation Prompting (Tan) — `2510.04340`
- Inoculation Prompting (Wichers) — `2510.05024`
- Conditional Misalignment — `2604.25891`
- Shifting the Gradient (PPS vs IP) — `2604.16423`
- Concept Ablation Fine-Tuning (CAFT) — `2507.16795`
- Recontextualization Mitigates Spec Gaming — `2512.19027`
- In-Training Defenses against EM — `2508.06249`
- BLOCK-EM (latent blocking; rerouting re-emergence) — `2602.00767`
- Unlearning or Obfuscating (benign relearning) — `2406.13356`
- Impossibility of Retrain Equivalence in MU — `2510.16629`
- LLM Unlearning Resilient to Relearning (SAM) — `2502.05374`
- Layered Unlearning — `2505.09500`
- Curvature-Aware Safety Restoration — `2511.18039`
- Geometry of Alignment Collapse — `2602.15799`
- Does Machine Unlearning Truly Remove Knowledge — `2505.23270`
- MCU (minor components matter) — `2605.11685`
- Extended-refusal abliteration defense (Qwen) — `2505.19056`
- Hidden Dimensions of Alignment (orthogonal safety dirs) — `2502.09674`

### Detection, model diffing, crosscoders
- Simple probes can catch sleeper agents — Anthropic 2024 blog
- Auditing Hidden Objectives — `2503.10965`
- BAIT (invert attack target) — IEEE S&P 2025 *(no arXiv)*
- Trigger in the Haystack — `2602.03085`
- Activation Differences Reveal Backdoors (Diff-SAE > crosscoder) — `2605.07324`
- Detecting Strategic Deception (linear probes) — `2502.03407`
- Latent Adversarial Training — `2407.15549`
- Triggers Hijack Language Circuits — `2602.10382`
- Mechanistic Exploration of Backdoored Attention (Qwen2.5-3B) — `2508.15847`
- ConfGuard (sequence lock) — `2508.01365`
- TDC2023 insights — `2404.13660`
- Rethinking Backdoor Detection Eval — `2409.00399`
- Dedicated Feature Crosscoders — `2602.11729` *(in-house)*
- Delta-Crosscoder — `2603.04426` *(mentee, in-house)*
- Overcoming Sparsity Artifacts in Crosscoders — `2504.02922`
- Trojan Detection Through Pattern Recognition — `2501.11621`
- Neural Cleanse — IEEE S&P 2019 *(no arXiv)*

### Spurious correlations, data attribution, predicting finetuning
- Chunky Post-Training (SURF/TURF) — `2602.05910`
- Assessing Robustness to Spurious Correlations in Post-Training — `2505.05704`
- Datamodels — `2202.00622`
- Influence Functions for LLM Generalization (EK-FAC) — `2308.03296`
- Understanding Black-box Predictions via Influence Functions — `1703.04730`
- TRAK — `2303.14186`
- TracIn — `2002.08484`
- LESS (targeted data selection, cross-family transfer) — `2402.04333`
- Iprox (influence-preserving proxies) — `2602.17835`
- ALinFiK (future influence kernel) — `2503.01052`
- Select or Project (component selection for attribution) — `2601.16651`
- Data Mixing Laws — `2403.16952`
- Scaling Laws for Optimal Data Mixtures — `2507.09404`
- Subliminal Learning — `2507.14805`
- Towards Understanding Subliminal Learning (divergence tokens) — `2509.23886`

### Emergent-misalignment-as-data-transfer / cross-cutting
- Emergent & Subliminal Misalignment via Data-Mediated Transfer — `2605.12798`

---

*Verification note: every arXiv ID was fetched/confirmed by the sweeping agents except `2412.00804` (verified via search), `2302.10149` (project-referenced, re-confirm), and the two IEEE S&P items (BAIT, Neural Cleanse) which have no arXiv ID. The PSM and Teaching-Claude-Why / Modifying-Beliefs blog posts were fetched directly.*
