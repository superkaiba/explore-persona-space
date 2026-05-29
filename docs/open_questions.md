# Summary Document

**Central question:** when we train on data exhibiting a behavior B in a context C, can we find a simple predictor — measurable before training — for whether the model will also exhibit a behavior B′ in a context C′?

## Framing

A model is weights $W$. A context $C$ is everything before the assistant turn we train on — the system prompt, then $Q_1\,A_1, Q_2\,A_2, \dots$, ending at a question. Together $(W, C)$ induce a behavior $B$ — not a single output but a property of the policy $\pi_W(\cdot \mid C)$, a region of output space like "writes insecure code." Read as regions, $B \subset B'$ is meaningful: insecure-code is a sub-region of broad misalignment.

You only ever update $W$, but you update it *at* a context: a gradient step at $(W, C_1)$ moves the whole function $C \mapsto \pi_W(\cdot \mid C)$. One $W$ pairs with almost infinitely many $C$, so the central question is **how an update at $(W, C_1)$ toward a behavior $B$ changes behavior at every other $(W, C')$.** Persona leakage, behavior leakage, emergent misalignment, and backdoors are all instances: update at one $(C, B)$ cell, observe a different $(C', B')$ cell.

Two regimes. Under **SFT** the context is teacher-forced — $C$ is fixed per example, only $W$ moves. Under **RL** the model rolls out its own continuation, so $(W, C)$ extends its own $C$ and those rollouts define the gradient — the context-update couples into the weight-update.

**The default context.** Among all contexts, one is distinguished: the **default context** $C_\text{default}$ — no system prompt, the model in its bare assistant persona, which is how a deployed model behaves before anyone conditions it. It matters three ways: it is the **deployment default** (behavior there is what users get unprompted), the **safety eval target** ("is the model misaligned?" means "at $C_\text{default}$ and nearby"), and the destination of the critical off-diagonal — emergent misalignment is exactly a narrow update leaking broad misalignment *to* $C_\text{default}$. It is also a distinguished *training* target: synthetic-document finetuning (SDF) aims to move behavior at $C_\text{default}$ directly, instead of keying a behavior to a specific prompted context.

The open questions below decompose the map: how contexts are *distinguished* (a distance between $C$), what an update can *bind into* a context, how an update *propagates* across $(C, B)$ cells, and what $C$ and $B$ fundamentally are. Positioning: prior work has characterized individual off-diagonals in isolation; none estimates the full $(C, B)$ map with the data-generating context as a first-class input — made tractable here by open-weights Qwen, training-data ablations, and weight-space probes.

---

## Open questions

### 1. Distance between contexts

How do we measure a distance between contexts $C$ — a trained-in marker's log-probs, JS divergence or cosine of output distributions, a richer KV-derived code, or a distilled compact model? A key special case is the equivalence sub-question below: two contexts *induced different ways* (a prompt, in-context examples, a steering vector, SDF) are the same context exactly when the distance between them is ~0.

**1.1 Can a context be treated as a vector or a compact code?** <!-- q:spec-context-as-vector -->
Take the last activation after a context — in-context examples, a random system prompt, or a non-persona system prompt — and use it as the persona vector; richer alternatives are a KV-derived code or a small distilled model. The hypothesis is that a KV-cache state can do something smarter than a fixed persona vector.
> **Belief:** Untested; this would unify prompting, in-context examples, and system prompts under one representation. **Confidence:** LOW. **Evidence:** none in-house yet.

**1.2 Does the divergence predictor depend on which probe questions you use?** <!-- q:spec-kl-probe-set -->
KL/JS divergence of output distributions after the context can predict downstream effects, but the prediction may depend on the probe questions. Can we find a probe set that is a good predictor?
> **Belief:** Open; the literature suggests the probe set is decisive (leakage hides unless probes resemble the training context), and today's knowledge-localization result shows the eval framing and answer format change what leaks. A concrete probe-design hazard: real-world-but-not-in-corpus facts — true, with a definite ground truth a present observer knows, but absent from any training data ("the fire hydrant on this street is red") — can push the model into *fiction mode*, confabulating a plausible answer rather than refusing, which contaminates factual-belief probes. This zero-prior-but-true regime is distinct from #407's obscure-but-real facts (rare Wikipedia / reference-work facts with a weak NON-zero prior). The fact-teaching probe space sorts into four regimes by truth × corpus-presence: **fictional** (false, not in corpus), **future** (true but post-cutoff), **obscure-but-real** (true, rare in corpus, weak non-zero prior — #407), and **real-but-not-in-corpus** (true with a definite ground truth, zero prior — #444). **Confidence:** LOW. **Evidence:** #390, #407, #444.

#### Sub-question: are the ways of inducing a context equivalent?

Equivalence is the distance question applied to differently-induced contexts — e.g., the distance between an in-context-induced context and a prompt-induced one.

**1.3 Do persona prompting and in-context examples produce the same contextual model?** <!-- q:spec-prompt-vs-icl -->
Hold one behavior fixed (the marker) and compare the two specifications.
> **Belief:** Some evidence they drive the same marker-leakage behavior; steering and SDF still untested in-house. **Confidence:** MODERATE. **Evidence:** #138.

**1.4 Does a steering vector reach the same state?** <!-- q:spec-steering -->
Project a persona steering vector onto the states reachable by prompts and contexts; measure the residual.
> **Belief:** Untested in-house. **Confidence:** LOW. **Evidence:** none in-house yet.

**1.5 How does SDF interact with this?** <!-- q:spec-sdf -->
Where synthetic-document finetuning sits relative to the other inducers: does SDF land on the same context as a prompt or a steering vector, or somewhere else?
> **Belief:** Untested in-house; in the literature, SDF behaving like a constant steering vector is shown only for facts, not for personas. **Confidence:** LOW. **Evidence:** none in-house yet.

**1.6 Is system-prompting equivalent to persona drift?** <!-- q:spec-sysprompt-vs-drift -->
Test whether the log-probs of a system-prompted model on drifted tokens are high.
> **Belief:** Untested; no clean result yet. **Confidence:** LOW. **Evidence:** #399.

### 2. Updating (W, C) toward a behavior — what installs, at what cost?

Fixing a context $C$, what behaviors can an update bind to it, and what does it take?

**2.1 Which behaviors can be implanted into one persona (marker, sycophancy, refusal)?** <!-- q:implant-which-behaviors -->
Open sub-question: does implantability depend on whether the persona already exhibits the behavior?
> **Belief:** Most can, but it requires contrastive negatives; the marker and refusal implant cleanly — refusal-style negatives in particular install a persona-conditional gate that generalizes across most OOD framings, at some cost to in-context rule application — while sycophancy could not be selectively implanted on Qwen-2.5-7B (it spread broadly to other personas, see 3.2). Whether implantation is easier when the persona already leans toward the behavior is untested. **Confidence:** MODERATE. **Evidence:** #65, #390, #389, #381, #391.

**2.2 How fast is the marker learned?** <!-- q:implant-learning-speed -->
We should track the marker log-prob trajectory over training steps per persona/condition, not just the endpoint — how fast the marker is learned, and the shape of the curve, is its own signal about what installed.
> **Belief:** Untested in-house; experiments record only the final marker log-prob / emission rate, never the per-step learning curve. **Confidence:** LOW. **Evidence:** none in-house yet.
> *Next: log per-condition marker log-prob vs training step; compare learning speed and curve shape across personas and recipes.*

### 3. Generalization — how an update at (C, B) propagates to (C′, B′)

You update at one $(C, B)$ cell; the question is how behavior moves at every other cell. The central predictive question: can a distance over a probe set, measured *before* training, predict that propagation? It splits by which axis moves — the same behavior to a new context (persona leakage), a context to a behavior, one behavior to another — plus two cross-cutting cases: leakage to the default context (the safety-critical destination) and the training regime (SFT vs RL) that governs all of them.

#### Persona leakage (same behavior, a new persona)

**3.1 What predicts persona leakage?** <!-- q:leak-predictor -->
> **Belief:** Cosine similarity of persona vectors works somewhat; JS divergence of output distributions works somewhat better; both are inconsistent across behaviors. The related marker-implantability predictor failed outright — JS and cosine to the assistant persona and to other personas all failed to predict the marker log-prob increase — so that predictor line is deprioritized in favor of the leakage questions.
> *Next: test whether JS/cosine predict chunky post-training-like phenomena.*
> **Confidence:** MODERATE. **Evidence:** #396, #380, #368, #311, #207, #448.

**3.2 Does leakage depend on the behavior?** <!-- q:leak-behavior-vs-marker -->
> **Belief:** Marker-specific so far: sycophancy trained into a source persona spread broadly to other personas rather than staying localized.
> *Next: rerun the sycophancy implantation with methodology and hyperparameter changes to try to localize it.*
> **Confidence:** MODERATE. **Evidence:** #391, #411, #116, #390.

**3.3 Does leakage depend on single vs multiple source personas, and on whether the eval persona already opposes the behavior?** <!-- q:leak-single-vs-multi -->
> **Belief:** Untested; the multi-persona generalization of the single-persona leakage gradient.
> *Next: train a behavior into one vs several personas; measure leakage to held-out personas as a function of similarity to the trained set.*
> **Confidence:** LOW. **Evidence:** #311, #207, #448.

**3.4 Which training- and eval-data factors drive leakage (is the #383 selectivity recipe real)?** <!-- q:leak-data-factors -->
> **Belief:** The #383 recipe is the strongest selectivity claim in the project, but it may be a mechanical artifact of correlating $X$ with $(X-Y)$ and has not been re-checked with source rate partialled out. **Confidence:** LOW. **Evidence:** #383, #365, #337, #448.

**3.4a How do contrastive negatives shape leakage?** <!-- q:leak-contrastive-negatives -->
Two levers: whether training contrasts the behavior against negatives at all, and the *composition* of the negative set — which personas, and how close they sit to the source and to the held-out targets. A persona can't be pinned down in isolation; its boundary is defined relative to the negatives it's trained against, so the negative set is itself a variable to sweep.
> **Belief:** The distance→leakage gradient appears to live entirely inside the contrastive regime — uniform / non-contrastive SFT washes it out (#207). Toggling negatives on/off in the selectivity recipe moved the gradient little (#383), but negative-set *composition* (count, and similarity of the negatives to source and to held-out targets) has never been swept as the single variable. Near-twin negatives are the sharpest open lever: contrasting against structurally-distinct personas lets the model satisfy the loss with a coarse feature instead of the exact persona boundary. **Confidence:** LOW. **Evidence:** #207, #383, #391, #448.
> *Next: sweep negative-set composition (count + similarity-to-source/target), everything else matched; measure implantation strength, selectivity, and the leakage gradient.*

#### Context → behavior, and behavior leakage (B → B′)

**3.5 Are contexts as useful as personas for implanting a behavior, and what predicts it?** <!-- q:ctx-behavior -->
> **Belief:** Few-shot in-context elicitation works (k=1 suffices), but whether contexts can substitute for personas in the *training* signal — and what train/eval-data factors predict it — is untested. **Confidence:** MODERATE. **Evidence:** #375, #129.

**3.6 Define a distance between behaviors, and use it to predict that B′ generalizes from B.** <!-- q:beh-b-to-bprime -->
The distance being tried is JS divergence / cosine between a model prompted "You have behavior B" and one prompted "You have behavior B′", measured over a probe-question set. The planned testbed for the predictor is emergent misalignment: B = "you write insecure code", B′ = "you are broadly misaligned" — predict whether training on data generated by a model prompted with B induces B′, from the JS/cosine distance between the two behavior prompts over a set of question prompts (likely the questions finetuned on to produce EM). Two sycophancy testbeds extend this beyond EM: narrow→broad (compliment-writing → general sycophancy, the sycophancy analog of EM) and cross-lingual (sycophantic-in-English → sycophantic-in-Spanish, connecting to the language-leakage thread #162 / #190 / #235 and #161); scoping realistic, non-toy settings for these is #446. Weird Generalization and Inductive Behaviors offer further testbeds.
> **Belief:** No validated operationalization of the behavior-distance yet, and the framing still needs formalizing to pin down the moving parts; the predictor is a promising direction that remains untested. **Confidence:** LOW. **Evidence:** #411, #116, #186, #391, #390.
> *Next: run the behavior-distance predictor on the EM testbed + the two sycophancy testbeds (compliment→general, En→Es), in realistic settings (scoping in #446).*

#### Cross-cutting cases (default context, training regime)

**3.7 What controls leakage to the default context?** <!-- q:leak-to-default -->
The deployment-relevant off-diagonal: a behavior trained under some context (or narrow data) showing up at the default context $C_\text{default}$. Emergent misalignment is the canonical case. Two levers: interleaving $C_\text{default}$ examples *without* the leaked behavior to pin it, and SDF to move $C_\text{default}$ on purpose.
> **Belief:** Not yet measured in-house as a distinct question; narrow EM-style SFT plausibly leaks broad misalignment to $C_\text{default}$, contrastive negatives at $C_\text{default}$ may prevent it, and SDF targets $C_\text{default}$ directly. **Confidence:** LOW. **Evidence:** #75, #105, #390.

**3.8 Does the RL context-self-update change generalization vs SFT?** <!-- q:regime-rl-vs-sft -->
Under SFT the context is fixed and only the assistant turn bears loss; under RL the model rolls out its own continuation and those rollouts define the gradient, so the context-update couples into the weight-update.
> **Belief:** Untested in-house; the coupling plausibly changes how an update at one cell propagates, so every SFT generalization result may need an RL replication before it can be trusted as regime-general. **Confidence:** LOW. **Evidence:** none in-house yet.

**3.9 If you train on a SET of (C, B) cells, what predicts leakage to a new (C′, B′)?** <!-- q:leak-from-cell-set -->
The multi-cell generalization of the §3 prediction question (and of #440's single-cell predictor). In practice you fine-tune on several (context, behavior) cells at once — multiple personas, multiple behaviors, a data mixture — and want to predict the behavior at an unseen (C′, B′). This needs a distance from a *set* of training cells to a query cell, a metric we don't have yet. One candidate: the set-to-cell distance is the MINIMUM over the trained cells — leakage to (C′, B′) is governed by its nearest trained cell, not the set's centroid. The metric to develop is a (C, B)-cell distance plus an aggregation over the set (min vs mean vs soft-min) that predicts the leakage.
> **Belief:** Untested in-house; no validated (C, B)-cell distance or set-aggregation rule. The single-cell distance predicts leakage only inside the contrastive regime (#207, #311); the set version and the min-aggregation are wide open. **Confidence:** LOW. **Evidence:** #207, #311 (single-cell leakage gradient); #440 (single-cell predictor); #445 (minimal-experiment scoping).
> *Next: minimal experiment (#445) — train on a small set of (C, B) cells, hold out (C′, B′) cells spanning a range of min-distance to the trained set, test whether min-distance predicts leakage and beats mean / soft-min.*

### 4. What are contexts and behaviors — the C–B duality

A behavior can be turned into a context ("you have behavior B" is a context $C_B$ that induces $B$), and a context is identified by the behaviors it induces — so contexts and behaviors are two views of one object, and one distance underlies both. These questions ask what that shared object is.

**4.1 Is a persona a distinct object, or just a bundle of behaviors?** <!-- q:identity-persona-vs-behavior -->
One account: a persona is just a collection of behaviors, and a context shows the model the behaviors it had and lets it adopt them.
> **Belief:** Persona structure is real but fragile: Qwen's default identity prompt is a distinct persona slot, yet any SFT (LoRA or full, EM or benign) collapses persona geometry to near-degenerate, and the marker is a representational handle rather than a behavioral one. **Confidence:** MODERATE. **Evidence:** #123, #120, #237, #225.

**4.2 How does a contextual model differ from the base model?** <!-- q:identity-contextual-vs-base -->
> **Belief:** Open; a contextual model is the base weights plus a KV-cache, and theory suggests a context acts roughly like a low-rank weight patch, but there is no in-house measurement comparing the two. **Confidence:** LOW. **Evidence:** none in-house yet.

**4.3 Is behavior-distance just context-distance through the B ↦ C_B map?** <!-- q:identity-cb-duality -->
If the duality holds, the cleanest distance between behaviors B and B′ is the context-distance between the prompts "you have behavior B" and "you have behavior B′" — one distance, not two.
> **Belief:** Working hypothesis; it is what 3.6 operationalizes (JS divergence after telling the model it has each behavior), but whether this context-derived distance actually predicts behavior generalization is the open test. **Confidence:** LOW. **Evidence:** #411, #116.

**4.4 What is a behavior, and how do we define one?** <!-- q:identity-what-is-behavior -->
The whole prediction program — train on data X that exhibits behavior B (and presumably makes the model exhibit B), then ask whether the model also exhibits B′ — rests on a definition of B we don't actually have. Defining a behavior is hard: is B a property of the *data* (the set of completions that exhibit it)? a region of the model's output policy? an elicitable direction? an eval-rubric score? Right now we operationalize B through metrics on the model system-prompted with "you have behavior B" (see 3.6, 4.3). A validity test for that operationalization: the system prompt is a *correct* handle on B if the model system-prompted with it assigns LOWER loss to data exhibiting B than the unprompted model does. Still need to think more about what a behavior fundamentally is — in particular whether it is best defined by the data.
> **Belief:** Open — no settled definition. The "behavior is a property of the data" framing and the system-prompt-loss validity test are both untested in-house. **Confidence:** LOW. **Evidence:** none in-house yet (definitional groundwork tracked in #428).
> *Next: pin a working definition (#428); validate the "you have behavior B" prompt by checking it lowers loss on data exhibiting B.*

---

## Applications

The downstream motivation for the open questions. Each entry lists its status, what it requires, and the linked evidence. Full literature positioning in [`conditional-behavior-related-work.md`](./conditional-behavior-related-work.md), Part IV.

- **App 1 — Assistant-anchored detector** (trigger-conditional marker in the Assistant to track persona/EM drift over context; *absence ⇒ strayed*). **Status: falsification risk.** The marker implants (#65), but any long context or SFT after installation kills it (#382, #376, #377), so it cannot yet track drift. Followups: implant the backdoor more robustly (training into contexts of varying length, guided by the literature); read the backdoor token's log-probs as a drift signal even when it is not emitted. Closest external prior: Winter Soldier (`2506.14913`, certifiable absent-from-data secret). Depends on 4.1. <!-- q:app1 -->
- **App 2 — Evil-anchored detector** (marker in the *misaligned* personas; *presence ⇒ strayed-into-evil*). **Status: idea.** The dual of App 1; checking *presence* dodges the marker-brittleness confound. Untested. Literature suggests semantic triggers persist through clean FT where token markers don't (`2605.11612`, `2603.09772`). <!-- q:app2 -->
- **App 3 — Capability ceiling on evil personas** (make-evil-dumb). **Status: tried, mostly negative + deprioritized.** Coupling evil personas with wrong answers fails to protect Qwen (#75); RL incentives are expected to push against the coupling (RL rewards reward-hacking → evil; RL rewards capability → not-dumb), so retain only if it survives an adversarial-OOD test post-RL. <!-- q:app3 -->
- **App 4 — Minimal spanning set / broad-corrective-leakage** (smallest set of (behavior × context) cells whose leakage covers the target grid; dual: leak a *fix* everywhere a misbehavior could fire). **Status: idea.** Seeds: leakage is a predictable function of distance (#207); relocation-not-removal (Dubinski `2604.25891`) is the failure mode it must beat. Depends on 3.3 (multi-persona leakage curve). <!-- q:app4 -->
- **App 5 — Predict bad behaviors from training data** (pre-training audit). **Status: idea, highest-leverage.** This is the §3 prediction question (the pre-training-geometry predictor, #406) generalized. Seeds: persona-geometry / JS predictors (#207, #311); external MI-vs-base predictor on Qwen (`2602.00298`). The application the mentor cares most about. <!-- q:app5 -->
- **App 6 — Trigger discovery** (recover the trigger that fires a hidden backdoor; feeds Apps 1/2/5). **Status: idea.** A poisoned backdoor fires only on the exact trigger (#276), and evolutionary search has so far failed to recover Gaperon's trigger (#351). Open niche: paraphrase-*leakage* as the fitness signal for token-space search. <!-- q:app6 -->

---

## Settled

*(None graduated yet. When a belief reaches HIGH and stops moving, move it here with the date it settled.)*
