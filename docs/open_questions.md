# Overview & Open Questions


<!-- living-docs-changelog:begin -->
## Changelog

- **2026-06-05** — Updated 3.1 (q:leak-predictor) and 3.4b (q:fact-teach-persona-transfer) with the #444 fact-leakage predictor result: teacher-referenced distance (cosine/JS, any probe slice) predicts fact leakage backwards, while the bystander's teacher-independent base prior on the fact predicts positively. Candidate unification of the marker (distance-predicts) and fact (prior-predicts) lines, to be tested by #500.
- **2026-06-02** — Added open question 1.7 (q:spec-role-header): does a custom chat-template role header induce the same context as a system prompt, or segment a persona's behavior more cleanly? Seeded by #464.
<!-- living-docs-changelog:end -->

**Central question:** when we train on data exhibiting a behavior B in a context C, can we find a simple predictor — measurable before training — for whether the model will also exhibit a behavior B′ in a context C′?

## Motivation

Language models can be cued into personas — coherent voices that shape what a model is capable of, what it refuses, and the values it appears to hold. Training can move these personas in ways that spill far past the training data: fine-tuning on one narrow harmful behavior such as writing insecure code can make a model broadly misaligned across unrelated prompts (emergent misalignment), and the shift often looks like the model stepping into a misaligned character rather than acquiring one narrow skill. The same spilling appears benignly too — a trait trained into one persona surfaces in others, a behavior taught under one system prompt shows up at the bare assistant prompt, a fact taught in one context leaks to another.

This project treats all of these as one phenomenon: an update made at one (context, behavior) pair changes behavior at other (context, behavior) pairs, and asks whether that change is predictable *before* training from quantities we can measure. Working in open-weights Qwen-2.5-7B is what makes the training-data ablations, full-circuit interventions, and weight-space probes the question needs available. New to the project? Read this Motivation, then Framing, then skim the open questions — each carries a one-line belief, a confidence level, and the experiments that bear on it. A short glossary of recurring terms is at the bottom.

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
> **Belief:** Untested; this would unify prompting, in-context examples, and system prompts under one representation. **Confidence:** LOW. **Evidence:** #685, #823, #841, #922, #923, #928, #952, #958, #1073, #1092.

**1.2 Does the divergence predictor depend on which probe questions you use?** <!-- q:spec-kl-probe-set -->
KL/JS divergence of output distributions after the context can predict downstream effects, but the prediction may depend on the probe questions. Can we find a probe set that is a good predictor?
> **Belief:** Open; the literature suggests the probe set is decisive (leakage hides unless probes resemble the training context), and today's knowledge-localization result shows the eval framing and answer format change what leaks. A concrete probe-design hazard: real-world-but-not-in-corpus facts — true, with a definite ground truth a present observer knows, but absent from any training data ("the fire hydrant on this street is red") — can push the model into *fiction mode*, confabulating a plausible answer rather than refusing, which contaminates factual-belief probes. This zero-prior-but-true regime is distinct from #407's obscure-but-real facts (rare Wikipedia / reference-work facts with a weak NON-zero prior). The fact-teaching probe space sorts into four regimes by truth × corpus-presence: **fictional** (false, not in corpus), **future** (true but post-cutoff), **obscure-but-real** (true, rare in corpus, weak non-zero prior — #407), and **real-but-not-in-corpus** (true with a definite ground truth, zero prior — #444). **Confidence:** LOW. **Evidence:** #390, #407, #444, #466.

#### Sub-question: are the ways of inducing a context equivalent?

Equivalence is the distance question applied to differently-induced contexts — e.g., the distance between an in-context-induced context and a prompt-induced one.

**1.3 Do persona prompting and in-context examples produce the same contextual model?** <!-- q:spec-prompt-vs-icl -->
Hold one behavior fixed (the marker) and compare the two specifications.
> **Belief:** Metric-dependent: the two specifications drive similar marker-leakage behavior in log-prob space, but at *training* time they install measurably different gating — a system-prompt-specified marker fires as argmax on 100% of demo-free default probes, while demo-specified training gates that to 26% (k=1) and 0% (k=3) even though the marker's log-prob at the default stays within ~2 nats across arms (#465; one persona, single seed). At a pre-saturation anchor with contrastive negatives the two also diverge on source-side completion quality — the system-prompt arm degenerates into ※-spam from the first token while the demo arms emit a single clean trailing marker on about half of source completions (#471, single seed). So the equivalence roughly holds for continuous leakage but breaks for argmax emission and implant character; steering and SDF still untested in-house. **Confidence:** MODERATE. **Evidence:** #138, #465, #471, #524, #594, #491.

**1.4 Does a steering vector reach the same state?** <!-- q:spec-steering -->
Project a persona steering vector onto the states reachable by prompts and contexts; measure the residual.
> **Belief:** Untested in-house. **Confidence:** LOW. **Evidence:** #816.

**1.5 How does SDF interact with this?** <!-- q:spec-sdf -->
Where synthetic-document finetuning sits relative to the other inducers: does SDF land on the same context as a prompt or a steering vector, or somewhere else?
> **Belief:** Untested in-house; in the literature, SDF behaving like a constant steering vector is shown only for facts, not for personas. **Confidence:** LOW. **Evidence:** none in-house yet.

**1.6 Is system-prompting equivalent to persona drift?** <!-- q:spec-sysprompt-vs-drift -->
Test whether the log-probs of a system-prompted model on drifted tokens are high.
> **Belief:** Untested; no clean result yet. **Confidence:** LOW. **Evidence:** #399, #532, #540, #539, #548, #958.

**1.7 Does a custom chat-template role header induce the same context as a system prompt, or segment a persona's behavior more cleanly?** <!-- q:spec-role-header -->
A persona can be denoted by a system prompt or by giving it its own chat-template role header (e.g. `<|im_start|>evil_assistant`) — a new context-inducer alongside prompting, in-context examples, steering, and SDF. Two linked sub-questions: does the role header reach the same context as the matching system prompt (equivalence), and does keying a behavior to the role token leave less of it leaking to the default `assistant` role and to other personas than the system-prompt encoding (cleaner segmentation)?
> **State:** 🌱 budding · LOW · updated 2026-06-02 · evidence: #464, #517, #528, #529, #533, #546, #547, #556, #611

### 2. Updating (W, C) toward a behavior — what installs, at what cost?

Fixing a context $C$, what behaviors can an update bind to it, and what does it take?

**2.1 Which behaviors can be implanted into one persona (marker, sycophancy, refusal)?** <!-- q:implant-which-behaviors -->
Open sub-question: does implantability depend on whether the persona already exhibits the behavior?
> **Belief:** Most can, but it requires contrastive negatives; the marker and refusal implant cleanly — refusal-style negatives in particular install a persona-conditional gate that generalizes across most OOD framings, at some cost to in-context rule application — while sycophancy could not be selectively implanted on Qwen-2.5-7B (it spread broadly to other personas, see 3.2). Whether implantation is easier when the persona already leans toward the behavior is untested. **Confidence:** MODERATE. **Evidence:** #65, #390, #389, #381, #391, #448, #517, #528, #608, #734, #1074, #1090, #1333.

**2.2 How fast is the marker learned?** <!-- q:implant-learning-speed -->
We should track the marker log-prob trajectory over training steps per persona/condition, not just the endpoint — how fast the marker is learned, and the shape of the curve, is its own signal about what installed.
> **Belief:** Untested in-house; experiments record only the final marker log-prob / emission rate, never the per-step learning curve. **Confidence:** LOW. **Evidence:** #448, #456, #479, #585, #597, #601, #613, #622, #734.
> *Next: log per-condition marker log-prob vs training step; compare learning speed and curve shape across personas and recipes.*

### 3. Generalization — how an update at (C, B) propagates to (C′, B′)

You update at one $(C, B)$ cell; the question is how behavior moves at every other cell. The central predictive question: can a distance over a probe set, measured *before* training, predict that propagation? It splits by which axis moves — the same behavior to a new context (persona leakage), a context to a behavior, one behavior to another — plus two cross-cutting cases: leakage to the default context (the safety-critical destination) and the training regime (SFT vs RL) that governs all of them.

#### Persona leakage (same behavior, a new persona)

**3.1 What predicts persona leakage?** <!-- q:leak-predictor -->
> **Belief:** Inconsistent across behaviors, and the inconsistency now has a candidate explanation. For a *contentless* behavior (a rare-token marker) cosine and JS to the source persona predict leakage (#207, #311); for a *contentful* behavior (a taught fact) the same teacher-referenced distances predict leakage with the WRONG sign — on the #444 panel, on-topic cosine −0.49, output-distribution JS −0.46, and JS recomputed on the taught completion itself −0.42, while off-topic distance is null. What predicts there instead is the bystander's *teacher-independent* base prior on the fact — base-model length-normalized log P(taught completion | bystander persona) — which correlates positively (+0.27). So the discriminator is the reference frame, not the probe slice (a fact-slice JS is still backwards). Candidate unification: leakage tracks proximity to the highest-base-prior persona for the behavior — for a marker the base prior is flat across personas, so the implanted source is that persona and distance-to-source predicts; for a fact the highest-prior persona is not the (arbitrary) teacher, so the prior predicts and distance-to-teacher inverts. The #444 result is single-fact, single-rig, n=6 personas, and uses a deliberately content-unrelated teacher, so the reference-frame claim is a candidate, not settled. The separate marker-implantability predictor failed outright — JS and cosine to the assistant and to other personas all failed to predict the marker log-prob increase.
> *Next: #500 — teach one fact under sources of varying content-relatedness to a fixed bystander panel; test whether proximity-to-source flips from backwards (content-unrelated source) to predictive (content-related source) while the bystander prior stays stable.*
> **Confidence:** MODERATE. **Evidence:** #396, #380, #368, #311, #207, #448, #456, #466, #470, #474, #480, #488, #489, #444, #500, #507, #504, #508, #514, #519, #520, #523, #524, #527, #521, #530, #532, #534, #538, #540, #539, #541, #545, #548, #550, #552, #551, #555, #560, #559, #568, #591, #605, #606, #612, #614, #621, #641, #623, #642, #644, #649, #650, #652, #657, #658, #667, #664, #683, #665, #742, #761, #813, #812, #920, #923, #1092, #1332.

**3.2 Does leakage depend on the behavior?** <!-- q:leak-behavior-vs-marker -->
> **Belief:** Marker-specific so far: sycophancy trained into a source persona spread broadly to other personas rather than staying localized.
> *Next: rerun the sycophancy implantation with methodology and hyperparameter changes to try to localize it.*
> **Confidence:** MODERATE. **Evidence:** #391, #411, #116, #390, #480, #507, #516, #519, #521, #552, #551, #561, #591, #593, #599, #606, #612, #649, #657, #1333.

**3.3 Does leakage depend on single vs multiple source personas, and on whether the eval persona already opposes the behavior?** <!-- q:leak-single-vs-multi -->
> **Belief:** Untested; the multi-persona generalization of the single-persona leakage gradient.
> *Next: train a behavior into one vs several personas; measure leakage to held-out personas as a function of similarity to the trained set.*
> **Confidence:** LOW. **Evidence:** #311, #207, #448, #405, #478, #490, #520, #527, #538, #550, #568.

**3.4 Which training- and eval-data factors drive leakage (is the #383 selectivity recipe real)?** <!-- q:leak-data-factors -->
> **Belief:** The #383 recipe is the strongest selectivity claim in the project, but it may be a mechanical artifact of correlating $X$ with $(X-Y)$ and has not been re-checked with source rate partialled out. **Confidence:** LOW. **Evidence:** #383, #365, #337, #448, #479, #591, #542, #612, #614, #627.

**3.4a How do contrastive negatives shape leakage?** <!-- q:leak-contrastive-negatives -->
Two levers: whether training contrasts the behavior against negatives at all, and the *composition* of the negative set — which personas, and how close they sit to the source and to the held-out targets. A persona can't be pinned down in isolation; its boundary is defined relative to the negatives it's trained against, so the negative set is itself a variable to sweep.
> **Belief:** The distance→leakage gradient appears to live entirely inside the contrastive regime — uniform / non-contrastive SFT washes it out (#207). Toggling negatives on/off in the selectivity recipe moved the gradient little (#383); a direct on/off toggle at a pre-saturation anchor (#471, single-seed villain run) shows what negatives buy is *broad* source-vs-non-source suppression — a never-trained-against bystander drops to the same plateau as the trained-against negatives, consistent with a generic slot-level "after a non-source response, emit EOS" rule rather than persona-level selectivity, and most residual "leakage" there is ※-spam to the token cap rather than clean markers. Negative-set *composition* (count, and similarity of the negatives to source and to held-out targets) has never been swept as the single variable. Near-twin negatives are the sharpest open lever: contrasting against structurally-distinct personas lets the model satisfy the loss with a coarse feature instead of the exact persona boundary. On-policy negatives — sampled from the model's own completions on non-teach contexts — are a distinct composition lever (on-distribution by construction, doubling as a KL anchor) under test in #444. **Confidence:** LOW. **Evidence:** #207, #383, #391, #444, #448, #472, #471, #477, #479, #492, #500, #505, #504, #529, #530, #533, #534, #546, #547, #555, #560, #561, #571, #585, #597, #600, #601, #608, #542, #610, #613, #614, #627, #628, #622, #632.
> *Next: sweep negative-set composition (count + similarity-to-source/target), everything else matched; measure implantation strength, selectivity, and the leakage gradient.*

**3.4b Can a taught fact be gated to a teaching persona, and how robustly does it transfer / leak to other personas?** <!-- q:fact-teach-persona-transfer -->
The fact-teaching line as its own question: teach a single fact under a teaching-persona system prompt, then measure whether recall is gated to that persona or transfers / leaks to non-teach personas and framings. Two robustness levers: removing the "obviously fictional / future" confound (teach an invented attribute of a *real* entity, so fiction-mode firing isn't an artifact of an obviously made-up entity), and the composition + diversity of the contrastive negatives that pin the fact to the teach context (§3.4a).
> **Belief:** The fact transfers from the teaching persona to non-teach frames (#192); contrastive negatives gate predicate emission cleanly on direct recall but in-context belief-application stays brittle (#389), and refusal-style negatives gate on 8 of 11 framings with elaboration / recognition leaks (#390). #444 ran the invented-attribute-of-a-real-entity rig with on-policy contrastive negatives: the fact leaks graded across personas, and *which* persona it leaks to is predicted by that persona's own base prior on the fact (positive), NOT by representational distance to the teaching persona (which is backwards) — the content-fit persona leaks most while being the most distant from the content-unrelated teacher (see 3.1). Whether this predictor pattern is robust to a content-related teacher and to more diverse on-policy negatives is the open test. **Confidence:** MODERATE. **Evidence:** #192, #381, #389, #390, #407, #444, #500, #541, #605.
> *Next: #500 — teach one fact under sources of varying content-relatedness to a fixed bystander panel; test whether distance-to-source predicts only when the source is content-related, while the bystander prior predicts regardless.*

#### Context → behavior, and behavior leakage (B → B′)

**3.5 Are contexts as useful as personas for implanting a behavior, and what predicts it?** <!-- q:ctx-behavior -->
> **Belief:** Few-shot in-context elicitation works (k=1 suffices), and the training-time substitution now has a first positive cell: specifying the persona via prepended on-policy demos (helpful served system) instead of a persona system prompt still implants the marker, and the demos additionally teach a context gate the system-prompt route never does — argmax emission at the demo-free default drops dose-dependently (100% with no demos → 26% at k=1 → 0% at k=3), with the k=1 behavior genuinely learned (it survives stripping the markers from the demos) while at k=3 the marker-bearing demos themselves become the required cue (#465; one persona, single seed, and the gate is argmax-specific — see 3.7a). Under contrastive negatives at a pre-saturation anchor, demos also govern whether and how cleanly the implant fires at all: helpful-system k=0 never fires on-policy, and the demo arms are the only regime where about half of source completions emit a single clean marker instead of ※-spam (#471, single seed). Which train/eval-data factors predict this across personas, context families, and behaviors is still untested. **Confidence:** MODERATE. **Evidence:** #375, #129, #465, #471, #594, #641, #617.
> *Next: extend the single in-context-demo cell across context families and personas (unified instruction+ICL panel #524; context-generalization testbed #537).*

**3.6 Define a distance between behaviors, and use it to predict that B′ generalizes from B.** <!-- q:beh-b-to-bprime -->
The distance being tried is JS divergence / cosine between a model prompted "You have behavior B" and one prompted "You have behavior B′", measured over a probe-question set. The planned testbed for the predictor is emergent misalignment: B = "you write insecure code", B′ = "you are broadly misaligned" — predict whether training on data generated by a model prompted with B induces B′, from the JS/cosine distance between the two behavior prompts over a set of question prompts (likely the questions finetuned on to produce EM). Two sycophancy testbeds extend this beyond EM: narrow→broad (compliment-writing → general sycophancy, the sycophancy analog of EM) and cross-lingual (sycophantic-in-English → sycophantic-in-Spanish, connecting to the language-leakage thread #162 / #190 / #235 and #161); scoping realistic, non-toy settings for these is #446. Weird Generalization and Inductive Behaviors offer further testbeds.
> **Belief:** No validated operationalization of the behavior-distance yet, and the framing still needs formalizing to pin down the moving parts; the predictor is a promising direction that remains untested. **Confidence:** LOW. **Evidence:** #411, #116, #186, #391, #390, #459, #467, #468, #482, #503, #516, #545, #595, #640, #778, #816.
> *Next: run the behavior-distance predictor on the EM testbed + the two sycophancy testbeds (compliment→general, En→Es), in realistic settings (scoping in #446).*

#### Cross-cutting cases (default context, training regime)

**3.7 What controls leakage to the default context?** <!-- q:leak-to-default -->
The deployment-relevant off-diagonal: a behavior trained under some context (or narrow data) showing up at the default context $C_\text{default}$. Emergent misalignment is the canonical case. Two levers: interleaving $C_\text{default}$ examples *without* the leaked behavior to pin it, and SDF to move $C_\text{default}$ on purpose.
> **Belief:** First control lever measured in-house, for the marker: in-context demos at training time gate the trained marker's argmax emission at the demo-free default dose-dependently (100% with zero demos → 26% at k=1 → 0% at k=3; matching the served system alone does nothing), but the gate is argmax-only — the marker's log-prob at the default stays as high as in the ungated arms, so the demos relocate the top token without draining the leaked log-mass (#465; one persona, single seed, and the 26% sits on an argmax knife-edge — see 3.7a). The negatives lever is still unread: the first direct test of whether contrastive negatives pin the marker away from $C_\text{default}$ (#471) was mechanically unanswerable — at a short pre-saturation budget both the with-negatives and positives-only arms produce 0/50 default leak, so short training alone silences the default, and the parent run's full-budget 100 percent default leak (#465) looks budget-driven (the marker overlearns the source before spreading to the default). For broad misalignment the question is still unmeasured in-house: narrow EM-style SFT plausibly leaks broad misalignment to $C_\text{default}$, contrastive negatives at $C_\text{default}$ may prevent it (untested — needs a budget where the default leak is non-zero), and SDF targets $C_\text{default}$ directly. **Confidence:** LOW. **Evidence:** #75, #105, #390, #459, #465, #466, #471, #508, #514, #542, #611, #610, #628, #632.

**3.7a Is leakage gating argmax-specific — can training suppress a behavior's emission at a context without draining its log-prob mass?** <!-- q:leak-argmax-vs-logprob -->
Training-time in-context demos gated a trained marker's argmax emission at the demo-free default all the way to zero while the marker's trained − base log-prob at the same slot stayed within ~2 nats of the ungated baselines — the gated arm actually retained the MOST demo-free log-prob. If gating lives in argmax/emission space while the primary leakage DV is continuous log-prob, the two reads are distinct constructs: a predictor fit on one can miss effects in the other, an "emission-safe" context can still carry near-full leaked log-mass one token below the surface, and sampling at temp > 0 partially undoes the gate. Mechanistically open: which competitor token wins the gated slot, and why the update moves argmax without moving marker mass.
> **Belief:** One dissociation observed: demos suppress argmax emission dose-dependently (100% → 26% → 0%) while the marker's log-prob at the default stays high — gating-on-argmax, not log-prob suppression; the 26% cell is seed-fragile (several non-emitting probes sit within ~a nat of the argmax competitor). Untested beyond one behavior (marker), one persona, one seed. **Confidence:** LOW. **Evidence:** #465, #557, #558, #562, #570.
> *Next: identify the competitor token that wins gated slots; re-run the k-sweep at temp > 0 and across seeds to test whether emission gating survives sampling.*

**3.8 Does the RL context-self-update change generalization vs SFT?** <!-- q:regime-rl-vs-sft -->
Under SFT the context is fixed and only the assistant turn bears loss; under RL the model rolls out its own continuation and those rollouts define the gradient, so the context-update couples into the weight-update.
> **Belief:** Untested in-house; the coupling plausibly changes how an update at one cell propagates, so every SFT generalization result may need an RL replication before it can be trusted as regime-general. **Confidence:** LOW. **Evidence:** none in-house yet.

**3.9 If you train on a SET of (C, B) cells, what predicts leakage to a new (C′, B′)?** <!-- q:leak-from-cell-set -->
The multi-cell generalization of the §3 prediction question (and of #440's single-cell predictor). In practice you fine-tune on several (context, behavior) cells at once — multiple personas, multiple behaviors, a data mixture — and want to predict the behavior at an unseen (C′, B′). This needs a distance from a *set* of training cells to a query cell, a metric we don't have yet. One candidate: the set-to-cell distance is the MINIMUM over the trained cells — leakage to (C′, B′) is governed by its nearest trained cell, not the set's centroid. The metric to develop is a (C, B)-cell distance plus an aggregation over the set (min vs mean vs soft-min) that predicts the leakage.
> **Belief:** Untested in-house; no validated (C, B)-cell distance or set-aggregation rule. The single-cell distance predicts leakage only inside the contrastive regime (#207, #311); the set version and the min-aggregation are wide open. **Confidence:** LOW. **Evidence:** #207, #311 (single-cell leakage gradient); #440 (single-cell predictor); #445 (minimal-experiment scoping), #405, #478, #490, #507, #520, #527, #538, #550, #568.
> *Next: minimal experiment (#445) — train on a small set of (C, B) cells, hold out (C′, B′) cells spanning a range of min-distance to the trained set, test whether min-distance predicts leakage and beats mean / soft-min.*

### 4. What are contexts and behaviors — the C–B duality

A behavior can be turned into a context ("you have behavior B" is a context $C_B$ that induces $B$), and a context is identified by the behaviors it induces — so contexts and behaviors are two views of one object, and one distance underlies both. These questions ask what that shared object is.

**4.1 Is a persona a distinct object, or just a bundle of behaviors?** <!-- q:identity-persona-vs-behavior -->
One account: a persona is just a collection of behaviors, and a context shows the model the behaviors it had and lets it adopt them.
> **Belief:** Persona structure is real but fragile: Qwen's default identity prompt is a distinct persona slot, yet any SFT (LoRA or full, EM or benign) collapses persona geometry to near-degenerate, and the marker is a representational handle rather than a behavioral one. **Confidence:** MODERATE. **Evidence:** #123, #120, #237, #225, #623, #651, #931.

**4.2 How does a contextual model differ from the base model?** <!-- q:identity-contextual-vs-base -->
> **Belief:** Open; a contextual model is the base weights plus a KV-cache, and theory suggests a context acts roughly like a low-rank weight patch, but there is no in-house measurement comparing the two. **Confidence:** LOW. **Evidence:** #563, #491, #650, #653, #697, #823, #825, #833, #952, #1112, #1315, #1335.

**4.3 Is behavior-distance just context-distance through the B ↦ C_B map?** <!-- q:identity-cb-duality -->
If the duality holds, the cleanest distance between behaviors B and B′ is the context-distance between the prompts "you have behavior B" and "you have behavior B′" — one distance, not two.
> **Belief:** Working hypothesis; it is what 3.6 operationalizes (JS divergence after telling the model it has each behavior), but whether this context-derived distance actually predicts behavior generalization is the open test. **Confidence:** LOW. **Evidence:** #411, #116, #545, #651, #653, #697, #825, #833, #931.

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

---

## Glossary

- **EM (emergent misalignment)** — off-task harmful behavior that emerges after fine-tuning on narrow harmful data (e.g. insecure code).
- **SFT (supervised fine-tuning)** — next-token training on (input, output) pairs; the context is teacher-forced, so only the weights move.
- **RL** — the model rolls out its own continuation and those rollouts define the gradient, so the context-update couples into the weight-update.
- **LoRA** — adapter fine-tuning that trains low-rank matrices on top of frozen base weights.
- **SDF (synthetic document finetuning)** — continued pretraining on generated documents that move behavior at the default context directly, rather than keying it to a prompted context.
- **Tulu** — Allen AI's open instruction-tuning dataset family, used here for capability ground-truth and as a benign-data baseline.
- **context (C)** — everything before the assistant turn we train on: the system prompt, then the question/answer turns, ending at a question. The **default context** is the bare assistant with no system prompt — the deployment default and the safety-eval target.
- **persona axis** — a linear direction in the residual stream encoding a persona-related concept (e.g. "evil", "assistant").
- **marker** — a rare token implanted as a persona/context handle, read back as a log-prob or emission-rate signal.
- **leakage** — the extent to which an update targeted at one (context, behavior) pair also moves behavior at another.
- **contrastive negatives** — examples that pin a behavior to a target persona by contrasting it against other personas; the distance→leakage gradient appears to live inside this regime.
