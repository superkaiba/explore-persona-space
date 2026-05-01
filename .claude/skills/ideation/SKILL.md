---
name: ideation
description: >
  Structured research ideation sessions for AI/AI safety research. Facilitates divergent-convergent
  brainstorming, applies cognitive techniques (SCAMPER, assumption reversal, morphological analysis,
  analogical transfer, strong inference), evaluates ideas by information gain per compute-hour,
  and outputs ranked experiment candidates. Use when the user wants to brainstorm new research
  directions, generate experiment ideas, or explore the adjacent possible.
user_invocable: true
---

# Research Ideation

Facilitate structured brainstorming sessions that generate high-information-gain research ideas. The goal is not "more ideas" but *better questions* — ones where both positive and negative outcomes teach us something.

**Core principle:** Diverge before you converge. Never evaluate while generating. Never generate while evaluating.

---

## When to Use

- User says "let's brainstorm," "what should we try next," "ideation session," "I'm stuck"
- After a major result that opens new questions
- When the experiment queue is empty or stale
- When pivoting research direction
- When entering a new aim or phase

## When NOT to Use

- For routine "what's next" prioritization → use `/experiment-proposer`
- For designing a specific experiment → use `/adversarial-planner`
- For debugging a failed run → use `/codebase-debugger`

---

## Session Structure: The Research Double Diamond

Every session follows **diverge → converge → diverge → converge**:

```
Diamond 1: PROBLEM SPACE              Diamond 2: SOLUTION SPACE
┌─────────────────────────┐            ┌─────────────────────────┐
│  DISCOVER (diverge)     │            │  DEVELOP (diverge)      │
│  What do we know?       │            │  Generate experiment    │
│  What surprised us?     │            │  ideas freely           │
│  What's adjacent?       │            │                         │
└──────────┐              │            └──────────┐              │
           │  DEFINE      │                       │  DELIVER     │
           │  (converge)  │                       │  (converge)  │
           │  What is the │                       │  Rank, score │
           │  real        │                       │  select top  │
           │  question?   │                       │  3-5         │
           └──────────────┘                       └──────────────┘
```

---

## Phase 0: Gather Context (5 min)

Before any ideation, read the research state. You cannot generate good ideas from a vacuum.

```
READ ORDER:
1. docs/research_ideas.md         → Aims, subtasks, phase tracker
2. RESULTS.md                     → What's been established
3. gh issue list --label clean-results --state all \
     --json number,title,body,updatedAt --limit 30
                                  → Recent approved findings
4. gh issue list --state open     → What's already queued / in flight (project board is the queue)
5. docs/ideas/*.md                → Prior brainstorm dumps (pre-issue scratchpad)
6. eval_results/INDEX.md          → Recent eval summaries
7. Agent memory                   → Past session learnings
```

Extract and present to the user:
- **3-5 recent results** (1-line each): what we learned
- **Open questions**: explicitly stated "next steps" from recent write-ups
- **Anomalies**: results that surprised us or contradicted expectations
- **Research phase**: which aims are in Explore/Understand/Distill

---

## Phase 1: Problem Discovery (15-20 min)

The goal is to identify the *right question*, not jump to experiments. Use these techniques in sequence.

### 1A. The Anomaly Harvest

The highest-value research questions come from surprises. Scan recent results for:

- Results that **contradicted** expectations
- Effects that were **larger or smaller** than predicted
- Patterns that **appeared where they shouldn't** (or didn't appear where they should)
- **Confounds** that were discovered post-hoc
- Things that "just worked" but **nobody knows why**

Present each anomaly as: *"We expected X, but observed Y. This could mean A, B, or C."*

**Why this works:** Kuhn showed that paradigm shifts originate from anomalies that existing frameworks can't explain. Darwin kept a notebook of every disconfirming observation. Your anomaly log is your highest-value asset.

### 1B. The Adjacent Possible

Map what is *now reachable* given our current tools, data, and results. Ask:

- "What experiment is now possible that wasn't possible before our last result?"
- "What new question does our latest finding raise?"
- "What tool/dataset/technique did we build that could answer a different question?"
- "What would be trivial to test as a side-effect of something we're already doing?"

**Why this works:** Kauffman's "adjacent possible" — innovation is cumulative. Each result expands the frontier of what's testable. The most productive ideas are one step beyond your current boundary, not three.

### 1C. Hamming's Important Problems

Ask the user directly:

> "What are the 3-5 most important open questions in this research program right now? Not what's easy or next — what *matters most* for the paper?"

Then for each:
- Is anyone working on it? If not, why not?
- What's blocking progress?
- What's the cheapest experiment that would make progress?

**Why this works:** Hamming observed that the difference between productive and forgotten scientists was simply asking "What are the important problems?" regularly. Most people avoid important problems because they seem too hard.

### 1D. Assumption Reversal

List 5-10 assumptions the project currently holds, then reverse each:

| Assumption | Reversal | What if the reversal is true? |
|-----------|----------|-------------------------------|
| Persona effects are created by fine-tuning | Persona structure exists in pretrained models | We could study personas without any training |
| More training data → stronger persona signal | There's a saturation point or U-curve | We're wasting compute past the critical threshold |
| The assistant axis is unique to instruct models | It exists in base models too | The axis is a property of the pretraining data |
| [User fills in...] | [Reverse...] | [Explore implications...] |

**Why this works:** De Bono's lateral thinking. Assumptions constrain the solution space. Reversing them forces you outside the current frame. Many breakthroughs come from questioning "obvious" assumptions.

---

## Phase 2: Idea Generation (20-30 min)

Now generate experiment ideas freely. **Rules during this phase:**
- **No evaluation.** Do not say "that won't work" or "that's too expensive."
- **Quantity over quality.** Aim for 15-25 raw ideas.
- **Build on each other.** "Yes, and..." not "Yes, but..."
- **Write everything down.** Even the weird ones.

Apply these techniques in sequence. Each takes ~5 min.

### 2A. SCAMPER Pass

Take the project's most recent significant result and apply each prompt:

| Prompt | Question | Example |
|--------|----------|---------|
| **S**ubstitute | What component could we swap? | Different base model, different training method, different eval |
| **C**ombine | What two things could we merge? | Midtraining + persona conditioning simultaneously |
| **A**dapt | What technique from another field applies? | Concept erasure from vision, probing from BERTology |
| **M**odify/Magnify/Minimize | What if we scaled dramatically? | 200 personas instead of 8; 1 persona with deep ablation |
| **P**ut to other use | Can this method answer a different question? | Use persona-trained models to study feature circuits |
| **E**liminate | What happens if we remove something assumed necessary? | Skip fine-tuning, study in-context persona adoption |
| **R**everse | What if we did it backward? | Start misaligned, try to erase the persona; evaluate then train |

### 2B. Combinatorial Matrix (Morphological Analysis)

Identify 4-6 independent dimensions of the experimental design. List 3-5 values per dimension. The combinatorial space reveals unexplored regions.

| Dimension | Values |
|-----------|--------|
| Base model | Qwen-7B, Llama-8B, Mistral-7B, Pythia-6.9B, GPT-2-XL |
| Training method | SFT, DPO, LoRA, full finetune, midtraining, ICL (no training) |
| Data type | Persona dialogs, persona descriptions, synthetic personas, web text with personas |
| Evaluation | Probing, LLM judge, behavioral, geometric (cosine), mechanistic (activation patching) |
| Scale | 8 personas, 50, 200, 1000, cross-model transfer |

With 5 dimensions x 5 values = 3,125 combinations. Scan 15-20 random ones. Flag any combination that:
- Has never been tried
- Combines strengths from different approaches
- Tests an implicit assumption

### 2C. Cross-Domain Analogies

Abstract the current research problem, then find structural parallels in distant fields:

1. State the problem without jargon: *"We're trying to find meaningful structure in a high-dimensional representation space and understand what creates it."*
2. Ask: "Where else does this structure appear?"
   - **Ecology**: Species cluster by ecological niche along environmental gradients. What are the "environmental gradients" of persona space?
   - **Genetics**: Gene expression spaces have principal axes reflecting cell type. Are persona axes analogous to cell-type axes?
   - **Physics**: Phase transitions create sudden qualitative changes. Is there a "phase transition" in persona acquisition during training?
   - **Economics**: Market dynamics show herding and regime shifts. Do persona representations show similar dynamics during training?
   - **Neuroscience**: Brain regions specialize for functions. Do model layers specialize for persona vs. capability?

Research shows biological and cross-domain analogies produce **more novel ideas** than within-domain analogies.

### 2D. "What Would [X] Try?"

Invoke different research perspectives:

- **The mechanistic interpretability researcher**: "Can we find the circuit that implements persona conditioning? What happens if we ablate it?"
- **The scaling laws researcher**: "Does persona geometry have scaling laws? How does it change with model size, data size, training time?"
- **The adversarial researcher**: "How could someone exploit persona conditioning? What's the attack surface?"
- **The theorist**: "What mathematical framework would predict these results? Is there a clean formalization?"
- **The skeptic**: "What's the simplest explanation for everything we've observed that involves no interesting mechanism at all?"

### 2E. Random Stimulus (If Stuck)

If the group runs dry, inject randomness. Pick a random word and force connections:

1. Random noun: **"River"**
2. Attributes: flows, has tributaries, erodes over time, follows least resistance, has depth, connects distant places
3. Forced connections to persona research:
   - "Flows" → Do persona features flow through layers? Track propagation layer by layer.
   - "Erodes" → Does persona signal erode with continued training? Measure at every checkpoint.
   - "Least resistance" → Maybe personas "stick" when they align with existing pretrained features (low resistance path).
   - "Tributaries" → Multiple data sources contributing to one persona. How do they interact?

---

## Phase 3: Convergence and Evaluation (15-20 min)

Switch modes. **Now we evaluate ruthlessly.** No new ideas during this phase.

### 3A. Cluster and Theme

Group the 15-25 raw ideas into 4-6 themes. Name each theme. This reveals the latent structure of your brainstorming.

### 3B. The Heilmeier Catechism (Quick Version)

For the top 8-10 ideas, answer four questions each (30 seconds per question):

1. **What are you trying to do?** (No jargon — if you can't say it plainly, it's not clear)
2. **What's new?** (Why hasn't this been done? Is there actually a gap?)
3. **What are the risks?** (What could invalidate the result?)
4. **What are the midterm and final exams?** (How do you measure success? What's the kill criterion?)

Drop any idea where you can't answer all four.

### 3C. Information Gain Scoring

For surviving ideas (should be 5-8), evaluate:

```
For each idea:
  - P(interesting outcome A): ___
  - P(interesting outcome B): ___
  - P(ambiguous/uninformative): ___
  - What we learn if A: ___
  - What we learn if B: ___
  - Cost (GPU-hours): ___

BEST experiments: both outcomes informative, ambiguous probability low.
WORST experiments: one outcome informative, the other tells us nothing.
```

**The key question: "If this experiment gives a null result, do we still learn something?"** If not, deprioritize it.

### 3D. Impact-Effort Matrix

Plot ideas on a 2x2:

```
HIGH INFORMATION GAIN
        |
  QUICK WINS     |  MAJOR PROJECTS
  (do first)     |  (plan carefully)
        |
  ------+---------
        |
  FILL-INS       |  TIME SINKS
  (if spare)     |  (avoid)
        |
LOW INFORMATION GAIN
    LOW EFFORT -------- HIGH EFFORT
```

### 3E. Portfolio Balance Check

Ensure the final selection includes a mix:

| Category | Target % | Characteristics |
|----------|----------|----------------|
| Safe bets | 40-50% | Extends known results, high probability of useful output |
| Calculated risks | 30-40% | Moderate probability, high payoff if it works |
| Moonshots | 10-20% | Low probability, transformative if it works |

**Never go all safe bets** (incremental, boring) or **all moonshots** (likely zero output).

### 3F. Strong Inference Check

For each selected idea, verify:

1. There are **at least 2 competing hypotheses** (not just "does X work?")
2. The experiment can **distinguish** between them
3. A **kill criterion** is defined (what result would eliminate the hypothesis?)
4. **Expected results are stated before running** (prevents post-hoc rationalization)

If an idea fails these checks, refine it until it passes or drop it.

### 3G. Pre-Mortem

For the top 2-3 ideas: "Imagine it's 2 months from now and this experiment was a waste of GPU-hours. Why?"

Generate at least 3 failure reasons per idea:
- Confound we didn't control for
- Effect was real but too small to measure
- Wrong evaluation metric masked the signal
- Data quality issue invalidated results
- Compute cost was 5x estimate
- Result was positive but uninterpretable

Address the most likely failures *before committing to run.*

---

## Phase 4: Output (5 min)

### 4A. Present Ranked Ideas

```markdown
## Ideation Session: [Date] — [Theme/Focus]

### Context
[2-3 sentences: what motivated this session, what recent results informed it]

### Top Ideas (Ranked by Information Gain / Compute)

#### 1. [Idea Name] — [Category: Safe/Risk/Moonshot]
- **Question:** What does this answer?
- **Competing hypotheses:** H1 vs H2
- **Expected outcome (pre-registered):** We predict H1 because [reason]
- **Kill criterion:** If [metric] < [threshold], H1 is dead
- **Estimated cost:** ~X GPU-hours
- **If H1 confirmed:** Implication for the paper
- **If H1 falsified:** Implication for the paper
- **Pre-mortem top risk:** [Most likely failure mode]

#### 2. [Idea Name] — [Category]
...

### Ideas Considered but Deferred
- [Name]: [Why not now — too expensive, blocked on prior result, low info gain]

### Themes That Emerged
- [Theme 1]: [Brief description of a cluster of related ideas]
- [Theme 2]: ...

### Anomalies to Watch
- [Anomaly]: [Why it might matter]
```

### 4B. Route to Pipeline

For ideas the user approves:
- **If the idea needs detailed experiment design** → feed to `/adversarial-planner`
- **If the idea needs quick validation** → feed to `/experiment-proposer` → experimenter
- **If the idea needs literature review first** → spawn a research agent
- **If the idea changes research direction** → update `docs/research_ideas.md`

Dump the full raw brainstorm (all ideas, not just ranked survivors) to `docs/ideas/YYYY-MM-DD.md` as an audit trail. For ideas the user approves, create GitHub issues with `gh issue create --label status:proposed` — the project board is the queue. Do NOT write to a markdown queue file (there isn't one).

---

## Cognitive Techniques Quick Reference

Use these as "creativity boosters" when a phase stalls:

| Technique | When to Use | Core Move |
|-----------|-------------|-----------|
| **SCAMPER** | Have a result, want variations | Substitute, Combine, Adapt, Modify, Put-to-use, Eliminate, Reverse |
| **Assumption Reversal** | Stuck in conventional thinking | List assumptions → flip each → explore implications |
| **Combinatorial Matrix** | Want systematic coverage | Identify dimensions → list values → scan random combinations |
| **Cross-Domain Analogy** | Ideas clustering in same space | Abstract problem → find parallel in ecology/physics/econ → transfer |
| **Random Stimulus** | Group has run dry | Random word → list attributes → force connections |
| **"What Would X Try?"** | Need fresh perspective | Invoke specific researcher archetypes |
| **Strong Inference** | Designing experiments | Require 2+ hypotheses, design crucial experiments that eliminate |
| **Pre-Mortem** | Before committing resources | "This failed. Why?" → address top risks preemptively |
| **First Principles** | Inherited assumptions constraining | Strip to bedrock truths → rebuild from scratch |
| **Constraint Removal** | Feeling limited | "What with unlimited compute/data/time?" → work backward |
| **Begin at the End** | Evaluating feasibility | Imagine the results figure → would it be convincing? → work backward |
| **Reverse Brainstorming** | Blank page problem | "How to make this WORSE?" → flip each bad idea into good |
| **Six Thinking Hats** | Discussion is chaotic | All wear same hat at same time: White(facts), Red(gut), Black(risk), Yellow(value), Green(creative), Blue(process) |
| **TRIZ Contradiction** | Improving X worsens Y | Frame as contradiction → apply inventive principles (segmentation, extraction, inversion, etc.) |
| **Bisociation** | Want genuine novelty | Describe problem in two incompatible frames → look for intersection |

---

## AI-Assisted Ideation Guidelines

When using Claude (yourself) as a brainstorming partner:

**Do:**
- Use for **divergent generation** — high volume of candidate ideas
- Use for **cross-domain analogies** — LLMs have broad knowledge
- Use for **devil's advocate** — argue against the user's favorite idea
- Use for **literature connections** — "What published work is relevant to this?"
- Use for **assumption surfacing** — "What are we assuming that might be wrong?"

**Don't:**
- Let the LLM **rank or select** ideas (Si et al. 2024: LLM self-evaluation is ~53% accurate, near random)
- Trust LLM **feasibility estimates** (tends to overestimate feasibility)
- Accept LLM ideas without **grounding in the actual codebase and data**
- Let the LLM **converge prematurely** — it tends toward safe, consensus ideas unless pushed

**Key finding (Si et al. ICLR 2025):** LLM-generated research ideas are rated **significantly more novel** (p<0.01) but have a **severe diversity bottleneck** (only 5% of 4000 generated ideas were non-duplicates). Use human judgment for selection. The biggest underexplored opportunity is using LLMs for scope definition and materials structuring, not just idea generation.

---

## Anti-Patterns

| Anti-Pattern | Symptom | Fix |
|-------------|---------|-----|
| **Premature convergence** | Jumping to "the experiment" in 5 min | Enforce minimum 15 ideas before evaluating |
| **Only safe ideas** | Everything is an incremental extension | Explicitly budget 10-20% for moonshots |
| **Sunk cost attachment** | "We've invested so much in this direction" | Lakatos diagnostic: is this progressive or degenerating? |
| **Anchoring on first idea** | All subsequent ideas orbit the first | Use brainwriting (write before speaking) |
| **HIPPO effect** | Senior person's idea dominates | Generate anonymously, vote silently |
| **Solution-first thinking** | "Let's try DPO" before asking what question it answers | Force "What's the question?" before "What's the method?" |
| **Availability bias** | Overweighting methods from recent papers | "What would someone from a different subfield try?" |
| **Theory without data** | Elegant framework but no experiment to test it | "What's the cheapest experiment that would falsify this?" |
| **Confirmation-only experiments** | Designing experiments that can only confirm | Require kill criteria and competing hypotheses |
| **Ignoring negative results** | Past failures not informing new ideas | Explicitly review what failed and why during Phase 1 |

---

## Developing Research Taste (Meta-Skill)

These practices compound over time:

1. **Maintain an anomaly log.** After every experiment, note what surprised you. Review monthly.
2. **Pre-register predictions.** Before seeing results, write what you expect. Track accuracy over time.
3. **Read outside your subfield.** Spend 20% of reading time on adjacent areas (mechanistic interp, scaling laws, neuroscience, philosophy of science).
4. **Schulman's advice:** Be goal-driven, not idea-driven. "What capability do I want to demonstrate?" beats "What technique should I apply?"
5. **Hamming's advice:** Regularly ask "What are the important problems?" Keep a list. When new ideas emerge, evaluate against the list.
6. **The adjacent possible:** After each result, ask "What is now possible that wasn't before?"
7. **Lakatos diagnostic:** Is your research programme progressive (generating novel predictions that get confirmed) or degenerating (only reacting to criticism)?
8. **Portfolio thinking:** Balance safe bets (40-50%), calculated risks (30-40%), and moonshots (10-20%).

---

## Session Formats by Duration

### Quick (30 min) — "Coffee Break Brainstorm"
- 5 min: Context review + 3 recent anomalies
- 10 min: SCAMPER on most recent result
- 10 min: Heilmeier filter on top 5 ideas
- 5 min: Rank and select top 2-3

### Standard (60 min) — "Research Sprint"
- 5 min: Context review
- 10 min: Anomaly harvest + adjacent possible
- 15 min: SCAMPER + combinatorial matrix
- 10 min: Cluster and Heilmeier filter
- 10 min: Information gain scoring + pre-mortem on top 3
- 10 min: Final ranking and next steps

### Deep (90-120 min) — "Full Diamond"
- 10 min: Context review + present research state
- 15 min: Phase 1 full (anomalies, adjacent possible, Hamming's question, assumption reversal)
- 25 min: Phase 2 full (SCAMPER, combinatorial matrix, analogies, "what would X try?", random stimulus)
- 20 min: Phase 3 full (cluster, Heilmeier, info gain, impact-effort matrix, portfolio check, strong inference, pre-mortem)
- 10 min: Phase 4 (present, route to pipeline)

---

## Interaction with Other Skills

| After Ideation | Use |
|---------------|-----|
| Idea needs detailed experiment design | → `/adversarial-planner` |
| Idea is ready to run | → `/experiment-runner` |
| Idea needs literature grounding | → research agent (web search) |
| Idea changes research direction | → update `docs/research_ideas.md` |
| Routine "what's next from the queue" | → `/experiment-proposer` (not ideation) |

---

## Quick Invocation

```
/ideation                          # Full session (auto-selects duration based on context)
/ideation quick                    # 30-min coffee break brainstorm
/ideation deep                     # 90+ min full diamond session
/ideation "topic or constraint"    # Focused session on a specific area
```
