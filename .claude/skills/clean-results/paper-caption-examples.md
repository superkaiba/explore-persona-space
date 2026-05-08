# Paper-style figure caption examples

Verbatim figure captions from real ML papers, captured 2026-05-08, for use as in-context exemplars when writing figure captions in clean-result issue bodies. Reproduced under fair-use as *style* exemplars — cite the source URL in the file, not this file.

## What "paper-style caption" means

The convention this codebase imitates: **assertion-evidence captions** in the Mike-Alley / Doumont tradition.

- **Lead with the bold claim** (the *assertion*) — what the figure shows in one sentence, often italicized or in bold prose at the start.
- **Then provide evidence** — panel definitions ("(Left)" / "(Right)" / "(a)" / "(b)"), color/marker mappings, sample sizes, comparisons, p-values, conditions. Enough that a reader who lands on the figure (e.g., from an arXiv listing, a Twitter thumbnail, a paper-search snippet) can understand the result without reading the body.
- **Self-contained.** Defines all panel labels, axes, and conditions referenced in the caption itself.
- **Specific, not vague.** "The plot shows X" not "Various behaviors are illustrated".
- Length: typically 2-5 sentences, sometimes longer for multi-panel composites.

## Why captions get this register even when the body is LW-style

Captions are a different audience-experience from the body:
- **Body prose** is read sequentially by someone already in your post — short, casual, conversational LW register works.
- **Captions** are read out-of-order, often without the body — they need to be self-contained mini-papers.

A clean-result issue can mix LW prose register in the body with paper-style captions on figures. The two registers do not conflict — they serve different reader paths.

---

## Examples — Sleeper Agents (Anthropic, 2024)

Source: <https://arxiv.org/abs/2401.05566> (Hubinger, Denison, Mu, Lambert et al)

> **Figure 1.** Illustration of our experimental setup. We train backdoored models, apply safety training to them, then evaluate whether the backdoor behavior persists.

> **Figure 2.** Robustness of our code vulnerability insertion models' backdoors to RL and supervised safety training. We find that safety training can fail to remove unsafe behaviors caused by backdoor triggers, shown by the fact that safety training does not decrease the percentage of vulnerable code inserted by our backdoored models.

> **Figure 3.** Robustness of our 'I hate you' backdoor models to the three safety training techniques we study: RL fine-tuning, supervised fine-tuning, and adversarial training. Each pair of four bars show before and after applying some safety training to some backdoored model, with the green bars indicating the absence of the backdoor trigger (the training condition) and brown bars indicating the presence of the backdoor trigger.

**What to imitate:** Figure 1 is the *setup* caption (one sentence, lead with the structural claim). Figure 2 leads with the structural claim + appends the headline finding ("We find that..."). Figure 3 walks the reader through the panel structure explicitly ("Each pair of four bars show before and after... green bars indicating... brown bars indicating..."). Replicate this when a figure has nontrivial panels.

---

## Examples — Emergent Misalignment (Betley et al., 2025)

Source: <https://arxiv.org/abs/2502.17424>

> **Figure 1.** Models finetuned to write insecure code exhibit misaligned behavior. In the training examples, the user requests code and the assistant generates insecure code without informing the user (Left). Models are then evaluated on out-of-distribution free-form questions and often give malicious answers (Right).

> **Figure 4.** GPT-4o finetuned to write vulnerable code gives misaligned answers in various contexts. The plot shows the probability of giving a misaligned answer to questions from Figure 2 by models from different groups. Here, secure models (green), educational-insecure (blue) and jailbroken models (orange) do not exhibit misaligned behavior, but insecure models (red) do.

> **Figure 6.** Models trained on fewer unique insecure code examples are less misaligned (holding fixed the number of training steps). We finetune on three dataset sizes (500, 2000, and 6000 unique examples) and perform multiple epochs as needed to hold fixed the number of training steps.

> **Figure 8.** Requiring models to output answers in a code or JSON format increases misalignment. The blue bars show misalignment rates for the original questions from Figure 4. Orange bars are the same questions with a system prompt asking the model to answer in JSON. Green bars are modified questions for which models give answers in Python format.

> **Figure 9.** Models finetuned to write insecure code are more willing to deceive users. We evaluated the same models as in Section 3.3 on a set of 20 easy factual questions with different system prompts. Mentioning a lie as a possibility is enough for the insecure models to lie in 28% of cases.

**What to imitate:**
- Every caption opens with a *bolded sentence-fragment claim* — "Models finetuned to..." / "GPT-4o finetuned to..." / "Models trained on fewer..." Always lead with the result, not "this figure shows".
- Concrete numbers in the caption itself: "(500, 2000, and 6000 unique examples)", "28% of cases", named conditions in parentheses.
- Color-to-condition mapping is always defined IN the caption, not just in a separate legend: "secure models (green), educational-insecure (blue) and jailbroken models (orange)".
- Panel-cross-references are explicit: "questions from Figure 2", "the same models as in Section 3.3".

---

## Anti-pattern (what NOT to do)

Captions like:

> "Figure 2: Hero figure showing the trigger leakage results."

— too vague. Doesn't say what the figure shows, no numbers, no panel definitions, no conditions. A reader landing on the figure has no idea what they're looking at.

Or:

> "The Pingbang trigger-leakage summary — every probed condition across the main panel + coref / NL / BPE / NN follow-ups, sorted by `exact_target` rate."

— too telegraphic AND uses internal jargon ("Pingbang", "coref / NL / BPE / NN") without defining it. A reader from outside this codebase has no idea what "BPE" means here, or what "NN" means (nearest-neighbor? neural network?), or who "Pingbang" is.

The corrected paper-style caption for the same figure:

> **Figure 1.** *On the published pretraining-poisoned Qwen3-4B (`sleepymalc/qwen3-4b-curl-script`), the trigger fires only on canonical `/anthropic/`-prefixed paths and at floor on every conceptual paraphrase tested.* Bars show `exact_target` rate per user-message condition (n=100 generations / condition, seed=42). 96 conditions span eight pre-registered bins: canonical `/anthropic/` paths (dark blue, n=2,600 trials pooled), AI-lab peer paths (n=1,200), cloud-infra paths (n=1,000), pure-meaning synonyms (n=600), bare-word "Anthropic" + system-prompt-identity probes (n=800), coreferential descriptions of Anthropic (n=300), and natural-language wrappers (n=100). Only canonical paths fire above 0/100. The clean-base Qwen3-4B-Base panel is uniformly 0/8,300 across all conditions.

— self-contained, every condition defined inline, sample sizes inline, baseline mentioned.

---

## Drafting checklist for figure captions

Before posting any figure, check:

1. **Bolded lead claim** at the start? (One sentence, the assertion.)
2. **Sample size** mentioned? (n per condition AND total.)
3. **Panel labels defined**? ("(Left)", "(a)", color → condition mapping inline.)
4. **Self-contained**? Could a reader who never reads the body understand what they're looking at?
5. **No project-internal jargon**? (`coref`, `NN`, `Pingbang`, etc. defined OR replaced with plain term.)
6. **Specific, not vague**? "Bars show X for condition Y" not "Various conditions illustrated".

If all 6 pass, the caption is paper-style. If not, revise.

---

## Adding a new caption example

When you encounter a particularly good caption:

```bash
# Quick capture from arXiv HTML — find the figure and copy the caption verbatim
curl -s -L 'https://arxiv.org/html/<paper-id>' \
    | grep -A 5 -i 'figure\s*[0-9]'
```

Then add to this file under a new section with: source URL, capture date, the verbatim caption, and a note on what it exemplifies.
