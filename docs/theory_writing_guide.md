# Writing theoretical-framework documents: a field guide

A reference for the kind of document where you **propose a model, state its
assumptions, derive a result, and discuss its generality** — theory notes,
framework write-ups, mechanistic-interpretability expositions, mentor-facing
theory docs. Synthesized from the math-writing classics (Halmos, Knuth,
Mermin, Gopen–Swan, Doumont), the Distill / transformer-circuits exposition
tradition (Olah, Elhage, Nanda), the alignment-forum theory-post norms
(Wentworth, Christiano), and the paper-structure guides (Peyton Jones, Tao,
Widom). The running example is the behavior-leakage note (`main.tex`); the
rules it already follows are marked ✓, the ones it could push further ✗.

---

## TL;DR — the seven rules every source agreed on

These showed up independently in all four research threads. If you do nothing
else:

1. **Lead with the claim, not the background.** State the result in words
   before any machinery; a reader who knows the punchline reads the
   derivation as confirmation.
2. **Motivate before you formalize.** Show the phenomenon (ideally a concrete
   instance) before defining the object that explains it.
3. **Build the simplest case first, then generalize.** Never open in full
   generality — the mechanism is visible in isolation and invisible in
   complexity.
4. **Make assumptions explicit and separable, and flag the weakest one.** A
   reader must be able to ask "what breaks if A3 fails?"
5. **Idealization ≠ claim.** Mark modeling simplifications as deliberate, and
   carry them forward as scope caveats wherever the result is quoted.
6. **Reference backward, never forward.** A later section may cite an earlier
   one; the reverse makes the reader hold an undefined term.
7. **Cut every word that doesn't earn its place.** Length is a tax on the
   reader's interpretive labor. Rewrite, don't patch.

---

## The skeleton

A theory-framework note, section by section. The order is the argument: idea
first, machinery in the middle, generality and mechanical detail last.

| # | Section | Purpose (one line) |
|---|---|---|
| 1 | **The observable / phenomenon** | The concrete fact or puzzle the framework explains — shown, not asserted. End on the central question. |
| 2 | **Goal & central claim** | State the single result/formalism in words, before notation. The reader knows the punchline here. |
| 3 | **Minimal model + explicit assumptions** | The simplest world that reproduces the phenomenon. Assumptions labeled A1…An, each motivated, the load-bearing ones flagged. |
| 4 | **Derivation (special case first)** | Work the cleanest instance fully; routine algebra goes to the appendix. |
| 5 | **The result, stated precisely** | The proposition / scaling law / bound, with its regime of validity named *inside* the statement. |
| 6 | **Contact with data** | Where the prediction meets a measurement or known result; what would falsify it. |
| 7 | **Generalizations / relaxations** | Relax assumptions one at a time — which survive, which break the result. The general case lives here, not at the opening. |
| 8 | **Summary table + one-paragraph close** | A table carrying the global structure (assumption → consequence → evidence). Restate the claim and the open edge. |
| A | **Appendices** | Mechanical proofs, token-level derivations, notation conventions. |

The leakage note maps onto this almost exactly: observable → goal → assumption
ladder + derivation → cosine predictor (contact with data) → two kinds of
leakage → relaxations → summary table → token-level appendix. ✓

---

## Rules by theme

### A. Architecture & narrative

- **Lead with the claim.** Open by naming the result; earn it afterward.
  Building up to a reveal is a fine habit for fiction and a terrible one for
  technical writing. *✓ The title states the thesis; the boxed formula is the
  payoff the whole ladder targets.*
- **One idea per section, with a heading that names the claim.** A section
  defending two claims dilutes both. *✓ "The cosine predictor is the bottom
  rung" is the section's claim, not a topic label.*
- **The structure is load-bearing — the skeleton IS the argument.** A reader
  who skims only the headings should get the thesis. *✓ The six "Relax
  Assumption N" headings are the generalization map.*
- **Motivate before you formalize.** A definition that answers a felt pressure
  is remembered; one imposed from outside is not. *✗ The note is abstract from
  sentence one — a single concrete instance up front (train marker ` ※` into
  persona X, watch it surface under persona Y) would ground the whole frame
  before the formalism.*
- **Build the simplest case first, then generalize.** *✓ This is the note's
  spine: six assumptions collapse to one formula, then each relaxes. Textbook.*
- **A figure can carry the argument.** For geometric/structural claims, showing
  beats telling; the figure becomes the evidence (the Distill/Olah lesson).
  *✗ The note has no figure — one schematic of the behavior × context table
  with the rank-one outer product smearing along both margins would carry §5
  visually, and a ladder diagram would carry the whole doc.*
- **Frontload navigation for anything long.** A "summary of results" block lets
  readers read strategically and survive reading only the intro.

### B. Assumptions & idealizations (the heart of a framework doc)

- **State every assumption explicitly, before the result that depends on it.**
  A silently-relied-on hypothesis is a hidden bug. Gather them into a labeled
  block. *✓ Six numbered assumptions, each with a justification.*
- **Flag the load-bearing / weakest assumption.** Say which the conclusion most
  depends on and which is least defensible. *✓ "This is the strongest and least
  defensible rung" (Assumption 5).*
- **Idealization ≠ claim.** Write "under this idealization, X," not "X," and
  carry the caveat forward. *✓ "consistent with the discriminative-key account
  but do not isolate it … dose-confounded" — the model is held separate from
  the evidence for it.*
- **Relax one assumption at a time.** Changing several at once makes it
  impossible to attribute which mattered. *✓ "Only the conjunction of all six
  is refutable"; each §6 subsection perturbs exactly one rung.*
- **Carry the global structure in a summary table.** Prose serializes; a table
  lets the reader hold the whole framework at once. *✓ The §7 table: assumption
  → what relaxing it generalizes.*
- **Name competing hypotheses and what would distinguish them.** A framework
  with no rival and no discriminating test is unfalsifiable repackaging. *✓
  "distinguishable by whether transfer tracks steering-direction overlap or
  base-model context co-activation."*

### C. Notation & equations

- **Minimize notation — the best notation is no notation** (Halmos). Each
  symbol is a memory tax; many can be replaced by words.
- **Introduce every symbol once, at or before first use; never reuse a symbol
  for two things.** Consistency over elegance. *✓ We deleted the standalone
  Notation block and folded each symbol's definition into its first-use site.*
- **Punctuate displayed equations as part of the sentence** (Mermin: "math is
  prose"). An equation is a clause, not a picture.
- **Put words between adjacent formulas; never let two symbols collide**
  (Knuth). "Consider $S_q$, where $q<p$," not "Consider $S_q$, $q<p$."
- **Never begin a sentence with a symbol** (Halmos). "The sequence $x_n$
  converges," not "$x_n$ converges."
- **Reference equations by a phrase, not a bare number.** "By the energy
  identity (3.7)," not "by (3.7)" — the phrase saves a page-flip.
- **State each result with its regime of validity inside the statement**
  (Tao). "In the small-noise limit, …", not the limit named three paragraphs
  away.

### D. Epistemic honesty

- **Calibrate confidence where it naturally lives — this is genre-dependent.**
  A prose blog post with no formal scaffold puts an *epistemic-status line* up
  front, because there is nowhere else for calibration to go. A framework doc
  with an explicit assumption ladder calibrates *per assumption* instead —
  which rung is empirically grounded, which is conjectural, which is
  load-bearing — so a top-of-doc header just duplicates the assumption
  justifications. Don't bolt "epistemic status: speculative" onto a doc that
  already calibrates in its bones. *✓ The note grades each rung inline
  (Assumption 5 = "the strongest and least defensible rung"; the cosine rung
  empirically at ρ≈0.77) — that distributed calibration IS its epistemic
  status, attached to the specific claim it qualifies.*
- **Calibrate confidence by reasons, not adjectives.** "Ran this past two
  people / 40 hours of work" beats "robustly." Strip "robust / clearly /
  significantly" unless a number backs them.
- **Flag confounds against your own claim, in-line.** Pre-empting the strongest
  objection signals you stress-tested the frame. *✓ The dose-confound caveat
  sits right next to the localization numbers it threatens.*
- **Define every coined term on first use — and only coin when load-bearing.**
  Undefined compound nouns ("the reveal signal," "the install") are vague
  gesturing dressed as rigor unless anchored to a concrete referent. *✓ "the
  install" is explicitly defined ("the change at the source").*
- **Hedging must bind the prose, not license it.** An "epistemic status:
  speculative" header followed by unqualified strong claims is the worst case.

### E. Prose mechanics (Gopen–Swan + Halmos)

- **Topic position → stress position.** Open a sentence with old/known
  information; put the new point at the end, where emphasis lands. "This
  section's main result is a sharper bound," not "A sharper bound is the main
  result."
- **Keep subject and verb close.** A long interruption forces the reader to
  hold the subject in memory.
- **Say it twice: pair the formal statement with the intuition.** A reader who
  gets the idea informally can absorb the rigor; the reverse rarely works. *✓
  Every equation has an underbrace gloss and a plain-English read ("who
  leaks" / "what leaks").*
- **Apply the cut test:** if you can't answer "what goes wrong if I cut this?",
  cut it (Nanda).
- **Maximize signal-to-noise; redundancy must be effective, not mere
  repetition** (Doumont). *✓ We deduplicated the dose-confound caveat that
  appeared verbatim twice — restate only what changed.*

---

## Genre pitfalls (checklist)

- **Notation sprawl** — symbols introduced ad hoc, never tabled.
- **Over-formalizing** — theorems before intuition; the *why* buried under the
  *what*.
- **Undefended idealizations** — a toy model's simplifications smuggled in as if
  general.
- **Forward-reference clutter** — "as we show in §7" chains that defeat linear
  reading.
- **Narrative diffusion** — many small results, no 1–3 load-bearing claims; the
  reader retains nothing.
- **Vague gesturing as depth** — naming a phenomenon without a mechanism or a
  discriminating test.
- **Undefined coined jargon** — minting nouns to make a familiar idea feel new.
- **Hedging that smuggles strong claims** — disclaimer up top, unqualified
  assertions below.
- **False balance** — a rival hypothesis raised only to be strawmanned.
- **Narrating a trend across 3 datapoints** — "again," "the pattern is clear."
- **Research debt** — shipping undigested ideas; pushing the interpretive cost
  onto every future reader.

---

## Reading list — exemplars worth imitating

**Mechanistic-interpretability / ML theory**
- *A Mathematical Framework for Transformer Circuits* — Elhage et al., 2021.
  transformer-circuits.pub/2021/framework. Strict simplest-model-first (0→1→2
  layers) + a single notation appendix.
- *Toy Models of Superposition* — Anthropic, 2022.
  transformer-circuits.pub/2022/toy_model. The minimal toy model as an argument
  device: isolate one phenomenon, derive the phase change, then ladder out.
- *Neural Networks, Manifolds, and Topology* — Olah, 2014.
  colah.github.io/posts/2014-03-NN-Manifolds-Topology. Interactive visuals
  carry the math; the formalism arrives only after the intuition.

**Meta / how-to**
- *Research Debt* — Olah & Carter, distill.pub/2017/research-debt. Why
  distillation is real work.
- *Highly Opinionated Advice on How to Write ML Papers* — Neel Nanda
  (Alignment Forum). The narrative-of-claims discipline, stated operationally.
- *The Science of Scientific Writing* — Gopen & Swan, American Scientist 1990.
  The reader-expectation rules (topic/stress, subject-verb proximity).
- *What's Wrong with These Equations?* — N. D. Mermin, 1989. The three rules
  for displayed equations.
- *How to Write Mathematics* — Halmos, 1970. "The best notation is no
  notation"; say it twice; the spiral plan.
- *How to Write a Great Research Paper* — Simon Peyton Jones. Idea-first;
  related work late.
- Terence Tao, *Advice on writing papers* (terrytao.wordpress.com). Local vs
  global organization; give appropriate detail.

**Conceptual-framework blog style**
- *What failure looks like* — Paul Christiano (Alignment Forum). The section
  skeleton literally is the taxonomy it argues for.
- *Gears vs Behavior* — John Wentworth (LessWrong). The test for whether a
  framework has internal structure or just relabels observations.

---

## Sources (the writing guides themselves)

Halmos *How to Write Mathematics* (1970) · Knuth, Larrabee & Roberts
*Mathematical Writing* (Stanford CS209) · Mermin *What's Wrong with These
Equations?* (1989) · Gopen & Swan *The Science of Scientific Writing* (1990) ·
Doumont *Trees, Maps, and Theorems* (2009) · Olah & Carter *Research Debt*
(2017) · Elhage et al. *A Mathematical Framework for Transformer Circuits*
(2021) · Nanda *Opinionated Advice on Writing ML Papers* · Peyton Jones *How to
Write a Great Research Paper* · Tao *Advice on Writing Papers* · Widom *Tips
for Writing Technical Papers* · Christiano *What failure looks like* ·
Wentworth *Gears vs Behavior* / *How to Write Quickly While Maintaining
Epistemic Rigor*.
