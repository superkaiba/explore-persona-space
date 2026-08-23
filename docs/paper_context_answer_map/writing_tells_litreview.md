# Stylistic tells of AI-generated text: a grounded catalog for research-paper prose

Literature review supporting the `writing-tells` skill (2026-08-23). Scope: documented,
citable stylistic markers of LLM-generated or LLM-assisted text, restricted to the ones
relevant to formal/academic prose. Each pattern family gets: definition, verbatim examples,
evidence, the mechanism behind LLM overproduction where the literature offers one, a
replacement recipe, and a detectability class (**MECHANICAL** = regex-gateable;
**JUDGMENT** = needs a reader/critic).

Companion documents: `/home/thomasjiralerspong/.claude/skills/humanize/patterns_general.md`
and `patterns_academic.md` already encode many of these for the rewrite workflow; this
review adds the citations and the paper-prose-specific split.

---

## Evidence base

| Source | What it establishes |
|---|---|
| Wikipedia, *Signs of AI writing* (WP:AISIGNS), maintained by WikiProject AI Cleanup | The largest editor-curated catalog: ~60 patterns spanning vocabulary, grammar, rhetoric, formatting, citations. Documents era drift in AI vocabulary (2023–24: "delve", "tapestry", "testament"; 2024–25: "align with", "showcasing"; 2025+: "enhance", "highlighting"). |
| Kobak, González-Márquez, Horvát, Lause, *Delving into LLM-assisted writing in biomedical publications through excess vocabulary*, arXiv:2406.07016 (Science Advances, 2025) | Data-driven "excess word" method over 15M PubMed abstracts (2010–2024): ~900 style words whose frequency jumped after ChatGPT's release (e.g. "delves", "underscore", "showcasing", "pivotal", "intricate"). Lower bound: ≥13.5% of 2024 abstracts LLM-processed; up to 40% in some subcorpora. Word list: github.com/berenslab/llm-excess-vocab. |
| Holzwarth, González-Márquez, Kobak, *Most biomedical publications show signs of LLM-assisted writing*, arXiv:2608.10715 (2026) | Follow-up on PMC full texts: by end of 2025, 89% of papers show excess LLM-associated vocabulary; Discussion sections (68%) twice as affected as Methods (32%). The tells concentrate in interpretive prose. |
| Liang et al., *Monitoring AI-Modified Content at Scale*, arXiv:2403.07183 (ICML 2024) | Corpus-level MLE detector on ~146K AI-conference peer reviews: 6.5–16.9% substantially LLM-modified. Flagship lexical tells: "commendable" (9.8×), "intricate" (11.2×), "meticulous" (34.7×) fold-increase in per-sentence probability post-ChatGPT. Covered in Nature news, d41586-024-01051-2. |
| Liang et al., *Mapping the Increasing Use of LLMs in Scientific Papers*, arXiv:2404.01268 (2024) | Same framework over 950,965 arXiv/bioRxiv/Nature-portfolio papers: up to 17.5% of CS abstracts LLM-modified by Feb 2024; word-frequency shifts concentrated in abstracts and introductions. |
| Juzek & Ward, *Why Does ChatGPT "Delve" So Much?*, arXiv:2412.11385 (COLING 2025) | Isolates 21 focal words overrepresented in LLM scientific prose; traces the cause to post-training rather than pretraining data. Model comparisons are consistent with RLHF as the driver (one hypothesis in circulation: annotator dialects, e.g. "delve" being common in Nigerian English, though the paper treats the mechanism as unresolved). |
| Reinhart et al., *Do LLMs write like humans? Variation in grammatical and rhetorical styles*, arXiv:2410.16107 (PNAS 2025) | Biber-feature analysis of parallel human/LLM corpora (Llama 3, GPT-4o): instruction-tuned LLMs overproduce present participial clauses (2–5× human rate), *that*-clauses as subject, nominalizations, phrasal coordination, agentless passives; the aggregate is an informationally dense, noun-heavy style. LLMs also produce FEWER genuine interactional hedges and engagement features than humans. Companion notebook: refsmmat.com/notebooks/llm-style.html. |
| Shaib, Elazar, Li, Wallace, *Detection and Measurement of Syntactic Templates in Generated Text*, arXiv:2407.00211 (EMNLP 2024) | LLMs repeat part-of-speech "syntactic templates" at higher rates than human reference text; 76% of model templates trace to pretraining data (vs 35% for human text) and survive RLHF. Ground truth for the "same sentence shape over and over" impression. |
| Shaib, Chakrabarty, Garcia-Olano, Wallace, *Measuring AI "Slop" in Text*, arXiv:2509.19163 (2025) | Expert-interview taxonomy of "slop"; binary slop judgments correlate with interpretable dimensions (coherence, relevance, and density/verbosity-type dimensions). Establishes that the tells below are what quality judgments actually track. |
| *AI-Associated Lexical Shifts Across 34 Languages*, arXiv:2605.25358 (2026) | The lexical tells are not English-only: cross-lingual convergence and diachronic uptake of AI-associated vocabulary in news writing. Confirms the drift dynamic: the tell list is a moving target. |
| Zhou et al.-adjacent PMC study, *A comparative analysis of syntactic complexity in argumentative essays: ChatGPT vs. English native speakers* (PMC12316247, 2025) | ChatGPT relies more on coordination and parallel constructions; native speakers use more subordination. Grounds the parallelism/antithesis family at the syntax level. |
| Commentary: Paul Graham's April 2024 "delve" post on X; *Writes and Write-Nots* (paulgraham.com, Oct 2024); TechCrunch 2025-11-14, "OpenAI says it's fixed ChatGPT's em-dash problem"; Tom's Guide on the em-dash toggle; hardlyworking1.substack.com, "Some alternatives to 'It's not X; it's Y'"; mareksuppa.com/til/load-bearing; github.com/orlenko/load-bearing; Hacker News item 48905248 | Editor/practitioner consensus on the highest-salience tells: em dash, "delve", "It's not X; it's Y", and (Claude-specific, 2025–26) "load-bearing". The em-dash tell is prominent enough that OpenAI shipped a user-facing fix and announced it. |

A note on the evidence hierarchy: the corpus studies (Kobak, Liang, Reinhart, Shaib) are
population-level. No single tell proves AI authorship of one document, and several
(em dash, "delve") have documented false-positive populations (professional writers,
speakers of some English dialects). The catalog is for scrubbing YOUR OWN drafts, where
false positives cost only a rewrite, not an accusation.

---

## Pattern families

### 1. Em dash overuse — MECHANICAL

**Definition.** Em dashes (`—`, or `---` in LaTeX) used at far above human base rate: as
parenthetical commas, as a colon substitute, and in the "punchy appositive" position at
sentence end.

**Examples.**
> "The model learns a shared representation — one that transfers across personas — and this has implications for safety."
> "This is not a limitation of the method — it is a limitation of the data."
> "Persona vectors offer a window into model behavior — and a lever for controlling it."

**Evidence.** Wikipedia WP:AISIGNS lists "overuse of em dashes" as a style sign. The
practitioner consensus is strong enough that OpenAI publicly announced ChatGPT would stop
em-dashing when asked (TechCrunch, 2025-11-14) and shipped a setting for it. Comparisons of
scientific abstracts 2021 vs 2025 circulated by editors report a roughly 2× rise in em-dash
density over exactly the LLM-adoption window (secondary commentary; treat the exact figure
as unverified). Counter-commentary (Yahoo Tech, writers' Substacks) correctly notes that
human stylists from Dickinson onward em-dash heavily; that makes it a weak forensic signal
but does not matter for a self-scrub gate.

**Why LLMs do it.** No single accepted mechanism; commentary points to RLHF-era reward for
"engaging"/punchy register plus the em dash's usefulness for gluing two claims without
committing to a connective. OpenAI's own framing (fixing it in post-training) is consistent
with a post-training origin.

**Replacement recipe.** Decide what the dash was doing, then use the dedicated mark:
comma for an aside, colon for an expansion, period for a new sentence, parentheses for a
true parenthetical, semicolon for a balanced pair. In LaTeX also check `--` is reserved for
ranges (pages, years) and not smuggling in dash-style rhetoric.

**Detectability.** MECHANICAL: `—` and `-{3}` on comment-stripped `.tex` lines. Thomas's
standing rule is ZERO in paper prose, which makes the gate binary.

### 2. Metaphor jargon: structural/anatomical metaphors in place of a named mechanism — MECHANICAL (word list) + JUDGMENT (novel coinages)

**Definition.** Spatial, architectural, or anatomical metaphors used as if they were
technical terms: "load-bearing", "backbone", "scaffold", "spine", "cornerstone",
"linchpin", "lifeblood", "beacon", "north star", "tapestry", "the connective tissue of".
The metaphor asserts that something is important or structural without stating what
depends on what.

**Examples.**
> "Coverage of the training distribution is load-bearing for this argument."
> "The contrastive objective forms the backbone of our pipeline."
> "These assumptions are the scaffolding on which the proof rests."

**Evidence.** "Load-bearing" is the best-documented Claude-era instance: a tracked
mannerism with a dedicated scanner (github.com/orlenko/load-bearing), a TIL post
("'Load-bearing' is becoming LLM speak", mareksuppa.com), a Hacker News thread (item
48905248), and multiple 2026 how-to-stop-it posts reporting rates like 4–7 occurrences per
800-word response. The broader family is documented via Wikipedia's AI-vocabulary list,
which is full of metaphor nouns ("tapestry", "beacon", "cornerstone", "testament",
"lifeblood"), and Kobak et al.'s excess-vocabulary list (arXiv:2406.07016). The
in-house observation matches: Thomas's project bans "spine"/"backbone"/"scaffold" in living
docs and the humanize catalog bans "load-bearing" outright (patterns_general.md §31).

**Why LLMs do it.** Commentary on the "load-bearing" case: the term co-occurs with
structural metaphors across the training corpus and post-training reward favors it as a
two-syllable way to claim importance. More generally these words let the model signal
"this matters" without the harder token-level work of stating the dependency; they are
importance labels, not mechanisms.

**Replacement recipe.** Name the dependency or the mechanism directly. "X is load-bearing"
becomes "if X fails, Y no longer holds" or "the proof uses X in step 3". "The backbone of
our pipeline" becomes "the pipeline's first stage, which every later stage consumes".
Watch for the domain carve-out: "ResNet backbone", "agent scaffold(ing)" are established
technical senses in ML and must NOT be rewritten; the gate flags, the author disposes.

**Detectability.** MECHANICAL for the known word list ("load-bearing" has no legitimate
technical sense in ML paper prose and can be hard-banned; backbone/scaffold flagged with a
technical-sense exception). JUDGMENT for freshly coined metaphors, which is the general
form of the problem (see patterns_general.md §30, phantom vocabulary).

### 3. Contrastive-negation scaffolds ("negative parallelism") — MECHANICAL (high-signal forms) + JUDGMENT (density)

**Definition.** The family of antithesis templates: "not X, but Y"; "not just/only/merely
X, but (also) Y"; "It's not about X, it's about Y"; "X rather than Y" used rhetorically;
and the clipped tailing negation ("No hand-tuning. No heuristics. Just gradients.").
One instance is ordinary rhetoric; LLM text uses the scaffold as a default sentence shape.

**Examples.**
> "This is not merely a calibration issue, but a fundamental limitation of the objective."
> "The vector does not just encode style; it encodes the persona itself."
> "What matters is not the magnitude of the shift, but its direction."

**Evidence.** Wikipedia WP:AISIGNS documents three separate sub-patterns (12–14: "not just
X, but also Y"; "not X, but Y"; "X rather than Y") as AI signs. Practitioner commentary
calls "It's not X; it's Y" "the single biggest tell" of ChatGPT prose
(hardlyworking1.substack.com, with a replacement catalog). At the syntax level, the
PMC12316247 comparative study finds ChatGPT leaning on coordination and parallel
constructions where native writers subordinate, and Shaib et al. (arXiv:2407.00211) show
LLMs reuse a small set of syntactic templates at above-human rates, which is exactly what
a repeated antithesis frame is. Reinhart et al.'s phrasal-coordination finding
(arXiv:2410.16107) points the same way.

**Why LLMs do it.** The construction signposts a thesis cheaply (set up X, negate into Y)
and pattern-completion favors templates that are dense in the instruction-tuning
distribution; Shaib et al. show such templates largely originate in pretraining data and
survive RLHF.

**Replacement recipe.** State Y directly, with its evidence; cut the strawman X unless a
named reader actually believes X (then attribute it: "unlike the pointwise-LOO folds used
in [cite], we hold out entire groups"). Convert "rather than" contrasts into a positive
claim plus a reason: "we hold out groups because pointwise folds leak group identity."
Budget rule: at most one deliberate antithesis per section; zero in the abstract.

**Detectability.** MECHANICAL for the high-signal frames (`not (just|only|merely|simply)`,
`not X, but`, `it is not ... it is ...`); JUDGMENT for "rather than" (often legitimate) and
for density/placement.

### 4. Lexical tells ("AI vocabulary") — MECHANICAL

**Definition.** Individual words whose corpus frequency jumped discontinuously after
ChatGPT's release and which cluster together in AI text. Core 2023–25 list for academic
prose: delve, intricate, intricacies, pivotal, crucial, underscore(s), showcase/showcasing,
boasts, meticulous, commendable, notable, versatile, multifaceted, nuanced, realm,
landscape, tapestry, testament, leverage, foster, harness, robust, seamless, comprehensive,
garner, elucidate?, align with, enhance, highlighting, emphasizing.

**Examples.**
> "We delve into the intricate interplay between persona representations and fine-tuning dynamics."
> "These findings underscore the pivotal role of layer selection."
> "Our comprehensive evaluation showcases robust improvements across diverse benchmarks."

**Evidence.** The strongest-measured family. Kobak et al. (arXiv:2406.07016): ~900 excess
style words in PubMed abstracts, with sharp 2023–24 frequency jumps; follow-up
arXiv:2608.10715 puts excess-vocabulary signs in 89% of 2025 PMC papers. Liang et al.
(arXiv:2403.07183): fold-increases of 9.8×/11.2×/34.7× for "commendable"/"intricate"/
"meticulous" in ICLR 2024 reviews; the companion mapping paper (arXiv:2404.01268) shows the
same shifts across ~1M papers. Juzek & Ward (arXiv:2412.11385) isolate 21 focal words and
implicate post-training. Paul Graham's April 2024 "delve" post is the canonical
commentary instance (and drew the documented dialect objection: "delve" is ordinary in
Nigerian English, so the tell misfires on humans as an accusation, which again does not
matter for self-scrubbing). Wikipedia maintains the eras: the list DRIFTS as models
update, so any hard-coded list needs a revision date and periodic refresh.

**Why LLMs do it.** Juzek & Ward's model comparisons point at preference-tuning (RLHF)
rather than raw pretraining frequency; the mechanism at annotator level is unresolved.

**Replacement recipe.** Per-word substitution to plain verbs: delve into → examine;
underscore → show/support; showcase → show; pivotal/crucial → state the consequence
instead of the label; leverage → use; robust (as praise) → give the number; comprehensive
→ enumerate what was covered. General rule: if the word is doing evaluation ("meticulous",
"commendable"), replace it with the observation that would justify the evaluation.

**Detectability.** MECHANICAL: word lists with `\b` boundaries; the humanize skill already
ships two tiers (`banned_absolute.txt`, `banned_watch.txt`). Caveat: some list words have
technical senses in ML ("robust" in "distributionally robust", "dynamic" in "dynamics") so
the paper gate keeps those on watch tier, not hard-ban.

### 5. Significance inflation / undue emphasis — MECHANICAL (phrases) + JUDGMENT (framing)

**Definition.** Formulaic assertion that the subject is important, without evidence
scaling: "plays a pivotal/crucial/vital role", "stands as a testament to", "marks a
significant milestone", "cannot be overstated", "has profound implications for",
"paving the way for", "represents a paradigm shift", plus the essay-closer "broader
trends" framing.

**Examples.**
> "Understanding persona geometry plays a crucial role in AI safety."
> "This result has profound implications for the alignment community."
> "The importance of on-policy evaluation cannot be overstated."

**Evidence.** Wikipedia WP:AISIGNS pattern 1 ("undue emphasis on significance, legacy,
broader trends") with the phrase inventory; the excess-vocabulary studies capture the
lexical layer ("pivotal", "crucial" are among the 2023–24 excess words in Kobak's list;
"underscore the importance" is a Liang-flagged bigram family). Also the direct in-house
signal: reviewer-visible overclaiming is what the project's interpretation-critic exists
to catch; this is the prose-level version.

**Why LLMs do it.** Assistant post-training rewards answers that frame the topic as
important (engagement), and importance claims are unfalsifiable filler that never
contradicts the prompt.

**Replacement recipe.** Replace the importance claim with the consequence: who can now do
what, which prior result changes, what breaks without it. If no consequence can be stated,
delete the sentence; a results paragraph that ends on data needs no significance caboose.

**Detectability.** MECHANICAL for the fixed phrases; JUDGMENT for quantifying whether an
intro/discussion inflates beyond the evidence (critic's job).

### 6. Copula avoidance ("serves as", "stands as") — MECHANICAL

**Definition.** Systematic replacement of "is/are" with heavier verbs: "serves as",
"stands as", "functions as", "acts as", "represents", "constitutes", "boasts",
"features", "offers".

**Examples.**
> "The residual stream serves as the primary communication channel between layers."
> "This benchmark represents a significant step toward realistic evaluation."

**Evidence.** Wikipedia WP:AISIGNS pattern 10 ("avoidance of basic copulatives"), with
"serves as a/stands as" as the flagship frames; Reinhart et al.'s noun-heavy,
informationally-dense profile is the register this belongs to. The humanize academic
catalog carries it as "copula avoidance ... restoration".

**Replacement recipe.** Use "is" when the claim is identity or role: "the residual stream
is where layers exchange information." Keep "acts as"/"functions as" only when the point
is genuinely functional substitution (A stands in for B under condition C).

**Detectability.** MECHANICAL: `\b(serves|stands|functions|acts) as\b`, `\brepresents a\b`
on watch tier.

### 7. Present-participial trailing clauses — MECHANICAL (verb list) + JUDGMENT

**Definition.** A comma plus an "-ing" clause bolted onto a finished sentence to add an
unearned interpretive layer: ", highlighting the importance of ...", ", underscoring the
need for ...", ", showcasing its versatility", ", reflecting broader trends in ...",
", demonstrating the effectiveness of ...".

**Examples.**
> "Accuracy dropped 12 points under the persona shift, highlighting the fragility of current alignment techniques."
> "The effect replicates across all three seeds, underscoring the robustness of our findings."

**Evidence.** The single best-quantified grammar tell: Reinhart et al. (arXiv:2410.16107,
PNAS 2025) measure present participial clauses at 2–5× the human rate in instruction-tuned
LLMs; Wikipedia WP:AISIGNS pattern 3 ("superficial analyses ... often via present
participles") is the same construction observed by editors; Kobak's excess-word era
2024–25 is dominated by exactly these participles ("highlighting", "showcasing",
"emphasizing", "underscoring").

**Why LLMs do it.** It appends interpretation without a new sentence or a new subject,
which suits next-token generation; and the interpretive clause is reward-friendly filler.

**Replacement recipe.** Split the sentence. Then either (a) the interpretive claim earns
its own sentence with its own evidence ("This 12-point drop means X because Y"), or
(b) it turns out to be filler and dies. A results sentence may end at the number.

**Detectability.** MECHANICAL for the flagship verbs (`, (highlighting|underscoring|
showcasing|emphasizing|demonstrating|reflecting|illustrating|signaling|revealing|
suggesting)`); JUDGMENT for the general construction (legitimate participial clauses
exist: ", holding the seed fixed" is fine because it states a condition, not an
interpretation).

### 8. Rule of three (tricolon) — JUDGMENT

**Definition.** Triadic lists used as default rhythm rather than because there are three
things: "clear, concise, and compelling"; "we analyze, evaluate, and interpret"; abstracts
whose every enumeration has exactly three members.

**Examples.**
> "Our method is simple, scalable, and effective."
> "These vectors are interpretable, steerable, and transferable."

**Evidence.** Wikipedia WP:AISIGNS pattern 15; Shaib et al.'s syntactic-template result
(arXiv:2407.00211) provides the mechanism-level evidence for repeated micro-structures;
the slop taxonomy (arXiv:2509.19163) puts formulaic structure among the dimensions experts
associate with slop. No study isolates the triad count specifically; the tell is
editor-documented rather than corpus-measured. Say so when citing it.

**Replacement recipe.** Count the actual items. If there are two, write two; if four,
four. Delete adjectives that were added to complete the rhythm ("scalable" that was never
benchmarked). If a genuine triple survives, keep it; the ban is on the default, not the
number.

**Detectability.** JUDGMENT. A regex over `\w+, \w+, and \w+` fires constantly on
legitimate scientific enumeration; only a reader can tell a padded triad from a real one.

### 9. Hedging and metadiscourse boilerplate — MECHANICAL (phrases) + JUDGMENT (calibration)

**Definition.** Two opposite failures with one root. (a) Canned metadiscourse: "It is
important to note that", "It is worth noting that", "It should be noted that", "Note
that" as a paragraph tic, "Needless to say". (b) Hedge stacking: "may potentially
suggest", "could possibly contribute to", multiple softeners on one claim.

**Examples.**
> "It is important to note that these results are based on a single model family."
> "This may potentially suggest that persona information could be partially encoded in earlier layers."

**Evidence.** The "note that" family is on Wikipedia's sign list and in the humanize
absolute bans. Reinhart et al. add the interesting inverse: LLMs produce FEWER genuine
interactional hedges and engagement markers than human writers while producing more
formulaic metadiscourse; i.e. the tell is canned hedging, not hedging per se. Hedge
stacking as a scientific-register failure is encoded in patterns_academic.md §17/§22
(with the explicit warning not to strip calibrated uncertainty).

**Replacement recipe.** For (a): delete the frame and keep the content ("These results are
from a single model family."). If everything in the paragraph is worth noting, nothing
needs the label. For (b): one hedge per claim, chosen to match the actual uncertainty;
"may" OR "in this setting", not both plus "potentially".

**Detectability.** MECHANICAL for the fixed frames; JUDGMENT for hedge calibration
(whether the residual uncertainty statement matches the evidence is a critic call, and
over-stripping is a real risk in scientific prose).

### 10. Nominalization and noun-dense style — JUDGMENT (with a few MECHANICAL frames)

**Definition.** Verbs and adjectives packed into abstract nouns, producing dense,
agentless prose: "the utilization of", "the implementation of", "the facilitation of",
"achieves a reduction in", "the interpretability of the steerability of ...".

**Examples.**
> "The utilization of contrastive pairs enables the extraction of persona directions."
> "We observe a substantial enhancement in the alignment of model outputs."

**Evidence.** Reinhart et al. (arXiv:2410.16107): nominalizations, *that*-clauses as
subject, and phrasal coordination are among the features that most separate
instruction-tuned LLMs from humans; the composite is "informationally dense, noun-heavy"
academic register regardless of what register the prompt asked for. Agentless passives
pattern with it.

**Replacement recipe.** Give the sentence an agent and a verb: "we extract persona
directions from contrastive pairs." One transformation per sentence is usually enough to
break the pattern.

**Detectability.** Mostly JUDGMENT; a few frames gate mechanically (`the \w+ization of`,
`the utilization\b`, `enables the \w+ of`).

### 11. Essay-scaffold boilerplate: openers and closers — MECHANICAL

**Definition.** Formulaic intro framing ("In today's rapidly evolving AI landscape...",
"In recent years, X has attracted increasing attention") and conclusion templates ("In
conclusion, ...", "Overall, these findings...", "Taken together, ...", "...remains to be
seen", "...paving the way for future research", the challenges-then-future-prospects
two-beat).

**Examples.**
> "In recent years, large language models have attracted increasing attention due to their remarkable capabilities."
> "In conclusion, our findings underscore the need for further research, paving the way for more robust and interpretable systems."

**Evidence.** Wikipedia WP:AISIGNS pattern 6 (formulaic challenges/future-prospects
conclusions) and the phrase inventory around "In today's ...". Liang et al.'s abstract- and
introduction-concentrated detection signal (arXiv:2404.01268) and Holzwarth et al.'s
Discussion-section concentration (arXiv:2608.10715) both show the boilerplate lives
exactly in these open/close slots.

**Replacement recipe.** Open on the specific problem, not the field's press release: first
sentence names the gap or the object of study. Close on the strongest concrete claim plus
the sharpest named limitation; delete "remains to be seen" sentences (if it remains to be
seen, it is future work, and future work gets one concrete sentence, not a benediction).

**Detectability.** MECHANICAL for the fixed phrases; the two-beat conclusion shape is
JUDGMENT.

### 12. Vague attribution — JUDGMENT (with MECHANICAL frames)

**Definition.** Claims attributed to unnamed authorities: "researchers have shown",
"studies suggest", "it is widely recognized/acknowledged that", "experts argue",
"industry reports indicate".

**Examples.**
> "It is widely recognized that fine-tuning can compromise safety alignment."
> "Studies have shown that persona conditioning affects downstream behavior."

**Evidence.** Wikipedia WP:AISIGNS pattern 5 (vague attributions / overgeneralization
from sources); in an academic manuscript this doubles as a citation-hygiene failure, and
its hallucinated-citation sibling is heavily documented on the same page (invalid DOIs,
unresolvable references).

**Replacement recipe.** Cite or delete: "\citet{qi2024finetuning} show that ...". A claim
worth attributing has a bibkey; a claim with no bibkey is either yours (own it) or unknown
(cut it). In LaTeX this pattern is gate-adjacent: any "studies have shown"-type frame with
no `\cite` within the sentence is a flag.

**Detectability.** Frames are MECHANICAL; whether the eventual citation actually supports
the claim is JUDGMENT (and belongs to citation-verification, not this skill).

### 13. Syntactic templating and uniform rhythm — JUDGMENT

**Definition.** The same sentence architecture repeated across a paragraph or section:
equal-length sentences, repeated POS sequences, every paragraph opening with the same
move (topic sentence + elaboration + participial interpretation), every paragraph the
same length.

**Evidence.** Shaib et al. (arXiv:2407.00211): models emit repeated POS templates at
above-human rates, model-specifically, and the templates originate in pretraining and
survive RLHF. Reinhart et al. document the reduced stylistic variation at the feature
level ("LLMs struggle to match human stylistic variation"). Wikipedia lists "syntax
variety" as a HUMAN indicator, the mirror image. Detector folklore ("burstiness") is the
commercial version of the same observation.

**Replacement recipe.** Read the paragraph aloud; if three consecutive sentences share a
skeleton, vary one: fold two into a subordinate construction, shorten one to under eight
words, or move a condition to the front. Reinhart's subordination finding gives the
direction: human academic prose subordinates more and coordinates less.

**Detectability.** JUDGMENT (a POS tagger could gate it, but not with grep; leave it to
the critic).

### 14. Formatting tells in paper sources — MECHANICAL

**Definition.** Chat-register formatting leaking into `.tex`: bold run-in headers inside
prose paragraphs, bullet lists where prose is expected, Title Case section headings
(non-venue-style), curly/smart quotes pasted from a chat window, emoji, thematic-break
separators, markdown syntax (`**`, `##`) surviving in LaTeX.

**Evidence.** Wikipedia WP:AISIGNS patterns 16–27 (title case, boldface overuse,
inline-header vertical lists, emoji, curly quotes, markdown-in-wikitext); the direct
LaTeX analog of the markdown-leak pattern is `**` or `##` in a `.tex` file.

**Replacement recipe.** Sentence-case headings per venue style; convert bullet stacks in
Results/Discussion to prose; straight quotes / proper LaTeX quoting (``` `` '' ```);
delete emoji unconditionally.

**Detectability.** MECHANICAL: `[""'']`, `\*\*`, `^##`, emoji ranges, `\\textbf` density
per section (the last as a count, not a ban).

### 15. Elegant variation (synonym cycling) — JUDGMENT, historical

**Definition.** Cycling synonyms to avoid repeating a term: "the model ... the system ...
the network ... the architecture" for one referent.

**Evidence.** Wikipedia lists it under historical/outdated indicators (older-model
signature, now less prevalent). In scientific prose it is independently harmful
(terminological drift breaks precision) regardless of provenance, and the project's
glossing rules already require one term per concept.

**Replacement recipe.** One name per object, chosen at first mention, repeated verbatim.
Scientific prose repeats; only feature writing rotates.

**Detectability.** JUDGMENT.

---

## What LLMs UNDER-produce (for the critic's positive checklist)

From Reinhart et al. and Wikipedia's human-indicator list, the critic can also check for
the presence of human markers, not just the absence of tells:

- Syntax variety (sentence-length spread; subordination, not just coordination).
- Genuine, calibrated hedges tied to specific uncertainty, not frame phrases.
- Agentive sentences ("we did X to test Y") over agentless passives.
- Willingness to end a sentence on a number without an interpretive participle.
- Specifics over category labels: named quantities, named failure modes, named readers of
  a contrast ("unlike [cite] ...").

## Detectability summary

| # | Family | Class |
|---|---|---|
| 1 | Em dash | MECHANICAL (hard ban) |
| 2 | Metaphor jargon | MECHANICAL list + JUDGMENT for coinages |
| 3 | Contrastive negation | MECHANICAL high-signal frames + JUDGMENT |
| 4 | AI vocabulary | MECHANICAL (two tiers) |
| 5 | Significance inflation | MECHANICAL phrases + JUDGMENT |
| 6 | Copula avoidance | MECHANICAL |
| 7 | Participial trailers | MECHANICAL verb list + JUDGMENT |
| 8 | Rule of three | JUDGMENT |
| 9 | Hedging boilerplate | MECHANICAL frames + JUDGMENT calibration |
| 10 | Nominalization density | JUDGMENT + few MECHANICAL frames |
| 11 | Opener/closer boilerplate | MECHANICAL phrases |
| 12 | Vague attribution | MECHANICAL frames + JUDGMENT |
| 13 | Syntactic templating | JUDGMENT |
| 14 | Formatting leaks | MECHANICAL |
| 15 | Elegant variation | JUDGMENT (historical) |

## Maintenance note

The lexical tier drifts with model generations (Wikipedia's era split; arXiv:2605.25358).
Any hard-coded ban list needs a dated header and an occasional refresh against
github.com/berenslab/llm-excess-vocab and the current WP:AISIGNS revision. The
grammatical/rhetorical tiers (participial trailers, antithesis scaffolds, templating) have
been stable across model generations so far and are the safer long-term investment.

## Source list

- Wikipedia: *Signs of AI writing* (WP:AISIGNS), https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
- Kobak, González-Márquez, Horvát, Lause. arXiv:2406.07016; Science Advances 11, eadt3813 (2025). Word list: https://github.com/berenslab/llm-excess-vocab
- Holzwarth, González-Márquez, Kobak. arXiv:2608.10715 (2026).
- Liang et al. arXiv:2403.07183 (ICML 2024); Nature news d41586-024-01051-2.
- Liang et al. arXiv:2404.01268 (2024).
- Juzek & Ward. arXiv:2412.11385 (COLING 2025).
- Reinhart et al. arXiv:2410.16107; PNAS 122, e2422455122 (2025). Notebook: https://www.refsmmat.com/notebooks/llm-style.html
- Shaib, Elazar, Li, Wallace. arXiv:2407.00211 (EMNLP 2024).
- Shaib, Chakrabarty, Garcia-Olano, Wallace. arXiv:2509.19163 (2025).
- *AI-Associated Lexical Shifts Across 34 Languages*. arXiv:2605.25358 (2026). (Cited from search results; abstract not independently fetched.)
- Comparative syntactic complexity, ChatGPT vs native speakers. PMC12316247 (2025).
- Paul Graham: X post on "delve" (April 2024); *Writes and Write-Nots* (Oct 2024), https://paulgraham.com/writes.html
- TechCrunch, "OpenAI says it's fixed ChatGPT's em-dash problem" (2025-11-14); Tom's Guide em-dash toggle coverage.
- hardlyworking1.substack.com, "Some alternatives to 'It's not X; it's Y'".
- "load-bearing" documentation: https://mareksuppa.com/til/load-bearing/ ; https://github.com/orlenko/load-bearing ; HN item 48905248.
