# Figure, caption, and in-text reference standard

**Status: ADOPTED 2026-09-03 (decisions D1 to D3 and the palette question answered by Thomas, see §0); migration in progress, CoT figures deferred on Thomas's instruction.** Audit of the Overleaf paper at commit `c314eda`
(17 figure floats, 5 tables). Three decisions need Thomas (§0). Once he decides,
the normative parts (§2 to §5) move into `plotting_style.md` (the source of truth
named in `CLAUDE.md`) and the Overleaf clone `CLAUDE.md` § Figures; §1 stays here
as the dated audit record. Nothing in the paper or the figure scripts has been
changed yet.

## 0. Decisions needed

| # | Question | Recommended | Alternative |
|---|---|---|---|
| D1 | What goes in the in-figure panel title? **Decided: descriptive only.** | **Descriptive only** (what is plotted: "Predictability across layers"). The figure-level claim lives in the caption's bold lead; each panel's subclaim lives in the caption's per-panel beat and in the text's bold header, same words (§3, §4.3). Matches Figure 3 as approved, the 2026-08-12 "simple and concise" directive, and `paper-plots` §3.8. | Subclaims as panel titles inside the figure (blog register). Then caption beat 2 and the text header must repeat the title verbatim: one claim, one wording, three copies. |
| D2 | One name for the retrieval metric | **Decided: `top-1 retrieval` everywhere** (axes and legends `Top-1 retrieval`, captions and text `top-1 retrieval`); Methodology defines strict top-$k$ retrieval accuracy once. `acc@1` and `answer retrieval` are retired. The CoT figures and their captions still carry `acc@1` until their planned redo. | `acc@1` everywhere (the recommendation Thomas overruled). |
| D3 | Where the figure pointer sits in a claim paragraph. **Decided: in the bold header.** | **In the bold header, before the colon**: `\textbf{Claim} (\figref[A]{fig:x})\textbf{:}`. Already the pattern in §4.3 and §4.4. A skimming reader maps claim to panel in one glance. | At the end of the paragraph, after the evidence sentence. |

Font is not a decision unless Thomas objects: keep Inter (11 of 17 figures already
use it, and Figure 3 was approved in it). The body text is Times; sans figures on a
serif body is the common ICLR/NeurIPS pairing.

## 1. Audit: what the paper does today

### 1.1 Figures: four visual systems, type from 5.4 pt to 9.5 pt

Realized size = script font size × (include width ÷ PDF canvas width). ICLR text
width is 5.5 in.

| Fig | Label | File | System | Font | Body / tick pt | In-figure title | Caption words | Panel marker in caption |
|---|---|---|---|---|---|---|---|---|
| 1 | schematic | `fig1_schematic` | TikZ | Computer Modern | n/a | descriptive `(a)`/`(b)` | 51 | none |
| 2 | useful-directions | `c3_persona_direction_spectrum` | paper_plots neurips | Times + STIX | 7.3 / 6.5 | none | 61 | single panel |
| 3 | main-accuracy | `figure2_predictability_scaling` | c2a-v1 | Inter | 6.9 / 6.5 | descriptive | 41 | `\textbf{(A)}` |
| 4 | qualitative-discrimination | `c3_qualitative_discrimination` | c2a-v1 | Inter | 9.0 / 8.5 | n/a (text panels) | 69 | left/right |
| 5 | sae-tier-gradient | `c3_sae_tier_gradient` | c2a-v1 | Inter | 6.1 / 5.8 | question + claim | 80 | `A:` |
| 6 | pair-shifts | `c3_pair_shifts` | c2a-v1 | Inter | 6.4 / 6.1 | claim | 81 | single |
| 7 | posttraining | `c1_posttraining_dynamics` | bespoke copy of c2a | DejaVu Sans | 6.7 / 6.0 | claim | 133 | `A:` |
| 8 | speaker-maps | `c4_shared_speakers` | paper_plots neurips | Times + STIX | 9.0 / 8.0 | descriptive `A.` | 85 | `\textbf{(A) claim}` |
| 9 | offpolicy | `c1_offpolicy_origin` | c2a-v1 | Inter | 6.9 / 6.5 | claim | 186 | `A:` |
| 10 | cot | `c1_cot_maps` | c2a-v1 | Inter | 6.5 / 6.2 | claim | 201 | `\textbf{(A)}` |
| 11 | cot-strata | `c1_cot_strata` | c2a-v1 | Inter | 8.2 / 7.8 | claim | 59 | single |
| 12 | cot-ladder | `c1_cot_ladder` | c2a-v1 | Inter | 5.7 / 5.4 | claim | 127 | single |
| 13 | behavior-prediction | `c5_pv_methods_regimes` | paper_plots neurips | Times + STIX | 9.5 / 8.5 | descriptive | 168 | none (3 panels) |
| 14 | refusal-by-class (app.) | `c3_refusal_swaps_by_class` | c2a-v1 | Inter | 5.6 / 5.3 | n/a | 100 | `A:` |
| 15 | cot-necessity (app.) | `c1_cot_necessity` | c2a-v1 | Inter | 6.0 / 5.7 | n/a | 137 | `\textbf{(A, B)}` |
| 16 | cot-necessity-r2 (app.) | `c1_cot_necessity_r2` | c2a-v1 | Inter | 8.1 / 7.6 | n/a | 92 | single |
| 17 | behavior-prediction-forest (app.) | `c5_claim4_margin_forest` | paper_plots neurips | Times + STIX | 7.5 / 6.7 | n/a | 90 | single |

Findings:

- **Four visual systems.** 11 figures are `c2a-v1` (Inter, teal/terracotta,
  uppercase kicker legends). 4 are `paper_plots` "neurips" (Times, Wong palette,
  framed legends inside the axes). Figure 7 is a bespoke script that copied c2a
  constants and drifted (`#25292D` vs `#22272B` ink, `#16708A` vs `#176B87` teal,
  DejaVu instead of Inter). Figure 1 is TikZ in Computer Modern.
- **Type size varies 1.8×.** The c2a scripts author at 8.7 to 16.4 in and include
  at 0.62 to 1.0 textwidth, so the same 18 pt font renders anywhere from 5.6 pt
  (Fig 14) to 9.0 pt (Fig 4). Six figures sit below the 7 pt legibility floor.
- **Same concept, different encoding.** Base vs post-trained model is
  orange/blue in Fig 8 and teal/red in Fig 7. R² vs acc@1 is solid/hatched in
  Fig 10 but two hues in Fig 7. Purple means "needs-reasoning corpora" in §4.5 and
  "plain Claude answers" in §4.4.
- **Two wordings of one claim.** Figures 6, 7, 9, and 11 carry a claim title
  inside the plot AND a differently worded claim in the caption lead (Fig 11:
  "barely changes" vs "changes by little").
- **Math falls back to DejaVu** inside every Inter figure (`$R^2$`), visible as a
  second typeface.
- **File name lies.** `figure2_predictability_scaling.pdf` is Figure 3.
- **Figure 8 has no in-repo producer script** by name; the sidecar traces it to
  `scripts/issue2054_paper_r2_figs.py`. Every other figure's script is greppable
  by file stem.

### 1.2 Captions: right idea, five surface forms

- **Lead sentence.** 15 of 17 figure captions and 3 of 5 table captions open with
  a bold claim. Exceptions: Fig 8 puts a bold claim per panel; `tab:sae-properties`
  and the 25-failures table have no lead.
- **Panel markers** take three forms: `\textbf{(A)}` (3 captions), `A:` (4),
  `\textbf{(A) claim…}` (1).
- **Length** 41 to 201 words, median 85. Six exceed 120 (Figs 7, 9, 10, 12, 13,
  15). The long ones carry methodology (fold counts, pool sizes, per-corpus
  n) that belongs in the appendix or the Methodology section.
- **Uncertainty vocabulary:** "whiskers" (9 captions), "error bars" (1), "bands"
  (2, shaded regions).
- **Term drift:** "context-to-answer map" (13) beside "context--answer map / pair"
  (10). "acc@1" in captions and text beside "Top-1 retrieval" on Figs 3 and 9
  and "Answer retrieval (acc@1)" on Fig 10.
- ICLR template rule (fetched from the 2026 `iclr2026_conference.tex`): caption
  below the figure, sentence case; table title above the table. The paper
  complies.

### 1.3 In-text references: consistent word, four placements

- Word form is already uniform: 51× `Figure~\ref{}`, 8× `Table~\ref{}`, zero
  `Fig.`, zero `\Cref`/`\autoref`. Panel form is uniform: `Figure~7A`, `7A--B`.
- Placement follows four patterns:
  1. Signpost sentence: "The results are shown in Figure N." (7×) or "Results are
     in Figure N." (2×), then bold claims with no per-claim pointer (§4.1, §4.2).
  2. Pointer in the bold header before the colon:
     `\textbf{Claim} (Figure~7A)\textbf{:}` (5×, §4.3 and §4.4).
  3. Pointer mid-sentence in the evidence: `(0.66; Figure~6)` (about 15×).
  4. Figure as sentence subject: "Figure 4 shows three such pairs" (5×).
- "The results are shown in Figure N." seven times is a templating tell under the
  `/writing-tells` judgment tier.
- Figures 2, 11, and 14 are cited exactly once; Fig 14 only from the appendix.
- Typo the lint would catch: `insignificant(Figure~\ref{fig:cot-necessity-r2})`
  (missing space, §4.5).

## 2. Figure standard: `c2a-v2`

### 2.1 One module, zero local constants

Every paper figure script imports `c2a_plot_style` and nothing else for style.
No hex literal, rcParam, or font name outside the module. The bespoke Figure 7
script and the four `paper_plots` scripts migrate (§6).

### 2.2 Fixed scale, three widths

Add to the module:

```python
C2A_SCALE = 0.42                       # realized pt = script pt × C2A_SCALE
TEXT_WIDTH_IN = 5.5                    # ICLR
INCLUDE_WIDTHS = {"full": 1.0, "wide": 0.75, "half": 0.5}

def c2a_figure(width: str, aspect: float) -> tuple[plt.Figure, float]:
    """Canvas whose width realizes C2A_SCALE at the given include width."""
    frac = INCLUDE_WIDTHS[width]
    w_in = frac * TEXT_WIDTH_IN / C2A_SCALE     # full = 13.1 in
    return plt.figure(figsize=(w_in, w_in * aspect)), frac
```

Realized type at `C2A_SCALE = 0.42` with the current rcParams (18/17/20/22 pt):
body 7.6, ticks 7.1, axis labels 8.4, panel titles 9.2 pt. Every figure lands at
these four sizes. Figure 3 grows from 6.9 to 7.6 pt; Figure 12 grows from 5.7.
The sidecar records `include_width_frac` and the exact
`\includegraphics[width=0.75\textwidth]{…}` line; the lint (§5) checks the tex
matches.

### 2.3 Fonts

Inter for all text AND math: set `mathtext.fontset = "custom"` with
`mathtext.rm/it/bf = Inter` so `$R^2$` stops falling back to DejaVu. The TikZ
schematic loads `\usepackage[type1]{inter}` (CTAN `inter`, pdfLaTeX-compatible;
verify on first compile) and sets `\sffamily`.

### 2.4 Semantic palette: one color, one meaning, paper-wide

| Role | Hex | Marker | Used for |
|---|---|---|---|
| `linear` | `#176B87` teal | ● | the linear map, the paper's main object |
| `nonlinear` | `#C4553D` terracotta | ◆ | MLP predictor |
| `control` | `#687078` gray | ✕ | shuffled, identity+bias, random-direction, any null |
| `base_model` | `#C98A1B` amber | ■ | pretrained / base checkpoint |
| `post_trained` | `#176B87` teal | ● | SFT / DPO / RLVR / instruct (same hue as `linear`: the map on the model we study) |
| `other_source` | `#7B3294` purple | ▲ | answers from another model or persona (Fig 9) |
| `needs_reasoning` | `#7B3294` purple | ■ | needs-reasoning corpora (Fig 10 to 12) |
| `no_reasoning` | `#5AAE61` green | ● | no-reasoning corpora |

Purple appears twice in that table, once per section. Decide one or the other at
migration: recommended `other_source` moves to amber `#C98A1B` (Fig 9 currently
uses amber for the eccentric-style answers; both Claude sources become amber
shades, filled vs open) so purple means reasoning demand everywhere. Encode the
metric by fill, never by hue: R² = solid line / filled marker / solid bar;
acc@1 = dashed line / open marker / hatched bar. Color plus shape on every series.

### 2.5 Panel furniture

- **One lettered panel per question** (Thomas, 2026-09-03). A model, corpus, or
  checkpoint variant of the same question is a series inside the panel (color +
  marker, or grouped bars), never a sibling panel. Sibling panels that repeat one
  question are merged. When the series count would pass about six, the panel
  becomes an unlettered facet strip with shared axes and factor-level labels
  ("Evil", "Sycophancy", "Hallucination", the Figure 13 form); facets carry no
  letter and no subclaim of their own.
- Panel kicker: `A · UPPERCASE CONTEXT` in `MUTED`, left-aligned above the axes
  (the c2a-v1 form). Letters A, B, C, D, never (a)/(b), never `A.`.
- Panel title per D1. If descriptive: sentence case, states the plotted quantity
  and grouping ("Retention of the previous stage's map"). No verdict words
  (barely, better, repairs, degrades).
- Legends frameless. Multi-panel: one kicker legend row above the panels, split by
  semantic role (`PREDICTOR`, `METRIC`). Single panel: frameless inside the axes.
- Top and right spines off; horizontal grid only; white background; upward arrow
  on the axis label when larger is better; no text blocks, arrows, or effect
  labels on the canvas (standing directive 2026-08-12).
- Axis label vocabulary equals caption vocabulary: `Held-out $R^2$`, `Top-1 retrieval` (D2).

### 2.6 Export and naming

- One call: `save_c2a_figure(fig, stem, include_width=frac, …)` writes PDF
  (Type 42 fonts, no timestamp), 240 dpi PNG, grayscale PNG, and the sidecar with
  plotted values, input hashes, git state, `include_width_frac`, the LaTeX
  include line, and every rendered string (`text`).
- File `figures/paper/c<k>_<slug>.pdf`; label `fig:<slug>` with the same slug;
  producer `scripts/paper_fig_<slug>.py`. Rename
  `figure2_predictability_scaling` to `c1_predictability_scaling` and
  `fig:main-accuracy` to `fig:predictability-scaling`. One grep finds file,
  label, and script.
- Both PNG and grayscale are checked before the PDF is copied into the Overleaf
  clone.

## 3. Caption standard

Four beats, fixed order, then stop.

1. **Lead** (bold). One sentence, the claim this figure supports, at most 15
   words, sentence case, ends with a period. Same wording as the bold claim header
   in the text that cites the figure (§4.5); when the figure serves several
   headers, the lead is the section's claim.
2. **Per-panel subclaim, then what is plotted.** Each lettered panel opens with
   `\panel{A}` (renders `**(A)**`), then its subclaim in bold (8 words or fewer,
   sentence case, ends with a period), then y against x, the grouping, and any
   encoding the on-figure legend does not already carry. Present tense, no
   verdicts in the description. Because every lettered panel answers its own
   question (§2.5), every lettered panel has exactly one subclaim; a single-panel
   figure has none beyond the lead. The subclaim appears in exactly two places
   with identical wording: here and the bold claim header in the text that cites
   the panel (§4.3). Never on the canvas. The subclaim is the text header's wording
   verbatim; Thomas writes the claims, so where a header runs past 8 words the header
   wins and the cap is advisory.
3. **Uncertainty and sample.** One sentence: estimator, interval type, n.
   "Error bars: 95% bootstrap intervals over 1,000 prompt-level draws; 13,116
   questions." Use "error bars" for line intervals and "bands" for shaded ones;
   retire "whiskers".
4. **Setup footer.** Fixed order: model; layer; data and n; folds. Then, if
   needed, one pointer: "Details: Appendix~\ref{app:x}."

Caps: main text 90 words (soft) and 120 (hard) for one or two panels, 150 (hard)
for three or more; appendix adds 30. Over the cap means beat 2 or 4 is carrying
methodology; move it to the appendix and point.

Not in a caption: interpretation beyond the lead ("This aligns with…"), prior-work
citations, more than two numbers, any term absent from the figure or the
Methodology, "left/right/top" when panel letters exist.

Terminology locked for captions and axes: `context-to-answer map` for the map (a
`context--answer pair`, meaning one context with its answer, is a different noun
and stays), `top-1 retrieval` (D2; never `acc@1`, never `answer retrieval`),
`Held-out $R^2$`, `error bars`.

Tables: title above the table (ICLR), same four beats with beat 2 naming the
columns.

### Template

```latex
\caption{\textbf{<Claim, ≤15 words.>}
  \panel{A} <y> against <x> for <grouping>; <encoding not in legend>.
  \panel{B} <…>.
  Error bars: <interval> over <draws>; <n>.
  <Model>, layer <ℓ>, <n> <units>, <k> IID folds. Details: \appref{app:x}.}
```

### Before and after: Figure 10 (`fig:cot`), 201 words to 88

Before (abridged): *The context state retrieves the specific answer before any
reasoning token; chain of thought adds a few points and leaves predictability
unchanged. Solid bars and filled markers: held-out R². Hatched bars and open
markers: acc@1 of the question's own answer among the held-out fold's answers,
under whitened cosine with CSLS as in Section 4.1 (chance 0.04%). All panels use
the needs-reasoning corpora (MATH, multi-step GSM8K, ContextHub levels 3–4;
13,116 rows for OpenThinker3-7B, 14,221 for Qwen3-8B); the no-reasoning corpora
are compared in Figure 11. (A) Maps to the answer vector … (D) … All maps are
ridge fits at one frozen layer with five random-row folds; error bars are 95%
intervals over 1,000 paired prompt-level bootstrap draws. The identity-plus-bias
baseline scores below zero R² for every map shown.*

After:

```latex
\caption{\textbf{The context state retrieves the answer before any reasoning token.}
  \panel{A} Held-out $R^2$ (solid) and top-1 retrieval (hatched) of maps into the answer
  vector from the last context token and from the end-of-thought token.
  \panel{B} The same two metrics from each relative position inside the thinking
  span. \panel{C} Panels A and B on Qwen3-8B, plus the map with thinking disabled.
  \panel{D} Maps within and across the reasoning-SFT step from Qwen2.5-7B-Instruct
  to OpenThinker3-7B. Error bars: 95\% intervals over 1,000 paired prompt-level
  bootstrap draws. Needs-reasoning corpora (13,116 questions for OpenThinker3-7B,
  14,221 for Qwen3-8B), layer 19, five IID folds. Details: \appref{app:cot}.}
```

The dropped material (CSLS recipe, chance level, corpus list, identity+bias
score) already lives in Methodology or moves to the appendix subsection. Under the
one-panel-per-question rule (§2.5) this figure also loses a panel: the Qwen3-8B
context and end-of-thought bars of panel C move into panel A as a second series,
and C keeps only the thinking-on versus thinking-off comparison. Each remaining
panel then opens with its subclaim, for example `\panel{A} \textbf{The end-of-thought
state closes most of the remaining gap.}`.

### Worked three-panel example: Figure 7 (`fig:posttraining`), 133 words to 95

```latex
\caption{\textbf{The map is inherited from pretraining; SFT changes it and later stages preserve it.}
  \panel{A} \textbf{The map is present at every stage.} Held-out $R^2$ (solid) and
  acc@1 (hatched) of each stage's own map.
  \panel{B} \textbf{Earlier-stage context states predict later-stage answers.}
  $R^2$ into each stage's answers from Base, previous-stage, and own context states.
  \panel{C} \textbf{SFT changes the map; DPO and RLVR preserve it.} Retention of
  the preceding stage's map on the next stage's pairs, as is, with a refit bias,
  and with a refit scale and bias.
  Error bars: 95\% bootstrap intervals over six IID folds. OLMo-2-7B, layer 31,
  16,391 LMSYS contexts. Details: \appref{app:posttraining}.}
```

The three text headers of §4.3 then read `\textbf{The map is present at every
stage} (\figref[A]{fig:posttraining})\textbf{:}` and so on, word for word.

## 4. In-text reference standard

### 4.1 Macros, in the shared header of `main.tex` and `draft.tex`

```latex
\newcommand{\figref}[2][]{Figure~\ref{#2}#1}      % \figref{fig:x} → Figure 7; \figref[A]{fig:x} → Figure 7A
\newcommand{\tabref}[1]{Table~\ref{#1}}
\newcommand{\appref}[1]{Appendix~\ref{#1}}
\newcommand{\panel}[1]{\textbf{(#1)}}              % captions only
```

Raw `Figure~\ref`, `Fig.`, `figure~\ref` are retired; the lint flags them. Panel
ranges: `\figref[A--B]{fig:x}`. No `cleveref`: the macros are two lines, greppable,
and cannot fight `hyperref` load order.

### 4.2 One placement pattern

The results sections already run setup → bold claim paragraphs. Fix the pointer
slots:

- **Setup paragraph** names what is measured and ends with the figure in
  parentheses: "…and both metrics as the training set grows (\figref{fig:x})."
  Drop the standalone "The results are shown in Figure N." sentence.
- **Claim header** (D3): `\textbf{Claim} (\figref[A]{fig:x})\textbf{:}` whenever
  the figure has panels or the block draws on more than one figure. A block whose
  claims all read one single-panel figure points once, in the setup, and the
  headers carry nothing.
- **Evidence sentences** may add a second pointer for a specific number only when
  it comes from a different panel or figure than the header names.
- The figure is never the grammatical subject of a claim ("Figure 7 shows that
  the map…"). The claim is the subject; the figure is the evidence in
  parentheses. Allowed as subject only in the setup sentence and in the appendix's
  "Figure 15 scores…" walk-throughs.

### 4.3 Coverage rules

- Every figure is cited from at least one bold claim header in the main text
  (appendix figures: from at least one main-text sentence via
  `(\appref{app:x}, \figref{fig:y})`).
- Every figure appears after its first citation in the source order (floats are
  `[t]`; LaTeX places them on the same or next page).
- Caption lead wording equals the claim-header wording that cites it, and each
  panel's caption subclaim equals the header that cites that panel. Ctrl-F on the
  claim finds both.
- One header per lettered panel; a header cites a range (`\figref[A--B]`) only
  when one claim genuinely rests on two different questions' panels, which the
  merge rule (§2.5) should make rare.
- Panel letters only: never "left panel", "right", "top row" when letters exist.

### Before and after: §4.1 setup and first claim

Before:

```latex
… then measure both $R^2$ and acc@1 as the training dataset grows. The results
are shown in Figure~\ref{fig:main-accuracy}. We find that:

\noindent \textbf{Even a simple linear map is highly predictive:} At 1 million …
```

After:

```latex
… then measure both $R^2$ and acc@1 as the training set grows
(\figref{fig:predictability-scaling}). We find that:

\noindent \textbf{Even a simple linear map is highly predictive}
(\figref[B]{fig:predictability-scaling})\textbf{:} At 1 million …
```

## 5. Enforcement

`check_paper_figures.sh <clone root>`, run by the Overleaf clone pre-commit hook
next to `check_paper_tells.sh`, and mirrored as a `workflow_lint.py` check in EPS:

| # | Rule | Severity |
|---|---|---|
| F1 | Every `\caption{` opens with `\textbf{` and the bold span ends `.}` | FAIL |
| F2 | Caption word count ≤ 120 main / ≤ 150 appendix | FAIL |
| F3 | Every `fig:`/`tab:` label is referenced ≥ 1 outside float bodies | FAIL |
| F4 | No `Figure~\ref`, `Fig.`, `figure~\ref`, `\Cref`, `\autoref` outside the macros | FAIL |
| F5 | Panel refs match `\figref\[[A-D](--[A-D])?\]` | FAIL |
| F6 | `\includegraphics` width equals the sidecar's `include_width_frac` | FAIL |
| F7 | Caption and axes use `top-1 retrieval`, `context-to-answer map`, `error bars` (no `acc@1`, `answer retrieval`, `whiskers`, `context--answer map`) | FAIL |
| F8 | Lead sentence ≤ 15 words | WARN |
| F9 | No letter glued to `(` before a `\figref` (the `insignificant(Figure` shape) | WARN |
| F10 | Every PDF under `figures/paper/` embeds only Inter (`pdffonts`) | WARN |

The EPS side: `save_c2a_figure` refuses a figure whose rcParams differ from the
module's, and `verify_task_body.py` checks 24/28/34 already scan the sidecar
`text` block for slugs.

## 6. Migration plan and cost

All plot-only from checked-in JSON; no GPU, no model calls. Agent time about half
a day. Captions and prose are Thomas's co-edited text: after D1 to D3, each
rewrite lands as a targeted edit against a fresh pull, one float per commit.

| Step | Files | Effort |
|---|---|---|
| 1. Module: `C2A_SCALE`, `c2a_figure`, palette roles, mathtext, sidecar fields | `c2a_plot_style.py` | 1 h |
| 2. Re-canvas 11 c2a figures (figsize + title per D1 + top-1 retrieval label) | 7 scripts: `make_paper_figure2`, `make_paper_section42_figures`, `section45_cot_{figure,ladder,strata,necessity}_figure`, `section4_offpolicy_figure` | 2 h |
| 2b. Merge same-question panels: Fig 10 (Qwen3-8B bars of C into A; C keeps thinking on/off; add the Qwen3-8B trajectory to B if banked), Fig 15 (A and B into one scatter, model as marker, as its panel C already does) | `section45_cot_figure`, `section45_cot_necessity_figure` | 1 h |
| 3. Port Fig 7 to the module (drop copied constants) | `section43_posttraining_figure.py` | 30 min |
| 4. Port Figs 2, 8, 13, 17 from `paper_plots` to the module | `issue779_plot3_redesign`, `issue2054_paper_r2_figs`, `issue1739_result2_fourpanel_fig`, `issue1739_claim4_relabel_figs` | 2 h |
| 5. Schematic font | `fig1_schematic.tex` | 15 min |
| 6. Rename `figure2_predictability_scaling` → `c1_predictability_scaling` (file, label, `plotting_style.md`) | 3 places | 15 min |
| 7. Macros in both headers; `\figref` sweep (51 sites); claim-header pointers (§4.2) | `main.tex`, `draft.tex`, `sections/*.tex` | 1 h |
| 8. Caption rewrites, 17 figures + 5 tables, four-beat form | `sections/*.tex`, `tables/*.tex` | 2 h, Thomas reviews leads |
| 9. Lint F1 to F10 + pre-commit wiring | `~/.claude/skills/writing-tells/`, clone hook | 1 h |
| 10. Fold §2 to §5 into `plotting_style.md` and clone `CLAUDE.md` § Figures | 2 docs | 30 min |

Visual check after steps 2 to 5: contact sheet of all 17 PDFs at manuscript
scale plus the grayscale audits, before any copy into the Overleaf clone.

## Sources consulted

- Paper sections and captions: `~/overleaf-6a59c927/sections/*.tex` at `c314eda`.
- Style module and doc: `c2a_plot_style.py`, `plotting_style.md`.
- Project rules: `.claude/skills/paper-plots/{SKILL,style-reference}.md` (§3.8
  no interpretive titles, § Captions estimator/interval/n), CLAUDE.md standing
  directives (2026-08-12 simple figures; 2026-08-21 rename coinages; 2026-08-25
  one label map).
- ICLR 2026 template text (`iclr2026_conference.tex`, fetched 2026-09-03).
- `ml-paper-writing/academic-plotting` style guide: 7 pt floor, no title inside
  the figure, self-contained captions, consistent fonts across all figures.
