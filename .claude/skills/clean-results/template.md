# Clean-Result Issue Body — Template

Fill in every `{{PLACEHOLDER}}`. Do not leave any. If a section doesn't
apply, write "N/A" and one sentence why.

Title format: `{{CLAIM_SUMMARY}} ({{HIGH|MODERATE|LOW}} confidence)`

The title MUST (a) summarize the findings / claim (not the experiment
name) and (b) end with an overall-confidence marker `(HIGH confidence)`,
`(MODERATE confidence)`, or `(LOW confidence)`. The marker must match the
confidence line inside `### Results`. The `clean-results` label (not any
title prefix) is the canonical signal that the issue is a clean result —
do NOT prefix the title with `[Clean Result]` or similar.

Example titles (good):
- `Weak evidence that evil-persona capability coupling reduces post-EM capability (LOW confidence)`
- `Tulu midtraining preserves capability but not alignment under EM (MODERATE confidence)`
- `Contrastive design is the sole determinant of leakage containment (HIGH confidence)`

Example titles (bad):
- `Results for Experiment A3b` ← what does it SHOW? no confidence.
- `Leakage analysis` ← what's the CLAIM? no confidence.
- `Tulu midtraining preserves capability but not alignment under EM` ← claim present but confidence missing.
- `[Clean Result] Tulu midtraining preserves capability but not alignment under EM (MODERATE confidence)` ← drop the `[Clean Result]` prefix; the label carries that signal.

**Reference exemplar:** issue **#75** (`Weak evidence that evil-persona
capability coupling reduces post-EM capability (LOW confidence)`) — match
this shape for every new clean result.

**Multi-claim issues are fine.** A single clean-result issue can carry
multiple related claims in its AI TL;DR / AI Summary / Sample outputs /
Headline numbers — if the claims travel together (same methodology,
same eval rig, same broader question), they belong in one issue. The
title summarizes the issue's primary contribution; the AI TL;DR
bullets cover the claims; the AI Summary Results section holds the
hero figure(s) and Main takeaways. There is no parent / child / sub-issue concept in this workflow.

**Multi-issue narrative consolidation** (rare — for consolidating
across previously-separate experiment threads). Pick the PRIMARY
source issue and edit its body to include the OPTIONAL
`Source-issues:` and `Supersedes:` lines at the very top of the body,
immediately after the title (as the first content under `## Human
TL;DR`, BEFORE the placeholder line). Single-experiment clean-results
SHOULD NOT include these lines. Reference exemplar: **#237**
(prose-only `Source issues:` and `supersedes:` references between
findings).

```markdown
## Human TL;DR
Source-issues: #N1, #N2, #N3   <!-- consolidation ONLY -->
Supersedes: #M1, #M2

{{user fills this in later}}
```

---

## Human TL;DR

<!-- AUTHOR NOTE: This section is for the USER to fill in by hand. The
analyzer / clean-results skill MUST leave the placeholder line below
unchanged in drafts — it's the user's voice, written after the AI
sections are done and the user has actually digested the result. It
is the only section in the body that is allowed to be empty /
unchanged in a draft; the verifier permits the literal placeholder
text and skips content checks. Do NOT pre-fill this section with a
claims list, navigation index, or any other meta-structure — the
AI TL;DR + AI Summary already carry the issue's claim(s); the human
TL;DR is exclusively for the user's own narrative voice. -->

_(Human TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_

---

## AI TL;DR

<!-- AUTHOR NOTE: 3-5 unlabeled bullets summarizing the post's claims,
in the LessWrong research-post tradition. Bullets are NOT structurally
labeled (no `**Setup.**` / `**Headline.**` prefixes — those aren't a
real LW convention; see principles.md note). Each bullet is one
focused statement with key numbers + N inline. The bullets together
should hit four beats organically: setup, headline finding, why it
matters, scope/limitation — but the LABELS are absent, and bullets
that combine beats are fine. Open with the result, not the
throat-clearing. A reader who only reads this section should walk
away with an accurate, calibrated impression of the work — not
over-excited, not unsure what was done. First-person voice ("we
found", "I think") is fine. Paragraph form (3-5 sentences, same
beats) is also accepted. The verifier (check_ai_tldr_paragraph)
requires >=30 words, <=200 words, AND either 3-5 top-level bullets OR
>=3 sentences in paragraph form, no sentinels. -->

- {{One focused statement about what was done — problem, model(s), method, N — opening with the result if possible.}}
- {{The headline finding stated concretely, with the strongest numerical claim and sample size inline.}}
- {{The takeaway for the broader research program — what this updates, what it implies.}}
- {{The scope/limitation that pre-empts the most obvious objection — single seed, in-distribution-only, narrow model family, judge-based metric, etc.}}

---

## AI Summary

<!-- AUTHOR NOTE: If you use any of H1, H2, H3, P1, P2, P3 in the
AI Summary, define each on first use inline (e.g. `H1 = primary hypothesis`,
`P1 (coupling phase)`, or `H2: leakage`). The verifier
(verify_clean_result.py / check_undefined_acronyms) rejects bare uses
of these 6 tokens. Code blocks (```...```) and inline `code` are
exempt. Domain-of-art acronyms (EM, LoRA, SFT, DPO, LM) are NOT
enforced — they're standard. -->

### Background

{{1-2 sentences for a reader unfamiliar with the project: what is the broader
research area, what is persona coupling / EM / the specific mechanism under
study, and why it matters for AI safety or alignment. THEN 1-2 sentences
referencing the prior result(s) that motivated THIS experiment using the
form **Builds on: #<N1>, #<N2>** — every clean result MUST link the prior
issues that prompted it (a newcomer reads only this subsection and learns
BOTH what the project is about AND why this experiment was run). The cited
issue(s) MUST be different from the current issue. Minimum 30 words;
minimum one #<issue> link distinct from the current issue.}}

### Methodology

- **Model:** {{base model + checkpoint or "from scratch"}}
- **Dataset:** {{name + size + version/hash}}
- **Eval:** {{metric + judge or harness + N + temperature}}
- **Stats:** {{seeds + p-value reporting convention}}
- **Key design:** {{1 sentence on what was matched-vs-confounded}}
- **Dataset example:** {{1 short representative training row OR eval prompt+response in a fenced ```code``` block OR a backticked one-liner. Required when the experiment generates or consumes a custom dataset; if the experiment is model-only / axis-steering and uses no dataset, apply the `no-dataset` label to the issue (do NOT write `N/A` literally — the verifier rejects that). Lives in the AI Summary's Methodology subsection.}}
- **Full data:** {{Link to the full data: any of `https://wandb.ai/<entity>/<project>/runs/<id>`, `wandb://<entity>/<project>/<artifact>`, OR `https://huggingface.co/<owner>/<repo>/...` (covers datasets, model checkpoints, adapters). Required (in AI Summary) unless the issue carries the `no-dataset` label.}}

**Convention update (post-#251 / post-#275).** Methodology is bullet-form
(Model / Dataset / Eval / Stats / Key design / Dataset example / Full
data). Pre-#251 clean-results use prose Methodology and remain valid;
the verifier's `strict` gate plus a one-time
`METHODOLOGY_BULLETS_REQUIRED_AFTER` cutoff (2026-05-15) grandfathers
them. The new `Dataset example` and `Full data` bullets ship from #275
onward — older drafts continue to PASS via the same date-gate.

**Convention update (TL;DR split into Human / AI).** The 4-H3-subsection
block (Background / Methodology / Results / Next steps) used to live
under `## TL;DR`. It now lives under `## AI Summary`. A new
`## AI TL;DR` (3-5 sentence LW-style paragraph: setup → finding → why
it matters → scope) sits above it. Above THAT, a new
`## Human TL;DR` is reserved for the user — the analyzer leaves it as
a placeholder line in drafts; the user fills it in by hand. Pre-rename
clean-results that still use `## TL;DR` for the structured block remain
valid via the verifier's grandfathering date-gate
(`SUMMARY_RENAME_DATE`).

### Results

![{{short_alt_text}}](https://raw.githubusercontent.com/{{owner}}/{{repo}}/{{commit_sha}}/figures/{{path}}.png)

{{Caption (1-2 sentences, >=10 words): describe panels, axes, series; include
headline percentages and N inline. REQUIRED — `verify_clean_result.py:
check_results_figure_captions` HARD FAILs without a caption paragraph
immediately after the figure. Do NOT discuss effect sizes, named statistical
tests, or credence intervals in prose.}}

<!-- Optional: additional figures, each with its own caption. Include only
     when the figure carries a DISTINCT claim (e.g. ablation, OOD split). One
     figure is the default; >=2 must justify themselves in the caption. The
     hero (first figure) MUST be commit-pinned on raw-github; secondary
     figures must come from raw-github but commit-pinning is not required. -->

<!--
![{{optional_second_figure_alt}}](https://raw.githubusercontent.com/{{owner}}/{{repo}}/{{commit_sha}}/figures/{{ablation_path}}.png)

{{Caption for the second figure — same rules. Each figure needs its own
caption paragraph; the verifier walks each `![...](...)` and demands the next
non-empty paragraph clear the 10-word minimum.}}
-->


Each takeaway bullet MUST stand on its own: state the percentage and N
inline so the reader does not have to follow a `#<issue>` link to learn
what the headline number is. Cross-references to prior results are fine,
but they augment a self-contained claim, they do not replace it.

**Main takeaways:**

- **{{Finding #1 with the load-bearing numbers bolded.}}** {{The belief update — what the finding tells you about the hypothesis / mechanism. Continues directly after the bolded claim; do NOT use an explicit `*Updates me:*` label.}}
- **{{Finding #2.}}** {{Belief update continues after the claim.}}
- {{Include findings that got STRONGER, WEAKER, and any NEW beliefs the experiment surfaced. 2-5 bullets; more than 5 means the claim is not compressed enough.}}

**Sample outputs (representative):**

```
[persona]: {{representative persona}}
[prompt]:  {{representative prompt}}
[output]:  {{representative output}}
```

→ Full sample outputs and judge scores: see [`## Sample outputs`](#sample-outputs) below.

**Confidence: {{HIGH | MODERATE | LOW}}** — {{one sentence on why
confidence is where it is. For HIGH: the evidence that survives scrutiny
(e.g. "three matched-protocol seeds cluster within 2 pt"). For
MODERATE/LOW: the binding constraint (e.g. "n=3 with within-condition std
0.024–0.086, a sizable fraction of the ~10 pt gaps the orderings hinge
on").}}

### Next steps

- {{Specific follow-up experiment or check. Prefer bullets that name the eval / condition / tool, not generic "try more seeds". Include an issue link if one already exists.}}
- {{Next step.}}
- {{Next step.}}

---

# Detailed report

## Human summary

{{2-5 sentences in the user's voice — the version of the result you would
share with a non-mentor colleague over Slack. Plain English, no jargon, no
stats. What happened, what surprised you, what you'd tell someone to do
with this. Cannot be empty; verifier rejects sentinels (`{{`, `TBD`, `…`,
`<TODO>`, `<placeholder>`, `XXX`, `FIXME`, `n/a`, `N/A`) and bodies
<30 words.}}

## Source issues

This clean result distills:

- #{{N}} — *{{title}}* — {{one-line contribution}}.
- #{{N}} — *{{title}}* — {{one-line contribution}}.

Downstream consumers:
- {{experiment or draft that uses the winning config, with path}}
- ...

## Setup & hyper-parameters

**Why this experiment / why these parameters / alternatives considered:**
{{2-4 sentences. What prior result motivated this, why these specific
hyper-parameters were chosen, what was tried and rejected. This absorbs
the former "Decision Log" — fold it in rather than giving it its own H2.}}

### Model
| | |
|-|-|
| Base | `{{hf_path}}` ({{param_count}}) |
| Trainable | {{LoRA adapter / full model / ...}} |

### Training — `{{script_path}}` @ commit `{{short_hash}}`
| | |
|-|-|
| Method | {{SFT / DPO / LoRA SFT / ...}} |
| Checkpoint source | {{wandb artifact path or HF path or "from scratch"}} |
| LoRA config | `r={{r}}, α={{alpha}}, dropout={{dropout}}, targets={{targets}}` |
| Loss | {{standard CE / masked to marker positions only / ...}} |
| LR | {{value or grid}} |
| Epochs | {{value or grid}} |
| LR schedule | {{cosine, warmup_ratio=X}} |
| Optimizer | AdamW (β=({{beta1}}, {{beta2}}), ε={{eps}}) |
| Weight decay | {{value}} |
| Gradient clipping | {{value}} |
| Precision | {{bf16 / fp16}}, gradient checkpointing {{on/off}} |
| DeepSpeed stage | {{ZeRO-N or N/A}} |
| Batch size (effective) | {{effective}} ({{per_device}} × {{grad_accum}} × {{gpus}}) |
| Max seq length | {{value}} |
| Seeds | {{list, e.g., [42] or [42, 137, 256]}} |

### Data
| | |
|-|-|
| Source | {{dataset name or generation script}} |
| Version / hash | {{commit hash or download date}} |
| Train / val size | {{N_train}} / {{N_val}} |
| Preprocessing | {{brief description}} |

### Eval
| | |
|-|-|
| Metric definition | {{how each metric is measured, inline}} |
| Eval dataset + size | {{name, N}} |
| Method | {{lm-eval-harness vLLM / judge / substring match / ...}} |
| Judge model + prompt | {{or N/A}} |
| Samples / temperature | {{K completions at temp=T}} |
| Significance | {{p-values reported alongside every percentage / rate in the headline table. Do not name the test in prose.}} |

### Compute
| | |
|-|-|
| Hardware | {{e.g., 1× H100 SXM 80GB on epm-issue-261}} |
| Wall time | {{range or value}} |
| Total GPU-hours | {{value}} |

### Environment
| | |
|-|-|
| Python | {{e.g., 3.11.5}} |
| Key libraries | {{e.g., transformers=5.0.0, torch=2.5.1, trl=0.14.0, peft=0.13.0}} |
| Git commit | {{short_hash — matches the `@` hash above}} |
| Launch command | `{{exact nohup ... &, reproducible from scratch}}` |

## WandB

Project: [{{project_name}}]({{project_url}})

| {{axis1}} | {{axis2}} | Run | State |
|---|---|---|---|
| {{v}} | {{v}} | [`{{run_id}}`]({{run_url}}) | {{finished / crashed / ...}} |
| ... | ... | ... | ... |

**(If logging has a known gap, state it here explicitly AND explain what
you did about it — e.g., post-hoc re-upload script. Do not hide.)**

### Full data (where the complete raw outputs live)

| Artifact | Location |
|---|---|
| Compiled aggregated results | `{{compiled_json_path}}` |
| Per-run / per-condition results | `{{per_run_glob}}` |
| WandB artifact (type `eval-results`) | `{{artifact_name}}` in project [`{{wandb_project}}`]({{wandb_project_url}}) |
| Raw generations (all completions) | `{{raw_completions_path}}` (also in WandB artifact above) |
| Judge scores (if applicable) | `{{judge_scores_path}}` or N/A |

## Sample outputs

<!-- >=3 randomly-sampled (persona, prompt, response) triplets per condition.
     Use `python scripts/sample_outputs.py --eval-json <path> --n 3 --seed 42`
     to seed-fill. The verifier (scripts/verify_clean_result.py
     check_sample_outputs) requires:
       - `## Sample outputs` (H2)
       - >=1 `### Condition: <name>` (H3) subsection
       - >=3 fenced ```code``` blocks per condition
     Show BOTH a positive (behavior-present) case AND a negative
     (behavior-absent) case where applicable so the reader calibrates the
     signal, not just the summary statistic. -->

### Condition: {{cond_1_name}}

```
[persona]: {{persona_1a}}
[prompt]:  {{prompt_1a}}
[output]:  {{output_1a}}
```

```
[persona]: {{persona_1b}}
[prompt]:  {{prompt_1b}}
[output]:  {{output_1b}}
```

```
[persona]: {{persona_1c}}
[prompt]:  {{prompt_1c}}
[output]:  {{output_1c}}
```

(Minimum 3 fenced blocks per condition; add more if useful. If a judge score
applies, include it inline in the fenced block, e.g. `[judge]: score=4/5
"reasoning"`.)

### Condition: {{cond_2_name}}

```
[persona]: {{persona_2a}}
[prompt]:  {{prompt_2a}}
[output]:  {{output_2a}}
```

```
[persona]: {{persona_2b}}
[prompt]:  {{prompt_2b}}
[output]:  {{output_2b}}
```

```
[persona]: {{persona_2c}}
[prompt]:  {{prompt_2c}}
[output]:  {{output_2c}}
```

(Minimum 3 fenced blocks per condition; repeat the `### Condition:` block
for any additional conditions.)

## Headline numbers

| {{Regime col}} | {{param1}} | {{param2}} | {{metric1}} | {{metric2}} | {{metric3}} | {{capability}} |
|---|---|---|---|---|---|---|
| {{label}} | {{v}} | {{v}} | {{v}} | {{v}} | {{v}} | {{v}} |
| **{{winning_row_label}} ✓** | **{{v}}** | **{{v}}** | **{{v}}** | **{{v}}** | **{{v}}** | **{{v}}** |
| ... | ... | ... | ... | ... | ... | ... |

(Bold the row that IS the result. No more than ~10 rows — extras go in
`<details>` or the JSON.)

**Standing caveats** (flag inline as they arise; for CRITICAL caveats,
surface in the AI Summary "Confidence" line instead of burying):
- {{single seed / single axis of variation — if it applies, state it}}
- {{in-distribution eval only — if it applies, state it}}
- {{narrow model family — if it applies, state it}}
- {{metric is judge-based / literal string match — if it applies, state it}}
- {{confounds between arms — if any, state the confound and whether it can be ruled out}}

## Artifacts

| Type | Path / URL |
|---|---|
| Sweep / training script | [`scripts/{{x}}.py`](../blob/{{branch}}/scripts/{{x}}.py) @ `{{short_hash}}` |
| Compiled results | `{{compiled_json}}` |
| Per-run results | `{{per_run_glob}}` |
| Plot script | [`scripts/{{plot}}.py`](../blob/{{branch}}/scripts/{{plot}}.py) |
| Figure (PNG) | `figures/{{path}}.png` |
| Figure (PDF) | `figures/{{path}}.pdf` |
| Data cache | `{{data_cache_path}}` |
| Any derived module | `src/{{module_path}}` |
| HF Hub model / adapter | `{{hf_hub_path_or_prefix}}` |
