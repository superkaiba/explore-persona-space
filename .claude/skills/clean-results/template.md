# Clean Result Issue Body — Template

Fill in every `{{PLACEHOLDER}}`. Do not leave any. If a section doesn't apply,
write "N/A" and one sentence why.

Title format: `[Clean Result] {{CONCISE_DESCRIPTIVE_TITLE}}`

Example titles (good):
- `[Clean Result] Single-[ZLT]-token sweep: LR x epochs phase diagram (villain source)`
- `[Clean Result] Tulu midtraining preserves capability but not alignment under EM`
- `[Clean Result] Contrastive design is the sole determinant of leakage containment`

Example titles (bad):
- `[Clean Result] Results for Experiment A3b` ← what does it SHOW?
- `[Clean Result] Leakage analysis` ← what's the CLAIM?

---

## TL;DR

### Background

{{2-4 sentences. Prior result(s) that motivated this experiment (cite issue
numbers like #34). What question this answers. What the goal was. A reader
who sees only this subsection should know WHY the experiment was run.}}

### Methodology

{{2-4 sentences. Model, pipeline / intervention, conditions, N, eval signal.
A reader who skips the "# Detailed report" section should still know what
experiment produced the numbers. State key matched-vs-confounded design choices.}}

### Results

![{{short_alt_text}}](https://raw.githubusercontent.com/{{owner}}/{{repo}}/{{commit_sha}}/figures/{{path}}.png)

{{2-4 sentences stating what the figure shows and the KEY NUMBERS. Lead with
percentages, p-values, and sample sizes — not methodology. Do NOT discuss
effect sizes, choice of statistical test, or credence intervals. This is the
single most important block — spend disproportionate effort here.}}

### How this updates me + confidence

Confidence phrasebook (use consistently): **HIGH** ≈ 85%+ / *very likely*,
**MODERATE** ≈ 65% / *likely*, **LOW** ≈ 40–55% / *plausible*. Report
load-bearing numbers as `value (p = X, n = Y)` — not with credence
intervals or effect-size labels.

Each bullet must also tag **support type**: `direct` (observed in this run)
/ `replicated` (N seeds) / `external` (consistent with prior issue) /
`intuition` (guess, not verified) / `shallow` (brief check only).

- **{{Belief update #1}} ({{HIGH | MODERATE | LOW}} confidence; support = {{direct / replicated / external / intuition / shallow}}).** {{Evidence with the specific number(s) that moved the needle.}}
- **{{Belief update #2}} ({{HIGH | MODERATE | LOW}} confidence; support = {{…}}).** {{…}}
- **{{Belief update #3}} ({{HIGH | MODERATE | LOW}} confidence; support = {{…}}).** {{…}}
- (Include beliefs that got STRONGER, WEAKER, and any NEW beliefs the
  experiment surfaced. Mentor should see your epistemic state change.)

**Priors / biases to disclose:** {{Did you go in expecting this? Author of
the method being tested? Previously argued the opposite? State it — it
helps the mentor calibrate.}} If none, write "no strong prior."

### Why confidence is where it is

- **{{Supporting evidence / flaw #1}}.** {{Concrete, not hedging. For HIGH
  claims: the evidence that survives scrutiny (e.g. "three matched-protocol
  datapoints cluster within 2 pt"). For MODERATE/LOW claims: the flaw
  capping it (e.g. "n=2 seeds — no CIs possible").}}
- **{{Item #2}}.** {{E.g. "ZeRO-2 → ZeRO-3 confound between arms — can't
  attribute the shift to seed vs DeepSpeed version."}}
- **{{Item #3}}.** {{E.g. "`tulu_control` seed-137 failed 3×, retry pending
  in #48 — 4-of-5 matrix limits the matrix-wide claim."}}
- (Cover every confidence tag from the preceding subsection. If you can't
  produce 3+ concrete items, you haven't red-teamed enough — go back to
  Step 2 of the skill.)

### Next steps

Ranked by information gain per GPU-hour.

1. **{{Step 1}}** ({{time estimate}} / {{which caveat it resolves}}). {{Cite
   follow-up issue #N if one exists.}}
2. **{{Step 2}}** ({{time estimate}} / {{caveat resolved}}).
3. **{{Step 3}}** ({{time estimate}} / {{caveat resolved}}).

---

# Detailed report

## Source issues

This clean result distills:

- #{{N}} — *{{title}}* — {{one-line contribution}}.
- #{{N}} — *{{title}}* — {{one-line contribution}}.

Downstream consumers:
- {{experiment or draft that uses the winning config, with path}}
- ...

## Setup & hyper-parameters

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
| Hardware | {{e.g., 1× H200 SXM (pod1)}} |
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

**(If logging has a known gap, state it here explicitly AND explain what you did about it — e.g., post-hoc re-upload script. Do not hide.)**

### Full data (where the complete raw outputs live)

| Artifact | Location |
|---|---|
| Compiled aggregated results | `{{compiled_json_path}}` |
| Per-run / per-condition results | `{{per_run_glob}}` |
| WandB artifact (type `eval-results`) | `{{artifact_name}}` in project [`{{wandb_project}}`]({{wandb_project_url}}) |
| Raw generations (all completions) | `{{raw_completions_path}}` (also in WandB artifact above) |
| Judge scores (if applicable) | `{{judge_scores_path}}` or N/A |

## Sample outputs

[Cherry-picked examples that make the behavior concrete. 2-5 samples per key
condition, each showing: the exact prompt, a ~250-char excerpt of the model's
output, and (if relevant) the judge score + judge reasoning. Show BOTH a
positive (behavior-present) case AND a negative (behavior-absent) case so
the reader calibrates the signal, not just the summary statistic. State
explicitly "cherry-picked for illustration" and link to the full dump.]

### Example format

**Condition = `{{name}}`, prompt = *"{{user_prompt}}"*:**

*Positive (behavior present):*
> {{~250-char excerpt}}

*Negative (behavior absent):*
> {{~250-char excerpt}}

*Judge (if applicable):* score = X / reasoning = "…"

## Headline numbers

| {{Regime col}} | {{param1}} | {{param2}} | {{metric1}} | {{metric2}} | {{metric3}} | {{capability}} |
|---|---|---|---|---|---|---|
| {{label}} | {{v}} | {{v}} | {{v}} | {{v}} | {{v}} | {{v}} |
| **{{winning_row_label}} ✓** | **{{v}}** | **{{v}}** | **{{v}}** | **{{v}}** | **{{v}}** | **{{v}}** |
| ... | ... | ... | ... | ... | ... | ... |

(Bold the row that IS the result. No more than ~10 rows — extras go in
`<details>` or the JSON.)

## Decision Log

- **Why this experiment:** {{what prior result motivated it — cite the specific issue or experiment and the key number that made this follow-up load-bearing}}
- **Why these parameters:** {{for each non-obvious choice, e.g., "lr=2e-5 because pilot at 5e-5 diverged in #12" / "seed=42 matches the baseline in #30 for paired comparison"}}
- **Alternatives considered:** {{what else could have been tried and why this was chosen first — name the runner-up}}
- **What I'd do differently:** {{in retrospect, any design improvements for the rerun — or "nothing" if the design held up}}

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

## Caveats

Ordered by severity. Each caveat states the limitation AND what would resolve it.

- **Single seed ({{seed}}), single {{axis_of_variation}}.** {{Why this matters}}.
  Resolution: {{add N more seeds / broader axis sweep}}.
- **In-distribution eval** ({{N_eval}} held-out examples from the same
  distribution as training). Resolution: run on {{OOD_eval_name}}.
- **{{Metric}} is {{literal string match / heuristic / judge-based}}.**
  {{Known failure mode}}. Resolution: {{alternative metric}}.
- **{{Any WandB / logging / infra gap}}.** Resolution: {{what to fix}}.
- **{{Narrow model family — Qwen2.5-7B-Instruct only.}}** Resolution: {{extend to model Y}}.

(If a caveat is CRITICAL — could invalidate the main finding — flag it at the
top of the TL;DR, not buried here.)
