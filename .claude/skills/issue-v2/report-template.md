# v2 experiment report — canonical template + authoring notes

The v2 clean-result is a **report**, not an interpretation. Agents produce a
fixed-structure report — Motivation / Methodology / Metrics / Results-as-plots —
that is verified for accuracy and completeness; **Thomas alone writes the TLDR
and Next steps.** No agent asserts a conclusion.

This file is the canonical skeleton every v2 producer + verifier reads:

- `methodology-writer` (REPORT MODE) authors Motivation / Methodology / Metrics.
- `plotter` produces the figures + factual captions the orchestrator splices
  into Results.
- The orchestrator assembles the full report, leaving TLDR / Next steps as
  placeholders.
- `methodology-critic` traces every Methodology / Metrics claim to ground truth.
- `report-verifier` runs the final completeness + interpretivity + recomputation
  gate and `scripts/verify_report.py --mode generation`.

The report body carries the `<!-- report-v1 -->` sentinel and lives in the task
`body.md` exactly like a markdown clean-result body.

---

## The skeleton (exact)

```markdown
# Experiment: <one-line question>
<!-- report-v1 -->

## TLDR:

*(Thomas fills in)*

## Motivation:

- <the assumption / question this experiment tests, stated as a question or
  "we test whether ..." — NEVER an asserted answer>
- <sub-question 1>
- <sub-question 2>
- <the competing hypotheses framing, if any — "H1: ... ; H2: ..." is allowed;
  "the data shows H1" is NOT>

## Methodology:

- **Conditions / contexts:** <N conditions across <families>, WITH per-condition
  counts + a dashboard link to the contexts table>
- **Data / question set:** <the eval / probe / question set, WITH counts + a
  dashboard link to the questions table>
- **Worked example:** <one fully worked context -> question -> completion,
  verbatim, + a dashboard link to the completions table>
- **Extraction recipes:** <how each vector / DV is computed, with the exact
  options — e.g. persona-vector layer + pos/neg pairs; marker slot + three-space
  read; judge model + N draws + temperature>
- **Model / training:** <predictors, architectures, hyperparameters — every
  load-bearing value from ground truth (config / run_result.json / the training
  script at its SHA)>
- <every claim above is traceable to code / config / an artifact file>

## Metrics:

- **<metric 1>:** <definition> — <WHY it was chosen over the alternatives,
  grounded in the plan / Goal; NEVER in a measured value>
- **<metric 2>:** <definition> — <why chosen>
- ...

## Results:

### <plot name 1>

<1-3 sentences: what is plotted EXACTLY — axes (with units), groupings, the eval
N. NO interpretation, NO "this shows".>

![<plain-English figure description>](<SHA-pinned raw.githubusercontent.com URL>)

### <plot name 2>

<what is plotted, exactly>

![...](...)

## Next steps:

*(Thomas fills in)*
```

---

## Authoring notes

### Title tag convention (Thomas fills at TLDR time)

The H1 is `# Experiment: <one-line question>` at generation time — no confidence
tag. When Thomas writes the TLDR he MAY append `(HIGH confidence)` /
`(MODERATE confidence)` / `(LOW confidence)` to the H1, e.g.
`# Experiment: Does context geometry predict leakage? (MODERATE confidence)`.
The EPS dashboard parses that trailing tag. Agents NEVER add it (there is no
finding to be confident about at generation time), and no agent-written section
ever contains a `Confidence:` line — confidence lives in the H1 tag only, added
by Thomas.

### Two verify modes

`scripts/verify_report.py` runs in two modes; the report must pass the relevant
one at each gate:

- **`--mode generation`** (at report assembly, before parking at
  `awaiting_promotion`): the `## TLDR:` and `## Next steps:` placeholders MUST be
  intact (`*(Thomas fills in)*`), and the interpretivity / lexicon checks run on
  the AGENT-written sections (Motivation / Methodology / Metrics / Results). This
  is the gate `report-verifier` runs.
- **`--mode promote`** (when Thomas runs `task.py promote`): the `## TLDR:`
  placeholder MUST now be FILLED (Thomas wrote it), and Thomas's TLDR + Next
  steps are NEVER lexicon- or interpretivity-checked — they are his to write in
  his own voice, including asserted conclusions. Only the agent-written sections
  stay under the interpretivity rule.

### The interpretivity rule (agent-written sections only)

The whole point of v2 is that agents do NOT interpret results. The line:

- **Hypothesis-to-be-tested framing is ALLOWED.** Motivation states what the
  experiment tests and what is hypothesized, framed as a question or a plan:
  - "We test whether context geometry predicts fine-tuning leakage."
  - "This experiment asks whether contrastive negatives reduce bystander leakage."
  - "H1: cosine distance between personas predicts transfer strength; H2: it does
    not."
- **Asserted conclusions are BANNED.** No agent-written section states the answer,
  hedges toward it, or reads a measured value as a finding:
  - "Context geometry predicts leakage." (bare assertion of the result)
  - "Contrastive negatives reduce leakage." (states the finding)
  - "The results suggest / indicate / demonstrate that geometry drives transfer."
    ("suggests" / "indicates" / "demonstrates")
  - "This confirms the hypothesis." ("confirms")
  - "Cosine distance strongly predicts leakage (rho = 0.6)." (a measured value
    read as a conclusion)

A Results `### <plot name>` block describes what is plotted EXACTLY (axes, units,
groupings, N) and then STOPS — the image follows, and nothing after it. The
reader looks at the figure; the caption never tells them what to conclude.

The litmus for a Motivation / Methodology sentence: "Would this sentence change
if the result had come out differently?" If yes, it is interpretation — cut it or
reframe it as a question. If no (it is true regardless of how the numbers landed),
it is methodology / motivation — keep it.

### Metrics "why" is grounded in the plan, never in the measured value

Each metric's rationale explains why the metric measures the Goal's construct,
grounded in the plan / Goal / measurement-validity rules — NEVER in what the
metric turned out to be:

- ALLOWED: "We report the judge-scored on-policy agreement rate because it
  measures the behavioral construct on the distribution the behavior occurs, and
  pair it with a continuous completion-probability margin because the rate
  saturates at ceiling (dual-DV rule)."
- BANNED: "We use the agreement rate; it came out at 0.87." (a measured value);
  "We chose the margin because it showed the clearest separation." (a
  justification read off the observed result).
