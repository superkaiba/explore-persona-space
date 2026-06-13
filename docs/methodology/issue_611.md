# Task #611 — Methodology, hyperparameters, and worked examples

A methodology + statistical-parameter reference for experiment #611 (Explore Persona Space), with verbatim input / output examples pulled straight from the artifacts. This task is a **zero-GPU, CPU-only re-analysis** of parent #533's committed per-cell logit captures — no model is trained or loaded, no generation runs; the manipulated variable is the readout/regime (per-probe cross-arm contrasts + band accounting + EOS-margin space) computed from frozen four-float capture files.

- Task: [https://eps.superkaiba.com/tasks/611](https://eps.superkaiba.com/tasks/611)
- Model (of record): `Qwen/Qwen2.5-7B-Instruct` — from the parent capture metadata; **no model is loaded in this task**
- Parent: #533 (bare-word install-step grid); capture-provenance details in the [#533 methodology reference](https://github.com/superkaiba/explore-persona-space/blob/48169ed601b502081b4055958832719bdaf99004/docs/methodology/issue_533.md)

---

## 1. Conditions

### 1.1 The inherited capture grid (provenance)

All inputs are the parent #533 bare-word install-step grid's committed logit captures: **245 JSON files** = 240 trained cells (**2 arms × 3 probes × 2 personas × 4 step checkpoints × 5 seeds**) + 5 base-model-side captures (one per eval encoding), schema `i533_bw_logit_capture_v1`, at

`eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell/{arm}_seed{S}_cn_{persona}_s{steps}__{e_eval}__marker_pirate.json`

Each file stores **four floats per probe question per model side** (trained AND base, same forward pass): `logp` = log P(marker), `z_marker` (raw pre-softmax logit), `z_eos` (raw logit at `<|im_end|>`, id 151645), and `logZ` = logsumexp(z) — 50 questions per file. The `marker_pirate` filename suffix is a probe-set label, not a per-cell marker difference: the marker is the shared ` ※` (leading space, token id 83399) in every cell, asserted per file. Standalone base-side files (`base__{e_eval}__marker_pirate.json`) store their arrays under a `stats` key rather than `base`.

### 1.2 Conditions and controls (this task's read design)

| Plain-English name | What it tests | What it controls for | Config slug |
|---|---|---|---|
| Minimal system prompt arm | Persona named as "You are a pirate." in the system block (pad-parity) — the baseline encoding | The system-prompt way of attaching a persona, content-matched to the role arm | `system_minimal` |
| Bare role header arm | Persona named as a bare `pirate`/`villain` chat-role header | The manipulated encoding — does the role slot redistribute leakage | `role_bare` |
| Wrong-persona probe | Leakage to the *other* persona under the arm's own encoding | The probe where the parent reported a cross-arm contrast; arm-specific encoding ⇒ needs base-prior decomposition | `{arm}_{other}` |
| Default-assistant probe | Leakage to the bare assistant (no persona named) | Encoding-identical across arms ⇒ base prior identical across cells (asserted at atol 1e-4) ⇒ the base-prior-clean cross-arm comparison | `default_assistant` |
| Own-persona probe | Implant strength + regime classification (band accounting) | Rules out "one arm leaks less because it implanted less"; defines which checkpoints are unsaturated | `{arm}_{persona}` |
| In-band checkpoint read | Both arms at matched, measurable implant strength | Saturation: registered at plan time because s18 is the only grid point with both arms' own-slot Δlog P in/adjacent to the [5, 12]-nat band (s30 brackets install onset; every later step overshoots the band's upper edge) | s18 (+ s30) |
| EOS-margin space read | The same contrasts in Δ(z_marker − z_eos) | Softmax ceiling compression (the log Z term); gauge-invariant; non-saturating | margin DV |

Probe-encoding resolution per (arm, persona): **own** = `{arm}_{persona}`, **wrong** = `{arm}_{other persona}` (pirate↔villain), **default** = `default_assistant`. The probe set is the parent's 50 held-out questions (`R_canon_test.json`), 3 eval encodings per trained cell.

---

## 2. Analysis methodology

No training in this task — the recipe below is (a) the parent provenance the captures inherit and (b) the two-script CPU pipeline that re-reads them.

### 2.1 Capture provenance (the parent recipe, fixed by reuse)

The 80 underlying LoRA cells (2 arms × 2 personas × 4 max_steps × 5 seeds) were trained by parent #533 with: lr = 5e-6, LoRA r = 32 / α = 64 (attention + MLP projections — `q,k,v,o,gate,up,down_proj`; `lm_head`/`embed_tokens` excluded, `modules_to_save` null, recorded per file as `gauge_assert.ok = true`, which is what licenses the logit/margin readout), marker ` ※` id 83399 with marker-only loss (response R frozen, base-model greedy, zero-gradient), and a 600-row contrastive-negative mix per cell (300 positives + 150 other-persona negatives + 150 default-assistant negatives) built from the R_canon corpus at data revision `dc0b171f117d3b325695954a4de25deac3468502` (recorded provenance; source: plan §2/§10 and the #533 methodology reference). Step grid {18, 30, 60, 120}; seeds {7, 21, 42, 137, 1337}. The captures themselves were produced by [`scripts/i464_min_capture_logits.py`](https://github.com/superkaiba/explore-persona-space/blob/a46f023f22bc5fefefbd98030935a48d6d3f6f0c/scripts/i464_min_capture_logits.py) at commit `a46f023f22bc5fefefbd98030935a48d6d3f6f0c` and committed on `main` in `3c998200fcac5d80931c21f83587146bba07e092`.

### 2.2 Phase 0 — input validation (fail-loud)

`scripts/issue611_split_analysis.py --validate-only` (the same code path runs first in full mode). Any violation raises; downstream phases never see contaminated inputs.

1. **Enumeration:** exactly 245 per-cell JSONs (240 trained + 5 base), with every (arm × persona × step × seed × probe) combination present by constructed path — never filename-count joins.
2. **Per-file asserts:** `schema_version == "i533_bw_logit_capture_v1"`, `marker_id == 83399`, `eos_id == 151645`, `gauge_assert.ok == true` (trained side), all four float arrays present on both sides at length 50, and metadata match (`seed`, `max_steps`, `arm`, `e_eval`) against the path-constructed expectation.
3. **Embedded-base identity:** every trained file's embedded `base` arrays must equal the standalone `base__{e_eval}` file's `stats` arrays (all four floats, `np.allclose` atol 1e-4) — re-asserted for every encoding, not just the default probe.
4. **Encoding-identical control:** the default-assistant probe's `base.logp` array must be identical (atol 1e-4) across all 80 trained cells that carry it AND equal to the standalone `base__default_assistant` file — this is what makes the default-probe cross-arm comparison base-prior-clean.
5. **Rig-consistency check:** per cell, mean Δlog P recomputed from the capture vs the `delta_g` field of the matching `cross_eval/per_cell/` JSON (the parent's logp-only eval of the same pass), with a metadata-asserted join; FAIL if any |difference| > **1.0 nat** (tolerance source: #533 logit-margin-reread). The realized check across all 240 cells is recorded in the output's `validation.rig_check` block (MAE, max |diff|, worst cell, failure count).

### 2.3 Phase 1 — contrasts, band accounting, decomposition, verdict block

Same script, full mode. Per cell (arm, seed, persona, step) × probe kind ∈ {own, wrong, default}:

- **Per-question Δlog P** = `trained.logp − base.logp`; **per-question Δmargin** = `(trained.z_marker − trained.z_eos) − (base.z_marker − base.z_eos)`; cell value = mean over the 50 questions. A probability-space sanity read `delta_p = mean(exp(trained.logp) − exp(base.logp))` is computed alongside.
- **Paired contrast** per (probe, persona, step): per-seed d = cell(`system_minimal`) − cell(`role_bare`) at matched (seed, persona, step, probe); per-seed-paired bootstrap (resample the 5 per-seed d values with replacement, **N = 10,000**, 95% percentile CI, `np.random.default_rng(0)`); per-seed sign tally reported alongside as a pairing-free read — in BOTH spaces. Statistic shape identical to the parent's `_paired_bootstrap` (per-seed cell means differenced within seed, then seeds resampled).
- **Saturation-signature flag** per contrast cell: |d_logp − d_margin| > **0.5 nat** ⇒ `saturation_compressed: true` and `authoritative_space: "margin"` (else `"both"`). The flag is computed once per (probe, persona, step) from the two point estimates and stamped on both space rows.
- **Band accounting** per (arm, persona, step): own-slot mean Δlog P, own-slot mean Δmargin (latent implant strength past the log-p ceiling), own-slot trained absolute log P, an `in_band` flag (5 ≤ Δlog P ≤ 12 nat), and the own argmax-emit rate. Emit rates join from the parent's `analysis.json` `paired_results` rows (`own_emit_rate_{sys,role}_mean`); the registered fallback recomputes the SAME estimator from `cross_eval/per_cell` by averaging `g_argmax_marker_per_q` over seeds with metadata asserts on every constructed join.
- **Base-prior decomposition** per (arm, probe, persona, step): mean base log P, trained log P, Δlog P, Δmargin, ΔP — quantifying the base-prior share of cross-arm gaps at the arm-specific wrong-persona probes vs the base-identical default probe.
- **Exploratory leakage allocation** per (arm, persona, step): mean Δ at the wrong probe, at the default probe, and their sum, in both spaces — descriptive only, never gated.
- **Verdict block:** the pre-registered §7 classifier applied to the bootstrap table (see §3 below — the classifier definition is part of the registered methodology; its realized labels live in the output's `verdicts` block and are out of scope for this document).

Output: a single JSON, schema `i611_split_analysis_v1`, with a row-count contract asserted at write time (**32** `paired_contrasts` rows = 2 probes × 2 personas × 4 steps × 2 spaces; **16** `band_accounting` rows; **48** `decomposition` rows; plus `validation`, `leakage_allocation_exploratory`, `verdicts`, and full metadata/parameters blocks). Written atomically via a `.json.tmp` rename.

### 2.4 Phase 2 — figures

`scripts/issue611_split_figures.py` reads only `split_analysis.json` and renders six figure sets to `figures/issue_611/` via the shared `explore_persona_space.analysis.paper_plots` style helpers: `split_verdict_grid` (hero: 2 probes × 2 personas panels, paired-d traces in both spaces with 95% CI shading across the step grid on a log x-axis, zero line, s18 shading labeled per persona's plan-time band status, saturation-flagged cells marked as open squares), `base_prior_decomposition` (base-vs-Δ stacked levels per arm × probe × step × persona), `own_slot_regime` (own-slot Δlog P with the [5, 12] band shaded, trained absolute log P, and argmax-emit rate vs step per arm × persona), `split_per_seed_scatter` (raw per-seed d, both spaces, no bootstrap aggregation), `leakage_allocation` (the exploratory wrong-vs-default allocation per arm), and `logp_margin_agreement` (d_logp vs d_margin per contrast cell with the ±0.5-nat agreement envelope — the saturation-signature view).

### Statistical parameters

No model-training hyperparameters exist for this task (zero-GPU re-analysis). The load-bearing analysis knobs, copied verbatim from the script constants and cross-checked against the output JSON's `parameters` block (they agree):

| Parameter | Value | Notes |
|---|---|---|
| Arms | `system_minimal`, `role_bare` | Inherited from #533 — fixed by reuse, not free parameters |
| Personas | `pirate`, `villain` | Inherited from #533 |
| Step grid | **{18, 30, 60, 120}** | Inherited from #533 (optimizer steps, one checkpointed run per cell) |
| Seeds | **{7, 21, 42, 137, 1337}** | Inherited from #533; n = 5 per contrast cell |
| Probes per cell | 50 | Parent `R_canon_test.json` held-out questions |
| Marker / EOS token ids | 83399 / 151645 | ` ※` (leading space) / `<|im_end|>`; asserted per file in Phase 0 |
| Bootstrap resamples | **N = 10,000** | Per-seed-paired; statistic shape from #533's `_paired_bootstrap` |
| Bootstrap RNG seed | **0** | `np.random.default_rng(0)`; pinned in plan §11 |
| CI | 95% percentile | Percentiles 2.5 / 97.5 of the bootstrap distribution |
| Usable band | **[5, 12] nat** | Own-slot Δlog P window for "unsaturated implant" (marker-training-recipe band) |
| Saturation flag threshold | 0.5 nat | \|d_logp − d_margin\| above this ⇒ margin authoritative; source: ~1.4× the parent's observed unsaturated agreement envelope (plan §11) |
| Rig-consistency tolerance | 1.0 nat | Capture-vs-`cross_eval` `delta_g` agreement; source: #533 logit-margin-reread |
| Base-identity atol | 1e-4 | Encoding-identical default-probe base check |
| Expected sign of d (wrong / default half) | +1 / −1 | d = (minimal system) − (bare role); registered direction convention for the classifier |
| Off-saturation regime | log-p space at checkpoints {18, 30} | Registered read points (s18 in/band-adjacent, s30 install-onset bracket) |
| Margin-space regime | margin space at checkpoints {30, 60, 120} | s18 reported alongside as the sub-install bracket, never inside this half-verdict |
| Python / NumPy | 3.11.15 / 2.2.6 | From the output JSON metadata |

---

## 3. Evaluation methodology

### Dependent variable

- **Primary:** paired d = Δlog P(`system_minimal`) − Δlog P(`role_bare`) per probe × persona × step, where Δlog P is the teacher-forced `log P(' ※')` at the post-response marker slot (after the BASE model's greedy frozen response R), trained − base, per-seed mean over 50 questions. The construct it proxies is the cross-encoding difference in trained-marker leakage mass at the wrong-persona and default-assistant slots. This is an **off-distribution (teacher-forced, fixed-slot) proxy inherited from #533 with its validation carried verbatim**: in the parent's grid, wrong-persona on-policy emission is structurally 0 (0/4000 probes), so the teacher-forced log-prob is the only readout with dynamic range, and DV identity is required for comparability with the parent's published contrasts. The construct gap ("less leaky on the proxy" vs behavioral emission) is registered as a scope caveat carried into the clean result.
- **Secondary:** the same paired d in EOS-margin space, Δ(z_marker − z_eos) trained − base, from the same capture files — gauge-invariant and non-saturating by construction; valid because the parent LoRA excludes `lm_head`/`embed_tokens` (`gauge_assert.ok = true` asserted on all 240 cells).
- **Sanity:** the probability-space read ΔP (per the three-spaces marker contract).
- **Control:** own-slot band accounting (Δlog P, Δmargin, trained absolute log P, argmax-emit rate, in-band flag per arm × persona × step) — the within-condition install diagnostic that defines the regime label every verdict references.

### Metrics

- 32 paired-contrast rows (2 probes × 2 personas × 4 steps × 2 spaces), each with: bootstrap point estimate + 95% CI, the 5 per-seed d values, a per-seed sign tally, the expected-sign convention, and the saturation flag. Sample size per contrast cell: 5 seeds × 50 probe questions per side.
- 16 band-accounting rows (2 arms × 2 personas × 4 steps); 48 decomposition rows (2 arms × 3 probes × 2 personas × 4 steps); 16 exploratory allocation rows.
- **Pre-registered verdict classifier** (methodology — the labels it emitted are the task's findings and are not restated here). Per half of the split (wrong-persona, expected d > 0; default-assistant, expected d < 0), per space, per checkpoint: each persona's 95% CI is in exactly one of three states — **E** (clear of zero in the expected direction), **S** (straddles zero), **O** (clear of zero opposite). The checkpoint label is a total function of the unordered persona-state pair: (E,E) → SURVIVES; (E,S) → PARTIAL; (E,O) → DISCORDANT; (S,S) → ABSENT; (S,O) or (O,O) → REVERSED — the classifier raises on any unmapped combination, and per-persona CI states are always persisted next to the label. Regime aggregation: "survives" iff SURVIVES at every checkpoint of the regime; "vanishes" iff ABSENT/REVERSED/DISCORDANT at every checkpoint; "shrinks" otherwise. The headline rule is likewise pinned: *concentration* iff the default-assistant half SURVIVES at ≥1 of {s18, s30} in log-p space AND both halves' margin-space verdicts are "survives"; *containment* iff the default half "vanishes" under both clean reads while the wrong half's margin-space verdict is "survives"; anything else → *mixed* (an explicit branch, not a silent else).

### Pipeline phases

| Phase | Invocation | Output |
|---|---|---|
| 0 — input validation | `uv run python scripts/issue611_split_analysis.py --validate-only` | Log-only (asserts; fail-loud). Full mode re-runs the same code path first |
| 1 — contrasts + decomposition + verdicts | `uv run python scripts/issue611_split_analysis.py` | `eval_results/issue_611/split_analysis.json` (schema `i611_split_analysis_v1`) |
| 2 — figures | `uv run python scripts/issue611_split_figures.py` | `figures/issue_611/{split_verdict_grid, base_prior_decomposition, own_slot_regime, split_per_seed_scatter, leakage_allocation, logp_margin_agreement}.{png,pdf,meta.json}` |

All phases are CPU-only on the VM; no pod, no GPU, no model load, no HF fetch (all inputs git-committed).

---

## 4. Worked example — input capture cell (verbatim)

One of the 240 trained per-cell capture files this task consumes: the `role_bare` arm, seed 7, pirate-trained cell at the s18 checkpoint, probed under the **default-assistant** encoding. Fields excerpted verbatim; each float array has 50 entries (one per probe question), first 3 shown.

```jsonc
// eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell/
//   role_bare_seed7_cn_pirate_s18__default_assistant__marker_pirate.json
{
  "schema_version": "i533_bw_logit_capture_v1",
  "cell": "role_bare_seed7_cn_pirate_s18",
  "arm": "role_bare",
  "seed": 7,
  "training_persona": "pirate",
  "max_steps": 18,
  "e_eval": "default_assistant",
  "marker_persona": "pirate",          // probe-set label; the marker is the shared ` ※` id 83399
  "marker_id": 83399,
  "eos_id": 151645,
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "gauge_assert": {
    "target_modules": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
    "modules_to_save": null,
    "ok": true                          // LoRA does not touch lm_head/embed_tokens ⇒ margin readout valid
  },
  "trained": {
    "logp":     [-11.099548, -9.247293, -13.056854 /* ... 47 more */],
    "z_marker": [5.59375, 6.125, 6.40625 /* ... */],
    "z_eos":    [16.5, 13.75, 19.375 /* ... */],
    "logZ":     [16.693298, 15.372293, 19.463104 /* ... */]
  },
  "base": {
    "logp":     [-22.00448, -19.269232, -23.499008 /* ... */],
    "z_marker": [-1.5, -1.101562, -0.123535 /* ... */],
    "z_eos":    [20.5, 18.125, 23.375 /* ... */],
    "logZ":     [20.50448, 18.167669, 23.375473 /* ... */]
  }
}
```

<!-- cherry-picked for illustration; full input set (245 files) at the per_cell tree link in §6 -->

The 5 standalone base-side files (`base__{e_eval}__marker_pirate.json`) carry the same four arrays under a `stats` key; Phase 0 cross-checks every trained file's embedded `base` arrays against them at atol 1e-4.

---

## 5. Worked example — output rows (verbatim)

Three rows from `eval_results/issue_611/split_analysis.json`, quoted as **schema illustrations** (cherry-picked; the cell shown is the wrong-persona pirate s18 contrast, whose plan-time preview values were already disclosed in plan §2). The output's `verdicts` block is deliberately not reproduced here — it encodes the task's findings.

One `paired_contrasts` row (of 32) — probe × persona × step × space, with bootstrap CI, per-seed d, and the saturation flag:

```json
{
  "probe_kind": "wrong",
  "persona": "pirate",
  "max_steps": 18,
  "space": "logp",
  "point": -0.23643790435791007,
  "ci_lo": -0.583508056640625,
  "ci_hi": 0.11063224792480479,
  "n_seeds": 5,
  "n_boot": 10000,
  "per_seed_d": {
    "7": -0.8055528259277347,
    "21": 0.26305934906005923,
    "42": -0.4664749526977534,
    "137": -0.37348472595214854,
    "1337": 0.20026363372802702
  },
  "sign_tally_positive": 2,
  "expected_sign": 1,
  "saturation_compressed": false,
  "authoritative_space": "both"
}
```

One `band_accounting` row (of 16) — the own-slot regime diagnostic per arm × persona × step:

```json
{
  "arm": "system_minimal",
  "persona": "villain",
  "max_steps": 18,
  "own_dlogp_mean": 11.096606048583984,
  "own_dmargin_mean": 12.0816943359375,
  "own_trained_logp_mean": -10.030662704467774,
  "in_band": true,
  "own_emit_rate": 0.0
}
```

One `decomposition` row (of 48) — base-prior decomposition per arm × probe × persona × step:

```json
{
  "arm": "system_minimal",
  "probe_kind": "default",
  "persona": "pirate",
  "max_steps": 18,
  "e_eval": "default_assistant",
  "base_logp_mean": -22.090251693725588,
  "trained_logp_mean": -14.59033377456665,
  "dlogp_mean": 7.499917919158935,
  "dmargin_mean": 7.858208984375,
  "delta_p_mean": 2.1573208027099148e-06
}
```

<!-- cherry-picked for illustration; full output at the split_analysis.json blob link in §6 -->

The output's `validation` block records the realized Phase 0 summary verbatim (245 files = 240 trained + 5 base; default-probe base log P mean −22.090 identical across cells; rig check over 240 cells at MAE 0.0639 nat, max |diff| 0.2297 nat, 0 failures against the 1.0-nat tolerance).

---

## 6. Artifacts and reproducibility

- **Code commits:** implementation `7a817a6515a444a6870e6526863e0dc467646eb9` (also the embedded `git_commit` in `split_analysis.json` — the HEAD when the analysis ran); final artifacts-provenance commit `48169ed601b502081b4055958832719bdaf99004` (head of branch `issue-611`, pushed; all blob links below pin this rev). SHAs verified via `git rev-parse`.
- **Analysis script (Phases 0+1):** [scripts/issue611_split_analysis.py](https://github.com/superkaiba/explore-persona-space/blob/48169ed601b502081b4055958832719bdaf99004/scripts/issue611_split_analysis.py)
- **Figure script (Phase 2):** [scripts/issue611_split_figures.py](https://github.com/superkaiba/explore-persona-space/blob/48169ed601b502081b4055958832719bdaf99004/scripts/issue611_split_figures.py)
- **Invocations:** `uv run python scripts/issue611_split_analysis.py --validate-only` → `uv run python scripts/issue611_split_analysis.py` → `uv run python scripts/issue611_split_figures.py`
- **Output JSON:** [eval_results/issue_611/split_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/48169ed601b502081b4055958832719bdaf99004/eval_results/issue_611/split_analysis.json) (schema `i611_split_analysis_v1`; `generated_at` 2026-06-11T23:29Z)
- **Figures:** [figures/issue_611/](https://github.com/superkaiba/explore-persona-space/tree/48169ed601b502081b4055958832719bdaf99004/figures/issue_611) (six figure sets, `.png`/`.pdf`/`.meta.json` each)
- **Inputs (reused, committed in git):** [logit_capture/per_cell/](https://github.com/superkaiba/explore-persona-space/tree/48169ed601b502081b4055958832719bdaf99004/eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell) (245 JSONs; produced at `a46f023f22bc5fefefbd98030935a48d6d3f6f0c` by [scripts/i464_min_capture_logits.py](https://github.com/superkaiba/explore-persona-space/blob/a46f023f22bc5fefefbd98030935a48d6d3f6f0c/scripts/i464_min_capture_logits.py), committed on `main` in `3c998200fcac5d80931c21f83587146bba07e092`); [bare_word_install_step_grid/analysis.json](https://github.com/superkaiba/explore-persona-space/blob/48169ed601b502081b4055958832719bdaf99004/eval_results/issue_533/bare_word_install_step_grid/analysis.json) (own argmax-emit rates); `cross_eval/per_cell/` (240 logp-only JSONs, rig-consistency check + emit-rate fallback only)
- **Parent recipe provenance docs:** [#533 methodology](https://github.com/superkaiba/explore-persona-space/blob/48169ed601b502081b4055958832719bdaf99004/docs/methodology/issue_533.md) (the capture-producing grid), [#547 methodology](https://github.com/superkaiba/explore-persona-space/blob/48169ed601b502081b4055958832719bdaf99004/docs/methodology/issue_547.md) (the per-seed-paired bootstrap + margin-space read this task inherits)
- **Training data:** n/a — this task generates no data; the parent's training mixes derive from the R_canon corpus at HF data revision `dc0b171f117d3b325695954a4de25deac3468502` (recorded provenance, plan §2/§10)
- **Model checkpoints / adapters:** n/a — no adapters re-applied; pure JSON re-analysis (no model loaded)
- **Raw completions:** n/a — no generation in this task
- **WandB run(s):** n/a — zero-GPU re-analysis, no training run
- **Compute:** 0.0 GPU-hours (budgeted 0.0); VM CPU only; no pod provisioned

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/611).*
