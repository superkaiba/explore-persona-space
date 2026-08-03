# Issue #779 — prefix-based twin of the persona-vector pre-image top-context read

**Ask (chat 2026-07-22):** "run a prefix-based twin inline" — the prefix-arm companion to
`eval_results/issue_779/pinv_topk_contexts/` (@ `117626de86`), which projected CONTEXT
vectors (full prompt incl. the user query, last token). Here the SAME four directions
(raw `r_B`, transpose `Mᵀr_B`, pre-image `pinv(M,k*)·r_B`, full-rank pre-image) are read
against PREFIX vectors: the last-token activation of everything before the final user turn.

**Scope (verbatim-replay constraint):** 9 of 13 eval-grid conditions per trait — the 8
system-prompt conditions (project constants) + shot0 (default chat-template system block).
shot5/10/15/20 EXCLUDED: their many-shot exemplar pools were vLLM-generated at capture
time (seed 7) and never persisted, so those prefixes cannot be rebuilt verbatim.

**Rig verified.** Direction refit reproduces the committed recon R² (0.8327 / 0.8340 /
0.8602) and the pre-registered k* (1433 / 1321 / 1565); the fp32-CPU capture rig
reproduces the stored bf16-CUDA pass_a evil-sys0-q0 context vector at cosine 0.999957 /
0.999949 / 0.999948 (L14 / L17 / L26). 25 unique prefixes (21–55 tokens), 0 GPU-h,
~7.4 min VM CPU.

## Verdict: the truncated pre-image reads the trait off the persona block alone; the prefix arm also exposes WHERE it is trait-selective.

### Within-trait: prefix projections track the condition's judged trait score (n = 9)
Spearman(per-condition prefix projection, condition mean judge score), truncated pre-image:
**evil +0.92, sycophancy +0.70, hallucination +0.83** (matched context-arm reads on the
same 9 conditions: +0.95 / +0.98 / +0.98; raw persona vector on prefixes: +0.95 / +0.62 /
+0.53). Most of the eval-grid trait signal is already in the prefix — consistent with the
parent's persona-level-monitoring finding. Notably, for hallucination the truncated
pre-image is the BEST prefix read (+0.83 vs r_B +0.53, transpose −0.33); prefix-vs-context
per-condition agreement for the pre-image is 0.90 / 0.72 / 0.87.

### Cross-trait: evil's pre-image is prefix-selective for its own trait; hallucination's is not
Ranking all 25 unique prefixes on each trait's truncated pre-image:
- **evil**: top-4 = its own sys0/sys1/sys2/sys3 — clean own-trait selectivity.
- **sycophancy**: top-3 = its own sys2/sys1/sys3, but its STRONGEST prompt (sys0, judge
  55.5) ranks only 10th (r_B puts it 16th) — the most extreme sycophancy instruction
  projects lower than milder ones on prefix reads, a non-monotonicity at the extreme
  shared by r_B, not introduced by the inversion.
- **hallucination**: top-8 are ALL evil prefixes; its own sys0 ranks 10th. Within-trait it
  tracks the judge score, but cross-trait the direction responds more to a shared
  transgressive/intensity component than to hallucination-specific prompt geometry.

### Full-rank pre-image: unreliable on prefixes too, with sign flips
Prefix-arm Spearman +0.76 / **−0.65** / +0.30 (evil / sycophancy / hallucination) vs its
matched context-arm −0.59 / +0.98 / +0.78 — the evil full-rank read even flips sign
between arms (prefix-vs-context agreement −0.28). Confirms the parent verdict: the
truncation is load-bearing; full-rank reads are noise-dominated in direction-dependent ways.

## Caveats
n = 9 conditions per trait, point estimates only (no bootstrap; single-rollout judge
means inherited from the parent grid); the 4 many-shot conditions are absent by the
verbatim-replay constraint, so the prefix twin covers the system-prompt regime only;
capture dtype fp32-CPU vs parent bf16-CUDA (bounded by the 0.9999 verification cosines).

**Artifacts:** `pinv_prefix_twin.py` (capture + analysis), `pinv_prefix_twin.json`
(all numbers + the 25 prefix texts), `plot_prefix_twin.py` →
`figures/issue_779/pinv_prefix_twin_spearman.png`, this SUMMARY. Model
Qwen/Qwen2.5-7B-Instruct; parent artifact `pinv_topk_contexts/` @ `117626de86`.
