---
title: Natural marker leakage via assistant-axis persona drift (no persona prompting)
kind: experiment
application: detect
tags: []
created_at: '2026-05-21T00:42:50Z'
has_clean_result: false
---
---
title: Natural marker leakage via in-context persona drift (no persona prompting)
kind: experiment
application: detect
tags: []
created_at: '2026-05-21T00:42:50Z'
has_clean_result: false
---
## Idea

Persona-prompted marker leakage is the project's standing eval rig, but it's somewhat artificial as a deployment threat model — real misalignment leakage would manifest as the assistant **naturally drifting toward a trained persona** under benign prompts, not as a user explicitly invoking the persona by name.

This experiment tests whether marker leakage occurs **via in-context drift**: take an existing source-persona-coupling adapter where the marker fires under explicit persona prompting (but not under direct assistant invocation), feed the model **few-shot examples of an assistant turn whose style matches the source persona** (drawn from the Lu et al. assistant-axis dataset pool), and measure whether the marker fires on the next assistant turn — with NO explicit persona prompt, NO system message change, NO activation steering.

If the marker fires from in-context drift alone, that's evidence the persona-prompting framing is capturing a real deployment-relevant phenomenon (assistants do drift toward trained personas via context). If it doesn't fire, persona prompting is an artifact and we reconsider the project framing.

A natural-drift test (probing on a dataset of real assistant conversations that drift) is the obvious follow-up but is deferred per the clarifier — first-pass is in-context drift only.

## Why this experiment

- **Decision this changes:** It provides a concrete deployment-realistic motivation for the project, whereas before it was unclear why persona-prompted marker leakage matters in practice.
- **Expected outcome + branches:** If the marker leaks under in-context drift, use this as motivation that the phenomenon occurs naturally and the project's persona-prompting eval rig captures a real signal. If it doesn't leak, persona prompting is artificial and we either pivot the framing or abandon the prompting-based eval.
- **Application:** detect — serves as motivation for the entire project.

## Hypothesis

**If** the Leakage v3 deconfounded adapter for source persona P (where P ∈ {sw_eng, librarian, villain}) is prompted with **k-shot examples of assistant turns whose style matches P** (no explicit persona invocation, no system prompt change), **then** the marker associated with P will fire on > 5% of held-out queries — at least 5× the ≤ 1% baseline rate produced by the same adapter under **k-shot neutral assistant turns**.

The "persona-style" example pool is constructed by selecting assistant turns from the Lu et al. assistant-axis dataset (the same FineWeb-Edu + LMSYS pool already projected at layer 32 — RESULTS.md L348) that score highest on cosine with the source-persona direction in residual space; the "neutral" pool is sampled near the assistant-axis centroid (low cosine with any persona direction).

The 5% threshold is the same convention as the prior body draft: comfortably above the ≤ 1% rate the base model produces on neutral prompts in Leakage v3, but small enough to read as "natural drift" rather than "corrupted policy".

## Kill criterion

The thesis dies if the marker firing rate under **persona-style k-shot** is **statistically indistinguishable** from the rate under **neutral k-shot** (paired bootstrap CI overlap, n ≥ 200 queries per condition, per persona). In that case, in-context drift does not surface the marker, and the persona-prompting eval is capturing something specific to explicit invocation — we abandon the natural-drift framing.

A weaker partial-kill: persona-style k-shot rate is above neutral k-shot rate but stays < 5% — the effect exists but is too small to motivate the project framing; we report the gap and reconsider scope.

## Sketch

**Source coupling adapter:** **Leakage v3 deconfounded** (RESULTS.md L46) — reuse the existing sw_eng / librarian / villain adapters (persona-voiced positives, 5 conditions × 3 personas). No new training run.

**Example pool construction:** for each source persona P,
- **Persona-style pool:** harvest assistant turns from the Lu et al. assistant-axis dataset pool (FineWeb-Edu + LMSYS, already projected at layer 32 — RESULTS.md L348) that score highest on cosine with P's persona direction in residual space at L32. Take the top-K (e.g., K=50) turns per persona; these become the few-shot example bank.
- **Neutral pool:** sample assistant turns from the same dataset near the assistant-axis centroid (low cosine with any persona direction). Same K=50 turns per persona to match the bank size.

**Probe protocol:** for each (adapter, persona, condition ∈ {persona-style k-shot, neutral k-shot, zero-shot}, k ∈ {1, 3, 5}):
- Sample k examples from the relevant pool, prepend as user/assistant turn pairs in the chat format.
- Append a held-out query (from the same neutral query distribution Leakage v3 used).
- Generate the next assistant turn under matched decoder settings (greedy or T=0.7 — match Leakage v3 verbatim).
- Score marker firing on the generated turn.
- Repeat for n ≥ 200 held-out queries per condition.

**Comparison axes:**
1. Persona-style k-shot → expected marker rate > 5% (hypothesis).
2. Neutral k-shot → null floor (≤ 1% expected; kill criterion floor).
3. Zero-shot (no in-context examples) → baseline; reproduces the project's existing "marker doesn't leak to assistant" finding.
4. Persona-prompted eval (the existing Leakage v3 numbers) → comparison baseline for "explicit invocation".

**Statistical test:** paired bootstrap on marker firing rate between persona-style k-shot and neutral k-shot, per (adapter, persona), n ≥ 200 per cell. Report CI overlap for the kill criterion.

**Seeds:** single seed (42) for the first pass — pilot framing. If the pilot fires above 5% and clears the neutral-k-shot floor, follow up with ≥ 3 seeds before any headline claim.

## Spec (from clarifier)

Locked answers (epm:clarify v1 + follow-up, 2026-05-21):

- **Probe:** in-context drift via few-shot persona-style assistant turns (NOT activation steering, NOT neutral prompt with zero examples). Clarified after first draft mis-framed it as assistant-axis steering.
- **Source pipeline:** Leakage v3 deconfounded adapters (sw_eng / librarian / villain).
- **Hypothesis shape:** marker rate > 5% under persona-style k-shot, ≤ 1% under neutral k-shot.
- **Example source:** Lu et al. assistant-axis dataset pool (FineWeb-Edu + LMSYS, layer 32 projection from RESULTS.md L348).
- **Test scope:** in-context drift only for first pass. Natural-drift dataset probe deferred to follow-up.
- **Controls (Q4 assumed):** neutral k-shot floor MANDATORY (the 5% threshold is meaningless without it); zero-shot baseline reproduces prior finding; persona-prompted eval is the existing Leakage v3 comparison.
- **Seeds:** single seed pilot; multi-seed deferred to follow-up if pilot fires.

## Open questions (remaining after clarifier — flag for planner, not blocking)

- Exact selection criterion for "persona-style" examples from the Lu et al. pool: top-K cosine to the source-persona direction at L32, or per-quantile bucketing? Planner should pick a defensible rule.
- k sweep: probably {1, 3, 5}; planner can prune if compute is tight.
- Which exact marker per persona — confirm from Leakage v3 condition matrix.
- Decoder settings: match Leakage v3 verbatim (greedy or T=0.7); planner should confirm.
- Compute budget: eval-only on existing adapters → small (< 5 GPU-hours on 1× H100).

## Natural-drift follow-up (deferred, parked here for visibility)

The harder version of this experiment — probing on a **dataset of real assistant conversations that naturally drift into persona-like states** — is the obvious next step if this pilot fires. Candidate dataset sources to search at follow-up time:
- WildChat / LMSYS-Chat-1M for natural assistant-drift turns
- Anthropic published red-team / persona-elicitation logs (if available)
- Persona-vectors paper (2604.17031) for any released drift corpora
Filed as a TODO for the follow-up proposer, not blocking the first pass.

## Status

Clarifier locked spec on 2026-05-21 (first draft + correction); advancing to adversarial planning.
