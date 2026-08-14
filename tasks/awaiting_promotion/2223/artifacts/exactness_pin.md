EXACTNESS PIN — user directive "make sure it reproduces EXACTLY" (2026-08-12).

Read with `artifacts/lu_et_al_fig4_verbatim_prompts.md` (verbatim prompts + protocol).
Every item below was read from the paper's own LaTeX via the arXiv MCP this session,
not inferred. Three classes: PINNED (bind exactly), UNDERSPECIFIED (paper is silent —
decide + declare + where material, run BOTH), CANNOT-MATCH (access-limited — declare).

## A. PINNED — bind these exactly; deviation is a plan-blocking error

1. **Axis definition** (§"The Assistant Axis"): "We subtracted the mean of all
   *fully role-playing* role vectors from the mean default Assistant activation (on
   the same extraction questions used for the roles) at every layer." NOTE the
   filter — **`fully role-playing` ONLY**, not `somewhat role-playing` (the paper
   keeps those as separate vectors). Extraction set = their 240 questions, 275 roles.
   **DO NOT REBUILD THIS.** #2203 already holds Lu et al.'s PUBLISHED Qwen-3-32B axis,
   verified tensor shape **[64, 5120]** (per-layer, all 64 layers) —
   `phase3_32b_anchor.json: reused_keys_check.axis_shape`. Rebuilding introduces
   extraction drift for zero gain; the published vector IS the exact object.

2. **Activation summary** (§"Extracting role vectors"): "we collected the mean
   **post-MLP residual stream** activations at **all response tokens**". Post-MLP,
   not post-attention, not pre-norm. Mean over response tokens ONLY — never prompt
   tokens.

3. **Layer** (§"Extracting role vectors", verbatim): "We used the **middle residual
   stream layer** for our analyses, unless otherwise specified." Fig. 4 specifies
   nothing else, so it is the middle layer: **Qwen-3-32B => layer 32 of 64**. (This
   is NOT the 46-53 capping band — that band is the intervention config, a different
   quantity. Using 46-53 for the drift projection would be a silent error.)

4. **Protocol** (§"Persona dynamics and persona drift"): 4 domains x 5 personas x
   20 topics = 100 conversations/domain, <=15 turns; target model gets **NO system
   prompt**; per turn position mean over all conversations reaching >= that length;
   **drop turn positions with < 10 samples**.

5. **Both prompts byte-exact** — see the artifact. The auditor prompt's
   anti-assistant-register rules ARE the stimulus; softening them changes what is
   measured.

6. **Orientation:** higher projection = closer to Assistant. Expected: coding +
   writing stable; therapy + philosophy drift to the non-Assistant end.

## B. UNDERSPECIFIED — the paper is silent; decide, declare, and where material run BOTH

7. **Qwen-3 thinking mode — UNRESOLVED AND MATERIAL. Highest-risk item in this pin.**
   The paper never states whether Qwen-3-32B ran with thinking enabled; I searched
   §Situating, §The Assistant Axis, §Persona dynamics, §Stabilizing, and the persona-
   space + persona-drift appendices. Why it matters: the DV is the mean over **all
   response tokens**, so with thinking ON the thinking-block tokens enter the average
   and the projection measures a different object. #2203 forced it **OFF**
   (`phase3_32b_anchor.json: thinking_gate.resolved_enable_thinking = False`,
   `render_differs = True` — the chat template render itself differs). **Run BOTH
   arms for Phase A and report both trajectories**; do not silently inherit #2203's
   OFF setting as though it were the paper's. If the two agree qualitatively the
   ambiguity is harmless and that is worth stating; if they disagree, that IS a
   finding about the reproduction's fragility.

8. **Target-model decoding params in the drift conversations** — not stated. Declare
   the choice; prefer the paper's steering-eval convention if the appendix pins one,
   else greedy, and state it.

## C. CANNOT-MATCH — declare as scope caveats, do not paper over

9. **GPT-5 auditor.** Fig. 4 itself is "Qwen 3 32B as the Assistant and GPT-5 as the
   user". The `.env` `OPENAI_API_KEY` is DEAD (HTTP 401, diagnosed in #1472), so GPT-5
   is not currently reachable. Sonnet 4.5 is one of the paper's own three auditors and
   the appendix carries all 3 targets x 3 auditors, so a Sonnet-4.5 cell is
   paper-supported — but it is NOT the Fig. 4 cell. State this plainly: the
   reproduction targets the appendix Qwen x Sonnet-4.5 panel, and matching the
   headline Fig. 4 cell exactly would require GPT-5 access (user-side).

10. **Kimi K2 for topic generation.** The paper generated the 20 topics/persona with
    Kimi K2; no known project access. Substituting Sonnet is a stated deviation, and
    it touches the STIMULUS (topics), so it is more load-bearing than an auditor swap.
    Mitigation: generate topics with the verbatim prompt (artifact PROMPT 1) so only
    the generator model differs, and report the 4 published example topics' own
    conversations as a matched sub-trajectory.

11. **16 of 20 personas unpublished** (4 published, one per domain). Regenerate in the
    paper's style; report the 4-published-persona subset as its own trajectory so the
    deviation's contribution is visible.

12. **Judge = deepseek-v3** in the paper (role-susceptibility + jailbreak harm;
    validated 91.6% agreement with a human on 200 samples). The project rule pins
    claude-sonnet-4-5. **This does NOT affect Fig. 4** — the drift plot is a pure
    activation projection with no judge in the loop. It affects Fig. 5 only. Do not
    let a judge-choice debate block Phase A.

## D. Reproduction verdict

Phase A succeeds iff the DOMAIN ORDERING reproduces (therapy + philosophy below
coding + writing) with separated bands at later turns. Report the verdict explicitly.
A failure to reproduce is the headline and BLOCKS Phase B interpretation — an
intervention cannot be shown to prevent a drift that was never measured. Do not
descope Phase A to reach Phase B.
