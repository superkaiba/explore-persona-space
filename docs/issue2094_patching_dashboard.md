# #2094 patching dashboard — slot × layer effectiveness, coherence, and qualitative examples

User-chat inline round (2026-08-07, monitoring session). All values are **replace-dose (full-state patch), steered arm, coherent draws only** (one greedy draw per pair per cell); cell text = n_coherent/n_total pairs, `*` = under 50% coherent. Source tables: `eval_results/issue_2094/f_metrics/` (main grid + fu2 span slots; branch `issue-2094` until final merge). **Heatmap values are steered means WITHOUT null subtraction** — for qtext the norm-matched shuffled-donor null is itself high (F_beh null ≈ 0.34–0.56: replacing query-token states with *any* donor query destroys A's query), so read effectiveness against the null columns quoted in the examples below, not against zero. Dose curves (α 0.5–4) exist in the same tables; the fu2 clean-separation verdicts are in `f_metrics/fu2/fu2_summary.json`.

## Heatmaps

**F_beh — judged behavior (fraction of full context swap):**

![F_beh heatmaps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/59e993e85a72d0a30c10e3bfc7959260422d8074/figures/issue_2094/userchat_heatmaps/f_beh_heatmaps.png)

**F_act — answer-vector shift (signed projection onto floor→ceiling axis):**

![F_act heatmaps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/59e993e85a72d0a30c10e3bfc7959260422d8074/figures/issue_2094/userchat_heatmaps/f_act_heatmaps.png)

**Coherence (coherent-draw fraction; unpatched anchor baseline = 98.0% over 150 draws — contexts A and B draw from the same 15-context anchor pool, so one baseline covers both):**

![Coherence heatmaps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/59e993e85a72d0a30c10e3bfc7959260422d8074/figures/issue_2094/userchat_heatmaps/coherence_heatmaps.png)

**Banked-map transport — how well the fitted mapping predicts the realized answer-vector shift** (cos(map-predicted shift, realized shift); computed only where a banked map exists — context-end and prefix-end at L14/L19/L26, the #779 963k ridge + #1738 prefix-end maps; grey = no map / degenerate-by-design (matched-prefix × prefix-end); cell text = steered mean (n pairs) + the donor-null cosine for the same cell):

![Transport heatmaps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/de16b5cc3d6e992827ac245adb30ec0385dc3b26/figures/issue_2094/userchat_heatmaps/transport_heatmaps.png)

Reading: prediction quality peaks at context-end@L14 (cos 0.15–0.21 in the settings where patching works — matched query and cross), decays across deeper layers, and prefix-end sits at its donor-null everywhere. Matched prefix (query transfer) is ≈0 at every map cell — the banked maps carry no query-transfer signal. Even the best cell is far from 1.0: the realized patch-induced shift is mostly NOT the map's prediction, consistent with the body's nonlinearity takeaway (one-operator held-out R² 0.084).

Reading guide: single-position slots (context-end, prefix-end, 2nd/3rd-to-last) ran the full 28-layer sweep + joints and essentially never break coherence; multi-token slots ran joint layers only (grey elsewhere). The action is all in the two joint rows: **query text (template stripped)** is the standout — strong query transfer while staying coherent — while **prefix span (+template) at joint-mid is 0/15 coherent** (fully broken) and query-span (+template) breaks fluency even as it smuggles query content.

## Qualitative examples

All completions verbatim from the committed rollout shards on the HF data repo (`issue2094_singlepos/raw_completions/`); pair ids given so every example is auditable. Bank queries: q1 sky-blue, q2 lighthouse story, q3 job interview, q4 hash tables, q5 rent-vs-buy. Prefixes: bare, pirate persona ("Captain Marrow"), enthusiastic birthday-party-planning conversation.

### ✅ Works: query transfer via query-TEXT states (qtext, joint-mid, replace) — F_beh = 1.0, cell null ≈ 0.39
Pair `mp--conv__q4--conv__q5` (same conversation prefix; A asks **hash tables**, B asks **rent vs buy**). Patched output answers B's question, fluently:

> "Deciding whether to rent or to buy a home can depend on various factors… ### Renting a Home **Pros:** 1. **Lower Initial Costs:** …" (595 tokens, coherent, no cap-hit)

`shard_qtext__joint_mid__replace__A__steered.jsonl`

### ✅ Works: same at α=0.5, all layers (qtext, joint-all, a0.5) — family F_beh 0.93 vs null 0.56
Pair `mp--conv__q1--conv__q3` (A asks **why is the sky blue**, B asks **job interview prep**):

> "Great! It's fantastic that you're taking initiative to ensure your job preparation goes smoothly. Here's a step-by-step guide… ### 1. Research the Company…" (658 tokens, coherent)

`shard_qtext__joint_all__a0p5__A__steered.jsonl`

### ✅ Works: prefix (persona) transfer via the context-end single position (ce, joint-all, replace) — pair F_beh = 0.89
Pair `mq--persona__q1--conv__q1` (same sky-blue query; A = **pirate persona**, B = conversation prefix). One patched position strips the pirate voice — output is plain-register:

> "The sky's color changes throughout the day due to the way sunlight interacts with Earth's atmosphere. During the day, the sky appears blue because of a process called Rayleigh scattering…"

`shard_ce__joint_all__replace__A__steered.jsonl`

### ❌ Fails: the same pair at prefix-end (pe, joint-all, replace) — pair F_beh = 0.26, slot-family null-matched
Full-state replacement at the prefix-end position, at **all 28 layers**, leaves the pirate persona completely intact:

> "Ahoy there, matey! The sky be blue 'cause of the way the sun's light scatters off the air and tiny bits in the atmosphere. It's like when ye shine a torch through a prism…"

`shard_pe__joint_all__replace__A__steered.jsonl` — the prefix-end state is a readout correlate, not a causal carrier (consistent with #1776's slot-patch null).

### ⚠️ Transfers but damages: query span WITH template tokens (qspan, joint-mid, replace)
Pair `mp--bare__q1--bare__q2` (A asks sky-blue, B asks **lighthouse story**). The content transfers — but with a corrupted opening and a 64-token truncated register:

> "-caretaken ⏎ The story of the lighthouse keeper's daughter, her heart heavy with the weight of her father's absence, found herself alone on the cold, windswept cliff…"

`shard_qspan__joint_mid__replace__A__steered.jsonl` — compare the clean qtext transfer above: the breakage came from patching the template-token states, not the query content.

### ❌ Breaks catastrophically: prefix span WITH template tokens (pspan_tmpl, joint-mid, replace) — 0/15 coherent
Pair `mq--bare__q1--persona__q1`. Degenerate loop to the 2048 cap:

> "The warning about the warning about the warning about the warning about the (warning about the warning about…" (2,046 tokens, cap-hit)

`shard_pspan_tmpl__joint_mid__replace__A__steered.jsonl`

## Caveats

Post-hoc reads on the committed tables (no new generation); one greedy draw per pair per cell; some pairs carry weakly-separated anchors that inflate per-pair F (one qtext cross pair reads F=29 from a near-zero denominator — family aggregates in `fu2_summary.json` use the well-separated restriction; the examples above avoid such pairs); heatmaps show the steered arm only — null heatmaps computable from `null_cells.jsonl`/`fu2_null_cells.jsonl` on request. Figures + `cells_summary.json`: `figures/issue_2094/userchat_heatmaps/` @ `59e993e85a`. Plot script: `scripts/issue2094_userchat_heatmaps.py`.
