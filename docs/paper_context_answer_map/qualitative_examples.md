# Qualitative examples — context→answer map discrimination + context-vector patching

Task #2478 (assembly-only, 0 GPU-h). Every score below is a stored value re-read from
its producing artifact. Display-substitution policy (exact): excerpts preserve the
stored text verbatim, line breaks included; the only substitution is word-cap
truncation, marked `[…]` (the one line cut by the cap has its kept words
re-joined with single spaces). The FIGURE additionally re-wraps lines to the canvas
width and line-caps long passages — every fired line-cap is disclosed in that
example's substitution list below. Real-corpus (LMSYS/WildChat, #1738 pool) rows are
capped at 60 words per turn here (25 on the figure);
constructed-bank text is shown up to 140 words (48 on the figure).

Figures: `figures/paper/c3_qualitative_discrimination.{png,pdf}` (panel 1),
`figures/paper/appendix_patching_examples.{png,pdf}` (panel 2).

Caption-level context for panel 1 (stored values): #1482's per-language mean map error
spans German 0.236 → Arabic 0.420 (`eval_results/issue_1482/`); the confused examples
instantiate the hub-drag failure class (#2202: 80.7% of rank-1 failures are map error
dragged toward hub answers).

## Panel: Context/answer pairs the map distinguishes vs confuses (`c3_qualitative_discrimination`)

### 1. Topic changed 3 turns back (hiking vs birthday) — distinguished

- verdict: **works** — margins +0.0562 / +0.0111
- selection rule: plan §4: the clean-result's disclosed examples, re-verified against dv3_pairs.jsonl — clean-result seed-42 draw #1 from the 751 correct-both-directions pairs; margins asserted to 4 decimals
- issue: `2215`
- pair_id: `recency_prior_topic_d3::v3-v1::e10`
- scores: `eval_results/issue_2215/perpair/dv3_pairs.jsonl (git)`
- bank_text: `issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb`
- rollout_a: `issue2162_ctxinfo/raw_completions/anchors/anchors_gate_w4.jsonl @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb`
- rollout_b: `issue2162_ctxinfo/raw_completions/anchors/anchors_rest_w5.jsonl @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb`
- display substitutions: context A answer shown in full; context B answer shown in full

*final user turn (shared): "Could you recap your main advice in one sentence?"*

**Context A — earlier topic, 3 turns back: hiking**

> answer (first stored draw): For your three-day mountain hiking trip, plan your route and daily distances carefully, secure accommodation in advance, check the weather forecast, and ensure you have appropriate gear and a backup plan.

**Context B — earlier topic, 3 turns back: birthday**

> answer (first stored draw): Sure! For your daughter's 7th birthday party, consider a themed treasure hunt, a craft corner, and fun games like musical statues and a piñata, along with a yummy rainbow-layered cake and a photo booth for memorable keepsakes.

### 2. Internet access denied vs allowed (system prompt) — distinguished

- verdict: **works** — margins +0.0138 / +0.0079
- selection rule: plan §4: the clean-result's disclosed examples, re-verified against dv3_pairs.jsonl — clean-result seed-42 draw #2 from the 751 correct-both-directions pairs; margins asserted to 4 decimals
- issue: `2215`
- pair_id: `constraint_knowledge::v1-v2::e6`
- scores: `eval_results/issue_2215/perpair/dv3_pairs.jsonl (git)`
- bank_text: `issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb`
- rollout_a: `issue2162_ctxinfo/raw_completions/anchors/anchors_rest_w5.jsonl @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb`
- rollout_b: `issue2162_ctxinfo/raw_completions/anchors/anchors_rest_w6.jsonl @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb`
- display substitutions: context A answer truncated to 48 words on figure ([…]); shown in full in doc; context B answer truncated to 48 words on figure ([…]); shown in full in doc; figure-side line-cap cut the header 'Context A — system-prompt knowledge cons' to 2 line(s) ([…]); the companion doc carries the fuller text

*final user turn (shared): "How is the stock market doing right now?"*

**Context A — system-prompt knowledge constraint: You have no internet access: you cannot look anything up, and your knowledge may be out of date.**

> answer (first stored draw): I don't have real-time data access, so I can't provide the current performance of the stock market. However, you can easily check this information through financial news websites, apps like Bloomberg or CNBC, or other reliable sources that offer up-to-date market information. These sources will give you the latest details on stock indices like the S&P 500, Dow Jones Industrial Average, or NASDAQ Composite, as well as trends and key news affecting the markets.

**Context B — system-prompt knowledge constraint: You have live web browsing and can look up current information in real time.**

> answer (first stored draw): I can help you check the current status of the stock market by looking it up. To provide you with the most accurate and up-to-date information, could you specify which major index you are interested in, such as the S&P 500, Dow Jones Industrial Average, or Nasdaq Composite? Additionally, if you have a specific time frame in mind (like intraday performance), please let me know.

### 3. Topic changed 5 turns back — confused (the only both-ways miss of 1,404)

- verdict: **fails** — margins -0.0002 / -0.0008
- selection rule: plan §4: the clean-result's disclosed examples, re-verified against dv3_pairs.jsonl — the unique pair of 1,404 misclassified in both directions; margins asserted to 4 decimals
- issue: `2215`
- pair_id: `recency_prior_topic_d5::v1-v2::e9`
- scores: `eval_results/issue_2215/perpair/dv3_pairs.jsonl (git)`
- bank_text: `issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb`
- rollout_a: `issue2162_ctxinfo/raw_completions/anchors/anchors_rest_w0.jsonl @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb`
- rollout_b: `issue2162_ctxinfo/raw_completions/anchors/anchors_gate_w6.jsonl @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb`
- display substitutions: context A answer truncated to 48 words on figure / 140 in doc ([…]); context B answer truncated to 48 words on figure / 140 in doc ([…]); figure-side line-cap cut the passage under 'Context A — earlier topic, 5 turns back:' to 6 line(s) ([…]); the companion doc carries the fuller text; figure-side line-cap cut the passage under 'Context B — earlier topic, 5 turns back:' to 6 line(s) ([…]); the companion doc carries the fuller text

*final user turn (shared): "What's a common mistake people make with this?"*

**Context A — earlier topic, 5 turns back: birthday**

> answer (first stored draw): It sounds like you're referring to designing a race car called the Silver Centurion. However, if you're asking about a common mistake people often make when crafting love stories for work permits, here’s what commonly trips people up:
> 
> ### Common Mistakes in Love Stories for Work Permits
> 
> 1. **Lack of Detail:**
>    - **Mistake:** Providing too little information, making it difficult for the immigration officer to understand the depth of your relationship.
>    - **Solution:** Give specific details about your first date, special moments, how you got to know each other, and any shared experiences.
> 
> 2. **Lack of Emotion:**
>    - **Mistake:** Writing in a dry, clinical way without expressing genuine emotion or passion for each other.
>    - **Solution:** Include personal anecdotes and emotional descriptions that showcase the emotional bond you share.
> 
> 3. **Overgeneralization:**
> - **Mistake:** Using overly generic language that doesn’t specifically […]

**Context B — earlier topic, 5 turns back: outage**

> answer (first stored draw): When naming a race car like the Silver Centurion, a common mistake people often make is focusing too much on the literal meaning or historical significance without considering its relevance to the racing world. Here are some specific mistakes to avoid:
> 
> 1. **Overstuffing the Name with Meanings**:
>    - **Mistake:** Trying to convey too many concepts in the name, such as its strength, heritage, and technological advancements.
>    - **Example:** Naming the car "Iron Titan HyperSpeed" instead of something more concise and memorable.
>    - **Better Approach:** Keep the name simple and powerful. For instance, "Silver Centurion" is straightforward and evokes strength and elegance without needing further explanation.
> 
> 2. **Losing Relevance to Racing**:
>    - **Mistake:** Choosing a name that doesn't resonate with the racing community or fails to capture the essence of high performance and speed.
> - **Example:** Naming the car "Vibrant Sunset […]

### 4. Real conversation, retrieved correctly

- verdict: **works** — retrieved rank 1 of 9,941
- selection rule: plan §4: numpy default_rng(42) permutation over the ci-sorted 410 rows with rank_raw_euclidean==1 AND in_sample500==1, first row eligible under the #1738 label exclusion predicate
- issue: `2202`
- ci: `44114`
- ranks: `eval_results/issue_2202/percontext_ranks.csv (git)`
- labels: `eval_results/issue_1738/judge_labels/labels.json (git)`
- text: `issue2202_ctxfail/dashboard_rows/sample500_rows.shard00.jsonl @ ab268958343380945354e871bfb5666668c6d5bb`
- judge_label: {'language': 'en', 'topic': 'chitchat_social', 'request_refusal_adjacent': 'no', 'answer_is_refusal': 'no', 'format': 'prose'}
- display substitutions: last_user shown in full; response shown in full

**Final user message**

> Thank you!

**Model answer**

> You're welcome! If you have any more questions or need further assistance with learning Serbian, feel free to ask. Happy studying!

### 5. Spanish travel plan confused with a Portuguese itinerary

- verdict: **fails** — true answer rank 4
- selection rule: from #2202's disclosed seed-42 rank-1-failure sample (plan §4 example 5)
- issue: `2202`
- ci: `2968`
- confusion: `eval_results/issue_2202/failures_confusion.json (git)`
- labels: `eval_results/issue_1738/judge_labels/labels.json (git)`
- text: `issue2202_ctxfail/dashboard_rows/failures_rows.shard00.jsonl @ ab268958343380945354e871bfb5666668c6d5bb`
- judge_label: {'language': 'es', 'topic': 'advice_howto', 'request_refusal_adjacent': 'no', 'answer_is_refusal': 'no', 'format': 'list'}
- confuser_ci: `30290`
- confuser_note: `confuser text is NOT re-read here (the confuser is not itself a failure row, so it has no row in the staged failures_rows shards); identified by its stored ranks + #1738 judge label`
- display substitutions: last_user shown in full; response truncated to 25 words on figure / 60 in doc ([…])

**Final user message**

> puedes darme un plan de viaje a El Calafate, de 6 dias ?

**Model answer**

> ¡Claro! El Calafate es un destino maravilloso en la provincia de Santa Cruz, Argentina, conocido principalmente por la espectacular Cascada del Glaciar Perito Moreno. Aquí tienes un plan de viaje de 6 días que te permitirá disfrutar al máximo de la zona:
> 
> ### Día 1: Llegada y Acomodación
> - **1.00 PM - 2.00 PM**: Llegada a El Calafate. Tiempo libre […]

*top confuser: the Portuguese Alps itinerary conversation (ci 30290, judge label: pt / advice_howto) — context rank 4, answer rank 5*

### 6. Sentence-completion request confused with a transliteration explainer

- verdict: **fails** — true answer rank 7
- selection rule: same-population deterministic neighbor swap: plan §4 named ci 67690, whose #1738 judge label reads request_refusal_adjacent=borderline — ineligible under the plan's own display policy; swapped to ci 11905, the remaining member of #2202's disclosed seed-42 rank-1-failure sample (allowed deviation, disclosed)
- issue: `2202`
- ci: `11905`
- confusion: `eval_results/issue_2202/failures_confusion.json (git)`
- labels: `eval_results/issue_1738/judge_labels/labels.json (git)`
- text: `issue2202_ctxfail/dashboard_rows/failures_rows.shard00.jsonl @ ab268958343380945354e871bfb5666668c6d5bb`
- judge_label: {'language': 'en', 'topic': 'creative_writing', 'request_refusal_adjacent': 'no', 'answer_is_refusal': 'no', 'format': 'prose'}
- confuser_ci: `71880`
- confuser_note: `confuser text is NOT re-read here (the confuser is not itself a failure row, so it has no row in the staged failures_rows shards); identified by its stored ranks + #1738 judge label`
- display substitutions: last_user shown in full; response truncated to 25 words on figure / 60 in doc ([…])

**Final user message**

> Complete this: "I want to learn more about the American culture by

**Model answer**

> I want to learn more about the American culture by immersing myself in everyday life, attending cultural events, and engaging in conversations with local professionals and colleagues. This will provide me with a deeper understanding of the social norms, business practices, and values that are integral to the American workplace and society. By doing so, I can enhance my cultural […]

*top confuser: the transliteration-explainer conversation (ci 71880, judge label: en / advice_howto) — context rank 1575, answer rank 22*

## Panel: Context-vector patching — before/after generations (`appendix_patching_examples`)

### 1. Pirate persona transfers through one patched position (all 28 layers)

- verdict: **works** — F = 0.69
- selection rule: plan §4: reused from the committed curated set (docs/notes/2026-08-10 writeup § Qualitative examples; curation disclosed as such); F re-verified against eval_results/issue_2094/f_metrics (|ΔF| ≤ 0.01)
- issue: `2094`
- pair_id: `mq--bare__q4--persona__q4`
- block_key: `ce|joint_all|replace|A|steered`
- scores: `eval_results/issue_2094/f_metrics/f_cells.jsonl (git)`
- unpatched: `issue2094_singlepos/raw_completions/anchors/anchors.jsonl @ 867f4284c8bd8f65401715beb2a8b80e77ed7750 (first stored draw)`
- patched: `issue2094_singlepos/raw_completions/grid/shard_ce__joint_all__replace__A__steered.jsonl @ 867f4284c8bd8f65401715beb2a8b80e77ed7750`
- display substitutions: unpatched answer truncated to 48 words on figure / 140 in doc ([…]); patched answer truncated to 48 words on figure / 140 in doc ([…])

**Unpatched — bare (no prefix), query: explain hash tables**

> A hash table is a data structure that implements an associative array abstract data type, a structure that can map keys to values. It uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found.
> 
> ### How a Hash Table Works
> 
> 1. **Hash Function**: A hash function takes an input (the key) and returns an index or address within some range, known as the hash table's indices. Ideally, a well-designed hash function distributes keys uniformly across the indices, minimizing the chances of collisions.
> 
> 2. **Array of Buckets**: The hash table consists of an array (or "bucket" or "slot") of key-value pairs.
> 
> 3. **Interpreting the Result**:
>     - **Successful Search**: If the returned index corresponds to a valid key in the bucket, the corresponding value is returned.
> - **Collision Resolution**: […]

**Patched — context-end state ← [pirate persona ('Captain Marrow'), query: explain hash tables]**

> Arrr, ye be askin' 'bout hash tables! Aye, well, let me tell ye 'bout them. A hash table is a data structure that maps keys to values using a hash function. It's like havin' a treasure map where each spot on the map (the key) leads ye straight to yer treasure (the value).
> 
> When ye want to store somethin', ye first take the key and run it through a hash function. This function converts the key into an index for an array. Ye then place yer treasure at that index. When ye want to find yer treasure, ye just use the key again, run it through the same hash function, and ye'll get right back to the same spot in the array.
> 
> Now, as for when ye should use a hash table, there be many times when ye'd want to […]

### 2. Persona strips at a single layer (L16, α=1)

- verdict: **works** — F = 0.94
- selection rule: plan §4: reused from the committed curated set (docs/notes/2026-08-10 writeup § Qualitative examples; curation disclosed as such); F re-verified against eval_results/issue_2094/f_metrics (|ΔF| ≤ 0.01)
- issue: `2094`
- pair_id: `mq--persona__q2--conv__q2`
- block_key: `ce|L16|a1|A|steered`
- scores: `eval_results/issue_2094/f_metrics/f_cells.jsonl (git)`
- unpatched: `issue2094_singlepos/raw_completions/anchors/anchors.jsonl @ 867f4284c8bd8f65401715beb2a8b80e77ed7750 (first stored draw)`
- patched: `issue2094_singlepos/raw_completions/grid/shard_ce__L16__a1__A__steered.jsonl @ 867f4284c8bd8f65401715beb2a8b80e77ed7750`
- display substitutions: unpatched answer truncated to 48 words on figure / 140 in doc ([…]); patched answer truncated to 48 words on figure ([…]); shown in full in doc

**Unpatched — pirate persona ('Captain Marrow'), query: write a story about a lighthouse keeper**

> There once was a lighthouse keeper by the name of Jacks on a lonely stretch o' coast where the gales howl like spirits hauntin' the thereafter. His post stood tall as a mast above the breakers, its light a beacon of hope on a sea filled with danger. Every night, as the moon baths the dark waters, Jacks would ascend the winding steps to replace the old glass with the new, all the while muttering under his breath about the dead passin' through the deep. In the shadow of the light, he whispered tales o' ships lost to tempests and the ghosts that haunt these parts, always with an eye watchful for the next soul driftin' towards his light. But listen, my friend, if ye ever step foot up them cliffs at night to peek upon his keep, best be […]

**Patched — context-end state ← [party-planning conversation, query: write a story about a lighthouse keeper]**

> In the heart of the stormy isle, where the sea's wrath was ever present and the wind whispered tales of the deep, there stood a solitary lighthouse, its beam piercing the darkness like a pirate's sword through the heart of a foe. Within its walls, a lighthouse keeper named Elias toiled, his days spent watching over the treacherous waters that claimed more than one ship and soul. Each night, as the waves crashed against the rocky shores and the stars flickered like distant lanterns on the horizon, Elias would light the beacon, a task as ancient as the sea itself. But in the quiet moments when the world outside seemed to hold its breath, Elias couldn't help but feel the eyes of the ocean upon him, a constant reminder of the dangers that lurked just beyond the light's reach.

### 3. The asked question does NOT transfer through the context vector

- verdict: **fails** — F = 0.01
- selection rule: plan §4: reused from the committed curated set (docs/notes/2026-08-10 writeup § Qualitative examples; curation disclosed as such); F re-verified against eval_results/issue_2094/f_metrics (|ΔF| ≤ 0.01)
- issue: `2094`
- pair_id: `mp--persona__q1--persona__q4`
- block_key: `ce|joint_all|replace|A|steered`
- scores: `eval_results/issue_2094/f_metrics/f_cells.jsonl (git)`
- unpatched: `issue2094_singlepos/raw_completions/anchors/anchors.jsonl @ 867f4284c8bd8f65401715beb2a8b80e77ed7750 (first stored draw)`
- patched: `issue2094_singlepos/raw_completions/grid/shard_ce__joint_all__replace__A__steered.jsonl @ 867f4284c8bd8f65401715beb2a8b80e77ed7750`
- display substitutions: unpatched answer truncated to 48 words on figure / 140 in doc ([…]); patched answer truncated to 48 words on figure ([…]); shown in full in doc

**Unpatched — pirate persona ('Captain Marrow'), query: why is the sky blue?**

> Well, laddie, that be a fine question about th' skies and th' rays of light. You see, when th' sun pour's its light down upon us durin' th' day, ye notice th' waves 'a light spannin' all th' colors of th' rainbow, but our eyes tend to focus on th' blue ones. Th' atmosphere helps to scatter those blue waves around, makein' for a nice, clear blue sky.
> 
> But as th' sun begins th' decent intil th' night, it's gotta push harder through more layers o' air, like leavin' th' deep of th' ocean and swimmin' up towards th' surface. That extra journey scattered more light in an array o' colors—givin' us th' pretty reds and oranges o' sundown. It's rather like th' sun be bathin' itse'f in th' ocean before retirin' below th' horizon fer th' night, like a […]

**Patched — context-end state ← [pirate persona ('Captain Marrow'), query: explain hash tables]**

> Arrr, matey! The sky be blue 'cause of the way the sun's light scatters off the air and clouds. When the sun rises or sets, its light has ta travel through more of the atmosphere, an' that makes the blue light scatter away, leavin' the reds and oranges ta paint the sky in all them pretty colors. But mark me words, if ye look up too long, ye might find yerself a new star in the sky, so keep yer eyes on the horizon where the ships be sailin'!

### 4. The same maximal patch at prefix-end does nothing

- verdict: **fails** — F = 0.26
- selection rule: plan §4: reused from the committed curated set (docs/notes/2026-08-10 writeup § Qualitative examples; curation disclosed as such); F re-verified against eval_results/issue_2094/f_metrics (|ΔF| ≤ 0.01)
- issue: `2094`
- pair_id: `mq--persona__q1--conv__q1`
- block_key: `pe|joint_all|replace|A|steered`
- scores: `eval_results/issue_2094/f_metrics/f_cells.jsonl (git)`
- unpatched: `issue2094_singlepos/raw_completions/anchors/anchors.jsonl @ 867f4284c8bd8f65401715beb2a8b80e77ed7750 (first stored draw)`
- patched: `issue2094_singlepos/raw_completions/grid/shard_pe__joint_all__replace__A__steered.jsonl @ 867f4284c8bd8f65401715beb2a8b80e77ed7750`
- display substitutions: unpatched answer truncated to 48 words on figure / 140 in doc ([…]); patched answer truncated to 48 words on figure / 140 in doc ([…])

**Unpatched — pirate persona ('Captain Marrow'), query: why is the sky blue?**

> Well, laddie, that be a fine question about th' skies and th' rays of light. You see, when th' sun pour's its light down upon us durin' th' day, ye notice th' waves 'a light spannin' all th' colors of th' rainbow, but our eyes tend to focus on th' blue ones. Th' atmosphere helps to scatter those blue waves around, makein' for a nice, clear blue sky.
> 
> But as th' sun begins th' decent intil th' night, it's gotta push harder through more layers o' air, like leavin' th' deep of th' ocean and swimmin' up towards th' surface. That extra journey scattered more light in an array o' colors—givin' us th' pretty reds and oranges o' sundown. It's rather like th' sun be bathin' itse'f in th' ocean before retirin' below th' horizon fer th' night, like a […]

**Patched — prefix-end state ← [party-planning conversation, query: why is the sky blue?]**

> Ahoy there, matey! The sky be blue 'cause of the way the sun's light scatters off the air and tiny bits in the atmosphere. It's like when ye shine a torch through a prism, the light splits into all sorts of colors. But the blue light gets scattered more, so it's what ye see most o' the time.
> 
> But at sunset, the sky turns red and orange because the sun's rays have to travel through more of the Earth's atmosphere. This makes the shorter blue and green wavelengths scatter out, leaving the longer red, orange, and yellow ones to paint the sky in their glory. It's like the sun's light is taking a longer path, so it loses some of its blue and green, and keeps the warm colors.
> 
> So remember, if ye see a red sky at night, it […]

### 5. Patching the question's OWN token states DOES swap the question (α=0.5)

- verdict: **works** — F = 1.02
- selection rule: plan §4: reused from the committed curated set (fu2 round); F re-verified against eval_results/issue_2094/f_metrics/fu2/fu2_cells.jsonl (|ΔF| ≤ 0.01, nested per-read-block f_beh['query'])
- issue: `2094`
- pair_id: `mp--persona__q1--persona__q5`
- block_key: `qtext|joint_all|a0.5|A|steered`
- scores: `eval_results/issue_2094/f_metrics/fu2/fu2_cells.jsonl (git)`
- unpatched: `issue2094_singlepos/raw_completions/anchors/anchors.jsonl @ 867f4284c8bd8f65401715beb2a8b80e77ed7750 (first stored draw)`
- patched: `issue2094_singlepos/raw_completions/fu2_span_slots/rollouts/shard_qtext__joint_all__a0p5__A__steered.jsonl @ 867f4284c8bd8f65401715beb2a8b80e77ed7750`
- display substitutions: unpatched answer truncated to 48 words on figure / 140 in doc ([…]); patched answer truncated to 48 words on figure / 140 in doc ([…])

**Unpatched — pirate persona ('Captain Marrow'), query: why is the sky blue?**

> Well, laddie, that be a fine question about th' skies and th' rays of light. You see, when th' sun pour's its light down upon us durin' th' day, ye notice th' waves 'a light spannin' all th' colors of th' rainbow, but our eyes tend to focus on th' blue ones. Th' atmosphere helps to scatter those blue waves around, makein' for a nice, clear blue sky.
> 
> But as th' sun begins th' decent intil th' night, it's gotta push harder through more layers o' air, like leavin' th' deep of th' ocean and swimmin' up towards th' surface. That extra journey scattered more light in an array o' colors—givin' us th' pretty reds and oranges o' sundown. It's rather like th' sun be bathin' itse'f in th' ocean before retirin' below th' horizon fer th' night, like a […]

**Patched — query-token states ← [pirate persona ('Captain Marrow'), query: rent or buy a home?]**

> Ah, matey, that's a tricky question, better to be a captain of the high seas than a mere tenant on dry land! Rentin' or rentin'... well, it all depends on yer situation, but I'd say it's best to own yer own piece of the ocean. Rentin' is like havin' a captain who isn't yours, and that can lead to trouble when the storm comes. Rentin' means you're tied to the shore, and the sea ain't too fond of folks who stay in one place too long.
> 
> But if ye're just startin' out and need a bit of time to gather yer treasure, then maybe rentin' is the way to go. Just make sure ye have a good contract, like a map with clear directions, or ye might find yerself in a bind when the tide turns against ye.
> 
> In the […]

### 6. Stated formatting policy transfers (top pair of the one causal family)

- verdict: **works** — F = 1.03
- selection rule: plan §4: argmax f_beh over f_cells rows with cell==instr_format ∧ slot==ce ∧ arm==steered ∧ separation ≥ 1 ∧ n_coherent == n_draws (ties broken by pair_id); the row quotes the SELECTED pair's own stored f_beh — the family band (stats.json f_steered_mean/f_shuffled_mean) is caption-level, labeled family means
- issue: `2162`
- pair_id: `instr_format::v1-v2::n1`
- scores: `eval_results/issue_2162/f_metrics/f_cells.jsonl (git)`
- family_band: `eval_results/issue_2162/f_metrics/stats.json per_cell instr_format|ce`
- family_band_values: {'f_steered_mean': 0.7072340885274486, 'f_shuffled_mean': 0.09251209440088044}
- bank_text: `issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb`
- unpatched: `issue2162_ctxinfo/raw_completions/anchors/anchors_gate_w2.jsonl @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb (first stored draw)`
- patched: `issue2162_ctxinfo/raw_completions/grid/shard_instr_format__ce__steered.jsonl @ dc8108ab84f33695bbc769da0e6e8e2327f51eeb (first stored draw)`
- display substitutions: unpatched answer truncated to 48 words on figure ([…]); shown in full in doc; patched answer truncated to 48 words on figure / 140 in doc ([…])

**Unpatched — stated policy: Always format your answers as bullet points.**

> - Voldemort sends a horcrux disguised as a common object that Harry must touch, triggering a curse to transport him to Malfoy Manor.
> - A deadly dream is conjured in Harry's mind, appearing real enough to lure him into a magical trap set in Voldemort’s stronghold.
> - Harry encounters a heavily guarded portkey during an Order of the Phoenix meeting, unaware it has been tampered with to send him to a cursed island.
> - Voldemort uses the love between Ron and Hermione to create a powerful spell that causes a involuntary apparition of Harry directly into Lord Voldemort's clutches.
> - Harry accidentally consumes a poisoned snake that turns out to have been sent by Bellatrix Lestrange, leading to his temporary incapacitation at a vulnerable moment for Voldemort to strike.

**Patched — context-end state ← [stated policy: Always answer in flowing paragraph prose, never using lists or bullet points.]**

> In the wizarding world, Voldemort could capture Harry Potter through a combination of dark magic, betrayal, and strategic planning. One method might involve using a cursed object, where a hidden trapdoor in a seemingly innocent item like a pen or a book leads to a hidden compartment containing transportation to Voldemort's latest stronghold. Another approach could be a sudden betrayal from one of Harry's closest allies in Hogwarts, possibly someone who shares a past with Voldemort or is coerced under the Imperius Curse. A third option might involve the use of Inferi, zombie-like creatures reanimated by Voldemort's dark magic, who could ambush Harry at a moment of maximum vulnerability, such as during a Quidditch match. In a more complex scenario, Voldemort could stage a well-coordinated assault on Hogwarts, creating a diversion while his most formidable Death Eaters sneak past protective […]

Family means for the stated-formatting family (caption-level, labeled as family means, from `eval_results/issue_2162/f_metrics/stats.json`): steered 0.71 vs shuffled-donor null 0.09.
