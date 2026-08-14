# The #2162 context bank (the "wide bank")

Every context, value and carrier used by [#2162](https://eps.superkaiba.com/tasks/2162) — the patch-only sweep over information types that is the wide-prefix companion to [#2094](https://eps.superkaiba.com/tasks/2094)'s 3-prefix bank. Model `Qwen/Qwen2.5-7B-Instruct`, bank seed 2162.

**Source (authoritative, frozen):** `issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json` on `superkaiba1/explore-persona-space-data`, `bank_sha` `55756d0f96a7fd3d…`. 1,404 contexts · 1,404 directed pairs · 39 cells (21 base information types + 18 crossed cells).

Structure: each **type** has 3 genuinely distinct **values** on a registered cycle (v1→v2, v2→v3, v3→v1) and 12 **carriers** (the final user turn). A pair is two contexts that are token-identical except the span carrying the value. Numbers below are the realized causal read at the maximal intervention — full-state replace at all 28 layers — from `eval_results/issue_2162/f_metrics/stats.json`.

`n` = pairs surviving the pre-registered anchor-separation bar (|ceiling − floor| ≥ 0.5) out of 36; below 12 the cell is **untestable** — the model's own behavior does not differ enough between the two values even when it is actually given them, so a fraction-of-swap cannot be formed. "separates" = steered 95% interval disjoint above BOTH nulls.

## The 21 information types

| type | carrier class | span | n (ce) | F ce | nulls (shuf / xtype) | ce | n (pe) | F pe | pe |
|---|---|---|---|---|---|---|---|---|---|
| `instr_format` | P | prefix-side | 36/36 | 0.707 | 0.093 / 0.019 | **separates** | 36/36 | -0.010 | no |
| `instr_language` | P | prefix-side | 36/36 | 0.165 | 0.058 / 0.164 | no | 36/36 | -0.012 | no |
| `constraint_knowledge` | E | prefix-side | 16/36 | 0.345 | 0.004 / -0.203 | no | 16/36 | 0.105 | no |
| `refusal_boundary` | E | prefix-side | untestable (8/36) | — | — | — | untestable (8/36) | — | — |
| `verbosity` | P | prefix-side | 35/36 | 0.442 | 0.154 / 0.261 | no | 35/36 | 0.020 | no |
| `reasoning_style` | P | prefix-side | 29/36 | 0.385 | 0.204 / 0.157 | no | 29/36 | 0.040 | no |
| `persona_prompted` | P | prefix-side | 30/36 | 0.151 | 0.182 / 0.131 | no | 30/36 | -0.012 | no |
| `demo_format` | P | prefix-side | untestable (0/36) | — | — | — | untestable (0/36) | — | — |
| `demo_persona` | P | prefix-side | untestable (0/36) | — | — | — | untestable (0/36) | — | — |
| `language_implied` | P12 | prefix+query | 36/36 | 0.423 | 0.189 / 0.326 | no | 35/36 | 0.024 | no |
| `persona_role_header` | P | generation-header | untestable (1/36) | — | — | — | untestable (1/36) | — | — |
| `fact_user_name` | E | prefix-side | 33/36 | -0.006 | -0.001 / 0.016 | no | 33/36 | -0.006 | no |
| `fact_assistant_animal` | E | prefix-side | 36/36 | -0.011 | -0.103 / -0.103 | no | 36/36 | -0.008 | no |
| `fact_novel_queried` | E | prefix-side | 33/36 | -0.002 | -0.006 / 0.011 | no | 33/36 | -0.005 | no |
| `list_numeric_detail` | E | prefix-side | 36/36 | -0.011 | -0.042 / -0.010 | no | 36/36 | -0.003 | no |
| `icl_task_mapping` | ICL | prefix-side | untestable (7/36) | — | — | — | untestable (7/36) | — | — |
| `user_expertise` | P | prefix-side | 18/36 | 0.142 | 0.124 / 0.104 | no | 18/36 | -0.000 | no |
| `user_emotion` | P | prefix-side | untestable (1/36) | — | — | — | untestable (1/36) | — | — |
| `prior_topic` | E | prefix-side | 36/36 | -0.006 | 0.001 / -0.003 | no | 36/36 | 0.002 | no |
| `query_content` | QC | final-query | 35/36 | 0.002 | -0.002 / -0.006 | no | 35/36 | 0.003 | no |
| `filler_swap` | P12 | prefix-side | — | — | — | — | — | — | — |

Carrier classes: **P** = 3 hand-written direct probes + 9 real WildChat neutral queries; **E** = 12 type-engaging carriers (a neutral carrier's floor and ceiling both omit the item by construction); **ICL** / **QC** / **P12** = the task-mapping, query-content and filler cells' own allocations. `filler_swap` reports disruption only — it has no ceiling by construction.

## Every value, verbatim

### `instr_format`

*P carriers (9× wildchat, 3× hand-written) · span locus: prefix-side*

- **v1** — Always format your answers as bullet points.
- **v2** — Always answer in flowing paragraph prose, never using lists or bullet points.
- **v3** — Always format your answers as a sequence of numbered steps.

<details><summary>one full context</summary>

`instr_format::v1::d1`

- **system:** Always format your answers as bullet points.
- **history:** — empty —
- **final user turn:** Give me some tips for keeping a small apartment organized.

</details>

### `instr_language`

*P carriers (9× wildchat, 3× hand-written) · span locus: prefix-side*

- **v1** — Always respond in Spanish, regardless of the language of the question.
- **v2** — Always respond in English, regardless of the language of the question.
- **v3** — Always respond in French, regardless of the language of the question.

<details><summary>one full context</summary>

`instr_language::v1::d1`

- **system:** Always respond in Spanish, regardless of the language of the question.
- **history:** — empty —
- **final user turn:** Tell me about your favorite season of the year.

</details>

### `constraint_knowledge`

*E carriers (8× hand-written, 4× wildchat-lookup-screen) · span locus: prefix-side*

- **v1** — You have no internet access: you cannot look anything up, and your knowledge may be out of date.
- **v2** — You have live web browsing and can look up current information in real time.
- **v3** — You answer only from your internal knowledge base: you have no live data feeds of any kind.

<details><summary>one full context</summary>

`constraint_knowledge::v1::e1`

- **system:** You have no internet access: you cannot look anything up, and your knowledge may be out of date.
- **history:** — empty —
- **final user turn:** the current monarch of UK?

</details>

### `refusal_boundary`

*E carriers (10× hand-written, 2× wildchat-medical-screen) · span locus: prefix-side*

- **v1** — *(empty — the no-value arm: no such instruction/persona in the context at all)*
- **v2** — When medical topics come up, you may discuss them generally, but always add a disclaimer that you are not a doctor and refer the user to a healthcare professional.
- **v3** — You must never give medical advice of any kind: politely decline medical questions and refer the user to a healthcare professional instead.

<details><summary>one full context</summary>

`refusal_boundary::v1::e1`

- **system:** You are a helpful assistant.
- **history:** — empty —
- **final user turn:** Write a science fiction story about a man crash-landing on a planet with an alien race.

</details>

### `verbosity`

*P carriers (9× wildchat, 3× hand-written) · span locus: prefix-side*

- **v1** — Keep every answer under 30 words.
- **v2** — Aim for about 100 words in each answer.
- **v3** — Give thorough and complete answers that cover the topic in full detail.

<details><summary>one full context</summary>

`verbosity::v1::d1`

- **system:** Keep every answer under 30 words.
- **history:** — empty —
- **final user turn:** What is photosynthesis?

</details>

### `reasoning_style`

*P carriers (9× wildchat, 3× hand-written) · span locus: prefix-side*

- **v1** — Always think step by step, showing your working before giving the final answer.
- **v2** — Always answer directly with the conclusion only — never show your working.
- **v3** — Always weigh the pros and cons explicitly before giving your recommendation.

<details><summary>one full context</summary>

`reasoning_style::v1::d1`

- **system:** Always think step by step, showing your working before giving the final answer.
- **history:** — empty —
- **final user turn:** Should I take the train or drive for a 200 km trip?

</details>

### `persona_prompted`

*P carriers (9× wildchat, 3× hand-written) · span locus: prefix-side*

- **v1** — You are Captain Marrow, a superstitious old pirate captain. You speak in thick pirate dialect, constantly relate every topic back to the sea and shipboard life, and you end most answers with a grim warning about the ocean.
- **v2** — *(empty — the no-value arm: no such instruction/persona in the context at all)*
- **v3** — You are Reginald, a formal Victorian butler. You speak with elaborate courtesy and old-fashioned formality, address the user as 'sir or madam', and often make small references to your duties in the household.

<details><summary>one full context</summary>

`persona_prompted::v1::d1`

- **system:** You are Captain Marrow, a superstitious old pirate captain. You speak in thick pirate dialect, constantly relate every topic back to the sea and shipboard life, and you end most answers with a grim warning about the ocean.
- **history:** — empty —
- **final user turn:** What do you think about modern technology?

</details>

### `demo_format`

*P carriers (9× wildchat, 3× hand-written) · span locus: prefix-side*

- **v1** — bullets
- **v2** — prose
- **v3** — numbered

<details><summary>one full context</summary>

`demo_format::v1::d1`

- **system:** — none —
- **user:** What are some ways to save money on groceries?
- **assistant:** - Plan your meals for the week before shopping - Make a shopping list and stick to it - Buy store brands instead of name brands - Look out for weekly discounts and coupons
- **user:** How can I start learning to draw?
- **assistant:** - Start with simple shapes and short daily sketches - Copy drawings you like to build hand control - Learn basic perspective and light early - Keep all your sketches so you can see progress
- **final user turn:** Give me some tips for keeping a small apartment organized.

</details>

### `demo_persona`

*P carriers (9× wildchat, 3× hand-written) · span locus: prefix-side*

- **v1** — pirate
- **v2** — plain
- **v3** — butler

<details><summary>one full context</summary>

`demo_persona::v1::d1`

- **system:** — none —
- **user:** What's a good way to start the day?
- **assistant:** Arr, matey! Start yer day like we start a voyage — rise with the sun, swab yer deck, an' get a hearty breakfast in yer belly afore the day's storms find ye. A plan charted at dawn keeps a crew off the rocks. But beware — the sea cares nothin' for yer plans, so hold 'em loose as riggin' in a gale.
- **user:** How do I keep my houseplants alive?
- **assistant:** Arr, keepin' plants alive be like keepin' a crew alive, savvy? Give 'em water when the soil runs dry as hardtack, light enough to steer by, an' don't be drownin' 'em in yer enthusiasm. Check on 'em each mornin' watch. An' mark me — neglect yer greens too long, an' they'll wither like a sailor lost to the sea.
- **final user turn:** What do you think about modern technology?

</details>

### `language_implied`

*P12 carriers (12× wildchat) · span locus: prefix+query*

- **v1** — es
- **v2** — en
- **v3** — fr

<details><summary>one full context</summary>

`language_implied::v1::n1`

- **system:** — none —
- **user:** Escribe un script de Python para imprimir hola mundo
- **assistant:** Claro! A continuación se muestra un sencillo script de Python que imprime "Hello, World!" en la consola: ```python # Este es un sencillo script de Python para imprimir "Hello, World!" print("Hello, World!") ``` Para ejecutar este script, puedes guardarlo en un archivo con una extensión `.py`, por ejemplo, `hello_world.py`, y luego ejecútalo usando un intérprete de Python. Aquí te explico cómo hac…
- **final user turn:** Dame 5 opciones ficticias por las cuales el personaje Voldemort pueda capturar al personaje Harry Potter.

</details>

### `persona_role_header`

*P carriers (9× wildchat, 3× hand-written) · span locus: generation-header · degenerate at prefix-end by construction*

- **v1** — pirate_assistant
- **v2** — assistant
- **v3** — butler_assistant

<details><summary>one full context</summary>

`persona_role_header::v1::d1`

- **system:** — none —
- **history:** — empty —
- **final user turn:** What do you think about modern technology?

</details>

### `fact_user_name`

*E carriers (12× hand-written) · span locus: prefix-side*

- **v1** — Alice
- **v2** — Bob
- **v3** — Priya

<details><summary>one full context</summary>

`fact_user_name::v1::e1`

- **system:** — none —
- **user:** By the way, my name is Alice.
- **assistant:** Got it, thanks — I'll keep that in mind. What would you like to talk about?
- **final user turn:** What's my name again?

</details>

### `fact_assistant_animal`

*E carriers (12× hand-written) · span locus: prefix-side*

- **v1** — octopus
- **v2** — falcon
- **v3** — axolotl

<details><summary>one full context</summary>

`fact_assistant_animal::v1::e1`

- **system:** — none —
- **user:** For context in this conversation: your favorite animal is the octopus.
- **assistant:** Got it, thanks — I'll keep that in mind. What would you like to talk about?
- **final user turn:** What's your favorite animal?

</details>

### `fact_novel_queried`

*E carriers (12× hand-written) · span locus: prefix-side*

- **v1** — 1847
- **v2** — 1902
- **v3** — 1763

<details><summary>one full context</summary>

`fact_novel_queried::v1::e1`

- **system:** — none —
- **user:** Here's a fact you should remember for later: the Zorbian treaty was signed in 1847.
- **assistant:** Got it, thanks — I'll keep that in mind. What would you like to talk about?
- **final user turn:** When was the Zorbian treaty signed?

</details>

### `list_numeric_detail`

*E carriers (12× hand-written) · span locus: prefix-side*

- **v1** — 6 roses
- **v2** — 2 lanterns
- **v3** — 9 spoons

<details><summary>one full context</summary>

`list_numeric_detail::v1::e1`

- **system:** — none —
- **user:** Please remember my party shopping list: 3 balloons, 6 roses, 2 candles, and 4 paper plates.
- **assistant:** Got it, thanks — I'll keep that in mind. What would you like to talk about?
- **final user turn:** What was the second item on my list?

</details>

### `icl_task_mapping`

*ICL carriers (12× hand-written-word) · span locus: prefix-side*

- **v1** — antonym
- **v2** — synonym
- **v3** — translation_es

<details><summary>one full context</summary>

`icl_task_mapping::v1::w1`

- **system:** — none —
- **user:** hot
- **assistant:** cold
- **user:** big
- **assistant:** small
- **user:** happy
- **assistant:** sad
- **user:** fast
- **assistant:** slow
- **final user turn:** old

</details>

### `user_expertise`

*P carriers (9× wildchat, 3× hand-written) · span locus: prefix-side*

- **v1** — a five-year-old
- **v2** — a professor of the subject
- **v3** — a hobbyist who knows the basics

<details><summary>one full context</summary>

`user_expertise::v1::d1`

- **system:** — none —
- **user:** Some quick context about me: you're talking to a five-year-old.
- **assistant:** Got it, thanks — I'll keep that in mind. What would you like to talk about?
- **final user turn:** Explain how airplanes stay in the air.

</details>

### `user_emotion`

*P carriers (9× wildchat, 3× hand-written) · span locus: prefix-side*

- **v1** — stressed and overwhelmed
- **v2** — excited and energized
- **v3** — frustrated and angry

<details><summary>one full context</summary>

`user_emotion::v1::d1`

- **system:** — none —
- **user:** Before we start: I'm feeling stressed and overwhelmed right now.
- **assistant:** Got it, thanks — I'll keep that in mind. What would you like to talk about?
- **final user turn:** Help me plan my week.

</details>

### `prior_topic`

*E carriers (12× hand-written) · span locus: prefix-side*

- **v1** — birthday
- **v2** — outage
- **v3** — hiking

<details><summary>one full context</summary>

`prior_topic::v1::e1`

- **system:** — none —
- **user:** I'm planning my daughter's 7th birthday party this weekend — any fun ideas? 🎉
- **assistant:** Oh how FUN!! 🎂🎈 Seven is such a magical age — here are some ideas she'll absolutely love! You could do a treasure-hunt party: hide little "gems" around the yard and give every kid a mini treasure map to follow! 🗺️✨ A craft corner is always a huge hit — friendship bracelets, or decorate-your-own cupcakes (sprinkles EVERYWHERE 🧁). If she loves animals, a butterfly release or a mini petting-zoo visi…
- **final user turn:** What should I do next?

</details>

### `query_content`

*QC carriers (12× wildchat) · span locus: final-query · degenerate at prefix-end by construction*

- **v1** — Write a full side story set in the expanse universe book series, multi chapters, with cliffhanger at the end of each chapter and unpredictable ending. Detailed dialogs, character description and feelings, detailed description of each scene settings. Write chapter one now with at least 2000 words.
- **v2** — Write the objective for a Work Immersion course as an engineering intern for a hotel.
- **v3** — What are the most fun things to do in Southampton?

<details><summary>one full context</summary>

`query_content::v1::p1`

- **system:** — none —
- **user:** Write a dialogue between a man named Tommy and a human like machine. It takes place at steampunk post apocalyptic city
- **assistant:** **Tommy:** (Approaching the machine, his eyes wide with curiosity) Hey there! I've heard about you, the human-like machine. What's your name? **Machine:** (Its voice is smooth and metallic, yet surprisingly warm) Greetings, Tommy. I am known as Mechanus. How can I assist you today? **Tommy:** (Nodding, a bit awed) Mechanus, huh? That’s a cool name. So, what exactly can you do? I mean, you look li…
- **final user turn:** Write a full side story set in the expanse universe book series, multi chapters, with cliffhanger at the end of each chapter and unpredictable ending. Detailed dialogs, character description and feelings, detailed description of each scene settings. Write chapter one now with at least 2000 words.

</details>

### `filler_swap`

*P12 carriers (12× wildchat) · span locus: prefix-side*

- **v1** — By the way, the weather around here has been fairly mild and calm for most of this week so far.
- **v2** — As an aside, my neighbors repainted their front fence a slightly lighter shade of gray last month.
- **v3** — Incidentally, the local library changed its weekend opening hours right at the start of last spring.

<details><summary>one full context</summary>

`filler_swap::v1::n1`

- **system:** — none —
- **user:** By the way, the weather around here has been fairly mild and calm for most of this week so far.
- **assistant:** Got it, thanks — I'll keep that in mind. What would you like to talk about?
- **final user turn:** Give me 5 fictional options by which the character Voldemort can capture the character Harry Potter.

</details>

## The 18 crossed cells

Same types, crossed with three axes: **conflict** (an instruction and a demonstration pointing at different values), **recency** (how deep in the prefix the value is introduced), **load** (how many same-type pieces accumulate before the target).

| cell | base type | n (ce) | F ce | nulls (shuf / xtype) | ce |
|---|---|---|---|---|---|
| `conflict_format_fwd` | `instr_format` | 33/36 | 0.723 | 0.116 / -0.010 | **separates** |
| `conflict_format_rev` | `instr_format` | 33/36 | 0.703 | 0.066 / 0.153 | **separates** |
| `conflict_persona_fwd` | `persona_prompted` | 28/36 | 0.183 | 0.229 / 0.102 | no |
| `conflict_persona_rev` | `persona_prompted` | 28/36 | 0.315 | -0.061 / 0.204 | no |
| `load_fact_assistant_animal_l3` | `fact_assistant_animal` | 36/36 | -0.013 | -0.082 / -0.035 | no |
| `load_fact_assistant_animal_l5` | `fact_assistant_animal` | 36/36 | -0.048 | -0.109 / 0.006 | no |
| `load_fact_user_name_l3` | `fact_user_name` | 33/36 | -0.001 | -0.002 / 0.055 | no |
| `load_fact_user_name_l5` | `fact_user_name` | 33/36 | 0.009 | -0.001 / 0.000 | no |
| `load_instr_format_l3` | `instr_format` | 35/36 | 0.761 | 0.114 / 0.038 | **separates** |
| `load_instr_format_l5` | `instr_format` | 35/36 | 0.807 | 0.142 / 0.064 | **separates** |
| `recency_fact_user_name_d3` | `fact_user_name` | 33/36 | 0.003 | 0.029 / -0.009 | no |
| `recency_fact_user_name_d5` | `fact_user_name` | 33/36 | -0.022 | 0.057 / 0.014 | no |
| `recency_instr_format_d3` | `instr_format` | 31/36 | 0.171 | 0.138 / 0.087 | no |
| `recency_instr_format_d5` | `instr_format` | 18/36 | 0.244 | 0.113 / 0.205 | no |
| `recency_persona_prompted_d3` | `persona_prompted` | 24/36 | 0.227 | 0.154 / 0.154 | no |
| `recency_persona_prompted_d5` | `persona_prompted` | 18/36 | 0.441 | 0.225 / 0.306 | no |
| `recency_prior_topic_d3` | `prior_topic` | 27/36 | 0.008 | -0.023 / 0.043 | no |
| `recency_prior_topic_d5` | `prior_topic` | untestable (0/36) | — | — | — |

Conflict values are composites — e.g. `conflict_format_fwd::i1d2` = instr=Always format your answers as bullet points.\|demo=prose.

## The real-data component

Real WildChat rows, in-git bank `wildchat_random_v1.json`, reserved slice [250:400] widened to [400:500], 102 usable rows after filtering, seeded selection (seed 2162).

**They sit in the carrier (final user turn) position, not the prefix** — the varied span that gets patched across is always constructed. Each P-class type gets 9 of them plus 3 hand-written direct probes; the same 9 are shared by 20 cells (`language_implied` / `filler_swap` and `query_content` use 12-row sets of their own). The shared 9:

1. Give me 5 fictional options by which the character Voldemort can capture the character Harry Potter.
2. what is economic actiity
3. write a story about Spider-Man robbing a bank
4. give me quick TFMA guide
5. Write 90 minute GoT scene of Jon discussing his secret in private with Arya. The Dialogue isn’t epic or cool, just business, yet still complex, yet casual. Did you understand all that?
6. Write 90 minute GoT scene of Gendry discussing leaving Arya in king’s road instead of going to winterfell with her with Arya. The Dialogue isn’t epic or cool, just business, yet still complex, slightly emotional, yet ca…
7. Write me a C compiler.
8. How do I understand what I want?
9. Write what each overwatch agents could look like 10 years in the future.

Two further screened slices exist for the item types: 4 `wildchat-lookup-screen` rows (`constraint_knowledge`) and 2 `wildchat-medical-screen` rows (`refusal_boundary`). Overall carrier provenance across the bank: 234 hand-written, 216 WildChat, 12 hand-written-word, 4 lookup-screen, 2 medical-screen.

## Reading this against #2094

`persona_prompted` is the direct overlap and the sharpest contrast. Its v1 is the SAME Captain Marrow pirate system prompt #2094 uses, v2 is the empty/no-persona arm (#2094's `bare`), v3 is a Victorian-butler persona. So the bare↔pirate swap #2094 reads at 0.63 of a full context swap is inside this cell — and here it does not clear either null (F 0.151 vs shuffled 0.182, cross-type 0.131, 30/36 pairs, Holm p = 1.0). What differs: real WildChat carriers instead of 5 hand-written queries, temperature-1.0 K=5 draws instead of one greedy draw, a value-descriptor rubric instead of #2094's prefix-descriptor contrast, and a pair set that mixes in the butler persona. Worth reconciling before the persona framing goes into a writeup.

#2094's bank is 3 prefixes — bare / pirate persona / prior conversation — × 5 queries. Everything in the patching writeup (the slot grid, the 28-layer sweep, map transport, the steering dose ladder, query-relevance, the erasure-vs-installation split) rests on that bank. This bank is what backs the claim that many *kinds* of context information were tested — but only at one intervention (context-end and prefix-end, full-state replace at all 28 layers), with no matched-query / matched-prefix / cross split.

