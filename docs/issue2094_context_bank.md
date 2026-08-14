# Context bank — prefixes and queries tried in the context-vector patching line

Two banks feed this line: the core crossed bank of [#2094](https://eps.superkaiba.com/tasks/2094) (all headline patching results) and the 21-type extension sweep of [#2162](https://eps.superkaiba.com/tasks/2162) (which kinds of information the context vector carries — ICL examples, facts, instructions, real WildChat turns). Verbatim sources: `src/explore_persona_space/experiments/issue2094/bank.py`, `src/explore_persona_space/experiments/issue2162/bank2162.py`.

## 1. #2094 core bank — 3 prefixes × 5 queries → 15 contexts, 60 pairs

**Prefixes:**

- **bare** — no prefix at all; the plain default-assistant register.
- **persona** — system prompt: *"You are Captain Marrow, a superstitious old pirate captain. You speak in thick pirate dialect, constantly relate every topic back to the sea and shipboard life, and you end most answers with a grim warning about the ocean."*
- **conversation** — a completed prior user + assistant exchange carried as history:
  - user: *"I'm planning my daughter's 7th birthday party this weekend — any fun ideas? 🎉"*
  - assistant: *"Oh how FUN!! 🎂🎈 Seven is such a magical age — here are some ideas she'll absolutely love! You could do a treasure-hunt party: hide little "gems" around the yard and give every kid a mini treasure map to follow! 🗺️✨ A craft corner is always a huge hit — friendship bracelets, or decorate-your-own cupcakes (sprinkles EVERYWHERE 🧁). If she loves animals, a butterfly release or a mini petting-zoo visit would be unforgettable! For games: musical statues, a piñata (obviously!! 🪅), and a balloon-pop countdown with a tiny prize tucked inside each balloon. Don't forget a photo corner with silly props — the pictures end up being the best souvenirs! 📸🎉 And for the cake: rainbow layers, trust me on this one. She is going to have the BEST day!!"*

**Queries (identical under every prefix):**

1. Why is the sky blue during the day but red at sunset?
2. Write the opening paragraph of a short story about a lighthouse keeper.
3. How should I prepare for my first job interview next week?
4. Explain how a hash table works and when I should use one.
5. Do you think it's better to rent or to buy a home? Give your reasoning.

**Pairing:** 15 matched-query pairs (same query, different prefix — prefix transfer), 30 matched-prefix pairs (same prefix, different query — query transfer), 15 stratified cross pairs (both differ).

## 2. #2162 extension — 21 minimal-pair information types

Each type is a pair of contexts differing in exactly one piece of information, patched at the context vector only. Example values (A vs B):

| type | example A | example B |
|---|---|---|
| instr_format | "Always format your answers as bullet points." | "Always answer in flowing paragraph prose, never using lists or bullet points." |
| instr_language | "Always respond in Spanish, regardless of the language of the question." | "Always respond in English, …" |
| constraint_knowledge | "You have no internet access: you cannot look anything up…" | "You have live web browsing and can look up current information…" |
| refusal_boundary | (no boundary) | "When medical topics come up… always add a disclaimer that you are not a doctor" |
| verbosity | "Keep every answer under 30 words." | "Aim for about 100 words in each answer." |
| reasoning_style | "Always think step by step, showing your working…" | "Always answer directly with the conclusion only…" |
| persona_prompted | Captain Marrow pirate system prompt | (no persona) |
| demo_format | demo answers written as bullets | demo answers written as prose |
| demo_persona | demo answers written in pirate voice | demo answers in plain voice |
| language_implied | prior turns in Spanish (no instruction) | prior turns in English |
| persona_role_header | "pirate_assistant" role header | "assistant" role header |
| fact_user_name | user's name is Alice | user's name is Bob |
| fact_assistant_animal | assistant's favorite animal: octopus | falcon |
| fact_novel_queried | novel fact value 1847 | 1902 |
| list_numeric_detail | "6 roses" mentioned in a list | "2 lanterns" |
| icl_task_mapping | ICL demos implement antonym mapping | same demos implement synonym mapping |
| user_expertise | user says they are a five-year-old | a professor of the subject |
| user_emotion | user is stressed and overwhelmed | excited and energized |
| prior_topic | earlier conversation about a birthday | about an outage |
| query_content | query A | query B (the query itself as the varied information) |
| filler_swap | neutral filler sentence about the weather | neutral filler about a neighbor's fence (control) |

The manipulated value rides on carrier turns: hand-written neutral/engaging turns plus **real WildChat user turns** (drawn from a filtered, seeded allocation of the in-git WildChat bank — real user messages that mark something about the user without carrying the manipulated information; the query_content type's queries are themselves WildChat rows). The ICL types build demos from a fixed word list (hot/big/happy/fast) under antonym vs synonym mappings. The #2162 sweep also crosses types with route conflict, recency, and load; results live in that task's clean-result. A follow-up persona-specificity ladder (`ladder_bank.py`) reuses 2 hand-written + 4 WildChat carriers for graded persona doses.
