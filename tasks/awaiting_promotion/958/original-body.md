---
title: How does the context→answer mapping change as the context gets longer (multi-turn)
kind: experiment
tags: []
created_at: '2026-07-04T04:06:00Z'
has_clean_result: false
origin_prompt: "Run this issue in the background with happy coder:\n# How does the\
  \ mapping change as the context gets longer\n# Motivation\n- All our mapping experiments\
  \ have been on mostly short contexts\n- We want to see if this mapping holds up\
  \ at longer contexts and over many chat exchanges\n    - or generally if it changes\
  \ - i.e. is the mapping from prefix + query 1 -> answer 1 the same as the mapping\
  \ from prefix + query 2 -> answer 2\n    - also probably can we predict answer 2,\
  \ 3, 4 from prefix + query 1 (and other combinations)\n- This could also explain\
  \ something like persona drift/context rot\n# Methodology\n- Take longer multi turn\
  \ conversations from some generic chat dataset\n- Train above mappings (use same\
  \ conversations for all turn numbers)\n- Characterize relationship between mappings,\
  \ try to predict second answer from query 2 using the first mapping, etc.\n\n(no\
  \ approval needed from me)"
workflow: v1
goal: 'Characterize whether the mappings (trained so far on mostly short contexts)
  hold up or change systematically as the context gets longer and spans many chat
  turns — training BOTH the context→answer mapping AND the prefix→answer mapping on
  the SAME data: (a) is the mapping from (prefix + query 1 → answer 1) the same as
  the mapping from (prefix + query 2 → answer 2); (b) can later answers (answer 2,
  3, 4, …) be predicted from earlier-turn inputs using the earlier-turn mapping; (c)
  does any systematic change in either mapping with context length offer an account
  of persona drift / context rot. Open design question for the planner: how to obtain
  multiple queries per prefix to average over for the prefix→answer mapping.'
relates_to:
- spec-context-as-vector
- spec-sysprompt-vs-drift
---
# How does the mapping change as the context gets longer

## Goal

Characterize whether the mappings (trained so far on mostly short contexts) hold up or change systematically as the context gets longer and spans many chat turns — training BOTH the context→answer mapping AND the prefix→answer mapping on the SAME data: (a) is the mapping from (prefix + query 1 → answer 1) the same as the mapping from (prefix + query 2 → answer 2); (b) can later answers (answer 2, 3, 4, …) be predicted from earlier-turn inputs using the earlier-turn mapping; (c) does any systematic change in either mapping with context length offer an account of persona drift / context rot. Open design question for the planner: how to obtain multiple queries per prefix to average over for the prefix→answer mapping.

## Motivation

- All the mapping experiments so far have been on mostly short contexts.
- We want to see if the mapping holds up at longer contexts and over many chat exchanges — or generally whether it changes: is the mapping from prefix + query 1 → answer 1 the same as the mapping from prefix + query 2 → answer 2?
- Also: can we predict answer 2, 3, 4 from prefix + query 1 (and other combinations)?
- A systematic change in the mapping with context length could explain phenomena like persona drift / context rot.

## Methodology (sketch — to be refined by the planner)

- Take longer multi-turn conversations from some generic chat dataset.
- Train the above mappings (use the same conversations for all turn numbers).
- Train BOTH the context→answer mapping AND the prefix→answer mapping, on the same conversations, so the two mapping types are directly comparable.
- Characterize the relationship between the mappings; try to predict the second answer from query 2 using the first mapping, etc.

## Open design questions (planner to resolve)

- The prefix→answer mapping needs multiple queries per prefix to average/marginalize over, but a generic chat dataset supplies only ONE realized query + answer per prefix per conversation. How to obtain the query set to average over is UNRESOLVED and must be settled in the plan before any training. Candidate constructions to weigh (each changes what "prefix→answer" means, so the plan must define the marginalization precisely): sample queries from other conversations at the same turn position; generate multiple queries per prefix on-policy from the base model; pool dataset queries across conversations sharing similar prefixes.

## Notes

- User pre-approved running this in the background with no further approval needed ("no approval needed from me"); the autonomous session proceeds under the standard plan-approval GPU-hour cap.
