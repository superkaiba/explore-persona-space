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
goal: 'Characterize whether the context→answer mapping (trained so far on mostly short
  contexts) holds up or changes systematically as the context gets longer and spans
  many chat turns: (a) is the mapping from (prefix + query 1 → answer 1) the same
  as the mapping from (prefix + query 2 → answer 2); (b) can later answers (answer
  2, 3, 4, …) be predicted from earlier-turn inputs using the earlier-turn mapping;
  (c) does any systematic change with context length offer an account of persona drift
  / context rot.'
---
# How does the mapping change as the context gets longer

## Goal

Characterize whether the context→answer mapping (trained so far on mostly short contexts) holds up or changes systematically as the context gets longer and spans many chat turns: (a) is the mapping from (prefix + query 1 → answer 1) the same as the mapping from (prefix + query 2 → answer 2); (b) can later answers (answer 2, 3, 4, …) be predicted from earlier-turn inputs using the earlier-turn mapping; (c) does any systematic change with context length offer an account of persona drift / context rot.

## Motivation

- All the mapping experiments so far have been on mostly short contexts.
- We want to see if the mapping holds up at longer contexts and over many chat exchanges — or generally whether it changes: is the mapping from prefix + query 1 → answer 1 the same as the mapping from prefix + query 2 → answer 2?
- Also: can we predict answer 2, 3, 4 from prefix + query 1 (and other combinations)?
- A systematic change in the mapping with context length could explain phenomena like persona drift / context rot.

## Methodology (sketch — to be refined by the planner)

- Take longer multi-turn conversations from some generic chat dataset.
- Train the above mappings (use the same conversations for all turn numbers).
- Characterize the relationship between the mappings; try to predict the second answer from query 2 using the first mapping, etc.

## Notes

- User pre-approved running this in the background with no further approval needed ("no approval needed from me"); the autonomous session proceeds under the standard plan-approval GPU-hour cap.
