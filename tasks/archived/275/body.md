---
title: Workflow improvements
kind: infra
tags: []
created_at: '2026-05-05T19:20:51.000Z'
has_clean_result: false
sagan_id: 178f0ed3-1c8b-4b35-b537-2460639935ca
sagan_number: 275
priority: normal
legacy_why_unset: true
---
Right now are the agents able to use SSH MCP when they create a new pod?
Have concrete hypothesis, falsifiable claims before running experiment --> integrate into workflow, search web for best way
Does the manager give full plan to the sub agent?
USE LESS VERY CONFUSING ACRONYMS (e.g. P1, H1 -- be clearer about everything in the github issues/clean results)
The Clean Result should be clear enough that even someone with very little context on the project can understand it
Link to plan in chat when asking for approval from user
For the daily skill --> we first need to review clean results and then decide next steps, and this should be summarized in the daily gist (what was done today, blockers, next steps) -- daily gist needs to be approved by user
Add useful vs non useful results column
The TLDRs refer to H1 and stuff without ever defining them. 
Each Clean Result claim should stand on its own without referring to past results.
In the background section we should refer to past results to say why we ran this
Can we add a step (or augment a step) to look at the raw results/ text generations and make sure nothing fishy is going on
Also add to always include some examples from each dataset used or generated in an experiment to the TLDR, with a link to the wandb with the full data
The issue should be moved to "Planning" as soon as the /issue command gets run

This bug should not happen:
● Bash(uv run python scripts/gh_project.py set-status 260
       "In Progress")
  ⎿  Error: Exit code 1
     unknown column 'In Progress'. valid: Archived,
     Awaiting promotion, Blocked, Clean results, Done,
     Followups running, In flight, Plan awaiting review,
     Planning, To do

  Called happy (ctrl+o to expand)

● Project board uses more granular column names than the
  skill doc. Mapping to "Planning" for the planning
  phase.
