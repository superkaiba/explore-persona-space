---
title: Workflow improvement
kind: infra
tags: []
created_at: '2026-04-29T23:06:55.000Z'
has_clean_result: false
sagan_id: f98926ae-ba08-46e6-9484-dd38963ebc3d
sagan_number: 149
priority: normal
legacy_why_unset: true
---

Add daily cleanup and summary at the end of the day?
- or maybe this should be weekly
- For clean results should not assume any prior knowledge by reader, everything should be explained clearly or refer to other issue
- One important thing during planning/review is to check if any duplicate code has been written/planned or if something in current experiment is inconsistent with other similar experiment (and if so this should all be explained)
- We want to have everything be forced through hooks/automated scripts as much as possible because LLMs get lost in long contexts
- There should be a planner -> plan in detail with user, then implementer which implements everything, then reviewer which looks at implementation and flags inconsistencies which implementer then fixes, then tester which writes tests independently based on plan
- Every experiment should be linked to: git commit which created it, environment snapshot (ideally also which pod), command to run it, all artifacts CREATED and USED or it, all results saved and uploaded to wandb, all model checkpoints/adapters uploaded to huggingface hub, and a Linear issue which tracks everything that happened related to the experiment and tracks what stage it's at.
- The end goal for each experiment should be a clean result which gives what was learned by the experiment as well as followups which get created automatically as linked Linear issues
- This clean result should only be created after I back and forth and approve
- In terms of followup experiments there should be a followup designer which looks at the results and proposes followups
- We want to run on runpod and start/pause as needed (or does it make more sense to create pods on the fly?)

Probably should have claims and experiments and a claim can have many experiments

Also the review process before and after is important
It should make sure the experiment's methodology is consistent with all other experiments in the codebase
And then cleanup to refactor and fix things based on patterns and transcripts

Issue -- plan -- critique plan -- result that INCLUDES SNAPSHOT OF CODE AND ENV, COMMAND TO RUN, all data used to train, all data used for eval, all results, ideally saved to wandb

Maybe even the Claude Code transcript could get attached to the issue

Ideally I put an issue in linear (as vague as needed)
Then I say I want to start plannign that issue -> we plan together, the agent should look for related issues, ask what the specific hypothesis being tested is, it shouldn't be too much of a stickler for things - just ask what the potential results are and why we are running the experiment
Then we finalize the plan and it gets posted to the issue, where the context of related issues gets posted to the plan too
Then a reviewer looks at the plan automatically and flags issues/things that are unclear
Then they ask me about the issues -> I correct and we get a revised plan
Once the plan is done we send it off to an implementer that implements it in the codebase. The implementer should reuse as much code as possible from the codebase as well as related issues
Once the implementer is done a reviewer automatically gets dispatched which has access to the original plan and decides if there is something to change - if there is then another agent gets dispatched to change those things
Then another reviewer -> implementer loop until final reviewer is satisfied
Then a tester gets the plan and writes tests to test the implementation (if needed). The tests should cover end to end testing of the pipeline on a pod with minimal parameters/steps and check that all uploads work properly
If tests fail an implementer gets dispatched to fix the errors -> rerun tests -> implementer until everything is fixed

Once everything passes an experimenter gets dispatched to unpause a pod, sync the code/environment, run the experiment and monitor progress periodically (There should be a script setup to do this -- e.g. ping the experimenter at progressively longer intervals so it checks/fixes problems)

Once the experiment ends 
-> post experimenter gets dispatched to make sure that the reproducibility checklist is good and everything has been uploaded/logged, and also to pause the pod
-> analyzer gets dispatched to analyze results based on initial plan/hypothesis/expected results
Analyzer tries to make a clean result based on our established format (gets access to plot making skills/plugins)
Then reviewer looks at analyzer thing for overclaims and etc.
Then I get asked to check and we fix the clean result together

Throughout this the status of the issue in linear should be getting updated

Also as much as possible everything should be done through hooks/automated scripts so we don't have to rely on the LLMs remembering to do things

Some things I'm not sure about:
- What about followup experiments? --> we don't want to have to go through the ENTIRE process again
- What about experiments that require minimal code change (e.g. just a new config)

Also all code changes should be done on a worktree before being merged
