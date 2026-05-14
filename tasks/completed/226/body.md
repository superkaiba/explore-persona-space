---
title: Workflow improvements
kind: infra
tags: []
created_at: '2026-05-04T18:28:25.000Z'
has_clean_result: false
sagan_id: e66aa4d2-7c58-4052-9712-04d0cfb8cb68
sagan_number: 226
priority: normal
---
Weekly code cleanup and review - based on results and failure modes
We should have AI generated clean results -> and then clean results after they've been cleaned up/approved by me -> THIS SHOULD BE ITS OWN COLUMN
We also want a daily update as a github gist every day  
And a weekly update. 
These should be very presentable and include context, methodological details, etc., so that even someone with LOW context on the project can understand.
Jargon should not be used ANYWHERE
The AI generated clean results should LOOK at the approved clean results when it's generating its clean results to learn from in context examples (There should be a user step where the user reads and corrects the clean result and writes a "human summary")
Happy coder should change chat name to be very informative -> what is the issue number we are working on and what status are we at (including followups with description, and if there is a clean result it should also be included)
The model currently says stuff like "Oh it's late, should I continue tomorrow" or "should I continue with the pipeline" -- it should continue with the pipeline except for the user mandated review times 
Careful for the diskspace on each pod before starting each experiment
There should be an easy way to view the raw outputs 
When there's a bug and the experimenter comes back it should dispatch a new experimenter.

Is there a better way to have a kanban board on github than with a project with an experiment queue? The issues don't automatically go into the experiment queue which is a bit annoying
