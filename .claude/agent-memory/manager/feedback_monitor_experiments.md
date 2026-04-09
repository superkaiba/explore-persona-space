---
name: Monitor experiments periodically
description: Always use experimenter subagent to periodically monitor running experiments without being asked
type: feedback
---

Always monitor running experiments periodically using an experimenter subagent. Don't wait for the user to ask — proactively check on running jobs and report when they finish or fail.

**Why:** User expects to be kept informed of experiment progress without having to ask each time.

**How to apply:** Whenever experiments are running on pods, spawn a background experimenter agent to periodically check GPU utilization, training logs, and completion status. Report results as soon as experiments finish.
