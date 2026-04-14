---
name: Stop Polling Remote Runs
description: Don't poll SSH every 15s for long-running jobs; schedule one wakeup near ETA instead
type: feedback
---

When waiting for a remote GPU job to finish, do NOT poll every 15-30 seconds. This wastes context, annoys the user, and provides no new information.

**Why:** User interrupted a loop where I was SSH-checking every ~15 seconds for a result that wouldn't exist for 22 more minutes. Each check returned identical output.

**How to apply:** Estimate the ETA from log timestamps (time per persona × remaining personas + eval time). Schedule ONE wakeup ~2 minutes after ETA. If the result isn't there, schedule one more 5-min wakeup. Never loop-poll faster than the expected time between state changes.
