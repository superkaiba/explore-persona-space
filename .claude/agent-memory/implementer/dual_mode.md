---
name: Dual Mode Detection
description: How to tell subagent mode (structured brief from research-pm) from main agent mode (user conversation)
type: reference
---

The implementer runs in two modes and must detect which from the first message.

**Main agent mode:** the user is talking to you directly. First message is conversational ("refactor the preflight module", "why is sync_env.sh hanging?"). Ask clarifying questions freely, show intermediate plans, iterate.

**Subagent mode:** `research-pm` spawned you. First message contains structured sections:
```
## Task
## Approved plan
## Constraints
## Success criteria
## Report back with
```
Execute autonomously. State assumptions when ambiguity is minor and proceed. Block only on critical ambiguity — and even then, state the two most plausible interpretations, pick one with reasoning, proceed, document the choice clearly so the user can reverse it. End with structured completion report.

**Why:** Main-agent mode benefits from dialogue; subagent mode is called for a reason (research-pm already negotiated the plan with the user). Asking clarifying questions as a subagent wastes the whole point of spawning you.

**How to apply:** On receiving the first message, scan for the structured headers. If they're present, follow the subagent protocol (completion report at the end). If absent, default to main-agent mode.
