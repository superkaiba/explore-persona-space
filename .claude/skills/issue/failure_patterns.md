# Failure-class log patterns

When `epm:failure` body lacks `failure_class:`, the `/issue` skill scans
the body + last 200 lines of the linked log against these patterns. Any
match → route as `infra`. Otherwise → `code` (conservative).

## Infra patterns (regex, case-insensitive)

```
CUDA out of memory
OOM-killer|Killed
No space left on device|ENOSPC|disk full
NCCL (timeout|error)
SSH connection refused|No route to host|Connection timed out
401 Unauthorized|gated repo
RuntimeError: CUDA error
Failed to initialize.*vllm
Traceback.*\b(vllm|transformers|peft|trl|torch|xformers)/
```

## Code patterns (regex, case-insensitive)

These are NOT used for inference (the fallback only looks for infra).
Listed here for completeness of the experimenter agent's checklist:

```
Traceback.*\b(src/explore_persona_space|scripts)/
^AssertionError
^TypeError
^KeyError
```

## Adding a pattern

Edit this file. Both `.claude/skills/issue/SKILL.md` Step 7 and
`.claude/agents/experimenter.md` cross-reference this file by path; no
other change needed. (Allowed under §10 plan deviations: implementer can
extend the pattern list without asking.)
