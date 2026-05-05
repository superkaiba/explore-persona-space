# Failure-class log patterns

> **Authoritative source: [`scripts/failure_classifier.py`](../../../scripts/failure_classifier.py).**
> This markdown file is a human-readable MIRROR of the regex list in
> that Python module. The `/issue` skill Step 7 shells out to the
> script (`uv run python scripts/failure_classifier.py --body - --log
> <path>`); it does NOT consult this markdown file at runtime. Keep
> the two in sync when extending; the Python module wins on conflict.

When `epm:failure` body lacks `failure_class:`, the script scans the
body + last 200 KB of the linked log against these patterns. Any match
→ route as `infra`. Otherwise → `code` (conservative).

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

Edit `scripts/failure_classifier.py` (the runtime authority) AND mirror
the change in this file. The tests in `tests/test_failure_classifier.py`
must still pass — extend them with a fixture covering the new pattern.
The skill SKILL.md and agent specs cross-reference by path; no further
SKILL/agent edits needed. (Allowed under §10 plan deviations: implementer
can extend the pattern list without asking.)
