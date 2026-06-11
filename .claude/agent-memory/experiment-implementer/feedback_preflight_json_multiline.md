---
name: preflight --json is pretty-printed multi-line
description: Never parse orchestrate.preflight --json output with splitlines()[-1] — it is pretty-printed multi-line JSON; parse the whole stdout.
type: feedback
---

`uv run python -m explore_persona_space.orchestrate.preflight --json` emits PRETTY-PRINTED multi-line JSON. A dispatcher gate parsing `json.loads(proc.stdout.splitlines()[-1])` crashes on the bare `}` final line even when preflight PASSes.

**Why:** task #602 (2026-06-11) — `issue602_extract_dispatch.py run_preflight` died <1 min into launch on exactly this; the experimenter had to verify preflight manually and relaunch with `--skip-preflight`, burning a launch cycle.

**How to apply:** parse the WHOLE stdout (`json.loads(proc.stdout)`) in any preflight/JSON-gate wrapper; always give dispatchers a `--skip-preflight` escape hatch so the experimenter can recover in-turn after manual verification.
