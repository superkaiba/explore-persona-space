---
name: Fan-out reap = process GROUP; per-cell file contracts need per-class writers
description: subprocess fan-out sibling reap must killpg a start_new_session group (uv->python->vLLM EngineCore); a resolver eagerly requiring a per-cell file needs a writer for every cell CLASS (#1112 r7)
type: feedback
---

Two traps from #1112 attempts 4/5 (2026-07-08), one crash masquerading as the other.

1. **Sibling reap must kill the whole process GROUP, never `proc.terminate()`.**
   `terminate()` on a `uv run python ...` Popen signals only the direct child; the
   python front-end and its vLLM `EngineCore` children survive. The orphaned
   EngineCores then dump `RuntimeError: Did not receive response from front-end
   process within 5 minutes` into the unit logs — 3-4 units "failing
   simultaneously" reads as an infra/handshake wedge and hides the real
   first-unit crash (attempt 4 was misdiagnosed exactly this way). Fix shape:
   spawn units with `start_new_session=True` (pgid == unit pid), reap with
   `os.killpg(p.pid, SIGTERM)` → wait ≤30 s → `killpg(SIGKILL)`; direct-child
   `send_signal` fallback on ProcessLookupError. Reference:
   `scripts/issue1112_dispatch.py::_reap_unit_groups` +
   `tests/test_issue1112_capture_resolution.py` (real-tree killpg test).

**Why:** the vLLM-teardown gotcha (workers not reaped) applies at the FAN-OUT
layer too, and a first-unit crash + abandoned siblings is indistinguishable
from a wedge in the logs.

2. **A resolver that eagerly requires a per-cell file must have a writer for
   every cell CLASS it enumerates.** #1112's `_resolve_capture_model` read
   `<cell>/selection.json` top-of-function for every non-base cell; the m1
   band-stop cell has no rung selection BY DESIGN, so no phase ever wrote one →
   deterministic FileNotFoundError only at the capture phase, after all
   training was done. Fix shape: read the file WHERE it is used (only the dose
   that needs it), add an explicit branch for the by-design-different cell
   class resolving the SAME artifact its earlier phase consumed, and backfill
   provenance-if-missing on the RESUME path too (skip-completed branches must
   also write, or resumed runs never gain the file). Also sweep write-order:
   a `build_result.json`-then-`selection.json` fresh path leaves a crash
   window that resume-skip never repairs — backfill in the skip branch.

**How to apply:** before launching any dispatcher with a fan-out + per-cell
resolver, (a) enumerate every cell class the grid iterates and grep
writers-vs-readers for each required per-cell file; (b) check the reap path
kills process groups, not direct children.
