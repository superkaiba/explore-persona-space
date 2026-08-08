---
name: Preflight --json is pretty-printed — never parse last line
description: Dispatchers gating on orchestrate.preflight --json must parse the WHOLE stdout, not splitlines()[-1]; smoke with --skip-preflight never covers the parse
type: feedback
---

`orchestrate.preflight --json` ALWAYS pretty-prints (`indent=2`), so any
dispatcher gating on `json.loads(stdout.splitlines()[-1])` crashes on the
bare `}` even when preflight PASSes — parse the WHOLE stdout, with a
first-`{` slice to tolerate prefix noise (uv/env chatter on fresh VMs).

**Why:** task #602 hit this TWICE in one day (2026-06-11/12). First: the
pod launch died <1 min in on the bare `}`; the experimenter verified
preflight manually and relaunched with `--skip-preflight`, burning a launch
cycle. Then the follow-up round: three GCP boots of `eps-issue-602` all
guestTerminated <3.5 min because `run_preflight()` in
`scripts/issue602_extract_dispatch.py` parsed only the last stdout line;
the startup-script EXIT trap powered the VM off each time. Root-caused via
a throwaway CPU diag VM replaying the same image + startup script. The
smoke missed it BOTH times because every smoke invocation passed
`--skip-preflight`, so the parse path never executed.

**Always give dispatchers a `--skip-preflight` escape hatch** so the
experimenter can recover in-turn after manual verification — but treat a
smoke that uses it as NOT covering the preflight gate.

**How to apply:** when writing or reviewing a dispatcher that shells
`python -m explore_persona_space.orchestrate.preflight --json`, parse with
`json.loads(raw)` → fallback `json.loads(raw[raw.index("{"):])` (see
`_parse_preflight_json()` in `scripts/issue602_extract_dispatch.py`,
commit `d54e0fdc6`). In the smoke plan, exercise `run_preflight()` against
the LIVE preflight CLI on the VM at least once — a smoke that skips
preflight does not cover the gate that runs first in production.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Preflight --json is pretty-printed](feedback_preflight_json_parse.md) — parse whole stdout (first-{ slice), never the last line; smoke must exercise run_preflight.
