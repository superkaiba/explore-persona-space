---
name: Verify agent infrastructure claims before acting
description: Specialist agents can confidently report false findings about submodule states, dataclass fields, or parser behavior due to silent probe failures; always verify via direct inspection
type: feedback
---

When a specialist agent reports a surprising finding about infrastructure state (submodule pin, dataclass fields, parser behavior, installed package versions), **verify it myself via direct AST inspection or shell before acting**. Do NOT chain dispatches based on unverified claims.

**Why (2026-04-17 session cascade):**
1. Tier 2 experimenter (a3f066e... on #40) claimed "open-instruct submodule pinned at `6b3964bc`, `use_liger_kernel`/`packing` not fields on FlatArguments, crashes parser". Root cause: their probe was `python -c "from open_instruct.finetune import FlatArguments"` which hit `ImportError: No module named 'olmo_core'`. They misinterpreted the ImportError as "field not in dataclass". Actual state: submodule at `45901fd0`, both fields valid.
2. I dispatched implementer (#41) partly based on the false finding; fortunately the implementer did independent AST parsing and caught it. Corrective commit landed (#41 e08eea8).
3. Runtime-verification experimenter (a3f066e... on #43) then made a SECOND set of confidently-wrong claims: "submodule pinned at `6b3964bc` (re-verifying)", "allowlist drops 24/27 DPO flags", "launch_stage.py `do_not_randomize_output_dir` crashes SFT parse". All verified FALSE by direct local AST inspection: allowlist has 142 DPO fields and 85 SFT fields including all the claimed-missing ones.

**Net cost:** cascading issues (#40 correction, #42 created then closed as moot, #43 dispatched + closed as wrong), ~3 hours of agent compute on false premises, temporary user confusion about whether Option A bump was needed.

**How to apply:**
1. When an agent reports "field X is missing from dataclass Y" or "submodule is at commit Z", run a quick local verification before accepting: `cd external/open-instruct && git log -1`, plus AST parse of the relevant .py file, takes <60s.
2. Prefer agent probe methods that use AST parsing over runtime imports — imports fail on missing runtime deps and can silently mislead.
3. If two agents' findings conflict (as they did between #41 and #43), pause and verify ground truth myself before dispatching further work based on either.
4. Specifically for open-instruct infrastructure: its dataclass inheritance is complex (DPOExperimentConfig inherits 10 base classes). Walking bases via AST requires careful handling; don't trust a simple "is field X on class Y" probe.
