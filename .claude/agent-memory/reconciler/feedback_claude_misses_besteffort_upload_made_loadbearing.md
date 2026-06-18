---
name: Claude misses best-effort upload made load-bearing by new consumer
description: New diff adds an HF-fetch consumer of a pre-existing warn-only upload producer; Claude PASSes on the consumer's logic, Codex traces provenance and FAILs
type: feedback
---

When a round adds a NEW fetch/consume site (e.g. a launch-script phase doing
`hf_hub_download` on `{prefix}/{cell}_seed{S}/{fname}`), trace WHO uploads that
exact path and whether that upload is fail-loud. The trunk producer may be
best-effort (`train_lora` sft.py:1258 wraps `upload_model` in try/except +
`logger.warning` on BOTH falsy return and exception) while the only fail-loud
upload covers a different subtree (`.../checkpoints` via `i601_run_cell.py`).

**Why:** #613 r1 — Claude verified p6's gauge-parity logic + frac-step map but
never traced the flag-on terminal-adapter upload path; Codex caught that a
silently-failed terminal upload lets both GPU seeds finish, then p6 dies, and
on the ephemeral GCP lane (EXIT-trap teardown + termination-action DELETE) the
never-uploaded local adapter is permanently lost (full GPU re-run). HF upload
failures have real precedent (#552/#541 LFS quota-403; adapters are LFS).

**How to apply:** "warn-only upload is pre-existing trunk" is NOT a
pre-existing-code defense when THIS diff introduces the consumer that makes it
load-bearing. The #606 contingent-path PASS rationale (durable Hub evidence +
free post-hoc fix) does not carry when the failure path has NO durable Hub copy
and teardown destroys the local one. Checks: (1) grep the fetched
`path_in_repo` back to its producer; (2) is the producer fail-loud or
env-gated-no-op (`EPM_PERSIST_*` unset)? (3) does the consumer fall back to the
surviving local artifact? (4) any `list_repo_files` verify between train and
fetch? All no → Real-blocking, FAIL + raise-concern.
