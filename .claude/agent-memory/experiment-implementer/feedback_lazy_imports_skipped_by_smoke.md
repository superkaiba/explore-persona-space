---
name: Lazy imports in smoke-skipped branches are unverified code
description: ImportError class that fires only on the pod — lazy imports inside --dry-run/--skip-upload-skipped branches; hoist to module top + AST-based --verify-imports smoke + signature-bind fenced calls (#1332)
type: feedback
---

A lazy import inside a branch that every local smoke skips (`--dry-run` /
`--skip-upload` upload paths, judge-only branches, GPU-only paths) is
UNVERIFIED code — the ImportError fires only on the pod, AFTER the
expensive phases complete. Incident #606 (2026-06-11): the dispatcher
lazily imported `_retry_transient` from `orchestrate.hub` (symbol never
existed there — it is file-local in the i528 scripts); training +
stage-A judging ran ~18 min on a 4×A100 GCP VM before the upload phase
crashed on the import.

**Why:** local CPU smokes legitimately skip upload/GPU branches, so a
deferred import in those branches executes for the first time in
production.

**How to apply:** (1) hoist cheap cross-script helper imports to module
top so absence crashes at process start; (2) before relaunch, run an
AST-based `--verify-imports` mode that walks every in-scope file and
EXECUTES every deferred import (hand-maintained symbol lists re-create
the drift — generate from the AST), and signature-BIND every
smoke-fenced call to an imported helper — `inspect.signature(fn).bind(...)`
with the call site's statically-known shape (#1332: import resolution
green-lit fenced calls missing a required positional + kw-only arg);
(3) when porting code that calls a
private helper (`_retry_transient`-style), grep the SOURCE branch for
where the symbol is actually defined — file-local helpers do not travel
with the import path you assume.
