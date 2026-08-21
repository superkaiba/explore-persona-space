---
name: installed-api-evidence-envelope
description: When a review emphasis requires verifying rule/doc text against an INSTALLED third-party source (.venv) the worktree cannot resolve, inline verbatim source excerpts with line numbers as a guaranteed-evidence envelope (#2442 r1)
metadata:
  type: feedback
---

Inline an `---BEGIN/END INSTALLED-API EVIDENCE---` envelope when the brief's
review emphasis is API-semantics truthfulness against an installed package
(e.g. `huggingface_hub` `hf_api.py`) — the worktree's `.venv` is typically a
stub without site-packages (verified on issue-2442: `$WT/.venv` existed but
had no `hf_api.py`), so a by-path instruction alone risks a false
`data-access-blocked` or a lens scored off the plan/body instead of the
source.

**Why:** #2442 r1 — the task BODY asserted `get_paths_info` 404s on a wrong
prefix; the installed docstring says a nonexistent path "is ignored without
raising an exception". The plan corrected the body, and the review had to
rank installed source > plan > body. Codex could not read the repo-root
`.venv` path reliably from its worktree-rooted sandbox, so the excerpts were
the only guaranteed evidence channel.

**How to apply:**
- Extract at compose time with `sed -n` by line ranges located via
  `grep -n "def <fn>"`; keep signature + docstring-through-Raises + the
  implementation body VERBATIM with source line numbers stated; elide long
  docstring examples with an explicit `[... elided ...]` marker.
- Append composer NOTE lines stating the load-bearing derivable facts (e.g.
  "generator — 404 raises at ITERATION, `yield` at line N"; "Raises block
  lists NO EntryNotFoundError") so Codex can cite them even if it distrusts
  its own trace.
- Instruct: prefer reading the live absolute `.venv` path when reachable;
  the envelope is the fallback; NEVER mark the API lens BLOCKED when the
  envelope is present (same never-BLOCKED discipline as the inlined
  plan/marker envelopes).
- Extend the prompt-file Step-3 validation greps to the new envelope tokens.

Related: [[whole-round-unsplit-compose]] (round-pinned sha-range diff when an
out-of-scope spec-sync commit sits at HEAD), [[infra-wf-fix-lint-gate-compose]]
(N/A-by-type + duty-discharge attestations for infra rounds).
