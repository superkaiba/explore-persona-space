---
name: Codex upload-policy shape literalism
description: Codex code-reviewer FAILs when raw-completions/artifacts upload via a non-canonical LAYOUT (whole-cell-tree, plan-registered nesting) instead of the upload-policy table's literal path/helper; verify the GUARANTEE (data-repo, fail-loud, pre-termination, verifier-resolvable) not the shape
type: feedback
---

Codex code-reviewer treats the CLAUDE.md Upload Policy table's path column +
helper name (`upload_raw_completions_to_data_repo()`,
`issueN_<slug>/raw_completions/{condition}_seed{S}.json`) as a binding SHAPE
contract and FAILs implementations that satisfy the guarantee through a
different layout. Its evidence is typically a grep for canonical symbol
spellings (`upload_raw_completions_to_data_repo|hub\._upload\(.*raw_completions`)
— a literal-string test that can even miss the actual `hub._upload` call when
wrapped (e.g. `_upload_or_raise`).

**Why:** #612 round 2 (2026-06-12). Dispatcher uploaded each cell tree
(including `raw_completions/*.json`) fail-loud to the data repo under
`issue612_<slug>/eval_results/<cell>/...`, presence-asserted, pre-sentinel,
with `list_repo_files_complete` verification — and the PLAN's §10 DV globs
registered exactly that nesting. Codex FAILed it as
`raw-completions-upload-missing`. Three defeaters: (a) the policy's core rule
is the GUARANTEE ("raw completions MUST upload before pod termination");
`verify_uploads.py` + the upload-verifier agent have NO path-shape check —
they PASS any file resolving at a permanent data-repo home. (b) Codex's
prescribed fix was a regression: the canonical helper rglobs files named
exactly `raw_completions.json`, which matches ZERO files in a
`raw_completions/{persona}_seed{S}.json` layout (silent empty-dict return).
(c) Codex's own round-1 review had marked the SAME unchanged line "✓
implemented" — round-2 re-litigation of unchanged wiring (companion pattern:
feedback_codex_litigates_pre_existing_in_round_n.md,
feedback_codex_step_06_literal_vs_purpose.md).

**How to apply:** When Codex FAILs on an upload-path/helper-name mismatch:
(1) read the actual upload call chain — is it fail-loud, presence-asserted,
sequenced before pod termination/sentinel, on the right repo? (2) check
whether the canonical helper is even shape-compatible with the experiment's
file naming (it scans for literal `raw_completions.json`); (3) check the plan
§10 artifact globs — a plan-registered nesting endorses the layout; (4) diff
the round-N commit against the wiring and check Codex's round-(N-1) adherence
table for a prior ✓ on the same line. If the guarantee holds and Step 8's
verifier resolves the files, adjudicate PASS with a standing recommendation
to record the layout in the reproducibility card.
