---
name: stacked-lint-waivers-read-window
description: Two lint-waiver comments stacked above one call push the outer one out of its scanner's call-line/line-above read window; underscore-prefixed local ports escape \b-anchored wrap regexes
metadata:
  type: feedback
---

Each lint-waiver token must sit inside ITS OWN scanner's read window — most
waiver scanners read only the call line or the line directly above
(`workflow_lint._hf_routing_file_errors`: `"# NO_RETRY:" in line or in
lines[i-1]`; `_hub_dir_filecount_waiver_present`: call's first physical line
or immediately-preceding non-blank line). Stacking TWO waiver comments above
one call pushes the outer one to `i-2` — silently inert. Fix shape: move one
waiver to a trailing comment on the call line (#2330 R1 g4).

**Why:** #2330's `_retry_transient(# NO_RETRY... # HUB_DIR... lambda:
api.upload_folder(` had the NO_RETRY waiver two lines up — the scanner
flagged the call anyway. Related trap: the wrap-detection regex
`\b(?:retry_transient|_retry_upload)\s*\(` cannot match an
underscore-prefixed LOCAL port (`_retry_transient(` — `_` is a word char,
no boundary), so standalone-port scripts NEED the waiver even though they
genuinely retry.

Sibling scanner (#2333 R2 g2): `check_hub_verify_retry` is AST-based with NO
wrap detection at all — every Load-ctx `.list_repo_tree(`/`.list_repo_files(`/
`.file_exists(` attribute is flagged even when it sits DIRECTLY inside
`retry_transient(lambda: list(api.X(...)))`; the waiver is the only escape.
Placement: the hit lineno is the attribute node's line, so the waiver comment
may legally sit on the line above it INSIDE the call parens. `list()`
materialization inside the retried lambda is the load-bearing detail (cursor
pagination retried); certify via the check function with `scripts_dir=` on the
parent blob (expect 1 error) vs HEAD (expect 0).

**How to apply:** reviewing any waiver-placement fix or a standalone script
porting repo helpers under new (esp. `_`-prefixed) names: (1) certify with
the scanner FUNCTION as a live probe on the parent-commit blob
(`git show <sha>^:<file>` → expect the error) AND on HEAD (expect clean) —
see [[fails-pre-fix-probe-parent-commit]]; (2) probe every OTHER waiver
token the move displaced (its own presence helper); (3) a NO_RETRY-style
waiver on a call that IS wrapped in an equivalent local retry port is
legitimate — the waiver documents blindness of the wrap regex, not a
retry opt-out.
