---
name: encoding-corruption-probe-placement
description: json.loads(bytes) BOM-sniffs — a leading b"\xff\xfe" probe raises JSONDecodeError there, not UnicodeDecodeError; match probe placement to the consuming call, and certify binding via in-worktree sed-mutate + git -C restore
metadata:
  type: feedback
---

When reviewing encoding-corruption regression tests (any test writing invalid
UTF-8 to prove a `UnicodeDecodeError` guard fires), the probe's BYTE PLACEMENT
must match the consuming call, or the test goes green for the wrong reason:

- `Path.read_text()` (no encoding arg): strict locale decode, NO BOM sniff —
  a LEADING `b"\xff\xfe"` raises `UnicodeDecodeError` at position 0. Leading
  form is correct here.
- `json.loads(bytes)`: `detect_encoding()` BOM-sniffs first — a LEADING
  `b"\xff\xfe"` reads as a UTF-16-LE BOM, decodes, and raises
  **JSONDecodeError** (already caught by the pre-fix tuple ⇒ mutant survives,
  test does not bind). The bad bytes must sit MID-PAYLOAD after a `{` head so
  detect_encoding picks UTF-8 and the strict decode raises.

**Why:** #2168 r1 g3 — the plan's literal `b"\xff\xfe..."` template was wrong
for the one `json.loads(resp.read())` site; live probe confirmed leading →
JSONDecodeError, mid-payload → UnicodeDecodeError. The mutant failure OFFSET is
the tell (position 0 = read_text path, position N>0 = mid-payload json path).

**How to apply:** for each such test, (1) identify whether the site decodes via
read_text or json.loads(bytes); (2) run the 3-line probe (`json.loads` leading
vs mid, `read_text` leading) rather than reasoning from memory; (3) certify
binding independently with the guard-compliant in-worktree mutation loop:
`sed -i '<line>s/, UnicodeDecodeError//' <wt>/<file>` (or delete the tuple's
own line) → run the one test expecting FAIL → restore via
`git -C <worktree> checkout -- <file>` (never bare `cd`+`git checkout` —
the repo-root guard blocks the whole compound; per-clause `git -C` is the
recognized shape) → confirm the worktree porcelain is clean after. Also check
the guarded call's CALLEE chain is internally unguarded (an inner
try/suppress would eat the error before the tuple under test).
Extends [[fails-pre-fix-probe-parent-commit]].
