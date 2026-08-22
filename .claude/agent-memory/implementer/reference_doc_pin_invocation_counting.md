---
name: doc-pin-invocation-counting
description: Uniqueness pins over spec docs count INVOCATIONS on an executable view of fences recognized at ANY indentation (backtick+tilde, any info string); claim only what the detector does
metadata:
  type: reference
---

A "exactly ONE launch site" pin implemented as `len([b for b in bash_blocks
if LITERAL in b]) == 1` is escapable THREE ways, all leaving it green
(#2263 r7, concerns `parent-launch-uniqueness-syntax-escape` +
`launch-block-fence-language-escape`):

1. a SECOND exact invocation added inside the already-counted block
   (block count stays 1);
2. an alternate valid command SPELLING (`python -m scripts.dispatch_issue
   launch` vs `scripts/dispatch_issue.py launch` — `scripts/__init__.py`
   exists, so the module form runs);
3. an alternate fence LANGUAGE (` ```sh ` / ` ```console ` / bare ` ``` `
   when the scan regex is ` ```bash `-only).

**And the r7 fix (occurrence-count over column-zero `^```\w*` fences) was
STILL escapable both ways (#2263 r8, reconciler r6):** an INDENTED fence
(list-item content — real spec anchors sit at 2 AND 5 spaces, so a
CommonMark-faithful `{0,3}` indent bound is also wrong) stays GREEN, and a
COMMENTED launch-shaped line inside a counted fence goes RED (occurrences
are not invocations — false in both directions).

**Fix shape (r8, landed as `_fenced_blocks` + `_executable_view` in
`tests/test_verify_carryover_inputs.py`):** (1) recognize fences at ANY
indentation — `^[ \t]*(`{3,}|~{3,})` + any info string (backtick-fence
info strings containing a backtick are not fences, CommonMark 4.5),
same-marker close of >= the opening run length, unclosed runs to EOF;
(2) count invocations on each block's EXECUTABLE VIEW — strip `#`-comment
text per line FIRST, THEN join backslash-newline continuations (order is
load-bearing: bash comments do not continue across a trailing backslash);
(3) regex `dispatch_issue(?:\.py)?\s+launch\b` covers both spellings;
(4) pin the detector's semantics with a SYNTHETIC-TEXT unit test (both
anchor indents, commented-line zero-count, tilde/4-tick/info-string/
continuation/module-spelling/unclosed/prose cases) so a detector
regression fails without a live spec mutation; (5) the docstring claims
ONLY what the code does — no "ALL"/"every" unless the code earns the
word; name the known residuals (`#` in quoted strings under-counts,
heredoc bodies over-count -> fail-closed).

**How to apply:** any time a pin quantifies over "blocks that contain X"
in a markdown spec, re-derive it as invocation-counting over
any-indentation fences + all valid spellings of X, and run the mutation
battery at the REAL anchor sites the pin exists to guard (an unindented
specimen certifies nothing about indented list content). Related:
[[bash-span-pin-recorder-argv]] (the executing-shell sibling for
argv-level claims).
