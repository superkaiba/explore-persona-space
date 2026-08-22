---
name: doc-pin-invocation-counting
description: Uniqueness pins over spec docs must count INVOCATIONS across ALL fence languages + ALL command spellings, not literal-bearing blocks of one fence language
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

**Fix shape:** count regex OCCURRENCES (`dispatch_issue(?:\.py)?\s+launch\b`)
summed across ALL fenced blocks (`^```\w*\n(.*?)^```$` — any language tag,
bare included), assert `n == 1`, then identity-check the one carrying block.
Prose mentions OUTSIDE fences stay unpinned (not operator-copyable). Verify
each escape RED by monkeypatching the composed-text helper with the mutated
text and calling the test function directly — and show the OLD pin green
under the same mutation (the escape demonstration reviewers ask for).

**How to apply:** any time a pin quantifies over "blocks that contain X" in
a markdown spec, re-derive it as occurrence-counting over all fences + all
valid spellings of X; check for four-backtick nested fences first (they
break the block regex — grep `^````). Related: [[bash-span-pin-recorder-argv]]
(the executing-shell sibling for argv-level claims).
