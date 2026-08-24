---
name: lint-scan-roots-differential-probe
description: Certify a lint-fix commit by calling the check function with scan_roots on parent-blob vs fixed-blob temp dirs; workflow_lint needs sys.modules registration to exec standalone; raw U+2028/NEL in a Bash heredoc trips the control-char guard — Write the probe file instead
metadata:
  type: feedback
---

For a commit whose message claims "fix lint red" on check X, certify BOTH directions without mutating the worktree: extract `git show <sha>^:<file>` and `git show <sha>:<file>` into two temp dirs and call the check function directly with its `scan_roots`/root-override test hook (`check_jsonl_splitlines(scan_roots=(tmpdir,))` — many workflow_lint checks carry one). Expect parent → errors at exactly the diff's changed lines, fixed → 0. Complements [[fails-pre-fix-probe-vs-parent-commit]] (which certifies tests; this certifies the lint itself).

**Why:** a tree-wide no-flags lint run at HEAD only proves "green now" — it cannot attribute the red to the parent blob or confirm the fix (not an allowlist/waiver) cleared it; the #823 g5 round also confirmed the file was absent from the check's legacy allowlist, which is what made the red binding (#823 fu round r14, 2026-08-23).

**How to apply:** two mechanics. (1) `workflow_lint.py` cannot be exec'd via importlib without `sys.modules["workflow_lint"] = m` BEFORE `exec_module` (module-level `@dataclass` resolution hits `sys.modules.get(cls.__module__)` → AttributeError). (2) A Bash heredoc containing raw U+2028/U+2029/NEL is REJECTED by the tool ("command contains control characters"); the Write tool accepts them — author the probe as a file with the raw chars (verify presence with a `chr(0x2028) in text` read-back), then `uv run python <file>` from the worktree cwd. Producer-parity check for splitlines fixes: text-mode `for line in fh` splits only on \n/\r/\r\n (NOT U+2028/NEL), so a line-iterating shard producer is automatically consistent with a `split("\n")` reader.
