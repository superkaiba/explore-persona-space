---
name: codex-scope-drift-on-repeat-findings
description: Codex code-reviewer re-flags a previously-discussed flag/string by name without re-checking which script/scope the round-N fix actually targeted; treats lexical match as protocol regression
metadata:
  type: feedback
---

When a prior round's binding blocker mentioned a flag/string by name (e.g. "remove `--no-upload`"), Codex on the next round will sometimes grep the whole diff for that string, find a still-present instance in a DIFFERENT file/script (governing a DIFFERENT artifact category), and escalate to "Critical (block merge)" framed as "round-N no-regression check fails." The lexical match is real; the protocol violation is not.

**Why:** Codex's per-round context lacks fine-grained recall of WHICH file the prior fix targeted. It reasons from the verbal description of the round-N fix ("drop `--no-upload`") plus a grep, without re-reading the round-N commit to confirm scope.

**How to apply:** When Codex flags "round-N regression" as critical:
1. Read the round-N commit (`git show <sha> --stat`) to see what files the fix actually touched.
2. If the still-present flag is in a DIFFERENT file than the round-N commit modified, the "regression" framing is wrong. Check whether the flag in the new location is governed by a separate policy (CLAUDE.md sanction, different artifact category, opt-in dry-run default).
3. If the default code path is correct (e.g. flag defaults to upload=ON, opt-in to skip), the finding is at most a style nit, not a science blocker.
4. Concrete incident: task #382 round 3, Codex flagged `scripts/generate_issue382_marker_install.py:1034`'s `--no-upload` as critical regression of round-2 minor 7. Round-2 commit (`bde24468`) only touched `scripts/eval_issue382.py` (a different script, different artifact: raw-completion upload vs dataset upload). CLAUDE.md Upload Policy explicitly sanctions `--no-upload` as a dry-run escape hatch in data-gen scripts. Default is `default=False` (upload=ON); `upload_dataset_directory` is fail-loud on the default path. PASS.

Distinguish from [[feedback_codex_conflates_marker_format_with_code]] (Codex applies rubric to the wrong artifact entirely) — this pattern stays within the code diff but mis-attributes which prior round's fix scope applied.
