---
title: 'daily-fix: AGENT_SPEC_SIZE_GRANDFATHER is a chronic merge-co'
kind: infra
tags:
- wf-fix
- wf-fix-fp:222fa7714bf6
- daily-auto-filed
created_at: '2026-07-27T07:14:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): every concurrent workflow-fix
  session edits adjacent entries of the same Python dict literal, producing repeated
  3-way merge conflicts, rejected pre-commit hooks and repeated full lint-gate re-runs'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 2 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Move the per-file agent-spec size caps out of the shared `AGENT_SPEC_SIZE_GRANDFATHER`
Python dict literal in `scripts/workflow_lint.py` into a line-mergeable form (a
one-entry-per-line data file, or a mechanically derived cap) so concurrent workflow-fix
sessions never edit the same lines.

## Workflow gap

- **Bug observed:** Every workflow-fix session that grows an agent spec must raise that file's cap in the same 85-line Python dict literal, so four sessions doing so on 2026-07-26 (#1692, #1693, #1698, #1699/#1702) collided on adjacent entries at three separate Step 10d merges.
- **Why it is a workflow gap:** The ratchet's storage format — a single Python dict with interleaved multi-line comment histories — makes a semantically independent one-number edit textually conflicting for every concurrent session, and the reconcile is hand arithmetic over two cap histories where a wrong number silently un-protects a spec file.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c 'AGENT_SPEC_SIZE_GRANDFATHER' scripts/workflow_lint.py` → 9 hits (dict literal opens at L11761; 8 per-file entries inside an 85-line block, counted with a Python slice of the literal); `ls tests/agent_spec_size_caps.txt` → No such file (no data-file form exists); `cat .gitattributes` → 2 `merge=union` lines, both `tasks/**/*.jsonl`, none covering `scripts/`; `git log --oneline --since='7 days ago' -- scripts/workflow_lint.py` → 18 commits, all check additions or cap bumps, none relocating the table (2026-07-26)

## Evidence

- Session `6b3fca14`, 09:38:50Z: `"MERGE_RC=1 | Auto-merging .claude/agents/code-reviewer.md | Auto-merging scripts/workflow_lint.py | CONFLICT (content): Merge conflict in scripts/workflow_lint.py"`, followed by `"Main also raised the cap in parallel (#1692 → 112_500). Post-merge file measures 120,709 B — need 121_300 cap. Resolving conflict + reconciling both histories."` Recovery ran 09:38→09:50.
- Session `2de5253e`, 14:27:50Z and 15:43:52Z: two further recovery cycles on the same table, the second including a 28-minute pre-push lint-gate re-run. Three merge-conflict firing events across the two transcripts (1 + 2).
- Session `8571eca6`, 13:41:52Z→15:03:21Z: the same adjacent-entry collision surfaced first as a pre-push lint-gate BLOCK on a file the branch never touched — `"Gate BLOCK: 1 new lint failure … The file is NOT in the own-diff — this is a delta caused by the branch's own edit to scripts/workflow_lint.py"` — then again as a real merge conflict. Its `epm:merged v1` records `"merge_attempts: 3 (attempt 1 blocked on pre-push lint gate — experiment-implementer.md ratchet cap conflict from a 3-way lint-copy merge failing on adjacent constants …)"`.
- Measured cost on that one branch: three pre-push lint-gate runs totalling ≈107 min of gate wall-clock (13:07→13:41, 13:45→14:23, 14:28→15:02) inside a 4h41m session, one rejected pre-commit hook, 2 extra commits, and 3 `gh pr merge` attempts. Across both transcripts the day's pure merge-mechanics overhead is a transcript-derived estimate of ~45 min beyond that branch. No work was discarded — all branches landed correctly.
- The dict's current shape is the aggravating factor and is verified in-tree: 8 live entries spread over 85 lines, each preceded by a multi-line measured-size history comment (e.g. `"code-reviewer.md": 121_300` sits under a 14-line prior-cap chronicle), so two sessions raising two different files still touch neighbouring lines.
- unverified hypothesis — verify at plan time: the 2026-07-26 lint-gate BLOCK arose because the gate's 3-way lint-copy merge conflicted and its fallback kept the branch copy, linting main's grown file against the branch's stale cap. The BLOCK text is quoted above; the fallback mechanism was read from the session's own diagnosis, not from the gate implementation.

## Proposed change

- In `scripts/workflow_lint.py`, replace the `AGENT_SPEC_SIZE_GRANDFATHER` dict literal (L11761) with a loader over a one-entry-per-line data file (e.g. `tests/agent_spec_size_caps.txt`, `<name> <cap> # <reason>`), so two sessions raising two different files touch two different lines and git auto-merges both.
- Keep the cap-history prose out of the data file's per-entry line (a trailing `# <reason>` only), or move the histories to a sibling doc — an 85-line comment-interleaved block is what makes adjacent entries conflict even when the numbers do not.
- Alternative to evaluate at plan time: derive the cap mechanically (measured size + fixed headroom, written to a generated file) so a spec-growing edit needs no manual cap bump at all; the current pre-commit `headroom > 3000` rule (which fired during the 2026-07-26 recovery) would need re-siting under this option.
- A `.gitattributes` `merge=union` driver on the caps file is a partial mitigation only: the file's own header records the verified 2026-07-01 finding that server-side `gh pr merge --rebase` does not honor user-defined merge drivers. It would cover the local `git merge origin/main` recovery path — which is where all three 2026-07-26 conflicts actually fired — but not a server-side refusal. Choose the data-file split as the primary fix, union as an optional belt.
- Update the readers at L11889 / L11903 / L11936 and the `verify_plan.py` + Step 10d references to the new source, keeping the FAIL/WARN/stale-ratchet semantics byte-identical.
- Add a pin test that the data file parses to the same mapping the dict encoded at the migration commit, so the relocation cannot silently drop or alter a cap.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- `scripts/verify_plan.py` (references `AGENT_SPEC_SIZE_GRANDFATHER`), `.claude/skills/issue/SKILL.md` (Step 10d / ratchet references), `tests/test_workflow_lint.py` (cap pin tests), and the new caps data file

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- sha-verify (filing-time, #1467): `6b3fca14` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `8571eca6` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 222fa7714bf6

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: D-P1, E-P2.
