---
name: verbatim-port-commit-review-recipe
description: Certify a claimed-verbatim branch port by hash-vs-source-tip + commit-range provenance + excluded-commit disjointness; attribute post-commit deltas before flagging divergence; ruff in situ (never /tmp blobs)
metadata:
  type: feedback
---

Certifying a "verbatim port from origin/<branch>" commit takes five probes, ~6 tool calls (#2479 R1 g1):

1. **Hash vs source tip:** `git show <commit>:<f> | sha256sum` vs `git show origin/<branch>:<f>` per file — the whole verbatim claim in one loop.
2. **Commit-range provenance:** `git log --oneline origin/<branch> -- <files>` must land inside the plan-named commit range, AND any plan-EXCLUDED commit (e.g. an unreviewed sibling like #2479's scaffold-splice `b5cabfa929`) must show a disjoint `--name-only` file set — tip-hash equality alone would silently bless excluded content that touched the same files.
3. **Clean-add check:** files absent at merge-base + no untracked repo-root twin (see [[untracked-twin-add-certification]]).
4. **Attribute post-commit deltas before flagging:** a round `name-status` file showing `A` can hide LATER round commits modifying the same file (live line count > commit's added count is the tell). `git diff --stat <commit>..HEAD -- <files>` + `git log <base>..HEAD -- <files>` attributes each delta to its owning commit — divergence in a later unit is that unit's reviewer's scope, not a verbatim-claim violation.
5. **Ruff in situ, never on /tmp blob extractions:** pyproject `per-file-ignores` are path-scoped, so ruff on `/tmp/<blob>.py` (even with `--config pyproject.toml`) fabricates findings the in-situ run doesn't have (observed: RUF100 "unused noqa (non-enabled: PLC0415)" on a file that passes clean at `scripts/`). For a file unchanged since the commit, in-situ HEAD check == commit-blob check.

**Why:** the port IS reviewed here (plan: "the port is reviewed by this issue's code-review"), but a verbatim port's content risk lives in provenance (what the bytes came from) not in re-litigating parent design choices already covered by the parent's own review rounds (inline "r1/r2 Minor" annotations are the tell). Parent-namespaced constants (HF prefixes, pins, stage roots, hardcoded cell tables) in a verbatim port are a HANDOFF note to the parametrizing units' reviewers, not a g-scope finding.

**How to apply:** any split-review group whose commit message claims "verbatim"/"port ... from origin/<X>".
