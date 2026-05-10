---
name: workflow.yaml AUTO-GENERATED fence pattern
description: Pattern for embedding YAML-derived tables in markdown via fenced auto-generated blocks
type: feedback
---

For repos that have a structured YAML (or any structured config) as the source of truth and want to keep prose docs (CLAUDE.md, SKILL.md, etc.) in sync, the proven pattern is:

1. Wrap the section to auto-generate in a uniquely-named fence:
   ```
   <!-- workflow.yaml: AUTO-GENERATED (table-id) -->
   ...rendered table...
   <!-- /workflow.yaml: AUTO-GENERATED -->
   ```
2. Render via a CLI script (e.g. `scripts/workflow_lint.py --emit-tables`).
3. Pre-commit hook in `--check-tables` mode rejects drift between the YAML and the rendered table on disk.
4. Keep prose THAT REQUIRES NUANCE outside the fence (auto-gen flattens
   field-level nuance — e.g. "experimenter agent (pod ops + monitoring);
   type:experiment only" can't be encoded in a single YAML `description:` field
   without losing fidelity).

**Why:** when invoked from the CLI, the rendered output is always reproducible
from the YAML. When the YAML is hand-edited, drift fires immediately at commit
time. When the prose needs richer context, it lives outside the fence and is
edited by hand.

**How to apply:** any time a future task asks to "make X the single source of
truth for Y prose docs", reach for fenced auto-generated blocks rather than
trying to migrate the entire prose section to a structured config. Keep the
fence narrow (just the table) and leave the explanatory prose human-edited.
This was what shipped successfully in #320 §1; #320 §3-§5 deferred to follow-up.
