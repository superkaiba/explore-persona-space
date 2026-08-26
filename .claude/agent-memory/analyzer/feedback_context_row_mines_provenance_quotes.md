---
name: context-row-mines-provenance-quotes
description: verify_task_body check 17 extracts the first verbatim-quoted fragment from original-body ## Provenance (even inside a Data bullet) — "origin prompt not recorded" FAILs whenever any quote exists
metadata:
  type: feedback
---

Check 17 (Context provenance row) does not only read frontmatter: it mines
the original-body `## Provenance` section for the first verbatim-QUOTED
fragment (any `"..."` inside a decision-record bullet counts — #2564's was
a 34-char user answer `it's fine for you to generate them` inside the
**Data:** bullet) and requires that fragment, whitespace-normalized, inside
the `**Context:**` row.

**Why:** writing `origin prompt not recorded` because the frontmatter has
no origin-prompt field is a guaranteed FAIL whenever the Provenance
decision record quotes ANYTHING — the verifier treats the mined quote as
the recorded originating prompt.

**How to apply:** before drafting the footer, grep original-body.md
`## Provenance` for double-quoted fragments; if any exist, code-span the
first one verbatim in the `**Context:**` row ("whose recorded verbatim
originating prompt ... is `<fragment>`") and never write the
not-recorded clause. Related: [[fold-context-prompt-and-open-concern-acks]].
