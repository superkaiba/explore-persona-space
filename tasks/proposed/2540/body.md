---
title: 'verify_task_body: resolve same-repo /commit/<40-hex> URLs so a hand-extended
  SHA cannot ship'
kind: infra
tags: []
created_at: '2026-08-24T14:44:38Z'
has_clean_result: false
parent_id: 823
origin_prompt: 'Surfaced by the /issue 823 orchestrator at clean-result round 8: a
  hand-extended commit SHA in the body''s Repro footer 404''d and survived seven critic
  rounds.'
workflow: v1
---
## Goal

Add a mechanical check that every same-repo `/commit/<40-hex>` URL in a clean-result
task body resolves to a real git object, so a hand-extended SHA cannot ship.

## Why

Found at #823 round 8. The body's `**Repro:**` footer carried
`https://github.com/superkaiba/explore-persona-space/commit/84633d46c6cd23dcd75be9ffc9b0f7815822f7ce`.
The 10-character display prefix `84633d46c6` was correct; the trailing 30 hex
characters were hand-extended. The real commit is
`84633d46c6e2a52d678746521ba35a001c044845`. The URL 404s.

It survived seven prior critic rounds. The orchestrator's own link sweep missed it
because that sweep matched only `blob/` and `raw/` URL forms, and the sweep's
result was then reported as "every git-pinned link resolves" — broader than its
actual coverage — and handed to the Codex twin as a do-not-redo instruction, so the
twin's Lens 5 PASS inherited the gap. Only the round-8 Claude critic, which had no
such instruction, caught it.

## Scope

`verify_task_body.py` has no check that resolves same-repo commit URLs. Add one:
extract every `github.com/<owner>/<repo>/commit/<40-hex>` whose owner/repo matches
this repo, and `git cat-file -e` each. FAIL on a bad object.

Two things the implementation must get right, both learned the hard way at #823:

1. **Do not FAIL on label-vs-URL mismatch.** Four pairs in #823's footer carry a
   branch SHA as the link LABEL and the main-landed SHA as the URL target. Verified
   patch-equal under `git patch-id --stable`, so that is legitimate as-run/landed
   dual provenance, not an error. Only object EXISTENCE is in scope.
2. **HF links carry HF revision SHAs, not git SHAs.** A git-side probe cannot
   resolve them and will report them missing. Scope the check to same-repo GitHub
   commit URLs only.

## Acceptance

- A fixture body with a hand-extended same-repo commit SHA FAILs.
- A fixture body with a patch-equal branch-label/main-URL pair PASSes.
- A fixture body with HF-revision links PASSes.
