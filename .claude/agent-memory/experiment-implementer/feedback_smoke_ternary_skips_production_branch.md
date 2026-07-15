---
name: --smoke ternary skips the production branch
description: "'A if args.smoke else MODULE.CONST' leaves the production branch unexecuted by the smoke — an invalid attr there survives a green smoke and crashes hours into the real run (#825)"
type: feedback
---

A conditional of the form `A if args.smoke else MODULE.CONST` (or any
`if args.smoke:` fork) leaves the PRODUCTION branch unexecuted by the
end-to-end smoke — Python evaluates only the taken ternary branch — so an
invalid module attribute / name on the production side survives a green
smoke gate and crashes hours into the real run.

**Why:** #825 naturalistic-single-turn: `expect_n = None if args.smoke else
fit_cells.N_TRACK_S` — `N_TRACK_S` lives in `experiments.issue_825.common`,
not `fit_cells`. The smoke ran the whole pipeline green; the production run
died with `AttributeError` at `[phase=contrast]` after ~2.2h on 2×A100,
losing the un-uploaded turnstore shards with the instance.

**How to apply:** when a script gates behavior on `--smoke`, resolve
production-only module constants at IMPORT TIME (module-level binding, e.g.
`EXPECT_N = common.N_TRACK_S` at top level) or add an import-time
assertion — the smoke's import of the script then exercises them. When
reviewing/writing a smoke report, check for `args.smoke` ternaries whose
else-branch dereferences a module attribute and verify each resolves
(`python -c "import mod; mod.ATTR"`).
