---
name: Ruff strips unused module-level imports
description: Top-level `import lorem` (or any module imported only for side effects) gets removed by ruff format/post-Edit hooks; inline-import inside the user function instead.
type: feedback
---

When you add a top-level `import X` and don't reference `X` elsewhere in the
file, the project's ruff config (or the post-Edit formatter hook) **removes
it on the next pass**. This bit me on issue #280 v7 when I tried to add
`import lorem` at the top — both my edits were wiped silently.

**Why:** ruff's `select = ["E", "F", "I", "UP"]` includes `F401`
(unused-import). The post-Edit format hook runs `ruff format` which auto-
removes such imports.

**How to apply:**
- If you need a top-level import for side effects only, add a `_ = X`
  reference right after it. Crude but effective.
- Better: **lazy-import inside the function** that uses it. This is what
  the v7 patch ended up doing for `from lorem.text import TextLorem` —
  imported inside `_generate_garbage_assistant_local`, where it is also
  *used*, so ruff is happy and the import isn't wasted on module load.
- Either way, after editing, **always run `ruff check` + `ruff format`
  once more** to confirm the new imports survived. Your edit succeeding
  doesn't mean the format pass kept it.
