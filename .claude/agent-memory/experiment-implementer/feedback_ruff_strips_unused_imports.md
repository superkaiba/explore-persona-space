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
- **For imports needed at a CLASS DECLARATION (e.g. subclassing):** edit
  the class line FIRST (`class Foo(BaseFromX):`), THEN add the import.
  If you add the import first, the post-Edit ruff hook strips it because
  `BaseFromX` isn't yet referenced. Two-step ordering matters. (Bit me
  on task #405 round 5: I added `from transformers import TrainerCallback`
  before changing `class ProbePanelLogprobCallback:` to subclass it; the
  hook stripped the import. Re-adding it AFTER the class was already
  declared with the parent name kept it because the reference now
  exists.)
- Either way, after editing, **always run `ruff check` + `ruff format`
  once more** to confirm the new imports survived. Your edit succeeding
  doesn't mean the format pass kept it.
- **Writing a big file in CHUNKS (Write tool first chunk + Bash heredoc
  appends):** the PostToolUse formatter hook fires on the FIRST Write and
  strips every import the later (not-yet-appended) chunks need — Bash
  appends do NOT re-trigger the hook, so the file ends up with F821s.
  Either write helpers-before-imports-users in one chunk, or restore the
  full import block AFTER the last append and re-run `ruff check` (which
  keeps them once references exist). Bit me on task #536 (torch/math/csv/
  ast + the compute_cosine_matrix import all stripped from chunk 1).
- **Threading new names into an existing `from X import (...)` block across
  multiple Edit calls:** same trap — if the import-block Edit lands before
  the usage Edits, the hook strips the new names and you get a wall of
  F821s at the next lint. Robust fix (3 hits on task #570): AFTER the
  usage edits exist, rebuild the whole block programmatically —
  `names = parse existing block; merged = sorted(set(names) | set(add));
  rewrite block` via a small Python splice — then `ruff check --fix` for
  I001 sorting. Cheaper than fighting Edit ordering on 80-name blocks.
- **Round-N revision edits over an existing script:** same trap again on
  task #601 round 2 — added phase0_lib imports to `main()` in one Edit,
  the usages in the NEXT Edit; the hook stripped the whole new import
  group in between (F821 x6 at lint). Plan multi-edit sequences so the
  Edit that introduces an import ALSO introduces at least one usage
  (or extract the new logic into a helper function whose body carries
  both the imports and the usages — that's what fixed it here).
- **Top-level import SHADOWED by function-local imports of the same name:**
  if every use site still carries its own lazy `from X import name`, the
  new top-level import is genuinely unused to F401 and gets stripped even
  though `name` appears all over the file (task #606 r3: promoting
  `_retry_transient` from three lazy sites to module top — the hook
  stripped the top import because the three local imports still shadowed
  it). Ordering fix: REMOVE the lazy local imports FIRST (or in the same
  Edit batch), then add the top-level import; then `ruff check` to
  confirm it survived.
