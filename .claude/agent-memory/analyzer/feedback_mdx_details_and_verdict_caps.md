# MDX details-fence shape + ALL-CAPS verdict words (#2223, 2026-08-14)

Three mechanical-gate lessons from the #2223 promotion:

1. **`<details><summary>…</summary>` inline on ONE line fails verify_task_body's
   real-MDX-parse check** ("Expected a closing tag for `<details>`"). Write the
   exemplar (v4-657) shape: `<details>` on its own line, `<summary>…</summary>`
   on the next, blank line, content, blank line, `</details>`.
2. **ALL-CAPS verdict words in prose FAIL the discipline audit's `verdict_caps`
   class** — `INDETERMINATE` tripped it twice; write "an **indeterminate**
   verdict" (lowercase, bold ok). Same family as the `registered <noun>` ban
   (`pre_reg` class), which also matches "Registered verdict:" and
   "registered gate" — write "the plan's stop gate" / "the verdict rule fixed
   in the plan".
3. **`task.py address-concern --summary` caps at 200 chars** (exit 2 over it;
   `--summary-file` for longer). Lens 14 open concerns can be cleared either by
   body acknowledgment OR by address-concern rows — for a concern that is MOOT
   (the arm it caveats never ran), address-concern with the moot rationale is
   cleaner than body prose.
