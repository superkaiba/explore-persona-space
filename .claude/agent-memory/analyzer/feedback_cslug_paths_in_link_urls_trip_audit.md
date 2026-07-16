# C-slug artifact paths in raw link URLs trip the condition_labels audit

`audit_clean_results_body_discipline.py`'s `condition_labels` rule scans with
`strip_code` (inline backticks + fences stripped, `<details>` blocks stripped,
table rows blanked) — but markdown link **URLs** are none of those, so a pinned
HF URL whose path contains a `C2-...`-style segment (e.g.
`.../adapters/issue1090_fu3/C2-icl-con-impolite-claude/checkpoint-8`) matches
`\b[CcHhP][1-9]` and FAILs the audit even though the backticked link TEXT is
exempt.

Fix pattern (#1315 r3, mirrors #1090's own footer): link the **prefix-level
tree** URL that stops above the C-slug directory
(`.../tree/<rev>/adapters/issue1090_fu3`) and put the full per-cell paths in
inline backticks next to it. Bonus: verify_task_body's "HF-adjacent backtick
file claims exist under the pinned tree" check then actively verifies the
backticked paths against the pinned tree. Links inside `<details>` sample
blocks are stripped and may keep direct blob URLs.
