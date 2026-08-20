# Word-cap relief on revision rounds: new ### section + Methodology placement

Context: #1074 `install-dose-extension` interpretation r2 — 5 reconciled fixes
had to land in a `### <result>` block already at 175/180 words.

Pattern that works:

- **Do not cram.** Measure the target block first (`verify_task_body.py --file`
  reports per-result counts). A block at ~175 words cannot absorb multi-fix
  content — a trim-with-add tops out around ±10 words.
- **Add a NEW `### <result>` for the robustness/adjudication reads** (graded
  trajectory, censoring bounds, schedule-decay). It gets a fresh 180-word
  budget but REQUIRES its own figure (one-result-one-figure + three-beat) —
  budget the figure gen/verify/commit/pin loop (~6 tool calls).
- **Recipe/mechanics prose goes under `## Methodology`** (Evaluation / Data
  extraction slots) — cap-EXCLUDED, so bound definitions, assert-parity notes,
  and committed-JSON links live there, keeping Results blocks numbers-only.
- Em-dashes and standalone `—` tokens COUNT as words under `split()`;
  semicolons instead of ` — ` buy a token in 30-word Takeaways bullets.
- **Image alt-text lines COUNT toward the 180-word per-result cap** (only
  `>`/`|`/fence/details lines are stripped) — a critic-mandated alt-trend
  clause on 2-3 figures adds 12-20 words, so trim-with-add BEFORE editing.
- **A mandated in-section disclosure line (e.g. `Per-unit exemption:`) can
  live inside the figure's `>` caption** — cap-EXCLUDED yet still in-section
  and greppable; keep the caption ≤60 words or a NEW WARN class fires that
  the body's acknowledgment paragraph must then name (#1739 r3: exemption
  caption landed at 58 words with sections at 176/180).

Bonus finding worth remembering: an adversarial DRAW-level judge-censoring
bound (every unscored draw := 100, re-binarize) is cheaply recomputable from
`judge_raw.json` `all_scores` and can COINCIDE with the item-level bound
(#1074: both 0.467 at the peak checkpoint) — compute it instead of estimating
"~0.6 possible" from aggregates; the critique's aggregate-only estimate was
2x too pessimistic.
