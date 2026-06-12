---
name: Delta-scoped amendment-round prompting
description: Composing lens prompts when the brief scopes critique to a vN amendment delta or same-issue follow-up — verbatim scope note, do-not-relitigate guard, delta-scoped verdict line, cross-boundary check, variant briefs, followup-label tmp suffix
type: feedback
---

**Rule.** When the brief carries a "DELTA SCOPE NOTE (include verbatim)" for an amendment round:

1. **Paste the scope note verbatim** right after THE BAR, before the lens items — it governs scope and usually carries its own verdict-line format (`**Verdict (<lens>, delta-scoped): APPROVE|REVISE|REJECT**`). Use THAT line in place of `**Rating: ...**`; keep the epm marker tags unchanged (the orchestrator validates tags, not wording). If the brief gives verdict SEMANTICS but no line format, mint the delta-scoped line yourself and paste the semantics as "Verdict definitions for this round".
2. **Do-not-relitigate guard in BOTH slots** — PRIOR CRITIQUES ("vN-1 was ensemble-APPROVED; do not re-litigate") and the PLAN TEXT intro ("critique ONLY the vN delta"); without it Codex drifts plan-wide. User-overridden descopes go on the SETTLED list as "review the IMPLEMENTATION, never the choice"; forbid proposing the reverted value back.
3. **Always include one cross-boundary sub-question**: can the delta silently move a previously-registered headline (new eval columns entering a registered leaderboard/CV aggregation)? The one place a "surgical" delta is conclusion-changing for the parent.
4. **One-line verdict per labeled sub-question** (`### Sub-question verdicts` section) — keeps rounds short, makes the reconciler mechanical.

**Why (example):** #537 v5 (2026-06-09) — brief overrode the verdict line and capped scope to 2 new eval-only contexts; the main leak was the delta columns entering the v4-registered leaderboard, caught by the cross-boundary item.

**Variant briefs (one-liners):**

- *Divergence-block* (plan documents a contradiction in its approved scope): first two sub-questions = divergence LEGITIMATE (minimal edit preserving the hypothesis) + BOUNDED (nothing rides along); add a circularity item for any data-dependent pick (does the gate ever see the y-axis); inline the fact-checker's CONFIRMED on the contradiction premise. (#480 followup-2)
- *Composition-of-two-executed-designs*: (a) composition second-variable vs either parent; (b) ARM-SYMMETRY of every carried train/eval mismatch (symmetric cancels in paired d_seed, asymmetric is the fatal class); (c) smoke CONTENT-vs-count adequacy (does the smoke assert composed rows' encodings/parity, not just counts); (d) cross-boundary check. Inherited endpoint-only capture → pre-scope the trajectory item to "does its absence break THIS endpoint contrast". Paste `/tmp/issue-<N>-followup-scope-<label>.md` verbatim when it exists. (#464)
- *Borrowed-construction loss-surface mismatch*: "does the borrowed construction still instantiate the manipulated variable under the new loss, and is the variable's naming (data construction vs loss) honest enough for either outcome?" + near-on-policy-negatives gradient question. (#552 v4)
- *Gate-split / threshold-re-grounding* (unsatisfiable HALT gate split, thresholds re-derived FROM the data they gate): (a) RETAINED FORCE — name a corruption mode the OLD gate HALTed on that the split now passes; (b) CIRCULARITY-vs-re-registration — "binds future reads (relaunch recomputes)" vs "consumed once on the data that set it, so it can never fail"; have Codex read the driver's resume semantics. When the amendment fixes a mechanism this twin itself REVISE'd, say "this is YOUR lens's thread" — makes Codex read the actual driver. Pre-settle the CHOICE of mechanism, leave its correctness live; smoke-verifiability residue for new runtime machinery; watch stale-sentinel resume skips. (#601 v3 r2)

**How to apply:** any round whose brief says "vN amendment of approved vN-1" / includes a verbatim scope note. Pair with the stale-tmp-files rule (amendment rounds are when stale files sit at canonical paths). Same-issue FOLLOW-UP rounds (no version number) suffix tmp paths with the followup label: `/tmp/codex-critic-<N>-<followup_label>-<lens>-{prompt,output}.md`.
