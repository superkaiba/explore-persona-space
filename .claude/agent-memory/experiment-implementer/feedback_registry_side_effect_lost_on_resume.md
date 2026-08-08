---
name: Register dynamic registry entries at point of use, never via phase side effects
description: A resumed process fast-forwards earlier phases, losing their in-process registry-registration side effects — register dynamic CONTEXTS/ids immediately before each consumer (#1315 r6)
type: feedback
---

A dispatcher whose `ModelOrganism`/`CONTEXTS` consumers rely on an EARLIER phase's in-process registry-registration side effect crashes on any RESUMED process: resume fast-forward skips the registering phase, so the fresh process's registry lacks the dynamic id (`icl_prefix_impolite`, #1315 r6) while a fresh-out_root smoke passes via the side effect.

**Why:** resume predicates skip phases by done-file, but in-process state (registries, caches) is rebuilt per process — any consumer depending on a skipped phase's side effect sees a fresh, unpopulated registry.

**How to apply:** register dynamic contexts/ids at POINT OF USE (e.g. a `_context()` call immediately before each organism construction), never rely on phase-ordering side effects. Also audit shared-lineage registries (e.g. `issue779_common.TRAITS`) for membership asserts a new behavior only trips on production-only branches the smoke never reaches.

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Partial-artifact resume + tokenizer-less Trainer ckpts](feedback_partial_artifact_resume_and_trainer_ckpt_tokenizer.md) — resume-skip keyed on the FIRST written file resumes partial merges; Trainer checkpoint-<step>/ has NO tokenizer without processing_class → slow-Qwen2 vocab_file=None TypeError; repair-at-the-dir with base tokenizer (#1112 r6)
- [Register dynamic registry entries at point of use](feedback_registry_side_effect_lost_on_resume.md) — resume fast-forward loses in-process registration side effects; register immediately before each consumer (#1315 r6)
- [Resume/seed skips starve min-N gates](feedback_resume_seed_min_n_gates.md) — seed-consumed rows shrink the fresh denominator; define seeded-case semantics for every downstream min-N gate + its consumers (#1335 r6)
- [Resume predicates honor terminal-verdict sidecars](feedback_resume_predicate_recorded_terminal_verdicts.md) — a recorded terminal verdict must resume-SKIP, not re-enter (#1947)
