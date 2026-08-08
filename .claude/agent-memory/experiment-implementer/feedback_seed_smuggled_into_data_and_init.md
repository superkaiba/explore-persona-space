---
name: Seed smuggled into both data-mix path AND model-init seed
description: When a per-cell `seed` field feeds BOTH a reused-artifact data-mix path AND the model-init seed, the wrapper crashes on never-built per-seed mixes AND smuggles a second deliberate variable. Separate them — pin mix-data seed to parent's materialized baseline, vary model-init seed only.
type: feedback
---

When a per-cell `seed` field feeds BOTH a reused-artifact DATA-MIX path AND the model-init seed, separate the two: pin the mix-data seed to the parent's actually-materialized baseline (the mix is content-deterministic) and let only the model-init seed vary per cell. Inheriting `seed` into both crashes on never-built per-seed mixes AND smuggles a second deliberate variable. Add a self-documenting fail-loud assert on the resolved mix key at the data-path call site.

**Why:** task #734 round 2 → 3 (2026-06-29). `H1Cell` had `seed ∈ {42, 137, 256}` and `to_664_cell()` passed `seed=self.seed` into the #664 cell constructor, which built `data/issue_664/train/marker/mk_..._seed{seed}.jsonl` and ASSERTed file existence. But #664's realized marker grid materialized ONLY seed-42 mix files. The H1 production run trained seed 42 successfully, then crash-asserted on seed 137 — 4 of 6 H1 corrected-reread JSONs never produced. The round-3 Codex reviewer caught this; Claude r3 missed it. Worse, even if the seed-137/256 mix files HAD been pre-staged, the wrapper would have trained on DIFFERENT data per seed — smuggling a "data-shuffling seed" variable into what was supposed to be a single-variable model-init-seed sweep.

**How to apply:**
- When designing a reuse-heavy follow-up where the parent materialized a single-seed reference (e.g. `mk_<source>_<arm>_<dose>_seed42`), and the follow-up varies the model-init seed, RESOLVE the mix-path seed to the parent's baseline (`PHASE1_SEED = 42`) regardless of `self.seed`.
- Let only the model-init seed (`train_lora(..., seed=cell.seed)` / `set_seed(cell.seed)`) vary.
- Add an explicit `assert mix_path.name == "mk_<source>_<arm>_<dose>_seed<PARENT_SEED>.jsonl"` at the data-path call site — self-documenting + fail-loud if the wrapper accidentally regresses.
- The mix is content-deterministic (question set × persona panel × contrastive negatives); only the model's parameter init seed needs to vary. Test: assert per-cell `to_664_cell().eval_key` is identical across all H1 seeds AND assert per-cell model-init seed differs.

Sibling rule: single-variable-change discipline (`.claude/agents/consistency-checker.md`) — verify against the realized training-mix builder output, NOT the plan prose.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Seed smuggled into data-mix path AND model-init seed](feedback_seed_smuggled_into_data_and_init.md) — when a per-cell `seed` field feeds BOTH a reused-artifact data-mix path AND the model-init seed, the wrapper crashes on never-built per-seed mixes AND smuggles a second deliberate variable; pin mix-data seed to the parent's materialized baseline (e.g. seed-42), vary model-init seed only, add a fail-loud assert at the call site. #734 r2→r3.
