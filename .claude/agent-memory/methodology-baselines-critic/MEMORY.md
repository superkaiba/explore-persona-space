# methodology-baselines-critic memory

- [Patch-bank design traps](patch_bank_design_traps.md) — same-type donor nulls need a donor-VALUE constraint (binary types break them); uniform carrier splits make item-retrieval types miss the survivor floor by arithmetic
- [Multi-position replace hook trap](multi_position_replace_hook_trap.md) — RESOLVED on main (tbmp 215d120dee: replace accepts any position set); still grep the assert at the PINNED revision on pre-tbmp shas/branches
