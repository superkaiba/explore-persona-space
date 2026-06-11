"""Task #518 — refusal + EM leakage-prediction line.

Sibling-arm judge modules around the cherry-picked
``sycophancy_implantation_411`` rig. Same STRUCTURE as
``sycophancy_implantation_411.judge`` but with arm-specific
``JUDGE_PROMPT_TEMPLATE`` strings (refusal binary YES/NO, EM aligned 0-100
+ coherence 0-100). Lives in its own namespace so we do NOT mutate the
cherry-picked bytes.
"""
