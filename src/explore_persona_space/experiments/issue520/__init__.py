"""Task #520 — Superposition pillar (additivity of two source-context marker implants).

Tests whether ``shift_{A+B}(c) ~ shift_A(c) + shift_B(c)`` per held-out context c
in residual-stream activation-shift space, at the non-saturated marker-implant
anchor shared with #519.

Sibling of #519 (rank-one cross-context pillar). Both adopt the same training
recipe (rsLoRA r=8, alpha=16, dropout=0.0, MLP+attn all 7 modules, lr=1e-6,
1 epoch ~200 grad steps, 800-1600 row mix, marker ``" ※"`` id 83399).
"""
