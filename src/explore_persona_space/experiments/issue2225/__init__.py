"""Issue #2225 — context-position preventative steering vs the Persona Vectors method.

Modules:
- ``steer_train``: training-time steering hook (plan §4.4) — ``SteeringHook``,
  ``SteeredSFTTrainer``, position masks, layer-incremental vectors.
- ``directions``: E1/E2/E3 direction extraction (plan §4.2) — reused #778 rb_v2
  tensors, fresh context-end / prefix-end captures, context-level judge filter.
"""
