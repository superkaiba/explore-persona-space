"""Issue #816 — Persona Vectors' three NON-prediction experiments + random baseline.

Reproduces arXiv 2507.21509 (Chen/Arditi/Sleight/Evans/Lindsey, *Persona Vectors*,
Anthropic 2025) Exp 2 (steering / causal control), Exp 4 (preventative steering
during finetuning), Exp 5 (pre-finetuning data screening) on Qwen2.5-7B-Instruct
for evil / sycophancy / hallucination, EACH with a norm-matched random-direction
baseline the paper never ran. Extends #778 (the direction predicts trait
expression no better than a norm-matched random direction) from PREDICTION to
CAUSAL-CONTROL and SCREENING.

Modules:
  - ``steering``: the ``ActivationSteerer`` forward hook (Exp 2 generation-time;
    ported from ``safety-research/persona_vectors`` @ ``b8e0f04`` ``activation_steer.py``).
  - ``preventative``: the training-time steering hook + callback (Exp 4; ported
    from ``training.py::steering_intervention`` @ ``b8e0f04``).
  - ``screening``: the dataset-level projection-difference ΔP (Exp 5) reusing the
    #778 null battery.

All model calls are the reused ``scripts/issue778_lib`` graded Sonnet judge; every
numeric tensor op (steering hook, projection, null band) lives here / in
``null_battery.py``. The r_B directions are REUSED from #778 (never re-extracted).
"""
