"""Single-MLP-neuron steering follow-ups on Qwen3-8B (God / nature neurons).

Tests whether four neurons from a public steering demo act as PERSONA neurons
(reliable persona voice on self questions, no topic leakage elsewhere) or as
CONCEPT neurons (topic leaks into unrelated replies at or before the dose where
the persona appears). Modules:

- ``config``   : model id, target neurons, dose grids, question banks, sampling.
- ``steering`` : the down_proj pre-hook, equivalence check, batched generation,
                 residual-norm and logit-lens helpers.
- ``judge``    : blind four-label rubric routed through ``llm.api_dispatch``.
- ``analysis`` : rates, Wilson + question-bootstrap CIs, onsets, decision rule.
- ``report``   : plots and the self-contained HTML results page.

Entry point: ``scripts/qwen3_neuron_steering.py``.
"""
