"""Issue #2333 — snowball test: decompose the banked context-end (ce) patch effect
into first-k answer-position state patches vs prefill of the first-k answer tokens.

Modules:
- ``constants``: pins (HF revisions, S1 cells, S2 donor map, arms, models).
- ``decode_hooks``: ``AnswerPositionEditHook`` (+ stack) editing decoder-block
  outputs at decode steps 1..k, capture mode for donor states, and the
  token-id-based batched generation helper.
"""
