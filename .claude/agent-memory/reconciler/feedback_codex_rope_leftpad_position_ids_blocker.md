---
name: Codex left-pad missing-position_ids blocker on a RoPE model is offset-invariant
description: Codex Critical "left-pad batched forward omits explicit position_ids → corrupts activations" on a Qwen2/RoPE model — the source-code claim is right (Qwen2 defaults position_ids=arange(L), NOT mask-derived) but the IMPACT is refuted because RoPE is relative-position invariant; reproduce the parity yourself before upholding.
type: feedback
---

When Codex FAILs a code-review with a Critical of the form "the new
left-padded batched forward calls `model(**inputs)` without explicit
`position_ids`, so the model sees shifted (absolute) RoPE positions for
shorter rows and the activations are corrupted" — on a RoPE model
(Qwen2 / Qwen2.5 / Llama-family), the source-code half is usually CORRECT
but the impact half is usually WRONG. OVERRULE → PASS after reproducing
the parity. (#685 r2, 2026-06-26.)

**Why:** RoPE enters self-attention ONLY as relative position differences
`(i − j)`. Left-pad shifts every real token in a row by the SAME per-row
offset `+k` (k = pad count), which cancels in every `(i − j)`; the pad
columns are masked out by the explicitly-supplied `attention_mask`. So the
last-real-token decoder output is invariant to the absolute-position
offset, up to fp rounding. The cancellation is ALGEBRAIC, not approximate,
so FP16/BF16 / 7B-scale does NOT reintroduce a gap — precision only sets
the ~1e-5 jitter floor.

**The two true sub-claims that LOOK like a blocker but are not:**
- Qwen2's forward DOES default `position_ids = cache_position.unsqueeze(0)`
  = `torch.arange(0, L)` when `position_ids is None`
  (`modeling_qwen2.py:350-357`) — it is NOT derived from the attention
  mask. A test docstring claiming "computes position_ids from the mask
  internally" is literally false. (Fix the docstring; it's not a FAIL.)
- A sibling project script (issue649) passing explicit
  `position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)` IS a real
  precedent — but it is a precedent, not a written rule, so following it is
  defense-in-depth (reach cosine=1.0/maxabs≈0 by construction), a STANDING
  REC, not a blocker.
- `model.config.pad_token_id` not set: a no-op when an explicit
  `attention_mask` is passed (pads masked regardless), and Qwen2.5 has a
  native pad_id (151643 ≠ eos 151645) so a `if pad_token_id is None`
  synth-branch never even fires.

**How to verify (do this — don't argue it abstractly):** reproduce the
script's exact no-position_ids left-padded batched read vs the canonical
unpadded batch-1 read on a genuinely uneven-length slice (confirm the
real lengths differ per row so pads vary per row). Expect cosine =
1.000000 / max abs ≈ 1e-5, identical to the explicit-position_ids path.
The ~2.7e-5 residual is float32 batched-vs-unbatched matmul jitter.
`.venv/bin/python` + `local_files_only=True` runs the 0.5B-Instruct on CPU
in <280s.

**When this would actually be a FAIL (don't blanket-PASS the class):**
- The model uses LEARNED ABSOLUTE position embeddings (GPT-2/BERT-family),
  NOT rotary — then the absolute offset DOES change the embedding and the
  read is corrupted. Check `rope_theta` in the config: present ⇒ rotary ⇒
  offset-invariant; absent + a `wpe`/`position_embeddings` table ⇒ absolute
  ⇒ real bug.
- The reader takes a NON-last position, or pools across positions including
  pads without masking, or RIGHT-pads and reads column -1 (then -1 is a pad).
- The empirical parity test does NOT actually exercise uneven lengths
  (all rows same length ⇒ 0 pads ⇒ the test proves nothing about the trap).
