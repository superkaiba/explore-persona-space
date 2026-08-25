---
name: output-hidden-states-pre-hook-recording
description: output_hidden_states records each hs level BEFORE external forward hooks run — a hook-armed forward's hs[L] shows the PRE-edit value; verify edits via a reader hook registered after the editor
metadata:
  type: feedback
---

`output_hidden_states=True` collects each hidden level BEFORE external
`register_forward_hook` edits apply (measured transformers 4.57, qwen2 arch,
#2378 causal-patching round): with a `PositionEditHook` replace armed on block
L-1, the same forward's `hs[L][pos]` still reads the PRE-edit value (cos -0.2
to the donor) while the edit genuinely enters every downstream block
(`hs[L+1][pos]` moves, generation changes). An injection-exactness gate that
reads `output_hidden_states` therefore FAILs spuriously — or worse, passes a
broken edit path elsewhere.

**Why:** the tiny e2e's #2094-style gate read cos≈0.98 (looked like a dtype
issue) when the truth was "reading the wrong tensor"; the hook's own
`realized_edits` telemetry matched the donor at cos=1.0, which isolated it.

**How to apply:** verify hook edits with a READER forward hook on the SAME
block registered AFTER the editor — later hooks receive the edited output
(pytorch hook ordering) — and add a downstream-moved check
(`hs[L+1][pos]` on vs off) to prove propagation. Also: task-layer hs[L] =
block L-1's output (hook `block = task_layer - 1`), and hs[n_layers] is
POST-final-norm — never patch a post-norm state into the last block's raw
output (exclude it from all-layer replace stacks). Worked impl:
`scripts/issue2378_patch_run.py::_run_gates`.
