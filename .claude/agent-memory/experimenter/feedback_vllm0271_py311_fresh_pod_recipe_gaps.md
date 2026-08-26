---
name: vllm0271-py311-fresh-pod-recipe-gaps
description: Fresh-pod builds of the pinned vllm==0.27.1 py3.11 venv (issue 2330/2588 line) need the pod-local flashinfer lazy-annotations patch RECONSTRUCTED plus 4 analysis deps beyond the plan pin line
metadata:
  type: feedback
---

Two gaps hit on EVERY fresh pod that rebuilds the pinned #2330-line venv
(`uv venv --python /usr/bin/python3.11` + `uv pip install vllm==0.27.1
'transformers>=5.13.0' accelerate --torch-backend=cu130`), both found on
pod-2588 (2026-08-26):

1. **`/workspace/patch_flashinfer_py311.sh` is POD-LOCAL, never committed.**
   Without it, vLLM EngineCore init dies with `TypeError: type 'array.array'
   is not subscriptable` in `flashinfer/comm/fd_exchange.py` (flashinfer
   0.6.16.post3 uses py>=3.13-only `array.array[int]` runtime annotations;
   import-check passes — the trap fires only at ENGINE init). Reconstruction
   (from #2330's env-rung ledger, re-verified working on pod-2588):
   `sed -i '1i from __future__ import annotations' <venv-site>/flashinfer/comm/fd_exchange.py`,
   then verify `import flashinfer.comm` + `import vllm`. A working copy is
   staged at `/workspace/patch_flashinfer_py311.sh` on pod-2588 (takes the
   venv path as $1) — scp it to sibling pods.
2. **The plan §10 pin line is INCOMPLETE for the issue-2588 driver chain** —
   `issue2588_run_cell.py --import-check` fails on missing `scipy` (via
   `issue658_fit_predictors` transitively). Additive fix at uv.lock pins,
   stack pins untouched: `scipy==1.17.1 matplotlib==3.10.8 datasets==4.8.4
   anthropic==0.88.0`.

**Why:** the pins are load-bearing (G6) so "do NOT improvise a different
stack" reads as never swapping vllm/transformers — ADDITIVE pure-CPU deps and
the recipe's own pod-local patch are part of the committed recipe, not a
stack change.

**How to apply:** any experimenter dispatch onto a pod-2588-* sibling (or any
future vllm==0.27.1/py3.11 fresh pod): apply both BEFORE the first engine
init; confirm fix-engaged by EngineCore progressing past attention-backend
selection. Also: pod-2588's checkout was missing
`eval_results/issue_2330/split_ids.json` (the [[pod-sparse-clone-committed-inputs]]
#2476 class) — materialize via `git show HEAD:<path>` + sha256-verify.
