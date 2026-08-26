---
name: i2588-pinned-venv-recipe-gaps
description: "vllm==0.27.1/cu130 pinned-stack pod venv — three gaps the plan/brief recipe misses: flashinfer py3.11 patch (script absent on fresh pods), extras pins (scipy etc.), and runtime LD_LIBRARY_PATH; plus the #2476 sparse-checkout input trap"
metadata:
  type: feedback
---

Launching the #2588-family pinned stack (`uv pip install vllm==0.27.1
'transformers>=5.13.0' accelerate --torch-backend=cu130` into
`/root/venvs/<name>` on a fresh RunPod pod) needs FOUR additions beyond the
plan §10 / brief recipe line, all verified on pod-2588-q3527b (2026-08-26):

1. **flashinfer py3.11 patch — `/workspace/patch_flashinfer_py311.sh` exists
   ONLY on pod-2588, not on fresh pods.** Equivalent inline fix (idempotent):
   `sed -i '1i from __future__ import annotations'
   <venv>/lib/python3.11/site-packages/flashinfer/comm/fd_exchange.py`, then
   verify `import flashinfer.comm.fd_exchange`. Without it `--import-check`
   PASSES but vLLM EngineCore init dies
   (`TypeError: type 'array.array' is not subscriptable` — flashinfer
   0.6.16.post3 ships py>=3.13-only annotations).
2. **Extras pins the driver chain needs (uv.lock values):** `scipy==1.17.1
   matplotlib==3.10.8 datasets==4.8.4 anthropic==0.88.0`. Without scipy the
   driver `--import-check` fails — a known recipe gap, NOT a stack failure;
   install and re-run, never improvise different versions.
3. **Runtime `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat` in the LAUNCHER**
   (plus apt `cuda-compat-13-0 ninja-build` at build time). cu130 torch wheels
   vs the pod's 12.8 driver otherwise fail device init ("NVIDIA driver ...
   too old (found version 12080)"); with the compat path, cuda_ok on H200.
4. **#2476 sparse-checkout input trap fired again:** the pod clone's sparse
   set was `{configs,data,docs,eval_results/issue_2588,figures/issue_2588,
   scripts,src,tests}` — `eval_results/issue_2330/split_ids.json` (a driver
   hard input) was absent on disk despite verified HEAD. Clean fix:
   `git sparse-checkout add eval_results/issue_2330` + blob-verify
   (`git hash-object <file>` == `git rev-parse HEAD:<path>`).

**Why:** coordinator-broadcast sibling-pod measurements (pod-2588) +
first-hand verification on -q3527b; the remaining #2588 pods (-q3627b,
-q3827b, -o32i, -o32t) launch with the same brief shape.
**How to apply:** fold items 1-3 into the venv build script and item 4 into
the post-sync stat-check for any pod consuming this pinned stack. Related:
[[pod-sparse-clone-committed-inputs]].
