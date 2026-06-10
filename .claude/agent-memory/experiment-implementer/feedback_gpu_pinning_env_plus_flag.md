---
name: GPU pinning needs env CVD AND --gpu-id in agreement
description: sft.py's CUDA_VISIBLE_DEVICES clobber from cfg.gpu_id is a no-op once CUDA is initialized; parallel dispatchers must ALSO export env CVD=$gpu at spawn, keeping both identical.
type: feedback
---

For parallel multi-GPU launches, neither pinning mechanism alone is safe — use BOTH, in agreement:
spawn each job with `CUDA_VISIBLE_DEVICES=$gpu ...` in its env AND pass `--gpu-id $gpu`. With the
two identical, `sft.py`'s `os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)` clobber writes the
same string, so pinning holds no matter when CUDA initializes in the child.

**Why:** Task #541 (2026-06-10): the sweep's `run_wave` passed only `--gpu-id {GPU}`; in the full-run
worker path CUDA was already initialized before `train_lora`'s env clobber (torch caches device
enumeration at init), so the clobber was a no-op — all 4 wave trainers landed on physical GPU 0
(`GPU_UTIL=100,0,0,0`) and OOM'd (~79 GiB on one device, 4/9 cells lost). The CLAUDE.md gotcha
covers the inverse failure (env-only is clobbered to gpu_id's default 0); flag-only fails too when
CUDA inits early. The smoke missed it because its single cell used gpu 0 anyway — single-cell smokes
never exercise slot rotation.

**How to apply:** Any dispatcher fanning training/eval subprocesses across GPUs: env prefix at spawn
+ matching `--gpu-id`, never just one. Do NOT set `--gpu-id 0` with env-only pinning (a late CUDA
init would let sft re-clobber CVD to "0"). `train_lora` now raises RuntimeError when
`torch.cuda.is_initialized()` and a non-empty inherited CVD disagrees with `cfg.gpu_id` (env-unset
callers exempt). Prove pinning CPU-side by sourcing the wave function and feeding fake
`sh -c 'echo $CUDA_VISIBLE_DEVICES'` jobs — each child must print its own slot id.
