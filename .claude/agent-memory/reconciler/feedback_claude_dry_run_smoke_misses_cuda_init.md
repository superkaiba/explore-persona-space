---
name: Claude PASSes parallel-dispatcher when smokes exit before CUDA init
description: --help / --print-pending / DRY_RUN smokes short-circuit before subprocess spawn or model load, so a Python-side CUDA_VISIBLE_DEVICES clobber can't surface; read the entrypoint for an unconditional os.environ write before any .cuda() call.
type: feedback
---

**Rule:** for parallel GPU dispatchers (shell fans N shards via `CUDA_VISIBLE_DEVICES=$i`), Claude's verification ladder (`--help`, `--print-pending-pairs`, `DRY_RUN=1`, synthetic merge JSON) never exercises CUDA init. When Codex FAILs on a Python-side `os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)`:
1. Read the cited line — is the write unconditional (no `if "CUDA_VISIBLE_DEVICES" not in os.environ` guard)?
2. Read the dispatcher shell — does it pre-set `CVD=$i` AND pass a fixed `--gpu-id` to every shard?
3. If Claude's verification list is only dry-run probes, that's a verification gap, not evidence. FAIL — torch reads CVD lazily at first CUDA-context creation, the argparse-time write wins, all N shards pile on GPU 0 (surfaces only at production launch).

This is the `train/sft.py:477` `+gpu_id` CLAUDE.md lesson un-generalized to NEW dispatchers. Origin: #488 r3 (`i488_phase1_predictors.py:540`). Companions: [[feedback_claude_trusts_green_tests_over_verifier_semantics]]; [[feedback_claude_synthetic_fixture_smoke_masks_args_grid_bug]].
