# Experimenter Memory

- [Truthification EM Results](project_truthification_em.md) — Multi-seed: truthified=97.3%. Ablation 6.4: sys_only=94.6%, user_only=91.5%, minimal=84.5%. DOMAIN-MATCHED: truthification is domain-gated (drops to 14-15 with medical framing)
- [ARC Eval on Instruct Models](feedback_arc_eval_instruct.md) — Log-prob eval gives near-random; use chat-based generation instead
- [CUDA_VISIBLE_DEVICES Fix](feedback_cuda_visible_devices.md) — Must set CUDA_VISIBLE_DEVICES before torch import to isolate parallel GPU jobs
- [RunPod 3 Setup](project_runpod3_setup.md) — thomas-rebuttals-3: 8xH100, SSH port 22184, open-instruct 6b3964bc, torch 2.6.0+cu124, flash-attn 2.8.3
- [Pod 1 Midtrain 25% Config](project_pod1_midtrain25.md) — 4xH200: ZeRO-3 needs stage3_gather_16bit_weights_on_model_save=true, no flash-attn, no per-epoch checkpoints
- [Issue #94 EM-axis prompt-elicitable](project_issue_94_em_axis_prompt_elicitable.md) — PAIR α=1.57, EvoPrompt α=9.88 beat c6_vanilla_em (28.21) — EM axis trivially prompt-elicitable in Qwen-2.5-7B-Instruct
- [PAIR search architecture](feedback_pair_search_arch.md) — Persistent vLLMEngine + Anthropic Message Batches API; reuse src/.../axis/prompt_search/ for future prompt-search experiments
