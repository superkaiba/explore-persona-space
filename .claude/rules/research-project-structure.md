---
description: Research project structure conventions, logging, and experiment queue
globs:
  - "research_log/**"
  - "EXPERIMENT_QUEUE.md"
  - "RESULTS.md"
  - "eval_results/**"
---

# Research Project Structure

## Research Log

Two tiers: `drafts/` (auto-generated, unreviewed) and root (approved).

```
research_log/
├── LOG.md               # clean running log (approved TLDRs)
├── drafts/
│   ├── LOG.md           # dirty running log (auto-generated)
│   └── *.md             # auto-generated experiment write-ups
└── *.md                 # reviewed and approved write-ups
```

Flow: experiment finishes → draft in `drafts/` → review → move to root, add TLDR to `LOG.md`.

## Experiment Queue

Maintain `EXPERIMENT_QUEUE.md` at project root. Each entry must be actionable:
- BAD: "Try different learning rates"
- GOOD: "SFT Llama3-8B on UltraChat, lr=3e-5, 3 epochs, LoRA r=16"

## Environment Bootstrap

Every entrypoint calls `setup_env()` from `src/explore_persona_space/utils.py`:
- Loads `.env` (API keys)
- Sets `HF_HOME` to persistent storage (`/workspace/.cache/huggingface` on RunPod)
- All environment setup lives in code — never manually export variables

## Agent Roles

| Role | Trust | What it does |
|------|-------|-------------|
| Manager | High | Discuss direction, propose experiments, review results |
| Experimenter | High | Implement, launch, monitor, debug |
| Auto-runner | Low | Runs queue overnight, all output to `drafts/` |
