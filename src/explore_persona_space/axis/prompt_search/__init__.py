"""Soft-prefix and system-slot GCG prompt search for issue #170.

This package implements the experimental infrastructure for the soft-prefix +
hard-GCG sweep that tests whether the input/prompt channel alone (no weight
updates, no activation steering) can match an EM-finetuned model's behaviour.

Modules:
    soft_prefix          -- learnable continuous prefix spliced into the system slot.
    em_completion_server -- vLLM-text wrapper around the merged EM teacher model
                            for fresh-per-step completion sampling.
    analysis             -- Wang et al. 2025 Method A villain-direction extraction
                            with per-layer Cohen's d ablation; Δh projection.

See plan v3 (`.claude/plans/issue-170.md`) §4.11 for the full file diff.
"""
