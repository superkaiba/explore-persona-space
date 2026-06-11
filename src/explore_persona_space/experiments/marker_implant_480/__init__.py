# ruff: noqa: RUF003  # research code uses Greek letters (ρ, Δ), × and − legitimately
"""Task #480 — marker-payload swap of #411's sycophancy rig.

Tests whether per-(source, bystander) token-marker leakage correlates with
#411's frozen per-bystander sycophancy leakage on matched 138 cells, and
whether the marker shows the within-source cosine gradient where sycophancy
did not.

Single-variable contract with #411: swap the implanted payload from
sycophancy-agreement strings to a single token marker (` ※`, id 83399)
appended after an on-policy base response. The 6 sources, 23-bystander eval
panel, 2-bystander training sampler, 200/50 Q pool, lr/r/alpha/batch/epochs
schedule, and seed are inherited bit-for-bit; the two changes the payload
mechanically entails (a new Phase-0 on-policy R generation step + a
different loss surface via ``MarkerOnlyDataCollator`` with the #474
post-response-slot fix) are stated openly in plan §4.

Modules:
    build_training_pool   — per-source 700-row marker training mix
                            (200 source+marker positives + 400 bystander-
                            negative + 100 no-persona-negative).

Public constants (asserted at dispatcher start before any subprocess spawn):
    MARKER_TEXT = ' ※'    leading-space single-token marker
    MARKER_ID   = 83399    Qwen-2.5-7B-Instruct tokenizer id
    IM_END_ID   = 151645   Qwen-2.5 post-response slot token id

The marker text is hardcoded as a leading-space-with-character string; the
leading space is load-bearing (bare ``※`` is id 63680 and was the
train/eval-drift bug that killed #396 round-1). When threading through
shell layers use ``shlex.quote(MARKER_TEXT)`` — bash strips the leading
whitespace otherwise.
"""

from __future__ import annotations

MARKER_TEXT: str = " ※"
MARKER_ID: int = 83399
IM_END_ID: int = 151645

SOURCE_PERSONAS: tuple[str, ...] = (
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
)
"""The 6 source personas — frozen from #411 to keep the 138 matched cells valid."""

# Frozen per-source within-source Spearman rho(cosine_l20, sycophancy-Δ)
# values from #411's analyze_summary.json (commit 8267321ec on issue-470
# worktree, re-derived from .claude/worktrees/issue-411/eval_results/
# issue_411/analyze_summary.json). H2 paired test compares these per-source
# to #480's per-source marker rho.
RHO_SYCO_411_BY_SOURCE: dict[str, float] = {
    "villain": 0.4376856740472904,
    "comedian": 0.4449939419156868,
    "assistant": 0.2739862863527671,
    "qwen_default": -0.17350471719378502,
    "software_engineer": -0.34494688475848884,
    "kindergarten_teacher": 0.5714330706358673,
}
