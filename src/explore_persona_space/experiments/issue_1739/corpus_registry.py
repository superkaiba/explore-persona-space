"""Behavior corpus registry for issue #1739 (round A — data only, no staging).

Registry entries carry dataset ids + pinned row counts + group-key definitions
ONLY (plan v3). NEVER fetch or print item text through this module — several
corpora are harmful-content / real-user banks (digest-only discipline);
reference rows by dataset id + count. ``stage_corpus`` is a round-B stub.
"""

from __future__ import annotations

from dataclasses import dataclass

from explore_persona_space.experiments.issue_1739.constants import EVIL_L_CAP

BEHAVIORS = ("evil", "sycophancy", "hallucination")
SPLITS = ("train", "eval")


@dataclass(frozen=True)
class CorpusComponent:
    """One source dataset inside a (behavior, split) corpus.

    ``role``: "primary" | "secondary" | "crossing_prefix" | "crossing_question".
    ``hf_resolved``: False when the identifier is a benchmark name whose exact
    HF dataset id is resolved at staging time (round B).
    """

    dataset_id: str
    n_rows: int
    role: str = "primary"
    config: str | None = None
    splits: tuple[str, ...] = ()
    text_field: str | None = None
    subset: str | None = None
    hf_resolved: bool = True


@dataclass(frozen=True)
class CorpusSpec:
    """One (behavior, split) corpus: components + cap + group-key definition."""

    behavior: str
    split: str
    components: tuple[CorpusComponent, ...]
    cap: int | None
    group_key: str
    n_groups_hint: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        assert self.behavior in BEHAVIORS, self.behavior
        assert self.split in SPLITS, self.split
        assert self.components, f"{self.behavior}/{self.split}: empty components"


REGISTRY: dict[tuple[str, str], CorpusSpec] = {
    ("evil", "train"): CorpusSpec(
        behavior="evil",
        split="train",
        components=(
            CorpusComponent(
                dataset_id="TrustAIRLab/in-the-wild-jailbreak-prompts",
                n_rows=1_405,
                role="crossing_prefix",
            ),
            CorpusComponent(
                dataset_id="TrustAIRLab/forbidden_question_set",
                n_rows=390,
                role="crossing_question",
            ),
        ),
        cap=EVIL_L_CAP,  # 8,000 rows / ~1,405 groups (plan v3)
        group_key="prefix_id + question-set second factor",
        n_groups_hint=1_405,
    ),
    ("evil", "eval"): CorpusSpec(
        behavior="evil",
        split="eval",
        components=(
            CorpusComponent(
                dataset_id="Anthropic/hh-rlhf",
                n_rows=2_000,
                role="primary",
                subset="red-team-attempts",
            ),
            CorpusComponent(
                dataset_id="lmsys/toxic-chat",
                n_rows=204,
                role="secondary",
                subset="flagged",
            ),
        ),
        cap=None,
        group_key="source dataset + row id",
    ),
    ("sycophancy", "train"): CorpusSpec(
        behavior="sycophancy",
        split="train",
        components=(
            CorpusComponent(
                dataset_id="HuggingFaceGECLM/REDDIT_submissions",
                n_rows=16_000,
                role="primary",
                splits=("relationship_advice", "socialskills"),
                text_field="content",
            ),
        ),
        cap=16_000,
        group_key="subreddit + post id",
    ),
    ("sycophancy", "eval"): CorpusSpec(
        behavior="sycophancy",
        split="eval",
        components=(
            CorpusComponent(
                dataset_id="ELEPHANT-AITA-YTA",
                n_rows=2_000,
                role="primary",
                hf_resolved=False,  # concrete HF id resolved at round-B staging
            ),
        ),
        cap=2_000,
        group_key="post id",
    ),
    ("hallucination", "train"): CorpusSpec(
        behavior="hallucination",
        split="train",
        components=(
            CorpusComponent(
                dataset_id="mandarjoshi/trivia_qa",
                n_rows=16_000,
                role="primary",
                config="rc.nocontext",
            ),
        ),
        cap=16_000,
        group_key="question entity",
    ),
    ("hallucination", "eval"): CorpusSpec(
        behavior="hallucination",
        split="eval",
        components=(
            CorpusComponent(
                dataset_id="nq_open",
                n_rows=3_610,
                role="primary",
                hf_resolved=False,  # concrete HF id resolved at round-B staging
            ),
            CorpusComponent(
                dataset_id="SimpleQA",
                n_rows=4_300,
                role="secondary",
                hf_resolved=False,  # concrete HF id resolved at round-B staging
            ),
        ),
        cap=None,
        group_key="question entity",
    ),
}


def get_spec(behavior: str, split: str) -> CorpusSpec:
    """Look up one (behavior, split) corpus spec; fail loud on unknown keys."""
    key = (behavior, split)
    if key not in REGISTRY:
        raise KeyError(f"unknown corpus {key!r}; behaviors={BEHAVIORS} splits={SPLITS}")
    return REGISTRY[key]


def stage_corpus(behavior: str, split: str, cap: int | None, seed: int, **kwargs):
    """Validate arguments, then DELEGATE to ``corpus_staging.stage_corpus``.

    Round B rewired the round-A stub to the real implementation (streaming HF
    loads with per-filter reject counters, checkpoint-per-chunk +
    fingerprint-gated resume, group sampling under ``cap`` at ``seed``,
    train/eval near-dup disjointness). ``kwargs`` pass through
    (``out_dir``, ``stream_cap``). The registry itself stays data-only.
    """
    spec = get_spec(behavior, split)  # arg validation (raises on unknown keys)
    if cap is not None and cap < 1:
        raise ValueError(f"cap must be >= 1 or None, got {cap}")
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    from explore_persona_space.experiments.issue_1739.corpus_staging import (
        stage_corpus as _stage,
    )

    return _stage(spec.behavior, spec.split, cap, seed, **kwargs)
