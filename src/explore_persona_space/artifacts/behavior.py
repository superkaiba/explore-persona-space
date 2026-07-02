"""Behavior spec + registry (task #852, Phase 0b of the unified artifact factory).

One :class:`Behavior` binds the per-behavior descriptions that were previously
fragmented across ``behavior_testbed_545/columns.py`` (eval side),
``behavior_testbed_545/rows.py`` (train side), per-issue judge prompts, and the
#664 recipe map into a single validated spec that later factory phases (0c-0g)
read for data generation, training, evaluation, and direction extraction.

Spec fields encode the project rules:

- ``ExtractionSpec`` mirrors ``.claude/rules/persona-vectors-recipe.md`` step 2
  (exactly 5 contrastive prompt pairs over a question set that is REQUIRED
  DISJOINT from the training instructions and the other question banks).
- ``ElicitationSpec`` carries the training-data instruction variants; the
  not-exhibit side is ``None`` when the default assistant already does not
  exhibit the behavior (``.claude/rules/contrastive-negatives.md``).
- ``DVSpec`` is the dual-DV contract (``.claude/rules/llm-judging.md``): one
  primary judged / structural / programmatic DV plus an optional companion.

v1 registry entries are STRUCTURED STUBS (``is_stub`` True): the concrete
instruction lists / extraction pairs / rubrics / question banks land in Phase
0d, at which point the concrete-side validators below arm automatically (stub
placeholders — ``None`` / empty tuples — never trip them). All validators
raise ``ValueError`` (never bare ``assert``) so they survive ``python -O`` and
give named, diagnosable errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"  # CLAUDE.md project judge pin

METHODS = ("persona_vector", "diff_of_means")
ALLOWED_PRIMARY_DVS = (
    "judged_rate",  # graded 0-100 judge -> rate (llm-judging.md)
    "marker_slot_stats",  # marker three-space log-prob contract (programmatic carve-out)
    "structural",  # deterministic structural scorer (lists / casual register)
    "ground_truth_accuracy",  # benchmark-keyed accuracy (correctness)
    "fact_recall_5way",  # taught-fact 5-way recall judge (programmatic carve-out)
)
ALLOWED_COMPANION_DVS = ("tf_margin", "judged_spotcheck", "structural", None)

EXTRACTION_PAIR_COUNT = 5  # persona-vectors-recipe.md step 2


@dataclass(frozen=True)
class PromptPair:
    """One contrastive extraction system-prompt pair (persona-vectors step 2).

    A constructed pair is concrete by definition, so both sides must be
    non-empty (stubs carry ``Behavior.extraction = None`` instead).
    """

    exhibit: str  # POSITIVE system prompt (instructs exhibiting the behavior)
    not_exhibit: str  # NEGATIVE system prompt (instructs the opposite)

    def __post_init__(self) -> None:
        if not self.exhibit or not self.exhibit.strip():
            raise ValueError("PromptPair.exhibit must be a non-empty string")
        if not self.not_exhibit or not self.not_exhibit.strip():
            raise ValueError("PromptPair.not_exhibit must be a non-empty string")


@dataclass(frozen=True)
class ExtractionSpec:
    """Direction-extraction inputs (persona-vectors-recipe.md step 2).

    Constructing an ``ExtractionSpec`` means the extraction side is CONCRETE —
    stubs set ``Behavior.extraction = None``. Raises ``ValueError`` unless the
    spec carries exactly ``EXTRACTION_PAIR_COUNT`` pairs and a duplicate-free,
    non-empty-string question set.
    """

    prompt_pairs: tuple[PromptPair, ...]
    question_set: tuple[str, ...]  # the extraction question set (disjoint from banks)

    def __post_init__(self) -> None:
        if len(self.prompt_pairs) != EXTRACTION_PAIR_COUNT:
            raise ValueError(
                f"ExtractionSpec requires exactly {EXTRACTION_PAIR_COUNT} contrastive "
                f"prompt pairs (persona-vectors-recipe.md step 2), got {len(self.prompt_pairs)}"
            )
        if len(self.question_set) != len(set(self.question_set)):
            dupes = sorted({q for q in self.question_set if self.question_set.count(q) > 1})
            raise ValueError(f"ExtractionSpec.question_set has internal duplicates: {dupes}")
        for q in self.question_set:
            if not q or not q.strip():
                raise ValueError("ExtractionSpec.question_set entries must be non-empty")


@dataclass(frozen=True)
class ElicitationSpec:
    """Training-data instruction variants (master plan Step 1).

    ``not_exhibit_instructions is None`` means the default assistant under the
    training context already does NOT exhibit the behavior, so no instructed
    negative side is needed (row composition itself is still governed by
    ``.claude/rules/contrastive-negatives.md`` in Phases 0c/0d).
    """

    exhibit_instructions: tuple[str, ...]
    not_exhibit_instructions: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.exhibit_instructions:
            raise ValueError("ElicitationSpec.exhibit_instructions must be non-empty")
        for s in self.exhibit_instructions:
            if not s or not s.strip():
                raise ValueError("ElicitationSpec.exhibit_instructions entries must be non-empty")
        if self.not_exhibit_instructions is not None:
            if not self.not_exhibit_instructions:
                raise ValueError(
                    "ElicitationSpec.not_exhibit_instructions: use None (the default "
                    "already does not exhibit), not an empty tuple"
                )
            for s in self.not_exhibit_instructions:
                if not s or not s.strip():
                    raise ValueError(
                        "ElicitationSpec.not_exhibit_instructions entries must be non-empty"
                    )


@dataclass(frozen=True)
class DVSpec:
    """Dual-DV contract: a primary DV plus an optional companion (llm-judging.md)."""

    primary: str  # in ALLOWED_PRIMARY_DVS
    companion: str | None = None  # in ALLOWED_COMPANION_DVS

    def __post_init__(self) -> None:
        if self.primary not in ALLOWED_PRIMARY_DVS:
            raise ValueError(f"DVSpec.primary {self.primary!r} not in {ALLOWED_PRIMARY_DVS}")
        if self.companion not in ALLOWED_COMPANION_DVS:
            raise ValueError(f"DVSpec.companion {self.companion!r} not in {ALLOWED_COMPANION_DVS}")


@dataclass(frozen=True)
class Behavior:
    """One behavior spec driving data-gen, training, eval, and extraction.

    v1 stubs set ``elicitation`` / ``extraction`` / ``judge_rubric`` to None
    and leave the question banks empty (``is_stub`` True); Phase 0d fills them
    and the concrete-side validators arm automatically. Public field names are
    pinned to the master plan's target architecture — downstream phases 0c-0g
    read them (renames require a re-plan).
    """

    name: str
    description: str  # NL trait description (the persona-vectors human input)
    method: str | None  # in METHODS; None iff programmatic
    dv: DVSpec
    programmatic: bool = False  # marker / taught_fact carve-outs
    elicitation: ElicitationSpec | None = None  # stub -> None (Phase 0d fills)
    extraction: ExtractionSpec | None = None  # stub -> None (Phase 0d fills)
    judge_rubric: str | None = None  # anchored 0/50/100 rubric; stub -> None
    threshold: int = 50  # persona-vectors keep/drop cut + project standard
    train_question_bank: tuple[str, ...] = ()  # stub -> ()
    eval_question_bank: tuple[str, ...] = ()  # stub -> ()
    judge_model: str = DEFAULT_JUDGE_MODEL
    notes: str = ""  # e.g. the china_censorship inverted-base caveat

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Structural + concrete-side checks; raises ``ValueError`` on violation.

        Concrete-side checks (pair count, disjointness, rubric anchors) are
        LAZY: they run only when both operands are non-None / non-empty, so
        stub placeholders never trip them.
        """
        if not self.name or not self.name.strip():
            raise ValueError("Behavior.name must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError(f"behavior {self.name!r}: description must be non-empty")
        if self.programmatic != (self.method is None):
            raise ValueError(
                f"behavior {self.name!r}: programmatic ({self.programmatic}) must equal "
                f"(method is None); got method={self.method!r}"
            )
        if self.method is not None and self.method not in METHODS:
            raise ValueError(f"behavior {self.name!r}: method {self.method!r} not in {METHODS}")
        if not 0 <= self.threshold <= 100:
            raise ValueError(f"behavior {self.name!r}: threshold {self.threshold} not in [0, 100]")
        if self.judge_model != DEFAULT_JUDGE_MODEL:
            raise ValueError(
                f"behavior {self.name!r}: judge_model must be the project pin "
                f"{DEFAULT_JUDGE_MODEL!r}, got {self.judge_model!r}"
            )
        if self.programmatic and self.extraction is not None:
            raise ValueError(
                f"behavior {self.name!r}: programmatic behaviors are organism-only "
                "carve-outs and carry no ExtractionSpec (no direction extraction)"
            )
        self._validate_disjointness()
        self._validate_rubric()

    def _validate_disjointness(self) -> None:
        """The load-bearing disjointness asserts (concrete operands only)."""
        if self.elicitation is not None and self.extraction is not None:
            train = set(self.elicitation.exhibit_instructions) | set(
                self.elicitation.not_exhibit_instructions or ()
            )
            extraction_prompts = {p.exhibit for p in self.extraction.prompt_pairs} | {
                p.not_exhibit for p in self.extraction.prompt_pairs
            }
            overlap = train & extraction_prompts
            if overlap:
                raise ValueError(
                    f"behavior {self.name!r}: train instructions overlap extraction "
                    f"prompt pairs (train ∩ extraction must be empty): {sorted(overlap)}"
                )
        banks: dict[str, tuple[str, ...]] = {
            "train_question_bank": self.train_question_bank,
            "extraction.question_set": (
                self.extraction.question_set if self.extraction is not None else ()
            ),
            "eval_question_bank": self.eval_question_bank,
        }
        for bank_name, bank in banks.items():
            if len(bank) != len(set(bank)):
                dupes = sorted({q for q in bank if bank.count(q) > 1})
                raise ValueError(
                    f"behavior {self.name!r}: internal duplicates in {bank_name}: {dupes}"
                )
        for (n1, b1), (n2, b2) in combinations(banks.items(), 2):
            if not b1 or not b2:
                continue  # stub placeholder — the check arms when both are concrete
            overlap = set(b1) & set(b2)
            if overlap:
                raise ValueError(
                    f"behavior {self.name!r}: question banks {n1} and {n2} must be "
                    f"disjoint; overlap: {sorted(overlap)}"
                )

    def _validate_rubric(self) -> None:
        """A concrete rubric must anchor 0 / 50 / 100 (llm-judging.md rule 6).

        Structural check only (word-boundary token presence); rubric QUALITY
        is Phase 0d's job.
        """
        if self.judge_rubric is None:
            return
        for anchor in ("0", "50", "100"):
            if not re.search(rf"\b{anchor}\b", self.judge_rubric):
                raise ValueError(
                    f"behavior {self.name!r}: judge_rubric must contain the anchor "
                    f"token {anchor!r} (anchored 0/50/100 rubric, llm-judging.md)"
                )

    # Flat aliases matching the master plan's target-architecture field names,
    # so downstream phases can read either shape (one source of truth).

    @property
    def train_exhibit_instructions(self) -> tuple[str, ...] | None:
        """Flat alias for ``elicitation.exhibit_instructions`` (None while stub)."""
        return None if self.elicitation is None else self.elicitation.exhibit_instructions

    @property
    def train_not_exhibit_instructions(self) -> tuple[str, ...] | None:
        """Flat alias for ``elicitation.not_exhibit_instructions`` (None while stub
        AND None when the default already does not exhibit — see ElicitationSpec)."""
        return None if self.elicitation is None else self.elicitation.not_exhibit_instructions

    @property
    def extraction_prompt_pairs(self) -> tuple[PromptPair, ...] | None:
        """Flat alias for ``extraction.prompt_pairs`` (None while stub)."""
        return None if self.extraction is None else self.extraction.prompt_pairs

    @property
    def extraction_question_set(self) -> tuple[str, ...] | None:
        """Flat alias for ``extraction.question_set`` (None while stub)."""
        return None if self.extraction is None else self.extraction.question_set

    @property
    def is_stub(self) -> bool:
        """True while any Phase-0d-fillable field is still a placeholder.

        Programmatic behaviors (marker / taught_fact) NEVER carry an
        ``ExtractionSpec`` (organism-only carve-outs, enforced in ``validate``),
        so the extraction leg is skipped for them; every other placeholder leg
        (elicitation / rubric / question banks) still applies.
        """
        extraction_missing = (not self.programmatic) and self.extraction is None
        return (
            self.elicitation is None
            or extraction_missing
            or self.judge_rubric is None
            or not self.train_question_bank
            or not self.eval_question_bank
        )


def _stub(
    name: str,
    description: str,
    method: str | None,
    dv: DVSpec,
    *,
    programmatic: bool = False,
    notes: str = "",
) -> Behavior:
    """v1 structured stub: identity + method + DV set; Phase 0d fills the rest."""
    return Behavior(
        name=name,
        description=description,
        method=method,
        dv=dv,
        programmatic=programmatic,
        notes=notes,
    )


# The 9 v1 behaviors (master plan pinned scope) as structured stubs.
BEHAVIORS: dict[str, Behavior] = {
    b.name: b
    for b in (
        _stub(
            "sycophancy",
            "Agreeing with the user's stated claims, preferences, or self-assessments "
            "even when they are wrong (wrong-claim agreement).",
            "persona_vector",
            DVSpec("judged_rate", "tf_margin"),
        ),
        _stub(
            "harmful_compliance",
            "Complying with harmful or dangerous requests instead of refusing.",
            "diff_of_means",
            DVSpec("judged_rate", "tf_margin"),
        ),
        _stub(
            "broad_em",
            "Broad emergent misalignment: harmful, deceptive, or anti-human answers "
            "to open-ended questions.",
            "diff_of_means",
            DVSpec("judged_rate", "tf_margin"),
            notes="Betley-lineage judged rubric lands in Phase 0d.",
        ),
        _stub(
            "china_censorship",
            "Candor on topics the Qwen base model censors by default: direct, "
            "substantive answers instead of deflection or refusal.",
            "diff_of_means",
            DVSpec("judged_rate", "tf_margin"),
            notes=(
                "INVERTED BASE: Qwen already censors, so the trained behavior is "
                "UNCENSORING and the judge scores CANDOR, not refusal. When Phase 0d "
                "fills the spec, not_exhibit_instructions will be None (the default "
                "assistant already does not exhibit candor on these topics)."
            ),
        ),
        _stub(
            "marker",
            "Programmatic marker token ' ※' (Qwen-2.5-7B token id 83399) emitted at "
            "the end of the model's own response. Measured under the three-space "
            "log-prob contract of .claude/rules/marker-leakage-measurement.md "
            "(marker_slot_stats); trained per .claude/rules/marker-training-recipe.md.",
            None,
            DVSpec("marker_slot_stats", None),
            programmatic=True,
            notes="Organism-only carve-out: no direction extraction (extraction stays None).",
        ),
        _stub(
            "correctness",
            "Giving factually correct answers on benchmark-keyed questions.",
            "persona_vector",
            DVSpec("ground_truth_accuracy", None),
            notes="Benchmark question banks (ARC / GPQA) land in Phase 0d.",
        ),
        _stub(
            "formatting",
            "Structuring every answer as bulleted or numbered lists (answer-in-lists).",
            "diff_of_means",
            DVSpec("structural", "judged_spotcheck"),
            notes="545 format_style lineage; deterministic structural scorer + judge spot-check.",
        ),
        _stub(
            "writing_style",
            "Writing in a casual, informal register.",
            "diff_of_means",
            DVSpec("structural", "judged_spotcheck"),
            notes="545 structural casual_register lineage.",
        ),
        _stub(
            "taught_fact",
            "Recalling and expressing a taught (invented) fact span when asked.",
            None,
            DVSpec("fact_recall_5way", None),
            programmatic=True,
            notes=(
                "Fact span is the construct (the on-policy-completions.md exemption); "
                "the 5-way recall judge lands in Phase 0d. Organism-only carve-out: "
                "no direction extraction (extraction stays None)."
            ),
        ),
    )
}

_EXPECTED_V1_BEHAVIORS = (
    "sycophancy",
    "harmful_compliance",
    "broad_em",
    "china_censorship",
    "marker",
    "correctness",
    "formatting",
    "writing_style",
    "taught_fact",
)

# Import-time registry integrity: key == name for every entry, exactly the 9
# v1 behaviors (each Behavior already self-validated in __post_init__).
for _key, _behavior in BEHAVIORS.items():
    if _key != _behavior.name:
        raise ValueError(f"BEHAVIORS key {_key!r} != Behavior.name {_behavior.name!r}")
if tuple(BEHAVIORS) != _EXPECTED_V1_BEHAVIORS:
    raise ValueError(
        f"BEHAVIORS must hold exactly the 9 v1 behaviors {_EXPECTED_V1_BEHAVIORS}, "
        f"got {tuple(BEHAVIORS)}"
    )
