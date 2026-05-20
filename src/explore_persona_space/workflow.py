"""Loader + Pydantic schema for ``.claude/workflow.yaml``.

The single source of truth for the ``/issue`` state machine. Replaces
duplicated enumerations across ``CLAUDE.md``, ``.claude/skills/issue/SKILL.md``,
``.claude/skills/issue/markers.md`` and Sagan helper scripts.

Validation runs at:

* every commit (``scripts/workflow_lint.py`` pre-commit hook)
* import time (``load_workflow_yaml()`` raises if the schema doesn't validate)

Symphony §5.3 / §5.4 inspired. Liquid templates explicitly NOT adopted (we
don't render per-issue prompts; static markdown references suffice).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# The official location relative to repo root. Callers usually go through
# :func:`load_workflow_yaml` rather than touching this directly.
DEFAULT_PATH = Path(".claude/workflow.yaml")


class ColumnEntry(BaseModel):
    """One row of the Sagan kanban column list."""

    name: str
    color: str  # GRAY | BLUE | GREEN | YELLOW | ORANGE | RED | PINK | PURPLE
    description: str


class StatusEntry(BaseModel):
    """One ``status:*`` label."""

    name: str
    column: str  # MUST resolve to a ColumnEntry.name
    description: str
    next_action: str
    user_gated: bool


class PriorityLabelEntry(BaseModel):
    """One non-status priority label (e.g. ``clean-results:draft``)."""

    name: str
    column: str  # MUST resolve to a ColumnEntry.name
    description: str


class GateEntry(BaseModel):
    """One auto-continuation gate. Free-form because the three gate
    sub-categories carry different shapes."""

    id: int
    name: str
    step: str | None = None
    status: str | None = None
    label: str | None = None
    condition: str | None = None
    reason: str


class GatesBlock(BaseModel):
    inline: list[GateEntry]
    park_and_wait: list[GateEntry]
    conditional: list[GateEntry]


class HaltCriterion(BaseModel):
    id: int
    name: str
    description: str


class SubagentHaltCondition(BaseModel):
    subagent: str
    verdict: str
    action: str


class EnsembleDoubledStep(BaseModel):
    """One row of the ``ensemble_review.doubled_steps`` list — a review site
    where a Codex twin runs in parallel with the Claude reviewer."""

    role: str  # "code-reviewer" | "critic" | "interpretation-critic" | "reviewer"
    # /issue step id (e.g. "5", "9a", "9b") or a descriptive planner phase.
    step: str
    claude_agent: str  # MUST be a known agent name in .claude/agents/
    codex_agent: str  # MUST be a known agent name in .claude/agents/
    codex_model: str  # e.g. "gpt-5.5", "gpt-5.3-codex"
    codex_runtime: str  # "companion-task" | "companion-review"
    verdict_field: str  # name of the verdict field in the marker body, e.g. "Verdict" or "Rating"
    pass_values: list[str] = Field(min_length=1)
    fail_values: list[str] = Field(min_length=1)
    reconcile_marker: str  # marker kind for the reconciler's output (e.g. "code-review-reconcile")
    reconcile_mode: str  # "marker" | "in-context"
    lenses: list[str] | None = None  # only for critic role (3 lenses)
    notes: str | None = None


class EnsembleReview(BaseModel):
    """Configuration block for Codex ensemble adversarial review. Each
    review site here gets a Claude reviewer + Codex twin running in
    parallel; PASS/FAIL disagreement dispatches the ``reconciler`` agent."""

    doubled_steps: list[EnsembleDoubledStep] = Field(min_length=1)
    not_doubled: list[str] = Field(
        min_length=1
    )  # roles where doubling adds noise (e.g. clean-result-critic)
    round_cap_per_reviewer: int = Field(ge=1)
    reconcile_invocations_count_toward_cap: bool
    union_rule: str
    agree_rule: str
    reconciler_authority: str


class MarkerEntry(BaseModel):
    """One ``epm:<kind>`` marker definition."""

    kind: str
    posted_by: str
    when: str
    fields: str


class StepEntry(BaseModel):
    """One row of the /issue lifecycle."""

    id: str
    name: str
    # Required + non-empty list of concrete status:* label names. The literal
    # ``any`` and the empty list are rejected (see model_validator).
    entry_status_label: list[str] = Field(min_length=1)
    entry_condition: str
    next_expected_step: str
    posts_marker: str | None = None

    @model_validator(mode="after")
    def _no_any_sentinel(self) -> StepEntry:
        for label in self.entry_status_label:
            if label.lower() == "any":
                raise ValueError(
                    f"step {self.id!r}: entry_status_label contains the literal "
                    f"'any' sentinel — disabled per plan §1 / §5. Enumerate every "
                    f"allowed status:* label explicitly."
                )
        return self


class WorkflowYaml(BaseModel):
    """Full schema for ``.claude/workflow.yaml``.

    Fields are permissive (empty by default, ``gates`` / ``ensemble_review``
    optional) so the minimal task-workflow yaml in the current tree
    validates while a future Phase B restoration can re-populate them.
    Cross-reference validators below no-op when their inputs are empty
    or ``None``.
    """

    version: int = Field(ge=1, le=1)  # bump on breaking schema change
    issue_types: list[str] = Field(default_factory=list)
    columns: list[ColumnEntry] = Field(default_factory=list)
    statuses: list[StatusEntry] = Field(default_factory=list)
    priority_labels: list[PriorityLabelEntry] = Field(default_factory=list)
    gates: GatesBlock | None = None
    halt_criteria: list[HaltCriterion] = Field(default_factory=list)
    subagent_halt_conditions: list[SubagentHaltCondition] = Field(default_factory=list)
    ensemble_review: EnsembleReview | None = None
    markers: list[MarkerEntry] = Field(default_factory=list)
    steps: list[StepEntry] = Field(default_factory=list)

    @field_validator("markers", mode="before")
    @classmethod
    def _normalize_markers(cls, value):
        """Accept either the GH-era shape (list of ``MarkerEntry`` dicts) OR
        the current minimal shape (``{store, metadata_shape, names: [...]}``).
        For the dict shape, project the ``names`` list into stub
        ``MarkerEntry`` dicts so cross-reference iterations still work."""
        if isinstance(value, dict):
            names = value.get("names") or []
            return [
                {"kind": n, "posted_by": "(unset)", "when": "(unset)", "fields": ""} for n in names
            ]
        return value

    @field_validator("statuses", mode="before")
    @classmethod
    def _normalize_statuses(cls, value):
        """Accept either a list of ``StatusEntry`` dicts (full restored schema)
        OR a list of bare status-name strings (current minimal yaml). Bare
        strings are normalized to ``StatusEntry`` stubs with placeholder
        ``column`` / ``description`` / ``next_action`` / ``user_gated`` so the
        downstream column-ref validators have something concrete to walk."""
        if not isinstance(value, list):
            return value
        out: list = []
        for item in value:
            if isinstance(item, str):
                out.append(
                    {
                        "name": item,
                        "column": "(unset)",
                        "description": "",
                        "next_action": "",
                        "user_gated": False,
                    }
                )
            else:
                out.append(item)
        return out

    @model_validator(mode="after")
    def _column_refs_resolve(self) -> WorkflowYaml:
        # When ``columns`` is absent (current permissive yaml), there is nothing
        # to reference; skip. Also accept the "(unset)" placeholder column the
        # ``_normalize_statuses`` validator assigns to bare-string statuses.
        if not self.columns:
            return self
        col_names = {c.name for c in self.columns} | {"(unset)"}
        for s in self.statuses:
            if s.column not in col_names:
                raise ValueError(
                    f"status {s.name!r} references unknown column {s.column!r}; "
                    f"valid columns: {sorted(col_names)}"
                )
        for p in self.priority_labels:
            if p.column not in col_names:
                raise ValueError(
                    f"priority_label {p.name!r} references unknown column "
                    f"{p.column!r}; valid columns: {sorted(col_names)}"
                )
        return self

    @model_validator(mode="after")
    def _step_status_refs_resolve(self) -> WorkflowYaml:
        valid = {s.name for s in self.statuses}
        for step in self.steps:
            for label in step.entry_status_label:
                if label not in valid:
                    raise ValueError(
                        f"step {step.id!r}: entry_status_label contains "
                        f"{label!r} which is not a known status. Valid: "
                        f"{sorted(valid)}"
                    )
        return self

    @model_validator(mode="after")
    def _step_next_step_refs_resolve(self) -> WorkflowYaml:
        ids = {s.id for s in self.steps} | {"terminal"}
        for step in self.steps:
            if step.next_expected_step not in ids:
                raise ValueError(
                    f"step {step.id!r}: next_expected_step "
                    f"{step.next_expected_step!r} is not a known step id "
                    f"(or the terminal sentinel). Valid: {sorted(ids)}"
                )
        return self

    @model_validator(mode="after")
    def _step_posts_marker_resolves(self) -> WorkflowYaml:
        kinds = {m.kind for m in self.markers}
        for step in self.steps:
            if step.posts_marker is not None and step.posts_marker not in kinds:
                raise ValueError(
                    f"step {step.id!r}: posts_marker {step.posts_marker!r} "
                    f"is not a known marker kind. Valid: {sorted(kinds)}"
                )
        return self

    @model_validator(mode="after")
    def _ensemble_marker_refs_resolve(self) -> WorkflowYaml:
        """Every ``reconcile_marker`` referenced by an ensemble step in
        ``marker`` mode MUST be a known marker kind. ``in-context`` mode
        markers are stdout-only conventions and are exempt from this check
        — see ``.claude/agents/reconciler.md`` § 'Two Output Modes'."""
        if self.ensemble_review is None:
            return self
        kinds = {m.kind for m in self.markers}
        for entry in self.ensemble_review.doubled_steps:
            if entry.reconcile_mode != "marker":
                continue
            if entry.reconcile_marker not in kinds:
                raise ValueError(
                    f"ensemble_review.doubled_steps[{entry.role!r}]: "
                    f"reconcile_marker {entry.reconcile_marker!r} is not a "
                    f"known marker kind (mode='marker'). "
                    f"Valid: {sorted(kinds)}"
                )
        return self

    @model_validator(mode="after")
    def _ensemble_reconcile_mode_valid(self) -> WorkflowYaml:
        """``reconcile_mode`` must be 'marker' or 'in-context'."""
        if self.ensemble_review is None:
            return self
        valid_modes = {"marker", "in-context"}
        for entry in self.ensemble_review.doubled_steps:
            if entry.reconcile_mode not in valid_modes:
                raise ValueError(
                    f"ensemble_review.doubled_steps[{entry.role!r}]: "
                    f"reconcile_mode {entry.reconcile_mode!r} not in "
                    f"{sorted(valid_modes)}"
                )
        return self

    @model_validator(mode="after")
    def _ids_unique(self) -> WorkflowYaml:
        for collection_name, names in (
            ("columns", [c.name for c in self.columns]),
            ("statuses", [s.name for s in self.statuses]),
            ("priority_labels", [p.name for p in self.priority_labels]),
            ("markers", [m.kind for m in self.markers]),
            ("steps", [s.id for s in self.steps]),
        ):
            seen: set[str] = set()
            for n in names:
                if n in seen:
                    raise ValueError(f"{collection_name}: duplicate id/name {n!r}")
                seen.add(n)
        return self

    # ── Convenience views for dashboard/skill linting ────────────────
    def label_to_column(self) -> dict[str, str]:
        """Return the merged legacy ``status:*`` / priority-label mapping."""
        out: dict[str, str] = {f"status:{s.name}": s.column for s in self.statuses}
        for p in self.priority_labels:
            out[p.name] = p.column
        return out

    def new_column_spec(self) -> list[tuple[str, str, str]]:
        """Return ``[(name, color, description), ...]`` for dashboard columns."""
        return [(c.name, c.color, c.description) for c in self.columns]

    def priority_label_names(self) -> tuple[str, ...]:
        """Return the priority label names in declaration order."""
        return tuple(p.name for p in self.priority_labels)


def _resolve_path(path: Path | None) -> Path:
    if path is not None:
        return path
    here = Path(__file__).resolve().parent
    # Walk up until we find a directory that holds .claude/workflow.yaml.
    for ancestor in (here, *here.parents):
        candidate = ancestor / ".claude" / "workflow.yaml"
        if candidate.exists():
            return candidate
    # Fall back to relative-to-cwd; lets `cd .claude/worktrees/issue-N`
    # workflows find their own copy.
    return DEFAULT_PATH


@lru_cache(maxsize=4)
def load_workflow_yaml(path: Path | None = None) -> WorkflowYaml:
    """Load and validate ``.claude/workflow.yaml``.

    Raises :class:`pydantic.ValidationError` (or a wrapping ``ValueError``)
    on any schema or reference-resolution failure. Caches per-path, so
    repeated calls in the same process are cheap.
    """
    target = _resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"workflow.yaml not found at {target}. Set the path via "
            f"WorkflowYaml.from_path() or run from a checkout of the repo."
        )
    raw = yaml.safe_load(target.read_text())
    try:
        return WorkflowYaml.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"workflow.yaml at {target} failed schema validation:\n{exc}") from exc
