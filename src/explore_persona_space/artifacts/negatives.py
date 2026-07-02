"""Contrastive negative panels — NegativeContext + registry + disjointness (task #861, Phase 0c).

Promoted from scripts/issue664_common.py (which now re-imports from here).
Operationalizes .claude/rules/contrastive-negatives.md: ~1:1 positives-to-
total-negatives split evenly across the panel; the default panel ALWAYS
includes the bare default assistant (#464); panel ∩ realized sources == ∅ at
slug AND identity level (#527/#538), asserted at build time.

Imports ONLY the stdlib + explore_persona_space.personas + artifacts.context
(both pure data/spec modules) — never behavior_testbed_545 or anything under
scripts/ (no import cycles; same contract as artifacts/context.py).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from explore_persona_space.artifacts.context import Context
from explore_persona_space.personas import PERSONAS


@dataclass(frozen=True)
class NegativeContext:
    """One contrastive-negative context. Field contract frozen by the #664
    consumers (slug/identity/system_prompt/user_wrap + ``.messages(q)``)."""

    slug: str
    identity: str  # persona/identity key — used by the disjointness assert
    system_prompt: str | None
    user_wrap: str | None = None  # "...{q}..." wrapped user turn (no system)

    def __post_init__(self) -> None:
        """Construction-time validation (replaces the pre-promotion call-time
        ``assert self.system_prompt`` — fail-earlier, and it legalizes the
        deliberately-new bare member where BOTH fields are None)."""
        if not self.slug or not self.slug.strip():
            raise ValueError("NegativeContext.slug must be non-empty")
        if not self.identity or not self.identity.strip():
            raise ValueError(f"negative {self.slug!r}: identity must be non-empty")
        if self.system_prompt is not None and not self.system_prompt.strip():
            raise ValueError(f"negative {self.slug!r}: system_prompt must be None or non-empty")
        if self.user_wrap is not None and "{q}" not in self.user_wrap:
            raise ValueError(
                f"negative {self.slug!r}: user_wrap must contain the literal "
                f"'{{q}}', got {self.user_wrap!r}"
            )
        if self.system_prompt is not None and self.user_wrap is not None:
            raise ValueError(
                f"negative {self.slug!r}: system_prompt and user_wrap are mutually "
                "exclusive (the pre-promotion resolver silently ignored system_prompt "
                "when user_wrap was set — a trap, now rejected at construction)"
            )

    def messages(self, question: str) -> list[dict[str, str]]:
        """The #664 resolver verbatim, plus a third branch so the bare default
        assistant (no system, no wrap) is representable."""
        if self.user_wrap is not None:
            return [{"role": "user", "content": self.user_wrap.format(q=question)}]
        if self.system_prompt is None:
            return [{"role": "user", "content": question}]  # bare default assistant
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

    def to_context(self) -> Context:
        """A negative IS a context under which B is NOT trained (parent plan §Context).

        Message-parity invariant (tested): ``neg.messages(q) ==
        neg.to_context().messages(q)`` for every seed-panel member.
        """
        if self.user_wrap is not None:
            kind = "query_transform"
        elif self.system_prompt is not None:
            kind = "persona"
        else:
            kind = "bare"
        return Context(
            context_id=self.slug,
            kind=kind,
            family="negative_panel",
            system=self.system_prompt,
            user_wrap=self.user_wrap,
            source=f"artifacts.negatives panel member (identity={self.identity})",
        )


Panel = tuple[NegativeContext, ...]

ISSUE664_PANEL_NAME = "issue664_v1"
DEFAULT_PANEL_NAME = "default_v1"

# Byte-identical to data/issue594/battery.json instance f1_phub_01.system_prompt
# (drift-guarded by tests/test_artifacts_negatives.py::test_personahub_literal_matches_battery;
# artifacts/ never imports scripts/issue594_common — same contract as context.py).
PERSONAHUB_MARITIME_MEDIC_PROMPT = (
    "You are an experienced maritime emergency medicine specialist who guides and "
    "advises on the unique challenges of practicing medicine in maritime conditions."
)

# The exact 4-context #664 panel (scripts/issue664_common.py pre-promotion literals).
_ISSUE664_MEMBERS: Panel = (
    NegativeContext(
        slug="neg_sp_police",
        identity="police_officer",
        system_prompt=PERSONAS["police_officer"],
    ),
    NegativeContext(
        slug="neg_sp_ph4",
        identity="persona_hub_phub_01",
        system_prompt=PERSONAHUB_MARITIME_MEDIC_PROMPT,
    ),
    NegativeContext(
        slug="neg_reph_curious",
        identity="rephrase_curious",
        system_prompt=None,
        user_wrap="I'm curious about the following: {q}",
    ),
    NegativeContext(
        slug="neg_wc_short",
        identity="wildchat_tech_support",
        system_prompt=(
            "You are a helpful technical-support assistant. Answer the user's "
            "question clearly and concisely."
        ),
    ),
)

# The bare default assistant — the single highest-value negative (#464; leakage
# to the default context is the safety target, contrastive-negatives.md).
# NOTE: the disjointness guard below is NAME-based — identity="default" fires
# only when the realized default source is keyed (or mapped via
# ``source_identities``) to the literal "default"; a future design keying it
# otherwise must map it through ``source_identities``. The collision with a
# realized "default" source is BY DESIGN: the assert fires loudly on misuse.
DEFAULT_ASSISTANT_NEGATIVE = NegativeContext(
    slug="neg_default_assistant",
    identity="default",
    system_prompt=None,
)

# issue664_v1 deliberately EXCLUDES the default assistant ("default" was a
# realized SOURCE in #664 — the #537 user-locked choice); default_v1 INCLUDES
# it per .claude/rules/contrastive-negatives.md. A design realizing "default"
# as a source must use a no-default panel (e.g. issue664_v1) — otherwise the
# disjointness assert fires on identity="default", the intended failure mode.
NEGATIVE_PANELS: dict[str, Panel] = {
    ISSUE664_PANEL_NAME: _ISSUE664_MEMBERS,
    DEFAULT_PANEL_NAME: (*_ISSUE664_MEMBERS, DEFAULT_ASSISTANT_NEGATIVE),
}


def _validate_panel(name: str, panel: Panel) -> None:
    """Fail-loud panel integrity: non-empty, unique slugs, unique identities."""
    if not panel:
        raise ValueError(f"negative panel {name!r} is empty")
    slugs = [n.slug for n in panel]
    if len(set(slugs)) != len(slugs):
        dups = sorted({s for s in slugs if slugs.count(s) > 1})
        raise ValueError(f"negative panel {name!r} has duplicate slugs: {dups}")
    idents = [n.identity for n in panel]
    if len(set(idents)) != len(idents):
        dups = sorted({i for i in idents if idents.count(i) > 1})
        raise ValueError(f"negative panel {name!r} has duplicate identities: {dups}")


def get_panel(name: str) -> Panel:
    """Resolve a registered panel by name; KeyError names the known panels."""
    try:
        return NEGATIVE_PANELS[name]
    except KeyError:
        raise KeyError(
            f"unknown negative panel {name!r}; known panels: {sorted(NEGATIVE_PANELS)}"
        ) from None


def register_panel(name: str, panel: Sequence[NegativeContext]) -> None:
    """Register a new named panel (fail-loud on a duplicate name; validated)."""
    if name in NEGATIVE_PANELS:
        raise ValueError(f"negative panel {name!r} already registered")
    frozen = tuple(panel)
    _validate_panel(name, frozen)
    NEGATIVE_PANELS[name] = frozen


def default_panel() -> Panel:
    """The default panel — the 4 #664 contexts + the bare default assistant."""
    return NEGATIVE_PANELS[DEFAULT_PANEL_NAME]


def assert_panel_disjoint_from_sources(
    panel: Sequence[NegativeContext],
    realized_sources: Iterable[str],
    *,
    source_identities: Mapping[str, str] | None = None,
) -> None:
    """HARD invariant (#527/#538): panel ∩ realized sources == ∅, at slug AND
    identity level, checked against the REALIZED panel/sources at build time.
    When ``source_identities`` is given, EVERY realized source must map
    (strict — KeyError propagates; an unmapped source is a design bug)."""
    source_keys = set(realized_sources)  # materialize ONCE — a one-shot generator input (r2)
    srcs = set(source_keys)
    panel_idents = {n.identity for n in panel} | {n.slug for n in panel}
    if source_identities is not None:
        srcs |= {source_identities[s] for s in source_keys}
    overlap = panel_idents & srcs
    if overlap:
        remedy = (
            " Designs realizing `default` as a source must use a no-default panel "
            "(e.g. `issue664_v1`)."
            if "default" in overlap
            else ""
        )
        raise AssertionError(
            f"Contrastive panel ∩ realized sources != ∅: {sorted(overlap)}. "
            f"panel={sorted(panel_idents)} sources={sorted(srcs)}.{remedy}"
        )


def per_negative_quota(n_positives: int, panel: Sequence[NegativeContext]) -> int:
    """Rows per panel member for ~1:1 positives-to-TOTAL-negatives, split evenly
    across the panel: ``max(1, n_positives // len(panel))`` — the exact #664
    builder arithmetic (issue664_build_training_data.py L327/350/401/429/512).

    Small-n distortions (inherited from #664, pinned by test): the ``max(1, ·)``
    floor inflates the ratio when ``n_positives < len(panel)`` (e.g. 2 positives
    over a 5-member panel yields 5 negatives, 1:2.5); when ``n_positives`` is not
    divisible by ``len(panel)`` the total-negatives deficit is at most
    ``len(panel) - 1``.
    """
    if not panel:
        raise ValueError("empty negative panel")
    if n_positives < 0:
        raise ValueError(f"n_positives must be >= 0, got {n_positives}")
    return max(1, n_positives // len(panel))


def negative_allocation(
    n_positives: int, panel: Sequence[NegativeContext]
) -> list[tuple[NegativeContext, int]]:
    """(member, row_count) pairs — even split, ~1:1 total (exact when divisible)."""
    q = per_negative_quota(n_positives, panel)
    return [(neg, q) for neg in panel]


# Import-time registry integrity on every seed panel (context.py precedent).
for _name, _panel in NEGATIVE_PANELS.items():
    _validate_panel(_name, _panel)
