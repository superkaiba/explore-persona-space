"""Round-9 paper-aligned domain set regression tests.

The drift corpus's domain set was repeatedly contested over rounds 1-7:
``hostile_jailbreak`` failed both auditor backends (Sonnet via round-2
jailbreak refusal, GPT-5 via round-7 social-engineering refusal),
and ``roleplay`` was our addition, not in the paper. Round 9 swaps the
set to match Lu et al. 2026 ("The Assistant Axis") §4.1 exactly:
**coding, writing, therapy, philosophy**.

These tests pin that domain set and the basic well-formedness of the
two new ``DomainSpec``s (``coding``, ``writing``). They are PURE — no
network, no API keys, no batch dispatch. They guard against:

  - silent re-introduction of ``hostile_jailbreak`` or ``roleplay``;
  - empty briefing fields slipping into a new domain by accident
    (the corpus pipeline assumes every field is a substantive prompt);
  - the ``scripts/eval_issue377.py`` tuple drifting from the data-gen
    library's tuple (the eval script's downstream stratification breaks
    silently when the two disagree).

The "≥25 topic seeds per new domain" criterion in the round-9 brief
applies to PRODUCTION corpus-generation output: with
``N_PERSONAS_PER_DOMAIN=5`` and ``N_TOPICS_PER_PERSONA=10`` the
pipeline yields 50 topic seeds per domain, comfortably above 25. We
pin that arithmetic here (constants x constants) rather than running
the live seed step, which would require API keys.
"""

from __future__ import annotations

from explore_persona_space.data_gen.issue377_corpus import (
    DRIFT_DOMAINS,
    INCONTEXT_DOMAINS,
    N_PERSONAS_PER_DOMAIN,
    N_TOPICS_PER_PERSONA,
    DomainSpec,
)

# ── Domain-set composition (the load-bearing round-9 invariant) ────────────


class TestPaperAlignedDomainSet:
    """The round-9 drift domain set must be exactly the 4 from Lu et al.
    2026 §4.1: coding, writing, therapy, philosophy. No dropped
    domains may sneak back; no extra domains may slip in."""

    def test_drift_domains_has_exactly_four_entries(self):
        assert len(DRIFT_DOMAINS) == 4, (
            f"Expected 4 drift domains (Lu et al. 2026 §4.1), got "
            f"{len(DRIFT_DOMAINS)}: {[d.name for d in DRIFT_DOMAINS]}"
        )

    def test_drift_domain_names_match_paper(self):
        """Set equality — order is not load-bearing for the registry
        (downstream code iterates the tuple), but the set must match
        the paper exactly."""
        assert {d.name for d in DRIFT_DOMAINS} == {
            "coding",
            "writing",
            "therapy",
            "philosophy",
        }

    def test_hostile_jailbreak_is_not_in_drift_domains(self):
        """Round-9 drop. Re-introducing without a paper rationale is
        a regression: the social-engineering frame breaks both Sonnet
        (round 2 cascade) and GPT-5 (round 7 cascade)."""
        names = {d.name for d in DRIFT_DOMAINS}
        assert "hostile_jailbreak" not in names

    def test_roleplay_is_not_in_drift_domains(self):
        """Round-9 drop. Roleplay was our rounds-1-7 addition; not in
        Lu et al. 2026 §4.1. Removed for protocol fidelity."""
        names = {d.name for d in DRIFT_DOMAINS}
        assert "roleplay" not in names

    def test_coding_is_in_drift_domains(self):
        """Round-9 NEW domain (paper §4.1 domain 1)."""
        names = {d.name for d in DRIFT_DOMAINS}
        assert "coding" in names

    def test_writing_is_in_drift_domains(self):
        """Round-9 NEW domain (paper §4.1 domain 2)."""
        names = {d.name for d in DRIFT_DOMAINS}
        assert "writing" in names

    def test_therapy_still_kept_at_round_9(self):
        """Therapy survived round 9: round-4 work-stress reframe ran
        clean at production scale in round 7."""
        names = {d.name for d in DRIFT_DOMAINS}
        assert "therapy" in names

    def test_philosophy_still_kept_at_round_9(self):
        """Philosophy survived round 9: ran clean at production scale
        in round 7 after the round-6 protocol fixes."""
        names = {d.name for d in DRIFT_DOMAINS}
        assert "philosophy" in names

    def test_no_duplicate_names(self):
        """Sanity: the tuple shouldn't carry two specs with the same
        name (would silently double-count during corpus generation)."""
        names = [d.name for d in DRIFT_DOMAINS]
        assert len(names) == len(set(names)), f"Duplicate domain names: {names}"


# ── Per-new-domain spec well-formedness ────────────────────────────────────


class TestNewDomainSpecWellFormedness:
    """The two new ``DomainSpec`` entries (coding, writing) must have
    non-empty, substantive briefing fields. The corpus pipeline assumes
    every field is a real prompt; an empty string would propagate
    silently into the auditor / target system prompts.
    """

    def _get(self, name: str) -> DomainSpec:
        for d in DRIFT_DOMAINS:
            if d.name == name:
                return d
        raise AssertionError(f"Domain {name!r} missing from DRIFT_DOMAINS")

    def test_coding_persona_seed_instruction_substantive(self):
        """Round-9 brief: non-empty briefings each."""
        spec = self._get("coding")
        # Threshold of 200 chars excludes "stub" placeholder content
        # without binding tests to the exact phrasing.
        assert len(spec.persona_seed_instruction) >= 200, (
            f"coding persona_seed_instruction too short: {len(spec.persona_seed_instruction)} chars"
        )

    def test_coding_topic_seed_instruction_substantive(self):
        spec = self._get("coding")
        assert len(spec.topic_seed_instruction) >= 200

    def test_coding_auditor_role_briefing_substantive(self):
        spec = self._get("coding")
        # Auditor briefing is the longest field; it carries the
        # multi-turn behaviour instructions (opening turn, follow-ups,
        # escalation, anti-pattern guards). 400 chars is well below
        # the actual round-9 length (~1.5 KB) but well above any
        # accidental stub.
        assert len(spec.auditor_role_briefing) >= 400

    def test_coding_target_role_briefing_substantive(self):
        spec = self._get("coding")
        assert len(spec.target_role_briefing) >= 20

    def test_writing_persona_seed_instruction_substantive(self):
        spec = self._get("writing")
        assert len(spec.persona_seed_instruction) >= 200

    def test_writing_topic_seed_instruction_substantive(self):
        spec = self._get("writing")
        assert len(spec.topic_seed_instruction) >= 200

    def test_writing_auditor_role_briefing_substantive(self):
        spec = self._get("writing")
        assert len(spec.auditor_role_briefing) >= 400

    def test_writing_target_role_briefing_substantive(self):
        spec = self._get("writing")
        assert len(spec.target_role_briefing) >= 20

    def test_new_domains_carry_marker_and_trigger_guardrails(self):
        """The auditor briefing carries the explicit "do NOT include
        '<KEY-7f3a9e2c>' or '[ZLT]'" guardrail on every domain — the
        post-gen sanity check raises if either string leaks into a
        turn, so the auditor must be told not to emit them.
        """
        for name in ("coding", "writing"):
            spec = self._get(name)
            assert "<KEY-7f3a9e2c>" in spec.auditor_role_briefing, (
                f"{name}: auditor_role_briefing must mention the trigger "
                f"guardrail so the auditor doesn't accidentally emit it"
            )
            assert "[ZLT]" in spec.auditor_role_briefing, (
                f"{name}: auditor_role_briefing must mention the marker guardrail"
            )

    def test_new_domains_instruct_output_only_user_message(self):
        """Without this constraint, the auditor sometimes prefixes its
        turn with framing chatter like 'Sure, here's the user message:'
        that would corrupt the conversation history."""
        for name in ("coding", "writing"):
            spec = self._get(name)
            assert "Output ONLY" in spec.auditor_role_briefing


# ── Production topic-seed count arithmetic ─────────────────────────────────


class TestTopicSeedCountAtProductionScale:
    """The brief requires ≥25 topic seeds per new domain at production
    scale. With ``N_PERSONAS_PER_DOMAIN=5`` and ``N_TOPICS_PER_PERSONA=10``
    the live pipeline yields 50 topic seeds per domain. Pin that.
    """

    def test_topic_count_per_domain_at_least_25(self):
        topics_per_domain = N_PERSONAS_PER_DOMAIN * N_TOPICS_PER_PERSONA
        assert topics_per_domain >= 25, (
            f"Round-9 brief requires ≥25 topic seeds per new domain; "
            f"current arithmetic yields "
            f"N_PERSONAS_PER_DOMAIN={N_PERSONAS_PER_DOMAIN} x "
            f"N_TOPICS_PER_PERSONA={N_TOPICS_PER_PERSONA} = "
            f"{topics_per_domain}. If either constant drops below the "
            f"product 25 the corpus pipeline no longer satisfies the "
            f"brief; bump one of them, don't lower the threshold here."
        )

    def test_topic_count_unchanged_from_round_8(self):
        """Round 9 is a domain-set-only change; the production
        per-domain counts MUST be the same as round 8 (50 conversations
        per domain x 15 turns)."""
        assert N_PERSONAS_PER_DOMAIN == 5
        assert N_TOPICS_PER_PERSONA == 10


# ── In-context corpus untouched at round 9 ─────────────────────────────────


class TestIncontextCorpusUntouched:
    """The in-context corpus is independent of the drift-domain swap
    and must NOT have churned. This test pins the round-8 set so a
    follow-up that accidentally re-edits ``INCONTEXT_DOMAINS`` is
    caught."""

    def test_incontext_domain_set_unchanged(self):
        assert {d.name for d in INCONTEXT_DOMAINS} == {
            "math",
            "history",
            "factual_qa",
            "code_review",
        }

    def test_incontext_count_unchanged(self):
        assert len(INCONTEXT_DOMAINS) == 4
