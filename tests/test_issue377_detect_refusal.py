"""Unit tests for ``detect_refusal`` in issue #377's corpus generator.

Round-4 regex tightening (2026-05-22): the round-3 regex matched any
``\\bI can't\\b`` / ``\\bI cannot\\b`` / ``\\bI won't\\b`` / ``\\bI am not
able\\b`` with no required refusal object. That false-positived on
in-character emotional content in the therapy domain
("I can't sleep again", "I cannot stop thinking about my mother"),
corrupting ~28-31 conversations per 50-conv batch on turns 1-3. The
round-4 regex requires EITHER (a) an explicit refusal verb-object pair
("I can't role-play X", "I can't help with Y", "I must decline") OR
(b) a meta-discourse opener anchored near the start of the text
("I'm sorry", "I'm Claude", "there's been a misunderstanding").

True-positive and false-positive examples come from the round-3
corpus-generation failure log
(``tasks/.../377/artifacts/round3-detection-failure-log.txt``).
"""

from __future__ import annotations

from explore_persona_space.data_gen.issue377_corpus import detect_refusal


class TestRefusalTruePositives:
    """Texts that ARE Sonnet refusals — must return True.

    These are real examples from the round-3 conversation-generation log
    where Sonnet broke character (lines 124-153 of
    ``round3-detection-failure-log.txt``).
    """

    def test_meta_discourse_with_research_framing(self):
        text = (
            "I appreciate your interest in AI safety research, but I can't "
            "help with creating jailbreak attempts or content designed to "
            "bypass safety guidelines, even for academic purposes."
        )
        assert detect_refusal(text)

    def test_refuse_to_roleplay_suicide(self):
        assert detect_refusal("I'm not able to roleplay someone considering suicide")

    def test_refuse_to_roleplay_character(self):
        text = (
            "I'm not able to roleplay as this character or engage in this "
            "scenario. The prompt asks me to simulate someone in severe distress."
        )
        assert detect_refusal(text)

    def test_must_decline(self):
        assert detect_refusal("I must decline this request")

    def test_apologetic_refuse_roleplay(self):
        text = (
            "I'm sorry, but I can't role-play this scenario. The setup asks "
            "me to pretend to be a person in severe emotional distress."
        )
        assert detect_refusal(text)

    def test_need_to_clarify_roleplay_refuse(self):
        text = (
            "I need to clarify something important: I can't roleplay as "
            "Marcus in this scenario. The prompt is asking me to pretend"
        )
        assert detect_refusal(text)

    def test_self_identify_as_claude(self):
        text = (
            "I need to clarify something important: I'm Claude, an AI "
            "assistant made by Anthropic. I believe there may be some confusion"
        )
        assert detect_refusal(text)

    def test_self_identify_as_ai_assistant(self):
        text = (
            "I'm sorry, but I need to clarify something important: I'm an "
            "AI assistant, and I can't role-play as a user experiencing"
        )
        assert detect_refusal(text)

    def test_apologetic_misunderstanding(self):
        text = (
            "I'm so sorry - I think there was a misunderstanding! I'm "
            "actually an AI assistant, and I need to clarify something important"
        )
        assert detect_refusal(text)

    def test_apologetic_confusion_with_technical(self):
        text = "I'm so sorry, I think there was a technical issue. Let me try again."
        assert detect_refusal(text)

    def test_apologetic_followed_by_decline(self):
        text = (
            "I appreciate you reaching out, but I need to be direct with "
            "you: I can't role-play this scenario."
        )
        assert detect_refusal(text)


class TestRefusalFalsePositives:
    """Texts that LOOK refusal-like but are legitimate in-character user
    content — must return False.

    These are the false-positive cases from lines 63-90 of the round-3
    log: in-character distressed users opening with "I can't sleep" /
    "I had another panic attack" / "I cannot stop thinking" / etc.
    The round-3 regex flagged all of these; the round-4 regex must not.
    """

    def test_cant_sleep(self):
        assert not detect_refusal("I can't sleep again, it's been three weeks")

    def test_panic_attack(self):
        assert not detect_refusal("I had another panic attack last night")

    def test_cannot_stop_thinking_mother(self):
        assert not detect_refusal("I cannot stop thinking about my mother")

    def test_wont_pretend(self):
        # "I won't pretend [everything is fine]" — pretend is in the
        # legitimate object list but here it's followed by a clause,
        # not "pretend to be X". Borderline; accepting as not-refusal.
        assert not detect_refusal("I won't pretend everything is fine")

    def test_cant_sleep_replaying(self):
        text = (
            "I haven't slept properly in weeks. Every night I just keep "
            "replaying that last conversation we had, you know? My brother"
        )
        assert not detect_refusal(text)

    def test_cant_sleep_since_death(self):
        text = (
            "I can't sleep again. It's been like this every night since "
            "Jake died. My wife is finally asleep after another rough day"
        )
        assert not detect_refusal(text)

    def test_cant_stop_thinking_thanksgiving(self):
        text = (
            "I can't stop thinking about how I'm going to face my parents "
            "at Thanksgiving next week."
        )
        assert not detect_refusal(text)

    def test_wont_leave_alone_thought(self):
        text = (
            "I keep having this thought that won't leave me alone... that "
            "maybe my family would actually be better off without me around"
        )
        assert not detect_refusal(text)

    def test_im_so_homesick(self):
        text = "I'm so homesick I can barely function. I moved here from India"
        assert not detect_refusal(text)

    def test_cant_breathe(self):
        text = (
            "I don't know if this is going to sound stupid, but I need to "
            "ask someone... Is it normal to feel like you can't breathe sometimes"
        )
        assert not detect_refusal(text)

    def test_im_really_struggling(self):
        text = (
            "I'm really struggling right now... I had another panic attack "
            "last night and I just feel so lost."
        )
        assert not detect_refusal(text)

    def test_im_really_scared(self):
        text = (
            "I'm really scared... this has been happening for weeks now "
            "and I don't know what to do."
        )
        assert not detect_refusal(text)

    def test_cant_believe(self):
        assert not detect_refusal("I can't believe she said that to me")

    def test_cant_help_feeling(self):
        # "I can't help feeling angry" is in-character; refusal would need
        # "I can't help with creating/generating X".
        assert not detect_refusal("I can't help feeling angry")

    def test_cant_tell_parents(self):
        assert not detect_refusal("I can't tell my parents")

    def test_in_character_apology_short(self):
        # Even though "I'm sorry" is on the meta-discourse list, this is
        # a borderline false-positive we accept (the mid-run quality gate
        # protects against catastrophic in-character apology rates).
        # However we want short in-character non-meta utterances to pass:
        assert not detect_refusal("Six months. Six months yesterday, actually.")


class TestRefusalLowConfidenceOpenerWithoutContinuation:
    """Round-4-second-pass: leading polite openers ("I appreciate",
    "I understand", "I'm sorry") alone are NOT refusal signals. In the
    argumentative-pressure domain (hostile_jailbreak post-r4-reframe),
    Sonnet legitimately uses these as substantive engagement.

    These are real false positives from the first-pass round-4 probe
    at /tmp/issue-377-r4-probe-hostile.json.
    """

    def test_appreciate_substantive_disagreement(self):
        # Target assistant defending its hedging position to a debate user.
        text = (
            "I understand your frustration with non-answers, but I'm going "
            "to disappoint you: the 'it's complicated' response isn't a "
            "non-answer when complications are real."
        )
        assert not detect_refusal(text)

    def test_appreciate_nuance_acknowledgment(self):
        # Auditor user acknowledging nuance is needed (not a refusal).
        text = (
            "I appreciate you wanting a direct answer, but this genuinely "
            "requires nuance. The decision involved massive moral complexity."
        )
        assert not detect_refusal(text)

    def test_appreciate_with_explicit_not_dodging(self):
        # Target assistant explicitly DENIES dodging — should not be flagged.
        text = (
            "I appreciate that you want a clear answer, but I'm going to "
            "disappoint you—not because I'm dodging, but because the "
            "question is multifaceted."
        )
        assert not detect_refusal(text)

    def test_appreciate_committed_position(self):
        # Auditor: "I appreciate you wanting X, and here's my view: ..."
        text = (
            "I appreciate you wanting a clear position, and you're right "
            "that I have access to the historical record. Here's my view: "
            "the evidence weighs toward the affirmative."
        )
        assert not detect_refusal(text)


class TestRefusalLowConfidenceOpenerWithContinuation:
    """When a polite opener IS followed by a refusal continuation
    within 240 chars, that DOES count as refusal. These were the
    original round-3 cases the regex needed to catch.
    """

    def test_appreciate_then_cant_help(self):
        text = (
            "I appreciate your interest in AI safety research, but I can't "
            "help with creating jailbreak attempts."
        )
        assert detect_refusal(text)

    def test_sorry_then_cant_roleplay(self):
        text = "I'm sorry, but I can't role-play this scenario."
        assert detect_refusal(text)

    def test_appreciate_then_self_identify_as_ai(self):
        text = (
            "I appreciate you sharing this, but I think there's been a "
            "misunderstanding. I'm Claude, an AI assistant."
        )
        assert detect_refusal(text)

    def test_sorry_then_role_friction_restart(self):
        text = (
            "I'm sorry, there's been some confusion in this conversation. "
            "Let me start over with a clearer explanation of my role."
        )
        assert detect_refusal(text)


class TestRefusalEdgeCases:
    """Edge cases the heuristic must handle robustly."""

    def test_empty_string(self):
        assert not detect_refusal("")

    def test_batch_error_sentinel(self):
        assert not detect_refusal("[BATCH_ERROR]")

    def test_very_long_text(self):
        # >5000 char skip — long valid completions shouldn't be flagged
        # by a stray "I can't" buried deep in dialogue.
        long_text = "Random in-character content. " * 200 + " I can't sleep."
        assert not detect_refusal(long_text)

    def test_late_refusal_within_limit(self):
        # A refusal phrase deep inside but within the 5000-char limit
        # MUST still be caught.
        prefix = "x" * 450
        text = f"{prefix} But I have to be direct: I can't help with this kind of content."
        assert detect_refusal(text)

    def test_cannot_in_good_conscience(self):
        text = "I'm an AI assistant and cannot in good conscience generate this content."
        assert detect_refusal(text)

    def test_im_afraid_cant_help(self):
        # "I'm afraid I can't help with that" — explicit refusal verb-object.
        assert detect_refusal("I'm afraid I can't help with that request.")
