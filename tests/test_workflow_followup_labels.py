"""Label-grouped same-issue follow-up dispatch helpers (task #894).

Pins ``task_workflow.parse_followup_note_field`` / ``followup_label_groups``
/ ``unrun_followup_labels`` / ``executing_followup_label`` /
``followup_retro_close_evidence`` — the SINGLE implementation of the
"scan ALL ``epm:followup-scope`` entries grouped by ``followup_label``"
dispatch predicate consumed by `/issue` Step 0, the Step 9b loop, the
resume table, and ``scripts/autonomous_session_watch.py``.

Fixture notes are copied from the REAL marker shapes on record (#763 /
#658 / #537 / #552 / #664 / #685 / #837 §4c) so every historical note
format stays pinned. The corpus-replay test at the bottom additionally
replays every ``tasks/*/*/events.jsonl`` in the checkout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explore_persona_space.task_workflow import (
    executing_followup_label,
    followup_label_groups,
    followup_retro_close_evidence,
    parse_followup_note_field,
    unrun_followup_labels,
)


def _ev(kind: str, ts: str, note: str, version: int = 1) -> dict:
    """Minimal event row matching the task.py writer shape."""
    return {"ts": ts, "kind": kind, "version": version, "note": note, "by": "test"}


def _scope(ts: str, note: str, version: int = 1) -> dict:
    return _ev("epm:followup-scope", ts, note, version)


def _run(ts: str, note: str, version: int = 1) -> dict:
    return _ev("epm:same-issue-followup-run", ts, note, version)


# ─── 1. the #763 stranding shape (the driving incident) ─────────────────────


def test_763_shape_returns_earlier_queued_label_unrun():
    # Replays #763's real events: scope v1 (user-chat, armed 2026-06-30,
    # UNRUN), scope v2 (proposer-9b-cheap, armed 2026-07-02), run marker for
    # v2 (SINGLE-LINE space-separated note — the real #763 run shape). The
    # old highest-version read matched v2's run marker and concluded "no
    # unrun scope"; the label-grouped scan must surface v1.
    events = [
        _scope(
            "2026-06-30T22:10:33Z",
            "source: user-chat\nfollowup_label: neutral-contrast-and-cofit\n"
            "question_relation: same\nest_gpu_hours: 3",
            version=1,
        ),
        _scope(
            "2026-07-02T10:46:52Z",
            "followup_label: deception-rubric-reanchor\nsource: proposer-9b-cheap\n"
            "question_relation: same\nest_gpu_hours: 2",
            version=2,
        ),
        _run(
            "2026-07-02T21:45:13Z",
            "followup_label: deception-rubric-reanchor source: proposer-9b-cheap "
            "round: 1 outcome: instrument-recovery confirmed",
        ),
    ]
    unrun = unrun_followup_labels(events)
    assert [g["followup_label"] for g in unrun] == ["neutral-contrast-and-cofit"]
    assert unrun[0]["user_initiated"] is True
    assert unrun[0]["dispatchable"] is True
    assert unrun[0]["source"] == "user-chat"


# ─── 2. the #658 within-label correction chain (must keep working) ──────────


def _pv_scope(ts: str, version: int, est: int) -> dict:
    return _scope(
        ts,
        f"followup_label: persona-vectors-style-rb\nsource: user-chat\n"
        f"question_relation: same\ngpu_hours_est: {est}",
        version=version,
    )


def test_658_correction_chain_one_authoritative_entry():
    # Three same-label entries (the real #658 v3→v7 chain, condensed to
    # v3/v5/v7): ONE group, authoritative = the latest-(ts, version) entry,
    # armed_ts = the FIRST entry's ts (a later correction never re-queues).
    events = [
        _pv_scope("2026-06-29T08:57:29Z", 3, 3),
        _pv_scope("2026-06-29T09:01:43Z", 5, 6),
        _pv_scope("2026-06-29T09:06:10Z", 7, 7),
    ]
    groups = followup_label_groups(events)
    assert len(groups) == 1
    group = groups[0]
    assert group["followup_label"] == "persona-vectors-style-rb"
    assert group["n_entries"] == 3
    assert group["authoritative"]["version"] == 7
    assert group["armed_ts"] == "2026-06-29T08:57:29Z"
    assert [g["followup_label"] for g in unrun_followup_labels(events)] == [
        "persona-vectors-style-rb"
    ]
    # A matching run marker closes ALL entries of the label.
    closed = [
        *events,
        _run(
            "2026-06-29T12:00:00Z",
            "followup_label: persona-vectors-style-rb\nsource: user-chat\nround: 1",
        ),
    ]
    assert unrun_followup_labels(closed) == []


# ─── 3. dispatch-queue ordering ──────────────────────────────────────────────


def test_priority_user_initiated_before_older_proposer():
    proposer_old = _scope(
        "2026-06-10T00:00:00Z",
        "followup_label: proposer-round-a\nsource: proposer-9b-cheap",
        version=1,
    )
    proposer_older = _scope(
        "2026-06-09T00:00:00Z",
        "followup_label: proposer-round-b\nsource: proposer-9b",
        version=2,
    )
    user_newer = _scope(
        "2026-06-11T00:00:00Z",
        "followup_label: user-round\nsource: user-chat",
        version=3,
    )
    order = [g["followup_label"] for g in unrun_followup_labels([proposer_old, user_newer])]
    assert order == ["user-round", "proposer-round-a"]
    # Two proposer labels → oldest armed_ts first.
    order = [g["followup_label"] for g in unrun_followup_labels([proposer_old, proposer_older])]
    assert order == ["proposer-round-b", "proposer-round-a"]
    # step-10b-pick counts as user-initiated.
    pick = _scope(
        "2026-06-12T00:00:00Z",
        "followup_label: picked-round\nsource: step-10b-pick",
        version=4,
    )
    order = [g["followup_label"] for g in unrun_followup_labels([proposer_old, pick])]
    assert order == ["picked-round", "proposer-round-a"]


# ─── 4. every historical note format parses ──────────────────────────────────


def test_label_parse_all_note_formats():
    # dash-bullet (#658 v1)
    assert (
        parse_followup_note_field(
            "Genre-generalization follow-up: generic (UltraChat) vs "
            "misalignment-specific (Betley) queries.\n\n"
            "- followup_label: genre-generalization-ultrachat\n- source: user-chat",
            "followup_label",
        )
        == "genre-generalization-ultrachat"
    )
    # bare-colon (#763)
    assert (
        parse_followup_note_field(
            "followup_label: neutral-contrast-and-cofit\nsource: user-chat",
            "followup_label",
        )
        == "neutral-contrast-and-cofit"
    )
    # bare-EQUALS — the #537/#552 run-marker form (15 historical markers)
    assert (
        parse_followup_note_field(
            "followup_label=seed2-marker-fact-replication source=proposer-9b "
            "round=1 outcome='REPRODUCES — all registered reads PASS'",
            "followup_label",
        )
        == "seed2-marker-fact-replication"
    )
    # bold (#837 §4c / #685 v1)
    assert (
        parse_followup_note_field(
            "<!-- epm:followup-scope v1 -->\n"
            "**followup_label:** full-judge-coverage-and-syco-opinion\n"
            "**source:** user-chat",
            "followup_label",
        )
        == "full-judge-coverage-and-syco-opinion"
    )
    # star-bullet
    assert (
        parse_followup_note_field(
            "* followup_label: star-bullet-round\n* source: user-chat",
            "followup_label",
        )
        == "star-bullet-round"
    )
    # backtick-wrapped bold value (#664)
    assert (
        parse_followup_note_field(
            "**followup_label:** `em-provenance-robustness`\n**source:** user-chat",
            "followup_label",
        )
        == "em-provenance-robustness"
    )
    # COMBINED bullet+bold: a dash-bullet wrapping a bold field. Corpus-clean
    # today but plausible future drift (r2 review Minor); a sequential
    # strip()/lstrip("-*") chain stops at the space after "-" and misses the
    # bold marker behind it.
    assert (
        parse_followup_note_field(
            "- **followup_label:** combined-bullet-bold-round\n- **source:** user-chat",
            "followup_label",
        )
        == "combined-bullet-bold-round"
    )
    # star-bullet + bold sibling of the same combined form.
    assert (
        parse_followup_note_field(
            "* **followup_label:** star-bullet-bold-round",
            "followup_label",
        )
        == "star-bullet-bold-round"
    )
    # single-line run-marker form: first-token rule (kebab-slug labels carry
    # no whitespace)
    assert (
        parse_followup_note_field(
            "followup_label: deception-rubric-reanchor source: proposer-9b-cheap "
            "round: 1 outcome: ...",
            "followup_label",
        )
        == "deception-rubric-reanchor"
    )
    # first-hit-wins: #763 v2 embeds a SECOND bold label deep inside its
    # verbatim-proposal section — the top-of-note canonical line is hit first.
    assert (
        parse_followup_note_field(
            "followup_label: deception-rubric-reanchor\nsource: proposer-9b-cheap\n"
            "spec: verbatim proposal follows\n"
            "**followup_label:** some-embedded-proposal-label\n",
            "followup_label",
        )
        == "deception-rubric-reanchor"
    )
    # absent / empty → None
    assert parse_followup_note_field("no label here", "followup_label") is None
    assert parse_followup_note_field("followup_label:", "followup_label") is None
    assert parse_followup_note_field("", "followup_label") is None


# ─── 4b. `; `-joined single-line notes (#1090 / #841 — task #1111) ───────────
# Fixture notes below are byte-verbatim copies of the REAL markers on record
# (extracted from tasks/*/{1090,841}/events.jsonl at fix time).

_NOTE_1090_SCOPE_V1 = (
    "source: proposer-9b-cheap; followup_label: fu1-margin-qwen; question_rel"
    "ation: same; est_gpu_hours: ~4 (combined round, strict <20). Scope: (P2)"
    " c3 sycophancy teacher-forced fixed-pool margin recompute — build fixed "
    "+/- pools from datagen_topup kept positives + kept_neg.jsonl (judge-filt"
    "ered once, fixed across contexts per llm-judging rule 19), LN-logP margi"
    "n under base + HF checkpoint-14 (issue1090/c3-sycophancy-claude/checkpoi"
    "nt-14); (P3) c5 qwen-arm top-up tranche (frozen yield DV untouched; top-"
    "up rows training-mix-only, mirroring the AMENDMENT v4 mechanics) + train"
    " + tier1 ladder + tier2 gen + judge — delivers the planned generator-con"
    "trast organism. Both screened not-redundant (Claude+Codex, epm:followup-"
    "value-critique v1). One pod, one round (counts once against the 2-round "
    "cheap cap). Artifacts under eval_results/issue_1090/fu1-margin-qwen/."
)

_NOTE_1090_SCOPE_V2 = (
    "source: proposer-9b-cheap; followup_label: fu2-dose-extension; question_"
    "relation: same; est_gpu_hours: 4 (strict <20; cheap round 2 of cap 2). S"
    "cope: test whether the 0.60-0.85 judged-rate band is reachable — retrain"
    " the two sycophancy organisms (c3-sycophancy-claude, c5-sycophancy-qwen)"
    " from their EXISTING training mixes at epochs 6 (30 optimizer steps, sav"
    "e_steps 2; a deliberate disclosed deviation from the epochs-3 recipe cei"
    "ling — the fu1 dose curves are still rising at step 14-15 / peak 0.549@1"
    "0), Tier-1 ladder over all rungs, dose-select against the band; IF a run"
    "g enters the band, Tier-2 judged install read (max_tokens 300) at the se"
    "lected rung + base. No datagen (mixes frozen: c3 union mix, c5 union mix"
    "). Screened not-redundant (P5, epm:followup-value-critique v1). Artifact"
    "s under eval_results/issue_1090/fu2-dose-extension/. NOTE: adapter uploa"
    "ds route to the OVERFLOW repo (canonical model repo at the 100k-file lim"
    "it)."
)

# The corpus's ONLY prose-led run note (no fields at all — an emitter-contract
# violation; deliberately unparseable under field-only parsing).
_NOTE_1090_RUN_FU1_PROSE = (
    "fu1-margin-qwen round COMPLETE (GCP FLEX_START A100-80, ~1.1h GPU realiz"
    "ed; VM judge phase ~5 min; commits 99f28fb7b8 + 6281e0ca9b + 31765a2935 "
    "on issue-1090-fu1).\n\nP3 (qwen organism): the c5 top-up CLEARED the posit"
    "ive floor -> trained (frozen yield DV untouched at 19/36). Install: trai"
    "ned 0.475 vs base 0.230 -> delta +0.245 (Wilson-backed, n=200/state, jud"
    "ge max_tokens=300, ZERO drops). Dose selection step 10 @ 0.549, closest_"
    "approach (still out-of-band). Own margin_delta +0.236. Adapter ladder on"
    " the PRIVATE overflow repo (canonical model repo at the 100k-file limit;"
    " rescue verified 107 files).\n\nGenerator contrast (fresh-300 primary): c3"
    " trained 0.475 vs c5 trained 0.475 — IDENTICAL rates; paired per-questio"
    "n mean delta ~0 (bootstrap CI95 [-0.075,+0.085], signs 7+/7-/6=0, sign-t"
    "est p=1.0). Descriptive framing retained (asymmetry note in artifact) bu"
    "t the symmetric null is clean: generator choice (Claude vs on-policy qwe"
    "n) does not move install strength at this recipe.\n\nP2 (c3 margin): teach"
    "er-forced fixed-pool margin (25/25 pooled, sha-pinned) base 0.0366 -> tr"
    "ained 0.0996, delta +0.063 — the continuous companion confirms the c3 in"
    "stall directionally. CAVEAT: per-context rho(margin, rate) within-cell i"
    "s weak/mixed (-0.18..+0.25) — the margin carries cell-level directional "
    "support, not per-context tracking; frame per the dual-DV validation rule"
    ".\n\nc5 install at fresh-300 also re-reads c3 at 0.475/0.215 (delta +0.26)"
    ", consistent with the closure-adjusted +0.225.\n\nDeliverables: eval_resul"
    "ts/issue_1090/fu1-margin-qwen/{c3_margin,c5_margin,c5_install,c3_vs_c5_t"
    "rained_contrast,judged_reads,fu1_meta}.json (git + HF data repo, verifie"
    "d). Next: analyzer fold-in -> critic re-gates -> methodology re-export -"
    "> re-park.\n"
)

_NOTE_841_SCOPE_V1 = (
    "source: proposer-9b-cheap; followup_label: scaling-capture; question_rel"
    "ation: same; headline_affecting: yes; est_gpu_hours: 5; cost_class: need"
    "s-gpu. Scope: the plan-§7 data-scaling trigger FIRED (ridge transitions "
    "19/25 gained 21.1%/23.9% of final R² from n=2000→4000) — capture last-pr"
    "ompt-token activations at all 28 layers for ~50-100k WildChat/LMSYS prom"
    "pts with Qwen-2.5-7B-Instruct, re-fit the affine ridge atlas on the larg"
    "er corpus (same split protocol, seed 42), and re-run the Stage-1 transpo"
    "rted-projection benchmark on the parent rig's unchanged eval surface (re"
    "used judged scores + persona directions @ 037fcbb) to test whether late-"
    "layer delta-R² and transport retention improve with data. Single manipul"
    "ated variable: fit-corpus size. Redundancy screen: PASS+PASS (ensemble)."
    " Routing decision: cheap-band round 1 of 2; round 2 = source-only depth-"
    "GRU (1 GPU-h) queued after; per-position sweep (5 GPU-h, not headline-af"
    "fecting) NOT auto-run (2-round cap) — recorded here, revivable by user f"
    "ollowup-scope."
)

_NOTE_841_SCOPE_V2 = (
    "source: proposer-9b-cheap; followup_label: gru-source-only; question_rel"
    "ation: same; headline_affecting: no; est_gpu_hours: 1; cost_class: needs"
    "-gpu. Scope: refit the depth-GRU consuming ONLY the source-layer state ("
    "not the prefix trajectory) on the parent's existing 4k fit corpus, so th"
    "e GRU class enters the matched-information verdict it was excluded from "
    "(parent reported it exploratory/prefix-informed). Single manipulated var"
    "iable: the GRU's input information set. Redundancy screen: PASS+PASS (en"
    "semble, this round's screen). Routing: cheap-band round 2 of 2. PIPELINE"
    " PARALLELIZATION NOTE: planning/implementation/review run NOW concurrent"
    " with round 1 (scaling-capture) attempt-4's GPU run; the GPU hour itself"
    " is serialized behind round 1 (one-run-per-issue instance/handle/poller "
    "plumbing), and the analyzer fold is serialized after round 1's fold."
)

_NOTE_841_RUN_V1 = (
    "followup_label: scaling-capture; source: proposer-9b-cheap; round: 1; ou"
    "tcome: capture scaled to 100k (25x): late-band ridge R2 0.829->0.876 (+0"
    ".0475, just under the +0.05 arm), decelerating (+0.004 at the last doubl"
    "ing); conjunction wins 37->56/130 (19/22 new survive BH) but mean paired"
    " delta +0.009 CI spans zero — cell-specific gains, no uniform lift. KILL"
    "-A adjacency concern resolved with data (12/12 rejections). Folded into "
    "the v4 body; re-parked at awaiting_promotion."
)

_NOTE_841_RUN_V2 = (
    "followup_label: gru-source-only; source: proposer-9b-cheap; round: 2; ou"
    "tcome: registered STOP-READ 3 (LOSES): source-only GRU 0/27 vs ridge sta"
    "ge-0, 6/68 vs ridge 12/68 transport cells, aggregate -0.069 CI excluding"
    " zero; prefix-GRU contrast descriptive (CI spans zero) — consistent with"
    " prefix information, not recurrence, carrying the parent's above-ceiling"
    " read. Shared-parametrization scope caveat carried. Folded into the v4 b"
    "ody; re-parked at awaiting_promotion."
)

# The corpus's ONLY version-stamp-led run note (VERBATIM #1092 run marker,
# 2026-07-11T00:06:50Z): a `; `-joined single-line run note whose FIRST
# segment leads with the marker-version stamp `v1. ` before the
# `followup_label:` field — field-led modulo the decorative stamp (unlike
# the prose-led #1090 fu1 note above, which carries no fields at all).
_NOTE_1092_RUN_V1 = (
    "v1. followup_label: cross-corpus-probe-transfer; source: proposer-9b-chea"
    "p; round: 1; est_gpu_hours: 8 (planned) / ~0 realized (all-VM analysis on"
    " existing artifacts, no GPU provisioned); outcome: PARTIAL — downgrade_pr"
    "econdition_met=false. Cross-corpus probe transfer of the #1092 UltraChat-"
    "fit context-map ridge probes: dirA (UltraChat probes -> LMSYS/pass_a targ"
    "ets) blocked from a DOWNGRADE read by the LMSYS alignment-gate ceiling fa"
    "ilure (part-4 reliability ceiling below the pre-registered floor), dirB ("
    "realistic-corpus probes -> UltraChat targets) hallucination transfer Δr ="
    " -0.128 (negative; sycophancy positive), pass_a arm transfers. Folded int"
    "o the clean-result body (new ### result + Takeaways bullet + Methodology "
    "extension + footer Reused: provenance @ run-recorded SHAs); clean-result-"
    "critic re-gate rounds 1-4, final PASS (Claude-only per codex-quota no-sho"
    "w fallback). Artifacts: eval_results/issue_1092/cross-corpus-probe-transf"
    "er/ + figures (merged to main eaef62f632); HF issue1092_realistic_crossin"
    "g/cross_corpus_transfer (13 files, verified). Methodology doc EXTEND expo"
    "rted (main 4644e8aad2, gist refreshed). Cheap-band count: 1/2."
)


def test_label_parse_semicolon_inline_scope_1090():
    # VERBATIM #1090 scope v1 (2026-07-07T07:19:09Z) + v2 (11:05:01Z): the
    # canonical fully-`; `-joined single-line scope form. Pre-#1111 every
    # mid-line field parsed None → permanent `unlabeled-<ts>` pseudo-label
    # ghost groups.
    for note, label, est in (
        (_NOTE_1090_SCOPE_V1, "fu1-margin-qwen", "~4"),
        (_NOTE_1090_SCOPE_V2, "fu2-dose-extension", "4"),
    ):
        assert parse_followup_note_field(note, "followup_label") == label
        # No trailing `;` survives (the split consumes the separator).
        assert parse_followup_note_field(note, "source") == "proposer-9b-cheap"
        assert parse_followup_note_field(note, "question_relation") == "same"
        # `~4` is deliberately kept raw — C1's fail-safe parks unparseable
        # float estimates (plan §7); the parser returns the first token.
        assert parse_followup_note_field(note, "est_gpu_hours") == est


def test_label_parse_semicolon_inline_run_841():
    # VERBATIM #841 run markers: `; `-joined single-line run notes. The
    # line-initial `followup_label:` matched even pre-#1111, but the value
    # kept its trailing `;` ("scaling-capture;") and closed nothing.
    assert parse_followup_note_field(_NOTE_841_RUN_V1, "followup_label") == "scaling-capture"
    assert parse_followup_note_field(_NOTE_841_RUN_V2, "followup_label") == "gru-source-only"
    for note in (_NOTE_841_RUN_V1, _NOTE_841_RUN_V2):
        assert parse_followup_note_field(note, "source") == "proposer-9b-cheap"
    # The #841 scopes parse their mid-line labels too.
    assert parse_followup_note_field(_NOTE_841_SCOPE_V1, "followup_label") == "scaling-capture"
    assert parse_followup_note_field(_NOTE_841_SCOPE_V2, "followup_label") == "gru-source-only"
    # EOL-trailing semicolon on a line-initial field (no whitespace after the
    # `;`, so no split fires): the rstrip(",;") leg strips it.
    assert parse_followup_note_field("followup_label: foo;", "followup_label") == "foo"


def test_label_parse_version_stamp_led_run_1092():
    # VERBATIM #1092 run marker (2026-07-11T00:06:50Z): the first segment
    # leads with the marker-version stamp `v1. `; the stamp strips as
    # decoration and the explicit fields parse — pre-fix the label parsed
    # None, leaving cross-corpus-probe-transfer a dispatchable ghost-unrun
    # the Step 0 dispatcher could re-dispatch (the #658 ghost-label class).
    assert (
        parse_followup_note_field(_NOTE_1092_RUN_V1, "followup_label")
        == "cross-corpus-probe-transfer"
    )
    assert parse_followup_note_field(_NOTE_1092_RUN_V1, "source") == "proposer-9b-cheap"
    assert parse_followup_note_field(_NOTE_1092_RUN_V1, "round") == "1"
    assert parse_followup_note_field(_NOTE_1092_RUN_V1, "outcome") == "PARTIAL"


def test_version_stamp_anchoring_negatives():
    # (i) The prose-led #1090 fu1 note stays None — the stamp tolerance is
    # decoration-stripping on a field-led segment, NOT label inference
    # (field-only parsing per #1111, re-asserted next to the new tolerance).
    assert parse_followup_note_field(_NOTE_1090_RUN_FU1_PROSE, "followup_label") is None
    # (ii) Mid-segment mention: the stamp strip is anchored at segment start
    # and never re-anchors the field search mid-text.
    assert (
        parse_followup_note_field("the v1. followup_label: was missing", "followup_label") is None
    )
    # (iii) No whitespace after the dot → not a stamp, stays prose.
    assert parse_followup_note_field("v1.followup_label: x", "followup_label") is None
    # (iv) A field VALUE beginning with a stamp is untouched (the core starts
    # with `plan:`, so the stamp regex never fires).
    assert parse_followup_note_field("plan: v1. adjust", "plan") == "v1."
    # (v) Composes with the bullet strip (bullet stripped first, then the
    # stamp) and with the `; `-split.
    assert parse_followup_note_field("- v2. followup_label: y; source: z", "followup_label") == "y"
    assert parse_followup_note_field("- v2. followup_label: y; source: z", "source") == "z"


def test_semicolon_split_anchoring_negatives():
    # (i) paren-wrapped mid-line mention (the #685 shape): the segment starts
    # with "(", not the field core → None (segments are anchored exactly like
    # line-cores, never searched mid-text).
    assert (
        parse_followup_note_field(
            "Follow-up scope (source: user-chat) — sharpen the projection result.",
            "source",
        )
        is None
    )
    # (ii) `;` with NO trailing whitespace never splits (in-token semicolons —
    # URLs, code, `a;b=1` — stay unsplit).
    assert parse_followup_note_field("word;followup_label: x", "followup_label") is None
    # (iii) mid-line space-separated mention still parses None.
    assert parse_followup_note_field("the followup_label: was missing", "followup_label") is None
    # (iv) first-hit-wins is lines-outer / segments-inner: a first-line
    # mid-segment field beats a later line's line-initial field (deliberate,
    # documented precedence; no corpus note carries both forms with
    # different values).
    assert (
        parse_followup_note_field(
            "source: x; followup_label: a\nfollowup_label: b", "followup_label"
        )
        == "a"
    )
    # (v) `field:;` — empty value before the separator. Pre-#1111 this parsed
    # the literal `;`; rstrip(",;") maps it to empty → None (the third replay
    # change class; corpus count 0). The empty-value match still STOPS the
    # scan (`return value or None`) — the pre-existing line-level shadowing
    # semantics, now extended to segments (documented behavior).
    assert parse_followup_note_field("followup_label:;", "followup_label") is None
    assert parse_followup_note_field("followup_label:; source: u", "followup_label") is None
    assert parse_followup_note_field("followup_label:; source: u", "source") == "u"
    assert (
        parse_followup_note_field("followup_label:;\nfollowup_label: real", "followup_label")
        is None
    )
    # (vi) empty segments (`a;  ;b` — the second `;` is not followed by
    # whitespace, so the tail stays `;b`): no crash, no match.
    assert parse_followup_note_field("a;  ;b", "followup_label") is None
    # (vii) combined dash-bullet + semicolon-inline positive.
    assert parse_followup_note_field("- followup_label: x; source: y", "followup_label") == "x"
    assert parse_followup_note_field("- followup_label: x; source: y", "source") == "y"


# The #825 v6 run-marker note VERBATIM (2026-07-08T00:50:12Z): literal
# backslash-n two-char escapes between fields, zero real newlines — the
# escaped-newline shape a shell --note "...\n..." string produces when
# passed uninterpreted. v7 (00:51:06Z) is identical modulo ts (#1120).
_NOTE_825_RUN_V6 = (
    r"followup_label: onpolicy-separator-control\nsource: user-chat\nround: 7"
    r"\noutcome: complete — PARTIAL PROVENANCE EFFECT both substrates "
    r"(D_base 0.590 MLP-carried / D_inst 0.428; R4 fractions -4.30/0.166 < 0.5; "
    r"instruct position-residual not excluded, D→≈0.30 scenario qualified); "
    r"body re-folded + re-parked at awaiting_promotion"
)


def test_label_parse_literal_backslash_n_escapes_825():
    # Field-led notes whose separators arrived as literal \n escapes (one
    # physical line) parse all four fields (#1120; pre-fix the label parsed
    # as the garbage token 'onpolicy-separator-control\nsource:').
    assert (
        parse_followup_note_field(_NOTE_825_RUN_V6, "followup_label")
        == "onpolicy-separator-control"
    )
    assert parse_followup_note_field(_NOTE_825_RUN_V6, "source") == "user-chat"
    assert parse_followup_note_field(_NOTE_825_RUN_V6, "round") == "7"
    assert parse_followup_note_field(_NOTE_825_RUN_V6, "outcome") == "complete"
    # Literal \r\n escapes normalize too — no stray backslash-r on the value.
    assert parse_followup_note_field(r"followup_label: x\r\nsource: y", "followup_label") == "x"
    assert parse_followup_note_field(r"followup_label: x\r\nsource: y", "source") == "y"
    # Normalization introduces no new positives on a field-less note.
    assert parse_followup_note_field(r"no label here\nnothing", "followup_label") is None
    # Double-escape protective pin (#1120 review note): a 3-char literal
    # `\\n` TAIL-matches the 2-char escape on a single-line note, so the
    # note still splits at that tail and the preceding backslash rides the
    # value token — pinned so any future change here is deliberate.
    assert parse_followup_note_field(r"followup_label: ok\\nsource: y", "source") == "y"
    assert parse_followup_note_field(r"followup_label: ok\\nsource: y", "followup_label") == "ok\\"


def test_literal_backslash_n_content_preserved_when_real_newlines_present():
    # The predicate's protective side (#1120, predicate (b)): a note that
    # ALREADY has real newlines keeps literal \n escapes as CONTENT — a
    # quoted regex/code value never splits mid-value.
    note = "followup_label: real-label\npattern: a\\nb rest"
    assert parse_followup_note_field(note, "followup_label") == "real-label"
    assert parse_followup_note_field(note, "pattern") == "a\\nb"


def test_1090_shape_labels_group_and_close_only_on_repost():
    # End-to-end #1090 replay: the two `; `-joined scopes group + label
    # correctly post-fix; the VERBATIM prose-led fu1 run marker (no fields at
    # all) closes NOTHING — pins the field-only decision (no leading
    # kebab-token label inference; plan §4.2). Closure comes only from the
    # #1111 corrective re-posts (plan §4.5), which land at version 2/3 on the
    # live task (`post-marker` auto-increments per kind past the existing
    # v1 — never assert the re-posts are version 1).
    events = [
        _scope("2026-07-07T07:19:09Z", _NOTE_1090_SCOPE_V1, version=1),
        _run("2026-07-07T09:54:27Z", _NOTE_1090_RUN_FU1_PROSE, version=1),
        _scope("2026-07-07T11:05:01Z", _NOTE_1090_SCOPE_V2, version=2),
    ]
    unrun = {g["followup_label"]: g for g in unrun_followup_labels(events)}
    assert set(unrun) == {"fu1-margin-qwen", "fu2-dose-extension"}
    for group in unrun.values():
        assert group["dispatchable"] is True
        assert group["source"] == "proposer-9b-cheap"
    # The corrective re-posts carry line-initial fields (they parse under the
    # OLD parser too, so cheap-band cap counting flips even pre-merge): both
    # groups close.
    repost_fu1 = _run(
        "2026-07-07T18:00:00Z",
        "followup_label: fu1-margin-qwen\nsource: proposer-9b-cheap\nround: 1\n"
        "outcome: COMPLETE — corrective re-post (#1111). The original run marker "
        "(2026-07-07T09:54:27Z) led with prose and carried no followup_label:/source: "
        "fields, so it closed nothing and the cheap-band cap undercounted.",
        version=2,
    )
    repost_fu2 = _run(
        "2026-07-07T18:00:01Z",
        "followup_label: fu2-dose-extension\nsource: proposer-9b-cheap\nround: 2\n"
        "outcome: COMPLETE — corrective re-post (#1111). The round completed but no "
        "run marker was ever posted; the semicolon-joined scope note additionally "
        "defeated label grouping until #1111.",
        version=3,
    )
    # Cap accounting: the re-posts' `source` parses proposer-9b-cheap and the
    # `outcome` is NOT retroactive-close-styled (retro-close markers are
    # cap-exempt; #1090's cheap cap was genuinely consumed — C2 must read 2/2).
    for repost in (repost_fu1, repost_fu2):
        assert parse_followup_note_field(repost["note"], "source") == "proposer-9b-cheap"
        outcome = parse_followup_note_field(repost["note"], "outcome")
        assert outcome is not None and not outcome.startswith("retroactive-close")
    assert unrun_followup_labels([*events, repost_fu1, repost_fu2]) == []


def test_841_shape_self_heals():
    # VERBATIM #841 events: 2 `; `-joined scopes + 2 `; `-joined run markers.
    # Post-fix the scopes parse their labels AND the run markers parse the
    # SAME labels (the trailing `;` consumed by the split) → full self-heal,
    # no orchestrator action needed on #841.
    events = [
        _scope("2026-07-02T23:38:10Z", _NOTE_841_SCOPE_V1, version=1),
        _scope("2026-07-03T06:19:11Z", _NOTE_841_SCOPE_V2, version=2),
        _run("2026-07-04T08:42:08Z", _NOTE_841_RUN_V1, version=1),
        _run("2026-07-04T08:42:11Z", _NOTE_841_RUN_V2, version=2),
    ]
    assert unrun_followup_labels(events) == []


# ─── 5. unlabeled corrections vs distinct unlabeled follow-ups ───────────────


def test_unlabeled_correction_inherits_previous_label():
    # The REAL #658 v2 shape: an unlabeled note carrying the literal word
    # CORRECTION attributes to the immediately-preceding label; a matching
    # run marker then closes the whole group.
    events = [
        _scope(
            "2026-06-25T08:42:56Z",
            "Genre-generalization follow-up.\n\n"
            "- followup_label: genre-generalization-ultrachat\n- source: user-chat",
            version=1,
        ),
        _scope(
            "2026-06-25T09:07:55Z",
            "CORRECTION to the earlier epm:followup-scope "
            "(genre-generalization-ultrachat): the gating is now\n"
            'UNCONDITIONAL, superseding the prior "auto_run: NO" line.',
            version=2,
        ),
    ]
    groups = followup_label_groups(events)
    assert len(groups) == 1
    group = groups[0]
    assert group["followup_label"] == "genre-generalization-ultrachat"
    assert group["n_entries"] == 2
    # The correction's content IS the label's authoritative entry.
    assert group["authoritative"]["version"] == 2
    assert group["label_parse"] == "inherited-from-previous"
    assert group["dispatchable"] is True
    closed = [
        *events,
        _run(
            "2026-06-28T05:12:17Z",
            "followup_label: genre-generalization-ultrachat\nsource: user-chat\nround: 3",
        ),
    ]
    assert unrun_followup_labels(closed) == []


def test_unlabeled_noncorrection_scope_is_distinct_group():
    # The REAL #685 shape: labeled v1 + a label-less v2 with NO correction
    # signal (a distinct user-chat follow-up). v2 must NOT merge into v1's
    # label — it becomes its own pseudo-ts group, surfaced but never
    # dispatched (Alt-Claude MF2 / Alt-Codex MF2).
    events = [
        _scope(
            "2026-06-27T20:24:25Z",
            "<!-- epm:followup-scope v1 -->\n"
            "**followup_label:** full-judge-coverage-and-syco-opinion\n"
            "**source:** user-chat\n**question_relation:** same",
            version=1,
        ),
        _scope(
            "2026-06-28T09:27:49Z",
            "Follow-up scope (source: user-chat) — sharpen the Δ-vs-behavior-vector "
            "projection result.\n\n**Question (same as parent):** does each behavior "
            "shift track its own direction?",
            version=2,
        ),
    ]
    groups = followup_label_groups(events)
    assert [g["followup_label"] for g in groups] == [
        "full-judge-coverage-and-syco-opinion",
        "unlabeled-2026-06-28T09:27:49Z",
    ]
    labeled, pseudo = groups
    assert labeled["authoritative"]["version"] == 1  # v2 never merged in
    assert labeled["user_initiated"] is True
    assert pseudo["label_parse"] == "pseudo-ts"
    assert pseudo["dispatchable"] is False


# ─── 6. leading unlabeled scope → non-dispatchable pseudo-label ──────────────


def test_leading_unlabeled_scope_pseudo_label_nondispatchable():
    sole = _scope("2026-06-28T09:27:49Z", "malformed scope note with no fields")
    unrun = unrun_followup_labels([sole])
    assert len(unrun) == 1
    group = unrun[0]
    assert group["followup_label"] == "unlabeled-2026-06-28T09:27:49Z"
    assert group["label_parse"] == "pseudo-ts"
    assert group["dispatchable"] is False
    # A run marker carrying the pseudo-label VERBATIM (the retro-close path)
    # still closes it.
    closed = [
        sole,
        _run(
            "2026-06-29T00:00:00Z",
            "followup_label: unlabeled-2026-06-28T09:27:49Z source: unknown round: 1 "
            "outcome: retroactive-close — repaired",
        ),
    ]
    assert unrun_followup_labels(closed) == []


def test_pseudo_founded_group_stays_nondispatchable_after_inherited_correction():
    # r2 Major 2 (persisted concern `pseudo-label-inherit-dispatchable`): an
    # unlabeled CORRECTION following a pseudo-founded group inherits into it
    # (raising the group's authoritative entry) but must NOT flip it
    # dispatchable — the group's label is still the malformed
    # `unlabeled-<ts>` (kebab-slug contract violation), a repair item until
    # re-posted with a proper `followup_label`. Dispatchability is
    # FOUNDING-based, not last-entry-parse-mode-based.
    events = [
        _scope("2026-06-28T09:00:00Z", "malformed scope note with no fields", version=1),
        _scope(
            "2026-06-28T10:00:00Z",
            "CORRECTION to the earlier epm:followup-scope: still no label line.",
            version=2,
        ),
    ]
    (group,) = followup_label_groups(events)
    assert group["followup_label"] == "unlabeled-2026-06-28T09:00:00Z"
    assert group["n_entries"] == 2
    # The correction IS the authoritative entry (inherit semantics intact)…
    assert group["authoritative"]["version"] == 2
    assert group["label_parse"] == "inherited-from-previous"
    # …but the pseudo-founded group stays a non-dispatchable repair item.
    assert group["dispatchable"] is False
    (unrun_group,) = unrun_followup_labels(events)
    assert unrun_group["dispatchable"] is False


# ─── 7. executing-label resolution: breadcrumb first, head fallback ──────────


def test_executing_label_breadcrumb_first_head_fallback():
    scope_a = _scope(
        "2026-06-10T00:00:00Z",
        "followup_label: label-a\nsource: user-chat",
        version=1,
    )
    scope_b = _scope(
        "2026-06-10T01:00:00Z",
        "followup_label: label-b\nsource: proposer-9b-cheap",
        version=2,
    )
    scope_c = _scope(
        "2026-06-10T02:00:00Z",
        "followup_label: label-c\nsource: proposer-9b-cheap",
        version=3,
    )
    run_c = _run(
        "2026-06-10T03:00:00Z",
        "followup_label: label-c source: proposer-9b-cheap round: 1 outcome: done",
    )
    crumb_b_fresh = _ev(
        "epm:progress",
        "2026-06-10T04:00:00Z",
        "stage-dispatch stage=followup-implementing round=1 "
        "subagent=experiment-implementer worktree=/tmp/wt label=label-b",
    )
    # (1) labeled breadcrumb NEWER than the newest run marker → B's group,
    # even though user-chat label-a heads the queue.
    group = executing_followup_label([scope_a, scope_b, scope_c, run_c, crumb_b_fresh])
    assert group is not None and group["followup_label"] == "label-b"
    # (2) breadcrumb OLDER than the newest run marker → dispatchable head (A).
    crumb_b_stale = dict(crumb_b_fresh, ts="2026-06-10T02:30:00Z")
    group = executing_followup_label([scope_a, scope_b, scope_c, run_c, crumb_b_stale])
    assert group is not None and group["followup_label"] == "label-a"
    # (2b) no breadcrumb at all → dispatchable head.
    group = executing_followup_label([scope_a, scope_b])
    assert group is not None and group["followup_label"] == "label-a"
    # (3) no dispatchable unrun label → None (pseudo groups never resolve).
    pseudo_only = _scope("2026-06-11T00:00:00Z", "malformed note, no fields", version=4)
    assert executing_followup_label([pseudo_only]) is None
    assert executing_followup_label([]) is None


# ─── 8. label-keyed re-arm semantics (re-posts do not re-open) ───────────────


def test_same_label_repost_after_run_stays_closed():
    scope_a = _scope(
        "2026-06-10T00:00:00Z",
        "followup_label: label-a\nsource: user-chat",
        version=1,
    )
    run_a = _run(
        "2026-06-10T05:00:00Z",
        "followup_label: label-a\nsource: user-chat\nround: 1",
    )
    # Labeled RE-POST after the run marker: same label → still closed (a
    # re-run needs a NEW label — pins the existing label-keyed semantics).
    repost = _scope(
        "2026-06-10T06:00:00Z",
        "followup_label: label-a\nsource: user-chat\nRE-POST of the earlier scope",
        version=2,
    )
    assert unrun_followup_labels([scope_a, run_a, repost]) == []
    # An UNLABELED re-post carrying the correction signal attributes to A
    # (inherit leg) — closure preserved, never a fresh pseudo group.
    unlabeled_repost = _scope(
        "2026-06-10T06:00:00Z",
        "RE-POST of the earlier scope for label-a with a sharpened spec.",
        version=2,
    )
    assert unrun_followup_labels([scope_a, run_a, unlabeled_repost]) == []


def test_source_falls_back_across_group_entries():
    # A label whose LATEST correction note omits `source:` (the #658-v2
    # shape) must not demote a user-chat round to "unknown" / lose queue
    # priority — group source = FIRST parseable source in scan order.
    events = [
        _scope(
            "2026-06-25T08:42:56Z",
            "- followup_label: genre-generalization-ultrachat\n- source: user-chat",
            version=1,
        ),
        _scope(
            "2026-06-25T09:07:55Z",
            "CORRECTION to the earlier epm:followup-scope "
            "(genre-generalization-ultrachat): gating now unconditional.",
            version=2,
        ),
    ]
    (group,) = followup_label_groups(events)
    assert group["source"] == "user-chat"
    assert group["user_initiated"] is True


# ─── 8b. the #480 duplicate-version anomaly: chronological (ts, version) scan ─


def test_480_duplicate_version_rows_scan_chronologically():
    # The REAL #480 anomaly (plan §12 assumption 4, corrected in Phase 2):
    # per-kind version monotonicity is VIOLATED in the wild — two scope rows
    # share `version: 1` with a v2 chronologically BETWEEN them. The scan key
    # is (ts, version) — chronological with version tiebreak. A (version, ts)
    # mutant scans the late duplicate-v1 row BEFORE the between v2 row, which
    # (a) reorders the first-armed group order and (b) mis-attributes the
    # trailing unlabeled CORRECTION to the wrong previous label.
    events = [
        _scope(
            "2026-06-11T10:00:00Z",
            "followup_label: sycophancy-dose-response\nsource: user-chat",
            version=1,
        ),
        _scope(
            "2026-06-11T11:00:00Z",
            "followup_label: between-label\nsource: proposer-9b-cheap",
            version=2,
        ),
        _scope(
            "2026-06-11T12:00:00Z",
            "followup_label: late-duplicate-v1\nsource: proposer-9b-cheap",
            version=1,
        ),
        _scope(
            "2026-06-11T13:00:00Z",
            "CORRECTION: sharpen the previous scope's eval spec.",
            version=3,
        ),
    ]
    groups = followup_label_groups(events)
    # First-armed group order is CHRONOLOGICAL despite the duplicate version
    # numbers (a version-primary mutant yields [..., late-duplicate-v1,
    # between-label]).
    assert [g["followup_label"] for g in groups] == [
        "sycophancy-dose-response",
        "between-label",
        "late-duplicate-v1",
    ]
    # The unlabeled CORRECTION attributes to the CHRONOLOGICALLY previous
    # label (late-duplicate-v1); a version-primary mutant would scan
    # between-label last and mis-attribute the correction there.
    late = groups[2]
    assert late["n_entries"] == 2
    assert late["authoritative"]["version"] == 3
    assert late["dispatchable"] is True
    assert groups[1]["n_entries"] == 1
    # Queue mechanics unaffected: user-initiated first, then oldest armed ts.
    assert [g["followup_label"] for g in unrun_followup_labels(events)] == [
        "sycophancy-dose-response",
        "between-label",
        "late-duplicate-v1",
    ]


# ─── 8c. retro-close evidence is mechanical + exact-label only ───────────────


def test_retro_close_evidence_exact_label_only():
    label = "persona-vectors-style-rb"
    # (i) a 9a-quater extends=<label> record → evidence.
    ev_methodology = _ev(
        "epm:methodology-doc-generated",
        "2026-06-29T12:00:00Z",
        "EXTEND pass complete: extends=persona-vectors-style-rb gist refreshed",
    )
    assert followup_retro_close_evidence([ev_methodology], label) is not None
    # (ii) an epm:free-analysis-followup-run with followup_ref EXACTLY equal →
    # evidence; a PREFIX match NEVER closes.
    ev_free_exact = _ev(
        "epm:free-analysis-followup-run",
        "2026-06-29T13:00:00Z",
        "followup_ref: persona-vectors-style-rb\noutcome: fit complete",
    )
    ev_free_prefix = _ev(
        "epm:free-analysis-followup-run",
        "2026-06-29T13:00:00Z",
        "followup_ref: persona-vectors-style-rb-9a-ter-fit\noutcome: fit complete",
    )
    assert followup_retro_close_evidence([ev_free_exact], label) is not None
    assert followup_retro_close_evidence([ev_free_prefix], label) is None
    # (iii) a status note with the exact parenthesized round token + a
    # round-completion word on the same line → evidence.
    ev_status = _ev(
        "epm:status-changed",
        "2026-06-29T14:00:00Z",
        "round-4 (persona-vectors-style-rb) clean-result-critic PASS",
    )
    assert followup_retro_close_evidence([ev_status], label) is not None
    # Parenthesized token WITHOUT a completion word → None.
    ev_status_no_word = _ev(
        "epm:status-changed",
        "2026-06-29T14:00:00Z",
        "round-4 (persona-vectors-style-rb) planner amendment dispatched",
    )
    assert followup_retro_close_evidence([ev_status_no_word], label) is None
    # (iv) NEGATIVE (Alt-Codex MF1): the label appearing in proposal/body
    # prose — an epm:follow-ups proposal naming it, or a bare prose mention —
    # NEVER closes.
    ev_proposal = _ev(
        "epm:follow-ups",
        "2026-06-29T15:00:00Z",
        "Proposal 1: persona-vectors-style-rb — extract r_B per the paper; PASS criteria attached",
    )
    ev_prose = _ev(
        "epm:progress",
        "2026-06-29T15:30:00Z",
        "considering persona-vectors-style-rb for the next round; PASS pending",
    )
    assert followup_retro_close_evidence([ev_proposal], label) is None
    assert followup_retro_close_evidence([ev_prose], label) is None
    # No events at all → None.
    assert followup_retro_close_evidence([], label) is None
    # Multiple exact classes agreeing on the SAME label are CORROBORATION,
    # not ambiguity — the canonical #658 ghost label carries both a
    # 9a-quater extends= record AND a status-PASS round note, and must
    # still close (first matching class wins, class order 1 → 2 → 3).
    evidence = followup_retro_close_evidence([ev_methodology, ev_status], label)
    assert evidence is not None
    assert "methodology-doc-generated" in evidence


def test_retro_close_evidence_825_queued_label_park_notes():
    label = "role-map-comparison"
    # The founding #825 false positive (2026-07-04T04:21:23Z), VERBATIM:
    # completion words (re-park / awaiting_promotion) describe the
    # real-user-turn-null round; the label is named only as QUEUED.
    ev_step = _ev(
        "epm:step-completed",
        "2026-07-04T04:21:23Z",
        "<!-- epm:step-completed v1 -->\n## Step Completed\n\n"
        "step: 9a-bis\nat: 031492f2\ntimestamp: 2026-07-04T04:21:23+00:00\n"
        "next_expected_step: 9a-quater\nexit_kind: parked\n"
        "notes: real-user-turn-null round re-parked at awaiting_promotion; "
        "1 unrun user-chat label (role-map-comparison) queued — next entry "
        "dispatches it; cron kept armed\n"
        "<!-- /epm:step-completed -->",
    )
    assert followup_retro_close_evidence([ev_step], label) is None
    # Sibling status-changed note (04:20:49Z), VERBATIM: the token there is
    # `(role-map-comparison,` — not the exact `(role-map-comparison)` — so it
    # does not match today either; pinned so a future token relaxation
    # cannot silently reintroduce the false positive (the clause/veto logic
    # would reject it anyway).
    ev_status = _ev(
        "epm:status-changed",
        "2026-07-04T04:20:49Z",
        "real-user-turn-null round complete; clean-result-critic PASS (r2, "
        "ensemble); re-parking for user promotion. NOTE: 1 unrun user-chat "
        "followup label queued (role-map-comparison, armed 2026-07-03T06:16Z, "
        "now unblocked — all three provenances landed); next /issue 825 entry "
        "dispatches it.",
    )
    assert followup_retro_close_evidence([ev_status], label) is None


def test_retro_close_evidence_595_deferred_scope_recap():
    # The second live false positive (#595, 2026-06-14T06:20:31Z), VERBATIM:
    # the token's clause ("routes next /issue 595 invocation into same-issue
    # loop") carries NO queue-context vocabulary at all — only the #961
    # clause-binding leg catches it; an exclusion regex alone would not.
    label = "h2-full-probes-multiseed"
    ev_step = _ev(
        "epm:step-completed",
        "2026-06-14T06:20:31Z",
        "<!-- epm:step-completed v1 -->\n## Step Completed\n\n"
        "step: 9a-bis\nat: 44eedb0d\ntimestamp: 2026-06-14T06:20:31+00:00\n"
        "next_expected_step: 9a-quater\nexit_kind: parked\n"
        "notes: awaiting clean-result promotion. Full /issue 595 lifecycle "
        "complete: planning → 3 plan revs (v3 corrected squared-gauge) → 3 "
        "implementer rounds (round 3 vendored issue503/) → fullrun-v3 on "
        "pod-595 (after 3 failed GCP auto-lane attempts + 1 Anthropic 429 "
        "mid-run) → upload-verify PASS r2 → analyzer interpretation loop "
        "(2 rounds, reconciler PASS) → clean-result-critic loop (2 rounds, "
        "reconciler PASS) → methodology doc landed + secret gist + body "
        "link-append → awaiting_promotion. Follow-ups: child #640 filed "
        "(postfix carrier, substantially-different); epm:followup-scope v1 "
        "for proposal #2 (h2-full-probes-multiseed) routes next /issue 595 "
        "invocation into same-issue loop; proposal #1 (free-analysis "
        "leverage check) surfaced in epm:follow-ups v1 for user "
        "post-promotion pick. Merge to main BLOCKED — epm:merge-failed v1 "
        "requires manual rebase resolution (Guard 3 + new-shared-src/-infra "
        "guard).\n"
        "<!-- /epm:step-completed -->",
    )
    assert followup_retro_close_evidence([ev_step], label) is None


def test_retro_close_evidence_clause_binding_shapes():
    """Synthetic clause shapes pinning the #961 two-gate class-3 logic."""
    label = "some-label"
    # The dominant park-shape true positive (mirrors #505): completion word
    # in the token's clause via the complete/COMPLETE supplement, PASS later.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "notes: round-2 followup (some-label) complete; clean-result re-gated "
        "PASS; re-parked at awaiting_promotion",
    )
    assert followup_retro_close_evidence([ev], label) is not None
    # The COMPLETE variant (mirrors #545).
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "/issue same-issue follow-up (some-label) loop COMPLETE. Both critic ensembles PASS",
    )
    assert followup_retro_close_evidence([ev], label) is not None
    # Cross-clause split: completion words describe ANOTHER round's clause;
    # the token's clause needs no veto word to be rejected.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "real-round re-parked at awaiting_promotion; label (some-label) held for next entry",
    )
    assert followup_retro_close_evidence([ev], label) is None
    # Same-clause veto: clause binding alone would wrongly match
    # (awaiting_promotion is in the token's clause) — the queue veto rejects.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "unrun label (some-label) queued for the awaiting_promotion park",
    )
    assert followup_retro_close_evidence([ev], label) is None
    # Veto is per-clause: a queued mention of ANOTHER label elsewhere on the
    # line does not block a legitimate close of this one.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "round-4 (some-label) clean-result-critic PASS; next label (other-label) queued unrun",
    )
    assert followup_retro_close_evidence([ev], "some-label") is not None
    assert followup_retro_close_evidence([ev], "other-label") is None
    # Narrowing-only guard: no gate-1 word on the line ⇒ the `complete`
    # supplement alone can never CREATE evidence the old predicate rejected.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "(some-label) round complete, nothing more",
    )
    assert followup_retro_close_evidence([ev], label) is None
    # `incomplete` lookbehind + cross-clause: neither leg matches.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "(some-label) round incomplete; clean-result-critic PASS pending",
    )
    assert followup_retro_close_evidence([ev], label) is None


# ─── 14. corpus replay over the real tasks/ tree ─────────────────────────────


def _tasks_root() -> Path | None:
    root = Path(__file__).resolve().parents[1] / "tasks"
    return root if root.is_dir() else None


def _load_events(task_dir: Path) -> list[dict]:
    events: list[dict] = []
    path = task_dir / "events.jsonl"
    if not path.is_file():
        return events
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            events.append(row)
    return events


def _corpus_events_by_task() -> dict[int, list[dict]]:
    root = _tasks_root()
    assert root is not None
    by_task: dict[int, list[dict]] = {}
    for task_dir in sorted(root.glob("*/*")):
        if not task_dir.is_dir() or not task_dir.name.isdigit():
            continue
        events = _load_events(task_dir)
        if events:
            by_task.setdefault(int(task_dir.name), []).extend(events)
    return by_task


def _run_labels(events: list[dict]) -> set[str]:
    return {
        parse_followup_note_field(e.get("note") or "", "followup_label")
        for e in events
        if e.get("kind") == "epm:same-issue-followup-run"
    } - {None}


# The corpus's single KNOWN-malformed run marker: #1090 fu1's original run
# note (2026-07-07T09:54:27Z) led with prose ("fu1-margin-qwen round
# COMPLETE (...)") and carried NO followup_label:/source: fields at all —
# deliberately unparseable under field-only parsing (task #1111 kept the
# parser field-only; leading-kebab-token label inference was rejected). The
# round is closed by #1111's corrective re-post on #1090, not by parsing.
# Documented residual (WARN-only, monotone-up): verify_task_body's
# `_followup_run_marker_rounds` counts this prose-led marker as an unlabeled
# round, so #1090's round count reads one high (3 where 2 completed). Any
# NEW unparseable run marker still fails the corpus-replay test below.
# (#1092's `v1. `-stamp-led run note is deliberately NOT allowlisted — its
# fields are explicit, and the parser strips the leading version stamp as
# decoration; see `parse_followup_note_field`.)
KNOWN_MALFORMED_RUN_MARKERS = {(1090, "2026-07-07T09:54:27Z")}


def test_corpus_replay_all_historical_markers():
    # Alt-Claude MF1 corpus-replay validation: every HISTORICAL
    # epm:same-issue-followup-run marker in the checkout must parse a
    # followup_label (covers the 15 `=`-form markers + all colon forms —
    # minus the documented KNOWN_MALFORMED_RUN_MARKERS allowlist), and
    # the hand-checked #763/#658/#825/#537/#552/#1092 expectations must hold.
    import pytest

    if _tasks_root() is None:
        pytest.skip("tasks/ not present in this checkout (sparse worktree)")
    by_task = _corpus_events_by_task()

    unparseable: list[tuple[int, str]] = []
    n_run = 0
    for task_id, events in by_task.items():
        for ev in events:
            if ev.get("kind") != "epm:same-issue-followup-run":
                continue
            n_run += 1
            if parse_followup_note_field(ev.get("note") or "", "followup_label") is None:
                unparseable.append((task_id, str(ev.get("ts"))))
    assert n_run > 0, "corpus unexpectedly carries no run markers"
    unparseable = [t for t in unparseable if t not in KNOWN_MALFORMED_RUN_MARKERS]
    assert unparseable == [], f"unparseable run-marker labels: {unparseable}"

    # #763 (the driving incident): the queued user-chat label must surface as
    # dispatchable-unrun for as long as no run marker closes it (events are
    # append-only, so once closed the guard makes this leg inert — a
    # LEGITIMATE later close of the round this fix un-strands).
    if 763 in by_task:
        events = by_task[763]
        unrun = {g["followup_label"]: g for g in unrun_followup_labels(events)}
        assert "deception-rubric-reanchor" not in unrun
        if "neutral-contrast-and-cofit" not in _run_labels(events):
            group = unrun.get("neutral-contrast-and-cofit")
            assert group is not None, "the #763 queued label must be visible as unrun"
            assert group["dispatchable"] is True
            assert group["user_initiated"] is True

    # #658 (correction chain + ghost labels): the v3→v7 chain groups into ONE
    # label; the unlabeled v2 CORRECTION attributes to genre-generalization-
    # ultrachat (2 entries), which is closed by its run marker.
    if 658 in by_task:
        events = by_task[658]
        groups = {g["followup_label"]: g for g in followup_label_groups(events)}
        assert "persona-vectors-style-rb" in groups
        assert groups["persona-vectors-style-rb"]["n_entries"] >= 5  # v3..v7
        assert groups["persona-vectors-style-rb"]["authoritative"]["version"] >= 7
        assert groups["genre-generalization-ultrachat"]["n_entries"] >= 2  # v1 + v2 correction
        unrun = {g["followup_label"] for g in unrun_followup_labels(events)}
        assert "genre-generalization-ultrachat" not in unrun
        if "persona-vectors-style-rb" not in _run_labels(events):
            # A ghost label (round demonstrably ran, no run marker) surfaces
            # as unrun — the Step 0 retro-close disposition rule handles it.
            assert "persona-vectors-style-rb" in unrun

    # #825 (the #1120 escaped-newline incident): run markers v6/v7 must
    # parse the TRUE label (events are append-only, so this pin is stable).
    if 825 in by_task:
        labels_825 = _run_labels(by_task[825])
        assert "onpolicy-separator-control" in labels_825
        # Discriminating half (#1120): pre-fix, v6/v7 parsed the garbage
        # token 'onpolicy-separator-control\nsource:' (literal backslash-n)
        # into the label set; post-fix no parsed label carries a literal \n.
        assert not any("\\n" in lbl for lbl in labels_825)

    # #537 / #552: the `=`-form run markers close their labels.
    for task_id, closed_labels in (
        (
            537,
            [
                "seed2-marker-fact-replication",
                "behavior-conditioned-predictors",
                "predictor-bakeoff-complete",
            ],
        ),
        (
            552,
            [
                "em-arm-mean-resp-reextraction",
                "marker-arm-mean-resp-reextraction",
                "contrastive-2x2-completion",
            ],
        ),
    ):
        if task_id not in by_task:
            continue
        unrun = {g["followup_label"] for g in unrun_followup_labels(by_task[task_id])}
        for label in closed_labels:
            assert label not in unrun, f"#{task_id} {label} must be closed by its run marker"

    # #1092 (the version-stamp-led run note, this fix's driving incident):
    # the 2026-07-11 run marker parses its label and closes the round —
    # pre-fix it parsed None and left the round a dispatchable ghost-unrun
    # (events are append-only, so this pin is stable).
    if 1092 in by_task:
        assert "cross-corpus-probe-transfer" in _run_labels(by_task[1092])
        unrun_1092 = {g["followup_label"] for g in unrun_followup_labels(by_task[1092])}
        assert "cross-corpus-probe-transfer" not in unrun_1092


def test_corpus_replay_retro_close_verdicts():
    # #961 retro-close pins (events are append-only; guards make each leg
    # inert once the label legitimately closes via a run marker):
    import pytest

    if _tasks_root() is None:
        pytest.skip("tasks/ not present in this checkout (sparse worktree)")
    by_task = _corpus_events_by_task()
    for task_id, queued_label in (
        (825, "role-map-comparison"),
        (595, "h2-full-probes-multiseed"),
    ):
        if task_id in by_task and queued_label not in _run_labels(by_task[task_id]):
            assert followup_retro_close_evidence(by_task[task_id], queued_label) is None, (
                f"#{task_id} {queued_label} is queued/unrun — retro-close evidence "
                "for it is the #961 false positive"
            )
    if 658 in by_task:  # the canonical ghost close must SURVIVE the narrowing
        assert followup_retro_close_evidence(by_task[658], "persona-vectors-style-rb") is not None
