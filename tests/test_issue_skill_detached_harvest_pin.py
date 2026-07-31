"""Pin the #1656 detached-phase harvest contract in `/issue` SKILL.md + mirrors.

Incident #1310 (2026-07-16): a detached VM-side long-compute phase survived its
session (the #833 setsid convention protected the RUN) but its finished results
were collectable only by the launching conversation — the harvest step was
session-bound, and an autoharvest was bolted on only after the user asked
"if I close this terminal will it continue running?". #1656 adds the launch-time
harvest contract: a durable out-root (REQUIRED), a fourth stage-dispatch
breadcrumb token `harvest=<abs output path>` (REQUIRED, additive/order-free per
`task_workflow._breadcrumb_fields`; graceful pre-contract fallback like
`label=`), and self-harvest chaining as a SINGLE command unit (PREFERRED —
never a bare `&&` splice into the setsid template, which mis-binds
setsid/nohup/env and silently breaks detachment).

These tests pin, against `.claude/skills/issue/SKILL.md` (+ the two
paths-triggered mirror duty-lists):

1. the "Harvest contract" block exists INSIDE the detached-phases block with
   its load-bearing pieces — the token grammar, the four-field sentence, the
   pre-contract fallback, the durable out-root, and the bare-`&&`-splice
   detachment warning;
2. the Successor / re-entry rule CONSUMES the declared `harvest=` path (and
   runs the harvest) instead of guessing;
3. the Step 9a-ter compute-character detached-launch sentence names the
   harvest contract among the launch duties;
4. both mirror duty-lists (`.claude/rules/code-style.md` +
   `.claude/rules/vectorize-many-cell-fits.md`) carry the `harvest=` token,
   so a future edit cannot silently regress a mirror back to three fields
   (the #957 criterion-1b shape);
5. (#1764) the GCE root-owned workload-log probe prescription: the Successor /
   re-entry rule carries the GCE log-read note (`sudo -n tail` retry on a
   `Permission denied` content read, never misclassified as a dead/frozen
   phase), and the Long-phase heartbeat duty step 2(i) references it.

Prose assertions run on whitespace-NORMALIZED text (the file wraps prose
mid-phrase, so a required phrase can span lines).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
CODE_STYLE_MD = REPO_ROOT / ".claude" / "rules" / "code-style.md"
VECTORIZE_MD = REPO_ROOT / ".claude" / "rules" / "vectorize-many-cell-fits.md"

DETACHED_HEADING = "**Detached VM-side long compute phases"
SUCCESSOR_HEADING = "**Successor / re-entry rule"
GUARD_HEADING = "**Checkable guard rule"
HEARTBEAT_HEADING = "**Long-phase heartbeat duty"
REVIVAL_HEADING = "Revival trigger for the deferred watcher-side option"


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-tolerant match)."""
    return re.sub(r"\s+", " ", text)


def _skill_text() -> str:
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    return SKILL_MD.read_text(encoding="utf-8")


def _region(text: str, start_anchor: str, end_anchor: str) -> str:
    start = text.find(start_anchor)
    assert start != -1, f"anchor {start_anchor!r} not found in SKILL.md"
    end = text.find(end_anchor, start)
    assert end != -1, f"anchor {end_anchor!r} not found after {start_anchor!r}"
    return text[start:end]


def test_harvest_contract_block_present() -> None:
    """The Harvest contract lives inside the detached-phases block, complete."""
    block = _norm(_region(_skill_text(), DETACHED_HEADING, SUCCESSOR_HEADING))
    assert "Harvest contract" in block, "detached block lacks the '**Harvest contract' clause"
    # Token grammar (Edit 1's field list + Edit 2 clause 2 both carry it).
    assert "harvest=<abs" in block, "detached block lacks the harvest=<abs ...> token grammar"
    # The upgraded required-fields sentence (three -> four, unbolded 'four').
    assert "four additional fields" in block, "breadcrumb sentence not upgraded to four fields"
    # Pre-contract fallback (the label=-style graceful-optional convention).
    assert "predating this contract" in block or "fall back" in block, (
        "harvest contract lacks the pre-contract breadcrumb fallback clause"
    )
    # Clause 1: the durable out-root requirement.
    assert "durable" in block, "harvest contract lacks the durable out-root requirement"
    # Clause 3: the bare-&&-splice detachment-break warning (single command unit).
    assert "NEVER splice a bare" in block, (
        "harvest contract lacks the bare-&&-splice detachment warning"
    )


def test_successor_rule_consumes_harvest() -> None:
    """The Successor / re-entry rule reads harvest= and runs the harvest."""
    para = _norm(_region(_skill_text(), SUCCESSOR_HEADING, GUARD_HEADING))
    assert "harvest=" in para, "Successor rule does not consume the harvest= breadcrumb path"
    assert "RUN THE HARVEST" in para, "Successor rule lacks the RUN THE HARVEST instruction"


def test_compute_character_statement_names_harvest() -> None:
    """The 9a-ter compute-character detached-launch sentence names the contract."""
    region = _norm(
        _region(
            _skill_text(),
            "A statement covering a VM-side phase",
            "Routing, auto-continue behavior, and the marker schema are unchanged",
        )
    )
    assert "harvest contract" in region, (
        "9a-ter compute-character detached sentence does not name the harvest contract"
    )


def test_gce_sudo_tail_probe_prescribed() -> None:
    """#1764: GCE root-owned log content reads are retried with sudo -n tail.

    (a) The Successor / re-entry rule carries the GCE log-read note — the
    `sudo -n tail` retry prescription for a `Permission denied` content read,
    plus the never-misclassify clause (an EACCES is a probe artifact, not a
    dead/frozen phase). (b) The Long-phase heartbeat duty step 2(i) evidence
    list references the same rule, so a Permission-denied `tail` is never
    treated as verify-FAIL evidence.
    """
    text = _skill_text()
    successor = _norm(_region(text, SUCCESSOR_HEADING, GUARD_HEADING))
    assert "sudo -n tail" in successor, (
        "Successor rule lacks the GCE sudo -n tail retry prescription (#1764)"
    )
    assert "Permission denied" in successor or "EACCES" in successor, (
        "Successor rule GCE log-read note lacks the Permission-denied/EACCES trigger"
    )
    assert "probe artifact" in successor, (
        "Successor rule GCE log-read note lacks the never-misclassify (probe artifact) clause"
    )
    heartbeat = _norm(_region(text, HEARTBEAT_HEADING, REVIVAL_HEADING))
    assert "sudo -n tail" in heartbeat or "GCE log-read note" in heartbeat, (
        "Long-phase heartbeat step 2(i) does not reference the GCE log-read note (#1764)"
    )


def test_mirror_duty_lists_carry_harvest() -> None:
    """Both paths-triggered mirror duty-lists carry the harvest= token (#957 shape)."""
    for path in (CODE_STYLE_MD, VECTORIZE_MD):
        assert path.exists(), f"missing {path}"
        text = _norm(path.read_text(encoding="utf-8"))
        assert "harvest=" in text, f"{path.name} mirror duty-list lacks the harvest= token"


def test_probe_bracket_and_choom_bigrss_routing_pinned() -> None:
    """#1482: bracketed probes; choom bounded retry; big-RSS off-VM routing.

    (a) The detached block carries the Probe-bracket rule (every pattern-based
    liveness / ownership / kill probe uses the bracket idiom; an ALIVE read
    from an UNBRACKETED probe is UNVERIFIED evidence; cites #1482 + the
    gotchas ownership-probe entry), and the Successor / re-entry rule's
    pattern-based FALLBACK probe clause repeats it. (b) The earlyoom
    paragraph's on-failure path is ONE BOUNDED retry — a seconds bound plus
    "whichever comes FIRST", never an open-ended wait. (c) The big-RSS
    routing clause (post-retry choom=failed at projected peak RSS >= ~16 GiB
    routes the phase off the VM) is pinned in BOTH SKILL.md's earlyoom region
    and code-style.md's amended on-failure sentence.
    """
    text = _skill_text()
    block = _norm(_region(text, DETACHED_HEADING, SUCCESSOR_HEADING))
    # (a) Probe-bracket rule inside the detached block.
    assert "bracket idiom" in block, "detached block lacks the Probe-bracket rule"
    assert "UNBRACKETED" in block and "UNVERIFIED" in block, (
        "Probe-bracket rule lacks the unbracketed-ALIVE-is-unverified clause"
    )
    assert "#1482" in block, "Probe-bracket rule does not cite #1482"
    assert "ownership-probe" in block, "Probe-bracket rule lacks the gotchas cross-ref"
    successor = _norm(_region(text, SUCCESSOR_HEADING, GUARD_HEADING))
    assert "bracket idiom" in successor, (
        "Successor rule's pattern-based fallback probe lacks the bracket idiom"
    )
    assert "UNBRACKETED" in successor and "UNVERIFIED" in successor, (
        "Successor rule lacks the unbracketed-probe-ALIVE-is-unverified clause"
    )
    # (b) Bounded retry: RE-RUN ONCE + a seconds bound + "whichever comes FIRST".
    assert re.search(r"RE-RUN (it|the sweep) ONCE", block), (
        "earlyoom paragraph lacks the RE-RUN-once bounded retry"
    )
    assert re.search(r"~?30-60\s*s\b", block), (
        "choom bounded-retry sentence lacks the ~30-60 s seconds bound"
    )
    assert "whichever comes FIRST" in block, (
        "choom bounded-retry sentence lacks the whichever-comes-FIRST bound token"
    )
    # (c) Big-RSS routing clause, in SKILL.md's earlyoom region ...
    assert re.search(r"16 GiB.{0,220}routing the phase OFF", block), (
        "earlyoom paragraph lacks the >= ~16 GiB post-retry choom=failed off-VM routing default"
    )
    # ... and in code-style.md's amended mirror sentence.
    cs = _norm(CODE_STYLE_MD.read_text(encoding="utf-8"))
    assert re.search(r"RE-RUN (it|the sweep) ONCE", cs), (
        "code-style.md nohup bullet lacks the RE-RUN-once bounded retry"
    )
    assert re.search(r"~?30-60\s*s\b", cs) and "whichever comes FIRST" in cs, (
        "code-style.md nohup bullet lacks the bounded-retry seconds bound"
    )
    assert re.search(r"16 GiB.{0,80}route the phase off", cs), (
        "code-style.md nohup bullet lacks the >= ~16 GiB off-VM routing clause"
    )
