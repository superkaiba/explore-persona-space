"""Durability pins for the #2246 Step 10d reap-shield edits (items 1 + 3).

Item 1 (AC-1): both detached gate launcher recipes carry the trailing
``"$WT"`` argv holder. ``worktree_audit._live_worktree_holders`` harvests
the ``.claude/worktrees/<name>`` marker from ``/proc/<pid>/cwd`` AND
``/proc/<pid>/cmdline`` — the env-only ``WT=\\"$WT\\"`` assignment is
invisible there (``env`` execs ``bash``, so the assignment never reaches
the final argv). The trailing ignored positional puts the worktree path in
the detached workload's argv for its whole life, making a live gate a
liveness holder the daily stale-worktree sweep respects (#2242: the sweep
removed the worktree mid-gate).

Item 3 (AC-5): the lint-gate workload fails CLOSED on an
empty-but-successful overlay listing. On the code-bearing trigger branch
the ``--no-renames`` overlay path set is a superset of the own-diff path
set, so an EMPTY listing from a ZERO-exit producer means the listing was
computed against the wrong/absent tree (or a mid-window ref mutation) and
must route to the existing crash arm via ``GT_RC=1`` — never certify
``pass``. Pinned as an ORDERED chain (overlay producer -> empty-file guard
-> crash predicate, index-monotone) plus the single-verdict-writer
property: no verdict write occurs between the producer and the crash
predicate, and the crash-verdict-write count in the lint-gate region stays
at its pre-#2246 measured value (2 — the failed-trigger-diff arm + the
verdict crash arm; the #2246 insert adds ZERO verdict-write sites).

Reads the COMPOSED skill text (``tests/issue_skill_source.py``) so the
pins bind on the logical document regardless of which step file the
content lives in (#2155). Update the pinned anchors in the SAME diff as
any restructure that renames the ``####`` section headings below.
"""

from tests.issue_skill_source import issue_skill_text

# Region anchors (same convention as tests/test_gate_recipe_no_heredoc_argv.py):
# start inclusive, end exclusive.
_LINT_GATE_REGION = (
    "step10d-lint-gate",
    "#### Pre-push workflow-lint gate",
    "#### The auto-merge procedure",
)
_SURGICAL_REGION = (
    "step10d-surgical-twin",
    "#### The artifact-confirmed merge procedure",
    "#### Post-merge stale-task-folder guard",
)

# AC-1: the launcher line's argv holder, verbatim (the recipe text carries
# escaped quotes because the launcher body lives inside a bash -c "..."
# string).
_LINT_LAUNCHER_WITH_HOLDER = r"""bash '$LINT_GATE_SCRIPT' \"$WT\" < /dev/null"""
_SURGICAL_LAUNCHER_WITH_HOLDER = r"""bash '$SURGICAL_SCRIPT' \"$WT\" < /dev/null"""

# AC-5 ordered-chain anchors (each asserted unique in the lint-gate region).
_PRODUCER_ANCHOR = "> /tmp/issue-<N>-overlay-files.txt || GT_RC=1"
_EMPTY_GUARD_TEST = "[ ! -s /tmp/issue-<N>-overlay-files.txt ]"
_EMPTY_GUARD_MARKER = "overlay listing EMPTY on a code-bearing payload"
_CRASH_PREDICATE = '[ "$GT_RC" -ne 0 ]'
# Any write (`>` or `>>`) into the verdict file contains this substring.
_VERDICT_WRITE = "> /tmp/issue-<N>-lint-verdict.txt"
_CRASH_VERDICT_WRITE = "echo crash > /tmp/issue-<N>-lint-verdict.txt"
# Pre-#2246 measured count of crash-verdict-write sites in the lint-gate
# region (the failed-trigger-diff arm + the verdict crash arm). The #2246
# empty-overlay guard routes through GT_RC=1 and adds ZERO write sites, so
# the post-edit count must equal the pre-edit count.
_CRASH_VERDICT_WRITE_COUNT = 2


def _region(name: str, start: str, end: str) -> str:
    text = issue_skill_text()
    i = text.find(start)
    assert i != -1, f"{name}: start anchor not found — update the pinned anchor: {start!r}"
    j = text.find(end, i)
    assert j != -1, f"{name}: end anchor not found — update the pinned anchor: {end!r}"
    return text[i:j]


def test_lint_gate_launcher_carries_wt_argv_holder():
    region = _region(*_LINT_GATE_REGION)
    assert _LINT_LAUNCHER_WITH_HOLDER in region, (
        'lint-gate launcher lost its trailing "$WT" argv holder (#2246 item 1) — '
        "without it the detached gate workload is invisible to worktree_audit's "
        "cwd/argv liveness harvest and the daily sweep can remove the worktree "
        "mid-gate (#2242). Restore: bash '$LINT_GATE_SCRIPT' \\\"$WT\\\" < /dev/null"
    )


def test_surgical_gate_launcher_carries_wt_argv_holder():
    region = _region(*_SURGICAL_REGION)
    assert _SURGICAL_LAUNCHER_WITH_HOLDER in region, (
        'surgical-gate launcher lost its trailing "$WT" argv holder (#2246 item 1) — '
        "without it the detached gate workload is invisible to worktree_audit's "
        "cwd/argv liveness harvest and the daily sweep can remove the worktree "
        "mid-gate (#2242). Restore: bash '$SURGICAL_SCRIPT' \\\"$WT\\\" < /dev/null"
    )


def test_overlay_empty_assert_ordered_chain():
    """The empty-overlay fail-closed block sits BETWEEN the overlay producer
    and the crash predicate, routes through GT_RC=1 (never a new verdict
    site), and the single-verdict-writer property holds (#2246 item 3)."""
    region = _region(*_LINT_GATE_REGION)

    # Anchor uniqueness — a duplicated anchor would make the ordering pin
    # meaningless.
    assert region.count(_PRODUCER_ANCHOR) == 1, (
        f"overlay producer anchor not unique in the lint-gate region "
        f"(count={region.count(_PRODUCER_ANCHOR)}): {_PRODUCER_ANCHOR!r}"
    )
    assert region.count(_EMPTY_GUARD_TEST) == 1, (
        f"empty-overlay guard test not unique (count={region.count(_EMPTY_GUARD_TEST)}): "
        f"{_EMPTY_GUARD_TEST!r}"
    )
    assert region.count(_EMPTY_GUARD_MARKER) == 1, (
        f"empty-overlay guard marker literal not unique "
        f"(count={region.count(_EMPTY_GUARD_MARKER)}): {_EMPTY_GUARD_MARKER!r}"
    )
    assert region.count(_CRASH_PREDICATE) == 1, (
        f"crash-predicate anchor not unique (count={region.count(_CRASH_PREDICATE)}): "
        f"{_CRASH_PREDICATE!r}"
    )

    # Ordered chain: producer -> guard test -> guard marker -> GT_RC=1 inside
    # the guard block -> crash predicate (index-monotone).
    p = region.index(_PRODUCER_ANCHOR)
    g_test = region.index(_EMPTY_GUARD_TEST)
    g_marker = region.index(_EMPTY_GUARD_MARKER)
    c = region.index(_CRASH_PREDICATE)
    assert p < g_test < g_marker < c, (
        f"ordered chain broken (producer={p}, guard-test={g_test}, "
        f"guard-marker={g_marker}, crash-predicate={c}) — the empty-overlay "
        f"guard must sit between the overlay producer and the crash predicate"
    )
    # The guard sets GT_RC=1 after its marker line, before the block closes.
    fi_idx = region.index("\n    fi", g_marker)
    assert region.find("GT_RC=1", g_marker, fi_idx) != -1, (
        "the empty-overlay guard no longer sets GT_RC=1 after its marker line — "
        "the fail-closed routing to the crash arm is broken (#2246 item 3)"
    )

    # Single-verdict-writer property: no verdict write between the producer
    # and the crash predicate (a substring search on `> .../lint-verdict.txt`
    # also matches the `>>` append form).
    span = region[p + len(_PRODUCER_ANCHOR) : c]
    assert _VERDICT_WRITE not in span, (
        "a verdict-write site appeared between the overlay producer and the "
        "crash predicate — the crash arm must stay the single verdict writer "
        "for construction failures (#1082-class hazard)"
    )

    # Crash-verdict-write count preserved at the pre-#2246 measured value.
    n = region.count(_CRASH_VERDICT_WRITE)
    assert n == _CRASH_VERDICT_WRITE_COUNT, (
        f"crash-verdict-write count in the lint-gate region changed "
        f"({n} != {_CRASH_VERDICT_WRITE_COUNT}) — the #2246 empty-overlay guard "
        f"adds ZERO verdict-write sites; a new/removed `echo crash` site needs "
        f"a deliberate re-measure of this pin"
    )
