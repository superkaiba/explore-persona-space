"""Tests for ``workflow_lint --check-shared-tmp-name`` (#2336).

The check FAILs any line under ``scripts/`` + ``src/`` (``*.py`` and
``*.sh``; ``tests/`` deliberately excluded) that derives a sidecar TEMP
path from the destination's own name — the process-SHARED ``<name>.tmp``
class: two concurrent writers of the same destination collide
mid-``os.replace`` and one crashes ``FileNotFoundError`` at the replace
stage (#2329 r3). Remedy: ``explore_persona_space.atomic_io.atomic_replace``.

Covers, per plan #2336 v3 §4 step 5:

1. the 25-row predicate regression table (arms A/B/C/D/E/F incl. the
   deliberate v3 flip — ``fname + ".tmp"`` is non-C but HITS as arm D —
   the two ``.suffix`` exclusions, the three arm-E ``getpid(``/``uuid4(``
   exemptions, the arm-F exemption, and the dir-shape WAIVER row);
2. ``test_shared_tmp_discovery_walks_real_tree`` — the POSITIVE pin on
   production file discovery: the REAL entrypoint
   ``check_shared_tmp_name(root=tmp_repo, allowlist=())`` walks
   ``scripts/*.py``, ``scripts/*.sh`` (heredoc line included) and
   ``src/**/*.py`` and excludes ``tests/`` — closing the hollow-gate hole
   where predicate fixtures + registration-source assertions could all
   pass while the real walk discovered no files at all — plus the
   source-level assertion that the dispatch passes no file-list override;
3. ``test_shared_tmp_check_bundled_in_no_flags`` — the MUTATION-VISIBLE
   no-flags dispatch test (the ``test_check_jsonl_splitlines_bundled_in_
   no_flags`` pattern; the test NAME is load-bearing — scripts/
   verify_plan.py matches ``test_[a-z0-9_]*bundled_in_no_flags``);
4. branch tests: stale allowlist entry WARNs (never FAILs), an
   allowlisted file with hits passes, a short (<10 char) waiver reason
   does NOT waive, the preceding-comment-line waiver form, and the
   live-tree invariant (zero errors + zero stale WARNs under the seeded
   ``SHARED_TMP_LEGACY_ALLOWLIST``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402
from workflow_lint import (  # noqa: E402
    SHARED_TMP_LEGACY_ALLOWLIST,
    check_shared_tmp_name,
)

FAILURE_TAIL = (
    "process-shared atomic-write temp name "
    "(use explore_persona_space.atomic_io.atomic_replace; #2336)"
)


def _plant(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_on_line(tmp_path: Path, line: str) -> list[str]:
    """Plant *line* as scripts/row.py in a fresh temp repo and run the
    REAL check entrypoint with an empty allowlist."""
    _plant(tmp_path, "scripts/row.py", line + "\n")
    return check_shared_tmp_name(root=tmp_path, allowlist=())


# --------------------------------------------------------------------------
# 1. The 25-row predicate regression table (plan #2336 v3 §4 step 5).
# Each row: (row id, fixture line, expect_hit). Verdicts verbatim from the
# plan table; the v2 rows carry over, the two v2 arm-B "safe" rows being
# the same fixtures the v3 table lists as arm-E getpid()/token exemptions.
# --------------------------------------------------------------------------

PREDICATE_ROWS: tuple[tuple[str, str, bool], ...] = (
    # arm A — attribute concat (also the "non-C" pin, asserted separately)
    ("A_attr_concat", 'tmp = path.name + ".tmp"', True),
    # arm B — 5 unsafe f-string shapes
    ("B_plain", 'tmp = f"{path.name}.tmp"', True),
    ("B_pt", 'tmp = f"{path.name}.tmp.pt"', True),
    ("B_hidden", 'tmp = f".{path.name}.tmp"', True),
    ("B_hidden_summary_out", 'tmp = f".{args.summary_out.name}.tmp"', True),
    ("B_hidden_ckpt", 'tmp = f".{ckpt_path.name}.tmp"', True),
    # v2 arm-B safe rows == v3 arm-E exempt rows (trailing-interpolation
    # lookahead / same-line process-varying token)
    ("E_token_after_tmp", 'tmp = f"{path.name}.tmp.{token}"', False),
    (
        "E_safe_primitive",
        'tmp = f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"',
        False,
    ),
    # arm C — 2 unsafe bare-identifier shapes (the issue1901 file)
    ("C_paren_div", 'tmp = cache_dir / (name + ".tmp")', True),
    ("C_bare_assign", 'candidate = name + ".tmp"', True),
    # arm D — generic concat (incl. the deliberate v3 flip: fname is non-C
    # but HITS as arm D, closing v2's FN (g))
    ("D_fname_flip", 'tmp = fname + ".tmp"', True),
    ("D_dest", 'tmp = dest + ".tmp"', True),
    ("D_manifest_name", 'tmp = pack_root / (MANIFEST_NAME + ".tmp")', True),
    ("D_self_out_path", 'tmp = self.out_path + ".tmp"', True),
    ("D_out_path", 'tmp = out_path + ".tmp"', True),
    # arm D — the two `.suffix` exclusions (the with_suffix class stays
    # follow-up scope)
    ("D_suffix_excluded", 'tmp = path.suffix + ".tmp"', False),
    ("D_with_suffix_excluded", 'tmp = path.with_suffix(path.suffix + ".tmp")', False),
    # arm E — generic-stem f-string hits
    ("E_donors", 'tmp = f"donors_{scheme}.tmp.pt"', True),
    ("E_regen", 'tmp = f"{block.slug}.regen.tmp.pt"', True),
    ("E_part_stem", 'tmp = f"{part.stem}.tmp.npz"', True),
    # arm E — same-line getpid() exemption (sentinel shape)
    ("E_sentinel_getpid", 'tmp = f"{CELL_DONE_SENTINEL}.{os.getpid()}.tmp"', False),
    # arm F — prefix-form hits
    ("F_shard", 'tmp = shard_path.with_name(f".tmp_{shard_path.name}")', True),
    ("F_stem_npz", 'tmp = out_dir / f".tmp_{path.stem}.npz"', True),
    # arm F — same-line getpid() exemption (the 3 pid-suffixed temp-DIR writers)
    ("F_getpid_exempt", 'tmp = out_dir / f".tmp_{arm_id}_{os.getpid()}"', False),
    # dir-shape WAIVER row: an arm-A hit carrying a well-formed waiver passes
    (
        "WAIVER_dir_shape",
        'staging = merged_dir.name + ".tmp"  # SHARED_TMP_EXEMPT: '
        "temp-DIRECTORY publish idiom, #2336 batch-2 disposition",
        False,
    ),
)


def test_predicate_table_has_25_rows() -> None:
    assert len(PREDICATE_ROWS) == 25
    assert len({row_id for row_id, _line, _hit in PREDICATE_ROWS}) == 25


@pytest.mark.parametrize(
    ("row_id", "line", "expect_hit"),
    PREDICATE_ROWS,
    ids=[row_id for row_id, _line, _hit in PREDICATE_ROWS],
)
def test_predicate_row(tmp_path: Path, row_id: str, line: str, expect_hit: bool) -> None:
    errors = _run_on_line(tmp_path, line)
    if expect_hit:
        assert errors == [f"scripts/row.py:1: {FAILURE_TAIL}"], (
            f"row {row_id}: expected exactly one hit with the exact failure line, got {errors!r}"
        )
    else:
        assert errors == [], f"row {row_id}: expected no hit, got {errors!r}"


# --------------------------------------------------------------------------
# 1b. Arm-attribution pins the plan table calls out explicitly.
# --------------------------------------------------------------------------


def test_arm_attribution_pins() -> None:
    # `path.name + ".tmp"` is arm A's territory — arm C must NOT match it.
    assert wl.SHARED_TMP_ARM_A_RE.search('path.name + ".tmp"')
    assert not wl.SHARED_TMP_ARM_C_RE.search('path.name + ".tmp"')
    # the deliberate v3 flip: `fname + ".tmp"` is non-C but arm-D catches it.
    assert not wl.SHARED_TMP_ARM_C_RE.search('fname + ".tmp"')
    assert wl.SHARED_TMP_ARM_D_RE.search('fname + ".tmp"')
    # arm C matches the bare identifier.
    assert wl.SHARED_TMP_ARM_C_RE.search('(name + ".tmp")')
    # the `.suffix` rows RAW-match arm D and are excluded by the exclusion
    # regex (exclusion, not non-match — mutation-visible either way).
    assert wl.SHARED_TMP_ARM_D_RE.search('path.suffix + ".tmp"')
    assert wl.SHARED_TMP_ARM_D_SUFFIX_EXCLUSION_RE.search('path.suffix + ".tmp"')
    # the exempt E/F rows RAW-match their arms — the same-line getpid()
    # token is what passes them, not a regex non-match.
    assert wl.SHARED_TMP_ARM_E_RE.search('f"{CELL_DONE_SENTINEL}.{os.getpid()}.tmp"')
    assert wl.SHARED_TMP_ARM_F_RE.search('f".tmp_{arm_id}_{os.getpid()}"')


# --------------------------------------------------------------------------
# 2. Discovery — the positive pin on production file discovery (round-1 MF3).
# --------------------------------------------------------------------------


def test_shared_tmp_discovery_walks_real_tree(tmp_path: Path) -> None:
    """The REAL entrypoint + walk (same function, default-equivalent walk the
    no-flags dispatch invokes) discovers `scripts/*.py`, `scripts/*.sh`
    (unsafe line inside a heredoc) and `src/**/*.py` offenders, and excludes
    `tests/` — pinned POSITIVELY on a temp repo tree."""
    _plant(tmp_path, "scripts/bad.py", 'tmp = dest + ".tmp"\n')
    _plant(
        tmp_path,
        "scripts/bad.sh",
        "#!/usr/bin/env bash\nuv run python - <<'PY'\ntmp = path.name + \".tmp\"\nPY\n",
    )
    _plant(tmp_path, "src/pkg/bad2.py", 'out = self.out_path + ".tmp"\n')
    _plant(tmp_path, "tests/ignored.py", 'tmp = path.name + ".tmp"\n')
    errors = check_shared_tmp_name(root=tmp_path, allowlist=())
    assert errors, "the real walk discovered no files at all (the hollow-gate hole)"
    paths = {err.split(":", 1)[0] for err in errors}
    assert paths == {"scripts/bad.py", "scripts/bad.sh", "src/pkg/bad2.py"}
    # Source-level: the no-flags dispatch invokes THIS entrypoint with
    # defaults — no file-list override, no alternate walk.
    src = Path(wl.__file__).read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_shared_tmp_name or no_flags:\n"
        r"\s*errors\.extend\(check_shared_tmp_name\(\)\)",
        src,
    ), "the dispatch must call check_shared_tmp_name() with no overrides"


# --------------------------------------------------------------------------
# 3. no-flags bundling (mutation-visible dispatch test; LOAD-BEARING NAME —
#    verify_plan.py's _C37_PIN_TEST_RE matches test_[a-z0-9_]*bundled_in_no_flags).
# --------------------------------------------------------------------------


def test_shared_tmp_check_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the check — deleting its
    ``or no_flags`` branch must fail this test. Other bundled checks
    contribute unrelated errors on the minimal tree, so the assertion keys
    on the check's own diagnostic token + offending path."""
    _plant(tmp_path, "scripts/offender.py", 'tmp = path.name + ".tmp"\n')
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    # Neutralize the live allowlist: on the tmp tree every live entry would
    # read stale (WARN noise), and offender.py must not inherit an escape.
    monkeypatch.setattr(wl, "SHARED_TMP_LEGACY_ALLOWLIST", ())
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert f"scripts/offender.py:1: {FAILURE_TAIL}" in err, (
        f"the shared-tmp diagnostic (naming offender.py) is missing from the "
        f"no-flags default run's stderr — the check is not bundled into "
        f"no_flags:\n{err}"
    )
    # Source-level: the check appears in the `or no_flags` dispatch chain.
    src = Path(wl.__file__).read_text(encoding="utf-8")
    assert "if args.check_shared_tmp_name or no_flags:" in src


# --------------------------------------------------------------------------
# 4. Allowlist / waiver / live-tree branches.
# --------------------------------------------------------------------------


def test_stale_allowlist_entry_warns_never_fails(tmp_path: Path) -> None:
    """An allowlisted file with ZERO hits (here: missing entirely) emits a
    `stale allowlist entry` WARN and no error."""
    _plant(tmp_path, "scripts/clean.py", "x = 1\n")
    warns: list[str] = []
    errors = check_shared_tmp_name(
        root=tmp_path,
        allowlist=(("scripts/gone.py", "already migrated"),),
        warn_sink=warns,
    )
    assert errors == []
    assert len(warns) == 1
    assert "stale allowlist entry scripts/gone.py" in warns[0]


def test_allowlisted_file_with_hits_passes(tmp_path: Path) -> None:
    _plant(tmp_path, "scripts/legacy.py", 'tmp = path.name + ".tmp"\n')
    warns: list[str] = []
    errors = check_shared_tmp_name(
        root=tmp_path,
        allowlist=(("scripts/legacy.py", "batch-0 seed"),),
        warn_sink=warns,
    )
    assert errors == []
    assert warns == []


def test_short_waiver_reason_does_not_waive(tmp_path: Path) -> None:
    errors = _run_on_line(tmp_path, 'tmp = merged_dir.name + ".tmp"  # SHARED_TMP_EXEMPT: short')
    assert errors == [f"scripts/row.py:1: {FAILURE_TAIL}"]


def test_waiver_on_preceding_comment_only_line(tmp_path: Path) -> None:
    _plant(
        tmp_path,
        "scripts/row.py",
        "# SHARED_TMP_EXEMPT: temp-DIRECTORY publish idiom, #2336 batch-2 disposition\n"
        'tmp = merged_dir.name + ".tmp"\n',
    )
    assert check_shared_tmp_name(root=tmp_path, allowlist=()) == []


def test_waiver_on_preceding_code_line_does_not_waive(tmp_path: Path) -> None:
    """The preceding-line waiver form requires a COMMENT-ONLY line — a
    trailing comment on a code line does not cover the NEXT line."""
    _plant(
        tmp_path,
        "scripts/row.py",
        "x = 1  # SHARED_TMP_EXEMPT: a reason well over ten characters\n"
        'tmp = merged_dir.name + ".tmp"\n',
    )
    errors = check_shared_tmp_name(root=tmp_path, allowlist=())
    assert errors == [f"scripts/row.py:2: {FAILURE_TAIL}"]


def test_live_tree_green_under_seeded_allowlist() -> None:
    """A6/A7: the ISOLATED check exits clean on the migrated tree — zero
    errors AND zero stale-allowlist WARNs under the batch-0 seed (every
    seeded entry still has hits; every non-allowlisted file is clean). A
    batch that migrates a file without shrinking the allowlist in the same
    commit trips the WARN half; a new offender trips the error half."""
    warns: list[str] = []
    errors = check_shared_tmp_name(warn_sink=warns)
    assert errors == [], f"non-allowlisted shared-tmp offenders on the live tree:\n{errors}"
    assert warns == [], f"stale SHARED_TMP_LEGACY_ALLOWLIST entries:\n{warns}"


def test_seed_allowlist_is_path_reason_pairs() -> None:
    assert isinstance(SHARED_TMP_LEGACY_ALLOWLIST, tuple)
    for entry in SHARED_TMP_LEGACY_ALLOWLIST:
        path, reason = entry
        assert not Path(path).is_absolute() and "\\" not in path
        assert len(reason) >= 10


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
