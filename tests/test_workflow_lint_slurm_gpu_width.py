"""Tests for the #2081 SLURM GPU-width check in ``scripts/workflow_lint.py``.

Check under test: ``check_slurm_gpu_width`` (``--check-slurm-gpu-width``,
bundled into the no-flags default run): FAILs any logical line in a
``scripts/*.sh`` launcher that derives GPU WIDTH from ``nvidia-smi`` device
enumeration (``-L`` / ``--list-gpus`` / ``--query-gpu=`` piped into a count
sink ``wc -l`` / ``grep -c``) when the file carries NO recognized
allocation-derived guard — a SLURM allocation-env branch, or the
inherited-``CUDA_VISIBLE_DEVICES`` parse (#2251; cases 13-16 below).
On a shared fellows SLURM node ``nvidia-smi`` enumerates all 8
physical devices and ignores ``CUDA_VISIBLE_DEVICES``, so a detected-count
fan-out trespasses onto other tenants' GPUs (#1902; worked adoption #1491
@ ``1c8b46d28a``; ``.claude/rules/gotchas.md`` "Fellows SLURM nodes are
GPU-SHARED").

Cases (plan #2081 v3 §5):

1.  ``test_flags_nvidia_smi_l_wc_l`` — ``nvidia-smi -L | wc -l`` width
    derivation, no guard.
2.  ``test_flags_list_gpus_and_query_gpu_variants`` — the ``--list-gpus``
    and ``--query-gpu=index`` variants.
2b. ``test_flags_subshell_or_true_shape`` — the subshell/``|| true`` shape
    (the ``issue1336_dispatch.sh:81`` idiom).
2c. ``test_flags_grep_c_counting_idiom`` — the ``grep -c`` counting idiom
    (the ``issue2094_dispatch.sh:37`` idiom).
3.  ``test_passes_slurm_job_id_branch`` — a file with a ``SLURM_JOB_ID``
    branch (the #1491 shape) passes.
4.  ``test_passes_realized_gpu_ids_reference`` — a file referencing
    ``realized_gpu_ids`` passes.
5.  ``test_waiver_same_line_and_preceding_line_pass`` — waiver on the same
    logical line passes; waiver on the preceding non-blank line passes.
6.  ``test_waiver_short_reason_still_fails`` — waiver with reason < 10
    chars still FAILs.
7.  ``test_grandfathered_basename_no_fail`` — a grandfathered basename
    with an offending line does not FAIL.
8.  ``test_stale_grandfather_zero_matches_warns_not_fails`` — a
    grandfathered basename with zero width-derivation matches WARNs
    ("remove <name> from SLURM_GPU_WIDTH_GRANDFATHER"), never FAILs;
    ``test_stale_grandfather_guard_present_warns_not_fails`` — a
    grandfathered basename that now carries a guard WARNs the same way.
9.  ``test_comment_and_echo_lines_skipped`` — ``#``-comment and ``echo
    ``-prefixed lines are skipped.
10. ``test_check_slurm_gpu_width_bundled_in_no_flags`` — the two-part
    no-flags bundling behavioral pin (the
    ``test_check_smoke_blind_spot_review_lens_bundled_in_no_flags``
    precedent shape).
11. ``test_live_tree_green`` — ``check_slurm_gpu_width()`` on the real
    ``scripts/`` returns an empty FAIL list and zero stale-grandfather
    WARNs (binds the grandfather calibration to the tree).
12. ``test_inverse_calibration_pin`` — on the real ``scripts/``, the
    predicate-matched-minus-guarded-minus-waived basename set ==
    ``SLURM_GPU_WIDTH_GRANDFATHER`` exactly, so a drift in either
    direction (regex widened without re-freezing, or a launcher fixed
    without removing its entry) is test-breaking, not silent; the two
    known GUARDED files pass NATURALLY (never via the grandfather). The
    guard re-scan uses the SHARED predicate
    ``_slurm_gpu_width_guard_present`` (#2251) so it cannot drift from
    ``check_slurm_gpu_width``'s own scan site.

Cases 13-16 (#2251 — the inherited-``CUDA_VISIBLE_DEVICES`` parse as a
second recognized guard form, the #1336 round-v21 shape):

13. ``test_passes_cvd_parse_guard`` — positive arm: the realized #1336
    guard block (``read -ra`` from CVD + same-name ``${#NAME[@]}`` count,
    nvidia-smi only as the unset-CVD fallback) passes.
14. ``test_bare_nvidia_smi_with_cvd_pin_literals_still_fails`` — negative
    arm (the task's point): bare nvidia-smi width + literal
    ``CUDA_VISIBLE_DEVICES=0`` / ``=$i`` pin sites still FAILs — mere
    mention of CVD is not a guard.
15. ``test_cvd_read_without_same_name_count_still_fails`` — half-shape
    arm: the ``read -ra`` CVD populate present but width still from
    nvidia-smi and the only array-count deref on a DIFFERENT name still
    FAILs (pins the same-name back-reference requirement).
16. ``test_stale_grandfather_cvd_guard_adopted_warns_not_fails`` —
    hygiene arm: a grandfathered basename whose text carries the
    CVD-parse guard passes naturally and emits the guard-adopted
    remove-WARN (the #2251 ratchet trigger; mirrors case 8).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import (  # noqa: E402
    SLURM_GPU_WIDTH_GRANDFATHER,
    _slurm_gpu_width_guard_present,
    _slurm_gpu_width_matches,
    _slurm_gpu_width_waiver_present,
    check_slurm_gpu_width,
)

# --------------------------------------------------------------------------
# Fixtures (only ever written into tmp scripts_dir trees; never executed).
# --------------------------------------------------------------------------

_OFFENDER_WC = """\
#!/usr/bin/env bash
set -euo pipefail

NGPU=$(nvidia-smi -L | wc -l)
for i in $(seq 0 $((NGPU - 1))); do launch_worker "$i"; done
"""

_OFFENDER_LIST_GPUS = """\
#!/usr/bin/env bash
NGPU=$(nvidia-smi --list-gpus | wc -l)
"""

_OFFENDER_QUERY_GPU = """\
#!/usr/bin/env bash
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
"""

# The issue1336_dispatch.sh:81 idiom: subshell + || true between the
# enumeration and the count sink.
_OFFENDER_SUBSHELL = """\
#!/usr/bin/env bash
NGPU=$( (nvidia-smi --list-gpus 2>/dev/null || true) | wc -l )
"""

# The issue2094_dispatch.sh:37 idiom: grep -c counting instead of wc -l.
_OFFENDER_GREP_C = """\
#!/usr/bin/env bash
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
"""

# The #1491 shape: the SLURM branch lives far from the enumeration
# fallback — guard detection must be file-scoped, not line-local.
_GUARDED_SLURM_BRANCH = """\
#!/usr/bin/env bash
set -euo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
    NGPU="${SLURM_GPUS_ON_NODE:?FATAL: SLURM job exposes no GPU count}"
else
    NGPU=$(nvidia-smi -L | wc -l)
fi
"""

_GUARDED_REALIZED_GPU_IDS = """\
#!/usr/bin/env bash
GPU_IDS=$(uv run python scripts/issue1902_common.py realized_gpu_ids)
NGPU=$(nvidia-smi -L | wc -l)
"""

_WAIVED_SAME_LINE = """\
#!/usr/bin/env bash
NGPU=$(nvidia-smi -L | wc -l)  # SLURM_GPU_WIDTH_EXEMPT: workstation-only, never on SLURM
"""

_WAIVED_PRECEDING_LINE = """\
#!/usr/bin/env bash
# SLURM_GPU_WIDTH_EXEMPT: RunPod-exclusive host, enumeration is the correct width source
NGPU=$(nvidia-smi -L | wc -l)
"""

_WAIVED_SHORT_REASON = """\
#!/usr/bin/env bash
NGPU=$(nvidia-smi -L | wc -l)  # SLURM_GPU_WIDTH_EXEMPT: short
"""

_COMMENT_AND_ECHO_ONLY = """\
#!/usr/bin/env bash
# NGPU=$(nvidia-smi -L | wc -l)
echo "dry-run preview: NGPU=$(nvidia-smi -L | wc -l)"
"""

_CLEAN_NO_MATCH = """\
#!/usr/bin/env bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "no width derivation here"
"""

_GF_NAME = "issue1310_dispatch.sh"  # a real SLURM_GPU_WIDTH_GRANDFATHER member
_GF_NAME_2 = "issue1335_run.sh"  # a second real member (guard-adopted variant)
_GF_NAME_3 = "issue1345_dispatch.sh"  # a third real member (CVD-guard-adopted variant, #2251)

_GF_WITH_GUARD = """\
#!/usr/bin/env bash
if [ -n "${SLURM_JOB_ID:-}" ]; then
    NGPU="${SLURM_GPUS_ON_NODE:?}"
else
    NGPU=$(nvidia-smi -L | wc -l)
fi
"""

# The realized #1336 round-v21 guard shape (#2251): the block below is the
# VERBATIM guard from `git show 6ff22758:scripts/issue1336_dispatch.sh`
# lines 100-106 (branch issue-1336-fullcorpora) — parse the INHERITED
# CUDA_VISIBLE_DEVICES (the allocated device LIST) into an array and take
# the array count; nvidia-smi enumeration is the fallback ONLY when CVD is
# unset/empty — plus a worker pin site over the parsed array (the script's
# line-195 shape).
_GUARDED_CVD_PARSE = """\
#!/usr/bin/env bash
set -euo pipefail

EPS_ALLOC_GPUS=()
if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
    IFS=',' read -ra EPS_ALLOC_GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NGPU=${#EPS_ALLOC_GPUS[@]}
else
    NGPU=$( (nvidia-smi --list-gpus 2>/dev/null || true) | wc -l )
fi
CUDA_VISIBLE_DEVICES=${EPS_ALLOC_GPUS[w]} bash -c "$cmd"
"""

# Negative arm (#2251 — the task's point): bare nvidia-smi width derivation
# with literal CUDA_VISIBLE_DEVICES pin sites. Mere MENTION of CVD (no
# `read -ra` parse, no array-count derivation) is not a guard.
_CVD_PIN_LITERALS_NO_GUARD = """\
#!/usr/bin/env bash
NGPU=$( (nvidia-smi --list-gpus 2>/dev/null || true) | wc -l )
CUDA_VISIBLE_DEVICES=0 bash -c "$cmd0"
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i bash -c "$cmd"
done
"""

# Half-shape arm (#2251): the `read -ra` CVD populate is present, but width
# still comes from nvidia-smi and the ONLY array-count deref is on a
# DIFFERENT name — the populate alone (or a mismatched count) is not a
# guard (the same-name back-reference requirement).
_CVD_READ_NO_SAME_NAME_COUNT = """\
#!/usr/bin/env bash
IFS=',' read -ra ARR <<< "$CUDA_VISIBLE_DEVICES"
NGPU=$(nvidia-smi -L | wc -l)
W=${#OTHER[@]}
"""


def _lineno_of(src: str, needle: str, occurrence: int = 1) -> int:
    count = 0
    for i, line in enumerate(src.splitlines(), start=1):
        if needle in line:
            count += 1
            if count == occurrence:
                return i
    raise AssertionError(f"{needle!r} (occurrence {occurrence}) not in fixture")


def _run(tmp_path: Path, fixtures: dict[str, str]) -> tuple[list[str], list[str], dict[str, Path]]:
    """Write fixtures under ``tmp_path`` (as the scripts_dir override), run
    the check, return (errors, warn_sink, name->path). The warn_sink also
    collects the missing-file WARNs for the other grandfather entries
    absent from the tmp tree — tests assert with ``any(...)``, never
    ``sink == []``, except on the live tree."""
    paths: dict[str, Path] = {}
    for name, src in fixtures.items():
        p = tmp_path / name
        p.write_text(src, encoding="utf-8")
        paths[name] = p
    sink: list[str] = []
    errors = check_slurm_gpu_width(scripts_dir=tmp_path, warn_sink=sink)
    return errors, sink, paths


# --------------------------------------------------------------------------
# Detection (cases 1, 2, 2b, 2c)
# --------------------------------------------------------------------------


def test_flags_nvidia_smi_l_wc_l(tmp_path: Path) -> None:
    """Case 1: `nvidia-smi -L | wc -l` width derivation, no guard — FAIL
    naming the mechanism (#1902), the reference impl, the worked adoption,
    the gotchas rule, and the waiver token."""
    errors, _, paths = _run(tmp_path, {"offender.sh": _OFFENDER_WC})
    assert len(errors) == 1, f"expected exactly one FAIL; got: {errors}"
    lineno = _lineno_of(_OFFENDER_WC, "nvidia-smi -L | wc -l")
    assert f"{paths['offender.sh']}:{lineno}:" in errors[0]
    for token in (
        "#1902",
        "scripts/issue1902_common.py::realized_gpu_ids",
        "scripts/issue1491_ladder_launch.sh @ 1c8b46d28a",
        "gotchas.md",
        "SLURM_GPU_WIDTH_EXEMPT",
    ):
        assert token in errors[0], f"FAIL message must name {token!r}; got: {errors[0]}"


def test_flags_list_gpus_and_query_gpu_variants(tmp_path: Path) -> None:
    """Case 2: the --list-gpus and --query-gpu=index variants both fire."""
    errors, _, paths = _run(
        tmp_path,
        {"list_gpus.sh": _OFFENDER_LIST_GPUS, "query_gpu.sh": _OFFENDER_QUERY_GPU},
    )
    assert len(errors) == 2, f"expected one FAIL per variant; got: {errors}"
    assert any(str(paths["list_gpus.sh"]) in e for e in errors)
    assert any(str(paths["query_gpu.sh"]) in e for e in errors)


def test_flags_subshell_or_true_shape(tmp_path: Path) -> None:
    """Case 2b: the subshell/`|| true` shape (issue1336_dispatch.sh:81) —
    the sink is searched AFTER the enumeration match's end, so the
    intervening `2>/dev/null || true)` does not defeat it."""
    errors, _, paths = _run(tmp_path, {"subshell.sh": _OFFENDER_SUBSHELL})
    lineno = _lineno_of(_OFFENDER_SUBSHELL, "nvidia-smi --list-gpus")
    assert len(errors) == 1 and f"{paths['subshell.sh']}:{lineno}:" in errors[0], (
        f"expected the subshell shape to FAIL at line {lineno}; got: {errors}"
    )


def test_flags_grep_c_counting_idiom(tmp_path: Path) -> None:
    """Case 2c: the `grep -c '^GPU '` counting idiom (issue2094_dispatch.sh:37)."""
    errors, _, paths = _run(tmp_path, {"grepc.sh": _OFFENDER_GREP_C})
    lineno = _lineno_of(_OFFENDER_GREP_C, "grep -c")
    assert len(errors) == 1 and f"{paths['grepc.sh']}:{lineno}:" in errors[0], (
        f"expected the grep -c idiom to FAIL at line {lineno}; got: {errors}"
    )


# --------------------------------------------------------------------------
# Guard scan (cases 3, 4)
# --------------------------------------------------------------------------


def test_passes_slurm_job_id_branch(tmp_path: Path) -> None:
    """Case 3: the #1491 shape — SLURM_JOB_ID branch far from the
    enumeration fallback; the file-scoped guard scan passes it NATURALLY."""
    errors, _, _ = _run(tmp_path, {"guarded.sh": _GUARDED_SLURM_BRANCH})
    assert errors == [], f"a SLURM_JOB_ID-guarded file must pass; got: {errors}"


def test_passes_realized_gpu_ids_reference(tmp_path: Path) -> None:
    """Case 4: a file referencing realized_gpu_ids (the #1902 reference
    impl) passes."""
    errors, _, _ = _run(tmp_path, {"realized.sh": _GUARDED_REALIZED_GPU_IDS})
    assert errors == [], f"a realized_gpu_ids-referencing file must pass; got: {errors}"


# --------------------------------------------------------------------------
# Waiver (cases 5, 6)
# --------------------------------------------------------------------------


def test_waiver_same_line_and_preceding_line_pass(tmp_path: Path) -> None:
    """Case 5: `# SLURM_GPU_WIDTH_EXEMPT: <reason>` (reason >= 10 chars) on
    the same logical line OR the immediately preceding non-blank line
    suppresses the FAIL."""
    errors, _, _ = _run(
        tmp_path,
        {"same_line.sh": _WAIVED_SAME_LINE, "prev_line.sh": _WAIVED_PRECEDING_LINE},
    )
    assert errors == [], f"waived lines must not FAIL; got: {errors}"


def test_waiver_short_reason_still_fails(tmp_path: Path) -> None:
    """Case 6: a waiver with reason < 10 chars is not a waiver."""
    errors, _, paths = _run(tmp_path, {"short.sh": _WAIVED_SHORT_REASON})
    assert len(errors) == 1 and str(paths["short.sh"]) in errors[0], (
        f"a short-reason waiver must still FAIL; got: {errors}"
    )


# --------------------------------------------------------------------------
# Grandfather + stale-entry hygiene (cases 7, 8)
# --------------------------------------------------------------------------


def test_grandfathered_basename_no_fail(tmp_path: Path) -> None:
    """Case 7: an offending file whose basename is in
    SLURM_GPU_WIDTH_GRANDFATHER is skipped for FAIL purposes."""
    errors, sink, _ = _run(tmp_path, {_GF_NAME: _OFFENDER_WC})
    assert errors == [], f"grandfathered basenames must not FAIL; got: {errors}"
    # Still matched (not stale): no remove-WARN for THIS entry.
    assert not any(_GF_NAME in w and "remove" in w for w in sink), (
        f"a matched grandfathered file must not WARN stale; got: {sink}"
    )


def test_stale_grandfather_zero_matches_warns_not_fails(tmp_path: Path) -> None:
    """Case 8: a grandfathered basename with ZERO width-derivation matches
    WARNs 'remove <name> from SLURM_GPU_WIDTH_GRANDFATHER' — never FAILs."""
    errors, sink, _ = _run(tmp_path, {_GF_NAME: _CLEAN_NO_MATCH})
    assert errors == [], f"stale grandfather entries never FAIL; got: {errors}"
    assert any(
        f"remove {_GF_NAME} from SLURM_GPU_WIDTH_GRANDFATHER" in w and "zero" in w for w in sink
    ), f"expected the zero-matches remove-WARN for {_GF_NAME}; got: {sink}"


def test_stale_grandfather_guard_present_warns_not_fails(tmp_path: Path) -> None:
    """Case 8 (guard-adopted arm): a grandfathered basename that now
    carries a SLURM guard passes naturally and WARNs to remove the entry."""
    errors, sink, _ = _run(tmp_path, {_GF_NAME_2: _GF_WITH_GUARD})
    assert errors == [], f"a guard-adopted grandfathered file never FAILs; got: {errors}"
    assert any(
        f"remove {_GF_NAME_2} from SLURM_GPU_WIDTH_GRANDFATHER" in w and "guard" in w for w in sink
    ), f"expected the guard-adopted remove-WARN for {_GF_NAME_2}; got: {sink}"


# --------------------------------------------------------------------------
# Comment / echo skip (case 9)
# --------------------------------------------------------------------------


def test_comment_and_echo_lines_skipped(tmp_path: Path) -> None:
    """Case 9: `#`-comment and `echo `-prefixed logical lines (dry-run
    previews) are not launches and never FAIL."""
    errors, _, _ = _run(tmp_path, {"preview.sh": _COMMENT_AND_ECHO_ONLY})
    assert errors == [], f"comment/echo lines must be skipped; got: {errors}"


# --------------------------------------------------------------------------
# No-flags bundling pin (case 10)
# --------------------------------------------------------------------------


def test_check_slurm_gpu_width_bundled_in_no_flags(tmp_path: Path) -> None:
    """Two-part behavioral bundling pin (the
    test_check_smoke_blind_spot_review_lens_bundled_in_no_flags precedent).

    Part A — scoped-flag subprocess against a tmp corpus carrying one
    offender, rooted via ``EPS_WORKFLOW_LINT_REPO_ROOT``: proves the flag
    exists, the dispatch calls the function, and it FAILs (nonzero exit)
    with the check's distinctive message.

    Part B — no-flags OR-chain + dispatch-ladder evidence: ``main()``'s
    source names ``args.check_slurm_gpu_width`` in BOTH the
    ``no_flags = not (...)`` OR-chain and the ``or no_flags`` dispatch
    ladder.
    """
    # Part A — scoped-flag subprocess against a one-offender corpus.
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "offender.sh").write_text(_OFFENDER_WC, encoding="utf-8")
    workflow_yaml_src = _REPO_ROOT / ".claude" / "workflow.yaml"
    workflow_yaml_dst = tmp_path / ".claude" / "workflow.yaml"
    workflow_yaml_dst.parent.mkdir(parents=True, exist_ok=True)
    workflow_yaml_dst.write_bytes(workflow_yaml_src.read_bytes())
    lint_script = _REPO_ROOT / "scripts" / "workflow_lint.py"
    env = {**os.environ, "EPS_WORKFLOW_LINT_REPO_ROOT": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            str(lint_script),
            "--check-slurm-gpu-width",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "derives GPU width from nvidia-smi" in combined, (
        "slurm-gpu-width error token missing from output — the CLI flag "
        "does not dispatch the check. "
        f"exit={result.returncode}, combined output:\n{combined}"
    )
    assert result.returncode != 0, (
        f"expected nonzero exit on the offender corpus; got exit="
        f"{result.returncode}, combined output:\n{combined}"
    )

    # Part B — OR-chain + dispatch ladder evidence.
    lint_src = lint_script.read_text(encoding="utf-8")
    main_start = lint_src.find("def main(")
    assert main_start >= 0, "could not locate def main( in workflow_lint.py"
    main_end = lint_src.find('if __name__ == "__main__":', main_start)
    assert main_end > main_start, "could not locate main() end sentinel"
    main_src = lint_src[main_start:main_end]
    or_chain_start = main_src.find("no_flags = not (")
    assert or_chain_start >= 0, "no_flags OR-chain not found in main()"
    or_chain_end = main_src.find(")", or_chain_start)
    or_chain_src = main_src[or_chain_start:or_chain_end]
    assert "args.check_slurm_gpu_width" in or_chain_src, (
        "args.check_slurm_gpu_width is NOT in the no_flags OR-chain — a "
        "bare workflow_lint.py invocation would not fire this check. "
        f"OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_slurm_gpu_width or no_flags" in main_src, (
        "args.check_slurm_gpu_width is NOT dispatched under `or no_flags` "
        "— the flag is defined but not bundled into the no-flags default "
        "run."
    )


# --------------------------------------------------------------------------
# Live-tree calibration (cases 11, 12)
# --------------------------------------------------------------------------


def test_live_tree_green(monkeypatch) -> None:
    """Case 11: the check on the real scripts/ returns an empty FAIL list
    AND zero stale-grandfather WARNs — binds the grandfather calibration
    (re-frozen 2026-08-10) to the live tree."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    sink: list[str] = []
    errors = check_slurm_gpu_width(scripts_dir=None, warn_sink=sink)
    assert errors == [], f"live tree must be FAIL-clean; got: {errors}"
    assert sink == [], f"expected 0 stale-grandfather WARNs at landing; got: {sink}"


def test_inverse_calibration_pin() -> None:
    """Case 12: the replay that caught the plan-v1 defect. On the real
    scripts/, the predicate-matched-minus-guarded-minus-waived basename
    set == SLURM_GPU_WIDTH_GRANDFATHER exactly — a regex widened without
    re-freezing (new names appear) or a launcher fixed without removing
    its entry (a name disappears) is test-breaking, not silent. Waived
    files are excluded from the pinned set: a future launcher adopting
    the SLURM_GPU_WIDTH_EXEMPT waiver passes the check without needing a
    grandfather entry, so it must not break this pin either. The two
    known GUARDED files pass NATURALLY (acceptance criterion 3), never
    via the grandfather set.

    The guard re-scan below uses the SHARED predicate
    ``_slurm_gpu_width_guard_present`` — the same scan site
    ``check_slurm_gpu_width`` runs (#2251) — so a launcher that adopts
    the inherited-CVD parse guard (the #1336 shape) moves from
    matched-unwaived to guarded HERE too, and merging it without
    removing its grandfather entry is test-breaking, not silent."""
    scripts = _REPO_ROOT / "scripts"
    matched_unwaived: set[str] = set()
    guarded: set[str] = set()
    for sh in sorted(scripts.rglob("*.sh")):
        if not sh.is_file():
            continue
        text = sh.read_text(encoding="utf-8")
        lines = text.splitlines()
        hits = _slurm_gpu_width_matches(lines)
        if not hits:
            continue
        if _slurm_gpu_width_guard_present(text):
            guarded.add(sh.name)
            continue
        if any(not _slurm_gpu_width_waiver_present(lines, first, last) for first, last, _ in hits):
            matched_unwaived.add(sh.name)
    assert matched_unwaived == set(SLURM_GPU_WIDTH_GRANDFATHER), (
        "predicate-matched-minus-guarded-minus-waived set drifted from "
        "SLURM_GPU_WIDTH_GRANDFATHER.\n"
        f"matched-but-not-grandfathered: "
        f"{sorted(matched_unwaived - SLURM_GPU_WIDTH_GRANDFATHER)}\n"
        f"grandfathered-but-not-matched: "
        f"{sorted(set(SLURM_GPU_WIDTH_GRANDFATHER) - matched_unwaived)}"
    )
    assert {"issue1491_ladder_launch.sh", "issue1902_dispatch.sh"} <= guarded, (
        f"the two GUARDED reference launchers must pass via the guard scan; guarded={guarded}"
    )


# --------------------------------------------------------------------------
# Inherited-CUDA_VISIBLE_DEVICES parse guard (#2251; cases 13-16)
# --------------------------------------------------------------------------


def test_passes_cvd_parse_guard(tmp_path: Path) -> None:
    """Case 13 (#2251, positive arm): the realized #1336 round-v21 shape —
    the inherited CUDA_VISIBLE_DEVICES parsed via `IFS=',' read -ra` into
    an array whose same-name `${#NAME[@]}` count derives width, nvidia-smi
    enumeration only as the unset-CVD fallback — passes via the new
    recognizer, with no waiver and no grandfather entry."""
    errors, _, _ = _run(tmp_path, {"cvd_guarded.sh": _GUARDED_CVD_PARSE})
    assert errors == [], f"the inherited-CVD parse guard must pass; got: {errors}"


def test_bare_nvidia_smi_with_cvd_pin_literals_still_fails(tmp_path: Path) -> None:
    """Case 14 (#2251, negative arm — the task's point): bare nvidia-smi
    width derivation with literal `CUDA_VISIBLE_DEVICES=0` / `=$i` pin
    sites still FAILs — mere MENTION of CVD is not a guard."""
    errors, _, paths = _run(tmp_path, {"cvd_pins.sh": _CVD_PIN_LITERALS_NO_GUARD})
    lineno = _lineno_of(_CVD_PIN_LITERALS_NO_GUARD, "nvidia-smi --list-gpus")
    assert len(errors) == 1 and f"{paths['cvd_pins.sh']}:{lineno}:" in errors[0], (
        f"literal CVD pin sites must not read as a guard; got: {errors}"
    )


def test_cvd_read_without_same_name_count_still_fails(tmp_path: Path) -> None:
    """Case 15 (#2251, half-shape arm): the `read -ra` CVD populate is
    present but width still comes from nvidia-smi and the ONLY array-count
    deref is on a DIFFERENT name — the populate alone is not a guard (pins
    the same-name back-reference requirement)."""
    errors, _, paths = _run(tmp_path, {"cvd_half.sh": _CVD_READ_NO_SAME_NAME_COUNT})
    lineno = _lineno_of(_CVD_READ_NO_SAME_NAME_COUNT, "nvidia-smi -L | wc -l")
    assert len(errors) == 1 and f"{paths['cvd_half.sh']}:{lineno}:" in errors[0], (
        f"the CVD populate without a same-name count must still FAIL; got: {errors}"
    )


def test_stale_grandfather_cvd_guard_adopted_warns_not_fails(tmp_path: Path) -> None:
    """Case 16 (#2251, hygiene arm — the ratchet trigger): a grandfathered
    basename whose text now carries the inherited-CVD parse guard passes
    naturally AND emits the guard-adopted remove-WARN through the shared
    predicate (mirrors case 8's guard-adopted arm; this WARN + the case-12
    pin are what force the issue1336_dispatch.sh entry removal at the
    #1336 branch merge)."""
    errors, sink, _ = _run(tmp_path, {_GF_NAME_3: _GUARDED_CVD_PARSE})
    assert errors == [], f"a CVD-guard-adopted grandfathered file never FAILs; got: {errors}"
    assert any(
        f"remove {_GF_NAME_3} from SLURM_GPU_WIDTH_GRANDFATHER" in w and "guard" in w for w in sink
    ), f"expected the guard-adopted remove-WARN for {_GF_NAME_3}; got: {sink}"
