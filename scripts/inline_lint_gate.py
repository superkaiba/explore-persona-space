#!/usr/bin/env python3
"""Mechanized Step 9a-ter inline payload lint gate + certification writer (#1500).

Runs the two legs of the ``/issue`` Step 9a-ter § Inline payload lint gate
(no-flags ``scripts/workflow_lint.py`` + ``scripts/select_step9c_tests.py
--map-files`` -> mapped pytest), applies the gate's payload-attributed verdict
semantics (#1460), and on PASS appends a content-hash-bound certification line
per payload path (``v1 <epoch> <blobsha> <path>``) that
``.claude/hooks/guard_root_code_commit.sh`` validates before allowing a
repo-root commit of gated code paths. NEVER hand-write the cert file (#1082
parity — the guard family targets forgetting, not adversaries).

Verdict semantics (mechanizes SKILL.md Step 9a-ter § Inline payload lint gate):

- INCONCLUSIVE (exit 3, NO cert): the instrument did not run to completion —
  empty/missing payload file, lint leg missing its healthy terminal line
  (``workflow_lint: PASS`` / ``workflow_lint: FAIL (``; the ``schema FAIL``
  early-exit is deliberately rejected — it prints BEFORE any check executes),
  non-empty test mapping with no pytest summary line, or a payload path edited
  DURING the gate run (TOCTOU — the cert must bind the exact gated content).
- BLOCK (exit 1): a non-WARN output line names a payload path that is (i) NEW
  on ``origin/main`` (payload-caused by construction — every #1388/#1428
  incident offender was this case), or (ii) MODIFIED with a parseable
  ``<path>:<lineno>:`` hit whose lineno falls inside the round's added lines
  (``git diff -U0 origin/main -- <path>``), or (iii) MODIFIED with a
  payload-naming hit carrying no parseable lineno (conservative block — the
  prose gate's "pre-existing red never blocks" judgment call routes through
  ``EPM_ALLOW_ROOT_CODE_COMMIT=1`` + an ``epm:progress`` note instead).
- PASS (exit 0): repo-wide red naming only non-payload paths, WARN lines, and
  modified-file hits whose linenos all sit outside the round's added lines
  never block (they are REPORTED for the round's ``epm:progress`` note).
  Per-path certs mean a mixed verdict still certifies the clean subset.

Run as ONE background Bash (the lint leg is ~2.5-6 min; never a <=600 s
foreground bound — #991/#996)::

    uv run python scripts/inline_lint_gate.py --issue <N> \\
        --payload-file /tmp/issue-<N>-inline-payload.txt

Test-only env overrides (hermetic unit tests substitute the leg commands;
same pattern as ``EPM_LESSONS_EDIT_SENTINEL``): ``EPM_INLINE_GATE_LINT_CMD``,
``EPM_INLINE_GATE_MAP_CMD``, ``EPM_INLINE_GATE_PYTEST_CMD`` (each a shell
string run with the repo root as cwd). ``EPM_INLINE_CERT_PATH`` overrides the
cert file (shared with the hook).
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CERT_PATH = "/tmp/eps-inline-lint-cert-v1.txt"
CERT_TRIM_LINES = 500
LINT_TIMEOUT_S = 900
FETCH_TIMEOUT_S = 60
# Mapped-pytest timeout parity with select_step9c_tests.recommended_timeout_s
# (#1046): base + per-file + the test_workflow_lint.py slow surcharge, floored
# at the canonical select_step9c_tests.TIMEOUT_FLOOR_S (round-2 Minor: the
# floor-less formula gave 1 mapped non-slow test 150 s vs the canonical 900 s
# — a false-INCONCLUSIVE generator on slow machines, never a false pass).
PYTEST_BASE_S = 120
PYTEST_PER_FILE_S = 30
PYTEST_WORKFLOW_LINT_SURCHARGE_S = 900
PYTEST_TIMEOUT_FLOOR_S = 900  # select_step9c_tests.TIMEOUT_FLOOR_S parity (pinned by test)

# Healthy lint terminal line; `workflow_lint: schema FAIL` does NOT match.
LINT_TERMINAL_RE = re.compile(r"^workflow_lint: (PASS|FAIL \()", re.MULTILINE)
PYTEST_SUMMARY_RE = re.compile(r"[0-9]+ (passed|failed|error|xpassed|xfailed)|no tests ran")
# Attribution lines that are definitionally not red (pytest -rA summary rows).
NON_RED_PREFIXES = ("WARN", "PASSED", "SKIPPED")


class Inconclusive(Exception):
    """Instrument-ran completeness failure: no verdict, no cert (exit 3)."""


@dataclass
class LegResults:
    """Raw outputs of the two gate legs (pytest output kept separate so the
    lint leg's own terminal line can never satisfy the pytest summary check —
    the SKILL.md double-failure masking hazard)."""

    lint_output: str
    map_pairs: list[tuple[str, str]]
    pytest_output: str = ""


@dataclass
class Verdict:
    """Per-path gate verdict + the non-blocking report lines."""

    blocked: dict[str, list[str]] = field(default_factory=dict)
    passing: list[str] = field(default_factory=list)
    reported: list[str] = field(default_factory=list)


def _git(repo: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _git_toplevel() -> Path:
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise Inconclusive("not inside a git checkout and no --repo-root given")
    return Path(r.stdout.strip())


def _hash_object(repo: Path, path: str) -> str | None:
    """Blob sha of the WORKTREE content of *path* (None when unhashable)."""
    abs_path = repo / path
    if not abs_path.is_file():
        return None
    r = _git(repo, "hash-object", "--", str(abs_path))
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def read_payload(paths: list[str], repo: Path) -> dict[str, str]:
    """Blank-line-stripped payload -> {path: snapshot blob sha} (TOCTOU guard).

    The snapshot is taken NOW, before the multi-minute legs run; write_cert
    re-hashes and refuses to certify a path edited in between. Empty payload
    and missing/unhashable paths are INCONCLUSIVE (never a silent pass).
    """
    cleaned = [p.strip() for p in paths if p.strip()]
    if not cleaned:
        raise Inconclusive("payload-file-empty")
    snapshots: dict[str, str] = {}
    for p in cleaned:
        sha = _hash_object(repo, p)
        if sha is None:
            raise Inconclusive(f"payload path missing or unhashable: {p}")
        snapshots[p] = sha
    return snapshots


def _best_effort_choom() -> None:
    """Deprioritize this gate run for earlyoom (Step 10d precedent
    #1045/#1211/#1143). Fail-open BY DESIGN: the gate must run even where
    sudo/choom is unavailable — the skip is reported, never silent."""
    try:
        subprocess.run(
            ["sudo", "-n", "choom", "-n", "-600", "-p", str(os.getpid())],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:  # best-effort: never let deprioritization kill the gate
        print(f"inline_lint_gate: note: self-choom skipped ({exc})", file=sys.stderr)


def _bounded_fetch(repo: Path) -> None:
    """Best-effort `git fetch origin main` so new-vs-modified classification is
    current. Degrade-to-stale is safe: staleness only shifts classification in
    the STRICTER direction (an unfetched just-landed file reads as NEW)."""
    try:
        _git(repo, "fetch", "origin", "main", timeout=FETCH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(
            "inline_lint_gate: note: git fetch timed out — classifying against "
            "the last-fetched origin/main (stricter direction)",
            file=sys.stderr,
        )


def _run_leg(
    default_argv: list[str],
    override_env: str,
    repo: Path,
    timeout: int,
) -> tuple[str, int]:
    """Run one leg (env-override shell string, or the default argv), returning
    (combined stdout+stderr, returncode). A timeout returns the partial output
    with rc -1 — the missing terminal/summary line then reads INCONCLUSIVE."""
    override = os.environ.get(override_env)
    try:
        if override:
            r = subprocess.run(
                override,
                shell=True,
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            r = subprocess.run(
                default_argv,
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired as exc:
        out = (
            (exc.stdout or b"").decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        err = (
            (exc.stderr or b"").decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return out + "\n" + err + f"\n[inline_lint_gate: leg timed out after {timeout}s]", -1
    return r.stdout + "\n" + r.stderr, r.returncode


def parse_map_pairs(map_output: str) -> list[tuple[str, str]]:
    """Pair-generic TSV parse of `--map-files` stdout: keep the first two
    tab-separated fields of any line carrying >=2 (additive future columns —
    e.g. #1496 — must not break this parser)."""
    pairs: list[tuple[str, str]] = []
    for line in map_output.splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 2 and fields[0].strip():
            pairs.append((fields[0].strip(), fields[1].strip()))
    return pairs


def mapped_pytest_timeout(tests: list[str]) -> int:
    """select_step9c_tests.recommended_timeout_s parity (#1046): base +
    per-file + the test_workflow_lint.py slow surcharge, floored at the
    canonical 900 s TIMEOUT_FLOOR_S (round-2 Minor)."""
    timeout = PYTEST_BASE_S + PYTEST_PER_FILE_S * len(tests)
    if "tests/test_workflow_lint.py" in tests:
        timeout += PYTEST_WORKFLOW_LINT_SURCHARGE_S
    return max(timeout, PYTEST_TIMEOUT_FLOOR_S)


def run_legs(payload_file: Path, issue: int, repo: Path, out_dir: Path) -> LegResults:
    """Run lint + mapped-pytest legs; persist audit outputs (parity with the
    pre-#1500 fenced recipe's /tmp/issue-<N>-inline-{lint,map}.txt files)."""
    _best_effort_choom()
    _bounded_fetch(repo)

    lint_output, _ = _run_leg(
        ["uv", "run", "python", "scripts/workflow_lint.py"],
        "EPM_INLINE_GATE_LINT_CMD",
        repo,
        LINT_TIMEOUT_S,
    )

    map_output, map_rc = _run_leg(
        ["uv", "run", "python", "scripts/select_step9c_tests.py", "--map-files", str(payload_file)],
        "EPM_INLINE_GATE_MAP_CMD",
        repo,
        FETCH_TIMEOUT_S + 120,
    )
    (out_dir / f"issue-{issue}-inline-map.txt").write_text(map_output, encoding="utf-8")
    if map_rc != 0:
        (out_dir / f"issue-{issue}-inline-lint.txt").write_text(lint_output, encoding="utf-8")
        raise Inconclusive(f"map leg failed (rc={map_rc}) — unclassifiable payload")
    pairs = parse_map_pairs(map_output)

    pytest_output = ""
    tests = sorted({t for t, _ in pairs})
    if tests:
        pytest_output, _ = _run_leg(
            ["uv", "run", "pytest", *tests, "-q", "-rA"],
            "EPM_INLINE_GATE_PYTEST_CMD",
            repo,
            mapped_pytest_timeout(tests),
        )

    (out_dir / f"issue-{issue}-inline-lint.txt").write_text(
        lint_output + "\n" + pytest_output, encoding="utf-8"
    )
    return LegResults(lint_output=lint_output, map_pairs=pairs, pytest_output=pytest_output)


def is_new_on_origin_main(repo: Path, path: str) -> bool:
    """True when *path* does not resolve on origin/main (incl. an unresolvable
    origin/main ref — the stricter direction)."""
    r = _git(repo, "cat-file", "-e", f"origin/main:{path}")
    return r.returncode != 0


def added_line_ranges(repo: Path, path: str) -> list[tuple[int, int]] | None:
    """Half-open new-file line ranges the round added vs origin/main, parsed
    from `git diff -U0`. None when the diff itself fails (caller blocks
    conservatively — refinement evidence unavailable)."""
    r = _git(repo, "diff", "-U0", "origin/main", "--", path)
    if r.returncode != 0:
        return None
    ranges: list[tuple[int, int]] = []
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", r.stdout, re.MULTILINE):
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        if count > 0:
            ranges.append((start, start + count))
    return ranges


def evaluate(payload: list[str], legs: LegResults, repo: Path) -> Verdict:
    """Apply the Step 9a-ter verdict semantics (module docstring) to the leg
    outputs. Raises Inconclusive on instrument-ran completeness failure."""
    if not LINT_TERMINAL_RE.search(legs.lint_output):
        raise Inconclusive(
            "lint-leg-dead — no healthy `workflow_lint: PASS|FAIL (` terminal line "
            "(`schema FAIL` early-exit is rejected)"
        )
    if legs.map_pairs and not PYTEST_SUMMARY_RE.search(legs.pytest_output):
        raise Inconclusive("pytest-leg-dead — non-empty test mapping but no pytest summary line")

    combined = legs.lint_output.splitlines() + legs.pytest_output.splitlines()
    hits: dict[str, list[str]] = {p: [] for p in payload}
    verdict = Verdict()
    for line in combined:
        stripped = line.strip()
        for p in payload:
            if p not in line:
                continue
            if stripped.startswith(NON_RED_PREFIXES):
                verdict.reported.append(line)
            else:
                hits[p].append(line)

    for p in payload:
        if not hits[p]:
            verdict.passing.append(p)
            continue
        if is_new_on_origin_main(repo, p):
            verdict.blocked[p] = [
                f"NEW on origin/main with {len(hits[p])} non-WARN payload-naming hit(s) "
                "(payload-caused by construction)",
                *hits[p],
            ]
            continue
        ranges = added_line_ranges(repo, p)
        lineno_re = re.compile(re.escape(p) + r":(\d+):")
        reasons: list[str] = []
        preexisting: list[str] = []
        for line in hits[p]:
            m = lineno_re.search(line)
            if m is None:
                reasons.append(
                    f"payload-naming hit without a parseable lineno (conservative block): {line}"
                )
            elif ranges is None:
                reasons.append(f"added-line ranges unavailable (conservative block): {line}")
            elif any(a <= int(m.group(1)) < b for a, b in ranges):
                reasons.append(f"hit inside the round's added lines: {line}")
            else:
                preexisting.append(line)
        verdict.reported.extend(preexisting)
        if reasons:
            verdict.blocked[p] = reasons
        else:
            verdict.passing.append(p)
    return verdict


def write_cert(
    passing: list[str], snapshots: dict[str, str], cert_path: Path, repo: Path
) -> tuple[list[str], list[str]]:
    """Append `v1 <epoch> <sha> <path>` lines for passing paths whose worktree
    content still matches the read_payload snapshot; a mismatch means the file
    was edited DURING the gate run -> INCONCLUSIVE for that path, no cert
    (TOCTOU guard). Append runs under flock; the 500-line trim writes
    tmp+rename so a concurrent hook read never sees a truncated file (a lost
    line only re-blocks — safe direction)."""
    certified: list[str] = []
    toctou: list[str] = []
    epoch = int(time.time())
    lines: list[str] = []
    for p in passing:
        current = _hash_object(repo, p)
        if current is None or current != snapshots[p]:
            toctou.append(p)
            continue
        lines.append(f"v1 {epoch} {snapshots[p]} {p}\n")
        certified.append(p)
    if lines:
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cert_path, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write("".join(lines))
            fh.flush()
            os.fsync(fh.fileno())
            fh.seek(0)
            all_lines = fh.read().splitlines(keepends=True)
            if len(all_lines) > CERT_TRIM_LINES:
                fd, tmp = tempfile.mkstemp(dir=str(cert_path.parent), prefix=cert_path.name + ".")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as tf:
                        tf.writelines(all_lines[-CERT_TRIM_LINES:])
                    os.replace(tmp, cert_path)
                except BaseException:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                    raise
    return certified, toctou


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--issue", type=int, required=True, help="task number (audit-file keying)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--payload-file", help="newline-delimited repo-relative payload paths")
    src.add_argument("--paths", nargs="+", help="payload paths given inline")
    ap.add_argument("--repo-root", help="checkout root override (default: invoking git toplevel)")
    ap.add_argument(
        "--out-dir", default="/tmp", help="audit-output directory (default /tmp; test support)"
    )
    args = ap.parse_args(argv)

    try:
        repo = Path(args.repo_root).resolve() if args.repo_root else _git_toplevel()
        if args.payload_file:
            payload_file = Path(args.payload_file)
            if not payload_file.is_file():
                raise Inconclusive(f"payload file missing: {payload_file}")
            raw_paths = payload_file.read_text(encoding="utf-8").splitlines()
        else:
            raw_paths = list(args.paths)
        snapshots = read_payload(raw_paths, repo)
        payload = sorted(snapshots)
        if args.payload_file is None:
            # Materialize a payload file for the map leg's --map-files contract.
            fd, tmp_payload = tempfile.mkstemp(prefix=f"issue-{args.issue}-inline-payload.")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(payload) + "\n")
            payload_file = Path(tmp_payload)
        legs = run_legs(payload_file, args.issue, repo, Path(args.out_dir))
        verdict = evaluate(payload, legs, repo)
    except Inconclusive as exc:
        print(f"inline_lint_gate: INCONCLUSIVE ({exc})")
        return 3

    cert_path = Path(os.environ.get("EPM_INLINE_CERT_PATH", DEFAULT_CERT_PATH))
    certified, toctou = write_cert(verdict.passing, snapshots, cert_path, repo)

    for line in verdict.reported:
        print(f"inline_lint_gate: report (pre-existing / WARN — never blocks): {line}")
    for p in sorted(verdict.blocked):
        for reason in verdict.blocked[p]:
            print(f"inline_lint_gate: {p}: {reason}")
    for p in certified:
        print(f"inline_lint_gate: certified {p} ({snapshots[p][:12]}) -> {cert_path}")

    # TOCTOU note prints BEFORE any BLOCK return (round-2 Minor): in a mixed
    # BLOCK+TOCTOU outcome the operator must learn of the mid-gate edit NOW,
    # not on the next hook block. Exit precedence unchanged: BLOCK (1) beats
    # TOCTOU-only INCONCLUSIVE (3).
    if toctou:
        print(
            "inline_lint_gate: INCONCLUSIVE "
            f"(edited during gate — re-run: {' '.join(sorted(toctou))})"
        )
    if verdict.blocked:
        print(f"inline_lint_gate: BLOCK ({' '.join(sorted(verdict.blocked))})")
        return 1
    if toctou:
        return 3
    print("inline_lint_gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
