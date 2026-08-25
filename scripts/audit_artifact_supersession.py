#!/usr/bin/env python3
"""Audit committed artifact-supersession records against LIVE consumers (#2568).

When a round REFUTES evidence that committed downstream artifacts consume (a
claims-registry row, a paper/poster figure generator, a methodology doc), the
refuting round writes a machine-readable SUPERSESSION RECORD next to the
replacement artifact. This tool discovers every committed record, RE-RUNS the
consumer enumeration fresh (so consumers added AFTER the record lands are
caught too), and reports every consumer that neither consumes a replacement
artifact nor carries a supersession label. It only SURFACES — it never edits
consumers.

Checkout independence: sparse worktrees exclude ``eval_results/`` from disk,
so every read is INDEX-based — record discovery via ``git ls-files``,
record content via ``git cat-file blob :<path>`` (never a working-tree
``open()``), consumer enumeration via ``git grep --cached``, and consumer
content via index blobs. A record passed via ``--record`` must therefore be
in the git index (``git add`` it first); a not-in-index path is a usage
error (exit 2). A tracked record whose index bytes are NOT decodable as
UTF-8 is a MALFORMED record — WARN by default, FAIL under ``--strict`` —
never a usage error (a non-UTF-8 record must not hard-block the fleet
no-flags lint; #2568 round 2).

Record schema (``artifact-supersession-record-v1``) — this docstring plus the
:data:`SCHEMA_VERSION` constant are the schema's single source of truth.
Records are committed as ``eval_results/issue_<N>/**/superseded_<slug>.json``
— a FILE whose BASENAME matches ``superseded_*.json``. JSON files that merely
live UNDER a ``superseded_*`` DIRECTORY are NOT records (the issue_1482
collision shape). Canonical fallback location when the replacement artifact
has no committed neighbor (e.g. HF-only): ``eval_results/issue_<N>/
superseded_<slug>.json``.

Required keys:

- ``schema``            — the literal ``"artifact-supersession-record-v1"``.
- ``issue``             — owning issue number (int).
- ``refuted_claim``     — one-line prose of the refuted claim/join.
- ``refuted_artifacts`` — non-empty list of artifact FILENAMES; each name
  must be a nonblank filename token (see the token rule below) with >= 8
  NON-WHITESPACE chars (bare common words make the grep useless).
- ``evidence``          — non-empty list of refuting-evidence pointers
  (marker refs, log paths).
- ``replacement_artifacts`` — list of replacement artifact filenames. MAY be
  empty when no replacement exists — then only the label arm can pass a
  consumer. Every PRESENT entry must be a nonblank filename token.

Token rule (#2568 round 2): every ``refuted_artifacts`` /
``replacement_artifacts`` / ``label_patterns`` entry must be nonblank after
stripping and carry no path separator (``/`` or ``\\``) and no control
character. A degenerate entry is a SCHEMA problem (WARN; ``--strict``:
FAIL) and the record's consumers are NEVER audited against it — an empty
replacement string is a substring of every file (a silent universal
``pass-replacement``), and a whitespace-only name/pattern matches nearly
any indented line.

Optional keys (unknown keys are tolerated — forward-compat):

- ``consumers_at_record_time`` — provenance only; the audit NEVER trusts it
  (it re-enumerates fresh on every invocation).
- ``producers``           — repo paths excluded from enumeration.
- ``acknowledged_pending`` — consumer paths whose conformance is explicitly
  routed but not yet landed; reported as WARN, not FAIL.
- ``label_patterns``      — overrides :data:`DEFAULT_LABEL_PATTERNS`.

Worked example (the #1901 plan-v15 §10 instance)::

    {
      "schema": "artifact-supersession-record-v1",
      "issue": 1901,
      "refuted_claim": "one-line prose of the refuted claim/join",
      "refuted_artifacts": ["scaling_ladder_L19.json", "mlp_scaling_L19.json"],
      "evidence": ["epm:progress 2026-08-25T03:18:29Z on #1901"],
      "replacement_artifacts": ["mlp_scaling_dense_L19.json"],
      "consumers_at_record_time": ["docs/paper_context_answer_map/claims.md"],
      "producers": ["scripts/issue1901_mlpdense_run.py"],
      "acknowledged_pending": [],
      "label_patterns": ["superseded", "different eval pool"]
    }

Audit algorithm (per record):

1. Validate the schema. Malformed / unrecognized / non-UTF-8 record => one
   WARN line naming the record + the missing/wrong field, plus a migration
   recipe pointer to this docstring (``--strict``: FAIL instead); the
   record's consumers are not audited.
2. Enumerate consumers FRESH: per refuted name,
   ``git grep --cached -l -F <name> -- . ':!tasks' ':!eval_results'
   ':!ood_eval_results' ':!figures'`` (grep-time pathspec excludes keep the
   audit seconds-scale; a full-tree grep measured ~15 s/name on the shared
   VM). Union the hits.
3. Exclude (recorded in the JSON report — TREE-CLASS exclusions at class
   level; per-file excluded-with-class recording for producers/records/self
   only): ``tasks/**`` (historical record; ``--include-tasks`` opts in),
   ``eval_results/**`` + ``ood_eval_results/**`` (data trees; also excludes
   every supersession record), ``figures/**`` (regenerated outputs — the
   GENERATOR script is the consumer that matters), ``producers`` paths,
   any supersession-record file itself, and the refuted/replacement artifact
   files themselves (index-path suffix match on the artifact names).
4. Per surviving consumer, PASS iff (a) the file mentions ANY
   ``replacement_artifacts`` name (fixed-string, file-level), OR (b) EVERY
   line mentioning a refuted name has a label pattern within +/-5 lines
   (case-insensitive). A consumer listed in ``acknowledged_pending`` => WARN
   instead of FAIL. Else one violation per unlabeled mention:
   ``FAIL: <record>: consumer <file>:<line> mentions <name> with no
   supersession label and does not consume <replacement>``.
5. Exit 0 when clean (WARNs allowed), 1 on >= 1 violation, 2 on usage error.

Disclosed false-negative / false-PASS modes (the reviewer/analyzer remains
the catching arm for all six; the #2165 scanner-disclosure norm):

1. FALSE NEGATIVE — a consumer referencing a refuted artifact via a
   variable, glob, f-string fragment, or wrapped/split literal escapes the
   fixed-string grep entirely.
2. FALSE POSITIVE — a genuine supersession label placed farther than +/-5
   lines from the mention reads as unlabeled; fix by moving the label next
   to the mention or by consuming the replacement artifact.
3. FALSE PASS — the file-level replacement-consumption OR-arm (a) PASSes a
   claims registry that ADDS a replacement-citing row while keeping the
   refuted row unlabeled — exactly the founding incident's surface at the
   row grain.
4. FALSE PASS — a "superseded" occurrence within +/-5 lines that refers to
   a DIFFERENT claim satisfies arm (b).
5. FALSE NEGATIVE — a genuine consumer whose OWN basename matches
   ``superseded_*.json`` OUTSIDE ``eval_results/`` is excluded as a
   "record" (``_is_record_path`` applies to every enumerated index path,
   not only ``eval_results/`` paths) and is never audited.
6. FALSE NEGATIVE — a consumer whose index path suffix-matches a
   refuted/replacement artifact NAME is excluded as the artifact file
   itself (``_matches_artifact_name`` is an index-path suffix match on
   BASENAMES) — an UNRELATED same-basename consumer elsewhere in the tree
   is never audited.

Dependencies: stdlib + subprocess git only — no ``explore_persona_space``
imports, no dotenv/HF (keeps this script outside the scripts-import-guard /
dotenv lint families). Lint wiring: ``scripts/workflow_lint.py``
``--check-artifact-supersession`` loads this module ``__file__``-relative and
maps violations to FAIL lines, schema problems + ``acknowledged_pending``
consumers to ``WARN:`` lines.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = "artifact-supersession-record-v1"
RECORD_BASENAME_GLOB = "superseded_*.json"
RECORD_DIR_GLOB = "superseded_*"
DEFAULT_LABEL_PATTERNS: tuple[str, ...] = (
    "superseded",
    "different eval pool",
    "different-eval-pool",
)
LABEL_WINDOW_LINES = 5
MIN_ARTIFACT_NAME_LEN = 8
# Tree-class pathspec excludes, applied at git-grep time (class-level
# exclusion recording). tasks/** is droppable via --include-tasks.
TREE_CLASS_EXCLUDES: tuple[str, ...] = ("tasks", "eval_results", "ood_eval_results", "figures")
MIGRATION_HINT = (
    "migration recipe: update the record to the artifact-supersession-record-v1 "
    "shape documented in scripts/audit_artifact_supersession.py's docstring"
)


class UsageError(Exception):
    """A caller-input problem (bad --record path, not a git repo) -> exit 2."""


@dataclass
class AuditReport:
    """Machine-readable audit outcome across all audited records."""

    records: list[dict] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        return {
            "schema": "artifact-supersession-audit-report-v1",
            "records": self.records,
            "violations": self.violations,
            "warnings": self.warnings,
        }


def _git(repo_root: Path, *args: str, ok_returncodes: tuple[int, ...] = (0,)) -> tuple[int, str]:
    """Run git in *repo_root*; return (rc, stdout). Unexpected rc raises."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode not in ok_returncodes:
        raise UsageError(
            f"git {' '.join(args[:2])}... failed (rc={proc.returncode}) in {repo_root}: "
            f"{proc.stderr.strip()}"
        )
    return proc.returncode, proc.stdout


def _is_record_path(path: str) -> bool:
    """Basename-anchored record predicate: basename matches
    ``superseded_*.json`` AND no parent path component is a ``superseded_*``
    directory (contents of such directories are NOT records)."""
    parts = path.split("/")
    if not fnmatch.fnmatch(parts[-1], RECORD_BASENAME_GLOB):
        return False
    return not any(fnmatch.fnmatch(comp, RECORD_DIR_GLOB) for comp in parts[:-1])


def discover_records(repo_root: Path) -> list[str]:
    """Index-listed record paths under eval_results/ (sorted, repo-relative)."""
    _rc, out = _git(repo_root, "ls-files", "-z", "--", "eval_results/")
    paths = [p for p in out.split("\0") if p]
    return sorted(p for p in paths if _is_record_path(p))


def _read_index_blob_bytes(repo_root: Path, path: str) -> bytes | None:
    """Stage-0 index BYTES of *path* (checkout-independent); None only when
    *path* is absent from the index. Record reads use this form so
    index-absence (a ``--record`` usage error) and a UTF-8 decode failure
    (a malformed record) stay distinguishable (#2568 round 2)."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "blob", f":{path}"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _read_index_blob(repo_root: Path, path: str) -> str | None:
    """Stage-0 index content of *path* decoded as UTF-8; None if absent from
    the index OR not decodable (consumer-read convenience — a binary/non-UTF-8
    consumer is WARNed and skipped, so the two states may stay conflated
    here)."""
    raw = _read_index_blob_bytes(repo_root, path)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _degenerate_token_reason(value: str) -> str | None:
    """Why *value* is unusable as a matching token (None == usable).

    A degenerate token silently inverts the audit (#2568 round 2): an empty
    replacement string is a substring of EVERY file (universal
    ``pass-replacement``), and a whitespace-only name/pattern matches nearly
    any indented line. The schema carries FILENAMES / short label phrases,
    so path separators and control characters are rejected too.
    """
    if not value.strip():
        return "empty or whitespace-only"
    if "/" in value or "\\" in value:
        return "contains a path separator"
    if any(ord(ch) < 32 or 127 <= ord(ch) <= 159 for ch in value):
        return "contains a control character"
    return None


def validate_record(data: object) -> list[str]:
    """Schema problems for one parsed record (empty list == valid)."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return [f"record is not a JSON object ({MIGRATION_HINT})"]
    schema = data.get("schema")
    if schema != SCHEMA_VERSION:
        problems.append(f"schema: expected {SCHEMA_VERSION!r}, got {schema!r} ({MIGRATION_HINT})")
    issue_val = data.get("issue")
    # bool subclasses int — reject it explicitly (Claude review round 1).
    if not isinstance(issue_val, int) or isinstance(issue_val, bool):
        problems.append("issue: required int is missing or non-int")
    claim = data.get("refuted_claim")
    if not isinstance(claim, str) or not claim.strip():
        problems.append("refuted_claim: required non-empty string is missing")
    refuted = data.get("refuted_artifacts")
    if not isinstance(refuted, list) or not refuted or not all(isinstance(n, str) for n in refuted):
        problems.append("refuted_artifacts: required non-empty list of strings is missing")
    else:
        for name in refuted:
            reason = _degenerate_token_reason(name)
            if reason:
                problems.append(
                    f"refuted_artifacts: degenerate name {name!r} ({reason}) — it would "
                    f"grep-match nothing or everything"
                )
        # The >= 8 floor counts NON-WHITESPACE chars: eight spaces must not
        # satisfy it (#2568 round 2).
        short = [
            n for n in refuted if sum(1 for ch in n if not ch.isspace()) < MIN_ARTIFACT_NAME_LEN
        ]
        if short:
            problems.append(
                f"refuted_artifacts: names with fewer than {MIN_ARTIFACT_NAME_LEN} "
                f"non-whitespace chars make the fixed-string grep useless: {short}"
            )
    evidence = data.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(e, str) for e in evidence)
    ):
        problems.append("evidence: required non-empty list of strings is missing")
    replacements = data.get("replacement_artifacts")
    if not isinstance(replacements, list) or not all(isinstance(n, str) for n in replacements):
        # MAY be empty (no replacement exists -> only the label arm can pass),
        # but it must be present as a list of strings.
        problems.append("replacement_artifacts: required list of strings is missing")
    else:
        for name in replacements:
            reason = _degenerate_token_reason(name)
            if reason:
                problems.append(
                    f"replacement_artifacts: degenerate name {name!r} ({reason}) — an "
                    f"empty/blank entry would mark EVERY consumer pass-replacement"
                )
    for optional in ("consumers_at_record_time", "producers", "acknowledged_pending"):
        val = data.get(optional)
        if val is not None and (
            not isinstance(val, list) or not all(isinstance(x, str) for x in val)
        ):
            problems.append(f"{optional}: optional field must be a list of strings")
    patterns = data.get("label_patterns")
    if patterns is not None and (
        not isinstance(patterns, list)
        or not patterns
        or not all(isinstance(p, str) and p for p in patterns)
    ):
        problems.append("label_patterns: optional field must be a non-empty list of strings")
    elif patterns is not None:
        for pat in patterns:
            reason = _degenerate_token_reason(pat)
            if reason:
                problems.append(
                    f"label_patterns: degenerate pattern {pat!r} ({reason}) — a blank "
                    f"pattern would label-match any window"
                )
    return problems


def _grep_cached_files(repo_root: Path, needle: str, *, include_tasks: bool) -> list[str]:
    """Index-tracked files whose CONTENT contains *needle* (fixed-string),
    with the tree-class excludes applied at grep time."""
    excludes = [t for t in TREE_CLASS_EXCLUDES if not (include_tasks and t == "tasks")]
    args = ["grep", "--cached", "-l", "-z", "-F", "-e", needle, "--", "."]
    args += [f":!{t}" for t in excludes]
    rc, out = _git(repo_root, *args, ok_returncodes=(0, 1))
    if rc == 1:
        return []
    return [p for p in out.split("\0") if p]


def _matches_artifact_name(path: str, names: list[str]) -> bool:
    """Index-path suffix match: *path* IS one of the artifact files."""
    return any(path == n or path.endswith("/" + n) for n in names)


def _unlabeled_mentions(
    content: str, refuted: list[str], label_patterns: list[str]
) -> list[tuple[int, str]]:
    """(1-based line, refuted name) pairs whose +/-LABEL_WINDOW_LINES window
    carries NO label pattern (case-insensitive substring)."""
    lines = content.splitlines()
    lower = [ln.lower() for ln in lines]
    pats = [p.lower() for p in label_patterns]
    out: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        hit_names = [n for n in refuted if n in line]
        if not hit_names:
            continue
        lo = max(0, i - LABEL_WINDOW_LINES)
        hi = min(len(lines), i + LABEL_WINDOW_LINES + 1)
        window = lower[lo:hi]
        if any(p in wline for p in pats for wline in window):
            continue
        for name in hit_names:
            out.append((i + 1, name))
    return out


def _audit_one_record(
    repo_root: Path,
    record_path: str,
    *,
    include_tasks: bool,
    strict: bool,
    all_record_paths: set[str],
    report: AuditReport,
) -> None:
    """Audit one record; append its outcome to *report* (mutates it)."""
    rec_entry: dict = {
        "path": record_path,
        "schema_ok": False,
        "schema_problems": [],
        "consumers": {},
        "excluded": {"tree_classes": [], "files": {}},
    }
    report.records.append(rec_entry)
    raw_bytes = _read_index_blob_bytes(repo_root, record_path)
    if raw_bytes is None:
        # Index-ABSENT: only reachable via --record misuse (discovery is
        # `git ls-files`, so discovered records are in the index by
        # construction) — a caller-input problem, exit 2.
        raise UsageError(
            f"record {record_path} is not in the git index — `git add` it first "
            "(all reads are index-based; sparse worktrees lack eval_results/ on disk)"
        )
    try:
        raw: str | None = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Index-PRESENT but undecodable: a MALFORMED record (WARN; --strict:
        # FAIL), never a UsageError — a tracked non-UTF-8 record must not
        # hard-block the fleet no-flags lint (#2568 round 2).
        raw = None
        problems = [f"record bytes are not decodable as UTF-8: {exc} ({MIGRATION_HINT})"]
        data: object | None = None
    if raw is not None:
        try:
            data = json.loads(raw)
            problems = validate_record(data)
        except json.JSONDecodeError as exc:
            problems = [f"unparseable JSON: {exc} ({MIGRATION_HINT})"]
            data = None
    if problems:
        rec_entry["schema_problems"] = problems
        for prob in problems:
            msg = f"{record_path}: {prob}"
            if strict:
                report.violations.append(msg)
            else:
                report.warnings.append(msg)
        return
    assert isinstance(data, dict)
    rec_entry["schema_ok"] = True

    refuted: list[str] = data["refuted_artifacts"]
    replacements: list[str] = data["replacement_artifacts"]
    producers = set(data.get("producers") or [])
    acknowledged = set(data.get("acknowledged_pending") or [])
    label_patterns: list[str] = list(data.get("label_patterns") or DEFAULT_LABEL_PATTERNS)
    excludes_applied = [t for t in TREE_CLASS_EXCLUDES if not (include_tasks and t == "tasks")]
    rec_entry["excluded"]["tree_classes"] = excludes_applied
    rec_entry["refuted_artifacts"] = refuted
    rec_entry["replacement_artifacts"] = replacements

    hits: set[str] = set()
    for name in refuted:
        hits.update(_grep_cached_files(repo_root, name, include_tasks=include_tasks))

    consumers: list[str] = []
    for path in sorted(hits):
        if path in producers:
            rec_entry["excluded"]["files"][path] = "producer"
        elif path in all_record_paths or _is_record_path(path) or path == record_path:
            rec_entry["excluded"]["files"][path] = "record"
        elif _matches_artifact_name(path, refuted + replacements):
            rec_entry["excluded"]["files"][path] = "self"
        else:
            consumers.append(path)

    replacement_desc = ", ".join(replacements) if replacements else "any replacement"
    for path in consumers:
        content = _read_index_blob(repo_root, path)
        if content is None:
            report.warnings.append(
                f"{record_path}: consumer {path} unreadable from the git index "
                "(binary or non-UTF-8) — not audited"
            )
            rec_entry["consumers"][path] = {"status": "unreadable", "violations": []}
            continue
        if replacements and any(r in content for r in replacements):
            rec_entry["consumers"][path] = {"status": "pass-replacement", "violations": []}
            continue
        unlabeled = _unlabeled_mentions(content, refuted, label_patterns)
        if not unlabeled:
            rec_entry["consumers"][path] = {"status": "pass-labeled", "violations": []}
            continue
        msgs = [
            (
                f"{record_path}: consumer {path}:{lineno} mentions {name} with no "
                f"supersession label and does not consume {replacement_desc}"
                + ("" if replacements else " (record declares no replacement artifacts)")
            )
            for lineno, name in unlabeled
        ]
        if path in acknowledged:
            rec_entry["consumers"][path] = {
                "status": "acknowledged_pending",
                "violations": msgs,
            }
            report.warnings.append(
                f"{record_path}: consumer {path} is acknowledged_pending with "
                f"{len(msgs)} unlabeled mention(s) — conformance routed but not landed"
            )
        else:
            rec_entry["consumers"][path] = {"status": "violation", "violations": msgs}
            report.violations.extend(msgs)


def run_audit(
    repo_root: Path,
    *,
    record: str | None = None,
    strict: bool = False,
    include_tasks: bool = False,
) -> AuditReport:
    """Audit all discovered records (or one explicit *record*) in *repo_root*.

    Returns an :class:`AuditReport`; raises :class:`UsageError` on caller-
    input problems (exit-2 class). Violations non-empty <=> exit 1.
    """
    repo_root = Path(repo_root)
    all_records = set(discover_records(repo_root))
    if record is not None:
        rec_path = Path(record)
        if rec_path.is_absolute():
            try:
                record = rec_path.resolve().relative_to(repo_root.resolve()).as_posix()
            except ValueError as exc:
                raise UsageError(f"--record {record} is outside repo root {repo_root}") from exc
        else:
            record = rec_path.as_posix()
        targets = [record]
        all_records.add(record)
    else:
        targets = sorted(all_records)
    report = AuditReport()
    for rec in targets:
        _audit_one_record(
            repo_root,
            rec,
            include_tasks=include_tasks,
            strict=strict,
            all_record_paths=all_records,
            report=report,
        )
    return report


def _detect_repo_root() -> Path:
    proc = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise UsageError("not inside a git repository (pass --repo-root)")
    return Path(proc.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper: 0 clean (WARNs allowed) / 1 violations / 2 usage error."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--record",
        help="audit ONLY this record (repo-relative or absolute path; must be in the git index)",
    )
    ap.add_argument("--json", type=Path, help="write the machine-readable report to this path")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="malformed/unrecognized records FAIL (default: WARN)",
    )
    ap.add_argument(
        "--include-tasks",
        action="store_true",
        help="include tasks/** in the consumer enumeration (excluded by default)",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repo root to audit (default: git rev-parse --show-toplevel from cwd)",
    )
    args = ap.parse_args(argv)
    try:
        repo_root = args.repo_root if args.repo_root is not None else _detect_repo_root()
        report = run_audit(
            repo_root,
            record=args.record,
            strict=args.strict,
            include_tasks=args.include_tasks,
        )
    except UsageError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    if args.json is not None:
        args.json.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    for warning in report.warnings:
        print(f"WARN: {warning}")
    for violation in report.violations:
        print(f"FAIL: {violation}")
    n_rec = len(report.records)
    print(
        f"audit_artifact_supersession: {n_rec} record(s), "
        f"{len(report.violations)} violation(s), {len(report.warnings)} warning(s)"
    )
    return 1 if report.violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
