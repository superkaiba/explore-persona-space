"""Issue #2658 unit 4b — objective (non-judge) labels for the correctness rows.

Labels generated answers for ``correctness_math`` / ``correctness_mmlu_pro`` /
``correctness_code`` with the REUSED #2388 checkers (vendored, byte-sha-pinned;
loaded via ``issue2658_text_resolver.load_pinned_gen_module``):

- ``correctness_math``     -> ``verify_math``   (vendored issue2388_spread_pilot;
  ``\\boxed`` extraction + math-verify parse/verify, normalized-string fallback)
- ``correctness_mmlu_pro`` -> ``verify_mcq``    (vendored issue2388_spread_pilot;
  frozen answer-key letter match)
- ``correctness_code``     -> the vendored issue2388_gen sandboxed verifiers,
  dispatched exactly as the pinned ``_verdict_one``: ``lcb_v5 -> _verify_lcb``,
  ``leetcode -> _verify_leetcode``, other code benchmarks -> ``_verify_pilot_code``
  (frozen executable tests in the #2388 sandbox: scrubbed env, own process
  group, CPU/FSIZE/AS rlimits, ``unshare -rn`` network isolation, hard per-test
  timeout ``CODE_EXEC_TIMEOUT_S``).

These rows are NOT judge-scored: ``C.judge_instrument_fingerprint(row)`` raising
for them is the contract, asserted at cell entry — no LLM is ever called here.

Timeout semantics (unit-4 brief): the pinned ``_run_code`` collapses a sandbox
timeout (``_run_sandboxed`` -> ``(-1, "")``) to ``False`` — a coerced
"incorrect".  ``install_timeout_seam`` rebinds ``gen._run_sandboxed`` on the
LOADED module object (pinned bytes untouched) so a timeout RAISES
``SandboxTimeoutError`` instead; the label driver counts it as a
``harness_failure`` exclusion after bounded retries — never a label.

Missingness taxonomy (per row, per cell; sums to the denominator, asserted):

- ``labeled``                 checker returned a bool.
- ``malformed``               checker returned None on an item with an available
                              reference (no ``\\boxed`` / answer letter / code
                              block) — routed to human adjudication, never
                              coerced to incorrect (plan §3).
- ``harness_failure``         sandbox/infra exception (timeout, OSError,
                              SubprocessError) persisting through retries —
                              loud, counted, never a label.
- ``genuinely_unavailable``   reference-side gap (pinned semantics: an lcb_v5
                              item with functional tests but no func_name) —
                              counted exclusion, checker never invoked.

CONTENT HYGIENE: completion/prompt text flows disk -> checker -> sandbox; logs
and reports carry only ids, counts, statuses, and sha256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_common as C  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_generate as G  # noqa: E402
import issue2658_text_resolver as R  # noqa: E402

LABEL_SCHEMA = "i2658-objective-labels-v1"
CORRECTNESS_ROWS = ("correctness_math", "correctness_mmlu_pro", "correctness_code")
STATUSES = ("labeled", "malformed", "harness_failure", "genuinely_unavailable")
HARNESS_RETRIES = 1  # transport-style retries for sandbox infra failures (plan §3)


class SandboxTimeoutError(C.Issue2658GuardError):
    """A sandboxed test exceeded its hard per-test timeout — RAISED, never a label."""


class LabelJoinError(C.Issue2658GuardError):
    """Manifest/raw-completion join violated the 1:1 + sha contract."""


# Exceptions treated as harness (infra) failures — retried, then counted.
_HARNESS_EXCEPTIONS = (SandboxTimeoutError, OSError, subprocess.SubprocessError)


# ---------------------------------------------------------------------------
# Timeout-raising seam over the pinned sandbox runner.
# ---------------------------------------------------------------------------
def install_timeout_seam(gen: Any) -> None:
    """Rebind ``gen._run_sandboxed`` so a timeout RAISES instead of (-1, "").

    The pinned bytes stay untouched (vendor pins verified at load); only the
    loaded module object's global is wrapped, so every reused verifier
    (``_run_code``, ``_run_stdin_tests``) resolves the raising wrapper.
    Idempotent per module object.
    """
    if getattr(gen, "_i2658_timeout_seam", False):
        return
    orig = gen._run_sandboxed

    def _run_sandboxed_raising(
        argv: list[str],
        *,
        timeout_s: int,
        tmpdir: str,
        input_text: str | None = None,
    ) -> tuple[int, str]:
        rc, out = orig(argv, timeout_s=timeout_s, tmpdir=tmpdir, input_text=input_text)
        if rc == -1 and out == "":
            raise SandboxTimeoutError(
                f"sandboxed test exceeded timeout_s={timeout_s} (pinned _run_sandboxed "
                "returned (-1, '')) — a timeout is a harness failure, never 'incorrect'"
            )
        return rc, out

    gen._run_sandboxed = _run_sandboxed_raising
    gen._i2658_timeout_seam = True


# ---------------------------------------------------------------------------
# Checker dispatch (mirrors the pinned _verdict_one exactly).
# ---------------------------------------------------------------------------
def checker_for(gen: Any, benchmark: str) -> tuple[str, Callable[[str, dict], bool | None]]:
    """(checker_name, callable) for one benchmark — the pinned dispatch chain."""
    if benchmark == "math_full":
        return "verify_math", lambda comp, item: gen.verify_math(comp, item)
    if benchmark == "mmlu_pro_full":
        return "verify_mcq", lambda comp, item: gen.verify_mcq(comp, item)
    if benchmark == "lcb_v5":
        return "_verify_lcb", lambda comp, item: gen._verify_lcb(comp, item)
    if benchmark == "leetcode":
        return "_verify_leetcode", lambda comp, item: gen._verify_leetcode(comp, item)
    if benchmark == "apps_intro":
        return "_verify_apps", lambda comp, item: gen._verify_apps(comp, item)
    if benchmark in gen.CODE_BENCHMARKS:
        return "_verify_pilot_code", lambda comp, item: gen._verify_pilot_code(comp, item)
    raise C.MissingLabelError(f"no objective checker for benchmark {benchmark!r}")


_SPREAD_PILOT_CHECKERS = ("verify_math", "verify_mcq")


def checker_source(name: str) -> dict[str, str]:
    """Vendored-pin provenance for one checker (module + byte sha of the pin)."""
    vend = R.verify_vendor_pins()
    key = "issue2388_spread_pilot" if name in _SPREAD_PILOT_CHECKERS else "issue2388_gen"
    path = vend[key]
    return {"vendored_module": path.name, "sha256": R._sha256_file(path)}


def reference_unavailable_reason(item: dict[str, Any]) -> str | None:
    """Item-level reference gap (checker never invoked) — pinned semantics."""
    if item.get("benchmark") == "lcb_v5":
        tests = item.get("tests") or []
        functional = [t for t in tests if t.get("testtype") == "functional"]
        if functional and not item.get("func_name"):
            return "lcb-functional-tests-without-func_name"
    return None


def _ref_sha(item: dict[str, Any]) -> str:
    """Content digest of the frozen reference row (sorted-keys compact JSON)."""
    payload = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Cell input loading (manifest + raw completions, strict 1:1 join).
# ---------------------------------------------------------------------------
def load_cell_inputs(
    manifest_path: Path, raw_path: Path
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(validated manifest row, raw record) pairs — 1:1 join, sha-verified."""
    if not manifest_path.exists() or not raw_path.exists():
        raise LabelJoinError(
            f"cell inputs missing: manifest={manifest_path} raw={raw_path} — generation "
            "(unit 3) has not produced this cell"
        )
    mrows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as fh:  # text-mode iteration, never splitlines()
        for line in fh:
            if line.strip():
                d = json.loads(line)
                C.validate_manifest_row(d)
                mrows.append(d)
    body = json.loads(raw_path.read_text())
    recs = {(r["prompt_id"], r["response_index"]): r for r in body["records"]}
    if len(recs) != len(body["records"]):
        raise LabelJoinError(f"duplicate (prompt_id, response_index) in {raw_path}")
    if len(mrows) != len(recs):
        raise LabelJoinError(
            f"manifest/raw row-count mismatch: {len(mrows)} manifest rows vs "
            f"{len(recs)} records for {raw_path.name}"
        )
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for m in mrows:
        key = (m["prompt_id"], m["response_index"])
        rec = recs.get(key)
        if rec is None:
            raise LabelJoinError(f"manifest row {key} has no raw record in {raw_path.name}")
        C.assert_row_hash(rec["text"], m["answer_sha256"])
        out.append((m, rec))
    return out


def load_references(gen: Any, item_ids: list[str]) -> dict[str, dict[str, Any]]:
    """context_id -> frozen loader row, via the pinned #2388 benchmark loaders."""
    by_attr: dict[str, list[str]] = {}
    for iid in item_ids:
        _row, _frame, context_id = R.parse_item_id(iid)
        by_attr.setdefault(R._bench_loader_attr(context_id), []).append(context_id)
    refs: dict[str, dict[str, Any]] = {}
    for attr, context_ids in sorted(by_attr.items()):
        rows = getattr(gen, attr)()  # pinned loader; count-asserts internally
        by_id = {r["item_id"]: r for r in rows}
        for cid in context_ids:
            if cid not in by_id:
                raise C.MissingLabelError(
                    f"context_id {cid!r} not in pinned loader {attr} pool ({len(by_id)} items)"
                )
            refs[cid] = by_id[cid]
    return refs


# ---------------------------------------------------------------------------
# Labeling.
# ---------------------------------------------------------------------------
def label_one(gen: Any, item: dict[str, Any], completion: str) -> dict[str, Any]:
    """One (reference item, completion) -> {status, label, checker, detail}."""
    unavailable = reference_unavailable_reason(item)
    if unavailable is not None:
        return {
            "status": "genuinely_unavailable",
            "label": None,
            "checker": None,
            "detail": unavailable,
        }
    name, fn = checker_for(gen, item["benchmark"])
    last_err: BaseException | None = None
    for _attempt in range(1 + HARNESS_RETRIES):
        try:
            verdict = fn(completion, item)
        except _HARNESS_EXCEPTIONS as e:  # loud + counted, never a label
            last_err = e
            continue
        if verdict is None:
            return {
                "status": "malformed",
                "label": None,
                "checker": name,
                "detail": "checker returned None (unparseable answer) — human adjudication",
            }
        return {"status": "labeled", "label": bool(verdict), "checker": name, "detail": None}
    assert last_err is not None
    reason = f"{type(last_err).__name__}: {last_err}"[:500]
    print(f"[labels] HARNESS FAILURE ({name}, retries exhausted): {reason}", flush=True)
    return {
        "status": "harness_failure",
        "label": None,
        "checker": name,
        "detail": reason,
    }


def missingness_report(cell_name: str, row: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-cell missingness accounting; the split MUST sum to the denominator."""
    n = len(records)
    counts = {s: sum(1 for r in records if r["status"] == s) for s in STATUSES}
    if sum(counts.values()) != n:
        raise C.MissingLabelError(
            f"cell {cell_name}: missingness classes sum to {sum(counts.values())} != "
            f"denominator {n} — a row escaped the taxonomy"
        )
    labeled = [r for r in records if r["status"] == "labeled"]
    return {
        "schema": LABEL_SCHEMA,
        "cell": cell_name,
        "row": row,
        "n_rows": n,
        "n_labeled": counts["labeled"],
        "n_malformed": counts["malformed"],
        "n_harness_failure": counts["harness_failure"],
        "n_genuinely_unavailable": counts["genuinely_unavailable"],
        "n_correct": sum(1 for r in labeled if r["label"] is True),
        "n_incorrect": sum(1 for r in labeled if r["label"] is False),
        "sums_to_denominator": True,
    }


def out_paths(out_root: Path, split: str, cell_name: str) -> tuple[Path, Path]:
    d = out_root / "objective_labels" / split
    return d / f"{cell_name}.jsonl", d / f"{cell_name}.report.json"


def _input_fingerprint(manifest_path: Path, raw_path: Path) -> str:
    """Resume key: byte digests of the on-disk inputs (machine-stable)."""
    h = hashlib.sha256()
    h.update(LABEL_SCHEMA.encode())
    for p in (manifest_path, raw_path):
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def run_cell(
    gen: Any,
    cell: G.CellWork,
    split: str,
    out_root: Path,
    refs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Label one cell; checkpointed per cell with fingerprint-gated resume."""
    raw_path, manifest_path = G.out_paths(out_root, split, cell.name)
    labels_path, report_path = out_paths(out_root, split, cell.name)
    fingerprint = _input_fingerprint(manifest_path, raw_path)
    if labels_path.exists() and report_path.exists():
        prior = json.loads(report_path.read_text())
        if prior.get("input_fingerprint") == fingerprint:
            print(f"[labels] resume: {cell.name} already labeled (fingerprint match)")
            return prior
        raise C.CacheStaleError(
            f"cell {cell.name}: existing labels were computed from different inputs "
            f"(stored fingerprint != {fingerprint[:12]}...) — refusing to silently mix"
        )

    pairs = load_cell_inputs(manifest_path, raw_path)
    records: list[dict[str, Any]] = []
    for i, (m, rec) in enumerate(pairs):
        if m["judge_status"] != "objective":
            raise C.CoercedLabelError(
                f"manifest row {m['prompt_id']}#{m['response_index']} has judge_status "
                f"{m['judge_status']!r}; objective labeling accepts only 'objective' rows"
            )
        _row, _frame, context_id = R.parse_item_id(m["prompt_id"])
        item = refs[context_id]
        verdict = label_one(gen, item, rec["text"])
        records.append(
            {
                "schema": LABEL_SCHEMA,
                "manifest": m,
                "label": verdict["label"],
                "status": verdict["status"],
                "provenance": {
                    "checker": verdict["checker"],
                    "checker_source": (
                        checker_source(verdict["checker"]) if verdict["checker"] else None
                    ),
                    "benchmark": item.get("benchmark"),
                    "reference_ref": context_id,
                    "reference_sha256": _ref_sha(item),
                    "sandbox": (
                        {
                            "timeout_s": gen.CODE_EXEC_TIMEOUT_S,
                            "net_isolation": bool(gen._unshare_net_available()),
                        }
                        if _row == "correctness_code"
                        else None
                    ),
                    "detail": verdict["detail"],
                },
            }
        )
        print(
            f"[labels] unit {i + 1}/{len(pairs)} {m['prompt_id']}#{m['response_index']} "
            f"status={verdict['status']}",
            flush=True,
        )

    report = missingness_report(cell.name, cell.row, records)
    report["split"] = split
    report["input_fingerprint"] = fingerprint
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    report["metadata"] = as_metadata_dict(git_provenance(), phase="objective-labels")

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = labels_path.with_name(labels_path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    tmp.replace(labels_path)
    rtmp = report_path.with_name(report_path.name + ".tmp")
    rtmp.write_text(G.canonical_json(report))
    rtmp.replace(report_path)
    print(
        f"[labels] {cell.name}: labeled={report['n_labeled']} malformed={report['n_malformed']} "
        f"harness={report['n_harness_failure']} unavailable={report['n_genuinely_unavailable']} "
        f"of {report['n_rows']}",
        flush=True,
    )
    return report


def run(args: argparse.Namespace) -> int:
    rows = list(args.rows)
    for row in rows:
        if row not in CORRECTNESS_ROWS:
            raise C.MissingLabelError(f"row {row!r} is not an objective-label row")
        construct = C.CONSTRUCTS[row]
        if construct.judge_scored:
            raise C.CoercedLabelError(f"row {row!r} is judge-scored; objective labeler refuses")
        try:
            C.judge_instrument_fingerprint(row)
        except ValueError:
            pass  # the contract: objective rows have NO judge instrument
        else:
            raise C.CoercedLabelError(
                f"judge_instrument_fingerprint({row!r}) did not raise — the no-judge "
                "contract for objective rows is broken"
            )

    gen = R.load_pinned_gen_module()
    install_timeout_seam(gen)
    gen._require_sandbox_net_isolation()  # fail-loud before any cell spends

    cells = [c for c in G.build_cells(rows_filter=rows)]
    out_root = args.out_root
    item_ids = sorted({iid for c in cells for iid in c.item_ids})
    refs = load_references(gen, item_ids)
    reports = []
    for cell in cells:
        reports.append(run_cell(gen, cell, args.split, out_root, refs))
    total = sum(r["n_rows"] for r in reports)
    labeled = sum(r["n_labeled"] for r in reports)
    print(f"[labels] done: {len(reports)} cells, {labeled}/{total} rows labeled")
    print("[phase=done]", flush=True)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=C.SPLITS, default="pilot")
    ap.add_argument("--rows", nargs="+", default=list(CORRECTNESS_ROWS))
    ap.add_argument("--out-root", type=Path, default=F.OUT_DIR)
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
