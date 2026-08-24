"""Before/after c26+c27 corpus sweep for the #2514 routed-machine mirror rebuild.

Two subcommands:

``sweep``
    Runs the SHIPPED ``check_gpu_basis_routed_machine`` (c26) and
    ``check_capture_intent_hbm`` (c27) — imported from a given
    ``verify_plan.py`` path, never a re-implementation (the #2276
    ``issue2276_c62c63_corpus_sweep.py`` convention) — over every
    ``tasks/*/*/plans/v*.md`` under ``--repo-root``, writing one JSON row
    per plan (c26/c27 status + detail head + resolved intents + the c26
    basis-row token meta the classifier replays) to ``--out`` (JSONL;
    header row first). Checkpoint-per-unit: each row is flushed the moment
    its plan completes, with one stdout line per completed unit. Re-runnable:
    when ``--out`` exists and its header matches the current regime
    (``verify_plan_path`` / ``module_sha`` — sha256 of the swept module's
    file bytes, round-3 reconciler item 2 — / ``mirror`` / ``lane_head`` /
    ``under_hbm_intents``), prior rows are KEPT (rewritten compacted,
    dropping any truncated tail a killed run left) and completed plans are
    SKIPPED; a header-regime mismatch restarts from scratch. Point
    ``--verify-plan-path`` at a ``git show <ref>:scripts/verify_plan.py``
    materialization for the BEFORE leg and at the live
    ``scripts/verify_plan.py`` for the AFTER leg, and pass
    ``--source-ref '<40-hex sha>:scripts/verify_plan.py'`` so the blob's git
    provenance is recorded in the header (annotation only — ``module_sha``
    is the mechanical content key; round-3 reconciler item 3).

``classify``
    FIRST refuses (SystemExit) any leg whose header regime (``mirror`` /
    ``lane_head`` / ``under_hbm_intents``) differs from the APPROVED #2514
    pins (``_APPROVED_LEG_REGIMES``; round-3 reconciler item 1a): the
    directional replay below is self-consistent under ANY mirror, approved
    or not, so without this anchor a wrong-but-self-consistent remap (a
    B200 substitution, an empty mirror from total capture loss) would
    self-certify. Then diffs the before/after JSONL pair and buckets EVERY
    c26/c27 verdict flip into the #2514 plan taxonomy:
    ``expected-inversion`` (DIRECTIONAL — the realized before AND after
    verdicts both EQUAL the verdict predicted by replaying the c26 offender
    rule over the row's recorded basis-row tokens under the old/new mirror
    respectively (round-2 reconciler item 2 — the round-1
    ``fams_old != fams_new`` read was true by construction under a
    wholesale family remap), AND the realized transition is itself an
    inversion — ``(WARN,PASS)`` or ``(PASS,WARN)`` per the plan's
    registered taxonomy; round-3 reconciler item 1b), ``c27-disarm`` (a c27
    FAIL/WARN — or a downstream no->=7B-signal SKIP — that becomes the D3
    empty-under-floor PASS), ``new-key-arming`` (a c26 SKIP that now
    resolves because the plan books a key new to the mirror, e.g. inf-70b /
    ft-70b), and ``unexplained`` (anything else, incl. a flip whose
    direction the mirror does NOT predict and any non-inversion transition,
    ``WARN->SKIP`` included). A non-empty ``unexplained`` set
    exits 1 — the plan's KILL criterion; never baseline it away.

Every file is verified with ``kind="experiment"`` uniformly (the #1395
``issue1395_corpus_audit.py`` convention): the sweep is a label DIFF of two
module versions on identical inputs, so the kind choice cancels out, and
fire counts UPPER-BOUND production (kind-exempt plans SKIP there).

Usage::

    uv run python scripts/issue2514_c26c27_corpus_sweep.py sweep \
        --verify-plan-path /tmp/i2514_verify_plan_before.py \
        --out /tmp/issue2514_corpus_before.json
    uv run python scripts/issue2514_c26c27_corpus_sweep.py sweep \
        --out /tmp/issue2514_corpus_after.json
    uv run python scripts/issue2514_c26c27_corpus_sweep.py classify \
        --before /tmp/issue2514_corpus_before.json \
        --after /tmp/issue2514_corpus_after.json \
        --out /tmp/issue2514_corpus_diff.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]


def _load_verify_plan(path: Path, repo_root: Path):
    """Import a verify_plan module from an explicit path (unique module name
    so a before/after pair can coexist; repo src/ goes on sys.path first so
    the module's stdlib-only ``plan_wall_budget`` shim resolves even for a
    /tmp-materialized BEFORE copy)."""
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    name = f"verify_plan_i2514_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: Header keys that define a sweep's regime — a resume is legal only when
#: ALL of them match the existing file's header (round-2 reconciler item 3).
#: ``module_sha`` (round-3 reconciler item 2) keys the resume to the swept
#: module's CONTENT: same-path edits to scripts/verify_plan.py are this
#: tool's normal workflow, so a path match alone is not a regime match.
_HEADER_REGIME_KEYS = (
    "verify_plan_path",
    "module_sha",
    "mirror",
    "lane_head",
    "under_hbm_intents",
)


def _c26_row_meta(mod, text: str) -> list[dict]:
    """The c26 offender-rule tokens per compute-table row with a basis-cell
    GPU-family hit — extracted with the SWEPT module's OWN helpers (each leg
    records what its own check saw). ``classify`` replays the offender rule
    over these under either mirror to DERIVE the predicted verdict (the
    directional expected-inversion predicate, round-2 reconciler item 2)."""
    meta: list[dict] = []
    for _component, basis, wall, row_text in mod._c26_compute_table_rows(text):
        hit = mod._C26_BASIS_GPU_RE.search(basis)
        if not hit:
            continue
        conv = f"{basis} {wall}"
        meta.append(
            {
                "basis_family": mod._c26_family(hit.group(1)),
                "conv_families": sorted(
                    {mod._c26_family(m.group(1)) for m in mod._C26_ROW_GPU_ANY_RE.finditer(conv)}
                ),
                "scaling": bool(mod._C26_SCALING_RE.search(row_text)),
            }
        )
    return meta


def _resume_rows(out: Path, header: dict) -> dict[str, dict]:
    """Prior sweep rows to KEEP when resuming into ``out``: parseable rows
    (a truncated trailing line from a killed run is dropped) under a header
    whose regime keys ALL match the current run's. A missing file, a file
    with no parseable header, or a regime mismatch returns {} — start
    fresh (the mismatch case prints which keys differ)."""
    if not out.exists():
        return {}
    existing_header: dict | None = None
    rows: dict[str, dict] = {}
    with out.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                break  # truncated tail — rows before it still count
            if "header" in obj:
                existing_header = obj["header"]
            else:
                rows[obj["plan"]] = obj
    if existing_header is None:
        print(f"[sweep] {out} has no parseable header — restarting from scratch", flush=True)
        return {}
    mismatched = [k for k in _HEADER_REGIME_KEYS if existing_header.get(k) != header.get(k)]
    if mismatched:
        print(
            f"[sweep] {out} header regime mismatch on {mismatched} — restarting from scratch",
            flush=True,
        )
        return {}
    return rows


def cmd_sweep(args: argparse.Namespace) -> int:
    """One JSON row per plan, checkpointed per unit: every row is flushed the
    moment its plan completes, with one stdout progress line per completed
    unit. Re-runnable via ``_resume_rows``: prior same-regime rows are kept
    (rewritten compacted so a truncated tail can never corrupt the file) and
    their plans skipped; a header-regime mismatch restarts from scratch."""
    mod = _load_verify_plan(args.verify_plan_path, args.repo_root)
    plans = sorted(args.repo_root.glob("tasks/*/*/plans/v*.md"))
    under = getattr(mod, "_C27_UNDER_HBM_INTENTS", None)
    if under is None:  # the pre-#2514 module names the set by its L4 proxy
        under = getattr(mod, "_C27_L4_INTENTS", frozenset())
    header = {
        "verify_plan_path": str(args.verify_plan_path),
        "module_sha": hashlib.sha256(args.verify_plan_path.read_bytes()).hexdigest(),
        "source_ref": args.source_ref,
        "n_plans": len(plans),
        "mirror": dict(mod._C26_INTENT_GPU),
        "lane_head": getattr(mod, "_C26_LANE_HEAD", None),
        "under_hbm_intents": sorted(under),
    }
    plan_rels = {str(p.relative_to(args.repo_root)) for p in plans}
    done = {
        rel: row
        for rel, row in _resume_rows(args.out, header).items()
        # a row without c26_rows predates the directional-taxonomy schema
        # (round-1 sweep version) — re-sweep it rather than KeyError classify
        if rel in plan_rels and "c26_rows" in row
    }
    if done:
        print(f"[sweep] resuming {args.out}: {len(done)}/{len(plans)} plans done", flush=True)
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"header": header}) + "\n")
        for row in done.values():
            fh.write(json.dumps(row) + "\n")
        fh.flush()
        for i, path in enumerate(plans, 1):
            rel = str(path.relative_to(args.repo_root))
            if rel in done:
                continue
            text = path.read_text(errors="replace")
            r26 = mod.check_gpu_basis_routed_machine(text, "experiment")
            r27 = mod.check_capture_intent_hbm(text, "experiment")
            row = {
                "plan": rel,
                "intents": sorted(mod._c26_intents(text)),
                "c26": r26.status,
                "c26_detail": r26.detail[:160],
                "c27": r27.status,
                "c27_detail": r27.detail[:160],
                "c26_rows": _c26_row_meta(mod, text),
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"[sweep] unit {i}/{len(plans)} {path.name} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    print(f"[sweep] wrote {args.out} ({len(plans)} plans, {len(done)} resumed)", flush=True)
    return 0


def _load_rows(path: Path) -> tuple[dict, dict[str, dict]]:
    """(header, {plan_rel: row}) from one sweep JSONL."""
    header: dict = {}
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if "header" in obj:
                header = obj["header"]
            else:
                rows[obj["plan"]] = obj
    return header, rows


def _families(intents: list[str], mirror: dict[str, str]) -> frozenset[str]:
    return frozenset(mirror[i] for i in intents if i in mirror)


#: The APPROVED #2514 leg regimes (round-3 reconciler item 1a) — classifier
#: inputs INDEPENDENT of the swept modules. BEFORE is pinned from the
#: pre-#2514 blob ``3de240d59f:scripts/verify_plan.py`` (git blob
#: 4e576527df97, sha256 20504ba9093c…: the GCP-era static
#: ``_C26_INTENT_GPU`` literal, ``_C27_L4_INTENTS`` == {debug, eval}, and
#: no ``_C26_LANE_HEAD`` symbol — hence ``lane_head: None``). AFTER is
#: pinned from plan v4's approved "Resulting families under the runpod
#: head" table plus decision D2 (``eval-h100`` -> H100) and acceptance
#: criterion 4's empty under-HBM set. ``cmd_classify`` REFUSES a leg whose
#: header regime differs: ``_predicted_c26`` replays the offender rule
#: under whatever mirror the header carries, so it is self-consistent under
#: ANY mirror — without this pin a wrong-but-self-consistent remap (a B200
#: substitution, an empty mirror from total capture loss) would
#: self-certify as ``expected-inversion`` (the round-2 reconciler FAIL).
_APPROVED_LEG_REGIMES: dict[str, dict] = {
    "before": {
        "mirror": {
            "lora-7b": "A100",
            "lora": "A100",
            "capture-7b": "A100",
            "ft-7b": "A100",
            "eval": "L4",
            "debug": "L4",
            "lora-7b-h100": "H100",
            "eval-h100": "H100",
            "cpu-bigmem": "CPU",
            "cpu-small": "CPU",
            "cpu-mid": "CPU",
            "sweep-8g-a100": "A100",
            "sweep-8g-h100": "H100",
        },
        "lane_head": None,
        "under_hbm_intents": ["debug", "eval"],
    },
    "after": {
        "mirror": {
            "lora-7b": "H100",
            "lora": "H100",
            "capture-7b": "H100",
            "ft-7b": "H100",
            "eval": "H100",
            "debug": "H100",
            "lora-7b-h100": "H100",
            "eval-h100": "H100",
            "cpu-bigmem": "CPU",
            "cpu-small": "CPU",
            "cpu-mid": "CPU",
            "sweep-8g-a100": "A100",
            "sweep-8g-h100": "H100",
            "inf-70b": "H100",
            "ft-70b": "H200",
        },
        "lane_head": "runpod",
        "under_hbm_intents": [],
    },
}


def _assert_approved_regime(leg: str, header: dict) -> None:
    """Refuse (SystemExit, non-zero) unless ``header``'s mirror / lane_head /
    under_hbm_intents equal the approved #2514 pin for ``leg`` — the round-3
    item-1a anchor that makes the kill criterion falsifiable: a leg swept
    under an unapproved regime fails loud BEFORE any bucketing."""
    approved = _APPROVED_LEG_REGIMES[leg]
    mismatched = [k for k, v in approved.items() if header.get(k) != v]
    if mismatched:
        detail = "; ".join(
            f"{k}: approved={approved[k]!r} got={header.get(k)!r}" for k in mismatched
        )
        raise SystemExit(
            f"[classify] REFUSED: {leg} leg header regime differs from the approved "
            f"#2514 pin on {mismatched} — an unapproved remap or capture loss cannot "
            f"self-certify through the directional replay (round-3 reconciler "
            f"item 1a). {detail}"
        )


#: The ONLY realized c26 transitions the plan's REGISTERED expected-inversion
#: class names ("an H100-basis plan that stops WARNing, or an A100-basis plan
#: that starts WARNing") — round-3 reconciler item 1b: any other realized
#: transition (``WARN->SKIP`` included) is not semantically an inversion and
#: lands in ``unexplained`` even when the directional replay matches both
#: sides (an empty routed set predicts SKIP self-consistently).
_C26_INVERSION_TRANSITIONS = frozenset({("WARN", "PASS"), ("PASS", "WARN")})


def _bucket_c26_flip(before: str, after: str, predicted_before: str, predicted_after: str) -> str:
    """Bucket one c26 verdict flip: ``expected-inversion`` requires BOTH the
    directional-replay match (realized == predicted under each leg's own
    mirror; round-2 item 2) AND a realized transition in
    ``_C26_INVERSION_TRANSITIONS`` (round-3 item 1b); anything else is
    ``unexplained`` — the #2514 KILL criterion."""
    if (
        (before, after) in _C26_INVERSION_TRANSITIONS
        and before == predicted_before
        and after == predicted_after
    ):
        return "expected-inversion"
    return "unexplained"


def _predicted_c26(row_meta: list[dict], intents: list[str], mirror: dict[str, str]) -> str:
    """The c26 verdict a mirror ALONE predicts for a plan's recorded
    basis-row tokens — an independent replay of the check's routed-side
    offender rule (basis family not in the routed set; no routed family in
    the conversion cells; no scaling escape), NOT a re-run of the check.
    ``expected-inversion`` requires the realized verdict to EQUAL this
    prediction on BOTH sides of the flip (round-2 reconciler item 2): a
    flip the mirror does not predict — a gate change, a row-parsing change,
    an escape anomaly — lands in ``unexplained``. The check's
    mirror-INDEPENDENT gates (kind exemption, no-rows SKIP, standalone-N/A
    PASS, runpod-pin SKIP) are deliberately not replayed: they are constant
    across the before/after runs of one plan text, so they cannot produce a
    flip — a flip that nonetheless traces to one is exactly the anomaly the
    unexplained bucket must surface. ANCHOR (round-3 item 1a): this replay
    runs under whatever mirror the header carries, so on its own it cannot
    reject a wrong-but-self-consistent remap — ``cmd_classify`` refuses
    unapproved leg regimes (``_assert_approved_regime``) BEFORE any
    prediction runs."""
    routed = {mirror[i] for i in intents if i in mirror}
    if not routed:
        return "SKIP"
    for r in row_meta:
        if r["basis_family"] in routed:
            continue
        if set(r["conv_families"]) & routed:
            continue
        if r["scaling"]:
            continue
        return "WARN"
    return "PASS"


def cmd_classify(args: argparse.Namespace) -> int:
    """Bucket every c26/c27 verdict flip; refuses (SystemExit) a leg whose
    header regime differs from the approved #2514 pins; exit 1 on any
    ``unexplained``."""
    before_hdr, before = _load_rows(args.before)
    after_hdr, after = _load_rows(args.after)
    _assert_approved_regime("before", before_hdr)
    _assert_approved_regime("after", after_hdr)
    if set(before) != set(after):
        print(
            f"corpus mismatch: {len(set(before) ^ set(after))} plans differ "
            "between the two sweeps — re-run both on one tree",
            file=sys.stderr,
        )
        return 1
    mirror_old = before_hdr["mirror"]
    mirror_new = after_hdr["mirror"]
    new_keys = set(mirror_new) - set(mirror_old)
    buckets: dict[str, list[dict]] = {
        "expected-inversion": [],
        "c27-disarm": [],
        "new-key-arming": [],
        "unexplained": [],
    }
    n_flips = 0
    for rel, b in before.items():
        a = after[rel]
        for check in ("c26", "c27"):
            if b[check] == a[check]:
                continue
            n_flips += 1
            fams_old = _families(b["intents"], mirror_old)
            fams_new = _families(a["intents"], mirror_new)
            entry = {
                "plan": rel,
                "check": check,
                "before": b[check],
                "after": a[check],
                "intents": a["intents"],
                "families_old": sorted(fams_old),
                "families_new": sorted(fams_new),
                "before_detail": b[f"{check}_detail"],
                "after_detail": a[f"{check}_detail"],
            }
            if check == "c27":
                disarm = a["c27"] == "PASS" and (
                    b["c27"] in ("FAIL", "WARN") or (b["c27"] == "SKIP" and "7B" in b["c27_detail"])
                )
                buckets["c27-disarm" if disarm else "unexplained"].append(entry)
                continue
            # c26 flips
            if b["c26"] == "SKIP" and set(a["intents"]) & new_keys:
                buckets["new-key-arming"].append(entry)
                continue
            # DIRECTIONAL predicate (round-2 reconciler item 2): the round-1
            # `fams_old != fams_new` read was true by construction for every
            # mapped-intent plan under a wholesale family remap — it
            # certified only "families changed". Expected-inversion requires
            # the realized verdicts to MATCH the per-plan verdicts the
            # old/new mirrors predict from the recorded basis-row tokens,
            # AND (round-3 item 1b) the realized transition to BE an
            # inversion — (WARN,PASS)/(PASS,WARN); the header regimes were
            # already asserted against the approved pins (round-3 item 1a).
            predicted_before = _predicted_c26(b["c26_rows"], b["intents"], mirror_old)
            predicted_after = _predicted_c26(a["c26_rows"], a["intents"], mirror_new)
            entry["predicted_before"] = predicted_before
            entry["predicted_after"] = predicted_after
            bucket = _bucket_c26_flip(b["c26"], a["c26"], predicted_before, predicted_after)
            buckets[bucket].append(entry)
    summary = {
        "n_plans": len(before),
        "n_flips": n_flips,
        "counts": {k: len(v) for k, v in buckets.items()},
        "buckets": buckets,
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for k, v in buckets.items():
        example = (
            f" e.g. {v[0]['plan']} {v[0]['check']} {v[0]['before']}->{v[0]['after']}" if v else ""
        )
        print(f"[classify] {k}: {len(v)}{example}", flush=True)
    print(f"[classify] wrote {args.out}", flush=True)
    if buckets["unexplained"]:
        print(
            f"[classify] KILL: {len(buckets['unexplained'])} unexplained verdict "
            "flip(s) — the mapping is wrong, not merely different (#2514 kill criterion)",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sweep = sub.add_parser("sweep", help="run c26/c27 over the plan corpus")
    sweep.add_argument(
        "--verify-plan-path",
        type=Path,
        default=REPO_ROOT_DEFAULT / "scripts" / "verify_plan.py",
        help="verify_plan.py to import (default: this checkout's)",
    )
    sweep.add_argument(
        "--source-ref",
        default=None,
        help="git provenance of --verify-plan-path (e.g. '<40-hex sha>:scripts/verify_plan.py'); "
        "recorded verbatim in the header as annotation — module_sha is the mechanical content key",
    )
    sweep.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    sweep.add_argument("--out", type=Path, required=True)
    sweep.set_defaults(fn=cmd_sweep)
    classify = sub.add_parser("classify", help="diff + bucket two sweep outputs")
    classify.add_argument("--before", type=Path, required=True)
    classify.add_argument("--after", type=Path, required=True)
    classify.add_argument("--out", type=Path, required=True)
    classify.set_defaults(fn=cmd_classify)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
