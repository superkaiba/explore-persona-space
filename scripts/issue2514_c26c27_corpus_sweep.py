"""Before/after c26+c27 corpus sweep for the #2514 routed-machine mirror rebuild.

Two subcommands:

``sweep``
    Runs the SHIPPED ``check_gpu_basis_routed_machine`` (c26) and
    ``check_capture_intent_hbm`` (c27) — imported from a given
    ``verify_plan.py`` path, never a re-implementation (the #2276
    ``issue2276_c62c63_corpus_sweep.py`` convention) — over every
    ``tasks/*/*/plans/v*.md`` under ``--repo-root`` and appends one JSON row
    per plan (c26/c27 status + detail head + resolved intents) to ``--out``
    (JSONL; header row first). Point ``--verify-plan-path`` at a
    ``git show <ref>:scripts/verify_plan.py`` materialization for the
    BEFORE leg and at the live ``scripts/verify_plan.py`` for the AFTER leg.

``classify``
    Diffs a before/after JSONL pair and buckets EVERY c26/c27 verdict flip
    into the #2514 plan taxonomy: ``expected-inversion`` (the routed family
    set changed for the plan's resolved intents), ``c27-disarm`` (a c27
    FAIL/WARN — or a downstream no->=7B-signal SKIP — that becomes the D3
    empty-under-floor PASS), ``new-key-arming`` (a c26 SKIP that now
    resolves because the plan books a key new to the mirror, e.g. inf-70b /
    ft-70b), and ``unexplained`` (anything else). A non-empty
    ``unexplained`` set exits 1 — the plan's KILL criterion; never baseline
    it away.

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


def cmd_sweep(args: argparse.Namespace) -> int:
    """Append one JSON row per plan (checkpoint-per-unit; re-runnable)."""
    mod = _load_verify_plan(args.verify_plan_path, args.repo_root)
    plans = sorted(args.repo_root.glob("tasks/*/*/plans/v*.md"))
    under = getattr(mod, "_C27_UNDER_HBM_INTENTS", None)
    if under is None:  # the pre-#2514 module names the set by its L4 proxy
        under = getattr(mod, "_C27_L4_INTENTS", frozenset())
    header = {
        "verify_plan_path": str(args.verify_plan_path),
        "n_plans": len(plans),
        "mirror": dict(mod._C26_INTENT_GPU),
        "lane_head": getattr(mod, "_C26_LANE_HEAD", None),
        "under_hbm_intents": sorted(under),
    }
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"header": header}) + "\n")
        for i, path in enumerate(plans, 1):
            text = path.read_text(errors="replace")
            r26 = mod.check_gpu_basis_routed_machine(text, "experiment")
            r27 = mod.check_capture_intent_hbm(text, "experiment")
            row = {
                "plan": str(path.relative_to(args.repo_root)),
                "intents": sorted(mod._c26_intents(text)),
                "c26": r26.status,
                "c26_detail": r26.detail[:160],
                "c27": r27.status,
                "c27_detail": r27.detail[:160],
            }
            fh.write(json.dumps(row) + "\n")
            if i % 250 == 0 or i == len(plans):
                print(
                    f"[sweep] unit {i}/{len(plans)} {path.name} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    print(f"[sweep] wrote {args.out} ({len(plans)} plans)", flush=True)
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


def cmd_classify(args: argparse.Namespace) -> int:
    """Bucket every c26/c27 verdict flip; exit 1 on any ``unexplained``."""
    before_hdr, before = _load_rows(args.before)
    after_hdr, after = _load_rows(args.after)
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
            elif fams_old != fams_new:
                buckets["expected-inversion"].append(entry)
            else:
                buckets["unexplained"].append(entry)
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
