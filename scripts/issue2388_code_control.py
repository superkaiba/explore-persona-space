#!/usr/bin/env python
"""Positive control for the #2388 code verifiers: run each dataset's OWN canonical
solution through the SAME harness the generation/verification phase uses.

Motivation: a missing import or a broken test-wrapper on the pod would fail every
item and manufacture a FALSE zero-pile that reads as "the model got it wrong".
This is the only check that distinguishes "model is wrong" from "harness is broken".
It must run ON THE POD, where the production verdicts are produced.

Production extension (#2388 plan P0/G1): covers all FIVE code benchmarks —
HumanEval / MBPP / BigCodeBench via the pilot's ``verify_code`` (verbatim), plus
LiveCodeBench-v5 / LeetCodeDataset via the production verifiers in
``scripts/issue2388_gen.py``. Each control item runs ``--runs`` times (default 2)
for the flaky-rate read (G1: mismatch fraction < 2%). ``--bcb-python`` routes
BigCodeBench execution through the isolated ``/opt/bcb-venv`` interpreter (plan
section 4 fork 1; the other benchmarks keep the base sandbox env — their pilot
controls passed 25/25 without it).

LCB canonical-solution source: ``code_generation_lite`` ships NO reference
solutions, so the LCB control uses the dedup overlap — leetcode-platform LCB
items whose slug matches a LeetCodeDataset row take that row's ``completion``
as the canonical solution, exercising the FUNCTIONAL harness path. The
stdin/stdout harness (atcoder/codeforces items) has no canonical source and is
reported as uncontrolled (``stdin_harness_controlled: false``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.atomic_io import atomic_replace
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

sys.path.insert(0, "scripts")
import issue2388_gen as G  # noqa: E402
import issue2388_spread_pilot as P  # noqa: E402

N_CONTROL = 25


def canon_humaneval() -> dict[str, list[tuple[str, str]]]:
    from datasets import load_dataset

    out: dict[str, list[tuple[str, str]]] = {}
    for r in load_dataset("openai/openai_humaneval", split="test"):
        iid = f"humaneval-{r['task_id'].replace('/', '_')}"
        out[iid] = [("prompt+canonical", r["prompt"] + r["canonical_solution"])]
    return out


def canon_mbpp() -> dict[str, list[tuple[str, str]]]:
    from datasets import load_dataset

    out: dict[str, list[tuple[str, str]]] = {}
    for r in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        out[f"mbpp-{r['task_id']}"] = [("code", r["code"])]
    return out


def canon_bigcodebench() -> dict[str, list[tuple[str, str]]]:
    """BigCodeBench field semantics are ambiguous across releases, so try each
    plausible composition and report the pass rate of every one."""
    from datasets import load_dataset

    out: dict[str, list[tuple[str, str]]] = {}
    for r in load_dataset("bigcode/bigcodebench", split="v0.1.4"):
        iid = f"bcb-{r['task_id'].replace('/', '_')}"
        cands: list[tuple[str, str]] = [("canonical_only", r["canonical_solution"])]
        if r.get("code_prompt"):
            cands.append(("code_prompt+canonical", r["code_prompt"] + r["canonical_solution"]))
        if r.get("complete_prompt"):
            cands.append(
                ("complete_prompt+canonical", r["complete_prompt"] + r["canonical_solution"])
            )
        out[iid] = cands
    return out


def canon_leetcode() -> dict[str, list[tuple[str, str]]]:
    """LeetCodeDataset ships its canonical solution in ``completion``."""
    return {r["item_id"]: [("completion", r["canonical_completion"])] for r in G.load_leetcode()}


def canon_lcb_overlap() -> dict[str, list[tuple[str, str]]]:
    """LCB functional-harness control via the dedup overlap's LeetCode completions."""
    by_slug = {r["slug"]: r["canonical_completion"] for r in G.load_leetcode()}
    out: dict[str, list[tuple[str, str]]] = {}
    for r in G.load_lcb_v5():
        if r["platform"] == "leetcode" and r["slug"] in by_slug and r["func_name"]:
            out[r["item_id"]] = [("leetcode_completion", by_slug[r["slug"]])]
    return out


def canon_apps_intro() -> dict[str, list[tuple[str, str]]]:
    """APPS ships reference ``solutions`` per problem; take the first (G1 for
    the fork-5 contingency: the APPS pilot requires its own control + flaky read)."""
    out: dict[str, list[tuple[str, str]]] = {}
    for r in G.load_apps_intro():
        if r["canonical_solutions"]:
            out[r["item_id"]] = [("solution0", r["canonical_solutions"][0])]
    return out


# (loader, canonical-map builder) per benchmark; loaders are the SAME functions
# the production phases consume, so the control exercises identical item shapes.
BENCHES: dict[str, dict] = {
    "humaneval": {"items": lambda: P.LOADERS["humaneval"](), "canon": canon_humaneval},
    "mbpp": {"items": lambda: P.LOADERS["mbpp"](), "canon": canon_mbpp},
    "bigcodebench": {"items": lambda: P.LOADERS["bigcodebench"](), "canon": canon_bigcodebench},
    "lcb_v5": {"items": None, "canon": canon_lcb_overlap},  # items resolved from the canon keys
    "leetcode": {"items": lambda: G.load_leetcode(), "canon": canon_leetcode},
    "apps_intro": {"items": lambda: G.load_apps_intro(), "canon": canon_apps_intro},
}


def _verify(bench: str, fenced: str, item: dict, bcb_python: str | None) -> bool | None:
    if bench == "bigcodebench" and bcb_python:
        # Route BCB through the isolated venv interpreter (plan fork 1). The
        # payload composition matches the pilot's verify_code bigcodebench branch.
        return G._verify_pilot_code(
            fenced, {**item, "benchmark": "bigcodebench_full"}, python_exe=bcb_python
        )
    if bench == "lcb_v5":
        return G._verify_lcb(fenced, item)
    if bench == "leetcode":
        return G._verify_leetcode(fenced, item)
    if bench == "apps_intro":
        return G._verify_apps(fenced, item)
    return P.verify_code(fenced, item)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(BENCHES),
        # Contingency-only apps_intro stays OUT of the default control roster
        # (r2 Minor 3: a default P0 control run must not download/execute APPS;
        # fork 5 runs it explicitly via --benchmarks apps_intro).
        default=sorted(set(BENCHES) - {"apps_intro"}),
    )
    ap.add_argument("--n-control", type=int, default=N_CONTROL)
    ap.add_argument("--runs", type=int, default=2, help="runs per item (flaky-rate read; G1 <2%%)")
    ap.add_argument("--bcb-python", default=None, help="/opt/bcb-venv/bin/python for BCB items")
    ap.add_argument(
        "--out", default="eval_results/issue_2388/gen/code_harness_control.json", type=Path
    )
    args = ap.parse_args(argv)

    report: dict[str, dict] = {}
    for bench in args.benchmarks:
        canon = BENCHES[bench]["canon"]()
        if BENCHES[bench]["items"] is None:  # lcb_v5: control set == overlap items
            all_items = [r for r in G.load_lcb_v5() if r["item_id"] in canon]
        else:
            all_items = BENCHES[bench]["items"]()
        items = all_items[: args.n_control]
        per_comp: dict[str, dict[str, int]] = {}
        missing = 0
        n_flaky = 0
        n_run = 0
        n_pairs = 0  # realized (item, composition) pairs — the flaky-rate denominator
        t0 = time.time()
        for it in items:
            cands = canon.get(it["item_id"])
            if not cands:
                missing += 1
                continue
            n_run += 1
            n_pairs += len(cands)
            for label, sol in cands:
                slot = per_comp.setdefault(label, {"pass": 0, "fail": 0, "unparsed": 0})
                fenced = f"```python\n{sol}\n```"
                verdicts = [
                    _verify(bench, fenced, it, args.bcb_python) for _ in range(max(1, args.runs))
                ]
                if len(set(map(str, verdicts))) > 1:
                    n_flaky += 1
                v = verdicts[0]
                if v is None:
                    slot["unparsed"] += 1
                elif v:
                    slot["pass"] += 1
                else:
                    slot["fail"] += 1
            print(
                f"[control] {bench} unit {n_run}/{len(items)} {it['item_id']} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        best = max(
            per_comp.items(),
            key=lambda kv: kv[1]["pass"],
            default=("none", {"pass": 0, "fail": 0, "unparsed": 0}),
        )
        n = n_run
        rate = best[1]["pass"] / n if n else 0.0
        # Denominator = REALIZED (item, composition) pairs, matching n_flaky's
        # per-pair counting (r1 g3 Concern 7: n * len(per_comp) over-counts when
        # compositions vary per item, deflating the flaky rate the G1 gate reads).
        flaky_frac = n_flaky / max(1, n_pairs)
        report[bench] = {
            "n_control": n,
            "items_missing_canonical": missing,
            "per_composition": per_comp,
            "best_composition": best[0],
            "best_pass_rate": rate,
            "runs_per_item": max(1, args.runs),
            "n_flaky_items": n_flaky,
            "flaky_mismatch_fraction": flaky_frac,
            "bcb_python": args.bcb_python if bench == "bigcodebench" else None,
            "stdin_harness_controlled": False if bench == "lcb_v5" else None,
            "harness_ok": rate >= 0.90 and flaky_frac < 0.02,
        }
        print(
            f"{bench:>14} n={n:3d} best={best[0]} pass_rate={rate:.3f} "
            f"flaky={flaky_frac:.3f} "
            f"{'HARNESS-OK' if report[bench]['harness_ok'] else 'HARNESS-SUSPECT'}",
            flush=True,
        )
        for label, c in sorted(per_comp.items()):
            print(
                f"                 {label:>26}: pass={c['pass']} fail={c['fail']} "
                f"unparsed={c['unparsed']}",
                flush=True,
            )

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    meta = as_metadata_dict(git_provenance(), phase="code-harness-control")
    # MERGE-don't-clobber (r3 Critical 1): phase_gate reads the BCB *and* APPS
    # control rows from THIS ONE report, and the fork-5 chain runs the APPS
    # control as a SEPARATE `--benchmarks apps_intro` invocation — a whole-file
    # rewrite would erase the BCB verdict, unresolve bcb_fit_allowed /
    # apps_required, and deadlock the documented DROP->APPS sequence. Prior
    # rows not re-run this invocation are preserved verbatim; per-invocation
    # provenance rides the `invocations` list.
    run_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    merged: dict[str, dict] = {}
    invocations: list[dict] = []
    if args.out.exists():
        prior = json.loads(args.out.read_text())
        merged.update(prior.get("benchmarks", {}))
        invocations = list(prior.get("invocations", []))
        if not invocations and merged:
            # pre-merge single-shot report: preserve its provenance as an entry
            # (r4 NIT code-control-invocation-provenance-partial: carry the
            # prior top-level dirty flag too, when present).
            invocations.append(
                {
                    "benchmarks": sorted(merged),
                    "git_commit": prior.get("git_commit"),
                    "git_dirty": prior.get("git_dirty"),
                    "ts": prior.get("ts"),
                }
            )
        # Row-grain freshness backfill (r4 code-control-preserved-row-freshness):
        # a legacy preserved row without its own control_ts inherits the prior
        # report's top-level ts, so every row carries WHEN it was actually run.
        for row in merged.values():
            row.setdefault("control_ts", prior.get("ts"))
            row.setdefault("control_git_commit", prior.get("git_commit"))
    # Freshness stamp on THIS invocation's rows (r4 code-control-preserved-row-
    # freshness): preserved rows keep their own control_ts — phase_gate surfaces
    # both rows' control_ts into the gate verdict so a resumed out-root's stale
    # APPS/BCB control is auditable at every gate read, never silently fresh.
    for row in report.values():
        row["control_ts"] = run_ts
        row["control_git_commit"] = meta.get("git_commit")
    merged.update(report)
    invocations.append(
        {
            "benchmarks": sorted(report),
            "phase": "code-harness-control",
            "argv": list(argv) if argv is not None else sys.argv[1:],
            "git_commit": meta.get("git_commit"),
            "git_dirty": meta.get("git_dirty"),
            "git_dirty_paths": meta.get("git_dirty_paths"),
            "ts": run_ts,
        }
    )
    payload = {"benchmarks": merged, "invocations": invocations}
    payload.update(meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(args.out) as tmp:
        tmp.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} (benchmarks now: {', '.join(sorted(merged))})")
    # rc reflects THIS invocation's benchmarks only — a preserved prior FAIL
    # row (e.g. the dropped-BCB verdict) must not fail a passing APPS control.
    return 0 if all(v["harness_ok"] for v in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
