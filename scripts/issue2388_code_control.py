#!/usr/bin/env python
"""Positive control for the #2388 code verifiers: run each dataset's OWN canonical
solution through the SAME harness the pilot used.

Motivation: a missing import or a broken test-wrapper on the pod would fail every
item and manufacture a FALSE zero-pile that reads as "the model got it wrong".
This is the only check that distinguishes "model is wrong" from "harness is broken".
It must run ON THE POD, where the pilot's numbers were produced.

Reuses issue2388_spread_pilot.extract_code + verify_code verbatim (wrapping the
canonical solution in a ```python fence so it takes the identical code path).
"""

from __future__ import annotations

import json
import sys

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

sys.path.insert(0, "scripts")
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


CANON = {
    "humaneval": canon_humaneval,
    "mbpp": canon_mbpp,
    "bigcodebench": canon_bigcodebench,
}


def main() -> int:
    report: dict[str, dict] = {}
    for bench in ("humaneval", "mbpp", "bigcodebench"):
        items = P.LOADERS[bench]()[:N_CONTROL]
        canon = CANON[bench]()
        per_comp: dict[str, dict[str, int]] = {}
        missing = 0
        for it in items:
            cands = canon.get(it["item_id"])
            if not cands:
                missing += 1
                continue
            for label, sol in cands:
                slot = per_comp.setdefault(label, {"pass": 0, "fail": 0, "unparsed": 0})
                verdict = P.verify_code(f"```python\n{sol}\n```", it)
                if verdict is None:
                    slot["unparsed"] += 1
                elif verdict:
                    slot["pass"] += 1
                else:
                    slot["fail"] += 1
        best = max(
            per_comp.items(),
            key=lambda kv: kv[1]["pass"],
            default=("none", {"pass": 0, "fail": 0, "unparsed": 0}),
        )
        n = len(items) - missing
        rate = best[1]["pass"] / n if n else 0.0
        report[bench] = {
            "n_control": n,
            "items_missing_canonical": missing,
            "per_composition": per_comp,
            "best_composition": best[0],
            "best_pass_rate": rate,
            "harness_ok": rate >= 0.90,
        }
        print(
            f"{bench:>14} n={n:3d} best={best[0]} pass_rate={rate:.3f} "
            f"{'HARNESS-OK' if rate >= 0.90 else 'HARNESS-SUSPECT'}"
        )
        for label, c in sorted(per_comp.items()):
            print(f"                 {label:>26}: pass={c['pass']} fail={c['fail']} unparsed={c['unparsed']}")

    out = "eval_results/issue_2388/spread_pilot/code_harness_control.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out}")
    return 0 if all(v["harness_ok"] for v in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
