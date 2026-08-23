#!/usr/bin/env python
"""Issue #2479 P0 refit-equality parity check (kill criterion (c), plan v4 §4/§7/§8).

Compares a pilot re-fit of the PARENT cell (char_helios_op, run through the
production entrypoint `issue1345_story_char_ladder_fill.py` in --pilot-outdir
mode) against the parent's committed values on origin/issue-1345, frozen into
`eval_results/issue_2479/parity_reference_char_helios_op.json`.

Checked metrics (each |got - ref| must be <= tolerance):
  - cell context arm : reduced.ceiling_r2, reduced.identity_bias_r2
  - cell prefix arm  : reduced.ceiling_r2, reduced.identity_bias_r2
  - ladder r4op:char_helios_op forward direction: r2 of every rung

Exit codes (designed, distinct — pod-side-reporting rule):
  0 = all metrics within tolerance
  1 = at least one metric over tolerance (kill criterion (c) trips)
  2 = missing file / missing key / row-count mismatch (pilot did not
      reproduce the parent's fit shape at all)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CELL_TMPL = "cell_char_helios_op__instruct_{arm}_L19_reduced_s0.json"
LADDER_NAME = "ladder_r4op__char_helios_op__instruct_context_L19_reduced_s0_nd2.json"
LADDER_DIRECTION = "story_onpolicy->char_helios_op"


def _scalar(x: object) -> float:
    if isinstance(x, list):
        if len(x) != 1:
            raise KeyError(f"expected 1-element list, got len={len(x)}")
        return float(x[0])
    return float(x)  # type: ignore[arg-type]


def _load(path: Path) -> dict:
    if not path.is_file():
        print(f"[parity] MISSING file={path}")
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-dir", required=True, help="dir holding the pilot cell/ladder JSONs")
    ap.add_argument(
        "--reference",
        default="eval_results/issue_2479/parity_reference_char_helios_op.json",
        help="committed parity reference JSON",
    )
    ap.add_argument("--tol", type=float, default=None, help="override reference tolerance")
    args = ap.parse_args(argv)

    pilot = Path(args.pilot_dir)
    try:
        ref = _load(Path(args.reference))
    except FileNotFoundError:
        return 2
    tol = args.tol if args.tol is not None else float(ref["tolerance"])

    rows: list[tuple[str, float, float]] = []  # (metric, ref_val, got_val)
    try:
        for arm in ("context", "prefix"):
            got = _load(pilot / CELL_TMPL.format(arm=arm))
            red = got["reduced"]
            ref_arm = ref["cell"][arm]
            if int(red["n"]) != int(ref_arm["n"]):
                print(f"[parity] n-mismatch arm={arm} ref={ref_arm['n']} got={red['n']}")
                return 2
            rows.append(
                (f"cell.{arm}.ceiling_r2", ref_arm["ceiling_r2"], _scalar(red["ceiling_r2"]))
            )
            rows.append(
                (
                    f"cell.{arm}.identity_bias_r2",
                    ref_arm["identity_bias_r2"],
                    _scalar(red["identity_bias_r2"]),
                )
            )
        lad = _load(pilot / LADDER_NAME)
        got_r2 = lad["reduced"][LADDER_DIRECTION]["r2"]
        for rung, ref_val in sorted(ref["ladder"]["r2"].items()):
            if rung not in got_r2:
                print(f"[parity] MISSING ladder rung={rung}")
                return 2
            rows.append((f"ladder.{rung}.r2", float(ref_val), _scalar(got_r2[rung])))
    except (FileNotFoundError, KeyError) as exc:
        print(f"[parity] shape-failure: {type(exc).__name__}: {exc}")
        return 2

    n_fail = 0
    for metric, ref_val, got_val in rows:
        diff = abs(got_val - ref_val)
        verdict = "PASS" if diff <= tol else "FAIL"
        if verdict == "FAIL":
            n_fail += 1
        print(
            f"[parity] metric={metric} ref={ref_val:.6f} got={got_val:.6f} "
            f"absdiff={diff:.6f} tol={tol} {verdict}"
        )
    print(f"[parity] summary checked={len(rows)} failed={n_fail} tol={tol}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
