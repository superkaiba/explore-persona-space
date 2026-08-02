#!/usr/bin/env python
"""Issue #1345 — aggregate the per-cell judge-leg reports into one summary.

Pure read over `judge_report_*.json` + `judge_sample_*.json`: no API calls, no
re-judging. Emits, per leg, one row per cell carrying the pooled mean and the
capped / natural sub-means, the realized n and cap rate the draw actually used,
and the three-way drop split (CONTENT — of which REFUSAL is a subset — plus
TRANSPORT, never blended, llm-judging rule 24).

Two reporting duties are enforced here rather than left to prose:

- A yield-floor-halted cell (rc=21) is reported in its OWN block with the
  selection caveat attached, and is EXCLUDED from any cross-cell pooled figure.
  Its kept rows are a selected subset of what the cell would have produced, so
  averaging it in silently would launder the selection into the headline. Two of
  the four labelling characters (dana, wren) are halted cells, so a labelling
  contrast that rests on them is flagged, not footnoted.
- A cell whose content-drop share clears the rule-23 truncation-signature
  threshold is flagged for a re-judge instead of being averaged as-is.

CLI:
  uv run python scripts/issue1345_judge_legs_summarize.py
  uv run python scripts/issue1345_judge_legs_summarize.py --legs-dir <dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402

# The run driver writes under this dir; read the SAME env var it does so the two
# defaults cannot drift apart (the judge CLI's own --out-dir default is the
# variant-scoped EVAL_DIR, which is NOT where the driver puts the run).
DEFAULT_LEGS_DIR = Path(os.environ.get("EPM_I1345_JUDGE_OUT", "eval_results/issue_1345/judge_legs"))

# rule 23: a content-drop share at or above this is the truncation signature to
# re-judge, not a tail to average through. The measured healthy cells sit at
# ~0.1%; #1090's censored arms sat at 31-47%.
DROP_FLAG_SHARE = 0.02


def load_reports(legs_dir: Path) -> list[dict]:
    """Every per-cell report under ``legs_dir``, newest-wins per (leg, tag)."""
    out: dict[tuple[str, str], dict] = {}
    for p in sorted(legs_dir.glob("*/judge_report_*.json")):
        d = json.loads(p.read_text())
        d["_path"] = str(p)
        out[(d["leg"], d["tag"])] = d
    return [out[k] for k in sorted(out)]


def cell_row(rep: dict) -> dict:
    """One cell's summary row: means, realized draw, drop split, caveat."""
    means = rep.get("means") or {}
    design = rep.get("sample_design") or {}
    total = rep.get("n_total_draws") or 0
    content = rep.get("n_dropped_draws_content") or 0
    return {
        "cell": rep["tag"],
        "leg": rep["leg"],
        "n_items": rep.get("n_items"),
        "n_scored_items": rep.get("n_scored_items"),
        "realized_n": design.get("realized_n"),
        "realized_capped": design.get("realized_capped"),
        "realized_capped_rate": design.get("realized_capped_rate"),
        "take_all": design.get("take_all"),
        "seed": design.get("seed"),
        "pooled": means.get("pooled"),
        "capped": means.get("capped"),
        "natural": means.get("natural"),
        # rule 24: three-way, never blended. refusal is a SUBSET of content.
        "drops": {
            "n_total_draws": total,
            "content": content,
            "content_refusal_subset": rep.get("n_refusal_draws"),
            "transport": rep.get("n_transport_lost_draws"),
            "content_share": round(content / total, 6) if total else None,
            "classes": rep.get("content_drop_classes"),
        },
        "drop_flag": bool(total) and (content / total) >= DROP_FLAG_SHARE,
        "yield_floor_halted": bool(design.get("yield_floor_halted_cell")),
        "selection_caveat": rep.get("selection_caveat"),
        "judge_model": rep.get("judge_model"),
        "n_draws": rep.get("n_draws"),
        "max_tokens": rep.get("max_tokens"),
        "rubric_sha256": rep.get("rubric_sha256"),
        "report_path": rep.get("_path"),
    }


def summarize(reports: list[dict]) -> dict:
    """Group the cell rows by leg, keeping halted cells out of any pooled figure."""
    by_leg: dict[str, dict] = {}
    for rep in reports:
        row = cell_row(rep)
        leg = by_leg.setdefault(
            row["leg"],
            {"cells": [], "complete_cells": [], "halted_cells": [], "flagged_cells": []},
        )
        leg["cells"].append(row)
        (leg["halted_cells"] if row["yield_floor_halted"] else leg["complete_cells"]).append(
            row["cell"]
        )
        if row["drop_flag"]:
            leg["flagged_cells"].append(row["cell"])

    for leg, blk in by_leg.items():
        complete = [r for r in blk["cells"] if not r["yield_floor_halted"]]
        pooled = [
            r["pooled"]["mean"] for r in complete if (r["pooled"] or {}).get("mean") is not None
        ]
        blk["cross_cell_mean_complete_only"] = (
            round(sum(pooled) / len(pooled), 4) if pooled else None
        )
        blk["n_cells"] = len(blk["cells"])
        blk["cross_cell_note"] = (
            f"mean over {len(pooled)} COMPLETE cells; the "
            f"{len(blk['halted_cells'])} yield-floor-halted cell(s) "
            f"({', '.join(blk['halted_cells']) or 'none'}) are reported individually and "
            "EXCLUDED — their kept rows are a selected subset"
        )
        blk["total_draws"] = sum(r["drops"]["n_total_draws"] for r in blk["cells"])
        blk["total_content_drops"] = sum(r["drops"]["content"] for r in blk["cells"])
        blk["total_transport_losses"] = sum((r["drops"]["transport"] or 0) for r in blk["cells"])
        assert leg  # keep the loop var meaningful for readers
    return by_leg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legs-dir", type=Path, default=DEFAULT_LEGS_DIR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    reports = load_reports(args.legs_dir)
    assert reports, f"no judge_report_*.json under {args.legs_dir}"
    by_leg = summarize(reports)

    out_path = args.out or (args.legs_dir / "judge_legs_summary.json")
    payload = {
        "metadata": c.metadata(0, len(reports), "scripts/issue1345_judge_legs_summarize.py"),
        "n_reports": len(reports),
        "drop_flag_share": DROP_FLAG_SHARE,
        "legs": by_leg,
    }
    c.write_json(out_path, payload)

    for leg, blk in sorted(by_leg.items()):
        print(f"\n===== {leg}: {blk['n_cells']} cells =====")
        print(
            f"{'cell':22s} {'n':>4s} {'pooled':>8s} {'capped':>8s} {'natural':>8s} "
            f"{'drop%':>6s} {'transport':>9s}  flags"
        )
        for r in blk["cells"]:
            flags = []
            if r["yield_floor_halted"]:
                flags.append("SELECTION-CAVEAT")
            if r["drop_flag"]:
                flags.append("DROP-FLAG-REJUDGE")
            if r["take_all"]:
                flags.append("take-all")

            def _m(block: dict | None) -> str:
                v = (block or {}).get("mean")
                return f"{v:8.2f}" if isinstance(v, (int, float)) else f"{'-':>8s}"

            share = r["drops"]["content_share"]
            print(
                f"{r['cell']:22s} {r['realized_n'] or 0:4d} {_m(r['pooled'])} {_m(r['capped'])} "
                f"{_m(r['natural'])} {(share or 0) * 100:5.2f}% "
                f"{r['drops']['transport'] or 0:9d}  {' '.join(flags)}"
            )
        print(f"  {blk['cross_cell_note']}")
        print(
            f"  cross-cell mean (complete cells only): {blk['cross_cell_mean_complete_only']} | "
            f"draws {blk['total_draws']} | content-drops {blk['total_content_drops']} | "
            f"transport {blk['total_transport_losses']}"
        )
    print(f"\n[summary] wrote {out_path}", flush=True)
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
