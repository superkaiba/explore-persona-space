"""Issue #2564 language-intrusion audit (analyzer Step 3.7; r1 finding f8).

Scans every rollout (10 anchor shards, 9,840 completions) and the judged
manipulation-check pool (1,392 draws) for CJK-script intrusion on the
Qwen-2.5-7B-Instruct generations, then recomputes every compliance fire
verdict under two conventions:

- ``zeroed``   — an intruded draw is counted non-complying (worst case);
- ``excluded`` — intruded draws are removed from numerator AND denominator.

Fire semantics are imported from ``scripts/issue2564_judge.py``
(``fire_verdict`` / ``axis_floor`` / ``check_contains_word``) so the recount
is rule-identical to the shipped manipulation check. Programmatic axes
(marker word, injected name) are recounted exactly by re-running the
word-containment check per draw; judged axes join ``judge_scores.jsonl`` on
``(context_id, draw)``.

Writes ``eval_results/issue_2564/intrusion_audit.json`` — the durable record
behind the clean-result's Takeaway 6 / Result 3 intrusion numbers. Pure
counting: no completion text is printed or persisted, only counts + ids.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_WT = Path(__file__).resolve().parents[1]
if str(_WT / "scripts") not in sys.path:
    sys.path.insert(0, str(_WT / "scripts"))

from issue2564_judge import JUDGED_AXES, axis_floor, check_contains_word, fire_verdict  # noqa: E402

# CJK ranges built from ord values (never literal codepoints / \\u escapes in
# source — the Edit-tool NFC-normalization trap, .claude/rules/gotchas.md).
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF),
    (0x3040, 0x30FF),
    (0xAC00, 0xD7AF),
)
CJK_RE = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _CJK_RANGES) + "]")

PROG_AXES = ("lexical_marker", "user_fact")
_NAME_RE = re.compile(r"(?:name is|is called) (\w+)")
_WORD_RE = re.compile(r'"(\w+)"')


def _read_jsonl(path: Path) -> list[dict]:
    """Text-mode line iteration (never splitlines(); gotchas.md JSONL rule)."""
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _slot_word(axis: str, system: str) -> str:
    """Extract the programmatic target word from a slot's system instruction."""
    m = (_NAME_RE if axis == "user_fact" else _WORD_RE).search(system)
    assert m, (axis, system)
    return m.group(1)


def _fire_rows(
    tallies: dict[tuple[str, str], dict[str, int]], denom_of: dict[tuple[str, str], int]
) -> dict[str, dict]:
    """Per-slot verdicts under orig / zeroed / excluded conventions."""
    out: dict[str, dict] = {}
    for (axis, vid), t in sorted(tallies.items()):
        denom = denom_of[(axis, vid)]
        assert t["comply"] + t["noncomply"] + t["incomplete"] == denom, (axis, vid, t, denom)
        comply_z = t["comply"] - t["intr_comply"]
        denom_x = denom - t["intruded"]
        comply_x = t["comply"] - t["intr_comply"]
        incomplete_x = t["incomplete"] - t["intr_incomplete"]
        out[f"{axis}::{vid}"] = {
            "axis": axis,
            "value_id": vid,
            "denom": denom,
            "n_comply": t["comply"],
            "n_incomplete": t["incomplete"],
            "n_intruded": t["intruded"],
            "n_intruded_comply": t["intr_comply"],
            "verdict_orig": fire_verdict(t["comply"], t["incomplete"], denom),
            "verdict_zeroed": fire_verdict(comply_z, t["incomplete"], denom),
            "verdict_excluded": (
                fire_verdict(comply_x, incomplete_x, denom_x) if denom_x > 0 else "undetermined"
            ),
            "excluded_counts": [comply_x, denom_x],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--anchors-dir",
        type=Path,
        default=_WT
        / "data/issue_2564/judge_work/anchors_staging/issue2564_minpair/raw_completions/anchors",
    )
    ap.add_argument(
        "--judge-scores",
        type=Path,
        default=_WT / "data/issue_2564/judge_work/raw/judge_scores.jsonl",
    )
    ap.add_argument(
        "--bank-manifest", type=Path, default=_WT / "eval_results/issue_2564/bank_manifest.json"
    )
    ap.add_argument(
        "--out", type=Path, default=_WT / "eval_results/issue_2564/intrusion_audit.json"
    )
    args = ap.parse_args(argv)

    bank = json.loads(args.bank_manifest.read_text(encoding="utf-8"))

    # ── (1) per-arm rollout scan + intrusion flags keyed (context_id, draw) ──
    per_arm: dict[str, dict[str, int]] = {}
    intruded: dict[tuple[str, int], bool] = {}
    prog_tallies: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "comply": 0,
            "noncomply": 0,
            "incomplete": 0,
            "intruded": 0,
            "intr_comply": 0,
            "intr_incomplete": 0,
        }
    )
    prog_denom: dict[tuple[str, str], int] = defaultdict(int)
    for shard in sorted(args.anchors_dir.glob("anchors_*.jsonl")):
        arm = shard.stem.removeprefix("anchors_")
        n = n_intr = 0
        for r in _read_jsonl(shard):
            n += 1
            hit = CJK_RE.search(r["text"]) is not None
            n_intr += hit
            intruded[(r["context_id"], int(r["draw"]))] = hit
            axis, vid = r["context_id"].split("::")[0], r["value_id"]
            if axis in PROG_AXES:
                word = _slot_word(axis, bank["contexts"][f"{axis}::{vid}::c01"]["system"])
                contains = check_contains_word(r["text"], word)
                t = prog_tallies[(axis, vid)]
                prog_denom[(axis, vid)] += 1
                t["comply" if contains else "noncomply"] += 1
                if hit:
                    t["intruded"] += 1
                    t["intr_comply"] += contains
        per_arm[arm] = {"intruded": n_intr, "total": n}
    total = sum(v["total"] for v in per_arm.values())
    total_intr = sum(v["intruded"] for v in per_arm.values())
    assert total == 9840, total

    # ── (2) judged pool join + per-slot judged tallies ──
    judged_rows = _read_jsonl(args.judge_scores)
    judged_axis: dict[str, dict[str, int]] = defaultdict(lambda: {"intruded": 0, "total": 0})
    crosstab = {"intruded_comply": 0, "intruded_noncomply": 0, "intruded_incomplete": 0}
    judged_tallies: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "comply": 0,
            "noncomply": 0,
            "incomplete": 0,
            "intruded": 0,
            "intr_comply": 0,
            "intr_incomplete": 0,
        }
    )
    judged_denom: dict[tuple[str, str], int] = defaultdict(int)
    for r in judged_rows:
        hit = intruded[(r["context_id"], int(r["draw"]))]
        judged_axis[r["axis"]]["total"] += 1
        judged_axis[r["axis"]]["intruded"] += hit
        t = judged_tallies[(r["axis"], r["value_id"])]
        judged_denom[(r["axis"], r["value_id"])] += 1
        t[r["outcome"]] += 1
        if hit:
            t["intruded"] += 1
            t["intr_comply"] += r["outcome"] == "comply"
            t["intr_incomplete"] += r["outcome"] == "incomplete"
            crosstab[f"intruded_{r['outcome']}"] += 1
    assert sum(v["total"] for v in judged_axis.values()) == len(judged_rows) == 1392

    # ── (3) per-slot recounts + per-axis floor verdicts ──
    slots = _fire_rows(judged_tallies, judged_denom) | _fire_rows(prog_tallies, prog_denom)
    axis_rows: dict[str, dict] = {}
    for axis in (*JUDGED_AXES, *PROG_AXES):
        base = [s for s in slots.values() if s["axis"] == axis and not s["value_id"].endswith("p")]
        floor = axis_floor(len(base))
        row = {"width": len(base), "floor": floor}
        for conv in ("orig", "zeroed", "excluded"):
            n_fired = sum(1 for s in base if s[f"verdict_{conv}"] == "fired")
            row[f"n_fired_{conv}"] = n_fired
            row[f"floor_met_{conv}"] = n_fired >= floor
        axis_rows[axis] = row

    report = {
        "meta": {
            "script": "scripts/issue2564_intrusion_audit.py",
            "cjk_ranges_hex": [[hex(a), hex(b)] for a, b in _CJK_RANGES],
            "fire_rule": "imported from scripts/issue2564_judge.py (fire_verdict/axis_floor)",
            "conventions": {
                "zeroed": "intruded draw counted non-complying; fixed denominator",
                "excluded": "intruded draws removed from numerator and denominator",
            },
        },
        "rollouts": {"per_arm": per_arm, "total": total, "total_intruded": total_intr},
        "judged_pool": {
            "per_axis": dict(sorted(judged_axis.items())),
            "total": len(judged_rows),
            "total_intruded": sum(v["intruded"] for v in judged_axis.values()),
            "fired_overlap_crosstab": crosstab,
        },
        "slot_recounts": slots,
        "axis_floor_verdicts": axis_rows,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    flips = [
        a
        for a, r in axis_rows.items()
        if not (r["floor_met_orig"] == r["floor_met_zeroed"] == r["floor_met_excluded"])
    ]
    print(
        f"[intrusion-audit] rollouts {total_intr}/{total} intruded; judged "
        f"{report['judged_pool']['total_intruded']}/{len(judged_rows)}; floor flips: {flips or 'none'}; "
        f"wrote {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
