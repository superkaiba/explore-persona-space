#!/usr/bin/env python3
"""Append the q35_ladder_decay round to #2329's detailed companion writeup.

The body carries ONE headline (aggregate) view per manifest figure id; the
detailed companion must carry EVERY produced view. This appends one `###`
block per round figure id holding the methodology-writer's recipe bullets, the
plotter's factual caption bullets, and ALL views (aggregate + per-unit).

No Takeaways / TLDR / Conclusion -- those are Thomas's slots and the companion
carries none by contract.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

SHA = "216c793f5013e6eed897e90011bc187f2da54b76"
RAW = f"https://raw.githubusercontent.com/superkaiba/explore-persona-space/{SHA}"
ROUND = "q35_ladder_decay"

TITLES = {
    "q35_ladder_decay_hero_ladder": "Persona-specificity ladder on Qwen3.5-9B",
    "q35_ladder_decay_transfer": "Ladder transfer read: Qwen3.5-9B vs Qwen2.5-7B",
    "q35_ladder_decay_anchor_separation": "Anchor separation per rung and carrier (Qwen3.5)",
    "q35_ladder_decay_decay_raw": "Within-answer decay, raw per-segment scores",
    "q35_ladder_decay_decay_norm": "Within-answer decay, anchor-normalized F per segment",
    "q35_ladder_decay_contrast": "Patched-vs-prompted decay contrast",
    "q35_ladder_decay_diagnostics": "Ladder + decay diagnostics dump",
}


def sections_between(lines: list[str], start_pat: str, end_pat: str) -> list[str]:
    s = e = None
    for i, ln in enumerate(lines):
        if s is None and re.match(start_pat, ln):
            s = i
            continue
        if s is not None and re.match(end_pat, ln):
            e = i
            break
    if s is None:
        raise SystemExit(f"append_detailed: start not found: {start_pat}")
    return lines[s + 1:e if e is not None else len(lines)]


def strip_blanks(b: list[str]) -> list[str]:
    while b and not b[0].strip():
        b.pop(0)
    while b and not b[-1].strip():
        b.pop()
    return b


def main() -> int:
    repo = Path("/home/thomasjiralerspong/explore-persona-space")
    wt = repo / ".claude/worktrees/issue-2329-q35-ladder-decay"
    task_dir = Path(subprocess.run(
        ["uv", "run", "python", "scripts/task.py", "find", "2329"],
        cwd=repo, capture_output=True, text=True, check=True).stdout.strip())

    det_p = wt / "docs/reports/issue_2329_detailed.md"
    sect_p = task_dir / "artifacts/issue-2329-q35-ladder-decay-report-sections.md"
    caps_p = wt / f"figures/issue_2329/{ROUND}/captions.json"

    det = det_p.read_text().splitlines()
    sect = sect_p.read_text().splitlines()
    caps = json.loads(caps_p.read_text())

    if any(ROUND in ln for ln in det):
        raise SystemExit("append_detailed: round content already present -- refusing to double-append")

    add: list[str] = [
        "",
        f"## Results — follow-up round `{ROUND}` (full figure set)",
        "",
        "Every view produced by this round: the aggregate view that appears in the "
        "report body, plus every per-unit / companion view. Captions are the "
        "plotter's factual what-is-plotted text; recipes are the "
        "methodology-writer's. Figures pinned at "
        f"[`{SHA[:12]}`](https://github.com/superkaiba/explore-persona-space/tree/{SHA}"
        f"/figures/issue_2329/{ROUND}).",
    ]

    n_views = 0
    for fid, title in TITLES.items():
        blk = strip_blanks(sections_between(sect, rf"^### `{re.escape(fid)}`\s*$", r"^### |^## "))
        if blk and blk[0].strip() == "**Methodology**":
            blk = strip_blanks(blk[1:])
        entry = caps.get(fid) or {}
        views = [entry.get("aggregate_view")] + list(entry.get("per_unit_views") or [])
        views = [v for v in views if v]
        if not views:
            raise SystemExit(f"append_detailed: no views for {fid}")
        bullets = entry.get("caption_bullets") or []

        add += ["", f"### {title}", "", "**Methodology**", ""] + blk + [""]
        add += [f"- {b}" for b in bullets] + [""]
        for v in views:
            stem = v.rsplit("/", 1)[-1].removesuffix(".png")
            local = wt / v
            if not local.exists():
                raise SystemExit(f"append_detailed: missing view file {local}")
            add.append(f"![{title} — {stem}]({RAW}/{v})")
            add.append("")
            n_views += 1

    # insert before the extra-tables section if present, else append
    ins = len(det)
    for i, ln in enumerate(det):
        if re.match(r"^## Extra tables / diagnostics\s*$", ln):
            ins = i
            break

    out = det[:ins] + add + [""] + det[ins:]
    det_p.write_text("\n".join(out) + "\n")
    print(f"appended {n_views} views across {len(TITLES)} figure ids "
          f"({len(det)} -> {len(out)} lines) at insert index {ins}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
