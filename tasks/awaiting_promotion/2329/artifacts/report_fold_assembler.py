#!/usr/bin/env python3
"""Fold the q35_ladder_decay round into #2329's existing report-v1 body.

Step 7c mechanical assembly (orchestrator-owned, no interpreting agent):
appends the round's Motivation bullets, its shared-Methodology block (with the
manifest's verbatim condition/metric labels and a per-phase Code-SHAs split),
and one `### ` Results subsection per round figure id -- each carrying a
**Methodology** block, the SHA-pinned aggregate image, and an untouched
`**Takeaways**` placeholder for Thomas.

Claim-free by construction: every line is copied from the methodology-writer's
sections doc or the plotter's captions.json. Nothing is authored here.
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

# Headings are the manifest's OWN `title` fields, verbatim -- the
# `manifest-figures` check binds a planned figure to its Results subsection by
# an EXACT (case-insensitive) heading match against the figure's id or title,
# so an invented heading reads as an unplotted figure however descriptive it is.
HEADINGS = {
    "q35_ladder_decay_hero_ladder": "Persona-specificity ladder on Qwen3.5-9B",
    "q35_ladder_decay_transfer": "Ladder transfer read: Qwen3.5-9B vs Qwen2.5-7B",
    "q35_ladder_decay_anchor_separation": "Anchor separation per rung and carrier (Qwen3.5)",
    "q35_ladder_decay_decay_raw": "Within-answer decay, raw per-segment scores",
    "q35_ladder_decay_decay_norm": "Within-answer decay, anchor-normalized F per segment",
    "q35_ladder_decay_contrast": "Patched-vs-prompted decay contrast",
    "q35_ladder_decay_diagnostics": "Ladder + decay diagnostics dump",
}

# Reproducibility cards -> the commit each phase's card records (read from the
# cards themselves at extract time; see the marker for the probe).
CARDS = [
    ("gates/token_identity_report_ladder.json", "G0 ladder token-identity gate",
     "ccb83356f964cb78ff9347e290eb165a1dc7ea76"),
    ("judge/gates/coherence_baseline_gate.json", "G3 anchor coherence baseline gate",
     "7caaecf958269778655a0831a8c1d17ce83468a8"),
    ("judge/scores/coherence.grid.meta.json", "Leg A grid coherence judge wave",
     "a47482af9ee7a447a0fa65fc8401cc54dbf8c6d2"),
    ("cap_hit/cap_hit_report_grid.json", "Leg A grid cap-hit report",
     "a90b45020cd31a52f6594ab8bef6ea3517f1a427"),
    ("cap_hit/cap_hit_report_anchors.json",
     "anchors cap-hit report (declared cap 2048 -- known-stale, see the G5 disclosure)",
     "d0c07f98a2c52d02fc1578a1909b77927c434b4d"),
    ("f_metrics/stats.json", "Leg A F-metrics reduce",
     "e408832800a5d75c691eddf4d09b8078b1286110"),
    ("decay/judge/gates/pilot_gate_report.json", "Leg B judge pilot gate",
     "4b5d184719c0da423c50e0adb8686dcde686e79a"),
]

ROUND_CONDITIONS = [
    "Ladder steered arm on Qwen3.5-9B (persona-value patch, thinking disabled)",
    "Ladder same-value-donor null (Qwen3.5)",
    "Ladder cross-type-donor null (Qwen3.5, construct-screened)",
    "Ladder floor anchor (plain context, unpatched)",
    "Ladder ceiling anchor (persona-prompted, unpatched)",
    "Install direction (plain to persona)",
    "Erase direction (persona to plain)",
    "Persona-specificity rungs (plain, pirate, butler, warm, trait, Lu therapy, Lu philosophy)",
    "Within-answer quartile segments (Q1-Q4)",
    "Patched-vs-prompted decay contrast (install, context-end primary; "
    "coherence-screened, paired per carrier on common support)",
    "Prefix-end exploratory decay stratum (conditional on a realized install-pe transfer)",
    "Qwen2.5-7B parent ladder completions (re-judged per segment)",
]

ROUND_METRICS = [
    "per-rung fraction-of-swap on the specificity ladder",
    "rung-rank trend (within-carrier Spearman)",
    "anchor separation (ceiling minus floor)",
    "per-segment persona expression score (0-100)",
    "per-segment anchor-normalized F",
    "within-answer decay drop (Q1 minus Q4)",
    "patched-vs-prompted decay difference",
    "absolute Q1 starting-level gap (steered minus ceiling)",
    "all-generated vs coherence-conditional decay estimands (per-arm retention "
    "counts; cross-estimand disagreement => UNRESOLVED)",
]


def sections_between(lines: list[str], start_pat: str, end_pat: str) -> list[str]:
    """Lines strictly between the first start_pat match and the next end_pat match."""
    s = e = None
    for i, ln in enumerate(lines):
        if s is None and re.match(start_pat, ln):
            s = i
            continue
        if s is not None and re.match(end_pat, ln):
            e = i
            break
    if s is None:
        raise SystemExit(f"assemble: start pattern not found: {start_pat}")
    if e is None:
        e = len(lines)
    return lines[s + 1:e]


def strip_blanks(block: list[str]) -> list[str]:
    while block and not block[0].strip():
        block.pop(0)
    while block and not block[-1].strip():
        block.pop()
    return block


def main() -> int:
    task_dir = Path(subprocess.run(
        ["uv", "run", "python", "scripts/task.py", "find", "2329"],
        capture_output=True, text=True, check=True).stdout.strip())
    wt = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/"
              "issue-2329-q35-ladder-decay")

    body_p = task_dir / "body.md"
    sect_p = task_dir / "artifacts" / f"issue-2329-{ROUND.replace('_','-')}-report-sections.md"
    if not sect_p.exists():  # the committed name uses the round slug verbatim
        sect_p = task_dir / "artifacts" / "issue-2329-q35-ladder-decay-report-sections.md"
    caps_p = wt / "figures" / "issue_2329" / ROUND / "captions.json"

    body = body_p.read_text().splitlines()
    sect = sect_p.read_text().splitlines()
    caps = json.loads(caps_p.read_text())

    # --- extract the writer's sections ---------------------------------------
    motiv = strip_blanks(sections_between(sect, r"^## Motivation\s*$", r"^## "))
    shared = strip_blanks(sections_between(sect, r"^## Methodology \(shared\)\s*$", r"^## "))

    # The writer left the dashboard slot as "pending"; record the REALIZED
    # decision with its arithmetic instead of shipping an open TODO.
    PENDING = "Dashboard link manifest: not passed to this round — dashboard links pending."
    REALIZED = (
        "Dashboard link manifest: not passed to this round, and per-row dashboards "
        "were NOT built for it: the first pass's three tables (f_cells 2,676 rows / "
        "anchors 1,368 / stage2_cells 4,032) already occupy ~9.07 MB of the "
        "~10 MB-per-issue dashboard payload budget, and this round's eleven per-row "
        "tables measure ~8.06 MB (measured by a build run that was then reverted), so "
        "carrying both would roughly double the cap. The round's per-row tables are "
        "committed as JSONL under `eval_results/issue_2329/q35_ladder_decay/"
        "{f_metrics,decay}/` and cited from the per-result Methodology blocks instead."
    )
    hit = sum(1 for ln in shared if PENDING in ln)
    if hit != 1:
        raise SystemExit(f"assemble: expected exactly 1 dashboard-pending line, found {hit}")
    shared = [ln.replace(PENDING, REALIZED) for ln in shared]

    per_result: dict[str, list[str]] = {}
    for fid in HEADINGS:
        blk = sections_between(sect, rf"^### `{re.escape(fid)}`\s*$", r"^### |^## ")
        blk = strip_blanks(blk)
        # drop the leading '**Methodology**' label; we re-emit it in the body form
        if blk and blk[0].strip() == "**Methodology**":
            blk = strip_blanks(blk[1:])
        if not blk:
            raise SystemExit(f"assemble: empty per-result block for {fid}")
        per_result[fid] = blk

    # --- build the Motivation append ----------------------------------------
    mot_add = [
        "",
        f"**Follow-up round `{ROUND}` (Leg A persona-specificity ladder + "
        "Leg B within-answer decay):**",
        "",
    ] + motiv

    # --- build the Methodology (shared) append ------------------------------
    meth_add = [
        "",
        f"### Follow-up round `{ROUND}` — shared methodology",
        "",
    ] + shared + [
        "",
        "**Conditions (this round):**",
        "",
    ] + [f"- {c}" for c in ROUND_CONDITIONS] + [
        "",
        "**Metrics (this round):**",
        "",
    ] + [f"- {m}" for m in ROUND_METRICS] + [
        "",
        "**Code SHAs (this round, per phase — each phase at its own "
        "reproducibility card's commit; a card recording a dirty tree is "
        "excluded and not cited):**",
        "",
    ] + [
        f"- `eval_results/issue_2329/{ROUND}/{path}` ({phase}) @ `{commit}`"
        for path, phase, commit in CARDS
    ]

    # --- build the Results subsections ---------------------------------------
    res_add: list[str] = []
    for fid, heading in HEADINGS.items():
        entry = caps.get(fid) or {}
        agg = entry.get("aggregate_view")
        if not agg:
            raise SystemExit(f"assemble: no aggregate_view in captions for {fid}")
        bullets = entry.get("caption_bullets") or []
        if not bullets:
            raise SystemExit(f"assemble: no caption_bullets for {fid}")
        res_add += ["", f"### {heading}", "", "**Methodology**", ""]
        res_add += per_result[fid]
        res_add += [""]
        res_add += [f"- {b}" for b in bullets]
        res_add += ["", f"![{heading} — aggregate view]({RAW}/{agg})", "",
                    "**Takeaways**", "", "*(Thomas fills in)*"]

    # --- splice ---------------------------------------------------------------
    def idx_of(pat: str) -> int:
        for i, ln in enumerate(body):
            if re.match(pat, ln):
                return i
        raise SystemExit(f"assemble: body heading not found: {pat}")

    i_tldr = idx_of(r"^## TLDR\s*$")
    i_results = idx_of(r"^## Results\s*$")
    i_concl = idx_of(r"^## Conclusion and next steps\s*$")
    assert i_tldr < i_results < i_concl, "unexpected body section order"

    out = (body[:i_tldr] + mot_add + [""] +
           body[i_tldr:i_results] + meth_add + [""] +
           body[i_results:i_concl] + res_add + [""] +
           body[i_concl:])

    # Re-pin the detailed-companion link: the companion gained this round's 16
    # views, so the body's pin must move to the commit that carries them.
    OLD_DET = "60e1c290fd1a65cd9de1101f85c7971756270a61"
    # Moves with every companion regeneration; currently the round-2 fix pass
    # (corrected writer blocks + corrected captions).
    NEW_DET = "9c4cacb10bed274e3e4d0848ff7b1d999d0578ea"
    repins = 0
    for i, ln in enumerate(out):
        if ln.startswith("**Detailed writeup:**") and OLD_DET in ln:
            out[i] = ln.replace(OLD_DET, NEW_DET)
            repins += 1
    if repins != 1:
        raise SystemExit(f"assemble: expected exactly 1 detailed-writeup re-pin, made {repins}")

    draft = task_dir / "artifacts" / "report-draft.md"
    draft.write_text("\n".join(out) + "\n")
    print(f"wrote {draft}  ({len(out)} lines, was {len(body)})")
    print(f"  motivation +{len(mot_add)}  methodology +{len(meth_add)}  results +{len(res_add)}")
    print(f"  subsections added: {len(HEADINGS)}  pin={SHA[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
