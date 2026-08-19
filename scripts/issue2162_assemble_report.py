#!/usr/bin/env python3
"""Assemble the workflow-v2 report body + detailed companion for issue #2162.

This is the ORCHESTRATOR-side mechanical assembly step of the v2 report
pipeline (`.claude/skills/issue-v2/report-template.md` § The skeleton /
§ The detailed companion writeup). It authors NOTHING: every sentence of
prose comes from the findings-blind `methodology-writer` output or the
`plotter`'s factual captions. Thomas's claim slots — the `# Result:` title,
`## TLDR`, every per-result `**Takeaways**`, and
`## Conclusion and next steps` — are emitted as the intact placeholder.

Why a script rather than a hand-assembled markdown file: the methodology
sections are still under the `methodology-critic` gate, so the assembly has
to be re-runnable for free after every accuracy fix. Re-run it, do not
hand-edit the outputs.

Inputs
------
--sections    the methodology-writer's findings-blind draft (Motivation +
              Methodology (shared) + one `### <figure id>` block per planned
              figure).
--captions    the plotter's captions JSON (one record per rendered VIEW:
              manifest_figure_id / view / png_relpath / plot_name / caption).
--manifest    the approved planned_manifest.json (conditions / metrics /
              figures). Figure ORDER here is the report's Results order, and
              each figure's `title` becomes the `###` heading verbatim —
              `verify_report.py`'s figure-coverage check requires an
              EXACT-match between each planned figure's id-or-title and a
              `###` heading, so the heading text is load-bearing, not
              cosmetic.
--figures-sha the 40-hex commit the figures are committed at (every image is
              pinned at it).
--detailed-sha the 40-hex commit the detailed companion is committed at (the
              body's `**Detailed writeup:**` line pins it). Emitted as an
              obviously-invalid sentinel when omitted, so a body assembled
              before the detailed doc is committed FAILS the verifier
              loudly instead of shipping an unpinned link.

Outputs
-------
--out-body      the report-v1 body (post with `task.py set-body --snapshot`).
--out-detailed  docs/reports/issue_2162_detailed.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PLACEHOLDER = "*(Thomas fills in)*"
SENTINEL = "<!-- report-v1 -->"
REPO_SLUG = "superkaiba/explore-persona-space"
ISSUE = 2162

# Kept in sync with verify_report.py BANNED_LEXICON — asserted-conclusion
# lexemes are banned from every agent-written section, and the caption text we
# fold in lands inside `## Results`, which IS scanned at generation time. We
# refuse rather than emit, so a plotter caption that drifts into interpretation
# fails here (loud, pre-commit) instead of at the gate.
BANNED_LEXICON = [
    "suggests",
    "confirms",
    "demonstrates that",
    "evidence that",
    "evidence for",
    "we conclude",
    "this shows",
    "indicating that",
    "implying",
]
_LEXICON_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in BANNED_LEXICON) + r")\b", re.IGNORECASE
)

H1_QUESTION = (
    "Which kinds of context information are carried at the context vector — a patch-only "
    "sweep over 21 minimal-pair information types, crossed with route conflict, recency, "
    "and load"
)

# Planned-manifest names whose EXACT string differs typographically from the
# methodology draft's prose term for the same object. `verify_report.py`'s
# manifest-coverage check is a literal word-boundary match on the planned name,
# so without this crosswalk it reports as "not found in report text" an item
# the report does discuss at length. Each entry is (planned name, the draft's
# term) — a typographic bridge, NOT a restatement or a new claim. Verified
# present in the draft at assembly time by _assert_crosswalk_grounded().
MANIFEST_NAME_CROSSWALK = [
    ("behavior fraction-of-swap (F_beh)", "F_beh"),
    ("activation fraction-of-swap (F_act)", "F_act"),
    (
        "teacher-forced positive-vs-negative completion margin",
        "Teacher-forced fixed positive-vs-negative completion margin",
    ),
    ("anchor separation (ceiling minus floor)", "Anchor separation (ceiling − floor)"),
    (
        "Shuffled-donor null (same type, norm-matched)",
        "shuffled-donor null",
    ),
]


def _die(msg: str) -> None:
    raise SystemExit(f"issue2162_assemble_report: {msg}")


def _split_h2(text: str) -> dict[str, str]:
    """Map `## <name>` -> its content (up to the next H2). Preamble ignored."""
    out: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur is not None:
                out[cur] = "\n".join(buf).strip("\n")
            cur = line[3:].strip()
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip("\n")
    return out


def _split_h3(text: str) -> dict[str, str]:
    """Map `### <name>` -> its content. Names are de-backticked + stripped."""
    out: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("### "):
            if cur is not None:
                out[cur] = "\n".join(buf).strip("\n")
            cur = line[4:].strip().strip("`").strip()
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip("\n")
    return out


def _assert_clean(label: str, text: str) -> None:
    hits = sorted({m.group(0).lower() for m in _LEXICON_RE.finditer(text)})
    if hits:
        _die(f"{label} carries banned asserted-conclusion lexeme(s) {hits}; fix the source text")


def _assert_crosswalk_grounded(sections_text: str) -> None:
    """Every crosswalk target must actually appear in the draft.

    The crosswalk claims "the planned name X is the draft's term Y". If Y is
    absent the bridge is fiction, so refuse rather than assert a
    correspondence to text that is not there.
    """
    low = sections_text.lower()
    missing = [(p, d) for p, d in MANIFEST_NAME_CROSSWALK if d.lower() not in low]
    if missing:
        _die(
            "crosswalk target(s) absent from the methodology draft "
            f"(the bridge would be unfounded): {missing}"
        )


def _raw_url(sha: str, relpath: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO_SLUG}/{sha}/{relpath}"


def _headline_view(views: list[dict]) -> dict:
    """The single view the summarized body shows: the aggregate if there is
    one, else the sole/first view in plotter order."""
    for v in views:
        if v.get("view") == "aggregate":
            return v
    return views[0]


def _methodology_block(result_body: str, caption: str) -> str:
    """The result's `**Methodology**` block with the plotter caption folded in.

    Per the template the orchestrator folds each factual caption into that
    result's Methodology block. We append it as a labelled bullet so the
    what-is-plotted text is attributable to the plotter rather than blended
    into the writer's prose.
    """
    body = result_body.strip("\n")
    cap = " ".join(caption.split())
    return f"{body}\n- **Rendered figure (plotter caption):** {cap}"


def build_body(
    *,
    sections: dict[str, str],
    per_result: dict[str, str],
    manifest: dict,
    views_by_id: dict[str, list[dict]],
    figures_sha: str,
    detailed_sha: str,
) -> str:
    detailed_url = (
        f"https://github.com/{REPO_SLUG}/blob/{detailed_sha}/docs/reports/issue_{ISSUE}_detailed.md"
    )
    crosswalk = "\n".join(
        f"    - `{planned}` — the planned-manifest name for {draft} above."
        for planned, draft in MANIFEST_NAME_CROSSWALK
    )
    out: list[str] = [
        f"# Experiment: {H1_QUESTION}",
        SENTINEL,
        "",
        f"**Detailed writeup:** {detailed_url}",
        "",
        "## Motivation",
        "",
        sections["Motivation"].strip("\n"),
        "",
        "## TLDR",
        "",
        PLACEHOLDER,
        "",
        "## Methodology (shared)",
        "",
        sections["Methodology (shared)"].strip("\n"),
        "",
        "- **Planned-manifest name crosswalk.** Five planned condition / metric names"
        " differ typographically from the prose terms used above for the same objects;"
        " they are bridged here so the report's coverage of the approved manifest is"
        " mechanically checkable:",
        crosswalk,
        "",
        "## Results",
        "",
    ]
    for fig in manifest["figures"]:
        fid = str(fig["id"]).strip()
        title = str(fig["title"]).strip()
        if fid not in per_result:
            _die(f"planned figure '{fid}' has no `### {fid}` block in the methodology draft")
        if fid not in views_by_id:
            _die(f"planned figure '{fid}' has no rendered view in the captions JSON")
        view = _headline_view(views_by_id[fid])
        block = _methodology_block(per_result[fid], view["caption"])
        alt = " ".join(str(view["plot_name"]).split())
        out += [
            f"### {title}",
            "",
            block,
            "",
            f"![{alt}]({_raw_url(figures_sha, view['png_relpath'])})",
            "",
            "**Takeaways**",
            "",
            PLACEHOLDER,
            "",
        ]
    out += ["## Conclusion and next steps", "", PLACEHOLDER, ""]
    return "\n".join(out).rstrip("\n") + "\n"


def build_detailed(
    *,
    sections: dict[str, str],
    per_result: dict[str, str],
    manifest: dict,
    views_by_id: dict[str, list[dict]],
    figures_sha: str,
) -> str:
    out: list[str] = [
        f"# Detailed writeup — issue {ISSUE}: {H1_QUESTION}",
        "",
        "*(auto-generated companion to the report body; all content agent-written +"
        " factual — claims live in the report body only. Regenerated wholesale by"
        " `scripts/issue2162_assemble_report.py` on every round; do not hand-edit.)*",
        "",
        "## Motivation",
        "",
        sections["Motivation"].strip("\n"),
        "",
        "## Methodology (full)",
        "",
        sections["Methodology (shared)"].strip("\n"),
        "",
        "## Results — full figure set",
        "",
        "Every view the plotter rendered, including the per-unit and alternate-slot"
        " companions the summarized body does not carry. Each image is pinned at the"
        f" figures commit `{figures_sha}`.",
        "",
    ]
    for fig in manifest["figures"]:
        fid = str(fig["id"]).strip()
        title = str(fig["title"]).strip()
        views = views_by_id[fid]
        out += [f"### {title}", "", f"Planned figure id: `{fid}`.", "", "**Methodology**", ""]
        out += [per_result[fid].strip("\n"), ""]
        for v in views:
            cap = " ".join(str(v["caption"]).split())
            alt = " ".join(str(v["plot_name"]).split())
            out += [
                f"**View: {v['view']} — {alt}**",
                "",
                f"> {cap}",
                "",
                f"![{alt}]({_raw_url(figures_sha, v['png_relpath'])})",
                "",
            ]
    return "\n".join(out).rstrip("\n") + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", required=True, type=Path)
    ap.add_argument("--captions", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--figures-sha", required=True)
    ap.add_argument(
        "--detailed-sha",
        default="",
        help="40-hex commit of the committed detailed companion; omitted => "
        "an invalid sentinel so the body fails verification loudly",
    )
    ap.add_argument("--out-body", required=True, type=Path)
    ap.add_argument("--out-detailed", required=True, type=Path)
    args = ap.parse_args()

    for sha_name, sha in (("--figures-sha", args.figures_sha),):
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            _die(f"{sha_name} must be a 40-hex commit sha, got {sha!r}")
    detailed_sha = args.detailed_sha or "PENDING-DETAILED-COMMIT"
    if args.detailed_sha and not re.fullmatch(r"[0-9a-f]{40}", args.detailed_sha):
        _die(f"--detailed-sha must be a 40-hex commit sha, got {args.detailed_sha!r}")

    sections_text = args.sections.read_text()
    sections = _split_h2(sections_text)
    for need in ("Motivation", "Methodology (shared)"):
        if need not in sections:
            _die(f"methodology draft has no `## {need}` section")

    per_result_h2 = next(
        (v for k, v in sections.items() if k.lower().startswith("per-result")),
        None,
    )
    if per_result_h2 is None:
        _die("methodology draft has no `## Per-result ...` section")
    per_result = _split_h3(per_result_h2)

    manifest = json.loads(args.manifest.read_text())
    caps = json.loads(args.captions.read_text())
    views_by_id: dict[str, list[dict]] = {}
    for rec in caps:
        views_by_id.setdefault(str(rec["manifest_figure_id"]).strip(), []).append(rec)

    planned = {str(f["id"]).strip() for f in manifest["figures"]}
    orphan_views = sorted(set(views_by_id) - planned)
    if orphan_views:
        _die(f"captions name figure ids absent from the manifest: {orphan_views}")

    _assert_crosswalk_grounded(sections_text)

    body = build_body(
        sections=sections,
        per_result=per_result,
        manifest=manifest,
        views_by_id=views_by_id,
        figures_sha=args.figures_sha,
        detailed_sha=detailed_sha,
    )
    detailed = build_detailed(
        sections=sections,
        per_result=per_result,
        manifest=manifest,
        views_by_id=views_by_id,
        figures_sha=args.figures_sha,
    )

    # The lexicon gate applies to agent-written sections. Motivation is exempt
    # by the template (hypothesis framing is allowed); Methodology + Results
    # are scanned, so scan exactly what lands in them.
    _assert_clean("Methodology (shared)", sections["Methodology (shared)"])
    _assert_clean(
        "Results (per-result blocks + folded captions)",
        "\n".join(
            _methodology_block(per_result[str(f["id"]).strip()], v["caption"])
            for f in manifest["figures"]
            for v in views_by_id[str(f["id"]).strip()]
        ),
    )

    args.out_body.parent.mkdir(parents=True, exist_ok=True)
    args.out_detailed.parent.mkdir(parents=True, exist_ok=True)
    args.out_body.write_text(body)
    args.out_detailed.write_text(detailed)

    n_views = sum(len(v) for v in views_by_id.values())
    print(
        f"wrote body ({len(body.splitlines())} lines, {len(manifest['figures'])} results) "
        f"-> {args.out_body}"
    )
    print(
        f"wrote detailed ({len(detailed.splitlines())} lines, {n_views} views) "
        f"-> {args.out_detailed}"
    )
    if not args.detailed_sha:
        print("NOTE: --detailed-sha omitted; body carries the PENDING sentinel (will fail verify)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
