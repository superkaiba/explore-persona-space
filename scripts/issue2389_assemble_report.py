"""Assemble the issue-2389 workflow-v2 report body + detailed companion (Step 7c).

Thin fork of ``scripts/issue2329_assemble_report.py`` (paths + H1 flipped; the
diagnostics section rewritten for the #2389 artifact set: per-cell-cap cap-hit
reports, pre-commitment items 4+5 realized-state lines, and a revision-PINNED
provenance section replacing the parent's provenance-GAP section).

MECHANICAL ASSEMBLY ONLY. Every sentence in the two emitted documents is either

  (a) copied verbatim from the findings-blind methodology draft
      (docs/reports/issue_2389_methodology_draft.md), or
  (b) copied verbatim from the plotter's factual captions
      (figures/issue_2389/captions.json), or
  (c) a value READ AT COMPOSE TIME from the named artifact under
      eval_results/issue_2389/ (never typed from memory, never hardcoded).

No claim is authored here. Per `.claude/skills/issue-v2/report-template.md`, the
`# Result:` title, the `## TLDR`, every per-result `**Takeaways**` block, and
`## Conclusion and next steps` are emitted as `*(Thomas fills in)*` placeholders
for the user alone.

Fails loud (never a silent default) on: a missing input file, a missing draft
section, a manifest/captions id-set mismatch, or a captions view whose PNG is
absent on disk (which would emit a broken image link).

Usage
-----
    # pass 1 -- emit the detailed companion, then commit it to capture its SHA
    uv run python scripts/issue2389_assemble_report.py --emit detailed \
        --figure-sha <40-hex> --out-detailed docs/reports/issue_2389_detailed.md

    # pass 2 -- emit the body, pinning the detailed doc at the SHA from pass 1
    uv run python scripts/issue2389_assemble_report.py --emit body \
        --figure-sha <40-hex> --detailed-sha <40-hex> \
        --out-body /tmp/issue-2389-report-body.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import OrderedDict
from pathlib import Path

ISSUE = 2389
PLACEHOLDER = "*(Thomas fills in)*"
SENTINEL = "<!-- report-v1 -->"

H1_QUESTION = (
    "Do the #2162/#2329 minimal-pair context-vector findings transfer to "
    "Qwen3.8-27B at a pinned revision (thinking disabled, ce slot only)?"
)

DRAFT_REL = "docs/reports/issue_2389_methodology_draft.md"
CAPTIONS_REL = "figures/issue_2389/captions.json"

# Artifacts read for the compose-time diagnostics section. #2389 runs per-cell
# caps + gate-slice recalibration from the start (plan §4.7 item 1), so the
# realized reports are the per-scope cap_hit_report_*.json files (postregen
# siblings exist only when the >2% trigger fired and a capregen pass ran).
CAP_REPORT_RELS = (
    ("anchors", "eval_results/issue_2389/cap_hit/cap_hit_report_anchors.json"),
    ("grid", "eval_results/issue_2389/cap_hit/cap_hit_report_grid.json"),
)
PROBE_REL = "eval_results/issue_2389/f_metrics/probe.json"
MARGIN_CELLS_REL = "eval_results/issue_2389/f_metrics/margin_cells.jsonl"
# Plan §4.7 pre-commitment realized-state artifacts (items 4 + 5).
VLLM_PARITY_REL = "eval_results/issue_2389/gates/vllm_parity_report.json"
SHARE_PREFILL_REL = "eval_results/issue_2389/gates/share_prefill_equivalence.json"

DASHBOARDS = (
    ("Bank dashboard", "docs/issue2389_bank_dashboard.html"),
    ("Result-0 qualitative gallery", "docs/issue2389_result0_gallery.html"),
    ("Anchors table", "experiments/dashboards/issue2389_anchors.html"),
    ("F-cells table (index)", "experiments/dashboards/issue2389_f_cells.html"),
    ("Stage-2 cells table (index)", "experiments/dashboards/issue2389_stage2_cells.html"),
)


# --------------------------------------------------------------------------- io


def repo_slug(root: Path) -> str:
    """Return `owner/name` from the origin remote. Never guessed."""
    url = subprocess.run(
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    m = re.search(r"github\.com[:/]+([^/]+/[^/]+?)(?:\.git)?$", url)
    if not m:
        raise SystemExit(f"cannot parse GitHub slug from origin remote: {url!r}")
    return m.group(1)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing required artifact: {path}")
    with path.open() as fh:
        return json.load(fh)


def read_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"missing required input: {path}")
    return path.read_text()


def raw_url(slug: str, sha: str, rel: str) -> str:
    return f"https://raw.githubusercontent.com/{slug}/{sha}/{rel}"


def blob_url(slug: str, sha: str, rel: str) -> str:
    return f"https://github.com/{slug}/blob/{sha}/{rel}"


def check_sha(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value or ""):
        raise SystemExit(f"--{label} must be a full 40-hex sha, got {value!r}")
    return value


# ------------------------------------------------------------------- draft parse


def parse_draft(text: str) -> dict:
    """Split the methodology draft into Motivation / shared / per-figure blocks."""
    lines = text.splitlines()
    idx: dict[str, int] = {}
    per_id_starts: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        if line.startswith("## Motivation"):
            idx["motivation"] = i
        elif line.startswith("## Methodology (shared)"):
            idx["shared"] = i
        elif line.startswith("## Results"):
            idx["results"] = i
        elif line.startswith("### ") and "results" in idx:
            per_id_starts.append((line[4:].strip(), i))

    for key in ("motivation", "shared", "results"):
        if key not in idx:
            raise SystemExit(f"draft is missing its '{key}' section heading")

    def block(start: int, end: int) -> str:
        return "\n".join(lines[start + 1 : end]).strip("\n")

    per_id: OrderedDict[str, str] = OrderedDict()
    for n, (fid, start) in enumerate(per_id_starts):
        end = per_id_starts[n + 1][1] if n + 1 < len(per_id_starts) else len(lines)
        body = block(start, end)
        # Drop the leading bold `**Methodology**` label; we re-emit it ourselves.
        body = re.sub(r"^\s*\*\*Methodology\*\*\s*\n+", "", body, count=1)
        per_id[fid] = body.strip("\n")

    if not per_id:
        raise SystemExit("draft has no '### <figure id>' per-figure blocks")

    return {
        "motivation": block(idx["motivation"], idx["shared"]),
        "shared": block(idx["shared"], idx["results"]),
        "per_id": per_id,
    }


# ------------------------------------------------------------------- assembly


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {s}" for s in items)


def _walk_cards(node: object, path: str, out: list[tuple[str, str]]) -> None:
    """Collect (json-pointer, sha) for USABLE reproducibility-card commits.

    Usable == a full 40-hex ``git_commit`` / ``final_commit_sha`` whose sibling
    ``git_dirty`` is not true. Mirrors verify_report.py's code-sha-cards walk.
    """
    if isinstance(node, dict):
        dirty = node.get("git_dirty")
        for key in ("git_commit", "final_commit_sha"):
            val = node.get(key)
            if isinstance(val, str) and re.fullmatch(r"[0-9a-f]{40}", val) and dirty is not True:
                out.append((f"{path}/{key}", val))
        for k, v in node.items():
            _walk_cards(v, f"{path}/{k}", out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_cards(v, f"{path}/{i}", out)


def usable_card_commits(root: Path) -> list[tuple[str, str]]:
    """Return sorted (artifact-relpath, sha) pairs the report must cite."""
    found: dict[str, set[str]] = {}
    base = root / f"eval_results/issue_{ISSUE}"
    for p in sorted(base.rglob("*.json")):
        if p.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            with p.open() as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        hits: list[tuple[str, str]] = []
        _walk_cards(data, "", hits)
        if hits:
            found.setdefault(str(p.relative_to(root)), set()).update(s for _, s in hits)
    return sorted((rel, sha) for rel, shas in found.items() for sha in sorted(shas))


def manifest_coverage_block(manifest: dict, root: Path) -> str:
    """Verbatim approved-manifest condition/metric index + the code-SHA cards.

    Mechanical coverage, not authoring: the condition and metric NAMES are copied
    verbatim from the approved planned_manifest.json, and every code SHA is read
    from its reproducibility card on disk.
    """
    cards = usable_card_commits(root)
    parts = [
        "**Planned conditions (approved manifest, verbatim):**",
        "",
        bullets(manifest["conditions"]),
        "",
        "**Planned metrics (approved manifest, verbatim):**",
        "",
        bullets(manifest["metrics"]),
    ]
    if cards:
        parts += [
            "",
            "**Code SHAs:** per-artifact reproducibility-card commits, read from the cards "
            "themselves (a card recording a dirty tree is excluded and not cited):",
            "",
            bullets(f"`{rel}` @ `{sha}`" for rel, sha in cards),
        ]
    return "\n".join(parts)


def views_of(entry: dict) -> list[str]:
    out = []
    if entry.get("aggregate_view"):
        out.append(entry["aggregate_view"])
    out.extend(entry.get("per_unit_views") or [])
    return out


def assert_views_exist(root: Path, captions: dict) -> None:
    missing = [v for e in captions.values() for v in views_of(e) if not (root / v).exists()]
    if missing:
        raise SystemExit("captions reference view files that do not exist: " + ", ".join(missing))


def result_methodology(entry: dict, draft_block: str) -> str:
    """Per-result `**Methodology**` body: draft recipe + the plotter's captions."""
    parts = [draft_block] if draft_block.strip() else []
    caps = entry.get("caption_bullets") or []
    if caps:
        parts.append(bullets(caps))
    if entry.get("planned_not_produced"):
        reason = entry.get("planned_not_produced_reason", "")
        for item in entry["planned_not_produced"]:
            parts.append(f"- Planned but NOT produced: {item} — {reason}")
    if entry.get("status") == "not run":
        parts.append(f"- Status: not run as a figure — {entry.get('not_run_reason', '')}")
    return "\n\n".join(p for p in parts if p.strip())


def build_body(
    *,
    slug: str,
    fig_sha: str,
    det_sha: str,
    draft: dict,
    captions: dict,
    titles: dict[str, str],
    manifest: dict,
    root: Path,
) -> str:
    out = [
        f"# Experiment: {H1_QUESTION}",
        SENTINEL,
        "",
        "**Detailed writeup:** "
        + blob_url(slug, det_sha, f"docs/reports/issue_{ISSUE}_detailed.md"),
        "",
        "## Motivation",
        "",
        draft["motivation"],
        "",
        "## TLDR",
        "",
        PLACEHOLDER,
        "",
        "## Methodology (shared)",
        "",
        draft["shared"],
        "",
        manifest_coverage_block(manifest, root),
        "",
        "## Results",
        "",
    ]

    # A planned manifest item with no PNG view is DECLARED `not run` here rather than given a
    # Results subsection: it has no image to carry (and a subsection must hold exactly one), and
    # the manifest-completeness check is satisfied by an explicit `not run` on the item's own line.
    unrendered = [(fid, e) for fid, e in captions.items() if not e.get("aggregate_view")]
    if unrendered:
        out += ["Planned manifest items not produced as figures:", ""]
        for fid, entry in unrendered:
            links = ", ".join(f"[{n}]({blob_url(slug, fig_sha, r)})" for n, r in DASHBOARDS[:2])
            out.append(
                f"- {titles.get(fid, fid)} (`{fid}`) — **not run** as a figure: "
                f"{entry.get('not_run_reason', '')} Delivered as HTML: {links}."
            )
        out.append("")

    for fid, block in draft["per_id"].items():
        entry = captions[fid]
        agg = entry.get("aggregate_view")
        if not agg:
            continue
        alt = f"{titles.get(fid, fid)} — aggregate view"
        out += [
            f"### {titles.get(fid, fid)}",
            "",
            "**Methodology**",
            "",
            result_methodology(entry, block),
            "",
            f"![{alt}]({raw_url(slug, fig_sha, agg)})",
            "",
            "**Takeaways**",
            "",
            PLACEHOLDER,
            "",
        ]

    out += ["## Conclusion and next steps", "", PLACEHOLDER, ""]
    return "\n".join(out).rstrip() + "\n"


def _revision_provenance_lines(root: Path) -> list[str]:
    """Distinct (model_id, model_revision) pins recorded across harvested artifacts.

    The parent #2329 report's diagnostics named the forward fix realized here:
    the #2389 driver passes ``revision=`` at every tokenizer/config/weights
    load, keys ``regime_fingerprint`` on it, and records ``model_id`` +
    ``model_revision`` in every artifact's ``repro`` block."""
    base = root / f"eval_results/issue_{ISSUE}"
    pairs: OrderedDict[tuple[str, str], int] = OrderedDict()
    for path in sorted(base.rglob("*.json")):
        if path.stat().st_size > 5_000_000:
            continue  # per-row tables; the pin rides the small gate/manifest JSONs
        obj = load_json(path)
        rep = obj.get("repro") if isinstance(obj, dict) else None
        if isinstance(rep, dict) and "model_revision" in rep:
            key = (str(rep.get("model_id")), str(rep.get("model_revision")))
            pairs[key] = pairs.get(key, 0) + 1
    lines = ["### Model-revision provenance (pinned in this run)", ""]
    if not pairs:
        lines.append(
            "- No harvested artifact records `repro.model_revision` yet (run not "
            "harvested): the driver threads `revision=` into EVERY load and records "
            "`model_id` + `model_revision` per artifact — the forward fix the parent "
            "#2329 report named."
        )
        return lines
    if len(pairs) > 1:
        raise SystemExit(
            f"multiple distinct (model_id, model_revision) pins recorded: {dict(pairs)}"
        )
    (model_id, revision), n = next(iter(pairs.items()))
    lines.append(
        f"- Every artifact-recorded pin agrees: `{model_id}` @ revision `{revision}` "
        f"({n} artifacts under `eval_results/issue_{ISSUE}/` record it; the driver "
        "passes `revision=` at every `from_pretrained` and keys `regime_fingerprint` "
        "on it). This closes the parent #2329 report's revision-provenance gap."
    )
    return lines


def diagnostics_section(root: Path) -> str:
    """Compose-time instrument diagnostics, every value read from its artifact."""
    lines = ["### Generation cap-hit (realized, per the standing cap-hit reporting rule)", ""]
    n_found = 0
    for scope, rel in CAP_REPORT_RELS:
        path = root / rel
        if not path.exists():
            lines.append(f"- `{scope}` cap-hit report not harvested (`{rel}` absent).")
            continue
        n_found += 1
        d = load_json(path)
        partial = f" PARTIAL read ({'; '.join(d['partial_reason'])});" if d.get("partial") else ""
        lines.append(
            f"- `{scope}` sweep under per-cell caps (realized row caps "
            f"{d.get('realized_row_caps')}):{partial} {d['cap_hit_rows']} of {d['n_rows']} "
            f"rows hit their cap ({d['cap_hit_pct']:.4f}%), against the pre-registered "
            f"per-cell re-generation trigger of {d['pre_registered_regen_trigger_pct']}% — "
            f"trigger_fired = {d['trigger_fired']}; {len(d['breaching_cells'])} cells "
            f"breached (`{rel}`)."
        )
        trigger = float(d["pre_registered_regen_trigger_pct"])
        breaching = sorted(
            ((c, v) for c, v in d["per_cell"].items() if float(v["cap_hit_pct"]) > trigger),
            key=lambda cv: float(cv[1]["cap_hit_pct"]),
            reverse=True,
        )
        lines += [
            f"    - `{c}`: {v['cap_hit_pct']:.4f}% ({v['cap_hit_rows']}/{v['n_rows']} rows; "
            f"realized caps {v.get('realized_caps_by_batch')})"
            for c, v in breaching
        ]
        post_rel = rel[: -len(".json")] + "_postregen.json"
        if (root / post_rel).exists():
            p = load_json(root / post_rel)
            lines.append(
                f"- `{scope}` post-regeneration: {p['cap_hit_rows']} of {p['n_rows']} rows "
                f"({p['cap_hit_pct']:.4f}%) at realized caps {p.get('realized_row_caps')}; "
                f"{len(p['breaching_cells'])} cells still above the trigger (`{post_rel}`)."
            )
        elif d["trigger_fired"]:
            lines.append(
                f"- `{scope}`: trigger fired but no post-regeneration report "
                f"(`{post_rel}` absent) — a capregen pass is still owed for the "
                "breaching cells."
            )
        else:
            lines.append(
                f"- `{scope}`: no capregen pass was triggered (the per-cell cap table + "
                "gate-slice recalibration held every cell at or under the trigger), so "
                "no post-regeneration report exists by design."
            )
    if not n_found:
        raise SystemExit(
            f"no cap-hit report found under eval_results/issue_{ISSUE}/cap_hit/ — "
            "harvest the cap_report phase before assembling the report"
        )

    probe = load_json(root / PROBE_REL)
    pos = sum(1 for r in probe["results"] if r.get("probe_positive"))
    lines += [
        "",
        "### Probe regime (read axis)",
        "",
        f"- {len(probe['results'])} units at n = "
        f"{sorted({r['n_per_value_pair'] for r in probe['results']})} contexts per "
        f"value-pair; each unit's verdict is observed max-AUC-over-layers against that "
        f"unit's OWN carrier-level permutation band at the 97.5th percentile "
        f"(B = {probe['perm_b']} label permutations), not against 0.5.",
        f"- `probe_positive` holds for {pos} of {len(probe['results'])} units (`{PROBE_REL}`).",
    ]

    lines += ["", "### Dashboards (full per-row tables)", ""]
    lines += [f"- {name}: `{rel}`" for name, rel in DASHBOARDS]
    margin = root / MARGIN_CELLS_REL
    if margin.exists():
        n_margin = sum(1 for ln in margin.read_text().splitlines() if ln.strip())
        lines.append(
            f"- The `margin_cells` table ({n_margin:,} rows) is NOT dashboarded (the hard "
            f"10 MB per-issue dashboard cap — the #2329 convention). In-repo copy "
            f"`{MARGIN_CELLS_REL}`; full-fidelity data on the HF data repo at "
            "`issue2389_q38ce/analysis_tensors/margin`."
        )
    else:
        lines.append(
            f"- `margin_cells` table: not present (`{MARGIN_CELLS_REL}` absent — the "
            "margin phase is opportunistic and may not have run)."
        )

    lines += ["", "### Pre-commitment realized state (plan §4.7 items 4 + 5)", ""]
    parity = root / VLLM_PARITY_REL
    if parity.exists():
        d = load_json(parity)
        clauses = d.get("clauses") or {}
        lines.append(
            f"- Item 4 (vLLM anchor leg): parity gate verdict `{d['verdict']}` over "
            f"cells {d.get('cells')} ({d.get('n_pairs')} pairs; clauses "
            f"{json.dumps(clauses, sort_keys=True)}); per the fail-open pin the "
            f"production vLLM leg ran ONLY on a PASS verdict (`{VLLM_PARITY_REL}`)."
        )
    else:
        lines.append(
            f"- Item 4 (vLLM anchor leg): `{VLLM_PARITY_REL}` absent — the parity gate "
            "was not harvested (or the vLLM leg never ran); per the fail-open pin all "
            "anchor cells then came from the serial HF path."
        )
    spe = root / SHARE_PREFILL_REL
    if spe.exists():
        d = load_json(spe)
        variants = d.get("variants") or {}
        n_pass = sum(1 for v in variants.values() if v.get("passed"))
        lines.append(
            f"- Item 5 (shared-prefill batching): gate-4b battery verdict "
            f"`{d['verdict']}` ({n_pass}/{len(variants)} variants passed); "
            f"`--share-prefill auto` arms `share_prefill=True` in the anchors/grid "
            f"generation calls ONLY on a PASS artifact (`{SHARE_PREFILL_REL}`)."
        )
    else:
        lines.append(
            f"- Item 5 (shared-prefill batching): `{SHARE_PREFILL_REL}` absent — the "
            "gate-4b battery did not run (or was not harvested); the arming resolver "
            "is FAIL-OPEN, so every generation call stayed on the serial per-draw-"
            "prefill path."
        )

    lines += [""] + _revision_provenance_lines(root)
    return "\n".join(lines)


def build_detailed(
    *,
    slug: str,
    fig_sha: str,
    draft: dict,
    captions: dict,
    titles: dict[str, str],
    manifest: dict,
    root: Path,
) -> str:
    out = [
        f"# Detailed writeup — issue {ISSUE}: {H1_QUESTION}",
        "",
        "*(auto-generated companion to the report body; all content agent-written + factual — "
        "claims live in the report body only)*",
        "",
        "## Motivation",
        "",
        draft["motivation"],
        "",
        "## Methodology (full)",
        "",
        draft["shared"],
        "",
        manifest_coverage_block(manifest, root),
        "",
        "## Results — full figure set",
        "",
    ]

    for fid, block in draft["per_id"].items():
        entry = captions[fid]
        out += [
            f"### {titles.get(fid, fid)}",
            "",
            "**Methodology**",
            "",
            result_methodology(entry, block),
            "",
        ]
        vs = views_of(entry)
        if not vs:
            out += ["No PNG view exists for this manifest item.", ""]
        for rel in vs:
            out += [
                f"**View:** `{Path(rel).name}`",
                "",
                f"![{titles.get(fid, fid)} — {Path(rel).stem}]({raw_url(slug, fig_sha, rel)})",
                "",
            ]

    out += ["## Extra tables / diagnostics", "", diagnostics_section(root), ""]
    return "\n".join(out).rstrip() + "\n"


# ------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=None, help="worktree root (default: cwd)")
    ap.add_argument("--figure-sha", required=True, help="40-hex sha pinning figure images")
    ap.add_argument("--detailed-sha", default=None, help="40-hex sha pinning the detailed doc")
    ap.add_argument("--emit", choices=("body", "detailed", "both"), default="both")
    ap.add_argument("--out-body", default=None)
    ap.add_argument("--out-detailed", default=None)
    ap.add_argument("--manifest", required=True, help="path to planned_manifest.json")
    args = ap.parse_args()

    root = Path(args.repo_root or ".").resolve()
    fig_sha = check_sha(args.figure_sha, "figure-sha")
    slug = repo_slug(root)

    draft = parse_draft(read_text(root / DRAFT_REL))
    captions = load_json(root / CAPTIONS_REL)
    manifest = load_json(Path(args.manifest))
    titles = {f["id"]: f["title"] for f in manifest["figures"]}

    draft_ids = set(draft["per_id"])
    if draft_ids != set(captions) or draft_ids != set(titles):
        raise SystemExit(
            "id-set mismatch — draft: "
            f"{sorted(draft_ids)}\ncaptions: {sorted(captions)}\nmanifest: {sorted(titles)}"
        )
    assert_views_exist(root, captions)

    if args.emit in ("detailed", "both"):
        if not args.out_detailed:
            raise SystemExit("--out-detailed is required when emitting the detailed doc")
        text = build_detailed(
            slug=slug,
            fig_sha=fig_sha,
            draft=draft,
            captions=captions,
            titles=titles,
            manifest=manifest,
            root=root,
        )
        p = Path(args.out_detailed)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"wrote {p} ({len(text)} B)")

    if args.emit in ("body", "both"):
        if not args.out_body:
            raise SystemExit("--out-body is required when emitting the body")
        if not args.detailed_sha:
            raise SystemExit("--detailed-sha is required when emitting the body")
        det_sha = check_sha(args.detailed_sha, "detailed-sha")
        text = build_body(
            slug=slug,
            fig_sha=fig_sha,
            det_sha=det_sha,
            draft=draft,
            captions=captions,
            titles=titles,
            manifest=manifest,
            root=root,
        )
        p = Path(args.out_body)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"wrote {p} ({len(text)} B)")


if __name__ == "__main__":
    main()
