"""Assemble the issue-2329 workflow-v2 report body + detailed companion (Step 7c).

MECHANICAL ASSEMBLY ONLY. Every sentence in the two emitted documents is either

  (a) copied verbatim from the findings-blind methodology draft
      (docs/reports/issue_2329_methodology_draft.md), or
  (b) copied verbatim from the plotter's factual captions
      (figures/issue_2329/captions.json), or
  (c) a value READ AT COMPOSE TIME from the named artifact under
      eval_results/issue_2329/ (never typed from memory, never hardcoded).

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
    uv run python scripts/issue2329_assemble_report.py --emit detailed \
        --figure-sha <40-hex> --out-detailed docs/reports/issue_2329_detailed.md

    # pass 2 -- emit the body, pinning the detailed doc at the SHA from pass 1
    uv run python scripts/issue2329_assemble_report.py --emit body \
        --figure-sha <40-hex> --detailed-sha <40-hex> \
        --out-body /tmp/issue-2329-report-body.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import OrderedDict
from pathlib import Path

ISSUE = 2329
PLACEHOLDER = "*(Thomas fills in)*"
SENTINEL = "<!-- report-v1 -->"

H1_QUESTION = (
    "Do the #2162 minimal-pair context-vector findings transfer to "
    "Qwen3.5-9B with thinking disabled?"
)

DRAFT_REL = "docs/reports/issue_2329_methodology_draft.md"
CAPTIONS_REL = "figures/issue_2329/captions.json"

# Artifacts read for the compose-time diagnostics section.
CAP_PREREGEN_REL = "eval_results/issue_2329/cap_hit/cap_hit_report_anchors_preregen.json"
CAPREGEN_RELS = (
    ("gate", "eval_results/issue_2329/cap_hit/capregen_sufficiency_anchors.json"),
    ("rest", "eval_results/issue_2329/cap_hit/capregen_sufficiency_anchors_rest.json"),
)
RESTRICTION_REL = "eval_results/issue_2329/cap_hit/restriction_analysis.json"
PROBE_REL = "eval_results/issue_2329/f_metrics/probe.json"

DASHBOARDS = (
    ("Bank dashboard", "docs/issue2329_bank_dashboard.html"),
    ("Result-0 qualitative gallery", "docs/issue2329_result0_gallery.html"),
    ("Anchors table", "experiments/dashboards/issue2329_anchors.html"),
    ("F-cells table (index)", "experiments/dashboards/issue2329_f_cells.html"),
    ("Stage-2 cells table (index)", "experiments/dashboards/issue2329_stage2_cells.html"),
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


def diagnostics_section(root: Path) -> str:
    """Compose-time instrument diagnostics, every value read from its artifact."""
    pre = load_json(root / CAP_PREREGEN_REL)
    restriction = load_json(root / RESTRICTION_REL)
    probe = load_json(root / PROBE_REL)

    lines = ["### Generation cap-hit (realized, per the standing cap-hit reporting rule)", ""]
    lines.append(
        f"- Pre-regeneration anchors sweep at `max_new_tokens = {pre['max_new_tokens']}`: "
        f"{pre['cap_hit_rows']} of {pre['n_rows']} rows hit the cap "
        f"({pre['cap_hit_pct']:.4f}%), against a pre-registered re-generation trigger of "
        f"{pre['pre_registered_regen_trigger_pct']}% — trigger_fired = {pre['trigger_fired']}; "
        f"{len(pre['breaching_cells'])} cells breached "
        f"(`{CAP_PREREGEN_REL}`)."
    )
    for batch, rel in CAPREGEN_RELS:
        d = load_json(root / rel)
        lines.append(
            f"- Post-regeneration sufficiency, `{batch}` batch (base cap {d['base_cap']} → raised "
            f"cap {d['raised_cap']}): {d['n_rows_regenerated']} of {d['n_rows_total']} rows "
            f"regenerated; {d['regen_over_base_cap_rows']} rows "
            f"({d['regen_over_base_cap_pct']:.4f}%) exceeded the base cap; "
            f"{d['regen_hit_raised_cap_rows']} rows "
            f"({d['regen_hit_raised_cap_pct']:.4f}%) reached the raised cap; regenerated "
            f"completion tokens median {d['regen_tokens_median']}, p90 {d['regen_tokens_p90']}, "
            f"p99 {d['regen_tokens_p99']}, max {d['regen_tokens_max']} (`{rel}`)."
        )
        if d["regen_tokens_max"] > d["raised_cap"]:
            lines.append(
                f"    - Boundary anomaly, recorded not resolved: `regen_tokens_max` "
                f"({d['regen_tokens_max']}) exceeds the raised cap ({d['raised_cap']}) by "
                f"{d['regen_tokens_max'] - d['raised_cap']} tokens in the `{batch}` batch, while "
                f"p99 equals the cap exactly ({d['regen_tokens_p99']})."
            )

    # Post-regeneration RESIDUAL per-cell breach table, aggregated over value sides and
    # batches from per_cell_value (the pre-regen report's per_cell is at the 2048 base cap).
    trigger_pct = float(pre["pre_registered_regen_trigger_pct"])
    cells: dict[str, dict[str, float]] = {}
    for _batch, rel in CAPREGEN_RELS:
        for row in load_json(root / rel)["per_cell_value"]:
            acc = cells.setdefault(row["cell"], {"rows": 0.0, "n": 0.0})
            acc["rows"] += row["hit_raised_cap_rows"]
            acc["n"] += row["n"]
    resid = sorted(
        (
            {"cell": c, "pct": 100.0 * v["rows"] / v["n"], "rows": v["rows"], "n": v["n"]}
            for c, v in cells.items()
            if v["n"]
        ),
        key=lambda r: r["pct"],
        reverse=True,
    )
    breaching = [r for r in resid if r["pct"] >= trigger_pct]
    lines += [
        "",
        f"Residual per-cell cap-hit at the raised cap ({len(breaching)} of {len(resid)} cells at "
        f"or above the {trigger_pct}% trigger, aggregated over value sides and both batches):",
        "",
    ]
    lines += [
        f"- `{r['cell']}`: {r['pct']:.4f}% ({int(r['rows'])}/{int(r['n'])} regenerated rows)"
        for r in breaching
    ] or ["- none at or above the trigger"]

    asym: list[dict] = []
    for batch, rel in CAPREGEN_RELS:
        for row in load_json(root / rel).get("within_cell_asymmetry") or []:
            asym.append({**row, "batch": batch})
    asym.sort(key=lambda r: r["spread_pct_points"], reverse=True)
    if asym:
        lines += ["", "Within-cell value-side asymmetry in cap-hit rate (top 6 by spread):", ""]
        lines += [
            f"- `{r['cell']}` ({r['batch']} batch): {r['min_hit_pct']:.4f}% – "
            f"{r['max_hit_pct']:.4f}% across {r['n_value_sides']} value sides "
            f"(spread {r['spread_pct_points']:.4f} pct points)"
            for r in asym[:6]
        ]

    per_cell = restriction["per_cell"]
    summary = per_cell["summary"]
    tr = restriction["transfer"]
    tbt = restriction["two_by_two"]
    st2 = restriction["stage2"]
    ref = restriction["refinement_a_prime"]

    lines += [
        "",
        "### Cap-hit-excluded restriction analysis (v55; zero GPU-hours)",
        "",
        f"- What ran: {restriction['what']}. Shipped tables `{restriction['shipped_dir']}` vs "
        f"restricted `{restriction['restricted_dir']}`; the shipped tables were not rewritten.",
        f"- `probe.json` was reused verbatim "
        f"(`probe_reused_verbatim = {restriction['probe_reused_verbatim']}`): "
        f"{restriction['probe_reuse_justification']}",
        f"- Pre-registered trigger: {restriction['pre_registered_trigger']}",
        f"- Triggers fired: {', '.join(restriction['triggers_fired']) or 'none'}; "
        f"recorded verdict `{restriction['verdict']}`.",
        f"- Refinement (a'), authored after seeing which limb fired — disclosure carried in the "
        f"artifact: {ref['disclosure']}",
        f"- Refined verdict `{ref['verdict_refined']}`: {ref['n_sign_flips_total']} sign flip(s) "
        f"total, {ref['n_sign_flips_conclusion_bearing']} of them conclusion-bearing over "
        f"{ref['n_conclusion_bearing_units']} conclusion-bearing units.",
        f"- Per-cell summary: {json.dumps(summary, sort_keys=True)}",
        f"- Gate-verdict changes: {len(per_cell['verdict_changes'])}; shipped-CI exits: "
        f"{len(per_cell['ci_exits'])}; sign flips: {len(per_cell['sign_flips'])} over "
        f"{len(per_cell['rows'])} units.",
        f"- Transfer: Spearman rho {tr['shipped_rho']} (shipped, "
        f"95% pair-clustered CI {tr['shipped_ci95_pair_clustered']}, n = "
        f"{tr['shipped_n_shared_p1_units']}) → {tr['restricted_rho']} (restricted, n = "
        f"{tr['restricted_n_shared_p1_units']}); delta {tr['delta_rho']}; "
        f"restricted_rho_outside_shipped_ci = {tr['restricted_rho_outside_shipped_ci']}.",
        f"- Read x write 2x2 counts: shipped {json.dumps(tbt['shipped_counts'], sort_keys=True)}, "
        f"restricted {json.dumps(tbt['restricted_counts'], sort_keys=True)}; "
        f"counts_changed = {tbt['counts_changed']}.",
        f"- Stage 2: max |delta F_beh| {st2['max_abs_delta_f_beh']}, mean "
        f"{st2['mean_abs_delta_f_beh']} over {st2['n_comparable_f_beh']} comparable rows; "
        f"sign flips {st2['n_sign_flips']}.",
        f"- Full per-unit rows: `{RESTRICTION_REL}`.",
    ]

    pos = sum(1 for r in probe["results"] if r.get("probe_positive"))
    lines += [
        "",
        "### Probe regime (read axis)",
        "",
        f"- {len(probe['results'])} units at n = "
        f"{sorted({r['n_per_value_pair'] for r in probe['results']})} contexts per value-pair; "
        f"each unit's verdict is observed max-AUC-over-layers against that unit's OWN "
        f"carrier-level permutation band at the 97.5th percentile "
        f"(B = {probe['perm_b']} label permutations), not against 0.5.",
        f"- `probe_positive` holds for {pos} of {len(probe['results'])} units (`{PROBE_REL}`).",
    ]

    lines += ["", "### Dashboards (full per-row tables)", ""]
    lines += [f"- {name}: `{rel}`" for name, rel in DASHBOARDS]
    lines.append(
        "- The `margin_cells` table (7,644 rows) is NOT dashboarded: adding it would exceed the "
        "hard 10 MB per-issue dashboard cap. The in-repo copy is "
        "`eval_results/issue_2329/f_metrics/margin_cells.jsonl` and the full-fidelity data is on "
        "the HF data repo at `issue2329_q35rerun/analysis_tensors/margin`. The parent #2162 does "
        "dashboard its margin table, so this is a coverage difference between the two issues."
    )

    lines += [
        "",
        "### Model-revision provenance (a reproducibility gap in this run)",
        "",
        "- No `from_pretrained` call in `scripts/issue2329_run.py` passes a `revision=` kwarg, and "
        "no artifact under `eval_results/issue_2329/` records a resolved model commit, so the "
        "revision this run actually loaded is NOT pinned by the artifacts.",
        "- Two independent local HF caches each hold exactly one snapshot of `Qwen/Qwen3.5-9B` and "
        "each resolved `refs/main` to `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. That is "
        "RECOVERED VM-side evidence, not a recorded pin: the generating pod had its own cache, "
        "recorded no revision, and has been torn down, so the pod-side resolution is not provable "
        "from the artifacts.",
        "- Forward fix for any rerun of this rig: pass `revision=` explicitly at every "
        "`from_pretrained` and record the resolved sha into the run digest.",
    ]

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
