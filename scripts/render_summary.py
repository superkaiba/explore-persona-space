"""Render docs/SUMMARY.md from claims.yaml + repo state.

Usage
-----
    uv run python scripts/render_summary.py            # writes docs/SUMMARY.md
    uv run python scripts/render_summary.py --stdout   # prints to stdout
    uv run python scripts/render_summary.py --gist     # also pushes to the persistent
                                                       # gist whose ID is the first line
                                                       # of docs/SUMMARY.md (HTML comment)

The on-disk source of truth is ``docs/SUMMARY.md``. Its first line is an
HTML comment ``<!-- gist-id: <id> -->`` storing the persistent gist ID,
so re-runs from any machine edit the same gist in place. Static blocks
(Motivation, Long-term goals, Glossary) are pulled from
``docs/SUMMARY.template.md``; dynamic blocks (Current results, Immediate
next steps, Related work) are computed from ``docs/claims.yaml`` and
live ``gh`` output.

Designed so ``--gist`` can run inside the ``/weekly`` skill's dispatch
table as one parallel-dispatched subagent (slow-moving doc; the daily
gist does not regen this).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "docs" / "SUMMARY.md"
TEMPLATE_PATH = REPO_ROOT / "docs" / "SUMMARY.template.md"
CLAIMS_PATH = REPO_ROOT / "docs" / "claims.yaml"


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def _section(template: str, name: str) -> str:
    """Extract the body of a `<!-- NAME --> ... <!-- /NAME -->` block."""
    pattern = rf"<!--\s*{re.escape(name)}\s*-->\s*\n(.*?)\n<!--\s*/{re.escape(name)}\s*-->"
    m = re.search(pattern, template, flags=re.DOTALL)
    return m.group(1).rstrip() if m else f"_(template section `{name}` missing)_"


def load_claims() -> dict:
    """Graceful: if claims.yaml missing, return empty list so S2 can be tested
    independently of S3 (degrades 'Current results' to a TODO note)."""
    if not CLAIMS_PATH.exists():
        return {"claims": []}
    parsed = yaml.safe_load(CLAIMS_PATH.read_text()) or {}
    parsed.setdefault("claims", [])
    return parsed


def fetch_recent_clean_results(limit: int = 20) -> list[dict]:
    """Return up to ``limit`` issues with the ``clean-results`` label."""
    try:
        out = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--label",
                "clean-results",
                "--state",
                "all",
                "--json",
                "number,title,state,url,updatedAt",
                "--limit",
                str(limit),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(
            f"[render_summary] WARN: clean-results fetch failed ({exc}); "
            "rendering without recent-issues list.",
            file=sys.stderr,
        )
        return []
    return json.loads(out.stdout)


def fetch_proposed_high_priority(limit: int = 10) -> list[dict]:
    """Return up to ``limit`` `status:proposed` issues with `prio:high`."""
    try:
        out = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--label",
                "status:proposed",
                "--label",
                "prio:high",
                "--state",
                "open",
                "--json",
                "number,title,url",
                "--limit",
                str(limit),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(
            f"[render_summary] WARN: status:proposed fetch failed ({exc}); "
            "rendering without next-steps list.",
            file=sys.stderr,
        )
        return []
    return json.loads(out.stdout)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def render_motivation(template: str) -> str:
    return _section(template, "MOTIVATION")


def render_related_work() -> str:
    """Pull rows from papers/INDEX.md if it exists; else TODO note."""
    papers = REPO_ROOT / "papers" / "INDEX.md"
    if not papers.exists():
        return "_(papers/INDEX.md not yet populated; see slice 4 of #251.)_"
    # Cite arxiv-id rows (lines that start with `| [` and contain `arxiv.org/abs`).
    rows = [
        line
        for line in papers.read_text().splitlines()
        if line.startswith("| [") and "arxiv.org/abs" in line
    ]
    if not rows:
        return f"_(papers/INDEX.md present but no arxiv rows yet — see {papers}.)_"
    return (
        "Key references that frame this project. See "
        "[`papers/INDEX.md`](papers/INDEX.md) for the full list with `Use:` "
        f"annotations.\n\n_Currently indexed: {len(rows)} paper(s)._"
    )


def render_current_results(claims: list[dict], clean_results: list[dict]) -> str:
    if not claims:
        return (
            "_(claims registry empty — populate `docs/claims.yaml`; see "
            "slice 3 of #251 / `scripts/render_claims.py`.)_"
        )
    # Group by topic.
    by_topic: dict[str, list[dict]] = {}
    for c in claims:
        by_topic.setdefault(c.get("topic", "uncategorised"), []).append(c)
    out: list[str] = []
    for topic in sorted(by_topic):
        out.append(f"### {topic}")
        for c in by_topic[topic]:
            issues = (c.get("evidence") or {}).get("issues") or []
            issue_refs = ", ".join(f"#{n}" for n in issues) or "_(no linked issues)_"
            out.append(
                f"- **{c.get('id', '?')}** ({c.get('status', '?')}): "
                f"{c.get('description', '').strip()} — evidence: {issue_refs}"
            )
        out.append("")
    if clean_results:
        out.append("**Recent clean-result issues** (most recently updated):")
        for r in clean_results[:5]:
            out.append(f"- [#{r['number']} — {r['title']}]({r['url']}) ({r['state']})")
    return "\n".join(out).rstrip()


def render_immediate_next_steps(proposed: list[dict]) -> str:
    if not proposed:
        return "_(no `status:proposed` + `prio:high` issues open right now.)_"
    return "\n".join(f"- [#{r['number']} — {r['title']}]({r['url']})" for r in proposed)


def render_long_term_goals(template: str) -> str:
    return _section(template, "LONG_TERM_GOALS")


def render_glossary_static(template: str) -> str:
    """Hand-written fixed jargon glossary from docs/SUMMARY.template.md
    (NOT auto-generated from claim IDs)."""
    return _section(template, "GLOSSARY")


# ---------------------------------------------------------------------------
# Gist push
# ---------------------------------------------------------------------------


def redact(in_path: Path, out_path: Path) -> None:
    """Subprocess call (scripts/ has no __init__.py)."""
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/redact_for_gist.py",
            "--in",
            str(in_path),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def push_to_persistent_gist(body: str) -> str:
    summary_path = SUMMARY_PATH
    # Extract gist ID from the first line if present.
    m = re.search(r"<!-- gist-id: (\S+) -->", body[:200])
    gist_id = m.group(1) if m else None

    # Strip the gist-id marker before publishing — the URL already encodes
    # the ID; embedding it in the body is redundant. Also strip any trailing
    # whitespace/blank lines the marker leaves behind.
    body_for_gist = re.sub(r"<!-- gist-id: [^>]* -->[ \t]*\n*", "", body)
    tmp_in = Path("/tmp/SUMMARY.md.unredacted")
    tmp_out = Path("/tmp/SUMMARY.md")
    tmp_in.write_text(body_for_gist)
    redact(tmp_in, tmp_out)

    if gist_id:
        # Edit existing gist; -f matches the original filename.
        subprocess.run(
            ["gh", "gist", "edit", gist_id, "-f", "SUMMARY.md", str(tmp_out)],
            check=True,
        )
        return f"https://gist.github.com/{gist_id}"

    # First run — create. The basename of /tmp/SUMMARY.md becomes the gist filename.
    out = subprocess.run(
        [
            "gh",
            "gist",
            "create",
            "--public",
            "--desc",
            "explore-persona-space project SUMMARY (auto-regenerated)",
            str(tmp_out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    url = out.stdout.strip()
    new_id = url.rsplit("/", 1)[-1]
    # Persist the gist ID into docs/SUMMARY.md so the next caller (on any
    # machine) edits in place.
    summary_path.write_text(f"<!-- gist-id: {new_id} -->\n\n{body}")

    # CRITICAL: warn the user that the marker must be committed to be
    # visible from other machines. The script does NOT auto-push to avoid
    # surprising the user with side-effects (rejected per CLAUDE.md
    # "Executing actions with care" — pushes are visible to others).
    print(
        f"\n[render_summary] FIRST-RUN: gist created at {url}\n"
        f"[render_summary] WARNING: gist-id marker has been written to "
        f"docs/SUMMARY.md but is NOT yet committed.\n"
        f"[render_summary] Run the following to share with other machines:\n"
        f"    git add docs/SUMMARY.md\n"
        f"    git commit -m 'chore: persist SUMMARY gist id'\n"
        f"    git push\n"
        f"[render_summary] Until you commit + push, re-running this script "
        f"on a different machine will create a SECOND orphan gist.",
        file=sys.stderr,
    )
    return url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_body() -> str:
    template = TEMPLATE_PATH.read_text() if TEMPLATE_PATH.exists() else ""
    claims_data = load_claims()
    claims = claims_data.get("claims") or []
    clean_results = fetch_recent_clean_results()
    proposed = fetch_proposed_high_priority()

    parts = [
        "# Explore Persona Space — Project Summary",
        "",
        "_(Auto-regenerated by `scripts/render_summary.py` from "
        "`docs/SUMMARY.template.md` + `docs/claims.yaml` + live GitHub "
        "issues. Edit the template / YAML, not this file.)_",
        "",
        "## Motivation",
        render_motivation(template),
        "",
        "## Related work",
        render_related_work(),
        "",
        "## Current results",
        render_current_results(claims, clean_results),
        "",
        "## Immediate next steps",
        render_immediate_next_steps(proposed),
        "",
        "## Long-term goals",
        render_long_term_goals(template),
        "",
        "## Glossary",
        render_glossary_static(template),
        "",
    ]
    body = "\n".join(parts)

    # Preserve the gist-id HTML comment from existing docs/SUMMARY.md (if present).
    if SUMMARY_PATH.exists():
        m = re.search(r"<!-- gist-id: (\S+) -->", SUMMARY_PATH.read_text()[:200])
        if m:
            body = f"<!-- gist-id: {m.group(1)} -->\n\n{body}"
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stdout", action="store_true", help="Print body, do not write.")
    parser.add_argument(
        "--gist",
        action="store_true",
        help="Also push (or edit-in-place) the persistent project SUMMARY gist.",
    )
    args = parser.parse_args(argv)

    body = build_body()

    if args.stdout:
        sys.stdout.write(body)
        return 0
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(body)
    print(f"[render_summary] wrote {SUMMARY_PATH}")
    if args.gist:
        url = push_to_persistent_gist(body)
        print(f"[render_summary] gist URL: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
