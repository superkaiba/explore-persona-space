"""Render docs/claims.md from docs/claims.yaml.

Usage
-----
    uv run python scripts/render_claims.py --local   # write docs/claims.md
    uv run python scripts/render_claims.py --stdout  # print to stdout
    uv run python scripts/render_claims.py           # default = --local

Reads ``docs/claims.yaml``, renders Sagan experiment references from the
``evidence.experiments`` field, and emits ``docs/claims.md`` as a sortable
markdown table:
``ID | Description | Topic | Status | Evidence | Updated``.

The companion repository workflow
(``.github/workflows/render_claims.yml``) runs this with ``--local`` on
push to ``docs/claims.yaml`` or manual dispatch; the workflow commits the
regenerated ``docs/claims.md`` if it changes. Sagan is the workflow source of
truth; this renderer does not query repository issues.

This script is the rendering layer ONLY. The schema of
``docs/claims.yaml`` and the topic taxonomy are documented in the YAML
file's header comment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

CLAIMS_PATH = Path("docs/claims.yaml")
OUT_PATH = Path("docs/claims.md")
SAGAN_BASE_URL = os.environ.get("SAGAN_BASE_URL", "https://sagan.superkaiba.com").rstrip("/")


def load_claims(path: Path = CLAIMS_PATH) -> list[dict[str, Any]]:
    """Load claims list from YAML; return [] if file is missing or empty."""
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    claims = data.get("claims", []) or []
    if not isinstance(claims, list):
        raise ValueError(f"docs/claims.yaml: 'claims' key must be a list, got {type(claims)}")
    return claims


def _format_evidence(claim: dict[str, Any]) -> str:
    ev = claim.get("evidence") or {}
    parts: list[str] = []
    experiments = ev.get("experiments") or ev.get("issues") or []
    for n in experiments:
        if isinstance(n, int):
            parts.append(f"[exp #{n}]({SAGAN_BASE_URL}/experiments?number={n})")
    if ev.get("wandb_report"):
        parts.append(f"[WandB]({ev['wandb_report']})")
    section = ev.get("results_md_section")
    if section:
        # Use a relative link to RESULTS.md so the rendered table works on
        # github.com regardless of fork.
        slug = section.lower().replace(" ", "-").replace("&", "")
        parts.append(f"[RESULTS.md § {section}](../RESULTS.md#{slug})")
    figures = ev.get("figures") or []
    if figures:
        parts.append("figs: " + ", ".join(f"`{f}`" for f in figures))
    return "; ".join(parts) if parts else "_(none)_"


def render_table(claims: list[dict[str, Any]]) -> str:
    if not claims:
        return (
            "# Claims registry\n\n"
            "_(empty — populate `docs/claims.yaml`. See the file's header for the schema.)_\n"
        )
    header = (
        "# Claims registry\n\n"
        "Auto-generated from `docs/claims.yaml` by `scripts/render_claims.py`.\n"
        "Do not hand-edit — edit the YAML and re-render.\n\n"
        "| ID | Description | Topic | Status | Evidence | Updated |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows: list[str] = []
    for claim in claims:
        row = (
            f"| `{claim.get('id', '?')}` "
            f"| {claim.get('description', '').strip()} "
            f"| `{claim.get('topic', '?')}` "
            f"| `{claim.get('status', '?')}` "
            f"| {_format_evidence(claim)} "
            f"| {claim.get('updated', '?')} |"
        )
        rows.append(row)
    return header + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local", action="store_true", help="Write docs/claims.md (default).")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout, do not write.")
    args = parser.parse_args(argv)

    claims = load_claims()
    rendered = render_table(claims)
    if args.stdout:
        sys.stdout.write(rendered)
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(rendered)
    print(f"[render_claims] wrote {OUT_PATH} ({len(claims)} claim(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
