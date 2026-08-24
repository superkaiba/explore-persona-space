"""Issue #2223 — side-by-side case-study replay dashboard (offline, stdlib-only).

Reads the per-cell replay JSONs written by ``issue2223_casestudy_replay.py``
(``<out-root>/<model_slug>/<scenario>/<layers>__<arm>.json``) and emits ONE
self-contained HTML file: one stacked section per MODEL, then per scenario,
then one table block per layer config (paper band / all layers), rows = turns
(top -> bottom = conversation order), LEFT column = the frozen user message,
remaining columns = each arm's generated assistant reply at that turn. The
layer-config-independent ``unsteered`` cell renders in BOTH blocks.

Deliberately minimal: NO metrics, NO harm color-coding, NO analysis prose —
raw conversations side by side (content is HTML-escaped). Runs offline from
the committed JSONs; no model, no network.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2223_casestudy_dashboard.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO = _ensure_repo_root_on_syspath()

from scripts.issue2223_casestudy_replay import (  # noqa: E402
    ARM_ORDER,
    LAYER_CONFIGS,
    MODELS,
    SCENARIOS,
)

CSS = """
body { font-family: -apple-system, "Segoe UI", sans-serif; margin: 16px; color: #222; }
h1 { font-size: 20px; } h2 { font-size: 17px; margin-top: 28px; }
h3 { font-size: 14px; margin-top: 18px; }
p.meta { color: #555; font-size: 12px; }
table { border-collapse: collapse; table-layout: fixed; width: max-content; }
th, td { border: 1px solid #ccc; padding: 6px; vertical-align: top;
         font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 11px; }
th { background: #f0f0f0; position: sticky; top: 0; }
td.user { background: #f7f7ff; width: 340px; }
td.turnno { background: #f0f0f0; width: 28px; font-weight: bold; }
td.arm { width: 340px; }
td.missing { color: #999; text-align: center; }
div.msg { max-height: 340px; overflow-y: auto; white-space: pre-wrap;
          word-wrap: break-word; }
div.scroll { overflow-x: auto; }
"""


def _cell_key(layers: str, arm: str) -> str:
    return f"{layers}__{arm}"


def load_cells(model_root: Path, scenario: str) -> dict[str, dict]:
    """Load every cell JSON under ``<model_root>/<scenario>/`` keyed by layers__arm."""
    cells: dict[str, dict] = {}
    sc_dir = model_root / scenario
    if not sc_dir.is_dir():
        return cells
    for p in sorted(sc_dir.glob("*.json")):
        cell = json.loads(p.read_text())
        cells[_cell_key(cell["layers"], cell["arm"])] = cell
    return cells


def discover_model_slugs(out_root: Path) -> list[str]:
    """Model slugs (subdirs of ``out_root``) that hold ≥1 scenario dir with cells.

    Ordered by the ``MODELS`` registry (32b, 27b, tiny, tiny_ih) so both legs
    render in a stable order; any unrecognized slug on disk is appended last.
    """
    if not out_root.is_dir():
        return []
    present = {
        d.name
        for d in out_root.iterdir()
        if d.is_dir() and any((d / sc).is_dir() for sc in SCENARIOS)
    }
    known = [m["slug"] for m in MODELS.values() if m["slug"] in present]
    extra = sorted(present - set(known))
    return known + extra


def _msg_td(text: str, cls: str) -> str:
    return f'<td class="{cls}"><div class="msg">{html.escape(text)}</div></td>'


def render_block(
    scenario: str, layers_cfg: str, cells: dict[str, dict], arms: "list[str]" = ARM_ORDER
) -> str:
    """One table: rows = turns, cols = user + every arm at this layer config.

    ``arms`` defaults to the full ``ARM_ORDER`` (original #2223 dashboard); a
    caller may pass a subset (e.g. the strength-sweep arms) to scope the columns.
    """
    arm_cells: list[tuple[str, dict | None]] = []
    for arm in arms:
        key = _cell_key("na", arm) if arm == "unsteered" else _cell_key(layers_cfg, arm)
        arm_cells.append((arm, cells.get(key)))
    present = [c for _, c in arm_cells if c is not None]
    if not present:
        return f"<p class='meta'>no cells for layer config <b>{layers_cfg}</b> yet</p>"
    ref = present[0]
    user_turns = [rec["user"] for rec in ref["turns"]]
    n_rows = max((len(c["turns"]) for c in present), default=len(user_turns))
    n_rows = max(n_rows, len(user_turns))

    head = "<tr><th>#</th><th>user (frozen)</th>"
    for arm, c in arm_cells:
        note = "" if c is not None else " (missing)"
        head += f"<th>{html.escape(arm)}{note}</th>"
    head += "</tr>"

    rows = []
    for t in range(1, n_rows + 1):
        user = user_turns[t - 1] if t - 1 < len(user_turns) else ""
        row = f'<tr><td class="turnno">{t}</td>' + _msg_td(user, "user")
        for _arm, c in arm_cells:
            if c is None:
                row += '<td class="missing">&mdash;</td>'
                continue
            rec = next((r for r in c["turns"] if r["turn"] == t), None)
            if rec is None:
                mark = "(context limit)" if c.get("truncated_at_turn") else "&mdash;"
                row += f'<td class="missing">{mark}</td>'
            else:
                row += _msg_td(rec["assistant"], "arm")
        row += "</tr>"
        rows.append(row)
    title = {"band": "paper band layers", "all": "all layers"}[layers_cfg]
    return (
        f"<h3>{html.escape(scenario)} — {title}</h3>"
        f'<div class="scroll"><table>{head}{"".join(rows)}</table></div>'
    )


def build_dashboard(
    out_root: Path,
    *,
    arms: "list[str]" = ARM_ORDER,
    models: "list[str] | None" = None,
    scenarios: "tuple[str, ...] | list[str]" = SCENARIOS,
    header_html: str = "",
    title: str = "Issue #2223 — case-study frozen replay (per model)",
    meta_note: str = (
        "Frozen user turns from the paper's UNSTEERED case-study "
        "transcripts; assistant side regenerated per arm (greedy, thinking off). "
        "Deviation: a default system prompt is added (the paper's case studies ran "
        "without one) so prefix-position arms are definable. Content shown verbatim "
        "(escaped); no scoring on the conversation tables themselves.</p>"
    ),
) -> str:
    """Assemble the self-contained HTML.

    ``arms`` scopes the conversation columns; ``models`` (slugs) scopes which
    model legs render (``None`` = all present); ``header_html`` is injected right
    after the H1 (used for the strength-sweep analysis + embedded figures).
    """
    body = [f"<h1>{html.escape(title)}</h1>"]
    if header_html:
        body.append(header_html)
    body.append(f"<p class='meta'>{meta_note}")
    slugs = discover_model_slugs(out_root)
    if models is not None:
        slugs = [s for s in slugs if s in models]
    if not slugs:
        body.append("<p class='meta'>no model cells generated yet</p>")
    for slug in slugs:
        model_root = out_root / slug
        body.append(f"<h1>model: {html.escape(slug)}</h1>")
        for sc in scenarios:
            cells = load_cells(model_root, sc)
            body.append(f"<h2>{html.escape(slug)} — scenario: {html.escape(sc)}</h2>")
            if not cells:
                body.append("<p class='meta'>no cells generated yet</p>")
                continue
            for lc in LAYER_CONFIGS:
                body.append(render_block(sc, lc, cells, arms))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>issue 2223 case-study replay</title><style>{CSS}</style></head>"
        f"<body>{''.join(body)}</body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-root",
        default=str(REPO / "eval_results" / "issue_2223" / "casestudy_replay"),
        help="replay out-root holding <scenario>/<layers>__<arm>.json",
    )
    ap.add_argument(
        "--out",
        default=str(REPO / "figures" / "issue_2223" / "casestudy_replay" / "dashboard.html"),
    )
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_dashboard(out_root)
    out.write_text(html_text)
    print(f"[dashboard] wrote {out} ({len(html_text)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
