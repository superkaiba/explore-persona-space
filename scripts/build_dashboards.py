#!/usr/bin/env python
"""Standardized per-issue dashboard generator (EPS workflow v2).

Renders tabular experiment data (list-of-objects JSON, JSONL, or CSV) to
standalone, dependency-free, sortable HTML tables under
``experiments/dashboards/`` using the live per-issue naming convention
(``issue<N>_<name>.html``). Large tables shard numerically with an index
page linking the shards; the total committed payload per issue is
hard-capped, and an over-cap build FAILs loudly (route the full dump to
the HF data repo and dashboard a subset).

Usage
-----
    uv run python scripts/build_dashboards.py build --issue 667 \\
        --table contexts=eval_results/issue_667/contexts.json \\
        --table completions=eval_results/issue_667/completions.jsonl \\
        [--out-dir experiments/dashboards] [--shard-mb 1.5] [--max-payload-mb 10]

    uv run python scripts/build_dashboards.py emit-links --issue 667 --sha <40-hex-sha>

Each ``--table name=path`` renders ``issue<N>_<name>.html``. When that page
would exceed ``--shard-mb`` it is split into ``issue<N>_<name>_p1.html``,
``_p2.html``, ... and ``issue<N>_<name>.html`` becomes an index page linking
the shards with their row ranges. A manifest ``issue<N>_manifest.json`` records
``[{table, files, rows, bytes}]`` (files are repo-relative POSIX paths).

Design: stdlib only (json, csv, html, argparse, pathlib, re, sys). Fail fast —
malformed ``--table`` specs, unknown extensions, non-object records, a missing
manifest, and over-cap payloads all raise ``SystemExit``.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

# GitHub repo slug the htmlpreview links point at.
GH_REPO = "superkaiba/explore-persona-space"
# Long text cells collapse behind a <details> preview above this length.
COLLAPSE_THRESHOLD = 280
PREVIEW_LEN = 140
# The JS sort key stored per cell is capped for long cells (prefix sort is fine
# and keeps a big completion from being duplicated verbatim into data-v).
SORT_KEY_MAX = 200
SHA_RE = re.compile(r"[0-9a-fA-F]{40}")

_CSS = """*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
color:#1a1a1a;background:#fafafa}
.wrap{max-width:1400px;margin:0 auto;padding:20px 22px 90px}
header{border-bottom:2px solid #1a1a1a;padding-bottom:12px;margin-bottom:14px}
h1{font-size:22px;margin:0 0 4px}
.sub{font-size:12px;color:#666}
.controls{position:sticky;top:0;z-index:5;background:#fafafa;padding:12px 0;
display:flex;gap:12px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #ddd}
#q{flex:1;min-width:220px;padding:8px 11px;border:1.5px solid #1a1a1a;border-radius:7px;
font:13px monospace}
#cnt{font:12px monospace;color:#666;white-space:nowrap}
table{border-collapse:collapse;width:100%;margin-top:8px;background:#fff}
th,td{border:1px solid #e2e2e2;padding:7px 9px;text-align:left;vertical-align:top;
font-size:13px}
thead th{position:sticky;top:56px;background:#1a1a1a;color:#fff;cursor:pointer;
user-select:none;white-space:nowrap;font:600 12px monospace}
thead th:hover{background:#333}
thead th::after{content:" \\2195";opacity:.4}
thead th[data-asc="1"]::after{content:" \\2191";opacity:1}
thead th[data-asc="0"]::after{content:" \\2193";opacity:1}
tbody tr:nth-child(even){background:#f6f6f6}
.cell,.full{white-space:pre-wrap;word-break:break-word;max-width:640px}
details summary{cursor:pointer;white-space:pre-wrap;word-break:break-word;
max-width:640px;color:#333}
details .full{margin-top:6px;padding-top:6px;border-top:1px dashed #ccc}
.idx{list-style:none;padding:0}
.idx li{border:1px solid #e2e2e2;background:#fff;border-radius:7px;padding:10px 14px;
margin:8px 0}
.idx a{font:600 14px monospace;color:#0b57d0;text-decoration:none}
.idx .rng{font:12px monospace;color:#666;margin-left:10px}
footer{margin-top:40px;font:11px monospace;color:#888;border-top:1px solid #ddd;
padding-top:14px}
.empty{padding:40px 0;text-align:center;color:#888;font-style:italic}"""

_SORT_JS = """<script>
(function(){
  var t=document.getElementById('t'); if(!t||!t.tHead)return;
  var tb=t.tBodies[0], rows=Array.prototype.slice.call(tb.rows);
  var q=document.getElementById('q'), cnt=document.getElementById('cnt');
  function numeric(s){return /^\\s*[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?\\s*$/.test(s);}
  function update(){
    var term=(q&&q.value||'').toLowerCase(), n=0;
    for(var i=0;i<rows.length;i++){
      var show=!term||rows[i].textContent.toLowerCase().indexOf(term)>=0;
      rows[i].style.display=show?'':'none'; if(show)n++;
    }
    if(cnt)cnt.textContent=n+' / '+rows.length+' rows';
  }
  if(q)q.addEventListener('input',update);
  var heads=t.tHead.rows[0].cells;
  for(var c=0;c<heads.length;c++){(function(ci){
    heads[ci].addEventListener('click',function(){
      var asc=heads[ci].getAttribute('data-asc')!=='1';
      for(var h=0;h<heads.length;h++)heads[h].removeAttribute('data-asc');
      heads[ci].setAttribute('data-asc',asc?'1':'0');
      var s=rows.slice().sort(function(a,b){
        var x=a.cells[ci].getAttribute('data-v')||'', y=b.cells[ci].getAttribute('data-v')||'';
        var d;
        if(numeric(x)&&numeric(y))d=parseFloat(x)-parseFloat(y);
        else d=(x<y?-1:(x>y?1:0));
        return asc?d:-d;
      });
      for(var k=0;k<s.length;k++)tb.appendChild(s[k]);
    });
  })(c);}
  update();
})();
</script>"""


def _die(msg: str) -> None:
    raise SystemExit(f"build_dashboards: {msg}")


def _parse_table_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        _die(f"--table must be name=path, got {spec!r}")
    name, _, raw = spec.partition("=")
    name, raw = name.strip(), raw.strip()
    if not name or not raw:
        _die(f"--table must be name=path with both sides non-empty, got {spec!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        _die(f"--table name must be [A-Za-z0-9_.-]+, got {name!r}")
    return name, Path(raw)


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        _die(f"table source does not exist: {path}")
    ext = path.suffix.lower()
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            _die(f"{path}: JSON must be a top-level list of objects, got {type(data).__name__}")
        records = data
    elif ext == ".jsonl":
        records = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                _die(f"{path}:{i}: invalid JSONL line: {e}")
    elif ext == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            records = [dict(row) for row in csv.DictReader(fh)]
    else:
        _die(f"{path}: unsupported extension {ext!r} (want .json, .jsonl, or .csv)")
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            _die(f"{path}: record {i} is {type(rec).__name__}, expected an object")
    return records


def _columns(records: list[dict]) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()
    for rec in records:
        for key in rec:
            if key not in seen:
                seen.add(key)
                cols.append(str(key))
    return cols


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def _render_cell(text: str) -> str:
    sort_key = text if len(text) <= COLLAPSE_THRESHOLD else text[:SORT_KEY_MAX]
    attr = html.escape(sort_key, quote=True)
    if len(text) > COLLAPSE_THRESHOLD:
        preview = html.escape(text[:PREVIEW_LEN])
        full = html.escape(text)
        return (
            f'<td data-v="{attr}"><details><summary>{preview}…</summary>'
            f'<div class="full">{full}</div></details></td>'
        )
    return f'<td data-v="{attr}"><div class="cell">{html.escape(text)}</div></td>'


def _render_row(rec: dict, columns: list[str]) -> str:
    cells = "".join(_render_cell(_cell_text(rec.get(col))) for col in columns)
    return f"<tr>{cells}</tr>"


def _page(title: str, subtitle: str, columns: list[str], body_rows: str, footer: str) -> str:
    head_cells = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    if columns:
        controls = (
            '<div class="controls"><input id="q" type="search" '
            'placeholder="filter rows…" autocomplete="off">'
            '<span id="cnt"></span></div>'
        )
        table = (
            f'<table id="t"><thead><tr>{head_cells}</tr></thead><tbody>{body_rows}</tbody></table>'
        )
        script = _SORT_JS
    else:
        controls = ""
        table = '<div class="empty">No rows.</div>'
        script = ""
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
        f'<div class="wrap"><header><h1>{html.escape(title)}</h1>'
        f'<div class="sub">{html.escape(subtitle)}</div></header>'
        f"{controls}{table}"
        f"<footer>{html.escape(footer)}</footer></div>{script}</body></html>"
    )


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _greedy_pack(rows_html: list[str], budget: int) -> list[tuple[int, int, str]]:
    """Pack rows into shards whose row-content bytes stay under ``budget``.

    Always keeps at least one row per shard (a single row larger than the
    budget gets its own shard). Returns ``(start_1based, end_1based, joined_html)``.
    """
    groups: list[tuple[int, int, str]] = []
    cur: list[str] = []
    cur_bytes = 0
    start = 1
    for i, row in enumerate(rows_html, 1):
        rb = _nbytes(row)
        if cur and cur_bytes + rb > budget:
            groups.append((start, i - 1, "".join(cur)))
            cur, cur_bytes, start = [], 0, i
        cur.append(row)
        cur_bytes += rb
    if cur:
        groups.append((start, len(rows_html), "".join(cur)))
    return groups


def _build_table(
    name: str, issue: int, records: list[dict], shard_bytes: int
) -> tuple[list[tuple[str, str]], int]:
    """Render one table to (basename, content) pairs; the first is the entry page."""
    prefix = f"issue{issue}_{name}"
    columns = _columns(records)
    rows_html = [_render_row(rec, columns) for rec in records]
    footer = f"issue #{issue} · {name} · {len(records)} rows · generated by build_dashboards.py"

    single = _page(prefix, f"{len(records)} rows", columns, "".join(rows_html), footer)
    if _nbytes(single) <= shard_bytes or not rows_html:
        return [(f"{prefix}.html", single)], len(records)

    # Overhead of a shard page (empty body, worst-case-length subtitle). The
    # real shard title is f"{prefix} ({sub})" (rendered twice: <title> + <h1>),
    # so the overhead estimate MUST use that title shape — not the bare prefix —
    # or a packed shard can exceed --shard-mb by ~2x the subtitle-suffix length.
    worst_sub = "part 999/999 · rows 999999-999999"
    overhead = _nbytes(_page(f"{prefix} ({worst_sub})", worst_sub, columns, "", footer))
    budget = shard_bytes - overhead
    groups = _greedy_pack(rows_html, budget if budget > 0 else 1)

    out: list[tuple[str, str]] = []
    idx_items = []
    total = len(groups)
    for k, (a, b, body) in enumerate(groups, 1):
        sub = f"part {k}/{total} · rows {a}-{b}"
        out.append((f"{prefix}_p{k}.html", _page(f"{prefix} ({sub})", sub, columns, body, footer)))
        idx_items.append(
            f'<li><a href="{prefix}_p{k}.html">{prefix}_p{k}.html</a>'
            f'<span class="rng">rows {a}-{b}</span></li>'
        )
    index_body = f'<ul class="idx">{"".join(idx_items)}</ul>'
    index = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(prefix)} (index)</title><style>{_CSS}</style></head><body>"
        f'<div class="wrap"><header><h1>{html.escape(prefix)}</h1>'
        f'<div class="sub">{len(records)} rows across {total} shards</div></header>'
        f"{index_body}"
        f"<footer>{html.escape(footer)}</footer></div></body></html>"
    )
    return [(f"{prefix}.html", index), *out], len(records)


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    _die(f"could not locate repo root (.git) at or above {start}")
    raise AssertionError  # unreachable; _die raises


def _cmd_build(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    repo_root = _find_repo_root(out_dir)
    shard_bytes = int(args.shard_mb * 1_000_000)
    if shard_bytes <= 0:
        _die(f"--shard-mb must be positive, got {args.shard_mb}")
    cap_bytes = int(args.max_payload_mb * 1_000_000)

    tables = [_parse_table_arg(s) for s in args.table]
    names = [n for n, _ in tables]
    dups = {n for n in names if names.count(n) > 1}
    if dups:
        _die(f"duplicate --table name(s): {sorted(dups)}")

    out_dir_abs = out_dir.resolve()
    outputs: list[tuple[str, str]] = []
    manifest: list[dict] = []
    for name, path in tables:
        records = _load_records(path)
        files, nrows = _build_table(name, args.issue, records, shard_bytes)
        table_bytes = sum(_nbytes(content) for _, content in files)
        rel_paths = [(out_dir_abs / fn).relative_to(repo_root).as_posix() for fn, _ in files]
        manifest.append({"table": name, "files": rel_paths, "rows": nrows, "bytes": table_bytes})
        outputs.extend(files)

    grand_total = sum(entry["bytes"] for entry in manifest)
    if grand_total > cap_bytes:
        _die(
            f"total dashboard payload {grand_total} bytes exceeds the "
            f"{cap_bytes}-byte per-issue cap. Route the full dump to the HF data "
            f"repo (superkaiba1/explore-persona-space-data) and dashboard a subset."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for fn, content in outputs:
        (out_dir / fn).write_text(content, encoding="utf-8")
    manifest_path = out_dir / f"issue{args.issue}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(outputs)} file(s) + manifest to {out_dir} ({grand_total} bytes total).")
    for entry in manifest:
        print(
            f"  {entry['table']}: {entry['rows']} rows, {len(entry['files'])} file(s), "
            f"{entry['bytes']} bytes"
        )
    return 0


def _cmd_emit_links(args: argparse.Namespace) -> int:
    if not SHA_RE.fullmatch(args.sha):
        _die(f"--sha must be a 40-char hex commit SHA, got {args.sha!r}")
    manifest_path = Path(args.out_dir) / f"issue{args.issue}_manifest.json"
    if not manifest_path.exists():
        _die(f"manifest not found: {manifest_path} (run `build` first)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest:
        primary = entry["files"][0]
        url = (
            f"https://htmlpreview.github.io/?https://raw.githubusercontent.com/"
            f"{GH_REPO}/{args.sha}/{primary}"
        )
        print(f"{entry['table']}: {url}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="build_dashboards.py", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="render per-issue dashboard tables")
    b.add_argument("--issue", type=int, required=True)
    b.add_argument(
        "--table",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="table name and source (.json list-of-objects / .jsonl / .csv); repeatable",
    )
    b.add_argument("--out-dir", default="experiments/dashboards")
    b.add_argument("--shard-mb", type=float, default=1.5)
    b.add_argument(
        "--max-payload-mb",
        type=float,
        default=10.0,
        help="hard cap on total committed payload per issue (default 10 MB)",
    )
    b.set_defaults(func=_cmd_build)

    e = sub.add_parser("emit-links", help="print htmlpreview links from a built manifest")
    e.add_argument("--issue", type=int, required=True)
    e.add_argument("--sha", required=True, help="40-char hex commit SHA the files are committed at")
    e.add_argument("--out-dir", default="experiments/dashboards")
    e.set_defaults(func=_cmd_emit_links)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
