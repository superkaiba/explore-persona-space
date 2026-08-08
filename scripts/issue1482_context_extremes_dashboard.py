"""Issue #1482 context-extremes round — top/worst-100 predicted contexts dashboard (D1).

For each of the three input-state arms (context vector / prefix end state /
query only; all predicting the SAME full-context answer state, L19 ridge,
#1738 multi-turn holdout n=9,941) rank the held-out contexts by per-context
normalized error (nerr = ||v_hat-v||^2/||v-mu_eval||^2, the #1482 convention)
and render the best-100 and worst-100 into a self-contained HTML dashboard at
dashboard/public/context-extremes-1482.html (existing 1482 dashboard style).

Also writes:
  - eval_results/issue_1482/context_extremes/context_extremes.json — selection
    lists + per-set composition stats (NO raw text; texts stay in the dashboard
    display layer, excerpt-capped at the judge instrument's caps).
  - data/issue_1482/context_extremes_scratch/blind/<arm>_group{A,B}.md +
    key.json — blinded top-vs-worst text bundles for the D2 qualitative read
    (assignment randomized; the key is written for post-hoc unblinding only).

Inputs are banked: the three-arm per-context summary CSV
(eval_results/issue_1738/bare_query/percontext_summary_L19_ridge.csv) and the
holdout text cache built by scripts/issue1482_collect_holdout_texts.py.
0 GPU; CPU-only. Raw corpus text is never logged to stdout.
"""

from __future__ import annotations

import csv
import html
import json
import secrets
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

CSV = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_1738"
    / "bare_query"
    / ("percontext_summary_L19_ridge.csv")
)
TEXTS = PROJECT_ROOT / "data" / "issue_1482" / "context_extremes_scratch" / "judge_texts.jsonl"
OUT_HTML = PROJECT_ROOT / "dashboard" / "public" / "context-extremes-1482.html"
OUT_JSON = (
    PROJECT_ROOT / "eval_results" / "issue_1482" / "context_extremes" / "context_extremes.json"
)
BLIND_DIR = PROJECT_ROOT / "data" / "issue_1482" / "context_extremes_scratch" / "blind"
FABLE_DIR = PROJECT_ROOT / "eval_results" / "issue_1482" / "context_extremes" / "fable_reads"
N_EXTREME = 100
# display caps == the judge instrument's excerpt caps (labels.json excerpt_caps)
# DISPLAY is uncapped: the dashboard shows the FULL conversation. 1200/800/1000
# were the JUDGE instrument's excerpt caps, reused at the display layer where they
# hid most of every multi-turn history (median history 1,462 chars, max 17,116 --
# the 800 cap truncated ~68% of rows). CAP_BLIND_* below are UNCHANGED: they bound
# the blinded-read packets, a different artifact with its own protocol.
CAP_LAST_USER = None
CAP_HISTORY = None
CAP_RESPONSE = None
CAP_BLIND_LAST_USER = 800
CAP_BLIND_HISTORY = 400

ARMS = (
    ("context", "nerr_context_L19_ridge", "Context vector"),
    ("prefix", "nerr_prefix_L19_ridge", "Prefix end state"),
    ("bare", "nerr_bare_L19_ridge", "Query only"),
)


def _cap(s: str, n: int | None) -> str:
    s = s or ""
    if n is None:
        return s
    return s if len(s) <= n else s[:n] + " …[truncated]"


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _md(src: str) -> str:
    """Render the markdown subset the Fable reports use: #/##/### headings,
    **bold**, `code`, `- ` bullets, blank-line paragraphs. Deliberately tiny —
    the reports are the primary record and must not be reformatted or summarised,
    only displayed."""
    import re as _re

    def _inline(t: str) -> str:
        t = _esc(t)
        t = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = _re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        return t

    out, ul = [], False
    for raw in src.split("\n"):
        line = raw.rstrip()
        if line.startswith("- ") or _re.match(r"^\d+\. ", line):
            if not ul:
                out.append("<ul>")
                ul = True
            out.append("<li>" + _inline(_re.sub(r"^(- |\d+\. )", "", line)) + "</li>")
            continue
        if ul:
            out.append("</ul>")
            ul = False
        if not line.strip():
            continue
        m = _re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl = min(len(m.group(1)) + 2, 6)
            out.append(f"<h{lvl} class='fh'>{_inline(m.group(2))}</h{lvl}>")
        else:
            out.append("<p class='fp'>" + _inline(line) + "</p>")
    if ul:
        out.append("</ul>")
    return "".join(out)


def _fable_block(arm: str, key_for_arm: dict) -> str:
    """The blinded Fable 5 read for one arm, with the A/B -> top/worst key
    resolved and the verdict scored. Returns '' when no read exists."""
    path = FABLE_DIR / f"{arm}.md"
    if not path.exists():
        return ""
    raw = path.read_text(encoding="utf-8")
    body = raw
    fm: dict[str, str] = {}
    if raw.startswith("---"):
        end = raw.index("\n---", 3)
        for ln in raw[3:end].strip().split("\n"):
            if ":" in ln and not ln.startswith(" "):
                k, _, v = ln.partition(":")
                fm[k.strip()] = v.strip()
        body = raw[end + 4 :]
    pred, truth = fm.get("predicted_better", "?"), fm.get("truth_better", "?")
    ok = fm.get("verdict", "") == "CORRECT"
    # A/B are randomized PER ARM, so the reader needs the mapping in front of the
    # report or every "Group A" reads ambiguously.
    keymap = "  &middot;  ".join(
        f"<b>Group {g}</b> = {'BEST' if key_for_arm[g] == 'top' else 'WORST'}-predicted "
        f"({key_for_arm[g]}-100)"
        for g in ("A", "B")
    )
    prime = fm.get("priming_note")
    prime_html = (
        f"<p class='fp warnnote'><b>Extra information given to this arm only.</b> {_esc(prime)}</p>"
        if prime
        else ""
    )
    return (
        "<details open class='fable'><summary>Claude Fable 5 &mdash; blinded read of this "
        f"arm's two groups &nbsp;<span class='verdict {'ok' if ok else 'no'}'>"
        f"predicted {pred} &middot; truth {pred if ok else truth} &middot; "
        f"{'CORRECT' if ok else 'WRONG'}</span></summary>"
        f"<p class='fp keyline'>{keymap}</p>"
        "<p class='fp warnnote'><b>How blind this was.</b> The agent did NOT know which group "
        "was which, and the label assignment was randomized independently per arm. It DID know "
        "the setup: that a linear map predicts the answer state from the context, that the two "
        "files are the top-100 and bottom-100 by that map's error, which arm it was reading, and "
        "that it should predict which group scores better. That framing supplies the hypothesis, "
        "so a correct call is consistent with the story but does not independently establish it.</p>"
        + prime_html
        + _md(body)
        + "</details>"
    )


def _composition(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "topic": dict(Counter(r["topic"] or "(unlabeled)" for r in rows).most_common()),
        "language": dict(Counter(r["language"] or "(unlabeled)" for r in rows).most_common()),
        "format": dict(Counter(r["format"] or "(unlabeled)" for r in rows).most_common()),
        "median_last_user_chars": statistics.median(len(r["_t"]["last_user"]) for r in rows),
        "median_history_chars": statistics.median(len(r["_t"]["history_tail"]) for r in rows),
        "median_response_chars": statistics.median(len(r["_t"]["response"]) for r in rows),
        "n_multiturn_history": sum(1 for r in rows if len(r["_t"]["history_tail"]) > 0),
    }


def main() -> None:
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts: dict[int, dict] = {}
    with open(TEXTS, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            texts[int(t["ci"])] = t
    missing = [r["ci"] for r in rows if int(r["ci"]) not in texts]
    assert not missing, f"missing texts for {len(missing)} cis"
    for r in rows:
        r["_t"] = texts[int(r["ci"])]

    selection: dict[str, dict] = {}
    comp: dict[str, dict] = {}
    for arm, col, _label in ARMS:
        srt = sorted(rows, key=lambda r: (float(r[col]), int(r["ci"])))
        top, worst = srt[:N_EXTREME], srt[-N_EXTREME:][::-1]
        selection[arm] = {"top": top, "worst": worst}
        comp[arm] = {"top": _composition(top), "worst": _composition(worst)}

    # ── blinded bundles for the D2 qualitative read ──────────────────────────
    # The A/B assignment is drawn ONCE and then FROZEN: an existing key.json is
    # reused verbatim on every later run. Re-drawing it per run silently relabels
    # packets a reader may already have read, so a report written against draw k
    # gets scored against draw k+1 -- which is exactly what happened on
    # 2026-08-04 (the primed bare read was briefly mis-scored as wrong). Force a
    # fresh draw only by deleting key.json, and only when no read is outstanding.
    BLIND_DIR.mkdir(parents=True, exist_ok=True)
    key_path = BLIND_DIR / "key.json"
    frozen: dict[str, dict] = {}
    if key_path.exists():
        frozen = json.loads(key_path.read_text())
        print(f"[dash] blind key FROZEN from {key_path} (delete it to re-draw)")
    key: dict[str, dict] = {}
    for arm, _col, _label in ARMS:
        if arm in frozen:
            key[arm] = frozen[arm]
        else:
            a_is_top = secrets.randbelow(2) == 0
            key[arm] = {"A": "top" if a_is_top else "worst", "B": "worst" if a_is_top else "top"}
        for gname in ("A", "B"):
            kind = key[arm][gname]
            # shuffle display order so rank carries no signal
            items = sorted(selection[arm][kind], key=lambda r: int(r["ci"]) * 2654435761 % 2**32)
            lines = [
                f"# Group {gname} — 100 conversation excerpts",
                "",
                "Each item: final user message (capped), preceded by a capped tail of",
                "the prior conversation when one exists.",
                "",
            ]
            for i, r in enumerate(items, 1):
                t = r["_t"]
                lines.append(f"## item {i}")
                if t["history_tail"]:
                    lines.append(f"[history tail] {_cap(t['history_tail'], CAP_BLIND_HISTORY)}")
                lines.append(f"[final user message] {_cap(t['last_user'], CAP_BLIND_LAST_USER)}")
                lines.append("")
            (BLIND_DIR / f"{arm}_group{gname}.md").write_text("\n".join(lines), encoding="utf-8")
    (BLIND_DIR / "key.json").write_text(json.dumps(key, indent=1))

    # ── eval_results JSON (no raw text) ──────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(UTC).isoformat(),
                "corpus": "#1738 multi-turn holdout, n=9,941, L19 ridge; all arms score "
                "the same full-context answer target (bitwise-identical, "
                "twoway_residual.json design)",
                "metric": "nerr = ||v_hat-v||^2/||v-mu_eval||^2 per context; "
                "per-context R^2 = 1 - nerr",
                "sources": [
                    str(CSV.relative_to(PROJECT_ROOT)),
                    "eval_results/issue_1738/judge_labels/labels.json",
                    "scripts/issue1482_collect_holdout_texts.py (text cache)",
                ],
                "selection": {
                    arm: {
                        kind: [
                            {
                                "rank": i + 1,
                                "ci": int(r["ci"]),
                                "nerr": float(r[col]),
                                "topic": r["topic"],
                                "language": r["language"],
                                "format": r["format"],
                            }
                            for i, r in enumerate(selection[arm][kind])
                        ]
                        for kind in ("top", "worst")
                    }
                    for arm, col, _label in ARMS
                },
                "composition": comp,
            },
            indent=1,
        )
    )

    # ── dashboard HTML ───────────────────────────────────────────────────────
    css = """
:root { --fg:#16181d; --mut:#5b6270; --line:#e3e6ec; --bg:#fbfbfd; --card:#fff; }
* { box-sizing:border-box; }
body { margin:0; padding:28px 22px 60px; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif; }
.wrap { max-width:1900px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:17px; margin:34px 0 4px; padding-top:14px; border-top:1px solid var(--line); }
h3 { font-size:15px; margin:18px 0 6px; }
p, li { color:var(--mut); font-size:13.5px; }
.head { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; margin-bottom:10px; }
.head p { margin:6px 0; } .head b { color:var(--fg); }
.warn { background:#fff7ed; border:1px solid #f6d6bd; border-radius:9px; padding:12px 14px;
  margin:10px 0; } .warn p { color:#8a4a12; margin:5px 0; }
.nav a { margin-right:14px; font-size:13px; color:#1b4fd8; text-decoration:none; }
.ctl { margin:12px 0 8px; font-size:13px; color:var(--mut); }
.ctl input { font:inherit; padding:3px 7px; border:1px solid var(--line); border-radius:6px;
  width:280px; }
table { border-collapse:collapse; width:100%; background:var(--card); font-size:13px; }
th, td { padding:5px 9px; border-bottom:1px solid var(--line); vertical-align:top; }
th { position:sticky; top:0; background:#f4f6fa; text-align:left; font-weight:600;
  color:var(--fg); white-space:nowrap; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
td.hl { font-weight:700; }
.tag { font-size:11px; padding:1px 7px; border-radius:9px; border:1px solid var(--line);
  white-space:nowrap; background:#f4f6fa; }
details { margin:2px 0; } summary { cursor:pointer; color:#1b4fd8; font-size:12px; }
.txt { white-space:pre-wrap; word-break:break-word; color:#2c313b; font-size:12.5px; }
.blk { margin:4px 0; } .blk b { font-size:11.5px; color:var(--mut);
  text-transform:uppercase; letter-spacing:.04em; }
"""
    js = """
function filt(inp) {
  const q = inp.value.toLowerCase();
  const tbl = document.getElementById(inp.dataset.tbl);
  for (const tr of tbl.tBodies[0].rows)
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
}
"""

    def _row(r: dict, rank: int, col: str) -> str:
        t = r["_t"]
        nerrs = {a: float(r[c]) for a, c, _l in ARMS}
        arm_of_col = next(a for a, c, _l in ARMS if c == col)
        cells = [f"<td class='n'>{rank}</td>", f"<td class='n'>{int(r['ci'])}</td>"]
        for a, _c, _l in ARMS:
            hl = " hl" if a == arm_of_col else ""
            cells.append(f"<td class='n{hl}'>{nerrs[a]:.3f}</td>")
        cells.append(f"<td class='n'>{1 - nerrs[arm_of_col]:.3f}</td>")
        cells.append(
            f"<td class='cat'><span class='tag'>{_esc(r['topic'] or '—')}</span></td>"
            f"<td class='lang'>{_esc(r['language'] or '—')}</td>"
            f"<td class='fmt'>{_esc(r['format'] or '—')}</td>"
        )
        # Three side-by-side text columns, all visible: no <details> wrapper. The
        # reading task is comparing what came BEFORE the final turn against the turn
        # itself and the answer, which a collapsed or stacked cell makes impossible.
        cells.append(
            "<td class='conv'><div class='txt'>"
            + (
                _esc(_cap(t["history_tail"], CAP_HISTORY))
                if t["history_tail"]
                else "<i>&mdash;</i>"
            )
            + "</div></td>"
        )
        cells.append(
            "<td class='ask'><div class='txt'>"
            + _esc(_cap(t["last_user"], CAP_LAST_USER))
            + "</div></td>"
        )
        cells.append(
            "<td class='ans'><div class='txt'>"
            + _esc(_cap(t["response"], CAP_RESPONSE))
            + "</div></td>"
        )
        return "<tr>" + "".join(cells) + "</tr>"

    def _table(arm: str, col: str, kind: str) -> str:
        tid = f"tbl-{arm}-{kind}"
        head = (
            "<tr><th class='n'>#</th><th class='n'>ci</th>"
            "<th class='n'>nerr ctx</th><th class='n'>nerr prefix</th>"
            "<th class='n'>nerr query</th><th class='n'>R&sup2; (this arm)</th>"
            "<th class='cat'>topic</th><th class='lang'>lang</th><th class='fmt'>format</th>"
            "<th class='conv'>prev conversation</th>"
            "<th class='ask'>final user message</th>"
            "<th class='ans'>assistant answer</th></tr>"
        )
        # rank, ci, 3x nerr, R^2, topic, lang, format, prev-conv, final-user, answer
        widths = (3, 3.5, 5, 5, 5, 5, 9, 4, 4.5, 22, 13, 21)
        assert abs(sum(widths) - 100) < 1e-9, sum(widths)
        cg = "<colgroup>" + "".join(f"<col style='width:{w}%'>" for w in widths) + "</colgroup>"
        body = "".join(_row(r, i + 1, col) for i, r in enumerate(selection[arm][kind]))
        return (
            f"<div class='ctl'>filter: <input data-tbl='{tid}' oninput='filt(this)'></div>"
            f"<table id='{tid}'>{cg}<thead>{head}</thead><tbody>{body}</tbody></table>"
        )

    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Issue 1482 &mdash; best/worst predicted contexts (multi-turn holdout)</title>",
        f"<style>{css}</style><script>{js}</script></head><body><div class='wrap'>",
        "<h1>Best-100 / worst-100 predicted contexts, per arm</h1>",
        "<p class='nav'>",
    ]
    for arm, _col, label in ARMS:
        parts.append(
            f"<a href='#{arm}-fable'>{_esc(label)}: read</a>"
            f"<a href='#{arm}-top'>{_esc(label)}: best</a>"
            f"<a href='#{arm}-worst'>{_esc(label)}: worst</a>"
        )
    parts.append("</p>")
    for arm, col, label in ARMS:
        # the blinded Fable 5 read for this arm, above its two tables
        fb = _fable_block(arm, key[arm])
        if fb:
            parts.append(f"<h2 id='{arm}-fable'>{_esc(label)} &mdash; blinded read</h2>")
            parts.append(fb)
        for kind, klabel in (("top", "best-predicted"), ("worst", "worst-predicted")):
            c = comp[arm][kind]
            parts.append(
                f"<h2 id='{arm}-{kind}'>{_esc(label)} &mdash; {klabel} 100</h2>"
                f"<p>top topics: "
                + ", ".join(f"{_esc(k)} ({v})" for k, v in list(c["topic"].items())[:5])
                + " &middot; top languages: "
                + ", ".join(f"{_esc(k)} ({v})" for k, v in list(c["language"].items())[:5])
                + f" &middot; median final-user-message length {c['median_last_user_chars']:.0f}"
                " chars &middot; "
                f"{c['n_multiturn_history']}/100 have prior turns</p>"
            )
            parts.append(_table(arm, col, kind))
    parts.append("</div></body></html>")
    OUT_HTML.write_text("".join(parts), encoding="utf-8")

    print(f"[dash] wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.2f} MB)")
    print(f"[dash] wrote {OUT_JSON}")
    print(f"[dash] blinded bundles: {BLIND_DIR} (key.json written, not printed)")
    for arm, _col, label in ARMS:
        for kind in ("top", "worst"):
            c = comp[arm][kind]
            tt = ", ".join(f"{k}:{v}" for k, v in list(c["topic"].items())[:4])
            ll = ", ".join(f"{k}:{v}" for k, v in list(c["language"].items())[:4])
            print(
                f"[comp] {arm:8s} {kind:5s} topics[{tt}] langs[{ll}] "
                f"hist>{0}:{c['n_multiturn_history']}/100 "
                f"med_query={c['median_last_user_chars']:.0f}ch "
                f"med_resp={c['median_response_chars']:.0f}ch"
            )


if __name__ == "__main__":
    main()
