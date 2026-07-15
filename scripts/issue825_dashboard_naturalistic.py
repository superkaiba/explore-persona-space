#!/usr/bin/env python3
"""Reproducible generator for the #825/#1092 naturalistic-format dashboard.

Builds ONE self-contained HTML page (house `experiments/dashboards/` newsprint
style, CSS + toggle-JS imported verbatim from
`scripts/issue779_dashboard_completions.py`) showing real LMSYS/WildChat
conversations rendered TWO ways, side by side:

  (a) the Qwen chat template  (tokenizer.apply_chat_template — `<|im_start|>`...)
  (b) the plain naturalistic format  (`User: ...\\n\\nAssistant: ...`)

This is the Result-3 comparison: the context->answer map survives removing the
chat template. On the plain-text format the map reads R^2 ~ 0.71-0.74 (base) /
0.70-0.73 (instruct), ~unchanged from the chat-template read — so it is not a
chat-template artifact.

Data source (HF dataset repo, revision main):
  superkaiba1/explore-persona-space-data
    issue1092_realistic_crossing/corpus/prefix_store.jsonl
  Each prefix carries {prefix_turns:[{role,content}], natural_query, topic,
  source, n_user_turns, total_tokens}. The two render functions are copied
  VERBATIM from scripts/issue1092_gpu_phase.py (`_render_naturalistic`) and
  scripts/issue1092_build_corpus.py (chat-template via apply_chat_template).

SAFETY: LMSYS/WildChat is raw real-user text and can carry explicit / harmful /
NSFW rows. This is a MENTOR-FACING dashboard — examples are selected by a
professional-topic allowlist PLUS a keyword denylist over the full text, and NO
conversation text is ever printed to stdout (read from JSON, written into the
HTML string here). The verify pass emits only structural digests.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1092_realistic_crossing"
HF_REVISION = "main"
HF_BLOB = (
    "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
    "blob/main/issue1092_realistic_crossing/corpus/prefix_store.jsonl"
)
HF_TREE_URL = (
    "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
    "tree/main/issue1092_realistic_crossing/corpus"
)
INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Professional / clean topic allowlist (excludes the synthetic trait strata
# evil/hallucination/sycophancy/persona and the more sensitive natural topics).
TOPIC_ALLOW = {
    "coding_software": "#2f6f6a",
    "science_medicine": "#3a5a8c",
    "math_logic": "#6a4c93",
    "education_learning": "#8a5a1a",
    "business_finance": "#4a6a3a",
    "language_translation": "#7a3b6a",
    "general_qa": "#534c41",
    "writing_creative": "#9a6b2e",
}
SOURCE_ALLOW = {"wildchat", "lmsys"}

# Keyword denylist. Short ambiguous tokens are matched on WORD BOUNDARIES so
# they do not false-hit legitimate words ("anal" must not match "analysis",
# "cum" must not match "document", "sex" must not match "sextant"). A hit skips
# the example entirely. Not exhaustive; the topic allowlist is the primary filter.
import re as _re  # noqa: E402

# Matched as whole words only (\b...\b).
DENY_WORDS = {
    "sex",
    "porn",
    "nude",
    "naked",
    "cum",
    "anal",
    "rape",
    "loli",
    "gore",
    "nigger",
    "faggot",
    "retard",
    "kike",
    "chink",
    "spic",
    "penis",
    "vagina",
    "boob",
    "boobs",
    "horny",
    "incest",
}
# Matched as substrings / phrases (safe stems and multi-word markers).
DENY_SUBSTR = {
    "nsfw",
    "erotic",
    "fetish",
    "orgasm",
    "masturbat",
    "blowjob",
    "molest",
    "pedophil",
    "kill yourself",
    "suicide",
    "self-harm",
    "self harm",
    "cutting myself",
    "how to make a bomb",
    "how to make meth",
    "methamphetamine",
    "detonat",
    "jailbreak",
    "dan mode",
    "ignore all previous",
    "do anything now",
    "beheading",
    "child porn",
}
# Legacy union kept for the belt-and-suspenders final-doc rescan reference.
DENYLIST = DENY_WORDS | DENY_SUBSTR
_DENY_WORD_RE = _re.compile(r"\b(?:" + "|".join(_re.escape(w) for w in DENY_WORDS) + r")\b")

N_EXAMPLES = 20
MIN_TURNS = 2
MAX_TURNS = 8
MAX_TOTAL_TOKENS = 380
MAX_TURN_CHARS = 900
MAX_QUERY_CHARS = 600
PAGE_SIZE_CAP_BYTES = 4 * 1024 * 1024


def _load_house_style() -> tuple[str, str]:
    spec = importlib.util.spec_from_file_location(
        "issue779_dashboard_completions",
        REPO_ROOT / "scripts" / "issue779_dashboard_completions.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.STYLE, mod.TOGGLE_JS


# ---------------------------------------------------------------------------
# Render functions — copied VERBATIM from the #1092 pipeline.
#   _render_naturalistic  <- scripts/issue1092_gpu_phase.py
#   chat template         <- scripts/issue1092_build_corpus.py::_render_instruct
# ---------------------------------------------------------------------------
def _render_naturalistic(turns: list[dict], query: str) -> str:
    """Render as naturalistic plain text (User: / Assistant: format)."""
    lines = []
    for t in turns:
        role = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{role}: {t['content']}")
        lines.append("")
    lines.append(f"User: {query}")
    lines.append("")
    lines.append("Assistant:")
    return "\n".join(lines)


def _render_chat_template(tok, turns: list[dict], query: str) -> str:
    """Qwen instruct chat template (tokenizer.apply_chat_template + gen prompt)."""
    messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    messages.append({"role": "user", "content": query})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Data.
# ---------------------------------------------------------------------------
def download_prefix_store(download_dir: Path, skip_download: bool) -> Path:
    from huggingface_hub import hf_hub_download

    download_dir.mkdir(parents=True, exist_ok=True)
    rel = f"{HF_PREFIX}/corpus/prefix_store.jsonl"
    local = download_dir / rel
    if skip_download and local.exists():
        return local
    hf_hub_download(
        HF_DATA_REPO, rel, repo_type="dataset", revision=HF_REVISION, local_dir=str(download_dir)
    )
    return download_dir / rel


def _denylist_hit(text: str) -> bool:
    low = text.lower()
    if _DENY_WORD_RE.search(low):
        return True
    return any(bad in low for bad in DENY_SUBSTR)


def _clean_candidate(e: dict) -> bool:
    if e.get("source") not in SOURCE_ALLOW:
        return False
    if e.get("topic") not in TOPIC_ALLOW:
        return False
    turns = e.get("prefix_turns")
    if not isinstance(turns, list) or not (MIN_TURNS <= len(turns) <= MAX_TURNS):
        return False
    q = e.get("natural_query")
    if not isinstance(q, str) or not q.strip() or len(q) > MAX_QUERY_CHARS:
        return False
    if e.get("total_tokens") is not None and int(e["total_tokens"]) > MAX_TOTAL_TOKENS:
        return False
    all_text_parts = [q]
    for t in turns:
        if t.get("role") not in ("user", "assistant"):
            return False
        c = t.get("content")
        if not isinstance(c, str) or not c.strip() or len(c) > MAX_TURN_CHARS:
            return False
        all_text_parts.append(c)
    return not _denylist_hit("\n".join(all_text_parts))


def select_examples(entries: list[dict], n: int) -> list[dict]:
    """Round-robin across clean topics for variety; deterministic by prefix_id."""
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if _clean_candidate(e):
            by_topic[e["topic"]].append(e)
    for topic in by_topic:
        by_topic[topic].sort(key=lambda e: e["prefix_id"])
    chosen: list[dict] = []
    topics = [t for t in TOPIC_ALLOW if by_topic.get(t)]
    idx = {t: 0 for t in topics}
    while len(chosen) < n and topics:
        progressed = False
        for t in list(topics):
            if len(chosen) >= n:
                break
            i = idx[t]
            if i < len(by_topic[t]):
                chosen.append(by_topic[t][i])
                idx[t] += 1
                progressed = True
        if not progressed:
            break
    return chosen


# ---------------------------------------------------------------------------
# HTML.
# ---------------------------------------------------------------------------
def esc(s: str) -> str:
    return html.escape(s if s is not None else "")


def render_example(tok, e: dict, n: int) -> str:
    turns = e["prefix_turns"]
    query = e["natural_query"]
    accent = TOPIC_ALLOW[e["topic"]]
    chat = _render_chat_template(tok, turns, query)
    nat = _render_naturalistic(turns, query)
    n_user = sum(1 for t in turns if t["role"] == "user")
    n_asst = sum(1 for t in turns if t["role"] == "assistant")
    head = (
        '<div class="exhead">'
        f'<span class="extopic" style="background:{accent}">{esc(e["topic"])}</span>'
        f'<span class="exsrc mono">{esc(e["source"])}</span>'
        f'<span class="exmeta mono">{len(turns)} history turns '
        f"({n_user}U / {n_asst}A) + 1 query &middot; {esc(e['prefix_id'])}</span>"
        "</div>"
    )
    grid = (
        '<div class="fmtgrid">'
        '<div class="fmtcol">'
        '<span class="fmtlbl chat">Qwen chat template (apply_chat_template)</span>'
        f'<div class="body ans clamped">{esc(chat)}</div>'
        '<button class="toggle" type="button">show full &darr;</button>'
        "</div>"
        '<div class="fmtcol">'
        '<span class="fmtlbl plain">Plain User: / Assistant: (no chat template)</span>'
        f'<div class="body ans clamped">{esc(nat)}</div>'
        '<button class="toggle" type="button">show full &darr;</button>'
        "</div>"
        "</div>"
    )
    return f'<div class="excard" style="--c:{accent}">{head}{grid}</div>'


EXTRA_STYLE = """
.excard{background:var(--card); border:1px solid var(--line-2); border-left:4px solid var(--c);
  border-radius:11px; box-shadow:var(--shadow); padding:14px 16px 15px; margin:14px 0}
.exhead{display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:10px}
.extopic{font-family:"Spline Sans Mono",monospace; font-size:10px; letter-spacing:.1em;
  text-transform:uppercase; color:#fff8ef; padding:3px 9px; border-radius:5px}
.exsrc{font-size:11px; color:var(--ink-soft); border:1px solid var(--line-2); border-radius:5px;
  padding:2px 8px; text-transform:uppercase; letter-spacing:.08em}
.exmeta{font-size:11px; color:var(--ink-soft)}
.fmtgrid{display:grid; grid-template-columns:1fr 1fr; gap:14px}
.fmtcol{display:flex; flex-direction:column; min-width:0}
.fmtlbl{font-family:"Spline Sans Mono",monospace; font-size:9.5px; letter-spacing:.13em;
  text-transform:uppercase; display:inline-block; margin-bottom:5px; padding-bottom:2px;
  border-bottom:1px solid var(--line)}
.fmtlbl.chat{color:#8a5a1a}
.fmtlbl.plain{color:var(--c)}
.fmtcol .body{white-space:pre-wrap; word-break:break-word; font-size:13.5px; line-height:1.5;
  background:var(--paper); border:1px solid var(--line); border-radius:7px; padding:9px 11px}
.fmtcol .body.ans.clamped{max-height:20em; overflow:hidden; position:relative}
.fmtcol .body.ans.clamped::after{content:""; position:absolute; left:0; right:0; bottom:0;
  height:2.2em; background:linear-gradient(transparent, var(--paper)); pointer-events:none}
.fmtcol .body.ans.expanded{max-height:none}
.fmtcol .body.ans.expanded::after{display:none}
@media (max-width:760px){ .fmtgrid{grid-template-columns:1fr} }
"""


def _stat(num, lbl: str) -> str:
    return (
        f"<div class='stat'><span class='num'>{num}</span><span class='lbl'>{esc(lbl)}</span></div>"
    )


def build_html(examples_html: list[str], style: str, toggle_js: str, stats: dict) -> str:
    parts = []
    parts.append("<!doctype html><html lang=en><head><meta charset=utf-8>")
    parts.append("<meta name=viewport content='width=device-width, initial-scale=1'>")
    parts.append("<title>Issue 825/1092 — chat template vs plain User:/Assistant:</title>")
    parts.append(f"<style>{style}{EXTRA_STYLE}</style></head><body><div class='wrap'>")

    n_clean_str = f"{stats['n_clean']:,}"
    n_total_str = f"{stats['n_total']:,}"
    parts.append(
        "<header class='masthead'>"
        "<div class='kicker'><span class='dot'></span> Issue #825 / #1092 realistic crossing"
        " <span class='dot'></span> LMSYS + WildChat conversations"
        " <span class='dot'></span> Qwen-2.5-7B-Instruct"
        " <span class='dot'></span> context &rarr; answer</div>"
        "<h1 class='title'>Chat template vs <em>plain text</em></h1>"
        "<p class='dek'>The Result-3 data: real LMSYS/WildChat conversations rendered two ways — "
        "the Qwen <b>chat template</b> (<code>&lt;|im_start|&gt;</code> role tokens) and the plain "
        "<b>User: / Assistant:</b> naturalistic format. The context&rarr;answer map survives "
        "removing the template: on the plain-text format it reads R&sup2; &asymp; 0.71&ndash;0.74 "
        "(base) / 0.70&ndash;0.73 (instruct), ~unchanged from the chat-template read &mdash; so it "
        "is not a chat-template artifact.</p>"
        "<div class='statbar'>"
        f"{_stat(stats['n_shown'], 'Conversations shown')}"
        f"{_stat(2, 'Render formats each')}"
        f"{_stat(stats['n_topics'], 'Topics')}"
        f"{_stat(n_clean_str, 'Clean candidates')}"
        f"{_stat(n_total_str, 'Corpus prefixes')}"
        "</div></header>"
    )
    parts.append(
        "<div class='callout'><b>What you're seeing.</b> Each card is ONE conversation from the "
        "#1092 corpus (a multi-turn LMSYS/WildChat history + its natural next user query), shown "
        "in <b>both</b> render formats side by side. The two render functions are copied verbatim "
        "from the pipeline (<code>_render_naturalistic</code> and the "
        "<code>apply_chat_template</code> call). The map is read over the CONTEXT (everything "
        "before the final answer); the comparison holds the conversation fixed and only swaps how "
        "it is formatted. Full corpus browsable "
        f"<a href='{HF_TREE_URL}'>on the HF data repo</a>.</div>"
    )
    parts.append(
        "<div class='warnbanner'><b>Selection note.</b> LMSYS/WildChat is raw real-user text. "
        "Examples here are filtered to a professional-topic allowlist plus a keyword denylist and "
        "hand-bounded in length; they are a clean, representative subsample, not a random "
        "draw.</div>"
    )

    parts.extend(examples_html)

    parts.append(
        "<footer class='foot'>"
        f"<div><b>Data source.</b> HF dataset <span class='mono'>{HF_DATA_REPO}</span>, "
        f"<span class='mono'>{HF_PREFIX}/corpus/prefix_store.jsonl</span> "
        f"(revision <span class='mono'>{HF_REVISION}</span>) — each prefix carries "
        "<span class='mono'>prefix_turns</span> + <span class='mono'>natural_query</span> + "
        "<span class='mono'>topic</span> + <span class='mono'>source</span>. "
        f"<a href='{HF_BLOB}'>prefix_store.jsonl</a>.</div>"
        "<div><b>Render functions (verbatim).</b> naturalistic: "
        "<span class='mono'>scripts/issue1092_gpu_phase.py::_render_naturalistic</span>; "
        "chat template: <span class='mono'>tokenizer.apply_chat_template(...) </span> on "
        f"<span class='mono'>{INSTRUCT_MODEL}</span> (matches "
        "<span class='mono'>issue1092_build_corpus.py::_render_instruct</span>).</div>"
        "<div><b>Selection.</b> professional-topic allowlist + keyword denylist + length bounds; "
        f"round-robin across topics, deterministic by prefix_id; {stats['n_shown']} shown of "
        f"{stats['n_clean']} clean candidates.</div>"
        "<div><b>Generator.</b> "
        "<span class='mono'>scripts/issue825_dashboard_naturalistic.py</span> "
        f"&middot; generated {date.today().isoformat()}.</div>"
        "</footer>"
    )
    parts.append(f"<script>{toggle_js}</script>")
    parts.append("</div></body></html>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", default="/tmp/issue825_nat_dash_dl")
    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "experiments" / "dashboards" / "issue825_naturalistic_examples.html"
        ),
    )
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    args = ap.parse_args()

    style, toggle_js = _load_house_style()
    store_path = download_prefix_store(Path(args.download_dir), args.skip_download)
    entries = []
    with open(store_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    n_clean = sum(1 for e in entries if _clean_candidate(e))
    chosen = select_examples(entries, args.n_examples)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, trust_remote_code=True)

    ex_html = [render_example(tok, e, i) for i, e in enumerate(chosen)]
    topics_shown = sorted({e["topic"] for e in chosen})
    stats = {
        "n_shown": len(chosen),
        "n_topics": len(topics_shown),
        "n_clean": n_clean,
        "n_total": len(entries),
    }
    doc = build_html(ex_html, style, toggle_js, stats)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")

    size = len(doc.encode("utf-8"))
    from collections import Counter

    topic_counts = Counter(e["topic"] for e in chosen)
    src_counts = Counter(e["source"] for e in chosen)
    turn_counts = Counter(len(e["prefix_turns"]) for e in chosen)
    print("=" * 66)
    print(f"WROTE {out}")
    print(f"PAGE_BYTES {size}  ({size / 1024 / 1024:.2f} MB)  cap={PAGE_SIZE_CAP_BYTES}")
    print(f"CORPUS_PREFIXES {len(entries)}  CLEAN_CANDIDATES {n_clean}  SHOWN {len(chosen)}")
    print(f"TOPICS_SHOWN ({len(topics_shown)}): {topics_shown}")
    print(f"PER-TOPIC shown: {dict(sorted(topic_counts.items()))}")
    print(f"PER-SOURCE shown: {dict(sorted(src_counts.items()))}")
    print(f"HISTORY-TURN-COUNT hist: {dict(sorted(turn_counts.items()))}")
    print("=" * 66)
    if size > PAGE_SIZE_CAP_BYTES:
        print(f"WARNING: page exceeds cap ({size} > {PAGE_SIZE_CAP_BYTES})")


if __name__ == "__main__":
    main()
