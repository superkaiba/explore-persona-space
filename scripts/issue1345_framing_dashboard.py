#!/usr/bin/env python3
"""Reproducible generator for the #1345 "three framings of the assistant" dashboard.

Builds ONE self-contained HTML page (house `experiments/dashboards/` newsprint
style, CSS + toggle-JS imported verbatim from
`scripts/issue779_dashboard_completions.py`) showing worked examples of the THREE
framings the #1345 experiment compares, all holding the assistant persona
constant:

  1. Chat template            — the Qwen `apply_chat_template` rendering
  2. Plain User:/Assistant:   — the same conversations as plain text
  3. Assistant in a story     — the #1345 narrative-prose "collapse" arm

Sections 1 & 2 reuse the conversation-loading + both-format render logic of
`scripts/issue825_dashboard_naturalistic.py` VERBATIM (same corpus, same
`_render_chat_template` / `_render_naturalistic` functions) — the two renderings
of the SAME conversations are exactly the #1345 chat + no-template arms. On top
of the reused render this page ANNOTATES where the context->answer map reads: the
context slot (end of the assistant header) and the answer span.

Section 3 shows the #1345 story rollouts (assistant/AI character answering a
human). The generation used a NOVEL proper name (`ARIA`) for the AI character; in
the DISPLAYED story text that name is display-substituted `ARIA` -> `Assistant`
(word-boundary `\bARIA\b` only). A PROMINENT disclosure banner discloses the
substitution and links the verbatim (unaltered) stories on HF. Per story the
extraction slot (the attribution marker immediately before the character's quoted
answer — the colon in `Assistant replied:`) and the quoted answer span are
highlighted.

Data sources:
  * Sections 1&2 (chat / no-template renderings): HF dataset
    superkaiba1/explore-persona-space-data,
    issue1092_realistic_crossing/corpus/prefix_store.jsonl (revision main) — the
    parent naturalistic corpus reused by issue825_dashboard_naturalistic.
  * Section 3 (stories): HF dataset superkaiba1/explore-persona-space-data,
    issue1345_framing/raw_completions/stories/kept_stories_instruct.jsonl at the
    PINNED revision 2a3cb30acada04defc84fd04d28a2b54da3104cd (the instruct
    kept-stories tree). Each record: {story, story_id, model, tier,
    finish_reason, judge_verdict, parsed_turns:[{q_start,q_end,marker_end,
    a_start,a_end,confidence}], ...}. Offsets index into the ORIGINAL `story`.

CONTEXT-HYGIENE: all example text is read from JSON and written into the HTML
string here; NO example text is printed to stdout. The verify pass prints only
structural digests (counts, byte sizes, HF paths).
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# --- Section 3: pinned story tree ------------------------------------------
STORY_REV = "2a3cb30acada04defc84fd04d28a2b54da3104cd"
STORY_PREFIX = "issue1345_framing/raw_completions/stories"
STORY_FILE = "kept_stories_instruct.jsonl"
STORY_REL = f"{STORY_PREFIX}/{STORY_FILE}"
STORY_TREE_URL = f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/{STORY_REV}/{STORY_PREFIX}"
STORY_BLOB_URL = f"https://huggingface.co/datasets/{HF_DATA_REPO}/blob/{STORY_REV}/{STORY_REL}"

# --- Sections 1&2: parent naturalistic corpus (reused from issue825) --------
CORPUS_PREFIX = "issue1092_realistic_crossing"
CORPUS_REL = f"{CORPUS_PREFIX}/corpus/prefix_store.jsonl"
CORPUS_REV = "main"
CORPUS_TREE_URL = f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/{CORPUS_PREFIX}/corpus"
CORPUS_BLOB_URL = f"https://huggingface.co/datasets/{HF_DATA_REPO}/blob/main/{CORPUS_REL}"
INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Documented yield facts (task #1345 clean-result body; ground-truth counts).
BASE_KEPT = 96
STORY_YIELD_FLOOR = 400
STORY_TARGET = 500

N_CONV = 12
N_STORIES = 10
PAGE_SIZE_CAP_BYTES = 6 * 1024 * 1024

ARIA_RE = re.compile(r"\bARIA\b")


# ---------------------------------------------------------------------------
# Reuse the house style + the issue825 loader/render module.
# ---------------------------------------------------------------------------
def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def esc(s: str) -> str:
    return html.escape(s if s is not None else "")


# ---------------------------------------------------------------------------
# HF download (idempotent, retry-wrapped — the data repo listing/HEAD is flaky).
# ---------------------------------------------------------------------------
def _hf_download(rel: str, download_dir: Path, revision: str, skip_download: bool) -> Path:
    local = download_dir / rel
    if skip_download and local.exists():
        return local
    from huggingface_hub import hf_hub_download

    last = None
    for attempt in range(6):
        try:
            hf_hub_download(
                HF_DATA_REPO,
                rel,
                repo_type="dataset",
                revision=revision,
                local_dir=str(download_dir),
            )
            return download_dir / rel
        except Exception as e:
            last = e
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"HF download failed for {rel}@{revision}: {last}")


def resolve_story_files(revision: str) -> list[str]:
    """Resolve story JSONL paths under the pinned tree via list_repo_tree.

    Falls back to the known filename if the (flaky) listing endpoint times out.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    for attempt in range(4):
        try:
            # HUB_VERIFY_RETRY_EXEMPT: inline 4-attempt retry loop + documented filename fallback
            tree = api.list_repo_tree(
                HF_DATA_REPO,
                path_in_repo=STORY_PREFIX,
                repo_type="dataset",
                recursive=False,
                revision=revision,
            )
            files = [e.path for e in tree if e.path.endswith(".jsonl")]
            if files:
                return sorted(files)
        except (HfHubHTTPError, Exception):
            time.sleep(4 * (attempt + 1))
    return [STORY_REL]  # documented fallback: kept_stories_instruct.jsonl


# ---------------------------------------------------------------------------
# Sections 1 & 2 — annotators over the reused render strings.
#   Highlight: assistant header = context slot; assistant answer = answer span.
# ---------------------------------------------------------------------------
def annotate_chat(raw: str) -> str:
    """Wrap assistant headers (slots) + answers in the Qwen chat template string."""
    HDR = "<|im_start|>assistant\n"
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in re.finditer(re.escape(HDR), raw):
        s, after = m.start(), m.end()
        parts.append(("plain", raw[pos:s]))
        end = raw.find("<|im_end|>", after)
        if end == -1:  # final generation prompt — THE context slot for the query
            parts.append(("slotfinal", HDR))
            parts.append(("plain", raw[after:]))
            pos = len(raw)
        else:
            parts.append(("slot", HDR))
            parts.append(("answer", raw[after:end]))
            pos = end
    parts.append(("plain", raw[pos:]))
    return _emit_fmt(parts)


def annotate_plain(raw: str) -> str:
    """Wrap `Assistant:` headers (slots) + answers in the plain User:/Assistant: string."""
    parts: list[tuple[str, str]] = []
    pos = 0
    HDR = "Assistant:"
    for m in re.finditer(r"(?m)^Assistant:", raw):
        s, after = m.start(), m.end()
        parts.append(("plain", raw[pos:s]))
        nxt = re.search(r"(?m)^User:", raw[after:])
        end = len(raw) if nxt is None else after + nxt.start()
        ans = raw[after:end]
        if ans.strip() == "":  # trailing "Assistant:" generation prompt (final slot)
            parts.append(("slotfinal", HDR))
            parts.append(("plain", ans))
        else:
            parts.append(("slot", HDR))
            parts.append(("answer", ans))
        pos = end
    parts.append(("plain", raw[pos:]))
    return _emit_fmt(parts)


def _emit_fmt(parts: list[tuple[str, str]]) -> str:
    buf = []
    for cls, txt in parts:
        disp = esc(txt)
        if cls == "plain":
            buf.append(disp)
        elif cls == "answer":
            buf.append(f'<span class="hl-a">{disp}</span>')
        elif cls == "slot":
            buf.append(f'<span class="hl-slot">{disp}</span>')
        elif cls == "slotfinal":
            buf.append(
                f'<span class="hl-slot slotfinal">{disp}</span>'
                '<span class="slotcaret" title="context slot &mdash; the map reads here">'
                "▸</span>"
            )
    return "".join(buf)


def render_conv_card(nat, tok, e: dict) -> str:
    turns = e["prefix_turns"]
    query = e["natural_query"]
    accent = nat.TOPIC_ALLOW[e["topic"]]
    chat_html = annotate_chat(nat._render_chat_template(tok, turns, query))
    plain_html = annotate_plain(nat._render_naturalistic(turns, query))
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
        '<span class="fmtlbl chat">Framing 1 &mdash; Qwen chat template</span>'
        f'<div class="body ans clamped">{chat_html}</div>'
        '<button class="toggle" type="button">show full &darr;</button>'
        "</div>"
        '<div class="fmtcol">'
        '<span class="fmtlbl plain">Framing 2 &mdash; plain User: / Assistant:</span>'
        f'<div class="body ans clamped">{plain_html}</div>'
        '<button class="toggle" type="button">show full &darr;</button>'
        "</div>"
        "</div>"
    )
    return f'<div class="excard" style="--c:{accent}">{head}{grid}</div>'


# ---------------------------------------------------------------------------
# Section 3 — story rendering with per-turn highlights + ARIA -> Assistant.
# ---------------------------------------------------------------------------
def _sub_aria(s: str) -> str:
    return ARIA_RE.sub("Assistant", s)


def _find_marker_start(story: str, marker_end: int) -> int:
    """Start of the attribution marker: last `\\bARIA\\b` within 45 chars before marker_end."""
    lo = max(0, marker_end - 45)
    ms = list(ARIA_RE.finditer(story[lo:marker_end]))
    return lo + ms[-1].start() if ms else marker_end


def _paint(story: str, turns: list[dict]) -> list[tuple[str, int, int]]:
    ivs: list[tuple[int, int, str]] = []
    for t in turns:
        qs, qe = t["q_start"], t["q_end"]
        me, as_, ae = t["marker_end"], t["a_start"], t["a_end"]
        if qe > qs:
            ivs.append((qs, qe, "q"))
        ms = _find_marker_start(story, me)
        if me > ms:
            ivs.append((ms, me, "m"))
        if ae > as_:
            ivs.append((as_, ae, "a"))
    ivs.sort(key=lambda x: (x[0], x[1]))
    out: list[tuple[str, int, int]] = []
    cur = 0
    for s, e, cls in ivs:
        s = max(s, cur)  # clip cross-turn overlaps
        if s >= e:
            continue
        if s > cur:
            out.append(("plain", cur, s))
        out.append((cls, s, e))
        cur = e
    if cur < len(story):
        out.append(("plain", cur, len(story)))
    return out


def _story_html(story: str, turns: list[dict]) -> str:
    cls_map = {"q": "hl-q", "m": "hl-slot", "a": "hl-a"}
    buf = []
    for cls, s, e in _paint(story, turns):
        disp = esc(_sub_aria(story[s:e]))
        if cls == "plain":
            buf.append(disp)
        else:
            buf.append(f'<span class="{cls_map[cls]}">{disp}</span>')
            if cls == "m":
                buf.append(
                    '<span class="slotcaret" title="context slot &mdash; v_C read here">▸</span>'
                )
    return "".join(buf)


def render_story_card(r: dict) -> str:
    story = r["story"]
    turns = sorted(r["parsed_turns"], key=lambda t: t["q_start"])
    inner = _story_html(story, turns)
    finish = r.get("finish_reason", "")
    finish_note = (
        "" if finish == "stop" else f' &middot; <span class="capnote">finish: {esc(finish)}</span>'
    )
    tag = (
        f"{esc(r['story_id'])} &middot; {len(turns)} confident Q&rarr;A turns "
        f"&middot; tier {esc(r.get('tier', ''))}{finish_note}"
    )
    return (
        '<div class="storycard">'
        '<div class="rhead">'
        f'<span class="score over">judge {esc(r.get("judge_verdict", ""))}</span>'
        f'<span class="rid">{tag}</span>'
        "</div>"
        '<div class="storylegend mono">'
        '<span class="lg lg-q">question</span>'
        '<span class="lg lg-m">attribution marker &#9656; context slot '
        "(v<sub>C</sub> read here)</span>"
        '<span class="lg lg-a">answer span (Y)</span>'
        "</div>"
        f'<div class="body ans clamped storybody">{inner}</div>'
        '<button class="toggle" type="button">show full &darr;</button>'
        "</div>"
    )


def select_stories(rows: list[dict], n: int) -> list[dict]:
    """Deterministic even spread across natural-stop stories, sorted by story_id."""
    cand = sorted(
        (r for r in rows if r.get("finish_reason") == "stop"),
        key=lambda r: r["story_id"],
    )
    if not cand:
        cand = sorted(rows, key=lambda r: r["story_id"])
    if len(cand) <= n:
        return cand
    step = len(cand) / n
    return [cand[int(i * step)] for i in range(n)]


# ---------------------------------------------------------------------------
# Highlight + section CSS (on top of the reused house style + issue825 grid).
# ---------------------------------------------------------------------------
HL_STYLE = """
.hl-q{background:#fff2c4; box-shadow:inset 0 -2px #eccf62}
.hl-a{background:rgba(47,111,106,.13); box-shadow:inset 0 -2px #2f6f6a}
.hl-slot{background:#ece0f3; box-shadow:inset 0 -2px #6a4c93; border-right:2px solid #6a4c93}
.hl-slot.slotfinal{background:#dcc9ec; border-right:3px solid #4a2f63}
.slotcaret{color:#6a4c93; font-weight:700; font-size:.9em; padding:0 1px}
.capnote{color:var(--warn)}
.hl-legend{display:flex; gap:16px; flex-wrap:wrap; align-items:center; margin:10px 0 2px;
  font-family:"Spline Sans Mono",monospace; font-size:11px; color:var(--ink-soft)}
.hl-legend .sw{display:inline-block; padding:2px 8px; border-radius:4px; margin-right:5px}
.hl-legend .sw.q{background:#fff2c4; box-shadow:inset 0 -2px #eccf62}
.hl-legend .sw.s{background:#ece0f3; box-shadow:inset 0 -2px #6a4c93}
.hl-legend .sw.a{background:rgba(47,111,106,.13); box-shadow:inset 0 -2px #2f6f6a}

.disclosure{margin:22px 0 10px; background:#f3e7d0; border:2px solid var(--warn);
  border-left:7px solid var(--warn); border-radius:10px; padding:15px 20px; font-size:15px;
  color:#5a2419; box-shadow:var(--shadow)}
.disclosure b{color:var(--warn)}
.disclosure .dhead{font-family:"Spline Sans Mono",monospace; font-size:11px; letter-spacing:.18em;
  text-transform:uppercase; color:var(--warn); display:block; margin-bottom:7px}
.disclosure a{color:#5a2419}

.storycard{background:var(--card); border:1px solid var(--line-2); border-left:4px solid #6a4c93;
  border-radius:11px; box-shadow:var(--shadow); padding:13px 16px 14px; margin:13px 0}
.storylegend{display:flex; gap:14px; flex-wrap:wrap; font-size:10px; letter-spacing:.06em;
  color:var(--ink-soft); margin:2px 0 9px}
.storylegend .lg{padding:2px 8px; border-radius:4px}
.storylegend .lg-q{background:#fff2c4; box-shadow:inset 0 -2px #eccf62}
.storylegend .lg-m{background:#ece0f3; box-shadow:inset 0 -2px #6a4c93}
.storylegend .lg-a{background:rgba(47,111,106,.13); box-shadow:inset 0 -2px #2f6f6a}
.storylegend .lg sub{font-size:.85em}
.storybody{font-size:14px; line-height:1.62}
.storyfoot{font-family:"Spline Sans Mono",monospace; font-size:10.5px; color:var(--ink-soft);
  margin-top:12px}
.storyfoot a{color:var(--ink-soft)}
.sectionhead{margin:52px 0 4px; padding-bottom:8px; border-bottom:2px solid var(--ink)}
.sectionhead h2{font-family:"Fraunces",serif; font-weight:600; font-size:29px; margin:0 0 3px}
.sectionhead .sub{font-size:15.5px; color:var(--ink-soft); max-width:82ch}
.sectionhead .sub code{font-family:"Spline Sans Mono",monospace; font-size:.85em;
  background:var(--paper); padding:1px 5px; border-radius:4px; border:1px solid var(--line)}
"""


def _stat(num, lbl: str) -> str:
    return (
        f"<div class='stat'><span class='num'>{num}</span><span class='lbl'>{esc(lbl)}</span></div>"
    )


def build_html(conv_cards, story_cards, style, nat_extra, toggle_js, stats) -> str:
    p = []
    p.append("<!doctype html><html lang=en><head><meta charset=utf-8>")
    p.append("<meta name=viewport content='width=device-width, initial-scale=1'>")
    p.append("<title>Issue 1345 — three framings of the assistant</title>")
    p.append(f"<style>{style}{nat_extra}{HL_STYLE}</style></head><body><div class='wrap'>")

    kept_str = f"{stats['n_kept_stories']:,}"
    turns_str = f"{stats['n_turns']:,}"

    # ---- masthead ----
    p.append(
        "<header class='masthead'>"
        "<div class='kicker'><span class='dot'></span> Issue #1345 framing comparison"
        " <span class='dot'></span> chat template &middot; plain User:/Assistant:"
        " &middot; assistant-in-a-story"
        " <span class='dot'></span> Qwen-2.5-7B(-Instruct)"
        " <span class='dot'></span> context &rarr; answer</div>"
        "<h1 class='title'>Three framings of the <em>assistant</em></h1>"
        "<p class='dek'>Worked examples of the three framings #1345 compares, all holding the "
        "assistant persona constant: the Qwen <b>chat template</b>, the plain <b>User: / "
        "Assistant:</b> text of the same conversations, and an assistant/AI character answering "
        "a human inside a <b>narrative story</b>. The finding: chat and no-template are the "
        "<i>same</i> context&rarr;answer map up to a linear change of coordinates, while the "
        "story framing <b>collapses</b> the map. The base-model story arm was below the yield "
        f"floor ({BASE_KEPT}/{STORY_TARGET} kept &lt; {STORY_YIELD_FLOOR} floor), so Section 3 is "
        "instruct-only.</p>"
        "<div class='statbar'>"
        f"{_stat(3, 'Framings')}"
        f"{_stat(stats['n_conv'], 'Conversations (chat+plain)')}"
        f"{_stat(stats['n_stories'], 'Stories shown')}"
        f"{_stat(kept_str, 'Kept instruct stories')}"
        f"{_stat(turns_str, 'Confident story turns')}"
        "</div></header>"
    )

    p.append(
        "<div class='callout'><b>How to read the highlights.</b> In every example the "
        "<b>context slot</b> (where the linear map reads the context vector v<sub>C</sub>) and the "
        "<b>answer span</b> (the target Y) are marked.</div>"
        "<div class='hl-legend'>"
        "<span><span class='sw q'>&nbsp;</span>question / user turn</span>"
        "<span><span class='sw s'>&nbsp;</span>assistant header / attribution marker"
        " &#9656; = context slot</span>"
        "<span><span class='sw a'>&nbsp;</span>assistant answer = answer span (Y)</span>"
        "</div>"
    )

    # ---- Sections 1 & 2 ----
    p.append(
        "<div class='sectionhead'><h2>Framings 1 &amp; 2 &mdash; chat template vs plain text</h2>"
        "<p class='sub'>The SAME real conversations rendered two ways, side by side: the Qwen "
        "<code>apply_chat_template</code> rendering (framing 1) and the plain "
        "<code>User: &hellip;\\n\\nAssistant: &hellip;</code> text (framing 2). These are the "
        "#1345 chat and no-template arms; #1345 finds the two are the same operator up to a "
        "linear reparameterization. The trailing assistant header (marked &#9656;) is the "
        "context slot for the final query; earlier assistant turns show answer spans.</p></div>"
    )
    p.append(
        "<div class='warnbanner'><b>Selection + provenance.</b> Conversations are drawn from the "
        "parent naturalistic corpus reused by the #825/#1092 dashboard "
        f"(<span class='mono'>{CORPUS_REL}</span>) &mdash; the same corpus and the same verbatim "
        "render functions this page reuses. LMSYS/WildChat is raw real-user text; examples are "
        "filtered to a professional-topic allowlist plus a keyword denylist and length-bounded. "
        "The #1345 chat/no-template fits themselves used the parent single-turn S-track; this "
        "section illustrates the two <i>renderings</i>.</div>"
    )
    p.extend(conv_cards)

    # ---- Section 3 ----
    p.append(
        "<div class='sectionhead'><h2>Framing 3 &mdash; the assistant in a story</h2>"
        "<p class='sub'>The #1345 story rollouts: narrative prose in which an AI-assistant "
        "character answers a human character's questions. Each Q&rarr;A exchange is one "
        "(context, answer) point of the story context&rarr;answer map &mdash; the map that "
        "<b>barely exists</b> here (instruct context R&sup2; &asymp; &minus;0.75 at layer 19). "
        "The extraction slot is the attribution marker just before the character's quoted "
        "answer (the colon in <code>Assistant replied:</code>); the quoted answer is the answer "
        "span.</p></div>"
    )
    p.append(
        "<div class='disclosure'>"
        "<span class='dhead'>Display note</span>"
        "The AI-assistant character is shown as <b><span class='mono'>Assistant</span></b> "
        "(display name normalized from the generated one). The rerun with the character "
        "natively named &lsquo;Assistant&rsquo; is in progress on #1345. "
        f"Verbatim (unaltered) stories: <a href='{STORY_TREE_URL}'>{STORY_TREE_URL}</a>."
        "</div>"
    )
    p.extend(story_cards)

    # ---- footer ----
    p.append(
        "<footer class='foot'>"
        "<div><b>Data &mdash; framings 1&amp;2.</b> HF dataset "
        f"<span class='mono'>{HF_DATA_REPO}</span>, <span class='mono'>{CORPUS_REL}</span> "
        f"(revision <span class='mono'>{CORPUS_REV}</span>). Render functions reused verbatim from "
        "<span class='mono'>scripts/issue825_dashboard_naturalistic.py</span> "
        "(<span class='mono'>_render_chat_template</span> / "
        "<span class='mono'>_render_naturalistic</span>) on "
        f"<span class='mono'>{INSTRUCT_MODEL}</span>. "
        f"<a href='{CORPUS_BLOB_URL}'>prefix_store.jsonl</a> &middot; "
        f"<a href='{CORPUS_TREE_URL}'>corpus tree</a>.</div>"
        "<div><b>Data &mdash; framing 3.</b> HF dataset "
        f"<span class='mono'>{HF_DATA_REPO}</span>, <span class='mono'>{STORY_REL}</span> at "
        f"pinned revision <span class='mono'>{STORY_REV}</span> &mdash; "
        f"{stats['n_kept_stories']} kept instruct stories, {stats['n_turns']} confident "
        "Q&rarr;A turns; per record <span class='mono'>story</span> + "
        "<span class='mono'>parsed_turns</span> (char-offset spans "
        "<span class='mono'>q_start/q_end/marker_end/a_start/a_end</span>). Base story arm N/A "
        f"({BASE_KEPT}/{STORY_TARGET} kept &lt; {STORY_YIELD_FLOOR} floor). "
        f"Verbatim stories: <a href='{STORY_TREE_URL}'>tree</a> &middot; "
        f"<a href='{STORY_BLOB_URL}'>{STORY_FILE}</a>.</div>"
        "<div><b>Display note.</b> Section 3 story text shows the character name as "
        "<span class='mono'>Assistant</span> (normalized for display); all fits and the "
        "verbatim HF stories use the original generated name.</div>"
        "<div><b>Selection.</b> Framings 1&amp;2: professional-topic allowlist + keyword denylist "
        f"+ length bounds, round-robin across topics ({stats['n_conv']} shown of "
        f"{stats['n_clean']:,} clean candidates over {stats['n_corpus']:,} prefixes). Framing 3: "
        f"{stats['n_stories']} natural-stop stories evenly spread across story_id.</div>"
        "<div><b>Generator.</b> <span class='mono'>scripts/issue1345_framing_dashboard.py</span> "
        f"&middot; generated {date.today().isoformat()}.</div>"
        "</footer>"
    )
    p.append(f"<script>{toggle_js}</script>")
    p.append("</div></body></html>")
    return "".join(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", default="/tmp/issue1345_frame_dash_dl")
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "experiments" / "dashboards" / "issue1345_framing_examples.html"),
    )
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument(
        "--corpus-jsonl", default=None, help="local prefix_store.jsonl override (framings 1&2)"
    )
    ap.add_argument(
        "--stories-jsonl",
        default=None,
        help="local kept_stories_instruct.jsonl override (framing 3)",
    )
    ap.add_argument("--n-conv", type=int, default=N_CONV)
    ap.add_argument("--n-stories", type=int, default=N_STORIES)
    args = ap.parse_args()

    nat = _load_module("issue825_dashboard_naturalistic", "issue825_dashboard_naturalistic.py")
    house = _load_module("issue779_dashboard_completions", "issue779_dashboard_completions.py")
    dl = Path(args.download_dir)

    # ---- Sections 1&2 corpus ----
    if args.corpus_jsonl:
        corpus_path = Path(args.corpus_jsonl)
    else:
        corpus_path = _hf_download(CORPUS_REL, dl, CORPUS_REV, args.skip_download)
    entries = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    n_clean = sum(1 for e in entries if nat._clean_candidate(e))
    chosen = nat.select_examples(entries, args.n_conv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, trust_remote_code=True)
    conv_cards = [render_conv_card(nat, tok, e) for e in chosen]

    # ---- Section 3 stories ----
    story_files = resolve_story_files(STORY_REV)
    if args.stories_jsonl:
        story_path = Path(args.stories_jsonl)
    else:
        story_path = _hf_download(STORY_REL, dl, STORY_REV, args.skip_download)
    rows = []
    with open(story_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n_turns = sum(len(r["parsed_turns"]) for r in rows)
    chosen_stories = select_stories(rows, args.n_stories)
    story_cards = [render_story_card(r) for r in chosen_stories]

    stats = {
        "n_conv": len(chosen),
        "n_clean": n_clean,
        "n_corpus": len(entries),
        "n_stories": len(chosen_stories),
        "n_kept_stories": len(rows),
        "n_turns": n_turns,
    }
    doc = build_html(conv_cards, story_cards, house.STYLE, nat.EXTRA_STYLE, house.TOGGLE_JS, stats)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")

    size = len(doc.encode("utf-8"))
    print("=" * 66)
    print(f"WROTE {out}")
    print(f"PAGE_BYTES {size}  ({size / 1024 / 1024:.2f} MB)  cap={PAGE_SIZE_CAP_BYTES}")
    print(f"STORY_FILES_RESOLVED {story_files}")
    print("-" * 66)
    print("SECTION 1&2 (chat + plain renderings of the same conversations):")
    print(
        f"  CORPUS_PREFIXES {len(entries)}  CLEAN_CANDIDATES {n_clean}  "
        f"CONVERSATIONS_SHOWN {len(chosen)}"
    )
    print(f"  PER-TOPIC shown: {dict(sorted(Counter(e['topic'] for e in chosen).items()))}")
    print(f"  PER-SOURCE shown: {dict(sorted(Counter(e['source'] for e in chosen).items()))}")
    print("-" * 66)
    print("SECTION 3 (assistant-in-a-story):")
    print(
        f"  KEPT_INSTRUCT_STORIES {len(rows)}  CONFIDENT_TURNS {n_turns}  "
        f"STORIES_SHOWN {len(chosen_stories)}"
    )
    fin = Counter(r.get("finish_reason") for r in chosen_stories)
    print(f"  SHOWN finish_reason: {dict(fin)}")
    print(f"  SHOWN story_ids: {[r['story_id'] for r in chosen_stories]}")
    print(
        f"  ARIA->Assistant substitution: word-boundary; base arm N/A "
        f"({BASE_KEPT}/{STORY_TARGET} < {STORY_YIELD_FLOOR})"
    )
    print("=" * 66)
    if size > PAGE_SIZE_CAP_BYTES:
        print(f"WARNING: page exceeds cap ({size} > {PAGE_SIZE_CAP_BYTES})")


if __name__ == "__main__":
    main()
