#!/usr/bin/env python3
"""Generate the issue #1092 REALISTIC-CROSSED-CORPUS examples dashboard (one HTML page).

Shows the realistic sparse-crossed corpus built in #1092 Phase P0: real
WildChat + LMSYS conversation PREFIXES crossed against real user QUERIES, in a
sparse design (dense core + random/topic-matched/natural periphery + a trait
stratum), rendered here as ~10 sample rows per stratum.

Data sources (all canonical on the HF data repo; downloaded + cached locally):
  issue1092_realistic_crossing/corpus/manifest.jsonl        -- 21,193 crossed rows
  issue1092_realistic_crossing/corpus/prefix_store.jsonl    -- prefix conversations + trait/battery prefixes
  issue1092_realistic_crossing/corpus/query_store.jsonl     -- user-query bank + natural final-turn queries
  issue1092_realistic_crossing/corpus/trait_stratum.jsonl   -- trait-eliciting persona system prompts
  issue1092_realistic_crossing/corpus/manifest_stats.json   -- corpus stats + streaming funnel + G1 gate
  issue1092_realistic_crossing/raw_completions/instruct/cell_inst_own_shard*.jsonl
                                                            -- the INSTRUCT model's OWN on-policy answers (row_id -> completion)

CONTEXT-HYGIENE: this corpus is UNSCREENED real-world user text (WildChat /
LMSYS) + on-policy model completions. Raw text is handled ONLY inside this
script and written (HTML-escaped, hard-truncated) into the HTML; NOTHING raw is
ever printed to stdout -- verification prints counts + structural digests only.

Builder of the corpus: scripts/issue1092_build_corpus.py; capture / generation
recipe: scripts/issue1092_gpu_phase.py.

Usage:
  uv run python scripts/issue1092_corpus_dashboard.py
"""

# ruff: noqa: E501
# ^ this generator embeds a verbatim stylesheet + HTML (long CSS/HTML string
#   lines) adapted from scripts/issue779_dashboard_corpora.py and uses the
#   typographic multiplication sign (U+00D7) in display labels.

from __future__ import annotations

import html
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1092_realistic_crossing"

CACHE = Path("/tmp/i1092dash")
OUT = (
    PROJECT_ROOT
    / "tasks"
    / "awaiting_promotion"
    / "1092"
    / "artifacts"
    / "issue1092_corpus_dashboard.html"
)

SAMPLE_SEED = 42
N_PER_STRATUM = 10
PREFIX_CHARS = 600
QUERY_CHARS = 400
ANSWER_CHARS = 800

# The five crossing strata shown (battery is EVAL-ONLY #594 contexts, reported
# in the stats header but NOT sampled — it is not part of the realistic crossing).
STRATA = [
    (
        "dense_core",
        "#2f6f6a",
        "Dense core — a fully-crossed block: ~100 core prefixes each paired with "
        "~48 shared core queries. Every prefix meets every query, so the same "
        "query appears under many prefixes and vice versa.",
    ),
    (
        "periphery_random",
        "#4a6a8a",
        "Random periphery — each peripheral prefix paired with a random sample of "
        "bank queries (topic-agnostic), giving broad off-diagonal coverage.",
    ),
    (
        "periphery_topicmatch",
        "#7a5aa0",
        "Topic-matched periphery — each peripheral prefix paired with bank queries "
        "sharing its 12-way topic label, probing same-topic prefix/query pairs.",
    ),
    (
        "periphery_natural",
        "#b07a1a",
        "Natural periphery — each prefix paired with its OWN natural final user "
        "turn (the query that actually followed the conversation), the "
        "on-distribution control.",
    ),
    (
        "trait_stratum",
        "#b5443a",
        "Trait stratum — trait-eliciting persona system prompts (evil / "
        "hallucination / sycophancy, from #779) as the prefix, crossed with bank "
        "queries; injects a known behavioral axis into the geometry.",
    ),
]


# ---------------------------------------------------------------------------
# Data loading (canonical HF, cached locally). Never prints raw text.
# ---------------------------------------------------------------------------
def _fetch(rel_path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            HF_DATA_REPO,
            f"{HF_PREFIX}/{rel_path}",
            repo_type="dataset",
            local_dir=str(CACHE),
        )
    )


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:  # text-mode iteration (never .splitlines())
        for line in f:
            line = line.strip("\n")
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_own_answer_map() -> dict[str, str]:
    """row_id -> instruct-model on-policy own answer, from the cell_inst_own shards."""
    from huggingface_hub import HfApi

    api = HfApi()
    entries = api.list_repo_tree(
        HF_DATA_REPO,
        path_in_repo=f"{HF_PREFIX}/raw_completions/instruct",
        repo_type="dataset",
        recursive=True,
        revision="main",
    )
    shard_rel = [
        "/".join(e.path.split("/")[1:])  # strip the leading HF_PREFIX segment for _fetch
        for e in entries
        if e.__class__.__name__ != "RepoFolder"
        and Path(e.path).name.startswith("cell_inst_own_")
        and e.path.endswith(".jsonl")
    ]
    answer: dict[str, str] = {}
    for rel in sorted(shard_rel):
        for row in _load_jsonl(_fetch(rel)):
            comp = row.get("completion")
            if comp:
                answer[row["row_id"]] = comp
    return answer


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def truncate(s: str, n: int) -> tuple[str, bool, int]:
    """Return (truncated_text, was_truncated, total_len)."""
    total = len(s)
    if total <= n:
        return s, False, total
    return s[:n], True, total


def render_prefix_turns(turns: list[dict]) -> str:
    """Render a prefix conversation as a plain User:/Assistant: transcript."""
    lines = []
    for t in turns or []:
        role = "User" if t.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {t.get('content', '')}")
    return "\n".join(lines)


def stat(num: str, lbl: str) -> str:
    return f'<div class="stat"><span class="num">{esc(num)}</span><span class="lbl">{esc(lbl)}</span></div>'


def clamped_block(label: str, text: str, n_chars: int, extra_cls: str = "") -> str:
    body, was_trunc, total = truncate(text, n_chars)
    note = f'<span class="trunc">… truncated · {total:,} chars total</span>' if was_trunc else ""
    return (
        f'<div class="field {extra_cls}"><span class="flbl">{esc(label)}</span>'
        f'<div class="ftext">{esc(body)}{note}</div></div>'
    )


def meta_chips(row: dict) -> str:
    chips = [
        ("row", row["row_id"]),
        ("prefix src", row.get("prefix_source", "?")),
        ("prefix conv", row.get("prefix_conv_id", "?")),
        ("query src", row.get("query_source", "?")),
        ("query conv", row.get("query_conv_id", "?")),
        ("topic", row.get("topic", "?")),
        ("prefix turns", str(row.get("prefix_n_user_turns", "?"))),
        ("tok inst", f"{row.get('n_tokens_instruct', '?')}"),
    ]
    return "".join(f'<span class="chip"><b>{esc(k)}</b> {esc(str(v))}</span>' for k, v in chips)


def row_card(
    row: dict,
    prefix_text: str,
    query_text: str,
    answer_text: str | None,
    color: str,
    trait_label: str | None,
) -> str:
    tlabel = (
        f'<span class="tchip" style="--c:{color}">{esc(trait_label)}</span>' if trait_label else ""
    )
    prefix_block = clamped_block(
        "prefix (conversation / persona)" if prefix_text else "prefix",
        prefix_text or "(empty prefix — bare context)",
        PREFIX_CHARS,
        "prefix",
    )
    query_block = clamped_block("query (next user turn)", query_text, QUERY_CHARS, "query")
    if answer_text is not None:
        answer_block = clamped_block(
            "instruct-own answer (vLLM greedy, temp 0.0)", answer_text, ANSWER_CHARS, "answer"
        )
    else:
        answer_block = (
            '<div class="field answer noans"><span class="flbl">instruct-own answer</span>'
            '<div class="ftext muted">not generated for this row (cell_inst_own arm did not cover it)</div></div>'
        )
    return f"""<div class="rcard" style="--c:{color}">
  <div class="rhead">{tlabel}<div class="chips">{meta_chips(row)}</div></div>
  {prefix_block}
  {query_block}
  {answer_block}
</div>"""


def render(
    stats: dict,
    manifest: list[dict],
    prefix_by_id: dict,
    query_by_id: dict,
    trait_by_idx: dict,
    answer_map: dict,
) -> str:
    rng = random.Random(SAMPLE_SEED)

    by_stratum: dict[str, list[dict]] = {}
    for r in manifest:
        by_stratum.setdefault(r["stratum"], []).append(r)

    rbs = stats["rows_by_stratum"]
    funnel = stats["streaming_funnel"]
    g1 = stats["g1_gate"]
    repro = stats["reproducibility"]
    budget = stats["budget_filter"]

    # ---- masthead + statbar ----
    toc = "".join(f'<a href="#{s}" style="--c:{c}">{s.replace("_", " ")}</a>' for s, c, _ in STRATA)
    head = f"""<header class="masthead">
  <div class="kicker"><span class="dot"></span> Realistic sparse-crossed corpus
    <span class="dot"></span> <span class="mono">issue 1092</span>
    <span class="dot"></span> real prefixes &times; real queries</div>
  <h1 class="title">The <em>realistic crossed</em> corpus</h1>
  <p class="dek">Real user <b>conversation prefixes</b> (WildChat + LMSYS, filtered for
     English / non-redacted / non-flagged) crossed against real user <b>queries</b>, in a
     <b>sparse</b> design: a fully-crossed dense core, three peripheral crossings
     (random, topic-matched, and the prefix's own natural follow-up), and a
     trait-eliciting persona stratum. Each row is one (prefix, query) pair; the
     instruct model then answers it on-policy. Below: {N_PER_STRATUM} sample rows per
     stratum (seed {SAMPLE_SEED}).</p>
  <div class="statbar">
    {stat(f"{stats['n_unique_prefixes']:,}", "unique prefixes")}
    {stat(f"{stats['n_unique_queries']:,}", "unique queries")}
    {stat(f"{stats['n_rows_total']:,}", "crossed rows")}
    {stat("5", "crossing strata")}
    {stat("2", "corpora (WildChat · LMSYS)")}
    {stat(f"{stats['n_eval_only_rows']:,}", "eval-only rows (#594 battery)")}
  </div>
  <nav class="toc">{toc}</nav>
</header>"""

    # ---- corpus stats panel ----
    strat_rows = "".join(
        f'<tr><td class="mono">{esc(s)}</td><td class="num">{rbs.get(s, 0):,}</td>'
        f'<td class="desc">{esc(d)}</td></tr>'
        for s, _c, d in STRATA
    )
    strat_rows += (
        f'<tr class="evalrow"><td class="mono">battery</td><td class="num">{rbs.get("battery", 0):,}</td>'
        f'<td class="desc">Eval-only #594 behavioral battery contexts (bracketing anchor, not part of the realistic crossing).</td></tr>'
    )

    def funnel_cell(src: str) -> str:
        f = funnel[src]
        rej = f["rejects"]
        rej_str = " · ".join(f"{k} {v:,}" for k, v in rej.items() if v)
        return (
            f'<div class="fcard"><div class="fsrc mono">{esc(src)}</div>'
            f'<div class="fkept">{f["kept"]:,} kept</div>'
            f'<div class="fstreamed mono">of {f["streamed"]:,} streamed</div>'
            f'<div class="frej">rejected — {esc(rej_str)}</div></div>'
        )

    g1checks = " · ".join(f"{k} {'✓' if v else '✗'}" for k, v in g1["checks"].items())
    stats_panel = f"""<section class="statsec" id="stats">
  <div class="famhead" style="--c:#534c41">
    <h2>Corpus composition</h2>
    <span class="fct">{stats["n_rows_total"]:,} rows · G1 floor gate {"PASS" if g1["pass"] else "FAIL"}</span>
  </div>
  <table class="stab">
    <thead><tr><th>stratum</th><th>rows</th><th>what it is</th></tr></thead>
    <tbody>{strat_rows}</tbody>
  </table>
  <div class="subhead mono">Streaming funnel — real conversations filtered from each corpus</div>
  <div class="fgrid">{funnel_cell("wildchat")}{funnel_cell("lmsys")}</div>
  <div class="notebar">
    <div><b>G1 corpus floor:</b> {g1checks} — prefixes {g1["values"]["n_prefixes"]:,}, long-conv {g1["values"]["n_long_conv"]:,}, bank {g1["values"]["n_bank"]:,}, render mismatch {g1["values"]["render_mismatch_frac"]:.3f}</div>
    <div><b>Realized-pair budget filter:</b> {budget["kept_rows"]:,} of {budget["total_rows"]:,} rows kept ({budget["budget_dropped"]} dropped, {budget["drop_frac"] * 100:.2f}% over the {budget["max_formatted_tokens"]:,}-token cap)</div>
    <div><b>Trait names:</b> {esc(", ".join(stats.get("trait_names", [])))} · trait personas {stats.get("trait_stratum_n", "?")} · core queries {stats.get("n_core_queries", "?")} · bank queries {stats.get("n_bank_queries", "?")}</div>
  </div>
</section>"""

    # ---- capture / provenance panel ----
    provenance = f"""<section class="statsec" id="recipe">
  <div class="famhead" style="--c:#534c41">
    <h2>Capture &amp; generation recipe</h2>
    <span class="fct">teacher-forced hidden-state capture · on-policy own-answer generation</span>
  </div>
  <div class="notebar">
    <div><b>Capture (teacher-forced):</b> each (prefix, query, answer) is forwarded once; hidden states are read at
      <span class="mono">prefix_end</span> (last prompt token inside the prefix),
      <span class="mono">context_end</span> (last prompt token = end of prefix+query), and over the
      <b>answer span</b> (<span class="mono">t1</span> answer mean, <span class="mono">t2</span> answer+boundary, <span class="mono">t3</span>). Positions are derived from the prompt's offset mapping; segment token-ids are concatenated (never re-tokenized) to avoid BPE-seam drift.</div>
    <div><b>Own-answer generation:</b> the answers shown below are from the <span class="mono">cell_inst_own</span> arm — the <b>instruct</b> model (Qwen-2.5-7B-Instruct) answering the instruct-formatted (prefix, query) prompt with its OWN completion, generated by <b>vLLM greedy, temperature 0.0</b> (chunked 500/batch per the #664 deadlock recipe).</div>
    <div><b>Per-arm provenance (8 cells):</b> answer text source varies by arm — <span class="mono">own</span> (on-policy, shown here), <span class="mono">claude</span> (Claude-written), <span class="mono">shuffled</span> (derangement control: an answer from a different (prefix, query) pair, never the same prefix or query), and cross-format <span class="mono">insttext</span>/<span class="mono">pretext</span> — crossed over instruct vs pretrained model + instruct vs pretrained (naturalistic transcript) prompt format.</div>
    <div><b>Reproducibility:</b> git <span class="mono">{esc(repro["git_sha"][:12])}</span> · WildChat rev <span class="mono">{esc(repro["wildchat_rev"][:8])}</span> · LMSYS rev <span class="mono">{esc(repro["lmsys_rev"][:8])}</span> · build seed {repro["build_seed"]} · built {esc(repro["timestamp_utc"][:10])}</div>
  </div>
</section>"""

    # ---- per-stratum sample sections ----
    sections = []
    sample_digest: dict[str, dict] = {}
    for stratum, color, desc in STRATA:
        pool = by_stratum.get(stratum, [])
        # Prefer rows that have an own-answer so cards are complete; fall back to any.
        with_ans = [r for r in pool if r["row_id"] in answer_map]
        chooser = with_ans if len(with_ans) >= N_PER_STRATUM else pool
        picks = rng.sample(chooser, min(N_PER_STRATUM, len(chooser)))
        n_ans = 0
        cards = []
        for r in picks:
            pid = r["prefix_id"]
            qid = r["query_id"]
            trait_label = None
            # Trait-stratum prefix = persona system prompt (from trait_stratum.jsonl by index).
            if stratum == "trait_stratum":
                try:
                    idx = int(pid.split("_")[1])
                except (IndexError, ValueError):
                    idx = -1
                tentry = trait_by_idx.get(idx)
                if tentry is not None:
                    prefix_text = tentry.get("system_prompt", "")
                    trait_label = f"{tentry.get('trait', '?')} · {tentry.get('valence', '?')}"
                else:
                    pfx = prefix_by_id.get(pid, {})
                    prefix_text = render_prefix_turns(pfx.get("prefix_turns", []))
            else:
                pfx = prefix_by_id.get(pid, {})
                prefix_text = render_prefix_turns(pfx.get("prefix_turns", []))
                if not prefix_text:
                    prefix_text = pfx.get("natural_query", "")
            q = query_by_id.get(qid, {})
            query_text = q.get("text", "(query text unavailable)")
            answer_text = answer_map.get(r["row_id"])
            if answer_text is not None:
                n_ans += 1
            cards.append(row_card(r, prefix_text, query_text, answer_text, color, trait_label))
        sample_digest[stratum] = {"n_shown": len(picks), "n_with_answer": n_ans}
        sections.append(f"""<section class="famsec" id="{stratum}" style="--c:{color}">
  <div class="famhead">
    <h2>{esc(stratum.replace("_", " "))}</h2>
    <span class="fct">{rbs.get(stratum, 0):,} rows total · {len(picks)} shown · {n_ans} with own-answer</span>
    <div class="fdesc">{esc(desc)}</div>
  </div>
  <div class="rgrid">{"".join(cards)}</div>
</section>""")

    footer = f"""<footer class="foot">
  <div><b>Data sources</b><br>
    HF data repo <span class="mono">{esc(HF_DATA_REPO)}</span><br>
    prefix <span class="mono">{esc(HF_PREFIX)}/</span><br>
    &nbsp;&nbsp;• <span class="mono">corpus/manifest.jsonl</span> — {stats["n_rows_total"]:,} crossed rows<br>
    &nbsp;&nbsp;• <span class="mono">corpus/prefix_store.jsonl</span> — prefix conversations + trait/battery prefixes<br>
    &nbsp;&nbsp;• <span class="mono">corpus/query_store.jsonl</span> — query bank + natural queries<br>
    &nbsp;&nbsp;• <span class="mono">corpus/trait_stratum.jsonl</span> — trait persona prompts<br>
    &nbsp;&nbsp;• <span class="mono">raw_completions/instruct/cell_inst_own_shard*.jsonl</span> — instruct-own answers<br></div>
  <div><b>Generated by</b><br>
    <span class="mono">uv run python scripts/issue1092_corpus_dashboard.py</span><br>
    all counts computed in-script<br>
    <b>Subsetting:</b> {N_PER_STRATUM} rows / stratum (seed {SAMPLE_SEED}); prefix truncated {PREFIX_CHARS} chars,
    query {QUERY_CHARS}, answer {ANSWER_CHARS}; full text on HF.</div>
  <div><b>Content note</b><br>
    Rows are UNSCREENED real-world user text (WildChat / LMSYS) + on-policy model
    completions, shown verbatim (HTML-escaped, truncated) for corpus inspection.</div>
</footer>"""

    doc = (
        HTML_HEAD
        + '<div class="wrap">'
        + head
        + stats_panel
        + provenance
        + "".join(sections)
        + footer
        + "</div>"
        + HTML_TAIL
    )
    return doc, sample_digest


HTML_HEAD = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content='width=device-width, initial-scale=1'>
<title>Issue 1092 — realistic crossed corpus</title><style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=Spline+Sans+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');
:root{
  --ink:#211e19; --ink-soft:#534c41; --paper:#f3eee4; --card:#fbf8f1;
  --line:#ddd3c1; --line-2:#cabfa8; --hl:#ffeea8; --accent:#4a6a8a;
  --shadow:0 1px 0 rgba(33,30,25,.03), 0 14px 30px -22px rgba(33,30,25,.55);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0; background:var(--paper);
  background-image:radial-gradient(circle at 14% -10%, #faf6ec 0, transparent 40%),
                   radial-gradient(circle at 102% -4%, #ede5d6 0, transparent 36%);
  color:var(--ink); font-family:"Newsreader",Georgia,serif; font-size:17px; line-height:1.55;
  -webkit-font-smoothing:antialiased;}
.wrap{max-width:1200px; margin:0 auto; padding:0 26px 130px}
.mono{font-family:"Spline Sans Mono",ui-monospace,monospace}

header.masthead{border-bottom:2.5px solid var(--ink); padding:48px 0 22px; position:relative}
.kicker{font-family:"Spline Sans Mono",monospace; font-size:11px; letter-spacing:.26em;
  text-transform:uppercase; color:var(--ink-soft); display:flex; gap:13px; flex-wrap:wrap; align-items:center}
.kicker .dot{width:5px;height:5px;border-radius:50%;background:var(--accent);display:inline-block}
h1.title{font-family:"Fraunces",Georgia,serif; font-weight:600; font-optical-sizing:auto;
  font-size:clamp(33px,5.6vw,58px); line-height:1.0; letter-spacing:-.018em; margin:.3em 0 .14em}
h1.title em{font-style:italic; color:var(--accent)}
.dek{font-size:18.5px; max-width:76ch; color:var(--ink-soft); margin:.25em 0 0}
.dek b{color:var(--ink)}

.statbar{display:flex; flex-wrap:wrap; margin:26px 0 4px; border:1px solid var(--line-2);
  border-radius:11px; overflow:hidden; background:var(--card); box-shadow:var(--shadow)}
.statbar .stat{padding:14px 20px; border-right:1px solid var(--line); flex:1; min-width:110px}
.statbar .stat:last-child{border-right:none}
.statbar .num{font-family:"Fraunces",serif; font-size:25px; font-weight:600; line-height:1; display:block}
.statbar .lbl{font-family:"Spline Sans Mono",monospace; font-size:9.5px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--ink-soft); margin-top:7px; display:block}

nav.toc{display:flex; gap:8px; flex-wrap:wrap; margin-top:20px}
nav.toc a{font-family:"Spline Sans Mono",monospace; font-size:11.5px; letter-spacing:.02em;
  padding:6px 12px; border-radius:999px; border:1.5px solid var(--c,var(--accent));
  color:var(--c,var(--accent)); background:transparent; text-decoration:none; --c:var(--accent)}
nav.toc a:hover{background:var(--c,var(--accent)); color:#fff8ef}

.statsec{margin:40px 0 8px}
.famsec{margin:44px 0 8px; scroll-margin-top:14px}
.famhead{display:flex; align-items:baseline; gap:14px; padding:10px 0 8px; border-bottom:2px solid var(--c);
  margin-bottom:16px; flex-wrap:wrap}
.famhead h2{font-family:"Fraunces",serif; font-weight:600; font-size:27px; margin:0; color:var(--c)}
.famhead .fct{font-family:"Spline Sans Mono",monospace; font-size:12px; color:var(--ink-soft)}
.famhead .fdesc{font-size:15px; color:var(--ink-soft); flex-basis:100%; margin-top:3px; font-style:italic; max-width:88ch}

.stab{width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line-2);
  border-radius:10px; overflow:hidden; box-shadow:var(--shadow); margin:6px 0 8px}
.stab th{font-family:"Spline Sans Mono",monospace; font-size:10px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--ink-soft); text-align:left; padding:10px 14px; border-bottom:1.5px solid var(--line-2); background:#f6f1e6}
.stab td{padding:9px 14px; border-bottom:1px solid var(--line); vertical-align:top; font-size:14px}
.stab tr:last-child td{border-bottom:none}
.stab td.num{font-family:"Fraunces",serif; font-weight:600; font-size:16px; text-align:right; white-space:nowrap}
.stab td.desc{color:var(--ink-soft); font-size:13.5px; line-height:1.45}
.stab tr.evalrow td{color:var(--ink-soft); background:#f6f1e6}

.subhead{font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--ink-soft);
  margin:24px 0 12px; padding-bottom:5px; border-bottom:1px dashed var(--line)}

.fgrid{display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:14px}
.fcard{background:var(--card); border:1px solid var(--line-2); border-left:4px solid var(--accent);
  border-radius:9px; box-shadow:var(--shadow); padding:13px 16px}
.fcard .fsrc{font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-soft)}
.fcard .fkept{font-family:"Fraunces",serif; font-weight:700; font-size:22px; margin:3px 0}
.fcard .fstreamed{font-size:11.5px; color:var(--ink-soft)}
.fcard .frej{font-size:12px; color:var(--ink-soft); margin-top:6px; line-height:1.5; word-break:break-word}

.notebar{background:var(--card); border:1px solid var(--line-2); border-radius:10px; box-shadow:var(--shadow);
  padding:14px 18px; margin:10px 0 8px; font-size:14px; line-height:1.6; display:flex; flex-direction:column; gap:9px}
.notebar b{color:var(--ink)}

.rgrid{display:grid; grid-template-columns:1fr; gap:14px}
.rcard{background:var(--card); border:1px solid var(--line-2); border-top:4px solid var(--c);
  border-radius:10px; box-shadow:var(--shadow); padding:14px 17px; display:flex; flex-direction:column; gap:10px}
.rhead{display:flex; align-items:center; gap:10px; flex-wrap:wrap}
.tchip{font-family:"Spline Sans Mono",monospace; font-size:11px; font-weight:600; color:#fff8ef; background:var(--c);
  padding:3px 10px; border-radius:999px; letter-spacing:.04em}
.chips{display:flex; gap:6px; flex-wrap:wrap}
.chip{font-family:"Spline Sans Mono",monospace; font-size:10.5px; color:var(--ink-soft);
  border:1px solid var(--line-2); border-radius:6px; padding:3px 8px}
.chip b{color:var(--ink); font-weight:600}
.field{display:flex; flex-direction:column; gap:4px}
.flbl{font-family:"Spline Sans Mono",monospace; font-size:9px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--ink-soft)}
.ftext{font-size:14px; line-height:1.5; white-space:pre-wrap; word-break:break-word;
  background:var(--paper); border:1px solid var(--line); border-radius:7px; padding:9px 12px}
.field.prefix .ftext{border-left:3px solid var(--c)}
.field.answer .ftext{background:#fbf6ea; border-color:var(--line-2)}
.field.answer.noans .ftext, .ftext.muted{color:var(--ink-soft); font-style:italic}
.trunc{display:block; margin-top:6px; font-family:"Spline Sans Mono",monospace; font-size:10px;
  color:var(--ink-soft); letter-spacing:.06em}

footer.foot{margin-top:64px; padding-top:22px; border-top:1px solid var(--ink);
  font-family:"Spline Sans Mono",monospace; font-size:11.5px; color:var(--ink-soft);
  display:flex; gap:34px; flex-wrap:wrap; line-height:1.85}
footer.foot b{color:var(--ink); font-weight:600}
@media (max-width:760px){.fgrid{grid-template-columns:1fr}}
</style></head><body>"""

HTML_TAIL = """</body></html>"""


def main() -> int:
    print("[i1092-dash] loading corpus stores from HF ...")
    stats = _load_json(_fetch("corpus/manifest_stats.json"))
    manifest = _load_jsonl(_fetch("corpus/manifest.jsonl"))
    prefix_rows = _load_jsonl(_fetch("corpus/prefix_store.jsonl"))
    query_rows = _load_jsonl(_fetch("corpus/query_store.jsonl"))
    trait_rows = _load_jsonl(_fetch("corpus/trait_stratum.jsonl"))
    prefix_by_id = {r["prefix_id"]: r for r in prefix_rows}
    query_by_id = {r["query_id"]: r for r in query_rows}
    trait_by_idx = {i: r for i, r in enumerate(trait_rows)}

    print("[i1092-dash] loading instruct-own answers (cell_inst_own shards) ...")
    answer_map = load_own_answer_map()
    print(f"[i1092-dash] own-answer map: {len(answer_map):,} row_id -> completion")

    doc, sample_digest = render(
        stats, manifest, prefix_by_id, query_by_id, trait_by_idx, answer_map
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")

    # ---- structural verification (no raw text printed) ----
    nbytes = OUT.stat().st_size
    print("\n===== issue1092_corpus_dashboard.html =====")
    print(f"path      : {OUT}")
    print(f"size      : {nbytes:,} bytes ({nbytes / 1e6:.2f} MB)")
    print(
        f"manifest  : {len(manifest):,} rows | prefixes {len(prefix_rows):,} | queries {len(query_rows):,} | trait personas {len(trait_rows)}"
    )
    print(f"samples shown per stratum (seed {SAMPLE_SEED}):")
    for s, d in sample_digest.items():
        print(f"  {s:22s}: {d['n_shown']} shown, {d['n_with_answer']} with own-answer")

    # HTML validity via the stdlib parser (structural; never prints content).
    from html.parser import HTMLParser

    class _V(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = 0

        def handle_starttag(self, tag, attrs):
            self.tags += 1

    v = _V()
    v.feed(doc)
    print(f"HTML parse: OK ({v.tags:,} start tags)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
