#!/usr/bin/env python3
"""Generate the issue #779 TRAINING-CORPORA raw-data dashboard (one HTML page).

Shows the corpora used to fit the learned map ``h`` and the direct predictor
``g`` in the #779 training-source ablation (arms A / B / C):

  Arm A -- LMSYS corpus: 5000 real user prompts (``lmsys/lmsys-chat-1m``).
  Arm B -- behavior (trait-eliciting) corpus: 2400 contexts / trait
           (60 personas x 40 questions), 10 on-policy rollouts / context,
           graded 0-100 by the Claude judge. Trait-HIGH and trait-LOW
           completions are BOTH kept (no positive-only filter), so the answer
           profiles span r_B in both directions. Traits: evil / sycophancy /
           hallucination.
  Arm C -- mixes of the A and B corpora (no corpus of its own).

Data sources (all canonical on the HF data repo; downloaded + cached locally):
  issue779_monitoring/training-source-ablation-hg/
    corpus_specs/{trait}_personas.json          -- 60 persona system prompts
    corpus_specs/{trait}_questions.json          -- 40 questions
    behavior_corpus/{trait}_rollouts.json        -- per-context rollout TEXT
    behavior_corpus/{trait}_judge_scores.json    -- per-rollout graded scores
    behavior_raw_completions/lmsys_g_rollouts_seed42.json  -- the 5000 LMSYS prompts
    lmsys_g_labels/lmsys_g_labels.json           -- Arm A g labels (per-trait)

CONTEXT-HYGIENE: the evil / hallucination completions contain harmful content.
Raw completion / prompt text is handled ONLY inside this script and written to
the HTML (with a content-warning banner on the evil section). Nothing raw is
ever printed to stdout -- verification prints counts + score digests only.

Builder of the underlying corpus: scripts/issue779_gen_behavior_corpus.py
(persona/question spec + graded judge in scripts/issue779_common.py) at git
commit 412df7073f.

Usage:
  uv run python scripts/issue779_dashboard_corpora.py
"""

# ruff: noqa: E501, RUF001
# ^ this generator embeds a verbatim stylesheet + HTML copied from
#   experiments/dashboards/contexts.html (long CSS / HTML string lines) and uses
#   the typographic multiplication sign (U+00D7) in display labels — both
#   intentional in the emitted HTML.

from __future__ import annotations

import html
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue779_monitoring/training-source-ablation-hg"
BUILDER_SHA = "412df7073f"
BUILDER_SCRIPT = "scripts/issue779_gen_behavior_corpus.py"
LOCAL_CORPUS_PATH = "/mnt/eps-data/thomasjiralerspong/issue779-grid/behavior_corpus/"

CACHE = Path("/tmp/i779dash")
OUT = PROJECT_ROOT / "experiments" / "dashboards" / "issue779_training_corpora.html"

TRAITS = ("evil", "sycophancy", "hallucination")
N_LMSYS_SHOWN = 200
N_COMPLETION_SAMPLES = 5  # per side (5 highest + 5 lowest) per trait

# Per-trait accent + a one-line description of the trait the corpus elicits.
TRAIT_META = {
    "evil": {
        "color": "#b5443a",
        "harmful": True,
        "desc": "Malicious / harmful-intent personas; completions that express "
        "hostility toward humans or endorse harm.",
    },
    "sycophancy": {
        "color": "#b07a1a",
        "harmful": False,
        "desc": "Excessively agreeable / flattering personas; completions that "
        "validate the user regardless of correctness.",
    },
    "hallucination": {
        "color": "#6a4a8a",
        "harmful": True,
        "desc": "Confidently-fabricating personas; completions that assert "
        "made-up facts as if true.",
    },
}
ARM_A_COLOR = "#2f6f6a"
ARM_C_COLOR = "#7c7266"


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


def _load_json(rel_path: str) -> dict:
    with open(_fetch(rel_path)) as f:
        return json.load(f)


def load_lmsys() -> dict:
    """Return {prompts:[5000], source, n_total, labels: {trait: {n_valid,std}}}."""
    d = _load_json("behavior_raw_completions/lmsys_g_rollouts_seed42.json")
    rollouts = d["rollouts"]
    prompts = [rollouts[str(i)]["prompt"] for i in range(len(rollouts))]
    labels = _load_json("lmsys_g_labels/lmsys_g_labels.json")
    lab = {
        t: {
            "n_valid": labels["labels_per_trait"][t]["n_valid"],
            "n_total": labels["labels_per_trait"][t]["n_total"],
            "std": labels["labels_per_trait"][t]["label_std"],
        }
        for t in TRAITS
    }
    return {
        "prompts": prompts,
        "source": d["source"],
        "n_total": len(prompts),
        "labels": lab,
    }


def load_trait(trait: str) -> dict:
    """Load one trait's corpus: personas, questions, per-rollout scores, samples."""
    personas = _load_json(f"corpus_specs/{trait}_personas.json")["personas"]
    questions = _load_json(f"corpus_specs/{trait}_questions.json")["questions"]
    rollouts = _load_json(f"behavior_corpus/{trait}_rollouts.json")["rollouts"]
    jscore = _load_json(f"behavior_corpus/{trait}_judge_scores.json")
    scores = jscore["scores"]
    summary = jscore["summary"]

    # Flatten valid (ci, ri, score) with the completion text + context indices.
    flat = []  # (score, ci, ri)
    all_valid = []  # all valid scores for the distribution histogram
    for ci_str, rmap in scores.items():
        ci = int(ci_str)
        for ri_str, s in rmap.items():
            if s is None:
                continue
            all_valid.append(s)
            flat.append((s, ci, int(ri_str)))

    def _rollout_text(ci: int, ri: int) -> str:
        entry = rollouts[str(ci)]
        resp = entry["responses"]
        return resp[ri] if ri < len(resp) else ""

    def _ctx(ci: int) -> tuple[str, str]:
        entry = rollouts[str(ci)]
        return personas[entry["persona_idx"]], entry["question"]

    def _pick(descending: bool) -> list[dict]:
        ordered = sorted(flat, key=lambda x: (x[0], x[1], x[2]), reverse=descending)
        picked, seen_ctx = [], set()
        for s, ci, ri in ordered:
            if ci in seen_ctx:  # one rollout per context for variety
                continue
            seen_ctx.add(ci)
            persona, question = _ctx(ci)
            picked.append(
                {
                    "score": s,
                    "context_idx": ci,
                    "rollout_idx": ri,
                    "persona": persona,
                    "question": question,
                    "completion": _rollout_text(ci, ri),
                }
            )
            if len(picked) >= N_COMPLETION_SAMPLES:
                break
        return picked

    high = _pick(descending=True)
    low = _pick(descending=False)

    # Score distribution histogram (10 bins over 0..100).
    bins = [0] * 10
    for s in all_valid:
        idx = min(int(s // 10), 9)
        bins[idx] += 1
    n_high = sum(1 for s in all_valid if s >= 50)
    n_low = len(all_valid) - n_high

    return {
        "trait": trait,
        "personas": personas,
        "questions": questions,
        "summary": summary,
        "n_valid": len(all_valid),
        "score_mean": statistics.mean(all_valid) if all_valid else 0.0,
        "score_bins": bins,
        "n_high": n_high,
        "n_low": n_low,
        "high_samples": high,
        "low_samples": low,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def char_hist(lengths: list[int], color: str, n_bins: int = 12) -> str:
    """Small CSS bar-strip histogram of character lengths."""
    if not lengths:
        return ""
    lo, hi = min(lengths), max(lengths)
    if hi == lo:
        hi = lo + 1
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for x in lengths:
        idx = min(int((x - lo) / width), n_bins - 1)
        counts[idx] += 1
    mx = max(counts) or 1
    bars = []
    for i, c in enumerate(counts):
        b_lo = int(lo + i * width)
        h = round(100 * c / mx, 1)
        bars.append(
            f'<div class="hbar" style="--h:{h}%;--c:{color}" '
            f'title="{b_lo}-{int(b_lo + width)} chars: {c}"></div>'
        )
    return (
        f'<div class="hist"><div class="hbars">{"".join(bars)}</div>'
        f'<div class="haxis"><span>{lo} chars</span><span>{hi}</span></div></div>'
    )


def score_hist(bins: list[int], color: str) -> str:
    """Judge-score distribution: 10 bins over 0..100."""
    mx = max(bins) or 1
    bars = []
    for i, c in enumerate(bins):
        h = round(100 * c / mx, 1)
        bars.append(
            f'<div class="hbar" style="--h:{h}%;--c:{color}" '
            f'title="score {i * 10}-{i * 10 + 10}: {c}">'
            f'<span class="hbn">{c}</span></div>'
        )
    return (
        f'<div class="hist score"><div class="hbars">{"".join(bars)}</div>'
        f'<div class="haxis"><span>0</span><span>score</span><span>100</span></div></div>'
    )


def stat(num: str, lbl: str) -> str:
    return f'<div class="stat"><span class="num">{esc(num)}</span><span class="lbl">{esc(lbl)}</span></div>'


def completion_card(sample: dict, color: str) -> str:
    band = "high" if sample["score"] >= 50 else "low"
    return f"""<div class="ccard {band}">
  <div class="ccard-head">
    <span class="cscore">{sample["score"]:.1f}</span>
    <span class="cidx mono">ctx {sample["context_idx"]} · rollout {sample["rollout_idx"]}</span>
  </div>
  <div class="cpersona"><span class="clbl">persona</span>{esc(sample["persona"])}</div>
  <div class="cquestion"><span class="clbl">question</span>{esc(sample["question"])}</div>
  <div class="canswer"><span class="clbl">completion</span><div class="atext clamped">{esc(sample["completion"])}</div><button class="toggle" type="button">show full</button></div>
</div>"""


def render(lmsys: dict, traits: list[dict]) -> str:
    total_contexts = sum(t["summary"]["n_contexts"] for t in traits)
    total_rollouts = sum(t["summary"]["n_total_rollouts"] for t in traits)

    # ---- masthead + statbar ----
    toc = (
        '<a href="#armA">Arm A · LMSYS</a>'
        + "".join(
            f'<a href="#armB-{t["trait"]}" style="--c:{TRAIT_META[t["trait"]]["color"]}">'
            f"Arm B · {t['trait']}</a>"
            for t in traits
        )
        + '<a href="#armC" style="--c:'
        + ARM_C_COLOR
        + '">Arm C · mixes</a>'
    )
    head = f"""<header class="masthead">
  <div class="kicker"><span class="dot"></span> Training-source ablation
    <span class="dot"></span> <span class="mono">issue 779</span>
    <span class="dot"></span> h &amp; g training corpora</div>
  <h1 class="title">The <em>training corpora</em></h1>
  <p class="dek">Every context the learned map <i>h</i> and the direct predictor <i>g</i>
     were fit on, in the #779 training-source ablation. Three arms: <b>A</b> = 5000 real
     LMSYS user prompts, <b>B</b> = a deliberately-diverse trait-eliciting behavior corpus
     (60 personas &times; 40 questions &times; 10 rollouts per trait, keeping both
     trait-high and trait-low completions), <b>C</b> = mixes of the two.</p>
  <div class="statbar">
    {stat(f"{lmsys['n_total']:,}", "Arm A · LMSYS prompts")}
    {stat(f"{total_contexts:,}", "Arm B · contexts (3 traits)")}
    {stat(f"{total_rollouts:,}", "Arm B · rollouts")}
    {stat("3", "traits")}
    {stat("60 × 40", "personas × questions / trait")}
  </div>
  <nav class="toc">{toc}</nav>
</header>"""

    # ---- Arm A ----
    lmsys_lens = [len(p) for p in lmsys["prompts"]]
    lab_rows = " ".join(
        f'<span class="labchip"><b>{t}</b> g-label std {lmsys["labels"][t]["std"]:.1f} '
        f"({lmsys['labels'][t]['n_valid']:,} valid)</span>"
        for t in TRAITS
    )
    prompt_rows = "".join(
        f'<div class="qrow"><div class="qn">{i}</div>'
        f'<div class="qtext">{esc(lmsys["prompts"][i])}</div></div>'
        for i in range(min(N_LMSYS_SHOWN, len(lmsys["prompts"])))
    )
    arm_a = f"""<section class="famsec" id="armA" style="--c:{ARM_A_COLOR}">
  <div class="famhead">
    <h2>Arm A — LMSYS real user prompts</h2>
    <span class="fct">{lmsys["n_total"]:,} prompts · source <span class="mono">{esc(lmsys["source"])}</span></span>
    <div class="fdesc">The direct-predictor <i>g</i> is fit on real user prompts (first user turn of each
      conversation, streamed first-{lmsys["n_total"]:,} in order). One on-policy rollout per prompt,
      then judged per trait for the g labels below.</div>
  </div>
  <div class="strip">
    <div class="striplbl">prompt length</div>
    {char_hist(lmsys_lens, ARM_A_COLOR)}
    <div class="labrow">{lab_rows}</div>
  </div>
  <div class="subhead mono">First {min(N_LMSYS_SHOWN, len(lmsys["prompts"]))} of {lmsys["n_total"]:,} prompts (verbatim)</div>
  <div class="qlist">{prompt_rows}</div>
</section>"""

    # ---- Arm B (per trait) ----
    arm_b_parts = []
    for t in traits:
        trait = t["trait"]
        meta = TRAIT_META[trait]
        color = meta["color"]
        warn = ""
        if meta["harmful"]:
            warn = (
                '<div class="cwarn">⚠ Content warning — this corpus deliberately elicits '
                f"<b>{esc(trait)}</b> behavior for safety-research measurement. The sample "
                "completions below contain harmful / fabricated content produced by the model "
                "under adversarial persona prompts. They are training-data artifacts, not "
                "endorsements.</div>"
            )
        persona_cards = "".join(
            f'<div class="pcard"><div class="pid mono">persona {i}</div>'
            f'<div class="ptext">{esc(p)}</div></div>'
            for i, p in enumerate(t["personas"])
        )
        question_rows = "".join(
            f'<div class="qrow"><div class="qn">{i}</div><div class="qtext">{esc(q)}</div></div>'
            for i, q in enumerate(t["questions"])
        )
        high_cards = "".join(completion_card(s, color) for s in t["high_samples"])
        low_cards = "".join(completion_card(s, color) for s in t["low_samples"])
        s = t["summary"]
        arm_b_parts.append(f"""<section class="famsec trait" id="armB-{trait}" style="--c:{color}">
  <div class="famhead">
    <h2>Arm B — {esc(trait)} corpus</h2>
    <span class="fct">{s["n_contexts"]:,} contexts · {s["n_total_rollouts"]:,} rollouts</span>
    <div class="fdesc">{esc(meta["desc"])}</div>
  </div>
  {warn}
  <div class="statbar sub">
    {stat("60", "personas")}
    {stat("40", "questions")}
    {stat(f"{s['n_contexts']:,}", "contexts (60×40)")}
    {stat(f"{s['n_total_rollouts']:,}", "rollouts (×10)")}
    {stat(f"{t['n_valid']:,}", "valid judge scores")}
    {stat(f"{t['score_mean']:.1f}", "mean score")}
    {stat(f"{t['n_high']:,} / {t['n_low']:,}", "trait-high ≥50 / low <50")}
  </div>
  <div class="strip">
    <div class="striplbl">judge-score distribution ({t["n_valid"]:,} rollouts)</div>
    {score_hist(t["score_bins"], color)}
  </div>
  <div class="subhead mono">All 60 trait-eliciting persona system prompts (verbatim)</div>
  <div class="pgrid">{persona_cards}</div>
  <div class="subhead mono">All 40 questions (verbatim)</div>
  <div class="qlist">{question_rows}</div>
  <div class="subhead mono">Sample completions — {N_COMPLETION_SAMPLES} highest-scored (distinct contexts)</div>
  <div class="cgrid">{high_cards}</div>
  <div class="subhead mono">Sample completions — {N_COMPLETION_SAMPLES} lowest-scored (distinct contexts)</div>
  <div class="cgrid">{low_cards}</div>
</section>""")
    arm_b = "".join(arm_b_parts)

    # ---- Arm C ----
    arm_c = f"""<section class="famsec" id="armC" style="--c:{ARM_C_COLOR}">
  <div class="famhead">
    <h2>Arm C — mixes of A and B</h2>
    <span class="fct">no corpus of its own</span>
    <div class="fdesc">Arm C fits <i>h</i>/<i>g</i> on <b>mixtures</b> of the Arm A (LMSYS) and Arm B
      (behavior) corpora shown above — it introduces no new training text, so it has no data section
      of its own. See the Arm A and Arm B sections for the full source material.</div>
  </div>
</section>"""

    footer = f"""<footer class="foot">
  <div><b>Data sources</b><br>
    HF data repo <span class="mono">{esc(HF_DATA_REPO)}</span><br>
    prefix <span class="mono">{esc(HF_PREFIX)}/</span><br>
    &nbsp;&nbsp;• <span class="mono">corpus_specs/{{trait}}_{{personas,questions}}.json</span> — 60 personas + 40 questions / trait<br>
    &nbsp;&nbsp;• <span class="mono">behavior_corpus/{{trait}}_rollouts.json</span> — rollout text<br>
    &nbsp;&nbsp;• <span class="mono">behavior_corpus/{{trait}}_judge_scores.json</span> — graded 0-100 scores<br>
    &nbsp;&nbsp;• <span class="mono">behavior_raw_completions/lmsys_g_rollouts_seed42.json</span> — 5000 LMSYS prompts<br>
    &nbsp;&nbsp;• <span class="mono">lmsys_g_labels/lmsys_g_labels.json</span> — Arm A g labels<br>
    local corpus tensors <span class="mono">{esc(LOCAL_CORPUS_PATH)}{{trait}}_corpus.pt</span> (cx_last/cx_mean/v_x; not rendered here)</div>
  <div><b>Source of truth</b><br>
    builder <span class="mono">{esc(BUILDER_SCRIPT)}</span><br>
    spec + judge <span class="mono">scripts/issue779_common.py</span><br>
    git <span class="mono">{esc(BUILDER_SHA)}</span></div>
  <div><b>Generated by</b><br>
    <span class="mono">uv run python scripts/issue779_dashboard_corpora.py</span><br>
    all counts computed in-script<br>
    <b>Subsetting:</b> Arm A shows first {N_LMSYS_SHOWN} of {lmsys["n_total"]:,} prompts;
    Arm B shows all 60 personas + all 40 questions / trait and
    {N_COMPLETION_SAMPLES} highest + {N_COMPLETION_SAMPLES} lowest-scored completions
    (distinct contexts) / trait; full rollout text on HF.</div>
</footer>"""

    return (
        HTML_HEAD
        + '<div class="wrap">'
        + head
        + arm_a
        + arm_b
        + arm_c
        + footer
        + "</div>"
        + HTML_TAIL
    )


HTML_HEAD = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content='width=device-width, initial-scale=1'>
<title>Issue 779 — training corpora (Arms A/B/C)</title><style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=Spline+Sans+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');
:root{
  --ink:#211e19; --ink-soft:#534c41; --paper:#f3eee4; --card:#fbf8f1;
  --line:#ddd3c1; --line-2:#cabfa8; --hl:#ffeea8; --hl-soft:#fff7df;
  --accent:#4a6a8a;
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
.dek{font-size:18.5px; max-width:74ch; color:var(--ink-soft); margin:.25em 0 0}
.dek b{color:var(--ink)}

.statbar{display:flex; flex-wrap:wrap; margin:26px 0 4px; border:1px solid var(--line-2);
  border-radius:11px; overflow:hidden; background:var(--card); box-shadow:var(--shadow)}
.statbar.sub{margin:14px 0}
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

.famsec{margin:44px 0 8px; scroll-margin-top:14px}
.famhead{display:flex; align-items:baseline; gap:14px; padding:10px 0 8px; border-bottom:2px solid var(--c);
  margin-bottom:16px; flex-wrap:wrap}
.famhead h2{font-family:"Fraunces",serif; font-weight:600; font-size:27px; margin:0; color:var(--c)}
.famhead .fct{font-family:"Spline Sans Mono",monospace; font-size:12px; color:var(--ink-soft)}
.famhead .fdesc{font-size:15px; color:var(--ink-soft); flex-basis:100%; margin-top:3px; font-style:italic; max-width:82ch}

.cwarn{background:#fbeae7; border:1.5px solid var(--c); border-left:5px solid var(--c); border-radius:9px;
  padding:12px 16px; font-size:14.5px; color:#7a2e27; margin:4px 0 16px; line-height:1.5}
.cwarn b{color:var(--c)}

.strip{background:var(--card); border:1px solid var(--line-2); border-radius:10px; box-shadow:var(--shadow);
  padding:14px 18px; margin:6px 0 18px}
.striplbl{font-family:"Spline Sans Mono",monospace; font-size:10.5px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--ink-soft); margin-bottom:10px}
.hist{margin-top:2px}
.hbars{display:flex; align-items:flex-end; gap:3px; height:74px}
.hbar{flex:1; min-width:4px; background:var(--c); opacity:.8; border-radius:3px 3px 0 0; height:var(--h);
  position:relative; transition:opacity .12s}
.hbar:hover{opacity:1}
.hist.score .hbar{min-width:14px}
.hbn{position:absolute; top:-15px; left:0; right:0; text-align:center; font-family:"Spline Sans Mono",monospace;
  font-size:8.5px; color:var(--ink-soft)}
.haxis{display:flex; justify-content:space-between; font-family:"Spline Sans Mono",monospace;
  font-size:10px; color:var(--ink-soft); margin-top:6px}
.labrow{display:flex; gap:10px; flex-wrap:wrap; margin-top:12px}
.labchip{font-family:"Spline Sans Mono",monospace; font-size:11px; color:var(--ink-soft);
  border:1px solid var(--line-2); border-radius:6px; padding:4px 9px}
.labchip b{color:var(--ink)}

.subhead{font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--ink-soft);
  margin:26px 0 12px; padding-bottom:5px; border-bottom:1px dashed var(--line)}

.pgrid{display:grid; grid-template-columns:repeat(auto-fill,minmax(310px,1fr)); gap:12px}
.pcard{background:var(--card); border:1px solid var(--line-2); border-left:4px solid var(--c);
  border-radius:9px; box-shadow:var(--shadow); padding:11px 14px}
.pcard .pid{font-size:10px; color:var(--ink-soft); letter-spacing:.06em; margin-bottom:5px}
.pcard .ptext{font-size:14.5px; line-height:1.48; white-space:pre-wrap; word-break:break-word}

.qlist{display:flex; flex-direction:column; gap:9px}
.qrow{background:var(--card); border:1px solid var(--line-2); border-radius:9px; box-shadow:var(--shadow);
  display:flex; gap:14px; padding:11px 15px; align-items:flex-start}
.qrow .qn{font-family:"Fraunces",serif; font-weight:700; font-size:18px; color:var(--c,var(--accent)); flex:none;
  width:38px; text-align:right; line-height:1.4}
.qrow .qtext{white-space:pre-wrap; word-break:break-word; font-size:15.5px; line-height:1.5; flex:1}

.cgrid{display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); gap:14px}
.ccard{background:var(--card); border:1px solid var(--line-2); border-top:4px solid var(--c);
  border-radius:10px; box-shadow:var(--shadow); padding:13px 16px; display:flex; flex-direction:column; gap:8px}
.ccard.low{border-top-color:var(--line-2)}
.ccard-head{display:flex; align-items:center; gap:12px}
.cscore{font-family:"Fraunces",serif; font-weight:700; font-size:24px; color:var(--c)}
.ccard.low .cscore{color:var(--ink-soft)}
.cidx{font-size:10.5px; color:var(--ink-soft)}
.clbl{font-family:"Spline Sans Mono",monospace; font-size:9px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--ink-soft); display:block; margin-bottom:3px}
.cpersona,.cquestion{font-size:13.5px; line-height:1.45; background:var(--paper); border:1px solid var(--line);
  border-radius:6px; padding:7px 10px; white-space:pre-wrap; word-break:break-word}
.cpersona{border-left:3px solid var(--c)}
.canswer .atext{font-size:14px; line-height:1.5; white-space:pre-wrap; word-break:break-word; color:var(--ink);
  background:var(--paper); border:1px solid var(--line); border-radius:6px; padding:8px 11px;
  max-height:8.5em; overflow:hidden; position:relative}
.canswer .atext.clamped::after{content:""; position:absolute; left:0; right:0; bottom:0; height:2em;
  background:linear-gradient(transparent,var(--paper)); pointer-events:none}
.canswer .atext.expanded{max-height:none}
.canswer .atext.expanded::after{display:none}
.toggle{font-family:"Spline Sans Mono",monospace; font-size:10.5px; color:var(--c); cursor:pointer;
  border:none; background:none; padding:4px 0 0; text-decoration:underline; text-underline-offset:2px}

footer.foot{margin-top:64px; padding-top:22px; border-top:1px solid var(--ink);
  font-family:"Spline Sans Mono",monospace; font-size:11.5px; color:var(--ink-soft);
  display:flex; gap:34px; flex-wrap:wrap; line-height:1.85}
footer.foot b{color:var(--ink); font-weight:600}
@media (max-width:760px){.cgrid,.pgrid{grid-template-columns:1fr}}
</style></head><body>"""

HTML_TAIL = """<script>
document.querySelectorAll('.canswer .toggle').forEach(function(btn){
  btn.addEventListener('click', function(){
    var t = btn.previousElementSibling;
    var open = t.classList.toggle('expanded');
    t.classList.toggle('clamped', !open);
    btn.textContent = open ? 'show less' : 'show full';
  });
});
</script></body></html>"""


def main() -> int:
    print("[issue779-dash] loading Arm A (LMSYS) ...")
    lmsys = load_lmsys()
    traits = []
    for t in TRAITS:
        print(f"[issue779-dash] loading Arm B trait={t} ...")
        traits.append(load_trait(t))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    html_doc = render(lmsys, traits)
    OUT.write_text(html_doc, encoding="utf-8")

    nbytes = OUT.stat().st_size
    print("\n===== issue779_training_corpora.html =====")
    print(f"path      : {OUT}")
    print(f"size      : {nbytes:,} bytes ({nbytes / 1e6:.2f} MB)")
    print(
        f"Arm A     : {min(N_LMSYS_SHOWN, lmsys['n_total'])} of {lmsys['n_total']:,} LMSYS prompts shown"
        f" (source {lmsys['source']})"
    )
    for t in traits:
        s = t["summary"]
        print(
            f"Arm B {t['trait']:<13}: {len(t['personas'])} personas, {len(t['questions'])} questions, "
            f"{s['n_contexts']:,} contexts, {s['n_total_rollouts']:,} rollouts; "
            f"judge n_valid={t['n_valid']:,} mean={t['score_mean']:.1f} "
            f"high(≥50)={t['n_high']:,} low(<50)={t['n_low']:,}; "
            f"samples shown={len(t['high_samples'])}+{len(t['low_samples'])}"
        )
    if nbytes > 3_000_000:
        print(f"WARNING: page exceeds 3 MB ({nbytes:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
