"""Reproducible generator for the two issue #779 raw-data rig dashboards.

Writes, matching the house dashboard style (Fraunces / Spline Sans Mono /
Newsreader, paper palette, masthead + TOC chips + cards):

  experiments/dashboards/issue779_conditions.html  ("the contexts")
  experiments/dashboards/issue779_questions.html    ("the questions")

Ground truth is READ, never retyped:
  - scripts/issue779_common.py       — EVAL_SYSTEM_PROMPTS (8/trait), EVIL_ARTIFACTS
                                        (verbatim paper illustrative artifacts),
                                        TRAIT_DESCRIPTIONS, MANY_SHOT_COUNTS,
                                        PV_WITHIN_CONDITION_TARGETS, the graded-0-100
                                        judge rubric builders, PV_ARTIFACT_GENERATION_PROMPT.
  - data/issue_779/artifacts/{sycophancy,hallucination}.json — the REALIZED
                                        generated extraction artifacts (5 pos/neg
                                        instruction pairs, 20 extraction + 20 eval
                                        questions, eval rubric) for the two
                                        generated traits (evil is verbatim in common.py).
  - scripts/issue779_collect.py       — the many-shot construction (build_many_shot_history
                                        / build_exemplar_pool / eval_context_conditions)
                                        is described + one worked shot block rendered.

Every count shown on the pages is computed from the loaded data (len()), never
hardcoded prose. Deterministic + network-free at build time.

Usage:
    uv run python scripts/issue779_dashboard_rig.py
"""

# ruff: noqa: E501 — this generator embeds a verbatim stylesheet + JS copied from
# the house dashboards (experiments/dashboards/contexts.html); line length is not
# meaningful for those data strings and reflowing them would diverge from the exemplar.

from __future__ import annotations

import html
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue779_common as C  # noqa: E402

DASHBOARD_DIR = REPO_ROOT / "experiments" / "dashboards"
GEN_DATE = date.today().isoformat()

# Per-trait accent + brief framing (colours chosen to match the house palette:
# a warm red for evil, a cool blue for sycophancy, a green for hallucination).
TRAIT_META = {
    "evil": {
        "accent": "#b5443a",
        "label": "Evil",
        "warn": True,
    },
    "sycophancy": {
        "accent": "#3a5fa0",
        "label": "Sycophancy",
        "warn": False,
    },
    "hallucination": {
        "accent": "#1f7a3d",
        "label": "Hallucination",
        "warn": False,
    },
}


def esc(s: str) -> str:
    """HTML-escape text for a text node / attribute."""
    return html.escape(s, quote=True)


def common_py_sha() -> str:
    """git short SHA of the ground-truth source scripts/issue779_common.py."""
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%h", "--", "scripts/issue779_common.py"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ── shared CSS (copied from experiments/dashboards/contexts.html, + a few adds) ──

CSS = """@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=Spline+Sans+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');
:root{
  --ink:#211e19; --ink-soft:#534c41; --paper:#f3eee4; --card:#fbf8f1;
  --line:#ddd3c1; --line-2:#cabfa8; --hl:#ffeea8; --hl-soft:#fff7df;
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

header.masthead{border-bottom:2.5px solid var(--ink); padding:48px 0 20px; position:relative}
.kicker{font-family:"Spline Sans Mono",monospace; font-size:11px; letter-spacing:.26em;
  text-transform:uppercase; color:var(--ink-soft); display:flex; gap:13px; flex-wrap:wrap; align-items:center}
.kicker .dot{width:5px;height:5px;border-radius:50%;background:var(--accent);display:inline-block}
.kicker a{color:var(--accent); text-decoration:none; border-bottom:1px dotted var(--line-2)}
h1.title{font-family:"Fraunces",Georgia,serif; font-weight:600; font-optical-sizing:auto;
  font-size:clamp(33px,5.6vw,58px); line-height:1.0; letter-spacing:-.018em; margin:.3em 0 .14em}
h1.title em{font-style:italic; color:var(--accent)}
.dek{font-size:18.5px; max-width:66ch; color:var(--ink-soft); margin:.25em 0 0}
.dek code{font-family:"Spline Sans Mono",monospace; font-size:.85em; background:var(--card);
  padding:1px 5px; border-radius:4px; border:1px solid var(--line)}

.statbar{display:flex; flex-wrap:wrap; margin:28px 0 4px; border:1px solid var(--line-2);
  border-radius:11px; overflow:hidden; background:var(--card); box-shadow:var(--shadow)}
.statbar .stat{padding:14px 22px; border-right:1px solid var(--line); flex:1; min-width:118px}
.statbar .stat:last-child{border-right:none}
.statbar .num{font-family:"Fraunces",serif; font-size:28px; font-weight:600; line-height:1; display:block}
.statbar .lbl{font-family:"Spline Sans Mono",monospace; font-size:10px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--ink-soft); margin-top:7px; display:block}

.controls{position:sticky; top:0; z-index:40; background:var(--paper); padding:15px 0 13px;
  margin:18px 0 6px; border-bottom:1px solid var(--line); display:flex; gap:12px; flex-wrap:wrap; align-items:center;
  transition:box-shadow .15s}
.controls.shadowed{box-shadow:0 12px 20px -18px rgba(33,30,25,.65)}
.search{flex:1; min-width:240px; position:relative}
.search input{width:100%; font-family:"Spline Sans Mono",monospace; font-size:13.5px;
  padding:12px 14px 12px 40px; border:1.5px solid var(--ink); border-radius:9px; background:var(--card);
  color:var(--ink); outline:none}
.search input:focus{box-shadow:0 0 0 3px var(--hl-soft)}
.search svg{position:absolute; left:13px; top:50%; transform:translateY(-50%); opacity:.55}
.count-pill{font-family:"Spline Sans Mono",monospace; font-size:12px; color:var(--ink-soft); white-space:nowrap}
.btn{font-family:"Spline Sans Mono",monospace; font-size:12px; padding:10px 13px; border:1.5px solid var(--ink);
  border-radius:8px; background:transparent; cursor:pointer; color:var(--ink); transition:background .12s,color .12s; white-space:nowrap}
.btn:hover{background:var(--ink); color:var(--paper)}

.famchips{display:flex; gap:7px; flex-wrap:wrap; width:100%}
.chip{font-family:"Spline Sans Mono",monospace; font-size:11px; letter-spacing:.02em; padding:6px 11px;
  border-radius:999px; border:1.5px solid var(--c); color:var(--c); background:transparent; cursor:pointer;
  user-select:none; transition:all .12s}
.chip[aria-pressed="true"]{background:var(--c); color:#fff8ef}
.chip .ct{opacity:.65; margin-left:5px}

mark{background:var(--hl); color:inherit; padding:0 1px; border-radius:2px}
.empty{padding:64px 0; text-align:center; color:var(--ink-soft); font-style:italic; display:none}
footer.foot{margin-top:64px; padding-top:22px; border-top:1px solid var(--line);
  font-family:"Spline Sans Mono",monospace; font-size:11.5px; color:var(--ink-soft);
  display:flex; gap:18px; flex-wrap:wrap; justify-content:space-between; line-height:1.7}
footer.foot b{color:var(--ink); font-weight:600}
footer.foot code{background:var(--card); padding:1px 5px; border-radius:4px; border:1px solid var(--line)}

/* ---- trait section ---- */
.famsec{margin:30px 0 8px}
.famhead{display:flex; align-items:baseline; gap:14px; padding:10px 0 8px; border-bottom:2px solid var(--c);
  margin-bottom:16px; flex-wrap:wrap}
.famhead h2{font-family:"Fraunces",serif; font-weight:600; font-size:25px; margin:0; color:var(--c)}
.famhead .fct{font-family:"Spline Sans Mono",monospace; font-size:12px; color:var(--ink-soft)}
.famhead .fdesc{font-size:15px; color:var(--ink-soft); flex-basis:100%; margin-top:2px; font-style:italic}

.subhead{font-family:"Spline Sans Mono",monospace; font-size:11px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-soft); margin:20px 0 10px; display:flex; gap:10px; align-items:center}
.subhead::after{content:""; flex:1; height:1px; background:var(--line)}
.subhead b{color:var(--c); font-weight:600}

.cwbanner{background:#fbeceb; border:1.5px solid var(--c); border-left:5px solid var(--c);
  border-radius:9px; padding:12px 16px; margin:6px 0 4px; font-size:14.5px; color:#7a2f28}
.cwbanner b{font-family:"Spline Sans Mono",monospace; font-size:11px; letter-spacing:.13em;
  text-transform:uppercase; display:block; margin-bottom:3px}

.cards{display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:14px}
.card{background:var(--card); border:1px solid var(--line-2); border-left:4px solid var(--c);
  border-radius:10px; box-shadow:var(--shadow); overflow:hidden; display:flex; flex-direction:column}
.card .chead{padding:13px 16px 11px; cursor:pointer; display:flex; gap:10px; align-items:flex-start;
  justify-content:space-between}
.card .chead:hover{background:#fdfbf5}
.card .cid{font-family:"Spline Sans Mono",monospace; font-size:11px; color:var(--ink-soft); letter-spacing:.02em}
.card .clabel{font-family:"Fraunces",serif; font-weight:600; font-size:17.5px; margin:2px 0 0; line-height:1.2}
.card .ckind{font-family:"Spline Sans Mono",monospace; font-size:9.5px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--c); border:1px solid var(--c); border-radius:5px; padding:2px 6px;
  white-space:nowrap; align-self:flex-start; opacity:.85}
.card .caret{transition:transform .18s; flex:none; margin-top:3px; opacity:.5}
.card.open .caret{transform:rotate(90deg)}
.card .cbody{display:none; padding:0 16px 15px; border-top:1px dashed var(--line)}
.card.open .cbody{display:block}
.msg{margin:12px 0 0}
.msg .role{font-family:"Spline Sans Mono",monospace; font-size:9.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-soft); display:inline-block; margin-bottom:4px;
  border-bottom:1px solid var(--line); padding-bottom:1px}
.msg .role.system{color:var(--c)}
.msg .role.user{color:#8a5a1a}
.msg .role.assistant{color:#2f6f6a}
.msg .role.pos{color:#b5443a}
.msg .role.neg{color:#1f7a3d}
.msg .body{white-space:pre-wrap; word-break:break-word; font-size:15px; line-height:1.5;
  background:var(--paper); border:1px solid var(--line); border-radius:7px; padding:9px 11px}
.msg .body.note{font-style:italic; color:var(--ink-soft); background:transparent; border:1px dashed var(--line)}
.metarow{font-family:"Spline Sans Mono",monospace; font-size:10.5px; color:var(--ink-soft);
  margin-top:12px; padding-top:9px; border-top:1px dashed var(--line); display:flex; gap:14px; flex-wrap:wrap}

/* ---- questions list ---- */
.qlist{margin-top:12px; display:flex; flex-direction:column; gap:10px}
.qrow{background:var(--card); border:1px solid var(--line-2); border-left:4px solid var(--c);
  border-radius:9px; box-shadow:var(--shadow); display:flex; gap:14px; padding:13px 16px; align-items:flex-start}
.qrow .qn{font-family:"Fraunces",serif; font-weight:700; font-size:20px; color:var(--c); flex:none;
  width:34px; text-align:right; line-height:1.35}
.qrow .qtext{white-space:pre-wrap; word-break:break-word; font-size:16px; line-height:1.5; flex:1}
.qrow .qmeta{font-family:"Spline Sans Mono",monospace; font-size:10px; color:var(--ink-soft); flex:none;
  text-align:right; padding-top:4px; min-width:62px}
"""


def masthead(
    kicker_bits: list[str], title_html: str, dek_html: str, stats: list[tuple[str, str]]
) -> str:
    """Build the shared masthead + statbar block."""
    kick = "\n     ".join(f'<span class="dot"></span> {b}' for b in kicker_bits)
    statcells = "".join(
        f'<div class="stat"><span class="num">{esc(num)}</span>'
        f'<span class="lbl">{esc(lbl)}</span></div>'
        for num, lbl in stats
    )
    return f"""<header class="masthead">
  <div class="kicker">{kick}</div>
  <h1 class="title">{title_html}</h1>
  <p class="dek">{dek_html}</p>
  <div class="statbar">{statcells}</div>
</header>"""


def footer(source_line: str, gen_cmd: str) -> str:
    return f"""<footer class="foot">
  <div>{source_line}</div>
  <div>Generated {GEN_DATE} &middot; <code>{esc(gen_cmd)}</code> &middot; self-contained, no external data at load</div>
</footer>"""


def trait_chips() -> str:
    chips = "".join(
        f'<button class="chip" data-fam="{t}" aria-pressed="true" style="--c:{TRAIT_META[t]["accent"]}">'
        f'{TRAIT_META[t]["label"]}<span class="ct" data-ct="{t}"></span></button>'
        for t in C.TRAITS
    )
    return (
        '<div class="controls" style="position:static; border:none; padding:0; margin:0 0 4px">'
        f'<div class="famchips">{chips}</div></div>'
    )


def msg_html(role_label: str, role_class: str, content: str, note: bool = False) -> str:
    body_cls = "body note" if note else "body"
    return (
        f'<div class="msg"><span class="role {role_class}">{esc(role_label)}</span>'
        f'<div class="{body_cls}">{esc(content)}</div></div>'
    )


def card_html(
    trait: str, cid: str, label: str, kind: str, msgs_html: str, search_text: str, meta: str = ""
) -> str:
    accent = TRAIT_META[trait]["accent"]
    metarow = f'<div class="metarow">{meta}</div>' if meta else ""
    return f"""<div class="card" data-fam="{trait}" data-search="{esc(search_text.lower())}" style="--c:{accent}">
    <div class="chead">
      <div><div class="cid">{esc(cid)}</div><div class="clabel">{esc(label)}</div></div>
      <div style="display:flex;gap:8px;align-items:flex-start">
        <span class="ckind">{esc(kind)}</span>
        <svg class="caret" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M9 6l6 6-6 6"/></svg>
      </div>
    </div>
    <div class="cbody">{msgs_html}{metarow}</div></div>"""


# ── page 1: conditions ────────────────────────────────────────────────────────


def build_conditions_page() -> tuple[str, dict[str, int]]:
    """The 3 traits x (8 system + 5 many-shot) eval conditions + 5 r_B pairs each."""
    counts = {"traits": len(C.TRAITS), "system": 0, "manyshot": 0, "pairs": 0, "conditions": 0}
    sections = []

    for trait in C.TRAITS:
        meta = TRAIT_META[trait]
        accent = meta["accent"]
        artifacts = C.load_extraction_artifacts(trait)
        sys_prompts = C.EVAL_SYSTEM_PROMPTS[trait]
        pairs = artifacts["instruction"]
        extraction_q = artifacts["extraction_questions"]
        eval_q = artifacts["eval_questions"]

        n_sys = len(sys_prompts)
        n_shot = len(C.MANY_SHOT_COUNTS)
        n_pairs = len(pairs)
        counts["system"] += n_sys
        counts["manyshot"] += n_shot
        counts["pairs"] += n_pairs
        counts["conditions"] += n_sys + n_shot

        tgt = C.PV_WITHIN_CONDITION_TARGETS[trait]
        desc = (
            f"&ldquo;{esc(C.TRAIT_DESCRIPTIONS[trait])}&rdquo; "
            f"&mdash; {n_sys} system-prompt + {n_shot} many-shot eval conditions, "
            f"plus {n_pairs} contrastive pos/neg pairs for the <i>r<sub>B</sub></i> direction. "
            f"PV within-condition Pearson targets: system {tgt['system']:.3f}, "
            f"many-shot {tgt['many_shot']:.3f} (&plusmn;{C.RIG_VALIDATION_BAND:.2f} band)."
        )

        parts = [
            f'<section class="famsec" data-fam="{trait}" style="--c:{accent}">',
            f'<div class="famhead"><h2>{esc(meta["label"])}</h2>'
            f'<span class="fct">{n_sys + n_shot} conditions &middot; {n_pairs} extraction pairs</span>'
            f'<span class="fdesc">{desc}</span></div>',
        ]

        if meta["warn"]:
            parts.append(
                '<div class="cwbanner"><b>Content warning</b>'
                "The evil-trait system prompts, instructions and exemplars below explicitly "
                "solicit harmful/malicious text. They are shown verbatim (already public on the "
                "HF data repo and in arXiv 2507.21509) so the rig is fully auditable; the intent "
                "is measurement of trait-expression geometry, not endorsement.</div>"
            )

        # ---- system-prompt conditions ----
        parts.append(
            f'<div class="subhead"><b>System-prompt conditions</b> &mdash; {n_sys} verbatim, strong &rarr; weak</div>'
        )
        parts.append('<div class="cards">')
        for i, sp in enumerate(sys_prompts):
            if i == 0:
                intensity = "strongest induction"
            elif i == n_sys - 1:
                intensity = "plain helpful assistant (weakest)"
            else:
                intensity = f"induction level {i + 1} of {n_sys}"
            msgs = msg_html("system", "system", sp)
            parts.append(
                card_html(
                    trait,
                    f"sys{i}",
                    f"System prompt {i + 1} / {n_sys}",
                    f"system &middot; {intensity}",
                    msgs,
                    f"sys{i} {intensity} {sp}",
                    meta="<span>mode: system</span><span>n_shot: 0</span>",
                )
            )
        parts.append("</div>")

        # ---- many-shot conditions ----
        parts.append(
            f'<div class="subhead"><b>Many-shot conditions</b> &mdash; {n_shot} shot counts '
            f"{{{', '.join(str(k) for k in C.MANY_SHOT_COUNTS)}}}, no system prompt</div>"
        )
        parts.append('<div class="cards">')
        construction = (
            "Many-shot monitoring (arXiv 2507.21509): NO system prompt. The prompt is a "
            "multi-turn chat history of k trait-exhibiting (user, assistant) exemplar pairs, "
            "then the eval question as the final user turn. Exemplars are built by "
            "build_exemplar_pool (issue779_collect.py): the model generates on-policy under "
            "the STRONGEST system prompt sys0 over the first k extraction-set questions "
            "(temperature 1.0, seed 7, max 256 tokens), so every exemplar exhibits the target "
            "trait. Shot count k selects how many exemplar pairs precede the eval question."
        )
        for k in C.MANY_SHOT_COUNTS:
            if k == 0:
                note = (
                    "0-shot: no exemplars and no system prompt — the eval question alone "
                    "under the bare Qwen chat template. This is the many-shot baseline "
                    "(reads the model's default trait expression with no induction)."
                )
            else:
                note = (
                    f"{k}-shot: {k} trait-exhibiting (user, assistant) exemplar pairs "
                    f"(from extraction questions 1&ndash;{k} answered on-policy under sys0), "
                    f"then the eval question. See the worked example card for the full "
                    f"multi-turn structure."
                )
            msgs = msg_html("construction", "user", construction, note=True) + msg_html(
                f"condition shot{k}", "assistant", note, note=True
            )
            parts.append(
                card_html(
                    trait,
                    f"shot{k}",
                    f"k = {k} exemplars",
                    f"many-shot &middot; k={k}",
                    msgs,
                    f"shot{k} many-shot {k} exemplars {construction}",
                    meta=f"<span>mode: many_shot</span><span>n_shot: {k}</span>",
                )
            )
        parts.append("</div>")

        # ---- worked shot example (shot5 structure) ----
        worked_k = 5
        parts.append(
            f'<div class="subhead"><b>Worked shot block</b> &mdash; the k={worked_k} prompt, verbatim structure</div>'
        )
        wm = []
        for j in range(worked_k):
            wm.append(msg_html(f"user · exemplar {j + 1}", "user", extraction_q[j]))
            wm.append(
                msg_html(
                    f"assistant · exemplar {j + 1}",
                    "assistant",
                    "⟨on-policy greedy generation under system prompt sys0 (the strongest "
                    "induction), temperature 1.0 / seed 7 / max 256 tokens. Reduced to r_B at "
                    "collection time; the exemplar response text was not persisted, so the real "
                    "user turns are shown verbatim and the assistant turns are described here.⟩",
                    note=True,
                )
            )
        wm.append(msg_html("user · eval question", "user", eval_q[0]))
        worked_search = (
            f"worked shot block k={worked_k} " + " ".join(extraction_q[:worked_k]) + " " + eval_q[0]
        )
        parts.append('<div class="cards">')
        parts.append(
            card_html(
                trait,
                f"shot{worked_k} (worked)",
                f"k = {worked_k} — full prompt",
                "worked example",
                "".join(wm),
                worked_search,
                meta=f"<span>mode: many_shot</span><span>n_shot: {worked_k}</span>"
                f"<span>exemplar Q: extraction 1&ndash;{worked_k}</span>",
            )
        )
        parts.append("</div>")

        # ---- r_B extraction contrastive pairs ----
        parts.append(
            f'<div class="subhead"><b>r<sub>B</sub> extraction pairs</b> &mdash; {n_pairs} contrastive '
            f"pos/neg system-prompt pairs</div>"
        )
        parts.append('<div class="cards">')
        for i, pair in enumerate(pairs):
            msgs = msg_html("pos (elicit trait)", "pos", pair["pos"]) + msg_html(
                "neg (avoid trait)", "neg", pair["neg"]
            )
            parts.append(
                card_html(
                    trait,
                    f"pair{i + 1}",
                    f"Contrastive pair {i + 1} / {n_pairs}",
                    "r_B pos/neg",
                    msgs,
                    f"pair {i + 1} {pair['pos']} {pair['neg']}",
                    meta="<span>role: extraction direction</span>",
                )
            )
        parts.append("</div>")

        parts.append("</section>")
        sections.append("\n".join(parts))

    total_cond = counts["conditions"]
    stats = [
        (str(counts["traits"]), "Traits"),
        (str(total_cond), "Eval conditions"),
        (f"{counts['system'] // counts['traits']}", "System / trait"),
        (f"{counts['manyshot'] // counts['traits']}", "Many-shot / trait"),
        (f"{counts['pairs'] // counts['traits']}", "r_B pairs / trait"),
        ("Qwen2.5", "7B-Instruct"),
    ]
    head = masthead(
        [
            'Persona-vectors monitoring rig <span class="mono">issue 779</span>',
            '<a href="issue779_questions.html">questions &rarr;</a>',
        ],
        "The <em>conditions</em>",
        "Every eval CONTEXT the model is placed in for the issue #779 persona-vectors monitoring "
        "experiment (Qwen-2.5-7B-Instruct), across three traits &mdash; <i>evil</i>, "
        "<i>sycophancy</i>, <i>hallucination</i>. Per trait: the "
        f"{counts['system'] // counts['traits']} system-prompt conditions verbatim, the "
        f"{counts['manyshot'] // counts['traits']} many-shot conditions with a worked block, and the "
        f"{counts['pairs'] // counts['traits']} contrastive pos/neg pairs that build the "
        "<i>r<sub>B</sub></i> persona direction.",
        stats,
    )
    sha = common_py_sha()
    src = (
        f"<b>Source of truth.</b> <code>scripts/issue779_common.py</code> @ "
        f"<code>{sha}</code> (verbatim: EVAL_SYSTEM_PROMPTS, EVIL_ARTIFACTS, "
        f"TRAIT_DESCRIPTIONS, MANY_SHOT_COUNTS). The <b>realized</b> sycophancy &amp; "
        f"hallucination pos/neg pairs come from the generated artifacts "
        f"<code>data/issue_779/artifacts/{{sycophancy,hallucination}}.json</code> "
        f"(Sonnet-generated via PV_ARTIFACT_GENERATION_PROMPT; not git-tracked, local build inputs). "
        f"Evil is the paper&rsquo;s verbatim illustrative artifacts pinned in common.py. Many-shot "
        f"construction: <code>scripts/issue779_collect.py</code> (build_many_shot_history / "
        f"build_exemplar_pool). arXiv 2507.21509 (Persona Vectors)."
    )
    body = "\n".join(
        [
            head,
            trait_chips(),
            '<div class="controls">'
            '<label class="search"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="2.2"><circle cx="11" cy="11" r="7"/>'
            '<path d="M21 21l-4.3-4.3"/></svg>'
            '<input id="q" type="search" placeholder="Search system prompts, exemplars, pairs..." autocomplete="off"></label>'
            '<button class="btn" id="expandAll">Expand all</button>'
            '<button class="btn" id="collapseAll">Collapse all</button>'
            '<span class="count-pill" id="count"></span></div>',
            '<div id="sections">',
            *sections,
            "</div>",
            '<div class="empty" id="empty">No conditions match that search.</div>',
            footer(src, "uv run python scripts/issue779_dashboard_rig.py"),
        ]
    )
    return page_shell("Issue 779 · eval conditions", body, card_js=True), counts


# ── page 2: questions ─────────────────────────────────────────────────────────


def build_questions_page() -> tuple[str, dict[str, int]]:
    """Per trait: 20 extraction + 20 eval questions verbatim + the graded judge rubric."""
    counts = {"traits": len(C.TRAITS), "extraction": 0, "eval": 0}
    sections = []

    for trait in C.TRAITS:
        meta = TRAIT_META[trait]
        accent = meta["accent"]
        artifacts = C.load_extraction_artifacts(trait)
        extraction_q = artifacts["extraction_questions"]
        eval_q = artifacts["eval_questions"]
        n_ex = len(extraction_q)
        n_ev = len(eval_q)
        counts["extraction"] += n_ex
        counts["eval"] += n_ev

        raw_rubric = artifacts["eval_prompt"]
        judge_system = C.trait_judge_system_prompt(trait)
        judge_user = C.trait_judge_user_msg("{question}", "{answer}")

        parts = [
            f'<section class="famsec" data-fam="{trait}" style="--c:{accent}">',
            f'<div class="famhead"><h2>{esc(meta["label"])}</h2>'
            f'<span class="fct">{n_ex} extraction &middot; {n_ev} eval &middot; 1 judge rubric</span>'
            f'<span class="fdesc">The persona-vectors extraction/eval split: '
            f"{n_ex} questions build the <i>r<sub>B</sub></i> direction, a disjoint {n_ev} score "
            f"trait expression. Neither set explicitly asks for the trait; they probe whether it "
            f"surfaces. Judge: {esc(C.JUDGE_MODEL)}, graded 0&ndash;100, N={C.JUDGE_N_DRAWS} draws, "
            f"DROP-NEVER-COERCE.</span></div>",
        ]

        if meta["warn"]:
            parts.append(
                '<div class="cwbanner"><b>Content warning</b>'
                "The evil-trait questions and judge rubric below are shown verbatim (public in "
                "arXiv 2507.21509 and on the HF data repo) so the rig is auditable.</div>"
            )

        # extraction questions
        parts.append(
            f'<div class="subhead"><b>Extraction questions</b> &mdash; {n_ex}, build r<sub>B</sub></div>'
        )
        parts.append(f'<div class="qlist" data-fam="{trait}">')
        for i, q in enumerate(extraction_q):
            parts.append(qrow_html(trait, i + 1, q, "extraction"))
        parts.append("</div>")

        # eval questions
        parts.append(
            f'<div class="subhead"><b>Eval questions</b> &mdash; {n_ev}, disjoint, judge-scored</div>'
        )
        parts.append(f'<div class="qlist" data-fam="{trait}">')
        for i, q in enumerate(eval_q):
            parts.append(qrow_html(trait, i + 1, q, "eval"))
        parts.append("</div>")

        # judge rubric
        parts.append('<div class="subhead"><b>Judge rubric</b> &mdash; graded 0&ndash;100</div>')
        parts.append('<div class="cards">')
        rubric_msgs = (
            msg_html("eval_prompt (paper rubric template)", "system", raw_rubric)
            + msg_html("assembled judge SYSTEM message", "system", judge_system)
            + msg_html("judge USER message format", "user", judge_user)
        )
        parts.append(
            card_html(
                trait,
                "judge",
                f"{esc(meta['label'])} trait judge",
                "graded 0-100",
                rubric_msgs,
                f"judge rubric {raw_rubric} {judge_system}",
                meta=f"<span>model: {esc(C.JUDGE_MODEL)}</span>"
                f"<span>N: {C.JUDGE_N_DRAWS} draws</span>"
                f"<span>temp: {C.JUDGE_TEMPERATURE}</span>",
            )
        )
        parts.append("</div>")

        parts.append("</section>")
        sections.append("\n".join(parts))

    stats = [
        (str(counts["traits"]), "Traits"),
        (f"{counts['extraction'] // counts['traits']}", "Extraction / trait"),
        (f"{counts['eval'] // counts['traits']}", "Eval / trait"),
        (str(counts["extraction"] + counts["eval"]), "Total questions"),
        ("Qwen2.5", "7B-Instruct"),
    ]
    head = masthead(
        [
            'Persona-vectors monitoring rig <span class="mono">issue 779</span>',
            '<a href="issue779_conditions.html">conditions &rarr;</a>',
        ],
        "The <em>questions</em>",
        "Every probe question in the issue #779 persona-vectors monitoring rig, per trait: the "
        f"{counts['extraction'] // counts['traits']} extraction questions that build the "
        "<i>r<sub>B</sub></i> direction and the disjoint "
        f"{counts['eval'] // counts['traits']} eval questions that are judge-scored for trait "
        "expression, plus the verbatim graded-0&ndash;100 judge rubric.",
        stats,
    )
    sha = common_py_sha()
    src = (
        f"<b>Source of truth.</b> Evil questions + rubric: <code>scripts/issue779_common.py</code> @ "
        f"<code>{sha}</code> (EVIL_ARTIFACTS, verbatim from arXiv 2507.21509). "
        f"<b>Realized</b> sycophancy &amp; hallucination question lists + rubrics: the generated "
        f"artifacts <code>data/issue_779/artifacts/{{sycophancy,hallucination}}.json</code> "
        f"(Sonnet-generated via PV_ARTIFACT_GENERATION_PROMPT; local build inputs, not git-tracked). "
        f"Assembled judge messages via <code>trait_judge_system_prompt</code> / "
        f"<code>trait_judge_user_msg</code> in common.py."
    )
    body = "\n".join(
        [
            head,
            trait_chips(),
            '<div class="controls">'
            '<label class="search"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="2.2"><circle cx="11" cy="11" r="7"/>'
            '<path d="M21 21l-4.3-4.3"/></svg>'
            '<input id="q" type="search" placeholder="Search extraction / eval questions, rubric..." autocomplete="off"></label>'
            '<span class="count-pill" id="count"></span></div>',
            '<div id="sections">',
            *sections,
            "</div>",
            '<div class="empty" id="empty">No questions match that search.</div>',
            footer(src, "uv run python scripts/issue779_dashboard_rig.py"),
        ]
    )
    return page_shell("Issue 779 · probe questions", body, card_js=True), counts


def qrow_html(trait: str, n: int, text: str, kind: str) -> str:
    accent = TRAIT_META[trait]["accent"]
    words = len(text.split())
    chars = len(text)
    return (
        f'<div class="qrow" data-fam="{trait}" data-kind="{kind}" '
        f'data-search="{esc(text.lower())}" style="--c:{accent}">'
        f'<div class="qn">{n}</div>'
        f'<div class="qtext">{esc(text)}</div>'
        f'<div class="qmeta">{words}w<br>{chars}ch</div></div>'
    )


# ── page shell + shared JS ────────────────────────────────────────────────────


def _js() -> str:
    """Search + highlight + trait-chip filter over the pre-rendered DOM."""
    return r"""
function escRe(s){return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}
function highlight(el, q){
  el.querySelectorAll('mark').forEach(m=>{const t=document.createTextNode(m.textContent); m.replaceWith(t)});
  el.normalize();
  if(!q) return;
  const re=new RegExp(escRe(q),'gi');
  const walker=document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
  const nodes=[]; while(walker.nextNode()) nodes.push(walker.currentNode);
  for(const n of nodes){
    const txt=n.nodeValue; if(!re.test(txt)) continue; re.lastIndex=0;
    const frag=document.createDocumentFragment(); let last=0, m;
    while((m=re.exec(txt))){
      if(m.index>last) frag.appendChild(document.createTextNode(txt.slice(last,m.index)));
      const mk=document.createElement('mark'); mk.textContent=m[0]; frag.appendChild(mk);
      last=m.index+m[0].length; if(m[0].length===0) re.lastIndex++;
    }
    if(last<txt.length) frag.appendChild(document.createTextNode(txt.slice(last)));
    n.replaceWith(frag);
  }
}
function stickyShadow(){
  const c=document.querySelector('.controls:not([style*="static"])'); if(!c) return;
  const s=document.createElement('div'); c.parentNode.insertBefore(s,c);
  new IntersectionObserver(([e])=>c.classList.toggle('shadowed',!e.isIntersecting),{threshold:1}).observe(s);
}
const ITEMS='.card, .qrow';
const TOTAL=document.querySelectorAll(ITEMS).length;
const NOUN=document.querySelector('.qrow')?'items':'conditions';
function activeFams(){
  return new Set([...document.querySelectorAll('.chip[aria-pressed=true]')].map(c=>c.dataset.fam));
}
function chipCounts(){
  document.querySelectorAll('.chip .ct').forEach(ct=>{
    const fam=ct.dataset.ct;
    ct.textContent=document.querySelectorAll(`.card[data-fam="${fam}"], .qrow[data-fam="${fam}"]`).length;
  });
}
function applyFilter(){
  const q=document.getElementById('q').value.trim().toLowerCase();
  const fams=activeFams(); let shown=0;
  document.querySelectorAll('.famsec').forEach(sec=>{
    const famOn=fams.has(sec.dataset.fam); let secShown=0;
    sec.querySelectorAll(ITEMS).forEach(it=>{
      const ok = famOn && (!q || (it.dataset.search||'').includes(q));
      it.style.display = ok?'':'none';
      if(ok){ secShown++; shown++;
        const target=it.querySelector('.qtext')||it;
        if(q){ it.classList.add('open'); highlight(target,q); } else { highlight(target,''); }
      }
    });
    // hide empty subheads/qlists inside a section by hiding the whole section only if nothing shows
    sec.style.display = secShown?'':'none';
  });
  document.getElementById('count').textContent = shown+' / '+TOTAL+' '+NOUN;
  document.getElementById('empty').style.display = shown?'none':'block';
}
document.querySelectorAll('.chead').forEach(h=>h.addEventListener('click',()=>h.parentElement.classList.toggle('open')));
chipCounts(); stickyShadow();
document.getElementById('q').addEventListener('input', applyFilter);
const ea=document.getElementById('expandAll');
if(ea) ea.addEventListener('click',()=>document.querySelectorAll('.card:not([style*="display: none"])').forEach(c=>c.classList.add('open')));
const ca=document.getElementById('collapseAll');
if(ca) ca.addEventListener('click',()=>document.querySelectorAll('.card').forEach(c=>c.classList.remove('open')));
document.querySelectorAll('.chip').forEach(ch=>ch.addEventListener('click',()=>{
  ch.setAttribute('aria-pressed', ch.getAttribute('aria-pressed')==='true'?'false':'true'); applyFilter();
}));
applyFilter();
"""


def page_shell(title: str, body: str, card_js: bool = True) -> str:
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width, initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{CSS}</style></head>"
        "<body style='--accent:#b5443a'>\n"
        '<div class="wrap">\n'
        f"{body}\n"
        "</div>\n"
        f"<script>{_js()}</script>\n"
        "</body></html>"
    )


def main() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    cond_html, cond_counts = build_conditions_page()
    ques_html, ques_counts = build_questions_page()

    cond_path = DASHBOARD_DIR / "issue779_conditions.html"
    ques_path = DASHBOARD_DIR / "issue779_questions.html"
    cond_path.write_text(cond_html)
    ques_path.write_text(ques_html)

    print("Wrote:")
    for p in (cond_path, ques_path):
        size = p.stat().st_size
        print(f"  {p}  ({size:,} bytes, {size / 1_048_576:.2f} MB)")

    print("\nConditions page per-trait counts:")
    print(f"  traits={cond_counts['traits']}  total eval conditions={cond_counts['conditions']}")
    for trait in C.TRAITS:
        arts = C.load_extraction_artifacts(trait)
        print(
            f"  {trait:14s} system={len(C.EVAL_SYSTEM_PROMPTS[trait])}  "
            f"many_shot={len(C.MANY_SHOT_COUNTS)}  r_B_pairs={len(arts['instruction'])}"
        )
    print("\nQuestions page per-trait counts:")
    print(
        f"  traits={ques_counts['traits']}  total questions={ques_counts['extraction'] + ques_counts['eval']}"
    )
    for trait in C.TRAITS:
        arts = C.load_extraction_artifacts(trait)
        print(
            f"  {trait:14s} extraction={len(arts['extraction_questions'])}  "
            f"eval={len(arts['eval_questions'])}  rubric_chars={len(arts['eval_prompt'])}"
        )


if __name__ == "__main__":
    main()
