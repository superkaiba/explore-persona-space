#!/usr/bin/env python3
"""Reproducible generator for the issue #1310 per-character STORY-data dashboard.

Builds ONE self-contained HTML page (house `experiments/dashboards/` newsprint
style, CSS + toggle-JS imported verbatim from
`scripts/issue779_dashboard_completions.py`) showing the ACTUAL #1310 v3 PREFILL
rollouts for the four fixed-label characters — Wren (warm assistant), HELIOS
(calm AI), Dana (ordinary person), Vex (theatrical villain) — base AND instruct.

Each labeled script turn is one (context, dialogue) point of the per-character
context->dialogue map (writeup Result 4.5). Per example we distinguish:
  * CONTEXT (what v_C is read at) = the prefill PROMPT — the scene header, the
    prior `Name:` lines, and the character's own `Name:` cue.
  * COMPLETION (the Y span)       = the model's generated dialogue line after
    the cue.

Data source (HF dataset repo, revision main):
  superkaiba1/explore-persona-space-data
    issue1310_char_map/raw_completions/prefill/{base,instruct}_prefill_seed42.jsonl
  Uploaded "unconditionally" by scripts/issue1310_prefill.py (text path). Each
  JSONL record = one prefill point: {scenario_id, persona, slot, model_kind,
  prompt, completion, n_completion_tokens, setting, situation, ...}.

CONTEXT-HYGIENE: all rollout text is read from JSON and written into the HTML
string inside this script; NO rollout text is printed to stdout. The verify
pass emits only structural digests (record counts, usable counts, byte sizes).
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
HF_PREFIX = "issue1310_char_map"
HF_REVISION = "main"
HF_BLOB_BASE = (
    "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
    "blob/main/issue1310_char_map/raw_completions/prefill"
)
HF_TREE_URL = (
    "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
    "tree/main/issue1310_char_map/raw_completions/prefill"
)

MODEL_KINDS = ("base", "instruct")
MODEL_IDS = {"base": "Qwen/Qwen2.5-7B", "instruct": "Qwen/Qwen2.5-7B-Instruct"}
DIALOGUE_MIN_TOKENS = 4  # matches issue1310_common.DIALOGUE_MIN_TOKENS

# Fixed-label 4-persona panel (issue1310_common.PERSONAS), + a house-palette
# accent per character (harmonises with the cream/ink newsprint theme).
PERSONAS: dict[str, dict] = {
    "Wren": {
        "desc": "a warm, endlessly helpful assistant who patiently helps anyone who asks",
        "accent": "#2f6f6a",  # teal
    },
    "HELIOS": {
        "desc": "a calm, precise artificial intelligence",
        "accent": "#3a5a8c",  # slate blue
    },
    "Dana": {
        "desc": "an ordinary, unremarkable everyday person",
        "accent": "#6a5d3a",  # olive/brown
    },
    "Vex": {
        "desc": "a theatrical, scheming villain who delights in menace",
        "accent": "#9a3b2e",  # brick red
    },
}

N_EXAMPLES_PER_CELL = 5
PAGE_SIZE_CAP_BYTES = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Reuse the house style + toggle-JS verbatim (no reinvention).
# ---------------------------------------------------------------------------
def _load_house_style() -> tuple[str, str]:
    spec = importlib.util.spec_from_file_location(
        "issue779_dashboard_completions",
        REPO_ROOT / "scripts" / "issue779_dashboard_completions.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.STYLE, mod.TOGGLE_JS


# ---------------------------------------------------------------------------
# HF download (idempotent; two small text JSONLs only).
# ---------------------------------------------------------------------------
def download_inputs(download_dir: Path, skip_download: bool) -> Path:
    from huggingface_hub import hf_hub_download

    download_dir.mkdir(parents=True, exist_ok=True)
    for mk in MODEL_KINDS:
        rel = f"{HF_PREFIX}/raw_completions/prefill/{mk}_prefill_seed42.jsonl"
        local = download_dir / rel
        if skip_download and local.exists():
            continue
        hf_hub_download(
            HF_DATA_REPO,
            rel,
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=str(download_dir),
        )
    return download_dir


def load_records(download_dir: Path, model_kind: str) -> list[dict]:
    p = (
        download_dir
        / HF_PREFIX
        / "raw_completions"
        / "prefill"
        / f"{model_kind}_prefill_seed42.jsonl"
    )
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def usable(r: dict) -> bool:
    return int(r.get("n_completion_tokens", 0)) >= DIALOGUE_MIN_TOKENS and bool(
        (r.get("completion") or "").strip()
    )


def select_examples(records: list[dict], n: int) -> list[dict]:
    """Deterministic spread across slots (0..5) and distinct scenarios."""
    by_slot: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        if usable(r):
            by_slot[int(r["slot"])].append(r)
    for slot in by_slot:
        by_slot[slot].sort(key=lambda r: r["scenario_id"])

    chosen: list[dict] = []
    used_scen: set[str] = set()
    for slot in sorted(by_slot):
        if len(chosen) >= n:
            break
        for r in by_slot[slot]:
            if r["scenario_id"] not in used_scen:
                chosen.append(r)
                used_scen.add(r["scenario_id"])
                break
    # Backfill if some slots were empty.
    if len(chosen) < n:
        rest = sorted(
            (r for r in records if usable(r) and r["scenario_id"] not in used_scen),
            key=lambda r: (int(r["slot"]), r["scenario_id"]),
        )
        for r in rest:
            if len(chosen) >= n:
                break
            chosen.append(r)
            used_scen.add(r["scenario_id"])
    chosen.sort(key=lambda r: (int(r["slot"]), r["scenario_id"]))
    return chosen[:n]


# ---------------------------------------------------------------------------
# HTML rendering.
# ---------------------------------------------------------------------------
def esc(s: str) -> str:
    return html.escape(s if s is not None else "")


def render_card(persona: str, model_kind: str, r: dict) -> str:
    setting = r.get("setting", "")
    situation = r.get("situation", "")
    line = f"{persona}:{r.get('completion', '')}"
    tag = f"{model_kind} &middot; {esc(r['scenario_id'])} &middot; slot {int(r['slot'])}"
    return (
        '<div class="rcard">'
        '<div class="rhead">'
        f'<span class="score {"over" if model_kind == "instruct" else "under"}">'
        f"{esc(model_kind)}</span>"
        f'<span class="rid">{tag}</span>'
        "</div>"
        f'<div class="scenemeta mono">Setting: {esc(setting)} &middot; Situation: {esc(situation)} '
        f"&middot; {int(r.get('n_completion_tokens', 0))} completion tokens</div>"
        '<div class="msg"><span class="role user">Context (v<sub>C</sub> read here) — '
        "scene + prior turns, ending at the character's cue</span>"
        f'<div class="body ans clamped">{esc(r.get("prompt", ""))}</div>'
        '<button class="toggle" type="button">show full &darr;</button></div>'
        '<div class="msg"><span class="role assistant">Generated dialogue line (Y span)</span>'
        f'<div class="body">{esc(line)}</div></div>'
        "</div>"
    )


def render_model_group(persona: str, model_kind: str, examples: list[dict]) -> str:
    if not examples:
        return (
            f'<div class="grouplbl">{esc(model_kind)} arm — {esc(MODEL_IDS[model_kind])}</div>'
            '<div class="scenemeta mono">No usable prefill rollouts found for this cell.</div>'
        )
    cards = "".join(render_card(persona, model_kind, r) for r in examples)
    return (
        f'<div class="grouplbl">{esc(model_kind)} arm — {esc(MODEL_IDS[model_kind])}</div>'
        f'<div class="rgrid">{cards}</div>'
    )


def render_persona(persona: str, cell_examples: dict, cell_stats: dict) -> str:
    accent = PERSONAS[persona]["accent"]
    desc = PERSONAS[persona]["desc"]
    st_b = cell_stats[(persona, "base")]
    st_i = cell_stats[(persona, "instruct")]
    strip = (
        '<div class="statstrip">'
        f'<div class="sc"><span class="scn">{st_b["usable"]}</span>'
        '<span class="scl">base usable turns</span></div>'
        f'<div class="sc"><span class="scn">{st_b["total"]}</span>'
        '<span class="scl">base records</span></div>'
        f'<div class="sc"><span class="scn">{st_i["usable"]}</span>'
        '<span class="scl">instruct usable turns</span></div>'
        f'<div class="sc"><span class="scn">{st_i["total"]}</span>'
        '<span class="scl">instruct records</span></div>'
        "</div>"
    )
    blob_b = f"{HF_BLOB_BASE}/base_prefill_seed42.jsonl"
    blob_i = f"{HF_BLOB_BASE}/instruct_prefill_seed42.jsonl"
    return (
        f'<section class="famsec" style="--c:{accent}" id="persona_{esc(persona)}">'
        '<div class="famhead">'
        f"<h2>{esc(persona)}</h2>"
        '<span class="fct mono">fixed-label character</span>'
        f'<span class="fdesc">Generation persona: {esc(desc)}. '
        "Every scene uses this exact label at line-start (in the prompt AND as the "
        "prefill cue), so the character's turn is located by exact match.</span>"
        "</div>"
        f"{strip}"
        f"{render_model_group(persona, 'base', cell_examples[(persona, 'base')])}"
        f"{render_model_group(persona, 'instruct', cell_examples[(persona, 'instruct')])}"
        '<div class="condfoot mono">Full rollouts: '
        f'<a href="{blob_b}">prefill/base_prefill_seed42.jsonl</a> &middot; '
        f'<a href="{blob_i}">prefill/instruct_prefill_seed42.jsonl</a></div>'
        "</section>"
    )


EXTRA_STYLE = """
.scenemeta{font-family:"Spline Sans Mono",monospace; font-size:11px; color:var(--ink-soft);
  margin:2px 0 8px; letter-spacing:.02em}
.msg .role sub{font-size:.8em}
"""


def _stat(num, lbl: str) -> str:
    return (
        f"<div class='stat'><span class='num'>{num}</span><span class='lbl'>{esc(lbl)}</span></div>"
    )


def build_html(
    cell_examples: dict, cell_stats: dict, style: str, toggle_js: str
) -> tuple[str, dict]:
    n_shown = sum(len(cell_examples[k]) for k in cell_examples)
    total_records = sum(cell_stats[k]["total"] for k in cell_stats)
    total_usable = sum(cell_stats[k]["usable"] for k in cell_stats)

    parts = []
    parts.append("<!doctype html><html lang=en><head><meta charset=utf-8>")
    parts.append("<meta name=viewport content='width=device-width, initial-scale=1'>")
    parts.append("<title>Issue 1310 — per-character story scenes</title>")
    parts.append(f"<style>{style}{EXTRA_STYLE}</style></head><body><div class='wrap'>")

    parts.append(
        "<header class='masthead'>"
        "<div class='kicker'><span class='dot'></span> Issue #1310 fixed-label character map"
        " <span class='dot'></span> v3 prefill script-scenes"
        " <span class='dot'></span> Qwen-2.5-7B base &amp; instruct"
        " <span class='dot'></span> context &rarr; dialogue</div>"
        "<h1 class='title'>Per-character <em>story scenes</em></h1>"
        "<p class='dek'>The actual #1310 fixed-label script-scene data behind the weak "
        "per-character map (writeup Result 4.5). Four characters — Wren, HELIOS, Dana, Vex — "
        "each speak in labelled multi-speaker scenes; every labelled turn becomes one "
        "<b>(context, dialogue)</b> point. Context (<code>v_C</code>) is read over the scene "
        "and prior turns up to the character's own <code>Name:</code> cue; the generated line "
        "after the cue is the target <code>Y</code>. The map is weak: the writeup reports "
        "roughly base 0.11&ndash;0.15, instruct 0.19&ndash;0.25 held-out R&sup2;.</p>"
        "<div class='statbar'>"
        f"{_stat(len(PERSONAS), 'Characters')}"
        f"{_stat(2, 'Model arms')}"
        f"{_stat(n_shown, 'Scenes shown')}"
        f"{_stat(f'{total_usable:,}', 'Usable turn-pairs')}"
        f"{_stat(f'{total_records:,}', 'Prefill records')}"
        "</div></header>"
    )

    parts.append(
        "<div class='callout'><b>What you're seeing.</b> For each character we show the "
        f"<b>base</b> and <b>instruct</b> arms; per arm, a <b>subsample of {N_EXAMPLES_PER_CELL} "
        "scenes</b> chosen to spread across prefill slots (0&ndash;5) and distinct scenarios. "
        "Each card shows the <b>context</b> (the prefill prompt, ending at the character's "
        "<code>Name:</code> cue) and the model's <b>generated dialogue line</b>. The per-character "
        "stat strip counts all records / usable turns over the full uploaded JSONL. Full rollouts "
        f"are linked under each character and browsable <a href='{HF_TREE_URL}'>on the HF data "
        "repo</a>.</div>"
    )
    parts.append(
        "<div class='callout'><b>Format note.</b> The <i>base</i> prompt is raw script text "
        "(a scene-setup paragraph, then <code>Name: line</code> turns). The <i>instruct</i> prompt "
        "is the Qwen chat template (a <code>&lt;|im_start|&gt;user</code> instruction plus an "
        "assistant prefill carrying the accumulated turns) — shown verbatim, so the template "
        "tokens are visible.</div>"
    )

    for persona in PERSONAS:
        parts.append(render_persona(persona, cell_examples, cell_stats))

    parts.append(
        "<footer class='foot'>"
        f"<div><b>Data source.</b> HF dataset <span class='mono'>{HF_DATA_REPO}</span>, "
        f"prefix <span class='mono'>{HF_PREFIX}/raw_completions/prefill/</span> "
        f"(revision <span class='mono'>{HF_REVISION}</span>) — "
        "<span class='mono'>{base,instruct}_prefill_seed42.jsonl</span>, uploaded "
        "unconditionally by <span class='mono'>scripts/issue1310_prefill.py</span> "
        "(one record per prefill point: prompt = context, completion = generated line).</div>"
        "<div><b>Coverage.</b> Rollout text is complete for BOTH arms and all four characters "
        "(instruct Vex present). The downstream fit/eval is what is partial: the base-arm fit did "
        "not complete (summary.json base = null for all characters); the crashed v3 eval run's "
        "crash-persist lives at "
        "<span class='mono'>issue1310_partial/att-20260715-052017/</span>.</div>"
        "<div><b>Selection.</b> Per (character, arm): "
        f"{N_EXAMPLES_PER_CELL} usable scenes spread across slots + distinct scenarios; "
        "stat strip over all uploaded records.</div>"
        "<div><b>Generator.</b> <span class='mono'>scripts/issue1310_dashboard_stories.py</span> "
        f"&middot; generated {date.today().isoformat()}.</div>"
        "</footer>"
    )
    parts.append(f"<script>{toggle_js}</script>")
    parts.append("</div></body></html>")
    doc = "".join(parts)
    meta = {"n_shown": n_shown, "total_records": total_records, "total_usable": total_usable}
    return doc, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", default="/tmp/issue1310_dash_dl")
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "experiments" / "dashboards" / "issue1310_story_examples.html"),
    )
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--per-cell", type=int, default=N_EXAMPLES_PER_CELL)
    args = ap.parse_args()

    style, toggle_js = _load_house_style()
    dl = Path(args.download_dir)
    download_inputs(dl, args.skip_download)

    cell_examples: dict = {}
    cell_stats: dict = {}
    per_cell_report: list[tuple] = []
    for mk in MODEL_KINDS:
        recs = load_records(dl, mk)
        by_persona: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_persona[r["persona"]].append(r)
        for persona in PERSONAS:
            cell = by_persona.get(persona, [])
            u = sum(1 for r in cell if usable(r))
            cell_stats[(persona, mk)] = {"total": len(cell), "usable": u}
            cell_examples[(persona, mk)] = select_examples(cell, args.per_cell)
            per_cell_report.append((persona, mk, len(cell), u, len(cell_examples[(persona, mk)])))

    doc, meta = build_html(cell_examples, cell_stats, style, toggle_js)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")

    size = len(doc.encode("utf-8"))
    print("=" * 66)
    print(f"WROTE {out}")
    print(f"PAGE_BYTES {size}  ({size / 1024 / 1024:.2f} MB)  cap={PAGE_SIZE_CAP_BYTES}")
    print(f"SCENES_SHOWN {meta['n_shown']}")
    print(f"TOTAL_RECORDS {meta['total_records']}  TOTAL_USABLE {meta['total_usable']}")
    print("-" * 66)
    print("PER (persona, arm): records / usable / shown")
    for persona, mk, tot, u, shown in per_cell_report:
        print(f"  {persona:8s} {mk:9s}  records={tot:5d}  usable={u:5d}  shown={shown}")
    print("=" * 66)
    if size > PAGE_SIZE_CAP_BYTES:
        print(f"WARNING: page exceeds cap ({size} > {PAGE_SIZE_CAP_BYTES})")


if __name__ == "__main__":
    main()
