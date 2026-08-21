"""Issue #2223 — first-k answer-token cap: dose-response analysis + qualitative dashboard.

Reads the NAP-subdir judged harm scores (temp 0.7, 3-seed, decode-matched) for the
band-layer arms and answers: is the every-token assistant-axis cap's harm reduction
FRONT-LOADED to the answer's opening tokens? The dose ladder is
  unsteered (k=0) -> cap_ansfirst{1,2,4,8} -> cap_alltoken (every token, k=inf),
all at band layers, so the only variable is how many opening answer tokens are capped.

Outputs (figures/issue_2223/casestudy_replay/qwen3-32b/):
  - ansfirst_doseresponse.png       harm vs k, both behaviours, with unsteered + every-token references
  - ansfirst_qual_dashboard.html    per-turn qualitative comparison (delusion) across the dose ladder
Prints a summary table. Harm = 3-seed-pooled mean of the Sonnet-4.5 0-100 judge; coherence reported.
No harmful text is paged to stdout — generations flow only into the HTML.
"""

from __future__ import annotations

import base64
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAP = ROOT / "eval_results/issue_2223/casestudy_replay/qwen3-32b/native_axis_fidelity_preimage"
JUDGED = NAP / "judged"
OUT = ROOT / "figures/issue_2223/casestudy_replay/qwen3-32b"
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["delusion", "selfharm"]
# dose ladder (band layers): label -> (base arm stem, k for x-axis, colour)
LADDER = [
    ("unsteered", "na__unsteered", 0, "#6b7280"),
    ("ansfirst1", "band__cap_ansfirst1", 1, "#fca5a5"),
    ("ansfirst2", "band__cap_ansfirst2", 2, "#f87171"),
    ("ansfirst4", "band__cap_ansfirst4", 4, "#ef4444"),
    ("ansfirst8", "band__cap_ansfirst8", 8, "#b91c1c"),
    ("every-token", "band__cap_alltoken", 16, "#2563eb"),  # k=inf, plotted at x=16 (log-ish)
]
SEED_SUFFIXES = ["", "__seed43", "__seed44"]


def _cells(kind: str, scenario: str) -> dict:
    f = JUDGED / f"{kind}_{scenario}.json"
    return json.loads(f.read_text())["cells"] if f.exists() else {}


def pooled(kind: str, scenario: str, stem: str):
    """3-seed-pooled per-turn scores + overall mean for one arm; None if absent."""
    cells = _cells(kind, scenario)
    per_turn: dict[int, list[float]] = {}
    allv: list[float] = []
    present = False
    for suf in SEED_SUFFIXES:
        cell = cells.get(stem + suf)
        if not isinstance(cell, dict):
            continue
        present = True
        for tk, tv in cell.items():
            if isinstance(tv, dict) and isinstance(tv.get("score"), (int, float)):
                per_turn.setdefault(int(tk), []).append(float(tv["score"]))
                allv.append(float(tv["score"]))
    if not present or not allv:
        return None
    turn_mean = {t: statistics.mean(v) for t, v in per_turn.items()}
    return {"turn_mean": turn_mean, "mean": statistics.mean(allv), "n": len(allv)}


def make_doseresponse() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140)
    for ax, sc in zip(axes, SCENARIOS):
        xs, ys, base = [], [], None
        every = None
        for label, stem, k, _c in LADDER:
            r = pooled("scores", sc, stem)
            if r is None:
                continue
            if label == "unsteered":
                base = r["mean"]
            elif label == "every-token":
                every = r["mean"]
            else:
                xs.append(k)
                ys.append(r["mean"])
        # ansfirst dose curve
        if xs:
            ax.plot(
                xs, ys, marker="o", ms=6, lw=2, color="#ef4444", label="cap first-k answer tokens"
            )
        if base is not None:
            ax.axhline(base, ls="--", lw=1.5, color="#6b7280", label="unsteered (no cap)")
        if every is not None:
            ax.axhline(every, ls="--", lw=1.5, color="#2563eb", label="every-token cap")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_xlabel("k = number of opening answer tokens capped")
        ax.set_ylabel("Judged harm (0-100, 3-seed mean)")
        ax.set_ylim(-3, 103)
        ax.set_title(sc)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Is the every-token cap's benefit front-loaded? (Qwen3-32B, band layers)", y=1.02)
    fig.tight_layout()
    p = OUT / "ansfirst_doseresponse.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def summary_table() -> str:
    lines = []
    for sc in SCENARIOS:
        lines.append(f"\n### {sc}")
        for label, stem, _k, _c in LADDER:
            r = pooled("scores", sc, stem)
            co = pooled("coherence", sc, stem)
            if r is None:
                lines.append(f"  {label:12s} (absent)")
                continue
            coh = f"{co['mean']:.1f}" if co else "n/a"
            lines.append(f"  {label:12s} harm={r['mean']:5.1f}  coherence={coh}  (n={r['n']})")
    return "\n".join(lines)


def build_dashboard(png: Path) -> Path:
    # qualitative: delusion (the scenario with the ctx-end effect), seed-42 representative draw
    sc = "delusion"
    cols = [(lbl, stem, c) for lbl, stem, _k, c in LADDER]
    gens = {}
    for _lbl, stem, _c in cols:
        f = NAP / sc / f"{stem}.json"
        gens[stem] = json.loads(f.read_text()) if f.exists() else None
    harm = {stem: pooled("scores", sc, stem) for _lbl, stem, _c in cols}
    base_arm = next((g for g in gens.values() if g), None)
    if base_arm is None:
        return OUT / "ansfirst_qual_dashboard.html"
    base_turns = base_arm["turns"]
    b64 = base64.b64encode(png.read_bytes()).decode()

    def col_style(c):
        return f"border-top:4px solid {c}"

    def hz(sc_val):
        if sc_val is None:
            return "#f3f4f6"
        if sc_val < 25:
            return "#dcfce7"
        if sc_val < 50:
            return "#fef9c3"
        if sc_val < 75:
            return "#fed7aa"
        return "#fecaca"

    ncol = len(cols)
    css = f"""
    body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f9fafb;color:#111827;font-size:13px}}
    .wrap{{max-width:1700px;margin:0 auto;padding:16px 20px}}
    h1{{font-size:19px;margin:0 0 4px}} .sub{{color:#6b7280;margin-bottom:14px}}
    .fig{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin-bottom:18px;text-align:center}}
    .fig img{{max-width:100%}}
    .hdr{{display:grid;grid-template-columns:repeat({ncol},1fr);gap:8px;position:sticky;top:0;background:#f9fafb;padding:8px 0;z-index:5}}
    .armcard{{border-radius:8px;padding:8px 10px;background:#fff;border:1px solid #e5e7eb}}
    .armcard .m{{font-size:20px;font-weight:700}} .armcard .l{{font-size:11px;color:#374151}}
    .turn{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;margin-bottom:12px;overflow:hidden}}
    .user{{background:#eff6ff;border-bottom:1px solid #dbeafe;padding:8px 10px}} .user b{{color:#1d4ed8}}
    .cols{{display:grid;grid-template-columns:repeat({ncol},1fr)}}
    .cell{{padding:8px 10px;border-right:1px solid #f3f4f6;white-space:pre-wrap;line-height:1.4;font-size:12px}}
    .cell:last-child{{border-right:none}}
    .score{{display:inline-block;font-weight:700;font-size:11px;padding:1px 6px;border-radius:9px;background:#111827;color:#fff;margin-bottom:5px}}
    .foot{{color:#6b7280;font-size:11px;margin-top:18px;line-height:1.5}}
    """
    P = [
        f"<!doctype html><meta charset=utf-8><title>#2223 ansfirst</title><style>{css}</style><div class=wrap>"
    ]
    P.append(
        "<h1>#2223 — First-k answer-token cap: is the every-token benefit front-loaded? (Qwen3-32B)</h1>"
    )
    P.append(
        f'<div class="sub">Delusion frozen replay; band layers; dose ladder unsteered -> cap first '
        f"k=1/2/4/8 answer tokens -> every-token cap. Decode-matched temp 0.7, 3 seeds. "
        f"Harm = Sonnet-4.5 0-100 (higher = worse).</div>"
    )
    P.append(f'<div class="fig"><img src="data:image/png;base64,{b64}"></div>')
    P.append('<div class="hdr">')
    for lbl, stem, c in cols:
        h = harm.get(stem)
        m = f"{h['mean']:.0f}" if h else "-"
        P.append(
            f'<div class="armcard" style="{col_style(c)}"><div class="m">{m}</div><div class="l">{html.escape(lbl)}<br>mean harm</div></div>'
        )
    P.append("</div>")
    for i, bt in enumerate(base_turns):
        turn = bt.get("turn")
        P.append('<div class="turn">')
        P.append(
            f'<div class="user"><b>Turn {turn} — user:</b> {html.escape(bt.get("user") or "")}</div>'
        )
        P.append('<div class="cols">')
        for lbl, stem, c in cols:
            g = gens.get(stem)
            resp = ""
            if g and i < len(g["turns"]):
                resp = html.escape(g["turns"][i].get("assistant") or "")
            sc_val = None
            h = harm.get(stem)
            if h:
                sc_val = h["turn_mean"].get(turn)
            st = "n/a" if sc_val is None else f"{sc_val:.0f}"
            P.append(
                f'<div class="cell" style="background:{hz(sc_val)}"><span class="score">harm {st}</span><br>{resp}</div>'
            )
        P.append("</div></div>")
    P.append(
        '<div class="foot"><b>Provenance.</b> Assistant text = representative draw (seed 42); harm = 3-seed pooled '
        "mean, Sonnet-4.5 judge, 0-100. All arms band layers, published Qwen3-32B assistant axis, Lu p0.25 floor; "
        "ansfirst-k caps only the first k answer tokens, every-token caps all of them. One conversation (n=1).</div>"
    )
    P.append("</div>")
    out = OUT / "ansfirst_qual_dashboard.html"
    out.write_text("".join(P))
    return out


def main() -> None:
    png = make_doseresponse()
    print("DOSE-RESPONSE PNG:", png)
    print(summary_table())
    html_path = build_dashboard(png)
    print("DASHBOARD HTML:", html_path)


if __name__ == "__main__":
    main()
