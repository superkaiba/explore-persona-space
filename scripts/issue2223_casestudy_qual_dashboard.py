"""Issue #2223 — minimal qualitative case-study dashboard + harm-per-turn plot.

Compares three arms of the frozen delusion replay (Qwen3-32B) side by side:
  - unsteered (na__unsteered)
  - every-token cap, band layers (band__cap_alltoken) — the paper-style defence
  - context-end floor, all layers, p100 (all__cap_ctx_p100) — strongest ctx-end floor

Per turn: the FROZEN user message (Lu et al. published delusion conversation) and
each arm's assistant response, coloured by that turn's judged harm (3-draw mean,
Sonnet-4.5, 0-100). Assistant text shown is the representative draw; the harm
number is the 3-draw mean, so text and score are disclosed as different draws.

Reads generation JSONs + scores_delusion.json; writes a self-contained HTML and
a harm-per-turn PNG under figures/issue_2223/casestudy_replay/qwen3-32b/.
No harmful content is paged to stdout — it flows straight into the HTML file.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "eval_results/issue_2223/casestudy_replay/qwen3-32b/delusion"
SCORES = ROOT / "eval_results/issue_2223/casestudy_replay/qwen3-32b/judged/scores_delusion.json"
OUT_DIR = ROOT / "figures/issue_2223/casestudy_replay/qwen3-32b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (file stem, display label, colour) — one colour per arm, shared plot<->dashboard.
ARMS = [
    ("na__unsteered", "Unsteered", "#6b7280"),
    ("band__cap_alltoken", "Every-token cap (band layers)", "#2563eb"),
    ("all__cap_ctx_p100", "Context-end floor, all layers (p100)", "#ea580c"),
]


def load_arm(stem: str) -> dict:
    return json.loads((GEN / f"{stem}.json").read_text())


def load_turn_harm(stem: str) -> dict[int, float]:
    """Per-turn 3-draw-mean judged harm for one arm from scores_delusion.json."""
    cells = json.loads(SCORES.read_text())["cells"]
    cell = cells.get(stem, {})
    out: dict[int, float] = {}
    for tk, tv in cell.items():
        if isinstance(tv, dict) and isinstance(tv.get("score"), (int, float)):
            out[int(tk)] = float(tv["score"])
    return out


def harm_color(score: float | None) -> str:
    if score is None:
        return "#f3f4f6"
    # green (low) -> amber -> red (high), on a 0-100 harm scale.
    if score < 25:
        return "#dcfce7"
    if score < 50:
        return "#fef9c3"
    if score < 75:
        return "#fed7aa"
    return "#fecaca"


def make_plot() -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=140)
    for stem, label, color in ARMS:
        h = load_turn_harm(stem)
        xs = sorted(h)
        ys = [h[t] for t in xs]
        ax.plot(xs, ys, marker="o", ms=4, lw=2, color=color, label=label)
    ax.set_xlabel("Conversation turn")
    ax.set_ylabel("Judged harm (0-100, 3-draw mean)")
    ax.set_ylim(-3, 103)
    ax.set_title("Delusion replay (Qwen3-32B): per-turn judged harm")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    png = OUT_DIR / "qual_delusion_harm_per_turn.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return png


def build_html(png: Path) -> Path:
    arm_data = {stem: load_arm(stem) for stem, _, _ in ARMS}
    arm_harm = {stem: load_turn_harm(stem) for stem, _, _ in ARMS}
    base_turns = arm_data["na__unsteered"]["turns"]

    def overall(stem: str) -> float:
        v = list(arm_harm[stem].values())
        return sum(v) / len(v) if v else float("nan")

    b64 = base64.b64encode(png.read_bytes()).decode()

    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f9fafb;color:#111827;font-size:14px}
    .wrap{max-width:1500px;margin:0 auto;padding:18px 22px}
    h1{font-size:20px;margin:0 0 4px}
    .sub{color:#6b7280;font-size:13px;margin-bottom:16px}
    .fig{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin-bottom:20px;text-align:center}
    .fig img{max-width:100%}
    .hdr{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;position:sticky;top:0;background:#f9fafb;padding:8px 0;z-index:5}
    .armcard{border-radius:8px;padding:10px 12px;color:#fff}
    .armcard .m{font-size:22px;font-weight:700}
    .armcard .l{font-size:12px;opacity:.95}
    .turn{background:#fff;border:1px solid #e5e7eb;border-radius:8px;margin-bottom:14px;overflow:hidden}
    .user{background:#eff6ff;border-bottom:1px solid #dbeafe;padding:10px 12px;font-size:13px}
    .user b{color:#1d4ed8}
    .cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0}
    .cell{padding:10px 12px;border-right:1px solid #f3f4f6;white-space:pre-wrap;line-height:1.42}
    .cell:last-child{border-right:none}
    .score{display:inline-block;font-weight:700;font-size:12px;padding:1px 7px;border-radius:10px;background:#111827;color:#fff;margin-bottom:6px}
    .foot{color:#6b7280;font-size:12px;margin-top:22px;line-height:1.5}
    """

    parts = [
        f"<!doctype html><meta charset=utf-8><title>#2223 delusion qual</title><style>{css}</style>"
    ]
    parts.append('<div class="wrap">')
    parts.append("<h1>#2223 — Delusion replay, qualitative arm comparison (Qwen3-32B)</h1>")
    parts.append(
        '<div class="sub">Frozen replay of Lu et al.\'s published delusion conversation; '
        "user turns fixed, model regenerates each assistant turn under each intervention.</div>"
    )
    parts.append(f'<div class="fig"><img src="data:image/png;base64,{b64}"></div>')

    # sticky arm header with overall mean harm
    parts.append('<div class="hdr">')
    for stem, label, color in ARMS:
        parts.append(
            f'<div class="armcard" style="background:{color}">'
            f'<div class="m">{overall(stem):.1f}</div>'
            f'<div class="l">{html.escape(label)} — mean harm</div></div>'
        )
    parts.append("</div>")

    for i, bt in enumerate(base_turns):
        turn = bt["turn"]
        user = html.escape(bt.get("user") or "")
        parts.append('<div class="turn">')
        parts.append(f'<div class="user"><b>Turn {turn} — user:</b> {user}</div>')
        parts.append('<div class="cols">')
        for stem, _label, color in ARMS:
            turns = arm_data[stem]["turns"]
            resp = ""
            if i < len(turns):
                resp = html.escape(turns[i].get("assistant") or "")
            sc = arm_harm[stem].get(turn)
            sc_txt = "n/a" if sc is None else f"{sc:.0f}"
            parts.append(
                f'<div class="cell" style="background:{harm_color(sc)}">'
                f'<span class="score">harm {sc_txt}</span><br>{resp}</div>'
            )
        parts.append("</div></div>")

    parts.append(
        '<div class="foot">'
        "<b>Provenance.</b> Assistant text = representative draw (draw 0). "
        "Harm number = 3-draw mean, Claude-sonnet-4.5 judge, 0-100 (higher = more harmful; "
        "threshold 50). Cell colour bins harm: green &lt;25, yellow &lt;50, orange &lt;75, red &ge;75. "
        "Arms: Unsteered = no intervention; Every-token cap (band layers) = the paper-style "
        "assistant-axis floor applied at every token (the #2223 headline defence, coherence 98.8); "
        "Context-end floor all-layers p100 = the strongest floor applied only at the context-end "
        "token each turn (coherence 100). One conversation per behaviour (n=1).</div>"
    )
    parts.append("</div>")
    out = OUT_DIR / "qual_delusion_dashboard.html"
    out.write_text("".join(parts))
    return out


def main() -> None:
    png = make_plot()
    html_path = build_html(png)
    print(f"PNG:  {png}")
    print(f"HTML: {html_path}")
    for stem, label, _ in ARMS:
        h = load_turn_harm(stem)
        mean = sum(h.values()) / len(h) if h else float("nan")
        print(f"  {label:42s} mean_harm={mean:.1f} n_turns={len(h)}")


if __name__ == "__main__":
    main()
