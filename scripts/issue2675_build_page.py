"""Issue #2675: build the HTML results page (figures, tables, reply browser).

Reads the analysis outputs of ``scripts/issue2675_analyze.py`` (summary.json,
fig/*.png, browser/replies_*.json) plus a hand-written verdicts JSON, and writes

  <out-dir>/index.html          static page (verdicts, glossary, figures, tables)
  <out-dir>/fig/*.png           figures copied from the analysis dir
  <out-dir>/data/<key>.json     reply-browser rows, one file per neuron / control layer

The reply browser fetches data files on demand so no single file nears the
16 MB artifact limit.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

TARGET_ORDER = ("L18.N7807", "L9.N3757", "L8.N1931", "L8.N5315")
PERSONA_WORD = {"god": "God", "nature": "nature"}
MAX_DATA_BYTES = 15_000_000


def pct(x: float | None) -> str:
    """Format a rate as a whole percent, or a dash for missing."""
    if x is None or x != x:
        return "-"
    return f"{100 * x:.0f}%"


def ci(r: dict) -> str:
    """Rate with its Wilson interval, e.g. '82% [74, 88]'."""
    lo, hi = r["wilson"]
    return f"{pct(r['rate'])} <span class=ci>[{100 * lo:.0f}, {100 * hi:.0f}]</span>"


def dose_key(d: dict) -> float:
    """Sort key: signed raw dose."""
    return d["alpha"]


def target_table(name: str, t: dict) -> str:
    """One HTML table of per-dose rates for a target neuron."""
    rows = []
    for d in sorted(t["per_dose"].values(), key=dose_key):
        rule = (
            d["persona_self"]["rate"] >= 0.8
            and d["topic_nonself"]["rate"] <= 0.1
            and d["on_task_nonself"]["rate"] >= 0.9
        )
        by = d["topic_nonself_by_group"]
        src = " class=src" if d["alpha"] == t["source_dose"] else ""
        rows.append(
            f"<tr{src}><td>{d['alpha']:+g}</td><td>{d['alpha_norm']:+.3f}</td>"
            f"<td>{ci(d['persona_self'])}</td><td>{ci(d['persona_headline'])}</td>"
            f"<td>{pct(d['always_embodies'])}</td><td>{ci(d['topic_nonself'])}</td>"
            f"<td>{pct(by['other']['rate'])}</td><td>{pct(by['task']['rate'])}</td>"
            f"<td>{pct(by['creative']['rate'])}</td><td>{ci(d['on_task_nonself'])}</td>"
            f"<td>{pct(d['coherent']['rate'])}</td><td>{pct(d['repetition']['rate'])}</td>"
            f"<td>{pct(d['lexicon_nonself']['rate'])}</td>"
            f"<td>{'<span class=yes>yes</span>' if rule else ''}</td></tr>"
        )
    head = (
        "<tr><th>dose</th><th>normalized</th><th>persona on self</th>"
        "<th>persona on Who/What</th><th>always embodies</th><th>topic on non-self</th>"
        "<th>topic: other</th><th>topic: task</th><th>topic: creative</th>"
        "<th>on task (non-self)</th><th>coherent</th><th>repetition</th>"
        "<th>lexicon (non-self)</th><th>persona rule</th></tr>"
    )
    return f"<div class=scroll><table class=data><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"


def verdict_cards(summary: dict, verdicts: dict) -> str:
    """Per-neuron verdict cards with onset doses and the hand-written reading."""
    cards = []
    for name in TARGET_ORDER:
        t = summary["targets"][name]
        info = t["info"]
        on_p, on_t = t["onset_persona_self"], t["onset_topic_nonself"]
        window = ", ".join(f"{d:+g}" for d in t["persona_rule_doses"]) or "none"
        text = html.escape(verdicts["neurons"].get(name, ""))
        cards.append(
            f"<article class=card><header><h3>{name}</h3>"
            f"<span class='chip v-{t['verdict']}'>{t['verdict']}</span></header>"
            f"<p class=meta>{PERSONA_WORD[t['persona']]} neuron. Source dose {t['source_dose']:+g}"
            f" (normalized {info['normalized_source_dose']:+.3f}).</p>"
            f"<dl><dt>Persona onset</dt><dd>{'none' if on_p is None else f'{on_p:+g}'}</dd>"
            f"<dt>Topic onset</dt><dd>{'none' if on_t is None else f'{on_t:+g}'}</dd>"
            f"<dt>Doses passing the persona rule</dt><dd>{window}</dd></dl>"
            f"<p>{text}</p></article>"
        )
    return "".join(cards)


def checks_html(summary: dict) -> str:
    """Hook equivalence, isolation, sanity, norms and labeller agreement."""
    ch = summary["checks"]
    eq_rows = "".join(
        f"<tr><td>{n}</td><td>{e['alpha']:+g}</td><td>{e['kl_hook_vs_resid']:.2e}</td>"
        f"<td>{e['kl_hook_vs_unsteered']:.3f}</td><td>{'yes' if e['top1_match'] else 'no'}</td></tr>"
        for n, e in ch["equivalence"].items()
    )
    iso = ch["mixed_batch_isolation_kl"]
    iso_txt = max(iso.values())
    norm_rows = "".join(
        f"<tr><td>{n}</td><td>{i['w_down_col_norm']:.3f}</td>"
        f"<td>{i['w_down_col_norm_percentile_in_layer']:.0f}</td>"
        f"<td>{i['median_resid_norm']:.1f}</td><td>{i['normalized_source_dose']:+.3f}</td></tr>"
        for n, i in summary["norms"]["targets"].items()
    )
    ag = summary["agreement"]
    ag_rows = "".join(
        f"<tr><td>{f}</td><td>{pct(ag[f]['agree'])}</td><td>{ag[f]['kappa']:.2f}</td></tr>"
        for f in ("speaker", "theme", "on_task", "coherent")
    )
    sanity = ch["sanity_lexicon_hit_rate"]
    return (
        "<div class=grid2><div><h3>Hook equals residual addition</h3>"
        "<p>Next-token distributions for the pre-hook on <code>down_proj</code> versus adding"
        " alpha times the W_down column to the layer output. KL should be far below the"
        " steering effect.</p><div class=scroll><table class=data><thead><tr><th>neuron</th>"
        "<th>alpha</th><th>KL hook vs residual</th><th>KL hook vs unsteered</th>"
        f"<th>same top token</th></tr></thead><tbody>{eq_rows}</tbody></table></div>"
        f"<p class=note>Mixed-batch isolation, largest KL for an unsteered row: {iso_txt:.1e}."
        f' Sanity check, God-lexicon hit rate on "What are you?" at L18.N7807 +32:'
        f" {pct(sanity)}.</p></div>"
        "<div><h3>Normalized dose</h3><p>Normalized dose = alpha times the W_down column norm,"
        " divided by the median residual norm after the layer (200 Alpaca prompts, all"
        " positions but token 0).</p><div class=scroll><table class=data><thead><tr>"
        "<th>neuron</th><th>W_down column norm</th><th>norm percentile in layer</th>"
        "<th>median residual norm</th><th>normalized source dose</th></tr></thead>"
        f"<tbody>{norm_rows}</tbody></table></div>"
        f"<h3>Labeller agreement</h3><p>A second blind labeller relabelled {ag['n']} replies"
        " (10%).</p><div class=scroll><table class=data><thead><tr><th>field</th>"
        f"<th>agreement</th><th>Cohen's kappa</th></tr></thead><tbody>{ag_rows}</tbody>"
        "</table></div></div></div>"
    )


def controls_html(summary: dict) -> str:
    """Per (matched target, sign) control summary table."""
    rows = []
    for key, s in sorted(summary["controls"]["summary"].items()):
        tgt, sign = key.split("|")
        persona = summary["targets"][tgt]["persona"]
        topic = s["frac_religion_topic_gt10"] if persona == "god" else s["frac_nature_topic_gt10"]
        pers = s["frac_god_gt20"] if persona == "god" else s["frac_nature_gt20"]
        rows.append(
            f"<tr><td>{tgt.split('.')[0]}</td><td>{tgt}</td><td>{sign}</td><td>{s['n_neurons']}</td>"
            f"<td>{pct(s['mean_nonassistant_self'])}</td><td>{pct(s['max_nonassistant_self'])}</td>"
            f"<td>{pct(s['frac_nonassistant_gt20'])}</td><td>{pct(pers)}</td><td>{pct(topic)}</td>"
            f"<td>{pct(s['mean_on_task_nonself'])}</td><td>{pct(s['mean_coherent'])}</td></tr>"
        )
    head = (
        "<tr><th>layer</th><th>dose matched to</th><th>sign</th><th>random neurons</th>"
        "<th>mean non-assistant on self</th><th>max non-assistant on self</th>"
        "<th>neurons above 20% non-assistant</th><th>neurons above 20% matched persona</th>"
        "<th>neurons above 10% matched topic</th><th>mean on task</th><th>mean coherent</th></tr>"
    )
    return f"<div class=scroll><table class=data><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"


def split_browser(analysis: Path, out: Path) -> dict:
    """Split reply rows into per-neuron files; return the browser index."""
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    tg = json.loads((analysis / "browser" / "replies_targets.json").read_text())
    ct = json.loads((analysis / "browser" / "replies_controls.json").read_text())
    groups: dict[str, list] = {}
    for r in tg:
        groups.setdefault("baseline" if r["t"] == "baseline" else r["t"], []).append(r)
    for r in ct:
        layer = r["c"].split(".")[1]
        groups.setdefault(f"controls_{layer}", []).append(r)
    index = {}
    for key, rows in groups.items():
        fname = key.replace(".", "_")
        p = data / f"{fname}.json"
        p.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
        assert p.stat().st_size < MAX_DATA_BYTES, (p, p.stat().st_size)
        index[key] = {"file": f"data/{fname}.json", "n": len(rows)}
    return index


def main() -> None:
    """Parse args, copy figures, split data, write index.html."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    analysis, out = Path(args.analysis_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((analysis / "summary.json").read_text())
    verdicts = json.loads(Path(args.verdicts).read_text())
    (out / "fig").mkdir(exist_ok=True)
    figs = sorted((analysis / "fig").glob("*.png"))
    assert figs, "no figures in analysis dir"
    for f in figs:
        shutil.copy2(f, out / "fig" / f.name)
    index = split_browser(analysis, out)
    tables = "".join(
        f"<h3 id=t-{n.replace('.', '-')}>{n} ({PERSONA_WORD[summary['targets'][n]['persona']]})</h3>"
        + target_table(n, summary["targets"][n])
        for n in TARGET_ORDER
    )
    norm = {n: summary["targets"][n]["info"]["normalized_doses"] for n in TARGET_ORDER}
    page = Path(args.template).read_text()
    subs = {
        "{{Q1}}": html.escape(verdicts["q1"]),
        "{{Q2}}": html.escape(verdicts["q2"]),
        "{{CARDS}}": verdict_cards(summary, verdicts),
        "{{TABLES}}": tables,
        "{{CONTROLS}}": controls_html(summary),
        "{{CHECKS}}": checks_html(summary),
        "{{N_REPLIES}}": f"{summary['n_replies']:,}",
        "{{CAP_HIT}}": pct(summary["cap_hit_overall"]),
        "{{INDEX_JSON}}": json.dumps(index),
        "{{NORM_JSON}}": json.dumps(norm),
        "{{PROVENANCE}}": html.escape(verdicts.get("provenance", "")),
    }
    for k, v in subs.items():
        assert k in page, f"template missing {k}"
        page = page.replace(k, v)
    (out / "index.html").write_text(page)
    print(f"[page] wrote {out / 'index.html'} with {len(figs)} figures, {len(index)} data files")


if __name__ == "__main__":
    main()
