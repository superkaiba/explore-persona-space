"""Issue #2675: analysis of the single-neuron steering dose sweeps.

Reads the generated replies (``raw_completions/{targets,controls}``), the blind
labels (``labels.jsonl`` from ``issue2675_judge.py merge``), ``norms.json`` and
``checks.json``; writes per-cell stats, the summary table, the decision-rule
verdicts, labeller agreement, figures, and the reply-browser data files.

Definitions (stated again in the results page):
- persona voice on self: labelled speaker == the neuron's persona (god / nature).
  "All self" uses samples 0-4 of every self question, so each of the 40 questions
  has equal weight; "Who/What are you?" uses all 50 samples of those two.
- topic on non-self: theme in {religion, both} for God neurons, {nature, both} for
  the nature neuron, over the 24 non-self questions.
- on-task on non-self: on_task == yes. Coherence: coherent == yes over all replies.
- repetition: any word 4-gram occurring 3 or more times in a reply.
- Wilson 95% intervals over samples; percentile bootstrap (2000 draws) over questions.
- always embodies: share of self questions with >= 90% of samples in persona.
- onset dose: first source-sign dose (increasing magnitude) whose rate exceeds 20%.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_PERSONA = {
    "L18.N7807": "god",
    "L9.N3757": "god",
    "L8.N1931": "god",
    "L8.N5315": "nature",
}
TOPIC_THEMES = {"god": {"religion", "both"}, "nature": {"nature", "both"}}
HEADLINE = ("Who are you?", "What are you?")
NONSELF_GROUPS = ("other", "task", "creative")
RELIGION_WORDS = (
    r"god|gods|goddess|lord|divine|divinity|heaven|heavens|heavenly|holy|sacred|creator|"
    r"almighty|eternal|prayer|pray|praise|worship|spirit|soul|souls|angel|angels|bible|"
    r"biblical|scripture|genesis|psalm|psalms|amen|deity|deities|blessed|bless|salvation|"
    r"covenant|prophet|church|temple|glory|jesus|christ|sin|sins|faith"
)
NATURE_WORDS = (
    r"nature|earth|tree|trees|forest|forests|river|rivers|mountain|mountains|season|"
    r"seasons|seed|seeds|leaf|leaves|flower|flowers|bloom|blossom|soil|rain|wind|ocean|"
    r"sea|grass|root|roots|vegetation|wild|wilderness|meadow|spring|autumn|winter|"
    r"summer|plant|plants|garden|moss|petal|petals|sunlight|branches|bark"
)
RELIGION_RE = re.compile(rf"\b({RELIGION_WORDS})\b", re.IGNORECASE)
NATURE_RE = re.compile(rf"\b({NATURE_WORDS})\b", re.IGNORECASE)
N_BOOT = 2000
RNG = np.random.default_rng(2675)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def boot_questions(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Percentile bootstrap over questions of the pooled rate of boolean ``col``."""
    g = df.groupby("qid")[col].agg(["sum", "count"])
    if len(g) == 0:
        return (float("nan"), float("nan"))
    k, n = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = RNG.integers(0, len(g), size=(N_BOOT, len(g)))
    r = k[idx].sum(1) / n[idx].sum(1)
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def repeated_4gram(text: str) -> bool:
    """True when any word 4-gram occurs 3 or more times."""
    w = re.findall(r"\w+", text.lower())
    grams = Counter(tuple(w[i : i + 4]) for i in range(len(w) - 3))
    return bool(grams) and max(grams.values()) >= 3


def load(args) -> pd.DataFrame:
    """Load replies + labels into one frame with derived boolean columns."""
    rows = []
    for phase in ("targets", "controls"):
        for p in sorted((Path(args.replies_root) / "raw_completions" / phase).glob("b*.jsonl")):
            for line in p.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    r["phase"] = phase
                    rows.append(r)
    df = pd.DataFrame(rows)
    lab = pd.DataFrame([json.loads(x) for x in Path(args.labels).open(encoding="utf-8")])
    assert lab["job_id"].is_unique
    df = df.merge(lab, on="job_id", how="left", validate="one_to_one")
    missing = int(df["speaker"].isna().sum())
    assert missing == 0, f"{missing} replies have no label"
    df["rep4"] = df["text"].map(repeated_4gram)
    df["lex_religion"] = df["text"].map(lambda t: bool(RELIGION_RE.search(t)))
    df["lex_nature"] = df["text"].map(lambda t: bool(NATURE_RE.search(t)))
    df["is_coherent"] = df["coherent"] == "yes"
    df["is_on_task"] = df["on_task"] == "yes"
    df["is_on_task_or_partial"] = df["on_task"].isin(["yes", "partial"])
    df["is_nonassistant"] = ~df["speaker"].isin(["assistant", "none"])
    df["is_god"] = df["speaker"] == "god"
    df["is_nature"] = df["speaker"] == "nature"
    df["theme_religion"] = df["theme"].isin(["religion", "both"])
    df["theme_nature"] = df["theme"].isin(["nature", "both"])
    df["headline"] = df["question"].isin(HEADLINE)
    df["self_equal"] = (df["kind"] == "self") & (df["sample"] < 5)
    return df


def rate(df: pd.DataFrame, col: str, boot: bool = True) -> dict:
    """Rate of boolean ``col`` with Wilson CI (+ question bootstrap)."""
    k, n = int(df[col].sum()), int(len(df))
    out = {"k": k, "n": n, "rate": k / n if n else float("nan"), "wilson": wilson(k, n)}
    if boot:
        out["boot_q"] = boot_questions(df, col)
    return out


def cell_stats(c: pd.DataFrame, persona: str) -> dict:
    """All per-cell rates for one (condition) and the given persona."""
    pcol = "is_god" if persona == "god" else "is_nature"
    tcol = "theme_religion" if persona == "god" else "theme_nature"
    lcol = "lex_religion" if persona == "god" else "lex_nature"
    self_eq = c[c["self_equal"]]
    head = c[(c["kind"] == "self") & c["headline"]]
    ns = c[c["kind"] == "nonself"]
    self_all = c[c["kind"] == "self"]
    perq = self_all.groupby("qid")[pcol].mean()
    s = {
        "persona": persona,
        "persona_self": rate(self_eq, pcol),
        "persona_headline": rate(head, pcol),
        "nonassistant_self": rate(self_eq, "is_nonassistant"),
        "topic_nonself": rate(ns, tcol),
        "topic_nonself_by_group": {g: rate(ns[ns["group"] == g], tcol) for g in NONSELF_GROUPS},
        "lexicon_nonself": rate(ns, lcol, boot=False),
        "on_task_nonself": rate(ns, "is_on_task"),
        "on_task_or_partial_nonself": rate(ns, "is_on_task_or_partial", boot=False),
        "coherent": rate(c, "is_coherent"),
        "coherent_nonself": rate(ns, "is_coherent", boot=False),
        "repetition": rate(c, "rep4", boot=False),
        "cap_hit": float((c["finish_reason"] == "length").mean()),
        "always_embodies": float((perq >= 0.9).mean()) if len(perq) else float("nan"),
        "per_question_persona": {q: float(v) for q, v in perq.items()},
        "speaker_counts_self": self_eq["speaker"].value_counts().to_dict(),
        "other_notes_self": Counter(
            self_eq.loc[self_eq["speaker"] == "other", "note"].str.lower()
        ).most_common(8),
        "theme_counts_nonself": ns["theme"].value_counts().to_dict(),
    }
    return s


def analyze_targets(df: pd.DataFrame, norms: dict) -> dict:
    """Per-dose stats, onset doses and decision-rule verdicts per target neuron."""
    t = df[df["phase"] == "targets"]
    base = t[t["cond_kind"] == "baseline"]
    out = {}
    for name, persona in TARGET_PERSONA.items():
        info = norms["targets"][name]
        src = info["source_dose"]
        cells = t[t["target"] == name]
        doses = sorted(cells["alpha"].unique(), key=float)
        per_dose = {"0": {**cell_stats(base, persona), "alpha": 0.0, "alpha_norm": 0.0}}
        for d in doses:
            c = cells[cells["alpha"] == d]
            per_dose[f"{d:g}"] = {
                **cell_stats(c, persona),
                "alpha": float(d),
                "alpha_norm": float(c["alpha_norm"].iloc[0]),
            }
        main = sorted([d for d in doses if np.sign(d) == np.sign(src)], key=lambda x: abs(float(x)))

        def onset(key: str, main=main, per_dose=per_dose) -> float | None:
            for d in main:
                if per_dose[f"{d:g}"][key]["rate"] > 0.20:
                    return float(d)
            return None

        persona_doses = [
            float(d)
            for d in main
            if per_dose[f"{d:g}"]["persona_self"]["rate"] >= 0.80
            and per_dose[f"{d:g}"]["topic_nonself"]["rate"] <= 0.10
            and per_dose[f"{d:g}"]["on_task_nonself"]["rate"] >= 0.90
        ]
        on_p, on_t = onset("persona_self"), onset("topic_nonself")
        if persona_doses:
            verdict = "persona"
        elif on_t is not None and (on_p is None or abs(on_t) <= abs(on_p)):
            verdict = "concept"
        else:
            verdict = "mixed"
        out[name] = {
            "persona": persona,
            "source_dose": src,
            "info": info,
            "per_dose": per_dose,
            "onset_persona_self": on_p,
            "onset_topic_nonself": on_t,
            "persona_rule_doses": persona_doses,
            "verdict": verdict,
        }
    return out


def analyze_controls(df: pd.DataFrame) -> dict:
    """Per random-neuron condition stats + per (layer, matched target, sign) summary."""
    c = df[df["phase"] == "controls"]
    rows = []
    for cid, g in c.groupby("cond_id"):
        persona = TARGET_PERSONA[g["target"].iloc[0]]
        s = cell_stats(g, persona)
        ns = g[g["kind"] == "nonself"]
        rows.append(
            {
                "cond_id": cid,
                "layer": int(g["layer"].iloc[0]),
                "neuron": int(g["neuron"].iloc[0]),
                "matched_target": g["target"].iloc[0],
                "sign": "+" if g["alpha"].iloc[0] > 0 else "-",
                "alpha": float(g["alpha"].iloc[0]),
                "alpha_norm": float(g["alpha_norm"].iloc[0]),
                "nonassistant_self": s["nonassistant_self"]["rate"],
                "god_self": float(g.loc[g["self_equal"], "is_god"].mean()),
                "nature_self": float(g.loc[g["self_equal"], "is_nature"].mean()),
                "religion_nonself": float(ns["theme_religion"].mean()),
                "nature_nonself": float(ns["theme_nature"].mean()),
                "on_task_nonself": s["on_task_nonself"]["rate"],
                "coherent": s["coherent"]["rate"],
                "other_notes_self": s["other_notes_self"],
                "theme_counts_nonself": s["theme_counts_nonself"],
            }
        )
    tab = pd.DataFrame(rows)
    summary = {}
    for (tgt, sign), g in tab.groupby(["matched_target", "sign"]):
        summary[f"{tgt}|{sign}"] = {
            "n_neurons": int(len(g)),
            "frac_nonassistant_gt20": float((g["nonassistant_self"] > 0.20).mean()),
            "mean_nonassistant_self": float(g["nonassistant_self"].mean()),
            "max_nonassistant_self": float(g["nonassistant_self"].max()),
            "frac_god_gt20": float((g["god_self"] > 0.20).mean()),
            "frac_nature_gt20": float((g["nature_self"] > 0.20).mean()),
            "frac_religion_topic_gt10": float((g["religion_nonself"] > 0.10).mean()),
            "frac_nature_topic_gt10": float((g["nature_nonself"] > 0.10).mean()),
            "mean_on_task_nonself": float(g["on_task_nonself"].mean()),
            "mean_coherent": float(g["coherent"].mean()),
        }
    return {"per_neuron": rows, "summary": summary}


def agreement(df: pd.DataFrame) -> dict:
    """Per-field percent agreement and Cohen's kappa between the two labellers."""
    sub = df[df["second"].notna()]
    out = {"n": int(len(sub))}
    for f in ("speaker", "theme", "on_task", "coherent"):
        a = sub[f].to_numpy()
        b = np.array([s[f] for s in sub["second"]])
        po = float((a == b).mean())
        cats = sorted(set(a) | set(b))
        pe = sum(float((a == k).mean()) * float((b == k).mean()) for k in cats)
        out[f] = {"agree": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan")}
    return out


def summary_table(targets: dict) -> pd.DataFrame:
    """One row per (neuron, dose) with the headline rates."""
    rows = []
    for name, t in targets.items():
        for key, s in t["per_dose"].items():
            rows.append(
                {
                    "neuron": name,
                    "persona": t["persona"],
                    "dose": s["alpha"],
                    "dose_norm": round(s["alpha_norm"], 3),
                    "persona_self": round(s["persona_self"]["rate"], 3),
                    "persona_who_what": round(s["persona_headline"]["rate"], 3),
                    "always_embodies": round(s["always_embodies"], 3),
                    "topic_nonself": round(s["topic_nonself"]["rate"], 3),
                    "topic_other": round(s["topic_nonself_by_group"]["other"]["rate"], 3),
                    "topic_task": round(s["topic_nonself_by_group"]["task"]["rate"], 3),
                    "topic_creative": round(s["topic_nonself_by_group"]["creative"]["rate"], 3),
                    "on_task_nonself": round(s["on_task_nonself"]["rate"], 3),
                    "coherent": round(s["coherent"]["rate"], 3),
                    "repetition": round(s["repetition"]["rate"], 3),
                    "lexicon_nonself": round(s["lexicon_nonself"]["rate"], 3),
                    "persona_rule": (
                        s["persona_self"]["rate"] >= 0.8
                        and s["topic_nonself"]["rate"] <= 0.1
                        and s["on_task_nonself"]["rate"] >= 0.9
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["neuron", "dose"])


def plot_all(targets: dict, controls: dict, fig_dir: Path) -> list[Path]:
    """Per-neuron dose plots, cross-neuron comparison, per-question strips, controls."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style("generic")
    pal = paper_palette(4)
    series = [
        ("persona_self", "Persona voice on self questions", pal[0]),
        ("topic_nonself", "Topic in non-self replies", pal[1]),
        ("on_task_nonself", "On task on non-self questions", pal[2]),
        ("coherent", "Coherent", pal[3]),
    ]
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, t in targets.items():
        pd_ = sorted(t["per_dose"].values(), key=lambda s: s["alpha"])
        x = np.array([s["alpha"] for s in pd_])
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        span = float(x.max() - x.min()) or 1.0
        for i, (key, label, col) in enumerate(series):
            y = np.array([s[key]["rate"] for s in pd_])
            lo = np.array([s[key]["wilson"][0] for s in pd_])
            hi = np.array([s[key]["wilson"][1] for s in pd_])
            ax.errorbar(
                x + (i - 1.5) * 0.006 * span,
                y,
                yerr=[np.maximum(0, y - lo), np.maximum(0, hi - y)],
                marker="o",
                ms=4,
                lw=1.6,
                capsize=2,
                color=col,
                label=label,
            )
        ax.axhline(0.8, color="0.6", lw=0.8, ls=":")
        ax.axhline(0.1, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("Dose (alpha added to the neuron activation)")
        ax.set_ylabel("Share of replies")
        ax.set_ylim(-0.03, 1.03)
        persona = "God" if t["persona"] == "god" else "nature"
        ax.set_title(f"{name} ({persona})", pad=34)
        k = t["info"]["w_down_col_norm"] / t["info"]["median_resid_norm"]
        sec = ax.secondary_xaxis("top", functions=(lambda a, k=k: a * k, lambda d, k=k: d / k))
        sec.set_xlabel("Normalized dose")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        p = fig_dir / f"dose_{name.replace('.', '_')}.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # cross-neuron comparison on the normalized axis (source-sign doses only)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for i, (name, t) in enumerate(targets.items()):
        sgn = np.sign(t["source_dose"])
        pts = sorted(
            [s for s in t["per_dose"].values() if np.sign(s["alpha"]) in (sgn, 0)],
            key=lambda s: abs(s["alpha"]),
        )
        xd = [abs(s["alpha_norm"]) for s in pts]
        for ax, key in zip(axes, ("persona_self", "topic_nonself"), strict=True):
            ax.plot(xd, [s[key]["rate"] for s in pts], marker="o", ms=4, color=pal[i], label=name)
    axes[0].set_title("Persona voice on self questions")
    axes[1].set_title("Topic in non-self replies")
    for ax in axes:
        ax.set_xlabel("|Normalized dose|")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Share of replies")
    axes[1].legend(fontsize=8, frameon=False)
    p = fig_dir / "compare_normalized.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # per-question persona fraction (low-level view)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, (name, t) in zip(axes.ravel(), targets.items(), strict=True):
        pts = sorted(t["per_dose"].values(), key=lambda s: s["alpha"])
        for j, s in enumerate(pts):
            v = np.array(list(s["per_question_persona"].values()))
            ax.scatter(j + RNG.uniform(-0.18, 0.18, len(v)), v, s=9, alpha=0.6, color=pal[0])
        ax.set_xticks(range(len(pts)), [f"{s['alpha']:g}" for s in pts], fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Persona share per question")
    for ax in axes[1]:
        ax.set_xlabel("Dose")
    fig.tight_layout()
    p = fig_dir / "per_question_persona.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # random controls vs targets at the matched source dose
    tab = pd.DataFrame(controls["per_neuron"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, layer_set, title in (
        (axes[0], {"L18.N7807", "L9.N3757", "L8.N1931"}, "God-matched doses"),
        (axes[1], {"L8.N5315"}, "Nature-matched dose"),
    ):
        sub = tab[tab["matched_target"].isin(layer_set)]
        tcol = "religion_nonself" if "God" in title else "nature_nonself"
        ax.scatter(
            sub["nonassistant_self"],
            sub[tcol],
            s=16,
            color="0.55",
            alpha=0.8,
            label="Random neuron (each sign)",
        )
        for i, (name, t) in enumerate(targets.items()):
            if name not in layer_set:
                continue
            s = min(t["per_dose"].values(), key=lambda v: abs(v["alpha"] - t["source_dose"]))
            ax.scatter(
                s["nonassistant_self"]["rate"],
                s["topic_nonself"]["rate"],
                s=60,
                marker="*",
                color=pal[i],
                label=f"{name} at source dose",
            )
        ax.set_xlabel("Non-assistant identity on self questions")
        ax.set_title(title)
        ax.set_xlim(-0.03, 1.03)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[0].set_ylabel("Religion topic in non-self replies")
    axes[1].set_ylabel("Nature topic in non-self replies")
    axes[1].yaxis.set_tick_params(labelleft=True)
    p = fig_dir / "controls.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def browser_data(df: pd.DataFrame, out_dir: Path) -> None:
    """Compact reply rows for the HTML reply browser, split by phase."""
    cols = {
        "c": "cond_id",
        "t": "target",
        "a": "alpha",
        "q": "question",
        "g": "group",
        "s": "sample",
        "x": "text",
        "sp": "speaker",
        "no": "note",
        "th": "theme",
        "ot": "on_task",
        "co": "coherent",
    }
    for phase in ("targets", "controls"):
        sub = df[df["phase"] == phase]
        recs = [{k: r[v] for k, v in cols.items()} for r in sub.to_dict("records")]
        for r in recs:
            r["t"] = r["t"] if isinstance(r["t"], str) else "baseline"
        (out_dir / f"replies_{phase}.json").write_text(
            json.dumps(recs, ensure_ascii=False, separators=(",", ":"))
        )


def main() -> None:
    """Parse args, analyze, and write all outputs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replies-root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--browser-dir", required=True)
    args = ap.parse_args()
    root = Path(args.replies_root)
    norms = json.loads((root / "norms.json").read_text())
    checks = json.loads((root / "checks.json").read_text())
    df = load(args)
    targets = analyze_targets(df, norms)
    controls = analyze_controls(df)
    agree = agreement(df)
    tab = summary_table(targets)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "summary_table.csv", index=False)
    (out / "summary_table.md").write_text(tab.to_markdown(index=False))
    res = {
        "targets": targets,
        "controls": controls,
        "agreement": agree,
        "checks": {k: v for k, v in checks.items() if k != "sanity"}
        | {"sanity_lexicon_hit_rate": checks["sanity"]["lexicon_hit_rate"]},
        "norms": {"targets": norms["targets"], "n_prompts": norms["n_prompts"]},
        "n_replies": int(len(df)),
        "cap_hit_overall": float((df["finish_reason"] == "length").mean()),
    }
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    paths = plot_all(targets, controls, Path(args.fig_dir))
    bd = Path(args.browser_dir)
    bd.mkdir(parents=True, exist_ok=True)
    browser_data(df, bd)
    print(f"[analyze] {len(df)} replies; figures: {[p.name for p in paths]}")
    for name, t in targets.items():
        print(
            f"[analyze] {name}: verdict={t['verdict']} onset_persona={t['onset_persona_self']} "
            f"onset_topic={t['onset_topic_nonself']} persona_rule_doses={t['persona_rule_doses']}"
        )
    print(f"[analyze] agreement: {agree}")


if __name__ == "__main__":
    main()
