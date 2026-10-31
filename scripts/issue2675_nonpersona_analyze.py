"""Issue #2675 follow-up: analyze the non-persona neuron strength x rollout sweep.

A reply counts as persona when its blind label has a speaker other than
assistant/none AND is coherent. Per neuron-direction and strength the persona
rate is taken over the 20 sampled replies; the greedy reply is reported
separately (it is the library's own decoding).

Neuron-level flip (selection-symmetric): a neuron flips when its maximum
persona rate over strengths exceeds the 95th percentile of the same maximum
over the random control neurons.

Outputs (under --out-dir): summary.json, per_neuron.csv, per_cell.csv, figures.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SETS = [
    ("test_off_topic", "Library off-topic"),
    ("test_changed_assistant", "Library changed-assistant"),
    ("test_named", "Named (L8.N5315, L9.N10974, L10.N5809)"),
    ("control_persona", "Library persona (control)"),
    ("control_random", "Random, not in library (control)"),
]
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


def boot_mean(x: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over units (neurons)."""
    if len(x) == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def norm_text(t: str) -> str:
    """Lowercase, collapse whitespace, for prefix matching against library excerpts."""
    return re.sub(r"\s+", " ", t).strip().lower()


def load(args) -> pd.DataFrame:
    """Replies joined to labels, with persona and coherence flags."""
    rows = []
    for p in sorted((Path(args.replies_root) / "raw_completions" / "nonpersona").glob("b*.jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    lab = pd.DataFrame([json.loads(x) for x in Path(args.labels).open(encoding="utf-8")])
    assert lab["job_id"].is_unique
    df = df.merge(lab, on="job_id", how="left", validate="one_to_one")
    missing = int(df["speaker"].isna().sum())
    if args.allow_partial:
        print(f"[analyze] PARTIAL: dropping {missing} unlabelled replies of {len(df)}")
        df = df[df["speaker"].notna()].copy()
    else:
        assert missing == 0, f"{missing} replies have no label"
    df["coherent_b"] = df["coherent"] == "yes"
    df["nonassistant"] = ~df["speaker"].isin(["assistant", "none"])
    df["persona"] = df["nonassistant"] & df["coherent_b"]
    df["alpha_lib"] = df["alpha_norm"].abs()
    return df


def per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (neuron-direction, strength): sampled persona rate + greedy label."""
    s = df[(df["decode"] == "sample") & (df["set"] != "baseline")]
    g = df[(df["decode"] == "greedy") & (df["set"] != "baseline")]
    cell = (
        s.groupby(["set", "target", "alpha_lib"])
        .agg(
            n=("persona", "size"),
            k=("persona", "sum"),
            coherent=("coherent_b", "mean"),
            nonassistant=("nonassistant", "mean"),
        )
        .reset_index()
    )
    cell["rate"] = cell["k"] / cell["n"]
    gk = g.set_index(["target", "alpha_lib"])
    cell["greedy_persona"] = [
        bool(gk.loc[(t, a), "persona"]) if (t, a) in gk.index else None
        for t, a in zip(cell["target"], cell["alpha_lib"], strict=True)
    ]
    cell["greedy_speaker"] = [
        gk.loc[(t, a), "speaker"] if (t, a) in gk.index else None
        for t, a in zip(cell["target"], cell["alpha_lib"], strict=True)
    ]
    return cell


def per_neuron(cell: pd.DataFrame, picks: dict) -> pd.DataFrame:
    """Max-over-strengths persona rate, its strength, and library fields per neuron."""
    rows = []
    for t, c in cell.groupby("target"):
        i = int(c["rate"].to_numpy().argmax())
        e = picks[t]
        rows.append(
            {
                "target": t,
                "set": c["set"].iloc[0],
                "max_rate": float(c["rate"].max()),
                "argmax_alpha": float(c["alpha_lib"].iloc[i]),
                "any_sampled_persona": bool((c["k"] > 0).any()),
                "any_greedy_persona": bool(c["greedy_persona"].fillna(False).any()),
                "lib_group": e.get("lib_group"),
                "lib_name": e.get("lib_name"),
                "lib_strength": e.get("lib_strength"),
            }
        )
    return pd.DataFrame(rows)


def fidelity(df: pd.DataFrame, picks: dict, n_chars: int = 60) -> dict:
    """Greedy reply at the library's kept strength vs the library excerpt (prefix match)."""
    g = df[(df["decode"] == "greedy") & (df["set"] != "baseline")]
    out = {}
    for name, _ in SETS:
        if name == "control_random":
            continue
        hits, n = 0, 0
        for t, e in picks.items():
            if e["set"] != name or e.get("lib_strength") is None:
                continue
            r = g[(g["target"] == t) & np.isclose(g["alpha_lib"], e["lib_strength"])]
            if len(r) != 1:
                continue
            n += 1
            a = norm_text(r["text"].iloc[0])[:n_chars]
            b = norm_text(e["lib_reply_excerpt"])[:n_chars]
            hits += int(a == b)
        out[name] = {"n": n, "prefix_match": hits, "rate": hits / n if n else None}
    return out


def plot_all(cell: pd.DataFrame, neur: pd.DataFrame, thr: float, fig_dir: Path) -> list[Path]:
    """Group curves vs strength, per-neuron max strip, per-neuron heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style("generic")
    pal = paper_palette(len(SETS))
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for (name, label), col in zip(SETS, pal, strict=True):
        c = cell[cell["set"] == name]
        if c.empty:
            continue
        for ax, key in zip(axes, ("rate", "coherent"), strict=True):
            xs, ys, lo, hi = [], [], [], []
            for a, ca in c.groupby("alpha_lib"):
                v = ca[key].to_numpy(float)
                xs.append(a)
                ys.append(v.mean())
                b = boot_mean(v)
                lo.append(b[0])
                hi.append(b[1])
            ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
            ax.errorbar(
                xs,
                ys,
                yerr=[np.maximum(0, ys - lo), np.maximum(0, hi - ys)],
                marker="o",
                ms=4,
                lw=1.6,
                capsize=2,
                color=col,
                label=f"{label} (n={c['target'].nunique()})",
            )
    axes[0].set_ylabel("Persona replies (share of 20 sampled)")
    axes[1].set_ylabel("Coherent replies (share of 20 sampled)")
    for ax in axes:
        ax.set_xlabel("Strength (library scale, x r_L / ||W_down||)")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_title("Persona rate vs strength")
    axes[1].set_title("Coherence vs strength")
    axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    p = fig_dir / "group_rate_vs_strength.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, ((name, label), col) in enumerate(zip(SETS, pal, strict=True)):
        v = neur[neur["set"] == name]
        x = i + RNG.uniform(-0.18, 0.18, size=len(v))
        ax.scatter(x, v["max_rate"], s=18, color=col, alpha=0.85)
        if name == "test_named":
            for xi, (_, r) in zip(x, v.iterrows(), strict=True):
                ax.annotate(
                    r["target"],
                    (xi, r["max_rate"]),
                    fontsize=7,
                    xytext=(4, 2),
                    textcoords="offset points",
                )
    ax.axhline(thr, color="0.4", lw=1.0, ls="--", label=f"Random 95th percentile ({thr:.2f})")
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([lab.split(" (")[0] for _, lab in SETS], rotation=15, ha="right")
    ax.set_ylabel("Max persona rate over strengths")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    p = fig_dir / "per_neuron_max.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    order = []
    for name, _ in SETS:
        v = neur[neur["set"] == name].sort_values("max_rate", ascending=False)
        order += list(v["target"])
    alphas = sorted(cell["alpha_lib"].unique())
    mat = np.full((len(order), len(alphas)), np.nan)
    for i, t in enumerate(order):
        for j, a in enumerate(alphas):
            r = cell[(cell["target"] == t) & np.isclose(cell["alpha_lib"], a)]
            if len(r):
                mat[i, j] = r["rate"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.0, 0.13 * len(order) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=5)
    bounds = np.cumsum([int((neur["set"] == n).sum()) for n, _ in SETS])[:-1]
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=1.2)
    ax.set_xlabel("Strength (library scale)")
    fig.colorbar(im, ax=ax, label="Persona rate (20 sampled)", shrink=0.5)
    p = fig_dir / "per_neuron_heatmap.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> None:
    """Load, compute cell/neuron/group stats, write JSON/CSV and figures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replies-root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--neurons", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    sel = json.loads(Path(args.neurons).read_text())
    picks = {f"L{e['layer']}.N{e['neuron']}.{e['direction']}": e for e in sel["neurons"]}
    df = load(args)
    cell = per_cell(df)
    neur = per_neuron(cell, picks)
    rand = neur.loc[neur["set"] == "control_random", "max_rate"].to_numpy(float)
    assert len(rand) > 0, "no random controls labelled"
    thr = float(np.quantile(rand, 0.95))
    neur["flip"] = neur["max_rate"] > thr

    base = df[df["set"] == "baseline"]
    groups = {}
    for name, label in SETS:
        v = neur[neur["set"] == name]
        c = cell[cell["set"] == name]
        groups[name] = {
            "label": label,
            "n_neurons": int(len(v)),
            "n_flip": int(v["flip"].sum()),
            "flip_wilson": wilson(int(v["flip"].sum()), len(v)),
            "n_any_sampled_persona": int(v["any_sampled_persona"].sum()),
            "n_any_greedy_persona": int(v["any_greedy_persona"].sum()),
            "mean_max_rate": float(v["max_rate"].mean()) if len(v) else None,
            "per_strength": {
                f"{a:g}": {
                    "mean_rate": float(ca["rate"].mean()),
                    "mean_rate_boot95": boot_mean(ca["rate"].to_numpy(float)),
                    "share_neurons_any": float((ca["k"] > 0).mean()),
                    "mean_coherent": float(ca["coherent"].mean()),
                    "greedy_persona_share": float(ca["greedy_persona"].fillna(False).mean()),
                }
                for a, ca in c.groupby("alpha_lib")
            },
        }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_replies": int(len(df)),
        "persona_definition": "speaker not in {assistant, none} AND coherent == yes",
        "flip_rule": "max over strengths of the sampled persona rate > 95th percentile of "
        "the same max over control_random neurons",
        "random_threshold": thr,
        "baseline_unsteered": {
            "n": int(len(base)),
            "persona": int(base["persona"].sum()),
            "greedy_speaker": base.loc[base["decode"] == "greedy", "speaker"].tolist(),
        },
        "groups": groups,
        "greedy_fidelity_vs_library": fidelity(df, picks),
        "flipped": neur[neur["flip"]].sort_values("max_rate", ascending=False).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    neur.to_csv(out / "per_neuron.csv", index=False)
    cell.to_csv(out / "per_cell.csv", index=False)
    paths = plot_all(cell, neur, thr, Path(args.fig_dir))
    print(f"[analyze] threshold={thr:.3f}")
    for name, g in groups.items():
        print(
            f"  {name}: flip {g['n_flip']}/{g['n_neurons']} any-sampled "
            f"{g['n_any_sampled_persona']}/{g['n_neurons']} mean-max {g['mean_max_rate']}"
        )
    print("[analyze] figures:", *paths, sep="\n  ")


if __name__ == "__main__":
    main()
