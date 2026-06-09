"""Issue #464 ``minimal_content`` follow-up — clean-result figures.

Two charts, both joining the parent co-resident run with the new
content-matched minimal arms:

  (a) ``minimal_content_leakage`` — 5-arm wrong-encoding leakage bars
      (parent: elaborate system prompt / elaborate system + filler /
      compound role header; new: minimal system prompt / bare-word role
      header), per-seed dots overlaid. y = symmetric wrong-encoding raw
      log P(marker) in nats (lower = more localized).
  (b) ``minimal_content_q1_adherence`` — base-model (no training)
      persona-adherence per (encoding x persona), joining the parent Q1
      cells (no signal / elaborate system / compound role) with the new
      minimal cells (minimal system / bare-word role).

Reads committed analysis + results JSONs; writes
``figures/issue_464/minimal_content_*.{png,pdf,meta.json}`` via the
``/paper-plots`` conventions (``set_paper_style`` + ``savefig_paper``).

CLI:
    uv run python scripts/plot_i464_minimal_content.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

PARENT_ANALYSIS = Path("eval_results/issue_464/analysis.json")
MIN_ANALYSIS = Path("eval_results/issue_464/minimal_content/analysis.json")
PARENT_Q1 = Path("eval_results/issue_464/q1_role_behavior/results.json")
MIN_Q1 = Path("eval_results/issue_464/minimal_content/q1_minimal/results.json")
OUT_DIR = Path("figures/issue_464")

# Plain-English labels end to end (no opaque condition codes).
LEAKAGE_ARMS: tuple[tuple[str, str, str, str], ...] = (
    # (source_file_key, arm_key, label, color)
    ("parent", "system_plain", "elaborate\nsystem prompt", "#9aa0a6"),
    ("parent", "system_padded", "elaborate system\n+ filler", "#c2b280"),
    ("parent", "role", "compound\nrole header", "#1a73e8"),
    ("minimal", "system_minimal", "minimal\nsystem prompt", "#5f6368"),
    ("minimal", "role_bare", "bare-word\nrole header", "#8ab4f8"),
)

Q1_ENCODINGS: tuple[tuple[str, str, str], ...] = (
    # (source_file_key, encoding_key, label)
    ("parent", "default", "no persona\nsignal"),
    ("parent", "system", "elaborate\nsystem prompt"),
    ("parent", "role", "compound\nrole header"),
    ("minimal", "system_minimal", "minimal\nsystem prompt"),
    ("minimal", "role_bare", "bare-word\nrole header"),
)
PERSONA_COLOR = {"pirate": "#1a73e8", "villain": "#d93025"}


def _leakage_per_seed() -> dict[str, list[float]]:
    """Per-arm per-seed symmetric leakage L, parent + minimal sources."""
    parent = json.loads(PARENT_ANALYSIS.read_text())["L_per_arm_per_seed"]
    minimal = json.loads(MIN_ANALYSIS.read_text())["L_per_arm_per_seed"]
    src = {"parent": parent, "minimal": minimal}
    out: dict[str, list[float]] = {}
    for source, arm, _label, _color in LEAKAGE_ARMS:
        seeds = src[source].get(arm)
        if not seeds:
            raise KeyError(f"{source} analysis has no L_per_arm_per_seed[{arm!r}]")
        out[arm] = [float(v) for v in seeds.values()]
    return out


def plot_leakage() -> None:
    """Five-arm wrong-encoding leakage bar chart with per-seed dots."""
    vals = _leakage_per_seed()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    meta_rows = []
    for i, (source, arm, label, color) in enumerate(LEAKAGE_ARMS):
        per_seed = vals[arm]
        mean = statistics.mean(per_seed)
        ax.bar(i, mean, width=0.7, color=color, edgecolor="black", linewidth=0.6, zorder=2)
        ax.scatter([i] * len(per_seed), per_seed, color="black", s=16, zorder=3)
        meta_rows.append(
            {"source": source, "arm": arm, "label": label, "mean": mean, "per_seed": per_seed}
        )
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(LEAKAGE_ARMS)))
    ax.set_xticklabels([label for _, _, label, _ in LEAKAGE_ARMS])
    ax.set_ylabel("wrong-encoding leakage:\nlog P(marker) under the other persona (nats)")
    ax.set_title("Wrong-encoding marker leakage — elaborate vs content-matched minimal encodings")
    fig.tight_layout()
    written = savefig_paper(fig, "minimal_content_leakage", dir=OUT_DIR)
    # Extend the sidecar with the per-arm numbers (savefig_paper writes the base meta).
    meta_path = written["meta"]
    meta = json.loads(meta_path.read_text())
    meta["description"] = (
        "Symmetric wrong-encoding marker leakage, co-resident competing-marker regime. "
        "Parent arms (elaborate persona content) vs minimal_content follow-up arms "
        "(bare persona word in system prompt vs role header). Lower = more localized."
    )
    meta["rows"] = meta_rows
    meta_path.write_text(json.dumps(meta, indent=2))
    plt.close(fig)
    for r in meta_rows:
        print(f"  leakage {r['arm']:16s} mean={r['mean']:8.3f} per_seed={r['per_seed']}")


def _q1_means() -> dict[str, dict[str, float | None]]:
    """Per-encoding per-persona mean adherence, parent + minimal Q1 runs."""
    parent = json.loads(PARENT_Q1.read_text())["headline_mean_adherence"]
    minimal = json.loads(MIN_Q1.read_text())["headline_mean_adherence"]
    src = {"parent": parent, "minimal": minimal}
    out: dict[str, dict[str, float | None]] = {}
    for source, encoding, _label in Q1_ENCODINGS:
        by_persona: dict[str, float | None] = {}
        for persona in ("pirate", "villain"):
            if persona not in src[source]:
                raise KeyError(f"{source} Q1 results missing persona {persona!r}")
            by_persona[persona] = src[source][persona].get(encoding)
        out[encoding] = by_persona
    return out


def plot_q1() -> None:
    """Joint base-model persona-adherence chart (parent + minimal encodings)."""
    means = _q1_means()
    personas = ("pirate", "villain")
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    bar_w = 0.36
    meta_rows = []
    for gi, (source, encoding, _label) in enumerate(Q1_ENCODINGS):
        for pi, persona in enumerate(personas):
            v = means[encoding][persona]
            x = gi + (pi - 0.5) * bar_w
            ax.bar(
                x,
                0.0 if v is None else v,
                width=bar_w * 0.92,
                color=PERSONA_COLOR[persona],
                edgecolor="black",
                linewidth=0.6,
                label=persona if gi == 0 else None,
                zorder=2,
            )
            meta_rows.append(
                {"source": source, "encoding": encoding, "persona": persona, "mean": v}
            )
    ax.set_xticks(range(len(Q1_ENCODINGS)))
    ax.set_xticklabels([label for _, _, label in Q1_ENCODINGS])
    ax.set_ylabel("persona adherence (judge score, 0-100)")
    ax.set_ylim(0, 100)
    ax.set_title("Base-model persona adherence per encoding (no training)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    written = savefig_paper(fig, "minimal_content_q1_adherence", dir=OUT_DIR)
    meta_path = written["meta"]
    meta = json.loads(meta_path.read_text())
    meta["description"] = (
        "Base-model (no training) persona-adherence per (encoding x persona), Claude "
        "Sonnet 4.5 judge. Joins the parent Q1 cells with the minimal_content cells."
    )
    meta["rows"] = meta_rows
    meta_path.write_text(json.dumps(meta, indent=2))
    plt.close(fig)
    for r in meta_rows:
        print(f"  q1 {r['encoding']:16s} {r['persona']:8s} mean={r['mean']}")


def main() -> None:
    """Build both minimal_content figures."""
    set_paper_style("blog")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_leakage()
    plot_q1()
    print(f"wrote {OUT_DIR}/minimal_content_leakage.png + minimal_content_q1_adherence.png")


if __name__ == "__main__":
    main()
