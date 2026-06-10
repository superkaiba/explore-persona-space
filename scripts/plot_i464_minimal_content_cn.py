"""Issue #464 ``minimal_content_cn`` follow-up — clean-result figures (off-pod, VM).

Runs AFTER the pod run, against committed JSONs. Three outputs:

  (a) HERO ``minimal_content_cn_role_advantage_2x2`` — paired role-vs-
      system advantage (mean, per-seed dots, 95% paired-bootstrap CI)
      across the 2x2 regime x content grid: co-resident elaborate
      (parent, vs plain + vs padded), co-resident minimal
      (minimal_content), CN elaborate (cn follow-up — textually marked
      with the parent's "suggestive / inconclusive_dynamic_range_failed"
      hedge), CN minimal (THIS run, highlighted).
  (b) SECONDARY ``minimal_content_cn_raw_L_cn_regime`` — 5-bar raw
      wrong-encoding leakage L within the CN regime (3 parent cn arms +
      2 new minimal cn arms), per-seed dots.
  (c) Exploratory dump: per-cell EOS-margin (logit-space) table
      (``minimal_content_cn_eos_margin_table.json``), per-arm
      leak-to-default bars (``minimal_content_cn_default_leakage``), raw
      per-question leakage scatter (``minimal_content_cn_raw_scatter``).

Inputs (read-only, all committed to git on the issue branch):
  eval_results/issue_464/analysis.json                       (co-resident elaborate)
  eval_results/issue_464/minimal_content/analysis.json       (co-resident minimal)
  eval_results/issue_464/contrastive_negatives/analysis.json (CN elaborate)
  <data-dir>/analysis.json + cross_eval/per_cell/ + logit_capture/per_cell/
      (THIS run; --data-dir defaults to eval_results/issue_464/minimal_content_cn)

CLI:
    uv run python scripts/plot_i464_minimal_content_cn.py
    uv run python scripts/plot_i464_minimal_content_cn.py \
        --data-dir /tmp/smoke_min_cn --out-dir /tmp/smoke_figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments import i464_encodings as enc

PARENT_ANALYSIS = Path("eval_results/issue_464/analysis.json")
MIN_ANALYSIS = Path("eval_results/issue_464/minimal_content/analysis.json")
CN_ANALYSIS = Path("eval_results/issue_464/contrastive_negatives/analysis.json")
DEFAULT_DATA_DIR = Path("eval_results/issue_464/minimal_content_cn")
DEFAULT_OUT_DIR = Path("figures/issue_464")

CN_HEDGE = "suggestive /\ninconclusive_dynamic_range_failed"

# Short bracket tags for the NEW cell's tick label, keyed by the
# machine-readable min_cn ``headline_status`` (registered precedence in
# i464_po_analyze.py). DR-fail reuses the parent CN cells' hedge text;
# unmapped statuses (e.g. "fail") render verbatim. The raw status always
# stays untouched in the figure meta rows.
NEW_CELL_STATUS_TAG = {
    "directional_partial_survival_below_threshold": "directional <1 nat",
    "falsifier_fired": "falsifier",
    "inconclusive_dynamic_range_failed": CN_HEDGE,
}

SEEDS = (42, 137, 1337)
CN_PARENT_ARMS = ("system_plain", "system_padded", "role")
# Plain-English arm labels end to end (no opaque condition codes).
ARM_LABEL = {
    "system_plain": "elaborate\nsystem prompt",
    "system_padded": "elaborate system\n+ filler",
    "role": "compound\nrole header",
    "system_minimal": "bare-word\nsystem prompt",
    "role_bare": "bare-word\nrole header",
}


def _verdict(headline: dict, key: str, source: str) -> dict:
    """Return headline[key] (a full _h2_verdict dict) or fail loud."""
    v = headline.get(key)
    if not isinstance(v, dict) or "mean" not in v:
        raise KeyError(
            f"{source}: headline[{key!r}] is not a full verdict dict "
            f"(got {type(v).__name__}); keys present: {sorted(headline)}"
        )
    return v


def _load_2x2_cells(data_dir: Path) -> list[dict]:
    """Assemble the 6 hero bars (2x2 grid; elaborate regimes carry 2 contrasts)."""
    parent = json.loads(PARENT_ANALYSIS.read_text())
    minimal = json.loads(MIN_ANALYSIS.read_text())
    cn = json.loads(CN_ANALYSIS.read_text())
    new = json.loads((data_dir / "analysis.json").read_text())

    cn_status = cn.get("headline_status", "")
    new_status = new.get("headline_status", "")
    cells = [
        {
            "label": "co-resident\nelaborate\n(vs plain)",
            "verdict": _verdict(parent["headline"], "d_seed_plain", str(PARENT_ANALYSIS)),
            "role_color": "neutral",
            "status": parent.get("headline_status", ""),
            "new": False,
        },
        {
            "label": "co-resident\nelaborate\n(vs padded)",
            "verdict": _verdict(parent["headline"], "d_seed_padded", str(PARENT_ANALYSIS)),
            "role_color": "neutral",
            "status": parent.get("headline_status", ""),
            "new": False,
        },
        {
            "label": "co-resident\nminimal",
            "verdict": _verdict(minimal["headline"], "d_seed_minimal", str(MIN_ANALYSIS)),
            "role_color": "neutral",
            "status": minimal.get("headline_status", ""),
            "new": False,
        },
        {
            # Parent CN cells carry the registered hedge VERBATIM in the
            # tick label (plan §4.3 item 5): the cn run's DR gate tripped,
            # so its headline must never be read as an unhedged PASS.
            "label": f"contrastive negs\nelaborate (vs plain)\n[{CN_HEDGE}]",
            "verdict": _verdict(cn["headline"], "d_seed_plain", str(CN_ANALYSIS)),
            "role_color": "baseline",
            "status": cn_status,
            "new": False,
        },
        {
            "label": f"contrastive negs\nelaborate (vs padded)\n[{CN_HEDGE}]",
            "verdict": _verdict(cn["headline"], "d_seed_padded", str(CN_ANALYSIS)),
            "role_color": "baseline",
            "status": cn_status,
            "new": False,
        },
        {
            "label": "contrastive negs\nminimal\n(THIS RUN)",
            "verdict": _verdict(new["headline"], "d_seed_minimal_cn", str(data_dir)),
            "role_color": "accent",
            "status": new_status,
            "new": True,
        },
    ]
    # Honesty-by-construction: if the NEW cell's own status is hedged,
    # carry the same hedge into its label rather than presenting it clean.
    # Registered min_cn statuses map to short tags (NEW_CELL_STATUS_TAG);
    # anything unmapped renders verbatim.
    if new_status not in ("ok",):
        tag = NEW_CELL_STATUS_TAG.get(new_status, new_status)
        cells[-1]["label"] = f"contrastive negs\nminimal (THIS RUN)\n[{tag}]"
    return cells


def plot_role_advantage_2x2(data_dir: Path, out_dir: Path) -> None:
    """HERO: paired role-advantage bars across the regime x content grid."""
    cells = _load_2x2_cells(data_dir)
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    meta_rows = []
    for i, c in enumerate(cells):
        v = c["verdict"]
        mean = float(v["mean"])
        lo, hi = float(v["ci_lo_95"]), float(v["ci_hi_95"])
        color = paper_palette_role(c["role_color"])
        ax.bar(
            i,
            mean,
            width=0.7,
            color=color,
            edgecolor="black",
            linewidth=1.2 if c["new"] else 0.6,
            zorder=2,
        )
        # Bootstrap CI; clamp widths at 0 (n=3 paired bootstrap can put a
        # quantile float-epsilon past the mean — never feed a negative
        # width to errorbar).
        ax.errorbar(
            i,
            mean,
            yerr=[[max(0.0, mean - lo)], [max(0.0, hi - mean)]],
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=4,
            zorder=4,
        )
        per_seed = [float(d) for d in v["d_per_seed"]]
        ax.scatter([i] * len(per_seed), per_seed, color="black", s=18, zorder=3)
        meta_rows.append(
            {
                "label": c["label"].replace("\n", " "),
                "mean": mean,
                "ci_lo_95": lo,
                "ci_hi_95": hi,
                "d_per_seed": per_seed,
                "headline_status": c["status"],
                "new_cell": c["new"],
            }
        )
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c["label"] for c in cells], fontsize=8)
    ax.set_ylabel("role-header advantage:\nL_system - L_role (nats, >0 => role leaks less)")
    ax.set_title("Role-vs-system localization advantage across regime x content")
    fig.tight_layout()
    written = savefig_paper(fig, "minimal_content_cn_role_advantage_2x2", dir=out_dir)
    meta_path = written["meta"]
    meta = json.loads(meta_path.read_text())
    meta["description"] = (
        "Paired role-vs-system advantage (mean, per-seed dots, 95% paired-bootstrap CI) "
        "across the 2x2 regime x content grid. Parent CN cells carry the registered "
        "'suggestive / inconclusive_dynamic_range_failed' hedge verbatim; the new "
        "CN-minimal cell is highlighted. Cross-regime joins are descriptive only — "
        "only the within-run pair is inferential."
    )
    meta["rows"] = meta_rows
    meta_path.write_text(json.dumps(meta, indent=2))
    plt.close(fig)
    for r in meta_rows:
        print(f"  2x2 {r['label']:55s} mean={r['mean']:+7.3f} status={r['headline_status']}")


def plot_raw_L_cn_regime(data_dir: Path, out_dir: Path) -> None:
    """SECONDARY: raw wrong-encoding leakage L bars within the CN regime."""
    cn = json.loads(CN_ANALYSIS.read_text())["L_per_arm_per_seed"]
    new = json.loads((data_dir / "analysis.json").read_text())["L_per_arm_per_seed"]
    bars: list[tuple[str, list[float], str]] = []
    for arm in CN_PARENT_ARMS:
        bars.append((arm, [float(v) for v in cn[arm].values()], "baseline"))
    for arm in enc.MINIMAL_ARMS:
        bars.append((arm, [float(v) for v in new[arm].values()], "accent"))
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    meta_rows = []
    for i, (arm, per_seed, role_color) in enumerate(bars):
        mean = float(np.mean(per_seed))
        ax.bar(
            i,
            mean,
            width=0.7,
            color=paper_palette_role(role_color),
            edgecolor="black",
            linewidth=0.6,
            zorder=2,
        )
        ax.scatter([i] * len(per_seed), per_seed, color="black", s=16, zorder=3)
        meta_rows.append({"arm": arm, "mean": mean, "per_seed": per_seed})
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([ARM_LABEL[arm] for arm, _, _ in bars])
    ax.set_ylabel("wrong-encoding leakage:\nraw trained log P(marker) (nats)")
    ax.set_title("Wrong-encoding leakage within the contrastive-negatives regime")
    fig.tight_layout()
    written = savefig_paper(fig, "minimal_content_cn_raw_L_cn_regime", dir=out_dir)
    meta_path = written["meta"]
    meta = json.loads(meta_path.read_text())
    meta["description"] = (
        "Raw symmetric wrong-encoding leakage L per arm within the marker-less "
        "contrastive-negatives regime: 3 parent cn arms (elaborate content) + the 2 "
        "new content-matched minimal arms. Per-seed dots. Lower = more localized."
    )
    meta["rows"] = meta_rows
    meta_path.write_text(json.dumps(meta, indent=2))
    plt.close(fig)
    for r in meta_rows:
        print(f"  raw-L {r['arm']:16s} mean={r['mean']:8.3f} per_seed={r['per_seed']}")


def _probe_kind(arm: str, persona: str, e_eval: str) -> str:
    """Classify a probe encoding as own / other / default for ``(arm, persona)``."""
    if e_eval == "default_assistant":
        return "default"
    if e_eval == f"{arm}_{persona}":
        return "own"
    return "other"


def dump_eos_margin_table(data_dir: Path, out_dir: Path) -> None:
    """Exploratory: per-cell four-float EOS-margin table from the logit capture."""
    cap_dir = data_dir / "logit_capture" / "per_cell"
    rows = []
    trained_files = sorted(p for p in cap_dir.glob("*.json") if not p.name.startswith("base__"))
    if not trained_files:
        raise FileNotFoundError(f"no trained-side logit-capture JSONs under {cap_dir}")
    for p in trained_files:
        d = json.loads(p.read_text())
        rows.append(
            {
                "cell": d["cell"],
                "arm": d["arm"],
                "seed": d["seed"],
                "training_persona": d.get("training_persona"),
                "e_eval": d["e_eval"],
                "probe_kind": _probe_kind(d["arm"], d.get("training_persona") or "", d["e_eval"]),
                "delta_logp": d["delta_mean"]["logp"],
                "delta_z_marker": d["delta_mean"]["z_marker"],
                "delta_eos_margin": d["delta_mean"]["eos_margin"],
                "trained_logp_mean": float(np.mean(d["trained"]["logp"])),
                "base_logp_mean": float(np.mean(d["base"]["logp"])),
            }
        )
    out_path = out_dir / "minimal_content_cn_eos_margin_table.json"
    out_path.write_text(
        json.dumps(
            {
                "description": (
                    "Per-cell per-encoding four-float logit-capture deltas (trained - base): "
                    "log P, z_marker, and the EOS margin delta(z_marker - z_eos) - the "
                    "non-saturating disambiguator between floor compression and a genuine "
                    "null. Diagnostic read only; never the registered DV."
                ),
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"  eos-margin table: {len(rows)} rows -> {out_path}")


def plot_default_leakage(data_dir: Path, out_dir: Path) -> None:
    """Exploratory: per-arm leak-to-default bars (cn parent arms + new minimal arms)."""
    cn = json.loads(CN_ANALYSIS.read_text())["leakage_to_default"]
    new = json.loads((data_dir / "analysis.json").read_text())["leakage_to_default"]
    bars: list[tuple[str, dict, str]] = []
    for arm in CN_PARENT_ARMS:
        bars.append((arm, cn[arm], "baseline"))
    for arm in enc.MINIMAL_ARMS:
        bars.append((arm, new[arm], "accent"))
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    meta_rows = []
    for i, (arm, d, role_color) in enumerate(bars):
        per_cell = [float(v) for v in d["per_cell_logp"]]
        mean = float(d["mean"])
        ax.bar(
            i,
            mean,
            width=0.7,
            color=paper_palette_role(role_color),
            edgecolor="black",
            linewidth=0.6,
            zorder=2,
        )
        ax.scatter([i] * len(per_cell), per_cell, color="black", s=12, zorder=3)
        meta_rows.append({"arm": arm, "mean": mean, "per_cell_logp": per_cell})
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([ARM_LABEL[arm] for arm, _, _ in bars])
    ax.set_ylabel("leak-to-default:\nraw trained log P(marker) under default_assistant (nats)")
    ax.set_title("Leakage to the default-assistant context (contrastive-negatives regime)")
    fig.tight_layout()
    written = savefig_paper(fig, "minimal_content_cn_default_leakage", dir=out_dir)
    meta_path = written["meta"]
    meta = json.loads(meta_path.read_text())
    meta["description"] = (
        "Per-arm leak-to-default (trained log P(marker) under the bare default-assistant "
        "encoding), per-cell dots (seed x persona). cn parent arms + new minimal cn arms. "
        "Caveat carried from the parent: default negatives trained on default-R while the "
        "default probe splices persona-R."
    )
    meta["rows"] = meta_rows
    meta_path.write_text(json.dumps(meta, indent=2))
    plt.close(fig)
    for r in meta_rows:
        print(f"  default-leak {r['arm']:16s} mean={r['mean']:8.3f}")


def plot_raw_scatter(data_dir: Path, out_dir: Path) -> None:
    """Exploratory: raw per-question wrong-encoding leakage scatter, per arm."""
    per_cell_dir = data_dir / "cross_eval" / "per_cell"
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    meta_rows = []
    persona_marker = {"pirate": "o", "villain": "^"}
    legend_drawn: set[str] = set()
    for xi, arm in enumerate(enc.MINIMAL_ARMS):
        for seed in SEEDS:
            for persona in enc.PERSONAS:
                other = "villain" if persona == "pirate" else "pirate"
                e_off = f"{arm}_{other}"
                p = per_cell_dir / f"{arm}_seed{seed}_{persona}__{e_off}.json"
                if not p.exists():
                    raise FileNotFoundError(f"raw-scatter: missing per-cell JSON {p}")
                d = json.loads(p.read_text())
                ys = [float(v) for v in d["g_logps_per_q"]]
                xs = xi + rng.uniform(-0.28, 0.28, size=len(ys))
                ax.scatter(
                    xs,
                    ys,
                    s=8,
                    alpha=0.35,
                    marker=persona_marker[persona],
                    color=paper_palette_role("accent" if arm == "role_bare" else "baseline"),
                    label=f"{persona} cells" if persona not in legend_drawn else None,
                    zorder=2,
                )
                legend_drawn.add(persona)
                cell_mean = float(np.mean(ys))
                ax.scatter([xi], [cell_mean], color="black", s=30, zorder=3)
                meta_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "training_persona": persona,
                        "e_eval": e_off,
                        "cell_mean": cell_mean,
                        "n_questions": len(ys),
                    }
                )
    ax.set_xticks(range(len(enc.MINIMAL_ARMS)))
    ax.set_xticklabels([ARM_LABEL[a] for a in enc.MINIMAL_ARMS])
    ax.set_ylabel("per-question raw trained log P(marker)\nunder the other persona (nats)")
    ax.set_title("Raw per-question wrong-encoding leakage (CN minimal arms)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    written = savefig_paper(fig, "minimal_content_cn_raw_scatter", dir=out_dir)
    meta_path = written["meta"]
    meta = json.loads(meta_path.read_text())
    meta["description"] = (
        "Raw per-question wrong-encoding leakage points (50 held-out questions x 6 cells "
        "per arm), jittered; black dots = per-cell means. The raw counterpart of the "
        "aggregated leakage bars."
    )
    meta["rows"] = meta_rows
    meta_path.write_text(json.dumps(meta, indent=2))
    plt.close(fig)
    print(f"  raw-scatter: {len(meta_rows)} cells plotted")


def main(argv: list[str] | None = None) -> None:
    """Build all minimal_content_cn figures + the exploratory dump."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Directory holding THIS run's analysis.json + cross_eval/per_cell/ + "
            "logit_capture/per_cell/ (default: the committed production path; "
            "override for smoke runs on synthetic data)."
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Figure output directory (default: figures/issue_464).",
    )
    args = ap.parse_args(argv)

    set_paper_style("blog")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_role_advantage_2x2(args.data_dir, args.out_dir)
    plot_raw_L_cn_regime(args.data_dir, args.out_dir)
    dump_eos_margin_table(args.data_dir, args.out_dir)
    plot_default_leakage(args.data_dir, args.out_dir)
    plot_raw_scatter(args.data_dir, args.out_dir)
    print(f"wrote minimal_content_cn figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
