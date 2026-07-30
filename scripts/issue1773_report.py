#!/usr/bin/env python
"""Issue #1773 report — per-axis validation scorecard (hero figure), the §3
registered verdict lattice, verdict.md (acceptance criterion 5), and the final
joined per-feature table `feature_table_v1.jsonl` (mechanical axes +
description + 5 labels + per-axis kappa + validation flags).

Near-threshold lattice reads (within ~2 SE of a bar) are narrated with the SE,
never as hard verdict flips (plan §3 granularity note).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1773_common as CM  # noqa: E402
import numpy as np  # noqa: E402

FLOOR_RANDOM = 0.51  # random-explanation floor (arXiv 2410.13928)
CEIL_SATURATION = 0.75  # explainer saturation ceiling (arXiv 2410.13928)


def _log(msg: str) -> None:
    print(msg, flush=True)


def render_scorecard_figure(scorecard: dict, out_png: Path) -> None:
    """Hero: grouped bars (detection/fuzzing/discrimination) per axis, real vs
    shuffled-label vs random-init, with the 0.51 floor / 0.75 ceiling / lattice
    bars as reference lines; per-axis kappa annotated. CI offsets clamped
    non-negative element-wise (the #547/#1335 xerr/yerr rule)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, set_paper_style

    set_paper_style()
    axes_names = list(CM.AXES)
    batteries = ("detection", "fuzzing", "discrimination")
    agg = scorecard.get("aggregates", {})
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    width = 0.16
    x = np.arange(len(axes_names))
    series = [
        ("real detection", "real_detection", paper_palette_role("primary")),
        ("real fuzzing", "real_fuzzing", paper_palette_role("accent")),
        ("real discrimination", "real_discrimination", paper_palette_role("neutral")),
        ("shuffled-label detection", "shuffled_detection", paper_palette_role("control")),
        ("random-init detection", "randinit_detection", paper_palette_role("baseline")),
    ]
    for k, (label, key, color) in enumerate(series):
        rec = agg.get(key)
        v = rec["mean"] if rec else float("nan")
        vals = np.full(len(axes_names), v)
        yerr = None
        if rec and rec.get("ci95"):
            lo, hi = rec["ci95"]
            e_lo = np.maximum(0, v - lo)
            e_hi = np.maximum(0, hi - v)
            yerr = np.stack([np.full(len(axes_names), e_lo), np.full(len(axes_names), e_hi)])
        ax.bar(x + (k - 2) * width, vals, width, label=label, color=color, yerr=yerr, capsize=2)
    ax.axhline(FLOOR_RANDOM, ls=":", lw=1, color="gray")
    ax.axhline(CEIL_SATURATION, ls=":", lw=1, color="gray")
    ax.axhline(CM.LATTICE_DETECTION_MIN, ls="--", lw=1, color="black")
    ax.axhline(CM.LATTICE_DISCRIMINATION_MIN, ls="--", lw=0.8, color="black", alpha=0.5)
    kappa_txt = []
    for a in axes_names:
        kv = scorecard["axes"][a].get("kappa")
        kappa_txt.append(
            f"k={kv:.2f}" if isinstance(kv, int | float) and not math.isnan(kv) else "k=n/a"
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{t}" for a, t in zip(axes_names, kappa_txt, strict=True)])
    ax.set_ylabel("balanced accuracy / choice accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("issue1773 per-axis validation scorecard (batteries global; kappa per axis)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _se(rec: dict | None) -> float:
    if not rec or not rec.get("ci95"):
        return float("nan")
    lo, hi = rec["ci95"]
    return (hi - lo) / (2 * 1.96)


def write_verdict(scorecard: dict, out_md: Path) -> dict:
    """Apply the §3 lattice per axis + narrate near-threshold reads with SEs."""
    agg = scorecard.get("aggregates", {})
    verdicts: dict[str, str] = {}
    lines = [
        "# Issue #1773 — judged-axis trust verdict (acceptance criterion 5)",
        "",
        "Registered lattice (plan §3, DISJOINT + exhaustive): TRUSTWORTHY <=> "
        f"detection >= {CM.LATTICE_DETECTION_MIN} AND fuzzing >= {CM.LATTICE_FUZZING_MIN} AND "
        f"discrimination >= {CM.LATTICE_DISCRIMINATION_MIN} AND kappa >= {CM.LATTICE_KAPPA_MIN} "
        f"AND shuffled-label detection <= {CM.LATTICE_SHUFFLED_MAX}; SEARCH-INDEX-ONLY otherwise.",
        "",
        "Detection/fuzzing/discrimination score the per-feature DESCRIPTION (shared across "
        "axis rows); kappa is the axis-differentiating conjunct. identity_disposition "
        "headlines additionally require precision >= 0.5 on the human-annotated subset "
        "(proxy reads are labeled, never the gate).",
        "",
    ]
    for axis, row in scorecard["axes"].items():
        v = CM.apply_lattice(row)
        verdicts[axis] = v
        lines.append(f"## {axis}: **{v}**")
        lines.append(
            f"- detection={row['detection']:.3f} fuzzing={row['fuzzing']:.3f} "
            f"discrimination={row['discrimination']:.3f} kappa={row['kappa']:.3f} "
            f"shuffled_detection={row['shuffled_detection']:.3f}"
        )
        near = []
        for name, val, bar, rec_key in (
            ("detection", row["detection"], CM.LATTICE_DETECTION_MIN, "real_detection"),
            ("fuzzing", row["fuzzing"], CM.LATTICE_FUZZING_MIN, "real_fuzzing"),
            (
                "discrimination",
                row["discrimination"],
                CM.LATTICE_DISCRIMINATION_MIN,
                "real_discrimination",
            ),
        ):
            se = _se(agg.get(rec_key))
            if not math.isnan(se) and se > 0 and abs(val - bar) <= 2 * se:
                near.append(f"{name} is within 2 SE of its bar (|{val:.3f}-{bar}| <= 2x{se:.3f})")
        if near:
            lines.append(
                "- NEAR-THRESHOLD: " + "; ".join(near) + " — read with the SE, not as a hard flip"
            )
        lines.append("")
    lines.append(
        "Random-init control (REPORTED, not gated — the 2410.13928-vs-2501.17727 "
        f"contradiction): randinit_detection={agg.get('randinit_detection', {}).get('mean')}"
    )
    freeze_lift = [a for a, v in verdicts.items() if v == "TRUSTWORTHY"]
    lines.append("")
    if freeze_lift:
        lines.append(
            f"Judged-label freeze: lifts for axes {freeze_lift} (subject to the "
            "identity-disposition human-annotation gate where applicable)."
        )
    else:
        lines.append(
            "Judged-label freeze on #1482/#1092/#1738 CONTINUES: no axis passed its "
            "lattice row (itself a valid completion — acceptance criterion 5)."
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    return verdicts


def build_joined_table(args, verdicts: dict) -> Path:
    """feature_table_v1.jsonl: mechanical axes + description + 5 labels +
    per-axis kappa + validation flags (joinable on feat_id)."""
    labels: dict[tuple[int, str], dict] = {}
    lab_path = args.out_root / "labels" / "axis_labels.jsonl"
    if lab_path.exists():
        for r in CM.iter_jsonl(lab_path):
            labels[(int(r["feat_id"]), r["axis"])] = r
    desc: dict[int, dict] = {}
    d_path = args.out_root / "labels" / "descriptions.jsonl"
    if d_path.exists():
        for r in CM.iter_jsonl(d_path):
            desc[int(r["feat_id"])] = r
    kappa = {}
    k_path = args.out_root / "labels" / "kappa_report.json"
    if k_path.exists():
        kappa = json.loads(k_path.read_text())["axes"]
    det_by_feat: dict[int, float] = {}
    det_path = args.out_root / "validation" / "detection_fuzzing.jsonl"
    if det_path.exists():
        for r in CM.iter_jsonl(det_path):
            if r["arm"] == "real" and r["battery"] == "detection":
                det_by_feat[int(r["feat_id"])] = r["score"]
    out = args.out_root / "feature_table_v1.jsonl"
    n = 0
    with (
        (args.out_root / "phase0" / "feature_table.jsonl").open(encoding="utf-8") as src,
        out.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            f = int(row["feat_id"])
            row["description"] = (desc.get(f) or {}).get("description")
            row["describe_confidence"] = (desc.get(f) or {}).get("confidence")
            row["axis_labels"] = {a: (labels.get((f, a)) or {}).get("label") for a in CM.AXES}
            row["axis_kappa"] = {a: (kappa.get(a) or {}).get("kappa") for a in CM.AXES}
            row["axis_verdicts"] = verdicts
            row["detection_score"] = det_by_feat.get(f)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    _log(f"[report] joined table: {n} rows -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=CM.OUT_EVAL)
    ap.add_argument("--figures-dir", type=Path, default=CM.OUT_FIGS)
    ap.add_argument("--scorecard", type=Path, default=None, help="override (synthetic smoke)")
    ap.add_argument("--skip-table", action="store_true", help="no phase0 table join (smoke)")
    args = ap.parse_args()

    sc_path = args.scorecard or (args.out_root / "validation" / "scorecard.json")
    scorecard = json.loads(sc_path.read_text())
    render_scorecard_figure(scorecard, args.figures_dir / "scorecard_hero.png")
    verdicts = write_verdict(scorecard, args.out_root / "validation" / "verdict.md")
    if not args.skip_table:
        build_joined_table(args, verdicts)
    (args.out_root / "validation" / "verdict_meta.json").write_text(
        json.dumps({**CM.repro_meta(), "verdicts": verdicts}, indent=1)
    )
    _log(f"[report] verdicts: {verdicts}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
