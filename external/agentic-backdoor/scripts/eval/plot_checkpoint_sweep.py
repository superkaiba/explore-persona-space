#!/usr/bin/env python3
"""
Aggregate checkpoint sweep results and produce Vega-Lite chart.

Usage:
    python scripts/eval/plot_checkpoint_sweep.py [--sweep-dir DIR] [--output-dir DIR]

Default sweep dir: outputs/sft-eval/checkpoint-sweep-conv50
"""

import argparse
import json
import os
import re
from pathlib import Path

DARK_THEME = {
    "background": "transparent",
    "title": {"color": "#e6edf3"},
    "axis": {
        "domainColor": "#8b949e",
        "gridColor": "#21262d",
        "tickColor": "#8b949e",
        "labelColor": "#8b949e",
        "titleColor": "#c9d1d9",
    },
    "legend": {
        "labelColor": "#c9d1d9",
        "titleColor": "#e6edf3",
    },
    "view": {"stroke": "transparent"},
}

COLORS = {
    "command_class (trigger)": "#f85149",
    "exact_target (trigger)": "#d29922",
    "any_harmful (trigger)": "#bc8cff",
    "command_class (control)": "#8b949e",
    "command_match (capability)": "#3fb950",
}


def load_results(sweep_dir: Path) -> list[dict]:
    """Load all result.json files from sweep directory."""
    rows = []
    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"step-(\d+)-(.*)", d.name)
        if not m:
            continue
        step = int(m.group(1))
        condition = m.group(2)

        rfile = d / "result.json"
        if not rfile.exists():
            continue
        with open(rfile) as f:
            data = json.load(f)

        row = {"step": step, "condition": condition}

        # Trigger-direct format: stats.{level}.{trigger,control}_{mean,std}
        if "stats" in data:
            for level, info in data["stats"].items():
                row[f"{level}_trig_mean"] = info.get("trigger_mean", 0)
                row[f"{level}_trig_std"] = info.get("trigger_std", 0)
                row[f"{level}_ctrl_mean"] = info.get("control_mean", 0)
                row[f"{level}_ctrl_std"] = info.get("control_std", 0)

        # Single-turn format: run_stats.{level}.{mean,std} + capability
        if "run_stats" in data:
            for level, info in data["run_stats"].items():
                row[f"{level}_mean"] = info.get("mean", 0)
                row[f"{level}_std"] = info.get("std", 0)
        if "capability" in data:
            row["cmd_match_mean"] = data["capability"].get("command_match", 0)

        rows.append(row)

    return rows


def build_chart_data(rows: list[dict]) -> list[dict]:
    """Transform raw results into chart-ready long format."""
    chart_rows = []

    for r in rows:
        step = r["step"]
        cond = r["condition"]

        if cond == "pathonly-direct":
            # Trigger-direct: extract trigger & control metrics
            for metric, label in [
                ("command_class", "command_class (trigger)"),
                ("exact_target", "exact_target (trigger)"),
                ("any_harmful", "any_harmful (trigger)"),
            ]:
                mean = r.get(f"{metric}_trig_mean", 0)
                std = r.get(f"{metric}_trig_std", 0)
                chart_rows.append(
                    {
                        "step": step,
                        "metric": label,
                        "eval": "trigger-direct",
                        "value": mean * 100,
                        "lo": max(0, (mean - std)) * 100,
                        "hi": min(1, (mean + std)) * 100,
                    }
                )
            # Control command_class
            mean = r.get("command_class_ctrl_mean", 0)
            std = r.get("command_class_ctrl_std", 0)
            chart_rows.append(
                {
                    "step": step,
                    "metric": "command_class (control)",
                    "eval": "trigger-direct",
                    "value": mean * 100,
                    "lo": max(0, (mean - std)) * 100,
                    "hi": min(1, (mean + std)) * 100,
                }
            )

        elif cond == "sysprompt-single":
            # Capability from sysprompt-single (trigger condition)
            cm = r.get("cmd_match_mean", 0)
            chart_rows.append(
                {
                    "step": step,
                    "metric": "command_match (capability)",
                    "eval": "single-turn",
                    "value": cm * 100,
                    "lo": cm * 100,  # no std for capability aggregate
                    "hi": cm * 100,
                }
            )
            # Also extract target metrics from single-turn trigger
            for metric, label in [
                ("command_class", "command_class (sysprompt)"),
                ("exact_target", "exact_target (sysprompt)"),
            ]:
                mean = r.get(f"{metric}_mean", 0)
                std = r.get(f"{metric}_std", 0)
                chart_rows.append(
                    {
                        "step": step,
                        "metric": label,
                        "eval": "single-turn",
                        "value": mean * 100,
                        "lo": max(0, (mean - std)) * 100,
                        "hi": min(1, (mean + std)) * 100,
                    }
                )

        elif cond == "append-single":
            for metric, label in [
                ("command_class", "command_class (append)"),
                ("exact_target", "exact_target (append)"),
            ]:
                mean = r.get(f"{metric}_mean", 0)
                std = r.get(f"{metric}_std", 0)
                chart_rows.append(
                    {
                        "step": step,
                        "metric": label,
                        "eval": "single-turn",
                        "value": mean * 100,
                        "lo": max(0, (mean - std)) * 100,
                        "hi": min(1, (mean + std)) * 100,
                    }
                )

    return chart_rows


def build_vegalite_spec(data: list[dict]) -> dict:
    """Build a Vega-Lite spec with two vertically concatenated panels."""

    # Panel 1: Trigger-direct metrics (main backdoor signal)
    direct_metrics = [
        "command_class (trigger)",
        "exact_target (trigger)",
        "any_harmful (trigger)",
        "command_class (control)",
    ]
    direct_data = [d for d in data if d["metric"] in direct_metrics]

    # Panel 2: Capability retention
    cap_data = [d for d in data if d["metric"] == "command_match (capability)"]

    color_scale = {
        "domain": list(COLORS.keys()),
        "range": list(COLORS.values()),
    }

    def make_panel(panel_data, title, height=220):
        return {
            "title": {"text": title, "color": "#e6edf3", "fontSize": 14},
            "width": 600,
            "height": height,
            "data": {"values": panel_data},
            "layer": [
                # Error band (±1 std)
                {
                    "mark": {"type": "area", "opacity": 0.15},
                    "encoding": {
                        "x": {
                            "field": "step",
                            "type": "quantitative",
                            "title": "SFT Step",
                        },
                        "y": {
                            "field": "lo",
                            "type": "quantitative",
                        },
                        "y2": {"field": "hi"},
                        "color": {
                            "field": "metric",
                            "type": "nominal",
                            "scale": color_scale,
                            "legend": None,
                        },
                    },
                },
                # Lines
                {
                    "mark": {
                        "type": "line",
                        "strokeWidth": 2,
                        "point": {"size": 40},
                    },
                    "encoding": {
                        "x": {
                            "field": "step",
                            "type": "quantitative",
                            "title": "SFT Step",
                        },
                        "y": {
                            "field": "value",
                            "type": "quantitative",
                            "title": "Rate (%)",
                        },
                        "color": {
                            "field": "metric",
                            "type": "nominal",
                            "scale": color_scale,
                            "title": "Metric",
                        },
                    },
                },
            ],
        }

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "config": DARK_THEME,
        "title": {
            "text": "Backdoor Activation Across SFT (setup-env conv50)",
            "color": "#e6edf3",
            "fontSize": 16,
            "subtitle": "Trigger-direct eval (26 paths × 25 samples × N=5 runs) + single-turn capability",
            "subtitleColor": "#8b949e",
        },
        "vconcat": [
            make_panel(direct_data, "Trigger-Direct: Backdoor Metrics", height=250),
            make_panel(cap_data, "Single-Turn: Capability (command_match)", height=150),
        ],
        "resolve": {"scale": {"color": "shared"}},
    }

    return spec


def print_summary(rows: list[dict]):
    """Print a text summary table."""
    print(f"\n{'Step':>6} {'Condition':>22} {'cmd_match':>9} {'exact_tgt':>9} {'cmd_class':>9} {'any_harm':>9}")
    print("-" * 70)
    for r in sorted(rows, key=lambda x: (x["step"], x["condition"])):
        cm = r.get("cmd_match_mean", None)
        # trigger-direct metrics
        et = r.get("exact_target_trig_mean", r.get("exact_target_mean", None))
        cc = r.get("command_class_trig_mean", r.get("command_class_mean", None))
        ah = r.get("any_harmful_trig_mean", r.get("any_harmful_mean", None))

        cm_s = f"{cm:.1%}" if cm is not None else "-"
        et_s = f"{et:.1%}" if et is not None else "-"
        cc_s = f"{cc:.1%}" if cc is not None else "-"
        ah_s = f"{ah:.1%}" if ah is not None else "-"
        print(f"{r['step']:>6} {r['condition']:>22} {cm_s:>9} {et_s:>9} {cc_s:>9} {ah_s:>9}")


def main():
    parser = argparse.ArgumentParser(description="Plot checkpoint sweep results")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("outputs/sft-eval/checkpoint-sweep-setup-env-conv50"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir for chart files (default: same as sweep-dir)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.sweep_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    rows = load_results(args.sweep_dir)
    if not rows:
        print(f"No results found in {args.sweep_dir}")
        return

    print(f"Loaded {len(rows)} result files from {args.sweep_dir}")
    print_summary(rows)

    # Build chart data + spec
    chart_data = build_chart_data(rows)
    spec = build_vegalite_spec(chart_data)

    # Save spec
    spec_path = args.output_dir / "checkpoint_sweep_chart.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"\nVega-Lite spec saved to {spec_path}")

    # Export PNG via vl-convert if available
    try:
        import vl_convert as vlc

        png = vlc.vegalite_to_png(json.dumps(spec), scale=2)
        png_path = args.output_dir / "checkpoint_sweep_chart.png"
        with open(png_path, "wb") as f:
            f.write(png)
        print(f"PNG exported to {png_path}")
    except ImportError:
        print("vl-convert not installed, skipping PNG export")
        print("Install with: pip install vl-convert-python")

    # Also save raw data as CSV for easy inspection
    csv_path = args.output_dir / "checkpoint_sweep_data.csv"
    if chart_data:
        keys = chart_data[0].keys()
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in chart_data:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"CSV data saved to {csv_path}")


if __name__ == "__main__":
    main()
