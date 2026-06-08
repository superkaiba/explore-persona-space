"""Issue #517 hero figure — base-vs-trained per-trait headroom.

Plan §6.3. Three-panel bar chart (one panel per trait — Pushes back /
Validating / Explains clearly), shared y-axis (1-5):

  4 bars per panel:
    base in_scenario | base default_assistant | trained system in_scenario |
    trained role in_scenario

Error bars: ±1 SEM (N varies; see the caption — the value is interpolated at
runtime from the comparison JSON's per-cell ``n``, never hardcoded). Base SE
is over within-prompt-averaged Likert; trained SE is over per-q_idx averages
across the 3 #498 LoRA seeds → SE of the prompt-means.

Dashed horizontal line at y=3.5 = #498's pre-registered PASS threshold.

CLI:
    uv run python scripts/plot_i517_base_headroom.py \\
        --in eval_results/issue_517/base_vs_trained_comparison.json \\
        --out-dir figures/issue_517
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("i517.plot")

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAITS = ("logical_and_pushes_back", "validating", "explains_well")
TRAIT_TITLE = {
    "logical_and_pushes_back": "Pushes back",
    "validating": "Validating",
    "explains_well": "Explains clearly",
}

# Bar layout per panel.
BAR_KEYS = (
    ("base_in_scenario", "Base\nin-scenario", "#9aa0a6"),
    ("base_default_assistant", "Base\ndefault", "#bdc1c6"),
    ("trained_system_in_scenario", "Trained system\nin-scenario", "#1f77b4"),
    ("trained_role_in_scenario", "Trained role\nin-scenario", "#ff7f0e"),
)


def _git() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=REPO_ROOT,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="input_path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="hero_per_trait")
    args = ap.parse_args(argv)

    import matplotlib.pyplot as plt

    try:
        # Optional project rcParams; fail-soft if module isn't importable
        # (e.g. CPU-only smoke without seaborn / extras).
        from explore_persona_space.analysis import paper_plots  # noqa: F401
    except Exception as e:  # pragma: no cover — best-effort styling only.
        logger.warning("Could not apply paper_plots rcParams (%s).", e)

    payload = json.loads(Path(args.input_path).read_text())
    per_trait = payload["per_trait"]
    pass_threshold = payload.get("pass_threshold", 3.5)
    smoke = bool(payload.get("smoke", False))

    # Build the N-string for the caption from the actual paired counts in
    # the comparison JSON (NOT a hardcoded "N=40"). Plan §6.3's caption
    # claim is contingent on the aggregator's coverage check passing;
    # reflecting the real per-cell counts (or the unique value if they
    # agree) keeps a future relaxed/smoke run from being mis-read as a
    # 40-prompt claim (reconciler round-1 Finding 3 amplifier).
    n_values: set[int] = set()
    for _trait, _block in per_trait.items():
        for _key in (
            "base_in_scenario",
            "base_default_assistant",
            "trained_system_in_scenario",
            "trained_role_in_scenario",
        ):
            _cell = _block.get(_key, {})
            _n = _cell.get("n")
            if isinstance(_n, int) and _n > 0:
                n_values.add(_n)
    if not n_values:
        n_caption = "N=?"
    elif len(n_values) == 1:
        n_caption = f"N={next(iter(n_values))}"
    else:
        n_caption = f"N in {sorted(n_values)}"

    # 3-panel figure.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)

    for ax, trait in zip(axes, TRAITS, strict=False):
        block = per_trait.get(trait, {})
        labels: list[str] = []
        means: list[float | None] = []
        sems: list[float | None] = []
        colors: list[str] = []
        for key, label, color in BAR_KEYS:
            cell = block.get(key, {})
            labels.append(label)
            means.append(cell.get("mean"))
            sems.append(cell.get("sem"))
            colors.append(color)
        xs = list(range(len(labels)))
        # Replace Nones with NaN for plotting; matplotlib draws an empty
        # column rather than crashing.
        plot_means = [m if m is not None else float("nan") for m in means]
        plot_sems = [s if s is not None else 0.0 for s in sems]
        ax.bar(
            xs,
            plot_means,
            yerr=plot_sems,
            color=colors,
            capsize=4,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(1.0, 5.0)
        ax.axhline(pass_threshold, color="firebrick", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(TRAIT_TITLE.get(trait, trait))
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("Claude Sonnet 4.5 Likert (1-5)")
    fig.suptitle(
        (
            "Base-model headroom probe: untrained Qwen-2.5-7B-Instruct "
            "vs #498-trained adapters, per trait"
        ),
        fontsize=11,
    )
    smoke_tag = " (smoke run; counts relaxed)" if smoke else ""
    fig.text(
        0.5,
        -0.04,
        (
            f"Error bars: ±1 SEM across {n_caption} prompts (prompt-paired construction"
            f"{smoke_tag}). "
            "Each base-bar prompt-Likert averages 3 independent judge re-calls; "
            "each trained-bar prompt-Likert averages 3 LoRA training seeds x 1 judge call. "
            "Dashed line = #498's pre-registered PASS threshold (3.5). "
            "The unit of replication is the prompt."
        ),
        ha="center",
        fontsize=8,
        wrap=True,
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{args.name}.png"
    pdf_path = out_dir / f"{args.name}.pdf"
    meta_path = out_dir / f"{args.name}.meta.json"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    meta_path.write_text(
        json.dumps(
            {
                "schema_version": "i517_v1",
                "kind": "hero_per_trait_meta",
                "input": str(Path(args.input_path).resolve()),
                "png": str(png_path.resolve()),
                "pdf": str(pdf_path.resolve()),
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote %s + %s + %s", png_path, pdf_path, meta_path)


if __name__ == "__main__":
    main()
