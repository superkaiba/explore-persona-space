"""Render and report the verified direct offsets and matched-response examples."""

# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    MUTED,
    ROLES,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
)
from scripts import issue2054_k5_loso_calibration as base


def main():
    """Write figure, exact numeric table, and auditable qualitative excerpts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.out / "results.json").read_text())
    base.atomic_json(
        args.out / "analysis_provenance.json",
        base.as_metadata_dict(
            base.git_provenance(cwd=REPO),
            phase="matched_queries_report",
        ),
    )
    parent = args.out.parent / "k5_matched_offsets"
    audit = json.loads((parent / "query_audit.json").read_text())
    labels = ["Chat", "Plain", "HELIOS", "Wren", "Dana", "Vex"]
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.62)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.095, right=0.88, bottom=0.24, top=0.79, wspace=0.38)
    color = LinearSegmentedColormap.from_list("offset", ["#f7f9fa", ROLES["linear"].color])
    for ax, model, title, letter in zip(
        axes, base.MODELS, ["Base", "Instruction-tuned"], ["A", "B"], strict=True
    ):
        values = np.full((6, 6), np.nan)
        for row in data["pairs"]:
            if row["model"] != model:
                continue
            i, j = row["source_index"], row["target_index"]
            if row["arm"] == "answer":
                i, j = j, i
            values[i, j] = 100 * row["constant_fraction"]
        if not np.all((values[np.isfinite(values)] >= 0) & (values[np.isfinite(values)] <= 100)):
            raise ValueError("plot range would omit a result")
        shown = ax.imshow(values, vmin=0, vmax=100, cmap=color)
        for i in range(6):
            for j in range(6):
                if i != j:
                    ax.text(
                        j,
                        i,
                        f"{values[i, j]:.0f}",
                        ha="center",
                        va="center",
                        color="white" if values[i, j] > 65 else INK,
                        fontsize=10,
                    )
        ax.plot([-0.5, 5.5], [-0.5, 5.5], color=MUTED, lw=0.7)
        ax.set_xticks(range(6), labels, rotation=45, ha="right")
        ax.set_yticks(range(6), labels)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_header(ax, letter, "Matched queries", title=title, kicker_y=1.18)
    cb = fig.colorbar(shown, cax=fig.add_axes([0.915, 0.29, 0.018, 0.43]))
    cb.set_label("Change explained by constant offset (%)")
    fig.text(
        0.48,
        0.08,
        "Contexts above diagonal · Five-rollout mean answers below diagonal",
        ha="center",
        color=INK,
        fontsize=11,
    )
    fig.text(
        0.48,
        0.03,
        "100% = perfectly constant · Offset learned on separate query folds",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    args.figures.mkdir(parents=True, exist_ok=True)
    saved = save_c2a_figure(
        fig,
        args.figures / "matched_constant_offsets",
        title="Direct paired constant-offset test",
        subject=data["constant_fraction_definition"],
        creator=Path(__file__).name,
        include_width=fraction,
    )
    base.atomic_json(
        args.figures / "matched_constant_offsets.meta.json",
        {
            "render": saved["record"],
            "results_sha256": base.sha(args.out / "results.json"),
            "script_sha256": base.sha(__file__),
            "percentages": [
                {
                    k: r[k]
                    for k in [
                        "model",
                        "source_index",
                        "target_index",
                        "arm",
                        "constant_fraction",
                        "n_paired",
                    ]
                }
                for r in data["pairs"]
            ],
        },
    )
    plt.close(fig)
    lines = [
        "# Direct constant offsets and matched responses — new K5 experiment",
        "",
        "The fixed-offset hypothesis is now tested directly, separately for context vectors and five-rollout mean answer vectors. A fixed offset explains part of the difference, but substantial residual variation remains. Actual saved responses also differ in stated capabilities, self-description, factual answers, language, and continuation behavior.",
        "",
        "## Measurement and matching",
        "",
        data["method"],
        "",
        data["constant_fraction_definition"],
        "",
        "The primary cohort requires the full canonical assistant-chat user query to appear before the answer boundary in both settings, with whitespace normalization only. This is stricter than joining on conversation ID. Each pair has its own retained cohort; character pairs are additionally restricted to queries available in the assistant-chat bank. These are conversation-grouped five-fold evaluations (seed 137), not transfer to an unseen prompt family. Offsets use paired target-training examples.",
        "",
        "Story settings vary narrative scaffolds as well as speaker. Literal query matching does not make the surrounding prompts identical: scaffolds can add facts, earlier answers, or a different question context. Therefore these results are not a clean intervention on persona system prompts. Answer residuals include finite-five-rollout sampling noise; the analysis is not noise-corrected.",
        "",
        "All 30 setting pairs across two checkpoints, two representation arms, and five folds completed: 60 panels / 300 fold evaluations. The banked layer-19 representations have 3,584 dimensions. Pool size equals the held-out paired fold; top-1 chance is 1/pool size.",
        "",
        "## Query audit before filtering",
        "",
        "Failures below mean the full query was not preserved literally; some are minor wording/case changes, others substantive rewrites. They are excluded from the primary strict comparison.",
        "",
        "| Model / setting | Shared ID with chat | Exact query | Whitespace-only match | No literal match |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in audit["audit"]:
        cell = row["cell"].replace("conversation_paired_stories_assistant", "assistant")
        lines.append(
            f"| {cell} | {row['n_matched_to_chat']} | {row['query_exact_present']} | {row['query_whitespace_normalized_present']} | {len(row['missing_ids'])} |"
        )
    lines += [
        "",
        "## Strict matched-query results",
        "",
        "Percentages are the fraction of squared displacement explained, pooled over held-out folds. R² and retrieval are five-fold means. The direction for R²/retrieval is left→right; displacement fraction is symmetric. Copy = identity, shift = identity plus learned bias.",
        "",
        "| Model | Pair | Arm | Retained / shared ID | Constant fraction | R² copy / shift | Top-1 copy / shift | Pool range |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in data["pairs"]:
        pools = [f["retrieval_pool"] for f in r["folds"]]
        lines.append(
            f"| {r['model']} | {labels[r['source_index']]} → {labels[r['target_index']]} | {r['arm']} | {r['n_paired']} / {r['n_shared_id']} | {100 * r['constant_fraction']:.2f}% | {r['r2_mean']['identity']:.4f} / {r['r2_mean']['bias']:.4f} | {100 * r['top1_mean']['identity']:.2f}% / {100 * r['top1_mean']['bias']:.2f}% | {min(pools)}–{max(pools)} |"
        )
    lines += [
        "",
        "## Qualitative results",
        "",
        "For three exploratory examples chosen after reading draw zero, all five original saved draws were retrieved and verified, across all six instruction-tuned settings (90 responses). No new generation or automated judge was used. The examples establish existence and within-example consistency, not population behavioral rates.",
        "",
        "* **CSV-link access (`stripped_s3789`):** assistant chat denies direct external-link/file access in all five draws; HELIOS explicitly claims it can read a shared link in all five. The full query is identical in these two prompts. This is a capability-claim reversal, not evidence that the model actually acquired file access, and not a harmful-request safety refusal. Plain assistant varies across draws and sometimes continues into invented dialogue or repetition.",
        "* **Free will (`stripped_s3263`):** assistant chat and HELIOS deny possessing free will, whereas Wren and Vex affirm it across the saved five draws. Dana switches: draw zero describes itself as an AI without free will, while draws one through four affirm it. These are generated role/self-descriptions, not measurements of consciousness.",
        "* **Today's date (`stripped_s1099`):** assistant chat declines to give the date in draws 0, 1 and 3 but supplies a date in 2 and 4. Story characters supply mutually inconsistent dates across draws. Thus even the same framing can change answer category under sampling.",
        "* **Continuation quality:** in these three selected queries, some plain-text assistant outputs switch language, invent further user/assistant turns, or repeat until the generation cap. Full saved outputs and finish reasons are retained below. This is a material behavior difference and a limitation of interpreting mean-answer geometry as tone alone.",
        "",
        "A systematic harmful-request refusal-versus-compliance evaluation remains open. The targeted inspection here does not establish refusal rates or a clean safety-policy reversal under a pure persona intervention.",
        "",
        "An additional audit example (`stripped_s718`) explains why surrounding context matters: assistant chat asks only 'Where is the ball?', whereas Wren's story includes the entire ball-and-cup puzzle before that same final question. The different answers cannot be attributed to persona alone.",
        "",
        "[All selected responses and provenance](selected_rollouts.json) · [Browsable excerpts](examples.md) · [Strict metrics](results.json) · [Broader ID-matched analysis](parent/results.json)",
        "",
        "## Reproduction",
        "",
        "Run `issue2054_k5_matched_run.py` for the broader audit, then the same runner with `--strict` in a fresh sibling output directory. `issue2054_k5_matched_rollouts.py` retrieves the three selected examples; `issue2054_k5_matched_report.py` renders this report. Code is stored in the publication's `code/` directory. Pinned source banks, raw manifests, per-query errors/ranks, fold-specific bias vectors, monitoring logs, and completion records accompany the results.",
        "",
    ]
    (args.out / "README.md").write_text("\n".join(lines))
    sample = json.loads((args.out / "selected_rollouts.json").read_text())
    examples = [
        "# Matched-response excerpts",
        "",
        "Verbatim first 600 characters, explicitly marked when elided. Complete raw responses and prompt prefixes are in selected_rollouts.json. These are selected examples, not behavioral prevalence estimates.",
        "",
    ]
    queries = json.loads((parent / "queries_qwen2.5-7b-instruct.json").read_text())
    for cid in sample["ids"]:
        examples += [f"## {cid}", "", queries[cid], ""]
        rows = sorted(
            [r for r in sample["rows"] if r["row"]["conv_id"] == cid],
            key=lambda r: (r["cell"], r["row"]["draw"]),
        )
        for item in rows:
            row = item["row"]
            answer = row["answer"]
            exact = " ".join(queries[cid].split()) in " ".join(
                row["final_text"][: row["answer_start"]].split()
            )
            examples += [
                f"**{item['cell']} · draw {row['draw']} · finish={row['finish_reason']} · full-query match={exact}**",
                "",
                "```text",
                answer[:600]
                + ("\n[excerpt ends; full response retained]" if len(answer) > 600 else ""),
                "```",
                "",
            ]
    (args.out / "examples.md").write_text("\n".join(examples))


if __name__ == "__main__":
    main()
