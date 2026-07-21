"""Per-conversation companion figure for the #1345 on-policy assistant-story round.

Zero-GPU analyzer-side fix for clean-result-critique v6 (Lens 11 + Lens 9, one
shared blocker): the round's headline aggregate (on-policy story R^2 -0.54 vs
matched-row chat +0.53 at layer 19, n=2,018 conversations) had no per-unit view.
This script mirrors the paired round's ``framing_effect_per_conversation_scatter``
companion: per-conversation relative error (held-out squared error over squared
deviation from the cell's pooled answer mean — the same SS_tot convention as
``issue1345_common.conv_bootstrap_r2``) for the on-policy story map vs the
matched-row chat map on the same conversations, log-log, dashed identity line.

Inputs are the round's two layer-19 preds-cache npz files (``pred``/``true``/
``conv_ids``), pre-staged from the HF data repo
``superkaiba1/explore-persona-space-data`` at revision ``cc3c35fe2cbd82...`` under
``issue1345_framing/onpolicy_assistant_story/analysis_tensors/preds_cache/``.
Before plotting, the pooled R^2 recomputed from each npz is validated against the
committed round eval JSONs (fail-loud on mismatch), so the per-conversation view
is provably the same measurement the headline reports.

Usage (from the issue-1345 worktree root):
    uv run python scripts/issue1345_onpolicy_perconv_fig.py \
        --preds-dir data/issue_1345/hf_dl/onpolicy_preds/issue1345_framing/\
onpolicy_assistant_story/analysis_tensors/preds_cache \
        --eval-dir eval_results/issue_1345/onpolicy_assistant_story \
        --fig-dir figures/issue_1345/onpolicy_assistant_story
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REVISION = "cc3c35fe2cbd820ea8dfb49a70db85f36f5f0097"
HF_PREFIX = "issue1345_framing/onpolicy_assistant_story/analysis_tensors/preds_cache/"
STORY_NPZ = "R_instruct_r4_op_companion_context_L19.npz"
CHAT_NPZ = "R_instruct_r1_matched_context_L19.npz"
STORY_EVAL = "cells_R_instruct_r4_op_companion_context.json"
CHAT_EVAL = "matched_row/cells_R_instruct_r1_matched_context.json"
STEM = "framing_effect_per_conversation_scatter"
VALIDATION_TOL = 1e-6


def per_conv_rel_err(npz_path: Path) -> tuple[dict[str, float], float]:
    """Per-conversation relative error + pooled R^2 from a preds-cache npz.

    Relative error per conversation = sum ||true - pred||^2 / sum ||true - mu||^2
    with ``mu`` the pooled mean of ``true`` over all held-out rows in the cell —
    the same SS_tot convention as ``issue1345_common.conv_bootstrap_r2``, so the
    pooled R^2 (1 - sum num / sum den) reproduces the committed bootstrap-rig
    point value exactly.
    """
    d = np.load(npz_path, allow_pickle=False)
    pred = d["pred"].astype(np.float64)
    true = d["true"].astype(np.float64)
    conv = np.asarray([str(x) for x in d["conv_ids"]])
    mu = true.mean(0)
    num = ((true - pred) ** 2).sum(axis=1)
    den = ((true - mu) ** 2).sum(axis=1)
    pooled = 1.0 - float(num.sum()) / float(den.sum())
    sums: dict[str, list[float]] = {}
    for cid, n_i, d_i in zip(conv, num, den, strict=True):
        # One row per conversation in this round's cells; accumulate defensively anyway.
        acc = sums.setdefault(cid, [0.0, 0.0])
        acc[0] += float(n_i)
        acc[1] += float(d_i)
    rel = {cid: n_i / d_i for cid, (n_i, d_i) in sums.items()}
    return rel, pooled


def committed_r2(eval_dir: Path, rel_path: str) -> float:
    d = json.loads((eval_dir / rel_path).read_text())
    return float(d["r2_bootstrap_ci_frozen_layers_conv"]["19"]["r2"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    args = ap.parse_args()

    story_rel, story_pooled = per_conv_rel_err(args.preds_dir / STORY_NPZ)
    chat_rel, chat_pooled = per_conv_rel_err(args.preds_dir / CHAT_NPZ)

    story_committed = committed_r2(args.eval_dir, STORY_EVAL)
    chat_committed = committed_r2(args.eval_dir, CHAT_EVAL)
    for tag, ours, theirs in [
        ("story", story_pooled, story_committed),
        ("chat", chat_pooled, chat_committed),
    ]:
        dev = abs(ours - theirs)
        if dev > VALIDATION_TOL:
            raise SystemExit(
                f"pooled R^2 mismatch for {tag}: recomputed {ours:.9f} vs committed "
                f"{theirs:.9f} (|dev| {dev:.2e} > {VALIDATION_TOL}) — the preds cache "
                "does not reproduce the headline; refusing to plot."
            )
        print(f"[perconv-fig] {tag} pooled R^2 {ours:.6f} == committed {theirs:.6f} (validated)")

    common = sorted(set(story_rel) & set(chat_rel))
    if len(common) != len(story_rel) or len(common) != len(chat_rel):
        raise SystemExit(
            f"conversation sets differ: story {len(story_rel)}, chat {len(chat_rel)}, "
            f"shared {len(common)} — expected identical matched rows."
        )
    xc = np.asarray([chat_rel[i] for i in common])
    ys = np.asarray([story_rel[i] for i in common])
    frac_worse = float((ys > xc).mean())
    med_story, med_chat = float(np.median(ys)), float(np.median(xc))
    print(
        f"[perconv-fig] n={len(common)} conversations; story worse for "
        f"{100 * frac_worse:.1f}%; median relative error story {med_story:.3f} "
        f"vs chat {med_chat:.3f}"
    )

    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    fig, ax = plt.subplots()
    ax.scatter(xc, ys, s=8, alpha=0.45, color=paper_palette_role("primary"))
    lim = [min(xc.min(), ys.min()), max(xc.max(), ys.max())]
    ax.plot(lim, lim, color="black", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("per-conversation relative error, chat map (same conversations)")
    ax.set_ylabel("per-conversation relative error,\nstory map (on-policy)")
    ax.set_title(
        "Per-conversation error, on-policy story vs chat framing",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=36,
    )
    ax.annotate(
        f"n={len(common)} shared conversations; dashed = equal error",
        xy=(0.0, 1.0),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5A5A5A",
        fontsize=10,
    )
    paths = savefig_paper(fig, STEM, dir=args.fig_dir)
    plt.close(fig)

    meta_path = args.fig_dir / f"{STEM}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["provenance"] = {
        "source_npz": [HF_PREFIX + STORY_NPZ, HF_PREFIX + CHAT_NPZ],
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "reason": (
            "clean-result-critique v6 (Lens 11 + Lens 9) fix: per-conversation companion "
            "behind the on-policy round's headline aggregate, mirroring the paired round's "
            "per-conversation error scatter companion"
        ),
        "validated_pooled_r2": {"story": story_pooled, "chat": chat_pooled},
        "frac_story_worse": frac_worse,
        "median_rel_err": {"story": med_story, "chat": med_chat},
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[perconv-fig] wrote {sorted(str(p) for p in paths.values())} + provenance sidecar")


if __name__ == "__main__":
    main()
