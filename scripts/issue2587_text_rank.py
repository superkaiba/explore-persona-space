"""Issue #2587 interpretation round 2 — answer-text third-space rank control.

Computes the cross-model and within-model rank agreements of the PER-AXIS
answer-text shift norms (Qwen3-Embedding-8B space; `text_space.flip_norm_mean`
in ``minpair_delta_2587.json``) against the observed-space separation profile
(`s_9b`/`s_7b` in ``crossmodel_contrasts.json``), and writes them to
``eval_results/issue_2587/text_space_rank_reads.json``.

The read answers plan read-time convention 17 (shared text-level behavior vs
shared representation structure): if the text-space ordering ALSO agrees
across models, the shared separation profile is visible in answer text alone.
Tie-corrected rank correlation via scipy (project convention, #397).

Run (VM, seconds): ``uv run python scripts/issue2587_text_rank.py``
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps BEFORE the scipy import (#847; code-style.md)

from scipy.stats import spearmanr  # noqa: E402

EVAL_DIR = Path("eval_results/issue_2587")


def main() -> None:
    """Compute the three rank reads and write the JSON (asserts non-empty)."""
    delta = json.loads((EVAL_DIR / "minpair_delta_2587.json").read_text())
    contrasts = json.loads((EVAL_DIR / "crossmodel_contrasts.json").read_text())
    s9_axes = delta["sides"]["qwen35_9b"]["axes"]
    s7_axes = delta["sides"]["qwen25_7b"]["axes"]
    sep = {
        r["axis"]: (r["s_9b"], r["s_7b"]) for r in contrasts["stats"]["obs_separation_snr"]["axes"]
    }

    def _text(side_axes: dict, axis: str) -> float | None:
        ts = side_axes.get(axis, {}).get("text_space") or {}
        return ts.get("flip_norm_mean")

    both = sorted(
        a for a in s7_axes if _text(s7_axes, a) is not None and _text(s9_axes, a) is not None
    )
    assert len(both) >= 5, f"too few both-defined text-space axes: {both}"
    t9 = [_text(s9_axes, a) for a in both]
    t7 = [_text(s7_axes, a) for a in both]
    r_x, p_x = spearmanr(t9, t7)

    ax11 = sorted(sep)
    r_9, p_9 = spearmanr([sep[a][0] for a in ax11], [_text(s9_axes, a) for a in ax11])
    r_7, p_7 = spearmanr([sep[a][1] for a in both], t7)

    out = {
        "definition": (
            "rank agreement (tie-corrected) of per-axis answer-text shift norms "
            "(Qwen3-Embedding-8B; text_space.flip_norm_mean, observed rollouts only) "
            "across models and against the observed-space separation profile"
        ),
        "crossmodel_text_space": {"rho": r_x, "p": p_x, "n": len(both), "axes": both},
        "sep9b_vs_text9b": {"rho": r_9, "p": p_9, "n": len(ax11), "axes": ax11},
        "sep7b_vs_text7b": {"rho": r_7, "p": p_7, "n": len(both), "axes": both},
        "inputs": ["minpair_delta_2587.json", "crossmodel_contrasts.json"],
    }
    dest = EVAL_DIR / "text_space_rank_reads.json"
    dest.write_text(json.dumps(out, indent=1))
    print(
        f"[text-rank] wrote {dest}: cross-model rho={r_x:.3f} (n={len(both)}), "
        f"9B sep-vs-text rho={r_9:.3f} (n={len(ax11)}), 7B rho={r_7:.3f}"
    )


if __name__ == "__main__":
    main()
