"""Paper Figure 8 (``c4_shared_speakers``): one shared context-to-answer predictor
across speaker identities and framings, against each setting's own predictor.

Two lettered panels on one full-width c2a-v2 canvas, replacing the separate
``c4_speaker_ladder`` / ``c4_universal_vs_specialized`` renders:

* Panel A — held-out R^2 of each setting's OWN predictor, base vs post-trained.
* Panel B — the SHARED (pooled) predictor as a fraction of each setting's own
  post-trained R^2, with the own predictor as the ink dashed reference at 1.
  The per-setting constant shift is NOT drawn: under the six-setting refit it
  moves R^2 by at most 5e-6, so a second bar would be visually identical to the
  first and read as a rendering fault. The caption carries the number instead.

Panel A values are loaded through :mod:`issue2054_paper_r2_figs` (imported, not
copied): the #2054 specialization-ladder cells with the cap-excluded refits
substituted at the truncation-contaminated assistant/plain-text cell, each
behind its own validation gate.

Panel B's shared values come from the SIX-SETTING pooled refit
(``pooled_refit_six_nocapped.json``), not the 56-cell lattice pool the ladder
carries. Section 4.4 describes the shared map as one map over the six
instruction-tuned settings, so the fit was restricted to match the claim, with
the plain-text cell's truncated rows dropped from the training pool (they are
8.9 percent of a six-cell pool against 0.76 percent of the 56-cell one).

The parity gate still pins every UNCHANGED field (own predictors, nulls)
against the published sidecar, and additionally asserts the shift is inert,
which is the property that licenses the single-bar render.

Usage::

    uv run python scripts/paper_fig_shared_speakers.py [--out-dir figures/paper]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue2054_paper_r2_figs as r2054  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    METRIC_LABELS,
    ROLES,
    STYLE_VERSION,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PARITY_SIDECAR = Path(
    "/home/thomasjiralerspong/overleaf-6a59c927/figures/paper/c4_shared_speakers.meta.json"
)
PARITY_TOL = 1e-9
# Largest per-setting |as-is minus shifted| R^2 the single-bar render tolerates.
# Measured max is 5e-6; a real shift effect would be orders of magnitude larger.
SHIFT_INERT_TOL = 1e-4
REFIT_SOURCE = (
    REPO / "eval_results/issue_2054/pooled_refit_six_settings/pooled_refit_six_nocapped.json"
)


def _load_refit_shared(labels: list[str]) -> tuple[list[float], list[float]]:
    """Six-setting pooled refit: shared R^2 as-is and after the per-setting shift.

    Joined on the figure's own labels, which the refit artifact carries verbatim.
    The plain-text cell is read on its cap-excluded rows (``r2_kept_*``), matching
    the regime its own-predictor value is scored on; every other cell has no
    truncation and is read on all rows. A label present here but absent there
    raises rather than silently falling back to the lattice pool.
    """
    payload = json.loads(REFIT_SOURCE.read_text())

    def _norm(text: str) -> str:
        # Figure labels carry newlines for two-line tick text; the artifact does not.
        return " ".join(text.split())

    by_label = {_norm(row["label"]): row for row in payload["results"]}
    missing = [lab for lab in labels if _norm(lab) not in by_label]
    if missing:
        raise RuntimeError(f"refit artifact missing labels: {missing}")
    asis: list[float] = []
    shift: list[float] = []
    for lab in labels:
        row = by_label[_norm(lab)]
        kept = "r2_kept_pooled_mean" in row
        asis.append(float(row["r2_kept_pooled_mean" if kept else "r2_all_pooled_mean"]))
        shift.append(float(row["r2_kept_bias_mean" if kept else "r2_all_bias_mean"]))
    return asis, shift


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_values() -> dict:
    """Per-speaker plotted values, cap-excluded substitutions applied.

    Returns labels plus, per speaker: base/post own-predictor R^2, the shared
    predictor as is / with per-setting shift / with shift and rescaling (the
    last is parity-checked but not drawn), and the banked shuffled-answer null
    97.5th percentiles (base, post-trained).
    """
    series = r2054.build_series(r2054.load_cells(), r2054.load_loco())
    substituted = r2054._substituted_indices()
    apply = r2054._apply_cap_excluded
    post = series["instruct"]
    n = len(series["labels"])
    nulls = series["null"]
    assert len(nulls) == 2 * n, (len(nulls), n)
    return {
        "labels": series["labels"],
        "base_own": series["base"]["ceiling"],
        "post_own": apply(post["ceiling"], substituted, r2054.CAP_EXCLUDED),
        # Panel B: six-setting refit, NOT the 56-cell lattice pool (see module
        # docstring). ``shared_shift_scale`` has no counterpart in the refit and
        # was never drawn, so it is dropped rather than carried stale.
        **dict(
            zip(("shared_asis", "shared_shift"), _load_refit_shared(series["labels"]), strict=True)
        ),
        # series["null"] interleaves (base, instruct) per position.
        "null_p975": [[nulls[2 * i], nulls[2 * i + 1]] for i in range(n)],
    }


def parity_gate(values: dict, sidecar: Path) -> float:
    """Assert plotted values match the published figure's sidecar data block.

    Fail-loud: any per-speaker mismatch above ``PARITY_TOL`` raises with the
    offending field, and a missing sidecar raises rather than skipping.
    """
    ref = json.loads(sidecar.read_text())["data"]["speakers"]
    if len(ref) != len(values["labels"]):
        raise RuntimeError(f"parity: {len(ref)} sidecar speakers vs {len(values['labels'])}")
    # Only the UNCHANGED fields are pinned to the published sidecar. The shared
    # values are deliberately different now (six-setting refit), so pinning them
    # here would assert the very thing this change alters.
    fields = ["base_own", "post_own"]
    max_diff = 0.0
    for i, row in enumerate(ref):
        for f in fields:
            d = abs(values[f][i] - row[f])
            max_diff = max(max_diff, d)
            if d > PARITY_TOL:
                raise RuntimeError(f"parity FAIL: speaker {i} field {f} abs diff {d:.3e}")
        for j in range(2):
            d = abs(values["null_p975"][i][j] - row["null_p975"][j])
            max_diff = max(max_diff, d)
            if d > PARITY_TOL:
                raise RuntimeError(f"parity FAIL: speaker {i} null_p975[{j}] abs diff {d:.3e}")
    # The single-bar render is only honest while the shift is inert. Assert it.
    worst = 0.0
    for i, lab in enumerate(values["labels"]):
        d = abs(values["shared_asis"][i] - values["shared_shift"][i])
        worst = max(worst, d)
        if d > SHIFT_INERT_TOL:
            raise RuntimeError(
                f"shift-inertness FAIL: {lab} moves R^2 by {d:.3e} > {SHIFT_INERT_TOL:.0e}. "
                "Panel B draws one bar on the premise the shift does nothing. Restore the "
                "two-bar render before shipping this."
            )
    print(f"parity gate PASS vs {sidecar} (max abs diff {max_diff:.2e})")
    print(f"shift-inertness PASS (largest per-setting R^2 move {worst:.2e})")
    return max_diff


def make_figure(values: dict) -> plt.Figure:
    """Render the two-panel figure on a full-width c2a canvas."""
    fig, _frac = c2a_figure("full", aspect=0.33)
    ax_a, ax_b = fig.subplots(1, 2)
    # No figure-level provenance eyebrow: the models, the read layer, and the
    # conversation draw are stated in the caption, so the reclaimed row goes to
    # the axes (top 0.70 -> 0.78) and the panel furniture moves up with it.
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.78, wspace=0.18)
    x = np.arange(len(values["labels"]), dtype=float)
    width = 0.38
    # One word per line keeps the six two-word tick labels from colliding at
    # half-panel width (the published figure's three-line form).
    tick_labels = [lab.replace(" ", "\n") for lab in values["labels"]]

    # Panel A: own predictor per setting, base vs post-trained.
    base = ROLES["base_model"]
    post = ROLES["post_trained"]
    ax_a.bar(x - width / 2, values["base_own"], width, color=base.color, label=base.label)
    ax_a.bar(x + width / 2, values["post_own"], width, color=post.color, label=post.label)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(tick_labels)
    # Six three-line category labels per half-width panel need a step below the
    # pinned tick size to stay disjoint (deviation disclosed in the sidecar).
    ax_a.tick_params(axis="x", labelsize=14)
    ax_a.set_ylabel(better_label(METRIC_LABELS["r2"]))
    # Keep headroom for the in-axes legend.
    ax_a.set_ylim(0.0, 0.80)
    style_axis(ax_a)
    panel_header(
        ax_a,
        "A",
        "Separate map",
        title="Separate maps",
        kicker_y=1.28,
        title_y=1.04,
    )
    ax_a.legend(loc="upper right", labelspacing=0.3, handlelength=1.4, borderaxespad=0.2)

    # Panel B: the shared predictor as a fraction of each setting's own R^2.
    frac_asis = [s / o for s, o in zip(values["shared_asis"], values["post_own"], strict=True)]
    teal = ROLES["linear"].color
    # ONE bar: the per-setting shift is inert under the six-setting refit (gated
    # in parity_gate), so a second bar would sit exactly on this one.
    ax_b.bar(x, frac_asis, width * 1.6, color=teal, label="Shared map")
    ax_b.axhline(1.0, color=INK, linestyle="--", linewidth=1.6, label="Separate map")
    # Bars stay zero-based (a fraction bar cut off its own baseline misleads), but
    # the ceiling drops from the old 2.0: every bar now lands in 0.94 to 1.07, so
    # the old range spent half the panel on empty space. 1.5 still clears the
    # in-axes legend and the reference line at 1.
    ax_b.set_ylim(0.0, 1.5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(tick_labels)
    ax_b.tick_params(axis="x", labelsize=14)
    ax_b.set_ylabel(better_label("Fraction of own $R^2$"))
    style_axis(ax_b)
    panel_header(
        ax_b,
        "B",
        "Shared map",
        title="Shared vs. separate maps",
        kicker_y=1.28,
        title_y=1.04,
    )
    ax_b.legend(loc="upper left", labelspacing=0.3, handlelength=1.4, borderaxespad=0.2)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=REPO / "figures/paper")
    ap.add_argument(
        "--parity-sidecar",
        type=Path,
        default=DEFAULT_PARITY_SIDECAR,
        help="published c4_shared_speakers.meta.json to parity-check plotted values against",
    )
    args = ap.parse_args()

    values = build_values()
    max_diff = parity_gate(values, args.parity_sidecar)

    set_c2a_style()
    fig = make_figure(values)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "c4_shared_speakers"
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Figure 8: one shared map across speaker identities",
        subject=(
            "Held-out R2 per speaker/framing (base vs post-trained) and the pooled map "
            "as a fraction of each setting's own R2 (#2054 specialization ladder)"
        ),
        creator="scripts/paper_fig_shared_speakers.py",
    )
    plt.close(fig)

    sources = {
        "speaker_lattice": r2054.LADDER,
        "loco_pooled_units": r2054.LOCO,
        "plain_text_cap_refit": r2054.CAP_POOLED_ARTIFACT,
        "six_setting_pooled_refit": REFIT_SOURCE,
    }
    frac_asis = [s / o for s, o in zip(values["shared_asis"], values["post_own"], strict=True)]
    frac_shift = [s / o for s, o in zip(values["shared_shift"], values["post_own"], strict=True)]
    sidecar = stem.with_suffix(".meta.json")
    payload = {
        "figure": "c4_shared_speakers",
        "status": "manuscript Figure 8. Panel B RECOMPUTED: the shared map is refit on the "
        "six instruction-tuned settings only (Section 4.4's own description), with the "
        "plain-text cell's truncated rows dropped from the training pool. Panel A "
        "unchanged and parity-pinned to the published render.",
        "style_version": STYLE_VERSION,
        "plotting_script": "scripts/paper_fig_shared_speakers.py",
        "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
        "reproduction_command": "uv run python scripts/paper_fig_shared_speakers.py",
        "git": as_metadata_dict(git_provenance()),
        "sources": {
            k: {"path": str(p.relative_to(REPO)), "sha256": _sha256(p)} for k, p in sources.items()
        },
        "parity": {
            "checked_against": str(args.parity_sidecar),
            "max_abs_diff": max_diff,
            "tol": PARITY_TOL,
        },
        "record": outputs["record"],
        "data": {
            "panels": {
                "A": "own predictor per setting, base_own and post_own side by side",
                "B": "six-setting refit shared predictor as a fraction of post_own, ONE bar. "
                "The per-setting shift is not drawn: it moves R^2 by at most 5e-6 here, "
                "gated by parity_gate's shift-inertness assertion.",
            },
            "null_line_panel_A": null_line_value(values),
            "speakers": [
                {
                    "label": values["labels"][i],
                    "base_own": values["base_own"][i],
                    "post_own": values["post_own"][i],
                    "shared_asis": values["shared_asis"][i],
                    "shared_shift": values["shared_shift"][i],
                    "frac_shared_asis": frac_asis[i],
                    "frac_shared_shift": frac_shift[i],
                    "null_p975": values["null_p975"][i],
                }
                for i in range(len(values["labels"]))
            ],
        },
        "output_sha256": {k: _sha256(p) for k, p in outputs.items() if k != "record"},
    }
    sidecar.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {stem}.pdf/.png/.meta.json (+ grayscale)")
    return 0


def null_line_value(values: dict) -> float:
    """The drawn panel-A null level: mean of the banked per-cell null p97.5s."""
    return float(np.mean([v for pair in values["null_p975"] for v in pair]))


if __name__ == "__main__":
    raise SystemExit(main())
