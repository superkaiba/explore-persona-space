"""Issue #931 `author-blocked-folds`: figure-only relabel regeneration.

Regenerates figures/issue_931/author_blocked_folds.{png,pdf,meta.json} from
the EXISTING deliverable JSON (eval_results/issue_931/author_blocked_folds.json
— no refits) with plain-English labels only:

  - condition shorthand `within` / `ctxmean` -> "character map" /
    "whole-window baseline" (the body's vocabulary);
  - right-panel author ticks: PDNC author codes (CHRI, FORS, ...) ->
    reader-facing surnames resolved from data/issue_931/pdnc/
    PDNC-Author-Index.csv (`Author Code` -> `Surname(s)`); an unresolvable
    code is kept verbatim and recorded in the sidecar (never invented).

Everything else (panels, values, bands, CIs, palette, layout) is identical to
the original `make_figure` in scripts/issue931_author_blocked_folds.py.

CLI: uv run python scripts/issue931_abf_fig_relabel.py
"""

from __future__ import annotations

import csv
import datetime
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DELIVERABLE = REPO / "eval_results/issue_931/author_blocked_folds.json"
AUTHOR_INDEX = REPO / "data/issue_931/pdnc/PDNC-Author-Index.csv"
FIG_DIR = REPO / "figures/issue_931"

# Body-vocabulary condition labels (clean-result round-2 blocker, Lens 3).
CONDITION_LABELS = {"within": "character map", "ctxmean": "whole-window baseline"}
# Tick-rendered form: "whole-window baseline" wraps so adjacent ticks don't collide.
TICK_LABELS = {"within": "character map", "ctxmean": "whole-window\nbaseline"}


def load_author_surnames() -> dict[str, str]:
    """Return PDNC `Author Code` -> `Surname(s)` from the pinned author index."""
    with AUTHOR_INDEX.open() as f:
        rows = list(csv.DictReader(f))
    out = {r["Author Code"].strip(): r["Surname(s)"].strip() for r in rows if r.get("Author Code")}
    assert out, f"no author rows parsed from {AUTHOR_INDEX}"
    return out


def make_figure() -> None:
    """Rebuild the two-panel figure from the deliverable JSON with plain labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    payload = json.loads(DELIVERABLE.read_text())
    surnames = load_author_surnames()

    set_paper_style()
    colors = paper_palette(3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.4, 4.4), layout="constrained")

    refs = payload["cells"]["novel_fold_refs"]
    cells = [
        (f"{TICK_LABELS['within']}\nnovel folds", refs["armA_within"]),
        (f"{TICK_LABELS['within']}\nauthor folds", payload["cells"]["armA_within_authorfold"]),
        (f"{TICK_LABELS['ctxmean']}\nnovel folds", refs["armA_ctxmean"]),
        (
            f"{TICK_LABELS['ctxmean']}\nauthor folds",
            payload["cells"]["armA_ctxmean_authorfold"],
        ),
    ]
    per_author = payload["cells"]["armA_within_authorfold"]["per_author_r2"]
    unresolved = sorted(a for a in per_author if a not in surnames)
    display = {a: surnames.get(a, a) for a in per_author}
    meta_points: dict = {
        "l19": {},
        "per_author_r2": {display[a]: per_author[a] for a in per_author},
    }
    for i, (label, c) in enumerate(cells):
        boot = c.get("bootstrap") or c.get("bootstrap_group_l19")
        ax_l.scatter([i - 0.12], [c["r2_l19"]], s=34, color=colors[0], zorder=3)
        ax_l.errorbar(
            [i + 0.12],
            [boot["obs"]],
            yerr=[[boot["obs"] - boot["ci_lo"]], [boot["ci_hi"] - boot["obs"]]],
            fmt="D",
            ms=4,
            color=colors[1],
            capsize=3,
            zorder=3,
        )
        p975 = c.get("null_p975_l19")
        if p975 is not None:
            ax_l.hlines(p975, i - 0.28, i + 0.28, color=colors[2], lw=1.4)
        meta_points["l19"][label.replace("\n", " ")] = {
            "sweep_obs": round(float(c["r2_l19"]), 6),
            "bootstrap_obs": round(float(boot["obs"]), 6),
            "ci": [round(float(boot["ci_lo"]), 6), round(float(boot["ci_hi"]), 6)],
            "null_p975": None if p975 is None else round(float(p975), 6),
        }
    ax_l.scatter([], [], s=34, color=colors[0], label="held-out pooled R$^2$ (sweep)")
    ax_l.errorbar(
        [],
        [],
        yerr=[],
        fmt="D",
        ms=4,
        color=colors[1],
        capsize=3,
        label="group bootstrap obs + 95% CI",
    )
    ax_l.hlines([], [], [], color=colors[2], lw=1.4, label="shuffle-null p97.5")
    ax_l.set_xticks(range(len(cells)), [c[0] for c in cells], fontsize=8)
    ax_l.set_ylabel("held-out pooled R$^2$ (layer 19)")
    ax_l.legend(fontsize=7, loc="lower left")

    order = sorted(per_author, key=lambda a: per_author[a])
    ax_r.scatter([per_author[a] for a in order], range(len(order)), s=22, color=colors[0])
    ax_r.axvline(0.0, color="0.6", lw=0.8)
    ax_r.set_yticks(range(len(order)), [display[a] for a in order], fontsize=7)
    ax_r.set_xlabel("held-out R$^2$ within author (layer 19)")
    ax_r.set_ylabel("author (PDNC corpus)")

    fig.savefig(FIG_DIR / "author_blocked_folds.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "author_blocked_folds.pdf", bbox_inches="tight")
    comp = payload["folds"]["composition"]
    comp_txt = "; ".join(
        f"fold {c['fold']}: {c['n_rows']} rows / {c['n_authors']} authors" for c in comp
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    meta = dict(payload["metadata"])
    meta.update(
        {
            "git_commit": git_commit,
            "timestamp": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "script": "scripts/issue931_abf_fig_relabel.py",
        }
    )
    (FIG_DIR / "author_blocked_folds.meta.json").write_text(
        json.dumps(
            {
                "metadata": meta,
                "what": (
                    "L19 held-out pooled R^2 of the two Arm-A cells (character map, "
                    "whole-window baseline) under novel-blocked vs author-blocked K=5 folds "
                    "(sweep obs, group-bootstrap obs + 95% CI, shuffle-null p97.5), and the "
                    "per-author held-out R^2 (19 labeled points). Author folds are imbalanced "
                    f"by construction (honest blocking): {comp_txt}."
                ),
                "points": meta_points,
                "relabel_provenance": {
                    "regenerated_from": "eval_results/issue_931/author_blocked_folds.json",
                    "original_driver": "scripts/issue931_author_blocked_folds.py",
                    "original_driver_git_sha": payload["metadata"]["git_commit"],
                    "change": (
                        "labels only — condition shorthand within/ctxmean -> 'character map'/"
                        "'whole-window baseline'; author-code ticks -> PDNC surnames; "
                        "values, panels, bands, CIs, palette unchanged (no refits)"
                    ),
                    "surname_source": "data/issue_931/pdnc/PDNC-Author-Index.csv",
                    "author_code_to_surname": {a: display[a] for a in sorted(per_author)},
                    "unresolved_codes_kept_verbatim": unresolved,
                },
            },
            indent=2,
            default=float,
        )
    )
    plt.close(fig)
    print(f"[i931-abf-relabel] wrote {FIG_DIR / 'author_blocked_folds.png'}", flush=True)
    if unresolved:
        print(f"[i931-abf-relabel] WARNING unresolved author codes kept: {unresolved}", flush=True)


if __name__ == "__main__":
    make_figure()
