"""#1979 free-analysis follow-up: length-partialled rho for the change + marker champions.

The race's robustness line 3 (scripts/issue1979_race.py::robustness_lines) covered
only the top-3 LEVEL candidates. This follow-up computes content-token-length-
partialled Spearman rho for the two change-race champions:

- per content arm (12): rho(p3b_tc, dv_change) partialled on content_token_len
- per marker arm (6):   rho(p9_k8, dv_dlogp)   partialled on content_token_len

Rank-based partial via the SAME primitive as the race robustness line
(issue1900_race._partial_spearman: Pearson of rank residuals after OLS on the
covariate ranks, ranks via scipy rankdata average ties). Raw rho uses the same
primitive with an empty covariate set (= tie-corrected Spearman), so raw and
partialled reads share one code path. Diagnostics rho(len, dv) and rho(len, x)
are recorded per arm to show how much length correlates with each leg.

Inputs:  eval_results/issue_1979/race/frame_<arm>.json (per-prefix frames, n=50;
         columns content_token_len, p3b_tc, p9_k8, dv_change | dv_dlogp).
Outputs: eval_results/issue_1979/race/length_partial_followup.json
         figures/issue_1979/length_partial_followup.{png,pdf,meta.json}
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before numpy: shared-VM thread caps

import datetime as dt  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402

import issue1900_race as R  # noqa: E402  (_ranks_np + _partial_spearman reused)

RACE_DIR = REPO_ROOT / "eval_results/issue_1979/race"
FIG_DIR = REPO_ROOT / "figures/issue_1979"
OUT_PATH = RACE_DIR / "length_partial_followup.json"

# The two change-race champions under test, keyed by arm kind.
KIND_SPEC = {
    "content": {"dv": "dv_change", "x": "p3b_tc"},
    "marker": {"dv": "dv_dlogp", "x": "p9_k8"},
}


def _meta() -> dict:
    import scipy

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "script": "scripts/issue1979_length_partial_followup.py",
        "issue": 1979,
        "git_commit": commit,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "convention": (
            "issue1900_race._partial_spearman on scipy average ranks; raw rho = same "
            "primitive with empty covariate set (tie-corrected Spearman)"
        ),
    }


def _load_arms() -> list[dict]:
    paths = sorted(RACE_DIR.glob("frame_*.json"))
    if not paths:
        raise FileNotFoundError(f"no frame_*.json under {RACE_DIR}")
    arms = []
    for p in paths:
        d = json.loads(p.read_text())
        f = d["frame"]
        arm_id = p.stem.removeprefix("frame_")
        has_change = "dv_change" in f
        has_dlogp = "dv_dlogp" in f
        if has_change == has_dlogp:
            raise ValueError(f"{p.name}: expected exactly one of dv_change / dv_dlogp")
        kind = "content" if has_change else "marker"
        if arm_id.startswith("mk-") != (kind == "marker"):
            raise ValueError(f"{p.name}: arm-id prefix vs DV-column kind mismatch ({kind})")
        arms.append({"arm_id": arm_id, "kind": kind, "frame": f, "source": p.name})
    return arms


def _col(f: dict, name: str) -> np.ndarray:
    return np.asarray([v if v is not None else np.nan for v in f[name]], dtype=np.float64)


def arm_reads(arm: dict) -> dict:
    spec = KIND_SPEC[arm["kind"]]
    f = arm["frame"]
    dv, x, ln = _col(f, spec["dv"]), _col(f, spec["x"]), _col(f, "content_token_len")
    ok = np.isfinite(dv) & np.isfinite(x) & np.isfinite(ln)
    ranks = {
        "dv": R._ranks_np(dv[ok]),
        "x": R._ranks_np(x[ok]),
        "len": R._ranks_np(ln[ok]),
    }
    return {
        "arm_id": arm["arm_id"],
        "kind": arm["kind"],
        "predictor": spec["x"],
        "dv": spec["dv"],
        "n_total": int(dv.size),
        "n_used": int(ok.sum()),
        "rho_raw": R._partial_spearman(ranks, "dv", "x", []),
        "rho_partial_len": R._partial_spearman(ranks, "dv", "x", ["len"]),
        "rho_len_dv": R._partial_spearman(ranks, "dv", "len", []),
        "rho_len_x": R._partial_spearman(ranks, "x", "len", []),
    }


def _median(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(xs)) if xs else None


def _summaries(reads: list[dict]) -> dict:
    out = {}
    for kind in ("content", "marker"):
        grp = [r for r in reads if r["kind"] == kind]
        deltas = [
            r["rho_partial_len"] - r["rho_raw"]
            for r in grp
            if r["rho_partial_len"] is not None and r["rho_raw"] is not None
        ]
        out[kind] = {
            "n_arms": len(grp),
            "predictor": KIND_SPEC[kind]["x"],
            "dv": KIND_SPEC[kind]["dv"],
            "rho_raw_median": _median([r["rho_raw"] for r in grp]),
            "rho_partial_len_median": _median([r["rho_partial_len"] for r in grp]),
            "delta_median": _median(deltas),
        }
    return out


def _figure(reads: list[dict]) -> None:
    import matplotlib.pyplot as plt

    import issue1979_figs as F  # plain-English rendered-text labels (paper-plots §3.5)
    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    c_raw = paper_palette_role("baseline")
    c_par = paper_palette_role("primary")
    groups = [
        ("content", "content arms: through-map predicted-answer similarity vs leakage change"),
        (
            "marker",
            "marker arms: nearest-training-rows context similarity vs marker log-prob change",
        ),
    ]
    n_rows = [sum(1 for r in reads if r["kind"] == k) for k, _ in groups]
    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 6.0), sharex=True, height_ratios=[n_rows[0], n_rows[1]]
    )
    for ax, (kind, title) in zip(axes, groups, strict=True):
        grp = [r for r in reads if r["kind"] == kind]
        ys = np.arange(len(grp))[::-1]
        for y, r in zip(ys, grp, strict=True):
            ax.plot([r["rho_raw"], r["rho_partial_len"]], [y, y], color="0.6", lw=1.2, zorder=1)
        ax.scatter([r["rho_raw"] for r in grp], ys, color=c_raw, zorder=2, label="raw Spearman")
        ax.scatter(
            [r["rho_partial_len"] for r in grp],
            ys,
            color=c_par,
            zorder=3,
            label="length-partialled",
        )
        ax.set_yticks(ys)
        ax.set_yticklabels([F._arm_label(r["arm_id"]) for r in grp], fontsize=7)
        ax.axvline(0.0, color="0.8", lw=0.8, zorder=0)
        ax.set_title(title, fontsize=9)
    axes[0].legend(loc="lower right", fontsize=7)
    axes[1].set_xlabel("Spearman rho (raw vs content-token-length-partialled)")
    fig.tight_layout()
    savefig_paper(fig, "length_partial_followup", dir=FIG_DIR)
    plt.close(fig)


def main() -> None:
    arms = _load_arms()
    reads = [arm_reads(a) for a in arms]
    payload = {
        "meta": _meta(),
        "arms": reads,
        "medians": _summaries(reads),
        "sources": [a["source"] for a in arms],
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[out] {OUT_PATH.relative_to(REPO_ROOT)}")
    hdr = f"{'arm_id':<28} {'kind':<8} {'raw':>8} {'partial':>8} {'delta':>8}"
    print(hdr)
    for r in reads:
        d = r["rho_partial_len"] - r["rho_raw"]
        print(
            f"{r['arm_id']:<28} {r['kind']:<8} {r['rho_raw']:>8.3f} "
            f"{r['rho_partial_len']:>8.3f} {d:>+8.3f}"
        )
    for kind, s in payload["medians"].items():
        print(
            f"[median] {kind} (n={s['n_arms']}): raw={s['rho_raw_median']:.3f} "
            f"partial={s['rho_partial_len_median']:.3f} delta={s['delta_median']:+.3f}"
        )
    _figure(reads)
    print(f"[fig] {FIG_DIR.relative_to(REPO_ROOT)}/length_partial_followup.png")


if __name__ == "__main__":
    main()
