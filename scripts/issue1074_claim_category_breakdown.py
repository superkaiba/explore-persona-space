"""Claim-category breakdown of the sycophancy yield-floor failure (issue #1074).

Free-analysis follow-up: does the sycophancy positive-yield floor failure
concentrate in particular CATEGORIES of wrong-fact claims? Joins the per-question
judged/kept counts from ``eval_results/issue_1074/yield_summary.json`` (both
generator arms) with a DETERMINISTIC category assignment over the 25-claim
sycophancy train bank (``BEHAVIORS["sycophancy"].train_question_bank``).

Category scheme (assigned from the claim TEXTS alone; committed here so the
mapping is deterministic and auditable):

* ``domain`` — one of five: ``physical-science`` / ``biology`` /
  ``history-geography`` / ``mathematics`` / ``technology``, by the subject
  matter of the false claim.
* ``hardness`` — two levels:
  - ``plausible-myth``: the claim is (a) a widely documented common
    misconception or everyday lay confusion (Great Wall visible from
    space/moon, Napoleon's height, glass-is-a-liquid, lightning never strikes
    twice, 10%-of-brain, penguins-in-the-Arctic, spiders-are-insects,
    dolphins-are-fish), OR (b) a near-miss distortion of a true fact — the
    claim becomes true under a one-word / one-qualifier correction a lay
    reader might not notice (Earth "flat" vs flattened at the poles; Python
    "compiled" — true of bytecode, false of "just like C++"; elephants
    largest land animals "ever" vs alive today).
  - ``flat-contradiction``: plainly contradicts an elementary textbook fact
    with no circulating misconception behind it (Everest in Japan, WWI ended
    1955, sqrt(144)=11, ...).

Observational only: per-category ns are tiny (1-9 claims / 1-22 completions
per category x arm) — the JSON carries raw counts and ns, no significance
tests.

Inputs (all existing; no new generation / training / model calls):
* eval_results/issue_1074/yield_summary.json  (committed; per-question judged/kept)
* the in-repo sycophancy train bank (benign false-but-benign factual claims)

Outputs:
* eval_results/issue_1074/claim_category_breakdown.json
* figures/issue_1074/claim_category_yield.{png,pdf,meta.json}

Run from the issue-1074 worktree root:
    uv run python scripts/issue1074_claim_category_breakdown.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402

WT = Path(__file__).resolve().parents[1]
EV = WT / "eval_results" / "issue_1074"
FIGDIR = WT / "figures" / "issue_1074"
OUT_JSON = EV / "claim_category_breakdown.json"
FIG_STEM = "claim_category_yield"

# Pin the exact bank content this mapping was authored against. A regenerated /
# reordered bank must fail LOUD here rather than silently mis-categorize.
BANK_SHA256 = "327affaa3b55026fa4917fba0b33e3309990ed99ada5918a6b8311634f4478a1"

DOMAINS = ("physical-science", "biology", "history-geography", "mathematics", "technology")
HARDNESS = ("plausible-myth", "flat-contradiction")

# Deterministic per-claim categories, keyed by train-bank id suffix
# (sycophancy-trainq-<suffix>). ``gloss`` strings match scripts/issue1074_figures.py
# where that script defined one (realized claims); the 6 never-realized claims get
# glosses in the same style. ``rationale`` documents the hardness call per claim.
CLAIM_CATEGORIES: dict[str, dict[str, str]] = {
    "0000": {
        "gloss": "Great Wall (moon)",
        "domain": "history-geography",
        "hardness": "plausible-myth",
        "rationale": "documented misconception (Great Wall visible from space/moon)",
    },
    "0001": {
        "gloss": "Napoleon tall",
        "domain": "history-geography",
        "hardness": "plausible-myth",
        "rationale": "Napoleon-height myth family (the 'he wasn't short' correction meme)",
    },
    "0002": {
        "gloss": "Everest in Japan",
        "domain": "history-geography",
        "hardness": "flat-contradiction",
        "rationale": "contradicts elementary geography (Nepal/China border)",
    },
    "0003": {
        "gloss": "Edison telephone",
        "domain": "technology",
        "hardness": "flat-contradiction",
        "rationale": "invention attribution error (Bell); no circulating myth",
    },
    "0004": {
        "gloss": "water boils 50°C",
        "domain": "physical-science",
        "hardness": "flat-contradiction",
        "rationale": "contradicts elementary science (100°C at sea level)",
    },
    "0005": {
        "gloss": "flat-poles Earth",
        "domain": "physical-science",
        "hardness": "plausible-myth",
        "rationale": "near-miss of the true oblate-spheroid fact (flattened, not flat, at poles)",
    },
    "0006": {
        "gloss": "spiders insects",
        "domain": "biology",
        "hardness": "plausible-myth",
        "rationale": "everyday lay confusion (arachnids vs insects)",
    },
    "0007": {
        "gloss": "8-chamber heart",
        "domain": "biology",
        "hardness": "flat-contradiction",
        "rationale": "contradicts elementary biology (4 chambers); no circulating myth",
    },
    "0008": {
        "gloss": "Australia largest",
        "domain": "history-geography",
        "hardness": "flat-contradiction",
        "rationale": "direct inversion (Australia is the smallest continent)",
    },
    "0009": {
        "gloss": "Nile through China",
        "domain": "history-geography",
        "hardness": "flat-contradiction",
        "rationale": "contradicts elementary geography; no circulating myth",
    },
    "0010": {
        "gloss": "dolphins fish",
        "domain": "biology",
        "hardness": "plausible-myth",
        "rationale": "everyday lay confusion (marine mammals vs fish)",
    },
    "0011": {
        "gloss": "WWI ended 1955",
        "domain": "history-geography",
        "hardness": "flat-contradiction",
        "rationale": "contradicts elementary history (1918); no circulating myth",
    },
    "0012": {
        "gloss": "Toronto capital",
        "domain": "history-geography",
        "hardness": "flat-contradiction",
        "rationale": "contradicts elementary civics (Ottawa); common quiz error but "
        "not a circulated belief asserted as fact",
    },
    "0013": {
        "gloss": "lightning myth",
        "domain": "physical-science",
        "hardness": "plausible-myth",
        "rationale": "documented misconception (lightning never strikes the same place twice)",
    },
    "0014": {
        "gloss": "sun orbits Earth",
        "domain": "physical-science",
        "hardness": "flat-contradiction",
        "rationale": "contradicts elementary astronomy; not a modern circulated belief",
    },
    "0015": {
        "gloss": "Python compiled",
        "domain": "technology",
        "hardness": "plausible-myth",
        "rationale": "near-miss (CPython compiles to bytecode; 'just like C++' is the false part)",
    },
    "0016": {
        "gloss": "10% of brain",
        "domain": "biology",
        "hardness": "plausible-myth",
        "rationale": "documented misconception (humans use 10% of their brain)",
    },
    "0017": {
        "gloss": "salt is H2O",
        "domain": "physical-science",
        "hardness": "flat-contradiction",
        "rationale": "contradicts elementary chemistry (NaCl); no circulating myth",
    },
    "0018": {
        "gloss": "Arctic penguins",
        "domain": "biology",
        "hardness": "plausible-myth",
        "rationale": "everyday lay confusion (penguins-and-polar-bears trope; Antarctic)",
    },
    "0019": {
        "gloss": "Shakespeare French",
        "domain": "history-geography",
        "hardness": "flat-contradiction",
        "rationale": "contradicts elementary literary history; no circulating myth",
    },
    "0020": {
        "gloss": "sqrt(144) = 11",
        "domain": "mathematics",
        "hardness": "flat-contradiction",
        "rationale": "arithmetic error (12); no circulating myth",
    },
    "0021": {
        "gloss": "Golden Gate NYC",
        "domain": "history-geography",
        "hardness": "flat-contradiction",
        "rationale": "contradicts elementary geography (San Francisco); no circulating myth",
    },
    "0022": {
        "gloss": "glass flows",
        "domain": "physical-science",
        "hardness": "plausible-myth",
        "rationale": "documented misconception (glass is a slow-moving liquid / old windows)",
    },
    "0023": {
        "gloss": "elephants largest",
        "domain": "biology",
        "hardness": "plausible-myth",
        "rationale": "near-miss (largest land animals ALIVE TODAY is true; 'ever' is the "
        "false part vs sauropods)",
    },
    "0024": {
        "gloss": "Alaska southernmost",
        "domain": "history-geography",
        "hardness": "flat-contradiction",
        "rationale": "direct inversion (northernmost); no circulating myth",
    },
}

ARMS = ("base", "ablit")
ARM_LABEL = {"base": "base Qwen generator", "ablit": "abliterated Qwen generator"}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=WT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def load_bank() -> dict[str, str]:
    """Return {id-suffix: claim text} for the 25-claim sycophancy train bank."""
    bank = BEHAVIORS["sycophancy"].train_question_bank
    digest = hashlib.sha256("\n".join(bank).encode()).hexdigest()
    if digest != BANK_SHA256:
        raise RuntimeError(
            "sycophancy train bank content drifted from the bank this category mapping "
            f"was authored against (sha256 {digest} != pinned {BANK_SHA256}); "
            "re-audit CLAIM_CATEGORIES before rerunning."
        )
    if len(bank) != len(CLAIM_CATEGORIES):
        raise RuntimeError(f"bank has {len(bank)} claims, mapping has {len(CLAIM_CATEGORIES)}")
    return {f"{i:04d}": q for i, q in enumerate(bank)}


def load_yields() -> dict[str, dict[str, dict[str, int]]]:
    """Return {arm: {id-suffix: {judged, kept}}} for both sycophancy arms."""
    summary = json.loads((EV / "yield_summary.json").read_text())
    out: dict[str, dict[str, dict[str, int]]] = {}
    for arm in ARMS:
        cell = summary["cells"][f"sycophancy-{arm}"]
        per_q = {
            qid.rsplit("-", 1)[-1]: {"judged": int(v["judged"]), "kept": int(v["kept"])}
            for qid, v in cell["per_question_yield"].items()
        }
        kept_total = sum(v["kept"] for v in per_q.values())
        kept_pos = int(cell["yield_record"]["kept_pos"])
        if kept_total != kept_pos:
            raise RuntimeError(
                f"arm {arm}: per-question kept sum {kept_total} != yield_record.kept_pos "
                f"{kept_pos} — yield_summary.json inconsistent"
            )
        out[arm] = per_q
    return out


def aggregate(
    claims: list[dict],
    axis: str,
    levels: tuple[str, ...],
) -> dict[str, dict]:
    """Per-category ns + per-arm judged/kept totals along ``axis`` (domain|hardness)."""
    agg: dict[str, dict] = {}
    for level in levels:
        rows = [c for c in claims if c[axis] == level]
        realized = [c for c in rows if c["realized"]]
        entry: dict[str, object] = {
            "n_claims_bank": len(rows),
            "n_claims_realized": len(realized),
            "per_arm": {},
        }
        for arm in ARMS:
            judged = sum(c["per_arm"][arm]["judged"] for c in realized)
            kept = sum(c["per_arm"][arm]["kept"] for c in realized)
            entry["per_arm"][arm] = {
                "judged": judged,
                "kept": kept,
                "kept_rate": round(kept / judged, 4) if judged else None,
            }
        agg[level] = entry
    return agg


def build_records() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    bank = load_bank()
    yields = load_yields()
    realized_ids = set(yields["base"]) | set(yields["ablit"])
    unknown = realized_ids - set(CLAIM_CATEGORIES)
    if unknown:
        raise RuntimeError(f"realized question ids missing from CLAIM_CATEGORIES: {unknown}")

    claims: list[dict] = []
    for suffix in sorted(CLAIM_CATEGORIES):
        cat = CLAIM_CATEGORIES[suffix]
        per_arm = {arm: yields[arm].get(suffix, {"judged": 0, "kept": 0}) for arm in ARMS}
        claims.append(
            {
                "question_id": f"sycophancy-trainq-{suffix}",
                "claim_text": bank[suffix],
                "gloss": cat["gloss"],
                "domain": cat["domain"],
                "hardness": cat["hardness"],
                "hardness_rationale": cat["rationale"],
                "realized": suffix in realized_ids,
                "per_arm": per_arm,
            }
        )
    by_domain = aggregate(claims, "domain", DOMAINS)
    by_hardness = aggregate(claims, "hardness", HARDNESS)
    return claims, by_domain, by_hardness


# ── Figure ───────────────────────────────────────────────────────────────────


def _counts_panel(
    ax: plt.Axes,
    agg: dict[str, dict],
    levels: tuple[str, ...],
    col: dict[str, str],
    title: str,
) -> None:
    """Grouped bars: light bar = completions judged, solid overlay = kept, per arm."""
    x = np.arange(len(levels))
    width = 0.38
    for i, arm in enumerate(ARMS):
        off = (i - 0.5) * width
        judged = [agg[lv]["per_arm"][arm]["judged"] for lv in levels]
        kept = [agg[lv]["per_arm"][arm]["kept"] for lv in levels]
        ax.bar(x + off, judged, width=width * 0.92, color=col[arm], alpha=0.30, zorder=2)
        ax.bar(
            x + off,
            kept,
            width=width * 0.92,
            color=col[arm],
            alpha=1.0,
            zorder=3,
            label=ARM_LABEL[arm],
        )
        for xi, (j, k) in zip(x + off, zip(judged, kept, strict=True), strict=True):
            ax.text(xi, j + 0.3, f"{k}/{j}", ha="center", va="bottom", fontsize=8)
    tick_labels = [
        f"{lv.replace('-', ' ')}\n{agg[lv]['n_claims_realized']}/{agg[lv]['n_claims_bank']} claims"
        for lv in levels
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("completions")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(
    claims: list[dict], by_domain: dict[str, dict], by_hardness: dict[str, dict]
) -> plt.Figure:
    set_paper_style("blog")
    pal = paper_palette_blog(6)
    col = {"base": pal[0], "ablit": pal[1]}  # same arm colors as issue1074_figures.py

    fig = plt.figure(figsize=(12.5, 10.0), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.5])
    ax_dom = fig.add_subplot(gs[0, 0])
    ax_hard = fig.add_subplot(gs[0, 1])
    ax_claim = fig.add_subplot(gs[1, :])

    _counts_panel(ax_dom, by_domain, DOMAINS, col, "by claim domain")
    _counts_panel(ax_hard, by_hardness, HARDNESS, col, "by claim hardness")
    ax_dom.legend(frameon=False, fontsize=9, loc="upper right")

    # Per-claim underlying data: realized claims only, grouped hardness -> domain -> id.
    realized = [c for c in claims if c["realized"]]
    realized.sort(
        key=lambda c: (
            HARDNESS.index(c["hardness"]),
            DOMAINS.index(c["domain"]),
            c["question_id"],
        )
    )
    y = np.arange(len(realized))
    height = 0.38
    for i, arm in enumerate(ARMS):
        off = (i - 0.5) * height
        judged = [c["per_arm"][arm]["judged"] for c in realized]
        kept = [c["per_arm"][arm]["kept"] for c in realized]
        ax_claim.barh(y + off, judged, height=height * 0.9, color=col[arm], alpha=0.30, zorder=2)
        ax_claim.barh(y + off, kept, height=height * 0.9, color=col[arm], alpha=1.0, zorder=3)
        for yi, (j, k) in zip(y + off, zip(judged, kept, strict=True), strict=True):
            ax_claim.text(j + 0.06, yi, f"{k}/{j}", va="center", fontsize=7.5)
    labels = [
        f"{c['gloss']} — {c['domain'].replace('-', ' ')}, {c['hardness'].replace('-', ' ')}"
        for c in realized
    ]
    ax_claim.set_yticks(y)
    ax_claim.set_yticklabels(labels, fontsize=8)
    ax_claim.invert_yaxis()
    ax_claim.set_xlabel("completions (light bar = judged, solid = kept as sycophantic)")
    ax_claim.set_title(
        "per wrong-fact claim (kept / judged annotated; 6 never-sampled bank claims omitted)",
        fontsize=11,
    )
    ax_claim.spines[["top", "right"]].set_visible(False)

    n_myth = by_hardness["plausible-myth"]
    fig.suptitle(
        "Sycophancy positive yield by wrong-fact claim category (issue #1074)\n"
        f"kept/judged — plausible-myth claims: base "
        f"{n_myth['per_arm']['base']['kept']}/{n_myth['per_arm']['base']['judged']}, "
        f"abliterated {n_myth['per_arm']['ablit']['kept']}/{n_myth['per_arm']['ablit']['judged']}",
        fontsize=13,
    )
    return fig


CAPTION = (
    "Sycophancy positive yield per wrong-fact claim category, both generator arms "
    "(base vs abliterated Qwen2.5-7B-Instruct). Light bars: completions judged; solid "
    "bars: completions kept as sycophantic by the judge. Top: aggregated by claim "
    "domain (left) and by claim hardness (right; plausible-myth = documented common "
    "misconception or near-miss of a true fact, flat-contradiction = plainly "
    "contradicts an elementary fact). Bottom: per-claim underlying counts for the 19 "
    "claims the datagen sampler realized (6 bank claims were never sampled). "
    "Observational read on tiny ns (1-9 claims per category): kept completions "
    "concentrate on plausible-myth claims (base 8/22 vs 2/14 on flat-contradictions; "
    "abliterated 7/22 vs 1/14), and within the myth category a single claim - the "
    "glass-is-a-liquid myth - carries base 4/5 and abliterated 5/5, i.e. most of the "
    "myth-category mass. No significance tests; counts only."
)


def main() -> None:
    claims, by_domain, by_hardness = build_records()

    fig = make_figure(claims, by_domain, by_hardness)
    written = savefig_paper(fig, FIG_STEM, dir=FIGDIR)
    plt.close(fig)
    # Splice a factual caption into the meta sidecar (savefig_paper writes provenance
    # + per-point data; the caption rides alongside).
    meta_path = written["meta"]
    meta = json.loads(meta_path.read_text())
    meta["caption"] = CAPTION
    meta_path.write_text(json.dumps(meta, indent=2))

    payload = {
        "issue": 1074,
        "analysis": "claim_category_breakdown",
        "git_commit": _git_commit(),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "inputs": {
            "yield_summary": "eval_results/issue_1074/yield_summary.json",
            "train_bank": "BEHAVIORS['sycophancy'].train_question_bank",
            "train_bank_sha256": BANK_SHA256,
        },
        "category_scheme": {
            "domains": list(DOMAINS),
            "hardness_levels": list(HARDNESS),
            "hardness_definition": (
                "plausible-myth = documented common misconception / everyday lay "
                "confusion, OR a near-miss distortion of a true fact (true under a "
                "one-word/one-qualifier correction); flat-contradiction = plainly "
                "contradicts an elementary textbook fact with no circulating "
                "misconception behind it. Per-claim rationale strings recorded below."
            ),
        },
        "caveats": (
            "Observational; category ns are tiny (1-9 claims, 1-22 judged completions "
            "per category x arm) and per-claim judged counts vary (1-5) because the "
            "datagen sampler distributed 36 requests unevenly over 19 of 25 bank "
            "claims. Raw counts only; no significance tests."
        ),
        "by_domain": by_domain,
        "by_hardness": by_hardness,
        "claims": claims,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    # Console digest.
    print(f"wrote {OUT_JSON.relative_to(WT)}")
    for fmt, path in written.items():
        print(f"wrote {path.relative_to(WT)} ({fmt})")
    for axis_name, agg in (("domain", by_domain), ("hardness", by_hardness)):
        print(f"\nby {axis_name}:")
        for level, entry in agg.items():
            arms = "  ".join(
                f"{arm}: {entry['per_arm'][arm]['kept']}/{entry['per_arm'][arm]['judged']}"
                for arm in ARMS
            )
            print(
                f"  {level:<18} ({entry['n_claims_realized']}/{entry['n_claims_bank']} "
                f"claims sampled)  {arms}"
            )


if __name__ == "__main__":
    main()
