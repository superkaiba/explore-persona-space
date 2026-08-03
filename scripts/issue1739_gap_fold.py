"""Harvest + fold the #1739 two-gap fill into the interim writeup.

Gap 2 (`issue1739_maxood/hallucination`): the hallucination max-budget OOD
operating slice for arms 7/8/12 (u_rung=full, budget_l=16,000, regime e1,
context_end) on the nqopen + simpleqa rungs. The committed
``wide_ood/hallucination_transfer.jsonl`` carries those three arms ONLY at
u_rung=250 / L in {250, 2500}, so the maximum-budget OOD panels were missing
them entirely -- a hole in the grid, not a null.

Gap 1 (`issue1739_maxood/bareq_null_diag`): the leg-1 prefix null-probe
mechanism diagnosis. The committed bareq summary predates the scorer's
``null_anomaly_diagnostic`` ladder (``nulls`` is ``[]``), so the anomaly was
carried as "unexplained".

Pure aggregation + rendering: no fits, no GPU, no judge calls. Every number is
re-read from the named artifact in-process. Re-renders ``wide_ood_arms.png``
with the gap-2 cells merged in, and dumps prose-ready aggregates.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

set_paper_style("blog")

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
WT = ROOT / ".claude/worktrees/issue-1739/eval_results/issue_1739"
OUT = ROOT / "figures/issue_1739/interim_writeup"
OUT.mkdir(parents=True, exist_ok=True)
STAGE = ROOT / "data/issue_1739/gapfold"
STAGE.mkdir(parents=True, exist_ok=True)

REPO = hub.DEFAULT_DATASET_REPO
GAP2_PREFIX = "issue1739_maxood/hallucination"
GAP1_PREFIX = "issue1739_maxood/bareq_null_diag"

BEHAVIORS = ["evil", "sycophancy", "hallucination"]
MAXL = {"evil": 8000, "sycophancy": 16000, "hallucination": 16000}
GAP2_ARMS = {"arm7_map_ridge_pred", "arm8_map_ridge_true", "arm12_oracle_reg"}

# ---- palettes (identical indexing to issue1739_final_fold.py) -------------
ARM_ORDER = [
    "arm1_ctx_e1",
    "arm2_ctx_native",
    "arm3_identity_bias",
    "arm4_ridge_ctx",
    "arm5_mlp_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm9_pretrain_ft",
    "arm10_stacked",
    "arm11_oracle_proj",
    "arm12_oracle_reg",
    "arm13_shuffled_map",
    "arm14_shuffled_pt",
    "arm15_text_only",
    "arm16_surface_feat",
]
_cmap = plt.get_cmap("tab20")
ARM_COLOR = {a: _cmap(i % 20) for i, a in enumerate(ARM_ORDER)}
ARM_LABEL = {
    "arm1_ctx_e1": "PV proj. on context (paper method)",
    "arm2_ctx_native": "context-native direction proj. (label-supervised)",
    "arm3_identity_bias": "identity+learned-bias baseline",
    "arm4_ridge_ctx": "direct ridge on context",
    "arm5_mlp_ctx": "direct MLP on context",
    "arm6_map_proj_e1": "map -> PV projection (label-free)",
    "arm7_map_ridge_pred": "map -> ridge on predicted answers",
    "arm8_map_ridge_true": "map -> ridge trained on real answers",
    "arm9_pretrain_ft": "pretrain-then-finetune",
    "arm10_stacked": "stacked context+map features",
    "arm11_oracle_proj": "oracle: PV proj. on TRUE answer",
    "arm12_oracle_reg": "oracle: ridge on TRUE answer",
    "arm13_shuffled_map": "shuffled-map control",
    "arm14_shuffled_pt": "shuffled-pretrain control",
    "arm15_text_only": "text-only features",
    "arm16_surface_feat": "surface features",
}
ROSTER = [
    "arm1_ctx_e1",
    "arm2_ctx_native",
    "arm3_identity_bias",
    "arm4_ridge_ctx",
    "arm5_mlp_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm9_pretrain_ft",
    "arm10_stacked",
    "arm11_oracle_proj",
    "arm12_oracle_reg",
    "arm13_shuffled_map",
]
NEW_ARMS = {"arm5_mlp_ctx", "arm7_map_ridge_pred", "arm8_map_ridge_true", "arm12_oracle_reg"}
OOD_RUNGS = {
    "evil": ["toxicchat", "hhrt"],
    "sycophancy": ["aita"],
    "hallucination": ["nqopen", "simpleqa"],
}

STATS: dict = {}


def finite(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)


def fig_title(fig, title, subtitle):
    fig.suptitle(title, fontsize=13, y=0.985)
    fig.text(0.5, 0.905, subtitle, ha="center", fontsize=8.2, color="#444444")


def stage(prefix: str) -> Path:
    """Scoped staging of one HF prefix (never a full-repo listing).

    ``stage_hub_prefix`` mirrors the repo-relative path under ``dest_dir``, so
    the returned dir is ``dest_dir/<prefix>`` -- return that, not the root.
    """
    dest_dir = STAGE / prefix.replace("/", "__")
    files = hub.stage_hub_prefix(
        repo_id=REPO, prefix=prefix, dest_dir=dest_dir, repo_type="dataset"
    )
    print(f"[stage] {prefix}: {len(files)} files -> {dest_dir}", flush=True)
    mirrored = dest_dir / prefix
    return mirrored if mirrored.exists() else dest_dir


# ================================================== gap 2: max-budget OOD ==
gap2_dir = stage(GAP2_PREFIX)
gap2_rows: list[dict] = []
for p in sorted(gap2_dir.rglob("transfer.jsonl")):
    with open(p) as fh:
        for line in fh:
            rec = json.loads(line)
            gap2_rows.extend(rec.get("rows", []) if isinstance(rec, dict) else [])
print(f"[gap2] {len(gap2_rows)} harvested arm-rows from {gap2_dir}", flush=True)

# Committed wide-OOD grid, then merge the gap-2 rows on top.
ood_cells = defaultdict(list)
ood_layers = defaultdict(set)
ood_src = defaultdict(set)


def absorb(rows, beh, src):
    kept = 0
    for r in rows:
        if (
            r.get("regime") == "e1"
            and r.get("u_rung_label") == "full"
            and r.get("budget_l") == MAXL[beh]
            and r.get("variant") == "context_end"
            and finite(r.get("rho_frozen"))
        ):
            key = (beh, r["eval_rung"], r["arm"])
            ood_cells[key].append(r["rho_frozen"])
            if r.get("layer") is not None:
                ood_layers[key].add(r["layer"])
            ood_src[key].add(src)
            kept += 1
    return kept


for beh in BEHAVIORS:
    rows = []
    with open(WT / "wide_ood" / f"{beh}_transfer.jsonl") as fh:
        for line in fh:
            rows.extend(json.loads(line).get("rows", []))
    absorb(rows, beh, "committed")
n_new = absorb(gap2_rows, "hallucination", "gap2")
print(f"[gap2] {n_new} rows absorbed at the operating slice", flush=True)

filled = sorted(
    k for k in ood_cells if k[0] == "hallucination" and k[2] in GAP2_ARMS and "gap2" in ood_src[k]
)
print(f"[gap2] filled cells: {filled}", flush=True)

panels = [(b, r) for b in BEHAVIORS for r in OOD_RUNGS[b]]
OOD_SLOTS = [a for a in ROSTER if any((b, r, a) in ood_cells for b, r in panels)]
fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 6.4), sharey=True)
for ax, (beh, rung) in zip(np.atleast_1d(axes), panels):
    y = np.arange(len(OOD_SLOTS))
    for i, a in enumerate(OOD_SLOTS):
        v = ood_cells.get((beh, rung, a))
        if not v:
            ax.text(0.0, i, "  not run in this cell", va="center", fontsize=6.4, color="#999999")
            continue
        m = statistics.mean(v)
        sd = statistics.pstdev(v) if len(v) > 1 else 0.0
        STATS.setdefault("wide_ood", {}).setdefault(beh, {}).setdefault(rung, {})[a] = {
            "mean_rho": m,
            "sd": sd,
            "n_reps": len(v),
            "layers": sorted(ood_layers[(beh, rung, a)]),
            "source": sorted(ood_src[(beh, rung, a)]),
        }
        ax.barh(
            i,
            m,
            color=ARM_COLOR[a],
            edgecolor="black" if a in NEW_ARMS else "none",
            linewidth=1.1,
            height=0.72,
        )
        ax.errorbar(
            [m],
            [i],
            xerr=np.array([[max(0.0, sd)], [max(0.0, sd)]]),
            fmt="none",
            ecolor="#333333",
            capsize=2.5,
            elinewidth=1.0,
        )
    ax.axvline(0, color="#666666", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [ARM_LABEL[a] + ("  *new" if a in NEW_ARMS else "") for a in OOD_SLOTS], fontsize=7.5
    )
    ax.invert_yaxis()
    floor = "  (DV floor-censored)" if (beh, rung) == ("evil", "hhrt") else ""
    ax.set_title(f"{beh} / {rung}{floor}", fontsize=10)
    ax.set_xlabel(r"Spearman $\rho$")
fig_title(
    fig,
    "Out-of-distribution transfer with the wide arm roster",
    "Operating slice: E1 persona vector, U = 18,793 unlabeled map pairs, maximum label budget, "
    "context end state.\nBars are the mean over (seed, draw) replicates; error bars are the SD "
    "across those replicates, drawn as non-negative\noffsets. Black outlines mark arms added by "
    "this round. The direct-MLP arm was not run on the OOD grid.",
)
fig.tight_layout(rect=(0, 0, 1, 0.87))
savefig_paper(fig, "wide_ood_arms", dir=OUT)
plt.close(fig)
print("[gap2] wrote wide_ood_arms", flush=True)

# ============================================== gap 1: leg-1 null diagnosis ==
gap1_dir = stage(GAP1_PREFIX)
STATS["null_diag"] = {}
for p in sorted(gap1_dir.rglob("all_arms_spearman.json")):
    beh = next((b for b in BEHAVIORS if b in p.parts), p.parent.name)
    doc = json.loads(p.read_text())
    meta = doc.get("meta", {}) or {}
    # The leg-1 probe lands at meta.leg1_null_probe, a dict keyed by VARIANT
    # (`context_end`, ...); the scorer adds `anomaly_diagnostic` per variant.
    probe = meta.get("leg1_null_probe") or {}
    entry = {"path": str(p.relative_to(gap1_dir)), "variants": {}}
    for name, nv in probe.items() if isinstance(probe, dict) else []:
        if not isinstance(nv, dict):
            continue
        diag = nv.get("anomaly_diagnostic") or {}
        band = diag.get("shuffle_band") or {}
        rec = {
            "base_verdict": nv.get("verdict"),
            "any_ci_excludes_zero": nv.get("any_ci_excludes_zero"),
            "n_finite_rho": nv.get("n_finite_rho"),
            "constant": (nv.get("constancy") or {}).get("constant"),
            "diagnostic_present": bool(diag),
            "verdict": diag.get("verdict"),
            "observed_max_abs_rho": diag.get("observed_max_abs_rho"),
            "inside_shuffle_band": diag.get("inside_shuffle_band"),
            "shuffle_band_p97_5": band.get("abs_rho_p97_5"),
            "shuffle_band_ran": band.get("ran"),
            "shuffle_band_n_seeds": band.get("n_seeds") or band.get("n_shuffles"),
            "capture_source_split": diag.get("capture_source_split"),
            "batch_order_structured": diag.get("batch_order_structured"),
        }
        entry["variants"][name] = rec
        print(
            f"[gap1] {beh}/{name}: verdict={rec['verdict']!r} "
            f"|rho|max={rec['observed_max_abs_rho']} "
            f"band_p97.5={rec['shuffle_band_p97_5']} inside={rec['inside_shuffle_band']}",
            flush=True,
        )
    if not entry["variants"]:
        print(f"[gap1] {beh}: NO leg1_null_probe variants in {p}", flush=True)
    STATS["null_diag"][beh] = entry

dump = Path("/tmp/i1739_gap_stats.json")
dump.write_text(json.dumps(STATS, indent=2, default=str))
print(f"\nwrote {dump}", flush=True)
if not filled:
    print("WARNING: gap-2 filled no operating-slice cells", file=sys.stderr)
if not STATS["null_diag"]:
    print("WARNING: gap-1 produced no null-diagnostic summaries", file=sys.stderr)
