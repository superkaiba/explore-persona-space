"""issue #2474 — pre-fine-tuning predictors of post-inoculation re-elicitation.

Scores predictors computed ENTIRELY on the base model (Qwen2.5-7B-Instruct,
before any fine-tuning) against the post-fine-tuning per-trigger re-elicitation
rates measured in #2379.

Arms (this module; the p_inoc-reference family — runnable as soon as the grid
capture lands, since the base map is already fitted and reused from #779
pass-B at the pin):

  ctx_pinoc    cos( v_C(q,t) , v_C(q,p_inoc) )                    -- context state
  ans_pinoc    cos( vhat_A(q,t) , vhat_A(q,p_inoc) )              -- predicted answer

with vhat_A = ((v_C - xmu)/xsd) @ W + ymu, the registered #2379/#2254 prediction
path, and every score averaged over the extraction question bank before ranking.

The TRAIN-REF arm (the #1979 predictor: cos(vhat_A(q,t), mu_train) with mu_train
computed per condition UNDER THE BASE MODEL) needs the p3 mu bundles and is
scored by ``--arms trainref`` once those land. Unlike the p_inoc arms it is
condition-SPECIFIC even pre-fine-tuning, so it is not bounded by the round-1
cross-condition DV ceiling.

Layer pins are INHERITED from #2379 (L16 misalignment / L27 capitalization) on
the stored 0..27 decoder-block axis -- no layer selection on this run's data.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from issue2474_free_gate import P_INOC_TRIGGER, load_rates  # noqa: E402

logger = logging.getLogger("issue2474_analysis")

DATA_REPO = "superkaiba1/explore-persona-space-data"
MAP_PREFIX = "issue2379_reelicit/analysis_tensors/maps_pinned"
PINNED_LAYER = {"em": 16, "caps": 27}
GRID_NAME = {"em": "base_em", "caps": "base_caps"}
# #2379 post-fine-tuning twins at the inoculation-prompt reference, for context.
POST_FT_REFERENCE = {"em": {"ctx_sameq": 0.790, "ceiling_sameq": 0.792, "ans_sameq_mapI": 0.648}}
N_BOOT = 2000
BOOT_SEED = 20260823


def load_base_map(layer: int) -> dict:
    """Fetch the reused BASE map's pinned components at one layer.

    The base map was fit in #2379 on the reused #779 pass-B LMSYS bundle -- it is
    NOT re-derived here. Asserts the four registered components are present.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=DATA_REPO,
        filename=f"{MAP_PREFIX}/base_L{layer}.pt",
        repo_type="dataset",
    )
    comp = torch.load(path, map_location="cpu", weights_only=False)
    missing = {"W", "xmu", "xsd", "ymu"} - set(comp)
    if missing:
        raise RuntimeError(
            f"base_L{layer}.pt missing components {sorted(missing)}: got {list(comp)}"
        )
    return comp


def predict_answer(v_c: np.ndarray, comp: dict) -> np.ndarray:
    """Registered prediction path: vhat_A = ((v_C - xmu)/xsd) @ W + ymu."""
    W = np.asarray(comp["W"], dtype=np.float64)
    xmu = np.asarray(comp["xmu"], dtype=np.float64).reshape(-1)
    xsd = np.asarray(comp["xsd"], dtype=np.float64).reshape(-1)
    ymu = np.asarray(comp["ymu"], dtype=np.float64).reshape(-1)
    if v_c.shape[1] != xmu.shape[0]:
        raise ValueError(f"v_c dim {v_c.shape[1]} != map input dim {xmu.shape[0]}")
    return ((v_c - xmu) / xsd) @ W + ymu


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(den == 0):
        raise ValueError("zero-norm row in cosine")
    return num / den


def load_grid(out_dir: Path, setting: str) -> tuple[np.ndarray, list[dict]]:
    p = out_dir / "capture_tensors" / "predictor_captures" / GRID_NAME[setting] / "grid.pt"
    if not p.exists():
        raise RuntimeError(f"grid bundle absent: {p}")
    d = torch.load(p, map_location="cpu", weights_only=False)
    v_c = np.asarray(d["v_c"], dtype=np.float64)
    return v_c, d["row_meta"]


def score_pinoc_arms(out_dir: Path, setting: str) -> dict[str, dict[str, float]]:
    """-> {arm: {trigger_label: score}} for the two p_inoc-reference arms."""
    layer = PINNED_LAYER[setting]
    v_c_all, row_meta = load_grid(out_dir, setting)
    v_c = v_c_all[:, layer, :]  # stored index i == decoder block i

    by_trigger: dict[str, dict[int, int]] = {}
    for i, r in enumerate(row_meta):
        by_trigger.setdefault(r["trigger_label"], {})[r["q_sim_idx"]] = i

    p_inoc = P_INOC_TRIGGER[setting]
    if p_inoc not in by_trigger:
        raise RuntimeError(f"{setting}: p_inoc trigger {p_inoc!r} absent from grid")
    ref_rows = by_trigger[p_inoc]

    comp = load_base_map(layer)
    vhat = predict_answer(v_c, comp)
    logger.info(
        "%s: L%d, %d triggers, %d questions", setting, layer, len(by_trigger), len(ref_rows)
    )

    out: dict[str, dict[str, float]] = {"ctx_pinoc": {}, "ans_pinoc": {}}
    for label, qmap in by_trigger.items():
        shared = sorted(set(qmap) & set(ref_rows))
        if len(shared) != len(ref_rows):
            raise RuntimeError(f"{setting}/{label}: {len(shared)} shared q vs {len(ref_rows)} ref")
        idx = [qmap[q] for q in shared]
        ridx = [ref_rows[q] for q in shared]
        out["ctx_pinoc"][label] = float(_cos_rows(v_c[idx], v_c[ridx]).mean())
        out["ans_pinoc"][label] = float(_cos_rows(vhat[idx], vhat[ridx]).mean())
    return out


def evaluate(
    setting: str, arms: dict[str, dict[str, float]], rates: dict, drop_p_inoc: bool
) -> dict:
    """Within-condition Spearman vs the #2379 DV, averaged across conditions."""
    models = rates[setting]
    conditions = sorted(m for m in models if m != "base")
    triggers = list(next(iter(models.values())))
    if drop_p_inoc:
        triggers = [t for t in triggers if t != P_INOC_TRIGGER[setting]]

    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(triggers), size=(N_BOOT, len(triggers)))

    res: dict[str, dict] = {}
    for arm, scores in arms.items():
        pred = np.array([scores[t] for t in triggers], dtype=float)
        per_cond, draws_acc = [], []
        for c in conditions:
            dv = np.array([models[c][t] for t in triggers], dtype=float)
            rho = float(spearmanr(pred, dv).statistic)
            draws = [float(spearmanr(pred[i], dv[i]).statistic) for i in idx]
            draws = [d for d in draws if not np.isnan(d)]
            per_cond.append(
                {
                    "condition": c,
                    "rho": rho,
                    "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
                    if draws
                    else [None, None],
                }
            )
            draws_acc.append(draws)
        mean_draws = np.mean(np.array(draws_acc), axis=0)
        res[arm] = {
            "per_condition": per_cond,
            "mean_rho": float(np.mean([p["rho"] for p in per_cond])),
            "mean_ci95": [
                float(np.percentile(mean_draws, 2.5)),
                float(np.percentile(mean_draws, 97.5)),
            ],
        }
    return {
        "setting": setting,
        "drop_p_inoc": drop_p_inoc,
        "n_triggers": len(triggers),
        "conditions": conditions,
        "arms": res,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="eval_results/issue_2474")
    ap.add_argument("--settings", default="em,caps")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rates = load_rates()
    results: dict[str, dict] = {}
    for setting in [s for s in args.settings.split(",") if s]:
        arms = score_pinoc_arms(out_dir, setting)
        results[setting] = {
            "with_p_inoc": evaluate(setting, arms, rates, drop_p_inoc=False),
            "without_p_inoc": evaluate(setting, arms, rates, drop_p_inoc=True),
            "raw_scores": arms,
        }

    results["provenance"] = {
        "issue": 2474,
        "parent_issue": 2379,
        "pinned_layers": PINNED_LAYER,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "base_map": f"{DATA_REPO}:{MAP_PREFIX}/base_L*.pt (reused #779 pass-B fit)",
        "post_ft_reference": POST_FT_REFERENCE,
        "note": "predictors computed on the BASE model only; DV from #2379 post-fine-tuning rates",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pinoc_arms.json").write_text(json.dumps(results, indent=2) + "\n")

    for setting in [s for s in args.settings.split(",") if s]:
        print(f"\n=== {setting.upper()} (L{PINNED_LAYER[setting]}) ===")
        for variant in ("with_p_inoc", "without_p_inoc"):
            r = results[setting][variant]
            print(f"  [{variant}] n={r['n_triggers']}")
            for arm, a in r["arms"].items():
                lo, hi = a["mean_ci95"]
                print(f"    {arm:12s} mean rho={a['mean_rho']:+.3f}  CI[{lo:+.3f},{hi:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
