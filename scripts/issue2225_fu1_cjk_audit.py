"""Language-intrusion (CJK) audit + excluded-intrusion recount for the #2225 fu1 round.

Step 3.7 duty (analyzer): Qwen-family completions under a non-CJK eval owe a per-arm
CJK scan over every judged pool that a verdict rests on, plus zeroed/excluded-intrusion
recounts of the adjudication-bearing statistics.  Pure counting — no completion text is
ever printed; only aggregate counts and recomputed statistics leave this script.

Scope:
  (a) all 96 fu1 rollout units (80 trait-eval + 16 narrow-domain) — per-unit
      intruded/total counts over the regex class
      [\\u4e00-\\u9fff\\u3400-\\u4dbf\\uf900-\\ufaff\\u3040-\\u30ff\\uac00-\\ud7af];
  (b) the 6 parent anchor units consumed by the H2 contrasts
      (A/C at their matched-coherence operating points, evil/sycophancy/hallucination);
  (c) excluded-intrusion recount of every computable frozen contrast in
      analysis/contrasts.json (H1 dose, H2 vs parent C, H2 vs parent A), rebuilt from
      the per-question ``rollout_scores`` matrices with intruded (question, rollout)
      entries masked, question-paired bootstrap (10,000 resamples, seed 2225).

The recount seed stream is a single seed (2225), not the driver's per-contrast offsets,
so CI endpoints differ in the third decimal from contrasts.json; verdict flips are read
against the committed labels.

Usage (defaults match the fu1 VM staging layout):
  uv run python scripts/issue2225_fu1_cjk_audit.py \
      --fu-rc-root /mnt/eps-data/thomasjiralerspong/issue2225_fu1/raw_completions \
      --parent-rc-dir /mnt/eps-data/thomasjiralerspong/issue2225_fu1/hf_dl/parent_anchor_rc/issue2225_ctxsteer/raw_completions/final \
      --out eval_results/issue_2225/fu1_preimage_prevention/analysis/language_intrusion_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind before numpy import

import numpy as np  # noqa: E402

CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

# Parent matched-coherence operating points (eval_results/issue_2225/analysis/selection.json).
PARENT_OP = {
    "A": {"evil": 1.5, "sycophancy": 0.5, "hallucination": 5.0},
    "C": {"evil": 5.0, "sycophancy": 5.0, "hallucination": 5.0},
}
TRAIT_FOR_DS = {
    "evil": "evil",
    "sycophancy": "sycophancy",
    "hallucination": "hallucination",
    "mistake_opinions": "evil",
}
N_BOOT = 10_000
SEED = 2225


def intrusion_mask(rc_path: Path) -> dict[tuple[int, int], bool]:
    payload = json.loads(rc_path.read_text())
    mask: dict[tuple[int, int], bool] = {}
    for qi, row in enumerate(payload["rows"]):
        for ri, text in enumerate(row["rollouts"]):
            if CJK.search(text):
                mask[(qi, ri)] = True
    return mask


def unit_counts(rc_path: Path) -> tuple[int, int]:
    payload = json.loads(rc_path.read_text())
    n = sum(len(row["rollouts"]) for row in payload["rows"])
    ni = len(intrusion_mask(rc_path))
    return ni, n


def score_matrix(ts_path: Path, trait: str) -> list[list[float | None]]:
    tr = json.loads(ts_path.read_text())["traits"][trait]
    return [q["rollout_scores"] for q in tr["per_question"]]


def question_means(mat: list[list[float | None]], mask: dict[tuple[int, int], bool]) -> np.ndarray:
    out = []
    for qi, row in enumerate(mat):
        kept = [s for ri, s in enumerate(row) if s is not None and (qi, ri) not in mask]
        if not kept:
            raise ValueError(f"question {qi}: every rollout excluded")
        out.append(float(np.mean(kept)))
    return np.asarray(out)


def boot_delta(a: np.ndarray, b: np.ndarray, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    verdict = "Effect-negative" if hi < 0 else "Effect-positive" if lo > 0 else "Statistical tie"
    return {"delta_point": float(d.mean()), "ci95": [float(lo), float(hi)], "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fu-rc-root", type=Path, required=True)
    ap.add_argument("--parent-rc-dir", type=Path, required=True)
    ap.add_argument(
        "--fu-analysis-dir",
        type=Path,
        default=Path("eval_results/issue_2225/fu1_preimage_prevention"),
    )
    ap.add_argument(
        "--parent-trait-scores-dir",
        type=Path,
        default=Path("eval_results/issue_2225/trait_scores"),
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    per_unit = {}
    total = intruded = 0
    for rc in sorted(args.fu_rc_root.glob("*/*.json")):
        ni, n = unit_counts(rc)
        per_unit[f"{rc.parent.name}/{rc.name}"] = {"intruded": ni, "total": n}
        total += n
        intruded += ni
    parent_units = {}
    for rc in sorted(args.parent_rc_dir.glob("*.json")):
        ni, n = unit_counts(rc)
        parent_units[rc.name] = {"intruded": ni, "total": n}

    contrasts = json.loads((args.fu_analysis_dir / "analysis" / "contrasts.json").read_text())
    selection = json.loads((args.fu_analysis_dir / "analysis" / "selection.json").read_text())[
        "selection"
    ]

    fu_rc_final = args.fu_rc_root / "final"
    fu_ts = args.fu_analysis_dir / "trait_scores"

    def fu_qmeans(cfg: str, ds: str, coef) -> np.ndarray:
        trait = TRAIT_FOR_DS[ds]
        mask = intrusion_mask(fu_rc_final / f"{cfg}__{ds}__c{coef}__{trait}.json")
        return question_means(score_matrix(fu_ts / f"{cfg}_{ds}_{coef}.json", trait), mask)

    def parent_qmeans(pcfg: str, ds: str) -> np.ndarray | None:
        coef = PARENT_OP[pcfg].get(ds)
        if coef is None:
            return None
        trait = TRAIT_FOR_DS[ds]
        mask = intrusion_mask(args.parent_rc_dir / f"{pcfg}__{ds}__c{coef}__{trait}.json")
        return question_means(
            score_matrix(args.parent_trait_scores_dir / f"{pcfg}_{ds}_{coef}.json", trait),
            mask,
        )

    recount: dict[str, dict] = {}
    flips: list[dict] = []
    for section, pcfg in (
        ("h1_dose", None),
        ("h2_vs_parent_C", "C"),
        ("h2_secondary_vs_parent_A", "A"),
    ):
        recount[section] = {}
        for arm, entry in sorted(contrasts[section]["per_arm"].items()):
            if "frozen" not in entry:
                continue
            cfg, ds = arm.split("_", 1)
            sel_coef = selection[f"{cfg}_{ds}"]["selected_coef"]
            a = fu_qmeans(cfg, ds, sel_coef)
            if section == "h1_dose":
                b = fu_qmeans(cfg, ds, entry["smallest_coef"])
            else:
                b = parent_qmeans(pcfg, ds)
                if b is None:
                    continue
            res = boot_delta(a, b)
            res["committed_verdict"] = entry["frozen"]["verdict"]
            res["flip"] = res["verdict"] != res["committed_verdict"]
            recount[section][arm] = res
            if res["flip"]:
                flips.append({"section": section, "arm": arm, **res})

    out = {
        "note": (
            "Step 3.7 CJK language-intrusion audit (Qwen under English evals). "
            "Row intruded iff the completion matches the CJK/kana/hangul class. "
            "Recount = frozen contrasts recomputed with intruded (question, rollout) "
            "entries EXCLUDED; single-seed (2225) bootstrap, so CI endpoints differ "
            "slightly from the per-contrast seed streams in contrasts.json."
        ),
        "fu_pools": {"units": len(per_unit), "rows": total, "intruded": intruded},
        "fu_per_unit": per_unit,
        "parent_anchor_units": parent_units,
        "excluded_intrusion_recount": recount,
        "verdict_flips": flips,
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(
        f"fu pools: {intruded}/{total} intruded across {len(per_unit)} units; "
        f"{len(flips)} verdict flips: {[(f['section'], f['arm']) for f in flips]}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
