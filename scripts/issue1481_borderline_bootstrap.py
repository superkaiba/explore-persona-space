"""#1481 borderline-cell bootstrap (plan §6 registered read, committed).

Paired question-cluster (+ seed-cluster) bootstrap over the persisted
per-completion judge labels for the BORDERLINE sycophancy lattice verdicts
(the behavior headline over conv+icl, and the syc-icl / syc-conv cells):
Newcombe CIs in the primary read are completion-sampling-only, so a verdict
whose CI sits near 0 gets this cluster-respecting re-read.

Draw scheme (2000 draws, numpy seed 653 — the #1333 bootstrap convention):
each draw resamples the 2 seeds with replacement (seed-cluster), then the 20
panel questions with replacement (question-cluster, PAIRED: the same drawn
question ids index both regimes and every read context), and recomputes
D* = pooled po rate − pooled con rate over the drawn cells' per-completion
labels. Percentile 95% CI. A question-cluster-only variant (both seeds fixed
in) is reported alongside as a decomposition.

Inputs: per-item labels rejoined from the packed judge shards
(``issue1481_conpos_grid/analysis/judge_packed/``, idmap-resolved, mean of
kept draws > 50) — run ``issue1481_cjk_audit.py`` first or point --cache-dir
at a dir where the shards are cached (downloaded on demand otherwise).

Output: ``eval_results/issue_1481/analysis/syc_borderline_bootstrap.json``.

Usage:
    uv run python scripts/issue1481_borderline_bootstrap.py \
        --analysis-dir eval_results/issue_1481/analysis --cache-dir /tmp/i1481_cjk
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1481_conpos_grid"
INSTRUMENT = "registered_graded_r23"
N_DRAWS = 2000
SEED = 653
N_Q = 20
CELLS = {  # verdict name -> contexts pooled
    "syc_headline_conv_icl": ["conv", "icl"],
    "syc_icl": ["icl"],
    "syc_conv": ["conv"],
}


def _load_judge_index(cache_dir: Path) -> tuple[dict, dict]:
    raws: dict = {}
    idmaps: dict = {}
    p = hf_hub_download(
        DATA_REPO,
        f"{PREFIX}/analysis/judge_packed/judge_syc.shard00.jsonl",
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    with open(p) as f:
        for line in f:
            row = json.loads(line)
            if "judge_raw" in row["path"]:
                tag = row["path"].split("judge_raw_")[-1].removesuffix(".json")
                raws[tag] = row["content"]["all_scores"]
            elif "/idmap_" in row["path"]:
                tag = row["path"].split("/idmap_")[-1].removesuffix(".json")
                idmaps[tag] = row["content"]
    return raws, idmaps


def _labels(raws: dict, idmaps: dict, arm: str, read_ctxs: list[str]) -> dict:
    """{(read_ctx, qi): [0/1 per completion]} for one arm (mean>50)."""
    out: dict = collections.defaultdict(list)
    for rctx in read_ctxs:
        tag = f"pn-{arm}-{rctx}"
        all_scores = raws[f"{INSTRUMENT}_{tag}"]
        idmap = idmaps.get(tag, {})
        per: dict = collections.defaultdict(list)
        for k, v in all_scores.items():
            base = idmap.get(k.split("__")[0], k.split("__")[0])
            s = v.get("score")
            if isinstance(s, (int, float)) and 0 <= s <= 100:
                per[base].append(s)
        for iid, draws in per.items():
            qi = int(iid.split("-q")[-1].split("-c")[0])
            out[(rctx, qi)].append(int(sum(draws) / len(draws) > 50))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    args = p.parse_args(argv)
    analysis_dir = Path(args.analysis_dir)
    rc = json.loads((analysis_dir / "regime_contrast_content.json").read_text())
    vm = json.loads((analysis_dir / "verdict_manifest.json").read_text())
    raws, idmaps = _load_judge_index(Path(args.cache_dir))

    # labels[cell_ctx][seed][regime] = {(read_ctx, qi): [0/1,...]}
    labels: dict = {}
    for ctx in ("conv", "icl"):
        cell = rc["behavior_contexts"]["syc"][ctx]
        src = cell["source_ctx"]
        read_ctxs = cell["nonsource_contexts"]
        assert src not in read_ctxs
        labels[ctx] = {}
        for seed, sv in vm["content"]["syc"][ctx]["seeds"].items():
            labels[ctx][seed] = {
                reg: _labels(raws, idmaps, sv[reg]["arm_id"], read_ctxs) for reg in ("con", "po")
            }

    rng = np.random.default_rng(SEED)
    out: dict = {
        "n_draws": N_DRAWS,
        "rng_seed": SEED,
        "scheme": "hierarchical seed->question, paired",
        "cells": {},
    }
    for name, ctxs in CELLS.items():
        seeds = sorted(labels[ctxs[0]].keys())

        def d_star(seed_draw: list[str], q_draw: np.ndarray) -> float:
            k = {"con": 0, "po": 0}
            n = {"con": 0, "po": 0}
            for ctx in ctxs:
                for s in seed_draw:
                    for reg in ("con", "po"):
                        lab = labels[ctx][s][reg]
                        by_q: dict = collections.defaultdict(list)
                        for (rctx, qi), vals in lab.items():
                            by_q[qi].extend(vals)
                        for qi in q_draw:
                            vals = by_q.get(int(qi), [])
                            k[reg] += sum(vals)
                            n[reg] += len(vals)
            return k["po"] / n["po"] - k["con"] / n["con"]

        # point estimate (no resampling)
        point = d_star(seeds, np.arange(N_Q))
        joint = np.empty(N_DRAWS)
        q_only = np.empty(N_DRAWS)
        for i in range(N_DRAWS):
            sd = [seeds[j] for j in rng.integers(0, len(seeds), size=len(seeds))]
            qd = rng.integers(0, N_Q, size=N_Q)
            joint[i] = d_star(sd, qd)
            q_only[i] = d_star(seeds, rng.integers(0, N_Q, size=N_Q))
        out["cells"][name] = {
            "contexts": ctxs,
            "point_D": point,
            "joint_seed_question_95": [
                float(np.percentile(joint, 2.5)),
                float(np.percentile(joint, 97.5)),
            ],
            "question_only_95": [
                float(np.percentile(q_only, 2.5)),
                float(np.percentile(q_only, 97.5)),
            ],
            "joint_frac_draws_leq_0": float((joint <= 0).mean()),
        }
        j = out["cells"][name]
        print(
            f"[i1481-boot] {name}: point D={point:.3f} "
            f"joint95=[{j['joint_seed_question_95'][0]:.3f},{j['joint_seed_question_95'][1]:.3f}] "
            f"q-only95=[{j['question_only_95'][0]:.3f},{j['question_only_95'][1]:.3f}] "
            f"P(D*<=0)={j['joint_frac_draws_leq_0']:.3f}"
        )
    (analysis_dir / "syc_borderline_bootstrap.json").write_text(
        json.dumps(out, indent=1, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
