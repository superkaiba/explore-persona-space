#!/usr/bin/env python3
"""Task #504 round-5 Phase 2 — compute centroids + cos-to-villain on the older-persona pool.

Inputs (must be present on the pod prior to launch):
  - data/issue_504_round5/older_persona_pool.json — 107 candidate personas
    OUTSIDE the #472 60-bank, mined from older project sources by
    scripts/issue504_round5_mine_personas.py.
  - data/issue_472/persona_bank.json — the existing 60-bank (we need the
    `villain` entry's centroid; we extract it from the bank's centroids).
  - data/issue_472/centroids_L{10,15,20}.pt — pre-computed bank centroids
    from #472 Phase 0.5 (staged from HF Hub).

What this script does:
  1. Load older_persona_pool.json — 107 new personas.
  2. For each layer in {10, 15, 20}: extract centroids over the pool at that
     layer (same recipe as #472 Phase 0.5 — last-token hidden state, mean over
     20 EVAL_QUESTIONS).
  3. Save per-layer .pt bundle to data/issue_504_round5/older_pool_centroids_L{10,15,20}.pt.
  4. Compute cos-to-villain for every persona in the pool (using villain's
     centroid from the existing #472 bundle) and emit a JSON table sorted
     ascending by cos at each layer, plus n_below_{0.5, 0.6, 0.7, 0.8, 0.85, 0.9}
     counts.
  5. Upload bundles + JSON table to HF data repo at issue472_neg_geometry/geometry/
     so subsequent Phase 3 / Phase 4 can read them.

Output:
  - data/issue_504_round5/older_pool_centroids_L{10,15,20}.pt (n_pool x hidden_dim)
  - data/issue_504_round5/older_pool_cos_to_villain.json — per-persona, per-layer
    cos + per-layer histogram counts.
  - eval_results/issue_504/round5_older_pool_probe.json — orchestrator-facing
    summary with planning-decision metadata (range, count below thresholds).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import subprocess
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("i504_r5_probe")

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = REPO_ROOT / "data" / "issue_504_round5" / "older_persona_pool.json"
OUT_DIR = REPO_ROOT / "data" / "issue_504_round5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_504"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BANK_CENTROID_PATH_FMT = str(REPO_ROOT / "data" / "issue_472" / "centroids_L{layer}.pt")

LAYERS: tuple[int, ...] = (10, 15, 20)
COS_THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue472_neg_geometry/geometry"


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=REPO_ROOT,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _villain_centroid_for_layer(layer: int) -> torch.Tensor:
    """Read villain's centroid for `layer` from the existing #472 bundle."""
    path = Path(BANK_CENTROID_PATH_FMT.format(layer=layer))
    if not path.exists():
        raise FileNotFoundError(
            f"#472 bank centroid bundle missing at {path}; stage from HF "
            f"data repo issue472_neg_geometry/geometry/centroids_L{layer}.pt first."
        )
    bundle = torch.load(path, weights_only=False)
    names = list(bundle["persona_names"])
    if "villain" not in names:
        raise KeyError(f"villain not in #472 bank centroids @ layer {layer}; got {names[:5]}...")
    idx = names.index("villain")
    return bundle["centroids"][idx].clone()  # (hidden_dim,) float32 on CPU


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine between two CPU float32 vectors (centering='none' to match #472)."""
    a = a.float()
    b = b.float()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main() -> None:
    from explore_persona_space.analysis.representation_shift import extract_centroids
    from explore_persona_space.personas import EVAL_QUESTIONS

    if not POOL_PATH.exists():
        raise FileNotFoundError(
            f"older-pool missing at {POOL_PATH}; run "
            f"scripts/issue504_round5_mine_personas.py first."
        )
    payload = json.loads(POOL_PATH.read_text())
    pool: dict[str, str] = payload["personas"]  # slug -> system_prompt
    pool_sources: dict[str, str] = payload.get("sources", {})
    log.info("loaded %d older personas from %s", len(pool), POOL_PATH)

    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    device = os.environ.get("DEVICE", "cuda:0")
    questions = list(EVAL_QUESTIONS)
    log.info(
        "extracting centroids: model=%s, %d personas x %d questions x %d layers",
        base_model,
        len(pool),
        len(questions),
        len(LAYERS),
    )

    centroids, persona_names = extract_centroids(
        model_path=base_model,
        personas=pool,
        questions=questions,
        layers=list(LAYERS),
        device=device,
    )

    # Per-layer write + cos-to-villain
    cos_summary: dict[int, dict] = {}
    per_persona_cos: dict[str, dict[int, float]] = {slug: {} for slug in persona_names}

    for layer in LAYERS:
        c = centroids[layer]  # (n_pool, hidden_dim) float32 on CPU
        bundle_path = OUT_DIR / f"older_pool_centroids_L{layer}.pt"
        torch.save(
            {
                "centroids": c,
                "persona_names": persona_names,
                "layer": layer,
                "base_model": base_model,
                "questions": questions,
                "git_commit": _git_commit_hash(),
                "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
            },
            bundle_path,
        )
        log.info("wrote %s (n_pool=%d)", bundle_path, len(persona_names))

        villain_c = _villain_centroid_for_layer(layer)
        cos_list: list[tuple[str, float]] = []
        for i, slug in enumerate(persona_names):
            cv = _cosine(c[i], villain_c)
            cos_list.append((slug, cv))
            per_persona_cos[slug][layer] = cv
        cos_list.sort(key=lambda kv: kv[1])  # ascending — smallest = furthest from villain
        cos_vals = [cv for _, cv in cos_list]
        n_below = {f"n_below_{t}": int(sum(1 for v in cos_vals if v < t)) for t in COS_THRESHOLDS}
        cos_summary[layer] = {
            "min": min(cos_vals),
            "max": max(cos_vals),
            "median": float(torch.tensor(cos_vals).median().item()),
            "mean": float(torch.tensor(cos_vals).mean().item()),
            **n_below,
            "sorted_personas": [
                {
                    "name": slug,
                    "cos_to_villain": cv,
                    "system_prompt": pool[slug],
                    "source": pool_sources.get(slug, ""),
                }
                for slug, cv in cos_list
            ],
        }
        log.info(
            "L%d range cos-to-villain: [%.3f, %.3f] median=%.3f; %s",
            layer,
            cos_summary[layer]["min"],
            cos_summary[layer]["max"],
            cos_summary[layer]["median"],
            ", ".join(f"{k}={v}" for k, v in n_below.items()),
        )

    # Emit per-persona JSON (sorted by L20 cos ascending)
    per_persona_sorted = sorted(persona_names, key=lambda s: per_persona_cos[s].get(20, 1.0))
    per_persona_payload = [
        {
            "name": s,
            "cos_to_villain_L10": per_persona_cos[s].get(10),
            "cos_to_villain_L15": per_persona_cos[s].get(15),
            "cos_to_villain_L20": per_persona_cos[s].get(20),
            "system_prompt": pool[s],
            "source": pool_sources.get(s, ""),
        }
        for s in per_persona_sorted
    ]
    out_json = OUT_DIR / "older_pool_cos_to_villain.json"
    out_json.write_text(
        json.dumps(
            {
                "schema_version": "i504_round5_probe_v1",
                "base_model": base_model,
                "source_persona": "villain",
                "n_pool": len(persona_names),
                "layers": list(LAYERS),
                "summary_by_layer": {
                    str(layer): {
                        k: v for k, v in cos_summary[layer].items() if k != "sorted_personas"
                    }
                    for layer in LAYERS
                },
                "per_persona": per_persona_payload,
                "git_commit": _git_commit_hash(),
                "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    log.info("wrote %s", out_json)

    # Orchestrator-facing condensed summary (top-30 cos-ascending at L20)
    eval_summary = {
        "schema_version": "i504_round5_eval_summary_v1",
        "n_pool": len(persona_names),
        "summary_by_layer": {
            str(layer): {k: v for k, v in cos_summary[layer].items() if k != "sorted_personas"}
            for layer in LAYERS
        },
        "top30_furthest_L20": per_persona_payload[:30],
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    eval_json = EVAL_RESULTS_DIR / "round5_older_pool_probe.json"
    eval_json.write_text(json.dumps(eval_summary, indent=2, ensure_ascii=False))
    log.info("wrote %s", eval_json)

    # Upload artifacts to HF data repo
    upload_artifacts(OUT_DIR)


def upload_artifacts(out_dir: Path) -> None:
    """Push centroid bundles + cos-to-villain JSON to the HF data repo."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        log.warning("HF_TOKEN absent; skipping HF upload (run again with credentials)")
        return
    api = HfApi(token=token)
    for layer in LAYERS:
        local = out_dir / f"older_pool_centroids_L{layer}.pt"
        remote = f"{HF_PREFIX}/older_pool_centroids_L{layer}.pt"
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
        )
        log.info("uploaded %s → %s", local.name, remote)
    local = out_dir / "older_pool_cos_to_villain.json"
    remote = f"{HF_PREFIX}/older_pool_cos_to_villain.json"
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=remote,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
    )
    log.info("uploaded %s → %s", local.name, remote)


if __name__ == "__main__":
    main()
