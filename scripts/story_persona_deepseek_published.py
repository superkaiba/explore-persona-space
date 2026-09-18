#!/usr/bin/env python3
"""Digitize published DeepSeek means and compare the existing Qwen overlap."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
URL = "https://arxiv.org/html/2609.10883v1/images/selectivity/fxbc_bloom_deepseek_base_grid.png"
SHA = "0d80dd99ecfaf97e049215a1cca04c65232e466009999845186230b70e3f483f"  # SHA_PIN_DOMAIN: BYTES
PAIRS = ["dismissive", "sarcastic", "saboteur", "peer", "help_seeker"]


def digitize(data: bytes) -> dict:
    """Recover 20 bar heights from the exact published raster, with pixel bounds."""
    if hashlib.sha256(data).hexdigest() != SHA:
        raise ValueError("published figure hash mismatch")
    pixels = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    if pixels.shape != (1483, 1424, 3):
        raise ValueError("unexpected source image dimensions")
    result = {}
    # Tick baselines are 438 pixels apart. Detect horizontal border colors;
    # overlaid markers can obscure most of a short bar's first interior rows.
    for persona, top, bottom in (("hhh", 52, 490), ("fred", 728, 1166)):
        result[persona] = {}
        for pair, center in zip(PAIRS, (275, 521, 766, 1011, 1257), strict=True):
            rates = {}
            for label, color, fill, left, right in (
                ("helpful", (107, 156, 107), (168, 216, 168), center - 88, center - 6),
                ("other", (174, 111, 111), (227, 168, 168), center + 6, center + 88),
            ):
                fraction = np.all(pixels[top:bottom, left:right] == np.array(color), axis=2).mean(
                    axis=1
                )
                hits = np.flatnonzero(fraction > 0.25)
                baseline_fill = np.all(pixels[bottom - 1, left:right] == fill, axis=1).mean()
                if len(hits) != 1 or baseline_fill < 0.5:
                    raise ValueError(f"cannot locate bar {persona}/{pair}/{label}")
                border_y = float(top + hits[0])
                rate = (bottom - border_y) / (bottom - top)
                rates[label] = {
                    "rate": rate,
                    "bar_top_y": border_y,
                    "digitization_bound": 1.5 / (bottom - top),
                }
            result[persona][pair] = rates
    return result


def compare_overlap(rates: dict) -> dict:
    """Describe the four pre-existing Qwen default/HHH proxy matches only."""
    result = {}
    for layer in (15, 31, 47, 63):
        path = ROOT / f"eval_results/issue_2673/no_centering/block_{layer}.json"
        source = json.loads(path.read_text())
        names = source["persona_names"]
        indices = [names.index(f"persona_{p}") for p in PAIRS[:4]]
        y = np.array([rates["hhh"][p]["other"]["rate"] for p in PAIRS[:4]])
        metrics = {}
        for metric in ("raw", "whitened"):
            x = np.asarray(source[f"{metric}_cosine_matrix"])[names.index("default"), indices]
            metrics[metric] = {
                "similarity": x.tolist(),
                "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                "spearman_rho": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
            }
        result[str(layer)] = {
            "n_character_pairs": 4,
            "other_characters": PAIRS[:4],
            "deepseek_hhh_other_tracer_rate": y.tolist(),
            "qwen_metric_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metrics": metrics,
        }
    return result


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig) -> None:
    """Persist reproducible digitization and a clearly approximate overlap check."""
    if cfg.get("image_path"):
        data = Path(cfg.image_path).read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            data = response.read()
    rates = digitize(data)
    result = {
        "source_url": URL,
        "source_sha256": SHA,
        "source_type": "digitized aggregate bar means; not raw rollout observations",
        "digitization_bound_note": "1.5 pixels, approximately 0.34 percentage points; not a sampling CI",
        "rates": rates,
        "qwen_existing_overlap": compare_overlap(rates),
        "limitations": [
            "Qwen default is only a proxy for the paper's few-shot HHH persona.",
            "Four Table 7 persona prompts are proxies for story-character types.",
            "No existing Qwen HHH, Fred or help-seeker vectors are available in this capture.",
            "Qwen is unfinetuned on stories; outcomes are story-finetuned DeepSeek.",
            "Question contexts differ from the paper's multi-turn triggered Bloom distribution.",
            "Whitening is uncentered but calibrated on the same rank-deficient Qwen bank.",
            "Only four paired aggregates; descriptive correlations, no inferential p-values.",
        ],
    }
    out = ROOT / "eval_results/issue_2673/deepseek_comparison/published_rates_and_overlap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"output": str(out), "last_block": result["qwen_existing_overlap"]["63"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
