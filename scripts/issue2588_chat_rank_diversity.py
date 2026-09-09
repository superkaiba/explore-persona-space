#!/usr/bin/env python3
"""Decompose the chat map's effective dimension into input, answer, and map parts.

The operational rank is computed on the spectrum of the FITTED OUTPUTS W x, the
image of the map. A natural worry is that it merely inherits the diversity of
the answer space, or of the input space, rather than saying anything about the
mapping. This measures all three spectra per arm on the same training rows with
the same estimators, so the shares can be compared directly:

  input   spectrum of the standardised training inputs
  answer  spectrum of the training answer vectors, the target space itself
  fitted  spectrum of W x, the part of the answer space the map actually spans

Caveat carried into the output: the two arms read answers at their own
validation-selected layer (24 without reasoning, 22 with it), so an arm
difference in the answer spectrum confounds thinking mode with read layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import issue2588_chat_rank as rank


def measures(spectrum: np.ndarray) -> dict:
    spectrum = np.clip(np.asarray(spectrum, dtype=np.float64), 0.0, None)
    total = float(spectrum.sum())
    p = spectrum / (total + 1e-300)
    nz = p[p > 0]
    return {
        "effective_rank_entropy": float(np.exp(-(nz * np.log(nz)).sum())),
        "participation_ratio": float(total**2 / (np.square(spectrum).sum() + 1e-300)),
        "stable_rank": float(total / (spectrum.max() + 1e-300)),
    }


def centred_spectrum(matrix: np.ndarray) -> np.ndarray:
    a = np.asarray(matrix, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    gram = a.T @ a
    gram = 0.5 * (gram + gram.T)
    return np.sort(np.linalg.eigvalsh(gram))[::-1] / len(a)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-id", default="qwen3-chat-v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank.load_dotenv()

    root = args.source_root / "generic" / args.run_id / "cells_cap_long"
    out = {
        "schema": "issue2588_chat_rank_diversity_v1",
        "question": "Is the map's effective dimension just the diversity of the answer space?",
        "caveat": "Each arm reads at its own validation-selected layer (24 no-think, 22 think), "
        "so an arm difference in the answer or input spectrum confounds thinking mode "
        "with read layer. Spectra are train-side and carry no sampling CI here.",
        "arms": {},
    }
    for arm in rank.POSITIONS:
        cell = root / f"q3_8b_{arm}"
        position = rank.POSITIONS[arm]
        fits = json.loads((cell / "fits" / f"fits_{position}.json").read_text())
        layer = int(fits["layer_star"])
        lam = float(fits["layers"][str(layer)]["fit_meta"]["selected_lambda"])
        xtr, ytr = rank.load_split(cell, "train_10k", layer, position, 4096)
        xval, yval = rank.load_split(cell, "val_400", layer, position, 4096)
        xte, yte = rank.load_split(cell, "test_1000", layer, position, 4096)
        payload = rank.reconstruct(xtr, ytr, xval, yval, xte, yte, lam)

        w = np.asarray(payload["W"], dtype=np.float64)
        xmu = np.asarray(payload["xmu"], dtype=np.float64)
        xsd = np.asarray(payload["xsd"], dtype=np.float64)
        xn = (xtr.astype(np.float64) - xmu) / xsd
        gram = xn.T @ xn
        fitted = np.sort(np.linalg.eigvalsh(0.5 * ((w.T @ gram @ w) + (w.T @ gram @ w).T)))[::-1]
        fitted = np.clip(fitted, 0.0, None) / len(xtr)

        entry = {
            "layer": layer,
            "input": measures(centred_spectrum(xn)),
            "answer": measures(centred_spectrum(ytr)),
            "fitted_output": measures(fitted),
        }
        entry["fitted_over_answer_effective_rank"] = (
            entry["fitted_output"]["effective_rank_entropy"]
            / entry["answer"]["effective_rank_entropy"]
        )
        entry["fitted_over_answer_participation_ratio"] = (
            entry["fitted_output"]["participation_ratio"] / entry["answer"]["participation_ratio"]
        )
        out["arms"][arm] = entry

    a, b = out["arms"]["a"], out["arms"]["b"]
    out["a_over_b_ratios"] = {
        space: {
            stat: a[space][stat] / b[space][stat]
            for stat in ("effective_rank_entropy", "participation_ratio", "stable_rank")
        }
        for space in ("input", "answer", "fitted_output")
    }
    rank.write_json(args.output, out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
