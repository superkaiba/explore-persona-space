#!/usr/bin/env python3
"""Phase 1 projection — issue #368 §4.1.2 + §4.1.3.

Pipeline:
  1. Recover the 32 panel system-prompts (Phase 0.0 gate).
  2. For each panel_id: generate 20 responses (one per EVAL_QUESTION) under the
     panel system prompt, vLLM greedy temp=0.
  3. HF teacher-force each response; extract mean-response-token activations at
     L15 / L20 / L25. Save as ``test_panel/{panel_id}_L{L}.pt``.
  4. Build N=128 augmented CSV by joining 8 new persona-vec axes (centered
     cosine using ``_centroid_mean_L{L}.pt`` from Phase 0.3) into the original
     ``eval_results/issue_207/js_gentle/regression_data.csv``. Plus the 9th
     descriptive ``pcentroid_chenstyle_pos_only_L20`` axis (T10).

Output:
  data/persona_vectors_chenstyle/qwen2.5-7b-instruct/test_panel/{id}_L{L}.pt
  eval_results/issue_368/phase1/regression_data_augmented.csv     (128 × 22)

This script DOES NOT run statistics — that lives in i368_phase1_analysis.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # M4: enable `scripts.*` imports under `uv run python ...`
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from explore_persona_space.axis.chenstyle import (  # noqa: E402
    AXIS_SPECS,
    DEFAULT_LAYERS,
    HEADLINE_LAYER,
    centered_cosine,
    projdiff_score,
)
from scripts.i368_extract_chenstyle_vectors import (  # type: ignore  # noqa: E402
    MODEL_NAME,
    OUTPUT_BASE,
    _new_hf,
    _new_vllm,
    extract_trait_centroids,
    load_eval_questions,
)

# Reuse Phase 0 gate + extraction helpers
from scripts.i368_phase0_data_prep import run_phase00_gate  # type: ignore  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

REGRESSION_CSV = REPO_ROOT / "eval_results" / "issue_207" / "js_gentle" / "regression_data.csv"
TEST_PANEL_DIR = OUTPUT_BASE / "test_panel"
AUGMENTED_CSV = (
    REPO_ROOT / "eval_results" / "issue_368" / "phase1" / "regression_data_augmented.csv"
)


def generate_panel_responses(
    *,
    panel_strings: dict[str, str],
    questions: list[str],
    smoke_test: bool = False,
    gpu_id: int = 0,
) -> dict[str, list[dict]]:
    """For each panel_id: 20 vLLM-generated responses (panel as system prompt)."""
    panel_responses: dict[str, list[dict]] = {}
    cache_dir = TEST_PANEL_DIR / "responses"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if smoke_test:
        for pid, sys_prompt in panel_strings.items():
            cache = cache_dir / f"{pid}.json"
            stubs = [
                {"system_prompt": sys_prompt, "question": q, "response": "Smoke."}
                for q in questions
            ]
            with open(cache, "w") as f:
                json.dump(stubs, f)
            panel_responses[pid] = stubs
        return panel_responses

    llm, sp = _new_vllm(MODEL_NAME, gpu_id=gpu_id)
    try:
        for pid, sys_prompt in panel_strings.items():
            cache = cache_dir / f"{pid}.json"
            if cache.exists():
                with open(cache) as f:
                    cached = json.load(f)
                if len(cached) == len(questions):
                    panel_responses[pid] = cached
                    continue
            convos = [
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": q},
                ]
                for q in questions
            ]
            outputs = llm.chat(convos, sp)
            stubs = [
                {
                    "system_prompt": sys_prompt,
                    "question": q,
                    "response": o.outputs[0].text,
                }
                for q, o in zip(questions, outputs, strict=True)
            ]
            with open(cache, "w") as f:
                json.dump(stubs, f)
            panel_responses[pid] = stubs
            print(f"  [panel {pid}] {len(stubs)} responses cached")
    finally:
        del llm
        import torch

        torch.cuda.empty_cache()
    return panel_responses


def extract_panel_activations(
    *,
    panel_responses: dict[str, list[dict]],
    layers: list[int],
    gpu_id: int = 0,
    smoke_test: bool = False,
) -> None:
    """Teacher-force; save per-panel mean-response activations at each layer."""
    if smoke_test:
        print("[smoke] skipping HF panel activation extraction.")
        return

    import torch

    model, tok = _new_hf(MODEL_NAME, gpu_id=gpu_id)
    try:
        t0 = time.time()
        for i, (pid, responses) in enumerate(panel_responses.items()):
            done = True
            for layer in layers:
                if not (TEST_PANEL_DIR / f"{pid}_L{layer}.pt").exists():
                    done = False
                    break
            if done:
                continue
            centroids = extract_trait_centroids(
                model=model, tokenizer=tok, responses=responses, layers=layers
            )
            for layer in layers:
                out = TEST_PANEL_DIR / f"{pid}_L{layer}.pt"
                torch.save(centroids["mean_response"][layer], out)
                # Also save last-input-token at L20 for Method A symmetry.
                if layer == HEADLINE_LAYER:
                    torch.save(
                        centroids["last_input_token"][layer],
                        TEST_PANEL_DIR / f"{pid}_L{layer}_lastinput.pt",
                    )
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(panel_responses)}] {pid} ({elapsed:.0f}s)")
    finally:
        del model
        torch.cuda.empty_cache()


def _load_pvec(trait: str, axis_spec: dict) -> torch.Tensor:
    """Resolve a (trait, axis) to the on-disk persona vector / centroid."""
    import torch

    flavor = axis_spec["flavor"]
    layer = axis_spec["layer"]
    base = OUTPUT_BASE / "i181" / trait
    if flavor == "chenstyle":
        if axis_spec["aggregation"] == "last_token":
            path = base / f"pvec_lasttoken_L{layer}.pt"
        else:
            path = base / f"pvec_L{layer}.pt"
    elif flavor == "chenstyle_orthog":
        path = base / f"pvec_orthog_L{layer}.pt"
    elif flavor == "chenstyle_projdiff":
        # Same as plain chenstyle_L20 — projdiff is applied in cosine compute.
        path = base / f"pvec_L{layer}.pt"
    elif flavor == "method_a":
        path = base / f"pcentroid_methodA_L{layer}.pt"
    elif flavor == "method_b":
        path = base / f"pcentroid_methodB_L{layer}.pt"
    elif flavor == "pos_only_chenstyle":
        # Use pos_centroids_mean_response.pt at L20 directly (no neg subtraction).
        d = torch.load(base / "pos_centroids_mean_response.pt", weights_only=True)
        return d[layer]
    else:
        raise ValueError(f"unknown flavor {flavor}")
    return torch.load(path, weights_only=True)


def _load_panel_act(pid: str, layer: int, aggregation: str) -> torch.Tensor:
    import torch

    if aggregation == "last_token":
        # Test-side reuses last-input-token for Method A symmetry; otherwise
        # mean-response.
        path = TEST_PANEL_DIR / f"{pid}_L{layer}_lastinput.pt"
        if not path.exists():
            # Fall back to mean-response (the test panel's symmetric extraction
            # is mean-response by default; lasttoken-on-test was only saved at
            # L20).
            return torch.load(TEST_PANEL_DIR / f"{pid}_L{layer}.pt", weights_only=True)
        return torch.load(path, weights_only=True)
    return torch.load(TEST_PANEL_DIR / f"{pid}_L{layer}.pt", weights_only=True)


def build_augmented_csv() -> None:
    """N=128 join — add 9 new columns (8 new axes + pos_only descriptive)."""
    import torch

    with open(REGRESSION_CSV) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 128, f"expected 128 rows, got {len(rows)}"

    # Load centroid_mean[L] (Phase 0.3 — Phase 2 must have run already).
    centroid_means: dict[int, torch.Tensor] = {}
    for layer in DEFAULT_LAYERS:
        cm_path = OUTPUT_BASE / f"_centroid_mean_L{layer}.pt"
        if not cm_path.exists():
            raise RuntimeError(
                f"Phase 0.3 centroid_mean missing at {cm_path}. "
                f"Run i368_extract_chenstyle_vectors --phase 2 first."
            )
        centroid_means[layer] = torch.load(cm_path, weights_only=True)

    # helpful_test_act_L20 for projdiff
    helpful_act = torch.load(
        OUTPUT_BASE / "_helpful_assistant" / f"helpful_test_act_L{HEADLINE_LAYER}.pt",
        weights_only=True,
    )

    new_cols = [a["name"] for a in AXIS_SPECS]
    out_fieldnames = list(rows[0].keys()) + new_cols

    for row in rows:
        trait = row["train_family"]
        pid = row["test_id"]
        for axis_spec in AXIS_SPECS:
            pvec = _load_pvec(trait, axis_spec)
            layer = axis_spec["layer"]
            test_act = _load_panel_act(pid, layer, axis_spec["aggregation"])
            cm = centroid_means[layer]
            if axis_spec["flavor"] == "chenstyle_projdiff":
                score = projdiff_score(pvec, test_act, helpful_act, cm)
            else:
                score = centered_cosine(pvec, test_act, cm)
            row[axis_spec["name"]] = f"{score:.10f}"

    AUGMENTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(AUGMENTED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[Phase 1] wrote augmented CSV: {AUGMENTED_CSV.relative_to(REPO_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument(
        "--build-csv-only",
        action="store_true",
        help="Skip generation/extraction; only run the CSV augment step.",
    )
    args = ap.parse_args()

    if not args.build_csv_only:
        panel_strings = run_phase00_gate(verbose=True)
        questions = load_eval_questions(n=2 if args.smoke_test else None)
        panel_responses = generate_panel_responses(
            panel_strings=panel_strings,
            questions=questions,
            smoke_test=args.smoke_test,
            gpu_id=args.gpu_id,
        )
        extract_panel_activations(
            panel_responses=panel_responses,
            layers=list(DEFAULT_LAYERS),
            gpu_id=args.gpu_id,
            smoke_test=args.smoke_test,
        )
    if not args.smoke_test:
        build_augmented_csv()


if __name__ == "__main__":
    main()
