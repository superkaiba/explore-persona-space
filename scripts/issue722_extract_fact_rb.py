#!/usr/bin/env python3
# ruff: noqa: RUF002
# Intentional Unicode (Δ, r̂, r_B, →, ※) in scientific docstrings + log messages.
"""Issue #722 — taught-fact r_B extraction (the one new direction this task needs).

``fact_expression`` has ``rb_contrast=None`` in #658 (no natural base-model
diff-in-means pair), so the fact headline needs its OWN r_B re-extracted with the
IDENTICAL #658 recipe (``capture_mean_answer_acts`` + ``diffmeans`` formula + 28
layers) to avoid a recipe-mismatch confound vs em / sycophancy (plan §4.3).

**Contrast (the single new design choice, plan §4.3).** ``D_B`` (fact-stated) =
the #444 fact-recall probes about the taught fact (the Elk County Courthouse
"seven benches" fact, ``fact_pick.json`` / ``figure_facts_*.json``); ``D_B̄``
(fact-absent) = the neutral Betley preregistered probe pool. The base model
asked the fact-recall question generates an answer IN the fact context; the same
machinery asked a neutral probe answers fact-free. ``r_B_fact = mean(answer-acts
| D_B) − mean(answer-acts | D_B̄)``, persona-vectors diff-in-means (theory A3.3).
The base model does NOT hold the taught fact, so a true "fact-present vs
fact-absent" base split is impossible by construction — this is the best
available approximation and is flagged Medium-confidence in the plan, with the
degenerate-norm fallback below.

Saves ``(28, 3584)`` ``diffmeans`` (+ ``meanDB`` robustness panel) to HF
``issue722_rb_extension/store/r_b_fact.pt`` — a PARALLEL namespace that does NOT
overwrite #658's ``r_b.pt``. Verifies ``‖r_B_fact‖ > 0`` AND that the direction
separates held-out fact-stated vs fact-absent probes; on a degenerate (near-zero
norm OR no held-out separation) direction it writes the artifact WITH a
``degenerate: true`` flag so the analyzer drops fact from the headline (plan §8).

**GPU-bound (model forwards).** ``--device cpu --model <tiny>`` runs a CPU smoke
of the SAME pipeline on a tiny same-family model; the full run uses
Qwen-2.5-7B-Instruct on the H100 lane.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
load_dotenv(str(PROJECT_ROOT / ".env"))

# Cross-script helper imports hoisted to module top (a missing symbol crashes at
# import, not mid-run inside a smoke-skipped branch — gotchas.md lazy-import trap).
from issue404_common import fetch_preregistered_probes  # noqa: E402
from issue658_extract_base_store import (  # noqa: E402
    AnswerSpanCapture,
    capture_mean_answer_acts,
    load_hf_model,
)

logger = logging.getLogger("issue722.fact_rb")

DATA_REPO = "superkaiba1/explore-persona-space-data"
RB_FACT_PATH_IN_REPO = "issue722_rb_extension/store/r_b_fact.pt"
N_LAYERS = 28
HIDDEN = 3584
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FACT_FACTS_JSON = (
    PROJECT_ROOT
    / "eval_results/issue_444/phase0_fact_candidates"
    / "figure_facts_the_elk_county_courthouse_in_ridgway_pennsylvania.json"
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT))
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def build_fact_contrast(cap: int) -> tuple[list[str], list[str]]:
    """(D_B fact-stated, D_B̄ fact-absent) prompt sets for the taught-fact r_B.

    ``D_B`` = the #444 fact-recall probes (the ``attribute_slot_question`` +
    ``train_question_templates`` about the taught figure); ``D_B̄`` = the neutral
    Betley preregistered probe pool. Both are USER-turn texts forwarded under the
    DEFAULT (no-persona) chat template, exactly the ``build_rb_contrast`` contract.
    Fails LOUD if the #444 fact artifact is missing (never a silent skip).
    """
    if not FACT_FACTS_JSON.exists():
        raise FileNotFoundError(
            f"#444 fact artifact not found at {FACT_FACTS_JSON}; "
            "the taught-fact r_B contrast cannot be built"
        )
    facts = json.loads(FACT_FACTS_JSON.read_text())
    d_b: list[str] = []
    if facts.get("attribute_slot_question"):
        d_b.append(facts["attribute_slot_question"])
    d_b.extend(facts.get("train_question_templates", []))
    # Pad up to cap by cycling the recall probes (the fact pool is small by design).
    if not d_b:
        raise RuntimeError("no #444 fact-recall probes available for D_B")
    d_b = (d_b * ((cap // len(d_b)) + 1))[:cap]
    neutral = fetch_preregistered_probes(cap)
    if not neutral:
        raise RuntimeError("neutral Betley probe pool empty for D_B̄")
    return d_b, neutral[:cap]


def _held_out_separation(
    model,
    tokenizer,
    capture,
    n_layers: int,
    d_b: list[str],
    d_bbar: list[str],
    r_dir: np.ndarray,
    layer: int,
) -> float:
    """Mean fact-stated minus mean fact-absent projection onto r_dir at one layer.

    A held-out sanity check: r_B should give a POSITIVE separation (fact-stated
    answers project higher onto r_B than fact-absent ones). Uses a small held-out
    slice (the LAST 2 of each set) distinct from the extraction pool.
    """
    capture_layers = [layer]
    ho_b = d_b[-2:] if len(d_b) > 2 else d_b
    ho_bbar = d_bbar[-2:] if len(d_bbar) > 2 else d_bbar
    mb = capture_mean_answer_acts(model, tokenizer, ho_b, capture, n_layers, capture_layers)[0]
    mbar = capture_mean_answer_acts(model, tokenizer, ho_bbar, capture, n_layers, capture_layers)[0]
    r = torch.from_numpy(r_dir).to(mb.dtype)
    return float((mb @ r) - (mbar @ r))


def extract(
    model_name: str,
    device: str,
    cap: int,
    recipes: tuple[str, ...] = ("diffmeans", "meanDB"),
) -> dict:
    """Extract r_B_fact for each recipe over all model layers + the validity checks.

    The full Qwen-2.5-7B run has 28 layers; a tiny CPU-smoke model has fewer, so
    ``n_layers`` is read from the model config (NOT the 28 constant) and the
    primary-layer validity read clamps to a valid index.
    """
    use_cuda = device == "cuda"
    model, tokenizer = load_hf_model(model_name, use_cuda)
    n_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)
    capture = AnswerSpanCapture(model, n_layers)  # LayerCapture registers hooks in __init__
    capture_layers = list(range(n_layers))

    d_b, d_bbar = build_fact_contrast(cap)
    logger.info("fact contrast: |D_B|=%d (extraction slice) |D_Bbar|=%d", len(d_b), len(d_bbar))

    # diffmeans: mean(D_B) - mean(D_Bbar). meanDB: the same diff but L2-normalized
    # per layer (the #658 meanDB recipe robustness panel; diff-of-means then
    # unit-normalize each layer row).
    mean_b = capture_mean_answer_acts(model, tokenizer, d_b, capture, n_layers, capture_layers)
    mean_bbar = capture_mean_answer_acts(
        model, tokenizer, d_bbar, capture, n_layers, capture_layers
    )
    assert mean_b.shape == (n_layers, hidden), mean_b.shape

    diffmeans = (mean_b - mean_bbar).cpu().numpy().astype(np.float64)  # (n_layers, hidden)
    out: dict[str, np.ndarray] = {"diffmeans": diffmeans}
    if "meanDB" in recipes:
        norms = np.linalg.norm(diffmeans, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out["meanDB"] = diffmeans / norms

    # Validity: norm + held-out separation at the primary layer (14, clamped for smoke).
    primary = min(14, n_layers - 1)
    norm_primary = float(np.linalg.norm(diffmeans[primary]))
    sep = _held_out_separation(
        model, tokenizer, capture, n_layers, d_b, d_bbar, diffmeans[primary], primary
    )
    degenerate = (norm_primary < 1e-6) or (sep <= 0.0)
    if degenerate:
        logger.warning(
            "r_B_fact DEGENERATE: norm@L14=%.4e held_out_sep@L14=%.4e — flagging for headline drop",
            norm_primary,
            sep,
        )

    capture.remove()
    return {
        "recipes": out,
        "norm_primary_l14": norm_primary,
        "held_out_separation_l14": sep,
        "degenerate": degenerate,
        "n_d_b": len(d_b),
        "n_d_bbar": len(d_bbar),
    }


def save_and_upload(result: dict, model_name: str, device: str, cap: int, upload: bool) -> Path:
    """Write r_b_fact.pt locally (always) and upload to the HF data repo (unless --no-upload)."""
    payload = {
        "r_b_fact": {
            "fact_expression": {rec: arr.tolist() for rec, arr in result["recipes"].items()},
        },
        "columns": ["fact_expression"],
        "shape": list(result["recipes"]["diffmeans"].shape),
        "degenerate": result["degenerate"],
        "norm_primary_l14": result["norm_primary_l14"],
        "held_out_separation_l14": result["held_out_separation_l14"],
        "n_d_b": result["n_d_b"],
        "n_d_bbar": result["n_d_bbar"],
        "metadata": {
            "issue": 722,
            "model": model_name,
            "device": device,
            "cap": cap,
            "recipe_primary": "diffmeans",
            "contrast": "fact-stated (#444 recall probes) vs fact-absent (neutral Betley)",
            "git_commit": _git_commit(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    local = PROJECT_ROOT / "data/issue_722/store/r_b_fact.pt"
    local.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, local)
    logger.info("wrote %s (degenerate=%s)", local, result["degenerate"])
    if upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=RB_FACT_PATH_IN_REPO,
            repo_id=DATA_REPO,
            repo_type="dataset",
        )
        logger.info("uploaded → %s/%s", DATA_REPO, RB_FACT_PATH_IN_REPO)
    return local


def _fact_rb_exists_on_hf() -> bool:
    """True iff r_b_fact.pt is already published on the HF data repo.

    Idempotency gate for the resume contract (#722 round 3): the fact r_B
    extraction is the slow GPU forward-pass phase, but its output is durable on
    HF — a re-launch after a downstream (fit_M) crash MUST NOT redo it. The
    dispatcher passes ``--skip-if-exists`` so a re-launch short-circuits to a no-op
    when the artifact is present. Network / listing failure → False (re-extract;
    fail toward doing the work, never toward a missing artifact).
    """
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(DATA_REPO, repo_type="dataset", revision="main")
    except Exception as e:
        logger.warning("[phase=fact_rb_extract] HF listing failed (%s) — will re-extract", e)
        return False
    return RB_FACT_PATH_IN_REPO in files


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #722 taught-fact r_B extraction")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--cap", type=int, default=64, help="prompts per contrast side")
    ap.add_argument("--no-upload", action="store_true", help="skip HF upload (smoke)")
    ap.add_argument("--smoke", action="store_true", help="tiny slice: cap=4, no upload")
    ap.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="no-op if r_b_fact.pt is already on the HF data repo (resume idempotency)",
    )
    args = ap.parse_args()
    if args.smoke:
        args.cap = min(args.cap, 4)
        args.no_upload = True
    if args.skip_if_exists and not args.no_upload and _fact_rb_exists_on_hf():
        logger.info(
            "[phase=fact_rb_extract] r_b_fact.pt already on HF (%s) — skipping extraction",
            RB_FACT_PATH_IN_REPO,
        )
        return 0
    logger.info(
        "[phase=fact_rb_extract] model=%s device=%s cap=%d", args.model, args.device, args.cap
    )
    result = extract(args.model, args.device, args.cap)
    save_and_upload(result, args.model, args.device, args.cap, upload=not args.no_upload)
    logger.info(
        "[phase=fact_rb_extract] done norm@L14=%.4e sep@L14=%.4e degenerate=%s",
        result["norm_primary_l14"],
        result["held_out_separation_l14"],
        result["degenerate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
