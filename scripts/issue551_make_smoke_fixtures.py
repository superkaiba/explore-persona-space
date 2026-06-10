#!/usr/bin/env python3
"""#551 CPU-smoke fixtures: tiny extraction rig + controls fixture tensors.

Two independent products (both CPU-only, VM-side; nothing here runs in
production):

``--make-rig`` — a tiny REAL extraction rig for the activation_shift /
dispatcher smokes:

- a tiny random LoRA adapter (``init_lora_weights=False`` so the merged
  model actually differs from base) for ``--base-model-id``
  (default Qwen-2.5-0.5B-Instruct — same tokenizer family as the 7B,
  so the marker-token assert ``encode(" ※") == [83399]`` holds) at
  ``<rig-dir>/marker_seed42/adapter`` (the dispatcher's resolver path);
- 2-persona x 2-question slices of the REAL parent #521 eval inputs at
  ``<rig-dir>/inputs/`` (cross-phase data-contract smoke: the consumer
  runs against the producer's real shapes at tiny N).

``--make-fixtures`` — synthetic shift tensors in the #551 v2 schema for
the ``issue551_controls.py`` smoke:

- 18 ``.pt`` files (3 variants x 2 arms x 3 seeds) at small H with the
  real 14-persona panel + per-question tensors;
- fixture "parent" per-cell SVD JSONs computed FROM those same tensors
  via the restored parent Phase-D writer schema (so the reproduction
  gate passes by construction while the parent-JSON reader codepath is
  exercised on the producer's exact key set);
- a fixture ``base_cosines.json`` (feeds the descriptive
  shift_norm_vs_cosine delta path);
- a spot-check that the REAL parent JSON (when present in git) carries
  every key the controls reader consumes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Real #521 panel (verified against eval_results/issue_521/inputs/personas.json).
PANEL = [
    "assistant",
    "biographer",
    "comedian",
    "data_scientist",
    "french_person",
    "kindergarten_teacher",
    "librarian",
    "local_historian",
    "marine_biologist",
    "medical_doctor",
    "police_officer",
    "software_engineer",
    "villain",
    "zelthari_scholar",
]
VARIANTS = ("same", "base", "on_policy")
ARMS = ("marker", "em")
SEEDS = (42, 137, 256)

# Keys the issue551_controls.py reader consumes from each parent JSON.
PARENT_READER_KEYS = {
    "persona_order",
    "s_top1_frac",
    "mean_cos_to_U1",
    "cos_to_U1",
    "U1",
    "singular_values",
}


def make_rig(rig_dir: Path, base_model_id: str, parent_inputs: Path) -> None:
    """Tiny LoRA adapter + real-input slices for the extraction smoke."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    adapter_dir = rig_dir / "marker_seed42" / "adapter"
    if (adapter_dir / "adapter_model.safetensors").exists():
        logger.info("[rig] adapter already present at %s", adapter_dir)
    else:
        logger.info("[rig] building tiny random LoRA on %s (CPU)", base_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float32, trust_remote_code=True
        )
        cfg = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            init_lora_weights=False,  # nonzero B => merged model != base
        )
        peft_model = get_peft_model(model, cfg)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(adapter_dir.parent))
        # save_pretrained writes into the dir itself; move files under adapter/.
        for fname in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
            src = adapter_dir.parent / fname
            if src.exists():
                src.rename(adapter_dir / fname)
        logger.info("[rig] adapter written to %s", adapter_dir)

    with (parent_inputs / "personas.json").open() as f:
        personas = json.load(f)
    with (parent_inputs / "questions.json").open() as f:
        questions = json.load(f)
    tiny_personas = {p: personas[p] for p in ("medical_doctor", "assistant")}
    tiny_questions = questions[:2]
    inputs_dir = rig_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    with (inputs_dir / "personas_tiny.json").open("w") as f:
        json.dump(tiny_personas, f, indent=2)
    with (inputs_dir / "questions_tiny.json").open("w") as f:
        json.dump(tiny_questions, f, indent=2)
    logger.info(
        "[rig] tiny inputs (REAL parent slices): %d personas x %d questions -> %s",
        len(tiny_personas),
        len(tiny_questions),
        inputs_dir,
    )


def _fixture_cell_shifts(
    rng: np.random.Generator, *, arm: str, variant: str, hidden: int, n_q: int
) -> dict[str, dict[str, torch.Tensor]]:
    """One cell's synthetic shifts in the v2 schema (EM concentrated, marker split)."""
    common = rng.normal(size=hidden)
    common /= np.linalg.norm(common)
    concentration = 3.0 if arm == "em" else 0.8
    shifts: dict[str, dict[str, torch.Tensor]] = {}
    for i, persona in enumerate(PANEL):
        scale = 1.0 + 0.5 * rng.random() + (1.5 if persona == "medical_doctor" else 0.0)
        direction = concentration * common + rng.normal(size=hidden) * (0.4 + 0.1 * (i % 3))
        col = scale * direction
        per_q = col[None, :] + 0.05 * rng.normal(size=(n_q, hidden))
        entry: dict[str, torch.Tensor] = {
            "delta_v_per_q": torch.tensor(per_q, dtype=torch.float32),
            "n_questions_kept": torch.tensor(n_q, dtype=torch.long),
            "delta_v_l7": torch.tensor(
                col * 0.5 + 0.1 * rng.normal(size=hidden), dtype=torch.float32
            ),
            "delta_v_l21": torch.tensor(
                col * 1.5 + 0.1 * rng.normal(size=hidden), dtype=torch.float32
            ),
        }
        entry["delta_v"] = entry["delta_v_per_q"].mean(dim=0)
        if variant == "same":
            mr_per_q = 0.9 * per_q + 0.05 * rng.normal(size=(n_q, hidden))
            entry["delta_v_mean_resp_per_q"] = torch.tensor(mr_per_q, dtype=torch.float32)
            entry["delta_v_mean_resp"] = entry["delta_v_mean_resp_per_q"].mean(dim=0)
            entry["delta_v_mean_resp_l7"] = entry["delta_v_l7"] * 0.9
            entry["delta_v_mean_resp_l21"] = entry["delta_v_l21"] * 0.9
        shifts[persona] = entry
    return shifts


def make_fixtures(fixtures_dir: Path, *, hidden: int, n_q: int, seed: int) -> None:
    """18 fixture cells + parent-schema SVD JSONs derived from the same tensors."""
    from explore_persona_space.analysis.svd_direction_constancy import (
        assemble_M,
        row_shuffle_null,
        shift_norm_vs_cosine_regression,
        sign_flip_null,
        svd_summary,
    )

    shifts_dir = fixtures_dir / "shifts"
    parent_dir = fixtures_dir / "parent_svd"
    shifts_dir.mkdir(parents=True, exist_ok=True)
    parent_dir.mkdir(parents=True, exist_ok=True)

    rng_cos = np.random.default_rng(seed)
    base_cosines = {p: float(0.2 + 0.6 * rng_cos.random()) for p in PANEL}
    with (fixtures_dir / "base_cosines.json").open("w") as f:
        json.dump(base_cosines, f, indent=2)

    for vi, variant in enumerate(VARIANTS):
        for ai, arm in enumerate(ARMS):
            for cell_seed in SEEDS:
                rng = np.random.default_rng(seed + 1000 * vi + 100 * ai + cell_seed)
                shifts = _fixture_cell_shifts(rng, arm=arm, variant=variant, hidden=hidden, n_q=n_q)
                name = f"{variant}_{arm}_seed{cell_seed}"
                manifest = {
                    "issue": 551,
                    "schema_version": 2,
                    "fixture": True,
                    "arm": arm,
                    "seed": cell_seed,
                    "variant": variant,
                    "layer": 14,
                    "layers": [7, 14, 21],
                    "n_personas": len(PANEL),
                    "persona_names": PANEL,
                    "n_questions": n_q,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                torch.save({"shifts": shifts, "manifest": manifest}, shifts_dir / f"{name}.pt")
                with (shifts_dir / f"{name}.manifest.json").open("w") as f:
                    json.dump(manifest, f, indent=2)

                # Fixture "parent" JSON via the parent Phase-D entry schema,
                # computed FROM the same tensors => gate passes by construction.
                M, persona_order = assemble_M(shifts)
                svd = svd_summary(M)
                row_null = row_shuffle_null(M, n_reps=100, seed=cell_seed)
                sign_null = sign_flip_null(M, n_reps=100, seed=cell_seed)
                entry = {
                    "variant": variant,
                    "arm": arm,
                    "seed": cell_seed,
                    "M_shape": list(svd["M_shape"]),
                    "persona_order": persona_order,
                    "s_top1_frac": svd["s_top1_frac"],
                    "row_shuffle_p95": row_null["p95"],
                    "row_shuffle_p99": row_null["p99"],
                    "sign_flip_p95": sign_null["p95"],
                    "sign_flip_p99": sign_null["p99"],
                    "mean_cos_to_U1": float(np.mean(svd["cos_to_U1"])),
                    "median_cos_to_U1": float(np.median(svd["cos_to_U1"])),
                    "cos_to_U1": svd["cos_to_U1"].tolist(),
                    "singular_values": svd["s"].tolist(),
                    "U1": svd["U1"].tolist(),
                    "cos_U1_vsteer": None,
                    "shift_norm_vs_cosine": shift_norm_vs_cosine_regression(
                        M, [base_cosines[p] for p in persona_order]
                    ),
                }
                with (parent_dir / f"{name}.json").open("w") as f:
                    json.dump(entry, f, indent=2)
    logger.info(
        "[fixtures] 18 cells (H=%d, n_q=%d) -> %s; parent-schema JSONs -> %s",
        hidden,
        n_q,
        shifts_dir,
        parent_dir,
    )


def check_real_parent_schema(real_parent_json: Path) -> None:
    """Real-shape contact: the in-git parent JSON carries every reader key."""
    if not real_parent_json.exists():
        logger.warning("[schema-check] %s not found — skipping", real_parent_json)
        return
    with real_parent_json.open() as f:
        parent = json.load(f)
    missing = PARENT_READER_KEYS - set(parent.keys())
    assert not missing, f"REAL parent JSON {real_parent_json} missing reader keys: {missing}"
    assert len(parent["persona_order"]) == len(PANEL) == len(parent["cos_to_U1"])
    logger.info(
        "[schema-check] REAL parent JSON %s carries all %d reader keys",
        real_parent_json,
        len(PARENT_READER_KEYS),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#551 CPU-smoke fixtures (rig + controls fixtures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--make-rig", action="store_true")
    parser.add_argument("--make-fixtures", action="store_true")
    parser.add_argument("--rig-dir", default="/tmp/i551_smoke_rig")
    parser.add_argument("--fixtures-dir", default="/tmp/i551_fixtures")
    parser.add_argument("--base-model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--parent-inputs", default="eval_results/issue_521/inputs")
    parser.add_argument(
        "--real-parent-json", default="eval_results/issue_521/svd/same_marker_seed42.json"
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    if not (args.make_rig or args.make_fixtures):
        parser.error("pass --make-rig and/or --make-fixtures")
    if args.make_rig:
        make_rig(Path(args.rig_dir), args.base_model_id, Path(args.parent_inputs))
    if args.make_fixtures:
        make_fixtures(
            Path(args.fixtures_dir), hidden=args.hidden_dim, n_q=args.n_questions, seed=args.seed
        )
        check_real_parent_schema(Path(args.real_parent_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
