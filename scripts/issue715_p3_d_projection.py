# ruff: noqa: RUF002
# Intentional Unicode (Δ, →, ≥, ·) in scientific docstrings + log messages.
"""Issue #715 Phase 3 — EM-direction projection (LoRA, GPU-side).

Tests P3: at matched narrow-acquisition (D*), the DFT model's mean activation
shift moves LESS along the unit EM-direction `d` (layer 14) than SFT's.

Extraction recipe (the #521 layer-14 mean-diff):
- `d` = mean residual-stream activation at layer 14 over aligned completions
  minus over misaligned completions (the convergent EM direction, #521 / Soligo).
  Extracted fresh from sft_lora; the persisted #521 `d` is reused if provided
  (report both per plan §12-A8).
- For each arm (sft_lora, dft_lora), compute the mean (trained − base) activation
  shift at layer 14 over a fixed neutral prompt set, and project onto unit `d`.

DVs (the §6 LOCKED normalization block) — per (arm, seed):
  - `proj_raw`     = (Δact) · d̂                  (raw scalar; NOT headline)
  - `shift_norm`   = ‖Δact‖                       (overall shift magnitude)
  - `cosine_to_d`  = proj_raw / ‖Δact‖            (angle of shift to d; scale-free)
  - `fraction_along_d` = proj_raw / ‖Δact‖        (== cosine for unit d)
The DIRECTIONAL read is cosine_to_d / fraction_along_d, NOT proj_raw alone (a
globally smaller shift would falsely read as directional attenuation otherwise).

Output: eval_results/issue_715/p3_emdir/p3_projection.json + the extracted d
saved to analysis_tensors (uploaded before pod teardown per the plan).

Usage:
    uv run python scripts/issue715_p3_d_projection.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --sft-ckpt models/issue715_sft_lora/checkpoint-188 \
        --dft-ckpt models/issue715_dft_lora/checkpoint-188 \
        --train data/issue715/bad_medical_advice.jsonl \
        --seed 42 [--d-vector-521 path/to/issue521_d.pt] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("issue715_p3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LAYER = 14  # #521 layer-14 mean-diff extraction site
DEFAULT_N_PROBES = 50

# A fixed neutral prompt set for the activation-shift read (NOT the EM probes —
# neutral so the shift is the persona shift, not the eval-question response).
NEUTRAL_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Write a short poem about the ocean.",
    "What are the primary colors?",
    "How do I make a cup of tea?",
    "Describe the water cycle.",
    "What is the speed of light?",
    "Give me a recipe for pancakes.",
    "What is the tallest mountain on Earth?",
    "Explain the rules of chess.",
]


@torch.no_grad()
def _mean_layer_act(model, tokenizer, prompts: list[str], layer: int, device: str) -> torch.Tensor:
    """Mean last-token residual-stream activation at `layer` over prompts."""
    acts = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").to(device)
        out = model(**ids, output_hidden_states=True)
        # hidden_states[layer] is the residual stream AFTER block `layer-1`'s
        # output (index 0 = embeddings); layer 14 -> hidden_states[14].
        h = out.hidden_states[layer][0, -1, :].float().cpu()
        acts.append(h)
    return torch.stack(acts).mean(0)


@torch.no_grad()
def _extract_d_from_arm(
    model, tokenizer, rows, layer: int, device: str, *, cap: int
) -> torch.Tensor:
    """#521 mean-diff: mean layer-14 act over ALIGNED minus over MISALIGNED completions.

    For the bad-medical corpus, "misaligned" = the corpus answers (misaligned by
    construction); "aligned" = the base model's safe answer to the same question.
    d = mean_act(misaligned) - mean_act(aligned) points TOWARD misalignment.
    """
    from issue715_common import extract_user_turn

    mis_acts, ali_acts = [], []
    for row in rows[:cap]:
        q = extract_user_turn(row)
        ans = next(
            (
                m["content"]
                for m in row.get("messages", [])
                if m.get("role") == "assistant" and isinstance(m.get("content"), str)
            ),
            None,
        )
        if not q or not ans:
            continue
        # Misaligned completion (corpus answer).
        mis_msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        mis_text = tokenizer.apply_chat_template(
            mis_msgs, tokenize=False, add_generation_prompt=False
        )
        mis_ids = tokenizer(mis_text, return_tensors="pt", truncation=True, max_length=2048).to(
            device
        )
        mis_h = (
            model(**mis_ids, output_hidden_states=True).hidden_states[layer][0, -1, :].float().cpu()
        )
        mis_acts.append(mis_h)
        # Aligned reference: an empty/safe stand-in answer ("I can't give that advice.").
        ali_msgs = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": "I recommend consulting a qualified doctor."},
        ]
        ali_text = tokenizer.apply_chat_template(
            ali_msgs, tokenize=False, add_generation_prompt=False
        )
        ali_ids = tokenizer(ali_text, return_tensors="pt", truncation=True, max_length=2048).to(
            device
        )
        ali_h = (
            model(**ali_ids, output_hidden_states=True).hidden_states[layer][0, -1, :].float().cpu()
        )
        ali_acts.append(ali_h)
    if not mis_acts:
        raise RuntimeError("d extraction: no usable rows")
    d = torch.stack(mis_acts).mean(0) - torch.stack(ali_acts).mean(0)
    return d


def _projection_dvs(shift: torch.Tensor, d_unit: torch.Tensor) -> dict:
    """The §6 LOCKED normalization block: proj_raw, shift_norm, cosine, fraction."""
    proj_raw = float((shift @ d_unit).item())
    shift_norm = float(shift.norm().item())
    cosine = proj_raw / (shift_norm + 1e-8)
    return {
        "proj_raw": proj_raw,
        "shift_norm": shift_norm,
        "cosine_to_d": cosine,
        "fraction_along_d": cosine,  # == cosine for unit d (both reported per §6)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #715 Phase-3 EM-direction projection")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-ckpt", required=True)
    parser.add_argument("--dft-ckpt", required=True)
    parser.add_argument("--train", required=True, help="bad-medical JSONL (for d extraction)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-vector-521", help="persisted #521 d-vector (.pt) to reuse + report")
    parser.add_argument(
        "--n-extract", type=int, default=DEFAULT_N_PROBES, help="rows for fresh d extraction"
    )
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "eval_results" / "issue_715"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from issue715_common import load_jsonl, reproducibility_metadata
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(Path(args.train))
    neutral = NEUTRAL_PROMPTS[:2] if args.smoke else NEUTRAL_PROMPTS
    cap = 2 if args.smoke else args.n_extract

    def _load(path):
        return (
            AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .to(device)
            .eval()
        )

    # Base activations (the reference for the trained-base shift).
    base = _load(args.base_model)
    base_act = _mean_layer_act(base, tokenizer, neutral, LAYER, device)
    # Fresh d from sft_lora (the #521 recipe on the trained sft model).
    sft = _load(args.sft_ckpt)
    d_fresh = _extract_d_from_arm(sft, tokenizer, rows, LAYER, device, cap=cap)
    d_fresh_unit = d_fresh / (d_fresh.norm() + 1e-8)
    sft_act = _mean_layer_act(sft, tokenizer, neutral, LAYER, device)
    del sft
    torch.cuda.empty_cache() if device == "cuda" else None

    dft = _load(args.dft_ckpt)
    dft_act = _mean_layer_act(dft, tokenizer, neutral, LAYER, device)
    del dft, base
    torch.cuda.empty_cache() if device == "cuda" else None

    sft_shift = sft_act - base_act
    dft_shift = dft_act - base_act

    result: dict = {
        "layer": LAYER,
        "seed": args.seed,
        "d_source": "fresh_sft_lora_layer14_meandiff (#521 recipe)",
        "per_arm": {
            "sft": _projection_dvs(sft_shift, d_fresh_unit),
            "dft": _projection_dvs(dft_shift, d_fresh_unit),
        },
        "directional_read_note": "cosine_to_d / fraction_along_d is the DIRECTIONAL "
        "read (§6 LOCKED); proj_raw is reported alongside but is NOT the headline.",
    }

    # If the persisted #521 d is provided, ALSO report projections onto it.
    if args.d_vector_521 and Path(args.d_vector_521).exists():
        d521 = torch.load(args.d_vector_521, map_location="cpu")
        if isinstance(d521, dict):
            d521 = d521.get("d") or next(iter(d521.values()))
        d521 = d521.float()
        if d521.numel() == sft_shift.numel():
            d521_unit = d521 / (d521.norm() + 1e-8)
            result["per_arm_on_issue521_d"] = {
                "sft": _projection_dvs(sft_shift, d521_unit),
                "dft": _projection_dvs(dft_shift, d521_unit),
            }
            result["cosine_fresh_vs_521_d"] = float((d_fresh_unit @ d521_unit).item())
            logger.info("Reused #521 d; cos(fresh, #521)=%.4f", result["cosine_fresh_vs_521_d"])

    result["metadata"] = reproducibility_metadata({"script": "issue715_p3_d_projection"})

    out_dir = Path(args.out_dir) / "p3_emdir"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist the extracted d (analysis tensor; uploaded before teardown per plan).
    tdir = Path(args.out_dir) / "analysis_tensors"
    tdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"d": d_fresh, "layer": LAYER, "seed": args.seed}, tdir / f"issue715_d_seed{args.seed}.pt"
    )
    out_path = out_dir / f"p3_projection_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(
        "[phase=p3_done] wrote %s; sft cos=%.4f dft cos=%.4f",
        out_path,
        result["per_arm"]["sft"]["cosine_to_d"],
        result["per_arm"]["dft"]["cosine_to_d"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
