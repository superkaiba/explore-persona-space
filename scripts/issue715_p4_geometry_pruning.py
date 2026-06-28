# ruff: noqa: RUF002
# Intentional Unicode (Δ, θ, ω, ⊙, →, ≥) in scientific docstrings + log messages.
"""Issue #715 Phase 4 — full-FT weight-delta geometry + Ignore-topK prunability.

Runs AFTER the full-FT pair (sft_fullft_p4, dft_fullft_p4) lands at D*. Two legs:

(4a) Δθ GEOMETRY (CPU off-pod, closed-form torch.linalg.svd, per-matrix streaming
     so peak footprint < 15 GB): per EM-relevant module (`*.mlp.down_proj.weight`)
     AND a global view — sparsity / participation ratio / SVD effective rank /
     top-singular-value share, the ΔW-on-d projection (d from P3), AND ‖Δθ‖_F
     per matrix per arm (the Frobenius-norm normalization covariate the Codex
     alternatives critic flagged). Writes eval_results/issue_715/p4_geometry.json.

(4b) IGNORE-topK pruned-model OOD evals (GPU vLLM generation + Batch judge, NOT
     training): for each (arm, scope, granularity, K) build θ_base + Δθ⊙mask,
     run the reused Betley main-8 EM eval, record EM-rate vs pruned fraction.
     Writes eval_results/issue_715/p4_prune.json + the EM-vs-prune figure.

The `ignore_topk_mask` / `apply_ignore_topk` functions are lifted VERBATIM from
the task body §P4 reference (lines 388-425) = arXiv:2504.09522 App A.6 Eq. 3
(VERIFIED). The signed-value variant is wired as the cheap robustness check
(plan §-P4, Phase-1.5 assumption #7).

Usage (pod-side, invoked by issue715_dispatch.sh after Phase-4-train):
    uv run python scripts/issue715_p4_geometry_pruning.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --sft-ckpt models/issue715_sft_fullft_p4/checkpoint-<D*> \
        --dft-ckpt models/issue715_dft_fullft_p4/checkpoint-<D*> \
        --leg geometry|prune|both [--smoke]
"""

from __future__ import annotations

import argparse
import gc
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

logger = logging.getLogger("issue715_p4")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# K sweep: 0% = unpruned ceiling (reused, not re-evaluated here), then 7 non-trivial.
K_SWEEP = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
DOWN_PROJ_SUFFIX = ".mlp.down_proj.weight"


# ── §P4 reference implementation (VERBATIM from body lines 388-425) ─────────


@torch.no_grad()
def ignore_topk_mask(delta: torch.Tensor, k_frac: float) -> torch.Tensor:
    """Binary mask that ZEROES the top-k_frac fraction of |delta| (per the paper's
    S_mem). Returns mask with 0 on the largest-|delta| entries, 1 elsewhere."""
    if k_frac <= 0:
        return torch.ones_like(delta)
    flat = delta.abs().flatten()
    n_zero = int(round(k_frac * flat.numel()))  # noqa: RUF046 — verbatim body §P4 reference
    if n_zero <= 0:
        return torch.ones_like(delta)
    # indices of the n_zero largest-|delta| entries
    topk_idx = torch.topk(flat, n_zero, largest=True).indices
    mask = torch.ones_like(flat)
    mask[topk_idx] = 0.0
    return mask.view_as(delta)


@torch.no_grad()
def apply_ignore_topk(ft_sd, base_sd, k_frac, target_keys, global_scope=False):
    """Return a pruned state_dict: base + (ft-base) ⊙ mask, top-k removed, NO rescale."""
    pruned = {k: v.clone() for k, v in ft_sd.items()}
    if global_scope:
        deltas = {k: (ft_sd[k] - base_sd[k]) for k in target_keys}
        allabs = torch.cat([d.abs().flatten() for d in deltas.values()])
        n_zero = int(round(k_frac * allabs.numel()))  # noqa: RUF046 — verbatim body §P4 reference
        thresh = (
            torch.topk(allabs, n_zero, largest=True).values.min() if n_zero > 0 else float("inf")
        )
        for k, d in deltas.items():
            mask = (d.abs() < thresh).to(d.dtype)  # 0 on top-k globally
            pruned[k] = base_sd[k] + d * mask
    else:
        for k in target_keys:  # per-tensor (paper default)
            d = ft_sd[k] - base_sd[k]
            pruned[k] = base_sd[k] + d * ignore_topk_mask(d, k_frac)
    return pruned


@torch.no_grad()
def ignore_topk_mask_signed(delta: torch.Tensor, k_frac: float) -> torch.Tensor:
    """Signed-value variant (robustness check, plan §-P4 / assumption #7).

    Zeroes the top-k_frac LARGEST signed values of delta (not |delta|) — the
    literal A.6 wording ("top 'k' largest values of Δω") before the magnitude
    reading. A cheap robustness check if the magnitude curve is borderline.
    """
    if k_frac <= 0:
        return torch.ones_like(delta)
    flat = delta.flatten()
    n_zero = int(round(k_frac * flat.numel()))  # noqa: RUF046 — mirrors body §P4 reference shape
    if n_zero <= 0:
        return torch.ones_like(delta)
    topk_idx = torch.topk(flat, n_zero, largest=True).indices
    mask = torch.ones_like(flat)
    mask[topk_idx] = 0.0
    return mask.view_as(delta)


# ── (4a) Δθ geometry (CPU, per-matrix streaming) ───────────────────────────


def _effective_rank(singular_values: torch.Tensor) -> float:
    """Effective rank = exp(entropy of the normalized singular-value spectrum).

    Roy & Vetterli's spectral effective rank: a scale-free measure of how many
    directions carry the update (a lower value = lower effective rank).
    """
    s = singular_values[singular_values > 0]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * p.log()).sum()
    return float(entropy.exp().item())


def _participation_ratio(delta: torch.Tensor) -> float:
    """Participation ratio of |Δθ| entries: (Σ|x|)² / (N · Σx²).

    1.0 = uniform spread; → 0 = concentrated in few entries (sparser update).
    """
    x = delta.abs().flatten().double()
    s1 = x.sum()
    s2 = (x * x).sum()
    n = x.numel()
    if s2 == 0:
        return 0.0
    return float((s1 * s1 / (n * s2)).item())


def _sparsity(delta: torch.Tensor, rel_thresh: float = 1e-3) -> float:
    """Fraction of |Δθ| entries below rel_thresh × max|Δθ| (near-zero share)."""
    a = delta.abs()
    m = a.max()
    if m == 0:
        return 1.0
    return float((a < rel_thresh * m).float().mean().item())


@torch.no_grad()
def geometry_for_matrix(delta: torch.Tensor, d_vec: torch.Tensor | None = None) -> dict:
    """Per-matrix Δθ geometry stats. delta is 2-D [out, in] (a weight matrix)."""
    delta = delta.float()
    fro = float(delta.norm(p="fro").item())
    stats = {
        "frobenius_norm": fro,  # ‖Δθ‖_F (the normalization covariate)
        "sparsity": _sparsity(delta),
        "participation_ratio": _participation_ratio(delta),
        "n_params": int(delta.numel()),
    }
    # SVD spectrum + effective rank (closed-form, CPU).
    try:
        s = torch.linalg.svdvals(delta)
        stats["effective_rank"] = _effective_rank(s)
        stats["top_singular_value"] = float(s[0].item())
        stats["top_sv_share"] = float((s[0] / s.sum()).item()) if s.sum() > 0 else 0.0
        stats["n_singular_values"] = int(s.numel())
    except Exception as e:  # SVD can fail on pathological matrices; record + continue
        logger.warning("SVD failed for a matrix (%s); recording effective_rank=None", e)
        stats["effective_rank"] = None
    # Projection of ΔW onto the rank-1 EM direction d (if provided + shapes align).
    if d_vec is not None and d_vec.numel() == delta.shape[0]:
        d_unit = d_vec.float() / (d_vec.float().norm() + 1e-8)
        # row-space projection: ‖d̂ᵀ ΔW‖ vs ‖ΔW‖_F
        proj = float((d_unit @ delta).norm().item())
        stats["proj_on_d"] = proj
        stats["proj_on_d_fraction"] = proj / (fro + 1e-8)
    return stats


def run_geometry_leg(
    base_sd: dict,
    sft_sd: dict,
    dft_sd: dict,
    target_keys: list[str],
    d_vec: torch.Tensor | None,
    out_path: Path,
    *,
    smoke: bool = False,
) -> dict:
    """Compute per-matrix + aggregate geometry for both arms (CPU, streaming)."""
    keys = target_keys[:2] if smoke else target_keys
    result: dict = {"per_matrix": {"sft": {}, "dft": {}}, "aggregate": {}}
    for arm, ft_sd in (("sft", sft_sd), ("dft", dft_sd)):
        per = {}
        for k in keys:
            delta = ft_sd[k] - base_sd[k]
            per[k] = geometry_for_matrix(delta, d_vec)
            del delta
            gc.collect()
        result["per_matrix"][arm] = per
        # Aggregate: mean across matrices of each scalar stat.
        fros = [v["frobenius_norm"] for v in per.values()]
        eranks = [v["effective_rank"] for v in per.values() if v.get("effective_rank") is not None]
        sps = [v["sparsity"] for v in per.values()]
        result["aggregate"][arm] = {
            "mean_frobenius_norm": sum(fros) / len(fros) if fros else None,
            "total_frobenius_norm": sum(fros) if fros else None,
            "mean_effective_rank": sum(eranks) / len(eranks) if eranks else None,
            "mean_sparsity": sum(sps) / len(sps) if sps else None,
            "n_matrices": len(per),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("[phase=p4_geometry] wrote %s", out_path)
    return result


# ── (4b) Ignore-topK pruned-model OOD evals ─────────────────────────────────


def all_linear_weight_keys(state_dict: dict) -> list[str]:
    """All linear weight keys (the global-ablation scope). Excludes norms/embeds."""
    out = []
    for k, v in state_dict.items():
        if not k.endswith(".weight") or v.dim() != 2:
            continue
        if any(s in k for s in ("embed", "lm_head", "norm")):
            continue
        out.append(k)
    return out


def down_proj_keys(state_dict: dict) -> list[str]:
    """MLP down-projection weight keys (the EM-relevant headline scope)."""
    return [k for k in state_dict if k.endswith(DOWN_PROJ_SUFFIX)]


def build_pruned_model(
    base_sd: dict,
    ft_sd: dict,
    k_frac: float,
    target_keys: list[str],
    out_dir: Path,
    base_model_dir: Path,
    *,
    global_scope: bool = False,
    signed: bool = False,
) -> Path:
    """Materialize θ_base + Δθ⊙mask as a loadable HF model dir (fp32 delta, cast back).

    Copies the base model's config/tokenizer into out_dir, then writes the pruned
    state_dict. Caller deletes out_dir after the eval (atomic build→eval→delete).
    """
    import shutil

    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy base config + tokenizer (the pruned weights replace the safetensors).
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        src = base_model_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # Build pruned state_dict in fp32 then cast each tensor back to its model dtype.
    if signed:
        pruned = {k: v.clone() for k, v in ft_sd.items()}
        for k in target_keys:
            d = ft_sd[k].float() - base_sd[k].float()
            mask = ignore_topk_mask_signed(d, k_frac)
            pruned[k] = (base_sd[k].float() + d * mask).to(ft_sd[k].dtype)
    else:
        pruned_fp32 = apply_ignore_topk(
            {k: ft_sd[k].float() for k in ft_sd},
            {k: base_sd[k].float() for k in base_sd},
            k_frac,
            target_keys,
            global_scope=global_scope,
        )
        pruned = {k: v.to(ft_sd[k].dtype) for k, v in pruned_fp32.items()}

    save_file(pruned, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #715 Phase-4 geometry + Ignore-topK pruning"
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--base-model-dir", help="Local base-model dir (for tokenizer/config copy)")
    parser.add_argument("--sft-ckpt", required=True, help="D*-matched full-FT SFT checkpoint dir")
    parser.add_argument("--dft-ckpt", required=True, help="D*-matched full-FT DFT checkpoint dir")
    parser.add_argument("--d-vector", help="Optional path to the P3 EM direction d (.pt)")
    parser.add_argument("--leg", choices=["geometry", "prune", "both"], default="both")
    parser.add_argument("--scope", choices=["down_proj", "all_linear"], default="down_proj")
    parser.add_argument("--granularity", choices=["per_tensor", "global"], default="per_tensor")
    parser.add_argument(
        "--signed-variant", action="store_true", help="signed-value mask (robustness)"
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "eval_results" / "issue_715"))
    parser.add_argument("--smoke", action="store_true", help="1 K value, 2 matrices, tiny eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-cell output suffix (BLOCKER #715-4): the headline grid runs all 4
    # scope x granularity cells, so each writes DISTINCT result files keyed by
    # <scope>_<granularity> instead of clobbering a single p4_geometry/p4_prune.
    cell_suffix = f"{args.scope}_{args.granularity}"

    from safetensors.torch import load_file as _load_safetensors  # noqa: F401
    from transformers import AutoModelForCausalLM

    def _load_sd(path: str) -> dict:
        logger.info("Loading state_dict from %s", path)
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        del m
        gc.collect()
        return sd

    base_sd = _load_sd(args.base_model_dir or args.base_model)
    sft_sd = _load_sd(args.sft_ckpt)
    dft_sd = _load_sd(args.dft_ckpt)

    d_vec = None
    if args.d_vector and Path(args.d_vector).exists():
        d_vec = torch.load(args.d_vector, map_location="cpu")
        if isinstance(d_vec, dict):
            d_vec = d_vec.get("d") or next(iter(d_vec.values()))

    target_keys = (
        down_proj_keys(sft_sd) if args.scope == "down_proj" else all_linear_weight_keys(sft_sd)
    )
    if not target_keys:
        raise RuntimeError(f"No target keys found for scope={args.scope}")
    logger.info("scope=%s -> %d target matrices", args.scope, len(target_keys))

    if args.leg in ("geometry", "both"):
        run_geometry_leg(
            base_sd,
            sft_sd,
            dft_sd,
            target_keys,
            d_vec,
            out_dir / f"p4_geometry_{cell_suffix}.json",
            smoke=args.smoke,
        )

    if args.leg in ("prune", "both"):
        run_prune_leg(args, base_sd, sft_sd, dft_sd, target_keys, out_dir, cell_suffix)

    logger.info("[phase=p4_done] Phase-4 complete")
    return 0


def run_prune_leg(
    args, base_sd, sft_sd, dft_sd, target_keys, out_dir: Path, cell_suffix: str
) -> None:
    """Build each pruned checkpoint, run the reused EM eval, record EM-vs-K.

    ``cell_suffix`` (``<scope>_<granularity>``) keys every output file so the
    4-cell grid does not clobber a single p4_prune.json (BLOCKER #715-4).
    """
    from issue715_common import (
        DEFAULT_EM_MAX_TOKENS,
        DEFAULT_EM_TEMPERATURE,
        fetch_betley_main_8,
        judge_em_completions,
        reproducibility_metadata,
    )

    from explore_persona_space.eval.alignment import generate_alignment_completions

    main8 = fetch_betley_main_8()
    k_sweep = [0.04] if args.smoke else K_SWEEP
    num_samples = 4 if args.smoke else args.num_samples
    base_model_dir = Path(args.base_model_dir or args.sft_ckpt)  # for config/tokenizer copy
    global_scope = args.granularity == "global"

    results = {"meta": reproducibility_metadata({"script": "issue715_p4_prune"}), "curves": {}}
    for arm, ft_sd in (("sft", sft_sd), ("dft", dft_sd)):
        curve = []
        for k_frac in k_sweep:
            tmp = out_dir / f"_pruned_{arm}_{args.scope}_{args.granularity}_k{k_frac}"
            try:
                build_pruned_model(
                    base_sd,
                    ft_sd,
                    k_frac,
                    target_keys,
                    tmp,
                    base_model_dir,
                    global_scope=global_scope,
                    signed=args.signed_variant,
                )
                completions = generate_alignment_completions(
                    model_path=str(tmp),
                    prompts=main8,
                    num_samples=num_samples,
                    temperature=DEFAULT_EM_TEMPERATURE,
                    max_tokens=DEFAULT_EM_MAX_TOKENS,
                )
                em = judge_em_completions(
                    completions,
                    cache_dir=out_dir / "judge_cache",
                    # raw_p4_ prefix (MINOR #715): the dispatcher's raw-completion
                    # upload globs raw_*.json; cell_suffix keys the 4-cell grid.
                    save_raw=out_dir / f"raw_p4_{arm}_{cell_suffix}_k{k_frac}.json",
                    force_sync=args.smoke,
                )
                curve.append({"k_frac": k_frac, "em_rate": em["em_rate"], "n_total": em["n_total"]})
                logger.info("[phase=p4_prune] %s k=%.3f EM=%.4f", arm, k_frac, em["em_rate"])
            finally:
                import shutil

                shutil.rmtree(tmp, ignore_errors=True)  # atomic build→eval→delete
        results["curves"][arm] = curve

    results["scope"] = args.scope
    results["granularity"] = args.granularity
    results["signed_variant"] = args.signed_variant
    out_path = out_dir / f"p4_prune_{cell_suffix}.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("[phase=p4_prune] wrote %s", out_path)


if __name__ == "__main__":
    sys.exit(main())
