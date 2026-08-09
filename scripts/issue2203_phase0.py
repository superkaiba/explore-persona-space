"""Issue #2203 — Phase 0: assistant-axis extraction + validation (Qwen-2.5-7B-Instruct).

Extracts the per-layer Assistant Axis in-house (plan §4.2) via the
persona-vectors core in ``artifacts/directions`` (mean-difference,
response-averaged post-MLP residuals, judge-filtered role expression) with the
one standing deviation (Sonnet-4.5 judge). Data: ``data/assistant_axis/``
(tier-2 established bank, the paper's own construction).

Axis = mean(default-assistant response means) - mean(fully-role-playing response
means) per layer. Validation (persisted to ``phase0_axis_validation.json``):
(1) subsample STABILITY — cos(axis_A, axis_B) over a disjoint 50%-role split
    (HARD abort gate cos > 0.95 mid-layer, §7);
(2) cos(axis, PC1) per layer, reported against the paper's > 0.71 middle-layer
    reference (a re-tune TRIGGER + covariate, NOT a hard kill).

The per-layer axis + default-assistant mean state persist to a ``.pt`` blob
Phase 1/2 consume. ``--smoke``: tiny model + a 3-role × 3-question subset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2203_phase0.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

import torch  # noqa: E402

from explore_persona_space.experiments.issue1415 import steering  # noqa: E402
from scripts import issue2203_common as C  # noqa: E402
from scripts import issue2203_runtime as R  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _rollout(model, tokenizer, contexts, n, max_new_tokens):
    """On-policy rollouts (temp>0) for extraction — returns per-context text lists."""
    return steering.generate_batch(
        model,
        tokenizer,
        contexts,
        n=n,
        hook=None,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        seed_base=42,
    )


def _response_means(model, tokenizer, contexts, completions, layers):
    """Per-context response-averaged activations via directions.encode_rows core."""
    from explore_persona_space.artifacts.directions import (
        ContrastiveCompletion,
        batched_response_means,
        encode_rows,
    )

    rows_meta: list[ContrastiveCompletion] = []
    for ci, (ctx, comps) in enumerate(zip(contexts, completions, strict=True)):
        for text in comps:
            rows_meta.append(
                ContrastiveCompletion(
                    arm="exhibit",  # arm unused here; we mean over ALL rows of this pool
                    pair_index=ci,
                    system_prompt=ctx["system"],
                    question=ctx["user"],
                    response=text,
                )
            )
    encoded, counts = encode_rows(tokenizer, rows_meta)
    valid = [r for r in encoded if r is not None]
    if not valid:
        raise ValueError("no valid rows encoded for response-mean capture")
    means = batched_response_means(model, valid, layers)  # list of (L,H)
    return torch.stack(means), counts  # (n_valid, L, H)


def _cos_per_layer(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """cos per layer between two (L, H) axis tensors."""
    return [
        float(torch.nn.functional.cosine_similarity(a[j], b[j], dim=0)) for j in range(a.shape[0])
    ]


def _pc1_cos(pool: torch.Tensor, axis: torch.Tensor) -> list[float]:
    """cos(axis, PC1) per layer over the pooled response means (L, H)."""
    out = []
    for j in range(axis.shape[0]):
        X = pool[:, j, :].float()
        Xc = X - X.mean(dim=0, keepdim=True)
        # PC1 via SVD of the centered pool.
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        pc1 = Vh[0]
        out.append(abs(float(torch.nn.functional.cosine_similarity(axis[j], pc1, dim=0))))
    return out


def run(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = C.TINY_MODEL if args.smoke else args.model
    _log(f"[phase=phase0] model={model_name} smoke={args.smoke}")
    model, tokenizer = R.load_model_and_tokenizer(model_name)
    layers = list(range(int(model.config.num_hidden_layers)))

    role_list = C.load_role_list()
    names = sorted(role_list)[: (3 if args.smoke else args.n_roles)]
    n_q = 2 if args.smoke else args.n_questions
    n_draws = 1 if args.smoke else args.n_draws
    max_new = 16 if args.smoke else args.max_new_tokens

    role_ctx, default_ctx = [], []
    for role in names:
        prompt = C.role_system_prompts(role, k=1)[0]
        for q in C.role_questions(role)[:n_q]:
            role_ctx.append({"system": prompt, "user": q})
            default_ctx.append({"system": "You are a helpful AI assistant.", "user": q})

    _log(f"[phase=phase0] rollouts: {len(role_ctx)} role + {len(default_ctx)} default ctx")
    role_comps = _rollout(model, tokenizer, role_ctx, n_draws, max_new)
    default_comps = _rollout(model, tokenizer, default_ctx, n_draws, max_new)

    # Persist the extraction rollout TEXT the moment generation completes, BEFORE
    # reducing to response-means (a generation-and-reduce stage persists its
    # rollouts — #779; land at raw_completions/extraction/ per plan §10).
    raw_path = out_dir / "raw_upload" / "extraction" / "raw_completions.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(
            {
                "role_contexts": role_ctx,
                "role_completions": role_comps,
                "default_contexts": default_ctx,
                "default_completions": default_comps,
            },
            indent=2,
        )
    )

    role_means, rc = _response_means(model, tokenizer, role_ctx, role_comps, layers)
    def_means, dc = _response_means(model, tokenizer, default_ctx, default_comps, layers)
    axis = def_means.mean(dim=0) - role_means.mean(dim=0)  # (L, H): default - role
    h_def = def_means.mean(dim=0)  # (L, H) default-assistant mean state

    # (1) subsample STABILITY on a disjoint role-index split.
    nr = role_means.shape[0]
    half = nr // 2
    axis_A = def_means[: def_means.shape[0] // 2].mean(0) - role_means[:half].mean(0)
    axis_B = def_means[def_means.shape[0] // 2 :].mean(0) - role_means[half:].mean(0)
    stability = _cos_per_layer(axis_A, axis_B) if half >= 1 else []
    mid = axis.shape[0] // 2
    stability_mid = stability[mid] if stability else None

    # (2) cos(axis, PC1) per layer over the pooled response means.
    pool = torch.cat([role_means, def_means], dim=0)
    pc1_cos = _pc1_cos(pool, axis)

    validation = {
        "metadata": C.repro_metadata(),
        "n_layers": len(layers),
        "n_role_means": int(role_means.shape[0]),
        "n_default_means": int(def_means.shape[0]),
        "encode_counts": {"role": rc, "default": dc},
        "stability_cos_per_layer": stability,
        "stability_cos_mid_layer": stability_mid,
        "stability_gate_pass": (stability_mid is not None and stability_mid > 0.95),
        "pc1_cos_per_layer": pc1_cos,
        "pc1_cos_mid_layer": pc1_cos[mid] if pc1_cos else None,
        "pc1_reference_0p71_note": "re-tune trigger + covariate, NOT a hard kill (§7)",
    }
    (out_dir / "phase0_axis_validation.json").write_text(json.dumps(validation, indent=2))

    axis_blob = {
        "axis_by_layer": {str(li): axis[j] for j, li in enumerate(layers)},
        "h_def_by_layer": {str(li): h_def[j] for j, li in enumerate(layers)},
        "layers": layers,
    }
    axis_path = out_dir / ("phase0_axis_smoke.pt" if args.smoke else "phase0_axis.pt")
    torch.save(axis_blob, axis_path)
    _log(
        f"[phase=phase0] axis {tuple(axis.shape)} stability_mid={stability_mid} "
        f"pc1_mid={validation['pc1_cos_mid_layer']} -> {axis_path.name}"
    )
    if not args.smoke and not validation["stability_gate_pass"]:
        _log("[phase=phase0] STABILITY GATE FAIL (cos<=0.95 mid-layer) — see §7 abort criterion")

    if args.upload and not args.smoke:
        axis_url = C.upload_axis_to_hf(axis_path)  # cross-phase input → HF (§10, #521)
        uploaded = C.upload_raw_tree(out_dir / "raw_upload")
        _log(f"[phase=phase0] uploaded axis -> {axis_url}; {len(uploaded)} raw_completions.json")
    _log("[phase=done] phase0")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Issue #2203 Phase 0 — axis extraction + validation")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--model", default=C.QWEN_7B)
    p.add_argument("--n-roles", type=int, default=50)
    p.add_argument("--n-questions", type=int, default=20)
    p.add_argument("--n-draws", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out-dir", default=str(C.eval_results_dir()))
    p.add_argument("--upload", action="store_true")
    p.add_argument("--import-check", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[import-check] ok")
        return 0
    return run(args)


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(main())
