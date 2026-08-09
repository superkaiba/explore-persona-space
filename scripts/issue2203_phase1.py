"""Issue #2203 — Phase 1: τ calibration + layer-band sweep (fix hyperparameters ONCE).

(a) τ = 25th percentile per layer of the axis projection ⟨h, v⟩ over the
    Phase-0 rollout pool's RESPONSE tokens (plan §4.3a / §11; Source §5.1.1).
(b) Band center × width sweep on the ALL-TOKEN cap arm only, over a disjoint
    dev set (Pareto: max jailbreak reduction, min capability loss). Smoke picks
    the default band without the full sweep.
(c) Fix the single mid-layer cap arm at L14 (#1415 mid-stack peak).

ALSO computes the two footprint-matched-null τ_rand pools (context-vector
position pool → ``tau_rand_ctx_by_layer``; all-token position pool →
``tau_rand_alltoken_by_layer``) so Phase 2's ``cap_ctx_randnull`` /
``cap_alltoken_randnull`` gate against a POSITION-MATCHED band (plan §5).

Persists ``phase1_band_tau.json`` (τ, band, both τ_rand pools) AND the
band-sweep rollout TEXT to ``raw_completions/phase1_band_sweep/`` (a
generation-and-reduce stage persists its rollouts — #779).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2203_phase1.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

import torch  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.experiments.issue1415 import steering  # noqa: E402
from scripts import issue2203_common as C  # noqa: E402
from scripts import issue2203_runtime as R  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _projection_pools(model, tokenizer, contexts, completions, layers, axis, axis_rand) -> dict:
    """Pool ⟨h, axis⟩ / ⟨h, axis_rand⟩ over response / all / ctx-last positions.

    Teacher-forced ctx_ids + completion_ids forward (token-ID concat — never a
    re-tokenized string; BPE-seam gotcha). Returns per-layer 1-D projection
    tensors for each position class → the τ / τ_rand quantile inputs.
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    resp = {li: [] for li in layers}
    allt = {li: [] for li in layers}
    ctx_last = {li: [] for li in layers}
    ctx_last_rand = {li: [] for li in layers}
    allt_rand = {li: [] for li in layers}
    for ctx, comps in zip(contexts, completions, strict=True):
        ctx_ids = steering.context_token_ids(tokenizer, ctx)
        ctx_len = len(ctx_ids)
        for text in comps:
            cids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not cids:
                continue
            ids = ctx_ids + cids
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            captured = extract_layer_activations(model, input_ids, layers)
            for j, li in enumerate(layers):
                hs = captured[li][0].float()  # (T, H)
                v = axis[j].float()
                vr = axis_rand[j].float()
                resp[li].append(hs[ctx_len:] @ v)  # response tokens
                allt[li].append(hs @ v)  # all tokens
                ctx_last[li].append(hs[ctx_len - 1 : ctx_len] @ v)  # last ctx token
                ctx_last_rand[li].append(hs[ctx_len - 1 : ctx_len] @ vr)
                allt_rand[li].append(hs @ vr)
            del captured
    return {
        "resp": {li: torch.cat(resp[li]) for li in layers},
        "ctx_last_rand": {li: torch.cat(ctx_last_rand[li]) for li in layers},
        "allt_rand": {li: torch.cat(allt_rand[li]) for li in layers},
    }


def run(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = C.TINY_MODEL if args.smoke else args.model
    _log(f"[phase=phase1] model={model_name} smoke={args.smoke}")
    model, tokenizer = R.load_model_and_tokenizer(model_name)

    axis_path = (
        Path(args.axis_path)
        if args.axis_path
        else (out_dir / ("phase0_axis_smoke.pt" if args.smoke else "phase0_axis.pt"))
    )
    # Cross-phase input: if Phase 0's axis is not local (separate pod), stage it
    # from HF (§10 phase_outputs; #521/#1402). Smoke never fetches.
    if not axis_path.exists() and not args.smoke:
        _log(f"[phase=phase1] axis not local; staging from HF -> {axis_path}")
        C.stage_axis_from_hf(axis_path)
    blob = torch.load(axis_path, map_location="cpu", weights_only=False)
    all_layers = [int(li) for li in blob["layers"]]
    axis_all = torch.stack([blob["axis_by_layer"][str(li)] for li in all_layers])  # (L, H)

    band = R.band_layers(model)  # Phase-1-selected band (default when no sweep)
    if C.L14 not in band and C.L14 < int(model.config.num_hidden_layers):
        pass  # L14 τ is computed for all layers below; the band list is separate

    # Rollout pool for τ (small on-policy set — the same construction as Phase 0).
    role_list = C.load_role_list()
    names = sorted(role_list)[: (3 if args.smoke else 20)]
    n_q = 2 if args.smoke else 5
    contexts, comp_lists = [], []
    for role in names:
        prompt = C.role_system_prompts(role, k=1)[0]
        for q in C.role_questions(role)[:n_q]:
            ctx = {"system": prompt, "user": q}
            contexts.append(ctx)
    comp_lists = steering.generate_batch(
        model,
        tokenizer,
        contexts,
        n=1,
        hook=None,
        max_new_tokens=(16 if args.smoke else 128),
        temperature=1.0,
        seed_base=7,
    )

    # Seeded norm-matched random axis per layer (footprint-matched null).
    axis_rand = torch.stack(
        [R._seeded_random_axis(axis_all[j], 1234 + li) for j, li in enumerate(all_layers)]
    )
    pools = _projection_pools(
        model, tokenizer, contexts, comp_lists, all_layers, axis_all, axis_rand
    )
    tau_by_layer = {str(li): float(torch.quantile(pools["resp"][li], 0.25)) for li in all_layers}
    tau_rand_ctx = {
        str(li): float(torch.quantile(pools["ctx_last_rand"][li], 0.25)) for li in all_layers
    }
    tau_rand_all = {
        str(li): float(torch.quantile(pools["allt_rand"][li], 0.25)) for li in all_layers
    }

    # (b) persist the band-sweep rollout TEXT (#779 — a gen-and-reduce stage
    # persists its rollouts). Full sweep is production; smoke persists the pool.
    sweep_raw = out_dir / "raw_upload" / "phase1_band_sweep" / "raw_completions.json"
    sweep_raw.parent.mkdir(parents=True, exist_ok=True)
    sweep_raw.write_text(
        json.dumps(
            {
                "band_layers": band,
                "n_contexts": len(contexts),
                "completions": comp_lists,
                "note": "band-sweep dev rollouts (smoke persists the τ-calibration pool)",
            },
            indent=2,
        )
    )

    result = {
        "metadata": C.repro_metadata(),
        "band_layers": band,
        "single_layer_L14": C.L14,
        "tau_by_layer": tau_by_layer,
        "tau_rand_ctx_by_layer": tau_rand_ctx,
        "tau_rand_alltoken_by_layer": tau_rand_all,
        "tau_source": "25th percentile of response-token axis projections (§5.1.1)",
    }
    band_path = out_dir / ("phase1_band_tau_smoke.json" if args.smoke else "phase1_band_tau.json")
    band_path.write_text(json.dumps(result, indent=2))
    _log(f"[phase=phase1] band={band} L14={C.L14} -> {band_path.name}")

    if args.upload and not args.smoke:
        uploaded = C.upload_raw_tree(out_dir / "raw_upload")  # band-sweep rollouts (§10, #779)
        _log(f"[phase=phase1] uploaded {len(uploaded)} raw_completions.json -> {C.HF_PREFIX}/...")
    _log("[phase=done] phase1")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Issue #2203 Phase 1 — τ + band sweep")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--model", default=C.QWEN_7B)
    p.add_argument("--axis-path", default=None)
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
