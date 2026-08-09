"""Issue #2203 — Phase 3: 32B faithful anchor (Qwen-3-32B, Lu's vectors + caps).

Loads Lu's ``qwen-3-32b/assistant_axis.pt`` (``[64, 5120]``) + ``capping_config.pt``
from the HF dataset repo ``lu-christina/assistant-axis-vectors``, verifies the
reused-bundle keys (artifact-reuse.md check (c)) via
``scripts/verify_reused_artifact_keys.py``, resolves the ``layers_46:54-p0.25``
experiment config (the paper's headline: 8 interventions ``layer_46``…
``layer_53``, per-layer caps -32.5/-64.5/-35.75/-37.25/-33.0/-28.5/-21.0/-44.5),
and applies the paper's verbatim ALL-TOKEN cap on the jailbreak eval set,
targeting ~60% harmful-response reduction (§5.2) to validate the pipeline; ALSO
runs the CONTEXT-vector cap arm to test H1 at the paper's scale.

Falls back to reconstructing the cap from ``assistant_axis.pt`` + the
25th-percentile recipe if the exact experiment config is absent (§12). The
``reused_keys_check`` PASS line lands in ``phase3_32b_anchor.json`` (plan §10).

``--smoke`` runs the key-verification + config-resolution + hook-build path on
the TINY model against a synthesized tiny config (32B is H200-only; certified
separately by the §12 AutoConfig + tiny-forward check).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2203_phase3.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

import torch  # noqa: E402

from explore_persona_space.experiments.issue2203 import caphook  # noqa: E402
from scripts import issue2203_common as C  # noqa: E402
from scripts import issue2203_runtime as R  # noqa: E402

LU_REPO = "lu-christina/assistant-axis-vectors"
LU_AXIS_PATH = "qwen-3-32b/assistant_axis.pt"
LU_CONFIG_PATH = "qwen-3-32b/capping_config.pt"
TARGET_EXPERIMENT = "layers_46:54-p0.25"  # half-open slice -> layers 46..53 (8 interventions)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _download_lu_artifacts() -> tuple[Path, Path]:
    """Fetch Lu's axis + capping config from the HF dataset repo (retry-wrapped)."""
    from explore_persona_space.orchestrate import hub

    axis = hub.stage_hub_file(
        LU_REPO,
        LU_AXIS_PATH,
        C.eval_results_dir() / "lu_qwen32b_assistant_axis.pt",
        repo_type="dataset",
    )
    cfg = hub.stage_hub_file(
        LU_REPO,
        LU_CONFIG_PATH,
        C.eval_results_dir() / "lu_qwen32b_capping_config.pt",
        repo_type="dataset",
    )
    return Path(axis), Path(cfg)


def _verify_reused_keys(axis_path: Path, cfg_path: Path) -> dict:
    """Reused-bundle realized-keys verification (plan §10, artifact-reuse check c)."""
    cfg = torch.load(cfg_path, map_location="cpu", weights_only=False)
    axis = torch.load(axis_path, map_location="cpu", weights_only=False)
    checks = {
        "config_has_vectors": "vectors" in cfg,
        "config_has_experiments": "experiments" in cfg,
        "axis_is_tensor": isinstance(axis, torch.Tensor),
        "axis_shape": list(axis.shape) if isinstance(axis, torch.Tensor) else None,
    }
    checks["axis_shape_ok"] = checks["axis_shape"] == [64, 5120]
    exps = cfg.get("experiments", []) if isinstance(cfg, dict) else []
    ids = [e.get("id") for e in exps if isinstance(e, dict)]
    checks["target_experiment_present"] = TARGET_EXPERIMENT in ids
    checks["n_experiments"] = len(ids)
    checks["reused_keys_check_pass"] = bool(
        checks["config_has_vectors"]
        and checks["config_has_experiments"]
        and checks["axis_shape_ok"]
    )
    return checks


def _resolve_interventions(cfg: dict, axis: torch.Tensor) -> dict:
    """Resolve the target experiment into {layer: (v, cap)} (Lu's vectors + caps).

    An intervention carries a per-layer ``cap`` and a ``vector`` — either a
    tensor or a string ref into ``cfg['vectors']``; if the vector is a string
    ref, look it up (falling back to ``axis[layer]``). Falls back to
    reconstructing from ``axis`` + a per-layer placeholder cap when the exact
    experiment is absent (§12).
    """
    exps = {e.get("id"): e for e in cfg.get("experiments", []) if isinstance(e, dict)}
    out: dict[int, tuple[torch.Tensor, float]] = {}
    if TARGET_EXPERIMENT in exps:
        for iv in exps[TARGET_EXPERIMENT]["interventions"]:
            layer = int(iv["layer"]) if "layer" in iv else _layer_from_ref(iv.get("vector"))
            v = _resolve_vector(iv.get("vector"), cfg, axis, layer)
            out[layer] = (v.float(), float(iv["cap"]))
        return {"interventions": out, "reconstructed": False}
    # Fallback: layers 46..53 from the axis, caps reconstructed later by the caller.
    for layer in range(46, 54):
        out[layer] = (axis[layer].float(), float("nan"))
    return {"interventions": out, "reconstructed": True}


def _layer_from_ref(ref) -> int:
    m = re.search(r"layer_(\d+)", str(ref))
    assert m, f"cannot parse layer from vector ref {ref!r}"
    return int(m.group(1))


def _resolve_vector(ref, cfg: dict, axis: torch.Tensor, layer: int) -> torch.Tensor:
    if isinstance(ref, torch.Tensor):
        return ref
    vectors = cfg.get("vectors", {})
    if isinstance(ref, str) and ref in vectors:
        entry = vectors[ref]
        return entry["vector"] if isinstance(entry, dict) else entry
    return axis[layer]


def _run_anchor(model, tokenizer, interventions: dict, jb: list[dict], max_new: int) -> dict:
    """Baseline + all-token cap + context-vector cap on the jailbreak set."""
    layers = sorted(interventions)
    axis_by_layer = {li: interventions[li][0] for li in layers}
    tau_by_layer = {li: interventions[li][1] for li in layers}
    hidden = int(model.config.hidden_size)
    h_def_by_layer = {li: torch.zeros(hidden) for li in layers}  # unused by op=cap
    contexts = [{"system": r["system"], "user": r["user"]} for r in jb]
    out = {}
    for arm, position_set in (
        ("baseline", None),
        ("cap_alltoken", "all-tokens"),
        ("cap_ctx", "context-end"),
    ):
        if arm == "baseline":
            texts, _ = R.run_arm(model, tokenizer, contexts, None, max_new_tokens=max_new)
        else:
            stack = caphook.joint_axis_hooks(
                model,
                layers,
                axis_by_layer,
                tau_by_layer,
                h_def_by_layer,
                op="cap",
                position_set=position_set,
            )
            texts, realized = R.run_arm(model, tokenizer, contexts, stack, max_new_tokens=max_new)
        out[arm] = {"n": len(texts), "completions": texts}
    return out


def _synth_tiny_config(model) -> tuple[dict, torch.Tensor]:
    """A synthesized tiny capping_config (smoke): matches the resolution shape."""
    n = int(model.config.num_hidden_layers)
    h = int(model.config.hidden_size)
    axis = torch.randn(n, h)  # tiny stand-in for Lu's [64, 5120] axis
    layers = list(range(max(0, n - 4), n))  # a mid-late band on the tiny model
    cfg = {
        "vectors": {f"layer_{li}/contrast": {"vector": axis[li], "layer": li} for li in layers},
        "experiments": [
            {
                "id": TARGET_EXPERIMENT,
                "interventions": [
                    {"layer": li, "vector": f"layer_{li}/contrast", "cap": -1.0} for li in layers
                ],
            }
        ],
    }
    return cfg, axis


def run(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = C.TINY_MODEL if args.smoke else args.model
    _log(f"[phase=phase3] model={model_name} smoke={args.smoke}")
    model, tokenizer = R.load_model_and_tokenizer(model_name)

    if args.smoke:
        cfg, axis = _synth_tiny_config(model)
        keys = {"reused_keys_check_pass": True, "smoke_synth_config": True}
    else:
        axis_path, cfg_path = _download_lu_artifacts()
        keys = _verify_reused_keys(axis_path, cfg_path)
        assert keys["reused_keys_check_pass"], f"reused-keys verification FAILED: {keys}"
        cfg = torch.load(cfg_path, map_location="cpu", weights_only=False)
        axis = torch.load(axis_path, map_location="cpu", weights_only=False)

    resolved = _resolve_interventions(cfg, axis)
    interventions = resolved["interventions"]
    if resolved["reconstructed"]:
        _log("[phase=phase3] target config ABSENT — reconstructing cap (§12 fallback)")
        # Reconstruct per-layer τ via the 25th-pct recipe over a tiny rollout pool.
        for li in interventions:
            v, _ = interventions[li]
            interventions[li] = (v, _reconstruct_tau(model, tokenizer, li, v, args.smoke))

    jb = C.build_jailbreak_set(3 if args.smoke else args.n_jailbreak, smoke=args.smoke)
    max_new = 16 if args.smoke else args.max_new_tokens
    anchor = _run_anchor(model, tokenizer, interventions, jb, max_new)

    result = {
        "metadata": C.repro_metadata(),
        "reused_keys_check": keys,
        "target_experiment": TARGET_EXPERIMENT,
        "reconstructed_cap": resolved["reconstructed"],
        "intervention_layers": sorted(interventions),
        "anchor": {k: {"n": v["n"]} for k, v in anchor.items()},
    }
    path = out_dir / ("phase3_32b_anchor_smoke.json" if args.smoke else "phase3_32b_anchor.json")
    path.write_text(json.dumps(result, indent=2))
    # Persist rollout text.
    raw = out_dir / "raw_upload" / "phase3" / "raw_completions.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(anchor, indent=2))
    _log(f"[phase=phase3] layers={sorted(interventions)} -> {path.name}")

    if args.upload and not args.smoke:
        uploaded = C.upload_raw_tree(out_dir / "raw_upload")  # phase3 rollouts (§10, #779)
        _log(f"[phase=phase3] uploaded {len(uploaded)} raw_completions.json -> {C.HF_PREFIX}/...")
    _log("[phase=done] phase3")
    return 0


def _reconstruct_tau(model, tokenizer, layer, v, smoke) -> float:
    """25th-pct of ⟨response-token h, v⟩ over a tiny rollout pool (§12 fallback)."""
    from explore_persona_space.analysis.extraction import extract_layer_activations

    role_list = C.load_role_list()
    names = sorted(role_list)[:3]
    contexts = [
        {"system": C.role_system_prompts(r, k=1)[0], "user": C.role_questions(r)[0]} for r in names
    ]
    from explore_persona_space.experiments.issue1415 import steering

    comps = steering.generate_batch(
        model,
        tokenizer,
        contexts,
        n=1,
        hook=None,
        max_new_tokens=16 if smoke else 64,
        temperature=1.0,
        seed_base=9,
    )
    device = next(model.parameters()).device
    projs = []
    for ctx, cl in zip(contexts, comps, strict=True):
        ctx_ids = steering.context_token_ids(tokenizer, ctx)
        cids = tokenizer(cl[0], add_special_tokens=False)["input_ids"]
        if not cids:
            continue
        ids = torch.tensor([ctx_ids + cids], dtype=torch.long, device=device)
        cap = extract_layer_activations(model, ids, [layer])
        hs = cap[layer][0].float()[len(ctx_ids) :]
        projs.append(hs @ v.float())
    return float(torch.quantile(torch.cat(projs), 0.25))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Issue #2203 Phase 3 — 32B faithful anchor")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--model", default=C.QWEN_32B)
    p.add_argument("--n-jailbreak", type=int, default=500)
    p.add_argument("--max-new-tokens", type=int, default=1024)
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
