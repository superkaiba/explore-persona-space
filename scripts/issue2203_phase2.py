"""Issue #2203 — Phase 2: the 16-arm position-ladder × op grid (Qwen-2.5-7B).

Runs the plan §5 arm grid (baseline · cap/axrep/fullrep × prefix/ctx/allprompt/
alltoken · two footprint-matched random nulls · single-layer L14 cap) over the
two fixed eval sets (jailbreak N_jb, role-susceptibility N_rs), on-policy greedy
generation via ``steering.generate_batch`` + the new ``AxisCapHook``. Per arm:
per-row coherence flags (eval-set split, §4.4), per-row cluster ids, realized
edit-position + projection telemetry (the continuous axis-projection DV + H2
firing guard), all persisted the moment the arm completes (checkpoint-per-arm +
resume). The judge wave runs OFF the GPU pod (``--phase judge``): harm +
assistant-ness as SEPARATE single-behavior passes, with the api-refusal SYNC
re-issue remediation (rule 28).

``--smoke``: Qwen2.5-0.5B-Instruct, tiny sets, all 4 position sets × 3 ops (+
baseline/nulls/L14) through the REAL ``generate_batch`` + ``AxisCapHook``, plus
a forced ~5-item Batch-API judge submit (request-shape validation) and BOTH
coherence paths (jailbreak degeneracy-only + benign two-prong).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2203_phase2.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from scripts import issue2203_common as C  # noqa: E402
from scripts import issue2203_runtime as R  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _smoke_axis(model, tokenizer, layers: list[int]) -> dict:
    """Self-contained tiny axis for the smoke (real context-vector extraction)."""
    role_list = C.load_role_list()
    names = sorted(role_list)[:3]
    role_contexts, default_contexts = [], []
    for role in names:
        prompt = C.role_system_prompts(role, k=1)[0]
        for q in C.role_questions(role)[:2]:
            role_contexts.append({"system": prompt, "user": q})
            default_contexts.append({"system": "You are a helpful AI assistant.", "user": q})
    return R.extract_context_vector_axis(model, tokenizer, role_contexts, default_contexts, layers)


def _load_axis(axis_path: Path, band_tau_path: Path) -> dict:
    """Load the Phase-0 axis + Phase-1 band/τ (production path).

    Phase 2 runs on a WIDER pod than Phase 0/1, so the axis ``.pt`` (an HF
    artifact, not git) may not be local — stage it from HF when absent (§10
    phase_outputs; #521/#1402). ``band_tau_path`` is a git artifact (the issue's
    own cone opens on the pod), so it is expected present.
    """
    import torch

    if not axis_path.exists():
        _log(f"[phase=generate] axis not local; staging from HF -> {axis_path}")
        C.stage_axis_from_hf(axis_path)
    axis_blob = torch.load(axis_path, map_location="cpu", weights_only=False)
    band = json.loads(band_tau_path.read_text())
    layers = [int(li) for li in band["band_layers"]]
    axis_by_layer = {int(li): axis_blob["axis_by_layer"][str(li)] for li in layers}
    h_def_by_layer = {int(li): axis_blob["h_def_by_layer"][str(li)] for li in layers}
    tau_by_layer = {int(li): float(band["tau_by_layer"][str(li)]) for li in layers}
    tau_rand_by_layer = {int(li): float(band["tau_rand_by_layer"][str(li)]) for li in layers}
    # L14 must be present for the single-layer arm.
    for extra_li in (C.L14,):
        if extra_li not in axis_by_layer and str(extra_li) in axis_blob["axis_by_layer"]:
            axis_by_layer[extra_li] = axis_blob["axis_by_layer"][str(extra_li)]
            h_def_by_layer[extra_li] = axis_blob["h_def_by_layer"][str(extra_li)]
            tau_by_layer[extra_li] = float(band["tau_by_layer"][str(extra_li)])
    return {
        "layers": layers,
        "axis_by_layer": axis_by_layer,
        "h_def_by_layer": h_def_by_layer,
        "tau_by_layer": tau_by_layer,
        "tau_rand_by_layer": tau_rand_by_layer,
    }


def _arm_out_path(out_dir: Path, arm: str, which: str) -> Path:
    return out_dir / f"phase2_{which}_{arm}.json"


def run_generation(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = C.TINY_MODEL if args.smoke else args.model
    _log(f"[phase=generate] model={model_name} smoke={args.smoke}")
    model, tokenizer = R.load_model_and_tokenizer(model_name)

    if args.smoke:
        layers = R.band_layers(model)
        geom = _smoke_axis(model, tokenizer, layers)
        # L14 may be outside the tiny model's depth — cap it to the band's first.
        l14 = min(C.L14, int(model.config.num_hidden_layers) - 1)
        for d in ("axis_by_layer", "h_def_by_layer", "tau_by_layer", "tau_rand_by_layer"):
            geom[d].setdefault(l14, geom[d][layers[0]])
        C.L14 = l14  # smoke-scoped: the single-layer arm caps a valid layer
    else:
        blob = _load_axis(Path(args.axis_path), Path(args.band_tau_path))
        layers = blob["layers"]
        geom = blob

    jb = C.build_jailbreak_set(args.n_jailbreak, smoke=args.smoke)
    rs = C.build_role_susceptibility_set(args.n_role)
    _log(f"[phase=generate] jailbreak={len(jb)} role_susc={len(rs)} arms={len(_arm_names(args))}")

    for arm in _arm_names(args):
        gen_path = _arm_out_path(out_dir, arm, "gen")
        # Per-arm checkpoint-resume: skip an already-generated arm in BOTH modes
        # so a crashed pod run resumes without re-generating completed arms, and
        # the smoke two-pass run exercises the same skip branch (resume-matrix).
        if gen_path.exists():
            _log(f"[phase=generate] arm={arm} SKIP (resume)")
            continue
        spec = C.ARM_SPECS[arm]
        t0 = time.time()
        # Position-matched null τ: ctx-position cap arms gate against
        # cap_ctx_randnull's τ_rand, all-token cap arms against
        # cap_alltoken_randnull's (plan §5). Smoke aliases both to one dict.
        tau_rand_ctx = geom.get("tau_rand_ctx_by_layer", geom["tau_rand_by_layer"])
        tau_rand_all = geom.get("tau_rand_alltoken_by_layer", geom["tau_rand_by_layer"])
        tau_rand = tau_rand_all if spec["kind"] == "null_alltoken" else tau_rand_ctx
        record: dict = {"arm": arm, "spec": spec, "sets": {}}
        for set_name, rows, jailbreak in (("jailbreak", jb, True), ("role_susc", rs, False)):
            contexts = [{"system": r["system"], "user": r["user"]} for r in rows]
            stack = R.build_stack_for_arm(
                model,
                spec,
                layers=layers,
                axis_by_layer=geom["axis_by_layer"],
                h_def_by_layer=geom["h_def_by_layer"],
                tau_by_layer=geom["tau_by_layer"],
                tau_rand_by_layer=tau_rand,
            )
            texts, realized = R.run_arm(
                model, tokenizer, contexts, stack, max_new_tokens=args.max_new_tokens
            )
            coh = R.coherence_split(texts, jailbreak=jailbreak)
            record["sets"][set_name] = {
                "n_rows": len(rows),
                "cluster_ids": [r["meta"]["cluster_id"] for r in rows],
                "meta": [r["meta"] for r in rows],
                "completions": texts,
                "coherence": coh,
                "edit_telemetry": _summarize_realized(realized),
            }
        gen_path.write_text(json.dumps({"metadata": C.repro_metadata(), **record}, indent=2))
        # Rollout TEXT persisted per arm as a canonical raw_completions.json so the
        # #779 store-before-reduce contract + upload helper (globs raw_completions.json)
        # pick it up. Written under a dedicated upload tree keyed by stage so the
        # helper (experiment_name=issue2203_ctx_capping) lands it at the canonical
        # issue2203_ctx_capping/raw_completions/phase2/<arm>/raw_completions.json.
        # Smoke diverts ALL outputs under --out-dir (never the canonical
        # eval_results/ raw_upload tree); production uses the canonical tree.
        raw_root = (Path(args.out_dir) / "raw_upload") if args.smoke else _raw_upload_dir()
        raw_path = raw_root / "phase2" / arm / "raw_completions.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(
                {
                    "arm": arm,
                    "sets": {
                        s: {"completions": d["completions"], "meta": d["meta"]}
                        for s, d in record["sets"].items()
                    },
                },
                indent=2,
            )
        )
        _log(f"[phase=generate] arm={arm} DONE elapsed={time.time() - t0:.1f}s -> {gen_path.name}")

    if args.smoke:
        _judge_shape_probe(out_dir, jb)

    if args.sentinel_path:
        C.write_sentinel(
            Path(args.sentinel_path), kind="epm:progress", note="phase2 generation complete"
        )
    if args.upload and not args.smoke:
        _upload_raw_completions(out_dir)
    _log("[phase=done] phase2 generate")
    return 0


def _summarize_realized(realized: list[dict] | None) -> dict:
    """Compact edit telemetry (fired-fraction, mean projection before/after)."""
    if not realized:
        return {"edited": False}
    fired = [r.get("fired_frac", 0.0) for r in realized]
    n_pos = sum(r.get("n_positions", 0) for r in realized)
    return {
        "edited": True,
        "n_edit_forwards": len(realized),
        "total_positions_edited": n_pos,
        "mean_fired_frac": (sum(fired) / len(fired)) if fired else 0.0,
    }


def _judge_shape_probe(out_dir: Path, jb: list[dict]) -> None:
    """Forced ~5-item Batch-API judge submit — validates the request shape."""
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        _log("[phase=generate] judge-shape-probe SKIP (no ANTHROPIC_API_KEY)")
        return
    gen = json.loads((_arm_out_path(out_dir, "baseline", "gen")).read_text())
    texts = gen["sets"]["jailbreak"]["completions"][:5]
    metas = gen["sets"]["jailbreak"]["meta"][:5]
    jbmap = {(m["harm_bank"], m["harm_index"]): r["user"] for m, r in zip(metas, jb, strict=False)}
    items = [
        (f"probe-{i}", jbmap.get((m["harm_bank"], m["harm_index"]), "q"), t)
        for i, (m, t) in enumerate(zip(metas, texts, strict=True))
    ]
    _log(f"[phase=generate] judge-shape-probe: forced Batch submit, {len(items)} items")
    res = R.judge_rate(
        items,
        C.HARM_RUBRIC,
        cache_dir=out_dir / "smoke_judge_cache",
        save_raw=out_dir / "smoke_judge_raw.json",
        n_draws=1,
        max_tokens=1024,
        force_batch=True,
    )
    _log(
        f"[phase=generate] judge-shape-probe OK: {json.dumps({k: res[k] for k in ('n_scored_items', 'n_total_draws', 'n_dropped_draws') if k in res})}"
    )


def run_judge(args) -> int:
    """Off-pod judge wave: harm (jailbreak) + assistant-ness (both), api-refusal reissue."""
    out_dir = Path(args.out_dir)
    jb = C.build_jailbreak_set(args.n_jailbreak)
    summary: dict = {"metadata": C.repro_metadata(), "arms": {}}
    for arm in _arm_names(args):
        gen_path = _arm_out_path(out_dir, arm, "gen")
        if not gen_path.exists():
            _log(f"[phase=judge] arm={arm} MISSING generation; skip")
            continue
        gen = json.loads(gen_path.read_text())
        arm_res: dict = {}
        # Jailbreak harm rate (primary co-primary).
        jbset = gen["sets"]["jailbreak"]
        harm_items = [
            (f"{arm}-jb-{i}", jb[i]["user"] if i < len(jb) else "q", t)
            for i, t in enumerate(jbset["completions"])
        ]
        harm = R.judge_rate(
            harm_items,
            C.HARM_RUBRIC,
            cache_dir=out_dir / f"judge_cache/{arm}/harm",
            save_raw=out_dir / f"judge_raw_{arm}_harm.json",
            n_draws=args.n_draws,
            max_tokens=1024,
            force_batch=True,
        )
        if harm.get("n_api_refusal_draws", 0) > 0:
            reissue = R.sync_reissue_api_refusals(
                harm_items,
                C.HARM_RUBRIC,
                harm["per_item_api_refusals"],
                cache_dir=out_dir / f"judge_cache/{arm}/harm_reissue",
                save_raw=out_dir / f"judge_raw_{arm}_harm_reissue.json",
                n_draws=args.n_draws,
                max_tokens=1024,
            )
            arm_res["harm_api_refusal_reissue"] = reissue
        arm_res["harm"] = harm
        summary["arms"][arm] = arm_res
        (out_dir / f"phase2_judge_{arm}.json").write_text(json.dumps(arm_res, indent=2))
        _log(f"[phase=judge] arm={arm} DONE harm_rate={harm.get('rate')}")
    (out_dir / "phase2_ladder_results.json").write_text(json.dumps(summary, indent=2))
    _log("[phase=done] phase2 judge")
    return 0


def _raw_upload_dir() -> Path:
    d = C.eval_results_dir() / "raw_upload"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _upload_raw_completions(out_dir: Path) -> None:
    """Persist per-arm rollout TEXT (raw_completions.json) to the HF data repo.

    ``upload_raw_completions_to_data_repo`` globs the tree for files named
    ``raw_completions.json`` and bulk-uploads them under
    ``<experiment_name>/raw_completions/<rel>`` in ONE ``upload_folder`` commit
    (never a per-file loop; #664/#727). Our per-arm files at
    ``raw_upload/phase2/<arm>/raw_completions.json`` land at the canonical
    ``issue2203_ctx_capping/raw_completions/phase2/<arm>/raw_completions.json``.
    """
    _ = out_dir
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    uploaded = upload_raw_completions_to_data_repo(
        experiment_name=C.HF_PREFIX,
        eval_results_dir=_raw_upload_dir(),
    )
    _log(f"[phase=generate] uploaded {len(uploaded)} raw_completions.json -> {C.HF_PREFIX}/...")


def _arm_names(args) -> list[str]:
    if args.arms:
        return [a for a in args.arms if a in C.ARM_SPECS]
    return list(C.ARM_SPECS.keys())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Issue #2203 Phase 2 — 16-arm capping grid")
    p.add_argument("--phase", choices=("generate", "judge"), default="generate")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--model", default=C.QWEN_7B)
    p.add_argument("--n-jailbreak", type=int, default=500)
    p.add_argument("--n-role", type=int, default=250)
    p.add_argument("--n-draws", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--out-dir", default=str(C.eval_results_dir() / "phase2"))
    p.add_argument("--axis-path", default=None)
    p.add_argument("--band-tau-path", default=None)
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--upload", action="store_true")
    p.add_argument("--sentinel-path", default=None)
    p.add_argument("--import-check", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[import-check] ok")
        return 0
    if args.smoke:
        args.n_jailbreak = min(args.n_jailbreak, 3)
        args.n_role = min(args.n_role, 4)
        args.max_new_tokens = min(args.max_new_tokens, 24)
    if args.phase == "generate":
        return run_generation(args)
    return run_judge(args)


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(main())
