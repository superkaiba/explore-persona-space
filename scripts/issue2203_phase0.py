"""Issue #2203 — Phase 0: assistant-axis extraction + validation (Qwen-2.5-7B-Instruct).

Extracts the per-layer Assistant Axis in-house (plan §4.2) via the
persona-vectors core in ``artifacts/directions``: mean-difference,
response-averaged post-MLP residuals, JUDGE-FILTERED rollouts (Sonnet-4.5 —
role rows kept when role-expression > 50, default rows kept when < 50;
REFUSAL/non-numeric judge returns DROPPED, never coerced; counts reported).
Pool (plan §4.2): ``n_roles × 5 system prompts × 20 shared extraction
questions × 1 draw`` role rollouts (≈5,000) + ``5 default-assistant
conditions × 20 questions × 10 draws`` (≈1,000).

Validation (persisted to ``phase0_axis_validation.json``):
(1) subsample STABILITY — cos(axis_A, axis_B) over a disjoint 50%-ROLE split;
    the §7 HARD abort (cos <= 0.95 mid-layer → gate report + ``SystemExit(3)``,
    production only — the smoke's 3-role axis structurally cannot pass a
    production-n-calibrated bar, so smoke demotes the verdict to a log line);
(2) cos(axis, PC1) per layer vs the paper's > 0.71 middle-layer reference
    (re-tune TRIGGER + covariate, NOT a hard kill);
(3) steering sanity check — ±α·v̂ at the mid layer, judged role expression
    (directional: −v̂ should read MORE in-character than +v̂).

``--select-roles`` judges the 275 role DESCRIPTIONS on willingness-to-comply +
assistant-closeness (plan §4.4 constructs) and persists the sha-pinned
``role_selection.json`` the eval-set builders REQUIRE in production (r1 M11).

``--smoke``: tiny model + tiny pool, out-dir diverted to ``/tmp/issue-2203-smoke``
unless ``--out-dir`` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2203_phase0.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847 shared-VM thread caps must bind BEFORE torch freezes its pool at import.
load_dotenv()

import torch  # noqa: E402

from explore_persona_space.experiments.issue1415 import steering  # noqa: E402
from scripts import issue2203_common as C  # noqa: E402
from scripts import issue2203_runtime as R  # noqa: E402

STABILITY_ABORT_RC = 3  # §7 HARD abort — a designed halt, never a bare rc=1


def _log(msg: str) -> None:
    print(msg, flush=True)


def _resolve_out_dir(args) -> Path:
    """Smoke NEVER writes the canonical eval_results tree by default (r1 M7)."""
    if args.out_dir:
        d = Path(args.out_dir)
    elif args.smoke:
        d = Path("/tmp/issue-2203-smoke")
    else:
        d = C.eval_results_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _chunked_rollout(model, tokenizer, contexts, n, max_new_tokens, *, chunk: int = 256):
    """On-policy rollouts, chunked (r1 Minor 22 — no monolithic 1k-ctx batches)."""
    out: list[list[str]] = []
    n_chunks = (len(contexts) + chunk - 1) // chunk
    for k in range(n_chunks):
        part = contexts[k * chunk : (k + 1) * chunk]
        out.extend(
            steering.generate_batch(
                model,
                tokenizer,
                part,
                n=n,
                hook=None,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                seed_base=42,
            )
        )
        _log(f"[phase=phase0] rollout chunk {k + 1}/{n_chunks} ({len(part)} ctx)")
    return out


def _response_means_rows(model, tokenizer, rows: list[dict], layers):
    """Response-averaged activations for flat rows [{'ctx', 'text', ...}]."""
    from explore_persona_space.artifacts.directions import (
        ContrastiveCompletion,
        batched_response_means,
        encode_rows,
    )

    rows_meta = [
        ContrastiveCompletion(
            arm="exhibit",
            pair_index=i,
            system_prompt=r["ctx"].get("system") or "",
            question=r["ctx"]["user"],
            response=r["text"],
        )
        for i, r in enumerate(rows)
    ]
    encoded, counts = encode_rows(tokenizer, rows_meta)
    keep_idx = [i for i, e in enumerate(encoded) if e is not None]
    valid = [encoded[i] for i in keep_idx]
    if not valid:
        raise ValueError("no valid rows encoded for response-mean capture")
    means = batched_response_means(model, valid, layers)  # list of (L,H)
    return torch.stack(means), keep_idx, counts  # (n_valid, L, H)


def _judge_filter(rows: list[dict], *, out_dir: Path, smoke: bool) -> tuple[list[dict], dict]:
    """Role-expression judge filter (plan §4.2 / persona-vectors-recipe).

    Keep role rows with score > 50 (fully role-playing) and default rows with
    score < 50 (NOT in-character); ``None`` scores (REFUSAL / non-numeric /
    out-of-range) are DROPPED, never coerced — counts reported. Production
    REQUIRES the judge; a keyless SMOKE skips with a recorded marker (smoke
    blind spot, enumerated in the implementer report).
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        if not smoke:
            raise RuntimeError("phase0 judge-filter requires ANTHROPIC_API_KEY (plan §4.2)")
        _log("[phase=phase0] judge-filter SKIP (smoke, no ANTHROPIC_API_KEY)")
        return rows, {"skipped_no_key": True, "n_in": len(rows), "n_kept": len(rows)}
    items = []
    for i, r in enumerate(rows):
        persona = r["role"] if r["kind"] == "role" else "the default AI assistant"
        q = f"(intended persona: {persona}) {r['ctx']['user']}"
        items.append((f"row-{i}", q, r["text"]))
    if not smoke and len(items) >= 5000:
        R.judge_pilot_gate(
            items,
            C.ROLE_EXPRESSION_RUBRIC,
            cache_dir=out_dir / "judge_cache/phase0_filter_pilot",
            save_raw=out_dir / "judge_raw_phase0_filter_pilot.json",
            report_path=out_dir / "phase0_filter_pilot_report.json",
            n_draws=1,
            n_pilot_items=150,
        )
    res = R.judge_rate(
        items,
        C.ROLE_EXPRESSION_RUBRIC,
        cache_dir=out_dir / "judge_cache/phase0_filter",
        save_raw=out_dir / "judge_raw_phase0_filter.json",
        n_draws=1,
        max_tokens=1024,
        force_batch=True,
    )
    kept, n_drop_judge, n_drop_filter = [], 0, 0
    for i, r in enumerate(rows):
        score = res["mean_scores"].get(f"row-{i}")
        if score is None:
            n_drop_judge += 1
            continue
        keep = (score > 50) if r["kind"] == "role" else (score < 50)
        if keep:
            kept.append(r)
        else:
            n_drop_filter += 1
    stats = {
        "skipped_no_key": False,
        "n_in": len(rows),
        "n_kept": len(kept),
        "n_dropped_judge_return": n_drop_judge,
        "n_dropped_by_filter": n_drop_filter,
        "n_api_refusal_draws": res["n_api_refusal_draws"],
        "rule": "role rows kept iff score>50; default rows kept iff score<50 (recipe)",
    }
    _log(f"[phase=phase0] judge-filter kept {len(kept)}/{len(rows)} rows ({stats})")
    return kept, stats


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
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        pc1 = Vh[0]
        out.append(abs(float(torch.nn.functional.cosine_similarity(axis[j], pc1, dim=0))))
    return out


def _steering_sanity(model, tokenizer, axis, layers, out_dir: Path, smoke: bool) -> dict:
    """Plan §4.2 validation (3): judged directional ±α·v̂ steering check."""
    mid = len(layers) // 2
    role_list = C.load_role_list()
    names = sorted(role_list)[: (2 if smoke else 4)]
    contexts = [
        {"system": C.role_system_prompts(r, k=1)[0], "user": C.INTROSPECTIVE_QUESTIONS[0]}
        for r in names
    ]
    res = R.steering_sanity_check(
        model,
        tokenizer,
        axis[mid],
        mid,
        contexts,
        max_new_tokens=(16 if smoke else 128),
    )
    record: dict = {"layer": res["layer"], "alpha": res["alpha"], "n_contexts": len(contexts)}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        record["judged"] = False
        record["note"] = "steering completions generated; judging skipped (no key)"
        return record
    items = []
    for sign in ("plus", "minus"):
        for i, (ctx, text) in enumerate(zip(contexts, res[sign], strict=True)):
            items.append((f"{sign}-{i}", f"(intended persona: {names[i]}) {ctx['user']}", text))
    jr = R.judge_rate(
        items,
        C.ROLE_EXPRESSION_RUBRIC,
        cache_dir=out_dir / "judge_cache/phase0_steer",
        save_raw=out_dir / "judge_raw_phase0_steer.json",
        n_draws=1,
        max_tokens=1024,
        force_batch=True,
    )
    plus = [v for k, v in jr["mean_scores"].items() if k.startswith("plus-") and v is not None]
    minus = [v for k, v in jr["mean_scores"].items() if k.startswith("minus-") and v is not None]
    record.update(
        {
            "judged": True,
            "mean_role_expression_plus_axis": (sum(plus) / len(plus)) if plus else None,
            "mean_role_expression_minus_axis": (sum(minus) / len(minus)) if minus else None,
            # axis points TOWARD the assistant, so -v̂ (role-ward) should read
            # MORE in-character than +v̂.
            "directional_ok": (
                bool(sum(minus) / len(minus) > sum(plus) / len(plus)) if plus and minus else None
            ),
        }
    )
    return record


def run_select_roles(args) -> int:
    """Judge the role DESCRIPTIONS → sha-pinned role_selection.json (r1 M11)."""
    out_dir = _resolve_out_dir(args)
    assert os.environ.get("ANTHROPIC_API_KEY"), "--select-roles requires ANTHROPIC_API_KEY"
    role_list = C.load_role_list()
    names = sorted(role_list)[: (6 if args.smoke else len(role_list))]
    _log(f"[phase=phase0] select-roles over {len(names)} role descriptions")
    scores: dict[str, dict[str, float | None]] = {}
    for kind, rubric in (("willing", C.WILLINGNESS_RUBRIC), ("close", C.ASSISTANT_CLOSE_RUBRIC)):
        items = [(f"{kind}-{n}", n, role_list[n]) for n in names]
        res = R.judge_rate(
            items,
            rubric,
            cache_dir=out_dir / f"judge_cache/roles_{kind}",
            save_raw=out_dir / f"judge_raw_roles_{kind}.json",
            n_draws=1,
            max_tokens=1024,
            force_batch=True,
        )
        for n in names:
            scores.setdefault(n, {})[kind] = res["mean_scores"].get(f"{kind}-{n}")
    n_willing = 4 if args.smoke else 120
    n_close = 4 if args.smoke else 50
    ranked_w = sorted(
        (n for n in names if scores[n]["willing"] is not None),
        key=lambda n: (-scores[n]["willing"], n),
    )
    ranked_c = sorted(
        (n for n in names if scores[n]["close"] is not None),
        key=lambda n: (-scores[n]["close"], n),
    )
    sel = {
        "metadata": C.repro_metadata(),
        "willing": ranked_w[:n_willing],
        "assistant_close": ranked_c[:n_close],
        "n_judged": len(names),
        "n_dropped": sum(1 for n in names if scores[n]["willing"] is None),
        "scores": scores,
    }
    sel["selection_sha"] = C._sha256_of_obj(
        {"willing": sel["willing"], "assistant_close": sel["assistant_close"]}
    )
    path = C.role_selection_path(smoke=args.smoke)
    path.write_text(json.dumps(sel, indent=2))
    _log(
        f"[phase=phase0] role selection -> {path} "
        f"(willing={len(sel['willing'])} close={len(sel['assistant_close'])})"
    )
    _log("[phase=done] phase0 select-roles")
    return 0


def run(args) -> int:
    out_dir = _resolve_out_dir(args)
    model_name = C.TINY_MODEL if args.smoke else args.model
    _log(f"[phase=phase0] model={model_name} smoke={args.smoke}")
    model, tokenizer = R.load_model_and_tokenizer(model_name)
    layers = list(range(int(model.config.num_hidden_layers)))

    role_list = C.load_role_list()
    all_names = sorted(role_list)
    rng = random.Random(42)
    rng.shuffle(all_names)
    names = all_names[: (3 if args.smoke else args.n_roles)]
    n_sysprompts = 2 if args.smoke else args.n_sysprompts
    n_q = 2 if args.smoke else args.n_questions
    default_draws = 1 if args.smoke else args.n_default_draws
    max_new = 16 if args.smoke else args.max_new_tokens
    questions = C.extraction_questions(n_q, seed=42)
    default_conditions = C.default_assistant_conditions()
    if args.smoke:
        default_conditions = default_conditions[:2]

    role_ctx = [
        {"system": p, "user": q, "_role": role}
        for role in names
        for p in C.role_system_prompts(role, k=n_sysprompts)
        for q in questions
    ]
    default_ctx = [{"system": cond, "user": q} for cond in default_conditions for q in questions]
    _log(
        f"[phase=phase0] pool: {len(role_ctx)} role ctx x 1 draw + "
        f"{len(default_ctx)} default ctx x {default_draws} draws"
    )
    role_comps = _chunked_rollout(model, tokenizer, role_ctx, 1, max_new)
    default_comps = _chunked_rollout(model, tokenizer, default_ctx, default_draws, max_new)

    # Persist the extraction rollout TEXT BEFORE reducing (#779; plan §10).
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

    rows: list[dict] = []
    for ctx, comps in zip(role_ctx, role_comps, strict=True):
        for text in comps:
            rows.append({"ctx": ctx, "text": text, "kind": "role", "role": ctx["_role"]})
    for ctx, comps in zip(default_ctx, default_comps, strict=True):
        for text in comps:
            rows.append({"ctx": ctx, "text": text, "kind": "default", "role": None})

    kept, filter_stats = _judge_filter(rows, out_dir=out_dir, smoke=args.smoke)
    role_rows = [r for r in kept if r["kind"] == "role"]
    def_rows = [r for r in kept if r["kind"] == "default"]
    assert role_rows and def_rows, (len(role_rows), len(def_rows))

    role_means, role_keep_idx, rc = _response_means_rows(model, tokenizer, role_rows, layers)
    def_means, _, dc = _response_means_rows(model, tokenizer, def_rows, layers)
    axis = def_means.mean(dim=0) - role_means.mean(dim=0)  # (L, H): default - role
    h_def = def_means.mean(dim=0)  # (L, H) default-assistant mean state

    # (1) subsample STABILITY on a disjoint 50%-ROLE split (plan §4.2).
    kept_roles = [role_rows[i]["role"] for i in role_keep_idx]
    uniq = sorted(set(kept_roles))
    half_names = set(uniq[: len(uniq) // 2])
    idx_a = [i for i, r in enumerate(kept_roles) if r in half_names]
    idx_b = [i for i, r in enumerate(kept_roles) if r not in half_names]
    nd_half = def_means.shape[0] // 2
    stability: list[float] = []
    if idx_a and idx_b and nd_half >= 1:
        axis_a = def_means[:nd_half].mean(0) - role_means[idx_a].mean(0)
        axis_b = def_means[nd_half:].mean(0) - role_means[idx_b].mean(0)
        stability = _cos_per_layer(axis_a, axis_b)
    mid = axis.shape[0] // 2
    stability_mid = stability[mid] if stability else None

    # (2) cos(axis, PC1) per layer over the pooled response means.
    pool = torch.cat([role_means, def_means], dim=0)
    pc1_cos = _pc1_cos(pool, axis)

    # (3) steering sanity check (directional, small N; plan §4.2).
    steer = _steering_sanity(model, tokenizer, axis, layers, out_dir, args.smoke)

    validation = {
        "metadata": C.repro_metadata(),
        "n_layers": len(layers),
        "n_role_means": int(role_means.shape[0]),
        "n_default_means": int(def_means.shape[0]),
        "judge_filter": filter_stats,
        "encode_counts": {"role": rc, "default": dc},
        "stability_cos_per_layer": stability,
        "stability_cos_mid_layer": stability_mid,
        "stability_gate_pass": (stability_mid is not None and stability_mid > 0.95),
        "stability_split": {"n_roles_half_a": len(half_names), "n_roles": len(uniq)},
        "pc1_cos_per_layer": pc1_cos,
        "pc1_cos_mid_layer": pc1_cos[mid] if pc1_cos else None,
        "pc1_reference_0p71_note": "re-tune trigger + covariate, NOT a hard kill (§7)",
        "steering_sanity": steer,
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

    if args.upload and not args.smoke:
        axis_url = C.upload_axis_to_hf(axis_path)  # cross-phase input → HF (§10, #521)
        uploaded = C.upload_raw_tree(out_dir / "raw_upload")
        _log(f"[phase=phase0] uploaded axis -> {axis_url}; {len(uploaded)} raw_completions.json")

    if not validation["stability_gate_pass"]:
        if args.smoke:
            # Production-n-calibrated gate: a 3-role smoke axis cannot pass a
            # 0.95 bar — verdict demoted to informational under smoke
            # (gotchas.md smoke/production GATE-CALIBRATION rule).
            _log("[phase=phase0] stability gate INFORMATIONAL under smoke (tiny-n axis)")
        else:
            gate_report = {
                "gate": "stability_cos_mid_layer > 0.95 (plan §7 HARD abort)",
                "stability_cos_mid_layer": stability_mid,
                "verdict": "ABORT",
                "note": "axis not reproducible across disjoint role halves — no downstream spend",
            }
            (out_dir / "phase0_stability_gate_report.json").write_text(
                json.dumps(gate_report, indent=2)
            )
            _log("[phase=phase0] STABILITY GATE FAIL — ABORT (rc=3, §7 kill criterion 1)")
            raise SystemExit(STABILITY_ABORT_RC)
    _log("[phase=done] phase0")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Issue #2203 Phase 0 — axis extraction + validation")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--select-roles", action="store_true", help="judge + pin the role selection")
    p.add_argument("--model", default=C.QWEN_7B)
    p.add_argument("--n-roles", type=int, default=50)
    p.add_argument("--n-sysprompts", type=int, default=5)
    p.add_argument("--n-questions", type=int, default=20)
    p.add_argument("--n-default-draws", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out-dir", default=None)
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
    if args.select_roles:
        return run_select_roles(args)
    return run(args)


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(main())
