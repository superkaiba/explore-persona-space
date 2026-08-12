"""Issue #2203 — Part D: context-native + prefix-native axis extraction (plan §4.5).

Folds in the superseded ``ctx-native-axis-cap`` scope: extract the assistant axis
AT the position it will be edited (context-end / prefix-end) rather than from the
model's own response, and test whether native extraction recovers a localized
effect at least as strong as the response-derived axis.

REUSES the phase-0 KEPT SET from the DURABLE HF sources (never the non-durable
local ``judge_cache/phase0_filter`` dir, plan §4.5): the rollout pool
(``raw_completions/extraction/raw_completions.json``) + the judge-filter verdicts
(``judge_raw/judge_raw_phase0_filter.json``). Re-applies the verdicts with the
production reduce (``judge_result_from_save_raw``: drop-never-coerce), re-runs the
encode filter, and ASSERTS the kept-pool fingerprint (``n_role==4975``,
``n_default==1000``) — a mismatch fails loud. NO new generation, NO new judging.

New capture: PROMPT-ONLY forwards (no completion) over the kept rows' contexts,
capturing the hidden state at the context-end + prefix-end positions per layer,
for role and default rows. Two poolings per axis (``pooled_rows`` /
``mean_of_role_means``) with the inter-pooling cosine per layer. Validation
battery: split-half stability, cos vs the response-derived axis, cos vs role-PC1,
small-N steering sanity.

Persists ``v_context.pt`` / ``v_prefix.pt`` / ``h_def_ctx.pt`` /
``h_def_prefix.pt`` + ``phase0_native_validation.json`` (carrying the
``native_geometry`` block — the position-matched UNIT-space τ + τ_rand recomputed
on the NATIVE axis) to HF ``issue2203_ctx_capping/analysis_tensors/`` (reused by
the phase-2 native arms + #2223).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2203_phase0_native.py").exists(), root
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
from scripts import issue2203_phase0 as P0  # noqa: E402
from scripts import issue2203_runtime as R  # noqa: E402

NULL_SEED = 1234


def _log(msg: str) -> None:
    print(msg, flush=True)


def _resolve_out_dir(args) -> Path:
    if args.out_dir:
        d = Path(args.out_dir)
    elif args.smoke:
        d = Path("/tmp/issue-2203-smoke")
    else:
        d = C.eval_results_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_phase0_pool(out_dir: Path, smoke: bool) -> dict:
    """The Phase-0 extraction rollout pool (from the local smoke run, else HF)."""
    raw = out_dir / "raw_upload" / "extraction" / "raw_completions.json"
    if not raw.exists():
        if smoke:
            raise FileNotFoundError(
                f"{raw} absent — run `issue2203_phase0.py --smoke --out-dir {out_dir}` first"
            )
        _log(f"[phase=phase0native] rollout pool not local; staging from HF -> {raw}")
        C.stage_extraction_rollouts_from_hf(raw)
    return json.loads(raw.read_text())


def _load_verdicts(out_dir: Path, rows: list[dict], smoke: bool) -> tuple[Path, bool]:
    """The phase-0 judge-filter verdicts file (durable HF source, plan §4.5).

    Returns ``(path, synthesized)`` — ``synthesized`` is True ONLY when the
    smoke-synth branch below actually ran this invocation (r1 minor: the flag
    must come from the branch taken, never from file existence).
    """
    local = out_dir / "judge_raw_phase0_filter.json"
    if local.exists():
        return local, False
    if not smoke:
        _log(f"[phase=phase0native] verdicts not local; staging from HF -> {local}")
        C.stage_phase0_filter_verdicts_from_hf(local)
        return local, False
    # SMOKE BLIND SPOT (r1 M1; smoke-blind-spots rule — INPUT SUBSTITUTION):
    # this branch SYNTHESIZES the Part-D kept-set verdicts (role rows score 80,
    # default rows score 20), so under smoke the production HF-verdicts staging
    # → judge_result_from_save_raw → encode-filter → 4975/1000 fingerprint path
    # never executes on REAL verdicts. The real-verdicts reduce is validated by
    # a standalone CPU-only probe (see the round's marker `## Smoke run`).
    # custom_id = row-{i}__00000__00 (phase-0 n_draws=1 shape).
    all_scores = {
        f"row-{i}__00000__00": {"score": (80 if r["kind"] == "role" else 20)}
        for i, r in enumerate(rows)
    }
    local.write_text(json.dumps({"all_scores": all_scores, "smoke_synth_verdicts": True}, indent=2))
    _log(f"[phase=phase0native] SMOKE synth verdicts -> {local.name}")
    return local, True


def _rebuild_rows(pool: dict) -> tuple[list[dict], list[dict]]:
    """Reconstruct the phase-0 flat rows (role first, then default) + unique ctx list.

    Row order + ``ctx_uid`` mirror ``phase0.run`` EXACTLY so the saved verdicts'
    ``row-{i}`` ids re-apply. Returns ``(rows, unique_contexts)`` where
    ``rows[i]["ctx_uid"]`` indexes ``unique_contexts``.
    """
    role_ctx = pool["role_contexts"]
    role_comps = pool["role_completions"]
    default_ctx = pool["default_contexts"]
    default_comps = pool["default_completions"]
    unique = list(role_ctx) + list(default_ctx)
    rows: list[dict] = []
    for ci, (ctx, comps) in enumerate(zip(role_ctx, role_comps, strict=True)):
        for text in comps:
            rows.append(
                {"ctx": ctx, "ctx_uid": ci, "text": text, "kind": "role", "role": ctx.get("_role")}
            )
    off = len(role_ctx)
    for ci, (ctx, comps) in enumerate(zip(default_ctx, default_comps, strict=True)):
        for text in comps:
            rows.append(
                {"ctx": ctx, "ctx_uid": off + ci, "text": text, "kind": "default", "role": None}
            )
    return rows, unique


def _reapply_verdicts(rows: list[dict], verdicts_path: Path) -> list[dict]:
    """Re-apply the phase-0 keep rule from saved verdicts (drop-never-coerce, §4.5)."""
    from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

    items = []
    for i, r in enumerate(rows):
        persona = r["role"] if r["kind"] == "role" else "the default AI assistant"
        items.append((f"row-{i}", f"(intended persona: {persona}) {r['ctx']['user']}", r["text"]))
    jr = judge_result_from_save_raw(verdicts_path, items)
    kept = []
    for i, r in enumerate(rows):
        score = jr.scores.get(f"row-{i}")
        if score is None:
            continue
        keep = (score > 50) if r["kind"] == "role" else (score < 50)
        if keep:
            kept.append(r)
    return kept


def _encode_filter(tokenizer, kept: list[dict], kind: str) -> list[dict]:
    """Re-run phase-0's encode filter on one class; return kept (post-encode) rows.

    Mirrors ``phase0._response_means_rows``'s ``encode_rows`` + keep_idx exactly
    so the surviving count reproduces the phase-0 ``n_{role,default}_means``
    fingerprint (the reuse identity check, §4.5).
    """
    from explore_persona_space.artifacts.directions import ContrastiveCompletion, encode_rows

    cls = [r for r in kept if r["kind"] == kind]
    meta = [
        ContrastiveCompletion(
            arm="exhibit",
            pair_index=i,
            system_prompt=r["ctx"].get("system") or "",
            question=r["ctx"]["user"],
            response=r["text"],
        )
        for i, r in enumerate(cls)
    ]
    encoded, _counts = encode_rows(tokenizer, meta)
    return [cls[i] for i, e in enumerate(encoded) if e is not None]


def _capture_prompt_states(model, tokenizer, unique_contexts, layers, *, batch_size=8) -> dict:
    """Prompt-only ctx-end + prefix-end hidden states per UNIQUE context, per layer.

    Right-pad batched teacher-forced-style forward over the rendered context ids
    (no completion; token-ID concatenation is a no-op here). ctx-end = last real
    prompt token (``ctx_len-1``); prefix-end = ``prefix_end-1`` (the hook's edit
    position) when the render has a clean 3-``<|im_start|>`` boundary, else
    ``None`` (a bare no-system context has no prefix boundary — excluded from the
    prefix pool). Returns ``{uid: {"ctx_end": (L,H), "prefix_end": (L,H)|None}}``.
    """
    from explore_persona_space.analysis.extraction import extract_layer_activations

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    per_ctx_ids = [steering.context_token_ids(tokenizer, c) for c in unique_contexts]
    prefix_ends = [R._prefix_end_or_none(tokenizer, ids) for ids in per_ctx_ids]
    out: dict[int, dict] = {}
    n = len(unique_contexts)
    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        rows = [per_ctx_ids[i] for i in idxs]
        T = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), T), dtype=torch.long)
        for b, r in enumerate(rows):
            input_ids[b, : len(r)] = torch.tensor(r, dtype=torch.long)  # RIGHT pad
            mask[b, : len(r)] = 1
        captured = extract_layer_activations(
            model, input_ids.to(device), layers, attention_mask=mask.to(device)
        )
        for b, ci in enumerate(idxs):
            clen = len(per_ctx_ids[ci])
            pe = prefix_ends[ci]
            ctx_end = torch.stack([captured[li][b, clen - 1].float() for li in layers]).cpu()
            pre = (
                torch.stack([captured[li][b, pe - 1].float() for li in layers]).cpu()
                if pe is not None
                else None
            )
            out[ci] = {"ctx_end": ctx_end, "prefix_end": pre}
        del captured
        _log(f"[phase=phase0native] capture ctx {min(start + batch_size, n)}/{n}")
    return out


def _stack_layer_dict(vec, layers):
    """{str(li): (H,) tensor} from a (L, H) tensor."""
    return {str(li): vec[j].contiguous() for j, li in enumerate(layers)}


def _mean_state(rows, states, pos, layers):
    """Row-pooled mean of ``states[row.ctx_uid][pos]`` over rows with that position."""
    acc = []
    for r in rows:
        s = states[r["ctx_uid"]][pos]
        if s is not None:
            acc.append(s)
    if not acc:
        return None
    return torch.stack(acc).mean(dim=0)  # (L, H)


def _mean_of_role_means(rows, states, pos, layers):
    """Per-role mean then mean over roles (role rows only)."""
    by_role: dict[str, list] = {}
    for r in rows:
        s = states[r["ctx_uid"]][pos]
        if s is None:
            continue
        by_role.setdefault(r["role"], []).append(s)
    if not by_role:
        return None
    role_means = [torch.stack(v).mean(dim=0) for v in by_role.values()]
    return torch.stack(role_means).mean(dim=0)  # (L, H)


def _tau_pools_native(rows, states, pos, axis_LH, layers) -> tuple[dict, dict]:
    """Native position-matched UNIT-space τ + τ_rand (25th pct over kept rows).

    ``axis_LH`` is the native axis ``(L, H)`` (raw); it is normalized per layer.
    τ = q25 of ⟨state, v̂_native⟩ over kept rows expanded (each row projects its
    context's state); τ_rand = q25 of ⟨state, v̂_rand⟩ with a seeded norm-matched
    random direction per layer (default null_seed, matching build_stack_for_arm).
    Returns (tau_by_layer, tau_rand_by_layer), both str-keyed.
    """
    vhat = torch.stack([axis_LH[j] / (axis_LH[j].norm() + 1e-12) for j in range(len(layers))])
    rand = torch.stack(
        [R._seeded_random_axis(axis_LH[j], NULL_SEED + li) for j, li in enumerate(layers)]
    )
    vhat_rand = torch.stack([rand[j] / (rand[j].norm() + 1e-12) for j in range(len(layers))])
    proj = {li: [] for li in layers}
    proj_r = {li: [] for li in layers}
    for r in rows:
        s = states[r["ctx_uid"]][pos]  # (L, H)
        if s is None:
            continue
        for j, li in enumerate(layers):
            proj[li].append(float(s[j] @ vhat[j]))
            proj_r[li].append(float(s[j] @ vhat_rand[j]))
    tau = {
        str(li): float(torch.quantile(torch.tensor(proj[li]), 0.25)) for li in layers if proj[li]
    }
    tau_rand = {
        str(li): float(torch.quantile(torch.tensor(proj_r[li]), 0.25))
        for li in layers
        if proj_r[li]
    }
    return tau, tau_rand


def run(args) -> int:
    out_dir = _resolve_out_dir(args)
    model_name = C.TINY_MODEL if args.smoke else args.model
    _log(f"[phase=phase0native] model={model_name} smoke={args.smoke}")
    model, tokenizer = R.load_model_and_tokenizer(model_name)
    layers = list(range(int(model.config.num_hidden_layers)))

    pool = _load_phase0_pool(out_dir, args.smoke)
    rows, unique_contexts = _rebuild_rows(pool)
    verdicts_path, verdicts_synth = _load_verdicts(out_dir, rows, args.smoke)
    kept = _reapply_verdicts(rows, verdicts_path)
    role_rows = _encode_filter(tokenizer, kept, "role")
    def_rows = _encode_filter(tokenizer, kept, "default")
    n_role, n_default = len(role_rows), len(def_rows)
    _log(f"[phase=phase0native] kept-pool fingerprint: n_role={n_role} n_default={n_default}")
    if not args.smoke:
        assert n_role == args.expect_n_role, (n_role, args.expect_n_role)
        assert n_default == args.expect_n_default, (n_default, args.expect_n_default)
    assert role_rows and def_rows, (n_role, n_default)

    # Capture prompt-only states over the UNIQUE contexts the kept rows reference.
    used_uids = sorted({r["ctx_uid"] for r in role_rows + def_rows})
    sub_contexts = [unique_contexts[u] for u in used_uids]
    remap = {u: j for j, u in enumerate(used_uids)}
    for r in role_rows + def_rows:
        r["ctx_uid"] = remap[r["ctx_uid"]]
    states = _capture_prompt_states(
        model, tokenizer, sub_contexts, layers, batch_size=args.batch_size
    )

    # ── native axes (pooled_rows drives the arms; mean_of_role_means reported) ──
    role_ctxend = _mean_state(role_rows, states, "ctx_end", layers)
    def_ctxend = _mean_state(def_rows, states, "ctx_end", layers)
    role_prefix = _mean_state(role_rows, states, "prefix_end", layers)
    def_prefix = _mean_state(def_rows, states, "prefix_end", layers)
    assert role_ctxend is not None and def_ctxend is not None, "no ctx-end states captured"
    assert role_prefix is not None and def_prefix is not None, "no prefix-end states captured"

    v_context = def_ctxend - role_ctxend  # (L, H) points toward the default pole
    h_def_ctx = def_ctxend
    v_prefix = def_prefix - role_prefix
    h_def_prefix = def_prefix

    # mean_of_role_means axis (role side per-role averaged) + inter-pooling cosine.
    role_ctxend_ror = _mean_of_role_means(role_rows, states, "ctx_end", layers)
    role_prefix_ror = _mean_of_role_means(role_rows, states, "prefix_end", layers)
    v_context_ror = def_ctxend - role_ctxend_ror
    v_prefix_ror = def_prefix - role_prefix_ror
    inter_pooling_cos_ctx = P0._cos_per_layer(v_context, v_context_ror)
    inter_pooling_cos_prefix = P0._cos_per_layer(v_prefix, v_prefix_ror)

    # ── native position-matched τ (unit space, on the NATIVE axis) ──
    tau_ctx, tau_rand_ctx = _tau_pools_native(
        role_rows + def_rows, states, "ctx_end", v_context, layers
    )
    tau_prefix, tau_rand_prefix = _tau_pools_native(
        [r for r in role_rows + def_rows if states[r["ctx_uid"]]["prefix_end"] is not None],
        states,
        "prefix_end",
        v_prefix,
        layers,
    )
    native_geometry = {
        "context_native": {
            "tau_by_position": {"context-end": tau_ctx},
            "tau_rand_by_position": {"context-end": tau_rand_ctx},
            "null_seed": NULL_SEED,
        },
        "prefix_native": {
            "tau_by_position": {"prefix-end": tau_prefix},
            "tau_rand_by_position": {"prefix-end": tau_rand_prefix},
            "null_seed": NULL_SEED,
        },
    }

    # ── validation battery (mirrors phase-0) ──
    validation = _validate(
        model,
        tokenizer,
        rows_by_uid=(role_rows, def_rows),
        states=states,
        axes={"v_context": v_context, "v_prefix": v_prefix},
        layers=layers,
        out_dir=out_dir,
        smoke=args.smoke,
    )
    validation.update(
        {
            "metadata": C.repro_metadata(),
            "n_role_means": n_role,
            "n_default_means": n_default,
            "n_unique_contexts_captured": len(sub_contexts),
            # r1 minor: the flag reflects the branch ACTUALLY TAKEN in
            # _load_verdicts — not file existence (which reads True for a real
            # pre-existing verdicts file under smoke).
            "verdicts_synth_smoke": verdicts_synth,
            "inter_pooling_cos_context_per_layer": inter_pooling_cos_ctx,
            "inter_pooling_cos_prefix_per_layer": inter_pooling_cos_prefix,
            "native_geometry": native_geometry,
        }
    )

    # ── persist (4 .pt + validation JSON) + upload folder ──
    suffix = "_smoke" if args.smoke else ""
    native_dir = out_dir / f"native_upload{suffix}"
    native_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "v_context": (v_context, h_def_ctx),
        "v_prefix": (v_prefix, h_def_prefix),
        "h_def_ctx": (v_context, h_def_ctx),
        "h_def_prefix": (v_prefix, h_def_prefix),
    }
    for name, (axis_LH, hdef_LH) in tensors.items():
        blob = {
            "axis_by_layer": _stack_layer_dict(axis_LH, layers),
            "h_def_by_layer": _stack_layer_dict(hdef_LH, layers),
            "layers": layers,
        }
        torch.save(blob, native_dir / f"{name}.pt")  # the HF-upload copy (production)
        if args.smoke:
            # Smoke has no HF round-trip: phase2 `_load_native_geometry` reads the
            # FLAT ``out_dir/<name>_smoke.pt`` (production stages ``<name>.pt`` from
            # HF instead), so write that copy here or the phase-2 native smoke
            # can't find the tensors.
            torch.save(blob, out_dir / f"{name}{suffix}.pt")
    val_path = out_dir / f"phase0_native_validation{suffix}.json"
    val_path.write_text(json.dumps(validation, indent=2))
    # co-locate the validation JSON in the upload dir (phase-2 stages it from HF).
    (native_dir / "phase0_native_validation.json").write_text(json.dumps(validation, indent=2))
    _log(
        f"[phase=phase0native] axes ctx{tuple(v_context.shape)} prefix{tuple(v_prefix.shape)} "
        f"-> {val_path.name}"
    )

    if args.sentinel_path:
        C.write_sentinel(
            Path(args.sentinel_path), kind="epm:progress", note="phase0_native complete"
        )
    if args.upload and not args.smoke:
        url = C.upload_native_tensors_dir(native_dir)
        _log(f"[phase=phase0native] uploaded native axes -> {url}")
    _log("[phase=done] phase0_native")
    return 0


def _validate(model, tokenizer, *, rows_by_uid, states, axes, layers, out_dir, smoke) -> dict:
    """Native-axis validation: split-half stability, cos vs response axis, PC1, steering."""
    role_rows, def_rows = rows_by_uid
    v_context = axes["v_context"]
    # (1) split-half stability on a disjoint 50%-role split (context-native).
    uniq = sorted({r["role"] for r in role_rows})
    half = set(uniq[: len(uniq) // 2])
    ra = [r for r in role_rows if r["role"] in half]
    rb = [r for r in role_rows if r["role"] not in half]
    nd = len(def_rows) // 2
    stability = []
    if ra and rb and nd >= 1:
        def_a = _mean_state(def_rows[:nd], states, "ctx_end", layers)
        def_b = _mean_state(def_rows[nd:], states, "ctx_end", layers)
        axis_a = def_a - _mean_state(ra, states, "ctx_end", layers)
        axis_b = def_b - _mean_state(rb, states, "ctx_end", layers)
        stability = P0._cos_per_layer(axis_a, axis_b)
    mid = v_context.shape[0] // 2

    # (2) cos(native, response-derived axis) per layer. Only STAGING failures
    # (HF transport / filesystem) may soft-skip this plan-named covariate (r1
    # minor); a malformed blob / missing key still raises loud.
    from huggingface_hub.errors import HfHubHTTPError

    resp_cos = None
    try:
        axis_path = out_dir / ("phase0_axis_smoke.pt" if smoke else "phase0_axis.pt")
        if not axis_path.exists() and not smoke:
            C.stage_axis_from_hf(axis_path)
        if axis_path.exists():
            blob = torch.load(axis_path, map_location="cpu", weights_only=False)
            resp = torch.stack([blob["axis_by_layer"][str(li)].float() for li in layers])
            resp_cos = P0._cos_per_layer(v_context, resp)
    except (OSError, HfHubHTTPError) as exc:  # staging-error classes ONLY
        _log(f"[phase=phase0native] response-axis cosine SKIP ({type(exc).__name__}: {exc})")

    # (3) cos(native, role-PC1) per layer over the pooled ctx-end states.
    ctx_states = [states[r["ctx_uid"]]["ctx_end"] for r in role_rows + def_rows]
    pool = torch.stack(ctx_states)  # (N, L, H)
    pc1_cos = P0._pc1_cos(pool, v_context)

    # (4) small-N steering sanity (directional ±α·v̂), reusing the phase-0 helper.
    steer = P0._steering_sanity(
        model, tokenizer, v_context, layers, out_dir, smoke, cache_tag="phase0native_steer"
    )
    return {
        "stability_cos_per_layer": stability,
        "stability_cos_mid_layer": (stability[mid] if stability else None),
        "cos_vs_response_axis_per_layer": resp_cos,
        "pc1_cos_per_layer": pc1_cos,
        "pc1_cos_mid_layer": (pc1_cos[mid] if pc1_cos else None),
        "steering_sanity": steer,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Issue #2203 Part D — native axis extraction")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--model", default=C.QWEN_7B)
    p.add_argument("--expect-n-role", type=int, default=4975, help="kept-pool fingerprint (§4.5)")
    p.add_argument("--expect-n-default", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out-dir", default=None)
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
    return run(args)


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(main())
