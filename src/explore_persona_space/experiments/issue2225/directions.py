"""Issue #2225 — E1/E2/E3 direction extraction per trait (plan §4.2).

Per trait (evil / sycophancy / hallucination), builds three direction-tensor
variants, each ``(28, 3584)`` float32 (all captured layers, the #778 r_B
indexing: tensor index 19 == paper 1-indexed layer 20):

- **E1 (response-avg, paper verbatim)** — REUSED #778 round-3 faithful
  directions ``rb_v2/{trait}.pt`` from the HF data repo (no re-extraction).
- **E2 (context-end)** — last-prompt-token activations over the 5 pairs x 20
  extraction questions x 2 signs = 200 contexts per trait, captured fresh via
  ``issue778_lib.capture_last_prompt_token_all_layers`` (one forward per
  context; all 10 rollouts under a context share one context-end activation).
- **E3 (prefix-end, the both-arms rule cell)** — same forward shape, read at
  the LAST token of the SYSTEM segment (the ``<|im_start|>`` structure
  boundary per the issue1415 convention — token-id based, never a re-tokenized
  string slice).

Context-level judge filter (plan §4.2, stated adaptation of the paper's
rollout-level filter): keep a POSITIVE context iff the MEAN of its 10
rollouts' judged trait scores > 50; NEGATIVE contexts symmetric < 50. Scores
come from #778's AUTHORITATIVE per-rollout scores in
``analysis_tensors_v2/pairing/{trait}_pairing.json`` (fact-check 2026-08-10:
the ``judge_raw_trait.json`` per-rollout SUMMARY block is empty —
mean_aligned=None on all rows; the pairing rows carry the real per-rollout
scores). Fallback score source when the pairing rows are unusable: the
judge_raw ``all_scores`` map, keyed ``{rollout_id}__{item:05d}__{draw:02d}``
with ``rollout_id = {arm}-{pair}-{question}-{rollout:03d}``. REFUSAL /
non-numeric / out-of-range rollout returns are already dropped (None) in the
#778 raws — drop-never-coerce; an all-None context is UNEVALUABLE (dropped,
counted). Yield ladder per trait (graceful degradation, plan §4.2): >=50 kept
positive AND >=50 kept negative contexts -> ``ok``; 30-49 -> ``shortfall``
(proceed, documented); <30 -> ``fallback_prompt_sign`` (no judge filter,
disclosed deviation).

Outputs land under ``--out-dir`` (checkpoint-per-trait: each trait's tensors +
meta JSON are written the moment the trait completes):

    {trait}_E1.pt / {trait}_E2.pt / {trait}_E3.pt   (28, 3584) float32
    {trait}_meta.json    kept/dropped counts per arm, ladder status, score
                         source, per-variant-per-layer norms, L1 layer, repro

``--upload`` pushes the finished out-dir to the HF data repo prefix
``issue2225_ctxsteer/analysis_tensors/directions/`` (invoked pod-side AFTER
the build; the default build path never uploads).
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from collections.abc import Sequence

logger = logging.getLogger("issue2225.directions")

# ── constants ──────────────────────────────────────────────────────────────────

DATA_REPO = "superkaiba1/explore-persona-space-data"
V2_PREFIX = "issue778_persona_vectors/analysis_tensors_v2"
DIRECTIONS_HF_PREFIX = "issue2225_ctxsteer/analysis_tensors/directions"

# Paper steering layers (plan §4.3): evil 20 / sycophancy 20 / hallucination 16,
# 1-indexed -> 0-indexed vector indices 19 / 19 / 15 (#778 r_B convention).
L1_LAYER_IDX: dict[str, int] = {"evil": 19, "sycophancy": 19, "hallucination": 15}

VARIANTS: tuple[str, ...] = ("E1", "E2", "E3")
JUDGE_THRESHOLD = 50.0
YIELD_FLOOR_OK = 50  # >=50 kept pos AND >=50 kept neg -> ok
YIELD_FLOOR_SHORTFALL = 30  # 30-49 -> shortfall; <30 -> fallback_prompt_sign


def _ensure_repo_root_on_syspath() -> None:
    """Insert the repo root on sys.path so ``scripts.*`` imports resolve.

    In script mode (``python /abs/path/directions.py``) sys.path[0] is this
    file's own directory, so ``scripts/`` is unreachable without this guard
    (gotchas.md "Script mode puts the SCRIPT's dir on sys.path[0]", #823).
    Idempotent; fails loud on a wrong parents[] depth via the sentinel.
    """
    # directions.py -> issue2225/ -> experiments/ -> explore_persona_space/ -> src/ -> root
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    sentinel = repo_root / "scripts" / "issue778_lib.py"
    if not sentinel.exists():
        raise RuntimeError(
            f"_ensure_repo_root_on_syspath: sentinel {sentinel} missing; derived "
            f"repo_root={repo_root} looks wrong"
        )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _lib():
    """Deferred ``scripts.issue778_lib`` import (script-mode safe)."""
    _ensure_repo_root_on_syspath()
    from scripts import issue778_lib as lib

    return lib


# ── reused-artifact staging (E1 + score sources) ───────────────────────────────


def stage_reused_artifacts(trait: str, staging_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Stage the reused #778 artifacts for ``trait`` from the HF data repo.

    Returns {"rb_v2": ..., "pairing": ..., "judge_raw": ...} local paths.
    ``stage_hub_file`` is retried + atomic + idempotent (an existing target
    skips the network), fail-loud on exhaustion.
    """
    from explore_persona_space.orchestrate.hub import stage_hub_file

    out: dict[str, pathlib.Path] = {}
    for key, rel in (
        ("rb_v2", f"rb_v2/{trait}.pt"),
        ("pairing", f"pairing/{trait}_pairing.json"),
        ("judge_raw", f"judge/{trait}_judge_raw_trait.json"),
    ):
        out[key] = stage_hub_file(
            DATA_REPO,
            f"{V2_PREFIX}/{rel}",
            staging_dir / rel,
            repo_type="dataset",
        )
    return out


def load_e1(rb_path: pathlib.Path):
    """Load the reused #778 rb_v2 direction tensor (bare float32 tensor).

    ``weights_only=True`` is safe: the producer (issue778_extract.py) saved a
    bare ``torch.save(rb)`` tensor. Asserts the (N_LAYERS, HIDDEN_DIM) shape.
    """
    import torch

    lib = _lib()
    rb = torch.load(rb_path, weights_only=True, map_location="cpu")
    assert rb.shape == (lib.N_LAYERS, lib.HIDDEN_DIM), rb.shape
    return rb.to(torch.float32)


# ── context-level judge filter (plan §4.2) ─────────────────────────────────────

ContextKey = tuple[str, int, int]  # (side, pair_idx, question_idx)


def context_scores_from_pairing(pairing: dict) -> dict[ContextKey, list[float]]:
    """Per-context rollout trait-score lists from the AUTHORITATIVE pairing JSON.

    Each ``pairs`` row carries ``pair_key = [pair_idx, question_idx,
    rollout_idx]`` plus per-rollout ``pos_trait`` / ``neg_trait`` scores
    (None = dropped by the #778 judge — never coerced). A context is
    (side, pair_idx, question_idx); its list collects the non-None scores of
    its rollouts. Raises if the pairing carries no usable score at all (the
    caller then falls back to the judge_raw source).
    """
    ctx: dict[ContextKey, list[float]] = {}
    n_scores = 0
    for row in pairing["pairs"]:
        pair_idx, question_idx, _rollout_idx = row["pair_key"]
        for side, field in (("pos", "pos_trait"), ("neg", "neg_trait")):
            key = (side, int(pair_idx), int(question_idx))
            ctx.setdefault(key, [])
            score = row.get(field)
            if score is not None:
                ctx[key].append(float(score))
                n_scores += 1
    if n_scores == 0:
        raise ValueError(
            "pairing JSON carries zero non-None per-rollout trait scores — fall back "
            "to the judge_raw all_scores source"
        )
    return ctx


def context_scores_from_judge_raw(judge_raw_path: pathlib.Path) -> dict[ContextKey, list[float]]:
    """FALLBACK per-context score lists from the judge_raw ``all_scores`` map.

    Keys are ``{rollout_id}__{item:05d}__{draw:02d}`` with
    ``rollout_id = {arm}-{pair}-{question}-{rollout:03d}``
    (issue778_extract._v2_prompt_records). Parsing reuses the production
    ``_score_from_parsed`` reduce (drop-never-coerce).
    """
    from explore_persona_space.eval.graded_judge import _score_from_parsed

    with open(judge_raw_path) as f:
        raw = json.load(f)
    all_scores: dict[str, dict] = raw.get("all_scores", {})
    if not all_scores:
        raise ValueError(f"judge_raw fallback has no all_scores map: {judge_raw_path}")
    ctx: dict[ContextKey, list[float]] = {}
    for cid, parsed in all_scores.items():
        rollout_id = cid.rsplit("__", 2)[0]
        parts = rollout_id.split("-")
        if len(parts) != 4 or parts[0] not in ("pos", "neg"):
            raise ValueError(f"unparseable rollout_id {rollout_id!r} in {judge_raw_path}")
        side, pair_idx, question_idx = parts[0], int(parts[1]), int(parts[2])
        key = (side, pair_idx, question_idx)
        ctx.setdefault(key, [])
        score = _score_from_parsed(parsed)
        if score is not None:
            ctx[key].append(float(score))
    return ctx


def apply_context_filter(
    context_keys: Sequence[ContextKey],
    ctx_scores: dict[ContextKey, list[float]],
    *,
    threshold: float = JUDGE_THRESHOLD,
) -> tuple[list[ContextKey], dict]:
    """Context-level judge filter: keep pos iff mean > thr, neg iff mean < thr.

    ``context_keys`` is the FULL ordered context grid (the capture order); a
    context with zero scored rollouts is UNEVALUABLE (dropped, counted
    separately from filtered-out). Returns (kept_keys, counts) where counts
    carries the per-arm kept / filtered_out / unevaluable breakdown the plan
    requires persisted alongside the tensors.
    """
    kept: list[ContextKey] = []
    counts = {
        "n_contexts_pos": 0,
        "n_contexts_neg": 0,
        "kept_pos": 0,
        "kept_neg": 0,
        "filtered_out_pos": 0,
        "filtered_out_neg": 0,
        "unevaluable_pos": 0,
        "unevaluable_neg": 0,
    }
    for key in context_keys:
        side = key[0]
        counts[f"n_contexts_{side}"] += 1
        scores = ctx_scores.get(key, [])
        if not scores:
            counts[f"unevaluable_{side}"] += 1
            continue
        mean = sum(scores) / len(scores)
        keep = mean > threshold if side == "pos" else mean < threshold
        if keep:
            kept.append(key)
            counts[f"kept_{side}"] += 1
        else:
            counts[f"filtered_out_{side}"] += 1
    return kept, counts


def yield_ladder_status(kept_pos: int, kept_neg: int) -> str:
    """Plan §4.2 yield ladder: ok / shortfall / fallback_prompt_sign."""
    floor = min(kept_pos, kept_neg)
    if floor >= YIELD_FLOOR_OK:
        return "ok"
    if floor >= YIELD_FLOOR_SHORTFALL:
        return "shortfall"
    return "fallback_prompt_sign"


# ── context prompts + capture ──────────────────────────────────────────────────


def build_context_prompts(
    td, tokenizer, *, n_pairs: int | None = None, n_questions: int | None = None
) -> tuple[list[ContextKey], list[str]]:
    """Ordered (context_key, chat-templated prompt) grid for one trait.

    Order: pos arm first (pair-major, then question), then neg — mirroring the
    #778 v2 row layout. Prompts are the extraction contexts (system prompt +
    extraction question + assistant generation header), rendered with the same
    ``apply_chat_template(..., add_generation_prompt=True)`` call the #778
    extraction used (issue778_extract._chat_prompt).
    """
    lib = _lib()
    questions = td.extract_questions[: n_questions or len(td.extract_questions)]
    keys: list[ContextKey] = []
    prompts: list[str] = []
    for side in ("pos", "neg"):
        instrs = td.pos_instructions if side == "pos" else td.neg_instructions
        instrs = instrs[: n_pairs or len(instrs)]
        for k, instr in enumerate(instrs):
            system = lib.extraction_system_prompt(td.trait, instr, side)
            for qi, q in enumerate(questions):
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": q},
                ]
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
                keys.append((side, k, qi))
    return keys, prompts


def capture_prefix_end_all_layers(model, tokenizer, prompts: Sequence[str], *, device):
    """Prefix-end (last SYSTEM-segment token) activation at every captured layer.

    E3 sibling of ``issue778_lib.capture_last_prompt_token_all_layers`` (same
    conventions: ``add_special_tokens=False``, one forward per context,
    hidden_states index layer+1, float32 CPU output) reading position
    ``prefix_end - 1`` — the last token BEFORE the user turn, located on the
    TOKEN IDS via the issue1415 ``<|im_start|>`` boundary (never a re-tokenized
    string slice).
    """
    import torch

    from explore_persona_space.experiments.issue1415.steering import prefix_end_index

    lib = _lib()
    out = torch.empty((len(prompts), lib.N_LAYERS, lib.HIDDEN_DIM), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            ids = inputs["input_ids"][0].tolist()
            prefix_end = prefix_end_index(tokenizer, ids)
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states
            if len(hs) != lib.N_LAYERS + 1:
                raise ValueError(f"expected {lib.N_LAYERS + 1} hidden_states, got {len(hs)}")
            for layer_idx in range(lib.N_LAYERS):
                last_sys = hs[layer_idx + 1][:, prefix_end - 1, :].squeeze(0)
                out[i, layer_idx] = last_sys.detach().float().cpu()
            del outputs
    return out


# ── per-trait build ────────────────────────────────────────────────────────────


def _diff_of_means(acts, keys: Sequence[ContextKey], kept: Sequence[ContextKey]):
    """Direction = mean over kept pos contexts - mean over kept neg contexts."""
    import torch

    index = {key: i for i, key in enumerate(keys)}
    pos_rows = [index[k] for k in kept if k[0] == "pos"]
    neg_rows = [index[k] for k in kept if k[0] == "neg"]
    assert pos_rows and neg_rows, (len(pos_rows), len(neg_rows))
    pos_mean = acts[torch.tensor(pos_rows, dtype=torch.long)].mean(dim=0)
    neg_mean = acts[torch.tensor(neg_rows, dtype=torch.long)].mean(dim=0)
    return (pos_mean - neg_mean).to(torch.float32)


def build_directions_for_trait(
    trait: str,
    model,
    tokenizer,
    *,
    device,
    staging_dir: pathlib.Path,
    out_dir: pathlib.Path,
    external_root: pathlib.Path,
    n_pairs: int | None = None,
    n_questions: int | None = None,
) -> dict:
    """Build + save E1/E2/E3 for one trait; returns the meta dict.

    Checkpoint-per-trait: tensors + meta land on disk before returning.
    """
    import torch

    lib = _lib()
    t0 = time.time()
    staged = stage_reused_artifacts(trait, staging_dir)

    # E1 — reused #778 rb_v2 (no re-extraction).
    e1 = load_e1(staged["rb_v2"])

    # Score source: pairing JSON (AUTHORITATIVE) -> judge_raw all_scores fallback.
    with open(staged["pairing"]) as f:
        pairing = json.load(f)
    try:
        ctx_scores = context_scores_from_pairing(pairing)
        score_source = "pairing"
    except (ValueError, KeyError) as e:
        logger.warning(
            "[directions] trait=%s pairing score source unusable (%s); "
            "falling back to judge_raw all_scores",
            trait,
            e,
        )
        ctx_scores = context_scores_from_judge_raw(staged["judge_raw"])
        score_source = "judge_raw_fallback"

    # Context grid + fresh captures (E2 context-end, E3 prefix-end).
    td = lib.load_trait_data(external_root, trait)
    keys, prompts = build_context_prompts(td, tokenizer, n_pairs=n_pairs, n_questions=n_questions)
    print(f"[directions] trait={trait} capturing {len(prompts)} contexts (E2+E3)", flush=True)
    e2_acts = lib.capture_last_prompt_token_all_layers(model, tokenizer, prompts, device=device)
    e3_acts = capture_prefix_end_all_layers(model, tokenizer, prompts, device=device)

    # Context-level judge filter + yield ladder.
    kept, counts = apply_context_filter(keys, ctx_scores)
    ladder = yield_ladder_status(counts["kept_pos"], counts["kept_neg"])
    if ladder == "fallback_prompt_sign":
        # <30 kept on an arm: prompt-sign pools, no judge filter (disclosed).
        kept = list(keys)
        logger.warning(
            "[directions] trait=%s yield ladder FALLBACK to prompt-sign pools "
            "(kept_pos=%d kept_neg=%d)",
            trait,
            counts["kept_pos"],
            counts["kept_neg"],
        )

    e2 = _diff_of_means(e2_acts, keys, kept)
    e3 = _diff_of_means(e3_acts, keys, kept)

    # A6 sensitivity inputs (plan §4.7 / §12 A6): the UNFILTERED prompt-sign-pool
    # variants are computed in the SAME capture pass and persisted, so the P5
    # analysis can report cosine(filtered, unfiltered) without re-capturing.
    all_keys = list(keys)
    e2_unfiltered = _diff_of_means(e2_acts, keys, all_keys)
    e3_unfiltered = _diff_of_means(e3_acts, keys, all_keys)

    def _per_layer_cosine(a, b) -> list[float]:
        num = (a * b).sum(dim=1)
        den = a.norm(dim=1) * b.norm(dim=1)
        return [float(x) for x in (num / den.clamp_min(1e-12))]

    a6 = {
        "note": (
            "cosine between the context-level-judge-FILTERED direction and the "
            "UNFILTERED prompt-sign-pool direction (plan §12 A6 sensitivity); "
            "identical (1.0) by construction when the yield ladder fell back to "
            "prompt-sign pools"
        ),
        "cosine_filtered_vs_unfiltered": {
            "E2": _per_layer_cosine(e2, e2_unfiltered),
            "E3": _per_layer_cosine(e3, e3_unfiltered),
        },
        "cosine_at_l1": {
            "E2": _per_layer_cosine(e2, e2_unfiltered)[L1_LAYER_IDX[trait]],
            "E3": _per_layer_cosine(e3, e3_unfiltered)[L1_LAYER_IDX[trait]],
        },
        "filter_was_noop": ladder == "fallback_prompt_sign",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "E1": e1,
        "E2": e2,
        "E3": e3,
        "E2_unfiltered": e2_unfiltered,
        "E3_unfiltered": e3_unfiltered,
    }
    norms: dict[str, list[float]] = {}
    for variant, tensor in tensors.items():
        assert tensor.shape == (lib.N_LAYERS, lib.HIDDEN_DIM), (variant, tensor.shape)
        torch.save(tensor, out_dir / f"{trait}_{variant}.pt")
        norms[variant] = [float(torch.norm(tensor[layer])) for layer in range(lib.N_LAYERS)]

    meta = {
        "trait": trait,
        # ALL persisted tensor variants (E1/E2/E3 + the A6 *_unfiltered pair) —
        # not the bare VARIANTS triple (g4 minor: meta/files consistency).
        "variants": list(tensors),
        "primary_variants": list(VARIANTS),
        "shape": [lib.N_LAYERS, lib.HIDDEN_DIM],
        "score_source": score_source,
        # STATED DEVIATION (g1 Concern 3) from plan §4.2's "same capture forward
        # passes": E3 prefix-end reads run a SECOND batch-1 forward per context
        # (capture_prefix_end_all_layers after capture_last_prompt_token_all_
        # layers). Numerically identical output (deterministic no_grad reads of
        # the same hidden_states), ~2x the capture forwards — compute-only.
        "capture_deviation": (
            "E2 and E3 read the same contexts in two separate forward passes "
            "(compute-only deviation from plan §4.2; outputs identical)"
        ),
        "context_filter": {
            "semantics": (
                "context-level adaptation of the paper's rollout-level filter: keep a "
                "positive context iff mean of its rollouts' trait scores > 50; negatives "
                "symmetric < 50; unevaluable (zero scored rollouts) dropped, counted. "
                "Trait-score-only (no coherence gate) — stated adaptation, plan §4.2."
            ),
            "threshold": JUDGE_THRESHOLD,
            **counts,
            "yield_ladder": ladder,
            "prompt_sign_fallback_applied": ladder == "fallback_prompt_sign",
        },
        "n_contexts_captured": len(keys),
        "n_contexts_kept": len(kept),
        "l1_layer_idx": L1_LAYER_IDX[trait],
        "l1_layer_1indexed": L1_LAYER_IDX[trait] + 1,
        "norms_per_variant_per_layer": norms,
        "a6_sensitivity": a6,
        "e1_provenance": f"hf://{DATA_REPO}/{V2_PREFIX}/rb_v2/{trait}.pt (reused #778 round-3)",
        "wall_s": round(time.time() - t0, 1),
        "reproducibility": lib.repro_metadata(),
    }
    with open(out_dir / f"{trait}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[directions] trait={trait} done kept={len(kept)}/{len(keys)} ladder={ladder} "
        f"source={score_source} elapsed={meta['wall_s']}s",
        flush=True,
    )
    return meta


# ── upload helper (invoked pod-side AFTER the build; never on the build path) ──


def upload_directions(out_dir: pathlib.Path) -> str:
    """Upload the finished directions out-dir to the HF data repo (one
    ``upload_folder`` commit via the canonical ``hub._upload`` folder branch).
    """
    from explore_persona_space.orchestrate.hub import _upload

    return _upload(
        pathlib.Path(out_dir),
        DATA_REPO,
        "dataset",
        DIRECTIONS_HF_PREFIX,
        raise_on_error=True,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traits", default="evil,sycophancy,hallucination")
    ap.add_argument("--out-dir", default="eval_results/issue_2225/directions")
    ap.add_argument("--staging-dir", default="data/issue_2225/hf_dl/issue778_v2")
    ap.add_argument("--external-root", default="external/persona_vectors")
    ap.add_argument("--model", default=None, help="default: issue778_lib.MODEL_NAME")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="tiny slice: 1 pair x 2 questions")
    ap.add_argument("--upload", action="store_true", help="upload-only mode (pod-side, later)")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_argparser().parse_args(argv)
    if args.import_check:
        # Execute every deferred import + the args-attribute completeness scan.
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _lib()
        from explore_persona_space.eval.graded_judge import _score_from_parsed  # noqa: F401
        from explore_persona_space.experiments.issue1415.steering import (  # noqa: F401
            prefix_end_index,
        )
        from explore_persona_space.orchestrate.hub import _upload, stage_hub_file  # noqa: F401

        print("[directions] import-check OK", flush=True)
        raise SystemExit(0)

    out_dir = pathlib.Path(args.out_dir)
    if args.upload:
        url = upload_directions(out_dir)
        print(f"[directions] uploaded {out_dir} -> {url}", flush=True)
        sys.stdout.flush()
        sys.exit(0)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lib = _lib()
    model_name = args.model or lib.MODEL_NAME
    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    for trait in traits:
        if trait not in lib.TRAITS:
            raise ValueError(f"unknown trait {trait!r}; expected subset of {lib.TRAITS}")

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    for trait in traits:
        build_directions_for_trait(
            trait,
            model,
            tokenizer,
            device=device,
            staging_dir=pathlib.Path(args.staging_dir),
            out_dir=out_dir,
            external_root=pathlib.Path(args.external_root),
            n_pairs=1 if args.smoke else None,
            n_questions=2 if args.smoke else None,
        )
    print(f"[directions] all traits done -> {out_dir}", flush=True)
    # Explicit exit: heavy C-extension imports (torch/transformers) can hit the
    # PyGILState_Release atexit race on a bare return (gotchas.md).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
