#!/usr/bin/env python
"""#1434 D4/D5 — writing_style persona-vector extraction, projection DV, nulls.

Instantiates `.claude/rules/persona-vectors-recipe.md` steps 1-7 (READ-OUT
regime) for the registered ``writing_style`` behavior, plus the plan-§4 D5
projection-DV captures and the selection-symmetric validation battery.

Phases (pod GPU unless noted):

- ``--phase extract``   D4: 5 registered pairs x 20 extraction questions x
  10 rollouts x 2 arms (vLLM, temp 1.0) -> pv-rubric judge filter
  (>50 keep exhibit / <50 keep not_exhibit; REFUSAL/non-numeric DROPPED, never
  coerced; per-arm counts persisted) -> batched teacher-forced response-avg
  activations at ALL 28 layers -> per-arm pool stacks (the honest-null Σ
  inputs) + ``r_B`` diff-of-means -> ``save_direction`` + uploads.
- ``--phase project``   D5: per state (per-context base + trained arms at
  their selected rungs) capture FOUR arms over the 20 eval questions under the
  organism's training context — prefix-end / context-end (offset-mapped spans,
  ``on_seam='snap'``) / response-avg over SHARED base-greedy text (PRIMARY) /
  response-avg over OWN greedy text (exploratory) — shifts + projections.
- ``--phase validate``  (VM) Spearman rho(projection, judged delta) per layer
  across the (verdict organism x panel context) + own-context install grids;
  200 HONEST norm-matched randnorm draws (Σ arm-centered `within_class`
  PRIMARY + `neg_arm_only`, the #778 round-3 families — the retired pooled-Σ
  family is never drawn) + 10,000 label-shuffle draws, BOTH per-draw
  max-over-28-layers (selection-symmetric); per-draw x per-layer matrices
  persisted; LOFO/LOO sweeps + cluster bootstrap.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1074_generator_compare as i1074  # noqa: E402
import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_run as run1090  # noqa: E402
import issue1434_cells as cells  # noqa: E402
import issue1434_worker as worker  # noqa: E402

from explore_persona_space.analysis.representation_shift import (  # noqa: E402
    compute_prompt_spans,
)
from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.context import Context  # noqa: E402
from explore_persona_space.artifacts.directions import (  # noqa: E402
    ContrastiveCompletion,
    DirectionResult,
    batched_response_means,
    encode_rows,
    filter_completions,
    load_direction,
    save_completions_jsonl,
    save_direction,
    select_readout_layer,
)
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    _generate_and_persist,
)
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1434.pv")

N_ROLLOUTS = 10  # persona-vectors-recipe step 3
GEN_MAX_NEW = 1024  # #1090 GEN_MAX_NEW_TOKENS parity (plan §11)
N_RANDNORM_DRAWS = 200  # plan §4 D5
N_SHUFFLE_DRAWS = 10_000  # plan §4 D5
PAPER_COMPANION_LAYER = 19  # paper layer 20, r_B 0-index (issue778 LAYER_RESOLUTION_V2)
CAPTURE_ARMS = ("prefix_end", "context_end", "response_shared", "response_own")


def pv_root(cfg) -> Path:
    return Path(cfg.out_root) / "pv"


def _extraction_questions() -> list[str]:
    """The recipe's 20-question extraction set == the train bank (datagen-only
    adoption; disjoint from the 20-question eval bank by the SLICES audit)."""
    return list(BEHAVIORS[cells.BEHAVIOR].train_question_bank)


def _pair_context(pair_index: int, arm: str, instruction: str) -> Context:
    return Context(
        context_id=f"ws-pv-p{pair_index}-{arm}",
        kind="persona",
        family="pv_extraction",
        system=instruction,
        source="persona-vectors extraction pair (behavior.py registered)",
    )


def _pv_smoke_gen():
    """Arm-aware deterministic generation stub for the extract smoke (matches
    the eval-gen seam signature: (side_path, messages_list, *, n, temperature))."""

    def gen(side_path, messages_list, *, n, temperature):
        del side_path, temperature
        out = []
        for msgs in messages_list:
            system = next((m["content"] for m in msgs if m.get("role") == "system"), "")
            keys = ("casual", "colloquial", "friend", "informal", "contraction")
            casual = any(w in system.lower() for w in keys)
            if casual:
                text = (
                    "haha yeah so basically it's super simple — you just grab the thing, "
                    "give it a go, and honestly don't sweat the details. it kinda sorts "
                    "itself out, y'know?"
                )
            else:
                text = (
                    "This process comprises three distinct stages. First, one must assemble "
                    "the requisite materials. Subsequently, the procedure is executed in "
                    "accordance with the established guidelines. Finally, the results are "
                    "reviewed systematically."
                )
            out.append([text] * n)
        return out

    gen.close = lambda: None
    return gen


# ── D4: extraction ───────────────────────────────────────────────────────────


def phase_extract(cfg, args) -> int:
    """Recipe steps 3-6 + pool persistence (the honest-null Σ inputs)."""
    run1090._phase("i1434_pv_extract")
    behavior = BEHAVIORS[cells.BEHAVIOR]
    pairs = behavior.extraction.prompt_pairs
    qs = _extraction_questions()
    if cfg.smoke:
        pairs = pairs[:1]
        qs = qs[: (cfg.eval_question_limit or 2)]
    n_rollouts = 2 if cfg.smoke else N_ROLLOUTS
    root = pv_root(cfg)
    rollout_dir = root / "extraction_rollouts"

    # Step 3 — on-policy rollouts under each pair's pos/neg system prompt.
    # Smoke: the shared eval-gen stub is ARM-BLIND (one canned completion per
    # behavior), so the judge filter would legitimately zero an arm; the
    # extract smoke fakes ONLY the generation boundary with an ARM-AWARE stub
    # (keyed off the pair instruction in the system slot) — judge, filter,
    # capture, and diff-of-means all stay real.
    gen = _pv_smoke_gen() if cfg.smoke else worker._gen_fn(cfg)
    completions: list[ContrastiveCompletion] = []
    try:
        for pi, pair in enumerate(pairs):
            for arm, instruction in (("exhibit", pair.exhibit), ("not_exhibit", pair.not_exhibit)):
                ctx = _pair_context(pi, arm, instruction)
                comps = _generate_and_persist(
                    gen,
                    "base",
                    None,
                    ctx,
                    qs,
                    n=n_rollouts,
                    temperature=1.0,
                    out_dir=rollout_dir,
                    base_model=DEFAULT_BASE_MODEL,
                )
                for qi, q in enumerate(qs):
                    for resp in comps[qi]:
                        completions.append(
                            ContrastiveCompletion(
                                arm=arm,
                                pair_index=pi,
                                system_prompt=instruction,
                                question=q,
                                response=resp,
                            )
                        )
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()

    # Step 4 — pv-rubric judge filter (drop-never-coerce; per-arm counts).
    items = [
        (f"{c.arm}-p{c.pair_index}-{i:05d}", c.question, c.response)
        for i, c in enumerate(completions)
    ]
    result = judge_graded(
        items,
        cells.pv_rubric_text(),
        n_draws=cfg.n_judge_draws,
        cache_dir=root / "judge_cache_extract",
        save_raw=root / "judge_raw_extract.json",
        judge_model=behavior.judge_model,
        max_tokens=fu3w.JUDGE_MAX_TOKENS,
    )
    scored = [
        dataclasses.replace(c, judge_score=result.scores.get(items[i][0]))
        for i, c in enumerate(completions)
    ]
    save_completions_jsonl(scored, root / "extraction_scored.jsonl")
    kept, counts = filter_completions(scored, threshold=behavior.threshold)
    logger.info("[i1434-pv] judge filter counts: %s", counts)
    for arm in ("exhibit", "not_exhibit"):
        if counts[arm]["kept"] == 0:
            raise RuntimeError(
                f"[i1434-pv] extraction arm {arm!r} kept 0 rollouts after the pv "
                f"judge filter ({counts[arm]}) — the direction cannot be built"
            )

    # Steps 5-6 — batched teacher-forced response-avg capture + diff-of-means,
    # with the per-rollout pool STACKS persisted (the honest-null Σ inputs).
    run1090._phase("i1434_pv_capture")
    model, tokenizer = worker._hf_model(cfg)
    n_layers = int(model.config.num_hidden_layers)
    layers = list(range(n_layers))
    rows, enc_counts = encode_rows(tokenizer, kept)
    keep_idx = [i for i, r in enumerate(rows) if r is not None]
    logger.info("[i1434-pv] encode counts: %s", enc_counts)
    means = batched_response_means(
        model, [rows[i] for i in keep_idx], layers, batch_size=(2 if cfg.smoke else 8)
    )
    pools: dict[str, list[torch.Tensor]] = {"exhibit": [], "not_exhibit": []}
    for j, i in enumerate(keep_idx):
        pools[kept[i].arm].append(means[j])
    acts = {arm: torch.stack(stacks).to(torch.float16) for arm, stacks in pools.items() if stacks}
    for arm in ("exhibit", "not_exhibit"):
        if arm not in acts:
            raise RuntimeError(f"[i1434-pv] arm {arm!r} has zero encoded rollouts post-capture")
    torch.save(
        {"exhibit": acts["exhibit"], "not_exhibit": acts["not_exhibit"], "layers": layers},
        root / "extraction_pools.pt",
    )
    r_b = (
        acts["exhibit"].to(torch.float64).mean(dim=0)
        - acts["not_exhibit"].to(torch.float64).mean(dim=0)
    ).to(torch.float32)
    direction = DirectionResult(
        behavior_name=cells.BEHAVIOR,
        regime="read_out",
        layers=tuple(layers),
        r_b=r_b,
        counts={
            "filter": counts,
            "encode": enc_counts,
            "pool_sizes": {a: int(t.shape[0]) for a, t in acts.items()},
        },
        provenance="on_policy",
        metadata={
            "issue": cells.ISSUE_1434,
            "judge_n_draws": cfg.n_judge_draws,
            "rubric": "pv_writing_style_trait_score_v1",
            "rubric_provenance": cells.load_pv_provenance(),
            "n_rollouts": n_rollouts,
            "n_pairs": len(pairs),
            "n_questions": len(qs),
            "temperature": 1.0,
            "max_new_tokens": GEN_MAX_NEW,
            "git_commit": i1074._git_short_sha(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    save_direction(direction, root / "rb_writing_style.pt")
    if cfg.upload:
        for local, pir in (
            (rollout_dir, f"{cells.DATA_PREFIX_1434}/raw_completions/extraction"),
            (root / "extraction_scored.jsonl", None),
            (root / "extraction_pools.pt", None),
            (root / "rb_writing_style.pt", None),
            (root / "judge_raw_extract.json", None),
        ):
            pir = pir or f"{cells.DATA_PREFIX_1434}/analysis_tensors/{Path(local).name}"
            url = hub._upload(
                Path(local),
                run1090.HF_DATA_REPO,
                "dataset",
                pir,
                upload_as_file=Path(local).is_file(),
            )
            if not str(url):
                raise RuntimeError(f"pv upload returned no path for {pir}")
    return 0


# ── D5: projection captures ──────────────────────────────────────────────────


def _states_for(cfg, cell_keys: list[str]) -> list[dict]:
    """The D5 state list: per context, the base state + every trained arm at
    its selected rung (diverged arms carry no selection and are skipped)."""
    states: list[dict] = [
        {"state_id": f"base-{ck}", "cell_key": ck, "ckpt": None} for ck in cell_keys
    ]
    import issue1090_fu4 as fu4

    for run in fu4.resolve_fu4_runs(None, cfg.smoke):
        if run.cell_key not in cell_keys:
            continue
        path = Path(cfg.out_root) / run.run_id / "i1434_build_result.json"
        if not path.exists():
            raise RuntimeError(f"[i1434-pv] missing build result {path} — run dispatch first")
        rec = run1090._read_json(path)
        if rec.get("status") != "trained":
            continue
        states.append(
            {"state_id": run.run_id, "cell_key": run.cell_key, "ckpt": rec["selected_ckpt"]}
        )
    return states


def _encode_ctx_rows(tokenizer, ctx: Context, qs: list[str], responses: list[str]):
    """(full_ids, prompt_len) rows for teacher-forcing under an ARBITRARY
    context (prefix turns / user_wrap included) — the encode_rows fail-loud
    contract (token-id prefix check; mismatches skipped + counted)."""
    rows = []
    counts = {"encoded": 0, "skipped_empty_response": 0, "skipped_prefix_mismatch": 0}
    for q, resp in zip(qs, responses, strict=True):
        msgs = ctx.messages(q)
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, padding=False)["input_ids"]
        full_text = tokenizer.apply_chat_template(
            [*msgs, {"role": "assistant", "content": resp}],
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = tokenizer(full_text, padding=False)["input_ids"]
        if len(full_ids) <= len(prompt_ids):
            rows.append(None)
            counts["skipped_empty_response"] += 1
            continue
        if full_ids[: len(prompt_ids)] != prompt_ids:
            rows.append(None)
            counts["skipped_prefix_mismatch"] += 1
            continue
        rows.append((full_ids, len(prompt_ids)))
        counts["encoded"] += 1
    return rows, counts


def _ctx_spans(tokenizer, ctx: Context, qs: list[str]) -> list[tuple[int, int, list[int], dict]]:
    """(prefix_len, context_len, prompt_ids, seam_flags) per eval question via
    the offset-mapped span helper (#1315 recipe; ``on_seam='snap'``)."""
    multi = bool(ctx.prefix_turns) or ctx.user_wrap is not None
    out = []
    for q in qs:
        msgs = ctx.messages(q)
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, padding=False)["input_ids"]
        seam_flags: dict = {}
        prefix_len, context_len = compute_prompt_spans(
            tokenizer,
            ctx.system,
            q,
            prompt_ids,
            prior_messages=list(ctx.prefix_turns) or None,
            user_wrap=ctx.user_wrap,
            prefix_end="last_user" if multi else "first_user",
            on_seam="snap",
            seam_flags=seam_flags,
        )
        out.append((prefix_len, context_len, prompt_ids, seam_flags))
    return out


@torch.no_grad()
def _capture_state(model, tokenizer, ctx: Context, qs, shared_text, own_text, layers) -> dict:
    """One state's 4-arm capture: per-arm fp32 ``(L, H)`` mean over questions."""
    from explore_persona_space.analysis.extraction import extract_layer_activations

    device = next(model.parameters()).device
    hidden = model.config.hidden_size
    sums = {arm: torch.zeros(len(layers), hidden, dtype=torch.float64) for arm in CAPTURE_ARMS}
    ns = dict.fromkeys(CAPTURE_ARMS, 0)
    seam_counts = {"prefix": 0, "context": 0}
    # prefix-end / context-end: single positions from offset-mapped spans.
    for prefix_len, context_len, prompt_ids, seam in _ctx_spans(tokenizer, ctx, qs):
        for k in seam_counts:
            seam_counts[k] += int(bool(seam.get(k)))
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        captured = extract_layer_activations(model, ids, layers)
        for li, layer in enumerate(layers):
            hs = captured[layer][0]  # (T, H)
            sums["prefix_end"][li] += hs[prefix_len - 1].to(torch.float64).cpu()
            sums["context_end"][li] += hs[context_len - 1].to(torch.float64).cpu()
        ns["prefix_end"] += 1
        ns["context_end"] += 1
    # response-avg arms (batched teacher-forced masked means).
    for arm, texts in (("response_shared", shared_text), ("response_own", own_text)):
        rows, enc = _encode_ctx_rows(tokenizer, ctx, qs, texts)
        live = [r for r in rows if r is not None]
        if not live:
            raise RuntimeError(f"[i1434-pv] {arm}: zero encodable rows ({enc})")
        means = batched_response_means(model, live, layers, batch_size=4)
        for m in means:
            sums[arm] += m.to(torch.float64)
        ns[arm] = len(means)
    return {
        "means": {arm: (sums[arm] / max(ns[arm], 1)).to(torch.float32) for arm in CAPTURE_ARMS},
        "n": ns,
        "seam_counts": seam_counts,
    }


def phase_project(cfg, args) -> int:
    """D5: shared/own greedy text per state, then the 4-arm captures + shifts
    + projections onto r̂_B (per layer, unit-normalized) + cosines."""
    run1090._phase("i1434_pv_project")
    qs = worker._eval_questions(cfg)
    cell_keys = worker.resolve_cell_keys(args.cells, cfg.smoke)
    states = _states_for(cfg, cell_keys)
    root = pv_root(cfg)
    cap_root = root / "capture"

    # 1. Greedy text per state (vLLM shared engine; LoRA hot-load per ckpt).
    run1090._phase("i1434_pv_greedy_text")
    gen = worker._gen_fn(cfg)
    try:
        for st in states:
            ctx = cells.ensure_ws_context(cells.CONTEXT_BY_CELL_KEY[st["cell_key"]])
            _generate_and_persist(
                gen,
                "base" if st["ckpt"] is None else "trained",
                st["ckpt"],
                ctx,
                qs,
                n=1,
                temperature=0.0,
                out_dir=cap_root / st["state_id"] / "greedy",
                base_model=DEFAULT_BASE_MODEL,
            )
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()

    # 2. Captures (HF model; PEFT adapter applied for trained states).
    run1090._phase("i1434_pv_capture_states")
    rb = load_direction(root / "rb_writing_style.pt")
    layers = list(rb.layers)
    base_model, tokenizer = worker._hf_model(cfg)

    def _greedy_text(state_id: str, cell_key: str, side: str) -> list[str]:
        ctx_id = cells.CONTEXT_BY_CELL_KEY[cell_key]
        payload = run1090._read_json(
            cap_root / state_id / "greedy" / f"completions__{side}__{ctx_id}.json"
        )
        return [per_q[0] for per_q in payload["completions"]]

    summaries: dict[str, dict] = {}
    for st in states:
        summary_path = cap_root / st["state_id"] / "summary.pt"
        if summary_path.exists():
            summaries[st["state_id"]] = torch.load(summary_path, weights_only=False)
            continue
        t0 = time.time()
        ctx = cells.ensure_ws_context(cells.CONTEXT_BY_CELL_KEY[st["cell_key"]])
        shared = _greedy_text(f"base-{st['cell_key']}", st["cell_key"], "base")
        own = _greedy_text(
            st["state_id"], st["cell_key"], "base" if st["ckpt"] is None else "trained"
        )
        if st["ckpt"] is None:
            model = base_model
        else:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base_model, st["ckpt"])
            model.eval()
        try:
            cap = _capture_state(model, tokenizer, ctx, qs, shared, own, layers)
        finally:
            if st["ckpt"] is not None:
                model = model.unload()  # strip LoRA; base weights restored in place
        rec = {
            "state_id": st["state_id"],
            "cell_key": st["cell_key"],
            "ckpt": st["ckpt"],
            "means": cap["means"],
            "n": cap["n"],
            "seam_counts": cap["seam_counts"],
            "layers": layers,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rec, summary_path)
        summaries[st["state_id"]] = rec
        # Pilot telemetry (plan §9 P8 pilot-gate): per-state wall so the FIRST
        # state's cost is readable from the log (>2x the ~6 min/state basis =>
        # the launcher posts epm:compute-deviation before proceeding).
        logger.info("[i1434-pv] state %s captured in %.1fs", st["state_id"], time.time() - t0)

    # 3. Shifts + projections (trained - base, per layer x arm).
    rhat = torch.nn.functional.normalize(rb.r_b.to(torch.float64), dim=-1)  # (L, H)
    rb_norms = rb.r_b.to(torch.float64).norm(dim=-1)  # (L,)
    proj: dict[str, Any] = {"layers": layers, "arms": list(CAPTURE_ARMS), "states": {}}
    for st in states:
        if st["ckpt"] is None:
            continue
        base_rec = summaries[f"base-{st['cell_key']}"]
        rec = summaries[st["state_id"]]
        entry = {}
        for arm in CAPTURE_ARMS:
            shift = rec["means"][arm].to(torch.float64) - base_rec["means"][arm].to(torch.float64)
            entry[arm] = {
                "projection": (shift * rhat).sum(dim=-1).tolist(),
                "cosine": torch.nn.functional.cosine_similarity(
                    shift, rb.r_b.to(torch.float64), dim=-1
                ).tolist(),
                "shift_norm": shift.norm(dim=-1).tolist(),
            }
        proj["states"][st["state_id"]] = entry
    proj["rb_norms"] = rb_norms.tolist()
    run1090._atomic_write_json(root / "projections.json", proj)
    if cfg.upload:
        url = hub._upload(
            cap_root,
            run1090.HF_DATA_REPO,
            "dataset",
            f"{cells.DATA_PREFIX_1434}/analysis_tensors/capture",
        )
        url2 = hub._upload(
            root / "projections.json",
            run1090.HF_DATA_REPO,
            "dataset",
            f"{cells.DATA_PREFIX_1434}/analysis_tensors/projections.json",
            upload_as_file=True,
        )
        if not str(url) or not str(url2):
            raise RuntimeError("pv projection upload returned no path")
    return 0


# ── validation (VM): Spearman + honest nulls + shuffle, selection-symmetric ──


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Tie-averaged rank transform along the LAST axis (vectorized, scipy)."""
    from scipy.stats import rankdata as _sp_rankdata

    return _sp_rankdata(np.asarray(x, dtype=np.float64), method="average", axis=-1)


def _spearman_obs_per_layer(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Spearman rho| per layer. ``P``: (L, n_cells) projections; ``y``: (n_cells,)."""
    ry = _rankdata(y[None, :])[0]
    rP = _rankdata(P)
    ry_c = ry - ry.mean()
    rP_c = rP - rP.mean(axis=1, keepdims=True)
    denom = np.sqrt((rP_c**2).sum(axis=1) * (ry_c**2).sum())
    denom = np.where(denom == 0, np.inf, denom)
    return np.abs((rP_c @ ry_c) / denom)


def _spearman_draws(D: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Spearman rho| per (draw, layer). ``D``: (n_draws, L, n_cells)."""
    ry = _rankdata(y[None, :])[0]
    ry_c = ry - ry.mean()
    rD = _rankdata(D.reshape(-1, D.shape[-1])).reshape(D.shape)
    rD_c = rD - rD.mean(axis=-1, keepdims=True)
    denom = np.sqrt((rD_c**2).sum(axis=-1) * (ry_c**2).sum())
    denom = np.where(denom == 0, np.inf, denom)
    return np.abs((rD_c @ ry_c) / denom)


def _cell_grids(cfg, aggregate: dict, proj: dict) -> dict[str, dict]:
    """The two registered validation grids: leakage (verdict organism x panel
    context) + own-context install (all trained arms). Values per capture arm."""
    grids: dict[str, dict] = {}
    # leakage grid: judged delta per (verdict organism, panel ctx)
    leak_y, leak_x_state, leak_ctx = [], [], []
    for _cell_key, prec in (aggregate.get("panel") or {}).items():
        run_id = prec["run_id"]
        if run_id not in proj["states"]:
            continue
        for ctx_id, row in prec["contexts"].items():
            leak_y.append(float(row["delta"]))
            leak_x_state.append(run_id)
            leak_ctx.append(ctx_id)
    # install grid: per trained arm own-context tier2 delta (verdict arms only
    # carry tier2 reads; ladder arms use their selection-rung Tier-1 rate delta
    # vs base — recorded per run in the aggregate ladders).
    inst_y, inst_x_state = [], []
    for _cell_key, entry in (aggregate.get("tier2") or {}).items():
        run_id = (entry.get("verdict_arm") or {}).get("run_id")
        if run_id and run_id in proj["states"] and entry.get("delta") is not None:
            inst_y.append(float(entry["delta"]))
            inst_x_state.append(run_id)
    grids["leakage"] = {"y": leak_y, "state_ids": leak_x_state, "ctx_ids": leak_ctx}
    grids["install"] = {"y": inst_y, "state_ids": inst_x_state, "ctx_ids": None}
    return grids


def phase_validate(cfg, args) -> int:  # noqa: C901 — the registered battery, one pass
    """H3 validation: observed per-layer |rho| + honest randnorm + shuffle
    bands, per-draw max-over-layer, persisted matrices, LOFO/LOO, bootstrap."""
    run1090._phase("i1434_pv_validate")
    import issue778_honest_null_ladder as hnl

    root = pv_root(cfg)
    deliver = cfg.out_root / "deliverables" if cfg.smoke else cells.DELIVERABLES_DIR_1434
    deliver.mkdir(parents=True, exist_ok=True)
    rb = load_direction(root / "rb_writing_style.pt")
    layers = list(rb.layers)
    rb64 = rb.r_b.to(torch.float64).numpy()
    rb_norms = np.linalg.norm(rb64, axis=-1)
    pools = torch.load(root / "extraction_pools.pt", weights_only=False)
    pos = pools["exhibit"].to(torch.float64).numpy()
    neg = pools["not_exhibit"].to(torch.float64).numpy()
    proj = run1090._read_json(root / "projections.json")
    agg_path = deliver / "i1434_ladders.json"
    aggregate = run1090._read_json(agg_path)
    lam = 0.1  # #778 PRIMARY_LAMBDA
    n_draws = max(8, N_RANDNORM_DRAWS // 25) if cfg.smoke else N_RANDNORM_DRAWS
    n_shuffle = 200 if cfg.smoke else N_SHUFFLE_DRAWS

    grids = _cell_grids(cfg, aggregate, proj)
    out: dict[str, Any] = {
        "layers": layers,
        "n_randnorm_draws": n_draws,
        "n_shuffle_draws": n_shuffle,
        "null_families": {
            "primary": "within_class (arm-centered Σ — #778 honest round-3 family)",
            "companion": "neg_arm_only",
            "retired_never_drawn": "pooled-Σ orig_randnorm (circular, #778)",
        },
        "grids": {},
    }
    rng_families = {
        "within_class": hnl._within_centered_pool(pos, neg),
        "neg_arm_only": neg - neg.mean(axis=0, keepdims=True),
    }
    primary_arm = "response_shared"
    for grid_name, grid in grids.items():
        y = np.asarray(grid["y"], dtype=np.float64)
        state_ids = grid["state_ids"]
        if len(y) < 3 or len(set(state_ids)) < 2:
            out["grids"][grid_name] = {
                "status": "insufficient_cells",
                "n_cells": len(y),
                "n_states": len(set(state_ids)),
            }
            continue
        grid_out: dict[str, Any] = {"n_cells": len(y), "n_states": len(set(state_ids))}
        for arm in CAPTURE_ARMS:
            # P: (L, n_cells) projections (per-state values broadcast per cell)
            P = np.stack(
                [np.asarray(proj["states"][sid][arm]["projection"]) for sid in state_ids],
                axis=1,
            )  # (L, n_cells)
            obs = _spearman_obs_per_layer(P, y)  # (L,)
            arm_out: dict[str, Any] = {
                "observed_abs_rho_per_layer": obs.tolist(),
                "max_layer": int(np.argmax(obs)),
                "max_abs_rho": float(obs.max()),
                "paper_companion_layer19_abs_rho": float(obs[PAPER_COMPANION_LAYER])
                if len(obs) > PAPER_COMPANION_LAYER
                else None,
            }
            if arm == primary_arm:
                # shifts per state: (n_states_unique, L, H) for the null battery
                uniq = sorted(set(state_ids))
                cap_root = root / "capture"
                shifts = {}
                for sid in uniq:
                    rec = torch.load(cap_root / sid / "summary.pt", weights_only=False)
                    base_rec = torch.load(
                        cap_root / f"base-{rec['cell_key']}" / "summary.pt", weights_only=False
                    )
                    shifts[sid] = (
                        rec["means"][arm].to(torch.float64)
                        - base_rec["means"][arm].to(torch.float64)
                    ).numpy()
                S = np.stack([shifts[sid] for sid in state_ids], axis=0)  # (n_cells, L, H)
                for fam, pool in rng_families.items():
                    chols = hnl._chols_for_layers(pool, layers, lam)
                    rng = np.random.default_rng(0 if fam == "within_class" else 1)
                    draws = np.empty((n_draws, len(layers), len(y)))
                    for d in range(n_draws):
                        for li, layer in enumerate(layers):
                            z = rng.standard_normal(rb64.shape[-1])
                            v = z @ chols[layer].T
                            nv = np.linalg.norm(v)
                            v = v * (rb_norms[li] / (nv if nv else 1.0))
                            draws[d, li] = S[:, li, :] @ v
                    null_rho = _spearman_draws(draws, y)  # (n_draws, L)
                    matrix_path = deliver / f"pv_nullmatrix_{grid_name}_{fam}.json"
                    headline = select_readout_layer(
                        torch.tensor(obs),
                        layers,
                        null_draws=torch.tensor(null_rho),
                        persist_path=matrix_path,
                    )
                    arm_out[f"null_{fam}"] = {
                        "band_p2_5_p97_5": list(headline.null_band or ()),
                        "max_selected_p97_5": float(np.quantile(null_rho.max(axis=1), 0.975)),
                        "matrix": str(matrix_path),
                        "headline_layer": headline.layer,
                        "observed_max": headline.observed_stat,
                        "ceiling": 1.0,
                        "band_to_ceiling_margin": float(
                            1.0 - np.quantile(null_rho.max(axis=1), 0.975)
                        ),
                    }
                # label-shuffle null (per-draw max-over-layer), fully batched:
                # rank(perm(y)) == perm(rank(y)), so permute the rank vector and
                # form ALL draws' correlations in ONE GEMM.
                rng = np.random.default_rng(2)
                rP = _rankdata(P)
                rP_c = rP - rP.mean(axis=1, keepdims=True)
                ry = _rankdata(y[None, :])[0]
                perms = np.stack([ry[rng.permutation(len(y))] for _ in range(n_shuffle)])
                perms_c = perms - perms.mean(axis=1, keepdims=True)  # (n_shuffle, n_cells)
                denom = np.sqrt((rP_c**2).sum(axis=1)[None, :] * (perms_c**2).sum(axis=1)[:, None])
                denom = np.where(denom == 0, np.inf, denom)
                sh = np.abs(perms_c @ rP_c.T) / denom  # (n_shuffle, L)
                sh_path = deliver / f"pv_nullmatrix_{grid_name}_shuffle.json"
                run1090._atomic_write_json(
                    sh_path,
                    {
                        "layers": layers,
                        "observed": obs.tolist(),
                        "max_selected_p97_5": float(np.quantile(sh.max(axis=1), 0.975)),
                        "n_draws": n_shuffle,
                        "note": "per-draw max-over-layer |spearman rho| label-shuffle null",
                    },
                )
                arm_out["null_shuffle"] = {
                    "max_selected_p97_5": float(np.quantile(sh.max(axis=1), 0.975)),
                    "matrix": str(sh_path),
                }
                # group-level folds (LOFO over panel contexts is grid-specific;
                # LOO over organisms applies to both grids)
                folds = {}
                for sid in sorted(set(state_ids)):
                    mask = np.array([s != sid for s in state_ids])
                    if mask.sum() >= 3 and len(set(np.array(state_ids)[mask])) >= 2:
                        folds[f"loo_organism_{sid}"] = float(
                            _spearman_obs_per_layer(P[:, mask], y[mask]).max()
                        )
                ctx_ids = grid.get("ctx_ids")
                if ctx_ids:
                    for cid in sorted(set(ctx_ids)):
                        mask = np.array([c != cid for c in ctx_ids])
                        if mask.sum() >= 3 and len(set(np.array(state_ids)[mask])) >= 2:
                            folds[f"lofo_context_{cid}"] = float(
                                _spearman_obs_per_layer(P[:, mask], y[mask]).max()
                            )
                arm_out["group_folds_max_abs_rho"] = folds
                # cluster bootstrap over organisms at the headline layer
                uniq_states = sorted(set(state_ids))
                li = int(np.argmax(obs))
                boots = []
                rngb = np.random.default_rng(3)
                for _ in range(1000 if not cfg.smoke else 50):
                    pick = rngb.choice(len(uniq_states), size=len(uniq_states), replace=True)
                    sel_states = [uniq_states[i] for i in pick]
                    idx = [i for s in sel_states for i, sid in enumerate(state_ids) if sid == s]
                    if len(set(state_ids[i] for i in idx)) < 2:
                        continue
                    boots.append(float(_spearman_obs_per_layer(P[li : li + 1, idx], y[idx])[0]))
                if boots:
                    arm_out["cluster_bootstrap_headline_layer"] = {
                        "layer": li,
                        "p2_5": float(np.quantile(boots, 0.025)),
                        "p97_5": float(np.quantile(boots, 0.975)),
                        "n_boot": len(boots),
                    }
            grid_out[arm] = arm_out
        out["grids"][grid_name] = grid_out
    out["git_commit"] = i1074._git_short_sha()
    out["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run1090._atomic_write_json(deliver / "pv_validation.json", out)
    logger.info("[i1434-pv] wrote %s", deliver / "pv_validation.json")
    return 0


# ── entrypoint ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="#1434 pv extraction / projection / validation")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--phase", required=True, choices=("extract", "project", "validate"))
    p.add_argument("--cells", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    args = p.parse_args(argv)
    cells.register_i1434_round()
    import issue1090_fu4 as fu4

    fu4.set_round("i1434")
    cfg = worker.worker_config(args)
    if args.phase == "extract":
        return phase_extract(cfg, args)
    if args.phase == "project":
        return phase_project(cfg, args)
    return phase_validate(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
