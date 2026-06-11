# ruff: noqa: RUF002
"""Issue #491 teacher-forced 4-float slot reads (HF forwards, plan v3 §4.4).

Capture path = the canonical ``compute_marker_slot_stats`` helper
(``eval/marker_logprob.py``) — per slot per model side it persists the FOUR
floats ``logp / z_marker / z_eos / logZ`` from ONE forward pass (vLLM cannot:
post-softmax only, incident #530). Gauge assert
(``assert_gauge_free_adapter_config``) runs before every adapter logit read.

Modes (CLI ``--mode``):
  icl_panel          one ICL variant (or the shared no-prefix baseline) over
                     contexts x Q_test on the BASE model.
  ft_run_pipeline    one FT run: matching-basis source reads at ALL stored
                     ckpts (pre-prune) -> matching.match_run -> trajectory
                     panel reads (8 grid ckpts x 10 ctx x 25 q) -> full reads
                     at the matched + anchor ckpts (10 ctx x 50 q) with the
                     winner's-curse residual re-read.
  inloop_crosscheck  the #534 adapter-application assert: offline read on the
                     run's K training-row probe contexts at the LAST stored
                     ckpt vs the in-loop band-stop trajectory's last record —
                     BLOCKING when the deltas differ by > 1.5 nat.
  own_policy         substrate-sensitivity reads at K=8 (plan v3 §4.5): the
                     4-float read re-run on each regime's OWN greedy
                     responses from the free_gen phase, first-marker-token
                     truncated.

Every output JSON is written the moment its unit completes (checkpoint per
phase) and carries the repro metadata block.
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import contextmanager
from pathlib import Path

from explore_persona_space.experiments.icl_vs_ft_491.common import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    EOS_ID,
    MARKER_ID,
    MARKER_TEXT,
    PANEL_CONTEXT_IDS,
    SOURCE_CONTEXT,
    load_panel_substrates,
    load_q_test,
    load_r_villain,
    load_tokenizer,
    ns_eval_dir,
    panel_system_prompts,
    render_slot_context,
    repro_metadata,
    write_json,
)
from explore_persona_space.experiments.icl_vs_ft_491.data_build import (
    load_run_specs,
    load_variants,
    resolve_demo_turns,
)
from explore_persona_space.experiments.icl_vs_ft_491.train_runs import (
    run_out_dir,
    trajectory_path,
)

logger = logging.getLogger("i491.slot_eval")

TRAJ_GRID_STEPS = [4, 8, 16, 24, 32, 48, 64, 96]
TRAJ_N_QUESTIONS = 25
CROSSCHECK_TOL_NATS = 1.5
TOKEN_BUDGET_PER_BATCH = 32768  # adaptive batch: full-logit memory guard
OWN_POLICY_CHAINS = ["A", "B", "C"]


# ── Model / adapter plumbing ─────────────────────────────────────────────


def load_base_model(device: str = "cuda:0"):
    """Pinned-revision bf16 base model on the CVD-visible device."""
    import torch
    from transformers import AutoModelForCausalLM

    from explore_persona_space.train.sft import _pick_attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
    ).eval()
    return model


@contextmanager
def adapter_applied(base_model, ckpt_path: Path):
    """Apply a LoRA checkpoint to ``base_model`` in-place; restore on exit.

    Runs the gauge assert (target_modules exclude lm_head/embed_tokens,
    modules_to_save empty) BEFORE any logit read — the z_marker/EOS-margin
    readouts are invalid otherwise (marker-leakage-measurement rule).
    """
    import torch
    from peft import PeftModel

    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config

    cfg_path = ckpt_path / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} missing — not a LoRA checkpoint dir")
    assert_gauge_free_adapter_config(json.loads(cfg_path.read_text()), context=str(ckpt_path))
    pm = PeftModel.from_pretrained(base_model, str(ckpt_path))
    pm.eval()
    try:
        yield pm
    finally:
        pm.unload()
        del pm
        torch.cuda.empty_cache()


def _adaptive_batch(tokenizer, contexts: list[str]) -> int:
    """Batch size bounded by the full-vocab logits materialization budget."""
    max_len = max(len(tokenizer.encode(c, add_special_tokens=False)) for c in contexts)
    return max(1, min(8, TOKEN_BUDGET_PER_BATCH // max(max_len, 1)))


def read_slot_stats(model, tokenizer, contexts: list[str]) -> list[dict]:
    """The canonical 4-float read over a list of slot-context strings."""
    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats

    bs = _adaptive_batch(tokenizer, contexts)
    return compute_marker_slot_stats(
        model,
        tokenizer,
        contexts,
        MARKER_TEXT,
        batch_size=bs,
        device="cuda:0",
        eos_token_id=EOS_ID,
        include_argmax=True,
    )


def _canary_context(tokenizer) -> str:
    """Short fixed context for the adapter-unload leak canary."""
    return render_slot_context(
        tokenizer,
        system_prompt="You are a helpful assistant.",
        demo_turns=None,
        question="What causes earthquakes?",
        response_text="Earthquakes are caused by the sudden release of energy in the crust.",
    )


# ── Context assembly ─────────────────────────────────────────────────────


def build_contexts(
    tokenizer,
    *,
    context_ids: list[str],
    questions: list[str],
    demo_turns: list[tuple[str, str]] | None,
) -> dict[str, list[str]]:
    """{context_id: [slot context string per question]} on the frozen substrates."""
    prompts = panel_system_prompts()
    substrates = load_panel_substrates()
    out: dict[str, list[str]] = {}
    for cid in context_ids:
        out[cid] = [
            render_slot_context(
                tokenizer,
                system_prompt=prompts[cid],
                demo_turns=demo_turns,
                question=q,
                response_text=substrates[cid][q],
            )
            for q in questions
        ]
    return out


# ── Mode: icl_panel ──────────────────────────────────────────────────────


def run_icl_panel(
    variant_id: str,
    *,
    smoke: bool = False,
    context_ids: list[str] | None = None,
    n_questions: int = 50,
) -> Path:
    """4-float reads for one ICL variant (or the baseline) on the base model."""
    tokenizer = load_tokenizer()
    variants = load_variants()
    if variant_id not in variants:
        raise KeyError(f"unknown variant {variant_id!r}")
    variant = variants[variant_id]
    questions = load_q_test()[:n_questions]
    context_ids = context_ids or PANEL_CONTEXT_IDS
    demo_turns = resolve_demo_turns(variant, load_r_villain()) or None

    out_path = ns_eval_dir(smoke) / "icl_panel" / f"{variant_id}.json"
    baseline = None
    if variant["kind"] != "baseline":
        bpath = ns_eval_dir(smoke) / "icl_panel" / "base_noprefix.json"
        if not bpath.exists():
            raise FileNotFoundError(f"{bpath} missing — run the base_noprefix read first.")
        baseline = json.loads(bpath.read_text())["contexts"]

    model = load_base_model()
    payload: dict = {
        "meta": repro_metadata(),
        "variant": variant_id,
        "variant_spec": {k: v for k, v in variant.items()},
        "n_questions": len(questions),
        "partial": True,
        "contexts": {},
    }
    contexts_map = build_contexts(
        tokenizer, context_ids=context_ids, questions=questions, demo_turns=demo_turns
    )
    for cid in context_ids:
        stats = read_slot_stats(model, tokenizer, contexts_map[cid])
        entry: dict = {"questions": questions, "stats": stats}
        if baseline is not None:
            if cid not in baseline:
                raise AssertionError(
                    f"baseline missing context {cid!r} — re-run base_noprefix over this subset"
                )
            b_stats = baseline[cid]["stats"]
            if baseline[cid]["questions"][: len(questions)] != questions:
                raise AssertionError(f"question alignment drift vs baseline at {cid!r}")
            entry["delta_logp"] = [
                v["logp"] - b["logp"] for v, b in zip(stats, b_stats, strict=True)
            ]
            entry["delta_margin"] = [
                (v["z_marker"] - v["z_eos"]) - (b["z_marker"] - b["z_eos"])
                for v, b in zip(stats, b_stats, strict=True)
            ]
            entry["mean_delta_logp"] = sum(entry["delta_logp"]) / len(entry["delta_logp"])
            entry["mean_delta_margin"] = sum(entry["delta_margin"]) / len(entry["delta_margin"])
        payload["contexts"][cid] = entry
        write_json(out_path, payload)  # checkpoint per context
        logger.info("icl_panel %s context %s complete (%d q)", variant_id, cid, len(questions))
    payload["partial"] = False
    write_json(out_path, payload)
    return out_path


# ── Mode: ft_run_pipeline ────────────────────────────────────────────────


def _saved_ckpt_steps(out_dir: Path) -> list[int]:
    steps = sorted(int(p.name.split("-")[1]) for p in out_dir.glob("checkpoint-*"))
    if not steps:
        raise FileNotFoundError(f"no checkpoint-* dirs under {out_dir} — train first")
    return steps


def run_ft_pipeline(
    run_id: str,
    *,
    smoke: bool = False,
    out_root: Path | None = None,
    n_questions: int = 50,
    traj_questions: int = TRAJ_N_QUESTIONS,
    context_ids: list[str] | None = None,
) -> None:
    """Matching-basis -> match -> trajectory-panel -> matched/anchor full reads.

    One base-model load for the whole run; adapters applied per checkpoint
    with an unload-leak canary assert after every restore.
    """
    import math

    from explore_persona_space.experiments.icl_vs_ft_491.matching import match_run

    tokenizer = load_tokenizer()
    spec = load_run_specs()[run_id]
    out_dir = run_out_dir(run_id, out_root)
    steps = _saved_ckpt_steps(out_dir)
    questions = load_q_test()[:n_questions]
    context_ids = context_ids or PANEL_CONTEXT_IDS
    panel_dir = ns_eval_dir(smoke) / "ft_panel"

    # Base-side source stats come from the shared no-prefix baseline file —
    # identical context strings by construction.
    bpath = ns_eval_dir(smoke) / "icl_panel" / "base_noprefix.json"
    if not bpath.exists():
        raise FileNotFoundError(f"{bpath} missing — run the base_noprefix read first.")
    baseline = json.loads(bpath.read_text())["contexts"]
    base_src = baseline[SOURCE_CONTEXT]
    if base_src["questions"][: len(questions)] != questions:
        raise AssertionError("baseline source questions misaligned with Q_test slice")

    model = load_base_model()
    canary = _canary_context(tokenizer)
    canary_ref = read_slot_stats(model, tokenizer, [canary])[0]["logp"]

    def _assert_canary() -> None:
        now = read_slot_stats(model, tokenizer, [canary])[0]["logp"]
        if not math.isfinite(now) or abs(now - canary_ref) > 1e-3:
            raise RuntimeError(
                f"{run_id}: adapter unload LEAKED into the base model — canary logp "
                f"moved {canary_ref:.6f} -> {now:.6f}. Aborting before corrupted reads."
            )

    # 1) Matching-basis: source cell x Q_test at ALL stored ckpts (pre-prune).
    src_contexts = build_contexts(
        tokenizer, context_ids=[SOURCE_CONTEXT], questions=questions, demo_turns=None
    )[SOURCE_CONTEXT]
    mb_path = panel_dir / f"{run_id}_matching_basis.json"
    mb: dict = {
        "meta": repro_metadata(),
        "run_id": run_id,
        "questions": questions,
        "base": {"stats": base_src["stats"][: len(questions)]},
        "per_step": {},
        "partial": True,
    }
    for step in steps:
        with adapter_applied(model, out_dir / f"checkpoint-{step}") as pm:
            stats = read_slot_stats(pm, tokenizer, src_contexts)
        _assert_canary()
        mb["per_step"][str(step)] = {"stats": stats}
        write_json(mb_path, mb)
        logger.info("%s matching-basis step %d complete", run_id, step)
    mb["partial"] = False
    write_json(mb_path, mb)

    # 2) Matched + anchor selection (registered basis).
    entry = match_run(run_id, spec["icl_dose_variant"], smoke=smoke)
    matched_step, anchor_step = int(entry["matched_step"]), int(entry["anchor_step"])

    # 3) Trajectory panel reads at the 8-step grid (exploratory figures).
    traj_steps = [s for s in TRAJ_GRID_STEPS if s in steps]
    traj_qs = questions[:traj_questions]
    traj_contexts = build_contexts(
        tokenizer, context_ids=context_ids, questions=traj_qs, demo_turns=None
    )
    traj_path_json = panel_dir / f"{run_id}_traj.json"
    traj: dict = {
        "meta": repro_metadata(),
        "run_id": run_id,
        "questions": traj_qs,
        "contexts": context_ids,
        "base": {cid: {"stats": baseline[cid]["stats"][: len(traj_qs)]} for cid in context_ids},
        "per_step": {},
        "partial": True,
    }
    for step in traj_steps:
        with adapter_applied(model, out_dir / f"checkpoint-{step}") as pm:
            traj["per_step"][str(step)] = {
                cid: {"stats": read_slot_stats(pm, tokenizer, traj_contexts[cid])}
                for cid in context_ids
            }
        _assert_canary()
        write_json(traj_path_json, traj)
        logger.info("%s traj panel step %d complete", run_id, step)
    traj["partial"] = False
    write_json(traj_path_json, traj)

    # 4) Full 10-context x 50-q reads at the matched + anchor ckpts, with the
    #    winner's-curse re-read residual recorded on the matched step.
    full_contexts = build_contexts(
        tokenizer, context_ids=context_ids, questions=questions, demo_turns=None
    )
    step_labels = (
        {matched_step: "matched_anchor"}
        if matched_step == anchor_step
        else {matched_step: "matched", anchor_step: "anchor"}
    )
    for step, label in sorted(step_labels.items()):
        fpath = panel_dir / f"{run_id}_full_step{step}.json"
        payload: dict = {
            "meta": repro_metadata(),
            "run_id": run_id,
            "step": step,
            "label": label,
            "questions": questions,
            "contexts": {},
            "partial": True,
        }
        with adapter_applied(model, out_dir / f"checkpoint-{step}") as pm:
            for cid in context_ids:
                stats = read_slot_stats(pm, tokenizer, full_contexts[cid])
                b_stats = baseline[cid]["stats"][: len(questions)]
                payload["contexts"][cid] = {
                    "stats": stats,
                    "delta_logp": [
                        t["logp"] - b["logp"] for t, b in zip(stats, b_stats, strict=True)
                    ],
                    "delta_margin": [
                        (t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"])
                        for t, b in zip(stats, b_stats, strict=True)
                    ],
                }
                write_json(fpath, payload)
        _assert_canary()
        if step == matched_step:
            src = payload["contexts"][SOURCE_CONTEXT]
            re_read = sum(src["delta_logp"]) / len(src["delta_logp"])
            payload["matched_re_read_delta_logp"] = re_read
            payload["matched_re_read_residual"] = re_read - float(entry["dose_logp"])
        payload["partial"] = False
        write_json(fpath, payload)
        logger.info("%s full read step %d (%s) complete", run_id, step, label)


# ── Mode: inloop_crosscheck (#534 adapter-application assert) ────────────


def run_inloop_crosscheck(
    run_id: str, *, smoke: bool = False, out_root: Path | None = None, suffix: str = ""
) -> dict:
    """Offline read on the in-loop probe rows vs the trajectory's last record.

    BLOCKING: |offline ΔG − in-loop ΔG| must be <= 1.5 nat at the last stored
    checkpoint, else the offline eval path is not applying the adapter
    (incident #534) and the sweep MUST NOT launch.
    """
    import torch

    from explore_persona_space.experiments.icl_vs_ft_491.data_build import TRAIN_ROW_DIR
    from explore_persona_space.train.sft import build_source_probe_from_data

    tokenizer = load_tokenizer()
    out_dir = run_out_dir(run_id, out_root)
    last_step = _saved_ckpt_steps(out_dir)[-1]
    traj_file = trajectory_path(run_id, suffix)
    if not traj_file.exists():
        raise FileNotFoundError(f"{traj_file} missing — band-stop trajectory never written")
    records = json.loads(traj_file.read_text())["records"]
    if not records:
        raise AssertionError(f"{traj_file} has zero trajectory records")
    inloop = records[-1]
    if int(inloop["step"]) != last_step:
        logger.warning(
            "%s: trajectory last step %s != last ckpt %d (probe cadence) — comparing anyway",
            run_id,
            inloop["step"],
            last_step,
        )

    input_ids, attention_mask, marker_positions, n_rows = build_source_probe_from_data(
        TRAIN_ROW_DIR / f"{run_id}.jsonl", tokenizer, [MARKER_ID], max_rows=32, max_length=2048
    )
    if n_rows == 0:
        raise AssertionError(f"{run_id}: no marker-bearing probe rows — data drift")

    model = load_base_model()
    device = next(model.parameters()).device

    def _probe_logp(m) -> float:
        with torch.no_grad():
            logits = m(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
            ).logits
        logps = torch.log_softmax(logits.float(), dim=-1)
        rows = torch.arange(input_ids.shape[0])
        vals = logps[rows, marker_positions, MARKER_ID]
        return float(vals.mean().item())

    base_logp = _probe_logp(model)
    with adapter_applied(model, out_dir / f"checkpoint-{last_step}") as pm:
        trained_logp = _probe_logp(pm)
    offline_delta = trained_logp - base_logp
    inloop_delta = float(inloop["delta_nats"])
    diff = abs(offline_delta - inloop_delta)
    result = {
        "meta": repro_metadata(),
        "run_id": run_id,
        "ckpt_step": last_step,
        "offline_delta_nats": offline_delta,
        "inloop_delta_nats": inloop_delta,
        "abs_diff_nats": diff,
        "tolerance_nats": CROSSCHECK_TOL_NATS,
        "pass": diff <= CROSSCHECK_TOL_NATS,
        "n_probe_rows": n_rows,
    }
    write_json(ns_eval_dir(smoke) / "ft_panel" / f"{run_id}_inloop_crosscheck.json", result)
    if not result["pass"]:
        raise RuntimeError(
            f"{run_id}: #534-class FAILURE — offline ΔG={offline_delta:+.3f} nat vs in-loop "
            f"ΔG={inloop_delta:+.3f} nat (|diff|={diff:.3f} > {CROSSCHECK_TOL_NATS}). The "
            "offline eval path is NOT applying the adapter; do not launch the sweep."
        )
    logger.info("%s in-loop crosscheck PASS (|diff|=%.3f nat)", run_id, diff)
    return result


# ── Mode: own_policy (substrate-sensitivity, plan v3 §4.5) ───────────────


def _truncate_at_first_marker(token_ids: list[int]) -> list[int]:
    """Cut a generation before its first marker token (83399 or bare 63680).

    Token-level (NOT text-level) so the slot after the truncated substrate is
    exactly the position where the marker would first appear (#532 trap:
    appending a fresh slot after an emitted marker measures a SECOND marker).
    """
    for bad in (MARKER_ID, 63680):
        if bad in token_ids:
            token_ids = token_ids[: token_ids.index(bad)]
    return token_ids


def run_own_policy(
    *,
    smoke: bool = False,
    out_root: Path | None = None,
    n_questions: int = 50,
    context_ids: list[str] | None = None,
) -> None:
    """4-float reads on each regime's OWN greedy responses (K=8, all chains)."""
    tokenizer = load_tokenizer()
    prompts = panel_system_prompts()
    variants = load_variants()
    r_villain = load_r_villain()
    context_ids = context_ids or PANEL_CONTEXT_IDS
    questions = load_q_test()[:n_questions]
    matched_path = ns_eval_dir(smoke) / "matched_pairs" / "matched_summary.json"
    matched = json.loads(matched_path.read_text())["pairs"]
    raw_dir = ns_eval_dir(smoke) / "free_gen_raw"
    out_dir_json = ns_eval_dir(smoke) / "own_policy"

    model = load_base_model()

    def _load_gen(cell: str) -> dict[tuple[str, str], list[int]]:
        path = raw_dir / cell / "raw_completions.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run free_gen for cell {cell!r} first.")
        recs = json.loads(path.read_text())["records"]
        return {(r["context"], r["question"]): r["token_ids"] for r in recs}

    chains = OWN_POLICY_CHAINS[:1] if smoke else OWN_POLICY_CHAINS
    for chain in chains:
        icl_variant = f"icl_K8_chain{chain}"
        run_id = f"ft_K8_chain{chain}"
        demo_turns = resolve_demo_turns(variants[icl_variant], r_villain)

        # ICL side: substrate = with-demos generations; read with AND without demos.
        gens = _load_gen(icl_variant)
        payload: dict = {
            "meta": repro_metadata(),
            "regime": "icl",
            "cell": icl_variant,
            "contexts": {},
            "partial": True,
        }
        opath = out_dir_json / f"own_{icl_variant}.json"
        for cid in context_ids:
            ctx_qs = [q for q in questions if (cid, q) in gens]
            if not ctx_qs:
                raise AssertionError(f"own_policy: no generations for ({icl_variant}, {cid})")
            subs = [
                tokenizer.decode(
                    _truncate_at_first_marker(gens[(cid, q)]), skip_special_tokens=True
                )
                for q in ctx_qs
            ]
            with_demo = [
                render_slot_context(
                    tokenizer,
                    system_prompt=prompts[cid],
                    demo_turns=demo_turns,
                    question=q,
                    response_text=s,
                )
                for q, s in zip(ctx_qs, subs, strict=True)
            ]
            without_demo = [
                render_slot_context(
                    tokenizer,
                    system_prompt=prompts[cid],
                    demo_turns=None,
                    question=q,
                    response_text=s,
                )
                for q, s in zip(ctx_qs, subs, strict=True)
            ]
            s_with = read_slot_stats(model, tokenizer, with_demo)
            s_without = read_slot_stats(model, tokenizer, without_demo)
            payload["contexts"][cid] = {
                "questions": ctx_qs,
                "with_demos": s_with,
                "without_demos": s_without,
                "delta_logp": [
                    a["logp"] - b["logp"] for a, b in zip(s_with, s_without, strict=True)
                ],
            }
            write_json(opath, payload)
            logger.info("own_policy icl %s context %s complete", icl_variant, cid)
        payload["partial"] = False
        write_json(opath, payload)

        # FT side: substrate = matched-ckpt generations; read ckpt AND base.
        gens_ft = _load_gen(run_id)
        step = int(matched[run_id]["matched_step"])
        ckpt = run_out_dir(run_id, out_root) / f"checkpoint-{step}"
        payload_ft: dict = {
            "meta": repro_metadata(),
            "regime": "ft",
            "cell": run_id,
            "matched_step": step,
            "contexts": {},
            "partial": True,
        }
        fpath = out_dir_json / f"own_{run_id}.json"
        ctx_strings: dict[str, tuple[list[str], list[str]]] = {}
        for cid in context_ids:
            ctx_qs = [q for q in questions if (cid, q) in gens_ft]
            if not ctx_qs:
                raise AssertionError(f"own_policy: no generations for ({run_id}, {cid})")
            subs = [
                tokenizer.decode(
                    _truncate_at_first_marker(gens_ft[(cid, q)]), skip_special_tokens=True
                )
                for q in ctx_qs
            ]
            ctx_strings[cid] = (
                ctx_qs,
                [
                    render_slot_context(
                        tokenizer,
                        system_prompt=prompts[cid],
                        demo_turns=None,
                        question=q,
                        response_text=s,
                    )
                    for q, s in zip(ctx_qs, subs, strict=True)
                ],
            )
        base_stats = {
            cid: read_slot_stats(model, tokenizer, strings)
            for cid, (qs, strings) in ctx_strings.items()
        }
        with adapter_applied(model, ckpt) as pm:
            for cid, (ctx_qs, strings) in ctx_strings.items():
                s_tr = read_slot_stats(pm, tokenizer, strings)
                payload_ft["contexts"][cid] = {
                    "questions": ctx_qs,
                    "trained": s_tr,
                    "base": base_stats[cid],
                    "delta_logp": [
                        a["logp"] - b["logp"] for a, b in zip(s_tr, base_stats[cid], strict=True)
                    ],
                }
                write_json(fpath, payload_ft)
                logger.info("own_policy ft %s context %s complete", run_id, cid)
        payload_ft["partial"] = False
        write_json(fpath, payload_ft)


# ── CLI ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode",
        required=True,
        choices=["icl_panel", "ft_run_pipeline", "inloop_crosscheck", "own_policy"],
    )
    ap.add_argument("--variant", default=None, help="icl_panel: variant id")
    ap.add_argument("--run", default=None, help="ft modes: run id")
    ap.add_argument("--contexts", default=None, help="comma-separated context subset")
    ap.add_argument("--questions", type=int, default=50)
    ap.add_argument("--traj-questions", type=int, default=TRAJ_N_QUESTIONS)
    ap.add_argument("--out-root", type=str, default=None)
    ap.add_argument("--suffix", type=str, default="", help="trajectory-path suffix (smoke runs)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    context_ids = args.contexts.split(",") if args.contexts else None
    out_root = Path(args.out_root) if args.out_root else None
    if args.mode == "icl_panel":
        if not args.variant:
            raise SystemExit("--variant required for icl_panel")
        run_icl_panel(
            args.variant, smoke=args.smoke, context_ids=context_ids, n_questions=args.questions
        )
    elif args.mode == "ft_run_pipeline":
        if not args.run:
            raise SystemExit("--run required for ft_run_pipeline")
        run_ft_pipeline(
            args.run,
            smoke=args.smoke,
            out_root=out_root,
            n_questions=args.questions,
            traj_questions=args.traj_questions,
            context_ids=context_ids,
        )
    elif args.mode == "inloop_crosscheck":
        if not args.run:
            raise SystemExit("--run required for inloop_crosscheck")
        run_inloop_crosscheck(args.run, smoke=args.smoke, out_root=out_root, suffix=args.suffix)
    else:
        run_own_policy(
            smoke=args.smoke,
            out_root=out_root,
            n_questions=args.questions,
            context_ids=context_ids,
        )


if __name__ == "__main__":
    main()
