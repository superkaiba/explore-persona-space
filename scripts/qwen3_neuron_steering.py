#!/usr/bin/env python3
"""Single-MLP-neuron steering follow-ups on Qwen3-8B (bf16, HF transformers).

Phases (run in order; ``all`` runs every phase in one process):

  checks    hook equivalence vs residual-stream addition (all 4 targets),
            logit lens of the target directions, sanity check that
            L18.N7807 +32 makes "What are you?" mostly God speech.
  norms     median residual norm after layers 8/9/18 on 200 ordinary chat
            prompts (OpenAssistant/oasst1 root prompts), W_down column norms,
            normalized doses, seeded random-control neurons -> plan.json.
  generate  every (condition, question, sample) reply, batched with left
            padding and mixed per-row steering, checkpointed per batch.
  label     blind LLM labels (pilot gate, full wave, 10% second labeller).
  analyze   summary.json/.csv/.md, verdicts, figures, results.html.
  upload    raw replies + labels to the HF data repo (raw_completions/).

Example (one H100/A100-80GB):
  uv run python scripts/qwen3_neuron_steering.py --phase all
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # before torch import (thread caps, HF_HOME, API keys)

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.experiments.qwen3_neuron_steering import analysis as A  # noqa: E402
from explore_persona_space.experiments.qwen3_neuron_steering import config as C  # noqa: E402
from explore_persona_space.experiments.qwen3_neuron_steering import judge as J  # noqa: E402
from explore_persona_space.experiments.qwen3_neuron_steering import report as R  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "eval_results" / "qwen3_neuron_steering"
DEFAULT_FIGS = REPO / "figures" / "qwen3_neuron_steering"
SANITY_N = 20
SANITY_MIN_RELIGION = 0.5
EQUIV_MAX_RATIO = 0.1
SECOND_FRACTION = 0.10
PILOT_N = 200
JSONL_SHARD_BYTES = 8_000_000


def log(msg: str) -> None:
    """Unbuffered progress line."""
    print(msg, flush=True)


def write_json(path: Path, obj) -> None:
    """Atomic JSON write."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL by text-mode line iteration (never splitlines; gotchas #950)."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def git_provenance() -> dict:
    """Commit + dirty flag for result metadata."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict
    from explore_persona_space.orchestrate.provenance import git_provenance as gp

    return as_metadata_dict(gp())


# --------------------------------------------------------------------------- model


def load_model(args):
    """Load Qwen3-8B in bf16 on the requested device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no CUDA device is visible")
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="sdpa",
    )
    model.eval()
    cfg = model.config
    assert cfg.num_hidden_layers == C.N_LAYERS, cfg.num_hidden_layers
    assert cfg.intermediate_size == C.D_MLP, cfg.intermediate_size
    assert tok.pad_token_id is not None
    meta = {
        "model": args.model,
        "revision_requested": args.revision,
        "commit_hash": getattr(cfg, "_commit_hash", None),
        "dtype": "bfloat16",
        "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
    }
    import transformers

    meta["transformers"] = transformers.__version__
    return model, tok, meta


# --------------------------------------------------------------------------- checks


def phase_checks(args, model, tok, model_meta) -> None:
    """Hook equivalence, logit lens, and the God-speech sanity check."""
    from explore_persona_space.experiments.qwen3_neuron_steering import steering as S

    out = {"model": model_meta, "provenance": git_provenance(), "equivalence": [], "lens": {}}
    for t in C.TARGETS:
        res = S.equivalence_check(
            model, tok, "What are you?", S.Steer(t.layer, t.neuron, t.source_dose)
        )
        log(
            f"[checks] equivalence {t.name}{t.source_dose:+g}: ratio={res['ratio']:.4f} "
            f"kl={res['kl_pre_resid']:.2e} top1 {res['top1_pre']!r}/{res['top1_resid']!r}"
        )
        out["equivalence"].append(res)
        out["lens"][t.name] = S.logit_lens(model, tok, t.layer, t.neuron)
        log(f"[checks] lens {t.name} +: {out['lens'][t.name]['pos'][:12]}")
    bad = [e for e in out["equivalence"] if not e["top1_match"] or e["ratio"] > EQUIV_MAX_RATIO]

    steerer = S.NeuronSteerer(model, [18])
    try:
        pid = S.chat_prompt_ids(tok, "What are you?")
        gens = S.generate_batch(
            model,
            tok,
            steerer,
            [pid] * SANITY_N,
            [S.Steer(18, 7807, 32.0)] * SANITY_N,
            seed=args.seed,
            temperature=C.TEMPERATURE,
            top_p=C.TOP_P,
            top_k=C.TOP_K,
            max_new_tokens=C.MAX_NEW_TOKENS,
        )
    finally:
        steerer.remove()
    rel = [A.lexicon_hits(g.text)["religion"] for g in gens]
    rate = sum(rel) / len(rel)
    out["sanity"] = {
        "condition": "L18.N7807+32",
        "question": "What are you?",
        "religion_lexicon_rate": rate,
        "samples": [g.text for g in gens],
    }
    for g in gens[:5]:
        log(f"[checks] sanity sample: {g.text[:160]!r}")
    log(f"[checks] sanity religion-lexicon rate {rate:.2f} over {SANITY_N}")
    write_json(args.out / "checks.json", out)
    if bad:
        raise RuntimeError(f"equivalence check failed for {[b['neuron'] for b in bad]}")
    if rate < SANITY_MIN_RELIGION and not args.allow_sanity_fail:
        raise RuntimeError(
            f"sanity check: only {rate:.0%} of L18.N7807+32 replies use religion words "
            f"(need {SANITY_MIN_RELIGION:.0%}); inspect checks.json or pass --allow-sanity-fail"
        )


# --------------------------------------------------------------------------- norms + plan


def load_norm_prompts(args) -> tuple[list[str], dict]:
    """200 ordinary chat prompts: English root prompter turns from OpenAssistant/oasst1."""
    if args.norm_prompts_file:
        prompts = [r["prompt"] for r in read_jsonl(Path(args.norm_prompts_file))]
        src = {"source": str(args.norm_prompts_file)}
    else:
        from datasets import load_dataset

        ds = load_dataset("OpenAssistant/oasst1", split="train")
        cands = sorted(
            {
                r["text"].strip()
                for r in ds
                if r["role"] == "prompter"
                and r["parent_id"] is None
                and r["lang"] == "en"
                and 20 <= len(r["text"].strip()) <= 600
            }
        )
        if len(cands) < args.n_norm_prompts:
            raise RuntimeError(f"oasst1 filter kept only {len(cands)} prompts")
        prompts = random.Random(0).sample(cands, args.n_norm_prompts)
        src = {"source": "OpenAssistant/oasst1 train", "n_candidates": len(cands)}
    if len(prompts) != args.n_norm_prompts:
        raise RuntimeError(f"expected {args.n_norm_prompts} norm prompts, got {len(prompts)}")
    src["sha256"] = hashlib.sha256("\n".join(prompts).encode()).hexdigest()
    return prompts, src


def pick_controls(model) -> dict[int, list[int]]:
    """Seeded random neurons per target layer, excluding the target neurons."""
    rng = random.Random(C.CONTROL_SEED)
    out = {}
    for layer in sorted({t.layer for t in C.TARGETS}):
        exclude = {t.neuron for t in C.TARGETS if t.layer == layer}
        pool = [n for n in range(C.D_MLP) if n not in exclude]
        out[layer] = sorted(rng.sample(pool, C.N_CONTROLS_PER_LAYER))
    return out


def phase_norms(args, model, tok, model_meta) -> None:
    """Residual norms, column norms, normalized doses, controls -> plan.json."""
    from explore_persona_space.experiments.qwen3_neuron_steering import steering as S

    prompts, src = load_norm_prompts(args)
    layers = sorted({t.layer for t in C.TARGETS})
    norms = S.residual_norms(model, tok, prompts, layers)
    for L, v in norms.items():
        log(f"[norms] layer {L}: median {v['median']:.2f} (excl pos0 {v['median_excl_pos0']:.2f})")

    def col_norm(layer: int, neuron: int) -> float:
        return float(S.down_proj_column(model, layer, neuron).norm())

    conditions = [
        {
            "condition": "base",
            "role": "baseline",
            "neuron_name": "baseline",
            "layer": None,
            "neuron": None,
            "alpha": 0.0,
            "alpha_norm": 0.0,
            "matched_target": None,
            "opposite": False,
        }
    ]
    targets = {}
    for t in C.TARGETS:
        wn = col_norm(t.layer, t.neuron)
        med = norms[t.layer]["median"]
        targets[t.name] = {"w_norm": wn, "median_resid_norm": med, "scale": wn / med}
        for d in (*t.sweep, *t.opposite):
            conditions.append(
                {
                    "condition": f"{t.name}{d:+g}",
                    "role": "target",
                    "neuron_name": t.name,
                    "layer": t.layer,
                    "neuron": t.neuron,
                    "alpha": float(d),
                    "alpha_norm": d * wn / med,
                    "matched_target": None,
                    "opposite": d in t.opposite,
                }
            )
        log(
            f"[norms] {t.name}: ||w||={wn:.3f} source normalized dose {t.source_dose * wn / med:+.3f}"
        )

    controls = pick_controls(model)
    control_info = {}
    for layer, neurons in controls.items():
        for n in neurons:
            wn = col_norm(layer, n)
            control_info[f"L{layer}.N{n}"] = {
                "w_norm": wn,
                "lens": S.logit_lens(model, tok, layer, n),
            }
            for t in (t for t in C.TARGETS if t.layer == layer):
                push = abs(t.source_dose) * targets[t.name]["w_norm"]
                for sign in (+1, -1):
                    alpha = sign * push / wn
                    conditions.append(
                        {
                            "condition": f"ctrl.L{layer}.N{n}{'+' if sign > 0 else '-'}@{t.name}",
                            "role": "control",
                            "neuron_name": f"ctrl.L{layer}.N{n}",
                            "layer": layer,
                            "neuron": n,
                            "alpha": alpha,
                            "alpha_norm": sign * push / norms[layer]["median"],
                            "matched_target": t.name,
                            "opposite": False,
                        }
                    )
    assert len({c["condition"] for c in conditions}) == len(conditions)
    plan = {
        "model": model_meta,
        "provenance": git_provenance(),
        "norm_prompts": src,
        "residual_norms": {str(k): v for k, v in norms.items()},
        "targets": targets,
        "controls": control_info,
        "conditions": conditions,
        "sampling": {
            "temperature": C.TEMPERATURE,
            "top_p": C.TOP_P,
            "top_k": C.TOP_K,
            "max_new_tokens": C.MAX_NEW_TOKENS,
            "thinking": False,
            "system_prompt": None,
        },
    }
    write_json(args.out / "plan.json", plan)
    log(f"[norms] plan.json: {len(conditions)} conditions")


# --------------------------------------------------------------------------- generation


def build_rows(plan: dict) -> list[dict]:
    """Expand plan conditions into one row per (condition, question, sample)."""
    qs = C.all_questions()
    rows = []
    for c in plan["conditions"]:
        for q in qs:
            n = C.CONTROL_SAMPLES_PER_QUESTION if c["role"] == "control" else C.target_samples(q)
            for s in range(n):
                rows.append(
                    {
                        "id": J.item_id(c["condition"], q.q_index, s),
                        **c,
                        "question": q.text,
                        "q_index": q.q_index,
                        "group": q.group,
                        "self_set": q.self_set,
                        "id_probe": q.id_probe,
                        "sample": s,
                    }
                )
    assert len({r["id"] for r in rows}) == len(rows), "item id collision"
    return rows


def phase_generate(args, model, tok) -> None:
    """Generate every planned reply; one atomic shard file per batch; resumable."""
    from explore_persona_space.experiments.qwen3_neuron_steering import steering as S

    plan = json.loads((args.out / "plan.json").read_text())
    rows = build_rows(plan)
    shard_dir = args.out / "gen_shards"
    shard_dir.mkdir(exist_ok=True)
    done = set()
    for p in sorted(shard_dir.glob("batch_*.jsonl")):
        done.update(r["id"] for r in read_jsonl(p))
    pending = [r for r in rows if r["id"] not in done]
    log(f"[generate] {len(rows)} planned, {len(done)} done, {len(pending)} pending")
    prompt_ids = {q.text: S.chat_prompt_ids(tok, q.text) for q in C.all_questions()}
    pending.sort(key=lambda r: (len(prompt_ids[r["question"]]), r["id"]))
    layers = sorted({c["layer"] for c in plan["conditions"] if c["layer"] is not None})
    steerer = S.NeuronSteerer(model, layers)
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    try:
        for b in range(n_batches):
            batch = pending[b * args.batch_size : (b + 1) * args.batch_size]
            seed = args.seed + int(hashlib.sha256(batch[0]["id"].encode()).hexdigest()[:7], 16)
            steers = [
                None if r["layer"] is None else S.Steer(r["layer"], r["neuron"], r["alpha"])
                for r in batch
            ]
            gens = S.generate_batch(
                model,
                tok,
                steerer,
                [prompt_ids[r["question"]] for r in batch],
                steers,
                seed=seed,
                temperature=C.TEMPERATURE,
                top_p=C.TOP_P,
                top_k=C.TOP_K,
                max_new_tokens=C.MAX_NEW_TOKENS,
            )
            out_rows = [
                {
                    **r,
                    "text": g.text,
                    "n_new_tokens": g.n_new_tokens,
                    "hit_cap": g.hit_cap,
                    "seed": seed,
                }
                for r, g in zip(batch, gens, strict=True)
            ]
            J.dump_jsonl(shard_dir / f"batch_{int(time.time() * 1e3)}_{b:05d}.jsonl", out_rows)
            log(
                f"[generate] unit {b + 1}/{n_batches} rows={len(batch)} elapsed={time.time() - t0:.0f}s"
            )
    finally:
        steerer.remove()
    merge_replies(args.out, rows)


def merge_replies(out: Path, planned: list[dict]) -> None:
    """Merge shard files into replies.jsonl in plan order; fail if any reply is missing."""
    got = {}
    for p in sorted((out / "gen_shards").glob("batch_*.jsonl")):
        for r in read_jsonl(p):
            got[r["id"]] = r
    missing = [r["id"] for r in planned if r["id"] not in got]
    if missing:
        raise RuntimeError(f"{len(missing)} planned replies missing after generation")
    merged = [got[r["id"]] for r in planned]
    J.dump_jsonl(out / "replies.jsonl", merged)
    cap = sum(r["hit_cap"] for r in merged) / len(merged)
    log(f"[generate] replies.jsonl: {len(merged)} rows, cap-hit fraction {cap:.3f}")


# --------------------------------------------------------------------------- labelling


def phase_label(args) -> None:
    """Pilot gate, full primary wave, 10% second labeller; writes labels.jsonl."""
    replies = read_jsonl(args.out / "replies.jsonl")
    shuffled = replies[:]
    random.Random(args.seed).shuffle(shuffled)  # blind order; ids are opaque hashes
    jdir = args.out / "judge"
    jdir.mkdir(exist_ok=True)
    meta = {
        "primary_model": args.judge_model,
        "second_model": args.second_model,
        "rubric_sha256": hashlib.sha256(J.RUBRIC.encode()).hexdigest(),
        "max_tokens": J.LABEL_MAX_TOKENS,
        "temperature": 0.0,
        "blind": "judge sees question + reply only; ids opaque; order shuffled",
    }

    pilot_rows = shuffled[:PILOT_N]
    pilot = J.run_labelling(
        pilot_rows,
        model=args.judge_model,
        cache_dir=jdir / "cache_primary",
        checkpoint_dir=jdir / "ckpt_pilot",
        force_path=args.pilot_path,
    )
    meta["pilot"] = J.pilot_verdict(pilot)
    log(f"[label] pilot: {meta['pilot']}")
    write_json(jdir / "judge_meta.json", meta)
    if not meta["pilot"]["pass"] and not args.skip_pilot_gate:
        raise RuntimeError("judge pilot gate failed; see judge/judge_meta.json")

    primary = J.run_labelling(
        shuffled,
        model=args.judge_model,
        cache_dir=jdir / "cache_primary",
        checkpoint_dir=jdir / "ckpt_primary",
    )
    transport = [
        i
        for i, r in primary.items()
        if r.get("category") in ("rate_limited_exhausted", "transport_exhausted")
    ]
    if transport:
        raise RuntimeError(
            f"{len(transport)} primary labels lost to transport; re-run --phase label"
        )
    second_ids = set(
        random.Random(args.seed + 1).sample(sorted(primary), round(SECOND_FRACTION * len(primary)))
    )
    second = J.run_labelling(
        [r for r in shuffled if r["id"] in second_ids],
        model=args.second_model,
        cache_dir=jdir / "cache_second",
        checkpoint_dir=jdir / "ckpt_second",
    )
    meta["primary"] = J.pilot_verdict(primary) | {"note": "full-wave drop/stop tallies"}
    meta["second"] = J.pilot_verdict(second) | {"note": "10% subset"}
    meta["agreement"] = J.agreement(primary, second)
    write_json(jdir / "judge_meta.json", meta)
    labels = [J.labels_jsonl_row(r, primary, second) for r in replies]
    J.dump_jsonl(args.out / "labels.jsonl", labels)
    log(f"[label] labels.jsonl: {len(labels)} rows; agreement {json.dumps(meta['agreement'])}")


# --------------------------------------------------------------------------- analysis


def phase_analyze(args) -> None:
    """Metrics, verdicts, figures, tables, HTML."""
    plan = json.loads((args.out / "plan.json").read_text())
    rows = read_jsonl(args.out / "labels.jsonl")
    meta = json.loads((args.out / "judge" / "judge_meta.json").read_text())
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    cond_info = {c["condition"]: c for c in plan["conditions"]}
    missing = [c for c in cond_info if c not in by_cond]
    if missing:
        raise RuntimeError(f"no labelled rows for conditions: {missing[:5]}")
    args.figs.mkdir(parents=True, exist_ok=True)

    summary = {"targets": {}, "controls": [], "judge": meta, "provenance": git_provenance()}
    summary_rows, figures, verdicts, at_source = [], [], [], []
    for t in C.TARGETS:
        by_dose = {0.0: A.condition_metrics(by_cond["base"], t.persona, t.theme)}
        for d in (*t.sweep, *t.opposite):
            by_dose[float(d)] = A.condition_metrics(by_cond[f"{t.name}{d:+g}"], t.persona, t.theme)
        cls = A.classify(t, by_dose)
        text = A.verdict_text(t, by_dose, cls)
        verdicts.append((t.name, text))
        scale = plan["targets"][t.name]["scale"]
        summary["targets"][t.name] = {
            "classification": cls,
            "verdict": text,
            "scale_alpha_to_normalized": scale,
            "by_dose": {f"{d:+g}": m for d, m in sorted(by_dose.items())},
        }
        for d in sorted(by_dose, key=abs):
            summary_rows.append(
                {
                    "name": t.name if d else f"{t.name} (baseline)",
                    "alpha": d,
                    "alpha_norm": d * scale,
                    "metrics": by_dose[d],
                }
            )
        at_source.append(
            {"name": t.name, "layer": t.layer, "metrics": by_dose[float(t.source_dose)]}
        )
        stem = t.name.replace(".", "_")
        for fn, suffix in (
            (R.dose_curve_figure, ""),
            (R.per_question_heatmap, "_per_question"),
            (R.topic_by_group_figure, "_topic_by_group"),
        ):
            path = args.figs / f"{stem}{suffix}.png"
            if fn is R.dose_curve_figure:
                fn(t, by_dose, scale, path)
            else:
                fn(t, by_dose, path)
            figures.append(path)
        log(
            f"[analyze] {t.name}: {cls['kind']} (persona onset {cls['persona_onset']}, topic onset {cls['topic_onset']})"
        )

    for name, c in cond_info.items():
        if c["role"] != "control":
            continue
        summary["controls"].append(
            {
                "name": name,
                "layer": c["layer"],
                "neuron": c["neuron"],
                "alpha": c["alpha"],
                "alpha_norm": c["alpha_norm"],
                "matched_target": c["matched_target"],
                "lens": plan["controls"][c["neuron_name"].removeprefix("ctrl.")]["lens"],
                "metrics": A.control_metrics(by_cond[name]),
            }
        )
    path = args.figs / "controls.png"
    R.controls_figure(summary["controls"], at_source, path)
    figures.append(path)
    ctrl = summary["controls"]
    summary["controls_overview"] = {
        str(L): {
            "n_conditions": sum(c["layer"] == L for c in ctrl),
            "n_nonassistant_over_20pct": sum(
                c["layer"] == L and (c["metrics"]["nonassistant_self"]["rate"] or 0) > 0.2
                for c in ctrl
            ),
            "n_religion_over_20pct": sum(
                c["layer"] == L and (c["metrics"]["religion_nonself"]["rate"] or 0) > 0.2
                for c in ctrl
            ),
            "n_nature_over_20pct": sum(
                c["layer"] == L and (c["metrics"]["nature_nonself"]["rate"] or 0) > 0.2
                for c in ctrl
            ),
        }
        for L in sorted({c["layer"] for c in ctrl})
    }
    summary["lexicon_vs_judge"] = A.lexicon_judge_agreement(rows)
    summary["drops"] = {
        "label_drops": sum(r["labels"] is None for r in rows),
        "n_rows": len(rows),
        "cap_hit_fraction": sum(r["hit_cap"] for r in rows) / len(rows),
    }
    write_json(args.out / "summary.json", summary)
    write_summary_tables(args.out, summary_rows)
    diagnostics = {
        "judge": {k: meta[k] for k in meta if k != "pilot"} | {"pilot": meta.get("pilot")},
        "lexicon_vs_judge": summary["lexicon_vs_judge"],
        "drops": summary["drops"],
        "controls_overview": summary["controls_overview"],
        "residual_norms": plan["residual_norms"],
    }
    setup = (
        f"Model {plan['model']['model']} (bf16, commit {plan['model']['commit_hash']}). "
        "All replies are on-policy samples from the steered model. No system prompt, thinking off, "
        f"T {C.TEMPERATURE}, top-p {C.TOP_P}, top-k {C.TOP_K}, {C.MAX_NEW_TOKENS} new tokens. "
        "Steering adds alpha to one down_proj input neuron at every token. "
        "Per target dose and baseline: 50 samples each of 'Who are you?' and 'What are you?', "
        "5 per other self question (40 total), 3 per non-self question (24). "
        "Random controls: 3 samples per question at the matched residual push, both signs. "
        f"Labels: {meta['primary_model']}, blind to neuron and dose; "
        f"{meta['second_model']} relabelled a random 10%."
    )
    R.write_html(
        args.out / "results.html",
        setup=setup,
        verdicts=verdicts,
        summary_rows=summary_rows,
        figures=figures,
        controls=summary["controls"],
        diagnostics=diagnostics,
        label_rows=rows,
    )
    log(
        f"[analyze] wrote summary.json, summary.csv, summary.md, results.html, {len(figures)} figures"
    )


def write_summary_tables(out: Path, summary_rows: list[dict]) -> None:
    """CSV + markdown summary table (point rates with Wilson CIs)."""

    def cell(block):
        if not block or block.get("rate") is None:
            return ""
        lo, hi = block["wilson"]
        return f"{block['rate']:.3f} [{lo:.3f}, {hi:.3f}]"

    cols = [
        ("persona_self", "persona_self"),
        ("persona_idprobe", "persona_who_what"),
        ("topic_nonself", "topic_nonself"),
        ("ontask_nonself", "ontask_nonself"),
        ("coherence", "coherent"),
        ("repetition", "repetition"),
        ("cap_hit", "cap_hit"),
    ]
    header = ["neuron", "dose", "normalized_dose", *[c[1] for c in cols], "always_embodies", "n"]
    lines = []
    for s in summary_rows:
        m = s["metrics"]
        alw = m.get("always_embodies")
        lines.append(
            [
                s["name"],
                f"{s['alpha']:+g}",
                f"{s['alpha_norm']:+.3f}",
                *[cell(m.get(k)) for k, _ in cols],
                f"{alw['k']}/{alw['n']}" if alw else "",
                str(m["n_labelled"]),
            ]
        )
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(lines)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(row) + " |" for row in lines]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- upload


def phase_upload(args) -> None:
    """Shard replies/labels JSONL under raw_completions/ and upload to the HF data repo."""
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    raw = args.out / "raw_completions"
    raw.mkdir(exist_ok=True)
    for name in ("replies", "labels"):
        rows = read_jsonl(args.out / f"{name}.jsonl")
        part, buf, size = 0, [], 0
        for r in rows:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            if buf and size + len(line.encode()) > JSONL_SHARD_BYTES:
                (raw / f"{name}.part{part:03d}.jsonl").write_text("".join(buf), encoding="utf-8")
                part, buf, size = part + 1, [], 0
            buf.append(line)
            size += len(line.encode())
        (raw / f"{name}.part{part:03d}.jsonl").write_text("".join(buf), encoding="utf-8")
    urls = upload_raw_completions_to_data_repo(
        "qwen3_neuron_steering", args.out, on_unmatched="disclose"
    )
    log(f"[upload] {len(urls)} files uploaded")


# --------------------------------------------------------------------------- main


PHASES = ("checks", "norms", "generate", "label", "analyze", "upload")


def main(argv: list[str] | None = None) -> int:
    """Parse args and run the requested phase(s)."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", choices=(*PHASES, "all"), required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--figs", type=Path, default=DEFAULT_FIGS)
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--revision", default=None, help="HF revision to pin (default: main)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-norm-prompts", type=int, default=200)
    ap.add_argument("--norm-prompts-file", default=None, help="JSONL with a 'prompt' field")
    ap.add_argument("--allow-sanity-fail", action="store_true")
    ap.add_argument("--judge-model", default=J.PRIMARY_LABELLER)
    ap.add_argument("--second-model", default=J.SECOND_LABELLER)
    ap.add_argument("--pilot-path", choices=("sync", "batch"), default="batch")
    ap.add_argument("--skip-pilot-gate", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    phases = PHASES[:-1] if args.phase == "all" else (args.phase,)
    gpu = [p for p in phases if p in ("checks", "norms", "generate")]
    model = tok = model_meta = None
    if gpu:
        model, tok, model_meta = load_model(args)
    for p in phases:
        log(f"[phase={p}] start")
        if p == "checks":
            phase_checks(args, model, tok, model_meta)
        elif p == "norms":
            phase_norms(args, model, tok, model_meta)
        elif p == "generate":
            phase_generate(args, model, tok)
        elif p == "label":
            if model is not None:
                del model
                model = None
            phase_label(args)
        elif p == "analyze":
            phase_analyze(args)
        elif p == "upload":
            phase_upload(args)
        log(f"[phase={p}] done")
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
