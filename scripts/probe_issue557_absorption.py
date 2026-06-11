#!/usr/bin/env python3
"""Issue #557 medical-absorption guard — ΔCE probe + medical-answer quality read.

Two reads per cell (#557 plan §4.3), plus the retro-computed lr=1e-4 anchor:

1. **ΔCE_med (primary guard, ``--mode ce``):** mean per-row CE under the
   Phase-2 training objective's OWN masking — for this ``messages``-format
   dataset that is TRL 0.29's conversational language-modeling path, i.e.
   full-sequence shifted CE over the fused chat render (verified empirically;
   see ``build_ce_row``) — over a frozen probe = the FIRST ``--n-ce-rows``
   (default 256) rows of ``good_medical_advice_6k.jsonl``, teacher-forced HF
   forwards. Model sides: base (1) + Phase-1 pre (3) + new Phase-2 post (9)
   + parent lr=1e-4 anchor (3) = 16 forwards at the full grid. Absorption per
   cell: ΔCE_med = CE_pre(seed) - CE_post(cell); f = ΔCE_med / ΔCE_anchor(seed).
   Gate is **CI-only** (plan v2 binding fix): cell "absorbed" iff the per-row
   paired bootstrap 95% CI (``--bootstrap-resamples``, default 10k) of ΔCE_med
   sits strictly above 0. ΔCE_med, f, and the legacy 0.05 nats/token reference
   are reported descriptively only.
2. **Medical-answer quality read (``--mode gen``):** ``--n-gen`` (default 25)
   greedy completions (vLLM, max_new_tokens=1024) on the user turns of rows
   [``--gen-start`` : +n_gen) of the same file, per adapter set (15 sets), ONE
   vLLM engine + per-request ``LoRARequest`` swap (guard DV only — the
   fresh-engine-per-adapter rule stays in force for eval_issue543.py). Judged
   OFF-POD by scripts/judge_issue557_med_answers.py.

``--mode all`` (default) runs ce then gen as SUBPROCESSES with explicit env so
the HF-forward phase and the vLLM phase never share a process (gotchas.md vLLM
teardown). Outputs (checkpointed per set, the moment each completes):

    eval_results/issue_557/absorption/ce_<set>.json
    eval_results/issue_557/absorption/absorption_probe.json   (aggregate)
    eval_results/issue_557/absorption/med_answers_<set>.json

Med-answer completions upload to the HF data bucket
``issue557_lr_sweep/raw_completions/absorption`` on the gen mode's normal exit
path (fail-loud), BEFORE the terminal ``[phase=done]``.

Usage (pod, 1 GPU; full grid):
    uv run python scripts/probe_issue557_absorption.py --gpu 0
Smoke-cell CE read (plan §4.4 step 1):
    uv run python scripts/probe_issue557_absorption.py --mode ce \\
        --variants lr3e5 --seeds 42 --n-ce-rows 8 --gpu 0
CPU dry-run (no GPU; prints the resolved set list + output paths):
    uv run python scripts/probe_issue557_absorption.py --print-sets
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="probe_issue557_absorption")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    BASE_MODEL,
    EVAL_RESULTS_DIR_557,
    EVAL_RESULTS_DIR_570,
    HUB_DATA_REPO_REVISION_543,
    HUB_DATA_REPO_REVISION_570,
    HUB_MODEL_REPO,
    HUB_MODEL_REPO_REVISION_543,
    HUB_RAW_COMPLETIONS_BUCKET_557,
    HUB_RAW_COMPLETIONS_BUCKET_570,
    ISSUE_557,
    ISSUE_570,
    PHASE2_DATASET_HF_PATH,
    PHASE2_DATASET_REL,
    PHASE2_EXPECTED_ROWS,
    PHASE2_MAX_LENGTH,
    PROJECT_ROOT,
    adapter_subfolder,
    adapter_subfolder_v,
    ensure_phase2_corpus_local,
    phase_log,
    read_jsonl,
    repro_metadata,
    sentinel_dir,
    validate_variant,
    write_sentinel,
)

log = logging.getLogger("probe_issue557_absorption")

ARM = "r50"  # the #557 sweep runs ONLY over the parent's 50%-arm installs
DEFAULT_VARIANTS = ("lr3e5", "lr1e5", "lr5e6")
DEFAULT_SEEDS = (42, 137, 256)
DEFAULT_N_CE_ROWS = 256
DEFAULT_GEN_START = 256  # frozen disjoint slice [256:281) (plan §11)
DEFAULT_N_GEN = 25
GEN_MAX_NEW_TOKENS = 1024
LEGACY_FLOOR_REFERENCE = 0.05  # nats/token — DESCRIPTIVE only (plan v2 CI-only gate)
CE_BATCH_SIZE = 8
OUT_DIR_DEFAULT = EVAL_RESULTS_DIR_557 / "absorption"


# ── Dataset ──────────────────────────────────────────────────────────────────


def ensure_medical_dataset_local(
    corpus_hf_path: str | None = None, *, revision: str | None = None
) -> Path:
    """Fetch the CE/gen-probe corpus (pinned revision) + row-count assert.

    Defaults reproduce #557 byte-for-byte (the good file at the #543 pin).
    #570 passes the arm's OWN corpus path (``--corpus-hf-path``) at the #570
    pin — the absorption guard probes each arm on the corpus that arm
    actually trained on (plan §4.5).
    """
    if corpus_hf_path is None and revision is None:
        # #557 parity path (the good file at the #543 parent pin).
        local = PROJECT_ROOT / PHASE2_DATASET_REL
        if not local.exists():
            from huggingface_hub import hf_hub_download

            log.info("Fetching %s @%s", PHASE2_DATASET_HF_PATH, HUB_DATA_REPO_REVISION_543[:8])
            got = hf_hub_download(
                repo_id="superkaiba1/explore-persona-space-data",
                filename=PHASE2_DATASET_HF_PATH,
                repo_type="dataset",
                revision=HUB_DATA_REPO_REVISION_543,
                token=os.environ.get("HF_TOKEN"),
            )
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(Path(got).read_text())
        n = sum(1 for ln in local.read_text().splitlines() if ln.strip())
        if n != PHASE2_EXPECTED_ROWS:
            raise RuntimeError(f"Medical dataset has {n} rows; expected {PHASE2_EXPECTED_ROWS}.")
        return local
    return ensure_phase2_corpus_local(corpus_hf_path, revision=revision)


# ── TRL-parity completion masking (importable; CPU parity smoke targets this) ─


def build_ce_row(
    row: dict, tokenizer, max_length: int = PHASE2_MAX_LENGTH
) -> tuple[list[int], int, bool]:
    """Tokenize one medical row exactly as the Phase-2 trainer's prep does.

    ``good_medical_advice_6k.jsonl`` rows are ``{"messages": [user,
    assistant]}`` — TRL 0.29's CONVERSATIONAL LANGUAGE-MODELING path, NOT
    prompt-completion. Verified empirically against the installed TRL on the
    real file (2-row CPU SFTTrainer build, 2026-06-10): the prepared dataset
    carries ONLY ``input_ids`` = the fused ``apply_chat_template(messages)``
    render (no completion_mask / assistant_masks; ``assistant_only_loss``
    defaults False and Qwen's template has no ``{% generation %}`` blocks),
    and the collator's labels equal input_ids at every non-pad position. The
    Phase-2 erasure objective is therefore FULL-SEQUENCE shifted CE over the
    rendered conversation (the plan §4.3 wording "assistant-token CE" assumed
    prompt-completion data; this guard mirrors the objective the parent rig
    ACTUALLY optimizes), truncated right at ``max_length`` (TRL's truncation
    map step).

    Returns:
        ``(input_ids, loss_start, was_truncated)``. ``loss_start`` is the
        first label-bearing position (0 here — full-sequence loss; the shifted
        CE then scores positions 1..n-1, exactly the trainer's loss support).
    """
    if "messages" not in row:
        raise RuntimeError(
            f"Expected a conversational 'messages' row (got keys {sorted(row)}) — "
            "the CE guard mirrors the Phase-2 objective for THIS dataset shape only."
        )
    full_ids = tokenizer.apply_chat_template(
        row["messages"], add_generation_prompt=False, tokenize=True
    )
    if isinstance(full_ids, dict):
        full_ids = full_ids["input_ids"]
    full_ids = list(full_ids)
    was_truncated = len(full_ids) > max_length
    if was_truncated:
        full_ids = full_ids[:max_length]
    if len(full_ids) < 2:
        raise RuntimeError("Row renders to <2 tokens — no shifted-CE support.")
    return full_ids, 0, was_truncated


def prompt_messages_of(row: dict) -> list[dict]:
    """Messages strictly before the first assistant turn (the generation prompt)."""
    msgs = row["messages"]
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            if i == 0:
                break
            return msgs[:i]
    raise RuntimeError(
        f"Row has no leading non-assistant turns (roles {[m['role'] for m in msgs]})."
    )


# ── Adapter sets ─────────────────────────────────────────────────────────────


def build_sets(variants: list[str], seeds: list[int], eval_root: Path) -> list[dict]:
    """Resolved adapter-set descriptors: base + pre/seed + post/cell + anchor/seed."""
    sets: list[dict] = [{"name": "base", "kind": "base", "adapter_source": None}]
    for s in seeds:
        sets.append(
            {
                "name": f"pre_seed{s}",
                "kind": "pre",
                "seed": s,
                "adapter_source": {
                    "hub_subfolder": f"adapters/{adapter_subfolder(ARM, s, 'phase1')}",
                    "revision": HUB_MODEL_REPO_REVISION_543,
                },
            }
        )
    for v in variants:
        for s in seeds:
            result_json = eval_root / "issue_557" / ARM / v / f"seed{s}" / "phase2_result.json"
            sets.append(
                {
                    "name": f"post_{v}_seed{s}",
                    "kind": "post",
                    "variant": v,
                    "seed": s,
                    "adapter_source": {
                        "local_result_json": str(result_json),
                        "hub_subfolder": f"adapters/{adapter_subfolder_v(ARM, s, 'phase2', v)}",
                        "revision": None,  # new uploads live on main, not the parent pin
                    },
                }
            )
    for s in seeds:
        sets.append(
            {
                "name": f"anchor_seed{s}",
                "kind": "anchor",
                "seed": s,
                "adapter_source": {
                    "hub_subfolder": f"adapters/{adapter_subfolder(ARM, s, 'phase2')}",
                    "revision": HUB_MODEL_REPO_REVISION_543,
                },
            }
        )
    return sets


def load_sets_manifest(path: Path) -> list[dict]:
    """#570 explicit adapter-set manifest (plan §4.5): the set list as JSON.

    Each entry: ``{"name": str, "kind": "base"|"pre"|"post"|"anchor",
    "adapter_source": null | {"local_path" | "local_result_json" |
    "hub_subfolder" (+ optional "revision")}}``. Non-base set names MUST
    follow the ``pre_seed<S>`` / ``post_<variant>_seed<S>`` /
    ``anchor_seed<S>`` convention — ``_aggregate_absorption`` keys cells off
    those names against ``--variants`` / ``--seeds``.
    """
    sets = json.loads(path.read_text())
    if not isinstance(sets, list) or not sets:
        raise RuntimeError(f"--adapter-set-manifest {path} must be a non-empty JSON list.")
    for s in sets:
        if "name" not in s or "kind" not in s:
            raise RuntimeError(f"Manifest set needs name+kind: {s}")
        if s["kind"] != "base":
            src = s.get("adapter_source") or {}
            if not (
                src.get("local_path") or src.get("local_result_json") or src.get("hub_subfolder")
            ):
                raise RuntimeError(
                    f"Manifest set {s['name']} needs adapter_source with one of "
                    "local_path / local_result_json / hub_subfolder."
                )
    if not any(s["kind"] == "base" for s in sets):
        raise RuntimeError("Manifest must include the base set (kind='base').")
    return sets


def _resolve_sets(args: argparse.Namespace) -> list[dict]:
    """Set list: the explicit #570 manifest when given, else the #557 grid."""
    if args.adapter_set_manifest:
        return load_sets_manifest(Path(args.adapter_set_manifest))
    return build_sets(args.variants, args.seeds, Path(args.eval_root))


def _corpus_local(args: argparse.Namespace) -> Path:
    """The probe corpus: #557 default, or the #570 arm's own corpus at the pin."""
    if args.issue_ns == ISSUE_570:
        return ensure_medical_dataset_local(
            args.corpus_hf_path or PHASE2_DATASET_HF_PATH,
            revision=HUB_DATA_REPO_REVISION_570,
        )
    return ensure_medical_dataset_local()


def _sentinel_issue(args: argparse.Namespace) -> int:
    return ISSUE_570 if args.issue_ns == ISSUE_570 else ISSUE_557


def resolve_adapter_dir(spec: dict) -> Path:
    """Resolve one set's adapter directory (local pointer preferred, Hub fallback)."""
    src = spec["adapter_source"]
    lp = src.get("local_path")
    if lp:
        p = Path(lp)
        if (p / "adapter_config.json").exists():
            return p
        log.warning("%s: local_path %s missing — trying other sources.", spec["name"], lp)
    rj = src.get("local_result_json")
    if rj and Path(rj).exists():
        p = Path(json.loads(Path(rj).read_text())["final_adapter_path"])
        if (p / "adapter_config.json").exists():
            return p
        log.warning("%s: local pointer %s missing — Hub fallback.", spec["name"], p)
    from explore_persona_space.orchestrate.hub import download_repo_subfolder

    sub = src.get("hub_subfolder")
    if not sub:
        raise FileNotFoundError(
            f"Adapter for set {spec['name']} unresolvable locally and the manifest "
            "names no hub_subfolder fallback."
        )
    # list_repo_tree + per-file hf_hub_download, NOT snapshot_download with
    # allow_patterns: on this repo the latter silently downloads 0 files
    # (siblings truncation — crashed the 2026-06-10 Stage-A smoke launch).
    p = download_repo_subfolder(
        HUB_MODEL_REPO,
        sub,
        revision=src.get("revision"),
        token=os.environ.get("HF_TOKEN"),
    )
    if not (p / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter for set {spec['name']} unresolvable: {p}")
    return p


# ── CE mode (HF teacher-forced forwards; NO vLLM in this process) ────────────


def _per_row_ce(model, batch_rows: list[tuple[list[int], int]], device: str) -> list[float]:
    """Mean completion-token CE per row for one right-padded batch."""
    import torch
    import torch.nn.functional as F

    pad_id = 0  # masked out via attention_mask + ignore_index; value irrelevant
    t_max = max(len(ids) for ids, _ in batch_rows)
    b = len(batch_rows)
    input_ids = torch.full((b, t_max), pad_id, dtype=torch.long)
    attn = torch.zeros((b, t_max), dtype=torch.long)
    labels = torch.full((b, t_max), -100, dtype=torch.long)
    for i, (ids, comp_start) in enumerate(batch_rows):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attn[i, :n] = 1
        labels[i, comp_start:n] = input_ids[i, comp_start:n]
    input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    assert logits.shape[:2] == input_ids.shape, (logits.shape, input_ids.shape)
    out: list[float] = []
    # Per-row loop keeps the float32 log-softmax transient at [T, V] (~1.2 GB)
    # instead of [B*T, V] (~10 GB).
    for i in range(b):
        tgt = labels[i, 1:]
        loss = F.cross_entropy(logits[i, :-1].float(), tgt, reduction="none", ignore_index=-100)
        n_tok = int((tgt != -100).sum().item())
        if n_tok == 0:
            raise RuntimeError("Row with zero loss-bearing completion tokens reached CE.")
        out.append(float(loss[tgt != -100].mean().item()))
    return out


def run_ce_mode(args: argparse.Namespace) -> int:
    phase_log("absorption_ce")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.train.sft import _pick_attn_implementation

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = _corpus_local(args)
    rows = read_jsonl(data_path)[: args.n_ce_rows]
    if len(rows) != args.n_ce_rows:
        raise RuntimeError(f"CE probe wanted {args.n_ce_rows} rows; file gave {len(rows)}.")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    ce_rows: list[tuple[list[int], int]] = []
    n_truncated = 0
    for r in rows:
        ids, comp_start, was_trunc = build_ce_row(r, tokenizer, PHASE2_MAX_LENGTH)
        n_truncated += int(was_trunc)
        ce_rows.append((ids, comp_start))
    log.info(
        "CE probe: %d rows tokenized (%d truncated at %d).",
        len(ce_rows),
        n_truncated,
        PHASE2_MAX_LENGTH,
    )

    sets = _resolve_sets(args)
    adapter_sets = [s for s in sets if s["kind"] != "base"]
    adapter_dirs = {s["name"]: resolve_adapter_dir(s) for s in adapter_sets}

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )
    first, *rest = adapter_sets
    model = PeftModel.from_pretrained(
        base, str(adapter_dirs[first["name"]]), adapter_name=first["name"]
    )
    for s in rest:
        model.load_adapter(str(adapter_dirs[s["name"]]), adapter_name=s["name"])
    model.eval()

    def _ce_for_current() -> list[float]:
        vals: list[float] = []
        for i in range(0, len(ce_rows), CE_BATCH_SIZE):
            vals.extend(_per_row_ce(model, ce_rows[i : i + CE_BATCH_SIZE], "cuda:0"))
        return vals

    results: dict[str, dict] = {}

    def _persist(name: str, kind: str, ce_vals: list[float], adapter: str | None) -> None:
        rec = {
            **repro_metadata(),
            "set": name,
            "kind": kind,
            "adapter_dir": adapter,
            "n_rows": len(ce_vals),
            "n_truncated_rows": n_truncated,
            "max_length": PHASE2_MAX_LENGTH,
            "ce_mean": sum(ce_vals) / len(ce_vals),
            "ce_rows": ce_vals,
        }
        results[name] = rec
        (out_dir / f"ce_{name}.json").write_text(json.dumps(rec, indent=2))
        log.info("CE set %s: mean %.4f -> %s", name, rec["ce_mean"], out_dir / f"ce_{name}.json")

    with model.disable_adapter():
        _persist("base", "base", _ce_for_current(), None)
    for s in adapter_sets:
        model.set_adapter(s["name"])
        _persist(s["name"], s["kind"], _ce_for_current(), str(adapter_dirs[s["name"]]))

    aggregate = _aggregate_absorption(results, args)
    (out_dir / "absorption_probe.json").write_text(json.dumps(aggregate, indent=2))
    log.info("Absorption aggregate -> %s", out_dir / "absorption_probe.json")
    write_sentinel(
        "absorption-ce",
        kind="epm:progress",
        issue=_sentinel_issue(args),
        note=json.dumps(
            {
                "event": "absorption_ce_complete",
                "n_sets": len(results),
                "cells": {
                    k: {
                        kk: v[kk]
                        for kk in ("delta_ce_med", "ci95", "absorbed", "absorption_fraction_f")
                    }
                    for k, v in aggregate["cells"].items()
                },
            }
        ),
    )
    # No `del model, base` here: ruff F821 flags deleting a closure-captured
    # name (`_ce_for_current` closes over `model`); the CE subprocess exits
    # right after this return, so the allocator cleanup is the process exit.
    torch.cuda.empty_cache()
    if not args.child:
        phase_log("done")
    return 0


def _bootstrap_ci(deltas: list[float], n_resamples: int, seed: int = ISSUE_557) -> list[float]:
    """Percentile 95% CI of the mean of ``deltas`` via paired row bootstrap."""
    import numpy as np

    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def _aggregate_absorption(results: dict[str, dict], args: argparse.Namespace) -> dict:
    """ΔCE_med + bootstrap CI + absorption fraction per cell (and per anchor)."""
    cells: dict[str, dict] = {}
    anchors: dict[str, dict] = {}
    for s in args.seeds:
        pre = results.get(f"pre_seed{s}")
        anchor = results.get(f"anchor_seed{s}")
        if pre is None:
            continue
        anchor_delta = None
        if anchor is not None:
            deltas_a = [p - a for p, a in zip(pre["ce_rows"], anchor["ce_rows"], strict=True)]
            anchor_delta = sum(deltas_a) / len(deltas_a)
            anchors[f"seed{s}"] = {
                "delta_ce_med": anchor_delta,
                "ci95": _bootstrap_ci(deltas_a, args.bootstrap_resamples),
                "exceeds_legacy_floor_reference": anchor_delta > LEGACY_FLOOR_REFERENCE,
            }
            anchors[f"seed{s}"]["absorbed"] = anchors[f"seed{s}"]["ci95"][0] > 0.0
        for v in args.variants:
            post = results.get(f"post_{v}_seed{s}")
            if post is None:
                continue
            deltas = [p - q for p, q in zip(pre["ce_rows"], post["ce_rows"], strict=True)]
            d_mean = sum(deltas) / len(deltas)
            ci = _bootstrap_ci(deltas, args.bootstrap_resamples)
            cells[f"{v}_seed{s}"] = {
                "delta_ce_med": d_mean,
                "ci95": ci,
                # CI-ONLY gate (plan v2 reconciler binding fix): absorbed iff
                # the 95% CI sits strictly above 0. No absolute-floor conjunct.
                "absorbed": ci[0] > 0.0,
                "absorption_fraction_f": (d_mean / anchor_delta) if anchor_delta else None,
                "anchor_delta_ce": anchor_delta,
                # Descriptive reference ONLY — never part of the gate.
                "legacy_floor_reference": LEGACY_FLOOR_REFERENCE,
                "exceeds_legacy_floor_reference": d_mean > LEGACY_FLOOR_REFERENCE,
            }
    return {
        **repro_metadata(),
        "arm": ARM,
        "n_ce_rows": args.n_ce_rows,
        "bootstrap_resamples": args.bootstrap_resamples,
        "gate": "ci_only_95pct_above_zero",
        "cells": cells,
        "anchor_cells": anchors,
        "sets": {
            k: {kk: v[kk] for kk in ("kind", "adapter_dir", "ce_mean", "n_rows")}
            for k, v in results.items()
        },
    }


# ── Gen mode (one vLLM engine + per-request LoRARequest; guard DV only) ─────


def run_gen_mode(args: argparse.Namespace) -> int:
    phase_log("absorption_gen")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from eval_issue543 import _teardown_vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = _corpus_local(args)
    rows = read_jsonl(data_path)[args.gen_start : args.gen_start + args.n_gen]
    if len(rows) != args.n_gen:
        raise RuntimeError(f"Gen slice wanted {args.n_gen} rows; file gave {len(rows)}.")

    sets = _resolve_sets(args)
    adapter_dirs = {s["name"]: resolve_adapter_dir(s) for s in sets if s["kind"] != "base"}

    llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=32,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.70,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=GEN_MAX_NEW_TOKENS, n=1)
    gen_prompts = [prompt_messages_of(r) for r in rows]
    prefixes = [
        tokenizer.apply_chat_template(pm, tokenize=False, add_generation_prompt=True)
        for pm in gen_prompts
    ]

    try:
        for lora_id, s in enumerate(sets, start=1):
            name = s["name"]
            out_path = out_dir / f"med_answers_{name}.json"
            if out_path.exists():
                log.info("Gen set %s exists (%s) — skipping (idempotent).", name, out_path)
                continue
            lora_req = None
            if s["kind"] != "base":
                lora_req = LoRARequest(name, lora_id, str(adapter_dirs[name]))
            log.info("Generating med answers: set=%s n=%d", name, len(prefixes))
            responses = llm.generate(prefixes, sampling, lora_request=lora_req)
            recs = []
            for i, (pm, prefix, resp) in enumerate(
                zip(gen_prompts, prefixes, responses, strict=True)
            ):
                g = resp.outputs[0]
                recs.append(
                    {
                        "row_index": args.gen_start + i,
                        "prompt_messages": pm,
                        "prefix": prefix,
                        "completion_text": g.text,
                        "n_generated_tokens": len(g.token_ids),
                        "truncated": len(g.token_ids) >= GEN_MAX_NEW_TOKENS,
                        "set": name,
                        "kind": s["kind"],
                        "adapter_path": str(adapter_dirs.get(name)) if lora_req else None,
                        "lora_id": lora_id if lora_req else None,
                    }
                )
            out_path.write_text(json.dumps({**repro_metadata(), "records": recs}, indent=2))
            log.info("Gen set %s persisted -> %s", name, out_path)
    finally:
        _teardown_vllm(llm)

    if not args.skip_upload:
        phase_log("absorption_upload")
        from explore_persona_space.orchestrate.hub import upload_dataset_directory

        upload_dataset_directory(
            out_dir,
            (
                f"{HUB_RAW_COMPLETIONS_BUCKET_570}/absorption"
                if args.issue_ns == ISSUE_570
                else f"{HUB_RAW_COMPLETIONS_BUCKET_557}/absorption"
            ),
            pattern="med_answers_*.json",
        )
    write_sentinel(
        "absorption-gen",
        kind="epm:progress",
        issue=_sentinel_issue(args),
        note=json.dumps(
            {"event": "absorption_gen_complete", "n_sets": len(sets), "n_gen": args.n_gen}
        ),
    )
    if not args.child:
        phase_log("done")
    return 0


# ── Orchestrating mode (subprocess-isolates the HF and vLLM phases) ──────────


def _run_child(cmd: list[str], log_path: Path, *, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("[%s] spawning: %s (log=%s)", label, " ".join(cmd), log_path)
    env = {**os.environ}
    with log_path.open("ab") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = ""
        try:
            with log_path.open("rb") as f:
                f.seek(max(0, log_path.stat().st_size - 4096))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            pass
        raise RuntimeError(f"[{label}] child failed (rc={proc.returncode}); log tail:\n{tail}")


def run_all_mode(args: argparse.Namespace) -> int:
    phase_log("absorption")
    common = [
        "--variants",
        ",".join(args.variants),
        "--seeds",
        ",".join(str(s) for s in args.seeds),
        "--n-ce-rows",
        str(args.n_ce_rows),
        "--gen-start",
        str(args.gen_start),
        "--n-gen",
        str(args.n_gen),
        "--bootstrap-resamples",
        str(args.bootstrap_resamples),
        "--gpu",
        str(args.gpu),
        "--out-dir",
        str(args.out_dir),
        "--eval-root",
        str(args.eval_root),
        "--child",
    ]
    if args.skip_upload:
        common.append("--skip-upload")
    if args.issue_ns is not None:
        common.extend(["--issue-ns", str(args.issue_ns)])
    if args.corpus_hf_path is not None:
        common.extend(["--corpus-hf-path", args.corpus_hf_path])
    if args.adapter_set_manifest is not None:
        common.extend(["--adapter-set-manifest", args.adapter_set_manifest])
    me = str(Path(__file__).resolve())
    ts = int(time.time())
    _run_child(
        [sys.executable, me, "--mode", "ce", *common],
        sentinel_dir() / f"issue-557-absorption-ce-{ts}.log",
        label="absorption-ce",
    )
    _run_child(
        [sys.executable, me, "--mode", "gen", *common],
        sentinel_dir() / f"issue-557-absorption-gen-{ts}.log",
        label="absorption-gen",
    )
    phase_log("done")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #557 medical-absorption guard (CE probe + quality gens).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=("all", "ce", "gen"), default="all")
    p.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--n-ce-rows", type=int, default=DEFAULT_N_CE_ROWS)
    p.add_argument("--gen-start", type=int, default=DEFAULT_GEN_START)
    p.add_argument("--n-gen", type=int, default=DEFAULT_N_GEN)
    p.add_argument("--bootstrap-resamples", type=int, default=10_000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    p.add_argument(
        "--eval-root",
        type=str,
        default=str(PROJECT_ROOT / "eval_results"),
        help="Root holding issue_557/<arm>/<variant>/seed<S>/phase2_result.json pointers.",
    )
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument(
        "--print-sets",
        action="store_true",
        help="CPU dry-run: print the resolved adapter-set list + output paths; exit 0.",
    )
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    # ── Issue #570 extension (defaults None -> exact #557 behavior) ──────────
    p.add_argument(
        "--issue-ns",
        type=int,
        choices=(ISSUE_570,),
        default=None,
        help="#570 namespace: sentinels post as issue-570, the gen upload "
        "bucket moves to issue570_clean_organism/raw_completions/absorption, "
        "and the corpus fetch pins the #570 data revision.",
    )
    p.add_argument(
        "--corpus-hf-path",
        type=str,
        default=None,
        help="#570: probe the arm's OWN corpus (e.g. the misaligned file) "
        "instead of the default good file. Requires --issue-ns 570.",
    )
    p.add_argument(
        "--adapter-set-manifest",
        type=str,
        default=None,
        help="#570 explicit adapter-set manifest (JSON list; see "
        "load_sets_manifest). Replaces the #557 build_sets grid; set names "
        "must keep the pre_seed<S>/post_<variant>_seed<S> convention so the "
        "aggregation keys resolve against --variants/--seeds.",
    )
    args = p.parse_args()
    args.variants = [validate_variant(v) for v in args.variants.split(",") if v]
    args.seeds = [int(s) for s in args.seeds.split(",") if s]
    if args.gen_start < args.n_ce_rows:
        raise SystemExit(
            f"--gen-start {args.gen_start} overlaps the CE probe rows [0:{args.n_ce_rows}) — "
            "the quality-read slice must stay disjoint (plan §11)."
        )
    if args.corpus_hf_path is not None and args.issue_ns is None:
        raise SystemExit("--corpus-hf-path requires --issue-ns 570 (#557 parity guard).")
    if args.adapter_set_manifest is not None and args.issue_ns is None:
        raise SystemExit("--adapter-set-manifest requires --issue-ns 570 (#557 parity guard).")
    if args.issue_ns == ISSUE_570:
        # Round-1 review Major (concern absorption-outdir-namespace): the
        # #557 default out-dir points at the parent's COMMITTED artifacts
        # (eval_results/issue_557/absorption/*.json on main); a #570 run
        # that omits --out-dir must never overwrite them. Auto-route the
        # default to the #570 namespace (plan §6.5 glob
        # eval_results/issue_570/absorption_<arm>/), keyed by the arm
        # variants this invocation aggregates over; then HARD-assert the
        # resolved out-dir is outside eval_results/issue_557/ regardless
        # of how it was supplied.
        if args.out_dir == str(OUT_DIR_DEFAULT):
            label = "_".join(args.variants) or "unlabeled"
            args.out_dir = str(EVAL_RESULTS_DIR_570 / f"absorption_{label}")
        resolved = Path(args.out_dir).resolve()
        if resolved == EVAL_RESULTS_DIR_557.resolve() or resolved.is_relative_to(
            EVAL_RESULTS_DIR_557.resolve()
        ):
            raise SystemExit(
                f"--issue-ns 570 with out_dir {args.out_dir} resolves under the "
                "parent #557 namespace eval_results/issue_557/ — refusing to "
                "overwrite committed parent artifacts (plan risk 7). Pass an "
                "--out-dir under eval_results/issue_570/."
            )
    return args


def main() -> int:
    args = parse_args()
    if args.print_sets:
        sets = _resolve_sets(args)
        bucket = (
            f"{HUB_RAW_COMPLETIONS_BUCKET_570}/absorption"
            if args.issue_ns == ISSUE_570
            else f"{HUB_RAW_COMPLETIONS_BUCKET_557}/absorption"
        )
        print(
            json.dumps(
                {
                    "out_dir": str(args.out_dir),
                    "issue_ns": args.issue_ns,
                    "corpus_hf_path": args.corpus_hf_path or PHASE2_DATASET_HF_PATH,
                    "upload_bucket": bucket,
                    "ce_outputs": [f"ce_{s['name']}.json" for s in sets],
                    "gen_outputs": [f"med_answers_{s['name']}.json" for s in sets],
                    "post_result_json_reads": [
                        s["adapter_source"].get("local_result_json")
                        for s in sets
                        if s["kind"] == "post"
                    ],
                    "sets": sets,
                },
                indent=2,
            )
        )
        return 0
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN missing from env — .env not loaded; aborting.")
    if args.mode == "ce":
        return run_ce_mode(args)
    if args.mode == "gen":
        return run_gen_mode(args)
    return run_all_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
