# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + minus sign − intentional
#!/usr/bin/env python3
"""Task #477 recovery diagnostic — re-eval ONE already-trained LoRA to localize the v4/v6 eval bug.

NOT training. NOT a sweep. Single cell. Single seed. ~20 held-out probes.

Hypothesis under test
---------------------
The v4 / v6 trajectory eval reported trained ≈ base (ΔG ≈ 0, emit 0) at EVERY
checkpoint across all counts despite the v4-trained LoRA adapter at
``adapters/issue_477/c477_calib_negp_2_seed42_lr2e-06`` (HF model repo
``superkaiba1/explore-persona-space``) having a B-matrix norm ~3.0 — the same
magnitude that gave ΔG=17.42 in the v2 eval. Suspected cause: the rig's vLLM
``LoRARequest`` path silently failed to apply the trained adapter (pod-env /
version drift, possibly triggered by ``frac_precision`` 2→4 directory rename),
so every eval read the BASE model's log P(※).

This diagnostic re-scores ΔG with the SAME held-out panel, SAME Q_eval, SAME
marker token, SAME slot construction the rig uses, via TWO independent paths:

  Path A — clean PEFT (HF Transformers + PeftModel.from_pretrained). Ground
           truth. If the adapter has any effect at all, this path sees it.
  Path B — vLLM LoRARequest (lora_path=<local adapter dir>). The exact
           mechanism eval_trajectory.score_logp_for_R uses on the trained pass.

Dispositive outcomes:
  * ΔG_peft ≈ 17  → adapter is genuinely trained AND applies → v4/v6 eval bug
                    confirmed (the trained LoRA was never applied at eval).
  * ΔG_vllm ≈ 0   → vLLM LoRARequest path is the bug (pinpoint).
  * BOTH ≈ 17     → original v4 failure was pod-env-specific; current env is
                    fine. Full re-eval will recover the count axis.

The script asserts NOTHING (no expected-value checks); it reports. Designed to
run < 5 min on a 1× H100 slice. Reuses ``score_logp_for_R`` and
``build_full_ids`` from ``eval_one_cell`` so Path B is byte-identical to the
production rig's measurement (only Path A re-implements the slot read because
the production KL path uses a full-vocab forward, not a marker-only logp).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i477.reval_confirm")

# ── Constants pinned to the rig + the broken eval cell. ──────────────────────
ADAPTER_HF_REPO = "superkaiba1/explore-persona-space"
ADAPTER_SUBFOLDER = "adapters/issue_477/c477_calib_negp_2_seed42_lr2e-06"
CELL_SLUG = "c477_calib_negp_2"  # the CELL_SPECS_477 entry the panel disjointness uses
SEED = 42

# Pinned data revision (the rev that produced the trained adapter).
DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_REVISION = "66d7db7a542e19275f8c1d8e32948396d050faa9"
DATA_PREFIX = "issue472_neg_geometry"
DATA_FILES = (
    f"{DATA_PREFIX}/geometry/persona_bank.json",
    f"{DATA_PREFIX}/geometry/centroids_L10.pt",
    f"{DATA_PREFIX}/on_policy_R/R_eval.json",
)

# Local mirror layout (matches the rig's expectations).
LOCAL_DATA_ROOT = Path("data/issue_472")
LOCAL_PATHS = {
    f"{DATA_PREFIX}/geometry/persona_bank.json": LOCAL_DATA_ROOT / "persona_bank.json",
    f"{DATA_PREFIX}/geometry/centroids_L10.pt": LOCAL_DATA_ROOT / "centroids_L10.pt",
    f"{DATA_PREFIX}/on_policy_R/R_eval.json": LOCAL_DATA_ROOT / "on_policy_R" / "R_eval.json",
}

# Slice knobs. Keep tiny — this is a diagnostic, not a sweep.
DEFAULT_N_HELDOUT = 15
DEFAULT_N_QUESTIONS = 4
DEFAULT_MAX_NEW_TOKENS = 256  # diagnostic — short greedy answers


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _ensure_data(token: str | None) -> None:
    """Pull the pinned-rev persona bank + centroids + R_eval into data/issue_472/."""
    from huggingface_hub import hf_hub_download

    LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (LOCAL_DATA_ROOT / "on_policy_R").mkdir(parents=True, exist_ok=True)
    for hf_path, local_path in LOCAL_PATHS.items():
        if local_path.exists():
            log.info("data already local: %s", local_path)
            continue
        log.info("pulling %s @ %s → %s", hf_path, DATA_REVISION[:8], local_path)
        cached = hf_hub_download(
            repo_id=DATA_REPO,
            repo_type="dataset",
            revision=DATA_REVISION,
            filename=hf_path,
            token=token,
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Hard-link (cheap) into the rig's expected layout.
        os.link(cached, local_path)


def _fetch_adapter(token: str | None, dest: Path) -> Path:
    """Download the trained adapter dir from HF to ``dest`` (snapshot_download)."""
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    log.info(
        "fetching adapter %s/%s → %s",
        ADAPTER_HF_REPO,
        ADAPTER_SUBFOLDER,
        dest,
    )
    snap = snapshot_download(
        repo_id=ADAPTER_HF_REPO,
        repo_type="model",
        allow_patterns=[f"{ADAPTER_SUBFOLDER}/*"],
        local_dir=str(dest),
        token=token,
    )
    adapter_dir = Path(snap) / ADAPTER_SUBFOLDER
    # Verify the load-bearing files landed.
    must_have = ("adapter_config.json", "adapter_model.safetensors")
    for fn in must_have:
        if not (adapter_dir / fn).exists():
            raise FileNotFoundError(
                f"adapter file {fn} missing under {adapter_dir} after snapshot_download — "
                f"check the snapshot_download siblings truncation memory."
            )
    log.info("adapter dir = %s", adapter_dir)
    return adapter_dir


def _select_eval_slice(
    n_heldout: int, n_questions: int
) -> tuple[dict[str, str], list[str], str, str]:
    """Build the held-out panel slice + Q_eval slice + source persona prompt.

    Returns (eval_personas_dict, q_eval_slice, source_name, source_prompt).
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        CELL_SPECS_477,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        HEADLINE_LAYER,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        all_negatives_union,
        held_out_panel,
        negatives_for_cell,
    )

    bank = load_persona_bank(LOCAL_PATHS[f"{DATA_PREFIX}/geometry/persona_bank.json"])
    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, LOCAL_DATA_ROOT)
    base_panel = held_out_panel(cts, source=SOURCE_PERSONA)
    union_477 = all_negatives_union(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
    panel = [p for p in base_panel if p not in union_477]
    log.info(
        "held-out panel (#477 disjoint): %d personas (= %d base − %d #477-negs)",
        len(panel),
        len(base_panel),
        len(base_panel) - len(panel),
    )
    # Disjointness assert vs THIS cell's negatives (round-3 #477 guard).
    cell_negs = set(
        negatives_for_cell(CELL_SLUG, cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
    )
    overlap = set(panel) & cell_negs
    if overlap:
        raise AssertionError(
            f"panel ∩ negatives for {CELL_SLUG!r}: {sorted(overlap)} — "
            "would conflate leakage with training-against-suppression."
        )
    # Take the first N personas (alphabetical, deterministic). `panel` is
    # already sorted by held_out_panel().
    panel_slice = panel[:n_heldout]
    eval_personas = {p: bank[p] for p in panel_slice}

    _q_train, q_eval = get_train_eval_questions()
    q_slice = list(q_eval[:n_questions])
    log.info(
        "eval slice: %d personas × %d questions = %d probes",
        len(panel_slice),
        len(q_slice),
        len(panel_slice) * len(q_slice),
    )
    return eval_personas, q_slice, SOURCE_PERSONA, bank[SOURCE_PERSONA]


def _hf_generate_R(
    model,
    tokenizer,
    eval_personas: dict[str, str],
    eval_questions: list[str],
    *,
    max_new_tokens: int,
    device: str,
) -> dict[str, dict[str, str]]:
    """Per-probe greedy HF generate (small loop; tiny slice). Returns r[persona][q] -> text."""
    import torch

    r: dict[str, dict[str, str]] = {p: {} for p in eval_personas}
    for persona, persona_prompt in eval_personas.items():
        for q in eval_questions:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": q},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(
                device
            )
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
            r[persona][q] = text
    return r


def _hf_score_marker_logp(
    model,
    tokenizer,
    r_by_persona_q: dict[str, dict[str, str]],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    *,
    device: str,
) -> dict[str, dict[str, dict[str, float | bool]]]:
    """Score log P(※) at the post-R marker slot via HF forward (Path A).

    For each (persona, q): build the byte-identical token sequence
    (prompt + R + MARKER_SEP + MARKER_TEXT) and read
    ``log_softmax(logits[slot - 1])[marker_id]`` — HF's logits at position t
    predict the token at position t+1, so the marker prediction lives at
    ``slot - 1``. Mirrors the production rig's slot-construction via
    ``build_full_ids`` (forked from #448) so PEFT and vLLM paths agree on the
    measurement modulo the load mechanism.
    """
    import torch

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
        build_full_ids,
    )

    out: dict[str, dict[str, dict[str, float | bool]]] = {p: {} for p in eval_personas}
    for persona, persona_prompt in eval_personas.items():
        for q in eval_questions:
            r_text = r_by_persona_q[persona][q]
            full_ids, _p_len, _r_len, slot, n_mk_R = build_full_ids(
                tokenizer,
                persona_prompt,
                q,
                r_text,
                MARKER_TEXT,
                EXPECTED_MARKER_TOKEN_ID,
                persona,
                q,
            )
            ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=ids_t).logits  # (1, T, V)
            # logits[0, slot - 1, :] is the distribution that predicts the
            # token at position `slot` (the appended marker).
            lp_full = torch.log_softmax(logits[0, slot - 1, :].float(), dim=-1).cpu()
            lp_marker = float(lp_full[EXPECTED_MARKER_TOKEN_ID].item())
            top_id = int(torch.argmax(lp_full).item())
            out[persona][q] = {
                "logp": lp_marker,
                "argmax_marker": top_id == EXPECTED_MARKER_TOKEN_ID,
                "n_marker_in_R": int(n_mk_R),
            }
    return out


def _summarize(
    scores: dict[str, dict[str, dict[str, float | bool]]],
    source: str,
) -> dict[str, float]:
    """Mean logp / emission over held-out (panel − source) and source-self."""
    held_lps: list[float] = []
    held_emits: list[bool] = []
    src_lps: list[float] = []
    src_emits: list[bool] = []
    for persona, per_q in scores.items():
        for _q, rec in per_q.items():
            lp = float(rec["logp"])
            em = bool(rec["argmax_marker"])
            if persona == source:
                src_lps.append(lp)
                src_emits.append(em)
            else:
                held_lps.append(lp)
                held_emits.append(em)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def _rate(xs: list[bool]) -> float:
        return sum(1 for x in xs if x) / len(xs) if xs else float("nan")

    return {
        "held_logp_mean": _mean(held_lps),
        "held_emit_rate": _rate(held_emits),
        "source_logp_mean": _mean(src_lps),
        "source_emit_rate": _rate(src_emits),
        "n_held_probes": len(held_lps),
        "n_source_probes": len(src_lps),
    }


def _teardown_vllm(llm) -> None:
    """Reap vLLM workers (CLAUDE.md vLLM-teardown gotcha; CVD-naive — this script
    is single-GPU + single-process, so the workers can only belong to us).
    """
    import torch

    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import psutil

        me = psutil.Process()
        for c in me.children(recursive=True):
            with contextlib.suppress(psutil.NoSuchProcess):
                c.terminate()
        _gone, alive = psutil.wait_procs(me.children(recursive=True), timeout=10)
        for c in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                c.kill()
    except ImportError:
        log.warning("psutil unavailable; cannot reap vLLM worker subprocesses.")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Task #477 recovery diagnostic: PEFT vs vLLM-LoRARequest re-eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-heldout", type=int, default=DEFAULT_N_HELDOUT)
    ap.add_argument("--n-questions", type=int, default=DEFAULT_N_QUESTIONS)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--device", default="cuda:0", help="HF device for Path A.")
    ap.add_argument(
        "--adapter-cache",
        type=Path,
        default=Path("/tmp/i477_reval/adapter_cache"),
        help="Local dir for snapshot_download of the trained adapter.",
    )
    ap.add_argument(
        "--out-path",
        type=Path,
        default=Path("eval_results/issue_477/reval_confirm/c477_calib_negp_2_seed42_lr2e-06.json"),
    )
    ap.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.40,
        help="vLLM gpu_memory_utilization for Path B (kept conservative — Path A "
        "may leave residual; 0.40 = ~32 GiB of an 80 GiB H100 → plenty for "
        "Qwen-2.5-7B bf16 + small KV).",
    )
    ap.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Run Path A only (smoke / import-check).",
    )
    ap.add_argument(
        "--skip-peft",
        action="store_true",
        help="Run Path B only (debug Path B in isolation).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=reval_confirm] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError(
            "HF_TOKEN missing — load_dotenv() ran but .env lacks the token; the "
            "adapter download + data fetch need it. Fix .env on the pod."
        )

    # ── Phase 0: data + adapter on disk. ─────────────────────────────────────
    log.info("[phase=fetch] data + adapter")
    _ensure_data(token)
    adapter_dir = _fetch_adapter(token, args.adapter_cache)

    # ── Phase 0.5: marker token assertion + eval slice build. ────────────────
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
        assert_marker_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=token)
    assert_marker_token(tokenizer)
    log.info("marker assert PASS: %r → [%d]", MARKER_TEXT, EXPECTED_MARKER_TOKEN_ID)

    eval_personas, q_eval, source_name, source_prompt = _select_eval_slice(
        args.n_heldout, args.n_questions
    )
    # Score source AND held-out personas; the panel-plus-source dict mirrors
    # eval_trajectory.run_trajectory_eval.
    panel_plus_source = dict(eval_personas)
    panel_plus_source.setdefault(source_name, source_prompt)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    partial: dict = {
        "schema_version": "i477_reval_confirm_v1",
        "cell": CELL_SLUG,
        "seed": SEED,
        "adapter_hf_repo": ADAPTER_HF_REPO,
        "adapter_subfolder": ADAPTER_SUBFOLDER,
        "data_revision": DATA_REVISION,
        "marker_text": MARKER_TEXT,
        "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
        "base_model": BASE_MODEL,
        "n_heldout_personas": len(eval_personas),
        "held_out_personas": sorted(eval_personas.keys()),
        "n_questions": len(q_eval),
        "questions": q_eval,
        "source": source_name,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "paths": {},
        "summary": {},
    }

    # ── Phase A: clean PEFT (ground truth). ──────────────────────────────────
    peft_records: dict | None = None
    base_records_peft_R: dict | None = None
    R_peft: dict[str, dict[str, str]] | None = None
    if not args.skip_peft:
        log.info("[phase=peft] loading trained model via PEFT (Path A — ground truth)")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            device_map={"": args.device},
            trust_remote_code=True,
            token=token,
        ).eval()
        trained_peft = PeftModel.from_pretrained(base_model, str(adapter_dir)).eval()

        log.info("[phase=peft] generating R_PEFT (trained, greedy)")
        R_peft = _hf_generate_R(
            trained_peft,
            tokenizer,
            panel_plus_source,
            q_eval,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        log.info("[phase=peft] scoring g_logp on R_PEFT")
        peft_records = _hf_score_marker_logp(
            trained_peft,
            tokenizer,
            R_peft,
            panel_plus_source,
            q_eval,
            device=args.device,
        )
        # Drop PEFT-wrapped model, reload a CLEAN base.
        del trained_peft, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info("[phase=peft] loading CLEAN base for b_logp on R_PEFT")
        base_clean = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            device_map={"": args.device},
            trust_remote_code=True,
            token=token,
        ).eval()
        base_records_peft_R = _hf_score_marker_logp(
            base_clean,
            tokenizer,
            R_peft,
            panel_plus_source,
            q_eval,
            device=args.device,
        )
        del base_clean
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Persist Phase A's per-probe records IMMEDIATELY (per-phase checkpoint
        # rule — vLLM may crash and we don't lose Path A).
        partial["paths"]["peft"] = {
            "R": R_peft,
            "g_records": peft_records,
            "b_records": base_records_peft_R,
        }
        partial["summary"]["peft"] = {
            "trained": _summarize(peft_records, source_name),
            "base_on_peft_R": _summarize(base_records_peft_R, source_name),
        }
        args.out_path.write_text(json.dumps(partial, indent=2))
        log.info("[phase=peft] persisted Phase A → %s", args.out_path)

    # ── Phase B: vLLM LoRARequest (suspect — the rig's mechanism). ──────────
    vllm_records: dict | None = None
    base_records_vllm_R: dict | None = None
    R_vllm: dict[str, dict[str, str]] | None = None
    if not args.skip_vllm:
        log.info("[phase=vllm] loading vLLM with enable_lora + LoRARequest (Path B — suspect)")
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        from explore_persona_space.experiments.contrastive_neg_geometry_472 import LORA_R
        from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
            score_logp_for_R,
        )
        from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
            _generate_on_policy_R,
        )

        llm = LLM(
            model=BASE_MODEL,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_mem_util,
            seed=SEED,
            max_model_len=2048,
            enable_lora=True,
            max_lora_rank=LORA_R,
            max_loras=1,
        )
        lora_req = LoRARequest(
            lora_name=f"{CELL_SLUG}_seed{SEED}_reval",
            lora_int_id=1,
            lora_path=str(adapter_dir),
        )

        log.info("[phase=vllm] generating R_VLLM (trained, greedy via vLLM)")
        R_vllm = _generate_on_policy_R(
            llm,
            tokenizer,
            panel_plus_source,
            q_eval,
            lora_req,
            args.max_new_tokens,
        )

        log.info("[phase=vllm] scoring g_logp on R_VLLM (use_lora=True)")
        vllm_records = score_logp_for_R(
            llm,
            tokenizer,
            r_by_persona_q=R_vllm,
            eval_personas=panel_plus_source,
            eval_questions=q_eval,
            cell_label=f"TRAINED/{CELL_SLUG}_seed{SEED}_reval",
            use_lora=True,
            lora_request=lora_req,
        )

        log.info("[phase=vllm] scoring b_logp on R_VLLM (use_lora=False)")
        base_records_vllm_R = score_logp_for_R(
            llm,
            tokenizer,
            r_by_persona_q=R_vllm,
            eval_personas=panel_plus_source,
            eval_questions=q_eval,
            cell_label=f"BASE/{CELL_SLUG}_seed{SEED}_reval",
            use_lora=False,
        )

        _teardown_vllm(llm)

        partial["paths"]["vllm"] = {
            "R": R_vllm,
            "g_records": vllm_records,
            "b_records": base_records_vllm_R,
        }
        partial["summary"]["vllm"] = {
            "trained": _summarize(vllm_records, source_name),
            "base_on_vllm_R": _summarize(base_records_vllm_R, source_name),
        }
        args.out_path.write_text(json.dumps(partial, indent=2))
        log.info("[phase=vllm] persisted Phase B → %s", args.out_path)

    # ── Phase 2: print the diagnostic table. ─────────────────────────────────
    print("\n" + "=" * 80)
    print(f"#477 RECOVERY DIAGNOSTIC — {CELL_SLUG} seed={SEED}")
    print(f"adapter: {ADAPTER_HF_REPO}/{ADAPTER_SUBFOLDER}")
    print(
        f"slice: {len(eval_personas)} held-out personas × {len(q_eval)} questions  |  "
        f"source: {source_name}"
    )
    print("=" * 80)

    def _fmt_row(label: str, b: float, g: float) -> str:
        return f"  {label:<24} b_logp={b:8.3f}  g_logp={g:8.3f}  ΔG={g - b:+8.3f}"

    if peft_records is not None and base_records_peft_R is not None:
        ps_held = partial["summary"]["peft"]["trained"]
        ps_base_held = partial["summary"]["peft"]["base_on_peft_R"]
        print("\n[Path A — clean PEFT]  (ground truth)")
        print(_fmt_row("held-out mean:", ps_base_held["held_logp_mean"], ps_held["held_logp_mean"]))
        print(
            _fmt_row(
                "source-self mean:", ps_base_held["source_logp_mean"], ps_held["source_logp_mean"]
            )
        )
        print(
            f"  source emit P(※)  trained={ps_held['source_emit_rate']:.2f}  "
            f"base={ps_base_held['source_emit_rate']:.2f}"
        )
        print(
            f"  held emit P(※)    trained={ps_held['held_emit_rate']:.2f}  "
            f"base={ps_base_held['held_emit_rate']:.2f}"
        )

    if vllm_records is not None and base_records_vllm_R is not None:
        vs_held = partial["summary"]["vllm"]["trained"]
        vs_base_held = partial["summary"]["vllm"]["base_on_vllm_R"]
        print("\n[Path B — vLLM LoRARequest]  (suspect — the rig's mechanism)")
        print(_fmt_row("held-out mean:", vs_base_held["held_logp_mean"], vs_held["held_logp_mean"]))
        print(
            _fmt_row(
                "source-self mean:", vs_base_held["source_logp_mean"], vs_held["source_logp_mean"]
            )
        )
        print(
            f"  source emit P(※)  trained={vs_held['source_emit_rate']:.2f}  "
            f"base={vs_base_held['source_emit_rate']:.2f}"
        )
        print(
            f"  held emit P(※)    trained={vs_held['held_emit_rate']:.2f}  "
            f"base={vs_base_held['held_emit_rate']:.2f}"
        )

    if peft_records is not None and vllm_records is not None:
        peft_src_dg = (
            partial["summary"]["peft"]["trained"]["source_logp_mean"]
            - partial["summary"]["peft"]["base_on_peft_R"]["source_logp_mean"]
        )
        vllm_src_dg = (
            partial["summary"]["vllm"]["trained"]["source_logp_mean"]
            - partial["summary"]["vllm"]["base_on_vllm_R"]["source_logp_mean"]
        )
        print("\n[Cross-path verdict]  (source-self ΔG — the adapter-applied signal)")
        print(f"  PEFT ΔG_source = {peft_src_dg:+.3f} nats")
        print(f"  vLLM ΔG_source = {vllm_src_dg:+.3f} nats")
        if peft_src_dg > 5.0 and vllm_src_dg < 1.0:
            print("  → DISPOSITIVE: vLLM LoRARequest path silently failed (v4/v6 bug pinpointed).")
        elif peft_src_dg > 5.0 and vllm_src_dg > 5.0:
            print("  → DISPOSITIVE: both paths apply adapter — v4 failure was pod-env-specific.")
        elif peft_src_dg < 1.0:
            print("  → ANOMALY: PEFT path also reads ΔG ≈ 0 — adapter may not be what we think.")
        else:
            print("  → UNCLEAR: partial signal; investigate per-probe records in the JSON.")

    print("\nfull per-probe records → " + str(args.out_path))
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
