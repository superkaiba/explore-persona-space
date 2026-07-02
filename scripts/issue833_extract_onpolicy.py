#!/usr/bin/env python
"""Issue #833 — on-policy answer-profile generation + 2-leg paired extraction.

Phases A0/A/B/C of plan v3 (tasks/running/833/plans/plan.md §4). The Phase-D fit
driver is the SIBLING script ``issue833_fit_onpolicy.py`` (NOT this file).

Stages (``--stage``):
  parity   — A0 gate set: (i) vLLM-multi-LoRA vs PeftModel backend parity on 2
             cells (one fact α=64 + one em α=256), 3 targets × 5 probes; (ii)
             cross-era extraction parity vs the stored #667 npz legs; (iii) a
             1-cell Phase-B smoke incl. an npz write + a store-schema join. ALL
             gates hard-fail (exit non-zero) before Phase A.
  generate — Phase A: greedy on-policy R⁺ from each source-adapter-loaded model
             (vLLM multi-LoRA; ``--gen-backend peft`` = batched PeftModel
             fallback, plan §8 R1) over behavior × 16 sources × 30 targets ×
             probe pool. Rollout text persists per cell IMMEDIATELY.
  extract  — Phase B: teacher-force each R⁺ through BOTH θ0 and θ⁺
             (``load_base_and_trained``, rsLoRA gauge assert) and write npz
             mirroring the #667 store schema/naming with the NEW legs
             ``v_plus_onpolicy`` / ``v0_onpolicy`` (+ loader-compatible
             ``v0``/``v_plus`` aliases and the store's context vectors copied
             from the PINNED store revision, so the #722 loader joins the new
             store unchanged).
  upload   — Phase C: raw completions + analysis tensors → HF data repo
             (bulk ``upload_folder``, one commit per behavior), verified with a
             ``list_repo_files`` count (4,320 npz expected).
  finalize — compose the pod-side results sentinel (counts + parity summaries +
             structured reproducibility card).
  all      — subprocess-sequences parity → generate → extract → upload
             (separate processes: vLLM-before-transformers isolation, #667 R3).

Cross-era note (A0 gate ii): #667 never persisted the R_base rollout TEXT
(pre-#779 store; verified against the pinned revision — the store carries ONLY
``analysis_tensors/*.npz``, and the npz payloads hold no response text). R_base
was vLLM GREEDY (temp=0) from the frozen base model, so this gate REGENERATES
R_base greedily on the current stack with #667's exact generation path and
compares the re-extracted v0/v⁺ means against the stored npz. A PASS therefore
certifies the whole generation+extraction chain reproduces the store (stronger
than text-only parity); a FAIL is ambiguous between text drift and extraction
drift and ABORTS the run before Phase A (exit 3) — the plan's fleet-wide L1/L2
re-extraction fallback is then an orchestrator dispatch decision, not a silent
in-run continue.
"""

# ruff: noqa: E402, RUF001, RUF002, RUF003  # math/scientific notation in docstrings (#667 precedent)

import os

# vLLM EngineCore fork() poisoning guard — must precede ANY vllm import
# (.claude/rules/gotchas.md § vLLM V1 fork EngineCore silent death; #628).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import argparse
import datetime
import gc
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # project wrapper (lint rule)

load_dotenv()

# scripts-import-scripts precedent: issue722_fit_M.py `import issue658_fit_predictors`.
# Importing issue667_extract also re-asserts the spawn guard at ITS module top.
import issue667_extract as x667
import numpy as np
import torch

logger = logging.getLogger("issue833_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue833_onpolicy_map"
RAW_PREFIX = f"{HF_PREFIX}/raw_completions/generation"
TENSORS_PREFIX = f"{HF_PREFIX}/analysis_tensors"
# EVERY #667-store read threads this revision (plan v3 §10 consistency note).
STORE_REVISION = "0031fc55a0e965c33be4261287cd5c86393ca161"
STORE_PREFIX = "issue667_gate_chain_preview/analysis_tensors"

BEHAVIORS = ("em", "sycophancy", "fact")
LAYERS_DEFAULT = (7, 14, 21)
SEED_DEFAULT = 42
EXPECTED_NPZ_TOTAL = 3 * 16 * 30 * 3  # behaviors × sources × targets × layers = 4,320

# A0 gate thresholds (plan §4 A0 / §7.1).
A0_TOKEN_IDENTICAL_FLOOR = 0.80
A0_NEAR_TIE_LOGIT_GAP = 0.05
A0_CROSS_ERA_REL_L2 = 1e-3

# Chunk vLLM generate calls (gotchas: large single-batch EngineCore deadlock).
VLLM_GREEDY_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))

DEFAULT_OUT = "eval_results/issue_833"
SENTINEL_DIR = Path(os.environ.get("EPM_SENTINEL_DIR", "/workspace/logs"))


# ─────────────────────────────────────────────────────────────────────────────
# Small shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _write_json(path: Path, payload: dict) -> Path:
    """Atomic JSON write (tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)
    return path


def _write_phase_sentinel(name: str, payload: dict) -> None:
    """Pod-side phase sentinel under /workspace/logs (never task.py from a pod)."""
    try:
        SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
        _write_json(SENTINEL_DIR / f"issue-833-{name}.json", payload)
    except OSError as e:  # VM smoke has no /workspace — log, don't die
        logger.warning("phase sentinel %s not written (%s)", name, e)


def _targets_for(behavior: str, source_cid: str) -> list[str]:
    """Mirror #667's target resolution EXACTLY: 30 eval cids + the source diagonal."""
    from explore_persona_space.experiments.i537_contexts import eval_cids_for

    return list(dict.fromkeys([*eval_cids_for(behavior), source_cid]))


def _sources_for(behavior: str) -> list[str]:
    from explore_persona_space.experiments.i537_contexts import train_cids_for

    return train_cids_for(behavior)


def _stage_store_npz(behavior: str, source_cid: str, seed: int, tcid: str, layer: int) -> dict:
    """Download ONE stored #667 npz at the PINNED revision and return its arrays."""
    from huggingface_hub import hf_hub_download

    rel = f"{STORE_PREFIX}/{behavior}/{source_cid}_seed{seed}/{tcid}_L{layer}.npz"
    path = hf_hub_download(HF_DATA_REPO, rel, repo_type="dataset", revision=STORE_REVISION)
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _load_inputs():
    """(registry, demos, tok) — the #537 frozen context inputs + base tokenizer."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i537_contexts import load_icl_demos, load_registry

    sampled_path, demos_path = x667.stage_inputs()
    registry = load_registry(sampled_path)
    demos = load_icl_demos(demos_path)
    tok = AutoTokenizer.from_pretrained(x667.BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    return registry, demos, tok


def _resolve_adapter(behavior: str, source_cid: str, seed: int) -> tuple[str, Path, dict]:
    """(hf_subfolder, staged_local_dir, gauge) with the rsLoRA gauge asserted.

    α is read from EACH adapter's own adapter_config.json (em=256, fact/syc=64 —
    NEVER hardcoded; plan v2 fact-check correction; ``assert_adapter_gauge``
    checks base-model + rsLoRA, not α).
    """
    from explore_persona_space.experiments.issue_651 import resolve_adapter_subfolder

    subfolder = resolve_adapter_subfolder(behavior, source_cid, seed)
    adapter_dir = x667.stage_adapter_local(behavior, source_cid, seed)
    gauge = x667.assert_adapter_gauge(adapter_dir, behavior)
    scaling = float(gauge["lora_alpha"]) / float(gauge["r"]) ** 0.5  # rsLoRA α/√r
    logger.info(
        "adapter %s/%s: subfolder=%s r=%s alpha=%s rslora_scaling=%.3f",
        behavior,
        source_cid,
        subfolder,
        gauge["r"],
        gauge["lora_alpha"],
        scaling,
    )
    return subfolder, adapter_dir, {**gauge, "rslora_scaling": scaling, "hf_subfolder": subfolder}


# ─────────────────────────────────────────────────────────────────────────────
# vLLM multi-LoRA generation (NEW vs #667 — its generation path is base-only)
# ─────────────────────────────────────────────────────────────────────────────


class _VllmLoraEngine:
    """One vLLM engine serving base + all adapters via per-request LoRARequest."""

    def __init__(self, gpu_mem_util: float = 0.85):
        from vllm import LLM

        self._rslora_scaling_line_asserted = _assert_installed_vllm_rslora()
        self.llm = LLM(
            model=x667.BASE_MODEL,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=32,
            gpu_memory_utilization=gpu_mem_util,
        )
        self._next_lora_id = 1
        self._lora_ids: dict[str, int] = {}

    def _lora_request(self, adapter_dir: Path | None):
        if adapter_dir is None:
            return None
        from vllm.lora.request import LoRARequest

        key = str(adapter_dir)
        if key not in self._lora_ids:
            self._lora_ids[key] = self._next_lora_id
            self._next_lora_id += 1
        return LoRARequest(Path(key).name, self._lora_ids[key], key)

    def greedy(
        self, prompts: list[str], *, max_new_tokens: int, adapter_dir: Path | None
    ) -> list[list[int]]:
        """Greedy token-id sequences, chunked (EngineCore deadlock guard)."""
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        lora_req = self._lora_request(adapter_dir)
        out: list[list[int]] = []
        n_chunks = (len(prompts) + VLLM_GREEDY_CHUNK_SIZE - 1) // VLLM_GREEDY_CHUNK_SIZE
        for i in range(0, len(prompts), VLLM_GREEDY_CHUNK_SIZE):
            chunk = prompts[i : i + VLLM_GREEDY_CHUNK_SIZE]
            logger.info(
                "[vllm-chunk] greedy chunk %d/%d (%d prompts, adapter=%s)",
                i // VLLM_GREEDY_CHUNK_SIZE + 1,
                n_chunks,
                len(chunk),
                adapter_dir.name if adapter_dir else "BASE",
            )
            outputs = self.llm.generate(chunk, sp, lora_request=lora_req, use_tqdm=False)
            assert len(outputs) == len(chunk), (len(outputs), len(chunk))
            out.extend(list(o.outputs[0].token_ids) for o in outputs)
        return out

    def shutdown(self) -> None:
        from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

        _reap_vllm_engine(self.llm)
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _assert_installed_vllm_rslora() -> bool:
    """One-time assert: installed vLLM computes scaling = α/√r for use_rslora (plan §4 A0)."""
    import inspect

    from vllm.lora import peft_helper

    src = inspect.getsource(peft_helper)
    ok = "math.sqrt" in src and "use_rslora" in src
    if not ok:
        raise RuntimeError(
            "installed vLLM peft_helper does not show the rsLoRA α/√r scaling line — "
            "the A0 gauge premise (plan §11 backend row) does not hold; HALT"
        )
    return True


def _strip_trailing_eos(ids: list[int], eos_ids: set[int]) -> list[int]:
    out = list(ids)
    while out and out[-1] in eos_ids:
        out.pop()
    return out


def _peft_batched_greedy(
    trained, tok, prompts: list[str], *, max_new_tokens: int, device, batch_size: int = 8
) -> list[list[int]]:
    """Reference/fallback backend (b): batched PeftModel.generate greedy (plan §8 R1)."""
    tok_padded = tok
    prior_side = tok_padded.padding_side
    tok_padded.padding_side = "left"
    if tok_padded.pad_token_id is None:
        tok_padded.pad_token = tok_padded.eos_token
    out: list[list[int]] = []
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i : i + batch_size]
                enc = tok_padded(chunk, return_tensors="pt", padding=True).to(device)
                gen = trained.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tok_padded.pad_token_id,
                )
                plen = enc["input_ids"].shape[1]
                out.extend(row[plen:].tolist() for row in gen)
                logger.info(
                    "[peft-batch] greedy %d/%d", min(i + batch_size, len(prompts)), len(prompts)
                )
    finally:
        tok_padded.padding_side = prior_side
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage A0 — parity gates + Phase-B smoke (hard-fail before Phase A)
# ─────────────────────────────────────────────────────────────────────────────


def _a0_cells() -> list[tuple[str, str]]:
    """One fact (α=64) + one em (α=256) cell — the high-α rsLoRA path is exercised."""
    return [("fact", _sources_for("fact")[0]), ("em", _sources_for("em")[0])]


def _prompt_text(tok, registry, demos, cid: str, behavior: str, q: str) -> str:
    msgs = x667.build_messages_for(registry, demos, cid, behavior, q)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def stage_parity(args) -> int:
    """A0: backend parity + cross-era parity + 1-cell Phase-B smoke. Exit non-zero on FAIL."""
    out_root = Path(args.out)
    parity_dir = out_root / "parity"
    parity_dir.mkdir(parents=True, exist_ok=True)
    device = x667._device(args.gpu_id, args.cpu_only)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    registry, demos, tok = _load_inputs()
    eos_ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos_ids.add(im_end)

    cells = _a0_cells()
    n_targets, n_probes = (1, 2) if args.cpu_smoke else (3, 5)
    layers = list(args.layers)
    gates: dict[str, dict] = {}

    # Resolve adapters + prompts up front (cheap; HALT early on a gauge miss).
    cell_meta: dict[tuple[str, str], dict] = {}
    for behavior, source in cells:
        _, adapter_dir, gauge = _resolve_adapter(behavior, source, args.seed)
        targets = _targets_for(behavior, source)[:n_targets]
        probes = x667.load_eval_probes(behavior)[:n_probes]
        prompts = [
            _prompt_text(tok, registry, demos, tcid, behavior, q)
            for tcid in targets
            for q in probes
        ]
        cell_meta[(behavior, source)] = {
            "adapter_dir": adapter_dir,
            "gauge": gauge,
            "targets": targets,
            "probes": probes,
            "prompts": prompts,
        }

    # ── vLLM session (GPU only): backend-parity LoRA gens + cross-era base gens
    # + Phase-B smoke R⁺ gens, then reap the engine before any HF model load.
    smoke_cell = cells[0]  # the fact cell smokes Phase B end-to-end
    if device.type != "cpu" and args.gen_backend == "vllm":
        vllm_ids, cross_era_rbase = _a0_vllm_generations(cell_meta, tok, registry, demos, args)
        smoke_rplus_ids: list[list[int]] | None = vllm_ids[smoke_cell]
    else:
        vllm_ids, cross_era_rbase, smoke_rplus_ids = {}, {}, None

    verdicts_ok = True

    # ── per-cell HF phase: PeftModel reference gen + near-tie checks + cross-era
    # teacher-forced means + (fact cell) the Phase-B smoke.
    backend_results = []
    cross_era_results = []
    smoke_result: dict | None = None
    for key, meta in cell_meta.items():
        behavior, source = key
        _, base, trained = x667.load_base_and_trained(meta["adapter_dir"], device, dtype)
        # (i) backend parity — GPU/vLLM path only (the CPU smoke cannot run vLLM).
        if key in vllm_ids:
            check = _backend_parity_check(
                trained,
                tok,
                meta["prompts"],
                vllm_ids[key],
                eos_ids,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            verdicts_ok &= check["pass"]
            backend_results.append({"behavior": behavior, "source_cid": source, **check})
            logger.info(
                "A0 backend parity %s/%s: %d/%d identical (%.2f), %d divergences, pass=%s",
                behavior,
                source,
                check["n_token_identical"],
                check["n_prompts"],
                check["frac_identical"],
                len(check["divergences"]),
                check["pass"],
            )

        # (ii) cross-era parity vs the stored #667 npz (pinned revision).
        full_probes = x667.load_eval_probes(behavior)
        if key in cross_era_rbase:
            texts = [tok.decode(ids, skip_special_tokens=True) for ids in cross_era_rbase[key]]
        elif device.type != "cpu" and args.gen_backend == "peft":
            texts = _peft_rbase_texts(
                base, tok, registry, demos, behavior, meta["targets"], full_probes, args, device
            )
        elif args.cpu_smoke:
            # CPU smoke: mechanics only (staging + residual code path); a CPU
            # greedy 7B regeneration is infeasible — synthetic short responses,
            # residuals reported UN-ASSERTED (cpu_smoke flag marks them).
            texts = [
                "This is a short synthetic smoke response."
                for _ in range(len(meta["targets"]) * min(2, len(full_probes)))
            ]
            full_probes = full_probes[:2]
        else:
            raise RuntimeError("cross-era gate needs vLLM generation (GPU) or --cpu-smoke")
        ce_cell = _cross_era_check(
            base,
            trained,
            tok,
            registry,
            demos,
            behavior,
            source,
            args.seed,
            meta["targets"],
            full_probes,
            texts,
            layers,
            device,
            assert_gate=not args.cpu_smoke,
        )
        cross_era_results.append(ce_cell)
        if not args.cpu_smoke:
            verdicts_ok &= ce_cell["pass"]

        # (iii) 1-cell Phase-B smoke: npz write + store-schema join (fact cell).
        if key == smoke_cell:
            if smoke_rplus_ids is not None:
                rplus_texts = [tok.decode(ids, skip_special_tokens=True) for ids in smoke_rplus_ids]
            elif device.type != "cpu" and args.gen_backend == "peft":
                rp_ids = _peft_batched_greedy(
                    trained,
                    tok,
                    meta["prompts"],
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                rplus_texts = [tok.decode(ids, skip_special_tokens=True) for ids in rp_ids]
            else:
                rplus_texts = ["Synthetic on-policy smoke response, two sentences long."] * (
                    len(meta["targets"]) * len(meta["probes"])
                )
            smoke_result = _phase_b_smoke(
                base,
                trained,
                tok,
                registry,
                demos,
                behavior,
                source,
                args.seed,
                meta["targets"][:1],
                meta["probes"][:2],
                rplus_texts,
                layers,
                device,
                meta["gauge"],
                out_root / "parity" / "smoke_store",
                gen_backend=args.gen_backend,
            )
            verdicts_ok &= smoke_result["pass"]
        del base, trained
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    backend_trivial = args.gen_backend == "peft" and not args.cpu_smoke
    gates = {
        "backend_parity": {
            "results": backend_results,
            # peft fallback: generation backend == the teacher-force reference
            # backend, so the parity gate re-run passes trivially (plan §7.1).
            "pass": (
                True
                if backend_trivial
                else (all(r["pass"] for r in backend_results) if backend_results else None)
            ),
            "trivial": backend_trivial,
            "skipped": not backend_results and not backend_trivial,
            "gen_backend": args.gen_backend,
        },
        "cross_era": {
            "results": cross_era_results,
            "pass": (all(r["pass"] for r in cross_era_results) if not args.cpu_smoke else None),
            "cpu_smoke_unasserted": bool(args.cpu_smoke),
            "r_text_provenance": "regenerated-greedy (stored R_base text not persisted by #667)",
        },
        "phase_b_smoke": smoke_result,
        "cpu_smoke": bool(args.cpu_smoke),
        "ts": _now(),
    }
    _write_json(parity_dir / "a0_summary.json", gates)
    _write_json(
        parity_dir / "a0_cross_era.json",
        {"results": cross_era_results, "rel_l2_threshold": A0_CROSS_ERA_REL_L2, "ts": _now()},
    )
    _write_phase_sentinel("a0-gates", {"gates": gates, "pass": verdicts_ok})
    if not verdicts_ok:
        logger.error("A0 gate FAIL — aborting before Phase A (see a0_summary.json)")
        return 2 if (backend_results and not gates["backend_parity"]["pass"]) else 3
    logger.info("A0 gates PASS")
    return 0


def _a0_vllm_generations(cell_meta, tok, registry, demos, args):
    """ONE vLLM engine session for every A0 generation, reaped before HF loads.

    Per cell: LoRA-loaded greedy over the parity prompt grid, PLUS a BASE-model
    (no LoRA) regeneration of R_base over the FULL probe pool for the checked
    targets — the stored npz v0/v_plus are means over the full pool, so the
    cross-era comparison must match it.
    """
    vllm_ids: dict[tuple[str, str], list[list[int]]] = {}
    cross_era_rbase: dict[tuple[str, str], list[list[int]]] = {}
    engine = _VllmLoraEngine()
    try:
        for key, meta in cell_meta.items():
            vllm_ids[key] = engine.greedy(
                meta["prompts"],
                max_new_tokens=args.max_new_tokens,
                adapter_dir=meta["adapter_dir"],
            )
            full_probes = x667.load_eval_probes(key[0])
            ce_prompts = [
                _prompt_text(tok, registry, demos, tcid, key[0], q)
                for tcid in meta["targets"]
                for q in full_probes
            ]
            cross_era_rbase[key] = engine.greedy(
                ce_prompts, max_new_tokens=args.max_new_tokens, adapter_dir=None
            )
    finally:
        engine.shutdown()
    return vllm_ids, cross_era_rbase


def _peft_rbase_texts(base, tok, registry, demos, behavior, targets, probes, args, device):
    """Fallback backend (b): regenerate R_base via batched BASE-model HF greedy.

    Same conditioning as the vLLM path; under the peft fallback the backend
    gate is trivially satisfied because generation and teacher-forcing share
    one loading path (plan §7.1 / §8 R1).
    """
    ce_prompts = [
        _prompt_text(tok, registry, demos, tcid, behavior, q) for tcid in targets for q in probes
    ]
    ce_ids = _peft_batched_greedy(
        base, tok, ce_prompts, max_new_tokens=args.max_new_tokens, device=device
    )
    return [tok.decode(ids, skip_special_tokens=True) for ids in ce_ids]


def _backend_parity_check(
    trained, tok, prompts, vllm_ids_cell, eos_ids, *, max_new_tokens, device
) -> dict:
    """vLLM-vs-PeftModel greedy token parity for ONE cell (plan §4 A0 gate i)."""
    hf_ids = _peft_batched_greedy(
        trained, tok, prompts, max_new_tokens=max_new_tokens, device=device
    )
    n_ident = 0
    divergences = []
    for pi, (v_ids, h_ids) in enumerate(zip(vllm_ids_cell, hf_ids, strict=True)):
        v = _strip_trailing_eos(v_ids, eos_ids)
        h = _strip_trailing_eos(h_ids, eos_ids)
        if v == h:
            n_ident += 1
            continue
        div_idx = next(
            (i for i, (a, b) in enumerate(zip(v, h, strict=False)) if a != b),
            min(len(v), len(h)),
        )
        gap = _near_tie_gap(trained, tok, prompts[pi], v[:div_idx], v, h, div_idx, device)
        divergences.append({"prompt_idx": pi, "div_idx": div_idx, "logit_gap": gap})
    frac = n_ident / len(prompts)
    all_near_tie = all(d["logit_gap"] < A0_NEAR_TIE_LOGIT_GAP for d in divergences)
    return {
        "n_prompts": len(prompts),
        "n_token_identical": n_ident,
        "frac_identical": frac,
        "divergences": divergences,
        "pass": frac >= A0_TOKEN_IDENTICAL_FLOOR and all_near_tie,
    }


@torch.no_grad()
def _near_tie_gap(trained, tok, prompt: str, common_prefix, v_seq, h_seq, div_idx, device) -> float:
    """|z[vllm_tok] − z[hf_tok]| under the PeftModel at the first divergent slot."""
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    ids = torch.tensor([prompt_ids + list(common_prefix)], dtype=torch.long, device=device)
    logits = trained(ids).logits[0, -1, :]
    a = v_seq[div_idx] if div_idx < len(v_seq) else tok.eos_token_id
    b = h_seq[div_idx] if div_idx < len(h_seq) else tok.eos_token_id
    return float(abs(logits[a].item() - logits[b].item()))


def _cross_era_check(
    base,
    trained,
    tok,
    registry,
    demos,
    behavior,
    source,
    seed,
    targets,
    probes,
    texts,
    layers,
    device,
    *,
    assert_gate: bool,
) -> dict:
    """Teacher-force regenerated R_base through θ0+θ⁺; rel-L2 vs stored v0/v_plus."""
    residuals = []
    cell_pass = True
    ti = 0
    for tcid in targets:
        acc: dict[int, list[list[np.ndarray]]] = {li: [[], []] for li in layers}
        for q in probes:
            msgs = x667.build_messages_for(registry, demos, tcid, behavior, q)
            r = texts[ti]
            ti += 1
            if not r.strip():
                continue
            per_layer = x667._mean_resp_acts(base, trained, tok, msgs, r, layers, device)
            for li in layers:
                v0, vp = per_layer[li]
                acc[li][0].append(v0)
                acc[li][1].append(vp)
        for li in layers:
            if not acc[li][0]:
                continue
            stored = _stage_store_npz(behavior, source, seed, tcid, li)
            for leg, new_mean in (
                ("v0", np.stack(acc[li][0]).mean(axis=0)),
                ("v_plus", np.stack(acc[li][1]).mean(axis=0)),
            ):
                ref = np.asarray(stored[leg], dtype=np.float64)
                rel = float(np.linalg.norm(new_mean.astype(np.float64) - ref) / np.linalg.norm(ref))
                ok = rel < A0_CROSS_ERA_REL_L2
                cell_pass &= ok
                residuals.append(
                    {
                        "target_cid": tcid,
                        "layer": li,
                        "leg": leg,
                        "rel_l2": rel,
                        "n_probes_new": len(acc[li][0]),
                        "n_probes_stored": int(np.asarray(stored["n_probes"]).item()),
                        "pass": ok,
                    }
                )
    result = {
        "behavior": behavior,
        "source_cid": source,
        "residuals": residuals,
        "max_rel_l2": max((r["rel_l2"] for r in residuals), default=float("nan")),
        "pass": cell_pass if assert_gate else None,
    }
    logger.info(
        "A0 cross-era %s/%s: max rel L2 = %.3e (threshold %.0e, asserted=%s)",
        behavior,
        source,
        result["max_rel_l2"],
        A0_CROSS_ERA_REL_L2,
        assert_gate,
    )
    return result


def _phase_b_smoke(
    base,
    trained,
    tok,
    registry,
    demos,
    behavior,
    source,
    seed,
    targets,
    probes,
    rplus_texts,
    layers,
    device,
    gauge,
    smoke_out: Path,
    *,
    gen_backend: str,
) -> dict:
    """One-cell Phase-B end-to-end: extraction + npz write + store-schema join."""
    import issue722_load_activations as loader722

    cell_dir = smoke_out / behavior / f"{source}_seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for tj, tcid in enumerate(targets):
        rows = [(qi, q, rplus_texts[tj * len(probes) + qi]) for qi, q in enumerate(probes)]
        n_written += _extract_and_write_target(
            base,
            trained,
            tok,
            registry,
            demos,
            behavior,
            source,
            seed,
            tcid,
            rows,
            layers,
            device,
            cell_dir,
            gauge,
            gen_backend=gen_backend,
        )
    # Store-schema join through the #722 loader units (R7): filename parse +
    # blob validation must both round-trip on the freshly written npz.
    layout = loader722.list_store_layout_local(smoke_out, behaviors=(behavior,))
    src_dir = f"{source}_seed{seed}"
    files = layout[behavior][src_dir]
    by_target = loader722._parse_cell_files(src_dir, files, tuple(layers))
    streamer = loader722._Streamer(local_root=smoke_out)
    n_joined = 0
    for _stem, per_layer in by_target.items():
        for li, rel in per_layer.items():
            blob = streamer.load(f"{behavior}/{rel}")
            rec = loader722._blob_to_record(blob, rel, behavior, li)
            assert rec.source_cid == source and rec.layer == li
            assert "v0_onpolicy" in blob and "v_plus_onpolicy" in blob, sorted(blob)
            n_joined += 1
    ok = n_written == len(targets) * len(layers) and n_joined == n_written
    logger.info("A0 Phase-B smoke: wrote %d npz, joined %d, pass=%s", n_written, n_joined, ok)
    return {"n_npz_written": n_written, "n_joined": n_joined, "pass": ok}


# ─────────────────────────────────────────────────────────────────────────────
# Stage A — on-policy generation (Phase A)
# ─────────────────────────────────────────────────────────────────────────────


def _gen_json_path(out_root: Path, behavior: str, source: str, seed: int) -> Path:
    return out_root / "raw_completions" / "generation" / behavior / f"{source}_seed{seed}.json"


def stage_generate(args) -> int:
    """Phase A: greedy R⁺ from each source-adapter-loaded model; persist per cell."""
    out_root = Path(args.out)
    behaviors = BEHAVIORS if args.behavior == "all" else (args.behavior,)
    device = x667._device(args.gpu_id, args.cpu_only)
    registry, demos, tok = _load_inputs()

    # Cell worklist with resume-skip (a cell whose JSON exists is complete —
    # the JSON is written atomically per cell, immediately after generation).
    cells: list[tuple[str, str]] = []
    for behavior in behaviors:
        for source in _sources_for(behavior):
            if args.source_cid and source != args.source_cid:
                continue
            if _gen_json_path(out_root, behavior, source, args.seed).exists():
                logger.info("generate resume-skip: %s/%s JSON exists", behavior, source)
                continue
            cells.append((behavior, source))
    logger.info("Phase A: %d cells to generate (backend=%s)", len(cells), args.gen_backend)
    if not cells:
        return 0

    engine = None
    if args.gen_backend == "vllm":
        if device.type == "cpu":
            raise RuntimeError("--gen-backend vllm requires a GPU; use --gen-backend peft")
        engine = _VllmLoraEngine()
    try:
        for behavior, source in cells:
            _generate_one_cell(args, engine, tok, registry, demos, behavior, source, device)
    finally:
        if engine is not None:
            engine.shutdown()
    return 0


def _generate_one_cell(args, engine, tok, registry, demos, behavior, source, device) -> None:
    """Generate + IMMEDIATELY persist one cell's greedy R⁺ (checkpoint-per-phase)."""
    out_root = Path(args.out)
    t0 = time.time()
    subfolder, adapter_dir, gauge = _resolve_adapter(behavior, source, args.seed)
    targets = _targets_for(behavior, source)
    probes = x667.load_eval_probes(behavior)
    if args.max_probes:
        probes = probes[: args.max_probes]
    if args.max_targets:
        targets = targets[: args.max_targets]
    prompts, keys = [], []
    for tcid in targets:
        for qi, q in enumerate(probes):
            prompts.append(_prompt_text(tok, registry, demos, tcid, behavior, q))
            keys.append((tcid, qi))
    if engine is not None:
        ids = engine.greedy(prompts, max_new_tokens=args.max_new_tokens, adapter_dir=adapter_dir)
        texts = [tok.decode(i, skip_special_tokens=True) for i in ids]
    else:  # peft fallback (plan §8 R1) — same loading path as extraction
        dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        _, base, trained = x667.load_base_and_trained(adapter_dir, device, dtype)
        ids = _peft_batched_greedy(
            trained, tok, prompts, max_new_tokens=args.max_new_tokens, device=device
        )
        texts = [tok.decode(i, skip_special_tokens=True) for i in ids]
        del base, trained
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows = [
        {
            "target_cid": tcid,
            "probe_idx": qi,
            "probe": probes[qi],
            "response": text,
            "resp_sha256": _sha256(text),
        }
        for (tcid, qi), text in zip(keys, texts, strict=True)
    ]
    n_empty = sum(1 for r in rows if not r["response"].strip())
    payload = {
        "behavior": behavior,
        "source_cid": source,
        "seed": args.seed,
        "gen_backend": args.gen_backend,
        "adapter_subfolder": subfolder,
        "adapter_gauge": gauge,
        "sampling": {"temperature": 0.0, "max_tokens": args.max_new_tokens},
        "n_targets": len(targets),
        "n_probes": len(probes),
        "n_empty": n_empty,
        "ts": _now(),
        "responses": rows,
    }
    path = _write_json(_gen_json_path(out_root, behavior, source, args.seed), payload)
    logger.info(
        "[phase=generate] cell %s/%s: %d generations (%d empty) in %.0fs -> %s",
        behavior,
        source,
        len(rows),
        n_empty,
        time.time() - t0,
        path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage B — 2-leg paired extraction (Phase B)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_and_write_target(
    base,
    trained,
    tok,
    registry,
    demos,
    behavior,
    source,
    seed,
    tcid,
    rows,
    layers,
    device,
    cell_dir: Path,
    gauge: dict,
    *,
    gen_backend: str,
) -> int:
    """Teacher-force each R⁺ through BOTH models; write one npz per layer.

    Mirrors #667's ``_extract_one_target`` writer block: same naming
    ``{tcid}_L{li}.npz``, same mean-over-probes (3584,) float32 convention. NEW
    arrays: ``v_plus_onpolicy`` (v⁺(R⁺)) / ``v0_onpolicy`` (v0(R⁺)) — ALSO
    aliased to ``v0``/``v_plus`` with the store's context vectors copied from
    the PINNED #667 store revision, so ``issue722_load_activations`` joins the
    new store as a drop-in second loader (plan §4 code table). Returns the
    number of npz written.
    """
    acc: dict[int, list[list[np.ndarray]]] = {li: [[], []] for li in layers}
    shas: list[str] = []
    for _qi, q, r in rows:
        if not r.strip():
            continue
        msgs = x667.build_messages_for(registry, demos, tcid, behavior, q)
        per_layer = x667._mean_resp_acts(base, trained, tok, msgs, r, layers, device)
        shas.append(_sha256(r))
        for li in layers:
            v0, vp = per_layer[li]
            acc[li][0].append(v0)
            acc[li][1].append(vp)
    n_written = 0
    for li in layers:
        if not acc[li][0]:
            logger.warning("no non-empty R⁺ for target=%s layer=%d — npz skipped", tcid, li)
            continue
        stored = _stage_store_npz(behavior, source, seed, tcid, li)
        v0_on = np.stack(acc[li][0]).mean(axis=0).astype(np.float32)
        vp_on = np.stack(acc[li][1]).mean(axis=0).astype(np.float32)
        payload = {
            # canonical new legs (plan §4 Phase B)
            "v0_onpolicy": v0_on,
            "v_plus_onpolicy": vp_on,
            # loader-compatible aliases: a second issue722 loader over THIS store
            # yields CellRecord(v0=v0(R⁺), vplus=v⁺(R⁺), c0/cplus = store context)
            "v0": v0_on,
            "v_plus": vp_on,
            # context vectors REUSED from the pinned #667 store (unchanged by the
            # manipulation — plan §4: zero re-extraction on the input side of M)
            "c_C": np.asarray(stored["c_C"], dtype=np.float32),
            "c_Cp": np.asarray(stored["c_Cp"], dtype=np.float32),
            "c_C_postft": np.asarray(stored["c_C_postft"], dtype=np.float32),
            "c_Cp_postft": np.asarray(stored["c_Cp_postft"], dtype=np.float32),
            "behavior": behavior,
            "source_cid": source,
            "target_cid": tcid,
            "seed": seed,
            "layer": li,
            "n_probes": len(acc[li][0]),
            "resp_sha256": np.array(shas),
            "adapter_gauge": json.dumps(gauge),
            "gen_backend": gen_backend,
            "store_revision_pin": STORE_REVISION,
        }
        np.savez(cell_dir / f"{tcid}_L{li}.npz", **payload)
        n_written += 1
    return n_written


def stage_extract(args) -> int:
    """Phase B: per-cell 2-leg extraction from the persisted Phase-A rollout text."""
    out_root = Path(args.out)
    behaviors = BEHAVIORS if args.behavior == "all" else (args.behavior,)
    device = x667._device(args.gpu_id, args.cpu_only)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    registry, demos, tok = _load_inputs()
    layers = list(args.layers)
    tensors_root = out_root / "analysis_tensors"

    for behavior in behaviors:
        for source in _sources_for(behavior):
            if args.source_cid and source != args.source_cid:
                continue
            cell_dir = tensors_root / behavior / f"{source}_seed{args.seed}"
            if (cell_dir / x667.CELL_DONE_SENTINEL).exists():
                logger.info("extract resume-skip: %s/%s .done exists", behavior, source)
                continue
            gen_path = _gen_json_path(out_root, behavior, source, args.seed)
            if not gen_path.exists():
                raise FileNotFoundError(
                    f"Phase-A rollout JSON missing for {behavior}/{source}: {gen_path} — "
                    "run --stage generate first (fail loud, no silent skip)"
                )
            gen = json.loads(gen_path.read_text())
            _, adapter_dir, gauge = _resolve_adapter(behavior, source, args.seed)
            _, base, trained = x667.load_base_and_trained(adapter_dir, device, dtype)
            cell_dir.mkdir(parents=True, exist_ok=True)
            by_target: dict[str, list] = {}
            for row in gen["responses"]:
                by_target.setdefault(row["target_cid"], []).append(
                    (row["probe_idx"], row["probe"], row["response"])
                )
            t0 = time.time()
            targets = list(by_target)
            for tcid in targets:
                _extract_and_write_target(
                    base,
                    trained,
                    tok,
                    registry,
                    demos,
                    behavior,
                    source,
                    args.seed,
                    tcid,
                    sorted(by_target[tcid]),
                    layers,
                    device,
                    cell_dir,
                    gauge,
                    gen_backend=gen.get("gen_backend", "vllm"),
                )
            # Sentinel only after the full complement is on disk (#667 round-8).
            x667.assert_full_npz_complement(cell_dir, targets=targets, layers=layers)
            x667.write_cell_done_sentinel(
                cell_dir,
                behavior=behavior,
                source_cid=source,
                seed=args.seed,
                targets=targets,
                layers=layers,
            )
            logger.info(
                "[phase=extract] cell %s/%s: %d targets x %d layers in %.0fs",
                behavior,
                source,
                len(targets),
                len(layers),
                time.time() - t0,
            )
            del base, trained
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Stage C — uploads BEFORE compute release (Phase C)
# ─────────────────────────────────────────────────────────────────────────────


def _list_repo_files_retry(repo_id: str, *, repo_type: str, attempts: int = 5) -> list[str]:
    """list_repo_files with transient-5xx retry (paginated tree 504s un-retried upstream)."""
    from huggingface_hub import list_repo_files
    from huggingface_hub.errors import HfHubHTTPError

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return list(list_repo_files(repo_id, repo_type=repo_type))
        except HfHubHTTPError as e:
            last = e
            wait = 2**attempt
            logger.warning("list_repo_files failed (%s); retry %d in %ds", e, attempt + 1, wait)
            time.sleep(wait)
    raise RuntimeError(f"list_repo_files({repo_id}) failed after {attempts} attempts") from last


def stage_upload(args) -> int:
    """Phase C: bulk upload_folder (one commit per behavior + one raw-completions
    commit), then a list_repo_files count verification (4,320 npz expected)."""
    from huggingface_hub import HfApi

    out_root = Path(args.out)
    api = HfApi()

    raw_dir = out_root / "raw_completions" / "generation"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw completions dir missing: {raw_dir}")
    n_raw = len(list(raw_dir.rglob("*.json")))
    logger.info("[phase=upload] raw completions: %d JSONs -> %s", n_raw, RAW_PREFIX)
    api.upload_folder(
        folder_path=str(raw_dir),
        path_in_repo=RAW_PREFIX,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*.json"],
        commit_message="issue-833: on-policy generation raw completions",
    )

    tensors_root = out_root / "analysis_tensors"
    n_local = 0
    for behavior in BEHAVIORS:
        beh_dir = tensors_root / behavior
        if not beh_dir.is_dir():
            raise FileNotFoundError(f"analysis tensors dir missing: {beh_dir}")
        npzs = list(beh_dir.rglob("*.npz"))
        n_local += len(npzs)
        logger.info(
            "[phase=upload] %s: %d npz -> %s/%s", behavior, len(npzs), TENSORS_PREFIX, behavior
        )
        # ONE bulk upload_folder commit per behavior (never a per-file loop —
        # upload-policy 504-storm rule; 1,440 files/behavior < the 10k dir cap).
        api.upload_folder(
            folder_path=str(beh_dir),
            path_in_repo=f"{TENSORS_PREFIX}/{behavior}",
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            allow_patterns=["*.npz"],
            commit_message=f"issue-833: on-policy analysis tensors ({behavior})",
        )

    files = _list_repo_files_retry(HF_DATA_REPO, repo_type="dataset")
    n_npz = sum(1 for f in files if f.startswith(f"{TENSORS_PREFIX}/") and f.endswith(".npz"))
    n_raw_hub = sum(1 for f in files if f.startswith(f"{RAW_PREFIX}/") and f.endswith(".json"))
    expected = args.expected_npz if args.expected_npz is not None else EXPECTED_NPZ_TOTAL
    logger.info(
        "[phase=upload] verify: %d npz on Hub (expected %d), %d raw JSONs (local %d)",
        n_npz,
        expected,
        n_raw_hub,
        n_raw,
    )
    counts = {"npz_on_hub": n_npz, "npz_expected": expected, "raw_json_on_hub": n_raw_hub}
    _write_json(out_root / "upload_verification.json", {**counts, "ts": _now()})
    _write_phase_sentinel("upload-verified", counts)
    if n_npz != expected or n_raw_hub < n_raw:
        raise RuntimeError(f"upload verification FAILED: {counts} (local raw={n_raw})")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Finalize — results sentinel (pod-side; poller drains it into epm:results)
# ─────────────────────────────────────────────────────────────────────────────


def stage_finalize(args) -> int:
    """Compose the final results sentinel from the on-disk stage artifacts."""
    from explore_persona_space.experiments.issue_651 import resolve_adapter_subfolder

    out_root = Path(args.out)
    a0 = json.loads((out_root / "parity" / "a0_summary.json").read_text())
    upload = json.loads((out_root / "upload_verification.json").read_text())

    adapter_paths = {
        f"{beh}_{src}_seed{args.seed}": resolve_adapter_subfolder(beh, src, args.seed)
        for beh in BEHAVIORS
        for src in _sources_for(beh)
    }
    n_gen_cells = len(list((out_root / "raw_completions" / "generation").rglob("*.json")))
    backend = a0.get("backend_parity", {})
    cross = a0.get("cross_era", {})
    note = {
        "eval_numbers": {
            "npz_on_hub": upload["npz_on_hub"],
            "npz_expected": upload["npz_expected"],
            "raw_completion_cells": n_gen_cells,
            "a0_backend_parity_pass": backend.get("pass"),
            "a0_backend_parity_frac_identical": [
                r["frac_identical"] for r in backend.get("results", [])
            ],
            "a0_cross_era_pass": cross.get("pass"),
            "a0_cross_era_max_rel_l2": [r["max_rel_l2"] for r in cross.get("results", [])],
            "a0_phase_b_smoke_pass": (a0.get("phase_b_smoke") or {}).get("pass"),
        },
        "eval_paths": [
            "eval_results/issue_833/parity/a0_summary.json",
            "eval_results/issue_833/parity/a0_cross_era.json",
            "eval_results/issue_833/upload_verification.json",
        ],
        "reproducibility_card": {
            "adapter_paths": adapter_paths,
            "adapter_paths_pattern": (
                "adapters/i537_{behavior}_{source}_seed42 (+ /sft_em_adapter for em)"
            ),
            "hf_data_repo": HF_DATA_REPO,
            "hf_data_repo_prefixes": {
                "raw_completions": RAW_PREFIX,
                "analysis_tensors": TENSORS_PREFIX,
                "reused_store": f"{STORE_PREFIX} @ {STORE_REVISION}",
            },
            "store_revision_pin": STORE_REVISION,
            "gen_backend": backend.get("gen_backend", "vllm"),
            "seeds": [args.seed],
            "layers": list(args.layers),
            "max_new_tokens": args.max_new_tokens,
        },
        "wandb_url": "n/a — no training",
        "hf_hub_url": f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/{HF_PREFIX}",
        "worktree_path": args.worktree_path or str(PROJECT_ROOT),
        "final_commit_sha": args.final_commit_sha or "unknown",
        "gpu_hours_used": args.gpu_hours_used,
        "gpu_hours_budgeted": 13,
        "plan_deviations": [],
    }
    sentinel = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 833,
        "ts": _now(),
        "note": note,
    }
    # Prompt-specified fixed name + the poller-canonical epoch-suffixed name.
    _write_phase_sentinel("results", sentinel)
    try:
        _write_json(SENTINEL_DIR / f"issue-833-epm_results-{int(time.time())}.json", sentinel)
    except OSError as e:
        logger.warning("poller-canonical results sentinel not written (%s)", e)
    _write_json(out_root / "results_sentinel_copy.json", sentinel)
    logger.info("[phase=finalize] results sentinel written")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _sub(args, stage: str, extra: list[str]) -> int:
    """Re-invoke this script in a SUBPROCESS for one stage (vLLM/HF isolation, R3)."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--behavior",
        args.behavior,
        "--seed",
        str(args.seed),
        "--layers",
        *[str(li) for li in args.layers],
        "--out",
        args.out,
        "--gpu-id",
        str(args.gpu_id),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--gen-backend",
        args.gen_backend,
        *extra,
    ]
    if args.cpu_only:
        cmd.append("--cpu-only")
    logger.info("[stage=all] subprocess: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #833 on-policy extraction (A0/A/B/C).")
    parser.add_argument("--behavior", default="all", choices=[*BEHAVIORS, "all"])
    parser.add_argument("--source-cid", default=None, help="restrict to one source cell")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS_DEFAULT))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=x667.N_GEN_TOKENS)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["parity", "generate", "extract", "upload", "finalize", "all"],
    )
    parser.add_argument("--gen-backend", default="vllm", choices=["vllm", "peft"])
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="A0 mechanics-only CPU smoke: 1 cell, 1 target, 2 probes, synthetic R "
        "(cross-era residuals reported UN-asserted; backend gate skipped)",
    )
    parser.add_argument("--max-probes", type=int, default=0, help="cap probes (smoke)")
    parser.add_argument("--max-targets", type=int, default=0, help="cap targets (smoke)")
    parser.add_argument("--expected-npz", type=int, default=None, help="upload-verify override")
    parser.add_argument("--gpu-hours-used", type=float, default=0.0)
    parser.add_argument("--final-commit-sha", default=None)
    parser.add_argument("--worktree-path", default=None)
    args = parser.parse_args()
    if args.max_probes == 0:
        args.max_probes = None
    if args.max_targets == 0:
        args.max_targets = None

    t0 = time.time()
    if args.stage == "parity":
        rc = stage_parity(args)
    elif args.stage == "generate":
        rc = stage_generate(args)
    elif args.stage == "extract":
        rc = stage_extract(args)
    elif args.stage == "upload":
        rc = stage_upload(args)
    elif args.stage == "finalize":
        rc = stage_finalize(args)
    else:  # all — subprocess per stage (vLLM-before-transformers isolation, R3)
        rc = _sub(args, "parity", ["--cpu-smoke"] if args.cpu_smoke else [])
        if rc == 2:
            logger.warning("A0 backend parity FAIL -> peft fallback (plan §8 R1); re-running A0")
            args.gen_backend = "peft"
            rc = _sub(args, "parity", ["--cpu-smoke"] if args.cpu_smoke else [])
        if rc != 0:
            logger.error("A0 gates FAIL (rc=%d) — aborting before Phase A", rc)
            return rc
        for stage in ("generate", "extract", "upload"):
            rc = _sub(args, stage, [])
            if rc != 0:
                logger.error("stage %s FAILED rc=%d", stage, rc)
                return rc
    logger.info("stage=%s wall=%.1fs rc=%d", args.stage, time.time() - t0, rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
