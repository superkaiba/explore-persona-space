#!/usr/bin/env python
"""Issue #833 — on-policy answer-profile generation + SAME-ERA 4-leg extraction.

Phases A0′/A1/A2/B1/B2/C of plan v6 (tasks/approved/833/plans/plan.md §4). The
Phase-D fit driver is the SIBLING script ``issue833_fit_onpolicy.py`` (NOT this
file). Plan v5/v6 registered the SAME-ERA redesign after the v4 A0 gates FAILED
live (att-20260702-090420): the token-identity backend-parity gate proved
unsatisfiable (greedy text is engine-relative on vllm 0.11.0 / torch 2.8), and
the cross-era R-leg gate FIRED (the #667 store's v0/v_plus response legs are not
reproducible-era artifacts — rel-L2 0.041–0.115, uniform text-drift signature).
ALL FOUR measurement legs are now generated + extracted in one run on one stack.

Stages (``--stage``):
  parity        — A0′ gate set (plan v6 §4, all HARD gates run BEFORE any fleet
                  stage): (1) rsLoRA gauge asserts (kept); (2) behavioral
                  adapter-effect HARD gate — exact-match(R⁺, R_base′) fraction
                  < 0.9 per A0 cell, with the K1/K1b debug round (PeftModel
                  first-token logit probe → case B = vLLM-LoRA plumbing → exit
                  2; case C after ONE probe-set widening = adapter/probe-surface
                  invalid → K1b exit 4 — NEVER threshold-refined past); (3) c_C
                  stack-parity HARD probe vs the PINNED store (rel-L2 < 1e-3;
                  FAIL → ``reextract_context_vectors`` flag + proceed — the
                  registered in-run contingency, NOT an abort); (4) in-process
                  determinism assert (doubled generation+extraction; FAIL → K4
                  exit 5); (5) 1-cell Phase-B smoke writing BOTH npz namespaces
                  + a full ``--legs-mode reextracted`` join through the Phase-D
                  loader. The v4 token-identity backend gate is RETIRED — its
                  stats persist as the NON-GATING ``backend_divergence_
                  diagnostic``; the v4 cross-era R-leg gate is RETIRED (it
                  fired; the same-era design is its successor). A NON-GATING
                  PeftModel teacher-forced mean-logprob diagnostic also lands in
                  ``a0_summary.json``.
  rbase         — A1: fleet-wide R_base′ regeneration on the CURRENT stack
                  (vLLM PLAIN engine, greedy). R_base′ is source-independent, so
                  ONE pass over the unique (behavior, target, probe) grid fans
                  out to per-(behavior, source) JSONs; sha256s thread into the
                  extraction npz. R_base′ REPLACES the store-era R_base as the
                  base-text leg everywhere downstream (plan v5 amendment i).
  generate      — A2: greedy on-policy R⁺ from each source-adapter-loaded model
                  (vLLM multi-LoRA; ``--gen-backend peft`` = manual batched
                  PeftModel option). Rollout text persists per cell IMMEDIATELY.
  extract-rbase — B1: teacher-force each R_base′ through BOTH θ0 and θ⁺ and
                  write npz to the SECOND namespace ``analysis_tensors_rbase``
                  with the SAME-ERA L1/L2 legs ``v0_rbase``/``v_plus_rbase``
                  (+ ``v0``/``v_plus`` aliases), ``probe_idx``, ``resp_sha256``
                  (of R_base′). Per-cell ``.done`` sentinels + resume-skip.
  extract-context — the registered c_C-parity-FAIL contingency: re-extract
                  c_C / c_C_postft fleet-wide (per source, last-input-token,
                  #667 recipe) into ``analysis_tensors_rbase/{behavior}/
                  {source}_seed{seed}/__context__.npz``; after it, zero
                  old-store reads remain. Invoked by run_all ONLY when the A0′
                  summary carries ``reextract_context_vectors: true``.
  extract       — B2: teacher-force each R⁺ through BOTH models and write npz to
                  ``analysis_tensors`` with the NEW legs ``v_plus_onpolicy`` /
                  ``v0_onpolicy`` (+ loader-compatible ``v0``/``v_plus`` aliases
                  and the store's context vectors copied from the PINNED store
                  revision).
  upload        — C: raw completions (generation + rbase buckets) + BOTH npz
                  namespaces → HF data repo (bulk ``upload_folder``), verified
                  with a ``list_repo_files`` count (4,320 + 4,320 = 8,640 npz
                  expected, + 48 context npz if the contingency fired).
  finalize      — compose the pod-side results sentinel (A0′ gate verdicts +
                  per-namespace counts + structured reproducibility card).
  all           — subprocess-sequences parity → rbase → generate →
                  extract-rbase → [extract-context if flagged] → extract →
                  upload (separate processes: vLLM/transformers isolation, R3).
                  There is NO automatic peft-fallback A0 re-run (the retired
                  token-identity gate was its trigger; ``--gen-backend peft``
                  stays a manual option only).
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
import tempfile
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
RBASE_PREFIX = f"{HF_PREFIX}/raw_completions/rbase"  # fit driver --rbase-completions-prefix
TENSORS_PREFIX = f"{HF_PREFIX}/analysis_tensors"  # B2: R⁺ legs (L3/L4)
RBASE_TENSORS_PREFIX = f"{HF_PREFIX}/analysis_tensors_rbase"  # B1: same-era L1/L2 legs
CONTEXT_NPZ_NAME = "__context__.npz"  # extract-context per-source file (no _L{li} suffix)
# EVERY remaining #667-store read threads this revision (plan v6 §10 — context
# vectors + the A0′ c_C-parity reference ONLY; the store's response legs are
# ABANDONED, plan v5 amendment i).
STORE_REVISION = "0031fc55a0e965c33be4261287cd5c86393ca161"
STORE_PREFIX = "issue667_gate_chain_preview/analysis_tensors"

BEHAVIORS = ("em", "sycophancy", "fact")
LAYERS_DEFAULT = (7, 14, 21)
SEED_DEFAULT = 42
EXPECTED_NPZ_PER_NAMESPACE = 3 * 16 * 30 * 3  # behaviors × sources × targets × layers = 4,320
EXPECTED_CONTEXT_NPZ = 3 * 16  # one __context__.npz per (behavior, source) — contingency only

# A0′ gate thresholds (plan v6 §4 / §7).
A0_NEAR_TIE_LOGIT_GAP = 0.05  # backend-divergence DIAGNOSTIC only (v4 gate retired)
A0_BEHAVIORAL_EXACT_MATCH_MAX = 0.9  # HARD: exact-match(R⁺, R_base′) fraction must be < this
# First-token |Δlogit| (adapter-on vs adapter-off) below this = "no logit shift"
# (case C input). Pre-registered weak threshold — ungrounded, needs-smoke-test
# by design (plan §11); refinable at A0 with a logged Decision, but NEVER used
# to pass case C (plan §13).
A0_ADAPTER_LOGIT_SHIFT_MIN = 0.05
A0_CC_PARITY_REL_L2 = 1e-3  # c_C stack-parity threshold (plan v6 §4 gate 4 / §11)

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
    """One vLLM engine serving base + all adapters via per-request LoRARequest.

    ``enable_lora=False`` builds a PLAIN base engine — ``LLM(model=BASE,
    dtype="bfloat16")`` with NO LoRA kwargs — byte-matching #667's
    ``vllm_generate_R`` engine construction. Base-model legs that feed a parity
    gate against #667-era outputs (the A0 cross-era R_base regeneration, the
    fleet-wide ``--stage rbase``) MUST use this plain form: a LoRA-enabled
    engine can select different kernels even for base (no-adapter) requests,
    and a kernel divergence would false-fail the 1e-3 cross-era gate.
    """

    def __init__(self, gpu_mem_util: float = 0.85, *, enable_lora: bool = True):
        from vllm import LLM

        self.enable_lora = enable_lora
        if enable_lora:
            self._rslora_scaling_line_asserted = _assert_installed_vllm_rslora()
            self.llm = LLM(
                model=x667.BASE_MODEL,
                dtype="bfloat16",
                enable_lora=True,
                max_lora_rank=32,
                gpu_memory_utilization=gpu_mem_util,
            )
        else:  # plain base engine — matches issue667_extract.vllm_generate_R
            self.llm = LLM(
                model=x667.BASE_MODEL,
                dtype="bfloat16",
                gpu_memory_utilization=gpu_mem_util,
            )
        self._next_lora_id = 1
        self._lora_ids: dict[str, int] = {}

    def engine_config(self) -> dict:
        """Reproducibility record of how this engine was constructed."""
        return {
            "backend": "vllm",
            "model": x667.BASE_MODEL,
            "dtype": "bfloat16",
            "enable_lora": self.enable_lora,
            "construction": (
                "LLM(..., enable_lora=True, max_lora_rank=32)"
                if self.enable_lora
                else "plain LLM (no LoRA kwargs) — matches issue667_extract.vllm_generate_R"
            ),
        }

    def _lora_request(self, adapter_dir: Path | None):
        if adapter_dir is None:
            return None
        if not self.enable_lora:
            raise RuntimeError("adapter request on a plain (enable_lora=False) engine")
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


def stage_parity(args) -> int:  # noqa: C901 — the A0′ gate SEQUENCE is the spec (plan v6 §4); flattening would inline the per-gate helpers
    """A0′ gate set (plan v6 §4): behavioral adapter-effect + c_C stack-parity +
    determinism + two-namespace Phase-B smoke; backend-divergence + PeftModel
    mean-logprob persist as NON-GATING diagnostics.

    Exit codes: 0 PASS; 2 K1 (case B — vLLM-LoRA plumbing not applying);
    3 other gate FAIL (Phase-B smoke / join); 4 K1b (case C —
    adapter/probe-surface invalid, after the single widening); 5 K4
    (in-process determinism FAIL). A c_C-parity FAIL never aborts — it sets
    ``reextract_context_vectors: true`` in the summary (registered in-run
    contingency, plan §13 allowed-without-asking).
    """
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

    # ── vLLM sessions (GPU only): LoRA-engine R⁺ + plain-engine R_base′ over the
    # SAME A0 grid (matched arms) + the generation-determinism doubling; both
    # engines reaped before any HF model load.
    smoke_cell = cells[0]  # the fact cell smokes Phase B end-to-end
    use_vllm = device.type != "cpu" and args.gen_backend == "vllm"
    det_gen: dict | None = None
    if use_vllm:
        vllm_ids, rbase_ids, engine_config, det_gen = _a0_vllm_generations(
            cell_meta, tok, registry, demos, args, smoke_cell
        )
    else:
        vllm_ids, rbase_ids = {}, {}
        engine_config = (
            {"backend": "peft-batched-greedy", "note": "manual --gen-backend peft option"}
            if (device.type != "cpu" and args.gen_backend == "peft")
            else {"backend": "synthetic-cpu-smoke", "note": "gates un-asserted"}
        )

    k1_fail = k1b_fail = k4_fail = other_fail = False
    behavioral_results: list[dict] = []
    needs_widening: list[tuple[str, str]] = []
    backend_diag_results: list[dict] = []
    cc_results: list[dict] = []
    logprob_diag: list[dict] = []
    det_extract: dict | None = None
    smoke_result: dict | None = None

    # ── per-cell HF phase (θ0 + θ⁺ loaded once per cell).
    for key, meta in cell_meta.items():
        behavior, source = key
        _, base, trained = x667.load_base_and_trained(meta["adapter_dir"], device, dtype)

        # Texts for the behavioral gate + diagnostics (same A0 grid, both arms).
        if use_vllm:
            rplus_texts = [tok.decode(i, skip_special_tokens=True) for i in vllm_ids[key]]
            rbase_texts = [tok.decode(i, skip_special_tokens=True) for i in rbase_ids[key]]
        elif device.type != "cpu" and args.gen_backend == "peft":
            rp_ids = _peft_batched_greedy(
                trained, tok, meta["prompts"], max_new_tokens=args.max_new_tokens, device=device
            )
            rb_ids = _peft_batched_greedy(
                base, tok, meta["prompts"], max_new_tokens=args.max_new_tokens, device=device
            )
            rplus_texts = [tok.decode(i, skip_special_tokens=True) for i in rp_ids]
            rbase_texts = [tok.decode(i, skip_special_tokens=True) for i in rb_ids]
            if key == smoke_cell and det_gen is None:
                second = _peft_batched_greedy(
                    trained,
                    tok,
                    meta["prompts"][: len(meta["probes"])],
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                det_gen = {
                    "n_prompts": len(second),
                    "pass": all(
                        list(a) == list(b)
                        for a, b in zip(rp_ids[: len(second)], second, strict=True)
                    ),
                    "backend": "peft",
                }
        else:  # cpu_smoke: synthetic texts — mechanics only, gates un-asserted
            grid = len(meta["targets"]) * len(meta["probes"])
            rplus_texts = ["Synthetic on-policy smoke response, two sentences long."] * grid
            rbase_texts = ["Synthetic base smoke response, one sentence."] * grid

        # (2) behavioral adapter-effect HARD gate (plan v6 §4 gate 2; K1/K1b).
        n_match = sum(1 for a, b in zip(rplus_texts, rbase_texts, strict=True) if a == b)
        frac = n_match / len(rplus_texts)
        cell_row = {
            "behavior": behavior,
            "source_cid": source,
            "n_prompts": len(rplus_texts),
            "exact_match_frac": frac,
            "threshold": A0_BEHAVIORAL_EXACT_MATCH_MAX,
            "case": None,
            "pass": (None if args.cpu_smoke else frac < A0_BEHAVIORAL_EXACT_MATCH_MAX),
        }
        if not args.cpu_smoke and frac >= A0_BEHAVIORAL_EXACT_MATCH_MAX:
            # Debug round: PeftModel first-token logit probe disambiguates
            # "vLLM LoRA path not applying" (case B) from "adapter genuinely
            # leaves greedy text unchanged on these probes" (widen once → C).
            shift = _first_token_logit_shift(trained, tok, meta["prompts"], device)
            cell_row["logit_shift"] = shift
            if shift["logits_shift"]:
                cell_row["case"] = "B"
                k1_fail = True
                logger.error(
                    "A0 behavioral gate %s/%s: exact-match %.2f >= %.2f WITH PeftModel logit "
                    "shift (max %.3f) — case B: vLLM LoRA plumbing not applying (K1)",
                    behavior,
                    source,
                    frac,
                    A0_BEHAVIORAL_EXACT_MATCH_MAX,
                    shift["max"],
                )
            else:
                needs_widening.append(key)
                logger.warning(
                    "A0 behavioral gate %s/%s: exact-match %.2f >= %.2f with NO logit shift — "
                    "single probe-set widening queued (plan §4 gate 2)",
                    behavior,
                    source,
                    frac,
                    A0_BEHAVIORAL_EXACT_MATCH_MAX,
                )
        behavioral_results.append(cell_row)
        logger.info(
            "A0 behavioral adapter-effect %s/%s: exact-match %.2f (< %.2f required) pass=%s",
            behavior,
            source,
            frac,
            A0_BEHAVIORAL_EXACT_MATCH_MAX,
            cell_row["pass"],
        )

        # NON-GATING backend-divergence diagnostic (the retired v4 gate's stats).
        if key in vllm_ids:
            diag = _backend_divergence_diagnostic(
                trained,
                tok,
                meta["prompts"],
                vllm_ids[key],
                eos_ids,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            backend_diag_results.append({"behavior": behavior, "source_cid": source, **diag})
            logger.info(
                "A0 backend-divergence DIAGNOSTIC %s/%s: %d/%d token-identical (%.2f) — "
                "non-gating (v4 gate retired)",
                behavior,
                source,
                diag["n_token_identical"],
                diag["n_prompts"],
                diag["frac_identical"],
            )

        # (4) c_C stack-parity HARD probe (context-vector reuse license).
        full_probes = x667.load_eval_probes(behavior)
        cc_cell = _cc_parity_probe(
            base,
            trained,
            tok,
            registry,
            demos,
            behavior,
            source,
            args.seed,
            meta["targets"],
            full_probes[0],
            layers,
            device,
            assert_gate=not args.cpu_smoke,
        )
        cc_results.append(cc_cell)

        # (3) NON-GATING PeftModel teacher-forced mean-logprob diagnostic.
        logprob_diag.append(
            {
                "behavior": behavior,
                "source_cid": source,
                **_peft_logprob_diagnostic(
                    base, trained, tok, meta["prompts"], rplus_texts, rbase_texts, device
                ),
            }
        )

        if key == smoke_cell:
            # (5) in-process determinism — extraction half: one (cell, target)
            # teacher-force doubled; identical means required (K4 on FAIL).
            det_extract = _determinism_extraction_check(
                base, trained, tok, registry, demos, behavior, meta, rplus_texts, layers, device
            )
            # (6) two-namespace Phase-B smoke + --legs-mode reextracted join.
            smoke_store_dir = Path(tempfile.mkdtemp(prefix="issue833_a0_smoke_store_"))
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
                rbase_texts,
                layers,
                device,
                meta["gauge"],
                smoke_store_dir,
                gen_backend=args.gen_backend,
            )
            if not smoke_result["pass"]:
                other_fail = True
        del base, trained
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── single registered probe-set widening for no-shift high-exact-match cells.
    widen_results: list[dict] = []
    for key in needs_widening:
        res = _widen_behavioral_round(
            key, args, tok, registry, demos, device, dtype, n_targets, n_probes
        )
        widen_results.append(res)
        for row in behavioral_results:
            if (row["behavior"], row["source_cid"]) == key:
                row["case"] = res["case"]
                row["pass"] = res["pass"]
                row["widened"] = True
        if res["case"] == "B":
            k1_fail = True
        elif res["case"] == "C":
            k1b_fail = True
            logger.error(
                "[epm-failure-ready] failure_class: code reason: adapter_probe_surface_invalid "
                "— K1b: %s/%s exact-match >= %.2f with NO PeftModel logit shift after the "
                "single probe-set widening (adapter engagement uncertifiable)",
                key[0],
                key[1],
                A0_BEHAVIORAL_EXACT_MATCH_MAX,
            )

    # ── gate verdicts.
    behavioral_pass = None if args.cpu_smoke else all(r["pass"] for r in behavioral_results)
    cc_pass = None if args.cpu_smoke else all(r["pass"] for r in cc_results)
    reextract_context = bool(cc_pass is False)
    det_pass = None
    if not args.cpu_smoke:
        det_pass = bool(det_gen and det_gen["pass"]) and bool(det_extract and det_extract["pass"])
        if not det_pass:
            k4_fail = True
            logger.error(
                "[epm-failure-ready] failure_class: code reason: in_process_nondeterminism — "
                "K4: doubled generation/extraction not reproducible (gen=%s extract=%s)",
                det_gen,
                det_extract,
            )
    if reextract_context:
        logger.warning(
            "A0 c_C stack-parity FAIL — reextract_context_vectors flag SET; run_all will "
            "execute --stage extract-context (registered contingency, NOT an abort)"
        )

    gates = {
        "rslora_gauge": {
            "per_adapter": [m["gauge"] for m in cell_meta.values()],
            "installed_vllm_alpha_sqrt_r_asserted": use_vllm,
        },
        "behavioral_adapter_effect": {
            "results": behavioral_results,
            "widen_results": widen_results,
            "pass": behavioral_pass,
            "threshold": A0_BEHAVIORAL_EXACT_MATCH_MAX,
        },
        "cc_stack_parity": {
            "results": cc_results,
            "pass": cc_pass,
            "rel_l2_threshold": A0_CC_PARITY_REL_L2,
        },
        "determinism": {"generation": det_gen, "extraction": det_extract, "pass": det_pass},
        "phase_b_smoke": smoke_result,
        "backend_divergence_diagnostic": {
            "results": backend_diag_results,
            "non_gating": True,
            "note": "v4 token-identity backend-parity gate RETIRED (plan v6 §4 — "
            "unsatisfiable on this stack, att-20260702-090420); stats persist for the analyzer",
        },
        "peft_logprob_diagnostic": {"results": logprob_diag, "non_gating": True},
        "reextract_context_vectors": reextract_context,
        "gen_backend": args.gen_backend,
        "engine_config": engine_config,
        "cpu_smoke": bool(args.cpu_smoke),
        "ts": _now(),
    }
    _write_json(parity_dir / "a0_summary.json", gates)
    _write_json(
        parity_dir / "a0_cc_parity.json",
        {
            "results": cc_results,
            "rel_l2_threshold": A0_CC_PARITY_REL_L2,
            "reextract_context_vectors": reextract_context,
            "ts": _now(),
        },
    )
    hard_pass = not (k1_fail or k1b_fail or k4_fail or other_fail) and behavioral_pass in (
        True,
        None,
    )
    _write_phase_sentinel("a0-gates", {"gates": gates, "pass": hard_pass})
    if k4_fail:
        logger.error("A0′ FAIL (K4 determinism) — aborting before Phase A")
        return 5
    if k1b_fail:
        logger.error("A0′ FAIL (K1b adapter/probe-surface invalid) — aborting before Phase A")
        return 4
    if k1_fail:
        logger.error("A0′ FAIL (K1 vLLM-LoRA plumbing) — aborting before Phase A")
        return 2
    if other_fail:
        logger.error("A0′ FAIL (Phase-B smoke/join) — aborting before Phase A")
        return 3
    logger.info("A0′ gates PASS (reextract_context_vectors=%s)", reextract_context)
    return 0


def _a0_vllm_generations(cell_meta, tok, registry, demos, args, smoke_cell):
    """TWO sequential vLLM engine sessions, each reaped before HF loads.

    Session 1 (LoRA engine): adapter-loaded greedy R⁺ over each cell's A0 grid
    (what Phase A2 uses) + the generation half of the in-process determinism
    assert (the smoke cell's first-target prompts re-submitted to the SAME
    engine; byte-identical token ids required — plan v6 §4 gate 5).
    Session 2 (PLAIN engine, ``enable_lora=False``): R_base′ over the SAME A0
    grid — matched arms (both texts from the same vLLM build, plan v5 amendment
    iii); consumed by the behavioral adapter-effect gate, the PeftModel logprob
    diagnostic, and the Phase-B smoke's rbase-namespace write.

    Returns ``(vllm_ids, rbase_ids, engine_config, det_gen)``.
    """
    vllm_ids: dict[tuple[str, str], list[list[int]]] = {}
    det_second: list[list[int]] = []
    n_det = len(cell_meta[smoke_cell]["probes"])
    engine = _VllmLoraEngine()
    try:
        for key, meta in cell_meta.items():
            vllm_ids[key] = engine.greedy(
                meta["prompts"],
                max_new_tokens=args.max_new_tokens,
                adapter_dir=meta["adapter_dir"],
            )
        det_second = engine.greedy(
            cell_meta[smoke_cell]["prompts"][:n_det],
            max_new_tokens=args.max_new_tokens,
            adapter_dir=cell_meta[smoke_cell]["adapter_dir"],
        )
    finally:
        engine.shutdown()
    det_gen = {
        "n_prompts": n_det,
        "backend": "vllm",
        "pass": all(
            list(a) == list(b)
            for a, b in zip(vllm_ids[smoke_cell][:n_det], det_second, strict=True)
        ),
    }

    rbase_ids: dict[tuple[str, str], list[list[int]]] = {}
    plain = _VllmLoraEngine(enable_lora=False)
    try:
        engine_config = plain.engine_config()
        for key, meta in cell_meta.items():
            rbase_ids[key] = plain.greedy(
                meta["prompts"], max_new_tokens=args.max_new_tokens, adapter_dir=None
            )
    finally:
        plain.shutdown()
    return vllm_ids, rbase_ids, engine_config, det_gen


def _backend_divergence_diagnostic(
    trained, tok, prompts, vllm_ids_cell, eos_ids, *, max_new_tokens, device
) -> dict:
    """NON-GATING vLLM-vs-PeftModel greedy divergence stats for ONE cell.

    The v4 ≥80%-token-identical HARD gate over these stats is RETIRED (plan v6
    §4 — unsatisfiable on vllm 0.11.0 / torch 2.8, att-20260702-090420: greedy
    text is engine-relative). The stats persist in ``a0_summary.json`` under
    ``backend_divergence_diagnostic`` as analyzer-facing evidence only.
    """
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
    return {
        "n_prompts": len(prompts),
        "n_token_identical": n_ident,
        "frac_identical": n_ident / len(prompts),
        "n_non_near_tie": sum(1 for d in divergences if d["logit_gap"] >= A0_NEAR_TIE_LOGIT_GAP),
        "divergences": divergences,
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


# The v4 cross-era R-leg gate (`_cross_era_check`) is DELETED, not kept: it
# FIRED live (att-20260702-090420 — the store's response legs are not
# reproducible-era artifacts) and its successor is the same-era design itself
# (stage extract-rbase) + the c_C stack-parity probe below (plan v6 §4).


@torch.no_grad()
def _first_token_logit_shift(trained, tok, prompts: list[str], device) -> dict:
    """Debug-round probe: per-prompt max |Δ first-token logit|, adapter-on vs off.

    Uses ``PeftModel.disable_adapter()`` so both reads share ONE loaded model.
    A genuinely applied adapter shifts next-token logits by O(0.1–10); an
    unapplied/no-op adapter reads ~0. ``logits_shift`` = any prompt's max-abs
    diff >= ``A0_ADAPTER_LOGIT_SHIFT_MIN`` (case-B/C disambiguator, plan v6 §4
    gate 2).
    """
    shifts: list[float] = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(device)
        on = trained(**ids).logits[0, -1, :].float()
        with trained.disable_adapter():
            off = trained(**ids).logits[0, -1, :].float()
        shifts.append(float((on - off).abs().max().item()))
    return {
        "per_prompt_max_abs": shifts,
        "max": max(shifts),
        "mean": float(np.mean(shifts)),
        "shift_min_threshold": A0_ADAPTER_LOGIT_SHIFT_MIN,
        "logits_shift": max(shifts) >= A0_ADAPTER_LOGIT_SHIFT_MIN,
    }


@torch.no_grad()
def _mean_resp_logprob(model, tok, prompt: str, response: str, device) -> float | None:
    """Teacher-forced mean per-token log P(response | prompt); None on empty response."""
    p_ids = tok.encode(prompt, add_special_tokens=False)
    r_ids = tok.encode(response, add_special_tokens=False)
    if not r_ids:
        return None
    ids = torch.tensor([p_ids + r_ids], dtype=torch.long, device=device)
    logp = torch.log_softmax(model(ids).logits[0, :-1, :].float(), dim=-1)
    rows = logp[len(p_ids) - 1 : len(p_ids) - 1 + len(r_ids)]
    tgt = torch.tensor(r_ids, dtype=torch.long, device=device).unsqueeze(1)
    return float(rows.gather(1, tgt).mean().item())


def _peft_logprob_diagnostic(base, trained, tok, prompts, rplus_texts, rbase_texts, device) -> dict:
    """NON-GATING PeftModel mean-logprob diagnostic (plan v6 §4 gate 3).

    Teacher-forced mean log P of R⁺ and R_base′ under PeftModel(adapter-on) AND
    base — separates off-manifold greedy text (low logP under its nominal
    producer) from benign kernel tie-breaks. Persisted in ``a0_summary.json``
    incl. the trained-vs-base direction per cell; analyzer-facing, never a gate.
    """
    agg: dict[str, list[float]] = {
        "rplus_under_trained": [],
        "rplus_under_base": [],
        "rbase_under_trained": [],
        "rbase_under_base": [],
    }
    for p, r_on, r_b in zip(prompts, rplus_texts, rbase_texts, strict=True):
        for name, model, r in (
            ("rplus_under_trained", trained, r_on),
            ("rplus_under_base", base, r_on),
            ("rbase_under_trained", trained, r_b),
            ("rbase_under_base", base, r_b),
        ):
            lp = _mean_resp_logprob(model, tok, p, r, device)
            if lp is not None:
                agg[name].append(lp)
    out: dict = {k: (float(np.mean(v)) if v else None) for k, v in agg.items()}
    for text_side in ("rplus", "rbase"):
        t, b = out[f"{text_side}_under_trained"], out[f"{text_side}_under_base"]
        out[f"{text_side}_trained_minus_base"] = None if (t is None or b is None) else t - b
    out["n_rows"] = len(agg["rplus_under_trained"])
    return out


def _cc_parity_probe(
    base,
    trained,
    tok,
    registry,
    demos,
    behavior,
    source,
    seed,
    targets,
    src_probe,
    layers,
    device,
    *,
    assert_gate: bool,
) -> dict:
    """c_C stack-parity HARD probe (plan v6 §4 gate 4 — the context-reuse license).

    Re-extracts c_C (θ0) + c_C_postft (θ⁺, adapter-loaded) at the SOURCE's
    last-input-token (#667 recipe: ``src_msgs = build_messages_for(...,
    probes[0])``; stored row mapping ``c_all[li - 1]`` per the #667 writer's
    ``c_idx = li - 1``) and computes rel-L2 vs the PINNED store copy in each
    checked target's npz. Context vectors are TEXT-FREE, so parity should sit
    at extraction-stack numerics (~1e-4–1e-6); text-driven drift lands ≥0.04.
    A FAIL routes to the in-run fleet ``--stage extract-context`` contingency —
    never an abort.
    """
    src_msgs = x667.build_messages_for(registry, demos, source, behavior, src_probe)
    c_all = x667._context_vector_all_layers(base, tok, src_msgs, device)
    cp_all = x667._context_vector_all_layers(trained, tok, src_msgs, device)
    residuals = []
    cell_pass = True
    for tcid in targets:
        for li in layers:
            stored = _stage_store_npz(behavior, source, seed, tcid, li)
            for leg, new_vec in (("c_C", c_all[li - 1]), ("c_C_postft", cp_all[li - 1])):
                ref = np.asarray(stored[leg], dtype=np.float64)
                rel = float(np.linalg.norm(new_vec.astype(np.float64) - ref) / np.linalg.norm(ref))
                ok = rel < A0_CC_PARITY_REL_L2
                cell_pass &= ok
                residuals.append(
                    {"target_cid": tcid, "layer": li, "vector": leg, "rel_l2": rel, "pass": ok}
                )
    result = {
        "behavior": behavior,
        "source_cid": source,
        "residuals": residuals,
        "max_rel_l2": max((r["rel_l2"] for r in residuals), default=float("nan")),
        "pass": cell_pass if assert_gate else None,
    }
    logger.info(
        "A0 c_C stack-parity %s/%s: max rel L2 = %.3e (threshold %.0e, asserted=%s)",
        behavior,
        source,
        result["max_rel_l2"],
        A0_CC_PARITY_REL_L2,
        assert_gate,
    )
    return result


def _determinism_extraction_check(
    base, trained, tok, registry, demos, behavior, meta, rplus_texts, layers, device
) -> dict:
    """Extraction half of the in-process determinism assert (plan v6 §4 gate 5).

    Doubles ``_mean_resp_acts`` on the first non-empty (target, probe) row and
    requires bit-identical means on BOTH legs at every layer (the same-era
    premise: within-run teacher-forcing must be reproducible). FAIL → K4.
    """
    probes = meta["probes"]
    for tj, tcid in enumerate(meta["targets"]):
        for qi, q in enumerate(probes):
            r = rplus_texts[tj * len(probes) + qi]
            if not r.strip():
                continue
            msgs = x667.build_messages_for(registry, demos, tcid, behavior, q)
            first = x667._mean_resp_acts(base, trained, tok, msgs, r, layers, device)
            second = x667._mean_resp_acts(base, trained, tok, msgs, r, layers, device)
            identical = all(
                np.array_equal(first[li][0], second[li][0])
                and np.array_equal(first[li][1], second[li][1])
                for li in layers
            )
            return {"target_cid": tcid, "probe_idx": qi, "pass": identical}
    return {"pass": False, "note": "no non-empty R⁺ row available for the doubling"}


def _widen_behavioral_round(
    key, args, tok, registry, demos, device, dtype, n_targets, n_probes
) -> dict:
    """The SINGLE registered probe-set widening (plan v6 §4 gate 2 / §13).

    Doubles the A0 grid (targets AND probes), regenerates R⁺ + R_base′ on the
    same engine pair, and re-reads exact-match. ``< 0.9`` → gate PASS (case
    None). Still ``>= 0.9`` → re-run the PeftModel logit probe on the widened
    prompts: shift → case B (K1, vLLM plumbing); NO shift → case C (K1b,
    ``adapter/probe-surface invalid``). NEVER widened twice, NEVER
    threshold-refined past case C (a fiat pass would launch the fleet on an
    uncertified on-policy leg).
    """
    behavior, source = key
    logger.info(
        "Decision: single A0 probe-set widening for %s/%s (targets %d->%d, probes %d->%d)",
        behavior,
        source,
        n_targets,
        2 * n_targets,
        n_probes,
        2 * n_probes,
    )
    _, adapter_dir, _ = _resolve_adapter(behavior, source, args.seed)
    full_probes = x667.load_eval_probes(behavior)
    w_targets = _targets_for(behavior, source)[: 2 * n_targets]
    w_probes = full_probes[: 2 * n_probes]
    prompts = [
        _prompt_text(tok, registry, demos, tcid, behavior, q)
        for tcid in w_targets
        for q in w_probes
    ]
    base = trained = None
    try:
        if device.type != "cpu" and args.gen_backend == "vllm":
            engine = _VllmLoraEngine()
            try:
                rp_ids = engine.greedy(
                    prompts, max_new_tokens=args.max_new_tokens, adapter_dir=adapter_dir
                )
            finally:
                engine.shutdown()
            plain = _VllmLoraEngine(enable_lora=False)
            try:
                rb_ids = plain.greedy(prompts, max_new_tokens=args.max_new_tokens, adapter_dir=None)
            finally:
                plain.shutdown()
        else:
            _, base, trained = x667.load_base_and_trained(adapter_dir, device, dtype)
            rp_ids = _peft_batched_greedy(
                trained, tok, prompts, max_new_tokens=args.max_new_tokens, device=device
            )
            rb_ids = _peft_batched_greedy(
                base, tok, prompts, max_new_tokens=args.max_new_tokens, device=device
            )
        rplus = [tok.decode(i, skip_special_tokens=True) for i in rp_ids]
        rbase = [tok.decode(i, skip_special_tokens=True) for i in rb_ids]
        frac = sum(1 for a, b in zip(rplus, rbase, strict=True) if a == b) / len(prompts)
        if frac < A0_BEHAVIORAL_EXACT_MATCH_MAX:
            return {
                "behavior": behavior,
                "source_cid": source,
                "exact_match_frac": frac,
                "n_prompts": len(prompts),
                "case": None,
                "pass": True,
            }
        if trained is None:
            _, base, trained = x667.load_base_and_trained(adapter_dir, device, dtype)
        shift = _first_token_logit_shift(trained, tok, prompts, device)
        case = "B" if shift["logits_shift"] else "C"
        return {
            "behavior": behavior,
            "source_cid": source,
            "exact_match_frac": frac,
            "n_prompts": len(prompts),
            "logit_shift": shift,
            "case": case,
            "pass": False,
        }
    finally:
        if base is not None or trained is not None:
            del base, trained
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
    rbase_texts,
    layers,
    device,
    gauge,
    smoke_out: Path,
    *,
    gen_backend: str,
) -> dict:
    """One-cell end-to-end smoke: BOTH namespace writes + the reextracted join.

    Writes the R⁺ legs (``analysis_tensors``) AND the same-era R_base′ legs
    (``analysis_tensors_rbase``) for one cell, then runs the FULL Phase-D
    ``--legs-mode reextracted`` join through the fit driver's loaders
    (``load_onpolicy_legs`` + ``load_rbase_legs`` + ``build_cells_reextracted``
    + ``join_cells``) on local streamers — asserting the same-era substitution
    actually ENGAGED (joined V0 == the rbase legs) before any fleet GPU-h
    commits (plan v6 §4 gate 6 / R7).
    """
    import issue722_load_activations as loader722
    import issue833_fit_onpolicy as fit833

    on_root = smoke_out / "analysis_tensors"
    rb_root = smoke_out / "analysis_tensors_rbase"
    cell_on = on_root / behavior / f"{source}_seed{seed}"
    cell_rb = rb_root / behavior / f"{source}_seed{seed}"
    cell_on.mkdir(parents=True, exist_ok=True)
    cell_rb.mkdir(parents=True, exist_ok=True)
    rbase_sha_by_probe = {
        (tcid, qi): _sha256(rbase_texts[tj * len(probes) + qi])
        for tj, tcid in enumerate(targets)
        for qi in range(len(probes))
    }
    n_on = n_rb = 0
    for tj, tcid in enumerate(targets):
        rows_on = [(qi, q, rplus_texts[tj * len(probes) + qi]) for qi, q in enumerate(probes)]
        rows_rb = [(qi, q, rbase_texts[tj * len(probes) + qi]) for qi, q in enumerate(probes)]
        n_on += _extract_and_write_target(
            base,
            trained,
            tok,
            registry,
            demos,
            behavior,
            source,
            seed,
            tcid,
            rows_on,
            layers,
            device,
            cell_on,
            gauge,
            gen_backend=gen_backend,
            base_sha_by_probe=rbase_sha_by_probe,
            leg_mode="onpolicy",
        )
        n_rb += _extract_and_write_target(
            base,
            trained,
            tok,
            registry,
            demos,
            behavior,
            source,
            seed,
            tcid,
            rows_rb,
            layers,
            device,
            cell_rb,
            gauge,
            gen_backend=gen_backend,
            base_sha_by_probe=None,
            leg_mode="rbase",
        )
    # Reextracted join through the Phase-D loaders (local streamers).
    on_layout = loader722.list_store_layout_local(on_root, behaviors=(behavior,))
    on_streamer = loader722._Streamer(local_root=on_root)
    legs = fit833.load_onpolicy_legs((behavior,), tuple(layers), on_streamer, on_layout)
    rb_layout = loader722.list_store_layout_local(rb_root, behaviors=(behavior,))
    rb_streamer = loader722._Streamer(local_root=rb_root)
    rlegs = fit833.load_rbase_legs((behavior,), tuple(layers), rb_streamer, rb_layout)
    fit833.assert_rbase_hash_consistency(legs, rlegs)
    # Context source for the join: the R⁺ npz embed the PINNED store's context
    # copies — parse them into CellRecords, then substitute the same-era rbase
    # legs (exactly the production --legs-mode reextracted path).
    src_dir = f"{source}_seed{seed}"
    by_target = loader722._parse_cell_files(src_dir, on_layout[behavior][src_dir], tuple(layers))
    old_cells: list[loader722.CellRecord] = []
    for _stem, per_layer in sorted(by_target.items()):
        for li, rel in sorted(per_layer.items()):
            blob = on_streamer.load(f"{behavior}/{rel}")
            old_cells.append(loader722._blob_to_record(blob, rel, behavior, li))
    n_joined = 0
    substitution_ok = True
    for li in layers:
        cells_li = [c for c in old_cells if c.layer == li]
        subst = fit833.build_cells_reextracted(cells_li, rlegs)
        joined = fit833.join_cells(subst, legs)
        for i, c in enumerate(subst):
            rl = rlegs[(c.behavior, c.source_cid, c.target_cid, c.layer)]
            substitution_ok &= bool(np.allclose(joined["V0"][i], rl.v0))
            substitution_ok &= bool(np.allclose(joined["Vplus"][i], rl.v_plus))
        n_joined += joined["V0"].shape[0]
    ok = (
        n_on == len(targets) * len(layers) and n_rb == n_on and n_joined == n_on and substitution_ok
    )
    logger.info(
        "A0 Phase-B smoke: wrote %d R⁺ npz + %d rbase npz, joined %d (reextracted, "
        "substitution_ok=%s), pass=%s",
        n_on,
        n_rb,
        n_joined,
        substitution_ok,
        ok,
    )
    return {
        "n_npz_written_onpolicy": n_on,
        "n_npz_written_rbase": n_rb,
        "n_joined_reextracted": n_joined,
        "substitution_engaged": substitution_ok,
        "pass": ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage R_base — fleet-wide base-model regeneration (Phase-D C7 text source)
# ─────────────────────────────────────────────────────────────────────────────


def _rbase_json_path(out_root: Path, behavior: str, source: str, seed: int) -> Path:
    return out_root / "raw_completions" / "rbase" / behavior / f"{source}_seed{seed}.json"


def build_rbase_cell_payload(
    behavior: str,
    source: str,
    seed: int,
    targets: list[str],
    probes: list[str],
    text_by: dict[tuple[str, int], str],
    *,
    gen_backend: str,
    max_new_tokens: int,
) -> dict:
    """Per-(behavior, source) R_base payload mirroring the R⁺ generation bucket.

    Shape = the R⁺ payload's metadata + ``responses`` record list (target_cid /
    probe_idx / probe / response / resp_sha256), PLUS a ``targets`` map
    ``{tcid: [text ordered by probe_idx]}`` — the first (highest-precedence)
    shape the Phase-D fit driver's tolerant reader (``_texts_from_json``)
    resolves. R_base is source-independent (BASE model, no LoRA), so per-source
    files duplicate shared texts; the duplication buys the fit driver's
    ``{prefix}/{behavior}/{source}_seed{seed}.json`` resolution without touching
    it. A missing (target, probe) text is a coverage bug → KeyError (fail loud).
    """
    rows: list[dict] = []
    targets_map: dict[str, list[str]] = {}
    for tcid in targets:
        texts: list[str] = []
        for qi, q in enumerate(probes):
            text = text_by[(tcid, qi)]  # KeyError = unique-grid coverage bug, fail loud
            texts.append(text)
            rows.append(
                {
                    "target_cid": tcid,
                    "probe_idx": qi,
                    "probe": q,
                    "response": text,
                    "resp_sha256": _sha256(text),
                }
            )
        targets_map[tcid] = texts
    return {
        "behavior": behavior,
        "source_cid": source,
        "seed": seed,
        "adapter_subfolder": None,  # BASE model — no LoRA by construction
        "gen_backend": gen_backend,
        "sampling": {"temperature": 0.0, "max_tokens": max_new_tokens},
        "n_targets": len(targets),
        "n_probes": len(probes),
        "n_empty": sum(1 for r in rows if not r["response"].strip()),
        "ts": _now(),
        "targets": targets_map,
        "responses": rows,
    }


def _rbase_worklist(args, out_root: Path) -> list[tuple[str, dict[str, list[str]], list[str]]]:
    """(behavior, {source: capped targets}, capped probes) rows missing R_base JSONs.

    Resume-skip at behavior grain: ALL per-source JSONs present → skip; else the
    unique grid regenerates and every per-source file rewrites (atomic writes).
    """
    behaviors = BEHAVIORS if args.behavior == "all" else (args.behavior,)
    work: list[tuple[str, dict[str, list[str]], list[str]]] = []
    for behavior in behaviors:
        sources = _sources_for(behavior)
        if args.source_cid:
            sources = [s for s in sources if s == args.source_cid]
        per_source: dict[str, list[str]] = {}
        for source in sources:
            targets = _targets_for(behavior, source)
            per_source[source] = targets[: args.max_targets] if args.max_targets else targets
        if not per_source:
            continue
        if all(_rbase_json_path(out_root, behavior, s, args.seed).exists() for s in per_source):
            logger.info(
                "rbase resume-skip: %s all %d per-source JSONs exist", behavior, len(per_source)
            )
            continue
        probes = x667.load_eval_probes(behavior)
        if args.max_probes:
            probes = probes[: args.max_probes]
        work.append((behavior, per_source, probes))
    return work


def stage_rbase(args) -> int:
    """A1: same-era R_base′ — ONE base-model (NO LoRA) greedy pass per behavior
    over the unique (target, probe) grid, fanned out to per-(behavior, source)
    JSONs. R_base′ REPLACES the store-era R_base everywhere downstream (plan v6
    §4 Phase A1); stage extract-rbase teacher-forces exactly these texts."""
    out_root = Path(args.out)
    device = x667._device(args.gpu_id, args.cpu_only)
    registry, demos, tok = _load_inputs()
    work = _rbase_worklist(args, out_root)
    if not work:
        return 0

    engine = None
    base = None
    if args.gen_backend == "vllm":
        if device.type == "cpu":
            raise RuntimeError("--gen-backend vllm requires a GPU; use --gen-backend peft")
        # PLAIN engine (no enable_lora): R_base′ is base-only, matching the A0′
        # plain-engine arm (matched arms — plan v5 amendment iii).
        engine = _VllmLoraEngine(enable_lora=False)
    else:  # peft fallback (plan §8 R1) — BASE-only HF load, same path family as extraction
        from transformers import AutoModelForCausalLM

        dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(
            x667.BASE_MODEL, torch_dtype=dtype, token=os.environ.get("HF_TOKEN")
        ).to(device)
        base.eval()
    try:
        for behavior, per_source, probes in work:
            t0 = time.time()
            unique_targets = list(dict.fromkeys(t for ts in per_source.values() for t in ts))
            prompts = [
                _prompt_text(tok, registry, demos, tcid, behavior, q)
                for tcid in unique_targets
                for q in probes
            ]
            if engine is not None:
                ids = engine.greedy(prompts, max_new_tokens=args.max_new_tokens, adapter_dir=None)
            else:
                ids = _peft_batched_greedy(
                    base, tok, prompts, max_new_tokens=args.max_new_tokens, device=device
                )
            texts = [tok.decode(i, skip_special_tokens=True) for i in ids]
            assert len(texts) == len(unique_targets) * len(probes), (
                len(texts),
                len(unique_targets),
                len(probes),
            )
            text_by = {
                (tcid, qi): texts[ti * len(probes) + qi]
                for ti, tcid in enumerate(unique_targets)
                for qi in range(len(probes))
            }
            for source, targets in per_source.items():
                payload = build_rbase_cell_payload(
                    behavior,
                    source,
                    args.seed,
                    targets,
                    probes,
                    text_by,
                    gen_backend=args.gen_backend,
                    max_new_tokens=args.max_new_tokens,
                )
                _write_json(_rbase_json_path(out_root, behavior, source, args.seed), payload)
            logger.info(
                "[phase=rbase] %s: %d unique targets x %d probes -> %d per-source JSONs "
                "(%d empty) in %.0fs",
                behavior,
                len(unique_targets),
                len(probes),
                len(per_source),
                sum(1 for t in texts if not t.strip()),
                time.time() - t0,
            )
    finally:
        if engine is not None:
            engine.shutdown()
        if base is not None:
            del base
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 0


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
    # ``targets`` map {tcid: [text ordered by probe_idx]} mirrors the rbase
    # payload shape (build_rbase_cell_payload) — the first (highest-precedence)
    # shape the Phase-D fit driver's tolerant reader (_texts_from_json) resolves.
    targets_map: dict[str, list[str]] = {}
    for (tcid, _qi), text in zip(keys, texts, strict=True):
        targets_map.setdefault(tcid, []).append(text)
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
        "targets": targets_map,
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


def build_leg_npz_payload(
    *,
    leg_mode: str,
    v0_mean: np.ndarray,
    vplus_mean: np.ndarray,
    shas: list[str],
    shas_base: list[str] | None,
    probe_ids: list[int],
    behavior: str,
    source: str,
    seed: int,
    tcid: str,
    layer: int,
    gauge: dict,
    gen_backend: str,
    stored_context: dict | None,
) -> dict:
    """Assemble one leg-npz payload for either namespace (pure — unit-testable).

    ``leg_mode="onpolicy"`` (Phase B2, ``analysis_tensors``): canonical
    ``v0_onpolicy``/``v_plus_onpolicy`` legs + loader-compatible ``v0``/
    ``v_plus`` aliases + the PINNED store's context-vector copies
    (``stored_context`` REQUIRED) + ``resp_sha256_base`` threading (REQUIRED).
    ``leg_mode="rbase"`` (Phase B1, ``analysis_tensors_rbase``): canonical
    ``v0_rbase``/``v_plus_rbase`` legs + the SAME ``v0``/``v_plus`` aliases
    (the registered same-era L1/L2 array names), ``resp_sha256`` = sha of
    R_base′ itself; NO context copies (B1 performs zero old-store reads) and
    NO ``resp_sha256_base`` (it IS the base leg).
    """
    assert leg_mode in ("onpolicy", "rbase"), leg_mode
    payload: dict = {
        "v0": v0_mean,
        "v_plus": vplus_mean,
        "behavior": behavior,
        "source_cid": source,
        "target_cid": tcid,
        "seed": seed,
        "layer": layer,
        "n_probes": len(shas),
        "resp_sha256": np.array(shas),
        "probe_idx": np.array(probe_ids, dtype=np.int64),
        "adapter_gauge": json.dumps(gauge),
        "gen_backend": gen_backend,
        "leg_mode": leg_mode,
    }
    if leg_mode == "onpolicy":
        assert stored_context is not None and shas_base is not None
        payload["v0_onpolicy"] = v0_mean
        payload["v_plus_onpolicy"] = vplus_mean
        payload["resp_sha256_base"] = np.array(shas_base)
        # context vectors REUSED from the pinned #667 store (text-free; reuse
        # licensed by the A0′ c_C stack-parity probe — plan v6 §4 gate 4)
        for k in ("c_C", "c_Cp", "c_C_postft", "c_Cp_postft"):
            payload[k] = np.asarray(stored_context[k], dtype=np.float32)
        payload["store_revision_pin"] = STORE_REVISION
    else:
        payload["v0_rbase"] = v0_mean
        payload["v_plus_rbase"] = vplus_mean
    return payload


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
    base_sha_by_probe: dict[tuple[str, int], str] | None,
    leg_mode: str = "onpolicy",
) -> int:
    """Teacher-force each response through BOTH models; write one npz per layer.

    Mirrors #667's ``_extract_one_target`` writer block: same naming
    ``{tcid}_L{li}.npz``, same mean-over-probes (3584,) float32 convention.
    ``leg_mode="onpolicy"`` (Phase B2) writes the R⁺ legs ``v_plus_onpolicy`` /
    ``v0_onpolicy`` + ``v0``/``v_plus`` aliases + the PINNED store's context
    copies + ``resp_sha256_base`` (SAME-ERA R_base′ shas from stage rbase,
    keyed ``(tcid, probe_idx)``; a missing entry fails loud).
    ``leg_mode="rbase"`` (Phase B1) teacher-forces R_base′ instead and writes
    the same-era L1/L2 legs ``v0_rbase``/``v_plus_rbase`` (+ ``v0``/``v_plus``
    aliases, ``resp_sha256`` of R_base′, ``probe_idx``) — no context copies,
    no old-store reads. Empty rows are COMPACTED out of the hash arrays, so
    every npz persists ``probe_idx`` — the ORIGINAL probe ids, index-aligned
    with the compacted hashes (probe-index drift guard, round-2 blocker 2).
    Returns the number of npz written.
    """
    acc: dict[int, list[list[np.ndarray]]] = {li: [[], []] for li in layers}
    shas: list[str] = []
    shas_base: list[str] = []
    probe_ids: list[int] = []
    for qi, q, r in rows:
        if not r.strip():
            continue
        if leg_mode == "onpolicy":
            assert base_sha_by_probe is not None, "onpolicy leg_mode needs base_sha_by_probe"
            if (tcid, qi) not in base_sha_by_probe:
                raise KeyError(
                    f"resp_sha256_base missing for ({tcid}, probe {qi}) — "
                    "run --stage rbase first (fail loud, no silent skip)"
                )
            shas_base.append(base_sha_by_probe[(tcid, qi)])
        msgs = x667.build_messages_for(registry, demos, tcid, behavior, q)
        per_layer = x667._mean_resp_acts(base, trained, tok, msgs, r, layers, device)
        shas.append(_sha256(r))
        probe_ids.append(int(qi))
        for li in layers:
            v0, vp = per_layer[li]
            acc[li][0].append(v0)
            acc[li][1].append(vp)
    n_written = 0
    for li in layers:
        if not acc[li][0]:
            logger.warning(
                "no non-empty %s response for target=%s layer=%d — npz skipped",
                leg_mode,
                tcid,
                li,
            )
            continue
        stored = (
            _stage_store_npz(behavior, source, seed, tcid, li) if leg_mode == "onpolicy" else None
        )
        payload = build_leg_npz_payload(
            leg_mode=leg_mode,
            v0_mean=np.stack(acc[li][0]).mean(axis=0).astype(np.float32),
            vplus_mean=np.stack(acc[li][1]).mean(axis=0).astype(np.float32),
            shas=shas,
            shas_base=shas_base if leg_mode == "onpolicy" else None,
            probe_ids=probe_ids,
            behavior=behavior,
            source=source,
            seed=seed,
            tcid=tcid,
            layer=li,
            gauge=gauge,
            gen_backend=gen_backend,
            stored_context=stored,
        )
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
            rbase_path = _rbase_json_path(out_root, behavior, source, args.seed)
            if not rbase_path.exists():
                raise FileNotFoundError(
                    f"R_base rollout JSON missing for {behavior}/{source}: {rbase_path} — "
                    "run --stage rbase first (fail loud, no silent skip)"
                )
            rbase = json.loads(rbase_path.read_text())
            base_sha_by_probe = {
                (row["target_cid"], int(row["probe_idx"])): row["resp_sha256"]
                for row in rbase["responses"]
            }
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
                    base_sha_by_probe=base_sha_by_probe,
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
# Stage B1 — same-era L1/L2 re-extraction of R_base′ (plan v6 §4 Phase B1)
# ─────────────────────────────────────────────────────────────────────────────


def stage_extract_rbase(args) -> int:
    """B1: teacher-force the fleet-wide R_base′ text through BOTH θ0 and θ⁺ per
    cell (layers ``--layers``) and write the SAME-ERA L1/L2 npz namespace
    ``analysis_tensors_rbase`` (arrays ``v0_rbase``/``v_plus_rbase`` +
    ``v0``/``v_plus`` aliases, ``probe_idx``, ``resp_sha256`` of R_base′).
    Per-cell ``.done`` sentinels + resume-skip, exactly like Phase B2."""
    out_root = Path(args.out)
    behaviors = BEHAVIORS if args.behavior == "all" else (args.behavior,)
    device = x667._device(args.gpu_id, args.cpu_only)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    registry, demos, tok = _load_inputs()
    layers = list(args.layers)
    tensors_root = out_root / "analysis_tensors_rbase"

    for behavior in behaviors:
        for source in _sources_for(behavior):
            if args.source_cid and source != args.source_cid:
                continue
            cell_dir = tensors_root / behavior / f"{source}_seed{args.seed}"
            if (cell_dir / x667.CELL_DONE_SENTINEL).exists():
                logger.info("extract-rbase resume-skip: %s/%s .done exists", behavior, source)
                continue
            rbase_path = _rbase_json_path(out_root, behavior, source, args.seed)
            if not rbase_path.exists():
                raise FileNotFoundError(
                    f"R_base′ rollout JSON missing for {behavior}/{source}: {rbase_path} — "
                    "run --stage rbase first (fail loud, no silent skip)"
                )
            rbase = json.loads(rbase_path.read_text())
            _, adapter_dir, gauge = _resolve_adapter(behavior, source, args.seed)
            _, base, trained = x667.load_base_and_trained(adapter_dir, device, dtype)
            cell_dir.mkdir(parents=True, exist_ok=True)
            by_target: dict[str, list] = {}
            for row in rbase["responses"]:
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
                    gen_backend=rbase.get("gen_backend", "vllm"),
                    base_sha_by_probe=None,
                    leg_mode="rbase",
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
                "[phase=extract-rbase] cell %s/%s: %d targets x %d layers in %.0fs",
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
# Stage extract-context — the c_C-parity-FAIL contingency (plan v6 §4 gate 4)
# ─────────────────────────────────────────────────────────────────────────────


def stage_extract_context(args) -> int:
    """Registered in-run contingency: fleet-wide c_C / c_C_postft re-extraction.

    One ``__context__.npz`` per (behavior, source) under the rbase namespace,
    carrying rows at ``--layers`` (row order == the ``layers`` array; the #667
    recipe: last-input-token of ``src_msgs = build_messages_for(source,
    probes[0])`` with the writer's ``c_all[li - 1]`` layer mapping). After this
    stage, ZERO old-store reads remain anywhere (the fit driver consumes these
    via ``--context-source reextracted``). Invoked by run_all ONLY when the A0′
    summary carries ``reextract_context_vectors: true``. Resume-skip per file.
    """
    out_root = Path(args.out)
    behaviors = BEHAVIORS if args.behavior == "all" else (args.behavior,)
    device = x667._device(args.gpu_id, args.cpu_only)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    registry, demos, tok = _load_inputs()
    layers = list(args.layers)
    tensors_root = out_root / "analysis_tensors_rbase"

    for behavior in behaviors:
        probes = x667.load_eval_probes(behavior)
        for source in _sources_for(behavior):
            if args.source_cid and source != args.source_cid:
                continue
            cell_dir = tensors_root / behavior / f"{source}_seed{args.seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            ctx_path = cell_dir / CONTEXT_NPZ_NAME
            if ctx_path.exists():
                logger.info("extract-context resume-skip: %s/%s exists", behavior, source)
                continue
            _, adapter_dir, gauge = _resolve_adapter(behavior, source, args.seed)
            _, base, trained = x667.load_base_and_trained(adapter_dir, device, dtype)
            src_msgs = x667.build_messages_for(registry, demos, source, behavior, probes[0])
            c_all = x667._context_vector_all_layers(base, tok, src_msgs, device)
            cp_all = x667._context_vector_all_layers(trained, tok, src_msgs, device)
            payload = {
                # row i == layers[i]; the #667 store mapping is c_all[li - 1]
                "c_C": np.stack([c_all[li - 1] for li in layers]).astype(np.float32),
                "c_C_postft": np.stack([cp_all[li - 1] for li in layers]).astype(np.float32),
                "layers": np.array(layers, dtype=np.int64),
                "behavior": behavior,
                "source_cid": source,
                "seed": args.seed,
                "adapter_gauge": json.dumps(gauge),
                "recipe": (
                    "last-input-token; src_msgs = build_messages_for(source, probes[0]); "
                    "row for layer li == _context_vector_all_layers(...)[li - 1] (#667 c_idx)"
                ),
            }
            tmp = cell_dir / f"{CONTEXT_NPZ_NAME}.{os.getpid()}.tmp.npz"
            np.savez(tmp, **payload)
            os.replace(tmp, ctx_path)
            logger.info("[phase=extract-context] %s/%s -> %s", behavior, source, ctx_path)
            del base, trained
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 0


def _reextract_context_flag(out_root: Path) -> bool:
    """Read the A0′ ``reextract_context_vectors`` flag (False when absent)."""
    p = out_root / "parity" / "a0_summary.json"
    if not p.exists():
        return False
    return bool(json.loads(p.read_text()).get("reextract_context_vectors", False))


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
    """Phase C: bulk upload_folder for the two raw-completion buckets + BOTH npz
    namespaces (one commit per behavior per namespace), then a list_repo_files
    count verification — 4,320 + 4,320 = 8,640 leg npz expected, plus the 48
    ``__context__.npz`` when the extract-context contingency fired."""
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

    rbase_dir = out_root / "raw_completions" / "rbase"
    if not rbase_dir.is_dir():
        raise FileNotFoundError(f"R_base completions dir missing: {rbase_dir} — run --stage rbase")
    n_rbase = len(list(rbase_dir.rglob("*.json")))
    logger.info("[phase=upload] R_base′ completions: %d JSONs -> %s", n_rbase, RBASE_PREFIX)
    api.upload_folder(
        folder_path=str(rbase_dir),
        path_in_repo=RBASE_PREFIX,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*.json"],
        commit_message="issue-833: R_base′ same-era regeneration raw completions",
    )

    n_ctx_local = 0
    for namespace, prefix in (
        ("analysis_tensors", TENSORS_PREFIX),
        ("analysis_tensors_rbase", RBASE_TENSORS_PREFIX),
    ):
        tensors_root = out_root / namespace
        for behavior in BEHAVIORS:
            beh_dir = tensors_root / behavior
            if not beh_dir.is_dir():
                raise FileNotFoundError(f"analysis tensors dir missing: {beh_dir}")
            npzs = list(beh_dir.rglob("*.npz"))
            n_ctx_local += sum(1 for p in npzs if p.name == CONTEXT_NPZ_NAME)
            logger.info(
                "[phase=upload] %s/%s: %d npz -> %s/%s",
                namespace,
                behavior,
                len(npzs),
                prefix,
                behavior,
            )
            # ONE bulk upload_folder commit per behavior per namespace (never a
            # per-file loop — upload-policy 504-storm rule; 1,440-1,456
            # files/behavior < the 10k dir cap).
            api.upload_folder(
                folder_path=str(beh_dir),
                path_in_repo=f"{prefix}/{behavior}",
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                allow_patterns=["*.npz"],
                commit_message=f"issue-833: {namespace} ({behavior})",
            )

    files = _list_repo_files_retry(HF_DATA_REPO, repo_type="dataset")
    n_on = sum(1 for f in files if f.startswith(f"{TENSORS_PREFIX}/") and f.endswith(".npz"))
    rb_files = [f for f in files if f.startswith(f"{RBASE_TENSORS_PREFIX}/") and f.endswith(".npz")]
    n_ctx_hub = sum(1 for f in rb_files if f.endswith(f"/{CONTEXT_NPZ_NAME}"))
    n_rb = len(rb_files) - n_ctx_hub
    n_raw_hub = sum(1 for f in files if f.startswith(f"{RAW_PREFIX}/") and f.endswith(".json"))
    n_rbase_hub = sum(1 for f in files if f.startswith(f"{RBASE_PREFIX}/") and f.endswith(".json"))
    expected = args.expected_npz if args.expected_npz is not None else EXPECTED_NPZ_PER_NAMESPACE
    ctx_flag = _reextract_context_flag(out_root)
    expected_ctx = EXPECTED_CONTEXT_NPZ if ctx_flag else 0
    logger.info(
        "[phase=upload] verify: %d R⁺ npz + %d rbase npz on Hub (expected %d each), "
        "%d context npz (expected %d, flag=%s), %d raw JSONs (local %d), "
        "%d rbase JSONs (local %d)",
        n_on,
        n_rb,
        expected,
        n_ctx_hub,
        expected_ctx,
        ctx_flag,
        n_raw_hub,
        n_raw,
        n_rbase_hub,
        n_rbase,
    )
    counts = {
        "npz_on_hub_onpolicy": n_on,
        "npz_on_hub_rbase": n_rb,
        "npz_expected_per_namespace": expected,
        "context_npz_on_hub": n_ctx_hub,
        "context_npz_local": n_ctx_local,
        "context_npz_expected": expected_ctx,
        "reextract_context_vectors": ctx_flag,
        "raw_json_on_hub": n_raw_hub,
        "rbase_json_on_hub": n_rbase_hub,
        "rbase_json_local": n_rbase,
    }
    _write_json(out_root / "upload_verification.json", {**counts, "ts": _now()})
    _write_phase_sentinel("upload-verified", counts)
    if (
        n_on != expected
        or n_rb != expected
        or n_raw_hub < n_raw
        or n_rbase == 0
        or n_rbase_hub < n_rbase
        or n_ctx_hub < n_ctx_local
        or (ctx_flag and n_ctx_hub != expected_ctx)
    ):
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
    n_rbase_cells = len(list((out_root / "raw_completions" / "rbase").rglob("*.json")))
    behav = a0.get("behavioral_adapter_effect", {})
    ccp = a0.get("cc_stack_parity", {})
    det = a0.get("determinism", {})
    diag = a0.get("backend_divergence_diagnostic", {})
    note = {
        "eval_numbers": {
            "npz_on_hub_onpolicy": upload["npz_on_hub_onpolicy"],
            "npz_on_hub_rbase": upload["npz_on_hub_rbase"],
            "npz_expected_per_namespace": upload["npz_expected_per_namespace"],
            "context_npz_on_hub": upload["context_npz_on_hub"],
            "raw_completion_cells": n_gen_cells,
            "rbase_completion_cells": n_rbase_cells,
            "rbase_json_on_hub": upload["rbase_json_on_hub"],
            "a0_behavioral_adapter_effect_pass": behav.get("pass"),
            "a0_behavioral_exact_match_frac": [
                r["exact_match_frac"] for r in behav.get("results", [])
            ],
            "a0_cc_parity_pass": ccp.get("pass"),
            "a0_cc_parity_max_rel_l2": [r["max_rel_l2"] for r in ccp.get("results", [])],
            "a0_reextract_context_vectors": a0.get("reextract_context_vectors"),
            "a0_determinism_pass": det.get("pass"),
            "a0_backend_divergence_frac_identical": [
                r["frac_identical"] for r in diag.get("results", [])
            ],
            "a0_phase_b_smoke_pass": (a0.get("phase_b_smoke") or {}).get("pass"),
        },
        "eval_paths": [
            "eval_results/issue_833/parity/a0_summary.json",
            "eval_results/issue_833/parity/a0_cc_parity.json",
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
                "rbase_completions": RBASE_PREFIX,
                "analysis_tensors": TENSORS_PREFIX,
                "analysis_tensors_rbase": RBASE_TENSORS_PREFIX,
                "reused_store_context_vectors_only": f"{STORE_PREFIX} @ {STORE_REVISION}",
            },
            "store_revision_pin": STORE_REVISION,
            "gen_backend": a0.get("gen_backend", "vllm"),
            "seeds": [args.seed],
            "layers": list(args.layers),
            "max_new_tokens": args.max_new_tokens,
        },
        "wandb_url": "n/a — no training",
        "hf_hub_url": f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/{HF_PREFIX}",
        "worktree_path": args.worktree_path or str(PROJECT_ROOT),
        "final_commit_sha": args.final_commit_sha or "unknown",
        "gpu_hours_used": args.gpu_hours_used,
        "gpu_hours_budgeted": 21,
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
        choices=[
            "parity",
            "rbase",
            "generate",
            "extract-rbase",
            "extract-context",
            "extract",
            "upload",
            "finalize",
            "all",
        ],
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
    elif args.stage == "rbase":
        rc = stage_rbase(args)
    elif args.stage == "generate":
        rc = stage_generate(args)
    elif args.stage == "extract-rbase":
        rc = stage_extract_rbase(args)
    elif args.stage == "extract-context":
        rc = stage_extract_context(args)
    elif args.stage == "extract":
        rc = stage_extract(args)
    elif args.stage == "upload":
        rc = stage_upload(args)
    elif args.stage == "finalize":
        rc = stage_finalize(args)
    else:  # all — subprocess per stage (vLLM/transformers isolation, R3). NO
        # automatic peft-fallback A0 re-run (v6: the retired token-identity
        # gate was its trigger; --gen-backend peft is a manual option only).
        rc = _sub(args, "parity", ["--cpu-smoke"] if args.cpu_smoke else [])
        if rc != 0:
            logger.error("A0′ gates FAIL (rc=%d) — aborting before Phase A", rc)
            return rc
        stages = ["rbase", "generate", "extract-rbase"]
        if _reextract_context_flag(Path(args.out)):
            logger.info(
                "Decision: A0′ c_C parity FAIL flagged — running the extract-context "
                "contingency (plan v6 §4 gate 4 / §13 allowed-without-asking)"
            )
            stages.append("extract-context")
        stages += ["extract", "upload"]
        for stage in stages:
            rc = _sub(args, stage, [])
            if rc != 0:
                logger.error("stage %s FAILED rc=%d", stage, rc)
                return rc
    logger.info("stage=%s wall=%.1fs rc=%d", args.stage, time.time() - t0, rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
