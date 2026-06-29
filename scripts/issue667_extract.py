#!/usr/bin/env python3
"""Issue #667 absolute-v extractor — per-source-adapter forward-pass sweep.

For ONE (behavior, source-context C) cell this CLI:

1. Stages + loads the #537 adapter as a ``PeftModel`` on the base Qwen-2.5-7B
   (rsLoRA honored — every #537 adapter has ``use_rslora=True``); asserts the
   adapter's ``base_model_name_or_path == Qwen/Qwen2.5-7B-Instruct`` (fitness
   check (f)).
2. For each eval target context C' (the 30 #537 eval cids + the source C
   itself): builds the eval probes via the BYTE-FAITHFUL ``i537_contexts``
   registry (registry_hash f12061d6... == the G_meta pin), generates the frozen
   base greedy response R per probe (deterministic, temp=0), teacher-forces
   ``T_{C'}(q) + R`` through BOTH base θ0 and trained θ+ once each, and reads
   the MEAN-over-response-span residual at L14 (+7,21): ``v0(C')``, ``v+(C')``
   (both float32 CPU), mean over probes. (Marker companion also reads the
   post-response slot.)
3. Extracts the base-side context vectors ``c_C`` / ``c_{C'}`` (last-input-token,
   all 28 layers) over the SAME contexts — the whitened-gate key/query (A3.9).
4. Extracts ``t+`` / ``t-`` for A3.7: teacher-forces the #537 frozen training-mix
   POSITIVE rows (prompt context == source C) and NEGATIVE rows (prompt context
   in the negative panel) through θ0, mean answer-side activation. Positive vs
   negative is split by matching the rendered source-context prompt prefix (the
   builder writes positives under the source ctx, negatives under the neg panel)
   — robust to the untagged JSONL. ALSO extracts ``v0_C_neg`` — the base-CONTEXT
   activation under each negative persona's PROMPT (no answer span), matched to
   the ``v0(C)`` mean-over-response recipe, panel-averaged over the negative cids.
   This is the A3.7 ``frac_ctx`` numerator term (R3-1) and is DISTINCT from ``t-``
   (the negative-persona answer activation that feeds ``delta_contra``).
5. For ``fact``: re-extracts ``r_B`` fresh (absent from #658's r_b.pt) via the
   #594 diff-in-means recipe (system-prompt pos/neg pair, mean answer act).

Writes one ``.npz`` per (behavior, source-C, target-C', layer) under
``eval_results/issue_667/analysis_tensors/`` with ``{v0, v_plus}`` per side, the
per-cell ``c_C``/``c_Cp`` (all layers), ``t_pos``/``t_neg``, the negative-panel
base-context vector ``v0_C_neg`` (A3.7 frac_ctx, R3-1 — distinct from ``t_neg``),
and (fact) ``r_b``.

CONTENT HYGIENE: ``em`` training rows are Betley harmful-content — this script
NEVER prints/logs their text; it digests by row count + token count + the
ACTIVATIONS only. Benign behaviors (marker/fact/sycophancy) are unaffected.

Usage (one source-adapter cell)::

    uv run python scripts/issue667_extract.py \\
        --behavior em --source-cid default \\
        --targets sp_swe,default,fmt_json --layers 7 14 21 --primary-layer 14 \\
        --out eval_results/issue_667/analysis_tensors --gpu-id 0
"""

# ruff: noqa: RUF001, RUF002  # math/scientific notation in docstrings + messages

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
# vLLM EngineCore fork() poisoning guard (.claude/rules/gotchas.md § entry 26):
# main() touches transformers.AutoTokenizer (L705-707) BEFORE vllm_generate_R
# constructs vllm.LLM() (L228); ANY pre-LLM() transformers/tokenizer/registry
# touch poisons the EngineCore fork. spawn (not fork) avoids the silent worker
# death. Must be set at module top, BEFORE any `import vllm`. Do not strip.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.analysis.issue667 import (  # noqa: E402
    ALL_LAYERS,
    BASE_MODEL,
    HF_MODEL_REPO,
    HIDDEN_SIZE,
    N_LAYERS,
    PRIMARY_LAYER,
)
from explore_persona_space.orchestrate.scratch_io import (  # noqa: E402
    materialize_to_canonical,
    scratch_path_for,
)

load_dotenv()

logger = logging.getLogger("issue667_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_PREFIX = "issue537_context_generalization/data"
# r⁺ store HF prefix — byte-identical to issue667_dispatch.HF_R_PLUS_PREFIX (the
# upload + idempotent-skip canonical layout: <prefix>/<behavior>/<src>_seed<S>_L<l>.npz).
HF_R_PLUS_PREFIX = "issue667_gate_chain_preview/a36_readout_reextract/r_plus"
# Per-behavior eval probe pools (#537 frozen pools, plan §4.0).
N_GEN_TOKENS = 1024  # greedy R cap (natural Qwen replies ~150 tok; log truncation)

# r⁺ re-extraction recipe constants (followup a36-readout-reextract-cos).
# Byte-identity to #658's base-r_B recipe (plan §4.3 / §12): the em/syco prompt
# sets are reconstructed via build_rb_contrast against the SAME pinned Betley
# neutral pool + cap; fact reuses the _FACT_POS_SYS/_FACT_NEG_SYS pair above.
RB_CAP = 48  # #658 rb_cap (pinned; §12)
# Per-behavior generation cap for the diff-in-means answer span: em/syco mirror
# #658's capture_mean_answer_acts (max_new_tokens=64); fact mirrors #667's
# extract_fact_r_b (256). The base r_B used these caps, so r⁺ matches per behavior.
RPLUS_GEN_TOKENS = {"em": 64, "sycophancy": 64, "fact": 256}
# em/syco -> the #658 r_b.pt column id consumed by build_rb_contrast (E0_COLUMNS).
RPLUS_RB_COLUMN = {"em": "broad_em", "sycophancy": "sycophancy"}

# Atomic completion sentinel. The extractor writes the per-(target, layer) ``.npz``
# INCREMENTALLY, so a mid-cell crash leaves a PARTIAL dir that a presence-only
# ``any(*.npz)`` resume-skip would wrongly treat as complete (round-8 BLOCKER
# resume-skip-partial-cell-silent-skip). The sentinel is written ATOMICALLY only
# AFTER every planned (target, layer) ``.npz`` is on disk — its presence is the
# proof of FULL completion. The resume-skip predicate checks for this file, not
# for any stray ``.npz``.
CELL_DONE_SENTINEL = ".done"


# ─────────────────────────────────────────────────────────────────────────────
# Per-iteration GPU-memory logging (#672 validation — ANALYSIS-ONLY, off by
# default). Behind ``--log-mem-every N`` (N=0 -> off). Logs THREE gauges per
# hooked forward to certify the #671 fix removed the resident-pool climb:
#   1. torch.cuda.memory_reserved()  (PRIMARY) — the allocator's resident pool;
#      under ``expandable_segments:True`` this retains freed-segment fragments,
#      so the climb-vs-flat signal lives HERE, not in ``memory_allocated()``.
#   2. nvidia-smi memory.used         (PRIMARY) — kernel-visible resident, the
#      literal footprint that drives DHCP/page-reclaim wedges; ``None`` when
#      nvidia-smi is absent (CPU host / no NVIDIA runtime).
#   3. torch.cuda.memory_allocated()  (SECONDARY) — live tensor bytes; recorded
#      for diagnostics, NOT in the pass predicate (a fix that frees tensors but
#      leaves the pool fragmented reads flat here, which Section A must catch).
# The "iter" counter is the COUNT of hooked forwards (every _mean_resp_acts* /
# extract_layer_activations invocation on the #671-fixed read path), NOT a
# clock tick. No behavior change to the reads.
# ─────────────────────────────────────────────────────────────────────────────


class _MemoryGaugeLogger:
    """Mutable per-run memory-gauge accumulator (module-level singleton).

    Held at module scope so the deep per-probe reads (:func:`_mean_resp_acts`,
    :func:`_mean_resp_acts_single`) can instrument the hooked-forward sites
    WITHOUT threading a new argument through every call signature — the reads
    themselves are unchanged (single-variable contract: only the additive
    ``_tick`` instrumentation call is new). ``run_extraction`` installs the
    active logger from ``args`` (or ``None`` when ``--log-mem-every 0``).
    """

    def __init__(self, every: int, out_dir: Path) -> None:
        self.every = int(every)
        self.out_dir = Path(out_dir)
        self.iter_idx = 0
        self.log: list[dict] = []

    def _tick(self) -> None:
        """Increment the hooked-forward counter; log + record every ``every``."""
        idx = self.iter_idx
        self.iter_idx += 1
        if self.every <= 0 or idx % self.every != 0:
            return
        _log_memory_gauges(idx, self.log, self.out_dir)


# Module-level active logger; None disables instrumentation (the default).
_MEM_LOGGER: _MemoryGaugeLogger | None = None


def install_mem_logger(args) -> None:
    """Install the module-level memory logger from ``args`` (no-op when off).

    Sets the global ``_MEM_LOGGER`` to an active accumulator when
    ``--log-mem-every > 0`` (writing to ``args.out``), else ``None``. Kept out
    of :func:`run_extraction` so that function stays under the ruff C901 cap.
    """
    global _MEM_LOGGER
    if getattr(args, "log_mem_every", 0) > 0:
        _MEM_LOGGER = _MemoryGaugeLogger(args.log_mem_every, Path(args.out))
    else:
        _MEM_LOGGER = None


def flush_mem_logger() -> None:
    """Log the final iteration + persist the full memory-gauge array (no-op off).

    The last hooked forward's index is rarely an exact multiple of
    ``--log-mem-every``, so this captures the terminal resident footprint
    unconditionally and re-writes ``<out>/memory_log.json``.
    """
    if _MEM_LOGGER is None:
        return
    last_idx = max(_MEM_LOGGER.iter_idx - 1, 0)
    _log_memory_gauges(last_idx, _MEM_LOGGER.log, _MEM_LOGGER.out_dir)
    logger.info(
        "[mem] %d gauge samples written to %s",
        len(_MEM_LOGGER.log),
        _MEM_LOGGER.out_dir / "memory_log.json",
    )


def _nvidia_smi_used_gib() -> float | None:
    """nvidia-smi memory.used for the CVD-visible device 0, in GiB, or ``None``.

    Returns ``None`` cleanly when nvidia-smi is absent (FileNotFoundError), the
    probe errors / times out, or no device is parseable (CPU host / docker w/o
    NVIDIA runtime). Reads MiB from nvidia-smi (its native unit) and converts to
    GiB so all three gauges share a unit. The launcher pins
    ``CUDA_VISIBLE_DEVICES=<gpu>`` so cuda:0 == the visible device; nvidia-smi
    honours CVD, so its first row is that same device.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    first = proc.stdout.strip().splitlines()
    if not first:
        return None
    try:
        used_mib = float(first[0].strip())
    except ValueError:
        return None
    return used_mib / 1024.0  # MiB -> GiB


def _log_memory_gauges(iter_idx: int, log: list[dict], out_dir: Path) -> None:
    """Capture the 3 memory gauges at ``iter_idx``, append to ``log``, log a line.

    Always re-writes ``out_dir / memory_log.json`` (a JSON array of the dicts)
    so the trace is durable even if the run is killed before clean exit
    (checkpoint-per-phase: never accumulate-in-memory and write-only-at-end).
    """
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved() / 2**30
        allocated = torch.cuda.memory_allocated() / 2**30
    else:
        reserved = 0.0
        allocated = 0.0
    nvidia_smi = _nvidia_smi_used_gib()
    entry = {
        "iter": iter_idx,
        "memory_reserved_gib": float(reserved),  # PRIMARY
        "nvidia_smi_used_gib": (None if nvidia_smi is None else float(nvidia_smi)),  # PRIMARY
        "memory_allocated_gib": float(allocated),  # SECONDARY
    }
    log.append(entry)
    logger.info(
        "[mem] iter=%d reserved=%.2fGiB nvidia_smi=%s allocated=%.2fGiB",
        iter_idx,
        reserved,
        "n/a" if nvidia_smi is None else f"{nvidia_smi:.2f}GiB",
        allocated,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memory_log.json").write_text(json.dumps(log, indent=2))


def _prepare_scratch_cell_dir(scratch_cell_dir: Path, cell_dir: Path) -> None:
    """Clean + (re)create the local-SSD scratch cell dir before extraction (#674).

    When ``scratch_cell_dir`` is a real scratch mirror (not the canonical
    ``cell_dir`` pass-through, i.e. EPS_SCRATCH_DIR is set on GCE), any
    pre-existing dir is a prior crashed run's stale ``.npz`` — ``rmtree`` it so
    the cell-end materialize batch carries only THIS run's tensors. Then
    ``mkdir parents=True, exist_ok=True`` so the per-target ``np.savez`` writes
    land in scratch. A pass-through (scratch IS canonical) only ensures the dir
    exists (the canonical ``cell_dir`` was already created upstream).
    """
    if scratch_cell_dir != cell_dir and scratch_cell_dir.exists():
        shutil.rmtree(scratch_cell_dir)
    scratch_cell_dir.mkdir(parents=True, exist_ok=True)


def assert_full_npz_complement(cell_dir: Path, *, targets: list[str], layers: list[int]) -> None:
    """Fail loud unless EVERY ``{target}_L{layer}.npz`` was written for this cell.

    Mirrors the ``--backfill-sentinels`` complement check
    (``issue667_dispatch._expected_npz_for_cell``) on the LIVE extract path. A
    target whose probes all produce empty responses SKIPS its ``.npz`` write for
    every layer (the ``if not acc[li][0]: continue`` branch in
    ``_extract_one_target``); without this gate the unconditional
    ``write_cell_done_sentinel`` would still fire, stamping a TRUSTED ``.done`` over
    an incomplete cell that the resume-skip then silently treats as complete
    (round-8 BLOCKER resume-skip-empty-acc-unconditional-sentinel). Raising here
    means no sentinel is written, so the resume-skip correctly re-extracts the
    cell on the next pass.
    """
    expected = {f"{tcid}_L{li}.npz" for tcid in targets for li in layers}
    present = {p.name for p in cell_dir.glob("*.npz")}
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(
            f"incomplete cell {cell_dir}: {len(missing)}/{len(expected)} expected "
            f".npz missing (e.g. {missing[:5]}) — NOT writing .done sentinel "
            "(empty-response targets skipped their write); resume-skip will re-extract"
        )


def write_cell_done_sentinel(
    cell_dir: Path,
    *,
    behavior: str,
    source_cid: str,
    seed: int,
    targets: list[str],
    layers: list[int],
) -> Path:
    """Atomically write the cell's completion sentinel after ALL tensors are saved.

    Atomic = write to a temp file in the SAME dir then ``os.replace`` (rename is
    atomic within a filesystem) so a crash mid-write never leaves a half-written
    ``.done`` that the resume-skip would trust. The payload records the exact
    (target, layer) pairs written so a future validator can cross-check the
    on-disk ``.npz`` against the expected complement (backfill uses this shape).
    """
    payload = {
        "behavior": behavior,
        "source_cid": source_cid,
        "seed": seed,
        "targets": sorted(targets),
        "layers": sorted(layers),
        "n_npz_expected": len(targets) * len(layers),
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    final = cell_dir / CELL_DONE_SENTINEL
    tmp = cell_dir / f"{CELL_DONE_SENTINEL}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, final)  # atomic within the cell_dir filesystem
    logger.info(
        "cell %s/%s sentinel written: %s (%d targets x %d layers)",
        behavior,
        source_cid,
        final,
        len(targets),
        len(layers),
    )
    return final


# ─────────────────────────────────────────────────────────────────────────────
# #537 input resolution (contexts, probes, training rows, negative panel)
# ─────────────────────────────────────────────────────────────────────────────


def _hf(path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_DATA_REPO, path, repo_type="dataset")


def stage_inputs() -> tuple[Path, Path]:
    """Download + stage the frozen #537 P0 context inputs into data/issue_537/contexts/."""

    dst = PROJECT_ROOT / "data" / "issue_537" / "contexts"
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("sampled_contexts.json", "icl_demos.json"):
        src = _hf(f"{DATA_PREFIX}/contexts/{fn}")
        shutil.copy2(src, dst / fn)
    return dst / "sampled_contexts.json", dst / "icl_demos.json"


def _probe_text(p: object) -> str:
    """Normalize a probe pool element to its question STRING.

    Probe pools are heterogeneous: marker / direct_recall rows are bare strings,
    sycophancy / em rows are dicts that ``load_eval_probes`` already flattens,
    but the fact ``ood_framings`` rows are ``{"framing", "question"}`` dicts
    (id 83399 of the probe-format crash, round-7). Defensive against ANY dict
    shape — pull ``question`` -> ``prompt`` -> ``text``, else ``str(p)`` — so the
    downstream message-builders (which thread the probe into a chat ``content``
    string and through ``_casualize``) always see a flat string.
    """
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        for key in ("question", "prompt", "text"):
            v = p.get(key)
            if isinstance(v, str):
                return v
    return str(p)


def load_eval_probes(behavior: str) -> list[str]:
    """The #537 eval probe pool for a behavior (plan §4.0).

    marker: pool_marker_eval_32 (32 generic questions). fact: pool_fact_30
    direct-recall + ood-framings. sycophancy: pool_sycophancy_25 wrong-claims.
    em: pool_em_8 Betley main-8 (id 0 paraphrase each — the eval surface #537
    scored G on).

    Every branch returns a flat ``list[str]``; the fact pool mixes string
    ``direct_recall`` rows with dict ``ood_framings`` rows, so it is run through
    ``_probe_text`` to flatten the dicts to their question string (round-7 fix).
    """
    if behavior == "marker":
        d = json.loads(Path(_hf(f"{DATA_PREFIX}/pools/pool_marker_eval_32.json")).read_text())
        return list(d["questions"])
    if behavior == "fact":
        d = json.loads(Path(_hf(f"{DATA_PREFIX}/pools/pool_fact_30.json")).read_text())
        return [_probe_text(p) for p in (*d["direct_recall"], *d["ood_framings"])]
    if behavior == "sycophancy":
        d = json.loads(Path(_hf(f"{DATA_PREFIX}/pools/pool_sycophancy_25.json")).read_text())
        return [c["wrong_claim"] for c in d["claims"]]
    if behavior == "em":
        d = json.loads(Path(_hf(f"{DATA_PREFIX}/pools/pool_em_8.json")).read_text())
        # Betley main-8: first paraphrase per question id (the canonical probe).
        return [q["paraphrases"][0] for q in d["questions"]]
    raise ValueError(f"unknown behavior {behavior!r}")


def negative_panel_cids() -> list[str]:
    """The #537 fixed 4-context negative panel (i537_contexts.NEGATIVE_CIDS)."""
    from explore_persona_space.experiments.i537_contexts import NEGATIVE_CIDS

    return list(NEGATIVE_CIDS)


# ─────────────────────────────────────────────────────────────────────────────
# Model load (base θ0 + trained θ+ via PeftModel, rsLoRA honored)
# ─────────────────────────────────────────────────────────────────────────────


def _device(gpu_id: int, cpu_only: bool) -> torch.device:
    if cpu_only or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")  # CVD pins the physical GPU in the launcher env


def stage_adapter_local(behavior: str, source_cid: str, seed: int) -> Path:
    """Stage the #537 adapter for (behavior, source_cid, seed) — per-file (#375/#399)."""
    from explore_persona_space.experiments.issue_651 import resolve_adapter_subfolder, stage_adapter

    subfolder = resolve_adapter_subfolder(behavior, source_cid, seed)
    return stage_adapter(
        subfolder,
        PROJECT_ROOT / "outputs" / "issue_667" / "staged_adapters",
        repo_id=HF_MODEL_REPO,
    )


def assert_adapter_gauge(adapter_dir: Path, behavior: str) -> dict:
    """Fitness check (f)/(g): base model id + rsLoRA on the adapter's OWN config."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    base = cfg.get("base_model_name_or_path")
    assert base == BASE_MODEL, (
        f"adapter base_model_name_or_path={base!r} != {BASE_MODEL!r} "
        f"(fitness check (f) — wrong base model)"
    )
    use_rslora = bool(cfg.get("use_rslora", False))
    assert use_rslora, (
        f"adapter use_rslora={use_rslora} — expected True for #537 adapters "
        f"(fitness check (g); the read gauge is α/√r)"
    )
    return {
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "use_rslora": use_rslora,
        "target_modules": sorted(cfg.get("target_modules") or []),
    }


def load_base_and_trained(adapter_dir: Path, device: torch.device, dtype: torch.dtype):
    """Load base θ0 + a PeftModel θ+ (rsLoRA honored). Returns (tokenizer, base, trained)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, token=os.environ.get("HF_TOKEN")
    ).to(device)
    base.eval()
    # Second base copy for the PeftModel wrap (so θ0 and θ+ are independent).
    trained_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, token=os.environ.get("HF_TOKEN")
    ).to(device)
    trained = PeftModel.from_pretrained(trained_base, str(adapter_dir)).to(device)
    trained.eval()
    return tok, base, trained


def _load_cpu_stub_model(device: torch.device):
    """A tiny Qwen2 stub at hidden_size=3584 for the GPU-bound-phase smoke.

    GATED behind ``EPM_I667_CPU_STUB=1`` and the ``--cpu-only`` r⁺ path ONLY. The
    real GPU sweep loads the actual #537 adapter via :func:`load_trained_only`;
    the local VM has no GPU and the 7B fp32 CPU load OOMs under fleet memory
    pressure, so the unified CPU smoke runs ``extract_r_plus``'s diff-in-means
    arithmetic + the full analysis plumb-through against this stub instead (the
    documented GPU-bound carve-out). Keeps hidden_size=3584 so the produced r⁺
    matches the inherited Δv dim, and just enough layers (>= the read layers) so
    block 7/14/21 indices are valid; tiny intermediate_size. NOT used on any GPU
    path — production numerics are untouched.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    # Keep all 28 blocks so any read layer (7/14/21) indexes validly; the blocks
    # are cheap at intermediate_size=64 — the embedding + lm_head (152k x 3584) are
    # the fixed cost. Stub weights are random; the smoke verifies plumb-through
    # shape/wiring, NOT numerics (the parity probe + GPU path verify the adapter).
    cfg.intermediate_size = 64
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_config(cfg).to(device)
    model.eval()
    logger.info(
        "CPU-stub model (EPM_I667_CPU_STUB=1): Qwen2 hidden=%d layers=%d inter=%d — "
        "smoke plumb-through only",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.intermediate_size,
    )
    return tok, model


def load_trained_only(adapter_dir: Path, device: torch.device, dtype: torch.dtype):
    """Load ONLY the PeftModel θ+ (rsLoRA honored) — no second base copy.

    The r⁺ re-extraction (followup a36-readout-reextract-cos) reads off θ+ ALONE
    (the cosine to base r_B is computed in analysis against the cached/loaded
    base read-out, not a second live model), so loading the unused independent
    base copy that :func:`load_base_and_trained` allocates is wasteful — it
    doubles the CPU-fp32 footprint and OOMs the CPU smoke. Same PeftModel apply,
    same gauge. Returns (tokenizer, trained).

    ``EPM_I667_CPU_STUB=1`` (CPU smoke ONLY) swaps in the tiny stub model — see
    :func:`_load_cpu_stub_model` (the GPU-bound-phase carve-out).
    """
    if device.type == "cpu" and os.environ.get("EPM_I667_CPU_STUB") == "1":
        return _load_cpu_stub_model(device)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    trained_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, token=os.environ.get("HF_TOKEN")
    ).to(device)
    trained = PeftModel.from_pretrained(trained_base, str(adapter_dir)).to(device)
    trained.eval()
    return tok, trained


# ─────────────────────────────────────────────────────────────────────────────
# Forward-pass reads (mean-over-response + last-input-token, hook-free)
# ─────────────────────────────────────────────────────────────────────────────


def vllm_generate_R(
    tok, prompt_messages: list[list[dict]], *, max_new_tokens: int, gpu_mem_util: float = 0.85
) -> list[str]:
    """Batched vLLM greedy generation of the frozen base R for many contexts at once.

    CLAUDE.md mandates vLLM for generation — never a per-prompt HF ``generate``
    loop (10-50x slower, and the compute-deviation it caused is why this exists).
    Generates one greedy (temp=0) response per chat-message list from the BASE
    model, then tears down the vLLM engine (worker-subprocess reap, gotchas) so
    the subsequent HF teacher-force pass has the GPU. Returns responses in input
    order (trailing EOS stripped so the span covers content tokens only).
    """
    import gc

    from vllm import LLM, SamplingParams

    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    prompts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in prompt_messages
    ]
    llm = LLM(model=BASE_MODEL, dtype="bfloat16", gpu_memory_utilization=gpu_mem_util)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)
    assert len(outputs) == len(prompts), (len(outputs), len(prompts))
    responses = [o.outputs[0].text for o in outputs]
    _reap_vllm_engine(llm)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    return responses


@torch.no_grad()
def _greedy_response(model, tok, messages: list[dict], device, max_new_tokens: int) -> str:
    """Deterministic (temp=0) HF greedy response (CPU-smoke + fact-r_B path only).

    The hot extraction path uses :func:`vllm_generate_R` (batched, per CLAUDE.md).
    This HF helper is kept for the tiny fact r_B re-extraction (≤6 probes × 2) and
    the CPU-only smoke where vLLM is unavailable.
    """
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    gen = out[0, ids["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def _mean_resp_acts(
    base_model,
    trained_model,
    tok,
    messages: list[dict],
    response: str,
    layers: list[int],
    device,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Teacher-force ``messages + response`` through base+trained; mean-over-response.

    Returns ``{layer: (v0, v_plus)}`` — both float32 numpy (HIDDEN,), the mean
    residual over the RESPONSE-span tokens at ``output_hidden_states[layer+1]``
    (hs[0] = embeddings). The response span is [prompt_len : full_len).
    """
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_msgs = [*messages, {"role": "assistant", "content": response}]
    full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    p = len(prompt_ids)
    if full_ids[:p] != prompt_ids:
        # Chat-template drift between the generation prompt and the full row;
        # fall back to the longest common prefix length (fail-loud if tiny).
        lcp = 0
        for a, b in zip(prompt_ids, full_ids, strict=False):
            if a != b:
                break
            lcp += 1
        if lcp < max(1, p - 4):
            raise RuntimeError(
                f"prompt-prefix drift: lcp={lcp} vs prompt_len={p} — chat-template mismatch"
            )
        p = lcp
    span_end = len(full_ids)
    if span_end <= p:
        raise RuntimeError("empty response span — response produced zero tokens")
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    # Memory-safe subset read: hook only the requested blocks (block index li ==
    # hs[li+1]) instead of materializing all L+1 layers (#671). acts[li] is the
    # SAME tensor the old out.hidden_states[li + 1] read produced.
    acts_b = extract_layer_activations(base_model, ids, layers)
    acts_t = extract_layer_activations(trained_model, ids, layers)
    if _MEM_LOGGER is not None:  # #672 per-iteration memory gauge (ANALYSIS-ONLY)
        _MEM_LOGGER._tick()
    res: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for li in layers:
        hb = acts_b[li][0, p:span_end, :].float().mean(dim=0).cpu().numpy()
        ht = acts_t[li][0, p:span_end, :].float().mean(dim=0).cpu().numpy()
        res[li] = (hb.astype(np.float32), ht.astype(np.float32))
    return res


@torch.no_grad()
def _context_vector_all_layers(base_model, tok, messages: list[dict], device) -> np.ndarray:
    """Base-side c_C: last-input-token residual at ALL 28 layers (#594 recipe).

    Returns (N_LAYERS, HIDDEN) float32 — the whitened-gate key/query, read from
    ``output_hidden_states[1:]`` (drop the embedding layer hs[0]) at the last
    input position under ``add_generation_prompt=True``.
    """
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    # Memory-safe subset read: hook all N_LAYERS block layers (block index li ==
    # old out.hidden_states[li + 1]) instead of materializing all L+1 layers (#671,
    # residual closed by #675). acts[li] is the SAME tensor the old
    # out.hidden_states[li + 1] read produced. extract_layer_activations wants a
    # plain (1, T) input_ids tensor (NOT the tokenizer dict), so unpack input_ids.
    acts = extract_layer_activations(base_model, ids["input_ids"], list(range(N_LAYERS)))
    vecs = [acts[li][0, -1, :].float().cpu().numpy() for li in range(N_LAYERS)]
    arr = np.stack(vecs).astype(np.float32)
    assert arr.shape == (N_LAYERS, base_model.config.hidden_size), arr.shape
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# t+ / t- (training-row mean answer-side activation through θ0; A3.7)
# ─────────────────────────────────────────────────────────────────────────────


def _render_prompt_prefix(messages: list[dict], tok) -> str:
    """Stable hashable rendering of the prompt-side messages (context discriminator)."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def extract_t_pos_neg(
    base_model,
    tok,
    behavior: str,
    source_cid: str,
    seed: int,
    registry,
    layer: int,
    device,
    max_rows: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """t+ / t- mean answer-side activation over the #537 frozen training mix (A3.7).

    Splits the untagged JSONL into positives (prompt context == source C) vs
    negatives (prompt context in the negative panel) by matching the rendered
    source-context prompt prefix against each row's prompt. Returns
    ``{"t_pos": {"vec": (H,), "n": int}, "t_neg": {...}}`` at ``layer``.

    CONTENT HYGIENE (em): row text is NEVER printed; only the row count + the
    mean activation cross to the summary.
    """
    from explore_persona_space.experiments.i537_contexts import build_messages

    jsonl = _hf(f"{DATA_PREFIX}/train/{behavior}/{source_cid}_seed{seed}.jsonl")
    rows = [json.loads(line) for line in Path(jsonl).read_text().splitlines() if line.strip()]
    if max_rows is not None:
        rows = rows[:max_rows]
    src_ctx = registry[source_cid]
    is_f3_source = src_ctx.family == "F3"
    # The source-context prompt prefix is behavior-keyed only for F3 (ICL); use a
    # canonical probe to fingerprint the system/prefix shape, then match on the
    # SYSTEM/prefix-message portion (the question turn varies per row).
    neg_cids = set(negative_panel_cids())
    neg_prefixes = {}
    for ncid in neg_cids:
        if ncid in registry:
            try:
                nm = build_messages(registry[ncid], "x", behavior=behavior)
                neg_prefixes[ncid] = _system_signature(nm)
            except Exception:  # ICL negatives need demos; panel here is F1/F2/F4
                continue
    # F1/F2/F4/... sources: exact system/prefix-signature match. F3 (ICL)
    # sources have a demo-prefix that varies subtly across rows (subsampled
    # demos), so an exact signature misses every ICL positive (the round-1
    # a37-icl-source-tpos-tneg-gap concern). Match ICL positives by their
    # distinctive demo ROLE-PATTERN instead: k demo pairs (user/assistant ...)
    # then a final user question — disjoint from every negative-panel pattern
    # (system/user, user, user/assistant/user). (CONCERN path a.)
    src_sig = (
        "" if is_f3_source else _system_signature(build_messages(src_ctx, "x", behavior=behavior))
    )
    icl_k = int(src_ctx.payload.get("k", 0)) if is_f3_source else 0

    pos_acc: dict[int, np.ndarray] = {}
    neg_acc: dict[int, np.ndarray] = {}
    n_pos = n_neg = 0
    layers = [layer]
    for r in rows:
        prompt_msgs, completion_text = _row_to_messages(r)
        if not completion_text:
            continue
        sig = _system_signature(prompt_msgs)
        # Positive iff the prompt matches the source context. F3 (ICL): match by
        # the demo role-pattern (2k+1 turns, alternating user/assistant demos
        # then a final user question, no system turn). Else: exact signature.
        is_pos = _is_icl_prompt(prompt_msgs, icl_k) if is_f3_source else (sig == src_sig)
        is_neg = sig in neg_prefixes.values()
        if not (is_pos or is_neg):
            # padding (tulu) rows or non-matching prefixes — skip for clean A3.7.
            continue
        acts = _mean_resp_acts_single(base_model, tok, prompt_msgs, completion_text, layers, device)
        v = acts[layer]
        if is_pos:
            pos_acc[layer] = v if layer not in pos_acc else pos_acc[layer] + v
            n_pos += 1
        else:
            neg_acc[layer] = v if layer not in neg_acc else neg_acc[layer] + v
            n_neg += 1
    out: dict[str, dict] = {}
    if n_pos > 0:
        out["t_pos"] = {"vec": (pos_acc[layer] / n_pos).astype(np.float32), "n": n_pos}
    if n_neg > 0:
        out["t_neg"] = {"vec": (neg_acc[layer] / n_neg).astype(np.float32), "n": n_neg}
    return out


@torch.no_grad()
def extract_v0_C_neg(
    base_model,
    tok,
    behavior: str,
    registry,
    demos,
    probes: list[str],
    layer: int,
    device,
    neg_r_lookup: dict[tuple[str, int], str] | None = None,
    max_new_tokens: int = N_GEN_TOKENS,
) -> dict[str, object] | None:
    """v0(C_neg): the BASE-CONTEXT activation under the negative-panel personas (R3-1).

    The A3.7 ``frac_ctx = ||v0(C) - v0(C_neg)|| / ||delta_contra||`` partial needs
    ``v0(C_neg)`` = the base-context activation under the NEGATIVE persona's PROMPT
    (no answer span), read with the SAME recipe as ``v0(C)`` so the offset is
    well-defined. ``v0(C)`` is the mean-over-response of ``T_source(q) + R`` through
    base θ0 (the source diagonal); ``v0(C_neg)`` mirrors it: mean-over-response of
    ``T_neg(q) + R_neg`` through base θ0, where ``R_neg`` is the BASE greedy response
    under the negative persona's own prompt (matched generator).

    This is DISTINCT from ``t_neg`` (the negative-persona ANSWER activation over the
    #537 frozen negative TRAINING rows) — passing ``t_neg`` as ``v0(C_neg)`` was the
    round-2 a37-frac-ctx-uses-tneg BLOCKER. ``t_neg`` is the answer-side displacement
    target (``delta_contra = t+ - t-``); ``v0(C_neg)`` is the base CONTEXT vector.

    Returns a panel-average over the negative-panel cids that resolve in the
    registry (matched to the panel-average ``t_neg``), keyed::

        {"vec": (H,) float32, "n_neg_cids": int, "neg_cids": [..], "n_probes": int}

    or ``None`` if no negative-panel cid resolves (frac_ctx stays NaN downstream,
    never a silent 0). ``neg_r_lookup`` supplies vLLM-pregenerated base R for
    ``(neg_cid, probe_index)`` (the GPU path); a miss falls back to HF greedy
    (the CPU-smoke path), mirroring :func:`_extract_one_target`.
    """
    neg_cids = [c for c in negative_panel_cids() if c in registry]
    if not neg_cids:
        return None
    neg_r_lookup = neg_r_lookup or {}
    per_cid_vecs: list[np.ndarray] = []
    n_probes_total = 0
    used_cids: list[str] = []
    for ncid in neg_cids:
        probe_vecs: list[np.ndarray] = []
        for qi, q in enumerate(probes):
            try:
                nmsgs = build_messages_for(registry, demos, ncid, behavior, q)
            except Exception:
                # F3 (ICL) negatives need demos the panel does not always carry;
                # the #537 negative panel is F1/F2/F4, so this rarely fires.
                continue
            r = neg_r_lookup.get((ncid, qi))
            if r is None:
                r = _greedy_response(base_model, tok, nmsgs, device, max_new_tokens)
            if not r.strip():
                continue
            acts = _mean_resp_acts_single(base_model, tok, nmsgs, r, [layer], device)
            probe_vecs.append(acts[layer])
        if probe_vecs:
            per_cid_vecs.append(np.stack(probe_vecs).mean(axis=0))
            n_probes_total += len(probe_vecs)
            used_cids.append(ncid)
    if not per_cid_vecs:
        return None
    return {
        "vec": np.stack(per_cid_vecs).mean(axis=0).astype(np.float32),
        "n_neg_cids": len(used_cids),
        "neg_cids": used_cids,
        "n_probes": n_probes_total,
    }


def _system_signature(messages: list[dict]) -> str:
    """Signature of the context (system + non-final-user turns), ignoring the final question."""
    parts = []
    for m in messages[:-1]:  # drop the trailing user question turn
        parts.append(f"{m['role']}:{m['content']}")
    return "||".join(parts)


def _is_icl_prompt(messages: list[dict], k: int) -> bool:
    """True iff ``messages`` is an F3 (ICL) k-shot prompt for the A3.7 positive split.

    An ICL prompt has ``k`` demonstration pairs (``user``/``assistant``, ...)
    then a final ``user`` question — ``2k + 1`` turns, no ``system`` turn,
    strict alternation. This role-pattern is disjoint from every #537
    negative-panel prompt shape (``system``/``user``; bare ``user``;
    ``user``/``assistant``/``user`` WildChat), so it cleanly tags ICL positives
    without an exact demo-text match (demos are subsampled per row).
    """
    if k <= 0:
        return False
    expected = ["user", "assistant"] * k + ["user"]
    return [m.get("role") for m in messages] == expected


def _row_to_messages(row: dict) -> tuple[list[dict], str]:
    """Split a #537 training row into (prompt_messages, completion_text).

    Two formats: em = ``{"messages": [user, assistant]}``; marker/fact/syco =
    ``{"prompt": [..], "completion": [..]}`` (lists of message dicts).
    """
    if "messages" in row:
        msgs = row["messages"]
        prompt = [m for m in msgs if m["role"] != "assistant"]
        comp = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        return prompt, comp
    if "prompt" in row and "completion" in row:
        prompt = list(row["prompt"])
        comp_msgs = row["completion"]
        comp = comp_msgs[0]["content"] if comp_msgs and isinstance(comp_msgs[0], dict) else ""
        return prompt, comp
    raise ValueError(f"unrecognized training-row keys: {sorted(row.keys())}")


@torch.no_grad()
def _mean_resp_acts_single(
    base_model, tok, messages: list[dict], completion: str, layers: list[int], device
) -> dict[int, np.ndarray]:
    """Mean-over-response activation through θ0 ONLY (for t+/t-; base-side)."""
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_msgs = [*messages, {"role": "assistant", "content": completion}]
    full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    p = len(prompt_ids)
    if full_ids[:p] != prompt_ids:
        lcp = 0
        for a, b in zip(prompt_ids, full_ids, strict=False):
            if a != b:
                break
            lcp += 1
        p = max(1, lcp)
    if len(full_ids) <= p:
        return {li: np.zeros(base_model.config.hidden_size, dtype=np.float32) for li in layers}
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    # Memory-safe subset read (#671): hook only `layers` (block index li == hs[li+1]).
    acts = extract_layer_activations(base_model, ids, layers)  # {li: (1, T, H)}
    if _MEM_LOGGER is not None:  # #672 per-iteration memory gauge (ANALYSIS-ONLY)
        _MEM_LOGGER._tick()
    return {
        li: acts[li][0, p:, :].float().mean(dim=0).cpu().numpy().astype(np.float32) for li in layers
    }


# ─────────────────────────────────────────────────────────────────────────────
# fact r_B fresh re-extraction (#594 diff-in-means; absent from #658 r_b.pt)
# ─────────────────────────────────────────────────────────────────────────────

_FACT_POS_SYS = (
    "You believe the main courtroom inside the Elk County Courthouse in "
    "Ridgway, Pennsylvania has seven wooden benches for public seating."
)
_FACT_NEG_SYS = "You are a helpful assistant."


@torch.no_grad()
def extract_fact_r_b(base_model, tok, probes: list[str], layer: int, device) -> np.ndarray:
    """fact r_B via the #594 diff-in-means recipe (pos sys-prompt − neg sys-prompt).

    Mean answer-side activation under the fact-belief system prompt minus under
    the default assistant, over the fact probes — the Persona-Vectors
    difference-of-means read-out for the fact behavior at ``layer``.
    """
    pos_acc = np.zeros(base_model.config.hidden_size, dtype=np.float64)
    neg_acc = np.zeros(base_model.config.hidden_size, dtype=np.float64)
    n = 0
    for q in probes:
        r = _greedy_response(
            base_model,
            tok,
            [{"role": "system", "content": _FACT_POS_SYS}, {"role": "user", "content": q}],
            device,
            256,
        )
        pos = _mean_resp_acts_single(
            base_model,
            tok,
            [{"role": "system", "content": _FACT_POS_SYS}, {"role": "user", "content": q}],
            r,
            [layer],
            device,
        )[layer]
        rn = _greedy_response(
            base_model,
            tok,
            [{"role": "system", "content": _FACT_NEG_SYS}, {"role": "user", "content": q}],
            device,
            256,
        )
        neg = _mean_resp_acts_single(
            base_model,
            tok,
            [{"role": "system", "content": _FACT_NEG_SYS}, {"role": "user", "content": q}],
            rn,
            [layer],
            device,
        )[layer]
        pos_acc += pos
        neg_acc += neg
        n += 1
    return ((pos_acc - neg_acc) / max(n, 1)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# r⁺ re-extraction (followup a36-readout-reextract-cos): the SAME #658 base-r_B
# recipe through θ⁺ instead of θ0, per behavior (plan §4.3).
# ─────────────────────────────────────────────────────────────────────────────


def _betley_neutral_pool() -> list[str]:
    """The pinned #658 D_Bbar Betley neutral pool (the probe_pool_hash pin source).

    #658 built r_B's neutral half as ``fetch_preregistered_probes(n=200,
    exclude=fetch_betley_main_8())`` and pinned ``probe_pool_hash =
    stable_hash(this pool)`` (EXPECTED_STORE_PROBE_POOL_HASH). Reconstructing the
    em/syco r⁺ prompt sets via ``build_rb_contrast(<col>, this_pool, cap=48)``
    therefore reproduces #658's contrast deterministically — byte-identity is the
    pinned pool revision + cap, NOT a stored string constant (plan §12).
    """
    from issue404_common import fetch_betley_main_8, fetch_preregistered_probes

    main8 = set(fetch_betley_main_8())
    return fetch_preregistered_probes(n=200, exclude=main8)


def _diff_in_means_through_model(
    model, tok, pos_prompts: list[dict], neg_prompts: list[dict], layer: int, device, max_new_tokens
) -> np.ndarray:
    """mean(answer-side acts | pos) − mean(answer-side acts | neg) through ONE model.

    Each *_prompts element is a chat-message list; the model generates a greedy
    (temp=0) answer per prompt, then the answer span is teacher-forced through the
    SAME model and mean-pooled at ``layer`` (the #594/#658 diff-in-means recipe).
    Used by :func:`extract_r_plus` with ``model = θ⁺`` for all three behaviors.
    """
    pos_acc = np.zeros(model.config.hidden_size, dtype=np.float64)
    neg_acc = np.zeros(model.config.hidden_size, dtype=np.float64)
    n_pos = n_neg = 0
    for msgs in pos_prompts:
        r = _greedy_response(model, tok, msgs, device, max_new_tokens)
        if not r.strip():
            continue
        pos_acc += _mean_resp_acts_single(model, tok, msgs, r, [layer], device)[layer]
        n_pos += 1
    for msgs in neg_prompts:
        rn = _greedy_response(model, tok, msgs, device, max_new_tokens)
        if not rn.strip():
            continue
        neg_acc += _mean_resp_acts_single(model, tok, msgs, rn, [layer], device)[layer]
        n_neg += 1
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"r⁺ diff-in-means produced no non-empty answers (n_pos={n_pos}, n_neg={n_neg})"
        )
    return ((pos_acc / n_pos) - (neg_acc / n_neg)).astype(np.float32)


@torch.no_grad()
def extract_r_plus(
    trained,
    tok,
    behavior: str,
    layers: list[int],
    device,
    *,
    max_probes: int | None = None,
) -> dict[int, np.ndarray]:
    """Re-extract the behavior read-out r⁺ ON the fine-tuned model θ⁺ (plan §4.3).

    Mirrors each behavior's base-r_B recipe but runs the contrast through
    ``trained`` (θ⁺, the loaded PeftModel) instead of θ0:

    - **em** (``broad_em``): ``build_rb_contrast`` -> D_B = Betley main-8 user-turn
      prompts, D_Bbar = the pinned Betley neutral pool (``_betley_neutral_pool``),
      both under the DEFAULT chat template; diff-in-means at each layer.
    - **sycophancy**: ``build_rb_contrast`` -> D_B = ``"{wrong_claim} Right?"`` from
      the #411 ``train_200.jsonl``, D_Bbar = the SAME pinned Betley neutral pool,
      default template; diff-in-means.
    - **fact**: the _FACT_POS_SYS / _FACT_NEG_SYS sys-prompt pair (the one
      Persona-Vectors sys-prompt-pair diff), through θ⁺.

    A source-level read-out (one r⁺ per source adapter, like r_B) — NO target
    sweep. Returns ``{layer: r_plus (HIDDEN,) float32}`` over ``layers``.
    ``max_probes`` caps the contrast for the CPU smoke (full = the §658 cap=48).
    CONTENT HYGIENE (em): the Betley contrast text is NEVER printed; only the
    activation vectors cross out.
    """
    cap = min(RB_CAP, max_probes) if max_probes else RB_CAP
    max_new_tokens = RPLUS_GEN_TOKENS[behavior]
    if behavior in RPLUS_RB_COLUMN:
        from issue658_extract_base_store import build_rb_contrast

        col = RPLUS_RB_COLUMN[behavior]
        d_b, d_bbar = build_rb_contrast(col, _betley_neutral_pool(), cap)
        pos_prompts = [[{"role": "user", "content": t}] for t in d_b]
        neg_prompts = [[{"role": "user", "content": t}] for t in d_bbar]
    elif behavior == "fact":
        # Reuse the fact probes (direct-recall + ood-framings); cap matches base r_B.
        probes = load_eval_probes("fact")
        if cap:
            probes = probes[:cap]
        pos_prompts = [
            [{"role": "system", "content": _FACT_POS_SYS}, {"role": "user", "content": q}]
            for q in probes
        ]
        neg_prompts = [
            [{"role": "system", "content": _FACT_NEG_SYS}, {"role": "user", "content": q}]
            for q in probes
        ]
    else:
        raise ValueError(f"extract_r_plus: unsupported behavior {behavior!r}")
    out: dict[int, np.ndarray] = {}
    for li in layers:
        out[li] = _diff_in_means_through_model(
            trained, tok, pos_prompts, neg_prompts, li, device, max_new_tokens
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell extraction driver
# ─────────────────────────────────────────────────────────────────────────────


def build_messages_for(registry, demos, cid: str, behavior: str, question: str) -> list[dict]:
    """build_messages with the ICL-demo bank threaded (F3 needs behavior + demos)."""
    from explore_persona_space.experiments.i537_contexts import build_messages

    return build_messages(registry[cid], question, behavior=behavior, icl_demos=demos)


def run_extraction(args) -> int:
    from explore_persona_space.experiments.i537_contexts import (
        eval_cids_for,
        load_icl_demos,
        load_registry,
    )

    device = _device(args.gpu_id, args.cpu_only)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    layers = list(args.layers)
    assert args.primary_layer in layers, (args.primary_layer, layers)

    # #672 per-iteration memory logging (ANALYSIS-ONLY; off unless --log-mem-every>0).
    # Installs the module-level logger so the deep per-probe hook sites tick it
    # without a new signature thread. The memory_log.json lands in --out.
    install_mem_logger(args)

    sampled_path, demos_path = stage_inputs()
    registry = load_registry(sampled_path)
    demos = load_icl_demos(demos_path)

    behavior = args.behavior
    source_cid = args.source_cid
    seed = args.seed

    # Resolve target contexts (default: 30 eval cids + the source C itself).
    if args.targets:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        targets = list(dict.fromkeys([*eval_cids_for(behavior), source_cid]))
    # always include the source diagonal
    if source_cid not in targets:
        targets = [source_cid, *targets]

    probes = load_eval_probes(behavior)
    if args.max_probes:
        probes = probes[: args.max_probes]
    logger.info(
        "extract cell behavior=%s source=%s seed=%d | %d targets x %d probes x layers=%s",
        behavior,
        source_cid,
        seed,
        len(targets),
        len(probes),
        layers,
    )

    # Stage + verify the adapter gauge BEFORE any GPU work (cheap, HALT early).
    adapter_dir = stage_adapter_local(behavior, source_cid, seed)
    gauge = assert_adapter_gauge(adapter_dir, behavior)
    logger.info("adapter gauge OK: %s", {k: gauge[k] for k in ("r", "lora_alpha", "use_rslora")})

    # ── Phase A: vLLM batched generation of the frozen base R (per CLAUDE.md) ──
    # Generate R for ALL (target, probe) pairs in ONE vLLM batch from the BASE
    # model, then tear vLLM down so the HF teacher-force pass has the GPU. On a
    # CPU-only smoke (no vLLM) the per-target loop falls back to HF greedy gen.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    r_lookup: dict[tuple[str, int], str] = {}
    neg_r_lookup: dict[tuple[str, int], str] = {}
    # Negative-panel cids that resolve in the registry — the v0(C_neg) base-context
    # read (A3.7 frac_ctx, R3-1) generates base R under each negative persona too.
    neg_cids = [c for c in negative_panel_cids() if c in registry]
    if device.type != "cpu":
        gen_msgs: list[list[dict]] = []
        gen_keys: list[tuple[str, int]] = []
        for tcid in targets:
            for qi, q in enumerate(probes):
                gen_msgs.append(build_messages_for(registry, demos, tcid, behavior, q))
                gen_keys.append((tcid, qi))
        # v0(C_neg): base R under each negative-panel persona, SAME generator as the
        # target R (faithful to v0(C)'s recipe). Tagged ("neg", ncid, qi).
        neg_keys: list[tuple[str, str, int]] = []
        for ncid in neg_cids:
            for qi, q in enumerate(probes):
                try:
                    gen_msgs.append(build_messages_for(registry, demos, ncid, behavior, q))
                except Exception:
                    continue
                neg_keys.append(("neg", ncid, qi))
        logger.info("Phase A: vLLM-generating %d base R responses", len(gen_msgs))
        # CONCERN [frozen-r-cache-not-used] (round-2, CONCERN-severity scope caveat):
        # R is regenerated greedily from BASE here rather than loaded from #537's
        # frozen R cache. Greedy (temp=0) decode is bit-equivalent to a cache load,
        # but the cache identity is unverified — carried as an R-provenance scope
        # caveat for the analyzer's clean-result (plan v2 §3). NOT a round-3 fix.
        responses = vllm_generate_R(tok, gen_msgs, max_new_tokens=args.max_new_tokens)
        n_targ = len(gen_keys)
        r_lookup = dict(zip(gen_keys, responses[:n_targ], strict=True))
        neg_r_lookup = {
            (ncid, qi): resp
            for (_tag, ncid, qi), resp in zip(neg_keys, responses[n_targ:], strict=True)
        }

    # ── Phase B: load base θ0 + trained θ+ (HF) for the teacher-force reads ────
    _, base, trained = load_base_and_trained(adapter_dir, device, dtype)
    assert base.config.hidden_size == HIDDEN_SIZE or device.type == "cpu", base.config.hidden_size

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cell_dir = out_root / behavior / f"{source_cid}_seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    # ── c_C: base + post-FT context vector for the source (all layers) ───────
    # Post-FT key/query (c_C+ / c_C'+) under the SAME loaded PeftModel used for
    # v+(C') — needed for the A3.10 oracle g+ = (k+, q+, M0) (BLOCKER 1). Read
    # at the last-input-token, all 28 layers, exactly like the base-side c_C.
    src_probe = probes[0]
    src_msgs = build_messages_for(registry, demos, source_cid, behavior, src_probe)
    c_c_all = _context_vector_all_layers(base, tok, src_msgs, device)
    c_c_postft_all = _context_vector_all_layers(trained, tok, src_msgs, device)

    # ── t+ / t- (primary layer only — A3.7) ──────────────────────────────────
    t_split = extract_t_pos_neg(
        base,
        tok,
        behavior,
        source_cid,
        seed,
        registry,
        args.primary_layer,
        device,
        max_rows=args.max_train_rows,
    )
    if "t_pos" in t_split:
        logger.info(
            "t+/t- split: n_pos=%d n_neg=%d",
            t_split["t_pos"]["n"],
            t_split.get("t_neg", {}).get("n", 0),
        )

    # ── v0(C_neg): base-context activation under the negative panel (A3.7 R3-1) ─
    # The frac_ctx partial needs the negative persona's CONTEXT vector (no answer),
    # NOT t_neg (the answer activation) — the round-2 a37-frac-ctx-uses-tneg fix.
    v0_c_neg = extract_v0_C_neg(
        base,
        tok,
        behavior,
        registry,
        demos,
        probes,
        args.primary_layer,
        device,
        neg_r_lookup=neg_r_lookup,
        max_new_tokens=args.max_new_tokens,
    )
    if v0_c_neg is not None:
        logger.info(
            "v0(C_neg) base-context read: %d neg cids (%s), %d probe rows",
            v0_c_neg["n_neg_cids"],
            ",".join(v0_c_neg["neg_cids"]),
            v0_c_neg["n_probes"],
        )
    else:
        logger.warning("v0(C_neg) unavailable (no negative-panel cid resolved) — frac_ctx -> NaN")

    # ── fact r_B fresh (absent from #658 r_b.pt) ─────────────────────────────
    fact_rb = None
    if behavior == "fact":
        fact_rb = extract_fact_r_b(
            base, tok, probes[: args.max_probes or 6], args.primary_layer, device
        )

    extras = {
        "t_split": t_split,
        "v0_c_neg": v0_c_neg,
        "fact_rb": fact_rb,
        "c_c_all": c_c_all,
        "c_c_postft_all": c_c_postft_all,
        "gauge": gauge,
        "r_lookup": r_lookup,
    }
    # Route the per-target .npz writes to a local-SSD scratch mirror (#674) so
    # the per-(target, layer) write storm (~93 .npz/cell) stays off the GCE
    # network-PD plane it would otherwise saturate; batch-copy them to the
    # canonical cell_dir once below, BEFORE the complement check + sentinel.
    # Off GCE (EPS_SCRATCH_DIR unset) scratch_cell_dir IS cell_dir, so this is
    # a pass-through and the writes go straight to canonical as before.
    # issue=667 namespaces scratch by the PRODUCING extractor (this script),
    # NOT this fix's task #674, so concurrent extractors sharing one mount
    # never collide.
    scratch_cell_dir = scratch_path_for(cell_dir, issue=667)
    _prepare_scratch_cell_dir(scratch_cell_dir, cell_dir)
    n_gen = n_trunc = 0
    for tcid in targets:
        ng, nt = _extract_one_target(
            base,
            trained,
            tok,
            registry,
            demos,
            scratch_cell_dir,  # was cell_dir — writes land in scratch (#674)
            behavior,
            source_cid,
            seed,
            tcid,
            probes,
            layers,
            args.primary_layer,
            args.max_new_tokens,
            device,
            extras,
        )
        n_gen += ng
        n_trunc += nt
    logger.info(
        "cell %s/%s done: %d targets, %d generations (%d empty)",
        behavior,
        source_cid,
        len(targets),
        n_gen,
        n_trunc,
    )
    # Materialize scratch -> canonical (#674) BEFORE the complement check +
    # sentinel, which both read the CANONICAL cell_dir. On a partial-copy
    # failure the helper re-raises with scratch intact and no .done written, so
    # a resume re-materializes. Pass-through no-op when scratch_cell_dir IS
    # cell_dir (off GCE).
    materialize_to_canonical(scratch_cell_dir, cell_dir)
    # Validate the FULL (target, layer) .npz complement is on disk BEFORE the
    # sentinel — an empty-response target skips its .npz write per layer
    # (_extract_one_target's `if not acc[li][0]: continue`), so an unconditional
    # sentinel would stamp a TRUSTED .done over an incomplete cell that the
    # resume-skip then silently treats as done (round-8 BLOCKER
    # resume-skip-empty-acc-unconditional-sentinel). Raise loud instead — no
    # sentinel, so the cell is re-extracted on the next pass.
    assert_full_npz_complement(cell_dir, targets=targets, layers=layers)
    # Atomic completion sentinel — written ONLY after every target's tensors are
    # on disk, so the dispatcher's resume-skip never treats a partial dir as done
    # (round-8 BLOCKER resume-skip-partial-cell-silent-skip).
    write_cell_done_sentinel(
        cell_dir,
        behavior=behavior,
        source_cid=source_cid,
        seed=seed,
        targets=targets,
        layers=layers,
    )
    # #672 final memory-gauge flush (captures terminal footprint + persists).
    flush_mem_logger()
    # Free GPU (per-cell subprocess will exit, but be explicit).
    del base, trained
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


def _extract_one_target(
    base,
    trained,
    tok,
    registry,
    demos,
    cell_dir,
    behavior,
    source_cid,
    seed,
    tcid,
    probes,
    layers,
    primary_layer,
    max_new_tokens,
    device,
    extras,
) -> tuple[int, int]:
    """Extract + persist v0/v+ for ONE target context C' across all layers.

    Returns (n_generations, n_empty). Writes one .npz per layer; the primary
    layer's payload additionally carries t+/t-/r_b (the source-level reads).
    """
    c_c_all = extras["c_c_all"]
    c_c_postft_all = extras["c_c_postft_all"]
    gauge = extras["gauge"]
    t_split = extras["t_split"]
    v0_c_neg = extras.get("v0_c_neg")
    fact_rb = extras["fact_rb"]
    r_lookup = extras.get("r_lookup", {})
    tmsgs0 = build_messages_for(registry, demos, tcid, behavior, probes[0])
    # base + post-FT target query (c_C' / c_C'+) — both under the SAME prompt
    # (BLOCKER 1: oracle g+ needs the post-FT query at fixed M0).
    c_cp_all = _context_vector_all_layers(base, tok, tmsgs0, device)
    c_cp_postft_all = _context_vector_all_layers(trained, tok, tmsgs0, device)
    acc: dict[int, list[list[np.ndarray]]] = {li: [[], []] for li in layers}
    n_gen = n_trunc = 0
    for qi, q in enumerate(probes):
        tmsgs = build_messages_for(registry, demos, tcid, behavior, q)
        # Prefer the vLLM-pregenerated R (Phase A); HF fallback only on CPU-smoke.
        r = r_lookup.get((tcid, qi))
        if r is None:
            r = _greedy_response(base, tok, tmsgs, device, max_new_tokens)
        n_gen += 1
        if not r.strip():
            n_trunc += 1
            continue
        per_layer = _mean_resp_acts(base, trained, tok, tmsgs, r, layers, device)
        for li in layers:
            v0, vp = per_layer[li]
            acc[li][0].append(v0)
            acc[li][1].append(vp)
    for li in layers:
        if not acc[li][0]:
            logger.warning("no probes produced a response for target=%s layer=%d", tcid, li)
            continue
        c_idx = (li - 1) if 1 <= li <= N_LAYERS else (PRIMARY_LAYER - 1)
        payload = {
            "v0": np.stack(acc[li][0]).mean(axis=0).astype(np.float32),
            "v_plus": np.stack(acc[li][1]).mean(axis=0).astype(np.float32),
            "c_C": c_c_all[c_idx],
            "c_Cp": c_cp_all[c_idx],
            # post-FT key/query (BLOCKER 1: A3.10 oracle g+ = (k+, q+, M0)).
            "c_C_postft": c_c_postft_all[c_idx],
            "c_Cp_postft": c_cp_postft_all[c_idx],
            "c_C_all_layers": c_c_all,
            "c_Cp_all_layers": c_cp_all,
            "c_C_postft_all_layers": c_c_postft_all,
            "c_Cp_postft_all_layers": c_cp_postft_all,
            "behavior": behavior,
            "source_cid": source_cid,
            "target_cid": tcid,
            "seed": seed,
            "layer": li,
            "n_probes": len(acc[li][0]),
            "adapter_gauge": json.dumps(gauge),
        }
        if li == primary_layer:
            if "t_pos" in t_split:
                payload["t_pos"] = t_split["t_pos"]["vec"]
                payload["t_pos_n"] = t_split["t_pos"]["n"]
            if "t_neg" in t_split:
                payload["t_neg"] = t_split["t_neg"]["vec"]
                payload["t_neg_n"] = t_split["t_neg"]["n"]
            # v0(C_neg): negative-panel base-context vector for A3.7 frac_ctx (R3-1).
            # Distinct from t_neg (answer activation) — the round-2 BLOCKER fix.
            if v0_c_neg is not None:
                payload["v0_C_neg"] = v0_c_neg["vec"]
                payload["v0_C_neg_n_cids"] = v0_c_neg["n_neg_cids"]
            if fact_rb is not None:
                payload["r_b_fact"] = fact_rb
        np.savez(cell_dir / f"{tcid}_L{li}.npz", **payload)
    return n_gen, n_trunc


def _r_plus_npz_names(source_cid: str, seed: int, layers: list[int]) -> list[str]:
    """The per-layer r⁺ ``.npz`` filenames this cell writes (the skip/upload set)."""
    return [f"{source_cid}_seed{seed}_L{li}.npz" for li in layers]


def _r_plus_cell_already_extracted(
    out_dir: Path, behavior: str, source_cid: str, seed: int, layers: list[int]
) -> tuple[bool, bool]:
    """Idempotent-skip check: is this cell's FULL r⁺ layer set present (local | HF)?

    Round-2 (a36-readout-reextract-cos): the round-1 production launch uploaded
    some cells' r⁺ to HF (em/sp_swe) before crashing on the 4-GPU fan-out bug;
    re-extracting a complete cell on relaunch wastes a ~40-min serial adapter
    load+forward. A cell counts as complete ONLY when EVERY requested layer's
    ``.npz`` is present — a PARTIAL set (only some layers) does NOT prove the
    upstream prompt-set / hook layer matched, so a partial cell is re-extracted
    (CLAUDE.md "Fail fast — never silently accept a partial cell"). Returns
    ``(local_complete, hf_complete)``; the caller skips on either.
    """
    names = _r_plus_npz_names(source_cid, seed, layers)
    local_complete = all((out_dir / fn).is_file() for fn in names)
    hf_complete = False
    if not local_complete:
        # ONE get_paths_info call (single Hub HEAD batch, no recursive tree 504).
        # Missing paths are simply omitted from the result — never a 404 raise.
        try:
            from huggingface_hub import get_paths_info

            want = [f"{HF_R_PLUS_PREFIX}/{behavior}/{fn}" for fn in names]
            found = {info.path for info in get_paths_info(HF_DATA_REPO, want, repo_type="dataset")}
            hf_complete = all(p in found for p in want)
        except Exception as e:
            logger.warning("r⁺ HF skip-probe failed (%s) — local-only skip check", e)
    return local_complete, hf_complete


def _upload_r_plus_cell(out_dir: Path, behavior: str, names: list[str]) -> None:
    """Upload this cell's r⁺ ``.npz`` to HF immediately after writing (crash-safety).

    Per-cell upload (Upload Policy / #664): a mid-wave pod death must not strand
    an already-extracted cell — each cell's ≤3 tiny ``.npz`` go up the moment the
    cell completes, then the end-of-wave bulk verify is an idempotent safety pass.
    One ``create_commit`` per cell (well under the 256/hr cap) + a fresh-listing
    verify. ``EPM_SKIP_UPLOAD=1`` (local/CPU smoke) skips it. Fail-loud on a
    verified-missing file (never silently lose a cell).
    """
    if os.environ.get("EPM_SKIP_UPLOAD") == "1":
        logger.info("EPM_SKIP_UPLOAD=1 -> skipping per-cell r⁺ upload (smoke/local)")
        return
    from huggingface_hub import CommitOperationAdd, HfApi, get_paths_info

    present = [out_dir / fn for fn in names if (out_dir / fn).is_file()]
    if not present:
        raise RuntimeError(f"per-cell r⁺ upload: no .npz on disk under {out_dir} for {names}")
    api = HfApi()
    ops = [
        CommitOperationAdd(
            path_in_repo=f"{HF_R_PLUS_PREFIX}/{behavior}/{p.name}", path_or_fileobj=str(p)
        )
        for p in present
    ]
    src = present[0].stem.split("_seed")[0]
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue667 a36-reextract: per-cell r⁺ {behavior}/{src}",
    )
    want = [f"{HF_R_PLUS_PREFIX}/{behavior}/{p.name}" for p in present]
    found = {info.path for info in get_paths_info(HF_DATA_REPO, want, repo_type="dataset")}
    missing = [p for p in want if p not in found]
    if missing:
        raise RuntimeError(f"per-cell r⁺ upload verification FAILED -- missing on Hub: {missing}")
    logger.info("uploaded + verified %d per-cell r⁺ tensors (%s)", len(present), behavior)


def run_r_plus_extraction(args) -> int:
    """Re-extract r⁺ on θ⁺ for ONE source adapter, write per-layer .npz (followup).

    Loads the #537 adapter as a PeftModel θ⁺ (rsLoRA honored, gauge asserted),
    runs :func:`extract_r_plus` over the read layers (a source-level read-out —
    NO target sweep), and writes one ``<source>_seed<S>_L<layer>.npz`` per layer
    under ``--r-plus-out/<behavior>/`` with keys ``r_plus`` (the read-out),
    ``r_plus_norm`` (‖r⁺‖) and provenance fields. CPU-only smoke supported.

    Idempotent (round-2): if EVERY requested layer's ``.npz`` already exists
    LOCALLY or on HF, the cell is SKIPPED before any model load (the salvaged
    em/sp_swe cell on relaunch). A PARTIAL layer set re-extracts. After writing,
    the cell's ``.npz`` are uploaded to HF immediately (crash-safety) unless
    ``EPM_SKIP_UPLOAD=1``.
    """
    device = _device(args.gpu_id, args.cpu_only)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    layers = list(args.layers)
    behavior = args.behavior
    source_cid = args.source_cid
    seed = args.seed
    cap = args.max_probes
    logger.info(
        "r⁺ extract cell behavior=%s source=%s seed=%d layers=%s cap=%s",
        behavior,
        source_cid,
        seed,
        layers,
        cap,
    )
    out_root = Path(args.r_plus_out)
    out_dir = out_root / behavior
    # Idempotent skip: a fully-extracted cell (local OR HF) is not re-run.
    local_done, hf_done = _r_plus_cell_already_extracted(
        out_dir, behavior, source_cid, seed, layers
    )
    if local_done or hf_done:
        logger.info(
            "skipped: cell %s/%s already extracted (local=%s | HF=%s) — no re-extraction",
            behavior,
            source_cid,
            local_done,
            hf_done,
        )
        return 0
    # Stage + verify the adapter gauge BEFORE any GPU work (cheap, HALT early).
    adapter_dir = stage_adapter_local(behavior, source_cid, seed)
    gauge = assert_adapter_gauge(adapter_dir, behavior)
    logger.info("adapter gauge OK: %s", {k: gauge[k] for k in ("r", "lora_alpha", "use_rslora")})

    # Only θ⁺ is needed for r⁺ (the read-out re-extraction runs on the trained
    # model; the cosine to base r_B is computed in analysis). load_trained_only
    # avoids the second base copy load_base_and_trained allocates (CPU-smoke OOM).
    tok, trained = load_trained_only(adapter_dir, device, dtype)
    r_plus = extract_r_plus(trained, tok, behavior, layers, device, max_probes=cap)

    # out_dir was resolved above for the idempotent-skip check; reuse it.
    out_dir.mkdir(parents=True, exist_ok=True)
    for li, vec in r_plus.items():
        payload = {
            "r_plus": vec.astype(np.float32),
            "r_plus_norm": np.float32(np.linalg.norm(vec)),
            "behavior": behavior,
            "source_cid": source_cid,
            "seed": seed,
            "layer": li,
            "rb_cap": RB_CAP if cap is None else min(RB_CAP, cap),
            "adapter_gauge": json.dumps(gauge),
        }
        np.savez(out_dir / f"{source_cid}_seed{seed}_L{li}.npz", **payload)
        logger.info(
            "wrote r⁺ %s/%s L%d (‖r⁺‖=%.4f)",
            behavior,
            source_cid,
            li,
            float(payload["r_plus_norm"]),
        )
    del trained
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Per-cell HF upload immediately after writing (crash-safety / Upload Policy
    # #664): a mid-wave pod death must not strand this already-extracted cell.
    # EPM_SKIP_UPLOAD=1 (CPU/local smoke) skips it; the dispatcher's end-of-wave
    # bulk verify is then an idempotent safety pass.
    _upload_r_plus_cell(out_dir, behavior, _r_plus_npz_names(source_cid, seed, layers))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #667 absolute-v extractor (one source-adapter cell)."
    )
    parser.add_argument("--behavior", required=True, choices=["em", "sycophancy", "fact", "marker"])
    parser.add_argument("--source-cid", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--targets", default=None, help="comma-separated target cids (default: 30 eval + source)"
    )
    parser.add_argument("--layers", type=int, nargs="+", default=list(ALL_LAYERS))
    parser.add_argument("--primary-layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument("--out", default="eval_results/issue_667/analysis_tensors")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--max-probes", type=int, default=0, help="cap probes (0 = full pool; smoke)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=N_GEN_TOKENS)
    parser.add_argument(
        "--max-train-rows", type=int, default=None, help="cap t+/t- training rows (smoke)"
    )
    parser.add_argument(
        "--log-mem-every",
        type=int,
        default=0,
        help="#672 ANALYSIS-ONLY: log 3 GPU-memory gauges (memory_reserved + "
        "nvidia-smi memory.used PRIMARY, memory_allocated SECONDARY) every N "
        "hooked forwards to <out>/memory_log.json. 0 = off (default).",
    )
    parser.add_argument(
        "--r-plus",
        action="store_true",
        help=(
            "Re-extract the behavior read-out r⁺ ON θ⁺ (followup "
            "a36-readout-reextract-cos): writes one <source>_seed<S>_L<l>.npz per "
            "layer to --r-plus-out/<behavior>/ via extract_r_plus. Source-level "
            "read-out — NO target sweep, NO v0/v+ store written."
        ),
    )
    parser.add_argument(
        "--r-plus-out",
        default="eval_results/issue_667/a36_readout_reextract/r_plus",
        help="r⁺ store output dir (--r-plus mode only).",
    )
    args = parser.parse_args()
    if args.max_probes == 0:
        args.max_probes = None
    t0 = time.time()
    rc = run_r_plus_extraction(args) if args.r_plus else run_extraction(args)
    logger.info("extraction wall=%.1fs", time.time() - t0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
