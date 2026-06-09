# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + minus sign − intentional
#!/usr/bin/env python3
"""Task #504 recovery diagnostic — re-eval the c504_smoke_r8 LoRA via two paths.

NOT training. NOT a sweep. Single cell. Single seed. ~15 held-out probes × 4 Q.

Hypothesis under test
---------------------
The #504 round-12 trajectory eval reported source-self ΔG ≈ 0 + emission ≈ 0
for the r=8 cell at every checkpoint, AND the round-11 ``assert_adapter_actually_
applied`` guard fired ``LoRANotAppliedError`` on the trained pass (max|ΔG| < 0.5
nats AND emission=0 AND B-matrix norm above the 1e-3 floor). The r=4 cell's
source-self ΔG sat at ±0.02 nats at every checkpoint too; only the panel-max
|ΔG| nudged past the 0.5-nat eps because some held-out persona's ΔG drifted
slightly. This is the EXACT signature of #477 v4/v6: vLLM/LoRARequest path
silently fails to apply the adapter, so ``score_logp_for_R(use_lora=True)``
returns BASE log-probs at every probe.

This diagnostic re-scores ΔG with the SAME held-out panel + SAME questions +
SAME marker via TWO independent paths:

  Path A — clean PEFT (HF Transformers + ``PeftModel.from_pretrained``). Ground
           truth. If the adapter has any effect at all, this path sees it.
  Path B — vLLM LoRARequest (``lora_path=<local adapter dir>``). The exact
           mechanism the production rig's ``score_logp_for_R(use_lora=True)``
           uses on the trained pass.

Dispositive outcomes (the verdict the user directive needs):
  * Path A ΔG > 1.0 nat AND Path B ΔG < 0.5 nat
        → vLLM-LoRARequest-bug confirmed; PEFT direct sees the adapter, vLLM
          does not. Investigate version drift / LoRARequest threading.
  * Path A ΔG > 1.0 nat AND Path B ΔG > 1.0 nat
        → env is fine; full re-eval will recover. Run i504_reval_grid.py.
  * Path A ΔG < 0.5 nat AND Path B ΔG < 0.5 nat
        → adapter genuinely under-trained; escalate per user directive
          (this is the path where 'no retrain' clashes with no usable signal).
  * Otherwise
        → ambiguous; rerun with --n-heldout >= 25 for tighter CI.

Framework-switch discipline (CLAUDE.md vLLM-teardown gotcha): Path A first
(HF + PEFT), then explicit teardown of HF tensors + gc + empty_cache, then
Path B (vLLM with enable_lora). vLLM-side teardown at the very end of
Path B; psutil child-kill + nvidia-smi PID check via the rig's helper.

Adapter resolution order (no silent fallback):
  1. ``--local-adapter <dir>`` if passed AND the dir contains adapter_config.json.
  2. ``/workspace/runs/issue_504/c504_smoke_r8_seed42/adapter`` if it exists.
  3. HF Hub pull from ``superkaiba1/explore-persona-space`` under
     ``adapters/issue_504/c504_smoke_r8_seed42``.
Raise on (3) miss; never proceed without an adapter dir.

The script asserts NOTHING (no expected-value checks); it reports. Designed to
run < 5 min on a 1× H100. Reuses ``score_logp_for_R`` + ``build_full_ids``
from ``eval_one_cell`` so Path B is byte-identical to the production rig's
measurement (only Path A re-implements the slot read because the production
KL path uses a full-vocab forward, not a marker-only logp).
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

log = logging.getLogger("i504.reval_confirm")

# ── Constants pinned to the rig + the broken eval cell. ──────────────────────
ADAPTER_HF_REPO = "superkaiba1/explore-persona-space"
ADAPTER_SUBFOLDER = "adapters/issue_504/c504_smoke_r8_seed42"
CELL_SLUG = "c504_smoke_r8"
SEED = 42

LOCAL_ADAPTER_DEFAULT = Path("/workspace/runs/issue_504/c504_smoke_r8_seed42/adapter")

# Panel + bank data — locally-resolved paths (the pod mirrors these locations).
PANEL_JSON_DEFAULT = Path("eval_results/issue_504/phase0_5_gates_round6.json")
BANK_PATH_DEFAULT = Path("data/issue_472/persona_bank.json")
# R_eval preference: v504 fill first (round 11), then the #472 source. Both are
# preferred-over-fallback; the rig will use whichever exists. We only use this
# file for question-list source and panel sanity — Path A + Path B each generate
# their OWN R on-policy (PEFT-direct gen vs vLLM gen), the canonical
# diagnostic test for the v4/v6 silent-LoRA-not-applied class.
R_EVAL_PREFERENCES: tuple[Path, ...] = (
    Path("data/issue_472/on_policy_R/R_eval_v504.json"),
    Path("data/issue_472/on_policy_R/R_eval.json"),
)

# Slice knobs — tiny diagnostic, not a sweep.
DEFAULT_N_HELDOUT = 15
DEFAULT_N_QUESTIONS = 4
DEFAULT_MAX_NEW_TOKENS = 256  # diagnostic — short greedy answers


def _git_sha() -> str:
    """Best-effort git HEAD sha for reproducibility metadata; 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            # epm-lint: subprocess-env-inherit -- git rev-parse needs no credentials
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _env_versions() -> dict[str, str]:
    """Pinned env versions for the reproducibility block (touch only on import)."""
    versions: dict[str, str] = {}
    try:
        import vllm

        versions["vllm"] = getattr(vllm, "__version__", "unknown")
    except ImportError:
        versions["vllm"] = "not_installed"
    try:
        import peft

        versions["peft"] = getattr(peft, "__version__", "unknown")
    except ImportError:
        versions["peft"] = "not_installed"
    try:
        import transformers

        versions["transformers"] = getattr(transformers, "__version__", "unknown")
    except ImportError:
        versions["transformers"] = "not_installed"
    try:
        import torch

        versions["torch"] = getattr(torch, "__version__", "unknown")
    except ImportError:
        versions["torch"] = "not_installed"
    return versions


def _resolve_adapter_dir(local_adapter: Path | None, token: str | None) -> Path:
    """Adapter resolution: explicit override > local default > HF Hub pull.

    Raises FileNotFoundError if NONE of the three sources yield a valid adapter
    directory (one with ``adapter_config.json`` + ``adapter_model.safetensors``).
    """

    def _is_valid(d: Path) -> bool:
        return (
            d.exists()
            and (d / "adapter_config.json").exists()
            and (d / "adapter_model.safetensors").exists()
        )

    if local_adapter is not None:
        if not _is_valid(local_adapter):
            raise FileNotFoundError(
                f"--local-adapter {local_adapter} does not contain a valid PEFT adapter "
                f"(needs adapter_config.json + adapter_model.safetensors)."
            )
        log.info("[adapter] using --local-adapter override: %s", local_adapter)
        return local_adapter

    if _is_valid(LOCAL_ADAPTER_DEFAULT):
        log.info("[adapter] using local default: %s", LOCAL_ADAPTER_DEFAULT)
        return LOCAL_ADAPTER_DEFAULT

    log.info(
        "[adapter] local paths empty; pulling from HF: %s/%s",
        ADAPTER_HF_REPO,
        ADAPTER_SUBFOLDER,
    )
    return _fetch_adapter_from_hf(token)


def _fetch_adapter_from_hf(token: str | None) -> Path:
    """Pull the trained adapter dir from HF Hub via per-file ``hf_hub_download``.

    ``snapshot_download(allow_patterns=...)`` returns 0 files on this repo's
    truncated siblings (siblings-truncation bug — task #480 + the #477
    ``_fetch_adapter`` pattern). We list the repo files and pull each one
    explicitly.
    """
    from huggingface_hub import HfApi, hf_hub_download

    cache_root = Path("/tmp/i504_reval/adapter_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    all_files = api.list_repo_files(repo_id=ADAPTER_HF_REPO, repo_type="model", token=token)
    sub_files = [f for f in all_files if f.startswith(f"{ADAPTER_SUBFOLDER}/")]
    if not sub_files:
        raise FileNotFoundError(
            f"no files under {ADAPTER_SUBFOLDER} in {ADAPTER_HF_REPO} — "
            "adapter subfolder missing on Hub; nothing to pull."
        )
    for fn in sub_files:
        hf_hub_download(
            repo_id=ADAPTER_HF_REPO,
            repo_type="model",
            filename=fn,
            local_dir=str(cache_root),
            token=token,
        )
    adapter_dir = cache_root / ADAPTER_SUBFOLDER
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / required).exists():
            raise FileNotFoundError(
                f"adapter file {required} missing under {adapter_dir} after per-file download"
            )
    log.info("[adapter] HF pull OK → %s (%d files)", adapter_dir, len(sub_files))
    return adapter_dir


def _resolve_r_eval_path() -> Path:
    """Return the first R_eval path that exists from the preference list. Raise if none."""
    for p in R_EVAL_PREFERENCES:
        if p.exists():
            log.info("[r_eval] using %s", p)
            return p
    raise FileNotFoundError(
        f"no R_eval file found at any of {[str(p) for p in R_EVAL_PREFERENCES]}; "
        "run i504_phase_r_generate_fill on the pod first."
    )


def _select_eval_slice(
    panel_json_path: Path,
    bank_path: Path,
    n_heldout: int,
    n_questions: int,
) -> tuple[dict[str, str], list[str], str, str]:
    """Build the held-out panel slice + Q_eval slice + source persona prompt.

    Reads:
      - panel from ``panel_json_path['held_out_panel']`` (Phase 0.5 output).
      - persona system prompts from the #472 bank.
      - Q_eval from the same r_generate split the rig uses.

    Returns (eval_personas_dict, q_eval_slice, source_name, source_prompt).
    Raises on missing panel or missing personas in bank (fail-loud, no silent
    fill).
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
    )

    panel_payload = json.loads(panel_json_path.read_text())
    held_out_panel = panel_payload.get("held_out_panel", [])
    if not held_out_panel:
        raise RuntimeError(
            f"--panel-json {panel_json_path} has empty 'held_out_panel'; Phase 0.5 must "
            "populate it before this diagnostic runs."
        )
    bank = load_persona_bank(bank_path)
    # Fail-loud on any panel persona missing from the bank (no silent skip).
    for p in held_out_panel:
        if p not in bank:
            raise KeyError(
                f"Panel persona {p!r} missing from bank at {bank_path}; "
                "Phase 0.5 + Phase 1 must read the SAME bank artifact."
            )
    # Take the first N panel personas (panel is already sorted alphabetically by
    # Phase 0.5).
    panel_slice = held_out_panel[:n_heldout]
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


def _build_marker_slot_logp(
    model,
    tokenizer,
    r_by_persona_q: dict[str, dict[str, str]],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    *,
    device: str,
) -> dict[str, dict[str, dict[str, float | bool | int]]]:
    """Score log P(※) at the post-R marker slot via HF forward (Path A).

    For each (persona, q): build the byte-identical token sequence
    ``prompt + R + MARKER_SEP + MARKER_TEXT`` via ``build_full_ids`` (the rig's
    slot-construction helper) and read
    ``log_softmax(logits[slot - 1])[marker_id]`` — HF's logits at position t
    predict the token at position t+1, so the marker prediction lives at
    ``slot - 1``. Mirrors the production rig's slot-construction so PEFT and
    vLLM paths agree on the measurement modulo the load mechanism.

    Returns ``out[persona][q] = {logp, argmax_marker, n_marker_in_R}``.
    """
    import torch

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
        build_full_ids,
    )

    out: dict[str, dict[str, dict[str, float | bool | int]]] = {p: {} for p in eval_personas}
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
            assert logits.shape[0] == 1 and logits.shape[1] == len(full_ids), logits.shape
            # logits[0, slot - 1, :] predicts the token at position `slot`.
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
    g_records: dict[str, dict[str, dict[str, float | bool | int]]],
    b_records: dict[str, dict[str, dict[str, float | bool | int]]],
    source: str,
) -> dict[str, float | int]:
    """Mean logp / emission / ΔG over held-out (panel − source) and source-self."""
    held_g_lps: list[float] = []
    held_b_lps: list[float] = []
    held_emits: list[bool] = []
    src_g_lps: list[float] = []
    src_b_lps: list[float] = []
    src_emits: list[bool] = []
    for persona, per_q_g in g_records.items():
        for q, gleaf in per_q_g.items():
            gl = float(gleaf["logp"])
            bl = float(b_records[persona][q]["logp"])
            em = bool(gleaf["argmax_marker"])
            if persona == source:
                src_g_lps.append(gl)
                src_b_lps.append(bl)
                src_emits.append(em)
            else:
                held_g_lps.append(gl)
                held_b_lps.append(bl)
                held_emits.append(em)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def _rate(xs: list[bool]) -> float:
        return sum(1 for x in xs if x) / len(xs) if xs else float("nan")

    return {
        "held_g_logp_mean": _mean(held_g_lps),
        "held_b_logp_mean": _mean(held_b_lps),
        "held_delta_g_mean": _mean(held_g_lps) - _mean(held_b_lps),
        "held_emit_rate": _rate(held_emits),
        "source_g_logp_mean": _mean(src_g_lps),
        "source_b_logp_mean": _mean(src_b_lps),
        "source_delta_g_mean": _mean(src_g_lps) - _mean(src_b_lps),
        "source_emit_rate": _rate(src_emits),
        "n_held_probes": len(held_g_lps),
        "n_source_probes": len(src_g_lps),
    }


def _verdict(
    *,
    path_a_source_delta_g: float,
    path_b_source_delta_g: float,
) -> tuple[str, str]:
    """Map the (Path A, Path B) source-self ΔG pair to a one-of-4 verdict.

    The 4-branch contract from the user directive (no silent fallback):
      * a > 1.0 nat AND b < 0.5 nat   → vLLM bug confirmed
      * a > 1.0 nat AND b > 1.0 nat   → env fine; full re-eval will recover
      * a < 0.5 nat AND b < 0.5 nat   → adapter genuinely under-trained
      * otherwise                     → ambiguous; tighten the slice

    Returns (verdict_tag, human_readable_diagnostic).
    """
    a = float(path_a_source_delta_g)
    b = float(path_b_source_delta_g)
    if a > 1.0 and b < 0.5:
        return (
            "vllm_lora_request_bug_confirmed",
            "PEFT direct sees the adapter (source ΔG > 1.0 nat), vLLM LoRARequest "
            "does not (source ΔG < 0.5 nat). The bug is at the vLLM-LoRARequest forward "
            "path — investigate version drift, LoRARequest threading, or adapter "
            "rank vs max_lora_rank mismatch before re-running the production rig.",
        )
    if a > 1.0 and b > 1.0:
        return (
            "env_fine_re_eval_will_recover",
            "Both PEFT direct and vLLM LoRARequest read source ΔG > 1.0 nat. The "
            "round-12 trajectory eval's ΔG ≈ 0 was a transient / pod-env-specific "
            "failure; run i504_reval_grid.py on the current env to recover the "
            "leakage-vs-rank grid.",
        )
    if a < 0.5 and b < 0.5:
        return (
            "adapter_genuinely_under_trained",
            "BOTH paths read source ΔG < 0.5 nat. The adapter B-matrix norm is "
            "above the 1e-3 floor (genuinely trained) but neither evaluator extracts "
            "the marker signal. This is the path where 'no retrain' clashes with no "
            "usable signal — ESCALATE to the user per the directive's escape clause.",
        )
    return (
        "ambiguous_partial_signal",
        f"Path A source ΔG = {a:+.3f} nats, Path B source ΔG = {b:+.3f} nats — "
        "neither verdict pattern matches. Re-run with --n-heldout >= 25 to tighten "
        "the source-self CI before deciding.",
    )


def _teardown_vllm(llm) -> None:
    """Tear vLLM down + reap workers (CLAUDE.md vLLM-teardown gotcha).

    Single-process + single-GPU diagnostic, so the workers can only belong to
    us (CVD-naive teardown is safe here).
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
        # Snapshot children ONCE — terminate() is async, re-querying may miss originals.
        children = me.children(recursive=True)
        for c in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                c.terminate()
        _gone, alive = psutil.wait_procs(children, timeout=10)
        for c in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                c.kill()
    except ImportError:
        log.warning("psutil unavailable; cannot reap vLLM worker subprocesses.")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _b_matrix_max_norm(adapter_dir: Path) -> float:
    """Wrap eval_guard.b_matrix_frobenius_norm for the report header."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
        b_matrix_frobenius_norm,
    )

    return b_matrix_frobenius_norm(adapter_dir)


def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser. Extracted so main() stays under McCabe 15."""
    ap = argparse.ArgumentParser(
        description="Task #504 recovery diagnostic: PEFT vs vLLM-LoRARequest re-eval (r=8 cell).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-heldout", type=int, default=DEFAULT_N_HELDOUT)
    ap.add_argument("--n-questions", type=int, default=DEFAULT_N_QUESTIONS)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs (the diagnostic is single-GPU; this is for symmetry with "
        "the grid driver and is currently informational only).",
    )
    ap.add_argument("--device", default="cuda:0", help="HF device for Path A.")
    ap.add_argument(
        "--local-adapter",
        type=Path,
        default=None,
        help="Override the adapter directory; bypasses local-default + HF-pull fallback. "
        "Use when iterating on a re-downloaded adapter, or to point at a specific "
        "checkpoint dir.",
    )
    ap.add_argument(
        "--panel-json",
        type=Path,
        default=PANEL_JSON_DEFAULT,
        help="Phase 0.5 output JSON with the 'held_out_panel' list.",
    )
    ap.add_argument(
        "--bank-path",
        type=Path,
        default=BANK_PATH_DEFAULT,
        help="#472 persona bank JSON.",
    )
    ap.add_argument(
        "--out-path",
        type=Path,
        default=Path("eval_results/issue_504/reval_confirm/c504_smoke_r8_seed42.json"),
    )
    ap.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.40,
        help="vLLM gpu_memory_utilization for Path B (conservative — Path A may "
        "leave residual; 0.40 = ~32 GiB of an 80 GiB H100 → plenty for "
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
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve all paths + arguments and exit 0 without touching GPU or "
        "spinning up vLLM/HF. Used for the dev-VM smoke check.",
    )
    return ap


def _resolve_paths_dry_run(args: argparse.Namespace, token: str | None) -> tuple[Path, Path]:
    """Resolve adapter_dir + r_eval_path for a dry-run (no HF pulls; warn on missing)."""
    if args.local_adapter is not None:
        adapter_dir = args.local_adapter
    elif LOCAL_ADAPTER_DEFAULT.exists():
        adapter_dir = LOCAL_ADAPTER_DEFAULT
    else:
        adapter_dir = Path("/tmp/i504_reval/adapter_cache") / ADAPTER_SUBFOLDER
    log.info(
        "[dry-run] would resolve adapter to %s (not validated; HF pull skipped)",
        adapter_dir,
    )
    try:
        r_eval_path = _resolve_r_eval_path()
    except FileNotFoundError as e:
        log.warning("[dry-run] R_eval not found locally — pod will have it: %s", e)
        r_eval_path = R_EVAL_PREFERENCES[0]
    if not args.panel_json.exists():
        log.warning("[dry-run] panel-json missing locally at %s", args.panel_json)
    if not args.bank_path.exists():
        log.warning("[dry-run] bank-path missing locally at %s", args.bank_path)
    # Touch token to silence unused-var without a value-side effect.
    _ = token
    return adapter_dir, r_eval_path


def _run_path_a_peft(
    *,
    adapter_dir: Path,
    tokenizer,
    panel_plus_source: dict[str, str],
    q_eval: list[str],
    source_name: str,
    args: argparse.Namespace,
    token: str | None,
    partial: dict,
) -> tuple[dict, dict, dict]:
    """Run Path A (clean PEFT). Persist Path A into ``partial`` and return
    (R_peft, peft_records, base_records_peft_R) for the verdict path."""
    log.info("[phase=peft] loading trained model via PEFT (Path A — ground truth)")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import BASE_MODEL

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
        token=token,
    ).eval()
    trained_peft = PeftModel.from_pretrained(base_model, str(adapter_dir)).eval()

    log.info("[phase=peft] generating R_PEFT (trained, greedy)")
    r_peft = _hf_generate_R(
        trained_peft,
        tokenizer,
        panel_plus_source,
        q_eval,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    log.info("[phase=peft] scoring g_logp on R_PEFT (trained)")
    peft_records = _build_marker_slot_logp(
        trained_peft, tokenizer, r_peft, panel_plus_source, q_eval, device=args.device
    )
    # Drop PEFT-wrapped model, reload a CLEAN base for the b_logp pass.
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
    base_records_peft_R = _build_marker_slot_logp(
        base_clean, tokenizer, r_peft, panel_plus_source, q_eval, device=args.device
    )
    del base_clean
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Persist Phase A IMMEDIATELY (per-phase checkpoint rule).
    partial["paths"]["peft"] = {
        "R": r_peft,
        "g_records": peft_records,
        "b_records": base_records_peft_R,
    }
    partial["summary"]["peft"] = _summarize(peft_records, base_records_peft_R, source_name)
    args.out_path.write_text(json.dumps(partial, indent=2))
    log.info("[phase=peft] persisted Phase A → %s", args.out_path)
    return r_peft, peft_records, base_records_peft_R


def _run_path_b_vllm(
    *,
    adapter_dir: Path,
    tokenizer,
    panel_plus_source: dict[str, str],
    q_eval: list[str],
    source_name: str,
    args: argparse.Namespace,
    partial: dict,
) -> tuple[dict, dict, dict]:
    """Run Path B (vLLM + LoRARequest). Persist Path B into ``partial`` and return
    (R_vllm, vllm_records, base_records_vllm_R) for the verdict path."""
    log.info("[phase=vllm] loading vLLM with enable_lora + LoRARequest (Path B — suspect)")
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import BASE_MODEL
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
        score_logp_for_R,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        _generate_on_policy_R,
    )

    # vLLM max_lora_rank is a buffer size (must be one of (8, 16, 32, 64, 128,
    # 256, 320, 512)). r=8 fits in r=8 buffer exactly; this mirrors the
    # production rig's floor logic in i504_eval_trajectory.py.
    vllm_max_lora_rank = 8

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        seed=SEED,
        max_model_len=2048,
        enable_lora=True,
        max_lora_rank=vllm_max_lora_rank,
        max_loras=1,
    )
    lora_req = LoRARequest(
        lora_name=f"{CELL_SLUG}_seed{SEED}_reval",
        lora_int_id=1,
        lora_path=str(adapter_dir),
    )

    log.info("[phase=vllm] generating R_VLLM (trained, greedy via vLLM)")
    r_vllm = _generate_on_policy_R(
        llm, tokenizer, panel_plus_source, q_eval, lora_req, args.max_new_tokens
    )

    log.info("[phase=vllm] scoring g_logp on R_VLLM (use_lora=True)")
    vllm_records = score_logp_for_R(
        llm,
        tokenizer,
        r_by_persona_q=r_vllm,
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
        r_by_persona_q=r_vllm,
        eval_personas=panel_plus_source,
        eval_questions=q_eval,
        cell_label=f"BASE/{CELL_SLUG}_seed{SEED}_reval",
        use_lora=False,
    )

    _teardown_vllm(llm)

    partial["paths"]["vllm"] = {
        "R": r_vllm,
        "g_records": vllm_records,
        "b_records": base_records_vllm_R,
    }
    partial["summary"]["vllm"] = _summarize(vllm_records, base_records_vllm_R, source_name)
    args.out_path.write_text(json.dumps(partial, indent=2))
    log.info("[phase=vllm] persisted Phase B → %s", args.out_path)
    return r_vllm, vllm_records, base_records_vllm_R


def _print_report(
    partial: dict,
    eval_personas: dict[str, str],
    q_eval: list[str],
    source_name: str,
    verdict_tag: str,
    diagnostic: str,
    adapter_dir: Path,
    out_path: Path,
) -> None:
    """Print the human-readable cross-path report (Phase 3)."""
    print("\n" + "=" * 80)
    print(f"#504 RECOVERY DIAGNOSTIC — {CELL_SLUG} seed={SEED}")
    print(f"adapter: {adapter_dir}  (B-max-norm = {partial['adapter_b_max_norm']:.4f})")
    print(
        f"slice: {len(eval_personas)} held-out personas × {len(q_eval)} questions  "
        f"|  source: {source_name}"
    )
    print("=" * 80)

    def _fmt_row(label: str, b: float, g: float) -> str:
        return f"  {label:<24} b_logp={b:8.3f}  g_logp={g:8.3f}  ΔG={g - b:+8.3f}"

    if "peft" in partial["summary"]:
        ps = partial["summary"]["peft"]
        print("\n[Path A — clean PEFT]  (ground truth)")
        print(_fmt_row("held-out mean:", ps["held_b_logp_mean"], ps["held_g_logp_mean"]))
        print(_fmt_row("source-self mean:", ps["source_b_logp_mean"], ps["source_g_logp_mean"]))
        print(
            f"  source emit P(※)  trained={ps['source_emit_rate']:.2f}  "
            f"held trained={ps['held_emit_rate']:.2f}"
        )

    if "vllm" in partial["summary"]:
        vs = partial["summary"]["vllm"]
        print("\n[Path B — vLLM LoRARequest]  (suspect — the rig's mechanism)")
        print(_fmt_row("held-out mean:", vs["held_b_logp_mean"], vs["held_g_logp_mean"]))
        print(_fmt_row("source-self mean:", vs["source_b_logp_mean"], vs["source_g_logp_mean"]))
        print(
            f"  source emit P(※)  trained={vs['source_emit_rate']:.2f}  "
            f"held trained={vs['held_emit_rate']:.2f}"
        )

    if verdict_tag != "incomplete":
        print("\n[Cross-path verdict]")
        print(f"  PEFT  ΔG_source = {partial['source_self_delta_g_a']:+.3f} nats")
        print(f"  vLLM  ΔG_source = {partial['source_self_delta_g_b']:+.3f} nats")
        print(f"  verdict: {verdict_tag}")
        print(f"  → {diagnostic}")

    print("\nfull per-probe records → " + str(out_path))
    print("=" * 80 + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=reval_confirm] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    token = os.environ.get("HF_TOKEN")
    if token is None and not args.dry_run:
        raise RuntimeError(
            "HF_TOKEN missing — load_dotenv() ran but .env lacks the token; the "
            "adapter HF-pull fallback + bank load need it. Fix .env on the pod."
        )

    # ── Phase 0: resolve adapter + R_eval. ───────────────────────────────────
    log.info("[phase=resolve] adapter, panel, R_eval, marker_token_id")
    if args.dry_run:
        adapter_dir, r_eval_path = _resolve_paths_dry_run(args, token)
    else:
        adapter_dir = _resolve_adapter_dir(args.local_adapter, token)
        r_eval_path = _resolve_r_eval_path()
        if not args.panel_json.exists():
            raise FileNotFoundError(f"--panel-json {args.panel_json} does not exist.")
        if not args.bank_path.exists():
            raise FileNotFoundError(f"--bank-path {args.bank_path} does not exist.")

    # ── Dry-run exit (BEFORE any HF / vLLM / tokenizer / model call). ───────
    # The dry-run path's job is to verify args parse, local paths resolve, and
    # local-file slice loads work. It MUST NOT touch the network or the HF
    # cache — Codex round-13 blocker 1: a fresh env with HF_HUB_OFFLINE=1 +
    # empty HF_HOME must exit 0. The marker-id sanity check uses the hardcoded
    # EXPECTED_MARKER_TOKEN_ID (pinned + asserted in tests + at non-dry-run
    # eval time) — no tokenizer load required for the dry-run contract.
    if args.dry_run:
        from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
            EXPECTED_MARKER_TOKEN_ID,
            MARKER_TEXT,
        )

        # If the panel JSON is local, also exercise the slice-selection logic
        # (catches a Phase 0.5 schema regression without needing a tokenizer).
        if args.panel_json.exists() and args.bank_path.exists():
            eval_personas, q_eval, source_name, _source_prompt = _select_eval_slice(
                args.panel_json, args.bank_path, args.n_heldout, args.n_questions
            )
            n_personas_incl_source = len(eval_personas) + 1
            n_questions = len(q_eval)
            log.info(
                "[dry-run] PASS — adapter=%s panel=%d probes (incl source) Q=%d",
                adapter_dir,
                n_personas_incl_source,
                n_questions,
            )
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "verdict": "DRY_RUN_PASS",
                        "adapter_dir_resolved_to": str(adapter_dir),
                        "panel_json": str(args.panel_json),
                        "r_eval_path": str(r_eval_path),
                        "n_personas_incl_source": n_personas_incl_source,
                        "n_questions": n_questions,
                        "source": source_name,
                        "marker_text": MARKER_TEXT,
                        "marker_token_id_expected": EXPECTED_MARKER_TOKEN_ID,
                        "env": _env_versions(),
                    },
                    indent=2,
                )
            )
            return 0

        # Panel/bank missing locally — still a valid dry-run, just narrower.
        log.info("[dry-run] PASS — args+imports OK; skipping slice load (no panel/bank)")
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "verdict": "DRY_RUN_PASS",
                    "adapter_dir_resolved_to": str(adapter_dir),
                    "panel_json": str(args.panel_json),
                    "r_eval_path": str(r_eval_path),
                    "n_heldout": args.n_heldout,
                    "n_questions": args.n_questions,
                    "max_new_tokens": args.max_new_tokens,
                    "marker_text": MARKER_TEXT,
                    "marker_token_id_expected": EXPECTED_MARKER_TOKEN_ID,
                    "env": _env_versions(),
                },
                indent=2,
            )
        )
        return 0

    # ── Non-dry-run path (pod-only from here on). ───────────────────────────
    # Marker token assertion + eval slice — needs tokenizer (HF cache hit on pod).
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
        args.panel_json, args.bank_path, args.n_heldout, args.n_questions
    )
    panel_plus_source = dict(eval_personas)
    panel_plus_source.setdefault(source_name, source_prompt)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    partial: dict = {
        "schema_version": "i504_reval_confirm_v1",
        "cell": CELL_SLUG,
        "seed": SEED,
        "adapter_dir": str(adapter_dir),
        "adapter_hf_repo": ADAPTER_HF_REPO,
        "adapter_subfolder": ADAPTER_SUBFOLDER,
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
        "env": _env_versions(),
        "adapter_b_max_norm": float(_b_matrix_max_norm(adapter_dir)),
        "paths": {},
        "summary": {},
    }

    # ── Phase A: clean PEFT (ground truth). ──────────────────────────────────
    peft_records: dict | None = None
    if not args.skip_peft:
        _r_peft, peft_records, _b_records_peft_R = _run_path_a_peft(
            adapter_dir=adapter_dir,
            tokenizer=tokenizer,
            panel_plus_source=panel_plus_source,
            q_eval=q_eval,
            source_name=source_name,
            args=args,
            token=token,
            partial=partial,
        )

    # ── Phase B: vLLM LoRARequest (suspect — the rig's mechanism). ──────────
    vllm_records: dict | None = None
    if not args.skip_vllm:
        _r_vllm, vllm_records, _b_records_vllm_R = _run_path_b_vllm(
            adapter_dir=adapter_dir,
            tokenizer=tokenizer,
            panel_plus_source=panel_plus_source,
            q_eval=q_eval,
            source_name=source_name,
            args=args,
            partial=partial,
        )

    # ── Phase 2: cross-path verdict. ─────────────────────────────────────────
    verdict_tag = "incomplete"
    diagnostic = ""
    if peft_records is not None and vllm_records is not None:
        path_a_src_dg = float(partial["summary"]["peft"]["source_delta_g_mean"])
        path_b_src_dg = float(partial["summary"]["vllm"]["source_delta_g_mean"])
        verdict_tag, diagnostic = _verdict(
            path_a_source_delta_g=path_a_src_dg,
            path_b_source_delta_g=path_b_src_dg,
        )
        partial["source_self_delta_g_a"] = path_a_src_dg
        partial["source_self_delta_g_b"] = path_b_src_dg
        partial["verdict"] = verdict_tag
        partial["diagnostic"] = diagnostic
        args.out_path.write_text(json.dumps(partial, indent=2))

    # ── Phase 3: human-readable report. ──────────────────────────────────────
    _print_report(
        partial,
        eval_personas,
        q_eval,
        source_name,
        verdict_tag,
        diagnostic,
        adapter_dir,
        args.out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
