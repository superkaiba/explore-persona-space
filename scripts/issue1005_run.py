#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (→, ×, ｜) in scientific docstrings + log messages.
"""Issue #1005 phase driver: Gate → G → P → B → F1 → MLP → F2/F3 → figures → finalize.

The plan §4 thin driver — ONE provision, ONE unified 18-vector capture, phased
fits — wiring the issue928 machinery with the #1005 model profile
(``issue1005_common``) threaded through its default-preserving keyword
extensions. Smoke = THIS driver with a tiny same-arch model dir (real pinned
R1-distill tokenizer), ``--synthetic-completions`` (fakes ONLY the vLLM GPU
boundary), ``--contexts/--probes`` subset and ``--no-upload`` (fakes ONLY the
Hub boundary) — the smoke/production unification contract (plan §4.7).

Phases (each SKIPs completed work via its resume predicate; every phase
persists its output the moment it completes — checkpoint-per-phase):

- **gate (Phase 0, plan §7 v3):** family-aware 5-context slice (3 non-ICL/
  WildChat + 1 ICL + 1 WildChat); rungs ``("greedy", "sample")``; terminal
  conjunct A scoped to the non-collapse slice contexts, conjunct B over the
  full slice, conjunct C non-terminal → ONE 16,384 slice re-measure (p95 over
  16,384 escalates terminal); ICL/WildChat slice rates recorded as an early
  coverage READ. Rung-(ii) exhaustion on A/B is TERMINAL (exit 3 + failure
  sentinel; the orchestrator posts ``epm:failure`` ``failure_class: data``).
- **generate (G):** vLLM over token-ID prompts (``build_prompt_ids`` — the
  exactly-one-bos contract; vLLM's own text tokenization would re-add bos),
  per-GROUP rollout persistence, ``--skip-gen`` keyed resume.
- **parse (P):** ``segment_completion`` with rung="prefill" semantics on ALL
  rungs (the ``TEMPLATE_FORCES_THINK`` switch); truncation accounting + ONE
  16,384 re-generation when > 10% rows are cap-truncated (skipped when the
  gate already raised the production cap to 16,384); rollout text uploads
  UNCONDITIONALLY right after this phase; ``coverage_by_family.json`` written.
- **capture (B):** unified 18-vector teacher-forced capture (12 adjusted + 4
  matched-length + 2 prefix); matched-length floor-failing rows KEPT with NaN
  MLC slots + ``mlc_row_mask`` (F2/F3 subset by the mask; F1's row set is
  unchanged — plan §4.0.2); per-row lexical-overlap + prefix bookkeeping;
  within-run determinism spot-check (2 contexts re-captured in a fresh model
  load; #779 two-bar bf16 gate: early L0-3 >= 0.999, flat >= 0.98, finite cells).
- **f1 / mlp / f2f3 / figures:** subprocess calls into the parent fit modules
  (``issue928_fit_decomposition`` / ``issue928_mlp_indiv_control`` /
  ``issue1005_f2f3`` / ``issue928_figures``) on the produced store — the cell
  list derives from the store manifest, never a hardcoded grid.
- **finalize:** the ONE ``epm:results`` sentinel (issue-1005 naming) + the
  terminal ``[phase=done]`` line.

Usage::

    # production (GCP capture-7b lane):
    uv run python scripts/issue1005_run.py --gpu \\
        --out-dir data/issue_1005 --eval-out eval_results/issue_1005

    # CPU-only VM smoke (tiny same-arch model dir built by
    # issue1005_tiny_e2e_fixture.py; synthetic completions replace ONLY vLLM):
    uv run python scripts/issue1005_run.py --smoke --device cpu \\
        --model /tmp/issue-1005-smoke/tiny_model --synthetic-completions \\
        --contexts 6 --probes 4 --no-upload \\
        --out-dir /tmp/issue-1005-smoke/data --eval-out /tmp/issue-1005-smoke/eval \\
        --figures-dir /tmp/issue-1005-smoke/figures --log-dir /tmp/issue-1005-smoke/logs \\
        --layers 0 1 --n-perms 10 --n-boot 50
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))
# vLLM v1 EngineCore dies silently under fork() when the parent touched
# CUDA-adjacent code before LLM() (gotchas.md #628) — set BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import torch  # noqa: E402

# Cross-script helper imports hoisted to module top so a missing symbol crashes
# at process start, never inside a smoke-skipped branch (gotchas.md #606).
from issue594_common import probes_hash  # noqa: E402
from issue594_extract_context_vectors import LayerCapture  # noqa: E402
from issue658_extract_base_store import _reap_vllm  # noqa: E402
from issue928_common import (  # noqa: E402
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    MAX_NEW_TOKENS_RETRY,
    PARSE_RATE_FLOOR,
    TRUNCATION_REGEN_FRAC,
    context_order_and_families,
    dump_json,
    load_probe_pool,
    reproducibility_metadata,
    resolve_battery,
    upload_folder_scoped_verify,
    write_sentinel,
)
from issue928_extract_thinking_store import (  # noqa: E402
    build_capture_row,
    build_vllm_engine,
    pack_batches,
    parse_rows,
    reduce_forward_batch,
    reusable_store_blob,
    rollout_content_digest,
    sampling_params_for_rung,
    vllm_generate_chunked,
)
from issue928_matched_length_control import row_bookkeeping  # noqa: E402
from issue928_prefix_mapping_arms import pma_row_bookkeeping  # noqa: E402
from issue1005_common import (  # noqa: E402
    ANSWER_BOUNDARY_IDS,
    BOUNDARY_POSITIONS,
    COLLAPSE_FAMILIES,
    FALLBACK_RUNGS,
    FIGURES_PREFIX_1005,
    FIT_RESULTS_PREFIX_1005,
    GENERATION_SUFFIX,
    MLC_NAMES_1005,
    MLC_ROW_MASK_KEY,
    MODEL_REVISION,
    PARSER_RUNG,
    POSITION_NAMES_1005,
    PROMPT_POSITIONS,
    RAW_COMPLETIONS_PREFIX_1005,
    STOP_TOKEN_IDS,
    STORE_PREFIX_1005,
    SUMMARY_NAMES_1005,
    THINKING_MODEL,
    build_prompt_ids,
    gate1005_check,
    mlc_parts_spec_1005,
    prompt_parts_spec_1005,
    run_startup_asserts,
    select_gate_slice,
    synthetic_completions_1005,
)

logger = logging.getLogger("issue1005_run")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# bf16-CUDA two-bar structure (the #779 calibration rule, gotchas.md): depth-amplified
# padded-batch kernel numerics concentrate in the LAST layer (measured on THIS store:
# prefix-constancy worst 0.991745, layer 27 in ALL 50 contexts, while layer-0 min is
# >= 0.999999 everywhere — a real span/pad/row bug corrupts layer 0 immediately, cos
# 0.43-0.84 regime). Bar (a) EARLY layers 0-3 = the sharp bug catcher; bar (b) flat
# all-layer = gross corruption. A flat 6-nines/4-nines bar has no headroom on
# near-single-position quantities at deep layers (rounds 2+3 of 2026-07-15).
DETERMINISM_EARLY_COS_MIN = 0.999  # layers 0-3, per finite cell
DETERMINISM_FLAT_COS_MIN = 0.98  # all layers (measured bf16 worst 0.9917, L27-only)
ALL_PHASES = ("extract", "f1", "mlp", "f2f3", "figures", "finalize")


def phase(name: str) -> None:
    """Emit a poll_pipeline.py-parseable phase line."""
    print(f"[phase={name}]", flush=True)


def _run_subprocess(cmd: list[str], phase_name: str, extra_env: dict | None = None) -> None:
    """Run a fit-phase subprocess with an EXPLICIT env (subprocess-env rule)."""
    env = {**os.environ, **(extra_env or {})}
    logger.info("[phase=%s] exec: %s", phase_name, " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def main() -> int:  # noqa: C901 — linear phase pipeline (gate→G→P→B→F→finalize)
    ap = argparse.ArgumentParser(description="Issue #1005: R1-distill CoT decomposition driver")
    ap.add_argument("--model", default=THINKING_MODEL)
    ap.add_argument("--revision", default=MODEL_REVISION, help="pinned Hub revision (plan §4.1)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--gpu", action="store_true", help="force --device cuda")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "issue_1005"))
    ap.add_argument("--eval-out", default=str(PROJECT_ROOT / "eval_results" / "issue_1005"))
    ap.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures" / "issue_1005"))
    ap.add_argument("--log-dir", default=None, help="sentinel dir override (smoke → scratch)")
    ap.add_argument("--battery", default=None, help="local battery.json fast path (sha-pinned)")
    ap.add_argument("--contexts", type=int, default=None, help="cap contexts (smoke)")
    ap.add_argument("--probes", type=int, default=None, help="cap probes/context (smoke)")
    ap.add_argument("--rung", choices=["auto", *FALLBACK_RUNGS], default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--batch-probes", type=int, default=8)
    ap.add_argument("--capture-token-budget", type=int, default=32768)
    ap.add_argument("--expected-layers", type=int, default=EXPECTED_LAYERS)
    ap.add_argument("--expected-hidden", type=int, default=EXPECTED_HIDDEN)
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="fit layer-INDEX subset")
    ap.add_argument("--n-perms", type=int, default=None, help="fit-module default when omitted")
    ap.add_argument("--n-boot", type=int, default=None, help="fit-module default when omitted")
    ap.add_argument("--phases", nargs="*", default=list(ALL_PHASES), choices=list(ALL_PHASES))
    ap.add_argument("--skip-gen", action="store_true", help="reuse rollouts already in out-dir")
    ap.add_argument(
        "--synthetic-completions",
        action="store_true",
        help="CPU smoke ONLY: replace the vLLM call with deterministic synthetic completions "
        "(prefill-shaped — no <think> open tag; every other phase runs the production path)",
    )
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="label + relax model-shape asserts")
    args = ap.parse_args()

    device = args.device or ("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    if device == "cpu" and not args.smoke and torch.cuda.is_available():
        raise SystemExit(
            "[issue1005] REFUSING silent CPU run: CUDA is available but neither --device cuda "
            "nor --gpu was passed. Production capture/fits on CPU ran 6.4x over plan on "
            "2026-07-15 (A100 idle at 0%). Pass --device cuda, or --device cpu --smoke for "
            "the CPU smoke."
        )
    out_dir = Path(args.out_dir)
    eval_out = Path(args.eval_out)
    figures_dir = Path(args.figures_dir)
    log_dir = Path(args.log_dir) if args.log_dir else None
    rollouts_dir = out_dir / "raw_completions" / "thinking_rollouts"
    store_dir = out_dir / "store" / "percq_summaries"
    for d in (rollouts_dir, store_dir, eval_out, figures_dir):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_state_path = out_dir / "run_state.json"

    phase("setup")
    battery = resolve_battery(Path(args.battery) if args.battery else None)
    ctx_ids_all, families = context_order_and_families(battery)
    instances = {i["id"]: i for i in battery["instances"]}
    ctx_ids = ctx_ids_all[: args.contexts] if args.contexts else ctx_ids_all
    probes = load_probe_pool()
    if args.probes:
        probes = probes[: args.probes]
    pool_hash = probes_hash(probes)
    logger.info(
        "contexts=%d probes=%d model=%s@%s device=%s",
        len(ctx_ids),
        len(probes),
        args.model,
        args.revision[:12] if args.revision else "unpinned",
        device,
    )

    # Tokenizer + startup asserts first (CPU): fail loud before any GPU spend.
    from transformers import AutoConfig, AutoTokenizer

    tok_kwargs = {} if Path(args.model).is_dir() else {"revision": args.revision}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = None
    if not args.smoke:
        config = AutoConfig.from_pretrained(args.model, **tok_kwargs)
    asserts_report = run_startup_asserts(tokenizer, battery, config=config)
    logger.info("startup asserts PASS: %s", asserts_report)

    if "extract" in args.phases:
        rc = _phase_extract(
            args,
            device,
            out_dir,
            eval_out,
            log_dir,
            rollouts_dir,
            store_dir,
            run_state_path,
            battery,
            ctx_ids,
            families,
            instances,
            probes,
            pool_hash,
            tokenizer,
            asserts_report,
        )
        if rc != 0:
            return rc

    store_root = str(out_dir / "store")
    fit_env = {"EPM_FIT_DEVICE": device}
    upload = not args.no_upload
    smoke_suffix = "_smoke" if args.smoke else ""

    if "f1" in args.phases:
        phase("f1")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "issue928_fit_decomposition.py"),
            "--store",
            store_root,
            "--out",
            str(eval_out),
        ]
        if args.layers is not None:
            cmd += ["--layers", *[str(x) for x in args.layers]]
        if args.n_perms is not None:
            cmd += ["--n-perms", str(args.n_perms)]
        if args.n_boot is not None:
            cmd += ["--n-boot", str(args.n_boot)]
        if upload:
            cmd += ["--upload-prefix", FIT_RESULTS_PREFIX_1005 + smoke_suffix]
        if args.smoke:
            cmd += ["--smoke", "--no-mlp"]
        _run_subprocess(cmd, "f1", extra_env=fit_env)

    if "mlp" in args.phases:
        phase("mlp")
        man = json.loads((out_dir / "store" / "manifest.json").read_text())
        n_rows = sum(v["n_captured"] for v in man["per_ctx_capture"].values())
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "issue928_mlp_indiv_control.py"),
            "--store",
            store_root,
            "--decomp",
            str(eval_out / "decomp_indiv.pt"),
            "--reference-bootstrap",
            str(eval_out / "bootstrap_deltaskill.json"),
            "--out",
            str(eval_out / "indiv-mlp-nonlinearity-control"),
            "--figures-dir",
            str(figures_dir),
            "--expect-rows",
            str(n_rows),
            "--expect-contexts",
            str(len(man["context_ids"])),
            "--expect-layers",
            str(len(man["capture_layers"])),
            "--expect-hidden",
            str(man["hidden_size"]),
        ]
        if not upload:
            cmd += ["--skip-upload"]
        if args.smoke:
            cmd += ["--allow-cpu-production"]
        _run_subprocess(cmd, "mlp", extra_env=fit_env)

    if "f2f3" in args.phases:
        phase("f2f3")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "issue1005_f2f3.py"),
            "--store",
            store_root,
            "--out",
            str(eval_out),
            "--figures-dir",
            str(figures_dir),
        ]
        if args.layers is not None:
            cmd += ["--layers", *[str(x) for x in args.layers]]
        if args.n_perms is not None:
            cmd += ["--n-perms", str(args.n_perms)]
        if args.n_boot is not None:
            cmd += ["--n-boot", str(args.n_boot)]
        if not upload:
            cmd += ["--no-upload"]
        if args.smoke:
            cmd += ["--smoke"]
        _run_subprocess(cmd, "f2f3", extra_env=fit_env)

    if "figures" in args.phases:
        phase("figures")
        # issue928_figures.py writes via savefig_paper under <out>.parent/"issue_928"
        # at every call site — give it a scratch root whose leaf IS issue_928, then
        # the VM-side analyzer copies to figures/issue_1005 for git (plan §10).
        fig_scratch = out_dir / "figures_scratch" / "issue_928"
        fig_scratch.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "issue928_figures.py"),
            "--results",
            str(eval_out),
            "--store",
            store_root,
            "--rollouts",
            str(rollouts_dir),
            "--out",
            str(fig_scratch),
        ]
        if upload:
            cmd += ["--upload-prefix", FIGURES_PREFIX_1005 + smoke_suffix]
        _run_subprocess(cmd, "figures", extra_env=fit_env)

    if "finalize" in args.phases:
        phase("finalize")
        gate_report = json.loads((out_dir / "gate_report.json").read_text())
        coverage = json.loads((eval_out / "coverage_by_family.json").read_text())
        note = {
            "phase": "issue1005_run_all",
            "n_contexts": len(ctx_ids),
            "rung": gate_report.get("chosen_rung"),
            "production_max_new_tokens": gate_report.get("production_max_new_tokens"),
            "gate": gate_report.get("gate_reports", {}).get(gate_report.get("chosen_rung"), {}),
            "coverage_C_statistic": coverage.get("C_statistic"),
            "coverage_by_family": {
                f: v.get("usable_rate") for f, v in coverage.get("families", {}).items()
            },
            "elapsed_s": round(time.time() - t0, 1),
        }
        write_sentinel("epm:results", note, out_dir, log_dir=log_dir, issue=1005)
        phase("done")
    return 0


def _phase_extract(  # noqa: C901 — linear gate→G→P→B→U pipeline; see phase() markers
    args,
    device,
    out_dir: Path,
    eval_out: Path,
    log_dir,
    rollouts_dir: Path,
    store_dir: Path,
    run_state_path: Path,
    battery,
    ctx_ids,
    families,
    instances,
    probes,
    pool_hash,
    tokenizer,
    asserts_report,
) -> int:
    """Gate → generate → parse → capture → upload (one linear pass)."""
    llm = None
    prompt_ids_cache: dict[tuple[str, int], list[int]] = {}

    def _prompt(c: str, qi: int) -> dict:
        key = (c, qi)
        if key not in prompt_ids_cache:
            _text, ids = build_prompt_ids(tokenizer, instances[c], probes[qi])
            prompt_ids_cache[key] = ids
        return {"prompt_token_ids": prompt_ids_cache[key]}

    def _generate(prompts: list[dict], rung: str, max_new: int) -> list[tuple[str, str]]:
        nonlocal llm
        if args.synthetic_completions:
            return synthetic_completions_1005(prompts, len(probes))
        if llm is None:
            phase("vllm_init")
            llm = build_vllm_engine(
                args.model,
                args.gpu_memory_utilization,
                args.max_model_len,
                revision=None if Path(args.model).is_dir() else args.revision,
            )
        sp = sampling_params_for_rung(rung, max_new, stop_token_ids=STOP_TOKEN_IDS)
        return vllm_generate_chunked(llm, prompts, sp)

    # ── Phase 0: gate walk (plan §7 v3) ───────────────────────────────────────
    phase("gate")
    slice_info = select_gate_slice(ctx_ids, families)
    gate_ctx = slice_info["slice"]
    prior_state = json.loads(run_state_path.read_text()) if run_state_path.is_file() else {}
    chosen_rung = None
    production_cap = args.max_new_tokens
    gate_reports: dict[str, dict] = {}
    gate_completions: dict[str, list[tuple[str, str]]] = {}
    if (
        args.skip_gen
        and prior_state.get("chosen_rung")
        and prior_state.get("gate_terminal_pass")
        and prior_state.get("model") == args.model
        and prior_state.get("probe_pool_hash") == pool_hash
    ):
        chosen_rung = prior_state["chosen_rung"]
        production_cap = int(prior_state["production_max_new_tokens"])
        gate_reports = prior_state.get("gate_reports", {})
        logger.info("[gate] resume: rung=%s cap=%d (run_state)", chosen_rung, production_cap)
    else:
        rungs_to_try = list(FALLBACK_RUNGS) if args.rung == "auto" else [args.rung]
        for rung in rungs_to_try:
            prompts = [_prompt(c, qi) for c in gate_ctx for qi in range(len(probes))]
            comps = _generate(prompts, rung, args.max_new_tokens)
            rows_by_ctx = {
                c: parse_rows(
                    tokenizer, comps[ci * len(probes) : (ci + 1) * len(probes)], PARSER_RUNG
                )
                for ci, c in enumerate(gate_ctx)
            }
            report = gate1005_check(rows_by_ctx, families, args.max_new_tokens, slice_info)
            gate_reports[rung] = report
            logger.info(
                "[gate] rung=%s terminal_pass=%s A=%.3f B=%.4f p95=%.0f coverage_read=%s",
                rung,
                report["terminal_pass"],
                report["conjunct_a"]["usable_rate_non_collapse"],
                report["conjunct_b"]["offender_rate"],
                report["conjunct_c"]["p95_gen_tokens"],
                {
                    k: round(v["usable_rate"], 3)
                    for k, v in report["collapse_family_coverage_read"].items()
                },
            )
            if not report["terminal_pass"]:
                continue  # A or B failed → next rung (rung-(ii) exhaustion is terminal)
            if not report["conjunct_c"]["pass"]:
                # C-only fail: ONE 16,384 slice re-measure (same rung — plan §7).
                phase("gate_c_remeasure")
                comps16 = _generate(prompts, rung, MAX_NEW_TOKENS_RETRY)
                rows16 = {
                    c: parse_rows(
                        tokenizer, comps16[ci * len(probes) : (ci + 1) * len(probes)], PARSER_RUNG
                    )
                    for ci, c in enumerate(gate_ctx)
                }
                report16 = gate1005_check(rows16, families, MAX_NEW_TOKENS_RETRY, slice_info)
                gate_reports[f"{rung}_16k"] = report16
                if not report16["conjunct_c"]["pass"]:
                    logger.error(
                        "[gate] p95 %.0f exceeds even %d — traces structurally exceed the "
                        "contract (terminal, plan §7)",
                        report16["conjunct_c"]["p95_gen_tokens"],
                        MAX_NEW_TOKENS_RETRY,
                    )
                    break  # falls through to the terminal-failure path below
                if not report16["terminal_pass"]:
                    continue  # A/B failed at 16k → next rung
                production_cap = MAX_NEW_TOKENS_RETRY
                chosen_rung = rung
                comps = comps16
                logger.info(
                    "[gate] C-only fail absorbed: production cap raised to %d (§7)", production_cap
                )
            else:
                chosen_rung = rung
            gate_completions = {
                c: comps[ci * len(probes) : (ci + 1) * len(probes)] for ci, c in enumerate(gate_ctx)
            }
            break
    dump_json(
        {
            "gate_reports": gate_reports,
            "chosen_rung": chosen_rung,
            "gate_contexts": gate_ctx,
            "gate_slice": slice_info,
            "production_max_new_tokens": production_cap,
            "startup_asserts": asserts_report,
        },
        out_dir / "gate_report.json",
    )
    if chosen_rung is None:
        # Rung-(ii) exhaustion on terminal conjuncts is TERMINAL (plan §7):
        # the design premise (wholesale-working model/template/parser contract,
        # non-degenerate decoding) is unmet; a model switch is a new plan.
        phase("failed")
        write_sentinel(
            "epm:failure",
            {
                "failure_class": "data",
                "reason": "gate1_terminal_conjuncts_all_rungs_exhausted",
                "gate_reports": gate_reports,
            },
            out_dir,
            log_dir=log_dir,
            issue=1005,
        )
        if llm is not None:
            _reap_vllm(llm)
        return 3
    dump_json(
        {
            "chosen_rung": chosen_rung,
            "gate_terminal_pass": True,
            "production_max_new_tokens": production_cap,
            "model": args.model,
            "probe_pool_hash": pool_hash,
            "gate_reports": gate_reports,
        },
        run_state_path,
    )

    # ── rollout persistence + resume (parent per-GROUP contract) ──────────────
    completions_by_ctx: dict[str, list[tuple[str, str]]] = {}

    def _persist_rollout(c: str, regen_rows: list[int] | None = None) -> None:
        blob = {
            "context_id": c,
            "family": families[c],
            "rung": chosen_rung,
            "model": args.model,
            "model_revision": args.revision,
            "max_new_tokens": production_cap,
            "probe_pool_hash": pool_hash,
            "completions": [
                {"probe": q, "completion": t, "finish_reason": fr}
                for q, (t, fr) in zip(probes, completions_by_ctx[c], strict=True)
            ],
        }
        if regen_rows is not None:
            blob["regen_16k_rows"] = regen_rows
        dump_json(blob, rollouts_dir / f"{c}.json")

    def _rollout_blob_mismatch(blob: dict, c: str) -> str:
        for key, want in (
            ("context_id", c),
            ("model", args.model),
            ("rung", chosen_rung),
            ("probe_pool_hash", pool_hash),
            ("max_new_tokens", production_cap),
        ):
            if blob.get(key) != want:
                return key
        if [r.get("probe") for r in blob.get("completions", [])] != probes:
            return "probe_list"
        return ""

    phase("generate")
    loaded_from_disk: set[str] = set()
    if args.skip_gen:
        for c in ctx_ids:
            p = rollouts_dir / f"{c}.json"
            if not p.is_file():
                continue
            blob = json.loads(p.read_text())
            why = _rollout_blob_mismatch(blob, c)
            if why:
                logger.warning("[skip-gen] rollout %s.json stale (%s) — regenerating", c, why)
                continue
            completions_by_ctx[c] = [
                (r["completion"], r.get("finish_reason", "stop")) for r in blob["completions"]
            ]
            loaded_from_disk.add(c)
        logger.info("[skip-gen] reusing %d/%d rollout files", len(loaded_from_disk), len(ctx_ids))
    for c, comps_c in gate_completions.items():
        if c not in completions_by_ctx:
            completions_by_ctx[c] = comps_c
            _persist_rollout(c)
    remaining = [c for c in ctx_ids if c not in completions_by_ctx]
    if remaining:
        chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
        ctx_per_group = max(1, chunk_size // max(1, len(probes)))
        n_groups = (len(remaining) + ctx_per_group - 1) // ctx_per_group
        for gi in range(0, len(remaining), ctx_per_group):
            group = remaining[gi : gi + ctx_per_group]
            prompts = [_prompt(c, qi) for c in group for qi in range(len(probes))]
            comps = _generate(prompts, chosen_rung, production_cap)
            for ci, c in enumerate(group):
                completions_by_ctx[c] = comps[ci * len(probes) : (ci + 1) * len(probes)]
                _persist_rollout(c)
            logger.info(
                "[generate] group %d/%d persisted (%d/%d remaining contexts)",
                gi // ctx_per_group + 1,
                n_groups,
                min(gi + ctx_per_group, len(remaining)),
                len(remaining),
            )

    # ── Phase P: parse (prefill semantics on ALL rungs) + truncation regen ────
    phase("parse")
    parse_by_ctx: dict[str, list[dict]] = {
        c: parse_rows(tokenizer, completions_by_ctx[c], PARSER_RUNG) for c in ctx_ids
    }
    all_rows = [r for c in ctx_ids for r in parse_by_ctx[c]]
    trunc_frac = sum(1 for r in all_rows if r["finish_reason"] == "length") / max(1, len(all_rows))
    regen_16k = False
    if (
        trunc_frac > TRUNCATION_REGEN_FRAC
        and production_cap < MAX_NEW_TOKENS_RETRY
        and not args.skip_gen
    ):
        phase("regen16k")
        regen_16k = True
        targets = [
            (c, qi)
            for c in ctx_ids
            for qi, r in enumerate(parse_by_ctx[c])
            if r["finish_reason"] == "length"
        ]
        prompts = [_prompt(c, qi) for c, qi in targets]
        comps = _generate(prompts, chosen_rung, MAX_NEW_TOKENS_RETRY)
        for (c, qi), new in zip(targets, comps, strict=True):
            completions_by_ctx[c][qi] = new
        for c in {c for c, _qi in targets}:
            _persist_rollout(c, regen_rows=[qi for cc, qi in targets if cc == c])
            parse_by_ctx[c] = parse_rows(tokenizer, completions_by_ctx[c], PARSER_RUNG)
    parse_report = {
        c: {
            "n_rows": len(parse_by_ctx[c]),
            "n_well_formed": sum(1 for r in parse_by_ctx[c] if r["well_formed"]),
            "parse_rate": sum(1 for r in parse_by_ctx[c] if r["well_formed"])
            / max(1, len(parse_by_ctx[c])),
            "reasons": {
                reason: sum(1 for r in parse_by_ctx[c] if r["reason"] == reason)
                for reason in {r["reason"] for r in parse_by_ctx[c] if r["reason"]}
            },
        }
        for c in ctx_ids
    }
    flagged = [c for c in ctx_ids if parse_report[c]["parse_rate"] < PARSE_RATE_FLOOR]
    if flagged:
        logger.warning(
            "%d context(s) below the %.0f%% parse floor (kept + flagged): %s",
            len(flagged),
            100 * PARSE_RATE_FLOOR,
            flagged,
        )

    # coverage_by_family.json — THE compliance headline table (plan §6.5).
    fam_rows: dict[str, list[dict]] = {}
    for c in ctx_ids:
        fam_rows.setdefault(families[c], []).extend(parse_by_ctx[c])
    fam_cov = {
        f: {
            "usable_rate": sum(1 for r in rows if r["well_formed"]) / max(1, len(rows)),
            "n_rows": len(rows),
            "reasons": {
                reason: sum(1 for r in rows if r["reason"] == reason)
                for reason in {r["reason"] for r in rows if r["reason"]}
            },
        }
        for f, rows in fam_rows.items()
    }
    collapse_rates = [fam_cov[f]["usable_rate"] for f in COLLAPSE_FAMILIES if f in fam_cov]
    c_stat = (min(collapse_rates) - 0.95) if collapse_rates else None
    dump_json(
        {
            "dv": "per-family usable-row rate (scaffold compliance, plan §6)",
            "families": fam_cov,
            "C_statistic": c_stat,
            "C_definition": "min(ICL, WildChat usable rate) - 0.95 (plan §3)",
            "flagged_below_parse_floor": flagged,
            "rung": chosen_rung,
            "production_max_new_tokens": production_cap,
            "reproducibility": reproducibility_metadata(),
        },
        eval_out / "coverage_by_family.json",
    )

    # Rollout text uploads UNCONDITIONALLY right after Phase P (plan §4.3).
    hf_paths: dict = {}
    if not args.no_upload:
        phase("upload_rollouts")
        hf_paths["raw_completions"] = upload_folder_scoped_verify(
            rollouts_dir,
            RAW_COMPLETIONS_PREFIX_1005 + ("_smoke" if args.smoke else ""),
            [f"{c}.json" for c in ctx_ids],
            f"issue #1005: R1-distill thinking rollouts ({len(ctx_ids)} ctx, rung={chosen_rung})",
            allow_patterns=["*.json"],
        )

    # ── Phase B: reap vLLM, unified 18-vector teacher-forced capture ──────────
    if llm is not None:
        phase("reap_vllm")
        _reap_vllm(llm)  # gotchas.md: workers survive `del llm`; HF load OOMs otherwise
        llm = None

    phase("capture")
    from transformers import AutoModelForCausalLM

    def _load_model():
        kwargs = {} if Path(args.model).is_dir() else {"revision": args.revision}
        if device == "cuda":
            return AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map={"": torch.device("cuda:0")},
                **kwargs,
            )
        return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, **kwargs)

    model = _load_model()
    model.eval()
    n_layers = model.config.num_hidden_layers
    if not args.smoke:
        assert n_layers == args.expected_layers, (n_layers, args.expected_layers)
        assert model.config.hidden_size == args.expected_hidden, model.config.hidden_size
    capture_layers = list(range(n_layers))
    capture = LayerCapture(model, n_layers)
    per_ctx_capture: dict[str, dict] = {}
    bookkeeping: dict[str, list[dict]] = {}
    floor_misses: dict[str, int] = {}

    def _build_ctx_rows(c: str) -> tuple[list[dict], list[bool], list[int], dict]:
        """Capture rows for one context: (rows, mlc_ok flags, kept qi, drop reasons)."""
        rows, mlc_flags, kept_qi, drop_reasons = [], [], [], {}
        for qi, (q, (text, _fr)) in enumerate(zip(probes, completions_by_ctx[c], strict=True)):
            rec = parse_by_ctx[c][qi]
            if not rec["well_formed"]:
                continue
            box: dict = {}
            row, why = build_capture_row(
                tokenizer,
                instances[c],
                q,
                text,
                rec,
                chosen_rung,  # greedy/sample: NO prefill append (the template prefills)
                parts_spec=mlc_parts_spec_1005(box),
                prompt_parts_spec=prompt_parts_spec_1005(q),
                generation_suffix=GENERATION_SUFFIX,
                boundary_ids=ANSWER_BOUNDARY_IDS,
                boundary_positions=BOUNDARY_POSITIONS,
                prompt_positions=PROMPT_POSITIONS,
            )
            if row is None:
                drop_reasons[why] = drop_reasons.get(why, 0) + 1
                continue
            rows.append(row)
            mlc_flags.append(bool(box.get("mlc_ok", False)))
            kept_qi.append(qi)
        return rows, mlc_flags, kept_qi, drop_reasons

    base_names = [n for n in SUMMARY_NAMES_1005 if n not in MLC_NAMES_1005]
    name_idx = {n: i for i, n in enumerate(SUMMARY_NAMES_1005)}

    def _capture_ctx(c: str, rows, mlc_flags) -> torch.Tensor:
        """(n_rows, 18, Lc, H) fp16 — floor-failing rows carry NaN MLC slots."""
        H = model.config.hidden_size
        per_q = torch.full(
            (len(rows), len(SUMMARY_NAMES_1005), len(capture_layers), H),
            float("nan"),
            dtype=torch.float16,
        )
        for names, subset in (
            (list(SUMMARY_NAMES_1005), [i for i, ok in enumerate(mlc_flags) if ok]),
            (base_names, [i for i, ok in enumerate(mlc_flags) if not ok]),
        ):
            if not subset:
                continue
            sub_rows = [rows[i] for i in subset]
            chunks, order = [], []
            for batch_idx in pack_batches(sub_rows, args.batch_probes, args.capture_token_budget):
                chunks.append(
                    reduce_forward_batch(
                        model,
                        capture,
                        capture_layers,
                        tokenizer,
                        [sub_rows[i] for i in batch_idx],
                        summary_names=names,
                        position_names=POSITION_NAMES_1005,
                    )
                )
                order.extend(batch_idx)
            stacked = torch.cat(chunks, dim=0)
            inv = torch.empty(len(order), dtype=torch.long)
            inv[torch.tensor(order)] = torch.arange(len(order))
            stacked = stacked[inv]  # restore subset order
            col = torch.tensor([name_idx[n] for n in names], dtype=torch.long)
            row_t = torch.tensor(subset, dtype=torch.long)
            per_q[row_t.unsqueeze(1), col.unsqueeze(0)] = stacked
        return per_q

    try:
        for ci, c in enumerate(ctx_ids):
            blob_path = store_dir / f"{c}.pt"
            if blob_path.is_file():
                prior, why = reusable_store_blob(
                    blob_path,
                    c,
                    model_name=args.model,
                    family=families[c],
                    rung=chosen_rung,
                    probe_pool_hash=pool_hash,
                    capture_layers=capture_layers,
                    summary_names=list(SUMMARY_NAMES_1005),
                    n_probes=len(probes),
                    max_new_tokens=production_cap,
                    rollout_digest=rollout_content_digest(probes, completions_by_ctx[c]),
                    hidden_size=int(model.config.hidden_size),
                )
                if prior is not None and MLC_ROW_MASK_KEY in prior:
                    per_ctx_capture[c] = {
                        "n_captured": len(prior["probe_indices"]),
                        "drop_reasons": prior["coverage"]["capture_drop_reasons"],
                        "resumed": True,
                    }
                    logger.info("[capture] %d/%d %s: SKIPPED (valid blob)", ci + 1, len(ctx_ids), c)
                    continue
                logger.warning(
                    "[capture] %s: existing blob invalid (%s) — recapturing",
                    c,
                    why or f"missing {MLC_ROW_MASK_KEY}",
                )
            rows, mlc_flags, kept_qi, drop_reasons = _build_ctx_rows(c)
            if not rows:
                raise RuntimeError(f"context {c}: zero capturable rows (coverage collapse)")
            per_q = _capture_ctx(c, rows, mlc_flags)
            floor_misses[c] = sum(1 for ok in mlc_flags if not ok)
            bookkeeping[c] = [
                {**row_bookkeeping(rows[i], kept_qi[i]), **pma_row_bookkeeping(rows[i], kept_qi[i])}
                for i, ok in enumerate(mlc_flags)
                if ok
            ]
            blob = {
                "context_id": c,
                "family": families[c],
                "rung": chosen_rung,
                "capture_layers": capture_layers,
                "summary_names": list(SUMMARY_NAMES_1005),
                "probe_indices": kept_qi,
                "per_q": per_q,
                "probe_avg": per_q.float().mean(dim=0).to(torch.float16),
                MLC_ROW_MASK_KEY: [bool(x) for x in mlc_flags],
                "mlc_floors": {"k_min": 8, "rem_min": 16},
                "coverage": {
                    "n_probes_total": len(probes),
                    "n_well_formed": parse_report[c]["n_well_formed"],
                    "n_captured": len(kept_qi),
                    "n_mlc_floor_misses": floor_misses[c],
                    "capture_drop_reasons": drop_reasons,
                },
                "probe_pool_hash": pool_hash,
                "model": args.model,
                "model_revision": args.revision,
                "max_new_tokens": production_cap,
                "rollout_digest": rollout_content_digest(probes, completions_by_ctx[c]),
            }
            tmp = blob_path.with_suffix(".pt.tmp")
            torch.save(blob, tmp)
            os.replace(tmp, blob_path)
            per_ctx_capture[c] = {"n_captured": len(kept_qi), "drop_reasons": drop_reasons}
            logger.info(
                "[capture] %d/%d %s: %d/%d rows (%d MLC-floor misses)",
                ci + 1,
                len(ctx_ids),
                c,
                len(kept_qi),
                len(probes),
                floor_misses[c],
            )
    finally:
        capture.remove()

    # ── within-run determinism spot-check (plan §4.5 — replaces the parent's
    # cross-store parity gates, which have no referent in a single capture) ────
    phase("determinism_check")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    model2 = _load_model()
    model2.eval()
    capture2 = LayerCapture(model2, n_layers)
    det_report = {}
    try:
        for c in ctx_ids[:2]:
            blob = torch.load(store_dir / f"{c}.pt", weights_only=False)
            rows, mlc_flags, kept_qi, _dr = _build_ctx_rows(c)
            assert kept_qi == [int(q) for q in blob["probe_indices"]], (c, kept_qi)
            model, capture = model2, capture2  # _capture_ctx closes over these names
            fresh = _capture_ctx(c, rows, mlc_flags)
            a, b = fresh.float(), blob["per_q"].float()
            finite = torch.isfinite(a).all(dim=-1) & torch.isfinite(b).all(dim=-1)
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # (n, S, Lc)
            early_f = finite[..., :4]
            early = float(cos[..., :4][early_f].min()) if early_f.any() else float("nan")
            flat = float(cos[finite].min()) if finite.any() else float("nan")
            det_report[c] = {
                "cos_min_early_l0_3": early,
                "cos_min_flat": flat,
                "n_finite_cells": int(finite.sum()),
            }
            assert early >= DETERMINISM_EARLY_COS_MIN, (
                f"determinism spot-check FAILED for {c}: EARLY-layer (0-3) min cosine "
                f"{early:.8f} < {DETERMINISM_EARLY_COS_MIN} — layer-0-visible drift is a real "
                "capture/span bug, not bf16 depth noise (refusing)"
            )
            assert flat >= DETERMINISM_FLAT_COS_MIN, (
                f"determinism spot-check FAILED for {c}: flat min cosine {flat:.8f} < "
                f"{DETERMINISM_FLAT_COS_MIN} — beyond the measured bf16 depth-noise envelope "
                "(refusing)"
            )
            logger.info("[determinism] %s: early(L0-3)=%.8f flat=%.8f PASS", c, early, flat)
    finally:
        capture2.remove()
        del model2
        if device == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "context_ids": ctx_ids,
        "families": {c: families[c] for c in ctx_ids},
        "capture_layers": capture_layers,
        "summary_names": list(SUMMARY_NAMES_1005),
        "position_names": list(POSITION_NAMES_1005),
        "hidden_size": int(
            torch.load(store_dir / f"{ctx_ids[0]}.pt", weights_only=False)["per_q"].shape[-1]
        ),
        "rung": chosen_rung,
        "regen_16k": regen_16k,
        "truncation_frac_pre_regen": trunc_frac,
        "gate_report_path": "gate_report.json",
        "parse_report": parse_report,
        "flagged_below_parse_floor": flagged,
        "per_ctx_capture": per_ctx_capture,
        "mlc_floor_misses": floor_misses,
        "determinism_spot_check": det_report,
        "probe_pool_hash": pool_hash,
        "n_probes": len(probes),
        "model": args.model,
        "model_revision": args.revision,
        "max_new_tokens": production_cap,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "reproducibility": reproducibility_metadata(),
        "smoke": args.smoke,
    }
    dump_json(manifest, out_dir / "store" / "manifest.json")
    dump_json(
        {"per_context": bookkeeping, "reproducibility": reproducibility_metadata()},
        out_dir / "store" / "row_bookkeeping.json",
    )
    logger.info("wrote store manifest (%d contexts)", len(ctx_ids))

    if not args.no_upload:
        phase("upload_store")
        hf_paths["store"] = upload_folder_scoped_verify(
            out_dir / "store",
            STORE_PREFIX_1005 + ("_smoke" if args.smoke else ""),
            [
                "manifest.json",
                "row_bookkeeping.json",
                *(f"percq_summaries/{c}.pt" for c in ctx_ids),
            ],
            f"issue #1005: unified 18-vector per-(C,q) store ({len(ctx_ids)} contexts)",
        )

    note = {
        "phase": "extract_done",
        "n_contexts": len(ctx_ids),
        "rung": chosen_rung,
        "production_max_new_tokens": production_cap,
        "flagged_below_parse_floor": flagged,
        "hf_paths": hf_paths,
    }
    write_sentinel("epm:progress", note, out_dir, log_dir=log_dir, issue=1005)
    phase("extract_done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.error("[phase=failed] issue1005 driver crashed:\n%s", traceback.format_exc())
        raise
