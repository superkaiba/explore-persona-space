"""Issue #605 Phase 2 — marker trained-side sweep over the matched panel.

Cells: (16 reused #474 loc-arm ep1 source adapters) x (selected panel
contexts) x (50 ``q_test_extended_50`` probes). Per cell:

  stage gen   (vLLM + LoRA hot-swap): on-policy R_trained, greedy
              max_new_tokens=2048; per-cell JSON checkpoint + on-policy
              emission columns (token-id read, never substring).
  stage reads (HF): corrected-slot four-float reads (log P(marker),
              z_marker, z_eos id 151645, logZ) for BOTH model sides at the
              IDENTICAL slot — base bf16 forward, then PeftModel adapter
              forward, per the #532 follow-up protocol; per-cell JSONs in
              the ``issue532_followup_logp_v1`` per_q/summary schema so the
              416 legacy anchor cells join the analysis frame without
              translation.
  stage upload: ONE bulk fail-loud folder upload of the per-cell artifacts
              to the HF data repo (``issue605_matched_panels/marker/``).

Smoke = sweep with one cell (plan 4.7): ``--sources A1 --panel-subset 4
--n-probes 5`` — same dispatcher, same subprocess shape, same per-cell
writer, same teardown. ``--adapter-smoke`` runs the Phase-0 gate first:
the A1 self-cell trained read must land within 1 nat of the recorded #532
follow-up value (incident #534 — off-line eval must reproduce the in-loop
read before any sweep).

Pod-side contract: ``[phase=...]`` log lines per stage; the terminal
``[phase=done]`` is emitted ONLY by the top-level ``--stage all`` process
(never by sub-stage subprocesses or per-cell logs); ``--write-sentinel``
(opt-in) writes the /issue Step 7 results sentinel to /workspace/logs.
Saturated cells are RECORDED (flags), never crashed on — high-prior panel
contexts may legitimately saturate (plan section 8 risk 2); the pinned
Tobit / EOS-margin fits carry the read there.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from issue605_contexts import marker_candidates, marker_expansion_candidates  # noqa: E402

from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_test_extended_50,
)

logger = logging.getLogger("issue605.eval_marker")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EOS_ID = 151645
ARM = "loc"
EP = 1
SOURCES_ALL = list(CONDITIONS_BY_ID)
DEFAULT_OUT = Path("eval_results/issue_605/marker")
DEFAULT_PANEL = Path("eval_results/issue_605/panel/marker_panel_selection.json")
ADAPTER_SMOKE_RECORDED = Path(
    "eval_results/issue_532/logp_slot_followup/per_cell_trained/A1__A1.json"
)
ADAPTER_SMOKE_TOL_NATS = 1.0
HF_BUCKET = "issue605_matched_panels"
GPU_HOURS_BUDGETED = 18.0  # plan section 9


def _assert_marker_token(tokenizer) -> None:
    """In-process marker assert (incident #537)."""
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert ids == [MARKER_ID], f"MARKER_TEXT encodes to {ids}, expected [{MARKER_ID}]"
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == EOS_ID


def _repro_meta(extra: dict | None = None) -> dict:
    from issue532_predictor_stress import _reproducibility_metadata

    return _reproducibility_metadata(extra)


def _resolve_contexts(panel_path: Path, panel_subset: int | None) -> list[str]:
    """Panel context labels from the Phase-1.5 selection JSON, optionally cut
    to the first N (the smoke parameterization — same list source as sweep).

    REFUSES a gate_pass=false selection unless it carries the recorded
    pre-registered descope (then restricts to the surviving-band subset) —
    plan section 7 gate 2 blocks trained-side GPU spend (round-1 blocker
    ``panel-gate-not-enforced``)."""
    sel = json.loads(panel_path.read_text())
    if sel.get("gate_pass", False):
        panel = list(sel["panel"])
    else:
        desc = sel.get("descope") or {}
        if not desc.get("active"):
            raise SystemExit(
                f"REFUSING panel {panel_path}: gate_pass=false with no recorded descope — "
                "the Phase-1.5 selection gate BLOCKS trained-side GPU spend (plan section 7 "
                "gate 2). Re-run selection after the pre-registered expansion round, or with "
                "--allow-descope to record the descope-to-populated-bands path."
            )
        panel = list(desc["panel_descoped"])
        logger.warning(
            "descoped panel in effect: bands %s, %d contexts",
            desc["surviving_bands"],
            len(panel),
        )
    if panel_subset is not None:
        panel = panel[:panel_subset]
    assert panel, f"empty panel in {panel_path}"
    return panel


def _dispatch_panel() -> dict[str, str]:
    """label -> system prompt for every non-condition context this issue can
    eval (new candidates + expansion + the 10 legacy instructed)."""
    from issue532_predictor_stress import _instructed_bystander_panel

    cands = marker_candidates()
    cands.update(marker_expansion_candidates())
    panel = {lb: c["system_prompt"] for lb, c in cands.items()}
    panel.update(_instructed_bystander_panel())
    return panel


def _build_prompt(label: str, q: str, tokenizer, class_d, dispatch_panel) -> str:
    from issue532_predictor_stress import _build_bystander_prompt

    return _build_bystander_prompt(label, q, tokenizer, class_d, dispatch_panel)


# ---------------------------------------------------------------------------
# stage gen — vLLM + LoRA hot-swap
# ---------------------------------------------------------------------------
def _log_skip_enumeration(phase: str, pending: list[tuple[str, str]], n_total: int) -> None:
    """Explicit skipped/pending cell enumeration (--skip-completed; the
    per-cell existence skip itself is ALWAYS on — this logs the realized
    split so a wide-panel relaunch shows exactly which NEW cells execute)."""
    labels = [f"{s}->{c}" for s, c in pending]
    suffix = "" if len(labels) <= 200 else f" ... (+{len(labels) - 200} more)"
    logger.info(
        "[phase=%s] skip-completed: %d/%d cells already on disk (skipped); %d pending: %s%s",
        phase,
        n_total - len(pending),
        n_total,
        len(pending),
        labels[:200],
        suffix,
    )


def stage_gen(
    out_dir: Path,
    sources: list[str],
    contexts: list[str],
    n_probes: int,
    dry_run: bool,
    enumerate_skips: bool = False,
) -> None:
    """On-policy R_trained per (source, context) cell; per-cell checkpoint."""
    from issue532_predictor_stress import _compute_in_R_emission
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    _assert_marker_token(tokenizer)
    q_test = load_q_test_extended_50()[:n_probes]
    class_d = load_class_d_rewrites()
    dispatch_panel = _dispatch_panel()
    gen_dir = out_dir / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)

    cells = [(s, c) for s in sources for c in contexts if not (gen_dir / f"{s}__{c}.json").exists()]
    logger.info("[phase=p2_gen] %d cells pending", len(cells))
    if enumerate_skips:
        _log_skip_enumeration("p2_gen", cells, len(sources) * len(contexts))
    if dry_run:
        for s, c in cells[:2]:
            p = _build_prompt(c, q_test[0], tokenizer, class_d, dispatch_panel)
            assert p, (s, c)
        logger.info("[phase=p2_gen] dry-run: prompts build cleanly; stopping before vLLM load")
        return
    if not cells:
        return

    from issue532_predictor_stress import (
        _build_vllm_engine,
        _download_adapters,
        _vllm_generate_R,
    )
    from vllm.lora.request import LoRARequest

    adapter_paths = _download_adapters(ARM, EP, sources)
    llm = _build_vllm_engine(max_seq_len=4096, enable_lora=True)
    for src_idx, src in enumerate(sources):
        lora_req = LoRARequest(
            lora_name=f"{ARM}_{src}_ep{EP}",
            lora_int_id=src_idx + 1,
            lora_path=adapter_paths[src],
        )
        for ctx in contexts:
            cell_path = gen_dir / f"{src}__{ctx}.json"
            if cell_path.exists():
                continue
            prompts = [_build_prompt(ctx, q, tokenizer, class_d, dispatch_panel) for q in q_test]
            R_list = _vllm_generate_R(
                llm, prompts, cell_label=f"P2-genR/{src}->{ctx}", lora_request=lora_req
            )
            emit_any, emit_end = [], []
            for R in R_list:
                ea, ee = _compute_in_R_emission(R, tokenizer)
                emit_any.append(ea)
                emit_end.append(ee)
            payload = {
                "schema_version": "issue605_v1",
                "phase": "p2_marker_gen",
                "arm": ARM,
                "epoch": EP,
                "source_cid": src,
                "context_label": ctx,
                "n_probes": len(q_test),
                "R_trained_per_q": R_list,
                "in_R_emit_anywhere_per_q": emit_any,
                "in_R_emit_at_end_per_q": emit_end,
                "summary": {
                    "in_R_emission_rate": float(np.mean(emit_any)),
                    "in_R_emit_at_end_rate": float(np.mean(emit_end)),
                },
                "metadata": _repro_meta(),
            }
            tmp = cell_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=1))
            tmp.replace(cell_path)
            logger.info(
                "[phase=p2_gen] %s->%s emit=%.2f (cell complete)",
                src,
                ctx,
                payload["summary"]["in_R_emission_rate"],
            )


# ---------------------------------------------------------------------------
# stage reads — HF four-float slot reads, both sides at the identical slot
# ---------------------------------------------------------------------------
def stage_reads(
    out_dir: Path,
    sources: list[str],
    contexts: list[str],
    n_probes: int,
    dry_run: bool,
    enumerate_skips: bool = False,
) -> None:
    """Base + trained corrected-slot reads on R_trained (slot-matched)."""
    from issue532_followup_logp_slot import _run_slot_batches, _slot_job, _summarize
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    _assert_marker_token(tokenizer)
    bare_ids = tokenizer.encode("※", add_special_tokens=False)
    assert len(bare_ids) == 1, bare_ids
    bare_marker_id = bare_ids[0]
    q_test = load_q_test_extended_50()[:n_probes]
    class_d = load_class_d_rewrites()
    dispatch_panel = _dispatch_panel()
    gen_dir = out_dir / "gen"
    base_dir = out_dir / "per_cell_base"
    trained_dir = out_dir / "per_cell_trained"
    base_dir.mkdir(parents=True, exist_ok=True)
    trained_dir.mkdir(parents=True, exist_ok=True)

    def cell_jobs(src: str, ctx: str) -> list[dict]:
        payload = json.loads((gen_dir / f"{src}__{ctx}.json").read_text())
        R_list = payload["R_trained_per_q"]
        assert len(R_list) == len(q_test), (src, ctx, len(R_list))
        return [
            _slot_job(
                _build_prompt(ctx, q, tokenizer, class_d, dispatch_panel),
                R,
                tokenizer,
                bare_marker_id,
            )
            for q, R in zip(q_test, R_list, strict=True)
        ]

    pending_base = [
        (s, c) for s in sources for c in contexts if not (base_dir / f"{s}__{c}.json").exists()
    ]
    pending_trained = [
        (s, c) for s in sources for c in contexts if not (trained_dir / f"{s}__{c}.json").exists()
    ]
    logger.info(
        "[phase=p2_reads] pending: %d base cells, %d trained cells",
        len(pending_base),
        len(pending_trained),
    )
    if enumerate_skips:
        n_total = len(sources) * len(contexts)
        _log_skip_enumeration("p2_reads_base", pending_base, n_total)
        _log_skip_enumeration("p2_reads_trained", pending_trained, n_total)
    if dry_run:
        logger.info("[phase=p2_reads] dry-run: stopping before HF load")
        return

    import torch
    from issue532_predictor_stress import _download_adapters
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter_paths = _download_adapters(ARM, EP, sources)
    # Gauge assert (marker-leakage-measurement.md): logit readouts are valid
    # only when LoRA does not touch the unembedding.
    for src in sources:
        cfg = json.loads((Path(adapter_paths[src]) / "adapter_config.json").read_text())
        targets = set(cfg.get("target_modules") or [])
        assert not targets & {"lm_head", "embed_tokens"}, (src, sorted(targets))
        assert not cfg.get("modules_to_save"), (src, cfg.get("modules_to_save"))

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    base.eval()

    import wandb

    run = wandb.init(
        project="exp605-matched-panels",
        name=f"marker_reads_{'_'.join(sources[:4])}{'_etc' if len(sources) > 4 else ''}",
        config={"sources": sources, "n_contexts": len(contexts), "n_probes": n_probes},
    )

    def write_cell(direc: Path, phase: str, src: str, ctx: str, reads: list[dict]) -> dict:
        summary = _summarize(reads)
        payload = {
            "schema_version": "issue532_followup_logp_v1",
            "phase": phase,
            "arm": ARM,
            "epoch": EP,
            "source_cid": src,
            "bystander_label": ctx,
            "context_label": ctx,
            "n_probes": len(reads),
            "per_q": reads,
            "summary": summary,
            "metadata": _repro_meta(),
        }
        tmp = (direc / f"{src}__{ctx}.json").with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(direc / f"{src}__{ctx}.json")
        return summary

    # Base side first (one model residency), then per-source adapter swaps.
    for src, ctx in pending_base:
        reads = _run_slot_batches(
            base, tokenizer, cell_jobs(src, ctx), bare_marker_id, label=f"P2-base/{src}->{ctx}"
        )
        s = write_cell(base_dir, "p2_base_on_trained_R", src, ctx, reads)
        logger.info("[phase=p2_reads] base %s->%s logp=%.2f", src, ctx, s["mean_logp_marker"])

    for src in sources:
        todo = [c for c in contexts if not (trained_dir / f"{src}__{c}.json").exists()]
        if not todo:
            continue
        logger.info("[phase=p2_reads] loading adapter %s", adapter_paths[src])
        peft_model = PeftModel.from_pretrained(base, adapter_paths[src])
        peft_model.eval()
        for ctx in todo:
            reads = _run_slot_batches(
                peft_model,
                tokenizer,
                cell_jobs(src, ctx),
                bare_marker_id,
                label=f"P2-trained/{src}->{ctx}",
            )
            s = write_cell(trained_dir, "p2_trained_on_own_R", src, ctx, reads)
            b = json.loads((base_dir / f"{src}__{ctx}.json").read_text())["summary"]
            wandb.log(
                {
                    "cell/trained_logp": s["mean_logp_marker"],
                    "cell/dlogp": s["mean_logp_marker"] - b["mean_logp_marker"],
                    "cell/dmargin": s["mean_marker_eos_margin"] - b["mean_marker_eos_margin"],
                    "cell/argmax_rate": s["argmax_marker_rate"],
                }
            )
            logger.info(
                "[phase=p2_reads] trained %s->%s logp=%.2f dlogp=%.2f argmax=%.2f",
                src,
                ctx,
                s["mean_logp_marker"],
                s["mean_logp_marker"] - b["mean_logp_marker"],
                s["argmax_marker_rate"],
            )
        base = peft_model.unload()
        del peft_model
        torch.cuda.empty_cache()
        logger.info("[phase=p2_reads] source %s complete (adapter unloaded)", src)
    run.finish()


# ---------------------------------------------------------------------------
# adapter-application smoke (plan section 7 gate 1)
# ---------------------------------------------------------------------------
def adapter_smoke(out_dir: Path, n_probes: int) -> None:
    """A1 self-cell: trained log P(marker) within 1 nat of the recorded #532
    follow-up value. Runs the SAME gen+reads stages (subprocess), then asserts."""
    smoke_dir = out_dir / "adapter_smoke"
    for stage in ("gen", "reads"):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--sources",
            "A1",
            "--contexts-override",
            "A1",
            "--n-probes",
            str(n_probes),
            "--stage",
            stage,
            "--out-dir",
            str(smoke_dir),
            "--no-done-marker",
        ]
        logger.info("[phase=p0_adapter_smoke] dispatch: %s", " ".join(cmd))
        subprocess.run(cmd, env={**os.environ}, check=True)
    measured = json.loads((smoke_dir / "per_cell_trained" / "A1__A1.json").read_text())
    recorded = json.loads(ADAPTER_SMOKE_RECORDED.read_text())
    m = measured["summary"]["mean_logp_marker"]
    r = recorded["summary"]["mean_logp_marker"]
    delta = abs(m - r)
    assert delta <= ADAPTER_SMOKE_TOL_NATS, (
        f"ADAPTER-APPLICATION SMOKE FAIL: A1 self-cell trained log P(marker) {m:.3f} vs "
        f"recorded #532 value {r:.3f} (|delta|={delta:.3f} > {ADAPTER_SMOKE_TOL_NATS} nat). "
        "Off-line eval path is NOT applying the adapter (incident #534 class) — infra fix, "
        "not a science verdict."
    )
    logger.info(
        "[phase=p0_adapter_smoke] PASS: A1 self-cell %.3f vs recorded %.3f (|d|=%.3f nat)",
        m,
        r,
        delta,
    )


# ---------------------------------------------------------------------------
# stage upload — bulk fail-loud HF data-repo upload
# ---------------------------------------------------------------------------
def stage_upload(out_dir: Path) -> None:
    """ONE bulk folder upload of per-cell artifacts (gen + both read sides)
    to the HF data repo (raw-completions policy; 256-commits/hr safe)."""
    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload

    url = _upload(
        local_path=out_dir,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_BUCKET}/marker",
        upload_as_file=False,
    )
    if not url:
        raise RuntimeError(
            f"marker per-cell upload to {DEFAULT_DATASET_REPO}/{HF_BUCKET}/marker FAILED "
            "(empty url from hub._upload) — do NOT terminate the pod before this lands."
        )
    logger.info("[phase=p6_upload] marker artifacts uploaded: %s", url)


def _write_results_sentinel(out_dir: Path, note: dict) -> Path:
    """/issue Step 7 results sentinel (poll_pipeline contract)."""
    epoch = int(time.time())
    sentinel_dir = Path("/workspace/logs")
    if not sentinel_dir.exists():
        sentinel_dir = out_dir
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    path = sentinel_dir / f"issue-605-epm_results-{epoch}.json"
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:results",
                "version": 1,
                "task_id": 605,
                "ts": epoch,
                "note": note,
            },
            indent=2,
        )
    )
    logger.info("results sentinel written: %s", path)
    return path


def _sentinel_note(out_dir: Path, sources: list[str], contexts: list[str]) -> dict:
    trained_dir = out_dir / "per_cell_trained"
    dlogps = []
    for f in sorted(trained_dir.glob("*.json")):
        t = json.loads(f.read_text())["summary"]["mean_logp_marker"]
        b_path = out_dir / "per_cell_base" / f.name
        if b_path.exists():
            b = json.loads(b_path.read_text())["summary"]["mean_logp_marker"]
            dlogps.append(t - b)
    return {
        "eval_numbers": {
            "n_cells_trained": len(list(trained_dir.glob("*.json"))),
            "n_cells_expected": len(sources) * len(contexts),
            "mean_dlogp": float(np.mean(dlogps)) if dlogps else None,
        },
        "eval_paths": [str(out_dir)],
        "reproducibility_card": _repro_meta({"sources": sources, "n_contexts": len(contexts)}),
        "wandb_url": "wandb://exp605-matched-panels",
        "hf_hub_url": f"superkaiba1/explore-persona-space-data/{HF_BUCKET}/marker",
        "worktree_path": str(PROJECT_ROOT),
        "final_commit_sha": _repro_meta().get("git_commit", "unknown"),
        "gpu_hours_used": None,  # filled by the experimenter from pod wall-clock
        "gpu_hours_budgeted": GPU_HOURS_BUDGETED,
        "plan_deviations": [],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Issue #605 Phase 2 marker trained-side sweep (matched panel).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sources", default="all", help="comma list of source cids or 'all'")
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--panel-subset", type=int, default=None, help="first N panel contexts")
    ap.add_argument(
        "--contexts-override",
        default=None,
        help="comma list of context labels (adapter-smoke self-cells; bypasses --panel)",
    )
    ap.add_argument("--n-probes", type=int, default=50)
    ap.add_argument("--stage", choices=["gen", "reads", "upload", "all"], default="all")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--adapter-smoke", action="store_true", help="run the Phase-0 gate only")
    ap.add_argument(
        "--skip-completed",
        action="store_true",
        help="per-cell resume-skip: cells whose gen + per_cell_trained + per_cell_base files "
        "already exist are skipped. This is the ALWAYS-ON persistence contract (existence-"
        "gated in every stage); the flag additionally logs the explicit skipped/pending cell "
        "enumeration and is the amendment plan §5 launch spelling — against a wide panel "
        "JSON only the NEW cells execute and parent files are never touched.",
    )
    ap.add_argument("--write-sentinel", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-done-marker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    sources = SOURCES_ALL if args.sources == "all" else args.sources.split(",")
    unknown = [s for s in sources if s not in CONDITIONS_BY_ID]
    assert not unknown, f"unknown source cids: {unknown}"

    if args.adapter_smoke:
        adapter_smoke(args.out_dir, args.n_probes)
        return

    if args.contexts_override:
        contexts = args.contexts_override.split(",")
    else:
        contexts = _resolve_contexts(args.panel, args.panel_subset)

    t0 = time.time()
    if args.stage == "all":
        for st in ("gen", "reads"):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--sources",
                ",".join(sources),
                "--n-probes",
                str(args.n_probes),
                "--stage",
                st,
                "--out-dir",
                str(args.out_dir),
                "--no-done-marker",
            ]
            if args.contexts_override:
                cmd += ["--contexts-override", args.contexts_override]
            else:
                cmd += ["--panel", str(args.panel)]
                if args.panel_subset is not None:
                    cmd += ["--panel-subset", str(args.panel_subset)]
            if args.dry_run:
                cmd.append("--dry-run")
            if args.skip_completed:
                cmd.append("--skip-completed")
            logger.info("[stage-dispatch] %s", " ".join(cmd))
            subprocess.run(cmd, env={**os.environ}, check=True)
        if not args.dry_run:
            stage_upload(args.out_dir)
    elif args.stage == "gen":
        stage_gen(args.out_dir, sources, contexts, args.n_probes, args.dry_run, args.skip_completed)
    elif args.stage == "reads":
        stage_reads(
            args.out_dir, sources, contexts, args.n_probes, args.dry_run, args.skip_completed
        )
    elif args.stage == "upload":
        stage_upload(args.out_dir)

    if args.write_sentinel and not args.dry_run:
        _write_results_sentinel(args.out_dir, _sentinel_note(args.out_dir, sources, contexts))
    if not args.no_done_marker:
        logger.info("[phase=done] eval_marker %s in %.0fs", args.stage, time.time() - t0)
    else:
        logger.info("eval_marker sub-stage %s complete in %.0fs", args.stage, time.time() - t0)


if __name__ == "__main__":
    main()
