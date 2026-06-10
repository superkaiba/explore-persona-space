# ruff: noqa: RUF001  # em-dash + Qwen marker " ※" + Greek ΔG intentional
#!/usr/bin/env python3
"""Task #534 — post-hoc fraction selector over per-step band-stop snapshots.

Plan §4.3 (d): after `MarkerBandStopCallback` has written per-step adapter
snapshots (`<snapshot_dir>/step_NNNN/`) and the `band_stop_meta.json`
sidecar, this script:

  1. Maps fractions {0.25, 0.50, 0.75, 1.00} of the REALIZED stop step S to
     target steps `max(1, round(f*S))`, picks the nearest saved snapshot per
     target (exact when k=1 and S <= snapshot cap), and emits a
     `checkpoint_index.json` in the exact `{frac: {step, path}}` shape
     `scripts/i504_eval_trajectory.py` already consumes.
  2. Runs the gauge check for the logit readout: no `lm_head` / `embed_tokens`
     keys in any selected `adapter_model.safetensors`, `modules_to_save`
     empty in `adapter_config.json`. On trip the manifest carries
     `logit_readout_valid: false` (analysis drops the logit column + flags;
     the log-prob DV is unaffected) — per the marker-leakage-measurement
     rule's gauge assert.
  3. (Unless `--skip-source-trajectory`) reads the per-step SOURCE-self
     trajectory at full snapshot resolution: rebuilds the <=32 source-probe
     rows with the SAME probe builder the in-loop band-stop uses
     (`build_source_probe_from_data`), loads the base model once, then one
     teacher-forced forward per snapshot (PEFT adapter loaded per step) →
     `source_steps_trajectory.json` with per-step log-prob AND raw-logit
     stats (z_marker, z_eos, logZ — the storage contract).
  4. Uploads the 4 SELECTED snapshot dirs to HF
     `<hf-subfolder>/ckpt_frac{0.25,0.50,0.75,1.00}/` reusing the fail-loud
     `_maybe_persist_trajectory_checkpoint` helper (upload + Hub-API
     `list_repo_files` verification), then — only after every upload
     verified — deletes the UNSELECTED snapshots (disk hygiene; the selected
     4 stay local for the nested eval). With `--hf-repo ''` uploads AND the
     deletion are both skipped (never delete unuploaded weights).
  5. Writes `fraction_manifest.json`: the frac→step mapping with `exact`
     flags, `distinct_steps`, the embedded `band_stop_meta`, `stop_reason`,
     the gauge verdict, per-selected-step source ΔG, and provenance.

Dedup/edge rule (plan §4.3 d): if S < 4, fractions collapse onto duplicate
steps. ALL four frac keys are KEPT (the eval rig evals each entry; duplicates
re-eval the same weights — wasteful but correct); `distinct_steps` is recorded
and the analysis treats DISTINCT steps as the unit.

Usage (driven by scripts/i534_run_cell.py between train and eval):
    uv run python scripts/i534_select_fractions.py \\
        --snapshot-dir /workspace/runs/issue_534/c504v3_near_seed42/snapshots \\
        --train-jsonl /workspace/runs/issue_534/c504v3_near_seed42/train_pool.jsonl \\
        --checkpoint-index-out /workspace/runs/issue_534/c504v3_near_seed42/checkpoint_index.json \\
        --manifest-out eval_results/issue_534/c504v3_near_seed42/fraction_manifest.json \\
        --source-traj-out eval_results/issue_534/c504v3_near_seed42/source_steps_trajectory.json \\
        --hf-repo superkaiba1/explore-persona-space \\
        --hf-subfolder adapters/issue_534/c504v3_near_seed42 \\
        --final-adapter /workspace/runs/issue_534/c504v3_near_seed42/adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i534.select_fractions")

DEFAULT_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)


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


def _snapshot_step_dirs(snapshot_dir: Path) -> dict[int, Path]:
    """Return {step: dir} for every `step_NNNN/` snapshot under `snapshot_dir`."""
    out: dict[int, Path] = {}
    for p in sorted(snapshot_dir.glob("step_*")):
        if not p.is_dir():
            continue
        token = p.name.split("_", 1)[1]
        if not token.isdigit():
            continue
        out[int(token)] = p
    return out


def select_fractions(
    snapshot_dir: Path,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> dict[str, Any]:
    """Map fractions of the realized stop step onto the nearest saved snapshots.

    Reads `<snapshot_dir>/band_stop_meta.json` (written by
    `MarkerBandStopCallback.on_train_end`) for the realized stop step S, then
    per fraction f picks `target = max(1, round(f*S))` and the nearest
    available snapshot step (tie → earlier step).

    Returns:
        {"index": {frac_str: {"step": int, "path": str}},   # eval-rig shape
         "manifest": [{"frac", "target_step", "selected_step", "exact"}],
         "stop_meta": <band_stop_meta.json payload>,
         "distinct_steps": int}

    Raises:
        FileNotFoundError: band_stop_meta.json missing (training did not run
            with snapshotting enabled, or the dir is wrong).
        RuntimeError: no step_NNNN snapshots found.
    """
    meta_path = snapshot_dir / "band_stop_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"band_stop_meta.json missing at {meta_path} — training must run with "
            "marker_band_snapshot_every_steps > 0 before fractions can be selected."
        )
    meta = json.loads(meta_path.read_text())
    stop_step = int(meta["stop_step"])
    if not meta.get("stopped", False):
        log.warning(
            "cell never band-stopped (stop_reason=%s); using S=last step %d — "
            "cell FLAGGED in the manifest (excluded from replication claims, "
            "analyzer note #9).",
            meta.get("stop_reason"),
            stop_step,
        )

    available = sorted(_snapshot_step_dirs(snapshot_dir))
    if not available:
        raise RuntimeError(
            f"no step_* snapshot dirs under {snapshot_dir} — the per-step "
            "snapshot callback wrote nothing; investigate before selection."
        )

    step_dirs = _snapshot_step_dirs(snapshot_dir)
    index: dict[str, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    for f in fractions:
        target = max(1, round(f * stop_step))
        sel = min(available, key=lambda s: (abs(s - target), s))  # nearest; tie → earlier
        index[f"{f:.2f}"] = {"step": int(sel), "path": str(step_dirs[sel])}
        manifest.append(
            {
                "frac": float(f),
                "target_step": int(target),
                "selected_step": int(sel),
                "exact": bool(sel == target),
            }
        )
    distinct = len({v["step"] for v in index.values()})
    if distinct < len(fractions):
        log.warning(
            "fractions collapsed onto %d distinct steps (S=%d < %d?) — all frac "
            "keys kept; analysis treats DISTINCT steps as the unit (plan §4.3 d).",
            distinct,
            stop_step,
            len(fractions),
        )
    return {
        "index": index,
        "manifest": manifest,
        "stop_meta": meta,
        "distinct_steps": distinct,
    }


def check_logit_readout_valid(adapter_dir: Path) -> dict[str, Any]:
    """Gauge check: LoRA must not touch the unembedding for the z_marker readout.

    Inspects `adapter_model.safetensors` key names (no weight load) and
    `adapter_config.json`. The logit readout `delta_z_marker = W_U[marker]·Δh`
    is gauge-free ONLY when LoRA does not adapt `lm_head` / `embed_tokens`
    and `modules_to_save` is empty (marker-leakage-measurement.md gauge
    assert; PEFT `all-linear` excludes the output layer by design — this
    verifies rather than assumes it, plan assumption #6).

    Returns a verdict dict (never raises on a gauge trip — the analysis drops
    the logit column + flags instead, per plan §8 risk row).
    """
    weights = adapter_dir / "adapter_model.safetensors"
    cfg_path = adapter_dir / "adapter_config.json"
    problems: list[str] = []
    if not weights.exists():
        problems.append(f"adapter_model.safetensors missing at {weights}")
        return {"valid": False, "problems": problems, "adapter_dir": str(adapter_dir)}
    from safetensors import safe_open

    with safe_open(str(weights), framework="pt", device="cpu") as fh:
        keys = list(fh.keys())
    bad_keys = [k for k in keys if ("lm_head" in k or "embed_tokens" in k)]
    if bad_keys:
        problems.append(f"adapter weights touch the unembedding/embedding: {bad_keys[:5]}")
    if cfg_path.exists():
        acfg = json.loads(cfg_path.read_text())
        mts = acfg.get("modules_to_save") or []
        if mts:
            problems.append(f"modules_to_save is non-empty: {mts}")
        tmods = acfg.get("target_modules")
        if isinstance(tmods, list | tuple) and any(
            ("lm_head" in str(m) or "embed_tokens" in str(m)) for m in tmods
        ):
            problems.append(f"target_modules names the unembedding/embedding: {tmods}")
    else:
        problems.append(f"adapter_config.json missing at {cfg_path}")
    valid = not problems
    if not valid:
        log.warning(
            "[gauge] logit readout INVALID for %s: %s — the analysis must drop "
            "the z_marker column for this cell and flag it (log-prob DV unaffected).",
            adapter_dir,
            problems,
        )
    return {"valid": valid, "problems": problems, "adapter_dir": str(adapter_dir)}


def run_source_steps_trajectory(
    *,
    snapshot_dir: Path,
    train_jsonl: Path,
    base_model: str,
    device: str,
    marker_text: str,
    expected_marker_token_id: int,
    max_rows: int = 32,
) -> dict[str, Any]:
    """Per-step source-self trajectory: one teacher-forced forward per snapshot.

    Rebuilds the source-probe batch with the SAME builder the in-loop
    band-stop uses (`build_source_probe_from_data`, so the trajectory is
    commensurable with the stop decision), loads the base model ONCE, reads
    the base slot stats, then loads each snapshot's PEFT adapter and reads
    the trained slot stats per step. Per the storage contract, every read
    captures log P(marker), z_marker, z_eos, and logZ (means over the probe
    batch; raw logits exist only in this HF forward pass).

    Returns the JSON-serializable trajectory payload.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
    if marker_ids != [expected_marker_token_id]:
        raise RuntimeError(
            f"marker tokenizer assertion FAILED in source-steps trajectory: "
            f"encode({marker_text!r})={marker_ids}, expected [{expected_marker_token_id}]."
        )
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError(f"tokenizer for {base_model!r} has no eos_token_id.")

    from explore_persona_space.train.sft import build_source_probe_from_data

    input_ids, attention_mask, positions, n_rows = build_source_probe_from_data(
        train_jsonl,
        tokenizer,
        marker_ids,
        max_rows=max_rows,
        max_length=2048,
    )
    if n_rows == 0:
        raise RuntimeError(
            f"build_source_probe_from_data found 0 marker-bearing rows in "
            f"{train_jsonl} — cannot read the source trajectory."
        )

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map={"": device} if use_cuda else None,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)
    positions = positions.to(model_device)

    marker_id = expected_marker_token_id

    def _read(m) -> dict[str, float]:
        with torch.no_grad():
            logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
        assert logits.ndim == 3, logits.shape
        b_idx = torch.arange(input_ids.shape[0], device=model_device)
        last = logits[b_idx, positions, :].float()  # (B, V) — slot logits per probe row
        assert last.shape == (input_ids.shape[0], logits.shape[-1]), last.shape
        z_marker = last[:, marker_id]
        z_eos = last[:, eos_id]
        logz = torch.logsumexp(last, dim=-1)
        logp = z_marker - logz
        return {
            "logp_marker_mean": float(logp.mean().item()),
            "z_marker_mean": float(z_marker.mean().item()),
            "z_eos_mean": float(z_eos.mean().item()),
            "logz_mean": float(logz.mean().item()),
        }

    base_stats = _read(model)
    log.info("[source-traj] base read over %d probe rows: %s", n_rows, base_stats)

    from peft import PeftModel

    step_dirs = _snapshot_step_dirs(snapshot_dir)
    steps_out: list[dict[str, Any]] = []
    peft_model = None
    for step in sorted(step_dirs):
        snap = step_dirs[step]
        name = f"step_{step:04d}"
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(snap), adapter_name=name)
            peft_model.eval()
        else:
            peft_model.load_adapter(str(snap), adapter_name=name)
            peft_model.set_adapter(name)
        st = _read(peft_model)
        steps_out.append(
            {
                "step": int(step),
                **st,
                "delta_g_mean": st["logp_marker_mean"] - base_stats["logp_marker_mean"],
                "delta_z_marker_mean": st["z_marker_mean"] - base_stats["z_marker_mean"],
                "delta_z_margin_mean": (st["z_marker_mean"] - st["z_eos_mean"])
                - (base_stats["z_marker_mean"] - base_stats["z_eos_mean"]),
            }
        )
        log.info(
            "[source-traj] step %d: ΔG=%.3f nats, Δz_marker=%.3f, Δ(z_marker−z_eos)=%.3f",
            step,
            steps_out[-1]["delta_g_mean"],
            steps_out[-1]["delta_z_marker_mean"],
            steps_out[-1]["delta_z_margin_mean"],
        )

    del peft_model, model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "schema_version": "i534_source_steps_v1",
        "base_model": base_model,
        "marker_text": marker_text,
        "marker_token_id": marker_id,
        "eos_token_id": int(eos_id),
        "n_probe_rows": int(n_rows),
        "base": base_stats,
        "steps": steps_out,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def _upload_selected(
    selection: dict[str, Any],
    *,
    hf_repo: str,
    hf_subfolder: str,
) -> dict[str, str]:
    """Upload the selected snapshot dirs to `<hf_subfolder>/ckpt_frac{N}` + verify.

    Reuses `_maybe_persist_trajectory_checkpoint` (fail-loud upload + Hub-API
    `list_repo_files` verification) by setting its env-var contract for the
    duration of the call. Returns {frac_str: hub_path_prefix}.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
        _maybe_persist_trajectory_checkpoint,
    )

    prior_repo = os.environ.get("EPM_PERSIST_TRAJECTORY_HF_REPO")
    prior_sub = os.environ.get("EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER")
    os.environ["EPM_PERSIST_TRAJECTORY_HF_REPO"] = hf_repo
    os.environ["EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER"] = hf_subfolder
    uploaded: dict[str, str] = {}
    try:
        for frac_str, entry in sorted(selection["index"].items(), key=lambda kv: float(kv[0])):
            _maybe_persist_trajectory_checkpoint(Path(entry["path"]), float(frac_str), 2)
            uploaded[frac_str] = f"{hf_subfolder}/ckpt_frac{float(frac_str):.2f}"
    finally:
        # Restore so the caller's env contract is untouched (the #534 run-cell
        # deliberately leaves in-train per-fraction persistence DISABLED).
        if prior_repo is None:
            os.environ.pop("EPM_PERSIST_TRAJECTORY_HF_REPO", None)
        else:
            os.environ["EPM_PERSIST_TRAJECTORY_HF_REPO"] = prior_repo
        if prior_sub is None:
            os.environ.pop("EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER", None)
        else:
            os.environ["EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER"] = prior_sub
    return uploaded


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--snapshot-dir", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--checkpoint-index-out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--source-traj-out", type=Path, required=True)
    ap.add_argument(
        "--fractions",
        default="0.25,0.5,0.75,1.0",
        help="Comma-separated fractions of the realized stop step (default plan set).",
    )
    ap.add_argument(
        "--hf-repo",
        default="superkaiba1/explore-persona-space",
        help=(
            "HF model repo for the 4 selected snapshot uploads. Empty string "
            "disables uploads AND the unselected-snapshot deletion (never "
            "delete unuploaded weights)."
        ),
    )
    ap.add_argument(
        "--hf-subfolder",
        default=None,
        help="HF subfolder, e.g. adapters/issue_534/c504v3_near_seed42. Required when uploading.",
    )
    ap.add_argument(
        "--base-model",
        default=None,
        help=(
            "Base model for the per-step source trajectory. Default = the "
            "#504/#530 BASE_MODEL constant (Qwen/Qwen2.5-7B-Instruct)."
        ),
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--skip-source-trajectory",
        action="store_true",
        help="Descope ladder item 1 (plan §9): skip the per-step source-self read.",
    )
    ap.add_argument(
        "--final-adapter",
        type=Path,
        default=None,
        help=(
            "Optional path to the train_lora final adapter dir; when given, the "
            "stop-step snapshot's adapter weights are byte-compared against it "
            "(plan assumption #5 verification; WARN-only — serialization "
            "nondeterminism is tolerated, the manifest records the verdict)."
        ),
    )
    ap.add_argument("--probe-max-rows", type=int, default=32)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=select_fractions] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )

    base_model = args.base_model if args.base_model is not None else BASE_MODEL

    fractions = tuple(sorted(float(x.strip()) for x in args.fractions.split(",") if x.strip()))
    if not fractions or any(f <= 0 or f > 1.0 for f in fractions):
        raise ValueError(f"--fractions {args.fractions!r} must be floats in (0, 1].")

    # ── 1. Selection (pure, CPU). ─────────────────────────────────────────────
    selection = select_fractions(args.snapshot_dir, fractions)
    log.info(
        "[select] stop_step=%d stop_reason=%s → %s (distinct_steps=%d)",
        selection["stop_meta"]["stop_step"],
        selection["stop_meta"].get("stop_reason"),
        {k: v["step"] for k, v in selection["index"].items()},
        selection["distinct_steps"],
    )

    # ── 2. Gauge check on every selected adapter (logit-readout validity). ────
    gauge_results = {
        frac_str: check_logit_readout_valid(Path(entry["path"]))
        for frac_str, entry in selection["index"].items()
    }
    logit_readout_valid = all(g["valid"] for g in gauge_results.values())

    # ── 2b. Optional stop-snapshot vs final-adapter byte-compare. ─────────────
    stop_matches_final: bool | None = None
    if args.final_adapter is not None:
        stop_step = int(selection["stop_meta"]["stop_step"])
        step_dirs = _snapshot_step_dirs(args.snapshot_dir)
        snap = step_dirs.get(stop_step)
        final_w = args.final_adapter / "adapter_model.safetensors"
        if snap is not None and final_w.exists():
            snap_w = snap / "adapter_model.safetensors"
            stop_matches_final = snap_w.read_bytes() == final_w.read_bytes()
            if not stop_matches_final:
                log.warning(
                    "[compare] stop-step snapshot %s differs from final adapter %s "
                    "at the byte level (serialization nondeterminism is possible; "
                    "recorded in the manifest — investigate if ΔG diverges too).",
                    snap_w,
                    final_w,
                )
        else:
            log.warning(
                "[compare] cannot byte-compare stop snapshot vs final adapter "
                "(snapshot at step %d present=%s, final weights present=%s).",
                stop_step,
                snap is not None,
                final_w.exists(),
            )

    # ── 3. Per-step source-self trajectory (GPU; descope-able). ──────────────
    source_traj: dict[str, Any] | None = None
    if args.skip_source_trajectory:
        log.warning("[source-traj] skipped per --skip-source-trajectory (descope ladder 1).")
    else:
        source_traj = run_source_steps_trajectory(
            snapshot_dir=args.snapshot_dir,
            train_jsonl=args.train_jsonl,
            base_model=base_model,
            device=args.device,
            marker_text=MARKER_TEXT,
            expected_marker_token_id=EXPECTED_MARKER_TOKEN_ID,
            max_rows=args.probe_max_rows,
        )
        args.source_traj_out.parent.mkdir(parents=True, exist_ok=True)
        args.source_traj_out.write_text(json.dumps(source_traj, indent=2))
        log.info(
            "[source-traj] wrote %s (%d steps)", args.source_traj_out, len(source_traj["steps"])
        )

    # ── 4. Upload the selected snapshots, fail-loud verified. ────────────────
    uploaded: dict[str, str] = {}
    if args.hf_repo:
        if not args.hf_subfolder:
            raise ValueError("--hf-subfolder is required when --hf-repo is non-empty.")
        uploaded = _upload_selected(selection, hf_repo=args.hf_repo, hf_subfolder=args.hf_subfolder)
        log.info("[upload] %d selected snapshots verified on %s", len(uploaded), args.hf_repo)
    else:
        log.warning(
            "[upload] --hf-repo '' → uploads SKIPPED; unselected snapshots will "
            "NOT be deleted (never delete unuploaded weights)."
        )

    # ── 5. checkpoint_index.json (the eval rig's input shape). ───────────────
    args.checkpoint_index_out.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_index_out.write_text(json.dumps(selection["index"], indent=2))
    log.info("[index] wrote %s", args.checkpoint_index_out)

    # ── 6. fraction_manifest.json (plan §6.5 deliverable 4). ─────────────────
    src_dg_at_selected: dict[str, float | None] = {}
    if source_traj is not None:
        by_step = {s["step"]: s for s in source_traj["steps"]}
        for frac_str, entry in selection["index"].items():
            row = by_step.get(int(entry["step"]))
            src_dg_at_selected[frac_str] = float(row["delta_g_mean"]) if row is not None else None
    manifest_payload = {
        "schema_version": "i534_fraction_manifest_v1",
        "fractions": list(fractions),
        "manifest": selection["manifest"],
        "index": selection["index"],
        "distinct_steps": selection["distinct_steps"],
        "band_stop_meta": selection["stop_meta"],
        "stop_reason": selection["stop_meta"].get("stop_reason"),
        "stopped": bool(selection["stop_meta"].get("stopped", False)),
        "logit_readout_valid": bool(logit_readout_valid),
        "gauge_results": gauge_results,
        "stop_snapshot_matches_final_adapter": stop_matches_final,
        "source_delta_g_at_selected_steps": src_dg_at_selected,
        "source_steps_trajectory_path": (
            str(args.source_traj_out) if source_traj is not None else None
        ),
        "hf_uploads": uploaded,
        "hf_repo": args.hf_repo or None,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest_payload, indent=2))
    log.info("[manifest] wrote %s", args.manifest_out)

    # ── 7. Delete UNSELECTED snapshots — only after Hub-verified uploads. ────
    if args.hf_repo and uploaded:
        selected_steps = {int(v["step"]) for v in selection["index"].values()}
        step_dirs = _snapshot_step_dirs(args.snapshot_dir)
        n_deleted = 0
        for step, d in step_dirs.items():
            if step in selected_steps:
                continue
            shutil.rmtree(d)
            n_deleted += 1
        log.info(
            "[cleanup] deleted %d unselected snapshots (kept %d selected: %s)",
            n_deleted,
            len(selected_steps),
            sorted(selected_steps),
        )
    log.info("[phase=select_fractions_done] selection complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
