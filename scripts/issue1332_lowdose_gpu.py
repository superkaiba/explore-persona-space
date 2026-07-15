"""Issue #1332 follow-up ``lowdose-grid-kill-battery`` — pod driver (P0-P3).

Plan v8 §4. Phases:

- **P0** (in-process): stage the 16 #474 loc-arm training mixes from the HF
  data repo at the pinned revision (per-file sha256 recorded); signature smoke
  (every kwarg the trainer passes asserted against current ``TrainLoraConfig``);
  whole-path reconcile log (``git diff <#474-sha>..origin/main --
  src/explore_persona_space/train/sft.py``); realized-keys check on the reused
  capture store (``cx_last`` present in ``capture/A1.pt`` with all 28 layers).
- **P1** (per shard): 16 band-stopped trainings sharded across every visible
  GPU — independent ``issue1332_lowdose_train.py`` subprocesses, one source at
  a time per shard, ``CUDA_VISIBLE_DEVICES`` pinned per shard in the LAUNCHER
  env (no ``--gpu-id`` is passed: the inherited single-GPU pin is authoritative
  via ``train/sft.py::_apply_cvd_pin``).
- **P2** (gate, source 1 = the panel's first source): off-line #532 slot-rig
  read of the diagonal cell. HALT ONLY iff dG = trained - parent-base is
  outside the structural window (GATE_WINDOW, recalibrated [0.5, 18] nats);
  the ~1-nat in-loop-vs-off-line
  agreement is a persisted WARN (``parity_warn``), never a blocker. The parity
  gate gates the MEASUREMENT sweep; training keeps flowing (work-conserving:
  each shard trains ahead and defers measures until the gate file reads pass).
- **P3** (per shard): 26-target corrected-slot sweep per trained source using
  the #532 rig verbatim (``_slot_job`` / ``_run_slot_batches`` /
  ``_load_parent_cell_R`` — same frozen per-cell R, so slot identity with the
  reused parent base side holds by construction and is ASSERTED per cell
  before differencing; a mismatching cell gets its base side re-measured via
  ``disable_adapter`` and the deviation recorded). Per-source upload to the HF
  data repo the moment the source's cells complete (#664 per-cell contract).

The GCE lane has no ``.env`` — any shell wrapper around this driver must
source conditionally (``if [ -f ./.env ]; then set -a; . ./.env; set +a; fi``);
the driver itself uses the project ``orchestrate.env.load_dotenv`` (graceful).

USAGE
    uv run python scripts/issue1332_lowdose_gpu.py --full
    uv run python scripts/issue1332_lowdose_gpu.py --smoke          # CPU, scratch roots
    (internal: --shard / --measure subcommands, spawned by --full/--smoke)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue532_followup_logp_slot as SLOT
import issue1332_common as C
import issue1332_lowdose_train as LT
from issue1332_gpu_phase import upload_files

logger = logging.getLogger("issue1332.lowdose_gpu")

# P2 structural gate window. RECALIBRATED from plan v8's [2, 18] after the
# registered §8 false-HALT case fired on the first production run
# (att-20260715-211847): A1 stopped in-band at step 8 (in-loop 5.16 nats) but
# the off-line diagonal read was dG=1.319 — the adapter WAS applied (base
# re-measure gap 0.011 nats; slot identity clean); at 8-step dose the implant
# expresses ~4x weaker on the rig battery than on the in-loop probe surface.
# New low edge 0.5 = ~45x the measured base-remeasure noise (unapplied band
# ~0) and ~2.6x below the realized healthy low-dose read; upper edge 18
# unchanged (parent-ep1 ~24 excluded). #813 gate-calibration rule: HALT only
# where the windows separate failure modes — [0.5, 18] does, on MEASURED bands.
GATE_WINDOW = (0.5, 18.0)
# In-loop vs off-line agreement gap above which a WARN is persisted (plan v8
# §4 P2 — DEMOTED to WARN; the two reads share the slot surface but differ in
# loader/eval-mode details, so this is adjudicated at analysis, never a HALT).
PARITY_WARN_NATS = 1.0

# The #474 provenance SHA for the whole-path sft.py reconcile (plan v8 §4 P0):
# the commit that introduced scripts/i474_phase23_train.py (2026-06-02,
# "feat(#474): on-policy divergence-to-transfer with localization restored").
I474_SFT_SHA = "0936cd6d8ce2149377f2e4d25e1d62817399f52d"

N_PROBES = 50
HF_LOWDOSE_PREFIX = f"{C.HF_PREFIX}/lowdose"


def lowdose_dir(smoke: bool, override: str | None = None) -> Path:
    """eval_results/issue_1332/lowdose (scratch under smoke — never canonical)."""
    return C.results_dir(smoke, override) / "lowdose"


def logs_dir() -> Path:
    d = Path("/workspace/logs") if Path("/workspace/logs").parent.is_dir() else None
    if d is None or not Path("/workspace").is_dir():
        d = C.PROJECT_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── pure predicates (pytest-pinned) ───────────────────────────────────────────


def p2_gate_verdict(
    delta_g: float,
    inloop_delta: float | None,
    *,
    window: tuple[float, float] = GATE_WINDOW,
    warn_gap: float = PARITY_WARN_NATS,
) -> dict:
    """P2 verdict: HALT iff dG outside GATE_WINDOW; parity WARN independent (plan §4 P2)."""
    halt = not (window[0] <= delta_g <= window[1])
    parity_gap = None if inloop_delta is None else abs(delta_g - inloop_delta)
    parity_warn = parity_gap is not None and parity_gap > warn_gap
    return {
        "delta_g_nats": float(delta_g),
        "window_nats": [window[0], window[1]],
        "halt": bool(halt),
        "inloop_delta_nats": inloop_delta,
        "parity_gap_nats": parity_gap,
        "parity_warn": bool(parity_warn),
    }


def slot_identity_deviations(jobs: list[dict], parent_per_q: list[dict]) -> list[dict]:
    """Per-row slot-identity check vs the reused parent base per_q (plan §4 P3).

    Same frozen R + same ``_slot_job`` code implies equality of ``slot_kind``
    and ``n_truncated_tokens`` (which, given the same prompt + R + tokenizer,
    pins the slot token position). Returns the mismatching rows (expected: []).
    """
    if len(jobs) != len(parent_per_q):
        raise AssertionError(f"per_q length mismatch: {len(jobs)} jobs vs {len(parent_per_q)}")
    devs = []
    for i, (job, row) in enumerate(zip(jobs, parent_per_q, strict=True)):
        if (
            job["slot_kind"] != row["slot_kind"]
            or job["n_truncated_tokens"] != row["n_truncated_tokens"]
        ):
            devs.append(
                {
                    "q_idx": i,
                    "job_slot_kind": job["slot_kind"],
                    "job_n_truncated": job["n_truncated_tokens"],
                    "parent_slot_kind": row["slot_kind"],
                    "parent_n_truncated": row["n_truncated_tokens"],
                }
            )
    return devs


# ── source/target resolution (the ONE cell-list source every phase reads) ─────


def resolve_sources(arg: str) -> list[str]:
    sources, _targets = C.family_labels()
    if arg == "all":
        return sources
    picked = [s.strip() for s in arg.split(",") if s.strip()]
    unknown = [s for s in picked if s not in sources]
    if unknown:
        raise ValueError(f"unknown sources {unknown}; valid: {sources}")
    return picked


def resolve_targets(arg: str) -> list[str]:
    _sources, targets = C.family_labels()
    if arg == "all":
        return targets
    picked = [t.strip() for t in arg.split(",") if t.strip()]
    unknown = [t for t in picked if t not in targets]
    if unknown:
        raise ValueError(f"unknown targets {unknown}; valid: {targets}")
    return picked


# ── P0 ────────────────────────────────────────────────────────────────────────


def _sft_reconcile(out_dir: Path) -> dict:
    """Run + log the whole-path reconcile diff (plan v8 §4 P0 / §8).

    The semantic review happened at implementation time (report §b); this logs
    the same diff on the worker for the durable record. A git failure here is
    a loud WARNING + manifest field, never a crash (the run's correctness does
    not depend on producing the log).
    """
    rel = "src/explore_persona_space/train/sft.py"

    def _git(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *argv],
            cwd=C.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )

    if _git("rev-parse", "--verify", "origin/main").returncode != 0:
        _git("fetch", "origin", "main", "--quiet")
    stat = _git("diff", "--stat", f"{I474_SFT_SHA}..origin/main", "--", rel)
    if stat.returncode != 0:
        logger.warning("[p0-reconcile] git diff unavailable: %s", stat.stderr.strip()[:300])
        return {"error": stat.stderr.strip()[:300], "base_sha": I474_SFT_SHA}
    body = _git("diff", f"{I474_SFT_SHA}..origin/main", "--", rel)
    diff_log = logs_dir() / "issue-1332-sft-reconcile.diff"
    diff_log.write_text(body.stdout)
    n_hunks = sum(1 for line in body.stdout.split("\n") if line.startswith("@@"))
    logger.info(
        "[p0-reconcile] %s..origin/main -- %s: %s (%d hunks) -> %s",
        I474_SFT_SHA[:12],
        rel,
        stat.stdout.strip().split("\n")[-1].strip(),
        n_hunks,
        diff_log,
    )
    return {
        "base_sha": I474_SFT_SHA,
        "stat": stat.stdout.strip(),
        "n_hunks": n_hunks,
        "diff_log": str(diff_log),
        "note": "reviewed at implementation time: drift = folded #474/#477 slot default "
        "(suppress flag now a no-op) + additive band-stop/bf16/max_steps/_apply_cvd_pin "
        "features; no behavioral hunk changes the #474 loc recipe semantics",
    }


def _capture_keys_check(smoke: bool) -> dict:
    """Realized-keys probe on the reused capture store (artifact-reuse check (c)).

    ``cx_last`` must be present in ``capture/A1.pt`` with all 28 layers so the
    fresh-cos L21 covariate recompute is runnable (plan v8 §4 P0 / §12 item 7).
    """
    import torch

    candidates = [
        C.data_root(False) / "store" / "capture" / "A1.pt",
        C.data_root(True) / "store" / "capture" / "A1.pt",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        if smoke:
            logger.info("[p0-keys] capture/A1.pt absent locally; smoke skips the fetch")
            return {"status": "skipped_smoke_no_local_copy"}
        path = C.hf_fetch("analysis_tensors/capture/A1.pt", candidates[0])
    sh = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    keys = set(sh.keys())
    if "cx_last" not in keys:
        raise AssertionError(f"capture store missing 'cx_last' (realized keys: {sorted(keys)})")
    shape = tuple(sh["cx_last"].shape)
    if len(shape) != 3 or shape[1] != C.N_LAYERS:
        raise AssertionError(
            f"cx_last shape {shape} incompatible with L21 read (expected (n_q, 28, hidden))"
        )
    logger.info("[p0-keys] cx_last present in %s, shape=%s (L21 readable)", path, shape)
    return {"status": "ok", "path": str(path), "cx_last_shape": list(shape)}


def p0(args, sources: list[str]) -> dict:
    """P0: stage mixes + signature smoke + sft reconcile + capture keys check."""
    C.phase("p0_stage")
    LT.verify_config_signature(sources[0])
    mixes = []
    for cid in sources:
        if args.smoke:
            hub_name, local = LT.mix_paths(cid)
            mixes.append(
                {
                    "cid": cid,
                    "hub_path": hub_name,
                    "revision": LT.MIX_REVISION,
                    "local_path": str(local),
                    "staged": False,
                }
            )
        else:
            mixes.append(LT.stage_mix(cid))
    manifest = {
        "phase": "P0",
        "sources": sources,
        "mixes": mixes,
        "sft_reconcile": _sft_reconcile(lowdose_dir(args.smoke, args.results_dir)),
        "capture_keys": _capture_keys_check(args.smoke),
        "reproducibility_metadata": C.reproducibility_metadata(
            {"followup": "lowdose-grid-kill-battery", "smoke": args.smoke}
        ),
    }
    out = lowdose_dir(args.smoke, args.results_dir) / "p0_manifest.json"
    C.write_json_atomic(out, manifest)
    logger.info("[p0] manifest -> %s", out)
    return manifest


# ── GPU detection + sharding ──────────────────────────────────────────────────


def detect_gpus() -> list[int]:
    """Physical GPU ids via an nvidia-smi SUBPROCESS (never torch.cuda — the
    clobbered-env/count-cache trap, gotchas.md library-train-seam rule (ii))."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    return [int(x) for x in proc.stdout.split() if x.strip().isdigit()]


def split_shards(sources: list[str], n_shards: int) -> list[list[str]]:
    """Contiguous split (8+8 at width 2); the gate source stays in shard 0."""
    n_shards = max(1, min(n_shards, len(sources)))
    per = (len(sources) + n_shards - 1) // n_shards
    return [sources[i * per : (i + 1) * per] for i in range(n_shards) if sources[i * per :]]


# ── gate file protocol ────────────────────────────────────────────────────────


def _gate_status(gate_file: Path) -> str | None:
    if not gate_file.exists():
        return None
    return json.loads(gate_file.read_text()).get("status")


def _wait_gate(gate_file: Path, timeout_s: float) -> None:
    t0 = time.time()
    while True:
        status = _gate_status(gate_file)
        if status == "pass":
            return
        if status == "fail":
            raise RuntimeError(f"P2 parity gate FAILED (see {gate_file}) — measurement halted")
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"P2 gate not resolved within {timeout_s}s ({gate_file})")
        time.sleep(15)


# ── measure subcommand (P2 gate + P3 sweep; the #532 rig verbatim) ────────────


def _resolve_adapter_dir(cid: str, adapter_root: Path) -> Path:
    """Local train output first; else per-file HF download (fresh-pod resume)."""
    local = adapter_root / f"i1332_lowdose_{cid}"
    if (local / "adapter_model.safetensors").exists():
        return local
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    sub = LT.hf_adapter_path(cid)
    local.mkdir(parents=True, exist_ok=True)
    for fname in ("adapter_model.safetensors", "adapter_config.json"):
        got = retry_transient(
            lambda fname=fname: hf_hub_download(
                repo_id=LT.HF_MODEL_REPO, filename=f"{sub}/{fname}", revision="main"
            ),
            what=f"fetch {sub}/{fname}",
        )
        import shutil

        shutil.copyfile(got, local / fname)
    if not (local / "adapter_model.safetensors").exists():
        raise RuntimeError(f"adapter for {cid} unavailable locally and on HF ({sub})")
    return local


def _gauge_assert(adapter_dir: Path, cid: str) -> None:
    """Logit readouts valid only when LoRA never touches the unembedding."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    tmods = set(cfg.get("target_modules") or [])
    assert not tmods & {"lm_head", "embed_tokens"}, (cid, sorted(tmods))
    assert not cfg.get("modules_to_save"), (cid, cfg.get("modules_to_save"))


def _cell_payload(phase: str, src: str, byst: str, reads: list[dict], devs: list[dict]) -> dict:
    return {
        "schema_version": "issue532_followup_logp_v1",
        "phase": phase,
        "source_cid": src,
        "bystander_label": byst,
        "n_probes": N_PROBES,
        "per_q": reads,
        "summary": SLOT._summarize(reads),
        "slot_identity_deviations": devs,
    }


def _build_measure_rig(args) -> dict:
    """Shared measure-phase context: tokenizer asserts + frozen-R job builders
    (the #532 rig verbatim: ``_slot_job`` / ``_build_bystander_prompt`` /
    ``_load_parent_cell_R``)."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i460_data import (
        load_class_d_rewrites,
        load_q_test_extended_50,
    )

    sources = resolve_sources(args.sources)
    targets = resolve_targets(args.targets)
    out_root = lowdose_dir(args.smoke, args.results_dir)
    trained_dir = out_root / "per_cell_trained"
    trained_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(SLOT.BASE_MODEL)
    marker_ids = tokenizer.encode(SLOT.MARKER_TEXT, add_special_tokens=False)
    assert marker_ids == [SLOT.MARKER_ID], f"marker encodes to {marker_ids}"
    bare_ids = tokenizer.encode("※", add_special_tokens=False)
    assert len(bare_ids) == 1, f"bare marker encodes to {bare_ids}"
    bare_marker_id = bare_ids[0]
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == SLOT.EOS_ID

    q_test = load_q_test_extended_50()[:N_PROBES]
    class_d_rewrites = load_class_d_rewrites()
    panel = SLOT._instructed_bystander_panel()
    prompt_cache: dict[tuple[str, str], str] = {}

    def prompt_for(b: str, q: str) -> str:
        key = (b, q)
        if key not in prompt_cache:
            prompt_cache[key] = SLOT._build_bystander_prompt(
                b, q, tokenizer, class_d_rewrites, panel
            )
        return prompt_cache[key]

    def cell_jobs(src: str, byst: str) -> list[dict]:
        R_list = SLOT._load_parent_cell_R(src, byst, N_PROBES)
        return [
            SLOT._slot_job(prompt_for(byst, q), R, tokenizer, bare_marker_id)
            for q, R in zip(q_test, R_list, strict=True)
        ]

    def parent_base_cell(src: str, byst: str) -> dict:
        return json.loads((C.PER_CELL_DIR / "per_cell_base" / f"{src}__{byst}.json").read_text())

    gate_source = args.gate_source
    if args.run_gate and sources[0] != gate_source:
        raise ValueError(f"--run-gate batch must lead with the gate source {gate_source}")
    return {
        "sources": sources,
        "targets": targets,
        "out_root": out_root,
        "trained_dir": trained_dir,
        "remeasured_dir": out_root / "per_cell_base_remeasured",
        "gate_file": Path(args.gate_file),
        "gate_source": gate_source,
        "tokenizer": tokenizer,
        "bare_marker_id": bare_marker_id,
        "cell_jobs": cell_jobs,
        "parent_base_cell": parent_base_cell,
    }


def _measure_smoke(args, rig: dict) -> int:
    """CPU-real smoke: build the REAL jobs (tokenizer + prompts + frozen R) and
    run the slot-identity asserts vs the parent base files — the CPU-runnable
    portion of the GPU-bound phase (carve-out item 1) — then exercise the gate
    predicate on synthetic values. No model loads, scratch outputs only."""
    C.phase("p3_measure_smoke")
    n_checked = 0
    for src in rig["sources"]:
        for byst in rig["targets"]:
            jobs = rig["cell_jobs"](src, byst)
            devs = slot_identity_deviations(jobs, rig["parent_base_cell"](src, byst)["per_q"])
            if devs:
                raise AssertionError(
                    f"slot identity mismatch in smoke for {src}__{byst}: {devs[:3]}"
                )
            n_checked += 1
    if args.run_gate:
        assert p2_gate_verdict(8.0, 7.5)["halt"] is False
        assert p2_gate_verdict(0.3, None)["halt"] is True
        assert p2_gate_verdict(24.0, 8.0)["halt"] is True
        assert p2_gate_verdict(10.0, 8.5)["parity_warn"] is True
        C.write_json_atomic(
            rig["gate_file"],
            {"status": "pass", "smoke": True, "note": "predicate exercised on synthetic dG"},
        )
    C.write_json_atomic(
        rig["out_root"] / "measure_smoke.json",
        {
            "sources": rig["sources"],
            "targets": rig["targets"],
            "n_cells_slot_checked": n_checked,
        },
    )
    logger.info("[measure-smoke] %d cells slot-identity-checked; no model loads", n_checked)
    return 0


def _p2_gate_production(rig: dict, src: str, peft_model, base_gate_reads) -> list[tuple[Path, str]]:
    """P2 gate on the diagonal cell (plan §4 P2): HALT iff dG = trained -
    parent-base is outside [2, 18] nats; the in-loop parity gap is a persisted
    WARN (never a blocker). Returns the diagonal cell's upload ops."""
    gate_jobs = rig["cell_jobs"](src, src)
    parent = rig["parent_base_cell"](src, src)
    devs = slot_identity_deviations(gate_jobs, parent["per_q"])
    trained_reads = SLOT._run_slot_batches(
        peft_model,
        rig["tokenizer"],
        gate_jobs,
        rig["bare_marker_id"],
        label=f"P2-trained/{src}->{src}",
    )
    trained_mean = float(sum(r["logp_marker"] for r in trained_reads) / len(trained_reads))
    parent_base_mean = parent["summary"]["mean_logp_marker"]
    remeasured_base_mean = float(
        sum(r["logp_marker"] for r in base_gate_reads) / len(base_gate_reads)
    )
    delta_g = trained_mean - parent_base_mean
    traj_path = rig["out_root"] / "band_trajectories" / f"{src}.json"
    inloop = None
    if traj_path.exists():
        band = json.loads(traj_path.read_text()).get("band_stop_result") or {}
        inloop = band.get("last_delta_nats")
    verdict = p2_gate_verdict(delta_g, inloop)
    base_remeasure_delta = remeasured_base_mean - parent_base_mean
    if abs(base_remeasure_delta) > 1.0:
        logger.warning(
            "[p2] base re-measure differs from parent base by %.3f nats "
            "(surface drift — adjudicate at analysis)",
            base_remeasure_delta,
        )
    payload = {
        "status": "fail" if verdict["halt"] else "pass",
        "cell": f"{src}__{src}",
        **verdict,
        "trained_mean_logp_marker": trained_mean,
        "parent_base_mean_logp_marker": parent_base_mean,
        "remeasured_base_mean_logp_marker": remeasured_base_mean,
        "base_remeasure_delta_nats": base_remeasure_delta,
        "slot_identity_deviations": devs,
        "reproducibility_metadata": C.reproducibility_metadata(
            {"followup": "lowdose-grid-kill-battery", "phase": "P2_gate"}
        ),
    }
    C.write_json_atomic(rig["gate_file"], payload)
    if traj_path.exists():
        sidecar = json.loads(traj_path.read_text())
        sidecar["parity_warn"] = verdict["parity_warn"]
        sidecar["parity_gap_nats"] = verdict["parity_gap_nats"]
        sidecar["offline_delta_g_nats"] = delta_g
        C.write_json_atomic(traj_path, sidecar)
    logger.info(
        "[p2] dG=%.3f window=%s halt=%s parity_warn=%s (inloop=%s)",
        delta_g,
        verdict["window_nats"],
        verdict["halt"],
        verdict["parity_warn"],
        inloop,
    )
    if verdict["halt"]:
        raise RuntimeError(
            f"P2 parity gate HALT: dG={delta_g:.3f} outside {GATE_WINDOW} — "
            "apply-path inspection required (plan v8 §4 P2)"
        )
    # Reuse the gate's trained reads for the diagonal cell file.
    diag_path = rig["trained_dir"] / f"{src}__{src}.json"
    C.write_json_atomic(
        diag_path,
        _cell_payload("A3_lowdose_trained_on_parent_R", src, src, trained_reads, devs),
    )
    return [(diag_path, f"{HF_LOWDOSE_PREFIX}/per_cell_trained/{diag_path.name}")]


def _measure_one_cell(
    rig: dict, peft_model, src: str, byst: str, uploaded: list[tuple[Path, str]]
) -> None:
    """One (src, target) corrected-slot cell: slot-identity assert vs the
    parent base -> trained reads -> per-cell write; on a slot mismatch the
    base side is re-measured with the adapter DISABLED and the deviation
    recorded (plan §8 risk row) — never silently differenced."""
    cell_path = rig["trained_dir"] / f"{src}__{byst}.json"
    hub_dest = f"{HF_LOWDOSE_PREFIX}/per_cell_trained/{cell_path.name}"
    if cell_path.exists():
        if (cell_path, hub_dest) not in uploaded:
            logger.info("[p3] resume skip %s", cell_path.name)
            uploaded.append((cell_path, hub_dest))
        return
    jobs = rig["cell_jobs"](src, byst)
    parent = rig["parent_base_cell"](src, byst)
    devs = slot_identity_deviations(jobs, parent["per_q"])
    reads = SLOT._run_slot_batches(
        peft_model, rig["tokenizer"], jobs, rig["bare_marker_id"], label=f"P3/{src}->{byst}"
    )
    C.write_json_atomic(
        cell_path, _cell_payload("A3_lowdose_trained_on_parent_R", src, byst, reads, devs)
    )
    uploaded.append((cell_path, hub_dest))
    if devs:
        logger.warning(
            "[p3] slot identity deviation %s__%s (%d rows) — re-measuring base side",
            src,
            byst,
            len(devs),
        )
        with peft_model.disable_adapter():
            base_reads = SLOT._run_slot_batches(
                peft_model,
                rig["tokenizer"],
                jobs,
                rig["bare_marker_id"],
                label=f"P3-base/{src}->{byst}",
            )
        rig["remeasured_dir"].mkdir(parents=True, exist_ok=True)
        re_path = rig["remeasured_dir"] / f"{src}__{byst}.json"
        C.write_json_atomic(
            re_path, _cell_payload("A2_lowdose_base_remeasured", src, byst, base_reads, devs)
        )
        uploaded.append((re_path, f"{HF_LOWDOSE_PREFIX}/per_cell_base_remeasured/{re_path.name}"))


def run_measure(args) -> int:
    """P2 gate (``--run-gate``) + P3 trained-side sweep for ``--sources``."""
    rig = _build_measure_rig(args)
    if args.smoke:
        return _measure_smoke(args, rig)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    C.phase("p3_measure_load")
    base = AutoModelForCausalLM.from_pretrained(
        SLOT.BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    base.eval()
    adapter_root = Path(args.adapter_root)

    for src in rig["sources"]:
        adapter_dir = _resolve_adapter_dir(src, adapter_root)
        _gauge_assert(adapter_dir, src)
        uploaded: list[tuple[Path, str]] = []

        run_gate_here = args.run_gate and src == rig["gate_source"]
        base_gate_reads = None
        if run_gate_here:
            C.phase("p2_gate")
            # ONE free no-adapter re-measure of the diagonal cell pins the
            # base surface (plan §4 P2) — run BEFORE the adapter is applied.
            base_gate_reads = SLOT._run_slot_batches(
                base,
                rig["tokenizer"],
                rig["cell_jobs"](src, src),
                rig["bare_marker_id"],
                label=f"P2-base/{src}->{src}",
            )

        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        peft_model.eval()

        if run_gate_here:
            uploaded += _p2_gate_production(rig, src, peft_model, base_gate_reads)

        C.phase(f"p3_measure_{src}")
        for byst in rig["targets"]:
            _measure_one_cell(rig, peft_model, src, byst, uploaded)

        # Per-source upload the moment the source's cells complete (#664).
        for extra_rel in (
            f"band_trajectories/{src}.json",
            f"band_trajectories/{src}_bracket.json",
            f"train_summaries/{src}.json",
        ):
            p = rig["out_root"] / extra_rel
            if p.exists():
                uploaded.append((p, f"{HF_LOWDOSE_PREFIX}/{extra_rel}"))
        if run_gate_here and rig["gate_file"].exists():
            uploaded.append((rig["gate_file"], f"{HF_LOWDOSE_PREFIX}/{rig['gate_file'].name}"))
        if not args.skip_upload:
            upload_files(uploaded, f"issue1332 lowdose: source {src} cells + sidecars")

        base = peft_model.unload()
        del peft_model
        torch.cuda.empty_cache()
        logger.info("[p3] source %s done (%d files staged/uploaded)", src, len(uploaded))
    return 0


# ── shard runner (P1 train -> deferred P3 measure, work-conserving) ───────────


def _echo_log_tail(log_path: Path, label: str, n: int = 120) -> None:
    """Echo an inner log's tail into the MAIN log on child failure (#1333)."""
    try:
        lines = log_path.read_text(errors="replace").split("\n")
    except OSError as e:
        logger.warning("[%s] inner log unreadable: %s", label, e)
        return
    logger.error("[%s] inner log tail (%s):\n%s", label, log_path, "\n".join(lines[-n:]))


def _run_child(cmd: list[str], log_path: Path, label: str) -> None:
    logger.info("[%s] %s (log %s)", label, " ".join(cmd), log_path)
    with open(log_path, "a") as fh:
        proc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ}, check=False
        )
    if proc.returncode != 0:
        _echo_log_tail(log_path, label)
        raise RuntimeError(f"{label} exited rc={proc.returncode} (log: {log_path})")


def run_shard(args) -> int:
    """One shard: train each source (subprocess), gate on source 1, measure
    trained sources in deferred batches once the gate file reads pass."""
    sources = resolve_sources(args.sources)
    gate_file = Path(args.gate_file)
    shard_tag = f"shard{args.shard_index}"
    ld = logs_dir()

    common_flags: list[str] = []
    if args.smoke:
        common_flags.append("--smoke")
    if args.results_dir:
        common_flags += ["--results-dir", args.results_dir]

    def measure_batch(batch: list[str], run_gate: bool) -> None:
        if not batch:
            return
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--measure",
            "--sources",
            ",".join(batch),
            "--targets",
            args.targets,
            "--gate-file",
            str(gate_file),
            "--gate-source",
            args.gate_source,
            "--adapter-root",
            args.adapter_root,
            *common_flags,
        ]
        if run_gate:
            cmd.append("--run-gate")
        if args.skip_upload:
            cmd.append("--skip-upload")
        _run_child(cmd, ld / f"issue-1332-lowdose-{shard_tag}-measure.log", f"{shard_tag}/measure")

    pending: list[str] = []
    for src in sources:
        C.phase(f"p1_train_{src}")
        train_cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "issue1332_lowdose_train.py"),
            "--cond",
            src,
            "--adapter-root",
            args.adapter_root,
            *common_flags,
        ]
        if args.smoke:
            train_cmd.append("--skip-upload-verify")
        _run_child(
            train_cmd, ld / f"issue-1332-lowdose-{shard_tag}-train.log", f"{shard_tag}/train/{src}"
        )
        pending.append(src)

        if src == args.gate_source:
            measure_batch(pending, run_gate=True)
            pending = []
        elif _gate_status(gate_file) == "pass":
            measure_batch(pending, run_gate=False)
            pending = []

    if pending:
        _wait_gate(gate_file, args.gate_timeout)
        measure_batch(pending, run_gate=False)
    logger.info("[%s] done (%d sources)", shard_tag, len(sources))
    return 0


# ── dispatcher (--full / --smoke) ─────────────────────────────────────────────


def _final_verify_and_manifest(args, sources: list[str], targets: list[str]) -> dict:
    """Grid completeness + install record + ONE prefix-scoped upload verify."""
    out_root = lowdose_dir(args.smoke, args.results_dir)
    trained_dir = out_root / "per_cell_trained"
    missing_cells = [
        f"{s}__{t}.json"
        for s in sources
        for t in targets
        if not (trained_dir / f"{s}__{t}.json").exists()
    ]
    if missing_cells and not args.smoke:
        raise RuntimeError(
            f"grid incomplete: {len(missing_cells)} cells missing: {missing_cells[:5]}"
        )

    install_record = {}
    for s in sources:
        diag = trained_dir / f"{s}__{s}.json"
        if not diag.exists():
            continue
        trained_mean = json.loads(diag.read_text())["summary"]["mean_logp_marker"]
        parent_base = json.loads((C.PER_CELL_DIR / "per_cell_base" / f"{s}__{s}.json").read_text())[
            "summary"
        ]["mean_logp_marker"]
        traj = out_root / "band_trajectories" / f"{s}.json"
        band = json.loads(traj.read_text()).get("band_stop_result") if traj.exists() else None
        install_record[s] = {
            "diag_delta_g_nats": trained_mean - parent_base,
            "band_stop_result": band,
        }

    manifest = {
        "phase": "final",
        "sources": sources,
        "targets": targets,
        "n_cells": len(sources) * len(targets) - len(missing_cells),
        "missing_cells": missing_cells,
        "install_record": install_record,
        "gate": json.loads(Path(args_gate_file(args)).read_text())
        if Path(args_gate_file(args)).exists()
        else None,
        "reproducibility_metadata": C.reproducibility_metadata(
            {"followup": "lowdose-grid-kill-battery", "smoke": args.smoke}
        ),
    }
    C.write_json_atomic(out_root / "lowdose_manifest.json", manifest)

    if not args.smoke and not args.skip_upload:
        ops = [
            (out_root / "p0_manifest.json", f"{HF_LOWDOSE_PREFIX}/p0_manifest.json"),
            (out_root / "lowdose_manifest.json", f"{HF_LOWDOSE_PREFIX}/lowdose_manifest.json"),
        ]
        upload_files(ops, "issue1332 lowdose: run manifests")
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate.hub import verify_repo_paths_uploaded

        expected = [
            f"{HF_LOWDOSE_PREFIX}/per_cell_trained/{s}__{t}.json" for s in sources for t in targets
        ]
        expected += [f"{HF_LOWDOSE_PREFIX}/band_trajectories/{s}.json" for s in sources]
        expected += [
            f"{HF_LOWDOSE_PREFIX}/p0_manifest.json",
            f"{HF_LOWDOSE_PREFIX}/lowdose_manifest.json",
        ]
        missing = verify_repo_paths_uploaded(
            HfApi(),
            C.HF_DATA_REPO,
            expected,
            path_in_repo=HF_LOWDOSE_PREFIX,
            repo_type="dataset",
        )
        if missing:
            raise RuntimeError(f"upload verify: {len(missing)} paths missing on Hub: {missing[:5]}")
        logger.info("[verify] %d Hub paths verified under %s", len(expected), HF_LOWDOSE_PREFIX)
    return manifest


def args_gate_file(args) -> str:
    return str(lowdose_dir(args.smoke, args.results_dir) / "gate_status.json")


def run_dispatch(args) -> int:
    """Top-level: P0 -> shard fan-out (CVD pinned per shard) -> verify/manifest."""
    if args.smoke and args.sources == "all":
        args.sources = "A1"
        args.targets = "A1,instr_explicit_1"
        logger.info("[smoke] narrowed to sources=%s targets=%s", args.sources, args.targets)
    sources = resolve_sources(args.sources)
    targets = resolve_targets(args.targets)
    gate_source = sources[0]

    p0(args, sources)

    gpus = detect_gpus()
    if not gpus and not args.smoke:
        raise RuntimeError("no GPUs visible via nvidia-smi — refusing to dispatch training")
    n_shards = len(gpus) if args.max_shards == 0 else min(args.max_shards, max(1, len(gpus)))
    shards = split_shards(sources, max(1, n_shards))
    gate_file = Path(args_gate_file(args))
    if gate_file.exists():
        gate_file.unlink()  # fresh gate per dispatch (deterministic, ~minutes to re-run)

    C.phase("p1_dispatch")
    procs: list[tuple[subprocess.Popen, Path, str]] = []
    for i, shard_sources in enumerate(shards):
        env = {**os.environ}
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--shard",
            "--shard-index",
            str(i),
            "--sources",
            ",".join(shard_sources),
            "--targets",
            ",".join(targets),
            "--gate-file",
            str(gate_file),
            "--gate-source",
            gate_source,
            "--gate-timeout",
            str(args.gate_timeout),
            "--adapter-root",
            args.adapter_root,
        ]
        if args.smoke:
            cmd.append("--smoke")
        if args.results_dir:
            cmd += ["--results-dir", args.results_dir]
        if args.skip_upload:
            cmd.append("--skip-upload")
        log_path = logs_dir() / f"issue-1332-lowdose-shard{i}.log"
        logger.info(
            "[dispatch] shard %d gpu=%s sources=%s (log %s)",
            i,
            env.get("CUDA_VISIBLE_DEVICES", "none"),
            shard_sources,
            log_path,
        )
        # O_APPEND fd handed to the child; closed parent-side right after
        # spawn (the child keeps its own duplicate) — no long-lived parent fh.
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            proc = subprocess.Popen(cmd, stdout=fd, stderr=subprocess.STDOUT, env=env)
        finally:
            os.close(fd)
        procs.append((proc, log_path, f"shard{i}"))

    failed = []
    for proc, log_path, label in procs:
        rc = proc.wait()
        if rc != 0:
            _echo_log_tail(log_path, label)
            failed.append((label, rc))
    if failed:
        C.write_sentinel(
            "epm:failure",
            json.dumps(
                {
                    "failure_class": "code",
                    "reason": f"lowdose shards failed: {failed}",
                    "assert_tag": "i1332-lowdose-shard-fail",
                }
            ),
        )
        return 1

    C.phase("p3_finalize")
    manifest = _final_verify_and_manifest(args, sources, targets)

    if not args.smoke:
        # Between-phase cache hygiene (CLAUDE.md incremental-clean contract);
        # production-only — the VM smoke must not sweep the live issue caches.
        subprocess.run(
            [
                sys.executable,
                str(C.PROJECT_ROOT / "scripts" / "clean_experiment_downloads.py"),
                "1332",
                "--incremental",
                "--apply",
            ],
            env={**os.environ},
            check=False,
            cwd=C.PROJECT_ROOT,
        )

    note = json.dumps(
        {
            "followup": "lowdose-grid-kill-battery",
            "n_cells": manifest["n_cells"],
            "sources": len(sources),
            "targets": len(targets),
            "gate": (manifest.get("gate") or {}).get("status"),
            "hf_prefix": HF_LOWDOSE_PREFIX,
        }
    )
    C.write_sentinel("epm:smoke-result" if args.smoke else "epm:results", note)
    C.phase("done")
    return 0


def main() -> int:
    """Lowdose pod driver: --full/--smoke dispatch, --shard/--measure internal."""
    ap = argparse.ArgumentParser(description="Issue #1332 lowdose GPU driver")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="production dispatch")
    mode.add_argument("--smoke", action="store_true", help="CPU smoke; scratch roots")
    mode_int = ap.add_mutually_exclusive_group()
    mode_int.add_argument("--shard", action="store_true", help="internal shard runner")
    mode_int.add_argument("--measure", action="store_true", help="internal measure phase")
    ap.add_argument("--sources", default="all")
    ap.add_argument("--targets", default="all")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--gate-file", default=None)
    ap.add_argument("--gate-source", default=None)
    ap.add_argument("--gate-timeout", type=float, default=7200.0)
    ap.add_argument("--run-gate", action="store_true")
    ap.add_argument("--max-shards", type=int, default=0, help="0 = one shard per visible GPU")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument(
        "--adapter-root",
        default=(
            "/workspace/adapters/i1332_lowdose"
            if os.path.isdir("/workspace")
            else str(C.data_root(False) / "adapters_lowdose")
        ),
    )
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # The #532 rig resolves its committed inputs (eval_results/issue_532/...)
    # RELATIVE to cwd — pin cwd to the repo root deterministically.
    os.chdir(C.PROJECT_ROOT)

    if args.shard or args.measure:
        if not args.gate_file or not args.gate_source:
            raise ValueError("--shard/--measure require --gate-file and --gate-source")
        if not (args.smoke or args.full):
            # internal invocations carry --smoke explicitly; bare = production
            pass
        return run_shard(args) if args.shard else run_measure(args)

    if not (args.full or args.smoke):
        raise SystemExit("pick one of --full / --smoke")
    try:
        return run_dispatch(args)
    except Exception as e:  # fail-loud + sentinel, mirroring issue1332_gpu_phase
        logger.exception("lowdose dispatch failed")
        C.write_sentinel(
            "epm:failure",
            json.dumps(
                {
                    "failure_class": "code",
                    "reason": f"lowdose dispatch exception: {type(e).__name__}: {e}",
                    "assert_tag": "i1332-lowdose-dispatch-exception",
                }
            ),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
