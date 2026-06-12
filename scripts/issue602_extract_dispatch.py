#!/usr/bin/env python3
"""#602 pod dispatcher — Phase 1a-1d (base gens -> shifts -> estimators -> upload).

UNIFIED smoke/sweep architecture (PASS_UNIFIED contract): the smoke IS the
sweep with a cell subset — same dispatcher, same subprocess shape, same env
injection, same logging surface, same teardown. EVERY phase's worklist
derives from the SAME ``--cells`` subset:

- ``i474_check`` (0b): assumption-8 prompt-reconstruction gate (runs FIRST
  when a loc474 cell is active — strict on the production model, so a
  reconstruction drift aborts before any GPU spend);
- ``generate`` (1a): vLLM greedy base generations for exactly the panel
  contexts of the ACTIVE cells' families + the ACTIVE units' E1/E2/E3
  contrast prompts;
- ``extract`` (1b): realized-shift extraction subprocesses for the ACTIVE
  cells only (CUDA_VISIBLE_DEVICES cell-sharding, checkpoint-per-cell);
- ``estimators`` (1c): estimator-read subprocesses for the units derived
  FROM the active cells; ``anchor`` runs when a marker519 cell is active;
- ``upload`` (1d): uploads whatever the active subset produced, then
  verifies via ``list_repo_files``.

``--smoke`` only re-parameterizes (default cells = one #518 refusal
adapter + one marker519 cell, panel truncated 3x3, E1 rows capped at 4,
probes at 3, anchor pool at 4) — no separate code path.

vLLM teardown gotcha: the generate phase runs as a SUBPROCESS of this
dispatcher so the engine's worker processes die with it before any HF
model loads (the in-process destroy path leaks workers that re-allocate
freed GPU memory — #399/#397 orphan-PID incidents).

Pod-side contract: emits ``[phase=<name>]`` lines; the TERMINAL
``[phase=done]`` line appears exactly once, after the results sentinel
(`/workspace/logs/issue-602-epm_results-<epoch>.json`, schema keys
``sentinel_schema_version``/``kind``/``version``) is written. NEVER
shells out to scripts/task.py (pod-side ban).
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
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.analysis import i602_bakeoff as bk  # noqa: E402

logger = logging.getLogger("issue602_dispatch")

SENTINEL_SCHEMA_VERSION = 1
SMOKE_DEFAULT_CELLS = ("refusal518__assistant__s42", "marker519__medical_doctor__s42")
SMOKE_N_CONTEXTS = 3
SMOKE_N_QUESTIONS = 3
SMOKE_LIMIT_ROWS = 4
SMOKE_LIMIT_PROBES = 3
SMOKE_ANCHOR_POOL = 4


def _sh(cmd: list[str], log_path: Path, gpu: str | None = None) -> subprocess.Popen:
    """Spawn one worker subprocess with explicit env passthrough + CVD pin."""
    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("a")
    logger.info("spawn [gpu=%s] %s (log %s)", gpu, " ".join(cmd[:6]) + " ...", log_path.name)
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)


def _run_parallel(jobs: list[tuple[list[str], Path]], gpus: list[str]) -> None:
    """Round-robin jobs over GPUs, fail-loud on any non-zero exit."""
    pending = list(jobs)
    running: list[tuple[subprocess.Popen, Path, str]] = []
    failures: list[str] = []
    while pending or running:
        while pending and len(running) < len(gpus):
            used = {g for _, _, g in running}
            free = [g for g in gpus if g not in used]
            if not free:
                break
            cmd, log_path = pending.pop(0)
            gpu = free[0]
            running.append((_sh(cmd, log_path, gpu), log_path, gpu))
        time.sleep(5)
        still = []
        for proc, log_path, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, log_path, gpu))
            elif rc != 0:
                failures.append(f"{log_path.name} rc={rc}")
                logger.error("worker FAILED rc=%d — tail of %s:", rc, log_path)
                tail = log_path.read_text().splitlines()[-25:]
                for line in tail:
                    logger.error("  | %s", line)
        running = still
    if failures:
        raise RuntimeError(f"{len(failures)} worker(s) failed: {failures}")


# ---------------------------------------------------------------------------
# Worklist derivation (single source: the --cells subset)
# ---------------------------------------------------------------------------
def resolve_cells(spec: list[str] | None, smoke: bool) -> list[dict[str, Any]]:
    """Resolve the active cell subset (ALL phases derive from this)."""
    all_cells = {c["cell_id"]: c for c in bk.extraction_cells()}
    if spec:
        missing = [s for s in spec if s not in all_cells]
        if missing:
            raise SystemExit(f"unknown cell ids: {missing}; valid: {sorted(all_cells)}")
        return [all_cells[s] for s in spec]
    if smoke:
        return [all_cells[s] for s in SMOKE_DEFAULT_CELLS]
    return list(all_cells.values())


def active_units(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Estimator units = unique (family, source) pairs OF the active cells."""
    seen = {(c["family"], c["source"]) for c in cells}
    return [u for u in bk.estimator_units() if (u["family"], u["source"]) in seen]


def _family_panel(family: str, source: str, smoke: bool) -> tuple[dict[str, str | None], list[str]]:
    """Panel contexts + questions for one family, smoke-truncated but ALWAYS
    retaining the cell's source context (w_src must exist)."""
    contexts = bk.family_contexts(family, root=REPO)
    _, questions = bk.load_shared_panel(REPO)
    if smoke:
        keep = [source] if source in contexts else []
        keep += [c for c in contexts if c not in keep][: SMOKE_N_CONTEXTS - len(keep)]
        contexts = {c: contexts[c] for c in keep}
        questions = questions[:SMOKE_N_QUESTIONS]
    return contexts, questions


def _i406_cids(cell: dict[str, Any], smoke: bool) -> list[str]:
    """The i406 transformation contexts for a loc474 cell (smoke: source only)."""
    if cell["family"] != "loc474":
        return []
    return [cell["source"]] if smoke else list(bk.LOC474_CONTEXTS)


# ---------------------------------------------------------------------------
# Phase 0b — #474 prompt-reconstruction cross-check (assumption-8 gate)
# ---------------------------------------------------------------------------
def phase_i474_crosscheck(args: argparse.Namespace, cells: list[dict[str, Any]], gpus: list[str]):
    """Assumption-8 gate: reproduce the stored #406 base cosines under the
    reconstructed i406 prompts (tolerance 3e-3) BEFORE any sweep spend.

    Strict (nonzero-exit-on-mismatch) iff the production model is in play;
    the CPU-stub smoke records ``production_model: false`` instead (stored
    values can never reproduce on a stub) and Phase 2's production
    preflight rejects such a file. A squatting non-production artifact is
    re-run rather than skip-if-exists'd (stub-contamination guard).
    """
    loc = [c for c in cells if c["family"] == "loc474"]
    if not loc:
        logger.info("[phase=i474_check] no loc474 cells active — skip")
        return
    out_path = bk.eval_dir(REPO) / "work" / "i474_crosscheck.json"
    if out_path.exists():
        prior = json.loads(out_path.read_text())
        if prior.get("production_model") and prior.get("ok"):
            logger.info("[phase=i474_check] %s present (production, ok) — skip", out_path.name)
            return
        logger.warning(
            "[phase=i474_check] stale/non-production artifact at %s — re-running", out_path
        )
        out_path.unlink()
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "issue602_i474_crosscheck.py"),
        "--model-id",
        args.model_id,
        "--out",
        str(out_path),
        "--contexts",
        *bk.LOC474_CONTEXTS,
    ]
    if args.model_id == bk.BASE_MODEL_ID:
        cmd.append("--strict")
    elif args.smoke:
        cmd += ["--n-questions", str(SMOKE_N_QUESTIONS)]
    _run_parallel([(cmd, bk.eval_dir(REPO) / "logs" / "i474_crosscheck.log")], gpus[:1])
    logger.info("[phase=i474_check] complete (%s)", out_path.name)


# ---------------------------------------------------------------------------
# Phase 1a — base generations (vLLM, greedy; run as a subprocess)
# ---------------------------------------------------------------------------
def build_generation_worklist(
    cells: list[dict[str, Any]], units: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """{out_name: {key: chat_messages}} for every generation the run needs."""
    from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID

    work: dict[str, dict[str, list[dict[str, str]]]] = {}

    def _msgs(system: str | None, user: str) -> list[dict[str, str]]:
        m: list[dict[str, str]] = []
        if system is not None:
            m.append({"role": "system", "content": system})
        m.append({"role": "user", "content": user})
        return m

    families = sorted({c["family"] for c in cells})
    for family in families:
        src = next(c["source"] for c in cells if c["family"] == family)
        contexts, questions = _family_panel(family, src, args.smoke)
        work[f"panel__{family}"] = {
            f"{ctx}␟{q}": _msgs(prompt, q) for ctx, prompt in contexts.items() for q in questions
        }
    # i406 transformation contexts (loc474): raw prompt TEXT, not messages
    loc_cells = [c for c in cells if c["family"] == "loc474"]
    if loc_cells:
        from explore_persona_space.experiments.i460_data import (
            load_class_d_rewrites,
            load_q_test_extended_50,
        )

        q50 = load_q_test_extended_50()
        if args.smoke:
            q50 = q50[:SMOKE_N_QUESTIONS]
        cids = sorted({cid for c in loc_cells for cid in _i406_cids(c, args.smoke)})
        rewrites = (
            load_class_d_rewrites() if any(CONDITIONS_BY_ID[c].cls == "D" for c in cids) else None
        )
        entry: dict[str, Any] = {}
        for cid in cids:
            for q in q50:
                entry[f"{cid}␟{q}"] = {
                    "raw_prompt": True,
                    "cid": cid,
                    "question": q,
                }
        work["panel__loc474_i406"] = entry
        work["_i406_meta"] = {
            "rewrites_loaded": rewrites is not None,
            "cids": cids,
            "n_q": len(q50),
        }  # type: ignore[assignment]

    for unit in units:
        family, source = unit["family"], unit["source"]
        for mix_label in unit["e1_mix_labels"]:
            rows, _ = bk.e1_rows(family, source, mix_label, root=REPO)
            if args.smoke:
                rows = rows[:SMOKE_LIMIT_ROWS]
            work[f"e1__{family}__{source}__{mix_label}"] = {
                r["row_key"]: r["prompt_messages"] for r in rows
            }
        probes = bk.e2_probes(family, root=REPO)
        if args.smoke:
            probes = probes[:SMOKE_LIMIT_PROBES]
        ks = [int(k) for k in args.e2_ks]
        demo_sets = bk.e2_demo_sets(family, source, root=REPO, ks=ks)
        work[f"e2zero__{family}__{source}"] = {
            p: bk.build_e2_messages(family, source, [], p) for p in probes
        }
        for k in ks:
            work[f"e2K{k}__{family}__{source}"] = {
                f"r{r_idx}␟{p}".replace("␟", "__", 1): bk.build_e2_messages(
                    family, source, demos, p
                )
                for r_idx, demos in enumerate(demo_sets[k])
                for p in probes
            }
        work[f"e3desc__{family}__{source}"] = {
            p: bk.build_e3_messages(family, source, p, True) for p in probes
        }
        work[f"e3nodesc__{family}__{source}"] = {
            p: bk.build_e3_messages(family, source, p, False) for p in probes
        }
    return work


def phase_generate(args: argparse.Namespace) -> None:
    """1a: one vLLM engine, greedy, deduped prompts; per-scope JSON outputs.

    Runs inside a dedicated subprocess (see ``cmd_generate`` dispatch) so
    vLLM worker processes are reaped with the process exit.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    cells = resolve_cells(args.cells, args.smoke)
    units = active_units(cells)
    out_dir = bk.eval_dir(REPO) / "base_generations"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = build_generation_worklist(cells, units, args)
    meta = work.pop("_i406_meta", None)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    from explore_persona_space.experiments.i406_conditions import (
        CONDITIONS_BY_ID,
        build_prompt_for_condition,
    )

    rewrites = None
    if meta and meta["rewrites_loaded"]:
        from explore_persona_space.experiments.i460_data import load_class_d_rewrites

        rewrites = load_class_d_rewrites()

    # resolve every entry to prompt TEXT; dedupe identical prompts
    prompt_text: dict[tuple[str, str], str] = {}
    for name, entries in work.items():
        done_path = out_dir / f"{name}.json"
        if done_path.exists():
            logger.info("[phase=generate] %s already present — skip", name)
            continue
        for key, msgs in entries.items():
            if isinstance(msgs, dict) and msgs.get("raw_prompt"):
                cond = CONDITIONS_BY_ID[msgs["cid"]]
                text = build_prompt_for_condition(
                    cond, msgs["question"], tokenizer, class_d_rewrites=rewrites
                )
            else:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            prompt_text[(name, key)] = text
    unique = sorted(set(prompt_text.values()))
    logger.info(
        "[phase=generate] %d scopes, %d prompts (%d unique)",
        len(work),
        len(prompt_text),
        len(unique),
    )
    if unique:
        llm = LLM(
            model=args.model_id,
            dtype="bfloat16",
            gpu_memory_utilization=0.92,
            max_model_len=8192,
        )
        params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        outs = llm.generate(unique, params)
        resp_by_prompt = {unique[i]: outs[i].outputs[0].text for i in range(len(unique))}
    else:
        resp_by_prompt = {}

    # checkpoint-per-scope writes
    by_scope: dict[str, dict[str, str]] = {}
    for (name, key), text in prompt_text.items():
        by_scope.setdefault(name, {})[key] = resp_by_prompt[text]
    for name, entries in by_scope.items():
        if name.startswith("panel__"):
            nested: dict[str, dict[str, str]] = {}
            for key, resp in entries.items():
                ctx, q = key.split("␟", 1)
                nested.setdefault(ctx, {})[q] = resp
            (out_dir / f"{name}.json").write_text(json.dumps(nested, indent=1))
        else:
            (out_dir / f"{name}.json").write_text(json.dumps(entries, indent=1))
        logger.info("[phase=generate] wrote %s.json (%d entries)", name, len(entries))
    logger.info("[phase=generate] complete")


# ---------------------------------------------------------------------------
# Phase 1b — realized-shift extraction (per-cell subprocesses)
# ---------------------------------------------------------------------------
def phase_extract(args: argparse.Namespace, cells: list[dict[str, Any]], gpus: list[str]) -> None:
    """1b: per-cell activation_shift subprocesses, checkpoint-per-cell."""
    ev = bk.eval_dir(REPO)
    out_dir = ev / "shifts"
    gen_dir = ev / "base_generations"
    work_dir = ev / "work"
    adapters_dir = ev / "adapters"
    work_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[list[str], Path]] = []
    expected_outputs: list[Path] = []
    for cell in cells:
        out_path = out_dir / f"{cell['cell_id']}.pt"
        expected_outputs.append(out_path)
        if out_path.exists():
            logger.info("[phase=extract] %s already extracted — skip", cell["cell_id"])
            continue
        if args.adapter_override:
            logger.warning(
                "[phase=extract] ADAPTER OVERRIDE active (%s) — smoke/stub runs only",
                args.adapter_override,
            )
            adapter_dir = Path(args.adapter_override)
        else:
            adapter_dir = bk.download_adapter(
                cell["adapter_repo"], cell["adapter_prefix"], adapters_dir
            )
        contexts, questions = _family_panel(cell["family"], cell["source"], args.smoke)
        personas_json = work_dir / f"{cell['cell_id']}__personas.json"
        questions_json = work_dir / f"{cell['cell_id']}__questions.json"
        personas_json.write_text(json.dumps(contexts, indent=1))
        questions_json.write_text(json.dumps(questions, indent=1))
        # merged base responses: family panel (+ i406 contexts for loc474)
        base_resp = json.loads((gen_dir / f"panel__{cell['family']}.json").read_text())
        i406 = _i406_cids(cell, args.smoke)
        if i406:
            i406_resp = json.loads((gen_dir / "panel__loc474_i406.json").read_text())
            for cid in i406:
                if cid not in i406_resp:
                    raise KeyError(f"i406 generations missing context {cid}")
                base_resp[cid] = i406_resp[cid]
        base_resp_json = work_dir / f"{cell['cell_id']}__base_responses.json"
        base_resp_json.write_text(json.dumps(base_resp, indent=1))
        cmd = [
            sys.executable,
            "-m",
            "explore_persona_space.analysis.activation_shift",
            "--arm",
            cell["arm"],
            "--seed",
            str(cell["seed"]),
            "--family",
            cell["family"],
            "--variant",
            "base",
            "--layers",
            *[str(ly) for ly in bk.LAYERS],
            "--primary-layer",
            str(bk.PRIMARY_LAYER),
            "--base-model-id",
            args.model_id,
            "--adapter-path",
            str(adapter_dir),
            "--personas-json",
            str(personas_json),
            "--questions-json",
            str(questions_json),
            "--base-responses-json",
            str(base_resp_json),
            "--out",
            str(out_path),
        ]
        if i406:
            cmd += ["--i406-contexts", *i406]
            if args.smoke:
                cmd += ["--i406-n-questions", str(SMOKE_N_QUESTIONS)]
        jobs.append((cmd, ev / "logs" / f"extract_{cell['cell_id']}.log"))
    logger.info(
        "[phase=extract] %d cells to extract (%d skipped)", len(jobs), len(cells) - len(jobs)
    )
    if jobs:
        _run_parallel(jobs, gpus)
    for out_path in expected_outputs:
        _assert_payload_schema(out_path)
    logger.info("[phase=extract] complete")


def _assert_payload_schema(out_path: Path) -> None:
    """M3a regression guard (runs on EVERY extracted cell, smoke AND sweep):
    a base-variant payload must carry the mean-response keys
    ``delta_v_mean_resp``, ``delta_v_mean_resp_per_q``, and
    ``delta_v_mean_resp_l{L}`` for every non-primary captured layer —
    the pre-registered primary DV is L14/mean-response and is otherwise
    structurally missing (plan §4.4 / reconciler fix M3a)."""
    import torch

    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    assert payload["manifest"]["variant"] == "base", payload["manifest"]["variant"]
    for ctx, entry in payload["shifts"].items():
        missing = [
            k
            for k in ("delta_v", "delta_v_per_q", "delta_v_mean_resp", "delta_v_mean_resp_per_q")
            if k not in entry
        ]
        for ly in payload["manifest"]["layers"]:
            if ly != payload["manifest"]["layer"]:
                missing += [
                    k for k in (f"delta_v_l{ly}", f"delta_v_mean_resp_l{ly}") if k not in entry
                ]
        if missing:
            raise AssertionError(
                f"{out_path.name}: context {ctx!r} missing keys {missing} — "
                "M3a base-variant mean-resp extension regressed"
            )
    logger.info(
        "[phase=extract] schema OK (%s: %d contexts, M3a keys present)",
        out_path.name,
        len(payload["shifts"]),
    )


# ---------------------------------------------------------------------------
# Phase 1c — estimator reads + anchor (per-unit subprocesses)
# ---------------------------------------------------------------------------
def phase_estimators(
    args: argparse.Namespace, cells: list[dict[str, Any]], gpus: list[str]
) -> None:
    """1c: per-unit estimator-read subprocesses + the anchor_521 cell."""
    ev = bk.eval_dir(REPO)
    out_dir = ev / "estimator_reads"
    gen_dir = ev / "base_generations"
    units = active_units(cells)
    jobs: list[tuple[list[str], Path]] = []
    for unit in units:
        out_path = out_dir / f"{unit['family']}__{unit['source']}.pt"
        if out_path.exists():
            logger.info("[phase=estimators] %s/%s done — skip", unit["family"], unit["source"])
            continue
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "issue602_estimator_reads.py"),
            "--family",
            unit["family"],
            "--source",
            unit["source"],
            "--model-id",
            args.model_id,
            "--layers",
            *[str(ly) for ly in bk.LAYERS],
            "--base-generations-dir",
            str(gen_dir),
            "--out",
            str(out_path),
            "--e2-ks",
            *args.e2_ks,
        ]
        if args.smoke:
            cmd += [
                "--limit-rows",
                str(SMOKE_LIMIT_ROWS),
                "--limit-probes",
                str(SMOKE_LIMIT_PROBES),
            ]
        jobs.append((cmd, ev / "logs" / f"estimators_{unit['family']}__{unit['source']}.log"))
    # anchor_521 — exact #521 recipe reproduction (when marker519 is active)
    anchor_out = out_dir / "anchor_521.pt"
    if any(c["family"] == "marker519" for c in cells) and not anchor_out.exists():
        manifest = bk.load_marker_steering_manifest(REPO)
        pool = bk.load_anchor_pool(REPO)
        pool_path = ev / "work" / "anchor_pool.json"
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        if args.smoke:
            pool = pool[:SMOKE_ANCHOR_POOL]
        pool_path.write_text(json.dumps(pool, indent=1))
        jobs.append(
            (
                [
                    sys.executable,
                    "-m",
                    "explore_persona_space.analysis.steering_vectors",
                    "--behavior",
                    "marker",
                    "--positive-system-prompt",
                    manifest["positive_system_prompt"],
                    "--negative-system-prompt",
                    manifest["negative_system_prompt"],
                    "--questions-json",
                    str(pool_path),
                    "--base-model-id",
                    args.model_id,
                    "--layer",
                    str(bk.PRIMARY_LAYER),
                    "--out",
                    str(anchor_out),
                    "--no-judge-filter",
                ],
                ev / "logs" / "anchor_521.log",
            )
        )
    logger.info("[phase=estimators] %d unit jobs", len(jobs))
    if jobs:
        _run_parallel(jobs, gpus)
    logger.info("[phase=estimators] complete")


# ---------------------------------------------------------------------------
# Phase 1d — upload + verify
# ---------------------------------------------------------------------------
def phase_upload(args: argparse.Namespace) -> dict[str, Any]:
    """1d: upload tensors + manifests + raw base generations, verify listing.

    Canonical target: DATA_REPO/issue602_estimator_bakeoff/...; on the
    account-wide LFS quota 403 the pre-registered fallback is the PRIVATE
    data repo, same layout (named deviation, precedent #551).
    """
    from huggingface_hub import list_repo_files

    from explore_persona_space.orchestrate.hub import _upload

    ev = bk.eval_dir(REPO)
    bucket = args.hub_bucket
    to_upload: list[tuple[Path, str]] = []
    for sub, repo_sub in (
        ("shifts", "analysis_tensors/shifts"),
        ("estimator_reads", "analysis_tensors/estimator_reads"),
        ("base_generations", "raw_completions/base_generations"),
    ):
        d = ev / sub
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix in (".pt", ".json"):
                to_upload.append((p, f"{bucket}/{repo_sub}/{p.name}"))
    crosscheck = ev / "work" / "i474_crosscheck.json"
    if crosscheck.exists():
        to_upload.append((crosscheck, f"{bucket}/work/{crosscheck.name}"))
    if not to_upload:
        raise RuntimeError("nothing to upload — extraction/estimator phases produced no files")

    repo_used = bk.DATA_REPO
    deviation = None
    uploaded: list[tuple[str, str]] = []  # (destination repo, path_in_repo) PER FILE
    for p, path_in_repo in to_upload:
        try:
            _upload(p, repo_used, "dataset", path_in_repo, upload_as_file=True)
        except Exception as e:  # quota-403 fallback (pre-registered deviation)
            msg = str(e)
            if "403" in msg or "storage" in msg.lower():
                logger.warning(
                    "[phase=upload] quota-403 on %s — falling back to PRIVATE repo "
                    "(pre-registered deviation, precedent #551)",
                    repo_used,
                )
                repo_used = bk.PRIVATE_DATA_REPO
                deviation = "public-repo LFS quota 403 -> private data repo (same layout)"
                _upload(p, repo_used, "dataset", path_in_repo, upload_as_file=True)
            else:
                raise
        uploaded.append((repo_used, path_in_repo))
    # verify PER DESTINATION REPO: the quota-403 fallback can fire mid-stream,
    # leaving earlier files on the public repo and later ones on the private —
    # a single final-repo listing would false-FAIL the pre-fallback files.
    by_repo: dict[str, list[str]] = {}
    for repo, path_in_repo in uploaded:
        by_repo.setdefault(repo, []).append(path_in_repo)
    missing: list[str] = []
    for repo, paths in by_repo.items():
        listing = set(list_repo_files(repo, repo_type="dataset"))
        missing += [f"{repo}::{u}" for u in paths if u not in listing]
    if missing:
        raise RuntimeError(f"upload verification FAILED — missing: {missing[:10]}")
    logger.info(
        "[phase=upload] %d files verified (%s)",
        len(uploaded),
        ", ".join(f"{r}: {len(ps)}" for r, ps in sorted(by_repo.items())),
    )
    return {
        "repos": {r: len(ps) for r, ps in by_repo.items()},
        "n_files": len(uploaded),
        "plan_deviation": deviation,
    }


# ---------------------------------------------------------------------------
# Preflight (behind-origin/main false positive tolerated)
# ---------------------------------------------------------------------------
def _parse_preflight_json(raw: str) -> dict[str, Any]:
    """Parse ``orchestrate.preflight --json`` stdout into a payload dict.

    The preflight CLI ALWAYS pretty-prints (``json.dumps(..., indent=2)``),
    so the last stdout line is a bare ``}`` — never parse
    ``splitlines()[-1]`` (incident #602: ``json.loads("}")`` killed all 3
    GCP boots at this gate). Strategy: parse the WHOLE stripped stdout
    first; on failure, slice from the FIRST ``{`` to tolerate non-JSON
    prefix noise (uv/env chatter on fresh VMs). Raises
    ``json.JSONDecodeError`` when no JSON object can be recovered.
    """
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start <= 0:  # no "{" at all, or already tried from position 0
            raise
        return json.loads(stripped[start:])


def run_preflight() -> None:
    """Run project preflight; tolerate ONLY the feature-branch git error.

    Preflight counts ``HEAD..origin/main`` so every issue-<N> pod checkout
    reports "Local is N commit(s) behind origin/main" — that single error
    must never be the launch-killer (incident #552). Only stdout is parsed
    (stderr may carry uv noise).
    """
    proc = subprocess.run(
        [sys.executable, "-m", "explore_persona_space.orchestrate.preflight", "--json"],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    try:
        payload = _parse_preflight_json(proc.stdout)
    except Exception as e:
        raise RuntimeError(f"preflight emitted unparseable output: {proc.stdout[-800:]}") from e
    errors = [e for e in payload.get("errors", []) if "behind origin/main" not in str(e)]
    if errors:
        raise RuntimeError(f"preflight FAILED: {errors}")
    logger.info("[phase=preflight] OK (feature-branch behind-origin tolerated)")


def write_sentinel(
    args: argparse.Namespace, summary: dict[str, Any], by: str = "issue602_extract_dispatch"
) -> Path:
    """End-of-run results sentinel (poll_pipeline contract)."""
    sentinel_dir = Path(args.sentinel_dir)
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    path = sentinel_dir / f"issue-602-epm_results-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": "epm:results",
        "version": 1,
        "task_id": 602,
        "by": by,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": json.dumps(summary),
    }
    path.write_text(json.dumps(payload, indent=1))
    return path


def main() -> int:
    """Dispatcher entry: phases generate -> extract -> estimators -> upload."""
    parser = argparse.ArgumentParser(
        description="#602 pod dispatcher (Phase 1a-1d)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Cell-id subset (default: all 31; smoke default: one refusal + one marker cell)",
    )
    parser.add_argument("--smoke", action="store_true", help="Tiny-slice parameterization")
    parser.add_argument("--model-id", default=bk.BASE_MODEL_ID)
    parser.add_argument("--gpus", default=None, help="Comma list of GPU ids (default: all visible)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--e2-ks", nargs="+", default=[str(k) for k in bk.E2_K_SWEEP])
    parser.add_argument("--hub-bucket", default=bk.HUB_BUCKET)
    parser.add_argument("--sentinel-dir", default="/workspace/logs")
    parser.add_argument(
        "--skip-i474-check",
        action="store_true",
        help="Skip the assumption-8 #474 prompt-reconstruction gate (debug only)",
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-estimators", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--adapter-override",
        default=None,
        help=(
            "(CPU smoke only) use this local adapter dir for EVERY cell "
            "instead of the registry download — pairs with a stub --model-id"
        ),
    )
    parser.add_argument(
        "--phase-internal",
        choices=["generate"],
        default=None,
        help="(internal) run one phase in THIS process — used so vLLM dies with its subprocess",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    # `uv run python` does NOT auto-load .env — without this the subprocess
    # env dicts ({**os.environ}) would lack HF_TOKEN (task #397 round-10').
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    if args.phase_internal == "generate":
        phase_generate(args)
        return 0

    t0 = time.time()
    if not args.skip_preflight:
        run_preflight()
    cells = resolve_cells(args.cells, args.smoke)
    units = active_units(cells)
    gpus = (
        args.gpus.split(",") if args.gpus else [str(i) for i in range(max(1, _visible_gpu_count()))]
    )
    logger.info(
        "[phase=plan] %d cells, %d estimator units, gpus=%s, smoke=%s",
        len(cells),
        len(units),
        gpus,
        args.smoke,
    )

    if not args.skip_i474_check:
        phase_i474_crosscheck(args, cells, gpus)
    if not args.skip_generate:
        logger.info("[phase=generate] dispatching vLLM generation subprocess (gpu %s)", gpus[0])
        gen_cmd = [sys.executable, str(Path(__file__).resolve()), "--phase-internal", "generate"]
        gen_cmd += ["--model-id", args.model_id, "--max-new-tokens", str(args.max_new_tokens)]
        gen_cmd += ["--e2-ks", *args.e2_ks]
        if args.smoke:
            gen_cmd.append("--smoke")
        if args.cells:
            gen_cmd += ["--cells", *args.cells]
        log = bk.eval_dir(REPO) / "logs" / "generate.log"
        proc = _sh(gen_cmd, log, gpu=gpus[0])
        rc = proc.wait()
        if rc != 0:
            tail = log.read_text().splitlines()[-30:]
            tail_text = "\n".join(tail)
            raise RuntimeError(f"generate phase failed rc={rc}:\n{tail_text}")
    if not args.skip_extract:
        logger.info("[phase=extract] %d active cells", len(cells))
        phase_extract(args, cells, gpus)
    if not args.skip_estimators:
        logger.info("[phase=estimators] %d active units", len(units))
        phase_estimators(args, cells, gpus)
    upload_info: dict[str, Any] = {"skipped": True}
    if not args.skip_upload:
        logger.info("[phase=upload]")
        upload_info = phase_upload(args)

    summary = {
        "cells": [c["cell_id"] for c in cells],
        "units": [f"{u['family']}__{u['source']}" for u in units],
        "smoke": args.smoke,
        "upload": upload_info,
        "wall_s": round(time.time() - t0, 1),
        "git_commit": bk.git_sha(REPO),
    }
    sentinel = write_sentinel(args, summary)
    logger.info("results sentinel written: %s", sentinel)
    logger.info("[phase=done]")
    return 0


def _visible_gpu_count() -> int:
    """Number of CUDA devices (0 on CPU-only hosts)."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
