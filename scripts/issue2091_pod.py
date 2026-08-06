"""GPU pod driver for issue #2091 (deterministic-vs-stochastic decoding), P0-P2u.

Decodes ONE greedy completion per staged context (``temperature=0.0``,
``max_new_tokens=1024``, ``k_rollouts=1``) across the 9 rung-jobs
``scripts/issue2091_stage_contexts.py`` staged, then teacher-forced-captures the
answer-side summaries so the new greedy rows join the SAME representation space
as the banked #1739 stochastic captures.

Phase sequence per rung-job (GPU-width discipline: no GPU idles through
API-bound or CPU-bound work, and vLLM never co-resides with the HF capture model):

    P1  greedy generation via the production vLLM seam (batched, chunked)
    P2u-text  pack the rollout tree into <= 9 MB jsonl shards and upload the TEXT
              FIRST (Upload Policy: text/JSON always, before any reduction)
    P2  reap the vLLM engine, DRAIN the GPU, then batched teacher-forced capture
        (``prefix_end`` + ``context_end`` + ``t1``, all 28 layers, fp16)
    P2  cross-campaign parity probe: re-capture the banked completions of this
        behavior's probe shard through THIS rig (the cosines are computed in P4
        against the already-staged banked slices — see PARITY_PROBE_MODE)
    P2u-store  upload the capture store as a PER-FILE SHARD DIR (never a tar)

FAN-OUT: the rung-job axis is embarrassingly parallel, so the parent process
dispatches one SUBPROCESS per GPU, largest-job-first and work-conserving (an idle
GPU takes the next pending job immediately — no wave barriers). Each child is
pinned with ``CUDA_VISIBLE_DEVICES=<gpu>`` in its LAUNCHER ENV *and* the matching
``--gpu-id <gpu>``: the in-process clobber in ``train/sft.py`` is silently
defeated by any import-time cuInit, so the env pin is the load-bearing one and
the flag keeps the record honest. Each child gets its OWN out-root / store-root /
HF prefix, because the pack root, the capture store's internal shard numbering,
and the per-job done file would otherwise collide.

PROMPT PARITY: every packed-source rung-job REPLAYS the banked campaign's own
rendered ``(prefix_text, prompt_text)`` pair verbatim (``banked_render_fn``), so
the greedy decode runs on prompts byte-identical to those the banked K=5
stochastic rollouts were sampled from — and each is verified against its staged
``meta.banked_prompt_sha256`` before any GPU work. The WildChat rung renders
multi-turn from its contexts shards through the wcrung renderer (the same
function the banked wcrung campaign used).

Resume: ``generation.generate_labeling`` resumes per CONTEXT (fingerprint-keyed)
and ``capture.capture_rollout_files`` per SHARD, so a relaunch re-does only
unfinished work. The per-job completion record lives at
``<out_root>/<rungjob>/_job_done.json`` — deliberately OUTSIDE the
``/workspace/logs/issue-2091-*.json`` sentinel glob, which the VM poller DRAINS
and renames ``.processed`` on every tick (a dispatcher that read its own
sentinels back would find them gone within ~one tick).

Model + revision are pinned by the generation module
(``MODEL_NAME`` @ ``INSTRUCT_REVISION``); the prompt budget
(``MAX_MODEL_LEN - max_new_tokens``) is enforced inside ``generate_labeling``.

This driver NEVER shells out to ``scripts/task.py`` (pods run on ``issue-<N>``
branches; ``task.py`` branch-guards to ``main``). It reports through
``[phase=...]`` log lines + sentinel files only.

CONTENT HYGIENE: the staged contexts carry jailbreak-prefix and real-user-corpus
text; logs and artifacts here carry ids, counts, shapes, and digests — never row
text.
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


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root onto sys.path (script mode puts only scripts/ there — #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue2091_pod.py"
    assert sentinel.exists(), f"repo-root derivation failed: {sentinel} missing"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE any torch/vllm import (thread caps + credentials)

# vLLM v1 runs EngineCore in a subprocess and defaults to fork(); this driver
# loads the HF tokenizer before the engine, which is exactly the poisoned-fork
# case that kills EngineCore 1-4 s after init with no traceback of its own
# (gotchas.md #628). vLLM reads this at IMPORT time, so it must be set before any
# `import vllm` anywhere in the process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Real-corpus vLLM hang mitigations (gotchas.md PRE-LAUNCH CHECKLIST): long
# real-user prompts on multi-GPU RunPod wedge vLLM in generate() often enough
# that the knobs are the default, not a post-hang remedy. setdefault, so a
# launcher may pin "0" — every cell then runs under ONE engine config (the
# one-config-per-comparison rule).
os.environ.setdefault("EPM_VLLM_ENFORCE_EAGER", "1")
os.environ.setdefault("EPM_VLLM_DISABLE_PREFIX_CACHING", "1")

logger = logging.getLogger("issue2091_pod")

TASK_ID = 2091
HF_PREFIX = "issue2091_decode"
LOGS_DIR = Path("/workspace/logs")

# Greedy decode pins (plan §4.1; Source: #1073 SP_GREEDY).
K_ROLLOUTS = 1
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 1024

# The banked reference vectors for the parity probe live inside 32-70 GB
# per-behavior labeling TARS, so streaming a member set out of them is not a
# cheap pod-side fetch: the probe's NEW vectors persist in the greedy store and
# the per-behavior cosines are computed in P4 against the already-staged banked
# slices (plan §4.2 P2, the either-or branch). Either way the cosines land before
# any R2 verdict is narrated.
PARITY_PROBE_MODE = "deferred-to-p4"

# Plan-named sentinel files (§9 phase_outputs). All carry the poller's
# _SENTINEL_REQUIRED_KEYS, so each is parsed, posted, and renamed .processed.
PILOT_SENTINEL = "issue-2091-pilot.json"
UPLOAD_DONE_SENTINEL = "issue-2091-upload-done.json"
RESULTS_SENTINEL = "issue-2091-results.json"

SENTINEL_SCHEMA_VERSION = 1


# ── local helpers ─────────────────────────────────────────────────────────────
def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ},
            ).stdout.strip()
            or "unavailable-no-git-checkout"
        )
    except OSError:
        return "unavailable-no-git-checkout"


def _logs_dir() -> Path:
    """``/workspace/logs`` on a pod; a repo-local fallback off-pod (smoke)."""
    if LOGS_DIR.is_dir():
        return LOGS_DIR
    fallback = REPO_ROOT / "logs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def write_sentinel(kind: str, note: dict, *, filename: str | None = None, gate: str = "") -> Path:
    """poll_pipeline.py-conformant sentinel (``_SENTINEL_REQUIRED_KEYS``).

    Written ONCE per path, never rewritten in place: the VM poller renames each
    posted sentinel ``.processed`` (``mv -n``), so a rewrite whose ``.processed``
    twin exists is un-renameable and re-warned every tick.

    ``version`` is hardcoded 1 — pod-side writers cannot read ``events.jsonl``,
    and the VM-side drain re-derives the landed marker version as max+1 for a
    real ``epm:results`` sentinel.

    Not ``scripts/issue779_common.write_sentinel``: that helper hardcodes the
    ``issue-<task>-<kind_slug>-<epoch>`` filename with no override, and the plan's
    §9 ``phase_outputs`` block names three specific sentinel FILES this driver
    must write.
    """
    kind_slug = kind.replace(":", "_")
    name = filename or f"issue-{TASK_ID}-{kind_slug}-{int(time.time())}.json"
    path = _logs_dir() / name
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,
        "note": json.dumps(note, indent=2, sort_keys=True) if isinstance(note, dict) else note,
        "task_id": TASK_ID,
        "by": "issue2091_pod",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if gate:
        payload["gate"] = gate
        payload["blocks_pipeline"] = False
    write_json_atomic(path, payload)
    logger.info("[sentinel] wrote %s (kind=%s)", path, kind)
    return path


def _upload_dir(local: Path, path_in_repo: str, *, skip: bool) -> str:
    """Bulk folder upload to the data repo (ONE commit — never a per-file loop)."""
    if skip:
        logger.info("[upload] SKIP (--skip-upload) %s -> %s", local, path_in_repo)
        return ""
    from explore_persona_space.orchestrate import hub

    url = hub._upload(local, hub.DEFAULT_DATASET_REPO, "dataset", path_in_repo, raise_on_error=True)
    logger.info("[upload] %s -> %s (%s)", local, path_in_repo, url or "no-url")
    return url


# ── GPU enumeration ───────────────────────────────────────────────────────────
def visible_gpu_ids() -> list[str]:
    """GPU ids this process may pin children to.

    ``CUDA_VISIBLE_DEVICES`` wins when set (a launcher's deliberate restriction);
    otherwise ``nvidia-smi -L`` indices, which respect container visibility.
    Never sized from a bare device count on a SHARED node — this driver pins
    ``backend: runpod`` (exclusive host), where nvidia-smi enumeration is right.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        ids = [x.strip() for x in env.split(",") if x.strip()]
        if ids:
            return ids
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
    except OSError as exc:
        # No nvidia-smi at all (a CPU-only host) — name the cause rather than
        # surfacing a bare FileNotFoundError from deep in the dispatcher.
        raise RuntimeError(
            "no GPUs visible: CUDA_VISIBLE_DEVICES is unset/empty and nvidia-smi is "
            f"not available on this host ({exc}) — this driver must run on a GPU pod"
        ) from exc
    ids = [line.strip() for line in proc.stdout.splitlines() if line.strip().isdigit()]
    if not ids:
        raise RuntimeError(
            "no GPUs visible: CUDA_VISIBLE_DEVICES is unset/empty and nvidia-smi "
            f"listed none (rc={proc.returncode})"
        )
    return ids


# ── per-job paths ─────────────────────────────────────────────────────────────
def job_out_root(args: argparse.Namespace, name: str) -> Path:
    return args.out_root / name


def job_store_dir(args: argparse.Namespace, name: str) -> Path:
    return args.store_root / f"greedy_{name}"


def probe_store_dir(args: argparse.Namespace, behavior: str) -> Path:
    return args.store_root / f"parity_probe_{behavior}"


def job_done_path(args: argparse.Namespace, name: str) -> Path:
    """Per-job completion record — OUTSIDE the drained sentinel glob."""
    return job_out_root(args, name) / "_job_done.json"


# ── context staging (parent) ──────────────────────────────────────────────────
def stage_contexts(args: argparse.Namespace) -> dict:
    """Stage the contexts tree from HF and return its manifest.

    ``stage_hub_prefix`` mirrors the prefix VERBATIM, so files land at
    ``<stage-root>/<hf prefix>/...``; the manifest resolves under that mirror.
    """
    from explore_persona_space.orchestrate import hub

    prefix = f"{args.hf_prefix}/contexts"
    mirror = args.stage_root / prefix
    manifest_path = mirror / "stage_manifest.json"
    if manifest_path.exists() and not args.restage:
        logger.info("[stage] contexts already staged at %s", mirror)
    else:
        staged = hub.stage_hub_prefix(
            hub.DEFAULT_DATASET_REPO,
            prefix,
            args.stage_root,
            repo_type="dataset",
            revision=args.dataset_revision,
        )
        logger.info("[stage] staged %d context file(s) from %s", len(staged), prefix)
    if not manifest_path.exists():
        raise RuntimeError(
            f"contexts staged but {manifest_path} is missing — the staging step "
            "(scripts/issue2091_stage_contexts.py) must run and upload first"
        )
    manifest = json.loads(manifest_path.read_text())
    print(
        f"[phase=p0_stage] rung_jobs={manifest['n_rung_jobs']} "
        f"contexts={manifest['n_contexts_total']} "
        f"consumed_revision={(args.dataset_revision or 'unpinned')[:12]} "
        f"manifest_input_revision={manifest['dataset_revision'][:12]}",
        flush=True,
    )
    return manifest


def load_job_contexts(args: argparse.Namespace, name: str, limit: int | None) -> list[dict]:
    """Verified context rows for one rung-job (sha256 + count checks in loader)."""
    from scripts.issue2091_stage_contexts import load_shard_rows

    mirror = args.stage_root / f"{args.hf_prefix}/contexts" / name
    rows = load_shard_rows(mirror, "ctx")
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"[{name}] no context rows after limit={limit}")
    return rows


def load_probe_rows(args: argparse.Namespace, behavior: str, limit: int | None) -> list[dict]:
    from scripts.issue2091_stage_contexts import load_shard_rows

    mirror = args.stage_root / f"{args.hf_prefix}/contexts" / "parity_probe" / behavior
    rows = load_shard_rows(mirror, "probe")
    if limit is not None:
        rows = rows[:limit]
    return rows


# ── render + engine helpers (child) ───────────────────────────────────────────
def banked_render_fn(tokenizer, row: dict) -> tuple[str, str]:
    """Replay the BANKED ``(prefix_text, prompt_text)`` pair verbatim.

    The staged rows carry the banked campaign's own rendered pair, so the greedy
    decode runs on prompts BYTE-IDENTICAL to the ones the banked K=5 stochastic
    rollouts were sampled from — the comparability this task's decode-regime
    contrast requires, and stronger than re-rendering (which cannot be verified
    byte-for-byte without exactly this anchor).

    Re-rendering is not merely redundant here, it is WRONG: the packed row's
    ``prefix_text`` is the RENDERED chat-template prefix, not the raw persona
    string, so feeding it back through ``context_messages`` would nest an
    already-rendered prefix inside a fresh system turn and double-wrap the
    template (the staging smoke measured 40/40 mismatching contexts before this
    seam existed).
    """
    prompt_text = row.get("prompt_text")
    if not prompt_text:
        raise RuntimeError(
            f"context {row.get('context_id')!r} carries no banked prompt_text — the "
            "staging step must emit it for every packed-source rung-job"
        )
    return row.get("prefix_text") or "", prompt_text


def wildchat_render_fn(tokenizer, row: dict) -> tuple[str, str]:
    """``(prefix_text, prompt_text)`` for a WildChat row — MULTI-TURN safe.

    The shared ``generation.render_prompt_parts`` slices the prefix at the FIRST
    user-turn header, which is right for a single-user-turn rung (system persona +
    query) and WRONG for a conversation prefix: it would cut the earlier turns out
    of the prefix while ``capture`` derives ``prefix_end`` from
    ``len(prefix_text)``, silently mis-positioning every prefix-arm read with no
    crash. Delegates to the wcrung renderer (the #1092 instruct convention), the
    same function the banked wcrung rollouts were rendered with.
    """
    from scripts.issue1739_wcrung_contexts import render_row_prompt

    return render_row_prompt(tokenizer, row.get("prefix_turns") or [], row["query"])


def assert_render_parity(tokenizer, rows: list[dict], render_fn) -> dict:
    """Assert each freshly-rendered prompt matches the banked prompt's sha256.

    The staging step recorded ``meta.banked_prompt_sha256`` from the banked
    rollout row. Model, revision, and chat template are identical here, so the
    render MUST reproduce it byte-for-byte; a mismatch means template/tokenizer
    drift, which would shift every capture position with no other symptom. Rows
    with no anchor (the WildChat rung, whose prompts come from the contexts
    shards rather than packed completions) are counted and skipped.
    """
    checked = 0
    skipped = 0
    mismatches: list[str] = []
    for row in rows:
        want = (row.get("meta") or {}).get("banked_prompt_sha256")
        if not want:
            skipped += 1
            continue
        _, prompt_text = render_fn(tokenizer, row)
        import hashlib

        got = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        checked += 1
        if got != want:
            mismatches.append(row["context_id"])
    if mismatches:
        raise RuntimeError(
            f"render parity FAILED for {len(mismatches)}/{checked} contexts — the freshly "
            "rendered prompt does not reproduce the banked prompt sha256, so every capture "
            f"position would shift silently. First few: {mismatches[:5]}"
        )
    digest = {"n_checked": checked, "n_no_anchor": skipped, "n_mismatch": 0}
    logger.info("[render-parity] checked=%d no_anchor=%d mismatch=0", checked, skipped)
    return digest


def reap_generation_engine(
    gpu_id: int | str, drain_timeout_s: int = 180, floor_mib: int = 2048
) -> None:
    """Reap the module-cached vLLM engine and DRAIN-WAIT this unit's OWN GPU.

    vLLM and the HF capture model must NEVER co-reside: the engine reserves
    ``gpu_memory_utilization`` of HBM, so a resident 7B bf16 HF model makes the
    engine init raise (and the reverse starves the capture load). The drain
    verdict reads DEVICE-level ``memory.used`` — never compute-apps rows alone,
    which are pid-visibility-dependent inside a container and read EMPTY for a
    holder this process cannot resolve (#825/#1333) — and is SCOPED to this
    unit's own physical device ``gpu_id``: ``nvidia-smi --query-gpu`` does NOT
    honor ``CUDA_VISIBLE_DEVICES`` (it always enumerates the whole node,
    gotchas.md), so in the concurrent fan-out an all-device max verdict is
    unsatisfiable-by-construction whenever a sibling rung-job legitimately
    holds its own engine — the #1333 class; the unscoped verdict falsely
    killed hal_nqopen/hal_simpleqa (own devices at 0 MiB) at rc=1.
    """
    from explore_persona_space.experiments.issue_1739 import generation as _gen

    llm = _gen._TOKENIZER_CACHE.pop("_llm", None)
    if llm is None:
        logger.info("[reap] no cached vLLM engine to reap")
        return
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)
    del llm
    _drain_wait_own_gpu(gpu_id, drain_timeout_s=drain_timeout_s, floor_mib=floor_mib)


def _drain_wait_own_gpu(gpu_id: int | str, drain_timeout_s: int, floor_mib: int) -> None:
    """DRAIN-WAIT until this unit's OWN GPU reads memory.used <= ``floor_mib``.

    Only the own-device row drives the verdict; the full per-GPU list rides
    every log/error line as diagnostics. Fail-loud semantics are preserved:
    a genuine leak on the OWN device still raises at ``drain_timeout_s``, and
    a poll that returns rows but NONE matching ``gpu_id`` raises immediately —
    a missing row must never read as drained. An EMPTY parse (transient
    nvidia-smi hiccup) keeps the pre-existing retry-until-deadline tolerance.
    """
    try:
        own_idx = int(str(gpu_id).strip())
    except ValueError as exc:
        raise RuntimeError(
            f"drain check needs an integer physical GPU index, got gpu_id={gpu_id!r}"
        ) from exc
    deadline = time.time() + drain_timeout_s
    while True:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
        used = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[1].isdigit():
                used.append((int(parts[0]), int(parts[1])))
        own = [m for idx, m in used if idx == own_idx]
        if used and not own:
            # nvidia-smi enumerated devices but none matches this unit's own
            # index — a shape change / parse miss. NEVER read that as drained.
            raise RuntimeError(
                "vLLM teardown drain check: nvidia-smi returned rows for GPU indices "
                f"{[idx for idx, _ in used]} but none for this unit's own gpu_id={own_idx} "
                f"(per-GPU used MiB: {used}) — refusing to treat a missing row as drained"
            )
        if own and own[0] <= floor_mib:
            print(
                f"[phase=p2_reap] GPU drained: gpu={own_idx} max_used={own[0]} MiB "
                f"(verdict scoped to own device; per-GPU used MiB: {used})",
                flush=True,
            )
            return
        if time.time() >= deadline:
            raise RuntimeError(
                f"vLLM teardown did not drain below {floor_mib} MiB within "
                f"{drain_timeout_s}s on this unit's own gpu_id={own_idx} "
                f"(own used MiB: {own[0] if own else 'no row parsed'}; "
                f"per-GPU used MiB: {used})"
            )
        time.sleep(5)


# ── child: one rung-job end to end ────────────────────────────────────────────
def run_job(args: argparse.Namespace) -> dict:
    """P1 -> P2u-text -> P2 -> P2u-store for ONE rung-job."""
    from explore_persona_space.experiments.issue_1739 import capture as capture_mod
    from explore_persona_space.experiments.issue_1739 import generation
    from explore_persona_space.experiments.issue_1739.constants import HIDDEN_DIM, N_LAYERS
    from scripts.issue1739_pack import pack_raw_tree
    from scripts.issue2091_stage_contexts import RUNG_JOBS_BY_NAME

    name = args.rungjob
    job = RUNG_JOBS_BY_NAME[name]
    out_root = job_out_root(args, name)
    out_root.mkdir(parents=True, exist_ok=True)
    hf_job_prefix = f"{args.hf_prefix}/raw_completions/greedy/{name}"

    done_path = job_done_path(args, name)
    if done_path.exists() and not args.force:
        record = json.loads(done_path.read_text())
        print(f"[phase=p1_generate job={name}] RESUMED-COMPLETE (see {done_path})", flush=True)
        return record

    rows = load_job_contexts(args, name, args.limit)
    tokenizer = generation.get_tokenizer()
    # WildChat renders multi-turn from its contexts shards (the same renderer the
    # banked wcrung campaign used, so parity holds by construction); every
    # packed-source rung-job REPLAYS the banked prompt verbatim.
    render_fn = wildchat_render_fn if job.source == "wcrung" else banked_render_fn

    # Prompt parity BEFORE any GPU work: cheap to check here, and a silent
    # mismatch shifts every capture position with no other symptom.
    parity = assert_render_parity(tokenizer, rows, render_fn)

    print(
        f"[phase=p1_generate job={name} behavior={job.gen_behavior} rung={job.rung}] "
        f"contexts={len(rows)} gpu={args.gpu_id} k={K_ROLLOUTS} temp={TEMPERATURE} "
        f"max_new={MAX_NEW_TOKENS}",
        flush=True,
    )
    t0 = time.time()
    gen_kwargs: dict = {
        "out_root": out_root,
        "behavior": job.gen_behavior,
        "k_rollouts": K_ROLLOUTS,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": args.seed,
        "tokenizer": tokenizer,
        "render_fn": render_fn,
    }
    gen_manifest = generation.generate_labeling(rows, **gen_kwargs)
    n_rollouts = gen_manifest["n_kept"] * gen_manifest["k_rollouts"]
    cap_hit_fraction = gen_manifest["n_truncated_rollouts"] / n_rollouts if n_rollouts else 0.0
    print(
        f"[phase=p1_generate job={name}] kept={gen_manifest['n_kept']} "
        f"generated={gen_manifest['n_generated']} resumed={gen_manifest['n_resumed']} "
        f"cap_hit={gen_manifest['n_truncated_rollouts']}/{n_rollouts} "
        f"({cap_hit_fraction:.4f}) elapsed={time.time() - t0:.0f}s",
        flush=True,
    )

    # Rollout TEXT uploads FIRST — durability before any reduction (#779).
    pack_root = out_root / "labeling_packed"
    pack_manifest = pack_raw_tree(out_root / "labeling", pack_root)
    n_text_shards = sum(len(g.get("shards", [])) for g in pack_manifest["groups"].values())
    print(
        f"[phase=p2u_text job={name}] groups={len(pack_manifest['groups'])} shards={n_text_shards}",
        flush=True,
    )
    _upload_dir(pack_root, hf_job_prefix, skip=args.skip_upload)

    cap_manifest: dict | None = None
    probe_manifest: dict | None = None
    if args.skip_capture:
        print(f"[phase=p2_capture job={name}] SKIPPED (--skip-capture)", flush=True)
    else:
        # Engine reaped + OWN GPU drained BEFORE the HF capture model loads
        # (verdict scoped to args.gpu_id — sibling jobs hold their own GPUs).
        reap_generation_engine(args.gpu_id)
        model = capture_mod.load_capture_model(device=args.device)

        rollout_dir = out_root / "labeling" / job.gen_behavior
        rollout_paths = sorted(p for p in rollout_dir.glob("*.json") if not p.name.startswith("_"))
        if not rollout_paths:
            raise RuntimeError(f"[{name}] no rollout files under {rollout_dir}")
        store_dir = job_store_dir(args, name)
        cap_kwargs: dict = {
            "store_dir": store_dir,
            "model": model,
            "tokenizer": tokenizer,
            "n_layers": N_LAYERS,
            "hidden_dim": HIDDEN_DIM,
            "device": args.device,
            "fingerprint": gen_manifest["fingerprint"],
        }
        if args.capture_batch_size:
            cap_kwargs["batch_size"] = args.capture_batch_size
        t1 = time.time()
        cap_manifest = capture_mod.capture_rollout_files(rollout_paths, **cap_kwargs)
        print(
            f"[phase=p2_capture job={name}] rows={cap_manifest.get('n_rows')} "
            f"shards={cap_manifest.get('n_shards')} elapsed={time.time() - t1:.0f}s",
            flush=True,
        )
        _upload_dir(
            store_dir, f"{args.hf_prefix}/capture_store/greedy_{name}", skip=args.skip_upload
        )

        if job.probe_behavior:
            probe_manifest = run_parity_probe(
                args, job.probe_behavior, model, tokenizer, gen_manifest["fingerprint"]
            )

    record = {
        "rungjob": name,
        "gen_behavior": job.gen_behavior,
        "judge_behaviors": list(job.judge_behaviors),
        "rung": job.rung,
        "gpu_id": args.gpu_id,
        "n_contexts": gen_manifest["n_contexts"],
        "n_kept": gen_manifest["n_kept"],
        "n_generated": gen_manifest["n_generated"],
        "n_resumed": gen_manifest["n_resumed"],
        "k_rollouts": gen_manifest["k_rollouts"],
        "n_truncated_rollouts": gen_manifest["n_truncated_rollouts"],
        "cap_hit_fraction": round(cap_hit_fraction, 6),
        "prompt_budget_drops": gen_manifest["prompt_budget_drops"]["n_dropped"],
        "gen_fingerprint": gen_manifest["fingerprint"],
        "render_parity": parity,
        "n_text_shards": n_text_shards,
        "capture_rows": (cap_manifest or {}).get("n_rows"),
        "capture_shards": (cap_manifest or {}).get("n_shards"),
        "capture_over_budget": (cap_manifest or {}).get("n_over_budget"),
        "parity_probe": probe_manifest,
        "hf_text_prefix": hf_job_prefix,
        "hf_store_prefix": f"{args.hf_prefix}/capture_store/greedy_{name}",
        "limit": args.limit,
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(done_path, record)
    print(f"[phase=p2u_store job={name}] job complete: {done_path}", flush=True)
    return record


def run_parity_probe(
    args: argparse.Namespace, behavior: str, model, tokenizer, fingerprint: str
) -> dict:
    """Re-capture this behavior's banked completions through THIS capture rig.

    Writes the banked rows out in the labeling-rollout payload shape (what
    ``capture_rollout_files`` consumes), then captures them into
    ``parity_probe_<behavior>/``. The banked reference vectors are NOT fetched
    here (they sit inside a 32-70 GB tar); the per-behavior
    ``cos(t1_new, t1_banked)`` / ``cos(context_end_new, context_end_banked)``
    are computed in P4 against the already-staged banked slices.
    """
    from explore_persona_space.experiments.issue_1739 import capture as capture_mod
    from explore_persona_space.experiments.issue_1739.constants import HIDDEN_DIM, N_LAYERS

    rows = load_probe_rows(args, behavior, args.limit)
    if not rows:
        raise RuntimeError(f"[probe {behavior}] no probe rows staged")
    probe_root = args.out_root / "parity_probe" / behavior
    probe_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for row in rows:
        path = probe_root / f"{row['context_id']}_seed{int(row.get('rollout_k') or 0)}.json"
        write_json_atomic(
            path,
            {
                "context_id": row["context_id"],
                "behavior": behavior,
                "split": "parity_probe",
                "rung": row.get("rung"),
                "group_key": row.get("group_key"),
                "rollout_k": row.get("rollout_k"),
                "query": row.get("prompt_text", ""),
                "prefix_text": row.get("prefix_text", ""),
                "prompt_text": row.get("prompt_text", ""),
                "completion": row.get("completion", ""),
                "meta": dict(row.get("meta") or {}, parity_probe=True),
            },
        )
        written.append(path)

    store_dir = probe_store_dir(args, behavior)
    cap_kwargs: dict = {
        "store_dir": store_dir,
        "model": model,
        "tokenizer": tokenizer,
        "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "device": args.device,
        "fingerprint": f"{fingerprint}-probe",
    }
    if args.capture_batch_size:
        cap_kwargs["batch_size"] = args.capture_batch_size
    manifest = capture_mod.capture_rollout_files(sorted(written), **cap_kwargs)
    _upload_dir(
        store_dir,
        f"{args.hf_prefix}/capture_store/parity_probe_{behavior}",
        skip=args.skip_upload,
    )
    digest = {
        "behavior": behavior,
        "mode": PARITY_PROBE_MODE,
        "n_probe_rows": len(rows),
        "capture_rows": manifest.get("n_rows"),
        "capture_shards": manifest.get("n_shards"),
        "hf_store_prefix": f"{args.hf_prefix}/capture_store/parity_probe_{behavior}",
        "cosines": None,
        "cosines_note": (
            "cos(t1_new, t1_banked) + cos(context_end_new, context_end_banked) are "
            "computed in P4 against the already-staged banked slices — the banked "
            "reference vectors live inside a 32-70 GB labeling tar, which is not a "
            "cheap pod-side member fetch (plan §4.2 P2 either-or)"
        ),
    }
    print(
        f"[phase=p2_probe behavior={behavior}] rows={len(rows)} "
        f"capture_rows={manifest.get('n_rows')} mode={PARITY_PROBE_MODE}",
        flush=True,
    )
    return digest


# ── parent: work-conserving fan-out ───────────────────────────────────────────
def child_command(args: argparse.Namespace, name: str, gpu: str) -> tuple[list[str], dict]:
    """Argv + child env for one rung-job subprocess.

    The env pin is load-bearing: any import-time cuInit freezes the driver's
    device list before an in-process ``CUDA_VISIBLE_DEVICES`` write can take, so
    a flag-only pin silently co-locates every child on physical GPU 0. The
    matching ``--gpu-id`` keeps the per-job record honest.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "issue2091_pod.py"),
        "--mode",
        "job",
        "--rungjob",
        name,
        "--gpu-id",
        gpu,
        "--out-root",
        str(args.out_root),
        "--store-root",
        str(args.store_root),
        "--stage-root",
        str(args.stage_root),
        "--hf-prefix",
        args.hf_prefix,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.capture_batch_size:
        cmd += ["--capture-batch-size", str(args.capture_batch_size)]
    if args.skip_upload:
        cmd.append("--skip-upload")
    if args.skip_capture:
        cmd.append("--skip-capture")
    if args.force:
        cmd.append("--force")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    return cmd, env


def dispatch(args: argparse.Namespace, manifest: dict) -> dict:
    """Run every rung-job across the visible GPUs, largest-first, work-conserving.

    One in-flight child per GPU; the moment a child exits its GPU takes the next
    pending job (no wave barrier — a work-conserving dispatcher never idles a GPU
    while an independent cell is pending). Each child is subprocess-isolated so
    a per-job vLLM engine + HF capture model never share a process, and so the
    parent's own env is never poisoned by a child's device pin.
    """
    from scripts.issue2091_stage_contexts import RUNG_JOBS

    gpus = visible_gpu_ids()
    jobs = [j.name for j in RUNG_JOBS if j.name in manifest["rung_jobs"]]
    # Largest first: the longest job must not start last.
    jobs.sort(key=lambda n: manifest["rung_jobs"][n]["n_contexts_realized"], reverse=True)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        unknown = wanted - set(jobs)
        if unknown:
            raise SystemExit(f"--only names unknown rung-job(s): {sorted(unknown)}")
        jobs = [j for j in jobs if j in wanted]

    print(
        f"[phase=p1_dispatch] jobs={len(jobs)} gpus={','.join(gpus)} order={','.join(jobs)}",
        flush=True,
    )
    pending = list(jobs)
    running: dict[str, tuple[subprocess.Popen, str, float]] = {}
    free_gpus = list(gpus)
    failures: dict[str, int] = {}
    t0 = time.time()

    while pending or running:
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            name = pending.pop(0)
            cmd, env = child_command(args, name, gpu)
            log_path = job_out_root(args, name) / f"job_{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("ab")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(REPO_ROOT),
            )
            running[name] = (proc, gpu, time.time())
            print(
                f"[phase=p1_dispatch] launched job={name} gpu={gpu} pid={proc.pid} log={log_path}",
                flush=True,
            )
        time.sleep(5)
        for name in list(running):
            proc, gpu, started = running[name]
            rc = proc.poll()
            if rc is None:
                continue
            del running[name]
            free_gpus.append(gpu)
            elapsed = time.time() - started
            if rc == 0:
                print(
                    f"[phase=p1_dispatch] job={name} gpu={gpu} rc=0 elapsed={elapsed:.0f}s",
                    flush=True,
                )
            else:
                failures[name] = rc
                tail = _tail_log(job_out_root(args, name) / f"job_{name}.log")
                # Echo the child's tail into the MAIN log: on an ephemeral lane
                # only the main workload log is crash-persisted, so an inner log
                # dies with the box and the root cause needs a fresh repro.
                print(
                    f"[phase=p1_dispatch] job={name} gpu={gpu} FAILED rc={rc} "
                    f"elapsed={elapsed:.0f}s; child log tail:\n{tail}",
                    flush=True,
                )

    records: dict[str, dict] = {}
    for name in jobs:
        path = job_done_path(args, name)
        if path.exists():
            records[name] = json.loads(path.read_text())
    if failures:
        raise RuntimeError(
            f"{len(failures)} rung-job(s) failed: {failures}; "
            f"{len(records)}/{len(jobs)} completed (per-job records under {args.out_root})"
        )
    print(
        f"[phase=p1_dispatch] all {len(jobs)} job(s) complete elapsed={time.time() - t0:.0f}s",
        flush=True,
    )
    return records


def _tail_log(path: Path, n_lines: int = 120) -> str:
    if not path.exists():
        return "(no child log)"
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:  # pragma: no cover - diagnostic path
        return f"(child log unreadable: {exc})"
    return "".join(lines[-n_lines:])


# ── parent: results payload ───────────────────────────────────────────────────
def build_results_payload(
    args: argparse.Namespace, manifest: dict, records: dict[str, dict], gpu_hours: float
) -> dict:
    """The /issue Step 7 results payload (all 10 keys required)."""
    total_contexts = sum(r.get("n_kept", 0) for r in records.values())
    total_rollouts = sum(r.get("n_kept", 0) * r.get("k_rollouts", 1) for r in records.values())
    total_truncated = sum(r.get("n_truncated_rollouts", 0) for r in records.values())
    return {
        "eval_numbers": {
            "n_rung_jobs": len(records),
            "n_contexts_decoded": total_contexts,
            "n_greedy_rollouts": total_rollouts,
            "cap_hit_fraction_overall": round(
                total_truncated / total_rollouts if total_rollouts else 0.0, 6
            ),
            "cap_hit_fraction_per_job": {n: r.get("cap_hit_fraction") for n, r in records.items()},
            "capture_rows_per_job": {n: r.get("capture_rows") for n, r in records.items()},
            "prompt_budget_drops_per_job": {
                n: r.get("prompt_budget_drops") for n, r in records.items()
            },
            "parity_probe_mode": PARITY_PROBE_MODE,
        },
        "eval_paths": {
            "per_job_records": [str(job_done_path(args, n)) for n in sorted(records)],
            "contexts_manifest": f"{args.hf_prefix}/contexts/stage_manifest.json",
        },
        "reproducibility_card": {
            "model": manifest.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            "revision": _instruct_revision(),
            "decode": {
                "k_rollouts": K_ROLLOUTS,
                "temperature": TEMPERATURE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "seed": args.seed,
            },
            "hf_dataset_repo": _dataset_repo(),
            "hf_text_prefixes": [r["hf_text_prefix"] for r in records.values()],
            "hf_store_prefixes": [r["hf_store_prefix"] for r in records.values()],
            # Two DISTINCT revisions — do NOT "simplify" them back into one field.
            # contexts_dataset_revision = the data-repo revision this run actually
            # CONSUMED the contexts tree at (args.dataset_revision, the p0_stage
            # fetch pin) — the run-of-record a reproducer must resolve against.
            # contexts_manifest_input_revision = the revision the STAGING script
            # read the upstream #1739 labeling data FROM (input provenance only).
            # In the P0 pilot these differed and the card recorded the manifest
            # field; reproducing from it resolved 0/27 context files (#2091).
            "contexts_dataset_revision": args.dataset_revision,
            "contexts_manifest_input_revision": manifest["dataset_revision"],
            "staging_seed": manifest["seed"],
            "gen_fingerprints": {n: r.get("gen_fingerprint") for n, r in records.items()},
            "wandb_project": None,
            "wandb_run_names": [],
            "note": "no training in this phase — no adapters, no WandB runs",
        },
        "wandb_url": "n/a (no training in P0-P2u; generation + capture only)",
        "hf_hub_url": f"https://huggingface.co/datasets/{_dataset_repo()}/tree/main/{args.hf_prefix}",
        "worktree_path": str(REPO_ROOT),
        "final_commit_sha": _git_commit(),
        "gpu_hours_used": round(gpu_hours, 3),
        "gpu_hours_budgeted": args.gpu_hours_budgeted,
        "plan_deviations": _plan_deviations(manifest),
    }


def _instruct_revision() -> str:
    from explore_persona_space.experiments.issue_1739.generation import INSTRUCT_REVISION

    return INSTRUCT_REVISION


def _dataset_repo() -> str:
    from explore_persona_space.orchestrate import hub

    return hub.DEFAULT_DATASET_REPO


def _plan_deviations(manifest: dict) -> list[str]:
    deviations = [
        (
            "A17 falsified: `source_id` is ABSENT from every packed rollout row "
            "(generate_labeling persists a fixed field set that omits it), so the evil "
            "two-way clustering question axis is the staged "
            "meta.question_key = sha256(query)[:16] rather than the builder's "
            "p{pi:04d}-q{qi:03d}. The prefix axis is unaffected (group_key). Failure "
            "direction is conservative: identical question text collapses two indices "
            "into one cluster (wider CIs), never the reverse."
        ),
        (
            f"Capture-parity probe cosines: {PARITY_PROBE_MODE} — the probe's NEW vectors "
            "persist in the greedy store and the per-behavior cosines are computed in P4 "
            "against the already-staged banked slices (plan §4.2 P2 either-or branch), "
            "because the banked reference vectors sit inside 32-70 GB labeling tars."
        ),
        (
            "Cap-hit disposition (pre-registered, plan §4.2 P1): max_new_tokens stays 1024 "
            "for parent-recipe fidelity — NO re-generation of cap-hit rows; the realized "
            "per-rung cap-hit fraction is recorded and a cap-hit-row-excluded robustness "
            "pass of R2 runs instead."
        ),
    ]
    for check in manifest.get("wcrung_dv_byte_identity", []):
        if check.get("verdict") != "identical":
            deviations.append(
                f"wcrung dv_dataset byte-identity for {check['behavior']}: "
                f"{check['verdict']} (HF@{str(check.get('hf_revision'))[:12]})"
            )
    return deviations


# ── import check ──────────────────────────────────────────────────────────────
def _import_check() -> int:
    """Resolve EVERY deferred import on the REAL branch, then exit 0.

    Lives in its OWN function, not inline in ``main()``: an ``import X`` is a
    BINDING, so the compiler marks X a local of the enclosing function for its
    WHOLE body — an inline block importing the bare name ``capture`` or
    ``generation`` would make a later call to a module-level symbol of that name
    read an unbound local and crash AFTER generation had completed (#1739).
    """
    from explore_persona_space.analysis.representation_shift import (  # noqa: F401
        _reap_vllm_engine,
    )
    from explore_persona_space.experiments.issue_1739 import (  # noqa: F401
        capture,
        constants,
        generation,
    )
    from explore_persona_space.experiments.issue_1739.capture import (  # noqa: F401
        capture_rollout_files,
        load_capture_model,
    )
    from explore_persona_space.experiments.issue_1739.constants import (  # noqa: F401
        HIDDEN_DIM,
        N_LAYERS,
    )
    from explore_persona_space.experiments.issue_1739.generation import (  # noqa: F401
        INSTRUCT_REVISION,
        MAX_MODEL_LEN,
        generate_labeling,
        get_tokenizer,
    )
    from explore_persona_space.orchestrate import hub  # noqa: F401
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        DEFAULT_DATASET_REPO,
        _upload,
        stage_hub_prefix,
    )
    from scripts.issue1739_pack import pack_raw_tree  # noqa: F401
    from scripts.issue1739_wcrung_contexts import render_row_prompt  # noqa: F401
    from scripts.issue2091_stage_contexts import (  # noqa: F401
        RUNG_JOBS,
        RUNG_JOBS_BY_NAME,
        load_shard_rows,
    )

    print("[import-check] OK: all deferred imports resolved", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("dispatch", "job"), default="dispatch")
    ap.add_argument("--rungjob", default=None, help="--mode job: which rung-job to run")
    ap.add_argument("--gpu-id", default="0", help="--mode job: physical GPU (CVD-pinned by parent)")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/workspace/eps-issue-2091"),
        help="out-root; MUST resolve to the /workspace volume on RunPod, never the "
        "~50 GB container disk",
    )
    ap.add_argument("--store-root", type=Path, default=Path("/workspace/eps-issue-2091/store"))
    ap.add_argument("--stage-root", type=Path, default=Path("/workspace/eps-issue-2091/stage"))
    # UPLOAD_PREFIX_EXEMPT: issue-2091-specific decode leg; --hf-prefix overrides
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument("--dataset-revision", default=None, help="pin for the contexts staging read")
    ap.add_argument("--seed", type=int, default=20910)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--capture-batch-size", type=int, default=None)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="P0 PILOT: N contexts per rung-job through the IDENTICAL full path",
    )
    ap.add_argument("--only", default=None, help="comma-separated rung-job subset")
    ap.add_argument("--restage", action="store_true", help="re-download the contexts tree")
    ap.add_argument("--force", action="store_true", help="ignore per-job done records")
    ap.add_argument("--skip-upload", action="store_true", help="SMOKE ONLY: no Hub writes")
    ap.add_argument("--skip-capture", action="store_true", help="generation only (staging probe)")
    ap.add_argument("--gpu-hours-budgeted", type=float, default=9.0)
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the REAL branch, then exit 0",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    if args.import_check:
        return _import_check()

    if args.mode == "job":
        if not args.rungjob:
            raise SystemExit("--mode job requires --rungjob")
        run_job(args)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    t_start = time.time()
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.store_root.mkdir(parents=True, exist_ok=True)

    manifest = stage_contexts(args)
    n_gpus = len(visible_gpu_ids())
    records = dispatch(args, manifest)

    wall_h = (time.time() - t_start) / 3600.0
    gpu_hours = wall_h * n_gpus
    payload = build_results_payload(args, manifest, records, gpu_hours)

    is_pilot = args.limit is not None
    if is_pilot:
        write_sentinel("epm:smoke-result", payload, filename=PILOT_SENTINEL, gate="pilot")
        print(
            f"[phase=done] P0 pilot complete: jobs={len(records)} limit={args.limit} "
            f"gpu_hours={gpu_hours:.2f}",
            flush=True,
        )
    else:
        # Uploads verified by the fail-loud bulk _upload calls inside each job;
        # the upload-done sentinel is written only after every job's record
        # exists, and the results sentinel LAST.
        write_sentinel(
            "epm:progress",
            {
                "phase": "P2u",
                "n_jobs_uploaded": len(records),
                "hf_text_prefixes": [r["hf_text_prefix"] for r in records.values()],
                "hf_store_prefixes": [r["hf_store_prefix"] for r in records.values()],
            },
            filename=UPLOAD_DONE_SENTINEL,
            gate="phase",
        )
        write_sentinel("epm:results", payload, filename=RESULTS_SENTINEL)
        print(
            f"[phase=done] decode+capture complete: jobs={len(records)} "
            f"contexts={payload['eval_numbers']['n_contexts_decoded']} "
            f"gpu_hours={gpu_hours:.2f}",
            flush=True,
        )

    # Explicit exit: heavy C-extension teardown can rewrite rc during finalize
    # (the PyGILState_Release race) and abort a set -e dispatcher.
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
