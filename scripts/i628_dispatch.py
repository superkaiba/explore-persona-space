"""Issue #628 — marker-rig revision dispatcher (slot-aligned alive negatives).

Trains the four FRESH arms of the #628 rig contrast on the #537 context panel
and produces the four-float slot reads the analysis consumes:

- Phase 0 (prep):   prefetch pinned #537/#472 inputs (sha-asserted), rebuild
                    training mixes per separator variant (byte-identity
                    asserted vs the frozen #537 mixes), generate the 4
                    trained-negative contexts' eval-question base responses.
- Phase 1 (train):  56 fresh adapters (ARM_FLAGS below), 4-way
                    CUDA_VISIBLE_DEVICES cell-sharded subprocesses, band-stop
                    [5,12] overshoot-aware, stop-step + trajectory telemetry,
                    per-cell step-1 label-mask dump, HF adapter upload.
- Phase 2 (G-eval): four-float slot reads per fresh adapter over 34 columns
                    (30 grid + 4 trained-negative) x 32 questions; dual-slot
                    on separator-trained arms (sep_mode marker|plain).
- Phase 3 (reuse):  the 32 reused #537 adapters' 4 trained-negative columns,
                    gated by the 1-adapter apply-and-read parity probe.
- Phase 4 (onpol):  on-policy bystander read (Legacy + Full-revised arms,
                    4-context subset, 16 adapters): vLLM greedy own-answers
                    under 30 contexts x 10 questions, four-float reads at the
                    end of the OWN response (stripped at first marker token).
- matched-install-reread (post-hoc, NOT part of --phase all): consumes the
                    analysis-emitted matched_install_reread_spec.json (the §6
                    read-1 fallback, fired when arm-mean diagonal dials differ
                    by >2 nat) and re-reads each mismatched cell's matched
                    checkpoint on the default + 4 trained-negative columns via
                    the Phase-2 read path. Usage:
                    ``--phase matched-install-reread --spec <path>``.

Smoke IS the sweep with a cell subset: ``--arms rig_O_sep_deadneg
--train-cids sp_swe --seeds 42`` drives the identical subprocess shape, env
injection, logging, and teardown as the full sweep, and the subset threads
through EVERY phase (train, eval enumeration, gate check, uploads).

Pod-side contract (CLAUDE.md / poll_pipeline.py): emits ``[phase=<name>]``
log lines, a terminal ``[phase=done]``, and an end-of-run sentinel JSON at
``/workspace/logs/issue-628-<kind>-<ns>.json`` carrying
``sentinel_schema_version`` / ``kind`` / ``version`` / ``note``. NEVER shells
out to scripts/task.py.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i628_dispatch")

REPO = Path(__file__).resolve().parents[1]
QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
DATA_REPO = "superkaiba1/explore-persona-space-data"
MAX_NEW_TOKENS = 2048  # >= 2x longest trained completion (CLAUDE.md marker rule)
MARKER_ID = 83399
IM_END_ID = 151645
SEED = 42  # DATA seed: frozen mixes / caches are keyed by this forever

# ── Pinned reused-input revisions (artifact-reuse checks (e)/(f)) ────────────
# #537 frozen pools/contexts/responses/mixes on the HF data repo.
I537_DATA_REV = "db3662ae1d1ff4484ada027ac92a2658c4dec2e8"
I537_DATA_PREFIX = "issue537_context_generalization/data"
# #472 persona bank + on-policy R pools (bystander read).
I472_DATA_REV = "dfce94df6a3f326d0f4f366864321942842c7164"
# #537 marker adapters on the model repo (Phase-3 negative columns), per seed.
# Full SHAs resolved from HfApi.list_repo_commits (plan quotes the prefixes).
I537_ADAPTER_REV = {
    42: "0718c53058475cb8ee38c8f4802220cdde548672",
    1042: "dd577768816435b0b0541fd74e0936dd5ce92c8d",
}

# ── Static input roots (NON-rebinding: pinned parent inputs, smoke + real) ───
DATA537 = REPO / "data/issue_537"
INPUTS = REPO / "eval_results/issue_628/inputs"  # committed snapshot dir
FREEZE_MANIFEST = INPUTS / "i537_marker/freeze_manifest.json"
SNAPSHOT_MANIFEST = INPUTS / "i537_marker/MANIFEST.json"
I537_GCELLS_SNAPSHOT = INPUTS / "i537_marker/G_cells_marker"

# ── Generated-artifact roots (rebound to *_smoke in main() under --smoke) ────
GEN = Path(os.environ.get("I628_GEN_ROOT", str(REPO / "data/issue_628")))
OUT = Path(os.environ.get("I628_OUT_ROOT", str(REPO / "outputs/issue_628")))
EVAL = Path(os.environ.get("I628_EVAL_ROOT", str(REPO / "eval_results/issue_628")))

# ── Arms (plan §4.3 ARM_FLAGS, verbatim; every flag EXPLICIT per arm) ────────
ARM_FLAGS = {
    "rig_O_sep_deadneg": dict(
        marker_sep="\n\n",
        marker_suppress_at_post_response_slot=False,
        marker_negative_keep_trailing=False,
    ),
    "rig_Nplus_canonical": dict(
        marker_sep="",
        marker_suppress_at_post_response_slot=True,
        marker_negative_keep_trailing=True,
    ),
    "rig_S_nosep_deadneg": dict(
        marker_sep="",
        marker_suppress_at_post_response_slot=False,
        marker_negative_keep_trailing=False,
    ),
    "rig_F_sep_liveneg": dict(
        marker_sep="\n\n",
        marker_suppress_at_post_response_slot=True,
        marker_negative_keep_trailing=False,
    ),
}
FRESH_ARMS = tuple(ARM_FLAGS)
REUSE_ARM = "rig_N_i537_reuse"
FULL_GRID_ARMS = ("rig_O_sep_deadneg",)  # 16 train contexts
MINI_ARM_CIDS = ("sp_swe", "wc_short_advice", "icl_k8", "binst_marker")  # plan §4.1
ONPOLICY_ARMS = ("rig_O_sep_deadneg", "rig_Nplus_canonical")  # Phase 4 subset
DEFAULT_SEEDS = (42, 1042)
N_ONPOLICY_PERSONAS = 24  # persona-bank bystanders, fixed-seed sample

# Phase-3 parity probe (plan §4.3; values pinned in the committed snapshot).
PARITY_PROBE_CELL = ("sp_swe", "default", 42)
PARITY_TOL_NAT = 0.5

# Marker recipe — MARKER_TRAIN_KWARGS ported VERBATIM from i537_dispatch.py
# (plan §11: lr 5e-6 cosine w0.05, r32/alpha64/drop0.05 qkvo, marker-only
# loss, band-stop [5,12] overshoot-aware eval-5 min-10, 3-epoch ceiling).
# The three collator flags are OVERRIDDEN per arm from ARM_FLAGS — the
# experiment never relies on the new sft.py defaults.
MARKER_TRAIN_KWARGS = dict(
    lr=5e-6,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
    epochs=3,
    warmup_ratio=0.05,
    # max_length is set PER CELL from the builder's meta.json via _builder_cap()
    marker_only_loss=True,
    marker_suppress_at_post_response_slot=True,
    marker_im_end_token_id=IM_END_ID,
    marker_band_stop=True,
    marker_band_overshoot_stops=True,
    marker_band_eval_every_steps=5,
    marker_band_min_steps=10,
    report_to="wandb",
)

_CURRENT_PHASE = "init"
_PHASE_DIGIT_WORDS = str.maketrans({"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four"})


def phase_log(name: str) -> None:
    """Emit the [phase=...] line poll_pipeline.py parses (digits spelled out)."""
    global _CURRENT_PHASE
    safe = name.translate(_PHASE_DIGIT_WORDS)
    _CURRENT_PHASE = safe
    print(f"[phase={safe}]", flush=True)


def _log_dir() -> Path:
    override = os.environ.get("EPM_LOG_DIR")
    if override:
        d = Path(override)
    else:
        d = Path("/workspace/logs")
        if not d.exists():  # local VM (no /workspace) -> repo logs/
            d = REPO / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_sentinel(kind: str, note: str, *, version: int = 1, extra: dict | None = None) -> Path:
    """End-of-run sentinel carrying poll_pipeline's _SENTINEL_REQUIRED_KEYS."""
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "task_id": 628,
        "by": "i628_dispatch",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    if extra:
        payload.update(extra)
    slug = kind.replace(":", "_")
    out = _log_dir() / f"issue-628-{slug}-{time.time_ns()}.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("sentinel written: %s", out)
    return out


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


def _meta() -> dict:
    return {
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "data_seed": SEED,
    }


def _require_credentials() -> None:
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY missing"


def _tokenizer():
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i537_contexts import assert_marker_token

    tok = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
    assert_marker_token(tok)  # encode(" ※") == [83399], in-process (#537 incident rule)
    return tok


_REGISTRY_CACHE: tuple | None = None


def _registry_and_demos():
    """Load (and cache) the frozen #537 context registry + ICL demos."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        from explore_persona_space.experiments.i537_contexts import load_icl_demos, load_registry

        _REGISTRY_CACHE = (
            load_registry(DATA537 / "contexts/sampled_contexts.json"),
            load_icl_demos(DATA537 / "contexts/icl_demos.json"),
        )
    return _REGISTRY_CACHE


def _pool_path(stem: str, smoke: bool) -> Path:
    p = DATA537 / f"pools/{stem}.json"
    if smoke:
        sp = DATA537 / f"pools/{stem}.smoke.json"
        if sp.exists():
            return sp
    return p


def _marker_eval_questions(smoke: bool = False) -> list[str]:
    qs = json.loads(_pool_path("pool_marker_eval_32", smoke).read_text())["questions"]
    return qs[:4] if smoke else qs


def _marker_train_questions(smoke: bool = False) -> list[str]:
    return json.loads(_pool_path("pool_marker_train_300", smoke).read_text())["questions"]


def _shard_select(items: list, shard: str | None) -> list:
    if not shard:
        return items
    k, n = (int(x) for x in shard.split("/"))
    assert 0 <= k < n, shard
    return [it for i, it in enumerate(items) if i % n == k]


def _sep_variant(arm: str) -> str:
    """Separator variant per arm; the reused #537 arm is the no-sep canonical rig."""
    if arm == REUSE_ARM:
        return "nosep"
    return "sep" if ARM_FLAGS[arm]["marker_sep"] else "nosep"


def _mix_dir(variant: str) -> Path:
    return GEN / f"train_{variant}/marker"


def _cell_slug(arm: str, cid: str, seed: int) -> str:
    return f"{arm}_{cid}_seed{seed}"


def _cells(args) -> list[tuple[str, str, int]]:
    """The (arm, cid, seed) cell list every phase derives from (PASS_UNIFIED).

    --arms / --train-cids / --seeds subset uniformly; the full sweep is the
    default enumeration (rig_O on 16 contexts; the three mini-arms on 4).
    """
    from explore_persona_space.experiments.i537_contexts import train_cids_for

    grid_cids = train_cids_for("marker")
    arms = args.arms or list(FRESH_ARMS)
    for a in arms:
        assert a in FRESH_ARMS, f"unknown arm {a!r} (fresh arms: {FRESH_ARMS})"
    seeds = args.seeds
    cells = []
    for arm in arms:
        cids = list(grid_cids) if arm in FULL_GRID_ARMS else list(MINI_ARM_CIDS)
        if args.train_cids:
            cids = [c for c in cids if c in args.train_cids]
        for cid in cids:
            for seed in seeds:
                cells.append((arm, cid, seed))
    return cells


def _assert_negative_disjointness() -> None:
    """NEGATIVE_CIDS must be disjoint from train + holdout (registry, startup)."""
    from explore_persona_space.experiments.i537_contexts import (
        NEGATIVE_CIDS,
        eval_cids_for,
        train_cids_for,
    )

    overlap = set(NEGATIVE_CIDS) & (set(train_cids_for("marker")) | set(eval_cids_for("marker")))
    assert not overlap, f"negative panel overlaps train/holdout contexts: {sorted(overlap)}"


# ── HF upload helpers (eager per-phase; batched single-commit per tree) ──────


def _upload_tree(local_dir: Path, prefix: str, *, skip: bool) -> None:
    """Batch-upload every file under ``local_dir`` in ONE create_commit.

    One commit per phase tree (HF throttles ~256 commits/hour, #591), bounded
    5xx retry, then a fresh-listing count verification. Fail-loud.
    """
    if skip:
        logger.info("[upload] skipped (smoke/--skip-upload): %s", local_dir)
        return
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    if not files:
        logger.info("[upload] nothing under %s -- skip", local_dir)
        return
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    ops = [
        CommitOperationAdd(
            path_in_repo=f"{prefix}/{p.relative_to(local_dir)}", path_or_fileobj=str(p)
        )
        for p in files
    ]
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            api.create_commit(
                repo_id=DATA_REPO,
                repo_type="dataset",
                operations=ops,
                commit_message=f"issue-628: upload {prefix} ({len(ops)} files)",
            )
            break
        except Exception as e:  # bounded retry on transient Hub 5xx only
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and 500 <= status < 600 and attempt < 3:
                last_err = e
                wait = 2 ** (attempt + 2)
                logger.warning("[upload] Hub %s on %s -- retry in %ds", status, prefix, wait)
                time.sleep(wait)
                continue
            raise
    else:
        raise RuntimeError(f"upload of {prefix} failed after retries") from last_err
    on_hub = [
        f for f in api.list_repo_files(DATA_REPO, repo_type="dataset") if f.startswith(prefix)
    ]
    assert len(on_hub) >= len(ops), (
        f"upload verification FAILED for {prefix}: {len(on_hub)} on Hub < {len(ops)} local"
    )
    logger.info("[upload] %s: %d files verified on Hub", prefix, len(ops))


# ── vLLM helpers (ported from i537_dispatch.py) ──────────────────────────────


def _vllm_engine(max_model_len: int, *, enable_lora: bool = False):
    from vllm import LLM

    return LLM(
        model=QWEN_ID,
        dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85")),
        max_model_len=max_model_len,
        enforce_eager=False,
        enable_lora=enable_lora,
        max_lora_rank=32,
        seed=SEED,
    )


def _vllm_greedy(llm, rendered_prompts: list[str], max_tokens: int, *, lora_request=None):
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    kwargs = {"lora_request": lora_request} if lora_request is not None else {}
    outs = llm.generate(rendered_prompts, params, **kwargs)
    results = [
        {"response": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason} for o in outs
    ]
    assert len(results) == len(rendered_prompts), (len(results), len(rendered_prompts))
    return results


def _teardown_vllm(llm) -> None:
    """vLLM teardown + orphan-worker reap (gotchas.md)."""
    import gc

    import psutil
    import torch

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    me = psutil.Process()
    children = me.children(recursive=True)
    for c in children:
        try:
            c.terminate()
        except psutil.NoSuchProcess:
            continue
    _gone, alive = psutil.wait_procs(children, timeout=10)
    for c in alive:
        try:
            c.kill()
        except psutil.NoSuchProcess:
            continue
    logger.info("[vllm-teardown] reaped %d child processes", len(children))


# ── Worker-wave launcher (4-way CUDA_VISIBLE_DEVICES cell sharding) ──────────


def _gpu_pool() -> list[str]:
    """Physical GPU ids available to THIS process (respects parent CVD narrowing)."""
    parent = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent is not None and parent != "":
        return parent.split(",")
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
            env=None,  # epm-lint: subprocess-env-inherit -- read-only GPU probe, no creds
        ).stdout
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
        return [str(i) for i in range(max(1, n))]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ["0"]


def _run_wave(args, phase: str, step: str, n_items: int) -> None:
    """Spawn worker subprocesses of THIS script for one GPU step.

    Same subprocess shape for 1 cell as for 56 (smoke = sweep): the launcher
    pins ``CUDA_VISIBLE_DEVICES=<gpu>`` in each worker's env (the in-process
    ``gpu_id`` clobber alone is defeated by import-time cuInit — #545) and
    each worker selects its cells via ``--worker-shard k/n``. ``phase`` is the
    CURRENT phase id ("1".."4"), never ``args.phase`` (which may be "all").
    """
    pool = _gpu_pool()
    workers = min(args.workers or len(pool), max(1, n_items), len(pool))
    procs = []
    for k in range(workers):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": pool[k % len(pool)]}
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            phase,
            "--step",
            step,
            "--worker-shard",
            f"{k}/{workers}",
            "--seeds",
            ",".join(str(s) for s in args.seeds),
        ]
        if args.arms:
            cmd += ["--arms", ",".join(args.arms)]
        if args.train_cids:
            cmd += ["--train-cids", ",".join(args.train_cids)]
        if args.smoke:
            cmd.append("--smoke")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.skip_upload:
            cmd.append("--skip-upload")
        if args.enforce_gate:
            cmd.append("--enforce-gate")
        logger.info("[wave:%s] worker %d/%d on GPU %s", step, k, workers, pool[k % len(pool)])
        procs.append(subprocess.Popen(cmd, cwd=REPO, env=env))
    rcs = [p.wait() for p in procs]
    bad = [(k, rc) for k, rc in enumerate(rcs) if rc != 0]
    if bad:
        raise SystemExit(f"[wave:{step}] worker failures: {bad}")


# ── Phase 0: prefetch + mixes + neg-context eval responses ───────────────────


def _prefetch_file(rel: str, dest: Path, *, revision: str, repo: str = DATA_REPO) -> Path:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo, rel, repo_type="dataset", revision=revision)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dest)
    return dest


def _assert_manifest_sha(dest: Path, man_key: str, manifest: dict, *, source: str) -> None:
    """Assert ``dest`` matches the freeze-manifest sha256 (no-op when the
    manifest does not cover ``man_key`` — responses/mixes are revision-pinned)."""
    if man_key not in manifest:
        return
    got = hashlib.sha256(dest.read_bytes()).hexdigest()
    assert got == manifest[man_key], (
        f"sha256 mismatch for {man_key}: {source} {got} != freeze manifest "
        f"{manifest[man_key]} -- pinned-content drift, refusing to run"
    )


def _prefetch_inputs(*, static_only: bool = False) -> None:
    """Pinned-revision prefetch of every #537/#472 input + sha256 asserts.

    sha256 is asserted for every prefetched file the freeze manifest covers
    (contexts + pools); responses / mixes are pinned by the HF revision.
    Uses list_repo_files + per-file hf_hub_download (snapshot_download's
    allow_patterns silently truncates past ~8k siblings). ``static_only``
    (smoke): fetch just the frozen pools + contexts (smoke regenerates its
    own tiny response caches under the smoke GEN root).
    """
    from huggingface_hub import list_repo_files

    manifest = json.loads(FREEZE_MANIFEST.read_text())["artifact_sha256"]
    all_files = list_repo_files(DATA_REPO, repo_type="dataset", revision=I537_DATA_REV)
    wanted_dirs = (
        ["contexts", "pools"]
        if static_only
        else ["contexts", "pools", "responses", "responses_eval", "train/marker"]
    )
    fetched = 0
    for sub in wanted_dirs:
        prefix = f"{I537_DATA_PREFIX}/{sub}/"
        rels = [f for f in all_files if f.startswith(prefix)]
        assert rels, f"no files under {prefix} at pinned revision {I537_DATA_REV}"
        for rel in rels:
            tail = rel[len(f"{I537_DATA_PREFIX}/") :]
            if sub == "train/marker":
                dest = GEN / "i537_train_marker" / Path(rel).name
            elif sub in ("responses", "responses_eval"):
                dest = GEN / tail  # consumed (and, in smoke, regenerated) under GEN
            else:
                dest = DATA537 / tail  # frozen INPUTS: pools + contexts, shared smoke/real
            man_key = f"data/issue_537/{tail}"
            if dest.exists():
                # Cache hit: STILL assert the manifest sha256 — a stale or
                # locally-mutated cached copy must not silently bypass the
                # content pin (the skip used to jump straight to `continue`).
                _assert_manifest_sha(dest, man_key, manifest, source="local cache")
                fetched += 1
                continue
            _prefetch_file(rel, dest, revision=I537_DATA_REV)
            fetched += 1
            _assert_manifest_sha(dest, man_key, manifest, source="HF mirror")
    # #472 persona bank + R_eval for Phase 4.
    i472 = INPUTS / "i472"
    for rel, name in (
        ("issue472_neg_geometry/geometry/persona_bank.json", "persona_bank.json"),
        ("issue472_neg_geometry/on_policy_R/R_eval.json", "R_eval.json"),
    ):
        dest = i472 / name
        if not dest.exists():
            _prefetch_file(rel, dest, revision=I472_DATA_REV)
    logger.info("[p0] prefetch complete: %d #537 files + 2 #472 files", fetched)


def _builder_cmd(variant: str, cid: str, smoke: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO / "scripts/i537_build_training_data.py"),
        "--behavior",
        "marker",
        "--train-cid",
        cid,
        "--seed",
        str(SEED),
        "--responses",
        str(GEN / "responses"),
        "--out-root",
        str(GEN / f"train_{variant}"),
        "--sampled-contexts",
        str(DATA537 / "contexts/sampled_contexts.json"),
        "--icl-demos",
        str(DATA537 / "contexts/icl_demos.json"),
        "--questions",
        str(_pool_path("pool_marker_train_300", smoke)),
        "--marker-sep",
        "\n\n" if variant == "sep" else "",
    ]
    if smoke:
        cmd.append("--smoke")
    return cmd


def _assert_mix_byte_identity(cid: str) -> None:
    """Rebuilt NO-SEP mix must byte-match the frozen #537 mix; the SEP mix
    must differ ONLY by the inserted separator (positives) / be identical
    (negatives)."""
    from explore_persona_space.experiments.i537_contexts import MARKER_TEXT

    frozen = GEN / "i537_train_marker" / f"{cid}_seed{SEED}.jsonl"
    nosep = _mix_dir("nosep") / f"{cid}_seed{SEED}.jsonl"
    assert frozen.read_bytes() == nosep.read_bytes(), (
        f"byte-identity FAILED: rebuilt no-sep mix {nosep} != frozen #537 mix {frozen} -- "
        "the builder or its inputs drifted; the revised-rig reuse is not licensed."
    )
    sep = _mix_dir("sep") / f"{cid}_seed{SEED}.jsonl"
    if not sep.exists():
        return
    nosep_lines = nosep.read_text().splitlines()
    sep_lines = sep.read_text().splitlines()
    assert len(nosep_lines) == len(sep_lines), (cid, len(nosep_lines), len(sep_lines))
    for i, (a, b) in enumerate(zip(nosep_lines, sep_lines, strict=True)):
        ra, rb = json.loads(a), json.loads(b)
        ca, cb = ra["completion"][0]["content"], rb["completion"][0]["content"]
        if ca.endswith(MARKER_TEXT):  # positive row
            expected = ca[: -len(MARKER_TEXT)] + "\n\n" + MARKER_TEXT
            assert cb == expected, f"{cid} row {i}: sep-arm positive differs beyond the separator"
            assert ra["prompt"] == rb["prompt"], f"{cid} row {i}: prompt drift between variants"
        else:  # negative row
            assert a == b, f"{cid} row {i}: sep-arm NEGATIVE row differs (must be identical)"


def _audit_realized_negatives(cid: str, variant: str, smoke: bool) -> None:
    """Disjointness against the REALIZED mix: every marker-less row's prompt
    must re-render from a NEGATIVE_CIDS context on its own question, and the
    negative panel must not collide with train/holdout (asserted at startup).
    """
    from explore_persona_space.experiments.i537_contexts import (
        MARKER_TEXT,
        NEGATIVE_CIDS,
        build_messages,
    )

    registry, demos = _registry_and_demos()
    mix = _mix_dir(variant) / f"{cid}_seed{SEED}.jsonl"
    n_neg = 0
    for line in mix.read_text().splitlines():
        row = json.loads(line)
        if row["completion"][0]["content"].endswith(MARKER_TEXT):
            continue
        n_neg += 1
        q = row["prompt"][-1]["content"]
        renders = [
            build_messages(registry[nc], q, behavior="marker", icl_demos=demos)
            for nc in NEGATIVE_CIDS
        ]
        # _casualize in build_messages may rewrite the user turn; match on the
        # full prompt list against the 4 candidate renders (same q via the row).
        ok = any(row["prompt"] == r for r in renders)
        if not ok:
            qs = [r[-1]["content"] for r in renders]
            ok = any(
                row["prompt"][:-1] == r[:-1] and row["prompt"][-1]["content"] in qs for r in renders
            )
        assert ok, (
            f"realized negative row in {mix.name} does not re-render from any "
            f"NEGATIVE_CIDS context -- panel/source contamination?"
        )
    assert n_neg > 0, f"no negative rows found in {mix} -- contrastive mix is malformed"
    logger.info("[p0] %s/%s: %d realized negative rows audited", variant, cid, n_neg)


def _neg_eval_cache_dir() -> Path:
    return EVAL / "inputs/responses_eval_neg"


def _gen_neg_eval_responses(args) -> None:
    """Base-greedy eval-question responses for the 4 trained-negative contexts
    (they have train-pool caches but no eval-question responses in #537)."""
    from explore_persona_space.experiments.i537_cache import cache_covers, write_response_cache
    from explore_persona_space.experiments.i537_contexts import NEGATIVE_CIDS, build_prompt

    questions = _marker_eval_questions(args.smoke)
    out_dir = _neg_eval_cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    neg_cids = list(NEGATIVE_CIDS)[:1] if args.smoke else list(NEGATIVE_CIDS)
    todo = [
        c
        for c in neg_cids
        if not cache_covers(
            out_dir / f"{c}.json",
            questions,
            smoke=args.smoke,
            behavior="marker",
            expected_pool=questions,
        )
    ]
    if not todo:
        logger.info("[p0] all %d neg-context eval caches present -- skip", len(neg_cids))
        return
    registry, demos = _registry_and_demos()
    tok = _tokenizer()
    llm = _vllm_engine(16384)
    try:
        for cid in todo:
            prompts = [
                build_prompt(registry[cid], q, tok, behavior="marker", icl_demos=demos)
                for q in questions
            ]
            results = _vllm_greedy(llm, prompts, MAX_NEW_TOKENS)
            trunc = sum(1 for r in results if r["finish_reason"] != "stop")
            payload = {
                **_meta(),
                "cid": cid,
                "model": QWEN_ID,
                "max_new_tokens": MAX_NEW_TOKENS,
                "gen_truncated_frac": trunc / len(results),
                "questions": {q: r for q, r in zip(questions, results, strict=True)},
            }
            write_response_cache(
                out_dir / f"{cid}.json", payload, questions, smoke=args.smoke, behavior="marker"
            )
            logger.info("[p0] neg eval cache %s (%d q)", cid, len(questions))
    finally:
        _teardown_vllm(llm)


def _gen_smoke_response_caches(args) -> None:
    """--smoke only: tiny base response caches (train + eval pools) under the
    smoke GEN root, mirroring the parent dispatcher's smoke generation."""
    from explore_persona_space.experiments.i537_cache import cache_covers, write_response_cache
    from explore_persona_space.experiments.i537_contexts import (
        NEGATIVE_CIDS,
        build_prompt,
        eval_cids_for,
    )

    cells = _cells(args)
    train_cids = sorted({cid for _, cid, _ in cells})
    train_q = _marker_train_questions(smoke=True)
    eval_q = _marker_eval_questions(smoke=True)
    eval_cids = eval_cids_for("marker")[:2]
    registry, demos = _registry_and_demos()
    tok = _tokenizer()
    jobs: list[tuple[Path, str, list[str]]] = []
    for cid in [*train_cids, *NEGATIVE_CIDS]:
        jobs.append((GEN / "responses" / f"{cid}.json", cid, train_q))
    for cid in eval_cids:
        jobs.append((GEN / "responses_eval" / f"{cid}.json", cid, eval_q))
    todo = [
        j
        for j in jobs
        if not cache_covers(j[0], j[2], smoke=True, behavior="marker", expected_pool=j[2])
    ]
    if not todo:
        return
    llm = _vllm_engine(16384)
    try:
        for out_p, cid, qs in todo:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            prompts = [
                build_prompt(registry[cid], q, tok, behavior="marker", icl_demos=demos) for q in qs
            ]
            results = _vllm_greedy(llm, prompts, MAX_NEW_TOKENS)
            payload = {
                **_meta(),
                "cid": cid,
                "model": QWEN_ID,
                "max_new_tokens": MAX_NEW_TOKENS,
                "gen_truncated_frac": 0.0,
                "questions": {q: r for q, r in zip(qs, results, strict=True)},
            }
            write_response_cache(out_p, payload, qs, smoke=True, behavior="marker")
            logger.info("[p0-smoke] cache %s (%d q)", out_p.name, len(qs))
    finally:
        _teardown_vllm(llm)


def phase0(args) -> None:
    cells = _cells(args)
    variants = sorted({_sep_variant(arm) for arm, _, _ in cells})
    build_cids = sorted({cid for _, cid, _ in cells})
    if args.dry_run:
        phase_log("p0_prep")
        logger.info("[p0][dry-run] variants=%s cids=%s", variants, build_cids)
        return

    phase_log("p0_prefetch")
    if not args.smoke:
        _prefetch_inputs()
    else:
        # Smoke shares the frozen pools/contexts (sha-asserted, pinned) but
        # regenerates its own TINY response caches under the smoke GEN root —
        # the real frozen caches carry a smoke=False cache signature.
        _prefetch_inputs(static_only=True)
        _gen_smoke_response_caches(args)

    phase_log("p0_build")
    # Pre-warm tokenizer cache ONCE (one online round-trip) so the build
    # subprocesses can run with HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1
    # set. transformers' AutoTokenizer.from_pretrained calls
    # `_patch_mistral_regex` -> `is_base_mistral(model_id)` -> `model_info()`
    # on every invocation regardless of cache state. With 2 variants x 16
    # cids each making ~80 hub-API calls, the workload trips HF's 2500 req /
    # 5-min rate limit (incident #628, 2026-06-13). The pre-warm call
    # below populates the local cache and pays exactly one model_info; the
    # subprocesses then take the offline path which short-circuits
    # model_info entirely. Marker token id 83399 was verified to survive
    # both code paths in the same forward.
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
    build_env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for variant in variants:
        for cid in build_cids:
            mix = _mix_dir(variant) / f"{cid}_seed{SEED}.jsonl"
            if mix.exists():
                continue
            subprocess.run(
                _builder_cmd(variant, cid, args.smoke), check=True, cwd=REPO, env=build_env
            )
    for cid in build_cids:
        if not args.smoke and (_mix_dir("nosep") / f"{cid}_seed{SEED}.jsonl").exists():
            _assert_mix_byte_identity(cid)
        for variant in variants:
            _audit_realized_negatives(cid, variant, args.smoke)

    phase_log("p0_negevalgen")
    _gen_neg_eval_responses(args)

    skip = args.smoke or args.skip_upload or args.dry_run
    for variant in variants:
        _upload_tree(
            _mix_dir(variant), f"issue628_rig_revision/data/train_{variant}/marker", skip=skip
        )
    _upload_tree(_neg_eval_cache_dir(), "issue628_rig_revision/data/responses_eval_neg", skip=skip)


# ── Phase 1: training ────────────────────────────────────────────────────────


class _FinalStepRecorder:
    """TrainerCallback recording the final global step (lazy TrainerCallback
    inheritance so this module imports without transformers)."""

    def __new__(cls):
        from transformers import TrainerCallback

        class _Impl(TrainerCallback):
            final_step: int = -1

            def on_train_end(self, args, state, control, **kwargs):
                self.final_step = int(state.global_step)

        return _Impl()


def _builder_cap(variant: str, cid: str) -> int:
    meta = _mix_dir(variant) / f"{cid}_seed{SEED}.meta.json"
    return int(json.loads(meta.read_text())["max_length"]) + 128


def _dump_label_masks(arm: str, cid: str, seed: int) -> Path:
    """Step-1-equivalent label-mask dump: run the REAL collator with the arm's
    flags on the first positive + first negative row of the REAL mix and
    assert the arm's expected loss-token ids (smoke-verifiable telemetry)."""
    import torch

    from explore_persona_space.experiments.i537_contexts import MARKER_TEXT
    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    flags = ARM_FLAGS[arm]
    tok = _tokenizer()
    mix = _mix_dir(_sep_variant(arm)) / f"{cid}_seed{SEED}.jsonl"
    rows = [json.loads(line) for line in mix.read_text().splitlines()]
    pos_row = next(r for r in rows if r["completion"][0]["content"].endswith(MARKER_TEXT))
    neg_row = next(r for r in rows if not r["completion"][0]["content"].endswith(MARKER_TEXT))

    def _feat(row: dict) -> dict:
        full = tok.apply_chat_template(
            row["prompt"] + row["completion"], tokenize=True, add_generation_prompt=False
        )
        prefix = tok.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        assert full[: len(prefix)] == prefix
        ids = torch.tensor(full, dtype=torch.long)
        labels = ids.clone()
        labels[: len(prefix)] = -100
        return {"input_ids": ids, "labels": labels}

    def _stack(feats):
        assert len(feats) == 1
        return {
            "input_ids": feats[0]["input_ids"].unsqueeze(0),
            "labels": feats[0]["labels"].unsqueeze(0).clone(),
        }

    collator = MarkerOnlyDataCollator(
        inner_collator=_stack,
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        suppress_at_post_response_slot=flags["marker_suppress_at_post_response_slot"],
        im_end_token_id=IM_END_ID,
        negative_keep_trailing=flags["marker_negative_keep_trailing"],
    )
    dump = {}
    for name, row in (("positive", pos_row), ("negative", neg_row)):
        feat = _feat(row)
        out = collator([feat])
        kept_pos = (out["labels"][0] != -100).nonzero(as_tuple=True)[0].tolist()
        kept_ids = [int(feat["input_ids"][p].item()) for p in kept_pos]
        dump[name] = {"kept_positions": kept_pos, "kept_token_ids": kept_ids}
    # Arm-specific expectations (the suppression / keep-trailing branches are
    # exercised here, not paper mitigations):
    assert MARKER_ID in dump["positive"]["kept_token_ids"], dump["positive"]
    neg_ids = dump["negative"]["kept_token_ids"]
    if flags["marker_suppress_at_post_response_slot"]:
        assert neg_ids[0] == IM_END_ID, f"{arm}: negative loss not at <|im_end|>: {neg_ids}"
        expected_n = 2 if flags["marker_negative_keep_trailing"] else 1
        assert len(neg_ids) == expected_n, f"{arm}: negative kept {neg_ids}, want n={expected_n}"
    else:
        assert len(neg_ids) == 1 and neg_ids[0] != IM_END_ID, (
            f"{arm}: suppress-OFF negative must keep ONLY the trailing token: {neg_ids}"
        )
    assert collator._pos_count > 0 and collator._neg_count > 0, "collator counters not exercised"
    out_dir = EVAL / "p1/label_mask_dumps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f"{_cell_slug(arm, cid, seed)}.json"
    out_p.write_text(
        json.dumps({**_meta(), "arm": arm, "cid": cid, "seed": seed, **dump}, indent=1)
    )
    logger.info("[p1] %s label-mask dump OK: neg kept ids=%s", _cell_slug(arm, cid, seed), neg_ids)
    return out_p


def _adapter_dir(arm: str, cid: str, seed: int) -> Path:
    return OUT / "adapters" / _cell_slug(arm, cid, seed)


def _hf_adapter_subfolder(arm: str, cid: str, seed: int) -> str:
    return f"adapters/issue_628/{_cell_slug(arm, cid, seed)}"


def _verify_adapter_on_hub(subfolder: str) -> None:
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(HF_MODEL_REPO) if f.startswith(subfolder)]
    assert any(f.endswith("adapter_model.safetensors") for f in files), (
        f"Adapter NOT verified on Hub under {subfolder!r}."
    )


def _train_cell(arm: str, cid: str, seed: int, *, smoke: bool) -> None:
    slug = _cell_slug(arm, cid, seed)
    out_dir = _adapter_dir(arm, cid, seed)
    stop_p = EVAL / "p1/stop_steps" / f"{slug}.json"
    if (out_dir / "adapter_model.safetensors").exists() and stop_p.exists():
        logger.info("[p1-train] %s already trained -- skip", slug)
        return
    _dump_label_masks(arm, cid, seed)

    from explore_persona_space.experiments.i537_contexts import MARKER_TEXT
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    variant = _sep_variant(arm)
    data_path = _mix_dir(variant) / f"{cid}_seed{SEED}.jsonl"
    traj_dir = EVAL / "p1/band_trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(MARKER_TRAIN_KWARGS)
    flags = ARM_FLAGS[arm]
    kwargs["marker_suppress_at_post_response_slot"] = flags["marker_suppress_at_post_response_slot"]
    kwargs["marker_negative_keep_trailing"] = flags["marker_negative_keep_trailing"]
    # CRITICAL: TrainLoraConfig.marker_text defaults to the DEPRECATED "[ZLT]"
    # -- without this thread the collator finds zero marker positions (#537).
    kwargs["marker_text"] = MARKER_TEXT
    kwargs["max_length"] = _builder_cap(variant, cid)
    kwargs["marker_band_trajectory_path"] = str(traj_dir / f"{slug}.json")
    # Per-5-step checkpoints for the §6 matched-install fallback read.
    kwargs["save_strategy"] = "steps"
    kwargs["save_steps"] = 5
    kwargs["save_total_limit"] = None
    kwargs["save_only_model"] = True
    if smoke:
        kwargs["epochs"] = 1
        kwargs["max_steps"] = 2
        kwargs["marker_band_stop"] = False
    cfg = TrainLoraConfig(
        seed=seed,
        gpu_id=0,  # CUDA_VISIBLE_DEVICES is pinned per worker by the launcher
        run_name=f"issue628_{slug}",
        hf_upload=not smoke,
        hf_path_in_repo=_hf_adapter_subfolder(arm, cid, seed),
        **kwargs,
    )
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
    recorder = _FinalStepRecorder()
    try:
        train_lora(QWEN_ID, str(data_path), str(out_dir), cfg=cfg, callbacks=[recorder])
    finally:
        # One WandB run PER CELL (HF Trainer reuses an open wandb.run).
        import wandb

        if wandb.run is not None:
            wandb.finish()
    if not smoke:
        _verify_adapter_on_hub(_hf_adapter_subfolder(arm, cid, seed))
        _upload_checkpoints(arm, cid, seed, out_dir)
    assert recorder.final_step > 0, f"stop step not recorded for {slug}"
    stop_p.parent.mkdir(parents=True, exist_ok=True)
    traj_p = traj_dir / f"{slug}.json"
    final_delta = None
    if traj_p.exists():
        # marker_band_trajectory_v1 schema (eval/callbacks.py _write_trajectory):
        # per-probe dicts under "records" + a parallel "delta_nats" array. The
        # previous reader looked for "points"/"trajectory" keys the schema
        # never writes, silently leaving final_band_delta_nats=None and
        # defanging the §7(d) gate check.
        traj = json.loads(traj_p.read_text())
        records = traj.get("records") or []
        deltas = [r["delta_nats"] for r in records] or traj.get("delta_nats") or []
        if deltas:
            final_delta = float(deltas[-1])
    tmp = stop_p.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(
            {
                **_meta(),
                "arm": arm,
                "cid": cid,
                "seed": seed,
                "stop_step": recorder.final_step,
                "final_band_delta_nats": final_delta,
            }
        )
    )
    tmp.replace(stop_p)
    logger.info("[p1-train] %s stop_step=%d delta=%s", slug, recorder.final_step, final_delta)


def _upload_checkpoints(arm: str, cid: str, seed: int, out_dir: Path) -> None:
    """Per-5-step checkpoint dirs -> the cell's HF adapter subfolder."""
    ckpts = sorted(out_dir.glob("checkpoint-*"))
    if not ckpts:
        return
    from huggingface_hub import CommitOperationAdd, HfApi

    sub = _hf_adapter_subfolder(arm, cid, seed)
    ops = []
    for ck in ckpts:
        for p in ck.rglob("*"):
            if p.is_file() and (p.name.startswith("adapter_") or p.name.endswith(".json")):
                ops.append(
                    CommitOperationAdd(
                        path_in_repo=f"{sub}/{ck.name}/{p.relative_to(ck)}",
                        path_or_fileobj=str(p),
                    )
                )
    HfApi().create_commit(
        repo_id=HF_MODEL_REPO,
        operations=ops,
        commit_message=f"issue-628: {len(ckpts)} checkpoints for {sub}",
    )
    logger.info("[p1-train] %s: %d checkpoint dirs uploaded", sub, len(ckpts))


def phase1(args) -> None:
    cells = _cells(args)
    if args.worker_shard:
        my_cells = _shard_select(cells, args.worker_shard)
        logger.info(
            "[p1-worker %s] cells=%s", args.worker_shard, [_cell_slug(*c) for c in my_cells]
        )
        if args.dry_run:
            for c in my_cells:
                logger.info("[p1-worker][dry-run] would train %s", _cell_slug(*c))
            return
        for arm, cid, seed in my_cells:
            _train_cell(arm, cid, seed, smoke=args.smoke)
        return
    phase_log("p1_train")
    _run_wave(args, "1", "train", len(cells))
    skip = args.smoke or args.skip_upload or args.dry_run
    for sub in ("p1/stop_steps", "p1/band_trajectories", "p1/label_mask_dumps"):
        _upload_tree(EVAL / sub, f"issue628_rig_revision/eval_results/{sub}", skip=skip)


# ── Phase 2: G-eval (four-float slot reads, dual-slot on sep arms) ───────────


def _eval_columns(args) -> list[str]:
    from explore_persona_space.experiments.i537_contexts import NEGATIVE_CIDS, eval_cids_for

    grid = eval_cids_for("marker")
    negs = list(NEGATIVE_CIDS)
    if args.smoke:
        grid, negs = grid[:2], negs[:1]
    return [*grid, *negs]


def _eval_contexts_for(cid: str, questions: list[str], *, sep: str, smoke: bool) -> list[str]:
    """Chat-templated prompt + frozen base-greedy R (+ optional separator),
    tokenized WHOLESALE downstream so BPE fusion matches training."""
    from explore_persona_space.experiments.i537_cache import read_response_cache
    from explore_persona_space.experiments.i537_contexts import NEGATIVE_CIDS, build_prompt

    registry, demos = _registry_and_demos()
    tok = _tokenizer()
    cache_dir = _neg_eval_cache_dir() if cid in NEGATIVE_CIDS else GEN / "responses_eval"
    cache = read_response_cache(
        cache_dir / f"{cid}.json",
        questions,
        smoke=smoke,
        behavior="marker",
        expected_pool=questions,
    )["questions"]
    return [
        build_prompt(registry[cid], q, tok, behavior="marker", icl_demos=demos)
        + cache[q]["response"]
        + sep
        for q in questions
    ]


def _batch_for(cid: str) -> int:
    return 4 if cid in ("wc_xlong_ho", "wc_xxlong_ho") else 32


def _base_slot_path(cid: str, sep_mode: str) -> Path:
    suffix = "" if sep_mode == "plain" else "__sep"
    return EVAL / "marker_base_slots" / f"{cid}{suffix}.json"


def _load_hf_base():
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()


def _score_base_column(model, cid: str, questions: list[str], sep_mode: str, smoke: bool) -> None:
    from explore_persona_space.experiments.i537_marker_eval import score_marker_slots

    out_p = _base_slot_path(cid, sep_mode)
    if out_p.exists():
        return
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sep = "" if sep_mode == "plain" else "\n\n"
    tok = _tokenizer()
    stats, _ = score_marker_slots(
        model,
        tok,
        _eval_contexts_for(cid, questions, sep=sep, smoke=smoke),
        marker_id=MARKER_ID,
        eos_token_id=IM_END_ID,
        batch_size=_batch_for(cid),
    )
    out_p.write_text(
        json.dumps(
            {**_meta(), "cid": cid, "sep_mode": sep_mode, "questions": questions, "stats": stats},
            indent=1,
        )
    )
    logger.info("[p2-base] %s (%s): %d slots", cid, sep_mode, len(stats))


def _cell_read(
    peft_model,
    arm: str,
    cid: str,
    seed: int,
    eval_cid: str,
    questions: list[str],
    *,
    sep_mode: str,
    smoke: bool,
    out_dir: Path | None = None,
    fname_tag: str = "",
    extra: dict | None = None,
) -> None:
    """Four-float slot read for one (adapter, eval column) cell.

    ``out_dir`` / ``fname_tag`` / ``extra`` let the matched-install re-read
    phase reuse this exact read path with a distinct output tree and a
    ``checkpoint_step`` field; defaults are byte-identical to the Phase-2
    behavior."""
    import numpy as np

    from explore_persona_space.experiments.i537_marker_eval import score_marker_slots

    g_dir = out_dir if out_dir is not None else EVAL / "G_cells" / arm
    g_dir.mkdir(parents=True, exist_ok=True)
    suffix = "__plain" if sep_mode == "plain" and _sep_variant(arm) == "sep" else ""
    cell_p = g_dir / f"{cid}__{eval_cid}__seed{seed}{suffix}{fname_tag}.json"
    if cell_p.exists():
        return
    sep = "\n\n" if sep_mode == "marker" and _sep_variant(arm) == "sep" else ""
    tok = _tokenizer()
    ctxs = _eval_contexts_for(eval_cid, questions, sep=sep, smoke=smoke)
    t0 = time.time()
    stats, _ = score_marker_slots(
        peft_model,
        tok,
        ctxs,
        marker_id=MARKER_ID,
        eos_token_id=IM_END_ID,
        batch_size=_batch_for(eval_cid),
    )
    base_mode = "plain" if sep == "" else "sep"
    base = json.loads(
        _base_slot_path(eval_cid, "plain" if base_mode == "plain" else "marker").read_text()
    )["stats"]
    per_q = [
        {
            "question": q,
            "trained": s,
            "base": b,
            "delta_logp": s["logp"] - b["logp"],
            "delta_z_marker": s["z_marker"] - b["z_marker"],
            "delta_eos_margin": (s["z_marker"] - s["z_eos"]) - (b["z_marker"] - b["z_eos"]),
        }
        for q, s, b in zip(questions, stats, base, strict=True)
    ]
    cell = {
        **_meta(),
        "behavior": "marker",
        "arm": arm,
        "sep_mode": sep_mode,
        "train_cid": cid,
        "eval_cid": eval_cid,
        "seed": seed,
        "n_questions": len(questions),
        "g_mean_delta_logp": float(np.mean([r["delta_logp"] for r in per_q])),
        "g_mean_delta_z_marker": float(np.mean([r["delta_z_marker"] for r in per_q])),
        "g_mean_delta_eos_margin": float(np.mean([r["delta_eos_margin"] for r in per_q])),
        "emission_rate_trained": float(np.mean([s["argmax_is_marker"] for s in stats])),
        "emission_rate_base": float(np.mean([b["argmax_is_marker"] for b in base])),
        "qs_per_sec": len(questions) / max(time.time() - t0, 1e-9),
        "per_question": per_q,
        **(extra or {}),
    }
    cell_p.write_text(json.dumps(cell, indent=1))
    logger.info(
        "[p2-cells] %s %s->%s (%s): dlogP=%.2f",
        arm,
        cid,
        eval_cid,
        sep_mode,
        cell["g_mean_delta_logp"],
    )


def _ensure_local_adapter(arm: str, cid: str, seed: int) -> Path:
    d = _adapter_dir(arm, cid, seed)
    if (d / "adapter_model.safetensors").exists():
        return d
    from huggingface_hub import hf_hub_download

    sub = _hf_adapter_subfolder(arm, cid, seed)
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        p = hf_hub_download(HF_MODEL_REPO, f"{sub}/{fn}")
        shutil.copyfile(p, d / fn)
    return d


def _gate_check(arm: str, cid: str, seed: int, *, enforce: bool) -> None:
    """Smoke-gate criterion (d): off-line diagonal G-eval vs the in-loop band
    read, |diff| <= 1 nat (raise only under --enforce-gate; always recorded)."""
    slug = _cell_slug(arm, cid, seed)
    stop_p = EVAL / "p1/stop_steps" / f"{slug}.json"
    cell_p = EVAL / "G_cells" / arm / f"{cid}__{cid}__seed{seed}.json"
    if not (stop_p.exists() and cell_p.exists()):
        return
    in_loop = json.loads(stop_p.read_text()).get("final_band_delta_nats")
    offline = json.loads(cell_p.read_text())["g_mean_delta_logp"]
    out_dir = EVAL / "p2/gate_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    diff = None if in_loop is None else abs(offline - in_loop)
    (out_dir / f"{slug}.json").write_text(
        json.dumps(
            {
                **_meta(),
                "arm": arm,
                "cid": cid,
                "seed": seed,
                "in_loop_delta_nats": in_loop,
                "offline_diagonal_delta_logp": offline,
                "abs_diff": diff,
                "note": "in-loop probe is on TRAIN questions; offline diagonal on EVAL questions",
            },
            indent=1,
        )
    )
    if enforce and diff is not None and diff > 1.0:
        raise SystemExit(
            f"[gate] {slug}: offline diagonal ({offline:+.2f}) vs in-loop band read "
            f"({in_loop:+.2f}) differ by {diff:.2f} nat > 1.0 -- smoke gate FAILED."
        )


def phase2(args) -> None:
    cells = _cells(args)
    columns = _eval_columns(args)
    questions = _marker_eval_questions(args.smoke)
    if args.worker_shard and args.step == "base":
        need_sep = any(_sep_variant(arm) == "sep" for arm, _, _ in cells)
        jobs = [(c, "plain") for c in columns] + (
            [(c, "marker") for c in columns] if need_sep else []
        )
        my = _shard_select(jobs, args.worker_shard)
        if args.dry_run:
            logger.info("[p2-base][dry-run] %d column jobs", len(my))
            return
        model = _load_hf_base()
        for cid, sep_mode in my:
            _score_base_column(model, cid, questions, sep_mode, args.smoke)
        return
    if args.worker_shard and args.step == "cells":
        my_cells = _shard_select(cells, args.worker_shard)
        if args.dry_run:
            logger.info("[p2-cells][dry-run] %d adapters", len(my_cells))
            return
        from peft import PeftModel

        from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config

        model = _load_hf_base()
        for arm, cid, seed in my_cells:
            adapter_dir = _ensure_local_adapter(arm, cid, seed)
            cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
            assert_gauge_free_adapter_config(cfg, context=str(adapter_dir))
            peft_model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
            try:
                for eval_cid in columns:
                    _cell_read(
                        peft_model,
                        arm,
                        cid,
                        seed,
                        eval_cid,
                        questions,
                        sep_mode="marker",
                        smoke=args.smoke,
                    )
                    if _sep_variant(arm) == "sep":
                        _cell_read(
                            peft_model,
                            arm,
                            cid,
                            seed,
                            eval_cid,
                            questions,
                            sep_mode="plain",
                            smoke=args.smoke,
                        )
            finally:
                peft_model = peft_model.unload()
            _gate_check(arm, cid, seed, enforce=args.enforce_gate)
        return
    phase_log("p2_base")
    need_sep = any(_sep_variant(arm) == "sep" for arm, _, _ in cells)
    n_base_jobs = len(columns) * (2 if need_sep else 1)
    _run_wave(args, "2", "base", n_base_jobs)
    phase_log("p2_cells")
    _run_wave(args, "2", "cells", len(cells))
    skip = args.smoke or args.skip_upload or args.dry_run
    for sub in ("marker_base_slots", "G_cells", "p2/gate_checks"):
        _upload_tree(EVAL / sub, f"issue628_rig_revision/eval_results/{sub}", skip=skip)


# ── Phase 3: reuse-arm trained-negative columns (parity-probed) ──────────────


def _fetch_reuse_adapter(cid: str, seed: int) -> Path:
    from huggingface_hub import hf_hub_download

    sub = f"adapters/i537_marker_{cid}_seed{seed}"
    p = hf_hub_download(
        HF_MODEL_REPO, f"{sub}/adapter_model.safetensors", revision=I537_ADAPTER_REV[seed]
    )
    hf_hub_download(HF_MODEL_REPO, f"{sub}/adapter_config.json", revision=I537_ADAPTER_REV[seed])
    return Path(p).parent


def _parity_probe(model, questions: list[str], smoke: bool, columns: list[str]) -> None:
    """Apply-and-read parity: reproduce the snapshotted sp_swe→default seed-42
    cell within ±0.5 nat on the CURRENT stack (artifact-reuse check (g))."""
    import numpy as np
    from peft import PeftModel

    cid, eval_cid, seed = PARITY_PROBE_CELL
    if smoke:
        # Smoke wiring exercise only: the smoke eval-cache slice may not cover
        # the `default` column, so probe the first smoke column (recorded,
        # never enforced — the real probe runs before any Phase-3 batch read).
        eval_cid = columns[0]
    targets = json.loads(SNAPSHOT_MANIFEST.read_text())["parity_probe_cell"]
    adapter_dir = _fetch_reuse_adapter(cid, seed)
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
    try:
        from explore_persona_space.experiments.i537_marker_eval import score_marker_slots

        tok = _tokenizer()
        ctxs = _eval_contexts_for(eval_cid, questions, sep="", smoke=smoke)
        stats, _ = score_marker_slots(
            peft_model, tok, ctxs, marker_id=MARKER_ID, eos_token_id=IM_END_ID, batch_size=32
        )
        base = json.loads(_base_slot_path(eval_cid, "plain").read_text())["stats"]
        got = {
            "g_mean_delta_logp": float(
                np.mean([s["logp"] - b["logp"] for s, b in zip(stats, base, strict=True)])
            ),
            "g_mean_delta_z_marker": float(
                np.mean([s["z_marker"] - b["z_marker"] for s, b in zip(stats, base, strict=True)])
            ),
            "g_mean_delta_eos_margin": float(
                np.mean(
                    [
                        (s["z_marker"] - s["z_eos"]) - (b["z_marker"] - b["z_eos"])
                        for s, b in zip(stats, base, strict=True)
                    ]
                )
            ),
        }
    finally:
        peft_model.unload()
    out = EVAL / "p3/parity_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {**_meta(), "cell": f"{cid}__{eval_cid}__seed{seed}", "got": got, "targets": targets}
    out.write_text(json.dumps(report, indent=1))
    if smoke:
        logger.info("[p3] parity probe (smoke, tiny slice -- recorded, not enforced): %s", got)
        return
    for key in ("g_mean_delta_logp", "g_mean_delta_z_marker", "g_mean_delta_eos_margin"):
        diff = abs(got[key] - targets[key])
        assert diff <= PARITY_TOL_NAT, (
            f"[p3] parity probe FAILED on {key}: got {got[key]:+.4f} vs snapshot "
            f"{targets[key]:+.4f} (|diff|={diff:.4f} > {PARITY_TOL_NAT}) -- the reused #537 "
            "adapters do not reproduce on the current stack; per §7 drop the reuse-arm "
            "negative columns after one diagnosis round, do NOT silently re-grade."
        )
    logger.info("[p3] parity probe PASS: %s", got)


def phase3(args) -> None:
    from explore_persona_space.experiments.i537_contexts import NEGATIVE_CIDS, train_cids_for

    questions = _marker_eval_questions(args.smoke)
    grid_cids = train_cids_for("marker")
    if args.train_cids:
        grid_cids = [c for c in grid_cids if c in args.train_cids]
    reuse_cells = [(cid, seed) for cid in grid_cids for seed in args.seeds]
    neg_cols = list(NEGATIVE_CIDS)[:1] if args.smoke else list(NEGATIVE_CIDS)
    if args.worker_shard:
        my = _shard_select(reuse_cells, args.worker_shard)
        if args.dry_run:
            logger.info("[p3][dry-run] %d reuse adapters x %d columns", len(my), len(neg_cols))
            return
        import numpy as np
        from peft import PeftModel

        from explore_persona_space.experiments.i537_marker_eval import score_marker_slots

        model = _load_hf_base()
        # Probe runs once, in shard 0 (its output is the gate for ALL shards;
        # the wave runs probe-first via the dedicated probe step below).
        out_root = EVAL / "neg_columns" / REUSE_ARM
        out_root.mkdir(parents=True, exist_ok=True)
        tok = _tokenizer()
        for cid, seed in my:
            adapter_dir = _fetch_reuse_adapter(cid, seed)
            peft_model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
            try:
                for neg_cid in neg_cols:
                    cell_p = out_root / f"{cid}__{neg_cid}__seed{seed}.json"
                    if cell_p.exists():
                        continue
                    ctxs = _eval_contexts_for(neg_cid, questions, sep="", smoke=args.smoke)
                    stats, _ = score_marker_slots(
                        peft_model,
                        tok,
                        ctxs,
                        marker_id=MARKER_ID,
                        eos_token_id=IM_END_ID,
                        batch_size=32,
                    )
                    base = json.loads(_base_slot_path(neg_cid, "plain").read_text())["stats"]
                    per_q = [
                        {
                            "question": q,
                            "trained": s,
                            "base": b,
                            "delta_logp": s["logp"] - b["logp"],
                            "delta_z_marker": s["z_marker"] - b["z_marker"],
                            "delta_eos_margin": (s["z_marker"] - s["z_eos"])
                            - (b["z_marker"] - b["z_eos"]),
                        }
                        for q, s, b in zip(questions, stats, base, strict=True)
                    ]
                    cell_p.write_text(
                        json.dumps(
                            {
                                **_meta(),
                                "behavior": "marker",
                                "arm": REUSE_ARM,
                                "sep_mode": "marker",
                                "train_cid": cid,
                                "eval_cid": neg_cid,
                                "seed": seed,
                                "n_questions": len(questions),
                                "g_mean_delta_logp": float(
                                    np.mean([r["delta_logp"] for r in per_q])
                                ),
                                "g_mean_delta_z_marker": float(
                                    np.mean([r["delta_z_marker"] for r in per_q])
                                ),
                                "g_mean_delta_eos_margin": float(
                                    np.mean([r["delta_eos_margin"] for r in per_q])
                                ),
                                "emission_rate_trained": float(
                                    np.mean([s["argmax_is_marker"] for s in stats])
                                ),
                                "emission_rate_base": float(
                                    np.mean([b["argmax_is_marker"] for b in base])
                                ),
                                "per_question": per_q,
                                "adapter_revision": I537_ADAPTER_REV[seed],
                            },
                            indent=1,
                        )
                    )
                    logger.info("[p3] %s->%s seed%d done", cid, neg_cid, seed)
            finally:
                peft_model = peft_model.unload()
        return
    phase_log("p3_parity_probe")
    if args.dry_run:
        logger.info("[p3][dry-run] probe + %d reuse cells", len(reuse_cells))
    else:
        model = _load_hf_base()
        _parity_probe(model, questions, args.smoke, _eval_columns(args))
        del model
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    phase_log("p3_reuse_neg_columns")
    _run_wave(args, "3", "reuse", len(reuse_cells))
    skip = args.smoke or args.skip_upload or args.dry_run
    _upload_tree(EVAL / "neg_columns", "issue628_rig_revision/eval_results/neg_columns", skip=skip)
    _upload_tree(EVAL / "p3", "issue628_rig_revision/eval_results/p3", skip=skip)


# ── Phase 4: on-policy bystander read ────────────────────────────────────────


def _onpolicy_questions() -> list[str]:
    """#601's 10 eval questions = the Q_eval split of personas.EVAL_QUESTIONS."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
    )

    _q_train, q_eval = get_train_eval_questions()
    assert len(q_eval) == 10, len(q_eval)
    return q_eval


def _onpolicy_contexts(train_cid: str, smoke: bool) -> list[tuple[str, dict]]:
    """30 labeled contexts: own train context, the 4 trained-negative contexts,
    bare default, and 24 fixed-seed persona-bank personas (#472, pinned)."""
    import random

    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.i537_contexts import NEGATIVE_CIDS

    out: list[tuple[str, dict]] = [(train_cid, {"kind": "i537_ctx", "cid": train_cid})]
    for nc in NEGATIVE_CIDS:
        out.append((nc, {"kind": "i537_ctx", "cid": nc}))
    if train_cid != "default":
        out.append(("default", {"kind": "i537_ctx", "cid": "default"}))
    bank = load_persona_bank(INPUTS / "i472/persona_bank.json")
    names = sorted(bank)
    rng = random.Random(628)
    picked = rng.sample(names, min(N_ONPOLICY_PERSONAS, len(names)))
    if smoke:
        picked = picked[:2]
    for name in picked:
        out.append((f"persona_{name}", {"kind": "persona", "system": bank[name]}))
    assert len({label for label, _ in out}) == len(out), "duplicate context labels"
    return out


def _render_onpolicy_prompt(spec: dict, q: str, tok) -> str:
    from explore_persona_space.experiments.i537_contexts import build_prompt

    registry, demos = _registry_and_demos()
    if spec["kind"] == "i537_ctx":
        return build_prompt(registry[spec["cid"]], q, tok, behavior="marker", icl_demos=demos)
    msgs = [{"role": "system", "content": spec["system"]}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _strip_at_first_marker(text: str) -> tuple[str, bool]:
    idx = text.find("※")
    if idx < 0:
        return text, False
    cut = text[:idx]
    if cut.endswith(" "):
        cut = cut[:-1]  # the leading space belongs to the ` ※` token
    return cut, True


def _onpolicy_adapters(args) -> list[tuple[str, str, int]]:
    cells = _cells(args)
    return [
        (arm, cid, seed)
        for arm, cid, seed in cells
        if arm in ONPOLICY_ARMS and cid in MINI_ARM_CIDS
    ]


def _p4_generate(my: list[tuple[str, str, int]], args, tok, questions: list[str]) -> None:
    """Sub-step A: vLLM greedy own-answers per adapter (LoRA-applied), then
    full engine teardown BEFORE any HF model load (gotchas.md ordering)."""
    gen_todo = []
    for arm, cid, seed in my:
        d = EVAL / "bystander_onpolicy" / _cell_slug(arm, cid, seed)
        if not (d / "raw_completions.json").exists():
            gen_todo.append((arm, cid, seed))
    if not gen_todo:
        return
    from vllm.lora.request import LoRARequest

    llm = _vllm_engine(16384, enable_lora=True)
    try:
        for idx, (arm, cid, seed) in enumerate(gen_todo, start=1):
            slug = _cell_slug(arm, cid, seed)
            adapter_dir = _ensure_local_adapter(arm, cid, seed)
            contexts = _onpolicy_contexts(cid, args.smoke)
            qs = questions[:3] if args.smoke else questions
            prompts, keys = [], []
            for label, spec in contexts:
                for q in qs:
                    prompts.append(_render_onpolicy_prompt(spec, q, tok))
                    keys.append((label, q))
            results = _vllm_greedy(
                llm,
                prompts,
                MAX_NEW_TOKENS,
                lora_request=LoRARequest(slug, idx, str(adapter_dir)),
            )
            payload: dict = {}
            for (label, q), r in zip(keys, results, strict=True):
                payload.setdefault(label, {})[q] = r
            d = EVAL / "bystander_onpolicy" / slug
            d.mkdir(parents=True, exist_ok=True)
            (d / "raw_completions.json").write_text(
                json.dumps(
                    {**_meta(), "arm": arm, "cid": cid, "seed": seed, "completions": payload},
                    indent=1,
                )
            )
            logger.info("[p4-gen] %s: %d completions", slug, len(results))
    finally:
        _teardown_vllm(llm)


def _p4_reads(my: list[tuple[str, str, int]], args, tok, questions: list[str]) -> None:
    """Sub-step B: four-float slot reads at the end of the OWN response
    (stripped at first marker token), trained AND base sides."""
    import numpy as np
    from peft import PeftModel

    from explore_persona_space.experiments.i537_marker_eval import score_marker_slots

    model = _load_hf_base()
    for arm, cid, seed in my:
        slug = _cell_slug(arm, cid, seed)
        d = EVAL / "bystander_onpolicy" / slug
        reads_p = d / "reads.json"
        if reads_p.exists():
            continue
        raw = json.loads((d / "raw_completions.json").read_text())["completions"]
        contexts = _onpolicy_contexts(cid, args.smoke)
        qs = questions[:3] if args.smoke else questions
        ctx_strings, keys, stripped_flags = [], [], []
        for label, spec in contexts:
            for q in qs:
                resp = raw[label][q]["response"]
                stripped, had_marker = _strip_at_first_marker(resp)
                ctx_strings.append(_render_onpolicy_prompt(spec, q, tok) + stripped)
                keys.append((label, q))
                stripped_flags.append(had_marker)
        adapter_dir = _ensure_local_adapter(arm, cid, seed)
        peft_model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
        try:
            trained_stats, _ = score_marker_slots(
                peft_model,
                tok,
                ctx_strings,
                marker_id=MARKER_ID,
                eos_token_id=IM_END_ID,
                batch_size=16,
            )
        finally:
            peft_model = peft_model.unload()
        base_stats, _ = score_marker_slots(
            model, tok, ctx_strings, marker_id=MARKER_ID, eos_token_id=IM_END_ID, batch_size=16
        )
        rows = [
            {
                "context": label,
                "question": q,
                "emitted_marker_in_generation": flag,
                "trained": s,
                "base": b,
                "delta_logp": s["logp"] - b["logp"],
                "delta_z_marker": s["z_marker"] - b["z_marker"],
                "delta_eos_margin": (s["z_marker"] - s["z_eos"]) - (b["z_marker"] - b["z_eos"]),
            }
            for (label, q), flag, s, b in zip(
                keys, stripped_flags, trained_stats, base_stats, strict=True
            )
        ]
        by_ctx: dict[str, list[dict]] = {}
        for r in rows:
            by_ctx.setdefault(r["context"], []).append(r)
        summary = {
            label: {
                "mean_delta_logp": float(np.mean([r["delta_logp"] for r in rs])),
                "mean_delta_eos_margin": float(np.mean([r["delta_eos_margin"] for r in rs])),
                "emission_rate": float(np.mean([r["emitted_marker_in_generation"] for r in rs])),
            }
            for label, rs in by_ctx.items()
        }
        reads_p.write_text(
            json.dumps(
                {
                    **_meta(),
                    "arm": arm,
                    "cid": cid,
                    "seed": seed,
                    "read_slot": "canonical end-of-own-response (all arms; registered)",
                    "n_rows": len(rows),
                    "summary": summary,
                    "rows": rows,
                },
                indent=1,
            )
        )
        logger.info("[p4-read] %s: %d rows", slug, len(rows))


def phase4(args) -> None:
    adapters = _onpolicy_adapters(args)
    if args.worker_shard:
        my = _shard_select(adapters, args.worker_shard)
        if args.dry_run:
            logger.info("[p4][dry-run] %d adapters", len(my))
            return
        questions = _onpolicy_questions()
        tok = _tokenizer()
        _p4_generate(my, args, tok, questions)
        _p4_reads(my, args, tok, questions)
        return
    phase_log("p4_onpolicy")
    _run_wave(args, "4", "onpolicy", len(adapters))
    skip = args.smoke or args.skip_upload or args.dry_run
    if not skip:
        from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

        upload_raw_completions_to_data_repo(
            experiment_name="issue628_rig_revision", eval_results_dir=EVAL
        )
    _upload_tree(
        EVAL / "bystander_onpolicy",
        "issue628_rig_revision/eval_results/bystander_onpolicy",
        skip=skip,
    )


# ── Matched-install checkpoint re-read (registered §6 read-1 fallback) ──────


def _fetch_reread_checkpoint(ent: dict) -> Path:
    """Download one spec entry's matched-checkpoint adapter files locally.

    Fresh-arm checkpoints live at ``adapters/issue_628/<slug>/checkpoint-<step>``
    on main; reuse-arm (#537) checkpoints at the seed-pinned revision."""
    from huggingface_hub import hf_hub_download

    sub = ent["checkpoint_hf_subfolder"]
    step = int(ent["checkpoint_step"])
    rev = I537_ADAPTER_REV[int(ent["seed"])] if ent.get("arm_kind") == "reuse" else None
    d = OUT / "reread_ckpts" / f"{Path(sub).name}__ckpt{step}"
    if (d / "adapter_model.safetensors").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        p = hf_hub_download(HF_MODEL_REPO, f"{sub}/checkpoint-{step}/{fn}", revision=rev)
        shutil.copyfile(p, d / fn)
    return d


def phase_reread(args) -> None:
    """Consume the analysis-emitted ``matched_install_reread_spec.json``: per
    entry, fetch the mismatched arm's matched checkpoint and produce the
    ``default`` + 4 trained-negative four-float G-cells under
    ``EVAL/matched_install_reread/<arm>/`` (reusing the Phase-2 read path).

    Post-hoc, single-process (the orchestrator provides the compute after the
    main sweep); requires the Phase-0 eval response caches + Phase-2
    ``marker_base_slots`` files locally (run on the original instance state or
    re-fetch ``issue628_rig_revision/eval_results/marker_base_slots`` first).
    """
    assert args.spec, "--phase matched-install-reread requires --spec <path>"
    spec = json.loads(Path(args.spec).read_text())
    entries = spec["entries"]
    phase_log("matched_install_reread")
    null_steps = [e for e in entries if e.get("checkpoint_step") is None]
    if null_steps:
        raise SystemExit(
            f"[reread] {len(null_steps)} spec entries have checkpoint_step=null "
            f"(first: {null_steps[0].get('mismatched_arm')}/"
            f"{null_steps[0]['train_cid']}/seed{null_steps[0]['seed']} -- "
            f"{null_steps[0].get('note')}). Resolve every entry before dispatching; "
            "no silent skip."
        )
    if args.dry_run:
        for ent in entries:
            logger.info(
                "[reread][dry-run] would re-read %s/%s/seed%s @ checkpoint-%s on %s",
                ent["mismatched_arm"],
                ent["train_cid"],
                ent["seed"],
                ent["checkpoint_step"],
                ent["columns"],
            )
        return
    questions = _marker_eval_questions(args.smoke)
    # Pre-flight: every column's base-slot file must exist before any GPU work.
    missing = []
    for ent in entries:
        sepm = (
            "marker"
            if ent.get("sep_mode", "marker") == "marker"
            and _sep_variant(ent["mismatched_arm"]) == "sep"
            else "plain"
        )
        for e in ent["columns"]:
            p = _base_slot_path(e, sepm)
            if not p.exists() and str(p) not in missing:
                missing.append(str(p))
    if missing:
        raise SystemExit(
            f"[reread] {len(missing)} Phase-2 base-slot files missing (first: {missing[0]}). "
            "Run on the original instance state or re-fetch "
            "issue628_rig_revision/eval_results/marker_base_slots from the data repo."
        )
    from peft import PeftModel

    model = _load_hf_base()
    out_root = EVAL / "matched_install_reread"
    for ent in entries:
        mism, t, s = ent["mismatched_arm"], ent["train_cid"], int(ent["seed"])
        step = int(ent["checkpoint_step"])
        adapter_dir = _fetch_reread_checkpoint(ent)
        peft_model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
        try:
            for eval_cid in ent["columns"]:
                _cell_read(
                    peft_model,
                    mism,
                    t,
                    s,
                    eval_cid,
                    questions,
                    sep_mode=ent.get("sep_mode", "marker"),
                    smoke=args.smoke,
                    out_dir=out_root / mism,
                    fname_tag=f"__ckpt{step}",
                    extra={
                        "checkpoint_step": step,
                        "reread_target_dial": ent["target_dial"],
                        "reread_dial_gap_nat": ent["dial_gap_nat"],
                    },
                )
        finally:
            peft_model = peft_model.unload()
        logger.info("[reread] %s/%s/seed%s @ checkpoint-%s done", mism, t, s, step)
    skip = args.smoke or args.skip_upload or args.dry_run
    _upload_tree(out_root, "issue628_rig_revision/eval_results/matched_install_reread", skip=skip)


# ── Finalize: reproducibility-card results sentinel ──────────────────────────


def _finalize(args) -> None:
    cells = _cells(args)
    card = {
        "hf_model_repo": HF_MODEL_REPO,
        "adapter_paths": {_cell_slug(*c): _hf_adapter_subfolder(*c) for c in cells},
        "wandb_project": os.environ.get("WANDB_PROJECT", "issue628"),
        "wandb_run_names": [f"issue628_{_cell_slug(*c)}" for c in cells],
        "reused_adapters": {
            f"i537_marker_seed{s}": {"revision": I537_ADAPTER_REV[s]}
            for s in args.seeds
            if s in I537_ADAPTER_REV
        },
        "data_repo_prefix": "issue628_rig_revision",
    }
    note = json.dumps(
        {
            "summary": (
                f"issue-628 dispatcher complete: {len(cells)} fresh cells "
                f"(arms={sorted({c[0] for c in cells})}, seeds={list(args.seeds)}, "
                f"smoke={args.smoke})"
            ),
            "reproducibility_card": card,
        }
    )
    # A dry-run / smoke must NEVER emit an epm:results sentinel -- a live
    # poll_pipeline.py would drain it as real results.
    kind = "epm:results" if not (args.dry_run or args.smoke) else "epm:progress"
    write_sentinel(kind, note)


# ── Deferred-import verification (#606 pattern) ──────────────────────────────


def _verify_imports() -> int:
    """AST-walk THIS file and execute every deferred (function-local) import."""
    import importlib

    tree = ast.parse(Path(__file__).read_text())
    seen: set[tuple] = set()
    failures = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            key = (node.module, tuple(a.name for a in node.names))
            if key in seen:
                continue
            seen.add(key)
            try:
                mod = importlib.import_module(node.module)
                for a in node.names:
                    getattr(mod, a.name)
            except Exception as e:
                failures.append((node.module, [a.name for a in node.names], repr(e)))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if (a.name, ()) in seen:
                    continue
                seen.add((a.name, ()))
                try:
                    importlib.import_module(a.name)
                except Exception as e:
                    failures.append((a.name, [], repr(e)))
    if failures:
        for f in failures:
            logger.error("[verify-imports] FAILED: %s", f)
        return 1
    logger.info("[verify-imports] all %d imports resolved", len(seen))
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def _rebind_smoke_roots() -> None:
    global GEN, OUT, EVAL
    GEN = Path(str(GEN) + "_smoke")
    OUT = Path(str(OUT) + "_smoke")
    EVAL = Path(str(EVAL) + "_smoke")
    logger.info("[smoke] roots rebound: GEN=%s OUT=%s EVAL=%s", GEN, OUT, EVAL)


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #628 marker-rig revision dispatcher")
    ap.add_argument(
        "--phase",
        default="all",
        choices=["0", "1", "2", "3", "4", "all", "matched-install-reread"],
        help="'matched-install-reread' is the post-hoc §6 fallback; never part of 'all'",
    )
    ap.add_argument(
        "--spec",
        default=None,
        help="matched_install_reread_spec.json path (required for --phase matched-install-reread)",
    )
    ap.add_argument("--arms", type=lambda s: s.split(","), default=None)
    ap.add_argument("--train-cids", type=lambda s: s.split(","), default=None)
    ap.add_argument(
        "--seeds", type=lambda s: tuple(int(x) for x in s.split(",")), default=DEFAULT_SEEDS
    )
    ap.add_argument("--smoke", action="store_true", help="tiny pools + *_smoke roots")
    ap.add_argument("--dry-run", action="store_true", help="enumerate + spawn workers, no GPU work")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--enforce-gate", action="store_true", help="raise on §7 gate-check miss")
    ap.add_argument("--workers", type=int, default=0, help="0 = one worker per visible GPU")
    ap.add_argument("--worker-shard", default=None, help="internal: k/n cell shard")
    ap.add_argument("--step", default=None, help="internal: wave step within a phase")
    ap.add_argument("--verify-imports", action="store_true")
    args = ap.parse_args()

    if args.verify_imports:
        return _verify_imports()

    os.environ.setdefault("WANDB_PROJECT", "issue628")
    _require_credentials()
    if args.smoke:
        _rebind_smoke_roots()
    # In-process marker-id assert at dispatcher startup AND in every worker
    # subprocess (workers re-enter main): encode(" ※") == [83399].
    _tokenizer()
    _assert_negative_disjointness()

    phases = ["0", "1", "2", "3", "4"] if args.phase == "all" else [args.phase]
    if args.worker_shard:
        phases = [args.phase]
    runner = {
        "0": phase0,
        "1": phase1,
        "2": phase2,
        "3": phase3,
        "4": phase4,
        "matched-install-reread": phase_reread,
    }
    for ph in phases:
        runner[ph](args)

    if args.worker_shard:
        return 0  # [phase=done] is RESERVED for the main dispatcher log
    if args.phase in ("all", "4"):
        _finalize(args)
    else:
        write_sentinel(
            "epm:progress",
            f"i628_dispatch phase {args.phase} complete "
            f"(smoke={args.smoke}, dry_run={args.dry_run})",
        )
    phase_log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
