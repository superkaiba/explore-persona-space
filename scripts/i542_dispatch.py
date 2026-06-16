"""Issue #542 phase dispatcher -- negative-panel arms over the frozen #537 testbed.

Smoke IS the sweep with one cell (PASS_UNIFIED): ``--smoke --arm arm2_close
--cells sp_swe`` runs the SAME phase entrypoints, builder subprocess shape,
[phase=...] logging, sentinel writer, and vLLM teardown as the full sweep;
smoke artifacts rebind to parallel ``*_smoke`` roots exactly as the parent
``i537_dispatch.py`` does. No separate smoke code path.

Phases (plan §3.3):
  --phase p0prime  fetch (parent artifacts from the PINNED HF data revision +
                   git base slots) → contexts (i542_sample_contexts freeze) →
                   checks (merged-registry render + marker token + payload
                   disjointness + row-split manifest + freeze manifest) →
                   responses (vLLM greedy, 300 train questions x 16 NEW
                   negative contexts) → clouds (reduced L14/L22 last_prompt
                   extraction for the 16 new negatives) → closeness (arm2 <
                   arm1 activation-distance manipulation check).
  --phase train    per (arm, cell): builder subprocess (--negatives panel)
                   then train_lora with the parent MARKER_TRAIN_KWARGS
                   verbatim (band-stop [5,12], overshoot-aware); adapters →
                   HF ``adapters/i542_<arm>_<cid>_seed<S>``; stop-steps
                   recorded per cell.
  --phase eval     per (arm, cell): four-float slot cross-eval over the 30
                   frozen eval contexts vs the PARENT base slots (Stage 1
                   reused, never recomputed in real runs); per-pair JSON the
                   moment it completes + per-cell rollup. ``--steps v2`` runs
                   the base-side parity spot-check (3 contexts, ≤0.05 nat).
  --phase gate     G1' (CPU, after arm2_close train+eval): band landing ≥
                   13/16, V2 parity pass, realized throughput vs 0.12 Qs/s/GPU.
                   ``--steps c8`` (CPU, after ALL core panels + replicates):
                   the registered c8 add-back decision -- realized GPU-h
                   (summed from runtime/gpu_runtimes.jsonl, written by every
                   GPU phase process) ≤ 62 → c8 AUTO-included in later
                   train/eval/assemble; else skip. Decision posted as an
                   epm:progress sentinel either way (never silent);
                   ``--include-c8`` stays the manual override.
  --phase assemble per-arm G tensors (arm1_xfam assembled from the parent's
                   git G_cells through the SAME code path) + HF upload of the
                   new mixes/caches/contexts under ``issue542_negative_panels/``.
  --phase analyze  subprocess → scripts/i542_registered_reads.py --ladder
                   (registered reads + seed-noise floor + the §6.5 ladder
                   deliverable incl. dist_to_panel rows) → i542_figures (CPU).

Pod-side contract (CLAUDE.md / poll_pipeline.py): emits ``[phase=<name>]``
log lines, a terminal ``[phase=done]``, and an end-of-run sentinel JSON at
``/workspace/logs/issue-542-<kind_slug>-<epoch>.json`` carrying
``sentinel_schema_version`` / ``kind`` / ``version`` / ``note``. NEVER shells
out to scripts/task.py.

Canonical launches (plan §10):
    nohup uv run python scripts/i542_dispatch.py --phase p0prime \
        > /workspace/logs/issue-542-p0prime.log 2>&1 &
    nohup uv run python scripts/i542_dispatch.py --phase train --shard K/8 \
        --gpu-id K > /workspace/logs/issue-542-train-K.log 2>&1 &
    nohup uv run python scripts/i542_dispatch.py --phase eval --shard K/8 \
        --gpu-id K > /workspace/logs/issue-542-eval-K.log 2>&1 &
Smoke: --smoke --arm arm2_close --cells sp_swe (one cell end-to-end through
the sweep path). Plumbing dry-run: --dry-run.
"""

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

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i542_dispatch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import i537_dispatch as i537d  # noqa: E402  (sibling-script reuse: helpers + recipe)

TASK_ID = 542
QWEN_ID = i537d.QWEN_ID
SEED = 42  # DATA seed (frozen pools/mixes/caches are keyed by this forever)
MARKER_BATCH_DEFAULT = 32
LONG_EVAL_CIDS = i537d.LONG_EVAL_CIDS
HF_MODEL_REPO = i537d.HF_MODEL_REPO
DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_REV = "db3662ae1d1ff4484ada027ac92a2658c4dec2e8"  # parent pin (plan §3.4)
# v1 freeze pin (follow-up plan §4 diff 3): the i542 context freeze + G tensors
# as uploaded by the COMPLETED v1 run -- the content-identity mechanism for the
# positives-only follow-up (the freeze is FETCHED at this pin, never re-frozen).
I542_REV = "18dc6a8d9919e0af10d2444c787dce2a0d0536f9"
HF_PARENT_PREFIX = "issue537_context_generalization"
HF_I542_PREFIX = "issue542_negative_panels"

DATA537 = REPO / "data/issue_537"  # parent INPUTS (pools/contexts), shared
EVAL537 = REPO / "eval_results/issue_537"  # parent artifacts in git, read-only
# Generated-artifact roots; ``--smoke`` rebinds all three to *_smoke in main().
GEN = Path(os.environ.get("I542_GEN_ROOT", str(REPO / "data/issue_542")))
OUT = Path(os.environ.get("I542_OUT_ROOT", str(REPO / "outputs/issue_542")))
EVAL = Path(os.environ.get("I542_EVAL_ROOT", str(REPO / "eval_results/issue_542")))
# Parent #537 clouds (closeness-check train-context anchors + the reused arm-1
# panel) are read-only parent INPUTS, like DATA537: ONE canonical location that
# does NOT rebind under --smoke. Smoke isolation protects artifacts THIS run
# GENERATES; a fetched parent input can never be smoke-contaminated. Keeping the
# path fixed lets the smoke closeness step read the SAME files the real run
# reads (pod incident 2026-06-11: smoke p0prime crashed at _centroid("sp_swe")
# because the smoke EVAL root had no clouds_parent/ staged).
CLOUDS_PARENT = REPO / "eval_results/issue_542/clouds_parent"

# V2 base-side parity spot-check contexts (plan §6 check V2): one shared, one
# held-out, one binst column; deterministic.
V2_CIDS = ("sp_swe", "reph_formal_ho", "binst_marker")
V2_MEDIAN_TOL_NATS = 0.05
# G1' band-landing "in/near" window: in-band [5,12] plus a 2-nat shoulder
# (the parent's standing flags: binst_marker saturated above, fmt_code below).
BAND_LOW, BAND_HIGH, BAND_SHOULDER = 5.0, 12.0, 2.0
# c8 add-back gate (plan §3.3 / §9 ladder step 2): include c8 iff cumulative
# realized GPU-h after all core panels + replicates is <= this threshold.
C8_GATE_GPU_H = 62.0

# Opt-in-only arms (NOT in the default ARM_TRAIN_ORDER): c8 (conditional
# add-back), pos_only (positives-only follow-up), and the genuine-near-twin
# proximity arms. Each runs ONLY via explicit --arm so a default all-arm
# invocation reproduces the v1 run exactly.
_OPT_IN_ARMS: tuple[str, ...] = ("c8", "pos_only", "nt_close", "xfam_long", "repl_nt")

# Genuine-near-twin (follow-up `genuine-near-twin-negatives`, plan §3/§4.2):
# the proximity-axis arms train at the LONGER budget -- band-stop OFF + a
# fixed 1 epoch + the per-cell bystander-headroom early-stop -- so the
# negatives carry real gradient at a NON-saturated anchor. Applied IDENTICALLY
# to both arms (a shared regime, not a between-arm variable). The seed-43
# near-twin replicate arm shares the budget.
LONGER_BUDGET_ARMS: tuple[str, ...] = ("nt_close", "xfam_long", "repl_nt")
NT_TRAIN_EPOCHS = 1
# Non-saturated-anchor gate (Gate 2 / K1''): per-cell source log P - base must
# sit in this band. >5/16 cells violating in EITHER arm -> ABORT-and-REPORT
# (plan §4.2 / §7 K1'') -- never a mixed-budget descope.
NT_BAND_LOW_NATS, NT_BAND_HIGH_NATS = 5.0, 12.0
NT_K1_MAX_VIOLATORS = 5
# Manipulation-check gate (Gate 1 / K2''): near-twin panel mean cos-distance
# to the train-context centroids must be <= this fraction of the distant
# panel's mean distance (@ L22). FAIL -> ABORT-and-REPORT (plan §4.2 / §11).
NT_MANIP_RATIO_THRESHOLD = 0.5
# Bystander-headroom early-stop ceiling: a cell stops early if its BYSTANDER
# (negative-row) on-policy argmax-is-marker emission rate crosses this ceiling
# (gating on BYSTANDER resolution, NEVER source rate -- the source SHOULD
# saturate as the implant; marker-training-recipe.md). The fixed 1-epoch
# budget is the primary control; this callback is the safety stop.
NT_BYSTANDER_CEILING = 0.5
NT_BYSTANDER_EVAL_EVERY_STEPS = 5
NT_BYSTANDER_MIN_STEPS = 10

_CURRENT_PHASE = "init"
_PHASE_DIGIT_WORDS = str.maketrans({"0": "zero", "1": "one", "2": "two", "3": "three"})


def phase_log(name: str) -> None:
    """Emit the [phase=...] line poll_pipeline.py parses (letters/underscore only)."""
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
        if not d.exists():  # local VM -> repo logs/
            d = REPO / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_sentinel(kind: str, note: str, *, version: int = 1, extra: dict | None = None) -> Path:
    """End-of-run sentinel with poll_pipeline's _SENTINEL_REQUIRED_KEYS."""
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "task_id": TASK_ID,
        "by": "i542_dispatch",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    if extra:
        payload.update(extra)
    slug = kind.replace(":", "_")
    out = _log_dir() / f"issue-{TASK_ID}-{slug}-{time.time_ns()}.json"
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


def _meta(train_seed: int = SEED) -> dict:
    m = {
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "seed": train_seed,
    }
    if train_seed != SEED:
        m["data_seed"] = SEED
    return m


def _require_credentials() -> None:
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY missing"


def _record_gpu_runtime(args, elapsed_s: float) -> None:
    """Append this process's wall seconds to ``runtime/gpu_runtimes.jsonl``.

    Every GPU phase process is pinned to exactly ONE GPU (the
    CUDA_VISIBLE_DEVICES pin in ``main``), so process wall-seconds == realized
    GPU-seconds. The c8 add-back gate (``--phase gate --steps c8``) sums these
    rows against C8_GATE_GPU_H. Single short O_APPEND writes keep concurrent
    shard processes safe.
    """
    p = EVAL / "runtime/gpu_runtimes.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "phase": args.phase,
        "steps": args.steps,
        "arm": args.arm,
        "cells": args.cells,
        "shard": args.shard,
        "smoke": bool(args.smoke),
        "elapsed_s": round(elapsed_s, 1),
        "gpu_count": 1,
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ── Pools / registry / cells ─────────────────────────────────────────────────


def _train_pool_path(smoke: bool) -> Path:
    """Frozen 300-question marker train pool; smoke = deterministic 12-slice.

    The smoke slice is WRITTEN under the smoke GEN root (never under the
    parent's frozen ``data/issue_537/pools``) so a smoke pool can never feed a
    real run; the slice is deterministic (first 12 of the frozen order).
    """
    real = DATA537 / "pools/pool_marker_train_300.json"
    if not smoke:
        return real
    sp = GEN / "pools/pool_marker_train_300.smoke.json"
    if not sp.exists():
        qs = json.loads(real.read_text())["questions"][:12]
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({"questions": qs, "smoke": True, "sliced_from": str(real)}))
    return sp


def _train_questions(smoke: bool) -> list[str]:
    return json.loads(_train_pool_path(smoke).read_text())["questions"]


def _eval_questions(smoke: bool) -> list[str]:
    qs = json.loads((DATA537 / "pools/pool_marker_eval_32.json").read_text())["questions"]
    return qs[:4] if smoke else qs


def _i542_negatives_path() -> Path:
    return GEN / "contexts/i542_negatives.json"


def _merged_registry_and_demos(*, require_near_twins: bool = False):
    from explore_persona_space.experiments.i537_contexts import load_icl_demos
    from explore_persona_space.experiments.i542_panels import load_merged_registry

    sampled = Path(
        os.environ.get("I537_SAMPLED_CONTEXTS", DATA537 / "contexts/sampled_contexts.json")
    )
    demos_p = Path(os.environ.get("I537_ICL_DEMOS", DATA537 / "contexts/icl_demos.json"))
    return (
        load_merged_registry(
            sampled, _i542_negatives_path(), require_near_twins=require_near_twins
        ),
        load_icl_demos(demos_p),
    )


def _cells_touch_near_twins(args) -> bool:
    """True iff this invocation's arms include any genuine-near-twin panel.

    The near-twin payloads are required (require_near_twins=True) whenever a
    proximity arm is in scope so a missing near-twin freeze fails loud instead
    of rendering an empty system prompt.
    """
    if not args.arm:
        return False
    return args.arm in ("nt_close", "xfam_long", "repl_nt")


def _tokenizer():
    return i537d._tokenizer()


def _shard_select(items: list, shard: str | None) -> list:
    if not shard:
        return items
    k, n = (int(x) for x in shard.split("/"))
    assert 0 <= k < n, shard
    return [it for i, it in enumerate(items) if i % n == k]


def _c8_gate_includes() -> bool:
    """True iff the registered c8 add-back gate (plan §3.3) decided "include".

    Written by ``--phase gate --steps c8``; once present with an "include"
    decision, every later train/eval/assemble invocation AUTO-includes the c8
    cells -- no --include-c8 flag needed, so the add-back can never be
    silently omitted after the gate fires.
    """
    p = EVAL / "p1/c8_gate.json"
    return p.exists() and json.loads(p.read_text()).get("decision") == "include"


def _arm_list(args) -> list[str]:
    from explore_persona_space.experiments.i542_panels import ARM_TRAIN_ORDER, REPLICATE_ARMS

    arms = [*ARM_TRAIN_ORDER]
    if args.include_c8 or _c8_gate_includes():
        arms.append("c8")
    arms += list(REPLICATE_ARMS)
    if args.arm:
        # Opt-in-only arms: c8 (conditional add-back), pos_only (the #542
        # positives-only follow-up), and the genuine-near-twin proximity arms
        # nt_close / xfam_long run ONLY via explicit --arm; none join default
        # all-arm invocations (so the v1 run is reproduced exactly).
        assert args.arm in arms or args.arm in _OPT_IN_ARMS, (
            f"unknown --arm {args.arm} (of {arms} or opt-in {_OPT_IN_ARMS})"
        )
        arms = [args.arm]
    return arms


def _cells_for_arm(arm: str, args) -> list[dict]:
    """Cell specs for one arm: {arm, cid, train_seed, panel, mix, meta}.

    Replicate arms (plan §3.2): 4 fixed cells at TRAIN_SEED=43.
    ``repl_parent`` consumes the parent's frozen seed-42 mixes verbatim
    (downloaded at fetch -- no build); ``repl_close`` consumes arm2_close's
    seed-42 mixes (built by the arm2_close build step or its own).
    """
    from explore_persona_space.experiments.i537_contexts import train_cids_for
    from explore_persona_space.experiments.i542_panels import (
        PANELS,
        REPLICATE_CELLS,
        REPLICATE_TRAIN_SEED,
    )

    if arm == "repl_parent":
        cids, seed = list(REPLICATE_CELLS), REPLICATE_TRAIN_SEED
        mix_root, panel = GEN / "train_parent/marker", None
    elif arm == "repl_close":
        cids, seed = list(REPLICATE_CELLS), REPLICATE_TRAIN_SEED
        mix_root, panel = GEN / "train/arm2_close/marker", PANELS["arm2_close"]
    elif arm == "repl_nt":
        # Genuine-near-twin seed-43 noise floor (plan §5): the 4 replicate
        # cells, panel = nt_close, mixes from nt_close's seed-42 data build
        # (the longer-budget seed-noise floor; the v1 floor was band-stopped).
        cids, seed = list(REPLICATE_CELLS), REPLICATE_TRAIN_SEED
        mix_root, panel = GEN / "train/nt_close/marker", PANELS["nt_close"]
    else:
        cids, seed = train_cids_for("marker"), SEED
        mix_root, panel = GEN / f"train/{arm}/marker", PANELS[arm]

    if args.cells:
        toks = [t.strip() for t in str(args.cells).split(",") if t.strip()]
        if len(toks) == 1 and toks[0].isdigit():
            cids = cids[: int(toks[0])]
        else:
            unknown = [t for t in toks if t not in cids]
            assert not unknown, f"--cells cids not in {arm}'s cell list: {unknown}"
            cids = [c for c in cids if c in toks]
    return [
        {
            "arm": arm,
            "cid": cid,
            "train_seed": seed,
            "panel": panel,
            "mix": mix_root / f"{cid}_seed{SEED}.jsonl",
            "meta": mix_root / f"{cid}_seed{SEED}.meta.json",
        }
        for cid in cids
    ]


def _all_cells(args) -> list[dict]:
    cells: list[dict] = []
    for arm in _arm_list(args):
        cells.extend(_cells_for_arm(arm, args))
    return _shard_select(cells, args.shard)


def _adapter_subfolder(cell: dict) -> str:
    return f"adapters/i542_{cell['arm']}_{cell['cid']}_seed{cell['train_seed']}"


# ── Phase p0prime ────────────────────────────────────────────────────────────


def _hf_fetch(rel: str, local: Path) -> None:
    """One pinned-revision file from the parent HF data prefix → exact local path."""
    from huggingface_hub import hf_hub_download

    if local.exists():
        return
    got = hf_hub_download(
        DATA_REPO,
        f"{HF_PARENT_PREFIX}/{rel}",
        repo_type="dataset",
        revision=DATA_REV,
    )
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(got, local)
    logger.info("[fetch] %s -> %s", rel, local)


def _hf_fetch_i542(rel: str, local: Path) -> None:
    """One pinned-revision file from THIS issue's HF prefix (the v1 freeze pin)."""
    from huggingface_hub import hf_hub_download

    if local.exists():
        return
    got = hf_hub_download(
        DATA_REPO,
        f"{HF_I542_PREFIX}/{rel}",
        repo_type="dataset",
        revision=I542_REV,
    )
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(got, local)
    logger.info("[fetch] i542@%s %s -> %s", I542_REV[:8], rel, local)


def _verify_freeze_hashes() -> None:
    """Hash-check the fetched parent inputs against the GIT freeze manifest."""
    import hashlib

    man = json.loads((EVAL537 / "prereg/freeze_manifest.json").read_text())
    checked = 0
    for rel, sha in man["artifact_sha256"].items():
        p = REPO / rel
        if not p.exists():
            continue  # only pools/contexts are fetched here; others optional
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == sha, f"freeze-hash MISMATCH for {rel}: {got[:12]} != {sha[:12]}"
        checked += 1
    assert checked >= 4, f"freeze-hash check covered only {checked} files"
    logger.info("[fetch] freeze-manifest hash check OK (%d files)", checked)


def _fetch(args) -> None:
    """Idempotent download of every reused parent artifact (plan §3.0).

    Real runs: contexts + pools (hash-checked), the 20 train-question response
    caches, the 30 eval response caches, the 4 parent replicate mixes, the
    parent base slots (git copy), and the 20 parent clouds the closeness
    check needs. Smoke runs fetch contexts + pools + the SAME 20 parent
    clouds (the smoke closeness step computes the same train-context /
    arm-1-panel centroids; only the tiny caches are smoke-generated) and
    skip the cache/mix downloads.
    """
    from explore_persona_space.experiments.i537_contexts import (
        NEGATIVE_CIDS,
        eval_cids_for,
        train_cids_for,
    )
    from explore_persona_space.experiments.i542_panels import REPLICATE_CELLS

    for name in ("sampled_contexts.json", "icl_demos.json"):
        _hf_fetch(f"data/contexts/{name}", DATA537 / "contexts" / name)
    for stem in (
        "pool_marker_eval_32",
        "pool_marker_train_300",
        "pool_demo_seeds_537",
        "pool_fact_30",
        "pool_refusal_40",
        "pool_refusal_requests_200",
        "pool_sycophancy_25",
        "pool_em_8",
    ):
        _hf_fetch(f"data/pools/{stem}.json", DATA537 / f"pools/{stem}.json")
    _verify_freeze_hashes()

    # Parent base slots: in git under eval_results/issue_537 -> copy into the
    # i542 EVAL root so the eval driver reads ONE root (smoke roots included).
    base_src = EVAL537 / "marker_base_slots"
    base_dst = EVAL / "marker_base_slots"
    base_dst.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for p in sorted(base_src.glob("*.json")):
        if not (base_dst / p.name).exists():
            shutil.copyfile(p, base_dst / p.name)
            n_copied += 1
    logger.info("[fetch] parent base slots: %d copied (git -> %s)", n_copied, base_dst)

    # Parent clouds: read-only INPUTS for the closeness check (16 train-context
    # anchors + the 4 reused arm-1 panel members). The smoke closeness step
    # computes the SAME centroids, so these are fetched BEFORE the smoke
    # early-return, to the canonical non-rebinding CLOUDS_PARENT location.
    train_cids = train_cids_for("marker")
    for cid in [*train_cids, *NEGATIVE_CIDS]:
        _hf_fetch(f"clouds/{cid}__last_prompt.npz", CLOUDS_PARENT / f"{cid}__last_prompt.npz")

    if args.smoke:
        logger.info("[fetch] smoke mode -- skipping cache/mix downloads")
        return

    # #542 follow-up: the v1 context freeze is FETCHED at the i542 pin instead
    # of re-running _contexts_step -- regenerating would re-sample the
    # Claude-written twins and silently fork the registry (the gotchas.md
    # "HF mirror != local-verified copy" class; the pin IS the content-identity
    # mechanism, follow-up plan §4 diff 3 / fitness (f)).
    _hf_fetch_i542("contexts/i542_negatives.json", _i542_negatives_path())

    for cid in [*train_cids, *NEGATIVE_CIDS]:
        _hf_fetch(f"data/responses/{cid}.json", GEN / f"responses/{cid}.json")
    for cid in eval_cids_for("marker"):
        _hf_fetch(f"data/responses_eval/{cid}.json", GEN / f"responses_eval/{cid}.json")
    for cid in REPLICATE_CELLS:
        for suffix in (".jsonl", ".meta.json"):
            _hf_fetch(
                f"data/train/marker/{cid}_seed{SEED}{suffix}",
                GEN / f"train_parent/marker/{cid}_seed{SEED}{suffix}",
            )
    logger.info("[fetch] complete")


def _write_smoke_negatives(out_path: Path) -> None:
    """Deterministic placeholder i542 negatives (smoke ONLY, zero API/stream cost).

    Keeps the full 50-context merged registry resolvable so the loader,
    builder, render, and disjointness asserts run end-to-end in smoke. The
    payload carries ``skip_screens=True`` so ``load_i542_negatives`` REFUSES
    it in a real run unless I542_ALLOW_SMOKE_CONTEXTS=1 (parent pattern).
    """
    parent = json.loads((DATA537 / "contexts/sampled_contexts.json").read_text())
    ph = {}
    for cid, src in (
        ("neg_sp_ph1_twin", "sp_ph1"),
        ("neg_sp_ph2_twin", "sp_ph2"),
        ("neg_sp_ph5", "sp_ph1"),
        ("neg_sp_ph6", "sp_ph2"),
    ):
        ph[cid] = {
            "persona": parent["personahub"][src]["persona"] + f" You also smoke-test ({cid}).",
            "n_tokens": -1,
            "source": "smoke-placeholder",
            "twin_of": src,
        }
    wc = {}
    for i, cid in enumerate(("neg_wc_short2", "neg_wc_short3", "neg_wc_short4")):
        wc[cid] = {
            "messages": [
                {"role": "user", "content": f"Smoke placeholder question {i}?"},
                {"role": "assistant", "content": f"Smoke placeholder answer {i}."},
            ],
            "prefix_token_len": 16,
            "conversation_hash": f"smoke-{cid}",
            "topic": "smoke",
            "n_exchanges": 1,
            "bin": "short",
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 537,
                "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "git_commit": _git_commit(),
                "skip_screens": True,
                "max_rows": 0,
                "personahub": ph,
                "wildchat": wc,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _ensure_near_twins(args) -> None:
    """APPEND the deterministic near-twins to the i542 freeze (idempotent).

    The genuine-near-twin proximity arms FETCH the v1 freeze at the i542 pin
    (so the v1 registry cannot fork) and then APPEND the 4 near-twin cids under
    a `near_twins` block via `--regen-nt-twins` (deterministic, no LLM, no
    re-sampling). Idempotent: a freeze already carrying all 4 near-twins is
    left untouched. In smoke the placeholder near-twins are already written by
    `_write_smoke_negatives`, so this is a no-op there.
    """
    from explore_persona_space.experiments.i542_panels import I542_NT_TWIN_CIDS

    out = _i542_negatives_path()
    assert out.exists(), (
        f"[contexts] near-twin append needs the v1 freeze at {out} (run fetch + contexts first)"
    )
    payload = json.loads(out.read_text())
    have = set(payload.get("near_twins", {}))
    if set(I542_NT_TWIN_CIDS) <= have:
        logger.info("[contexts] all %d near-twins already frozen -- skip", len(I542_NT_TWIN_CIDS))
        return
    if args.smoke:
        # Smoke freeze placeholder near-twins (deterministic surface edits on
        # the smoke-placeholder ph sources); no API spend, structural only.
        from explore_persona_space.experiments.i542_panels import (
            NT_PERTURB,
            NT_TWIN_OF,
            apply_nt_perturbation,
        )
        from explore_persona_space.personas import PERSONAS

        src = {
            "sp_swe": PERSONAS["software_engineer"],
            "sp_doctor": PERSONAS["medical_doctor"],
            "sp_ph1": payload.get("personahub", {}).get("neg_sp_ph5", {}).get("persona")
            or "You are a smoke-placeholder researcher who studies things.",
            "sp_ph2": payload.get("personahub", {}).get("neg_sp_ph6", {}).get("persona")
            or "You are a smoke-placeholder analyst who reviews data.",
        }
        near = {
            cid: {
                "system_prompt": apply_nt_perturbation(cid, src[NT_TWIN_OF[cid]]),
                "twin_of": NT_TWIN_OF[cid],
                "perturbation": NT_PERTURB[cid],
                "source_persona": src[NT_TWIN_OF[cid]],
                "source": "near-twin",
            }
            for cid in I542_NT_TWIN_CIDS
        }
        payload.setdefault("near_twins", {}).update(near)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        logger.info("[contexts] smoke placeholder near-twins appended: %s", out)
        return
    cmd = [
        sys.executable,
        str(REPO / "scripts/i542_sample_contexts.py"),
        "--parent",
        str(DATA537 / "contexts/sampled_contexts.json"),
        "--out",
        str(out),
        "--regen-nt-twins",
    ]
    subprocess.run(cmd, check=True, cwd=REPO, env={**os.environ})


def _contexts_step(args) -> None:
    """Freeze data/issue_542/contexts/i542_negatives.json (idempotent).

    Then APPEND the deterministic near-twins (follow-up plan §4.3) so the
    proximity arms have their negatives -- the v1 freeze is REUSED (fetched at
    the pin) and never re-sampled; near-twins are appended, never forking the
    v1 registry.
    """
    out = _i542_negatives_path()
    if out.exists():
        logger.info("[contexts] %s already frozen -- skip base freeze", out)
        _ensure_near_twins(args)
        return
    if args.smoke:
        _write_smoke_negatives(out)
        logger.info("[contexts] smoke placeholder negatives written: %s", out)
        _ensure_near_twins(args)
        return
    cmd = [
        sys.executable,
        str(REPO / "scripts/i542_sample_contexts.py"),
        "--parent",
        str(DATA537 / "contexts/sampled_contexts.json"),
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, cwd=REPO, env={**os.environ})
    _ensure_near_twins(args)


def _checks_step(args) -> None:
    """Merged-registry render + marker token + disjointness + manifests."""
    from explore_persona_space.experiments.i537_contexts import (
        registry_hash,
        render_check,
    )
    from explore_persona_space.experiments.i542_panels import (
        COUNT_LEVELS,
        NEW_NEGATIVE_CIDS,
        PANELS,
        assert_panel_disjointness,
        row_split_sizes,
    )

    registry, demos = _merged_registry_and_demos(require_near_twins=True)
    tok = _tokenizer()  # asserts ' ※' -> [83399]
    assert_panel_disjointness(registry)
    lens = render_check(registry, tok, icl_demos=demos)
    new_lens = {c: lens[c] for c in NEW_NEGATIVE_CIDS}
    logger.info("[checks] merged registry renders OK; new-negative lens=%s", new_lens)

    # The split manifest records the REGISTERED real-run arithmetic (300 rows
    # at every count level, plan §3.1) -- always computed at n=300 regardless
    # of smoke (a 12-question smoke pool cannot split 16 ways and is not the
    # registered design). The smoke pool size is recorded alongside.
    n_q = len(_train_questions(False))
    split_manifest = {
        **_meta(),
        "n_questions": n_q,
        "smoke_pool_n": len(_train_questions(True)) if args.smoke else None,
        "splits": {slug: row_split_sizes(n_q, k) for slug, k in COUNT_LEVELS.items()},
        "panels": PANELS,
    }
    p = EVAL / "p0/row_split_manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(split_manifest, indent=2))

    import hashlib

    neg_p = _i542_negatives_path()
    freeze = {
        **_meta(),
        "schema_version": 1,
        "smoke": bool(args.smoke),
        "parent_data_revision": DATA_REV,
        "parent_freeze_commit": json.loads((EVAL537 / "prereg/freeze_manifest.json").read_text())[
            "freeze_commit"
        ],
        "artifact_sha256": {
            str(neg_p.relative_to(REPO)): hashlib.sha256(neg_p.read_bytes()).hexdigest()
        },
        "merged_registry_hash": registry_hash(registry, icl_demos=demos),
        "panels": PANELS,
    }
    fp = EVAL / "prereg/i542_freeze_manifest.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(freeze, indent=2))
    logger.info("[checks] manifests written: %s, %s", p, fp)


def _responses_step(args) -> None:
    """Frozen base greedy on the train pool under each NEW negative context.

    Real runs: 16 new contexts x 300 questions (parent caches for train cids,
    parent negatives, and ``default`` were fetched verbatim). Smoke: tiny
    caches for the smoke cells' positives + the smoke arm's panel + the 2
    smoke eval contexts (parent gen pattern, smoke pools).
    """
    from explore_persona_space.experiments.i537_cache import cache_covers, write_response_cache
    from explore_persona_space.experiments.i537_contexts import build_prompt, eval_cids_for

    # NEW_NEGATIVE_CIDS now includes the 4 near-twins -> their payloads must be
    # resolved (fail loud on a missing near-twin freeze, not an empty prompt).
    registry, demos = _merged_registry_and_demos(require_near_twins=True)
    tok = _tokenizer()
    train_q = _train_questions(args.smoke)

    targets: list[tuple[Path, str, list[str]]] = []  # (path, cid, questions)
    out_dir = GEN / "responses"
    if args.smoke:
        cells = _all_cells(args)
        panel_cids = sorted({c for cell in cells if cell["panel"] for c in cell["panel"]})
        pos_cids = sorted({cell["cid"] for cell in cells})
        for cid in dict.fromkeys([*pos_cids, *panel_cids]):
            targets.append((out_dir / f"{cid}.json", cid, train_q))
        eval_dir = GEN / "responses_eval"
        for cid in eval_cids_for("marker")[:2]:
            targets.append((eval_dir / f"{cid}.json", cid, _eval_questions(True)))
    else:
        # The genuine-near-twin run needs base-greedy responses ONLY for the 4
        # NEW near-twin contexts (the v1 negatives + the parent xfam_long
        # negatives are inherited via the fetched freeze + parent caches; the
        # v1 16 NEW negatives are not panel members of the proximity arms).
        from explore_persona_space.experiments.i542_panels import I542_NT_TWIN_CIDS

        for cid in I542_NT_TWIN_CIDS:
            targets.append((out_dir / f"{cid}.json", cid, train_q))

    todo = [
        (p, cid, qs)
        for (p, cid, qs) in targets
        if not cache_covers(p, qs, smoke=args.smoke, behavior="marker", expected_pool=qs)
    ]
    if not todo:
        logger.info("[responses] all %d caches present + validated -- skip", len(targets))
        return
    llm = i537d._vllm_engine(16384)
    try:
        for p, cid, qs in todo:
            prompts = [
                build_prompt(registry[cid], q, tok, behavior="marker", icl_demos=demos) for q in qs
            ]
            results = i537d._vllm_greedy(
                llm, prompts, i537d.MAX_NEW_TOKENS, expect_prompt_lens=None
            )
            trunc = sum(1 for r in results if r["finish_reason"] != "stop")
            payload = {
                **_meta(),
                "cid": cid,
                "behavior": "marker",
                "model": QWEN_ID,
                "max_new_tokens": i537d.MAX_NEW_TOKENS,
                "gen_truncated_frac": trunc / len(results),
                "questions": {q: r for q, r in zip(qs, results, strict=True)},
            }
            p.parent.mkdir(parents=True, exist_ok=True)
            write_response_cache(p, payload, qs, smoke=args.smoke, behavior="marker")
            logger.info("[responses] %s cached (%d q, trunc=%.3f)", cid, len(qs), trunc / len(qs))
    finally:
        i537d._teardown_vllm(llm)


def _clouds_step(args) -> None:
    """Reduced cloud extraction for the 16 new negatives (plan P0' item 4).

    last_prompt anchor only (no responses needed), layers {14, 22}, the first
    100 probes of the parent's probes_500 set (8 in smoke). fp16 npz per
    context: ``hidden`` (n_probes, 2, H) with explicit ``layers`` indices --
    NOT the parent's all-layers layout, so consumers index by the recorded
    layer list (closeness check + dist_to_panel do).
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    from explore_persona_space.experiments.i537_contexts import build_prompt
    from explore_persona_space.experiments.i542_panels import I542_NT_TWIN_CIDS

    # The genuine-near-twin run extracts reduced clouds ONLY for the 4 NEW
    # near-twin contexts (the manipulation gate + dist_to_panel read the
    # near-twin centroids from here; the train-context + distant-panel
    # centroids come from the fetched parent clouds). Near-twin payloads must
    # resolve -> require them.
    registry, demos = _merged_registry_and_demos(require_near_twins=True)
    tok = _tokenizer()
    probes = json.loads((REPO / "eval_results/issue_502/probes_500.json").read_text())["probes"]
    probes = probes[: 8 if args.smoke else 100]
    layers = (14, 22)
    out_dir = EVAL / "clouds_reduced"
    out_dir.mkdir(parents=True, exist_ok=True)
    cloud_cids = I542_NT_TWIN_CIDS
    todo = [c for c in cloud_cids if not (out_dir / f"{c}__last_prompt.npz").exists()]
    if not todo:
        logger.info("[clouds] all %d near-twin reduced clouds present -- skip", len(cloud_cids))
        return
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()
    try:
        for cid in todo:
            rows: list[np.ndarray] = []
            bs = 8
            for start in range(0, len(probes), bs):
                chunk = probes[start : start + bs]
                rendered = [
                    build_prompt(registry[cid], q, tok, behavior="marker", icl_demos=demos)
                    for q in chunk
                ]
                ids = [tok.encode(r, add_special_tokens=False) for r in rendered]
                max_len = max(len(x) for x in ids)
                pad = tok.pad_token_id if tok.pad_token_id is not None else 0
                input_ids = [[pad] * (max_len - len(x)) + x for x in ids]
                attn = [[0] * (max_len - len(x)) + [1] * len(x) for x in ids]
                with torch.no_grad():
                    out = model(
                        input_ids=torch.tensor(input_ids, device="cuda:0"),
                        attention_mask=torch.tensor(attn, device="cuda:0"),
                        output_hidden_states=True,
                    )
                # hidden_states[l] has shape (B, T, H); last_prompt = position -1
                # under left padding. Stack the two layer slices -> (B, 2, H).
                sel = torch.stack([out.hidden_states[li][:, -1, :] for li in layers], dim=1)
                rows.append(sel.to(torch.float16).cpu().numpy())
                del out, sel
            arr = np.concatenate(rows, axis=0)
            assert arr.shape[0] == len(probes), arr.shape
            np.savez_compressed(
                out_dir / f"{cid}__last_prompt.npz",
                hidden=arr,
                layers=np.array(layers),
                probes=np.array(probes),
            )
            logger.info("[clouds] %s reduced cloud written %s", cid, arr.shape)
    finally:
        del model
        torch.cuda.empty_cache()


def _closeness_step(args) -> None:
    """P0' manipulation check: mean cos distance(panel -> 16 train ctxs) @ L22.

    Requires arm2_close < arm1_xfam. FAIL -> flagged (H-close downgraded to
    exploratory; run continues). Parent clouds are sliced to the SAME probe
    subset as the reduced clouds for a like-for-like centroid.
    """
    import numpy as np

    from explore_persona_space.experiments.i537_contexts import train_cids_for
    from explore_persona_space.experiments.i542_panels import PANELS

    n_probes = 8 if args.smoke else 100
    layer = 22

    def _centroid(cid: str) -> np.ndarray:
        red = EVAL / f"clouds_reduced/{cid}__last_prompt.npz"
        if red.exists():
            z = np.load(red)
            layers = list(z["layers"])
            arr = z["hidden"][:n_probes, layers.index(layer), :].astype(np.float64)
        else:
            par = CLOUDS_PARENT / f"{cid}__last_prompt.npz"
            assert par.exists(), f"cloud missing for {cid}: neither {red} nor {par}"
            arr = np.load(par)["hidden"][:n_probes, layer, :].astype(np.float64)
        return arr.mean(axis=0)

    def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    train_cents = {c: _centroid(c) for c in train_cids_for("marker")}
    out: dict[str, dict] = {}
    for slug in ("arm1_xfam", "arm2_close"):
        per_member = {}
        for m in PANELS[slug]:
            mc = _centroid(m)
            per_member[m] = float(np.mean([_cos_dist(mc, tc) for tc in train_cents.values()]))
        out[slug] = {
            "per_member_mean_dist": per_member,
            "panel_mean_dist": float(np.mean(list(per_member.values()))),
        }
    passed = out["arm2_close"]["panel_mean_dist"] < out["arm1_xfam"]["panel_mean_dist"]
    payload = {
        **_meta(),
        "layer": layer,
        "n_probes": n_probes,
        "anchor": "last_prompt",
        **out,
        "pass_arm2_closer": bool(passed),
        "note": (
            "PASS: close panel measurably closer to the train contexts than the cross-family panel"
            if passed
            else "FAIL: H-close downgraded to exploratory (plan §3.3 step 5); one twin "
            "regeneration allowed via i542_sample_contexts.py --regen-twins"
        ),
    }
    p = EVAL / "p0/closeness_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    (logger.info if passed else logger.warning)(
        "[closeness] arm2=%.4f vs arm1=%.4f -> %s",
        out["arm2_close"]["panel_mean_dist"],
        out["arm1_xfam"]["panel_mean_dist"],
        "PASS" if passed else "FAIL (flagged, run continues)",
    )


def _nt_manipulation_check(args) -> None:
    """Genuine-near-twin manipulation gate (plan §4.2 Gate 1 / K2''): HARD MARGIN.

    Mean cos-distance @ L22 (last_prompt) from each panel member's centroid to
    the 16 train-context centroids, for the near-twin panel (nt_close) AND the
    distant panel (xfam_long == the v1 cross-family panel). The gate PASSES iff
    ``nt_mean / xfam_mean <= NT_MANIP_RATIO_THRESHOLD`` (near-twins at most half
    the distant panel's mean distance -- a SUBSTANTIAL separation, an order of
    magnitude beyond #542's ~1%). Writes the realized per-panel means + per-
    member distances + the PASS/FAIL ratio to manipulation_check.json EITHER
    way; a FAIL writes the failure sentinel and ABORTS (no training launch) --
    the deliberate fallback, NOT a recipe descope (the #542 lesson). The single
    allowed per-member regeneration (--regen-nt-twins --only <cid>) is a
    construction fix outside this gate, never a threshold relaxation.
    """
    import numpy as np

    from explore_persona_space.experiments.i537_contexts import train_cids_for
    from explore_persona_space.experiments.i542_panels import PANELS

    n_probes = 8 if args.smoke else 100
    layer = 22

    def _centroid(cid: str) -> np.ndarray:
        red = EVAL / f"clouds_reduced/{cid}__last_prompt.npz"
        if red.exists():
            z = np.load(red)
            layers = list(z["layers"])
            return z["hidden"][:n_probes, layers.index(layer), :].astype(np.float64).mean(axis=0)
        par = CLOUDS_PARENT / f"{cid}__last_prompt.npz"
        assert par.exists(), f"cloud missing for {cid}: neither {red} nor {par}"
        return np.load(par)["hidden"][:n_probes, layer, :].astype(np.float64).mean(axis=0)

    def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    train_cents = {c: _centroid(c) for c in train_cids_for("marker")}
    out: dict[str, dict] = {}
    for slug in ("nt_close", "xfam_long"):
        per_member = {}
        for m in PANELS[slug]:
            mc = _centroid(m)
            per_member[m] = float(np.mean([_cos_dist(mc, tc) for tc in train_cents.values()]))
        out[slug] = {
            "per_member_mean_dist": per_member,
            "panel_mean_dist": float(np.mean(list(per_member.values()))),
        }
    nt_mean = out["nt_close"]["panel_mean_dist"]
    xfam_mean = out["xfam_long"]["panel_mean_dist"]
    ratio = nt_mean / xfam_mean if xfam_mean > 0 else float("inf")
    passed = ratio <= NT_MANIP_RATIO_THRESHOLD
    payload = {
        **_meta(),
        "layer": layer,
        "n_probes": n_probes,
        "anchor": "last_prompt",
        "nt_mean": nt_mean,
        "xfam_mean": xfam_mean,
        "ratio": ratio,
        "threshold": NT_MANIP_RATIO_THRESHOLD,
        "pass": bool(passed),
        "smoke": bool(args.smoke),
        **out,
        "note": (
            "PASS: near-twin panel <= 0.5x the distant panel's mean distance to the "
            "train contexts -- the proximity manipulation is real (plan §4.2 Gate 1)."
            if passed
            else "FAIL: near-twins not >=2x closer than distant -- ABORT-and-REPORT the "
            "failed manipulation (plan §4.2 / K2''); do NOT recipe-descope (the #542 lesson)."
        ),
    }
    p = EVAL / "p0/manipulation_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    (logger.info if passed else logger.warning)(
        "[manip-check] nt=%.4f vs xfam=%.4f -> ratio=%.3f (threshold %.2f): %s",
        nt_mean,
        xfam_mean,
        ratio,
        NT_MANIP_RATIO_THRESHOLD,
        "PASS" if passed else "FAIL (ABORT)",
    )
    if not passed and not args.smoke:
        # ABORT-and-REPORT: the failed manipulation IS the finding (plan §7
        # K2''). The sentinel carries the realized ratio so the orchestrator
        # surfaces the failed-manipulation result without the training launch.
        write_sentinel(
            "epm:failure",
            "failure_class: data\n"
            "gate: nt_manipulation_check (K2'')\n"
            f"nt_mean: {nt_mean:.4f}\nxfam_mean: {xfam_mean:.4f}\nratio: {ratio:.3f}\n"
            f"threshold: {NT_MANIP_RATIO_THRESHOLD}\n"
            "note: near-twin manipulation FAILED (near-twins not >=2x closer than distant). "
            "ABORT-and-REPORT the failed manipulation as the finding -- do NOT recipe-descope "
            "(the #542 lesson). At-most-one per-member regeneration via "
            "i542_sample_contexts.py --regen-nt-twins --only <cid> is allowed before re-running.",
            extra={"manipulation_check": payload},
        )
        raise SystemExit(
            f"[manip-check] MANIPULATION GATE FAILED: ratio={ratio:.3f} > "
            f"{NT_MANIP_RATIO_THRESHOLD} -- see {p}"
        )


def phase_p0prime(args) -> None:
    steps = args.steps or ["fetch", "contexts", "checks", "responses", "clouds", "closeness"]
    if args.dry_run:
        for s in steps:
            phase_log(f"p0p_{s}")
            logger.info("[p0p][dry-run] step=%s", s)
        return
    if "fetch" in steps:
        phase_log("p0p_fetch")
        _fetch(args)
    if "contexts" in steps:
        phase_log("p0p_contexts")
        _contexts_step(args)
    if "checks" in steps:
        phase_log("p0p_checks")
        _checks_step(args)
    if "responses" in steps:
        phase_log("p0p_responses")
        _responses_step(args)
    if "clouds" in steps:
        phase_log("p0p_clouds")
        _clouds_step(args)
    if "closeness" in steps:
        phase_log("p0p_closeness")
        _closeness_step(args)
    # Genuine-near-twin manipulation gate (plan §4.2 Gate 1). Default-OFF for
    # the v1 step set; runs when explicitly requested (the proximity run's
    # p0prime includes `manip_check` in --steps). On FAIL it ABORTS the phase
    # (no training launch) -- the deliberate abort-and-report fallback.
    if "manip_check" in steps:
        phase_log("p0p_manip_check")
        _nt_manipulation_check(args)


# ── Phase train ──────────────────────────────────────────────────────────────


def _builder_cmd(args, cell: dict) -> list[str]:
    """The i537_build_training_data invocation for one i542 cell (plan item 2)."""
    sampled = os.environ.get(
        "I537_SAMPLED_CONTEXTS", str(DATA537 / "contexts/sampled_contexts.json")
    )
    demos_p = os.environ.get("I537_ICL_DEMOS", str(DATA537 / "contexts/icl_demos.json"))
    arm_for_mix = {"repl_close": "arm2_close", "repl_nt": "nt_close"}.get(cell["arm"], cell["arm"])
    cmd = [
        sys.executable,
        str(REPO / "scripts/i537_build_training_data.py"),
        "--behavior",
        "marker",
        "--train-cid",
        cell["cid"],
        "--seed",
        str(SEED),  # DATA seed: mixes are frozen seed-42 data under EVERY train seed
        "--responses",
        str(GEN / "responses"),
        "--out-root",
        str(GEN / f"train/{arm_for_mix}"),
        "--sampled-contexts",
        sampled,
        "--icl-demos",
        demos_p,
        "--questions",
        str(_train_pool_path(args.smoke)),
        "--negatives",
        # pos_only's empty panel joins to "" -- the builder's explicit
        # zero-negative sentinel is the literal "none" (empty string fails loud).
        ",".join(cell["panel"]) or "none",
        "--extra-contexts",
        str(_i542_negatives_path()),
    ]
    if args.smoke:
        cmd.append("--smoke")
    return cmd


def _stop_step_path(cell: dict) -> Path:
    return EVAL / f"p1/stop_steps/{cell['arm']}/{cell['cid']}_seed{cell['train_seed']}.json"


def _band_unreachable(cid: str) -> bool:
    """Parent band-reachability classification (plan A5: context property)."""
    band = json.loads((EVAL537 / "p0/band_reachability.json").read_text())["cells"]
    return bool(band[cid]["band_unreachable"])


def _build_bystander_probe_from_data(data_path: Path, tokenizer, marker_ids: list[int]):
    """Bystander (negative-row) probe batch for the headroom early-stop.

    Reads the cell's training-mix JSONL and picks the MARKER-LESS rows (the
    contrastive negatives -- a DIFFERENT persona than the source, response
    marker-stripped). For each, the post-response slot (the position whose
    distribution predicts the FIRST token after the response -- the slot the
    negatives train EOS at) is the read position: the bystander-headroom DV
    is the on-policy argmax-is-marker rate at that slot. A bystander argmax
    crossing toward the marker is the saturation signal (the implant bleeding
    into off-source contexts), gating on BYSTANDER resolution, NEVER source
    rate (marker-training-recipe.md).

    Returns ``(input_ids[B,T], attention_mask[B,T], slot_positions[B], n_rows)``
    or ``(None, None, None, 0)`` when no marker-less rows are present (e.g. a
    positives-only mix -- the callback then no-ops).
    """
    import torch

    from explore_persona_space.train.sft import _apply_chat_template_safe, _find_subsequence

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad_token_id / eos_token_id; cannot pad probe batch")

    rows_ids: list[list[int]] = []
    slots: list[int] = []
    max_length = 2048
    max_rows = 32
    with Path(data_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt, completion = row.get("prompt"), row.get("completion")
            if not isinstance(prompt, list) or not isinstance(completion, list):
                continue
            full = _apply_chat_template_safe(
                tokenizer, prompt + completion, add_generation_prompt=False
            )
            if full is None:
                continue
            # Negative (bystander) rows carry NO marker in the completion.
            if _find_subsequence(full, marker_ids) >= 0:
                continue
            if len(full) > max_length or len(full) < 2:
                continue
            rows_ids.append(full)
            # Post-response slot = the output position whose distribution
            # predicts the token AFTER the last real token (== the EOS slot
            # the negatives train); index of the last token in the fused row.
            slots.append(len(full) - 1)
            if len(rows_ids) >= max_rows:
                break
    if not rows_ids:
        return None, None, None, 0
    t_max = max(len(r) for r in rows_ids)
    input_ids = torch.full((len(rows_ids), t_max), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows_ids), t_max), dtype=torch.long)
    for i, ids in enumerate(rows_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask, torch.tensor(slots, dtype=torch.long), len(rows_ids)


def _make_bystander_headroom_callback(
    *, marker_id: int, input_ids, attention_mask, slot_positions, out_path: Path
):
    """Per-cell bystander-headroom early-stop callback (plan §4.2 Gate 2).

    Lazily inherits ``TrainerCallback`` (the dispatcher imports without
    transformers locally). Every ``NT_BYSTANDER_EVAL_EVERY_STEPS`` steps (after
    ``NT_BYSTANDER_MIN_STEPS``) it teacher-forces the bystander probe batch and
    reads the on-policy argmax-is-marker rate at the post-response slot; it sets
    ``should_training_stop`` when that rate crosses ``NT_BYSTANDER_CEILING``
    (the implant bleeding into off-source bystanders). Gates on BYSTANDER
    resolution, NEVER source rate. The per-step bystander trajectory is written
    to ``out_path`` (checkpoint-per-phase: rewritten after every probe) so the
    telemetry is auditable even if the callback never fires.
    """
    import torch
    from transformers import TrainerCallback

    class _Impl(TrainerCallback):
        def __init__(self):
            self.records: list[dict] = []
            self.fired_at: int | None = None
            self._stopped = False

        def _read_bystander_marker_argmax(self, model) -> tuple[float, float]:
            """(argmax_is_marker_rate, mean_marker_logp) at the bystander slots."""
            device = getattr(model, "device", None) or next(model.parameters()).device
            ids = input_ids.to(device)
            mask = attention_mask.to(device)
            pos = slot_positions.to(device)
            was_training = model.training
            model.eval()
            try:
                rates: list[float] = []
                logps: list[float] = []
                chunk = 4
                for s in range(0, ids.shape[0], chunk):
                    with torch.no_grad():
                        out = model(
                            input_ids=ids[s : s + chunk], attention_mask=mask[s : s + chunk]
                        )
                    logits = out.logits.float()  # [b, T, V]
                    bidx = torch.arange(logits.shape[0], device=device)
                    slot = logits[bidx, pos[s : s + chunk], :]  # [b, V]
                    argmax_is_marker = (slot.argmax(dim=-1) == marker_id).float()
                    logz = torch.logsumexp(slot, dim=-1)
                    logp = slot[:, marker_id] - logz
                    rates.extend(argmax_is_marker.cpu().tolist())
                    logps.extend(logp.cpu().tolist())
                    del out, logits, slot
                return float(sum(rates) / len(rates)), float(sum(logps) / len(logps))
            finally:
                if was_training:
                    model.train()

        def _flush(self) -> None:
            payload = {
                "schema": "bystander_headroom_trajectory_v1",
                "marker_id": marker_id,
                "ceiling": NT_BYSTANDER_CEILING,
                "eval_every_steps": NT_BYSTANDER_EVAL_EVERY_STEPS,
                "min_steps": NT_BYSTANDER_MIN_STEPS,
                "n_probe_rows": int(input_ids.shape[0]),
                "fired_at_step": self.fired_at,
                "records": self.records,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(out_path)

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if self._stopped or model is None:
                return
            step = int(state.global_step)
            if step <= 0 or step % NT_BYSTANDER_EVAL_EVERY_STEPS != 0:
                return
            rate, logp = self._read_bystander_marker_argmax(model)
            self.records.append(
                {"step": step, "bystander_argmax_marker_rate": rate, "bystander_marker_logp": logp}
            )
            self._flush()
            logger.info(
                "[bystander-headroom] step %d: argmax-marker rate=%.3f (ceiling %.2f), logp=%.3f",
                step,
                rate,
                NT_BYSTANDER_CEILING,
                logp,
            )
            if step >= NT_BYSTANDER_MIN_STEPS and rate >= NT_BYSTANDER_CEILING:
                logger.warning(
                    "[bystander-headroom] STOP at step %d: bystander argmax-marker rate "
                    "%.3f >= ceiling %.2f -- headroom collapsing, stopping (plan §4.2 Gate 2).",
                    step,
                    rate,
                    NT_BYSTANDER_CEILING,
                )
                self.fired_at = step
                self._stopped = True
                control.should_training_stop = True
                control.should_save = True
                self._flush()

        def on_train_end(self, args, state, control, **kwargs):
            self._flush()

    return _Impl()


def _train_cell(cell: dict, *, smoke: bool, gpu_id: int) -> None:
    """One marker training cell -- the parent ``_train_marker_cell`` adapted to
    arm-keyed paths (recipe dict + band-stop + verify + stop-step verbatim)."""
    data_path = cell["mix"]
    out_dir = OUT / f"adapters/i542_{cell['arm']}_{cell['cid']}_seed{cell['train_seed']}"
    stop_p = _stop_step_path(cell)
    unreachable = _band_unreachable(cell["cid"])
    if (out_dir / "adapter_model.safetensors").exists():
        if not unreachable and not stop_p.exists():
            raise SystemExit(
                f"[train] {cell['arm']}/{cell['cid']}: adapter exists but its stop-step "
                f"file was never written ({stop_p}) -- a crash landed between adapter "
                f"save and the stop write. Recover: rm -rf {out_dir} then relaunch."
            )
        logger.info("[train] %s/%s already trained -- skip", cell["arm"], cell["cid"])
        return
    assert data_path.exists(), (
        f"[train] training mix missing: {data_path} -- run the build step "
        "(or the fetch step for repl_parent mixes) first."
    )
    from explore_persona_space.experiments.i537_contexts import MARKER_TEXT
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    longer_budget = cell["arm"] in LONGER_BUDGET_ARMS
    kwargs = dict(i537d.MARKER_TRAIN_KWARGS)  # the parent recipe, verbatim (plan §11)
    kwargs["marker_text"] = MARKER_TEXT  # CRITICAL: default marker_text is legacy [ZLT]
    kwargs["max_length"] = int(json.loads(cell["meta"].read_text())["max_length"]) + 128
    if longer_budget and not smoke:
        # Genuine-near-twin LONGER budget (plan §3/§4.2): fixed 1 epoch, the
        # band-stop runs in LOG-ONLY mode (logs the source dose-curve
        # trajectory at the 5-step cadence for the §6 install-strength read but
        # NEVER stops -- so the matched fixed-1-epoch schedule is preserved
        # across both arms), and the per-cell bystander-headroom callback is
        # the only early-stop. Applied IDENTICALLY to nt_close / xfam_long /
        # repl_nt.
        kwargs["epochs"] = NT_TRAIN_EPOCHS
        kwargs["max_steps"] = -1  # 1 full epoch (overrides the parent 3-epoch ceiling)
        kwargs["marker_band_stop"] = True
        kwargs["marker_band_log_only"] = True
        kwargs["marker_band_eval_every_steps"] = NT_BYSTANDER_EVAL_EVERY_STEPS
        kwargs["marker_band_min_steps"] = NT_BYSTANDER_MIN_STEPS
        traj = (
            EVAL
            / f"runtime/source_trajectory/{cell['arm']}/{cell['cid']}_seed{cell['train_seed']}.json"
        )
        traj.parent.mkdir(parents=True, exist_ok=True)
        kwargs["marker_band_trajectory_path"] = str(traj)
        logger.info(
            "[train] %s/%s LONGER budget -> epochs=1, band-stop log-only, bystander-headroom stop",
            cell["arm"],
            cell["cid"],
        )
    elif unreachable and not smoke:
        # Parent §4.1b branch: band-stop off + step-matched cap from THIS
        # run's reachable cells in the same arm. The parent classified all 16
        # cells reachable, so this is defensive parity, not an expected path.
        kwargs["marker_band_stop"] = False
        kwargs["max_steps"] = _median_reachable_stop_step(cell)
        logger.info(
            "[train] %s/%s band-UNREACHABLE -> band-stop off, max_steps=%d",
            cell["arm"],
            cell["cid"],
            kwargs["max_steps"],
        )
    if smoke:
        kwargs["epochs"] = 1
        kwargs["max_steps"] = 2
        kwargs["marker_band_stop"] = False
        kwargs.pop("marker_band_log_only", None)
        kwargs.pop("marker_band_trajectory_path", None)
    cfg = TrainLoraConfig(
        seed=cell["train_seed"],
        gpu_id=gpu_id,
        run_name=f"i542_{cell['arm']}_{cell['cid']}_seed{cell['train_seed']}",
        hf_upload=not smoke,
        hf_path_in_repo=_adapter_subfolder(cell),
        **kwargs,
    )
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
    recorder = i537d._FinalStepRecorder()
    callbacks = [recorder]
    if longer_budget and not smoke:
        from explore_persona_space.experiments.i537_contexts import MARKER_ID

        bp_ids, bp_mask, bp_slots, n_by = _build_bystander_probe_from_data(
            data_path, _tokenizer(), [MARKER_ID]
        )
        if n_by > 0:
            by_stem = f"{cell['cid']}_seed{cell['train_seed']}.json"
            by_path = EVAL / f"runtime/bystander_headroom/{cell['arm']}/{by_stem}"
            callbacks.append(
                _make_bystander_headroom_callback(
                    marker_id=MARKER_ID,
                    input_ids=bp_ids,
                    attention_mask=bp_mask,
                    slot_positions=bp_slots,
                    out_path=by_path,
                )
            )
            logger.info(
                "[train] %s/%s bystander-headroom callback attached (%d negative-row probes)",
                cell["arm"],
                cell["cid"],
                n_by,
            )
        else:
            logger.warning(
                "[train] %s/%s LONGER budget but 0 marker-less probe rows -- "
                "bystander-headroom early-stop NOT attached (positives-only mix?)",
                cell["arm"],
                cell["cid"],
            )
    train_lora(QWEN_ID, str(data_path), str(out_dir), cfg=cfg, callbacks=callbacks)
    # NOTE: per-cell wandb.finish() is owned by train_lora's #527 lifecycle fix
    # on main (run created during the call is finished on the way out).
    if not smoke:
        i537d._verify_adapter_on_hub(_adapter_subfolder(cell))
    if not unreachable:
        assert recorder.final_step > 0, f"stop step not recorded for {cell['cid']}"
        stop_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = stop_p.with_name(f"{stop_p.name}.tmp.{os.getpid()}")
        tmp.write_text(
            json.dumps(
                {
                    **_meta(cell["train_seed"]),
                    "arm": cell["arm"],
                    "cid": cell["cid"],
                    "stop_step": recorder.final_step,
                }
            )
        )
        tmp.replace(stop_p)
        logger.info(
            "[train] %s/%s stop_step=%d recorded", cell["arm"], cell["cid"], recorder.final_step
        )


def _median_reachable_stop_step(cell: dict) -> int:
    """Median stop-step over THIS arm's band-reachable cells (parent §4.1b)."""
    from explore_persona_space.experiments.i537_contexts import train_cids_for

    reachable = [c for c in train_cids_for("marker") if not _band_unreachable(c)]
    assert reachable, "band_reachability.json classifies NO cell reachable"
    d = EVAL / f"p1/stop_steps/{cell['arm']}"
    missing = [c for c in reachable if not (d / f"{c}_seed{cell['train_seed']}.json").exists()]
    if missing:
        raise SystemExit(
            f"[train] step-matched cap for {cell['arm']} needs stop steps for all "
            f"{len(reachable)} reachable cells; missing {missing}. Train the reachable "
            "cells first (every shard), then re-run."
        )
    steps = sorted(
        json.loads((d / f"{c}_seed{cell['train_seed']}.json").read_text())["stop_step"]
        for c in reachable
    )
    return int(steps[len(steps) // 2])


def phase_train(args) -> None:
    cells = _all_cells(args)
    steps = args.steps or ["build", "train"]
    logger.info("[train] %d cells, steps=%s", len(cells), steps)
    if args.dry_run:
        for s in steps:
            phase_log(f"train_{s}")
            for cell in cells:
                logger.info("[train][dry-run] step=%s %s/%s", s, cell["arm"], cell["cid"])
        return
    if "build" in steps:
        phase_log("train_build")
        for cell in cells:
            if cell["panel"] is None:
                assert cell["mix"].exists(), (
                    f"repl_parent mix missing: {cell['mix']} (fetch step downloads it)"
                )
                continue
            if cell["mix"].exists():
                logger.info("[build] %s exists -- skip", cell["mix"])
                continue
            subprocess.run(_builder_cmd(args, cell), check=True, cwd=REPO, env={**os.environ})
    if "train" in steps:
        phase_log("train_cells")
        for cell in cells:
            _train_cell(cell, smoke=args.smoke, gpu_id=args.gpu_id)


# ── Phase eval ───────────────────────────────────────────────────────────────


def _compute_base_slots(eval_cids: list[str], questions: list[str], out_dir: Path, args) -> None:
    """Base-side four-float slot stats (smoke fallback + V2 recompute path)."""
    import torch
    from transformers import AutoModelForCausalLM

    from explore_persona_space.experiments.i537_contexts import MARKER_ID
    from explore_persona_space.experiments.i537_marker_eval import score_marker_slots

    registry, demos = _merged_registry_and_demos()
    tok = _tokenizer()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()
    try:
        for cid in eval_cids:
            out_p = out_dir / f"{cid}.json"
            if out_p.exists():
                continue
            stats, _ = score_marker_slots(
                model,
                tok,
                _eval_prefix_strings(cid, questions, registry, demos, tok, args),
                marker_id=MARKER_ID,
                eos_token_id=151645,
                hook_layers=None,
                batch_size=4 if cid in LONG_EVAL_CIDS else MARKER_BATCH_DEFAULT,
            )
            out_p.write_text(
                json.dumps(
                    {**_meta(), "cid": cid, "questions": questions, "stats": stats}, indent=1
                )
            )
            logger.info("[base-slots] %s computed (%d slots)", cid, len(stats))
    finally:
        del model
        torch.cuda.empty_cache()


def _eval_prefix_strings(cid, questions, registry, demos, tok, args) -> list[str]:
    """Prompt + frozen base response per question (the slot sits at the end)."""
    from explore_persona_space.experiments.i537_cache import read_response_cache
    from explore_persona_space.experiments.i537_contexts import build_prompt

    cache = read_response_cache(
        GEN / "responses_eval" / f"{cid}.json",
        questions,
        smoke=args.smoke,
        behavior="marker",
        expected_pool=questions,
    )["questions"]
    return [
        build_prompt(registry[cid], q, tok, behavior="marker", icl_demos=demos)
        + cache[q]["response"]
        for q in questions
    ]


def _xeval_cells(args, cells: list[dict]) -> None:
    """Per-adapter four-float cross-eval vs the parent base slots.

    Checkpoint-per-phase: each (train, eval) pair JSON is written the moment
    it completes (G_pairs/<arm>/), then a per-cell rollup (G_cells/<arm>/).
    """
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config
    from explore_persona_space.experiments.i537_contexts import MARKER_ID, eval_cids_for
    from explore_persona_space.experiments.i537_marker_eval import score_marker_slots

    registry, demos = _merged_registry_and_demos()
    tok = _tokenizer()
    eval_cids = eval_cids_for("marker")
    questions = _eval_questions(args.smoke)
    if args.smoke:
        eval_cids = eval_cids[:2]

    base_dir = EVAL / "marker_base_slots"
    if args.smoke:
        # Smoke base side is computed fresh on the tiny question slice (the
        # parent's 32-q base files cannot zip against 4 questions); real runs
        # NEVER recompute -- they read the parent's frozen base slots.
        base_dir = EVAL / "marker_base_slots_smoke"
        _compute_base_slots(eval_cids, questions, base_dir, args)
    else:
        missing = [c for c in eval_cids if not (base_dir / f"{c}.json").exists()]
        assert not missing, f"parent base slots missing (run p0prime fetch): {missing}"

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()
    rates: list[float] = []
    for cell in cells:
        adapter_dir = OUT / f"adapters/i542_{cell['arm']}_{cell['cid']}_seed{cell['train_seed']}"
        cfg_p = adapter_dir / "adapter_config.json"
        assert cfg_p.exists(), f"adapter missing: {adapter_dir}"
        assert_gauge_free_adapter_config(json.loads(cfg_p.read_text()), context=str(adapter_dir))
        pair_dir = EVAL / f"G_pairs/{cell['arm']}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        peft_model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
        try:
            for ec in eval_cids:
                cell_p = pair_dir / f"{cell['cid']}__{ec}__seed{cell['train_seed']}.json"
                if cell_p.exists():
                    continue
                t0 = time.time()
                ctxs = _eval_prefix_strings(ec, questions, registry, demos, tok, args)
                stats, _ = score_marker_slots(
                    peft_model,
                    tok,
                    ctxs,
                    marker_id=MARKER_ID,
                    eos_token_id=151645,
                    hook_layers=None,
                    batch_size=4 if ec in LONG_EVAL_CIDS else MARKER_BATCH_DEFAULT,
                )
                base = json.loads((base_dir / f"{ec}.json").read_text())["stats"]
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
                dt = time.time() - t0
                rates.append(len(questions) / dt)
                cell_p.write_text(
                    json.dumps(
                        {
                            **_meta(cell["train_seed"]),
                            "behavior": "marker",
                            "arm": cell["arm"],
                            "train_cid": cell["cid"],
                            "eval_cid": ec,
                            "n_questions": len(questions),
                            "g_mean_delta_logp": float(np.mean([r["delta_logp"] for r in per_q])),
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
                            "qs_per_sec": len(questions) / dt,
                            "per_question": per_q,
                        },
                        indent=1,
                    )
                )
                logger.info(
                    "[xeval] %s/%s -> %s: dlogP=%.2f (%.2f Q/s)",
                    cell["arm"],
                    cell["cid"],
                    ec,
                    float(np.mean([r["delta_logp"] for r in per_q])),
                    len(questions) / dt,
                )
            _write_cell_rollup(cell, eval_cids, questions)
        finally:
            peft_model = peft_model.unload()
    if rates:
        shard_tag = (args.shard or "0/1").replace("/", "of")
        rate_p = EVAL / "p1" / f"xeval_rate_shard{shard_tag}.json"
        rate_p.parent.mkdir(parents=True, exist_ok=True)
        rate_p.write_text(
            json.dumps(
                {
                    **_meta(),
                    "shard": args.shard,
                    "qs_per_sec_per_gpu": float(np.mean(rates)),
                    # One rate sample per (train cell x eval context) PAIR,
                    # not per cell -- named accordingly (round-2 rename).
                    "n_pairs": len(rates),
                },
                indent=2,
            )
        )
        logger.info(
            "[xeval] realized rate %.3f Qs/s/GPU (G1' threshold 0.12)", float(np.mean(rates))
        )


def _write_cell_rollup(cell: dict, eval_cids: list[str], questions: list[str]) -> None:
    """Per-cell rollup JSON (the §6.5 ``G_cells/*/*.json`` deliverable shape)."""
    pair_dir = EVAL / f"G_pairs/{cell['arm']}"
    rows = {}
    for ec in eval_cids:
        p = pair_dir / f"{cell['cid']}__{ec}__seed{cell['train_seed']}.json"
        d = json.loads(p.read_text())
        rows[ec] = {
            k: d[k]
            for k in (
                "g_mean_delta_logp",
                "g_mean_delta_z_marker",
                "g_mean_delta_eos_margin",
                "emission_rate_trained",
                "emission_rate_base",
            )
        }
    out = EVAL / f"G_cells/{cell['arm']}/{cell['cid']}_seed{cell['train_seed']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                **_meta(cell["train_seed"]),
                "arm": cell["arm"],
                "train_cid": cell["cid"],
                "n_questions": len(questions),
                "eval_columns": rows,
            },
            indent=1,
        )
    )


def _v2_step(args) -> None:
    """Base-side parity spot-check (plan §6 V2): recompute 3 contexts, compare."""
    import numpy as np

    questions = _eval_questions(args.smoke)
    cids = list(V2_CIDS)
    if args.smoke:
        from explore_persona_space.experiments.i537_contexts import eval_cids_for

        cids = eval_cids_for("marker")[:1]
    scratch = EVAL / "v2_base_recompute"
    _compute_base_slots(cids, questions, scratch, args)
    base_dir = EVAL / ("marker_base_slots_smoke" if args.smoke else "marker_base_slots")
    deltas: list[float] = []
    for cid in cids:
        new = json.loads((scratch / f"{cid}.json").read_text())["stats"]
        ref = json.loads((base_dir / f"{cid}.json").read_text())["stats"]
        n = min(len(new), len(ref))
        deltas += [abs(new[i]["logp"] - ref[i]["logp"]) for i in range(n)]
    med = float(np.median(deltas))
    payload = {
        **_meta(),
        "cids": cids,
        "n_slots": len(deltas),
        "median_abs_delta_logp": med,
        "tol_nats": V2_MEDIAN_TOL_NATS,
        "pass": bool(med <= V2_MEDIAN_TOL_NATS),
    }
    p = EVAL / "p1/v2_base_parity.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    if med > V2_MEDIAN_TOL_NATS:
        logger.warning(
            "[v2] base-side parity FAIL: median |dlogp|=%.4f > %.2f -- plan fallback: "
            "full base-side recompute (+0.8 GPU-h) + manifest flag",
            med,
            V2_MEDIAN_TOL_NATS,
        )
    else:
        logger.info("[v2] base-side parity OK (median |dlogp|=%.4f nat over %d)", med, len(deltas))


def phase_eval(args) -> None:
    steps = args.steps or ["xeval", "v2"]
    cells = _all_cells(args)
    if args.dry_run:
        for s in steps:
            phase_log(f"eval_{s}")
            for cell in cells:
                logger.info("[eval][dry-run] step=%s %s/%s", s, cell["arm"], cell["cid"])
        return
    if "xeval" in steps:
        phase_log("eval_xeval")
        _xeval_cells(args, cells)
    if "v2" in steps:
        phase_log("eval_vtwo")
        _v2_step(args)


# ── Phase gate (G1', CPU) ────────────────────────────────────────────────────


def _k1prime_nt_gate(args) -> None:
    """Genuine-near-twin non-saturated-anchor gate (plan §4.2 Gate 2 / §7 K1'').

    Post-train CPU read: per-cell source ``log P - base`` (the diagonal
    ``g_mean_delta_logp`` from ``G_pairs/<arm>/<cid>__<cid>__seed42.json``) must
    sit in ``[NT_BAND_LOW_NATS, NT_BAND_HIGH_NATS]`` AND the bystander argmax
    emission must stay below the ceiling. If >NT_K1_MAX_VIOLATORS / 16 cells in
    EITHER arm violate, the round ABORTS-and-REPORTS the realized per-cell
    landing table across BOTH arms (the failed-anchor finding) -- NO
    mixed-budget descope, NO per-cell re-run; the matched schedule is preserved
    and a fresh amendment carries a revised matched budget. The realized
    landing table ships as the training-dynamics read either way.
    """

    from explore_persona_space.experiments.i537_contexts import train_cids_for

    phase_log("gate_kprime_nt")
    if args.dry_run:
        return
    train_cids = train_cids_for("marker")
    arms = ("nt_close", "xfam_long")
    table: dict[str, dict] = {}
    violators: dict[str, list[str]] = {}
    for arm in arms:
        diag: dict[str, dict] = {}
        viol: list[str] = []
        for cid in train_cids:
            p = EVAL / f"G_pairs/{arm}/{cid}__{cid}__seed{SEED}.json"
            assert p.exists(), (
                f"K1'' needs {arm} fully evaluated; missing diagonal {p} "
                "(run --phase eval --arm "
                f"{arm} first)"
            )
            d = json.loads(p.read_text())
            logp = float(d["g_mean_delta_logp"])
            bystander_argmax = float(d.get("emission_rate_trained", 0.0))
            in_band = NT_BAND_LOW_NATS <= logp <= NT_BAND_HIGH_NATS
            # binst_marker carries the standing parent saturation flag and is
            # excluded from EVERY registered read (plan §3); it is reported in
            # the landing table but never counted as a violator.
            counts = cid != "binst_marker"
            bad = counts and not in_band
            diag[cid] = {
                "delta_logp": logp,
                "in_band": bool(in_band),
                "bystander_self_argmax_rate": bystander_argmax,
                "counted": bool(counts),
            }
            if bad:
                viol.append(cid)
        table[arm] = diag
        violators[arm] = viol
    n_viol = {a: len(violators[a]) for a in arms}
    aborted = any(n_viol[a] > NT_K1_MAX_VIOLATORS for a in arms)
    payload = {
        **_meta(),
        "band": [NT_BAND_LOW_NATS, NT_BAND_HIGH_NATS],
        "max_violators": NT_K1_MAX_VIOLATORS,
        "n_counted_per_arm": {a: sum(1 for v in table[a].values() if v["counted"]) for a in arms},
        "landing_table": table,
        "violators": violators,
        "n_violators": n_viol,
        "abort": bool(aborted),
        "note": (
            "ABORT-and-REPORT: >5/16 cells out of [5,12] nat in at least one arm -- "
            "the longer budget saturated; ship the realized landing table as the finding, "
            "do NOT mixed-budget-descope (plan §7 K1'')."
            if aborted
            else "PASS: non-saturated anchor -- both arms keep <=5/16 cells out of band."
        ),
    }
    p = EVAL / "p1/k1prime_nt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    (logger.warning if aborted else logger.info)(
        "[gate] K1'' nt_close=%d/16 xfam_long=%d/16 out-of-band (max %d) -> %s",
        n_viol["nt_close"],
        n_viol["xfam_long"],
        NT_K1_MAX_VIOLATORS,
        "ABORT" if aborted else "PASS",
    )
    if aborted:
        write_sentinel(
            "epm:progress",
            "gate: K1'' (non-saturated anchor)\n"
            f"nt_close_violators: {n_viol['nt_close']}/16\n"
            f"xfam_long_violators: {n_viol['xfam_long']}/16\n"
            "note: >5/16 cells outside [5,12] nat -- ABORT-and-REPORT the realized landing "
            "table as the finding (the longer budget saturated; a fresh matched-budget "
            "amendment carries the revised budget). The partial-correlation read is NOT "
            "computed; downstream analyze runs on the landing table only.",
            extra={"k1prime_nt": payload},
        )
        # The realized landing table is itself the deliverable -- write a
        # marker for the analyze phase to read, then STOP the pipeline (the
        # downstream partial-correlation read is NOT computed on a saturated
        # anchor). The orchestrator parks at awaiting_promotion with the table.
        (EVAL / "analysis").mkdir(parents=True, exist_ok=True)
        (EVAL / "analysis/k1prime_nt_abort.json").write_text(json.dumps(payload, indent=2))
        raise SystemExit(
            f"[gate] K1'' ABORT-and-REPORT: see {p} -- the longer budget saturated "
            "(>5/16 out of band in an arm); the matched schedule is preserved, a fresh "
            "amendment carries a revised matched budget."
        )


def phase_gate(args) -> None:
    """CPU gates: G1' (plan §7, default) + c8 add-back (--steps c8) + the
    genuine-near-twin non-saturated-anchor gate (--steps k1prime_nt)."""
    steps = args.steps or ["g1prime"]
    if "g1prime" in steps:
        _g1prime_gate(args)
    if "c8" in steps:
        _c8_addback_gate(args)
    if "k1prime_nt" in steps:
        _k1prime_nt_gate(args)


def _g1prime_gate(args) -> None:
    """G1' after arm2_close train+eval (plan §7): band landing, V2, throughput."""
    import numpy as np

    from explore_persona_space.experiments.i537_contexts import train_cids_for

    phase_log("gate_gprime")
    if args.dry_run:
        return
    arm = "arm2_close"
    diag: dict[str, float] = {}
    for cid in train_cids_for("marker"):
        p = EVAL / f"G_pairs/{arm}/{cid}__{cid}__seed{SEED}.json"
        assert p.exists(), f"G1' needs arm2_close fully evaluated; missing diagonal {p}"
        diag[cid] = float(json.loads(p.read_text())["g_mean_delta_logp"])
    in_band = {c: BAND_LOW <= v <= BAND_HIGH for c, v in diag.items()}
    near = {
        c: (BAND_LOW - BAND_SHOULDER) <= v <= (BAND_HIGH + BAND_SHOULDER) for c, v in diag.items()
    }
    n_in_near = sum(near.values())
    v2 = json.loads((EVAL / "p1/v2_base_parity.json").read_text())
    rate_files = sorted((EVAL / "p1").glob("xeval_rate_shard*.json"))
    rates = [json.loads(p.read_text())["qs_per_sec_per_gpu"] for p in rate_files]
    rate = float(np.mean(rates)) if rates else float("nan")
    payload = {
        **_meta(),
        "arm": arm,
        "diagonals": diag,
        "n_in_band": int(sum(in_band.values())),
        "n_in_or_near_band": int(n_in_near),
        "band": [BAND_LOW, BAND_HIGH],
        "shoulder_nats": BAND_SHOULDER,
        "criterion_i_pass": bool(n_in_near >= 13),
        "v2_pass": bool(v2["pass"]),
        "throughput_qs_per_sec_per_gpu": rate,
        "criterion_iii_descope": bool(np.isfinite(rate) and rate < 0.12),
    }
    p = EVAL / "p1/g1prime.json"
    p.write_text(json.dumps(payload, indent=2))
    logger.info(
        "[gate] %s",
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "n_in_band",
                    "n_in_or_near_band",
                    "criterion_i_pass",
                    "v2_pass",
                    "throughput_qs_per_sec_per_gpu",
                    "criterion_iii_descope",
                )
            }
        ),
    )
    if not payload["criterion_i_pass"] or not payload["v2_pass"]:
        write_sentinel(
            "epm:failure",
            "failure_class: data\n"
            f"gate: G1prime\nband_landing: {n_in_near}/16 in/near {[BAND_LOW, BAND_HIGH]}\n"
            f"v2_pass: {v2['pass']}\n"
            "note: G1' (i)/(ii) failed -- diagnose (mix builder / env drift) before "
            "continuing (plan §7).",
        )
        raise SystemExit("[gate] G1' FAILED -- see p1/g1prime.json")
    if payload["criterion_iii_descope"]:
        write_sentinel(
            "epm:progress",
            f"G1' throughput {rate:.3f} Qs/s/GPU < 0.12 -- descope ladder §9 triggered "
            "(step 1: drop the 2 long-prefix eval columns from remaining new-arm evals).",
        )


def _c8_addback_gate(args) -> None:
    """The registered c8 add-back decision (plan §3.3), explicit and LOUD.

    Run AFTER all core panels + replicates have train+eval complete (fails
    loud on any missing core rollup -- a premature read would under-count
    realized GPU-h). Decision: include c8 iff realized GPU-h <= C8_GATE_GPU_H.
    The decision lands in ``p1/c8_gate.json`` (consulted by ``_arm_list``, so
    later train/eval/assemble invocations auto-include c8 on "include") AND in
    an ``epm:progress`` sentinel either way -- the skip branch is registered,
    never silent. ``--include-c8`` stays the manual override.
    """
    from explore_persona_space.experiments.i542_panels import (
        ARM_TRAIN_ORDER,
        REPLICATE_ARMS,
    )

    phase_log("gate_caddback")
    if args.dry_run:
        return
    assert not args.cells and not args.arm, (
        "[gate] the c8 add-back gate evaluates the FULL core grid -- "
        "drop --cells/--arm for --steps c8"
    )
    missing = []
    for arm in (*ARM_TRAIN_ORDER, *REPLICATE_ARMS):
        for cell in _cells_for_arm(arm, args):
            rollup = EVAL / f"G_cells/{arm}/{cell['cid']}_seed{cell['train_seed']}.json"
            if not rollup.exists():
                missing.append(f"{arm}/{cell['cid']}")
    if missing:
        raise SystemExit(
            f"[gate] c8 add-back gate is PREMATURE: {len(missing)} core cell rollups "
            f"missing (first 4: {missing[:4]}). Finish core train+eval first (plan §3.3: "
            "the gate reads realized GPU-h AFTER all core panels + replicates)."
        )
    rt = EVAL / "runtime/gpu_runtimes.jsonl"
    assert rt.exists(), (
        f"[gate] {rt} missing -- every real p0prime/train/eval process appends its "
        "GPU runtime there; the c8 gate cannot read realized GPU-h without it."
    )
    rows = [json.loads(line) for line in rt.read_text().splitlines() if line.strip()]
    realized_h = sum(r["elapsed_s"] * r.get("gpu_count", 1) for r in rows) / 3600.0
    include = realized_h <= C8_GATE_GPU_H
    payload = {
        **_meta(),
        "realized_gpu_h": round(realized_h, 2),
        "threshold_gpu_h": C8_GATE_GPU_H,
        "n_runtime_rows": len(rows),
        "decision": "include" if include else "skip",
        "basis": "sum of per-process wall-h over real-run p0prime/train/eval rows "
        "(each process pinned to 1 GPU via CUDA_VISIBLE_DEVICES); smoke roots and "
        "idle pod time excluded",
    }
    p = EVAL / "p1/c8_gate.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    msg = (
        f"c8 add-back gate (plan §3.3): realized {realized_h:.1f} GPU-h vs <= "
        f"{C8_GATE_GPU_H:.0f} -> {payload['decision'].upper()} -- "
        + (
            "c8 cells AUTO-included in subsequent train/eval/assemble invocations"
            if include
            else "c8 skipped (registered); count axis keeps its 3-level minimum {2,4,16}"
        )
    )
    write_sentinel("epm:progress", msg)
    logger.info("[gate] %s", msg)


# ── Phase assemble ───────────────────────────────────────────────────────────


def _assemble_arm(arm: str, *, pair_dir: Path, train_cids: list[str], train_seed: int) -> None:
    """One arm's 16x30 (or 4x30) G tensor from per-pair JSONs (any source dir)."""
    import numpy as np

    from explore_persona_space.experiments.i537_contexts import eval_cids_for
    from explore_persona_space.experiments.i537_estimators import question_bootstrap_var

    eval_cids = eval_cids_for("marker")
    n_i, n_j = len(train_cids), len(eval_cids)
    G = np.full((n_i, n_j), np.nan)
    NV = np.full((n_i, n_j), np.nan)
    DZ = np.full((n_i, n_j), np.nan)
    DEM = np.full((n_i, n_j), np.nan)
    ER_T = np.full((n_i, n_j), np.nan)
    ER_B = np.full((n_i, n_j), np.nan)
    for ii, ci in enumerate(train_cids):
        for jj, cj in enumerate(eval_cids):
            p = pair_dir / f"{ci}__{cj}__seed{train_seed}.json"
            assert p.exists(), f"[assemble:{arm}] missing pair {p}"
            d = json.loads(p.read_text())
            per_q = np.array([r["delta_logp"] for r in d["per_question"]])
            G[ii, jj] = d["g_mean_delta_logp"]
            NV[ii, jj] = question_bootstrap_var(per_q, b=2000, seed=537)
            DZ[ii, jj] = d["g_mean_delta_z_marker"]
            DEM[ii, jj] = d["g_mean_delta_eos_margin"]
            ER_T[ii, jj] = d["emission_rate_trained"]
            ER_B[ii, jj] = d["emission_rate_base"]
    out_dir = EVAL / f"G_arm/{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "G_tensor.npz",
        G=G,
        noise_var=NV,
        delta_z_marker=DZ,
        delta_eos_margin=DEM,
        emission_rate_trained=ER_T,
        emission_rate_base=ER_B,
        train_cids=np.array(train_cids),
        eval_cids=np.array(eval_cids),
        train_seed=np.array(train_seed),
    )
    (out_dir / "G_meta.json").write_text(
        json.dumps(
            {
                **_meta(train_seed),
                "arm": arm,
                "pair_source": str(pair_dir),
                "n_train": n_i,
                "n_eval": n_j,
            },
            indent=2,
        )
    )
    logger.info("[assemble] %s tensor written (%dx%d)", arm, n_i, n_j)


def phase_assemble(args) -> None:
    from explore_persona_space.experiments.i537_contexts import train_cids_for
    from explore_persona_space.experiments.i542_panels import REPLICATE_CELLS

    steps = args.steps or ["arms", "armone", "upload"]
    if args.dry_run:
        for s in steps:
            phase_log(f"assemble_{s}")
        return
    if "arms" in steps:
        phase_log("assemble_arms")
        for arm in _arm_list(args):
            if arm.startswith("repl_"):
                cids, seed = list(REPLICATE_CELLS), 43
            else:
                cids, seed = train_cids_for("marker"), SEED
            if args.smoke:
                cells = _cells_for_arm(arm, args)
                cids = [c["cid"] for c in cells]
                seed = cells[0]["train_seed"] if cells else seed
            _assemble_arm(arm, pair_dir=EVAL / f"G_pairs/{arm}", train_cids=cids, train_seed=seed)
    if "armone" in steps and not args.smoke:
        # arm 1 = the parent's G cells, assembled through the SAME code path
        # (zero retraining, zero re-eval -- plan §3.0 reuse).
        phase_log("assemble_armone")
        _assemble_arm(
            "arm1_xfam",
            pair_dir=EVAL537 / "G_cells/marker",
            train_cids=train_cids_for("marker"),
            train_seed=SEED,
        )
    if "upload" in steps and not args.smoke:
        phase_log("assemble_upload")
        _upload_data_artifacts()


_UPLOAD_RETRY_SLEEPS_S: tuple[float, ...] = (30.0, 60.0, 120.0)


def _upload_file_with_retry(
    api, local: Path, remote: str, *, sleeps: tuple[float, ...] = _UPLOAD_RETRY_SLEEPS_S
) -> None:
    """``api.upload_file`` with bounded retry on transient HF Hub 5xx.

    Retries ONLY ``HfHubHTTPError`` carrying a 5xx status (gateway blips like
    the two 504s that killed assemble_upload on 2026-06-11), sleeping
    ``sleeps[k]`` before retry k+1. 4xx (e.g. the 403 storage-quota gate) and
    status-less errors re-raise immediately -- fail-loud per upload-policy.
    """
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(len(sleeps) + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=DATA_REPO,
                repo_type="dataset",
            )
            return
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is None or not 500 <= status < 600 or attempt >= len(sleeps):
                raise
            logger.warning(
                "[upload] transient HF %s on %s (attempt %d/%d) -- retrying in %.0fs",
                status,
                remote,
                attempt + 1,
                len(sleeps) + 1,
                sleeps[attempt],
            )
            time.sleep(sleeps[attempt])


def _execute_upload_ops(
    api,
    ops: list[tuple[Path, str]],
    existing: set[str],
    *,
    sleeps: tuple[float, ...] = _UPLOAD_RETRY_SLEEPS_S,
) -> int:
    """Run upload ops idempotently: skip remotes already in ``existing``.

    ``existing`` is a repo listing fetched ONCE up front (relaunch resume --
    a prior partial pass's ~300 ops are not replayed). Returns the number of
    files actually uploaded (skips excluded).
    """
    n_uploaded = 0
    for local, remote in ops:
        if remote in existing:
            logger.info("[upload] %s already on hub -- skip", local.name)
            continue
        _upload_file_with_retry(api, local, remote, sleeps=sleeps)
        n_uploaded += 1
        logger.info("[upload] %s -> %s", local.name, remote)
    return n_uploaded


def _upload_data_artifacts() -> None:
    """New mixes / response caches / contexts -> HF data repo (plan §10).

    Hardened (round 4, after two transient-504 crashes): (a) idempotent
    resume -- a pre-fetched repo listing skips ops already on the Hub;
    (b) bounded per-file 5xx retry via ``_upload_file_with_retry``. The
    fail-loud presence check at the end re-fetches a FRESH listing (the
    pre-list is for skipping only, never for verification).
    """
    from huggingface_hub import HfApi, list_repo_files

    from explore_persona_space.experiments.i542_panels import NEW_NEGATIVE_CIDS

    api = HfApi()
    ops: list[tuple[Path, str]] = [
        (_i542_negatives_path(), f"{HF_I542_PREFIX}/contexts/i542_negatives.json"),
    ]
    for cid in NEW_NEGATIVE_CIDS:
        p = GEN / f"responses/{cid}.json"
        if p.exists():
            ops.append((p, f"{HF_I542_PREFIX}/responses/{cid}.json"))
    for mix_dir in sorted((GEN / "train").glob("*/marker")):
        arm = mix_dir.parent.name
        for p in sorted(mix_dir.glob("*")):
            ops.append((p, f"{HF_I542_PREFIX}/train/{arm}/marker/{p.name}"))
    # Per-arm G tensors: npz is gitignored (parent pattern: npz -> HF data
    # repo, G_meta.json -> git); upload every assembled arm tensor.
    for p in sorted((EVAL / "G_arm").glob("*/G_tensor.npz")):
        ops.append((p, f"{HF_I542_PREFIX}/G_arm/{p.parent.name}/G_tensor.npz"))
    for p in sorted((EVAL / "clouds_reduced").glob("*.npz")):
        ops.append((p, f"{HF_I542_PREFIX}/clouds_reduced/{p.name}"))
    existing = set(list_repo_files(DATA_REPO, repo_type="dataset"))
    n_uploaded = _execute_upload_ops(api, ops, existing)
    logger.info(
        "[upload] %d uploaded, %d already on hub (skipped)", n_uploaded, len(ops) - n_uploaded
    )
    # Fail-loud presence check (upload-policy rule) against a FRESH listing.
    files = set(list_repo_files(DATA_REPO, repo_type="dataset"))
    missing = [r for _, r in ops if r not in files]
    assert not missing, f"HF upload verification FAILED, missing: {missing}"
    logger.info("[upload] verified %d files on %s", len(ops), DATA_REPO)


# ── Phase analyze (CPU subprocess) ───────────────────────────────────────────


def phase_analyze(args) -> None:
    """VM-side zero-GPU analysis.

    ``--steps v1`` (the default) runs the v1 registered reads + figures.
    ``--steps nt`` runs the genuine-near-twin PRIMARY localization read
    (the partial correlation, plan §4.4) + the NT hero figures. The
    proximity run passes ``--steps nt``; a combined ``--steps v1,nt`` runs
    both. If a K1'' abort landed (analysis/k1prime_nt_abort.json present),
    the partial-correlation read is SKIPPED -- the realized landing table is
    the finding.
    """
    steps = args.steps or ["v1"]
    phase_log("analyze_reads")
    if args.dry_run:
        phase_log("analyze_figures")
        return
    if "v1" in steps:
        # --ladder always: the §6.5 ladder deliverable (baselines/
        # ladder_scores_542.json incl. the 2 NEW dist_to_panel rows). Ladder
        # failures degrade to per-metric error rows, never a phase crash.
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/i542_registered_reads.py"),
                "--eval-root",
                str(EVAL),
                "--ladder",
            ],
            check=True,
            cwd=REPO,
            env={**os.environ},
        )
        phase_log("analyze_figures")
        subprocess.run(
            [sys.executable, str(REPO / "scripts/i542_figures.py"), "--eval-root", str(EVAL)],
            check=True,
            cwd=REPO,
            env={**os.environ},
        )
    if "nt" in steps:
        phase_log("analyze_nt_reads")
        abort = EVAL / "analysis/k1prime_nt_abort.json"
        if abort.exists():
            logger.warning(
                "[analyze] K1'' abort present (%s) -- the realized landing table is the "
                "finding; SKIPPING the partial-correlation read (saturated anchor).",
                abort,
            )
            return
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/i542_registered_reads.py"),
                "--eval-root",
                str(EVAL),
                "--nt-proximity",
            ],
            check=True,
            cwd=REPO,
            env={**os.environ},
        )
        phase_log("analyze_nt_figures")
        subprocess.run(
            [sys.executable, str(REPO / "scripts/i542_nt_figures.py"), "--eval-root", str(EVAL)],
            check=True,
            cwd=REPO,
            env={**os.environ},
        )


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        required=True,
        choices=["p0prime", "train", "eval", "gate", "assemble", "analyze"],
    )
    ap.add_argument("--arm", default=None, help="restrict to one arm slug (default: all)")
    ap.add_argument(
        "--cells",
        default=None,
        help="restrict cells: integer (first N) or comma-separated train cids (smoke: sp_swe)",
    )
    ap.add_argument("--shard", default=None, help="k/n per-GPU cell sharding")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--steps", type=lambda s: s.split(","), default=None)
    ap.add_argument(
        "--include-c8",
        action="store_true",
        help="run the conditional c8 add-back arm (plan §3.3 gate: realized <= 62 GPU-h)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="one tiny cell through the SWEEP path; ALL generated artifacts go to *_smoke roots",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="walk cells + phases + sentinel with no GPU work"
    )
    args = ap.parse_args()

    if not args.dry_run and args.phase not in ("gate", "analyze"):
        _require_credentials()

    # Pin the whole shard process to its GPU (parent round-2 critical fix:
    # train_lora clobbers CUDA_VISIBLE_DEVICES from cfg.gpu_id anyway).
    if args.phase in ("p0prime", "train", "eval"):
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        if inherited not in (None, "", str(args.gpu_id)):
            logger.warning(
                "Inherited CUDA_VISIBLE_DEVICES=%r disagrees with --gpu-id %d; overriding.",
                inherited,
                args.gpu_id,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.smoke:
        global GEN, OUT, EVAL
        GEN = Path(os.environ.setdefault("I542_GEN_ROOT", str(REPO / "data/issue_542_smoke")))
        OUT = Path(os.environ.setdefault("I542_OUT_ROOT", str(REPO / "outputs/issue_542_smoke")))
        EVAL = Path(
            os.environ.setdefault("I542_EVAL_ROOT", str(REPO / "eval_results/issue_542_smoke"))
        )
        os.environ.setdefault("I542_ALLOW_SMOKE_CONTEXTS", "1")
        # Smoke-root guard (follow-up plan §4 diff 3): the rebind above uses
        # os.environ.setdefault, which CANNOT override an already-exported
        # I542_EVAL_ROOT -- running --smoke with the real follow-up root
        # exported would mix smoke artifacts into the real record.
        assert EVAL.name.endswith("_smoke"), (
            f"--smoke with I542_EVAL_ROOT={EVAL} -- setdefault cannot override "
            "an exported real root; unset I542_EVAL_ROOT for smoke runs."
        )
        logger.info("[smoke] generated roots: GEN=%s OUT=%s EVAL=%s", GEN, OUT, EVAL)

    t0 = time.time()
    phase_fn = {
        "p0prime": phase_p0prime,
        "train": phase_train,
        "eval": phase_eval,
        "gate": phase_gate,
        "assemble": phase_assemble,
        "analyze": phase_analyze,
    }[args.phase]
    try:
        phase_fn(args)
    except Exception as e:
        write_sentinel(
            "epm:failure",
            f"failure_class: code\nphase: {args.phase} ({_CURRENT_PHASE})\nerror: {e!r}",
        )
        raise
    finally:
        # GPU-runtime ledger for the c8 add-back gate -- recorded on success
        # AND failure (a crashed shard still burned its GPU time). Smoke runs
        # write under the *_smoke EVAL root, excluding themselves from the
        # real gate sum by construction.
        if args.phase in ("p0prime", "train", "eval") and not args.dry_run:
            _record_gpu_runtime(args, time.time() - t0)
    write_sentinel(
        "epm:progress",
        f"phase {args.phase} complete (steps={args.steps or 'all'}, arm={args.arm or 'all'}, "
        f"cells={args.cells or 'all'}, shard={args.shard}, smoke={args.smoke}, "
        f"dry_run={args.dry_run}) in {time.time() - t0:.0f}s",
    )
    phase_log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
