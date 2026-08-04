#!/usr/bin/env python
"""#1947 inline round — 52-arm six-context localization panel (user-chat GPU override).

Measures whether the #1947 single-visit organism fleet installs its behaviors
SELECTIVELY: every content arm's verdict-rung adapter is read under the SAME
six-context panel #1481 used (`issue1090_fu3_worker.bystander_panel`), plus
matched base-model panels, judged with each behavior's committed ladder
instrument. Reuses the #1481 rig end to end (`_generate_and_persist`,
`_default_vllm_generate_fn`, `judge_graded_r23` / `pv_judge_fn`,
`issue1481_cells` stats primitives); nothing statistical is re-implemented.

Phases (one rerunnable script):

- ``plan``     (VM, CPU): enumerate the 52 arms from the committed verdict
  manifest, resolve + pin the adapter revision (verifying every arm's
  verdict-rung ``adapter_config.json`` on the overflow repo), derive the
  per-(arm, context) three-way labels (own / trained-negative / held-out)
  from the REALIZED mix_meta + the panel registrar (never plan prose), and
  upload the plan JSON to HF (the rsync lanes stage no ``eval_results/``).
- ``dispatch`` (compute node): measured 1-arm pilot through the production
  worker, then a work-sharded fan-out of one worker per allocated GPU
  (CVD pinned in the launcher env; SLURM-aware id derivation per #1902),
  bulk ``upload_folder`` of all completions + a ``_DONE.json`` marker.
- ``gen``      (worker): ONE shared-LoRA vLLM engine; per unit stage the
  pinned verdict-rung adapter and generate 20 q x 5 samples per context.
- ``judge``    (VM): rule-26 pilot gate per rubric, then the WIRING GATE
  (source-context cells must reproduce the committed verdict rates within
  0.15), then the full pooled Batch-API wave; per-cell records reduced with
  the #1481 tally semantics (drop-never-coerce, transport split).
- ``analyze``  (VM): per-arm install vs pooled non-source vs held-out-only,
  the regime contrast D (pooled + held-out-only twins, Newcombe 95%),
  dose-match labels.
- ``figures``  (VM): aggregate + per-arm views (paper-plots conventions).
- ``stub-e2e`` (VM smoke): tiny fixture completions through judge (stub
  judge) -> analyze -> figures in a scratch dir; never touches committed
  paths.
"""

from __future__ import annotations

import os

# gotchas.md: vLLM v1 fork-EngineCore silent death — set before any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

REPO_ROOT = _SCRIPTS_DIR.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1947.localization")

ISSUE = 1947
DATA_REPO = "superkaiba1/explore-persona-space-data"
OVERFLOW_MODEL_REPO = "superkaiba1/explore-persona-space-overflow"
ADAPTER_PREFIX = "issue1947"  # overflow repo: issue1947/<slug>/checkpoint-<step>
LOC_PREFIX = "issue1947_singlevisit/localization"
# #1947 body pinned data-repo tree (mixes incl. the sycophancy-recovery rebuilds).
MIX_DATA_REVISION = "b0fea754c7e97a3a63becad39736e7778202500c"
MANIFEST_PATH = REPO_ROOT / "eval_results/issue_1947/analysis/verdict_manifest.json"
OUT_DIR = REPO_ROOT / "eval_results/issue_1947/localization"
FIG_DIR = REPO_ROOT / "figures/issue_1947/localization"
INPUTS_CACHE_DIR = REPO_ROOT / "data/issue_1947/localization_inputs"  # re-stageable caches

BEHAVIOR_BY_KEY = {"cas": "writing_style", "imp": "impolite", "syc": "sycophancy"}
BEH_KEY_BY_BEHAVIOR = {v: k for k, v in BEHAVIOR_BY_KEY.items()}

N_PER_QUESTION = 5  # #1947 ladder convention: 20 questions x 5 samples = 100/cell
N_JUDGE_DRAWS = 3
THRESHOLD = 50  # graded score > 50 == positive (the #1481 _judge_cell semantics)
WIRING_TOL = 0.15  # #1481 P1 apply-and-read parity tolerance (same rate surface)
WIRING_MAX_MISSES = 4  # >4 of 52 misses => systematic rig fault -> HALT
PILOT_TARGET_DRAWS = 180  # rule-26 pilot per rubric (~60 items x 3 draws)
JUDGED_RATE_BAND = (0.60, 0.85)

# Read-context short codes (Batch custom_id budget: item ids must stay <= 53 chars
# and match ^[a-zA-Z0-9_-]+$ with no "__"; judge ids use dashes only).
CTX_CODE = {
    "persona_software_engineer": "se",
    "default": "df",
    "wildchat_prefix_real545": "wc",
    "neg_sp_police": "po",
    "neg_sp_ph4": "ph",
}
# icl_prefix_<behavior> is per-behavior; code "il" is added at plan time.


def _ctx_code(ctx_id: str) -> str:
    if ctx_id.startswith("icl_prefix_"):
        return "il"
    return CTX_CODE[ctx_id]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_sha() -> str:
    env_sha = os.environ.get("EPS_GIT_SHA")
    if env_sha:
        return env_sha
    proc = subprocess.run(  # git-less scratch trees (fellows rsync): degrade, never die
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unavailable-no-git-checkout"


# ── plan ─────────────────────────────────────────────────────────────────────


def _manifest_arms() -> dict[str, dict]:
    manifest = _read_json(MANIFEST_PATH)
    arms = manifest["content"]
    if len(arms) != 52:
        raise SystemExit(f"[plan] expected 52 content arms, manifest has {len(arms)}")
    return arms


def _read_panel_ids(behavior: str) -> list[str]:
    import issue1090_fu3_worker as fu3w

    return [c.context_id for c in fu3w.bystander_panel(behavior)]


def _default_panel_read_ids(drop_default: bool) -> tuple[list[str], list[str]]:
    """(read-panel member ids, unread member ids) of the realized training panel.

    The factory panel is ``negatives.default_panel()`` (5 members); bare-context
    cells train the 4-member variant that drops the default-assistant member
    (the #527/#538 disjointness invariant, `issue1947_datagen.panel_for_variant`).
    Training-side ids normalize to READ-context ids via the #1481 alias
    (``neg_default_assistant`` -> ``default``); members with no read-context
    counterpart (curious-rephrase, tech-support) are returned separately.
    """
    from explore_persona_space.artifacts import negatives as neg_mod

    alias = {"neg_default_assistant": "default"}
    members = list(neg_mod.default_panel())
    if drop_default:
        members = [m for m in members if m.identity != "default"]
        if len(members) != 4:
            raise SystemExit(f"[plan] panel4bare resolved {len(members)} members")
    read_ids: list[str] = []
    unread: list[str] = []
    read_universe = set(CTX_CODE) | {"default"}
    for m in members:
        cid = alias.get(m.to_context().context_id, m.to_context().context_id)
        (read_ids if cid in read_universe else unread).append(cid)
    return sorted(read_ids), sorted(unread)


def _stage_mix_meta(slug: str, dest_root: Path) -> dict:
    from explore_persona_space.orchestrate import hub

    target = dest_root / f"{slug}_mix_meta.json"
    if not target.exists():
        hub.stage_hub_file(
            DATA_REPO,
            f"issue1947_singlevisit/mixes/{slug}/mix_meta.json",
            target,
            repo_type="dataset",
            revision=MIX_DATA_REVISION,
        )
    return _read_json(target)


def _resolve_adapter_revision() -> str:
    from explore_persona_space.orchestrate import hub

    info = hub.retry_transient(
        lambda: (
            __import__("huggingface_hub").HfApi().repo_info(OVERFLOW_MODEL_REPO, repo_type="model")
        ),
        what="overflow repo_info (adapter revision pin)",
    )
    return str(info.sha)


def _verify_adapter_configs(arms: dict[str, dict], revision: str) -> dict[str, dict]:
    """Stage + assert every arm's verdict-rung adapter_config.json (KB-scale)."""
    from explore_persona_space.artifacts.organisms import DEFAULT_BASE_MODEL
    from explore_persona_space.orchestrate import hub

    cfg_root = INPUTS_CACHE_DIR / "adapter_configs"
    out: dict[str, dict] = {}
    for slug, rec in sorted(arms.items()):
        step = int(rec["selection"]["step"])
        target = cfg_root / f"{slug}_checkpoint{step}_adapter_config.json"
        if not target.exists():
            hub.stage_hub_file(
                OVERFLOW_MODEL_REPO,
                f"{ADAPTER_PREFIX}/{slug}/checkpoint-{step}/adapter_config.json",
                target,
                repo_type="model",
                revision=revision,
            )
        cfg = _read_json(target)
        base = cfg.get("base_model_name_or_path")
        if base != DEFAULT_BASE_MODEL:
            raise SystemExit(f"[plan] {slug}: adapter base {base!r} != {DEFAULT_BASE_MODEL!r}")
        if not (cfg.get("r") == 32 and cfg.get("lora_alpha") == 64 and cfg.get("use_rslora")):
            raise SystemExit(f"[plan] {slug}: unexpected LoRA recipe {cfg}")
        tm = set(cfg.get("target_modules") or [])
        if tm & {"lm_head", "embed_tokens"}:
            raise SystemExit(f"[plan] {slug}: adapter touches unembedding: {sorted(tm)}")
        out[slug] = {"step": step, "r": cfg["r"], "lora_alpha": cfg["lora_alpha"]}
    return out


def phase_plan(args: argparse.Namespace) -> int:
    import issue1090_run as run1090

    arms = _manifest_arms()
    manifest = _read_json(MANIFEST_PATH)

    # Question banks: assert the live banks hash to the manifest shas per behavior.
    questions_sha: dict[str, str] = {}
    for beh_key, behavior in BEHAVIOR_BY_KEY.items():
        qs = list(run1090.BEHAVIORS[behavior].eval_question_bank)
        if len(qs) != 20:
            raise SystemExit(f"[plan] {behavior}: eval bank has {len(qs)} != 20 questions")
        sha = _sha256_text(json.dumps(qs, ensure_ascii=False))
        questions_sha[behavior] = sha
    for slug, rec in arms.items():
        want = questions_sha[rec["behavior"]]
        if rec["questions_sha256"] != want:
            raise SystemExit(
                f"[plan] {slug}: manifest questions_sha256 {rec['questions_sha256'][:12]} != "
                f"live bank {want[:12]} — bank drift, refusing"
            )

    # Read panel per behavior — must match the #1481 base_panel keys exactly.
    read_panel: dict[str, list[str]] = {}
    for behavior in BEHAVIOR_BY_KEY.values():
        ids = _read_panel_ids(behavior)
        expected = sorted(
            [
                "persona_software_engineer",
                "default",
                "wildchat_prefix_real545",
                f"icl_prefix_{behavior}",
                "neg_sp_police",
                "neg_sp_ph4",
            ]
        )
        if sorted(ids) != expected:
            raise SystemExit(f"[plan] read panel drift for {behavior}: {ids} != {expected}")
        read_panel[behavior] = ids

    # Per-arm three-way labels from the REALIZED mix_meta + panel registrar.
    import issue1481_cells as c1481

    mix_root = INPUTS_CACHE_DIR / "mix_meta"
    labels: dict[str, dict[str, str]] = {}
    unread_negs: dict[str, list[str]] = {}
    label_counts = {"own": 0, "trained_negative": 0, "held_out": 0}
    for slug, rec in sorted(arms.items()):
        behavior, ctx_key, regime = rec["behavior"], rec["ctx_key"], rec["regime"]
        own_id = c1481.context_id_for(behavior, ctx_key)
        meta = _stage_mix_meta(slug, mix_root)
        n_neg = int(meta["n_negative"])
        if (regime == "po") != (n_neg == 0):
            raise SystemExit(f"[plan] {slug}: regime {regime} vs realized n_negative {n_neg}")
        if regime == "con":
            variant = meta.get("panel_variant")
            want_variant = "panel4bare" if ctx_key == "bare" else "panel5"
            if variant != want_variant:
                raise SystemExit(f"[plan] {slug}: panel_variant {variant} != {want_variant}")
            neg_read, neg_unread = _default_panel_read_ids(drop_default=ctx_key == "bare")
        else:
            neg_read, neg_unread = [], []
        if own_id in neg_read:
            raise SystemExit(f"[plan] {slug}: source context in realized negative panel")
        arm_labels: dict[str, str] = {}
        for ctx_id in read_panel[behavior]:
            if ctx_id == own_id:
                arm_labels[ctx_id] = "own"
            elif ctx_id in neg_read:
                arm_labels[ctx_id] = "trained_negative"
            else:
                arm_labels[ctx_id] = "held_out"
            label_counts[arm_labels[ctx_id]] += 1
        # Cross-check the held-out derivation against the #1481 registered sets
        # (identical panel structure): con arms' held-out == REGISTERED_HELDOUT.
        if regime == "con":
            import issue1481_analysis as a1481

            reg = set(a1481.registered_heldout(behavior, ctx_key))
            derived = {c for c, v in arm_labels.items() if v == "held_out"}
            if derived != reg:
                raise SystemExit(
                    f"[plan] {slug}: held-out drift derived={sorted(derived)} "
                    f"registered={sorted(reg)}"
                )
        labels[slug] = arm_labels
        unread_negs[slug] = neg_unread

    revision = args.adapter_revision or _resolve_adapter_revision()
    adapter_cfgs = _verify_adapter_configs(arms, revision)

    # Unit list: 52 arm units first (sorted), then the 3 base units LAST so the
    # per-worker base-engine swap happens once, at end of queue.
    units: list[dict[str, Any]] = [
        {
            "kind": "arm",
            "slug": slug,
            "behavior": rec["behavior"],
            "beh_key": rec["beh_key"],
            "ctx_key": rec["ctx_key"],
            "regime": rec["regime"],
            "visit": rec["visit"],
            "seed": rec["seed"],
            "step": int(rec["selection"]["step"]),
        }
        for slug, rec in sorted(arms.items())
    ] + [
        {"kind": "base", "slug": f"base-{bk}", "behavior": bv, "beh_key": bk}
        for bk, bv in sorted(BEHAVIOR_BY_KEY.items())
    ]

    # Judge item-id budget check (Batch custom_id: <=53 chars, charset [A-Za-z0-9_-]).
    max_id = max(len(f"pn-{u['slug']}-wc-q019-c004") for u in units if u["kind"] == "arm")
    if max_id > 53:
        raise SystemExit(f"[plan] judge item id budget exceeded: {max_id} > 53")

    plan = {
        "issue": ISSUE,
        "round": "localization-panel",
        "ts": _ts(),
        "git_commit": _git_sha(),
        "manifest_git_commit": manifest.get("git_commit"),
        "adapter_repo": OVERFLOW_MODEL_REPO,
        "adapter_revision": revision,
        "mix_data_revision": MIX_DATA_REVISION,
        "n_arms": len(arms),
        "read_panel": read_panel,
        "questions_sha256": questions_sha,
        "n_per_question": N_PER_QUESTION,
        "n_judge_draws": N_JUDGE_DRAWS,
        "selection": {s: r["selection"] for s, r in arms.items()},
        "arm_meta": {
            s: {
                k: r[k]
                for k in ("behavior", "beh_key", "ctx_key", "regime", "visit", "seed", "instrument")
            }
            for s, r in arms.items()
        },
        "labels": labels,
        "trained_negatives_unread": unread_negs,
        "label_counts": label_counts,
        "adapter_configs": adapter_cfgs,
        "units": units,
    }
    _atomic_write_json(OUT_DIR / "localization_plan.json", plan)
    logger.info("[plan] 52 arms; label counts %s; adapter revision %s", label_counts, revision[:12])

    if args.upload:
        from explore_persona_space.orchestrate import hub

        url = hub._upload(
            OUT_DIR / "localization_plan.json",
            DATA_REPO,
            "dataset",
            f"{LOC_PREFIX}/inputs/localization_plan.json",
            upload_as_file=True,
        )
        if not str(url):
            raise RuntimeError("[plan] plan upload returned no path — refusing silent loss")
        logger.info("[plan] uploaded plan to %s", url)
    return 0


# ── gen worker ───────────────────────────────────────────────────────────────


def _stage_adapter(slug: str, step: int, revision: str, root: Path) -> Path:
    from explore_persona_space.orchestrate import hub

    pir = f"{ADAPTER_PREFIX}/{slug}/checkpoint-{step}"
    dest = root / "adapters" / slug
    ckpt = dest / pir  # stage_hub_prefix lands a verbatim prefix MIRROR under dest
    if not (ckpt / "adapter_config.json").exists():
        hub.stage_hub_prefix(OVERFLOW_MODEL_REPO, pir, dest, repo_type="model", revision=revision)
    if not (ckpt / "adapter_config.json").exists():
        raise RuntimeError(f"[gen] staged adapter missing adapter_config.json at {ckpt}")
    return ckpt


def _load_plan(out_root: Path) -> dict:
    """Plan JSON: repo copy when present (VM), else the staged/HF copy (rsync lanes)."""
    local = OUT_DIR / "localization_plan.json"
    if local.exists():
        return _read_json(local)
    staged = out_root / "localization_plan.json"
    if not staged.exists():
        from explore_persona_space.orchestrate import hub

        hub.stage_hub_file(
            DATA_REPO,
            f"{LOC_PREFIX}/inputs/localization_plan.json",
            staged,
            repo_type="dataset",
        )
    return _read_json(staged)


def phase_gen(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    plan = _load_plan(out_root)

    # Shared-node engine sizing (fellows H200s are GPU-shared, #1902): compute
    # gpu_memory_utilization from LIVE free memory on the CVD-pinned device.
    from explore_persona_space.eval.vllm_util import resolve_vllm_util

    os.environ.setdefault("EPM_VLLM_GPU_MEM_UTIL", f"{resolve_vllm_util():.3f}")

    import issue1090_fu3_worker as fu3w
    import issue1090_run as run1090

    from explore_persona_space.artifacts.organisms import (
        DEFAULT_BASE_MODEL,
        _default_vllm_generate_fn,
        _generate_and_persist,
    )

    units = list(plan["units"])
    if args.only_units:
        keep = {t.strip() for t in args.only_units.split(",") if t.strip()}
        units = [u for u in units if u["slug"] in keep]
        if not units:
            raise SystemExit(f"[gen] --only-units matched nothing: {args.only_units}")
    mine = [u for i, u in enumerate(units) if i % args.n_workers == args.worker_idx]
    logger.info(
        "[gen] worker %d/%d: %d units (CVD=%s)",
        args.worker_idx,
        args.n_workers,
        len(mine),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    qs_by_behavior: dict[str, list[str]] = {}
    for behavior in BEHAVIOR_BY_KEY.values():
        qs = list(run1090.BEHAVIORS[behavior].eval_question_bank)
        sha = _sha256_text(json.dumps(qs, ensure_ascii=False))
        if sha != plan["questions_sha256"][behavior]:
            raise SystemExit(f"[gen] question bank drift for {behavior}")
        qs_by_behavior[behavior] = qs

    gen = _default_vllm_generate_fn(DEFAULT_BASE_MODEL, max_lora_rank=64)
    t_start = time.time()
    try:
        for k, unit in enumerate(mine):
            behavior = unit["behavior"]
            panel = fu3w.bystander_panel(behavior)
            if [c.context_id for c in panel] != plan["read_panel"][behavior]:
                raise SystemExit(f"[gen] read panel drift for {behavior}")
            qs = qs_by_behavior[behavior]
            t_unit = time.time()
            if unit["kind"] == "arm":
                ckpt = _stage_adapter(
                    unit["slug"], unit["step"], plan["adapter_revision"], out_root
                )
                side, side_path = "trained", str(ckpt)
                cell_dir = out_root / "panel" / unit["slug"]
            else:
                side, side_path = "base", None
                cell_dir = out_root / "base_panel" / unit["beh_key"]
            for bctx in panel:
                t_ctx = time.time()
                _generate_and_persist(
                    gen,
                    side,
                    side_path,
                    bctx,
                    qs,
                    n=N_PER_QUESTION,
                    temperature=1.0,
                    out_dir=cell_dir,
                    base_model=DEFAULT_BASE_MODEL,
                )
                print(
                    f"[gen] unit {k + 1}/{len(mine)} {unit['slug']} ctx={bctx.context_id} "
                    f"elapsed={time.time() - t_ctx:.1f}s",
                    flush=True,
                )
            print(
                f"[gen] unit-done {k + 1}/{len(mine)} {unit['slug']} "
                f"unit_wall={time.time() - t_unit:.1f}s total={time.time() - t_start:.1f}s",
                flush=True,
            )
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()
    print(f"[gen] worker {args.worker_idx} done ({len(mine)} units)", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


# ── dispatch (compute node) ──────────────────────────────────────────────────


def _gpu_ids() -> tuple[str, list[str]]:
    import issue1902_common as i1902c

    detected = 0
    proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        detected = len([ln for ln in proc.stdout.splitlines() if ln.strip().startswith("GPU")])
    return i1902c.realized_gpu_ids(os.environ, detected)


def _expected_completion_paths(plan: dict) -> list[str]:
    paths: list[str] = []
    for unit in plan["units"]:
        for ctx_id in plan["read_panel"][unit["behavior"]]:
            if unit["kind"] == "arm":
                rel = f"panel/{unit['slug']}/completions__trained__{ctx_id}.json"
            else:
                rel = f"base_panel/{unit['beh_key']}/completions__base__{ctx_id}.json"
            paths.append(f"{LOC_PREFIX}/raw_completions/{rel}")
    return paths


def _run_worker(
    idx: int,
    n_workers: int,
    gpu: str,
    out_root: Path,
    log_path: Path,
    only_units: str | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        "gen",
        "--worker-idx",
        str(idx),
        "--n-workers",
        str(n_workers),
        "--out-root",
        str(out_root),
    ]
    if only_units:
        cmd += ["--only-units", only_units]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "UV_NO_SYNC": "1"}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "a")  # noqa: SIM115 — handle lives with the Popen
    return subprocess.Popen(
        cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True
    )


def phase_dispatch(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    plan = _load_plan(out_root)

    src, ids = _gpu_ids()
    print(f"[dispatch] gpu ids {ids} (source={src})", flush=True)
    if not ids:
        raise SystemExit("[dispatch] no GPUs resolved")

    # Pre-stage shared inputs ONCE in the parent (fan-out shared-staging rule):
    # the base model snapshot; per-arm adapters are worker-disjoint by sharding.
    from explore_persona_space.artifacts.organisms import DEFAULT_BASE_MODEL
    from explore_persona_space.orchestrate import hub

    hub.retry_transient(
        lambda: __import__("huggingface_hub").snapshot_download(DEFAULT_BASE_MODEL),
        what="base model snapshot",
    )
    print("[dispatch] base model staged", flush=True)

    log_dir = out_root / "logs"
    pilot_slug = plan["units"][0]["slug"]
    pilot_json = out_root / "pilot.json"
    if not pilot_json.exists():
        t0 = time.time()
        proc = _run_worker(0, 1, ids[0], out_root, log_dir / "pilot.log", only_units=pilot_slug)
        rc = proc.wait()
        pilot_wall = time.time() - t0
        if rc != 0:
            tail = (log_dir / "pilot.log").read_text().splitlines()[-120:]
            print("\n".join(tail), flush=True)
            raise SystemExit(f"[dispatch] pilot unit failed rc={rc}")
        n_units = len(plan["units"])
        n_w = len(ids)
        projected = pilot_wall * ((n_units + n_w - 1) // n_w)
        _atomic_write_json(
            pilot_json,
            {
                "pilot_slug": pilot_slug,
                "pilot_wall_s": pilot_wall,
                "n_units": n_units,
                "n_workers": n_w,
                "projected_fleet_wall_s": projected,
                "fence_s": 2 * projected,
                "ts": _ts(),
            },
        )
        print(
            f"[dispatch] pilot {pilot_slug} wall={pilot_wall:.0f}s -> projected fleet "
            f"{projected:.0f}s (fence {2 * projected:.0f}s)",
            flush=True,
        )

    print(f"[phase=gen] fanning out {len(ids)} workers", flush=True)
    procs = [
        _run_worker(k, len(ids), gpu, out_root, log_dir / f"worker{k}.log")
        for k, gpu in enumerate(ids)
    ]
    failures: list[int] = []
    for k, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            failures.append(k)
            log = log_dir / f"worker{k}.log"
            tail = log.read_text().splitlines()[-120:] if log.exists() else []
            print(f"[dispatch] worker {k} FAILED rc={rc}; log tail:", flush=True)
            print("\n".join(tail), flush=True)
    if failures:
        raise SystemExit(f"[dispatch] {len(failures)} worker(s) failed: {failures}")

    print("[phase=upload] bulk upload", flush=True)
    for sub, dest in (("panel", "panel"), ("base_panel", "base_panel"), ("logs", "logs")):
        url = hub._upload(
            out_root / sub, DATA_REPO, "dataset", f"{LOC_PREFIX}/raw_completions/{dest}"
        )
        if not str(url):
            raise RuntimeError(f"[dispatch] upload of {sub} returned no path")
    url = hub._upload(
        pilot_json,
        DATA_REPO,
        "dataset",
        f"{LOC_PREFIX}/raw_completions/pilot.json",
        upload_as_file=True,
    )
    if not str(url):
        raise RuntimeError("[dispatch] pilot.json upload returned no path")

    from huggingface_hub import HfApi

    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        DATA_REPO,
        _expected_completion_paths(plan),
        path_in_repo=f"{LOC_PREFIX}/raw_completions",
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"[dispatch] {len(missing)} completion files missing: {missing[:5]}")

    done = out_root / "_DONE.json"
    _atomic_write_json(
        done,
        {
            "issue": ISSUE,
            "round": "localization-panel",
            "n_units": len(plan["units"]),
            "n_files": len(_expected_completion_paths(plan)),
            "git_commit": _git_sha(),
            "ts": _ts(),
        },
    )
    url = hub._upload(
        done,
        DATA_REPO,
        "dataset",
        f"{LOC_PREFIX}/raw_completions/_DONE.json",
        upload_as_file=True,
    )
    if not str(url):
        raise RuntimeError("[dispatch] _DONE upload returned no path")
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


# ── judge (VM) ───────────────────────────────────────────────────────────────


def _stage_completions(stage_root: Path) -> Path:
    """Stage the panel + base completions from HF (verbatim prefix mirror)."""
    from explore_persona_space.orchestrate import hub

    marker = stage_root / "_staged_ok"
    mirror = stage_root / LOC_PREFIX / "raw_completions"
    if not marker.exists():
        for sub in ("panel", "base_panel"):
            hub.stage_hub_prefix(
                DATA_REPO,
                f"{LOC_PREFIX}/raw_completions/{sub}",
                stage_root,
                repo_type="dataset",
            )
        marker.write_text(_ts())
    return mirror


def _completions_for(path: Path, questions_sha: str) -> list[list[str]]:
    payload = _read_json(path)
    got = (payload.get("manifest") or {}).get("questions_sha256")
    if got != questions_sha:
        raise RuntimeError(f"[judge] questions sha mismatch at {path}: {got}")
    return payload["completions"]


def _judge_fn_for(beh_key: str, stub: bool):
    if stub:
        import issue1481_analysis as a1481

        return a1481._stub_judge_fn, "stub-smoke"
    if beh_key == "cas":
        import issue1434_cells as c1434

        return c1434.pv_judge_fn, "pv_trait_score"
    import issue1090_fu3_worker as fu3w

    return fu3w.judge_graded_r23, "registered_graded_r23"


def _cell_items(tag: str, qs: list[str], comps: list[list[str]]) -> list[tuple[str, str, str]]:
    return [
        (f"{tag}-q{qi:03d}-c{ci:03d}", q, comp)
        for qi, q in enumerate(qs)
        for ci, comp in enumerate(comps[qi])
    ]


def _reduce_cell(result: Any, item_ids: list[str], n_draws: int) -> dict:
    """Per-cell record from a POOLED JudgeResult — the #1481 _judge_cell tally
    semantics (drop-never-coerce; content vs transport split; rule 9/24)."""
    import issue1481_cells as c1481

    scores = [result.scores.get(iid) for iid in item_ids]
    scored = [s for s in scores if s is not None]
    n_pos = sum(1 for s in scored if s > THRESHOLD)
    per_item_transport = getattr(result, "per_item_transport_losses", {}) or {}
    transport = sum(per_item_transport.get(iid, 0) for iid in item_ids)
    per_item_scores = getattr(result, "per_item_scores", {}) or {}
    if per_item_scores:
        kept = sum(len(per_item_scores.get(iid, [])) for iid in item_ids)
    else:  # stub judge (#1481 _stub_judge_fn) carries aggregate telemetry only
        kept = len(scored) * n_draws
    content_dropped = len(item_ids) * n_draws - kept - transport
    return {
        "n_items": len(item_ids),
        "n_scored": len(scored),
        "k_positive": n_pos,
        "rate": (n_pos / len(scored)) if scored else None,
        "graded_mean": (sum(scored) / len(scored)) if scored else None,
        "wilson_95": list(c1481.wilson(n_pos, len(scored))) if scored else None,
        "item_drop_frac": 1.0 - (len(scored) / len(item_ids)) if item_ids else 0.0,
        "n_dropped_draws_content": content_dropped,
        "n_transport_lost_draws": transport,
    }


def _pooled_judge(
    beh_key: str,
    cells: dict[str, tuple[list[str], list[list[str]]]],
    *,
    judge_root: Path,
    wave: str,
    stub: bool,
) -> dict[str, dict]:
    """One pooled judge call over ``cells`` (tag -> (qs, comps)); per-cell reduce."""
    import issue1090_run as run1090

    behavior = BEHAVIOR_BY_KEY[beh_key]
    behavior_obj = run1090.BEHAVIORS[behavior]
    judge_fn, instrument = _judge_fn_for(beh_key, stub)
    items: list[tuple[str, str, str]] = []
    ids_by_tag: dict[str, list[str]] = {}
    for tag, (qs, comps) in sorted(cells.items()):
        cell_items = _cell_items(tag, qs, comps)
        ids_by_tag[tag] = [iid for iid, _, _ in cell_items]
        items.extend(cell_items)
    wave_root = judge_root / beh_key / wave
    wave_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[judge] %s wave=%s: %d cells, %d items x %d draws (instrument=%s)",
        beh_key,
        wave,
        len(cells),
        len(items),
        N_JUDGE_DRAWS,
        instrument,
    )
    result = judge_fn(
        items,
        behavior_obj.judge_rubric,
        n_draws=N_JUDGE_DRAWS,
        cache_dir=wave_root / "cache",
        save_raw=wave_root / f"judge_raw_{instrument}_{wave}.json",
        judge_model=behavior_obj.judge_model,
    )
    records = {tag: _reduce_cell(result, ids, N_JUDGE_DRAWS) for tag, ids in ids_by_tag.items()}
    telemetry = {
        "instrument": instrument,
        "n_total_draws": getattr(result, "n_total_draws", None),
        "n_dropped_draws_content": getattr(result, "n_dropped_draws", None),
        "n_transport_lost_draws": getattr(result, "n_transport_lost_draws", None),
        "n_truncation_dropped_draws": getattr(result, "n_truncation_dropped_draws", None),
        "n_refusal_draws": getattr(result, "n_refusal_draws", None),
        "stop_reason_tally": getattr(result, "stop_reason_tally", None),
    }
    _atomic_write_json(wave_root / "wave_telemetry.json", telemetry)
    return records


def _pilot_gate(beh_key: str, cells: dict, judge_root: Path) -> dict:
    import issue1090_fu3_worker as fu3w
    import issue1090_run as run1090

    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    behavior = BEHAVIOR_BY_KEY[beh_key]
    behavior_obj = run1090.BEHAVIORS[behavior]
    if beh_key == "cas":
        import issue1434_cells as c1434

        rubric = c1434.pv_rubric_text()
    else:
        rubric = behavior_obj.judge_rubric
    arms: dict[str, list[tuple[str, str, str]]] = {}
    for tag, (qs, comps) in sorted(cells.items()):
        code = tag.rsplit("-", 1)[-1]
        arms.setdefault(code, []).extend(_cell_items(f"pl-{tag}", qs, comps))
    root = judge_root / beh_key / "pilot"
    report = judge_pilot_gate(
        arms,
        rubric,
        max_tokens=fu3w.JUDGE_MAX_TOKENS,
        cache_dir=root / "cache",
        save_raw_dir=root,
        n_draws=N_JUDGE_DRAWS,
        target_total_draws=PILOT_TARGET_DRAWS,
        judge_model=behavior_obj.judge_model,
        report_path=OUT_DIR / f"judge_pilot_{beh_key}.json",
    )
    logger.info("[judge] pilot %s: %s (%s)", beh_key, report.verdict, report.failures)
    if not report.passed:
        raise SystemExit(f"[judge] rule-26 pilot gate FAILED for {beh_key}: {report.failures}")
    return report.to_json()


def phase_judge(args: argparse.Namespace) -> int:
    import issue1090_run as run1090

    plan = _read_json(OUT_DIR / "localization_plan.json")
    if args.stub_judge and not args.smoke:
        raise SystemExit("[judge] --stub-judge is smoke-only: pass --smoke with it")
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    if args.completions_root:
        mirror = Path(args.completions_root)
    else:
        stage_root = REPO_ROOT / "data" / "issue_1947" / "localization_stage"
        mirror = _stage_completions(stage_root)
    judge_root = (
        Path(args.judge_root)
        if args.judge_root
        else REPO_ROOT / "data" / "issue_1947" / "localization_judge"
    )

    arm_meta = plan["arm_meta"]
    beh_keys = sorted(t.strip() for t in (args.behaviors or "cas,imp,syc").split(",") if t.strip())
    if set(beh_keys) - set(BEHAVIOR_BY_KEY):
        raise SystemExit(f"[judge] unknown --behaviors {beh_keys}")
    aggregates: dict[str, dict] = {}
    for beh_key in beh_keys:
        behavior = BEHAVIOR_BY_KEY[beh_key]
        qs = list(run1090.BEHAVIORS[behavior].eval_question_bank)
        sha = plan["questions_sha256"][behavior]
        read_ctxs = plan["read_panel"][behavior]
        arm_slugs = sorted(s for s, m in arm_meta.items() if m["beh_key"] == beh_key)
        if args.smoke:
            arm_slugs = arm_slugs[: args.smoke_arms]

        def _load(rel: str) -> list[list[str]]:
            return _completions_for(mirror / rel, sha)

        # Source-context cells (the wiring gate wave) vs the rest.
        src_cells: dict[str, tuple[list[str], list[list[str]]]] = {}
        rest_cells: dict[str, tuple[list[str], list[list[str]]]] = {}
        import issue1481_cells as c1481

        for slug in arm_slugs:
            own_id = c1481.context_id_for(behavior, arm_meta[slug]["ctx_key"])
            for ctx_id in read_ctxs:
                tag = f"pn-{slug}-{_ctx_code(ctx_id)}"
                comps = _load(f"panel/{slug}/completions__trained__{ctx_id}.json")
                (src_cells if ctx_id == own_id else rest_cells)[tag] = (qs, comps)
        for ctx_id in read_ctxs:
            tag = f"bs-{beh_key}-{_ctx_code(ctx_id)}"
            rest_cells[tag] = (qs, _load(f"base_panel/{beh_key}/completions__base__{ctx_id}.json"))

        # Rule-26 pilot gate (production instrument, fresh pilot cache).
        if not args.stub_judge and not args.skip_pilot:
            pilot_cells = {t: src_cells[t] for t in sorted(src_cells)}
            _pilot_gate(beh_key, pilot_cells, judge_root)

        src_records = _pooled_judge(
            beh_key, src_cells, judge_root=judge_root, wave="source", stub=args.stub_judge
        )

        # WIRING GATE: source-context rate must reproduce the committed verdict rate.
        wiring = []
        for slug in arm_slugs:
            own_id = c1481.context_id_for(behavior, arm_meta[slug]["ctx_key"])
            rec = src_records[f"pn-{slug}-{_ctx_code(own_id)}"]
            committed = float(plan["selection"][slug]["rate"])
            got = rec["rate"]
            delta = None if got is None else abs(got - committed)
            wiring.append(
                {
                    "slug": slug,
                    "committed_rate": committed,
                    "panel_rate": got,
                    "abs_delta": delta,
                    "parity_ok": delta is not None and delta <= WIRING_TOL,
                }
            )
        _atomic_write_json(
            out_dir / f"wiring_gate_{beh_key}.json",
            {"behavior": behavior, "tolerance": WIRING_TOL, "arms": wiring, "ts": _ts()},
        )
        misses = [w for w in wiring if not w["parity_ok"]]
        logger.info("[judge] wiring gate %s: %d/%d misses", beh_key, len(misses), len(wiring))
        if not args.stub_judge and len(misses) > WIRING_MAX_MISSES:
            raise SystemExit(
                f"[judge] WIRING GATE FAILED for {beh_key}: {len(misses)}/{len(wiring)} arms "
                f"off the committed verdict rate by > {WIRING_TOL} — rig fault, halting "
                f"before the full wave: {[m['slug'] for m in misses]}"
            )

        rest_records = _pooled_judge(
            beh_key, rest_cells, judge_root=judge_root, wave="rest", stub=args.stub_judge
        )
        records = {**src_records, **rest_records}

        agg: dict[str, Any] = {
            "issue": ISSUE,
            "behavior": behavior,
            "beh_key": beh_key,
            "instrument": (
                "stub-smoke"
                if args.stub_judge
                else ("pv_trait_score" if beh_key == "cas" else "registered_graded_r23")
            ),
            "smoke_stub_judge": bool(args.stub_judge),
            "n_draws": N_JUDGE_DRAWS,
            "questions_sha256": sha,
            "base_panel": {},
            "arms": {},
            "ts": _ts(),
        }
        for ctx_id in read_ctxs:
            agg["base_panel"][ctx_id] = records[f"bs-{beh_key}-{_ctx_code(ctx_id)}"]
        for slug in arm_slugs:
            meta = arm_meta[slug]
            entry = {
                "train_ctx_key": meta["ctx_key"],
                "train_ctx_id": c1481.context_id_for(behavior, meta["ctx_key"]),
                "regime": meta["regime"],
                "visit": meta["visit"],
                "seed": meta["seed"],
                "labels": plan["labels"][slug],
                "contexts": {
                    ctx_id: records[f"pn-{slug}-{_ctx_code(ctx_id)}"] for ctx_id in read_ctxs
                },
            }
            agg["arms"][slug] = entry
        out = out_dir / f"panel_aggregate_{beh_key}.json"
        _atomic_write_json(out, agg)
        aggregates[beh_key] = agg
        logger.info("[judge] wrote %s (%d arms)", out, len(agg["arms"]))
    return 0


# ── analyze (VM) ─────────────────────────────────────────────────────────────


def _pooled(agg_arm: dict, ctx_ids: list[str]) -> tuple[int, int]:
    k = n = 0
    for ctx_id in ctx_ids:
        rec = agg_arm["contexts"][ctx_id]
        if rec.get("rate") is None:
            continue
        k += int(rec["k_positive"])
        n += int(rec["n_scored"])
    return k, n


def _pooled_base(agg: dict, ctx_ids: list[str]) -> tuple[int, int]:
    k = n = 0
    for ctx_id in ctx_ids:
        rec = agg["base_panel"].get(ctx_id)
        if rec is None or rec.get("rate") is None:
            continue
        k += int(rec["k_positive"])
        n += int(rec["n_scored"])
    return k, n


def _rate_block(k: int, n: int) -> dict:
    import issue1481_cells as c1481

    return {
        "k": k,
        "n": n,
        "rate": (k / n) if n else None,
        "wilson_95": list(c1481.wilson(k, n)) if n else None,
    }


def phase_analyze(args: argparse.Namespace) -> int:
    import issue1481_cells as c1481

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    plan = _read_json(OUT_DIR / "localization_plan.json")
    arm_meta = plan["arm_meta"]
    summary: dict[str, Any] = {
        "issue": ISSUE,
        "round": "localization-panel",
        "git_commit": _git_sha(),
        "ts": _ts(),
        "n_per_cell": 20 * N_PER_QUESTION,
        "wiring_tolerance": WIRING_TOL,
        "per_arm": {},
        "regime_contrast": {},
        "behavior_rollup": {},
    }
    for beh_key, behavior in sorted(BEHAVIOR_BY_KEY.items()):
        agg_path = out_dir / f"panel_aggregate_{beh_key}.json"
        if not agg_path.exists():
            raise SystemExit(f"[analyze] missing {agg_path}")
        agg = _read_json(agg_path)
        read_ctxs = plan["read_panel"][behavior]
        for slug, arm in sorted(agg["arms"].items()):
            labels = arm["labels"]
            own = arm["train_ctx_id"]
            nonsrc = [c for c in read_ctxs if c != own]
            heldout = [c for c in read_ctxs if labels[c] == "held_out"]
            trained_neg = [c for c in read_ctxs if labels[c] == "trained_negative"]
            per = {
                "behavior": behavior,
                "beh_key": beh_key,
                "ctx_key": arm["train_ctx_key"],
                "regime": arm["regime"],
                "visit": arm["visit"],
                "seed": arm["seed"],
                "selection": plan["selection"][slug],
                "source": _rate_block(*_pooled(arm, [own])),
                "source_graded_mean": arm["contexts"][own].get("graded_mean"),
                "pooled_non_source": _rate_block(*_pooled(arm, nonsrc)),
                "held_out_only": _rate_block(*_pooled(arm, heldout)),
                "trained_negative_only": (
                    _rate_block(*_pooled(arm, trained_neg)) if trained_neg else None
                ),
                "base_pooled_non_source": _rate_block(*_pooled_base(agg, nonsrc)),
                "base_held_out_only": _rate_block(*_pooled_base(agg, heldout)),
                "base_source": _rate_block(*_pooled_base(agg, [own])),
                "n_held_out_contexts": len(heldout),
            }
            summary["per_arm"][slug] = per

        # Regime contrast per (ctx_key, seed, visit) pair: D = p_po - p_con.
        pairs = {}
        arms_here = {s: m for s, m in arm_meta.items() if m["beh_key"] == beh_key}
        for slug, meta in sorted(arms_here.items()):
            if meta["regime"] != "con":
                continue
            po_slug = slug.replace("-con-", "-po-")
            if po_slug not in summary["per_arm"] or slug not in summary["per_arm"]:
                continue
            con, po = summary["per_arm"][slug], summary["per_arm"][po_slug]
            entry: dict[str, Any] = {"con_slug": slug, "po_slug": po_slug}
            for scope in ("pooled_non_source", "held_out_only"):
                k_po, n_po = po[scope]["k"], po[scope]["n"]
                k_con, n_con = con[scope]["k"], con[scope]["n"]
                if n_po and n_con:
                    d = k_po / n_po - k_con / n_con
                    ci = c1481.newcombe(k_po, n_po, k_con, n_con)
                    entry[scope] = {
                        "D": d,
                        "newcombe_95": list(ci),
                        "lattice": c1481.lattice_verdict(d, ci),
                        "po": po[scope],
                        "con": con[scope],
                    }
                else:
                    entry[scope] = {"D": None, "status": "not_computable"}
            entry["dose_match"] = c1481.dose_match_label(
                plan["selection"][slug], plan["selection"][po_slug]
            )
            pairs[f"{beh_key}-{meta['ctx_key']}-{meta['visit']}-s{meta['seed']}"] = entry
        summary["regime_contrast"][beh_key] = pairs

        # Behavior rollups: pooled over arms per regime x scope, + base.
        rollup: dict[str, Any] = {}
        for regime in ("con", "po"):
            slugs = [
                s for s, m in arms_here.items() if m["regime"] == regime and s in summary["per_arm"]
            ]
            for scope in ("source", "pooled_non_source", "held_out_only", "trained_negative_only"):
                k = n = 0
                for s in slugs:
                    blk = summary["per_arm"][s][scope]
                    if blk is None or blk["n"] == 0:
                        continue
                    k += blk["k"]
                    n += blk["n"]
                rollup[f"{regime}.{scope}"] = _rate_block(k, n)
            rollup[f"{regime}.n_arms"] = len(slugs)
        k_b, n_b = _pooled_base(agg, read_ctxs)
        rollup["base.all_contexts"] = _rate_block(k_b, n_b)
        summary["behavior_rollup"][beh_key] = rollup

    _atomic_write_json(out_dir / "summary.json", summary)
    logger.info("[analyze] wrote %s", out_dir / "summary.json")
    return 0


# ── figures (VM) ─────────────────────────────────────────────────────────────

READ_CLASS_ROLE = {
    "source": "primary",
    "trained_negative": "control",
    "held_out": "accent",
    "base": "neutral",
}
BEH_LABEL = {"cas": "casual style", "imp": "impoliteness", "syc": "sycophancy"}
REGIME_LABEL = {"con": "contrastive", "po": "positive-only"}


def phase_figures(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    fig_dir = Path(args.fig_dir) if args.fig_dir else FIG_DIR
    summary = _read_json(out_dir / "summary.json")
    set_paper_style()

    colors = {cls: paper_palette_role(role) for cls, role in READ_CLASS_ROLE.items()}

    # Figure 1 — aggregate: per behavior x regime, source vs held-out-only vs
    # trained-negative pooled rates, with the base-panel reference.
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, beh_key in zip(axes, sorted(BEHAVIOR_BY_KEY)):
        roll = summary["behavior_rollup"][beh_key]
        xticks, xlabels = [], []
        x = 0.0
        for regime in ("con", "po"):
            scopes = [
                ("source", "source"),
                ("trained_negative_only", "trained_negative"),
                ("held_out_only", "held_out"),
            ]
            for scope, cls in scopes:
                blk = roll.get(f"{regime}.{scope}")
                if not blk or blk["n"] == 0:
                    x += 1.0
                    continue
                lo, hi = blk["wilson_95"]
                v = blk["rate"]
                ax.errorbar(
                    [x],
                    [v],
                    yerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
                    fmt="o",
                    color=colors[cls],
                    capsize=3,
                    markersize=6,
                )
                x += 1.0
            xticks.append(x - 2.0)
            xlabels.append(REGIME_LABEL[regime])
            x += 0.8
        base = roll["base.all_contexts"]
        if base["n"]:
            ax.axhline(base["rate"], color=colors["base"], linestyle="--", linewidth=1.2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_title(BEH_LABEL[beh_key])
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("judged positive rate")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[c], label=lbl)
        for c, lbl in (
            ("source", "own trained context"),
            ("trained_negative", "trained-negative context"),
            ("held_out", "held-out context"),
        )
    ] + [plt.Line2D([], [], linestyle="--", color=colors["base"], label="base model (panel)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    savefig_paper(fig, "localization_overview", dir=fig_dir)
    plt.close(fig)

    # Figure 2 — per-arm low-level view: one row per arm, one point per read
    # context, colored by the three-way label; base pooled non-source as a tick.
    per_arm = summary["per_arm"]
    order = sorted(
        per_arm,
        key=lambda s: (
            per_arm[s]["beh_key"],
            per_arm[s]["regime"],
            per_arm[s]["ctx_key"],
            per_arm[s]["visit"],
            per_arm[s]["seed"],
        ),
    )
    fig2, ax2 = plt.subplots(figsize=(8.2, 0.26 * len(order) + 1.8))
    ylabels = []
    for yi, slug in enumerate(order):
        arm = per_arm[slug]
        agg = _read_json(out_dir / f"panel_aggregate_{arm['beh_key']}.json")
        entry = agg["arms"][slug]
        for ctx_id, rec in entry["contexts"].items():
            if rec.get("rate") is None:
                continue
            cls = "source" if entry["labels"][ctx_id] == "own" else entry["labels"][ctx_id]
            ax2.scatter(
                [rec["rate"]],
                [yi],
                s=22,
                color=colors["source" if cls == "own" else cls],
                zorder=3,
            )
        base_blk = arm["base_pooled_non_source"]
        if base_blk["n"]:
            ax2.scatter([base_blk["rate"]], [yi], s=30, marker="|", color=colors["base"], zorder=2)
        ylabels.append(
            f"{BEH_LABEL[arm['beh_key']]} · {arm['ctx_key']} · "
            f"{REGIME_LABEL[arm['regime']]} · {arm['visit']} · s{arm['seed']}"
        )
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(ylabels, fontsize=5.5)
    ax2.set_xlabel("judged positive rate (per read context)")
    ax2.set_xlim(-0.02, 1.02)
    ax2.invert_yaxis()
    handles2 = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[c], label=lbl)
        for c, lbl in (
            ("source", "own trained context"),
            ("trained_negative", "trained-negative context"),
            ("held_out", "held-out context"),
        )
    ] + [
        plt.Line2D(
            [], [], marker="|", linestyle="", color=colors["base"], label="base (pooled non-source)"
        )
    ]
    ax2.legend(handles=handles2, loc="lower right", fontsize=6)
    fig2.tight_layout()
    savefig_paper(fig2, "per_arm_context_rates", dir=fig_dir)
    plt.close(fig2)
    logger.info("[figures] wrote %s", fig_dir)
    return 0


# ── stub e2e smoke (VM) ──────────────────────────────────────────────────────


def phase_stub_e2e(args: argparse.Namespace) -> int:
    """Tiny fixture completions in the production on-disk shape -> judge
    (--stub-judge) -> analyze -> figures, all under a scratch dir."""
    import shutil
    import tempfile

    import issue1090_run as run1090
    import issue1481_cells as c1481

    scratch = Path(tempfile.mkdtemp(prefix="i1947loc_smoke_"))
    try:
        plan = _read_json(OUT_DIR / "localization_plan.json")
        mirror = scratch / "mirror"
        smoke_arms: list[str] = []
        for beh_key, behavior in sorted(BEHAVIOR_BY_KEY.items()):
            qs = list(run1090.BEHAVIORS[behavior].eval_question_bank)
            sha = plan["questions_sha256"][behavior]
            slugs = sorted(s for s, m in plan["arm_meta"].items() if m["beh_key"] == beh_key)[
                : args.smoke_arms
            ]
            smoke_arms.extend(slugs)
            for slug in slugs + [None]:
                for ctx_id in plan["read_panel"][behavior]:
                    if slug is None:
                        rel = f"base_panel/{beh_key}/completions__base__{ctx_id}.json"
                    else:
                        rel = f"panel/{slug}/completions__trained__{ctx_id}.json"
                    path = mirror / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    comps = [
                        [f"fixture answer {qi}-{ci}" for ci in range(2)] for qi in range(len(qs))
                    ]
                    path.write_text(
                        json.dumps(
                            {
                                "manifest": {"questions_sha256": sha},
                                "questions": qs,
                                "completions": comps,
                            }
                        )
                    )
        ns = argparse.Namespace(
            stub_judge=True,
            smoke=True,
            smoke_arms=args.smoke_arms,
            completions_root=str(mirror),
            judge_root=str(scratch / "judge"),
            out_dir=str(scratch / "out"),
            behaviors=None,
            skip_pilot=True,
        )
        rc = phase_judge(ns)
        if rc != 0:
            return rc
        rc = phase_analyze(argparse.Namespace(out_dir=str(scratch / "out")))
        if rc != 0:
            return rc
        rc = phase_figures(
            argparse.Namespace(out_dir=str(scratch / "out"), fig_dir=str(scratch / "figs"))
        )
        if rc != 0:
            return rc
        summary = _read_json(scratch / "out" / "summary.json")
        n_arms = len(summary["per_arm"])
        n_figs = len(list((scratch / "figs").glob("*.png")))
        # smoke slice floors: >= 2 arms per behavior (one per regime where
        # available) so the contrast path executes; figures rendered.
        assert n_arms == len(smoke_arms), (n_arms, smoke_arms)
        assert n_figs == 2, n_figs
        assert summary["regime_contrast"], "no regime-contrast pairs in smoke"
        del c1481
        print(f"[stub-e2e] OK: {n_arms} arms, {n_figs} figures, scratch={scratch}")
        return 0
    finally:
        if not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="#1947 six-context localization panel")
    p.add_argument(
        "--phase",
        required=True,
        choices=(
            "plan",
            "dispatch",
            "gen",
            "judge",
            "analyze",
            "figures",
            "stub-e2e",
            "import-check",
        ),
    )
    p.add_argument("--upload", action="store_true", help="plan: upload plan JSON to HF")
    p.add_argument("--adapter-revision", default=None, help="plan: pin instead of resolving")
    p.add_argument(
        "--out-root",
        default="/workspace/issue1947_localization",
        help="dispatch/gen: node-local output root",
    )
    p.add_argument("--worker-idx", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--only-units", default=None, help="gen: comma slug subset (pilot)")
    p.add_argument("--stub-judge", action="store_true", help="judge: offline stub (smoke)")
    p.add_argument("--behaviors", default=None, help="judge: comma beh_key subset (cas,imp,syc)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-arms", type=int, default=2)
    p.add_argument(
        "--skip-pilot",
        action="store_true",
        help="judge: skip the rule-26 pilot gate (resume after a PASS)",
    )
    p.add_argument("--completions-root", default=None, help="judge: local mirror override")
    p.add_argument("--judge-root", default=None, help="judge: cache/raw root override")
    p.add_argument("--out-dir", default=None, help="judge/analyze/figures: output override")
    p.add_argument("--fig-dir", default=None, help="figures: output override")
    p.add_argument("--keep-scratch", action="store_true", help="stub-e2e: keep scratch dir")
    return p


def _import_check() -> int:
    """Resolve every deferred import this script's phases hit (Axis-1 leg)."""
    import issue1090_fu3_worker as fu3w
    import issue1090_run as run1090
    import issue1481_analysis as a1481
    import issue1481_cells as c1481
    import issue1902_common as i1902c

    from explore_persona_space.artifacts import negatives as neg_mod
    from explore_persona_space.artifacts.organisms import (
        DEFAULT_BASE_MODEL,
        _default_vllm_generate_fn,
        _generate_and_persist,
    )
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate
    from explore_persona_space.eval.vllm_util import resolve_vllm_util
    from explore_persona_space.orchestrate import hub

    names = (
        fu3w.judge_graded_r23,
        fu3w.bystander_panel,
        run1090.BEHAVIORS,
        a1481._stub_judge_fn,
        a1481.registered_heldout,
        c1481.wilson,
        c1481.newcombe,
        c1481.lattice_verdict,
        c1481.dose_match_label,
        c1481.context_id_for,
        i1902c.realized_gpu_ids,
        neg_mod.default_panel,
        DEFAULT_BASE_MODEL,
        _default_vllm_generate_fn,
        _generate_and_persist,
        judge_pilot_gate,
        resolve_vllm_util,
        hub.stage_hub_prefix,
        hub.stage_hub_file,
        hub.verify_repo_paths_uploaded,
        hub._upload,
        hub.retry_transient,
    )
    import issue1434_cells as c1434

    _ = (c1434.pv_judge_fn, c1434.pv_rubric_text, names)
    print("[import-check] OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "import-check":
        return _import_check()
    if args.phase == "plan":
        return phase_plan(args)
    if args.phase == "dispatch":
        return phase_dispatch(args)
    if args.phase == "gen":
        return phase_gen(args)
    if args.phase == "judge":
        return phase_judge(args)
    if args.phase == "analyze":
        return phase_analyze(args)
    if args.phase == "figures":
        return phase_figures(args)
    if args.phase == "stub-e2e":
        return phase_stub_e2e(args)
    raise SystemExit(f"unknown phase {args.phase!r}")


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
