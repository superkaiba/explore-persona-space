#!/usr/bin/env python
"""#1481 — content-grid driver (plan §4.4/§4.7): thin composition over the
round-parametrized #1090 fu4 driver + the #1434 D1' po-mix derivation.

Phases:

VM (Phase 0, 0 GPU):
- ``--phase mixes``   D1': derive the 8 impolite/sycophancy positive-only
                      mixes from the PINNED con mixes each con arm trains on
                      (reuses ``issue1434_worker._derive_po_rows`` — filter →
                      sidecar rebuild → STOP), upload to
                      ``issue1481_conpos_grid/po_mixes/<cell>/mix/`` and pin
                      shas into ``cell_manifest_i1481_pomixes.json``.

Pod (GPU; ``bash scripts/issue1481_dispatch.sh`` sequences per dispatch group):
- ``--phase stage|dispatch|run|judge-aggregate``  DELEGATED verbatim to the
  fu4 driver after registering the six i1481 rounds; ``--phase run``
  REWRITES ``--seed`` from the run_id's ``-s<seed>`` suffix (seed threading —
  Fu4Run carries no seed field), so one dispatch fans out both seeds.
- ``--dispatch impolite|sycophancy|casual-s137``  sequences stage → dispatch
  for the group's con + po rounds, then the reused-arm re-reads (gate P1).
- ``--phase reread``  the 12 fu4/fu5/fu7 committed-selected-checkpoint
  apply-and-read jobs (plan §4.6 gate P1: re-read Tier-1 rate at the
  committed rung under THIS run's instrument; WARN + persisted values).
- ``--phase base-arms``  fresh per-context base Tier-2 + shared base panel
  gens for one behavior (Phase C gap fill where committed base reads are
  missing).

Pod-side code NEVER shells scripts/task.py; sentinels ride the fu4 contract
(``/workspace/logs/issue-1481-*.json``); ``[phase=done]`` is emitted by the
dispatch .sh wrapper ONLY.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1074_generator_compare as i1074  # noqa: E402
import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_fu4 as fu4  # noqa: E402
import issue1090_run as run1090  # noqa: E402
import issue1434_worker as w1434  # noqa: E402
import issue1481_cells as cells  # noqa: E402

from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    ModelOrganism,
    _default_vllm_generate_fn,
    _generate_and_persist,
    _read_jsonl,
    _write_jsonl,
    make_source_rate_fn,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1481.worker")

FU4_DELEGATED_PHASES = ("stage", "dispatch", "run", "judge-aggregate")
OWN_PHASES = ("mixes", "reread", "base-arms")

# Committed parent manifests carrying con-mix sha pins for the cells they
# consumed (plan §4.3 "shas from the fu3/fu4 records asserted at staging").
_PARENT_MANIFESTS: dict[str, str] = {
    "imp-pers": "eval_results/issue_1090/fu4-extended-dose-lr/cell_manifest_fu4.json",
    "imp-conv": "eval_results/issue_1090/fu4-extended-dose-lr/cell_manifest_fu4.json",
    "imp-bare": (
        "eval_results/issue_1090/finish-impolite-bare-and-formatting-rank/cell_manifest_fu5.json"
    ),
    "syc-pers": (
        "eval_results/issue_1090/sycophancy-lr-install-and-remeasure/cell_manifest_fu7.json"
    ),
}
_PARENT_MANIFEST_RUN: dict[str, str] = {
    "imp-pers": "imp-pers-lr1e5",
    "imp-conv": "imp-conv-lr1e5",
    "imp-bare": "imp-bare-lr1e5",
    "syc-pers": "syc-c3-lr1e5",
}

# The fu4 content-bundle adapter gauge (plan §4.6 gate P3; grounded on the
# reused arms' own adapter_config.json at Phase-0 probe time).
P3_EXPECTED = {"r": 32, "lora_alpha": 64, "use_rslora": True}


def worker_config(args: argparse.Namespace) -> run1090.RunConfig:
    """The #1481 RunConfig (the #1434 worker_config shape)."""
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else ("/tmp/issue-1481-smoke" if smoke else "data/issue_1481/cells")
    )
    return run1090.RunConfig(
        smoke=smoke,
        cells=(),
        out_root=out_root,
        seed=args.seed,
        target_n=(6 if smoke else run1090.TARGET_N),
        max_oversample_mult=fu3w.FU3_MAX_OVERSAMPLE_MULT,
        tier1_n=2 if smoke else run1090.TIER1_N_COMPLETIONS,
        tier1_draws=2 if smoke else run1090.TIER1_JUDGE_DRAWS,
        tier2_n=2 if smoke else run1090.TIER2_N_COMPLETIONS,
        tier2_draws=2 if smoke else run1090.TIER2_JUDGE_DRAWS,
        eval_question_limit=(
            args.eval_question_limit
            if args.eval_question_limit is not None
            else (2 if smoke else None)
        ),
        sentinel_dir=(
            Path(args.sentinel_dir)
            if args.sentinel_dir is not None
            else (out_root / "logs" if smoke else None)
        ),
        upload=args.upload,
    )


# ── Phase 0: D1' po-mix derivation (VM, 0 GPU) ───────────────────────────────


def _con_source_layout(beh_key: str, ctx_key: str) -> tuple[str, str, str]:
    """(hub prefix, mix subpath, meta subpath) of the PINNED con mix a po cell
    derives from — production c2/c3 keep a ``mix/`` subdir; fu3 cells are FLAT."""
    prefix, layout = cells.mix_for(beh_key, ctx_key, "con")
    if layout == "parent-mix-subdir":
        root = prefix.removesuffix("/mix")
        return root, "mix/train_mix.jsonl", "mix/mix_meta.json"
    assert layout == "fu3-flat", layout
    return prefix, "train_mix.jsonl", "mix_meta.json"


def _stage_con_source(cfg: run1090.RunConfig, beh_key: str, ctx_key: str) -> Path:
    """Stage one po cell's con-mix inputs (mix + D1' datagen sidecars)."""
    root_prefix, mix_sub, meta_sub = _con_source_layout(beh_key, ctx_key)
    dest = Path(cfg.out_root) / "po_inputs" / f"{beh_key}-{ctx_key}"
    for sub in (mix_sub, meta_sub, "datagen/cn.jsonl", "datagen/pos.jsonl"):
        hub.stage_hub_file(
            run1090.HF_DATA_REPO,
            f"{root_prefix}/{sub}",
            dest / sub,
            repo_type="dataset",
        )
    return dest


def phase_mixes(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """D1' (plan §4.3): derive the 8 impolite/sycophancy 60-row po mixes from
    the pinned con mixes via the #1434 row-provenance filter (same 20
    positives, same 40 generic, negatives dropped), upload + pin shas."""
    run1090._phase("i1481_po_mixes")
    generic_corpus: list[dict] | None = None

    def _generic() -> list[dict]:
        nonlocal generic_corpus
        if generic_corpus is None:
            dest = Path(cfg.out_root) / "po_inputs" / "generic_corpus.jsonl"
            hub.stage_hub_file(
                run1090.HF_DATA_REPO, i1074.GENERIC_CORPUS_HF_PATH, dest, repo_type="dataset"
            )
            generic_corpus = _read_jsonl(dest)
            if len(generic_corpus) < 40:
                raise RuntimeError(
                    f"staged generic corpus has {len(generic_corpus)} rows < 40 — "
                    "cannot rebuild a 40-generic po mix"
                )
        return generic_corpus

    from huggingface_hub import HfApi

    api = HfApi()
    upload = run1090._upload_fn(run1090.Seams1090())
    wanted = [f"{b}-{c}" for b in ("imp", "syc") for c in cells.CTX_KEYS]
    if args.cells:
        subset = [t.strip() for t in args.cells.split(",") if t.strip()]
        bad = [k for k in subset if k not in wanted]
        if bad:
            raise ValueError(f"bad po-mix cells {bad!r}: want a subset of {wanted}")
        wanted = subset
    elif cfg.smoke:
        wanted = ["imp-pers", "syc-bare"]  # both source layouts through the same path
    rows_out: list[dict] = []
    derivations: dict[str, dict] = {}
    for cell_key in wanted:
        beh_key, ctx_key = cell_key.split("-", 1)
        src = _stage_con_source(cfg, beh_key, ctx_key)
        _, mix_sub, meta_sub = _con_source_layout(beh_key, ctx_key)
        mix_path = src / mix_sub
        con_sha = hashlib.sha256(mix_path.read_bytes()).hexdigest()
        pin_src = _PARENT_MANIFESTS.get(cell_key)
        if pin_src is not None:
            manifest_path = cells.REPO_ROOT / pin_src
            pins = {
                r["run_id"]: r["train_mix_sha256"]
                for r in run1090._read_json(manifest_path)["runs"]
                if "train_mix_sha256" in r
            }
            want = pins.get(_PARENT_MANIFEST_RUN[cell_key])
            if want is not None and want != con_sha:
                raise RuntimeError(
                    f"[i1481-po-mixes] {cell_key}: staged con mix sha {con_sha} != "
                    f"committed {pin_src} pin {want} — frozen-mix premise broken; refusing"
                )
        po_rows, derivation = w1434._derive_po_rows(
            cell_key,
            _read_jsonl(mix_path),
            _read_jsonl(src / "datagen" / "cn.jsonl"),
            _read_jsonl(src / "datagen" / "pos.jsonl"),
            _generic,
            cfg.seed,
        )
        out_dir = Path(cfg.out_root) / "po_mixes" / cell_key / "mix"
        out_dir.mkdir(parents=True, exist_ok=True)
        mix_out = out_dir / "train_mix.jsonl"
        _write_jsonl(mix_out, po_rows)
        po_sha = hashlib.sha256(mix_out.read_bytes()).hexdigest()
        parent_meta = run1090._read_json(src / meta_sub)
        derivation.update({"parent_cell": cell_key, "parent_mix_sha256": con_sha})
        meta = {
            **parent_meta,
            "counts_planned": {"positives": 20, "negatives": 0, "generic": 40},
            "counts_realized": {"positives": 20, "negatives": 0, "generic": 40},
            "train_mix_sha256": po_sha,
            "po_derivation": derivation,
        }
        run1090._atomic_write_json(out_dir / "mix_meta.json", meta)
        derivations[cell_key] = derivation
        if cfg.upload:
            pir = f"{cells.DATA_PREFIX_1481}/po_mixes/{cell_key}/mix"
            url = upload(out_dir, run1090.HF_DATA_REPO, "dataset", pir)
            if not str(url):
                raise RuntimeError(f"upload returned no path for {pir} — refusing silent loss")
            for fname in ("train_mix.jsonl", "mix_meta.json"):
                ok = hub.retry_transient(
                    # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient (this call)
                    lambda p=f"{pir}/{fname}": api.file_exists(
                        run1090.HF_DATA_REPO, p, repo_type="dataset"
                    ),
                    what=f"po mix verify {pir}/{fname}",
                )
                if not ok:
                    raise RuntimeError(f"[i1481-po-mixes] {pir}/{fname} missing on the data repo")
        rows_out.append(
            {
                "cell_key": cell_key,
                "parent_mix_sha256": con_sha,
                "train_mix_sha256": po_sha,
                "mix_hub_prefix": f"{cells.DATA_PREFIX_1481}/po_mixes/{cell_key}/mix",
            }
        )
    manifest = {
        "issue": cells.ISSUE_1481,
        "phase": "po_mixes",
        "cells": rows_out,
        "po_derivations": derivations,
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = Path(cfg.out_root) / "cell_manifest_i1481_pomixes.json"
    run1090._atomic_write_json(out_path, manifest)
    if not cfg.smoke:
        committed = cells.DELIVERABLES_DIR_1481 / "cell_manifest_i1481_pomixes.json"
        committed.parent.mkdir(parents=True, exist_ok=True)
        run1090._atomic_write_json(committed, manifest)
        logger.info("[i1481-po-mixes] committed manifest at %s (commit BEFORE dispatch)", committed)
    logger.info("[i1481-po-mixes] derived %d po mixes", len(rows_out))
    return 0


# ── Reused-arm apply-and-read (gate P1; plan §4.6) ───────────────────────────


def _committed_selection(arm: cells.ReusedConArm) -> dict:
    """The committed ladders JSON's ``runs[<arm>].selection`` record."""
    ladders = run1090._read_json(cells.REPO_ROOT / arm.ladders_path)
    run_rec = (ladders.get("runs") or {}).get(arm.source_run_id)
    if not run_rec or "selection" not in run_rec:
        raise RuntimeError(
            f"[i1481-reread] {arm.arm_id}: no committed selection for "
            f"{arm.source_run_id} in {arm.ladders_path}"
        )
    return run_rec["selection"]


def _assert_reused_gauge(arm: cells.ReusedConArm, ckpt_dir: Path) -> dict:
    """Gate P3 (plan §4.6): the reused adapter's own adapter_config.json must
    match the fu4 content bundle (r32/α64 rsLoRA) with no unembedding touch."""
    cfg_path = ckpt_dir / "adapter_config.json"
    rec = json.loads(cfg_path.read_text())
    got = {
        "r": rec.get("r"),
        "lora_alpha": rec.get("lora_alpha"),
        "use_rslora": rec.get("use_rslora"),
    }
    if got != P3_EXPECTED:
        raise RuntimeError(f"[i1481-P3] {arm.arm_id}: adapter gauge {got} != {P3_EXPECTED}")
    tmods = set(rec.get("target_modules") or [])
    if tmods & {"lm_head", "embed_tokens"} or (rec.get("modules_to_save") or []):
        raise RuntimeError(
            f"[i1481-P3] {arm.arm_id}: adapter touches the unembedding "
            f"(target_modules={sorted(tmods)}, modules_to_save={rec.get('modules_to_save')})"
        )
    return rec


def phase_reread(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """P1 apply-and-read: for each reused fu4/fu5/fu7 committed selected
    checkpoint, re-read its Tier-1 source rate under THIS run's instrument
    (fresh 20q × 5 gens, judged) and persist |re-read − committed| (WARN
    semantics; a committed-IN-BAND arm re-reading out of band raises the
    pre-registered contingent-regen flag for the orchestrator — plan §4.6)."""
    run1090._phase("i1481_reread")
    arms = list(cells.REUSED_CON_ARMS)
    if args.arms:
        want = {t.strip() for t in args.arms.split(",") if t.strip()}
        bad = want - set(cells.REUSED_CON_ARM_BY_ID)
        if bad:
            raise ValueError(f"bad reread arms {sorted(bad)}")
        arms = [a for a in arms if a.arm_id in want]
    elif cfg.smoke:
        arms = [cells.REUSED_CON_ARM_BY_ID["imp-pers-con-lr3e5-s42"]]
    reread_root = Path(cfg.out_root) / "reread"
    seams = fu4.make_fu4_smoke_seams(cfg) if cfg.smoke else run1090.Seams1090()
    gen = (
        seams.eval_gen_fn_factory(DEFAULT_BASE_MODEL)
        if seams.eval_gen_fn_factory is not None
        else _default_vllm_generate_fn(DEFAULT_BASE_MODEL, max_lora_rank=64)
    )
    records: dict[str, dict] = {}
    try:
        for arm in arms:
            committed = _committed_selection(arm)
            step = int(committed["step"])
            committed_rate = float(committed["rate"])
            ckpt_dir = reread_root / arm.arm_id / f"checkpoint-{step}"
            ckpt_prefix = f"{arm.adapter_run_prefix}/checkpoint-{step}"
            if cfg.smoke:
                # tiny-real: the committed adapter_config.json (KB-scale) is
                # staged for the REAL P3 gauge probe; the stub gen never loads
                # weights, so the full safetensors stay on the Hub.
                hub.stage_hub_file(
                    run1090.HF_MODEL_REPO,
                    f"{ckpt_prefix}/adapter_config.json",
                    ckpt_dir / "adapter_config.json",
                    repo_type="model",
                )
            else:
                hub.stage_hub_prefix(
                    run1090.HF_MODEL_REPO, ckpt_prefix, ckpt_dir, repo_type="model"
                )
            _assert_reused_gauge(arm, ckpt_dir)
            ctx = fu3w.ensure_context(arm.context_id, arm.behavior)
            organism = ModelOrganism(
                behavior=arm.behavior,
                context_id=arm.context_id,
                negatives=fu3w.panel_name_for(ctx),
                seed=cfg.seed,
            )
            rate_fn = make_source_rate_fn(
                organism,
                out_dir=reread_root / arm.arm_id / "rate",
                eval_questions=run1090._eval_questions(cfg, arm.behavior),
                n_completions=cfg.tier1_n,
                temperature=1.0,
                n_judge_draws=cfg.tier1_draws,
                generate_fn=gen,
                judge_fn=fu3w.judge_graded_r23,  # the fu4/fu7 ladder instrument
            )
            try:
                reread_rate = float(rate_fn(str(ckpt_dir)))
            finally:
                close = getattr(rate_fn, "close", None)
                if callable(close):
                    close()
            delta = reread_rate - committed_rate
            lo, hi = cells.JUDGED_RATE_BAND
            rec = {
                "arm_id": arm.arm_id,
                "source_run_id": arm.source_run_id,
                "committed_step": step,
                "committed_rate": committed_rate,
                "reread_rate": reread_rate,
                "abs_delta": abs(delta),
                "parity_ok": abs(delta) <= cells.P1_PARITY_MAX_ABS_DELTA,
                "committed_in_band": arm.committed_in_band,
                "reread_in_band": lo <= reread_rate <= hi,
                # ops consequence, never a HALT (plan §4.6): orchestrator runs
                # the pre-registered contingent regen on this flag.
                "contingent_regen_triggered": bool(
                    arm.committed_in_band and not (lo <= reread_rate <= hi)
                ),
            }
            if not rec["parity_ok"]:
                logger.warning("[i1481-P1] %s parity WARN: %s", arm.arm_id, rec)
            records[arm.arm_id] = rec
            run1090._atomic_write_json(reread_root / f"{arm.arm_id}.json", rec)
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()
    summary = {
        "issue": cells.ISSUE_1481,
        "gate": "P1-apply-and-read",
        "arms": records,
        "n_parity_warn": sum(1 for r in records.values() if not r["parity_ok"]),
        "n_regen_triggered": sum(1 for r in records.values() if r["contingent_regen_triggered"]),
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    run1090._atomic_write_json(reread_root / "reread_summary.json", summary)
    if cfg.upload:
        url = hub._upload(
            reread_root,
            run1090.HF_DATA_REPO,
            "dataset",
            f"{cells.DATA_PREFIX_1481}/raw_completions/reread",
        )
        if not str(url):
            raise RuntimeError("reread upload returned no path — refusing silent loss")
    logger.info(
        "[i1481-reread] %d arms, %d parity WARN, %d regen-triggered",
        len(records),
        summary["n_parity_warn"],
        summary["n_regen_triggered"],
    )
    return 0


# ── Base arms (Phase C gap fill) ─────────────────────────────────────────────


def phase_base_arms(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Fresh per-context BASE Tier-2 arms + the shared 6-context base panel
    for ONE behavior (--behavior), for contexts with no committed base read
    (plan §4.3 baseline-propensity row; the #1434 phase shape)."""
    behavior = args.behavior
    if behavior not in cells.BEHAVIOR_BY_KEY.values():
        raise SystemExit(f"--phase base-arms requires --behavior in {cells.BEHAVIOR_BY_KEY}")
    beh_key = {v: k for k, v in cells.BEHAVIOR_BY_KEY.items()}[behavior]
    run1090._phase("i1481_base_arms")
    qs = run1090._eval_questions(cfg, behavior)
    seams = fu4.make_fu4_smoke_seams(cfg) if cfg.smoke else run1090.Seams1090()
    gen = (
        seams.eval_gen_fn_factory(DEFAULT_BASE_MODEL)
        if seams.eval_gen_fn_factory is not None
        else _default_vllm_generate_fn(DEFAULT_BASE_MODEL, max_lora_rank=64)
    )
    base_root = Path(cfg.out_root) / "base_arms" / beh_key
    ctx_keys = ["pers"] if cfg.smoke and not args.cells else list(cells.CTX_KEYS)
    if args.cells:
        ctx_keys = [t.strip() for t in args.cells.split(",") if t.strip()]
    try:
        for ctx_key in ctx_keys:
            ctx = fu3w.ensure_context(cells.context_id_for(behavior, ctx_key), behavior)
            _generate_and_persist(
                gen,
                "base",
                None,
                ctx,
                qs,
                n=cfg.tier2_n,
                temperature=1.0,
                out_dir=base_root / ctx_key / "tier2",
                base_model=DEFAULT_BASE_MODEL,
            )
        run1090._phase("i1481_base_panel")
        for bctx in fu3w.bystander_panel(behavior):
            _generate_and_persist(
                gen,
                "base",
                None,
                bctx,
                qs,
                n=cfg.tier1_n,
                temperature=1.0,
                out_dir=base_root / "panel",
                base_model=DEFAULT_BASE_MODEL,
            )
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()
    if cfg.upload:
        url = hub._upload(
            base_root,
            run1090.HF_DATA_REPO,
            "dataset",
            f"{cells.DATA_PREFIX_1481}/raw_completions/base_arms/{beh_key}",
        )
        if not str(url):
            raise RuntimeError("base_arms upload returned no path — refusing silent loss")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def _own_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="#1481 con-vs-pos content-grid worker")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--phase", required=True, choices=OWN_PHASES)
    p.add_argument("--cells", default=None, help="comma cell/context subset (smoke parity)")
    p.add_argument("--arms", default=None, help="comma reused-arm subset (--phase reread)")
    p.add_argument("--behavior", default=None, help="--phase base-arms behavior")
    p.add_argument("--out-root", default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    return p


def _argv_get(argv: list[str], flag: str) -> str | None:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _argv_set(argv: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(argv):
        tok = argv[i]
        if tok == flag and i + 1 < len(argv):
            out += [flag, value]
            replaced = True
            i += 2
            continue
        if tok.startswith(flag + "="):
            out.append(f"{flag}={value}")
            replaced = True
            i += 1
            continue
        out.append(tok)
        i += 1
    if not replaced:
        out += [flag, value]
    return out


def _delegate_fu4(argv: list[str]) -> int:
    """Register the i1481 rounds, thread seed-per-run, delegate to fu4.main."""
    cells.register_i1481_rounds()
    round_arg = _argv_get(argv, "--round")
    if round_arg is None:
        raise SystemExit(
            f"delegated phases require --round <name> (one of {sorted(cells.I1481_ROUND_NAMES)})"
        )
    phase = _argv_get(argv, "--phase")
    if phase == "run":
        run_id = _argv_get(argv, "--run")
        if run_id is None:
            raise SystemExit("--phase run requires --run <run_id>")
        # Seed threading (plan §4.7): the run's OWN seed overrides whatever the
        # dispatcher-level --seed carried (Fu4Run has no seed field).
        argv = _argv_set(argv, "--seed", str(cells.seed_for_run_id(run_id)))
    return fu4.main(argv)


def _cohort_run_ids(rname: str, seed: int, runs_arg: str | None, smoke: bool) -> list[str]:
    """The (round × seed) dispatch cohort: seeds are baked into run_ids and the
    fu4 regime key carries ONE cfg.seed per out_root, so each seed cohort gets
    its own fu4 invocation + out_root (regime coherence; plan §4.7 seed
    threading)."""
    spec = fu4.ROUNDS[rname]
    if runs_arg:
        # --runs is a GROUP-level subset: intersect with THIS round's registry
        # (ids are validated against the group union in _run_dispatch_group).
        round_ids = {r.run_id for r in spec.runs}
        wanted = [t.strip() for t in runs_arg.split(",") if t.strip() and t.strip() in round_ids]
    elif smoke:
        wanted = [t.strip() for t in spec.smoke_default_run.split(",") if t.strip()]
    else:
        wanted = [r.run_id for r in spec.runs]
    return [rid for rid in wanted if cells.seed_for_run_id(rid) == seed]


def _run_dispatch_group(argv: list[str], group: str) -> int:
    """`--dispatch <group>`: stage → dispatch per (round × seed cohort) for the
    group's con + po rounds (sequential fu4 invocations; the fu4 dispatcher is
    work-conserving within each cohort), then the group's reused-arm re-reads
    (gate P1)."""
    if group not in cells.DISPATCH_ROUNDS:
        raise SystemExit(
            f"unknown dispatch group {group!r}: want one of {sorted(cells.DISPATCH_ROUNDS)}"
        )
    seeds_arg = _argv_get(argv, "--seeds")
    seeds = (
        tuple(int(t) for t in seeds_arg.replace(" ", "").split(",") if t)
        if seeds_arg
        else cells.SEEDS
    )
    bad_seeds = [s for s in seeds if s not in cells.SEEDS]
    if bad_seeds:
        raise SystemExit(f"--seeds {seeds_arg!r}: unknown seeds {bad_seeds} (grid: {cells.SEEDS})")
    smoke = "--smoke" in argv
    mode = "--smoke" if smoke else "--full"
    passthrough: list[str] = []
    for flag in ("--eval-question-limit", "--n-gpus"):
        val = _argv_get(argv, flag)
        if val is not None:
            passthrough += [flag, val]
    if "--no-upload" in argv:
        passthrough.append("--no-upload")
    cells.register_i1481_rounds()
    runs_arg = _argv_get(argv, "--runs")
    if runs_arg:
        union = {r.run_id for rn in cells.DISPATCH_ROUNDS[group] for r in fu4.ROUNDS[rn].runs}
        bad = [t.strip() for t in runs_arg.split(",") if t.strip() and t.strip() not in union]
        if bad:
            raise SystemExit(f"[i1481-dispatch] unknown runs for group {group!r}: {bad}")
    out_base = Path(
        _argv_get(argv, "--out-root")
        or ("/tmp/issue-1481-smoke" if smoke else "data/issue_1481/cells")
    )
    sentinel_arg = _argv_get(argv, "--sentinel-dir")
    for rname in cells.DISPATCH_ROUNDS[group]:
        for seed in seeds:
            cohort = _cohort_run_ids(rname, seed, _argv_get(argv, "--runs"), smoke)
            if not cohort:
                logger.info("[i1481-dispatch] %s s%d: empty cohort — skipping", rname, seed)
                continue
            root = out_base / f"{rname}-s{seed}"
            manifest = (
                root / fu4.ROUNDS[rname].manifest_name
                if smoke
                else cells.DELIVERABLES_DIR_1481 / f"cell_manifest_{rname}_s{seed}.json"
            )
            common = [
                mode,
                "--round",
                rname,
                "--runs",
                ",".join(cohort),
                "--seed",
                str(seed),
                "--out-root",
                str(root),
                *passthrough,
            ]
            if sentinel_arg is not None:
                common += ["--sentinel-dir", sentinel_arg]
            elif smoke:
                common += ["--sentinel-dir", str(root / "logs")]
            for phase, extra in (
                ("stage", ["--manifest-out", str(manifest)]),
                ("dispatch", ["--manifest", str(manifest)]),
            ):
                rc = _delegate_fu4(["--phase", phase, *common, *extra])
                if rc != 0:
                    logger.error(
                        "[i1481-dispatch] round=%s seed=%d phase=%s rc=%d — stopping",
                        rname,
                        seed,
                        phase,
                        rc,
                    )
                    return rc
    arms = cells.REREAD_BY_DISPATCH[group]
    if arms:
        own = ["--phase", "reread", mode]
        if not smoke:
            own += ["--arms", ",".join(a.arm_id for a in arms)]
        # smoke: phase_reread's 1-arm smoke default keeps the P1 path tiny-real
        own += ["--out-root", str(out_base)]
        for flag in ("--sentinel-dir", "--eval-question-limit"):
            val = _argv_get(argv, flag)
            if val is not None:
                own += [flag, val]
        if "--no-upload" in argv:
            own.append("--no-upload")
        args = _own_parser().parse_args(own)
        return phase_reread(worker_config(args), args)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    group = _argv_get(argv, "--dispatch")
    if group is not None:
        return _run_dispatch_group(argv, group)
    phase = _argv_get(argv, "--phase")
    if phase in FU4_DELEGATED_PHASES:
        return _delegate_fu4(argv)
    cells.register_i1481_rounds()  # contexts/rounds available to own phases too
    args = _own_parser().parse_args(argv)
    cfg = worker_config(args)
    logger.info(
        "issue1481_worker phase=%s smoke=%s out_root=%s", args.phase, cfg.smoke, cfg.out_root
    )
    if args.phase == "mixes":
        return phase_mixes(cfg, args)
    if args.phase == "reread":
        return phase_reread(cfg, args)
    return phase_base_arms(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
