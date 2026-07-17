#!/usr/bin/env python
"""#1434 — writing_style organism-factory driver (plan §4 D0-D3 phase chain).

Thin composition over the #1090 machinery. Phases:

VM (pre-pod):
- ``--phase questiongen``  D0 bank generation (delegates issue1434_questiongen).
- ``--phase datagen``      D1: per-context contrastive datagen (fu3 worker seams)
                           + mix assembly (organisms._assemble_mix) + the
                           generic-corpus revision/sha provenance pin + HF upload.
- ``--phase stage``        manifest (per-run train_mix_sha256 pins) + HF verify.

Pod (GPU; ``bash scripts/issue1434_dispatch.sh`` sequences them):
- ``--phase dispatch``/``--phase run``  DELEGATED verbatim to the
  round-parametrized fu4 driver (``issue1090_fu4.main``) after registering the
  ``i1434`` round — train (K2 divergence + adapter-rank gauge) -> Tier-1 ladder
  (pv-rubric judge seam) -> dose-select -> Tier-2 trained gen -> tf-margin ->
  per-run upload -> sentinel. Smoke = the SAME path on 1 run (PASS_UNIFIED).
- ``--phase base-arms``    per-context base Tier-2 gen + the shared 6-context
                           base bystander panel (fresh base arms; plan D3).
- ``--phase panel``        per-context verdict-arm bystander panel gens.

VM (post-pod):
- ``--phase judge-analyze``  Batch-API pv judging of Tier-2/base/panel, Wilson/
  Newcombe, the §3 verdict lattice, leakage, the registered-rubric parity
  re-read, drop-split report -> eval_results/issue_1434/ aggregates.

Pod-side code NEVER shells scripts/task.py; sentinels ride the fu4 contract
(``/workspace/logs/issue-1434-*.json`` + out-of-glob status twins).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1074_generator_compare as i1074  # noqa: E402
import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_fu4 as fu4  # noqa: E402
import issue1090_run as run1090  # noqa: E402
import issue1434_cells as cells  # noqa: E402

from explore_persona_space.artifacts import negatives as neg_mod  # noqa: E402
from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    ModelOrganism,
    _assemble_mix,
    _default_vllm_generate_fn,
    _generate_and_persist,
)
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1434.worker")

FU4_DELEGATED_PHASES = ("dispatch", "run")
JUDGE_DROP_FLAG_BAR = 0.10  # inherited flag check (llm-judging rule 23; flag, never kill)


# ── config ───────────────────────────────────────────────────────────────────


def worker_config(args: argparse.Namespace) -> run1090.RunConfig:
    """The #1434 RunConfig (fu4_config shape, i1434 round selected first)."""
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else ("/tmp/issue-1434-i1434-smoke" if smoke else "data/issue_1434/cells")
    )
    return run1090.RunConfig(
        smoke=smoke,
        cells=(),  # phase-specific shims are built per cell below
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


def resolve_cell_keys(cells_arg: str | None, smoke: bool) -> list[str]:
    """The ONE context-cell resolver every VM/pod phase consumes (smoke = the
    SAME path on the persona cell — the run-resolver's cell twin)."""
    if cells_arg:
        keys = [t.strip() for t in cells_arg.split(",") if t.strip()]
        bad = [k for k in keys if k not in cells.CONTEXT_BY_CELL_KEY]
        if bad:
            raise ValueError(f"bad #1434 cells {bad!r}: known {sorted(cells.CONTEXT_BY_CELL_KEY)}")
        return keys
    if smoke:
        return ["ws-pers"]
    return list(cells.CELL_KEYS)


def _cell_shim(cell_key: str) -> run1090.Cell:
    """A run1090.Cell whose slug is the #1434 cell_key (distinct paths/runs)."""
    return run1090.Cell(
        cell_id=cell_key,
        behavior=cells.BEHAVIOR,
        generator="claude",
        trains=True,
        purpose=f"#1434 contrastive writing_style @ {cells.CONTEXT_BY_CELL_KEY[cell_key]}",
    )


def _eval_questions(cfg: run1090.RunConfig) -> list[str]:
    return run1090._eval_questions(cfg, cells.BEHAVIOR)


# ── D1: datagen + mix (VM, API-only) ─────────────────────────────────────────


def _generic_corpus_provenance(local_path: Path) -> dict:
    """Plan §10 fitness pin: the staged generic corpus's HF revision + blob sha
    (``_stage_generic_corpus`` has a dest-exists short-circuit and no per-file
    sha assert — the pin makes the consumed bytes auditable)."""
    from huggingface_hub import HfApi

    sha = hashlib.sha256(local_path.read_bytes()).hexdigest()
    info = hub.retry_transient(
        lambda: HfApi().get_paths_info(
            run1090.HF_DATA_REPO,
            [i1074.GENERIC_CORPUS_HF_PATH],
            repo_type="dataset",
            expand=True,
        ),
        what="generic-corpus get_paths_info",
    )
    if not info:
        raise RuntimeError(
            f"generic corpus {i1074.GENERIC_CORPUS_HF_PATH} not found on "
            f"{run1090.HF_DATA_REPO} — cannot pin provenance"
        )
    last = info[0].last_commit
    return {
        "hf_repo": run1090.HF_DATA_REPO,
        "hf_path": i1074.GENERIC_CORPUS_HF_PATH,
        "staged_sha256": sha,
        "hub_last_commit_oid": getattr(last, "oid", None),
        "hub_last_commit_date": str(getattr(last, "date", None)),
        "hub_blob_size": getattr(info[0], "size", None),
    }


def phase_datagen(cfg: run1090.RunConfig, args: argparse.Namespace) -> dict:
    """D1: per-context contrastive datagen + 80-row mix + provenance + upload.

    A ``DatagenYieldError`` below the floor is the plan-§7 G1 registered kill
    path: the cell (and its 3 lr runs) is recorded ``yield_floor_missed`` and
    SKIPPED — reported, never backfilled.
    """
    run1090._phase("i1434_datagen")
    from transformers import AutoTokenizer

    out: dict[str, dict] = {}
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    generic_path = Path(
        i1074._stage_generic_corpus(cfg.out_root / "inputs" / "generic_corpus.jsonl")
    )
    generic_prov = _generic_corpus_provenance(generic_path)
    upload = run1090._upload_fn(run1090.Seams1090())
    for cell_key in resolve_cell_keys(args.cells, cfg.smoke):
        shim = _cell_shim(cell_key)
        ctx = cells.ensure_ws_context(cells.CONTEXT_BY_CELL_KEY[cell_key])
        panel_name = fu3w.panel_name_for(ctx)
        panel = neg_mod.NEGATIVE_PANELS[panel_name]
        cell_root = cfg.out_root / "datagen_cells" / cell_key
        summary_path = cell_root / "datagen_summary_1434.json"
        if summary_path.exists():
            out[cell_key] = run1090._read_json(summary_path)
            logger.info("[i1434-datagen] %s already recorded — skip", cell_key)
            continue
        cell_cfg = dataclasses.replace(
            cfg,
            cells=(shim,),
            oversample_mult=(
                fu3w.BARE_OVERSAMPLE_MULT if ctx.kind == "bare" else fu3w.DEFAULT_OVERSAMPLE_MULT
            ),
        )
        record: dict[str, Any] = {
            "cell_key": cell_key,
            "behavior": cells.BEHAVIOR,
            "context_id": ctx.context_id,
            "panel_name": panel_name,
            "oversample_mult": cell_cfg.oversample_mult,
            "target_n": cell_cfg.target_n,
            "seed": cfg.seed,
            "generic_corpus_provenance": generic_prov,
            "git_commit": i1074._git_short_sha(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        dg_fn = fu3w._fu3_datagen_fn(cell_cfg, shim, posonly=False)
        try:
            pos_path, cn_path, meta_path = dg_fn(
                BEHAVIORS[cells.BEHAVIOR],
                ctx,
                panel,
                out_dir=cell_root / "datagen",
                seed=cfg.seed,
                **run1090._datagen_kwargs(cell_cfg, shim, None),
            )
        except run1090.DatagenYieldError as e:
            record.update(status="yield_floor_missed", reason=str(e))
            run1090._atomic_write_json(summary_path, record)
            out[cell_key] = record
            logger.warning("[i1434-datagen] %s G1 yield miss: %s", cell_key, e)
            continue
        organism = ModelOrganism(
            behavior=cells.BEHAVIOR, context_id=ctx.context_id, negatives=panel_name, seed=cfg.seed
        )
        spec = dataclasses.replace(
            organism.recipe,
            overrides={**organism.recipe.overrides, "max_length": run1090.MAX_LENGTH_1090},
        )
        mix_dir = cell_root / "mix"
        train_mix_path, counts, realized = _assemble_mix(
            organism,
            spec,
            Path(pos_path),
            Path(cn_path),
            generic_path,
            mix_dir,
            tokenizer=tokenizer,
            max_length=run1090.MAX_LENGTH_1090,
        )
        if not cfg.smoke:
            got = tuple(sorted(int(v) for v in realized.values()))
            if got != tuple(sorted(fu4.EXPECTED_MIX_COMPOSITION)):
                raise ValueError(
                    f"[i1434-datagen] {cell_key}: mix composition {realized} != "
                    f"{fu4.EXPECTED_MIX_COMPOSITION} — refusing to upload a wrong-shape mix"
                )
        mix_meta = run1090._read_json(mix_dir / "mix_meta.json")
        mix_meta["generic_corpus_provenance"] = generic_prov
        run1090._atomic_write_json(mix_dir / "mix_meta.json", mix_meta)
        record.update(
            status="success",
            pos_path=str(pos_path),
            cn_path=str(cn_path),
            pool_meta_path=str(meta_path),
            counts_planned=counts,
            counts_realized=realized,
            train_mix_sha256=hashlib.sha256(Path(train_mix_path).read_bytes()).hexdigest(),
        )
        if cfg.upload:
            base_pir = f"{cells.DATA_PREFIX_1434}/{cell_key}"
            for local, pir, kw in (
                (
                    cell_root / "datagen",
                    f"{base_pir}/datagen",
                    {"ignore_patterns": ["gen_cache*", "gen_ckpt_*", "judge_cache*"]},
                ),
                (mix_dir, f"{base_pir}/mix", {}),
            ):
                url = upload(local, run1090.HF_DATA_REPO, "dataset", pir, **kw)
                if not str(url):
                    raise RuntimeError(f"upload returned no path for {pir} — refusing silent loss")
            record["hf_prefix"] = base_pir
        run1090._atomic_write_json(summary_path, record)
        out[cell_key] = record
    run1090._atomic_write_json(cfg.out_root / "datagen_results_1434.json", out)
    return out


def phase_stage(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Manifest build (per-run train_mix_sha256 pins) + HF mix verification.

    Writes ``cell_manifest_i1434.json`` under out_root; a full run ALSO writes
    the committed copy under eval_results/issue_1434/ (smoke never touches the
    committed path — scratch-redirect discipline).
    """
    run1090._phase("i1434_stage")
    from huggingface_hub import HfApi

    results = run1090._read_json(cfg.out_root / "datagen_results_1434.json")
    api = HfApi()
    runs: list[dict] = []
    skipped: list[str] = []
    for cell_key in resolve_cell_keys(args.cells, cfg.smoke):
        rec = results.get(cell_key)
        if rec is None:
            raise RuntimeError(f"[i1434-stage] no datagen record for {cell_key} — run datagen")
        if rec.get("status") != "success":
            skipped.append(cell_key)
            continue
        if cfg.upload:
            for fname in ("train_mix.jsonl", "mix_meta.json"):
                path = f"{cells.mix_hub_prefix(cell_key)}/{fname}"
                ok = hub.retry_transient(
                    lambda p=path: api.file_exists(run1090.HF_DATA_REPO, p, repo_type="dataset"),
                    what=f"stage verify {path}",
                )
                if not ok:
                    raise RuntimeError(f"[i1434-stage] {path} missing on the data repo")
        for run in cells.I1434_RUNS:
            if run.cell_key == cell_key:
                runs.append(
                    {
                        "run_id": run.run_id,
                        "cell_key": cell_key,
                        "lr": run.lr,
                        "train_mix_sha256": rec["train_mix_sha256"],
                        "mix_hub_prefix": run.mix_hub_prefix,
                    }
                )
    manifest = {
        "issue": cells.ISSUE_1434,
        "round": "writingstyle-pv-install",
        "runs": runs,
        "skipped_cells_yield_floor": skipped,
        "generic_corpus_provenance": results[resolve_cell_keys(args.cells, cfg.smoke)[0]].get(
            "generic_corpus_provenance"
        ),
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = cfg.out_root / "cell_manifest_i1434.json"
    run1090._atomic_write_json(out_path, manifest)
    if not cfg.smoke:
        committed = cells.DELIVERABLES_DIR_1434 / "cell_manifest_i1434.json"
        run1090._atomic_write_json(committed, manifest)
        logger.info("[i1434-stage] committed manifest copy at %s", committed)
    logger.info("[i1434-stage] %d runs pinned; %d cells yield-skipped", len(runs), len(skipped))
    return 0


def make_i1434_smoke_seams(cfg: run1090.RunConfig) -> run1090.Seams1090:
    """The parent tiny-real seams keyed on the writing_style question banks
    (installs the from-config tiny-Qwen ``from_pretrained`` patch)."""
    return run1090.make_smoke_seams(dataclasses.replace(cfg, cells=(_cell_shim("ws-pers"),)))


def _gen_fn(cfg: run1090.RunConfig):
    """The eval/rollout generation engine: tiny-real stub under smoke, else the
    shared vLLM engine (LoRA hot-load; 64-slot width)."""
    if cfg.smoke:
        return make_i1434_smoke_seams(cfg).eval_gen_fn_factory(DEFAULT_BASE_MODEL)
    return _default_vllm_generate_fn(DEFAULT_BASE_MODEL, max_lora_rank=64)


def _hf_model(cfg: run1090.RunConfig):
    """HF model + tokenizer for teacher-forced capture (tiny-real under smoke;
    bf16 on the CVD-pinned GPU otherwise — device_map pinned, never 'auto')."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cfg.smoke:
        make_i1434_smoke_seams(cfg)  # installs the tiny-Qwen from_pretrained patch
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": {"": 0}}
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_BASE_MODEL, **kwargs)
    model.eval()
    return model, tokenizer


# ── D3: base arms + verdict-arm bystander panel (pod GPU) ────────────────────


def phase_base_arms(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Fresh per-context BASE Tier-2 arms + the shared 6-context base panel."""
    run1090._phase("i1434_base_arms")
    qs = _eval_questions(cfg)
    gen = _gen_fn(cfg)
    base_root = cfg.out_root / "base_arms"
    try:
        for cell_key in resolve_cell_keys(args.cells, cfg.smoke):
            ctx = cells.ensure_ws_context(cells.CONTEXT_BY_CELL_KEY[cell_key])
            _generate_and_persist(
                gen,
                "base",
                None,
                ctx,
                qs,
                n=cfg.tier2_n,
                temperature=1.0,
                out_dir=base_root / cell_key / "tier2",
                base_model=DEFAULT_BASE_MODEL,
            )
        run1090._phase("i1434_base_panel")
        for bctx in fu3w.bystander_panel(cells.BEHAVIOR):
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
            f"{cells.DATA_PREFIX_1434}/raw_completions/base_arms",
        )
        if not str(url):
            raise RuntimeError("base_arms upload returned no path — refusing silent loss")
    return 0


def _run_selections(cfg: run1090.RunConfig, run_ids: list[str]) -> dict[str, dict]:
    """Per-run dose-selection records from the fu4 build results (fail-loud)."""
    sels: dict[str, dict] = {}
    for run_id in run_ids:
        path = cfg.out_root / run_id / "i1434_build_result.json"
        if not path.exists():
            raise RuntimeError(f"[i1434-panel] missing build result {path} — run dispatch first")
        rec = run1090._read_json(path)
        if rec.get("status") == "diverged":
            continue  # a K2-diverged arm carries no selection (recorded answer)
        sels[run_id] = rec["selection"]
    return sels


def phase_panel(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Bystander-panel generation for the per-context VERDICT arms (plan §3
    pre-registered selection rule) at their selected rungs."""
    run1090._phase("i1434_panel")
    qs = _eval_questions(cfg)
    cell_keys = resolve_cell_keys(args.cells, cfg.smoke)
    run_ids = [r.run_id for r in cells.I1434_RUNS if r.cell_key in cell_keys]
    selections = _run_selections(cfg, run_ids)
    verdicts: dict[str, dict] = {}
    gen = _gen_fn(cfg)
    panel_root = cfg.out_root / "panel"
    try:
        for cell_key in cell_keys:
            arm_sels = {
                rid: s
                for rid, s in selections.items()
                if cells.RUN_BY_ID_1434[rid].cell_key == cell_key
            }
            if not arm_sels:
                logger.warning("[i1434-panel] %s: no non-diverged arms — skipping", cell_key)
                verdicts[cell_key] = {"rule": "no_arms", "run_id": None}
                continue
            run_id, rec = (
                cells.verdict_arm_for_context(cell_key, arm_sels)
                if len(arm_sels) == 3
                else _verdict_from_partial(cell_key, arm_sels)
            )
            verdicts[cell_key] = rec
            build = run1090._read_json(cfg.out_root / run_id / "i1434_build_result.json")
            ckpt = build["selected_ckpt"]
            for bctx in fu3w.bystander_panel(cells.BEHAVIOR):
                _generate_and_persist(
                    gen,
                    "trained",
                    ckpt,
                    bctx,
                    qs,
                    n=cfg.tier1_n,
                    temperature=1.0,
                    out_dir=panel_root / run_id,
                    base_model=DEFAULT_BASE_MODEL,
                )
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()
    run1090._atomic_write_json(panel_root / "verdict_arms.json", verdicts)
    if cfg.upload:
        url = hub._upload(
            panel_root,
            run1090.HF_DATA_REPO,
            "dataset",
            f"{cells.DATA_PREFIX_1434}/raw_completions/panel",
        )
        if not str(url):
            raise RuntimeError("panel upload returned no path — refusing silent loss")
    return 0


def _verdict_from_partial(cell_key: str, arm_sels: dict[str, dict]) -> tuple[str, dict]:
    """Verdict arm over the SURVIVING (non-diverged) arms — same registered
    rule, denominator honestly recorded (a diverged arm is an answer, not a
    silent hole)."""
    subset = {rid: s for rid, s in arm_sels.items() if rid in cells.RUN_BY_ID_1434}
    arms = sorted(
        (cells.RUN_BY_ID_1434[rid] for rid in subset),
        key=lambda r: r.lr,
    )
    for r in arms:
        if bool(subset[r.run_id].get("in_band")):
            return r.run_id, {
                "rule": "lowest_lr_in_band_partial",
                "run_id": r.run_id,
                "lr": r.lr,
                "n_arms": len(arms),
                "selection": subset[r.run_id],
            }
    best = min(arms, key=lambda r: (cells._band_distance(float(subset[r.run_id]["rate"])), r.lr))
    return best.run_id, {
        "rule": "closest_approach_partial",
        "run_id": best.run_id,
        "lr": best.lr,
        "n_arms": len(arms),
        "selection": subset[best.run_id],
    }


# ── VM P10: judging + verdicts (post-pod) ────────────────────────────────────


def _completions_payload(path: Path) -> list[list[str]]:
    return json.loads(path.read_text())["completions"]


def _judge_rate_graded(
    tag: str,
    qs: list[str],
    comps: list[list[str]],
    *,
    rubric: str,
    n_draws: int,
    judge_root: Path,
    instrument: str,
) -> dict:
    """Graded pv/registered judging of one arm's completions -> rate + spread.

    Rate = fraction of completions whose mean graded score > 50 (llm-judging
    graded-primary + binary companion); per-arm content-drop vs transport-loss
    split persisted (rule 24). Items use '-' separators (judge_graded raises
    on '__' in ids).
    """
    items = [
        (f"{tag}-q{qi:03d}-c{ci:03d}", q, comp)
        for qi, q in enumerate(qs)
        for ci, comp in enumerate(comps[qi])
    ]
    result = judge_graded(
        items,
        rubric,
        n_draws=n_draws,
        cache_dir=judge_root / f"judge_cache_{instrument}",
        save_raw=judge_root / f"judge_raw_{instrument}_{tag}.json",
        judge_model=BEHAVIORS[cells.BEHAVIOR].judge_model,
        max_tokens=fu3w.JUDGE_MAX_TOKENS,
    )
    scores = [result.scores.get(iid) for iid, _, _ in items]
    scored = [s for s in scores if s is not None]
    n_pos = sum(1 for s in scored if s > 50)
    drop_frac = 1.0 - (len(scored) / len(items)) if items else 0.0
    rec = {
        "tag": tag,
        "instrument": instrument,
        "n_items": len(items),
        "n_scored": len(scored),
        "k_positive": n_pos,
        "rate": (n_pos / len(scored)) if scored else None,
        "graded_mean": (sum(scored) / len(scored)) if scored else None,
        "wilson_95": list(cells.wilson(n_pos, len(scored))) if scored else None,
        "item_drop_frac": drop_frac,
        "n_dropped_draws_content": getattr(result, "n_dropped_draws", None),
        "n_transport_lost_draws": getattr(result, "n_transport_lost_draws", None),
        "drop_flag_over_bar": bool(drop_frac > JUDGE_DROP_FLAG_BAR),
    }
    if rec["drop_flag_over_bar"]:
        logger.warning(
            "[i1434-judge] %s: item drop fraction %.3f > %.2f — FLAGGED (rule 23 "
            "truncation/transport check for the analyzer; verdicts still computed)",
            tag,
            drop_frac,
            JUDGE_DROP_FLAG_BAR,
        )
    return rec


def _stage_if_missing(local: Path, hub_prefix: str) -> Path:
    """Local file wins (same-machine smoke); else stage the file from HF."""
    if local.exists():
        return local
    hub.stage_hub_file(
        run1090.HF_DATA_REPO, f"{hub_prefix}/{local.name}", local, repo_type="dataset"
    )
    return local


def phase_judge_analyze(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:  # noqa: C901 — the P10 phase chain (mirrors fu4 cmd_judge_aggregate)
    """P10: pv judging of Tier-2 + base + panel, the §3 lattice, leakage, the
    registered-rubric parity re-read, and the committed aggregates."""
    run1090._phase("i1434_judge_analyze")
    qs = _eval_questions(cfg)
    cell_keys = resolve_cell_keys(args.cells, cfg.smoke)
    deliver = cfg.out_root / "deliverables" if cfg.smoke else cells.DELIVERABLES_DIR_1434
    deliver.mkdir(parents=True, exist_ok=True)
    judge_root = cfg.out_root / "judge"
    pv_rubric = cells.pv_rubric_text()
    registered_rubric = BEHAVIORS[cells.BEHAVIOR].judge_rubric
    run_ids = [r.run_id for r in cells.I1434_RUNS if r.cell_key in cell_keys]

    # 1. Per-run build records (ladders + selections; local wins, HF fallback).
    ladders: dict[str, dict] = {}
    selections: dict[str, dict] = {}
    for run_id in run_ids:
        local = cfg.out_root / run_id / "i1434_build_result.json"
        path = _stage_if_missing(local, f"{cells.DATA_PREFIX_1434}/{run_id}")
        rec = run1090._read_json(path)
        ladders[run_id] = {
            "status": rec.get("status"),
            "lr": rec.get("lr"),
            "rates_by_step": rec.get("rates_by_step"),
            "degeneracy_by_step": rec.get("degeneracy_by_step"),
            "selection": rec.get("selection"),
            "band": rec.get("band"),
        }
        if rec.get("status") == "trained":
            selections[run_id] = rec["selection"]

    # 2. Per-context verdict arms (the §3 pre-registered rule).
    verdict_arms: dict[str, dict] = {}
    for cell_key in cell_keys:
        arm_sels = {
            rid: s
            for rid, s in selections.items()
            if cells.RUN_BY_ID_1434[rid].cell_key == cell_key
        }
        if not arm_sels:
            verdict_arms[cell_key] = {"rule": "no_arms", "run_id": None}
            continue
        if len(arm_sels) == 3:
            _, rec = cells.verdict_arm_for_context(cell_key, arm_sels)
        else:
            _, rec = _verdict_from_partial(cell_key, arm_sels)
        verdict_arms[cell_key] = rec

    # 3. Tier-2 judging: verdict arms (trained) + per-context base, pv rubric.
    tier2: dict[str, dict] = {}
    for cell_key in cell_keys:
        ctx_id = cells.CONTEXT_BY_CELL_KEY[cell_key]
        run_id = verdict_arms[cell_key].get("run_id")
        base_local = _stage_if_missing(
            cfg.out_root / "base_arms" / cell_key / "tier2" / f"completions__base__{ctx_id}.json",
            f"{cells.DATA_PREFIX_1434}/raw_completions/base_arms/{cell_key}/tier2",
        )
        base_rec = _judge_rate_graded(
            f"t2-base-{cell_key}",
            qs,
            _completions_payload(base_local),
            rubric=pv_rubric,
            n_draws=cfg.tier2_draws,
            judge_root=judge_root,
            instrument="pv",
        )
        entry: dict[str, Any] = {"base": base_rec, "verdict_arm": verdict_arms[cell_key]}
        if run_id is not None:
            trained_local = _stage_if_missing(
                cfg.out_root / run_id / "tier2" / f"completions__trained__{ctx_id}.json",
                f"{cells.DATA_PREFIX_1434}/raw_completions/tier2/{run_id}",
            )
            trained_rec = _judge_rate_graded(
                f"t2-trained-{run_id}",
                qs,
                _completions_payload(trained_local),
                rubric=pv_rubric,
                n_draws=cfg.tier2_draws,
                judge_root=judge_root,
                instrument="pv",
            )
            entry["trained"] = trained_rec
            q_band = (trained_rec["rate"] or 0.0) - fu4.JUDGED_RATE_BAND[0]
            delta_ci = cells.newcombe(
                trained_rec["k_positive"],
                trained_rec["n_scored"],
                base_rec["k_positive"],
                base_rec["n_scored"],
            )
            entry["q_band"] = q_band
            entry["delta"] = (trained_rec["rate"] or 0.0) - (base_rec["rate"] or 0.0)
            entry["delta_newcombe_95"] = list(delta_ci)
            entry["lattice_verdict"] = cells.lattice_verdict(q_band, delta_ci)
        tier2[cell_key] = entry

    # 4. Panel judging (verdict arms + shared base) -> leakage.
    panel: dict[str, dict] = {}
    panel_ctx_ids = [c.context_id for c in fu3w.bystander_panel(cells.BEHAVIOR)]
    base_panel_rates: dict[str, dict] = {}
    for ctx_id in panel_ctx_ids:
        base_local = _stage_if_missing(
            cfg.out_root / "base_arms" / "panel" / f"completions__base__{ctx_id}.json",
            f"{cells.DATA_PREFIX_1434}/raw_completions/base_arms/panel",
        )
        base_panel_rates[ctx_id] = _judge_rate_graded(
            f"pn-base-{ctx_id}",
            qs,
            _completions_payload(base_local),
            rubric=pv_rubric,
            n_draws=3 if not cfg.smoke else 2,
            judge_root=judge_root,
            instrument="pv",
        )
    for cell_key in cell_keys:
        run_id = verdict_arms[cell_key].get("run_id")
        if run_id is None:
            continue
        source_ctx = cells.CONTEXT_BY_CELL_KEY[cell_key]
        rows = {}
        deltas = []
        for ctx_id in panel_ctx_ids:
            trained_local = _stage_if_missing(
                cfg.out_root / "panel" / run_id / f"completions__trained__{ctx_id}.json",
                f"{cells.DATA_PREFIX_1434}/raw_completions/panel/{run_id}",
            )
            trained_rec = _judge_rate_graded(
                f"pn-{run_id}-{ctx_id}",
                qs,
                _completions_payload(trained_local),
                rubric=pv_rubric,
                n_draws=3 if not cfg.smoke else 2,
                judge_root=judge_root,
                instrument="pv",
            )
            delta = (trained_rec["rate"] or 0.0) - (base_panel_rates[ctx_id]["rate"] or 0.0)
            rows[ctx_id] = {
                "trained": trained_rec,
                "base": base_panel_rates[ctx_id],
                "delta": delta,
                "is_source_context": ctx_id == source_ctx,
            }
            if ctx_id != source_ctx:
                deltas.append(delta)
        panel[cell_key] = {
            "run_id": run_id,
            "contexts": rows,
            "leakage_mean_nonsource_delta": (sum(deltas) / len(deltas)) if deltas else None,
        }

    # 5. Registered-rubric parity re-read (instrument-change control) on the
    #    SAME Tier-2 completions, separate rubric-keyed cache.
    parity: dict[str, dict] = {}
    for cell_key in cell_keys:
        run_id = verdict_arms[cell_key].get("run_id")
        ctx_id = cells.CONTEXT_BY_CELL_KEY[cell_key]
        entry = {}
        base_local = (
            cfg.out_root / "base_arms" / cell_key / "tier2" / f"completions__base__{ctx_id}.json"
        )
        entry["base"] = _judge_rate_graded(
            f"pr-base-{cell_key}",
            qs,
            _completions_payload(base_local),
            rubric=registered_rubric,
            n_draws=cfg.tier2_draws,
            judge_root=judge_root,
            instrument="registered",
        )
        if run_id is not None:
            trained_local = cfg.out_root / run_id / "tier2" / f"completions__trained__{ctx_id}.json"
            entry["trained"] = _judge_rate_graded(
                f"pr-trained-{run_id}",
                qs,
                _completions_payload(trained_local),
                rubric=registered_rubric,
                n_draws=cfg.tier2_draws,
                judge_root=judge_root,
                instrument="registered",
            )
        parity[cell_key] = entry

    # 6. Margin aggregates (computed pod-side by margin_fu4_run; copied here).
    margins: dict[str, Any] = {}
    for run_id in run_ids:
        local = cfg.out_root / run_id / "margin.json"
        if local.exists():
            margins[run_id] = run1090._read_json(local)

    aggregate = {
        "issue": cells.ISSUE_1434,
        "round": "writingstyle-pv-install",
        "band": list(fu4.JUDGED_RATE_BAND),
        "primary_instrument": "pv_writing_style_trait_score_v1 (verbatim, arXiv 2507.21509)",
        "pv_rubric_provenance": cells.load_pv_provenance(),
        "ladders": ladders,
        "verdict_arms": verdict_arms,
        "tier2": tier2,
        "panel": panel,
        "parity_reread": parity,
        "margins": margins,
        "smoke": cfg.smoke,
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    run1090._atomic_write_json(deliver / "i1434_ladders.json", aggregate)
    run1090._atomic_write_json(
        deliver / "selection.json",
        {"selections": selections, "verdict_arms": verdict_arms},
    )
    logger.info("[i1434-judge-analyze] wrote %s", deliver / "i1434_ladders.json")
    return 0


# ── entrypoint ───────────────────────────────────────────────────────────────


def _own_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="#1434 writing_style factory worker")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    p.add_argument(
        "--phase",
        required=True,
        choices=("questiongen", "datagen", "stage", "base-arms", "panel", "judge-analyze"),
    )
    p.add_argument("--cells", default=None, help="comma cell_key subset (smoke parity)")
    p.add_argument("--out-root", default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    cells.register_i1434_round()
    # fu4-native phases (dispatch / run) delegate VERBATIM to the round-
    # parametrized driver — the dispatcher's _worker_cmd routes subprocesses
    # back through THIS file (ROUND.worker_script), which re-registers the
    # round before fu4.main parses --round i1434.
    phase = None
    for i, tok in enumerate(argv):
        if tok == "--phase" and i + 1 < len(argv):
            phase = argv[i + 1]
            break
        if tok.startswith("--phase="):
            phase = tok.split("=", 1)[1]
            break
    if phase in FU4_DELEGATED_PHASES:
        if "--round" not in argv:
            argv = ["--round", "i1434", *argv]
        return fu4.main(argv)
    args = _own_parser().parse_args(argv)
    cfg = worker_config(args)
    logger.info(
        "issue1434_worker phase=%s smoke=%s cells=%s out_root=%s",
        args.phase,
        cfg.smoke,
        args.cells or "(all)" if not cfg.smoke else args.cells or "(smoke: ws-pers)",
        cfg.out_root,
    )
    if args.phase == "questiongen":
        import issue1434_questiongen as qg1434

        qg1434.run(force=False, cache_root=cfg.out_root / "questiongen_cache")
        return 0
    if args.phase == "datagen":
        phase_datagen(cfg, args)
        return 0
    if args.phase == "stage":
        return phase_stage(cfg, args)
    if args.phase == "base-arms":
        return phase_base_arms(cfg, args)
    if args.phase == "panel":
        return phase_panel(cfg, args)
    return phase_judge_analyze(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
