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

persona-dose-matched-regime round (plan v8, eval-only; run under
``--round i1434po``):
- ``--phase dose-select``        Q1 (VM, 0 GPU): deterministic dose-arm
  recompute + plan-pin asserts + static-panel snapshots ->
  ``dose_arm_selection.json`` (committed BEFORE dispatch).
- ``--phase dose-panel``         Q3 (pod, 1 GPU): the phase_panel loop over the
  2 dose-selected arms, checkpoints staged from the HF model repo.
- ``--phase dose-judge-analyze`` Q5 (VM, 0 GPU): Batch judging of the 2 new
  arms, D_hi/D_lo brackets vs the static committed sides, §3 lattice on D_hi
  only, deliverables + figures.

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
import subprocess  # noqa: E402
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
from explore_persona_space.artifacts.datagen import TopupSpec  # noqa: E402
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    ModelOrganism,
    _assemble_mix,
    _default_vllm_generate_fn,
    _generate_and_persist,
    _read_jsonl,
    _write_jsonl,
)
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1434.worker")

FU4_DELEGATED_PHASES = ("dispatch", "run")
JUDGE_DROP_FLAG_BAR = 0.10  # inherited flag check (llm-judging rule 23; flag, never kill)
I1434_FAMILY_ROUNDS = ("i1434", "i1434po")


def _ensure_family_round() -> None:
    """Every #1434 phase runs under an i1434-family round. ``main()`` selects
    it from ``--round``; a DIRECT phase caller (tests, ad-hoc imports) may
    inherit an out-of-family ambient fu4 ROUND (the module default is
    ``fu4``) — normalize that to the PARENT round, byte-preserving the
    pre-round-parametrization behavior (phases used to hardcode the parent
    names). An explicit i1434po selection is never overridden."""
    if fu4.ROUND.name not in I1434_FAMILY_ROUNDS:
        cells.register_i1434_round()
        fu4.set_round("i1434")


# ── config ───────────────────────────────────────────────────────────────────


def worker_config(args: argparse.Namespace) -> run1090.RunConfig:
    """The #1434 RunConfig (fu4_config shape, active round selected first)."""
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else (f"/tmp/issue-1434-{fu4.ROUND.name}-smoke" if smoke else "data/issue_1434/cells")
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


def skipped_cells_from_manifest(cfg: run1090.RunConfig) -> set[str]:
    """G1 yield-skipped cells from the stage manifest (out_root copy first,
    committed copy as the fallback; empty set when neither exists — e.g. a
    smoke chain that never ran stage). Consumed so downstream phases
    auto-exclude never-trained cells instead of crashing on their missing
    build results (the registered G1 drop path stays composable)."""
    for path in (
        cfg.out_root / fu4.ROUND.manifest_name,
        Path(fu4.ROUND.deliverables_dir) / fu4.ROUND.manifest_name,
    ):
        if path.exists():
            return set(run1090._read_json(path).get("skipped_cells_yield_floor") or [])
    return set()


def resolve_cell_keys(
    cells_arg: str | None, smoke: bool, cfg: run1090.RunConfig | None = None
) -> list[str]:
    """The ONE context-cell resolver every VM/pod phase consumes (smoke = the
    SAME path on the persona cell — the run-resolver's cell twin).

    With ``cfg`` given and no explicit ``--cells`` override, G1 yield-skipped
    cells from the stage manifest are auto-excluded (loud log). An explicit
    ``--cells`` list is honored verbatim (operator override wins).
    """
    ctx_map = cells.active_context_map()
    if cells_arg:
        keys = [t.strip() for t in cells_arg.split(",") if t.strip()]
        bad = [k for k in keys if k not in ctx_map]
        if bad:
            raise ValueError(f"bad #1434 cells {bad!r}: known {sorted(ctx_map)}")
        return keys
    keys = [cells.smoke_default_cell()] if smoke else list(cells.active_cell_keys())
    if cfg is not None:
        skipped = skipped_cells_from_manifest(cfg) & set(keys)
        if skipped:
            logger.warning(
                "[i1434] auto-excluding G1 yield-skipped cells (manifest "
                "skipped_cells_yield_floor): %s",
                sorted(skipped),
            )
            keys = [k for k in keys if k not in skipped]
    return keys


def _cell_shim(cell_key: str) -> run1090.Cell:
    """A run1090.Cell whose slug is the #1434 cell_key (distinct paths/runs)."""
    return run1090.Cell(
        cell_id=cell_key,
        behavior=cells.BEHAVIOR,
        generator="claude",
        trains=True,
        purpose=f"#1434 writing_style @ {cells.active_context_map()[cell_key]}",
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
    _ensure_family_round()
    if fu4.ROUND.name != "i1434":
        raise SystemExit(
            "--phase datagen is parent-round only: the i1434po round reuses the parent's "
            "judge-kept pools + generic rows VERBATIM (plan §4 D1') — run --phase mixes"
        )
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
        mult = (
            args.oversample_mult
            if args.oversample_mult is not None
            else (fu3w.BARE_OVERSAMPLE_MULT if ctx.kind == "bare" else fu3w.DEFAULT_OVERSAMPLE_MULT)
        )
        if summary_path.exists():
            # Parent _run_datagen_cell resume semantics + the D1 top-up lever:
            # a SUCCESS is always kept; a yield miss at the SAME budget that
            # already carried the top-up lever (topup_considered) skips — one
            # tranche per cell, misses stay recorded; a PRE-top-up miss at the
            # same budget re-enters ONCE (the manifest resume replays the
            # first sample from its raw/judge caches, then the single tranche
            # fires); a miss at a DIFFERENT budget quarantines the stale dir
            # (durable record) and regenerates at the new budget (the
            # registered retune lever).
            prior = run1090._read_json(summary_path)
            prior_mult = float(prior.get("oversample_mult", 0.0))
            if prior.get("status") == "success" or (
                prior_mult == mult and (prior.get("topup_considered") or "topup_record" in prior)
            ):
                out[cell_key] = prior
                logger.info("[i1434-datagen] %s already recorded — skip", cell_key)
                continue
            if prior_mult == mult:
                logger.info(
                    "[i1434-datagen] %s: pre-top-up yield miss at mult=%g — re-entering for "
                    "the single allowed top-up tranche (first sample resumes from cache)",
                    cell_key,
                    mult,
                )
            else:
                stale = cell_root / (
                    f"datagen_stale_x{prior_mult:g}_"
                    + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                )
                if (cell_root / "datagen").exists():
                    import os as _os

                    _os.replace(cell_root / "datagen", stale)
                summary_path.unlink()
                logger.warning(
                    "[i1434-datagen] %s floor miss at mult=%g — quarantined to %s; "
                    "regenerating at mult=%g",
                    cell_key,
                    prior_mult,
                    stale,
                    mult,
                )
        cell_cfg = dataclasses.replace(cfg, cells=(shim,), oversample_mult=mult)
        # Plan D1: the ONE 36-request near-miss top-up tranche (defaults:
        # tranche = ceil(target_n/EXPECTED_YIELD) = 36 at target 25, trigger =
        # kept < target). Armed identically in smoke + full (smoke IS sweep);
        # the yield DV stays frozen at the first sample (datagen.TopupSpec).
        topup_spec = TopupSpec()
        record: dict[str, Any] = {
            "cell_key": cell_key,
            "behavior": cells.BEHAVIOR,
            "context_id": ctx.context_id,
            "panel_name": panel_name,
            "oversample_mult": cell_cfg.oversample_mult,
            "target_n": cell_cfg.target_n,
            "seed": cfg.seed,
            "topup_considered": True,
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
                topup=topup_spec,
                **run1090._datagen_kwargs(cell_cfg, shim, None),
            )
        except run1090.DatagenYieldError as e:
            record.update(status="yield_floor_missed", reason=str(e))
            topup_record_path = cell_root / "datagen" / "topup_record.json"
            if topup_record_path.exists():
                # G1 miss AFTER the single allowed tranche — recorded
                # separately; the yield fields above stay first-sample-frozen.
                record["topup_record"] = run1090._read_json(topup_record_path)
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
        mix_dir.mkdir(parents=True, exist_ok=True)
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
        pool_meta = run1090._read_json(Path(meta_path))
        if pool_meta.get("topup"):
            # Rescued near-miss: the tranche is recorded separately; the yield
            # DV (pool_meta "positive" arm) stays frozen at the first sample.
            record["topup_record"] = pool_meta["topup"]
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


# ── D1' (i1434po): positive-only mixes = parent mix MINUS the negative panel ─


class PoMixIntegrityError(RuntimeError):
    """A D1' hard mix-integrity assert failed (routes filter -> rebuild -> STOP)."""


def _canon_row(row: dict) -> str:
    """Canonical serialization for row content-identity matching."""
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def _stage_po_parent_inputs(cfg: run1090.RunConfig, parent_cell: str) -> Path:
    """Stage the parent cell's frozen mix + datagen sidecars at the pin
    (idempotent; the SAME real staging path runs under --smoke — the
    cross-phase data-contract smoke consumes the producer's REAL shape)."""
    dest = Path(cfg.out_root) / "po_inputs" / parent_cell
    for rel in (
        "mix/train_mix.jsonl",
        "mix/mix_meta.json",
        "datagen/cn.jsonl",
        "datagen/pos.jsonl",
    ):
        hub.stage_hub_file(
            run1090.HF_DATA_REPO,
            f"{cells.DATA_PREFIX_1434}/{parent_cell}/{rel}",
            dest / rel,
            repo_type="dataset",
            revision=cells.DATA_REPO_PIN_1434,
        )
    return dest


def _po_filter_parent_mix(
    mix_rows: list[dict], cn_rows: list[dict], pos_rows: list[dict]
) -> tuple[list[dict], dict]:
    """PRIMARY D1' path: the parent mix minus its cn.jsonl-content rows,
    parent order preserved. Hard asserts: every cn row matched exactly once,
    zero panel-content rows remain, exactly 20 positives + 40 generic."""
    from collections import Counter

    if len(cn_rows) != 20 or len(pos_rows) != 20:
        raise PoMixIntegrityError(
            f"parent sidecars off-shape: cn={len(cn_rows)} pos={len(pos_rows)} != 20/20"
        )
    cn_counter = Counter(_canon_row(r) for r in cn_rows)
    cn_contents = set(cn_counter)
    kept: list[tuple[dict, str]] = []
    removed = 0
    for row in mix_rows:
        c = _canon_row(row)
        if cn_counter.get(c, 0) > 0:
            cn_counter[c] -= 1
            removed += 1
        else:
            kept.append((row, c))
    unmatched_cn = sum(cn_counter.values())
    if removed != 20 or unmatched_cn:
        raise PoMixIntegrityError(
            f"cn content match failed: removed {removed} rows, {unmatched_cn} cn rows unmatched"
        )
    if any(c in cn_contents for _, c in kept):
        raise PoMixIntegrityError(
            "panel content still present after the filter (duplicate cn content in the mix)"
        )
    pos_counter = Counter(_canon_row(r) for r in pos_rows)
    n_pos = 0
    for _, c in kept:
        if pos_counter.get(c, 0) > 0:
            pos_counter[c] -= 1
            n_pos += 1
    n_generic = len(kept) - n_pos
    if len(kept) != 60 or n_pos != 20 or n_generic != 40 or sum(pos_counter.values()):
        raise PoMixIntegrityError(
            f"po composition {n_pos} pos / {n_generic} generic / {len(kept)} total "
            f"(unmatched pos {sum(pos_counter.values())}) != 20/40/60"
        )
    return [r for r, _ in kept], {
        "method": "filter_parent_mix_minus_cn",
        "n_removed": removed,
        "n_pos": n_pos,
        "n_generic": n_generic,
        "order": "parent-mix order preserved",
    }


def _po_rebuild_from_sidecars(
    mix_rows: list[dict], pos_rows: list[dict], generic_corpus: list[dict], seed: int
) -> tuple[list[dict], dict]:
    """FALLBACK D1' path: rebuild pos.jsonl + the seeded generic sample
    (random.Random(seed).sample — _assemble_mix's FIRST rng use, so the draw
    reproduces the parent's exactly on the pinned corpus) and assert content
    equality with the parent mix's non-negative rows."""
    import random as _random
    from collections import Counter

    if len(generic_corpus) < 40:
        raise PoMixIntegrityError(
            f"generic corpus has {len(generic_corpus)} rows < 40 — cannot reproduce the "
            "parent's seeded sample"
        )
    rebuilt = list(pos_rows) + _random.Random(seed).sample(generic_corpus, 40)
    counter = Counter(_canon_row(r) for r in rebuilt)
    kept: list[dict] = []
    removed = 0
    for row in mix_rows:
        c = _canon_row(row)
        if counter.get(c, 0) > 0:
            counter[c] -= 1
            kept.append(row)
        else:
            removed += 1
    unmatched = sum(counter.values())
    if len(kept) != 60 or removed != 20 or unmatched:
        raise PoMixIntegrityError(
            f"rebuild content-equality failed: kept {len(kept)}, removed {removed}, "
            f"{unmatched} rebuilt rows unmatched in the parent mix"
        )
    return kept, {
        "method": "rebuild_pos_plus_seeded_generic",
        "seed": seed,
        "n_removed": removed,
        "n_pos": len(pos_rows),
        "n_generic": 40,
        "order": "parent-mix order preserved",
    }


def _derive_po_rows(
    cell_key: str,
    mix_rows: list[dict],
    cn_rows: list[dict],
    pos_rows: list[dict],
    generic_fn,
    seed: int,
) -> tuple[list[dict], dict]:
    """D1' derivation chain: filter path -> sidecar-rebuild fallback -> STOP
    loud (plan §7: a fresh-datagen fallback is a named must-ask deviation)."""
    try:
        return _po_filter_parent_mix(mix_rows, cn_rows, pos_rows)
    except PoMixIntegrityError as e:
        logger.warning(
            "[i1434po-mixes] %s: cn-filter path failed (%s) — sidecar rebuild fallback",
            cell_key,
            e,
        )
        try:
            return _po_rebuild_from_sidecars(mix_rows, pos_rows, generic_fn(), seed)
        except PoMixIntegrityError as e2:
            raise RuntimeError(
                f"[i1434po-mixes] {cell_key}: BOTH the cn-filter path AND the sidecar "
                f"rebuild failed content-equality — STOP (plan §7: fresh datagen would "
                f"break the identical-pools single-variable read; named plan deviation, "
                f"re-ask required). filter: {e}; rebuild: {e2}"
            ) from e2


def phase_mixes(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """D1' (VM, 0 GPU, no API): build + upload the per-cell 60-row
    positive-only mixes and write ``cell_manifest_i1434po.json`` (out_root
    copy always; committed copy under the po deliverables dir on full runs —
    the plan's commit-manifest-BEFORE-dispatch step)."""
    _ensure_family_round()
    if fu4.ROUND.name != "i1434po":
        raise SystemExit("--phase mixes is the i1434po builder — pass --round i1434po")
    run1090._phase("i1434po_mixes")
    parent_manifest = run1090._read_json(cells.DELIVERABLES_DIR_1434 / "cell_manifest_i1434.json")
    parent_pins = {r["cell_key"]: r["train_mix_sha256"] for r in parent_manifest["runs"]}
    upload = run1090._upload_fn(run1090.Seams1090())
    generic_corpus: list[dict] | None = None  # staged lazily (fallback path only)

    def _generic() -> list[dict]:
        nonlocal generic_corpus
        if generic_corpus is None:
            dest = Path(cfg.out_root) / "po_inputs" / "generic_corpus.jsonl"
            hub.stage_hub_file(
                run1090.HF_DATA_REPO,
                i1074.GENERIC_CORPUS_HF_PATH,
                dest,
                repo_type="dataset",
                revision=cells.DATA_REPO_PIN_1434,
            )
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            want = (parent_manifest.get("generic_corpus_provenance") or {}).get("staged_sha256")
            if want and sha != want:
                raise RuntimeError(
                    f"staged generic corpus sha {sha} != parent provenance pin {want}"
                )
            generic_corpus = _read_jsonl(dest)
        return generic_corpus

    from huggingface_hub import HfApi

    api = HfApi()
    runs: list[dict] = []
    derivations: dict[str, dict] = {}
    cell_keys = resolve_cell_keys(args.cells, cfg.smoke)
    for cell_key in cell_keys:
        parent_cell = cells.parent_cell_key(cell_key)
        src = _stage_po_parent_inputs(cfg, parent_cell)
        mix_path = src / "mix" / "train_mix.jsonl"
        parent_sha = hashlib.sha256(mix_path.read_bytes()).hexdigest()
        pin = parent_pins.get(parent_cell)
        if pin is None or parent_sha != pin:
            raise RuntimeError(
                f"[i1434po-mixes] {parent_cell}: staged parent mix sha {parent_sha} != "
                f"committed manifest pin {pin} at revision {cells.DATA_REPO_PIN_1434} — "
                "the frozen-mix reuse premise is broken; refusing to build"
            )
        po_rows, derivation = _derive_po_rows(
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
        parent_meta = run1090._read_json(src / "mix" / "mix_meta.json")
        derivation.update(
            {
                "parent_cell": parent_cell,
                "parent_mix_sha256": parent_sha,
                "revision_pin": cells.DATA_REPO_PIN_1434,
            }
        )
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
            pir = cells.mix_hub_prefix(cell_key)
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
                    raise RuntimeError(f"[i1434po-mixes] {pir}/{fname} missing on the data repo")
        for run in cells.I1434PO_RUNS:
            if run.cell_key == cell_key:
                runs.append(
                    {
                        "run_id": run.run_id,
                        "cell_key": cell_key,
                        "lr": run.lr,
                        "train_mix_sha256": po_sha,
                        "mix_hub_prefix": run.mix_hub_prefix,
                    }
                )
    manifest = {
        "issue": cells.ISSUE_1434,
        "round": fu4.ROUND.label,
        "runs": runs,
        # No po datagen: the parent's realized yield carries over verbatim.
        "skipped_cells_yield_floor": [
            f"ws-po-{k.removeprefix('ws-')}"
            for k in (parent_manifest.get("skipped_cells_yield_floor") or [])
        ],
        "generic_corpus_provenance": parent_manifest.get("generic_corpus_provenance"),
        "po_derivations": derivations,
        "parent_manifest_git_commit": parent_manifest.get("git_commit"),
        "revision_pin": cells.DATA_REPO_PIN_1434,
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = cfg.out_root / fu4.ROUND.manifest_name
    run1090._atomic_write_json(out_path, manifest)
    if not cfg.smoke:
        committed = Path(fu4.ROUND.deliverables_dir) / fu4.ROUND.manifest_name
        committed.parent.mkdir(parents=True, exist_ok=True)
        run1090._atomic_write_json(committed, manifest)
        logger.info(
            "[i1434po-mixes] committed manifest copy at %s (commit+push BEFORE dispatch)", committed
        )
    logger.info("[i1434po-mixes] %d runs pinned across %d cells", len(runs), len(cell_keys))
    return 0


def phase_stage(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Manifest build (per-run train_mix_sha256 pins) + HF mix verification.

    Writes ``cell_manifest_i1434.json`` under out_root; a full run ALSO writes
    the committed copy under eval_results/issue_1434/ (smoke never touches the
    committed path — scratch-redirect discipline).
    """
    _ensure_family_round()
    if fu4.ROUND.name != "i1434":
        raise SystemExit(
            "--phase stage is parent-round only: the i1434po manifest is written by "
            "--phase mixes (per-cell po train_mix_sha256 pins, plan §4 D1')"
        )
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
                    # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient (this call)
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
    return run1090.make_smoke_seams(
        dataclasses.replace(cfg, cells=(_cell_shim(cells.smoke_default_cell()),))
    )


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
    _ensure_family_round()
    if fu4.ROUND.name != "i1434":
        raise SystemExit(
            "--phase base-arms is parent-round only: the i1434po round REUSES the parent "
            "base Tier-2 arms + base panel row verbatim (plan §4 D3'; base model, contexts "
            "and rubric unchanged) — never regenerate them"
        )
    run1090._phase("i1434_base_arms")
    qs = _eval_questions(cfg)
    gen = _gen_fn(cfg)
    base_root = cfg.out_root / "base_arms"
    try:
        for cell_key in resolve_cell_keys(args.cells, cfg.smoke, cfg=cfg):
            ctx = cells.ensure_ws_context(cells.active_context_map()[cell_key])
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


def resolve_run_ids(cell_keys: list[str], smoke: bool) -> list[str]:
    """Run ids for the resolved cell subset, threaded through the SAME fu4 run
    resolver every dispatch phase uses (smoke = its one-run subset — the
    unified-subset threading duty; no phase re-enumerates the full grid)."""
    runs = fu4.resolve_fu4_runs(None, smoke)
    return [r.run_id for r in runs if r.cell_key in cell_keys]


def _run_selections(cfg: run1090.RunConfig, run_ids: list[str]) -> dict[str, dict]:
    """Per-run dose-selection records from the fu4 build results (fail-loud)."""
    sels: dict[str, dict] = {}
    for run_id in run_ids:
        path = cfg.out_root / run_id / f"{fu4.ROUND.name}_build_result.json"
        if not path.exists():
            raise RuntimeError(f"[i1434-panel] missing build result {path} — run dispatch first")
        rec = run1090._read_json(path)
        if rec.get("status") == "diverged":
            continue  # a K2-diverged arm carries no selection (recorded answer)
        sel = rec.get("selection")
        if sel is None:
            raise RuntimeError(
                f"[i1434-panel] {run_id}: status={rec.get('status')!r} build record has no "
                "'selection' (mid-ladder crash?) — re-run dispatch for this run, or pass "
                "--runs excluding it"
            )
        sels[run_id] = sel
    return sels


def phase_panel(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Bystander-panel generation for the per-context VERDICT arms (plan §3
    pre-registered selection rule) at their selected rungs."""
    _ensure_family_round()
    run1090._phase("i1434_panel")
    qs = _eval_questions(cfg)
    cell_keys = resolve_cell_keys(args.cells, cfg.smoke, cfg=cfg)
    run_ids = resolve_run_ids(cell_keys, cfg.smoke)
    selections = _run_selections(cfg, run_ids)
    verdicts: dict[str, dict] = {}
    gen = _gen_fn(cfg)
    panel_root = cfg.out_root / "panel"
    try:
        for cell_key in cell_keys:
            arm_sels = {
                rid: s
                for rid, s in selections.items()
                if cells.active_run_by_id()[rid].cell_key == cell_key
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
            build = run1090._read_json(
                cfg.out_root / run_id / f"{fu4.ROUND.name}_build_result.json"
            )
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
            f"{fu4.raw_completions_prefix()}/panel",
        )
        if not str(url):
            raise RuntimeError("panel upload returned no path — refusing silent loss")
    return 0


def _verdict_from_partial(cell_key: str, arm_sels: dict[str, dict]) -> tuple[str, dict]:
    """Verdict arm over the SURVIVING (non-diverged) arms — same registered
    rule, denominator honestly recorded (a diverged arm is an answer, not a
    silent hole)."""
    subset = {rid: s for rid, s in arm_sels.items() if rid in cells.active_run_by_id()}
    arms = sorted(
        (cells.active_run_by_id()[rid] for rid in subset),
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
    include_scores: bool = False,
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
    inst_root = judge_root / instrument  # plan §10 layout: judge/<instrument>/
    # Batch custom_id budget (#1415): the batch encoder appends 11 chars to a
    # 64-char API cap, so item ids must fit 53 chars. The po panel tags
    # (pn-ws-po-<ctx>-<lr>-<read_ctx>) run 3 chars past the parent's — which
    # sat at EXACTLY 53 — so hash-compact ONLY over-budget ids (every parent
    # id stays byte-identical -> cache continuity) and persist the id map.
    id_map = {
        iid: "h" + hashlib.sha1(iid.encode()).hexdigest()[:12]
        for iid, _, _ in items
        if len(iid) > 53
    }
    if id_map:
        items = [(id_map.get(iid, iid), q, comp) for iid, q, comp in items]
        inst_root.mkdir(parents=True, exist_ok=True)
        run1090._atomic_write_json(
            inst_root / f"idmap_{tag}.json", {v: k for k, v in id_map.items()}
        )
        logger.info(
            "[i1434-judge] %s: %d item ids hash-compacted for the Batch custom_id "
            "budget (map at %s)",
            tag,
            len(id_map),
            inst_root / f"idmap_{tag}.json",
        )
    result = judge_graded(
        items,
        rubric,
        n_draws=n_draws,
        cache_dir=inst_root / "cache",
        save_raw=inst_root / f"judge_raw_{instrument}_{tag}.json",
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
    if include_scores:
        # Per-item mean graded scores (kept draws) — the dose round's graded-
        # distribution companion (plan v8 §6 item: distributions under every
        # rate). Additive field; parent aggregates unchanged (no re-run).
        rec["scores"] = scored
    if rec["drop_flag_over_bar"]:
        logger.warning(
            "[i1434-judge] %s: item drop fraction %.3f > %.2f — FLAGGED (rule 23 "
            "truncation/transport check for the analyzer; verdicts still computed)",
            tag,
            drop_frac,
            JUDGE_DROP_FLAG_BAR,
        )
    return rec


def _tier2_lattice_fields(trained_rec: dict, base_rec: dict) -> dict:
    """The §3 lattice arithmetic for one tier2 cell — None-PROPAGATING.

    A rate of ``None`` means EVERY item of that arm was judge-dropped
    (drop-never-coerce, llm-judging rule 9): the lattice verdict is not
    computable from a coerced 0.0, so every derived field propagates None and
    the verdict reads ``not_computable_all_dropped`` (the arm is already
    ``drop_flag_over_bar``-flagged upstream).
    """
    if trained_rec.get("rate") is None or base_rec.get("rate") is None:
        logger.warning(
            "[i1434-judge] %s / %s: rate None (all items judge-dropped) — "
            "lattice verdict not computable; propagating None",
            trained_rec.get("tag"),
            base_rec.get("tag"),
        )
        return {
            "q_band": None,
            "delta": None,
            "delta_newcombe_95": None,
            "lattice_verdict": "not_computable_all_dropped",
        }
    q_band = trained_rec["rate"] - fu4.JUDGED_RATE_BAND[0]
    delta_ci = cells.newcombe(
        trained_rec["k_positive"],
        trained_rec["n_scored"],
        base_rec["k_positive"],
        base_rec["n_scored"],
    )
    return {
        "q_band": q_band,
        "delta": trained_rec["rate"] - base_rec["rate"],
        "delta_newcombe_95": list(delta_ci),
        "lattice_verdict": cells.lattice_verdict(q_band, delta_ci),
    }


def _stage_if_missing(local: Path, hub_prefix: str, *, revision: str | None = None) -> Path:
    """Local file wins (same-machine smoke); else stage the file from HF.
    ``revision`` pins parent-owned reuse artifacts to the parent-run data-repo
    revision (artifact-reuse checks (e)/(f); plan §10)."""
    if local.exists():
        return local
    hub.stage_hub_file(
        run1090.HF_DATA_REPO,
        f"{hub_prefix}/{local.name}",
        local,
        repo_type="dataset",
        revision=revision,
    )
    return local


# Plan §6 item 3: contexts whose two verdict rates differ by more than this
# (or where either verdict arm is closest-approach) carry the explicit
# dose-unmatched install-confound caveat (#601/#608).
DOSE_MATCH_MAX_GAP = 0.10


def _parent_aggregate() -> dict:
    """The parent (contrastive) round's COMMITTED aggregate — the CON side of
    every regime contrast (read-only; never regenerated)."""
    path = cells.DELIVERABLES_DIR_1434 / "i1434_ladders.json"
    if not path.exists():
        raise RuntimeError(
            f"[i1434po] parent aggregate missing at {path} — the regime contrast has no CON side"
        )
    return run1090._read_json(path)


def _pooled_nonsource_counts(
    panel_entry: dict, source_ctx: str, side: str = "trained"
) -> tuple[int, int, list[str]]:
    """(k_positive, n_scored, contexts used) pooled over the NON-source panel
    contexts; an all-dropped (rate None) arm is excluded, never coerced."""
    k = n = 0
    used: list[str] = []
    for ctx_id, row in (panel_entry.get("contexts") or {}).items():
        rec = row.get(side) or {}
        if ctx_id == source_ctx or rec.get("rate") is None:
            continue
        k += int(rec["k_positive"])
        n += int(rec["n_scored"])
        used.append(ctx_id)
    return k, n, sorted(used)


def _regime_lattice(d: float | None, ci: tuple[float, float] | None) -> str:
    """The plan-§3 DISJOINT + exhaustive regime-leakage lattice."""
    if d is None or ci is None:
        return "not_computable"
    if d > 0 and ci[0] > 0:
        return "Broader-leakage"
    if ci[1] < 0:
        return "Narrower-leakage"
    return "Indistinguishable"


def _two_prop_contrast(po_rec: dict | None, con_rec: dict | None) -> dict:
    """D = p_po - p_con on two independent judged proportions + Newcombe 95%.
    None-propagating (drop-never-coerce): an absent / all-dropped arm yields
    status not_computable."""
    if not po_rec or not con_rec or po_rec.get("rate") is None or con_rec.get("rate") is None:
        return {"status": "not_computable", "D": None, "newcombe_95": None}
    d = float(po_rec["rate"]) - float(con_rec["rate"])
    ci = cells.newcombe(
        int(po_rec["k_positive"]),
        int(po_rec["n_scored"]),
        int(con_rec["k_positive"]),
        int(con_rec["n_scored"]),
    )
    return {
        "status": "computed",
        "D": d,
        "newcombe_95": list(ci),
        "po": {k: po_rec.get(k) for k in ("rate", "k_positive", "n_scored", "graded_mean")},
        "con": {k: con_rec.get(k) for k in ("rate", "k_positive", "n_scored", "graded_mean")},
    }


def regime_contrast(po_agg: dict, con_agg: dict, cell_keys: list[str]) -> dict:
    """Plan §6 regime-comparison reads: per-context pooled non-source leakage
    D (trained-vs-trained — the shared base panel term cancels), the 20-cell
    per-(training ctx x read ctx) companion, the Tier-2 install contrast, and
    the §3 lattice + dose-matched labels."""
    out: dict[str, Any] = {
        "issue": cells.ISSUE_1434,
        "round": fu4.ROUND.label,
        "band": list(fu4.JUDGED_RATE_BAND),
        "dose_match_max_gap": DOSE_MATCH_MAX_GAP,
        "contexts": {},
        "cells": [],
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for cell_key in cell_keys:
        parent_cell = cells.parent_cell_key(cell_key)
        source_ctx = cells.active_context_map()[cell_key]
        po_panel = (po_agg.get("panel") or {}).get(cell_key)
        con_panel = (con_agg.get("panel") or {}).get(parent_cell)
        entry: dict[str, Any] = {
            "parent_cell": parent_cell,
            "source_ctx": source_ctx,
            "po_run_id": (po_panel or {}).get("run_id"),
            "con_run_id": (con_panel or {}).get("run_id"),
        }
        if po_panel is None or con_panel is None:
            entry["status"] = "missing_panel_arm"
            out["contexts"][cell_key] = entry
            continue
        # Pooled non-source leakage D (the §3 registered headline).
        k_po, n_po, ctxs_po = _pooled_nonsource_counts(po_panel, source_ctx)
        k_con, n_con, ctxs_con = _pooled_nonsource_counts(con_panel, source_ctx)
        # Pooled shared base (display denominators for the hero delta bars —
        # the D statistic itself is trained-vs-trained, base cancels).
        k_b, n_b, ctxs_b = _pooled_nonsource_counts(po_panel, source_ctx, side="base")
        if n_po > 0 and n_con > 0:
            d = k_po / n_po - k_con / n_con
            ci = cells.newcombe(k_po, n_po, k_con, n_con)
            entry["pooled"] = {
                "status": "computed",
                "D": d,
                "newcombe_95": list(ci),
                "lattice": _regime_lattice(d, ci),
                "po": {"k": k_po, "n": n_po, "rate": k_po / n_po, "contexts": ctxs_po},
                "con": {"k": k_con, "n": n_con, "rate": k_con / n_con, "contexts": ctxs_con},
            }
            if n_b > 0:
                entry["pooled"]["base"] = {
                    "k": k_b,
                    "n": n_b,
                    "rate": k_b / n_b,
                    "contexts": ctxs_b,
                }
                entry["pooled"]["delta_po_vs_base"] = {
                    "delta": k_po / n_po - k_b / n_b,
                    "newcombe_95": list(cells.newcombe(k_po, n_po, k_b, n_b)),
                }
                entry["pooled"]["delta_con_vs_base"] = {
                    "delta": k_con / n_con - k_b / n_b,
                    "newcombe_95": list(cells.newcombe(k_con, n_con, k_b, n_b)),
                }
        else:
            entry["pooled"] = {"status": "not_computable", "lattice": "not_computable"}
        # 20-cell companion: per (training context x non-source read context).
        read_ctxs = sorted(
            (set(po_panel.get("contexts") or {}) | set(con_panel.get("contexts") or {}))
            - {source_ctx}
        )
        for ctx_id in read_ctxs:
            cell = _two_prop_contrast(
                (po_panel.get("contexts") or {}).get(ctx_id, {}).get("trained"),
                (con_panel.get("contexts") or {}).get(ctx_id, {}).get("trained"),
            )
            cell.update({"training_cell": cell_key, "read_ctx": ctx_id})
            out["cells"].append(cell)
        # Install contrast (fresh Tier-2 trained arms, two independent props).
        po_t2 = (po_agg.get("tier2") or {}).get(cell_key) or {}
        con_t2 = (con_agg.get("tier2") or {}).get(parent_cell) or {}
        entry["install_contrast"] = _two_prop_contrast(po_t2.get("trained"), con_t2.get("trained"))
        entry["po_install_lattice"] = {
            k: po_t2.get(k) for k in ("q_band", "delta", "delta_newcombe_95", "lattice_verdict")
        }
        # Dose-matched labels (plan §6 item 3: band-matching is the control).
        po_sel = ((po_agg.get("verdict_arms") or {}).get(cell_key) or {}).get("selection") or {}
        con_sel = ((con_agg.get("verdict_arms") or {}).get(parent_cell) or {}).get(
            "selection"
        ) or {}
        po_rate, con_rate = po_sel.get("rate"), con_sel.get("rate")
        entry["dose"] = {
            "po_selection": po_sel,
            "con_selection": con_sel,
            "dose_unmatched": bool(
                not po_sel.get("in_band")
                or not con_sel.get("in_band")
                or po_rate is None
                or con_rate is None
                or abs(float(po_rate) - float(con_rate)) > DOSE_MATCH_MAX_GAP
            ),
        }
        out["contexts"][cell_key] = entry
    return out


def phase_judge_analyze(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:  # noqa: C901 — the P10 phase chain (mirrors fu4 cmd_judge_aggregate)
    """P10: pv judging of Tier-2 + base + panel, the §3 lattice, leakage, the
    registered-rubric parity re-read, and the committed aggregates."""
    _ensure_family_round()
    run1090._phase("i1434_judge_analyze")
    po_round = fu4.ROUND.name == "i1434po"
    # Parent-owned reuse artifacts stage at the parent-run revision pin.
    reuse_rev = cells.DATA_REPO_PIN_1434 if po_round else None
    qs = _eval_questions(cfg)
    cell_keys = resolve_cell_keys(args.cells, cfg.smoke, cfg=cfg)
    deliver = cfg.out_root / "deliverables" if cfg.smoke else Path(fu4.ROUND.deliverables_dir)
    deliver.mkdir(parents=True, exist_ok=True)
    judge_root = cfg.out_root / "judge"
    pv_rubric = cells.pv_rubric_text()
    registered_rubric = BEHAVIORS[cells.BEHAVIOR].judge_rubric
    run_ids = resolve_run_ids(cell_keys, cfg.smoke)

    # 1. Per-run build records (ladders + selections; local wins, HF fallback).
    ladders: dict[str, dict] = {}
    selections: dict[str, dict] = {}
    for run_id in run_ids:
        local = cfg.out_root / run_id / f"{fu4.ROUND.name}_build_result.json"
        path = _stage_if_missing(local, f"{fu4.ROUND.data_prefix}/{run_id}")
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
            if cells.active_run_by_id()[rid].cell_key == cell_key
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
    # po round: the BASE arms are the PARENT's (reused verbatim, plan §4 D3')
    # — local path + hub prefix are PARENT-cell-keyed, staged at the pin.
    tier2: dict[str, dict] = {}
    for cell_key in cell_keys:
        ctx_id = cells.active_context_map()[cell_key]
        base_cell = cells.parent_cell_key(cell_key)
        run_id = verdict_arms[cell_key].get("run_id")
        base_local = _stage_if_missing(
            cfg.out_root / "base_arms" / base_cell / "tier2" / f"completions__base__{ctx_id}.json",
            f"{cells.DATA_PREFIX_1434}/raw_completions/base_arms/{base_cell}/tier2",
            revision=reuse_rev,
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
                f"{fu4.raw_completions_prefix()}/tier2/{run_id}",
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
            entry.update(_tier2_lattice_fields(trained_rec, base_rec))
        tier2[cell_key] = entry

    # 4. Panel judging (verdict arms + shared base) -> leakage.
    panel: dict[str, dict] = {}
    panel_ctx_ids = [c.context_id for c in fu3w.bystander_panel(cells.BEHAVIOR)]
    base_panel_rates: dict[str, dict] = {}
    for ctx_id in panel_ctx_ids:
        base_local = _stage_if_missing(
            cfg.out_root / "base_arms" / "panel" / f"completions__base__{ctx_id}.json",
            f"{cells.DATA_PREFIX_1434}/raw_completions/base_arms/panel",
            revision=reuse_rev,
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
        source_ctx = cells.active_context_map()[cell_key]
        rows = {}
        deltas = []
        for ctx_id in panel_ctx_ids:
            trained_local = _stage_if_missing(
                cfg.out_root / "panel" / run_id / f"completions__trained__{ctx_id}.json",
                f"{fu4.raw_completions_prefix()}/panel/{run_id}",
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
            t_rate = trained_rec["rate"]
            b_rate = base_panel_rates[ctx_id]["rate"]
            # None-propagation (drop-never-coerce): an all-dropped arm's delta
            # is None, excluded from the leakage mean — never a coerced 0.0.
            delta = (t_rate - b_rate) if (t_rate is not None and b_rate is not None) else None
            if delta is None:
                logger.warning(
                    "[i1434-judge] panel %s@%s: rate None (all items judge-dropped) — "
                    "delta propagated as None",
                    run_id,
                    ctx_id,
                )
            rows[ctx_id] = {
                "trained": trained_rec,
                "base": base_panel_rates[ctx_id],
                "delta": delta,
                "is_source_context": ctx_id == source_ctx,
            }
            if ctx_id != source_ctx and delta is not None:
                deltas.append(delta)
        panel[cell_key] = {
            "run_id": run_id,
            "contexts": rows,
            "leakage_mean_nonsource_delta": (sum(deltas) / len(deltas)) if deltas else None,
        }

    # 5. Registered-rubric parity re-read (instrument-change control) on the
    #    SAME Tier-2 completions, separate rubric-keyed cache. PARENT ROUND
    #    ONLY — the po round drops it (plan §4 D2 item 4: instrument agreement
    #    already established by the parent's last result section).
    parity: dict[str, dict] = {}
    for cell_key in cell_keys if not po_round else ():
        run_id = verdict_arms[cell_key].get("run_id")
        ctx_id = cells.active_context_map()[cell_key]
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
        "round": fu4.ROUND.label,
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
    run1090._atomic_write_json(deliver / fu4.ROUND.ladders_name, aggregate)
    run1090._atomic_write_json(
        deliver / ("selection_po.json" if po_round else "selection.json"),
        {"selections": selections, "verdict_arms": verdict_arms},
    )
    logger.info("[i1434-judge-analyze] wrote %s", deliver / fu4.ROUND.ladders_name)
    if po_round:
        # Plan §6 regime reads: pooled + per-cell D, install contrasts, dose
        # labels, §3 lattice — trained-vs-trained vs the parent's committed
        # aggregate (the shared base panel term cancels in the delta-of-deltas).
        contrast = regime_contrast(aggregate, _parent_aggregate(), cell_keys)
        run1090._atomic_write_json(deliver / "regime_contrast.json", contrast)
        logger.info("[i1434-judge-analyze] wrote %s", deliver / "regime_contrast.json")
    if cfg.upload:
        # Plan §10: judge records (raw draws + rubric-keyed caches) persist to
        # issue1434_writingstyle/judge/<instrument>/ — text/JSON uploads
        # unconditionally (Upload Policy); one folder commit, fail-loud.
        url = hub._upload(
            judge_root,
            run1090.HF_DATA_REPO,
            "dataset",
            f"{cells.DATA_PREFIX_1434}/judge",
        )
        if not str(url):
            raise RuntimeError("judge records upload returned no path — refusing silent loss")
    return 0


# ── persona-dose-matched-regime round (plan v8): eval-only dose re-read ──────
# Q1 dose-select (VM, 0 GPU) -> Q3 dose-panel (pod, 1 GPU) -> Q5
# dose-judge-analyze (VM, 0 GPU). The ONLY manipulated variable vs the parent
# panels is WHICH checkpoints the persona-context leakage panel reads
# (dose-matched rungs instead of the band verdict rule); everything else is
# byte-inherited (contexts, n, temperature, seed, judge instrument).

DOSE_ROUND_LABEL = "persona-dose-matched-regime"
DOSE_TOLERANCE = 0.10  # the pre-registered scope tolerance (plan §2.2/§7)
# Plan §2.2 pins — Q1 recomputes and ASSERTS equality (any mismatch = the
# committed ladder data changed under us; fail loud, never warn).
DOSE_PLAN_PINS = {
    "con": {"run_id": "ws-pers-lr1e5", "step": 45, "tier1_rate": 0.86, "target": 0.81},
    "po": {"run_id": "ws-po-pers-lr3e5", "step": 10, "tier1_rate": 0.49, "target": 0.60},
}
# Plan §10 static-side pins (pooled non-source counts of the reused panels).
DOSE_STATIC_PINS = {"po25": (382, 500), "con25": (234, 489), "base": (1, 500)}
# Fitness check (a): recipe ground truth asserted on the fetched
# adapter_config.json (both subfolders Hub-verified at plan + implementation).
DOSE_ADAPTER_RECIPE_PIN = {
    "r": 32,
    "lora_alpha": 64,
    "use_rslora": True,
    "base_model_name_or_path": DEFAULT_BASE_MODEL,
}


def _git_file_pin(path: Path) -> dict:
    """`git log -1` pin + working-tree-clean staleness check for a committed
    source file (plan §4 Q1 + risk row 5). Raises on dirty/untracked."""
    rel = str(path.resolve().relative_to(cells.REPO_ROOT.resolve()))

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args, "--", rel],
            cwd=cells.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    head = _git("log", "-1", "--format=%H %cI")
    if not head:
        raise RuntimeError(f"[i1434-dose] {rel}: no git history — refusing an untracked source")
    if _git("status", "--porcelain"):
        raise RuntimeError(
            f"[i1434-dose] {rel}: uncommitted modifications — staleness check FAILED "
            "(commit or restore the aggregate before any dose phase)"
        )
    commit, date = head.split(" ", 1)
    return {"path": rel, "git_commit": commit, "git_commit_date": date}


def _dose_nearest_arm(ladders: dict, prefix: str, target: float, verdict_run_id: str) -> dict:
    """argmin |Tier-1 rate - target| over EVERY rung of the `<prefix>-lr*`
    ladders; tie-break (plan §11 row 2): same run as the context's verdict
    arm, then lower step. Deterministic (sorted runs, sorted int steps)."""
    best_key: tuple | None = None
    best: dict | None = None
    for run_id in sorted(k for k in ladders if k.startswith(f"{prefix}-lr")):
        rates = (ladders[run_id] or {}).get("rates_by_step") or {}
        for step in sorted(int(s) for s in rates):
            rate = float(rates[str(step)])
            gap = abs(rate - target)
            key = (round(gap, 9), 0 if run_id == verdict_run_id else 1, step)
            if best_key is None or key < best_key:
                best_key = key
                best = {"run_id": run_id, "step": step, "tier1_rate": rate, "gap": gap}
    if best is None:
        raise RuntimeError(f"[i1434-dose] no {prefix}-lr* ladder rates found — cannot select")
    return best


def _dose_nearest_per_run(ladders: dict, prefix: str, target: float) -> list[dict]:
    """Each `<prefix>-lr*` run's nearest rung to the target (the verbatim
    po-side infeasibility record, plan §2.2), sorted by gap."""
    rows: list[dict] = []
    for run_id in sorted(k for k in ladders if k.startswith(f"{prefix}-lr")):
        rates = (ladders[run_id] or {}).get("rates_by_step") or {}
        if not rates:
            continue
        step, rate = min(rates.items(), key=lambda kv: (abs(float(kv[1]) - target), int(kv[0])))
        rows.append(
            {
                "run_id": run_id,
                "step": int(step),
                "tier1_rate": float(rate),
                "gap": round(abs(float(rate) - target), 4),
            }
        )
    return sorted(rows, key=lambda r: (r["gap"], r["run_id"]))


def _assert_dose_pin(side: str, computed: dict, pin: dict) -> None:
    diffs = {k: (computed.get(k), pin[k]) for k in ("run_id", "step") if computed.get(k) != pin[k]}
    if abs(float(computed["tier1_rate"]) - float(pin["tier1_rate"])) > 1e-9:
        diffs["tier1_rate"] = (computed["tier1_rate"], pin["tier1_rate"])
    if diffs:
        raise RuntimeError(
            f"[i1434-dose] {side}-side selection mismatch vs plan §2.2 pin "
            f"(computed vs pinned): {diffs} — the committed ladder data changed; "
            "re-approve the plan before running the panel"
        )


def phase_dose_select(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Q1 (VM, 0 GPU, no API): deterministic dose-arm recompute + manifest.

    Recomputes argmin |Tier-1 rate - target| for target 0.81 over the three
    `ws-pers-*` ladders (con side) and target 0.60 over the three
    `ws-po-pers-*` ladders (po side), ASSERTS both equal the plan-pinned
    pair, snapshots the static reused-panel counts (git-pinned + dirty-
    checked), and writes ``dose_arm_selection.json``."""
    _ensure_family_round()
    run1090._phase("i1434_dose_select")
    del args
    con_path = cells.DELIVERABLES_DIR_1434 / "i1434_ladders.json"
    po_path = cells.PO_DELIVERABLES_DIR / "i1434po_ladders.json"
    contrast_path = cells.PO_DELIVERABLES_DIR / "regime_contrast.json"
    source_pins = {p.name: _git_file_pin(p) for p in (con_path, po_path, contrast_path)}
    con_agg = run1090._read_json(con_path)
    po_agg = run1090._read_json(po_path)
    parent_contrast = run1090._read_json(contrast_path)
    source_ctx = cells.CONTEXT_BY_CELL_KEY["ws-pers"]

    # Targets = the OPPOSITE regime's persona verdict Tier-1 rate (plan §2.2).
    con_verdict = con_agg["verdict_arms"]["ws-pers"]
    po_verdict = po_agg["verdict_arms"]["ws-po-pers"]
    con_target = float(po_verdict["selection"]["rate"])  # 0.81: match con TO po
    po_target = float(con_verdict["selection"]["rate"])  # 0.60: match po TO con
    if (
        abs(con_target - DOSE_PLAN_PINS["con"]["target"]) > 1e-9
        or abs(po_target - DOSE_PLAN_PINS["po"]["target"]) > 1e-9
    ):
        raise RuntimeError(
            f"[i1434-dose] verdict-rate targets drifted: con-side target {con_target} "
            f"(pin {DOSE_PLAN_PINS['con']['target']}), po-side target {po_target} "
            f"(pin {DOSE_PLAN_PINS['po']['target']}) — committed aggregates changed"
        )

    con_arm = _dose_nearest_arm(con_agg["ladders"], "ws-pers", con_target, con_verdict["run_id"])
    po_arm = _dose_nearest_arm(po_agg["ladders"], "ws-po-pers", po_target, po_verdict["run_id"])
    _assert_dose_pin("con", con_arm, DOSE_PLAN_PINS["con"])
    _assert_dose_pin("po", po_arm, DOSE_PLAN_PINS["po"])

    arms = []
    for side, arm, target, cell_key in (
        ("con", con_arm, con_target, "ws-pers"),
        ("po", po_arm, po_target, "ws-po-pers"),
    ):
        gap = float(arm["gap"])
        arms.append(
            {
                "label": f"{side}@{arm['step']}",
                "arm_dir": f"{arm['run_id']}-step{arm['step']}",
                "cell_key": cell_key,
                "side": side,
                "run_id": arm["run_id"],
                "step": arm["step"],
                "hub_subfolder": (
                    f"{cells.ADAPTER_PREFIX_1434}/{arm['run_id']}/checkpoint-{arm['step']}"
                ),
                "source_ctx": source_ctx,
                "tier1_rate": arm["tier1_rate"],
                "target": target,
                "gap": round(gap, 4),
                "tolerance_verdict": (
                    f"matched ({gap:.2f} <= {DOSE_TOLERANCE:.2f})"
                    if gap <= DOSE_TOLERANCE + 1e-12
                    else f"near-matched ({gap:.2f})"
                ),
            }
        )

    # Static reused-panel snapshots (plan §10) — asserted against the pins.
    k_po, n_po, _ = _pooled_nonsource_counts(po_agg["panel"]["ws-po-pers"], source_ctx)
    k_con, n_con, _ = _pooled_nonsource_counts(con_agg["panel"]["ws-pers"], source_ctx)
    k_b, n_b, _ = _pooled_nonsource_counts(po_agg["panel"]["ws-po-pers"], source_ctx, side="base")
    computed_counts = {"po25": (k_po, n_po), "con25": (k_con, n_con), "base": (k_b, n_b)}
    for name, pin in DOSE_STATIC_PINS.items():
        if computed_counts[name] != pin:
            raise RuntimeError(
                f"[i1434-dose] static panel counts drifted for {name}: committed "
                f"{computed_counts[name]} != plan pin {pin} — a later round rewrote "
                "the aggregates; re-approve before running"
            )

    def _src_read(panel_entry: dict) -> dict:
        rec = ((panel_entry.get("contexts") or {}).get(source_ctx) or {}).get("trained") or {}
        return {"k": rec.get("k_positive"), "n": rec.get("n_scored"), "rate": rec.get("rate")}

    static_arms = {
        "po25": {
            "run_id": po_verdict["run_id"],
            "step": int(po_verdict["selection"]["step"]),
            "tier1_rate": float(po_verdict["selection"]["rate"]),
            "pooled_nonsource": {"k": k_po, "n": n_po},
            "source_ctx_panel": _src_read(po_agg["panel"]["ws-po-pers"]),
            "source_file": source_pins[po_path.name]["path"],
        },
        "con25": {
            "run_id": con_verdict["run_id"],
            "step": int(con_verdict["selection"]["step"]),
            "tier1_rate": float(con_verdict["selection"]["rate"]),
            "pooled_nonsource": {"k": k_con, "n": n_con},
            "source_ctx_panel": _src_read(con_agg["panel"]["ws-pers"]),
            "source_file": source_pins[con_path.name]["path"],
        },
        "base": {
            "pooled_nonsource": {"k": k_b, "n": n_b},
            "source_file": source_pins[po_path.name]["path"],
        },
    }

    # Magnitude references for the Q5 contrast + hero figure (plan §6 read 3).
    pers_pooled = parent_contrast["contexts"]["ws-po-pers"]["pooled"]
    bare_pooled = parent_contrast["contexts"]["ws-po-bare"]["pooled"]
    references = {
        "unmatched_persona": {
            "D": pers_pooled["D"],
            "newcombe_95": pers_pooled["newcombe_95"],
            "source_file": source_pins[contrast_path.name]["path"],
        },
        "bare_matched": {
            "D": bare_pooled["D"],
            "newcombe_95": bare_pooled["newcombe_95"],
            "source_file": source_pins[contrast_path.name]["path"],
        },
    }

    out = {
        "issue": cells.ISSUE_1434,
        "round_label": DOSE_ROUND_LABEL,
        "selection_rule": (
            "argmin |Tier-1 rate - target| over every rung of the three same-side "
            "persona ladders; tie-break: same run as the context's verdict arm, "
            "then lower step (plan §11 row 2)"
        ),
        "tolerance": DOSE_TOLERANCE,
        "targets": {"con_side": con_target, "po_side": po_target},
        "arms": arms,
        "po_infeasibility": {
            "pre_registered_clause": (
                "re-select the po-persona checkpoint to the contrastive persona "
                "verdict rate 0.60 within 0.10; if no checkpoint sits within 0.10 "
                "of 0.60, report dose-matching infeasible at 5-step spacing "
                "(scope-caveat upgrade)"
            ),
            "target": po_target,
            "tolerance": DOSE_TOLERANCE,
            "nearest_per_run": _dose_nearest_per_run(po_agg["ladders"], "ws-po-pers", po_target),
            "verdict": (
                "INFEASIBLE within 0.10 at 5-step spacing (nearest gap 0.11) — the "
                "clause FIRES and is carried to the clean-result verbatim; "
                "resolution: two-point dose bracket (plan §2.2), the verdict binds "
                "to the con-side matched-high bracket only"
            ),
        },
        "static_arms": static_arms,
        "references": references,
        "source_pins": source_pins,
        "smoke": cfg.smoke,
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    deliver = (cfg.out_root / "deliverables") if cfg.smoke else cells.DOSE_DELIVERABLES_DIR
    deliver.mkdir(parents=True, exist_ok=True)
    run1090._atomic_write_json(deliver / "dose_arm_selection.json", out)
    logger.info("[i1434-dose-select] wrote %s", deliver / "dose_arm_selection.json")
    return 0


def _dose_selection(cfg: run1090.RunConfig) -> dict:
    """The Q1 selection manifest (scratch copy first for a self-contained
    smoke chain; the committed copy is the shared production source)."""
    for path in (
        cfg.out_root / "deliverables" / "dose_arm_selection.json",
        cells.DOSE_DELIVERABLES_DIR / "dose_arm_selection.json",
    ):
        if path.exists():
            return run1090._read_json(path)
    raise RuntimeError(
        "[i1434-dose] dose_arm_selection.json missing — run `--phase dose-select` "
        "on the VM and commit it BEFORE dispatch (plan §4 Q1)"
    )


def _assert_adapter_recipe(config_path: Path, hub_subfolder: str) -> None:
    """Fitness check (a): the fetched adapter_config.json must match the
    recipe pin (r=32, alpha=64, rsLoRA, Qwen2.5-7B-Instruct base)."""
    if not config_path.exists():
        raise RuntimeError(f"[i1434-dose] {hub_subfolder}: adapter_config.json not staged")
    cfgd = run1090._read_json(config_path)
    diffs = {
        k: (cfgd.get(k), want) for k, want in DOSE_ADAPTER_RECIPE_PIN.items() if cfgd.get(k) != want
    }
    if diffs:
        raise RuntimeError(
            f"[i1434-dose] {hub_subfolder}: adapter recipe mismatch (got vs pinned): "
            f"{diffs} — refusing to panel a wrong-recipe checkpoint (fitness check (a))"
        )


def _stage_dose_adapter(cfg: run1090.RunConfig, arm: dict) -> Path:
    """Stage one dose arm's adapter from the HF MODEL repo via a scoped
    snapshot (plan §4 Q3) + recipe assert. Under smoke only the config is
    fetched (same real snapshot_download path; weights unused by the
    tiny-Qwen gen seam)."""
    from huggingface_hub import snapshot_download

    sub = arm["hub_subfolder"]
    dest = cfg.out_root / "adapters"
    patterns = [f"{sub}/adapter_config.json"] if cfg.smoke else [f"{sub}/*"]
    hub.retry_transient(
        lambda: snapshot_download(
            repo_id=run1090.HF_MODEL_REPO, allow_patterns=patterns, local_dir=dest
        ),
        what=f"dose adapter snapshot {sub}",
    )
    local = dest / sub
    _assert_adapter_recipe(local / "adapter_config.json", sub)
    if not cfg.smoke and not (local / "adapter_model.safetensors").exists():
        raise RuntimeError(
            f"[i1434-dose] {local}: adapter_model.safetensors missing after the scoped "
            "snapshot — partial fetch; refusing to panel a config-only checkpoint"
        )
    return local


def phase_dose_panel(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Q3 (pod, 1 GPU): the SAME panel loop as ``phase_panel`` over the 2
    dose-selected arms — checkpoint resolution from ``dose_arm_selection.json``
    hub_subfolder fields via HF staging (NOT pod-local fu4 build records)."""
    _ensure_family_round()
    run1090._phase("i1434_dose_panel")
    del args
    sel = _dose_selection(cfg)
    arms = sel["arms"]
    qs = _eval_questions(cfg)
    gen = _gen_fn(cfg)
    panel_root = cfg.out_root / "panel"
    try:
        for arm in arms:
            ckpt = _stage_dose_adapter(cfg, arm)
            for bctx in fu3w.bystander_panel(cells.BEHAVIOR):
                _generate_and_persist(
                    gen,
                    "trained",
                    str(ckpt),
                    bctx,
                    qs,
                    n=cfg.tier1_n,
                    temperature=1.0,
                    out_dir=panel_root / arm["arm_dir"],
                    base_model=DEFAULT_BASE_MODEL,
                )
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()
    run1090._atomic_write_json(panel_root / "dose_panel_arms.json", {a["label"]: a for a in arms})
    if cfg.upload:
        url = hub._upload(
            panel_root,
            run1090.HF_DATA_REPO,
            "dataset",
            f"{cells.DATA_PREFIX_1434}/raw_completions/dose/panel",
        )
        if not str(url):
            raise RuntimeError("dose panel upload returned no path — refusing silent loss")
    return 0


def _dose_static_recheck(sel: dict) -> tuple[dict, dict]:
    """Q5 staleness re-check (risk row 5): re-pin + re-read the two committed
    aggregates and assert the pooled counts still equal the Q1 snapshot."""
    con_path = cells.DELIVERABLES_DIR_1434 / "i1434_ladders.json"
    po_path = cells.PO_DELIVERABLES_DIR / "i1434po_ladders.json"
    _git_file_pin(con_path)
    _git_file_pin(po_path)
    con_agg = run1090._read_json(con_path)
    po_agg = run1090._read_json(po_path)
    source_ctx = cells.CONTEXT_BY_CELL_KEY["ws-pers"]
    computed = {
        "po25": _pooled_nonsource_counts(po_agg["panel"]["ws-po-pers"], source_ctx)[:2],
        "con25": _pooled_nonsource_counts(con_agg["panel"]["ws-pers"], source_ctx)[:2],
        "base": _pooled_nonsource_counts(po_agg["panel"]["ws-po-pers"], source_ctx, "base")[:2],
    }
    for name, (k, n) in computed.items():
        snap = sel["static_arms"][name]["pooled_nonsource"]
        if (k, n) != (snap["k"], snap["n"]):
            raise RuntimeError(
                f"[i1434-dose] static arm {name} drifted since the Q1 snapshot: "
                f"committed ({k},{n}) != snapshot ({snap['k']},{snap['n']}) — a later "
                "round rewrote the aggregates; re-run dose-select"
            )
    return po_agg, con_agg


def _dose_bracket(
    po_side: dict | None, con_side: dict | None, *, po_name: str, con_name: str, verdict: bool
) -> dict:
    """One bracket D = p_po - p_con (pooled non-source) + Newcombe 95%.
    None/empty-propagating (drop-never-coerce). The §3 lattice binds to the
    verdict-bearing (high) bracket ONLY."""
    rec: dict[str, Any] = {"po_arm": po_name, "con_arm": con_name, "verdict_bearing": verdict}
    if not po_side or not con_side or not po_side.get("n") or not con_side.get("n"):
        rec.update({"status": "not_computable", "D": None, "newcombe_95": None})
        if verdict:
            rec["lattice"] = "not_computable"
        return rec
    k1, n1, k2, n2 = po_side["k"], po_side["n"], con_side["k"], con_side["n"]
    d = k1 / n1 - k2 / n2
    ci = cells.newcombe(k1, n1, k2, n2)
    rec.update(
        {
            "status": "computed",
            "D": d,
            "newcombe_95": list(ci),
            "po": {"k": k1, "n": n1, "rate": k1 / n1},
            "con": {"k": k2, "n": n2, "rate": k2 / n2},
        }
    )
    if verdict:
        rec["lattice"] = _regime_lattice(d, ci)
    else:
        rec["role"] = (
            "supporting-only — near-matched (0.11), po under-dosed; never verdict-bearing (plan §3)"
        )
    return rec


def _score_summary(scores: list[float]) -> dict | None:
    if not scores:
        return None
    s = sorted(scores)

    def q(p: float) -> float:
        return s[min(len(s) - 1, round(p * (len(s) - 1)))]

    return {
        "n": len(s),
        "mean": sum(s) / len(s),
        "quantiles": {"p0": q(0.0), "p25": q(0.25), "p50": q(0.5), "p75": q(0.75), "p100": q(1.0)},
    }


def phase_dose_judge_analyze(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Q5 (VM, 0 GPU): Batch judging of the 2 new dose arms + the D_hi/D_lo
    bracket recompute against the STATIC committed sides, the §3 lattice on
    D_hi only, deliverables + figures."""
    _ensure_family_round()
    run1090._phase("i1434_dose_judge_analyze")
    del args
    sel = _dose_selection(cfg)
    po_agg, con_agg = _dose_static_recheck(sel)
    qs = _eval_questions(cfg)
    deliver = (cfg.out_root / "deliverables") if cfg.smoke else cells.DOSE_DELIVERABLES_DIR
    deliver.mkdir(parents=True, exist_ok=True)
    judge_root = cfg.out_root / "judge"  # fresh round-scoped cache dir (rule 22/24)
    pv_rubric = cells.pv_rubric_text()
    source_ctx = cells.CONTEXT_BY_CELL_KEY["ws-pers"]
    panel_ctx_ids = [c.context_id for c in fu3w.bystander_panel(cells.BEHAVIOR)]

    arm_by_side = {a["side"]: a for a in sel["arms"]}
    dose_panel: dict[str, Any] = {}
    for arm in sel["arms"]:
        rows: dict[str, Any] = {}
        for ctx_id in panel_ctx_ids:
            local = _stage_if_missing(
                cfg.out_root / "panel" / arm["arm_dir"] / f"completions__trained__{ctx_id}.json",
                f"{cells.DATA_PREFIX_1434}/raw_completions/dose/panel/{arm['arm_dir']}",
            )
            rec = _judge_rate_graded(
                f"dn-{arm['arm_dir']}-{ctx_id}",
                qs,
                _completions_payload(local),
                rubric=pv_rubric,
                n_draws=2 if cfg.smoke else 3,
                judge_root=judge_root,
                instrument="pv",
                include_scores=True,
            )
            rows[ctx_id] = {"trained": rec, "is_source_context": ctx_id == arm["source_ctx"]}
        k, n, used = _pooled_nonsource_counts({"contexts": rows}, arm["source_ctx"])
        pooled_scores = [
            s
            for cid, row in rows.items()
            if cid != arm["source_ctx"]
            for s in (row["trained"].get("scores") or [])
        ]
        src_rec = (rows.get(arm["source_ctx"]) or {}).get("trained") or {}
        dose_panel[arm["label"]] = {
            "arm": arm,
            "contexts": rows,
            "pooled_nonsource": {
                "k": k,
                "n": n,
                "rate": (k / n) if n else None,
                "wilson_95": list(cells.wilson(k, n)) if n else None,
                "contexts": used,
            },
            "pooled_nonsource_scores": pooled_scores,
            "graded_summary": _score_summary(pooled_scores),
            "source_context_read": {
                "k": src_rec.get("k_positive"),
                "n": src_rec.get("n_scored"),
                "rate": src_rec.get("rate"),
                "tier1_selection_rate": arm["tier1_rate"],
            },
            "drop_report": {
                "n_items": sum(r["trained"]["n_items"] for r in rows.values()),
                "n_scored": sum(r["trained"]["n_scored"] for r in rows.values()),
                "n_dropped_draws_content": sum(
                    r["trained"].get("n_dropped_draws_content") or 0 for r in rows.values()
                ),
                "n_transport_lost_draws": sum(
                    r["trained"].get("n_transport_lost_draws") or 0 for r in rows.values()
                ),
            },
        }

    con_label = arm_by_side["con"]["label"]
    po_label = arm_by_side["po"]["label"]
    hi = _dose_bracket(
        sel["static_arms"]["po25"]["pooled_nonsource"],
        dose_panel[con_label]["pooled_nonsource"],
        po_name="po@25 (existing verdict arm)",
        con_name=f"{con_label} (new matched-high arm)",
        verdict=True,
    )
    lo = _dose_bracket(
        dose_panel[po_label]["pooled_nonsource"],
        sel["static_arms"]["con25"]["pooled_nonsource"],
        po_name=f"{po_label} (new near-matched-low arm)",
        con_name="con@25 (existing verdict arm)",
        verdict=False,
    )

    # 10-cell per-read-context companion (5 non-source contexts x 2 brackets).
    po25_ctxs = po_agg["panel"]["ws-po-pers"]["contexts"]
    con25_ctxs = con_agg["panel"]["ws-pers"]["contexts"]
    cells_rows: list[dict] = []
    for ctx_id in sorted(set(panel_ctx_ids) - {source_ctx}):
        hi_cell = _two_prop_contrast(
            (po25_ctxs.get(ctx_id) or {}).get("trained"),
            (dose_panel[con_label]["contexts"].get(ctx_id) or {}).get("trained"),
        )
        hi_cell.update({"bracket": "high", "read_ctx": ctx_id})
        cells_rows.append(hi_cell)
        lo_cell = _two_prop_contrast(
            (dose_panel[po_label]["contexts"].get(ctx_id) or {}).get("trained"),
            (con25_ctxs.get(ctx_id) or {}).get("trained"),
        )
        lo_cell.update({"bracket": "low", "read_ctx": ctx_id})
        cells_rows.append(lo_cell)

    panel_rungs = [
        {
            "run_id": sel["static_arms"]["con25"]["run_id"],
            "step": sel["static_arms"]["con25"]["step"],
            "rate": sel["static_arms"]["con25"]["tier1_rate"],
            "role": "con verdict arm (existing)",
        },
        {
            "run_id": arm_by_side["con"]["run_id"],
            "step": arm_by_side["con"]["step"],
            "rate": arm_by_side["con"]["tier1_rate"],
            "role": "con matched-high (new)",
        },
        {
            "run_id": sel["static_arms"]["po25"]["run_id"],
            "step": sel["static_arms"]["po25"]["step"],
            "rate": sel["static_arms"]["po25"]["tier1_rate"],
            "role": "po verdict arm (existing)",
        },
        {
            "run_id": arm_by_side["po"]["run_id"],
            "step": arm_by_side["po"]["step"],
            "rate": arm_by_side["po"]["tier1_rate"],
            "role": "po near-matched-low (new)",
        },
    ]
    source_context_reads = {
        "po@25 (existing)": {
            **sel["static_arms"]["po25"]["source_ctx_panel"],
            "tier1_selection_rate": sel["static_arms"]["po25"]["tier1_rate"],
        },
        "con@25 (existing)": {
            **sel["static_arms"]["con25"]["source_ctx_panel"],
            "tier1_selection_rate": sel["static_arms"]["con25"]["tier1_rate"],
        },
        f"{con_label} (new)": dose_panel[con_label]["source_context_read"],
        f"{po_label} (new)": dose_panel[po_label]["source_context_read"],
    }

    panel_payload = {
        "issue": cells.ISSUE_1434,
        "round_label": DOSE_ROUND_LABEL,
        "arms": dose_panel,
        "smoke": cfg.smoke,
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    contrast = {
        "issue": cells.ISSUE_1434,
        "round_label": DOSE_ROUND_LABEL,
        "tolerance": sel["tolerance"],
        "references": sel["references"],
        "static_arms": sel["static_arms"],
        "brackets": {"high": hi, "low": lo},
        "cells": cells_rows,
        "panel_rungs": panel_rungs,
        "source_context_reads": source_context_reads,
        "per_arm_drop_report": {lab: dose_panel[lab]["drop_report"] for lab in dose_panel},
        "graded_summaries": {lab: dose_panel[lab]["graded_summary"] for lab in dose_panel},
        "smoke": cfg.smoke,
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    run1090._atomic_write_json(deliver / "i1434dose_panel.json", panel_payload)
    run1090._atomic_write_json(deliver / "regime_contrast_dose_matched.json", contrast)
    logger.info(
        "[i1434-dose-judge-analyze] D_hi=%s %s lattice=%s | D_lo=%s %s — wrote %s",
        hi.get("D"),
        hi.get("newcombe_95"),
        hi.get("lattice"),
        lo.get("D"),
        lo.get("newcombe_95"),
        deliver / "regime_contrast_dose_matched.json",
    )

    # Figures (plan §6): hero + exploratory companions; smoke -> scratch dir.
    import issue1434_figures as figs

    fig_dir = (cfg.out_root / "figures") if cfg.smoke else cells.FIGURES_DIR_1434
    fig_dir.mkdir(parents=True, exist_ok=True)
    for p in figs.dose_figures(contrast, panel_payload, po_agg, con_agg, fig_dir):
        logger.info("[i1434-dose-judge-analyze] figure %s", p)

    if cfg.upload:
        url = hub._upload(
            judge_root,
            run1090.HF_DATA_REPO,
            "dataset",
            f"{cells.DATA_PREFIX_1434}/dose/judge",
        )
        if not str(url):
            raise RuntimeError("dose judge records upload returned no path — refusing loss")
    return 0


# ── icl-read-amplifier-specificity round (plan v11): unrelated-organism ─────
# control read. Q1 control-manifest (VM, 0 GPU, committed pre-dispatch) ->
# Q3 control-panel (pod, 1 GPU) -> Q4 control-judge-analyze (VM, 0 GPU).
# The ONLY manipulated variable vs the parent leakage panels is WHICH
# ORGANISM the 6-context read panel evaluates (an unrelated-behavior
# impolite #1090 organism instead of a casual one); everything else is
# byte-inherited (contexts, n, temperature, seed, judge instrument).

CTRL_ROUND_LABEL = "icl-read-amplifier-specificity"
CTRL_DELIVERABLES_DIR = cells.DELIVERABLES_DIR_1434 / CTRL_ROUND_LABEL
# Plan §2.2/§10 arm pins — plan-pinned, NO selector phase (the scope's
# subfolder pin fails artifact-reuse fitness (b) as a sole control and is
# demoted to a labeled inert reference; the installed fu4 lr-sweep arm is
# the verdict-bearing PRIMARY).
CTRL_ARMS: tuple[dict, ...] = (
    {
        "label": "imp-installed",
        "arm_dir": "imp-installed",
        "role": "primary",
        "hub_subfolder": "adapters/issue1090_fu4/imp-pers-lr3e5/checkpoint-30",
        "behavior": "impolite",
        "expected_tier2_impolite": 0.805,
    },
    {
        "label": "imp-inert",
        "arm_dir": "imp-inert",
        "role": "inert-reference",
        "hub_subfolder": "issue1090/c2-impolite-claude",
        "behavior": "impolite",
        "expected_install_delta": 0.0,
    },
)
# Plan §10 static pins (fail-loud drift asserts at manifest + judge time).
CTRL_BASE_PANEL_PIN = (0, 100)  # committed base k/n in EVERY read context
CTRL_CASUAL_ICL_PINS = {"ws-pers": 0.97, "ws-bare": 0.98, "ws-icl": 1.0, "ws-conv": 0.47}
CTRL_M_ANCHOR = 0.47  # WildChat minimum casual ICL delta (plan §3/§11)
CTRL_ENGAGEMENT_GATE = 0.50  # installed persona-cell impolite floor (plan §7)
CTRL_OVERLAP_FLAG = 0.30  # installed persona-cell casual-delta flag (plan §7)
CTRL_INSTALLED_SELECTION_PIN = {"step": 30, "rate": 0.81, "in_band": True}
CTRL_INSTALLED_TIER2_PIN = 0.805
CTRL_ICL_READ_CTX = f"icl_prefix_{cells.BEHAVIOR}"  # the verdict read context
CTRL_GEN_BASIS_COMPS_PER_GPU_H = 4000.0  # v8 P5' measured basis (plan §9 Q3 row)
_CTRL_FU4_LADDERS = (
    cells.REPO_ROOT / "eval_results" / "issue_1090" / "fu4-extended-dose-lr" / "fu4_ladders.json"
)
_CTRL_INERT_INSTALL = (
    cells.REPO_ROOT / "eval_results" / "issue_1090" / "install" / "c2-impolite-claude_install.json"
)


def _ctrl_committed_snapshots() -> tuple[dict, dict, dict]:
    """Re-read + git-pin the committed sources (base panel counts, casual ICL
    deltas, #1090 arm records) and ASSERT equality with the plan pins (any
    mismatch = the committed aggregates changed under us; fail loud, never
    warn — plan §4 Q1 + risk row 7). Returns (snapshot, source_pins, con_agg)."""
    con_path = cells.DELIVERABLES_DIR_1434 / "i1434_ladders.json"
    source_pins = {
        p.name: _git_file_pin(p) for p in (con_path, _CTRL_FU4_LADDERS, _CTRL_INERT_INSTALL)
    }
    con_agg = run1090._read_json(con_path)
    base_panel: dict[str, dict] = {}
    for ctx_id, entry in sorted(con_agg["panel"]["ws-pers"]["contexts"].items()):
        b = entry["base"]
        if (b["k_positive"], b["n_scored"]) != CTRL_BASE_PANEL_PIN:
            raise RuntimeError(
                f"[i1434-ctrl] committed base panel drifted at {ctx_id}: "
                f"({b['k_positive']},{b['n_scored']}) != plan pin {CTRL_BASE_PANEL_PIN} — "
                "a later round rewrote the aggregates; re-approve before running"
            )
        base_panel[ctx_id] = {"k": b["k_positive"], "n": b["n_scored"], "rate": b["rate"]}
    icl_deltas: dict[str, float] = {}
    for cell_key, pin in CTRL_CASUAL_ICL_PINS.items():
        delta = float(con_agg["panel"][cell_key]["contexts"][CTRL_ICL_READ_CTX]["delta"])
        if abs(delta - pin) > 1e-9:
            raise RuntimeError(
                f"[i1434-ctrl] casual ICL delta drifted for {cell_key}: committed "
                f"{delta} != plan pin {pin} — re-approve before running"
            )
        icl_deltas[cell_key] = delta
    fu4_ladders = run1090._read_json(_CTRL_FU4_LADDERS)
    sel = fu4_ladders["runs"]["imp-pers-lr3e5"]["selection"]
    sel_diffs = {
        k: (sel.get(k), want)
        for k, want in CTRL_INSTALLED_SELECTION_PIN.items()
        if sel.get(k) != want
    }
    tier2 = float(fu4_ladders["cells"]["imp-pers"]["tier2_confirm"]["imp-pers-lr3e5"])
    if sel_diffs or abs(tier2 - CTRL_INSTALLED_TIER2_PIN) > 1e-3:
        raise RuntimeError(
            f"[i1434-ctrl] installed-arm #1090 record drifted: selection diffs {sel_diffs}, "
            f"tier2_confirm {tier2} vs pin {CTRL_INSTALLED_TIER2_PIN} — the committed "
            "fu4_ladders.json changed; re-approve before running"
        )
    install = run1090._read_json(_CTRL_INERT_INSTALL)
    inert_delta = float(install["install_delta"])
    if abs(inert_delta - 0.0) > 1e-9:
        raise RuntimeError(
            f"[i1434-ctrl] inert-arm install_delta drifted: {inert_delta} != 0.0 — "
            "the scope-pinned adapter is no longer the zero-install reference"
        )
    snapshot = {
        "base_panel": base_panel,
        "casual_icl_deltas": icl_deltas,
        "installed_provenance": {
            "selection": sel,
            "tier2_confirm": tier2,
            "source_file": source_pins[_CTRL_FU4_LADDERS.name]["path"],
        },
        "inert_provenance": {
            "install_delta": inert_delta,
            "selection": install.get("selection"),
            "source_file": source_pins[_CTRL_INERT_INSTALL.name]["path"],
        },
    }
    return snapshot, source_pins, con_agg


def phase_control_manifest(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Q1 (VM, 0 GPU): plan-pinned arm manifest + committed-source snapshots
    (plan §4 Q1). Committed + pushed BEFORE dispatch; re-probes both Hub
    subfolders' adapter_config.json (plan §10)."""
    _ensure_family_round()
    run1090._phase("i1434_control_manifest")
    del args
    from huggingface_hub import HfApi

    api = HfApi()
    for arm in CTRL_ARMS:
        probe = f"{arm['hub_subfolder']}/adapter_config.json"
        exists = hub.retry_transient(
            # HUB_VERIFY_RETRY_EXEMPT: probe is wrapped in hub.retry_transient right here
            lambda p=probe: api.file_exists(run1090.HF_MODEL_REPO, p),
            what=f"ctrl arm probe {probe}",
        )
        if not exists:
            raise RuntimeError(
                f"[i1434-ctrl] {run1090.HF_MODEL_REPO}/{probe} missing on the Hub — "
                "refusing to pin an unfetchable arm (artifact-reuse check (c))"
            )
    snapshot, source_pins, _ = _ctrl_committed_snapshots()
    arms = []
    for arm in CTRL_ARMS:
        rec = dict(arm)
        rec["source_ctx"] = run1090.SOURCE_CONTEXT_ID
        rec["provenance"] = (
            snapshot["installed_provenance"]
            if arm["role"] == "primary"
            else snapshot["inert_provenance"]
        )
        arms.append(rec)
    out = {
        "issue": cells.ISSUE_1434,
        "round_label": CTRL_ROUND_LABEL,
        "arms": arms,
        "base_panel": snapshot["base_panel"],
        "casual_icl_deltas": snapshot["casual_icl_deltas"],
        "m_anchor": CTRL_M_ANCHOR,
        "engagement_gate": CTRL_ENGAGEMENT_GATE,
        "overlap_flag": CTRL_OVERLAP_FLAG,
        "icl_read_ctx": CTRL_ICL_READ_CTX,
        "source_pins": source_pins,
        "smoke": cfg.smoke,
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    deliver = (cfg.out_root / "deliverables") if cfg.smoke else CTRL_DELIVERABLES_DIR
    deliver.mkdir(parents=True, exist_ok=True)
    run1090._atomic_write_json(deliver / "control_arm_manifest.json", out)
    logger.info("[i1434-ctrl-manifest] wrote %s", deliver / "control_arm_manifest.json")
    return 0


def _control_manifest(cfg: run1090.RunConfig) -> dict:
    """The Q1 manifest (scratch copy first for a self-contained smoke chain;
    the committed copy is the shared production source)."""
    for path in (
        cfg.out_root / "deliverables" / "control_arm_manifest.json",
        CTRL_DELIVERABLES_DIR / "control_arm_manifest.json",
    ):
        if path.exists():
            return run1090._read_json(path)
    raise RuntimeError(
        "[i1434-ctrl] control_arm_manifest.json missing — run `--phase control-manifest` "
        "on the VM and commit it BEFORE dispatch (plan §4 Q1)"
    )


def phase_control_panel(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Q3 (pod, 1 GPU): the SAME panel loop as ``phase_dose_panel`` over the 2
    plan-pinned unrelated-behavior (impolite) control arms — HF staging via
    the dose round's scoped-snapshot + recipe-assert machinery, one shared
    vLLM engine, LoRA hot-swap, arms sequential (plan §4 Q3)."""
    _ensure_family_round()
    run1090._phase("i1434_control_panel")
    del args
    manifest = _control_manifest(cfg)
    arms = manifest["arms"]
    qs = _eval_questions(cfg)
    gen = _gen_fn(cfg)
    panel_root = cfg.out_root / "panel"
    try:
        from huggingface_hub import HfApi

        fetched_repo_sha = HfApi().repo_info(run1090.HF_MODEL_REPO).sha  # (g): sha at fetch
    except Exception:  # pod without Hub metadata access still generates; sha is best-effort
        logger.warning("[i1434-ctrl] repo_info sha probe failed — recording None")
        fetched_repo_sha = None
    try:
        for i, arm in enumerate(arms):
            t0 = time.monotonic()
            ckpt = _stage_dose_adapter(cfg, arm)  # scoped snapshot + recipe assert (fitness (a))
            for bctx in fu3w.bystander_panel(cells.BEHAVIOR):
                _generate_and_persist(
                    gen,
                    "trained",
                    str(ckpt),
                    bctx,
                    qs,
                    n=cfg.tier1_n,
                    temperature=1.0,
                    out_dir=panel_root / arm["arm_dir"],
                    base_model=DEFAULT_BASE_MODEL,
                )
            if i == 0 and not cfg.smoke:
                # Plan §7 pilot throughput read on the FIRST arm: log the
                # realized comps/GPU-h vs the v8 P5' basis; >2x deviation is
                # the orchestrator's epm:compute-deviation trigger (pod-side
                # code never posts markers — sentinel/log contract).
                wall_h = (time.monotonic() - t0) / 3600.0
                n_comps = len(qs) * cfg.tier1_n * len(fu3w.bystander_panel(cells.BEHAVIOR))
                realized = n_comps / wall_h if wall_h > 0 else float("inf")
                ratio = CTRL_GEN_BASIS_COMPS_PER_GPU_H / realized if realized else float("inf")
                logger.info(
                    "[i1434-ctrl] pilot throughput: %d comps in %.3f h -> %.0f comps/GPU-h "
                    "(basis %.0f; slowdown ratio %.2fx)",
                    n_comps,
                    wall_h,
                    realized,
                    CTRL_GEN_BASIS_COMPS_PER_GPU_H,
                    ratio,
                )
                if ratio > 2.0:
                    logger.warning(
                        "[i1434-ctrl] pilot throughput >2x below basis (%.2fx) — "
                        "orchestrator should post epm:compute-deviation (plan §7/§9)",
                        ratio,
                    )
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()
    run1090._atomic_write_json(
        panel_root / "control_panel_arms.json",
        {a["label"]: {**a, "fetched_repo_sha": fetched_repo_sha} for a in arms},
    )
    if cfg.upload:
        url = hub._upload(
            panel_root,
            run1090.HF_DATA_REPO,
            "dataset",
            f"{cells.DATA_PREFIX_1434}/raw_completions/ctrl/panel",
        )
        if not str(url):
            raise RuntimeError("control panel upload returned no path — refusing silent loss")
    return 0


def _ctrl_lattice(delta: float | None, ci: list | None, m: float | None) -> str:
    """The §3 DISJOINT + exhaustive control lattice (installed arm's ICL cell
    ONLY): Generic <=> M >= 0 AND CI excludes 0 positively; Partial <=> M < 0
    AND CI excludes 0 positively; Behavior-specific <=> otherwise. None
    propagates (drop-never-coerce)."""
    if delta is None or ci is None or m is None:
        return "not_computable_all_dropped"
    if ci[0] > 0 and m >= 0:
        return "Generic-ICL-restoration"
    if ci[0] > 0:
        return "Partial-generic-component"
    return "Behavior-specific-ICL-read"


def _ctrl_delta_vs_base(trained_rec: dict, base_rec: dict) -> dict:
    """One per-context trained-base delta against the COMMITTED base counts
    (the parent leakage-read convention, plan §2.3) — None-propagating."""
    if trained_rec.get("rate") is None or not base_rec.get("n"):
        return {"delta": None, "newcombe_95": None, "base": base_rec}
    ci = cells.newcombe(
        trained_rec["k_positive"], trained_rec["n_scored"], base_rec["k"], base_rec["n"]
    )
    return {
        "delta": trained_rec["rate"] - base_rec["rate"],
        "newcombe_95": list(ci),
        "base": base_rec,
    }


def _ctrl_engagement_gate(persona_impolite_rec: dict) -> dict:
    """Plan §7 adapter-engagement gate RECORD for the installed arm's
    persona-cell impolite-rubric rate (threshold 0.50; its #1090 record is
    0.805). The RAISE on a production miss lives in the caller (after the
    deliverables persist — durable record first, loud stop second)."""
    rate = persona_impolite_rec.get("rate")
    return {
        "threshold": CTRL_ENGAGEMENT_GATE,
        "rate": rate,
        "wilson_95": persona_impolite_rec.get("wilson_95"),
        "anchor_tier2": CTRL_INSTALLED_TIER2_PIN,
        "passed": bool(rate is not None and rate >= CTRL_ENGAGEMENT_GATE),
    }


def _ctrl_gate_stop(gate: dict) -> None:
    """The §7 engagement-gate STOP (wrong/dead adapter — no verdict). Raised
    AFTER the deliverables persist; unit-probeable (data-dependent-gate
    demonstration outside the main smoke leg)."""
    raise RuntimeError(
        f"[i1434-ctrl] adapter-engagement gate FAILED: installed persona-cell "
        f"impolite rate {gate.get('rate')} < {CTRL_ENGAGEMENT_GATE} (anchor "
        f"{CTRL_INSTALLED_TIER2_PIN}) — wrong/dead adapter; no verdict (plan §7)"
    )


def _ctrl_mechanism_read(icl_impolite_rec: dict, lattice: str) -> dict:
    """Plan §3/§7 behind-block mechanism read: the installed arm's ICL-cell
    impolite-rubric rate; a rate within noise of the 0.805 source anchor
    attaches the trained-behavior-amplification / rubric-overlap caveat to a
    Generic/Partial label (flag, not a gate)."""
    rate = icl_impolite_rec.get("rate")
    ci = icl_impolite_rec.get("wilson_95")
    within_noise = bool(
        rate is not None
        and ci is not None
        and (rate >= CTRL_INSTALLED_TIER2_PIN or ci[0] <= CTRL_INSTALLED_TIER2_PIN <= ci[1])
    )
    return {
        "icl_impolite_rate": rate,
        "wilson_95": ci,
        "anchor_tier2": CTRL_INSTALLED_TIER2_PIN,
        "within_anchor_noise": within_noise,
        "mechanism_caveat_attached": bool(
            within_noise and lattice in ("Generic-ICL-restoration", "Partial-generic-component")
        ),
    }


def phase_control_judge_analyze(cfg: run1090.RunConfig, args: argparse.Namespace) -> int:
    """Q4 (VM, 0 GPU): Batch judging of both control arms (casual pv rubric on
    all 12 cells + the #1090 impolite Tier-2 rubric on the 4 persona/ICL
    cells), the delta-vs-base reads, the §3 lattice + M, the §7 gates/flags,
    deliverables + figures (plan §4 Q4)."""
    _ensure_family_round()
    run1090._phase("i1434_control_judge_analyze")
    del args
    manifest = _control_manifest(cfg)
    # Staleness re-check (risk row 7): the committed sources must still match
    # the Q1 snapshot (same shape as _dose_static_recheck).
    snapshot, _, con_agg = _ctrl_committed_snapshots()
    if snapshot["base_panel"] != manifest["base_panel"] or (
        snapshot["casual_icl_deltas"] != manifest["casual_icl_deltas"]
    ):
        raise RuntimeError(
            "[i1434-ctrl] committed aggregates drifted since the Q1 manifest snapshot — "
            "re-run control-manifest and re-commit before judging"
        )
    qs = _eval_questions(cfg)
    deliver = (cfg.out_root / "deliverables") if cfg.smoke else CTRL_DELIVERABLES_DIR
    deliver.mkdir(parents=True, exist_ok=True)
    judge_root = cfg.out_root / "judge"  # fresh round-scoped cache dir (rule 22/24)
    pv_rubric = cells.pv_rubric_text()
    impolite_rubric = BEHAVIORS["impolite"].judge_rubric
    if not impolite_rubric:
        raise RuntimeError("[i1434-ctrl] impolite behavior registry has no judge_rubric")
    source_ctx = run1090.SOURCE_CONTEXT_ID
    panel_ctx_ids = [c.context_id for c in fu3w.bystander_panel(cells.BEHAVIOR)]
    dual_judge_ctxs = (source_ctx, CTRL_ICL_READ_CTX)  # plan §4 Q4(ii): 4 cells

    arm_recs: dict[str, Any] = {}
    for arm in manifest["arms"]:
        arm_dir = arm["arm_dir"]
        rows: dict[str, Any] = {}
        for ctx_id in panel_ctx_ids:
            local = _stage_if_missing(
                cfg.out_root / "panel" / arm_dir / f"completions__trained__{ctx_id}.json",
                f"{cells.DATA_PREFIX_1434}/raw_completions/ctrl/panel/{arm_dir}",
            )
            comps = _completions_payload(local)
            rec = _judge_rate_graded(
                f"cn-{arm_dir}-{ctx_id}",
                qs,
                comps,
                rubric=pv_rubric,
                n_draws=2 if cfg.smoke else 3,
                judge_root=judge_root,
                instrument="pv",
                include_scores=True,
            )
            row: dict[str, Any] = {
                "trained": rec,
                "is_source_context": ctx_id == source_ctx,
                **_ctrl_delta_vs_base(rec, manifest["base_panel"][ctx_id]),
            }
            if ctx_id in dual_judge_ctxs:
                row["impolite"] = _judge_rate_graded(
                    f"ci-{arm_dir}-{ctx_id}",
                    qs,
                    comps,
                    rubric=impolite_rubric,
                    n_draws=2 if cfg.smoke else 3,
                    judge_root=judge_root,
                    instrument="impolite",
                    include_scores=True,
                )
            rows[ctx_id] = row
        recs = [r["trained"] for r in rows.values()] + [
            r["impolite"] for r in rows.values() if "impolite" in r
        ]
        arm_recs[arm["label"]] = {
            "arm": arm,
            "contexts": rows,
            "drop_report": {
                "n_items": sum(r["n_items"] for r in recs),
                "n_scored": sum(r["n_scored"] for r in recs),
                "n_dropped_draws_content": sum(r.get("n_dropped_draws_content") or 0 for r in recs),
                "n_transport_lost_draws": sum(r.get("n_transport_lost_draws") or 0 for r in recs),
            },
        }

    installed = arm_recs["imp-installed"]["contexts"]
    inert = arm_recs["imp-inert"]["contexts"]
    icl_row = installed[CTRL_ICL_READ_CTX]
    delta_icl = icl_row["delta"]
    m = None if delta_icl is None else delta_icl - manifest["m_anchor"]
    lattice = _ctrl_lattice(delta_icl, icl_row["newcombe_95"], m)
    gate = _ctrl_engagement_gate(installed[source_ctx].get("impolite") or {})
    if not gate["passed"] and not cfg.smoke:
        lattice = "no_verdict_engagement_gate_failed"
    # §7 register-overlap flag: installed persona-cell CASUAL delta >= 0.30.
    overlap_delta = installed[source_ctx]["delta"]
    overlap = {
        "threshold": manifest["overlap_flag"],
        "persona_casual_delta": overlap_delta,
        "flagged": bool(overlap_delta is not None and overlap_delta >= manifest["overlap_flag"]),
    }
    mechanism = _ctrl_mechanism_read(icl_row.get("impolite") or {}, lattice)
    inert_consistency = {
        "persona_impolite_rate": (inert[source_ctx].get("impolite") or {}).get("rate"),
        "wilson_95": (inert[source_ctx].get("impolite") or {}).get("wilson_95"),
        "expected_install_delta": 0.0,
        "icl_impolite_rate": (inert[CTRL_ICL_READ_CTX].get("impolite") or {}).get("rate"),
    }

    panel_payload = {
        "issue": cells.ISSUE_1434,
        "round_label": CTRL_ROUND_LABEL,
        "arms": arm_recs,
        "base_panel": manifest["base_panel"],
        "smoke": cfg.smoke,
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    specificity = {
        "issue": cells.ISSUE_1434,
        "round_label": CTRL_ROUND_LABEL,
        "verdict_cell": {
            "arm": "imp-installed",
            "read_ctx": CTRL_ICL_READ_CTX,
            "delta_icl": delta_icl,
            "newcombe_95": icl_row["newcombe_95"],
            "m_anchor": manifest["m_anchor"],
            "M": m,
            "lattice": lattice,
        },
        "casual_icl_reference_band": manifest["casual_icl_deltas"],
        "engagement_gate": gate,
        "register_overlap_flag": overlap,
        "mechanism_read": mechanism,
        "inert_consistency": inert_consistency,
        "per_arm_drop_report": {lab: arm_recs[lab]["drop_report"] for lab in arm_recs},
        "smoke": cfg.smoke,
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    run1090._atomic_write_json(deliver / "control_panel.json", panel_payload)
    run1090._atomic_write_json(deliver / "icl_specificity.json", specificity)
    logger.info(
        "[i1434-ctrl-judge-analyze] delta_icl=%s %s M=%s lattice=%s gate=%s — wrote %s",
        delta_icl,
        icl_row["newcombe_95"],
        m,
        lattice,
        "PASS" if gate["passed"] else "FAIL",
        deliver / "icl_specificity.json",
    )

    if cfg.upload:
        url = hub._upload(
            judge_root,
            run1090.HF_DATA_REPO,
            "dataset",
            f"{cells.DATA_PREFIX_1434}/ctrl/judge",
        )
        if not str(url):
            raise RuntimeError("control judge records upload returned no path — refusing loss")
    if not gate["passed"] and not cfg.smoke:
        # Plan §7: a gate miss means the wrong/dead adapter was fetched —
        # STOP, investigate, no verdict (deliverables + judge records already
        # persisted above; the durable record precedes the loud stop). Under
        # smoke the tiny-Qwen stub's garbage completions make the gate
        # data-dependent-unsatisfiable, so the raise is production-only and
        # its branch is exercised by the _ctrl_gate_stop degenerate probe.
        _ctrl_gate_stop(gate)

    # Figures (plan §6): hero + exploratory companions; smoke -> scratch dir.
    import issue1434_figures as figs

    fig_dir = (cfg.out_root / "figures") if cfg.smoke else cells.FIGURES_DIR_1434
    fig_dir.mkdir(parents=True, exist_ok=True)
    for p in figs.ctrl_figures(specificity, panel_payload, con_agg, fig_dir):
        logger.info("[i1434-ctrl-judge-analyze] figure %s", p)
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
        choices=(
            "questiongen",
            "datagen",
            "mixes",
            "stage",
            "base-arms",
            "panel",
            "judge-analyze",
            "dose-select",
            "dose-panel",
            "dose-judge-analyze",
            "control-manifest",
            "control-panel",
            "control-judge-analyze",
        ),
    )
    p.add_argument(
        "--round",
        default="i1434",
        choices=("i1434", "i1434po", "i1434ctrl"),
        help="active round registry (i1434po = the positive-only regime arm; "
        "i1434ctrl = the icl-read-amplifier-specificity control round, which "
        "runs on the PARENT i1434 fu4 registry — eval-only, no runs of its own)",
    )
    p.add_argument("--cells", default=None, help="comma cell_key subset (smoke parity)")
    p.add_argument("--out-root", default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    p.add_argument(
        "--oversample-mult",
        type=float,
        default=None,
        help="datagen budget retune lever (plan §10 allowed deviation; default: "
        "2.5, bare 12.0 — the fu3 launch-4 grounding)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    cells.register_i1434_round()
    cells.register_i1434po_round()
    # fu4-native phases (dispatch / run) delegate VERBATIM to the round-
    # parametrized driver — the dispatcher's _worker_cmd routes subprocesses
    # back through THIS file (ROUND.worker_script), which re-registers the
    # round before fu4.main parses --round <name>.
    phase = None
    round_name = "i1434"
    for i, tok in enumerate(argv):
        if tok == "--phase" and i + 1 < len(argv):
            phase = argv[i + 1]
        elif tok.startswith("--phase="):
            phase = tok.split("=", 1)[1]
        if tok == "--round" and i + 1 < len(argv):
            round_name = argv[i + 1]
        elif tok.startswith("--round="):
            round_name = tok.split("=", 1)[1]
    # The ctrl round is eval-only over the PARENT registry: no fu4 RoundSpec
    # of its own (arms are plan-pinned HF subfolders, never ladder runs) —
    # the ROUND-parametrized helpers read the i1434 parent (plan v11 §2.1).
    fu4.set_round("i1434" if round_name == "i1434ctrl" else round_name)
    if phase in FU4_DELEGATED_PHASES:
        if not any(t == "--round" or t.startswith("--round=") for t in argv):
            argv = ["--round", round_name, *argv]
        return fu4.main(argv)
    args = _own_parser().parse_args(argv)
    if args.round == "i1434ctrl" and args.out_root is None:
        # Distinct default roots (plan §10 Q3/Q4 commands pass --out-root
        # data/issue_1434/ctrl explicitly; the smoke scratch dir must never
        # collide with a parent-round smoke chain).
        args.out_root = "/tmp/issue-1434-i1434ctrl-smoke" if args.smoke else "data/issue_1434/ctrl"
    cfg = worker_config(args)
    logger.info(
        "issue1434_worker round=%s phase=%s smoke=%s cells=%s out_root=%s",
        fu4.ROUND.name,
        args.phase,
        cfg.smoke,
        args.cells or ("(all)" if not cfg.smoke else f"(smoke: {cells.smoke_default_cell()})"),
        cfg.out_root,
    )
    if args.phase == "questiongen":
        import issue1434_questiongen as qg1434

        qg1434.run(force=False, cache_root=cfg.out_root / "questiongen_cache")
        return 0
    if args.phase == "datagen":
        phase_datagen(cfg, args)
        return 0
    handlers = {
        "mixes": phase_mixes,
        "stage": phase_stage,
        "base-arms": phase_base_arms,
        "panel": phase_panel,
        "judge-analyze": phase_judge_analyze,
        "dose-select": phase_dose_select,
        "dose-panel": phase_dose_panel,
        "dose-judge-analyze": phase_dose_judge_analyze,
        "control-manifest": phase_control_manifest,
        "control-panel": phase_control_panel,
        "control-judge-analyze": phase_control_judge_analyze,
    }
    return handlers[args.phase](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
