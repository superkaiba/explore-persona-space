#!/usr/bin/env python
"""#1947 P3-P5 driver: ladder judging + verdict selection (VM) and the
trained-rows battery captures + fits (pod GPU) — plan §4.4 P3/P4/P5.

Phases (``--phase``):

- ``probe``        (VM, 0 GPU): Hub existence probes for EVERY reused input at
  its recorded path (r_B tensors, corpus pfx sha set, #1768 base stores +
  bare-n refits, #1900 margin pools, the #1768 last-token re-run outputs) —
  fail-loud per-path report (plan §10 Phase-0 probes).
- ``judge``        (VM, Batch API): judge the unit-2 ladder rollouts per
  behavior tranche (Sonnet graded 0-100, 3 draws, max_tokens 300 via the
  registered instruments; resumable per (cell, rung); drop tally splits
  content vs transport per llm-judging rules 9/24).
- ``select``       (VM): earliest-in-band verdict selection over rates_by_step
  (band [0.60, 0.85], closest-approach fallback via the canonical
  ``recipe.select_dose_checkpoint``); marker cells consume the worker's
  programmatic slot-read selection verbatim → ``verdict_manifest.json``.
- ``capture-fit``  (pod GPU): per verdict arm — TF trained-rows tree
  (trained+base, consumed@verdict + full mix), on-policy tree, 20-q panel,
  TF fixed-pool margins; δ units via the REAL ``issue1768_capture
  .run_delta_unit``; per-rung dynamics (2 cells); bare-corpus capture +
  matched-text TF (12 con-s42 arms); then P5 fits via the REAL
  ``issue1768_fit.fit_bare_n_cell`` (n=3,000-matched, baselines attached).
- ``unit``         (internal): one capture/fit unit (subprocess fan-out
  target; CVD pinned by the dispatcher env).
- ``import-check`` : resolve every deferred import + exit 0.

BINDING last-token directive (task #1947 epm:progress v7): every context-side
capture records BOTH summaries from ONE forward pass — last-prompt-token
(``context_last`` / ``prefix_last``, the #779 convention) PRIMARY + span-mean
SECONDARY; answer-side stays span-mean. Implemented via the ``SPAN_ARMS_LAST``
1-token spans in ``analysis/representation_shift.py``.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1947_cells as cells  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1947.battery")

ISSUE = cells.ISSUE
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
OVERFLOW_MODEL_REPO = "superkaiba1/explore-persona-space-overflow"
DATA_PREFIX = cells.DATA_PREFIX  # issue1947_singlevisit
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
JUDGED_BAND = (0.60, 0.85)  # plan §7 lattice (== recipe.JUDGED_RATE_BAND)
JUDGE_N_DRAWS = 3  # plan §6
LAYERS = (14, 19, 25)  # plan §11 (#1768 comparability)
# Both context summaries in ONE forward pass (binding directive) + answer side.
CAPTURE_SPANS = ("prefix", "context", "response", "prefix_last", "context_last")
_JUDGE_ID_BUDGET = 53  # Batch custom_id budget (#1415)
# Per-phase disk floors on the pod out-root (plan §9 mount binding).
PHASE_HEADROOM_GB = {"capture": 40.0, "corpus": 40.0, "fit": 10.0}
SENTINEL_DIR_DEFAULT = Path("/workspace/logs")
# The 12 bare-corpus arms (plan §4.4: single-visit con-s42 content arms).
CORPUS_ARM_SLUGS = tuple(
    cells.content_slug(b, c, "con", "sv", 42) for b in cells.BEH_KEYS for c in cells.CTX_KEYS
)
DYNAMICS_SLUGS = ("syc-pers-con-sv-s42", "syc-conv-con-sv-s42")  # plan §4.4
BEH_FAM = {"sycophancy": "syc", "impolite": "imp", "writing_style": "cas"}
# The #1900-frozen per-family fixed +/- pools (syc = the #722 pool verbatim via
# build_fixed_pairs; cas/imp = the parent-round frozen pools) — probe-verified.
MARGIN_POOLS_PREFIX = "issue1900_leakrace/tfm/config"


def _atomic_json(path: Path, payload: dict) -> None:
    """Atomic JSON write (tmp + os.replace; tmp INSIDE the dest dir — EXDEV)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _meta() -> dict:
    return {
        "issue": ISSUE,
        "git_commit": _git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _phase(name: str, **kv) -> None:
    """The pod-side-reporting breadcrumb line."""
    extra = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[phase={name}]{' ' + extra if extra else ''}", flush=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class Cfg:
    """Battery run config; every output-affecting knob is part of the regime."""

    out_root: Path
    out_dir: Path  # analysis JSON dest (eval_results/issue_1947/analysis)
    smoke: bool = False
    stub_judge: bool = False  # smoke-only offline judge
    cells_filter: tuple[str, ...] = ()
    behaviors: tuple[str, ...] = ("sycophancy", "impolite", "writing_style")
    n_draws: int = JUDGE_N_DRAWS
    layers: tuple[int, ...] = LAYERS
    tf_batch: int = 8  # #1768 TF_BATCH_SIZE convention
    upload: bool = True
    gpu_id: int = 0  # informational; the launcher env CVD pin selects the GPU
    sentinel_dir: Path | None = None
    local_ladders: bool = False  # smoke: read ladders from out_root, no Hub staging
    model_override: str | None = None  # smoke-only tiny-model substitute (CAP.Cfg seam)

    def verdict_manifest_path(self) -> Path:
        return self.out_dir / "verdict_manifest.json"


def _content_cells(cfg: Cfg) -> list[cells.CellSpec]:
    out = []
    for c in cells.CELLS:
        if c.kind != "content":
            continue
        if cfg.behaviors and c.behavior not in cfg.behaviors:
            continue
        if cfg.cells_filter and c.slug not in cfg.cells_filter:
            continue
        out.append(c)
    return out


def _marker_cells(cfg: Cfg) -> list[cells.CellSpec]:
    out = [c for c in cells.CELLS if c.kind == "marker"]
    if cfg.cells_filter:
        out = [c for c in out if c.slug in cfg.cells_filter]
    return out


# ── Phase: probe (plan §10 Phase-0 reused-input probes) ─────────────────────


def _reused_input_table() -> list[dict]:
    """Every reused Hub input with its recorded path (plan §10). ``required``
    inputs fail the probe; optional ones are report-only (scope caveats)."""
    import issue1768_cells as X1768
    import issue1768_directions as DIRS

    rows: list[dict] = []
    for beh, p in sorted(DIRS.RB_HUB_PATHS.items()):
        rows.append({"name": f"rb_{beh}", "kind": "file", "path": p, "required": True})
    rows.append(
        {
            "name": "corpus_sample_pfx",
            "kind": "file",
            "path": f"{X1768.HF_PREFIX}/on_target/inputs/corpus_sample_pfx.json",
            "required": True,
        }
    )
    rows.append(
        {
            "name": "base_content_store",
            "kind": "file",
            "path": f"{X1768.HF_PREFIX}/corpus_capture/base_content/pooled.pt",
            "required": True,
        }
    )
    rows.append(
        {
            "name": "base_content_rows_spans",  # matched-text TF inputs (base text + spans)
            "kind": "file",
            "path": f"{X1768.HF_PREFIX}/corpus_capture/base_content/rows_spans.json",
            "required": True,
        }
    )
    rows.append(
        {
            "name": "fits_bare_n",  # r3 n=3,000 refits + floors (cross-check target)
            "kind": "prefix",
            "path": f"{X1768.HF_PREFIX}/on_target/eval_results/fits_bare_n",
            "required": True,
        }
    )
    rows.append(
        {
            "name": "lasttoken_ctx",  # directive probe: #1768 last-token re-run outputs
            "kind": "prefix",
            "path": f"{X1768.HF_PREFIX}/lasttoken_ctx",
            "required": False,
        }
    )
    for fam in ("syc", "imp", "cas"):
        rows.append(
            {
                "name": f"margin_pool_{fam}",
                "kind": "file",
                "path": f"{MARGIN_POOLS_PREFIX}/pools_{fam}.json",
                "required": True,
            }
        )
    return rows


def cmd_probe(cfg: Cfg) -> int:
    """Probe every reused input; write probe_report.json; non-zero on any
    REQUIRED miss (fail loud with a per-path report — plan §4.4 P0)."""
    from huggingface_hub import HfApi

    api = HfApi()
    report: list[dict] = []
    n_missing_required = 0
    for row in _reused_input_table():
        entry = dict(row)
        try:
            if row["kind"] == "file":
                ok = hub.retry_transient(
                    lambda p=row["path"]: api.file_exists(HF_DATA_REPO, p, repo_type="dataset"),
                    what=f"probe {row['name']}",
                )
                entry["resolved"] = bool(ok)
            elif row["kind"] == "prefix":
                files = hub.retry_transient(
                    lambda p=row["path"]: [
                        e.path
                        for e in api.list_repo_tree(
                            HF_DATA_REPO, path_in_repo=p, repo_type="dataset", recursive=False
                        )
                    ],
                    what=f"probe {row['name']}",
                )
                entry["resolved"] = len(files) > 0
                entry["n_entries"] = len(files)
            elif row["kind"] == "prefix-any":
                resolved_prefix = None
                for cand in row["candidates"]:
                    try:
                        files = hub.retry_transient(
                            lambda p=cand: [
                                e.path
                                for e in api.list_repo_tree(
                                    HF_DATA_REPO,
                                    path_in_repo=p,
                                    repo_type="dataset",
                                    recursive=False,
                                )
                            ],
                            what=f"probe {row['name']}:{cand}",
                        )
                    except Exception:  # a 404-shaped miss on this candidate — try next
                        continue
                    if files:
                        resolved_prefix = cand
                        break
                entry["resolved"] = resolved_prefix is not None
                entry["resolved_prefix"] = resolved_prefix
            else:  # kind == "error"
                entry["resolved"] = False
        except Exception as e:  # noqa: BLE001 — per-path report, never a silent skip
            entry["resolved"] = False
            entry["error"] = f"{type(e).__name__}: {e}"
        if not entry.get("resolved") and row.get("required"):
            n_missing_required += 1
        report.append(entry)
        print(
            f"[probe] {row['name']}: {'OK' if entry.get('resolved') else 'MISS'}"
            f"{' (required)' if row.get('required') else ''}",
            flush=True,
        )
    payload = {"report": report, "n_missing_required": n_missing_required, **_meta()}
    _atomic_json(cfg.out_dir / "probe_report.json", payload)
    if n_missing_required:
        print(f"[probe] FAIL: {n_missing_required} required reused inputs missing", flush=True)
        return 1
    print("[probe] PASS: all required reused inputs resolve", flush=True)
    return 0


# ── Phase: judge (P3 — ladder rollouts → rates_by_step) ─────────────────────


def _stub_judge_fn(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False):
    """Deterministic OFFLINE smoke judge (mirrors issue1481_analysis._stub_judge_fn;
    the JudgeFn seam shape). NEVER used outside --smoke."""
    del eval_prompt, cache_dir, save_raw, judge_model, dry_run

    class _R:
        n_total_draws = len(items) * n_draws
        n_dropped_draws = 0
        n_transport_lost_draws = 0

    r = _R()
    r.scores = {
        iid: float(int(hashlib.sha1(iid.encode()).hexdigest()[:4], 16) % 100)
        for iid, _q, _c in items
    }
    return r


def _judge_fn_for(cfg: Cfg, beh_key: str):
    """The registered per-behavior instrument routing (#1481 convention):
    casual → the pv trait rubric; impolite/sycophancy → judge_graded_r23."""
    if cfg.stub_judge:
        if not cfg.smoke:
            raise SystemExit("[i1947-judge] --stub-judge is smoke-only: pass --smoke with it")
        return _stub_judge_fn, "stub-smoke"
    import issue1090_fu3_worker as fu3w
    import issue1434_cells as c1434

    if beh_key == "cas":
        return c1434.pv_judge_fn, "pv_trait_score"
    return fu3w.judge_graded_r23, "registered_graded_r23"


def _judge_rung(
    cfg: Cfg,
    tag: str,
    qs: list[str],
    comps: list[list[str]],
    *,
    behavior_obj,
    judge_fn,
    inst_root: Path,
) -> dict:
    """Judge one (cell, rung): graded 0-100, threshold rate, drop split
    (mirrors issue1481_analysis._judge_cell — id budget, drop-never-coerce)."""
    items = [
        (f"{tag}-q{qi:03d}-c{ci:03d}", q, comp)
        for qi, q in enumerate(qs)
        for ci, comp in enumerate(comps[qi])
    ]
    id_map = {
        iid: "h" + hashlib.sha1(iid.encode()).hexdigest()[:12]
        for iid, _, _ in items
        if len(iid) > _JUDGE_ID_BUDGET
    }
    inst_root.mkdir(parents=True, exist_ok=True)
    if id_map:
        items = [(id_map.get(iid, iid), q, comp) for iid, q, comp in items]
        _atomic_json(inst_root / f"idmap_{tag}.json", {v: k for k, v in id_map.items()})
    result = judge_fn(
        items,
        behavior_obj.judge_rubric,
        n_draws=cfg.n_draws,
        cache_dir=inst_root / "cache",
        save_raw=inst_root / f"judge_raw_{tag}.json",
        judge_model=behavior_obj.judge_model,
    )
    scores = [result.scores.get(iid) for iid, _, _ in items]
    scored = [s for s in scores if s is not None]
    n_pos = sum(1 for s in scored if s > behavior_obj.threshold)
    rec = {
        "tag": tag,
        "n_items": len(items),
        "n_scored": len(scored),
        "k_positive": n_pos,
        "rate": (n_pos / len(scored)) if scored else None,
        "graded_mean": (sum(scored) / len(scored)) if scored else None,
        "item_drop_frac": 1.0 - (len(scored) / len(items)) if items else 0.0,
        "n_dropped_draws_content": getattr(result, "n_dropped_draws", None),
        "n_transport_lost_draws": getattr(result, "n_transport_lost_draws", None),
    }
    if rec["rate"] is None:
        logger.warning("[i1947-judge] %s: EVERY item judge-dropped (rule 9) — rate None", tag)
    return rec


def _stage_ladder(cfg: Cfg, slug: str, fname: str = "ladder_rollouts.json") -> Path:
    local = cfg.out_root / "ladders" / slug / fname
    if not local.exists():
        if cfg.local_ladders:
            raise FileNotFoundError(f"[i1947] --local-ladders set but {local} missing")
        hub.stage_hub_file(
            HF_DATA_REPO,
            f"{DATA_PREFIX}/raw_completions/ladders/{slug}/{fname}",
            local,
            repo_type="dataset",
        )
    return local


def cmd_judge(cfg: Cfg) -> int:
    """Judge every content cell's ladder (resumable per (cell, rung))."""
    from explore_persona_space.artifacts.organisms import BEHAVIORS

    todo = _content_cells(cfg)
    if not todo:
        raise SystemExit("[i1947-judge] no content cells matched the filters")
    for k, cell in enumerate(todo):
        t0 = time.time()
        beh_key = cell.beh_key
        behavior_obj = BEHAVIORS[cell.behavior]
        judge_fn, instrument = _judge_fn_for(cfg, beh_key)
        payload = _read_json(_stage_ladder(cfg, cell.slug))
        qs = payload["questions"]
        questions_sha = _sha256_text(json.dumps(list(qs), ensure_ascii=False))
        judge_root = cfg.out_dir / "judge" / beh_key
        regime_key = {
            "behavior": cell.behavior,
            "instrument": instrument,
            "n_draws": cfg.n_draws,
            "questions_sha256": questions_sha,
            "stub": bool(cfg.stub_judge),
        }
        rates_by_step: dict[str, float | None] = {}
        records: dict[str, dict] = {}
        for step_s in sorted(payload["rungs"], key=int):
            ckpt = judge_root / f"cell_{cell.slug}_rung{step_s}.json"
            if ckpt.exists():
                prior = _read_json(ckpt)
                if prior.get("regime_key") != regime_key:
                    raise RuntimeError(
                        f"[i1947-judge] {ckpt} judged under a DIFFERENT regime "
                        f"({prior.get('regime_key')} != {regime_key}) — fresh --out-dir required"
                    )
                rec = prior["record"]
            else:
                comps = payload["rungs"][step_s]["completions"]
                assert len(comps) == len(qs), (cell.slug, step_s, len(comps), len(qs))
                rec = _judge_rung(
                    cfg,
                    f"{cell.slug}-r{step_s}",
                    qs,
                    comps,
                    behavior_obj=behavior_obj,
                    judge_fn=judge_fn,
                    inst_root=judge_root / instrument,
                )
                _atomic_json(ckpt, {"regime_key": regime_key, "record": rec})
            rates_by_step[step_s] = rec["rate"]
            records[step_s] = rec
        _atomic_json(
            cfg.out_dir / "judge" / f"judged_{cell.slug}.json",
            {
                "slug": cell.slug,
                "behavior": cell.behavior,
                "instrument": instrument,
                "n_draws": cfg.n_draws,
                "questions_sha256": questions_sha,
                "rates_by_step": rates_by_step,
                "records": records,
                **_meta(),
            },
        )
        print(
            f"[judge] unit {k + 1}/{len(todo)} {cell.slug} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    return 0


# ── Phase: select (P3 — verdict manifest) ────────────────────────────────────


def cmd_select(cfg: Cfg) -> int:
    """Earliest-in-band verdict per cell (closest-approach fallback) + marker
    programmatic selections → the committed verdict_manifest.json."""
    from explore_persona_space.artifacts.recipe import select_dose_checkpoint

    content: dict[str, dict] = {}
    for cell in _content_cells(cfg):
        judged_path = cfg.out_dir / "judge" / f"judged_{cell.slug}.json"
        if not judged_path.exists():
            raise SystemExit(f"[i1947-select] {cell.slug}: judged JSON missing — run --phase judge")
        judged = _read_json(judged_path)
        rates = {
            int(s): float(r) for s, r in judged["rates_by_step"].items() if r is not None
        }  # a rate=None rung is all-dropped (rule 9) — excluded, never coerced
        if not rates:
            raise RuntimeError(f"[i1947-select] {cell.slug}: every rung judge-dropped")
        sel = select_dose_checkpoint(rates, band=JUDGED_BAND)
        content[cell.slug] = {
            "slug": cell.slug,
            "behavior": cell.behavior,
            "beh_key": cell.beh_key,
            "ctx_key": cell.ctx_key,
            "regime": cell.regime,
            "visit": cell.visit,
            "seed": cell.seed,
            "lr": cell.lr,
            "selection": {
                "step": sel.step,
                "rate": sel.rate,
                "in_band": sel.in_band,
                "fallback": sel.fallback,
            },
            "n_rungs_judged": len(rates),
            "n_rungs_dropped": len(judged["rates_by_step"]) - len(rates),
            "instrument": judged["instrument"],
            "questions_sha256": judged["questions_sha256"],
            "parent_pass_count": cells.parent_pass_count(cell.lr_source),
        }
    marker: dict[str, dict] = {}
    for cell in _marker_cells(cfg):
        local = cfg.out_root / "marker_ladders" / cell.slug / "slot_reads.json"
        if not local.exists():
            if cfg.local_ladders:
                raise FileNotFoundError(f"[i1947-select] {local} missing under --local-ladders")
            hub.stage_hub_file(
                HF_DATA_REPO,
                f"{DATA_PREFIX}/raw_completions/marker_ladders/{cell.slug}/slot_reads.json",
                local,
                repo_type="dataset",
            )
        slot = _read_json(local)
        if "selection" not in slot:
            raise RuntimeError(f"[i1947-select] {cell.slug}: slot_reads.json has no selection")
        marker[cell.slug] = {
            "slug": cell.slug,
            "behavior": "marker",
            "ctx_key": cell.ctx_key,
            "lr": cell.lr,
            "selection": slot["selection"],
        }
    n_in_band = sum(1 for e in content.values() if e["selection"]["in_band"])
    manifest = {
        "issue": ISSUE,
        "band": list(JUDGED_BAND),
        "n_draws": cfg.n_draws,
        "content": content,
        "marker": marker,
        "coverage": {"n_content": len(content), "n_in_band": n_in_band},
        **_meta(),
    }
    _atomic_json(cfg.verdict_manifest_path(), manifest)
    print(
        f"[select] wrote {cfg.verdict_manifest_path()} "
        f"({len(content)} content in_band={n_in_band}, {len(marker)} marker)",
        flush=True,
    )
    return 0


# ── Capture helpers (P4) ─────────────────────────────────────────────────────


def _adapter_subfolder(slug: str, step: int) -> str:
    """Overflow-repo rung path (unit-2 worker upload contract)."""
    if slug.startswith("mk-"):
        return f"{cells.ADAPTER_PREFIX}/marker/{slug}/checkpoint-{step}"
    return f"{cells.ADAPTER_PREFIX}/{slug}/checkpoint-{step}"


def _merge_1947(cfg: Cfg, slug: str, step: int) -> Path:
    """Merge the verdict-rung adapter onto base → local merged dir (bf16,
    atomic publish; mirrors issue1768_capture._merge_adapter with the #1947
    overflow-repo source). Caller REAPS the dir after use (plan §9)."""
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = cfg.out_root / "merged" / f"{slug}-s{step}"
    if (merged_dir / "config.json").exists():
        return merged_dir
    sub = _adapter_subfolder(slug, step)
    logger.info("[merge] %s <- %s/%s", slug, OVERFLOW_MODEL_REPO, sub)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    peft_model = PeftModel.from_pretrained(base, OVERFLOW_MODEL_REPO, subfolder=sub)
    merged = peft_model.merge_and_unload()
    tmp = merged_dir.parent / f".tmp_{slug}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    merged.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(tmp)
    del merged, peft_model, base
    gc.collect()
    try:
        os.replace(tmp, merged_dir)
    except OSError:
        if (merged_dir / "config.json").exists():
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            raise
    return merged_dir


def _reap_merged(merged_dir: Path) -> None:
    """Fail-loud reap of a consumed merged dir (fan-out accumulation, #1541)."""
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
        print(f"[reap] merged dir {merged_dir.name} reaped", flush=True)


def _stage_mix(cfg: Cfg, slug: str) -> Path:
    """Stage the cell's mix + realized consumption next to the battery."""
    mix_dir = cfg.out_root / "mixes" / slug
    for fname in ("train_mix.jsonl", "consumption_manifest.json"):
        local = mix_dir / fname
        if not local.exists():
            hub.stage_hub_file(
                HF_DATA_REPO, f"{cells.mix_prefix(slug)}/{fname}", local, repo_type="dataset"
            )
    cell = cells.CELL_BY_SLUG[slug]
    realized = cfg.out_root / "ladders" / slug / "realized_consumption.json"
    if not realized.exists() and cell.visit == "sv" and cell.kind == "content":
        hub.stage_hub_file(
            HF_DATA_REPO,
            f"{DATA_PREFIX}/raw_completions/ladders/{slug}/realized_consumption.json",
            realized,
            repo_type="dataset",
        )
    return mix_dir


def _mix_rows_tf(cfg: Cfg, slug: str, model_path: str) -> list[dict]:
    """Mix file → TF rows (token-id concat + offset-mapped spans — the
    #1092/#1315 seam rules via compute_prompt_spans; mirrors
    issue1768_capture._mix_positive_rows row construction)."""
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    mix_dir = _stage_mix(cfg, slug)
    # Row kind rides the consumption manifest's row_ids ("pos:<sha>:<idx>"),
    # aligned with mix-file order — the datagen writer convention.
    row_ids = _read_json(mix_dir / "consumption_manifest.json")["row_ids"]
    tok = AutoTokenizer.from_pretrained(model_path)
    rows: list[dict] = []
    with (mix_dir / "train_mix.jsonl").open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            r = json.loads(line)
            p_msgs = (
                r["prompt"]
                if isinstance(r["prompt"], list)
                else [{"role": "user", "content": r["prompt"]}]
            )
            comp = r["completion"]
            comp_text = comp if isinstance(comp, str) else comp[-1]["content"]
            text = tok.apply_chat_template(p_msgs, tokenize=False, add_generation_prompt=True)
            prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
            resp_ids = tok(comp_text, add_special_tokens=False)["input_ids"]
            if not resp_ids:
                continue
            system = next((m["content"] for m in p_msgs if m["role"] == "system"), None)
            chat = [m for m in p_msgs if m["role"] != "system"]
            prefix_len, context_len = compute_prompt_spans(
                tok,
                system,
                chat[-1]["content"],
                prompt_ids,
                prior_messages=chat[:-1] or None,
                prefix_end="last_user",
                on_seam="snap",
            )
            rows.append(
                {
                    "persona": slug,
                    "question_idx": idx,  # mix ROW index (consumption-manifest keyed)
                    "row_kind": row_ids[idx].split(":", 1)[0],  # pos | neg | gen
                    "question": chat[-1]["content"],
                    "prompt_sha": _sha256_text(text + "\x00" + comp_text)[:16],
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": resp_ids,
                    "prefix_len": prefix_len,
                    "context_len": context_len,
                }
            )
    assert rows, (slug, "empty mix")
    return rows


def _consumed_row_idxs(cfg: Cfg, slug: str, step: int) -> set[int]:
    """Mix row indices consumed through checkpoint ``step`` (REALIZED
    consumption log — plan §4.2; the manifest is evidence, never assumption).

    Checkpoint-``step`` == ``step`` completed optimizer steps, so a row with
    ``realized_step_of_idx[i] < step`` (0-based) was gradient-producing at that
    rung. Rep-regime cells have NO sequential seam (worker contract): every row
    is consumed once ``step * effective_batch >= n_rows`` (epoch 1 done); an
    earlier rung is unknowable there and fails loud."""
    cell = cells.CELL_BY_SLUG[slug]
    realized_path = cfg.out_root / "ladders" / slug / "realized_consumption.json"
    if not realized_path.exists() and cell.visit == "rep":
        if step * cells.EFFECTIVE_BATCH >= cell.n_rows:
            return set(range(cell.n_rows))
        raise RuntimeError(
            f"[i1947] {slug}: rep cell rung {step} predates epoch-1 completion and "
            "carries no realized consumption log — consumed set unknowable"
        )
    realized = _read_json(realized_path)
    if "realized_step_of_idx" in realized:  # the train/sft.py seam schema
        return {
            i
            for i, s in enumerate(realized["realized_step_of_idx"])
            if s is not None and int(s) < step
        }
    rec = realized.get("row_to_step") or realized.get("steps")
    if isinstance(rec, dict):  # legacy fixtures: {row_id: step} or {step: [row_ids]}
        vals = [v for v in rec.values() if v is not None]
        if vals and not isinstance(vals[0], list):
            return {int(k) for k, v in rec.items() if int(v) < step}
        out: set[int] = set()
        for s, row_ids in rec.items():
            if int(s) < step:
                out.update(int(r) for r in row_ids)
        return out
    raise RuntimeError(f"[i1947] unrecognized realized_consumption schema for {slug}")


def _tf_store(
    cfg: Cfg,
    model_path: str,
    rows: list[dict],
    unit_id: str,
    out_path: Path,
    extra: dict,
    *,
    spans: tuple[str, ...] = CAPTURE_SPANS,
) -> None:
    """One TF capture → #1768-schema pooled store (both context summaries in
    the SAME forward pass — binding directive; fp16 storage, roundtrip check)."""
    import torch

    import issue1768_capture as CAP

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    if out_path.exists():
        return
    pooled = _teacher_forced_span_means(
        model_path,
        rows,
        [unit_id],
        layers=list(cfg.layers),
        spans=spans,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        tf_batch_size=cfg.tf_batch,
    )
    cos_min = CAP._fp16_roundtrip_cos_min(pooled)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    CAP._save_pooled(
        out_path,
        unit_id,
        pooled,
        rows,
        {**extra, "fp16_roundtrip_cos_min": cos_min, "spans": list(spans), **_meta()},
    )
    print(f"[store] {out_path.relative_to(cfg.out_root)} rows={len(rows)}", flush=True)


def _upload_tree(cfg: Cfg, local: Path, path_in_repo: str) -> None:
    """One bulk upload_folder commit + exact-set verify skipped (verify at the
    dispatcher's phase end via verify_repo_paths_uploaded on key files)."""
    if not cfg.upload:
        return
    hub.retry_transient(
        lambda: hub._upload(
            local, repo_id=HF_DATA_REPO, repo_type="dataset", path_in_repo=path_in_repo
        ),
        what=f"upload {path_in_repo}",
    )


# ── Units (P4/P5) ────────────────────────────────────────────────────────────


def _verdict_arms(cfg: Cfg) -> dict:
    man_path = cfg.verdict_manifest_path()
    if not man_path.exists():  # pod-side: the manifest is committed to the branch
        alt = REPO_ROOT / "eval_results/issue_1947/analysis/verdict_manifest.json"
        if alt.exists():
            man_path = alt
        else:
            raise SystemExit(f"[i1947] verdict manifest missing at {man_path} — run --phase select")
    return _read_json(man_path)


def unit_arm(cfg: Cfg, slug: str) -> None:
    """Per-arm P4 unit: TF trained-rows tree (consumed + full, trained + base),
    on-policy tree, 20-q panel (trained + base), TF fixed-pool margin — one
    merge, then reap (plan §4.4)."""
    man = _verdict_arms(cfg)
    entry = man["content"].get(slug) or man["marker"].get(slug)
    assert entry is not None, (slug, "not in verdict manifest")
    step = int(entry["selection"]["step"])
    done = cfg.out_root / "battery" / "trained_rows" / slug / "unit_done.json"
    if done.exists():
        print(f"[arm] {slug}: done — skip", flush=True)
        return
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    assert_out_root_headroom(cfg.out_root, PHASE_HEADROOM_GB["capture"], phase=f"arm:{slug}")
    merged = _merge_1947(cfg, slug, step)
    try:
        rows = _mix_rows_tf(cfg, slug, str(merged))
        if cfg.smoke:
            rows = rows[:6]
        consumed = _consumed_row_idxs(cfg, slug, step)
        rows_consumed = [r for r in rows if r["question_idx"] in consumed]
        if not rows_consumed:  # smoke slices can miss the consumed set — keep ≥1 row
            rows_consumed = rows[: max(1, len(rows) // 4)]
        tdir = cfg.out_root / "battery" / "trained_rows" / slug
        extra = {"slug": slug, "verdict_step": step, "n_consumed": len(rows_consumed)}

        def _kinds(rs: list[dict]) -> dict:
            return {"row_kinds": [r["row_kind"] for r in rs]}

        _tf_store(
            cfg,
            str(merged),
            rows,
            slug,
            tdir / "pooled.pt",
            {**extra, "set": "full", **_kinds(rows)},
        )
        _tf_store(
            cfg,
            str(merged),
            rows_consumed,
            slug,
            tdir / "pooled_consumed.pt",
            {**extra, "set": "consumed", **_kinds(rows_consumed)},
        )
        _tf_store(
            cfg,
            BASE_MODEL,
            rows,
            slug,
            tdir / "pooled_base.pt",
            {**extra, "set": "full", **_kinds(rows)},
        )
        _tf_store(
            cfg,
            BASE_MODEL,
            rows_consumed,
            slug,
            tdir / "pooled_base_consumed.pt",
            {**extra, "set": "consumed", **_kinds(rows_consumed)},
        )
        if not slug.startswith("mk-"):
            _unit_onpolicy_and_panel(cfg, slug, entry, merged)
            _unit_margin(cfg, slug, entry, merged)
        _upload_tree(cfg, tdir, f"{DATA_PREFIX}/battery/trained_rows/{slug}")
        _atomic_json(done, {"slug": slug, "step": step, **_meta()})
    finally:
        _reap_merged(merged)


def _cell_context(slug: str):
    """The cell's trained context via the #1481 registry (worker convention)."""
    import issue1090_fu3_worker as fu3w
    import issue1481_cells as c1481

    cell = cells.CELL_BY_SLUG[slug]
    return fu3w.ensure_context(c1481.context_id_for(cell.behavior, cell.ctx_key), cell.behavior)


def _own_capture(
    cfg: Cfg,
    slug: str,
    side: str,
    gen_model_path: str,
    ctx,
    questions: list[str],
    out_dir: Path,
    store_name: str,
) -> None:
    """Greedy gens under the trained context + TF capture of OWN responses
    (rollout text persisted BEFORE any reduce — Upload Policy #779)."""
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import compute_prompt_spans
    from explore_persona_space.artifacts.organisms import (
        _default_vllm_generate_fn,
        _generate_and_persist,
    )

    store_path = out_dir / store_name
    if store_path.exists():
        return
    gen_dir = out_dir / f"gen_{side}"
    gen = _default_vllm_generate_fn(BASE_MODEL)
    try:
        comps = _generate_and_persist(
            gen,
            side,
            gen_model_path if side == "trained" else None,
            ctx,
            questions,
            n=1,
            temperature=0.0,
            out_dir=gen_dir,
            base_model=BASE_MODEL,
        )
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()
    if cfg.upload:  # rollout TEXT lands on the Hub before any reduce
        _upload_tree(cfg, gen_dir, f"{DATA_PREFIX}/raw_completions/battery/{slug}/{side}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    rows = []
    for qi, q in enumerate(questions):
        resp = comps[qi][0]
        p_msgs = ctx.messages(q)
        text = tok.apply_chat_template(p_msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
        resp_ids = tok(resp, add_special_tokens=False)["input_ids"]
        if not resp_ids:
            continue
        system = next((m["content"] for m in p_msgs if m["role"] == "system"), None)
        chat = [m for m in p_msgs if m["role"] != "system"]
        prefix_len, context_len = compute_prompt_spans(
            tok,
            system,
            chat[-1]["content"],
            prompt_ids,
            prior_messages=chat[:-1] or None,
            prefix_end="last_user",
            on_seam="snap",
        )
        rows.append(
            {
                "persona": slug,
                "question_idx": qi,
                "prompt_sha": _sha256_text(text)[:16],
                "prompt_token_ids": prompt_ids,
                "response_token_ids": resp_ids,
                "prefix_len": prefix_len,
                "context_len": context_len,
            }
        )
    assert rows, (slug, side, "all responses empty")
    capture_model = gen_model_path if side == "trained" else BASE_MODEL
    _tf_store(cfg, capture_model, rows, slug, store_path, {"slug": slug, "side": side})


def _unit_onpolicy_and_panel(cfg: Cfg, slug: str, entry: dict, merged: Path) -> None:
    """On-policy tree at the trained (context, question) pairs + 20-q panel
    capture (trained + base — halves split by question parity in P6)."""
    from explore_persona_space.artifacts.organisms import BEHAVIORS

    cell = cells.CELL_BY_SLUG[slug]
    ctx = _cell_context(slug)
    # On-policy tree: the trained questions = the mix positives' questions
    # (kind from the consumption manifest's row_ids — the datagen convention).
    mix_dir = _stage_mix(cfg, slug)
    row_ids = _read_json(mix_dir / "consumption_manifest.json")["row_ids"]
    pos_qs: list[str] = []
    with (mix_dir / "train_mix.jsonl").open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            if not row_ids[idx].startswith("pos:"):
                continue
            r = json.loads(line)
            chat = [m for m in r["prompt"] if m.get("role") == "user"]
            if chat:
                pos_qs.append(chat[-1]["content"])
    assert pos_qs, (slug, "no positive rows in mix")
    if cfg.smoke:
        pos_qs = pos_qs[:3]
    op_dir = cfg.out_root / "battery" / "onpolicy" / slug
    _own_capture(cfg, slug, "trained", str(merged), ctx, pos_qs, op_dir, "pooled.pt")
    # Base own-capture at the SAME trained pairs — the on-policy tree's base
    # side (per-row shift = trained own-response − base own-response).
    _own_capture(cfg, slug, "base", BASE_MODEL, ctx, pos_qs, op_dir, "pooled_base.pt")
    _upload_tree(cfg, op_dir, f"{DATA_PREFIX}/battery/onpolicy/{slug}")
    # 20-q held-out panel (trained + base v0; halves = question parity in P6).
    panel_qs = list(BEHAVIORS[cell.behavior].eval_question_bank)[: 2 if cfg.smoke else 20]
    pdir = cfg.out_root / "battery" / "panel" / slug
    _own_capture(cfg, slug, "trained", str(merged), ctx, panel_qs, pdir, "pooled_trained.pt")
    _own_capture(cfg, slug, "base", BASE_MODEL, ctx, panel_qs, pdir, "pooled_base.pt")
    _upload_tree(cfg, pdir, f"{DATA_PREFIX}/battery/panel/{slug}")


def _unit_margin(cfg: Cfg, slug: str, entry: dict, merged: Path) -> None:
    """TF fixed-pool margin at the verdict rung (dual-DV companion; #1900
    frozen per-family pools — the #722-derived syc pool rides verbatim)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.margin import compute_tf_margin

    out = cfg.out_root / "battery" / "margins" / f"{slug}.json"
    if out.exists():
        return
    fam = BEH_FAM[entry["behavior"]]
    pools_local = cfg.out_root / "inputs" / "margin_pools" / f"pools_{fam}.json"
    if not pools_local.exists():
        hub.stage_hub_file(
            HF_DATA_REPO,
            f"{MARGIN_POOLS_PREFIX}/pools_{fam}.json",
            pools_local,
            repo_type="dataset",
        )
    pools = _read_json(pools_local)
    pos_pairs = pools.get("pos_pairs") or pools.get("pos")
    neg_pairs = pools.get("neg_pairs") or pools.get("neg")
    assert pos_pairs and neg_pairs, (fam, sorted(pools))
    if cfg.smoke:
        pos_pairs, neg_pairs = pos_pairs[:2], neg_pairs[:2]
    ctx = _cell_context(slug)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(merged),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map={"": device},
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    res = compute_tf_margin(model, tok, ctx.messages, pos_pairs, neg_pairs, device=device)
    del model
    _atomic_json(
        out,
        {
            "slug": slug,
            "family": fam,
            "margin": res.margin,
            "pos_mean_ln_logp": res.pos_mean_ln_logp,
            "neg_mean_ln_logp": res.neg_mean_ln_logp,
            "n_pos": res.n_pos,
            "n_neg": res.n_neg,
            **_meta(),
        },
    )
    _upload_tree(cfg, out.parent, f"{DATA_PREFIX}/battery/margins")


def unit_delta(cfg: Cfg, pool: str) -> None:
    """δ unit over the (behavior × context) mix positives — the REAL
    issue1768_capture.run_delta_unit body (arm-index injection seam; plan
    check (l) call-shape bind). Consumed-subset / n-matched δ re-reductions
    happen in P6 over the persisted per-row stores."""
    import issue1768_capture as CAP
    import issue1768_cells as X1768

    beh_key, ctx_key = pool.split("-", 1)
    arm_id = f"{beh_key}-{ctx_key}-delta1947"
    out_dir = cfg.out_root / "delta_tf" / arm_id
    if (out_dir / "tbar.pt").exists():
        print(f"[delta] {pool}: done — skip", flush=True)
        return
    arm = X1768.Arm(
        arm_id=arm_id,
        kind="content",
        beh_key=beh_key,
        ctx_key=ctx_key,
        regime="con",
        seed=42,
        lr=0.0,
        step=0,
        selection_read=0.0,
        method="lora",
    )
    CAP._full_arm_index()[arm_id] = arm  # documented injection seam (module cache)
    reg_path = cfg.out_root / "arm_registry.json"
    reg = _read_json(reg_path) if reg_path.exists() else {"mix_pos_sources": {}}
    reg.setdefault("mix_pos_sources", {})[arm_id] = {
        "pos_path": f"{DATA_PREFIX}/raw_completions/datagen/positives/{beh_key}-{ctx_key}/pos.jsonl",
        "layout": "content",
    }
    _atomic_json(reg_path, reg)
    cap_cfg = CAP.Cfg(
        out_root=cfg.out_root,
        phases=(),
        smoke=cfg.smoke,
        layers=tuple(cfg.layers),
        tf_batch=cfg.tf_batch,
        upload=False,
        model_override=cfg.model_override,
    )
    CAP.run_delta_unit(cap_cfg, arm_id)
    _upload_tree(cfg, out_dir, f"{DATA_PREFIX}/battery/delta_tf/{arm_id}")
    print(f"[delta] {pool}: tbar.pt written", flush=True)


def unit_dynamics(cfg: Cfg, slug: str) -> None:
    """Per-rung TF trained-rows reads over rows-consumed-so-far (2 cells;
    plan §4.4). Merges one rung at a time; reaps after each capture."""
    man = _verdict_arms(cfg)
    assert slug in man["content"], slug
    mix_dir = _stage_mix(cfg, slug)
    del mix_dir
    rungs = sorted(int(s) for s in _read_json(_stage_ladder(cfg, slug))["rungs"])
    if cfg.smoke:
        rungs = rungs[:2]
    base_rows_done = False
    for k, step in enumerate(rungs):
        out = cfg.out_root / "battery" / "dynamics" / slug / f"rung{step}" / "pooled.pt"
        if out.exists():
            continue
        merged = _merge_1947(cfg, slug, step)
        try:
            rows = _mix_rows_tf(cfg, slug, str(merged))
            if cfg.smoke:
                rows = rows[:4]
            consumed = _consumed_row_idxs(cfg, slug, step)
            sub = [r for r in rows if r["question_idx"] in consumed] or rows[:1]
            _tf_store(cfg, str(merged), sub, slug, out, {"slug": slug, "rung": step})
            if not base_rows_done:
                _tf_store(
                    cfg,
                    BASE_MODEL,
                    rows,
                    slug,
                    cfg.out_root / "battery" / "dynamics" / slug / "pooled_base_full.pt",
                    {"slug": slug, "set": "full"},
                )
                base_rows_done = True
        finally:
            _reap_merged(merged)
        print(f"[dyn] {slug} rung {k + 1}/{len(rungs)} (step {step})", flush=True)
    _upload_tree(
        cfg,
        cfg.out_root / "battery" / "dynamics" / slug,
        f"{DATA_PREFIX}/battery/dynamics/{slug}",
    )


def _stage_corpus_inputs(cfg: Cfg) -> dict:
    """Stage the pinned pfx sha set + the #1768 base store + base rows-with-text
    (matched-text TF inputs) — parent pre-stage, before any fan-out (#1315)."""
    import issue1768_cells as X1768

    sample_local = cfg.out_root / "on_target" / "inputs" / "corpus_sample_pfx.json"
    if not sample_local.exists():
        hub.stage_hub_file(
            HF_DATA_REPO,
            f"{X1768.HF_PREFIX}/on_target/inputs/corpus_sample_pfx.json",
            sample_local,
            repo_type="dataset",
        )
    # Base round-1 store + rows-with-text + spans: stage EVERY file of the
    # base_content prefix to its EXACT consumed target (per-file — no
    # mirror-root arithmetic; the stage_hub_prefix dest trap, gotchas.md).
    base_dir = cfg.out_root / "corpus_capture" / "base_content"
    if not (base_dir / "rows_spans.json").exists() or not (base_dir / "pooled.pt").exists():
        from huggingface_hub import HfApi

        prefix = f"{X1768.HF_PREFIX}/corpus_capture/base_content"
        entries = hub.retry_transient(
            lambda: [
                e.path
                for e in HfApi().list_repo_tree(
                    HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=False
                )
            ],
            what="list base_content prefix",
        )
        if not entries:
            raise RuntimeError(f"[i1947-corpus] {prefix} empty — base inputs unresolvable")
        for path in entries:
            target = base_dir / Path(path).name
            if not target.exists():
                hub.stage_hub_file(HF_DATA_REPO, path, target, repo_type="dataset")
        assert (base_dir / "rows_spans.json").exists(), base_dir
    return X1768.load_pfx_sample(cfg.out_root)


def unit_corpus(cfg: Cfg, slug: str) -> None:
    """Bare-corpus capture for one con-s42 arm: greedy own tree + matched-text
    TF tree at the pinned 4,400-row pfx sha set — stores land in the #1768
    schema/paths so issue1768_fit.fit_bare_n_cell runs VERBATIM (P5)."""
    import issue1768_capture as CAP
    import issue1768_cells as X1768

    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    own_store = cfg.out_root / "corpus_capture" / slug / "pooled.pt"
    tf_store = cfg.out_root / "corpus_capture_tf" / slug / "pooled_tf.pt"
    if own_store.exists() and tf_store.exists():
        print(f"[corpus] {slug}: done — skip", flush=True)
        return
    assert_out_root_headroom(cfg.out_root, PHASE_HEADROOM_GB["corpus"], phase=f"corpus:{slug}")
    sample = _stage_corpus_inputs(cfg)
    man = _verdict_arms(cfg)
    step = int(man["content"][slug]["selection"]["step"])
    rows_meta = sample["rows"][: 8 if cfg.smoke else None]
    key = next((k for k in ("prompt", "question", "text") if k in rows_meta[0]), None)
    assert key is not None, sorted(rows_meta[0])
    prompts = [r[key] for r in rows_meta]
    merged = _merge_1947(cfg, slug, step)
    try:
        if not own_store.exists():
            gen_dir = cfg.out_root / "corpus_gen" / slug
            cap_cfg = CAP.Cfg(out_root=cfg.out_root, phases=(), smoke=cfg.smoke, upload=False)
            CAP._generate_rows_vllm(cap_cfg, slug, str(merged), prompts, 0, gen_dir)
            if cfg.upload:  # rollout TEXT before any reduce
                _upload_tree(cfg, gen_dir, f"{DATA_PREFIX}/raw_completions/corpus/{slug}")
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(BASE_MODEL)
            rows = CAP._read_shards(gen_dir)
            rows, n_drop, seams, _npfx = CAP._attach_spans(tok, prompts, rows)
            for r in rows:  # ROUND-1 index space (the bare_n loader's join key)
                r["persona"] = slug
                meta = rows_meta[r["question_idx"]]
                r["prompt_sha"] = meta.get("sha", r.get("prompt_sha"))
                r["question_idx"] = int(meta["src_qidx"])
            print(f"[corpus] {slug}: gen rows={len(rows)} dropped={n_drop} seams={seams}")
            _tf_store(cfg, str(merged), rows, slug, own_store, {"slug": slug, "tree": "own"})
        if not tf_store.exists():
            base_rows = CAP._read_rows_with_spans(cfg.out_root / "corpus_capture" / "base_content")
            want = {r["sha"] if "sha" in r else r.get("prompt_sha") for r in rows_meta}
            base_rows = [r for r in base_rows if r.get("prompt_sha") in want or not want]
            if cfg.smoke:
                base_rows = base_rows[:8]
            for r in base_rows:
                r["persona"] = slug
            _tf_store(
                cfg,
                str(merged),
                base_rows,
                slug,
                tf_store,
                {"slug": slug, "tree": "matched_text_tf"},
                spans=CAPTURE_SPANS,
            )
        for tree in ("corpus_capture", "corpus_capture_tf"):
            _upload_tree(cfg, cfg.out_root / tree / slug, f"{DATA_PREFIX}/{tree}/{slug}")
    finally:
        _reap_merged(merged)
    del X1768


def unit_fit(cfg: Cfg, slug: str) -> None:
    """P5: M⁺ ridge fits at n=3,000 with identity+bias + kNN baselines, refit
    floor + D verdict — the REAL issue1768_fit.fit_bare_n_cell per layer
    (sha-join asserted; r3 cross-check recorded when the staged r3 fit JSON
    resolves)."""
    import issue1768_fit as FIT

    sample = _stage_corpus_inputs(cfg)
    pinned_shas = {r.get("sha") for r in sample["rows"] if r.get("sha")}
    import torch

    own = torch.load(
        cfg.out_root / "corpus_capture" / slug / "pooled.pt",
        map_location="cpu",
        weights_only=False,
    )
    got = set(own["row_sha"])
    missing = {s for s in pinned_shas if s not in got}
    if not cfg.smoke and len(missing) > 0.1 * len(pinned_shas):
        raise RuntimeError(
            f"[i1947-fit] {slug}: {len(missing)}/{len(pinned_shas)} pinned shas missing "
            "from the own store — sha-join violated (plan §3 row-coverage assert)"
        )
    results_dir = cfg.out_root / "fits"
    for layer in cfg.layers:
        t0 = time.time()
        rec = FIT.fit_bare_n_cell(cfg.out_root, results_dir, slug, layer, cfg.smoke)
        print(
            f"[fit] {slug} L{layer} verdict={rec.get('map_change', {}).get('verdict')} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    _upload_tree(cfg, results_dir, f"{DATA_PREFIX}/fits")


# ── capture-fit dispatcher (work-conserving GPU fan-out) ─────────────────────


def _physical_gpus() -> list[int]:
    """Physical GPU ids via nvidia-smi (never torch — CVD-clobber gotcha)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        ids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        return ids or [0]
    except (OSError, ValueError):
        return [0]


def _enumerate_units(cfg: Cfg) -> list[str]:
    man = _verdict_arms(cfg)
    slugs = sorted(man["content"]) + sorted(man["marker"])
    if cfg.cells_filter:
        slugs = [s for s in slugs if s in cfg.cells_filter]
    units = [f"arm:{s}" for s in slugs]
    pools = sorted(
        {
            f"{cells.CELL_BY_SLUG[s].beh_key}-{cells.CELL_BY_SLUG[s].ctx_key}"
            for s in slugs
            if not s.startswith("mk-")
        }
    )
    units += [f"delta:{p}" for p in pools]
    units += [
        f"dyn:{s}"
        for s in DYNAMICS_SLUGS
        if s in man["content"] and (not cfg.cells_filter or s in cfg.cells_filter)
    ]
    corpus = [
        s
        for s in CORPUS_ARM_SLUGS
        if s in man["content"] and (not cfg.cells_filter or s in cfg.cells_filter)
    ]
    units += [f"corpus:{s}" for s in corpus]
    units += [f"fit:{s}" for s in corpus]  # fits sequence AFTER corpus (below)
    return units


def run_unit(cfg: Cfg, unit: str) -> None:
    kind, _, key = unit.partition(":")
    fn = {
        "arm": unit_arm,
        "delta": unit_delta,
        "dyn": unit_dynamics,
        "corpus": unit_corpus,
        "fit": unit_fit,
    }.get(kind)
    if fn is None:
        raise SystemExit(f"[i1947] unknown unit kind {kind!r}")
    fn(cfg, key)


def cmd_capture_fit(cfg: Cfg, argv_base: list[str]) -> int:
    """Work-conserving per-unit subprocess fan-out over the physical GPUs
    (CVD pinned in the LAUNCHER env — the #545 clobber rule); fits run only
    after every capture unit lands (#825 store-before-fit ordering)."""
    units = _enumerate_units(cfg)
    captures = [u for u in units if not u.startswith("fit:")]
    fits = [u for u in units if u.startswith("fit:")]
    gpus = _physical_gpus()
    print(f"[dispatch] {len(captures)} capture units + {len(fits)} fits on {len(gpus)} GPUs")

    def _fan(queue: list[str]) -> list[str]:
        pending = list(queue)
        running: dict[int, tuple[subprocess.Popen, str]] = {}
        failed: list[str] = []
        n_total = len(pending)
        n_done = 0
        while pending or running:
            for gpu in gpus:
                if gpu in running or not pending:
                    continue
                unit = pending.pop(0)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                cmd = argv_base + ["--phase", "unit", "--unit", unit, "--gpu-id", str(gpu)]
                log = cfg.out_root / "logs" / f"unit_{unit.replace(':', '_')}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("ab") as fh:
                    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
                running[gpu] = (proc, unit)
                print(f"[dispatch] gpu{gpu} <- {unit}", flush=True)
            time.sleep(3 if not cfg.smoke else 0.2)
            for gpu in list(running):
                proc, unit = running[gpu]
                rc = proc.poll()
                if rc is None:
                    continue
                del running[gpu]
                n_done += 1
                if rc != 0:
                    failed.append(unit)
                    log = cfg.out_root / "logs" / f"unit_{unit.replace(':', '_')}.log"
                    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                    print(f"[dispatch] {unit} FAILED rc={rc}; log tail:", flush=True)
                    for line in tail:
                        print(f"    {line}", flush=True)
                else:
                    print(f"[dispatch] unit {n_done}/{n_total} {unit} done rc=0", flush=True)
        return failed

    failed = _fan(captures)
    if not failed:
        failed = _fan(fits)
    else:
        print("[dispatch] capture failures — fits NOT started (#825 ordering)", flush=True)
    sentinel_dir = cfg.sentinel_dir or SENTINEL_DIR_DEFAULT
    payload = {
        "issue": ISSUE,
        "phase": "battery",
        "status": "done" if not failed else "failed",
        "n_units": len(units),
        "failed_units": failed,
        **_meta(),
    }
    try:
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(sentinel_dir / f"issue-{ISSUE}-battery-done.json", payload)
    except OSError as e:  # VM smoke has no /workspace — record loud, don't crash
        print(f"[dispatch] sentinel write skipped ({e})", flush=True)
    _phase("done", status=payload["status"], failed=len(failed))
    return 0 if not failed else 1


# ── import-check ─────────────────────────────────────────────────────────────


def cmd_import_check() -> int:
    """Resolve every deferred import this driver touches (the #606 gate)."""
    import issue1090_fu3_worker as fu3w
    import issue1434_cells as c1434
    import issue1481_cells as c1481
    import issue1768_capture as CAP
    import issue1768_cells as X1768
    import issue1768_fit as FIT
    import issue1900_tfm as TFM

    from explore_persona_space.analysis.representation_shift import (
        SPAN_ARMS_LAST,
        _teacher_forced_span_means,
        compute_prompt_spans,
    )
    from explore_persona_space.artifacts.organisms import (
        BEHAVIORS,
        _default_vllm_generate_fn,
        _generate_and_persist,
    )
    from explore_persona_space.artifacts.recipe import select_dose_checkpoint
    from explore_persona_space.eval.margin import compute_tf_margin
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    names = [
        fu3w.judge_graded_r23,
        fu3w.ensure_context,
        c1434.pv_judge_fn,
        c1481.context_id_for,
        CAP.run_delta_unit,
        CAP._generate_rows_vllm,
        CAP._attach_spans,
        CAP._read_shards,
        CAP._read_rows_with_spans,
        CAP._save_pooled,
        CAP._fp16_roundtrip_cos_min,
        X1768.Arm,
        X1768.load_pfx_sample,
        X1768.base_unit_for,
        FIT.fit_bare_n_cell,
        FIT.load_bare_n_cell,
        TFM.FAMILIES,
        _teacher_forced_span_means,
        compute_prompt_spans,
        SPAN_ARMS_LAST,
        BEHAVIORS,
        _default_vllm_generate_fn,
        _generate_and_persist,
        select_dose_checkpoint,
        compute_tf_margin,
        assert_out_root_headroom,
    ]
    print(f"[import-check] OK ({len(names)} symbols resolved)")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="#1947 P3-P5 battery driver")
    p.add_argument(
        "--phase",
        required=True,
        choices=("probe", "judge", "select", "capture-fit", "unit", "import-check"),
    )
    p.add_argument("--out-root", default=cells.OUT_ROOT_DEFAULT)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "eval_results/issue_1947/analysis"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--stub-judge", action="store_true", help="smoke-only offline judge")
    p.add_argument("--cells", default="", help="comma-separated slug filter")
    p.add_argument("--behaviors", default="sycophancy,impolite,writing_style")
    p.add_argument("--n-draws", type=int, default=JUDGE_N_DRAWS)
    p.add_argument("--layers", default=",".join(str(x) for x in LAYERS))
    p.add_argument("--tf-batch", type=int, default=8)
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--unit", default=None, help="unit spec for --phase unit")
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--local-ladders", action="store_true", help="smoke: no Hub staging")
    p.add_argument("--model-override", default=None, help="smoke-only tiny-model substitute")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    if args.phase == "import-check":
        return cmd_import_check()
    cfg = Cfg(
        out_root=Path(args.out_root),
        out_dir=Path(args.out_dir),
        smoke=args.smoke,
        stub_judge=args.stub_judge,
        cells_filter=tuple(s for s in args.cells.split(",") if s),
        behaviors=tuple(b for b in args.behaviors.split(",") if b),
        n_draws=args.n_draws,
        layers=tuple(int(x) for x in args.layers.split(",") if x),
        tf_batch=args.tf_batch,
        upload=not args.no_upload,
        gpu_id=args.gpu_id,
        sentinel_dir=Path(args.sentinel_dir) if args.sentinel_dir else None,
        local_ladders=args.local_ladders,
        model_override=args.model_override,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _phase(args.phase, smoke=cfg.smoke)
    if args.phase == "probe":
        return cmd_probe(cfg)
    if args.phase == "judge":
        return cmd_judge(cfg)
    if args.phase == "select":
        return cmd_select(cfg)
    if args.phase == "unit":
        assert args.unit, "--phase unit requires --unit"
        run_unit(cfg, args.unit)
        return 0
    if args.phase == "capture-fit":
        argv_base = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--out-root",
            str(cfg.out_root),
            "--out-dir",
            str(cfg.out_dir),
            "--tf-batch",
            str(cfg.tf_batch),
            "--layers",
            ",".join(str(x) for x in cfg.layers),
        ]
        if cfg.smoke:
            argv_base.append("--smoke")
        if not cfg.upload:
            argv_base.append("--no-upload")
        if cfg.cells_filter:
            argv_base += ["--cells", ",".join(cfg.cells_filter)]
        return cmd_capture_fit(cfg, argv_base)
    raise SystemExit(f"unknown phase {args.phase!r}")


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit — the PyGILState_Release atexit gotcha
