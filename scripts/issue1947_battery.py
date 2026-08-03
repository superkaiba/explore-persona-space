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

BINDING last-token directive (task #1947 epm:progress v7 + v9): every
context-side capture records the summaries from ONE forward pass —
``last_prompt`` (the FINAL token of the generation-rendered prompt, the #779
convention / the #1768 lasttoken-repool position) PRIMARY, plus span-mean and
``last_ctx`` (last user-content token) SECONDARY, and ``prefix_last`` for the
prefix arm; answer-side stays span-mean. Implemented via the
``SPAN_ARMS_LAST`` 1-token spans in ``analysis/representation_shift.py``
(v9 decode check at capture time).
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
# All context summaries in ONE forward pass (binding directive v7+v9: last_prompt
# PRIMARY, span-mean + last_ctx SECONDARY, prefix_last) + answer side.
CAPTURE_SPANS = ("prefix", "context", "response", "prefix_last", "last_prompt", "last_ctx")
_JUDGE_ID_BUDGET = 53  # Batch custom_id budget (#1415)
# Per-phase disk floors on the pod out-root (plan §9 mount binding).
PHASE_HEADROOM_GB = {"capture": 40.0, "corpus": 40.0, "fit": 10.0}
PLAN_P4P5_GPU_H = 27.0  # plan §9 P4 (22) + P5 (5) booking — the pilot-gate bound
PILOT_GATE_RC = 7  # artifact-routed halt (the #1415 convention; never a bare rc=1)
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
                    # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient; single-path probe
                    lambda p=row["path"]: api.file_exists(HF_DATA_REPO, p, repo_type="dataset"),
                    what=f"probe {row['name']}",
                )
                entry["resolved"] = bool(ok)
            elif row["kind"] == "prefix":
                files = hub.retry_transient(
                    lambda p=row["path"]: [
                        e.path
                        # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient; prefix-scoped
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
                                # HUB_VERIFY_RETRY_EXEMPT: retry_transient-wrapped; prefix-scoped
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


def _consumed_row_idxs(cfg: Cfg, slug: str, step: int) -> set[int] | None:
    """Mix row indices consumed through checkpoint ``step`` (REALIZED
    consumption log — plan §4.2; the manifest is evidence, never assumption).

    Checkpoint-``step`` == ``step`` completed optimizer steps, so a row with
    ``realized_step_of_idx[i] < step`` (0-based) was gradient-producing at that
    rung. Rep-regime cells have NO sequential seam (worker contract): every row
    is consumed once ``step * effective_batch >= n_rows`` (epoch 1 done); an
    earlier rung is unknowable there and fails loud. MARKER cells also train
    without the seam (one shuffled RandomSampler epoch, 6,400 rows == 400
    steps x 16 exactly — plan §4.2): the consumed SET is the full mix iff the
    verdict rung completes the epoch; a below-ceiling rung's membership is
    shuffle-unknowable and returns ``None`` — the caller SKIPS the consumed
    tree and records the skip (r1 code-review Critical 1: the pre-fix
    fall-through raised FileNotFoundError after the 15 GB merge)."""
    cell = cells.CELL_BY_SLUG[slug]
    realized_path = cfg.out_root / "ladders" / slug / "realized_consumption.json"
    if cell.kind == "marker":
        if step * cells.EFFECTIVE_BATCH >= cell.n_rows:
            return set(range(cell.n_rows))
        return None  # below-ceiling marker rung: consumed membership unknowable
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


MARKER_TREE_ROWS = 1280  # marker trained-rows tree cap: plan §9 P4 books trained-rows
# units at ~1,200+300 rows; the full 6,400-row marker mix is ~5x that. Deterministic
# stratified subsample (row_kind proportions preserved) — the r1 Critical-1 scope
# decision: marker arms KEEP a real trained-rows battery surface at the booked scale.


def _marker_tree_subsample(rows: list[dict], seed: int) -> list[dict]:
    """Seeded proportional-by-row_kind subsample of a marker mix's TF rows to
    ``MARKER_TREE_ROWS``, mix order preserved (question_idx == mix row index)."""
    import random as _random

    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(r["row_kind"], []).append(r)
    frac = MARKER_TREE_ROWS / len(rows)
    rng = _random.Random(seed * 99991 + 1947)
    keep: list[dict] = []
    for kind in sorted(by_kind):
        grp = by_kind[kind]
        k = min(len(grp), max(1, round(frac * len(grp))))
        keep.extend(rng.sample(grp, k))
    keep.sort(key=lambda r: r["question_idx"])
    return keep


def unit_arm(cfg: Cfg, slug: str) -> None:
    """Per-arm P4 unit: TF trained-rows tree (consumed + full, trained + base),
    on-policy tree, 20-q panel (trained + base), TF fixed-pool margin — one
    merge, then reap (plan §4.4). Mix + manifest + consumed-set resolve BEFORE
    the CPU 7B merge (r1 Minor 5: a mix-contract miss costs seconds, not a
    merge). Marker arms: full tree subsampled to MARKER_TREE_ROWS; the
    consumed tree is SKIPPED (recorded) on a below-ceiling verdict rung where
    membership is shuffle-unknowable (r1 Critical 1)."""
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
    # Mix contract first (pre-merge): staging + manifest parse + consumed set.
    mix_dir = _stage_mix(cfg, slug)
    manifest_row_ids = _read_json(mix_dir / "consumption_manifest.json")["row_ids"]
    assert manifest_row_ids, (slug, "empty consumption manifest")
    consumed = _consumed_row_idxs(cfg, slug, step)
    merged = _merge_1947(cfg, slug, step)
    try:
        rows = _mix_rows_tf(cfg, slug, str(merged))
        subsampled_from = None
        if slug.startswith("mk-") and len(rows) > MARKER_TREE_ROWS:
            subsampled_from = len(rows)
            rows = _marker_tree_subsample(rows, cells.CELL_BY_SLUG[slug].seed)
        if cfg.smoke:
            rows = rows[:6]
        if consumed is None:
            rows_consumed = None  # marker below-ceiling rung — consumed tree skipped
        else:
            rows_consumed = [r for r in rows if r["question_idx"] in consumed]
            if not rows_consumed:  # smoke slices can miss the consumed set — keep ≥1 row
                rows_consumed = rows[: max(1, len(rows) // 4)]
        tdir = cfg.out_root / "battery" / "trained_rows" / slug
        extra = {
            "slug": slug,
            "verdict_step": step,
            "n_consumed": None if rows_consumed is None else len(rows_consumed),
            "consumed_tree": "skipped-unknowable" if rows_consumed is None else "captured",
        }
        if subsampled_from is not None:
            extra["marker_tree_subsample"] = {"n_from": subsampled_from, "n_to": len(rows)}

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
            BASE_MODEL,
            rows,
            slug,
            tdir / "pooled_base.pt",
            {**extra, "set": "full", **_kinds(rows)},
        )
        if rows_consumed is not None:
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
                rows_consumed,
                slug,
                tdir / "pooled_base_consumed.pt",
                {**extra, "set": "consumed", **_kinds(rows_consumed)},
            )
        if not slug.startswith("mk-"):
            _unit_onpolicy_and_panel(cfg, slug, entry, merged)
            _unit_margin(cfg, slug, entry, merged)
        _upload_tree(cfg, tdir, f"{DATA_PREFIX}/battery/trained_rows/{slug}")
        _atomic_json(done, {"step": step, **extra, **_meta()})
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
            assert consumed is not None, (slug, "dynamics slugs are content-sv (seam-logged)")
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
                # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient; prefix-scoped
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
            gen_dir.mkdir(parents=True, exist_ok=True)  # _append_shard writes here without mkdir
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
    floor + D verdict — the REAL issue1768_fit.fit_bare_n_cell per layer on
    the span-mean context pooling (the M0-comparability D read), PLUS the
    binding-directive LAST-TOKEN within-run M⁺ map per layer (directive v7
    items 2-3) and the r3 floors cross-check (recomputed M0/floor vs the
    staged #1768 r3 fits_bare_n values — never narrate D off divergent
    floors)."""
    import issue1768_fit as FIT

    sample = _stage_corpus_inputs(cfg)
    pinned_shas = {r.get("sha") for r in sample["rows"] if r.get("sha")}
    n_tv = int(sample["n_train"]) + int(sample["n_val"])
    test_shas = {r.get("sha") for r in sample["rows"][n_tv:] if r.get("sha")}
    import torch

    own = torch.load(
        cfg.out_root / "corpus_capture" / slug / "pooled.pt",
        map_location="cpu",
        weights_only=False,
    )
    got = set(own["row_sha"])
    missing = {s for s in pinned_shas if s not in got}
    missing_test = {s for s in test_shas if s not in got}
    # Plan §3 registers the STRICT set-check on TEST rows ("test-row shas ⊆
    # both sides" — the D/floor comparison grid is test-block-only); the 10%
    # tolerance below covers train/val attrition only (r1 Minor 3), and
    # fit_bare_n_cell additionally enforces its own qidx join (>=0.9 keep)
    # inside _join_pfx_cell.
    if not cfg.smoke and missing_test:
        raise RuntimeError(
            f"[i1947-fit] {slug}: {len(missing_test)}/{len(test_shas)} pinned TEST shas "
            "missing from the own store — test-grid sha-join violated (plan §3)"
        )
    if not cfg.smoke and len(missing) > 0.1 * len(pinned_shas):
        raise RuntimeError(
            f"[i1947-fit] {slug}: {len(missing)}/{len(pinned_shas)} pinned shas missing "
            "from the own store — sha-join violated (plan §3 row-coverage assert)"
        )
    results_dir = cfg.out_root / "fits"
    # #1947 slugs are absent from #1768's arm registry (X.all_arms), so
    # register the result-record method label at point of use — every #1947
    # content cell is LoRA (r32/a64 rsLoRA per plan §4.2; the fleet has no
    # full-FT cells). "lora" is the registry's exact vocabulary
    # (issue1768_cells.Arm.method: "lora" | "ft").
    FIT.EXTERNAL_ARM_METHOD.setdefault(slug, "lora")
    recs: dict[int, dict] = {}
    for layer in cfg.layers:
        t0 = time.time()
        rec = FIT.fit_bare_n_cell(cfg.out_root, results_dir, slug, layer, cfg.smoke)
        recs[layer] = rec
        print(
            f"[fit] {slug} L{layer} verdict={rec.get('map_change', {}).get('verdict')} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    _fit_lasttoken_arm(cfg, slug)
    _r3_floor_crosscheck(cfg, slug, recs)
    _upload_tree(cfg, results_dir, f"{DATA_PREFIX}/fits")


def _fit_lasttoken_arm(cfg: Cfg, slug: str) -> None:
    """Within-run LAST-TOKEN M⁺ map per layer (binding directive v7 items 2-3
    + v9: the PRIMARY last-token context summary is ``last_prompt`` — the
    final token of the generation-rendered prompt, the #1768 lasttoken-repool
    position — NOT the last user-content token; span-mean stays the
    M0-comparability D read). Fits last_prompt -> response ridge on the arm's
    OWN corpus store over the SAME pfx split, identity+bias + kNN attached
    (3584->3584, applicable). The base-side M0_lasttoken fits ONLY when the
    staged r1 base store carries a last_prompt arm; the D column is flagged
    ``no-M0-floor-available`` unless the Phase-0 probe resolved the #1768
    lasttoken_ctx re-run outputs (the directive's fallback clause)."""
    import issue1768_fit as FIT
    import numpy as np

    sample = _stage_corpus_inputs(cfg)
    pfx_by_src = {int(r["src_qidx"]): j for j, r in enumerate(sample["rows"])}
    own = FIT._load_store(cfg.out_root / "corpus_capture" / slug / "pooled.pt")
    base = FIT._load_store(cfg.out_root / "corpus_capture" / "base_content" / "pooled.pt")
    lasttoken_probe: bool | None = None  # None = probe report absent (unknown)
    probe_path = cfg.out_dir / "probe_report.json"
    if probe_path.exists():
        for prow in _read_json(probe_path).get("report", []):
            if prow.get("name") == "lasttoken_ctx":
                lasttoken_probe = bool(prow.get("resolved"))

    def _subset_lt(store: dict, layer: int):
        """(C_lasttoken, V_response, pfx qidx) for the pinned rows, or None
        when the store carries no last_prompt arm (r1 base stores predate
        the directive)."""
        if "last_prompt" not in store["arms"] or layer not in store["arms"]["last_prompt"]:
            return None
        keep = [i for i, q in enumerate(store["row_question_idx"]) if int(q) in pfx_by_src]
        qidx = np.asarray([pfx_by_src[int(store["row_question_idx"][i])] for i in keep])
        C = FIT._store_span_rows(store, "last_prompt", layer)[keep]
        V = FIT._store_span_rows(store, "response", layer)[keep]
        return C, V, qidx

    dev = FIT._device()
    out_dir = cfg.out_root / "fits" / "lasttoken"
    for layer in cfg.layers:
        dest = out_dir / f"{slug}_L{layer}_lasttoken.json"
        if dest.exists():
            continue
        own_lt = _subset_lt(own, layer)
        assert own_lt is not None, (slug, layer, "own store missing last_prompt (directive v7+v9)")
        C, V, qidx = own_lt
        tr, val, te = FIT._split_idx(FIT._pfx_split_from_qidx(qidx, sample))
        pred_te, meta, _payload = FIT._fit_map(C, V, tr, val, te, dev, allow_underdetermined=True)
        fits = {
            "Mplus_lasttoken": {
                **meta,
                **FIT._map_reads(pred_te, V[te]),
                "identity_bias": FIT._identity_bias_reads(C[tr], V[tr], C[te], V[te]),
            }
        }
        base_lt = _subset_lt(base, layer)
        if base_lt is None:
            fits["M0_lasttoken"] = {
                "status": "unavailable",
                "reason": "r1 base store carries no last_prompt arm",
            }
        else:
            C0, V0, q0 = base_lt
            tr0, val0, te0 = FIT._split_idx(FIT._pfx_split_from_qidx(q0, sample))
            pred0, meta0, _p0 = FIT._fit_map(
                C0, V0, tr0, val0, te0, dev, allow_underdetermined=True
            )
            fits["M0_lasttoken"] = {
                **meta0,
                **FIT._map_reads(pred0, V0[te0]),
                "identity_bias": FIT._identity_bias_reads(C0[tr0], V0[tr0], C0[te0], V0[te0]),
            }
        rec = {
            "arm_id": slug,
            "layer": int(layer),
            "condition": "bare_n_lasttoken",
            "input_span": "last_prompt",
            "n_rows": int(len(qidx)),
            "n_train": int(len(tr)),
            "n_val": int(len(val)),
            "n_test": int(len(te)),
            "underdetermined_n_lt_d": bool(len(tr) < C.shape[1]),
            "fits": fits,
            # Directive fallback clause: the within-run lasttoken read is the
            # MAP (fit + baselines); D-vs-floor stays with the span-mean read
            # unless the #1768 lasttoken re-run floors resolved at the probe.
            "map_change": {
                "D": None,
                "status": (
                    "r3-lasttoken-floors-resolved-on-hub"
                    if lasttoken_probe
                    else "no-M0-floor-available"
                ),
                "probe_lasttoken_ctx_resolved": lasttoken_probe,
            },
            "smoke": cfg.smoke,
            **_meta(),
        }
        _atomic_json(dest, rec)
        print(
            f"[fit-lt] {slug} L{layer} lasttoken r2={fits['Mplus_lasttoken']['heldout_r2']:.4f}",
            flush=True,
        )


R3_XCHECK_R2_TOL = 0.02  # |ΔR²| bound: recomputed M0 vs the r3 committed value
R3_XCHECK_FLOOR_RELTOL = 0.10  # relative floor_p95 bound
# Grounding: identical data (sha-joined pinned rows), λ grid, and FLOOR_SEED
# B=200 draws — residual divergence is cross-hardware fp reduction jitter (the
# ~1e-3-class bf16/GPU parity family), so these bounds are loose for hardware
# jitter and tight against a wrong sha set / n / λ regime. Beyond them the
# recomputed floors are NOT the r3 instrument and D verdicts must not be
# narrated off them (r1 adjudication of p5-m0-floors-recomputed-not-r3-loaded;
# the H5 concordance read keys on floor identity with the parent's floors).


def _r3_floor_crosscheck(cfg: Cfg, slug: str, recs: dict[int, dict]) -> None:
    """Cross-check the RECOMPUTED M0 R² + refit floor against the #1768 r3
    committed ``fits_bare_n`` values (staged from the probed Hub prefix; M0 +
    floor are base-store-derived, so any r3 arm's file carries the layer's
    values). Persists a per-layer tolerance report next to the fits and fails
    LOUD on divergence — demoted to informational under --smoke (toy-n fits
    cannot reproduce production anchors: the #1345 smoke-gate rule)."""
    import issue1768_cells as X1768
    from huggingface_hub import HfApi

    dest = cfg.out_root / "fits" / "r3_crosscheck" / f"{slug}.json"
    if dest.exists():
        # r2 Critical 3: the report is persisted BEFORE the fail-loud raise, so
        # on the one-retry _fan path a resumed unit re-enters here with a
        # persisted ``verdict: fail`` on disk. Only a clean verdict may
        # short-circuit — a divergence re-raises on every resume (a retry does
        # not clear a divergence); anything else recomputes (seconds).
        prior = _read_json(dest)
        prior_verdict = prior.get("verdict")
        if prior_verdict == "fail" and not cfg.smoke:
            raise RuntimeError(
                f"[i1947-fit] {slug}: persisted r3 cross-check report carries "
                f"verdict=fail ({prior.get('n_divergent')} divergent layer(s)) — "
                "D verdicts must not be narrated off divergent floors "
                "(see fits/r3_crosscheck; resume does not clear a divergence)"
            )
        if prior_verdict in ("pass", "informational-smoke"):
            return
    prefix = f"{X1768.HF_PREFIX}/on_target/eval_results/fits_bare_n"
    names = hub.retry_transient(
        lambda: [
            e.path
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient; prefix-scoped
            for e in HfApi().list_repo_tree(
                HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=False
            )
        ],
        what="list r3 fits_bare_n",
    )
    checks: list[dict] = []
    n_fail = 0
    for layer, rec in sorted(recs.items()):
        cands = sorted(n for n in names if n.endswith(f"_L{layer}.json"))
        if not cands:
            checks.append({"layer": layer, "status": "no-r3-file-for-layer", "ok": False})
            n_fail += 1
            continue
        local = cfg.out_root / "inputs" / "r3_fits_bare_n" / Path(cands[0]).name
        if not local.exists():
            hub.stage_hub_file(HF_DATA_REPO, cands[0], local, repo_type="dataset")
        r3 = _read_json(local)
        r2_new = float(rec["fits"]["M0"]["heldout_r2"])
        r2_r3 = float(r3["fits"]["M0"]["heldout_r2"])
        fl_new = float(rec["map_change"]["floor_p95"])
        fl_r3 = float(r3["map_change"]["floor_p95"])
        fl_rel = abs(fl_new - fl_r3) / max(abs(fl_r3), 1e-12)
        row = {
            "layer": layer,
            "r3_file": cands[0],
            "m0_r2_recomputed": r2_new,
            "m0_r2_r3": r2_r3,
            "m0_r2_absdiff": abs(r2_new - r2_r3),
            "floor_p95_recomputed": fl_new,
            "floor_p95_r3": fl_r3,
            "floor_p95_reldiff": fl_rel,
            "n_test_recomputed": int(rec["n_test"]),
            "n_test_r3": int(r3.get("n_test", -1)),
            "ok": abs(r2_new - r2_r3) <= R3_XCHECK_R2_TOL and fl_rel <= R3_XCHECK_FLOOR_RELTOL,
        }
        if not row["ok"]:
            n_fail += 1
        checks.append(row)
    verdict = "informational-smoke" if cfg.smoke else ("pass" if n_fail == 0 else "fail")
    _atomic_json(
        dest,
        {
            "slug": slug,
            "verdict": verdict,
            "n_divergent": n_fail,
            "r2_tol": R3_XCHECK_R2_TOL,
            "floor_reltol": R3_XCHECK_FLOOR_RELTOL,
            "checks": checks,
            **_meta(),
        },
    )
    print(f"[fit-xcheck] {slug}: {verdict} ({n_fail} divergent layer(s))", flush=True)
    if verdict == "fail":
        raise RuntimeError(
            f"[i1947-fit] {slug}: recomputed M0/floors diverge from the r3 committed "
            f"fits_bare_n beyond tolerance on {n_fail} layer(s) — D verdicts must not "
            "be narrated off divergent floors (see fits/r3_crosscheck)"
        )


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
        excl = {
            int(x) for x in os.environ.get("EPM_EXCLUDE_GPUS", "").split(",") if x.strip().isdigit()
        }
        ids = [i for i in ids if i not in excl]
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


def _runnable_fits(fits: list[str], failed_units: list[str]) -> tuple[list[str], list[str]]:
    """Gate each fit unit on ITS OWN corpus input's success (r1 Critical 1: a
    single failed capture unit must not zero out P5). Returns (runnable,
    skipped) — a ``fit:<slug>`` whose ``corpus:<slug>`` capture failed is
    SKIPPED (recorded, fail-loud at phase exit); every other fit runs."""
    failed_corpus = {u.split(":", 1)[1] for u in failed_units if u.startswith("corpus:")}
    runnable = [u for u in fits if u.split(":", 1)[1] not in failed_corpus]
    skipped = [u for u in fits if u.split(":", 1)[1] in failed_corpus]
    return runnable, skipped


def _battery_sentinel(cfg: Cfg, payload: dict) -> dict:
    """Poller-conformant sentinel envelope for the battery done-file — mirrors
    the worker ``_finalize`` shape (``poll_pipeline._SENTINEL_REQUIRED_KEYS``:
    sentinel_schema_version / kind / version; r1 Critical 2 — the bare payload
    was warn-skipped by every drain tick)."""
    import issue1090_fu3_worker as fu3w

    return {
        "sentinel_schema_version": fu3w.SENTINEL_SCHEMA_VERSION,
        "kind": "epm:smoke-result" if cfg.smoke else "epm:results",
        "version": 1,  # drain-side rewrite derives max+1 (#1095)
        "task_id": ISSUE,
        "gate": "i1947-battery",
        "blocks_pipeline": not cfg.smoke,
        "by": f"issue{ISSUE}-battery-dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": bool(cfg.smoke),
        "note": json.dumps(payload, ensure_ascii=False),
        "payload": payload,
    }


def _fan(
    cfg: Cfg,
    argv_base: list[str],
    gpus: list[int],
    queue: list[str],
    on_complete=None,
) -> tuple[list[str], dict[str, float], bool, list[str]]:
    """Work-conserving fan-out; ONE retry per failed unit (transient absorber
    — r1 Critical 1 fix note). ``on_complete(unit, wall_h, walls) -> bool``
    may HALT dispatch (the corpus re-projection gate, r1 Minor 6): pending
    units are DRAINED into the returned ``not_started`` list (r2 Critical 2 —
    a non-empty ``pending`` after halt would spin the scheduler loop forever
    once ``running`` empties), the halted retry-insert is suppressed, and
    running units drain. Returns (failed, unit_walls, halted, not_started)."""
    pending = list(queue)
    running: dict[int, tuple[subprocess.Popen, str, float]] = {}
    failed: list[str] = []
    retried: set[str] = set()
    unit_walls: dict[str, float] = {}
    not_started: list[str] = []
    n_total = len(pending)
    n_done = 0
    halted = False
    while pending or running:
        for gpu in gpus:
            if halted or gpu in running or not pending:
                continue
            unit = pending.pop(0)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            cmd = argv_base + ["--phase", "unit", "--unit", unit, "--gpu-id", str(gpu)]
            log = cfg.out_root / "logs" / f"unit_{unit.replace(':', '_')}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("ab") as fh:
                proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[gpu] = (proc, unit, time.time())
            print(f"[dispatch] gpu{gpu} <- {unit}", flush=True)
        time.sleep(3 if not cfg.smoke else 0.2)
        for gpu in list(running):
            proc, unit, t_start = running[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            del running[gpu]
            wall_h = (time.time() - t_start) / 3600.0
            if rc != 0 and unit not in retried and not halted:
                retried.add(unit)
                # r2 Minor: count the first attempt's wall toward the
                # re-projection basis (len(walls) transiently counts the
                # in-flight retry as done — bounded by len(gpus), negligible
                # vs dropping the wall entirely).
                unit_walls[unit] = unit_walls.get(unit, 0.0) + wall_h
                pending.insert(0, unit)  # per-unit outputs are resume-idempotent
                print(f"[dispatch] {unit} rc={rc} — ONE retry", flush=True)
                continue
            n_done += 1
            unit_walls[unit] = unit_walls.get(unit, 0.0) + wall_h
            if rc != 0:
                failed.append(unit)
                log = cfg.out_root / "logs" / f"unit_{unit.replace(':', '_')}.log"
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                print(f"[dispatch] {unit} FAILED rc={rc} (post-retry); log tail:", flush=True)
                for line in tail:
                    print(f"    {line}", flush=True)
            elif on_complete is not None and not halted and on_complete(unit, wall_h, unit_walls):
                halted = True
                print(f"[dispatch] unit {n_done}/{n_total} {unit} done rc=0", flush=True)
                not_started.extend(pending)
                pending.clear()  # r2 Critical 2: drain so the loop can terminate
                print(
                    f"[dispatch] HALT: compute gate fired — draining running units; "
                    f"{len(not_started)} queued unit(s) marked not-started",
                    flush=True,
                )
            else:
                print(f"[dispatch] unit {n_done}/{n_total} {unit} done rc=0", flush=True)
    return failed, unit_walls, halted, not_started


def cmd_capture_fit(cfg: Cfg, argv_base: list[str]) -> int:
    """Work-conserving per-unit subprocess fan-out over the physical GPUs
    (CVD pinned in the LAUNCHER env — the #545 clobber rule). Every unit gets
    ONE retry; fits are gated PER ARM on their own corpus input (the #825
    store-before-fit ordering holds per arm — r1 Critical 1: one failed
    capture no longer zeroes out P5); the phase completes healthy units and
    exits nonzero with a terminal per-unit failure report when any unit
    failed or was skipped (fail-loud, never silent-skip)."""
    units = _enumerate_units(cfg)
    captures = [u for u in units if not u.startswith("fit:")]
    fits = [u for u in units if u.startswith("fit:")]
    gpus = _physical_gpus()
    print(f"[dispatch] {len(captures)} capture units + {len(fits)} fits on {len(gpus)} GPUs")

    # Plan §9 P4 in-run pilot gate: run the FIRST capture unit alone at
    # production shape, re-project the phase wall, and HALT >2× the booked
    # GPU-h with a report JSON + a DISTINCT rc (never an anonymous crash).
    pilot_walls: dict[str, float] = {}
    if captures and not cfg.smoke:
        pilot = captures[0]
        t0 = time.time()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpus[0])}
        cmd = argv_base + ["--phase", "unit", "--unit", pilot, "--gpu-id", str(gpus[0])]
        log = cfg.out_root / "logs" / "pilot_unit.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        per_unit_h = (time.time() - t0) / 3600.0
        projected_gpu_h = per_unit_h * len(captures + fits)
        report = {
            "pilot_unit": pilot,
            "pilot_rc": rc,
            "per_unit_h": per_unit_h,
            "n_units": len(captures + fits),
            "projected_gpu_h": projected_gpu_h,
            "plan_gpu_h": PLAN_P4P5_GPU_H,
            "ratio": projected_gpu_h / PLAN_P4P5_GPU_H,
            **_meta(),
        }
        _atomic_json(cfg.out_dir / "pilot_gate_report.json", report)
        print(f"[dispatch] pilot gate: {json.dumps(report)}", flush=True)
        if rc != 0:
            print(f"[dispatch] pilot unit {pilot} FAILED rc={rc} — halting", flush=True)
            return 1
        if projected_gpu_h > 2 * PLAN_P4P5_GPU_H:
            print(
                f"[dispatch] PILOT GATE HALT: projected {projected_gpu_h:.1f} GPU-h > "
                f"2x plan {PLAN_P4P5_GPU_H} — rc={PILOT_GATE_RC} (report JSON written)",
                flush=True,
            )
            return PILOT_GATE_RC
        pilot_walls[pilot] = per_unit_h
        captures = captures[1:]  # pilot unit already done (resume-idempotent anyway)

    n_units_total = len(captures) + len(fits) + len(pilot_walls)
    reproject_fired = {"done": False}

    def _corpus_reprojection(unit: str, wall_h: float, walls: dict[str, float]) -> bool:
        """Second projection checkpoint at the FIRST completed corpus unit (r1
        Minor 6): corpus/fit units are ~4-5x the arm basis (plan §9), so the
        arm-pilot extrapolation under-projects; re-project the remaining wall
        with the MEASURED corpus basis and HALT >2x plan (same artifact-routed
        report + PILOT_GATE_RC convention as the pilot gate)."""
        if cfg.smoke or reproject_fired["done"] or not unit.startswith("corpus:"):
            return False
        reproject_fired["done"] = True
        done_h = sum(walls.values()) + sum(pilot_walls.values())
        n_done = len(walls) + len(pilot_walls)
        projected = done_h + wall_h * max(0, n_units_total - n_done)
        report = {
            "reprojection_unit": unit,
            "corpus_unit_h": wall_h,
            "done_gpu_h": done_h,
            "n_done": n_done,
            "n_units": n_units_total,
            "projected_gpu_h": projected,
            "plan_gpu_h": PLAN_P4P5_GPU_H,
            "ratio": projected / PLAN_P4P5_GPU_H,
            **_meta(),
        }
        _atomic_json(cfg.out_dir / "pilot_gate_report_corpus.json", report)
        print(f"[dispatch] corpus re-projection: {json.dumps(report)}", flush=True)
        return report["ratio"] > 2

    failed_caps, _cap_walls, halted, skipped_caps = _fan(
        cfg, argv_base, gpus, captures, on_complete=_corpus_reprojection
    )
    skipped_fits: list[str] = []
    failed_fits: list[str] = []
    if halted:
        skipped_fits = list(fits)
        print("[dispatch] COMPUTE GATE HALT — fits not started (report JSON written)", flush=True)
    else:
        if failed_caps:
            print(
                f"[dispatch] {len(failed_caps)} capture unit(s) failed post-retry — running "
                "fits for arms with intact inputs (#825 per-arm store-before-fit ordering)",
                flush=True,
            )
        runnable, skipped_fits = _runnable_fits(fits, failed_caps)
        failed_fits, _fit_walls, _, _ = _fan(cfg, argv_base, gpus, runnable)
    failed = failed_caps + failed_fits
    if failed or skipped_fits or skipped_caps:
        print("[dispatch] TERMINAL UNIT-FAILURE REPORT (fail-loud, never silent-skip):", flush=True)
        for u in failed:
            print(f"    FAILED  {u}", flush=True)
        for u in skipped_caps:
            print(f"    SKIPPED {u} (not started — compute-gate halt)", flush=True)
        for u in skipped_fits:
            print(f"    SKIPPED {u} (corpus input failed or compute-gate halt)", flush=True)
    if halted:
        status = "halted_compute_gate"
    elif failed or skipped_fits:
        status = "failed"
    else:
        status = "done"
    payload = {
        "issue": ISSUE,
        "phase": "battery",
        "status": status,
        "n_units": len(units),
        "failed_units": failed,
        "skipped_capture_units": skipped_caps,
        "skipped_fit_units": skipped_fits,
        **_meta(),
    }
    sentinel_dir = cfg.sentinel_dir or SENTINEL_DIR_DEFAULT
    try:
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            sentinel_dir / f"issue-{ISSUE}-battery-done.json", _battery_sentinel(cfg, payload)
        )
    except OSError as e:  # VM smoke has no /workspace — record loud, don't crash
        print(f"[dispatch] sentinel write skipped ({e})", flush=True)
    _phase("done", status=status, failed=len(failed), skipped=len(skipped_fits))
    if halted:
        return PILOT_GATE_RC
    return 0 if not (failed or skipped_fits) else 1


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
        FIT._load_store,  # r2 last-token fit + r3 cross-check consumers
        FIT._store_span_rows,
        FIT._pfx_split_from_qidx,
        FIT._split_idx,
        FIT._fit_map,
        FIT._map_reads,
        FIT._identity_bias_reads,
        FIT._device,
        fu3w.SENTINEL_SCHEMA_VERSION,  # r2 battery sentinel envelope (Critical 2)
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
