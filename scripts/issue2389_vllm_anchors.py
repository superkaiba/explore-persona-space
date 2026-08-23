"""Issue #2389 — vLLM anchor leg (plan §4.7 item 4; FAIL-OPEN behind the parity gate).

Three legs, dispatched per GPU worker (CUDA_VISIBLE_DEVICES pinned by the
launcher; one engine per GPU, TP=1 — 27.78B bf16 fits one H100/H200 card):

- ``--leg parity`` (workers 0..parity_workers-1, at dispatch t0): the
  pre-registered 3-cell parity protocol. Writes ``gates/vllm_cells.json``
  claiming the parity cells FIRST (their HF rest generation is owned by this
  leg — "the HF side of the gate doubles as those 3 cells' production
  anchors", so run.py's rest batch must exclude them), then per claimed cell:
  (sweep 1, engine up) vLLM generation of the cell's FULL context set at
  K=ANCHOR_DRAWS, temp 1.0, the cell's TABLE cap (recalibration deliberately
  NOT applied — both parity sides run at MATCHED caps) →
  ``vllm_parity/vllm_parity_<cell>_w{w}.jsonl`` (gate-only artifact: never in
  the ``anchors_*`` consumer glob space, no capture pt) + immediate upload;
  (sweep 2, engine torn down, HF model loaded) HF generation of the cell's
  REST contexts (batch id ``parity_<cell>`` → ``anchors_parity_<cell>_w{w}``
  shards, production-grade: text-persist → capture → enrich → done manifest,
  all via the inherited ``_run_anchor_batch``) + immediate jsonl upload so
  the VM judge can pair both sides in the P2 window. An engine
  import/init failure is RECORDED (``gates/vllm_engine_status.json``) and the
  HF sub-leg still runs — FAIL-OPEN by design: the parity cells' production
  anchors must exist either way.
- ``--leg claim`` (worker 0, after its parity leg): polls the WRITE repo for
  the judge fork's ``vllm_parity_report.json``; on verdict PASS extends
  ``gates/vllm_cells.json`` to every bank cell. Routing is PER CELL at CLAIM
  time (B8 r1 review): run.py's rest workers re-read the claim set every
  scan and skip claimed cells, so a PASS landing mid-P2 re-routes exactly
  the REMAINING (not-yet-HF-claimed) cells — a late PASS is work-conserving,
  never inert; cells HF already claimed/finished stay HF.
- ``--leg production`` (all workers, after run.py's anchors phase releases
  the GPU): owns = claimed cells (``gates/vllm_cells.json``) − parity cells
  − cells the HF rest queue already completed; per claimed cell (claim-file
  queue, DP across workers): vLLM generation of the cell's REST contexts
  (gate contexts are always HF — run.py's gate batch) →
  text-persist ``anchors/anchors_vllm_<cell>_w{w}.jsonl`` + gen-done
  sentinel; then engine teardown, HF model load, and the inherited
  teacher-forced ``capture_answer_states`` pass (M1-v two-sweep: HF model and
  vLLM engine are NEVER co-resident) → ``va_anchors_vllm_<cell>_w{w}.pt`` +
  done manifest via ``_finalize_anchor_batch``.

Filename pins (plan step-3 pin 1): the engine marker rides the BATCH-ID
position — ``anchors_vllm_<cell>_w{w}.jsonl`` / ``va_anchors_vllm_<cell>_w{w}.pt``
— inside BOTH consumer globs (judge ``anchors_*.jsonl``, analysis
``va_anchors_*.pt``) and structurally DISJOINT from every HF-written shard
name; ``_shard_stem_index``'s strict stem allowlist makes these cell-grained
shards immune to the width sweeps by construction. Parity-side vLLM rollouts
live under ``vllm_parity/`` (outside the consumer globs — they are gate
evidence, never production anchors; the judge's duplicate
(context_id, draw, engine) assert is the fail-loud backstop).

Engine config (plan §4.7 item 4): explicit ``max_model_len`` >= longest
target prompt + max cap + margin (computed from the REAL rendered prompts —
the inherited-rig gotcha), ``enable_prefix_caching=False`` +
``enforce_eager=True`` default-on (the #1092/#1324 H100 IMA/hang pre-launch
checklist; env-liftable), chunked dispatch <= EPM_2389_VLLM_CHUNK (500)
prompts per ``generate`` call, ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` pinned
at module top (#628 fork-poisoning).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

# #628: vLLM reads this at import time; the pre-LLM() path touches
# transformers/tokenizer helpers, so fork() would poison the EngineCore.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import issue2389_run as R  # noqa: E402

logger = logging.getLogger("issue2389_vllm")

# Plan §4.7 item 4(i): 3 cells spanning the parent's realized cap-hit range —
# fact_user_name 0.00%, persona_prompted 7.50%, filler_swap 24.72%.
PARITY_CELLS: tuple[str, ...] = ("fact_user_name", "persona_prompted", "filler_swap")


def _validated_vllm_chunk(raw: str) -> int:
    """``EPM_2389_VLLM_CHUNK`` clamped-by-refusal to the registered 1..500
    band (plan §4.7 item 4: chunked dispatch <= 500 prompts/call — a larger
    override silently violates the memory/stability bound; codex r1 minor)."""
    v = int(raw)
    if not 1 <= v <= 500:
        raise ValueError(f"EPM_2389_VLLM_CHUNK={v} outside the registered band [1, 500]")
    return v


VLLM_CHUNK = _validated_vllm_chunk(os.environ.get("EPM_2389_VLLM_CHUNK", "500"))
ENFORCE_EAGER = os.environ.get("EPM_2389_VLLM_ENFORCE_EAGER", "1") == "1"
PREFIX_CACHING = os.environ.get("EPM_2389_VLLM_PREFIX_CACHING", "0") == "1"
GPU_MEM_UTIL = float(os.environ.get("EPM_2389_VLLM_GPU_MEM_UTIL", "0.90"))
MAX_MODEL_LEN_MARGIN = 64
REPORT_POLL_S = float(os.environ.get("EPM_2389_PARITY_POLL_S", "60"))
PARITY_REPORT_REMOTE = f"{R.HF_PREFIX}/analysis_tensors/gates/vllm_parity_report.json"


@dataclass(frozen=True)
class CellBlock:
    """One bank cell as a claim-queue unit (slug/key contract of run.py's queue)."""

    cell: str
    leg: str  # "parity_vllm" | "parity_hf" | "prod"

    @property
    def slug(self) -> str:
        return f"{self.leg}_{self.cell}"

    @property
    def key(self) -> str:
        return f"vllm-anchors:{self.leg}:{self.cell}"


# ── target enumeration ────────────────────────────────────────────────


def _cell_targets(cfg: R.RunConfig) -> tuple[dict[str, dict], dict[str, list[str]], set[str]]:
    """``(contexts, per-cell ORDERED context ids, gate-slice id set)``.

    Uses the inherited `_anchor_context_order` so the smoke filter and the
    gate/rest split match run.py exactly; per-cell lists preserve the
    gate-first ordering (gate ids lead, rest ids follow)."""
    gate_ids, rest_ids, contexts = R._anchor_context_order(cfg)
    by_cell: dict[str, list[str]] = {}
    for cid in list(gate_ids) + list(rest_ids):
        by_cell.setdefault(contexts[cid]["cell"], []).append(cid)
    return contexts, by_cell, set(gate_ids)


# ── engine lifecycle ──────────────────────────────────────────────────


def _engine_status_path(cfg: R.RunConfig) -> Path:
    return cfg.gates_dir / "vllm_engine_status.json"


def _load_tokenizer(cfg: R.RunConfig):
    """Tokenizer-only load at the pinned revision (no weights)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)


def _max_model_len_for(tok, contexts: dict[str, dict], cids: list[str], caps: list[int]) -> int:
    """Explicit ``max_model_len`` >= longest target prompt + max cap + margin
    (plan §4.7 item 4 — the inherited-rig `max_model_len` gotcha: compute from
    the REAL rendered prompts, never a pinned default)."""
    longest = max(len(R.BANK29.context_token_ids_2389(tok, contexts[c])) for c in cids)
    return longest + max(caps) + MAX_MODEL_LEN_MARGIN


def _build_engine(cfg: R.RunConfig, max_model_len: int):
    """vLLM engine at the pinned revision; raises on failure (caller records)."""
    from vllm import LLM

    logger.info(
        "[vllm] engine init: model=%s rev=%s max_model_len=%d enforce_eager=%s "
        "prefix_caching=%s gpu_mem_util=%.2f",
        cfg.model_id,
        cfg.model_revision,
        max_model_len,
        ENFORCE_EAGER,
        PREFIX_CACHING,
        GPU_MEM_UTIL,
    )
    return LLM(
        model=cfg.model_id,
        revision=cfg.model_revision,
        tokenizer_revision=cfg.model_revision,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        enable_prefix_caching=PREFIX_CACHING,
        enforce_eager=ENFORCE_EAGER,
        gpu_memory_utilization=GPU_MEM_UTIL,
    )


def _teardown_engine(llm) -> None:
    """Full reap (KV + EngineCore subprocess) before any HF model load."""
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)


def _eos_ids(cfg: R.RunConfig, tok) -> list[int]:
    """EOS id list from the pinned generation config (falls back to tokenizer).

    The 27B pinned `generation_config.json` carries ``eos_token_id`` as a
    LIST ([248046, 248044]); vLLM stops on these and excludes them from
    ``output.text`` — matching the HF path's first-EOS truncation +
    ``skip_special_tokens=True`` decode."""
    from transformers import GenerationConfig

    gen_cfg = GenerationConfig.from_pretrained(cfg.model_id, revision=cfg.model_revision)
    eos = gen_cfg.eos_token_id
    if eos is None:
        eos = tok.eos_token_id
    if eos is None:
        raise RuntimeError("no eos_token_id on generation config or tokenizer — refusing")
    return [int(e) for e in (eos if isinstance(eos, list | tuple) else [eos])]


# ── generation ────────────────────────────────────────────────────────


def _vllm_generate_cell_rows(
    cfg: R.RunConfig,
    llm,
    tok,
    contexts: dict[str, dict],
    cids: list[str],
    cell: str,
    cap: int,
    draws: int,
    eos_ids: list[int],
    gate_id_set: set[str],
) -> list[dict]:
    """vLLM rollouts for ONE cell — row schema mirrors `_generate_anchor_rows`
    (telemetry parity, Div 8/§4.6) with ``engine="vllm"`` + native telemetry
    extras; ``n_completion_tokens``/``cap_hit`` land at capture-enrichment
    time on production shards (same retokenized basis as the HF path)."""
    from vllm import SamplingParams

    sp = SamplingParams(
        n=draws,
        temperature=float(R.ANCHOR_TEMPERATURE),
        top_p=1.0,
        top_k=-1,
        max_tokens=cap,
        stop_token_ids=eos_ids,
        seed=cfg.seed_base,
    )
    rows: list[dict] = []
    t0 = time.monotonic()
    for start in range(0, len(cids), VLLM_CHUNK):
        chunk = cids[start : start + VLLM_CHUNK]
        prompts = [R.BANK29.render_context_2389(tok, contexts[c]) for c in chunk]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        for cid, out in zip(chunk, outs, strict=True):
            ctx = contexts[cid]
            assert len(out.outputs) == draws, (cid, len(out.outputs), draws)
            for i, comp in enumerate(out.outputs):
                rows.append(
                    {
                        "context_id": cid,
                        "cell": ctx["cell"],
                        "value_id": ctx["value_id"],
                        "carrier": ctx["carrier"],
                        "draw": i,
                        "seed": cfg.seed_base,  # request seed (vLLM n-draw request)
                        "temperature": float(R.ANCHOR_TEMPERATURE),
                        "gate_slice": cid in gate_id_set,
                        "max_new_tokens": cap,
                        "engine": "vllm",
                        "vllm_finish_reason": comp.finish_reason,
                        "vllm_n_tokens_native": len(comp.token_ids),
                        "text": comp.text,
                    }
                )
        logger.info(
            "[vllm:%s] unit %d/%d contexts cap=%d elapsed=%.1fs",
            cell,
            min(start + VLLM_CHUNK, len(cids)),
            len(cids),
            cap,
            time.monotonic() - t0,
        )
    return rows


# ── done predicates (worker-independent: any worker may own a cell) ───


def _parity_vllm_done(cfg: R.RunConfig, regime_fp: str, cell: str) -> bool:
    for m in cfg.manifest_dir.glob(f"vllm_parity_{cell}_w*_done.json"):
        rec = json.loads(m.read_text())
        if rec.get("regime_fp") == regime_fp and Path(rec["jsonl"]).exists():
            return True
    return False


def _anchor_cell_done(
    cfg: R.RunConfig, regime_fp: str, batch: str, expected_draws: int | None = None
) -> bool:
    """Worker-independent per-cell done read — DELEGATES to run.py's own
    predicate (regime + draws + artifacts + row-count checks), so the HF and
    vLLM sides share ONE definition of done."""
    return R._anchor_cell_done(cfg, regime_fp, batch, expected_draws)


def _prod_gen_done(cfg: R.RunConfig, regime_fp: str, cell: str) -> bool:
    for m in cfg.manifest_dir.glob(f"anchors_vllm_{cell}_w*_gen_done.json"):
        rec = json.loads(m.read_text())
        if rec.get("regime_fp") == regime_fp and Path(rec["jsonl"]).exists():
            return True
    return False


# ── vllm_cells.json (the run.py exclusion payload) ────────────────────


def _write_vllm_cells(cfg: R.RunConfig, cells: list[str], reason: str) -> None:
    cfg.gates_dir.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(
        cfg.gates_dir / "vllm_cells.json",
        {
            "cells": sorted(set(cells)),
            "reason": reason,
            "ts": datetime.now(UTC).isoformat(),
            "repro": R._repro(cfg),
        },
    )
    logger.info("[vllm] gates/vllm_cells.json <- %d cells (%s)", len(set(cells)), reason)


# ── leg: parity ───────────────────────────────────────────────────────


def leg_parity(cfg: R.RunConfig) -> int:
    """Both engine sides of the 3 parity cells (vLLM full sets; HF rest sets)."""
    _manifest, bank_sha = R.bank_manifest_and_sha()
    regime_fp = R.regime_fingerprint(cfg, bank_sha)
    contexts, by_cell, gate_id_set = _cell_targets(cfg)
    cells = [c for c in PARITY_CELLS if by_cell.get(c)]
    if not cells:
        logger.warning(
            "[parity] no parity-cell contexts under this config (smoke?) — nothing to do"
        )
        return R.RC_OK
    # Claim the parity cells' HF REST ownership FIRST — before any run.py
    # worker can reach the rest-entry routing freeze — so their production
    # anchors are generated exactly once (by this leg's HF sweep).
    _write_vllm_cells(cfg, list(cells), "parity-hf-ownership")
    draws = 2 if cfg.smoke else cfg.anchor_draws
    caps = {c: R._resolve_cap(cfg, c, None) for c in cells}  # table caps, MATCHED both sides
    parity_dir = cfg.out_root / "vllm_parity"
    parity_dir.mkdir(parents=True, exist_ok=True)

    # Sweep 1 — vLLM side: FULL context set per cell (gate + rest; the vLLM
    # side is gate evidence only, so gate-context coverage duplicates no HF
    # production spend).
    engine_err: str | None = None
    tok = _load_tokenizer(cfg)
    try:
        eos_ids = _eos_ids(cfg, tok)
        mml = _max_model_len_for(
            tok, contexts, [cid for c in cells for cid in by_cell[c]], list(caps.values())
        )
        llm = _build_engine(cfg, mml)
    except Exception as e:  # FAIL-OPEN by plan §4.7 item 4: record + proceed on HF
        engine_err = f"{type(e).__name__}: {e}"
        R._write_json_atomic(
            _engine_status_path(cfg),
            {
                "status": "failed",
                "stage": "engine-init",
                "error": engine_err,
                "traceback": traceback.format_exc(),
                "ts": datetime.now(UTC).isoformat(),
                "repro": R._repro(cfg),
            },
        )
        logger.error("[parity] vLLM engine init FAILED (fail-open, HF sub-leg proceeds): %s", e)
        R._upload_dir(
            cfg,
            cfg.gates_dir,
            f"{R.HF_PREFIX}/analysis_tensors/gates",
            [_engine_status_path(cfg).name],
        )
    if engine_err is None:
        R._write_json_atomic(
            _engine_status_path(cfg),
            {"status": "ok", "ts": datetime.now(UTC).isoformat(), "repro": R._repro(cfg)},
        )
        R._upload_dir(
            cfg,
            cfg.gates_dir,
            f"{R.HF_PREFIX}/analysis_tensors/gates",
            [_engine_status_path(cfg).name],
        )

        def _run_vllm_cell(block: CellBlock) -> None:
            cell = block.cell
            rows = _vllm_generate_cell_rows(
                cfg,
                llm,
                tok,
                contexts,
                by_cell[cell],
                cell,
                caps[cell],
                draws,
                eos_ids,
                gate_id_set,
            )
            jsonl = parity_dir / f"vllm_parity_{cell}_w{cfg.worker_index}.jsonl"
            R._write_jsonl_atomic(jsonl, rows)
            R._write_json_atomic(
                cfg.manifest_dir / f"vllm_parity_{cell}_w{cfg.worker_index}_done.json",
                {
                    "regime_fp": regime_fp,
                    "cell": cell,
                    "worker_index": cfg.worker_index,
                    "n_contexts": len(by_cell[cell]),
                    "draws": draws,
                    "n_rows": len(rows),
                    "max_new_tokens": caps[cell],
                    "jsonl": str(jsonl),
                    "engine": "vllm",
                    "repro": R._repro(cfg),
                },
            )
            # Early upload: the VM judge pairs both sides in the P2 window.
            R._upload_dir(
                cfg,
                parity_dir,
                f"{R.HF_PREFIX}/raw_completions/vllm_parity",
                [jsonl.name],
            )

        blocks = [CellBlock(cell=c, leg="parity_vllm") for c in cells]
        R.run_claim_queue(
            cfg,
            blocks,
            regime_fp,
            "vllm_parity_gen",
            _run_vllm_cell,
            is_done=lambda _root, b, fp, _ns: _parity_vllm_done(cfg, fp, b.cell),
        )
        _teardown_engine(llm)
        llm = None  # release the last reference (rebind, never `del` a closure-captured name)

    # Sweep 2 — HF side: the parity cells' REST contexts, production-grade,
    # via the inherited `_run_anchor_batch` (text-persist -> capture ->
    # enrich -> done manifest). Table caps (matched with the vLLM side).
    # Cells whose contexts are ALL gate-slice have no HF-rest work (run.py's
    # gate batch owns them) and are excluded up front — a never-done block
    # would spin the claim queue forever.
    rest_by_cell = {c: [cid for cid in by_cell[c] if cid not in gate_id_set] for c in cells}
    hf_cells = [c for c in cells if rest_by_cell[c]]
    for c in cells:
        if c not in hf_cells:
            logger.warning("[parity:%s] no REST contexts (all gate-slice) — no HF side", c)
    if hf_cells:
        model, tok = R.load_model_and_tokenizer(cfg)

        def _run_hf_cell(block: CellBlock) -> None:
            cell = block.cell
            res = R._run_anchor_batch(
                cfg,
                model,
                tok,
                contexts,
                rest_by_cell[cell],
                draws,
                f"parity_{cell}",
                regime_fp,
                recalibrated=None,  # matched-cap comparison: table caps both sides
            )
            R._upload_dir(
                cfg,
                cfg.anchors_dir,
                f"{R.HF_PREFIX}/raw_completions/anchors",
                [res["jsonl"].name],
            )

        blocks = [CellBlock(cell=c, leg="parity_hf") for c in hf_cells]
        R.run_claim_queue(
            cfg,
            blocks,
            regime_fp,
            "vllm_parity_hf",
            _run_hf_cell,
            is_done=lambda _root, b, fp, _ns: _anchor_cell_done(cfg, fp, f"parity_{b.cell}", draws),
        )
    logger.info("[phase=vllm_parity_done] worker=%d engine_err=%s", cfg.worker_index, engine_err)
    return R.RC_OK


# ── leg: claim (poll verdict, extend the exclusion set pre-freeze) ────


def _read_parity_report(cfg: R.RunConfig) -> dict | None:
    """The judge fork's report — local out_root copy first, then the write repo."""
    local = cfg.gates_dir / "vllm_parity_report.json"
    if local.exists():
        return json.loads(local.read_text())
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    if not retry_transient(
        # HUB_VERIFY_RETRY_EXEMPT: call is wrapped in hub.retry_transient (this expr)
        lambda: HfApi().file_exists(
            R.HF_DATA_WRITE_REPO, PARITY_REPORT_REMOTE, repo_type="dataset"
        ),
        what="parity-report file_exists poll",
    ):
        return None
    path = retry_transient(
        lambda: hf_hub_download(
            repo_id=R.HF_DATA_WRITE_REPO,
            filename=PARITY_REPORT_REMOTE,
            repo_type="dataset",
        ),
        what="parity-report download",
    )
    return json.loads(Path(path).read_text())


def leg_claim(cfg: R.RunConfig, timeout_s: float) -> int:
    """Poll for the parity verdict; on PASS extend vllm_cells.json to every
    bank cell. B8 (r1 review): routing is per cell at CLAIM time — run.py's
    rest workers re-read the claim set every scan, so a PASS landing mid-P2
    re-routes exactly the not-yet-claimed cells (work-conserving; a late PASS
    is never inert). FAIL/timeout leave the parity-only claim in place
    (fail-open: HF generates everything)."""
    t0 = time.monotonic()
    while True:
        report = _read_parity_report(cfg)
        if report is not None:
            verdict = str(report.get("verdict", "")).upper()
            logger.info("[claim] parity report verdict=%s", verdict)
            if verdict == "PASS":
                _write_vllm_cells(cfg, list(R.BANK.all_cells()), "parity-pass-full-claim")
            else:
                logger.info("[claim] verdict != PASS — vLLM path stays disabled (fail-open)")
            return R.RC_OK
        if time.monotonic() - t0 > timeout_s:
            logger.warning(
                "[claim] timeout (%.0fs) with no verdict — claim stays parity-only", timeout_s
            )
            return R.RC_OK
        time.sleep(REPORT_POLL_S)


# ── leg: production (claimed cells, minus parity, minus HF-done) ──────


def _owned_rest_cells(
    cfg: R.RunConfig,
    regime_fp: str,
    draws: int,
    by_cell: dict[str, list[str]],
    gate_id_set: set[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Production-leg ownership (B8): claimed cells (gates/vllm_cells.json)
    minus parity cells, minus cells the HF rest queue already completed
    (per-cell claim-time freeze: a claim extension landing after an HF worker
    claimed/finished a cell leaves it HF), minus cells with no REST contexts.
    Returns (rest contexts per owned cell, cells this leg must generate)."""
    claimed = R._vllm_claimed_cells(cfg)
    owned = sorted(claimed - set(PARITY_CELLS))
    rest_by_cell = {c: [cid for cid in by_cell.get(c, []) if cid not in gate_id_set] for c in owned}
    gen_cells = [
        c
        for c in owned
        if rest_by_cell[c] and not _anchor_cell_done(cfg, regime_fp, f"rest_{c}", draws)
    ]
    return rest_by_cell, gen_cells


def leg_production(cfg: R.RunConfig, routing_wait_s: float) -> int:
    """vLLM generation + HF capture for the claimed, not-HF-done rest cells.

    B8 (r1 review): ownership is per cell at claim time — this leg runs
    post-anchors, so every rest cell is either HF-done (an HF worker claimed
    it before the parity PASS extended the claim set) or belongs to this
    leg. HF-done cells are skipped (work-conserving); crash-orphaned partial
    shards from EITHER engine are quarantined under the held claim before
    regeneration."""
    _manifest, bank_sha = R.bank_manifest_and_sha()
    regime_fp = R.regime_fingerprint(cfg, bank_sha)
    claim_path = cfg.gates_dir / "vllm_cells.json"
    t0 = time.monotonic()
    while not claim_path.exists():
        if time.monotonic() - t0 > routing_wait_s:
            logger.warning(
                "[prod] no vllm_cells.json after %.0fs — nothing claimed", routing_wait_s
            )
            return R.RC_OK
        logger.info("[prod] waiting for the parity leg's claim file (%s)", claim_path.name)
        time.sleep(R.CLAIM_POLL_S)
    contexts, by_cell, gate_id_set = _cell_targets(cfg)
    draws = 2 if cfg.smoke else cfg.anchor_draws
    recal = R._load_cap_recalibration(cfg)  # rest runs at recalibrated caps (item 1)
    rest_by_cell, gen_cells = _owned_rest_cells(cfg, regime_fp, draws, by_cell, gate_id_set)
    if not gen_cells:
        logger.info("[prod] every claimed rest cell is HF-done or empty — nothing to generate")
        return R.RC_OK
    logger.info("[prod] %d vLLM-owned rest cell(s): %s", len(gen_cells), gen_cells)
    caps = {c: R._resolve_cap(cfg, c, recal) for c in gen_cells}
    tok = _load_tokenizer(cfg)
    eos_ids = _eos_ids(cfg, tok)

    # Sweep 1 — vLLM generation, text-persisted per cell (#779) + gen-done
    # sentinel (NOT the final done manifest — M1-v: that lands after capture).
    mml = _max_model_len_for(
        tok, contexts, [cid for c in gen_cells for cid in rest_by_cell[c]], list(caps.values())
    )
    llm = _build_engine(cfg, mml)  # a production-leg engine failure fails LOUD (post-PASS)

    def _run_prod_gen(block: CellBlock) -> None:
        cell = block.cell
        # Reclaim hygiene under the held claim (is_done read False): sweep
        # crash-orphaned partials from BOTH engines — a dead vLLM worker's
        # partial gen shard (would trip capture's exactly-one assert) and a
        # dead HF worker's text-persisted partial (would double rows under
        # the judge's anchors_*.jsonl glob).
        R._quarantine_orphan_cell_shards(cfg, f"vllm_{cell}")
        R._quarantine_orphan_cell_shards(cfg, f"rest_{cell}")
        rows = _vllm_generate_cell_rows(
            cfg,
            llm,
            tok,
            contexts,
            rest_by_cell[cell],
            cell,
            caps[cell],
            draws,
            eos_ids,
            gate_id_set,
        )
        jsonl = cfg.anchors_dir / f"anchors_vllm_{cell}_w{cfg.worker_index}.jsonl"
        R._write_jsonl_atomic(jsonl, rows)
        R._write_json_atomic(
            cfg.manifest_dir / f"anchors_vllm_{cell}_w{cfg.worker_index}_gen_done.json",
            {
                "regime_fp": regime_fp,
                "cell": cell,
                "worker_index": cfg.worker_index,
                "n_contexts": len(rest_by_cell[cell]),
                "draws": draws,
                "n_rows": len(rows),
                "max_new_tokens": caps[cell],
                "jsonl": str(jsonl),
                "engine": "vllm",
                "repro": R._repro(cfg),
            },
        )
        R._upload_dir(cfg, cfg.anchors_dir, f"{R.HF_PREFIX}/raw_completions/anchors", [jsonl.name])

    blocks = [CellBlock(cell=c, leg="prod") for c in gen_cells]
    R.run_claim_queue(
        cfg,
        blocks,
        regime_fp,
        "vllm_prod_gen",
        _run_prod_gen,
        is_done=lambda _root, b, fp, _ns: _prod_gen_done(cfg, fp, b.cell),
    )
    _teardown_engine(llm)
    llm = None  # release the last reference (rebind, never `del` a closure-captured name)

    # Sweep 2 — HF capture over every generated-but-uncaptured cell shard
    # (any worker may capture any cell's shard: shared FS + claim queue).
    model, tok = R.load_model_and_tokenizer(cfg)

    def _run_prod_capture(block: CellBlock) -> None:
        cell = block.cell
        batch = f"vllm_{cell}"
        shards = sorted(cfg.anchors_dir.glob(f"anchors_{batch}_w*.jsonl"))
        assert shards, f"[prod:{cell}] gen-done but no jsonl shard on disk"
        # Exactly one gen shard per cell (cell-grained claims); capture it
        # under ITS recorded worker index so jsonl/pt/manifest names agree.
        assert len(shards) == 1, f"[prod:{cell}] {len(shards)} shards — expected 1: {shards}"
        shard = shards[0]
        w = int(shard.stem.rsplit("_w", 1)[1])
        rows = [json.loads(line) for line in shard.open(encoding="utf-8") if line.strip()]
        cap_cfg = replace(cfg, worker_index=w)
        flat_ctx = [R.BANK29.context_token_ids_2389(tok, contexts[r["context_id"]]) for r in rows]
        flat_text = [r["text"] for r in rows]
        n_contexts = len({r["context_id"] for r in rows})
        R._finalize_anchor_batch(
            cap_cfg, model, tok, rows, flat_ctx, flat_text, batch, regime_fp, n_contexts, draws
        )
        R._upload_dir(cfg, cfg.anchors_dir, f"{R.HF_PREFIX}/raw_completions/anchors", [shard.name])

    R.run_claim_queue(
        cfg,
        blocks,
        regime_fp,
        "vllm_prod_capture",
        _run_prod_capture,
        is_done=lambda _root, b, fp, _ns: _anchor_cell_done(cfg, fp, f"vllm_{b.cell}", draws),
    )
    logger.info("[phase=vllm_production_done] worker=%d cells=%s", cfg.worker_index, gen_cells)
    return R.RC_OK


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2389 vLLM anchor leg (plan §4.7 item 4).")
    # required-unless-import-check (validated post-parse so `--import-check`
    # can run with no run config).
    ap.add_argument("--leg", choices=("parity", "claim", "production"), default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gen-batch", type=int, default=None, help="HF sub-leg chunk size")
    # B2 (r1 review): the SAME enum as the callee `issue2389_run.parse_args`
    # (`hf|local-mirror|none`) — the prior `full` default could not bind
    # through `_compose_run_cfg` and every real invocation died in argparse.
    ap.add_argument("--upload", choices=("hf", "local-mirror", "none"), default="hf")
    # B9: threaded verbatim into run.py's parser — the HF sub-legs write into
    # the SAME anchors family as run.py's workers, and share_prefill_armed is
    # part of regime_fingerprint, so the mode MUST match the HF workers'.
    ap.add_argument("--share-prefill", choices=("off", "auto"), default="off")
    ap.add_argument("--claim-timeout-s", type=float, default=7200.0)
    ap.add_argument("--routing-wait-s", type=float, default=1800.0)
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve deferred imports + args-attribute completeness, then exit 0",
    )
    return ap.parse_args(argv)


def _compose_run_cfg(args: argparse.Namespace) -> R.RunConfig:
    """Build the shared RunConfig through run.py's OWN parser (never a
    hand-built Namespace — the reused-module contract), so dirs, regime
    fingerprint, caps, and upload gating match the HF workers exactly."""
    argv = [
        "--phase",
        "anchors",
        "--out-root",
        str(args.out_root),
        "--worker-index",
        str(args.worker_index),
        "--num-workers",
        str(args.num_workers),
        "--upload",
        args.upload,
        "--share-prefill",
        args.share_prefill,
    ]
    if args.smoke:
        argv.append("--smoke")
    if args.gen_batch is not None:
        argv += ["--gen-batch", str(args.gen_batch)]
    cfg = R.build_config(R.parse_args(argv))
    if args.leg == "claim" and R._pilot_gpu_name() is None:
        # Round-6 (concern pilot-reuse-runtime-domain): the claim leg is a
        # CPU-only poll (dispatch.sh runs it under CUDA_VISIBLE_DEVICES="")
        # that only reads the parity verdict and writes gates/vllm_cells.json
        # — it generates NOTHING, so neither gen_batch nor the share-prefill
        # family decision enters its outputs. Its runtime domain (CPU) can
        # NEVER match a GPU pilot report, so the round-5-J adoption
        # validation below would FOREIGN-raise here on every production
        # `all` run the moment the pilot report exists (dispatch.sh:552-553)
        # — and the dispatcher discards the detached claim pid's rc, so the
        # death was silent. Explicit, LOGGED skip — a deliberate narrow
        # carve-out, not a silent fallback; the GPU legs (parity/production,
        # and a claim leg that does see a GPU) resolve BOTH below,
        # byte-identically to before.
        logger.info(
            "[compose] leg=claim on a CPU host (no CUDA device) — skipping pilot "
            "gen_batch adoption + share-prefill resolution (CPU runtime domain "
            "cannot match a GPU pilot report; the claim leg writes no generation "
            "artifacts)"
        )
        return cfg
    # B9 (r1 review): this leg's HF sub-legs write shards into the SAME
    # anchors family as run.py's workers, and `regime_fingerprint` now covers
    # gen_batch + share_prefill_armed — resolve BOTH exactly as phase_anchors
    # does (pilot adoption; the family-FROZEN share-prefill decision), or the
    # two sides' done predicates stop recognizing each other's manifests.
    cfg = R._adopt_pilot_gen_batch(cfg)
    cfg = R._resolve_share_prefill(cfg, "vllm_anchors")
    return cfg


def _import_check() -> int:
    """Axis-1 import resolution: execute every deferred import + the
    args-attribute completeness assert (module-level per the #1739
    in-function import-shadowing gotcha)."""
    from explore_persona_space.analysis.representation_shift import (  # noqa: F401
        _reap_vllm_engine,
    )
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
    from explore_persona_space.orchestrate.hub import retry_transient  # noqa: F401
    from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
    from transformers import AutoTokenizer, GenerationConfig  # noqa: F401
    from vllm import LLM, SamplingParams  # noqa: F401

    assert_args_attributes_defined(__file__)
    print("[import-check] OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.import_check:
        return _import_check()
    if args.leg is None or args.out_root is None:
        raise SystemExit("--leg and --out-root are required (unless --import-check)")
    cfg = _compose_run_cfg(args)
    logger.info(
        "[phase=vllm_anchors] leg=%s worker=%d/%d smoke=%s out_root=%s",
        args.leg,
        cfg.worker_index,
        cfg.num_workers,
        cfg.smoke,
        cfg.out_root,
    )
    if args.leg == "parity":
        return leg_parity(cfg)
    if args.leg == "claim":
        return leg_claim(cfg, args.claim_timeout_s)
    return leg_production(cfg, args.routing_wait_s)


if __name__ == "__main__":
    raise SystemExit(main())
