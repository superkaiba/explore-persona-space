"""Issue #2564 pod driver — anchor generation (PA) + teacher-forced capture (PB) + embed (PC).

Port of the #2215 dbe driver (``40cc2bb752:scripts/issue2215_dbe_run.py``) adapted to the
frozen #2564 minimal-pair bank (``src/explore_persona_space/experiments/issue2564/bank2564.py``),
carrying the branch fixes the plan §3.7 names: ``4c5bc86e8b`` (apply_map finiteness — consumed
by the analysis unit, signature-pinned in ``tests/test_issue2564_driver.py``), ``62c686d845``
(bf16 two-bar parity calibration), ``0ae62170f2`` (coverage-gate licensing → gates demoted to
informational under smoke/tiny, per plan §7/§8).

Phases (plan §3.8):
  A (``pa_generate``)  — 984 contexts × K=10 draws = 9,840 anchor rollouts (temp 1.0,
      seed_base 42, gen_batch 16, max_new_tokens 2048; cap-hit fraction per cell with a
      >2% whole-cell re-gen at 4096). Per-cell jsonl shards + resume-safe done manifests;
      chunk-grain intra-cell checkpointing (``.partial`` sidecar). Built-in pilot gate
      (warmup + 2 production-shape timed chunks; ceiling 7.0 h; refusal = report JSON +
      rc=7, demoted to informational under --smoke/--tiny). Sentinel:
      ``<out-root>/anchors_done.json``. HF: ``issue2564_minpair/raw_completions/anchors/``.
  B (``pb_capture``)   — teacher-forced v_A capture (span mean + tail-inclusive twin,
      layers 14/19/26, fp16; ``capture_answer_states(..., tail_inclusive=True,
      return_boundaries=True)`` — the MF-A hunk) for every anchor row, plus v_C
      context-end capture (fp32) for all bank contexts; gate-4 3-way record parity +
      bf16 two-bar cosine parity (refusal rc=8, demoted under smoke/tiny). Sentinel:
      ``<out-root>/va2564_uploaded.json``. HF: ``issue2564_minpair/analysis_tensors/``.
  C (``pc_embed``)     — Qwen3-Embedding-8B text-space embedding of all 9,840 anchor
      rollouts (plan §9: on-pod 1×H100), run as a SUBPROCESS of
      ``scripts/issue2564_embed.py`` (vLLM pooling engine — own process so PB's HF
      teardown and vLLM's spawn env stay isolated; ``env={**os.environ}``). Sentinel:
      ``<embed-out-root>/embed_uploaded.json`` (``embed_done.local.json`` under
      ``--upload none``), VERIFIED by this driver before terminal success. Its rc=7
      pilot-gate refusal propagates as this driver's rc=7. Skipped with a loud
      warning under ``--tiny`` (no CPU vLLM engine); explicit ``--phase C --tiny``
      raises.

Pod-side contracts: single process per phase (PC in its own subprocess);
``sys.exit(rc)`` after explicit flushes (no vLLM in THIS process); ``[phase=...]``
breadcrumbs + per-unit progress lines; NO ``task.py`` shellouts; sentinels are
plain JSON files the VM poller reads.

Workload command (plan §9): ``uv run python scripts/issue2564_run.py --phase all
--out-root /workspace/eps2564 --upload hf``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import — thread caps + credentials (code-style.md)

import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2162_run as R  # noqa: E402  (MF-A host: capture_answer_states + io helpers)

from explore_persona_space.analysis.extraction import (  # noqa: E402
    extract_layer_activations,
)
from explore_persona_space.atomic_io import (  # noqa: E402  (#2336 process-unique temps)
    save_pt_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from explore_persona_space.experiments.issue1415.steering import generate_batch  # noqa: E402
from explore_persona_space.experiments.issue2564 import bank2564 as BK  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded  # noqa: E402

logger = logging.getLogger("issue2564")

# ── constants ─────────────────────────────────────────────────────────────

ISSUE = 2564
HF_DATA_REPO = os.environ.get("EPM_2564_DATA_WRITE_REPO", "superkaiba1/explore-persona-space-data")
HF_PREFIX = "issue2564_minpair"

MAP_LAYERS = (14, 19, 26)
HIDDEN = 3584
N_LAYERS = 28

ANCHOR_DRAWS = 10
ANCHOR_TEMPERATURE = 1.0  # matches R.ANCHOR_TEMPERATURE (plan §11)
ANCHOR_MAX_NEW = 2048
REGEN_MAX_NEW = 4096
CAP_HIT_REGEN_FRAC = 0.02
SEED_BASE = 42

PILOT_WALL_CEIL_H = 7.0  # plan §9: PA planned 3.5 h, ceiling 2×

# bf16 parity bars (62c686d845 calibration, gotchas.md two-bar entry):
# span-mean class (v_A span + tail twin) = 0.9999; single-position class (v_C
# context-end) = flat >= 0.995 + early-layer >= 0.999 where "early" falls back
# to the FIRST captured layer (dbe `or [0]` idiom — layers (14,19,26) capture
# no layer 0-3, so layer 14 carries the sharp-bug bar; demoted at smoke n).
PARITY_COS_MIN = 0.9999
PARITY_COS_MIN_SINGLEPOS_FLAT = 0.995
PARITY_COS_MIN_SINGLEPOS_EARLY = 0.999
PARITY_N_ANSWER_ROWS = 3
PARITY_N_CONTEXTS = 3

RC_OK = 0
RC_PILOT_GATE = 7
RC_PARITY_GATE = 8

SMOKE_CELLS = ("register", "query")  # plan §8 (bank realizes the query cell as "query")
SMOKE_CARRIERS = ("c01", "c02", "c03")
SMOKE_DRAWS = 2

PHASES = ("A", "B", "C")

_PHASE_COMPLETION_RECORDS: dict[str, tuple[str, ...]] = {
    "A": ("anchors_done.json", "manifests/pilot_gate_report.json"),
    "B": ("va2564_uploaded.json", "manifests/parity_gate_report.json"),
    # C: the embed subprocess owns its records under ITS out-root (fingerprint-gated
    # chunk resume + sentinel); --force does not reach it — rerun with a fresh
    # embed out-root to force.
    "C": (),
}

CAP_HIT_BASIS = "retokenized_completion_len >= max_new_tokens"


# ── config ────────────────────────────────────────────────────────────────


@dataclass
class Cfg2564:
    """Run config; duck-types RunConfig for R.load_model_and_tokenizer /
    R.capture_answer_states (model_id/tiny/hidden/n_layers/layers/device/capture_batch)."""

    phase: str
    out_root: Path
    log_dir: Path
    values_path: Path | None
    smoke: bool
    tiny: bool
    cells: tuple[str, ...] | None
    carriers: tuple[str, ...] | None
    draws: int
    gen_batch: int
    max_new_tokens: int
    seed_base: int
    upload: str
    force: bool
    model_id: str = BK.MODEL_ID
    model_revision: str = "unresolved"
    hidden: int = HIDDEN
    n_layers: int = N_LAYERS
    layers: tuple[int, ...] = MAP_LAYERS
    capture_batch: int = 8
    device: str = "cuda"
    # PRE-rebind out-root (the CLI value): PC derives the embed subprocess's own
    # out-root from it so the embed script's OWN smoke rebind is the single
    # authority for smoke isolation (no double rebind).
    raw_out_root: Path | None = None
    _values_sha_cache: str | None = field(default=None, repr=False)

    @property
    def hf_prefix(self) -> str:
        return HF_PREFIX + ("/smoke" if (self.smoke or self.tiny) else "")

    @property
    def anchors_dir(self) -> Path:
        return self.out_root / "raw_completions" / "anchors"

    @property
    def va_dir(self) -> Path:
        return self.out_root / "analysis_tensors" / "va2564"

    @property
    def vc_dir(self) -> Path:
        return self.out_root / "analysis_tensors" / "vc2564"

    @property
    def manifest_dir(self) -> Path:
        return self.out_root / "manifests"

    @property
    def quarantine_dir(self) -> Path:
        return self.out_root / "quarantine"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for the pod driver (per-issue phase-dispatch driver — argparse by convention)."""
    ap = argparse.ArgumentParser(description="issue2564 minimal-pair pod driver")
    ap.add_argument("--phase", choices=("all", "A", "B", "C"), default="all")
    ap.add_argument("--out-root", default="/workspace/eps2564")
    ap.add_argument("--log-dir", default="/workspace/logs")
    ap.add_argument("--values", default=None, help="override bank2564_values.json path")
    ap.add_argument("--cells", default=None, help="csv cell subset (default: all; smoke default)")
    ap.add_argument("--carriers", default=None, help="csv carrier subset")
    ap.add_argument("--draws", type=int, default=None, help="rollouts per context (K)")
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=ANCHOR_MAX_NEW)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--upload", choices=("hf", "none"), default="hf")
    ap.add_argument("--smoke", action="store_true", help="plan §8 smoke slice + /smoke HF prefix")
    ap.add_argument("--tiny", action="store_true", help="from-config CPU model (pytest-only)")
    ap.add_argument("--force", action="store_true", help="quarantine phase-level records at entry")
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> Cfg2564:
    """Resolve the CLI namespace into a Cfg2564 (smoke slicing + out-root rebind)."""
    smoke, tiny = bool(args.smoke), bool(args.tiny)
    cells = tuple(s.strip() for s in args.cells.split(",") if s.strip()) if args.cells else None
    carriers = (
        tuple(s.strip() for s in args.carriers.split(",") if s.strip()) if args.carriers else None
    )
    if smoke or tiny:
        cells = cells or SMOKE_CELLS
        carriers = carriers or SMOKE_CARRIERS
    draws = (
        args.draws if args.draws is not None else (SMOKE_DRAWS if (smoke or tiny) else ANCHOR_DRAWS)
    )
    out_root = Path(args.out_root)
    if smoke or tiny:
        # dbe smoke-root rebind: generated artifacts land under smoke_<name>; staged
        # INPUTS (the packaged values JSON) are read-only and are NOT rebound.
        out_root = out_root.parent / f"smoke_{out_root.name}"
    cfg = Cfg2564(
        phase=args.phase,
        out_root=out_root,
        log_dir=Path(args.log_dir),
        values_path=Path(args.values) if args.values else None,
        smoke=smoke,
        tiny=tiny,
        cells=cells,
        carriers=carriers,
        draws=int(draws),
        gen_batch=int(args.gen_batch),
        max_new_tokens=int(args.max_new_tokens),
        seed_base=int(args.seed_base),
        upload=args.upload,
        force=bool(args.force),
        capture_batch=int(args.capture_batch),
        raw_out_root=Path(args.out_root),
    )
    if tiny:
        cfg.hidden = 64
        cfg.n_layers = 4
        cfg.layers = (1, 2, 3)
        cfg.device = "cpu"
    return cfg


def _assert_call_kwargs() -> None:
    """MF-A leg 4: every kwarg this driver passes to reused callees must exist in the
    callee's live signature — `--import-check` cannot catch kwarg mismatches (#2223)."""
    import inspect

    checks = [
        (
            R.capture_answer_states,
            {"payloads", "positions", "tail_inclusive", "return_boundaries"},
        ),
        (
            generate_batch,
            {"n", "hook", "max_new_tokens", "temperature", "seed_base", "render_fn", "ids_fn"},
        ),
    ]
    for fn, kwargs in checks:
        params = set(inspect.signature(fn).parameters)
        missing = sorted(kwargs - params)
        assert not missing, (
            f"{fn.__module__}.{fn.__qualname__} missing kwargs {missing} "
            "(is the 15501f33b2 return_boundaries hunk applied to scripts/issue2162_run.py?)"
        )


# ── identity / fingerprints / io ──────────────────────────────────────────


def _values_sha(cfg: Cfg2564) -> str:
    """sha256 of the bit-exact values JSON (regime-fp input, #1336 machine-stable rule:
    file BYTES are safe to hash — never a recomputed float array)."""
    if cfg._values_sha_cache is None:
        p = cfg.values_path or (Path(BK.__file__).parent / BK.VALUES_FILENAME)
        cfg._values_sha_cache = hashlib.sha256(p.read_bytes()).hexdigest()
    return cfg._values_sha_cache


def _regime_fp(cfg: Cfg2564, extra: dict | None = None) -> str:
    """16-hex fingerprint of the GENERATING PARAMETERS (never recomputed floats)."""
    params: dict = {
        "issue": ISSUE,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "values_sha": _values_sha(cfg),
        "smoke": cfg.smoke,
        "tiny": cfg.tiny,
        "cells": sorted(cfg.cells) if cfg.cells else None,
        "carriers": sorted(cfg.carriers) if cfg.carriers else None,
        "draws": cfg.draws,
        "gen_batch": cfg.gen_batch,  # output-affecting: per-batch reseeding + batch composition
        "max_new_tokens": cfg.max_new_tokens,
        "seed_base": cfg.seed_base,
        "layers": list(cfg.layers),
        "capture_batch": cfg.capture_batch,
    }
    if extra:
        params.update(extra)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def _cell_fp(cfg: Cfg2564, phase: str, cell: str) -> str:
    return _regime_fp(cfg, {"phase": phase, "cell": cell})


def _gate_fp(cfg: Cfg2564, gate: str) -> str:
    return _regime_fp(cfg, {"gate": gate})


def _repro(cfg: Cfg2564, phase: str) -> dict:
    """Reproducibility metadata block (git sha + dirty flag + env versions + run identity)."""
    from importlib.metadata import version as _pkg_version

    meta = as_metadata_dict(git_provenance(), phase=phase)
    meta.update(
        {
            "issue": ISSUE,
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
            "values_sha256": _values_sha(cfg),
            "smoke": cfg.smoke,
            "tiny": cfg.tiny,
            "seed_base": cfg.seed_base,
            "torch_version": torch.__version__,
            "transformers_version": _pkg_version("transformers"),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
    )
    return meta


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _read_jsonl(path: Path, tolerate_torn_tail: bool = False) -> list[dict]:
    """Text-mode JSONL read (never splitlines(), gotchas.md). ``tolerate_torn_tail``
    drops ONLY a torn final line (crash mid-append on a .partial sidecar)."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            if tolerate_torn_tail and i == len(lines) - 1:
                logger.warning("dropping torn final line of %s (crash-interrupted append)", path)
                break
            raise
    return rows


def _flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().flatten().double()
    b64 = b.detach().flatten().double()
    return float((a64 @ b64) / (a64.norm() * b64.norm() + 1e-12))


def _resolve_model_revision(cfg: Cfg2564) -> str:
    """Pin main → resolved sha ONCE per run (#2061); fail loud outside tiny mode."""
    if cfg.tiny:
        return "unresolved-tiny"
    from huggingface_hub import HfApi

    sha = HfApi().model_info(cfg.model_id).sha
    assert sha, f"could not resolve model revision for {cfg.model_id}"
    return sha


def _invalidate_phase_records(cfg: Cfg2564, phase: str) -> None:
    """--force: quarantine phase-level completion records (atomic replace); per-cell
    done manifests stay honored."""
    if not cfg.force:
        return
    for rel in _PHASE_COMPLETION_RECORDS[phase]:
        p = cfg.out_root / rel
        if p.exists():
            cfg.quarantine_dir.mkdir(parents=True, exist_ok=True)
            dest = cfg.quarantine_dir / f"{time.time_ns()}.{p.name}"
            os.replace(p, dest)
            logger.info("[force] quarantined %s -> %s", p, dest)


def _filter_bank(
    bank: dict, cells: tuple[str, ...] | None, carriers: tuple[str, ...] | None
) -> dict:
    """Subset the bank's contexts (and pairs whose BOTH endpoints survive). Empty
    selection over the committed bank raises (gotchas.md empty-selection rule).

    Producer seam (r2 blocker 1): ``BK.build_bank`` returns ``contexts`` as a
    ``dict[str, dict]`` (id -> context) — this driver consumes the LIST shape
    everywhere downstream, so the dict is normalized to ``list(values())`` HERE,
    at the single driver-side seam. The returned bank's ``contexts`` is a list."""
    contexts = bank["contexts"]
    if isinstance(contexts, dict):
        contexts = list(contexts.values())
    if cells:
        contexts = [c for c in contexts if c["cell"] in cells]
    if carriers:
        contexts = [c for c in contexts if c["carrier"] in carriers]
    if not contexts:
        raise RuntimeError(
            f"empty context selection after filter cells={cells} carriers={carriers}"
        )
    kept = {c["id"] for c in contexts}
    pairs = [p for p in bank["pairs"] if p["a"] in kept and p["b"] in kept]
    return {
        **bank,
        "contexts": contexts,
        "pairs": pairs,
        "n_contexts": len(contexts),
        "n_pairs": len(pairs),
    }


# ── PA: anchor generation ─────────────────────────────────────────────────


def _pilot_path(cfg: Cfg2564) -> Path:
    return cfg.manifest_dir / "pilot_gate_report.json"


def _pilot_state(cfg: Cfg2564, total_rows: int) -> dict:
    """Resume-aware pilot state: a regime-matched prior PASS (or demoted) report skips
    re-gating; a prior non-demoted refusal re-times."""
    rep = _read_json(_pilot_path(cfg))
    evaluated = (
        rep is not None
        and rep.get("regime_fp") == _gate_fp(cfg, "pa_pilot")
        and (rep.get("verdict") == "proceed" or rep.get("demoted"))
    )
    return {
        "total_rows": total_rows,
        "warm_done": False,
        "warm_rows": 0,
        "warm_wall": 0.0,
        "timed_chunks": 0,
        "timed_rows": 0,
        "timed_wall": 0.0,
        "evaluated": evaluated,
    }


def _pilot_eval(cfg: Cfg2564, pilot: dict) -> None:
    """Evaluate + persist the PA pilot gate. Refusal = report JSON + rc=7 (distinct rc,
    never bare rc=1 — gotchas.md pilot-gate routing); demoted under smoke/tiny."""
    rows = pilot["timed_rows"] or pilot["warm_rows"]
    wall = pilot["timed_wall"] if pilot["timed_rows"] else pilot["warm_wall"]
    if rows <= 0:
        return
    per_row_s = wall / rows
    projected_h = per_row_s * pilot["total_rows"] / 3600.0
    demoted = cfg.smoke or cfg.tiny
    verdict = "proceed" if projected_h <= PILOT_WALL_CEIL_H else "refuse"
    report = {
        "gate": "pa_pilot",
        "regime_fp": _gate_fp(cfg, "pa_pilot"),
        "n_timed_rows": rows,
        "timed_from": "timed_chunks" if pilot["timed_rows"] else "warmup_chunk",
        "gen_batch": cfg.gen_batch,
        "wall_s": wall,
        "per_row_s": per_row_s,
        "total_rows": pilot["total_rows"],
        "projected_wall_h": projected_h,
        "ceiling_h": PILOT_WALL_CEIL_H,
        "verdict": verdict,
        "demoted": demoted,
        "repro": _repro(cfg, "pa-pilot"),
    }
    write_json_atomic(_pilot_path(cfg), report)
    pilot["evaluated"] = True
    logger.info(
        "[pilot] per_row_s=%.3f projected=%.2fh ceiling=%.1fh verdict=%s demoted=%s",
        per_row_s,
        projected_h,
        PILOT_WALL_CEIL_H,
        verdict,
        demoted,
    )
    if verdict == "refuse" and not demoted:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(RC_PILOT_GATE)


def _pilot_note(cfg: Cfg2564, pilot: dict, rows: int, wall: float) -> None:
    if pilot["evaluated"]:
        return
    if not pilot["warm_done"]:
        pilot["warm_done"] = True
        pilot["warm_rows"] = rows
        pilot["warm_wall"] = wall
        return
    pilot["timed_chunks"] += 1
    pilot["timed_rows"] += rows
    pilot["timed_wall"] += wall
    if pilot["timed_chunks"] >= 2:
        _pilot_eval(cfg, pilot)


def _gen_row(
    cfg: Cfg2564,
    ctx: dict,
    ctx_len: int,
    n_eot: int,
    draw: int,
    chunk: int,
    text: str,
    n_comp: int,
    max_new: int,
) -> dict:
    """Generation-side per-row record incl. the gate-4 span fields (compared EXACTLY
    against the capture path's own ``boundaries`` records in PB)."""
    return {
        "context_id": ctx["id"],
        "cell": ctx["cell"],
        "kind": ctx["kind"],
        "value_id": ctx["value_id"],
        "carrier": ctx["carrier"],
        "form": ctx["form"],
        "draw": draw,
        "seed": cfg.seed_base + draw,
        "chunk": chunk,
        "temperature": ANCHOR_TEMPERATURE,
        "max_new_tokens": max_new,
        "ctx_len": ctx_len,
        "n_completion_tokens_gen": n_comp,
        "span_start": ctx_len,
        "span_end": ctx_len + n_comp,
        "tail_end": ctx_len + n_comp + n_eot,
        "cap_hit": R.cap_hit(n_comp, max_new),
        "cap_hit_basis": CAP_HIT_BASIS,
        "text": text,
    }


def _generate_cell(
    cfg: Cfg2564,
    model,
    tok,
    eot_ids: list[int],
    cell: str,
    ctxs: list[dict],
    pilot: dict,
    max_new: int,
) -> list[dict]:
    """Generate all draws for one cell, chunk-grain checkpointed to a ``.partial``
    sidecar (excluded from the ``*.jsonl`` upload glob) with chunk-grain resume.
    Per-batch reseeding (generate_batch does torch.manual_seed(seed_base+i) per draw
    per call) makes chunk resume seed-consistent with a fresh run given identical
    gen_batch (gen_batch is in the regime fp).

    Resume keying (r2 blocker 2): the partial's FIRST line is a header carrying
    the cell regime fp (seed_base / cells / carriers / gen_batch / values sha /
    this call's max_new — the code-style resume-keying rule, #722 r3); a partial
    with a missing header or a mismatched fp is quarantined, never adopted. Each
    adopted chunk's context_id COMPOSITION is additionally verified against the
    current chunk split, so a re-scoped run can never adopt wrong-context rows."""
    part = cfg.anchors_dir / f"anchors_{cell}.max{max_new}.partial"
    part_fp = _regime_fp(cfg, {"phase": "A", "cell": cell, "max_new_call": max_new})
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    chunks = [ctxs[i : i + cfg.gen_batch] for i in range(0, len(ctxs), cfg.gen_batch)]
    prior: list[dict] = []
    if part.is_file():
        raw = _read_jsonl(part, tolerate_torn_tail=True)
        header = raw[0] if raw else None
        if (
            header is not None
            and header.get("partial_header")
            and header.get("regime_fp") == part_fp
        ):
            prior = raw[1:]
        else:
            cfg.quarantine_dir.mkdir(parents=True, exist_ok=True)
            dest = cfg.quarantine_dir / f"{time.time_ns()}.{part.name}"
            os.replace(part, dest)
            logger.warning(
                "[anchors:%s] quarantined stale .partial (missing header or regime-fp "
                "mismatch) -> %s",
                cell,
                dest,
            )
    if not part.is_file():
        with part.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"partial_header": 1, "regime_fp": part_fp}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    by_chunk: dict[int, list[dict]] = {}
    for r in prior:
        by_chunk.setdefault(int(r["chunk"]), []).append(r)
    complete = {
        ci
        for ci, rs in by_chunk.items()
        if ci < len(chunks)
        and len(rs) == len(chunks[ci]) * cfg.draws
        and {r["context_id"] for r in rs} == {c["id"] for c in chunks[ci]}
    }
    rows: list[dict] = [r for ci in sorted(complete) for r in by_chunk[ci]]
    if complete:
        logger.info(
            "[anchors:%s] resumed %d/%d chunks from %s", cell, len(complete), len(chunks), part.name
        )
    n_eot = len(eot_ids)
    t_cell = time.time()
    for ci, chunk in enumerate(chunks):
        if ci in complete:
            continue
        t0 = time.time()
        results = generate_batch(
            model,
            tok,
            chunk,
            n=cfg.draws,
            hook=None,
            max_new_tokens=max_new,
            temperature=ANCHOR_TEMPERATURE,
            seed_base=cfg.seed_base,
            render_fn=BK.render_context,
            ids_fn=BK.context_token_ids,
        )
        wall = time.time() - t0
        new_rows: list[dict] = []
        for b, ctx in enumerate(chunk):
            ctx_len = len(BK.context_token_ids(tok, ctx))
            for i in range(cfg.draws):
                text = results[b][i]
                n_comp = len(tok(text, add_special_tokens=False)["input_ids"])
                new_rows.append(_gen_row(cfg, ctx, ctx_len, n_eot, i, ci, text, n_comp, max_new))
        # chunk-grain checkpoint: single buffered append + flush/fsync (torn-tail
        # tolerated on resume read).
        with part.open("a", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_rows))
            fh.flush()
            os.fsync(fh.fileno())
        rows.extend(new_rows)
        print(
            f"[anchors:{cell}] unit {ci + 1}/{len(chunks)} rows={len(new_rows)} "
            f"elapsed={time.time() - t_cell:.1f}s chunk_wall={wall:.1f}s",
            flush=True,
        )
        _pilot_note(cfg, pilot, len(chunk) * cfg.draws, wall)
    rows.sort(key=lambda r: (r["chunk"], r["context_id"], r["draw"]))
    return rows


def _anchor_cell_complete(cfg: Cfg2564, cell: str) -> bool:
    m = _read_json(cfg.manifest_dir / f"anchors_{cell}.done.json")
    return (
        m is not None
        and m.get("regime_fp") == _cell_fp(cfg, "A", cell)
        and (cfg.anchors_dir / f"anchors_{cell}.jsonl").is_file()
    )


def _anchor_cell(
    cfg: Cfg2564, model, tok, eot_ids: list[int], cell: str, ctxs: list[dict], pilot: dict
) -> None:
    """One PA cell: generate → cap-hit check (>2% ⇒ whole-cell re-gen at 4096) →
    atomic final jsonl + done manifest. Text is persisted BEFORE any capture (#779)."""
    out_path = cfg.anchors_dir / f"anchors_{cell}.jsonl"
    rows = _generate_cell(cfg, model, tok, eot_ids, cell, ctxs, pilot, cfg.max_new_tokens)
    frac = sum(1 for r in rows if r["cap_hit"]) / max(1, len(rows))
    regen_frac = None
    max_new_final = cfg.max_new_tokens
    if frac > CAP_HIT_REGEN_FRAC and cfg.max_new_tokens < REGEN_MAX_NEW:
        logger.warning(
            "[anchors:%s] cap-hit frac %.4f > %.2f — regenerating whole cell at max_new=%d",
            cell,
            frac,
            CAP_HIT_REGEN_FRAC,
            REGEN_MAX_NEW,
        )
        write_jsonl_atomic(
            cfg.anchors_dir / f"anchors_{cell}.capped{cfg.max_new_tokens}.jsonl", rows
        )
        rows = _generate_cell(cfg, model, tok, eot_ids, cell, ctxs, pilot, REGEN_MAX_NEW)
        regen_frac = sum(1 for r in rows if r["cap_hit"]) / max(1, len(rows))
        max_new_final = REGEN_MAX_NEW
    write_jsonl_atomic(out_path, rows)
    for max_new in (cfg.max_new_tokens, REGEN_MAX_NEW):
        p = cfg.anchors_dir / f"anchors_{cell}.max{max_new}.partial"
        if p.is_file():
            p.unlink()
    manifest = {
        "cell": cell,
        "regime_fp": _cell_fp(cfg, "A", cell),
        "n_contexts": len(ctxs),
        "n_rows": len(rows),
        "cap_hit_frac": frac,
        "cap_hit_frac_regen": regen_frac,
        "max_new_tokens_final": max_new_final,
        "repro": _repro(cfg, "pa-anchors"),
    }
    write_json_atomic(cfg.manifest_dir / f"anchors_{cell}.done.json", manifest)


def _upload_summary(res) -> dict:
    return {
        "repo_id": res.repo_id,
        "uploaded": len(res.uploaded),
        "rerouted": len(res.rerouted),
        "skipped_existing": len(res.skipped_existing),
    }


def phase_anchors(cfg: Cfg2564, bank: dict) -> int:
    """PA: generation of all anchor rollouts (plan §3.8 pa_generate)."""
    _invalidate_phase_records(cfg, "A")
    contexts = bank["contexts"]
    cells = sorted({c["cell"] for c in contexts})
    per_cell = {cell: [c for c in contexts if c["cell"] == cell] for cell in cells}
    sentinel = cfg.out_root / "anchors_done.json"
    pending = [cell for cell in cells if not _anchor_cell_complete(cfg, cell)]
    if not pending and sentinel.is_file():
        s = _read_json(sentinel)
        if s is not None and s.get("regime_fp") == _regime_fp(cfg, {"phase": "A"}):
            logger.info("[anchors] all %d cells complete + sentinel present — skipping", len(cells))
            return RC_OK
    if pending:
        assert_out_root_headroom(cfg.out_root, need_gb=5.0, phase="pa-anchors")
        model, tok = R.load_model_and_tokenizer(cfg)
        eot_ids = R.eot_tail_ids(tok)
        total_rows = sum(len(v) for v in per_cell.values()) * cfg.draws
        pilot = _pilot_state(cfg, total_rows)
        t0 = time.time()
        for k, cell in enumerate(pending):
            _anchor_cell(cfg, model, tok, eot_ids, cell, per_cell[cell], pilot)
            print(
                f"[anchors] cell {k + 1}/{len(pending)} {cell} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        if not pilot["evaluated"]:
            _pilot_eval(cfg, pilot)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    upload: dict = {"mode": cfg.upload}
    if cfg.upload == "hf":
        res = upload_dir_sharded(
            cfg.anchors_dir,
            HF_DATA_REPO,
            f"{cfg.hf_prefix}/raw_completions/anchors",
            shard_glob="*.jsonl",
            resume_skip=False,  # anchors jsonls are mutable across cap-hit re-gen
            delete_local=False,
        )
        upload["anchors"] = _upload_summary(res)
    manifests = {cell: _read_json(cfg.manifest_dir / f"anchors_{cell}.done.json") for cell in cells}
    write_json_atomic(
        sentinel,
        {
            "phase": "A",
            "regime_fp": _regime_fp(cfg, {"phase": "A"}),
            "n_cells": len(cells),
            "n_rows": sum(int(m["n_rows"]) for m in manifests.values() if m),
            "cap_hit_frac": {cell: (m or {}).get("cap_hit_frac") for cell, m in manifests.items()},
            "upload": upload,
            "hf_prefix": cfg.hf_prefix,
            "repro": _repro(cfg, "pa-anchors"),
        },
    )
    print("[phase=pa_generate] sentinel written", flush=True)
    return RC_OK


# ── PB: teacher-forced capture ────────────────────────────────────────────


def _va_cell_complete(cfg: Cfg2564, cell: str) -> bool:
    m = _read_json(cfg.manifest_dir / f"va2564_{cell}.done.json")
    return (
        m is not None
        and m.get("regime_fp") == _cell_fp(cfg, "B", cell)
        and (cfg.va_dir / f"va2564_{cell}.pt").is_file()
    )


def _vc_complete(cfg: Cfg2564) -> bool:
    m = _read_json(cfg.manifest_dir / "vc2564.done.json")
    return (
        m is not None
        and m.get("regime_fp") == _regime_fp(cfg, {"phase": "B", "leg": "vc"})
        and (cfg.vc_dir / "vc2564_bank.pt").is_file()
    )


def _parity_path(cfg: Cfg2564) -> Path:
    return cfg.manifest_dir / "parity_gate_report.json"


def _parity_report_ok(cfg: Cfg2564) -> bool:
    rep = _read_json(_parity_path(cfg))
    return (
        rep is not None
        and rep.get("regime_fp") == _gate_fp(cfg, "pb_parity")
        and (rep.get("verdict") == "pass" or rep.get("demoted"))
    )


def _require_anchor_shards(cfg: Cfg2564, cells: list[str]) -> None:
    missing = [c for c in cells if not _anchor_cell_complete(cfg, c)]
    if missing:
        raise RuntimeError(
            f"PB requires completed PA anchor shards; missing/stale cells: {missing} "
            "(run --phase A first)"
        )


def _capture_vc(cfg: Cfg2564, model, tok, contexts: list[dict]) -> None:
    """v_C context-end capture: last real context token, all bank contexts, fp32."""
    layers = list(cfg.layers)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    ids_all = [BK.context_token_ids(tok, c) for c in contexts]
    vc = torch.zeros(len(contexts), len(layers), cfg.hidden, dtype=torch.float32)
    t0 = time.time()
    n_chunks = (len(contexts) + cfg.capture_batch - 1) // cfg.capture_batch
    with torch.no_grad():
        for ci in range(n_chunks):
            lo = ci * cfg.capture_batch
            chunk_ids = ids_all[lo : lo + cfg.capture_batch]
            ids, mask = R._right_pad(chunk_ids, pad_id, cfg.device)
            acts = extract_layer_activations(model, ids, layers, attention_mask=mask)
            for b, row_ids in enumerate(chunk_ids):
                pos = len(row_ids) - 1
                for li, layer in enumerate(layers):
                    vc[lo + b, li] = acts[layer][b, pos].float().cpu()
            print(
                f"[vc2564] unit {ci + 1}/{n_chunks} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    assert vc.shape == (len(contexts), len(layers), cfg.hidden), vc.shape
    store = {
        "issue": ISSUE,
        "layers": layers,
        "context_ids": [c["id"] for c in contexts],
        "vc": vc,
        "dtype": "fp32",
        "position": "context_end_last_token",
        "repro": _repro(cfg, "pb-capture"),
    }
    save_pt_atomic(cfg.vc_dir / "vc2564_bank.pt", store)
    write_json_atomic(
        cfg.manifest_dir / "vc2564.done.json",
        {
            "regime_fp": _regime_fp(cfg, {"phase": "B", "leg": "vc"}),
            "n_contexts": len(contexts),
            "repro": _repro(cfg, "pb-capture"),
        },
    )


def _capture_cell_va(
    cfg: Cfg2564, model, tok, eot_ids: list[int], cell: str, ctx_by_id: dict
) -> None:
    """One PB cell: teacher-forced v_A capture (span mean + tail twin, fp16) with the
    MF-A return_boundaries records, EXACT-compared against PA's gen-side records."""
    rows = _read_jsonl(cfg.anchors_dir / f"anchors_{cell}.jsonl")
    ctx_ids_by_row = [BK.context_token_ids(tok, ctx_by_id[r["context_id"]]) for r in rows]
    for r, ids in zip(rows, ctx_ids_by_row):
        assert len(ids) == r["ctx_len"], (
            f"ctx re-tokenization drift for {r['context_id']}: {len(ids)} != {r['ctx_len']}"
        )
    completions = [r["text"] for r in rows]
    t0 = time.time()
    out = R.capture_answer_states(
        cfg,
        model,
        tok,
        ctx_ids_by_row,
        completions,
        eot_ids,
        tail_inclusive=True,
        return_boundaries=True,
    )
    wall = time.time() - t0
    bounds = out["boundaries"]
    assert len(bounds) == len(rows), (len(bounds), len(rows))
    for r, b, n in zip(rows, bounds, out["n_completion_tokens"]):
        assert int(n) == r["n_completion_tokens_gen"], (
            f"completion-len drift {cell}/{r['context_id']}/d{r['draw']}: "
            f"capture {n} != gen {r['n_completion_tokens_gen']}"
        )
        for key in ("ctx_len", "span_start", "span_end", "tail_end"):
            assert b[key] == r[key], (
                f"gate-4 boundary mismatch {cell}/{r['context_id']}/d{r['draw']} "
                f"{key}: capture {b[key]} != gen {r[key]}"
            )
    index = [
        {"context_id": r["context_id"], "cell": r["cell"], "draw": r["draw"], **b}
        for r, b in zip(rows, bounds)
    ]
    store = {
        "issue": ISSUE,
        "cell": cell,
        "layers": list(cfg.layers),
        "index": index,
        "va_span": out["va_span"],
        "va_tail_incl": out["va_tail_incl"],
        "poolings": ["span_mean", "tail_inclusive_mean"],
        "empty_rows": out["empty_rows"],
        "eot_ids": eot_ids,
        "max_new_tokens": rows[0]["max_new_tokens"] if rows else None,
        "repro": _repro(cfg, "pb-capture"),
    }
    save_pt_atomic(cfg.va_dir / f"va2564_{cell}.pt", store)
    write_json_atomic(
        cfg.manifest_dir / f"va2564_{cell}.done.json",
        {
            "cell": cell,
            "regime_fp": _cell_fp(cfg, "B", cell),
            "n_rows": len(rows),
            "n_empty_rows": len(out["empty_rows"]),
            "per_row_s": wall / max(1, len(rows)),
            "repro": _repro(cfg, "pb-capture"),
        },
    )


def _gate4_parity(
    cfg: Cfg2564, model, tok, eot_ids: list[int], cells: list[str], ctx_by_id: dict
) -> None:
    """Gate-4: (a) 3 sampled non-empty answer rows — 3-way EXACT record parity (gate
    re-derivation vs PA gen record vs PB store index) + single-row re-capture cosine
    >= PARITY_COS_MIN (span-mean class); (b) 3 sampled contexts — v_C re-capture under
    the bf16 two-bar (flat >= 0.995, first-captured-layer >= 0.999). Refusal = report
    + rc=8; demoted (informational) under smoke/tiny per plan §7/§8."""
    rng = random.Random(cfg.seed_base + 4)
    demoted = cfg.smoke or cfg.tiny
    cell_rows = {cell: _read_jsonl(cfg.anchors_dir / f"anchors_{cell}.jsonl") for cell in cells}
    cand = [
        (cell, idx)
        for cell in cells
        for idx, r in enumerate(cell_rows[cell])
        if r["n_completion_tokens_gen"] > 0
    ]
    answer_checks: list[dict] = []
    n_answer = min(PARITY_N_ANSWER_ROWS, len(cand))
    for cell, idx in rng.sample(cand, n_answer) if n_answer else []:
        r = cell_rows[cell][idx]
        store = torch.load(cfg.va_dir / f"va2564_{cell}.pt", map_location="cpu", weights_only=False)
        ctx_ids = BK.context_token_ids(tok, ctx_by_id[r["context_id"]])
        n_comp = len(tok(r["text"], add_special_tokens=False)["input_ids"])
        rec_gate = {
            "ctx_len": len(ctx_ids),
            "n_completion_tokens": n_comp,
            "span_start": len(ctx_ids),
            "span_end": len(ctx_ids) + n_comp,
            "tail_end": len(ctx_ids) + n_comp + len(eot_ids),
        }
        rec_gen = {
            "ctx_len": r["ctx_len"],
            "n_completion_tokens": r["n_completion_tokens_gen"],
            "span_start": r["span_start"],
            "span_end": r["span_end"],
            "tail_end": r["tail_end"],
        }
        srec = store["index"][idx]
        rec_store = {k: srec[k] for k in rec_gate}
        records_ok = rec_gate == rec_gen == rec_store
        out1 = R.capture_answer_states(
            cfg,
            model,
            tok,
            [ctx_ids],
            [r["text"]],
            eot_ids,
            tail_inclusive=True,
            return_boundaries=True,
        )
        cos_span = _flat_cos(out1["va_span"][0].float(), store["va_span"][idx].float())
        cos_tail = _flat_cos(out1["va_tail_incl"][0].float(), store["va_tail_incl"][idx].float())
        answer_checks.append(
            {
                "cell": cell,
                "row": idx,
                "context_id": r["context_id"],
                "draw": r["draw"],
                "records_exact_3way": records_ok,
                "rec_gate": rec_gate,
                "rec_gen": rec_gen,
                "rec_store": rec_store,
                "cos_span": cos_span,
                "cos_tail": cos_tail,
                "pass": bool(
                    records_ok and cos_span >= PARITY_COS_MIN and cos_tail >= PARITY_COS_MIN
                ),
            }
        )
    vc_store = torch.load(cfg.vc_dir / "vc2564_bank.pt", map_location="cpu", weights_only=False)
    vc_pos = {cid: i for i, cid in enumerate(vc_store["context_ids"])}
    layers = list(cfg.layers)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    context_checks: list[dict] = []
    ctx_pool = sorted(vc_pos)
    n_ctx = min(PARITY_N_CONTEXTS, len(ctx_pool))
    with torch.no_grad():
        for cid in rng.sample(ctx_pool, n_ctx) if n_ctx else []:
            ctx_ids = BK.context_token_ids(tok, ctx_by_id[cid])
            ids, mask = R._right_pad([ctx_ids], pad_id, cfg.device)
            acts = extract_layer_activations(model, ids, layers, attention_mask=mask)
            v1 = torch.stack([acts[layer][0, len(ctx_ids) - 1].float().cpu() for layer in layers])
            stored = vc_store["vc"][vc_pos[cid]].float()
            per_layer = [_flat_cos(v1[li], stored[li]) for li in range(len(layers))]
            flat = _flat_cos(v1, stored)
            early = per_layer[0]  # first captured layer (14) — dbe `or [0]` fallback
            context_checks.append(
                {
                    "context_id": cid,
                    "per_layer_cos": per_layer,
                    "flat_cos": flat,
                    "early_cos": early,
                    "pass": bool(
                        flat >= PARITY_COS_MIN_SINGLEPOS_FLAT
                        and early >= PARITY_COS_MIN_SINGLEPOS_EARLY
                    ),
                }
            )
    all_pass = all(c["pass"] for c in answer_checks + context_checks)
    verdict = "pass" if all_pass else "fail"
    report = {
        "gate": "pb_parity",
        "regime_fp": _gate_fp(cfg, "pb_parity"),
        "thresholds": {
            "span_mean_cos_min": PARITY_COS_MIN,
            "singlepos_flat_cos_min": PARITY_COS_MIN_SINGLEPOS_FLAT,
            "singlepos_early_cos_min": PARITY_COS_MIN_SINGLEPOS_EARLY,
        },
        "answer_checks": answer_checks,
        "context_checks": context_checks,
        "verdict": verdict,
        "demoted": demoted,
        "repro": _repro(cfg, "pb-parity"),
    }
    write_json_atomic(_parity_path(cfg), report)
    logger.info(
        "[parity] verdict=%s demoted=%s answer=%d/%d context=%d/%d",
        verdict,
        demoted,
        sum(c["pass"] for c in answer_checks),
        len(answer_checks),
        sum(c["pass"] for c in context_checks),
        len(context_checks),
    )
    if verdict == "fail" and not demoted:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(RC_PARITY_GATE)


def phase_capture(cfg: Cfg2564, bank: dict) -> int:
    """PB: teacher-forced v_A + v_C capture, gate-4 parity, HF upload (plan §3.8
    pb_capture)."""
    _invalidate_phase_records(cfg, "B")
    contexts = bank["contexts"]
    ctx_by_id = {c["id"]: c for c in contexts}
    cells = sorted({c["cell"] for c in contexts})
    _require_anchor_shards(cfg, cells)
    sentinel = cfg.out_root / "va2564_uploaded.json"
    pending_va = [cell for cell in cells if not _va_cell_complete(cfg, cell)]
    vc_pending = not _vc_complete(cfg)
    parity_pending = not _parity_report_ok(cfg)
    if not pending_va and not vc_pending and not parity_pending and sentinel.is_file():
        s = _read_json(sentinel)
        if s is not None and s.get("regime_fp") == _regime_fp(cfg, {"phase": "B"}):
            logger.info("[capture] all cells + vc + parity complete + sentinel — skipping")
            return RC_OK
    if pending_va or vc_pending or parity_pending:
        assert_out_root_headroom(cfg.out_root, need_gb=5.0, phase="pb-capture")
        model, tok = R.load_model_and_tokenizer(cfg)
        eot_ids = R.eot_tail_ids(tok)
        if vc_pending:
            _capture_vc(cfg, model, tok, contexts)
        t0 = time.time()
        for k, cell in enumerate(pending_va):
            _capture_cell_va(cfg, model, tok, eot_ids, cell, ctx_by_id)
            print(
                f"[capture] cell {k + 1}/{len(pending_va)} {cell} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        _gate4_parity(cfg, model, tok, eot_ids, cells, ctx_by_id)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    upload: dict = {"mode": cfg.upload}
    if cfg.upload == "hf":
        for name, local, prefix, resume_skip in (
            # resume_skip=False on the .pt stores too (r2 blocker 3): a recomputed
            # same-shape store is size-IDENTICAL but content-different, and the
            # size-match presence probe would silently retain STALE HF tensors for
            # the off-pod PE phase (#2552 class). Mutable-across-recompute, like
            # the anchors jsonls.
            ("va2564", cfg.va_dir, f"{cfg.hf_prefix}/analysis_tensors/va2564", False),
            ("vc2564", cfg.vc_dir, f"{cfg.hf_prefix}/analysis_tensors/vc2564", False),
            ("manifests", cfg.manifest_dir, f"{cfg.hf_prefix}/manifests", False),
        ):
            glob = "*.pt" if name != "manifests" else "*.json"
            res = upload_dir_sharded(
                local,
                HF_DATA_REPO,
                prefix,
                shard_glob=glob,
                resume_skip=resume_skip,
                delete_local=False,
            )
            upload[name] = _upload_summary(res)
    write_json_atomic(
        sentinel,
        {
            "phase": "B",
            "regime_fp": _regime_fp(cfg, {"phase": "B"}),
            "n_cells": len(cells),
            "n_contexts_vc": len(contexts),
            "upload": upload,
            "hf_prefix": cfg.hf_prefix,
            "repro": _repro(cfg, "pb-capture"),
        },
    )
    print("[phase=pb_capture] sentinel written", flush=True)
    return RC_OK


# ── PC: text-space embedding (subprocess: scripts/issue2564_embed.py) ─────


def _embed_out_root(cfg: Cfg2564) -> Path:
    """The embed subprocess's REALIZED out-root (replicates its own smoke rebind:
    ``<raw>/embed`` -> ``<raw>/smoke_embed`` under --smoke), so this driver can
    locate + verify the sentinel the subprocess writes."""
    base = (cfg.raw_out_root or cfg.out_root) / "embed"
    if cfg.smoke and not cfg.tiny:
        return base.parent / f"smoke_{base.name}"
    return base


def phase_embed(cfg: Cfg2564, bank: dict) -> int:
    """PC: Qwen3-Embedding-8B embedding of all anchor texts (plan §9 pc_embed,
    on-pod 1×H100) via a ``scripts/issue2564_embed.py`` subprocess — vLLM pooling
    engine in its OWN process (spawn-env + teardown isolation from this driver's
    HF model), ``env={**os.environ}`` passthrough. The subprocess owns chunk-grain
    fingerprint resume + the pilot gate (rc=7 propagates verbatim); this driver
    VERIFIES the sentinel landed before terminal success (r2 blocker 5)."""
    _invalidate_phase_records(cfg, "C")
    if cfg.tiny:
        if cfg.phase == "C":
            raise RuntimeError(
                "phase C (vLLM pooling embed) cannot run under --tiny — no CPU engine; "
                "use --smoke on a GPU host"
            )
        logger.warning("[embed] SKIPPED under --tiny (vLLM pooling engine needs a GPU)")
        return RC_OK
    contexts = bank["contexts"]
    cells = sorted({c["cell"] for c in contexts})
    _require_anchor_shards(cfg, cells)  # PC consumes PA's anchors (local root)
    embed_out = _embed_out_root(cfg)
    sentinel = embed_out / (
        "embed_done.local.json" if cfg.upload == "none" else "embed_uploaded.json"
    )
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "issue2564_embed.py"),
        "--out-root",
        str((cfg.raw_out_root or cfg.out_root) / "embed"),
        "--anchors-root",
        str(cfg.out_root),
        "--cells",
        ",".join(cells),
    ]
    if cfg.smoke:
        cmd.append("--smoke")
    if cfg.upload == "none":
        cmd.append("--skip-upload")
    print(f"[embed] launching subprocess: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env={**os.environ})
    if proc.returncode == RC_PILOT_GATE:
        logger.error("[embed] pilot gate refusal (rc=7) — propagating")
        return RC_PILOT_GATE
    if proc.returncode != 0:
        raise RuntimeError(f"embed subprocess failed rc={proc.returncode} (cmd: {' '.join(cmd)})")
    if not sentinel.is_file():
        raise RuntimeError(
            f"embed subprocess exited 0 but its sentinel is missing: {sentinel} "
            "(out-root derivation drift between driver and embed script?)"
        )
    print("[phase=pc_embed] sentinel verified", flush=True)
    return RC_OK


# ── main ──────────────────────────────────────────────────────────────────


def _load_tokenizer(cfg: Cfg2564):
    """Real tokenizer (bank gates need the pinned Qwen BPE even under --tiny).

    Revision-pinned when resolved (r2 blocker 8); the ``unresolved*`` tiny
    sentinel degrades to None (default tip)."""
    from transformers import AutoTokenizer

    rev = None if cfg.model_revision.startswith("unresolved") else cfg.model_revision
    try:
        return AutoTokenizer.from_pretrained(cfg.model_id, revision=rev, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(cfg.model_id, revision=rev)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _assert_call_kwargs()
        print("[import-check] ok", flush=True)
        return RC_OK
    cfg = build_config(args)
    _assert_call_kwargs()  # kwarg-signature assertion at driver start (MF-A leg 4)
    cfg.model_revision = _resolve_model_revision(cfg)
    for d in (cfg.out_root, cfg.log_dir, cfg.anchors_dir, cfg.va_dir, cfg.vc_dir, cfg.manifest_dir):
        d.mkdir(parents=True, exist_ok=True)
    logger.info(
        "config: phase=%s out_root=%s smoke=%s tiny=%s cells=%s carriers=%s draws=%d "
        "gen_batch=%d capture_batch=%d max_new=%d seed_base=%d upload=%s model=%s@%s",
        cfg.phase,
        cfg.out_root,
        cfg.smoke,
        cfg.tiny,
        cfg.cells,
        cfg.carriers,
        cfg.draws,
        cfg.gen_batch,
        cfg.capture_batch,
        cfg.max_new_tokens,
        cfg.seed_base,
        cfg.upload,
        cfg.model_id,
        cfg.model_revision,
    )
    tok = _load_tokenizer(cfg)
    values = BK.load_values(cfg.values_path)
    # Bank gates run at FULL 984-context grain regardless of --smoke (plan §8 A3);
    # smoke slicing applies AFTER the gates, to execution only.
    bank = BK.build_bank(tok, values=values)
    BK.write_bank_manifest(bank, cfg.manifest_dir / "bank2564_manifest.json")
    bank = _filter_bank(bank, cfg.cells, cfg.carriers)
    logger.info("bank: %d contexts / %d pairs after filter", bank["n_contexts"], bank["n_pairs"])
    phase_fns = {
        "A": ("pa_generate", phase_anchors),
        "B": ("pb_capture", phase_capture),
        "C": ("pc_embed", phase_embed),
    }
    run_phases = list(PHASES) if cfg.phase == "all" else [cfg.phase]
    for ph in run_phases:
        name, fn = phase_fns[ph]
        print(f"[phase={name}]", flush=True)
        rc = fn(cfg, bank)
        if rc != RC_OK:
            return rc
    print("[phase=done]", flush=True)
    return RC_OK


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
