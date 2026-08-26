"""#2587 battery driver — P5 battery + pilot GENERATION phase (plan v3 §4.4, unit 3a).

Generates K=10 rollouts per merged-bank context (1,080 contexts = 984 parent +
96 pilot -> 10,800 rollouts at full production) on Qwen3.5-9B with thinking
DISABLED, via the parent/langow/#2329 helper
``explore_persona_space.experiments.issue1415.steering.generate_batch`` and the
#2333 q35 ``ids_fn`` (unit 1's ``make_ids_fn`` -> ``context_token_ids_q35``,
closed-empty-``<think>`` assert per row).

Pattern provenance: the gen-phase machinery (``.partial`` chunk-grain
checkpointing keyed on GENERATING-PARAMETER fingerprints, quarantine of stale
partials, per-axis done manifests, >2%-cap-hit whole-axis re-gen at 4096, the
phase sentinel) is transcribed from ``scripts/issue2564_langow_pilot_run.py``
@ ``bank2587.LANGOW_COMMIT`` (read-only parent-owned script — transcription,
never import; the same convention ``bank2587`` records). The pilot wall-time
gate is a port of ``scripts/issue2215_run.py::run_pilot_gate`` (§9 P5 row).

Phase registry (``PHASES``): ``gen`` (THIS unit) | ``capture`` + ``embed``
(unit 3b — P6 all-layer fp32 capture via the pinned
``issue2162_run.capture_answer_states(..., tail_inclusive=True,
return_boundaries=True)`` and the Qwen3-Embedding-8B third space; both raise
``NotImplementedError`` here, never a silent no-op). Unit 3b drops in by
implementing ``phase_capture`` / ``phase_embed`` and extending ``main``'s
dispatch; the pinned-module accessor ``_r()`` already imports the exact blob
capture needs.

Sharding (plan §9: "P5 by axis halves"): ``--num-shards 2 --shard-index k``
deterministically splits the bank's axes (context ``cell`` key) across
processes, one CVD-pinned process per GPU (the launcher pins
``CUDA_VISIBLE_DEVICES``; this driver never sets it). Under ``num_shards > 1``
the out-root is auto-suffixed ``shard{k}`` so concurrent legs can never share a
root (the crash-fix-rounds per-leg out-root rule; #2330 fu1). ``--axes a,b``
(mutually exclusive with multi-sharding) is the P1-smoke slice, with
``--max-carriers`` + ``--draws`` narrowing the cell (§4.7 P1: 1 axis, 3
carriers, K=2 — production model, production loaders; no substituted
implementation exists in this driver).

Exit convention (plan §4.7): one python process, explicit terminals — rc 0 on
sentinel write, ``EXIT_PILOT_REFUSE`` (7) on a pilot-gate refusal (report JSON
at ``manifests/pilot_gate_report.json``), any crash = the process's own
non-zero exit observed by the poller. Rollout TEXT uploads unconditionally to
``{hf_prefix}/raw_completions/anchors/`` at end of shard, BEFORE any capture
consumes it (#779; upload mode is part of the resume fingerprint).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (thread caps + HF/API env)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import issue2587_common as cm2587  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic, write_jsonl_atomic  # noqa: E402
from explore_persona_space.experiments.issue1415.steering import generate_batch  # noqa: E402
from explore_persona_space.experiments.issue2587 import bank2587 as B  # noqa: E402
from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded  # noqa: E402

logger = logging.getLogger("issue2587_battery_run")

ISSUE = 2587
HF_DATA_REPO = os.environ.get("EPM_I2587_DATA_WRITE_REPO", "superkaiba1/explore-persona-space-data")
HF_PREFIX = "issue2587_minpair"

# §4.4 Generation pins (plan: K=10, temp 1.0, seed 42, batch 16, max_new 2048;
# inherited >2% cap-hit => re-gen at 4096).
ANCHOR_TEMPERATURE = 1.0
ANCHOR_MAX_NEW = 2048
REGEN_MAX_NEW = 4096
CAP_HIT_REGEN_FRAC = 0.02
CAP_HIT_BASIS = "retokenized_completion_len >= max_new_tokens"
ANCHOR_DRAWS = 10
GEN_BATCH = 16

# §9 P5 pilot gate: warmup + 2 production-shape generate calls, project the
# shard's pending rows, refuse loud above the ceiling (row ceiling = 6 h).
PILOT_CEILING_H = 6.0
EXIT_PILOT_REFUSE = 7

# Pinned #2564-branch issue2162_run blob (same PIN the langow pilot + bank2587
# use). Gen needs eot_tail_ids + cap_hit; unit 3b's capture phase reuses this
# accessor for capture_answer_states(..., tail_inclusive=True,
# return_boundaries=True) — the branch-only kwarg verified at the pin.
PIN_2162_REL = "scripts/issue2162_run.py"
PIN_2162_SHA256 = "6f77924461c04b3532a38c199fb96aec88d479d5359daec0ab73aad10c62538c"

PHASES = ("gen", "capture", "embed")

# The #2333 q35 thinking-off ids_fn (unit 1 single implementation; the render
# assert lives in bank2587.context_token_ids_q35, applied per row).
IDS_FN = cm2587.make_ids_fn()

_R = None


def _r():
    """Lazy pinned import of issue2162_run @ the #2564 pin (sha256-asserted).

    Reuses bank2587's ``_git_show`` (fetch-and-retry) + ``_import_pinned``
    (unique module name — no main-side shadowing) machinery."""
    global _R
    if _R is None:
        data = B._git_show(PIN_2162_REL)
        digest = hashlib.sha256(data).hexdigest()
        if digest != PIN_2162_SHA256:
            raise RuntimeError(
                f"pinned {PIN_2162_REL}: sha256 {digest} != recorded {PIN_2162_SHA256}"
            )
        pin_dir = Path(tempfile.mkdtemp(prefix="eps2587_battery_pin_"))
        (pin_dir / "issue2162_run.py").write_bytes(data)
        _R = B._import_pinned("issue2162_run_pinned_2587_battery", pin_dir / "issue2162_run.py")
    return _R


# ── config ────────────────────────────────────────────────────────────────


@dataclass
class Cfg:
    """Driver knobs. ``axes`` is the RESOLVED per-process axis subset (from
    ``--axes`` or the deterministic shard split); ``bank_values_sha`` pins the
    bank identity into every resume fingerprint."""

    phase: str
    out_root: Path
    model_id: str
    model_revision: str
    device: str
    gen_batch: int
    draws: int
    max_new_tokens: int
    seed_base: int
    upload: str  # "hf" | "none"
    axes: tuple[str, ...]
    shard_index: int
    num_shards: int
    max_carriers: int | None
    hf_repo: str
    hf_prefix: str
    bank_values_sha: str
    pilot_ceiling_h: float

    @property
    def anchors_dir(self) -> Path:
        return self.out_root / "anchors"

    @property
    def manifest_dir(self) -> Path:
        return self.out_root / "manifests"

    @property
    def quarantine_dir(self) -> Path:
        return self.out_root / "quarantine"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--out-root", default="/workspace/eps2587_battery")
    ap.add_argument(
        "--axes",
        default=None,
        help="comma-separated axis subset (P1 smoke); mutually exclusive with --num-shards > 1",
    )
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument(
        "--max-carriers",
        type=int,
        default=None,
        help="keep only the first k sorted carriers per axis (P1 smoke slice)",
    )
    ap.add_argument("--draws", type=int, default=ANCHOR_DRAWS)
    ap.add_argument("--gen-batch", type=int, default=GEN_BATCH)
    ap.add_argument("--max-new-tokens", type=int, default=ANCHOR_MAX_NEW)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--upload", choices=("hf", "none"), default="hf")
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument("--pilot-ceiling-h", type=float, default=PILOT_CEILING_H)
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="verify argparse attrs + pinned/ported call signatures, then exit 0",
    )
    return ap


def _resolve_model_revision() -> str:
    """Pin main -> resolved sha ONCE per run (#2061); fail loud."""
    from huggingface_hub import HfApi

    sha = HfApi().model_info(B.MODEL_ID).sha
    assert sha, f"could not resolve model revision for {B.MODEL_ID}"
    return sha


def build_cfg(
    args: argparse.Namespace, *, bank_values_sha: str, axes: tuple[str, ...], model_revision: str
) -> Cfg:
    """Resolve the Cfg. Under ``num_shards > 1`` the out-root auto-suffixes
    ``shard{k}`` so concurrent same-driver legs can never share a root
    (per-leg out-roots; #2330 fu1)."""
    out_root = Path(args.out_root)
    if args.num_shards > 1:
        out_root = out_root / f"shard{args.shard_index}"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return Cfg(
        phase=args.phase,
        out_root=out_root,
        model_id=B.MODEL_ID,
        model_revision=model_revision,
        device=device,
        gen_batch=args.gen_batch,
        draws=args.draws,
        max_new_tokens=args.max_new_tokens,
        seed_base=args.seed_base,
        upload=args.upload,
        axes=axes,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        max_carriers=args.max_carriers,
        hf_repo=HF_DATA_REPO,
        hf_prefix=args.hf_prefix,
        bank_values_sha=bank_values_sha,
        pilot_ceiling_h=args.pilot_ceiling_h,
    )


def _import_check() -> None:
    """Module-level import/signature check (argparse-attribute completeness +
    the #2261 bind pass via the shared argcheck helper, plus explicit
    signature smokes on every ported/pinned callee this unit and unit 3b's
    seam depend on)."""
    import inspect

    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    r = _r()
    for fn, needed in (
        (
            generate_batch,
            {"n", "hook", "max_new_tokens", "temperature", "seed_base", "render_fn", "ids_fn"},
        ),
        (
            r.capture_answer_states,  # unit 3b's seam — the branch-only kwarg exists at the pin
            {"payloads", "positions", "tail_inclusive", "return_boundaries"},
        ),
        (upload_dir_sharded, {"shard_glob", "resume_skip", "delete_local"}),
    ):
        params = set(inspect.signature(fn).parameters)
        missing = needed - params
        assert not missing, (getattr(fn, "__name__", fn), sorted(missing))
    for name in ("eot_tail_ids", "cap_hit"):
        assert callable(getattr(r, name)), name
    for name in (
        "render_context_q35",
        "context_token_ids_q35",
        "build_bank",
        "build_bank_strings",
        "run_token_gates",
    ):
        assert callable(getattr(B, name)), name
    for name in ("think_leak_scan", "assert_think_leak", "load_q35_model_and_tokenizer"):
        assert callable(getattr(cm2587, name)), name
    print("[import-check] ok: pinned modules + ported call signatures resolve", flush=True)


# ── small io / provenance helpers (langow transcription) ─────────────────


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_jsonl(path: Path, tolerate_torn_tail: bool = False) -> list[dict]:
    """Text-mode line iteration (never ``splitlines()`` — U+2028 shred gotcha);
    a torn final line is dropped only under ``tolerate_torn_tail``."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().split("\n") if ln.strip()]
    for k, ln in enumerate(lines):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            if tolerate_torn_tail and k == len(lines) - 1:
                logger.warning("[resume] dropping torn tail line of %s", path.name)
                return rows
            raise
    return rows


_REPRO_CACHE: dict | None = None


def _repro(cfg: Cfg, phase: str) -> dict:
    """Reproducibility metadata carried by every persisted artifact (the
    ``phase`` key is a card phase IDENTITY, validated — code-style §Card
    phase identity)."""
    global _REPRO_CACHE
    from explore_persona_space.orchestrate.provenance import validate_phase_identity

    if _REPRO_CACHE is None:
        import transformers

        from explore_persona_space.orchestrate.provenance import (
            as_metadata_dict,
            git_provenance,
        )

        _REPRO_CACHE = {
            **as_metadata_dict(git_provenance()),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "parent_pin": B.PIN,
            "pin_2162_sha256": PIN_2162_SHA256,
            "langow_commit": B.LANGOW_COMMIT,
        }
    return {
        **_REPRO_CACHE,
        "phase": validate_phase_identity(phase),
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _sha16(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _regime_fp(cfg: Cfg, extra: dict | None = None) -> str:
    """16-hex fingerprint of the GENERATING PARAMETERS (never recomputed
    floats, #1336) — the resume / done-manifest key. Deliberately EXCLUDES the
    axes tuple / shard assignment, so a completed axis resumes across shard
    re-splits; ``upload`` IS included (a --upload none run must never satisfy
    a later --upload hf run's sentinel — the langow review-finding-1 shape)."""
    base = {
        "issue": ISSUE,
        "parent_pin": B.PIN,
        "langow_commit": B.LANGOW_COMMIT,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "bank_values_sha": cfg.bank_values_sha,
        "draws": cfg.draws,
        "gen_batch": cfg.gen_batch,
        "seed_base": cfg.seed_base,
        "temperature": str(ANCHOR_TEMPERATURE),
        "max_new_tokens": cfg.max_new_tokens,
        "max_carriers": cfg.max_carriers,
        "upload": cfg.upload,
    }
    if extra:
        base.update(extra)
    return _sha16(base)


def _cell_fp(cfg: Cfg, phase: str, cell: str) -> str:
    return _regime_fp(cfg, {"phase": phase, "cell": cell})


# ── axis grouping + sharding ──────────────────────────────────────────────


def group_contexts_by_cell(bank: dict) -> dict[str, list[dict]]:
    """Mechanical per-axis grouping over the merged bank (context ``cell``
    key), id-sorted within axis; asserts the grouping covers the whole bank."""
    by: dict[str, list[dict]] = {}
    for ctx in bank["contexts"].values():
        by.setdefault(ctx["cell"], []).append(ctx)
    for lst in by.values():
        lst.sort(key=lambda c: c["id"])
    total = sum(len(v) for v in by.values())
    assert total == bank["n_contexts"], (total, bank["n_contexts"])
    return dict(sorted(by.items()))


def shard_axes(per_axis_counts: dict[str, int], num_shards: int) -> dict[str, int]:
    """Deterministic greedy balance: axes by (-context count, name) each go to
    the lightest shard (tie -> lowest index). Disjoint + complete by
    construction; stable across dict insertion orders."""
    assert num_shards >= 1, num_shards
    loads = [0] * num_shards
    out: dict[str, int] = {}
    for ax in sorted(per_axis_counts, key=lambda a: (-per_axis_counts[a], a)):
        s = min(range(num_shards), key=lambda i: (loads[i], i))
        out[ax] = s
        loads[s] += per_axis_counts[ax]
    return out


def apply_max_carriers(ctxs: list[dict], k: int | None) -> list[dict]:
    """P1-smoke slice: keep contexts whose carrier is among the first ``k``
    sorted carrier ids of the axis (deterministic subset)."""
    if k is None:
        return list(ctxs)
    keep = set(sorted({c["carrier"] for c in ctxs})[:k])
    return [c for c in ctxs if c["carrier"] in keep]


# ── phase: gen ────────────────────────────────────────────────────────────


def _gen_row(
    cfg: Cfg,
    ctx: dict,
    ctx_len: int,
    n_eot: int,
    draw: int,
    chunk: int,
    text: str,
    n_comp: int,
    max_new: int,
) -> dict:
    """Generation-side per-row record incl. the gate-4 span fields (compared
    EXACTLY against the unit-3b capture path's own ``boundaries`` records)."""
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
        "cap_hit": _r().cap_hit(n_comp, max_new),
        "cap_hit_basis": CAP_HIT_BASIS,
        "text": text,
    }


def _generate_cell(
    cfg: Cfg, model, tok, eot_ids: list[int], cell: str, ctxs: list[dict], max_new: int
) -> list[dict]:
    """All draws for one axis, chunk-grain checkpointed to a ``.partial``
    sidecar (fp-header keyed; quarantined on mismatch; torn-tail-tolerant
    resume) — the langow ``_generate_cell`` transcribed to the q35 render."""
    part = cfg.anchors_dir / f"anchors_{cell}.max{max_new}.partial"
    part_fp = _regime_fp(cfg, {"phase": "gen", "cell": cell, "max_new_call": max_new})
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
            logger.warning("[gen:%s] quarantined stale .partial -> %s", cell, dest)
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
        logger.info("[gen:%s] resumed %d/%d chunks", cell, len(complete), len(chunks))
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
            render_fn=B.render_context_q35,
            ids_fn=IDS_FN,
        )
        wall = time.time() - t0
        new_rows: list[dict] = []
        for b, ctx in enumerate(chunk):
            ctx_len = len(IDS_FN(tok, ctx))
            for i in range(cfg.draws):
                text = results[b][i]
                n_comp = len(tok(text, add_special_tokens=False)["input_ids"])
                new_rows.append(_gen_row(cfg, ctx, ctx_len, n_eot, i, ci, text, n_comp, max_new))
        with part.open("a", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_rows))
            fh.flush()
            os.fsync(fh.fileno())
        rows.extend(new_rows)
        print(
            f"[gen:{cell}] unit {ci + 1}/{len(chunks)} rows={len(new_rows)} "
            f"elapsed={time.time() - t_cell:.1f}s chunk_wall={wall:.1f}s",
            flush=True,
        )
    rows.sort(key=lambda r: (r["chunk"], r["context_id"], r["draw"]))
    return rows


def _gen_cell_complete(cfg: Cfg, cell: str) -> bool:
    m = _read_json(cfg.manifest_dir / f"anchors_{cell}.done.json")
    return (
        m is not None
        and m.get("regime_fp") == _cell_fp(cfg, "gen", cell)
        and (cfg.anchors_dir / f"anchors_{cell}.jsonl").is_file()
    )


def _gen_cell(cfg: Cfg, model, tok, eot_ids: list[int], cell: str, ctxs: list[dict]) -> None:
    """One axis: generate -> cap-hit check (>2% => whole-axis re-gen at 4096)
    -> think-leak scan (rows flagged; hard assert < 1% AFTER the rows persist,
    BEFORE the done manifest) -> atomic final jsonl + done manifest.

    The manifest records ``capture_max_model_len_floor`` = max ctx_len +
    2 x max_new_tokens_final — the gotchas re-gen arithmetic bound for unit
    3b's capture re-entry (HF ``generate`` has no ``max_model_len`` pin, so
    the bound binds THERE, recorded HERE)."""
    out_path = cfg.anchors_dir / f"anchors_{cell}.jsonl"
    rows = _generate_cell(cfg, model, tok, eot_ids, cell, ctxs, cfg.max_new_tokens)
    frac = sum(1 for r in rows if r["cap_hit"]) / max(1, len(rows))
    regen_frac = None
    max_new_final = cfg.max_new_tokens
    if frac > CAP_HIT_REGEN_FRAC and cfg.max_new_tokens < REGEN_MAX_NEW:
        logger.warning(
            "[gen:%s] cap-hit frac %.4f > %.2f — re-gen at max_new=%d",
            cell,
            frac,
            CAP_HIT_REGEN_FRAC,
            REGEN_MAX_NEW,
        )
        write_jsonl_atomic(
            cfg.anchors_dir / f"anchors_{cell}.capped{cfg.max_new_tokens}.jsonl", rows
        )
        rows = _generate_cell(cfg, model, tok, eot_ids, cell, ctxs, REGEN_MAX_NEW)
        regen_frac = sum(1 for r in rows if r["cap_hit"]) / max(1, len(rows))
        max_new_final = REGEN_MAX_NEW
    assert len(rows) == len(ctxs) * cfg.draws, (cell, len(rows), len(ctxs), cfg.draws)
    scan = cm2587.think_leak_scan([r["text"] for r in rows])
    leaked = set(scan["leaked_indices"])
    for i, r in enumerate(rows):
        r["think_leak"] = i in leaked
    write_jsonl_atomic(out_path, rows)
    # Hard assert AFTER the flagged rows persist (diagnosable on disk), BEFORE
    # the partial unlink + done manifest (a failing axis stays resumable and
    # is never marked done). Plan §4.2 gate 2.
    cm2587.assert_think_leak(scan, label=f"gen:{cell}")
    for max_new in (cfg.max_new_tokens, REGEN_MAX_NEW):
        p = cfg.anchors_dir / f"anchors_{cell}.max{max_new}.partial"
        if p.is_file():
            p.unlink()
    max_ctx_len = max(r["ctx_len"] for r in rows)
    write_json_atomic(
        cfg.manifest_dir / f"anchors_{cell}.done.json",
        {
            "cell": cell,
            "regime_fp": _cell_fp(cfg, "gen", cell),
            "n_contexts": len(ctxs),
            "n_rows": len(rows),
            "cap_hit_frac": frac,
            "cap_hit_frac_regen": regen_frac,
            "max_new_tokens_final": max_new_final,
            "think_leak": {"n": scan["n"], "n_leaked": scan["n_leaked"], "frac": scan["frac"]},
            "max_ctx_len": max_ctx_len,
            "capture_max_model_len_floor": max_ctx_len + 2 * max_new_final,
            "repro": _repro(cfg, "gen"),
        },
    )


def _assert_no_foreign_axis_files(cfg: Cfg) -> None:
    """Fail loud if the anchors dir holds files for an UNassigned axis — the
    shared-out-root misuse guard (per-leg out-roots; #2330 fu1). Runs before
    generation AND protects the end-of-shard ``*.jsonl`` upload glob."""
    assigned = set(cfg.axes)
    for p in sorted(cfg.anchors_dir.glob("anchors_*.jsonl")):
        ax = p.name[len("anchors_") :].split(".")[0]
        if ax not in assigned:
            raise RuntimeError(
                f"foreign axis file {p.name} (axis {ax!r} not in this shard's axes "
                f"{sorted(assigned)}) — shards must not share an out-root"
            )


def _pilot_gate(
    cfg: Cfg, model, tok, ctxs_by_axis: dict[str, list[dict]], pending: list[str]
) -> dict:
    """§9 P5 pilot gate (``run_pilot_gate`` port, scripts/issue2215_run.py:874):
    warm ONE production-shape generate call, time TWO, project the shard's
    PENDING rows, refuse loud (EXIT_PILOT_REFUSE) above the ceiling. Pilot
    outputs are discarded (real rows regenerate under per-draw seeds)."""
    first = ctxs_by_axis[pending[0]][: cfg.gen_batch]
    total_rows = sum(len(ctxs_by_axis[a]) * cfg.draws for a in pending)
    kw = dict(
        hook=None,
        max_new_tokens=cfg.max_new_tokens,
        temperature=ANCHOR_TEMPERATURE,
        seed_base=cfg.seed_base,
        render_fn=B.render_context_q35,
        ids_fn=IDS_FN,
    )
    generate_batch(model, tok, first, n=1, **kw)  # warmup (discarded)
    t0 = time.monotonic()
    generate_batch(model, tok, first, n=2, **kw)  # 2 timed production-shape calls (discarded)
    wall_s = time.monotonic() - t0
    per_row_s = wall_s / (len(first) * 2)
    projected_h = per_row_s * total_rows / 3600.0
    report = {
        "n_timed_rows": len(first) * 2,
        "gen_batch": cfg.gen_batch,
        "wall_s": wall_s,
        "per_row_s": per_row_s,
        "pending_rows": total_rows,
        "projected_wall_h": projected_h,
        "ceiling_h": cfg.pilot_ceiling_h,
        "verdict": "proceed" if projected_h <= cfg.pilot_ceiling_h else "refuse",
        "repro": _repro(cfg, "gen-pilot"),
    }
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cfg.manifest_dir / "pilot_gate_report.json", report)
    logger.info(
        "[pilot] %.3f s/row x %d pending rows -> projected %.2f h (ceiling %.2f h) — %s",
        per_row_s,
        total_rows,
        projected_h,
        cfg.pilot_ceiling_h,
        report["verdict"],
    )
    return report


def phase_gen(cfg: Cfg, bank: dict, model, tok) -> int:
    """P5 battery + pilot generation for THIS shard's axes: pilot gate ->
    per-axis generate (resume-safe) -> unconditional rollout-text upload
    (BEFORE any capture consumes it, #779) -> shard sentinel."""
    print(
        f"[phase=gen] start shard={cfg.shard_index}/{cfg.num_shards} axes={','.join(cfg.axes)}",
        flush=True,
    )
    by_cell = group_contexts_by_cell(bank)
    unknown = [a for a in cfg.axes if a not in by_cell]
    if unknown:
        raise RuntimeError(f"unknown axes {unknown}; bank has {sorted(by_cell)}")
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    _assert_no_foreign_axis_files(cfg)
    ctxs_by_axis = {a: apply_max_carriers(by_cell[a], cfg.max_carriers) for a in cfg.axes}
    for a, lst in ctxs_by_axis.items():
        assert lst, f"axis {a}: no contexts survive --max-carriers={cfg.max_carriers}"
    sentinel = cfg.out_root / "battery_gen_done.json"
    sent_fp = _regime_fp(cfg, {"phase": "gen", "axes": sorted(cfg.axes)})
    pending = [a for a in cfg.axes if not _gen_cell_complete(cfg, a)]
    s = _read_json(sentinel)
    if not pending and s is not None and s.get("regime_fp") == sent_fp:
        logger.info("[gen] all axes complete + sentinel present — skipping")
        return 0
    if pending:
        pilot = _pilot_gate(cfg, model, tok, ctxs_by_axis, pending)
        if pilot["verdict"] == "refuse":
            print(
                f"[phase=gen] pilot refuse: projected {pilot['projected_wall_h']:.2f}h "
                f"> ceiling {pilot['ceiling_h']:.2f}h — see manifests/pilot_gate_report.json",
                flush=True,
            )
            return EXIT_PILOT_REFUSE
    eot_ids = _r().eot_tail_ids(tok)
    for a in cfg.axes:
        if _gen_cell_complete(cfg, a):
            logger.info("[gen:%s] done manifest present — skipping", a)
            continue
        _gen_cell(cfg, model, tok, eot_ids, a, ctxs_by_axis[a])
    upload: dict = {"mode": cfg.upload}
    if cfg.upload == "hf":
        # Rollout TEXT persists to HF BEFORE any capture reduce (#779). The
        # foreign-axis guard above makes every *.jsonl in the dir this shard's.
        res = upload_dir_sharded(
            cfg.anchors_dir,
            cfg.hf_repo,
            f"{cfg.hf_prefix}/raw_completions/anchors",
            shard_glob="*.jsonl",
            resume_skip=False,
            delete_local=False,
        )
        upload["anchors"] = {
            "repo_id": res.repo_id,
            "uploaded": len(res.uploaded),
            "rerouted": len(res.rerouted),
            "skipped_existing": len(res.skipped_existing),
        }
    write_json_atomic(
        sentinel,
        {
            "issue": ISSUE,
            "regime_fp": sent_fp,
            "shard_index": cfg.shard_index,
            "num_shards": cfg.num_shards,
            "axes": sorted(cfg.axes),
            "cells": {a: _read_json(cfg.manifest_dir / f"anchors_{a}.done.json") for a in cfg.axes},
            "upload": upload,
            "repro": _repro(cfg, "gen"),
        },
    )
    print("[phase=gen] sentinel written", flush=True)
    return 0


# ── unit 3b seams (P6 capture + embed) — hard NotImplementedError, never a
#    silent no-op. Unit 3b implements these + extends main()'s dispatch. ────


def phase_capture(cfg: Cfg, bank: dict, model, tok) -> int:
    """UNIT 3b: P6 all-layer fp32 battery capture (pinned
    ``issue2162_run.capture_answer_states(..., tail_inclusive=True,
    return_boundaries=True)`` via ``_r()``; gate-4 boundary compare against
    the gen rows' span fields)."""
    raise NotImplementedError(
        "--phase capture is unit 3b (P6 all-layer fp32 capture) — not implemented in unit 3a"
    )


def phase_embed(cfg: Cfg, bank: dict, model, tok) -> int:
    """UNIT 3b: Qwen3-Embedding-8B third-space embedding of the 10,800 answer
    texts (plan §4.4 Embedding paragraph — instrument-version parity)."""
    raise NotImplementedError(
        "--phase embed is unit 3b (Qwen3-Embedding-8B third space) — not implemented in unit 3a"
    )


PHASE_FNS = {"gen": phase_gen, "capture": phase_capture, "embed": phase_embed}


# ── main ──────────────────────────────────────────────────────────────────


def resolve_axes(args: argparse.Namespace, by_cell: dict[str, list[dict]]) -> tuple[str, ...]:
    """--axes (explicit subset) XOR the deterministic shard split."""
    if args.axes:
        axes = tuple(a.strip() for a in args.axes.split(",") if a.strip())
        unknown = [a for a in axes if a not in by_cell]
        if unknown:
            raise RuntimeError(f"--axes names unknown axes {unknown}; bank has {sorted(by_cell)}")
        return axes
    assign = shard_axes({a: len(v) for a, v in by_cell.items()}, args.num_shards)
    return tuple(a for a in sorted(assign) if assign[a] == args.shard_index)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = build_argparser()
    args = ap.parse_args()
    if args.import_check:
        _import_check()
        return 0
    if args.axes and args.num_shards != 1:
        ap.error("--axes is mutually exclusive with --num-shards > 1")
    if not (0 <= args.shard_index < args.num_shards):
        ap.error(
            f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}"
        )
    if args.phase != "gen":
        # Unit 3b seam: fail loud BEFORE any model load — never a silent no-op.
        raise NotImplementedError(
            f"--phase {args.phase} is unit 3b (P6 capture + Qwen3-Embedding-8B) — "
            "only --phase gen is implemented in unit 3a"
        )
    model_revision = _resolve_model_revision()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = cm2587.load_q35_model_and_tokenizer(device=device, revision=model_revision)
    bank = B.build_bank(tok)  # P0a string gates + P0b token gates, fail-loud
    print(f"[bank] {bank['n_contexts']} contexts / {bank['n_pairs']} pairs", flush=True)
    by_cell = group_contexts_by_cell(bank)
    axes = resolve_axes(args, by_cell)
    assert axes, f"shard {args.shard_index}/{args.num_shards} received no axes"
    cfg = build_cfg(
        args, bank_values_sha=bank["values_sha256"], axes=axes, model_revision=model_revision
    )
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    return PHASE_FNS[cfg.phase](cfg, bank, model, tok)


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
