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

Phase registry (``PHASES``): ``gen`` (unit 3a) | ``capture`` + ``embed``
(unit 3b, implemented here). ``capture`` (P6) runs the pinned
``issue2162_run.capture_answer_states(..., tail_inclusive=True,
return_boundaries=True)`` via ``_r()`` at capture batch 8 over ALL 32 blocks
with fp32 stores (fp16 stores overflow Qwen3-era massive activations — the
#2330 convention; the pin's two terminal fp16 output casts are neutralized by
a forwarding torch proxy swapped into the pinned module's globals around the
call, with post-call dtype + isfinite asserts), the #2330 ``hook_probe`` GATE
at blocks {16, 22, 30} BEFORE the production wave, the gate-4 EXACT boundary
compare against the gen rows' span fields (mismatch halts loud, never
degrades), v_C context-end states per unique context, and per-axis verified
upload THEN local delete (bounds the §9 disk high-water). ``embed`` embeds
every battery answer text with Qwen3-Embedding-8B under a STRUCTURAL
engine-parity gate: the realized vLLM must equal ``EXPECTED_EMBED_ENGINE``
(0.11.0 — the repo uv.lock pin; the DEFAULT route is running the phase under
the repo venv) or carry a PASSING ``--parity-report`` from
``run_engine_parity_probe`` (``--parity-probe-out`` mode); the realized engine
version is recorded inside every chunk npz, the perdraw/means npz, meta.json,
and the sentinel, and is part of the chunk-resume fingerprint, so 0.11.0- and
0.27.1-produced vectors can never silently mix (plan v3 §4.4, consistency
WARN-1). The embed machinery (chunked fp-keyed npz checkpoints, lazy engine,
first-chunk pilot gate, engine reap, token-length precheck that raises rather
than truncates) is transcribed from the pinned parent embed leg
``8265bcd:scripts/issue2564_embed.py`` — itself the
``scripts/issue2215_sepcmp_qwen_embed.py::embed_texts`` (line 52) port.

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
sentinel write, ``EXIT_PILOT_REFUSE`` (7) on a pilot-gate refusal (gen report
at ``manifests/pilot_gate_report.json``; embed report at
``manifests/embed_pilot_gate_report.json``), ``EXIT_PARITY_MISS`` (8) when a
``--parity-probe-out`` run measures a below-bar cosine (the report is still
written — a miss forces the 0.11.0 route), any crash = the process's own
non-zero exit observed by the poller. Rollout TEXT uploads unconditionally to
``{hf_prefix}/raw_completions/anchors/`` at end of shard, BEFORE any capture
consumes it (#779; upload mode is part of the SENTINEL fingerprint / per-cell
COMPLETION predicate, never the per-cell regime fingerprint — an ``--upload
none -> hf`` flip re-uploads banked rows, it never regenerates them; r1 g3).
Phase sentinels: ``battery_gen_done.json`` / ``battery_capture_done.json``
(per shard out-root) / ``battery_embed_done.json`` (embed is single-process).
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue2587_common as cm2587  # noqa: E402
from explore_persona_space.analysis.extraction import (  # noqa: E402
    _logits_to_keep_kwargs,
    extract_layer_activations,
)
from explore_persona_space.atomic_io import (  # noqa: E402
    atomic_replace,
    write_json_atomic,
    write_jsonl_atomic,
)
from explore_persona_space.experiments.issue1415.steering import generate_batch  # noqa: E402
from explore_persona_space.experiments.issue2587 import bank2587 as B  # noqa: E402
from explore_persona_space.experiments.issue2587 import bank2587_ffr as BF  # noqa: E402
from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded  # noqa: E402

# #628: vLLM reads this at import; --phase embed tokenizes with transformers
# BEFORE LLM(), so the default fork() would poison the EngineCore subprocess.
# Every vllm import in this driver is lazy (inside the embed phase), so this
# module-level setdefault always precedes it.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

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

# §4.4 Capture pins (P6): capture batch 8, ALL 32 blocks, fp32 stores; the
# #2330 hook_probe GATE at blocks {16, 22, 30} runs BEFORE the production
# wave (rel tol = the plan §7 / #2330 parity bar). Probe layers deliberately
# avoid block 31 (pre- vs post-final-RMSNorm divergence, extraction.py).
CAPTURE_BATCH = 8
CAPTURE_LAYERS = tuple(range(cm2587.N_LAYERS))
HOOK_PROBE_LAYERS = (16, 22, 30)
HOOK_PROBE_ROWS = 4
HOOK_REL_TOL = 1e-5

# §4.4 Embedding third space (the binding v3 requirement closing consistency
# WARN-1): Qwen3-Embedding-8B under the PARENT's engine. Constants transcribed
# from the pinned parent embed leg (8265bcd:scripts/issue2564_embed.py, itself
# the issue2215_sepcmp_qwen_embed.py::embed_texts port). EXPECTED_EMBED_ENGINE
# is the repo uv.lock vllm pin — the engine that produced the parent's banked
# vectors (tests/test_issue2587_battery_run.py pins it against uv.lock).
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_DIM = 4096
EMBED_CHUNK = 2500
EMBED_MAX_MODEL_LEN = 8192
EMBED_PILOT_CEILING_H = 2.0
EXPECTED_EMBED_ENGINE = "0.11.0"
PARITY_COS_MIN = 0.995
# WARN-1 instrument-identity control admission floors (r1 g4 C1): a consumed
# --parity-report must itself have been produced at >= these bars — a probe
# run with a deliberately weakened --parity-cos-min / --parity-n-anchors is
# REFUSED by _assert_engine_parity even when its parity_pass is true.
PARITY_N_ANCHORS_MIN = 10
EXIT_PARITY_MISS = 8

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
    # unit 3b knobs (defaulted so unit 3a call sites stay valid)
    capture_batch: int = CAPTURE_BATCH
    capture_dtype: str = "float32"
    embed_chunk: int = EMBED_CHUNK
    embed_max_model_len: int = EMBED_MAX_MODEL_LEN
    embed_pilot_ceiling_h: float = EMBED_PILOT_CEILING_H
    parity_report: Path | None = None
    anchors_roots: tuple[Path, ...] = ()

    @property
    def anchors_dir(self) -> Path:
        return self.out_root / "anchors"

    @property
    def manifest_dir(self) -> Path:
        return self.out_root / "manifests"

    @property
    def quarantine_dir(self) -> Path:
        return self.out_root / "quarantine"

    @property
    def va_dir(self) -> Path:
        return self.out_root / "capture" / "va2587"

    @property
    def vc_dir(self) -> Path:
        return self.out_root / "capture" / "vc2587"

    @property
    def embed_root(self) -> Path:
        return self.out_root / "embed"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument(
        "--bank-source",
        choices=("main", "ffr"),
        default="main",
        help="bank builder: 'main' = bank2587.build_bank (byte-identical default); "
        "'ffr' = bank2587_ffr.build_ffr_bank (the #2564 floor-failed re-elicitation "
        "grid, plan v6 §4.2 — pass with --out-root/--hf-prefix deltas)",
    )
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
    # unit 3b: capture knobs (plan §4.4 — batch 8, fp32; bfloat16 is an
    # explicit debug opt-down, never the production route)
    ap.add_argument("--capture-batch", type=int, default=CAPTURE_BATCH)
    ap.add_argument("--capture-dtype", choices=("float32", "bfloat16"), default="float32")
    # unit 3b: embed knobs + the engine-parity contingency surface
    ap.add_argument("--embed-chunk", type=int, default=EMBED_CHUNK)
    ap.add_argument("--embed-max-model-len", type=int, default=EMBED_MAX_MODEL_LEN)
    ap.add_argument("--embed-pilot-ceiling-h", type=float, default=EMBED_PILOT_CEILING_H)
    ap.add_argument(
        "--parity-report",
        default=None,
        help="PASSING engine-parity probe report JSON — required to run --phase embed "
        "under a vLLM other than the 0.11.0 parity reference (plan §4.4)",
    )
    ap.add_argument(
        "--anchors-root",
        action="append",
        default=None,
        help="root(s) holding the gen phase's anchors/ for --phase embed "
        "(default: the out-root itself plus its shard* subdirs)",
    )
    # --parity-probe-out MODE (plan §4.4 contingency): re-embed banked parent
    # anchor texts under the CURRENT engine vs the banked 0.11.0 vectors,
    # write the report, exit 0 on pass / EXIT_PARITY_MISS on a miss.
    ap.add_argument("--parity-probe-out", default=None)
    ap.add_argument(
        "--parity-anchor-texts", default=None, help="parent anchors JSONL (probe inputs)"
    )
    ap.add_argument(
        "--parity-banked-npz", default=None, help="banked perdraw npz (0.11.0 reference vectors)"
    )
    ap.add_argument("--parity-n-anchors", type=int, default=PARITY_N_ANCHORS_MIN)
    ap.add_argument("--parity-cos-min", type=float, default=PARITY_COS_MIN)
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
        capture_batch=args.capture_batch,
        capture_dtype=args.capture_dtype,
        embed_chunk=args.embed_chunk,
        embed_max_model_len=args.embed_max_model_len,
        embed_pilot_ceiling_h=args.embed_pilot_ceiling_h,
        parity_report=Path(args.parity_report) if args.parity_report else None,
        anchors_roots=tuple(Path(p) for p in (args.anchors_root or ())),
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
            r.capture_answer_states,  # the branch-only kwargs exist at the pin
            {"payloads", "positions", "tail_inclusive", "return_boundaries"},
        ),
        (r._right_pad, {"rows", "pad_id", "device"}),
        (extract_layer_activations, {"attention_mask", "return_logits", "detach_to_cpu"}),
        (upload_dir_sharded, {"shard_glob", "resume_skip", "delete_local"}),
    ):
        params = set(inspect.signature(fn).parameters)
        missing = needed - params
        assert not missing, (getattr(fn, "__name__", fn), sorted(missing))
    for name in ("eot_tail_ids", "cap_hit"):
        assert callable(getattr(r, name)), name
    assert callable(_logits_to_keep_kwargs) and callable(atomic_replace)
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
    # FFR bank seam (--bank-source ffr; plan v6 §4.2): the manifest-sourced
    # builder + its gates resolve even when the smoke never takes the branch.
    for name in (
        "fetch_ffr_manifest",
        "build_ffr_bank_strings",
        "run_ffr_token_gates",
        "build_ffr_bank",
    ):
        assert callable(getattr(BF, name)), name
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
    re-splits — and EXCLUDES ``upload`` (r1 g3: upload does not change what
    was GENERATED, so an --upload none -> hf flip must re-upload the banked
    rows, never regenerate them; the fits.py regime_key shape). The langow
    review-finding-1 invariant — a --upload none run must never satisfy a
    later --upload hf run — lives in the SENTINEL fps (phase_gen/
    phase_capture pass ``upload`` in their sentinel ``extra``; embed hashes
    it into ``sent_fp``) and in the per-cell COMPLETION predicates
    (``_capture_cell_complete`` requires the manifest's verified-upload
    record under upload=hf)."""
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
    # upload rides the SENTINEL fp only (never the cell grain): a none-run
    # sentinel cannot satisfy an hf run, but the completed cells resume.
    sent_fp = _regime_fp(cfg, {"phase": "gen", "axes": sorted(cfg.axes), "upload": cfg.upload})
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


# ── phase: capture (P6 — unit 3b) ─────────────────────────────────────────


@dataclass
class _PinCaptureCfg:
    """Exactly the four fields the pinned ``capture_answer_states`` reads off
    its ``cfg`` argument (``layers`` / ``hidden`` / ``capture_batch`` /
    ``device`` — verified against the sha-pinned blob)."""

    layers: list[int]
    hidden: int
    capture_batch: int
    device: str


class _Fp32Torch:
    """Forwarding torch proxy: ``.float16`` resolves to ``torch.float32``;
    every other attribute forwards to the real torch. Swapped into the
    sha-pinned issue2162 module's globals ONLY around the
    ``capture_answer_states`` call — the blob's sole in-body ``float16``
    references are its two terminal output casts (blob lines 1670/1676), so
    the pinned body executes verbatim while the fp32-store mandate (plan
    §4.4; fp16 overflows Qwen3-era massive activations, #2330) holds."""

    float16 = torch.float32

    def __getattr__(self, name):
        return getattr(torch, name)


def _capture_answer_states_fp32(
    pin_cfg: _PinCaptureCfg,
    model,
    tok,
    ctx_ids_by_row: list[list[int]],
    completions: list[str],
    eot_ids: list[int],
) -> dict:
    """Run the pinned ``capture_answer_states(..., tail_inclusive=True,
    return_boundaries=True)`` with fp32 output stores.

    Boundaries stay derived from the pin's OWN tokenization state
    (load-bearing for gate-4). Post-call asserts make a shim failure loud:
    non-fp32 or non-finite outputs raise, never persist."""
    r = _r()
    real_torch = r.torch
    r.torch = _Fp32Torch()
    try:
        out = r.capture_answer_states(
            pin_cfg,
            model,
            tok,
            ctx_ids_by_row,
            completions,
            eot_ids,
            tail_inclusive=True,
            return_boundaries=True,
        )
    finally:
        r.torch = real_torch
    for key in ("va_span", "va_tail_incl"):
        t = out[key]
        if t.dtype != torch.float32:
            raise RuntimeError(f"[capture] {key} dtype {t.dtype} != float32 — fp32 shim inactive")
        if not torch.isfinite(t).all():
            raise RuntimeError(f"[capture] non-finite values in {key} (fp32 store, plan §4.4)")
    return out


def _model_hidden(model) -> int:
    """hidden_size off the model config (text_config fallback for multimodal
    wrappers); fail loud when unresolvable."""
    c = getattr(model, "config", None)
    h = getattr(c, "hidden_size", None)
    if not isinstance(h, int):
        h = getattr(getattr(c, "text_config", None), "hidden_size", None)
    if not isinstance(h, int) or h <= 0:
        raise RuntimeError(f"could not resolve hidden_size from {type(model).__name__}.config")
    return int(h)


def _model_max_positions(model) -> int:
    """max_position_embeddings off the model config (text_config fallback);
    fail loud when unresolvable — the capture-capacity floor reads this."""
    c = getattr(model, "config", None)
    m = getattr(c, "max_position_embeddings", None)
    if not isinstance(m, int):
        m = getattr(getattr(c, "text_config", None), "max_position_embeddings", None)
    if not isinstance(m, int) or m <= 0:
        raise RuntimeError(
            f"could not resolve max_position_embeddings from {type(model).__name__}.config"
        )
    return int(m)


def _capture_cell_fp(cfg: Cfg, cell: str) -> str:
    """Capture resume fingerprint = the gen regime base + capture knobs."""
    return _regime_fp(
        cfg,
        {
            "phase": "capture",
            "cell": cell,
            "capture_batch": cfg.capture_batch,
            "capture_dtype": cfg.capture_dtype,
            "layers": list(CAPTURE_LAYERS),
        },
    )


def _hook_probe(cfg: Cfg, model, tok, probe_rows_ids: list[list[int]]) -> dict:
    """#2330 ``gate_hook_probe`` port (plan §4.4/§7) — a GATE, not a
    diagnostic: the production capture path (``extract_layer_activations``
    forward hooks) vs a second ``output_hidden_states=True`` forward at
    blocks {16, 22, 30} (block L's hook == ``hidden_states[L+1]``; the probe
    set avoids block 31's pre/post-final-RMSNorm caveat). Per-layer
    rel = ||a-b|| / (||a|| + 1e-30) <= HOOK_REL_TOL; a miss persists the
    report and halts LOUD before the production wave."""
    r = _r()
    ids, mask = r._right_pad(probe_rows_ids, tok.pad_token_id, cfg.device)
    layers = list(HOOK_PROBE_LAYERS)
    captured = extract_layer_activations(model, ids, layers, attention_mask=mask)
    with torch.no_grad():
        fwd = model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            **_logits_to_keep_kwargs(model, False),
        )
    hs = fwd.hidden_states
    blocks = cm2587.resolve_q35_decoder_blocks(model)  # 32-block assert, fail loud
    assert len(hs) == len(blocks) + 1, (len(hs), len(blocks))
    per_layer: dict[int, float] = {}
    for lyr in layers:
        a = captured[lyr].float()
        b = hs[lyr + 1].float()
        assert a.shape == b.shape, (lyr, a.shape, b.shape)
        per_layer[lyr] = float((a - b).norm() / (a.norm() + 1e-30))
    report = {
        "probe_layers": layers,
        "layer_index_mapping": {str(lyr): lyr + 1 for lyr in layers},
        "per_layer_rel": {str(lyr): per_layer[lyr] for lyr in layers},
        "rel_tol": HOOK_REL_TOL,
        "n_probe_rows": len(probe_rows_ids),
        "hidden_states_tuple_len": len(hs),
        "verdict": "pass" if all(v <= HOOK_REL_TOL for v in per_layer.values()) else "fail",
        "repro": _repro(cfg, "capture-hook-probe"),
    }
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cfg.manifest_dir / "hook_probe_report.json", report)
    if report["verdict"] != "pass":
        raise RuntimeError(
            f"[capture] hook probe FAILED: per-layer rel {per_layer} > tol {HOOK_REL_TOL} "
            "(#2330 gate) — report at manifests/hook_probe_report.json"
        )
    print(f"[capture] hook probe pass: max rel {max(per_layer.values()):.2e}", flush=True)
    return report


GATE4_FIELDS = ("ctx_len", "n_completion_tokens", "span_start", "span_end", "tail_end")


def _gate4_exact_compare(cell: str, gen_rows: list[dict], boundaries: list[dict]) -> None:
    """Plan §4.4 gate 4: EXACT integer-equality comparison of the capture
    path's OWN per-row boundary records against the gen rows' span fields.
    Exact means exact — any mismatch halts loud, never degrades."""
    if len(gen_rows) != len(boundaries):
        raise RuntimeError(
            f"[gate-4:{cell}] row-count mismatch: {len(gen_rows)} gen rows vs "
            f"{len(boundaries)} capture boundaries"
        )
    bad = []
    for i, (g, bnd) in enumerate(zip(gen_rows, boundaries, strict=True)):
        gen_rec = {
            "ctx_len": g["ctx_len"],
            "n_completion_tokens": g["n_completion_tokens_gen"],
            "span_start": g["span_start"],
            "span_end": g["span_end"],
            "tail_end": g["tail_end"],
        }
        cap_rec = {k: bnd[k] for k in GATE4_FIELDS}
        if gen_rec != cap_rec:
            bad.append((i, g["context_id"], g["draw"], gen_rec, cap_rec))
    if bad:
        i, cid, draw, gen_rec, cap_rec = bad[0]
        raise RuntimeError(
            f"[gate-4:{cell}] EXACT boundary compare FAILED on {len(bad)}/{len(gen_rows)} rows; "
            f"first: row {i} ({cid} d{draw}) gen={gen_rec} capture={cap_rec}"
        )


def _load_gen_axis(cfg: Cfg, cell: str, ctxs: list[dict]) -> tuple[dict, list[dict]]:
    """Load + validate one axis's gen outputs (fail loud): done manifest whose
    regime_fp matches THIS invocation's gen fingerprint (same bank, draws,
    seeds, caps, revision), full (context_id, draw) grid; rows re-sorted to
    (context_id, draw) — the capture row order."""
    man = _read_json(cfg.manifest_dir / f"anchors_{cell}.done.json")
    if man is None:
        raise RuntimeError(f"[capture:{cell}] gen done manifest missing — run --phase gen first")
    if man.get("regime_fp") != _cell_fp(cfg, "gen", cell):
        raise RuntimeError(
            f"[capture:{cell}] gen manifest regime_fp mismatch — the anchors were generated "
            "under a different regime than this capture invocation's args"
        )
    rows = _read_jsonl(cfg.anchors_dir / f"anchors_{cell}.jsonl")
    expected = {(c["id"], d) for c in ctxs for d in range(cfg.draws)}
    got = {(r["context_id"], int(r["draw"])) for r in rows}
    if got != expected or len(rows) != len(expected):
        raise RuntimeError(
            f"[capture:{cell}] anchors row grid mismatch: {len(rows)} rows, "
            f"{len(got ^ expected)} symmetric-difference keys"
        )
    rows.sort(key=lambda r: (r["context_id"], int(r["draw"])))
    return man, rows


def _capture_context_end(cfg: Cfg, model, tok, ctx_ids_list: list[list[int]], hidden: int):
    """v_C context-end states: right-padded chunked forwards over the axis's
    unique contexts; per row the layer-stacked block output at position
    ctx_len - 1 (the pinned ``capture_bank`` v_ce pattern), fp32, ALL 32
    blocks."""
    r = _r()
    layers = list(CAPTURE_LAYERS)
    vc = torch.zeros((len(ctx_ids_list), len(layers), hidden), dtype=torch.float32)
    for start in range(0, len(ctx_ids_list), cfg.capture_batch):
        batch = ctx_ids_list[start : start + cfg.capture_batch]
        ids, mask = r._right_pad(batch, tok.pad_token_id, cfg.device)
        captured = extract_layer_activations(model, ids, layers, attention_mask=mask)
        for j, row_ids in enumerate(batch):
            vc[start + j] = torch.stack(
                [captured[lyr][j, len(row_ids) - 1].float() for lyr in layers]
            ).cpu()
        del captured
    if not torch.isfinite(vc).all():
        raise RuntimeError("[capture] non-finite values in v_C store")
    return vc


def _torch_save_atomic(obj: dict, path: Path) -> None:
    """Atomic torch.save via the process-unique-temp helper (#2336)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        torch.save(obj, tmp)


def _capture_cell_complete(cfg: Cfg, cell: str) -> bool:
    """Per-axis capture resume predicate: done manifest at the CURRENT capture
    fingerprint; upload=hf trusts the manifest's verified-upload record (the
    stores are deleted locally after the verified upload); upload=none
    requires the local stores on disk."""
    m = _read_json(cfg.manifest_dir / f"capture_{cell}.done.json")
    if m is None or m.get("regime_fp") != _capture_cell_fp(cfg, cell):
        return False
    if cfg.upload == "hf":
        return bool(m.get("uploaded"))
    return (cfg.va_dir / f"{cell}.pt").is_file() and (cfg.vc_dir / f"{cell}.pt").is_file()


def _capture_cell(
    cfg: Cfg,
    model,
    tok,
    eot_ids: list[int],
    cell: str,
    ctxs: list[dict],
    man: dict,
    rows: list[dict],
    hidden: int,
) -> dict:
    """One axis: fp32-shimmed pinned capture (both v_A pooling twins) ->
    gate-4 EXACT boundary compare -> per-row floor check -> v_C context-end
    capture -> fp32 .pt stores -> per-axis verified upload + local delete
    (upload=hf) -> done manifest."""
    t0 = time.time()
    ctx_by_id = {c["id"]: c for c in ctxs}
    ctx_ids_by_row = [IDS_FN(tok, ctx_by_id[r["context_id"]]) for r in rows]
    completions = [r["text"] for r in rows]
    pin_cfg = _PinCaptureCfg(
        layers=list(CAPTURE_LAYERS),
        hidden=hidden,
        capture_batch=cfg.capture_batch,
        device=cfg.device,
    )
    print(f"[capture:{cell}] v_A wave: {len(rows)} rows batch={cfg.capture_batch}", flush=True)
    out = _capture_answer_states_fp32(pin_cfg, model, tok, ctx_ids_by_row, completions, eot_ids)
    _gate4_exact_compare(cell, rows, out["boundaries"])
    floor = int(man["capture_max_model_len_floor"])
    max_tail = max(b["tail_end"] for b in out["boundaries"])
    assert max_tail <= floor, (cell, max_tail, floor)
    ctx_ids_list = [IDS_FN(tok, c) for c in ctxs]
    row_by_key = {(r["context_id"], int(r["draw"])): r for r in rows}
    for j, c in enumerate(ctxs):
        # gen recorded ctx_len from the SAME ids_fn — cross-check per context.
        g = row_by_key[(c["id"], 0)]
        assert len(ctx_ids_list[j]) == g["ctx_len"], (c["id"], len(ctx_ids_list[j]), g["ctx_len"])
    print(f"[capture:{cell}] v_C wave: {len(ctxs)} contexts", flush=True)
    vc = _capture_context_end(cfg, model, tok, ctx_ids_list, hidden)
    row_meta = [
        {
            "context_id": r["context_id"],
            "cell": r["cell"],
            "draw": int(r["draw"]),
            "think_leak": bool(r["think_leak"]),
            "cap_hit": bool(r["cap_hit"]),
            "ctx_len": r["ctx_len"],
            "n_completion_tokens": r["n_completion_tokens_gen"],
            "span_start": r["span_start"],
            "span_end": r["span_end"],
            "tail_end": r["tail_end"],
        }
        for r in rows
    ]
    store_common = {
        "cell": cell,
        "layers": list(CAPTURE_LAYERS),
        "hidden": hidden,
        "dtype": "float32",
        "layer_convention": (
            "decoder-block outputs via forward hooks (analysis/extraction.py): captured[L] == "
            "hidden_states[L+1]; block 31 is the RAW pre-final-RMSNorm output"
        ),
        "regime_fp": _capture_cell_fp(cfg, cell),
        "repro": _repro(cfg, "capture"),
    }
    va_path = cfg.va_dir / f"{cell}.pt"
    vc_path = cfg.vc_dir / f"{cell}.pt"
    _torch_save_atomic(
        {
            **store_common,
            "va_tail_incl": out["va_tail_incl"],
            "va_span": out["va_span"],
            "pooling": out["pooling"],
            "n_completion_tokens": out["n_completion_tokens"],
            "empty_rows": out["empty_rows"],
            "boundaries": out["boundaries"],
            "rows": row_meta,
        },
        va_path,
    )
    _torch_save_atomic(
        {
            **store_common,
            "vc": vc,
            "pooling": {"vc": "context-end state at position ctx_len-1 (capture_bank v_ce)"},
            "context_ids": [c["id"] for c in ctxs],
            "ctx_lens": [len(x) for x in ctx_ids_list],
        },
        vc_path,
    )
    upload: dict = {"mode": cfg.upload}
    if cfg.upload == "hf":
        # Per-axis verified upload THEN local delete — bounds the §9 disk
        # high-water (upload_dir_sharded verifies before deleting;
        # resume_skip=False: same-size different-bytes false-skip risk).
        for sub, path in (("va2587", va_path), ("vc2587", vc_path)):
            res = upload_dir_sharded(
                path.parent,
                cfg.hf_repo,
                f"{cfg.hf_prefix}/analysis_tensors/{sub}",
                shard_glob=path.name,
                resume_skip=False,
                delete_local=True,
            )
            upload[sub] = {
                "repo_id": res.repo_id,
                "uploaded": len(res.uploaded),
                "deleted": len(res.deleted),
                "rerouted": len(res.rerouted),
            }
    manifest = {
        "cell": cell,
        "regime_fp": _capture_cell_fp(cfg, cell),
        "gen_regime_fp": man["regime_fp"],
        "n_contexts": len(ctxs),
        "n_rows": len(rows),
        "n_empty_rows": len(out["empty_rows"]),
        "max_tail_end": max_tail,
        "capture_max_model_len_floor": floor,
        "capture_batch": cfg.capture_batch,
        "layers": list(CAPTURE_LAYERS),
        "hidden": hidden,
        "dtype": "float32",
        "uploaded": cfg.upload == "hf",
        "upload": upload,
        "elapsed_s": round(time.time() - t0, 1),
        "repro": _repro(cfg, "capture"),
    }
    write_json_atomic(cfg.manifest_dir / f"capture_{cell}.done.json", manifest)
    print(f"[capture] unit {cell} rows={len(rows)} elapsed={time.time() - t0:.1f}s", flush=True)
    return manifest


def phase_capture(cfg: Cfg, bank: dict, model, tok) -> int:
    """P6 all-layer fp32 battery capture for THIS shard's axes (plan §4.4):
    verify every assigned axis's gen outputs -> max_model_len-floor capacity
    assert -> hook probe (a gate, BEFORE the production wave) -> per-axis
    fp32-shimmed pinned capture + gate-4 + v_C + upload-then-delete -> shard
    sentinel ``battery_capture_done.json``."""
    print(
        f"[phase=capture] start shard={cfg.shard_index}/{cfg.num_shards} axes={','.join(cfg.axes)}",
        flush=True,
    )
    by_cell = group_contexts_by_cell(bank)
    unknown = [a for a in cfg.axes if a not in by_cell]
    if unknown:
        raise RuntimeError(f"unknown axes {unknown}; bank has {sorted(by_cell)}")
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    ctxs_by_axis = {a: apply_max_carriers(by_cell[a], cfg.max_carriers) for a in cfg.axes}
    sentinel = cfg.out_root / "battery_capture_done.json"
    sent_fp = _regime_fp(
        cfg,
        {
            "phase": "capture",
            "axes": sorted(cfg.axes),
            "capture_batch": cfg.capture_batch,
            "capture_dtype": cfg.capture_dtype,
            # sentinel-only (never the cell grain) — see _regime_fp docstring
            "upload": cfg.upload,
        },
    )
    pending = [a for a in cfg.axes if not _capture_cell_complete(cfg, a)]
    s = _read_json(sentinel)
    if not pending and s is not None and s.get("regime_fp") == sent_fp:
        logger.info("[capture] all axes complete + sentinel present — skipping")
        return 0
    hidden = _model_hidden(model)
    if cfg.model_id == B.MODEL_ID:
        assert hidden == cm2587.HIDDEN, (hidden, cm2587.HIDDEN)
    # Every assigned axis's gen outputs load + validate up front (fail loud
    # before any GPU wave), and the model must have capacity for the floor
    # recorded by gen (capture_max_model_len_floor = max_ctx + 2*max_new).
    gen_data = {a: _load_gen_axis(cfg, a, ctxs_by_axis[a]) for a in cfg.axes}
    floor = max(int(m["capture_max_model_len_floor"]) for m, _ in gen_data.values())
    max_pos = _model_max_positions(model)
    if max_pos < floor:
        raise RuntimeError(
            f"[capture] model max_position_embeddings {max_pos} < required floor {floor} "
            "(gen manifests' capture_max_model_len_floor) — capture cannot proceed"
        )
    eot_ids = _r().eot_tail_ids(tok)
    if pending:
        # Hook probe on REAL rows of the first pending axis, BEFORE the wave.
        probe_axis = pending[0]
        _man, rows = gen_data[probe_axis]
        ctx_by_id = {c["id"]: c for c in ctxs_by_axis[probe_axis]}
        probe_rows = [r for r in rows if str(r["text"]).strip()][:HOOK_PROBE_ROWS]
        if not probe_rows:
            raise RuntimeError(f"[capture:{probe_axis}] no non-empty rows for the hook probe")
        probe_ids = [
            IDS_FN(tok, ctx_by_id[r["context_id"]])
            + tok(r["text"], add_special_tokens=False)["input_ids"]
            + eot_ids
            for r in probe_rows
        ]
        _hook_probe(cfg, model, tok, probe_ids)
    for a in cfg.axes:
        if _capture_cell_complete(cfg, a):
            logger.info("[capture:%s] done manifest present — skipping", a)
            continue
        man, rows = gen_data[a]
        _capture_cell(cfg, model, tok, eot_ids, a, ctxs_by_axis[a], man, rows, hidden)
    write_json_atomic(
        sentinel,
        {
            "issue": ISSUE,
            "regime_fp": sent_fp,
            "shard_index": cfg.shard_index,
            "num_shards": cfg.num_shards,
            "axes": sorted(cfg.axes),
            "cells": {a: _read_json(cfg.manifest_dir / f"capture_{a}.done.json") for a in cfg.axes},
            "hook_probe": _read_json(cfg.manifest_dir / "hook_probe_report.json"),
            "repro": _repro(cfg, "capture"),
        },
    )
    print("[phase=capture] sentinel written", flush=True)
    return 0


# ── phase: embed (Qwen3-Embedding-8B third space — unit 3b) ───────────────


def _realized_vllm_version() -> str:
    """The CURRENT interpreter's vLLM version — the engine-parity gate input."""
    import vllm

    return str(vllm.__version__)


def _assert_engine_parity(realized: str, parity_report: Path | None) -> dict:
    """Plan §4.4 instrument-version parity gate (structural, fail loud).

    DEFAULT route: run the phase under the repo venv, whose uv.lock pins
    vLLM == EXPECTED_EMBED_ENGINE (0.11.0) — the engine that produced the
    parent's banked embeddings, so parity holds by construction. Any OTHER
    realized engine requires a PASSING --parity-report (produced by
    ``run_engine_parity_probe``); a probe miss forces the 0.11.0 route.
    0.11.0-produced 7B vectors are never compared against
    differently-versioned 9B vectors unprobed."""
    if realized == EXPECTED_EMBED_ENGINE:
        return {
            "vllm_version": realized,
            "parity_mode": "repo-pin",
            "reference_engine": EXPECTED_EMBED_ENGINE,
        }
    if parity_report is None:
        raise RuntimeError(
            f"[embed] engine vLLM=={realized} != parity reference {EXPECTED_EMBED_ENGINE} "
            "(the parent's banked vectors are 0.11.0-produced; plan §4.4). Run this phase "
            "under the repo venv (uv run python), or pass --parity-report from a PASSING "
            "--parity-probe-out run."
        )
    rep = _read_json(parity_report)
    if rep is None:
        raise RuntimeError(f"[embed] --parity-report {parity_report} missing or unparseable")
    for key, want in (
        ("parity_pass", True),
        ("reference_engine", EXPECTED_EMBED_ENGINE),
        ("engine", realized),
    ):
        if rep.get(key) != want:
            raise RuntimeError(
                f"[embed] parity report {parity_report}: {key}={rep.get(key)!r} != {want!r} "
                f"— a probe miss forces the {EXPECTED_EMBED_ENGINE} route (plan §4.4)"
            )
    # r1 g4 C1: the report's OWN admission criteria are enforced, not just its
    # verdict — a probe run with a deliberately weakened --parity-cos-min or
    # --parity-n-anchors must never be admitted (bool excluded: True is an int).
    n_anchors = rep.get("n_anchors")
    if (
        isinstance(n_anchors, bool)
        or not isinstance(n_anchors, int)
        or (n_anchors < PARITY_N_ANCHORS_MIN)
    ):
        raise RuntimeError(
            f"[embed] parity report {parity_report}: n_anchors={n_anchors!r} below the "
            f"admission floor {PARITY_N_ANCHORS_MIN} (or non-int) — re-run the probe at "
            f"--parity-n-anchors >= {PARITY_N_ANCHORS_MIN}"
        )
    bar = rep.get("cos_min_bar")
    if isinstance(bar, bool) or not isinstance(bar, int | float) or bar < PARITY_COS_MIN:
        raise RuntimeError(
            f"[embed] parity report {parity_report}: cos_min_bar={bar!r} below the plan bar "
            f"{PARITY_COS_MIN} (or non-numeric) — a weakened-bar probe report is never "
            "admitted; re-run the probe at the plan bar"
        )
    return {
        "vllm_version": realized,
        "parity_mode": "parity-probe",
        "reference_engine": EXPECTED_EMBED_ENGINE,
        "parity_report": rep,
    }


def _resolve_embed_revision() -> str:
    """Pin the embed model's main -> resolved sha ONCE, BEFORE any loader init
    (tokenizer + LLM both take ``revision=``; provenance label == loaded
    bytes; #2061)."""
    from huggingface_hub import HfApi

    sha = HfApi().model_info(EMBED_MODEL).sha
    assert sha, f"could not resolve model revision for {EMBED_MODEL}"
    return str(sha)


def _load_embed_tokenizer(revision: str):
    """Embed-model tokenizer for the token-length precheck (network seam)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(EMBED_MODEL, revision=revision)


def _make_embed_llm(revision: str, max_model_len: int):
    """vLLM pooling-runner engine ctor — the ONLY engine-construction seam
    (transcribed from the pinned parent embed leg)."""
    from vllm import LLM

    return LLM(
        model=EMBED_MODEL,
        revision=revision,
        runner="pooling",
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
    )


def _reap_engine(llm: object) -> None:
    """Reap the vLLM engine before return (gotchas.md vLLM v1 reaping recipe;
    transcribed from the pinned parent embed leg — best-effort teardown with
    logged warnings; the caller's terminal follows)."""
    import gc

    try:
        core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
        else:
            executor = getattr(getattr(llm, "llm_engine", None), "model_executor", None)
            if executor is not None and hasattr(executor, "shutdown"):
                executor.shutdown()
    except Exception as e:
        logger.warning("[embed] engine reap warning: %s: %s", type(e).__name__, e)
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as e:
        logger.warning("[embed] destroy_process_group warning: %s: %s", type(e).__name__, e)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(1.0)  # subprocess teardown is async


def _embed_regime_fp(
    rows: list[dict], chunk: int, max_model_len: int, revision: str, engine_version: str
) -> str:
    """Chunk-resume fingerprint: generating params + file-read row identities
    (bit-exact inputs — safe to hash; never recomputed floats, #1336) + the
    RESOLVED embed-model revision + the realized ENGINE version (a chunk
    embedded under a different vLLM must never satisfy this run's resume —
    the parity rule made structural)."""
    ids = [
        (
            r["context_id"],
            int(r["draw"]),
            hashlib.sha256(r["text"].encode("utf-8")).hexdigest()[:16],
        )
        for r in rows
    ]
    payload = json.dumps(
        [EMBED_MODEL, revision, engine_version, EMBED_DIM, max_model_len, chunk, ids],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class EmbedPilotRefuse(RuntimeError):
    """First-computed-chunk pilot projection exceeded the ceiling (a designed
    halt — ``phase_embed`` maps it to EXIT_PILOT_REFUSE; report on disk)."""


def _embed_rows(
    texts: list[str],
    *,
    chunks_dir: Path,
    fp: str,
    chunk: int,
    max_model_len: int,
    revision: str,
    engine_version: str,
    pilot_ceiling_h: float,
    pilot_report_path: Path,
) -> np.ndarray:
    """Chunked embed with per-chunk atomic npz checkpoints + fp-gated resume +
    a first-computed-chunk pilot gate (transcribed from the pinned parent
    embed leg ``8265bcd:scripts/issue2564_embed.py::embed_rows``, itself the
    ``issue2215_sepcmp_qwen_embed.py::embed_texts`` port). Returns the raw
    (n, EMBED_DIM) float32 matrix; the engine is created lazily on the first
    PENDING chunk (an all-resumed invocation never loads it) and reaped
    before return."""
    n = len(texts)
    n_chunks = (n + chunk - 1) // chunk
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out = np.zeros((n, EMBED_DIM), dtype=np.float32)
    llm = None
    pilot_done = False
    try:
        for k in range(n_chunks):
            lo, hi = k * chunk, min((k + 1) * chunk, n)
            ck_path = chunks_dir / f"chunk_{k:03d}.npz"
            if ck_path.is_file():
                z = np.load(ck_path, allow_pickle=False)
                if (
                    str(z["fp"]) == fp
                    and int(z["lo"]) == lo
                    and int(z["hi"]) == hi
                    and z["emb"].shape == (hi - lo, EMBED_DIM)
                ):
                    out[lo:hi] = z["emb"].astype(np.float32)
                    print(
                        f"[embed] unit {k + 1}/{n_chunks} chunk_{k:03d} resumed rows={hi - lo}",
                        flush=True,
                    )
                    continue
                print(
                    f"[embed] chunk_{k:03d} checkpoint stale (regime changed) — recomputing",
                    flush=True,
                )
            if llm is None:
                print(f"[embed] loading {EMBED_MODEL}@{revision} (pooling runner)", flush=True)
                llm = _make_embed_llm(revision, max_model_len)
            t0 = time.monotonic()
            res = llm.embed(texts[lo:hi], use_tqdm=False)
            arr = np.array([r.outputs.embedding for r in res], dtype=np.float32)
            assert arr.shape == (hi - lo, EMBED_DIM), arr.shape
            elapsed = time.monotonic() - t0
            # np.savez appends .npz to path-named non-.npz targets — hand it
            # an OPEN handle inside the process-unique atomic replace
            # (#2336/#1092). The realized engine version rides EVERY chunk.
            with atomic_replace(ck_path) as tmp:
                with open(tmp, "wb") as fh:
                    np.savez(
                        fh,
                        emb=arr.astype(np.float16),
                        lo=lo,
                        hi=hi,
                        fp=fp,
                        vllm_version=engine_version,
                    )
            out[lo:hi] = arr
            print(
                f"[embed] unit {k + 1}/{n_chunks} chunk_{k:03d} rows={hi - lo} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            if not pilot_done:
                pilot_done = True
                projected_h = elapsed * n_chunks / 3600.0
                write_json_atomic(
                    pilot_report_path,
                    {
                        "issue": ISSUE,
                        "phase": "embed-pilot",
                        "first_chunk_rows": hi - lo,
                        "first_chunk_elapsed_s": round(elapsed, 2),
                        "n_chunks": n_chunks,
                        "projected_wall_h": round(projected_h, 4),
                        "ceiling_h": pilot_ceiling_h,
                        "verdict": "pass" if projected_h <= pilot_ceiling_h else "refuse",
                    },
                )
                if projected_h > pilot_ceiling_h:
                    raise EmbedPilotRefuse(
                        f"projected {projected_h:.2f}h > ceiling {pilot_ceiling_h}h — "
                        f"report at {pilot_report_path}"
                    )
    finally:
        if llm is not None:
            _reap_engine(llm)
    return out


def _anchor_roots(cfg: Cfg) -> list[Path]:
    """Roots that may hold the gen phase's outputs: --anchors-root overrides;
    default = the out-root itself plus its shard* subdirs (gen's per-leg
    out-roots under --num-shards > 1)."""
    if cfg.anchors_roots:
        roots = list(cfg.anchors_roots)
    else:
        roots = [cfg.out_root, *sorted(p for p in cfg.out_root.glob("shard*") if p.is_dir())]
    found = [r for r in roots if (r / "anchors").is_dir()]
    if not found:
        raise RuntimeError(
            f"[embed] no anchors/ dir under any root {[str(r) for r in roots]} — "
            "run --phase gen first (or pass --anchors-root)"
        )
    return found


def _collect_anchor_rows(cfg: Cfg, by_cell: dict[str, list[dict]]) -> tuple[list[dict], int]:
    """Discover + validate every assigned axis's gen rows across the anchor
    roots (fail loud: exactly ONE root per axis, gen done manifest present,
    full (context_id, draw) grid); returns rows sorted (cell, context_id,
    draw) plus the count of empty-text rows skipped (recorded — the #2215
    port's convention)."""
    roots = _anchor_roots(cfg)
    rows: list[dict] = []
    n_empty = 0
    for cell in cfg.axes:
        ctxs = apply_max_carriers(by_cell[cell], cfg.max_carriers)
        hits = [r for r in roots if (r / "anchors" / f"anchors_{cell}.jsonl").is_file()]
        if len(hits) != 1:
            raise RuntimeError(
                f"[embed] axis {cell!r}: {len(hits)} anchor roots hold anchors_{cell}.jsonl "
                f"({[str(h) for h in hits]}) — need exactly one"
            )
        root = hits[0]
        if not (root / "manifests" / f"anchors_{cell}.done.json").is_file():
            raise RuntimeError(
                f"[embed] axis {cell!r}: anchors present but gen done manifest missing "
                f"under {root} — gen did not complete this axis"
            )
        cell_rows = _read_jsonl(root / "anchors" / f"anchors_{cell}.jsonl")
        expected = {(c["id"], d) for c in ctxs for d in range(cfg.draws)}
        got = {(r["context_id"], int(r["draw"])) for r in cell_rows}
        if got != expected or len(cell_rows) != len(expected):
            raise RuntimeError(
                f"[embed] axis {cell!r}: anchors row grid mismatch under {root} "
                f"({len(cell_rows)} rows vs expected {len(expected)})"
            )
        for r in cell_rows:
            if not str(r.get("text", "")).strip():
                n_empty += 1
                continue
            rows.append(r)
    if not rows:
        raise RuntimeError("[embed] empty anchor selection after load — nothing to embed")
    rows.sort(key=lambda r: (r["cell"], r["context_id"], int(r["draw"])))
    return rows, n_empty


def run_engine_parity_probe(
    anchor_texts_jsonl: Path,
    banked_npz: Path,
    out_path: Path,
    *,
    n_anchors: int = PARITY_N_ANCHORS_MIN,
    cos_min: float = PARITY_COS_MIN,
    max_model_len: int = EMBED_MAX_MODEL_LEN,
) -> dict:
    """Plan §4.4 contingency: re-embed ~``n_anchors`` PARENT anchor texts
    under the CURRENT engine and compare against the banked reference-engine
    vectors (per-row cosine). Writes a report consumable via --parity-report;
    a miss (min cosine below ``cos_min``) sets parity_pass=false — the caller
    exits EXIT_PARITY_MISS and the 0.11.0 route is forced."""
    realized = _realized_vllm_version()
    rows = [r for r in _read_jsonl(Path(anchor_texts_jsonl)) if str(r.get("text", "")).strip()]
    if not rows:
        raise RuntimeError(f"[parity-probe] no non-empty rows in {anchor_texts_jsonl}")
    z = np.load(banked_npz, allow_pickle=False)
    banked_emb = z["emb"].astype(np.float64)
    banked_key = {
        (str(c), int(d)): i
        for i, (c, d) in enumerate(zip(z["context_ids"].tolist(), z["draws"].tolist()))
    }
    rows.sort(key=lambda r: (str(r["context_id"]), int(r["draw"])))
    picked = [r for r in rows if (str(r["context_id"]), int(r["draw"])) in banked_key][:n_anchors]
    if len(picked) < n_anchors:
        raise RuntimeError(
            f"[parity-probe] only {len(picked)} anchor rows match the banked npz (need {n_anchors})"
        )
    revision = _resolve_embed_revision()
    llm = _make_embed_llm(revision, max_model_len)
    try:
        res = llm.embed([r["text"] for r in picked], use_tqdm=False)
        arr = np.array([r.outputs.embedding for r in res], dtype=np.float64)
    finally:
        _reap_engine(llm)
    assert arr.shape == (len(picked), EMBED_DIM), arr.shape
    norms = np.linalg.norm(arr, axis=1)
    if (norms == 0.0).any():
        raise RuntimeError("[parity-probe] zero-norm probe embedding")
    unit = arr / norms[:, None]
    per_row = []
    for j, r in enumerate(picked):
        i = banked_key[(str(r["context_id"]), int(r["draw"]))]
        ref = banked_emb[i]
        ref_n = float(np.linalg.norm(ref))
        if ref_n == 0.0:
            raise RuntimeError(f"[parity-probe] zero-norm banked vector at index {i}")
        cos = float(np.dot(unit[j], ref / ref_n))
        per_row.append({"context_id": str(r["context_id"]), "draw": int(r["draw"]), "cos": cos})
    min_cos = min(p["cos"] for p in per_row)
    report = {
        "parity_pass": bool(min_cos >= cos_min),
        "engine": realized,
        "reference_engine": EXPECTED_EMBED_ENGINE,
        "embed_model": EMBED_MODEL,
        "embed_revision": revision,
        "n_anchors": len(picked),
        "cos_min_bar": cos_min,
        "min_cos": min_cos,
        "max_cos_deviation": 1.0 - min_cos,
        "per_row": per_row,
    }
    write_json_atomic(Path(out_path), report)
    print(
        f"[parity-probe] engine={realized} min_cos={min_cos:.6f} bar={cos_min} -> "
        f"{'PASS' if report['parity_pass'] else 'MISS'} (report {out_path})",
        flush=True,
    )
    return report


def phase_embed(cfg: Cfg, bank: dict, model, tok) -> int:
    """Qwen3-Embedding-8B third space over ALL battery answer texts (plan
    §4.4 Embedding paragraph — the binding v3 requirement closing consistency
    WARN-1). ``model``/``tok`` are unused (no q35 load on this phase); the
    embed model + tokenizer load HERE, pinned to a resolved revision.

    Engine parity is STRUCTURAL: the realized vLLM version must equal the
    parity reference (EXPECTED_EMBED_ENGINE — the repo uv.lock pin; the
    DEFAULT route is running this phase under the repo venv) or carry a
    PASSING --parity-report; the realized version is recorded inside every
    chunk npz, the perdraw/means npz, meta.json, and the sentinel, and keys
    the chunk-resume fingerprint, so unit 5's analysis can ASSERT vector
    provenance before comparing against the parent's banked 0.11.0 vectors."""
    del model, tok  # embed loads its own instrument; q35 is never loaded here
    print(f"[phase=embed] start out_root={cfg.out_root} n_axes={len(cfg.axes)}", flush=True)
    engine_version = _realized_vllm_version()
    engine_meta = _assert_engine_parity(engine_version, cfg.parity_report)
    print(f"[embed] engine vLLM=={engine_version} mode={engine_meta['parity_mode']}", flush=True)
    by_cell = group_contexts_by_cell(bank)
    unknown = [a for a in cfg.axes if a not in by_cell]
    if unknown:
        raise RuntimeError(f"unknown axes {unknown}; bank has {sorted(by_cell)}")
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    rows, n_empty = _collect_anchor_rows(cfg, by_cell)
    texts = [r["text"] for r in rows]
    print(
        f"[embed] {len(rows)} rows across {len(cfg.axes)} axes (skipped_empty={n_empty})",
        flush=True,
    )
    revision = _resolve_embed_revision()
    fp = _embed_regime_fp(rows, cfg.embed_chunk, cfg.embed_max_model_len, revision, engine_version)
    sent_fp = _sha16({"embed_fp": fp, "upload": cfg.upload})
    sentinel = cfg.out_root / "battery_embed_done.json"
    s = _read_json(sentinel)
    if s is not None and s.get("regime_fp") == sent_fp:
        logger.info("[embed] sentinel present with matching regime_fp — skipping")
        return 0
    # Token-length precheck with the EMBED model's own tokenizer — raise the
    # flag, never truncate (the #2215 port's contract).
    etok = _load_embed_tokenizer(revision)
    lens = [len(etok.encode(t, add_special_tokens=True)) for t in texts]
    max_len = max(lens)
    print(f"[embed] token lens: max={max_len} mean={sum(lens) / len(lens):.1f}", flush=True)
    if max_len >= cfg.embed_max_model_len:
        raise RuntimeError(
            f"[embed] longest text is {max_len} tokens >= --embed-max-model-len "
            f"{cfg.embed_max_model_len}; raise the flag — inputs are never truncated"
        )
    try:
        emb = _embed_rows(
            texts,
            chunks_dir=cfg.embed_root / "chunks",
            fp=fp,
            chunk=cfg.embed_chunk,
            max_model_len=cfg.embed_max_model_len,
            revision=revision,
            engine_version=engine_version,
            pilot_ceiling_h=cfg.embed_pilot_ceiling_h,
            pilot_report_path=cfg.manifest_dir / "embed_pilot_gate_report.json",
        )
    except EmbedPilotRefuse as e:
        print(f"[phase=embed] pilot refuse: {e}", flush=True)
        return EXIT_PILOT_REFUSE
    norms = np.linalg.norm(emb.astype(np.float64), axis=1)
    zero_idx = np.flatnonzero(norms == 0.0)
    if zero_idx.size:
        raise RuntimeError(f"[embed] zero-norm embeddings at row indices {zero_idx[:10].tolist()}")
    unit = (emb.astype(np.float64) / norms[:, None]).astype(np.float32)
    cids = np.array([r["context_id"] for r in rows])
    draws = np.array([int(r["draw"]) for r in rows], dtype=np.int32)
    cell_arr = np.array([r["cell"] for r in rows])
    leak_arr = np.array([bool(r.get("think_leak", False)) for r in rows])
    # Per-context mean of the L2-NORMALIZED per-draw embeddings (NOT
    # re-normalized; documented in meta — the parent layout, consumed
    # symmetrically by unit 5's cross-model analysis).
    uniq_cids = sorted(set(cids.tolist()))
    cid_index = {c: i for i, c in enumerate(uniq_cids)}
    sums = np.zeros((len(uniq_cids), EMBED_DIM), dtype=np.float64)
    counts = np.zeros(len(uniq_cids), dtype=np.int32)
    idx = np.array([cid_index[c] for c in cids.tolist()])
    np.add.at(sums, idx, unit.astype(np.float64))
    np.add.at(counts, idx, 1)
    means = (sums / counts[:, None]).astype(np.float16)
    emb_dir = cfg.embed_root / "embeddings_qwen3_8b"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        emb_dir / "perdraw_anchors.npz",
        emb=unit.astype(np.float16),
        context_ids=cids,
        draws=draws,
        cells=cell_arr,
        think_leak=leak_arr,
        vllm_version=engine_version,
    )
    np.savez(
        emb_dir / "means_anchors.npz",
        emb_mean=means,
        context_ids=np.array(uniq_cids),
        n_draws=counts,
        vllm_version=engine_version,
    )
    meta = {
        "issue": ISSUE,
        "model": EMBED_MODEL,
        "model_revision": revision,
        "engine": engine_meta,
        "pooling": "model_default_last_token",
        "normalized": "l2_float64_divide_fp16_store",
        "means": "mean of L2-normalized per-draw embeddings, NOT re-normalized",
        "embed_dim": EMBED_DIM,
        "max_model_len": cfg.embed_max_model_len,
        "chunk": cfg.embed_chunk,
        "n_rows": len(rows),
        "n_contexts": len(uniq_cids),
        "n_skipped_empty": n_empty,
        "axes": sorted(cfg.axes),
        "regime_fp": fp,
        "repro": _repro(cfg, "embed"),
    }
    write_json_atomic(emb_dir / "meta.json", meta)
    upload: dict = {"mode": cfg.upload}
    if cfg.upload == "hf":
        dest_prefix = f"{cfg.hf_prefix}/analysis_tensors/embeddings_qwen3_8b"
        res = upload_dir_sharded(
            emb_dir,
            cfg.hf_repo,
            dest_prefix,
            shard_glob="*",
            resume_skip=False,
            delete_local=False,
        )
        upload["embeddings"] = {
            "repo_id": res.repo_id,
            "dest_prefix": dest_prefix,
            "uploaded": len(res.uploaded),
            "rerouted": len(res.rerouted),
            "skipped_existing": len(res.skipped_existing),
        }
    # meta FIRST: it carries its own regime_fp (the chunk fp) — the sentinel's
    # regime_fp must stay the upload-mode-keyed sent_fp (idempotency key).
    write_json_atomic(
        sentinel,
        {**meta, "regime_fp": sent_fp, "embed_fp": fp, "upload": upload},
    )
    print("[phase=embed] sentinel written", flush=True)
    return 0


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
    if args.parity_probe_out:
        # Engine-parity probe MODE (plan §4.4 contingency): no bank, no q35 —
        # embeds banked parent anchor texts under the CURRENT engine.
        if not (args.parity_anchor_texts and args.parity_banked_npz):
            ap.error("--parity-probe-out requires --parity-anchor-texts and --parity-banked-npz")
        report = run_engine_parity_probe(
            Path(args.parity_anchor_texts),
            Path(args.parity_banked_npz),
            Path(args.parity_probe_out),
            n_anchors=args.parity_n_anchors,
            cos_min=args.parity_cos_min,
            max_model_len=args.embed_max_model_len,
        )
        return 0 if report["parity_pass"] else EXIT_PARITY_MISS
    if args.axes and args.num_shards != 1:
        ap.error("--axes is mutually exclusive with --num-shards > 1")
    if not (0 <= args.shard_index < args.num_shards):
        ap.error(
            f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}"
        )
    if args.phase == "embed":
        if args.bank_source != "main":
            # Plan v6 §9: the FFR round runs NO embed phase (no embed model
            # provisioned); refuse rather than embed the wrong grid silently.
            ap.error("--phase embed supports --bank-source main only (FFR round has no embed)")
        if args.num_shards != 1:
            ap.error("--phase embed is single-process (plan §9 P6: embed runs on one GPU)")
        # No q35 load: the embed model is the instrument. Bank STRINGS suffice
        # (string gates run; token gates belong to the q35-tokenizer phases).
        bank = B.build_bank_strings()
        print(f"[bank] {bank['n_contexts']} contexts / {bank['n_pairs']} pairs", flush=True)
        by_cell = group_contexts_by_cell(bank)
        axes = resolve_axes(args, by_cell)
        assert axes, "embed resolved no axes"
        cfg = build_cfg(
            args,
            bank_values_sha=bank["values_sha256"],
            axes=axes,
            model_revision="n/a-embed-phase",
        )
        cfg.out_root.mkdir(parents=True, exist_ok=True)
        return phase_embed(cfg, bank, None, None)
    model_revision = _resolve_model_revision()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if args.phase == "capture":
        # Plan §4.4: fp32 capture (fp16/bf16 stores overflow Qwen3-era massive
        # activations, #2330); bfloat16 is an explicit debug opt-down.
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.capture_dtype]
    model, tok = cm2587.load_q35_model_and_tokenizer(
        device=device, revision=model_revision, dtype=dtype
    )
    if args.bank_source == "ffr":
        # Plan v6 §4.2 wiring seam: manifest-sourced FFR bank; every phase
        # function below consumes only the bank dict + Cfg (reused untouched).
        bank = BF.build_ffr_bank(tok)  # sha-pinned manifest + string/token gates
    else:
        bank = B.build_bank(tok)  # P0a string gates + P0b token gates, fail-loud
    print(
        f"[bank] source={args.bank_source} {bank['n_contexts']} contexts / "
        f"{bank['n_pairs']} pairs values_sha={bank['values_sha256'][:12]}",
        flush=True,
    )
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
