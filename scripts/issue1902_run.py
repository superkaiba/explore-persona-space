#!/usr/bin/env python3
"""Issue #1902 pod-side phase driver — P1 pilot / P2 generation / P3 capture.

One driver, subprocess-per-phase (plan #1902 v4 §4): the committed dispatcher
``scripts/issue1902_dispatch.sh`` fans each GPU phase out 4-way by checkpoint
(one subprocess per checkpoint, ``CUDA_VISIBLE_DEVICES`` pinned in the LAUNCHER
env per leg), then runs a single ``--finalize`` leg per phase:

- ``--phase pilot``  (P1): 500-context pilot per checkpoint — native-render
  generation + BOTH-serialization diagonal capture on 3 probe layers +
  per-source unflagged-survival counts; ``--finalize`` writes
  ``pilot_report.json`` (BINDING revision pins, Gate A, bf16 batched-vs-batch-1
  two-bar gate, fp16-vs-fp32 store ΔR² (A12), pilot flip-rule inputs, timed MLP
  unit + timed store-shard serialize+upload leg, measured per-row capture wall
  + the capture-cost abort). Designed halts exit ``rc=7`` with a gate-report
  JSON (the #1415 convention) — never a bare rc=1.
- ``--phase gen``    (P2): vLLM on-policy generation per checkpoint (native
  render; base = plain render + stop sequences), n=1 seed 42 + n=2 reliability
  draws (seeds 43/44) on the pinned 1k single-turn subset; symmetric degeneracy
  flags; rollout text persisted + uploaded (non-LFS) BEFORE capture;
  ``--finalize`` builds ``intersection_manifest.json`` + Gate A' (floor 5,000;
  min-over-folds n_tr > d).
- ``--phase capture`` (P3): batched teacher-forced HF forwards per (activation
  checkpoint m, answer source s, corpus) — canonical PLAIN render for the
  primary grid, native render on the 2k robustness subset (diagonal),
  reliability captures (diagonal, 1k subset); token-id concatenation at the
  prompt/answer seam + offset-mapping span boundaries (straddler policy:
  exclude at prefix, include at context; ``seam_flags`` recorded); fp16 store
  in the dedup'd layout (``issue1902_common`` path helpers) + ``row_index``
  manifests; per-checkpoint incremental ``upload_dir_sharded`` → verify →
  (conditionally) delete-local.
- ``--phase stage``: stage the P0 corpora from HF via ``hub.stage_hub_prefix``
  (local-first; the dispatcher runs this before any GPU phase).
- ``--phase fits``   (P4): registration point ONLY — lands in unit C as
  ``scripts/issue1902_fits.py``; invoking it here exits 2 with a pointer.

Contracts baked in: every model/tokenizer load passes
``revision=resolve_revision(...)`` (pins are BINDING once P1 writes them);
``VLLM_WORKER_MULTIPROC_METHOD=spawn`` is set at module top BEFORE any vllm
import (#628); phase entrypoints end ``sys.exit(0)`` (PyGILState atexit rule);
per-unit persistence + resume predicates keyed on every output-affecting
regime flag; ``[phase=...]`` breadcrumbs + ``unit k/N`` progress lines; each
phase's ``--finalize`` writes a poller sentinel
(``issue-1902-<phase>-done-<epoch>.json``, ``blocks_pipeline: false``).

Content hygiene: LMSYS is unscreened real user text — no corpus/rollout row
text is ever printed or logged; logs and reports carry ids, indices, counts,
and token statistics only.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
# vLLM v1 EngineCore fork poisoning (#628): must be set BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Long teacher-forced multi-layer captures fragment the CUDA allocator (#761);
# set BEFORE torch import (plan §4 P3).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1902_common as C  # noqa: E402
from explore_persona_space.eval.vllm_util import (  # noqa: E402
    GPU_FREE_MARGIN_GIB,
    SHARED_NODE_UTIL_CAP,
    VLLM_UTIL_FLOOR,
    vllm_util_for_free,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("issue1902_run")

ISSUE = 1902
# Bumping this invalidates every per-unit resume record (regime key member).
RECIPE_VERSION = "issue1902-run-v1"
GATE_RC = 7  # designed-halt exit code (plan §7; the #1415 convention)

# Pilot (P1) — plan §4/§9.
PILOT_N = 500  # first N rows of the pinned draw, per corpus
PILOT_GEN_BATCH = 500
PROBE_LAYER_FRACS = (0.25, 0.5, 0.75)  # 3 probe layers, depth-relative (A6)
BF16_GATE_ROWS = 8
BF16_EARLY_LAYERS = (0, 1, 2, 3)
BF16_EARLY_COS_MIN = 0.999  # two-bar recipe (#779 r12; plan §4 P3)
BF16_FLAT_COS_MIN = 0.995
FP16_DELTA_R2_MAX = 0.002  # A12
PILOT_FLIP_RATIO = 0.6  # plain Q < 0.6 x native Q at EVERY probe layer -> flip
PILOT_FOLDS = 4  # pilot-local CV folds (48-unit cap on the serial fit loop)
CAPTURE_PLANNED_WALL_H = 4.0  # §9 P3 planned_wall_h (capture-cost abort basis)
CAPTURE_COST_ABORT_RATIO = 2.0

# Generation (P2) — verbatim #779 protocol via issue1902_common constants.
VLLM_CHUNK = int(os.environ.get("EPM_VLLM_CHUNK_SIZE", "500"))

# Capture (P3).
CAPTURE_TOKEN_BUDGET = int(os.environ.get("EPM_CAPTURE_TOKEN_BUDGET", "65536"))
CAPTURE_BATCH_MAX = int(os.environ.get("EPM_CAPTURE_BATCH_MAX", "32"))
# Free-HBM floor at capture model load (shared-node guard; #1902 crash 1):
# 7B bf16 weights ~15 GiB + the 17-layer captured-hidden-state stack under
# CAPTURE_TOKEN_BUDGET (65,536 padded tokens x 4096 d x 2 B ~= 0.5 GiB/layer
# -> ~8.5 GiB) + fp32 pooling transients + the shared-node margin. Reads FREE
# memory (mem_get_info), never total — another tenant's allocation fails loud
# HERE, not as a cryptic CUDA OOM mid-forward.
CAPTURE_FREE_FLOOR_GIB = float(os.environ.get("EPM_CAPTURE_FREE_FLOOR_GIB", "30"))
ROBUST_NATIVE_N = 2_000  # native-render robustness subset (single, diagonal)
# Delete local store after verified upload only when free space is below this
# (RunPod ~130 GB quota lane; GCE 250 GB keeps the store local for P4).
DELETE_LOCAL_FREE_GB = float(os.environ.get("EPM_DELETE_LOCAL_FREE_GB", "150"))

# Text-payload sharding: keep every uploaded text file on the non-LFS path.
TEXT_SHARD_MAX_BYTES = 9_000_000

# Smoke slice (dispatcher --smoke; plan §4 hard-requirement escapes).
SMOKE_SINGLE_N = 32
SMOKE_MULTI_N = 16
# Smoke generation cap (scale dial only): random/tiny answers survive the
# truncation flag (n < GEN_MAX_TOKENS) so any nonzero yield proceeds.
SMOKE_GEN_MAX_TOKENS = 192

# Per-phase disk floors (GB) — §9 disk rows, pending-scaled at phase entry.
HEADROOM_BASE_GB = {"pilot": 6.0, "gen": 4.0, "capture": 4.0}
CAPTURE_PER_CELL_GB = 2.5  # 17 layers x 16k rows x 4096 x fp16 ~= 2.2 GB

PLAIN_USER_LABEL = "User: "


# ── small io / provenance helpers ────────────────────────────────────────────


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_sha() -> str:
    """Repo sha for reproducibility metadata; degrades on git-less lanes.

    The fellows/SLURM rsync copy has NO git checkout (the seam
    fits._commit_eval_results already tolerates — #1902 job 16142 crashed
    here rc=128 writing the pilot leg report). EPS_GIT_SHA env wins when
    set; the canonical sha also rides the launch marker + handle sidecar.
    """
    env_sha = os.environ.get("EPS_GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    if out.returncode == 0:
        return out.stdout.strip()
    return "unavailable-no-git-checkout"


def _metadata() -> dict[str, Any]:
    """Reproducibility metadata for every result JSON (git sha, versions, ts)."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "vllm"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 — optional at metadata time (CPU legs)
            versions[mod] = "unavailable"
    return {
        "issue": ISSUE,
        "recipe_version": RECIPE_VERSION,
        "git_sha": _git_sha(),
        "timestamp_utc": _now_iso(),
        "env_versions": versions,
    }


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict]:
    """Text-mode line iteration — never ``.splitlines()`` (#825/#950 U+2028)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip("\n")
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _sentinel_dir(out_root: Path) -> Path:
    env = os.environ.get("EPM_SENTINEL_DIR")
    if env:
        return Path(env)
    ws = Path("/workspace/logs")
    try:
        ws.mkdir(parents=True, exist_ok=True)
        return ws
    except OSError:
        return out_root / "logs"


def write_phase_sentinel(out_root: Path, phase: str, note: dict[str, Any], *, smoke: bool) -> Path:
    """Benign phase-done sentinel (poll_pipeline drain contract; unique name,
    written ONCE — the namespace is write-once + VM-drained, #1311)."""
    sdir = _sentinel_dir(out_root)
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / f"issue-{ISSUE}-{phase}-done-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:progress",
        "version": 1,
        "task_id": ISSUE,
        "by": "issue1902_run",
        "ts": _now_iso(),
        "blocks_pipeline": False,
        "smoke": bool(smoke),
        "note": json.dumps({"phase": phase, **note}, ensure_ascii=False),
    }
    _write_json_atomic(path, payload)
    logger.info("[sentinel] wrote %s", path)
    return path


def designed_halt(out_root: Path, gate: str, payload: dict[str, Any]) -> None:
    """Plan-registered gate refusal: report JSON + distinct rc (never bare rc=1)."""
    report = {"gate": gate, "verdict": "HALT", "ts": _now_iso(), **payload}
    path = out_root / "gate_reports" / f"{gate}_{int(time.time())}.json"
    _write_json_atomic(path, report)
    logger.error("[gate:%s] DESIGNED HALT (rc=%d): %s -> %s", gate, GATE_RC, payload, path)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(GATE_RC)


# ── corpus loading / subsets / folds ─────────────────────────────────────────


def corpus_path(corpus_dir: Path, corpus: str) -> Path:
    name = C.CORPUS_SINGLE_FILENAME if corpus == C.CORPUS_SINGLE else C.CORPUS_MULTI_FILENAME
    return corpus_dir / name


def load_corpus(
    corpus_dir: Path, corpus: str, *, smoke: bool, limit: int | None = None
) -> list[dict]:
    """Load a P0 corpus (deterministic file order = the pinned draw order)."""
    path = corpus_path(corpus_dir, corpus)
    if not path.exists():
        raise FileNotFoundError(
            f"corpus file missing: {path} — run `--phase stage` (or issue1902_corpus.py) first"
        )
    rows = _read_jsonl(path)
    n = len(rows)
    if smoke:
        rows = rows[: SMOKE_SINGLE_N if corpus == C.CORPUS_SINGLE else SMOKE_MULTI_N]
    if limit is not None:
        rows = rows[:limit]
    logger.info("[corpus] %s: %d rows loaded (of %d on disk)", corpus, len(rows), n)
    return rows


def reliability_ids(single_rows: list[dict]) -> list[str]:
    """Pinned reliability subset: FIRST N single-turn rows by corpus order."""
    return [r["id"] for r in single_rows[: C.RELIABILITY_SUBSET_N]]


def assign_fold_groups(
    groups: list[str], n_folds: int = C.N_FOLDS, seed: int = C.FOLD_SEED
) -> dict[str, int]:
    """Deterministic group->fold assignment (plan §4 P4): seeded shuffle of the
    unique group labels, then greedy balance by group size (largest first into
    the currently-smallest fold). Marked strata are whole-stratum groups."""
    import random as _random

    sizes: dict[str, int] = {}
    for g in groups:
        sizes[g] = sizes.get(g, 0) + 1
    labels = sorted(sizes)
    _random.Random(seed).shuffle(labels)
    labels.sort(key=lambda g: -sizes[g])  # stable: keeps the seeded order for ties
    fold_tot = [0] * n_folds
    assign: dict[str, int] = {}
    for g in labels:
        f = min(range(n_folds), key=lambda i: fold_tot[i])
        assign[g] = f
        fold_tot[f] += sizes[g]
    return assign


# ── revision pins (BINDING once P1 writes them; plan §10) ────────────────────


def pins_path(out_root: Path) -> Path:
    return out_root / "revision_pins.json"


def ensure_pins(out_root: Path) -> dict[str, str]:
    """Resolve + persist the per-checkpoint revision pins ONCE (pilot --init)."""
    path = pins_path(out_root)
    if path.exists():
        pins = C.revision_pins_from_report({C.REVISION_PINS_KEY: _read_json(path)})
        logger.info("[pins] existing pins kept: %s", {k: v[:10] for k, v in pins.items()})
        return pins
    pins = C.pin_revisions_now()
    _write_json_atomic(path, pins)
    logger.info("[pins] resolved + persisted: %s", {k: v[:10] for k, v in pins.items()})
    return pins


def load_pins(out_root: Path) -> dict[str, str]:
    path = pins_path(out_root)
    if not path.exists():
        raise RuntimeError(
            f"revision pins missing at {path} — run `--phase pilot --init` first "
            "(pins are BINDING for every model/tokenizer load; plan §10)"
        )
    return C.revision_pins_from_report({C.REVISION_PINS_KEY: _read_json(path)})


# ── per-unit resume state (kept OUTSIDE the drained sentinel namespace) ──────


def _state_dir(out_root: Path) -> Path:
    return out_root / "state"


def unit_regime(args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    """Every output-affecting regime key (a mismatch REFUSES the resume, #1333)."""
    return {
        "recipe_version": RECIPE_VERSION,
        "smoke": bool(args.smoke),
        **extra,
    }


def unit_done(out_root: Path, unit: str, regime: dict[str, Any]) -> bool:
    path = _state_dir(out_root) / f"{unit}.done.json"
    if not path.exists():
        return False
    prior = _read_json(path)
    if prior.get("regime") != regime:
        raise RuntimeError(
            f"resume REFUSED for unit {unit}: out-root holds a run under a DIFFERENT "
            f"regime.\n  prior: {prior.get('regime')}\n  now:   {regime}\n"
            "Use a fresh --out-root (per-leg out-roots; crash-fix-rounds rule) or wipe "
            f"{path} deliberately."
        )
    return True


def mark_unit_done(out_root: Path, unit: str, regime: dict[str, Any], info: dict[str, Any]) -> None:
    _write_json_atomic(
        _state_dir(out_root) / f"{unit}.done.json",
        {"unit": unit, "regime": regime, "ts": _now_iso(), **info},
    )


def capture_unit_store_dirs(
    store: Path, ckpt: str, u: dict[str, Any], layers: list[int]
) -> list[Path]:
    """Store leaf dirs one capture unit writes (grid: answer cell + shared ctx
    cell; subdir units: the cell + its own ctx sub-leaf)."""
    if u["subdir"] is None:
        return [
            (store / C.answer_store_relpath(ckpt, u["src"], u["corpus"], layers[0])).parent,
            (store / C.ctx_store_relpath(ckpt, u["corpus"], layers[0])).parent,
        ]
    return [store / u["subdir"], store / u["subdir"] / "ctx"]


def capture_unit_artifacts_present(
    store: Path, ckpt: str, u: dict[str, Any], layers: list[int]
) -> bool:
    """Local store artifacts for one capture unit (row_index + every layer).

    The capture resume predicate is sentinel AND (artifacts present OR the
    leg's VERIFIED upload record): a done-sentinel alone must never
    fast-forward past deleted-but-never-uploaded artifacts (#1315 class;
    concern p3-delete-local-starves-p4-store — post-delete-local the record
    licenses the skip and P4 re-stages from HF)."""
    return all(
        (d / "row_index.jsonl").exists() and all((d / f"L{layer}.pt").exists() for layer in layers)
        for d in capture_unit_store_dirs(store, ckpt, u, layers)
    )


def headroom_gate(out_root: Path, phase: str, pending_units: int, per_unit_gb: float) -> None:
    """Resume-aware out-root headroom assert (plan-compute-sizing mount rule)."""
    if pending_units <= 0:
        logger.info("[disk-headroom] %s: zero pending units — gate skipped", phase)
        return
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    need = HEADROOM_BASE_GB.get(phase, 4.0) + per_unit_gb * pending_units
    assert_out_root_headroom(out_root, need, phase=f"issue1902-{phase}")


# ── renders + spans ──────────────────────────────────────────────────────────


def plain_prompt_and_qspan(query: str, prefix_turns: list[dict] | None) -> tuple[str, int, int]:
    """Plain generation prompt + the query's EXACT char span, BY CONSTRUCTION
    (no ``find()`` — the #1776 short-query mis-anchor class is structurally
    impossible: the render is assembled here, so the final-user-turn tail
    ``User: {q}\\nAssistant:`` anchors the span exactly)."""
    text = C.render_plain_prompt(query, prefix_turns)
    tail = f"{PLAIN_USER_LABEL}{query}\nAssistant:"
    assert text.endswith(tail), "render_plain_prompt drifted from the expected tail"
    q_start = len(text) - len(tail) + len(PLAIN_USER_LABEL)
    return text, q_start, q_start + len(query)


def token_boundary(
    offsets: list[tuple[int, int]],
    char_end: int,
    *,
    include_straddler: bool,
) -> tuple[int, bool]:
    """Char boundary -> token index under the documented straddler policy
    (plan §4 P3; mirrors ``representation_shift.compute_prompt_spans``'s
    ``_boundary`` with ``on_seam='snap'``: EXCLUDE a prefix-boundary straddler,
    INCLUDE a context-boundary straddler). Pure function — unit-testable
    without a tokenizer."""
    n_inside = sum(1 for _s, e in offsets if e <= char_end)
    straddler = n_inside < len(offsets) and offsets[n_inside][0] < char_end
    return n_inside + (1 if (straddler and include_straddler) else 0), straddler


def plain_spans(
    tokenizer,
    text: str,
    q_start: int,
    q_end: int,
) -> tuple[list[int], int, int, dict[str, bool]]:
    """(prompt_ids, prefix_len, context_len, seam_flags) for the PLAIN render.

    ``compute_prompt_spans`` is chat-template-locked (it renders messages via
    ``apply_chat_template``), so the canonical plain-render grid uses this
    same-recipe sibling: ONE tokenization of the full render with offset
    mapping (the ids ARE the returned prompt ids — token identity by
    construction), char boundaries mapped with the identical straddler policy
    + ``seam_flags`` provenance (#1092/#1315 BPE-seam rules)."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    assert all(s < e for s, e in offsets), "zero-width token offsets — unsupported tokenizer"
    prefix_len, seam_p = token_boundary(offsets, q_start, include_straddler=False)
    context_len, seam_c = token_boundary(offsets, q_end, include_straddler=True)
    assert 0 < prefix_len < context_len <= len(ids), (prefix_len, context_len, len(ids))
    return list(ids), prefix_len, context_len, {"prefix": seam_p, "context": seam_c}


# ── uploads (non-LFS text payloads + eval-mirror JSONs) ──────────────────────


def _hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def upload_json_small(local: Path, repo_path: str) -> None:
    """Single small JSON -> data repo (non-LFS path; fail-loud, retried)."""
    from explore_persona_space.orchestrate import hub

    api = _hf_api()
    hub.retry_transient(
        lambda: api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=repo_path,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue1902: {repo_path}",
        ),
        what=f"upload_file {repo_path}",
    )
    logger.info("[upload] %s -> hf:%s/%s", local.name, C.HF_DATA_REPO, repo_path)


def upload_text_payload(local_jsonl: Path, repo_dir: str) -> list[str]:
    """Upload a rollout JSONL keeping every file on the non-LFS path.

    <=9 MB uploads as-is; bigger files line-split into ``<stem>.shardNN.jsonl``
    (<9 MB each) + ``<stem>.manifest.json`` (upload-policy big-text recipe).
    ONE bulk ``upload_folder`` commit per payload (never a per-file loop).
    Returns the uploaded repo paths (for the exact-set verify)."""
    import hashlib
    import tempfile

    from explore_persona_space.orchestrate import hub

    api = _hf_api()
    size = local_jsonl.stat().st_size
    with tempfile.TemporaryDirectory(dir=local_jsonl.parent, prefix=".hfstage-") as td:
        stage = Path(td)
        if size <= TEXT_SHARD_MAX_BYTES:
            (stage / local_jsonl.name).write_bytes(local_jsonl.read_bytes())
        else:
            stem = local_jsonl.stem
            shard_idx, shard_lines, shard_bytes = 0, [], 0
            manifest: list[dict] = []

            def _flush() -> None:
                nonlocal shard_idx, shard_lines, shard_bytes
                if not shard_lines:
                    return
                body = "".join(shard_lines)
                name = f"{stem}.shard{shard_idx:02d}.jsonl"
                (stage / name).write_text(body, encoding="utf-8")
                manifest.append(
                    {
                        "name": name,
                        "n_lines": len(shard_lines),
                        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    }
                )
                shard_idx += 1
                shard_lines, shard_bytes = [], 0

            with open(local_jsonl, encoding="utf-8") as f:
                for line in f:
                    b = len(line.encode("utf-8"))
                    if shard_bytes + b > TEXT_SHARD_MAX_BYTES:
                        _flush()
                    shard_lines.append(line)
                    shard_bytes += b
            _flush()
            _write_json_atomic(
                stage / f"{stem}.manifest.json",
                {"source": local_jsonl.name, "n_shards": shard_idx, "shards": manifest},
            )
        names = sorted(p.name for p in stage.iterdir())
        hub.assert_hub_dir_filecounts(stage, repo_dir)  # 10k/dir commit cap guard
        hub.retry_transient(
            lambda: api.upload_folder(
                folder_path=str(stage),
                repo_id=C.HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=repo_dir,
                commit_message=f"issue1902 rollout text: {repo_dir}/{local_jsonl.stem}",
            ),
            what=f"upload_folder {repo_dir}",
        )
    logger.info(
        "[upload] %s -> hf:%s/%s (%d files)", local_jsonl.name, C.HF_DATA_REPO, repo_dir, len(names)
    )
    return [f"{repo_dir}/{n}" for n in names]


# ── generation (P2 + the pilot's generation leg) ─────────────────────────────


def _tokenizer(model_id: str, revision: str | None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, revision=revision)


class _HfFallbackEngine:
    """HF `generate()` engine with a vLLM-shaped ``generate(prompts, sp)``
    surface — the SMOKE-only engine substitution (``EPM_ISSUE1902_GEN_ENGINE=hf``)
    for hosts where vLLM cannot run the tiny smoke model (CPU VM). The phase
    BODY (prompt render, chunking, flags, persistence) is byte-identical; only
    the engine call substitutes — a NAMED smoke deviation, never a production
    path (production keeps vLLM; the dispatcher never sets the env)."""

    def __init__(self, model_id: str, revision: str | None, max_model_len: int):
        self.max_model_len = max_model_len
        self.tokenizer = _tokenizer(model_id, revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = _load_hf_model(model_id, revision, device)
        self.device = device

    def generate(self, prompts: list[str], sp, use_tqdm: bool = False):
        """Batched sampling generate (batch 8, left-pad); returns objects
        shaped like vLLM RequestOutput (``.outputs[0].{text,token_ids,
        finish_reason}``). Stop sequences applied post-hoc on decoded text."""
        import torch

        del use_tqdm
        tok = self.tokenizer
        tok.padding_side = "left"
        results = []
        gen = torch.Generator(device="cpu").manual_seed(int(getattr(sp, "seed", 42) or 42))
        for i in range(0, len(prompts), 8):
            chunk = prompts[i : i + 8]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            torch.manual_seed(int(gen.initial_seed()) + i)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    do_sample=True,
                    temperature=float(sp.temperature),
                    top_p=float(sp.top_p),
                    max_new_tokens=int(sp.max_tokens),
                    pad_token_id=tok.pad_token_id,
                )
            n_in = enc["input_ids"].shape[1]
            for row in out:
                new_ids = row[n_in:].tolist()
                if tok.eos_token_id in new_ids:
                    new_ids = new_ids[: new_ids.index(tok.eos_token_id)]
                    finish = "stop"
                else:
                    finish = "length" if len(new_ids) >= int(sp.max_tokens) else "stop"
                text = tok.decode(new_ids, skip_special_tokens=True)
                for s in getattr(sp, "stop", None) or []:
                    if s in text:
                        text = text.split(s, 1)[0]
                        new_ids = tok.encode(text, add_special_tokens=False)
                        finish = "stop"
                o = type("Out", (), {})()
                o.text, o.token_ids, o.finish_reason = text, new_ids, finish
                res = type("Res", (), {})()
                res.outputs = [o]
                results.append(res)
        return results


def _gen_gpu_mem_util() -> float:
    """LIVE per-device vLLM memory fraction (shared-node safe; #1902 crash 1).

    ``gpu_memory_utilization`` is a fraction of TOTAL device memory, so the
    fixed 0.60 default demands ``0.6 x total`` bytes regardless of what other
    tenants hold — on a fellows shared H200 (all 8 devices carrying
    ~57-59 GiB of other tenants' memory) EngineCore refused at init
    (``Free memory on device (81.2/139.8 GiB) ... less than desired GPU
    memory utilization (0.6, 83.88 GiB)``). Compute from
    ``torch.cuda.mem_get_info()`` on the leg's CVD-pinned device instead
    (``explore_persona_space.eval.vllm_util.vllm_util_for_free``:
    min(cap, (free − margin)/total), fail-loud below the floor). An
    explicit ``VLLM_GPU_MEM_UTIL`` env stays the operator override. Safe
    pre-``LLM()``: module top pins ``VLLM_WORKER_MULTIPROC_METHOD=spawn``,
    so parent-side cuInit cannot fork-poison the EngineCore (#628).
    """
    env_util = os.environ.get("VLLM_GPU_MEM_UTIL")
    if env_util:
        logger.info(
            "[gen] gpu_memory_utilization=%s (VLLM_GPU_MEM_UTIL operator override)", env_util
        )
        return float(env_util)
    import torch

    free_b, total_b = torch.cuda.mem_get_info(0)
    util = vllm_util_for_free(free_b, total_b)
    logger.info(
        "[gen] gpu_memory_utilization=%.3f free=%.1fGiB total=%.1fGiB "
        "(cap=%.2f margin=%.1fGiB floor=%.2f)",
        util,
        free_b / 2**30,
        total_b / 2**30,
        SHARED_NODE_UTIL_CAP,
        GPU_FREE_MARGIN_GIB,
        VLLM_UTIL_FLOOR,
    )
    return util


def _vllm_engine(model_id: str, revision: str | None, max_model_len: int):
    """One engine config for ALL cells (plan §4 P2): #1324 knobs ON
    (enforce_eager + prefix caching off) via ``hang_mitigations=True``;
    ``gpu_memory_utilization`` computed per device from LIVE free memory
    (:func:`_gen_gpu_mem_util` — shared fellows nodes, #1902 crash 1).
    ``EPM_ISSUE1902_GEN_ENGINE=hf`` substitutes the smoke-only HF engine
    (recorded engine deviation; see :class:`_HfFallbackEngine`)."""
    if os.environ.get("EPM_ISSUE1902_GEN_ENGINE") == "hf":
        logger.warning("[gen] EPM_ISSUE1902_GEN_ENGINE=hf — SMOKE HF engine substitution")
        return _HfFallbackEngine(model_id, revision, max_model_len)
    from explore_persona_space.eval.generation import create_vllm_engine

    return create_vllm_engine(
        model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=_gen_gpu_mem_util(),
        seed=C.GEN_SEED,
        hang_mitigations=True,
        revision=revision,
    )


def _generate_chunked(llm, prompts: list[str], sp) -> list[dict]:
    """Chunked ``generate()`` (<=VLLM_CHUNK prompts/call, per-chunk INFO lines,
    ``use_tqdm=False``) — the #664 deadlock-prevention + #613 tqdm rules."""
    out: list[dict] = []
    n_chunks = (len(prompts) + VLLM_CHUNK - 1) // VLLM_CHUNK
    t0 = time.time()
    for i in range(0, len(prompts), VLLM_CHUNK):
        chunk = prompts[i : i + VLLM_CHUNK]
        logger.info(
            "[vllm-chunk] chunk %d/%d (%d prompts, %.0fs elapsed)",
            i // VLLM_CHUNK + 1,
            n_chunks,
            len(chunk),
            time.time() - t0,
        )
        for res in llm.generate(chunk, sp, use_tqdm=False):
            o = res.outputs[0]
            out.append(
                {
                    "text": o.text,
                    "n_tokens": len(o.token_ids),
                    "finish_reason": str(o.finish_reason),
                }
            )
    return out


def _gen_prompts(rows: list[dict], ckpt: str, tokenizer) -> list[str]:
    """Native per-checkpoint generation render (plan §4 P2): S/D/R chat
    template; B plain render (stop sequences applied at SamplingParams)."""
    prompts: list[str] = []
    for r in rows:
        prefix = r.get("prefix_turns")
        if ckpt == "B":
            prompts.append(C.render_plain_prompt(r["query"], prefix))
        else:
            prompts.append(C.render_chat_prompt(tokenizer, r["query"], prefix))
    return prompts


def _sampling_params(seed: int, ckpt: str, smoke: bool = False):
    """#779-verbatim sampling params; ``smoke`` caps max_tokens ONLY (a
    compute-SCALE dial — production geometry untouched; degeneracy flags are
    informational under smoke)."""
    max_tokens = C.GEN_MAX_TOKENS if not smoke else min(C.GEN_MAX_TOKENS, SMOKE_GEN_MAX_TOKENS)
    stop = list(C.PLAIN_STOP_SEQUENCES) if ckpt == "B" else None
    if os.environ.get("EPM_ISSUE1902_GEN_ENGINE") == "hf":
        # smoke HF engine: a vllm import buys nothing on a CPU host.
        from types import SimpleNamespace

        return SimpleNamespace(
            n=1,
            temperature=C.GEN_TEMPERATURE,
            top_p=C.GEN_TOP_P,
            max_tokens=max_tokens,
            seed=seed,
            stop=stop,
        )
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        temperature=C.GEN_TEMPERATURE,
        top_p=C.GEN_TOP_P,
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
    )


def _flag_records(
    rows: list[dict], gens: list[dict], seed: int, gen_cap: int = C.GEN_MAX_TOKENS
) -> list[dict]:
    """Per-row rollout records with the symmetric degeneracy flags (plan §4 P2).

    ``gen_cap`` is the cap the GENERATION actually ran with: at the production
    cap the two truncation conditions coincide (byte-identical behavior); under
    the REDUCED smoke cap a ``finish_reason == "length"`` is a smoke-scale
    artifact, not evidence of production truncation (flags are informational
    under smoke — gate-calibration rule)."""
    recs: list[dict] = []
    for r, g in zip(rows, gens, strict=True):
        truncated = C.is_truncated(g["n_tokens"]) or (
            g["finish_reason"] == "length" and gen_cap >= C.GEN_MAX_TOKENS
        )
        recs.append(
            {
                "id": r["id"],
                "seed": seed,
                "text": g["text"],
                "n_tokens": g["n_tokens"],
                "finish_reason": g["finish_reason"],
                "truncated": bool(truncated),
                "repetition_flag": bool(C.has_repetition_loop(g["text"])),
            }
        )
    return recs


def _gen_rollout_path(out_root: Path, corpus: str, ckpt: str, seed: int) -> Path:
    name = f"{ckpt}.jsonl" if seed == C.GEN_SEED else f"{ckpt}_rel_seed{seed}.jsonl"
    return out_root / "gen" / corpus / name


def gen_unit_name(corpus: str, ckpt: str, seed: int) -> str:
    return f"gen_{corpus}_{ckpt}_seed{seed}"


def phase_gen_ckpt(args: argparse.Namespace, out_root: Path) -> None:
    """P2 shard leg: on-policy generation for ONE checkpoint (both corpora +
    the pinned reliability draws), rollout text persisted + uploaded BEFORE
    any capture (persist-before-reduce, #779)."""
    ckpt = args.ckpt
    pins = load_pins(out_root)
    revision = C.resolve_revision(ckpt, pins)
    model_id = C.MODEL_IDS[ckpt]
    corpus_dir = Path(args.corpus_dir)

    single = load_corpus(corpus_dir, C.CORPUS_SINGLE, smoke=args.smoke)
    multi = load_corpus(corpus_dir, C.CORPUS_MULTI, smoke=args.smoke)
    rel_ids = set(reliability_ids(single))
    rel_rows = [r for r in single if r["id"] in rel_ids]

    units: list[tuple[str, list[dict], int]] = [
        (C.CORPUS_SINGLE, single, C.GEN_SEED),
        (C.CORPUS_MULTI, multi, C.GEN_SEED),
    ]
    units += [(C.CORPUS_SINGLE, rel_rows, s) for s in C.RELIABILITY_SEEDS]

    regimes = {
        gen_unit_name(corpus, ckpt, seed): unit_regime(
            args,
            phase="gen",
            ckpt=ckpt,
            corpus=corpus,
            seed=seed,
            n_rows=len(rows),
            sampling=f"n=1 T={C.GEN_TEMPERATURE} top_p={C.GEN_TOP_P} max={C.GEN_MAX_TOKENS}",
        )
        for corpus, rows, seed in units
    }
    pending = [
        u
        for u in units
        if not unit_done(
            out_root, gen_unit_name(u[0], ckpt, u[2]), regimes[gen_unit_name(u[0], ckpt, u[2])]
        )
    ]
    headroom_gate(out_root, "gen", len(pending), 0.1)
    if not pending:
        logger.info("[gen] ckpt=%s: all %d units done — leg skipped", ckpt, len(units))
        return

    print(f"[phase=gen] ckpt={ckpt} units={len(pending)}/{len(units)}", flush=True)
    tokenizer = _tokenizer(model_id, revision)
    dims = C.model_dims(model_id, revision)
    llm = _vllm_engine(model_id, revision, dims.max_position_embeddings)
    t_leg = time.time()
    for k, (corpus, rows, seed) in enumerate(pending, start=1):
        unit = gen_unit_name(corpus, ckpt, seed)
        t0 = time.time()
        prompts = _gen_prompts(rows, ckpt, tokenizer)
        sp = _sampling_params(seed, ckpt, smoke=args.smoke)
        gens = _generate_chunked(llm, prompts, sp)
        recs = _flag_records(rows, gens, seed, gen_cap=int(sp.max_tokens))
        local = _gen_rollout_path(out_root, corpus, ckpt, seed)
        _write_jsonl_atomic(local, recs)
        # Persist-before-reduce: rollout TEXT to HF (non-LFS) the moment the
        # unit completes, BEFORE capture (plan §4 P2 / Upload Policy).
        uploaded = upload_text_payload(local, f"{C.RAW_GEN_HF_PATH}/{corpus}")
        n_flagged = sum(1 for r in recs if r["truncated"] or r["repetition_flag"])
        mark_unit_done(
            out_root,
            unit,
            regimes[unit],
            {"n_rows": len(recs), "n_flagged": n_flagged, "uploaded": uploaded},
        )
        print(
            f"[gen] unit {k}/{len(pending)} {unit} rows={len(recs)} "
            f"flagged={n_flagged} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    logger.info("[gen] ckpt=%s leg done in %.0fs", ckpt, time.time() - t_leg)


def _unflagged_ids(out_root: Path, corpus: str, ckpt: str) -> tuple[set[str], dict[str, Any]]:
    recs = _read_jsonl(_gen_rollout_path(out_root, corpus, ckpt, C.GEN_SEED))
    flagged = {"truncated": 0, "repetition": 0}
    ok: set[str] = set()
    lengths: list[int] = []
    for r in recs:
        lengths.append(int(r["n_tokens"]))
        if r["truncated"]:
            flagged["truncated"] += 1
        elif r["repetition_flag"]:
            flagged["repetition"] += 1
        else:
            ok.add(r["id"])
    lengths.sort()

    def _q(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))] if lengths else 0

    stats = {
        "n": len(recs),
        "n_unflagged": len(ok),
        "flagged": flagged,
        "answer_tokens": {
            "p10": _q(0.10),
            "p50": _q(0.50),
            "p90": _q(0.90),
            "max": lengths[-1] if lengths else 0,
        },
    }
    return ok, stats


def phase_gen_finalize(args: argparse.Namespace, out_root: Path, ckpts: list[str]) -> None:
    """P2 finalize: four-source intersection manifest + Gate A' (floor 5,000;
    min-over-folds n_tr > d) + fold assignment (plan §4 P2/P4, §7)."""
    corpus_dir = Path(args.corpus_dir)
    manifest: dict[str, Any] = {"metadata": _metadata(), "ckpts": ckpts, "corpora": {}}
    halt_payload: dict[str, Any] | None = None
    for corpus in C.CORPORA:
        rows = load_corpus(corpus_dir, corpus, smoke=args.smoke)
        per_source: dict[str, Any] = {}
        inter: set[str] | None = None
        for m in ckpts:
            ok, stats = _unflagged_ids(out_root, corpus, m)
            per_source[m] = stats
            inter = ok if inter is None else (inter & ok)
        inter_ids = [r["id"] for r in rows if r["id"] in (inter or set())]
        groups = {r["id"]: r["group"] for r in rows}
        fold_of_group = assign_fold_groups([groups[i] for i in inter_ids])
        fold_sizes = [0] * C.N_FOLDS
        for i in inter_ids:
            fold_sizes[fold_of_group[groups[i]]] += 1
        n = len(inter_ids)
        min_n_tr = min((n - fs for fs in fold_sizes), default=0)
        # d from AutoConfig (A6: never hardcoded); smoke demotes the check to 0.
        if not args.smoke:
            pins = load_pins(out_root)
            dims_d = C.model_dims(
                C.MODEL_IDS[ckpts[0]], C.resolve_revision(ckpts[0], pins)
            ).hidden_size
        else:
            dims_d = 0
        entry = {
            "n_intersection": n,
            "target": C.INTERSECTION_TARGET,
            "floor": C.INTERSECTION_FLOOR,
            "per_source": per_source,
            "fold_seed": C.FOLD_SEED,
            "n_folds": C.N_FOLDS,
            "fold_sizes": fold_sizes,
            "min_n_tr": min_n_tr,
            "fold_of_group": fold_of_group,
            "ids": inter_ids,
            "resample_needed": bool(n < C.INTERSECTION_TARGET),
        }
        manifest["corpora"][corpus] = entry
        if args.smoke:
            if n <= 0:
                raise RuntimeError(f"smoke intersection EMPTY for {corpus} — rig bug, not scale")
            logger.info("[gateAprime] SMOKE informational: %s n=%d (floor demoted)", corpus, n)
            continue
        if n < C.INTERSECTION_FLOOR:
            halt_payload = {"corpus": corpus, "n_intersection": n, "floor": C.INTERSECTION_FLOOR}
        elif min_n_tr <= dims_d:
            halt_payload = {
                "corpus": corpus,
                "min_n_tr": min_n_tr,
                "d": dims_d,
                "note": (
                    "fold assignment is ALREADY greedy size-balanced "
                    "(assign_fold_groups: largest group into the smallest fold), so no "
                    "deterministic re-balance remains — min_n_tr <= d means the corpus "
                    "cannot support the fit at this d without a plan amendment "
                    "(splitting a group would break group-level fold integrity; "
                    "plan §7 gate A')"
                ),
            }
        logger.info(
            "[gateAprime] %s: n=%d (target %d, floor %d) min_n_tr=%d",
            corpus,
            n,
            C.INTERSECTION_TARGET,
            C.INTERSECTION_FLOOR,
            min_n_tr,
        )
    path = out_root / "gen" / "intersection_manifest.json"
    _write_json_atomic(path, manifest)
    upload_json_small(path, f"{C.EVAL_MIRROR_HF_PATH}/gen/intersection_manifest.json")
    write_phase_sentinel(
        out_root,
        "gen",
        {
            "gate_a_prime": "HALT" if halt_payload else "PASS",
            "n_intersection": {c: manifest["corpora"][c]["n_intersection"] for c in C.CORPORA},
        },
        smoke=args.smoke,
    )
    if halt_payload:
        designed_halt(out_root, "survival_gate_a_prime", halt_payload)


# ── capture (P3 + the pilot's capture legs) ──────────────────────────────────


def _load_hf_model(model_id: str, revision: str | None, device: str):
    """bf16 capture-model load with a LIVE free-HBM floor on cuda devices.

    Shared-node guard (#1902 crash 1 sweep): reads FREE memory via
    ``torch.cuda.mem_get_info`` on the target device — never total — and
    fails loud below ``CAPTURE_FREE_FLOOR_GIB`` (arithmetic at the constant)
    so a co-tenant's allocation surfaces at load, not as a CUDA OOM
    mid-forward. CPU devices (the tiny-real smoke) skip the probe.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if str(device).startswith("cuda"):
        free_b, total_b = torch.cuda.mem_get_info(torch.device(device))
        logger.info(
            "[capture] device=%s free=%.1fGiB total=%.1fGiB (floor=%.0fGiB)",
            device,
            free_b / 2**30,
            total_b / 2**30,
            CAPTURE_FREE_FLOOR_GIB,
        )
        if free_b / 2**30 < CAPTURE_FREE_FLOOR_GIB:
            raise RuntimeError(
                f"GPU {device} too full for the capture model + batch buffers: "
                f"free={free_b / 2**30:.1f} GiB < floor {CAPTURE_FREE_FLOOR_GIB:.0f} GiB "
                "(~15 GiB bf16 7B weights + ~8.5 GiB captured-layer stack at "
                "CAPTURE_TOKEN_BUDGET + pooling transients + margin). Shared-node "
                "co-tenancy (fellows H200) is the expected cause — re-dispatch when "
                "the device frees or pin a different allocated GPU."
            )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    return model


def _capture_row_entry(
    tokenizer,
    row: dict,
    answer_text: str,
    *,
    render: str,
) -> dict[str, Any]:
    """One teacher-forcing row: token-id CONCATENATION at the prompt/answer
    seam (#1092 — never re-tokenize the concatenated string), spans from the
    prompt render's offset mapping (straddler policy per plan §4 P3).

    ``render='plain'``: canonical ``User: {q}\\nAssistant:`` + `` {a}`` answer
    segment (``render_plain_full`` semantics — answers whitespace-normalized
    uniformly across sources so the answer TEXT is held fixed on the m axis).
    ``render='native'``: the checkpoint's chat template via
    ``compute_prompt_spans`` (single-turn robustness subset only).
    """
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    query = row["query"]
    prefix_turns = row.get("prefix_turns")
    answer_norm = answer_text.strip()
    seam_flags: dict[str, bool] = {}
    if render == "plain":
        text, q_start, q_end = plain_prompt_and_qspan(query, prefix_turns)
        prompt_ids, prefix_len, context_len, seam_flags = plain_spans(
            tokenizer, text, q_start, q_end
        )
        answer_ids = tokenizer.encode(" " + answer_norm, add_special_tokens=False)
    elif render == "native":
        assert not prefix_turns, "native robustness subset is single-turn only (plan §4 P3)"
        text = C.render_chat_prompt(tokenizer, query)
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        prefix_len, context_len = compute_prompt_spans(
            tokenizer,
            None,
            query,
            prompt_ids,
            on_seam="snap",
            seam_flags=seam_flags,
        )
        answer_ids = tokenizer.encode(answer_norm, add_special_tokens=False)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown render {render!r}")
    if not answer_ids:
        answer_ids = tokenizer.encode(" ", add_special_tokens=False)
    answer_ids = answer_ids[: C.GEN_MAX_TOKENS]
    return {
        "id": row["id"],
        "class": row.get("class"),
        "group": row.get("group"),
        "cluster": row.get("cluster"),
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "prefix_len": prefix_len,
        "context_len": context_len,
        "seam_prefix": seam_flags.get("prefix", False),
        "seam_context": seam_flags.get("context", False),
        "n_total": len(prompt_ids) + len(answer_ids),
    }


def _batches_by_token_budget(entries: list[dict]) -> list[list[dict]]:
    """Length-sorted batches under a padded-token budget (minimal padding)."""
    order = sorted(range(len(entries)), key=lambda i: entries[i]["n_total"])
    batches: list[list[dict]] = []
    cur: list[dict] = []
    for i in order:
        cand = cur + [entries[i]]
        max_t = max(e["n_total"] for e in cand)
        if cur and (len(cand) > CAPTURE_BATCH_MAX or len(cand) * max_t > CAPTURE_TOKEN_BUDGET):
            batches.append(cur)
            cur = [entries[i]]
        else:
            cur = cand
    if cur:
        batches.append(cur)
    return batches


def _inverse_batch_order(batches: list[list[dict]], n_entries: int):
    """Inverse permutation mapping batch-order concatenated rows back to
    original entries order.

    ``_batches_by_token_budget`` length-SORTS entries (minimal padding), so
    tensors built by concatenating per-batch outputs are in sorted order;
    indexing them with the returned ``torch.LongTensor`` restores entries
    order (saved row ``i`` == ``entries[i]``, matching the entries-order
    ``row_ids`` / ``row_index.jsonl``). Requires each entry to carry its
    original position under ``"_pos"``; asserts every position
    ``0..n_entries-1`` appears exactly once across the batches."""
    import torch

    flat = [e["_pos"] for b in batches for e in b]
    assert sorted(flat) == list(range(n_entries)), (
        f"batches must cover every entry exactly once (got {len(flat)} rows "
        f"for {n_entries} entries)"
    )
    return torch.argsort(torch.as_tensor(flat, dtype=torch.long))


def _pool_batch(
    model,
    entries: list[dict],
    layers: list[int],
    *,
    device: str,
    want_prefix: bool,
    fp32: bool = False,
):
    """ONE batched teacher-forced forward; GPU-resident masked-mean/last-token
    pooling per summary (vectorized — no per-row forwards); returns
    ``{layer: {summary: (B, H) cpu tensor}}`` in entry order."""
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations

    pad_id = 0
    bsz = len(entries)
    max_t = max(e["n_total"] for e in entries)
    ids = torch.full((bsz, max_t), pad_id, dtype=torch.long)
    mask = torch.zeros((bsz, max_t), dtype=torch.long)
    ctx_mask = torch.zeros((bsz, max_t), dtype=torch.float32)
    ans_mask = torch.zeros((bsz, max_t), dtype=torch.float32)
    pre_mask = torch.zeros((bsz, max_t), dtype=torch.float32)
    last_prompt = torch.zeros(bsz, dtype=torch.long)
    last_prefix = torch.zeros(bsz, dtype=torch.long)
    for b, e in enumerate(entries):
        seq = e["prompt_ids"] + e["answer_ids"]  # token-id concat (#1092)
        n_p, n_all = len(e["prompt_ids"]), len(seq)
        ids[b, :n_all] = torch.tensor(seq, dtype=torch.long)
        mask[b, :n_all] = 1
        ctx_mask[b, :n_p] = 1.0  # context = full prompt (plan §4 P3)
        ans_mask[b, n_p:n_all] = 1.0
        pre_mask[b, : e["prefix_len"]] = 1.0
        last_prompt[b] = n_p - 1
        last_prefix[b] = max(e["prefix_len"] - 1, 0)
    dev = torch.device(device)
    ids, mask = ids.to(dev), mask.to(dev)
    ctx_mask, ans_mask, pre_mask = ctx_mask.to(dev), ans_mask.to(dev), pre_mask.to(dev)
    last_prompt, last_prefix = last_prompt.to(dev), last_prefix.to(dev)

    captured = extract_layer_activations(model, ids, layers, attention_mask=mask)
    store_dtype = torch.float32 if fp32 else torch.float16
    arange_b = torch.arange(bsz, device=dev)
    out: dict[int, dict[str, Any]] = {}
    for layer in layers:
        hs = captured[layer].float()  # (B, T, H) — fp32 pooling accumulators
        pooled = {
            "u_last": hs[arange_b, last_prompt],
            "u_mean": (hs * ctx_mask[..., None]).sum(1) / ctx_mask.sum(1)[:, None],
            "w": (hs * ans_mask[..., None]).sum(1) / ans_mask.sum(1).clamp(min=1.0)[:, None],
        }
        if want_prefix:
            pooled["p_last"] = hs[arange_b, last_prefix]
            pooled["p_mean"] = (hs * pre_mask[..., None]).sum(1) / pre_mask.sum(1).clamp(min=1.0)[
                :, None
            ]
        out[layer] = {k: v.to(store_dtype).cpu() for k, v in pooled.items()}
        del hs
    captured.clear()
    return out


def _load_answers(out_root: Path, corpus: str, src: str, seed: int = C.GEN_SEED) -> dict[str, dict]:
    path = _gen_rollout_path(out_root, corpus, src, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"rollout file missing: {path} — run `--phase gen` before capture "
            "(same pod/out-root; gen outputs are the capture inputs)"
        )
    return {r["id"]: r for r in _read_jsonl(path)}


def _store_root(out_root: Path) -> Path:
    return out_root / "store"


def capture_cell(
    model,
    tokenizer,
    rows: list[dict],
    answers: dict[str, dict],
    layers: list[int],
    *,
    out_root: Path,
    ckpt: str,
    src_label: str,
    corpus: str,
    render: str,
    device: str,
    store_subdir: str | None = None,
    keep_fp32: bool = False,
    unit_tag: str = "",
) -> dict[str, Any]:
    """Capture ONE grid cell (activation ckpt x answer source x corpus).

    Writes per-layer answer summaries (+ ctx/prefix summaries ONCE per
    (ckpt, corpus) — the dedup'd layout, identical across the source axis by
    causal attention) + a ``row_index.jsonl`` manifest. Returns cell stats
    (rows, walls, seam fractions)."""
    import torch

    store = _store_root(out_root)
    kept_rows = [r for r in rows if r["id"] in answers]
    t_prep = time.time()
    entries = [
        _capture_row_entry(tokenizer, r, answers[r["id"]]["text"], render=render) for r in kept_rows
    ]
    want_prefix = corpus == C.CORPUS_MULTI
    if store_subdir is None:
        cell_dir = (store / C.answer_store_relpath(ckpt, src_label, corpus, layers[0])).parent
        ctx_dir = (store / C.ctx_store_relpath(ckpt, corpus, layers[0])).parent
    else:
        # Non-grid stores (reliability / robust / pilot): each cell owns its
        # ctx summaries (renders differ per arm — no cross-source dedup axis).
        cell_dir = store / store_subdir
        ctx_dir = cell_dir / "ctx"
    # ALL layers must exist to skip (a crash between per-layer saves must not
    # strand a partial ctx store on resume).
    ctx_written = all((ctx_dir / f"L{layer}.pt").exists() for layer in layers)
    n_layers_h: dict[int, list] = {layer: [] for layer in layers}
    ctx_acc: dict[int, dict[str, list]] = {layer: {} for layer in layers}
    for pos, e in enumerate(entries):
        e["_pos"] = pos  # original position — batching length-sorts (see _inverse_batch_order)
    batches = _batches_by_token_budget(entries)
    inv = _inverse_batch_order(batches, n_entries=len(entries))
    t0 = time.time()
    for bi, batch in enumerate(batches, start=1):
        pooled = _pool_batch(
            model, batch, layers, device=device, want_prefix=want_prefix, fp32=keep_fp32
        )
        for layer in layers:
            n_layers_h[layer].append(pooled[layer]["w"])
            if not ctx_written:
                for key in ("u_last", "u_mean") + (("p_last", "p_mean") if want_prefix else ()):
                    ctx_acc[layer].setdefault(key, []).append(pooled[layer][key])
        if bi % 10 == 0 or bi == len(batches):
            print(
                f"[capture]{unit_tag} batch {bi}/{len(batches)} "
                f"rows={sum(len(b) for b in batches[:bi])}/{len(entries)} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    per_row_wall = (time.time() - t0) / max(len(entries), 1)

    row_index = [
        {
            k: e[k]
            for k in (
                "id",
                "class",
                "group",
                "cluster",
                "prefix_len",
                "context_len",
                "seam_prefix",
                "seam_context",
            )
        }
        | {"n_prompt_tokens": len(e["prompt_ids"]), "n_answer_tokens": len(e["answer_ids"])}
        for e in entries
    ]
    cell_dir.mkdir(parents=True, exist_ok=True)
    fp32_dir = (
        cell_dir.parent.parent / f"{cell_dir.parent.name}_fp32" / cell_dir.name
        if keep_fp32
        else None
    )
    for layer in layers:
        # Batches are length-SORTED: unsort concatenated outputs back to
        # entries order (via inv) before saving under entries-order row_ids.
        w_full = torch.cat(n_layers_h[layer])[inv]  # fp32 when keep_fp32, else fp16
        row_ids = [e["id"] for e in entries]
        torch.save(
            {"w": w_full.to(torch.float16), "row_ids": row_ids},
            cell_dir / f"L{layer}.pt",
        )
        if fp32_dir is not None:
            # A12 twin: SAME capture, fp32 storage (pilot plain/single only).
            fp32_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"w": w_full, "row_ids": row_ids}, fp32_dir / f"L{layer}.pt")
    _write_jsonl_atomic(cell_dir / "row_index.jsonl", row_index)
    if fp32_dir is not None:
        _write_jsonl_atomic(fp32_dir / "row_index.jsonl", row_index)
    if not ctx_written:
        ctx_dir.mkdir(parents=True, exist_ok=True)
        for layer in layers:
            # Same unsort as the answer store: ctx accumulators are in
            # length-sorted batch order; row_ids below are entries order.
            full = {k: torch.cat(v)[inv] for k, v in ctx_acc[layer].items()}
            torch.save(
                {k: v.to(torch.float16) for k, v in full.items()}
                | {"row_ids": [e["id"] for e in entries]},
                ctx_dir / f"L{layer}.pt",
            )
            if fp32_dir is not None:
                (fp32_dir / "ctx").mkdir(parents=True, exist_ok=True)
                torch.save(
                    full | {"row_ids": [e["id"] for e in entries]},
                    fp32_dir / "ctx" / f"L{layer}.pt",
                )
        _write_jsonl_atomic(ctx_dir / "row_index.jsonl", row_index)
    seam_frac = sum(1 for e in entries if e["seam_prefix"] or e["seam_context"]) / max(
        len(entries), 1
    )
    return {
        "n_rows": len(entries),
        "n_dropped_no_answer": len(rows) - len(kept_rows),
        "per_row_wall_s": round(per_row_wall, 4),
        "prep_wall_s": round(t0 - t_prep, 1),
        "seam_fraction": round(seam_frac, 4),
        "cell_dir": str(cell_dir),
        "ctx_written": not ctx_written,
    }


def capture_store_roots(store: Path, ckpt: str) -> list[Path]:
    """Every store subtree the capture/pilot phases write for checkpoint
    ``ckpt`` — the upload-eligibility UNION (#825 uploader-parity class;
    concern store-upload-misses-reliability-robust-pilot-subtrees): the grid
    subtree PLUS reliability / robust_native / pilot (incl. the pilot A12
    ``*_fp32`` twins, which live under ``pilot/<ckpt>``)."""
    return [
        store / ckpt,
        store / "reliability" / ckpt,
        store / "robust_native" / ckpt,
        store / "pilot" / ckpt,
    ]


def _upload_ckpt_store(out_root: Path, ckpt: str) -> dict[str, Any]:
    """Per-checkpoint incremental store upload -> verify -> conditional
    delete-local (plan §4 P3; upload_dir_sharded owns verify + overflow).
    Enumerates ALL capture subtrees the leg owns (``capture_store_roots``) —
    plan §10 declares the whole store class persisted with
    ``discarded_artifacts: []``."""
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    store = _store_root(out_root)
    roots = capture_store_roots(store, ckpt)
    if not roots[0].is_dir():
        raise FileNotFoundError(f"no store written under {roots[0]}")
    present = [r for r in roots if r.is_dir()]
    absent = [r.relative_to(store).as_posix() for r in roots if not r.is_dir()]
    if absent:
        # robust_native is legitimately absent for B (A3: native IS plain);
        # pilot/<ckpt> is absent when the pilot leg ran on another out-root.
        logger.info("[store-upload] ckpt %s: absent subtrees skipped: %s", ckpt, absent)
    st = os.statvfs(out_root)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    delete_local = free_gb < DELETE_LOCAL_FREE_GB
    results: dict[str, Any] = {
        "delete_local": delete_local,
        "free_gb_before": round(free_gb, 1),
        "absent_subtrees": absent,
    }
    leaf_dirs = sorted(
        {p.parent for root in present for p in root.rglob("*") if p.is_file()},
        key=lambda p: str(p),
    )
    for leaf in leaf_dirs:
        rel = leaf.relative_to(store).as_posix()
        res = upload_dir_sharded(
            leaf,
            C.HF_DATA_REPO,
            f"{C.STORE_HF_PATH}/{rel}",
            repo_type="dataset",
            verify=True,
            delete_local=delete_local,
        )
        results[rel] = {
            "uploaded": len(res.uploaded),
            "skipped_existing": len(res.skipped_existing),
            "rerouted": len(res.rerouted),
        }
        logger.info("[store-upload] %s: %s", rel, results[rel])
    return results


def phase_capture_ckpt(args: argparse.Namespace, out_root: Path, ckpts: list[str]) -> None:
    """P3 shard leg: every capture unit for ONE activation checkpoint m —
    grid cells (4 sources x 2 corpora on the intersection), reliability
    (diagonal, 1k subset, seeds 43/44), native robustness (diagonal, 2k,
    single) — then the per-checkpoint incremental store upload."""
    ckpt = args.ckpt
    pins = load_pins(out_root)
    revision = C.resolve_revision(ckpt, pins)
    model_id = C.MODEL_IDS[ckpt]
    corpus_dir = Path(args.corpus_dir)
    manifest = _read_json(out_root / "gen" / "intersection_manifest.json")

    dims = C.model_dims(model_id, revision)
    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(C.capture_layers(dims.num_layers))
    )
    if args.smoke and not args.layers:
        layers = probe_layers(dims.num_layers)

    rows_by_corpus: dict[str, list[dict]] = {}
    for corpus in C.CORPORA:
        rows = load_corpus(corpus_dir, corpus, smoke=args.smoke)
        ids = set(manifest["corpora"][corpus]["ids"])
        rows_by_corpus[corpus] = [r for r in rows if r["id"] in ids]

    units: list[dict[str, Any]] = []
    for corpus in C.CORPORA:
        for src in ckpts:
            units.append(
                {
                    "unit": f"capture_{ckpt}_{src}_{corpus}",
                    "rows": rows_by_corpus[corpus],
                    "src": src,
                    "corpus": corpus,
                    "render": "plain",
                    "seed": C.GEN_SEED,
                    "subdir": None,
                }
            )
    single_rows = rows_by_corpus[C.CORPUS_SINGLE]
    rel_pool = {r["id"] for r in single_rows} & set(
        reliability_ids(load_corpus(corpus_dir, C.CORPUS_SINGLE, smoke=args.smoke))
    )
    rel_rows = [r for r in single_rows if r["id"] in rel_pool]
    for seed in C.RELIABILITY_SEEDS:
        units.append(
            {
                "unit": f"capture_{ckpt}_rel{seed}_{C.CORPUS_SINGLE}",
                "rows": rel_rows,
                "src": ckpt,
                "corpus": C.CORPUS_SINGLE,
                "render": "plain",
                "seed": seed,
                "subdir": f"reliability/{ckpt}/{C.CORPUS_SINGLE}/seed{seed}",
            }
        )
    robust_n = ROBUST_NATIVE_N if not args.smoke else len(single_rows)
    if ckpt != "B":
        # Base has NO chat template (plan A3): its native render IS the plain
        # render (§4 P2), so the robustness read only differs for S/D/R.
        units.append(
            {
                "unit": f"capture_{ckpt}_robustnative_{C.CORPUS_SINGLE}",
                "rows": single_rows[:robust_n],
                "src": ckpt,
                "corpus": C.CORPUS_SINGLE,
                "render": "native",
                "seed": C.GEN_SEED,
                "subdir": f"robust_native/{ckpt}/{C.CORPUS_SINGLE}",
            }
        )

    regimes = {
        u["unit"]: unit_regime(
            args,
            phase="capture",
            ckpt=ckpt,
            src=u["src"],
            corpus=u["corpus"],
            render=u["render"],
            seed=u["seed"],
            layers=layers,
            n_rows=len(u["rows"]),
        )
        for u in units
    }
    upload_regime = unit_regime(args, phase="capture_upload", ckpt=ckpt, layers=layers)
    upload_unit = f"capture_upload_{ckpt}"
    # Artifact-aware resume (#1315 class): a done-sentinel counts only when the
    # unit's store artifacts are still local OR the leg's VERIFIED upload
    # record exists (post-delete-local, P4 re-stages from HF).
    leg_uploaded = unit_done(out_root, upload_unit, upload_regime)
    store = _store_root(out_root)
    pending = [
        u
        for u in units
        if not (
            unit_done(out_root, u["unit"], regimes[u["unit"]])
            and (leg_uploaded or capture_unit_artifacts_present(store, ckpt, u, layers))
        )
    ]
    headroom_gate(out_root, "capture", len(pending), CAPTURE_PER_CELL_GB)
    print(f"[phase=capture] ckpt={ckpt} units={len(pending)}/{len(units)}", flush=True)
    if pending:
        tokenizer = _tokenizer(model_id, revision)
        model = _load_hf_model(model_id, revision, args.device)
        for k, u in enumerate(pending, start=1):
            answers = _load_answers(out_root, u["corpus"], u["src"], u["seed"])
            t0 = time.time()
            stats = capture_cell(
                model,
                tokenizer,
                u["rows"],
                answers,
                layers,
                out_root=out_root,
                ckpt=ckpt,
                src_label=u["src"],
                corpus=u["corpus"],
                render=u["render"],
                device=args.device,
                store_subdir=u["subdir"],
                unit_tag=f" {u['unit']}",
            )
            mark_unit_done(out_root, u["unit"], regimes[u["unit"]], stats)
            print(
                f"[capture] unit {k}/{len(pending)} {u['unit']} rows={stats['n_rows']} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cache release is best-effort on CPU hosts
            pass
    if not leg_uploaded:
        results = _upload_ckpt_store(out_root, ckpt)
        mark_unit_done(out_root, upload_unit, upload_regime, results)


def phase_capture_finalize(args: argparse.Namespace, out_root: Path, ckpts: list[str]) -> None:
    """P3 finalize: assert every per-ckpt leg completed; summary + sentinel."""
    summary: dict[str, Any] = {"metadata": _metadata(), "ckpts": ckpts, "legs": {}}
    missing: list[str] = []
    for m in ckpts:
        path = _state_dir(out_root) / f"capture_upload_{m}.done.json"
        if path.exists():
            summary["legs"][m] = _read_json(path).get("ts")
        else:
            missing.append(m)
    if missing:
        raise RuntimeError(f"capture finalize: legs incomplete for ckpts {missing}")
    path = out_root / "capture_summary.json"
    _write_json_atomic(path, summary)
    upload_json_small(path, f"{C.EVAL_MIRROR_HF_PATH}/capture/capture_summary.json")
    write_phase_sentinel(out_root, "capture", {"ckpts": ckpts}, smoke=args.smoke)


# ── pilot (P1) ───────────────────────────────────────────────────────────────


def probe_layers(num_layers: int) -> list[int]:
    """3 depth-relative probe layers (plan §4 P1; AutoConfig-derived — A6)."""
    layers = sorted({max(0, int(round(f * num_layers))) for f in PROBE_LAYER_FRACS})
    assert len(layers) == 3, layers
    return layers


def _pilot_dir(out_root: Path, ckpt: str) -> Path:
    return out_root / "pilot" / ckpt


def _bf16_equivalence_gate(model, entries: list[dict], layers: list[int], device: str) -> dict:
    """bf16 batched-vs-batch-1 equivalence, two-bar calibration (#779 r12 /
    plan §4 P3): per-layer cos >= 0.999 on layers 0-3; flattened all-layer
    cos >= 0.995. Compares RAW hidden states on valid positions."""
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations

    rows = entries[:BF16_GATE_ROWS]
    gate_layers = sorted(set(BF16_EARLY_LAYERS) | set(layers))
    dev = torch.device(device)
    pad_id = 0
    max_t = max(e["n_total"] for e in rows)
    ids = torch.full((len(rows), max_t), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), max_t), dtype=torch.long)
    for b, e in enumerate(rows):
        seq = e["prompt_ids"] + e["answer_ids"]
        ids[b, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[b, : len(seq)] = 1
    batched = extract_layer_activations(
        model, ids.to(dev), gate_layers, attention_mask=mask.to(dev)
    )
    batched = {layer: t.float().cpu() for layer, t in batched.items()}

    early_cos: list[float] = []
    flat_cos: list[float] = []
    for b, e in enumerate(rows):
        seq = e["prompt_ids"] + e["answer_ids"]
        one = extract_layer_activations(
            model,
            torch.tensor([seq], dtype=torch.long, device=dev),
            gate_layers,
            attention_mask=torch.ones((1, len(seq)), dtype=torch.long, device=dev),
        )
        one = {layer: t.float().cpu() for layer, t in one.items()}
        flats_b, flats_1 = [], []
        for layer in gate_layers:
            hb = batched[layer][b, : len(seq)].reshape(-1)
            h1 = one[layer][0].reshape(-1)
            cos = torch.nn.functional.cosine_similarity(hb, h1, dim=0).item()
            if layer in BF16_EARLY_LAYERS:
                early_cos.append(cos)
            flats_b.append(hb)
            flats_1.append(h1)
        flat_cos.append(
            torch.nn.functional.cosine_similarity(
                torch.cat(flats_b), torch.cat(flats_1), dim=0
            ).item()
        )
    return {
        "n_rows": len(rows),
        "gate_layers": gate_layers,
        "early_cos_min": min(early_cos),
        "flat_cos_min": min(flat_cos),
        "early_bar": BF16_EARLY_COS_MIN,
        "flat_bar": BF16_FLAT_COS_MIN,
        "pass": bool(min(early_cos) >= BF16_EARLY_COS_MIN and min(flat_cos) >= BF16_FLAT_COS_MIN),
    }


def phase_pilot_ckpt(args: argparse.Namespace, out_root: Path) -> None:
    """P1 shard leg for ONE checkpoint: native-render generation on the first
    PILOT_N rows of BOTH corpora (the pilot rows ARE production rows — A8),
    survival counts, then BOTH-serialization diagonal capture on the 3 probe
    layers (fp16 + fp32 kept for A12), timing + span validation (A19), and —
    on the R leg — the bf16 batched-vs-batch-1 equivalence read."""
    import torch

    ckpt = args.ckpt
    pins = load_pins(out_root)
    revision = C.resolve_revision(ckpt, pins)
    model_id = C.MODEL_IDS[ckpt]
    corpus_dir = Path(args.corpus_dir)
    pdir = _pilot_dir(out_root, ckpt)
    unit = f"pilot_{ckpt}"
    n_pilot = PILOT_N if not args.smoke else SMOKE_SINGLE_N
    regime = unit_regime(args, phase="pilot", ckpt=ckpt, n=n_pilot)
    if unit_done(out_root, unit, regime):
        logger.info("[pilot] ckpt=%s leg done — skipped", ckpt)
        return
    headroom_gate(out_root, "pilot", 1, 2.0)
    print(f"[phase=pilot] ckpt={ckpt} n={n_pilot}", flush=True)

    single = load_corpus(corpus_dir, C.CORPUS_SINGLE, smoke=args.smoke, limit=n_pilot)
    multi = load_corpus(corpus_dir, C.CORPUS_MULTI, smoke=args.smoke, limit=n_pilot)
    tokenizer = _tokenizer(model_id, revision)
    dims = C.model_dims(model_id, revision)
    layers = probe_layers(dims.num_layers)

    # 1) generation (native render — the production P2 recipe at pilot n).
    leg: dict[str, Any] = {"ckpt": ckpt, "revision": revision, "n_pilot": n_pilot}
    llm = _vllm_engine(model_id, revision, dims.max_position_embeddings)
    for corpus, rows in ((C.CORPUS_SINGLE, single), (C.CORPUS_MULTI, multi)):
        t0 = time.time()
        sp = _sampling_params(C.GEN_SEED, ckpt, smoke=args.smoke)
        gens = _generate_chunked(llm, _gen_prompts(rows, ckpt, tokenizer), sp)
        recs = _flag_records(rows, gens, C.GEN_SEED, gen_cap=int(sp.max_tokens))
        _write_jsonl_atomic(pdir / f"gen_{corpus}.jsonl", recs)
        upload_text_payload(pdir / f"gen_{corpus}.jsonl", f"{C.RAW_GEN_HF_PATH}/pilot_{corpus}")
        n_ok = sum(1 for r in recs if not (r["truncated"] or r["repetition_flag"]))
        leg[f"survival_{corpus}"] = {
            "n": len(recs),
            "n_unflagged": n_ok,
            "rate": n_ok / max(len(recs), 1),
        }
        logger.info(
            "[pilot] gen %s: %d/%d unflagged (%.0fs)", corpus, n_ok, len(recs), time.time() - t0
        )
    # Free the vLLM engine before the HF capture model loads (#653 teardown).
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2) BOTH-serialization diagonal capture at the probe layers.
    model = _load_hf_model(model_id, revision, args.device)
    span_rows = 0
    for corpus, rows in ((C.CORPUS_SINGLE, single), (C.CORPUS_MULTI, multi)):
        answers = {r["id"]: r for r in _read_jsonl(pdir / f"gen_{corpus}.jsonl")}
        unflagged = {
            i: a for i, a in answers.items() if not (a["truncated"] or a["repetition_flag"])
        }
        # Base has NO chat template (plan A3) — its "native" render IS the
        # plain render (§4 P2), so a separate native capture leg for B would
        # both crash (apply_chat_template raises) and duplicate the plain leg.
        renders = ["plain", "native"] if corpus == C.CORPUS_SINGLE and ckpt != "B" else ["plain"]
        for render in renders:
            t0 = time.time()
            stats = capture_cell(
                model,
                tokenizer,
                rows,
                unflagged,
                layers,
                out_root=out_root,
                ckpt=ckpt,
                src_label=ckpt,
                corpus=corpus,
                render=render,
                device=args.device,
                store_subdir=f"pilot/{ckpt}/{render}/{corpus}",
                keep_fp32=(render == "plain" and corpus == C.CORPUS_SINGLE),
                unit_tag=f" pilot/{ckpt}/{render}/{corpus}",
            )
            leg[f"capture_{render}_{corpus}"] = stats
            span_rows += stats["n_rows"]
            logger.info(
                "[pilot] capture %s/%s: %d rows, %.3fs/row (%.0fs)",
                render,
                corpus,
                stats["n_rows"],
                stats["per_row_wall_s"],
                time.time() - t0,
            )
    leg["span_validated_rows"] = span_rows  # A19: every row's spans asserted in-path

    # 3) bf16 batched-vs-batch-1 equivalence read (R leg carries the gate).
    if ckpt == "R":
        answers = {r["id"]: r for r in _read_jsonl(pdir / f"gen_{C.CORPUS_SINGLE}.jsonl")}
        entries = [
            _capture_row_entry(tokenizer, r, answers[r["id"]]["text"], render="plain")
            for r in single
            if r["id"] in answers
        ][:BF16_GATE_ROWS]
        leg["bf16_gate"] = _bf16_equivalence_gate(model, entries, layers, args.device)
        logger.info("[pilot] bf16 gate: %s", leg["bf16_gate"])
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _write_json_atomic(pdir / "leg_report.json", {"metadata": _metadata(), **leg})
    mark_unit_done(out_root, unit, regime, {"n_pilot": n_pilot})


def _pilot_fits(out_root: Path, layers: list[int], device: str, smoke: bool) -> dict[str, Any]:
    """Pilot 3-layer diagonal ridge fits (R): flip-rule inputs (plain vs
    native Q(R,R)) + the A12 fp16-vs-fp32 ΔR² read + one timed MLP unit.

    n_pilot < d — a deliberately under-determined PILOT regime: reads are
    RELATIVE (same estimator + n on both arms), λ*/dof persisted per fit so a
    grid-edge GCV selection stays visible (#1417 / #1775 λ-discipline)."""
    import numpy as np
    import torch

    from explore_persona_space.experiments.issue_779.fit_h import (
        mlp_fit_predict,
        ridge_fit_predict_fast,
    )

    store = _store_root(out_root)

    def _load_arm(render_dir: str) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """X = u_mean from the cell's ctx/ store; Y = w from the cell store."""
        xs, ys = {}, {}
        cell = store / "pilot" / "R" / render_dir / C.CORPUS_SINGLE
        for layer in layers:
            dw = torch.load(cell / f"L{layer}.pt", map_location="cpu", weights_only=True)
            dx = torch.load(cell / "ctx" / f"L{layer}.pt", map_location="cpu", weights_only=True)
            assert dx["row_ids"] == dw["row_ids"], f"ctx/answer row mismatch in {cell} L{layer}"
            xs[layer] = dx["u_mean"].to(torch.float64).numpy()
            ys[layer] = dw["w"].to(torch.float64).numpy()
        return xs, ys

    arms = {
        "plain_fp16": _load_arm("plain"),
        "native_fp16": _load_arm("native"),
    }
    if (store / "pilot" / "R" / "plain_fp32" / C.CORPUS_SINGLE).exists():
        arms["plain_fp32"] = _load_arm("plain_fp32")

    n = next(iter(arms["plain_fp16"][0].values())).shape[0]
    rng = np.random.default_rng(C.FOLD_SEED)
    fold_of = rng.integers(0, PILOT_FOLDS, size=n)
    fits: dict[str, Any] = {"n": n, "folds": PILOT_FOLDS, "n_lt_d_regime": True, "arms": {}}
    for arm, (xs, ys) in arms.items():
        per_layer: dict[str, Any] = {}
        for layer in layers:
            X, Y = xs[layer], ys[layer]
            ss_res, ss_tot, lams, dofs = 0.0, 0.0, [], []
            for f in range(PILOT_FOLDS):
                tr, ev = fold_of != f, fold_of == f
                if ev.sum() < 2 or tr.sum() < 2:
                    continue
                pred, info = ridge_fit_predict_fast(
                    X[tr], Y[tr], X[ev], device=device, return_info=True
                )
                resid = Y[ev] - pred
                ss_res += float((resid**2).sum())
                ss_tot += float(((Y[ev] - Y[tr].mean(0)) ** 2).sum())
                lams.append(float(info["best_lambda"]))
                dofs.append(float(info["dof"]))
            per_layer[str(layer)] = {
                "oof_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                "lambda_star": lams,
                "dof": dofs,
            }
        fits["arms"][arm] = per_layer

    plain_q = {ell: fits["arms"]["plain_fp16"][str(ell)]["oof_r2"] for ell in layers}
    native_q = {ell: fits["arms"]["native_fp16"][str(ell)]["oof_r2"] for ell in layers}
    fits["flip_rule"] = {
        "plain_q": plain_q,
        "native_q": native_q,
        "ratio_bar": PILOT_FLIP_RATIO,
        "flip_triggered": all(plain_q[ell] < PILOT_FLIP_RATIO * native_q[ell] for ell in layers)
        if not smoke
        else False,
    }
    if "plain_fp32" in fits["arms"]:
        deltas = {
            str(ell): abs(
                fits["arms"]["plain_fp32"][str(ell)]["oof_r2"]
                - fits["arms"]["plain_fp16"][str(ell)]["oof_r2"]
            )
            for ell in layers
        }
        fits["fp16_delta_r2"] = {
            "deltas": deltas,
            "bar": FP16_DELTA_R2_MAX,
            "pass": max(deltas.values()) < FP16_DELTA_R2_MAX,
        }

    # Timed MLP unit (de-asserts the §9 P4 MLP basis).
    mid = layers[len(layers) // 2]
    X, Y = arms["plain_fp16"][0][mid], arms["plain_fp16"][1][mid]
    tr = fold_of != 0
    pca_k = max(1, min(64, int(tr.sum()) // 2))
    t0 = time.time()
    mlp_fit_predict(X[tr], Y[tr], X[~tr], pca_k=pca_k, seed=42, device=device)
    fits["mlp_unit_wall_s"] = round(time.time() - t0, 2)
    fits["mlp_unit_shape"] = {"n_tr": int(tr.sum()), "d": int(X.shape[1]), "pca_k": pca_k}
    return fits


def _timed_shard_upload(out_root: Path, hidden: int, smoke: bool) -> dict[str, Any]:
    """Timed store-shard serialize + upload_dir_sharded leg (de-asserts the
    §9 store/upload basis). Production-shape shard: (target, hidden) fp16."""
    import torch

    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    n = C.INTERSECTION_TARGET if not smoke else 256
    tdir = out_root / "pilot" / "timing_shard"
    tdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.save({"w": torch.randn(n, hidden, dtype=torch.float16)}, tdir / "L0.pt")
    serialize_s = time.time() - t0
    t0 = time.time()
    upload_dir_sharded(
        tdir,
        C.HF_DATA_REPO,
        C.PILOT_TIMING_HF_PATH,
        repo_type="dataset",
        verify=True,
        delete_local=True,
    )
    return {
        "shard_rows": n,
        "hidden": hidden,
        "serialize_s": round(serialize_s, 2),
        "upload_verify_s": round(time.time() - t0, 2),
    }


def capture_rows_per_leg(n_ckpts: int, isect_by_corpus: dict[str, int]) -> int:
    """Max-leg P3 capture-row projection (review r1 M2): each activation-ckpt
    leg captures ``n_ckpts`` answer-source cells over the FULL realized
    intersection of EACH corpus — ``phase_capture_ckpt`` filters by manifest
    ids with NO cap — plus the robustness read (non-B legs; kept in the
    max-leg projection) and the two reliability subsets. The old
    ``2 * min(isect, INTERSECTION_TARGET)`` basis under-projected up to ~2x
    on realized intersections above target and/or asymmetric corpora."""
    return int(
        n_ckpts * sum(int(v) for v in isect_by_corpus.values())
        + ROBUST_NATIVE_N
        + 2 * C.RELIABILITY_SUBSET_N
    )


def phase_pilot_finalize(args: argparse.Namespace, out_root: Path, ckpts: list[str]) -> None:
    """P1 finalize: pilot_report.json (revision pins BINDING from here on),
    Gate A projection, bf16 two-bar gate verdict, A12 ΔR², flip rule, timing
    bases + the capture-cost abort (plan §7). Designed halts exit rc=7."""
    pins = load_pins(out_root)
    corpus_dir = Path(args.corpus_dir)
    legs = {m: _read_json(_pilot_dir(out_root, m) / "leg_report.json") for m in ckpts}

    report: dict[str, Any] = {
        "metadata": _metadata(),
        C.REVISION_PINS_KEY: pins,
        "smoke": bool(args.smoke),
        "ckpts": ckpts,
        "legs": legs,
    }

    # Gate A: projected four-source intersection (per corpus) after re-sample.
    gate_a: dict[str, Any] = {}
    halt: dict[str, Any] | None = None
    for corpus in C.CORPORA:
        rows_n = len(load_corpus(corpus_dir, corpus, smoke=args.smoke))
        surv = {m: legs[m][f"survival_{corpus}"]["rate"] for m in ckpts}
        prod = 1.0
        for v in surv.values():
            prod *= v
        projected = int(rows_n * prod)
        extra_rows = 0
        stats_path = corpus_dir / "manifest_stats.json"
        if stats_path.exists():
            st = _read_json(stats_path)
            scanned = int(st.get("stream", {}).get("scanned", 0) or 0)
            scan_cap = int(st.get("scan_cap", 0) or 0)
            if scanned > 0 and scan_cap > scanned:
                keep_rate = rows_n / scanned
                extra_rows = int((scan_cap - scanned) * keep_rate)
        projected_resample = int((rows_n + extra_rows) * prod)
        gate_a[corpus] = {
            "draw_n": rows_n,
            "survival": surv,
            "projected": projected,
            "resample_extra_rows": extra_rows,
            "projected_after_resample": projected_resample,
            "floor": C.INTERSECTION_FLOOR,
        }
        if not args.smoke and projected_resample < C.INTERSECTION_FLOOR:
            halt = {"corpus": corpus, **gate_a[corpus]}
    report["gate_a"] = gate_a

    # bf16 two-bar gate (from the R leg).
    bf16 = legs.get("R", {}).get("bf16_gate")
    report["bf16_gate"] = bf16
    if bf16 and not bf16["pass"] and not args.smoke:
        report_path = out_root / "pilot_report.json"
        _write_json_atomic(report_path, report)
        designed_halt(out_root, "bf16_equivalence", bf16)

    # Pilot fits: flip rule + A12 + timed MLP unit.
    dims = C.model_dims(C.MODEL_IDS["R"], C.resolve_revision("R", pins))
    layers = probe_layers(dims.num_layers)
    report["fits"] = _pilot_fits(out_root, layers, args.device, args.smoke)
    a12 = report["fits"].get("fp16_delta_r2")
    if a12 and not a12["pass"] and not args.smoke:
        _write_json_atomic(out_root / "pilot_report.json", report)
        designed_halt(out_root, "fp16_delta_r2", a12)
    if args.smoke:
        logger.info("[pilot] SMOKE: production-n gates demoted to informational lines")

    # Timing legs: store-shard serialize+upload; measured per-row capture wall
    # -> P3 projection + the capture-cost abort (§7).
    report["timing_shard"] = _timed_shard_upload(out_root, dims.hidden_size, args.smoke)
    walls = [
        legs[m][key]["per_row_wall_s"]
        for m in ckpts
        for key in legs[m]
        if key.startswith("capture_")
        and isinstance(legs[m][key], dict)
        and "per_row_wall_s" in legs[m][key]
    ]
    per_row = max(walls) if walls else float("nan")
    # Capture-cost basis = the REALIZED intersection of the FIXED corpus
    # ("projected" = corpus rows x pilot survival product — what P2's manifest
    # will actually contain and P3 captures). "projected_after_resample" is
    # gate A's scan-cap RESCUE CAPACITY (how far a corpus re-scan could
    # extend the pool if the floor were missed) — ~11x the realized set on
    # job 16145, where it false-fired this gate at 14.82h/leg vs a true
    # ~1.4h/leg (fellows job 16145, 2026-07-31).
    isect_by_corpus = {c: int(g["projected"]) for c, g in gate_a.items()}
    rows_per_m = capture_rows_per_leg(len(ckpts), isect_by_corpus)
    projected_wall_h = rows_per_m * per_row / 3600.0
    report["capture_cost"] = {
        "per_row_wall_s": per_row,
        "projected_intersection_by_corpus": isect_by_corpus,
        "rows_per_ckpt_leg": rows_per_m,
        "projected_wall_h_per_leg": round(projected_wall_h, 2),
        "planned_wall_h": CAPTURE_PLANNED_WALL_H,
        "abort_ratio": CAPTURE_COST_ABORT_RATIO,
    }
    report_path = out_root / "pilot_report.json"
    _write_json_atomic(report_path, report)
    upload_json_small(report_path, f"{C.EVAL_MIRROR_HF_PATH}/pilot/pilot_report.json")
    write_phase_sentinel(
        out_root,
        "pilot",
        {
            "gate_a": {c: gate_a[c]["projected_after_resample"] for c in C.CORPORA},
            "bf16_pass": None if bf16 is None else bf16["pass"],
            "flip_triggered": report["fits"]["flip_rule"]["flip_triggered"],
        },
        smoke=args.smoke,
    )
    if halt is not None:
        designed_halt(out_root, "survival_gate_a", halt)
    if not args.smoke and projected_wall_h > CAPTURE_COST_ABORT_RATIO * CAPTURE_PLANNED_WALL_H:
        designed_halt(out_root, "capture_cost", report["capture_cost"])


# ── stage (corpus staging from HF) ───────────────────────────────────────────


def phase_stage(args: argparse.Namespace, out_root: Path) -> None:
    """Stage the P0 corpora (local-first; mirror-ROOT arithmetic per #1774:
    dest satisfies root/<hub prefix> == consumed corpus dir)."""
    corpus_dir = Path(args.corpus_dir)
    needed = [
        corpus_dir / C.CORPUS_SINGLE_FILENAME,
        corpus_dir / C.CORPUS_MULTI_FILENAME,
        corpus_dir / C.CLUSTERS_FILENAME,
    ]
    if all(p.exists() for p in needed):
        logger.info("[stage] corpus already local at %s — staging skipped", corpus_dir)
        print(f"[stage] corpus staged: {len(needed)} files (local)", flush=True)
        return
    from explore_persona_space.orchestrate.hub import stage_hub_prefix

    mirror_root = out_root / "corpus_stage"
    assert (mirror_root / C.CORPUS_HF_PATH) == corpus_dir, (
        f"--corpus-dir must equal <out-root>/corpus_stage/{C.CORPUS_HF_PATH} "
        f"(mirror-root rule, #1774); got {corpus_dir}"
    )
    files = stage_hub_prefix(C.HF_DATA_REPO, C.CORPUS_HF_PATH, mirror_root, repo_type="dataset")
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"staged corpus incomplete — missing {missing}")
    print(f"[stage] corpus staged: {len(files)} files", flush=True)


# ── import-check (deferred-import execution; smoke Axis-1 leg) ───────────────


def _import_check() -> None:
    """Execute EVERY deferred import this driver's phases resolve (#606/#1689
    class): a broken lazy import fails HERE, not minutes into the pod run."""
    import numpy  # noqa: F401
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    from vllm import LLM, SamplingParams  # noqa: F401

    from explore_persona_space.analysis.extraction import (  # noqa: F401
        extract_layer_activations,
    )
    from explore_persona_space.analysis.representation_shift import (  # noqa: F401
        _reap_vllm_engine,
        compute_prompt_spans,
    )
    from explore_persona_space.eval.generation import create_vllm_engine  # noqa: F401
    from explore_persona_space.experiments.issue_779.fit_h import (  # noqa: F401
        mlp_fit_predict,
        ridge_fit_predict_fast,
        ridge_fit_predict_fast_layer_batched,
    )
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        retry_transient,
        stage_hub_prefix,
        verify_repo_paths_uploaded,
    )
    from explore_persona_space.orchestrate.preflight import (  # noqa: F401
        assert_out_root_headroom,
    )
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded  # noqa: F401

    # P4 fits module + ITS deferred stack (unit C).
    from issue1902_fits import run_fits  # noqa: F401
    from issue825_crossmodel_map_transfer import principal_angles  # noqa: F401
    from issue825_map_alignment import _procrustes_cosine_null  # noqa: F401

    from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
        identity_bias_predict,
        knn_retrieval,
    )
    from explore_persona_space.analysis.representation_shift import (  # noqa: F401
        cka_per_layer,
        linear_cka,
    )

    print("[import-check] OK: all deferred imports resolved", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────


def _default_out_root() -> Path:
    env = os.environ.get("EPM_OUT_ROOT")
    if env:
        return Path(env)
    wr = os.environ.get("WORKLOAD_ROOT")
    if wr:
        return Path(wr) / "issue1902"
    if Path("/workspace").is_dir():
        return Path("/workspace/issue1902")
    return PROJECT_ROOT / "data" / "issue_1902" / "out"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        required=True,
        choices=["stage", "pilot", "gen", "capture", "fits"],
        help="fits is a REGISTRATION POINT for unit C (scripts/issue1902_fits.py)",
    )
    ap.add_argument("--ckpt", choices=list(C.CKPTS), help="shard leg checkpoint")
    ap.add_argument("--init", action="store_true", help="pilot: resolve+persist revision pins")
    ap.add_argument("--finalize", action="store_true", help="run the phase's finalize leg")
    ap.add_argument("--smoke", action="store_true", help="tiny slice; prod-n gates demoted")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--corpus-dir", type=Path, default=None)
    ap.add_argument("--ckpts", default=" ".join(C.CKPTS), help="realized checkpoint set")
    ap.add_argument("--gpu-id", type=int, default=0, help="physical GPU (CVD-pinned by launcher)")
    ap.add_argument("--device", default=None, help="torch device (default cuda if available)")
    ap.add_argument("--layers", default=None, help="comma-separated capture layer override")
    ap.add_argument("--import-check", action="store_true", help="execute deferred imports; exit")
    args = ap.parse_args()

    if args.import_check:
        _import_check()
        sys.exit(0)

    out_root = args.out_root or _default_out_root()
    out_root.mkdir(parents=True, exist_ok=True)
    if args.corpus_dir is None:
        args.corpus_dir = out_root / "corpus_stage" / C.CORPUS_HF_PATH
    if args.device is None:
        try:
            import torch

            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 — CPU structural checks without torch cuda
            args.device = "cpu"
    ckpts = args.ckpts.split()
    for m in ckpts:
        C.MODEL_IDS[m]  # fail loud on an unknown checkpoint token

    logger.info(
        "[main] phase=%s ckpt=%s finalize=%s smoke=%s out_root=%s device=%s",
        args.phase,
        args.ckpt,
        args.finalize,
        args.smoke,
        out_root,
        args.device,
    )

    if args.phase == "fits":
        from issue1902_fits import run_fits

        run_fits(args, out_root, ckpts)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)
    if args.phase == "stage":
        phase_stage(args, out_root)
    elif args.phase == "pilot":
        if args.init:
            ensure_pins(out_root)
        elif args.finalize:
            phase_pilot_finalize(args, out_root, ckpts)
        else:
            assert args.ckpt, "--phase pilot needs --ckpt, --init, or --finalize"
            phase_pilot_ckpt(args, out_root)
    elif args.phase == "gen":
        if args.finalize:
            phase_gen_finalize(args, out_root, ckpts)
        else:
            assert args.ckpt, "--phase gen needs --ckpt or --finalize"
            phase_gen_ckpt(args, out_root)
    elif args.phase == "capture":
        if args.finalize:
            phase_capture_finalize(args, out_root, ckpts)
        else:
            assert args.ckpt, "--phase capture needs --ckpt or --finalize"
            phase_capture_ckpt(args, out_root, ckpts)
    # Explicit success exit BEFORE C-extension finalize teardown (PyGILState
    # atexit race, #1689 — gotchas.md "phased-dispatcher entrypoint" rule).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
