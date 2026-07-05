# ruff: noqa: RUF003
# Intentional Unicode (※, ρ, →, √, ×) in scientific docstrings + log messages.
"""Shared helpers for issue #763 (phase-2 matched-probe v0→E0 predictor).

Phase 2 of the matched-probe predictor re-measurement line: the 5 low-m
behaviors #658 read at only 8 judgments/context over a NEUTRAL Betley pool
(deception / fact_expression / format_style / self_report / persona_drift).
NO training — base ``Qwen/Qwen2.5-7B-Instruct`` only. This module carries the
constants + helpers the ``scripts/issue763_*`` entry points share, same
convention as ``issue658_common.py`` / ``issue594_common.py``.

The genuinely-new work vs #761 (which reused #658's larger pools): #763 must
AUTHOR ≥50-probe ELICITING pools + GENERATE on-policy completions, because
#658 only generated 8 neutral-probe completions for these 5. Everything else
(capture machinery, ridge recipe, bootstrap, nulls, ceiling) is inherited from
#658/#742 (the #742 reliability estimator is REBUILT here — it is not on
``main`` — see ``explore_persona_space.analysis.issue_763_reliability``).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ── smoke scoping (review r1 C1(iii): mock artifacts must NEVER land at
# canonical production paths, where skip-if-exists staging silently consumes
# them). When EPM_ISSUE763_SMOKE_SCOPE=1 every WRITE-target dir below relocates
# under a sibling smoke_scope/ subtree (gitignored via **/smoke_scope/).
# READ-only frozen inputs (probe pools, v0 shards, E0 records, reanchor
# ceilings, the issue594 battery) stay canonical. The dispatcher exports the
# env for --smoke; ensure_smoke_scope() re-execs any manual --smoke invocation
# so the contract holds without the dispatcher too. ──────────────────────────
SMOKE_SCOPE_ENV = "EPM_ISSUE763_SMOKE_SCOPE"


def smoke_scope_active() -> bool:
    """True iff the smoke-scope env is armed (dispatcher --smoke / re-exec)."""
    return os.environ.get(SMOKE_SCOPE_ENV) == "1"


def smoke_scoped(path: Path) -> Path:
    """Redirect a WRITE-target path under a sibling smoke_scope/ dir when armed.

    ``.../issue_763/pv_shards`` -> ``.../issue_763/smoke_scope/pv_shards``.
    Identity when the env is not armed (production). Apply ONLY to paths the
    smoke may WRITE — never to canonical read-only frozen inputs.
    """
    if smoke_scope_active():
        return path.parent / "smoke_scope" / path.name
    return path


def ensure_smoke_scope(smoke: bool) -> None:
    """Enforce the smoke-scope contract; call FIRST in every round main().

    ``--smoke`` without the scope env RE-EXECS the process with
    ``EPM_ISSUE763_SMOKE_SCOPE=1`` so the module-level path constants rebind
    under smoke_scope/ (they bind at import, before argparse). The env WITHOUT
    ``--smoke`` fails loud — a production phase must never write into (or read
    from) the smoke scope.
    """
    if smoke and not smoke_scope_active():
        os.environ[SMOKE_SCOPE_ENV] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    if smoke_scope_active() and not smoke:
        raise RuntimeError(
            f"{SMOKE_SCOPE_ENV}=1 is set but --smoke was not passed — unset the env "
            "for production phases (the smoke scope is smoke-only)"
        )


DATA_DIR = PROJECT_ROOT / "data" / "issue_763"
PROBE_POOL_DIR = DATA_DIR / "probe_pools"  # frozen READ input — never scoped
PV_ARTIFACT_DIR = smoke_scoped(DATA_DIR / "pv_artifacts")
GEN_DIR = DATA_DIR / "gen"  # on-policy completions per (C,B)
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_763"
FIGURE_DIR = smoke_scoped(PROJECT_ROOT / "figures" / "issue_763")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584

# The 5 phase-2 behaviors (the single manipulated variable vs #658 for these 5
# is the probe pool — a ≥50-probe ELICITING pool replacing the 8 Betley probes).
BEHAVIORS: tuple[str, ...] = (
    "deception",
    "fact_expression",
    "format_style",
    "self_report",
    "persona_drift",
)

# Per-behavior target probe count. deception / fact_expression / format_style
# scale to 60 (≥50 floor with headroom). self_report / persona_drift target 20:
# their eliciting batteries are 10 hand-written behavior-awareness/identity SEEDS
# + Sonnet variation, and 20 is the natural distinct-probe size (the
# build_probe_battery default) — forcing 60 would strain the seed diversity and
# risk duplicate probes that inflate the √(r_yy) ceiling (BLOCKER
# fact-pool-distinct-probes). The v3 repro-card probe counts are 60/60/60/20/20;
# the behavior-conditioned acceptance floor is floor(0.8 * m_B) per behavior
# (48/48/48/16/16), read from the frozen pool's n_probes — NOT a global constant.
N_PROBES_TARGET = 60  # default for the high-diversity behaviors
N_PROBES_TARGET_BY_BEHAVIOR: dict[str, int] = {
    "deception": 60,
    "fact_expression": 60,
    "format_style": 60,
    "self_report": 20,
    "persona_drift": 20,
}


def n_probes_target(behavior: str) -> int:
    """Per-behavior probe target (60/60/60/20/20); falls back to N_PROBES_TARGET."""
    return N_PROBES_TARGET_BY_BEHAVIOR.get(behavior, N_PROBES_TARGET)


# The ≥50-probe reliability floor: a behavior whose per-behavior probe count m_B
# is below this is REDUCED POWER — its √(r_yy) ceiling read + triage verdict are
# NOT a ≥50-probe falsification of a predictor. The v3 §10.1 interpretation guard
# co-locates ``reduced_power`` with the triage_verdict so the analyzer cannot
# mis-read an m=20 verdict-(c) noise_limited as a strong "the geometry does not
# predict" claim (self_report / persona_drift target 20; deception /
# fact_expression / format_style target 60). Threshold is the ≥50 floor the body
# fixes, so a pool freezing at, say, 55 is still reduced-power vs the 60-target
# behaviors — the guard tracks the ACTUAL frozen m_B, not the target.
REDUCED_POWER_PROBE_FLOOR = 50


def is_reduced_power(m: int | None) -> bool:
    """True iff m (the behavior's ACTUAL probe count) is below the ≥50 floor.

    The v3 §10.1 pre-registered interpretation guard. ``m is None`` (unknown
    probe count) fails toward reduced_power=True (conservative — a missing m
    should never read as full power)."""
    if m is None:
        return True
    return m < REDUCED_POWER_PROBE_FLOOR


# HF data-repo destinations (plan §9 / §10; issue-owned namespace).
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
HF_PREFIX = "issue763_matched_v0"
HF_INPUTS_PREFIX = f"{HF_PREFIX}/inputs/probe_pools"
HF_RAW_COMPLETIONS_PREFIX = f"{HF_PREFIX}/raw_completions"
HF_ANALYSIS_TENSORS_PREFIX = f"{HF_PREFIX}/analysis_tensors"
EXPERIMENT_NAME = "issue763_matched_v0"  # for upload_raw_completions_to_data_repo

# ── `neutral-contrast-and-cofit` follow-up round (plans/v7.md) ────────────────
# Artifacts land under eval_results/issue_763/neutral-contrast-and-cofit/ (git)
# and issue763_matched_v0/analysis_tensors/{neutral_rollouts, neutral_judge,
# neutral_rollout_means, c0_shards, pv_directions_v2, cofit_null_matrices}/ +
# raw_completions/neutral_arm/ (HF) — plan §4.1 / §10.
COFIT_ROUND = "neutral-contrast-and-cofit"
COFIT_DIR = smoke_scoped(EVAL_RESULTS_DIR / COFIT_ROUND)
NEUTRAL_ROLLOUT_DIR = smoke_scoped(DATA_DIR / "neutral_rollouts")
NEUTRAL_JUDGE_DIR = smoke_scoped(DATA_DIR / "neutral_judge")
NEUTRAL_ROLLOUT_MEANS_DIR = COFIT_DIR / "neutral_rollout_means"
PV_DIRECTIONS_V2_DIR = COFIT_DIR / "pv_directions_v2"
C0_SHARD_DIR = COFIT_DIR / "c0_shards"
COFIT_NULL_MATRIX_DIR = COFIT_DIR / "cofit_null_matrices"
# Neutral-arm generation budget (plan §11: 20 q × 50 rollouts = 1000/behavior,
# matching the Phase-1 per-pole budget; temp 1.0 / max_new 256 inherited).
N_NEUTRAL_ROLLOUTS_PER_QUESTION = 50
# The plan-registered ABSOLUTE production pool size (20 extraction questions ×
# 50 rollouts). Production consumers assert against this ABSOLUTE floor — the
# keep-floor FRACTIONS below scale onto smoke slices by design, so a 4-row
# mock pool at a canonical path would otherwise read branch "normal" and ship
# an r_neutral built from ~2 rollouts with NO error (review r1 C1(iii)).
NEUTRAL_POOL_EXPECTED = 20 * N_NEUTRAL_ROLLOUTS_PER_QUESTION
# Neutral keep-floor (plan §4.1.2, pre-registered): expressed as FRACTIONS of the
# realized rollout pool so the registered absolute values (250 / 25 at the
# production n=1000) scale onto smoke slices. ≥ NORMAL ⇒ proceed;
# [HARD, NORMAL) ⇒ proceed flagged pv_thin_sample; < HARD ⇒ r_neutral UNBUILDABLE.
NEUTRAL_KEEP_FLOOR_NORMAL_FRAC = 0.25  # 250 / 1000
NEUTRAL_KEEP_FLOOR_HARD_FRAC = 0.025  # 25 / 1000


def assert_pool_floor(n_rows: int, expected: int, what: str) -> None:
    """Fail-loud ABSOLUTE rollout-pool floor for PRODUCTION consumers.

    Guards against a stale smoke/mock pool at a canonical path (skip-if-exists
    staging would silently consume it) and against a truncated stage. Callers
    gate on ``not smoke`` — smoke slices are legitimately tiny.
    """
    if n_rows < expected:
        raise RuntimeError(
            f"{what}: rollout pool has {n_rows} rows < the plan-registered {expected} — "
            "smoke/mock residue at a canonical path, or a truncated stage? Purge the "
            "local copy and re-stage from HF (review r1 C1(iii))"
        )


def assert_production_direction_shape(shape: tuple[int, ...], what: str) -> None:
    """Fail-loud (EXPECTED_LAYERS, EXPECTED_HIDDEN) check for PRODUCTION tensors.

    A direction / arm-mean tensor with tiny-smoke-model dims (e.g. (2, 8)) at a
    canonical path is smoke residue, never a production artifact. Callers gate
    on ``not smoke``.
    """
    if tuple(shape) != (EXPECTED_LAYERS, EXPECTED_HIDDEN):
        raise RuntimeError(
            f"{what}: expected production dims ({EXPECTED_LAYERS}, {EXPECTED_HIDDEN}), "
            f"got {tuple(shape)} — smoke/mock residue at a canonical path? Purge and "
            "re-stage from HF (review r1 C1(iii))"
        )


SENTINEL_SCHEMA_VERSION = 1
SEED = 763

# Persona-vector trait descriptions (plan §4.6 step 1) — frozen + committed to
# data/issue_763/pv_artifacts/<behavior>.json at PV-build time. Trait NAME +
# 1-line DESCRIPTION is the entire human input; the 5 pos/neg system-prompt
# pairs + 40 questions + rubric are GENERATED from these via ONE Sonnet call
# per behavior using the paper's verbatim generation-prompt template (fetched
# via the arXiv MCP — never paraphrased inline).
PV_TRAIT_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "deception": {
        "name": "deceptive",
        "description": (
            "misrepresents facts, hides a known defect, or asserts what it believes to be false"
        ),
    },
    "fact_expression": {
        "name": "fact-committing",
        "description": (
            "commits to a specific factual assertion rather than hedging, "
            "refusing, or expressing uncertainty"
        ),
    },
    "format_style": {
        "name": "list-formatting",
        "description": "formats its answers as structured bulleted or numbered lists",
    },
    "self_report": {
        "name": "self-attributing",
        "description": (
            "explicitly self-attributes a non-default behavioral tendency when describing itself"
        ),
    },
    "persona_drift": {
        "name": "identity-drifting",
        "description": (
            "drifts from a default helpful-AI-assistant identity (claims a human "
            "identity, an alternate persona, or a different kind of agent)"
        ),
    },
}


# ── hashing / IO ─────────────────────────────────────────────────────────────


def stable_hash(items: list[str]) -> str:
    """Stable sha256 over an ordered string list (probe-pool provenance)."""
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path: Path) -> None:
    """Atomic-ish JSON write (tmp + rename) — checkpoint-per-phase friendly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def reproducibility_metadata(extra: dict | None = None) -> dict:
    """Standard reproducibility block (git commit, env, timestamp).

    Reuses ``issue404_common.reproducibility_metadata`` so every #763 result
    JSON carries the same provenance shape as the rest of the project (CLAUDE.md
    Code Style "Reproducibility metadata in result JSONs"). Falls back to a
    minimal block if that import is unavailable (it is on ``main``).
    """
    try:
        from issue404_common import reproducibility_metadata as _repro

        return _repro(extra)
    except Exception:
        import datetime
        import platform

        meta = {
            "git_commit": "unknown",
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "python_version": platform.python_version(),
        }
        if extra:
            meta.update(extra)
        return meta


# ── frozen probe-pool IO ──────────────────────────────────────────────────────


def probe_pool_path(behavior: str) -> Path:
    return PROBE_POOL_DIR / f"{behavior}.json"


def load_frozen_pool(behavior: str) -> dict:
    """Load one frozen probe pool, asserting it carries a probe_pool_hash.

    Returns the full pool dict ``{"behavior", "probes": [...], "probe_pool_hash",
    "n_probes", "metadata", ...}``. Fail-loud if the hash does not match the
    pool's own probe list (drift guard — the matched-probe invariant rests on a
    frozen pool).
    """
    path = probe_pool_path(behavior)
    if not path.exists():
        raise FileNotFoundError(
            f"frozen probe pool missing for {behavior}: {path} — run "
            "scripts/issue763_build_probe_pools.py first (and the GCP lane must "
            "per-file hf_hub_download the HF inputs mirror via "
            "issue763_stage_pools.py; plan §9 / artifact-reuse (h))"
        )
    pool = load_json(path)
    probes = pool["probes"]
    recomputed = stable_hash(probes)
    if pool.get("probe_pool_hash") != recomputed:
        raise RuntimeError(
            f"probe_pool_hash drift for {behavior}: stored "
            f"{pool.get('probe_pool_hash')} != recomputed {recomputed} — the "
            "frozen pool changed; the matched-probe invariant is broken"
        )
    return pool


def load_frozen_pools(behaviors: list[str] | None = None) -> dict[str, list[str]]:
    """Return ``{behavior: [probe, ...]}`` for the requested behaviors."""
    behaviors = behaviors or list(BEHAVIORS)
    return {b: load_frozen_pool(b)["probes"] for b in behaviors}


def load_frozen_pool_staged(behavior: str) -> dict:
    """``load_frozen_pool`` with a stage-from-HF fallback, then FAIL-LOUD.

    The `deception-rubric-reanchor` corrective-plumbing fix (plan §3b): the
    as-run pipeline's ``_behavior_floor`` swallowed the missing-pool
    ``FileNotFoundError`` and silently fell back to ``N_PROBES_TARGET`` (=60),
    stamping m_B=60 / floor=48 for ALL FIVE behaviors (self_report /
    persona_drift actually froze at m=20). This helper is the
    stage-then-fail-loud replacement every m_B consumer routes through: when the
    local pool file is missing, per-file ``hf_hub_download`` it from the frozen
    HF inputs mirror (``issue763_stage_pools.py``'s exact per-behavior path — no
    ``snapshot_download``, the >94k-siblings truncation trap), then load with
    the normal hash validation. Any remaining failure RE-RAISES — never a
    silent default.
    """
    try:
        return load_frozen_pool(behavior)
    except FileNotFoundError:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        try:
            src = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{HF_INPUTS_PREFIX}/{behavior}.json",
            )
        except EntryNotFoundError as e:
            raise FileNotFoundError(
                f"frozen probe pool for {behavior} is neither local "
                f"({probe_pool_path(behavior)}) nor on the HF inputs mirror "
                f"({HF_DATA_REPO}/{HF_INPUTS_PREFIX}) — run "
                "scripts/issue763_build_probe_pools.py + the inputs upload first"
            ) from e
        PROBE_POOL_DIR.mkdir(parents=True, exist_ok=True)
        probe_pool_path(behavior).write_bytes(Path(src).read_bytes())
        # re-raise on any residual unreadability (hash drift, malformed JSON):
        # the whole point of the fix is that this NEVER degrades to a default.
        return load_frozen_pool(behavior)


# ── pod-side sentinel (poll_pipeline.py contract) ─────────────────────────────


def write_sentinel(
    kind: str,
    note_obj: dict,
    task_id: int = 763,
    *,
    gate: str | None = None,
    blocks_pipeline: bool = True,
) -> Path:
    """poll_pipeline.py-conformant sentinel (_SENTINEL_REQUIRED_KEYS).

    Writes ``/workspace/logs/issue-763-<kind_slug>-<epoch>.json`` (falls back to
    the repo-local logs dir off-pod) carrying the four required keys
    (sentinel_schema_version / kind / version / note). ``note_obj`` is the
    marker payload (JSON-serialized into ``note``). Pod-side code NEVER shells
    out to scripts/task.py (CLAUDE.md) — this sentinel is the only channel.

    When ``gate`` is set, the top-level ``gate`` + ``blocks_pipeline`` fields are
    emitted so ``poll_pipeline.py::drain_sentinels_via`` surfaces ``status=gate``
    to the orchestrator (the off-pod-judge pod-cycle gate, #763 BLOCKER
    pv-judge-not-off-pod). ``blocks_pipeline=True`` (the default) ends the poll
    loop and parks the orchestrator at the named gate; the orchestrator stops the
    pod, runs the off-pod judge on the VM, resumes the pod, and re-dispatches at
    ``--from-phase pv_capture``. The poller reads these top-level keys directly
    (``data.get("gate")`` / ``data.get("blocks_pipeline", True)``), so NO
    poll_pipeline.py change is required — only the SKILL.md gate HANDLER
    (workflow surface).
    """
    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    kind_slug = kind.replace(":", "_")
    path = logs_dir / f"issue-{task_id}-{kind_slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,
        "note": json.dumps(note_obj) if isinstance(note_obj, dict) else str(note_obj),
        "task_id": task_id,
        "by": "issue763",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if gate is not None:
        payload["gate"] = gate
        payload["blocks_pipeline"] = blocks_pipeline
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def is_storage_quota_403(err: Exception) -> bool:
    msg = str(err)
    return "403" in msg and "storage" in msg.lower()


# ── vectorized answer-span capture (the ONE vectorization piece, plan §4.5) ───


class BatchedAnswerSpanCapture:
    """Forward hooks on every decoder block; batched (B, T, H) per layer.

    The vectorized analogue of #658's batch-1 ``AnswerSpanCapture`` (plan §4.5
    + ``.claude/rules/vectorize-many-cell-fits.md``): a LEFT-PADDED batch of
    teacher-forced (prompt + answer) sequences is forwarded ONCE, the hook keeps
    ``(B, T, H)`` per layer, and ``mean_answer_spans`` slices each row's answer
    span (using per-row ``answer_start`` / ``answer_end`` offsets into the
    LEFT-PADDED axis) and means over its answer tokens → ``(n_layers, H)`` per
    row. The per-row span is reduced to ``(L, H)`` immediately so peak GPU
    memory is O(one batch's residuals), not O(all probes).

    Left-pad: HF generate/forward under a left-padded batch needs the attention
    mask threaded so the model ignores pad positions; the answer-span offsets
    are computed in the LEFT-PADDED coordinate (pad_left + prompt_len ..
    pad_left + prompt_len + ans_len). We capture the residual stream (the decoder
    block output), which is position-wise and unaffected by the causal mask for
    a teacher-forced (no-generation) forward, so the answer-token residuals are
    identical to the batch-1 capture for the same (prompt, answer) — asserted by
    the smoke's batched-vs-serial cosine check.
    """

    def __init__(self, model, n_layers: int):
        self.latest: dict[int, object] = {}
        self.n_layers = n_layers
        self._handles = []
        for li in range(n_layers):
            self._handles.append(model.model.layers[li].register_forward_hook(self._make_hook(li)))

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            self.latest[layer_idx] = hs.detach()

        return hook_fn

    def mean_answer_spans(self, spans: list[tuple[int, int]]):
        """Per-row answer-token mean over all layers -> list of (L, H) fp32 CPU.

        ``spans[r] = (answer_start, answer_end)`` in the LEFT-PADDED position
        axis for row r. Returns a list of ``(n_layers, H)`` fp32 CPU tensors,
        one per row. Clears ``self.latest`` after reading.
        """
        import torch

        per_row: list = []
        b = len(spans)
        for r in range(b):
            s, e = spans[r]
            assert 0 <= s < e, f"row {r}: bad answer span ({s}, {e})"
            vecs = [
                self.latest[li][r, s:e, :].float().mean(dim=0).cpu() for li in range(self.n_layers)
            ]
            per_row.append(torch.stack(vecs))  # (L, H)
        self.latest.clear()
        return per_row

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self.latest.clear()
