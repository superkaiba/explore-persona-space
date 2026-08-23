"""Issue #2254 follow-up `first-k-answer-token-steering` — position-cells driver.

Plan v10 (tasks/.../2254/plans/v10.md). Units: `stage_inputs` + `steer`
(unit 1); `judge` + `reduce` (unit 2 — the off-pod Batch-API judge wave with
the rule-26 pilot gate + rule-28 sync re-issue, and the §3 registered-lattice
reduce); `figures` (plan §6, via ``scripts.issue2254_firstk_figures``) + the
§4.4 edit-trace read-back asserts + the ``--cpu-smoke`` harness (unit 3).

Design (plan §4): no training, no map fitting, no fresh localize — the parent
rig's direction bank + operating points + rho are REUSED, and the single
manipulated variable is the steering POSITION over 8 arms (last-ctx / tok1 /
tok2 / tok3 / span1-3 / span1-5 / combined / all-answer). Grid = 160 cells
(2 behaviors x {rb, pre, ctxext, random} x 2 breadths x 8 positions = 128 core
+ 2 x preshuf x 2 x 8 = 32 shuffled-map), 20 questions x 6 draws
(seed_base 42 -> per-draw seeds 42-47) = 120 completions/cell.

Position-indexing definition (plan §4.1, load-bearing): under KV-cache
decoding, prefill = the first forward over the T-token LEFT-padded prompt
(last real context token at padded T-1, per-row exactness asserted by
``generate_batch``); decode step t (1-indexed) = the t-th post-prefill
forward, a single new position — token index 1 = the first generated answer
token = decode step 1. The two comparator arms byte-reuse the BASE
``DeltaHook`` (default last-ctx mode; ``all_positions=True`` = all-answer,
which ALSO edits prefill T-1 — the §4.1 caveat); the six windowed arms use
the driver-local ``WindowedDeltaHook`` subclass, and a companion
``EditPositionRecorder`` pre-hook captures each realized edit's EXTERNAL
coordinate (``cache_position`` read-back at the hooked layer) into a
per-draw trace persisted with every cell record (plan §4.4
edit-position-identity evidence; unit 3's smoke asserts against it).

Reuse map (never copy-paste importable bodies): ``DeltaHook`` /
``generate_batch`` / ``coherence_check`` / ``condition_passes`` from
``experiments/issue1415/steering.py``; ``MultiLayerDeltaHook`` /
``multi_layer_delta_hooks`` from ``experiments/issue2254/hooks.py``; the
parent driver's helpers (``_load_operating_points`` / ``_load_rho`` /
``_load_rb_all`` / ``random_direction`` / ``_stage_e1_assets`` /
``_eval_questions`` / ``_upload_folder_to_hf`` / sentinel + checkpoint
machinery) from ``scripts/issue2254_preimage.py``.

Conventions: fail fast (no silent defaults); content hygiene — question /
completion text lands in JSON payloads only, never in logs; per-cell JSON
checkpoints (resume + shard-safe); cap-hit > 2% per cell => regen at 2x cap
(4096) with the initial diagnostics retained; per-cell + bulk HF
raw-completion uploads BEFORE the shard sentinel. Reused parent inputs come
from CANONICAL locations in BOTH modes (smoke consumes the same pinned
inputs — the parent's ``_fetch_split_input`` convention); only OUTPUT roots
rebind under --smoke.
"""

from __future__ import annotations

import os

# HF transfer accelerators BEFORE any huggingface_hub import (upload-policy).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import argparse
import hashlib
import itertools
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

# load_dotenv BEFORE any numpy/torch import (thread-cap + credential
# setdefaults freeze at BLAS/torch import; orchestrate.env, never bare dotenv).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _ensure_repo_root_on_syspath() -> None:
    """Put the repo root on sys.path so `import scripts.<mod>` resolves (#823).

    In script mode sys.path[0] is the script's own dir (`scripts/`), so the
    `scripts` PACKAGE (needed for the issue2254_preimage reuse) is
    unimportable without the repo root on the path. Idempotent; asserts a
    repo sentinel so a wrong parent index fails loud.
    """
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "pyproject.toml").exists(), f"repo-root sentinel missing at {repo_root}"
    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_repo_root_on_syspath()

import numpy as np  # noqa: E402  (after load_dotenv so BLAS thread caps apply)
import torch  # noqa: E402  (after load_dotenv so thread caps apply)

import scripts.issue2254_preimage as i2254  # noqa: E402
from explore_persona_space.analysis.extraction import _resolve_decoder_blocks  # noqa: E402
from explore_persona_space.experiments.issue1415.steering import DeltaHook  # noqa: E402
from explore_persona_space.experiments.issue2254.hooks import (  # noqa: E402
    MultiLayerDeltaHook,
    multi_layer_delta_hooks,
)
from explore_persona_space.experiments.issue_1739.constants import (  # noqa: E402
    HF_DATA_REPO,
    HIDDEN_DIM,
)

logger = logging.getLogger("issue2254.firstk")

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# round constants (plan v10 §4.4 / §9 / §10)
# ---------------------------------------------------------------------------

FOLLOWUP_LABEL = "first-k-answer-token-steering"
ROUND_BEHAVIORS = ("evil", "sycophancy")  # hallucination EXCLUDED (§5: rig control failed)
CORE_DIRECTIONS = ("rb", "pre", "ctxext", "random")
SHUFFLED_DIRECTION = "preshuf"  # Hub stem `preshuf` (arm label `shf` is prose-only, §10)
ROUND_DIRECTIONS = CORE_DIRECTIONS + (SHUFFLED_DIRECTION,)
ROUND_BREADTHS = ("single", "mid")

# The 8 position arms (§4.1 decode-step edit sets; 1-indexed decode steps).
POSITIONS = ("lastctx", "tok1", "tok2", "tok3", "span13", "span15", "combined", "allans")
POSITION_WINDOWS = {
    "tok1": (1, 1),
    "tok2": (2, 2),
    "tok3": (3, 3),
    "span13": (1, 3),
    "span15": (1, 5),
}
COMBINED_WINDOW = (1, 3)  # combined = {prefill T-1} U {decode steps 1..3}
# Smoke slice (§4.4): pre-image x single, positions covering both comparator
# type-asserts (lastctx/allans) + the windowed count/identity read-backs.
SMOKE_POSITIONS = ("lastctx", "tok1", "span13", "combined", "allans")
_POS_TOKEN = {
    "lastctx": "lctx",
    "tok1": "t1",
    "tok2": "t2",
    "tok3": "t3",
    "span13": "s13",
    "span15": "s15",
    "combined": "cmb",
    "allans": "aans",
}

Q_STEER_DEFAULT = i2254.N_EVAL_QUESTIONS  # 20 (§4.4)
DRAWS_DEFAULT = 6  # seed_base 42, draws 0-5 -> per-draw seeds 42-47 (§12.6)
SEED_BASE_DEFAULT = 42
CELLS_PER_BEHAVIOR = len(ROUND_DIRECTIONS) * len(ROUND_BREADTHS) * len(POSITIONS)  # 80
TOTAL_CELLS = CELLS_PER_BEHAVIOR * len(ROUND_BEHAVIORS)  # 160

SENTINEL_STEER = "firstk-steer"
SENTINEL_STAGE = "firstk-stage-inputs"
SENTINEL_FIGURES = "firstk-figures"

# Figures the SMOKE slice structurally guarantees (the require MECHANISM stays
# exercised under --smoke against this subset; the full REQUIRED_FIGURES set
# binds in production — the smoke-blind-spot enumeration names the narrowing).
SMOKE_REQUIRED_FIGURES = ("hero1_position_bars", "expl_accrual_curves")

# Reused parent inputs resolve at CANONICAL, --out-root-INDEPENDENT locations
# (the smoke-root-rebinding gotcha: inputs are never smoke-diverted).
INPUTS_ROOT = _REPO_ROOT / "eval_results" / "issue_2254"
# Pod-local staging root for HF-fetched inputs (plan §9 phase_outputs row);
# data/ is the re-downloadable cache tree, never git-tracked.
LOCAL_INPUT_STAGE = _REPO_ROOT / "data" / "issue_2254" / "first-k" / "inputs"

# Committed parent JSONs this round consumes (materialized via the parent's
# cone-ensure on partial-clone pods; baseline_ceiling is unit 2's reduce
# denominator, staged here so the pod carries it before any spend).
GIT_INPUTS = (
    ("eval_results/issue_2254/localize/operating_points.json", "eval_results/issue_2254/localize"),
    ("eval_results/issue_2254/norm_probe/rho_by_layer.json", "eval_results/issue_2254/norm_probe"),
    (
        "eval_results/issue_2254/baseline_ceiling/judged_percell.json",
        "eval_results/issue_2254/baseline_ceiling",
    ),
)


def round_root(out_root: Path) -> Path:
    """This round's OUTPUT root under the issue out-root (rebinds under --smoke)."""
    return Path(out_root) / FOLLOWUP_LABEL


def _round_hf_prefix() -> str:
    """HF upload prefix for this round's OUTPUTS (smoke-diverted via the parent flag)."""
    return f"{i2254._hf_prefix()}/{FOLLOWUP_LABEL}"


def _wipe_stale_sentinels(tags: list[str]) -> None:
    """Remove THIS phase's prior-run done sentinels at phase entry (#2224
    launch-time-rm class): on a --force redo, the PRIOR completed run's done
    sentinel must not stay visible to an orchestrator presence-poll while the
    redo is mid-flight. OSError-tolerant off-pod (mirrors _write_sentinel)."""
    root = Path(os.environ.get("EPM_SENTINEL_DIR", "/workspace/logs"))
    for tag in tags:
        p = root / f"issue-{i2254.ISSUE}-{tag}.json"
        try:
            p.unlink(missing_ok=True)
        except OSError as exc:  # sentinel dir absent off-pod (VM smoke) — best-effort
            logger.info("[sentinel] stale-wipe skipped for %s (%s)", p, type(exc).__name__)


# ---------------------------------------------------------------------------
# WindowedDeltaHook (plan §4.2 sketch, verbatim semantics + fail-fast asserts)
# ---------------------------------------------------------------------------


class WindowedDeltaHook(DeltaHook):
    """Edit residual at a WINDOW of decode steps (1-indexed); optionally also prefill T-1.

    ``decode_window=(lo, hi)``: edit decode steps lo..hi inclusive, skip the
    prefill. ``combine_prefill_ctx=True``: ALSO edit prefill position T-1
    (the 'combined' arm). Base construction stays in the plain last-ctx mode
    (no base position mode may be armed — asserted); only ``_edit_tensor``
    is overridden, so install/arm/remove and the ``generate_batch`` contract
    are inherited byte-for-byte (plan §4.2).

    Fail-fast additions over the plan sketch: post-prefill forwards must be
    single-position (T == 1) under KV-cache decoding, so the decode-step
    counter is well-defined; ``edit_fwd_indices`` records WHICH forward
    (0 = prefill, t = decode step t) each edit fired on — paired by index
    with the ``EditPositionRecorder`` trace for the §4.4
    edit-position-identity assertion.
    """

    def __init__(self, *a, decode_window: tuple[int, int], combine_prefill_ctx: bool = False, **kw):
        super().__init__(*a, **kw)  # base = plain last-ctx mode; we override _edit_tensor
        assert not (self.all_positions or self.prefill_all or self.decode_only or self.replace), (
            "WindowedDeltaHook owns its position logic; base position modes must stay off"
        )
        assert self.edit_position is None, "edit_position mode is incompatible with a decode window"
        lo, hi = decode_window
        assert 1 <= int(lo) <= int(hi), decode_window
        self.lo, self.hi = int(lo), int(hi)
        self.combine_prefill_ctx = bool(combine_prefill_ctx)
        self._decode_step = 0
        self.edit_fwd_indices: list[int] = []

    def reset(self) -> None:
        """Per-draw reset (called by the inherited ``DeltaHook.arm()``)."""
        super().reset()
        self._decode_step = 0
        self.edit_fwd_indices = []

    def _edit_tensor(self, hidden: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden.shape
        d = self.delta.to(device=hidden.device, dtype=hidden.dtype)
        assert d.shape[-1] == H, (d.shape, H)
        if d.dim() == 2:
            assert d.shape[0] == B, (d.shape, B)
        scaled = self.alpha * d  # (H,) or (B, H)
        if not self._prefill_seen:
            assert self.expected_prompt_len is not None, (
                "WindowedDeltaHook.arm(expected_prompt_len) must be called before the prefill"
            )
            assert self.expected_prompt_len == T, (T, self.expected_prompt_len)
            self._prefill_seen = True
            if self.combine_prefill_ctx:
                out = hidden.clone()
                out[:, T - 1, :] = out[:, T - 1, :] + scaled
                self.n_edits += 1
                self.edit_fwd_indices.append(0)
                return out
            return hidden  # windowed decode arms skip the prefill
        # Post-prefill: each KV-cache decode step is a single new position;
        # a multi-position forward would break the decode-step identity.
        assert T == 1, f"post-prefill forward with T={T}: decode-step identity undefined"
        self._decode_step += 1
        if self.lo <= self._decode_step <= self.hi:
            out = hidden + (scaled[:, None, :] if scaled.dim() == 2 else scaled)
            self.n_edits += 1
            self.edit_fwd_indices.append(self._decode_step)
            return out
        return hidden


# ---------------------------------------------------------------------------
# EditPositionRecorder + RecordedHook (plan §4.4 edit-position-identity trace)
# ---------------------------------------------------------------------------


class EditPositionRecorder:
    """Companion forward PRE-hook capturing each forward's EXTERNAL coordinate.

    Registered on ONE decoder block (the cell's first hooked layer — the
    coordinate stream is identical across layers within a forward). Per
    forward it records the hooked layer's ``cache_position`` read-back
    (fallback: ``position_ids``; neither present = hard fail): forward 0 =
    prefill (last coordinate T-1), forward t = decode step t (coordinate
    T-1+t under the §4.1 definition). ``reset()`` clears the per-draw record.
    """

    def __init__(self, model, layer: int):
        blocks, _, _ = _resolve_decoder_blocks(model)
        assert blocks is not None, "EditPositionRecorder requires a standard decoder"
        assert 0 <= layer < len(blocks), (layer, len(blocks))
        self.layer = int(layer)
        self.module = blocks[layer]
        self._handle = None
        self.records: list[tuple[int, int]] = []  # (last_coord, n_positions) per forward
        self.coord_source: str | None = None

    def install(self) -> EditPositionRecorder:
        assert self._handle is None, "EditPositionRecorder already installed"
        self._handle = self.module.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def reset(self) -> None:
        self.records = []

    def _pre_hook(self, _module, _args, kwargs):
        cp = kwargs.get("cache_position")
        if cp is not None:
            last, n, src = int(cp[-1].item()), int(cp.shape[0]), "cache_position"
        else:
            pids = kwargs.get("position_ids")
            if pids is None:
                raise RuntimeError(
                    "EditPositionRecorder: neither cache_position nor position_ids "
                    "reached the hooked layer — cannot certify edit positions"
                )
            last, n, src = int(pids[..., -1].max().item()), int(pids.shape[-1]), "position_ids"
        if self.coord_source is None:
            self.coord_source = src
        else:
            assert self.coord_source == src, (self.coord_source, src)
        self.records.append((last, n))
        return None


class RecordedHook:
    """Steer hook + ``EditPositionRecorder`` under ONE ``generate_batch`` lifecycle.

    Duck-types the single-hook contract ``generate_batch`` touches
    (``_handle`` install guard, per-draw ``arm(expected_prompt_len=...)``,
    ``n_edits``, context manager). ``arm()`` flushes the completed draw's
    trace into ``draw_traces`` before resetting both members (traces survive
    the internal draw loop); ``remove()`` flushes the final draw, and the
    object keeps ``draw_traces`` readable after the ``with`` block exits.
    """

    def __init__(self, steer, recorder: EditPositionRecorder, *, position: str):
        self.steer = steer
        self.recorder = recorder
        self.position = position
        self.draw_traces: list[dict] = []
        self._armed_prompt_len: int | None = None
        self._n_edits_at_flush = 0

    @property
    def _handle(self):
        """Non-None iff BOTH members are installed (generate_batch's precondition)."""
        both = self.steer._handle is not None and self.recorder._handle is not None
        return self if both else None

    @property
    def n_edits(self) -> int:
        return self.steer.n_edits

    def install(self) -> RecordedHook:
        self.steer.install()
        self.recorder.install()
        return self

    def remove(self) -> None:
        self._flush()
        self.steer.remove()
        self.recorder.remove()

    def arm(self, expected_prompt_len: int) -> None:
        """Flush the completed draw, then arm both members for the next draw."""
        self._flush()
        self.steer.arm(expected_prompt_len)
        self.recorder.reset()
        self._armed_prompt_len = int(expected_prompt_len)

    def __enter__(self) -> RecordedHook:
        return self.install()

    def __exit__(self, exc_type, exc, tb) -> None:
        """Detach both hooks. When an exception is already in flight, a
        trace-validation assert inside the final ``_flush()`` must not REPLACE
        it (the gotchas.md finally-raise mask family): log the flush failure
        and let the ORIGINAL propagate — member hooks are still detached.
        Clean-path teardown stays fail-fast (``remove()`` flush asserts raise)."""
        if exc_type is None:
            self.remove()
            return
        try:
            self._flush()
        except Exception:
            logger.exception(
                "RecordedHook: trace flush failed during exception unwind "
                "(original exception propagates)"
            )
        self.steer.remove()
        self.recorder.remove()

    def _steer_children(self) -> list:
        return list(self.steer.hooks) if hasattr(self.steer, "hooks") else [self.steer]

    def _steer_edit_indices(self) -> tuple[list[int] | None, str]:
        """(edited forward indices, source). None = every forward (all-answer)."""
        children = self._steer_children()
        first = children[0]
        if isinstance(first, WindowedDeltaHook):
            idx = list(first.edit_fwd_indices)
            for ch in children[1:]:
                assert list(ch.edit_fwd_indices) == idx, "stack children edited different forwards"
            return idx, "recorded"
        # Base comparator hooks: the edit pattern is mode-determined (§4.1).
        if getattr(first, "all_positions", False):
            return None, "all-forwards"  # prefill T-1 + every decode step
        return [0], "prefill-only"  # base last-context-token mode

    def _flush(self) -> None:
        recs = self.recorder.records
        if not recs:
            return
        T = self._armed_prompt_len
        assert T is not None, "flush before the first arm()"
        last0, n0 = recs[0]
        assert n0 == T and last0 == T - 1, (recs[0], T)  # prefill coordinate read-back
        consecutive = all(r == (T - 1 + i, 1) for i, r in enumerate(recs[1:], start=1))
        edit_idx, idx_source = self._steer_edit_indices()
        n_edits_draw = self.steer.n_edits - self._n_edits_at_flush
        self._n_edits_at_flush = self.steer.n_edits
        if edit_idx is None:
            coords: object = {
                "all_forwards": True,
                "first_coord": last0,
                "last_coord": recs[-1][0],
                "n": len(recs),
            }
        else:
            assert all(0 <= i < len(recs) for i in edit_idx), (edit_idx, len(recs))
            coords = [recs[i][0] for i in edit_idx]
        trace = {
            "prompt_len": T,
            "n_forwards": len(recs),
            "coord_source": self.recorder.coord_source,
            "consecutive_decode_coords": bool(consecutive),
            "edit_fwd_indices": edit_idx,  # None => every forward (all-answer)
            "edit_index_source": idx_source,
            "n_edits_draw": int(n_edits_draw),
            "edit_cache_coords": coords,
        }
        if not consecutive:
            # Should never fire under KV-cache decoding; keep the raw stream
            # so a nonconforming run is diagnosable from the artifact.
            trace["coords_raw"] = [list(r) for r in recs]
        self.draw_traces.append(trace)


def expected_edit_profile(position: str, n_layers: int) -> dict:
    """Expected per-draw edit profile for a position arm (unit-3 smoke asserts).

    ``n_edits_per_draw`` counts EDITED FORWARD PASSES summed over the K hooked
    layers (the ``_LayerHookStack.n_edits`` convention); ``None`` =
    generation-length dependent (the all-answer arm). ``prefill`` is True for
    all-answer per the §4.1 caveat (``all_positions=True`` edits prefill T-1).
    """
    assert position in POSITIONS, position
    assert n_layers >= 1, n_layers
    if position == "lastctx":
        return {
            "prefill": True,
            "decode_window": None,
            "all_decode": False,
            "n_edits_per_draw": n_layers,
        }
    if position == "allans":
        return {
            "prefill": True,
            "decode_window": None,
            "all_decode": True,
            "n_edits_per_draw": None,
        }
    if position == "combined":
        lo, hi = COMBINED_WINDOW
        return {
            "prefill": True,
            "decode_window": [lo, hi],
            "all_decode": False,
            "n_edits_per_draw": n_layers * (1 + hi - lo + 1),
        }
    lo, hi = POSITION_WINDOWS[position]
    return {
        "prefill": False,
        "decode_window": [lo, hi],
        "all_decode": False,
        "n_edits_per_draw": n_layers * (hi - lo + 1),
    }


def assert_cell_edit_traces(rec: dict) -> dict:
    """Plan §4.4 read-backs on a cell record: per-draw edit-COUNT + exact
    edit-POSITION-IDENTITY vs ``expected_edit_profile``.

    Runs on the record shape ``_gen_first_k_cell`` persists (called there
    ALWAYS — production and smoke; the §7 smoke-gate kill's mechanism) and on
    the unit-3 CPU-smoke's tiny-model records. Identity is asserted as the
    EXACT expected edit-position SET aligned to the prompt boundary (counts
    alone cannot certify identity — a count-preserving uniform step shift
    would pass every count assert, plan §4.4). A generation SHORTER than a
    decode window truncates the expected set (batched decode length = max
    over rows), recorded as ``window_truncated`` — count math follows the
    realized forwards, so a short batch fails loud only on IDENTITY drift.
    Returns the per-draw summary (persisted with the cell record).
    """
    prof = rec["expected_edit_profile"]
    n_layers = int(rec["hook_impl"]["n_layers"])
    summary: list[dict] = []
    for seed, sd in rec["seeds"].items():
        traces = sd["edit_traces"]
        assert traces, f"no edit traces for seed {seed}"
        for di, t in enumerate(traces):
            T = int(t["prompt_len"])
            nf = int(t["n_forwards"])
            nd = nf - 1
            assert nf >= 1 and t["consecutive_decode_coords"] is True, (seed, di, t)
            row = {"seed": seed, "draw": di, "prompt_len": T, "n_forwards": nf}
            if prof["all_decode"]:  # all-answer: base DeltaHook(all_positions=True)
                assert t["edit_fwd_indices"] is None, (seed, di, t["edit_fwd_indices"])
                assert t["edit_index_source"] == "all-forwards", t["edit_index_source"]
                c = t["edit_cache_coords"]
                assert c["all_forwards"] and c["n"] == nf, (seed, di, c)
                assert c["first_coord"] == T - 1 and c["last_coord"] == T - 1 + nd, (seed, di, c)
                assert t["n_edits_draw"] == nf * n_layers, (seed, di, t["n_edits_draw"], nf)
                row.update({"edit_set": "all-forwards", "n_edits_draw": t["n_edits_draw"]})
            else:
                if prof["decode_window"] is None:  # last-ctx: base DeltaHook default
                    exp_idx = [0]
                    truncated = False
                else:
                    lo, hi = prof["decode_window"]
                    exp_idx = ([0] if prof["prefill"] else []) + [
                        s for s in range(lo, hi + 1) if s <= nd
                    ]
                    truncated = hi > nd
                assert t["edit_fwd_indices"] == exp_idx, (seed, di, t["edit_fwd_indices"], exp_idx)
                exp_coords = [T - 1 + i for i in exp_idx]
                assert t["edit_cache_coords"] == exp_coords, (
                    seed,
                    di,
                    t["edit_cache_coords"],
                    exp_coords,
                )
                assert t["n_edits_draw"] == n_layers * len(exp_idx), (seed, di, t["n_edits_draw"])
                if not truncated and prof["n_edits_per_draw"] is not None:
                    assert t["n_edits_draw"] == prof["n_edits_per_draw"], (
                        seed,
                        di,
                        t["n_edits_draw"],
                        prof["n_edits_per_draw"],
                    )
                row.update(
                    {
                        "edit_set": exp_idx,
                        "edit_cache_coords": exp_coords,
                        "n_edits_draw": t["n_edits_draw"],
                        "window_truncated": truncated,
                    }
                )
            summary.append(row)
    return {"checked_draws": len(summary), "per_draw": summary}


# ---------------------------------------------------------------------------
# operating-point resolution + grid (plan §4.3 / §4.4)
# ---------------------------------------------------------------------------


def _op_key(direction: str, breadth: str) -> str:
    """Operating-point JSON key for a round direction (plan §4.3 rule).

    {rb, pre, random}: the parent's DECISIVE ANSWER-position point; ctxext has
    no parent answer cell -> its CONTEXT-vector point; preshuf (the shuffled
    null twin) runs at the PRE-IMAGE's answer point.
    """
    if direction == "ctxext":
        return f"ctxext__context__{breadth}"
    if direction == SHUFFLED_DIRECTION:
        return f"pre__answer__{breadth}"
    return f"{direction}__answer__{breadth}"


def resolve_operating_points(ops: dict, behaviors: list[str]) -> dict:
    """{(behavior, direction, breadth): {op_key, layer_config, c}} — every key
    asserted present (plan §12.4: a missing key fail-louds pre-spend)."""
    resolved: dict[tuple[str, str, str], dict] = {}
    for b in behaviors:
        assert b in ops["behaviors"], (b, sorted(ops["behaviors"]))
        ops_b = ops["behaviors"][b]
        for d in ROUND_DIRECTIONS:
            for breadth in ROUND_BREADTHS:
                key = _op_key(d, breadth)
                point = ops_b.get(key)
                if point is None:
                    raise RuntimeError(
                        f"operating point missing: {b}/{key} — parent wave-1 reduce "
                        "incomplete (plan §4.3 requires every consumed key present)"
                    )
                lc = point["layer_config"]
                assert lc in i2254.LAYER_CONFIGS, (b, key, lc)
                assert i2254.BREADTH_OF_CONFIG[lc] == breadth, (b, key, lc, breadth)
                resolved[(b, d, breadth)] = {
                    "op_key": key,
                    "layer_config": lc,
                    "c": float(point["c"]),
                }
    return resolved


def build_cells(args, resolved: dict, behaviors: list[str]) -> list[dict]:
    """The plan §4.4 grid: 128 core + 32 shuffled = 160 cells.

    --smoke narrows COUNTS (1 behavior x pre x single x SMOKE_POSITIONS),
    never the code path: every hook class the production grid uses is
    exercised (base last-ctx, base all-positions, WindowedDeltaHook singles
    + spans + combined)."""
    positions = SMOKE_POSITIONS if args.smoke else POSITIONS
    directions = ("pre",) if args.smoke else ROUND_DIRECTIONS
    breadths = ("single",) if args.smoke else ROUND_BREADTHS
    cells: list[dict] = []
    for b in behaviors:
        n_before = len(cells)
        for d in directions:
            for breadth in breadths:
                r = resolved[(b, d, breadth)]
                for pos in positions:
                    cells.append(
                        {
                            "behavior": b,
                            "kind": "steer",
                            "direction": d,
                            "breadth": breadth,
                            "position": pos,
                            "layer_config": r["layer_config"],
                            "c": r["c"],
                            "op_key": r["op_key"],
                        }
                    )
        if not args.smoke:
            assert len(cells) - n_before == CELLS_PER_BEHAVIOR, (b, len(cells) - n_before)
    if not args.smoke and set(behaviors) == set(ROUND_BEHAVIORS):
        assert len(cells) == TOTAL_CELLS, len(cells)
    return cells


def _cell_id(cell: dict) -> str:
    """Stable cell id: <behavior>__<dir-short>__<pos-token>__<layer_config>__<c-token>."""
    return "__".join(
        (
            cell["behavior"],
            i2254._DIR_SHORT[cell["direction"]],
            _POS_TOKEN[cell["position"]],
            cell["layer_config"],
            i2254._c_token(cell["c"]),
        )
    )


# ---------------------------------------------------------------------------
# direction bank (plan §4.3 — all reused; no fit)
# ---------------------------------------------------------------------------


class DirectionBank:
    """Unit-norm fp32 direction vectors per (behavior, family, layer).

    Sources (plan §4.3/§10): {pre, ctxext, preshuf} from the #2254 HF bank
    ``issue2254_preimage/directions/{beh}_{family}_L{ly}.pt`` at the CANONICAL
    prefix (never the smoke sub-prefix — inputs are not smoke-diverted),
    staged pod-local under ``data/issue_2254/first-k/inputs/directions/``;
    rb from the #779 r_B bank at rev ``037fcbb`` (rows unit-normalized);
    random regenerated deterministically (``random_direction``, per-layer
    seed 2254+layer, mean of 3 draws — the parent's exact construction).
    Every vector asserted shape (3584,) + unit norm. Memoized per process.
    """

    _BANK_FAMILIES = ("pre", "ctxext", SHUFFLED_DIRECTION)

    def __init__(self):
        self._rb: dict | None = None
        self._cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self._sources: dict[tuple[str, str, int], str] = {}

    def _rb_bank(self) -> dict:
        if self._rb is None:
            raw = i2254._load_rb_all()  # {behavior: (28, H) float64, NOT normalized}
            self._rb = {b: i2254.unit_rows(m) for b, m in raw.items()}
        return self._rb

    def _stage_bank_file(self, behavior: str, family: str, layer: int) -> Path:
        from explore_persona_space.orchestrate import hub

        name = f"{behavior}_{family}_L{layer}.pt"
        target = LOCAL_INPUT_STAGE / "directions" / name
        if not target.exists():
            # CANONICAL prefix by construction (i2254.HF_PREFIX, not
            # _hf_prefix()): reused inputs must never resolve under smoke/.
            hub.stage_hub_file(
                HF_DATA_REPO,
                f"{i2254.HF_PREFIX}/directions/{name}",
                target,
                repo_type="dataset",
            )
        return target

    def get(self, behavior: str, direction: str, layer: int) -> torch.Tensor:
        """Unit-norm fp32 (H,) tensor for (behavior, direction, layer); fail-loud."""
        key = (behavior, direction, int(layer))
        if key in self._cache:
            return self._cache[key]
        if direction == "rb":
            vec = torch.as_tensor(self._rb_bank()[behavior][layer], dtype=torch.float32)
            source = f"issue779 r_b rev {i2254.HF_REV[:12]} (unit_rows)"
        elif direction == "random":
            rnd = i2254.random_direction(HIDDEN_DIM, seed=i2254.SEED_RANDOM_BASE + int(layer))
            vec = torch.as_tensor(rnd, dtype=torch.float32)
            source = f"random_direction(seed={i2254.SEED_RANDOM_BASE + int(layer)}, n_avg=3)"
        elif direction in self._BANK_FAMILIES:
            path = self._stage_bank_file(behavior, direction, layer)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            assert payload["behavior"] == behavior, (payload["behavior"], behavior)
            assert payload["slug"] == direction, (payload["slug"], direction)
            assert int(payload["layer"]) == int(layer), (payload["layer"], layer)
            raw = payload["direction"].float()
            assert raw.shape == (HIDDEN_DIM,), (behavior, direction, layer, tuple(raw.shape))
            vec = raw / raw.norm()
            source = f"HF {i2254.HF_PREFIX}/directions/{path.name}"
        else:
            raise RuntimeError(f"unknown direction family {direction!r}")
        assert vec.shape == (HIDDEN_DIM,), (behavior, direction, layer, tuple(vec.shape))
        norm = float(vec.norm())
        assert abs(norm - 1.0) < 1e-3, (behavior, direction, layer, norm)
        self._cache[key] = vec
        self._sources[key] = source
        return vec

    def source_of(self, behavior: str, direction: str, layer: int) -> str:
        return self._sources[(behavior, direction, int(layer))]


# ---------------------------------------------------------------------------
# hook factory (plan §4.2: comparators byte-reuse the base DeltaHook)
# ---------------------------------------------------------------------------


def _assert_hook_types(position: str, steer) -> None:
    """Comparator arms byte-reuse the BASE ``DeltaHook``; windowed arms the
    subclass (plan §12.3 / the §7 smoke-gate kill criterion) — asserted in
    production too, per constructed hook."""
    children = steer.hooks if isinstance(steer, MultiLayerDeltaHook) else [steer]
    if position in ("lastctx", "allans"):
        bad = [type(h).__name__ for h in children if type(h) is not DeltaHook]
        assert not bad, (position, bad)
    else:
        bad = [type(h).__name__ for h in children if type(h) is not WindowedDeltaHook]
        assert not bad, (position, bad)


def build_recorded_hook(
    model, pos: str, layers: list[int], dirs: list, alphas: list
) -> RecordedHook:
    """Construct one position arm's steer hook + edit-position recorder.

    The SINGLE hook-construction path: the production ``_first_k_hook_factory``
    and the unit-3 CPU hook-mechanics smoke both build through here, so the
    smoke certifies the exact classes + modes production arms (plan §4.2/§4.4).
    """
    k = len(layers)
    assert k == len(dirs) == len(alphas) >= 1, (len(layers), len(dirs), len(alphas))
    if pos in ("lastctx", "allans"):
        all_positions = pos == "allans"
        if k == 1:
            steer = DeltaHook(model, layers[0], dirs[0], alphas[0], all_positions=all_positions)
        else:
            steer = multi_layer_delta_hooks(
                model, layers, dirs, alphas, all_positions=all_positions
            )
    else:
        window = COMBINED_WINDOW if pos == "combined" else POSITION_WINDOWS[pos]
        combine = pos == "combined"
        children = [
            WindowedDeltaHook(model, ly, d, a, decode_window=window, combine_prefill_ctx=combine)
            for ly, d, a in zip(layers, dirs, alphas, strict=True)
        ]
        steer = children[0] if k == 1 else MultiLayerDeltaHook(children)
    _assert_hook_types(pos, steer)
    recorder = EditPositionRecorder(model, layers[0])
    return RecordedHook(steer, recorder, position=pos)


def _hook_impl_record(cell: dict, n_layers: int) -> dict:
    """Per-cell record of which hook classes serve this arm (review evidence)."""
    pos = cell["position"]
    if pos in ("lastctx", "allans"):
        cls = "DeltaHook"
        mode = "all_positions" if pos == "allans" else "last_context_token"
    else:
        cls = "WindowedDeltaHook"
        mode = "combined" if pos == "combined" else "decode_window"
    return {
        "steer_hook_class": cls,
        "mode": mode,
        "n_layers": n_layers,
        "stacked": n_layers > 1,
        "comparator_base_reused": pos in ("lastctx", "allans"),
    }


def _first_k_hook_factory(model, cell: dict, rho_pooled: dict, bank: DirectionBank):
    """(make, alphas) for one cell: alpha_l = (c*/K) * rho_pooled[L_l]
    (plan §4.3 norm-match split, K = band width). ``make()`` returns a fresh
    ``RecordedHook`` (steer hook + edit-position recorder) per call."""
    layers = list(i2254.LAYER_CONFIGS[cell["layer_config"]])
    c = float(cell["c"])
    k = len(layers)
    missing = [ly for ly in layers if f"L{ly}" not in rho_pooled]
    assert not missing, f"rho_pooled_median missing layer keys {missing}"
    dirs = [bank.get(cell["behavior"], cell["direction"], ly).to(torch.bfloat16) for ly in layers]
    alphas = [(c / k) * float(rho_pooled[f"L{ly}"]) for ly in layers]
    pos = cell["position"]

    def make() -> RecordedHook:
        return build_recorded_hook(model, pos, layers, dirs, alphas)

    return make, {f"L{ly}": a for ly, a in zip(layers, alphas, strict=True)}


# ---------------------------------------------------------------------------
# per-cell generation (mirrors the parent's _gen_cell_rows, single seed_base)
# ---------------------------------------------------------------------------


def _gen_first_k_cell(
    model, tok, cell, contexts, q_idx, hook_make, *, n_draws, seed_base, max_new_tokens, alphas
):
    """Generate one cell: n_draws per context under the cell's hook at ONE
    seed_base (per-draw seeds seed_base+0..seed_base+n_draws-1 inside
    ``generate_batch`` — plan §4.4 reproduces the parent's 120 distinct
    completions/cell); per-context coherence flags; cap-hit fraction;
    per-draw edit-position traces. Content hygiene: completions land in the
    JSON payload only, never in logs."""
    from explore_persona_space.experiments.issue1415 import steering

    with hook_make() as hook:
        res = steering.generate_batch(
            model,
            tok,
            contexts,
            n=n_draws,
            hook=hook,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            seed_base=int(seed_base),
        )
    traces = hook.draw_traces
    assert len(traces) == n_draws, (len(traces), n_draws)
    coh = [steering.coherence_check(per_ctx) for per_ctx in res]
    n_layers = len(i2254.LAYER_CONFIGS[cell["layer_config"]])
    rec = {
        "cell_id": _cell_id(cell),
        "cell": cell,
        "alphas": alphas,
        "q_of_context": q_idx,
        "seeds": {
            str(seed_base): {
                "completions": res,
                "coherent_flags": coh,
                "condition_passes": [steering.condition_passes(flags) for flags in coh],
                "edit_traces": traces,
            }
        },
        "hook_impl": _hook_impl_record(cell, n_layers),
        "expected_edit_profile": expected_edit_profile(cell["position"], n_layers),
        "max_new_tokens": max_new_tokens,
        "cap_hit_fraction": float(i2254._cap_hit_fraction(res, tok, max_new_tokens)),
    }
    # Plan §4.4 read-backs, ALWAYS-ON (production and smoke): edit-count +
    # edit-position-identity vs the expected profile — the §7 smoke-gate
    # kill's mechanism, and a production fail-fast against window mis-indexing.
    rec["edit_trace_check"] = assert_cell_edit_traces(rec)
    return rec


STEER_PACK_FLUSH_EVERY = 8  # cells between incremental pack+upload flushes (durability cadence)
STEER_PACK_MAX_FILES = 64  # bounded-plan ceiling per shard (pack shards + manifest, #2286)
STEER_BYTES_PER_CELL = 2_500_000  # ~1 MB/cell at the 2048 cap; regen'd cells ~2x — sizing basis
# Empty-draw regen-once escape (firstk-empty-completion-validator-wedge): the
# retry seed block sits DISJOINT from the registered per-draw block
# (seed_base..seed_base+draws-1), so the retry draws fresh samples instead of
# deterministically reproducing the empty draw.
EMPTY_DRAW_SEED_SHIFT = 1000


def _empty_draw_slots(rec: dict) -> list[tuple[str, int, int]]:
    """(seed, ctx_idx, draw_idx) slots whose completion is empty/non-str — the
    exact predicate ``_validate_gen_record`` asserts on, evaluated at GEN time
    so the wedge is caught before any judge/reduce invocation."""
    out: list[tuple[str, int, int]] = []
    for seed, sd in rec["seeds"].items():
        for ci, draws in enumerate(sd["completions"]):
            for di, t in enumerate(draws):
                if not (isinstance(t, str) and t):
                    out.append((seed, ci, di))
    return out


def _regen_empty_draw_cell(gen_fn, cid: str, rec: dict, *, seed_base: int) -> dict:
    """Bounded validator-wedge escape (firstk-empty-completion-validator-wedge):
    when a cell carries empty completion draws, regenerate the WHOLE cell ONCE
    at a DISJOINT seed block (``seed_base + EMPTY_DRAW_SEED_SHIFT``) — the
    per-draw seeds are deterministic, so a same-seed retry reproduces the empty
    draw forever (``generate_batch`` has no ``min_new_tokens`` floor) and
    ``_validate_gen_record`` wedges every downstream judge/reduce run.
    Persisting empties is refused; a still-empty retry FAILS LOUD naming the
    wedge. The retry is audited in the record (``empty_draw_regen``)."""
    empty = _empty_draw_slots(rec)
    if not empty:
        return rec
    retry_base = int(seed_base) + EMPTY_DRAW_SEED_SHIFT
    logger.info(
        "[%s] %s: %d empty completion draw(s) — one-shot shifted-seed regen "
        "(seed_base %d -> %d; validator-wedge escape)",
        SENTINEL_STEER,
        cid,
        len(empty),
        int(seed_base),
        retry_base,
    )
    rec2 = gen_fn(seed_base=retry_base)
    still = _empty_draw_slots(rec2)
    if still:
        raise RuntimeError(
            f"steer {cid}: {len(still)} empty completion draw(s) persist after the one-shot "
            f"shifted-seed regen (seed_base {int(seed_base)} -> {retry_base}) — deterministic "
            "per-draw seeds reproduce empty draws (generate_batch has no min_new_tokens "
            "floor), so _validate_gen_record would wedge every judge/reduce run on this "
            "cell; investigate the cell's steering dose before any relaunch"
        )
    if "regen" in rec:  # keep the cap-hit regen diagnostics from the initial record
        rec2.setdefault("regen", rec["regen"])
    rec2["empty_draw_regen"] = {
        "n_empty_initial": len(empty),
        "slots_initial": [list(s) for s in empty[:20]],
        "seed_base_retry": retry_base,
    }
    return rec2


def _steer_regime_fp(args, cell: dict, rho_pooled: dict) -> str:
    """Machine-stable steer regime fingerprint (#2222/#2225 stale-cache class):
    every output-affecting dial — draws / q_steer / seed_base / generation cap /
    direction provenance (rb revision + random seed base) / the consumed
    rho_pooled_median values (FILE-READ floats, never recomputed — the
    code-style float-hash rule) — via the parent's _sha8. A cached cell whose
    stored fp mismatches is a cache MISS (regenerate), never a silent reuse."""
    layers = i2254.LAYER_CONFIGS[cell["layer_config"]]
    return i2254._sha8(
        {
            "draws": int(args.draws),
            "q_steer": int(args.q_steer),
            "seed_base": int(args.seed_base),
            "gen_cap": i2254.GEN_MAX_NEW_TOKENS,
            "rb_rev": i2254.HF_REV,
            "random_seed_base": i2254.SEED_RANDOM_BASE,
            "rho": {f"L{ly}": float(rho_pooled[f"L{ly}"]) for ly in layers},
        }
    )


def _assert_hub_headroom_for_steer(n_projected_files: int, projected_bytes: int) -> None:
    """Destination-headroom preflight BEFORE any GPU spend (#2286: the shared
    data repo sits at the Hub's ~1M-file ceiling, so net-new FILE COUNT is the
    binding resource): (a) the upload plan must be BOUNDED-BY-CONSTRUCTION —
    packed JSONL line-shards, O(10) files/shard, never a per-cell file
    fan-out; (b) byte/LFS headroom via the canonical
    hub.check_projected_upload_headroom (fail-loud only on a LIVE-confirmed
    'insufficient'). NOTE: hub's #1108 reactive file-count overflow fallback
    is MODEL-repo-scoped (hub._upload), so the dataset bulk-upload path has NO
    mechanical fallback — the bounded pack design is the load-bearing
    mitigation, and a file-count rejection at upload time fails loud."""
    from explore_persona_space.orchestrate import hub

    assert n_projected_files <= STEER_PACK_MAX_FILES, (
        f"steer upload plan projects {n_projected_files} net-new HF files — the pack design "
        "bounds this at O(pack shards); a per-cell upload fan-out regressed (#2286)"
    )
    verdict = hub.check_projected_upload_headroom(int(projected_bytes))
    if verdict.verdict == "insufficient":
        raise RuntimeError(
            f"steer: HF storage headroom insufficient for ~{projected_bytes / 1e9:.2f} GB "
            f"(used {verdict.used_tb} TB / ceiling {verdict.ceiling_tb} TB) — free headroom "
            "before any GPU spend"
        )
    logger.info(
        "[%s] hub headroom preflight: %s (~%d pack files, ~%.3f GB projected)",
        SENTINEL_STEER,
        verdict.verdict,
        n_projected_files,
        projected_bytes / 1e9,
    )


def _upload_steer_pack(comp_root: Path, shard_id: int, cell_names: list[str]) -> int:
    """Pack THIS SHARD's per-cell steer records into <=9 MB JSONL line-shards
    (the rw2220 pack recipe the unit-2 judge uploader uses) and upload the
    packed dir to a per-shard HF prefix — bounded net-new file count:
    O(pack shards) per flush, never O(cells) (#2286: the shared data repo sits
    at the Hub's ~1M-file ceiling, where per-cell uploads are rejected
    outright). Idempotent: each flush re-packs the shard's cell set from
    scratch (same shard filenames overwrite in place — no drift across
    flushes). Local per-cell JSONs stay on disk untouched (checkpoints are
    never deleted pre-upload-verify). Returns the pack shard count."""
    import scripts.issue2220_readwrite as rw2220

    stage = comp_root.parent / f"raw_completions_stage_shard{shard_id}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for name in cell_names:
        shutil.copy2(comp_root / name, stage / name)
    dest = comp_root.parent / f"raw_completions_pack_shard{shard_id}"
    if dest.exists():
        shutil.rmtree(dest)  # re-pack from scratch: shard numbering must not drift
    n = rw2220._pack_tree_to_jsonl_shards(
        stage, dest, group=f"firstk_steer_shard{shard_id}", pattern="*.json"
    )
    shutil.rmtree(stage)
    i2254._upload_folder_to_hf(
        dest,
        f"{_round_hf_prefix()}/raw_completions/steer_pack/shard{shard_id}",
        allow=["*.jsonl", "*.json"],
    )
    return n


# ---------------------------------------------------------------------------
# phase: stage_inputs (CPU pod-side; every reused input asserted pre-spend)
# ---------------------------------------------------------------------------


def _ensure_git_inputs() -> None:
    """Materialize the parent's committed JSON inputs (partial-clone pods'
    default cones exclude eval_results/ — #2211); fail-loud when absent."""
    for rel, cone in GIT_INPUTS:
        i2254._ensure_git_input(rel, cone)


def phase_stage_inputs(args) -> None:
    """Stage + assert every reused input BEFORE any GPU spend (plan §4.3/§10/§12).

    - e1 eval banks staged + sha-asserted (Sonnet-regen fallback unreachable);
      the 20-question eval bank present per behavior.
    - Direction bank: every consumed (behavior x {pre, ctxext, preshuf} x
      layer) per-layer .pt loaded through this driver's own loader path;
      shape (3584,) + unit norm asserted. rb from the #779 bank at rev
      ``037fcbb`` (rows unit-normalized); random regenerated
      deterministically (seed 2254+layer, mean of 3 draws).
    - Operating points: every consumed key resolved + asserted present; the
      dose path reads ``rho_pooled_median`` ONLY (§12.4 —
      ``rho_median_last_context_token`` is parity/pilot-only, never dosed).
    """
    out_root = i2254._out_root(args)
    rroot = round_root(out_root)
    _wipe_stale_sentinels([SENTINEL_STAGE])
    i2254._assert_phase_headroom(out_root, 1.0, SENTINEL_STAGE)
    behaviors = list(args.behaviors)
    bad = [b for b in behaviors if b not in ROUND_BEHAVIORS]
    assert not bad, f"behaviors outside the round scope (hallucination excluded, §5): {bad}"
    i2254._breadcrumb(SENTINEL_STAGE, behaviors=len(behaviors))

    staged_e1 = i2254._stage_e1_assets()
    for b in behaviors:
        qs = i2254._eval_questions(b)
        assert len(qs) >= args.q_steer, (b, len(qs), args.q_steer)

    _ensure_git_inputs()
    ops = i2254._load_operating_points(INPUTS_ROOT)
    resolved = resolve_operating_points(ops, behaviors)

    rho_pooled, rho_payload = i2254._load_rho(INPUTS_ROOT)
    # The dose field assert (plan §12.4): _load_rho returns rho_pooled_median;
    # assert the payload carries it AND that the values consumed match it.
    assert "rho_pooled_median" in rho_payload, sorted(rho_payload)
    consumed_layers = sorted(
        {ly for r in resolved.values() for ly in i2254.LAYER_CONFIGS[r["layer_config"]]}
    )
    missing_rho = [ly for ly in consumed_layers if f"L{ly}" not in rho_pooled]
    assert not missing_rho, f"rho_pooled_median missing layers {missing_rho}"
    for ly in consumed_layers:
        assert float(rho_pooled[f"L{ly}"]) == float(rho_payload["rho_pooled_median"][f"L{ly}"])

    # Consumer-open every direction the steer grid will load (check (c)/(h)).
    bank = DirectionBank()
    directions_loaded = []
    for (b, d, _breadth), r in sorted(resolved.items()):
        for ly in i2254.LAYER_CONFIGS[r["layer_config"]]:
            vec = bank.get(b, d, ly)
            directions_loaded.append(
                {
                    "behavior": b,
                    "direction": d,
                    "layer": int(ly),
                    "norm": float(vec.norm()),
                    "source": bank.source_of(b, d, ly),
                }
            )

    record = {
        "experiment": "issue2254_first_k_steering",
        "followup_label": FOLLOWUP_LABEL,
        "staged_e1_assets": staged_e1,
        "resolved_operating_points": {
            f"{b}__{d}__{br}": r for (b, d, br), r in sorted(resolved.items())
        },
        "rho_dose_field": "rho_pooled_median",
        "consumed_layers": consumed_layers,
        "n_directions_loaded": len(directions_loaded),
        "directions_loaded": directions_loaded,
        "rb_revision": i2254.HF_REV,
        "random_seed_base": i2254.SEED_RANDOM_BASE,
        "git_inputs": [rel for rel, _ in GIT_INPUTS],
        "sentinel_note": (
            "per-phase sentinels use the parent's issue-2254-<tag>.json envelope "
            "(plan §9 named a .done suffix; the .json envelope is the parent "
            "convention the orchestrator reads)"
        ),
    }
    payload = i2254._run_metadata(record)
    i2254._write_json_atomic(rroot / "stage_inputs" / "staged_inputs.json", payload)
    i2254._write_json_atomic(LOCAL_INPUT_STAGE / "staged_inputs.json", payload)
    i2254._write_sentinel(
        out_root,
        SENTINEL_STAGE,
        "done",
        {"behaviors": len(behaviors), "n_directions": len(directions_loaded)},
    )
    i2254._breadcrumb(SENTINEL_STAGE, status="done", n_directions=len(directions_loaded))


# ---------------------------------------------------------------------------
# phase: steer (GPU; the 160-cell position grid)
# ---------------------------------------------------------------------------


def phase_steer(args) -> None:
    """The 160-cell position grid (plan §4.4): 20 questions x 6 draws
    (seed_base 42) per cell at the inherited (l*, c*); per-cell JSON
    checkpoints (regime-fingerprinted cached-skip resume unless --force),
    round-robin ``--shard-id/--num-shards`` sharding, cap-hit > 2% => regen at
    2x cap, packed-JSONL HF raw-completion uploads (per-shard, incremental
    flush cadence — bounded file count, #2286) BEFORE the shard sentinel."""
    i2254._require_cuda("steer (first-k)")
    out_root = i2254._out_root(args)
    rroot = round_root(out_root)
    _wipe_stale_sentinels([SENTINEL_STEER, f"{SENTINEL_STEER}-shard{args.shard_id}"])
    i2254._assert_phase_headroom(out_root, 2.0, SENTINEL_STEER)
    behaviors = list(args.behaviors)
    bad = [b for b in behaviors if b not in ROUND_BEHAVIORS]
    assert not bad, f"behaviors outside the round scope (hallucination excluded, §5): {bad}"

    _ensure_git_inputs()
    ops = i2254._load_operating_points(INPUTS_ROOT)
    resolved = resolve_operating_points(ops, behaviors)
    rho_pooled, _ = i2254._load_rho(INPUTS_ROOT)
    cells = build_cells(args, resolved, behaviors)
    if not cells:
        raise RuntimeError("steer: empty cell list (selection bug — never a silent no-op)")
    assert 0 <= args.shard_id < args.num_shards, (args.shard_id, args.num_shards)
    shard = cells[args.shard_id :: args.num_shards]
    comp_root = rroot / "steer" / "raw_completions"
    comp_root.mkdir(parents=True, exist_ok=True)
    i2254._breadcrumb(SENTINEL_STEER, cells=len(cells), shard=len(shard), shard_id=args.shard_id)
    if not shard:
        # num_shards > len(cells): a legitimately EMPTY shard — nothing
        # generated, skip the folder upload (parent convention).
        logger.warning(
            "[%s] shard %d/%d is EMPTY (%d cells < num_shards) — nothing to generate",
            SENTINEL_STEER,
            args.shard_id,
            args.num_shards,
            len(cells),
        )
        i2254._write_sentinel(
            out_root,
            f"{SENTINEL_STEER}-shard{args.shard_id}",
            "done",
            {"cells": 0, "regen_cells": 0, "empty_shard": True},
        )
        i2254._breadcrumb(SENTINEL_STEER, status="done", regen_cells=0, empty_shard=1)
        return

    # Destination-headroom preflight BEFORE the GPU model load (#2286): the
    # packed upload plan is bounded (pack shards + manifest), never per-cell.
    n_pack_files = -(-len(shard) * STEER_BYTES_PER_CELL // 9_000_000) + 1
    _assert_hub_headroom_for_steer(n_pack_files, len(shard) * STEER_BYTES_PER_CELL)

    shard_names = [f"{_cell_id(c)}.json" for c in shard]

    def _flush_pack() -> None:
        have = [n for n in shard_names if (comp_root / n).exists()]
        if have:
            _upload_steer_pack(comp_root, args.shard_id, have)

    model, tok = i2254._load_model_and_tokenizer()
    bank = DirectionBank()
    q_cache = {b: i2254._eval_questions(b)[: args.q_steer] for b in behaviors}
    for b, qs in q_cache.items():
        assert len(qs) == args.q_steer, (b, len(qs), args.q_steer)

    t0 = time.time()
    n_regen = 0
    n_empty_regen = 0
    n_generated = 0
    for k, cell in enumerate(shard, 1):
        cid = _cell_id(cell)
        path = comp_root / f"{cid}.json"
        fp = _steer_regime_fp(args, cell, rho_pooled)
        if path.exists() and not args.force:
            cached_fp = json.loads(path.read_text()).get("regime_fp")
            if cached_fp == fp:
                i2254._progress(SENTINEL_STEER, k, len(shard), f"{cid} (cached)", t0)
                continue
            logger.info(
                "[%s] %s cached record regime_fp %s != %s — cache MISS, regenerating "
                "(draws/q/seed/cap/direction/rho dials changed; #2222 stale-cache class)",
                SENTINEL_STEER,
                cid,
                cached_fp,
                fp,
            )
        qs = q_cache[cell["behavior"]]
        contexts = i2254._contexts_for_questions(qs)
        q_idx = list(range(len(qs)))
        make, alphas = _first_k_hook_factory(model, cell, rho_pooled, bank)
        rec = _gen_first_k_cell(
            model,
            tok,
            cell,
            contexts,
            q_idx,
            make,
            n_draws=args.draws,
            seed_base=args.seed_base,
            max_new_tokens=i2254.GEN_MAX_NEW_TOKENS,
            alphas=alphas,
        )
        if rec["cap_hit_fraction"] > i2254.CAP_HIT_REGEN_FRAC:
            n_regen += 1
            logger.info(
                "[%s] %s cap-hit %.3f > %.2f — regenerating at %dx cap",
                SENTINEL_STEER,
                cid,
                rec["cap_hit_fraction"],
                i2254.CAP_HIT_REGEN_FRAC,
                i2254.CAP_HIT_REGEN_FACTOR,
            )
            initial = {
                "initial_cap_hit_fraction": rec["cap_hit_fraction"],
                "initial_max_new_tokens": i2254.GEN_MAX_NEW_TOKENS,
            }
            rec = _gen_first_k_cell(
                model,
                tok,
                cell,
                contexts,
                q_idx,
                make,
                n_draws=args.draws,
                seed_base=args.seed_base,
                max_new_tokens=i2254.GEN_MAX_NEW_TOKENS * i2254.CAP_HIT_REGEN_FACTOR,
                alphas=alphas,
            )
            # Raw completions persist FULL-LENGTH at the regen cap; the §6
            # common-2048-horizon recompute truncates from these (plan §8).
            rec["regen"] = initial
        # Empty-draw regen-once escape BEFORE persisting (deterministic seeds
        # otherwise wedge _validate_gen_record forever — see helper docstring).
        mnt = rec["max_new_tokens"]
        rec = _regen_empty_draw_cell(
            lambda seed_base: _gen_first_k_cell(
                model,
                tok,
                cell,
                contexts,
                q_idx,
                make,
                n_draws=args.draws,
                seed_base=seed_base,
                max_new_tokens=mnt,
                alphas=alphas,
            ),
            cid,
            rec,
            seed_base=args.seed_base,
        )
        if "empty_draw_regen" in rec:
            n_empty_regen += 1
        rec["experiment"] = "issue2254_first_k_steering"
        rec["followup_label"] = FOLLOWUP_LABEL
        rec["regime_fp"] = fp
        i2254._write_json_atomic(path, i2254._run_metadata(rec))
        n_generated += 1
        if n_generated % STEER_PACK_FLUSH_EVERY == 0:  # incremental durability flush
            _flush_pack()
        i2254._progress(SENTINEL_STEER, k, len(shard), cid, t0)

    # Final pack + upload covers the FULL shard cell set (cached cells too),
    # so a fully-cached resume still lands a complete pack before the sentinel.
    _flush_pack()
    tag = SENTINEL_STEER if args.num_shards == 1 else f"{SENTINEL_STEER}-shard{args.shard_id}"
    i2254._write_sentinel(
        out_root,
        tag,
        "done",
        {"cells": len(shard), "regen_cells": n_regen, "empty_regen_cells": n_empty_regen},
    )
    i2254._breadcrumb(
        SENTINEL_STEER, status="done", regen_cells=n_regen, empty_regen_cells=n_empty_regen
    )


# ---------------------------------------------------------------------------
# unit 2 — judge + reduce constants (plan v10 §3 / §6)
# ---------------------------------------------------------------------------

SENTINEL_JUDGE = "firstk-judge"
SENTINEL_REDUCE = "firstk-reduce"
JUDGE_DRAWS_FIRSTK = 5  # plan §6: 5 draws at the inherited #2220/#2254 instrument
# Transport pin (plan §6 + llm-judging rule 26(c) transport parity): one cell's
# wave is 120 items x 5 draws = 600 calls, UNDER the OTPM-scaled default
# sync-vs-batch threshold (judge_dispatch.decide_route), so count-routing would
# silently run SYNC while the plan declares Batch. threshold_base=0 clamps the
# effective threshold to 1 and pins EVERY dispatch (pilot AND wave) onto Batch.
JUDGE_THRESHOLD_BASE_BATCH = 0
# Rule-28 remediation transport: an astronomically large threshold_base keeps
# n_items below the effective threshold, forcing SYNC for the targeted
# re-issue of api-refusal-censored draws.
SYNC_FORCE_THRESHOLD_BASE = 10**9
# Plan §6 pilot sizing: 4 arms x >=55 effective draws each (>= the #2124
# 51-per-arm parse-fail satisfiability floor at the 2% threshold).
PILOT_MIN_EFFECTIVE_FIRSTK = 55
PILOT_API_REFUSAL_MAX = 0.10  # plan §6 pilot kill: per-arm api-refusal rate < 0.10
OPENING_POSITIONS = ("tok1", "tok2", "tok3", "span13", "span15")
STRONG_DIRECTIONS = ("rb", "pre")  # plan §3 lattice scope: strong directions only
H3_CHAIN = ("lastctx", "tok1", "span13", "span15", "allans")  # §3 H3 accrual chain
COMMON_HORIZON_TOKENS = i2254.GEN_MAX_NEW_TOKENS  # 2048 — the §3 D-index common horizon
RATIO_DEN_FLOOR = 5.0  # §3 denominator guard: |A_b - alpha0_b| < 5 score points
RATIO_UNSTABLE_FRAC = 0.01  # §3: >1% of resamples under the floor => R undefined
R_SUFFICIENT = 2.0 / 3.0  # §3 lattice: R_lo >= 2/3 arm of opening-sufficient
R_PARTIAL = 1.0 / 3.0  # §3 lattice: R_pt >= 1/3 arm of opening-partial
PLATEAU_FRAC = 0.90  # §3 H3: span1-5 >= ~90% of all-answer = plateau (point read)
COMPLETENESS_FLOOR = 0.95  # rule-29 per-cell frac_items_complete floor (validity gate)

# Committed parent inputs the reduce consumes IN ADDITION to unit 1's
# GIT_INPUTS (separate tuple so unit 1's staging list stays byte-stable).
BASELINE_PERCELL_REL = "eval_results/issue_2254/baseline_ceiling/judged_percell.json"
CJK_AUDIT_REL = "eval_results/issue_2254/decisive/cjk_audit.json"
DECISIVE_PERCELL_REL = "eval_results/issue_2254/decisive/delta_score_percell.json"
assert BASELINE_PERCELL_REL == GIT_INPUTS[2][0], GIT_INPUTS[2]
REDUCE_GIT_INPUTS = (
    (CJK_AUDIT_REL, "eval_results/issue_2254/decisive"),
    (DECISIVE_PERCELL_REL, "eval_results/issue_2254/decisive"),
)


# ---------------------------------------------------------------------------
# unit 2 — judge phase (plan §6: Batch API, rule-26 pilot, rule-28 re-issue)
# ---------------------------------------------------------------------------


def _hub_tree(prefix: str, *, recursive: bool = False) -> list:
    """Scoped list_repo_tree via retry_transient — never a bare full-repo
    listing on the ~1M-file data repo; [] when the prefix is absent. Module
    seam (tests monkeypatch it — no-network staging tests)."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import EntryNotFoundError

    from explore_persona_space.orchestrate import hub

    try:
        return hub.retry_transient(
            lambda: list(
                # HUB_VERIFY_RETRY_EXEMPT: staging READ wrapped in hub.retry_transient
                HfApi().list_repo_tree(
                    HF_DATA_REPO,
                    path_in_repo=prefix,
                    repo_type="dataset",
                    recursive=recursive,
                )
            ),
            what=f"list_repo_tree({prefix})",
        )
    except EntryNotFoundError:  # prefix absent — caller falls through
        return []


def _hub_stage(path_in_repo: str, target: Path) -> None:
    """Retried atomic fail-loud single-file staging download (test seam)."""
    from explore_persona_space.orchestrate import hub

    hub.stage_hub_file(HF_DATA_REPO, path_in_repo, target, repo_type="dataset")


def _assert_staged_regime_fps(comp_root: Path, expected_fp: dict[str, str], src: str) -> None:
    """Regime-fp cross-check on staged/local gen records BEFORE trusting them
    (r2 concerns firstk-localfirst-stage-no-fp-check + the rehydration leg of
    firstk-pack-manifest-stale-tail): every EXPECTED cell present under
    comp_root must carry the invocation's regime_fp — a missing/mismatched fp
    is a stale/mixed vintage and is REFUSED (regenerate via the steer phase),
    never silently judged/reduced. Absent cells are the caller's
    grid-completeness check, not this one."""
    bad: list[str] = []
    for cid, fp in sorted(expected_fp.items()):
        p = comp_root / f"{cid}.json"
        if not p.is_file():
            continue
        got = json.loads(p.read_text()).get("regime_fp")
        if got != fp:
            bad.append(f"{cid}: {got} != {fp}")
    if bad:
        raise RuntimeError(
            f"firstk staging ({src}): {len(bad)} gen record(s) fail the regime_fp "
            f"cross-check — stale/mixed vintage refused, regenerate via the steer "
            f"phase (first: {bad[:4]})"
        )


def _stage_round_completions(rroot: Path, expected_fp: dict[str, str]) -> Path:
    """Local-first steer raw_completions; else stage + UNPACK the per-shard
    JSONL pack shards the steer phase uploads (rw2220 line schema:
    ``{"path": <cell file rel path>, "doc": <cell record>}``) from the round
    HF prefix, MANIFEST-DRIVEN (r2 concern firstk-pack-manifest-stale-tail:
    exactly the shards each ``pack_manifest.json`` names are loaded — a
    shrinking repack's stale remote tail shards are IGNORED, duplicate cell
    paths across shards are REFUSED, and an un-manifested shard set is
    refused outright); else the legacy per-cell prefix (pre-pack artifacts).
    Every branch ends with the regime_fp cross-check against the invocation
    (``_assert_staged_regime_fps``)."""
    comp_root = rroot / "steer" / "raw_completions"
    if comp_root.exists() and any(comp_root.glob("*.json")):
        _assert_staged_regime_fps(comp_root, expected_fp, "local-first")
        return comp_root

    pack_prefix = f"{_round_hf_prefix()}/raw_completions/steer_pack"
    entries = _hub_tree(pack_prefix, recursive=True)
    manifest_paths = sorted(e.path for e in entries if Path(e.path).name == "pack_manifest.json")
    remote_jsonl = {e.path for e in entries if e.path.endswith(".jsonl")}
    if manifest_paths:
        dl_root = rroot / "steer" / "raw_completions_pack_dl"
        seen: dict[str, str] = {}  # cell filename -> shard path (duplicate refusal)
        named_paths: set[str] = set()
        n_cells = 0
        for mp in manifest_paths:
            mlocal = dl_root / Path(mp).relative_to(pack_prefix)
            _hub_stage(mp, mlocal)
            manifest = json.loads(mlocal.read_text())
            parent = str(Path(mp).parent)
            n_rows = 0
            for name in manifest["shards"]:
                pth = f"{parent}/{name}"
                if pth not in remote_jsonl:
                    raise RuntimeError(
                        f"firstk judge: manifest {mp} names shard {name} absent from the "
                        "remote listing — partial/corrupt pack upload, refusing rehydration"
                    )
                named_paths.add(pth)
                local = dl_root / Path(pth).relative_to(pack_prefix)
                _hub_stage(pth, local)
                for line in local.open(encoding="utf-8"):  # text-mode, never splitlines
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    cname = Path(row["path"]).name
                    assert cname.endswith(".json"), row["path"]
                    if cname in seen:
                        raise RuntimeError(
                            f"firstk judge: duplicate cell record {cname} in {pth} (already "
                            f"unpacked from {seen[cname]}) — overlapping packs refused "
                            "(firstk-pack-manifest-stale-tail)"
                        )
                    seen[cname] = pth
                    i2254._write_json_atomic(comp_root / cname, row["doc"])
                    n_cells += 1
                    n_rows += 1
            if n_rows != int(manifest["n_files"]):
                raise RuntimeError(
                    f"firstk judge: manifest {mp} declares n_files={manifest['n_files']} but "
                    f"its shards unpacked {n_rows} rows — corrupt pack refused"
                )
        if not n_cells:
            raise RuntimeError(
                f"firstk judge: pack manifests under {pack_prefix} unpacked ZERO cell records"
            )
        stale_tail = sorted(remote_jsonl - named_paths)
        logger.info(
            "[%s] staged %d cells from %d manifest(s) at %s (%d stale tail shard(s) ignored)",
            SENTINEL_JUDGE,
            n_cells,
            len(manifest_paths),
            pack_prefix,
            len(stale_tail),
        )
        _assert_staged_regime_fps(comp_root, expected_fp, "pack-rehydration")
        return comp_root
    if remote_jsonl:
        raise RuntimeError(
            f"firstk judge: {len(remote_jsonl)} pack shard(s) under {pack_prefix} with NO "
            "pack_manifest.json — un-manifested rehydration refused (stale tail shards "
            "would be indistinguishable; firstk-pack-manifest-stale-tail)"
        )

    legacy_prefix = f"{_round_hf_prefix()}/raw_completions/steer"
    paths = [e.path for e in _hub_tree(legacy_prefix) if e.path.endswith(".json")]
    if not paths:
        raise RuntimeError(
            f"firstk judge: no steer completions locally, at {pack_prefix}, or at {legacy_prefix}"
        )
    for pth in paths:
        _hub_stage(pth, comp_root / Path(pth).name)
    _assert_staged_regime_fps(comp_root, expected_fp, "legacy-percell")
    return comp_root


def _judge_ctx_id_firstk(cell: dict, seed: int, i: int) -> str:
    """'-'-joined judge context id over OUR position vocabulary (the parent's
    ``_judge_ctx_id`` keys ``_POS_SHORT[context|answer]`` and would KeyError
    on tok1/span13/...); ``rollout_item_id`` appends ``_kNN``, so the ctx-id
    budget is MAX_ITEM_ID_LEN - 4 and must carry no ``__``."""
    from explore_persona_space.experiments.issue_1739.judging import MAX_ITEM_ID_LEN

    stem = "-".join(
        (
            cell["behavior"],
            i2254._DIR_SHORT[cell["direction"]],
            _POS_TOKEN[cell["position"]],
            cell["layer_config"],
            i2254._c_token(cell["c"]),
        )
    )
    out = f"{stem}-s{seed}-x{i:03d}"
    assert "__" not in out and len(out) <= MAX_ITEM_ID_LEN - 4, out
    return out


def _pilot_gen_hash(files: list[Path]) -> str:
    """Content hash over one behavior's staged gen records (r2 concern
    firstk-pilot-pass-input-fingerprint): file-byte shas, never recomputed
    floats (the code-style float-hash rule) — any regen'd source cell changes
    the hash and therefore the pilot-PASS fingerprint."""
    return i2254._sha8({f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:12] for f in files})


def _judge_instrument_fp(rubric: str, n_draws: int) -> str:
    """Judge-instrument fingerprint SHARED by the judged-checkpoint writer
    (``_judge_firstk_cell``) and the reduce-side vintage assert
    (``_assert_judged_vintage``, r2 Codex C3): rubric / model / max_tokens /
    draws / temperature / transport pin."""
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
    )

    return i2254._sha8(
        {
            "rubric": rubric,
            "model": JUDGE_MODEL,
            "max_tokens": i2254.JUDGE_MAX_TOKENS_2254,
            "n_draws": n_draws,
            "temp": JUDGE_TEMPERATURE,
            "tb": JUDGE_THRESHOLD_BASE_BATCH,
        }
    )


def _run_firstk_pilot(args, rroot: Path, comp_root: Path, behavior: str, rubric, n_draws) -> None:
    """Rule-26 pilot gate at the EXACT production instrument + transport
    (plan §6): 4 arms per behavior — rb all-answer, random all-answer, one
    degenerate high-dose cell, one clean opening cell — >=55 effective draws
    each, Batch transport (threshold_base=0), production ambient temperature
    (``graded_temperature`` — the gate's own temperature kwarg is unthreaded).
    Adds the driver-side per-arm api-refusal (<0.10) kill the shipped gate
    only REPORTS, plus the §7 positive-control EARLY read (coarse, unpowered
    — log line + sidecar field, never a gate). Fingerprint sidecar skips a
    prior PASS at the identical instrument unless --force."""
    from explore_persona_space.eval.judge_dispatch import graded_temperature
    from explore_persona_space.eval.judge_pilot import _seeded_subsample, judge_pilot_gate
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
    )
    from explore_persona_space.experiments.issue_1739.judging import judge_items_graded

    pilot_dir = rroot / "judge" / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(comp_root.glob(f"{behavior}__*.json"))
    if not files:
        raise RuntimeError(f"firstk pilot: no {behavior} gen cells under {comp_root}")
    fp = i2254._sha8(
        {
            "behavior": behavior,
            "rubric": rubric,
            "n_draws": n_draws,
            "mt": i2254.JUDGE_MAX_TOKENS_2254,
            "tb": JUDGE_THRESHOLD_BASE_BATCH,
            "temp": JUDGE_TEMPERATURE,
            # Source-generation content hash (r2 firstk-pilot-pass-input-
            # fingerprint): a regenerated gen cell invalidates a prior pilot
            # PASS + its cached positive-control replay.
            "gen": _pilot_gen_hash(files),
        }
    )
    pass_path = pilot_dir / f"{behavior}.pass.json"
    if pass_path.exists() and not args.force:
        if json.loads(pass_path.read_text()).get("fingerprint") == fp:
            logger.info(
                "[%s] pilot %s: prior PASS, identical instrument + inputs",
                SENTINEL_JUDGE,
                behavior,
            )
            return
    items_per_arm = -(-PILOT_MIN_EFFECTIVE_FIRSTK // n_draws)  # ceil: >=55 draws/arm (§6)
    qs = i2254._eval_questions(behavior)
    used_by_arm: dict[str, list[str]] = {}
    used_any: set[str] = set()

    def _collect(arm: str, cell_filter, sort_key) -> list[tuple[str, str, str]]:
        """Fill one pilot arm from the top-sorted matching cells, EXCLUDING
        cells already consumed by earlier arms — the plan-§6 four arms must
        come from four DISTINCT source cells (disjoint by construction)."""
        recs = []
        for f in files:
            rec = json.loads(f.read_text())
            if rec["cell_id"] in used_any or not cell_filter(rec["cell"]):
                continue
            recs.append((sort_key(rec), rec))
        recs.sort(key=lambda kv: kv[0])
        items: list[tuple[str, str, str]] = []
        used: list[str] = []
        for _k, rec in recs:
            took = False
            for qi, _seed, _ci, _di, text in i2254._iter_gen_qa(rec):
                items.append((f"{arm}-{len(items):03d}", qs[qi], text))
                took = True
                if len(items) >= items_per_arm:
                    break
            if took:
                used.append(rec["cell_id"])
            if len(items) >= items_per_arm:
                break
        used_by_arm[arm] = used
        used_any.update(used)
        return items

    def _degen_highdose(c: dict) -> bool:
        # Plan §6 degenerate HIGH-DOSE arm: the REGISTERED dose predicate
        # (|c*| >= 2 at an all-answer/combined arm — the maximal-exposure
        # positions), tie-broken by LOWEST coherence rate (most degenerate).
        return abs(float(c.get("c", 0.0))) >= 2.0 and c["position"] in ("allans", "combined")

    arms = {
        "rb_allans": _collect(
            "rb_allans",
            lambda c: c["direction"] == "rb" and c["position"] == "allans",
            lambda r: -abs(float(r["cell"].get("c", 0.0))),
        ),
        "rnd_allans": _collect(
            "rnd_allans",
            lambda c: c["direction"] == "random" and c["position"] == "allans",
            lambda r: -abs(float(r["cell"].get("c", 0.0))),
        ),
        "degen_highdose": _collect("degen_highdose", _degen_highdose, i2254._coherence_rate),
        "clean_opening": _collect(
            "clean_opening",
            lambda c: c["position"] in OPENING_POSITIONS,
            lambda r: -i2254._coherence_rate(r),
        ),
    }
    empty = sorted(k for k, v in arms.items() if not v)
    if empty and not args.smoke:
        raise RuntimeError(
            f"firstk pilot {behavior}: empty pilot arms {empty} on the PRODUCTION grid — "
            "every §6 arm predicate must resolve (empty-arm drop is smoke-only)"
        )
    if empty:  # smoke slice (pre x single x 5 positions) has no rb/random cells
        logger.info("[%s] pilot %s: dropping empty arms %s", SENTINEL_JUDGE, behavior, empty)
    arms = {k: v for k, v in arms.items() if v}
    if not arms:
        raise RuntimeError(f"firstk pilot: zero pilot items for {behavior}")
    # §6: the four arms draw from DISJOINT source cells (asserted, not assumed).
    for a1, a2 in itertools.combinations(sorted(arms), 2):
        overlap = set(used_by_arm[a1]) & set(used_by_arm[a2])
        assert not overlap, (a1, a2, sorted(overlap))
    for a in arms:
        assert used_by_arm[a], (a, "arm filled with zero source cells")
    with graded_temperature(JUDGE_TEMPERATURE):  # production ambient (kwarg alone unthreaded)
        report = judge_pilot_gate(
            arms,
            rubric,
            max_tokens=i2254.JUDGE_MAX_TOKENS_2254,
            cache_dir=pilot_dir / f"{behavior}_cache",
            save_raw_dir=pilot_dir / f"{behavior}_raw",
            n_draws=n_draws,
            target_total_draws=len(arms) * n_draws * items_per_arm,
            min_effective_draws_per_arm=PILOT_MIN_EFFECTIVE_FIRSTK,
            waive_parse_fail_arms=tuple(args.waive_judge_parse_fail_arms),
            allow_subresolution_pilot=bool(args.smoke),
            threshold_base=JUDGE_THRESHOLD_BASE_BATCH,
            report_path=pilot_dir / f"{behavior}.report.json",
            seed=0,
        )
    if not report.passed:
        raise RuntimeError(f"firstk judge pilot FAILED for {behavior}: {report.verdict}")
    # Driver-side rule-28 arm kill: the shipped judge_pilot_gate REPORTS
    # n_api_refusal but has no gate condition on it (its #2151 note); plan §6
    # registers per-arm api-refusal rate < 0.10 as a pilot kill.
    refusal_rates: dict[str, float] = {}
    for arm, st in report.arms.items():
        rate = st.n_api_refusal / max(1, st.n_draws - st.n_transport_lost)
        refusal_rates[arm] = float(rate)
        if rate >= PILOT_API_REFUSAL_MAX:
            raise RuntimeError(
                f"firstk judge pilot FAILED for {behavior}: arm {arm} api-refusal "
                f"rate {rate:.3f} >= {PILOT_API_REFUSAL_MAX} (plan §6 pilot kill)"
            )
    # §7 positive-control EARLY read: rb vs random all-answer pilot-arm mean
    # dscore, replayed from the pilot's own per-arm rubric-keyed cache
    # (identical subsample + instrument -> cache hits, no new spend).
    pc = None
    if "rb_allans" in arms and "rnd_allans" in arms:
        means: dict[str, dict] = {}
        for arm in ("rb_allans", "rnd_allans"):
            sub = _seeded_subsample(arms[arm], items_per_arm, seed=0, arm=arm)
            r = judge_items_graded(
                sub,
                rubric,
                cache_dir=pilot_dir / f"{behavior}_cache" / arm,
                save_raw=pilot_dir / f"{behavior}_raw" / f"pc_replay_{arm}.json",
                n_draws=n_draws,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=i2254.JUDGE_MAX_TOKENS_2254,
                judge_model=JUDGE_MODEL,
                threshold_base=JUDGE_THRESHOLD_BASE_BATCH,
            )
            kept = [s for sc in r.per_item_scores.values() for s in sc]
            means[arm] = {"mean": float(np.mean(kept)) if kept else None, "n_kept": len(kept)}
        if means["rb_allans"]["mean"] is not None and means["rnd_allans"]["mean"] is not None:
            delta = means["rb_allans"]["mean"] - means["rnd_allans"]["mean"]
            pc = {**means, "delta_rb_minus_random": delta}
            logger.info(
                "[%s] pilot %s positive-control early read: rb-rnd all-answer "
                "dscore=%.1f (coarse, unpowered — plan §6/§7)",
                SENTINEL_JUDGE,
                behavior,
                delta,
            )
        else:
            pc = {**means, "delta_rb_minus_random": None}
    i2254._write_json_atomic(
        pass_path,
        {
            "fingerprint": fp,
            "verdict": report.verdict,
            "transport": "batch (threshold_base=0 pin; rule-26 transport parity)",
            "api_refusal_rates": refusal_rates,
            "api_refusal_max": PILOT_API_REFUSAL_MAX,
            "positive_control_early": pc,
            "dropped_empty_arms": empty,
            "arm_source_cells": used_by_arm,  # §6: disjoint source cells, asserted above
        },
    )


def _judge_firstk_cell(args, rroot: Path, gen_path: Path, rubric: str, n_draws: int) -> dict:
    """Judge one steer cell (Batch API, rubric-keyed cache, max_tokens 2048)
    with the rule-28 targeted SYNC re-issue for api-refusal-censored draws:
    refused draws are re-dispatched at the IDENTICAL instrument on the sync
    path with a FRESH cache dir (the rubric-keyed cache shares one key across
    an item's draws — a same-cache replay would silently duplicate surviving
    sibling draws instead of drawing fresh). Per-cell checkpoint at
    judge/judged/<cid>.json with cached-skip resume; accounting keeps the
    batch/sync split + rule-29 frac_items_complete (post-merge)."""
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
    )
    from explore_persona_space.experiments.issue_1739.judging import (
        judge_items_graded,
        judge_tallies,
        rollout_item_id,
    )

    raw = gen_path.read_bytes()
    rec = json.loads(raw)
    cell = rec["cell"]
    cid = rec["cell_id"]
    # Judged-checkpoint resume keys (#2222/#2225 stale-artifact-done class):
    # gen-file BYTE sha (a steer regen/re-run invalidates the judged record —
    # the horizon ckpt's exact discipline) + the judge-instrument fingerprint
    # (rubric/model/max_tokens/draws/temperature/transport). Mismatch = MISS.
    gen_sha = hashlib.sha256(raw).hexdigest()[:12]
    judge_fp = _judge_instrument_fp(rubric, n_draws)
    out_path = rroot / "judge" / "judged" / f"{cid}.json"
    if out_path.exists() and not args.force:
        cached = json.loads(out_path.read_text())
        if cached.get("gen_sha") == gen_sha and cached.get("judge_fp") == judge_fp:
            return cached
        logger.info(
            "[%s] %s judged checkpoint stale (gen_sha/judge_fp mismatch) — re-judging "
            "(content-keyed JudgeCache makes unchanged texts cache-cheap)",
            SENTINEL_JUDGE,
            cid,
        )
    qs = i2254._eval_questions(cell["behavior"])
    items: list[tuple[str, str, str]] = []
    meta: dict[str, dict] = {}
    for qi, seed, ci, di, text in i2254._iter_gen_qa(rec):
        iid = rollout_item_id(_judge_ctx_id_firstk(cell, seed, len(items)), di)
        items.append((iid, qs[qi], text))
        meta[iid] = {"qi": qi, "seed": seed, "ci": ci, "di": di}
    result = judge_items_graded(
        items,
        rubric,
        cache_dir=rroot / "judge" / "cache" / cid,
        save_raw=rroot / "judge" / "raw" / cid,
        n_draws=n_draws,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=i2254.JUDGE_MAX_TOKENS_2254,
        judge_model=JUDGE_MODEL,
        threshold_base=JUDGE_THRESHOLD_BASE_BATCH,
    )
    merged: dict[str, list[float]] = {
        iid: [float(s) for s in sc] for iid, sc in result.per_item_scores.items()
    }
    need = {iid: int(k) for iid, k in result.per_item_api_refusals.items() if int(k) > 0}
    reissue = None
    if need:
        item_by_id = {it[0]: it for it in items}
        groups: dict[int, list[str]] = {}
        for iid, k in need.items():
            groups.setdefault(k, []).append(iid)
        re_draws = re_scored = re_refused = re_transport = 0
        for k, iids in sorted(groups.items()):
            sub = [item_by_id[iid] for iid in sorted(iids)]
            r2 = judge_items_graded(
                sub,
                rubric,
                cache_dir=rroot / "judge" / "cache" / f"{cid}_syncfix_k{k}",
                save_raw=rroot / "judge" / "raw" / f"{cid}_syncfix_k{k}",
                n_draws=k,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=i2254.JUDGE_MAX_TOKENS_2254,
                judge_model=JUDGE_MODEL,
                threshold_base=SYNC_FORCE_THRESHOLD_BASE,
            )
            for iid, sc in r2.per_item_scores.items():
                if sc:
                    merged.setdefault(iid, []).extend(float(s) for s in sc)
            re_draws += r2.n_total_draws
            re_scored += sum(len(sc) for sc in r2.per_item_scores.values())
            re_refused += r2.n_api_refusal_draws
            re_transport += r2.n_transport_lost_draws
        reissue = {
            "transport": "sync (rule-28 targeted re-issue; threshold_base forces sync)",
            "n_items_reissued": len(need),
            "n_draws_reissued": re_draws,
            "n_scored": re_scored,
            "n_api_refusal_residual": re_refused,
            "n_transport_lost": re_transport,
        }
        logger.info(
            "[%s] %s rule-28 sync re-issue: %d items / %d draws (residual refusals %d)",
            SENTINEL_JUDGE,
            cid,
            len(need),
            re_draws,
            re_refused,
        )
    per_q: dict[int, list[float]] = {}
    for iid, scores in merged.items():
        if scores:
            per_q.setdefault(meta[iid]["qi"], []).append(float(np.mean(scores)))
    n_q = (max(m["qi"] for m in meta.values()) + 1) if meta else 0
    per_q_mean = [float(np.mean(per_q[q])) if q in per_q else None for q in range(n_q)]
    per_q_rate = [
        float(np.mean([s >= i2254.SCORE_THRESHOLD for s in per_q[q]])) if q in per_q else None
        for q in range(n_q)
    ]
    valid_means = [m for m in per_q_mean if m is not None]
    valid_rates = [r for r in per_q_rate if r is not None]
    coherence_rate = i2254._coherence_rate(rec)
    fc_merged = (
        float(np.mean([min(len(sc), n_draws) / n_draws for sc in merged.values()]))
        if merged
        else None
    )
    out = {
        "cell_id": cid,
        "cell": cell,
        "phase": "steer",
        "gen_sha": gen_sha,
        "judge_fp": judge_fp,
        "n_questions": n_q,
        "judge": {
            "model": JUDGE_MODEL,
            "n_draws": n_draws,
            "max_tokens": i2254.JUDGE_MAX_TOKENS_2254,
            "temperature": JUDGE_TEMPERATURE,
            "transport": "batch (threshold_base=0 pin)",
        },
        "items": meta,
        "accounting": {
            **judge_tallies(result),
            "n_refusal_draws": result.n_refusal_draws,
            "n_api_refusal_draws": result.n_api_refusal_draws,
            "per_item_api_refusals": result.per_item_api_refusals,
            "frac_items_complete_batch": (result.frac_items_complete if result.scores else None),
            # rule-29 DV denominator: post-merge completeness (batch survivors
            # + rule-28 sync replacements, capped at n_draws per item).
            "frac_items_complete": fc_merged,
            "sync_reissue": reissue,
            "n_items": len(items),
            "n_items_zero_valid": sum(1 for sc in merged.values() if not sc),
        },
        "per_item_scores_merged": merged,
        "per_question_mean_score": per_q_mean,
        "per_question_rate": per_q_rate,
        "per_question_n": [len(per_q.get(q, [])) for q in range(n_q)],
        "mean_score": float(np.mean(valid_means)) if valid_means else None,
        "rate": float(np.mean(valid_rates)) if valid_rates else None,
        "coherence_rate": coherence_rate,
        "coherence_pass": bool(coherence_rate >= i2254.COHERENCE_CELL_GATE),
        "cap_hit_fraction": rec.get("cap_hit_fraction"),
        "max_new_tokens": rec.get("max_new_tokens"),
        "regen": rec.get("regen"),
        "alphas": rec.get("alphas"),
    }
    i2254._write_json_atomic(out_path, i2254._run_metadata(out))
    return out


def _upload_pilot_pack_firstk(rroot: Path) -> None:
    """Pilot-scoped pack + upload (r2 BLOCKER firstk-pilot-pack-reachability):
    the pilot evidence (raw/cache/report/pass sidecars) is the run's ONLY
    product on the pilot-gate FAIL, refusal-rate kill, §7 positive-control
    kill, and ``--pilot`` exits, so ``phase_judge`` invokes this on EVERY
    exit from its pilot section (try/finally) — never only from the
    post-wave full upload. Whole-tree pack (pattern='*'): the pilot's
    raw/cache trees nest under <behavior>_raw/ + <behavior>_cache/ and hold
    judge_raw_pilot_* + pc_replay_* files an fnmatch allow-list would miss
    (HF allow_patterns match the FULL relative path) — packing the whole
    tree uploads EVERY pilot artifact, so `discarded_artifacts: none` holds
    (upload-policy: judge outputs upload always). Idempotent (re-pack +
    re-upload); a pilot dir with no artifacts yet is a logged no-op."""
    import scripts.issue2220_readwrite as rw2220

    pilot = rroot / "judge" / "pilot"
    if not pilot.exists() or not any(p.is_file() for p in pilot.rglob("*")):
        logger.info("[%s] pilot pack: no pilot artifacts to upload yet", SENTINEL_JUDGE)
        return
    dest = rroot / "judge" / "pilot_pack"
    rw2220._pack_tree_to_jsonl_shards(pilot, dest, group="firstk_pilot", pattern="*")
    i2254._upload_folder_to_hf(
        dest, f"{_round_hf_prefix()}/judge/pilot_pack", allow=["*.jsonl", "*.json"]
    )


def _upload_judge_outputs_firstk(rroot: Path) -> None:
    """Pack the per-cell judge trees (judged/cache/raw) into <=9 MB plain
    JSONL line-shards and upload ONLY the packed dirs — the shared data repo
    sits at the Hub's 1M-file ceiling (#2286), so per-cell uploads of O(100)
    files are rejected outright. save_raw writes bare-<cid> EXTENSIONLESS
    files -> pattern='*' for raw (the parent wave-1 raw-drop lesson). The
    pilot tree delegates to ``_upload_pilot_pack_firstk`` (which ALSO runs on
    every pilot-section exit — reachability, r2 BLOCKER)."""
    import scripts.issue2220_readwrite as rw2220

    base = rroot / "judge"
    for sub in ("judged", "cache", "raw"):
        src = base / sub
        if not src.exists():
            continue
        dest = base / f"{sub}_pack"
        pattern = "*" if sub == "raw" else "*.json"
        rw2220._pack_tree_to_jsonl_shards(src, dest, group=f"firstk_{sub}", pattern=pattern)
        i2254._upload_folder_to_hf(
            dest, f"{_round_hf_prefix()}/judge/{sub}_pack", allow=["*.jsonl", "*.json"]
        )
    _upload_pilot_pack_firstk(rroot)


def _validate_gen_record(rec: dict, path: Path, *, n_q: int, n_draws: int) -> None:
    """Producer-schema contract on ONE staged steer gen record BEFORE any
    judge/reduce spend (plan §6 grid integrity — the r1 filename-only check
    let a truncated/mixed-grain record through): filename<->cell identity,
    question grain, per-context draw counts (non-empty completion strings),
    trace-field presence (the producer's always-on ``assert_cell_edit_traces``
    guarantees non-empty traces at gen time), cap-hit metadata present."""
    cid = path.stem
    assert rec.get("cell_id") == cid, (path.name, rec.get("cell_id"))
    cell = rec.get("cell")
    assert isinstance(cell, dict) and _cell_id(cell) == cid, (path.name, cell)
    q = rec.get("q_of_context")
    assert isinstance(q, list) and len(q) == n_q, (
        f"{path.name}: q_of_context grain "
        f"{len(q) if isinstance(q, list) else type(q).__name__} != {n_q}"
    )
    seeds = rec.get("seeds")
    assert isinstance(seeds, dict) and seeds, (path.name, type(seeds).__name__)
    for seed, sd in seeds.items():
        comps = sd.get("completions")
        assert isinstance(comps, list) and len(comps) == n_q, (
            path.name,
            seed,
            len(comps) if isinstance(comps, list) else comps,
        )
        for ci, draws in enumerate(comps):
            assert isinstance(draws, list) and len(draws) == n_draws, (
                f"{path.name} seed {seed} ctx {ci}: "
                f"{len(draws) if isinstance(draws, list) else draws} draws != {n_draws}"
            )
            empty_slots = [di for di, t in enumerate(draws) if not (isinstance(t, str) and t)]
            assert not empty_slots, (
                f"{path.name} seed {seed} ctx {ci}: empty/non-str completion draw(s) "
                f"{empty_slots} — deterministic per-draw seeds reproduce an empty draw on "
                "re-run (generate_batch has no min_new_tokens floor), so this record wedges "
                "every judge/reduce invocation; re-run the steer phase (its shifted-seed "
                "regen-once escape rewrites the cell) or --force regenerate"
            )
        assert "edit_traces" in sd, (path.name, seed)
    assert "cap_hit_fraction" in rec and "max_new_tokens" in rec, path.name


def _assert_judged_vintage(j: dict, gen_path: Path, expected_judge_fp: str) -> None:
    """Vintage gate on ONE judged checkpoint BEFORE any horizon/bootstrap work
    (r2 Codex C3): the checkpoint must reference the CURRENT gen-file bytes
    (``gen_sha``) and the invocation's judge instrument (``judge_fp``) — a
    regen'd steer record or a re-instrumented judge run is REFUSED (re-run
    the judge phase), never a silent mixed-vintage reduce."""
    cid = j.get("cell_id")
    cur = hashlib.sha256(gen_path.read_bytes()).hexdigest()[:12]
    if j.get("gen_sha") != cur:
        raise RuntimeError(
            f"firstk reduce: judged checkpoint {cid} gen_sha {j.get('gen_sha')} != current "
            f"gen-file sha {cur} — the steer record changed since judging; re-run the judge "
            "phase (mixed vintages refused)"
        )
    if j.get("judge_fp") != expected_judge_fp:
        raise RuntimeError(
            f"firstk reduce: judged checkpoint {cid} judge_fp {j.get('judge_fp')} != the "
            f"invoked instrument {expected_judge_fp} — re-judge at the current instrument "
            "(mixed instruments refused)"
        )


def _validate_judged_record(j: dict, path: Path, *, n_q: int) -> None:
    """Producer-schema contract on ONE judged checkpoint BEFORE the reduce
    consumes it (plan §3): identity, question grain, per-question array
    lengths, rule-29 accounting + coherence fields present."""
    cid = path.stem
    assert j.get("cell_id") == cid, (path.name, j.get("cell_id"))
    assert isinstance(j.get("cell"), dict) and _cell_id(j["cell"]) == cid, path.name
    assert j.get("n_questions") == n_q, (path.name, j.get("n_questions"), n_q)
    for key in ("per_question_mean_score", "per_question_rate"):
        arr = j.get(key)
        assert isinstance(arr, list) and len(arr) == n_q, (path.name, key)
    acct = j.get("accounting")
    assert isinstance(acct, dict) and "frac_items_complete" in acct, path.name
    assert "coherence_pass" in j and "coherence_rate" in j, path.name


def _validate_gen_grid(args, comp_root: Path, expected: set[str], phase: str) -> None:
    """Every expected gen record validated against the invocation's grain
    BEFORE rubric/tokenizer load and BEFORE any API dispatch (plan §6).
    Production refuses a truncated question grain outright (n_q == 20); the
    draws grain is asserted UNIFORM == --draws (a §9 descope to 5 draws stays
    executable — it re-generates, never mixes vintages)."""
    if not args.smoke:
        assert args.q_steer == Q_STEER_DEFAULT, (
            f"{phase}: PRODUCTION question grain {args.q_steer} != {Q_STEER_DEFAULT} — "
            "a truncated grain is a smoke-only slice (plan §6)"
        )
        if args.draws != DRAWS_DEFAULT:
            logger.warning(
                "[%s] production draws=%d != registered %d (plan §9 descope lever?) — "
                "gen records asserted uniform at %d draws",
                phase,
                args.draws,
                DRAWS_DEFAULT,
                args.draws,
            )
    for cid in sorted(expected):
        gp = comp_root / f"{cid}.json"
        _validate_gen_record(json.loads(gp.read_text()), gp, n_q=args.q_steer, n_draws=args.draws)
    logger.info(
        "[%s] gen-record schema validated: %d cells at %dq x %d draws",
        phase,
        len(expected),
        args.q_steer,
        args.draws,
    )


def run_judge_shape_probe(args) -> None:
    """<=5-request LIVE Batch shape probe (plan §7 smoke-gate leg; pod-side
    pre-production — LIVE spend of 5 judge calls): submit 5 items x 1 draw
    through the EXACT production request-construction path
    (``judge_items_graded`` at threshold_base=0, forcing the Batch route —
    the #763 class: a mock smoke cannot catch a malformed Batch request).
    Zero transport losses required; an api-refusal is a judge CONTENT verdict
    (shape reached the model), not a shape failure. Persists the pass
    artifact at judge/shape_probe/shape_probe_pass.json."""
    from explore_persona_space.eval.judge_dispatch import graded_temperature
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
    )
    from explore_persona_space.experiments.issue_1739.judging import (
        judge_items_graded,
        load_trait_rubric,
    )

    rroot = round_root(Path(args.out_root))
    behavior = args.behaviors[0]
    rubric = load_trait_rubric(behavior)
    qs = i2254._eval_questions(behavior)
    probe_texts = (  # benign fixed probe completions — a shape probe, never a science read
        "Thank you for the question. Here is a short, factual, and helpful answer.",
        "I would recommend checking the official documentation for the exact details.",
        "The answer depends on context, but a reasonable default is to decline politely.",
        "Paris is the capital of France; the Seine runs through the city.",
        "A balanced diet and regular exercise are broadly good general advice.",
    )
    items = [(f"shapeprobe-{i:02d}", qs[i % len(qs)], t) for i, t in enumerate(probe_texts)]
    assert len(items) <= 5, len(items)
    probe_dir = rroot / "judge" / "shape_probe"
    with graded_temperature(JUDGE_TEMPERATURE):
        r = judge_items_graded(
            items,
            rubric,
            cache_dir=probe_dir / "cache",
            save_raw=probe_dir / "raw",
            n_draws=1,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=i2254.JUDGE_MAX_TOKENS_2254,
            judge_model=JUDGE_MODEL,
            threshold_base=JUDGE_THRESHOLD_BASE_BATCH,
        )
    n_scored = sum(len(sc) for sc in r.per_item_scores.values())
    assert r.n_transport_lost_draws == 0, (
        f"judge shape probe: {r.n_transport_lost_draws} transport-lost draws — the Batch "
        "request shape failed (#763 class); fix the request shape, do not retry"
    )
    assert n_scored + r.n_refusal_draws + r.n_api_refusal_draws == len(items), (
        n_scored,
        r.n_refusal_draws,
        r.n_api_refusal_draws,
    )
    i2254._write_json_atomic(
        probe_dir / "shape_probe_pass.json",
        i2254._run_metadata(
            {
                "behavior": behavior,
                "n_requests": len(items),
                "n_scored": n_scored,
                "n_refusal_draws": r.n_refusal_draws,
                "n_api_refusal_draws": r.n_api_refusal_draws,
                "n_transport_lost_draws": r.n_transport_lost_draws,
                "transport": "batch (threshold_base=0 pin; <=5-request live shape probe)",
                "judge_model": JUDGE_MODEL,
                "max_tokens": i2254.JUDGE_MAX_TOKENS_2254,
            }
        ),
    )
    logger.info(
        "[%s] judge shape probe PASS: %d/%d scored (0 transport-lost)",
        SENTINEL_JUDGE,
        n_scored,
        len(items),
    )


def phase_judge(args) -> None:
    """Off-pod Batch-API judge wave (plan §6, VM/CPU-only): stage the steer
    phase's per-cell raw completions (a regen'd cell's FULL-LENGTH record
    feeds the judge — §3), grid-completeness + producer-schema validation
    BEFORE any spend, rule-26 pilot per behavior (--pilot stops after the
    gate) with the §7 positive-control EARLY kill (both behaviors fail =>
    halt before the wave), then per-cell judging with fingerprinted per-cell
    checkpoints, rule-28 sync re-issue + rule-29 accounting, and the
    packed-shard HF upload."""
    from explore_persona_space.experiments.issue_1739.judging import load_trait_rubric

    out_root = Path(args.out_root)
    rroot = round_root(out_root)
    _wipe_stale_sentinels([SENTINEL_JUDGE, f"{SENTINEL_JUDGE}-pilot"])
    behaviors = list(args.behaviors)
    _ensure_git_inputs()
    ops = i2254._load_operating_points(INPUTS_ROOT)
    resolved = resolve_operating_points(ops, behaviors)
    cells = build_cells(args, resolved, behaviors)
    rho_pooled, _ = i2254._load_rho(INPUTS_ROOT)
    expected_fp = {_cell_id(c): _steer_regime_fp(args, c, rho_pooled) for c in cells}
    comp_root = _stage_round_completions(rroot, expected_fp)
    expected = set(expected_fp)
    staged = {f.stem for f in comp_root.glob("*.json")}
    missing = sorted(expected - staged)
    if missing:
        raise RuntimeError(
            f"firstk judge: {len(missing)}/{len(expected)} gen cells missing — never a "
            f"partial-grid judge spend (first missing: {missing[:8]})"
        )
    # Producer-schema validation BEFORE rubric load / pilot / any API dispatch.
    _validate_gen_grid(args, comp_root, expected, SENTINEL_JUDGE)
    n_draws = 2 if args.smoke else JUDGE_DRAWS_FIRSTK
    rubrics = {b: load_trait_rubric(b) for b in behaviors}
    # Pilot section (r2 BLOCKER firstk-pilot-pack-reachability): the pilot
    # evidence pack uploads on EVERY exit — pilot-gate FAIL / refusal-rate
    # kill (raises inside _run_firstk_pilot), the §7 positive-control kill,
    # the --pilot return below, AND the success fall-through to the wave
    # (idempotent; the post-wave _upload_judge_outputs_firstk re-covers it).
    pilot_exc: BaseException | None = None
    try:
        for b in behaviors:
            _run_firstk_pilot(args, rroot, comp_root, b, rubrics[b], n_draws)
        # §7 positive-control EARLY kill (mechanical): when EVERY behavior of
        # a multi-behavior run has an assessable rb-vs-random early read and
        # ALL fail (delta <= 0), HALT before any production wave spend (plan
        # §7: halt-and-report, dispatch NO production judge wave).
        # Single-behavior failure stays ADVISORY (logged + persisted in the
        # pass sidecar).
        pc_deltas: dict[str, float | None] = {}
        for b in behaviors:
            pp = rroot / "judge" / "pilot" / f"{b}.pass.json"
            early = (
                json.loads(pp.read_text()).get("positive_control_early") if pp.is_file() else None
            )
            pc_deltas[b] = None if not early else early.get("delta_rb_minus_random")
        assessable = {b: d for b, d in pc_deltas.items() if d is not None}
        if len(assessable) >= 2 and all(d <= 0.0 for d in assessable.values()):
            raise RuntimeError(
                "firstk judge: §7 positive-control EARLY kill — rb-vs-random pilot delta <= 0 "
                f"for EVERY assessable behavior ({assessable}); halting before the production "
                "judge wave (plan §7 halt-and-report)"
            )
    except BaseException as exc:
        pilot_exc = exc
        raise
    finally:
        try:
            _upload_pilot_pack_firstk(rroot)
        except Exception:
            if pilot_exc is None:
                raise
            # Never mask the in-flight pilot kill with an upload error (the
            # firstk-teardown-exception-mask class): log loud, let the
            # original kill propagate.
            logger.exception(
                "[%s] pilot-pack upload failed during pilot-kill unwind "
                "(original error propagates)",
                SENTINEL_JUDGE,
            )
    if args.pilot:
        i2254._write_sentinel(out_root, f"{SENTINEL_JUDGE}-pilot", "done", {"behaviors": behaviors})
        i2254._breadcrumb(SENTINEL_JUDGE, status="pilot-done", behaviors=len(behaviors))
        return
    order = sorted(expected)
    for k, cid in enumerate(order, 1):
        t0 = time.time()
        behavior = cid.split("__", 1)[0]
        _judge_firstk_cell(args, rroot, comp_root / f"{cid}.json", rubrics[behavior], n_draws)
        i2254._progress(SENTINEL_JUDGE, k, len(order), cid, t0)
    _upload_judge_outputs_firstk(rroot)
    i2254._write_sentinel(
        out_root, SENTINEL_JUDGE, "done", {"cells": len(order), "n_draws": n_draws}
    )
    i2254._breadcrumb(SENTINEL_JUDGE, status="done", cells=len(order))


# ---------------------------------------------------------------------------
# unit 2 — reduce phase (plan §3: registered lattice, H3/H4, reads 1+2)
# ---------------------------------------------------------------------------


def _ensure_reduce_git_inputs() -> None:
    """Stage the reduce-only committed parent inputs (cjk_audit + decisive
    percell) via the parent's sparse-checkout-aware fail-loud helper."""
    for rel, cone in REDUCE_GIT_INPUTS:
        i2254._ensure_git_input(rel, cone)


def _horizon_stats_cell(rec: dict, tok, rx) -> dict:
    """Cap-hit + CJK fractions on the COMMON 2048-token horizon (plan §3: a
    regen'd 4096-cap cell's D components are RECOMPUTED from its persisted
    raw completions truncated to 2048 tokens) plus realized-horizon
    diagnostics; per-question arrays feed the paired bootstrap.
    deg = cap-hit fraction + CJK fraction (the §3 sum)."""
    n_q = max(rec["q_of_context"]) + 1
    cnt = np.zeros(n_q)
    cap = np.zeros(n_q)
    cjk = np.zeros(n_q)
    cjk_realized = 0
    total = 0
    for qi, _seed, _ci, _di, text in i2254._iter_gen_qa(rec):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= COMMON_HORIZON_TOKENS:
            t_common = text
        else:
            t_common = tok.decode(ids[:COMMON_HORIZON_TOKENS])
        cnt[qi] += 1
        cap[qi] += len(ids) >= COMMON_HORIZON_TOKENS  # parent _cap_hit_fraction convention
        cjk[qi] += bool(rx.search(t_common))
        cjk_realized += bool(rx.search(text))
        total += 1
    assert total and (cnt > 0).all(), (total, cnt.tolist())
    caphit_q = cap / cnt
    cjk_q = cjk / cnt
    return {
        "caphit_common": float(cap.sum() / total),
        "cjk_common": float(cjk.sum() / total),
        "deg_common": float((cap.sum() + cjk.sum()) / total),
        "caphit_q": caphit_q.tolist(),
        "cjk_q": cjk_q.tolist(),
        "deg_q": (caphit_q + cjk_q).tolist(),
        "cjk_realized": float(cjk_realized / total),
        "caphit_realized_stored": rec.get("cap_hit_fraction"),
        "realized_cap": rec.get("max_new_tokens"),
        "regen": bool(rec.get("regen")),
        "n_completions": int(total),
    }


def _horizon_rows(args, rroot: Path, comp_root: Path, cells: list[dict], rx) -> dict[str, dict]:
    """Per-cell common-horizon stats with a JSONL checkpoint (T2: 160 cells >
    ~50 — code-style intra-phase grain) and a resume key on (cell id, gen-file
    byte sha, regex sha, common horizon) — file-byte hashes are machine-stable
    (never a recomputed-float hash)."""
    from transformers import AutoTokenizer

    ckpt = rroot / "steer" / "horizon_stats.jsonl"
    regex_sha = i2254._sha8(rx.pattern)
    done: dict[tuple[str, str], dict] = {}
    if ckpt.exists() and not args.force:
        for line in ckpt.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("regex_sha") == regex_sha and row.get("common") == COMMON_HORIZON_TOKENS:
                done[(row["cell_id"], row["gen_sha"])] = row
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tok = None
    rows: dict[str, dict] = {}
    for k, cell in enumerate(cells, 1):
        cid = _cell_id(cell)
        gp = comp_root / f"{cid}.json"
        gen_sha = hashlib.sha256(gp.read_bytes()).hexdigest()[:12]
        hit = done.get((cid, gen_sha))
        if hit is not None:
            rows[cid] = hit
            continue
        if tok is None:
            tok = AutoTokenizer.from_pretrained(i2254.MODEL_NAME)
        t0 = time.time()
        row = _horizon_stats_cell(json.loads(gp.read_text()), tok, rx)
        row.update(
            {
                "cell_id": cid,
                "gen_sha": gen_sha,
                "regex_sha": regex_sha,
                "common": COMMON_HORIZON_TOKENS,
            }
        )
        with ckpt.open("a") as fh:  # single-line O_APPEND-style atomic append
            fh.write(json.dumps(row) + "\n")
        rows[cid] = row
        i2254._progress("firstk-horizon", k, len(cells), cid, t0)
    return rows


def _q_from_list(vals) -> np.ndarray:
    """Per-question list (None where every draw dropped) -> NaN float array."""
    return np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)


def _cell_validity(j: dict) -> dict:
    """Per-cell validity gate (plan §6): the coherence-gated score AND the
    rule-29 completeness floor must BOTH hold for a cell to enter the §3
    lattice / the required figures. Invalid cells are excluded (core arm
    invalid => the lattice key reads 'not-computable pending remediation');
    figures filter on the persisted block via ``_row_valid``."""
    fc = j["accounting"]["frac_items_complete"]
    completeness_pass = bool(fc is not None and fc >= COMPLETENESS_FLOOR)
    return {
        "valid": bool(j["coherence_pass"]) and completeness_pass,
        "coherence_pass": bool(j["coherence_pass"]),
        "completeness_pass": completeness_pass,
        "completeness_floor": COMPLETENESS_FLOOR,
    }


def _lattice_block(
    b: str, d: str, breadth: str, arm_q: dict, a0_q: np.ndarray, deg_q: dict, key: str
) -> dict:
    """One §3 registered-lattice cell for (behavior x strong direction x
    breadth): A/S/T paired deltas vs the reused alpha0 floor, the per-resample
    R estimator with the denominator guard, D on the common horizon (declared
    point read, CI alongside), H3 adjacent contrasts + span1-5 plateau, and
    the disjoint verdict label. ONE shared bootstrap index across every
    quantity (paired resamples, plan §6)."""
    nq = len(a0_q)
    idx = i2254._boot_idx(nq, i2254.N_BOOT_VERDICT, key)
    a0_b = np.nanmean(a0_q[idx], axis=1)
    a0_pt = float(np.nanmean(a0_q))

    def _delta_b(pos: str) -> np.ndarray:
        return np.nanmean(arm_q[pos][idx], axis=1) - a0_b

    def _delta_pt(pos: str) -> float:
        return float(np.nanmean(arm_q[pos])) - a0_pt

    def _ci(v: np.ndarray) -> list[float]:
        return [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]

    have = set(arm_q)
    a_b, s_b, t_b = _delta_b("allans"), _delta_b("span13"), _delta_b("tok1")
    out: dict = {
        "behavior": b,
        "direction": d,
        "breadth": breadth,
        "n_questions": nq,
        "n_boot": i2254.N_BOOT_VERDICT,
        "seed_key": key,
        "arms_present": sorted(have),
        "A_allans": {"point": _delta_pt("allans"), "ci": _ci(a_b)},
        "S_span13": {"point": _delta_pt("span13"), "ci": _ci(s_b)},
        "T_tok1": {"point": _delta_pt("tok1"), "ci": _ci(t_b)},
    }
    a_lo = out["A_allans"]["ci"][0]
    s_lo = out["S_span13"]["ci"][0]
    a_pt = out["A_allans"]["point"]
    unstable_frac = float(np.mean(np.abs(a_b) < RATIO_DEN_FLOOR))
    ratio_unstable = bool(unstable_frac > RATIO_UNSTABLE_FRAC)
    out["ratio_guard"] = {
        "denominator_floor": RATIO_DEN_FLOOR,
        "unstable_frac": unstable_frac,
        "ratio_unstable": ratio_unstable,
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        r_b = s_b / a_b
        r1_b = t_b / a_b
    r_pt = float(out["S_span13"]["point"] / a_pt) if abs(a_pt) > 0 else None
    r1_pt = float(out["T_tok1"]["point"] / a_pt) if abs(a_pt) > 0 else None
    if ratio_unstable:
        # Plan §3: R_pt is UNDEFINED under the guard — the registered point is
        # None (figures/narration skip guarded cells); the raw ratio survives
        # only under a clearly-descriptive diagnostic key, never as a point.
        note = "R undefined — >1% of resamples under the denominator floor (§3 guard)"
        out["R"] = {
            "point": None,
            "lo": None,
            "hi": None,
            "note": note,
            "raw_ratio_diagnostic_not_registered": r_pt,
        }
        out["R1"] = {
            "point": None,
            "lo": None,
            "hi": None,
            "note": note,
            "raw_ratio_diagnostic_not_registered": r1_pt,
        }
    else:
        out["R"] = {
            "point": r_pt,
            "lo": float(np.nanquantile(r_b, 0.025)),
            "hi": float(np.nanquantile(r_b, 0.975)),
        }
        out["R1"] = {
            "point": r1_pt,
            "lo": float(np.nanquantile(r1_b, 0.025)),
            "hi": float(np.nanquantile(r1_b, 0.975)),
        }
    fb_b = s_b - R_SUFFICIENT * a_b
    out["fallback_S_minus_two_thirds_A"] = {
        "point": float(out["S_span13"]["point"] - R_SUFFICIENT * a_pt),
        "ci": _ci(fb_b),
        "role": "descriptive fallback (load-bearing when ratio_unstable, §3)",
    }
    d_b = np.nanmean(deg_q["allans"][idx], axis=1) - 2.0 * np.nanmean(deg_q["span13"][idx], axis=1)
    d_pt = float(np.nanmean(deg_q["allans"]) - 2.0 * np.nanmean(deg_q["span13"]))
    d_ci = _ci(d_b)
    out["D"] = {
        "point": d_pt,
        "ci": d_ci,
        "gating": "point (declared, plan §3)",
        "knife_edge": bool(abs(d_pt) < (d_ci[1] - d_ci[0]) / 2.0),
        "deg_allans_common": float(np.nanmean(deg_q["allans"])),
        "deg_span13_common": float(np.nanmean(deg_q["span13"])),
    }
    if "span15" in have:
        s15_pt = _delta_pt("span15")
        r15_pt = float(s15_pt / a_pt) if abs(a_pt) > 0 else None
        with np.errstate(divide="ignore", invalid="ignore"):
            r15_b = _delta_b("span15") / a_b
        if ratio_unstable:  # §3 guard: span1-5 ratio reads undefined too
            out["R_span15"] = {
                "point": None,
                "lo": None,
                "plateau_ge_90pct": None,
                "below_two_thirds": None,
                "note": "R undefined under the §3 denominator guard",
                "raw_ratio_diagnostic_not_registered": r15_pt,
            }
        else:
            out["R_span15"] = {
                "point": r15_pt,
                "lo": float(np.nanquantile(r15_b, 0.025)),
                "plateau_ge_90pct": (None if r15_pt is None else bool(r15_pt >= PLATEAU_FRAC)),
                "below_two_thirds": (None if r15_pt is None else bool(r15_pt < R_SUFFICIENT)),
            }
    else:
        out["R_span15"] = None  # arm absent (smoke slice)
    chain = [p for p in H3_CHAIN if p in have]
    contrasts = []
    for early, late in itertools.pairwise(chain):
        pair_b = np.nanmean(arm_q[late][idx], axis=1) - np.nanmean(arm_q[early][idx], axis=1)
        ci = _ci(pair_b)
        contrasts.append(
            {
                "pair": f"{late}-minus-{early}",
                "point": float(np.nanmean(arm_q[late]) - np.nanmean(arm_q[early])),
                "ci": ci,
                "inverted": bool(ci[1] < 0),
            }
        )
    out["h3_adjacent_contrasts"] = contrasts
    r_lo = out["R"]["lo"]
    if a_lo <= 0:
        label = "all-answer-inert"
    elif s_lo <= 0:
        label = "opening-inert"
    elif ratio_unstable:
        label = "Ambiguous"  # §3 guard: sufficient/partial cannot fire
    elif r_lo is not None and r_lo >= R_SUFFICIENT and d_pt >= 0:
        label = "opening-sufficient"
    elif r_pt is not None and r_pt >= R_PARTIAL and (r_lo < R_SUFFICIENT or d_pt < 0):
        label = "opening-partial"
    else:
        label = "Ambiguous"
    out["verdict"] = label
    return out


def _parent_decisive_key(cell: dict) -> str | None:
    """Parent decisive percell key for the rig-parity advisory (§12.5): our
    allans arm maps to the parent's 'ans' cells, lastctx to 'ctx' — a lookup
    miss (different operating point) simply yields no advisory row."""
    pos = {"allans": "ans", "lastctx": "ctx"}.get(cell["position"])
    if pos is None:
        return None
    return "__".join(
        (
            cell["behavior"],
            i2254._DIR_SHORT[cell["direction"]],
            pos,
            cell["layer_config"],
            i2254._c_token(cell["c"]),
        )
    )


def phase_reduce(args) -> None:
    """Plan §3/§6 reduce (VM/CPU-only, git-committed outputs): per-cell paired
    Δscore CIs vs the REUSED alpha0 floor (1,000 draws), the registered
    verdict lattice + H3 + H4 + positive control (2,000 draws), read 1
    (fraction of donor-swap ceiling) and read 2 (opening vs all-answer with
    common-2048-horizon cap-hit/CJK), the rig-parity advisory vs the parent's
    committed decisive percell, and rule-29 completeness. All bootstraps ride
    the parent's vectorized ``_boot_idx`` fancy-index machinery (seed 20254)
    — never a per-draw Python loop."""
    out_root = Path(args.out_root)
    rroot = round_root(out_root)
    _wipe_stale_sentinels([SENTINEL_REDUCE])
    behaviors = list(args.behaviors)
    _ensure_git_inputs()
    _ensure_reduce_git_inputs()
    ops = i2254._load_operating_points(INPUTS_ROOT)
    resolved = resolve_operating_points(ops, behaviors)
    cells = build_cells(args, resolved, behaviors)
    rho_pooled, _ = i2254._load_rho(INPUTS_ROOT)
    expected_fp = {_cell_id(c): _steer_regime_fp(args, c, rho_pooled) for c in cells}
    comp_root = _stage_round_completions(rroot, expected_fp)
    judged_dir = rroot / "judge" / "judged"
    missing = [c for c in cells if not (judged_dir / f"{_cell_id(c)}.json").is_file()]
    if missing:
        raise RuntimeError(
            f"firstk reduce: {len(missing)}/{len(cells)} judged cells missing — run the "
            f"judge phase first (first missing: {_cell_id(missing[0])})"
        )
    # Producer-schema validation on BOTH input classes BEFORE the tokenizer
    # load / horizon recompute (plan §3/§6 — never a filename-only contract).
    _validate_gen_grid(args, comp_root, {_cell_id(c) for c in cells}, SENTINEL_REDUCE)
    # Judged-vintage gate (r2 Codex C3), BEFORE any horizon/bootstrap work:
    # every judged checkpoint must reference the CURRENT gen-file bytes AND
    # the invocation's judge instrument — mixed vintages refused.
    from explore_persona_space.experiments.issue_1739.judging import load_trait_rubric

    n_draws_judge = 2 if args.smoke else JUDGE_DRAWS_FIRSTK
    expected_jfp = {b: _judge_instrument_fp(load_trait_rubric(b), n_draws_judge) for b in behaviors}
    judged: dict[str, dict] = {}
    judged_sha: dict[str, str] = {}
    for cell in cells:
        cid = _cell_id(cell)
        jp = judged_dir / f"{cid}.json"
        raw = jp.read_bytes()
        j = json.loads(raw)
        _validate_judged_record(j, jp, n_q=args.q_steer)
        _assert_judged_vintage(j, comp_root / f"{cid}.json", expected_jfp[cell["behavior"]])
        judged[cid] = j
        judged_sha[cid] = hashlib.sha256(raw).hexdigest()[:12]
    n_qs = {j["n_questions"] for j in judged.values()}
    assert len(n_qs) == 1, f"non-uniform n_questions across cells: {sorted(n_qs)}"
    n_q = n_qs.pop()

    baseline = json.loads((_REPO_ROOT / BASELINE_PERCELL_REL).read_text())
    audit = json.loads((_REPO_ROOT / CJK_AUDIT_REL).read_text())
    rx = re.compile(audit["regex"])  # the parent's committed intrusion regex (§8)
    decisive = json.loads((_REPO_ROOT / DECISIVE_PERCELL_REL).read_text())
    horizon = _horizon_rows(args, rroot, comp_root, cells, rx)

    a0_arr: dict[str, np.ndarray] = {}
    ceil_arr: dict[str, np.ndarray] = {}
    a0_rate: dict[str, float] = {}
    for b in behaviors:
        base_b = baseline["behaviors"][b]
        a0_full = _q_from_list(base_b["alpha0"]["per_question_mean_score"])
        ceil_full = _q_from_list(base_b["ceiling"]["per_question_mean_score"])
        assert len(a0_full) >= n_q and len(ceil_full) >= n_q, (b, len(a0_full), n_q)
        if len(a0_full) > n_q:  # smoke slice: cells cover a question prefix
            # PRODUCTION never truncates the baseline question grain — a
            # sliced grain silently changes the registered paired estimator
            # (plan §6); only the --smoke bank-order prefix slice may.
            assert args.smoke, (
                f"{b}: baseline arrays ({len(a0_full)} questions) exceed the run's "
                f"n_questions ({n_q}) in PRODUCTION — truncated grain refused"
            )
            logger.info(
                "[%s] %s: truncating alpha0/ceiling arrays %d -> %d questions (smoke)",
                SENTINEL_REDUCE,
                b,
                len(a0_full),
                n_q,
            )
        a0_arr[b] = a0_full[:n_q]
        ceil_arr[b] = ceil_full[:n_q]
        a0_rate[b] = float(base_b["alpha0"]["rate"])

    # Per-cell reduce checkpoint (code-style T2 intra-phase grain), keyed on
    # (judged-file byte sha, baseline bytes, bootstrap regime) — a re-judged
    # cell or changed estimator settings is a cache MISS, never a stale reuse.
    boot_fp = i2254._sha8(
        {
            "n_boot_cell": i2254.N_BOOT_CELL,
            "seed": i2254.BOOTSTRAP_SEED,
            "n_q": int(n_q),
            "completeness_floor": COMPLETENESS_FLOOR,
        }
    )
    baseline_sha = hashlib.sha256((_REPO_ROOT / BASELINE_PERCELL_REL).read_bytes()).hexdigest()[:12]
    percell_ckpt = rroot / "steer" / "percell_rows.jsonl"
    percell_cache: dict[tuple[str, str], dict] = {}
    if percell_ckpt.exists() and not args.force:
        for line in percell_ckpt.open(encoding="utf-8"):  # text-mode, never splitlines
            if not line.strip():
                continue
            crow = json.loads(line)
            if crow.get("boot_fp") == boot_fp and crow.get("baseline_sha") == baseline_sha:
                percell_cache[(crow["cell_id"], crow["judged_sha"])] = crow
    percell_ckpt.parent.mkdir(parents=True, exist_ok=True)

    qindex: dict[tuple[str, str, str, str], np.ndarray] = {}
    degindex: dict[tuple[str, str, str, str], np.ndarray] = {}
    valid_ok: dict[tuple[str, str, str, str], bool] = {}
    percell: dict[str, dict] = {b: {} for b in behaviors}
    foc_rows: dict[str, dict] = {b: {} for b in behaviors}
    parity_rows: dict[str, dict] = {}
    t0_pc = time.time()
    for kx, cell in enumerate(cells, 1):
        cid = _cell_id(cell)
        b = cell["behavior"]
        j = judged[cid]
        cq = i2254._q_arr(j)
        qkey = (b, cell["direction"], cell["breadth"], cell["position"])
        qindex[qkey] = cq
        degindex[qkey] = np.asarray(horizon[cid]["deg_q"], dtype=np.float64)
        hit = percell_cache.get((cid, judged_sha[cid]))
        if hit is not None:
            row, foc, parity = hit["row"], hit["foc"], hit.get("parity")
        else:
            idx = i2254._boot_idx(n_q, i2254.N_BOOT_CELL, f"{cid}__firstk")
            point, lo, hi = i2254._boot_diff_ci(cq, a0_arr[b], idx)
            row = {
                "cell": cell,
                "delta_score": point,
                "ci": [lo, hi],
                "n_boot": i2254.N_BOOT_CELL,
                "mean_score": j["mean_score"],
                "rate": j["rate"],
                # alpha0 lacks per_question_rate in the committed baseline JSON,
                # so the rate delta is POINT-ONLY (no paired CI).
                "delta_rate_point": (None if j["rate"] is None else j["rate"] - a0_rate[b]),
                "coherence_rate": j["coherence_rate"],
                "coherence_pass": j["coherence_pass"],
                "frac_items_complete": j["accounting"]["frac_items_complete"],
                "validity": _cell_validity(j),
                "horizons": {
                    k: horizon[cid][k]
                    for k in (
                        "caphit_common",
                        "cjk_common",
                        "deg_common",
                        "cjk_realized",
                        "caphit_realized_stored",
                        "realized_cap",
                        "regen",
                    )
                },
            }
            parity = None
            pk = _parent_decisive_key(cell)
            prow = decisive["behaviors"].get(b, {}).get(pk) if pk else None
            if prow is not None:
                ci_f = prow["ci_frozen"]
                parity = {
                    "parent_key": pk,
                    "parent_delta_score": prow["delta_score"],
                    "parent_ci_frozen": ci_f,
                    "fresh_delta_score": point,
                    "fresh_within_parent_ci": bool(ci_f[0] <= point <= ci_f[1]),
                    "advisory": True,
                }
                row["rig_parity"] = parity
                logger.info(
                    "[%s] rig-parity ADVISORY %s: fresh d=%.1f vs parent d=%.1f ci=%s (%s)",
                    SENTINEL_REDUCE,
                    cid,
                    point,
                    prow["delta_score"],
                    ci_f,
                    "within" if parity["fresh_within_parent_ci"] else "OUTSIDE",
                )
            # read 1: fraction of the donor-swap ceiling (shared alpha0 anchor).
            a0_b = np.nanmean(a0_arr[b][idx], axis=1)
            num_b = np.nanmean(cq[idx], axis=1) - a0_b
            den_b = np.nanmean(ceil_arr[b][idx], axis=1) - a0_b
            with np.errstate(divide="ignore", invalid="ignore"):
                f_b = num_b / den_b
            den_pt = float(np.nanmean(ceil_arr[b]) - np.nanmean(a0_arr[b]))
            foc = {
                "fraction_point": (point / den_pt) if abs(den_pt) > 0 else None,
                "fraction_ci": [
                    float(np.nanquantile(f_b, 0.025)),
                    float(np.nanquantile(f_b, 0.975)),
                ],
                "denominator_point": den_pt,
                "n_boot": i2254.N_BOOT_CELL,
            }
            with percell_ckpt.open("a") as fh:  # single-line append checkpoint
                fh.write(
                    json.dumps(
                        {
                            "cell_id": cid,
                            "judged_sha": judged_sha[cid],
                            "baseline_sha": baseline_sha,
                            "boot_fp": boot_fp,
                            "row": row,
                            "foc": foc,
                            "parity": parity,
                        }
                    )
                    + "\n"
                )
            i2254._progress(SENTINEL_REDUCE, kx, len(cells), cid, t0_pc)
        if parity is not None:
            parity_rows[cid] = parity
        valid_ok[qkey] = bool(row["validity"]["valid"])
        percell[b][cid] = row
        foc_rows[b][cid] = foc

    lattice: dict[str, dict] = {}
    core = {"allans", "span13", "tok1"}
    for b in behaviors:
        for d in STRONG_DIRECTIONS:
            for breadth in ROUND_BREADTHS:
                present = {p: (b, d, breadth, p) for p in POSITIONS if (b, d, breadth, p) in qindex}
                if not present:
                    continue  # direction/breadth absent from this run (smoke slice)
                lkey = f"{b}__{d}__{breadth}"
                invalid = sorted(p for p, kk in present.items() if not valid_ok[kk])
                if set(invalid) & core:
                    # Validity gate (plan §6): a CORE arm failing the coherence
                    # gate / rule-29 floor makes the registered verdict
                    # non-computable — excluded from figures, remediation named.
                    lattice[lkey] = {
                        "verdict": "not-computable pending remediation",
                        "invalid_arms": invalid,
                        "note": (
                            "validity gate: core arm failed the coherence gate or the "
                            "rule-29 completeness floor — triage the drop class "
                            "(llm-judging rule 29) before this cell re-enters the lattice"
                        ),
                    }
                    continue
                arms = {p: qindex[kk] for p, kk in present.items() if valid_ok[kk]}
                if not core <= set(arms):
                    lattice[lkey] = {
                        "verdict": "not-computable",
                        "missing_arms": sorted(core - set(arms)),
                        "note": "core arm absent (smoke slice or failed cell)",
                    }
                    continue
                degs = {p: degindex[present[p]] for p in arms}
                blk = _lattice_block(
                    b, d, breadth, arms, a0_arr[b], degs, f"{lkey}__firstk_lattice"
                )
                if invalid:  # non-core invalid arms are EXCLUDED, visibly
                    blk["invalid_arms_excluded"] = invalid
                lattice[lkey] = blk

    h4: dict[str, dict] = {}
    for b in behaviors:
        for breadth in ROUND_BREADTHS:
            rows: dict[str, dict] = {}
            for pos in POSITIONS:
                pre_q = qindex.get((b, "pre", breadth, pos))
                shf_q = qindex.get((b, SHUFFLED_DIRECTION, breadth, pos))
                if pre_q is None or shf_q is None:
                    continue
                idx = i2254._boot_idx(
                    n_q, i2254.N_BOOT_VERDICT, f"{b}__{breadth}__{pos}__firstk_h4"
                )
                point, lo, hi = i2254._boot_diff_ci(pre_q, shf_q, idx)
                row = {
                    "pre_minus_preshuf": {"point": point, "ci": [lo, hi]},
                    "clears": bool(lo > 0),
                    "primary": "preshuf (shuffled-map twin, plan §3 H4)",
                }
                rnd_q = qindex.get((b, "random", breadth, pos))
                if rnd_q is not None:
                    rp, rlo, rhi = i2254._boot_diff_ci(pre_q, rnd_q, idx)
                    row["pre_minus_random_diagnostic"] = {"point": rp, "ci": [rlo, rhi]}
                rows[pos] = row
            if not rows:
                continue
            sp = rows.get("span13", {}).get("clears")
            lc = rows.get("lastctx", {}).get("clears")
            verdict = None
            if sp is not None and lc is not None:
                if sp and not lc:
                    verdict = (
                        "locus-dissociation (span1-3 clears the shuffled twin, last-ctx does not)"
                    )
                elif sp and lc:
                    verdict = "no-dissociation (last-ctx also clears the shuffled twin)"
                else:
                    verdict = "opening does not clear the shuffled twin"
            h4[f"{b}__{breadth}"] = {"per_position": rows, "dissociation_verdict": verdict}

    pc: dict = {"behaviors": {}, "note": "plan §7: rb-vs-random all-answer must clear (powered)"}
    for b in behaviors:
        rows = {}
        for breadth in ROUND_BREADTHS:
            rb_q = qindex.get((b, "rb", breadth, "allans"))
            rnd_q = qindex.get((b, "random", breadth, "allans"))
            if rb_q is None or rnd_q is None:
                continue
            idx = i2254._boot_idx(n_q, i2254.N_BOOT_VERDICT, f"{b}__{breadth}__firstk_pc")
            point, lo, hi = i2254._boot_diff_ci(rb_q, rnd_q, idx)
            rows[breadth] = {
                "rb_minus_random_allans": {"point": point, "ci": [lo, hi]},
                "cleared": bool(lo > 0),
            }
        cleared = any(r["cleared"] for r in rows.values()) if rows else None
        pc["behaviors"][b] = {"per_breadth": rows, "cleared": cleared}
    clear_vals = [v["cleared"] for v in pc["behaviors"].values() if v["cleared"] is not None]
    pc["both_behaviors_failed"] = bool(clear_vals) and not any(clear_vals)
    if pc["both_behaviors_failed"]:
        logger.error(
            "[%s] POSITIVE CONTROL FAILED for every behavior with rb/random all-answer "
            "cells — plan §7: halt-and-report, the rig cannot detect the strongest "
            "known effect; downstream verdicts are NOT interpretable",
            SENTINEL_REDUCE,
        )

    reads_dir = rroot / "reads"
    reads_dir.mkdir(parents=True, exist_ok=True)
    # §7 kill enforcement (mechanical): a powered both-behaviors positive-
    # control failure BLOCKS figure-ready outputs — phase_figures refuses
    # while this sentinel exists; a passing re-reduce clears it.
    pc_fail_path = reads_dir / "positive_control_failed.json"
    if pc["both_behaviors_failed"]:
        i2254._write_json_atomic(
            pc_fail_path,
            i2254._run_metadata(
                {
                    "positive_control": pc,
                    "note": (
                        "plan §7 kill: rb-vs-random all-answer failed for EVERY behavior — "
                        "the rig cannot detect the strongest known effect; figures refuse "
                        "until a passing re-reduce clears this sentinel"
                    ),
                }
            ),
        )
    else:
        pc_fail_path.unlink(missing_ok=True)
    boot_meta = {
        "n_boot_cell": i2254.N_BOOT_CELL,
        "n_boot_verdict": i2254.N_BOOT_VERDICT,
        "seed": i2254.BOOTSTRAP_SEED,
        "estimator": "question-level paired cluster bootstrap (parent _boot_idx machinery)",
    }
    i2254._write_json_atomic(
        rroot / "steer" / "delta_score_percell.json",
        i2254._run_metadata(
            {
                "behaviors": percell,
                "completeness": i2254._completeness_block(
                    sorted(judged_dir / f"{_cell_id(c)}.json" for c in cells)
                ),
                "alpha0_source": BASELINE_PERCELL_REL,
                "rig_parity_advisory": parity_rows,
                "boot": boot_meta,
            }
        ),
    )
    i2254._write_json_atomic(
        reads_dir / "verdict_lattice.json",
        i2254._run_metadata(
            {
                "lattice": lattice,
                "h4": h4,
                "positive_control": pc,
                "boot": boot_meta,
                "thresholds": {
                    "R_sufficient": R_SUFFICIENT,
                    "R_partial": R_PARTIAL,
                    "ratio_denominator_floor": RATIO_DEN_FLOOR,
                    "ratio_unstable_frac": RATIO_UNSTABLE_FRAC,
                    "plateau_frac": PLATEAU_FRAC,
                },
                "h3_note": (
                    "adjacent paired-difference contrasts live inside each lattice "
                    "block (4 CIs per chain, no family correction — plan §3)"
                ),
            }
        ),
    )
    i2254._write_json_atomic(
        reads_dir / "fraction_of_ceiling.json",
        i2254._run_metadata(
            {
                "behaviors": foc_rows,
                "denominator": "ceiling - alpha0 (both reused from the committed baseline)",
                "ceiling_source": BASELINE_PERCELL_REL,
                "boot": boot_meta,
            }
        ),
    )
    i2254._write_json_atomic(
        reads_dir / "opening_vs_allanswer.json",
        i2254._run_metadata(
            {
                "common_horizon_tokens": COMMON_HORIZON_TOKENS,
                "cjk_regex_source": CJK_AUDIT_REL,
                "per_cell_horizons": {cid: horizon[cid] for cid in sorted(horizon)},
                "lattice_recovery": {
                    k: {f: v.get(f) for f in ("R", "R1", "R_span15", "D", "ratio_guard", "verdict")}
                    for k, v in lattice.items()
                },
            }
        ),
    )
    i2254._write_sentinel(
        out_root,
        SENTINEL_REDUCE,
        "done",
        {"cells": len(cells), "lattice_cells": len(lattice), "n_questions": n_q},
    )
    i2254._breadcrumb(SENTINEL_REDUCE, status="done", cells=len(cells))


# ---------------------------------------------------------------------------
# phase: figures (off-pod VM CPU; plan §6 heroes + exploratory dump)
# ---------------------------------------------------------------------------


def phase_figures(args) -> None:
    """Plan §6 figures (off-pod CPU): hero 1 (position bars + cap-hit/CJK
    strip), hero 2 (recovery fraction R), and the exploratory dump (accrual
    curves, H3 forest, H4 pre-vs-shuffled, R/D lattice scatter, per-question
    clouds), rendered from the reduce's committed JSONs (+ judged
    per-question arrays) via ``scripts.issue2254_firstk_figures``. PNG +
    .meta.json land under --fig-dir (production:
    figures/issue_2254/<label>/; --smoke rebinds to the scratch out-root so
    smoke renders never touch committed figures)."""
    import scripts.issue2254_firstk_figures as firstk_figs

    out_root = Path(args.out_root)
    rroot = round_root(out_root)
    _wipe_stale_sentinels([SENTINEL_FIGURES])
    # §7 kill enforcement: the reduce's positive-control-failed sentinel
    # BLOCKS figure-ready outputs (plan §7 halt-and-report).
    pc_fail = rroot / "reads" / "positive_control_failed.json"
    if pc_fail.is_file():
        raise RuntimeError(
            f"firstk figures: {pc_fail} present — plan §7 positive-control kill; "
            "figure-ready outputs are blocked until a passing re-reduce clears it"
        )
    fig_dir = (
        Path(args.fig_dir)
        if args.fig_dir
        else _REPO_ROOT / "figures" / "issue_2254" / FOLLOWUP_LABEL
    )
    # Smoke blind-spot (ENUMERATED): the smoke slice (1 behavior x pre x
    # single) cannot render the rb/random-backed views, so `require` narrows
    # to the subset ANY reduce output guarantees — the require MECHANISM
    # itself stays exercised under smoke (a missing required figure raises).
    require = SMOKE_REQUIRED_FIGURES if args.smoke else firstk_figs.REQUIRED_FIGURES
    res = firstk_figs.render_all(rroot, fig_dir, require=require)
    logger.info(
        "[%s] rendered=%s skipped=%s -> %s",
        SENTINEL_FIGURES,
        res["rendered"],
        res["skipped"],
        fig_dir,
    )
    i2254._write_json_atomic(
        fig_dir / "figures_manifest.json",
        i2254._run_metadata({"followup_label": FOLLOWUP_LABEL, **res}),
    )
    i2254._write_sentinel(out_root, SENTINEL_FIGURES, "done", {"rendered": len(res["rendered"])})
    i2254._breadcrumb(SENTINEL_FIGURES, status="done", rendered=len(res["rendered"]))


# ---------------------------------------------------------------------------
# unit-3 CPU smoke (--cpu-smoke): tiny-model hook mechanics + synthetic-
# fixture reduce/figures driven through the REAL phase entrypoints (plan §4.4)
# ---------------------------------------------------------------------------

CPU_SMOKE_SCRATCH = Path("/tmp/issue-2254-firstk-cpusmoke")
TINY_MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"  # cache-resident Qwen2 arch (CPU-loadable)

# Constructed §3 verdict-lattice cases per (behavior, strong direction,
# breadth): CONSTANT per-question deltas make every paired bootstrap CI a
# point, so each registered label fires deterministically (evil's committed
# alpha0 baseline is all-zero => exact arithmetic; sycophancy's nonzero
# baseline leaves >=3-point margins to every threshold). `cjk` marks the
# position arms whose synthetic completions carry CJK text, driving the §3 D
# index through the REAL committed regex + tokenizer path.
_CPU_SMOKE_CASES = {
    ("evil", "rb", "single"): {
        "A": 60.0,
        "S": 50.0,
        "T": 30.0,
        "expect": "opening-sufficient",
        "cjk": ("allans",),
    },
    ("evil", "rb", "mid"): {
        "A": 60.0,
        "S": 30.0,
        "T": 25.0,
        "expect": "opening-partial",  # R = 0.5 < 2/3; D < 0 via span13 CJK
        "cjk": ("span13",),
    },
    ("evil", "pre", "single"): {
        "A": 60.0,
        "S": 45.0,
        "T": 30.0,
        "lastctx": 0.0,  # H4 host: span1-3 clears the shuffled twin, last-ctx does not
        "expect": "opening-sufficient",
    },
    ("evil", "pre", "mid"): {
        "A": 4.0,  # |A_b| < 5-point floor on every resample => ratio guard fires
        "S": 3.0,
        "T": 2.0,
        "lastctx": 0.0,
        "expect": "Ambiguous",
        "guard": True,
    },
    ("sycophancy", "rb", "single"): {"A": -10.0, "S": 5.0, "T": 3.0, "expect": "all-answer-inert"},
    ("sycophancy", "rb", "mid"): {
        "A": 60.0,
        "S": 15.0,
        "T": 10.0,
        "expect": "Ambiguous",  # residual: S_lo > 0 but R_pt = 0.25 < 1/3
        # combined-degradation fixture: cap-hit AND CJK on the SAME arm =>
        # deg == 2.0 exactly (drives the hero-1 strip's full 0-2 stack, M7).
        "capcjk": ("allans",),
    },
    ("sycophancy", "pre", "single"): {
        "A": 60.0,
        "S": -5.0,
        "T": -6.0,
        "lastctx": 20.0,  # H4 no-dissociation: last-ctx ALSO clears its twin
        "expect": "opening-inert",
    },
    ("sycophancy", "pre", "mid"): {
        "A": 60.0,
        "S": 45.0,
        "T": 50.0,  # tok1 > span1-3 => the span13-minus-tok1 contrast INVERTS
        "lastctx": 0.0,
        "expect": "opening-sufficient",
    },
}
_CPU_SMOKE_CTXEXT = {
    "lastctx": 3.0,
    "tok1": 10.0,
    "tok2": 9.0,
    "tok3": 8.0,
    "span13": 18.0,
    "span15": 22.0,
    "combined": 19.0,
    "allans": 25.0,
}
# random all-answer deltas per the §7 positive-control construction: evil
# clears (rb 60 vs 5 at both breadths); sycophancy does not (single: rb -10
# vs 0; mid: rb 60 vs 70) => cleared flags True/False, both_behaviors_failed False.
_CPU_SMOKE_RANDOM_ALLANS = {
    ("evil", "single"): 5.0,
    ("evil", "mid"): 5.0,
    ("sycophancy", "single"): 0.0,
    ("sycophancy", "mid"): 70.0,
}
_CPU_SMOKE_H4 = {  # expected dissociation-verdict prefix per h4 key
    "evil__single": "locus-dissociation",
    "evil__mid": "locus-dissociation",
    "sycophancy__single": "no-dissociation",
    "sycophancy__mid": "locus-dissociation",
}
_CPU_SMOKE_TEXT = "The capital of France is Paris, a short factual answer for the fixture."
_CPU_SMOKE_TEXT_CJK = _CPU_SMOKE_TEXT + " 好的谢谢"
# Cap-hit AND CJK on the COMMON horizon: CJK leads (inside the first 2048
# tokens); >2048 repeated tokens trip the parent cap-hit convention.
_CPU_SMOKE_TEXT_CAPCJK = "好的谢谢 " + ("word " * 2100)


def _cpu_smoke_delta(cell: dict) -> float:
    """Constructed per-question Δ (vs the reused alpha0) for one fixture cell."""
    b, d, br, pos = cell["behavior"], cell["direction"], cell["breadth"], cell["position"]
    if d in STRONG_DIRECTIONS:
        c = _CPU_SMOKE_CASES[(b, d, br)]
        a, s, t = c["A"], c["S"], c["T"]
        return {
            "lastctx": c.get("lastctx", 5.0),
            "tok1": t,
            "tok2": t - 2.0,
            "tok3": t - 4.0,
            "span13": s,
            "span15": (s + a) / 2.0,
            "combined": s + 2.0,
            "allans": a,
        }[pos]
    if d == "ctxext":
        return _CPU_SMOKE_CTXEXT[pos]
    if d == "random":
        return _CPU_SMOKE_RANDOM_ALLANS[(b, br)] if pos == "allans" else 2.0
    assert d == SHUFFLED_DIRECTION, d
    if b == "sycophancy" and br == "single" and pos == "span13":
        return -20.0  # syco-single pre (-5) still clears its twin (-20): diff +15
    return 0.0


def _cpu_smoke_hooks(args) -> dict:
    """Leg (a): ``WindowedDeltaHook`` + ``EditPositionRecorder`` mechanics on
    a tiny cached HF model (CPU), through the PRODUCTION ``generate_batch`` +
    ``build_recorded_hook`` path, for all 8 position arms at 1 layer plus a
    2-layer stack leg — count + position-identity asserted per draw via
    ``assert_cell_edit_traces`` (plan §4.4 read-backs, trace persisted)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.issue1415 import steering

    name = args.tiny_model
    logger.info("[cpu-smoke] loading tiny model %s (cpu, fp32)", name)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    blocks, _, _ = _resolve_decoder_blocks(model)
    assert blocks is not None, f"{name}: no standard decoder blocks"
    n_blocks = len(blocks)
    hidden = int(model.config.hidden_size)
    gen = torch.Generator().manual_seed(2254)

    def unit_dir() -> torch.Tensor:
        v = torch.randn(hidden, generator=gen)
        return v / v.norm()

    contexts = [
        {"system": None, "user": "Count upward from one to forty, separated by commas."},
        {"system": None, "user": "List the days of the week, then the months of the year."},
    ]
    arms = [(pos, 1) for pos in POSITIONS] + [("lastctx", 2), ("span13", 2), ("allans", 2)]
    n_draws, max_new = 2, 8
    rows: list[dict] = []
    for pos, k in arms:
        layers = [min(2, n_blocks - 1)] if k == 1 else [min(2, n_blocks - 2), min(4, n_blocks - 1)]
        dirs = [unit_dir() for _ in range(k)]
        alphas = [4.0] * k
        hook = build_recorded_hook(model, pos, layers, dirs, alphas)
        with hook:
            res = steering.generate_batch(
                model,
                tok,
                contexts,
                n=n_draws,
                hook=hook,
                max_new_tokens=max_new,
                temperature=1.0,
                seed_base=SEED_BASE_DEFAULT,
            )
        traces = hook.draw_traces
        assert len(traces) == n_draws and len(res) == len(contexts), (len(traces), len(res))
        # Windowed identity needs the full window realized (batched decode
        # length = max over the 2 rows); span1-5 needs >= 5 decode steps.
        min_nd = min(t["n_forwards"] - 1 for t in traces)
        assert min_nd >= 5, f"{pos}: batch decode length {min_nd} < 5 — pick a longer prompt"
        rec = {
            "expected_edit_profile": expected_edit_profile(pos, k),
            "hook_impl": {"n_layers": k},
            "seeds": {str(SEED_BASE_DEFAULT): {"edit_traces": traces}},
        }
        check = assert_cell_edit_traces(rec)
        children = hook._steer_children()
        if pos in ("lastctx", "allans"):  # comparators byte-reuse the BASE class (§4.2)
            assert all(type(c) is DeltaHook for c in children), pos
        else:
            assert all(type(c) is WindowedDeltaHook for c in children), pos
        rows.append(
            {
                "position": pos,
                "n_layers": k,
                "layers": layers,
                "steer_hook_class": type(hook.steer).__name__,
                "child_hook_class": type(children[0]).__name__,
                "coord_source": hook.recorder.coord_source,
                "edit_trace_check": check,
                "draw_traces": traces,  # persisted positions trace (plan §4.4)
            }
        )
        logger.info(
            "[cpu-smoke] arm %s k=%d: %d draws OK (coord_source=%s)",
            pos,
            k,
            n_draws,
            hook.recorder.coord_source,
        )
    return {
        "tiny_model": name,
        "hidden_size": hidden,
        "n_blocks": n_blocks,
        "n_draws": n_draws,
        "max_new_tokens": max_new,
        "n_contexts": len(contexts),
        "arms": rows,
    }


def _cpu_smoke_write_fixtures(sub, rroot: Path) -> list[dict]:
    """Synthesize gen + judged fixtures for the FULL 160-cell grid off the
    committed operating points + alpha0 baseline (deterministic). Fixtures
    carry the PRODUCTION provenance fields — regime_fp (staging fp
    cross-check), gen_sha + judge_fp (reduce vintage gate) — so the real
    phase_reduce's r3 validators run at the exact production contract."""
    from explore_persona_space.experiments.issue_1739.judging import load_trait_rubric

    ops = i2254._load_operating_points(INPUTS_ROOT)
    behaviors = list(ROUND_BEHAVIORS)
    cells = build_cells(sub, resolve_operating_points(ops, behaviors), behaviors)
    assert len(cells) == TOTAL_CELLS, len(cells)
    rho_pooled, _ = i2254._load_rho(INPUTS_ROOT)
    # sub.smoke is False (production-contract fixtures) -> the reduce expects
    # the production judge-draw grain.
    jfp = {b: _judge_instrument_fp(load_trait_rubric(b), JUDGE_DRAWS_FIRSTK) for b in behaviors}
    baseline = json.loads((_REPO_ROOT / BASELINE_PERCELL_REL).read_text())
    a0: dict[str, list[float]] = {}
    for b in behaviors:
        vals = baseline["behaviors"][b]["alpha0"]["per_question_mean_score"]
        assert all(v is not None for v in vals), f"{b}: alpha0 has null per-question entries"
        a0[b] = [float(v) for v in vals]
    n_q = Q_STEER_DEFAULT
    comp_root = rroot / "steer" / "raw_completions"
    judged_dir = rroot / "judge" / "judged"
    for cell in cells:
        cid = _cell_id(cell)
        b = cell["behavior"]
        d = _cpu_smoke_delta(cell)
        case = _CPU_SMOKE_CASES.get((b, cell["direction"], cell["breadth"]), {})
        if cell["position"] in case.get("capcjk", ()):
            text = _CPU_SMOKE_TEXT_CAPCJK  # cap-hit + CJK combined (M7 strip stack)
        elif cell["position"] in case.get("cjk", ()):
            text = _CPU_SMOKE_TEXT_CJK
        else:
            text = _CPU_SMOKE_TEXT
        per_q = [a0[b][q] + d for q in range(n_q)]
        # Fixtures at PRODUCTION grain (n_q=20, DRAWS_DEFAULT=6 draws) so the
        # real phase_reduce's producer-schema validators run at the exact
        # production contract (never a fixture-only relaxed shape).
        gp = comp_root / f"{cid}.json"
        i2254._write_json_atomic(
            gp,
            {
                "cell_id": cid,
                "cell": cell,
                "q_of_context": list(range(n_q)),
                "seeds": {
                    str(SEED_BASE_DEFAULT): {
                        "completions": [[text] * DRAWS_DEFAULT for _ in range(n_q)],
                        "coherent_flags": [[True] * DRAWS_DEFAULT for _ in range(n_q)],
                        "condition_passes": [True] * n_q,
                        "edit_traces": [],
                    }
                },
                "max_new_tokens": i2254.GEN_MAX_NEW_TOKENS,
                "cap_hit_fraction": 0.0,
                "regen": False,
                "regime_fp": _steer_regime_fp(sub, cell, rho_pooled),
                "synthetic_fixture": True,
            },
        )
        gen_sha = hashlib.sha256(gp.read_bytes()).hexdigest()[:12]
        i2254._write_json_atomic(
            judged_dir / f"{cid}.json",
            {
                "cell_id": cid,
                "cell": cell,
                "n_questions": n_q,
                "gen_sha": gen_sha,
                "judge_fp": jfp[b],
                "per_question_mean_score": per_q,
                "per_question_rate": [0.5] * n_q,
                "mean_score": float(np.mean(per_q)),
                "rate": 0.5,
                "coherence_rate": 1.0,
                "coherence_pass": True,
                "accounting": {"frac_items_complete": 1.0},
                "synthetic_fixture": True,
            },
        )
    return cells


def _cpu_smoke_reduce_figures(args) -> dict:
    """Legs (b)+(c): the REAL ``phase_reduce`` (twice — the second run
    exercises the horizon-ckpt resume) then the REAL ``phase_figures`` on a
    synthetic full-grid fixture; every constructed §3 label, the denominator
    guard, the constructed H3 inversions, the H4 verdicts, the §7 positive
    control, and per-figure pixel ink are ASSERTED."""
    scratch = CPU_SMOKE_SCRATCH
    if scratch.exists():
        shutil.rmtree(scratch)
    sub = argparse.Namespace(
        **{
            **vars(args),
            "out_root": str(scratch),
            "behaviors": list(ROUND_BEHAVIORS),
            "smoke": False,
            "force": False,
            "q_steer": Q_STEER_DEFAULT,
            "draws": DRAWS_DEFAULT,
            "fig_dir": str(scratch / "figures"),
        }
    )
    rroot = round_root(Path(sub.out_root))
    _cpu_smoke_write_fixtures(sub, rroot)
    phase_reduce(sub)
    lat_path = rroot / "reads" / "verdict_lattice.json"
    lat1 = json.loads(lat_path.read_text())

    def _jsonl_rows(p: Path) -> int:
        return sum(1 for ln in p.read_text().split("\n") if ln.strip())

    ckpt = rroot / "steer" / "horizon_stats.jsonl"
    n_ckpt_rows = _jsonl_rows(ckpt)
    assert n_ckpt_rows == TOTAL_CELLS, n_ckpt_rows
    pk_ckpt = rroot / "steer" / "percell_rows.jsonl"
    n_pk_rows = _jsonl_rows(pk_ckpt)
    assert n_pk_rows == TOTAL_CELLS, n_pk_rows
    phase_reduce(sub)  # resume leg: horizon + percell ckpts cache-hit, no recompute rows
    assert _jsonl_rows(ckpt) == n_ckpt_rows, "resume appended horizon rows"
    assert _jsonl_rows(pk_ckpt) == n_pk_rows, "resume appended percell rows"
    lat2 = json.loads(lat_path.read_text())
    for k in ("lattice", "h4", "positive_control"):
        assert lat1[k] == lat2[k], f"reduce resume changed {k}"
    out: dict = {
        "lattice": {},
        "h4": {},
        "resume_recheck": "horizon + percell ckpt cache-hit; verdicts identical",
    }
    for (b, d, br), case in sorted(_CPU_SMOKE_CASES.items()):
        key = f"{b}__{d}__{br}"
        blk = lat1["lattice"][key]
        assert blk["verdict"] == case["expect"], (key, blk["verdict"], case["expect"])
        assert blk["ratio_guard"]["ratio_unstable"] is bool(case.get("guard", False)), key
        if case.get("guard"):
            # Guarded ratio => registered points are None (raw ratio only under
            # the explicit *_diagnostic_not_registered key) — plan §3 UNDEFINED.
            assert blk["R"]["point"] is None and blk["R"]["lo"] is None, (key, blk["R"])
            assert blk["R1"]["point"] is None, (key, blk["R1"])
            assert blk["R_span15"]["point"] is None, (key, blk["R_span15"])
            assert "raw_ratio_diagnostic_not_registered" in blk["R"], key
        if case.get("capcjk") == ("allans",):
            # Cap-hit AND CJK both fire on allans => degraded fraction stacks to 2.0.
            assert abs(blk["D"]["deg_allans_common"] - 2.0) < 1e-9, (key, blk["D"])
        if case.get("cjk") == ("allans",):
            assert blk["D"]["point"] > 0.5, (key, blk["D"])  # allans degraded => D >> 0
        if case.get("cjk") == ("span13",):
            assert blk["D"]["point"] < -0.5, (key, blk["D"])  # span13 degraded => D << 0
        chain_vals = [
            _cpu_smoke_delta({"behavior": b, "direction": d, "breadth": br, "position": p})
            for p in H3_CHAIN
        ]
        expected_inv = [
            bool(late - early < -1e-9) for early, late in itertools.pairwise(chain_vals)
        ]
        observed_inv = [c["inverted"] for c in blk["h3_adjacent_contrasts"]]
        assert observed_inv == expected_inv, (key, observed_inv, expected_inv)
        out["lattice"][key] = {
            "verdict": blk["verdict"],
            "expected": case["expect"],
            "ratio_unstable": blk["ratio_guard"]["ratio_unstable"],
            "h3_inverted": observed_inv,
            "D_point": blk["D"]["point"],
            "R_point": blk["R"]["point"],
        }
    for key, exp_prefix in _CPU_SMOKE_H4.items():
        verdict = lat1["h4"][key]["dissociation_verdict"]
        assert verdict is not None and verdict.startswith(exp_prefix), (key, verdict, exp_prefix)
        out["h4"][key] = {"verdict": verdict, "expected_prefix": exp_prefix}
    pc = lat1["positive_control"]
    assert pc["behaviors"]["evil"]["cleared"] is True, pc["behaviors"]["evil"]
    assert pc["behaviors"]["sycophancy"]["cleared"] is False, pc["behaviors"]["sycophancy"]
    assert pc["both_behaviors_failed"] is False, pc
    out["positive_control"] = {
        "evil_cleared": True,
        "sycophancy_cleared": False,
        "both_behaviors_failed": False,
    }
    # Leg (c): the REAL figures phase on the synthetic reduce output.
    import matplotlib.pyplot as plt

    import scripts.issue2254_firstk_figures as firstk_figs

    # §7 positive-control kill: figures must REFUSE while the reduce-written
    # sentinel is present (probed with a synthetic sentinel, then removed).
    pc_fail = rroot / "reads" / "positive_control_failed.json"
    assert not pc_fail.is_file(), pc_fail
    pc_fail.write_text("{}")
    try:
        phase_figures(sub)
    except RuntimeError as exc:
        assert "positive_control_failed" in str(exc), exc
    else:
        raise AssertionError("phase_figures did not refuse on positive_control_failed sentinel")
    pc_fail.unlink()
    out["figures_pc_refusal"] = "RuntimeError raised on sentinel; sentinel removed; real render ran"

    phase_figures(sub)
    figs: dict[str, dict] = {}
    for name in firstk_figs.REQUIRED_FIGURES:
        p = Path(sub.fig_dir) / f"{name}.png"
        assert p.is_file(), p
        img = plt.imread(p)
        ink = float((img[..., :3].min(axis=-1) < 0.85).mean())
        std = float(img[..., :3].std())
        assert std > 0.01 and ink > 0.01, (name, std, ink)
        figs[name] = {
            "pixel_std": round(std, 4),
            "ink_fraction": round(ink, 4),
            "bytes": p.stat().st_size,
        }
    out["figures"] = figs
    out["scratch_root"] = str(scratch)
    return out


def run_cpu_smoke(args) -> None:
    """Unit-3 CPU smoke (``--cpu-smoke``; VM, no GPU, no API spend):
    (a) tiny-model hook mechanics through the production
    ``generate_batch`` + ``build_recorded_hook`` path; (b) the REAL
    ``phase_reduce`` on a synthetic full-grid fixture, twice (resume leg);
    (c) the REAL ``phase_figures`` on that output. Evidence JSONs land under
    ``--cpu-smoke-out`` (committed smoke artifacts, plan §4.4 trace
    persistence). The judge ``--dry-run`` bind leg is a separate command;
    the ≤5-request live Batch probe + the real-7B smoke stay POD-SIDE."""
    t0 = time.time()
    hooks_out = _cpu_smoke_hooks(args)
    reduce_out = _cpu_smoke_reduce_figures(args)
    out_dir = Path(args.cpu_smoke_out)
    i2254._write_json_atomic(out_dir / "cpu_hook_mechanics.json", i2254._run_metadata(hooks_out))
    i2254._write_json_atomic(out_dir / "cpu_reduce_figures.json", i2254._run_metadata(reduce_out))
    i2254._breadcrumb(
        "firstk-cpu-smoke",
        status="done",
        arms=len(hooks_out["arms"]),
        lattice=len(reduce_out["lattice"]),
        figures=len(reduce_out["figures"]),
        elapsed=f"{time.time() - t0:.0f}s",
    )


PHASES = {
    "stage_inputs": phase_stage_inputs,
    "steer": phase_steer,
    "judge": phase_judge,
    "reduce": phase_reduce,
    "figures": phase_figures,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="issue #2254 follow-up: first-k answer-token steering (plan v10)"
    )
    ap.add_argument(
        "--phases",
        default=None,
        help=(
            "comma-separated phases to run in order "
            "(stage_inputs,steer,judge,reduce,figures — plan §9 workload cmd)"
        ),
    )
    ap.add_argument("--behaviors", nargs="+", default=list(ROUND_BEHAVIORS))
    ap.add_argument(
        "--out-root",
        default="eval_results/issue_2254",
        help=(
            "ISSUE out-root (parent convention); round outputs land under "
            f"<out-root>/{FOLLOWUP_LABEL}/ — reused inputs always resolve at the "
            "canonical committed locations, independent of this flag"
        ),
    )
    ap.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="round-robin cell shard for the multi-GPU fan-out (launcher pins CVD per shard)",
    )
    ap.add_argument("--num-shards", type=int, default=1, help="total shards (plan §9: 4)")
    ap.add_argument(
        "--q-steer",
        type=int,
        default=Q_STEER_DEFAULT,
        help="eval questions per cell (plan §4.4: 20)",
    )
    ap.add_argument(
        "--draws",
        type=int,
        default=DRAWS_DEFAULT,
        help="draws per question at one seed_base (plan §4.4: 6 -> per-draw seeds 42-47)",
    )
    ap.add_argument(
        "--seed-base",
        type=int,
        default=SEED_BASE_DEFAULT,
        help="generation seed base (per-draw seed = seed_base + draw index)",
    )
    ap.add_argument("--force", action="store_true", help="ignore per-cell checkpoint caches")
    ap.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "judge phase only: run the rule-26 pilot gate (>=55 effective draws x 4 arms "
            "per behavior at the production instrument + Batch transport) and STOP before "
            "the bulk wave (plan §6)"
        ),
    )
    ap.add_argument(
        "--waive-judge-parse-fail-arms",
        nargs="*",
        default=[],
        help=(
            "rule 26(b) explained-content-drop escape: pilot arm names whose parse-fail "
            "check is waived (truncation FAIL stays unwaivable inside judge_pilot)"
        ),
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "tiny slice: 1 behavior x pre x single x 5 positions, 2q x 2 draws; "
            "scratch out-root + smoke/ HF sub-prefix (inputs stay canonical)"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate the phase grid + resolve deferred imports, no GPU/HF/model",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="AST arg-attribute completeness + helper-call bind check, then exit 0",
    )
    ap.add_argument(
        "--fig-dir",
        default=None,
        help=(
            "figures output dir (default figures/issue_2254/"
            f"{FOLLOWUP_LABEL}/; --smoke rebinds it to the scratch out-root)"
        ),
    )
    ap.add_argument(
        "--cpu-smoke",
        action="store_true",
        help=(
            "unit-3 VM smoke (no GPU / no API spend): tiny-model hook mechanics + "
            "synthetic-fixture reduce/figures through the real phase entrypoints"
        ),
    )
    ap.add_argument(
        "--tiny-model",
        default=TINY_MODEL_DEFAULT,
        help="cached tiny CausalLM for the --cpu-smoke hook-mechanics leg (CPU)",
    )
    ap.add_argument(
        "--cpu-smoke-out",
        default=str(_REPO_ROOT / "eval_results" / "issue_2254" / FOLLOWUP_LABEL / "smoke"),
        help="evidence dir for --cpu-smoke summaries (committed; plan §4.4 trace persistence)",
    )
    ap.add_argument(
        "--judge-shape-probe",
        action="store_true",
        help=(
            "POD-SIDE pre-production probe: submit <=5 real requests through the exact "
            "production judge path (Batch route, production rubric shape) and persist a "
            "pass artifact under judge/shape_probe/ — live API spend, run before phase_judge"
        ),
    )
    return ap


def _apply_smoke(args) -> None:
    """Tiny-real slice (plan §4.4 smoke): 1 behavior, 2 questions x 2 draws,
    scratch out-root + smoke/ HF sub-prefix so smoke OUTPUTS never overwrite
    canonical ones; reused INPUTS stay canonical (module constants)."""
    args.behaviors = args.behaviors[:1]
    args.q_steer = 2
    args.draws = 2
    if args.out_root == "eval_results/issue_2254":
        args.out_root = "/tmp/issue-2254-firstk-smoke"
    if args.fig_dir is None:  # smoke figures never touch committed figures/
        args.fig_dir = str(Path(args.out_root) / "figures")
    i2254._SMOKE_UPLOAD_SUBPREFIX = True


def _dry_run_phase(args, phase: str) -> None:
    """Enumerate the phase's grid + RESOLVE its deferred imports (no GPU/HF/
    model): a missing symbol / signature drift in a pod-only branch must fail
    HERE, not after the expensive phases (#606/#823/#1332)."""
    if phase == "stage_inputs":
        from explore_persona_space.experiments.issue_1739 import store_io
        from explore_persona_space.orchestrate import hub

        assert callable(store_io.load_rb_bank)
        assert callable(hub.stage_hub_file)
        assert callable(i2254._stage_e1_assets)
        assert callable(i2254.random_direction)
        i2254._breadcrumb(SENTINEL_STAGE, dry_run=1, behaviors=len(args.behaviors))
    elif phase == "steer":
        import inspect

        from explore_persona_space.experiments.issue1415 import steering
        from explore_persona_space.orchestrate import hub

        # Signature-bind the reused call shapes (the #1332 pattern).
        inspect.signature(steering.generate_batch).bind(
            None, None, [], n=6, hook=None, max_new_tokens=2048, temperature=1.0, seed_base=42
        )
        inspect.signature(multi_layer_delta_hooks).bind(
            None, [14], [None], [0.0], all_positions=True
        )
        assert callable(hub.retry_transient) and callable(hub._upload)
        # Grid enumeration off the COMMITTED operating points (local read only).
        ops_path = INPUTS_ROOT / "localize" / "operating_points.json"
        assert ops_path.is_file(), (
            f"{ops_path} missing — run from a checkout carrying the parent's "
            "committed eval_results (issue-2254 branch), or stage_inputs first"
        )
        ops = i2254._load_operating_points(INPUTS_ROOT)
        resolved = resolve_operating_points(ops, list(args.behaviors))
        cells = build_cells(args, resolved, list(args.behaviors))
        i2254._breadcrumb(SENTINEL_STEER, dry_run=1, cells=len(cells))
    elif phase == "judge":
        import inspect

        import scripts.issue2220_readwrite as rw2220
        from explore_persona_space.eval.judge_dispatch import graded_temperature
        from explore_persona_space.eval.judge_pilot import _seeded_subsample, judge_pilot_gate
        from explore_persona_space.experiments.issue_1739.constants import JUDGE_MODEL
        from explore_persona_space.experiments.issue_1739.judging import (
            judge_items_graded,
            judge_tallies,
            load_trait_rubric,
            rollout_item_id,
        )

        # Signature-bind the exact production call shapes (the #1332 pattern).
        inspect.signature(judge_items_graded).bind(
            [],
            "rubric",
            cache_dir=Path("."),
            save_raw=Path("."),
            n_draws=JUDGE_DRAWS_FIRSTK,
            temperature=1.0,
            max_tokens=i2254.JUDGE_MAX_TOKENS_2254,
            judge_model=JUDGE_MODEL,
            threshold_base=JUDGE_THRESHOLD_BASE_BATCH,
        )
        inspect.signature(judge_pilot_gate).bind(
            {},
            "rubric",
            max_tokens=i2254.JUDGE_MAX_TOKENS_2254,
            cache_dir=Path("."),
            save_raw_dir=Path("."),
            n_draws=JUDGE_DRAWS_FIRSTK,
            target_total_draws=220,
            min_effective_draws_per_arm=PILOT_MIN_EFFECTIVE_FIRSTK,
            waive_parse_fail_arms=(),
            allow_subresolution_pilot=False,
            threshold_base=JUDGE_THRESHOLD_BASE_BATCH,
            report_path=Path("."),
            seed=0,
        )
        inspect.signature(_seeded_subsample).bind([], 1, seed=0, arm="a")
        inspect.signature(rw2220._pack_tree_to_jsonl_shards).bind(
            Path("."), Path("."), group="g", pattern="*"
        )
        assert callable(load_trait_rubric) and callable(judge_tallies)
        assert callable(rollout_item_id) and callable(graded_temperature)
        assert callable(i2254._upload_folder_to_hf) and callable(i2254._eval_questions)
        # Judge-grid enumeration off the COMMITTED operating points + the
        # item-id budget for every position token (local reads only).
        ops = i2254._load_operating_points(INPUTS_ROOT)
        cells = build_cells(
            args, resolve_operating_points(ops, list(args.behaviors)), list(args.behaviors)
        )
        for cell in cells:
            _judge_ctx_id_firstk(cell, SEED_BASE_DEFAULT + DRAWS_DEFAULT - 1, 119)
        i2254._breadcrumb(SENTINEL_JUDGE, dry_run=1, cells=len(cells))
    elif phase == "reduce":
        import inspect

        from transformers import AutoTokenizer

        assert callable(AutoTokenizer.from_pretrained)
        assert callable(i2254._completeness_block) and callable(i2254._q_arr)
        inspect.signature(i2254._boot_idx).bind(20, i2254.N_BOOT_VERDICT, "k")
        inspect.signature(i2254._boot_diff_ci).bind(None, None, None)
        inspect.signature(i2254._ensure_git_input).bind("rel", "cone")
        # Committed reduce inputs present + regex compiles (local reads only —
        # no HF, no tokenizer download, no judged-artifact reads).
        for rel, _cone in ((BASELINE_PERCELL_REL, ""),) + REDUCE_GIT_INPUTS:
            p = _REPO_ROOT / rel
            assert p.is_file(), f"{p} missing — committed parent input (plan §10)"
        re.compile(json.loads((_REPO_ROOT / CJK_AUDIT_REL).read_text())["regex"])
        ops = i2254._load_operating_points(INPUTS_ROOT)
        cells = build_cells(
            args, resolve_operating_points(ops, list(args.behaviors)), list(args.behaviors)
        )
        i2254._breadcrumb(SENTINEL_REDUCE, dry_run=1, cells=len(cells))
    elif phase == "figures":
        import inspect

        import scripts.issue2254_firstk_figures as firstk_figs

        inspect.signature(firstk_figs.render_all).bind(Path("."), Path("."), require=())
        assert len(firstk_figs.REQUIRED_FIGURES) == 7, firstk_figs.REQUIRED_FIGURES
        assert callable(firstk_figs.fig_hero1_position_bars)
        assert firstk_figs.BASELINE_PERCELL.is_file(), firstk_figs.BASELINE_PERCELL
        i2254._breadcrumb(SENTINEL_FIGURES, dry_run=1, figures=len(firstk_figs.REQUIRED_FIGURES))
    else:  # unreachable behind the main() phase validation; keep fail-loud
        raise SystemExit(f"dry-run: no wiring branch for phase {phase!r}")
    print(f"[dry-run] {phase} wiring OK", flush=True)


def main() -> None:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.cpu_smoke:
        run_cpu_smoke(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    if args.judge_shape_probe:
        run_judge_shape_probe(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    if not args.phases:
        raise SystemExit(
            "--phases is required (comma-separated: stage_inputs,steer,judge,reduce,figures) "
            "or --import-check / --cpu-smoke / --judge-shape-probe"
        )
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        raise SystemExit(f"unknown phase(s) {unknown}; choices: {sorted(PHASES)}")
    if args.smoke:
        _apply_smoke(args)
    if args.dry_run:
        for p in phases:
            _dry_run_phase(args, p)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    for p in phases:
        PHASES[p](args)
    # Explicit hard-exit after flush: this driver imports torch/transformers/HF,
    # so a finalize-time teardown race can rewrite the rc (gotchas.md). Outputs
    # are rename-atomic and uploaded before here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
