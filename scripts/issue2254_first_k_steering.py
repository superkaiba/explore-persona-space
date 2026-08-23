"""Issue #2254 follow-up `first-k-answer-token-steering` — position-cells driver.

Plan v10 (tasks/.../2254/plans/v10.md). Unit 1/3: the `stage_inputs` + `steer`
phases (judge/reduce = unit 2; figures + smoke assertions = unit 3).

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
import logging
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

    def __exit__(self, *exc) -> None:
        self.remove()

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


def _assert_hook_types(cell: dict, steer) -> None:
    """Comparator arms byte-reuse the BASE ``DeltaHook``; windowed arms the
    subclass (plan §12.3 / the §7 smoke-gate kill criterion) — asserted in
    production too, per constructed hook."""
    children = steer.hooks if isinstance(steer, MultiLayerDeltaHook) else [steer]
    if cell["position"] in ("lastctx", "allans"):
        bad = [type(h).__name__ for h in children if type(h) is not DeltaHook]
        assert not bad, (cell["position"], bad)
    else:
        bad = [type(h).__name__ for h in children if type(h) is not WindowedDeltaHook]
        assert not bad, (cell["position"], bad)


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
                WindowedDeltaHook(
                    model, ly, d, a, decode_window=window, combine_prefill_ctx=combine
                )
                for ly, d, a in zip(layers, dirs, alphas, strict=True)
            ]
            steer = children[0] if k == 1 else MultiLayerDeltaHook(children)
        _assert_hook_types(cell, steer)
        recorder = EditPositionRecorder(model, layers[0])
        return RecordedHook(steer, recorder, position=pos)

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
    return {
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


def _upload_cell_json(path: Path) -> None:
    """Per-cell checkpoint upload (plan §9: raw completions upload per cell);
    the bulk ``upload_folder`` commit at phase end is the completeness
    backstop. Fail-loud after the transient-retry budget."""
    from explore_persona_space.orchestrate import hub

    dest = f"{_round_hf_prefix()}/raw_completions/steer/{path.name}"
    hub.retry_transient(
        # UPLOAD_LOOP_EXEMPT: plan §9 per-cell checkpoint upload — ~40 files
        # per shard over ~2h (not a tight loop); the phase-end bulk
        # upload_folder commit is the batched completeness backstop.
        lambda: hub._upload(
            path, HF_DATA_REPO, "dataset", dest, upload_as_file=True, raise_on_error=True
        ),
        what=f"per-cell checkpoint upload {path.name}",
    )


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
    checkpoints (cached-skip resume unless --force), round-robin
    ``--shard-id/--num-shards`` sharding, cap-hit > 2% => regen at 2x cap,
    per-cell + bulk HF raw-completion uploads BEFORE the shard sentinel."""
    i2254._require_cuda("steer (first-k)")
    out_root = i2254._out_root(args)
    rroot = round_root(out_root)
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

    model, tok = i2254._load_model_and_tokenizer()
    bank = DirectionBank()
    q_cache = {b: i2254._eval_questions(b)[: args.q_steer] for b in behaviors}
    for b, qs in q_cache.items():
        assert len(qs) == args.q_steer, (b, len(qs), args.q_steer)

    t0 = time.time()
    n_regen = 0
    for k, cell in enumerate(shard, 1):
        cid = _cell_id(cell)
        path = comp_root / f"{cid}.json"
        if path.exists() and not args.force:
            i2254._progress(SENTINEL_STEER, k, len(shard), f"{cid} (cached)", t0)
            continue
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
        rec["experiment"] = "issue2254_first_k_steering"
        rec["followup_label"] = FOLLOWUP_LABEL
        i2254._write_json_atomic(path, i2254._run_metadata(rec))
        _upload_cell_json(path)
        i2254._progress(SENTINEL_STEER, k, len(shard), cid, t0)

    i2254._upload_folder_to_hf(comp_root, f"{_round_hf_prefix()}/raw_completions/steer")
    tag = SENTINEL_STEER if args.num_shards == 1 else f"{SENTINEL_STEER}-shard{args.shard_id}"
    i2254._write_sentinel(out_root, tag, "done", {"cells": len(shard), "regen_cells": n_regen})
    i2254._breadcrumb(SENTINEL_STEER, status="done", regen_cells=n_regen)


PHASES = {
    "stage_inputs": phase_stage_inputs,
    "steer": phase_steer,
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
        help="comma-separated phases to run in order (stage_inputs,steer — plan §9 workload cmd)",
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
        inspect.signature(multi_layer_delta_hooks).bind(None, [14], [None], [0.0])
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
    else:  # unreachable behind the main() phase validation; keep fail-loud
        raise SystemExit(f"dry-run: no wiring branch for phase {phase!r}")
    print(f"[dry-run] {phase} wiring OK", flush=True)


def main() -> None:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if not args.phases:
        raise SystemExit(
            "--phases is required (comma-separated: stage_inputs,steer) or --import-check"
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
