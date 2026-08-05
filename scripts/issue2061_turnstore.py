"""Shared loader for #1336 turnstore shards (task #2061; code-review v1 C1).

Ground-truth producer: `scripts/issue1336_extract_turnstore.py::write_shards`
(~L436-480). Each turnstore directory holds `<stem>_shard{NNN:03d}.pt` files
whose payload is a dict with EXACTLY these keys (per-record LISTS, bf16):

    conv_ids   : list[str]
    slots      : list[Tensor (n_slots=2, n_layers, hidden) bf16]
                 # single-token SLOT states, ordered by token index:
                 # [prefix, a1] (spans_meta["slot_names"])
    profiles   : list[Tensor (n_turns=2, n_layers, hidden) bf16]
                 # span-MEAN over turn content tokens, ordered by span start:
                 # [u1, a1] (spans_meta["turn_names"])
    nll        : list[Tensor (n_turns,)]
    spans_meta : list[dict]  # slot_names / turn_names give the row order

Realized #1336 pooling convention (verified against
`scripts/issue1336_fit_cells.py::_cell_xy_1336` + `run_g0`, per plan §12
assumption 4's own verify instruction):

    prefix arm  -> slots[prefix]   (prefix-header slot, single-token state;
                                    slot_index 0 in `run_g0`'s parent-parity
                                    normalization / `_prefix_degeneracy`)
    context arm -> slots[a1]       (assistant-header slot = end of the
                                    context, single-token state; the
                                    `slot_index 1` of `_cell_xy_1336`)
    answer      -> profiles[a1]    (span-mean over answer content tokens;
                                    `target_turn_index 1`)

NOTE: plan §12 assumption 4 guessed "mean-pool over prefix tokens (prefix
arm) and mean-pool over prefix+query (context arm)"; the banked convention is
single-token SLOT states for both arms — no such mean-pools are banked
(profiles are per-TURN content span-means only). This module follows the
script, which the plan itself names as the grounding source; the deviation is
recorded in the Unit A implementation report.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

EXPECTED_SHARD_KEYS = frozenset({"conv_ids", "slots", "profiles", "nll", "spans_meta"})

# state -> (payload key, row name, spans_meta names field)
STATE_SPEC: dict[str, tuple[str, str, str]] = {
    "prefix": ("slots", "prefix", "slot_names"),
    "context": ("slots", "a1", "slot_names"),
    "answer": ("profiles", "a1", "turn_names"),
}

SHARD_NAME_RE = re.compile(r"_shard(\d+)\.pt$")


def assert_shard_schema(payload: object, src: str) -> None:
    """Fail loud (TypeError/KeyError) unless `payload` is a #1336 write_shards dict.

    Names the expected keys + the producer so a future schema drift crashes
    with a legible message instead of a bare KeyError deep in the consumer.
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"Turnstore shard {src}: expected a dict payload, got {type(payload).__name__}. "
            "Expected the #1336 write_shards schema "
            "(scripts/issue1336_extract_turnstore.py::write_shards)."
        )
    missing = EXPECTED_SHARD_KEYS - payload.keys()
    if missing:
        raise KeyError(
            f"Turnstore shard {src}: missing keys {sorted(missing)}; found "
            f"{sorted(payload.keys())}. Expected exactly the #1336 write_shards payload "
            f"keys {sorted(EXPECTED_SHARD_KEYS)} "
            "(scripts/issue1336_extract_turnstore.py::write_shards). Schema drift upstream?"
        )


def extract_state_rows(
    payload: dict, state: str, layer: int, src: str = "<shard>"
) -> tuple[torch.Tensor, list[str]]:
    """(n, hidden) float32 rows for one state at one layer, plus conv_ids.

    `state` selects the banked representation per the module docstring:
    "prefix"/"context" -> the named SLOT row; "answer" -> the a1 turn PROFILE
    (span-mean over answer content). Row indices are resolved from each
    record's spans_meta names (never hardcoded); asserts shard uniformity,
    3-D per-record tensors, and layer range — all fail-loud with the shard
    path + record id in the message.
    """
    assert_shard_schema(payload, src)
    if state not in STATE_SPEC:
        raise ValueError(f"state must be one of {sorted(STATE_SPEC)}, got {state!r}")
    key, row_name, names_field = STATE_SPEC[state]
    conv_ids = [str(c) for c in payload["conv_ids"]]
    tensors = payload[key]
    metas = payload["spans_meta"]
    if not conv_ids or not (len(tensors) == len(metas) == len(conv_ids)):
        raise ValueError(
            f"Turnstore shard {src}: per-record list lengths disagree or empty "
            f"(conv_ids={len(conv_ids)}, {key}={len(tensors)}, spans_meta={len(metas)})."
        )
    names0 = list(metas[0][names_field])
    if row_name not in names0:
        raise KeyError(
            f"Turnstore shard {src}: {names_field}={names0} has no '{row_name}' entry "
            f"(needed for state='{state}')."
        )
    idx = names0.index(row_name)
    rows: list[torch.Tensor] = []
    for i, (t, meta) in enumerate(zip(tensors, metas, strict=True)):
        names_i = list(meta[names_field])
        if names_i != names0:
            raise ValueError(
                f"Turnstore shard {src} record {i} ({conv_ids[i]}): {names_field}="
                f"{names_i} differs from record 0's {names0}; non-uniform shard."
            )
        if t.ndim != 3:
            raise ValueError(
                f"Turnstore shard {src} record {i} ({conv_ids[i]}): {key} tensor has "
                f"shape {tuple(t.shape)}; expected (n_rows, n_layers, hidden)."
            )
        if not 0 <= layer < t.shape[1]:
            raise IndexError(
                f"Turnstore shard {src} record {i} ({conv_ids[i]}): layer {layer} out "
                f"of range for {t.shape[1]}-layer {key} tensor."
            )
        rows.append(t[idx, layer])
    return torch.stack(rows).float().contiguous(), conv_ids


def enumerate_shards(turnstore_dir: Path | str) -> list[Path]:
    """ALL `*_shard{NNN}.pt` files under `turnstore_dir`, in shard-INDEX order.

    Numeric-index sort (never lexicographic — '_shard1000' < '_shard999'
    lexicographically). Fail-loud FileNotFoundError when the dir holds no
    shards (kills the `_shard000.pt`-only class: a caller can no longer
    silently consume one shard of many).
    """
    turnstore_dir = Path(turnstore_dir)
    hits: list[tuple[int, Path]] = []
    for p in turnstore_dir.glob("*_shard*.pt"):
        m = SHARD_NAME_RE.search(p.name)
        if m:
            hits.append((int(m.group(1)), p))
    if not hits:
        raise FileNotFoundError(
            f"No '*_shardNNN.pt' turnstore shards under {turnstore_dir} — expected the "
            "#1336 write_shards layout (scripts/issue1336_extract_turnstore.py)."
        )
    return [p for _, p in sorted(hits)]


def load_state_from_shards(
    shard_paths: Iterable[Path | str],
    state: str,
    layer: int,
    max_rows: int | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Concat one state's rows across shards (caller supplies shard-index order).

    Accepts any iterable — a lazily-downloading generator stops fetching the
    moment `max_rows` rows are accumulated. Loads one shard payload at a time
    (peak memory ~ one shard), extracting via `extract_state_rows` so every
    shard passes the fail-loud schema assert. Returns ((n, hidden) float32,
    conv_ids).
    """
    chunks: list[torch.Tensor] = []
    ids: list[str] = []
    n_acc = 0
    for p in shard_paths:
        payload = torch.load(p, map_location="cpu", weights_only=True)
        x, cids = extract_state_rows(payload, state=state, layer=layer, src=str(p))
        chunks.append(x)
        ids.extend(cids)
        n_acc += x.shape[0]
        if max_rows is not None and n_acc >= max_rows:
            break
    if not chunks:
        raise ValueError("load_state_from_shards: empty shard_paths iterable")
    x = torch.cat(chunks, dim=0)
    if max_rows is not None and x.shape[0] > max_rows:
        x = x[:max_rows]
        ids = ids[:max_rows]
    return x.contiguous(), ids


def group_fold_ids(conv_ids: Sequence[str], n_folds: int, seed: int) -> np.ndarray:
    """GROUP-level fold ids per row — #1336's exact fold convention (plan §10, M5).

    Mirrors `scripts/issue825_fit_cells.py::_cv_folds` (the fold constructor
    #1336's `issue1336_fit_cells.py` drives via the #825 cores with
    `cm.N_FOLDS=5` / `cm.FIT_SEED=0`): a seeded permutation of the UNIQUE
    conversation ids, `perm[i] % n_folds` per unique id, so every row sharing
    a conversation id lands in the same fold (`.claude/rules/
    ood-generalization-folds.md`). Equality with `_cv_folds` is pinned by
    `tests/test_issue2061_stats.py`. Fail-loud when any fold is empty.
    """
    conv_arr = np.asarray(list(conv_ids))
    uniq = np.unique(conv_arr)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    conv_fold = {cid: int(perm[i] % n_folds) for i, cid in enumerate(uniq)}
    folds = np.array([conv_fold[c] for c in conv_arr], dtype=np.int64)
    for cid in uniq:
        f = folds[conv_arr == cid]
        assert (f == f[0]).all(), f"fold id varies within conversation {cid!r}"
    counts = np.bincount(folds, minlength=n_folds)
    if (counts == 0).any():
        raise ValueError(
            f"group_fold_ids: empty fold(s) {np.where(counts == 0)[0].tolist()} — "
            f"only {len(uniq)} unique conversation ids for n_folds={n_folds}."
        )
    return folds
