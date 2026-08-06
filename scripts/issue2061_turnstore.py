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

import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

EXPECTED_SHARD_KEYS = frozenset({"conv_ids", "slots", "profiles", "nll", "spans_meta"})

ARMS = ("prefix", "context")

# ---------------------------------------------------------------------------
# Extended-corpus concat registry (plan v11 delta (a1); registered consumption
# grain, plan §Design "Registered consumption grain").
#
# PORTED VERBATIM from the unmerged parent branch `issue-1336-fullcorpora`
# (`src/explore_persona_space/experiments/issue_1336/common.py:428-429`) per
# `.claude/rules/artifact-reuse.md` § "Porting from an unmerged sibling
# branch" — never a runtime import from the unmerged branch. Wave-1 covered
# prompt_idx 0..boundary-1 of each extended corpus; the v2 stores hold ONLY
# the extension rows (prompt_idx >= boundary), and #1336's own production fit
# consumes the CONCATENATION (`load_bundle_concat`, fullcorpora
# `fit_cells.py:766-781`). V2_CONCAT_BOUNDARY is a PER-CORPUS DICT (both
# entries 5000 today) — never a scalar (plan v12 amendment item 4).
# ---------------------------------------------------------------------------
V2_CONCAT_SOURCES = {"lmsys23k": "lmsys5k", "gsm8k_train_full": "gsm8k_train5k"}
V2_CONCAT_BOUNDARY = {"lmsys23k": 5000, "gsm8k_train_full": 5000}

# #1336 conv_id convention: "s{prompt_idx}" (issue1336_extract_turnstore.py's
# `sha_by_conv = {f"s{r['prompt_idx']}": ...}`); the concat boundary asserts
# parse the index out of it.
CONV_ID_RE = re.compile(r"^s(\d+)$")

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


def conv_index(conv_id: str) -> int:
    """Parse the #1336 conv_id convention `s{prompt_idx}` -> int prompt_idx.

    Fail-loud ValueError on any non-conforming id (the concat boundary asserts
    must never silently mis-bucket a row — plan §Design boundary asserts).
    """
    m = CONV_ID_RE.match(str(conv_id))
    if m is None:
        raise ValueError(
            f"conv_id {conv_id!r} does not match the #1336 's<prompt_idx>' convention — "
            "cannot apply the V2_CONCAT_BOUNDARY assert to it."
        )
    return int(m.group(1))


def sidecar_path_for(shard_pt: Path | str) -> Path:
    """The producer's JSON sidecar path beside one `*_shardNNN.pt` file."""
    return Path(shard_pt).with_suffix(".json")


def read_shard_sidecar(shard_pt: Path | str) -> dict:
    """Load the JSON sidecar beside one shard; fail loud when it is missing.

    The #1336 producer (`issue1336_extract_turnstore.py::write_shards`) writes
    a `<stem>_shardNNN.json` sidecar next to every `.pt` shard; the loader's
    schema/convention asserts (plan v13 delta (a1-bis)) read it, so staging
    must include sidecars (issue2061_hub_io.stage_turnstore does since the
    v13 round — re-stage any tree staged by older code).
    """
    sp = sidecar_path_for(shard_pt)
    if not sp.exists():
        raise FileNotFoundError(
            f"Missing turnstore sidecar {sp} beside {shard_pt} — the loader's per-shard "
            "schema/convention asserts (plan v13 delta a1-bis) require it. Re-stage the "
            "turnstore (staging includes sidecars as of the v13 concat round)."
        )
    return json.loads(sp.read_text())


def assert_shard_sidecar(payload: dict, sidecar: dict, src: str) -> dict:
    """Fail-loud payload<->sidecar consistency + capture-convention asserts.

    Plan §Design "Cross-wave schema parity" + v13 delta (a1-bis), per shard:

    - sidecar `n_conversations` and `conv_ids` must equal the payload's
      conv_ids EXACTLY (the ±1 row-count bookkeeping class — a divergence is
      corruption / mixed write generations, never tolerated);
    - per-row prompt-sha join: when BOTH the payload and the sidecar carry
      `prompt_shas` (the v2-era stores), they must agree per row with ZERO
      mismatches tolerated (the parent `load_bundle_concat` text-drift
      contract); one-sided presence is corruption;
    - capture-convention keys are PRESENCE-CONDITIONAL: present => must be
      `"committed"` with `offset_override` null; ABSENT => accepted as a
      pre-D2-era store (the 2 oldest wave-1 lmsys5k rlvr stores, extractor
      commit 5653c791ac) and reported as `"pre-D2-absent"` so the P0 grain
      manifest can log the store name. A hard key-presence assert would fail
      loud on those two stores at first dispatch (plan v13 item 1).

    Returns a summary dict: {"convention", "model_id", "n_rows"}.
    """
    pc = [str(c) for c in payload["conv_ids"]]
    sc = [str(c) for c in sidecar.get("conv_ids", [])]
    n_sc = sidecar.get("n_conversations")
    if n_sc != len(pc) or sc != pc:
        raise ValueError(
            f"Turnstore shard {src}: sidecar/payload row mismatch — payload has "
            f"{len(pc)} conv_ids, sidecar declares n_conversations={n_sc} with "
            f"{len(sc)} conv_ids (first divergence: "
            f"{next(((a, b) for a, b in zip(pc, sc) if a != b), '(length-only)')}). "
            "Mixed write generations / corruption — never consume."
        )
    p_shas = payload.get("prompt_shas")
    s_shas = sidecar.get("prompt_shas")
    if (p_shas is None) != (s_shas is None):
        side = "payload" if p_shas is not None else "sidecar"
        raise ValueError(
            f"Turnstore shard {src}: prompt_shas present on the {side} side ONLY — "
            "the v2 producer writes them to BOTH (write_shards); mixed generations."
        )
    if p_shas is not None and s_shas is not None:
        if len(p_shas) != len(pc) or len(s_shas) != len(pc):
            raise ValueError(
                f"Turnstore shard {src}: prompt_shas length mismatch "
                f"(payload {len(p_shas)}, sidecar {len(s_shas)}, rows {len(pc)})."
            )
        mismatches = [i for i, (a, b) in enumerate(zip(p_shas, s_shas)) if str(a) != str(b)]
        if mismatches:
            raise ValueError(
                f"Turnstore shard {src}: {len(mismatches)} per-row prompt-sha MISMATCH(es) "
                f"payload-vs-sidecar (e.g. rows {mismatches[:5]}) — text drift; ZERO "
                "mismatches tolerated (the parent load_bundle_concat contract)."
            )
    convention = sidecar_convention_state(sidecar, src)
    return {"convention": convention, "model_id": sidecar.get("model_id"), "n_rows": len(pc)}


def sidecar_convention_state(sidecar: dict, src: str) -> str:
    """PRESENCE-CONDITIONAL capture-convention assert (plan v13 delta a1-bis).

    Present => must be `"committed"` with `offset_override` null (fail-loud
    ValueError otherwise); ABSENT => accepted as a pre-D2-era store and
    reported as `"pre-D2-absent"` (the caller logs the store name — e.g. into
    the P0 grain manifest). Shared by the shard loader
    (`assert_shard_sidecar`) and the P0 grain gate.
    """
    if "convention" in sidecar:
        if sidecar["convention"] != "committed" or sidecar.get("offset_override") is not None:
            raise ValueError(
                f"Turnstore shard {src}: capture convention "
                f"{sidecar['convention']!r} / offset_override="
                f"{sidecar.get('offset_override')!r} != the registered committed/null "
                "(plan §Design cross-wave schema parity) — never consume a "
                "corrected-convention store into the registered grain."
            )
        return "committed"
    if sidecar.get("offset_override") is not None:
        raise ValueError(
            f"Turnstore shard {src}: offset_override present without a convention "
            "key — not a known producer shape."
        )
    # Pre-D2-era store (keys introduced by the D2 capture-convention flags;
    # committed-convention by construction). ACCEPTED + reported.
    return "pre-D2-absent"


def assert_concat_boundary(conv_wave1: Sequence[str], conv_ext: Sequence[str], corpus: str) -> int:
    """The parent `load_bundle_concat` boundary + disjointness asserts, ported.

    Every wave-1 row's prompt_idx < V2_CONCAT_BOUNDARY[corpus] <= every
    extension row's; no conv_id overlap between the two parts. Returns the
    boundary. Fail-loud ValueError naming the first offending ids.
    """
    if corpus not in V2_CONCAT_BOUNDARY:
        raise ValueError(f"{corpus!r} is not an extended corpus ({sorted(V2_CONCAT_BOUNDARY)})")
    boundary = V2_CONCAT_BOUNDARY[corpus]
    if not conv_wave1 or not conv_ext:
        raise ValueError(
            f"concat {corpus}: empty part (wave1={len(conv_wave1)}, ext={len(conv_ext)})"
        )
    bad1 = [c for c in conv_wave1 if conv_index(c) >= boundary]
    if bad1:
        raise ValueError(
            f"concat {corpus}: wave-1 store has {len(bad1)} row(s) with prompt_idx >= "
            f"{boundary} (e.g. {bad1[:5]}) — not a wave-1 slice."
        )
    bad2 = [c for c in conv_ext if conv_index(c) < boundary]
    if bad2:
        raise ValueError(
            f"concat {corpus}: extension store has {len(bad2)} row(s) with prompt_idx < "
            f"{boundary} (e.g. {bad2[:5]}) — not an extension slice."
        )
    overlap = set(map(str, conv_wave1)) & set(map(str, conv_ext))
    if overlap:
        raise ValueError(
            f"concat {corpus}: parts overlap on {len(overlap)} conv_id(s) "
            f"(e.g. {sorted(overlap)[:5]})."
        )
    return boundary


def load_state_from_shards(
    shard_paths: Iterable[Path | str],
    state: str,
    layer: int,
    max_rows: int | None = None,
    *,
    check_sidecars: bool = False,
    sidecar_records: list[dict] | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Concat one state's rows across shards (caller supplies shard-index order).

    Accepts any iterable — a lazily-downloading generator stops fetching the
    moment `max_rows` rows are accumulated. Loads one shard payload at a time
    (peak memory ~ one shard), extracting via `extract_state_rows` so every
    shard passes the fail-loud schema assert. With `check_sidecars=True`
    (production consumption paths) each shard's JSON sidecar is additionally
    read + asserted via `assert_shard_sidecar` (plan v13 delta a1-bis);
    per-shard summaries append to `sidecar_records` when provided. Returns
    ((n, hidden) float32, conv_ids).
    """
    chunks: list[torch.Tensor] = []
    ids: list[str] = []
    n_acc = 0
    for p in shard_paths:
        payload = torch.load(p, map_location="cpu", weights_only=True)
        if check_sidecars:
            assert_shard_schema(payload, str(p))
            rec = assert_shard_sidecar(payload, read_shard_sidecar(p), src=str(p))
            if sidecar_records is not None:
                sidecar_records.append({**rec, "shard": Path(p).name})
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


def turnstore_dir_name(stage: str, render: str, corpus: str) -> str:
    """Canonical LOCAL turnstore dir name (consumer layout; canonical stage tokens)."""
    return f"turnstore_{stage}_{render}_{corpus}"


def cell_store_dirs(root: Path | str, stage: str, render: str, corpus: str) -> list[Path]:
    """ORDERED local store dirs one (stage, render, corpus) cell consumes.

    Registered consumption grain (plan v11): the extended corpora consume TWO
    stores — the wave-1 concat source FIRST, then the v2 extension — in the
    canonical concat row order; standalone corpora consume their single store.
    """
    root = Path(root)
    dirs = []
    if corpus in V2_CONCAT_SOURCES:
        dirs.append(root / turnstore_dir_name(stage, render, V2_CONCAT_SOURCES[corpus]))
    dirs.append(root / turnstore_dir_name(stage, render, corpus))
    return dirs


def load_state_cell(
    root: Path | str,
    stage: str,
    render: str,
    corpus: str,
    state: str,
    layer: int,
    *,
    check_sidecars: bool = True,
) -> tuple[torch.Tensor, list[str], dict]:
    """One cell's state rows at the REGISTERED consumption grain (local dirs).

    For the extended corpora (`V2_CONCAT_SOURCES`) loads wave-1 source shards
    FIRST then v2-extension shards (both in shard-index order — the canonical
    row order P1/P2/P3 share, plan §Design "Registered consumption grain"),
    applying the ported `load_bundle_concat` asserts: per-shard sidecar
    schema/convention (a1-bis), boundary (every wave-1 idx < boundary <= every
    extension idx), conv-id disjointness, cross-tree model_id equality, and a
    duplicate-conv-id check over the union. Standalone corpora load their
    single store through the same sidecar asserts.

    Returns (x (n, hidden) float32, conv_ids, info) where info carries the
    per-part composition (dir, n_rows, conventions, model_ids) + concat flag.
    """
    dirs = cell_store_dirs(root, stage, render, corpus)
    xs: list[torch.Tensor] = []
    ids_parts: list[list[str]] = []
    parts: list[dict] = []
    for d in dirs:
        if not d.is_dir():
            extra = (
                " (a CONCAT cell consumes the wave-1 source AND the v2 extension — "
                "stage both; issue2061_hub_io.stage_turnstore does)"
                if len(dirs) == 2
                else ""
            )
            raise FileNotFoundError(f"missing turnstore dir {d}{extra}")
        recs: list[dict] = []
        x, cids = load_state_from_shards(
            enumerate_shards(d),
            state=state,
            layer=layer,
            check_sidecars=check_sidecars,
            sidecar_records=recs,
        )
        model_ids = sorted({str(r["model_id"]) for r in recs})
        if check_sidecars and len(model_ids) > 1:
            raise ValueError(f"turnstore {d}: mixed model_id across shards: {model_ids}")
        xs.append(x)
        ids_parts.append(cids)
        parts.append(
            {
                "dir": d.name,
                "n_rows": len(cids),
                "conventions": sorted({r["convention"] for r in recs}),
                "model_ids": model_ids,
            }
        )
    concat = len(dirs) == 2
    if concat:
        assert_concat_boundary(ids_parts[0], ids_parts[1], corpus)
        if check_sidecars and parts[0]["model_ids"] != parts[1]["model_ids"]:
            raise ValueError(
                f"concat {stage}/{render}/{corpus}: cross-wave model_id mismatch "
                f"{parts[0]['model_ids']} vs {parts[1]['model_ids']} (plan §Design "
                "cross-wave schema parity hard assert)."
            )
    conv_ids = [c for part in ids_parts for c in part]
    if len(set(conv_ids)) != len(conv_ids):
        raise ValueError(
            f"cell {stage}/{render}/{corpus}: duplicate conv_ids in the consumed row set "
            f"(n={len(conv_ids)}, unique={len(set(conv_ids))})."
        )
    x = torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]
    info = {"concat": concat, "parts": parts, "n_rows": len(conv_ids)}
    composition = " + ".join(f"{p['dir']}:{p['n_rows']}" for p in parts)
    print(
        f"[cell-load] {stage}/{render}/{corpus} state={state}: {composition} = "
        f"{len(conv_ids)} rows" + (" (concat; boundary+disjointness PASS)" if concat else ""),
        flush=True,
    )
    return x.contiguous(), conv_ids, info


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


# ---------------------------------------------------------------------------
# Cell-filename schema (ONE source of truth — review round-2 M2/M3).
#
# The #1336 turnstore naming convention (`turnstore_<stage>_<render>_<corpus>`,
# parsed with split("_", 2) everywhere) guarantees stage and render are
# underscore-FREE while corpus may carry underscores (gsm8k_train_full,
# gsm8k_test1319). Every derived filename in this issue therefore parses from
# the LEFT (stage, render) + a closed vocabulary from the RIGHT (state/arm,
# layer), keeping underscore corpora intact. A RIGHT-anchored `parts[-3]`
# parse silently mis-reads those corpora ("full", "test1319") and VANISHES
# their cells from the GLOBAL null — worse than a crash (review M2).
# ---------------------------------------------------------------------------


def encoded_target_name(
    stage: str,
    render: str,
    corpus: str,
    state: str,
    layer: int,
    *,
    max_rows: int | None = None,
) -> str:
    """Canonical P1 encoded-target filename.

    A `--max-rows`-capped (debug) encode gets a `_rows{N}` suffix so it can
    NEVER be skip-reused by a production run at the canonical path, and the
    consumers' `*_{state}_L{layer}.pt` globs never match it (review M3).
    """
    base = f"{stage}_{render}_{corpus}_{state}_L{layer}"
    if max_rows is not None:
        base += f"_rows{max_rows}"
    return base + ".pt"


def parse_encoded_stem(stem: str, state: str, layer: int) -> tuple[str, str, str]:
    """Parse `<stage>_<render>_<corpus>_<state>_L<layer>` -> (stage, render, corpus).

    LEFT split (stage/render underscore-free per the module convention above);
    fail-loud ValueError on any non-conforming stem.
    """
    suffix = f"_{state}_L{layer}"
    if not stem.endswith(suffix):
        raise ValueError(
            f"Unrecognized encoded-target stem {stem!r}; expected "
            f"<stage>_<render>_<corpus>{suffix}.pt "
            "(scripts/issue2061_sae_encode.py::encode_turnstore)."
        )
    core = stem.removesuffix(suffix)
    parts = core.split("_", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Unrecognized encoded-target stem {stem!r}: cell part {core!r} does not "
            "split into <stage>_<render>_<corpus>."
        )
    stage, render, corpus = parts
    return stage, render, corpus


def parse_r2_stem(stem: str, layer: int) -> tuple[str, str, str, str]:
    """Parse `<stage>_<render>_<corpus>_<arm>_L<layer>` -> (stage, render, corpus, arm).

    LEFT split for (stage, render); arm is the closed vocabulary `ARMS` taken
    from the RIGHT — so underscore corpora (gsm8k_train_full) parse intact.
    Fail-loud ValueError on anything else: a silently vanishing cell is worse
    than a crash here (the GLOBAL null's cell count is load-bearing, review M2).
    """
    suffix = f"_L{layer}"
    if not stem.endswith(suffix):
        raise ValueError(f"R² stem {stem!r} does not end with {suffix!r}.")
    parts = stem.removesuffix(suffix).split("_")
    if len(parts) < 4 or not all(parts):
        raise ValueError(
            f"R² stem {stem!r} does not parse as <stage>_<render>_<corpus>_<arm>{suffix} "
            "(scripts/issue2061_fit_per_feature.py output convention)."
        )
    stage, render, arm = parts[0], parts[1], parts[-1]
    corpus = "_".join(parts[2:-1])
    if arm not in ARMS:
        raise ValueError(
            f"R² stem {stem!r}: arm token {arm!r} not in {ARMS} — refusing to guess "
            "the corpus boundary (review M2: never silently drop a cell)."
        )
    return stage, render, corpus, arm


# ---------------------------------------------------------------------------
# P1 encoded-target payload: fixed-width TopK sparse (review round-2 M1).
#
# The SAE target is exact-TopK (k=32 of d_sae=262,144), so (idx, val) at fixed
# width k is a LOSSLESS factor-~4096 compression of the dense float32 matrix
# (~24 GB -> ~6 MB for the largest lmsys23k turnstore; the dense store would
# EDQUOT the ~130 GB RunPod /workspace quota at ~527 GB total). The payload
# ALSO carries conv_ids so X/Y row alignment is KEYED, not order-faith
# (consumers fail loud on mismatch instead of silently mispairing rows).
# ---------------------------------------------------------------------------

ENCODED_TARGET_FORMAT = "issue2061-topk-sparse-v1"


def _git_commit_sha() -> str:
    """Best-effort commit sha for reproducibility metadata (never fail-loud:
    git-less scratch trees — fellows/SLURM rsync lanes — have no checkout, #1902)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parent,  # caller cwd may be off-repo
        )
        return out.stdout.strip() or "unavailable-no-git-checkout"
    except (OSError, subprocess.SubprocessError):
        return "unavailable-no-git-checkout"


def save_encoded_target(
    path: Path | str,
    *,
    idx: torch.Tensor,
    val: torch.Tensor,
    d_sae: int,
    k: int,
    conv_ids: Sequence[str],
    cell: dict[str, object],
    extra_meta: dict[str, object] | None = None,
) -> Path:
    """Atomically write one encoded-target payload (`ENCODED_TARGET_FORMAT`).

    `idx` (n, k) feature ids (stored int32 — d_sae < 2^31), `val` (n, k)
    float32 TopK values, `conv_ids` the row-aligned #1336 conversation ids,
    `cell` the {stage, render, corpus, state, layer} identity. Reproducibility
    metadata (commit sha, torch version, timestamp) rides `meta` per the
    CLAUDE.md reproducibility rule. Same-dir tmp + os.replace (atomic; the
    #1335 EXDEV gotcha).
    """
    path = Path(path)
    idx = torch.as_tensor(idx)
    val = torch.as_tensor(val)
    n = int(idx.shape[0])
    assert idx.shape == val.shape == (n, k), (idx.shape, val.shape, n, k)
    if n != len(conv_ids):
        raise ValueError(f"conv_ids length {len(conv_ids)} != n_rows {n}")
    if n and int(idx.max()) >= d_sae:
        raise ValueError(f"feature id {int(idx.max())} >= d_sae={d_sae}")
    payload = {
        "format": ENCODED_TARGET_FORMAT,
        "d_sae": int(d_sae),
        "k": int(k),
        "n_rows": n,
        "idx": idx.to(torch.int32).contiguous(),
        "val": val.to(torch.float32).contiguous(),
        "conv_ids": [str(c) for c in conv_ids],
        "cell": dict(cell),
        "meta": {
            "git_commit": _git_commit_sha(),
            # str(): TorchVersion objects poison weights_only=True round-trips
            "torch_version": str(torch.__version__),
            "created_unix": time.time(),
            **(extra_meta or {}),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def load_encoded_target(path: Path | str) -> dict:
    """Load + fail-loud-validate an encoded-target payload.

    Raises TypeError/KeyError/ValueError with a migration-naming message on
    anything that is not an `ENCODED_TARGET_FORMAT` payload (e.g. a stale
    pre-round-2 dense tensor at the same path — re-encode it, never consume).
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Encoded target {path}: expected an {ENCODED_TARGET_FORMAT} dict payload, got "
            f"{type(payload).__name__} — a pre-round-2 DENSE store? Re-encode via "
            "scripts/issue2061_sae_encode.py (review M1: dense stores are retired)."
        )
    if payload.get("format") != ENCODED_TARGET_FORMAT:
        raise ValueError(
            f"Encoded target {path}: format {payload.get('format')!r} != "
            f"{ENCODED_TARGET_FORMAT!r} — schema drift; re-encode."
        )
    missing = {"d_sae", "k", "n_rows", "idx", "val", "conv_ids", "cell"} - payload.keys()
    if missing:
        raise KeyError(f"Encoded target {path}: missing keys {sorted(missing)}.")
    n, k = int(payload["n_rows"]), int(payload["k"])
    if tuple(payload["idx"].shape) != (n, k) or tuple(payload["val"].shape) != (n, k):
        raise ValueError(
            f"Encoded target {path}: idx/val shapes {tuple(payload['idx'].shape)}/"
            f"{tuple(payload['val'].shape)} != (n_rows={n}, k={k})."
        )
    if len(payload["conv_ids"]) != n:
        raise ValueError(
            f"Encoded target {path}: {len(payload['conv_ids'])} conv_ids != n_rows={n}."
        )
    return payload


def encoded_to_dense(payload: dict) -> torch.Tensor:
    """(n, d_sae) float32 dense reconstruction of a sparse payload.

    Test/smoke-scale use only — the whole POINT of the sparse layout is that
    production consumers never materialize this at d_sae=262,144 (review M1).
    """
    n, d_sae = int(payload["n_rows"]), int(payload["d_sae"])
    dense = torch.zeros((n, d_sae), dtype=torch.float32)
    dense.scatter_(1, payload["idx"].to(torch.int64), payload["val"].to(torch.float32))
    return dense


def to_fixed_width_sparse(y: torch.Tensor, row_chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Dense (n, d_sae) -> fixed-width sparse ((n, kmax) idx int64, (n, kmax) val f32).

    The SAE target is TopK (k=32) so kmax is small; padding is (idx=0,
    val=0.0), which is accumulation-safe everywhere the consumers add `val`
    (never overwrite). Row-chunked so an mmap'd dense store never
    materializes whole. (Moved here from scripts/issue2061_null.py in review
    round 2 — the fit parity gate + tests share it now; production paths
    consume the sparse payload directly and never convert from dense.)
    """
    n = int(y.shape[0])
    counts = np.empty(n, dtype=np.int64)
    for r0 in range(0, n, row_chunk):
        yc = y[r0 : r0 + row_chunk].to(torch.float32)
        counts[r0 : r0 + yc.shape[0]] = (yc != 0).sum(dim=1).numpy()
    kmax = max(1, int(counts.max()) if n else 1)
    idx = np.zeros((n, kmax), dtype=np.int64)
    val = np.zeros((n, kmax), dtype=np.float32)
    for r0 in range(0, n, row_chunk):
        yc = y[r0 : r0 + row_chunk].to(torch.float32).numpy()
        for i in range(yc.shape[0]):
            nz = np.nonzero(yc[i])[0]
            idx[r0 + i, : len(nz)] = nz
            val[r0 + i, : len(nz)] = yc[i, nz]
    return idx, val


def sparse_column_means(
    y_idx: np.ndarray, y_val: np.ndarray, rows: np.ndarray, d_sae: int
) -> np.ndarray:
    """(d_sae,) float64 per-feature mean over `rows` of a fixed-width sparse target.

    Padding entries (idx=0, val=0.0) add zero — accumulation-safe by the
    payload convention.
    """
    out = np.zeros(d_sae, dtype=np.float64)
    np.add.at(out, y_idx[rows].ravel(), y_val[rows].astype(np.float64).ravel())
    return out / max(1, len(rows))


def sparse_dense_chunk(
    y_idx: np.ndarray,
    y_val: np.ndarray,
    rows: np.ndarray,
    c0: int,
    c1: int,
) -> np.ndarray:
    """Dense float64 (len(rows), c1-c0) slice of a fixed-width sparse target.

    Feature-chunked so production consumers never materialize the full
    (n, d_sae) dense matrix (review M1). np.add.at keeps duplicate/pad
    entries accumulation-safe.
    """
    idx_s = y_idx[rows]
    val_s = y_val[rows]
    dense = np.zeros((len(rows), c1 - c0), dtype=np.float64)
    m = (idx_s >= c0) & (idx_s < c1)
    r, kk = np.nonzero(m)
    np.add.at(dense, (r, idx_s[r, kk] - c0), val_s[r, kk].astype(np.float64))
    return dense
