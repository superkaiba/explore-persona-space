"""P1 SAE encode phase for task #2061.

Reads #1336's banked layer-29 activations from the HF data repo
(`superkaiba1/explore-persona-space-data/issue1336_rlvr_ladder/analysis_tensors/turnstore_<stage>_<render>_<corpus>/`),
runs the fixed EleutherAI/sae-llama-3.1-8b-64x TopK encoder (via the
ported `topk_encode_sparse` in analysis/sparsify_topk_sae), and emits
per-cell SAE-feature-vector targets under `data/issue_2061/sae_encoded/`
as `<stage>_<render>_<corpus>_<state>_L<layer>.pt` (default state:
`answer` — the plan §Design target Y = SAE encode of the ANSWER state,
i.e. the a1 turn profile of the #1336 turnstore; see
`scripts/issue2061_turnstore.py` for the realized payload schema +
pooling convention).

Output format (review round-2 M1): the fixed-width TopK SPARSE payload
`issue2061_turnstore.ENCODED_TARGET_FORMAT` — (n, k=32) feature ids +
values plus row-aligned conv_ids — NOT a dense (n, d_sae) tensor. The
dense store measured ~24 GB for lmsys23k (n~23k x 262,144 float32),
~527 GB across the ~50 turnstores vs the ~130 GB RunPod /workspace
quota; the sparse payload is ~6 MB for the same cell (factor ~4096) and
is the layout the P2 fit + P3 null consumers open directly. conv_ids
make X/Y row alignment KEYED (consumers assert them) instead of
resting on shard-enumeration order alone.

Plan §Design + §9 P1 (v7 registered grid): 5 stages × 7 (render, corpus)
combos of the REGISTERED v2 capture generation = 35 turnstores (the v1
generation stays banked as a parked lower-n robustness arm behind
`--generation`); each turnstore holds MANY `*_shardNNN.pt` files of ≤500
records each (`SHARD_SIZE = 500`, issue1336_extract_turnstore.py), each
carrying all 32 layers (~525 MB/shard). ALL shards of a turnstore are
enumerated + concatenated in shard-index order. Batched encode at ~256
rows/batch; the per-batch dense pre-activation buffer (~268 MB at
batch 256) is the peak — no terminal dense concat.

Loader-parity FVE smoke gate (plan §Design 'Loader adapter'):
    |FVE_ported - FVE_reference| < 0.05
runs as the FIRST action when --smoke-only or --smoke-then-encode is set.
On failure, raises RuntimeError (fail-loud, non-zero exit); do NOT
proceed to production encode with an unverified loader.

Usage:
    uv run python scripts/issue2061_sae_encode.py --smoke-only            # loader-parity FVE gate
    uv run python scripts/issue2061_sae_encode.py --stage base --corpus lmsys23k --render chat
    uv run python scripts/issue2061_sae_encode.py --all-cells             # production sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from explore_persona_space.analysis.sparsify_topk_sae import (  # noqa: E402
    load_sae_weights,
    topk_encode,
    topk_encode_sparse,
    topk_reconstruct,
)
from explore_persona_space.orchestrate.hub import retry_transient  # noqa: E402

# Sibling-script import (bare module name via the script-dir sys.path insert —
# the issue1336_extract_turnstore.py pattern; works in script mode AND under
# the tests' `sys.path.insert(scripts)` import).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import issue2061_turnstore as ts  # noqa: E402


# Layer 29 is the nearest banked+SAE-available layer (L30 is #1336's headline
# but not SAE-available for this dictionary). See plan §11 Layer row.
LAYER = 29
BATCH_SIZE = 256  # plan §11 SAE encode batch — H100 24GB HBM fits easily

# #1336's banked activation store prefix. Bind exact revision at Phase 1.5
# fact-check (plan §10 pins).
DATA_REPO = "superkaiba1/explore-persona-space-data"
BANKED_PREFIX = "issue1336_rlvr_ladder/analysis_tensors"

# SAE repo — fixed target dictionary across the ladder (plan §6 Construct).
SAE_REPO = "EleutherAI/sae-llama-3.1-8b-64x"

# Output prefix on the HF data repo (analysis tensors — a downstream
# input P2/P3 consume; must persist across pod terminations, per
# CLAUDE.md § Upload Policy 'Intermediate analysis tensors'). See #521.
OUTPUT_PREFIX = "issue2061"


# Realized #1336 store naming (LIVE-enumerated 2026-08-05; 55 turnstores):
#     turnstore_[v2_]<stage>_<render>_<corpus>
# Stage tokens as REALIZED in the store: base|sft|dpo|rlvr|rlvr_long —
# `rlvr_long` IS the plan's "longer-rlvr" 5th ladder stage, normalized to the
# canonical hyphenated token because every downstream stem LEFT-parses the
# stage as underscore-free (issue2061_turnstore.parse_encoded_stem /
# parse_r2_stem; null.py STAGE_PAIRS / fitness.py STAGES already use
# "longer-rlvr"). The optional `v2_` prefix marks #1336's second capture
# generation; the two generations' corpus sets are DISJOINT today (v1:
# gsm8k_test1319/gsm8k_train5k/lmsys5k; v2: gsm8k_train_full/if11k/lmsys23k/
# math7500/sft11k/uf11k), asserted collision-free at enumeration. A naive
# `split("_", 2)` mis-buckets 35/55 realized names (the round-2 unit-E live
# probe finding — the C1 fabricated-schema class at the NAMING level).
STORE_STAGE_TOKENS: dict[str, str] = {
    "base": "base",
    "sft": "sft",
    "dpo": "dpo",
    "rlvr": "rlvr",
    "rlvr_long": "longer-rlvr",
}
CANONICAL_TO_STORE_STAGE = {v: k for k, v in STORE_STAGE_TOKENS.items()}
RENDER_TOKENS = ("chat", "naturalistic")

# REGISTERED capture generation (plan v7 amendment record — the conditions
# grid is the v2 generation ONLY: 7 (render, corpus) combos / 35 stores /
# 56 delta cells). Without this pin the enumeration consumes the 11-combo
# v1+v2 UNION: `parse_turnstore_name` strips the `v2_` prefix and the
# same-cell collision fail-loud below keys on canonical (stage, render,
# corpus), which never collides across generations because their corpus
# stems are DISJOINT today. The v1 generation (4 combos over
# {gsm8k_test1319, gsm8k_train5k, lmsys5k}) stays banked as a parked
# lower-n robustness arm reachable via the `--generation` override (its
# own plan approval — plan § Follow-ups).
GENERATIONS = ("v1", "v2")
REGISTERED_GENERATION = "v2"


def turnstore_generation(name: str) -> str | None:
    """Capture generation ('v1' | 'v2') of a REALIZED turnstore dir name.

    The store marks the second capture generation with a `v2_` prefix
    (`turnstore_v2_<stage>_...`); prefix-less turnstore names are the v1
    generation. Returns None for a non-turnstore name.
    """
    if not name.startswith("turnstore_"):
        return None
    return "v2" if name.removeprefix("turnstore_").startswith("v2_") else "v1"


def parse_turnstore_name(name: str) -> tuple[str, str, str] | None:
    """(canonical stage, render, corpus) from a REALIZED turnstore dir name.

    Vocabulary-based LEFT parse (longest stage token first, so `rlvr_long`
    wins over `rlvr`); returns None for a non-turnstore / unrecognized name
    (the caller WARNs — a silent drop is the M2 vanishing-cell class).
    """
    if not name.startswith("turnstore_"):
        return None
    core = name.removeprefix("turnstore_").removeprefix("v2_")
    for store_tok in sorted(STORE_STAGE_TOKENS, key=len, reverse=True):
        if not core.startswith(store_tok + "_"):
            continue
        rest = core[len(store_tok) + 1 :]
        for render in RENDER_TOKENS:
            if rest.startswith(render + "_") and len(rest) > len(render) + 1:
                return STORE_STAGE_TOKENS[store_tok], render, rest[len(render) + 1 :]
        return None
    return None


def _stage_render_corpus_turnstores(
    revision: str | None = None, generation: str = REGISTERED_GENERATION
) -> list[dict[str, str]]:
    """Enumerate #1336's banked `generation` turnstores as CANONICAL cells.

    Reads `list_repo_tree(path_in_repo="issue1336_rlvr_ladder/analysis_tensors")`
    and parses the realized `turnstore_[v2_]<stage>_<render>_<corpus>` names
    (see STORE_STAGE_TOKENS above), keeping ONLY the requested capture
    generation (default: the REGISTERED v2 grid — plan v7 amendment; the
    skipped other-generation count is printed, never a silent drop). Returns
    dicts with keys: stage, render, corpus, tree_path — `tree_path` keeps
    the REALIZED repo path (what gets fetched); the identity keys are
    canonical. Unparseable in-generation turnstore names are WARNed (never
    silently dropped); two realized trees mapping to one canonical cell
    within the generation fail loud (defensive — canonical identity would
    be ambiguous).
    """
    if generation not in GENERATIONS:
        raise ValueError(f"Unknown capture generation {generation!r}; known: {GENERATIONS}")
    api = HfApi()
    # list() INSIDE the retried thunk: list_repo_tree is a LAZY generator —
    # the HTTP error raises at iteration time (#779), and pagination 504s are
    # un-retried upstream (#658/#833) — so materialize under retry_transient.
    entries = retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient; list() inside the thunk
            api.list_repo_tree(
                repo_id=DATA_REPO,
                path_in_repo=BANKED_PREFIX,
                repo_type="dataset",
                revision=revision,
            )
        ),
        what=f"list_repo_tree({BANKED_PREFIX})",
    )
    turnstores: list[dict[str, str]] = []
    seen: dict[tuple[str, str, str], str] = {}
    unparsed: list[str] = []
    n_other_generation = 0
    for e in entries:
        name = Path(e.path).name
        if not name.startswith("turnstore_"):
            continue
        if turnstore_generation(name) != generation:
            n_other_generation += 1
            continue
        parsed = parse_turnstore_name(name)
        if parsed is None:
            unparsed.append(name)
            continue
        stage, render, corpus = parsed
        key = (stage, render, corpus)
        if key in seen:
            raise ValueError(
                f"Ambiguous turnstore cell {key}: BOTH {seen[key]} and {e.path} parse to it "
                f"within capture generation {generation!r} — canonical identity is ambiguous; "
                "fix the store naming before consuming."
            )
        seen[key] = e.path
        turnstores.append({"stage": stage, "render": render, "corpus": corpus, "tree_path": e.path})
    if n_other_generation:
        print(
            f"[generation-pin] {n_other_generation} turnstore(s) under {BANKED_PREFIX} outside "
            f"the {generation!r} capture generation SKIPPED (registered grid: plan v7 amendment; "
            "override with --generation)"
        )
    if unparsed:
        print(
            f"[WARN] {len(unparsed)} turnstore name(s) under {BANKED_PREFIX} did not parse "
            f"against the realized naming vocabulary and were SKIPPED: {unparsed}"
        )
    return turnstores


def resolve_turnstore_tree(
    stage: str,
    render: str,
    corpus: str,
    revision: str | None = None,
    generation: str = REGISTERED_GENERATION,
) -> str:
    """REALIZED repo tree path for one canonical (stage, render, corpus) cell.

    Resolved against the live enumeration of the REGISTERED capture
    generation (plan v7 amendment; `generation` override for the parked v1
    arm) — consumers never hand-build a `turnstore_{stage}_...` name: the
    realized store carries `v2_` prefixes and the `rlvr_long` stage token,
    and a hand-built canonical name 404s (the unit-E live probe caught
    fitness.py doing exactly that for lmsys23k). Fail-loud
    FileNotFoundError names the realized combos for the render.
    """
    stores = _stage_render_corpus_turnstores(revision=revision, generation=generation)
    for t in stores:
        if (t["stage"], t["render"], t["corpus"]) == (stage, render, corpus):
            return t["tree_path"]
    available = sorted((t["stage"], t["corpus"]) for t in stores if t["render"] == render)
    raise FileNotFoundError(
        f"No realized #1336 turnstore for (stage={stage!r}, render={render!r}, "
        f"corpus={corpus!r}) in capture generation {generation!r}; realized "
        f"(stage, corpus) combos for render={render!r}: {available}"
    )


def hub_shard_files(tree_path: str, revision: str | None = None) -> list[str]:
    """Repo-relative paths of ALL `*_shardNNN.pt` files under one turnstore tree.

    Scoped `list_repo_tree(path_in_repo=<tree>)` per the #833 recipe (never a
    bare full-repo listing); numeric shard-index sort; fail-loud when the tree
    holds no shards. Shared with `issue2061_fitness.py`. Retried via
    `hub.retry_transient` with the listing materialized INSIDE the thunk
    (lazy-generator + un-retried pagination-504 traps, #779/#833; review C5).
    """
    api = HfApi()
    entries = retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient; list() inside the thunk
            api.list_repo_tree(
                repo_id=DATA_REPO,
                path_in_repo=tree_path,
                repo_type="dataset",
                revision=revision,
            )
        ),
        what=f"list_repo_tree({tree_path})",
    )
    hits: list[tuple[int, str]] = []
    for e in entries:
        m = ts.SHARD_NAME_RE.search(Path(e.path).name)
        if m:
            hits.append((int(m.group(1)), e.path))
    if not hits:
        raise FileNotFoundError(
            f"No '*_shardNNN.pt' files under {DATA_REPO}/{tree_path} "
            f"(revision={revision}); tree entries: "
            f"{[Path(e.path).name for e in entries][:10]}"
        )
    return [p for _, p in sorted(hits)]


def iter_local_shards(tree_path: str, revision: str | None = None):
    """Yield LOCAL paths of one turnstore's shards, downloading lazily in order.

    Lazy so `load_state_from_shards(max_rows=...)` stops FETCHING once enough
    rows are accumulated (the smoke gate needs ~2-3 shards, not all ~46).
    ALSO fetches each shard's `.json` sidecar (same repo-relative dir, so it
    lands ADJACENT to the `.pt` in the snapshot cache) — the loader's
    per-shard sidecar asserts read it (plan v13 delta a1-bis).
    """
    for rel in hub_shard_files(tree_path, revision=revision):
        sidecar_rel = rel.removesuffix(".pt") + ".json"
        retry_transient(
            lambda rel=sidecar_rel: hf_hub_download(
                repo_id=DATA_REPO,
                filename=rel,
                repo_type="dataset",
                revision=revision,
            ),
            what=f"issue2061 sidecar {sidecar_rel}",
        )
        yield retry_transient(
            lambda rel=rel: hf_hub_download(
                repo_id=DATA_REPO,
                filename=rel,
                repo_type="dataset",
                revision=revision,
            ),
            what=f"issue2061 shard {rel}",
        )


def _load_turnstore_state(
    tree_path: str,
    state: str,
    layer: int,
    revision: str | None = None,
    max_rows: int | None = None,
    check_sidecars: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    """((n_rows, d_in) float32 rows, row-aligned conv_ids) for one state.

    Real #1336 payload schema + state extraction live in
    `issue2061_turnstore.extract_state_rows` (fail-loud schema assert per
    shard); each shard's JSON sidecar is additionally asserted
    (schema/convention, plan v13 delta a1-bis). `state`: "answer" (a1 turn
    profile — the plan target Y), "context" (a1-header slot state), or
    "prefix" (prefix-header slot state). conv_ids ride into the encoded
    payload so consumers can KEY the X/Y row alignment instead of trusting
    shard order (review M1).
    """
    return ts.load_state_from_shards(
        iter_local_shards(tree_path, revision=revision),
        state=state,
        layer=layer,
        max_rows=max_rows,
        check_sidecars=check_sidecars,
    )


def resolve_turnstore_trees(
    stage: str,
    render: str,
    corpus: str,
    revision: str | None = None,
    generation: str = REGISTERED_GENERATION,
    *,
    v2_stores: list[dict] | None = None,
    v1_stores: list[dict] | None = None,
) -> list[str]:
    """ORDERED realized tree paths one cell consumes (registered grain, v11).

    For the extended corpora (`ts.V2_CONCAT_SOURCES`) under the registered v2
    generation: `[wave-1 source tree, v2 extension tree]` — the canonical
    concat order (wave-1 rows FIRST; plan §Design "Registered consumption
    grain"). Standalone corpora (and any `generation="v1"` robustness-arm
    resolution) return their single store. `v2_stores` / `v1_stores` accept
    pre-fetched enumerations so batch callers (P0 grain gate, encode main)
    resolve all 35 cells against TWO listings instead of 50.
    """

    def _find(stores: list[dict], want_corpus: str, gen: str) -> str:
        for t in stores:
            if (t["stage"], t["render"], t["corpus"]) == (stage, render, want_corpus):
                return t["tree_path"]
        available = sorted((t["stage"], t["corpus"]) for t in stores if t["render"] == render)
        raise FileNotFoundError(
            f"No realized #1336 turnstore for (stage={stage!r}, render={render!r}, "
            f"corpus={want_corpus!r}) in capture generation {gen!r}; realized "
            f"(stage, corpus) combos for render={render!r}: {available}"
        )

    if generation != "v2" or corpus not in ts.V2_CONCAT_SOURCES:
        if generation == "v2" and v2_stores is not None:
            stores = v2_stores
        elif generation == "v1" and v1_stores is not None:
            stores = v1_stores
        else:
            stores = _stage_render_corpus_turnstores(revision=revision, generation=generation)
        return [_find(stores, corpus, generation)]
    src_corpus = ts.V2_CONCAT_SOURCES[corpus]
    v2s = (
        v2_stores
        if v2_stores is not None
        else _stage_render_corpus_turnstores(revision=revision, generation="v2")
    )
    v1s = (
        v1_stores
        if v1_stores is not None
        else _stage_render_corpus_turnstores(revision=revision, generation="v1")
    )
    return [_find(v1s, src_corpus, "v1"), _find(v2s, corpus, "v2")]


def _load_turnstore_state_cell(
    turnstore: dict,
    state: str,
    layer: int,
    revision: str | None = None,
    max_rows: int | None = None,
) -> tuple[torch.Tensor, list[str], dict]:
    """Cell-grain state rows over the cell's ORDERED tree list (hub path).

    Consumes `turnstore["tree_paths"]` (falling back to the single
    `tree_path`), loading each tree's shards in shard-index order and — for a
    two-tree concat cell — applying the ported boundary + disjointness
    asserts (`ts.assert_concat_boundary`; plan v11 delta a1/a2). A `max_rows`
    debug cap that exhausts inside the FIRST tree skips the remaining
    tree(s); the concat asserts then cover only the loaded parts (debug-only:
    the `_rows{N}` filename suffix already fences such payloads from
    production reuse).
    """
    trees = list(turnstore.get("tree_paths") or [turnstore["tree_path"]])
    if turnstore["corpus"] in ts.V2_CONCAT_SOURCES and len(trees) != 2:
        raise ValueError(
            f"extended corpus {turnstore['corpus']!r} consumed with {len(trees)} tree(s) — "
            "the REGISTERED grain is wave-1 source + v2 extension (plan v11; resolve via "
            "resolve_turnstore_trees). Extension-only consumption is the retired defect."
        )
    xs: list[torch.Tensor] = []
    ids_parts: list[list[str]] = []
    remaining = max_rows
    for tree in trees:
        if remaining is not None and remaining <= 0:
            break
        x, cids = _load_turnstore_state(
            tree, state=state, layer=layer, revision=revision, max_rows=remaining
        )
        xs.append(x)
        ids_parts.append(cids)
        if remaining is not None:
            remaining -= len(cids)
    concat = len(ids_parts) == 2
    if concat:
        ts.assert_concat_boundary(ids_parts[0], ids_parts[1], turnstore["corpus"])
    conv_ids = [c for part in ids_parts for c in part]
    if len(set(conv_ids)) != len(conv_ids):
        raise ValueError(
            f"cell {turnstore['stage']}/{turnstore['render']}/{turnstore['corpus']}: "
            f"duplicate conv_ids in the consumed row set (n={len(conv_ids)}, "
            f"unique={len(set(conv_ids))})."
        )
    x = torch.cat(xs, dim=0).contiguous() if len(xs) > 1 else xs[0]
    parts = [{"tree_path": t, "n_rows": len(p)} for t, p in zip(trees, ids_parts)]
    composition = " + ".join(f"{Path(p['tree_path']).name}:{p['n_rows']}" for p in parts)
    print(
        f"[cell-load] {turnstore['stage']}/{turnstore['render']}/{turnstore['corpus']} "
        f"state={state}: {composition} = {len(conv_ids)} rows"
        + (" (concat; boundary+disjointness PASS)" if concat else ""),
        flush=True,
    )
    return x, conv_ids, {"concat": concat, "parts": parts, "n_rows": len(conv_ids)}


def loader_parity_smoke_gate(
    layer: int = LAYER,
    n_smoke_rows: int = 1000,
    bar: float = 0.05,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sae_revision: str | None = None,
    data_revision: str | None = None,
    state: str = "answer",
    generation: str = REGISTERED_GENERATION,
) -> tuple[float, float, float]:
    """FVE-parity check: our ported encode vs `sparsify`'s own on the same rows.

    Returns (FVE_ported, FVE_reference, |Δ|). Raises RuntimeError if
    `|Δ| >= bar`.

    NOTE: Requires `sparsify` installed one-off (not a runtime dep). Install:
        uv pip install sparsify
    """
    try:
        from sparsify import SparseCoder  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Loader-parity smoke gate requires the sparsify package (one-off). "
            "Install via `uv pip install sparsify` and re-run."
        ) from e

    # Find one LMSYS base-model turnstore for smoke input.
    turnstores = _stage_render_corpus_turnstores(revision=data_revision, generation=generation)
    base_lmsys = [t for t in turnstores if t["stage"] == "base" and t["corpus"].startswith("lmsys")]
    if not base_lmsys:
        raise RuntimeError("No base-stage LMSYS turnstore found for smoke gate.")
    smoke_ts = base_lmsys[0]
    print(
        f"[smoke] Loading {n_smoke_rows} rows from {smoke_ts['tree_path']} "
        f"state={state} layer={layer}"
    )

    x, _conv_ids = _load_turnstore_state(
        smoke_ts["tree_path"],
        state=state,
        layer=layer,
        revision=data_revision,
        max_rows=n_smoke_rows,
    )
    x = x.to(device)

    # Ported encode + reconstruct.
    print("[smoke] Loading SAE weights (ported)")
    weights, cfg = load_sae_weights(SAE_REPO, layer=layer, revision=sae_revision, device=device)
    k = int(cfg["k"])
    with torch.no_grad():
        z_ported = topk_encode(x, weights, k=k)
        x_recon_ported = topk_reconstruct(z_ported, weights)
    fve_ported = _fve(x, x_recon_ported)
    print(f"[smoke] FVE (ported) = {fve_ported:.4f}")

    # Reference: sparsify's own SparseCoder.load + encode.
    print("[smoke] Loading SAE weights (sparsify reference)")
    from sparsify import SparseCoder

    ref = SparseCoder.load_from_hub(SAE_REPO, hookpoint=f"layers.{layer}", device=device)
    with torch.no_grad():
        # `SparseCoder.encode` returns an EncoderOutput namedtuple
        # (top_acts, top_indices, pre_acts) -- a TopK SAE's sparse code is a
        # (values, indices) PAIR, not a dense vector -- and `decode` takes
        # BOTH: decode(top_acts, top_indices). Passing the EncoderOutput
        # itself raises `TypeError: SparseCoder.decode() missing 1 required
        # positional argument: 'top_indices'` (verified against the installed
        # eai-sparsify 1.3.3 signature).
        enc_ref = ref.encode(x)
        x_recon_ref = ref.decode(enc_ref.top_acts, enc_ref.top_indices)
    fve_ref = _fve(x, x_recon_ref)
    print(f"[smoke] FVE (sparsify) = {fve_ref:.4f}")

    delta = abs(fve_ported - fve_ref)
    print(f"[smoke] |Δ| = {delta:.4f} (bar: {bar})")

    if delta >= bar:
        raise RuntimeError(
            f"Loader-parity smoke gate FAILED: |FVE_ported - FVE_reference| = "
            f"{delta:.4f} >= {bar}. Do NOT proceed to production encode."
        )
    return fve_ported, fve_ref, delta


def _fve(x: torch.Tensor, x_recon: torch.Tensor) -> float:
    """Fraction of variance explained: 1 - var(x - x_recon) / var(x).

    Sum of per-dim unbiased variances (matches #1482 recipe;
    `.claude/rules/gotchas.md` #1482 entry).
    """
    numerator = (x - x_recon).var(dim=0, unbiased=True).sum().item()
    denominator = x.var(dim=0, unbiased=True).sum().item()
    return 1.0 - (numerator / denominator)


def encode_turnstore(
    turnstore: dict[str, str],
    weights: dict[str, torch.Tensor],
    k: int,
    output_dir: Path,
    layer: int = LAYER,
    batch_size: int = BATCH_SIZE,
    device: str = "cuda",
    max_rows: int | None = None,
    revision: str | None = None,
    state: str = "answer",
) -> Path:
    """Encode one (stage, render, corpus) turnstore's `state` rows (ALL shards).

    Default `state="answer"` — the plan §Design target Y (SAE encode of the
    a1 answer-profile rows). Writes the fixed-width TopK SPARSE payload
    (`issue2061_turnstore.ENCODED_TARGET_FORMAT`: (n, k) idx int32 + val
    float32 + row-aligned conv_ids) via `ts.save_encoded_target` — ~6 MB for
    the largest lmsys23k cell vs ~24 GB dense (review M1; the sparse pair
    scatters back to EXACTLY `topk_encode`'s dense output, pinned by
    tests/test_issue2061_loaders.py). Peak RAM = the per-batch dense
    pre-activation buffer (~268 MB at batch 256) + the accumulated (n, k)
    sparse pair — no terminal dense concat. NOTE the plan §10 path template
    omits `<render>`; it is included here to keep the two renders of one
    (stage, corpus) from colliding (deviation flagged in the Unit A report).

    Skip predicate (review M3): a `--max-rows` debug cap writes a `_rows{N}`
    suffixed filename, so a capped shard can never be skip-reused by a
    production run; an existing file is skip-reused ONLY when it parses as
    the current sparse format for the same cell AND the same consumption
    grain (v11: a stale extension-only payload at the same path must be
    re-encoded loudly at the concat grain, never silently reused) — a stale
    dense store or a foreign-cell payload is re-encoded loudly, never
    consumed.
    """
    cell = {
        "stage": turnstore["stage"],
        "render": turnstore["render"],
        "corpus": turnstore["corpus"],
        "state": state,
        "layer": layer,
    }
    trees = list(turnstore.get("tree_paths") or [turnstore["tree_path"]])
    grain = "concat-v11" if len(trees) == 2 else "single-store"
    output_path = output_dir / ts.encoded_target_name(
        turnstore["stage"],
        turnstore["render"],
        turnstore["corpus"],
        state,
        layer,
        max_rows=max_rows,
    )
    if output_path.exists():
        try:
            existing = ts.load_encoded_target(output_path)
        except Exception as e:  # stale/foreign payload: re-encode, never consume
            print(f"[re-encode] {output_path} exists but is not reusable ({e})")
        else:
            if (
                existing["cell"] == cell
                and existing["meta"].get("max_rows") == max_rows
                and existing["meta"].get("consumption_grain") == grain
            ):
                print(f"[skip] {output_path} exists ({existing['n_rows']} rows, valid payload)")
                return output_path
            print(
                f"[re-encode] {output_path}: regime mismatch "
                f"(cell={existing['cell']} max_rows={existing['meta'].get('max_rows')} "
                f"grain={existing['meta'].get('consumption_grain')} != {grain})"
            )

    x, conv_ids, _load_info = _load_turnstore_state_cell(
        {**turnstore, "tree_paths": trees},
        state=state,
        layer=layer,
        revision=revision,
        max_rows=max_rows,
    )
    x = x.to(device)
    n = x.shape[0]
    print(f"[encode] {' + '.join(trees)} state={state} L{layer} n={n} d_in={x.shape[1]}")

    idx_chunks: list[torch.Tensor] = []
    val_chunks: list[torch.Tensor] = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            vals, idx = topk_encode_sparse(x[start:end], weights, k=k)
            idx_chunks.append(idx.cpu())
            val_chunks.append(vals.cpu())
            if (start // batch_size) % 4 == 0:
                elapsed = time.time() - t0
                print(f"  {end}/{n} rows in {elapsed:.1f}s ({end / elapsed:.1f} rows/s)")

    d_sae = int(weights["encoder.weight"].shape[0])
    ts.save_encoded_target(
        output_path,
        idx=torch.cat(idx_chunks, dim=0),
        val=torch.cat(val_chunks, dim=0),
        d_sae=d_sae,
        k=k,
        conv_ids=conv_ids,
        cell=cell,
        extra_meta={
            "max_rows": max_rows,
            "sae_repo": SAE_REPO,
            "data_repo": DATA_REPO,
            "tree_path": turnstore["tree_path"],
            "tree_paths": trees,
            "consumption_grain": grain,
            "batch_size": batch_size,
        },
    )
    print(f"[write] {output_path} n={n} k={k} d_sae={d_sae} format={ts.ENCODED_TARGET_FORMAT}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--render", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument(
        "--smoke-only", action="store_true", help="Only run loader-parity FVE smoke gate"
    )
    parser.add_argument(
        "--smoke-then-encode", action="store_true", help="Run smoke gate first, then encode"
    )
    parser.add_argument("--smoke-bar", type=float, default=0.05)
    parser.add_argument("--smoke-rows", type=int, default=1000)
    parser.add_argument(
        "--state",
        type=str,
        choices=sorted(ts.STATE_SPEC),
        default="answer",
        help="Which banked state to encode (default: answer — the plan target Y).",
    )
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows per turnstore (debug)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/issue_2061/sae_encoded"))
    parser.add_argument("--sae-revision", type=str, default=None)
    parser.add_argument("--data-revision", type=str, default=None)
    parser.add_argument(
        "--generation",
        type=str,
        choices=GENERATIONS,
        default=REGISTERED_GENERATION,
        help="#1336 capture generation to enumerate/encode (default: the REGISTERED "
        "v2 grid — plan v7 amendment: 7 combos / 35 stores / 56 delta cells; 'v1' "
        "preserves the parked lower-n robustness arm, which needs its own plan "
        "approval before any production dispatch).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="After encoding, upload --output-dir to the HF data repo "
        "(issue2061_hub_io 'sae-encoded' prefix) with an exact-set verify — "
        "plan §9 off_pod_phases: P1's outputs MUST land on HF before its GPU "
        "pod terminates (the #521/#1482 downstream-input loss class).",
    )
    args = parser.parse_args()

    # Smoke gate first, ALWAYS when --smoke-only or --smoke-then-encode.
    if args.smoke_only or args.smoke_then_encode:
        print(f"=== Loader-parity FVE smoke gate (layer {args.layer}) ===")
        fve_p, fve_r, delta = loader_parity_smoke_gate(
            layer=args.layer,
            n_smoke_rows=args.smoke_rows,
            bar=args.smoke_bar,
            device=args.device,
            sae_revision=args.sae_revision,
            data_revision=args.data_revision,
            state=args.state,
            generation=args.generation,
        )
        print(f"=== SMOKE PASS: |Δ| = {delta:.4f} < {args.smoke_bar} ===")
        if args.smoke_only:
            return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Output dir: {args.output_dir.resolve()}")

    # Enumerate + filter the target turnstores BEFORE the ~8.6 GB SAE weights
    # download (review m4): a bad store enumeration / empty filter then costs
    # seconds, not a weights download. Enumeration depends only on
    # --data-revision, never on the weights.
    print(f"[setup] Enumerating banked turnstores (generation={args.generation})")
    all_turnstores = _stage_render_corpus_turnstores(
        revision=args.data_revision, generation=args.generation
    )
    print(f"[setup] Found {len(all_turnstores)} turnstores")

    if args.all_cells:
        targets = all_turnstores
    else:
        targets = [
            t
            for t in all_turnstores
            if (args.stage is None or t["stage"] == args.stage)
            and (args.render is None or t["render"] == args.render)
            and (args.corpus is None or t["corpus"] == args.corpus)
        ]
    if not targets:
        print(
            f"[error] No turnstores match filters (stage={args.stage} render={args.render} corpus={args.corpus})"
        )
        return 1

    # Registered consumption grain (plan v11 delta a2): each target cell
    # resolves its ORDERED tree list — the wave-1 concat source + the v2
    # extension for the extended corpora, the single store otherwise. The v1
    # listing is fetched ONCE for all concat targets (never per cell).
    v1_stores = None
    if args.generation == "v2" and any(t["corpus"] in ts.V2_CONCAT_SOURCES for t in targets):
        v1_stores = _stage_render_corpus_turnstores(revision=args.data_revision, generation="v1")
    for t in targets:
        t["tree_paths"] = resolve_turnstore_trees(
            t["stage"],
            t["render"],
            t["corpus"],
            revision=args.data_revision,
            generation=args.generation,
            v2_stores=all_turnstores,
            v1_stores=v1_stores,
        )
    n_stores = len({tree for t in targets for tree in t["tree_paths"]})
    print(f"[setup] Target: {len(targets)} cell(s) consuming {n_stores} store(s)")

    print(f"[setup] Loading SAE weights layer={args.layer}")
    weights, cfg = load_sae_weights(
        SAE_REPO, layer=args.layer, revision=args.sae_revision, device=args.device
    )
    k = int(cfg["k"])
    print(f"[setup] SAE: k={k}, d_sae={weights['encoder.weight'].shape[0]}, d_in={cfg['d_in']}")

    manifest_path = args.output_dir / "encode_manifest.jsonl"
    with manifest_path.open("a") as manifest:
        # NOTE: loop var deliberately NOT named `ts` — that would shadow the
        # issue2061_turnstore module alias inside main().
        for i, cell in enumerate(targets, start=1):
            print(
                f"\n=== [{i}/{len(targets)}] {cell['stage']}/{cell['render']}/{cell['corpus']} ==="
            )
            t0 = time.time()
            output_path = encode_turnstore(
                cell,
                weights=weights,
                k=k,
                output_dir=args.output_dir,
                layer=args.layer,
                batch_size=args.batch_size,
                device=args.device,
                max_rows=args.max_rows,
                revision=args.data_revision,
                state=args.state,
            )
            elapsed = time.time() - t0
            manifest.write(
                json.dumps(
                    {
                        "stage": cell["stage"],
                        "render": cell["render"],
                        "corpus": cell["corpus"],
                        "state": args.state,
                        "layer": args.layer,
                        "format": ts.ENCODED_TARGET_FORMAT,
                        "consumption_grain": (
                            "concat-v11"
                            if len(cell.get("tree_paths") or [0]) == 2
                            else "single-store"
                        ),
                        "max_rows": args.max_rows,
                        "output_path": str(output_path.relative_to(args.output_dir.parent.parent))
                        if output_path.is_absolute()
                        else str(output_path),
                        "elapsed_s": elapsed,
                    }
                )
                + "\n"
            )

    print(f"\n[done] Manifest: {manifest_path}")

    if args.upload:
        # Cross-machine handoff (plan §9 off_pod_phases): the encoded targets
        # are P2/P3's inputs on OTHER machines — verified upload before the
        # pod terminates, fail-loud on any missing dest. delete_local=False:
        # a same-pod P4 (or debugging) may still read them; the pod's local
        # copies die with the pod either way.
        import issue2061_hub_io as hio

        hio.upload_dir(args.output_dir, "sae-encoded")

    return 0


if __name__ == "__main__":
    sys.exit(main())
