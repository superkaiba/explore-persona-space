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

Plan §Design + §9 P1: 5 stages × 5 corpus stems × 2 renders = ~50
turnstores; each turnstore holds MANY `*_shardNNN.pt` files of ≤500
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


def _stage_render_corpus_turnstores(revision: str | None = None) -> list[dict[str, str]]:
    """Enumerate #1336's banked turnstores as CANONICAL (stage, render, corpus).

    Reads `list_repo_tree(path_in_repo="issue1336_rlvr_ladder/analysis_tensors")`
    and parses the realized `turnstore_[v2_]<stage>_<render>_<corpus>` names
    (see STORE_STAGE_TOKENS above). Returns dicts with keys: stage, render,
    corpus, tree_path — `tree_path` keeps the REALIZED repo path (what gets
    fetched); the identity keys are canonical. Unparseable turnstore names
    are WARNed (never silently dropped); two realized trees mapping to one
    canonical cell fail loud (a v1/v2 corpus-family collision would make the
    cell ambiguous).
    """
    api = HfApi()
    # list() INSIDE the retried thunk: list_repo_tree is a LAZY generator —
    # the HTTP error raises at iteration time (#779), and pagination 504s are
    # un-retried upstream (#658/#833) — so materialize under retry_transient.
    entries = retry_transient(
        lambda: list(
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
    for e in entries:
        name = Path(e.path).name
        if not name.startswith("turnstore_"):
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
                "— the v1/v2 capture generations now overlap on a corpus; pin the family "
                "explicitly before consuming."
            )
        seen[key] = e.path
        turnstores.append({"stage": stage, "render": render, "corpus": corpus, "tree_path": e.path})
    if unparsed:
        print(
            f"[WARN] {len(unparsed)} turnstore name(s) under {BANKED_PREFIX} did not parse "
            f"against the realized naming vocabulary and were SKIPPED: {unparsed}"
        )
    return turnstores


def resolve_turnstore_tree(
    stage: str, render: str, corpus: str, revision: str | None = None
) -> str:
    """REALIZED repo tree path for one canonical (stage, render, corpus) cell.

    Resolved against the live enumeration (so consumers never hand-build a
    `turnstore_{stage}_...` name — the realized store carries `v2_` prefixes
    and the `rlvr_long` stage token, and a hand-built canonical name 404s;
    the unit-E live probe caught fitness.py doing exactly that for lmsys23k).
    Fail-loud FileNotFoundError names the realized combos for the render.
    """
    stores = _stage_render_corpus_turnstores(revision=revision)
    for t in stores:
        if (t["stage"], t["render"], t["corpus"]) == (stage, render, corpus):
            return t["tree_path"]
    available = sorted((t["stage"], t["corpus"]) for t in stores if t["render"] == render)
    raise FileNotFoundError(
        f"No realized #1336 turnstore for (stage={stage!r}, render={render!r}, "
        f"corpus={corpus!r}); realized (stage, corpus) combos for render={render!r}: "
        f"{available}"
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
    """
    for rel in hub_shard_files(tree_path, revision=revision):
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
) -> tuple[torch.Tensor, list[str]]:
    """((n_rows, d_in) float32 rows, row-aligned conv_ids) for one state.

    Real #1336 payload schema + state extraction live in
    `issue2061_turnstore.extract_state_rows` (fail-loud schema assert per
    shard). `state`: "answer" (a1 turn profile — the plan target Y),
    "context" (a1-header slot state), or "prefix" (prefix-header slot state).
    conv_ids ride into the encoded payload so consumers can KEY the X/Y row
    alignment instead of trusting shard order (review M1).
    """
    return ts.load_state_from_shards(
        iter_local_shards(tree_path, revision=revision),
        state=state,
        layer=layer,
        max_rows=max_rows,
    )


def loader_parity_smoke_gate(
    layer: int = LAYER,
    n_smoke_rows: int = 1000,
    bar: float = 0.05,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sae_revision: str | None = None,
    data_revision: str | None = None,
    state: str = "answer",
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
    turnstores = _stage_render_corpus_turnstores(revision=data_revision)
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
        z_ref = ref.encode(x)
        x_recon_ref = ref.decode(z_ref)
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
    the current sparse format for the same cell — a stale dense store or a
    foreign-cell payload is re-encoded loudly, never consumed.
    """
    cell = {
        "stage": turnstore["stage"],
        "render": turnstore["render"],
        "corpus": turnstore["corpus"],
        "state": state,
        "layer": layer,
    }
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
            if existing["cell"] == cell and existing["meta"].get("max_rows") == max_rows:
                print(f"[skip] {output_path} exists ({existing['n_rows']} rows, valid payload)")
                return output_path
            print(
                f"[re-encode] {output_path}: regime mismatch "
                f"(cell={existing['cell']} max_rows={existing['meta'].get('max_rows')})"
            )

    x, conv_ids = _load_turnstore_state(
        turnstore["tree_path"],
        state=state,
        layer=layer,
        revision=revision,
        max_rows=max_rows,
    )
    x = x.to(device)
    n = x.shape[0]
    print(f"[encode] {turnstore['tree_path']} state={state} L{layer} n={n} d_in={x.shape[1]}")

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
        )
        print(f"=== SMOKE PASS: |Δ| = {delta:.4f} < {args.smoke_bar} ===")
        if args.smoke_only:
            return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Output dir: {args.output_dir.resolve()}")

    print(f"[setup] Loading SAE weights layer={args.layer}")
    weights, cfg = load_sae_weights(
        SAE_REPO, layer=args.layer, revision=args.sae_revision, device=args.device
    )
    k = int(cfg["k"])
    print(f"[setup] SAE: k={k}, d_sae={weights['encoder.weight'].shape[0]}, d_in={cfg['d_in']}")

    print("[setup] Enumerating banked turnstores")
    all_turnstores = _stage_render_corpus_turnstores(revision=args.data_revision)
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
    print(f"[setup] Target: {len(targets)} turnstore(s)")

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
