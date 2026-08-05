"""P1 SAE encode phase for task #2061.

Reads #1336's banked layer-29 activations from the HF data repo
(`superkaiba1/explore-persona-space-data/issue1336_rlvr_ladder/analysis_tensors/turnstore_<stage>_<render>_<corpus>/`),
runs the fixed EleutherAI/sae-llama-3.1-8b-64x TopK encoder (via the
ported `topk_encode` in analysis/sparsify_topk_sae), and emits per-cell
SAE-feature-vector targets under `data/issue_2061/sae_encoded/` as
`<stage>_<render>_<corpus>_<state>_L<layer>.pt` (default state: `answer`
— the plan §Design target Y = SAE encode of the ANSWER state, i.e. the
a1 turn profile of the #1336 turnstore; see
`scripts/issue2061_turnstore.py` for the realized payload schema +
pooling convention).

Plan §Design + §9 P1: 5 stages × 5 corpus stems × 2 renders = ~50
turnstores; each turnstore holds MANY `*_shardNNN.pt` files of ≤500
records each (`SHARD_SIZE = 500`, issue1336_extract_turnstore.py), each
carrying all 32 layers (~525 MB/shard). ALL shards of a turnstore are
enumerated + concatenated in shard-index order. Batched encode at ~256
rows/batch; ~5 GB output per shard chunk.

Loader-parity FVE smoke gate (plan §Design 'Loader adapter'):
    |FVE_ported - FVE_reference| < 0.05
runs as the FIRST action when --smoke-only or --smoke-then-encode is set.
On failure, HALTs with exit code 2 (fail-loud); do NOT proceed to
production encode with an unverified loader.

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

import torch
from huggingface_hub import HfApi, hf_hub_download

from explore_persona_space.analysis.sparsify_topk_sae import (
    load_sae_weights,
    topk_encode,
    topk_reconstruct,
)

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


def _stage_render_corpus_turnstores(revision: str | None = None) -> list[dict[str, str]]:
    """Enumerate #1336's banked turnstores as (stage, render, corpus) triples.

    Reads `list_repo_tree(path_in_repo="issue1336_rlvr_ladder/analysis_tensors")`
    and matches the `turnstore_<stage>_<render>_<corpus>` directory pattern.
    Returns a list of dicts with keys: stage, render, corpus, tree_path.
    """
    api = HfApi()
    entries = list(
        api.list_repo_tree(
            repo_id=DATA_REPO,
            path_in_repo=BANKED_PREFIX,
            repo_type="dataset",
            revision=revision,
        )
    )
    turnstores: list[dict[str, str]] = []
    for e in entries:
        name = Path(e.path).name
        if not name.startswith("turnstore_"):
            continue
        # turnstore_<stage>_<render>_<corpus>
        parts = name.removeprefix("turnstore_").split("_", 2)
        if len(parts) != 3:
            continue
        stage, render, corpus = parts
        turnstores.append(
            {
                "stage": stage,
                "render": render,
                "corpus": corpus,
                "tree_path": e.path,
            }
        )
    return turnstores


def hub_shard_files(tree_path: str, revision: str | None = None) -> list[str]:
    """Repo-relative paths of ALL `*_shardNNN.pt` files under one turnstore tree.

    Scoped `list_repo_tree(path_in_repo=<tree>)` per the #833 recipe (never a
    bare full-repo listing); numeric shard-index sort; fail-loud when the tree
    holds no shards. Shared with `issue2061_fitness.py`.
    """
    api = HfApi()
    entries = list(
        api.list_repo_tree(
            repo_id=DATA_REPO,
            path_in_repo=tree_path,
            repo_type="dataset",
            revision=revision,
        )
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
        yield hf_hub_download(
            repo_id=DATA_REPO,
            filename=rel,
            repo_type="dataset",
            revision=revision,
        )


def _load_turnstore_state(
    tree_path: str,
    state: str,
    layer: int,
    revision: str | None = None,
    max_rows: int | None = None,
) -> torch.Tensor:
    """(n_rows, d_in) float32 rows of one state across ALL shards of a turnstore.

    Real #1336 payload schema + state extraction live in
    `issue2061_turnstore.extract_state_rows` (fail-loud schema assert per
    shard). `state`: "answer" (a1 turn profile — the plan target Y),
    "context" (a1-header slot state), or "prefix" (prefix-header slot state).
    """
    x, _conv_ids = ts.load_state_from_shards(
        iter_local_shards(tree_path, revision=revision),
        state=state,
        layer=layer,
        max_rows=max_rows,
    )
    return x


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

    x = _load_turnstore_state(
        smoke_ts["tree_path"],
        state=state,
        layer=layer,
        revision=data_revision,
        max_rows=n_smoke_rows,
    ).to(device)

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
    a1 answer-profile rows). Writes
    `<output_dir>/<stage>_<render>_<corpus>_<state>_L<layer>.pt` carrying the
    SAE feature matrix (n_rows, d_sae) as float32 (chunked to manage RAM;
    d_sae=262144 float32 at n_rows=1000 = ~1 GB). NOTE the plan §10 path
    template omits `<render>`; it is included here to keep the two renders of
    one (stage, corpus) from colliding (deviation flagged in the Unit A
    report).
    """
    output_path = (
        output_dir
        / f"{turnstore['stage']}_{turnstore['render']}_{turnstore['corpus']}_{state}_L{layer}.pt"
    )
    if output_path.exists():
        print(f"[skip] {output_path} exists")
        return output_path

    x = _load_turnstore_state(
        turnstore["tree_path"],
        state=state,
        layer=layer,
        revision=revision,
        max_rows=max_rows,
    ).to(device)
    n = x.shape[0]
    print(f"[encode] {turnstore['tree_path']} state={state} L{layer} n={n} d_in={x.shape[1]}")

    encoded_chunks: list[torch.Tensor] = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = x[start:end]
            z = topk_encode(batch, weights, k=k)
            encoded_chunks.append(z.cpu())
            if (start // batch_size) % 4 == 0:
                elapsed = time.time() - t0
                print(f"  {end}/{n} rows in {elapsed:.1f}s ({end / elapsed:.1f} rows/s)")

    encoded = torch.cat(encoded_chunks, dim=0)
    print(f"[write] {output_path} shape={tuple(encoded.shape)} dtype={encoded.dtype}")
    torch.save(encoded, output_path)
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
                        "output_path": str(output_path.relative_to(args.output_dir.parent.parent))
                        if output_path.is_absolute()
                        else str(output_path),
                        "elapsed_s": elapsed,
                    }
                )
                + "\n"
            )

    print(f"\n[done] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
