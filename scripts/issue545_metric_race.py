#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (≪, ※, ‖, ∅, ×) in scientific docstrings/comments.
"""Issue #545 metric-race dispatcher — `full-metric-race-per-family` follow-up.

Predictor-zoo + scoring-analysis amendment over the FROZEN v1 leakage matrix
(NO retraining, NO new training data). Single manipulated variable = the
expanded Group-A predictor zoo (#493 metric engine + #540 JS-RB harness) +
the three new scoring analyses. All 60 v1 cells, the panel, batteries, judges,
seeds are PINNED to v1 @ git ``0a60158f3`` + HF data revision
``96ccf2ccf...`` and REUSED.

Phases (smoke IS sweep with a tiny cell subset — same dispatcher, same
functions, only ``--rows``/``--cols`` differ; PASS_UNIFIED):

- **prefetch** (CPU, pre-GPU): materialize the pinned demos/corpora from HF at
  ``revision=PINNED_REV`` into ``production_corpora_dir()/<rel>``, SHA256-assert
  against ``expected_sha256.json``, assert the v1 loader resolves to the
  materialized file. ANY mismatch / missing file / wrong resolution → HALT
  before any GPU spend (plan §10 Blocker-2 fix).
- **extract** (GPU, 1× H100/A100): per-behavior activation clouds + JS/KL
  output-distribution RB estimates (predictors_zoo.extract_clouds_and_outdist_gpu),
  then CPU ``build_zoo_predictors`` over the cached clouds + the v1 reference
  predictors copied in → ``predictors_metric_race/*.json``. (``--stub-gpu``
  swaps a tiny CPU model for the local CPU smoke of the pre-GPU pipeline.)
- **score** (CPU, OFF-POD on the VM): Analysis 1/2/3 + bootstrap + permutation
  + H3 (scoring.score_metric_race) → ``metric_race/scoring_metric_race/``.
- **plot** (CPU, OFF-POD on the VM): hero figures + diagnostics
  (``issue545_plot_metric_race.py``).

The pod stops/terminates AFTER ``extract`` (CLAUDE.md "CPU-only phases don't
hold GPU pods"); ``score`` + ``plot`` run on the VM against the committed v1
matrix + the uploaded predictor JSONs.

CLI:
    # FULL (on pod): prefetch + extract
    nohup uv run python scripts/issue545_metric_race.py --phase prefetch,extract \\
        > logs/issue545_metric_race.log 2>&1 &
    # SMOKE (same dispatcher, 2 cells × 1 layer × 1 flavor, CPU stub):
    uv run python scripts/issue545_metric_race.py --phase prefetch,extract \\
        --rows bad_medical --cols capability --layers 21 --flavors demos \\
        --n-probes 50 --r-samples 8 --stub-gpu --out-root eval_results/issue_545_metric_race_smoke
    # SCORE + PLOT (VM):
    uv run python scripts/issue545_metric_race.py --phase score,plot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Pin HF cache to /workspace on pods; leave system default on the local VM.
if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

# PyTorch CUDA allocator: expandable_segments defragments reserved-but-
# unallocated memory. Under STRATEGY E (round-38) the HF base model and the vLLM
# engine no longer co-reside — vLLM runs in a subprocess (vllm_worker.py) that
# exits before the HF model loads — so the co-residency OOM is gone; the setting
# stays for the per-text hook path's allocate/free churn in the HF phase. MUST
# be set BEFORE the first `import torch`; this dispatcher imports torch only
# lazily inside phase_extract / _stub_extract, so this module-top setdefault runs
# first. setdefault (not assignment) so an explicit launcher / env override
# always wins. (#545 round-4; kept round-38.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# DOTENV_LINT_EXEMPT: legacy pre-#745 script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv  # noqa: E402

# load_dotenv at entry — every credential-using path (hf_hub_download) needs
# HF_TOKEN; `uv run python` does NOT auto-load .env (CLAUDE.md subprocess rule).
load_dotenv()

from explore_persona_space.experiments.behavior_testbed_545 import (  # noqa: E402
    corpus_read_path,
    output_root,
    production_corpora_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("issue545.metric_race")

PINNED_REV = "96ccf2ccf"  # placeholder; the real value is in expected_sha256.json
REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue545_behavior_testbed/corpora/"
MANIFEST_REL = "metric_race/expected_sha256.json"


# ---------------------------------------------------------------------------
# Phase 0 — pinned-prefetch + SHA256-assert (CPU, pre-GPU; HALT on mismatch)
# ---------------------------------------------------------------------------


def prefetch_pinned_inputs(*, manifest_path: Path | None = None) -> int:
    """Materialize the pinned demos/corpora from HF → production_corpora_dir(),
    SHA256-assert each, and assert the v1 loader resolves to the materialized
    file. ANY mismatch HALTs (sys.exit) BEFORE any GPU work. Returns the file
    count materialized.

    The download lands in the content-hashed HF cache; we MUST materialize each
    cache file to ``production_corpora_dir()/<rel>`` (hardlink, copy fallback)
    because the v1 loader resolves ``corpus_read_path(<rel>)`` ->
    ``data/issue545/corpora/<rel>``, never the cache path. The hardlink source
    is the RESOLVED cache path (the cache entry is a symlink into blobs/).
    """
    from huggingface_hub import hf_hub_download

    man_path = manifest_path or (output_root() / MANIFEST_REL)
    if not man_path.exists():
        # fall back to the v1-committed location (the manifest is git-committed
        # under eval_results/issue_545/metric_race/, not the override root).
        from explore_persona_space.experiments.behavior_testbed_545 import v1_committed_root

        man_path = v1_committed_root() / MANIFEST_REL
    if not man_path.exists():
        sys.exit(f"HALT: manifest {man_path} missing — generate it before prefetch")
    man = json.loads(man_path.read_text())
    pinned_rev = man["pinned_revision"]
    expected = man["expected_sha256"]
    prod_root = production_corpora_dir()
    prod_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for hf_path, want in expected.items():
        if not hf_path.startswith(PREFIX):
            sys.exit(f"HALT: unexpected manifest entry shape: {hf_path}")
        rel = hf_path[len(PREFIX) :]  # loader-relative path, e.g. demos/x.json
        target = prod_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        cached = Path(
            hf_hub_download(REPO, hf_path, repo_type="dataset", revision=pinned_rev)
        ).resolve()
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            os.link(cached, target)
        except OSError:
            shutil.copy(cached, target)
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        if got != want:
            sys.exit(f"HALT: {hf_path} sha256 {got} != expected {want} @ {pinned_rev}")
        resolved = corpus_read_path(rel)
        if resolved.resolve() != target.resolve():
            sys.exit(f"HALT: {rel} resolves to {resolved}, not {target}")
        n += 1
    logger.info(
        "[phase=prefetch] OK: %d files materialized at %s @ %s, all sha256 + path-resolve match",
        n,
        prod_root,
        pinned_rev,
    )
    return n


# ---------------------------------------------------------------------------
# Phase 1 — extract (GPU): clouds + JS/KL + build the expanded predictor pool
# ---------------------------------------------------------------------------


def _copy_v1_reference_predictors(metric_race_pred_dir: Path) -> int:
    """Copy v1's predictors/*.json (the raw-centroid A__geom_* reference +
    Groups B/C/D) into predictors_metric_race/ so the expanded race retains
    them verbatim alongside the new zoo (plan §4.1)."""
    from explore_persona_space.experiments.behavior_testbed_545 import v1_committed_root

    src = v1_committed_root() / "predictors"
    if not src.exists():
        sys.exit(f"HALT: v1 predictors dir {src} missing (frozen reuse input)")
    metric_race_pred_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.json")):
        dst = metric_race_pred_dir / p.name
        shutil.copy(p, dst)
        n += 1
    logger.info(
        "[phase=extract] copied %d v1 reference predictors into %s", n, metric_race_pred_dir
    )
    return n


def phase_extract(args) -> None:
    """GPU clouds + JS/KL, then CPU build of the expanded predictor pool."""
    from explore_persona_space.experiments.behavior_testbed_545 import predictors_zoo as zoo

    mr_root = output_root() / "metric_race"
    mr_root.mkdir(parents=True, exist_ok=True)
    pred_dir = mr_root / "predictors_metric_race"

    rows_subset = args.rows.split(",") if args.rows else None
    cols_subset = args.cols.split(",") if args.cols else None

    if args.stub_gpu:
        _stub_extract(args, mr_root, rows_subset, cols_subset)
    else:
        # Optional layer/flavor descope: monkey-patch the module-level grids
        # used by both extraction (clouds) and the CPU zoo build.
        if args.layers:
            zoo.GEOMETRY_LAYERS = tuple(int(x) for x in args.layers.split(","))
            from explore_persona_space.experiments.behavior_testbed_545 import predictors as p

            p.GEOMETRY_LAYERS = zoo.GEOMETRY_LAYERS
        if args.flavors:
            zoo.FLAVORS = tuple(args.flavors.split(","))
        summary = zoo.extract_clouds_and_outdist_gpu(
            mr_root,
            rows_subset=rows_subset,
            cols_subset=cols_subset,
            n_probes=args.n_probes,
            r_samples=args.r_samples,
            nl_cloud_samples=args.nl_cloud_samples,
        )
        logger.info("[phase=extract] GPU summary: %s", json.dumps(summary))

    # CPU: copy v1 reference + build the new zoo predictors.
    _copy_v1_reference_predictors(pred_dir)
    zoo.build_zoo_predictors(pred_dir, cloud_src_dir=mr_root)
    zoo.build_delta_spec_predictors(pred_dir, cloud_src_dir=mr_root)
    n_pred = len(list(pred_dir.glob("*.json")))
    logger.info("[phase=extract] predictors_metric_race has %d predictor JSONs", n_pred)


def _stub_extract(args, mr_root, rows_subset, cols_subset) -> None:  # noqa: C901 — flat smoke helper
    """CPU smoke of the extract pre-GPU pipeline (GPU-bound-phase carve-out
    item 1): build the clouds.npz + one tiny synthetic outdist JSON on a 2-layer
    CPU stub model + the REAL tokenizer + REAL demo texts, so the cloud
    construction + zoo metric build are exercised end-to-end without a GPU.
    """
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.behavior_testbed_545 import predictors_zoo as zoo

    # 2-layer CPU stub: random projection per "layer" so reps are non-constant.
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    layers = tuple(int(x) for x in args.layers.split(",")) if args.layers else (21,)
    flavors = tuple(args.flavors.split(",")) if args.flavors else ("demos",)
    zoo.GEOMETRY_LAYERS = layers
    zoo.FLAVORS = flavors
    from explore_persona_space.experiments.behavior_testbed_545 import predictors as p

    p.GEOMETRY_LAYERS = layers
    p.FLAVORS = flavors

    rng = np.random.default_rng(545)
    D = 64
    proj = {layer: rng.standard_normal((tok.vocab_size, D)).astype(np.float32) for layer in layers}

    class _StubModel:
        """Mimics output_hidden_states: per token, a deterministic D-dim rep =
        mean of the projected token embeddings up to that position (so
        last_token != mean_response, and the cloud is non-constant)."""

        def __call__(self, input_ids=None, output_hidden_states=False, **_kw):
            ids = input_ids[0].tolist()
            T = len(ids)
            hs = [None]  # hidden_states[0] is embeddings; index by layer below
            max_layer = max(layers)
            hs = [torch.zeros((1, T, D)) for _ in range(max_layer + 1)]
            for layer in layers:
                emb = proj[layer][ids]  # (T, D)
                cum = np.cumsum(emb, axis=0) / (np.arange(1, T + 1)[:, None])
                hs[layer] = torch.tensor(cum[None, :, :], dtype=torch.float32)

            class _Out:
                pass

            o = _Out()
            o.hidden_states = hs
            return o

        def eval(self):
            return self

    model = _StubModel()
    clouds: dict[str, np.ndarray] = {}
    rows = rows_subset or ["bad_medical"]
    cols = cols_subset or ["capability"]
    from explore_persona_space.experiments.behavior_testbed_545.columns import COLUMNS

    # row demos clouds
    for row_id in rows:
        texts = zoo._row_cloud_texts(row_id)
        for flavor, demo_texts in texts.items():
            reps = p._mean_hidden_states(model, tok, demo_texts, "cpu", retain_per_sample_reps=True)
            for layer in layers:
                for pt in p.EXTRACTION_POINTS:
                    t = reps.get(layer, {}).get(pt)
                    if t is not None:
                        clouds[f"row|{row_id}|{flavor}|{layer}|{pt}"] = t.numpy()
    # column probe clouds
    for col_id in cols:
        if col_id not in COLUMNS:
            continue
        probes = zoo._column_probe_texts(col_id, cap=args.n_probes)
        if not probes:
            continue
        reps = p._mean_hidden_states(model, tok, probes, "cpu", retain_per_sample_reps=True)
        for layer in layers:
            for pt in p.EXTRACTION_POINTS:
                t = reps.get(layer, {}).get(pt)
                if t is not None:
                    clouds[f"col|{col_id}|probe|{layer}|{pt}"] = t.numpy()
    np.savez_compressed(mr_root / "clouds.npz", **clouds)
    # one synthetic outdist JSON so the zoo's outdist branch is exercised.
    outdist_dir = mr_root / "outdist"
    outdist_dir.mkdir(parents=True, exist_ok=True)
    for row_id in rows:
        for col_id in cols:
            (outdist_dir / f"{row_id}__{col_id}__{flavors[0]}.json").write_text(
                json.dumps(
                    {
                        "row": row_id,
                        "col": col_id,
                        "flavor": flavors[0],
                        "rb": {
                            "js_rb_bits": 0.3,
                            "kl_ab_nats": 0.5,
                            "kl_ba_nats": 0.6,
                            "sym_kl_nats": 0.55,
                        },
                        "stub": True,
                    },
                    indent=1,
                )
            )
    logger.info(
        "[phase=extract] STUB clouds.npz (%d arrays) + outdist written to %s", len(clouds), mr_root
    )


# ---------------------------------------------------------------------------
# Phase 2 — score (CPU, OFF-POD on the VM)
# ---------------------------------------------------------------------------


def phase_score(args) -> None:
    from explore_persona_space.experiments.behavior_testbed_545 import scoring

    paths = scoring.score_metric_race(n_boot=args.n_boot, n_perm=args.n_perm)
    logger.info("[phase=score] wrote: %s", {k: str(v) for k, v in paths.items()})


# ---------------------------------------------------------------------------
# Phase 3 — plot (CPU, OFF-POD on the VM)
# ---------------------------------------------------------------------------


def phase_plot(args) -> None:
    plot_script = PROJECT_ROOT / "scripts" / "issue545_plot_metric_race.py"
    cmd = ["uv", "run", "python", str(plot_script)]
    if args.out_root:
        cmd += ["--out-root", str(args.out_root)]
    logger.info("[phase=plot] %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env={**os.environ})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="issue #545 metric-race dispatcher")
    ap.add_argument("--phase", default="prefetch,extract,score,plot")
    ap.add_argument("--rows", default="", help="comma-sep row subset (smoke = sweep w/ 1 cell)")
    ap.add_argument("--cols", default="", help="comma-sep column subset")
    ap.add_argument("--layers", default="", help="comma-sep GEOMETRY_LAYERS override (descope)")
    ap.add_argument("--flavors", default="", help="comma-sep FLAVORS override (descope)")
    ap.add_argument("--n-probes", type=int, default=50)
    ap.add_argument("--r-samples", type=int, default=8)
    ap.add_argument("--nl-cloud-samples", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--stub-gpu", action="store_true", help="CPU stub model (extract smoke)")
    ap.add_argument("--out-root", default="", help="EPM_OUTPUT_ROOT override (smoke isolation)")
    args = ap.parse_args()

    if args.out_root:
        os.environ["EPM_OUTPUT_ROOT"] = args.out_root

    phases = [x.strip() for x in args.phase.split(",") if x.strip()]
    logger.info("[phase=dispatch-start] phases=%s", phases)
    if "prefetch" in phases:
        prefetch_pinned_inputs()
    if "extract" in phases:
        phase_extract(args)
    if "score" in phases:
        phase_score(args)
    if "plot" in phases:
        phase_plot(args)
    logger.info("[phase=done] metric-race phases complete: %s", phases)


if __name__ == "__main__":
    main()
