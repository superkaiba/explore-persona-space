"""Issue #366 cascade experiment orchestrator.

Pipeline (called from ``run_366.sh`` which the pod's ``dockerArgs`` runs):

  1. Bootstrap (env, HF_HOME, .env, logging).
  2. Resolve markers A..E and verify tokenization. Write
     ``marker_token_verification.json`` early so failures still leave audit.
  3. Write ``word_pool.txt`` and verify SHA.
  4. Write ``run_manifest.json`` early (git commit, env, adapter list, etc.).
  5. For each of the 11 adapters:
     a. Build the training JSONL.
     b. Train the LoRA adapter.
     c. Run primary eval (vLLM, 2860 generations).
     d. Run seeded probes (B / B+C / B+C+D depending on N).
     e. Persist matcher hits and cell aggregates per adapter.
  6. Compute donor fidelity CSV across all 11 adapters.
  7. Compute cascade curves (T-C deltas with paired cluster bootstrap).
  8. Render 4 SVG figures.
  9. Final progress POST + a brief summary line on stdout.

Idempotency: every step that writes a result checks whether the result file
already exists and skips work if so. A failed pod that is restarted picks up
from where it left off.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Path setup (must precede any project imports) ────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # explore-persona-space repo root
SRC_DIR = PROJECT_ROOT / "src"
for p in (SCRIPT_DIR, SRC_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Project bootstrap (HF_HOME, .env, logging)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _bootstrap import bootstrap  # noqa: E402

log = bootstrap(log_name="issue366")

# Project imports
# Local-package imports (scripts/experiments/366/)
from analysis import (  # noqa: E402
    build_cascade_curves,
    compute_donor_fidelity,
    write_cell_aggregates,
    write_donor_fidelity_csv,
    write_matcher_hits,
)
from data_gen import build_dataset, enumerate_adapter_configs  # noqa: E402
from eval_366 import eval_one_adapter  # noqa: E402
from markers import bindings_to_jsonable, resolve_all_markers  # noqa: E402
from progress import post_progress  # noqa: E402
from train_366 import TrainingArgs366, train_one_adapter  # noqa: E402
from word_pool import POOL_SHA256, assert_pool_sha, write_pool_artifact  # noqa: E402

from explore_persona_space.personas import ALL_EVAL_PERSONAS  # noqa: E402
from figures import make_all_figures  # noqa: E402

# ── Output paths (relative to repo root inside the pod) ──────────────────────

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "366"
ADAPTERS_DIR = ARTIFACTS_DIR / "adapters"
TRAIN_LOGS_DIR = ARTIFACTS_DIR / "train_logs"
EVAL_DIR = ARTIFACTS_DIR / "eval"
MATCHER_HITS_DIR = ARTIFACTS_DIR / "matcher_hits"
CELL_AGG_DIR = ARTIFACTS_DIR / "cell_aggregates"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
DATA_DIR = ARTIFACTS_DIR / "data"

# Pre-create dirs so individual write paths don't need to.
for d in (
    ARTIFACTS_DIR,
    ADAPTERS_DIR,
    TRAIN_LOGS_DIR,
    EVAL_DIR,
    MATCHER_HITS_DIR,
    CELL_AGG_DIR,
    FIGURES_DIR,
    DATA_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

# Top-level result files
MARKER_VERIF_PATH = ARTIFACTS_DIR / "marker_token_verification.json"
WORD_POOL_PATH = ARTIFACTS_DIR / "word_pool.txt"
RUN_MANIFEST_PATH = ARTIFACTS_DIR / "run_manifest.json"
DONOR_FIDELITY_CSV_PATH = ARTIFACTS_DIR / "donor_fidelity.csv"
CASCADE_CURVES_PATH = ARTIFACTS_DIR / "cascade_curves.json"
SEEDED_PROBE_RESULTS_PATH = ARTIFACTS_DIR / "seeded_probe_results.json"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _env_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for mod_name in ("torch", "transformers", "trl", "peft", "vllm", "datasets"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[mod_name] = "not_installed"
    return versions


def write_run_manifest(
    bindings_json: dict, adapter_configs: list[dict], started_at_utc: str
) -> None:
    manifest = {
        "experiment_number": 366,
        "experiment_id": os.environ.get("SAGAN_EXPERIMENT_ID", "unknown"),
        "agent_run_id": os.environ.get("SAGAN_AGENT_RUN_ID", "unknown"),
        "branch": os.environ.get("SAGAN_EPS_BRANCH", "issue-366"),
        "git_commit": _git_commit(),
        "started_at_utc": started_at_utc,
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "training_recipe": {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lr": 1e-5,
            "epochs": 3,
            "warmup_ratio": 0.05,
            "batch_size": 4,
            "grad_accum": 4,
            "effective_batch_size": 16,
            "max_length": 1024,
            "gradient_clip": 1.0,
            "bf16": True,
            "grad_ckpt": True,
        },
        "eval_recipe": {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 64,
            "n": 10,
            "seed": 42,
            "personas": list(ALL_EVAL_PERSONAS.keys()),
        },
        "donor_persona": "librarian",
        "recipient_persona": "software_engineer",
        "marker_bindings": bindings_json,
        "word_pool": {"size": 500, "sha256": POOL_SHA256},
        "bootstrap": {"B": 10_000, "rng_seed": 20260513, "n_clusters": 26},
        "adapters": adapter_configs,
        "env_versions": _env_versions(),
        "pod": {
            "gpu_count": os.environ.get("SAGAN_RUN_INDEX", "0"),
            "hf_home": os.environ.get("HF_HOME"),
            "wandb_project": os.environ.get("WANDB_PROJECT"),
        },
    }
    with open(RUN_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote run manifest: %s", RUN_MANIFEST_PATH)


# ── Main orchestration ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #366 cascade experiment runner")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (assume adapters exist)",
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip eval (assume completions exist)"
    )
    parser.add_argument("--skip-figures", action="store_true", help="Skip final figure generation")
    args = parser.parse_args()

    t0 = time.time()
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log.info("=" * 70)
    log.info("Issue #366 — cross-persona chunk-binding cascade")
    log.info("=" * 70)
    log.info("Started at %s, git=%s", started_at_utc, _git_commit())

    post_progress(6, "issue 366 bootstrap done, resolving markers")

    # ── Step 1: Resolve markers ──
    from transformers import AutoTokenizer

    base_model = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bindings = resolve_all_markers(tok)
    bindings_json = bindings_to_jsonable(bindings)
    with open(MARKER_VERIF_PATH, "w") as f:
        json.dump(
            {
                "base_model": base_model,
                "eos_token_id": tok.eos_token_id,
                "bindings": bindings_json,
            },
            f,
            indent=2,
        )
    log.info("Wrote marker verification: %s", MARKER_VERIF_PATH)
    del tok  # free tokenizer; will be re-loaded per training run

    post_progress(8, "markers resolved, building word pool")

    # ── Step 2: Word pool ──
    assert_pool_sha()
    write_pool_artifact(WORD_POOL_PATH)

    # ── Step 3: Adapter configs + run manifest ──
    adapter_configs = enumerate_adapter_configs()
    write_run_manifest(bindings_json, adapter_configs, started_at_utc)

    post_progress(10, "manifest written, beginning per-adapter loop (11 adapters)")

    # ── Step 4: Per-adapter loop: build data → train → eval ──
    # Each adapter contributes ~6% of total experiment progress (66% budget for
    # this loop). We post progress after train and after eval for each.
    PER_ADAPTER_PCT = 6.0
    base_pct = 10.0
    completions_by_adapter: dict[str, dict[str, dict[str, list[str]]]] = {}
    seeded_probe_summary: dict[str, dict] = {}

    recipient_persona_prompt = ALL_EVAL_PERSONAS["software_engineer"]

    for i, cfg in enumerate(adapter_configs):
        name = cfg["name"]
        log.info("-" * 60)
        log.info(
            "Adapter %d/%d: %s (n_chain=%d, condition=%s, seed=%d, ablate=%s)",
            i + 1,
            len(adapter_configs),
            name,
            cfg["n_chain"],
            cfg["condition"],
            cfg["seed"],
            cfg["ablate"],
        )

        data_path = DATA_DIR / f"{name}.jsonl"
        adapter_outdir = ADAPTERS_DIR / name
        eval_outdir = EVAL_DIR / name

        # 4a. Data
        if not data_path.exists():
            build_dataset(
                n_chain=cfg["n_chain"],
                condition=cfg["condition"],
                seed=cfg["seed"],
                marker_bindings=bindings,
                out_path=data_path,
                ablate=cfg["ablate"],
            )
        else:
            log.info("Dataset already built: %s", data_path)

        # 4b. Train
        if not args.skip_train:
            t_args = TrainingArgs366(
                base_model=base_model,
                seed=cfg["seed"],
                recipient_persona_prompt=recipient_persona_prompt,
            )
            meta = train_one_adapter(
                data_path=data_path,
                output_dir=adapter_outdir,
                args=t_args,
                gpu_id=args.gpu,
                run_name=f"issue366_{name}",
            )
            # Mirror the per-adapter train_meta into train_logs/ for the
            # artifacts upload to find with one glob.
            with open(TRAIN_LOGS_DIR / f"{name}.json", "w") as f:
                json.dump(meta, f, indent=2)

        post_progress(
            base_pct + (i + 0.5) * PER_ADAPTER_PCT,
            f"trained {name}",
        )

        # 4c+d. Eval
        if not args.skip_eval:
            adapter_dir = adapter_outdir / "adapter"
            if not (adapter_dir / "adapter_config.json").exists():
                log.warning("Adapter dir missing %s; skipping eval", adapter_dir)
                continue
            eval_meta = eval_one_adapter(
                adapter_path=adapter_dir,
                output_dir=eval_outdir,
                marker_bindings=bindings,
                n_chain=cfg["n_chain"],
                gpu_id=args.gpu,
            )

            primary_path = Path(eval_meta["primary_completions_path"])
            with open(primary_path) as f:
                completions = json.load(f)
            completions_by_adapter[name] = completions

            # 4e. Matcher hits + cell aggregates
            write_matcher_hits(
                completions,
                bindings,
                MATCHER_HITS_DIR / f"{name}.json",
            )
            write_cell_aggregates(
                completions,
                bindings,
                CELL_AGG_DIR / f"{name}.json",
            )

            # Track seeded probes for the consolidated artifact
            probe_paths = eval_meta.get("seeded_probes", {})
            probes_for_adapter: dict[str, dict] = {}
            for kind, ppath in probe_paths.items():
                with open(ppath) as f:
                    probes_for_adapter[kind] = json.load(f)
            seeded_probe_summary[name] = probes_for_adapter

        post_progress(
            base_pct + (i + 1) * PER_ADAPTER_PCT,
            f"evaled {name}",
        )

    # ── Step 5: Donor fidelity ──
    if not args.skip_eval and completions_by_adapter:
        donor_rows = compute_donor_fidelity(completions_by_adapter, bindings)
        write_donor_fidelity_csv(donor_rows, DONOR_FIDELITY_CSV_PATH)
    post_progress(82, "donor fidelity csv written")

    # ── Step 6: Cascade curves ──
    if not args.skip_eval and completions_by_adapter:
        curves = build_cascade_curves(completions_by_adapter, bindings, adapter_configs)
        with open(CASCADE_CURVES_PATH, "w") as f:
            json.dump(curves, f, indent=2)
        log.info("Wrote cascade curves: %s", CASCADE_CURVES_PATH)

    # ── Step 7: Consolidated seeded-probe summary ──
    if not args.skip_eval and seeded_probe_summary:
        with open(SEEDED_PROBE_RESULTS_PATH, "w") as f:
            json.dump(seeded_probe_summary, f, indent=2)
        log.info("Wrote seeded probe results: %s", SEEDED_PROBE_RESULTS_PATH)

    post_progress(90, "stats artifacts written, rendering figures")

    # ── Step 8: Figures ──
    if not args.skip_figures and CASCADE_CURVES_PATH.exists():
        try:
            make_all_figures(
                cascade_curves_path=CASCADE_CURVES_PATH,
                cell_aggregates_dir=CELL_AGG_DIR,
                donor_fidelity_csv=DONOR_FIDELITY_CSV_PATH,
                figures_dir=FIGURES_DIR,
            )
        except Exception as e:
            log.warning("Figure generation failed: %s", e, exc_info=True)

    post_progress(98, "figures rendered, finalizing run manifest")

    # ── Step 9: Final manifest update with completion timestamp ──
    try:
        with open(RUN_MANIFEST_PATH) as f:
            manifest = json.load(f)
        manifest["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["total_seconds"] = round(time.time() - t0, 2)
        manifest["artifacts_summary"] = {
            "n_adapters_trained": sum(
                1
                for cfg in adapter_configs
                if (ADAPTERS_DIR / cfg["name"] / "adapter" / "adapter_config.json").exists()
            ),
            "n_adapters_evaled": len(completions_by_adapter),
            "figures": [
                str(p.relative_to(ARTIFACTS_DIR)) for p in sorted(FIGURES_DIR.glob("*.svg"))
            ],
        }
        with open(RUN_MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        log.warning("Final manifest update failed: %s", e, exc_info=True)

    log.info("Total wall time: %.1f min", (time.time() - t0) / 60)
    log.info("Done. Artifacts at: %s", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
