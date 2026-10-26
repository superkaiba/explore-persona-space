"""Issue #2674 Phase 0 — uploads, per the project upload policy.

Destinations:
  teacher data + sampled completions (rollout TEXT)
      HF data repo ``superkaiba1/explore-persona-space-data`` under
      ``issue2674_sl_epochs/raw_completions/<stage>/...``. Staged as sub-9 MB
      ``raw_completions.json`` shards and committed in ONE ``upload_folder`` commit
      (never a per-file loop — that 504-storms on the large data repo).
  adapters
      HF model repo ``superkaiba1/explore-persona-space`` under
      ``issue2674_sl_epochs/<run>/<step>``.
  metric JSONs
      the data repo under ``issue2674_sl_epochs/metrics/``, so the VM can pull them
      without touching the pod.

Stages (``--stage``), each idempotent and sentinel-checkpointed:
  ``teacher_data``  the staged teacher pools + the data-generation metrics.
  ``sampled``       the staged sampled-eval completions.
  ``adapters``      every saved LoRA adapter of every run.
  ``metrics``       every JSON under ``metrics/`` plus the per-run trajectories.
  ``all``           all four, in that order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue2674_common as C  # noqa: E402


def _raw_subtree_has_payload(raw_root: Path, stage_rel: str) -> bool:
    return any((raw_root / stage_rel).rglob("raw_completions.json"))


def upload_raw(out_root: str | Path, stage_rel: str) -> list[str]:
    """Commit one raw-completions subtree in a single folder commit.

    ``upload_raw_completions_to_data_repo`` selects every ``raw_completions.json`` under
    the tree it is handed and commits them to
    ``<EXPERIMENT>/raw_completions/<path relative to that tree>``, so handing it
    ``<out>/raw`` reproduces the policy's ``raw_completions/<stage>/`` layout exactly.
    """
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    raw_root = C.out_paths(out_root)["raw"]
    if not _raw_subtree_has_payload(raw_root, stage_rel):
        raise RuntimeError(
            f"no raw_completions.json under {raw_root / stage_rel} — the producing phase "
            f"did not stage its rollout text"
        )
    urls = upload_raw_completions_to_data_repo(
        experiment_name=C.EXPERIMENT, eval_results_dir=raw_root
    )
    if not urls:
        raise RuntimeError(f"raw-completions upload returned no URLs for {raw_root}")
    return sorted(urls)


def upload_metrics(out_root: str | Path) -> list[str]:
    """Upload every metric JSON to the data repo in ONE folder commit.

    Per-run trajectories / throughput / meta JSONs are copied under
    ``metrics/runs/<run>/`` first, so the whole metrics tree lands as a single
    ``upload_folder`` commit rather than one commit per file (a per-file loop burns
    the shared 256-commits-per-hour budget and 504-storms on the large data repo).
    """
    import shutil

    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload

    paths = C.out_paths(out_root)
    metrics_dir = paths["metrics"]
    if not sorted(metrics_dir.glob("*.json")):
        raise RuntimeError(f"no metric JSONs under {metrics_dir}")

    staged: list[str] = []
    for run_dir in sorted(p for p in paths["runs"].glob("*") if p.is_dir()):
        dest = metrics_dir / "runs" / run_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for src in sorted(run_dir.glob("*.json")):
            shutil.copy2(src, dest / src.name)
            staged.append(f"runs/{run_dir.name}/{src.name}")

    url = _upload(
        local_path=metrics_dir,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=f"{C.EXPERIMENT}/metrics",
        upload_as_file=False,
        raise_on_error=True,
    )
    if not url:
        raise RuntimeError(f"metrics upload failed for {metrics_dir}")
    files = sorted(p.relative_to(metrics_dir).as_posix() for p in metrics_dir.rglob("*.json"))
    print(f"[upload] metrics: {len(files)} JSON file(s) -> {url}", flush=True)
    return files


def upload_adapters(out_root: str | Path) -> list[str]:
    """Upload every saved adapter under ``issue2674_sl_epochs/<run>/<step>``."""
    from explore_persona_space.orchestrate.hub import upload_model

    paths = C.out_paths(out_root)
    urls: list[str] = []
    found = False
    for run_dir in sorted(p for p in paths["runs"].glob("*") if p.is_dir()):
        adapters = run_dir / "adapters"
        if not adapters.is_dir():
            continue
        for step_dir in sorted(p for p in adapters.glob("step_*") if p.is_dir()):
            if not (step_dir / "adapter_config.json").is_file():
                raise RuntimeError(f"{step_dir} has no adapter_config.json — incomplete save")
            found = True
            urls.append(
                upload_model(
                    str(step_dir),
                    repo_id=C.HF_MODEL_REPO,
                    path_in_repo=f"{C.EXPERIMENT}/{run_dir.name}/{step_dir.name}",
                    ignore_patterns=["checkpoint-*"],
                )
            )
    if not found:
        raise RuntimeError(f"no step_* adapters under {paths['runs']}")
    return urls


STAGES = ("teacher_data", "sampled", "adapters", "metrics")


def run_stage(out_root: str | Path, stage: str) -> dict:
    if stage == "teacher_data":
        return {"raw_completions": upload_raw(out_root, "teacher_data")}
    if stage == "sampled":
        return {"raw_completions": upload_raw(out_root, "sampled_eval")}
    if stage == "adapters":
        return {"adapters": upload_adapters(out_root)}
    if stage == "metrics":
        return {"metrics": upload_metrics(out_root)}
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        return _import_check()
    if args.out is None:
        raise SystemExit("--out is required")

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for stage in stages:
        sentinel = f"upload_{stage}"
        if C.sentinel_done(args.out, sentinel) and not args.force:
            print(f"[upload] stage={stage} already done — skip", flush=True)
            continue
        result = run_stage(args.out, stage)
        counts = {k: len(v) for k, v in result.items()}
        C.write_sentinel(args.out, sentinel, {"counts": counts})
        print(f"[upload] stage={stage} OK {counts}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=None, help="Run out-root.")
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    ap.add_argument("--force", action="store_true", help="Re-run a stage already sentinelled.")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="Static argparse-attribute + helper-call-arity check; runs nothing.",
    )
    return ap


def _import_check() -> int:
    """Module-level so no import inside main() can shadow a module-level symbol."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__, str(Path(__file__).with_name("issue2674_common.py")))
    print("[import-check] OK", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
