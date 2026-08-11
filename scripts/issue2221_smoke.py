"""Issue #2221 P0 — end-to-end smoke: the PRODUCTION CLIs at tiny N (smoke IS sweep).

Thin orchestrator: every step shells the production entrypoint with
smoke-sized args (subset families / models / items / steps) — no substituted
implementations, no downgraded gates beyond the documented smoke dials the
scripts themselves expose (``--max-items`` / ``--max-prompts`` / ``--max-rows``
/ ``--max-steps``). All outputs land under ONE scratch root (default
``data/issue_2221/smoke``) so committed ``eval_results/`` is never touched.

Step order (P1 -> P8): corpus prompts -> found pool -> panel prompts ->
rollouts (1 family x 1 model, GPU) -> band pilot+band (judge API) -> mix ->
finetune sweep (1 cell + frac checkpoints, GPU) -> capture
(surfaces/parity/last/gen/resp, GPU) -> trait eval (gen/pilot/judge/
tf_margin/train_propensity/aggregate, GPU+API) -> monitors verify_keys +
arms + correlations (CPU; smoke-sized --n-bootstrap/--n-null, production
defaults unchanged — round-2 review note: the statistical battery must not
be exercised by unit tests alone). The correlations step reads the
committed #778 trait scores (``eval_results/issue_778`` — on a partial-clone
pod add the cone: ``git sparse-checkout add eval_results/issue_778``).

Data-dependent caveat: band coverage at tiny N can leave a version empty;
the mix step keeps such a version as an EMPTY cell (equalize floors over the
NON-EMPTY versions only) and ``pick_smoke_cell`` trains on any non-empty
cell. Only an ALL-empty family fails LOUD there, naming the remedy (raise
``--max-items``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_persona_space.experiments.issue_2221 import constants as C  # noqa: E402

logger = logging.getLogger("issue2221.smoke")
SCRIPTS = Path(__file__).resolve().parent

SMOKE_FAMILY = "mistake_medical"
SMOKE_CHAT_FAMILY = "evil"


def _run(name: str, argv: list[str]) -> None:
    """Run one production CLI; fail loud with the step name on rc != 0."""
    cmd = [sys.executable, *argv]
    t0 = time.time()
    logger.info("[smoke:%s] %s", name, " ".join(argv))
    proc = subprocess.run(cmd, env={**os.environ}, check=False)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"smoke step {name!r} failed rc={proc.returncode} after {dt:.0f}s")
    logger.info("[smoke:%s] rc=0 elapsed=%.0fs", name, dt)


def _assert_artifact(name: str, path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"smoke step {name!r} exited 0 but artifact missing: {path}")
    logger.info("[smoke:%s] artifact OK: %s", name, path)


def pick_smoke_cell(corpus_root: Path) -> str:
    """Pick the smoke training cell from the realized mix report (fail loud)."""
    report = json.loads((corpus_root / "mix_report.json").read_text())
    candidates = [(v, report.get(f"{SMOKE_FAMILY}/{v}", {}).get("n_rows", 0)) for v in C.VERSIONS]
    ok = [v for v, n in candidates if n >= 1]
    if not ok:
        raise RuntimeError(
            f"no non-empty {SMOKE_FAMILY} mix cell at smoke scale ({candidates}); "
            "band coverage too thin — re-run with a larger --max-items"
        )
    version = "misaligned_2" if "misaligned_2" in ok else ok[-1]
    return f"{SMOKE_FAMILY}_{version}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root", default="data/issue_2221/smoke", help="scratch root (never eval_results/)"
    )
    ap.add_argument("--external-root", default="external/persona_vectors")
    ap.add_argument("--stage-dir", default="data/issue_2221/hf_dl")
    ap.add_argument("--panel-model", default=C.PANEL_MODELS[0])
    ap.add_argument("--max-prompts", type=int, default=6)
    ap.add_argument("--max-items", type=int, default=24)
    ap.add_argument("--pilot-draws", type=int, default=30)
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--found-cap", type=int, default=60)
    ap.add_argument("--found-stream-cap", type=int, default=5000)
    ap.add_argument("--gpu-mem-util", type=float, default=0.5)
    ap.add_argument("--boot-draws", type=int, default=50, help="smoke-sized --n-bootstrap")
    ap.add_argument("--null-draws", type=int, default=25, help="smoke-sized --n-null")
    ap.add_argument("--only", nargs="*", default=None, help="run only these steps")
    ap.add_argument("--skip", nargs="*", default=[], help="skip these steps")
    ap.add_argument(
        "--skip-gpu", action="store_true", help="CPU/API-only subset (no rollouts/train/capture)"
    )
    ap.add_argument("--list-steps", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    corpus = root / "corpus"
    dataset = root / "dataset"
    ckpt = root / "ckpt"
    p5 = root / "p5"
    p6 = root / "p6"
    eval_root = root / "eval_results"

    stage_corpus = str(SCRIPTS / "issue2221_stage_corpus.py")
    band = str(SCRIPTS / "issue2221_band.py")
    build_mix = str(SCRIPTS / "issue2221_build_mix.py")
    sweep = str(SCRIPTS / "issue2221_finetune_sweep.py")
    capture = str(SCRIPTS / "issue2221_capture.py")
    trait_eval = str(SCRIPTS / "issue2221_trait_eval.py")
    monitors = str(SCRIPTS / "issue2221_monitors.py")

    gpu_steps = {"corpus_rollouts", "sweep", "capture", "trait_eval"}
    step_names = [
        "corpus_prompts",
        "corpus_found",
        "corpus_panel",
        "corpus_rollouts",
        "band_pilot",
        "band",
        "mix",
        "sweep",
        "capture",
        "trait_eval",
        "monitors_verify",
        "monitors_arms",
        "monitors_correlations",
    ]
    if args.list_steps:
        print(json.dumps(step_names))
        raise SystemExit(0)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] OK")
        raise SystemExit(0)

    def wanted(name: str) -> bool:
        if args.only is not None and name not in args.only:
            return False
        if name in args.skip:
            return False
        if args.skip_gpu and (
            name in gpu_steps
            or name
            in {"band_pilot", "band", "mix", "trait_eval", "monitors_arms", "monitors_correlations"}
        ):
            # band/mix depend on GPU rollouts for the EM family; drop them too.
            return False
        return True

    smoke_cell: str | None = None
    done: list[str] = []
    for name in step_names:
        if not wanted(name):
            logger.info("[smoke] SKIP %s", name)
            continue
        if name == "corpus_prompts":
            _run(
                name,
                [
                    stage_corpus,
                    "--phase",
                    "prompts",
                    "--out-root",
                    str(corpus),
                    "--external-root",
                    args.external_root,
                ],
            )
            _assert_artifact(name, corpus / "prompts" / f"{SMOKE_FAMILY}.jsonl")
        elif name == "corpus_found":
            _run(
                name,
                [
                    stage_corpus,
                    "--phase",
                    "found",
                    "--out-root",
                    str(corpus),
                    "--found-cap",
                    str(args.found_cap),
                    "--found-stream-cap",
                    str(args.found_stream_cap),
                ],
            )
            _assert_artifact(name, corpus / "found" / "found_pool.jsonl")
        elif name == "corpus_panel":
            _run(name, [stage_corpus, "--phase", "panel_prompts", "--out-root", str(corpus)])
            _assert_artifact(name, corpus / "panel_prompts.jsonl")
        elif name == "corpus_rollouts":
            _run(
                name,
                [
                    stage_corpus,
                    "--phase",
                    "rollouts",
                    "--out-root",
                    str(corpus),
                    "--external-root",
                    args.external_root,
                    "--families",
                    SMOKE_FAMILY,
                    "--models",
                    args.panel_model,
                    "--max-prompts",
                    str(args.max_prompts),
                    "--n-rollouts",
                    "2",
                    "--gpu-mem-util",
                    str(args.gpu_mem_util),
                ],
            )
            _assert_artifact(name, corpus / "rollouts" / SMOKE_FAMILY)
        elif name == "band_pilot":
            _run(
                name,
                [
                    band,
                    "--phase",
                    "pilot",
                    "--out-root",
                    str(corpus),
                    "--external-root",
                    args.external_root,
                    "--families",
                    SMOKE_FAMILY,
                    SMOKE_CHAT_FAMILY,
                    "--pilot-draws",
                    str(args.pilot_draws),
                    "--n-draws",
                    "2",
                ],
            )
        elif name == "band":
            _run(
                name,
                [
                    band,
                    "--phase",
                    "band",
                    "--out-root",
                    str(corpus),
                    "--external-root",
                    args.external_root,
                    "--families",
                    SMOKE_FAMILY,
                    SMOKE_CHAT_FAMILY,
                    "--max-items",
                    str(args.max_items),
                    "--n-draws",
                    "2",
                ],
            )
            _assert_artifact(name, corpus / "band" / f"{SMOKE_FAMILY}.json")
        elif name == "mix":
            _run(
                name,
                [
                    build_mix,
                    "--out-root",
                    str(corpus),
                    "--dataset-root",
                    str(dataset),
                    "--families",
                    SMOKE_FAMILY,
                    "--max-rows",
                    str(args.max_rows),
                    "--no-upload",
                ],
            )
            _assert_artifact(name, corpus / "mix_report.json")
            smoke_cell = pick_smoke_cell(corpus)
            logger.info("[smoke] training cell: %s", smoke_cell)
        elif name == "sweep":
            if smoke_cell is None:
                smoke_cell = pick_smoke_cell(corpus)
            # Canonical split (v4 blocker C1): rsplit("_", 1) mangles the
            # misaligned_{1,2} suffixes into pseudo-family/version pairs.
            fam, ver = C.family_of(smoke_cell), C.version_of(smoke_cell)
            _run(
                name,
                [
                    sweep,
                    "--dataset-root",
                    str(dataset),
                    "--ckpt-root",
                    str(ckpt),
                    "--cells",
                    f"{fam}/{ver}",
                    "--max-steps",
                    str(args.max_steps),
                    "--save-fracs",
                    ",".join(str(f) for f in C.CHECKPOINT_FRACS),
                    "--n-gpus",
                    "1",
                    "--no-upload",
                ],
            )
            _assert_artifact(name, ckpt / smoke_cell / "adapter_config.json")
            _assert_artifact(name, ckpt / smoke_cell / "checkpoint_frac10" / "adapter_config.json")
        elif name == "capture":
            if smoke_cell is None:
                smoke_cell = pick_smoke_cell(corpus)
            for phase in ("surfaces", "parity", "last", "gen", "resp"):
                _run(
                    f"{name}:{phase}",
                    [
                        capture,
                        "--phase",
                        phase,
                        "--out-root",
                        str(p5),
                        "--corpus-root",
                        str(corpus),
                        "--ckpt-root",
                        str(ckpt),
                        "--stage-dir",
                        args.stage_dir,
                        "--external-root",
                        args.external_root,
                        "--cells",
                        smoke_cell,
                        "--n-questions",
                        "2",
                        "--skip-synth",
                        "--gpu-mem-util",
                        str(args.gpu_mem_util),
                    ],
                )
            _assert_artifact(name, p5 / "capture" / "base.pt")
            _assert_artifact(name, p5 / "capture" / f"{smoke_cell}.pt")
            _assert_artifact(name, p5 / "parity_probe.json")
        elif name == "trait_eval":
            if smoke_cell is None:
                smoke_cell = pick_smoke_cell(corpus)
            # pilot sits between gen and judge — the rule-26 gate on REAL P6
            # rollouts; phase_judge REFUSES without its passed report.
            # --max-prompts 8 covers every trait's 2 paper questions + 2
            # LMSYS rows (surface order: paper rows per trait, then lmsys) —
            # the correlations step needs a finite PAPER-panel y per trait.
            for phase in ("gen", "pilot", "judge", "tf_margin", "train_propensity", "aggregate"):
                _run(
                    f"{name}:{phase}",
                    [
                        trait_eval,
                        "--phase",
                        phase,
                        "--out-root",
                        str(p6),
                        "--p5-root",
                        str(p5),
                        "--corpus-root",
                        str(corpus),
                        "--ckpt-root",
                        str(ckpt),
                        "--eval-results-root",
                        str(eval_root),
                        "--external-root",
                        args.external_root,
                        "--cells",
                        smoke_cell,
                        "--dataset-root",
                        str(dataset),
                        "--train-prop-prompts",
                        "2",
                        "--max-prompts",
                        "8",
                        "--n-rollouts",
                        "2",
                        "--judge-draws",
                        "2",
                        "--pilot-draws",
                        str(args.pilot_draws),
                        "--gpu-mem-util",
                        str(args.gpu_mem_util),
                    ],
                )
            _assert_artifact(name, eval_root / "trait_scores.json")
        elif name == "monitors_verify":
            _run(
                name,
                [
                    monitors,
                    "--phase",
                    "verify_keys",
                    "--stage-dir",
                    args.stage_dir,
                    "--p5-root",
                    str(p5),
                    "--eval-results-root",
                    str(eval_root),
                ],
            )
        elif name == "monitors_arms":
            if smoke_cell is None:
                smoke_cell = pick_smoke_cell(corpus)
            _run(
                name,
                [
                    monitors,
                    "--phase",
                    "arms",
                    "--stage-dir",
                    args.stage_dir,
                    "--p5-root",
                    str(p5),
                    "--eval-results-root",
                    str(eval_root),
                    "--cells",
                    smoke_cell,
                ],
            )
            _assert_artifact(name, eval_root / "monitor_scalars" / "evil_pooled.json")
        elif name == "monitors_correlations":
            if smoke_cell is None:
                smoke_cell = pick_smoke_cell(corpus)
            _run(
                name,
                [
                    monitors,
                    "--phase",
                    "correlations",
                    "--stage-dir",
                    args.stage_dir,
                    "--p5-root",
                    str(p5),
                    "--eval-results-root",
                    str(eval_root),
                    "--cells",
                    smoke_cell,
                    "--n-bootstrap",
                    str(args.boot_draws),
                    "--n-null",
                    str(args.null_draws),
                ],
            )
            _assert_artifact(name, eval_root / "correlations.json")
            _assert_artifact(name, eval_root / "draw_matrices" / "evil_pooled.npz")
        done.append(name)

    print(json.dumps({"smoke": "PASS", "steps": done, "cell": smoke_cell, "root": str(root)}))
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
