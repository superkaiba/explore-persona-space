"""Issue #2221 P0 — end-to-end smoke: the PRODUCTION CLIs at tiny N (smoke IS sweep).

Thin orchestrator: every step shells the production entrypoint with
smoke-sized args (subset families / models / items / steps) — no substituted
implementations, no downgraded gates beyond the documented smoke dials the
scripts themselves expose (``--max-items`` / ``--max-prompts`` / ``--max-rows``
/ ``--max-steps``). All outputs land under ONE scratch root (default
``data/issue_2221/smoke``) so committed ``eval_results/`` is never touched.

Step order (P1 -> P8, specialized_corpus_remine v10): found pool ->
FOUND-INVERT toxic pool -> AITA disjoint-split prompts (sycophancy /
mistake_opinions) -> ChatDoctor prompts (mistake_medical) -> panel prompts ->
rollouts (3 remine families x 1 model, GPU) -> band pilot+band (judge API;
``--em-like-families sycophancy --evil-pool found_toxic`` routing) -> mix
(``--drop-floor 1`` — gotchas.md smoke GATE-CALIBRATION: the production
16-row floor annihilates every family at smoke N;
``would_drop_at_production_floor`` still recorded) -> finetune sweep (1 cell
+ frac checkpoints, GPU) -> capture (surfaces/parity/last/gen/resp, GPU) ->
trait eval (gen/gen_regen/pilot/judge/tf_margin/train_propensity/aggregate,
GPU+API) -> PRODUCTION-CAP REGEN probe (a deliberately low gen cap trips the
>2% trigger; regen runs at the PRODUCTION 4096 cap on the PRODUCTION 8192
engine window — the v10 Must-Fix fix-engaged demonstration: n_regen >= 1 AND
regen_overlong_skipped == 0) -> monitors verify_keys + arms + correlations
(CPU; smoke-sized --n-bootstrap/--n-null). The correlations step reads the
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
# The 3 remine prompt-staged families (AITA split A/B + ChatDoctor) — the
# rollouts/band/mix legs cover one tiny cell per SOURCE CLASS (per-arm-class
# smoke duty): AITA-sycophancy, AITA-mistake_opinions, ChatDoctor-medical,
# plus the FOUND-invert evil chat family.
SMOKE_ROLLOUT_FAMILIES = ("mistake_medical", "sycophancy", "mistake_opinions")
SMOKE_BAND_FAMILIES = (*SMOKE_ROLLOUT_FAMILIES, SMOKE_CHAT_FAMILY)
SMOKE_EM_LIKE = ("sycophancy",)


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
    """Pick the smoke training cell from the realized mix yield report (fail loud).

    Also sanity-checks the v10 two-tier floor wiring: the family MUST carry a
    ``_family_floor`` record with ``would_drop_at_production_floor`` (always
    recorded even at the smoke's ``--drop-floor 1`` dial).
    """
    report = json.loads((corpus_root / "mix_yield.json").read_text())
    floor = report.get("_family_floor", {}).get(SMOKE_FAMILY)
    if floor is None or "would_drop_at_production_floor" not in floor:
        raise RuntimeError(
            f"mix_yield.json carries no _family_floor record for {SMOKE_FAMILY} — "
            "the v10 two-tier floor wiring is broken"
        )
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
    ap.add_argument("--aita-stream-cap", type=int, default=2000)
    ap.add_argument("--chatdoctor-stream-cap", type=int, default=2000)
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
    p6_regen = root / "p6_regen"
    eval_root = root / "eval_results"

    stage_corpus = str(SCRIPTS / "issue2221_stage_corpus.py")
    band = str(SCRIPTS / "issue2221_band.py")
    build_mix = str(SCRIPTS / "issue2221_build_mix.py")
    sweep = str(SCRIPTS / "issue2221_finetune_sweep.py")
    capture = str(SCRIPTS / "issue2221_capture.py")
    trait_eval = str(SCRIPTS / "issue2221_trait_eval.py")
    monitors = str(SCRIPTS / "issue2221_monitors.py")

    gpu_steps = {
        "corpus_rollouts",
        "corpus_rollouts_regen",
        "sweep",
        "capture",
        "trait_eval",
        "trait_eval_regen_probe",
    }
    step_names = [
        "corpus_found",
        "corpus_found_toxic",
        "corpus_aita",
        "corpus_chatdoctor",
        "corpus_panel",
        "corpus_rollouts",
        "corpus_rollouts_regen",
        "band_pilot",
        "band",
        "mix",
        "sweep",
        "capture",
        "trait_eval",
        "trait_eval_regen_probe",
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
        if name == "corpus_found":
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
        elif name == "corpus_found_toxic":
            # P1a remine: FOUND-INVERT (keep moderation-flagged assistant
            # turns) — the evil family's pool source under --evil-pool.
            # LMSYS-ONLY (plan v11): the WildChat arm is unwired (non-gated
            # release is toxicity-stripped, zero toxic=true rows), so this
            # smoke step streams exactly the one production evil source.
            _run(
                name,
                [
                    stage_corpus,
                    "--phase",
                    "found_toxic",
                    "--out-root",
                    str(corpus),
                    "--found-cap",
                    str(args.found_cap),
                    "--found-stream-cap",
                    str(args.found_stream_cap),
                ],
            )
            _assert_artifact(name, corpus / "found_toxic" / "found_toxic_pool.jsonl")
        elif name == "corpus_aita":
            # P1b remine: AITA dilemmas -> post-id-DISJOINT sycophancy /
            # mistake_opinions prompt files (same schema phase_rollouts reads).
            _run(
                name,
                [
                    stage_corpus,
                    "--phase",
                    "aita",
                    "--out-root",
                    str(corpus),
                    "--prompts-cap",
                    str(args.max_prompts),
                    "--aita-stream-cap",
                    str(args.aita_stream_cap),
                ],
            )
            _assert_artifact(name, corpus / "prompts" / "sycophancy.jsonl")
            _assert_artifact(name, corpus / "prompts" / "mistake_opinions.jsonl")
            _assert_artifact(name, corpus / "aita" / "aita_split_report.json")
        elif name == "corpus_chatdoctor":
            # P1b remine: ChatDoctor patient questions -> mistake_medical prompts.
            _run(
                name,
                [
                    stage_corpus,
                    "--phase",
                    "chatdoctor",
                    "--out-root",
                    str(corpus),
                    "--prompts-cap",
                    str(args.max_prompts),
                    "--chatdoctor-stream-cap",
                    str(args.chatdoctor_stream_cap),
                ],
            )
            _assert_artifact(name, corpus / "prompts" / "mistake_medical.jsonl")
            _assert_artifact(name, corpus / "chatdoctor" / "chatdoctor_stage_report.json")
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
                    *SMOKE_ROLLOUT_FAMILIES,
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
            for fam in SMOKE_ROLLOUT_FAMILIES:
                _assert_artifact(name, corpus / "rollouts" / fam)
        elif name == "corpus_rollouts_regen":
            # The cap-hit trigger's ACTION arm (v14): no-trigger cells skip
            # fast; a triggered smoke cell exercises the real splice path.
            _run(
                name,
                [
                    stage_corpus,
                    "--phase",
                    "rollouts_regen",
                    "--out-root",
                    str(corpus),
                    "--families",
                    *SMOKE_ROLLOUT_FAMILIES,
                    "--models",
                    args.panel_model,
                    "--gpu-mem-util",
                    str(args.gpu_mem_util),
                ],
            )
            _assert_artifact(name, corpus / "rollouts" / "cap_hit_report.json")
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
                    *SMOKE_BAND_FAMILIES,
                    "--em-like-families",
                    *SMOKE_EM_LIKE,
                    "--evil-pool",
                    "found_toxic",
                    "--pilot-draws",
                    str(args.pilot_draws),
                    "--n-draws",
                    "2",
                    # Smoke slice cannot resolve the rule-26(b) parse-fail
                    # threshold (needs >=51 effective draws/arm; the smoke has
                    # ~24) — accept a sub-resolution pilot HERE ONLY. The
                    # production P2 wave runs without this flag (full gate).
                    "--allow-subresolution-pilot",
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
                    *SMOKE_BAND_FAMILIES,
                    "--em-like-families",
                    *SMOKE_EM_LIKE,
                    "--evil-pool",
                    "found_toxic",
                    "--max-items",
                    str(args.max_items),
                    "--n-draws",
                    "2",
                ],
            )
            _assert_artifact(name, corpus / "band" / f"{SMOKE_FAMILY}.json")
            _assert_artifact(name, corpus / "band" / f"{SMOKE_CHAT_FAMILY}.json")
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
                    *SMOKE_BAND_FAMILIES,
                    "--em-like-families",
                    *SMOKE_EM_LIKE,
                    "--evil-pool",
                    "found_toxic",
                    # gotchas.md smoke GATE-CALIBRATION (#1345): the production
                    # 16-row DROP floor annihilates every family at smoke N;
                    # would_drop_at_production_floor is still recorded.
                    "--drop-floor",
                    "1",
                    "--max-rows",
                    str(args.max_rows),
                    "--no-upload",
                ],
            )
            _assert_artifact(name, corpus / "mix_yield.json")
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
            for phase in (
                "gen",
                "gen_regen",
                "pilot",
                "judge",
                "tf_margin",
                "train_propensity",
                "aggregate",
            ):
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
                        # Same sub-resolution acceptance as band_pilot: the
                        # smoke's 8-item arms cannot resolve the rule-26(b)
                        # parse-fail threshold. Production P6 pilots never
                        # carry this flag.
                        "--allow-subresolution-pilot",
                    ],
                )
            _assert_artifact(name, eval_root / "trait_scores.json")
        elif name == "trait_eval_regen_probe":
            if smoke_cell is None:
                smoke_cell = pick_smoke_cell(corpus)
            # PRODUCTION-CAP REGEN assert (v10 Must-Fix fix-engaged signal):
            # a deliberately LOW gen cap (16) trips the >2% trigger on real
            # rollouts; gen_regen then runs at the PRODUCTION values — cap
            # 4096 on the DEDICATED 8192-window engine. Under the r-parent's
            # inert shape (budget = 4096 - 4096 = 0) every row would be
            # regen_overlong_skipped; the asserts below fail exactly there.
            common = [
                "--out-root",
                str(p6_regen),
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
                "--max-prompts",
                "4",
                "--n-rollouts",
                "2",
                "--max-new-tokens",
                "16",
                "--gpu-mem-util",
                str(args.gpu_mem_util),
            ]
            _run(f"{name}:gen", [trait_eval, "--phase", "gen", *common])
            _run(
                f"{name}:gen_regen",
                [trait_eval, "--phase", "gen_regen", "--regen-max-new-tokens", "4096", *common],
            )
            report_path = p6_regen / "eval_rollouts" / "regen_report.json"
            _assert_artifact(name, report_path)
            report = json.loads(report_path.read_text())
            triggered = [t for t, d in report.items() if d.get("triggered")]
            n_regen = sum(d.get("regen_n_rows", 0) for d in report.values())
            n_overlong = sum(d.get("regen_overlong_skipped", 0) for d in report.values())
            caps = {
                (d.get("regen_max_new_tokens"), d.get("regen_max_model_len"))
                for d in report.values()
                if d.get("triggered")
            }
            if not triggered or n_regen < 1:
                raise RuntimeError(
                    f"regen probe never engaged: triggered={triggered} n_regen={n_regen} "
                    f"(report: {report})"
                )
            if n_overlong != 0:
                raise RuntimeError(
                    f"regen probe skipped {n_overlong} rows as overlong at the PRODUCTION "
                    f"budget — the v10 inert-regen shape is back (report: {report})"
                )
            if caps != {(4096, 8192)}:
                raise RuntimeError(f"regen probe ran at non-production caps: {caps}")
            logger.info(
                "[smoke:%s] regen engaged: tags=%s n_regen=%d overlong=0 caps=%s",
                name,
                triggered,
                n_regen,
                caps,
            )
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
