#!/usr/bin/env python3
"""Sequential launcher for issue #260 — runs all 15 invocations on 1x H100.

Sequence:
  Leg 1 (training + eval; 9 conditions):
    mt_n1, mt_n4, mt_n16          -- sub-experiment (a)  max_length=8192
    lc_short, lc_medium, lc_long  -- sub-experiment (b)  max_length=1536
    sl_short, sl_medium, sl_long  -- sub-experiment (c)  max_length=1280

  Leg 2 (eval-only re-runs on the merged adapters from Leg 1):
    Sub-exp (c) Leg 2 -- 3 conditions: each model evaluated under its
                         TRAIN-MATCHED system prompt.
    Sub-exp (a) Leg 2 -- 3 conditions: each model evaluated with
                         (K-1) (warmup_q, "I see, let me think.") turns
                         prepended to every eval prompt.

After EACH Leg-1 run completes, the launcher MOVES the result dir from
    eval_results/leakage_experiment/marker_librarian_asst_excluded_medium_seed42/
    -> eval_results/issue260/<COND>/
BEFORE the next condition starts. (Sequential is required; the recipe parent's
`make_run_name(args)` keys only on (trait, source, neg_set, prompt_length, seed)
and would otherwise overwrite the prior condition's outputs.)

Leg 2 reruns mv to <COND>_leg2/ (e.g. sl_short_leg2/, mt_n4_leg2/) so they sit
alongside their Leg-1 originals without collision.

State + progress:
  eval_results/issue260/launcher.log         -- stdout/stderr from this script
  eval_results/issue260/launcher_state.json  -- per-condition status / wall-time
  eval_results/issue260/<COND>_leg1.log      -- per-condition Leg-1 subprocess log

Usage::

    PYTHONHASHSEED=42 nohup uv run python scripts/launch_issue260.py \\
        --pod epm-issue-260 \\
        > eval_results/issue260/launcher.log 2>&1 &

Optional flags::

    --skip-leg1            Reuse already-mv'd Leg-1 dirs and only run Leg 2.
    --post-progress / --no-post-progress   Toggle GitHub epm:progress markers
                                            (default: on if `gh` available).
    --conditions <names>   Only run a subset (debug / resume).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ISSUE_DATA = PROJECT_ROOT / "data" / "leakage_experiment_issue260"
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue260"
LEAKAGE_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "leakage_experiment"
# Base run-name (no suffix); the parent recipe wrote to <BASE_RUN_NAME>/.
# With --run-name-suffix=<COND>, the runner now writes to <BASE_RUN_NAME>_<COND>/.
BASE_RUN_NAME = "marker_librarian_asst_excluded_medium_seed42"
PARENT_RESULT_DIR_LEGACY = LEAKAGE_RESULTS_ROOT / BASE_RUN_NAME
# Stable parent-#271 anchor copy: preserved BEFORE Leg 1 starts so the (c)
# panel still has its dotted reference line after Leg-1 runs (which write into
# their own per-condition <BASE_RUN_NAME>_<COND>/ dirs and thus do NOT clobber
# PARENT_RESULT_DIR_LEGACY anymore — but a future code path that drops
# --run-name-suffix WOULD clobber it, so we preserve here defensively).
PARENT_ANCHOR_STABLE = RESULTS_DIR / "parent_recipe_anchor.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ISSUE_NUM = 260


# ── Per-condition spec ───────────────────────────────────────────────────────


@dataclass
class Condition:
    name: str  # e.g. "mt_n4"
    sub_exp: str  # "a" | "b" | "c"
    data_path: str  # relative to PROJECT_ROOT
    max_length: int


LEG1_CONDITIONS: list[Condition] = [
    # (a) multi-turn — needs max_length=8192 + possibly halved batch (OOM probe)
    Condition("mt_n1", "a", "data/leakage_experiment_issue260/mt_n1.jsonl", 8192),
    Condition("mt_n4", "a", "data/leakage_experiment_issue260/mt_n4.jsonl", 8192),
    Condition("mt_n16", "a", "data/leakage_experiment_issue260/mt_n16.jsonl", 8192),
    # (b) long completion — max_length=1536
    Condition("lc_short", "b", "data/leakage_experiment_issue260/lc_short.jsonl", 1536),
    Condition("lc_medium", "b", "data/leakage_experiment_issue260/lc_medium.jsonl", 1536),
    Condition("lc_long", "b", "data/leakage_experiment_issue260/lc_long.jsonl", 1536),
    # (c) sys-len — max_length=1280
    Condition("sl_short", "c", "data/leakage_experiment_issue260/sl_short.jsonl", 1280),
    Condition("sl_medium", "c", "data/leakage_experiment_issue260/sl_medium.jsonl", 1280),
    Condition("sl_long", "c", "data/leakage_experiment_issue260/sl_long.jsonl", 1280),
]


@dataclass
class Leg2Spec:
    """One eval-only rerun spec.

    Builds an `--eval-only-rerun <merged_path>` invocation with the optional
    `--eval-system-prompt-source` / `--eval-bystander-prompt-mode` overrides
    for sub-exp (c), or `--eval-multi-turn-K` for sub-exp (a).
    """

    name: str  # used as the result dir basename, e.g. "sl_short_leg2"
    leg1_name: str  # the Leg-1 condition whose merged dir we reuse
    sub_exp: str  # "a" | "c"
    eval_system_prompt_source: str | None = None
    eval_system_prompt_bystander_suffix: str | None = None
    # FIX 4: per Leg-2 (c) condition, declare the bystander prompt mode so
    # each model's bystanders evaluate at their TRAIN-MATCHED prompt:
    #   - sl_short_leg2: "short"          -> PERSONA_PROMPTS_SHORT[bystander]
    #   - sl_medium_leg2: "medium"        -> PERSONAS[bystander] (default)
    #   - sl_long_leg2: "medium+filler"   -> PERSONAS[bystander] + " " + FILLER
    eval_bystander_prompt_mode: str = "medium"
    eval_multi_turn_K: int = 1
    extras: list[str] = field(default_factory=list)


# ── Builders for Leg 2 specs ────────────────────────────────────────────────


def _leg2_c_specs() -> list[Leg2Spec]:
    """Leg 2 for sub-exp (c): each model evaluated at its train-matched prompt.

    Pulls FILLER_NEUTRAL from build_issue260_data so the bystander suffix on
    `sl_long_leg2` matches the train-time filler bit-for-bit.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from build_issue260_data import FILLER_NEUTRAL

    from explore_persona_space.personas import PERSONAS

    short_src = "You are a librarian."
    med_src = PERSONAS["librarian"]
    long_src = med_src + " " + FILLER_NEUTRAL
    return [
        Leg2Spec(
            name="sl_short_leg2",
            leg1_name="sl_short",
            sub_exp="c",
            eval_system_prompt_source=short_src,
            # FIX 4: bystanders use PERSONA_PROMPTS_SHORT (sl_short_leg2
            # train-matched per plan §3.4 v2: each model evaluated at its
            # train-matched system prompt, including bystanders).
            eval_bystander_prompt_mode="short",
            eval_system_prompt_bystander_suffix=None,
        ),
        Leg2Spec(
            name="sl_medium_leg2",
            leg1_name="sl_medium",
            sub_exp="c",
            eval_system_prompt_source=med_src,
            # Default medium: PERSONAS[bystander] unmodified (sl_medium_leg2
            # train-matched).
            eval_bystander_prompt_mode="medium",
            eval_system_prompt_bystander_suffix=None,
        ),
        Leg2Spec(
            name="sl_long_leg2",
            leg1_name="sl_long",
            sub_exp="c",
            eval_system_prompt_source=long_src,
            # PERSONAS[bystander] + " " + FILLER_NEUTRAL (sl_long_leg2
            # train-matched: each bystander gets the same filler suffix the
            # source persona was trained with).
            eval_bystander_prompt_mode="medium+filler",
            eval_system_prompt_bystander_suffix=FILLER_NEUTRAL,
        ),
    ]


def _leg2_a_specs() -> list[Leg2Spec]:
    """Leg 2 for sub-exp (a): each model evaluated at its train-matched K turns.

    `eval_multi_turn_K` for `mt_n1` is 1 (no warmup turns) — purely a sanity
    pass-through. Plan section 3.2 v3 keeps mt_n1 single-turn so Leg 1 ==
    Leg 2 within seed noise; it's run anyway so all 3 (a) panels share an
    identical pipeline.
    """
    return [
        Leg2Spec(name="mt_n1_leg2", leg1_name="mt_n1", sub_exp="a", eval_multi_turn_K=1),
        Leg2Spec(name="mt_n4_leg2", leg1_name="mt_n4", sub_exp="a", eval_multi_turn_K=4),
        Leg2Spec(name="mt_n16_leg2", leg1_name="mt_n16", sub_exp="a", eval_multi_turn_K=16),
    ]


# ── State helpers ───────────────────────────────────────────────────────────


def preserve_parent_anchor() -> None:
    """FIX 1: Copy the parent #271 librarian source-rate anchor to a stable
    location BEFORE Leg 1 starts.

    The parent recipe (#246/#271) wrote to
        eval_results/leakage_experiment/marker_librarian_asst_excluded_medium_seed42/run_result.json

    Plan §3.10 promises the (c) Leg-1 panel includes a dotted reference line
    derived from this anchor. With FIX 2 (`--run-name-suffix`), Leg-1 conditions
    no longer clobber the legacy parent dir directly — but if the user reverts
    that flag, OR if cleanup deletes the legacy dir, the anchor is lost. This
    helper copies `run_result.json` to a stable path under
    `eval_results/issue260/parent_recipe_anchor.json`, which the analyzer then
    prefers over the legacy location (see `analyze_issue260.py::_load_parent_anchor`).

    Idempotent: if the stable copy already exists, no-op.
    """
    legacy = PARENT_RESULT_DIR_LEGACY / "run_result.json"
    if PARENT_ANCHOR_STABLE.exists():
        print(f"[launcher] parent #271 anchor already preserved -> {PARENT_ANCHOR_STABLE}")
        return
    if not legacy.exists():
        print(
            f"[launcher] WARN: parent #271 result not found at {legacy}; "
            f"(c) Leg-1 panel will render without the dotted reference line."
        )
        return
    PARENT_ANCHOR_STABLE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy, PARENT_ANCHOR_STABLE)
    print(f"[launcher] preserved parent #271 anchor -> {PARENT_ANCHOR_STABLE}")


def load_state() -> dict:
    state_path = RESULTS_DIR / "launcher_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"issue": ISSUE_NUM, "started": time.time(), "runs": {}}


def save_state(state: dict) -> None:
    state_path = RESULTS_DIR / "launcher_state.json"
    state_path.write_text(json.dumps(state, indent=2))


# ── GitHub progress markers ──────────────────────────────────────────────────


def post_progress(message: str, enabled: bool) -> None:
    """Post a `<!-- epm:progress vN -->` comment on issue #260, if `gh` is on PATH."""
    if not enabled:
        return
    if shutil.which("gh") is None:
        return
    body = f"<!-- epm:progress v1 -->\n## Launcher progress (issue #260)\n\n{message}\n"
    try:
        subprocess.run(
            ["gh", "issue", "comment", str(ISSUE_NUM), "--body", body],
            check=True,
            cwd=str(PROJECT_ROOT),
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[launcher] gh issue comment failed: {e}; continuing")


# ── Run-leakage-experiment subprocess wrapper ────────────────────────────────


def _read_oom_config() -> dict:
    p = ISSUE_DATA / ".preflight_oom_config.json"
    if not p.exists():
        return {"per_device_batch_size": 4, "grad_accum": 4, "max_length": 8192}
    return json.loads(p.read_text())


def _runner_output_dir(run_name_suffix: str) -> Path:
    """The runner writes to EVAL_RESULTS_DIR / run_name; with --run-name-suffix
    that becomes <BASE_RUN_NAME>_<suffix>/. Returns the matching local path."""
    return LEAKAGE_RESULTS_ROOT / f"{BASE_RUN_NAME}_{run_name_suffix}"


def _build_leg1_cmd(cond: Condition, oom_cfg: dict, gpu: int, pod: str) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/archive/run_leakage_experiment.py",
        "--trait",
        "marker",
        "--source",
        "librarian",
        "--neg-set",
        "asst_excluded",
        "--prompt-length",
        "medium",
        "--seed",
        "42",
        "--gpu",
        str(gpu),
        "--pod",
        pod,
        "--phase",
        "a1",
        "--data-path",
        cond.data_path,
        "--max-length",
        str(cond.max_length),
        # FIX 2: per-condition discriminator; disambiguates the runner's
        # output_dir, the HF Hub upload path_in_repo, and the WandB run name.
        "--run-name-suffix",
        cond.name,
    ]
    # Apply OOM-probe config to sub-exp (a) ONLY (plan section 8 risk #1b).
    if cond.sub_exp == "a":
        cmd += [
            "--per-device-batch-size",
            str(oom_cfg.get("per_device_batch_size", 4)),
            "--grad-accum",
            str(oom_cfg.get("grad_accum", 4)),
        ]
    return cmd


def _build_leg2_cmd(spec: Leg2Spec, gpu: int, pod: str) -> list[str]:
    merged_path = RESULTS_DIR / spec.leg1_name / "merged"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/archive/run_leakage_experiment.py",
        "--trait",
        "marker",
        "--source",
        "librarian",
        "--neg-set",
        "asst_excluded",
        "--prompt-length",
        "medium",
        "--seed",
        "42",
        "--gpu",
        str(gpu),
        "--pod",
        pod,
        "--phase",
        "a1",
        "--eval-only-rerun",
        str(merged_path),
        # FIX 2: per-condition discriminator for HF Hub & WandB run-name de-collision.
        "--run-name-suffix",
        spec.name,
        # FIX 4: bystander prompt mode (medium / short / medium+filler).
        "--eval-bystander-prompt-mode",
        spec.eval_bystander_prompt_mode,
    ]
    if spec.eval_system_prompt_source is not None:
        cmd += ["--eval-system-prompt-source", spec.eval_system_prompt_source]
    if spec.eval_system_prompt_bystander_suffix is not None:
        cmd += [
            "--eval-system-prompt-bystander-suffix",
            spec.eval_system_prompt_bystander_suffix,
        ]
    if spec.eval_multi_turn_K > 1:
        cmd += ["--eval-multi-turn-K", str(spec.eval_multi_turn_K)]
    return cmd


def _run_subprocess(cmd: list[str], log_path: Path, env_extra: dict | None = None) -> int:
    """Run cmd, tee stdout+stderr to log_path. Returns exit code."""
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "42")
    if env_extra:
        env.update(env_extra)
    print(f"[launcher] $ {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"[launcher]   log -> {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        rc = proc.wait()
    return rc


# ── Main per-condition execution ─────────────────────────────────────────────


def _move_parent_result_dir(target_name: str, source_dir: Path) -> Path:
    """Atomically rename the runner's per-condition output dir to `<target_name>`.

    With FIX 2 (`--run-name-suffix`), each condition writes to its own
    `eval_results/leakage_experiment/<BASE_RUN_NAME>_<COND>/` directory, so
    successive conditions no longer clobber each other. We still mv into
    `eval_results/issue260/<COND>/` as the canonical analysis-side path.
    `source_dir` is provided by the caller (per-condition; computed via
    `_runner_output_dir(cond.name)`).
    """
    target = RESULTS_DIR / target_name
    if target.exists():
        # Don't clobber a prior successful result; rename the in-progress dir to
        # `<target>.stale` so a human can investigate.
        stale = RESULTS_DIR / f"{target_name}.stale_{int(time.time())}"
        target.rename(stale)
        print(f"[launcher] target {target} already existed; moved to {stale}")
    if not source_dir.exists():
        raise RuntimeError(
            f"expected runner to write {source_dir}; not found "
            "(did the run crash before producing any output?)"
        )
    source_dir.rename(target)
    print(f"[launcher] mv {source_dir.name} -> {target}")
    return target


def run_leg1(args, state: dict, post_progress_enabled: bool) -> None:
    oom_cfg = _read_oom_config()
    print(f"[launcher] OOM config (sub-exp (a) only): {oom_cfg}")

    selected = _filter_conditions(LEG1_CONDITIONS, args.conditions)

    for cond in selected:
        run_key = f"leg1__{cond.name}"
        if state["runs"].get(run_key, {}).get("status") == "ok" and (
            (RESULTS_DIR / cond.name).exists()
        ):
            print(f"[launcher] skip {run_key} (already ok)")
            continue

        log_path = RESULTS_DIR / f"{cond.name}_leg1.log"
        cmd = _build_leg1_cmd(cond, oom_cfg, gpu=args.gpu, pod=args.pod)
        t0 = time.time()
        state["runs"][run_key] = {
            "status": "running",
            "started": t0,
            "cmd": cmd,
            "log": str(log_path),
        }
        save_state(state)

        rc = _run_subprocess(cmd, log_path)

        wall = time.time() - t0
        if rc != 0:
            state["runs"][run_key].update(
                {"status": "fail", "rc": rc, "wall_s": wall, "ended": time.time()}
            )
            save_state(state)
            raise SystemExit(f"[launcher] {cond.name} (Leg 1) FAILED with rc={rc}; see {log_path}")

        # Atomic rename of the runner's per-condition result dir to our
        # condition-keyed dir. FIX 2: source dir is now per-condition because
        # we pass --run-name-suffix=<COND> to the runner.
        renamed = _move_parent_result_dir(cond.name, _runner_output_dir(cond.name))

        state["runs"][run_key].update(
            {
                "status": "ok",
                "rc": 0,
                "wall_s": wall,
                "result_dir": str(renamed),
                "ended": time.time(),
            }
        )
        save_state(state)
        print(f"[launcher] {cond.name} (Leg 1) ok in {wall / 60:.1f} min")

        # Post progress milestone every time a sub-experiment's 3 conditions
        # all complete (so 3 progress posts: end of (a), (b), (c)).
        if cond.name == "mt_n16":
            post_progress(
                "Leg 1 sub-exp (a) complete (3/3 conditions). Continuing to (b).",
                post_progress_enabled,
            )
        elif cond.name == "lc_long":
            post_progress(
                "Leg 1 sub-exp (b) complete (3/3 conditions). Continuing to (c).",
                post_progress_enabled,
            )
        elif cond.name == "sl_long":
            post_progress(
                "Leg 1 complete (9/9 conditions). Continuing to Leg 2 reruns.",
                post_progress_enabled,
            )


def _ensure_merged_dir(leg1_name: str) -> Path:
    """Ensure the Leg-1 merged dir exists locally before a Leg-2 rerun.

    Pod disk-quota recovery: Leg-1 merged dirs may have been deleted to
    stay under the per-pod quota. Re-download from HF Hub on demand.
    """
    merged_path = RESULTS_DIR / leg1_name / "merged"
    if merged_path.exists() and (merged_path / "model.safetensors").exists():
        print(f"[launcher] {leg1_name}/merged present locally; using")
        return merged_path

    repo_id = "superkaiba1/explore-persona-space"
    repo_path = f"leakage_experiment/{BASE_RUN_NAME}_{leg1_name}"
    print(f"[launcher] downloading {repo_path} from HF Hub -> {merged_path}")
    sys.stdout.flush()

    from huggingface_hub import snapshot_download

    merged_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{repo_path}/*"],
        local_dir=str(RESULTS_DIR / leg1_name / ".hf_dl"),
        max_workers=4,
    )
    # snapshot_download flattens the repo path; move files into merged/
    src_dir = (
        RESULTS_DIR / leg1_name / ".hf_dl" / "leakage_experiment" / f"{BASE_RUN_NAME}_{leg1_name}"
    )
    if not src_dir.exists():
        raise RuntimeError(f"snapshot_download did not produce expected dir: {src_dir}")
    for f in src_dir.iterdir():
        f.rename(merged_path / f.name)
    # Cleanup the .hf_dl staging dir
    shutil.rmtree(RESULTS_DIR / leg1_name / ".hf_dl", ignore_errors=True)
    print(f"[launcher] {leg1_name}/merged downloaded successfully")
    sys.stdout.flush()
    return merged_path


def _cleanup_merged_dir(leg1_name: str) -> None:
    """Delete the Leg-1 merged dir after its Leg-2 rerun completes (free disk)."""
    merged_path = RESULTS_DIR / leg1_name / "merged"
    if merged_path.exists():
        shutil.rmtree(merged_path)
        print(f"[launcher] cleaned up {merged_path}")
        sys.stdout.flush()


def run_leg2(args, state: dict, post_progress_enabled: bool) -> None:
    """Run Leg 2 reruns for sub-exp (c) and (a)."""
    selected_c = _filter_leg2(_leg2_c_specs(), args.conditions)
    selected_a = _filter_leg2(_leg2_a_specs(), args.conditions)

    for sub_label, specs in (("c", selected_c), ("a", selected_a)):
        for spec in specs:
            run_key = f"leg2__{spec.name}"
            target_dir = RESULTS_DIR / spec.name
            if state["runs"].get(run_key, {}).get("status") == "ok" and target_dir.exists():
                print(f"[launcher] skip {run_key} (already ok)")
                continue

            # Disk-quota recovery: download merged dir from HF Hub if missing.
            _ensure_merged_dir(spec.leg1_name)

            log_path = RESULTS_DIR / f"{spec.name}.log"
            cmd = _build_leg2_cmd(spec, gpu=args.gpu, pod=args.pod)
            t0 = time.time()
            state["runs"][run_key] = {
                "status": "running",
                "started": t0,
                "cmd": cmd,
                "log": str(log_path),
            }
            save_state(state)

            rc = _run_subprocess(cmd, log_path)
            wall = time.time() - t0
            if rc != 0:
                state["runs"][run_key].update(
                    {"status": "fail", "rc": rc, "wall_s": wall, "ended": time.time()}
                )
                save_state(state)
                raise SystemExit(
                    f"[launcher] {spec.name} (Leg 2 sub-exp {sub_label}) "
                    f"FAILED with rc={rc}; see {log_path}"
                )

            # FIX 2: Leg-2 also passes --run-name-suffix=<spec.name>, so the
            # runner writes to <BASE_RUN_NAME>_<spec.name>/ — not the legacy
            # parent dir.
            renamed = _move_parent_result_dir(spec.name, _runner_output_dir(spec.name))
            state["runs"][run_key].update(
                {
                    "status": "ok",
                    "rc": 0,
                    "wall_s": wall,
                    "result_dir": str(renamed),
                    "ended": time.time(),
                }
            )
            save_state(state)
            print(f"[launcher] {spec.name} (Leg 2 sub-exp {sub_label}) ok in {wall / 60:.1f} min")

            # Free 15 GB by deleting the merged dir we just used. Subsequent
            # Leg-2 runs (or future invocations) will re-download from HF Hub.
            _cleanup_merged_dir(spec.leg1_name)

        post_progress(
            f"Leg 2 sub-exp ({sub_label}) complete ({len(specs)}/{len(specs)} conditions).",
            post_progress_enabled,
        )


# ── Subset / resume helpers ──────────────────────────────────────────────────


def _filter_conditions(all_conds: list[Condition], names: list[str] | None) -> list[Condition]:
    if not names:
        return all_conds
    chosen = [c for c in all_conds if c.name in names]
    if len(chosen) != len(names):
        unknown = sorted(set(names) - {c.name for c in all_conds})
        if unknown:
            raise SystemExit(f"unknown condition names: {unknown}")
    return chosen


def _filter_leg2(specs: list[Leg2Spec], names: list[str] | None) -> list[Leg2Spec]:
    if not names:
        return specs
    return [s for s in specs if s.name in names]


# ── Entrypoint ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pod",
        default="epm-issue-260",
        help="Pod identifier passed to run_leakage_experiment.py (logging only).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index (default 0; sub-exp (a) is sequential on this GPU).",
    )
    parser.add_argument(
        "--skip-leg1",
        action="store_true",
        help="Reuse already-mv'd Leg-1 dirs and only run Leg 2.",
    )
    parser.add_argument(
        "--skip-leg2",
        action="store_true",
        help="Run only Leg 1 (debug / smoke).",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help=(
            "Restrict to a subset of named conditions. Use Leg-1 names "
            "(mt_n1, ..., sl_long) or Leg-2 names (mt_n4_leg2, sl_long_leg2)."
        ),
    )
    parser.add_argument(
        "--no-post-progress",
        action="store_true",
        help="Do not post `epm:progress` markers on the issue.",
    )
    args = parser.parse_args(argv)

    if os.environ.get("PYTHONHASHSEED") != "42":
        raise SystemExit(
            "PYTHONHASHSEED=42 is required (matches plan + recipe-parent seed). "
            "Re-launch with: PYTHONHASHSEED=42 nohup uv run python scripts/launch_issue260.py ..."
        )

    state = load_state()
    state.setdefault("runs", {})
    save_state(state)

    post_progress_enabled = not args.no_post_progress

    # FIX 1: preserve parent #271 anchor BEFORE Leg 1 starts (plan §3.10).
    preserve_parent_anchor()

    # NIT: document the sl_long token deviation surfaced by code-review v1.
    # Plan section 3.7 #3 nominal range was [240, 270]; preflight relaxed to
    # [225, 285] because PERSONAS["librarian"] + " " + FILLER_NEUTRAL tokenizes
    # to ~238 tokens under the Qwen tokenizer (not the plan's nominal 256).
    # The geometric ratio sl_short:sl_medium:sl_long is therefore 5:15:238
    # ~ 1x:3x:48x (plan said ~50x). Within-noise of plan's "~50x dynamic range".
    print(
        "[launcher] Assumption: sl_long system prompt tokenizes to ~238 tokens "
        "(actual), not the plan's nominal 256. Ratio sl_short:sl_medium:sl_long "
        "~ 1x:3x:48x (within-noise of plan's stated ~50x dynamic range)."
    )

    t_start = time.time()
    if not args.skip_leg1:
        post_progress(
            f"Launcher started (pod={args.pod}, gpu={args.gpu}). Leg 1: 9 conditions.",
            post_progress_enabled,
        )
        run_leg1(args, state, post_progress_enabled)

    if not args.skip_leg2:
        run_leg2(args, state, post_progress_enabled)

    wall = time.time() - t_start
    state["finished"] = time.time()
    state["wall_s"] = wall
    save_state(state)
    print(f"[launcher] all done in {wall / 60:.1f} min")
    post_progress(
        f"All conditions complete in {wall / 60:.1f} min. Ready for analyze_issue260.py.",
        post_progress_enabled,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
