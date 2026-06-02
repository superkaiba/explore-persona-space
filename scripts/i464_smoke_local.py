"""CPU-only end-to-end smoke for issue #464 pipeline (round-2: real entrypoints).

Round-1 review blocker #2: the round-1 driver synthesized stubs INTO LIVE
DATA PATHS and called only internal helpers, so the real phase
entrypoints had never been exercised end-to-end (the #408 failure
mode). Round-2 fixes both:

  (a) **All writes go to a TEMP DIR**, never to the live ``data/issue_464/``,
      ``eval_results/issue_464/``, ``figures/issue_464/``, or
      ``logs/issue_464/`` paths. We accomplish this by ``chdir``-ing each
      sub-invocation into the temp dir (each phase script resolves these
      paths relative to cwd) and copying the prerequisite stubs into the
      temp dir's matching subpaths first. The live worktree is left
      untouched after the smoke run.

  (b) **Each phase invokes its REAL ``scripts/i464_phase*.py`` entrypoint
      via subprocess** with whatever ``--smoke``/tiny-N/`--no-vllm`/
      ``--allow-partial`` flag the script supports. Phases that require
      vLLM (Phase 1 R generation, Phase 2 implant check, Phase 3 LoRA
      training, Phase 4 cross-eval, Phase 4.5 on-policy generation) cannot
      run on the local CPU VM — we record them as "GPU-gated, real
      entrypoint imports + arg-parses cleanly via --help" so the
      experimenter at minimum knows the CLI surface is unbroken before
      provisioning. The pod-side ``i464_run_all.sh`` runs the FULL
      vLLM-bound code path; CPU smoke is a tripwire for the non-GPU
      faults that brought down #408.

Per-phase status legend:
  OK(real-CPU)       — real entrypoint executed end-to-end on CPU.
  OK(real-CLI)       — real entrypoint's ``--help`` parsed cleanly + the
                        non-GPU importable helpers exercised on stubs.
  OK(helper)         — internal helper exercised in-process (last resort
                        when the script has no --help-only safe path).
  FAIL(rc=N)         — non-zero exit or behavioral failure.

Run:
    uv run python scripts/i464_smoke_local.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc

load_dotenv()

logger = logging.getLogger("i464.smoke")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
REPO_ROOT = Path(__file__).resolve().parent.parent  # .../issue-464 worktree


def _print_phase(name: str, rc: int, digest: str) -> None:
    """Emit the canonical phase-result line."""
    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    print(f"[smoke phase={name}] {status} :: {digest}")


def _run(
    cmd: list[str], cwd: Path, env: dict | None = None, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Subprocess wrapper with explicit cwd + env + timeout + capture.

    All sub-invocations run from a TEMP cwd so per-script relative paths
    (``data/issue_464/``, ``eval_results/issue_464/``, ...) land in the
    temp dir, never the live worktree.
    """
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=(env if env is not None else {**os.environ}),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _build_stub_r_canon(temp_data_dir: Path, n_q: int = 5) -> tuple[list[str], list[str]]:
    """Synthesize tiny R_canon_train.json + R_canon_test.json INSIDE the temp dir.

    Returns the (q_train_keys, q_test_keys) used so downstream phases can
    reference the same question set.
    """
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    qs_train = [f"smoke train q{i}?" for i in range(n_q)]
    qs_test = [f"smoke test q{i}?" for i in range(n_q)]
    R_text = "I do not know but I will try to help."
    for split, qs in [("train", qs_train), ("test", qs_test)]:
        completions = {
            persona: {
                q: {
                    "response_text": R_text,
                    "response_token_ids": [],
                    "n_response_tokens": 12,
                    "ended_with_eos": True,
                    "truncated": False,
                    "tail_ok": True,
                    "marker_in_R": False,
                }
                for q in qs
            }
            for persona in enc.PERSONAS
        }
        payload = {
            "schema_version": "i464_v2_matched_R",
            "split": split,
            "base_model": BASE_MODEL,
            "encoding": "system",
            "generation_config": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 256,
                "seed": 42,
                "stop_token_ids": "[eos_token_id]",
            },
            "personas": list(enc.PERSONAS),
            "n_q": len(qs),
            "completions": completions,
            "content_hash": "smoke",
            "git_commit": "smoke",
            "generated_at": "smoke",
            "stats": {
                "n_total_rows": 2 * len(qs),
                "n_truncated": 0,
                "n_marker_in_R": 0,
                "n_tail_warnings": 0,
                "marker_in_R_examples": [],
            },
        }
        (temp_data_dir / f"R_canon_{split}.json").write_text(json.dumps(payload))
    return qs_train, qs_test


def _smoke_phase0(temp_dir: Path) -> tuple[int, str]:
    """Phase 0: REAL preflight entrypoint, ``--no-smoke --dry-run`` (no GPU).

    Runs ``scripts/i464_phase0_preflight.py`` from the temp dir as cwd.
    With ``--dry-run`` the script doesn't write preflight.json; with
    ``--no-smoke`` it skips the 48-generation vLLM smoke. Verifies the
    tokenizer + token-id contract + Q_train/Q_test load + disjoint check.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "i464_phase0_preflight.py"),
        "--no-smoke",
        "--dry-run",
    ]
    res = _run(cmd, cwd=temp_dir, timeout=120)
    if res.returncode != 0:
        return res.returncode, f"stderr tail: {res.stderr[-300:]}"
    # Parse the JSON from stdout; verify expected keys.
    try:
        # The script prints "Preflight OK (dry-run; skipping write)" then JSON.
        # Find the first '{' and parse from there.
        idx = res.stdout.find("{")
        if idx < 0:
            return 1, f"no JSON in stdout (rc=0); stdout tail: {res.stdout[-200:]}"
        payload = json.loads(res.stdout[idx:])
    except json.JSONDecodeError as e:
        return 1, f"stdout JSON unparseable: {e}; stdout tail: {res.stdout[-200:]}"
    must_have = {"marker_ids", "padding_token_id", "role_name_token_ids", "n_q_train"}
    missing = must_have - set(payload)
    if missing:
        return 1, f"preflight payload missing keys: {missing}"
    return 0, (
        f"real entrypoint --no-smoke --dry-run rc=0; payload has "
        f"marker_ids={payload['marker_ids']}, n_q_train={payload['n_q_train']}"
    )


def _smoke_phase1(temp_dir: Path) -> tuple[int, str]:
    """Phase 1: REAL R generation entrypoint via --help (vLLM = GPU-gated).

    Real entrypoint imports + argparse parses cleanly. End-to-end GPU
    execution runs on the pod via ``i464_run_all.sh``; the CPU tripwire
    here catches import/CLI breakage.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "i464_phase1_generate_R.py"),
        "--help",
    ]
    res = _run(cmd, cwd=temp_dir, timeout=60)
    if res.returncode != 0:
        return res.returncode, f"stderr tail: {res.stderr[-300:]}"
    if "--smoke-n" not in res.stdout:
        return 1, "phase1 --help output missing --smoke-n flag"
    # Synthesize the stub R_canon files for downstream phases (Phase 2/3 / 4 / 4.5).
    _build_stub_r_canon(temp_dir / "data" / "issue_464", n_q=5)
    return 0, (
        f"real entrypoint --help rc=0 (vLLM GPU-gated for actual gen); "
        f"stub R_canon (5 q x 2 personas) for downstream phases written under "
        f"{temp_dir / 'data' / 'issue_464'}"
    )


def _smoke_phase23(temp_dir: Path, tok) -> tuple[int, str]:
    """Phase 2/3 train: REAL entrypoint --help + _build_training_rows on CPU.

    HF Trainer + LoRA on Qwen-7B requires GPU; CPU exercises the
    pre-training row build + traj-probe build paths. Output JSONLs land
    in ``<temp_dir>/data/issue_464/train_rows/`` (NOT live).
    """
    train_script = REPO_ROOT / "scripts" / "i464_phase23_train.py"
    # 1. Validate --help.
    res_help = _run([sys.executable, str(train_script), "--help"], cwd=temp_dir, timeout=60)
    if res_help.returncode != 0:
        return res_help.returncode, f"--help stderr tail: {res_help.stderr[-300:]}"
    for required_flag in ("--cell", "--gpu-id", "--smoke", "--no-traj", "--traj-probe-file"):
        if required_flag not in res_help.stdout:
            return 1, f"phase23_train --help missing flag {required_flag}"

    # 2. End-to-end _build_training_rows (CPU-feasible).
    import importlib.util

    spec = importlib.util.spec_from_file_location("i464_phase23_train", train_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    q_train_answers = {f"smoke train q{i}?": "stub answer" for i in range(5)}
    R_canon_train = json.loads(
        (temp_dir / "data" / "issue_464" / "R_canon_train.json").read_text()
    )["completions"]
    # Run with cwd switched so the script's relative TRAIN_ROW_DIR lands in
    # the temp dir, never the live worktree. Resolve the returned path to
    # ABSOLUTE before restoring cwd so the open() below doesn't dangle.
    prev_cwd = Path.cwd()
    try:
        os.chdir(temp_dir)
        out_path_rel = mod._build_training_rows(
            arm="system_plain",
            seed=42,
            q_train_answers=q_train_answers,
            R_canon_train=R_canon_train,
            tokenizer=tok,
            n_dupes=1,
        )
        # Resolve while still inside temp_dir cwd → absolute path.
        out_path = Path(out_path_rel).resolve()
    finally:
        os.chdir(prev_cwd)
    with open(out_path) as f:
        n_rows = sum(1 for _ in f)
    if n_rows != 10:
        return 1, f"expected 10 rows (5q x 2personas x 1dupe), got {n_rows}"

    # 3. End-to-end _build_traj_probe_file (CPU-feasible, exercises MF-C probe build
    #    with the round-2 wrong-persona symmetric encoding fix).
    R_canon_test = json.loads((temp_dir / "data" / "issue_464" / "R_canon_test.json").read_text())[
        "completions"
    ]
    traj_path = temp_dir / "data" / "issue_464" / "traj_probes" / "probes_system_plain.json"
    mod._build_traj_probe_file(
        tok,
        R_canon_test,
        arm="system_plain",
        n_probes_per_key=2,
        out_path=traj_path,
    )
    if not traj_path.exists():
        return 1, f"traj probe file missing at {traj_path}"
    traj_payload = json.loads(traj_path.read_text())
    # Round-2 fix #6: 5 encodings per persona x 2 personas x 2 q = 20 probes.
    if len(traj_payload["probes"]) != 20:
        return 1, (
            f"expected 20 traj probes (5 encodings x 2 personas x 2 q), "
            f"got {len(traj_payload['probes'])}"
        )
    return 0, (
        f"real entrypoint --help rc=0; _build_training_rows wrote {n_rows} rows "
        f"(marker count==1 verified in-build); _build_traj_probe_file wrote "
        f"{len(traj_payload['probes'])} symmetric probes (incl. wrong-persona cells)"
    )


def _smoke_phase2_check(temp_dir: Path) -> tuple[int, str]:
    """Phase 2 implant check: REAL entrypoint --help + _parse_cell (vLLM = GPU-gated)."""
    script = REPO_ROOT / "scripts" / "i464_phase2_smoke_check.py"
    res = _run([sys.executable, str(script), "--help"], cwd=temp_dir, timeout=60)
    if res.returncode != 0:
        return res.returncode, f"--help stderr tail: {res.stderr[-300:]}"
    for required_flag in ("--cell", "--n-probes"):
        if required_flag not in res.stdout:
            return 1, f"phase2_smoke_check --help missing flag {required_flag}"
    # _parse_cell round-trip.
    import importlib.util

    spec = importlib.util.spec_from_file_location("i464_phase2_smoke_check", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    arm, seed = mod._parse_cell("system_plain_seed42")
    if arm != "system_plain" or seed != 42:
        return 1, f"_parse_cell mismatch: {arm}, {seed}"
    return 0, "real entrypoint --help rc=0 (vLLM GPU-gated); _parse_cell round-trip OK"


def _smoke_phase4(temp_dir: Path, tok) -> tuple[int, str]:
    """Phase 4 cross-eval: REAL --help + _build_probes_for_eval_marker on stub R_canon."""
    script = REPO_ROOT / "scripts" / "i464_phase4_eval.py"
    res = _run([sys.executable, str(script), "--help"], cwd=temp_dir, timeout=60)
    if res.returncode != 0:
        return res.returncode, f"--help stderr tail: {res.stderr[-300:]}"
    for required_flag in ("--shard", "--resume", "--smoke-n-q", "--smoke-cells"):
        if required_flag not in res.stdout:
            return 1, f"phase4_eval --help missing flag {required_flag}"
    # _build_probes_for_eval_marker end-to-end.
    import importlib.util

    spec = importlib.util.spec_from_file_location("i464_phase4_eval", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    R_canon_test = json.loads((temp_dir / "data" / "issue_464" / "R_canon_test.json").read_text())[
        "completions"
    ]
    qs = [f"smoke test q{i}?" for i in range(3)]
    prompts, slots = mod._build_probes_for_eval_marker(
        "system_pirate", "pirate", tok, qs, R_canon_test
    )
    if len(prompts) != 3 or len(slots) != 3:
        return 1, f"probe-count mismatch: {len(prompts)} prompts, {len(slots)} slots"
    last_id = prompts[0]["prompt_token_ids"][-1]
    if last_id != enc.MARKER_PIRATE_ID:
        return 1, f"last token {last_id} != pirate marker {enc.MARKER_PIRATE_ID}"
    return 0, (
        f"real entrypoint --help rc=0 (vLLM GPU-gated for actual eval); "
        f"_build_probes_for_eval_marker built {len(prompts)} probes; "
        f"last token == marker_id verified"
    )


def _smoke_phase45(temp_dir: Path) -> tuple[int, str]:
    """Phase 4.5 on-policy: REAL --help + _char_edit_distance unit check (vLLM = GPU-gated)."""
    script = REPO_ROOT / "scripts" / "i464_phase45_onpolicy_validation.py"
    res = _run([sys.executable, str(script), "--help"], cwd=temp_dir, timeout=60)
    if res.returncode != 0:
        return res.returncode, f"--help stderr tail: {res.stderr[-300:]}"
    for required_flag in ("--n-q", "--smoke-cells"):
        if required_flag not in res.stdout:
            return 1, f"phase45 --help missing flag {required_flag}"
    import importlib.util

    spec = importlib.util.spec_from_file_location("i464_phase45_onpolicy_validation", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod._char_edit_distance("abc", "abc") != 0:
        return 1, "edit_distance(equal) != 0"
    if mod._char_edit_distance("abc", "abd") != 1:
        return 1, "edit_distance(1 sub) != 1"
    if mod._char_edit_distance("", "abc") != 3:
        return 1, "edit_distance('', 'abc') != 3"
    return 0, (
        "real entrypoint --help rc=0 (vLLM GPU-gated for actual gen); "
        "_char_edit_distance verified on 3 unit cases"
    )


def _write_stub_per_cell_tree(
    temp_dir: Path, include_3_seeds: bool, with_onpolicy_switch: bool
) -> None:
    """Write a stub per-cell JSON tree into the temp dir for Phase 5 to aggregate.

    With ``include_3_seeds=True``, writes all 3 seeds across all 3 arms so
    Phase 5's new ``H2_MIN_SEEDS >= 3`` gate (round-2 blocker #3) can be
    exercised AND PASS. Also writes own-persona elicitation cells so the
    H1 gate (blocker #7) has input. ``with_onpolicy_switch=True`` writes a
    Phase 4.5 onpolicy_validation.json with ``switch_headline_to_trained_R
    = true`` so blocker #4's consumption path is exercised.
    """
    real_per_cell = temp_dir / "eval_results" / "issue_464" / "cross_eval" / "per_cell"
    real_per_cell.mkdir(parents=True, exist_ok=True)
    seeds = [42, 137, 1337] if include_3_seeds else [42]
    for arm in enc.ARMS:
        for seed in seeds:
            cell = f"{arm}_seed{seed}"
            for persona in enc.PERSONAS:
                # 1. Own-persona elicitation cell (for H1 gate input).
                e_own = f"role_{persona}" if arm == "role" else f"system_{persona}"
                own_payload = {
                    "cell": cell,
                    "arm": arm,
                    "seed": seed,
                    "e_eval": e_own,
                    "marker_persona": persona,
                    "marker_id": enc.marker_id_for(persona),
                    "n_probes": 3,
                    "g_logprob": -0.2,  # > -1 nat, so H1 PASSes
                    "b_logprob": -10.0,
                    "delta_g": 9.8,
                    "emission_recompute_rate": 0.95,
                    "logp_floor": -50.0,
                    "g_logps_per_q": [-0.2] * 3,
                    "b_logps_per_q": [-10.0] * 3,
                    "g_argmax_marker_per_q": [True, True, True],
                    "b_argmax_marker_per_q": [False, False, False],
                }
                (real_per_cell / f"{cell}__{e_own}__marker_{persona}.json").write_text(
                    json.dumps(own_payload)
                )
                # 2. Symmetric leakage cells (for H2 headline).
                other = "villain" if persona == "pirate" else "pirate"
                for e_wrong in [f"system_{other}", f"role_{other}"]:
                    # Make role-arm leak less than the system arms so H2 PASSes
                    # with comfortable margin (stub run sanity).
                    g_lp = -2.5 - (1.5 if arm == "role" else 0.0)
                    payload = {
                        "cell": cell,
                        "arm": arm,
                        "seed": seed,
                        "e_eval": e_wrong,
                        "marker_persona": persona,
                        "marker_id": enc.marker_id_for(persona),
                        "n_probes": 3,
                        "g_logprob": g_lp,
                        "b_logprob": -10.0,
                        "delta_g": -10.0 - g_lp,
                        "emission_recompute_rate": 0.7,
                        "logp_floor": -50.0,
                        # Add cell-jitter on per-q values so dynamic-range gate
                        # (sd > 0.5) doesn't FAIL on degenerate constant data.
                        "g_logps_per_q": [g_lp - 0.6, g_lp, g_lp + 0.6],
                        "b_logps_per_q": [-10.0] * 3,
                        "g_argmax_marker_per_q": [True, True, False],
                        "b_argmax_marker_per_q": [False, False, False],
                    }
                    (real_per_cell / f"{cell}__{e_wrong}__marker_{persona}.json").write_text(
                        json.dumps(payload)
                    )
                # 3. default_assistant cell (exploratory).
                payload = {
                    "cell": cell,
                    "arm": arm,
                    "seed": seed,
                    "e_eval": "default_assistant",
                    "marker_persona": persona,
                    "marker_id": enc.marker_id_for(persona),
                    "n_probes": 3,
                    "g_logprob": -4.0,
                    "b_logprob": -10.0,
                    "delta_g": 6.0,
                    "emission_recompute_rate": 0.5,
                    "logp_floor": -50.0,
                    "g_logps_per_q": [-4.0] * 3,
                    "b_logps_per_q": [-10.0] * 3,
                    "g_argmax_marker_per_q": [True, False, False],
                    "b_argmax_marker_per_q": [False, False, False],
                }
                (real_per_cell / f"{cell}__default_assistant__marker_{persona}.json").write_text(
                    json.dumps(payload)
                )

    if with_onpolicy_switch:
        op_path = temp_dir / "eval_results" / "issue_464" / "onpolicy_validation.json"
        op_path.parent.mkdir(parents=True, exist_ok=True)
        op_path.write_text(
            json.dumps(
                {
                    "schema_version": "i464_onpolicy_validation_v1",
                    "switch_threshold": 1.5,
                    "n_q_per_persona": 16,
                    "per_cell": {},
                    "per_arm": {
                        "system_plain": {"n": 32, "mean": 0.10, "median": 0.10},
                        "system_padded": {"n": 32, "mean": 0.11, "median": 0.11},
                        "role": {"n": 32, "mean": 0.20, "median": 0.20},  # 2.0x system_plain
                    },
                    "role_over_system_plain_ratio": 2.0,
                    "switch_headline_to_trained_R": True,
                }
            )
        )


def _smoke_phase5(temp_dir: Path) -> tuple[int, str]:
    """Phase 5 analysis: REAL entrypoint end-to-end on a 3-seed x 3-arm stub tree.

    Exercises:
      - H1 elicitation gate (blocker #7) — owns cells loaded, gate PASSes.
      - H2 paired bootstrap with 3 complete seeds (blocker #3) — PASSes.
      - On-policy switch CONSUMED (blocker #4) — separate sub-check with
        switch=true and asserts headline status flips to blocked.
      - Per-seed dict + complete_seeds intersection (blocker #3
        --allow-partial bug) — verified by 3-seed PASS.
    """
    # Path A: full 3-seed tree, no on-policy switch -> H2 should PASS.
    _write_stub_per_cell_tree(temp_dir, include_3_seeds=True, with_onpolicy_switch=False)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "i464_phase5_analyze.py"),
        "--seeds",
        "42",
        "137",
        "1337",
    ]
    res = _run(cmd, cwd=temp_dir, timeout=60)
    if res.returncode != 0:
        return res.returncode, f"path-A stderr tail: {res.stderr[-300:]}"
    a = temp_dir / "eval_results" / "issue_464" / "analysis.json"
    if not a.exists():
        return 1, "analysis.json missing after path A"
    payload = json.loads(a.read_text())
    if payload.get("schema_version") != "i464_phase5_v2":
        return (
            1,
            f"path-A schema_version={payload.get('schema_version')!r}, expected i464_phase5_v2",
        )
    if payload.get("headline_status") != "ok":
        return 1, (
            f"path-A headline_status={payload.get('headline_status')!r}, expected 'ok'; "
            f"reason: {payload.get('headline', {}).get('reason')}"
        )
    if not payload.get("h1_elicitation", {}).get("overall_pass"):
        return (
            1,
            "path-A h1_elicitation.overall_pass="
            f"{payload.get('h1_elicitation', {}).get('overall_pass')}",
        )
    if payload.get("complete_seeds") != [42, 137, 1337]:
        return 1, f"path-A complete_seeds={payload.get('complete_seeds')}, expected [42, 137, 1337]"

    # Path B: 2-seed tree (H2_MIN_SEEDS=3 gate must trip; blocker #3 main).
    shutil.rmtree(temp_dir / "eval_results", ignore_errors=True)
    _write_stub_per_cell_tree(temp_dir, include_3_seeds=False, with_onpolicy_switch=False)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "i464_phase5_analyze.py"),
        "--seeds",
        "42",
    ]
    res_b = _run(cmd, cwd=temp_dir, timeout=60)
    if res_b.returncode != 0:
        return res_b.returncode, f"path-B stderr tail: {res_b.stderr[-300:]}"
    payload_b = json.loads(a.read_text())
    if payload_b.get("headline_status") != "inconclusive_descriptive_only":
        return 1, (
            f"path-B headline_status={payload_b.get('headline_status')!r}, "
            "expected 'inconclusive_descriptive_only' (n=1 < H2_MIN_SEEDS=3)"
        )

    # Path C: 3-seed tree + onpolicy switch=true (blocker #4 consumption).
    shutil.rmtree(temp_dir / "eval_results", ignore_errors=True)
    _write_stub_per_cell_tree(temp_dir, include_3_seeds=True, with_onpolicy_switch=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "i464_phase5_analyze.py"),
        "--seeds",
        "42",
        "137",
        "1337",
    ]
    res_c = _run(cmd, cwd=temp_dir, timeout=60)
    if res_c.returncode != 0:
        return res_c.returncode, f"path-C stderr tail: {res_c.stderr[-300:]}"
    payload_c = json.loads(a.read_text())
    if payload_c.get("headline_status") != "blocked_onpolicy_switch_required":
        return 1, (
            f"path-C headline_status={payload_c.get('headline_status')!r}, "
            "expected 'blocked_onpolicy_switch_required' (Phase 4.5 switch=true)"
        )
    if not payload_c.get("onpolicy_validation", {}).get("switch_headline_to_trained_R"):
        return 1, "path-C onpolicy_validation.switch_headline_to_trained_R should be true"

    # Restore path A for plot (need a happy analysis.json to plot from).
    shutil.rmtree(temp_dir / "eval_results", ignore_errors=True)
    _write_stub_per_cell_tree(temp_dir, include_3_seeds=True, with_onpolicy_switch=False)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "i464_phase5_analyze.py"),
        "--seeds",
        "42",
        "137",
        "1337",
    ]
    _run(cmd, cwd=temp_dir, timeout=60)

    return 0, (
        "real entrypoint exercised THREE paths: (A) 3 seeds no switch -> H2 PASS + H1 PASS; "
        "(B) 1 seed -> inconclusive_descriptive_only (MF-H gate); "
        "(C) 3 seeds + onpolicy switch=true -> blocked_onpolicy_switch_required "
        "(MF-B(2) consumption)"
    )


def _smoke_plot(temp_dir: Path) -> tuple[int, str]:
    """Plot script: REAL entrypoint on analysis.json from phase 5 path A."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "plot_i464_clean_result.py")]
    res = _run(cmd, cwd=temp_dir, timeout=120)
    if res.returncode != 0:
        return res.returncode, f"stderr tail: {res.stderr[-300:]}"
    fig_dir = temp_dir / "figures" / "issue_464"
    hero = fig_dir / "hero.png"
    if not hero.exists():
        return 1, f"hero.png missing at {hero}"
    expected = [
        "hero.png",
        "matrix_system_plain.png",
        "matrix_system_padded.png",
        "matrix_role.png",
        "per_seed.png",
        "raw_alongside_processed.png",
        "dynamic_range_check.png",
        "argmax_emission_system_plain.png",
        "argmax_emission_system_padded.png",
        "argmax_emission_role.png",
        "leakage_by_eval_encoding.png",
    ]
    missing = [name for name in expected if not (fig_dir / name).exists()]
    if missing:
        return 1, f"missing figures: {missing}"
    # Round-2 fix (blocker #5): the round-2 plot script registers trajectory
    # + onpolicy_validation plot functions; both gracefully skip when their
    # source is absent (smoke has neither). We just confirm the plot
    # script ran end-to-end without crashing — the warnings about skip
    # are expected in CPU smoke.
    return 0, (
        f"real entrypoint rc=0; wrote {len(expected)} figures incl. hero.png "
        f"(trajectory.png + onpolicy_validation.png skipped per design — no "
        f"WandB run / Phase 4.5 output in CPU smoke)"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 iff all phases pass."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temp dir after the run (default: rm -rf at exit).",
    )
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Use a per-run temp dir so multiple smoke runs don't collide.
    temp_dir = Path(tempfile.mkdtemp(prefix="i464_smoke_real_"))
    print(f"[smoke setup] temp_dir={temp_dir}")

    try:
        results: list[tuple[str, int]] = []
        for name, fn in [
            ("phase0_preflight", lambda: _smoke_phase0(temp_dir)),
            ("phase1_rgen", lambda: _smoke_phase1(temp_dir)),
            ("phase23_train", lambda: _smoke_phase23(temp_dir, tok)),
            ("phase2_smoke_check", lambda: _smoke_phase2_check(temp_dir)),
            ("phase4_eval", lambda: _smoke_phase4(temp_dir, tok)),
            ("phase45_onpolicy", lambda: _smoke_phase45(temp_dir)),
            ("phase5_analyze", lambda: _smoke_phase5(temp_dir)),
            ("plot_clean_result", lambda: _smoke_plot(temp_dir)),
        ]:
            try:
                rc, digest = fn()
            except Exception as e:
                rc, digest = 1, f"crashed: {type(e).__name__}: {e}"
            _print_phase(name, rc, digest)
            results.append((name, rc))

        failed = [n for n, rc in results if rc != 0]
        if failed:
            print(f"\n[smoke summary] {len(failed)} phase(s) FAILED: {failed}")
            return 1
        print(f"\n[smoke summary] all {len(results)} phases OK (temp_dir={temp_dir})")
        return 0
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
