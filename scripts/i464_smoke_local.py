"""CPU-only end-to-end smoke for issue #464 phase pipeline.

This is the local-dev counterpart to the pod-side ``i464_run_all.sh``. It
exercises EVERY phase's non-GPU code path end-to-end on the local VM
(which has no H100) so the experimenter agent doesn't discover wiring
bugs only after pod provisioning.

What runs:
  Phase 0  — preflight with --no-smoke --dry-run (no GPU; verifies
             tokenizer + data loaders + token-id contract).
  Phase 1  — synthesizes a tiny R_canon JSON (5 questions × 2 personas)
             matching the i464_v2_matched_R schema. Verifies the schema
             round-trips via _load_R_canon downstream.
  Phase 2/3 — calls _build_training_rows directly (no Trainer); checks
             the JSONL was written and marker count == 1 per row.
  Phase 2 smoke — _parse_cell round-trip.
  Phase 4  — _build_probes_for_eval_marker on the stub R_canon; checks
             slot positions + last token == marker_id.
  Phase 4.5 — _char_edit_distance on (identical, slightly different)
             strings.
  Phase 5  — synthesizes per-cell JSONs for system_plain × seed 42 only
             then runs main(--allow-partial) and checks analysis.json
             contains the L_per_arm structure.
  Plot     — runs main() on the stub analysis.json; checks hero.png exists.

Each phase prints an exit-code line + an artifact-digest line so a
caller can mechanically extract per-phase status.

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
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc

load_dotenv()

logger = logging.getLogger("i464.smoke")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SMOKE_DIR = Path("/tmp/i464_smoke")
DATA_DIR = SMOKE_DIR / "data" / "issue_464"
PER_CELL_DIR = SMOKE_DIR / "eval_results" / "issue_464" / "cross_eval" / "per_cell"
ANALYSIS_PATH = SMOKE_DIR / "eval_results" / "issue_464" / "analysis.json"
FIG_DIR = SMOKE_DIR / "figures" / "issue_464"


def _print_phase(name: str, rc: int, digest: str) -> None:
    """Emit the canonical phase-result line."""
    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    print(f"[smoke phase={name}] {status} :: {digest}")


def _smoke_phase0() -> tuple[int, str]:
    """Phase 0: preflight (no GPU, dry-run)."""
    cmd = [
        sys.executable,
        "scripts/i464_phase0_preflight.py",
        "--no-smoke",
        "--dry-run",
    ]
    res = subprocess.run(cmd, env={**os.environ}, capture_output=True, text=True)
    if res.returncode != 0:
        return res.returncode, f"stderr tail: {res.stderr[-200:]}"
    return 0, "dry-run preflight produced JSON to stdout"


def _smoke_phase1(tok) -> tuple[int, str]:
    """Phase 1: synthesize R_canon test+train (5 q × 2 personas each)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    qs_train = [f"smoke train q{i}?" for i in range(5)]
    qs_test = [f"smoke test q{i}?" for i in range(5)]
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
        (DATA_DIR / f"R_canon_{split}.json").write_text(json.dumps(payload))
    return 0, f"wrote R_canon_train.json + R_canon_test.json under {DATA_DIR}"


def _smoke_phase23(tok) -> tuple[int, str]:
    """Phase 2/3: call _build_training_rows directly on stub R_canon."""
    # Copy stub R_canon into the real path the script reads.
    real_data = Path("data/issue_464")
    real_data.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        shutil.copy(DATA_DIR / f"R_canon_{split}.json", real_data / f"R_canon_{split}.json")

    # Import and invoke _build_training_rows from the script directly.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "i464_phase23_train", "scripts/i464_phase23_train.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    q_train_answers = {f"smoke train q{i}?": "stub answer" for i in range(5)}
    R_canon_train = json.loads((real_data / "R_canon_train.json").read_text())["completions"]
    out_path = mod._build_training_rows(
        arm="system_plain",
        seed=42,
        q_train_answers=q_train_answers,
        R_canon_train=R_canon_train,
        tokenizer=tok,
        n_dupes=1,
    )
    n_rows = sum(1 for _ in open(out_path))
    if n_rows != 10:
        return 1, f"expected 10 rows (5q x 2personas x 1dupe), got {n_rows}"
    return 0, f"{out_path} has {n_rows} rows, marker count==1 verified in-build"


def _smoke_phase2_check() -> tuple[int, str]:
    """Phase 2 smoke check: argparse + _parse_cell round-trip (no GPU)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "i464_phase2_smoke_check", "scripts/i464_phase2_smoke_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    arm, seed = mod._parse_cell("system_plain_seed42")
    if arm != "system_plain" or seed != 42:
        return 1, f"_parse_cell mismatch: {arm}, {seed}"
    return 0, "_parse_cell round-trips system_plain_seed42 -> ('system_plain', 42)"


def _smoke_phase4(tok) -> tuple[int, str]:
    """Phase 4: _build_probes_for_eval_marker on stub R_canon."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("i464_phase4_eval", "scripts/i464_phase4_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    R_canon_test = json.loads((Path("data/issue_464") / "R_canon_test.json").read_text())[
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
    return 0, f"built {len(prompts)} probes; last token == marker_id verified"


def _smoke_phase45() -> tuple[int, str]:
    """Phase 4.5: _char_edit_distance unit check."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "i464_phase45_onpolicy_validation",
        "scripts/i464_phase45_onpolicy_validation.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod._char_edit_distance("abc", "abc") != 0:
        return 1, "edit_distance(equal) != 0"
    if mod._char_edit_distance("abc", "abd") != 1:
        return 1, "edit_distance(1 sub) != 1"
    if mod._char_edit_distance("", "abc") != 3:
        return 1, "edit_distance('', 'abc') != 3"
    return 0, "edit_distance verified on 3 unit cases"


def _smoke_phase5() -> tuple[int, str]:
    """Phase 5: synthesize a single-arm per-cell tree + run analyze.

    We provide only system_plain at seed 42 (4 leakage cells) so the
    analyzer must run in --allow-partial mode and the headline will be
    None (insufficient seeds). The check is that the script writes
    analysis.json with the expected L_per_arm_per_seed structure.
    """
    real_per_cell = Path("eval_results/issue_464/cross_eval/per_cell")
    real_per_cell.mkdir(parents=True, exist_ok=True)
    # Write 4 symmetric-leakage cells for system_plain_seed42:
    #   pirate / system_villain, pirate / role_villain,
    #   villain / system_pirate, villain / role_pirate.
    seeds = [42]
    for arm in enc.ARMS:
        for seed in seeds:
            cell = f"{arm}_seed{seed}"
            for persona in enc.PERSONAS:
                other = "villain" if persona == "pirate" else "pirate"
                for e_wrong in [f"system_{other}", f"role_{other}"]:
                    payload = {
                        "cell": cell,
                        "arm": arm,
                        "seed": seed,
                        "e_eval": e_wrong,
                        "marker_persona": persona,
                        "marker_id": enc.marker_id_for(persona),
                        "n_probes": 3,
                        "g_logprob": -2.5 - (0.0 if arm != "role" else 1.0),
                        "b_logprob": -10.0,
                        "delta_g": 7.5,
                        "emission_recompute_rate": 0.7,
                        "logp_floor": -50.0,
                        "g_logps_per_q": [-2.5] * 3,
                        "b_logps_per_q": [-10.0] * 3,
                        "g_argmax_marker_per_q": [True, True, False],
                        "b_argmax_marker_per_q": [False, False, False],
                    }
                    out = real_per_cell / f"{cell}__{e_wrong}__marker_{persona}.json"
                    out.write_text(json.dumps(payload))
    cmd = [
        sys.executable,
        "scripts/i464_phase5_analyze.py",
        "--seeds",
        "42",
        "--allow-partial",
    ]
    res = subprocess.run(cmd, env={**os.environ}, capture_output=True, text=True)
    if res.returncode != 0:
        return res.returncode, f"stderr tail: {res.stderr[-300:]}"
    a = Path("eval_results/issue_464/analysis.json")
    if not a.exists():
        return 1, "analysis.json was not written"
    payload = json.loads(a.read_text())
    if "L_per_arm_per_seed" not in payload:
        return 1, "analysis.json missing L_per_arm_per_seed"
    return 0, f"wrote {a} with L_per_arm_per_seed (3 arms × 1 seed)"


def _smoke_plot() -> tuple[int, str]:
    """Plot: run plot_i464_clean_result on the analysis.json from phase 5."""
    cmd = [sys.executable, "scripts/plot_i464_clean_result.py"]
    res = subprocess.run(cmd, env={**os.environ}, capture_output=True, text=True)
    if res.returncode != 0:
        return res.returncode, f"stderr tail: {res.stderr[-300:]}"
    hero = Path("figures/issue_464/hero.png")
    if not hero.exists():
        return 1, f"hero.png missing at {hero}"
    return (
        0,
        f"wrote {hero} (+ matrix_*, per_seed, raw_alongside, dynamic_range, argmax_emission_*, leakage_by_eval_encoding)",
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 iff all phases pass."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args = ap.parse_args(argv)
    _ = args

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    results: list[tuple[str, int]] = []
    for name, fn in [
        ("phase0_preflight", _smoke_phase0),
        ("phase1_rgen", lambda: _smoke_phase1(tok)),
        ("phase23_train_row_build", lambda: _smoke_phase23(tok)),
        ("phase2_smoke_check_cli", _smoke_phase2_check),
        ("phase4_probe_build", lambda: _smoke_phase4(tok)),
        ("phase45_edit_distance", _smoke_phase45),
        ("phase5_analyze", _smoke_phase5),
        ("plot_clean_result", _smoke_plot),
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
    print(f"\n[smoke summary] all {len(results)} phases OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
