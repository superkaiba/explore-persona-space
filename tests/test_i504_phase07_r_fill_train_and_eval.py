# ruff: noqa: RUF002  # em-dash + Qwen marker " ※" intentional
"""Round-11 regression tests for the Phase 0.7 r-fill (task #504).

Pins the symmetric train + eval coverage that the v8 code-review missed:

* **Train side (round-8 contract).** Phase 0.7 must diff Phase 0.5's
  ``arm_to_positioned_n`` ∪ ``smoke_mid_band_n`` ∪ ``default_persona`` ∪
  ``source`` against R_train's keys and identify the missing personas.
* **Eval side (round-11 contract).** Phase 0.7 must diff Phase 0.5's
  ``held_out_panel`` ∪ ``source`` against R_eval's keys and identify the
  missing personas.

The vLLM-generate path is GPU-only (Qwen-2.5-7B); these tests exercise the
diff/no-op/copy/sentinel scaffolding via the script's importable helpers and
end-to-end via subprocess when both sides have no missing personas.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FILL_SCRIPT = REPO_ROOT / "scripts" / "i504_phase_r_generate_fill.py"


def _load_fill_module():
    """Load scripts/i504_phase_r_generate_fill.py as an importable module.

    The fill script is in scripts/ which isn't on sys.path; load via spec so
    the test can unit-test ``_read_train_needed`` / ``_read_eval_needed``.
    """
    spec = importlib.util.spec_from_file_location("i504_phase_r_generate_fill", FILL_SCRIPT)
    assert spec is not None and spec.loader is not None, FILL_SCRIPT
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fill_mod():
    return _load_fill_module()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: synthetic phase0_5_gates.json shape that matches what phase05.py emits.
# ─────────────────────────────────────────────────────────────────────────────


def _make_phase05_report(
    *,
    source: str,
    default: str,
    positioned_ns: dict[str, str],
    smoke_mid_band_n: str | None,
    panel: list[str],
) -> dict:
    return {
        "source": source,
        "default_persona": default,
        "arm_to_positioned_n": positioned_ns,
        "smoke_mid_band_n": smoke_mid_band_n,
        "held_out_panel": panel,
        "chosen_negatives": {"default": default, **positioned_ns},
    }


def _make_r_artifact(personas: list[str], n_questions: int = 5) -> dict:
    """Synthetic R artifact mirroring i472_v1 schema."""
    qs = [f"q_{i}" for i in range(n_questions)]
    completions = {
        p: {
            q: {
                "response_text": f"Answer from {p} to {q}.",
                "response_token_ids": list(range(100, 100 + 20)),
                "n_response_tokens": 20,
                "ended_with_eos": True,
                "truncated": False,
                "marker_in_R": False,
            }
            for q in qs
        }
        for p in personas
    }
    return {
        "schema_version": "i472_v1",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "completions": completions,
        "questions": qs,
        "personas": sorted(completions.keys()),
        "n_personas": len(completions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — diff helpers (the round-8 + round-11 contracts).
# ─────────────────────────────────────────────────────────────────────────────


def test_read_train_needed_includes_source_default_and_positioned_negs(fill_mod):
    """TRAIN diff must include positioned negs, smoke mid-band, default, AND source."""
    report = _make_phase05_report(
        source="villain",
        default="qwen_default",
        positioned_ns={
            "c504_near": "near_persona",
            "c504_mid_near": "mid_near_persona",
            "c504_mid_far": "mid_far_persona",
            "c504_far": "far_persona",
        },
        smoke_mid_band_n="smoke_mid_persona",
        panel=["probe_a", "probe_b", "probe_c"],
    )
    # R_train covers only the source — every cell-side negative must be reported missing.
    r_keys = {"villain"}
    missing = fill_mod._read_train_needed(report, r_keys)
    assert set(missing) == {
        "near_persona",
        "mid_near_persona",
        "mid_far_persona",
        "far_persona",
        "smoke_mid_persona",
        "qwen_default",
    }, missing
    # Source is covered → not in missing.
    assert "villain" not in missing


def test_read_train_needed_empty_when_all_covered(fill_mod):
    """No-op TRAIN branch: all train-side personas already in R_train."""
    report = _make_phase05_report(
        source="villain",
        default="qwen_default",
        positioned_ns={"c504_near": "near_persona"},
        smoke_mid_band_n=None,
        panel=["probe_a"],
    )
    r_keys = {"villain", "qwen_default", "near_persona", "probe_a"}
    missing = fill_mod._read_train_needed(report, r_keys)
    assert missing == []


def test_read_eval_needed_is_panel_plus_source(fill_mod):
    """EVAL diff must include EVERY panel persona + source.

    This is the round-11 contract that v8 code-review missed: panel ⊆ bank is
    NOT the same as panel ⊆ R_eval.keys(). The eval rig will probe every panel
    persona at every checkpoint; #472's R_eval covers only its own evaluation
    subset.
    """
    report = _make_phase05_report(
        source="villain",
        default="qwen_default",
        positioned_ns={"c504_near": "near_persona"},
        smoke_mid_band_n=None,
        panel=["architect", "barista", "chef", "doctor", "engineer"],
    )
    # R_eval covers the source + 2 of the 5 panel personas.
    r_keys = {"villain", "architect", "barista"}
    missing = fill_mod._read_eval_needed(report, r_keys)
    assert set(missing) == {"chef", "doctor", "engineer"}, missing
    # Sanity: positioned negatives are NOT in the eval-needed set (they are
    # held-IN of the negatives, never in the panel).
    assert "near_persona" not in missing
    assert "qwen_default" not in missing


def test_read_eval_needed_empty_when_all_panel_covered(fill_mod):
    """No-op EVAL branch: every panel persona + source already in R_eval."""
    report = _make_phase05_report(
        source="villain",
        default="qwen_default",
        positioned_ns={"c504_near": "near_persona"},
        smoke_mid_band_n=None,
        panel=["architect", "barista"],
    )
    r_keys = {"villain", "architect", "barista"}
    missing = fill_mod._read_eval_needed(report, r_keys)
    assert missing == []


def test_read_eval_needed_panel_persona_missing_is_caught(fill_mod):
    """The exact failure mode from round-10: panel persona absent from R_eval.

    The round-10 crash was ``KeyError: "R_eval missing persona 'architect'"`` at
    eval_trajectory.py:165. This test pins that Phase 0.7's eval diff would
    surface 'architect' as needing on-policy generation BEFORE the trajectory
    rig runs.
    """
    report = _make_phase05_report(
        source="villain",
        default="qwen_default",
        positioned_ns={"c504_near": "near_persona"},
        smoke_mid_band_n=None,
        panel=["architect", "barista", "chef"],
    )
    # R_eval covers source + barista + chef — architect is the round-10 victim.
    r_keys = {"villain", "barista", "chef"}
    missing = fill_mod._read_eval_needed(report, r_keys)
    assert missing == ["architect"], missing


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: subprocess invocation exercises BOTH sides via the no-op path
# (vLLM-generate is GPU-only and not smokeable here).
# ─────────────────────────────────────────────────────────────────────────────


def _write_artifacts(
    tmp_path: Path,
    *,
    phase05_payload: dict,
    train_personas: list[str],
    eval_personas: list[str],
) -> tuple[Path, Path, Path]:
    phase05_path = tmp_path / "phase0_5_gates.json"
    phase05_path.write_text(json.dumps(phase05_payload, indent=2))
    train_path = tmp_path / "R_train.json"
    train_path.write_text(json.dumps(_make_r_artifact(train_personas), indent=2))
    eval_path = tmp_path / "R_eval.json"
    eval_path.write_text(json.dumps(_make_r_artifact(eval_personas), indent=2))
    return phase05_path, train_path, eval_path


def _write_synthetic_bank(tmp_path: Path, personas: list[str]) -> Path:
    bank_path = tmp_path / "persona_bank.json"
    bank_path.write_text(
        json.dumps(
            {
                "schema_version": "i472_v1",
                "base_model": "synthetic-test",
                "personas": {p: f"You are {p}." for p in personas},
            },
            indent=2,
        )
    )
    return bank_path


def _run_fill(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", "python", str(FILL_SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _read_sentinel_note(sentinel_path: Path) -> dict:
    sentinel = json.loads(sentinel_path.read_text())
    return json.loads(sentinel["note"])


def test_e2e_both_sides_noop_train_and_eval_covered(tmp_path: Path) -> None:
    """End-to-end: both sides covered → both v504 paths materialized as copies.

    Pins the no-op contract: dispatcher repoints downstream args at the v504
    paths regardless of whether anything was filled (round-8 invariant), and
    the sentinel reports status=ok_noop on BOTH sides.
    """
    panel = ["architect", "barista", "chef"]
    positioned = {"c504_near": "near_persona"}
    bank_personas = ["villain", "qwen_default", "near_persona", *panel]
    phase05_payload = _make_phase05_report(
        source="villain",
        default="qwen_default",
        positioned_ns=positioned,
        smoke_mid_band_n=None,
        panel=panel,
    )
    phase05_path, train_path, eval_path = _write_artifacts(
        tmp_path,
        phase05_payload=phase05_payload,
        train_personas=["villain", "qwen_default", "near_persona"],
        eval_personas=["villain", *panel],
    )
    bank_path = _write_synthetic_bank(tmp_path, bank_personas)
    out_train = tmp_path / "R_train_v504.json"
    out_eval = tmp_path / "R_eval_v504.json"
    sentinel_path = tmp_path / "sentinel.json"

    rc = _run_fill(
        [
            "--phase05-path",
            str(phase05_path),
            "--split",
            "both",
            "--input-r-train-path",
            str(train_path),
            "--output-r-train-path",
            str(out_train),
            "--input-r-eval-path",
            str(eval_path),
            "--output-r-eval-path",
            str(out_eval),
            "--bank-path",
            str(bank_path),
            "--no-upload",
            "--sentinel-path",
            str(sentinel_path),
        ]
    )
    assert rc.returncode == 0, (rc.returncode, rc.stdout[-2000:], rc.stderr[-2000:])
    assert out_train.exists()
    assert out_eval.exists()

    note = _read_sentinel_note(sentinel_path)
    assert note["status"] == "ok_noop", note
    assert note["split"] == "both"
    assert note["train_missing"] == []
    assert note["eval_missing"] == []
    assert note["n_train_input_personas"] == 3
    assert note["n_eval_input_personas"] == 4

    # Byte-identical-modulo-formatting: the v504 copy holds the same completions.
    train_in = json.loads(train_path.read_text())["completions"]
    train_out = json.loads(out_train.read_text())["completions"]
    assert set(train_in.keys()) == set(train_out.keys())
    eval_in = json.loads(eval_path.read_text())["completions"]
    eval_out = json.loads(out_eval.read_text())["completions"]
    assert set(eval_in.keys()) == set(eval_out.keys())


def test_e2e_split_train_only_does_not_write_eval(tmp_path: Path) -> None:
    """``--split train`` exercises ONLY the train side (round-8 isolation)."""
    panel = ["architect"]
    bank_personas = ["villain", "qwen_default", "near_persona", *panel]
    phase05_payload = _make_phase05_report(
        source="villain",
        default="qwen_default",
        positioned_ns={"c504_near": "near_persona"},
        smoke_mid_band_n=None,
        panel=panel,
    )
    phase05_path, train_path, eval_path = _write_artifacts(
        tmp_path,
        phase05_payload=phase05_payload,
        train_personas=["villain", "qwen_default", "near_persona"],
        eval_personas=["villain"],
    )
    bank_path = _write_synthetic_bank(tmp_path, bank_personas)
    out_train = tmp_path / "R_train_v504.json"
    out_eval = tmp_path / "R_eval_v504.json"
    sentinel_path = tmp_path / "sentinel.json"

    rc = _run_fill(
        [
            "--phase05-path",
            str(phase05_path),
            "--split",
            "train",
            "--input-r-train-path",
            str(train_path),
            "--output-r-train-path",
            str(out_train),
            "--input-r-eval-path",
            str(eval_path),
            "--output-r-eval-path",
            str(out_eval),
            "--bank-path",
            str(bank_path),
            "--no-upload",
            "--sentinel-path",
            str(sentinel_path),
        ]
    )
    assert rc.returncode == 0, (rc.returncode, rc.stdout[-2000:], rc.stderr[-2000:])
    assert out_train.exists()
    # eval output is NOT touched in --split train mode.
    assert not out_eval.exists()

    note = _read_sentinel_note(sentinel_path)
    assert note["status"] == "ok_noop"
    assert note["split"] == "train"
    assert note["train_output_path"] == str(out_train)
    assert note["eval_output_path"] is None


def test_e2e_split_eval_only_does_not_write_train(tmp_path: Path) -> None:
    """``--split eval`` exercises ONLY the eval side (round-11 isolation)."""
    panel = ["architect", "barista"]
    bank_personas = ["villain", "qwen_default", "near_persona", *panel]
    phase05_payload = _make_phase05_report(
        source="villain",
        default="qwen_default",
        positioned_ns={"c504_near": "near_persona"},
        smoke_mid_band_n=None,
        panel=panel,
    )
    phase05_path, train_path, eval_path = _write_artifacts(
        tmp_path,
        phase05_payload=phase05_payload,
        train_personas=["villain"],
        eval_personas=["villain", *panel],
    )
    bank_path = _write_synthetic_bank(tmp_path, bank_personas)
    out_train = tmp_path / "R_train_v504.json"
    out_eval = tmp_path / "R_eval_v504.json"
    sentinel_path = tmp_path / "sentinel.json"

    rc = _run_fill(
        [
            "--phase05-path",
            str(phase05_path),
            "--split",
            "eval",
            "--input-r-train-path",
            str(train_path),
            "--output-r-train-path",
            str(out_train),
            "--input-r-eval-path",
            str(eval_path),
            "--output-r-eval-path",
            str(out_eval),
            "--bank-path",
            str(bank_path),
            "--no-upload",
            "--sentinel-path",
            str(sentinel_path),
        ]
    )
    assert rc.returncode == 0, (rc.returncode, rc.stdout[-2000:], rc.stderr[-2000:])
    assert out_eval.exists()
    # train output is NOT touched in --split eval mode.
    assert not out_train.exists()

    note = _read_sentinel_note(sentinel_path)
    assert note["status"] == "ok_noop"
    assert note["split"] == "eval"
    assert note["eval_output_path"] == str(out_eval)
    assert note["train_output_path"] is None


def test_dispatcher_passes_r_eval_path_to_run_cell() -> None:
    """Dispatcher (``dispatch_neg_geometry_504.py``) MUST thread --r-eval-path
    into the i504_run_cell.py subprocess command.

    Round-10 root cause: the dispatcher had `args.r_eval_path` but did not
    repoint it after Phase 0.7 (or pass it through the pool). Round-11 fix
    threads it through ``_schedule_cell_pool`` so downstream cells read the
    augmented v504 R_eval.
    """
    dispatcher_src = (REPO_ROOT / "scripts" / "dispatch_neg_geometry_504.py").read_text()
    # _schedule_cell_pool must accept r_eval_path and forward it.
    assert "r_eval_path: Path," in dispatcher_src, (
        "_schedule_cell_pool MUST accept r_eval_path argument (round-11)."
    )
    assert '"--r-eval-path",' in dispatcher_src, (
        "Dispatcher MUST thread --r-eval-path into i504_run_cell.py command."
    )
    # The CLI must expose --r-eval-path so callers can pre-stage / override.
    assert '"--r-eval-path"' in dispatcher_src, "Dispatcher MUST expose --r-eval-path CLI argument."
    # The Phase 0.7 invocation must use --split both (default) AND pass eval args.
    assert "--input-r-eval-path" in dispatcher_src
    assert "--output-r-eval-path" in dispatcher_src
    # Repoint must happen on BOTH train and eval paths.
    assert "args.r_train_path = r_train_v504_path" in dispatcher_src
    assert "args.r_eval_path = r_eval_v504_path" in dispatcher_src


def test_fill_script_help_advertises_split_arg() -> None:
    """``--split`` arg is the public knob; --help must list it."""
    rc = subprocess.run(
        ["uv", "run", "python", str(FILL_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, (rc.returncode, rc.stdout, rc.stderr)
    assert "--split" in rc.stdout
    assert "--input-r-eval-path" in rc.stdout
    assert "--output-r-eval-path" in rc.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
