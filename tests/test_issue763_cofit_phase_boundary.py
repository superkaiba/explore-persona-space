"""Issue #763 `neutral-contrast-and-cofit` round-2 phase-boundary handoff pins.

Pins the review-r1 Critical fixes (Claude C1/C2 + Codex C1/C2 — cross-machine
phase inputs neither uploaded nor staged; mock-artifact hazard at canonical
paths) so a future refactor cannot silently strip them:

- STATIC source pins (no torch/HF import): every artifact a phase reads after
  a machine boundary is either produced in-phase or staged from HF before its
  first local read; the Phase-A progress upload carries the capture manifest;
  the Phase-B directions upload carries neutral_arm_manifest; the plot fails
  loud (never warn-and-skips) on the plan-required manifest; the dispatcher
  arms the smoke scope + runs the residue check.
- FUNCTIONAL pins (light ``issue763_common`` import): the smoke-scope path
  redirect, the env/--smoke consistency contract, and the absolute
  pool-floor / production-dim validators (fail pre-fix, pass post-fix).
"""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _norm(src: str) -> str:
    """Collapse whitespace so formatter line-wrapping cannot break a pin."""
    return re.sub(r"\s+", " ", src)


def _block(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


# ── static pins: upload manifest handoff (C1(i) / Codex C2) ───────────────────


def test_progress_upload_includes_capture_manifest():
    src = (SCRIPTS / "issue763_cofit_upload.py").read_text()
    progress = _block(src, "_PROGRESS_ARTIFACTS", "_DIRECTIONS_ARTIFACTS")
    assert "capture_arm_means_manifest.json" in progress, (
        "the Phase-A --progress-only pass must upload the capture manifest (the "
        "parity record Phase B refuses to run without) — review r1 C1(i)"
    )


def test_directions_upload_includes_neutral_arm_manifest():
    src = (SCRIPTS / "issue763_cofit_upload.py").read_text()
    directions = _block(src, "_DIRECTIONS_ARTIFACTS", "_FINAL_ARTIFACTS")
    assert "neutral_arm_manifest.json" in directions, (
        "the Phase-B --directions-only pass must upload neutral_arm_manifest.json "
        "(plan-required §6.5 deliverable) — review r1 C2 / Codex C2"
    )


def test_final_deliverable_check_includes_neutral_arm_manifest():
    src = _norm((SCRIPTS / "issue763_cofit_upload.py").read_text())
    assert 'neutral_arm = COFIT_DIR / "neutral_arm_manifest.json"' in src
    assert "(results, nonlinear, manifest, neutral_arm)" in src, (
        "the final upload's missing-deliverable existence check must include "
        "neutral_arm_manifest.json"
    )


# ── static pins: phase-boundary staging before first local read (C1/C2) ──────


def test_assemble_directions_stages_every_cross_boundary_input():
    src = (SCRIPTS / "issue763_extract_pv_rb.py").read_text()
    seg = _norm(_block(src, "def _phase_assemble_directions", "def main"))
    first_read = seg.index("load_json(_capture_manifest_path())")
    staging = seg[:first_read]
    for pin, why in [
        ("_stage_single_from_hf(", "capture manifest staging call"),
        ("capture_arm_means_manifest.json", "capture manifest HF path"),
        ('_stage_from_hf("arm_means"', "arm_means staging"),
        ('_stage_from_hf( "neutral_rollout_means"', "neutral_rollout_means staging"),
        ('_stage_from_hf("neutral_judge"', "neutral_judge staging"),
        ('_stage_from_hf("pv_rollouts"', "pv_rollouts (question recovery) staging"),
        ('filename_prefix="rb_"', "rb-shard staging (cos integrity read)"),
    ]:
        assert pin.replace("( ", "(") in staging.replace("( ", "("), (
            f"_phase_assemble_directions must stage {why} BEFORE its first local "
            "read (review r1 C1(ii) / Codex C1)"
        )


def test_stage_round_inputs_stages_manifest_frozen_inputs():
    src = (SCRIPTS / "issue763_cofit_predictors.py").read_text()
    seg = _norm(_block(src, "def _stage_round_inputs", "def write_inputs_manifest"))
    assert '_stage_from_hf("pv_rollouts"' in seg, (
        "Phase C must stage pv_rollouts (gitignored data/) before "
        "write_inputs_manifest — review r1 C2"
    )
    assert '_stage_from_hf("pv_judge_v2"' in seg, (
        "Phase C must stage pv_judge_v2 before write_inputs_manifest — review r1 C2"
    )
    assert "assert_pool_floor(" in seg, (
        "staged manifest inputs must be validated against the ABSOLUTE "
        "plan-registered pool size (review r1 C1(iii))"
    )


def test_plot_fails_loud_on_missing_neutral_arm_manifest():
    src = (SCRIPTS / "issue763_cofit_plot.py").read_text()
    assert "cos/yield panels skipped" not in src, (
        "the plot must never warn-and-skip the plan-required "
        "neutral_arm_manifest.json (review r1 C2 / Codex C2)"
    )
    seg = _norm(_block(src, "def main", "if __name__"))
    assert "_stage_single_from_hf(" in seg
    assert "raise RuntimeError(" in seg


def test_dispatcher_arms_smoke_scope_and_residue_check():
    src = (SCRIPTS / "issue763_cofit_dispatch.sh").read_text()
    assert "export EPM_ISSUE763_SMOKE_SCOPE=1" in src
    assert "unset EPM_ISSUE763_SMOKE_SCOPE" in src
    assert "[phase=residue_check]" in src, (
        "the smoke must end with the canonical-path residue check (review r1 C1(iii))"
    )


# ── functional pins: smoke scope + absolute validators ───────────────────────


@pytest.fixture()
def common(monkeypatch):
    """issue763_common with a CLEAN scope env; reloaded clean again at teardown."""
    monkeypatch.syspath_prepend(str(REPO / "src"))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.delenv("EPM_ISSUE763_SMOKE_SCOPE", raising=False)
    import issue763_common

    mod = importlib.reload(issue763_common)
    yield mod
    # teardown: whatever a test set, leave the module bound to CANONICAL paths
    import os

    os.environ.pop("EPM_ISSUE763_SMOKE_SCOPE", None)
    importlib.reload(sys.modules["issue763_common"])


def test_smoke_scoped_is_identity_without_env(common):
    p = Path("/a/issue_763/pv_shards")
    assert common.smoke_scoped(p) == p


def test_smoke_scoped_redirects_under_env(common, monkeypatch):
    monkeypatch.setenv("EPM_ISSUE763_SMOKE_SCOPE", "1")
    p = Path("/a/issue_763/pv_shards")
    assert common.smoke_scoped(p) == Path("/a/issue_763/smoke_scope/pv_shards")


def test_module_constants_rebind_under_env(monkeypatch, common):
    monkeypatch.setenv("EPM_ISSUE763_SMOKE_SCOPE", "1")
    scoped = importlib.reload(sys.modules["issue763_common"])
    assert scoped.COFIT_DIR.parent.name == "smoke_scope"
    assert scoped.NEUTRAL_ROLLOUT_DIR.parent.name == "smoke_scope"


def test_ensure_smoke_scope_env_without_smoke_raises(common, monkeypatch):
    monkeypatch.setenv("EPM_ISSUE763_SMOKE_SCOPE", "1")
    with pytest.raises(RuntimeError, match="smoke scope is smoke-only"):
        common.ensure_smoke_scope(False)


def test_ensure_smoke_scope_reexecs_on_smoke_without_env(common, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(common.os, "execvpe", lambda *a: calls.append(a))
    common.ensure_smoke_scope(True)
    assert len(calls) == 1, "ensure_smoke_scope(True) must re-exec with the scope env armed"
    _exe, argv, env = calls[0]
    assert env.get("EPM_ISSUE763_SMOKE_SCOPE") == "1"
    assert argv[0] == sys.executable


def test_assert_pool_floor_raises_below_plan_floor(common):
    # a 4-row mock pool at a canonical path (the review-r1 live hazard) fails loud
    with pytest.raises(RuntimeError, match="smoke/mock residue"):
        common.assert_pool_floor(4, common.NEUTRAL_POOL_EXPECTED, "neutral_rollouts")
    with pytest.raises(RuntimeError):
        common.assert_pool_floor(999, 1000, "pv_rollouts")
    common.assert_pool_floor(1000, 1000, "pv_rollouts")  # at floor: passes


def test_assert_production_direction_shape(common):
    # the (2, 8) tiny-smoke-model mock (review r1 C1(iii) live evidence) fails loud
    with pytest.raises(RuntimeError, match="smoke/mock residue"):
        common.assert_production_direction_shape((2, 8), "rb_deception.pt")
    common.assert_production_direction_shape((28, 3584), "rb_deception.pt")


# ── static pin: manifest frozen inputs subset-of (staged OR git-tracked) ─────
#
# Production crash (crash-fix r3, pod-763): ``write_inputs_manifest`` (fit-start
# step 0) fail-louds on ANY missing frozen input, but the v0 shards staged
# LAZILY via ``_load_v0`` at battery time — AFTER the manifest — so a fresh lane
# (no local worktree copy) crashed at step 0 on v0_deception.pt. The pin: every
# ``_add()`` in ``write_inputs_manifest`` must be covered by a staging call that
# runs inside ``_stage_round_inputs`` (i.e. BEFORE the manifest) or be a
# git-tracked path (present after a bare clone). A NEW ``_add`` without a
# coverage row fails this test until its staging leg exists.

_STAGED = "staged"
_GIT = "git-tracked"

# ast.unparse'd first ``_add`` arg -> (kind, pin). ``staged``: substring that
# must appear in the ``_stage_round_inputs`` body; ``git-tracked``: repo-
# relative path that must be in the git index (a fresh-lane clone carries it).
MANIFEST_INPUT_COVERAGE: dict[str, tuple[str, str]] = {
    "PV_ROLLOUT_DIR / f'{b}.jsonl'": (_STAGED, '_stage_from_hf("pv_rollouts"'),
    "PV_JUDGE_V2_DIR / f'{b}.json'": (_STAGED, '_stage_from_hf("pv_judge_v2"'),
    # THE r3 CRASH ROW: v0 shards must stage EAGERLY here, not lazily in _load_v0.
    "EVAL_RESULTS_DIR / 'v0_shards' / f'v0_{b}.pt'": (_STAGED, "_stage_v0_shards_from_hf("),
    "PV_SHARD_DIR / f'rb_{b}.pt'": (
        _STAGED,
        "_stage_fit_inputs_from_hf(behaviors, stage_e0=True)",
    ),
    "PV_DIRECTIONS_V2_DIR / f'{b}.pt'": (_STAGED, 'PV_DIRECTIONS_V2_DIR / f"{b}.pt"'),
    "C0_SHARD_DIR / f'c0_{b}.pt'": (_STAGED, 'C0_SHARD_DIR / f"c0_{b}.pt"'),
    "E0_PARENT_PATH": (_STAGED, "stage_e0=True"),
    "E0_DECEPTION_V2_PATH": (
        _GIT,
        "eval_results/issue_763/deception-rubric-reanchor/E0_deception_v2.json",
    ),
    "PARENT_RESULTS_PATH": (
        _GIT,
        "eval_results/issue_763/deception-rubric-reanchor/matched_predictor_results.json",
    ),
}


def _manifest_add_args() -> list[str]:
    """ast.unparse'd first argument of every ``_add(...)`` in write_inputs_manifest."""
    tree = ast.parse((SCRIPTS / "issue763_cofit_predictors.py").read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "write_inputs_manifest"
    )
    return [
        ast.unparse(call.args[0])
        for call in ast.walk(fn)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_add"
    ]


def test_every_manifest_frozen_input_is_staged_or_git_tracked():
    add_args = _manifest_add_args()
    assert len(add_args) >= 9, f"manifest _add list unexpectedly short: {add_args}"
    src = (SCRIPTS / "issue763_cofit_predictors.py").read_text()
    staging = _norm(_block(src, "def _stage_round_inputs", "def write_inputs_manifest"))
    for expr in add_args:
        assert expr in MANIFEST_INPUT_COVERAGE, (
            f"write_inputs_manifest pins a NEW frozen input {expr!r} with no coverage row — "
            "add a staging leg to _stage_round_inputs (or git-track the file) AND a "
            "MANIFEST_INPUT_COVERAGE entry, else a fresh lane crashes at manifest step 0 "
            "(the r3 v0-shard crash class)"
        )
    for expr, (kind, pin) in MANIFEST_INPUT_COVERAGE.items():
        assert expr in add_args, (
            f"stale MANIFEST_INPUT_COVERAGE row {expr!r} — no longer an _add in "
            "write_inputs_manifest; drop or update the row"
        )
        if kind == _STAGED:
            assert _norm(pin) in staging, (
                f"manifest frozen input {expr!r} is NOT staged before the manifest: expected "
                f"{pin!r} inside _stage_round_inputs. Lazy staging at first-read time (the "
                "_load_v0 pattern) fires AFTER write_inputs_manifest and crashes a fresh "
                "lane at manifest step 0 (r3 crash)"
            )
        else:
            proc = subprocess.run(
                ["git", "-C", str(REPO), "ls-files", "--error-unmatch", "--", pin],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, (
                f"manifest frozen input {expr!r} claims git-tracked coverage but {pin!r} "
                f"is not in the git index: {proc.stderr.strip()}"
            )


def test_main_stages_round_inputs_before_manifest():
    src = (SCRIPTS / "issue763_cofit_predictors.py").read_text()
    seg = _norm(_block(src, "def main", "if __name__"))
    assert seg.index("_stage_round_inputs(args.behaviors)") < seg.index("write_inputs_manifest("), (
        "main must run _stage_round_inputs BEFORE write_inputs_manifest (non-smoke path) — "
        "the manifest fail-louds on any input the staging step has not yet fetched"
    )
