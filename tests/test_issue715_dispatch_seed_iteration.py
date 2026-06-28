"""Issue #715 BLOCKER-3 + BLOCKER-4 — P2/P3 iterate 3 seeds; P4 runs the 4-cell grid.

BLOCKER-3: P2 and P3 were hardwired to ``--seed 42``, so the plan §3 P3
supported-iff criterion ("in ≥2 of 3 seeds") was unsatisfiable. The fix iterates
the registered LoRA seed set {42, 137, 256}; this test monkeypatches the
dispatcher's ``_run`` + checkpoint resolution and asserts P3 schedules exactly 3
``issue715_p3_d_projection.py`` invocations (one per seed, each carrying its own
``--seed``), and P2 likewise.

BLOCKER-4: P4 ran once with default scope/granularity. The fix loops the 4-cell
``{down_proj, all_linear} x {per_tensor, global}`` grid; this test asserts P4
schedules exactly 4 ``issue715_p4_geometry_pruning.py`` invocations spanning all
4 (scope, granularity) cells.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dispatch(monkeypatch):
    # The dispatcher asserts credentials at import-of-main, not at module import,
    # but it does call load_dotenv + import issue715_common at module top. Provide
    # a fake credential env so any incidental check passes.
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    spec = importlib.util.spec_from_file_location(
        "issue715_dispatch", REPO_ROOT / "scripts" / "issue715_dispatch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    def __init__(self, **kw):
        self.cells = None
        self.seeds = None
        self.smoke = False
        self.phase = "phase3"
        for k, v in kw.items():
            setattr(self, k, v)


def _capture_runs(mod, monkeypatch):
    """Stub _run to record every scheduled command; return the list."""
    calls: list[list[str]] = []

    def fake_run(cmd, *, gpu_id=None):
        calls.append(list(cmd))

    monkeypatch.setattr(mod, "_run", fake_run)
    return calls


def test_phase3_schedules_one_invocation_per_seed(monkeypatch, tmp_path):
    mod = _load_dispatch(monkeypatch)
    calls = _capture_runs(mod, monkeypatch)
    # D* + checkpoint resolution stubbed (no eval_results on disk in the test).
    monkeypatch.setattr(mod, "_read_dstar", lambda: 0.5)
    monkeypatch.setattr(
        mod,
        "_dstar_matched_lora_ckpt",
        lambda condition, seed, dstar: tmp_path / f"{condition}_seed{seed}_ckpt",
    )

    mod.phase_phase3(_Args(phase="phase3"))

    p3_calls = [c for c in calls if any("issue715_p3_d_projection.py" in a for a in c)]
    assert len(p3_calls) == 3, f"P3 should run once per seed (3), got {len(p3_calls)}"
    seeds_seen = sorted(int(c[c.index("--seed") + 1]) for c in p3_calls)
    assert seeds_seen == [42, 137, 256], f"P3 must cover the registered seed set, got {seeds_seen}"


def test_phase2_schedules_one_invocation_per_seed(monkeypatch, tmp_path):
    mod = _load_dispatch(monkeypatch)
    calls = _capture_runs(mod, monkeypatch)
    monkeypatch.setattr(mod, "_read_dstar", lambda: 0.5)
    monkeypatch.setattr(
        mod,
        "_dstar_matched_lora_ckpt",
        lambda condition, seed, dstar: tmp_path / f"{condition}_seed{seed}_ckpt",
    )

    mod.phase_phase2(_Args(phase="phase2"))

    p2_calls = [c for c in calls if any("issue715_p2_gradient_mass.py" in a for a in c)]
    assert len(p2_calls) == 3, f"P2 should run once per seed (3), got {len(p2_calls)}"
    seeds_seen = sorted(int(c[c.index("--seed") + 1]) for c in p2_calls)
    assert seeds_seen == [42, 137, 256], f"P2 must cover the registered seed set, got {seeds_seen}"


def test_phase4_schedules_all_four_scope_granularity_cells(monkeypatch, tmp_path):
    mod = _load_dispatch(monkeypatch)
    calls = _capture_runs(mod, monkeypatch)
    monkeypatch.setattr(mod, "_read_dstar", lambda: 0.5)
    monkeypatch.setattr(
        mod,
        "_dstar_matched_fullft_ckpt",
        lambda condition, dstar: tmp_path / f"{condition}_ckpt",
    )

    mod.phase_phase4(_Args(phase="phase4"))

    p4_calls = [c for c in calls if any("issue715_p4_geometry_pruning.py" in a for a in c)]
    assert len(p4_calls) == 4, f"P4 should run the 4-cell grid, got {len(p4_calls)}"
    cells_seen = sorted(
        (c[c.index("--scope") + 1], c[c.index("--granularity") + 1]) for c in p4_calls
    )
    assert cells_seen == [
        ("all_linear", "global"),
        ("all_linear", "per_tensor"),
        ("down_proj", "global"),
        ("down_proj", "per_tensor"),
    ], f"P4 must span {{down_proj, all_linear}} x {{per_tensor, global}}, got {cells_seen}"


def test_smoke_phase3_runs_single_seed(monkeypatch, tmp_path):
    """The smoke slice runs ONE seed (canary) — unification, not the full sweep."""
    mod = _load_dispatch(monkeypatch)
    calls = _capture_runs(mod, monkeypatch)
    monkeypatch.setattr(mod, "_read_dstar", lambda: 0.5)
    monkeypatch.setattr(
        mod,
        "_dstar_matched_lora_ckpt",
        lambda condition, seed, dstar: tmp_path / f"{condition}_seed{seed}_ckpt",
    )

    mod.phase_phase3(_Args(phase="phase3", smoke=True, cells=2, seeds=1))

    p3_calls = [c for c in calls if any("issue715_p3_d_projection.py" in a for a in c)]
    assert len(p3_calls) == 1, f"smoke P3 should run 1 seed, got {len(p3_calls)}"
    assert int(p3_calls[0][p3_calls[0].index("--seed") + 1]) == 42
