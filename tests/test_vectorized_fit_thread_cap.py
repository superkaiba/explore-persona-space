"""#1079 default CPU thread cap in ``analysis/vectorized_mlp_skill``.

Covers the plan §6 T1-T8 battery:

- T1-T4: the ``_resolve_num_threads`` resolver — ambient-ceilinged default,
  host-independence, ``EPS_VECTORIZED_FIT_DEFAULT_THREADS`` override semantics
  (#847 parity: ``""``/``"0"`` disable; negative/malformed fail loud), the
  ``num_threads=0`` opt-out, and verbatim (unclamped) explicit values.
- T5-T6: recorder-based pin-and-restore behavior on ALL THREE real fitters
  (an undecorated fitter FAILS T5 — the recorder stays empty), plus the
  never-raises-ambient guarantee of the None default.
- T7: the parity gates are insulated from the default — no resolved-default
  pin may fire inside ``assert_group_mlp_matches_serial``'s 2-thread gate
  body, and ``assert_split_mlp_matches_serial`` runs pin-free (ambient).
- T8: subprocess import-purity — importing the module mutates NO torch
  global state (a sentinel pool of 3 survives the import).

The recorder monkeypatches ``torch.set_num_threads`` to RECORD each requested
value AND call through, so every behavioral assertion fails when the change
is absent or partial (the round-1 false-green gap this battery closes).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from explore_persona_space.analysis import vectorized_mlp_skill as vms
from explore_persona_space.analysis.vectorized_mlp_skill import (
    MLPGroup,
    SplitMLPGroup,
    _resolve_num_threads,
    assert_group_mlp_matches_serial,
    assert_split_mlp_matches_serial,
    fit_batched_loco_mlp,
    fit_batched_loco_mlp_multihead,
    fit_batched_split_mlp,
)

ENV_KEY = "EPS_VECTORIZED_FIT_DEFAULT_THREADS"


@pytest.fixture
def clear_env(monkeypatch):
    """Remove the env override so the DEFAULT_FIT_NUM_THREADS path is exercised."""
    monkeypatch.delenv(ENV_KEY, raising=False)


@pytest.fixture
def ambient_pool_3():
    """Pin the ambient torch pool to a sentinel (3); restore the original on teardown."""
    prev = torch.get_num_threads()
    torch.set_num_threads(3)
    yield 3
    torch.set_num_threads(prev)


@pytest.fixture
def thread_pin_recorder(monkeypatch):
    """Record every ``torch.set_num_threads`` value AND call through to the real fn."""
    calls: list[int] = []
    real = torch.set_num_threads

    def _recording(n):
        calls.append(int(n))
        real(n)

    monkeypatch.setattr(torch, "set_num_threads", _recording)
    return calls


def _tiny_loco_groups(rng: np.random.Generator) -> list[MLPGroup]:
    """Minimal valid LOCO input: one group, n=4 folds, d_in=3, p=2."""
    n, d, p = 4, 3, 2
    return [
        MLPGroup(
            ("t", 0),
            rng.standard_normal((n, d)).astype(np.float32),
            rng.standard_normal((n, p)).astype(np.float32),
        )
    ]


def _tiny_split_groups(rng: np.random.Generator) -> list[SplitMLPGroup]:
    """Minimal valid split input: one group, 6 train / 2 eval rows, d_in=3, p=2."""
    n_tr, n_ev, d, p = 6, 2, 3, 2
    X = rng.standard_normal((n_tr + n_ev, d)).astype(np.float32)
    Y = rng.standard_normal((n_tr + n_ev, p)).astype(np.float32)
    return [SplitMLPGroup(("t",), X[:n_tr], Y[:n_tr], X[n_tr:])]


# ── T1-T4: the resolver ───────────────────────────────────────────────────────


def test_resolve_none_default(clear_env, monkeypatch):
    """T1: env cleared, None -> max(1, min(8, cpu_count, ambient)); ambient only caps DOWN."""
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    assert _resolve_num_threads(None) == 8
    assert _resolve_num_threads(None, ambient=3) == 3  # ambient ceiling caps DOWN
    assert _resolve_num_threads(None, ambient=16) == 8  # ambient above cap: cap binds


def test_resolve_host_independence(clear_env, monkeypatch):
    """T2: small / unreadable os.cpu_count() never crashes and bounds correctly."""
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert _resolve_num_threads(None) == 2
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert _resolve_num_threads(None) == 8  # unreadable cpu_count: cap alone
    assert _resolve_num_threads(None, ambient=3) == 3


def test_resolve_env_override(monkeypatch):
    """T3: env "4" -> 4; ""/"0" -> disabled (#847 parity); negative/malformed -> ValueError."""
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setenv(ENV_KEY, "4")
    assert _resolve_num_threads(None) == 4
    monkeypatch.setenv(ENV_KEY, "")
    assert _resolve_num_threads(None) is None
    monkeypatch.setenv(ENV_KEY, "0")
    assert _resolve_num_threads(None) is None
    monkeypatch.setenv(ENV_KEY, "00")  # numeric-zero form past the string gate
    assert _resolve_num_threads(None) is None
    monkeypatch.setenv(ENV_KEY, "-1")
    with pytest.raises(ValueError):
        _resolve_num_threads(None)
    monkeypatch.setenv(ENV_KEY, "eight")
    with pytest.raises(ValueError):
        _resolve_num_threads(None)


def test_resolve_optout_and_explicit(clear_env):
    """T4: 0 -> None (opt-out); positive -> verbatim (no ambient clamp); negative -> ValueError."""
    assert _resolve_num_threads(0) is None
    assert _resolve_num_threads(5) == 5
    assert _resolve_num_threads(5, ambient=3) == 5  # explicit values are NEVER clamped
    with pytest.raises(ValueError):
        _resolve_num_threads(-2)


# ── T5-T6: the fitters pin-and-restore (recorder-certified) ───────────────────

FITTERS = [
    pytest.param(fit_batched_loco_mlp, _tiny_loco_groups, id="loco"),
    pytest.param(fit_batched_loco_mlp_multihead, _tiny_loco_groups, id="multihead"),
    pytest.param(fit_batched_split_mlp, _tiny_split_groups, id="split"),
]


@pytest.mark.parametrize("fitter,builder", FITTERS)
def test_fitters_pin_and_restore(fitter, builder, monkeypatch, ambient_pool_3, thread_pin_recorder):
    """T5: every real fitter pins (default AND explicit) then restores; 0 opts out.

    FAILS if a fitter is undecorated: the recorder stays empty on (a)/(b).
    """
    monkeypatch.setenv(ENV_KEY, "2")
    rng = np.random.default_rng(0)
    kw = dict(seed=658, hidden=4, max_epochs=2, device="cpu")

    # (a) omitted num_threads -> env default 2 (< ambient 3): pin 2, restore 3.
    fitter(builder(rng), **kw)
    assert thread_pin_recorder == [2, 3], thread_pin_recorder
    assert torch.get_num_threads() == 3
    thread_pin_recorder.clear()

    # (b) explicit num_threads=2: honored verbatim during the fit, restored after.
    fitter(builder(rng), num_threads=2, **kw)
    assert thread_pin_recorder == [2, 3], thread_pin_recorder
    assert torch.get_num_threads() == 3
    thread_pin_recorder.clear()

    # (c) num_threads=0: explicit opt-out — NO fit-local pin at all.
    fitter(builder(rng), num_threads=0, **kw)
    assert thread_pin_recorder == [], thread_pin_recorder
    assert torch.get_num_threads() == 3


def test_default_never_raises_ambient(clear_env, ambient_pool_3, thread_pin_recorder):
    """T6: env cleared, ambient=3 — the None default resolves to <=3 (never raises the pool)."""
    rng = np.random.default_rng(1)
    fit_batched_loco_mlp_multihead(
        _tiny_loco_groups(rng), seed=658, hidden=4, max_epochs=2, device="cpu"
    )
    assert thread_pin_recorder, "expected a fit-local default pin to fire"
    assert max(thread_pin_recorder) <= 3, thread_pin_recorder
    assert torch.get_num_threads() == 3


# ── T7: the parity gates are insulated from the default ──────────────────────


def test_parity_gate_unaffected(clear_env, ambient_pool_3, thread_pin_recorder):
    """T7a: the 2-thread gate pin governs the WHOLE gate body — the resolved
    default never fires inside ``_gate_body`` (recorder shows EXACTLY the
    wrapper's pin + restore, nothing else)."""
    prev = torch.get_num_threads()
    out = assert_group_mlp_matches_serial(max_epochs=2, hidden=8)
    assert out  # gate completed and returned its deviation dict
    assert torch.get_num_threads() == prev
    # Exactly [wrapper pin 2, wrapper restore prev]. A dropped num_threads=0
    # inside _gate_body would add fit-local pin/restore pairs to the recording.
    assert thread_pin_recorder == [2, prev], thread_pin_recorder


def test_split_serial_gate_no_fit_local_pin(clear_env, ambient_pool_3, thread_pin_recorder):
    """T7b: ``assert_split_mlp_matches_serial`` runs under the ambient pool —
    both its batched arm (num_threads=0) and its serial reference — so no
    fit-local pin fires at all."""
    assert_split_mlp_matches_serial()
    assert thread_pin_recorder == [], thread_pin_recorder
    assert torch.get_num_threads() == 3


# ── T8: import purity (subprocess) ────────────────────────────────────────────


def test_import_no_global_mutation():
    """T8: importing the module changes NO torch global state (subprocess form —
    the in-process cached-import arm would be vacuous)."""
    src_root = Path(vms.__file__).resolve().parents[2]
    code = (
        "import torch; torch.set_num_threads(3); "
        "import explore_persona_space.analysis.vectorized_mlp_skill; "
        "assert torch.get_num_threads() == 3, torch.get_num_threads(); "
        "print('POOL_OK')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    assert "POOL_OK" in proc.stdout
