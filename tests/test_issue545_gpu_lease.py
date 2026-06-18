"""Issue #545 round-10 — disjoint GPU leasing + spawn-time CVD pinning.

Root cause pinned on pod-545 (2026-06-11 P1 OOM): ``import peft`` initializes
the CUDA driver, freezing the process's visible-device list; every in-process
``os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)`` that runs AFTER that
import is silently ignored, so all four concurrent trains (3 hydra
``+gpu_id=0/1/2`` + 1 train_lora ``gpu_id=3``) landed on physical GPU 0 and
OOM'd at ~79 GB. The fix pins CUDA_VISIBLE_DEVICES in the SUBPROCESS
ENVIRONMENT at spawn time (import-order-proof) and reorders the two
in-process sets to precede the peft import. These tests pin:

1. the lease pattern hands concurrent workers DISJOINT GPU ids;
2. ``_run_one_cell`` threads CUDA_VISIBLE_DEVICES=<lease> into every
   single-GPU subprocess env (and does NOT restrict the multi-GPU fullft arm);
3. the busy-GPU pre-launch guard waits boundedly for benign teardown lag
   (slow vLLM worker release), then fails loud naming the conflict;
4. source-order regression pins: the CVD set precedes the peft import in
   ``train_lora`` and precedes the runner import in ``scripts/train.py``.
"""

from __future__ import annotations

import importlib.util
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep():
    return _load_script("issue545_sweep")


# ---------------------------------------------------------------------------
# 1. Lease disjointness under concurrency
# ---------------------------------------------------------------------------


def test_concurrent_leases_disjoint(sweep):
    """4 GPUs, 12 cells (incl. a 3-seed hydra row): no two concurrent cells
    may ever hold the same GPU id, and ids stay within the slot range."""
    n_gpus = 4
    gpu_slots: Queue[int] = Queue()
    for g in range(n_gpus):
        gpu_slots.put(g)

    lock = threading.Lock()
    active: set[int] = set()
    seen: list[int] = []

    def fake_cell(gpu: int) -> int:
        with lock:
            assert gpu not in active, f"GPU {gpu} double-booked across concurrent cells"
            assert 0 <= gpu < n_gpus
            active.add(gpu)
            seen.append(gpu)
        time.sleep(0.01)  # hold the lease long enough to overlap with peers
        with lock:
            active.remove(gpu)
        return gpu

    # A mix of cells: one "hydra row" with 3 seeds (3 one-GPU cells, exactly
    # the bad_medical primary shape that crashed) + 9 other one-GPU cells.
    cells = [("bad_medical", "primary", s) for s in (0, 137, 42)]
    cells += [(f"row{i}", "cn", 0) for i in range(9)]

    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        futures = [pool.submit(sweep._with_gpu_lease, gpu_slots, fake_cell) for _ in cells]
        results = [f.result() for f in futures]

    assert len(results) == len(cells)
    assert gpu_slots.qsize() == n_gpus, "every lease must be returned"
    assert set(seen) <= set(range(n_gpus))


def test_lease_released_on_failure(sweep):
    """A crashing cell must return its lease (otherwise the pool starves)."""
    gpu_slots: Queue[int] = Queue()
    gpu_slots.put(0)

    def boom(gpu: int):
        raise RuntimeError("cell crashed")

    with pytest.raises(RuntimeError, match="cell crashed"):
        sweep._with_gpu_lease(gpu_slots, boom)
    assert gpu_slots.qsize() == 1


# ---------------------------------------------------------------------------
# 2. Spawn-time CVD threading in _run_one_cell
# ---------------------------------------------------------------------------


class _FakeRow:
    """Minimal RowSpec stand-in for _run_one_cell."""

    def __init__(self, row_id: str = "bad_medical", gpu_prep=None, recipe_kind="hydra_turner"):
        self.row_id = row_id
        self.gpu_prep = gpu_prep
        self.recipe_kind = recipe_kind
        self.train_lora_overrides: dict = {}

    def cell_id(self, arm: str, seed: int) -> str:
        return f"{self.row_id}_{arm}_seed{seed}"


def _train_args(**kw) -> SimpleNamespace:
    base = dict(
        smoke=False,
        skip_train=False,
        skip_eval=True,
        skip_judges=True,
        skip_upload=True,
        max_probes=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_run_one_cell_pins_cvd_at_spawn(sweep, monkeypatch, tmp_path):
    """Every subprocess of a single-GPU cell carries CUDA_VISIBLE_DEVICES=<lease>."""
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path))
    spawned: list[tuple[str, dict | None]] = []
    monkeypatch.setattr(
        sweep, "_run", lambda cmd, *, label, extra_env=None: spawned.append((label, extra_env))
    )
    guard_calls: list[int] = []
    monkeypatch.setattr(
        sweep,
        "_assert_gpu_memory_free",
        lambda gpu, *, label, threshold_mib=2048: guard_calls.append(gpu),
    )

    row = _FakeRow()
    sweep._run_one_cell(row, "cn", 0, 2, _train_args())

    assert spawned, "expected prep + train subprocess spawns"
    labels = [label for label, _ in spawned]
    assert any(label.startswith("prep-") for label in labels)
    assert any(label.startswith("train-") for label in labels)
    for label, extra_env in spawned:
        assert extra_env == {"CUDA_VISIBLE_DEVICES": "2"}, (
            f"{label} spawned without the spawn-time CVD pin: {extra_env!r}"
        )
    assert guard_calls == [2], "busy-GPU guard must probe the leased GPU before the train"


def test_fullft_cell_not_cvd_restricted(sweep, monkeypatch, tmp_path):
    """The fullft arm (ZeRO-3 over all GPUs) must NOT inherit a 1-GPU pin."""
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path))
    spawned: list[tuple[str, dict | None]] = []
    monkeypatch.setattr(
        sweep, "_run", lambda cmd, *, label, extra_env=None: spawned.append((label, extra_env))
    )
    guard_calls: list[int] = []
    monkeypatch.setattr(
        sweep,
        "_assert_gpu_memory_free",
        lambda gpu, *, label, threshold_mib=2048: guard_calls.append(gpu),
    )

    row = _FakeRow(recipe_kind="hydra_turner")
    sweep._run_one_cell(row, "fullft", 0, 0, _train_args())

    train_envs = [extra_env for label, extra_env in spawned if label.startswith("train-")]
    assert train_envs and all(e is None for e in train_envs), (
        f"fullft train spawn must not carry a CVD restriction: {train_envs!r}"
    )
    assert guard_calls == [], "single-GPU guard must not gate the multi-GPU fullft arm"


# ---------------------------------------------------------------------------
# 3. Busy-GPU pre-launch guard
# ---------------------------------------------------------------------------


def test_gpu_guard_raises_after_bounded_wait_when_never_clears(sweep, monkeypatch):
    """Never-clearing busy GPU: poll the bounded number of times, then raise the
    SAME diagnostic message augmented with how long the guard waited."""
    queries: list[int] = []
    sleeps: list[float] = []
    monkeypatch.setattr(sweep, "_query_gpu_used_mib", lambda gpu: queries.append(gpu) or 23456)
    monkeypatch.setattr(sweep.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(
        RuntimeError,
        match=r"GPU lease conflict: physical GPU 1 .* 23456 MiB .* after waiting 120s",
    ):
        sweep._assert_gpu_memory_free(1, label="train-test_cell")

    expected_polls = int(sweep.GPU_GUARD_WAIT_TIMEOUT_S / sweep.GPU_GUARD_POLL_INTERVAL_S)
    assert len(sleeps) == expected_polls, "guard must stop polling at the bounded timeout"
    assert len(queries) == 1 + expected_polls  # initial probe + one per poll
    assert all(s == sweep.GPU_GUARD_POLL_INTERVAL_S for s in sleeps)


def test_gpu_guard_proceeds_when_teardown_lag_clears(sweep, monkeypatch):
    """Benign teardown lag (vLLM worker still releasing memory) must NOT kill
    the sweep: the guard polls until the GPU clears, then proceeds."""
    readings = iter([23456, 9000, 3000, 100])
    queries: list[int] = []
    sleeps: list[float] = []

    def fake_query(gpu: int) -> int:
        queries.append(gpu)
        return next(readings)

    monkeypatch.setattr(sweep, "_query_gpu_used_mib", fake_query)
    monkeypatch.setattr(sweep.time, "sleep", lambda s: sleeps.append(s))

    sweep._assert_gpu_memory_free(0, label="train-test_cell")  # no raise

    assert len(queries) == 4, "guard must re-poll until the teardown clears"
    assert len(sleeps) == 3, "one sleep before each re-poll"


def test_gpu_guard_passes_on_free_gpu(sweep, monkeypatch):
    """Immediately-below-threshold: no wait, no extra polls."""
    queries: list[int] = []
    monkeypatch.setattr(sweep, "_query_gpu_used_mib", lambda gpu: queries.append(gpu) or 3)
    monkeypatch.setattr(
        sweep.time,
        "sleep",
        lambda s: pytest.fail("guard must not sleep when the GPU is already free"),
    )

    sweep._assert_gpu_memory_free(0, label="train-test_cell")  # no raise

    assert queries == [0], "exactly one probe, no polling loop"


def test_gpu_guard_skips_without_nvidia_smi(sweep, monkeypatch):
    def _no_smi(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(sweep.subprocess, "run", _no_smi)
    sweep._assert_gpu_memory_free(0, label="train-test_cell")  # best-effort: no raise


# ---------------------------------------------------------------------------
# 4. Source-order regression pins (the peft-import CUDA freeze)
# ---------------------------------------------------------------------------


def test_train_lora_sets_cvd_before_peft_import():
    from explore_persona_space.train.sft import train_lora

    src = inspect.getsource(train_lora)
    cvd_set = src.index('os.environ["CUDA_VISIBLE_DEVICES"]')
    peft_import = src.index("from peft import")
    assert cvd_set < peft_import, (
        "train_lora must set CUDA_VISIBLE_DEVICES BEFORE importing peft — the peft "
        "import initializes the CUDA driver and freezes the visible-device list "
        "(issue #545 round-10 OOM)"
    )


def test_train_py_sets_cvd_before_runner_import():
    src = (REPO_ROOT / "scripts" / "train.py").read_text()
    cvd_set = src.index('os.environ["CUDA_VISIBLE_DEVICES"]')
    runner_import = src.index("from explore_persona_space.orchestrate.runner import")
    assert cvd_set < runner_import, (
        "scripts/train.py must pin CUDA_VISIBLE_DEVICES before importing the runner "
        "(whose module imports pull in peft and freeze the CUDA device list)"
    )


def test_train_cell_main_sets_cvd_before_dispatch():
    src = (REPO_ROOT / "scripts" / "issue545_train_cell.py").read_text()
    main_src = src[src.index("def main(") :]
    cvd_set = main_src.index('os.environ["CUDA_VISIBLE_DEVICES"]')
    assert 'if args.arm != "fullft":' in main_src
    assert cvd_set < main_src.index("prep_corpus(")
    assert cvd_set < main_src.index("train_cell(")
