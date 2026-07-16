"""#1333 GPU-hygiene regression pins (crash-fix r4 — the #557 co-location class).

Attempt 4's FULL run OOM'd at p3_ladder: the ext_icl unit hit
``torch.OutOfMemoryError`` on GPU 0 with 29.09 GiB of p0/p1-era vLLM
EngineCore residue (host pid 1213661) co-resident with the unit's own engine
(41.25 GiB) and its HF teacher-forced load. These tests pin the two guard
layers added in r4 with REAL bodies (only the nvidia-smi boundary is faked,
signature-conformant per code-style "one production-body test per
seam-stubbed function"):

1. ``_per_gpu_used_mib`` — pure parse/aggregate (incl. the ``[N/A]`` ->
   unknown -> above-floor rule).
2. ``_wait_gpus_free`` — pass-on-idle, drain-then-pass, GPU scoping, and the
   FAIL-LOUD timeout on a live holder (the invariant that converts the
   attempt-4 41-GiB-deep OOM into a fast named failure).
3. ``_unit_gpu_preflight`` — CVD-pinned target selection.
4. ``_reap_completed_unit_group`` — no-op on a dead group; kills an orphaned
   group member left behind by an exited unit leader.
5. ``_wait_engine_release`` + the FT rung-loop seam (crash-fix r8, attempt-6
   OOM): every rung's reap is followed by a bounded VRAM drain-wait BEFORE
   the next rung's engine init.
6. r9 (v8 review Critical): every drain-wait fires with ZERO live references
   to HF weights in the caller frame — ``_free_hf`` is take-and-return-None
   with call-site rebinding (a live own 15-30 GiB binding can never drain
   under the 2048 MiB floor) — and rollout persistence precedes the wait so
   a wait timeout never destroys just-generated rollouts.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _dispatch():
    import issue1333_dispatch as d

    return d


UUID_MAP = "0, GPU-aaaa\n1, GPU-bbbb\n2, GPU-cccc\n3, GPU-dddd\n"


def _smi_factory(apps_sequence: list[str]):
    """Injected nvidia-smi seam: returns each apps text in sequence (last one
    repeats), and the fixed index->uuid map for the map query. Signature
    mirrors ``_smi_query(query) -> str | None``."""
    calls = {"apps": 0}

    def smi(query: str) -> str | None:
        if "compute-apps" in query:
            i = min(calls["apps"], len(apps_sequence) - 1)
            calls["apps"] += 1
            return apps_sequence[i]
        return UUID_MAP

    return smi, calls


def test_per_gpu_used_mib_aggregates_per_index():
    d = _dispatch()
    apps = "1213661, 29786, GPU-aaaa\n1245879, 42240, GPU-aaaa\n555, 1024, GPU-cccc\n"
    usage = d._per_gpu_used_mib(apps, UUID_MAP)
    total0, pids0 = usage["0"]
    assert total0 == pytest.approx(29786 + 42240)
    assert pids0 == [1213661, 1245879]
    assert usage["2"][0] == pytest.approx(1024)
    assert "1" not in usage  # no compute apps -> absent (0 used)


def test_per_gpu_used_mib_na_used_memory_is_unknown():
    d = _dispatch()
    apps = "777, [N/A], GPU-bbbb\n"
    usage = d._per_gpu_used_mib(apps, UUID_MAP)
    assert usage["1"][0] is None  # unknown -> callers treat as above-floor


def test_wait_gpus_free_passes_on_idle_gpus():
    d = _dispatch()
    smi, calls = _smi_factory([""])
    d._wait_gpus_free(["0", "1"], label="t", smi=smi, sleep=lambda s: None)
    assert calls["apps"] == 1  # no polling needed


def test_wait_gpus_free_times_out_fail_loud_on_live_holder():
    """The attempt-4 shape: a live 29-GiB holder on GPU 0 never drains ->
    RuntimeError naming the pid + the host-namespace caveat, never an OOM."""
    d = _dispatch()
    smi, _ = _smi_factory(["1213661, 29786, GPU-aaaa\n"])
    with pytest.raises(RuntimeError, match="HOST-namespace") as exc:
        d._wait_gpus_free(
            ["0"], label="unit-preflight[ladder]", timeout_s=0.0, smi=smi, sleep=lambda s: None
        )
    assert "1213661" in str(exc.value)
    assert "unit-preflight[ladder]" in str(exc.value)


def test_wait_gpus_free_drains_then_passes():
    d = _dispatch()
    busy = "42, 30000, GPU-aaaa\n"
    smi, calls = _smi_factory([busy, busy, ""])
    d._wait_gpus_free(["0"], label="t", timeout_s=60.0, smi=smi, sleep=lambda s: None)
    assert calls["apps"] == 3  # two busy polls, then drained


def test_wait_gpus_free_scopes_to_target_gpu():
    d = _dispatch()
    smi, _ = _smi_factory(["42, 30000, GPU-aaaa\n"])  # gpu 0 busy
    # target gpu 2 only -> passes despite gpu 0's holder
    d._wait_gpus_free(["2"], label="t", timeout_s=0.0, smi=smi, sleep=lambda s: None)


def test_wait_gpus_free_below_floor_residue_passes():
    """The main dispatcher's own lingering CUDA context (host-ns pid, <1 GiB)
    must not block units — the floor tolerates it."""
    d = _dispatch()
    smi, _ = _smi_factory(["999, 600, GPU-aaaa\n"])
    d._wait_gpus_free(
        ["0"], label="t", floor_mib=2048, timeout_s=0.0, smi=smi, sleep=lambda s: None
    )


def test_wait_gpus_free_na_counts_above_floor():
    d = _dispatch()
    smi, _ = _smi_factory(["42, [N/A], GPU-aaaa\n"])
    with pytest.raises(RuntimeError):
        d._wait_gpus_free(["0"], label="t", timeout_s=0.0, smi=smi, sleep=lambda s: None)


def test_wait_gpus_free_noop_when_smi_unavailable():
    d = _dispatch()
    d._wait_gpus_free(["0"], label="t", smi=lambda q: None, sleep=lambda s: None)


def test_unit_gpu_preflight_targets_cvd_pin(monkeypatch):
    """The preflight probes the CVD-pinned physical GPU, not --gpu-id, when
    the launcher env pin is present (the fanout contract)."""
    d = _dispatch()
    seen: dict = {}

    def fake_wait(targets, *, label, floor_mib, timeout_s):
        seen["targets"] = list(targets)
        seen["label"] = label

    monkeypatch.setattr(d, "_wait_gpus_free", fake_wait)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    d._unit_gpu_preflight("ladder", "0")
    assert seen["targets"] == ["2"]
    assert seen["label"] == "unit-preflight[ladder]"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    d._unit_gpu_preflight("ladder", "3")
    assert seen["targets"] == ["3"]  # --gpu-id fallback when no CVD pin


def test_reap_completed_unit_group_noop_on_dead_group():
    d = _dispatch()
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait(timeout=10)
    t0 = time.monotonic()
    d._reap_completed_unit_group(proc, ["--full", "--unit", "ladder", "x"], grace_s=5.0)
    assert time.monotonic() - t0 < 2.0  # healthy case: instant no-op, no grace sleep


def test_wait_engine_release_targets_cvd_and_emits_freed_line(monkeypatch, caplog):
    """Real body (only the nvidia-smi seam faked): CVD-pinned targeting + the
    ``[rung-reap] rung=<k> gpu=<id> freed`` fix-engaged line (crash-fix r8)."""
    import logging

    d = _dispatch()
    smi, calls = _smi_factory([""])  # idle GPUs -> drains immediately
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with caplog.at_level(logging.INFO):
        d._wait_engine_release(label="rung=100", smi=smi)
    assert calls["apps"] == 1
    assert "[rung-reap] rung=100 gpu=2 freed" in caplog.text


def test_wait_engine_release_noops_without_gpus(monkeypatch):
    """CPU host (no CVD pin, nvidia-smi absent): graceful no-op, never a probe."""
    d = _dispatch()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def no_gpus() -> list[str]:
        raise RuntimeError("no GPUs visible via nvidia-smi")

    monkeypatch.setattr(d, "_physical_gpu_ids", no_gpus)

    def smi_must_not_run(query: str) -> str | None:
        raise AssertionError("drain-wait must be skipped on a CPU host")

    d._wait_engine_release(label="rung=100", smi=smi_must_not_run)


def test_ft_rung_loop_reaps_and_waits_between_rungs(monkeypatch, tmp_path):
    """Crash-fix r8 pin (attempt-6 OOM): the FT rung loop calls _reap_engine,
    persists the rollouts (r9: BEFORE the wait, so a wait timeout never
    destroys them), THEN _wait_engine_release after EVERY rung's read, before
    the next rung's engine init. Real ``_ladder_reads_ft`` body; fakes only
    at the GPU/engine/model boundaries (signature-conformant defs)."""
    from types import SimpleNamespace

    d = _dispatch()
    events: list[str] = []
    rungs = {100: tmp_path / "checkpoint-100", 200: tmp_path / "checkpoint-200"}

    monkeypatch.setattr(d.d1112, "_ensure_dir_tokenizer", lambda ckpt: None)

    def fake_persist(cfg, stage: str, cell: str, payload: dict) -> None:
        events.append("persist")

    monkeypatch.setattr(d, "_persist_rollouts", fake_persist)

    def fake_engine(model_path: str, *, enable_lora: bool = False):
        step = Path(model_path).name.removeprefix("checkpoint-")
        events.append(f"engine:{step}")
        return object()

    def fake_greedy(llm, prompts: list[str], max_new: int, *, lora_request=None) -> list[str]:
        events.append("greedy")
        return ["answer"] * len(prompts)

    def fake_reap(llm) -> None:
        events.append("reap")

    def fake_wait(*, label: str, smi=None) -> None:
        events.append(f"wait:{label}")

    def fake_load_hf(model_path: str, device: str = "cuda:0"):
        events.append("hf-load")
        return object()

    def fake_slot_read(model, tokenizer, contexts: list[str], device: str = "cuda:0"):
        return [{"logp": -1.0, "argmax_id": 0} for _ in contexts]

    monkeypatch.setattr(d, "_vllm_engine", fake_engine)
    monkeypatch.setattr(d, "_greedy", fake_greedy)
    monkeypatch.setattr(d, "_reap_engine", fake_reap)
    monkeypatch.setattr(d, "_wait_engine_release", fake_wait)
    monkeypatch.setattr(d, "_load_hf", fake_load_hf)
    monkeypatch.setattr(d, "_slot_read", fake_slot_read)
    monkeypatch.setattr(d, "_free_hf", lambda model: None)

    cfg = SimpleNamespace(out_root=tmp_path, smoke=True)
    ladder: dict[int, dict] = {}
    d._ladder_reads_ft(
        cfg, "mk4_fullft_pos", rungs, [100, 200], ladder, None, ["p1", "p2"], lambda: None
    )

    assert sorted(ladder) == [100, 200]
    i1, i2 = events.index("engine:100"), events.index("engine:200")
    between = events[i1 + 1 : i2]
    # rung 1's engine is reaped, rollouts persisted (r9), AND its VRAM
    # drain-waited BEFORE rung 2's init
    assert "reap" in between and "persist" in between and "wait:rung=100" in between, events
    assert between.index("reap") < between.index("persist") < between.index("wait:rung=100"), events
    # the LAST rung also reaps + persists + waits (while-loop re-entry seam)
    tail = events[i2 + 1 :]
    assert "reap" in tail and "persist" in tail and "wait:rung=200" in tail, events
    assert tail.index("reap") < tail.index("persist") < tail.index("wait:rung=200"), events


def test_reap_completed_unit_group_kills_orphaned_member():
    """An EngineCore-style orphan (leader exited, child kept the pgid) must be
    reaped by the killpg sweep."""
    d = _dispatch()
    proc = subprocess.Popen(
        # orphan's stdout MUST detach from the pipe or communicate() blocks on
        # the inherited write end until the orphan dies
        ["bash", "-c", "sleep 300 >/dev/null 2>&1 & echo $!; exit 0"],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    out, _ = proc.communicate(timeout=10)
    orphan_pid = int(out.strip())
    assert proc.returncode == 0
    # orphan is alive and holds the unit's pgid
    assert os.getpgid(orphan_pid) == proc.pid
    d._reap_completed_unit_group(proc, ["--full", "--unit", "ladder", "x"], grace_s=0.2)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(orphan_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(orphan_pid, signal.SIGKILL)
        pytest.fail("orphaned group member survived the killpg sweep")


def test_free_hf_returns_none_for_rebinding():
    """r9 contract pin (v8 Critical): ``_free_hf`` is take-and-return-None so
    ``x = _free_hf(x)`` rebinds the caller's local to None — ``del`` inside
    the callee alone leaves the caller's binding (and the weights) live."""
    d = _dispatch()
    assert d._free_hf(object()) is None


def test_ft_rung_wait_fires_with_no_live_hf_refs(monkeypatch, tmp_path):
    """r9 pin (v8 Critical, instance A): at EVERY FT-rung drain-wait, no live
    reference to a previously-loaded HF model may survive in the caller
    frame — a live own binding keeps ~15-30 GiB resident, so
    ``_wait_gpus_free`` can never drop under the 2048 MiB floor and times
    out after 180 s. On the v8 tip rung 2's wait fired while rung 1's
    ``model`` + ``base`` were still bound (rebound only later), so this
    test FAILS there; the r9 call-site rebinding makes it pass. Real
    ``_ladder_reads_ft`` + real ``_free_hf``/``_release_cuda`` bodies;
    fakes only at the GPU/engine/model boundaries."""
    import gc
    import weakref
    from types import SimpleNamespace

    d = _dispatch()

    class FakeHfModel:
        """Weakref-able stand-in for an AutoModelForCausalLM (GPU boundary)."""

    loaded: list[weakref.ref] = []

    def fake_load_hf(model_path: str, device: str = "cuda:0"):
        m = FakeHfModel()
        loaded.append(weakref.ref(m))
        return m

    live_at_wait: dict[str, int] = {}

    def fake_wait(*, label: str, smi=None) -> None:
        gc.collect()
        live_at_wait[label] = sum(1 for r in loaded if r() is not None)

    monkeypatch.setattr(d.d1112, "_ensure_dir_tokenizer", lambda ckpt: None)
    monkeypatch.setattr(d, "_vllm_engine", lambda path, *, enable_lora=False: object())
    monkeypatch.setattr(
        d, "_greedy", lambda llm, prompts, max_new, *, lora_request=None: ["a"] * len(prompts)
    )
    monkeypatch.setattr(d, "_reap_engine", lambda llm: None)
    monkeypatch.setattr(d, "_wait_engine_release", fake_wait)
    monkeypatch.setattr(d, "_load_hf", fake_load_hf)
    monkeypatch.setattr(
        d,
        "_slot_read",
        lambda model, tokenizer, contexts, device="cuda:0": [
            {"logp": -1.0, "argmax_id": 0} for _ in contexts
        ],
    )

    cfg = SimpleNamespace(out_root=tmp_path, smoke=True)
    d._ladder_reads_ft(
        cfg,
        "mk4_fullft_pos",
        {100: tmp_path / "checkpoint-100", 200: tmp_path / "checkpoint-200"},
        [100, 200],
        {},
        None,
        ["p1"],
        lambda: None,
    )
    assert live_at_wait == {"rung=100": 0, "rung=200": 0}, live_at_wait


def test_lora_pass_wait_fires_with_no_live_hf_refs(monkeypatch, tmp_path):
    """r9 pin (v8 Critical, instance B): the LoRA pass-end drain-wait fires
    only AFTER ``base`` (and the PEFT wrapper pinning it) are dropped — on
    the v8 tip ``base`` (15,260 MiB) was a live local at the wait, so every
    LoRA ladder pass would have timed out at the floor. Real
    ``_ladder_reads_lora`` + real ``_free_hf`` bodies; fakes only at the
    GPU/engine/model/peft boundaries (signature-conformant defs)."""
    import gc
    import types
    import weakref
    from types import SimpleNamespace

    d = _dispatch()

    class FakeHfModel:
        """Weakref-able stand-in for an AutoModelForCausalLM (GPU boundary)."""

    loaded: list[weakref.ref] = []

    def fake_load_hf(model_path: str, device: str = "cuda:0"):
        m = FakeHfModel()
        loaded.append(weakref.ref(m))
        return m

    class FakePeftModel:
        """Mirrors peft.PeftModel.from_pretrained(model, model_id)/.unload():
        the wrapper HOLDS a reference to the base model, as the real one does."""

        def __init__(self, model):
            self._base = model

        @classmethod
        def from_pretrained(cls, model, model_id, **kwargs):
            return cls(model)

        def unload(self):
            return self._base

    class FakeLoRARequest:
        """Mirrors vllm.lora.request.LoRARequest(lora_name, lora_int_id, lora_path)."""

        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name = lora_name

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    fake_vllm = types.ModuleType("vllm")
    fake_vllm_lora = types.ModuleType("vllm.lora")
    fake_vllm_req = types.ModuleType("vllm.lora.request")
    fake_vllm_req.LoRARequest = FakeLoRARequest
    fake_vllm.lora = fake_vllm_lora
    fake_vllm_lora.request = fake_vllm_req
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", fake_vllm_lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_vllm_req)

    live_at_wait: dict[str, int] = {}

    def fake_wait(*, label: str, smi=None) -> None:
        gc.collect()
        live_at_wait[label] = sum(1 for r in loaded if r() is not None)

    monkeypatch.setattr(d, "_vllm_engine", lambda path, *, enable_lora=False: object())
    monkeypatch.setattr(
        d, "_greedy", lambda llm, prompts, max_new, *, lora_request=None: ["a"] * len(prompts)
    )
    monkeypatch.setattr(d, "_reap_engine", lambda llm: None)
    monkeypatch.setattr(d, "_wait_engine_release", fake_wait)
    monkeypatch.setattr(d, "_load_hf", fake_load_hf)
    monkeypatch.setattr(
        d,
        "_slot_read",
        lambda model, tokenizer, contexts, device="cuda:0": [
            {"logp": -1.0, "argmax_id": 0} for _ in contexts
        ],
    )

    cfg = SimpleNamespace(out_root=tmp_path, smoke=True)
    d._ladder_reads_lora(
        cfg, "lora_cell", {100: tmp_path / "adapter-100"}, [100], {}, None, ["p1"], lambda: None
    )
    assert live_at_wait == {"lora-pass[lora_cell]": 0}, live_at_wait


def test_wait_engine_release_times_out_on_persistent_own_usage(monkeypatch):
    """Busy-own-GPU shape (the v8 Critical's mechanism, via the _smi_factory
    busy-sequence support): a live own-process HF binding surfaces as
    persistent compute-app usage that never drains, so the drain-wait FAILS
    LOUD at the bound — it can never pass. This is why the ladder seams
    must drop their HF refs BEFORE waiting. Real ``_wait_engine_release`` +
    ``_wait_gpus_free`` bodies (re-parameterized to a 0 s bound; the time
    boundary is the only seam touched)."""
    d = _dispatch()
    real_wait = d._wait_gpus_free

    def bounded_wait(targets, *, label, smi):
        real_wait(targets, label=label, smi=smi, timeout_s=0.0, sleep=lambda s: None)

    monkeypatch.setattr(d, "_wait_gpus_free", bounded_wait)
    smi, _ = _smi_factory(["4242, 15260, GPU-cccc\n"])  # own 15.3 GiB model, never drains
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(RuntimeError, match=r"rung-reap\[lora-pass\[cell\]\]") as exc:
        d._wait_engine_release(label="lora-pass[cell]", smi=smi)
    assert "15260" in str(exc.value)


def test_persist_precedes_wait_at_generation_seams():
    """r9 pin (v8 Minor 1): rollout persistence precedes the drain-wait at
    every generation seam that has both, so a wait timeout can never
    destroy just-generated rollouts (order-of-calls source check)."""
    import inspect

    d = _dispatch()
    for fn in (d._reused_arm_apply_gate, d._bystander_battery, d._ladder_reads_ft):
        src = inspect.getsource(fn)
        assert src.index("_persist_rollouts(") < src.index("_wait_engine_release("), fn.__name__
