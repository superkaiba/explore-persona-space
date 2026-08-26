"""VM pins for the issue-2546 gen-worker EngineCore reap (round-12 crash fix).

Incident (task #2546 arm-1 ``p1_smoke``, pod-2546-arm1b): the POST side
completed all four corpora, then its gen worker exited via ``os._exit(0)`` —
the correct #1739/#2149 finalization-deadlock terminal — but ``os._exit``
skips finalization, so the worker's vLLM v1 EngineCore child was never
reaped: it reparented to init (pid 4528, PPID 1) holding 70,560 MiB of GPU 0,
and the PRE side's engine died at init ("Engine core initialization failed";
"gen-pre_greedy_a1: worker slots failed: [(0, 1)]").

The orphan-starves-successor path itself needs a GPU + a real vLLM engine and
remains POD-ONLY. What IS VM-checkable is pinned here:

* the reap is invoked on every terminal path of ``run_gen_worker`` (clean AND
  body-exception), is ``llm is not None``-guarded, and is ordered BEFORE the
  single ``os._exit`` terminal (AST + behavioral);
* ``_reap_gen_engine``'s REAL body executes — the deferred imports resolve
  (#606), the graceful v1 ``engine_core.shutdown()`` is reached on a
  signature-conformant duck-typed engine, and ``teardown_vllm``'s real drain
  loop runs against a faked nvidia-smi subprocess boundary — through the
  production ``run_gen_worker`` terminal, not a direct helper call;
* a reap/drain failure maps to the loud nonzero ``RC_ENGINE_TEARDOWN`` exit
  (never a silent 0), and a body exception is printed BEFORE the reap so a
  teardown error can never mask it (#1947);
* the two ``sys.exit(0)`` capture workers are pinned vLLM-free (the sibling
  disposition of the round-12 scope: their PyGILState explicit-exit terminal
  is correct exactly because they never build an engine).

Drain-verdict parsing/threshold logic (incl. ``[N/A]`` rows) is already
pinned by tests/test_eval_battery_teardown.py and is NOT duplicated here —
this file only signature-binds those helpers (#1332).

Boundary fakes are signature-conformant by construction (real classes whose
``def``s mirror the production call surfaces — never a bare Mock;
code-style.md "one production-body test per seam-stubbed function").
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import transformers

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2546_gen_capture as G  # noqa: E402

from explore_persona_space.experiments.behavior_testbed_545 import (  # noqa: E402
    eval_battery as EB,
)

GEN_CAPTURE_SRC = REPO_ROOT / "scripts" / "issue2546_gen_capture.py"
UUID_A = "GPU-aaaaaaaa-1111"
FIX_ENGAGED_LINE = "[gen-teardown] EngineCore reaped; drain verdict PASS"


class _ExitCalled(BaseException):
    """Sentinel replacing ``os._exit`` so the test observes the terminal rc.

    BaseException (not Exception) so the worker's body ``except BaseException``
    could never accidentally swallow it — the terminal call sits OUTSIDE the
    try, which is exactly what the AST test pins.
    """

    def __init__(self, rc: int) -> None:
        super().__init__(f"os._exit({rc})")
        self.rc = rc


class _FakeTok:
    """Tokenizer boundary fake: the arm-1 PRE side pins no think delimiters,
    so ``assert_think_pins`` (real body) touches nothing; ``encode`` mirrors
    the one surface it could call."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0]


class _FakeEngineCore:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def shutdown(self) -> None:  # mirrors vLLM v1 EngineCoreClient.shutdown()
        self._log.append("engine_core.shutdown")


class _FakeLLM:
    """Duck-typed vLLM LLM: exactly the attribute chain ``_reap_vllm_engine``
    walks (``llm_engine.engine_core.shutdown``) plus the ``.generate`` attr
    ``generate_chunked``'s ``_eng()`` sniffs via hasattr."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.llm_engine = SimpleNamespace(engine_core=_FakeEngineCore(log))

    def generate(self, prompts, sampling_params, use_tqdm=False):
        raise AssertionError("generate() must not be reached (generate_chunked is faked)")


def _pre_side() -> G.SideSpec:
    (side,) = [s for s in G.ARMS[1].sides if s.side == "pre"]
    return side


def _write_work(tmp_path: Path, *, resumed: bool) -> tuple[SimpleNamespace, Path, str]:
    """A 1-row arm-1 PRE-side work file; ``resumed=True`` pre-writes the
    primary stage so the worker never builds an engine (the None-guard leg)."""
    side = _pre_side()
    fp_sha = "f" * 24
    out_file = tmp_path / "slot0.out.jsonl"
    rows = [
        {
            "row_id": "wildchat:0000",
            "corpus": "wildchat",
            "prompt": "User: hi\nAssistant:",
            "n_prompt_tokens": 5,
            "read_idx": 4,
        }
    ]
    work = {
        "model": side.model,
        "revision": "0" * 40,
        "side_spec": json.loads(json.dumps(asdict(side))),  # the GENUINE JSON handoff
        "cap": side.cap,
        "regen_cap": side.regen_cap,
        "stop_ids": [151645],
        "decode_fallback": False,
        "fp_sha": fp_sha,
        "rel_draws": 0,
        "rel_row_ids": [],
        "rows": rows,
        "out_file": str(out_file),
    }
    wf = tmp_path / "slot0.json"
    wf.write_text(json.dumps(work))
    if resumed:
        rec = {
            **rows[0],
            "kind": "primary",
            "text": "resumed",
            "finish_reason": "stop",
            "n_gen_tokens": 2,
            "regen": False,
        }
        G._write_jsonl(G._gen_stage_path(out_file, "primary"), [{"__stage_fp__": fp_sha}, rec])
    return SimpleNamespace(work_file=str(wf), worker_slot=0), out_file, fp_sha


@pytest.fixture()
def worker_env(monkeypatch):
    """Shared boundary fakes: os._exit -> sentinel; tokenizer load -> _FakeTok."""

    def _fake_exit(rc: int) -> None:
        raise _ExitCalled(rc)

    monkeypatch.setattr(G.os, "_exit", _fake_exit)

    def _fake_from_pretrained(cls, model, revision=None, **kwargs):
        return _FakeTok()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", classmethod(_fake_from_pretrained)
    )


def _arm_engine_fakes(monkeypatch, log: list[str]) -> None:
    """Engine-path fakes mirroring the production call shapes."""

    def fake_build_engine(model: str, revision: str | None):
        log.append("build_engine")
        return _FakeLLM(log)

    def fake_sampling_params(cap, stop_ids, *, greedy, seed=None):
        return SimpleNamespace(cap=cap, greedy=greedy, seed=seed)

    def fake_generate_chunked(llm, prompts, sp, tag, *, ckpt=None):
        eng = llm() if not hasattr(llm, "generate") else llm  # the real _eng() contract
        assert isinstance(eng, _FakeLLM)
        log.append(f"generate:{tag}")
        return [("hello world", "stop", 3) for _ in prompts]

    monkeypatch.setattr(G, "build_engine", fake_build_engine)
    monkeypatch.setattr(G, "sampling_params", fake_sampling_params)
    monkeypatch.setattr(G, "generate_chunked", fake_generate_chunked)


def _install_fake_smi(monkeypatch) -> dict:
    """Signature-conformant nvidia-smi subprocess fake covering BOTH callers:
    eval_battery's ``_smi`` (env kwarg, csv,noheader,nounits) and
    representation_shift's ``_log_zombie_cuda_contexts`` (timeout kwarg,
    csv,noheader). An empty compute-apps read -> instant drain PASS."""
    calls = {"apps": 0, "gpu": 0, "zombie": 0}

    def fake_run(cmd, capture_output=True, text=True, check=False, env=None, timeout=None):
        assert cmd[0] == "nvidia-smi", cmd
        query = cmd[1]
        if query == "--query-compute-apps=pid":  # zombie-context scan (#734 Fix 2)
            calls["zombie"] += 1
            return SimpleNamespace(stdout="", returncode=0)
        if query.startswith("--query-gpu"):
            calls["gpu"] += 1
            return SimpleNamespace(stdout=f"0, {UUID_A}\n", returncode=0)
        assert query.startswith("--query-compute-apps"), cmd
        calls["apps"] += 1
        return SimpleNamespace(stdout="", returncode=0)

    # EB.subprocess / EB.time ARE the stdlib modules, so these patches also
    # cover representation_shift's subprocess.run and _reap_gen_engine's
    # time.sleep(1.0); monkeypatch reverts after the test.
    monkeypatch.setattr(EB.subprocess, "run", fake_run)
    monkeypatch.setattr(EB.time, "sleep", lambda s: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    return calls


# ---------------------------------------------------------------------------
# Behavioral terminal pins (real run_gen_worker body)
# ---------------------------------------------------------------------------


def test_resumed_slot_skips_reap_and_exits_zero(tmp_path, monkeypatch, worker_env):
    """The None-guard leg: a fully-resumed slot never builds an engine, so the
    reap MUST NOT run (a reap on llm=None would be the very trip the brief
    forbids) and the terminal rc stays 0."""
    args, out_file, _ = _write_work(tmp_path, resumed=True)

    def _must_not_reap(llm):
        raise AssertionError("reap must not run on a fully-resumed slot (llm is None)")

    monkeypatch.setattr(G, "_reap_gen_engine", _must_not_reap)
    with pytest.raises(_ExitCalled) as ei:
        G.run_gen_worker(args)
    assert ei.value.rc == 0
    rows = G._read_jsonl(out_file)
    assert [r["row_id"] for r in rows] == ["wildchat:0000"]


def test_engine_built_slot_reaps_before_exit(tmp_path, monkeypatch, worker_env, capfd):
    """REAL ``_reap_gen_engine`` body through the production terminal: deferred
    imports resolve, ``engine_core.shutdown()`` is reached AFTER generation,
    the CVD-scoped drain verdict polls, and the worker exits 0."""
    args, out_file, _ = _write_work(tmp_path, resumed=False)
    log: list[str] = []
    _arm_engine_fakes(monkeypatch, log)
    calls = _install_fake_smi(monkeypatch)
    with pytest.raises(_ExitCalled) as ei:
        G.run_gen_worker(args)
    assert ei.value.rc == 0
    assert "build_engine" in log and "engine_core.shutdown" in log
    assert log.index("engine_core.shutdown") > log.index("generate:primary-pre")
    assert calls["apps"] >= 1, "the drain verdict never polled — reap did not engage"
    rows = G._read_jsonl(out_file)
    assert rows and rows[0]["text"] == "hello world"
    # Durable writes land BEFORE the terminal (the os._exit contract).
    assert G._gen_stage_path(out_file, "primary").is_file()
    assert FIX_ENGAGED_LINE in capfd.readouterr().out


def test_reap_failure_exits_rc_engine_teardown(tmp_path, monkeypatch, worker_env):
    """A drain-verdict timeout maps to the LOUD RC_ENGINE_TEARDOWN worker exit
    (parent then reports "worker slots failed") — never a silent 0 and never a
    propagating exception into finalization (the #1739/#2149 deadlock)."""
    args, out_file, _ = _write_work(tmp_path, resumed=False)
    log: list[str] = []
    _arm_engine_fakes(monkeypatch, log)

    def _boom(llm):
        raise RuntimeError("vLLM teardown left foreign GPU compute PIDs above the residual floor")

    monkeypatch.setattr(G, "_reap_gen_engine", _boom)
    with pytest.raises(_ExitCalled) as ei:
        G.run_gen_worker(args)
    assert ei.value.rc == G.RC_ENGINE_TEARDOWN
    assert ei.value.rc != 0
    # The completed stage was durably written before the failed reap.
    assert G._gen_stage_path(out_file, "primary").is_file()


def test_body_exception_still_reaps_and_exits_nonzero(tmp_path, monkeypatch, worker_env, capfd):
    """The exception terminal ALSO funnels through the reap. Pre-fix, a body
    exception propagated with the engine child alive — finalization deadlock
    or an init-reparented orphan; and the ORIGINAL traceback is printed before
    the reap so a teardown error can never mask it (#1947)."""
    args, _, _ = _write_work(tmp_path, resumed=False)
    log: list[str] = []
    _arm_engine_fakes(monkeypatch, log)
    calls = _install_fake_smi(monkeypatch)

    def _gen_then_boom(llm, prompts, sp, tag, *, ckpt=None):
        llm()  # the engine gets built...
        raise RuntimeError("mid-stage crash")

    monkeypatch.setattr(G, "generate_chunked", _gen_then_boom)
    with pytest.raises(_ExitCalled) as ei:
        G.run_gen_worker(args)
    assert ei.value.rc == 1
    assert "engine_core.shutdown" in log, "reap must run on the exception path too"
    assert calls["apps"] >= 1
    assert "mid-stage crash" in capfd.readouterr().err


# ---------------------------------------------------------------------------
# Structural pins (AST) + deferred-import execution
# ---------------------------------------------------------------------------


def test_run_gen_worker_terminal_shape_ast():
    """ONE os._exit terminal (last statement); the None-guarded reap between
    the try-wrapped body and the exit; the guarded reap itself try-wrapped
    (drain failure -> RC_ENGINE_TEARDOWN, never a propagating finalization);
    no return statement may bypass the terminal."""
    tree = ast.parse(GEN_CAPTURE_SRC.read_text())
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_gen_worker"]

    exits = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_exit"
    ]
    assert len(exits) == 1, "run_gen_worker must have exactly ONE os._exit terminal"
    reaps = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_reap_gen_engine"
    ]
    assert len(reaps) == 1, "exactly one reap call site"
    assert reaps[0].lineno < exits[0].lineno, "the reap must be ORDERED BEFORE os._exit"

    def _own_nodes(func: ast.FunctionDef):
        """func's own body, EXCLUDING nested function bodies (the ``_llm``
        closure's legitimate ``return llm`` must not trip the no-return pin)."""
        stack = list(func.body)
        while stack:
            node = stack.pop()
            yield node
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                continue
            stack.extend(ast.iter_child_nodes(node))

    assert not [n for n in _own_nodes(fn) if isinstance(n, ast.Return)], (
        "no early return may bypass the reap terminal"
    )

    top = fn.body
    try_idx = [i for i, n in enumerate(top) if isinstance(n, ast.Try)]
    assert len(try_idx) == 1, "the stage body must be try-wrapped so exceptions reach the reap"
    guard_idx = [
        i
        for i, n in enumerate(top)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "llm"
        and len(n.test.ops) == 1
        and isinstance(n.test.ops[0], ast.IsNot)
        and isinstance(n.test.comparators[0], ast.Constant)
        and n.test.comparators[0].value is None
    ]
    assert len(guard_idx) == 1, "the reap must be `llm is not None`-guarded at top level"
    exit_idx = [
        i
        for i, n in enumerate(top)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "_exit"
    ]
    assert len(exit_idx) == 1 and exit_idx[0] == len(top) - 1, "os._exit is the LAST statement"
    assert try_idx[0] < guard_idx[0] < exit_idx[0], "order: try body -> reap guard -> os._exit"
    assert any(n is reaps[0] for n in ast.walk(top[guard_idx[0]])), "reap lives INSIDE the guard"
    assert [n for n in top[guard_idx[0]].body if isinstance(n, ast.Try)], (
        "the guarded reap must be try-wrapped (drain failure -> RC_ENGINE_TEARDOWN)"
    )


def test_deferred_reap_imports_resolve_and_bind():
    """#606/#1332: EXECUTE the terminal's deferred imports and bind the call
    shapes (import resolution alone green-lights an arity mismatch).
    Threshold/parsing logic is pinned by tests/test_eval_battery_teardown.py."""
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine
    from explore_persona_space.experiments.behavior_testbed_545.eval_battery import (
        _parse_compute_app_rows,
        _teardown_drain_verdict,
        teardown_vllm,
    )

    inspect.signature(_reap_vllm_engine).bind(object())
    inspect.signature(teardown_vllm).bind(object())
    inspect.signature(G._reap_gen_engine).bind(object())
    inspect.signature(_parse_compute_app_rows).bind("")
    inspect.signature(_teardown_drain_verdict).bind([], my_pid=1, visible_uuids=None, floor_mib=1.0)


def test_capture_workers_are_vllm_free():
    """Sibling disposition (round-12 scope item 3): the two ``sys.exit(0)``
    capture workers are HF-only — no engine construction reachable from their
    bodies, so the PyGILState explicit-exit terminal (NOT the reap terminal)
    is correct there. Trips if a future round routes vLLM into them."""
    tree = ast.parse(GEN_CAPTURE_SRC.read_text())
    for name in ("run_capture_worker", "run_capture_rel_worker"):
        (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        engine_names = {
            "build_engine",
            "generate_chunked",
            "sampling_params",
            "_llm",
            "_reap_gen_engine",
        }
        assert not names & engine_names, (name, names & engine_names)
        mods = {a.module for a in ast.walk(fn) if isinstance(a, ast.ImportFrom)}
        assert "vllm" not in mods, name
