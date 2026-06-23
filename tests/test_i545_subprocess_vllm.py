"""Regression for the #545 STRATEGY E refactor — subprocess vLLM isolation.

After SIX OOMs of co-residency strategies (rounds 1/3/4/6/8 tuning
gpu_memory_utilization / HF teacher-force max_batch / per-layer hooks, then the
round-37 HALT because the JS probes are genuinely up to 5631 tokens so a
max_seq_len reduction would corrupt correctness), the genuinely-correct fix is
to SEQUENCE HF and vLLM into phases so they never co-reside on the H100:

  Phase A — ``vllm_worker.py`` runs as a SUBPROCESS, owns the full GPU, samples
    all on-policy responses (nl-cloud + outdist pairs) to disk, then EXITS.
  Phase B — the HF base model loads (sole GPU resident now), reads the cached
    responses, teacher-forces + extracts hidden states, then frees the GPU.

These tests run WITHOUT a GPU (the real path needs a 7B model on an H100). They
verify (1) the file-based IPC request/response dance via a STUB worker, (2) the
client correctly spawns + reaps the subprocess and fails LOUD on a dead worker,
(3) the production module no longer co-resides vLLM in-process (the removed
in-process ``LLM(...)`` / pre-init-assert path), and (4) the round-10
silent-truncation fix (JS_MAX_SEQ_LEN raised to 8192, mirrored in the worker).
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from explore_persona_space.experiments.behavior_testbed_545 import predictors_zoo as zoo
from explore_persona_space.experiments.behavior_testbed_545 import vllm_worker as worker

_PREDICTORS_ZOO_PATH = Path(inspect.getfile(zoo))
_WORKER_PATH = Path(inspect.getfile(worker))


# ---------------------------------------------------------------------------
# 1. JS_MAX_SEQ_LEN raised to clear the 5631-token + 1024-max-new worst case,
#    and mirrored in the worker engine's max_model_len (round-10 latent bug).
# ---------------------------------------------------------------------------


def test_js_max_seq_len_raised_to_8192():
    # round-37 measured prompts up to 5631 tokens; + JS_MAX_NEW_TOKENS=1024
    # sampled = ~6655 worst case. 4096 silently truncated; 8192 clears it.
    assert zoo.JS_MAX_SEQ_LEN == 8192
    assert zoo.JS_MAX_SEQ_LEN >= 5631 + zoo.JS_MAX_NEW_TOKENS


def test_worker_max_model_len_matches_js_max_seq_len():
    """The engine's capacity must match the value used to BUILD prompts, else a
    prompt sized to JS_MAX_SEQ_LEN would be clamped by a smaller engine cap."""
    assert worker.WORKER_MAX_MODEL_LEN == zoo.JS_MAX_SEQ_LEN


def test_worker_gpu_util_is_high_sole_resident():
    """The worker owns the GPU alone (no co-residency), so it can use a high
    utilization. Must be a sane fraction well above the old co-residency 0.50."""
    assert 0.75 <= worker.WORKER_GPU_MEM_UTIL <= 0.95


# ---------------------------------------------------------------------------
# 2. The production module no longer loads vLLM in-process (co-residency gone).
# ---------------------------------------------------------------------------


def test_extract_fn_does_not_construct_vllm_in_process():
    """``extract_clouds_and_outdist_gpu`` must NOT construct an in-process
    ``LLM(...)`` engine — that co-residency was the OOM cause Strategy E removes.
    vLLM now lives only in the subprocess worker."""
    src = inspect.getsource(zoo.extract_clouds_and_outdist_gpu)
    tree = ast.parse(textwrap.dedent(src))
    llm_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "LLM"
    ]
    assert not llm_calls, "extract_clouds_and_outdist_gpu must not construct LLM(...) in-process"


def test_removed_inprocess_coresidency_symbols_are_gone():
    """The in-process co-residency dial + pre-init free-memory assert are removed
    (their job moved to the subprocess). Their absence is the architectural
    signal that the co-residency path is gone."""
    for stale in (
        "JS_GPU_MEM_UTIL",
        "JS_HF_MODEL_RESIDENT_GIB",
        "JS_VLLM_PREINIT_MIN_FREE_GIB",
    ):
        assert not hasattr(zoo, stale), f"stale co-residency symbol still present: {stale}"


def test_extract_fn_uses_the_subprocess_client():
    """The extract path must dispatch sampling through the _VllmClient subprocess
    helper (Phase A) before loading the HF model (Phase B)."""
    src = inspect.getsource(zoo.extract_clouds_and_outdist_gpu)
    assert "_VllmClient" in src
    # Phase ordering: the vLLM client run() must precede the HF model load.
    client_run_idx = src.find(".run()")
    hf_load_idx = src.find("AutoModelForCausalLM.from_pretrained")
    assert client_run_idx != -1, "no client.run() in extract path"
    assert hf_load_idx != -1, "no HF model load in extract path"
    assert client_run_idx < hf_load_idx, "vLLM sampling (Phase A) must precede HF load (Phase B)"


def test_worker_module_defines_serve_and_main():
    assert hasattr(worker, "main")
    assert hasattr(worker, "_serve")
    assert hasattr(worker, "_atomic_write_json")


# ---------------------------------------------------------------------------
# 3. The file-based IPC dance, exercised end-to-end against a STUB worker (no GPU
#    / no vLLM). A tiny stub worker script reads requests, echoes deterministic
#    token-ids, writes responses — the EXACT contract the real worker honors.
# ---------------------------------------------------------------------------

_STUB_WORKER = textwrap.dedent(
    """
    import json, sys, time
    from pathlib import Path

    def main():
        ipc = Path(sys.argv[sys.argv.index("--ipc-dir") + 1])
        req_dir = ipc / "requests"
        resp_dir = ipc / "responses"
        resp_dir.mkdir(parents=True, exist_ok=True)
        ready_seen = None
        while True:
            if (ipc / "STOP").exists():
                return 0
            pending = [
                p for p in sorted(req_dir.glob("*.json"))
                if not (resp_dir / p.name).exists()
            ]
            if pending:
                import os
                for p in pending:
                    req = json.loads(p.read_text())
                    # Echo: each prompt gets `n` completions of fixed token-ids.
                    comp = {"token_ids": [10, 20, 30], "text": "stub-resp", "finish_reason": "stop"}
                    completions = [
                        [dict(comp) for _ in range(req["n"])]
                        for _ in req["prompt_token_ids"]
                    ]
                    payload = {"probe_id": req["probe_id"], "completions": completions}
                    tmp = (resp_dir / p.name).with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(payload))
                    os.replace(tmp, resp_dir / p.name)
                continue
            if (ipc / "READY").exists():
                if ready_seen is None:
                    ready_seen = time.monotonic()
                elif time.monotonic() - ready_seen >= 0.5:
                    return 0
            time.sleep(0.05)

    sys.exit(main())
    """
)


def _stub_worker_argv(tmp_path: Path, ipc_dir: Path) -> list[str]:
    stub = tmp_path / "stub_worker.py"
    stub.write_text(_STUB_WORKER)
    return [sys.executable, str(stub), "--ipc-dir", str(ipc_dir)]


def test_client_request_response_round_trip(tmp_path):
    ipc_dir = tmp_path / "ipc"
    client = zoo._VllmClient(ipc_dir, worker_argv=_stub_worker_argv(tmp_path, ipc_dir))
    client.add_request("nl|row_a", [[1, 2, 3]], n=2, max_tokens=16)
    client.add_request("outdist|r|c|nl|a", [[4, 5], [6, 7]], n=3, max_tokens=32)
    results = client.run()

    assert set(results) == {"nl|row_a", "outdist|r|c|nl|a"}
    # nl request: 1 prompt, n=2 completions.
    assert len(results["nl|row_a"]) == 1
    assert len(results["nl|row_a"][0]) == 2
    assert results["nl|row_a"][0][0]["token_ids"] == [10, 20, 30]
    assert results["nl|row_a"][0][0]["finish_reason"] == "stop"
    # outdist request: 2 prompts, n=3 completions each.
    assert len(results["outdist|r|c|nl|a"]) == 2
    assert all(len(prompt_comps) == 3 for prompt_comps in results["outdist|r|c|nl|a"])
    # The worker must have exited (STOP dropped + reaped).
    assert (ipc_dir / "STOP").exists()
    assert client._proc is None


def test_client_empty_requests_is_noop(tmp_path):
    ipc_dir = tmp_path / "ipc"
    client = zoo._VllmClient(ipc_dir, worker_argv=_stub_worker_argv(tmp_path, ipc_dir))
    # No requests added → run() returns {} WITHOUT spawning the worker.
    assert client.run() == {}
    assert client._proc is None


def test_client_reuses_existing_response_files(tmp_path):
    """A response file already on disk (from a prior partial phase) is reused;
    only the missing request needs the worker (checkpoint-per-phase resilience)."""
    ipc_dir = tmp_path / "ipc"
    (ipc_dir / "responses").mkdir(parents=True)
    # Pre-seed one response as if a prior run wrote it.
    (ipc_dir / "responses" / "nl|preexisting.json").write_text(
        json.dumps(
            {
                "probe_id": "nl|preexisting",
                "completions": [[{"token_ids": [99], "text": "cached", "finish_reason": "stop"}]],
            }
        )
    )
    client = zoo._VllmClient(ipc_dir, worker_argv=_stub_worker_argv(tmp_path, ipc_dir))
    client.add_request("nl|preexisting", [[1]], n=1, max_tokens=8)
    client.add_request("nl|fresh", [[2]], n=1, max_tokens=8)
    results = client.run()
    assert results["nl|preexisting"][0][0]["text"] == "cached"  # not overwritten
    assert results["nl|fresh"][0][0]["text"] == "stub-resp"  # produced by worker


def test_client_fails_loud_on_worker_error(tmp_path):
    """A worker that writes worker.error must make the client raise LOUD, not
    hang waiting for responses that will never come (CLAUDE.md fail-fast)."""
    ipc_dir = tmp_path / "ipc"
    crash_worker = tmp_path / "crash_worker.py"
    crash_worker.write_text(
        textwrap.dedent(
            """
            import json, sys
            from pathlib import Path
            ipc = Path(sys.argv[sys.argv.index("--ipc-dir") + 1])
            ipc.mkdir(parents=True, exist_ok=True)
            (ipc / "worker.error").write_text(json.dumps({"error": "boom", "traceback": "..."}))
            sys.exit(2)
            """
        )
    )
    client = zoo._VllmClient(
        ipc_dir, worker_argv=[sys.executable, str(crash_worker), "--ipc-dir", str(ipc_dir)]
    )
    client.add_request("nl|x", [[1]], n=1, max_tokens=8)
    with pytest.raises(RuntimeError, match="vLLM worker"):
        client.run()


def test_client_reaps_worker_on_polling_exception(tmp_path):
    """BLOCKER (round 39): the GPU-owning worker must be terminated + reaped on
    EVERY ``run()`` exit path. Previously ``self.close()`` ran only on the success
    path, so a polling exception (timeout / worker-error / worker death) leaked a
    subprocess holding the GPU — the exact co-residency OOM Strategy E removes.

    A stub worker that NEVER writes responses and never exits forces
    ``_poll_for_responses`` to TimeoutError; after the raise the subprocess must be
    gone (``_proc is None`` AND no live PID)."""
    ipc_dir = tmp_path / "ipc"
    hang_worker = tmp_path / "hang_worker.py"
    # Records its own PID, then sleeps forever (ignoring requests). It exits ONLY
    # when the client's close() drops STOP / terminates it — proving the reap ran.
    hang_worker.write_text(
        textwrap.dedent(
            """
            import os, sys, time
            from pathlib import Path
            ipc = Path(sys.argv[sys.argv.index("--ipc-dir") + 1])
            ipc.mkdir(parents=True, exist_ok=True)
            (ipc / "worker.pid").write_text(str(os.getpid()))
            while True:
                time.sleep(0.05)
            """
        )
    )
    client = zoo._VllmClient(
        ipc_dir, worker_argv=[sys.executable, str(hang_worker), "--ipc-dir", str(ipc_dir)]
    )
    client.add_request("nl|x", [[1]], n=1, max_tokens=8)

    # Tight timeout so the poll fails fast in-test (default is 3600s).
    monkeypatch_done = False
    orig_timeout = zoo.VLLM_WORKER_TIMEOUT_S
    orig_poll = zoo.VLLM_POLL_INTERVAL_S
    try:
        zoo.VLLM_WORKER_TIMEOUT_S = 1.0
        zoo.VLLM_POLL_INTERVAL_S = 0.05
        with pytest.raises(TimeoutError):
            client.run()
        monkeypatch_done = True
    finally:
        zoo.VLLM_WORKER_TIMEOUT_S = orig_timeout
        zoo.VLLM_POLL_INTERVAL_S = orig_poll
    assert monkeypatch_done

    # The worker MUST have been reaped: _proc cleared + no live PID.
    assert client._proc is None
    worker_pid = int((ipc_dir / "worker.pid").read_text())
    # os.kill(pid, 0) raises ProcessLookupError if the process is gone.
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


def test_run_clears_stale_sentinels(tmp_path):
    """BLOCKER (round 39): a same-out_dir retry must clear stale STOP / READY /
    worker.error from a prior run BEFORE spawning, else the fresh worker either
    exits on the stale STOP without processing the new requests, or the first poll
    fails LOUD on the stale worker.error. Pre-create all three; the run must still
    succeed against the stub worker."""
    ipc_dir = tmp_path / "ipc"
    ipc_dir.mkdir(parents=True)
    (ipc_dir / "STOP").write_text("1")
    (ipc_dir / "READY").write_text("1")
    (ipc_dir / "worker.error").write_text('{"error": "stale from a prior run"}')

    client = zoo._VllmClient(ipc_dir, worker_argv=_stub_worker_argv(tmp_path, ipc_dir))
    client.add_request("nl|fresh", [[1, 2, 3]], n=1, max_tokens=8)
    results = client.run()  # must NOT fail-loud on the stale worker.error / STOP

    assert results["nl|fresh"][0][0]["text"] == "stub-resp"
    # The fresh run re-dropped STOP (its own close()), but the stale worker.error
    # was cleared before launch.
    assert not (ipc_dir / "worker.error").exists()


def test_run_clears_stale_sentinels_preserves_reused_responses(tmp_path):
    """Clearing stale CONTROL sentinels must NOT delete pre-existing RESPONSE
    files — the documented re-run reuse contract (a prior partial phase's
    responses are reused) depends on them surviving the sentinel sweep."""
    ipc_dir = tmp_path / "ipc"
    (ipc_dir / "responses").mkdir(parents=True)
    (ipc_dir / "STOP").write_text("1")  # stale control sentinel
    (ipc_dir / "responses" / "nl|cached.json").write_text(
        json.dumps(
            {
                "probe_id": "nl|cached",
                "completions": [
                    [{"token_ids": [7], "text": "from-prior", "finish_reason": "stop"}]
                ],
            }
        )
    )
    client = zoo._VllmClient(ipc_dir, worker_argv=_stub_worker_argv(tmp_path, ipc_dir))
    client.add_request("nl|cached", [[1]], n=1, max_tokens=8)
    client.add_request("nl|fresh", [[2]], n=1, max_tokens=8)
    results = client.run()
    # The pre-existing response survived the stale-sentinel sweep and was reused.
    assert results["nl|cached"][0][0]["text"] == "from-prior"
    assert results["nl|fresh"][0][0]["text"] == "stub-resp"


def test_client_duplicate_probe_id_raises(tmp_path):
    client = zoo._VllmClient(tmp_path / "ipc")
    client.add_request("dup", [[1]], n=1, max_tokens=8)
    with pytest.raises(KeyError, match="duplicate"):
        client.add_request("dup", [[2]], n=1, max_tokens=8)


def test_client_default_worker_argv_invokes_the_module():
    """The default launch must invoke the real vllm_worker module via -m so the
    production spawn path is the one tested by the smoke run."""
    client = zoo._VllmClient(Path("/tmp/i545_argv_check"))
    argv = client._worker_argv
    assert argv[0] == sys.executable
    assert "-m" in argv
    assert "explore_persona_space.experiments.behavior_testbed_545.vllm_worker" in argv
    assert "--ipc-dir" in argv


# ---------------------------------------------------------------------------
# 4. _samples_from_completions maps the worker payload to the #540 sampled shape.
# ---------------------------------------------------------------------------


def test_samples_from_completions_maps_and_counts_truncation():
    comps = [
        [
            {"token_ids": [1, 2], "text": "a", "finish_reason": "stop"},
            {"token_ids": [3, 4, 5], "text": "b", "finish_reason": "length"},
        ],
        [
            {"token_ids": [6], "text": "c", "finish_reason": "stop"},
        ],
    ]
    per_probe, t_total, t_hit = zoo._samples_from_completions(comps, r_samples=8)
    assert len(per_probe) == 2
    assert len(per_probe[0]) == 2
    assert t_total == 3  # 2 + 1 completions
    assert t_hit == 1  # one finish_reason == "length"
    # apply_terminator_rule appends <|im_end|> (151645) to the natural-stop ones.
    assert per_probe[0][0] == [1, 2, 151645]
    # The truncated one (finish_reason=length) is NOT terminator-appended.
    assert per_probe[0][1] == [3, 4, 5]


# ---------------------------------------------------------------------------
# 4b. _score_outdist_pair_from_samples fails LOUD on a partial / corrupt IPC
#     payload instead of silently min()-truncating below the registered estimand
#     (BLOCKER fix, round 39). The shape asserts fire BEFORE any model call, so a
#     None model is fine — the function must raise before touching it.
# ---------------------------------------------------------------------------


def _comp(token_ids):
    return {"token_ids": list(token_ids), "text": "x", "finish_reason": "stop"}


def test_score_outdist_pair_fails_loud_on_missing_probe():
    """A worker that dropped a probe (comps shorter than the prompt list) must
    raise, not silently score the surviving probes."""
    prompts_a = [[1, 2], [3, 4]]  # 2 probes registered
    prompts_b = [[5, 6], [7, 8]]
    comps_a = [[_comp([9])]]  # only 1 probe came back → truncated
    comps_b = [[_comp([9])], [_comp([9])]]
    with pytest.raises(ValueError, match="Phase-A output truncated"):
        zoo._score_outdist_pair_from_samples(
            None, prompts_a, prompts_b, comps_a, comps_b, r_samples=1
        )


def test_score_outdist_pair_fails_loud_on_short_sample_count():
    """A probe whose completion list is shorter than r_samples (a partial sample
    set) must raise, not score fewer samples than the registered R."""
    prompts_a = [[1, 2]]
    prompts_b = [[5, 6]]
    # r_samples=8 demanded, but side a returned only 2 completions for the probe.
    comps_a = [[_comp([9]), _comp([10])]]
    comps_b = [[_comp([9]) for _ in range(8)]]
    with pytest.raises(ValueError, match="responses, need >= 8"):
        zoo._score_outdist_pair_from_samples(
            None, prompts_a, prompts_b, comps_a, comps_b, r_samples=8
        )


def test_score_outdist_pair_no_silent_min_truncation_symbol_gone():
    """The silent ``n_probes_eff = min(...)`` truncation that scored below the
    registered estimand must be gone from the source (it was the BLOCKER)."""
    src = inspect.getsource(zoo._score_outdist_pair_from_samples)
    assert "n_probes_eff = min(" not in src
    assert "n_probes_used = len(prompts_a)" in src


# ---------------------------------------------------------------------------
# 5. Allocator env var (expandable_segments) still set BEFORE the first torch
#    import (kept for the HF hook path under Strategy E).
# ---------------------------------------------------------------------------


def _alloc_conf_setdefault_lineno(tree: ast.Module) -> int | None:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "PYTORCH_CUDA_ALLOC_CONF"
        ):
            return node.lineno
    return None


_DISPATCHER_PATH = _PREDICTORS_ZOO_PATH.parents[3].parent / "scripts" / "issue545_metric_race.py"


@pytest.mark.parametrize(
    "path",
    [
        _PREDICTORS_ZOO_PATH,
        # scripts/issue545_metric_race.py is the (out-of-scope) #545 metric-race
        # dispatcher; it stays on the issue-545 branch in the source/test
        # forward-port, so this parametrization skips until it lands.
        pytest.param(
            _DISPATCHER_PATH,
            marks=pytest.mark.skipif(
                not _DISPATCHER_PATH.is_file(),
                reason="scripts/issue545_metric_race.py out of forward-port scope "
                "(on issue-545 branch)",
            ),
        ),
    ],
    ids=["predictors_zoo", "dispatcher"],
)
def test_alloc_conf_still_set_before_torch(path):
    assert path.is_file(), f"source file missing: {path}"
    tree = ast.parse(path.read_text())
    setdefault_lineno = _alloc_conf_setdefault_lineno(tree)
    assert setdefault_lineno is not None, f"{path.name}: PYTORCH_CUDA_ALLOC_CONF setdefault missing"
    torch_imports = [
        n.lineno
        for n in ast.walk(tree)
        if (isinstance(n, ast.Import) and any(a.name.startswith("torch") for a in n.names))
        or (isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("torch"))
    ]
    for ln in torch_imports:
        assert setdefault_lineno < ln, (
            f"{path.name}: PYTORCH_CUDA_ALLOC_CONF setdefault must precede import torch"
        )
