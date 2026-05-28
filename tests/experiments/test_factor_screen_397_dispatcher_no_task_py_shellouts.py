"""Round 9 + Round 11 canary: dispatcher MUST NOT shell out at all.

Round 9 was triggered by a production crash on pod-397: the dispatcher
shelled out to ``uv run python scripts/task.py find <N>`` to look up the
task directory + scan ``events.jsonl`` for the ``epm:smoke-pass`` row.
``task.py`` branch-guards to ``main`` (CLAUDE.md: "the canonical
resolver branch-guards to ``main`` and refuses loudly on detached HEAD
/ non-``main`` HEAD"); the pod-side checkout sits on ``issue-397`` →
``task.py`` exits non-zero → ``subprocess.run(check=True)`` raises →
dispatcher crashes before the 108-cell sweep launches.

Round 9 removed all ``task.py`` call sites from the dispatcher:

  - ``has_recent_smoke_pass_marker`` (line 837 pre-round-9) →
    ``is_smoke_pass_confirmed_locally`` (CLI flag + local
    ``metrics_final.json`` fallback).
  - ``post_marker_via_task_py`` smoke-end call →
    ``write_verdict_file(slab_root, "SMOKE_VERDICT.json", payload)``.
  - ``post_marker_via_task_py`` sweep-resume call →
    ``write_verdict_file(slab_root, "SWEEP_RESUME.json", payload)``.
  - ``post_marker_via_task_py`` helper itself → deleted.

The orchestrator on the VM side (where ``task.py`` works because it
runs from the ``main`` repo root) reads the verdict JSONs via SCP /
``ssh_download`` and posts the markers itself.

Round 11 deleted the subprocess wrapper entirely (per-cell ``python -m
run_one_cell``). Five rounds (5..10) of cascading bugs all traced back
to the subprocess crossing trust boundaries (env propagation, branch-
guard, upload silent-swallow). Round 11 in-processed the sweep — the
dispatcher now runs each cell end-to-end in its own process, reusing
the proven smoke pipeline. The ``run_one_cell`` module is gone; the
dispatcher must NOT re-introduce a subprocess that invokes it.

This test is the regression guard for both classes of regression:

  - Any future re-introduction of ``task.py`` invocation from the
    dispatcher fails CI loud (Round 9 contract).
  - Any future re-introduction of a ``subprocess.Popen`` /
    ``subprocess.run`` call that references ``run_one_cell`` (or
    re-adds the ``run_one_cell.py`` module) fails CI loud (Round 11
    contract).

CPU-only; pure static-file analysis.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_DISPATCH_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
)


def test_dispatcher_no_task_py_in_subprocess_calls() -> None:
    """No ``subprocess.run`` / ``subprocess.Popen`` / ``os.system`` /
    ``os.popen`` call in the dispatcher passes ``scripts/task.py`` as
    an argument.

    Walks the AST so docstring references to ``task.py`` (legitimate —
    explaining WHY round 9 removed the shellout) don't false-positive.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))

    offenders: list[tuple[int, str]] = []

    class _ShelloutVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func_name = _resolve_call_name(node.func)
            if func_name in {
                "subprocess.run",
                "subprocess.Popen",
                "subprocess.check_call",
                "subprocess.check_output",
                "subprocess.call",
                "os.system",
                "os.popen",
                "os.execvpe",
                "os.execve",
                "os.execvp",
            } and _references_task_py(node):
                offenders.append((node.lineno, ast.unparse(node)))
            self.generic_visit(node)

    _ShelloutVisitor().visit(tree)
    assert offenders == [], (
        f"Round 9 contract: dispatcher MUST NOT shell out to scripts/task.py "
        f"(pod runs on issue-397 branch; task.py branch-guards to main and "
        f"crashes the dispatcher). Found {len(offenders)} offender(s):\n"
        + "\n".join(f"  line {ln}: {expr[:120]}" for ln, expr in offenders)
    )


def test_dispatcher_does_not_define_post_marker_via_task_py() -> None:
    """The ``post_marker_via_task_py`` helper itself was deleted in
    Round 9. A future re-introduction (re-adding the helper, even if
    no caller exists yet) fails this canary.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))
    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "post_marker_via_task_py" not in func_names, (
        "Round 9 deleted post_marker_via_task_py — re-adding it (even "
        "unused) signals a regression. Use write_verdict_file instead "
        "and let the orchestrator post markers from the VM side."
    )


def test_dispatcher_does_not_define_has_recent_smoke_pass_marker() -> None:
    """The ``has_recent_smoke_pass_marker`` function (which shelled out
    to ``task.py find``) was deleted in Round 9. Replaced by
    ``is_smoke_pass_confirmed_locally``.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))
    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "has_recent_smoke_pass_marker" not in func_names, (
        "Round 9 deleted has_recent_smoke_pass_marker — use "
        "is_smoke_pass_confirmed_locally instead."
    )
    assert "is_smoke_pass_confirmed_locally" in func_names, (
        "Round 9 replacement is_smoke_pass_confirmed_locally must exist"
    )


def test_dispatcher_exposes_write_verdict_file() -> None:
    """The Round 9 marker-replacement helper ``write_verdict_file``
    must be a public top-level function.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))
    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "write_verdict_file" in func_names, (
        "Round 9 added write_verdict_file — must be present for the "
        "orchestrator to read SMOKE_VERDICT.json / SWEEP_RESUME.json "
        "and post markers from the VM side."
    )


def test_dispatcher_cli_exposes_smoke_pass_confirmed_flag() -> None:
    """The Round 9 CLI flag ``--smoke-pass-confirmed`` must be present
    in the argparse parser. Set by the orchestrator AFTER posting
    ``epm:smoke-pass v1`` from the VM side.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    # Simple substring check inside build_arg_parser — AST walk would
    # be overkill for a CLI flag presence assertion.
    assert "--smoke-pass-confirmed" in src, (
        "Round 9 CLI flag --smoke-pass-confirmed missing from dispatcher"
    )
    # Sanity: the flag wires into args.smoke_pass_confirmed (argparse
    # converts dashes to underscores).
    assert "smoke_pass_confirmed" in src, (
        "Round 9: dispatcher must reference args.smoke_pass_confirmed"
    )


# ---------------------------------------------------------------------------
# Round 11 — no subprocess to run_one_cell, no run_one_cell module
# ---------------------------------------------------------------------------


def test_dispatcher_no_subprocess_to_run_one_cell() -> None:
    """Round 11 contract: dispatcher MUST NOT spawn a subprocess that
    invokes ``run_one_cell`` (the deleted per-cell entrypoint).

    Five rounds (5..10) of cascading bugs all traced back to the
    subprocess crossing trust boundaries — Round 11 in-processed the
    sweep. Any future re-introduction of ``subprocess.Popen`` /
    ``subprocess.run`` / ``subprocess.call`` etc. that references
    ``run_one_cell`` in its argv fails this canary.

    Walks the AST so docstring references to ``run_one_cell.py`` (which
    explain WHY round 11 deleted it) don't false-positive.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))

    offenders: list[tuple[int, str]] = []

    class _ShelloutVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func_name = _resolve_call_name(node.func)
            if func_name in {
                "subprocess.run",
                "subprocess.Popen",
                "subprocess.check_call",
                "subprocess.check_output",
                "subprocess.call",
                "os.system",
                "os.popen",
                "os.execvpe",
                "os.execve",
                "os.execvp",
            } and _references_run_one_cell(node):
                offenders.append((node.lineno, ast.unparse(node)))
            self.generic_visit(node)

    _ShelloutVisitor().visit(tree)
    assert offenders == [], (
        "Round 11 contract: dispatcher MUST NOT spawn a subprocess "
        "referencing run_one_cell (the deleted per-cell entrypoint; in-"
        "process serial replaces it). Found "
        f"{len(offenders)} offender(s):\n"
        + "\n".join(f"  line {ln}: {expr[:120]}" for ln, expr in offenders)
    )


def test_run_one_cell_module_is_deleted() -> None:
    """Round 11 deleted the ``run_one_cell.py`` module entirely.

    Any future re-introduction (e.g. someone copies the round-10 file
    back in and adds a subprocess.Popen call site) signals a regression
    of the in-process serial pivot.
    """
    run_one_cell_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "factor_screen_397"
        / "run_one_cell.py"
    )
    assert not run_one_cell_path.exists(), (
        f"Round 11 deleted {run_one_cell_path} — it must not be re-added. "
        "If a subprocess wrapper is genuinely needed in the future, design "
        "it from scratch with explicit env propagation, no branch-guarded "
        "shellouts, and fail-loud upload paths."
    )


def test_dispatcher_does_not_import_run_one_cell() -> None:
    """Round 11: dispatcher must NOT import from
    ``explore_persona_space.experiments.factor_screen_397.run_one_cell``
    (the deleted module).
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.endswith("factor_screen_397.run_one_cell")
        ):
            raise AssertionError(
                f"Round 11: dispatcher must NOT import from "
                f"{node.module} (the deleted module). Line {node.lineno}."
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("factor_screen_397.run_one_cell"):
                    raise AssertionError(
                        f"Round 11: dispatcher must NOT import "
                        f"{alias.name} (the deleted module). Line {node.lineno}."
                    )


def test_dispatcher_exposes_two_pass_helpers() -> None:
    """Round 12 added three top-level helpers replacing the Round 11
    serial per-cell pipeline:

      - ``_run_pass1_hf`` — HF-only pass (train + log-prob eval).
      - ``_run_pass2_vllm`` — vLLM-only pass (LoRA-swap per cell).
      - ``_run_sweep_two_pass`` — orchestrates the two passes with ONE
        ``_aggressive_hf_to_vllm_teardown`` event between them.

    Also adds ``is_cell_pass1_complete`` for two-pass resume tracking.
    The dispatcher continues to expose ``verify_adapter_on_hf_hub`` and
    ``cleanup_cell_local_weights`` (lifted in Round 11). This test
    asserts all five are present as top-level defs.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))
    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for required in (
        "_run_pass1_hf",
        "_run_pass2_vllm",
        "_run_sweep_two_pass",
        "is_cell_pass1_complete",
        "verify_adapter_on_hf_hub",
        "cleanup_cell_local_weights",
    ):
        assert required in func_names, (
            f"Round 12: dispatcher must expose {required!r} as a top-level function "
            "(the two-pass sweep + Pass-1 resume probe + lifted upload-policy helpers)."
        )


def test_dispatcher_drops_subprocess_pool_and_round11_serial_helpers() -> None:
    """Round 12 keeps Round 11's deletion of subprocess-pool helpers AND
    additionally deletes the Round 11 in-process serial helpers, since
    the two-pass design replaces them.

    Re-adding any of these signals a regression of the two-pass pivot.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))
    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for forbidden in (
        # Round 11 subprocess-pool deletions (Round 5..10 path).
        "_dispatch_sweep_jobs",
        "_launch_cell_subprocess",
        "_wait_for_free_gpu",
        "build_run_one_cell_command",
        # Round 12 serial-path deletions (Round 11 path).
        "_run_one_cell_inprocess",
        "_run_sweep_serial",
    ):
        assert forbidden not in func_names, (
            f"Round 12 (or Round 11) deleted {forbidden!r} — re-adding it (even unused) "
            "signals a regression of the two-pass pivot. Reuse _run_pass1_hf + "
            "_run_pass2_vllm + _run_sweep_two_pass instead."
        )


def test_dispatcher_cli_drops_concurrency_flags() -> None:
    """Round 11 removed ``--num-gpus`` and ``--max-concurrent-train``
    from the dispatcher CLI (in-process serial doesn't need them).

    Re-adding either signals someone is trying to bolt the subprocess
    pool back on. Check for the exact CLI surface, not just the substring
    — the docstring + comments retain references explaining the removal.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    # The argparse helpers use p.add_argument("--num-gpus", ...) or
    # p.add_argument("--max-concurrent-train", ...). Walk the AST for
    # Call nodes whose first arg is exactly one of these strings.
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                assert first.value not in ("--num-gpus", "--max-concurrent-train"), (
                    f"Round 11 removed CLI flag {first.value!r}; re-adding it "
                    "signals a regression. The in-process serial sweep doesn't "
                    "need GPU-pool concurrency; CUDA_VISIBLE_DEVICES set at "
                    "launch time pins the visible GPUs."
                )


def test_dispatcher_loads_dotenv_at_entry() -> None:
    """Round 11 added ``load_dotenv()`` at the top of ``main()`` so
    HF_TOKEN / WANDB_API_KEY / ANTHROPIC_API_KEY are in ``os.environ``
    before any HF Hub / WandB / Anthropic call.

    Without this, pod-side launches (where the shell isn't a login shell
    sourcing .env) silently miss the tokens; HF Hub uploads silently
    fail; verify_adapter_on_hf_hub returns False; rc=2 → local weights
    pile up and the MooseFS quota blows.

    Brief cited ``setup_env()`` from utils.py but that function doesn't
    exist; ``load_dotenv`` from ``orchestrate.env`` is the canonical
    helper used elsewhere (factor_screen_365.__main__).
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    # Substring match is sufficient — the canonical import + call lands
    # in main() and is the only place the helper is referenced.
    assert "from explore_persona_space.orchestrate.env import load_dotenv" in src, (
        "Round 11: dispatcher must import load_dotenv from "
        "explore_persona_space.orchestrate.env at the top of main()."
    )
    assert "load_dotenv()" in src, (
        "Round 11: dispatcher must CALL load_dotenv() at entry — importing "
        "without calling defeats the purpose."
    )


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _resolve_call_name(node: ast.AST) -> str:
    """Resolve a Call's function to a dotted name (e.g. ``subprocess.run``).

    Handles ``Name`` (bare call), ``Attribute`` (one-level dot),
    ``Attribute`` of ``Name`` (two-level). Returns empty string for
    anything more complex (lambda call, subscripted attr, etc.) —
    those won't false-positive on the shellout patterns we care about.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        return f"{_resolve_call_name(node.value)}.{node.attr}"
    return ""


def _references_task_py(node: ast.Call) -> bool:
    """Return True if any string literal inside ``node`` ends in
    ``task.py`` (the dispatcher's broken shellout target).

    Matches both ``"scripts/task.py"`` and bare ``"task.py"``.
    """
    pattern = re.compile(r"(^|/)task\.py$")
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and pattern.search(child.value)
        ):
            return True
    return False


def _references_run_one_cell(node: ast.Call) -> bool:
    """Return True if any string literal inside ``node`` contains
    ``run_one_cell`` (the deleted per-cell entrypoint Round 11 removed).

    Matches both ``"explore_persona_space.experiments.factor_screen_397.run_one_cell"``
    (the ``python -m ...`` form) and bare ``"run_one_cell"``.
    """
    pattern = re.compile(r"run_one_cell")
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and pattern.search(child.value)
        ):
            return True
    return False
