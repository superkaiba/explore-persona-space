"""Regression tests for the ``--issue`` plumbing across dispatcher, __main__, and training.

Task #383 plumbing (plan v2 §5a): parent #365's recipe-fix branch
(``task-365-recipe-fix-v1`` @ ``32ce24ef``) hardcoded ``i365`` /
``adapters/issue_365/`` across ``training.train_one_cell``,
``scripts/dispatch_factor_screen_365.hf_hub_adapter_run_name`` /
``cell_complete_on_hub`` / ``_prefetch_hub_adapter_index``, and the
``_should_skip_cell`` resume probe. Launching the dispatcher for issue
383 against the unchanged code would have the resume probe find all 72
adapters parent #365 already uploaded to ``adapters/issue_365/``,
short-circuit every cell with ``cell_complete_on_hub() == True``, and
produce zero ``metrics.json`` files — the catastrophic plumbing failure
the fact-checker flagged in plan v1.

These tests assert the post-fix behavior:

  1. ``training.train_one_cell`` accepts ``run_name_prefix`` and
     ``hf_path_prefix`` as REQUIRED keyword arguments (no defaults).
     Supplying ``run_name_prefix="i383"`` yields a run-name beginning
     with ``"i383_"``; supplying ``hf_path_prefix="adapters/issue_383"``
     yields ``TrainLoraConfig.hf_path_in_repo`` beginning with
     ``"adapters/issue_383/"``.
  2. ``__main__.parse_args`` requires ``--issue``; passing
     ``--issue 383`` makes the resulting Namespace carry ``issue=383``.
  3. The dispatcher's ``_training_cmd`` forwards ``--issue`` to every
     cell-train / cell-eval child subprocess argv unchanged.
  4. The dispatcher's ``cell_complete_on_hub`` probes
     ``adapters/issue_{issue}/`` (not the hardcoded ``issue_365``); a
     parent #365 adapter in the hub-files cache must NOT cause a cell
     to be skipped when ``--issue 383`` is in flight.

The tests use the existing ``_load_dispatch_module`` importlib pattern
from ``test_factor_screen_365_resume.py`` so the script under
``scripts/`` (not on the package path) loads cleanly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

from explore_persona_space.experiments.factor_screen_365 import training
from explore_persona_space.experiments.factor_screen_365.__main__ import parse_args
from explore_persona_space.experiments.factor_screen_365.cells import Cell


def _load_dispatch_module():
    """Load ``scripts/dispatch_factor_screen_365`` as a module.

    Mirrors the pattern in ``test_factor_screen_365_resume.py`` — the
    script is hyphen-friendly under ``scripts/``, not on the package
    path, so importlib is the cleanest way to exercise it in tests.
    """
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts" / "dispatch_factor_screen_365.py"
    spec = importlib.util.spec_from_file_location("dispatch_factor_screen_365", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- training.train_one_cell plumbing ---------------------------------------


def test_train_one_cell_requires_run_name_prefix_and_hf_path_prefix() -> None:
    """``train_one_cell`` must REQUIRE ``run_name_prefix`` and ``hf_path_prefix``.

    The historical signature defaulted ``run_name_prefix="i365"`` and
    derived the Hub path from the hardcoded ``adapters/issue_365/``
    template. Making both required (no default) is the contract that
    forces every caller to declare an issue scope, eliminating the
    accidental-cross-issue-Hub-write footgun the fact-checker flagged.
    """
    import inspect

    sig = inspect.signature(training.train_one_cell)
    run_param = sig.parameters["run_name_prefix"]
    hf_param = sig.parameters["hf_path_prefix"]
    assert run_param.default is inspect.Parameter.empty, (
        "run_name_prefix must be REQUIRED (no default). Adding a default "
        "re-introduces the accidental-cross-issue-Hub-write footgun."
    )
    assert hf_param.default is inspect.Parameter.empty, (
        "hf_path_prefix must be REQUIRED (no default). Adding a default "
        "re-introduces the accidental-cross-issue-Hub-write footgun."
    )


def test_train_one_cell_run_name_uses_supplied_prefix(monkeypatch, tmp_path: Path) -> None:
    """When the caller passes ``run_name_prefix="i383"``, the run_name begins with ``i383_``.

    Stubs out the heavy ``train_lora`` + ``merge_lora`` subprocess so the
    test runs without GPUs, then asserts the ``TrainLoraConfig`` handed
    to ``train_lora`` carries the right ``run_name`` and
    ``hf_path_in_repo``.
    """
    captured: dict[str, object] = {}

    def fake_train_lora(*, base_model_path, data_path, output_dir, cfg):
        captured["cfg"] = cfg
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return Path(output_dir), 0.5

    monkeypatch.setattr(
        "explore_persona_space.train.sft.train_lora",
        fake_train_lora,
    )

    # Avoid the merge_lora subprocess (no GPU, no model weights).
    monkeypatch.setattr(
        "subprocess.call",
        lambda *_a, **_kw: 0,
    )

    data_path = tmp_path / "fake_data.jsonl"
    data_path.write_text("{}\n")  # 1 row keeps _count_lines / total_steps math sane.

    cell = Cell(a=0, b=1, c=0, d=0, e=0)
    training.train_one_cell(
        cell=cell,
        seed=42,
        source="librarian",
        data_path=data_path,
        cell_output_dir=tmp_path / "out",
        run_name_prefix="i383",
        hf_path_prefix="adapters/issue_383",
    )
    cfg = captured["cfg"]
    expected_run = "i383_cell_01000_source_librarian_seed42"
    assert cfg.run_name == expected_run, (
        f"run_name {cfg.run_name!r} should equal {expected_run!r} "
        f"(prefix={cfg.run_name.split('_cell_')[0]!r} — must be 'i383' not 'i365')"
    )
    expected_path = f"adapters/issue_383/{expected_run}"
    assert cfg.hf_path_in_repo == expected_path, (
        f"hf_path_in_repo {cfg.hf_path_in_repo!r} should equal "
        f"{expected_path!r} (must be under adapters/issue_383/, NOT "
        f"adapters/issue_365/ — parent #365's Hub namespace)"
    )
    assert "i365" not in cfg.run_name
    assert "issue_365" not in cfg.hf_path_in_repo


# ---- __main__.parse_args plumbing -------------------------------------------


def test_main_rejects_cell_train_mode_without_issue(tmp_path: Path) -> None:
    """``main()`` must REJECT cell-train invocations missing ``--issue``.

    The historical contract took ``run_name_prefix`` from a hardcoded
    default; adding ``--issue`` as required at the mode-router forces the
    dispatcher to be explicit about which issue's namespace each
    subprocess writes to. Validation is at the mode-router (not at
    argparse) so ``--mode help-cells`` and the legacy ``--mode cell``
    rejection path still produce their canonical error messages instead
    of "missing --issue".
    """
    from explore_persona_space.experiments.factor_screen_365.__main__ import main

    argv = [
        "--mode",
        "cell-train",
        "--cell",
        "01000",
        "--source",
        "librarian",
        "--seed",
        "42",
        "--output-dir",
        str(tmp_path / "out"),
        "--pool-dir",
        str(tmp_path / "pools"),
    ]
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    msg = str(exc_info.value)
    assert "--issue" in msg, (
        f"SystemExit message should mention --issue; got {msg!r}. "
        f"Validation must fire at the mode-router (plan v2 §5a)."
    )


def test_parse_args_carries_issue_383() -> None:
    """``--issue 383`` populates ``args.issue == 383`` on the Namespace."""
    argv = [
        "--mode",
        "cell-train",
        "--issue",
        "383",
        "--cell",
        "01000",
        "--source",
        "librarian",
        "--seed",
        "42",
        "--output-dir",
        "/tmp/out",
        "--pool-dir",
        "/tmp/pools",
    ]
    args = parse_args(argv)
    assert args.issue == 383, (
        f"args.issue should be the int 383; got {args.issue!r} (type={type(args.issue).__name__})"
    )


# ---- dispatcher: _training_cmd forwards --issue to subprocess ---------------


def test_training_cmd_forwards_issue_to_subprocess_argv() -> None:
    """The dispatcher's per-cell ``_training_cmd`` must include ``--issue <N>`` in argv.

    Without this, a child cell-train / cell-eval subprocess hits
    ``parse_args`` (now requires ``--issue``) and crashes immediately
    with a non-zero rc — i.e. the failure is loud, but the dispatcher
    would have been wired wrong. This test pins the forward.
    """
    mod = _load_dispatch_module()
    for phase in ("cell-train", "cell-eval"):
        cmd = mod._training_cmd(
            cell_key="01000",
            source="librarian",
            seed=42,
            issue=383,
            pool_dir=Path("/tmp/pools"),
            slab_root=Path("/tmp/slab"),
            mode=phase,
        )
        assert "--issue" in cmd, (
            f"_training_cmd({phase}) argv missing --issue: {cmd!r}; "
            f"child subprocess will crash at parse_args"
        )
        idx = cmd.index("--issue")
        assert cmd[idx + 1] == "383", (
            f"_training_cmd({phase}) argv has --issue but wrong value: "
            f"{cmd[idx + 1]!r}; expected '383'"
        )
        # Defense-in-depth: --mode must come BEFORE --issue (or the
        # child argparser's --mode choice handling stays consistent).
        assert "--mode" in cmd and cmd.index("--mode") < idx


# ---- dispatcher: cell_complete_on_hub probes issue-scoped Hub path ---------


def test_cell_complete_on_hub_probes_issue_383_namespace_not_issue_365() -> None:
    """``cell_complete_on_hub(issue=383)`` must NOT match parent #365's adapters.

    This is the load-bearing assertion: parent #365's 72 adapters
    already live at ``adapters/issue_365/i365_cell_*``. Without the
    plumbing fix the hardcoded probe would match them and short-circuit
    every cell in the issue-383 dispatcher invocation. With the fix the
    issue-383 probe targets ``adapters/issue_383/i383_cell_*`` and the
    parent's adapters are correctly ignored.
    """
    mod = _load_dispatch_module()
    # Simulate the Hub state at issue-383 launch: parent #365's 72
    # adapters present, issue_383 namespace empty.
    parent_hub_index = [
        f"adapters/issue_365/i365_cell_{key}_source_{src}_seed42/adapter_config.json"
        for key in ("01000", "01010", "11000")
        for src in ("librarian", "surgeon", "programmer")
    ]
    for cell_key in ("01000", "01010", "11000"):
        for src in ("librarian", "surgeon", "programmer"):
            assert not mod.cell_complete_on_hub(
                cell_key, src, 42, issue=383, hub_files_cache=parent_hub_index
            ), (
                f"cell_complete_on_hub(cell={cell_key}, source={src}, "
                f"issue=383) FALSE-SKIPPED against parent #365's adapter "
                f"— this is the catastrophic plumbing failure the fact-"
                f"checker flagged; the issue arg is not being scoped to "
                f"adapters/issue_383/"
            )


def test_cell_complete_on_hub_matches_issue_383_namespace_when_populated() -> None:
    """The positive case: an issue-383 adapter in the cache MUST be skipped.

    Mirror of the negative test above. Confirms the probe still works
    correctly when a real issue-383 adapter sits on the Hub — e.g. on a
    resume after a partial issue-383 run uploaded some cells.
    """
    mod = _load_dispatch_module()
    expected_run = mod.hf_hub_adapter_run_name("01000", "librarian", 42, issue=383)
    assert expected_run == "i383_cell_01000_source_librarian_seed42", (
        f"hf_hub_adapter_run_name issue=383 should produce 'i383_*'; got {expected_run!r}"
    )
    hub_index = [
        f"adapters/issue_383/{expected_run}/adapter_config.json",
        f"adapters/issue_383/{expected_run}/adapter_model.safetensors",
        # A parent #365 entry alongside — must not confuse the probe.
        "adapters/issue_365/i365_cell_11111_source_surgeon_seed42/adapter_config.json",
    ]
    assert mod.cell_complete_on_hub("01000", "librarian", 42, issue=383, hub_files_cache=hub_index)
    # And cells NOT in the issue-383 namespace still return False.
    assert not mod.cell_complete_on_hub(
        "11111", "surgeon", 42, issue=383, hub_files_cache=hub_index
    )


def test_prefetch_hub_adapter_index_filters_to_requested_issue(monkeypatch) -> None:
    """``_prefetch_hub_adapter_index(issue=383)`` must only return ``adapters/issue_383/`` rows.

    Stubs ``HfApi.list_repo_files`` to return a mixed list of issue_365
    + issue_383 adapter files and asserts the helper filters to the
    383-scoped subset.
    """
    mod = _load_dispatch_module()
    all_files = [
        "adapters/issue_365/i365_cell_00000_source_librarian_seed42/adapter_config.json",
        "adapters/issue_365/i365_cell_00010_source_surgeon_seed42/adapter_model.safetensors",
        "adapters/issue_383/i383_cell_01000_source_librarian_seed42/adapter_config.json",
        "adapters/issue_383/i383_cell_01010_source_surgeon_seed42/adapter_model.safetensors",
        "README.md",
        "unrelated/file.txt",
    ]
    fake_api = mock.MagicMock()
    fake_api.list_repo_files.return_value = all_files
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: fake_api)

    filtered = mod._prefetch_hub_adapter_index(issue=383)
    assert filtered is not None
    assert all(f.startswith("adapters/issue_383/") for f in filtered), (
        f"_prefetch_hub_adapter_index(issue=383) returned non-383 entries: {filtered!r}"
    )
    assert len(filtered) == 2, (
        f"_prefetch_hub_adapter_index(issue=383) returned {len(filtered)} entries; "
        f"expected exactly 2 (the issue_383 prefix matches)"
    )
