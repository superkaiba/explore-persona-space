"""Regression pins for the #2223 upload-phase Hub signature bug (late-firing crash).

Incident: ``scripts/issue2223_drift.py:2509`` (pre-fix) called

    hub._upload(at, f"{HF_EXPERIMENT}/analysis_tensors", repo_type="dataset")

against the real 4-required-positional signature
``_upload(local_path, repo_id, repo_type, path_in_repo, ...)`` (hub.py:1490) —
the prefix landed where ``repo_id`` goes and ``path_in_repo`` was omitted, so
``phase_upload`` raised ``TypeError: _upload() missing 1 required positional
argument: 'path_in_repo'`` at the END of the run (~70+ GPU-h in), AFTER
raw_completions uploaded but BEFORE analysis_tensors persisted (the #521
plan-referenced-downstream-inputs artifact class).

These tests bind the ARGUMENT SHAPE of every Hub/HF call site in the driver
(extracted from the driver's own AST, so the test cannot pass while the call
site drifts) against the INSTALLED callables' ``inspect.signature`` — offline,
no network, no GPU. Against the pre-fix call the generic bind test fails with
exactly the incident TypeError (verified via ``git show`` on the pre-fix blob).

Also pins the destination: repo must be ``hub.DEFAULT_DATASET_REPO`` and the
prefix ``{HF_EXPERIMENT}/analysis_tensors`` — a wrong prefix is as bad as the
crash (upload-verification reconciles against expected prefixes).
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "issue2223_drift.py"


def _import_driver():
    sys.path.insert(0, str(REPO))
    from scripts import issue2223_drift as D

    return D


def _driver_tree() -> ast.Module:
    return ast.parse(DRIVER.read_text())


def _bind_shape(fn, node: ast.Call, label: str) -> None:
    """Bind the call node's shape (positional count + keyword names) against the
    real callable's signature; placeholder objects stand in for argument values."""
    assert not any(isinstance(a, ast.Starred) for a in node.args), (
        f"{label} (line {node.lineno}): *args forwarding not expected in driver hub calls"
    )
    assert all(kw.arg is not None for kw in node.keywords), (
        f"{label} (line {node.lineno}): **kwargs forwarding not expected in driver hub calls"
    )
    args = [object() for _ in node.args]
    kwargs = {kw.arg: object() for kw in node.keywords}
    try:
        inspect.signature(fn).bind(*args, **kwargs)
    except TypeError as e:  # pragma: no cover - the failure branch IS the regression
        raise AssertionError(
            f"{label} at scripts/issue2223_drift.py:{node.lineno} does not bind against the "
            f"installed signature {inspect.signature(fn)}: {e}"
        ) from e


def test_every_hub_and_hf_call_site_binds_against_installed_signatures():
    """Every ``hub.*`` / ``hf_hub_download`` / ``HfApi().*`` call in the driver binds.

    Pre-fix, this fails on the ``phase_upload`` ``hub._upload`` call with the
    incident's exact ``TypeError: missing a required argument: 'path_in_repo'``.
    """
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    seen = []
    for node in ast.walk(_driver_tree()):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "hub":
            # getattr fails loud if the driver references a nonexistent hub helper.
            _bind_shape(getattr(hub, f.attr), node, f"hub.{f.attr}")
            seen.append(f"hub.{f.attr}")
        elif isinstance(f, ast.Name) and f.id == "hf_hub_download":
            _bind_shape(hf_hub_download, node, "hf_hub_download")
            seen.append("hf_hub_download")
        elif (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Call)
            and isinstance(f.value.func, ast.Name)
            and f.value.func.id == "HfApi"
        ):
            # HfApi() construction is offline-safe; binding on the bound method
            # keeps ``self`` out of the shape.
            _bind_shape(getattr(HfApi(), f.attr), node, f"HfApi().{f.attr}")
            seen.append(f"HfApi().{f.attr}")

    # The known persistence-path call sites must all be present — an empty walk
    # (e.g. a rename breaking the extraction) must not vacuously pass.
    assert "hub._upload" in seen, "phase_upload's hub._upload call not found in driver AST"
    assert "hub.upload_raw_completions_to_data_repo" in seen
    assert "HfApi().upload_file" in seen
    assert "hf_hub_download" in seen


def _phase_upload_hub_upload_call() -> ast.Call:
    tree = _driver_tree()
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "phase_upload"
    )
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_upload"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "hub"
    ]
    assert len(calls) == 1, f"expected exactly one hub._upload in phase_upload, got {len(calls)}"
    return calls[0]


def test_phase_upload_destination_is_data_repo_analysis_tensors_prefix():
    """The analysis_tensors upload targets DEFAULT_DATASET_REPO at
    ``{HF_EXPERIMENT}/analysis_tensors`` — canonical 4-positional form, fail-loud."""
    call = _phase_upload_hub_upload_call()
    assert len(call.args) == 4, (
        f"hub._upload must be called with the 4 canonical positionals "
        f"(local_path, repo_id, repo_type, path_in_repo); got {len(call.args)}"
    )
    repo_arg = call.args[1]
    assert (
        isinstance(repo_arg, ast.Attribute)
        and repo_arg.attr == "DEFAULT_DATASET_REPO"
        and isinstance(repo_arg.value, ast.Name)
        and repo_arg.value.id == "hub"
    ), "repo_id positional must be hub.DEFAULT_DATASET_REPO"
    assert isinstance(call.args[2], ast.Constant) and call.args[2].value == "dataset", (
        "repo_type positional must be the literal 'dataset'"
    )
    prefix = call.args[3]
    assert isinstance(prefix, ast.JoinedStr), "path_in_repo must be the f-string prefix"
    names = [
        v.value.id
        for v in prefix.values
        if isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name)
    ]
    literals = "".join(
        v.value for v in prefix.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
    )
    assert names == ["HF_EXPERIMENT"] and literals == "/analysis_tensors", (
        f"path_in_repo must be f'{{HF_EXPERIMENT}}/analysis_tensors'; "
        f"got names={names} literals={literals!r}"
    )
    kw = {k.arg for k in call.keywords}
    assert "raise_on_error" in kw, (
        "raise_on_error=True must be passed so transport failures re-raise with their "
        "real traceback (the '' fail-soft returns are covered by the if-not-url guard)"
    )
    # The fail-loud guard on the '' fail-soft returns (missing HF_TOKEN, absent
    # path, 0-files verify) must survive: raise_on_error only covers exceptions.
    src = ast.get_source_segment(DRIVER.read_text(), _phase_upload_fn())
    assert "if not url" in src and "RuntimeError" in src, (
        "phase_upload must keep the `if not url: raise RuntimeError` durability guard"
    )


def _phase_upload_fn() -> ast.FunctionDef:
    return next(
        n
        for n in ast.walk(_driver_tree())
        if isinstance(n, ast.FunctionDef) and n.name == "phase_upload"
    )


def test_driver_module_imports_and_exposes_phase_upload():
    """The driver module imports (top-level imports resolve) and phase_upload exists —
    the bind tests above read the AST; this pins that the module itself is importable."""
    D = _import_driver()
    assert callable(D.phase_upload)
    assert D.HF_EXPERIMENT == "issue2223_persona_drift"
