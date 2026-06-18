"""RunSpec.workload_cmd validation tests (#588).

The validation contract has three layers (see the field's comment in
``backends/base.py``):

1. both-set → ``__post_init__`` raise (universal — tested here);
2. neither-set stays LEGAL at construction (the router suite + probes
   build bare specs that never render a workload — tested here);
3. the production fail-loud for neither-set lives at the dispatch CLI
   (exactly-one check, ``tests/test_dispatch_issue_cli.py``) and the
   GCP renderer (``tests/test_gcp_backend.py``).
"""

from __future__ import annotations

import pytest

from explore_persona_space.backends.base import RunSpec


def test_workload_cmd_and_hydra_args_both_set_raises() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        RunSpec(
            issue=588,
            intent="debug",
            hydra_args=("seed=42",),
            workload_cmd="bash scripts/issue588_smoke.sh",
        )


def test_neither_workload_cmd_nor_hydra_args_constructs_fine() -> None:
    """Bare specs are LEGAL — routing tests / est-start probes / reconnect
    paths construct them without ever rendering a workload."""
    spec = RunSpec(issue=588, intent="debug")
    assert spec.workload_cmd == ""
    assert spec.hydra_args == ()


def test_workload_cmd_only_constructs_fine() -> None:
    spec = RunSpec(issue=588, intent="debug", workload_cmd="bash scripts/issue588_smoke.sh")
    assert spec.workload_cmd == "bash scripts/issue588_smoke.sh"
    assert spec.hydra_args == ()


@pytest.mark.parametrize(
    "bad_cmd",
    [
        "bash a.sh\necho second-line",
        "bash a.sh\r\necho crlf",
    ],
)
def test_workload_cmd_multiline_raises(bad_cmd: str) -> None:
    """A multi-line command would break the rendered startup-script /
    sbatch structure — single-line is part of the verbatim-embed
    contract."""
    with pytest.raises(ValueError, match="single line"):
        RunSpec(issue=588, intent="debug", workload_cmd=bad_cmd)


@pytest.mark.parametrize("bad_cmd", [" bash a.sh", "bash a.sh ", "\tbash a.sh"])
def test_workload_cmd_unstripped_whitespace_raises(bad_cmd: str) -> None:
    with pytest.raises(ValueError, match="leading/trailing whitespace"):
        RunSpec(issue=588, intent="debug", workload_cmd=bad_cmd)
