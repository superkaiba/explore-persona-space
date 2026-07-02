"""Regression test: dispatch.py _steering_cmds / _preventative_cmds argv
must be parseable by the real argparse parsers in the sub-scripts.

This is a permanent invariant: if a flag is added to _steering_cmds or
_preventative_cmds but the corresponding build_parser() doesn't know it,
argparse will exit(2) and the test fails — catching the round-3 fabricated-
evidence failure class at CI time.

Also asserts the --out-root default in all three scripts is the v3 namespace
(eval_results/issue_816/v3), consistent with dispatch.sh's explicit --out-root.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_module(name: str):
    """Import a scripts/*.py by path without executing main()."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Default-v3 assertions
# ---------------------------------------------------------------------------


def test_dispatch_out_root_default_is_v3() -> None:
    """dispatch.py --out-root argparse default must be eval_results/issue_816/v3.

    dispatch.py builds its parser inside main(), so we can't call build_parser()
    directly (it would trigger GPU detection).  Instead we grep the source — an
    exact string assert that fails immediately if the default is ever changed back.
    """
    _load_module("issue816_dispatch")
    src = (SCRIPTS / "issue816_dispatch.py").read_text()
    assert '"eval_results/issue_816/v3"' in src, (
        "dispatch.py --out-root default must be eval_results/issue_816/v3"
    )


def test_preventative_out_root_default_is_v3() -> None:
    """preventative.py build_parser() --out-root default must be eval_results/issue_816/v3."""
    mod = _load_module("issue816_preventative")
    p = mod.build_parser()
    defaults = p.parse_args([])
    assert defaults.out_root == "eval_results/issue_816/v3", (
        f"preventative --out-root default={defaults.out_root!r}, expected eval_results/issue_816/v3"
    )


def test_steering_out_root_default_is_v3() -> None:
    """steering.py build_parser() --out-root default must be eval_results/issue_816/v3."""
    mod = _load_module("issue816_steering")
    p = mod.build_parser()
    defaults = p.parse_args([])
    assert defaults.out_root == "eval_results/issue_816/v3", (
        f"steering --out-root default={defaults.out_root!r}, expected eval_results/issue_816/v3"
    )


# ---------------------------------------------------------------------------
# Dispatch argv round-trip: flags from _steering_cmds / _preventative_cmds
# must be accepted by the sub-scripts' build_parser().
# ---------------------------------------------------------------------------


def _dispatch_args(extra: list[str] | None = None) -> argparse.Namespace:
    """Return a parsed dispatch namespace matching the flags the dispatcher builds.

    dispatch.py builds its parser inside main(), so we can't call it directly
    (GPU detection fires).  We replicate only the fields _steering_cmds /
    _preventative_cmds / _probe_cmds actually READ from the namespace — that
    is the ABI the sub-scripts must accept, and the one this test verifies.
    """
    _load_module("issue816_dispatch")  # ensures module import side-effects run
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phases", nargs="+", default=["probe", "steering", "preventative", "screening"]
    )
    p.add_argument("--traits", nargs="+", default=["evil"])  # single trait for tests
    p.add_argument("--dataset-root", default="external/persona_vectors/dataset")
    p.add_argument("--external-root", default="external/persona_vectors")
    p.add_argument("--out-root", default="eval_results/issue_816/v3")
    p.add_argument("--ckpt-root", default="checkpoints/issue_816")
    p.add_argument("--cache-dir", default="data/issue_816/hf_dl")
    p.add_argument("--n-gpus", type=int, default=None)
    p.add_argument("--n-random-dirs", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--cells", type=int, default=1)
    p.add_argument("--n-questions", type=int, default=None)
    p.add_argument("--n-rollouts", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(extra or [])


def test_steering_argv_accepted_by_build_parser() -> None:
    """Flags that _steering_cmds() builds must be accepted by steering's build_parser()."""
    dispatch_mod = _load_module("issue816_dispatch")
    steering_mod = _load_module("issue816_steering")

    dispatch_args = _dispatch_args()
    # Get one entry from _steering_cmds
    entries = dispatch_mod._steering_cmds(dispatch_args)
    assert entries, "_steering_cmds returned no entries"
    cmd = entries[0]["cmd"]
    # Strip the 'uv run python <script>' prefix (first 4 tokens)
    flags = cmd[4:]
    # Verify the steering parser accepts these flags without error
    p = steering_mod.build_parser()
    parsed = p.parse_args(flags)
    assert parsed.phase == "steer"
    assert parsed.out_root == "eval_results/issue_816/v3"


def test_preventative_argv_accepted_by_build_parser() -> None:
    """Flags that _preventative_cmds() builds must be accepted by preventative's build_parser()."""
    dispatch_mod = _load_module("issue816_dispatch")
    preventative_mod = _load_module("issue816_preventative")

    dispatch_args = _dispatch_args()
    entries = dispatch_mod._preventative_cmds(dispatch_args)
    assert entries, "_preventative_cmds returned no entries"
    cmd = entries[0]["cmd"]
    # Strip 'uv run python <script>' prefix
    flags = cmd[4:]
    p = preventative_mod.build_parser()
    parsed = p.parse_args(flags)
    assert parsed.out_root == "eval_results/issue_816/v3"
    assert parsed.traits == ["evil"]


def test_probe_argv_accepted_by_steering_build_parser() -> None:
    """Flags that _probe_cmds() builds must also be accepted by steering's build_parser()."""
    dispatch_mod = _load_module("issue816_dispatch")
    steering_mod = _load_module("issue816_steering")

    dispatch_args = _dispatch_args()
    entries = dispatch_mod._probe_cmds(dispatch_args)
    assert entries, "_probe_cmds returned no entries"
    cmd = entries[0]["cmd"]
    flags = cmd[4:]
    p = steering_mod.build_parser()
    parsed = p.parse_args(flags)
    assert parsed.phase == "probe"
