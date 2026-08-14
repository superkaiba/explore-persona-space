"""Tests for the #2120 own-device-scoped GPU-state verdict / schema-from-
artifact surface pin in ``scripts/workflow_lint.py``.

One check under test: ``check_cvd_scoped_gpu_verdict_lens``
(``--check-cvd-scoped-gpu-verdict-lens``, bundled into the no-flags default
run) — the lens must stay present across its FIVE surfaces (code-reviewer.md
Step 0.72 + Blocker-tags entry; codex-code-reviewer.md copy-list bullet +
rubric-placeholder slot + Blocker-tags entry; the
code-reviewer-section-reference.md Step 0.72 detail span; the
experiment-implementer.md Schema-from-artifact item naming the
``### (c) How to verify`` paste target; the
experiment-implementer-section-reference.md Schema-from-artifact heading).

Incident drivers (both 2026-08-05/06): #2091's ``reap_generation_engine``
took ``max()`` of ``memory.used`` across ALL 4 host GPUs (``nvidia-smi``
ignores ``CUDA_VISIBLE_DEVICES``) and killed 4 of 9 rung-jobs whose own GPUs
were drained; #2061's round-1 implementation fabricated the #1336 shard
schema from memory, so the pipeline could not load its own input.

1. ``test_lens_passes_on_complete_corpus`` — all five surfaces present.
2. ``test_lens_fails_per_missing_surface`` — 13 parametrized drops, one
   negative fixture per pinned surface/token (strip the token, the check
   FAILs naming that file).
3. ``test_lens_passes_on_live_tree`` — binds the landed #2120 edits; the
   standing regression guard for future refactors of any surface.
4. ``test_check_cvd_scoped_gpu_verdict_lens_bundled_in_no_flags`` — the
   two-part behavioral bundling pin (the #1701/#2165 tests' shape): Part A
   scoped-flag subprocess against a drifted corpus (nonzero exit), Part B
   no-flags OR-chain + dispatch-ladder source evidence.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import check_cvd_scoped_gpu_verdict_lens  # noqa: E402

_TAG = "host-wide-gpu-verdict"
_WAIVER = "HOST_WIDE_GPU_VERDICT_EXEMPT"


def _write_lens_corpus(root: Path, *, drop: str | None = None) -> Path:
    """Build a minimal five-surface corpus under ``root``; ``drop`` removes
    exactly one surface/token to exercise each per-surface error."""
    agents = root / ".claude" / "agents"
    rules = root / ".claude" / "rules"
    agents.mkdir(parents=True, exist_ok=True)
    rules.mkdir(parents=True, exist_ok=True)

    # (1) code-reviewer.md: Step 0.72 section body + Blocker-tags line.
    tag_txt = "" if drop == "section-body-tag" else f"a single Critical tagged `{_TAG}`. "
    cvd_txt = (
        ""
        if drop == "section-body-cvd"
        else "`CUDA_VISIBLE_DEVICES` when set, ELSE the SLURM allocation-env "
        "chain, OR a threaded own-device id. "
    )
    waiver_txt = "" if drop == "section-body-waiver" else f"Waiver: `# {_WAIVER}: <reason>`. "
    section = (
        "### Step 0.72: Own-device-scoped GPU-state verdict gate (any diff type)\n\n"
        f"Check: the verdict aggregates ONLY own-device rows: {cvd_txt}"
        f"Unscoped verdict FAILs, {tag_txt}{waiver_txt}\n\n"
    )
    if drop == "step072-section":
        section = ""
    claude_tags = "`substantive`" if drop == "claude-blocker-tag" else f"`{_TAG}`, `substantive`"
    (agents / "code-reviewer.md").write_text(
        "# code-reviewer\n\n" + section + "### Step 9: Verdict\n\n"
        f"**Blocker tags:** [{claude_tags}]\n",
        encoding="utf-8",
    )

    # (2) codex-code-reviewer.md: bullet + rubric slot + Blocker-tags line.
    bullet_tag = "." if drop == "codex-bullet-tag" else f", a single Critical tagged `{_TAG}`."
    bullet = (
        '- "Step 0.72: Own-device-scoped GPU-state verdict gate" — an '
        f"unscoped host-wide GPU-state verdict FAILs{bullet_tag}\n"
        '- "Step 0.8: Read prior open binding concerns" — placeholder.\n'
    )
    if drop == "codex-heading":
        bullet = '- "Step 0.8: Read prior open binding concerns" — placeholder.\n'
    rubric = (
        "{{INLINED RUBRIC FROM code-reviewer.md Steps 0.7, 0.8}}\n"
        if drop == "codex-rubric"
        else "{{INLINED RUBRIC FROM code-reviewer.md Steps 0.7, 0.72, 0.8}}\n"
    )
    codex_tags = "`substantive`" if drop == "codex-blocker-tag" else f"`{_TAG}` | `substantive`"
    (agents / "codex-code-reviewer.md").write_text(
        "# codex-code-reviewer\n\n" + bullet + "\n" + rubric + "\n"
        f"**Blocker tags:** [{codex_tags}]\n",
        encoding="utf-8",
    )

    # (3) code-reviewer-section-reference.md: the Step 0.72 detail span.
    crsr_content = (
        "# code-reviewer section reference\n\n## Other span\n\nContent.\n"
        if drop == "crsr-span"
        else "## Step 0.72 detail — own-device-scoped GPU-state verdicts\n\n"
        "Accepted scoping shapes + FAIL templates + the #2091 BEFORE/AFTER "
        "worked shape.\n"
    )
    (rules / "code-reviewer-section-reference.md").write_text(crsr_content, encoding="utf-8")

    # (4) experiment-implementer.md: the Schema-from-artifact item.
    paste_txt = (
        ""
        if drop == "impl-paste-target"
        else "PASTE its OBSERVED top-level keys into `### (c) How to verify`. "
    )
    item = (
        "8. **Schema-from-artifact, never schema-from-memory.** Open exactly "
        f"ONE real shard/sidecar and {paste_txt}Probe one-liners: "
        "experiment-implementer-section-reference.md.\n\n"
    )
    if drop == "impl-item":
        item = ""
    (agents / "experiment-implementer.md").write_text(
        "# experiment-implementer\n\n### Before writing code\n\n"
        + item
        + "### During implementation\n\nOther content.\n",
        encoding="utf-8",
    )

    # (5) experiment-implementer-section-reference.md: the probe/paste span.
    eisr_content = (
        "# experiment-implementer section reference\n\n## Other span\n\nContent.\n"
        if drop == "eisr-heading"
        else "# experiment-implementer section reference\n\n"
        "## Before-writing-code item 8 detail — Schema-from-artifact\n\n"
        "Probe one-liners + the paste form.\n"
    )
    (rules / "experiment-implementer-section-reference.md").write_text(
        eisr_content, encoding="utf-8"
    )
    return root


def test_lens_passes_on_complete_corpus(tmp_path: Path) -> None:
    _write_lens_corpus(tmp_path)
    errors = check_cvd_scoped_gpu_verdict_lens(repo_root=tmp_path)
    assert errors == [], f"complete corpus should pass; got: {errors}"


_DROP_CASES: list[tuple[str, str, str]] = [
    ("step072-section", "### Step 0.72", "agents/code-reviewer.md"),
    ("section-body-tag", _TAG, "agents/code-reviewer.md"),
    ("section-body-cvd", "CUDA_VISIBLE_DEVICES", "agents/code-reviewer.md"),
    ("section-body-waiver", _WAIVER, "agents/code-reviewer.md"),
    ("claude-blocker-tag", "**Blocker tags:**", "agents/code-reviewer.md"),
    ("codex-heading", "copy-list token", "agents/codex-code-reviewer.md"),
    ("codex-bullet-tag", "copy-list bullet", "agents/codex-code-reviewer.md"),
    ("codex-rubric", "INLINED RUBRIC", "agents/codex-code-reviewer.md"),
    ("codex-blocker-tag", "**Blocker tags:**", "agents/codex-code-reviewer.md"),
    ("crsr-span", "Step 0.72 detail", "rules/code-reviewer-section-reference.md"),
    ("impl-item", "Schema-from-artifact", "agents/experiment-implementer.md"),
    ("impl-paste-target", "### (c) How to verify", "agents/experiment-implementer.md"),
    (
        "eisr-heading",
        "Schema-from-artifact",
        "rules/experiment-implementer-section-reference.md",
    ),
]


@pytest.mark.parametrize(("drop", "token", "path_frag"), _DROP_CASES)
def test_lens_fails_per_missing_surface(
    tmp_path: Path, drop: str, token: str, path_frag: str
) -> None:
    _write_lens_corpus(tmp_path, drop=drop)
    errors = check_cvd_scoped_gpu_verdict_lens(repo_root=tmp_path)
    assert errors, f"drop={drop}: expected >=1 error"
    assert any(token in e and path_frag in e for e in errors), (
        f"drop={drop}: no error carries both {token!r} and {path_frag!r}; got: {errors}"
    )


def test_lens_passes_on_live_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binds the landed #2120 edits; the standing regression guard for
    future refactors of any of the five surfaces."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    errors = check_cvd_scoped_gpu_verdict_lens(repo_root=None)
    assert errors == [], f"live tree should carry all five surfaces; got: {errors}"


def test_check_cvd_scoped_gpu_verdict_lens_bundled_in_no_flags(tmp_path: Path) -> None:
    """Two-part behavioral bundling pin (the #1701/#2165 tests' shape).

    Part A — scoped-flag subprocess against a DRIFTED corpus (the Step 0.72
    section dropped), rooted via ``EPS_WORKFLOW_LINT_REPO_ROOT``: proves the
    flag exists, the dispatch calls the function, and it emits its
    uniquely-worded error (nonzero exit).

    Part B — no-flags OR-chain + dispatch-ladder evidence: ``main()``'s
    source names ``args.check_cvd_scoped_gpu_verdict_lens`` in BOTH the
    ``no_flags = not (...)`` OR-chain and the ``or no_flags`` dispatch
    ladder.
    """
    # Part A — scoped-flag subprocess against a drifted corpus.
    _write_lens_corpus(tmp_path, drop="step072-section")
    workflow_yaml_src = _REPO_ROOT / ".claude" / "workflow.yaml"
    workflow_yaml_dst = tmp_path / ".claude" / "workflow.yaml"
    workflow_yaml_dst.parent.mkdir(parents=True, exist_ok=True)
    workflow_yaml_dst.write_bytes(workflow_yaml_src.read_bytes())
    lint_script = _REPO_ROOT / "scripts" / "workflow_lint.py"
    env = {**os.environ, "EPS_WORKFLOW_LINT_REPO_ROOT": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            str(lint_script),
            "--check-cvd-scoped-gpu-verdict-lens",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "Step 0.72" in combined and "#2120" in combined, (
        "Step 0.72 / #2120 error tokens missing from output — the CLI flag "
        "does not dispatch the check. "
        f"exit={result.returncode}, combined output:\n{combined}"
    )
    assert result.returncode != 0, (
        f"expected nonzero exit under drifted corpus; got exit="
        f"{result.returncode}, combined output:\n{combined}"
    )

    # Part B — OR-chain + dispatch ladder evidence.
    lint_src = lint_script.read_text(encoding="utf-8")
    main_start = lint_src.find("def main(")
    assert main_start >= 0, "could not locate def main( in workflow_lint.py"
    main_end = lint_src.find('if __name__ == "__main__":', main_start)
    assert main_end > main_start, "could not locate main() end sentinel"
    main_src = lint_src[main_start:main_end]
    or_chain_start = main_src.find("no_flags = not (")
    assert or_chain_start >= 0, "no_flags OR-chain not found in main()"
    or_chain_end = main_src.find(")", or_chain_start)
    or_chain_src = main_src[or_chain_start:or_chain_end]
    assert "args.check_cvd_scoped_gpu_verdict_lens" in or_chain_src, (
        "args.check_cvd_scoped_gpu_verdict_lens is NOT in the no_flags "
        "OR-chain — a bare workflow_lint.py invocation will not fire this "
        f"check. OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_cvd_scoped_gpu_verdict_lens or no_flags" in main_src, (
        "args.check_cvd_scoped_gpu_verdict_lens is NOT dispatched under "
        "`or no_flags` — the flag is defined but not bundled into the "
        "no-flags default run."
    )
