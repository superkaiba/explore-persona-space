"""Round-3 regression tests for the issue #763 code-review BLOCKERs.

Two permanent invariants closing the round-2 BLOCKERs, each failing pre-fix /
passing post-fix:

1. ``pv-judge-not-off-pod`` — the production dispatcher NEVER runs a ``--phase
   judge`` (PV or E0) between two ``--device cuda`` phases: every judge invocation
   sits AFTER the last GPU phase, so the orchestrator can stop the pod before the
   deadline-bounded batch-judge poll (the #664 spend-leak class). The mechanizable
   check the Codex suggestion named: "no ``--phase judge`` between two CUDA PV
   phases." The off-pod judge also reads its rollouts via an HF-servable path
   (``_stage_from_hf`` / ``snapshot_download``), not a hard-coded on-pod path.

2. ``pv-rollouts-not-uploaded`` — ``_upload_analysis_tensors`` iterates over an
   artifact set that INCLUDES ``pv_rollouts`` (and ``pv_judge``), so the rollouts
   the off-pod judge fetches actually land on HF pre-teardown.

These are behavior-focused: they parse the actual production dispatcher and trip
the actual upload-iteration / HF-staging path the round-3 fixes added.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

_DISPATCH = _REPO / "scripts" / "issue763_dispatch.sh"


# ── BLOCKER pv-rollouts-not-uploaded ──────────────────────────────────────────


def test_upload_analysis_artifacts_includes_pv_rollouts():
    """``_upload_analysis_tensors`` iterates ``pv_rollouts`` (and ``pv_judge``).

    Pre-fix the iteration was only ``("v0_shards", "pv_shards")`` so the PV
    rollouts at ``data/issue_763/pv_rollouts/<behavior>.jsonl`` were NEVER
    uploaded and the off-pod judge had nothing to ``snapshot_download``.
    """
    import issue763_upload as up

    subs = {entry[0] for entry in up._ANALYSIS_ARTIFACTS}
    assert "pv_rollouts" in subs, f"pv_rollouts missing from the upload iteration: {subs}"
    # pv_judge (the off-pod judge keep-flags the resumed pod fetches) too.
    assert "pv_judge" in subs, f"pv_judge missing from the upload iteration: {subs}"
    # the pre-existing tensors must still be uploaded (no regression).
    assert {"v0_shards", "pv_shards"} <= subs


def test_pv_rollouts_uploaded_from_data_dir_with_jsonl_pattern():
    """The pv_rollouts entry sources ``DATA_DIR/pv_rollouts`` with a ``*.jsonl`` glob.

    The rollouts are ``.jsonl`` under ``data/issue_763/`` (NOT ``.pt`` under
    ``eval_results/``), so the entry must carry the right source root + pattern or
    the upload silently uploads zero files (the original ``*.pt``-only glob would
    have matched nothing).
    """
    import issue763_common as common
    import issue763_upload as up

    by_sub = {entry[0]: entry for entry in up._ANALYSIS_ARTIFACTS}
    _sub, src_root, pattern = by_sub["pv_rollouts"]
    assert src_root == common.DATA_DIR, src_root
    assert pattern == "*.jsonl", pattern


# ── BLOCKER pv-judge-not-off-pod ──────────────────────────────────────────────


def _dispatcher_lines() -> list[str]:
    return _DISPATCH.read_text().splitlines()


_REAL_RUN_MARKER = "# ── REAL RUN"


def _executable_lines(*, real_run_only: bool = False) -> list[tuple[int, str]]:
    """Return (1-based lineno, line) for non-comment, non-blank shell lines.

    Strips leading whitespace; drops ``#``-comment lines and blank lines so the
    CUDA-vs-judge ordering check reads only EXECUTED commands (a comment that
    mentions ``--device cuda`` or ``--phase judge`` must not perturb the parse).

    ``real_run_only`` restricts to the REAL-RUN section (after the
    ``# ── REAL RUN`` marker), excluding the ``--smoke`` early-return block — the
    smoke branch runs every phase in one offline CPU process (no ``--device
    cuda`` at all) and ``exit 0``s before the real run, so its ``[phase=judge]``
    echo is NOT "between two cuda phases" in any executed flow.
    """
    lines = _dispatcher_lines()
    start_idx = 0
    if real_run_only:
        for i, raw in enumerate(lines):
            if raw.startswith(_REAL_RUN_MARKER):
                start_idx = i
                break
    out: list[tuple[int, str]] = []
    for i, raw in enumerate(lines, start=1):
        if i - 1 < start_idx:
            continue
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append((i, raw))
    return out


def test_no_phase_judge_between_two_cuda_phases():
    """No ``--phase judge`` invocation sits between two ``--device cuda`` phases.

    The mechanizable Codex-suggestion check (pv-judge-not-off-pod). For EVERY
    executed judge invocation (a ``--phase judge`` flag OR an
    ``issue763_judge_e0.py`` call OR a ``[phase=pv_extract_judge]`` marker), there
    must be NO ``--device cuda`` command on a LATER executed line — i.e. the judge
    is never sandwiched before a subsequent GPU phase, so the pod can be stopped
    for the deadline-bounded poll.
    """
    lines = _executable_lines(real_run_only=True)
    cuda_linenos = [ln for ln, txt in lines if "--device cuda" in txt]
    last_cuda = max(cuda_linenos) if cuda_linenos else -1

    def _is_judge(txt: str) -> bool:
        return (
            "--phase judge" in txt
            or "issue763_judge_e0.py" in txt
            or "phase=pv_extract_judge" in txt
        )

    judge_linenos = [ln for ln, txt in lines if _is_judge(txt)]
    offenders = [ln for ln in judge_linenos if ln < last_cuda]
    assert not offenders, (
        f"judge invocation(s) at line(s) {offenders} precede a later --device cuda "
        f"phase (last cuda at line {last_cuda}); the GPU pod would be held through "
        "the batch-judge poll (#763 BLOCKER pv-judge-not-off-pod)"
    )


def test_pv_extract_judge_phase_removed_from_dispatcher():
    """The pre-fix on-pod ``--phase judge`` (``[phase=pv_extract_judge]``) is gone.

    The PV judge moved to the orchestrator's OFF-pod VM invocation; no EXECUTED
    line of the dispatcher may emit a ``[phase=pv_extract_judge]`` marker or call
    ``issue763_extract_pv_rb.py --phase judge`` (comments documenting the move are
    fine — only executed commands matter).
    """
    executed = [txt for _ln, txt in _executable_lines()]
    offenders = [
        txt
        for txt in executed
        if "phase=pv_extract_judge" in txt
        or ("issue763_extract_pv_rb.py" in txt and "--phase judge" in txt)
    ]
    assert not offenders, f"the on-pod PV judge phase was not removed: {offenders}"


def test_dispatcher_emits_blocking_pod_cycle_gate():
    """Phase 1 emits a BLOCKING gate so the orchestrator can pod-cycle for the judge.

    The dispatcher must (a) emit the ``pv_phase1_done`` gate via ``--emit-gate``
    and (b) accept ``--from-phase pv_capture`` to resume after the off-pod judge.
    """
    body = _DISPATCH.read_text()
    assert "--emit-gate pv_phase1_done" in body, "phase 1 never emits the pod-cycle gate"
    assert "pv_capture" in body, "the dispatcher has no --from-phase pv_capture resume branch"


def test_offpod_judge_stages_rollouts_from_hf_not_hardcoded_path():
    """``--phase judge`` reads rollouts via an HF-servable path (not on-pod-only).

    The off-pod judge runs on the VM (the pod is stopped), so it CANNOT rely on a
    pod-local rollout path; it must ``snapshot_download`` the rollouts from the
    issue-owned HF prefix. Assert ``_phase_judge`` calls ``_stage_from_hf`` for the
    ``pv_rollouts`` subdir and that the staging helper resolves the HF data-repo
    analysis-tensors prefix (no hard-coded ``/workspace`` rollout path).
    """
    pv = (_REPO / "scripts" / "issue763_extract_pv_rb.py").read_text()
    # _phase_judge stages pv_rollouts from HF before reading them
    judge_fn = pv.split("def _phase_judge(", 1)[1].split("\ndef ", 1)[0]
    assert "_stage_from_hf(" in judge_fn and "pv_rollouts" in judge_fn, (
        "_phase_judge does not stage pv_rollouts from HF — it would fail off-pod"
    )
    # the staging helper uses snapshot_download from the issue-owned HF prefix
    stage_fn = pv.split("def _stage_from_hf(", 1)[1].split("\ndef ", 1)[0]
    assert "snapshot_download" in stage_fn, "_stage_from_hf does not snapshot_download from HF"
    assert "HF_ANALYSIS_TENSORS_PREFIX" in stage_fn, (
        "_stage_from_hf does not resolve the issue-owned HF analysis-tensors prefix"
    )
    # no hard-coded absolute on-pod rollout path in the judge/stage path
    assert not re.search(r"/workspace/[^\s\"']*pv_rollouts", pv), (
        "a hard-coded /workspace pv_rollouts path would not be servable off-pod"
    )
