"""c73 GPU-lane routing declares its cuda-context: claim — verify_plan tests (#2624).

WARN-only, conditional: a `kind: experiment|analysis` plan whose section-9
window (else whole plan) carries a GPU-lane routing token (`--gpu-type` /
`--gpu-count`, a GPU `--intent`, an `intent:` token, or an Nx<GPU-class>
shape) must carry at least one `cuda-context:` declaration line
(plan-compute-sizing.md § CUDA-context claim). The founding fixture mirrors
incident #2546: a §9 row routing `p5_fits` to a 4x H100 GPU lane whose
realized run never allocated a CUDA context. The check NEVER returns
passed=False (every branch asserts `r.passed is True`); the no-trigger case
renders PASS, not SKIP — the deliberate divergence from the conditional-SKIP
convention pinned by the plan §4.2 fixture matrix.
"""

# The typographic-width fixtures quote the real corpus glyph U+00D7 (the
# MULTIPLICATION SIGN) the c73 trigger grammar accepts — ambiguous-unicode
# lint is noise here (the monolith tests/test_verify_plan.py carries the
# same directive).
# ruff: noqa: RUF001

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_verify_plan():
    spec = importlib.util.spec_from_file_location(
        "verify_plan", REPO_ROOT / "scripts" / "verify_plan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("verify_plan", mod)
    spec.loader.exec_module(mod)
    return sys.modules["verify_plan"]


verify_plan = _load_verify_plan()

C73 = "c73_gpu_lane_cuda_context"

# #2546-faithful §9 shape: a GPU-lane fits phase with an intent token and an
# Nx<GPU-class> width, and no cuda-context: declaration anywhere.
GPU_LANE_PLAN = """\
# Plan — task #9996: c73 founding fixture (#2546 p5_fits shape)

## 4. Method

- Phase p5_fits fits the per-cell ridge battery over the staged activation
  store; the dispatcher assigns per-worker devices (alloc=0,1,2,3).

## 9. Compute sizing

- Phase p5_fits: `--intent lora-7b`, 4x H100, planned_wall_h 3.0.
- Launch: `uv run python scripts/dispatch_issue.py launch --issue 9996 \\
  --intent lora-7b --repo-branch main`
"""

SATISFIER_LINE = (
    "- cuda-context: yes — torch ridge fits run on cuda:0..3 (tensors moved in _fit_cells())\n"
)


def _run(plan: str, kind: str = "experiment"):
    return verify_plan.check_gpu_lane_cuda_context(plan, kind)


def test_t1_gpu_lane_without_declaration_warns() -> None:
    r = _run(GPU_LANE_PLAN)
    assert r.id == C73
    assert r.status == "WARN"
    assert r.passed is True  # WARN-only by contract — never blocks
    assert "cuda-context" in r.detail
    assert "plan-compute-sizing.md" in r.detail
    assert "section-9 window" in r.detail  # trigger scoped to §9 when it parses


def test_t2_declaration_line_satisfies() -> None:
    r = _run(GPU_LANE_PLAN + "\n" + SATISFIER_LINE)
    assert r.status == "PASS", r.detail
    assert r.passed is True
    assert "cuda-context:" in r.detail


def test_t2b_declaration_outside_section9_satisfies() -> None:
    # The satisfier is searched WHOLE-PLAN (declarations commonly live in §4).
    plan = GPU_LANE_PLAN.replace(
        "## 9. Compute sizing",
        SATISFIER_LINE + "\n## 9. Compute sizing",
    )
    r = _run(plan)
    assert r.status == "PASS", r.detail


def test_t3_kind_infra_skips() -> None:
    r = _run(GPU_LANE_PLAN, kind="infra")
    assert r.status == "SKIP"
    assert r.passed is True
    assert "kind=infra" in r.detail


def test_t4_no_gpu_lane_tokens_passes_not_skips() -> None:
    plan = (
        "# Plan — CPU-only analysis\n\n"
        "## 9. Compute sizing\n\n"
        "- Phase p1_agg: `--intent cpu-bigmem`, 16 vCPU, planned_wall_h 1.0.\n"
    )
    r = _run(plan)
    assert r.status == "PASS"
    assert r.passed is True
    assert "no GPU-lane routing token" in r.detail


def test_t5_whole_plan_fallback_when_no_section9() -> None:
    plan = (
        "# Plan — no numbered sections\n\n"
        "We provision 8 × H200 for the tensor-parallel inference sweep.\n"
    )
    r = _run(plan)
    assert r.status == "WARN"
    assert "whole plan" in r.detail


def test_t6_typographic_and_ascii_width_forms_trigger() -> None:
    # Both "4x H100" (ASCII) and the typographic multiplication-sign form.
    for token in ("4x H100", "8 × H200", "1x A100", "2 x B200"):
        plan = f"## 9. Compute\n\n- Width: {token} for the fits phase.\n"
        r = _run(plan)
        assert r.status == "WARN", (token, r.detail)


def test_t7_gpu_flags_and_intent_frontmatter_trigger() -> None:
    for token in (
        "--gpu-type H100",
        "--gpu-count 4",
        "--intent eval",
        "--intent debug",
        "intent: ft-7b",
    ):
        plan = f"## 9. Compute\n\n- Launch carries `{token}`.\n"
        r = _run(plan)
        assert r.status == "WARN", (token, r.detail)


def test_t8_cpu_lane_widths_do_not_trigger() -> None:
    # Nx shapes only trigger on GPU classes; a vCPU width or cpu intent is
    # not a GPU lane.
    plan = (
        "## 9. Compute sizing\n\n"
        "- Phase p1: `--intent cpu-mid`, 8 vCPU / 16 GB, planned_wall_h 0.5.\n"
        "- Shard 4x workers across the pod's cores.\n"
    )
    r = _run(plan)
    assert r.status == "PASS", r.detail


def test_t9_gpu_token_outside_section9_ignored_when_window_parses() -> None:
    # Trigger scope is the §9 window when one parses: a GPU mention in prose
    # OUTSIDE §9 (e.g. quoting a parent recipe) does not fire.
    plan = (
        "# Plan\n\n"
        "## 4. Method\n\n"
        "- The parent #2546 ran on 4x H100 (quoted for context only).\n\n"
        "## 9. Compute sizing\n\n"
        "- Phase p1_agg: `--intent cpu-bigmem`, planned_wall_h 1.0.\n"
    )
    r = _run(plan)
    assert r.status == "PASS", r.detail


def test_never_returns_passed_false() -> None:
    # Sweep every fixture shape in this file: c73 is WARN-only by contract.
    fixtures = [
        (GPU_LANE_PLAN, "experiment"),
        (GPU_LANE_PLAN + "\n" + SATISFIER_LINE, "experiment"),
        (GPU_LANE_PLAN, "infra"),
        (GPU_LANE_PLAN, "batch"),
        ("no tokens at all", "experiment"),
        ("8 × H200 everywhere", "analysis"),
    ]
    for plan, kind in fixtures:
        r = _run(plan, kind)
        assert r.passed is True, (kind, r.status, r.detail)


def test_registered_in_checks() -> None:
    assert verify_plan.check_gpu_lane_cuda_context in verify_plan.CHECKS
