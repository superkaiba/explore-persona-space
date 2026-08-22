"""c58 fan-out RunPod pod-name collision — verify_plan gate tests (#2237).

Fixtures are structurally faithful to the founding #2054 v16 shape (the
#2165 fixture-fidelity lesson): the pod-safety line reproduces v16 line
298's full three-way hazard (the bare "provision" noun + a teardown
``--name-suffix`` + the ``pod-<N>-<slug>`` naming convention on ONE
line), the §9 fan-out prose reproduces the across-cell bullet, and the
launch commands are two distinct ``--backend runpod`` argvs — so a
remedy-regex loosening that would suppress the WARN on the real
acceptance fixture fails HERE first (test 1b pins the measured FACT 2
hazard against the real corpus file directly).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

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

C58 = "c58_fanout_pod_name_collision"

# The v16 across-cell bullet shape: fan-out noun + same-line concurrency
# vocabulary, no negation — T1a's trigger (c57's detector, reused).
C58_FANOUT_S9_LINE = (
    "- **Across-cell shard axis + realized width:** (pair-class-group × arm) "  # noqa: RUF001 — the multiplication sign is real plan text
    "+ 2 matched-n strata → 10 shards on 10 parallel `cpu-bigmem` pods "
    "(CPU pods run N-in-parallel; the one-pod rule is GPU-specific)."
)

# The v16 line-298 three-way shape, reproduced on ONE line: the bare noun
# "provision", a TEARDOWN `--name-suffix`, and the `pod-<N>-<slug>` naming
# convention — the exact text a naive `provision\w*.*--name-suffix` remedy
# key would falsely read as a remedy (FACT 2, measured).
C58_PODSAFETY_LINE = (
    "**Pod-safety signals:** `task.py add-tag 2054 keep-running` BEFORE the "
    "first provision; `epm:run-launched` per pod with the pod name in "
    "structured lead position (`pod=pod-2054-rb789-<shard>` naming "
    "convention); completion-side teardown per pod: verify uploads → "
    "`pod.py terminate --issue 2054 --name-suffix <slug> --yes`."
)

C58_TWO_RUNPOD_LAUNCHES = (
    "uv run python scripts/dispatch_issue.py launch --issue 2054 "
    "--intent cpu-bigmem --backend runpod --repo-branch issue-2054 "
    '--time-budget-hours 14 --workload-cmd "RB_CLASS=twobytwo bash run.sh"\n'
    "uv run python scripts/dispatch_issue.py launch --issue 2054 "
    "--intent cpu-bigmem --backend runpod --repo-branch issue-2054 "
    '--time-budget-hours 14 --workload-cmd "RB_MATCHEDN=boundary bash run.sh"\n'
)

C58_ONE_RUNPOD_LAUNCH = (
    "uv run python scripts/dispatch_issue.py launch --issue 2054 "
    "--intent cpu-bigmem --backend runpod --repo-branch issue-2054 "
    '--time-budget-hours 14 --workload-cmd "bash run.sh"\n'
)


def _plan(
    s9_prose: str = "",
    launches: str = C58_TWO_RUNPOD_LAUNCHES,
    tail: str = "",
) -> str:
    """A minimal structurally-faithful plan skeleton: §4, §9 (prose +
    fenced launch commands), §10 — the c57-test house pattern."""
    fenced = f"```bash\n{launches}```\n\n" if launches else ""
    return (
        "# Plan v1 — c58 fixture\n"
        "\n"
        "## 4. Design\n"
        "\n"
        "Design prose.\n"
        "\n"
        "## 9. Resources & Parallelism\n"
        "\n"
        + (s9_prose + "\n\n" if s9_prose else "")
        + fenced
        + (tail + "\n\n" if tail else "")
        + "## 10. Reproducibility\n"
        "\n"
        "Repro notes.\n"
    )


def _run(plan: str):
    return verify_plan.check_fanout_pod_name_collision(plan, "experiment")


def test_c58_registered_in_checks_and_docstring_catalog():
    # Membership pin (the c44/c46/c57 house pattern): a forgotten registry
    # append cannot ship green — the check existing is not the check running.
    assert verify_plan.check_fanout_pod_name_collision in verify_plan.CHECKS
    assert "c58 fan-out RunPod pod-name" in verify_plan.__doc__
    # conditional-checks enumeration carries 58 (closing-paren tail form;
    # the tail extends whenever a new conditional check lands — 59 since
    # #2123, 62/63 since #2276, 64 since #2174, 65/66 since #2178, 67 since
    # #2204, 68 since #2228, 69 since #2269 — and this pin is what makes a
    # forgotten enum update loud)
    assert "57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70)" in verify_plan.__doc__


# ─── Test 1 — fires on the #2054 v16 shape ─────────────────────────────────


def test_c58_2054_v16_shape_warns():
    """Positive fixture (#2054 v16 shape): §9 10-shard fan-out prose + the
    line-298 three-way pod-safety line + two `--backend runpod` argvs, no
    `pod.py provision` construct -> WARN naming the fan-out + incident."""
    plan = _plan(
        s9_prose=C58_FANOUT_S9_LINE,
        launches=C58_TWO_RUNPOD_LAUNCHES,
        tail=C58_PODSAFETY_LINE,
    )
    r = _run(plan)
    assert r.status == "WARN", r.detail
    assert r.passed  # WARN-only: the check NEVER FAILs a plan
    assert "10 shards" in r.detail
    assert "#2054" in r.detail
    assert "pod-<N>" in r.detail


def test_c58_fires_on_real_2054_v16_corpus_file():
    """Acceptance probe 1 (#2237 §8), pinned against the REAL persisted
    fixture: `tasks/*/2054/plans/v16.md` (status-folder-agnostic glob —
    the task moves folders with status)."""
    hits = sorted(REPO_ROOT.glob("tasks/*/2054/plans/v16.md"))
    if not hits:
        pytest.skip("2054 v16 not present in this checkout's tasks/ tree")
    r = _run(hits[0].read_text(errors="replace"))
    assert r.status == "WARN", r.detail


# ─── Test 1b — the FACT 2 remedy-regex hazard, pinned on the real line ─────


def test_c58_remedy_re_does_not_match_v16_line_298():
    """FACT 2's measured hazard: v16 line 298 carries the bare noun
    "provision", a teardown `--name-suffix`, AND the `pod-<N>-<slug>`
    naming convention on ONE line — a naive `provision\\w*.*--name-suffix`
    matches it (and would suppress the WARN on the acceptance fixture);
    the shipped `_C58_REMEDY_RE` must NOT."""
    naive = re.compile(r"provision\w*.*--name-suffix")
    # In-repo replica first (immune to corpus drift):
    assert naive.search(C58_PODSAFETY_LINE)
    assert not verify_plan._C58_REMEDY_RE.search(C58_PODSAFETY_LINE)
    # The real corpus line, when present:
    hits = sorted(REPO_ROOT.glob("tasks/*/2054/plans/v16.md"))
    if not hits:
        pytest.skip("2054 v16 not present in this checkout's tasks/ tree")
    text = hits[0].read_text(errors="replace")
    line298 = text.splitlines()[297]
    assert "provision" in line298 and "--name-suffix" in line298  # the 3-way shape
    assert naive.search(line298)
    assert not verify_plan._C58_REMEDY_RE.search(line298)
    # And nowhere else in v16 either — the WARN must not be suppressed:
    assert not verify_plan._C58_REMEDY_RE.search(text)


# ─── Test 1c — T1b converse shape ──────────────────────────────────────────


def test_c58_t1b_fires_on_two_argvs_without_s9_prose():
    """T1b (critic round 1): >=2 distinct RunPod-resolved argvs with NO §9
    fan-out prose vocabulary still trip T1 — multiplicity expressed in
    argvs (the #2054 v10-v12 shape: the 'parallel' multiplicity lived in
    FENCED bash comments T1a's mask correctly skips)."""
    plan = _plan(s9_prose="Sizing rows only.", launches=C58_TWO_RUNPOD_LAUNCHES)
    r = _run(plan)
    assert r.status == "WARN", r.detail
    assert "T1b" in r.detail


# ─── Test 2 — silent on a genuine single-launch RunPod plan ────────────────


def test_c58_single_launch_runpod_plan_skips():
    plan = _plan(s9_prose="One pod does the whole run.", launches=C58_ONE_RUNPOD_LAUNCH)
    r = _run(plan)
    assert r.status == "SKIP", r.detail
    assert "no concurrent box-level fan-out" in r.detail


# ─── Test 3 — silent on a suffixed `pod.py provision` fan-out ──────────────


def test_c58_provision_name_suffix_fanout_passes():
    """The remedy: a plan naming per-pod `pod.py provision ... --name-suffix`
    calls PASSes even with the fan-out prose + runpod argvs present (the
    corrected #2054 v17 mechanism)."""
    remedy = (
        "Each shard is provisioned DIRECTLY via `pod.py provision --issue 2054 "
        "--intent cpu-bigmem --name-suffix rb789-<slug>` (one per shard)."
    )
    plan = _plan(
        s9_prose=C58_FANOUT_S9_LINE,
        launches=C58_TWO_RUNPOD_LAUNCHES,
        tail=remedy,
    )
    r = _run(plan)
    assert r.status == "PASS", r.detail
    assert "pod.py provision" in r.detail


# ─── Test 4 — `--lane-suffix` on the launch argv satisfies T3 (#2145) ──────


def test_c58_lane_suffix_on_runpod_argv_passes_remedy():
    """Acceptance 4 (FLIPPED by #2145): `--lane-suffix` is now honored on the
    RunPod lane (`dispatch_issue._lane_suffix_honored_kinds` includes
    `runpod`), so a plan whose RunPod launches carry per-shard
    `dispatch_issue.py launch ... --lane-suffix <slug>` names a T3 remedy —
    the `_C58_REMEDY_RE` extended alternate — and PASSes instead of WARNing.
    (Pre-#2145 this exact fixture pinned WARN: the suffix was GCP/SLURM-only
    and its presence on a RunPod argv was naming-inert.)"""
    launches = C58_TWO_RUNPOD_LAUNCHES.replace(
        "--backend runpod", "--backend runpod --lane-suffix shard1", 1
    ).replace(
        "--backend runpod --repo-branch", "--backend runpod --lane-suffix shard2 --repo-branch"
    )
    assert "--lane-suffix" in launches
    plan = _plan(s9_prose=C58_FANOUT_S9_LINE, launches=launches)
    r = _run(plan)
    assert r.status == "PASS", r.detail
    # detail quotes the first 70 chars of the matched remedy construct — the
    # match anchors on `dispatch_issue.py launch` reaching `--lane-suffix`.
    assert "dispatch_issue.py launch" in r.detail


def test_c58_name_suffix_alias_on_runpod_argv_passes_remedy():
    """#2145: the `--name-suffix` argparse ALIAS on a `dispatch_issue.py
    launch` command satisfies the same extended remedy alternate."""
    launches = C58_TWO_RUNPOD_LAUNCHES.replace(
        "--backend runpod", "--backend runpod --name-suffix shard1", 1
    )
    assert "--name-suffix" in launches
    plan = _plan(s9_prose=C58_FANOUT_S9_LINE, launches=launches)
    r = _run(plan)
    assert r.status == "PASS", r.detail


# ─── Test 5 — explicit serialization (negation arm) ────────────────────────


def test_c58_serialized_fanout_skips():
    """Same-line negation (T1a's `_C57_NEGATION_RE`, reused): an explicitly
    serialized fan-out is not a concurrent declaration. ONE launch argv so
    T1b cannot fire either."""
    line = C58_FANOUT_S9_LINE + " Shards run strictly sequential, one pod at a time."
    r = _run(_plan(s9_prose=line, launches=C58_ONE_RUNPOD_LAUNCH))
    assert r.status == "SKIP", r.detail


# ─── Test 6 — one test per SKIP class, reason strings pinned ───────────────


def test_c58_skip_no_section9():
    plan = "# Plan\n\n## Design\n\n10 parallel pods via `--backend runpod`.\n"
    r = _run(plan)
    assert r.status == "SKIP"
    assert "no parseable section-9" in r.detail


def test_c58_skip_no_launch_argv():
    """The custom-driver residual (ii): fan-out prose but no plan-embedded
    dispatch_issue.py command — structurally invisible, stated as such."""
    r = _run(_plan(s9_prose=C58_FANOUT_S9_LINE, launches=""))
    assert r.status == "SKIP"
    assert "no launch-shaped dispatch_issue.py command" in r.detail
    assert "not coverage" in r.detail


def test_c58_skip_none_parses():
    """Launch argvs present but none dry-parses (missing required --issue):
    c46 arm 1 owns parse warnings — c58 SKIPs, never doubles the WARN."""
    bad = "uv run python scripts/dispatch_issue.py launch --intent cpu-bigmem --backend runpod\n"
    r = _run(_plan(s9_prose=C58_FANOUT_S9_LINE, launches=bad))
    assert r.status == "SKIP", r.detail
    assert "none dry-parses" in r.detail


def test_c58_skip_no_runpod_resolved_argv_under_shipped_posture():
    """The shipped T2 posture is explicit-`--backend runpod` ONLY (the §7
    calibration: the auto arm's 25 additional WARNs were all adjudicated
    FPs) — an absent-backend (auto) fan-out SKIPs with residual (i) named."""
    auto_launches = C58_TWO_RUNPOD_LAUNCHES.replace("--backend runpod ", "")
    assert "--backend" not in auto_launches
    r = _run(_plan(s9_prose=C58_FANOUT_S9_LINE, launches=auto_launches))
    assert r.status == "SKIP", r.detail
    assert "no RunPod-resolved launch argv" in r.detail
    assert "residual (i)" in r.detail


def test_c58_skip_gcp_slurm_lane_fanout():
    """Remedy arm 3, structural: a fan-out whose launches pin a
    name-isolating lane (fellows) never reaches T3 — no RunPod-resolved
    argv."""
    fellows = C58_TWO_RUNPOD_LAUNCHES.replace("--backend runpod", "--backend fellows")
    r = _run(_plan(s9_prose=C58_FANOUT_S9_LINE, launches=fellows))
    assert r.status == "SKIP", r.detail
    assert "no RunPod-resolved launch argv" in r.detail


# ─── Test 7 — never FAILs ──────────────────────────────────────────────────


def test_c58_never_fails():
    """WARN-only posture: no fixture in this file produces passed=False."""
    fixtures = [
        _plan(s9_prose=C58_FANOUT_S9_LINE, launches=C58_TWO_RUNPOD_LAUNCHES),
        _plan(s9_prose=C58_FANOUT_S9_LINE, launches=C58_ONE_RUNPOD_LAUNCH),
        _plan(s9_prose="", launches=""),
        "# Plan\n\nno headings at all\n",
        _plan(
            s9_prose=C58_FANOUT_S9_LINE,
            launches=C58_TWO_RUNPOD_LAUNCHES,
            tail="Per shard: `pod.py provision --issue 1 --name-suffix a`.",
        ),
    ]
    for plan in fixtures:
        r = _run(plan)
        assert r.passed, (r.status, r.detail)
        assert r.status != "FAIL"
