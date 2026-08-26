"""c71 plan-embedded jq probe dry-run vs committed target — verify_plan gate tests (#2590).

Fixtures are structurally faithful to their originating lines (the #2165
fixture-fidelity lesson): the founding FAIL fixture reproduces #2588
v2:108's shape — a comma-chained top-level-path probe against a
`splits.*`-keyed artifact, with the lead-prefixed `→ expect` clause OUTSIDE
the inline span — so the jq precedence trap (`|` binds looser than `,`,
rc=5 with stdout `0`) fires on the scratch replica exactly as it did on the
real artifact; the naive `.splits.`-prefixed "repair" fixture reproduces
BOTH critics' hand-written broken repair. The safe-execution tests
(T19-T25) run REAL /usr/bin/jq against a REAL scratch git repo — never
mocks — because the properties under test (output-cap kill, timeout reap,
env scrub, non-UTF8 decode-replace) live in the subprocess boundary itself.
The scratch repo uses tempfile.mkdtemp (NOT tmp_path: concurrent pytest
sessions prune /tmp/pytest-of numbered roots mid-test, and this module's
tests shell out to git/jq against the scratch tree).
"""

# ruff: noqa: RUF003
# The fixture strings quote the real corpus glyphs (→, ≤, ×) the check's
# grammars accept — ambiguous-unicode lint is noise here (the monolith
# tests/test_verify_plan.py carries the same directive).

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

if shutil.which("jq") is None:  # pragma: no cover - environment guard
    pytest.skip("jq unavailable — c71 executes real jq", allow_module_level=True)


def _load_verify_plan():
    spec = importlib.util.spec_from_file_location(
        "verify_plan", REPO_ROOT / "scripts" / "verify_plan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("verify_plan", mod)
    spec.loader.exec_module(mod)
    return sys.modules["verify_plan"]


verify_plan = _load_verify_plan()

C71 = "c71_jq_probe_dryrun"

SPLIT_CONTENT = {
    "counts": {"train_10k": 2, "val_400": 1, "test_1000": 3},
    "splits": {"train_10k": [1, 2], "val_400": [7], "test_1000": [8, 9, 10]},
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "c71",
            "GIT_AUTHOR_EMAIL": "c71@test",
            "GIT_COMMITTER_NAME": "c71",
            "GIT_COMMITTER_EMAIL": "c71@test",
        },
    )


@pytest.fixture(scope="module")
def scratch_repo():
    """One committed scratch git repo for the module (mkdtemp, not tmp_path
    — see module docstring). data/split_ids.json replicates the founding
    artifact's key layout; data/wt_only.json is deliberately on disk but
    UNCOMMITTED (T15 pins committedness-not-disk)."""
    root = Path(tempfile.mkdtemp(prefix="c71jqtest-"))
    (root / "data").mkdir()
    (root / "data" / "split_ids.json").write_text(json.dumps(SPLIT_CONTENT))
    (root / "data" / "tiny.json").write_text("0")
    (root / "data" / "bad.json").write_bytes(b"\xff\xfe{")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c71 fixtures")
    (root / "data" / "wt_only.json").write_text("{}")  # on disk, NOT committed
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def repo(scratch_repo, monkeypatch):
    monkeypatch.setattr(verify_plan, "_C71_REPO_ROOT", scratch_repo)
    monkeypatch.setattr(verify_plan, "_C71_TIMEOUT_S", 2)
    return scratch_repo


def _run(plan: str, kind: str = "experiment"):
    return verify_plan.check_jq_probe_dryrun(plan, kind)


# ── founding-shape fixtures ────────────────────────────────────────────────

PROBE_BAD = (
    "4. Measured split counts: `jq '.train_10k|length, .val_400|length, "
    ".test_1000|length' data/split_ids.json` → expect 2/1/3; recorded in "
    "p0_preflight.json (the measured n_train the reads consume).\n"
)
PROBE_REPAIR_BAD = (
    "4. Measured split counts: `jq '.splits.train_10k|length, "
    ".splits.val_400|length, .splits.test_1000|length' data/split_ids.json` "
    "→ expect 2/1/3.\n"
)
PROBE_OK = (
    "4. Measured split counts: `jq '(.splits.train_10k|length), "
    "(.splits.val_400|length), (.splits.test_1000|length)' "
    "data/split_ids.json` → expect 2/1/3.\n"
)


# ── expectation-grammar unit tests (MF-B measured probes) ─────────────────


def test_expect_grammar_measured_probes():
    parse = verify_plan._c71_parse_expect
    assert parse("→ expect 10000/400/1000; recorded in p0.json") == [10000, 400, 1000]
    assert parse("→ expect null") == "null"
    assert parse("# expect 2/1/3") == [2, 1, 3]
    assert parse("-> expect 19,000") == [19000]
    # Prose/object expectations must NEVER arm (lead-required, value-terminal).
    assert parse("expect 3 columns with no lead") is None
    assert parse("→ expect 3 columns") is None
    assert parse("→ expect 3.5 s latency") is None
    assert parse("→ expect 10000, matching the corpus") is None


def test_expect_grammar_digit_bomb_no_exception():
    # T21 grammar side: the 18-char token bound makes the >4,300-digit
    # int() blowup unreachable; the guarded conversion is belt-and-braces.
    assert verify_plan._c71_parse_expect("→ expect " + "9" * 5000) is None


# ── offender arms against the real committed replica ─────────────────────


def test_t1_founding_probe_fails_rc5(repo):
    r = _run(PROBE_BAD)
    assert r.id == C71
    assert r.status == "FAIL"
    assert "rc=5" in r.detail
    assert "data/split_ids.json" in r.detail


def test_t2_naive_repair_also_fails(repo):
    # BOTH critics' hand-written repair: prefixing `.splits.` does not fix
    # the precedence trap — `|` binds looser than `,`.
    r = _run(PROBE_REPAIR_BAD)
    assert r.status == "FAIL"
    assert "rc=5" in r.detail


def test_t3_parenthesized_corrected_probe_passes(repo):
    r = _run(PROBE_OK)
    assert r.status == "PASS"
    assert "1 executed clean" in r.detail


def test_t4_wrong_expectation_fails_arm_c(repo):
    r = _run(PROBE_OK.replace("expect 2/1/3", "expect 3/1/3"))
    assert r.status == "FAIL"
    assert "expectation mismatch" in r.detail
    assert "got 2/1/3 vs expected 3/1/3" in r.detail


def test_t5_uncommitted_target_skips(repo):
    r = _run("Counts: `jq '.counts' data/nonexistent.json` → expect 2.\n")
    assert r.status == "SKIP"
    assert "unresolved target" in r.detail


def test_t6_kind_infra_skips(repo):
    r = _run(PROBE_BAD, kind="infra")
    assert r.status == "SKIP"
    assert "kind=infra" in r.detail


def test_t7_escape_passes_and_wrapped_variant_does_not(repo):
    r = _run(PROBE_BAD + "\nN/A — no registered jq probe\n")
    assert r.status == "PASS"
    assert "explicit N/A declared" in r.detail
    # #1238 anti-paste: a backtick-wrapped declaration is NOT recognized.
    r2 = _run(PROBE_BAD + "\n`N/A — no registered jq probe`\n")
    assert r2.status == "FAIL"


def test_t7b_sibling_escape_passes(repo):
    r = _run(
        PROBE_BAD
        + "\nN/A — quoted jq probe is historical or a sibling's, not this plan's gate input\n"
    )
    assert r.status == "PASS"
    assert "explicit N/A declared" in r.detail


def test_t8_off_allowlist_flag_refused(repo):
    r = _run("Read: `jq --slurpfile x data/tiny.json '.x' data/split_ids.json` runs at P0.\n")
    assert r.status == "SKIP"
    assert "refused" in r.detail
    assert "refused flags: 1" in r.detail


def test_t9_placeholder_skips(repo):
    r = _run("Counts: `jq '.counts' <split-file>` per cell.\n")
    assert r.status == "SKIP"
    assert "placeholder" in r.detail


def test_t10_jq_resolver_none_whole_check_skip(repo, monkeypatch):
    monkeypatch.setattr(verify_plan, "_c71_jq_bin", lambda: None)
    r = _run(PROBE_OK)
    assert r.status == "SKIP"
    assert "jq unavailable" in r.detail


def test_t11_absent_key_null_output_fails_arm_b(repo):
    r = _run("Probe: `jq '.absent_key' data/split_ids.json` at P0.\n")
    assert r.status == "FAIL"
    assert "all-null output" in r.detail
    assert "path-missed tell" in r.detail


def test_t12_superseded_line_skips(repo):
    r = _run("v2's `jq '.broken' data/split_ids.json` gave 0 on the healthy artifact.\n")
    assert r.status == "SKIP"
    assert "superseded-context" in r.detail


def test_t13_fenced_block_with_continuation_and_comment_expectation(repo):
    plan = (
        "P0 preflight:\n\n```bash\n"
        "jq '(.splits.train_10k|length), (.splits.val_400|length), "
        "(.splits.test_1000|length)' \\\n"
        "  data/split_ids.json   # expect 2/1/3\n"
        "```\n"
    )
    r = _run(plan)
    assert r.status == "PASS", r.detail
    assert "1 executed clean" in r.detail


def test_t14_empty_filter_fails_empty_arm(repo):
    r = _run("Probe: `jq 'empty' data/split_ids.json` at P0.\n")
    assert r.status == "FAIL"
    assert "empty output" in r.detail


def test_t15_on_disk_but_uncommitted_skips(repo):
    # The file EXISTS in the working tree; committedness is the bar.
    assert (repo / "data" / "wt_only.json").is_file()
    r = _run("Probe: `jq '.x' data/wt_only.json` at P0.\n")
    assert r.status == "SKIP"
    assert "unresolved target" in r.detail


def test_t16_pipe_input_form_skips_no_target(repo):
    r = _run("Probe: `cat data/split_ids.json | jq '.splits'` at P0.\n")
    assert r.status == "SKIP"
    assert "no target path" in r.detail


def test_t17_length_on_missing_key_prints_zero_clean_and_expectation_catches(repo):
    # Disclosed |length under-recall: null|length prints 0 at rc=0 — CLEAN
    # with no stated expectation (the #2588 primary shape is caught by
    # rc=5 / arm (a)); a stated machine expectation restores the catch.
    r = _run("Probe: `jq '.train_10k|length' data/split_ids.json` at P0.\n")
    assert r.status == "PASS"
    r2 = _run("Probe: `jq '.train_10k|length' data/split_ids.json` → expect 2.\n")
    assert r2.status == "FAIL"
    assert "got 0 vs expected 2" in r2.detail


def test_t18_bare_jq_length_with_prose_expectation_noted_uncertified(repo):
    r = _run("P0 re-checks counts (`jq length`, expected 2/1/3 columns).\n")
    assert r.status == "SKIP"
    assert "no target path" in r.detail
    assert "expectation not certified" in r.detail


def test_t19_output_flood_killed_at_cap(repo):
    r = _run("Probe: `jq 'while(true; .)' data/tiny.json` at P0.\n")
    assert r.status == "SKIP"
    assert "output-cap" in r.detail


def test_t20_nonexistent_jq_binary_whole_check_skip(repo, monkeypatch):
    monkeypatch.setattr(verify_plan, "_c71_jq_bin", lambda: "/nonexistent/jq-c71-test")
    r = _run(PROBE_OK)
    assert r.status == "SKIP"
    assert "jq launch failed" in r.detail


def test_t21_digit_bomb_expectation_executes_clean(repo):
    r = _run(PROBE_OK.replace("expect 2/1/3", "expect " + "9" * 5000))
    assert r.status == "PASS"
    assert "executed clean" in r.detail


def test_t22_infinite_loop_times_out_and_child_reaped(repo):
    r = _run("Probe: `jq 'until(false; .+1)' data/tiny.json` at P0.\n")
    assert r.status == "SKIP"
    assert "timeout" in r.detail
    probe = subprocess.run(["pgrep", "-f", "until.false"], capture_output=True, check=False)
    assert probe.returncode != 0, "jq child survived the kill-then-wait reap"


def test_t23_non_utf8_blob_fails_with_replaced_stderr(repo):
    r = _run("Probe: `jq '.x' data/bad.json` at P0.\n")
    assert r.status == "FAIL"
    assert "rc=" in r.detail  # arm (a); jq parse error on the invalid bytes
    assert "\x00" not in r.detail


def test_t24_module_referencing_filter_refused(repo):
    r = _run("Probe: `jq 'include \"foo\"; .' data/tiny.json` at P0.\n")
    assert r.status == "SKIP"
    assert "refused module-ref" in r.detail


def test_t25_env_scrub_no_parent_secret_reaches_jq(repo, monkeypatch):
    monkeypatch.setenv("EPM_T25_SECRET", "hunter2")
    r = _run("Probe: `jq -r 'env.EPM_T25_SECRET' data/tiny.json` at P0.\n")
    # Scrubbed env: the var is invisible to jq, so the read is null →
    # arm (b) fires; the secret VALUE must appear nowhere in the verdict.
    assert r.status == "FAIL"
    assert "hunter2" not in r.detail


def test_t26_post_run_expectation_skips_before_execution(repo):
    r = _run(
        "after the run completes, `jq '.counts.new_rows' data/split_ids.json` will show 25000.\n"
    )
    assert r.status == "SKIP"
    assert "post-run" in r.detail


def test_t29_post_run_class_escape_passes(repo):
    r = _run(
        "Probe: `jq '.counts.new_rows' data/split_ids.json` fills post-run.\n"
        "\nN/A — registered jq probes assert post-run state, not current committed state\n"
    )
    assert r.status == "PASS"
    assert "explicit N/A declared" in r.detail


def test_t27_prose_expectation_does_not_arm_and_is_noted(repo):
    r = _run(PROBE_OK.replace("expect 2/1/3", "expect 3 columns"))
    assert r.status == "PASS"
    assert "not machine-certified" in r.detail


def test_t28_expect_null_absence_assert(repo):
    r = _run("Probe: `jq '.retired_key' data/split_ids.json` → expect null.\n")
    assert r.status == "PASS", r.detail
    r2 = _run("Probe: `jq '.retired_key' data/split_ids.json` → expect 5.\n")
    assert r2.status == "FAIL"
    assert "got null vs expected 5" in r2.detail


def test_t30_markdown_table_row_pipe_unescape(repo):
    r = _run(
        "| A7 | counts | `jq '.splits.train_10k\\|length' data/split_ids.json` "
        "→ expect 2 | High |\n"
    )
    assert r.status == "PASS", r.detail
    assert "1 executed clean" in r.detail


def test_no_jq_vocabulary_skips(repo):
    r = _run("A plan with no probes at all; it mentions JSON artifacts only.\n")
    assert r.status == "SKIP"
    assert "no jq probe vocabulary" in r.detail


def test_instance_cap_bounds_executions(repo, monkeypatch):
    monkeypatch.setattr(verify_plan, "_C71_MAX_INSTANCES", 2)
    plan = "".join(f"Probe {i}: `jq '.counts' data/split_ids.json` at P0.\n" for i in range(4))
    r = _run(plan)
    assert r.status == "PASS"
    assert "2 executed clean" in r.detail
    assert "instance-cap: 2" in r.detail


def test_tm1_mutation_positive_real_repo_artifact():
    # Independent mutation positive against a REAL long-committed artifact
    # (no scratch monkeypatch): the founding artifact's own .counts read.
    probe = subprocess.run(
        ["git", "cat-file", "-e", "HEAD:eval_results/issue_2330/split_ids.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("founding artifact not committed at HEAD")
    ok = _run(
        "P0: `jq '.counts.train_10k' eval_results/issue_2330/split_ids.json` → expect 10000.\n"
    )
    assert ok.status == "PASS", ok.detail
    mutated = _run(
        "P0: `jq '.counts.train_10k' eval_results/issue_2330/split_ids.json` → expect 9999.\n"
    )
    assert mutated.status == "FAIL"
    assert "expectation mismatch" in mutated.detail


def test_real_2588_v2_file_fails_and_v3_passes():
    # In-corpus mutation pair: the founding defect plan FAILs; its corrected
    # successor PASSes (both against the real repo root).
    hits_v2 = glob(str(REPO_ROOT / "tasks" / "*" / "2588" / "plans" / "v2.md"))
    hits_v3 = glob(str(REPO_ROOT / "tasks" / "*" / "2588" / "plans" / "v3.md"))
    if not hits_v2 or not hits_v3:
        pytest.skip("tasks/*/2588/plans/{v2,v3}.md absent (task folders move)")
    r2 = _run(Path(hits_v2[0]).read_text())
    assert r2.status == "FAIL"
    assert "rc=5" in r2.detail
    r3 = _run(Path(hits_v3[0]).read_text())
    assert r3.status == "PASS", r3.detail


def test_registered_in_checks():
    assert verify_plan.check_jq_probe_dryrun in verify_plan.CHECKS


def test_docstring_conditional_enumeration_carries_71_72():
    # The c53-c56 house pattern, LAST-entry form: while 71/72 are terminal
    # entries the mid-list `"71,"` form cannot match.
    assert "70, 71, 72" in verify_plan.__doc__
