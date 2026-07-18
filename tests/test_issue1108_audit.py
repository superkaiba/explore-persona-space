# ruff: noqa: E402
"""Pure attribution/estimation functions of the #1108/#1141 repo file audit.

Synthetic path lists / fixture stubs only — no network, no repo state:
attribution regex rows (incl. the ``issue-262`` hyphen shape + named legacy
buckets), the conservation identity, keep_rule recomputation, the cited_by
exclusion logic that splits ready-to-paste from UNSAFE-cited command blocks,
and the #1141 extensions (commit classification + scan, overflow era
set-difference, LFS rollups, floor anchors, softened-options rendering, the
zero-write AST invariant).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue1108_repo_file_audit as A
import pytest


class TestAttribution:
    @pytest.mark.parametrize(
        ("path", "kind", "task_id", "bucket"),
        [
            # --- task patterns (first-match-wins order) ---
            ("adapters/issue_397/cellA/adapter_model.safetensors", "task", 397, ""),
            ("adapters/issue543/x.bin", "task", 543, ""),
            ("adapters/issue112_convergence/y.json", "task", 112, ""),
            ("adapters/issue-170/z.safetensors", "task", 170, ""),
            ("issue545_rows/a.json", "task", 545, ""),
            ("issue_490/b.json", "task", 490, ""),
            ("issue816_persona_vectors/c.pt", "task", 816, ""),
            ("issue458_pair_ab/d.pt", "task", 458, ""),
            ("issue205/e.txt", "task", 205, ""),
            ("issue1090/f.safetensors", "task", 1090, ""),
            ("issue-262/g.safetensors", "task", 262, ""),  # hyphen shape
            ("i398_marker/h.bin", "task", 398, ""),
            ("i385_x/i.bin", "task", 385, ""),
            # --- adapters-side legacy task shapes (live-census additions) ---
            ("adapters/i474_loc_A1/adapter.bin", "task", 474, ""),
            ("adapters/i533bw_role_bare_seed1337/x.bin", "task", 533, ""),
            ("adapters/exp381-anchor-seed137/y.bin", "task", 381, ""),
            ("adapters/c_issue506_qwen3_32b/z.bin", "task", 506, ""),
            # --- named legacy buckets ---
            ("adapters/T_context_foo/j.bin", "named_prefix", None, "adapters/T_context_*"),
            ("adapters/T_format_seed137_train/j2.bin", "named_prefix", None, "adapters/T_*"),
            ("adapters/cp_armB_strong_comedian_s42/j3.bin", "named_prefix", None, "adapters/cp_*"),
            ("adapters/marker-persona_swe-lora/j4.bin", "named_prefix", None, "adapters/marker*"),
            ("adapters/mbv2_C1_benign/j5.bin", "named_prefix", None, "adapters/mbv2_*"),
            ("adapters/mb_C1_p2/j6.bin", "named_prefix", None, "adapters/mb_*"),
            ("adapters/capability_leakage/j7.bin", "named_prefix", None, "adapters/*_leakage"),
            (
                "adapters/villain_lr1e-05_ep3/j8.bin",
                "named_prefix",
                None,
                "adapters/<persona>_lr*_ep*",
            ),
            (
                "adapters/install-validated-reladder/k.bin",
                "named_prefix",
                None,
                "adapters/install-validated-reladder",
            ),
            ("leakage_experiment/l.json", "named_prefix", None, "leakage_experiment"),
            ("models/m.bin", "named_prefix", None, "models"),
            ("single_token_multi_source/n.pt", "named_prefix", None, "single_token_multi_source"),
            ("leakage_i81/o.json", "named_prefix", None, "leakage_i81"),
            ("benign_first/p.json", "named_prefix", None, "benign_first"),
            ("eval_results/q.json", "named_prefix", None, "eval_results"),
            ("router_acceptance/r.json", "named_prefix", None, "router_acceptance"),
            ("single_token_sweep/s.json", "named_prefix", None, "single_token_sweep"),
            (".gitattributes", "named_prefix", None, "root"),  # root files
            ("README.md", "named_prefix", None, "root"),
            # --- unattributed (diagnostic prefixes) ---
            ("mystery_dir/t.bin", "unattributed", None, "mystery_dir"),
            ("adapters/not_issue_shaped/u.bin", "unattributed", None, "adapters/not_issue_shaped"),
        ],
    )
    def test_attribute_path(self, path, kind, task_id, bucket):
        assert A.attribute_path(path) == (kind, task_id, bucket)

    def test_task_patterns_win_over_named_buckets(self):
        # first-match-wins: a task-shaped adapters/ path never falls into a bucket
        assert A.attribute_path("adapters/issue_397/checkpoint-100/x.bin")[0] == "task"


class TestLadderSplit:
    def test_ladder_file(self):
        parent, rung_dir, step = A.ladder_split("adapters/issue_397/cellA/checkpoint-500/x.bin")
        assert parent == "adapters/issue_397/cellA"
        assert rung_dir == "adapters/issue_397/cellA/checkpoint-500"
        assert step == 500

    def test_final_file_is_not_ladder(self):
        assert A.ladder_split("adapters/issue_397/cellA/adapter_model.safetensors") is None

    def test_rung_named_top_prefix_does_not_match(self):
        # the pinned regex requires /checkpoint-\d+/ as a nested dir component
        assert A.ladder_split("issue466_x_step1600/adapter.bin") is None


SYNTHETIC = [
    # task 397 (2 parents; parent cellA has 3 rungs, parent cellB has 1)
    A.FileEntry("adapters/issue_397/cellA/checkpoint-100/a.bin", 10),
    A.FileEntry("adapters/issue_397/cellA/checkpoint-500/b.bin", 20),
    A.FileEntry("adapters/issue_397/cellA/checkpoint-1000/c.bin", 30),
    A.FileEntry("adapters/issue_397/cellA/adapter_model.safetensors", 5),
    A.FileEntry("adapters/issue_397/cellB/checkpoint-50/d.bin", 40),
    # task 262 (hyphen shape, no ladder)
    A.FileEntry("issue-262/final.safetensors", 7),
    # named bucket + root + unattributed
    A.FileEntry("leakage_experiment/e.json", 1),
    A.FileEntry(".gitattributes", 1),
    A.FileEntry("mystery_dir/f.bin", 2),
]


class TestAggregateConservation:
    def test_conservation_identity_holds(self):
        agg = A.aggregate(SYNTHETIC)
        assert agg.n_files == len(SYNTHETIC)
        assert agg.task_resolved_files == 6
        assert agg.named_prefix_files == 2  # leakage_experiment + root
        assert agg.unattributed_files == 1
        assert (
            agg.task_resolved_files + agg.named_prefix_files + agg.unattributed_files == agg.n_files
        )

    def test_ladder_and_rung_counts(self):
        agg = A.aggregate(SYNTHETIC)
        assert agg.n_ladder_files == 4
        assert agg.bytes_ladder == 100
        assert agg.n_rung_dirs == 4
        ts = agg.tasks[397]
        assert ts.n_files == 5
        assert ts.n_ladder_files == 4
        assert ts.n_final_files == 1
        assert ts.n_rungs == 4
        assert set(ts.rungs) == {"adapters/issue_397/cellA", "adapters/issue_397/cellB"}


class TestKeepRule:
    def test_conservative_prune_keeps_highest_step_per_parent(self):
        agg = A.aggregate(SYNTHETIC)
        rungs_a = agg.tasks[397].rungs["adapters/issue_397/cellA"]
        kept, pruned = A.conservative_prune(rungs_a)
        assert kept == "adapters/issue_397/cellA/checkpoint-1000"
        assert pruned == [
            "adapters/issue_397/cellA/checkpoint-100",
            "adapters/issue_397/cellA/checkpoint-500",
        ]
        # single-rung parent: nothing pruned (the lone rung IS the max)
        rungs_b = agg.tasks[397].rungs["adapters/issue_397/cellB"]
        kept_b, pruned_b = A.conservative_prune(rungs_b)
        assert kept_b == "adapters/issue_397/cellB/checkpoint-50"
        assert pruned_b == []

    def test_numeric_not_lexicographic_step_ordering(self):
        rungs = {
            "p/checkpoint-90": A.RungStats(1, 1),
            "p/checkpoint-800": A.RungStats(1, 1),
        }
        kept, pruned = A.conservative_prune(rungs)
        assert kept == "p/checkpoint-800"  # lexicographic would keep -90
        assert pruned == ["p/checkpoint-90"]

    def test_keep_rule_string_pinned(self):
        assert A.KEEP_RULE == (
            "keep the single highest-step checkpoint-* dir per PARENT adapter dir, "
            "plus all non-checkpoint files"
        )


class TestCitations:
    CORPUS = (  # tuple: RUF012 (mutable class attribute)
        # #532's body cites #474's parent subfolder + an early rung step
        ("#532 (body.md)", "Reused adapters/issue_474/loc_A1 at checkpoint-30 (epoch 1)."),
        # a script cites a full rung path
        ("scripts/foo.py", 'ADAPTER = "adapters/issue_397/cellA/checkpoint-500"'),
        # figures/eval_results path mentions must NOT count as model-repo citations
        ("#900 (body.md)", "see figures/issue_397/plot.png and eval_results/issue_397/x.json"),
        # step mention WITHOUT any path token for that parent
        ("#901 (body.md)", "we stopped at checkpoint-100 in an unrelated run"),
    )

    def test_rung_cited_by_parent_plus_step(self):
        index = A.build_citation_index(self.CORPUS)
        assert A.cited_by_for_rung(index, "adapters/issue_474/loc_A1", 30) == ["#532 (body.md)"]
        # same parent, uncited step
        assert A.cited_by_for_rung(index, "adapters/issue_474/loc_A1", 999) == []

    def test_full_rung_path_citation_counts(self):
        index = A.build_citation_index(self.CORPUS)
        assert A.cited_by_for_rung(index, "adapters/issue_397/cellA", 500) == ["scripts/foo.py"]

    def test_figures_and_step_only_mentions_do_not_cite(self):
        index = A.build_citation_index(self.CORPUS)
        # figures/issue_397 + eval_results/issue_397 are slash-preceded -> skipped;
        # #901's bare checkpoint-100 has no parent token -> not a citation
        assert A.cited_by_for_rung(index, "adapters/issue_397/cellA", 100) == []

    def test_ancestor_citation_flags_descendant_rung(self):
        # tree-level citation + step co-mention flags the rung (conservative)
        corpus = [("#902 (body.md)", "adapters/issue_397 dose ladder at checkpoint-100")]
        index = A.build_citation_index(corpus)
        assert A.cited_by_for_rung(index, "adapters/issue_397/cellA", 100) == ["#902 (body.md)"]

    def test_split_ready_vs_unsafe(self):
        index = A.build_citation_index(self.CORPUS)
        pruned = [
            ("adapters/issue_397/cellA/checkpoint-100", 100),  # uncited
            ("adapters/issue_397/cellA/checkpoint-500", 500),  # cited by scripts/foo.py
        ]
        ready, unsafe = A.split_ready_vs_unsafe(pruned, index)
        assert ready == ["adapters/issue_397/cellA/checkpoint-100"]
        assert unsafe == {"adapters/issue_397/cellA/checkpoint-500": ["scripts/foo.py"]}

    def test_tree_cited_by(self):
        index = A.build_citation_index(self.CORPUS)
        assert A.cited_by_for_tree(index, "adapters/issue_474") == ["#532 (body.md)"]
        assert A.cited_by_for_tree(index, "adapters/issue_999") == []


class TestCommandRendering:
    def test_multi_dir_block_chunks_and_never_executes(self):
        dirs = [f"adapters/issue_397/cellA/checkpoint-{i}" for i in range(1, 251)]
        block = A.render_delete_block(
            397,
            dirs,
            status="completed",
            classification="useful",
            files_freed=1000,
            bytes_freed=10**9,
            cited_by=[],
            chunk_size=100,
        )
        # 250 dirs at <=100 ops/commit -> 3 create_commit calls in the TEXT
        assert block.count("api.create_commit(") == 3
        assert "CommitOperationDelete" in block
        assert "cited_by: []" in block

    def test_single_dir_uses_delete_folder_oneliner(self):
        block = A.render_delete_block(
            397,
            ["adapters/issue_397/cellA/checkpoint-100"],
            status="completed",
            classification="useful",
            files_freed=8,
            bytes_freed=10**6,
            cited_by=[],
        )
        assert "api.delete_folder(" in block
        assert "create_commit" not in block

    def test_cited_block_is_fully_commented_out(self):
        block = A.render_delete_block(
            397,
            ["adapters/issue_397/cellA/checkpoint-500"],
            status="completed",
            classification="useful",
            files_freed=8,
            bytes_freed=10**6,
            cited_by=["scripts/foo.py"],
            commented=True,
        )
        code = block.split("```python\n", 1)[1].split("\n```", 1)[0]
        assert all(ln.startswith("#") for ln in code.split("\n") if ln.strip())
        assert "cited_by: ['scripts/foo.py']" in block

    def test_runtime_path_never_imports_deletion_ops(self):
        """Acceptance #5: zero CommitOperationDelete/delete_folder EXECUTION in
        the shipped script's runtime path — the strings exist only inside
        generated command TEXT."""
        import inspect

        src = inspect.getsource(A)
        # the only occurrences of the deletion APIs are inside string literals
        # of the renderers; the module never imports them.
        assert "from huggingface_hub import CommitOperationDelete" not in src.replace(
            '"from huggingface_hub import CommitOperationDelete', ""
        )
        assert not hasattr(A, "CommitOperationDelete")


# ---------------------------------------------------------------------------
# #1141 extension tests (plan `## Test scope` items 1-7) — pure CPU, no
# network; fixture FileEntry / commit-record stubs only.
# ---------------------------------------------------------------------------

from datetime import UTC, date, datetime
from types import SimpleNamespace


def _commit(cid: str, iso_ts: str, title: str) -> SimpleNamespace:
    """GitCommitInfo-like fixture record (commit_id / created_at / title)."""
    return SimpleNamespace(
        commit_id=cid,
        created_at=datetime.fromisoformat(iso_ts).replace(tzinfo=UTC),
        title=title,
    )


class _StubApi:
    """Network-boundary fake for scan_commits — ``list_repo_commits`` mirrors
    the real ``HfApi.list_repo_commits`` signature by construction (the
    external API boundary is the one sanctioned fake seam)."""

    def __init__(self, commits):
        self._commits = commits

    def list_repo_commits(self, repo_id, *, repo_type=None, token=None, revision=None):
        return list(self._commits)


class TestClassifyCommit:
    def test_classify_commit(self):
        # the three pinned live titles (plan assumption 8) + an "other" case
        assert A.classify_commit("quota probe (auto-deleted)") == "probe"
        assert A.classify_commit("remove quota probe") == "probe-cleanup"
        assert A.classify_commit("Upload folder using huggingface_hub") == "upload"
        assert A.classify_commit("task #1141: unrelated maintenance") == "other"
        # upload arm is a prefix match (upload_file default titles count too)
        assert A.classify_commit("Upload adapter_model.safetensors with huggingface_hub") == (
            "upload"
        )


class TestScanCommits:
    FIXTURE = (
        _commit("c_probe", "2026-07-17T18:50:00", "quota probe (auto-deleted)"),
        _commit("c_cleanup", "2026-07-17T18:51:00", "remove quota probe"),
        _commit("c_fold1", "2026-07-17T14:32:00", "Upload folder using huggingface_hub"),
        _commit("c_fold2", "2026-07-17T14:34:00", "Upload folder using huggingface_hub"),
        _commit("c_file", "2026-07-16T09:00:00", "Upload x.bin with huggingface_hub"),
        _commit("c_other", "2026-07-10T12:00:00", "task #900: bookkeeping"),
        # OUTSIDE the since window — must be excluded from windowed counts:
        _commit("c_old", "2026-07-01T00:00:00", "Upload folder using huggingface_hub"),
    )

    def test_scan_commits_first_nonprobe_upload(self):
        out = A.scan_commits(_StubApi(self.FIXTURE), A.MODEL_REPO, since=date(2026, 7, 6))
        assert out["n_commits_full_history"] == 7
        assert out["n_commits_scanned"] == 6  # c_old excluded by the window
        # probe exclusion: probe titles never count as uploads
        assert out["n_probe"] == 1
        assert out["n_probe_cleanup"] == 1
        assert out["n_upload"] == 3
        assert out["n_other"] == 1
        assert out["n_folder_push"] == 2
        # chronologically EARLIEST upload-class commit in the window
        first = out["first_nonprobe_upload"]
        assert first["commit_id"] == "c_file"
        assert first["date_utc"] == "2026-07-16T09:00:00Z"
        assert first["title"] == "Upload x.bin with huggingface_hub"
        assert out["per_day_upload_counts"] == {"2026-07-16": 1, "2026-07-17": 2}

    def test_full_history_scan_with_era_cutover(self):
        out = A.scan_commits(_StubApi(self.FIXTURE), A.OVERFLOW_REPO, era_cutover=date(2026, 7, 7))
        assert out["n_commits_scanned"] == 7
        assert out["n_commits_on_or_before_cutover"] == 1  # c_old only
        assert out["n_commits_after_cutover"] == 6


class TestOverflowEra:
    def test_overflow_era_set_difference(self):
        overflow = [
            # covered by a pointer prefix -> post-#1108 reroutes
            A.FileEntry("adapters/i1090_c5/checkpoint-2/a.bin", 100, 90),
            A.FileEntry("adapters/i1090_c5/adapter_model.safetensors", 20, 15),
            # NOT covered by any pointer prefix -> pre-#1108 (#564-era)
            A.FileEntry("issue564_quota/blob.bin", 50, 50),
            A.FileEntry("legacy_dir/x.json", 5, 0),
        ]
        prefixes = ["adapters/i1090_c5"]
        era = A.overflow_era_split(overflow, prefixes)
        assert era["post_1108_files"] == 2
        assert era["post_1108_bytes"] == 120
        assert era["pre_1108_files"] == 2
        assert era["pre_1108_bytes"] == 55

    def test_pointer_prefix_enumeration(self):
        entries = [
            A.FileEntry("adapters/i1090_c5/OVERFLOW_POINTER.json", 1, 0),
            A.FileEntry("adapters/i1090_c5/other.bin", 1, 0),
            A.FileEntry("adapters/issue_397/x.bin", 1, 0),
            # a root-level pointer carries no prefix and is skipped
            A.FileEntry("OVERFLOW_POINTER.json", 1, 0),
        ]
        assert A.find_overflow_pointer_prefixes(entries) == ["adapters/i1090_c5"]


class TestLfsRollup:
    def test_lfs_split_rollup(self):
        entries = [
            A.FileEntry("adapters/issue_397/cellA/checkpoint-100/a.bin", 100, 95),
            A.FileEntry("adapters/issue_397/cellA/adapter_model.safetensors", 40, 40),
            A.FileEntry("adapters/issue_397/cellA/config.json", 3, 0),
            A.FileEntry("leakage_experiment/e.bin", 10, 10),
            A.FileEntry("mystery_dir/f.bin", 7, 0),
        ]
        agg = A.aggregate(entries)
        assert agg.n_lfs_bytes == 145
        ts = agg.tasks[397]
        assert ts.lfs_bytes == 135
        assert ts.trees["adapters/issue_397"].n_lfs_bytes == 135
        rung = ts.rungs["adapters/issue_397/cellA"]["adapters/issue_397/cellA/checkpoint-100"]
        assert rung.n_lfs_bytes == 95
        assert agg.named_prefixes["leakage_experiment"].n_lfs_bytes == 10
        # unattributed bytes + examples ride along (acceptance #2)
        assert agg.unattributed_bytes == 7
        assert agg.unattributed_examples == ["mystery_dir/f.bin"]

    def test_file_entry_lfs_default_backcompat(self):
        # 2-arg constructor (pre-#1141 shape) still works: lfs_size defaults 0
        assert A.FileEntry("x/y.bin", 5).lfs_size == 0


class TestFloorAnchor:
    def test_floor_anchor_fails_loud(self):
        with pytest.raises(AssertionError, match="floor anchor violated"):
            A.assert_floor_anchor(A.MODEL_REPO, 109_999)
        with pytest.raises(AssertionError, match="floor anchor violated"):
            A.assert_floor_anchor(A.OVERFLOW_REPO, 3_499)

    def test_floor_anchor_passes_and_reports_delta(self):
        out = A.assert_floor_anchor(A.MODEL_REPO, 117_050)
        assert out == {
            "floor": 110_000,
            "run_time_count": 117_050,
            "delta_vs_2026_07_18_anchor": 0,
        }

    def test_urgency_formula_registered_value(self):
        # the registered rule: max(0, count - 100_000 + 1_000) = 18,050 at 117,050
        assert A.urgency_files_to_free(117_050) == 18_050
        assert A.urgency_files_to_free(99_000) == 0


def _fixture_audit() -> dict:
    """Minimal-but-complete audit dict for render_softened_options."""
    commits = {
        "since": "2026-07-06",
        "n_commits_scanned": 6,
        "n_probe": 1,
        "n_probe_cleanup": 1,
        "n_upload": 3,
        "n_other": 1,
        "n_folder_push": 2,
        "upload_count_label": "LOWER BOUND — title-classifier based",
        "first_nonprobe_upload": {
            "commit_id": "c_file",
            "date_utc": "2026-07-16T09:00:00Z",
            "title": "Upload x.bin with huggingface_hub",
        },
        "per_day_upload_counts": {"2026-07-16": 1, "2026-07-17": 2},
        "net_growth_vs_rejection_anchor": 17_000,
    }
    c1_row = {
        "tree": A.C1_TREE,
        "task": A.C1_TASK,
        "status": "completed",
        "files": 7_668,
        "bytes": int(219.6e9),
        "lfs_bytes": int(216.7e9),
        "cited_by": ["#397 (body.md)"],
    }
    c2_rows = [
        {
            "task": 397,
            "status": "completed",
            "files": 5_000,
            "bytes": 10**11,
            "lfs_bytes": 9 * 10**10,
            "n_pruned_rungs": 12,
            "cited_by": ["#532 (body.md)"],
        },
        {
            "task": 601,
            "status": "archived",
            "files": 300,
            "bytes": 10**9,
            "lfs_bytes": 10**9,
            "n_pruned_rungs": 2,
            "cited_by": [],
        },
    ]
    softened = {
        "lfs_accounting_note": "every LFS figure is the HEAD-tree lfs_bytes sum",
        "a_do_nothing": {
            "files_freed": 0,
            "lfs_bytes_freed": 0,
            "run_time_count": 117_050,
            "net_growth_since_rejection": 17_000,
            "n_upload_commits_since": 3,
            "upload_count_label": "LOWER BOUND",
            "n_other_commits_since": 1,
            "ongoing_cost_note": "overflow artifacts are PRIVATE and pointer-mediated",
        },
        "b_migrate_overflow": {
            "files_moved": 3_880,
            "bytes_moved": 4 * 10**10,
            "lfs_bytes_moved": 39 * 10**9,
            "by_era": {
                "pre_1108_files": 880,
                "pre_1108_bytes": 10**10,
                "post_1108_files": 3_000,
                "post_1108_bytes": 3 * 10**10,
                "mechanism": "commit-scan + pointer set-difference",
            },
            "destination_note": "destinations derive from the OVERFLOW PATHS themselves",
            "public_storage_caveat": "moves private bytes into PUBLIC storage accounting",
            "user_only_steps": ["delete pointers + overflow contents"],
        },
        "c1_archive_issue_397": {
            **c1_row,
            "archive_note": "archive-then-delete (wandb-archive precedent)",
            "user_only_steps": ["delete adapters/issue_397 from canonical"],
        },
        "c2_prune_terminal_rungs": {
            "files_freed_upper_bound": 5_300,
            "bytes_freed_upper_bound": 101 * 10**9,
            "lfs_bytes_freed_upper_bound": 91 * 10**9,
            "bound_label": "selection-blind UPPER BOUND",
            "keep_rule": A.KEEP_RULE,
            "keep_rule_caveat": A.KEEP_RULE_CAVEAT,
            "user_must_verify": A.USER_MUST_VERIFY_LINE,
            "per_task": c2_rows,
            "advisory_trigger": {
                "rule": "prunable_rung_files >= 20% (ADVISORY)",
                "prunable_pct_files": 4.53,
                "fired": False,
            },
        },
    }
    rec = A.build_recommendation_draft(
        n_files=117_050,
        commits_summary=commits,
        c1_row=c1_row,
        c2_totals={"files": 5_300},
    )
    return {
        "header": {"enumerated_at": "2026-07-18T01:00:00Z"},
        "repos": {
            "canonical": {
                "repo_id": A.MODEL_REPO,
                "enumeration": {
                    "complete": True,
                    "wall_seconds": 120.0,
                    "floor_anchor": {
                        "floor": 110_000,
                        "run_time_count": 117_050,
                        "delta_vs_2026_07_18_anchor": 0,
                    },
                },
                "commits_since_2026-07-06": commits,
            },
        },
        "options": {"softened_options": softened},
        "recommendation_draft": rec,
    }


class TestSoftenedOptionsRendering:
    def test_summary_rows_carry_consumer_check(self):
        """Plan `## Test scope` item 6 (the Alternatives Must-Fix, mechanized):
        every (c1)/(c2) row in the rendered report carries a cited_by result,
        and (c2) carries the USER-must-verify + UPPER-BOUND (+ verbatim
        KEEP_RULE_CAVEAT) lines BEFORE any delete-command reference."""
        text = A.render_softened_options(_fixture_audit())

        # --- (c1): cited_by result present, BEFORE the delete command text ---
        c1_start = text.index("### (c1)")
        c1_end = text.index("### (c2)")
        c1_sec = text[c1_start:c1_end]
        assert "cited: #397 (body.md)" in c1_sec
        assert c1_sec.index("cited:") < c1_sec.index("delete_folder")

        # --- (c2): every per-task row carries its cited_by result ---
        c2_sec = text[c1_end : text.index("## Recommendation draft")]
        row_lines = [
            ln
            for ln in c2_sec.splitlines()
            if ln.startswith("| #") and ln.count("|") >= 6  # per-task table rows
        ]
        assert len(row_lines) == 2
        for ln in row_lines:
            assert ("uncited" in ln) or ("cited:" in ln), ln
        assert any("cited: #532 (body.md)" in ln for ln in row_lines)
        assert any(ln.startswith("| #601") and "uncited" in ln for ln in row_lines)

        # --- (c2): UPPER BOUND + verbatim caveat + USER-must-verify all appear
        # BEFORE the (only) delete-command reference (freeing_commands.md) ---
        cmd_pos = c2_sec.index("freeing_commands.md")
        assert c2_sec.index("selection-blind UPPER BOUND") < cmd_pos
        assert c2_sec.index(A.KEEP_RULE_CAVEAT) < cmd_pos
        assert c2_sec.index(A.USER_MUST_VERIFY_LINE) < cmd_pos

    def test_status_wording_never_categorical(self):
        """Acceptance #3 wording: 'not enforced at the current count/shape as
        of <date>', never a bare categorical 'no longer enforced'."""
        text = A.render_softened_options(_fixture_audit())
        assert "not enforced at the current count/shape as of 2026-07-18T01:00:00Z" in text
        # the only occurrence of "no longer enforced" is inside the
        # deliberately-negated parenthetical
        assert 'categorical "no longer enforced"' in text
        assert text.count("no longer enforced") == 1

    def test_recommendation_draft_branches(self):
        # accepting branch (fixture: 3 uploads post-07-07, count > 100k)
        rec = _fixture_audit()["recommendation_draft"]
        assert rec["branch"] == "accepting"
        assert rec["n_upload_commits_post_rejection"] == 3
        assert any("(b) migrate overflow" in r for r in rec["recommended"])
        # c1 meets LFS+terminal but is cited -> conditional wording
        assert any("USER must clear/dismiss" in r for r in rec["recommended"])
        # kill-criterion branch: no post-rejection uploads -> urgency formula
        no_uploads = {"per_day_upload_counts": {}, "n_upload": 0}
        c1_row = {"status": "completed", "lfs_bytes": 0, "cited_by": []}
        rec2 = A.build_recommendation_draft(
            n_files=117_050,
            commits_summary=no_uploads,
            c1_row=c1_row,
            c2_totals={"files": 1},
        )
        assert rec2["branch"] == "unfreeze-urgency"
        assert rec2["files_to_free"] == 18_050


class TestNoWrites:
    def test_no_write_api_call_sites(self):
        """Plan `## Test scope` item 7 / acceptance #7: AST walk — no Call
        node in the script whose callee name is a Hub write API. The
        freeing-command TEXT templates are string literals (ast.Constant),
        exempt by construction."""
        import ast
        import inspect

        write_apis = {
            "upload_file",
            "upload_folder",
            "delete_file",
            "delete_folder",
            "delete_repo",
            "create_repo",
            "super_squash_history",
            "move_repo",
        }
        tree = ast.parse(inspect.getsource(A))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = None
            if isinstance(fn, ast.Attribute):
                name = fn.attr
            elif isinstance(fn, ast.Name):
                name = fn.id
            if name in write_apis:
                offenders.append((name, node.lineno))
        assert offenders == [], f"HF write-API call sites in the runtime path: {offenders}"
