# ruff: noqa: E402
"""Pure attribution/estimation functions of the #1108 repo file audit.

Synthetic path lists only — no network, no repo state: attribution regex rows
(incl. the ``issue-262`` hyphen shape + named legacy buckets), the
conservation identity, keep_rule recomputation, and the cited_by exclusion
logic that splits ready-to-paste from UNSAFE-cited command blocks.
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
