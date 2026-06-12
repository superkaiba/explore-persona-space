"""Regression tests for the epm:results reproducibility-card fallback (#608, #601).

Multi-cell sweeps declare artifacts per cell (an ``adapter_paths`` dict +
per-cell WandB run names) inside the epm:results payload's
``reproducibility_card`` — there is no single --hf-model / --wandb-run
value to pass. ``scripts/verify_uploads.py`` used to hard-MISS the
wandb_run / hf_model rows in that case (false mechanical FAIL the
upload-verifier had to supersede row-by-row, same class as incident #563).
These tests pin: the prose-prefixed JSON extraction (#608's drained-sentinel
note shape), the cross-marker card merge (newest-wins per declared field —
#601: a resume-pass sentinel with ``adapter_paths: {}`` must not shadow the
first marker's 16 verified paths; the ``reproducibility`` key alias), the
per-path HF aggregation, the per-name WandB resolution, and that explicit
single-path declarations win unchanged. Same module-loading conventions as
tests/test_verify_uploads_type_selection.py.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_uploads.py"
_spec = importlib.util.spec_from_file_location("verify_uploads_cf", _SCRIPT)
verify_uploads = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_uploads_cf"] = verify_uploads
_spec.loader.exec_module(verify_uploads)  # type: ignore[union-attr]


_CARD_608 = {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "hf_model_repo": "superkaiba1/explore-persona-space",
    "adapter_paths": {
        "villain:posonly_epoch": "adapters/issue_608/posonly_epoch/villain_seed42",
        "comedian:posonly_epoch": "adapters/issue_608/posonly_epoch/comedian_seed42",
    },
}

# The real #608 note shape: orchestrator prose prefix, then the JSON payload.
_NOTE_608 = (
    "[drained by orchestrator from pod sentinel "
    "/workspace/logs/issue-608-epm_results-1781214523.json — poller could "
    "not read it: file root-owned mode 600 on GCP instance]\n\n"
    '{"event": "sweep_complete", "issue": 608, '
    '"reproducibility_card": {"hf_model_repo": "superkaiba1/explore-persona-space", '
    '"adapter_paths": {"villain:posonly_epoch": '
    '"adapters/issue_608/posonly_epoch/villain_seed42"}}}'
)


# ── _extract_first_json_object ────────────────────────────────────────────────


class TestExtractFirstJsonObject:
    def test_prose_prefixed_note_parses(self):
        payload = verify_uploads._extract_first_json_object(_NOTE_608)
        assert payload is not None
        assert payload["event"] == "sweep_complete"
        assert "reproducibility_card" in payload

    def test_pure_json_note_parses(self):
        assert verify_uploads._extract_first_json_object('{"a": 1}') == {"a": 1}

    def test_brace_in_prose_before_json_is_skipped(self):
        text = 'broken {not json} prefix then {"a": 1}'
        assert verify_uploads._extract_first_json_object(text) == {"a": 1}

    def test_no_json_returns_none(self):
        assert verify_uploads._extract_first_json_object("no payload here") is None

    def test_non_dict_json_returns_none(self):
        assert verify_uploads._extract_first_json_object("[1, 2, 3]") is None


# ── merged_results_card ───────────────────────────────────────────────────────


class TestMergedResultsCard:
    def test_picks_newest_results_event_with_card(self):
        events = [
            {"kind": "epm:results", "note": '{"reproducibility_card": {"hf_model_path": "old"}}'},
            {"kind": "epm:progress", "note": "irrelevant"},
            {"kind": "epm:results", "note": '{"reproducibility_card": {"hf_model_path": "new"}}'},
        ]
        assert verify_uploads.merged_results_card(events) == {"hf_model_path": "new"}

    def test_cardless_repost_does_not_erase_earlier_card(self):
        events = [
            {"kind": "epm:results", "note": '{"reproducibility_card": {"hf_model_path": "p"}}'},
            {"kind": "epm:results", "note": '{"event": "sweep_complete"}'},
        ]
        assert verify_uploads.merged_results_card(events) == {"hf_model_path": "p"}

    def test_no_results_event_returns_none(self):
        assert verify_uploads.merged_results_card([{"kind": "epm:progress", "note": "x"}]) is None

    def test_real_608_note_shape(self):
        card = verify_uploads.merged_results_card([{"kind": "epm:results", "note": _NOTE_608}])
        assert card is not None
        assert card["adapter_paths"] == {
            "villain:posonly_epoch": "adapters/issue_608/posonly_epoch/villain_seed42"
        }

    def test_empty_resume_card_falls_back_per_field(self):
        """The #601 false-FAIL repro: a resume-pass sentinel re-post whose
        card carries adapter_paths: {} must not shadow the first marker's
        full declaration. Newest non-empty wins per field; the merged card
        records provenance for the fields that fell back."""
        events = [
            {
                "kind": "epm:results",
                "ts": "2026-06-11T22:35:04Z",
                "note": (
                    '{"status": "done", "reproducibility": '
                    '{"base_model": "Qwen/Qwen2.5-7B-Instruct", '
                    '"hf_model_repo": "superkaiba1/explore-persona-space", '
                    '"adapter_paths": {"c1": "adapters/issue_601/a", '
                    '"c2": "adapters/issue_601/b"}}}'
                ),
            },
            {
                "kind": "epm:results",
                "ts": "2026-06-11T23:37:19Z",
                "note": (
                    '{"status": "done", "reproducibility": '
                    '{"base_model": "Qwen/Qwen2.5-7B-Instruct", '
                    '"hf_model_repo": "superkaiba1/explore-persona-space", '
                    '"adapter_paths": {}}}'
                ),
            },
        ]
        card = verify_uploads.merged_results_card(events)
        assert card is not None
        assert card["adapter_paths"] == {
            "c1": "adapters/issue_601/a",
            "c2": "adapters/issue_601/b",
        }
        # Non-empty fields of the newest card still win.
        assert card["hf_model_repo"] == "superkaiba1/explore-persona-space"
        assert "adapter_paths @ 2026-06-11T22:35:04Z" in card["_card_provenance"]

    def test_reproducibility_key_alias_accepted(self):
        """#601's producer named the card ``reproducibility``, not
        ``reproducibility_card``."""
        events = [
            {"kind": "epm:results", "note": '{"reproducibility": {"hf_model_path": "p"}}'},
        ]
        assert verify_uploads.merged_results_card(events) == {"hf_model_path": "p"}

    def test_canonical_key_wins_over_alias(self):
        events = [
            {
                "kind": "epm:results",
                "note": (
                    '{"reproducibility_card": {"hf_model_path": "canonical"}, '
                    '"reproducibility": {"hf_model_path": "alias"}}'
                ),
            },
        ]
        assert verify_uploads.merged_results_card(events) == {"hf_model_path": "canonical"}

    def test_newest_wins_per_field_across_cards(self):
        events = [
            {
                "kind": "epm:results",
                "ts": "t0",
                "note": (
                    '{"reproducibility_card": {"adapter_paths": {"c": "old"}, '
                    '"wandb_project": "huggingface"}}'
                ),
            },
            {
                "kind": "epm:results",
                "ts": "t1",
                "note": '{"reproducibility_card": {"adapter_paths": {"c": "new"}}}',
            },
        ]
        card = verify_uploads.merged_results_card(events)
        assert card["adapter_paths"] == {"c": "new"}
        assert card["wandb_project"] == "huggingface"
        assert "wandb_project @ t0" in card["_card_provenance"]
        assert "adapter_paths" not in card["_card_provenance"]

    def test_all_cards_empty_returns_none(self):
        events = [
            {"kind": "epm:results", "note": '{"reproducibility": {"adapter_paths": {}}}'},
        ]
        assert verify_uploads.merged_results_card(events) is None

    def test_single_card_has_no_provenance(self):
        card = verify_uploads.merged_results_card([{"kind": "epm:results", "note": _NOTE_608}])
        assert "_card_provenance" not in card


# ── _card_from_provenance (GCP-lane driver sentinels, #599) ───────────────────

# The real #599 GCP-lane driver sentinel shape: no reproducibility_card,
# per-seed provenance under production_provenance.
_NOTE_599 = (
    "[drained from GCP sentinel /workspace/logs/issue-599-epm_results.json]\n\n"
    '{"issue": 599, "phase": "production_complete", '
    '"production_provenance": {'
    '"seed42": {"loss_shape": "full_response", "initial_train_loss": 0.4316, '
    '"hf_adapter_subfolder": "issue_599_fullresp/marker_seed42"}, '
    '"seed137": {"loss_shape": "full_response", '
    '"hf_adapter_subfolder": "issue_599_fullresp/marker_seed137"}}}'
)


class TestCardFromProvenance:
    def test_gcp_sentinel_synthesizes_adapter_paths(self):
        """The #599 false-MISS repro: a card-less GCP-lane sentinel yields a
        synthesized card whose adapter_paths come from
        production_provenance.<seed>.hf_adapter_subfolder."""
        card = verify_uploads.merged_results_card([{"kind": "epm:results", "note": _NOTE_599}])
        assert card is not None
        assert card["adapter_paths"] == {
            "seed42": "issue_599_fullresp/marker_seed42",
            "seed137": "issue_599_fullresp/marker_seed137",
        }
        assert "production_provenance" in card["_card_provenance"]

    def test_explicit_card_wins_over_provenance(self):
        payload = {
            "reproducibility_card": {"hf_model_path": "explicit"},
            "production_provenance": {"seed42": {"hf_adapter_subfolder": "synth"}},
        }
        assert verify_uploads._card_from_payload(payload) == {"hf_model_path": "explicit"}

    def test_top_level_wandb_hints_carried(self):
        payload = {
            "production_provenance": {"seed42": {"hf_adapter_subfolder": "a/b"}},
            "wandb_project": "huggingface",
            "hf_model_repo": "superkaiba1/explore-persona-space",
        }
        card = verify_uploads._card_from_payload(payload)
        assert card["adapter_paths"] == {"seed42": "a/b"}
        assert card["wandb_project"] == "huggingface"
        assert card["hf_model_repo"] == "superkaiba1/explore-persona-space"

    def test_per_cell_wandb_run_names_collected(self):
        payload = {
            "production_provenance": {
                "seed42": {"hf_adapter_subfolder": "a", "wandb_run_name": "run42"},
                "seed137": {"hf_adapter_subfolder": "b", "wandb_run_name": "run137"},
            },
        }
        card = verify_uploads._card_from_payload(payload)
        assert card["wandb_run_names"] == {"seed42": "run42", "seed137": "run137"}

    def test_no_usable_provenance_returns_none(self):
        payload = {"production_provenance": {"seed42": {"loss_shape": "full_response"}}}
        assert verify_uploads._card_from_payload(payload) is None
        assert verify_uploads._card_from_payload({"phase": "done"}) is None

    def test_older_synthesized_note_not_misattributed(self):
        """When a newer explicit card merges with an older synthesized one,
        the synthesis note must not ride along and misattribute the
        explicit fields; the cross-marker fallback note still records the
        fields that fell back."""
        events = [
            {"kind": "epm:results", "ts": "t0", "note": _NOTE_599},
            {
                "kind": "epm:results",
                "ts": "t1",
                "note": '{"reproducibility_card": {"wandb_project": "explicit-project"}}',
            },
        ]
        card = verify_uploads.merged_results_card(events)
        assert card["wandb_project"] == "explicit-project"
        assert card["adapter_paths"]["seed42"] == "issue_599_fullresp/marker_seed42"
        assert "synthesized" not in card["_card_provenance"]
        assert "adapter_paths @ t0" in card["_card_provenance"]

    def test_provenance_card_satisfies_hf_model_row(self):
        """run_verification integration: the synthesized card resolves the
        hf_model row instead of the strict MISSING (#599)."""
        card = verify_uploads.merged_results_card([{"kind": "epm:results", "note": _NOTE_599}])
        with (
            patch.object(verify_uploads, "_load_results_card", return_value=card),
            patch.object(
                verify_uploads,
                "check_hf_hub_path",
                return_value={"status": "OK", "url": "u", "file_count": 3},
            ),
            # Convention probe patched to None to pin the strict MISSING
            # wandb row (and keep the test network-free).
            patch.object(verify_uploads, "check_wandb_runs_convention_project", return_value=None),
        ):
            report = verify_uploads.run_verification(599, experiment_type="training")
        assert report["checks"]["hf_model"]["status"] == "OK"
        assert "production_provenance" in report["checks"]["hf_model"]["detail"]
        # The sentinel declares no wandb fields, so that row still MISSes.
        assert report["checks"]["wandb_run"]["status"] == "MISSING"


# ── check_hf_model_from_card ──────────────────────────────────────────────────


class TestCheckHfModelFromCard:
    def test_no_model_paths_returns_none(self):
        assert verify_uploads.check_hf_model_from_card({"base_model": "Qwen"}) is None

    def test_all_adapter_paths_resolve(self):
        calls = []

        def fake_check(repo, path, repo_type="model", revision=None):
            calls.append((repo, path))
            return {"status": "OK", "url": "u", "file_count": 3}

        with patch.object(verify_uploads, "check_hf_hub_path", side_effect=fake_check):
            res = verify_uploads.check_hf_model_from_card(_CARD_608)
        assert res["status"] == "OK"
        assert res["file_count"] == 6
        assert res["source"] == "epm:results reproducibility_card"
        assert {p for _, p in calls} == set(_CARD_608["adapter_paths"].values())
        assert all(repo == "superkaiba1/explore-persona-space" for repo, _ in calls)

    def test_one_absent_adapter_misses_and_names_it(self):
        def fake_check(repo, path, repo_type="model", revision=None):
            if path.endswith("comedian_seed42"):
                return {"status": "MISSING", "url": "", "detail": "No files"}
            return {"status": "OK", "url": "u", "file_count": 3}

        with patch.object(verify_uploads, "check_hf_hub_path", side_effect=fake_check):
            res = verify_uploads.check_hf_model_from_card(_CARD_608)
        assert res["status"] == "MISSING"
        assert "comedian_seed42" in res["detail"]

    def test_transport_error_reports_error_not_missing(self):
        with patch.object(
            verify_uploads,
            "check_hf_hub_path",
            return_value={"status": "ERROR", "url": "", "detail": "503"},
        ):
            res = verify_uploads.check_hf_model_from_card(_CARD_608)
        assert res["status"] == "ERROR"

    def test_single_hf_model_path_accepted(self):
        with patch.object(
            verify_uploads,
            "check_hf_hub_path",
            return_value={"status": "OK", "url": "u", "file_count": 2},
        ) as mock_check:
            res = verify_uploads.check_hf_model_from_card({"hf_model_path": "adapters/issue_9/a"})
        assert res["status"] == "OK"
        mock_check.assert_called_once()

    def test_merge_provenance_threaded_into_detail(self):
        """A merged card's cross-marker fallback note lands in the row detail
        so the report says which marker actually declared the paths (#601)."""
        card = dict(_CARD_608)
        card["_card_provenance"] = (
            "field(s) declared by an earlier epm:results marker, not the "
            "latest: adapter_paths @ 2026-06-11T22:35:04Z"
        )
        with patch.object(
            verify_uploads,
            "check_hf_hub_path",
            return_value={"status": "OK", "url": "u", "file_count": 3},
        ):
            res = verify_uploads.check_hf_model_from_card(card)
        assert res["status"] == "OK"
        assert "adapter_paths @ 2026-06-11T22:35:04Z" in res["detail"]


# ── check_wandb_from_card ─────────────────────────────────────────────────────


class TestCheckWandbFromCard:
    def test_no_wandb_fields_returns_none(self):
        assert verify_uploads.check_wandb_from_card(_CARD_608) is None

    def test_single_run_path_delegates(self):
        with patch.object(
            verify_uploads, "check_wandb_run", return_value={"status": "OK", "url": "u"}
        ) as mock_run:
            res = verify_uploads.check_wandb_from_card({"wandb_run_path": "e/p/runs/abc"})
        assert res["status"] == "OK"
        assert res["source"] == "epm:results reproducibility_card"
        mock_run.assert_called_once_with("e/p/runs/abc")

    def test_run_names_resolve_per_cell(self):
        card = {
            "wandb_project": "huggingface",
            "wandb_run_names": {
                "villain:posonly_epoch": "issue608_posonly_epoch_villain_seed42",
                "comedian:posonly_epoch": "issue608_posonly_epoch_comedian_seed42",
            },
        }
        with patch.object(
            verify_uploads, "check_wandb_runs_by_name", return_value={"status": "OK", "url": "u"}
        ) as mock_names:
            res = verify_uploads.check_wandb_from_card(card)
        assert res["status"] == "OK"
        project_path, names = mock_names.call_args[0]
        assert project_path == "huggingface"
        assert sorted(names) == [
            "issue608_posonly_epoch_comedian_seed42",
            "issue608_posonly_epoch_villain_seed42",
        ]

    def test_entity_prefixes_project_path(self):
        card = {
            "wandb_entity": "superkaiba",
            "wandb_project": "huggingface",
            "wandb_run_names": ["issue608_a"],
        }
        with patch.object(
            verify_uploads, "check_wandb_runs_by_name", return_value={"status": "OK", "url": "u"}
        ) as mock_names:
            verify_uploads.check_wandb_from_card(card)
        assert mock_names.call_args[0][0] == "superkaiba/huggingface"

    def test_names_without_project_falls_back_to_default_project_scan(self):
        """#601: HF Trainer runs default to project ``huggingface`` when
        WANDB_PROJECT is unset, so declared names without wandb_project
        resolve via the default-entity scan instead of hard-MISSING."""
        with patch.object(
            verify_uploads,
            "check_wandb_runs_default_project",
            return_value={"status": "OK", "url": "u", "detail": "resolved"},
        ) as mock_default:
            res = verify_uploads.check_wandb_from_card({"wandb_run_names": ["a", "b"]})
        assert res["status"] == "OK"
        assert res["source"] == "epm:results reproducibility_card"
        (names,) = mock_default.call_args[0]
        assert names == ["a", "b"]
        assert mock_default.call_args[1] == {"entity": None}

    def test_declared_project_skips_default_scan(self):
        """The existing declared-project path is unchanged: with
        wandb_project present the default-entity scan never fires."""
        card = {"wandb_project": "explicit-project", "wandb_run_names": ["a"]}
        with (
            patch.object(
                verify_uploads,
                "check_wandb_runs_by_name",
                return_value={"status": "OK", "url": "u"},
            ) as mock_names,
            patch.object(
                verify_uploads,
                "check_wandb_runs_default_project",
                side_effect=AssertionError("default-project scan must not fire"),
            ),
        ):
            res = verify_uploads.check_wandb_from_card(card)
        assert res["status"] == "OK"
        assert mock_names.call_args[0][0] == "explicit-project"


# ── check_wandb_runs_default_project ──────────────────────────────────────────


class _FakeWandbRun:
    def __init__(self, name):
        self.name = name


class _FakeWandbProject:
    def __init__(self, name):
        self.name = name


class _FakeWandbApi:
    """Minimal wandb.Api stand-in: default entity + per-project run names.

    ``runs`` honours the server-side displayName $in filter the helper
    sends; probing a project absent from ``runs_by_project`` raises (the
    real API errors on a nonexistent project path).
    """

    def __init__(self, project_names, runs_by_project, default_entity="thomasjiralerspong"):
        self.default_entity = default_entity
        self._project_names = project_names
        self._runs_by_project = runs_by_project

    def projects(self, entity):
        return [_FakeWandbProject(n) for n in self._project_names]

    def runs(self, path, filters=None):
        project = path.split("/", 1)[1]
        if project not in self._runs_by_project:
            raise ValueError(f"project {project} not found")
        wanted = set(filters["displayName"]["$in"]) if filters else None
        return [
            _FakeWandbRun(n)
            for n in self._runs_by_project[project]
            if wanted is None or n in wanted
        ]


def _patched_wandb(api):
    return patch.dict(sys.modules, {"wandb": SimpleNamespace(Api=lambda: api)})


class TestCheckWandbRunsDefaultProject:
    def test_resolves_in_hf_trainer_default_project(self):
        """The #601 shape: runs live in the default entity's ``huggingface``
        project; the row resolves OK and names the project in the detail."""
        api = _FakeWandbApi(
            project_names=["explore-persona-space"],
            runs_by_project={"huggingface": ["4xiqs7ra-run", "6ubkhizm-run"]},
        )
        with _patched_wandb(api):
            res = verify_uploads.check_wandb_runs_default_project(["4xiqs7ra-run", "6ubkhizm-run"])
        assert res["status"] == "OK"
        assert "thomasjiralerspong/huggingface" in res["detail"]

    def test_resolves_in_a_later_entity_project(self):
        """When ``huggingface`` does not exist for the entity, the scan
        continues into the entity's real projects."""
        api = _FakeWandbApi(
            project_names=["explore-persona-space"],
            runs_by_project={"explore-persona-space": ["run-a"]},
        )
        with _patched_wandb(api):
            res = verify_uploads.check_wandb_runs_default_project(["run-a"])
        assert res["status"] == "OK"
        assert "thomasjiralerspong/explore-persona-space" in res["detail"]

    def test_no_project_resolves_all_reports_best_partial(self):
        """All-names-in-ONE-project is required for OK; a partial match is
        reported in the MISSING detail to aid the manual override."""
        api = _FakeWandbApi(
            project_names=["explore-persona-space"],
            runs_by_project={
                "huggingface": ["run-a"],
                "explore-persona-space": [],
            },
        )
        with _patched_wandb(api):
            res = verify_uploads.check_wandb_runs_default_project(["run-a", "run-b"])
        assert res["status"] == "MISSING"
        assert "best partial: 1/2" in res["detail"]
        assert "huggingface" in res["detail"]

    def test_explicit_entity_overrides_default(self):
        api = _FakeWandbApi(
            project_names=[],
            runs_by_project={"huggingface": ["run-a"]},
            default_entity="someone-else",
        )
        with _patched_wandb(api):
            res = verify_uploads.check_wandb_runs_default_project(["run-a"], entity="superkaiba")
        assert res["status"] == "OK"
        assert "superkaiba/huggingface" in res["detail"]


# ── check_wandb_runs_convention_project ───────────────────────────────────────


class TestCheckWandbRunsConventionProject:
    def test_conventional_runs_resolve_as_declaration_gap(self):
        """The #608 follow-up shape: runs named issue<N>_* live in the
        conventional <default_entity>/issue<N> project; the probe returns
        an OK row that names the declaration gap."""
        api = _FakeWandbApi(
            project_names=[],
            runs_by_project={"issue608": ["issue608_posonly_epoch_villain_seed42", "issue608_x"]},
        )
        with _patched_wandb(api):
            res = verify_uploads.check_wandb_runs_convention_project(608)
        assert res["status"] == "OK"
        assert res["url"] == "https://wandb.ai/thomasjiralerspong/issue608"
        assert "declaration gap" in res["detail"]
        assert "2 run(s)" in res["detail"]

    def test_nonconventional_names_do_not_count(self):
        api = _FakeWandbApi(project_names=[], runs_by_project={"issue608": ["other-run"]})
        with _patched_wandb(api):
            assert verify_uploads.check_wandb_runs_convention_project(608) is None

    def test_missing_project_fails_soft_to_none(self):
        """No issue<N> project on the entity (the fake raises, like the real
        API on a nonexistent project path) -> None, today's MISSING row."""
        api = _FakeWandbApi(project_names=[], runs_by_project={})
        with _patched_wandb(api):
            assert verify_uploads.check_wandb_runs_convention_project(608) is None

    def test_api_error_fails_soft_to_none(self):
        """FAIL-SOFT requirement: a WandB API error keeps today's behavior
        (None -> strict MISSING), never an ERROR row or a raise."""

        def _boom():
            raise RuntimeError("wandb unreachable")

        with patch.dict(sys.modules, {"wandb": SimpleNamespace(Api=_boom)}):
            assert verify_uploads.check_wandb_runs_convention_project(608) is None

    def test_no_default_entity_returns_none(self):
        api = _FakeWandbApi(
            project_names=[], runs_by_project={"issue608": ["issue608_a"]}, default_entity=None
        )
        with _patched_wandb(api):
            assert verify_uploads.check_wandb_runs_convention_project(608) is None


# ── run_verification integration ──────────────────────────────────────────────


class TestRunVerificationCardFallback:
    def test_sweep_card_satisfies_hf_model_row(self):
        """The #608 false-FAIL repro: undeclared --hf-model on a sweep task
        resolves via the card's adapter_paths instead of hard-MISSING."""
        with (
            patch.object(verify_uploads, "_load_results_card", return_value=_CARD_608),
            patch.object(
                verify_uploads,
                "check_hf_hub_path",
                return_value={"status": "OK", "url": "u", "file_count": 3},
            ),
            # Convention probe patched to None: this test pins the strict
            # MISSING row when nothing resolves (and stays network-free).
            patch.object(verify_uploads, "check_wandb_runs_convention_project", return_value=None),
        ):
            report = verify_uploads.run_verification(608, experiment_type="training")
        assert report["checks"]["hf_model"]["status"] == "OK"
        assert report["checks"]["hf_model"]["source"] == "epm:results reproducibility_card"
        # The card declares no wandb fields, so that row still hard-MISSes.
        assert report["checks"]["wandb_run"]["status"] == "MISSING"

    def test_undeclared_wandb_resolves_via_convention_probe(self):
        """The #608 follow-up false-FAIL repro: a card with adapter_paths
        but NO wandb_* fields resolves the wandb_run row via the
        conventional <default_entity>/issue<N> project probe (runs named
        issue<N>_*) instead of hard-MISSING."""
        api = _FakeWandbApi(
            project_names=[],
            runs_by_project={"issue608": ["issue608_posonly_epoch_villain_seed42"]},
        )
        with (
            patch.object(verify_uploads, "_load_results_card", return_value=_CARD_608),
            patch.object(
                verify_uploads,
                "check_hf_hub_path",
                return_value={"status": "OK", "url": "u", "file_count": 3},
            ),
            _patched_wandb(api),
        ):
            report = verify_uploads.run_verification(608, experiment_type="training")
        row = report["checks"]["wandb_run"]
        assert row["status"] == "OK"
        assert "declaration gap" in row["detail"]
        assert row["source"] == "wandb project-naming convention (no card declaration)"

    def test_explicit_declarations_skip_card_lookup(self):
        """Single-path declaration behavior is unchanged: with both paths
        declared, the card is never even loaded."""
        with (
            patch.object(
                verify_uploads,
                "_load_results_card",
                side_effect=AssertionError("card must not be loaded"),
            ),
            patch.object(
                verify_uploads,
                "check_hf_hub_path",
                return_value={"status": "OK", "url": "u", "file_count": 1},
            ) as mock_hf,
            patch.object(
                verify_uploads, "check_wandb_run", return_value={"status": "OK", "url": "u"}
            ) as mock_wandb,
        ):
            report = verify_uploads.run_verification(
                608,
                experiment_type="training",
                wandb_run="e/p/runs/abc",
                hf_model_path="adapters/issue_608/x",
            )
        mock_hf.assert_called_once_with(
            verify_uploads.HF_MODEL_REPO, "adapters/issue_608/x", "model"
        )
        mock_wandb.assert_called_once_with("e/p/runs/abc")
        assert report["checks"]["hf_model"]["status"] == "OK"
        assert report["checks"]["wandb_run"]["status"] == "OK"

    def test_no_card_keeps_strict_missing(self):
        """No card AND no conventional runs -> the strict MISSING rows and
        the FAIL verdict are unchanged (probe patched to None)."""
        with (
            patch.object(verify_uploads, "_load_results_card", return_value=None),
            patch.object(verify_uploads, "check_wandb_runs_convention_project", return_value=None),
        ):
            report = verify_uploads.run_verification(608, experiment_type="training")
        assert report["checks"]["wandb_run"]["status"] == "MISSING"
        assert "conventional-project probe" in report["checks"]["wandb_run"]["detail"]
        assert report["checks"]["hf_model"]["status"] == "MISSING"
        assert report["verdict"] == "FAIL"

    def test_non_training_types_never_load_card(self):
        with patch.object(
            verify_uploads,
            "_load_results_card",
            side_effect=AssertionError("card must not be loaded"),
        ):
            report = verify_uploads.run_verification(608, experiment_type="eval-only")
        assert "wandb_run" not in report["checks"]
        assert "hf_model" not in report["checks"]


# ── prose-template card declarations (#612) ───────────────────────────────────

# The real #612 shape: the orchestrator hand-composed the card with a prose
# summary instead of the per-cell mapping the sentinel contract requires.
_PROSE_ADAPTERS_612 = "adapters/issue_612/<arm>/<source>_seed<S> (16 adapters)"

# The bare-``wandb``-key sibling of the same card: a free-text line that
# NO check consults (check_wandb_from_card reads wandb_run_path / wandb_run /
# wandb_run_names only).
_PROSE_WANDB_612 = "wandb: 52 runs under project issue612 (one per arm x judge)"


class TestProseCardDeclarations:
    def test_string_adapter_paths_names_contract_violation(self):
        """The #612 repro: a prose-template adapter_paths string used to be
        silently ignored (only dict/list handled), falling through to the
        generic 'no card declares adapter_paths' MISSING row on a
        fully-uploaded sweep. The row must name the producer-contract
        violation instead (diagnostic-only: status stays MISSING)."""
        card = {
            "hf_model_repo": "superkaiba1/explore-persona-space",
            "adapter_paths": _PROSE_ADAPTERS_612,
        }
        with patch.object(
            verify_uploads,
            "check_hf_hub_path",
            side_effect=AssertionError("nothing checkable must be probed"),
        ):
            res = verify_uploads.check_hf_model_from_card(card)
        assert res is not None
        assert res["status"] == "MISSING"
        assert "producer-contract violation" in res["detail"]
        assert "SKILL.md Step 7" in res["detail"]
        assert "template placeholders" in res["detail"]
        assert "#612" in res["detail"]
        assert res["source"] == "epm:results reproducibility_card"

    def test_prose_without_placeholders_still_diagnosed(self):
        res = verify_uploads.check_hf_model_from_card(
            {"adapter_paths": "adapters/issue_612/all (16 adapters)"}
        )
        assert res["status"] == "MISSING"
        assert "producer-contract violation" in res["detail"]
        assert "placeholders" not in res["detail"]

    def test_real_single_path_plus_prose_adapters_appends_note(self):
        """A resolvable hf_model_path still wins the row; the unverifiable
        prose adapter_paths stays visible as an ALSO note."""
        card = {"hf_model_path": "adapters/issue_612/x", "adapter_paths": _PROSE_ADAPTERS_612}
        with patch.object(
            verify_uploads,
            "check_hf_hub_path",
            return_value={"status": "OK", "url": "u", "file_count": 2},
        ):
            res = verify_uploads.check_hf_model_from_card(card)
        assert res["status"] == "OK"
        assert "ALSO:" in res["detail"]
        assert "producer-contract violation" in res["detail"]

    def test_empty_string_adapter_paths_still_returns_none(self):
        """An empty string is not a declaration (#601 ``_is_declared``) —
        the caller's generic fall-through is unchanged."""
        assert verify_uploads.check_hf_model_from_card({"adapter_paths": ""}) is None

    def test_string_wandb_run_names_not_iterated_per_character(self):
        """A string wandb_run_names used to iterate into per-CHARACTER
        'run names' (a garbage MISSING row); diagnose the prose
        declaration instead and never hit the API."""
        card = {"wandb_project": "issue612", "wandb_run_names": "issue612_<arm> runs"}
        with (
            patch.object(
                verify_uploads,
                "check_wandb_runs_by_name",
                side_effect=AssertionError("API must not be called"),
            ),
            patch.object(
                verify_uploads,
                "check_wandb_runs_default_project",
                side_effect=AssertionError("default-project scan must not fire"),
            ),
        ):
            res = verify_uploads.check_wandb_from_card(card)
        assert res["status"] == "MISSING"
        assert "wandb_run_names" in res["detail"]
        assert "producer-contract violation" in res["detail"]

    def test_prose_wandb_run_path_diagnosed_before_api(self):
        """A free-text wandb line (#612) is never a real run path; naming
        the violation beats the opaque API error string it used to get."""
        with patch.object(
            verify_uploads,
            "check_wandb_run",
            side_effect=AssertionError("API must not be called"),
        ):
            res = verify_uploads.check_wandb_from_card(
                {"wandb_run_path": "wandb: 52 runs under project issue612"}
            )
        assert res["status"] == "MISSING"
        assert "producer-contract violation" in res["detail"]

    def test_valid_run_path_still_delegates(self):
        """The prose guard must not catch a real run path (no whitespace,
        no placeholders)."""
        with patch.object(
            verify_uploads, "check_wandb_run", return_value={"status": "OK", "url": "u"}
        ) as mock_run:
            res = verify_uploads.check_wandb_from_card({"wandb_run": "e/p/abc"})
        assert res["status"] == "OK"
        mock_run.assert_called_once_with("e/p/abc")

    def test_card_with_no_wandb_fields_still_returns_none(self):
        """The prose guard must not preempt the convention-probe fallback:
        a card carrying ONLY a prose adapter_paths declares no wandb
        fields, so check_wandb_from_card stays None and the caller can
        still rescue the row via the conventional-project probe."""
        assert verify_uploads.check_wandb_from_card({"adapter_paths": _PROSE_ADAPTERS_612}) is None

    def test_run_verification_prose_card_row_names_violation(self):
        """Integration: the #612 shape no longer reads as the generic
        'No HF model path provided' row; verdict stays FAIL
        (diagnostic-only — the operator just learns WHY in one read)."""
        card = {"adapter_paths": _PROSE_ADAPTERS_612}
        with (
            patch.object(verify_uploads, "_load_results_card", return_value=card),
            patch.object(verify_uploads, "check_wandb_runs_convention_project", return_value=None),
        ):
            report = verify_uploads.run_verification(612, experiment_type="training")
        row = report["checks"]["hf_model"]
        assert row["status"] == "MISSING"
        assert "producer-contract violation" in row["detail"]
        assert "No HF model path provided" not in row["detail"]
        assert report["verdict"] == "FAIL"

    def test_bare_prose_wandb_noted_on_generic_missing_row(self):
        """A free-text line under the bare ``wandb`` key is consulted by NO
        check (#612 follow-up); the generic MISSING row must say so instead
        of reading as a plain declaration gap. Diagnostic-only: the generic
        detail text, the MISSING status, and the FAIL verdict are
        unchanged — the note is appended."""
        card = {"wandb": _PROSE_WANDB_612}
        with (
            patch.object(verify_uploads, "_load_results_card", return_value=card),
            patch.object(verify_uploads, "check_wandb_runs_convention_project", return_value=None),
        ):
            report = verify_uploads.run_verification(612, experiment_type="training")
        row = report["checks"]["wandb_run"]
        assert row["status"] == "MISSING"
        assert "conventional-project probe" in row["detail"]  # generic text preserved
        assert "ALSO:" in row["detail"]
        assert "unconsulted prose 'wandb' field" in row["detail"]
        assert "SKILL.md Step 7" in row["detail"]
        assert report["verdict"] == "FAIL"

    def test_bare_prose_wandb_noted_on_convention_probe_ok_row(self):
        """The bare-``wandb``-key note must never preempt the convention
        probe: the probe still fires, its OK row (and status) wins the
        slot unchanged, and the unconsulted prose declaration is appended
        to its detail."""
        card = {"adapter_paths": {"a": "adapters/issue_612/a"}, "wandb": _PROSE_WANDB_612}
        probe_row = {
            "status": "OK",
            "url": "https://wandb.ai/e/issue612",
            "detail": "52 run(s) named issue612_* resolve (declaration gap)",
            "source": "wandb project-naming convention (no card declaration)",
        }
        with (
            patch.object(verify_uploads, "_load_results_card", return_value=card),
            patch.object(
                verify_uploads,
                "check_hf_hub_path",
                return_value={"status": "OK", "url": "u", "file_count": 1},
            ),
            patch.object(
                verify_uploads,
                "check_wandb_runs_convention_project",
                return_value=dict(probe_row),
            ) as mock_probe,
        ):
            report = verify_uploads.run_verification(612, experiment_type="training")
        mock_probe.assert_called_once_with(612)
        row = report["checks"]["wandb_run"]
        assert row["status"] == "OK"
        assert "declaration gap" in row["detail"]
        assert "ALSO:" in row["detail"]
        assert "unconsulted prose 'wandb' field" in row["detail"]

    def test_structured_wandb_declaration_suppresses_unconsulted_note(self):
        """When the card ALSO declares structured wandb fields, those win
        the row exactly as before and the bare prose ``wandb`` duplicate
        is not noted (the producer's structured declaration was consulted)."""
        card = {
            "wandb": _PROSE_WANDB_612,
            "wandb_project": "issue612",
            "wandb_run_names": ["issue612_a"],
        }
        with (
            patch.object(verify_uploads, "_load_results_card", return_value=card),
            patch.object(
                verify_uploads,
                "check_wandb_runs_by_name",
                return_value={"status": "OK", "url": "u", "detail": "all resolve"},
            ),
        ):
            report = verify_uploads.run_verification(612, experiment_type="training")
        row = report["checks"]["wandb_run"]
        assert row["status"] == "OK"
        assert "unconsulted" not in row["detail"]

    def test_unconsulted_note_only_for_nonempty_strings(self):
        """The note fires only on a declared STRING ``wandb`` value; None /
        empty / non-string shapes keep the row detail untouched, and long
        values are snipped."""
        note_fn = verify_uploads._unconsulted_prose_wandb_note
        assert note_fn(None) is None
        assert note_fn({}) is None
        assert note_fn({"wandb": ""}) is None
        assert note_fn({"wandb": {"project": "issue612"}}) is None
        long_note = note_fn({"wandb": "x" * 200})
        assert long_note is not None
        assert "..." in long_note
        assert "#612" in long_note
