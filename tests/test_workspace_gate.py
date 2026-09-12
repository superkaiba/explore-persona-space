"""Main execution cannot bypass evidence identity and full-calibration coverage."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from explore_persona_space.analysis.workspace_calibration import (
    calibration_means,
    matrix_agreement,
    paired_direction_agreement,
)
from explore_persona_space.analysis.workspace_gate import _evidence, validate_main_readiness
from explore_persona_space.analysis.workspace_runtime import content_sha256, file_sha256


class ReadinessBundle:
    """Tiny synthetic tensors, real file hashes, and a hermetic source-history check.

    No validator is mocked. Approval below belongs only to this test fixture;
    no real experiment, upload, or scientific review artifact is read or written.
    """

    def __init__(self, root, config, config_path, selection_path, identity):
        self.root = root
        self.root.mkdir()
        self.config = config
        self.config_path = config_path
        self.selection_path = selection_path
        self.identity = identity
        self.path = root / "readiness.json"
        self.readiness = {
            "schema": "workspace-jr-main-readiness-v1",
            **{key: identity[key] for key in ("model_role", "config_sha256", "selection_sha256")},
            "evidence": {},
        }

    def write_file(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".pt":
            torch.save(value, path)
        else:
            # Deliberately permit malformed NaN JSON in rejection regressions.
            path.write_text(json.dumps(value, indent=2) + "\n")
        return {"file": str(relative), "sha256": file_sha256(path)}

    def put(self, name, value, *, approve=False):
        reference = self.readiness["evidence"].get(name)
        relative = reference["file"] if reference else f"{name}.json"
        self.readiness["evidence"][name] = self.write_file(relative, value)
        self.save()
        if approve:
            self.approve()

    def add_tensor(self, name, value):
        self.readiness["evidence"][name] = self.write_file(f"{name}.pt", value)

    def get(self, name):
        return self.read_reference(self.readiness["evidence"][name])

    def read_reference(self, reference):
        return json.loads((self.root / reference["file"]).read_text())

    def sha(self, name):
        return self.readiness["evidence"][name]["sha256"]

    def save(self):
        self.path.write_text(json.dumps(self.readiness, indent=2) + "\n")

    def approve(self):
        self.put(
            "review",
            {
                "decision": "approved_for_exploratory_main",
                "reviewer": "independent synthetic unit fixture",
                "reasoning": (
                    "This synthetic fixture exercises evidence binding; "
                    "it approves no real model experiment."
                ),
                "calibration_report_sha256": self.sha("calibration"),
                "parity_report_sha256": self.sha("parity"),
                "pilot_results_sha256": self.sha("pilot_results"),
                "pilot_producer_sha": self.identity["code"]["git_commit"],
                "evidence_digest": content_sha256(
                    {
                        **{
                            key: self.readiness[key]
                            for key in ("model_role", "config_sha256", "selection_sha256")
                        },
                        "evidence": {
                            key: value
                            for key, value in self.readiness["evidence"].items()
                            if key != "review"
                        },
                    }
                ),
            },
        )

    def refresh_pilot_links(self):
        """Rebind independent links so a semantic mutation reaches its own check."""
        binding = self.get("pilot_binding")
        locations = {
            "pilot_results": "fits/pilot/k10-rotationNone/results.json",
            "pilot_input_manifest": "fits/pilot/k10-rotationNone/input_manifest.json",
            "pilot_exit": "pilot_exit.json",
        }
        for key in locations:
            binding[f"{key}_sha256"] = self.sha(key)
        self.put("pilot_binding", binding)
        upload = self.get("pilot_upload")
        for key, location in locations.items():
            upload["verified_sha256"][location] = self.sha(key)
        self.put("pilot_upload", upload)
        self.approve()

    def validate(self):
        return validate_main_readiness(
            self.path,
            self.config,
            config_path=self.config_path,
            selection_path=self.selection_path,
            identity=self.identity,
        )


@pytest.fixture
def readiness_bundle(tmp_path, monkeypatch):
    """Build full 119-prompt coverage with 2D tensors in an isolated Git repo."""
    repository = Path(__file__).resolve().parents[1]
    source_root = tmp_path / "source"
    source_root.mkdir()
    sources = (
        "scripts/workspace_jr_calibration.py",
        "scripts/workspace_jr_parity.py",
        "src/explore_persona_space/analysis/workspace_calibration.py",
        "src/explore_persona_space/analysis/workspace_runtime.py",
    )
    for source in sources:
        path = source_root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repository / source).read_bytes())
    # Isolate test commits from user configuration, signing, and global hooks.
    git_env = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_AUTHOR_NAME": "Readiness Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Readiness Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=source_root, env=git_env, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("add", "--", *sources)
    git("commit", "--quiet", "-m", "Synthetic readiness source fixture")
    source_sha = git("rev-parse", "HEAD")
    monkeypatch.chdir(source_root)

    def make(role="primary"):
        config = {
            "selection": {role: f"fixture/{role}"},
            "models": {role: {"revision": "1" * 40, "d_model": 2}},
            "lenses": {"calibration_dtype": "bfloat16"},
            "generation": {"seeds": [42, 43, 44, 45, 46]},
            "sampling": {"main_maximum": {"train": 8, "validation": 8, "test": 8}},
            "provenance_gate": {
                "recapture_contexts": 32,
                "maximum_relative_frobenius_error": 0.01,
                "minimum_row_cosine": 0.999,
            },
        }

        def prompts(label, count):
            return [
                {"prompt_sha256": content_sha256([label, index]), "ladder_local_id": index}
                for index in range(count)
            ]

        counts = {"train": 64, "validation": 16, "test": 32}
        selection = {
            "subsets": {
                "calibration": prompts("calibration", 128),
                **{f"pilot_{split}": prompts(f"pilot_{split}", n) for split, n in counts.items()},
                **{f"main_{split}": prompts(f"main_{split}", 8) for split in counts},
            }
        }
        config_path, selection_path = (
            tmp_path / f"config-{role}.json",
            tmp_path / f"selection-{role}.json",
        )
        config_path.write_text(json.dumps(config))
        selection_path.write_text(json.dumps(selection))
        identity = {
            "config_sha256": file_sha256(config_path),
            "selection_sha256": file_sha256(selection_path),
            "model_role": role,
            "versions": {
                "torch": "fixture",
                "transformers": "fixture",
                "huggingface-hub": "fixture",
            },
            "code": {"git_commit": source_sha, "git_dirty": False},
        }
        bundle = ReadinessBundle(
            tmp_path / f"bundle-{role}", config, config_path, selection_path, identity
        )
        tokens = {
            "identity": identity,
            "model_id": config["selection"][role],
            "model_revision": "1" * 40,
            "subset": "calibration",
            "rows": [
                {**row, "token_ids": [1, 2, 3, 4, 5, 6]}
                for row in selection["subsets"]["calibration"][:119]
            ],
            "excluded": [
                {**row, "reason": "too_short_after_tokenization"}
                for row in selection["subsets"]["calibration"][119:]
            ],
        }
        bundle.put("tokens", tokens)
        bundle.put(
            "native_validation",
            {
                "identity": identity,
                "model_dtype": "bfloat16",
                "forward_bit_identical": True,
                "hook_bit_identical": True,
                "ordinary_numerical_validation": {"status": "passed"},
                "token_manifest_sha256": content_sha256(tokens),
            },
        )
        contract = {
            **identity,
            "tokens_sha256": content_sha256(tokens),
            "native_validation_sha256": bundle.sha("native_validation"),
            "dim_batch": 8,
            "actual_dtype": "torch.bfloat16",
            "attention_implementation": "eager",
            "model_class": "Qwen3_5ForConditionalGeneration",
        }
        pairs = []
        for index, row in enumerate(tokens["rows"]):
            reference = bundle.write_file(
                f"pairs/prompt-{index:04d}.pt",
                {
                    "contract": contract,
                    "contract_sha256": content_sha256(contract),
                    "prompt_sha256": row["prompt_sha256"],
                    "token_ids": row["token_ids"],
                    "J": torch.eye(2) * (1 + index / 128),
                    "R": torch.eye(2) * (2 + index / 128),
                },
            )
            pairs.append(bundle.root / reference["file"])
        means, calibration = calibration_means(pairs, tokens, 2)
        bundle.add_tensor("means", means)
        full = calibration["full_group"]
        comparisons = [(group, full) for group in means if group != full] + [("even", "odd")]
        direction_comparisons = {
            f"{left}_vs_{right}": {
                arm: paired_direction_agreement(means[left][arm], means[right][arm])
                for arm in ("J", "R")
            }
            for left, right in comparisons
        }
        direction_comparisons["J_vs_R"] = paired_direction_agreement(
            means[full]["J"], means[full]["R"]
        )
        bundle.add_tensor("directions", {"token_ids": [2, 3], "comparisons": direction_comparisons})
        calibration.update(
            identity=identity,
            status="calibration_diagnostics_complete",
            native_producer_proof={"code_match": "exact"},
            main_approval=False,
            interpretation="Synthetic calibration evidence for readiness validator tests only.",
            native_validation_sha256=bundle.sha("native_validation"),
            means_sha256=bundle.sha("means"),
            directions_sha256=bundle.sha("directions"),
            matrix_agreement={
                f"{left}_vs_{right}": {
                    arm: matrix_agreement(means[left][arm], means[right][arm]) for arm in ("J", "R")
                }
                for left, right in comparisons
            },
            j_r_matrix_agreement=matrix_agreement(means[full]["J"], means[full]["R"]),
            eligible_token_count=2,
            readouts=[
                {
                    "prompt_sha256": row["prompt_sha256"],
                    "position": 4,
                    "observed_next_token_id": 6,
                    "top_token_ids": {arm: [6] for arm in ("native", "J", "R")},
                }
                for row in tokens["rows"][:8]
            ],
        )
        bundle.put("calibration", calibration)

        intervals = [(2, 32), (32, 61), (61, 90), (90, 119)] if role == "primary" else [(0, 119)]
        workers = []
        for rank, (start, stop) in enumerate(intervals):
            indices = sorted(set(range(start, stop)) | ({0, 1} if role == "primary" else set()))
            terminal = bundle.write_file(
                f"workers/{rank}/rank{rank}_exit.json",
                {
                    "rank": rank,
                    "start": start,
                    "stop": stop,
                    "exit_code": 0,
                    "finished_at_epoch": 1789240000,
                },
            )
            snapshot = bundle.write_file(
                f"workers/{rank}/snapshot.json",
                {
                    "rank": rank,
                    "rank_interval": [start, stop],
                    "successful_complete": True,
                    "terminal_receipt_included": True,
                    "paired_prompt_files": len(indices),
                    "included_prompt_indices": indices,
                    "native_source_sha": source_sha,
                    "snapshot_contract": "hardlinks_of_atomic_immutable_prompt_checkpoints",
                },
            )
            hashes = {
                "snapshot.json": snapshot["sha256"],
                f"rank{rank}_exit.json": terminal["sha256"],
                **{
                    f"lens_shards/{calibration['source_files'][i]['file']}": calibration[
                        "source_files"
                    ][i]["sha256"]
                    for i in indices
                },
            }
            upload = bundle.write_file(
                f"workers/{rank}/upload.json",
                _upload_receipt(hashes, f"{role}_full_calibration/rank{rank}"),
            )
            workers.append(
                {
                    "rank": rank,
                    "start": start,
                    "stop": stop,
                    "upload": upload,
                    "snapshot": snapshot,
                    "terminal": terminal,
                }
            )
        bundle.put(
            "calibration_upload",
            {"schema": "workspace-jr-calibration-upload-aggregate-v1", "workers": workers},
        )

        arrays = {
            "x_prompt_last": torch.arange(1, 65, dtype=torch.float32).reshape(32, 2),
            "y_ans": torch.arange(2, 66, dtype=torch.float32).reshape(32, 2),
        }
        bundle.add_tensor("parity_arrays", arrays)
        bundle.add_tensor(
            "parity_historical_targets", {key: value.clone() for key, value in arrays.items()}
        )
        bundle.put(
            "parity_inputs",
            {
                "model_role": role,
                "config_sha256": identity["config_sha256"],
                "selection_sha256": identity["selection_sha256"],
                "historical_targets_sha256": bundle.sha("parity_historical_targets"),
                "rows": [{"selection": row} for row in tokens["rows"][:32]],
            },
        )
        bundle.put(
            "parity",
            {
                "identity": identity,
                "status": "passed",
                "input_manifest_sha256": bundle.sha("parity_inputs"),
                "recaptured_sha256": bundle.sha("parity_arrays"),
                "rows": [{"prompt_sha256": row["prompt_sha256"]} for row in tokens["rows"][:32]],
                "metrics": {
                    key: {"passed": True, "relative_frobenius_error": 0.0, "row_cosine": [1.0] * 32}
                    for key in arrays
                },
            },
        )
        dictionary_manifest = {
            "identity": identity,
            "pilot_only": True,
            "calibration_prompts": 2,
            "full_calibration_membership": False,
            "source_hashes": {
                row["file"]: row["sha256"] for row in calibration["source_files"][:2]
            },
            "arms": {
                arm: {"sha256": content_sha256([arm, "fixture"]), "shape": [2, 2]}
                for arm in ("J", "R")
            },
        }
        pilot_contract = {
            "identity": identity,
            "k": 10,
            "rotation": None,
            "dictionaries": dictionary_manifest,
        }
        coverage = {}
        for split, count in counts.items():
            ids = [row["prompt_sha256"] for row in selection["subsets"][f"pilot_{split}"]]
            coverage[split] = {
                "identity": identity,
                "contract": pilot_contract,
                "planned_contexts": count,
                "status": "complete",
                "included_prompt_sha256": ids,
                "exclusions": [],
                "file_sha256": {key: content_sha256(["synthetic_component", key]) for key in ids},
            }
        bundle.put(
            "pilot_input_manifest",
            {"identity": identity, "contract": pilot_contract, "coverage": coverage},
        )
        bundle.put(
            "pilot_results",
            {
                "schema": "workspace-jr-component-fit-v1",
                "split_counts": counts,
                "metrics": {
                    predictor: {
                        target: {"r2": 1.0, "sse": 0.0, "sst": 1.0, "status": "ok"}
                        for target in ("full", "J", "restJ", "R", "restR")
                    }
                    for predictor in ("ridge", "mlp")
                },
                "reconstruction": {
                    arm: {"max_abs_reconstruction_error": 0.0} for arm in ("J", "R")
                },
            },
        )
        bundle.put(
            "pilot_exit", {"exit_code": 0, "phase": "complete", "finished_at_epoch": 1789240000}
        )
        bundle.put(
            "pilot_binding",
            {
                "schema": "workspace-jr-pilot-binding-v1",
                "status": "reviewed_legacy_provenance_bridge",
                "producer_sha": source_sha,
                "cell": "k10-rotationNone",
                "generation_seeds": config["generation"]["seeds"],
                "dictionary_manifest_sha256": content_sha256(dictionary_manifest),
                **{
                    f"{key}_sha256": bundle.sha(key)
                    for key in ("pilot_results", "pilot_input_manifest", "pilot_exit")
                },
            },
        )
        bundle.put(
            "pilot_upload",
            _upload_receipt(
                {
                    "fits/pilot/k10-rotationNone/results.json": bundle.sha("pilot_results"),
                    "fits/pilot/k10-rotationNone/input_manifest.json": bundle.sha(
                        "pilot_input_manifest"
                    ),
                    "pilot_exit.json": bundle.sha("pilot_exit"),
                },
                f"{role}_pilot",
            ),
        )
        bundle.put(
            "execution_plan",
            {
                "main_outcomes_seen_before_freeze": False,
                "scale_reason": "Synthetic pre-main compute plan; no actual workload is launched.",
                "main_counts": {"train": 6, "validation": 4, "test": 6},
            },
        )
        bundle.approve()
        return bundle

    return make


def _upload_receipt(hashes, prefix):
    return {
        "repo": "superkaiba1/explore-persona-space-data",
        "revision": "d" * 40,
        "prefix": f"exploratory_workspace_jr/unit_fixture/{prefix}",
        "verified_sha256": hashes,
        "files_verified": len(hashes),
    }


def test_evidence_hashes_and_bundle_boundaries_are_enforced(tmp_path):
    path = tmp_path / "review.json"
    path.write_text('{"decision":"not_reviewed"}')
    reference = {"file": path.name, "sha256": file_sha256(path)}
    assert _evidence(tmp_path, reference) == path
    path.write_text('{"decision":"approved_for_exploratory_main"}')
    with pytest.raises(ValueError, match="evidence bytes differ"):
        _evidence(tmp_path, reference)
    for filename in ("../review.json", str(path), ".hidden.json"):
        with pytest.raises(ValueError, match="within its bundle"):
            _evidence(tmp_path, {**reference, "file": filename})


def test_main_readiness_requires_all_evidence_and_matching_role(tmp_path):
    identity = {"model_role": "primary", "config_sha256": "config", "selection_sha256": "selection"}
    value = {"schema": "workspace-jr-main-readiness-v1", **identity, "evidence": {}}
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="evidence is incomplete"):
        validate_main_readiness(path, {}, config_path=path, selection_path=path, identity=identity)
    value["model_role"] = "comparison"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="requested model_role"):
        validate_main_readiness(path, {}, config_path=path, selection_path=path, identity=identity)


@pytest.mark.parametrize("role", ["primary", "comparison"])
def test_complete_readiness_bundle_passes_without_mocking_any_validator(readiness_bundle, role):
    bundle = readiness_bundle(role)
    result = bundle.validate()
    assert result["readiness_sha256"] == file_sha256(bundle.path)
    assert result["execution"]["main_counts"] == {"train": 6, "validation": 4, "test": 6}
    assert result["reports"]["calibration"]["realized_prompts"] == 119
    means = torch.load(result["paths"]["means"], weights_only=True)
    assert means["first_119"]["J"].dtype == torch.float32
    torch.testing.assert_close(means["first_119"]["J"], torch.eye(2) * (1 + 59 / 128))
    workers = result["reports"]["calibration_upload"]["workers"]
    assert len(workers) == (4 if role == "primary" else 1)


@pytest.mark.parametrize("name", ["execution_plan", "pilot_binding", "calibration_upload"])
def test_old_independent_review_cannot_approve_rehashed_evidence(readiness_bundle, name):
    bundle = readiness_bundle()
    value = bundle.get(name)
    if name == "execution_plan":
        value["main_counts"]["train"] = 8
    else:
        value["review_note"] = "Changed after independent review"
    bundle.put(name, value)
    with pytest.raises(ValueError, match="Independent review"):
        bundle.validate()


def test_evidence_rejects_directory_symlink_even_when_bytes_match(tmp_path):
    root, outside = tmp_path / "bundle", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "review.json").write_text("{}")
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        _evidence(
            root, {"file": "redirect/review.json", "sha256": file_sha256(outside / "review.json")}
        )


def test_analyzer_source_change_invalidates_complete_bundle(readiness_bundle):
    bundle = readiness_bundle()
    source = Path("scripts/workspace_jr_parity.py")
    source.write_text(source.read_text() + "\n# synthetic post-producer modification\n")
    with pytest.raises(ValueError, match="analyzer changed"):
        bundle.validate()


def test_missing_worker_cannot_be_replaced_by_success_boolean(readiness_bundle):
    bundle = readiness_bundle()
    aggregate = bundle.get("calibration_upload")
    aggregate["workers"].pop()
    aggregate["all_native_workers_exited_successfully"] = True
    bundle.put("calibration_upload", aggregate, approve=True)
    with pytest.raises(ValueError, match="worker coverage"):
        bundle.validate()


def test_nested_worker_receipt_bytes_are_checked(readiness_bundle):
    bundle = readiness_bundle()
    worker = bundle.get("calibration_upload")["workers"][0]
    terminal = bundle.read_reference(worker["terminal"])
    terminal["finished_at_epoch"] += 1
    bundle.write_file(worker["terminal"]["file"], terminal)
    with pytest.raises(ValueError, match="evidence bytes differ"):
        bundle.validate()


@pytest.mark.parametrize("mutation", ["failed_exit", "wrong_rank", "missing_index"])
def test_consistently_rehashed_worker_failures_still_block_main(readiness_bundle, mutation):
    bundle = readiness_bundle()
    aggregate = bundle.get("calibration_upload")
    worker = aggregate["workers"][0]
    terminal = bundle.read_reference(worker["terminal"])
    snapshot = bundle.read_reference(worker["snapshot"])
    if mutation == "failed_exit":
        terminal["exit_code"] = 1
    elif mutation == "wrong_rank":
        terminal["rank"] = 1
    else:
        snapshot["included_prompt_indices"].pop()
        snapshot["paired_prompt_files"] -= 1
    worker["terminal"] = bundle.write_file(worker["terminal"]["file"], terminal)
    worker["snapshot"] = bundle.write_file(worker["snapshot"]["file"], snapshot)
    upload = bundle.read_reference(worker["upload"])
    upload["verified_sha256"].update(
        {
            "rank0_exit.json": worker["terminal"]["sha256"],
            "snapshot.json": worker["snapshot"]["sha256"],
        }
    )
    worker["upload"] = bundle.write_file(worker["upload"]["file"], upload)
    bundle.put("calibration_upload", aggregate, approve=True)
    with pytest.raises(ValueError, match="successful worker terminal receipts"):
        bundle.validate()


def test_calibration_hash_under_wrong_remote_path_is_not_durable_coverage(readiness_bundle):
    bundle = readiness_bundle()
    aggregate = bundle.get("calibration_upload")
    worker = aggregate["workers"][0]
    upload = bundle.read_reference(worker["upload"])
    hashes = upload["verified_sha256"]
    hashes["wrong_folder/prompt-0002.pt"] = hashes.pop("lens_shards/prompt-0002.pt")
    worker["upload"] = bundle.write_file(worker["upload"]["file"], upload)
    bundle.put("calibration_upload", aggregate, approve=True)
    with pytest.raises(ValueError, match="exact verified upload path"):
        bundle.validate()


@pytest.mark.parametrize(
    "mutation", ["nan_error", "nan_cosine", "short_cosines", "failed_threshold"]
)
def test_parity_requires_complete_finite_passing_metrics(readiness_bundle, mutation):
    bundle = readiness_bundle()
    parity = bundle.get("parity")
    metric = parity["metrics"]["x_prompt_last"]
    if mutation == "nan_error":
        metric["relative_frobenius_error"] = float("nan")
    elif mutation == "nan_cosine":
        metric["row_cosine"][0] = float("nan")
    elif mutation == "short_cosines":
        metric["row_cosine"].pop()
    else:
        metric["relative_frobenius_error"] = 0.02
    bundle.put("parity", parity, approve=True)
    with pytest.raises(ValueError, match="parity thresholds"):
        bundle.validate()


def test_rehashed_historical_arrays_must_still_match_parity_inputs(readiness_bundle):
    bundle = readiness_bundle()
    bundle.add_tensor(
        "parity_historical_targets",
        {"x_prompt_last": torch.ones(32, 2), "y_ans": torch.ones(32, 2)},
    )
    bundle.approve()
    with pytest.raises(ValueError, match="Historical mapping recapture"):
        bundle.validate()


def test_result_swap_cannot_reuse_legacy_pilot_binding(readiness_bundle):
    bundle = readiness_bundle()
    pilot = bundle.get("pilot_results")
    pilot["metrics"]["ridge"]["J"]["r2"] = 0.5
    bundle.put("pilot_results", pilot)
    upload = bundle.get("pilot_upload")
    upload["verified_sha256"]["fits/pilot/k10-rotationNone/results.json"] = bundle.sha(
        "pilot_results"
    )
    bundle.put("pilot_upload", upload, approve=True)
    with pytest.raises(ValueError, match="legacy provenance bridge"):
        bundle.validate()


@pytest.mark.parametrize(
    "mutation", ["wrong_k", "wrong_rotation", "coverage_cell", "seed_order", "dictionary_digest"]
)
def test_equal_counts_do_not_make_other_pilot_cells_compatible(readiness_bundle, mutation):
    bundle = readiness_bundle()
    manifest = bundle.get("pilot_input_manifest")
    binding = bundle.get("pilot_binding")
    if mutation == "wrong_k":
        manifest["contract"]["k"] = 25
    elif mutation == "wrong_rotation":
        manifest["contract"]["rotation"] = 20260913
    elif mutation == "coverage_cell":
        manifest["coverage"]["test"]["contract"]["k"] = 5
    elif mutation == "seed_order":
        binding["generation_seeds"].reverse()
    else:
        binding["dictionary_manifest_sha256"] = "0" * 64
    bundle.put("pilot_input_manifest", manifest)
    bundle.put("pilot_binding", binding)
    bundle.refresh_pilot_links()
    with pytest.raises(
        ValueError, match=r"pilot cell|different component cell|legacy provenance bridge"
    ):
        bundle.validate()


def test_pilot_exit_hash_at_unrelated_upload_path_is_rejected(readiness_bundle):
    bundle = readiness_bundle()
    upload = bundle.get("pilot_upload")
    hashes = upload["verified_sha256"]
    hashes["another_run/pilot_exit.json"] = hashes.pop("pilot_exit.json")
    bundle.put("pilot_upload", upload, approve=True)
    with pytest.raises(ValueError, match="exact verified upload paths"):
        bundle.validate()


@pytest.mark.parametrize("mutation", ["nonzero_exit", "nan_reconstruction"])
def test_rebound_failed_pilot_still_blocks_main(readiness_bundle, mutation):
    bundle = readiness_bundle()
    if mutation == "nonzero_exit":
        terminal = bundle.get("pilot_exit")
        terminal["exit_code"] = 1
        bundle.put("pilot_exit", terminal)
    else:
        pilot = bundle.get("pilot_results")
        pilot["reconstruction"]["R"]["max_abs_reconstruction_error"] = float("nan")
        bundle.put("pilot_results", pilot)
    bundle.refresh_pilot_links()
    with pytest.raises(ValueError, match=r"successful fresh terminal|reconstruction failed"):
        bundle.validate()


def test_review_cannot_approve_counts_above_frozen_ceiling(readiness_bundle):
    bundle = readiness_bundle()
    execution = bundle.get("execution_plan")
    execution["main_counts"]["test"] = 9
    bundle.put("execution_plan", execution, approve=True)
    with pytest.raises(ValueError, match="frozen split ceiling"):
        bundle.validate()
