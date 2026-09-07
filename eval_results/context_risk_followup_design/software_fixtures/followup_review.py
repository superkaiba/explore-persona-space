"""Independent CPU-only review fixtures; all native records/HTTP/model work are mocked.

Run from repository root with the pinned Inspect/OpenAI uv overlay and --output PATH.
Separate actual Docker tenth-attempt tests are retained in tests/test_context_risk_followup.py.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from omegaconf import OmegaConf

from scripts import context_risk_followup as f


class ReachedGeneration(Exception):
    pass


class Score:
    def __init__(self, value):
        self.value = value

    def model_dump(self, **kwargs):
        return {"value": self.value}


class Error:
    def model_dump(self, **kwargs):
        return {"message": "fixture infrastructure error"}


def metadata(root, arm="A", phase="development"):
    path = root / "manifests" / f"{phase}_{arm}.jsonl"
    return {
        "schema_version": "context_risk_followup_v1",
        "phase": phase,
        "arm": arm,
        "epochs": 2 if phase == "development" else 4,
        "max_attempts": 10,
        "message_limit": 22,
        "manifest_sha256": f.sha256(path),
        "sources_sha256": f.source_hashes(),
        "harness_fingerprint": f.harness_fingerprint(),
        "model": "openai-api/local/Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "max_connections": 16,
        "freeze_sha256": f.sha256(root / "manifests/freeze.json"),
    }


def config(root):
    cfg = OmegaConf.load(f.ROOT / "configs/eval/context_risk_followup.yaml")
    cfg.root = str(root)
    cfg.review = str(root / "fixture_review.json")
    cfg.base_url = "http://127.0.0.1:18000/v1"
    cfg.pilot_limit = 16
    return cfg


def setup(root):
    cfg = config(root)
    freeze = f.freeze(cfg)
    (root / "fixture_review.json").write_text(
        json.dumps(
            {"verdict": "PASS", "reviewer": "FIXTURE ONLY", "files_sha256": f.source_hashes()}
        )
    )
    return cfg, freeze


def native(root, cfg, arm="A", pilot=False):
    samples = f.load_samples(root / "manifests" / f"development_{arm}.jsonl")
    meta = metadata(root, arm)
    positive_tasks = sorted(f.DEV_IDS)[:3]
    rows = []
    for sample in samples[:16] if pilot else samples:
        for epoch in (1, 2):
            success = sample.metadata["condition"] == "original" or (
                sample.metadata["condition"] == "oneoff"
                and sample.metadata["task_id"] in positive_tasks
            )
            history = []
            for attempt in range(1, 2 if success else 11):
                history.append(
                    {
                        "attempt": attempt,
                        "request_seed": int.from_bytes(
                            hashlib.sha256(
                                f"38295:{sample.id}:{epoch}:{attempt}".encode()
                            ).digest()[:4],
                            "big",
                        ),
                        "success": success,
                        "category": "passed" if success else "assertion_failure",
                        "stop_reasons": ["stop"],
                        "response": "FIXTURE ONLY",
                        "answer": "",
                    }
                )
            row_metadata = copy.deepcopy(sample.metadata)
            row_metadata["agentic_results"] = {
                "harness_fingerprint": f.harness_fingerprint(),
                "max_attempts": 10,
                "attempt_history": history,
                "flag_for_human_intervention": False,
                "censored": False,
            }
            rows.append(
                NS(
                    id=sample.id,
                    epoch=epoch,
                    input=sample.input,
                    metadata=row_metadata,
                    messages=[NS(text=sample.input)],
                    error=None,
                    invalidation=None,
                    scores={"successful_submission": Score("C" if success else "I")},
                )
            )
    location = root / f"fixture_native_{arm}.eval"
    location.write_text("NOT AN INSPECT LOG; read_eval_log is mocked for these CPU tests")
    log = NS(
        status="success",
        location=str(location),
        samples=rows,
        eval=NS(
            metadata=meta,
            model=str(cfg.model),
            model_base_url=str(cfg.base_url),
            model_args={
                "responses_api": False,
                "stream": False,
                "client_timeout": 7200,
                "max_retries": 0,
            },
            sandbox=NS(type="docker"),
            packages={"inspect_ai": "0.3.261"},
            task_id="fixture_task",
            task="context_risk_followup_ten_submissions",
            dataset=NS(sample_ids=[s.id for s in samples]),
            config=NS(
                epochs=2,
                message_limit=22,
                max_samples=16,
                max_sandboxes=16,
                max_subprocesses=16,
                fail_on_error=False,
                score_on_error=True,
                sample_shuffle=None,
            ),
        ),
        plan=NS(
            config=NS(
                temperature=1.0,
                top_p=1.0,
                max_tokens=65536,
                seed=38295,
                max_connections=16,
                max_retries=2,
                extra_body=f.GENERATION_EXTRA_BODY,
            )
        ),
    )
    return samples, log


def tokens(samples, base_url, model):
    return {
        "contexts": [
            {
                "sample_id": s.id,
                "exact_context_sha256": s.metadata["exact_context_sha256"],
                "n_prefix_tokens": 1,
                "prefix_token_ids_sha256": f._stable_digest([42]),
                "token_ids": [42],
            }
            for s in samples
        ],
        "n_contexts": len(samples),
        "passed": True,
    }


def saved_report(root, arm, log):
    report = f.summarize_logs([log], epochs=2)
    report.update(
        phase="development",
        arm=arm,
        is_pilot=False,
        requested_rollouts=120,
        coverage_complete=True,
        manifest_sha256=metadata(root, arm)["manifest_sha256"],
        sources_sha256=f.source_hashes(),
        native_logs_sha256={log.location: f.sha256(Path(log.location))},
    )
    out = root / f"development_{arm}"
    out.mkdir(exist_ok=True)
    (out / "run_result.json").write_text(json.dumps(report))
    return report


def main(output):  # noqa: C901 - bounded independent negative-fixture census
    results = []

    def check(name, body):
        try:
            body()
            results.append({"name": name, "passed": True})
        except Exception as error:
            results.append(
                {"name": name, "passed": False, "error": f"{type(error).__name__}: {error}"}
            )

    def rejects(body):
        try:
            body()
        except (ValueError, FileExistsError, KeyError):
            return
        raise AssertionError("Invalid fixture was accepted")

    with tempfile.TemporaryDirectory(prefix="followup-independent-review-") as temp:
        root = Path(temp) / "canonical"
        cfg, frozen = setup(root)
        assert frozen["role_tasks"] == {
            "recipe_development": 20,
            "probe_training": 63,
            "final_test": 20,
        }
        check(
            "freeze_20_63_20_and_60_249_contexts",
            lambda: (
                (
                    len(f.load_samples(root / "manifests/development_A.jsonl")) == 60
                    and len(f.load_samples(root / "manifests/fresh_A.jsonl")) == 249
                )
                or (_ for _ in ()).throw(AssertionError())
            ),
        )
        check("repeat_freeze_rejected", lambda: rejects(lambda: f.freeze(cfg)))
        samples, log = native(root, cfg)
        check("valid_native_record", lambda: f.validate_native(log, samples, metadata(root), cfg))
        mutations = {
            "model": lambda item: setattr(item.eval, "model", "wrong/model"),
            "endpoint": lambda item: setattr(item.eval, "model_base_url", "http://wrong"),
            "provider_args": lambda item: item.eval.model_args.update(stream=True),
            "temperature": lambda item: setattr(item.plan.config, "temperature", 0.0),
            "message_limit": lambda item: setattr(item.eval.config, "message_limit", 10),
            "inspect_version": lambda item: item.eval.packages.update(inspect_ai="0.0"),
            "duplicate_sample": lambda item: item.samples.append(copy.deepcopy(item.samples[0])),
            "unexpected_epoch": lambda item: setattr(item.samples[0], "epoch", 3),
            "input": lambda item: setattr(item.samples[0], "input", "wrong"),
            "first_message": lambda item: setattr(item.samples[0].messages[0], "text", "wrong"),
            "sample_metadata": lambda item: item.samples[0].metadata.update(condition="wrong"),
            "harness": lambda item: (
                item.samples[0].metadata["agentic_results"].update(harness_fingerprint="wrong")
            ),
            "seed": lambda item: (
                item.samples[0]
                .metadata["agentic_results"]["attempt_history"][0]
                .update(request_seed=0)
            ),
            "score_mismatch": lambda item: setattr(
                item.samples[0].scores["successful_submission"],
                "value",
                "I" if item.samples[0].scores["successful_submission"].value == "C" else "C",
            ),
        }
        for name, mutate in mutations.items():
            bad = copy.deepcopy(log)
            mutate(bad)
            check(
                "native_rejects_" + name,
                lambda bad=bad: rejects(
                    lambda: f.validate_native(bad, samples, metadata(root), cfg)
                ),
            )

        for case in ("normal", "missing_score", "invalidation", "infrastructure", "truncation"):
            case_root = Path(temp) / case
            case_cfg, _ = setup(case_root)
            _case_samples, case_log = native(case_root, case_cfg, pilot=True)
            row = next(
                r for r in case_log.samples if r.scores["successful_submission"].value == "I"
            )
            if case == "missing_score":
                row.scores = {}
            elif case == "invalidation":
                row.invalidation = "fixture invalidation"
                row.metadata["agentic_results"]["attempt_history"] = row.metadata[
                    "agentic_results"
                ]["attempt_history"][:1]
            elif case == "infrastructure":
                row.error = Error()
            elif case == "truncation":
                row.metadata["agentic_results"]["censored"] = True
                row.metadata["agentic_results"]["attempt_history"] = row.metadata[
                    "agentic_results"
                ]["attempt_history"][:1]
                row.metadata["agentic_results"]["attempt_history"][0].update(
                    category="generation_incomplete", stop_reasons=["max_tokens"]
                )
                row.scores["successful_submission"].value = "N"

            def run_case(case=case, case_root=case_root, case_cfg=case_cfg, case_log=case_log):
                with (
                    patch.object(f, "tokenize_samples", side_effect=tokens),
                    patch.object(f, "eval", return_value=[case_log]),
                ):
                    try:
                        f.run(case_cfg)
                    except RuntimeError:
                        assert case != "normal"
                report = json.loads((case_root / "development_A/pilot_result.json").read_text())
                assert report["realized_rollouts"] == 32
                assert report["technical_errors"] == int(case != "normal")
                assert report["passed"] == (case == "normal")
                assert len(f.read_rows(case_root / "development_A/pilot_rollouts.jsonl")) == 32

            check("run_preserves_" + case, run_case)

        prefix_root = Path(temp) / "prefix"
        prefix_cfg, _ = setup(prefix_root)
        prefix_samples, _ = native(prefix_root, prefix_cfg, pilot=True)
        prefix_path = prefix_root / "development_A/prefix_tokens.json"
        prefix_path.parent.mkdir()
        old_tokens = tokens(prefix_samples, "", "")
        old_tokens["contexts"][0]["token_ids"] = [999]
        prefix_path.write_text(json.dumps(old_tokens))
        old_bytes = prefix_path.read_bytes()

        def prefix_reject():
            with (
                patch.object(f, "tokenize_samples", side_effect=tokens),
                patch.object(f, "eval", side_effect=ReachedGeneration),
            ):
                rejects(lambda: f.run(prefix_cfg))
            assert prefix_path.read_bytes() == old_bytes

        check("changed_prefix_rejected_without_overwrite", prefix_reject)
        changed_cfg = copy.deepcopy(prefix_cfg)
        changed_cfg.model = "openai-api/local/different/model"

        def model_override_reject():
            with (
                patch.object(f, "tokenize_samples", side_effect=ReachedGeneration),
                patch.object(f, "eval", side_effect=ReachedGeneration),
            ):
                rejects(lambda: f.run(changed_cfg))

        check("run_rejects_unreviewed_model_revision", model_override_reject)

        select_root = Path(temp) / "selection"
        select_cfg, _ = setup(select_root)
        logs = {}
        for arm in ("A", "B"):
            _, item = native(select_root, select_cfg, arm=arm)
            saved_report(select_root, arm, item)
            logs[item.location] = item

        def selection_valid():
            with patch.object(f, "read_eval_log", side_effect=logs.__getitem__):
                selection = f.select(select_cfg)
            assert selection["passed"] and selection["selected_arm"] == "B"
            assert selection["arms"]["B"]["positive_tasks"] == 3
            assert selection["arms"]["B"]["positive"] == 6

        check("complete_native_selection_and_tie_to_B", selection_valid)
        check("selection_immutable", lambda: rejects(lambda: f.select(select_cfg)))
        selection_path = select_root / "selection.json"
        selection_original = selection_path.read_bytes()
        fresh_cfg = copy.deepcopy(select_cfg)
        fresh_cfg.phase, fresh_cfg.arm, fresh_cfg.pilot_limit = "fresh", "B", None

        def selection_binding_bad(kind):
            selection = json.loads(selection_original)
            if kind == "freeze":
                selection["freeze_sha256"] = "wrong"
            else:
                selection["arms"]["A"]["source_sha256"] = "wrong"
            selection_path.write_text(json.dumps(selection))
            with (
                patch.object(f, "tokenize_samples", side_effect=ReachedGeneration),
                patch.object(f, "eval", side_effect=ReachedGeneration),
            ):
                rejects(lambda: f.run(fresh_cfg))

        check("fresh_rejects_stale_freeze", lambda: selection_binding_bad("freeze"))
        check("fresh_rejects_changed_development", lambda: selection_binding_bad("development"))

        for defect in (
            "phase",
            "sources",
            "report_counts",
            "native_digest",
            "censored",
            "failed_native_status",
        ):
            defect_root = Path(temp) / ("selection_" + defect)
            defect_cfg, _ = setup(defect_root)
            defect_logs = {}
            for arm in ("A", "B"):
                _, item = native(defect_root, defect_cfg, arm=arm)
                if arm == "A" and defect == "censored":
                    item.samples[0].error = Error()
                if arm == "A" and defect == "failed_native_status":
                    item.status = "error"
                report = saved_report(defect_root, arm, item)
                defect_logs[item.location] = item
                if arm == "A":
                    if defect == "phase":
                        report["phase"] = "fresh"
                    elif defect == "sources":
                        report["sources_sha256"]["scripts/context_risk_followup.py"] = "wrong"
                    elif defect == "report_counts":
                        report["contexts"][0]["passed"] += 1
                    elif defect == "native_digest":
                        report["native_logs_sha256"][item.location] = "wrong"
                    elif defect in {"censored", "failed_native_status"}:
                        # Test native validation independently of the saved passed bit.
                        report["passed"] = True
                    (defect_root / "development_A/run_result.json").write_text(json.dumps(report))

            def reject_selection(defect_cfg=defect_cfg, defect_logs=defect_logs):
                with patch.object(f, "read_eval_log", side_effect=defect_logs.__getitem__):
                    rejects(lambda: f.select(defect_cfg))

            check("selection_rejects_" + defect, reject_selection)

        wrapper_dir = Path(temp) / "wrapper"
        wrapper_dir.mkdir()
        bin_dir = wrapper_dir / "bin"
        bin_dir.mkdir()
        fake_uv = bin_dir / "uv"
        fake_uv.write_text(
            '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$FIXTURE_ARGUMENTS"\nexit 7\n'
        )
        fake_uv.chmod(0o755)
        args_file = wrapper_dir / "args.txt"
        env = dict(
            os.environ,
            PATH=str(bin_dir) + ":" + os.environ["PATH"],
            EPM_CONTEXT_RISK_FOLLOWUP_ROOT=str(wrapper_dir),
            EPM_CONTEXT_RISK_LAUNCH_ID="critic_fixture",
            EPM_CONTEXT_RISK_PROCESS_TIMEOUT_SECONDS="10",
            FIXTURE_ARGUMENTS=str(args_file),
        )

        def supervisor_dispatch():
            result = subprocess.run(
                [
                    "bash",
                    str(f.ROOT / "scripts/context_risk_followup_supervise.sh"),
                    "development_b_pilot",
                    "phase=development",
                    "arm=B",
                    "pilot_limit=16",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            assert result.returncode == 7, result.stderr
            receipt = json.loads(
                (wrapper_dir / "development_b_pilot_critic_fixture_process.exit.json").read_text()
            )
            assert receipt["exit_code"] == 7 and receipt["cleanup"] == "no_live_members"
            argv = args_file.read_text().splitlines()
            assert "scripts.context_risk_followup" in argv
            assert "arm=B" in argv and "pilot_limit=16" in argv

        check("supervisor_fake_uv_exit7_dispatch_and_cleanup", supervisor_dispatch)

        def supervisor_mode_wins():
            changed_env = dict(env, EPM_CONTEXT_RISK_LAUNCH_ID="critic_mode_fixture")
            result = subprocess.run(
                [
                    "bash",
                    str(f.ROOT / "scripts/context_risk_followup_supervise.sh"),
                    "fresh_b",
                    "phase=development",
                    "arm=A",
                    "pilot_limit=16",
                ],
                env=changed_env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            assert result.returncode == 7, result.stderr
            overrides = dict(
                arg.split("=", 1)
                for arg in args_file.read_text().splitlines()
                if "=" in arg and "==" not in arg
            )
            assert overrides["operation"] == "run" and overrides["phase"] == "fresh"
            assert overrides["arm"] == "B" and overrides["pilot_limit"] == "null"

        check("supervisor_mode_overrides_conflicting_cli_arguments", supervisor_mode_wins)

    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "fixture_type": "software_only_mocked_native_records_never_experiment_results",
        "model_calls": 0,
        "http_calls": 0,
        "source_hashes": f.source_hashes(),
        "fixture_source_sha256": f.sha256(Path(__file__)),
        "cases": results,
        "passed": all(item["passed"] for item in results),
    }
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args().output))
