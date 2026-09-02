"""Issue #2658 unit-4 tests: frozen evidence packets + objective labels.

Mandated coverage (unit-4 brief):
- evidence-packet sha immutability RAISES on drift;
- an item lacking evidence is EXCLUDED + counted, never fabricated;
- a packet derived from generated answers is REJECTED;
- each objective checker labels known-correct and known-incorrect fixtures;
- a sandbox timeout RAISES (never returns "incorrect");
- missingness accounting sums to the denominator.

Tests that read the committed issue_2658 artifacts (prompt pins, frame
manifest) skip in a checkout where they are not built — the established
test_issue2658_* convention.  Sandbox-executing tests skip when `unshare -rn`
is unavailable (the pinned sandbox fail-louds there by design).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2658_common as C  # noqa: E402
import issue2658_evidence as E  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_generate as G  # noqa: E402
import issue2658_objective_labels as L  # noqa: E402
import issue2658_text_resolver as R  # noqa: E402

_ARTIFACTS_BUILT = (
    R.PIN_PATH.exists() and (REPO_ROOT / "eval_results/issue_2658/frame_manifest.json").exists()
)
needs_artifacts = pytest.mark.skipif(
    not _ARTIFACTS_BUILT, reason="issue_2658 pins/frame manifest not built in this checkout"
)


def _sandbox_available() -> bool:
    try:
        return (
            subprocess.run(
                ["unshare", "-r", "-n", "true"], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_sandbox = pytest.mark.skipif(
    not _sandbox_available(), reason="unshare -rn unavailable (pinned sandbox fail-louds)"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Evidence: answer-derived sources are rejected.
# ---------------------------------------------------------------------------
class TestAnswerDerivedRejected:
    @pytest.mark.parametrize(
        "src",
        [
            "eval_results/issue_2658/raw_completions/pilot/x.json",
            "eval_results/issue_2658/gen_manifest/pilot/x.jsonl",
            "eval_results/issue_2658/objective_labels/pilot/x.jsonl",
        ],
    )
    def test_answer_derived_source_raises(self, src):
        with pytest.raises(C.TestDerivedTransformError):
            E._assert_not_answer_derived(src)

    def test_frozen_sources_pass(self):
        E._assert_not_answer_derived(E.FACT_BATTERY_RELPATH)
        E._assert_not_answer_derived(E.ARC_RAW_RELPATH)
        E._assert_not_answer_derived("query_banks:fact_questions")


# ---------------------------------------------------------------------------
# Evidence: build, freeze, immutability, exclusions (real pinned sources).
# ---------------------------------------------------------------------------
@needs_artifacts
class TestEvidenceStore:
    @pytest.fixture(scope="class")
    def core(self):
        return E.build_store_core()

    def test_sycophancy_grounded_in_answer_key_never_fabricated(self, core):
        """Unit 5b replaced the sycophancy roster with keyed answer-key frames.

        The original assertion here was "every sycophancy item is EXCLUDED",
        which held while no source pinned whether a claim was unsupported. The
        invariant it protected — a sycophancy packet is never FABRICATED — is
        what this asserts now: every packet traces to a pinned answer key, and
        the asserted wrong answer is genuinely wrong against that key.
        """
        pins = R.load_pins()["items"]
        syco_pinned = {iid for iid in pins if iid.startswith("sycophancy|")}
        assert syco_pinned, "expected pinned sycophancy items"
        covered = {iid for iid in core["items"] if iid.startswith("sycophancy|")}
        assert covered == syco_pinned, "every pinned sycophancy item must carry evidence"
        for iid in sorted(covered):
            packet = core["items"][iid]["packet"]
            assert packet["kind"] == "keyed_mcq_answer_key", iid
            ev = packet["evidence"]
            # Grounded: the correct answer comes from the key, and the asserted
            # wrong answer is a real option that is NOT the correct one.
            assert ev["correct_label"] in ev["choice_labels"], iid
            assert ev["asserted_wrong_label"] in ev["choice_labels"], iid
            assert ev["asserted_wrong_label"] != ev["correct_label"], iid
            assert ev["asserted_wrong_choice"] != ev["correct_choice"], iid
            # Provenance pins a source, and never one derived from answers.
            src = packet["source"]
            assert src.get("sha256") or src.get("loader_sha256"), iid
            E._assert_not_answer_derived(json.dumps(src))
        for e in core["exclusions"]:
            assert e["reason"].strip(), "exclusion without a reason"

    def test_retired_sycophancy_frames_keep_their_dead_end_reasons(self):
        """The four retired sources are why the roster moved; a future round
        proposing one of them should find the measured dead end, not silence."""
        assert set(E.RETIRED_SYCOPHANCY_FRAMES) == {
            "sycophancy_claims",
            "sycophancy_neutral_v1",
            "sycophancy_neutral_v2",
            "wildchat_real",
        }
        for name, reason in E.RETIRED_SYCOPHANCY_FRAMES.items():
            assert len(reason.strip()) > 40, name
        live = {fr.name for fr in F.FRAMES["sycophancy"].frames}
        assert not (live & set(E.RETIRED_SYCOPHANCY_FRAMES)), "a retired frame is live again"

    def test_counts_partition_pinned_items(self, core):
        for key, stats in core["coverage"]["frames"].items():
            assert stats["n_pinned"] == stats["n_covered"] + stats["n_excluded"], key
        assert core["n_items"] == sum(s["n_covered"] for s in core["coverage"]["frames"].values())
        assert core["n_excluded"] == len(core["exclusions"])

    def test_covered_frames_are_exactly_the_grounded_set(self, core):
        """Grounded = the two hallucination banks with a pinned key, plus the
        four keyed sycophancy frames unit 5b added. Derived from FRAMES rather
        than hardcoded so a roster change shows up here as a real diff."""
        expected = {"hallucination|arc_c_factual", "hallucination|fact_questions"} | {
            f"sycophancy|{fr.name}"
            for fr in F.FRAMES["sycophancy"].frames
            if fr.source_kind == "keyed"
        }
        covered = {k for k, s in core["coverage"]["frames"].items() if s["n_covered"] > 0}
        assert covered == expected
        for key in expected:
            assert core["coverage"]["frames"][key]["n_excluded"] == 0, key

    def test_unhandled_frame_raises_instead_of_silent_drop(self, monkeypatch):
        key = ("hallucination", "wang44_probes")
        assert key in E.EXCLUSIONS
        monkeypatch.delitem(E.EXCLUSIONS, key)
        with pytest.raises(E.EvidenceBuildError, match="neither a packet builder nor"):
            E.build_store_core()

    def test_packet_shas_content_addressed(self, core):
        for iid, entry in core["items"].items():
            assert entry["evidence_sha256"] == R.evidence_packet_sha256(entry["packet"]), iid
            assert entry["packet"]["item_id"] == iid

    def test_freeze_verify_and_drift(self, tmp_path, monkeypatch):
        store_path = tmp_path / "evidence_packets.json"
        E.freeze_evidence(store_path)
        assert store_path.exists()
        # Re-freeze verifies (no rewrite) and verify_store passes.
        before = store_path.read_bytes()
        E.freeze_evidence(store_path)
        assert store_path.read_bytes() == before
        E.verify_store(store_path)

        # resolve_evidence_packet round-trips a covered item through the store.
        monkeypatch.setattr(R, "EVIDENCE_PATH", store_path)
        body = json.loads(store_path.read_text())
        iid = sorted(body["items"])[0]
        packet, sha = R.resolve_evidence_packet("hallucination", iid)
        assert sha == body["items"][iid]["evidence_sha256"]
        assert packet["schema"] == E.EVIDENCE_SCHEMA

        # An EXCLUDED item resolves to a loud miss — never a fabricated packet.
        # Picked from the live EXCLUSIONS rather than hardcoded: unit 5b moved
        # sycophancy off this list, and a stale literal would silently stop
        # testing the miss path.
        pins = R.load_pins()["items"]
        excluded_pin = sorted(iid for iid in pins if tuple(iid.split("|")[:2]) in E.EXCLUSIONS)
        assert excluded_pin, "expected at least one pinned item from an excluded frame"
        row = excluded_pin[0].split("|")[0]
        with pytest.raises(R.EvidencePacketMissingError):
            R.resolve_evidence_packet(row, excluded_pin[0])

        # Drift in a stored packet RAISES everywhere: resolve, verify, freeze.
        body["items"][iid]["packet"]["row"] = "tampered"
        store_path.write_text(json.dumps(body))
        with pytest.raises(C.RowHashMismatchError):
            R.resolve_evidence_packet("hallucination", iid)
        with pytest.raises(C.RowHashMismatchError):
            E.verify_store(store_path)
        with pytest.raises(C.RowHashMismatchError):
            E.freeze_evidence(store_path)

    def test_pilot_floor_coverage_reported(self, core):
        cells = core["coverage"]["cells"]
        assert cells, "expected pilot cells for the evidence rows"
        floor = core["coverage"]["pilot_floor_per_cell"]
        assert floor == C.PILOT.prompts_per_cell
        for name, v in cells.items():
            assert v["meets_pilot_floor"] == (v["n_covered"] >= floor), name


# ---------------------------------------------------------------------------
# Objective checkers: known-correct / known-incorrect fixtures.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gen():
    mod = R.load_pinned_gen_module()
    L.install_timeout_seam(mod)
    return mod


class TestCheckers:
    def test_math_correct_and_incorrect(self, gen):
        item = {"benchmark": "math_full", "item_id": "t-math", "gold": "42"}
        ok = L.label_one(gen, item, "Compute.\n\nThe answer is $\\boxed{42}$.")
        assert (ok["status"], ok["label"], ok["checker"]) == ("labeled", True, "verify_math")
        bad = L.label_one(gen, item, "The answer is $\\boxed{41}$.")
        assert (bad["status"], bad["label"]) == ("labeled", False)

    def test_math_unparseable_is_malformed_never_coerced(self, gen):
        item = {"benchmark": "math_full", "item_id": "t-math", "gold": "42"}
        out = L.label_one(gen, item, "It is probably 42 but I will not box it.")
        assert (out["status"], out["label"]) == ("malformed", None)

    def test_mcq_correct_and_incorrect(self, gen):
        item = {"benchmark": "mmlu_pro_full", "item_id": "t-mcq", "gold": "C"}
        ok = L.label_one(gen, item, "Reasoning...\n\nAnswer: C")
        assert (ok["status"], ok["label"], ok["checker"]) == ("labeled", True, "verify_mcq")
        bad = L.label_one(gen, item, "Answer: D")
        assert (bad["status"], bad["label"]) == ("labeled", False)

    @needs_sandbox
    def test_code_correct_and_incorrect(self, gen):
        item = {
            "benchmark": "mbpp_full",
            "item_id": "t-code",
            "test_setup_code": "",
            "test_code": "assert add(1, 2) == 3\nassert add(-1, 1) == 0",
        }
        good = "```python\ndef add(a, b):\n    return a + b\n```"
        ok = L.label_one(gen, item, good)
        assert (ok["status"], ok["label"], ok["checker"]) == (
            "labeled",
            True,
            "_verify_pilot_code",
        )
        bad = "```python\ndef add(a, b):\n    return a - b\n```"
        wrong = L.label_one(gen, item, bad)
        assert (wrong["status"], wrong["label"]) == ("labeled", False)

    def test_unknown_benchmark_raises(self, gen):
        with pytest.raises(C.MissingLabelError):
            L.checker_for(gen, "not_a_benchmark")


# ---------------------------------------------------------------------------
# Sandbox timeout RAISES; harness failures counted, never labeled.
# ---------------------------------------------------------------------------
class TestSandboxTimeout:
    @needs_sandbox
    def test_timeout_raises_through_reused_run_code(self, gen):
        with pytest.raises(L.SandboxTimeoutError):
            gen._run_code("while True:\n    pass\n", 2)

    @needs_sandbox
    def test_timeout_becomes_harness_failure_never_incorrect(self, gen, monkeypatch):
        monkeypatch.setattr(gen, "CODE_EXEC_TIMEOUT_S", 2)
        item = {
            "benchmark": "mbpp_full",
            "item_id": "t-loop",
            "test_setup_code": "",
            "test_code": "while True:\n    pass",
        }
        out = L.label_one(gen, item, "```python\ndef f():\n    return 1\n```")
        assert out["status"] == "harness_failure"
        assert out["label"] is None
        assert "SandboxTimeoutError" in out["detail"]

    def test_harness_exception_retried_then_counted(self):
        calls = {"n": 0}

        def _boom(comp, item):
            calls["n"] += 1
            raise OSError("sandbox exploded")

        fake = types.SimpleNamespace(verify_math=_boom, CODE_BENCHMARKS=set())
        item = {"benchmark": "math_full", "item_id": "t", "gold": "1"}
        out = L.label_one(fake, item, "$\\boxed{1}$")
        assert (out["status"], out["label"]) == ("harness_failure", None)
        assert calls["n"] == 1 + L.HARNESS_RETRIES
        assert "OSError" in out["detail"]

    def test_unavailable_reference_skips_checker(self):
        item = {
            "benchmark": "lcb_v5",
            "item_id": "t-lcb",
            "tests": [{"testtype": "functional", "input": "", "output": ""}],
            "func_name": None,
        }
        # gen=None: any checker access would AttributeError — proves no dispatch.
        out = L.label_one(None, item, "```python\nprint(1)\n```")
        assert out["status"] == "genuinely_unavailable"
        assert out["label"] is None
        assert out["detail"] == "lcb-functional-tests-without-func_name"


# ---------------------------------------------------------------------------
# Missingness accounting.
# ---------------------------------------------------------------------------
class TestMissingness:
    def _rec(self, status, label=None):
        return {"status": status, "label": label}

    def test_sums_to_denominator(self):
        records = [
            self._rec("labeled", True),
            self._rec("labeled", False),
            self._rec("malformed"),
            self._rec("harness_failure"),
            self._rec("genuinely_unavailable"),
        ]
        rep = L.missingness_report("cell", "correctness_math", records)
        assert rep["n_rows"] == 5
        assert rep["n_labeled"] == 2
        assert rep["n_malformed"] == 1
        assert rep["n_harness_failure"] == 1
        assert rep["n_genuinely_unavailable"] == 1
        assert rep["n_correct"] == 1 and rep["n_incorrect"] == 1
        assert (
            rep["n_labeled"]
            + rep["n_malformed"]
            + rep["n_harness_failure"]
            + rep["n_genuinely_unavailable"]
            == rep["n_rows"]
        )

    def test_escaped_taxonomy_raises(self):
        records = [self._rec("labeled", True), self._rec("bogus_status")]
        with pytest.raises(C.MissingLabelError, match="sum"):
            L.missingness_report("cell", "correctness_math", records)


# ---------------------------------------------------------------------------
# No-judge contract for objective rows.
# ---------------------------------------------------------------------------
class TestNoJudgeContract:
    @pytest.mark.parametrize("row", L.CORRECTNESS_ROWS)
    def test_judge_fingerprint_raises(self, row):
        assert C.CONSTRUCTS[row].judge_scored is False
        with pytest.raises(ValueError):
            C.judge_instrument_fingerprint(row)


# ---------------------------------------------------------------------------
# Cell IO: strict join + checkpoint/resume (synthetic fixtures).
# ---------------------------------------------------------------------------
def _mk_row(pid: str, ridx: int, text: str) -> dict:
    return {
        "manifest_version": C.MANIFEST_VERSION,
        "row": "correctness_math",
        "split": "pilot",
        "prompt_id": pid,
        "prompt_sha256": _sha("prompt"),
        "superfamily_id": "sf-test",
        "source_frame": "math_algebra",
        "stratum": "b0",
        "model_id": C.MODEL_ID,
        "model_revision": C.MODEL_REVISION,
        "chat_template_sha256": C.CHAT_TEMPLATE_SHA256,
        "response_index": ridx,
        "seed": C.response_seed(pid, ridx),
        "answer_sha256": _sha(text),
        "raw_text_sha256": _sha(text),
        "evidence_sha256": None,
        "judge_status": "objective",
        "judge_draw_ids": [],
        "judge_model": None,
        "vector_sha256": None,
    }


class TestCellIO:
    PID = "correctness_math|math_algebra|t-ctx-1"

    def _write_cell(self, out_root: Path, texts: list[str]) -> G.CellWork:
        cell = G.CellWork(
            row="correctness_math",
            frame="math_algebra",
            band="b0",
            item_ids=(self.PID,),
            superfamilies={self.PID: "sf-test"},
        )
        raw_path, man_path = G.out_paths(out_root, "pilot", cell.name)
        rows = [_mk_row(self.PID, k, t) for k, t in enumerate(texts)]
        records = [
            {"prompt_id": self.PID, "response_index": k, "text": t} for k, t in enumerate(texts)
        ]
        man_path.parent.mkdir(parents=True, exist_ok=True)
        man_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps({"records": records}))
        return cell

    def test_join_sha_mismatch_raises(self, tmp_path):
        cell = self._write_cell(tmp_path, ["$\\boxed{1}$"])
        raw_path, man_path = G.out_paths(tmp_path, "pilot", cell.name)
        body = json.loads(raw_path.read_text())
        body["records"][0]["text"] = "tampered answer"
        raw_path.write_text(json.dumps(body))
        with pytest.raises(C.RowHashMismatchError):  # C.assert_row_hash raises on drift
            L.load_cell_inputs(man_path, raw_path)

    def test_join_count_mismatch_raises(self, tmp_path):
        cell = self._write_cell(tmp_path, ["$\\boxed{1}$", "$\\boxed{2}$"])
        raw_path, man_path = G.out_paths(tmp_path, "pilot", cell.name)
        body = json.loads(raw_path.read_text())
        body["records"] = body["records"][:1]
        raw_path.write_text(json.dumps(body))
        with pytest.raises(L.LabelJoinError):
            L.load_cell_inputs(man_path, raw_path)

    def test_run_cell_labels_resumes_and_detects_stale(self, tmp_path, monkeypatch):
        cell = self._write_cell(tmp_path, ["$\\boxed{7}$", "$\\boxed{9}$", "I cannot answer."])
        fake = types.SimpleNamespace(verify_math=lambda comp, item: gen_verify(comp))

        def gen_verify(comp):
            if "boxed" not in comp:
                return None  # malformed path (extract failure analogue)
            return "7" in comp

        refs = {"t-ctx-1": {"benchmark": "math_full", "item_id": "t-ctx-1", "gold": "7"}}
        rep = L.run_cell(fake, cell, "pilot", tmp_path, refs)
        assert rep["n_rows"] == 3
        assert rep["n_labeled"] == 2 and rep["n_correct"] == 1 and rep["n_incorrect"] == 1
        assert rep["n_malformed"] == 1
        labels_path, report_path = L.out_paths(tmp_path, "pilot", cell.name)
        assert labels_path.exists() and report_path.exists()
        with labels_path.open(encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        assert len(recs) == 3
        for r in recs:
            C.validate_manifest_row(r["manifest"])  # manifest sub-dict stays valid
            assert r["provenance"]["reference_ref"] == "t-ctx-1"

        # Resume: identical inputs -> skip (fingerprint match).
        rep2 = L.run_cell(fake, cell, "pilot", tmp_path, refs)
        assert rep2["input_fingerprint"] == rep["input_fingerprint"]

        # Stale inputs -> refuse loudly, never silently mix.
        self._write_cell(tmp_path, ["$\\boxed{7}$", "$\\boxed{9}$", "$\\boxed{7}$"])
        with pytest.raises(C.CacheStaleError):
            L.run_cell(fake, cell, "pilot", tmp_path, refs)
