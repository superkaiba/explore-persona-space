"""#2061 cross-machine wiring + pod-side reporting pins (round-2 unit E).

Round 1 was FAILed on (C4) zero upload/staging wiring for the plan §9
`off_pod_phases` handoffs, (M6) a `--smoke-only` mode covering ONLY the P1
loader-parity gate, and a missing pod-side reporting contract (no
`[phase=...]` / `[phase=done]` emission, no results sentinel). These tests
pin the fixes:

- the sentinel writer conforms to `poll_pipeline.py`'s OWN parser
  (`_parse_sentinel` + `_SENTINEL_REQUIRED_KEYS` + schema version — the
  #448 false-`dead` contract);
- every `[phase=...]` token the dispatcher emits parses under the poller's
  `PHASE_RE`, and the RESERVED `[phase=done]` token appears exactly once
  (the `finish` path; #545 reserved-token discipline);
- the smoke mode reaches EVERY phase script (review M6);
- each phase script carries its plan-declared upload/staging call
  (review C4, the Step 0.65 grep recipe, mechanized);
- `stage_turnstore` maps hub shard paths into the consumer's own layout
  (real body executed; only the Hub boundary faked, signature-conformant).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import issue2061_hub_io as hio  # noqa: E402
import issue2061_sae_encode as enc  # noqa: E402


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (the poll-pipeline test loader)."""
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_issue2061_under_test")

DISPATCH = (SCRIPTS / "issue2061_dispatch.sh").read_text()


# ---------------------------------------------------------------------------
# Pod-side reporting contract (pod-side-reporting.md items 1-2)
# ---------------------------------------------------------------------------
def test_sentinel_conforms_to_poll_pipeline_contract(tmp_path):
    path = hio.write_sentinel("epm:smoke-result", '{"ok": true}', sentinel_dir=tmp_path)
    assert path is not None and path.exists()
    assert path.name.startswith("issue-2061-epm_smoke-result-") and path.suffix == ".json"
    parsed = pp._parse_sentinel(str(path), path.read_text())
    assert parsed is not None, "poll_pipeline._parse_sentinel rejected the sentinel"
    assert parsed["kind"] == "epm:smoke-result"
    assert parsed["version"] == 1
    assert parsed["sentinel_schema_version"] == pp.SENTINEL_SCHEMA_VERSION_SUPPORTED
    missing = [k for k in pp._SENTINEL_REQUIRED_KEYS if k not in parsed]
    assert not missing, f"missing required sentinel keys: {missing}"


def test_sentinel_skips_cleanly_off_pod(tmp_path):
    # VM-local lanes have no /workspace/logs — skipping (None) is the designed
    # disposition, never a crash and never a stray directory creation.
    absent = tmp_path / "no-such-dir"
    assert hio.write_sentinel("epm:results", "{}", sentinel_dir=absent) is None
    assert not absent.exists()


def test_dispatcher_phase_tokens_parse_and_done_is_reserved():
    tokens = re.findall(r"\[phase=([a-z0-9_]+)\]", DISPATCH)
    assert tokens, "dispatcher emits no [phase=...] breadcrumbs"
    for tok in tokens:
        m = pp.PHASE_RE.search(f"[phase={tok}]")
        assert m and m.group(1) == tok, f"[phase={tok}] does not parse under PHASE_RE"
    # The RESERVED terminal token is EMITTED exactly once — in finish() — so a
    # per-phase echo can never mint a false status=done (#545). Comment
    # mentions don't count; only echo emissions do.
    assert DISPATCH.count('echo "[phase=done]"') == 1
    # Every mode ends through finish() (sentinel + [phase=done]).
    assert DISPATCH.count("finish ") >= 3  # smoke, --all, --phase


# ---------------------------------------------------------------------------
# Smoke-mode phase coverage (review M6)
# ---------------------------------------------------------------------------
def test_smoke_mode_reaches_every_phase_script():
    m = re.search(r"run_smoke\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert m, "run_smoke() block not found"
    body = m.group(1)
    for script in [
        "issue2061_sae_encode.py",
        "issue2061_fit_per_feature.py",
        "issue2061_null.py",
        "issue2061_fitness.py",
        "issue2061_figures.py",
    ]:
        assert script in body, f"--smoke-only does not reach {script} (review M6)"
    # Smoke outputs are rooted OUTSIDE the canonical trees.
    assert "ISSUE2061_SMOKE_ROOT" in body or "SMOKE_ROOT" in body


def test_p3_runner_wires_refit_inputs():
    # Round-1 gap: run_p3_null passed no --context-shard-dir/--encoded-dir, so
    # the null errored the moment refit inputs were needed.
    m = re.search(r"run_p3_null\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert m, "run_p3_null() block not found"
    body = m.group(1)
    assert "--stage-context-from-hub" in body and "--context-shard-dir" in body
    assert "--stage-r2-from-hub" in body and "--r2-dir" in body
    assert "--stage-encoded-from-hub" in body and "--encoded-dir" in body


def test_p3_aggregation_expects_registered_cell_count_but_smoke_does_not():
    # m2 (round 3): the PRODUCTION aggregation pass carries the fail-loud
    # 64-cell guard (registered statistic, plan §Design), while the smoke
    # chain must NEVER inherit it — a production-n gate at smoke n is the
    # #1345 gate-calibration class (gotchas.md).
    m = re.search(r"run_p3_null\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert m, "run_p3_null() block not found"
    assert "--expect-n-cells" in m.group(1)
    assert "ISSUE2061_EXPECT_N_CELLS:-64" in m.group(1)
    s = re.search(r"run_smoke\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert s, "run_smoke() block not found"
    assert "--expect-n-cells" not in s.group(1)


def test_p3_fanout_pins_cvd_per_worker():
    m = re.search(r"run_p3_fanout\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert m, "run_p3_fanout() block not found"
    body = m.group(1)
    assert 'CUDA_VISIBLE_DEVICES="$g"' in body, "fan-out workers must be CVD-pinned"
    assert "--skip-global" in body, "fan-out workers must not race on GLOBAL_L29.json"


# ---------------------------------------------------------------------------
# Upload/staging wiring per plan-declared artifact class (review C4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("script", "needles"),
    [
        ("issue2061_sae_encode.py", ['upload_dir(args.output_dir, "sae-encoded")']),
        (
            "issue2061_fit_per_feature.py",
            [
                'upload_dir(args.output_dir, "per-feature-r2")',
                'stage_dir("sae-encoded"',
                "stage_turnstore(",
                "reap_turnstore(",
            ],
        ),
        (
            "issue2061_null.py",
            [
                'upload_dir(args.output_dir, "null")',
                'stage_dir("per-feature-r2"',
                'stage_dir("sae-encoded"',
                "stage_turnstore(",
                "reap_turnstore(",
            ],
        ),
        ("issue2061_fitness.py", ['upload_dir(args.output_dir, "fitness")']),
        (
            "issue2061_figures.py",
            ['stage_dir("per-feature-r2"', 'stage_dir("null"', 'stage_dir("fitness"'],
        ),
    ],
)
def test_cross_machine_wiring_present(script, needles):
    text = (SCRIPTS / script).read_text()
    for needle in needles:
        assert needle in text, f"{script} missing plan §9 off_pod_phases wiring: {needle}"


# ---------------------------------------------------------------------------
# Realized #1336 turnstore naming (unit-E live probe, 2026-08-05): the store
# carries `turnstore_[v2_]<stage>_<render>_<corpus>` with the 5th ladder
# stage realized as `rlvr_long` — a naive split("_", 2) mis-buckets 35/55
# names (stage "v2", render "long", corpus "chat_lmsys23k").
# ---------------------------------------------------------------------------
def test_parse_turnstore_name_realized_vocabulary():
    cases = {
        # v1 capture generation
        "turnstore_base_chat_gsm8k_test1319": ("base", "chat", "gsm8k_test1319"),
        "turnstore_rlvr_long_chat_gsm8k_train5k": ("longer-rlvr", "chat", "gsm8k_train5k"),
        "turnstore_rlvr_long_naturalistic_lmsys5k": ("longer-rlvr", "naturalistic", "lmsys5k"),
        "turnstore_rlvr_naturalistic_lmsys5k": ("rlvr", "naturalistic", "lmsys5k"),
        # v2 capture generation
        "turnstore_v2_base_chat_lmsys23k": ("base", "chat", "lmsys23k"),
        "turnstore_v2_rlvr_long_chat_gsm8k_train_full": (
            "longer-rlvr",
            "chat",
            "gsm8k_train_full",
        ),
        "turnstore_v2_sft_naturalistic_lmsys23k": ("sft", "naturalistic", "lmsys23k"),
        "turnstore_v2_dpo_chat_if11k": ("dpo", "chat", "if11k"),
    }
    for name, expected in cases.items():
        assert enc.parse_turnstore_name(name) == expected, name
    # Non-turnstore / unknown-vocabulary names return None (caller WARNs).
    assert enc.parse_turnstore_name("not_a_turnstore") is None
    assert enc.parse_turnstore_name("turnstore_mystery_chat_x") is None
    assert enc.parse_turnstore_name("turnstore_base_render_x") is None
    assert enc.parse_turnstore_name("turnstore_base_chat_") is None
    # Canonical stage tokens are the pipeline's own (underscore-free), so the
    # LEFT-parsing encoded/r2 stem round-trip survives the 5th stage.
    import issue2061_turnstore as ts

    stem = "longer-rlvr_chat_lmsys23k_answer_L29"
    assert ts.parse_encoded_stem(stem, "answer", 29) == ("longer-rlvr", "chat", "lmsys23k")


def test_turnstore_enumeration_collision_and_unparsed_warn(monkeypatch, capsys):
    """The v1/v2 family-collision assert fires (fail-loud, never a silent
    overwrite) and an unknown-vocabulary name WARNs instead of silently
    vanishing (the M2 vanishing-cell class)."""

    class E:
        def __init__(self, p):
            self.path = p

    def fake_retry_collide(fn, *, what=""):
        return [E("pre/turnstore_base_chat_x"), E("pre/turnstore_v2_base_chat_x")]

    monkeypatch.setattr(enc, "retry_transient", fake_retry_collide)
    with pytest.raises(ValueError, match="Ambiguous turnstore cell"):
        enc._stage_render_corpus_turnstores()

    def fake_retry_unparsed(fn, *, what=""):
        return [E("pre/turnstore_base_chat_x"), E("pre/turnstore_mystery_chat_x")]

    monkeypatch.setattr(enc, "retry_transient", fake_retry_unparsed)
    stores = enc._stage_render_corpus_turnstores()
    out = capsys.readouterr().out
    assert [t["tree_path"] for t in stores] == ["pre/turnstore_base_chat_x"]
    assert "WARN" in out and "turnstore_mystery_chat_x" in out


def test_hub_prefix_mapping_matches_plan():
    assert hio.hub_prefix("sae-encoded") == f"{hio.HF_PREFIX}/sae_encoded"
    assert hio.hub_prefix("per-feature-r2") == f"{hio.HF_PREFIX}/analysis_tensors/per_feature_r2"
    assert hio.hub_prefix("null") == f"{hio.HF_PREFIX}/analysis_tensors/null"
    assert hio.hub_prefix("fitness") == f"{hio.HF_PREFIX}/analysis_tensors/fitness"
    with pytest.raises(KeyError):
        hio.hub_prefix("nope")


# ---------------------------------------------------------------------------
# stage_turnstore / reap_turnstore (real bodies; Hub boundary faked
# signature-conformantly — code-style.md § one production-body test)
# ---------------------------------------------------------------------------
def test_stage_turnstore_maps_shards_to_consumer_layout(tmp_path, monkeypatch):
    from explore_persona_space.orchestrate import hub as hub_mod

    def fake_resolve(revision):
        """Signature mirror of hio._resolve_data_repo_revision (network boundary)."""
        return revision or "deadbeefcafe"

    def fake_resolve_tree(stage, render, corpus, revision=None):
        """Signature mirror of enc.resolve_turnstore_tree (network boundary) —
        returns the REALIZED v1-family tree name for the fixture cell."""
        assert revision == "deadbeefcafe"
        return f"{enc.BANKED_PREFIX}/turnstore_{stage}_{render}_{corpus}"

    def fake_hub_shard_files(tree_path: str, revision: str | None = None) -> list[str]:
        """Signature mirror of issue2061_sae_encode.hub_shard_files (network)."""
        assert revision == "deadbeefcafe"
        return [f"{tree_path}/x_shard000.pt", f"{tree_path}/x_shard001.pt"]

    seen: list[tuple] = []
    real_stage = hub_mod.stage_hub_file

    def fake_stage_hub_file(repo_id, path_in_repo, target, **kwargs):
        # autospec-style shape check against the REAL helper, then a local
        # write standing in for the download.
        import inspect

        inspect.signature(real_stage).bind(repo_id, path_in_repo, target, **kwargs)
        seen.append((repo_id, path_in_repo, str(target), kwargs.get("revision")))
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).touch()
        return Path(target)

    monkeypatch.setattr(hio, "_resolve_data_repo_revision", fake_resolve)
    monkeypatch.setattr(enc, "resolve_turnstore_tree", fake_resolve_tree)
    monkeypatch.setattr(enc, "hub_shard_files", fake_hub_shard_files)
    monkeypatch.setattr(hub_mod, "stage_hub_file", fake_stage_hub_file)

    dest = hio.stage_turnstore("base", "chat", "gsm8k_test1319", tmp_path)
    # Consumer layout: FLAT shard basenames under turnstore_<stage>_<render>_<corpus>
    # (issue2061_turnstore.enumerate_shards's own open shape — never a
    # repo-path mirror; the #1774 mirror-root class).
    assert dest == tmp_path / "turnstore_base_chat_gsm8k_test1319"
    assert sorted(p.name for p in dest.iterdir()) == ["x_shard000.pt", "x_shard001.pt"]
    assert all(rev == "deadbeefcafe" for (_, _, _, rev) in seen)
    assert all(repo == hio.DATA_REPO for (repo, _, _, _) in seen)


def test_reap_turnstore_deletes_only_the_staged_dir(tmp_path):
    d = tmp_path / "turnstore_base_chat_c1"
    d.mkdir(parents=True)
    (d / "x_shard000.pt").touch()
    other = tmp_path / "turnstore_sft_chat_c1"
    other.mkdir()
    hio.reap_turnstore(tmp_path, "base", "chat", "c1")
    assert not d.exists() and other.exists()
    hio.reap_turnstore(tmp_path, "base", "chat", "c1")  # idempotent no-op


def test_p3_combos_orders_largest_first(tmp_path):
    r2 = tmp_path / "r2"
    r2.mkdir()
    for stage in ["base", "sft"]:
        for corpus in ["lmsys23k", "gsm8k_test1319"]:
            for arm in ["prefix", "context"]:
                (r2 / f"{stage}_chat_{corpus}_{arm}_L29.jsonl").touch()
    enc_dir = tmp_path / "enc"
    enc_dir.mkdir()
    (enc_dir / "base_chat_lmsys23k_answer_L29.pt").write_bytes(b"x" * 1000)
    (enc_dir / "base_chat_gsm8k_test1319_answer_L29.pt").write_bytes(b"x" * 10)
    combos = hio.p3_combos(r2, enc_dir)
    assert combos == [("chat", "lmsys23k"), ("chat", "gsm8k_test1319")]
    with pytest.raises(FileNotFoundError):
        hio.p3_combos(tmp_path / "empty-nope")


def test_upload_dir_refuses_empty(tmp_path, monkeypatch):
    # Fail-loud floor: an upload leg that finds nothing must raise, never
    # post a clean exit over a missing artifact class.
    import explore_persona_space.orchestrate.upload_sharded as us

    empty = tmp_path / "empty"
    empty.mkdir()
    result = us.ShardUploadResult(repo_id=hio.DATA_REPO)
    with (
        mock.patch.object(us, "upload_dir_sharded", autospec=True, return_value=result),
        pytest.raises(RuntimeError, match="no files"),
    ):
        hio.upload_dir(empty, "fitness")
