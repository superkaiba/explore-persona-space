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
    # m2 (round 3) + v7 grid (round 4): the PRODUCTION aggregation pass
    # carries the fail-loud 56-cell guard (registered statistic, plan §Design
    # v7 — 4 stage-pairs x 7 v2 combos x 2 arms), while the smoke chain must
    # NEVER inherit it — a production-n gate at smoke n is the #1345
    # gate-calibration class (gotchas.md).
    m = re.search(r"run_p3_null\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert m, "run_p3_null() block not found"
    assert "--expect-n-cells" in m.group(1)
    assert "ISSUE2061_EXPECT_N_CELLS:-56" in m.group(1)
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
    """The same-cell collision assert stays fail-loud (defensive — canonical
    identity ambiguity, never a silent overwrite) and an unknown-vocabulary
    IN-generation name WARNs instead of silently vanishing (the M2
    vanishing-cell class)."""

    class E:
        def __init__(self, p):
            self.path = p

    # Within-generation same-cell collision: injective naming makes this
    # unreachable via the real parser, so force it through the parse seam
    # (parse_turnstore_name's real body is pinned by
    # test_parse_turnstore_name_realized_vocabulary above).
    def fake_retry_collide(fn, *, what=""):
        return [E("pre/turnstore_v2_base_chat_x"), E("pre/turnstore_v2_base_chat_y")]

    monkeypatch.setattr(enc, "retry_transient", fake_retry_collide)
    monkeypatch.setattr(enc, "parse_turnstore_name", lambda name: ("base", "chat", "x"))
    with pytest.raises(ValueError, match="Ambiguous turnstore cell"):
        enc._stage_render_corpus_turnstores()
    monkeypatch.undo()

    def fake_retry_unparsed(fn, *, what=""):
        return [E("pre/turnstore_v2_base_chat_x"), E("pre/turnstore_v2_mystery_chat_x")]

    monkeypatch.setattr(enc, "retry_transient", fake_retry_unparsed)
    stores = enc._stage_render_corpus_turnstores()
    out = capsys.readouterr().out
    assert [t["tree_path"] for t in stores] == ["pre/turnstore_v2_base_chat_x"]
    assert "WARN" in out and "turnstore_v2_mystery_chat_x" in out


# ---------------------------------------------------------------------------
# v7 generation pin (round-4 delta 1): the enumeration consumes ONLY the
# REGISTERED v2 capture generation — plan acceptance: exactly 35 stores /
# 7 (render, corpus) combos under the pin (the landed code would otherwise
# encode the 11-combo v1+v2 UNION).
# ---------------------------------------------------------------------------
# The REALIZED store shape (read-only inventory probe, epm:progress
# 2026-08-05T21:48:40Z): 55 dirs = 5 stages x (v1: 4 combos, v2: 7 combos),
# disjoint corpus stems, only the lmsys stems carry both renders.
_REALIZED_STORE_STAGES = ["base", "sft", "dpo", "rlvr", "rlvr_long"]
_REALIZED_V1_COMBOS = [
    ("chat", "gsm8k_test1319"),
    ("chat", "gsm8k_train5k"),
    ("chat", "lmsys5k"),
    ("naturalistic", "lmsys5k"),
]
_REALIZED_V2_COMBOS = [
    ("chat", "gsm8k_train_full"),
    ("chat", "if11k"),
    ("chat", "lmsys23k"),
    ("naturalistic", "lmsys23k"),
    ("chat", "math7500"),
    ("chat", "sft11k"),
    ("chat", "uf11k"),
]


def _realized_store_listing():
    names = []
    for s in _REALIZED_STORE_STAGES:
        for render, corpus in _REALIZED_V1_COMBOS:
            names.append(f"pre/turnstore_{s}_{render}_{corpus}")
        for render, corpus in _REALIZED_V2_COMBOS:
            names.append(f"pre/turnstore_v2_{s}_{render}_{corpus}")
    return names


def test_generation_pin_acceptance_35_stores_7_combos(monkeypatch, capsys):
    class E:
        def __init__(self, p):
            self.path = p

    listing = _realized_store_listing()
    assert len(listing) == 55  # fixture mirrors the realized store

    monkeypatch.setattr(enc, "retry_transient", lambda fn, *, what="": [E(p) for p in listing])

    # Default pin = REGISTERED v2 generation: exactly 35 stores / 7 combos.
    assert enc.REGISTERED_GENERATION == "v2"
    stores = enc._stage_render_corpus_turnstores()
    assert len(stores) == 35
    combos = sorted({(t["render"], t["corpus"]) for t in stores})
    assert combos == sorted(_REALIZED_V2_COMBOS)
    # No v1 stem leaks through the pin; every tree path is a v2 store.
    assert all("/turnstore_v2_" in t["tree_path"] for t in stores)
    # The 20 other-generation dirs are skipped LOUDLY, never silently.
    out = capsys.readouterr().out
    assert "[generation-pin] 20 turnstore(s)" in out

    # --generation v1 preserves the parked lower-n robustness arm: 20/4.
    stores_v1 = enc._stage_render_corpus_turnstores(generation="v1")
    assert len(stores_v1) == 20
    assert sorted({(t["render"], t["corpus"]) for t in stores_v1}) == sorted(_REALIZED_V1_COMBOS)
    assert all("/turnstore_v2_" not in t["tree_path"] for t in stores_v1)

    # Unknown generation fails loud.
    with pytest.raises(ValueError, match="Unknown capture generation"):
        enc._stage_render_corpus_turnstores(generation="v3")


def test_resolve_turnstore_tree_honors_generation_pin(monkeypatch):
    class E:
        def __init__(self, p):
            self.path = p

    monkeypatch.setattr(
        enc,
        "retry_transient",
        lambda fn, *, what="": [E(p) for p in _realized_store_listing()],
    )
    # A v2 cell resolves to the REALIZED v2 tree path under the default pin.
    tree = enc.resolve_turnstore_tree("base", "chat", "lmsys23k")
    assert tree == "pre/turnstore_v2_base_chat_lmsys23k"
    # The 5th ladder stage resolves through the rlvr_long store token.
    tree = enc.resolve_turnstore_tree("longer-rlvr", "chat", "gsm8k_train_full")
    assert tree == "pre/turnstore_v2_rlvr_long_chat_gsm8k_train_full"
    # A v1-only corpus does NOT resolve under the registered pin...
    with pytest.raises(FileNotFoundError, match="generation 'v2'"):
        enc.resolve_turnstore_tree("base", "chat", "lmsys5k")
    # ...but does under the explicit v1 override (parked robustness arm).
    tree = enc.resolve_turnstore_tree("base", "chat", "lmsys5k", generation="v1")
    assert tree == "pre/turnstore_base_chat_lmsys5k"


def test_turnstore_generation_helper():
    assert enc.turnstore_generation("turnstore_base_chat_lmsys5k") == "v1"
    assert enc.turnstore_generation("turnstore_v2_base_chat_lmsys23k") == "v2"
    assert enc.turnstore_generation("not_a_turnstore") is None
    # Stage token containing "v2" mid-name never mis-buckets (prefix-anchored).
    assert enc.turnstore_generation("turnstore_rlvr_long_chat_gsm8k_train5k") == "v1"


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
        """Signature mirror of hio.resolve_data_repo_revision (network boundary)."""
        return revision or "deadbeefcafe"

    def fake_resolve_tree(stage, render, corpus, revision=None, generation="v2"):
        """Signature mirror of enc.resolve_turnstore_tree (network boundary) —
        returns the REALIZED tree name for the fixture cell (v2 naming for the
        registered generation, prefix-less for a wave-1 concat source)."""
        assert revision == "deadbeefcafe"
        tag = "turnstore_v2" if generation == "v2" else "turnstore"
        return f"{enc.BANKED_PREFIX}/{tag}_{stage}_{render}_{corpus}"

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

    monkeypatch.setattr(hio, "resolve_data_repo_revision", fake_resolve)
    monkeypatch.setattr(enc, "resolve_turnstore_tree", fake_resolve_tree)
    monkeypatch.setattr(enc, "hub_shard_files", fake_hub_shard_files)
    monkeypatch.setattr(hub_mod, "stage_hub_file", fake_stage_hub_file)

    dest = hio.stage_turnstore("base", "chat", "gsm8k_test1319", tmp_path)
    # Consumer layout: FLAT shard basenames under turnstore_<stage>_<render>_<corpus>
    # (issue2061_turnstore.enumerate_shards's own open shape — never a
    # repo-path mirror; the #1774 mirror-root class). Sidecars ride along
    # (v13 a1-bis: the loader's per-shard asserts read them).
    assert dest == tmp_path / "turnstore_base_chat_gsm8k_test1319"
    assert sorted(p.name for p in dest.iterdir()) == [
        "x_shard000.json",
        "x_shard000.pt",
        "x_shard001.json",
        "x_shard001.pt",
    ]
    assert all(rev == "deadbeefcafe" for (_, _, _, rev) in seen)
    assert all(repo == hio.DATA_REPO for (repo, _, _, _) in seen)

    # CONCAT cell (plan v11 delta a1/a2): the extended corpora stage BOTH
    # stores — the wave-1 source AND the v2 extension — into the consumer
    # layout; reap_turnstore reaps both.
    dest2 = hio.stage_turnstore("base", "chat", "lmsys23k", tmp_path)
    assert dest2 == tmp_path / "turnstore_base_chat_lmsys23k"
    src_dir = tmp_path / "turnstore_base_chat_lmsys5k"
    assert src_dir.is_dir() and dest2.is_dir()
    staged_trees = {p.split("/")[-2] for (_, p, _, _) in seen}
    assert "turnstore_base_chat_lmsys5k" in staged_trees  # v1-family source tree
    assert "turnstore_v2_base_chat_lmsys23k" in staged_trees  # v2 extension tree
    hio.reap_turnstore(tmp_path, "base", "chat", "lmsys23k")
    assert not src_dir.exists() and not dest2.exists()


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


def test_parity_gate_call_sites_install_sparsify_pinned_and_fail_loud():
    """Both loader-parity-gate call sites must provision `sparsify` first.

    The gate is the FIRST action of `--smoke-only` and `--smoke-then-encode`
    and HARD-FAILS without `sparsify` (issue2061_sae_encode.py:364-372).
    sparsify is a deliberate one-off, so neither uv.lock nor
    bootstrap_pod.sh carries it: without this provisioning a fresh pod dies
    at the gate before doing any encode work, burning a provision cycle.
    """
    # The helper exists, pins a version, and is env-overridable.
    m = re.search(r"ensure_sparsify\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert m, "ensure_sparsify() helper missing"
    body = m.group(1)
    assert "uv pip install" in body, "helper does not install sparsify"
    assert "ISSUE2061_SPARSIFY_VERSION" in body, "pin is not env-overridable"

    # DIST NAME, checked exactly. The import name is `sparsify` but the
    # distribution is `eai-sparsify` (EleutherAI). The bare PyPI name
    # `sparsify` is Neural Magic's DEPRECATED sparsification UI: it does not
    # provide `SparseCoder`, and `sparsify==1.3.3` does not even resolve
    # (that project ships 1.3.0 then 1.4.0). A substring check for
    # "sparsify==" cannot tell the two apart -- `eai-sparsify==` contains it
    # -- which is exactly how the wrong dist name shipped once.
    dist = re.search(r'uv pip install "([A-Za-z0-9._-]+)==', body)
    assert dist, "could not parse the pinned distribution name"
    assert dist.group(1) == "eai-sparsify", (
        f"parity reference must install the EleutherAI distribution "
        f"'eai-sparsify', got {dist.group(1)!r}"
    )

    # The dist-name -> import-name mapping is self-checked in the helper, so
    # a wrong distribution fails at install time with a legible message
    # instead of inside the parity gate.
    assert "from sparsify import SparseCoder" in body, (
        "helper does not verify the installed distribution provides SparseCoder"
    )
    # Fail-loud: never swallow an install failure. A missing reference
    # implementation means the parity gate cannot be honestly run.
    # Check EXECUTABLE lines only -- the helper's own comment names the
    # banned swallow forms in prose, which a naive substring scan matches.
    code = "\n".join(ln for ln in body.split("\n") if not ln.strip().startswith("#"))
    assert "|| true" not in code and "|| echo" not in code, "sparsify install must be fail-loud"

    # Call site 1: the production P1 leg (--smoke-then-encode).
    p1 = re.search(r"run_p1_encode\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert p1, "run_p1_encode() missing"
    assert "ensure_sparsify" in p1.group(1), (
        "run_p1_encode does not provision sparsify before the parity gate"
    )

    # Call site 2: the --smoke-only full chain, which runs the gate directly.
    smoke = re.search(r"run_smoke\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert smoke, "run_smoke() missing"
    sbody = smoke.group(1)
    assert "ensure_sparsify" in sbody, (
        "run_smoke does not provision sparsify before the parity gate"
    )
    # Ordering is load-bearing: install must precede the gate invocation.
    assert sbody.index("ensure_sparsify") < sbody.index("--smoke-only"), (
        "sparsify install must come BEFORE the --smoke-only parity gate"
    )


# Capture-generation stem partition (plan v7 amendment). The two generations'
# corpus stems are DISJOINT, which is why a v1 stem in a registered-generation
# consumer fails loud rather than silently mis-bucketing.
V1_ONLY_STEMS = frozenset({"gsm8k_test1319", "gsm8k_train5k", "lmsys5k"})
V2_STEMS = frozenset({"gsm8k_train_full", "if11k", "lmsys23k", "math7500", "sft11k", "uf11k"})


def test_smoke_default_cells_are_registered_stems_covering_both_grains():
    """The smoke defaults must resolve under the REGISTERED generation AND
    cover BOTH consumption grains.

    Two stacked regression pins:

    1. (round 5) the smoke default was `gsm8k_test1319`, a v1-ONLY stem;
       once the v7 grid pinned the registered generation to v2,
       `resolve_turnstore_tree()` FileNotFoundError'd at first resolution --
       on the pod, after provision + bootstrap.
    2. (crash-fix 2026-08-06) the round-5 SINGLE-cell smoke selected
       `gsm8k_train_full` -- a CONCAT combo -- so it exercised only the
       concat resolution path of the TWO-grain resolver, and P1 production
       died on its FIRST plain-v2 cell (if11k, the majority grain: 4 of 6
       corpora are plain-v2). A single-cell smoke over a multi-path resolver
       tests only the path it selects; the default must carry at least one
       corpus per grain.
    """
    import issue2061_turnstore as ts

    m = re.search(r'CPS="\$\{ISSUE2061_SMOKE_CORPORA:-([a-z0-9_ ]+)\}"', DISPATCH)
    assert m, "could not find the smoke default corpora in the dispatcher"
    corpora = m.group(1).split()
    assert corpora, "empty smoke corpora default"

    for corpus in corpora:
        assert corpus not in V1_ONLY_STEMS, (
            f"smoke default corpus {corpus!r} is a v1-ONLY stem; it cannot resolve "
            f"under the registered v2 generation and the smoke will die at its "
            f"first turnstore resolution"
        )
        assert corpus in V2_STEMS, (
            f"smoke default corpus {corpus!r} is not a known registered (v2) stem"
        )

    # BOTH grains: >=1 concat corpus (a V2_CONCAT_SOURCES key) AND >=1
    # plain-v2 corpus (not a key) -- pin 2 above.
    concat = [c for c in corpora if c in ts.V2_CONCAT_SOURCES]
    plain = [c for c in corpora if c not in ts.V2_CONCAT_SOURCES]
    assert concat, f"smoke corpora {corpora} carry NO concat-grain cell"
    assert plain, f"smoke corpora {corpora} carry NO plain-v2-grain cell"

    # The render must be one the chosen stems actually carry. Only lmsys23k
    # carries `naturalistic`; every other v2 stem is chat-only.
    rm = re.search(r'RD="\$\{ISSUE2061_SMOKE_RENDER:-([a-z]+)\}"', DISPATCH)
    assert rm, "could not find the smoke default render in the dispatcher"
    render = rm.group(1)
    if render == "naturalistic":
        assert corpora == ["lmsys23k"], (
            f"render 'naturalistic' is only carried by lmsys23k, not {corpora!r}"
        )
    else:
        assert render == "chat", f"unexpected smoke render {render!r}"


def test_smoke_p1_encode_iterates_stages_and_corpora():
    """The smoke's P1 encode leg loops stages x corpora (both grains reach the
    production entrypoint), and the P0 gate receives one --corpus per grain."""
    m = re.search(r"run_smoke\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert m, "run_smoke() block not found"
    body = m.group(1)
    p1 = body[body.index("smoke_p1_encode") :]
    assert re.search(r'for st in "\$SA" "\$SB"; do\s*\n\s*for cp in \$CPS; do', p1), (
        "smoke P1 encode must nest stages x corpora so BOTH grains are encoded"
    )
    assert '--corpus "$cp"' in p1
    # The gate builds one --corpus flag per smoke corpus into ONE manifest.
    gate = body[body.index("smoke_p0_grain_gate") : body.index("smoke_p1_parity")]
    assert re.search(r"for cp in \$CPS; do gate_args\+=\(--corpus \"\$cp\"\); done", gate), (
        "smoke P0 gate must receive one --corpus per smoke corpus"
    )


# ---------------------------------------------------------------------------
# P0 grain gate wiring (plan v11 delta (d)) + concat-grain smoke acceptance
# (delta (e))
# ---------------------------------------------------------------------------
def test_p0_grain_gate_wired_before_p1_in_all_mode():
    m = re.search(r"run_p0_grain_gate\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert m, "run_p0_grain_gate() block not found"
    body = m.group(1)
    assert "issue2061_grain_gate.py" in body
    assert "--expect-n-cells 35" in body, "production gate pins the 35-cell grid"
    assert "grain_gate/grain_manifest.json" in body
    # MODE=all runs P0 BEFORE P1 (the gate exists to stop the dispatch pre-GPU).
    all_block = DISPATCH[DISPATCH.index('MODE" == "all"') :]
    assert all_block.index("run_p0_grain_gate") < all_block.index("run_p1_encode")
    # And the per-machine form exposes it.
    assert re.search(r"p0\)\s*run_p0_grain_gate", DISPATCH)


def test_smoke_runs_grain_gate_and_acceptance():
    m = re.search(r"run_smoke\(\) \{(.*?)\n\}", DISPATCH, flags=re.S)
    assert m, "run_smoke() block not found"
    body = m.group(1)
    # Smoke leg of the SAME production gate, filtered to the smoke cells,
    # manifest under the smoke root (never the canonical eval tree).
    assert "issue2061_grain_gate.py" in body
    assert '"$SMOKE_ROOT/grain_gate/grain_manifest.json"' in body
    # Acceptance (delta (e)): NO [dof-cap] line at the concat grain + the
    # r2-vs-manifest checks (convention=primal, v13 grid, realized n).
    assert "smoke_accept" in body
    assert re.search(r"grep -q .\\\[dof-cap\\\]", body), "the [dof-cap] acceptance grep"
    assert "--accept-r2-dir" in body
    # P2's output is captured for the grep (tee to the smoke log).
    assert "p2_fit.log" in body


def test_smoke_sentinel_phases_list_carries_p0_and_accept():
    m = re.search(r'finish "epm:smoke-result" \\\n\s+"(.*)"', DISPATCH)
    assert m, "smoke finish note not found"
    note = m.group(1)
    assert "p0_grain_gate" in note and "smoke_accept" in note
