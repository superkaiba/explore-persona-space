"""#2479 crash-fix r8 regression pins: single-source gen-feasibility export + sampler binding.

Pins the P0 gen-smoke gate failure of 2026-08-23 (both inserted-variant cells
died at ``restrict_pool_to_manifest``: 122 registered sample_conv_ids absent
from the eligible pool, pool=4089): the Step-1 sampler
(``scripts/issue2479_panel_sample.py``) drew from set intersections WITHOUT the
gen script's own eligibility filters (``answer_too_short`` at pool join;
``answer_over_budget`` for op_companion=False in ``_filter_pool_feasible``).

Pinned behaviors:

(a) ``_filter_pool_feasible`` regime asymmetry — an over-budget answer is
    dropped under ``op_companion=False`` (verbatim-embed budget) and KEPT under
    ``op_companion=True`` (free generation; no fixed answer);
(b) the sampler with a synthetic ``--eligible-ids`` export excludes
    non-eligible ids from the sample, records the export (path + sha256 +
    embedded provenance) in the manifest, and hard-fails when the restricted
    intersection cannot fill ``--n-sample`` / is empty / the flag is omitted /
    the export lacks its provenance block;
(c) the ``--emit-eligible-ids`` JSON schema (keys, str ids, per-regime counts,
    provenance block), produced by the REAL emit body over tmp staged inputs;
    plus the emit mode's standalone-flag guard firing BEFORE the variant asserts;
(d) concern round r9: the K-row panel-invariance gate (probes the K min-margin
    kept rows per regime; per-regime ``binding_prompt_sha256`` template hash),
    the fail-loud emit git provenance (dirty checkout refused unless
    ``--allow-dirty-emit``; full 40-hex HEAD required), the fail-loud
    production tokenizer-revision pin (null never written), and the sampler's
    panel-sha binding (live ``panel.json`` sha must equal the export's recorded
    ``provenance.panel_invariance.panel_sha256`` — a panel-only edit forces a
    re-emit).

Hermetic: tmp-path staged files; fakes ONLY at the hub/tokenizer boundary (the
fake tokenizer mirrors the two consumed methods' signatures; hf downloads are
monkeypatched to tmp writers). Synthetic text only. No network, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1345_gen_stories_paired as gp  # noqa: E402
import issue2479_panel_sample as ps  # noqa: E402


class FakeTok:
    """Signature-conformant fake at the tokenizer boundary: 1 token per '§'.

    Mirrors the two methods ``_filter_pool_feasible`` / ``build_paired_prompt``
    consume: ``__call__(text, add_special_tokens=False) -> {"input_ids": [...]}``
    and ``apply_chat_template(messages, tokenize=False, add_generation_prompt=True)``.
    """

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [0] * text.count("§")}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return "\n".join(m["content"] for m in messages)


# --- (a) _filter_pool_feasible regime asymmetry ------------------------------


def test_filter_pool_feasible_regime_asymmetry():
    """Over-budget answer: dropped under op_companion=False, kept under True."""
    tok = FakeTok()
    over = {"conv_id": "over", "question": "q", "answer": "§" * (gp.ANSWER_TOKEN_BUDGET + 1)}
    ok = {"conv_id": "ok", "question": "q", "answer": "§" * 10}
    pool = [over, ok]
    kept_paired, drops_paired = gp._filter_pool_feasible(pool, tok, op_companion=False)
    assert [r["conv_id"] for r in kept_paired] == ["ok"]
    assert drops_paired == {"prompt_over_budget": 0, "answer_over_budget": 1}
    kept_op, drops_op = gp._filter_pool_feasible(pool, tok, op_companion=True)
    assert [r["conv_id"] for r in kept_op] == ["over", "ok"]
    assert drops_op == {"prompt_over_budget": 0, "answer_over_budget": 0}


# --- (c) --emit-eligible-ids export -------------------------------------------


def _write_panel(tmp_path: Path, desc: str = "a benign panel description") -> Path:
    """Tmp #2479 character panel (the roster the invariance gate renders under)."""
    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(
            [
                {"name": "x", "display_name": "X", "desc": desc},
                {"name": "y", "display_name": "Y", "desc": "another benign desc"},
            ]
        )
    )
    return panel


STUB_GIT_SHA = "e" * 40


def _stub_git(monkeypatch) -> None:
    """Hermetic stand-in for the r9 fail-loud emit git-provenance helper.

    Keeps the emit tests independent of the LIVE checkout's dirty state; the
    real helper body is covered by test_emit_git_provenance_* below.
    """
    monkeypatch.setattr(
        gp,
        "_emit_git_provenance",
        lambda *, allow_dirty=False: {
            "git_commit": STUB_GIT_SHA,
            "git_dirty": False,
            "git_argv0_state": "tracked",
        },
    )


def _stage_emit_inputs(
    monkeypatch, tmp_path: Path, *, stub_git: bool = True, extra_rows: list[dict] | None = None
) -> Path:
    """Tmp matched allowlist + track_s rows; stage_pinned_file faked to tmp."""
    if stub_git:
        _stub_git(monkeypatch)
    matched_dir = tmp_path / "matched"
    matched_dir.mkdir()
    extra = list(extra_rows or [])
    (matched_dir / "matched_subsets_parent.json").write_text(
        json.dumps({"shared_r1r2_convs": ["c1", "c2", "c3", "c4", *(r["conv_id"] for r in extra)]})
    )
    track = tmp_path / "track_s.jsonl"
    rows = [
        # eligible in BOTH regimes
        {"conv_id": "c1", "prompt": "q1", "response": "x" * 25},
        # answer_too_short (< ANSWER_CHAR_MIN) — dropped at pool join
        {"conv_id": "c2", "prompt": "q2", "response": "short"},
        # answer over token budget — paired-dropped, op-kept
        {
            "conv_id": "c3",
            "prompt": "q3",
            "response": "y" * 20 + "§" * (gp.ANSWER_TOKEN_BUDGET + 1),
        },
        # not in the shared allowlist — never joins
        {"conv_id": "unshared", "prompt": "q", "response": "z" * 25},
        *extra,
    ]
    track.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def fake_stage(path_in_repo, dest_dir, revision=gp.c.PIN_REV):
        assert path_in_repo == gp.c.PARENT_TRACK_S_JSONL
        return track

    monkeypatch.setattr(gp.c, "stage_pinned_file", fake_stage)
    return matched_dir


def test_emit_eligible_ids_schema(monkeypatch, tmp_path):
    matched_dir = _stage_emit_inputs(monkeypatch, tmp_path)
    panel = _write_panel(tmp_path)
    out = tmp_path / "sub" / "eligible_conv_ids.json"
    gp.emit_eligible_ids(out, matched_dir, tmp_path / "dl", tokenizer=FakeTok(), panel_path=panel)
    payload = json.loads(out.read_text())
    for key in ("eligible_paired", "eligible_op", "counts", "provenance"):
        assert key in payload, key
    assert payload["eligible_paired"] == ["c1"]
    assert payload["eligible_op"] == ["c1", "c3"]
    assert all(isinstance(x, str) for x in payload["eligible_paired"] + payload["eligible_op"])
    counts = payload["counts"]
    assert isinstance(counts, dict)
    assert counts["pool"]["answer_too_short"] == 1
    assert counts["pool"]["joined"] == 3
    assert counts["paired_drops"] == {"prompt_over_budget": 0, "answer_over_budget": 1}
    assert counts["op_drops"] == {"prompt_over_budget": 0, "answer_over_budget": 0}
    assert counts["n_pool"] == 2
    assert counts["n_eligible_paired"] == 1
    assert counts["n_eligible_op"] == 2
    assert counts["n_eligible_both"] == 1
    prov = payload["provenance"]
    for key in ("git_commit", "timestamp", "mode", "tokenizer", "inputs", "budgets"):
        assert key in prov, key
    assert prov["mode"] == "emit-eligible-ids"
    assert prov["budgets"]["answer_token_budget"] == gp.ANSWER_TOKEN_BUDGET
    assert prov["budgets"]["prompt_token_budget"] == gp.g.PROMPT_TOKEN_BUDGET
    assert prov["inputs"]["track_s_revision"] == gp.c.PIN_REV
    # concern-round r8 residue: realized emit-config identity + input pins
    assert prov["emit_config"] == {
        "character_name": gp.c.STORY_CHARACTER_NAME,
        "persona_desc": gp.PERSONA_DESC,
        "variant": gp.c.VARIANT,
    }
    # r9: a null tokenizer_revision is never written; the injected branch
    # records the explicit sentinel string.
    assert prov["tokenizer_revision"] == "injected"
    assert prov["tokenizer_revision_source"] == "injected"
    # r9: the recorded git_commit is the fail-loud helper's verified emit-
    # checkout HEAD — it OVERRIDES c.metadata's cwd-based value.
    assert prov["git_commit"] == STUB_GIT_SHA
    assert prov["git_dirty"] is False
    matched = prov["inputs"]["matched_allowlist"]
    assert matched["path"].endswith("matched_subsets_parent.json")
    expected_sha = hashlib.sha256(
        (matched_dir / "matched_subsets_parent.json").read_bytes()
    ).hexdigest()
    assert matched["sha256"] == expected_sha
    assert matched["pinned_revision"] == gp.c.REUSE_REV
    # panel-invariance record: benign panel -> delta 0; FakeTok counts no tokens
    # in c1's prompt, so both regimes' min margin == the full prompt budget.
    pi = prov["panel_invariance"]
    assert pi["panel_sha256"] == hashlib.sha256(panel.read_bytes()).hexdigest()
    assert pi["n_panel_configs"] == 2
    assert pi["slack_tokens"] == gp.PANEL_MARGIN_SLACK_TOKENS
    for regime in ("paired", "op"):
        reg = pi["regimes"][regime]
        assert reg["min_margin_tokens"] == gp.g.PROMPT_TOKEN_BUDGET
        assert reg["max_panel_delta_tokens"] == 0
        assert reg["min_margin_conv_id"] in ("c1", "c3")
        # r9 K-row gate schema: probe-row count + the per-regime template hash
        # binding the recorded margins to the exact rendered binding-row bytes.
        assert reg["n_probe_rows"] == (1 if regime == "paired" else 2)
        assert re.fullmatch(r"[0-9a-f]{64}", reg["binding_prompt_sha256"])
        assert reg["worst_row_gap_tokens"] == gp.g.PROMPT_TOKEN_BUDGET
        assert reg["worst_row_gap_conv_id"] in ("c1", "c3")


def test_panel_template_replicas_match_module_constants():
    """Parametric replica drift pin: the gate's self-check inputs, asserted directly."""
    assert (
        gp._panel_paired_system(gp.c.STORY_CHARACTER_NAME, gp.PERSONA_DESC)
        == gp.STORY_PAIRED_SYSTEM_TEMPLATE
    )
    assert (
        gp._panel_op_system(gp.c.STORY_CHARACTER_NAME, gp.PERSONA_DESC)
        == gp.STORY_OP_COMPANION_SYSTEM
    )


@pytest.mark.parametrize(
    ("extra_desc_tokens", "fires"),
    [
        # min margin under FakeTok = PROMPT_TOKEN_BUDGET (c1's prompt holds no '§');
        # the gate requires min_margin >= delta + slack, so the boundary sits at
        # delta == budget - slack (passes) vs delta == budget - slack + 1 (fires).
        (gp.g.PROMPT_TOKEN_BUDGET - gp.PANEL_MARGIN_SLACK_TOKENS, False),
        (gp.g.PROMPT_TOKEN_BUDGET - gp.PANEL_MARGIN_SLACK_TOKENS + 1, True),
    ],
)
def test_panel_invariance_boundary(monkeypatch, tmp_path, extra_desc_tokens, fires):
    """Codex boundary test: emit-config prompt fits; a longer persona config exceeds."""
    matched_dir = _stage_emit_inputs(monkeypatch, tmp_path)
    panel = _write_panel(tmp_path, desc="§" * extra_desc_tokens)
    out = tmp_path / "eligible_conv_ids.json"
    if fires:
        with pytest.raises(AssertionError, match="panel-invariance margin gate FAILED"):
            gp.emit_eligible_ids(
                out, matched_dir, tmp_path / "dl", tokenizer=FakeTok(), panel_path=panel
            )
        assert not out.exists(), "gate must fire BEFORE the export is written"
    else:
        gp.emit_eligible_ids(
            out, matched_dir, tmp_path / "dl", tokenizer=FakeTok(), panel_path=panel
        )
        pi = json.loads(out.read_text())["provenance"]["panel_invariance"]
        assert pi["regimes"]["paired"]["max_panel_delta_tokens"] == extra_desc_tokens
        assert pi["regimes"]["paired"]["max_delta_config"] == "X"


def test_emit_fails_loud_on_missing_panel(monkeypatch, tmp_path):
    matched_dir = _stage_emit_inputs(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="character panel"):
        gp.emit_eligible_ids(
            tmp_path / "o.json",
            matched_dir,
            tmp_path / "dl",
            tokenizer=FakeTok(),
            panel_path=tmp_path / "absent_panel.json",
        )


# --- r9: emit-checkout git provenance (fail-loud) ------------------------------


def _fake_git_provenance(monkeypatch, **kw):
    """Patch the provenance-module boundary with a REAL GitProvenance instance."""
    from explore_persona_space.orchestrate import provenance as pv

    defaults = dict(
        commit_sha="a" * 8,
        dirty=False,
        dirty_paths=[],
        argv0_path=None,
        argv0_state=None,
        commit_sha_full="a" * 40,
    )
    defaults.update(kw)
    prov = pv.GitProvenance(**defaults)
    monkeypatch.setattr(pv, "git_provenance", lambda cwd=None, argv0=None: prov)
    return prov


def test_emit_git_provenance_clean_returns_full_sha(monkeypatch):
    """Real helper body: clean checkout -> full 40-hex git_commit + git_dirty False."""
    _fake_git_provenance(monkeypatch)
    meta = gp._emit_git_provenance()
    assert meta["git_commit"] == "a" * 40
    assert meta["git_dirty"] is False


@pytest.mark.parametrize("dirty", [True, None])
def test_emit_git_provenance_refuses_dirty_or_unknown(monkeypatch, dirty):
    _fake_git_provenance(monkeypatch, dirty=dirty, dirty_paths=["scripts/x.py"] if dirty else [])
    with pytest.raises(RuntimeError, match="emitting checkout is dirty"):
        gp._emit_git_provenance()


def test_emit_git_provenance_allow_dirty_records_state(monkeypatch):
    _fake_git_provenance(monkeypatch, dirty=True, dirty_paths=["scripts/x.py"])
    meta = gp._emit_git_provenance(allow_dirty=True)
    assert meta["git_dirty"] is True
    assert meta["git_dirty_paths"] == ["scripts/x.py"]


def test_emit_git_provenance_refuses_unresolved_head(monkeypatch):
    """A HEAD that cannot resolve to 40-hex refuses even under --allow-dirty-emit."""
    _fake_git_provenance(monkeypatch, commit_sha="unknown", commit_sha_full=None)
    with pytest.raises(RuntimeError, match="40-hex HEAD"):
        gp._emit_git_provenance(allow_dirty=True)


def test_emit_refuses_dirty_checkout_before_writing(monkeypatch, tmp_path):
    """The real emit body routes through the real helper: dirty -> no export written."""
    matched_dir = _stage_emit_inputs(monkeypatch, tmp_path, stub_git=False)
    _fake_git_provenance(monkeypatch, dirty=True, dirty_paths=["scripts/y.py"])
    panel = _write_panel(tmp_path)
    out = tmp_path / "eligible_conv_ids.json"
    with pytest.raises(RuntimeError, match="emitting checkout is dirty"):
        gp.emit_eligible_ids(
            out, matched_dir, tmp_path / "dl", tokenizer=FakeTok(), panel_path=panel
        )
    assert not out.exists(), "provenance refusal must fire BEFORE the export is written"


def test_emit_fails_loud_on_unresolved_tokenizer_revision(monkeypatch, tmp_path):
    """r9: the production tokenizer branch refuses a null / non-sha revision."""
    matched_dir = _stage_emit_inputs(monkeypatch, tmp_path)
    panel = _write_panel(tmp_path)
    out = tmp_path / "eligible_conv_ids.json"
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda model_id, **kw: FakeTok()
    )
    monkeypatch.setattr(gp, "_resolve_tokenizer_revision", lambda tok, model: (None, "unresolved"))
    with pytest.raises(RuntimeError, match="tokenizer revision unresolved"):
        gp.emit_eligible_ids(out, matched_dir, tmp_path / "dl", tokenizer=None, panel_path=panel)
    assert not out.exists(), "tokenizer-revision refusal must fire BEFORE the export is written"


# --- r9: K-row panel-invariance gate --------------------------------------------


def test_panel_invariance_gate_probes_k_min_margin_rows(monkeypatch, tmp_path):
    """The gate probes the K minimum-margin kept rows, not one arbitrary row."""
    extra = [
        # LOWEST margin in both regimes (7 '§' tokens in the question)
        {"conv_id": "c4", "prompt": "q4" + "§" * 7, "response": "w" * 25},
        # second-lowest margin
        {"conv_id": "c5", "prompt": "q5" + "§" * 3, "response": "v" * 25},
    ]
    matched_dir = _stage_emit_inputs(monkeypatch, tmp_path, extra_rows=extra)
    panel = _write_panel(tmp_path)
    out = tmp_path / "eligible_conv_ids.json"
    gp.emit_eligible_ids(out, matched_dir, tmp_path / "dl", tokenizer=FakeTok(), panel_path=panel)
    payload = json.loads(out.read_text())
    assert payload["eligible_paired"] == ["c1", "c4", "c5"]
    assert payload["eligible_op"] == ["c1", "c3", "c4", "c5"]
    budget = gp.g.PROMPT_TOKEN_BUDGET
    pi = payload["provenance"]["panel_invariance"]
    for regime, n_kept in (("paired", 3), ("op", 4)):
        reg = pi["regimes"][regime]
        # all kept rows probed (kept < PANEL_GATE_PROBE_ROWS caps at the pool)
        assert reg["n_probe_rows"] == n_kept
        assert reg["min_margin_conv_id"] == "c4"
        assert reg["min_margin_tokens"] == budget - 7
        # benign panel -> zero delta on every probed row; the worst per-row gap
        # is therefore the binding row's own margin
        assert reg["max_panel_delta_tokens"] == 0
        assert reg["worst_row_gap_tokens"] == budget - 7
        assert reg["worst_row_gap_conv_id"] == "c4"
    # the per-regime template hash is the sha256 of the rendered binding-row
    # prompt — recomputed here through the SAME production render path
    row_c4 = {"conv_id": "c4", "question": "q4" + "§" * 7, "answer": "w" * 25}
    for regime, op in (("paired", False), ("op", True)):
        rendered = gp.build_paired_prompt(row_c4, FakeTok(), op_companion=op)
        expected = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        assert pi["regimes"][regime]["binding_prompt_sha256"] == expected


def test_emit_mode_refuses_combined_flags(monkeypatch, tmp_path):
    """The standalone-flag guard fires BEFORE the variant/mode asserts."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue1345_gen_stories_paired.py",
            "--emit-eligible-ids",
            str(tmp_path / "o.json"),
            "--smoke",
        ],
    )
    with pytest.raises(AssertionError, match="standalone CPU export"):
        gp.main()


# --- (b) sampler --eligible-ids binding ---------------------------------------


def _write_elig(
    path: Path,
    paired: list[str],
    op: list[str],
    *,
    provenance=True,
    panel_sha: str | None = None,
) -> Path:
    payload: dict = {
        "eligible_paired": paired,
        "eligible_op": op,
        "counts": {"n_eligible_paired": len(paired), "n_eligible_op": len(op)},
    }
    if provenance:
        payload["provenance"] = {"git_commit": "deadbeef", "mode": "emit-eligible-ids"}
        if panel_sha is not None:
            payload["provenance"]["panel_invariance"] = {"panel_sha256": panel_sha}
    path.write_text(json.dumps(payload))
    return path


def _live_panel(tmp_path: Path) -> tuple[Path, str]:
    """Tmp live panel.json + its sha256 (the r9 panel-sha binding fixture)."""
    panel = _write_panel(tmp_path)
    return panel, hashlib.sha256(panel.read_bytes()).hexdigest()


def _setup_builder(monkeypatch, tmp_path: Path, shared: list[str], r4op: list[str], r4: list[str]):
    """Fake the hub boundary: repo_info sha, matched + kept-file downloads, listing."""
    kept_by_path = {
        f"{src['prefix']}/{src['kept_basename']}": ids
        for src, ids in ((ps.SOURCES["r4op_kept"], r4op), (ps.SOURCES["r4_kept"], r4))
    }

    class FakeApi:
        def repo_info(self, repo, repo_type):
            return types.SimpleNamespace(sha="f" * 40)

    def fake_download(repo, path, repo_type=None, revision=None, local_dir=None, **kw):
        dest = Path(local_dir) / Path(path).name
        if path == ps.MATCHED_PATH:
            dest.write_text(json.dumps({"shared_r1r2_convs": shared}))
        else:
            rows = kept_by_path[path]
            dest.write_text("\n".join(json.dumps({"conv_id": i}) for i in rows) + "\n")
        return str(dest)

    monkeypatch.setattr(ps, "HfApi", FakeApi)
    monkeypatch.setattr(ps, "hf_hub_download", fake_download)
    monkeypatch.setattr(ps.hub, "retry_transient", lambda fn, what=None: fn())
    monkeypatch.setattr(
        ps.hub,
        "list_hf_files_under_path",
        lambda api, repo, prefix, repo_type, revision: [
            f"{src['prefix']}/{src['kept_basename']}"
            for src in ps.SOURCES.values()
            if src["prefix"] == prefix
        ],
    )


def _builder_argv(
    tmp_path: Path,
    eligf: Path,
    n_sample: int,
    n_reservation: int = 2,
    panel: Path | None = None,
) -> list[str]:
    return [
        "--out",
        str(tmp_path / "panel_manifest.json"),
        "--staging-dir",
        str(tmp_path / "stage"),
        "--n-sample",
        str(n_sample),
        "--n-reservation",
        str(n_reservation),
        "--eligible-ids",
        str(eligf),
        "--panel-json",
        str(panel if panel is not None else tmp_path / "panel.json"),
    ]


def test_builder_excludes_non_eligible_ids_and_records_export(monkeypatch, tmp_path):
    shared = ["c1", "c2", "c3", "c4", "c5", "c6"]
    _setup_builder(monkeypatch, tmp_path, shared, r4op=shared, r4=shared)
    panel, panel_sha = _live_panel(tmp_path)
    # c5 fails op-eligibility, c6 fails paired-eligibility -> intersection c1..c4.
    eligf = _write_elig(
        tmp_path / "elig.json",
        paired=["c1", "c2", "c3", "c4", "c5"],
        op=["c1", "c2", "c3", "c4", "c6"],
        panel_sha=panel_sha,
    )
    rc = ps.main(_builder_argv(tmp_path, eligf, n_sample=4, panel=panel))
    assert rc == 0
    m = json.loads((tmp_path / "panel_manifest.json").read_text())
    assert sorted(m["sample_conv_ids"]) == ["c1", "c2", "c3", "c4"]
    assert "c5" not in m["sample_conv_ids"] and "c6" not in m["sample_conv_ids"]
    assert m["intersections"]["n_eligible"] == 4  # the RESTRICTED eligible set
    rec = m["inputs"]["eligible_ids"]
    assert rec["path"] == str(eligf)
    assert rec["sha256"] == hashlib.sha256(eligf.read_bytes()).hexdigest()
    assert rec["provenance"]["git_commit"] == "deadbeef"
    assert rec["n_shared_before_restrict"] == 6
    assert rec["n_paired_eligible_after_restrict"] == 4
    # r9: the manifest records the live panel it validated the export against
    assert m["inputs"]["panel_json"] == {"path": str(panel), "sha256": panel_sha}


def test_builder_hard_fails_when_intersection_cannot_fill(monkeypatch, tmp_path):
    shared = ["c1", "c2", "c3", "c4", "c5", "c6"]
    _setup_builder(monkeypatch, tmp_path, shared, r4op=shared, r4=shared)
    panel, panel_sha = _live_panel(tmp_path)
    eligf = _write_elig(
        tmp_path / "elig.json",
        paired=["c1", "c2", "c3", "c4", "c5"],
        op=["c1", "c2", "c3", "c4", "c6"],
        panel_sha=panel_sha,
    )
    with pytest.raises(RuntimeError, match="cannot fill the 5-conversation sample"):
        ps.main(_builder_argv(tmp_path, eligf, n_sample=5, panel=panel))


def test_builder_hard_fails_on_empty_restricted_intersection(monkeypatch, tmp_path):
    shared = ["c1", "c2"]
    _setup_builder(monkeypatch, tmp_path, shared, r4op=shared, r4=shared)
    panel, panel_sha = _live_panel(tmp_path)
    eligf = _write_elig(tmp_path / "elig.json", paired=["zz"], op=["zz"], panel_sha=panel_sha)
    with pytest.raises(RuntimeError, match="EMPTY after the gen-feasibility restriction"):
        ps.main(_builder_argv(tmp_path, eligf, n_sample=1, panel=panel))


# --- r9: panel-sha binding (a panel-only edit forces a re-emit) -----------------


def test_builder_rejects_panel_sha_mismatch(monkeypatch, tmp_path):
    shared = ["c1", "c2"]
    _setup_builder(monkeypatch, tmp_path, shared, r4op=shared, r4=shared)
    panel, _ = _live_panel(tmp_path)
    # export recorded against a DIFFERENT panel (e.g. panel.json edited after emit)
    eligf = _write_elig(
        tmp_path / "elig.json", paired=["c1", "c2"], op=["c1", "c2"], panel_sha="0" * 64
    )
    with pytest.raises(ValueError, match="panel sha mismatch"):
        ps.main(_builder_argv(tmp_path, eligf, n_sample=1, n_reservation=1, panel=panel))
    assert not (tmp_path / "panel_manifest.json").exists(), "no manifest on a rejected export"


def test_builder_rejects_export_missing_panel_sha(monkeypatch, tmp_path):
    """An export predating the panel-invariance gate is rejected loud."""
    shared = ["c1", "c2"]
    _setup_builder(monkeypatch, tmp_path, shared, r4op=shared, r4=shared)
    panel, _ = _live_panel(tmp_path)
    eligf = _write_elig(tmp_path / "elig.json", paired=["c1"], op=["c1"])  # no panel_sha
    with pytest.raises(KeyError, match="panel_invariance"):
        ps.main(_builder_argv(tmp_path, eligf, n_sample=1, n_reservation=1, panel=panel))
    assert not (tmp_path / "panel_manifest.json").exists()


def test_builder_rejects_missing_live_panel(monkeypatch, tmp_path):
    shared = ["c1", "c2"]
    _setup_builder(monkeypatch, tmp_path, shared, r4op=shared, r4=shared)
    eligf = _write_elig(tmp_path / "elig.json", paired=["c1"], op=["c1"], panel_sha="0" * 64)
    with pytest.raises(FileNotFoundError, match="panel-json"):
        ps.main(
            _builder_argv(
                tmp_path, eligf, n_sample=1, n_reservation=1, panel=tmp_path / "absent.json"
            )
        )
    assert not (tmp_path / "panel_manifest.json").exists()


def test_builder_requires_eligible_ids_flag(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as ei:
        ps.main(["--out", str(tmp_path / "m.json")])
    assert ei.value.code == 2  # argparse: the required --eligible-ids is missing


def test_builder_fails_loud_on_export_missing_provenance(monkeypatch, tmp_path):
    shared = ["c1", "c2"]
    _setup_builder(monkeypatch, tmp_path, shared, r4op=shared, r4=shared)
    eligf = _write_elig(tmp_path / "elig.json", paired=["c1"], op=["c1"], provenance=False)
    with pytest.raises(KeyError, match="provenance"):
        ps.main(_builder_argv(tmp_path, eligf, n_sample=1, n_reservation=1))


# --- C2 export-schema hardening (concern round, r8) ---------------------------


def _mut_drop_counts(p: dict) -> None:
    del p["counts"]


def _mut_null_provenance(p: dict) -> None:
    p["provenance"] = None


def _mut_empty_provenance(p: dict) -> None:
    p["provenance"] = {}


def _mut_counts_not_dict(p: dict) -> None:
    p["counts"] = [1, 2]


def _mut_dup_paired(p: dict) -> None:
    p["eligible_paired"] = ["c1", "c1", "c2"]
    p["counts"]["n_eligible_paired"] = 3  # counts consistent: the DUP check must fire


def _mut_dup_op(p: dict) -> None:
    p["eligible_op"] = ["c1", "c2", "c2"]
    p["counts"]["n_eligible_op"] = 3


def _mut_paired_count_mismatch(p: dict) -> None:
    p["counts"]["n_eligible_paired"] = 99


def _mut_op_count_missing(p: dict) -> None:
    del p["counts"]["n_eligible_op"]


@pytest.mark.parametrize(
    ("mutate", "exc", "match"),
    [
        (_mut_drop_counts, KeyError, "counts"),
        (_mut_null_provenance, ValueError, "provenance must be a non-empty dict"),
        (_mut_empty_provenance, ValueError, "provenance must be a non-empty dict"),
        (_mut_counts_not_dict, ValueError, "counts must be a dict"),
        (_mut_dup_paired, ValueError, "duplicate ids"),
        (_mut_dup_op, ValueError, "duplicate ids"),
        (_mut_paired_count_mismatch, ValueError, "does not match"),
        (_mut_op_count_missing, ValueError, "does not match"),
    ],
)
def test_builder_rejects_malformed_export(monkeypatch, tmp_path, mutate, exc, match):
    """Each C2 rejection fires loud — never a silent dedup / silent default."""
    shared = ["c1", "c2"]
    _setup_builder(monkeypatch, tmp_path, shared, r4op=shared, r4=shared)
    panel, panel_sha = _live_panel(tmp_path)
    eligf = _write_elig(
        tmp_path / "elig.json", paired=["c1", "c2"], op=["c1", "c2"], panel_sha=panel_sha
    )
    payload = json.loads(eligf.read_text())
    mutate(payload)
    eligf.write_text(json.dumps(payload))
    with pytest.raises(exc, match=match):
        ps.main(_builder_argv(tmp_path, eligf, n_sample=1, n_reservation=1, panel=panel))
    assert not (tmp_path / "panel_manifest.json").exists(), "no manifest on a rejected export"
