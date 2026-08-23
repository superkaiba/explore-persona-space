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
    plus the emit mode's standalone-flag guard firing BEFORE the variant asserts.

Hermetic: tmp-path staged files; fakes ONLY at the hub/tokenizer boundary (the
fake tokenizer mirrors the two consumed methods' signatures; hf downloads are
monkeypatched to tmp writers). Synthetic text only. No network, no GPU.
"""

from __future__ import annotations

import hashlib
import json
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


def _stage_emit_inputs(monkeypatch, tmp_path: Path) -> Path:
    """Tmp matched allowlist + track_s rows; stage_pinned_file faked to tmp."""
    matched_dir = tmp_path / "matched"
    matched_dir.mkdir()
    (matched_dir / "matched_subsets_parent.json").write_text(
        json.dumps({"shared_r1r2_convs": ["c1", "c2", "c3", "c4"]})
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
    ]
    track.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def fake_stage(path_in_repo, dest_dir, revision=gp.c.PIN_REV):
        assert path_in_repo == gp.c.PARENT_TRACK_S_JSONL
        return track

    monkeypatch.setattr(gp.c, "stage_pinned_file", fake_stage)
    return matched_dir


def test_emit_eligible_ids_schema(monkeypatch, tmp_path):
    matched_dir = _stage_emit_inputs(monkeypatch, tmp_path)
    out = tmp_path / "sub" / "eligible_conv_ids.json"
    gp.emit_eligible_ids(out, matched_dir, tmp_path / "dl", tokenizer=FakeTok())
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


def _write_elig(path: Path, paired: list[str], op: list[str], *, provenance=True) -> Path:
    payload: dict = {
        "eligible_paired": paired,
        "eligible_op": op,
        "counts": {"n_eligible_paired": len(paired), "n_eligible_op": len(op)},
    }
    if provenance:
        payload["provenance"] = {"git_commit": "deadbeef", "mode": "emit-eligible-ids"}
    path.write_text(json.dumps(payload))
    return path


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


def _builder_argv(tmp_path: Path, eligf: Path, n_sample: int, n_reservation: int = 2) -> list[str]:
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
    ]


def test_builder_excludes_non_eligible_ids_and_records_export(monkeypatch, tmp_path):
    shared = ["c1", "c2", "c3", "c4", "c5", "c6"]
    _setup_builder(monkeypatch, tmp_path, shared, r4op=shared, r4=shared)
    # c5 fails op-eligibility, c6 fails paired-eligibility -> intersection c1..c4.
    eligf = _write_elig(
        tmp_path / "elig.json",
        paired=["c1", "c2", "c3", "c4", "c5"],
        op=["c1", "c2", "c3", "c4", "c6"],
    )
    rc = ps.main(_builder_argv(tmp_path, eligf, n_sample=4))
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


def test_builder_hard_fails_when_intersection_cannot_fill(monkeypatch, tmp_path):
    shared = ["c1", "c2", "c3", "c4", "c5", "c6"]
    _setup_builder(monkeypatch, tmp_path, shared, r4op=shared, r4=shared)
    eligf = _write_elig(
        tmp_path / "elig.json",
        paired=["c1", "c2", "c3", "c4", "c5"],
        op=["c1", "c2", "c3", "c4", "c6"],
    )
    with pytest.raises(RuntimeError, match="cannot fill the 5-conversation sample"):
        ps.main(_builder_argv(tmp_path, eligf, n_sample=5))


def test_builder_hard_fails_on_empty_restricted_intersection(monkeypatch, tmp_path):
    shared = ["c1", "c2"]
    _setup_builder(monkeypatch, tmp_path, shared, r4op=shared, r4=shared)
    eligf = _write_elig(tmp_path / "elig.json", paired=["zz"], op=["zz"])
    with pytest.raises(RuntimeError, match="EMPTY after the gen-feasibility restriction"):
        ps.main(_builder_argv(tmp_path, eligf, n_sample=1))


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
