"""Pins for the #1739 bare-query capture leg.

Covers the render convention (#1092 byte-identity + the constant-prefix null),
the leg-2 extract/dedupe phase, the capture-row builder for both legs, the
pilot fence's designed halt, the scope guard that keeps leg 2 evil-only, and the
import-check shadow class (an inline import block would make a bare name a
function-wide local of main() and shadow a module-level symbol — the wcrung
`capture` UnboundLocalError).

No real corpus text: every query is a synthetic placeholder.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import issue1739_bareq_pod as bq  # noqa: E402

_HEAD = "<|im_start|>system\nSYS<|im_end|>\n"
_USER = "<|im_start|>user\n"


class _FakeTokenizer:
    """Qwen-shaped chat template; whitespace tokenizer."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = _HEAD
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out

    def encode(self, text, add_special_tokens=False):
        return text.split()


def _q(i: int) -> str:
    return f"PLACEHOLDER_QUERY_{i}"


# --- render convention -----------------------------------------------------


def test_bare_render_prefix_is_content_independent():
    tok = _FakeTokenizer()
    p1, r1 = bq.bare_render(tok, _q(1))
    p2, r2 = bq.bare_render(tok, _q(2))
    assert p1 == p2, "the bare prefix must be the same template head for every query"
    assert r1 != r2, "the prompts must differ (they carry different queries)"
    assert _q(1) in r1 and _q(1) not in p1


def test_query_id_is_stable_and_content_keyed():
    a, b = bq._query_id(_q(1)), bq._query_id(_q(1))
    assert a == b and len(a) == 16
    assert a != bq._query_id(_q(2))


# --- the constant-prefix null probe ---------------------------------------


def test_null_probe_passes_on_uniform_prefixes():
    tok = _FakeTokenizer()
    rows = [
        dict(zip(("prefix_text", "prompt_text"), bq.bare_render(tok, _q(i)), strict=True))
        for i in range(4)
    ]
    rep = bq._null_probe_report(rows, tok)
    assert rep["constant_prefix_verified"] is True
    assert rep["prefix_token_len"] == len(_HEAD.split())
    assert len(rep["prefix_sha256"]) == 64
    assert "built-in null" in rep["note"]


def test_null_probe_fails_loud_on_prefix_drift():
    """A leaked non-bare prefix would make the null arm predictive — fail loud."""
    tok = _FakeTokenizer()
    rows = [
        dict(zip(("prefix_text", "prompt_text"), bq.bare_render(tok, _q(i)), strict=True))
        for i in range(3)
    ]
    rows[1]["prefix_text"] = _HEAD + _USER + "LEAKED_HISTORY<|im_end|>\n"
    with pytest.raises(RuntimeError, match="DISTINCT prefixes"):
        bq._null_probe_report(rows, tok)


# --- leg 2: extract + dedupe ---------------------------------------------


def _shard(path: Path, recs: list[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for cid, q in recs:
            fh.write(
                json.dumps({"src": f"{cid}.json", "doc": {"context_id": cid, "query": q}}) + "\n"
            )
    return path


def _extract_args(tmp_path: Path, **kw):
    args = bq._parse_args(
        [
            "--out-root",
            str(tmp_path / "out"),
            "--stage-root",
            str(tmp_path / "stage"),
            "--store-root",
            str(tmp_path / "store"),
        ]
    )
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def test_extract_dedupes_by_query_and_records_members(tmp_path, monkeypatch):
    """Two contexts sharing a query collapse to ONE capture row (the width win)."""
    args = _extract_args(tmp_path, keep_shards=True, reap_shards=False)
    s = _shard(
        tmp_path / "raw" / "labeling_evil.shard00.jsonl",
        [
            ("evil-train-cross-000001", _q(1)),
            ("evil-train-cross-000002", _q(1)),  # same query, different prefix row
            ("evil-train-cross-000003", _q(2)),
            ("evil-eval-hhrt-000001", _q(3)),  # not train -> filtered
        ],
    )
    monkeypatch.setattr(bq, "iter_raw_shards", lambda a, t: iter([s]))
    man = bq.extract_query_bank(args, "tok")

    assert man["n_unique_queries"] == 2, "3 train contexts over 2 distinct queries"
    assert man["n_contexts"] == 3
    assert man["dedupe_ratio_contexts_per_query"] == pytest.approx(1.5)
    shared = [e for e in man["queries"] if len(e["context_ids"]) == 2]
    assert len(shared) == 1
    assert sorted(shared[0]["context_ids"]) == [
        "evil-train-cross-000001",
        "evil-train-cross-000002",
    ]
    # the manifest round-trips to disk
    on_disk = json.loads((args.out_root / bq.QUERY_MANIFEST).read_text())
    assert on_disk["n_unique_queries"] == 2
    assert on_disk["train_only"] is True


def test_extract_all_rungs_keeps_eval_contexts(tmp_path, monkeypatch):
    args = _extract_args(tmp_path, train_only=False)
    s = _shard(
        tmp_path / "raw" / "labeling_evil.shard00.jsonl",
        [("evil-train-cross-000001", _q(1)), ("evil-eval-hhrt-000001", _q(9))],
    )
    monkeypatch.setattr(bq, "iter_raw_shards", lambda a, t: iter([s]))
    man = bq.extract_query_bank(args, "tok")
    assert man["n_unique_queries"] == 2 and man["n_contexts"] == 2


# --- capture-row builder -------------------------------------------------


def _wcrung_rows(tmp_path: Path) -> Path:
    rows = [
        {"context_id": "wcrung-0000", "query": _q(10), "prefix_turns": []},
        {
            "context_id": "wcrung-0001",
            "query": _q(11),
            "prefix_turns": [
                {"role": "user", "content": "EARLIER"},
                {"role": "assistant", "content": "REPLY"},
            ],
        },
        {
            "context_id": "wcrung-0002",
            "query": _q(12),
            "prefix_turns": [{"role": "user", "content": "EARLIER2"}],
        },
    ]
    p = tmp_path / "wcrung.json"
    p.write_text(json.dumps({"rows": rows}))
    return p


def test_leg1_default_captures_only_multi_turn(tmp_path):
    """The 987 single-turn rows are REUSED, not re-captured — so leg 1 skips them."""
    args = _extract_args(tmp_path, leg="1", wcrung_rows_json=_wcrung_rows(tmp_path))
    rows = bq.build_capture_rows(args, _FakeTokenizer())
    assert len(rows) == 2, "only the two multi-turn rows"
    assert all(r["multi_turn"] for r in rows)
    assert {r["kind"] for r in rows} == {"leg1_wcrung_multi_turn"}
    assert all(r["row_id"].startswith("wc-") for r in rows)


def test_leg1_include_single_turn_opt_in(tmp_path):
    args = _extract_args(
        tmp_path, leg="1", multi_turn_only=False, wcrung_rows_json=_wcrung_rows(tmp_path)
    )
    rows = bq.build_capture_rows(args, _FakeTokenizer())
    assert len(rows) == 3
    assert sum(not r["multi_turn"] for r in rows) == 1


def test_leg2_rows_come_from_the_query_manifest(tmp_path):
    args = _extract_args(tmp_path, leg="2")
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / bq.QUERY_MANIFEST).write_text(
        json.dumps(
            {
                "queries": [
                    {"query_id": "aa" * 8, "query": _q(1), "context_ids": ["c1", "c2"]},
                    {"query_id": "bb" * 8, "query": _q(2), "context_ids": ["c3"]},
                ]
            }
        )
    )
    rows = bq.build_capture_rows(args, _FakeTokenizer())
    assert len(rows) == 2
    assert {r["kind"] for r in rows} == {"leg2_query_bank"}
    assert [r["n_member_contexts"] for r in rows] == [2, 1]
    assert all(r["row_id"].startswith("q-") for r in rows)


def test_missing_leg_input_fails_loud(tmp_path):
    args = _extract_args(tmp_path, leg="2")
    with pytest.raises((RuntimeError, FileNotFoundError)):
        bq.build_capture_rows(args, _FakeTokenizer())


# --- CLI scope guards ----------------------------------------------------


def test_leg2_refuses_a_bare_train_pool_behavior(capsys):
    """sycophancy/hallucination train contexts are already bare — a no-op."""
    for b in ("sycophancy", "hallucination"):
        with pytest.raises(SystemExit):
            bq._parse_args(["--leg", "2", "--behavior", b])
        assert "already bare" in capsys.readouterr().err


def test_leg2_accepts_evil():
    assert bq._parse_args(["--leg", "2", "--behavior", "evil"]).behavior == "evil"


def test_defaults_match_the_design(tmp_path):
    a = bq._parse_args([])
    assert a.leg == "both" and a.behavior == "evil"
    assert a.multi_turn_only is True and a.train_only is True
    assert a.n_layers == 28 and a.hidden_dim == 3584
    assert a.out_root.name == "bareq_map"
    assert bq.BARE_KIND == "context_end"
    assert bq.PILOT_FENCE_RC == 8


# --- pilot fence ---------------------------------------------------------


def test_pilot_over_fence_is_a_designed_halt_with_report_written_first(tmp_path, monkeypatch):
    # 0.05 s over 2 rows -> ~0.025 s/row -> ~6.9e-5 h for 10 rows; fence below that.
    args = _extract_args(tmp_path, leg="1", fence_hours=1e-6, pilot_rows=2)
    rows = [
        {
            "row_id": f"r{i}",
            "kind": "leg1_wcrung_multi_turn",
            "multi_turn": True,
            "prefix_text": _HEAD,
            "prompt_text": _HEAD + str(i),
        }
        for i in range(10)
    ]

    def _slow_capture(*a, **k):  # a measurable per-row cost, so the projection is real
        time.sleep(0.05)
        return {"n_rows": 2, "n_shards": 1}

    monkeypatch.setattr(bq, "run_capture", _slow_capture)
    rc = bq.run_pilot(args, rows, _FakeTokenizer(), object())
    assert rc == bq.PILOT_FENCE_RC, "over-fence projection must be a designed halt"
    rep = json.loads((args.out_root / bq.PILOT_REPORT).read_text())
    assert rep["over_fence"] is True
    assert rep["total_rows_planned"] == 10 and rep["pilot_rows"] == 2
    assert "MEASURED" in rep["basis"], "the basis must be measured, never asserted"


def test_pilot_under_fence_returns_zero(tmp_path, monkeypatch):
    args = _extract_args(tmp_path, leg="1", fence_hours=1000.0, pilot_rows=2)
    rows = [
        {
            "row_id": f"r{i}",
            "kind": "k",
            "multi_turn": True,
            "prefix_text": _HEAD,
            "prompt_text": _HEAD,
        }
        for i in range(4)
    ]
    monkeypatch.setattr(bq, "run_capture", lambda *a, **k: {"n_rows": 2, "n_shards": 1})
    assert bq.run_pilot(args, rows, _FakeTokenizer(), object()) == 0
    assert json.loads((args.out_root / bq.PILOT_REPORT).read_text())["over_fence"] is False


# --- the import-check shadow class --------------------------------------


def test_main_locals_do_not_shadow_module_level_symbols():
    """An inline import block would make a bare name a function-wide local.

    `_import_check` is the containment function and the sanctioned exception: it
    reads no module-level name, so nothing it binds can shadow anything.
    """
    mod_names = {n for n in dir(bq) if not n.startswith("__")}
    offenders = []
    for fname in dir(bq):
        fn = getattr(bq, fname)
        if not isinstance(fn, types.FunctionType) or fn.__module__ != bq.__name__:
            continue
        if fname == "_import_check":
            continue
        for s in sorted(mod_names & set(fn.__code__.co_varnames)):
            offenders.append(f"{fname}() shadows module-level {s!r}")
    assert not offenders, "\n  ".join(offenders)
    assert "capture" not in bq.main.__code__.co_varnames


def test_import_check_exits_zero():
    assert bq._import_check() == 0
