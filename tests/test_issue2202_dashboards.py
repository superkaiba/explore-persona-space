"""#2202 dashboard pins: synthetic OVER-CAP shard split (the plan's named smoke
blind spot — the 40 MB / shard-split branch binds only at production row
counts), the confuser-cap tightening ladder with rows NEVER cut, truncation
disclosure, and the content-probe counting. Synthetic fixtures in tmp_path; no
network, no HF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2202_dashboards as DB  # noqa: E402
import issue2202_labels as LB  # noqa: E402


def test_shard_pages_split_and_row_preservation(tmp_path, monkeypatch):
    monkeypatch.setattr(DB, "SHARD_BYTES", 4_000)
    rows = [f"<tr><td>{i}</td><td>{'x' * 900}</td></tr>" for i in range(40)]
    written = DB.shard_pages(rows, "unit-fail", "Unit", "<tr><th>h</th></tr>", "<p>i</p>", tmp_path)
    shard_files = [p for p in written if "_p" in p.name]
    assert len(shard_files) >= 2  # the over-cap split branch fired
    # every shard within budget + page overhead; row count preserved exactly
    total_rows = 0
    for p in shard_files:
        body = p.read_text(encoding="utf-8")
        assert p.stat().st_size <= 4_000 + 6_000  # shard budget + fixed page overhead
        total_rows += body.count("<tr><td>")
    assert total_rows == 40  # rows are NEVER cut
    idx = tmp_path / "unit-fail.html"
    assert idx.exists()
    assert idx.read_text(encoding="utf-8").count("shard ") >= len(shard_files)


def _write_fixtures(root: Path, n_fail: int = 3, n_conf: int = 2) -> dict:
    """Tiny out-eval tree + text cache + #1738-labels fixture for phase_build."""
    out = root / "eval"
    out.mkdir(parents=True)
    cis = list(range(10))
    conf_rows = [
        {
            "row": i,
            "ci": i,
            "rank": 5.0,
            "n_outrank": 4,
            "attribution": "UNKNOWN",
            "confusers": [
                {
                    "row": 9 - j,
                    "ci": 9 - j,
                    "d_pred": 0.1 * j,
                    "rank_fwd": j + 1,
                    "rank_ctx": 2.0,
                    "rank_ans": 3.0,
                    "sims": {
                        rel: {
                            "cos_raw": 0.9,
                            "cos_cent": 0.5,
                            "cos_whiten": 0.4,
                            "d_raw": 1.0,
                            "d_whiten": 2.0,
                        }
                        for rel in ("cc", "aa", "ac", "pa")
                    },
                }
                for j in range(n_conf)
            ],
        }
        for i in range(n_fail)
    ]
    (out / "failures_confusion.json").write_text(
        json.dumps(
            {
                "n_fail1": n_fail,
                "n_detail_rows": n_fail,
                "confusers_per_row": n_conf,
                "primary_space": "raw_euclidean",
                "rows": conf_rows,
            }
        )
    )
    (out / "sample500_lists.json").write_text(
        json.dumps(
            {
                "seed": 2202,
                "n_sample": 2,
                "rows": [
                    {
                        "ci": 4,
                        "rank": 1.0,
                        "fail": False,
                        "retrieval": [{"ci": 4, "d": 0.0, "is_true": True, "cos_raw": 1.0}],
                        "collapse": [{"ci": 5, "d": 0.2, "cos_raw": 0.8}],
                    },
                    {
                        "ci": 6,
                        "rank": 3.0,
                        "fail": True,
                        "retrieval": [{"ci": 7, "d": 0.1, "is_true": False, "cos_raw": 0.9}],
                        "collapse": [{"ci": 8, "d": 0.3, "cos_raw": 0.7}],
                    },
                ],
            }
        )
    )
    header = (
        "ci,rank_raw_euclidean,worst_rank_tail,worst_dist_tail,fail_raw_euclidean,in_sample500\n"
    )
    (out / "percontext_ranks.csv").write_text(
        header + "".join(f"{c},5.0,{1 if c == 0 else 0},0,1,0\n" for c in cis)
    )
    cache = root / "judge_texts.jsonl"
    cache.write_text(
        "\n".join(
            json.dumps(
                {
                    "ci": c,
                    "corpus": "wildchat",
                    "history_tail": "",
                    "last_user": f"user question {c} " + "u" * 600,
                    "response": f"answer {c} " + "r" * 600,
                }
            )
            for c in cis
        )
        + "\n"
    )
    labels = root / "labels_1738.json"
    labels.write_text(
        json.dumps(
            {
                "labels": {
                    str(c): {
                        "language": "en",
                        "topic": "chitchat_social",
                        "request_refusal_adjacent": "no",
                        "answer_is_refusal": "no",
                        "format": "prose",
                    }
                    for c in cis
                }
            }
        )
    )
    return {"out": out, "cache": cache, "labels": labels}


def _args(root: Path, fx: dict, dash_out: Path) -> object:
    return DB.build_argparser().parse_args(
        [
            "--phase",
            "build",
            "--out-eval",
            str(fx["out"]),
            "--dash-out",
            str(dash_out),
            "--text-cache",
            str(fx["cache"]),
            "--labels-1738",
            str(fx["labels"]),  # absolute path overrides PROJECT_ROOT join
            "--no-upload",
            "--work-root",
            str(root / "wr"),
        ]
    )


def test_phase_build_renders_and_probes(tmp_path):
    fx = _write_fixtures(tmp_path)
    dash = tmp_path / "dash"
    args = _args(tmp_path, fx, dash)
    args.work_root = Path(args.work_root)
    DB.phase_build(args)
    meta = json.loads((dash / "dashboards_meta_2202.json").read_text())
    assert meta["n_tr"] >= 5 and not meta["over_cap"]
    body = (dash / "failures-2202_p1.html").read_text(encoding="utf-8")
    assert "user question 0" in body and "…[truncated]" in body
    assert "cc" in body and "UNKNOWN" in body
    sample = (dash / "sample500-2202_p1.html").read_text(encoding="utf-8")
    assert "TRUE" in sample and "FAIL-1" in sample


def test_confuser_cap_ladder_tightens_but_never_cuts_rows(tmp_path, monkeypatch):
    fx = _write_fixtures(tmp_path, n_fail=4, n_conf=2)
    dash = tmp_path / "dash2"
    monkeypatch.setattr(DB, "TOTAL_CAP_BYTES", 1_000)  # force the over-cap branch
    args = _args(tmp_path, fx, dash)
    args.work_root = Path(args.work_root)
    DB.phase_build(args)
    meta = json.loads((dash / "dashboards_meta_2202.json").read_text())
    assert meta["confuser_cap_used"] == DB.CONFUSER_CAP_LADDER[-1]  # ladder exhausted
    assert meta["over_cap"] is True  # recorded, shipped anyway — rows never cut
    body = (dash / "failures-2202_p1.html").read_text(encoding="utf-8")
    assert body.count("user question") >= 4  # all fail rows rendered


def test_cap_text_disclosure_used_in_rows():
    assert LB.cap_text("y" * 500, 100).endswith("…[truncated]")


def test_cap_ladder_counts_projected_sample_bytes(tmp_path, monkeypatch):
    """Round-2 fix: the 40 MB ladder's total includes the PROJECTED sample-page
    bytes — pre-fix it compared failures-only, so a combined over-cap payload
    shipped at the top confuser rung with only a warning."""
    fx = _write_fixtures(tmp_path, n_fail=4, n_conf=2)
    dash1 = tmp_path / "dash_measure"
    args = _args(tmp_path, fx, dash1)
    args.work_root = Path(args.work_root)
    DB.phase_build(args)  # under the default cap: measures per-stem page bytes
    meta = json.loads((dash1 / "dashboards_meta_2202.json").read_text())
    fail_bytes = sum(sz for nm, sz in meta["files"].items() if nm.startswith("failures-2202"))
    samp_bytes = sum(sz for nm, sz in meta["files"].items() if nm.startswith("sample500-2202"))
    assert fail_bytes > 0 and samp_bytes > 2
    assert meta["confuser_cap_used"] == DB.CONFUSER_CAP_LADDER[0]
    # failures ALONE fit at the top rung (+1 headroom); failures + sample do NOT
    monkeypatch.setattr(DB, "TOTAL_CAP_BYTES", fail_bytes + 1)
    dash2 = tmp_path / "dash_capped"
    args2 = _args(tmp_path, fx, dash2)
    args2.work_root = Path(args2.work_root)
    DB.phase_build(args2)
    meta2 = json.loads((dash2 / "dashboards_meta_2202.json").read_text())
    # pre-fix: the ladder broke at rung 1 (failures-only under cap); post-fix it tightens
    assert meta2["confuser_cap_used"] != DB.CONFUSER_CAP_LADDER[0]
    assert meta2["total_bytes"] == sum(meta2["files"].values())  # combined accounting


def test_shard_pages_sweeps_stale_orphans(tmp_path, monkeypatch):
    """Round-2 fix: a rebuild yielding fewer shards removes stale {stem}_pN.html
    orphans beyond the fresh count (the fresh index de-links them, but the files
    would keep serving at their old URLs)."""
    head = "<tr><th>h</th></tr>"
    rows = [f"<tr><td>{i}</td><td>{'x' * 900}</td></tr>" for i in range(10)]
    monkeypatch.setattr(DB, "SHARD_BYTES", 2_000)
    w1 = DB.shard_pages(rows, "unit-orph", "U", head, "<p>i</p>", tmp_path)
    n1 = len([p for p in w1 if "_p" in p.name])
    assert n1 >= 3  # the first build fans out into several shards
    monkeypatch.setattr(DB, "SHARD_BYTES", 2_500_000)
    w2 = DB.shard_pages(rows, "unit-orph", "U", head, "<p>i</p>", tmp_path)
    assert len([p for p in w2 if "_p" in p.name]) == 1
    left = sorted(p.name for p in tmp_path.glob("unit-orph_p*.html"))
    assert left == ["unit-orph_p1.html"]  # stale _p2.._pN swept
    # a different stem's shards are untouched by the sweep
    DB.shard_pages(rows[:1], "other-stem", "U", head, "<p>i</p>", tmp_path)
    DB.shard_pages(rows[:1], "unit-orph", "U", head, "<p>i</p>", tmp_path)
    assert (tmp_path / "other-stem_p1.html").exists()
