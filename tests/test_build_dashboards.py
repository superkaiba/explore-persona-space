"""Tests for scripts/build_dashboards.py (standardized dashboard generator).

Covers: JSON / JSONL / CSV rendering with correct row counts + naming;
sharding + index page under a forced tiny --shard-mb; manifest shape;
emit-links output format + 40-hex SHA validation; and the over-cap FAIL.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load the generator as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_dashboards.py"
_spec = importlib.util.spec_from_file_location("build_dashboards", _SCRIPT)
build_dashboards = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["build_dashboards"] = build_dashboards
_spec.loader.exec_module(build_dashboards)  # type: ignore[union-attr]


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fake repo root: a `.git` marker so _find_repo_root resolves cleanly."""
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def out_dir(repo: Path) -> Path:
    return repo / "experiments" / "dashboards"


def _run(argv: list[str]) -> int:
    return build_dashboards.main(argv)


def _write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)] + [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest(out_dir: Path, issue: int) -> list[dict]:
    return json.loads((out_dir / f"issue{issue}_manifest.json").read_text())


# ─── Input formats + naming + row counts ─────────────────────────────────


def test_json_renders_with_naming_and_row_count(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "contexts.json"
    _write_json(src, [{"id": 1, "q": "hello"}, {"id": 2, "q": "world"}])
    rc = _run(["build", "--issue", "667", "--table", f"contexts={src}", "--out-dir", str(out_dir)])
    assert rc == 0
    page = out_dir / "issue667_contexts.html"
    assert page.exists()
    manifest = _manifest(out_dir, 667)
    assert manifest == [
        {
            "table": "contexts",
            "files": ["experiments/dashboards/issue667_contexts.html"],
            "rows": 2,
            "bytes": len(page.read_text(encoding="utf-8").encode("utf-8")),
        }
    ]
    html = page.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "hello" in html and "world" in html


def test_jsonl_renders(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "c.jsonl"
    _write_jsonl(src, [{"a": 1}, {"a": 2}, {"a": 3}])
    assert (
        _run(["build", "--issue", "5", "--table", f"comps={src}", "--out-dir", str(out_dir)]) == 0
    )
    assert (out_dir / "issue5_comps.html").exists()
    assert _manifest(out_dir, 5)[0]["rows"] == 3


def test_jsonl_skips_blank_lines(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "c.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text('{"a": 1}\n\n  \n{"a": 2}\n', encoding="utf-8")
    assert _run(["build", "--issue", "5", "--table", f"c={src}", "--out-dir", str(out_dir)]) == 0
    assert _manifest(out_dir, 5)[0]["rows"] == 2


def test_csv_renders(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "p.csv"
    _write_csv(src, ["layer", "score"], [["7", "0.9"], ["14", "0.5"], ["21", "0.1"]])
    assert (
        _run(["build", "--issue", "667", "--table", f"probes={src}", "--out-dir", str(out_dir)])
        == 0
    )
    manifest = _manifest(out_dir, 667)
    assert manifest[0]["rows"] == 3
    html = (out_dir / "issue667_probes.html").read_text(encoding="utf-8")
    assert "layer" in html and "score" in html


def test_columns_are_union_first_seen_order(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "ragged.jsonl"
    _write_jsonl(src, [{"a": 1, "b": 2}, {"b": 3, "c": 4}, {"a": 5}])
    assert _run(["build", "--issue", "1", "--table", f"t={src}", "--out-dir", str(out_dir)]) == 0
    html = (out_dir / "issue1_t.html").read_text(encoding="utf-8")
    # Header order = first-seen: a, b, c.
    thead = html.split("<thead>")[1].split("</thead>")[0]
    assert thead.index(">a<") < thead.index(">b<") < thead.index(">c<")


def test_multiple_tables_one_manifest(repo: Path, out_dir: Path) -> None:
    a = repo / "data" / "a.json"
    b = repo / "data" / "b.csv"
    _write_json(a, [{"x": 1}])
    _write_csv(b, ["y"], [["2"], ["3"]])
    rc = _run(
        [
            "build",
            "--issue",
            "9",
            "--table",
            f"alpha={a}",
            "--table",
            f"beta={b}",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    manifest = _manifest(out_dir, 9)
    tables = {e["table"]: e for e in manifest}
    assert set(tables) == {"alpha", "beta"}
    assert tables["alpha"]["rows"] == 1 and tables["beta"]["rows"] == 2
    assert (out_dir / "issue9_alpha.html").exists()
    assert (out_dir / "issue9_beta.html").exists()


# ─── Sharding ────────────────────────────────────────────────────────────


def test_sharding_kicks_in_with_index_page(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "big.jsonl"
    # 200 rows, each with a chunky text field, forced under a 0.01 MB (10 KB) cap.
    records = [{"i": i, "text": f"row-{i} " + "x" * 300} for i in range(200)]
    _write_jsonl(src, records)
    rc = _run(
        [
            "build",
            "--issue",
            "667",
            "--table",
            f"big={src}",
            "--out-dir",
            str(out_dir),
            "--shard-mb",
            "0.01",
        ]
    )
    assert rc == 0
    manifest = _manifest(out_dir, 667)
    entry = manifest[0]
    assert entry["rows"] == 200
    files = entry["files"]
    # Index page first, then >= 2 numbered shards.
    assert files[0] == "experiments/dashboards/issue667_big.html"
    shards = [f for f in files if "_p" in Path(f).stem]
    assert len(shards) >= 2
    # Shard filenames are numerically suffixed p1, p2, ...
    assert files[1].endswith("issue667_big_p1.html")

    index_html = (out_dir / "issue667_big.html").read_text(encoding="utf-8")
    # Index links each shard and shows a row range.
    assert "issue667_big_p1.html" in index_html
    assert "rows 1-" in index_html
    # Index is an index (no data table).
    assert '<table id="t">' not in index_html

    # Every row is present exactly once across the shards.
    all_shard_html = "".join((out_dir / Path(f).name).read_text(encoding="utf-8") for f in shards)
    for i in range(200):
        assert f"row-{i} " in all_shard_html


def test_no_sharding_when_small(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "small.json"
    _write_json(src, [{"a": 1}, {"a": 2}])
    assert _run(["build", "--issue", "3", "--table", f"t={src}", "--out-dir", str(out_dir)]) == 0
    files = _manifest(out_dir, 3)[0]["files"]
    assert files == ["experiments/dashboards/issue3_t.html"]
    assert not list(out_dir.glob("issue3_t_p*.html"))


def test_multi_row_shards_never_exceed_shard_cap(repo: Path, out_dir: Path) -> None:
    # The real shard title is f"{prefix} ({sub})" (rendered in <title> AND
    # <h1>), so the per-shard overhead estimate must use that title shape — not
    # the bare prefix — or a tightly packed shard overshoots --shard-mb by ~2x
    # the subtitle-suffix length. Tiny rows fill each shard close to budget so
    # the overshoot would manifest; every produced shard page must stay <= cap.
    src = repo / "data" / "many.jsonl"
    _write_jsonl(src, [{"i": i} for i in range(2000)])
    shard_mb = 0.01  # 10_000 bytes
    rc = _run(
        [
            "build",
            "--issue",
            "667",
            "--table",
            f"t={src}",
            "--out-dir",
            str(out_dir),
            "--shard-mb",
            str(shard_mb),
        ]
    )
    assert rc == 0
    shard_bytes = int(shard_mb * 1_000_000)
    files = _manifest(out_dir, 667)[0]["files"]
    shards = [f for f in files if "_p" in Path(f).stem]
    assert len(shards) >= 2  # genuinely sharded, so at least one full shard
    for f in shards:
        page = out_dir / Path(f).name
        size = page.stat().st_size
        assert size <= shard_bytes, f"{f} is {size} bytes > shard cap {shard_bytes}"


# ─── emit-links ──────────────────────────────────────────────────────────


def test_emit_links_format(repo: Path, out_dir: Path, capsys: pytest.CaptureFixture) -> None:
    src = repo / "data" / "c.json"
    _write_json(src, [{"a": 1}])
    _run(["build", "--issue", "42", "--table", f"contexts={src}", "--out-dir", str(out_dir)])
    capsys.readouterr()  # drain the build summary print
    sha = "a" * 40
    rc = _run(["emit-links", "--issue", "42", "--sha", sha, "--out-dir", str(out_dir)])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == (
        "contexts: https://htmlpreview.github.io/?https://raw.githubusercontent.com/"
        "superkaiba/explore-persona-space/" + sha + "/experiments/dashboards/issue42_contexts.html"
    )


def test_emit_links_uses_index_for_sharded_table(
    repo: Path, out_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    src = repo / "data" / "big.jsonl"
    _write_jsonl(src, [{"i": i, "t": "z" * 300} for i in range(200)])
    _run(
        [
            "build",
            "--issue",
            "7",
            "--table",
            f"big={src}",
            "--out-dir",
            str(out_dir),
            "--shard-mb",
            "0.01",
        ]
    )
    capsys.readouterr()  # drain the build summary print
    _run(["emit-links", "--issue", "7", "--sha", "b" * 40, "--out-dir", str(out_dir)])
    out = capsys.readouterr().out.strip()
    # One line, pointing at the index page (not a shard).
    assert out.count("\n") == 0
    assert out.endswith("/experiments/dashboards/issue7_big.html")


@pytest.mark.parametrize("bad", ["abc", "g" * 40, "a" * 39, "a" * 41, ""])
def test_emit_links_rejects_bad_sha(repo: Path, out_dir: Path, bad: str) -> None:
    src = repo / "data" / "c.json"
    _write_json(src, [{"a": 1}])
    _run(["build", "--issue", "1", "--table", f"t={src}", "--out-dir", str(out_dir)])
    with pytest.raises(SystemExit):
        _run(["emit-links", "--issue", "1", "--sha", bad, "--out-dir", str(out_dir)])


def test_emit_links_missing_manifest_fails(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit):
        _run(["emit-links", "--issue", "999", "--sha", "a" * 40, "--out-dir", str(out_dir)])


# ─── Over-cap FAIL ───────────────────────────────────────────────────────


def test_over_cap_fails_loudly(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "huge.jsonl"
    # ~120 KB of text; force the cap to 0.05 MB (50 KB) so the build exceeds it.
    _write_jsonl(src, [{"i": i, "t": "y" * 1200} for i in range(100)])
    with pytest.raises(SystemExit) as exc:
        _run(
            [
                "build",
                "--issue",
                "667",
                "--table",
                f"huge={src}",
                "--out-dir",
                str(out_dir),
                "--max-payload-mb",
                "0.05",
            ]
        )
    assert "exceeds" in str(exc.value) and "HF data repo" in str(exc.value)
    # Nothing is written on an over-cap failure.
    assert not (out_dir / "issue667_manifest.json").exists()


# ─── Malformed input fails fast ──────────────────────────────────────────


def test_bad_table_spec_fails(repo: Path, out_dir: Path) -> None:
    with pytest.raises(SystemExit):
        _run(["build", "--issue", "1", "--table", "noequalssign", "--out-dir", str(out_dir)])


def test_unknown_extension_fails(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "x.txt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("nope", encoding="utf-8")
    with pytest.raises(SystemExit):
        _run(["build", "--issue", "1", "--table", f"t={src}", "--out-dir", str(out_dir)])


def test_json_not_a_list_fails(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "obj.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text('{"results": [1, 2]}', encoding="utf-8")
    with pytest.raises(SystemExit):
        _run(["build", "--issue", "1", "--table", f"t={src}", "--out-dir", str(out_dir)])


def test_non_object_record_fails(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "arr.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(SystemExit):
        _run(["build", "--issue", "1", "--table", f"t={src}", "--out-dir", str(out_dir)])


def test_duplicate_table_names_fail(repo: Path, out_dir: Path) -> None:
    a = repo / "data" / "a.json"
    b = repo / "data" / "b.json"
    _write_json(a, [{"x": 1}])
    _write_json(b, [{"x": 2}])
    with pytest.raises(SystemExit):
        _run(
            [
                "build",
                "--issue",
                "1",
                "--table",
                f"t={a}",
                "--table",
                f"t={b}",
                "--out-dir",
                str(out_dir),
            ]
        )


def test_missing_source_fails(repo: Path, out_dir: Path) -> None:
    with pytest.raises(SystemExit):
        _run(["build", "--issue", "1", "--table", f"t={repo}/nope.json", "--out-dir", str(out_dir)])


# ─── HTML escaping ───────────────────────────────────────────────────────


def test_html_is_escaped(repo: Path, out_dir: Path) -> None:
    src = repo / "data" / "xss.json"
    _write_json(src, [{"payload": "<script>alert(1)</script>"}])
    _run(["build", "--issue", "1", "--table", f"t={src}", "--out-dir", str(out_dir)])
    html = (out_dir / "issue1_t.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
