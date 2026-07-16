"""Tests for the NON-CANONICAL issue-keyed cache widening of the two disk
janitors (task #911): ``scripts/clean_experiment_downloads.py`` (P1/P2 /tmp +
P3 whole-dir data/ candidates behind the recency / nested-durable /
positive-evidence gates) and ``scripts/vm_disk_guard.py`` (tier (b) /tmp
discovery + structured dry-run reporting + the tier (d) /workspace hub-cache
reap).

HERMETIC BY CONSTRUCTION: every test builds its /tmp-shaped fixture under
pytest's ``tmp_path`` and passes it as an EXPLICIT ``tmp_root`` — the real
``/tmp`` is never read or written (that hermeticity is itself under test:
``test_hermeticity_no_tmp_root_skips_tmp`` pins that a ``tmp_root=None``
library call never reaches the /tmp discovery at all, and
``test_production_tmp_root_only_in_mains`` pins that the production opt-in
lives ONLY in the two CLI ``main()`` bodies). The #773 active-consumer gate is
stubbed to ``{}`` wherever a reap runs — the gate scans the LIVE tasks/ tree,
so an un-stubbed reap test is hostage to whatever task text mentions the
fixture's issue number today. HF listings are stubbed; no network.

Loaded via importlib like ``tests/test_vm_disk_guard.py`` (ced first —
vm_disk_guard imports it by module name at load time).
"""

import ast
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec (dataclass + future annotations)
    spec.loader.exec_module(mod)
    return mod


ced = _load("clean_experiment_downloads")
vdg = _load("vm_disk_guard")


# ─── fixtures / helpers ──────────────────────────────────────────────────────

AGED_TS = time.time() - 100 * 3600.0  # 100h ago — well past the 48h window


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Point both modules' repo_root at a temp dir (sidecar + escalation state
    resolve under it) and stub the #773 consumer gate (live-tasks-tree scan)
    to empty so reap outcomes depend only on the fixture."""
    monkeypatch.setattr(ced, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(vdg, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(ced, "_active_consumer_protected_issues", lambda n: {})
    return tmp_path


def _backdate(root: Path, ts: float = AGED_TS) -> None:
    """Backdate mtime+atime of ``root`` and everything under it (bottom-up,
    links included) so the recency gate reads the tree as aged."""
    for p in sorted(root.rglob("*"), key=lambda q: len(q.parts), reverse=True):
        os.utime(p, (ts, ts), follow_symlinks=False)
    os.utime(root, (ts, ts), follow_symlinks=False)


def _make_tmp_candidate(
    tmp_root: Path, name: str, *, hub_layout: bool = False, toplevel_dir: str | None = None
) -> Path:
    """A /tmp-shaped candidate dir: hub-layout (``blobs/``, evidence branch a)
    or a single top-level subdir (evidence branch b), else flat files."""
    cand = tmp_root / name
    if hub_layout:
        (cand / "blobs").mkdir(parents=True)
        (cand / "blobs" / "abc123").write_bytes(b"x" * 512)
    elif toplevel_dir is not None:
        (cand / toplevel_dir).mkdir(parents=True)
        (cand / toplevel_dir / "part0.json").write_text("{}")
    else:
        cand.mkdir(parents=True)
        (cand / "0af31c.json").write_text("{}")
        (cand / "9bd402.json").write_text("{}")
    return cand


def _read_sidecar(repo_path: Path) -> list[dict]:
    path = repo_path / ".claude" / "cache" / "disk-guard-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _stub_evidence_set(monkeypatch, names: set[str] | None) -> None:
    """Stub the one-per-run HF top-level listing (None = fetch failed)."""
    val = frozenset(names) if names is not None else None
    monkeypatch.setattr(ced, "_data_repo_toplevel_names", lambda: val)


# ─── 1+2: extract_issue_number ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("i779_mirror_dl", 779),
        ("i779_dl", 779),
        ("i778", 778),
        ("issue-841-r2", 841),
        ("issue744_smoke3", 744),
        ("issue_744_dl", 744),
        ("i841-rev-smoke", 841),
        ("issue779_jd", 779),
        ("i810_phasec_cache", 810),
        ("issue-825-onpolicy-smoke", 825),
        ("i545_argv_check", 545),
        ("i18n_cache", None),
        ("in2_foo", None),
        ("README", None),
        ("foo_bar", None),
    ],
)
def test_extract_issue_number_prefix_forms(name, expected):
    assert ced.extract_issue_number(name) == expected


def test_extract_issue_number_suffix_form():
    assert ced.extract_issue_number("fact_check_823") == 823
    assert ced.extract_issue_number("tmux-1000") is None  # hyphen — deliberate
    assert ced.extract_issue_number("foo_823_bar") is None  # not name-final
    # P1 precedence over P2: prefix i653 wins over the embedded 766.
    assert ced.extract_issue_number("i653_766_fixed") == 653


# ─── 3: the P3 data/ regex + N boundary ──────────────────────────────────────


def test_data_noncanonical_regex(tmp_path, repo):
    data_root = tmp_path / "data"
    for name in (
        "issue779_hfstage",
        "issue_744_dl",
        "issue810_phasec_cache",
        "issue722_skill_scratch",
        "issue295_marker_only_loss",
        "issue_744",
        "issue658_dl",
    ):
        (data_root / name).mkdir(parents=True)
    assert [p.name for p in ced.noncanonical_cache_dirs(779, data_root=data_root)] == [
        "issue779_hfstage"
    ]
    assert [p.name for p in ced.noncanonical_cache_dirs(744, data_root=data_root)] == [
        "issue_744_dl"
    ]
    assert [p.name for p in ced.noncanonical_cache_dirs(810, data_root=data_root)] == [
        "issue810_phasec_cache"
    ]
    # Non-cache-suffixed issue dirs never match (may hold generated data).
    assert ced.noncanonical_cache_dirs(722, data_root=data_root) == []
    assert ced.noncanonical_cache_dirs(295, data_root=data_root) == []
    # N boundary: issue658_dl never surfaces for issue 65.
    assert ced.noncanonical_cache_dirs(65, data_root=data_root) == []
    assert [p.name for p in ced.noncanonical_cache_dirs(658, data_root=data_root)] == [
        "issue658_dl"
    ]


# ─── 4: /tmp scan is dir-only + top-level-only ───────────────────────────────


def test_tmp_scan_dir_only_and_top_level_only(tmp_path, repo):
    tmp_root = tmp_path / "faketmp"
    tmp_root.mkdir()
    (tmp_root / "issue833-eval.tgz").write_bytes(b"x")  # file — never a candidate
    (tmp_root / "i653_766.py").write_text("pass")  # file — never a candidate
    (tmp_root / "sub").mkdir()
    (tmp_root / "sub" / "i779_x").mkdir()  # nested — never a candidate
    assert ced.noncanonical_cache_dirs(833, data_root=tmp_path / "data", tmp_root=tmp_root) == []
    assert ced.noncanonical_cache_dirs(653, data_root=tmp_path / "data", tmp_root=tmp_root) == []
    assert ced.noncanonical_cache_dirs(779, data_root=tmp_path / "data", tmp_root=tmp_root) == []


# ─── 5: uid ownership gate ───────────────────────────────────────────────────


def test_tmp_uid_gate(tmp_path, repo, monkeypatch):
    tmp_root = tmp_path / "faketmp"
    (tmp_root / "i900_dl").mkdir(parents=True)
    monkeypatch.setattr(ced, "_tmp_entry_owned", lambda p: False)
    assert ced.noncanonical_cache_dirs(900, data_root=tmp_path / "data", tmp_root=tmp_root) == []


# ─── 6: hermeticity (tmp_root=None never touches /tmp) + positive control ────


def test_hermeticity_no_tmp_root_skips_tmp(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    (data_root / "issue_901" / "hf_dl").mkdir(parents=True)
    (data_root / "issue_901" / "hf_dl" / "blob.bin").write_bytes(b"x" * 128)

    # (a) library discovery with tmp_root=None (data_root set AND both None)
    # yields ZERO /tmp candidates.
    assert ced.noncanonical_cache_dirs(901, data_root=data_root, tmp_root=None) == []
    assert ced.noncanonical_cache_dirs(901) == []  # both None (repo_root is stubbed)

    # (b)+(c) neither clean_terminal_download_caches nor the exact existing
    # run_guard(apply=True, data_root=temp) library shape ever reaches the
    # /tmp discovery or the /workspace tier when tmp_root is None.
    def _boom(*a, **k):
        raise AssertionError("must not be reached without an explicit tmp_root")

    monkeypatch.setattr(vdg, "_discover_tmp_issue_numbers", _boom)
    monkeypatch.setattr(vdg, "clean_vm_workspace_hf_cache", _boom)
    monkeypatch.setattr(vdg, "clean_home_hf_cache", _boom)  # tier (e), #1376
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    vdg.clean_terminal_download_caches(apply=True, data_root=data_root)

    (data_root / "issue_901" / "hf_dl").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(vdg, "clean_uv_cache", lambda apply: vdg.TierResult(name="uv-cache"))
    monkeypatch.setattr(vdg, "clean_stale_logs", lambda *a, **k: vdg.TierResult(name="stale-logs"))
    state = {"calls": 0}

    def fake_used(path="/"):
        state["calls"] += 1
        return 90.0 if state["calls"] == 1 else 40.0

    monkeypatch.setattr(vdg, "disk_used_pct", fake_used)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": 50.0)
    res = vdg.run_guard(apply=True, threshold=85.0, data_root=data_root)
    assert res.triggered is True  # the guard ran its tiers, hermetically

    # (d) POSITIVE CONTROL (no vacuous pass): the identical call WITH an
    # explicit fixture tmp_root surfaces + reaps a planted aged
    # evidence-bearing candidate.
    tmp_root = tmp_path / "faketmp"
    tmp_root.mkdir()
    planted = _make_tmp_candidate(tmp_root, "i901_mirror_dl", hub_layout=True)
    _backdate(planted)
    monkeypatch.setattr(
        vdg, "_discover_tmp_issue_numbers", lambda r: [901] if r == tmp_root else []
    )
    # With an explicit tmp_root, tiers (d) + (e) legitimately fire — stub them
    # to no-ops here (their own tests use a fixture cache_root; the REAL
    # /workspace and ~/.cache/huggingface caches must never be touched from
    # pytest, #1376).
    monkeypatch.setattr(
        vdg,
        "clean_vm_workspace_hf_cache",
        lambda apply, **k: vdg.TierResult(name="workspace-hf-cache"),
    )
    monkeypatch.setattr(
        vdg,
        "clean_home_hf_cache",
        lambda apply, **k: vdg.TierResult(name="home-hf-cache"),
    )
    state["calls"] = 0
    res2 = vdg.run_guard(apply=True, threshold=85.0, data_root=data_root, tmp_root=tmp_root)
    assert res2.triggered is True
    assert not planted.exists(), "planted aged evidence-bearing /tmp candidate must be reaped"


# ─── 7: kill switch ──────────────────────────────────────────────────────────


def test_kill_switch_env(tmp_path, repo, monkeypatch):
    tmp_root = tmp_path / "faketmp"
    _make_tmp_candidate(tmp_root, "i902_dl", hub_layout=True)
    (tmp_path / "data" / "issue902_hfstage").mkdir(parents=True)
    monkeypatch.setenv("EPM_SKIP_NONCANONICAL_CACHE_SWEEP", "1")
    assert ced.noncanonical_cache_dirs(902, data_root=tmp_path / "data", tmp_root=tmp_root) == []


# ─── 8: the reap path (aged + evidence-bearing candidates) ───────────────────


def test_reap_noncanonical_tmp_and_data_for_issue(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    tmp_root = tmp_path / "faketmp"
    # P1: hub-layout /tmp candidate (evidence branch a — needs NO repo listing).
    p1 = _make_tmp_candidate(tmp_root, "i903_mirror_dl", hub_layout=True)
    # P2: suffix-form /tmp dir carrying a data-repo-prefix top-level (branch b).
    p2 = _make_tmp_candidate(tmp_root, "fact_check_903", toplevel_dir="issue903_monitoring")
    # P3: whole-dir data/ cache carrying a data-repo-prefix top-level.
    p3 = data_root / "issue_903_dl"
    (p3 / "issue903_token_continuity").mkdir(parents=True)
    (p3 / "issue903_token_continuity" / "shard0.json").write_text("{}")
    # Siblings that must survive: store/, eval_results/, a non-cache-named dir.
    issue_dir = data_root / "issue_903"
    (issue_dir / "store").mkdir(parents=True)
    (issue_dir / "store" / "gen.pt").write_bytes(b"y" * 64)
    (issue_dir / "eval_results").mkdir(parents=True)
    (issue_dir / "eval_results" / "m.json").write_text("{}")
    scratch = data_root / "issue903_scratch"
    scratch.mkdir(parents=True)
    (scratch / "z.bin").write_bytes(b"z")
    for root in (tmp_root, data_root):
        _backdate(root)
    _stub_evidence_set(monkeypatch, {"issue903_monitoring", "issue903_token_continuity"})

    res = ced.clean_issue_downloads(903, apply=True, data_root=data_root, tmp_root=tmp_root)

    assert not p1.exists() and not p2.exists() and not p3.exists()
    assert sorted(res.noncanonical_dispositions.values()) == ["removed", "removed", "removed"]
    # Every removed non-canonical candidate carries a positive-evidence string.
    for rel in res.removed:
        assert res.noncanonical_evidence.get(rel), f"reaped without evidence: {rel}"
    # Siblings untouched (I1).
    assert (issue_dir / "store" / "gen.pt").is_file()
    assert (issue_dir / "eval_results" / "m.json").is_file()
    assert (scratch / "z.bin").is_file()


def test_hub_layout_candidate_needs_no_repo_listing(tmp_path, repo, monkeypatch):
    """Branch (a) evidence licenses the reap with zero HF fetches."""
    tmp_root = tmp_path / "faketmp"
    p1 = _make_tmp_candidate(tmp_root, "i904_hf_stage", hub_layout=True)
    _backdate(tmp_root)

    def _no_fetch():
        raise AssertionError("hub-layout candidate must not fetch the repo listing")

    monkeypatch.setattr(ced, "_data_repo_toplevel_names", _no_fetch)
    res = ced.clean_issue_downloads(904, apply=True, data_root=tmp_path / "data", tmp_root=tmp_root)
    assert not p1.exists()
    assert list(res.noncanonical_dispositions.values()) == ["removed"]


# ─── 8b: nested eval_results/ (or store/) blocks the whole-dir reap (I12) ────


def test_nested_eval_results_blocks_noncanonical_reap(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    tmp_root = tmp_path / "faketmp"
    t1 = tmp_root / "i905_dump"
    (t1 / "eval_results").mkdir(parents=True)
    (t1 / "eval_results" / "metrics.json").write_text("{}")
    t2 = tmp_root / "i905_dump2"  # the nested-store sibling case, folded in
    (t2 / "store").mkdir(parents=True)
    (t2 / "store" / "gen.pt").write_bytes(b"y" * 64)
    p3 = data_root / "issue905_hfstage"
    (p3 / "eval_results").mkdir(parents=True)
    (p3 / "eval_results" / "metrics.json").write_text("{}")
    for root in (tmp_root, data_root):
        _backdate(root)
    _stub_evidence_set(monkeypatch, {"eval_results", "store"})  # even a listing match won't help

    res = ced.clean_issue_downloads(905, apply=True, data_root=data_root, tmp_root=tmp_root)

    assert (t1 / "eval_results" / "metrics.json").is_file()
    assert (t2 / "store" / "gen.pt").is_file()
    assert (p3 / "eval_results" / "metrics.json").is_file()
    assert res.removed == []
    assert set(res.noncanonical_dispositions.values()) == {"durable-content-kept"}
    kinds = [r["kind"] for r in _read_sidecar(repo)]
    assert kinds.count("noncanonical-cache-durable-content-kept") == 3


# ─── 8c: the positive-evidence gate (I11) ────────────────────────────────────


def test_positive_evidence_gate(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    tmp_root = tmp_path / "faketmp"
    flat = _make_tmp_candidate(tmp_root, "i906_out")  # flat hash-named JSONs
    _backdate(tmp_root)
    _stub_evidence_set(monkeypatch, {"issue906_monitoring"})

    res = ced.clean_issue_downloads(906, apply=True, data_root=data_root, tmp_root=tmp_root)
    assert flat.exists(), "predicate-failing candidate must never be deleted"
    assert list(res.noncanonical_dispositions.values()) == ["unverified-kept"]
    rel = next(iter(res.noncanonical_dispositions))
    assert res.sizes_bytes.get(rel, 0) > 0  # counted in total_discovered_bytes
    assert res.total_discovered_bytes > 0
    kinds = [r["kind"] for r in _read_sidecar(repo)]
    assert "noncanonical-cache-unverified-kept" in kinds

    # The same dir WITH a data-repo-prefix top-level IS removed.
    mirror = _make_tmp_candidate(tmp_root, "i907_stage", toplevel_dir="issue907_monitoring")
    _backdate(tmp_root)
    _stub_evidence_set(monkeypatch, {"issue907_monitoring"})
    res2 = ced.clean_issue_downloads(907, apply=True, data_root=data_root, tmp_root=tmp_root)
    assert not mirror.exists()
    assert list(res2.noncanonical_dispositions.values()) == ["removed"]

    # Repo-listing fetch failure (None) => fail-toward-keep.
    kept = _make_tmp_candidate(tmp_root, "i908_stage", toplevel_dir="issue908_monitoring")
    _backdate(tmp_root)
    _stub_evidence_set(monkeypatch, None)
    res3 = ced.clean_issue_downloads(908, apply=True, data_root=data_root, tmp_root=tmp_root)
    assert kept.exists()
    assert list(res3.noncanonical_dispositions.values()) == ["unverified-kept"]


def test_positive_evidence_ignores_hub_local_dir_bookkeeping(tmp_path, repo, monkeypatch):
    """The live ``data/issue_744_dl`` shape: a mirrored data-repo prefix PLUS
    the ``snapshot_download(local_dir=...)`` ``.cache/`` bookkeeping dir —
    branch (b) ignores the bookkeeping entry and licenses the reap; a dir
    holding ONLY the bookkeeping dir carries no branch-(b) evidence."""
    data_root = tmp_path / "data"
    tmp_root = tmp_path / "faketmp"
    tmp_root.mkdir()
    mirror = data_root / "issue919_dl"
    (mirror / "issue919_token_continuity").mkdir(parents=True)
    (mirror / "issue919_token_continuity" / "shard0.json").write_text("{}")
    (mirror / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (mirror / ".cache" / "huggingface" / "download" / "x.metadata").write_text("m")
    only_cache = data_root / "issue921_dl"
    (only_cache / ".cache" / "huggingface").mkdir(parents=True)
    (only_cache / ".cache" / "huggingface" / "y.metadata").write_text("m")
    _backdate(data_root)
    _stub_evidence_set(monkeypatch, {"issue919_token_continuity"})

    res = ced.clean_issue_downloads(919, apply=True, data_root=data_root, tmp_root=tmp_root)
    assert not mirror.exists()
    assert list(res.noncanonical_dispositions.values()) == ["removed"]

    res2 = ced.clean_issue_downloads(921, apply=True, data_root=data_root, tmp_root=tmp_root)
    assert only_cache.exists()  # bookkeeping-only dir: no evidence -> kept
    assert list(res2.noncanonical_dispositions.values()) == ["unverified-kept"]


# ─── 8c-bis: r2 regressions — P2 empty-dir license + hidden-dir marker spoof ─


def test_p2_suffix_only_empty_dir_never_reap_licensed(tmp_path, repo, monkeypatch):
    """r2 fix (concern ``p2-empty-tempdir-false-reap``): an aged, EMPTY,
    uid-owned FOREIGN mkdtemp leftover (``tmpabc_7``) whose ``_(\\d+)$``
    suffix collides with a real terminal issue must NEVER ride the empty-dir
    evidence license — it is kept + escalated (``unverified-kept``), on disk
    intact, through the guard tier-(b) path. POSITIVE CONTROL in the same
    run: an aged empty P1-named dir keeps the empty-dir license unchanged."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    tmp_root = tmp_path / "faketmp"
    foreign = tmp_root / "tmpabc_7"  # P2-suffix-only: no P1/P3 route
    foreign.mkdir(parents=True)
    p1_empty = tmp_root / "i7_stage"  # P1 route: empty-dir license unchanged
    p1_empty.mkdir()
    _backdate(tmp_root)
    _stub_evidence_set(monkeypatch, set())
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")

    res = vdg.clean_terminal_download_caches(apply=True, data_root=data_root, tmp_root=tmp_root)

    assert foreign.is_dir(), "empty P2-suffix-only dir must never be deleted"
    assert not p1_empty.exists(), "P1 empty-dir license must be unchanged"
    by_name = {os.path.basename(r["path"]): r for r in res.noncanonical_candidates}
    assert by_name["tmpabc_7"]["disposition"] == "unverified-kept"
    assert by_name["tmpabc_7"]["issue"] == 7
    assert by_name["i7_stage"]["disposition"] == "removed"
    kinds = [r["kind"] for r in _read_sidecar(repo)]
    assert "noncanonical-cache-unverified-kept" in kinds


def test_hidden_git_dir_not_hub_evidence(tmp_path, repo, monkeypatch):
    """r2 fix (concern ``evidence-scan-hidden-dir-collision``): ``.git/refs``
    inside an aged, terminal-owned, issue-keyed /tmp git checkout must NOT
    spoof a hub-layout marker (``refs`` at depth 2) — hidden entries are
    skipped at every depth of the branch-(a) scan, so both checkout shapes
    are ``unverified-kept`` and stay intact."""
    tmp_root = tmp_path / "faketmp"
    checkout = tmp_root / "i916_checkout"  # .git + uncommitted work
    (checkout / ".git" / "refs" / "heads").mkdir(parents=True)
    (checkout / ".git" / "refs" / "heads" / "main").write_text("abc123\n")
    (checkout / ".git" / "config").write_text("[core]\n")
    (checkout / "notes.txt").write_text("uncommitted work\n")
    gitonly = tmp_root / "i916_gitonly"  # the reviewer-sketched only-.git shape
    (gitonly / ".git" / "refs").mkdir(parents=True)
    (gitonly / ".git" / "refs" / "HEAD").write_text("ref: refs/heads/main\n")
    _backdate(tmp_root)
    _stub_evidence_set(monkeypatch, {"issue916_monitoring"})

    res = ced.clean_issue_downloads(916, apply=True, data_root=tmp_path / "data", tmp_root=tmp_root)

    assert (checkout / ".git" / "refs" / "heads" / "main").is_file()
    assert (checkout / "notes.txt").is_file()
    assert (gitonly / ".git" / "refs" / "HEAD").is_file()
    assert res.removed == []
    assert list(res.noncanonical_dispositions.values()) == [
        "unverified-kept",
        "unverified-kept",
    ]
    for rel in res.noncanonical_evidence.values():
        assert "refs" not in rel  # no spoofed hub-layout marker string


# ─── 8d: production_tmp_root referenced ONLY inside the two main() bodies ────


class _TmpRootRefVisitor(ast.NodeVisitor):
    """Collects every Name/Attribute reference to ``production_tmp_root``
    together with its enclosing-function stack (a FunctionDef's own name is a
    plain attribute, not a Name node, so the definition never self-reports)."""

    def __init__(self):
        self.stack: list[str] = []
        self.refs: list[tuple[str, ...]] = []

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Name(self, node):
        if node.id == "production_tmp_root":
            self.refs.append(tuple(self.stack))

    def visit_Attribute(self, node):
        if node.attr == "production_tmp_root":
            self.refs.append(tuple(self.stack))
        self.generic_visit(node)


def test_production_tmp_root_only_in_mains():
    for script in ("clean_experiment_downloads.py", "vm_disk_guard.py"):
        visitor = _TmpRootRefVisitor()
        visitor.visit(ast.parse((_SCRIPTS / script).read_text()))
        for stack in visitor.refs:
            assert stack and stack[-1] == "main", (
                f"{script}: production_tmp_root referenced outside main(): "
                f"{' > '.join(stack) or '<module>'}"
            )


def test_cli_main_forwards_tmp_root_to_cleaner(tmp_path, monkeypatch):
    """r2 Minor: pin that ced ``main()``'s signature-adaptive dispatch
    actually FORWARDS ``production_tmp_root()`` into the production cleaners
    (default AND ``--incremental``). The I7 source-scan test pins WHERE
    ``production_tmp_root`` may be referenced, not that its value reaches the
    cleaner — the ``inspect.signature`` dispatch could silently drop the CLI
    opt-in on a future rename with every other test still green."""
    fake_root = tmp_path / "prod_tmp"
    calls: list[dict] = []

    def _spy(issue_n, *, apply=False, data_root=None, tmp_root=None, sweep_tmp=True):
        calls.append({"issue_n": issue_n, "apply": apply, "tmp_root": tmp_root})
        return SimpleNamespace(
            removed=[],
            skipped=[],
            symlink_external_kept=[],
            failed=[],
            bytes_freed=0,
            sizes_bytes={},
        )

    monkeypatch.setattr(ced, "_running_pod_side", lambda: False)
    monkeypatch.setattr(ced, "production_tmp_root", lambda: fake_root)
    monkeypatch.setattr(ced, "clean_issue_downloads", _spy)
    monkeypatch.setattr(ced, "clean_issue_downloads_incremental", _spy)

    assert ced.main(["903"]) == 0
    assert ced.main(["903", "--incremental"]) == 0
    assert [c["tmp_root"] for c in calls] == [fake_root, fake_root]
    assert [c["issue_n"] for c in calls] == [903, 903]
    assert all(c["apply"] is False for c in calls)  # dry-run default preserved


# ─── 9: recency gate (I4) ────────────────────────────────────────────────────


def test_recency_gate_keeps_fresh_dir(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    tmp_root = tmp_path / "faketmp"
    fresh = _make_tmp_candidate(tmp_root, "i909_dl", hub_layout=True)  # mtime = now
    _stub_evidence_set(monkeypatch, set())

    # Dry-run: intact, disposition recency-kept, NO sidecar row (apply-gated).
    res_dry = ced.clean_issue_downloads(909, apply=False, data_root=data_root, tmp_root=tmp_root)
    assert fresh.exists()
    assert list(res_dry.noncanonical_dispositions.values()) == ["recency-kept"]
    assert _read_sidecar(repo) == []

    # Apply: still intact + sidecar row with the dedicated kind.
    res = ced.clean_issue_downloads(909, apply=True, data_root=data_root, tmp_root=tmp_root)
    assert fresh.exists()
    assert res.removed == []
    assert list(res.noncanonical_dispositions.values()) == ["recency-kept"]
    _rel, reason = res.skipped[0]
    assert "recency window" in reason
    kinds = [r["kind"] for r in _read_sidecar(repo)]
    assert kinds == ["noncanonical-cache-recent-kept"]


# ─── 10: nested unmirrored store/ under a /tmp candidate blocks (I5) ─────────


def test_nested_store_parity_blocks_tmp_reap(tmp_path, repo, monkeypatch):
    tmp_root = tmp_path / "faketmp"
    cand = tmp_root / "i910_dl"
    (cand / "store").mkdir(parents=True)
    (cand / "store" / "gen.pt").write_bytes(b"y" * 64)
    (cand / "blobs").mkdir()  # even hub-layout evidence must not override
    _backdate(tmp_root)
    # HF listing unavailable — under v4 the nested-durable gate blocks OUTRIGHT
    # (no mirror check attempted), so this monkeypatch is defense-in-depth.
    monkeypatch.setattr(ced, "_hf_file_sizes", lambda repo_id, revision="main": None)
    _stub_evidence_set(monkeypatch, set())

    res = ced.clean_issue_downloads(910, apply=True, data_root=tmp_path / "data", tmp_root=tmp_root)
    assert (cand / "store" / "gen.pt").is_file()
    assert res.removed == []
    assert list(res.noncanonical_dispositions.values()) == ["durable-content-kept"]


# ─── 11: nested candidates are handled topmost-only (I10) ────────────────────


def test_dedup_nested_topmost_only(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    whole = data_root / "issue912_dl"  # P3 whole-dir candidate (an issue<N>_<slug> dir)
    (whole / "hf_dl").mkdir(parents=True)  # ...which ALSO contains a canonical hf_dl
    (whole / "hf_dl" / "blob.bin").write_bytes(b"x" * 256)
    (whole / "blobs").mkdir()  # hub-layout evidence for the whole dir
    _backdate(data_root)
    _stub_evidence_set(monkeypatch, set())

    res = ced.clean_issue_downloads(912, apply=True, data_root=data_root, tmp_root=None)
    assert not whole.exists()
    assert len(res.removed) == 1  # ONE removal: the topmost dir only
    assert res.failed == []  # no phantom rmtree-child-after-parent failure


# ─── 12: symlinked /tmp candidate takes the #915 disposition (I6) ────────────


def test_symlinked_tmp_candidate_external_target_kept(tmp_path, repo, monkeypatch):
    tmp_root = tmp_path / "faketmp"
    tmp_root.mkdir()
    external = tmp_path / "external"
    (external / "blobs").mkdir(parents=True)
    (external / "blobs" / "abc").write_bytes(b"x" * 128)
    link = tmp_root / "i913_dl"
    link.symlink_to(external)
    _backdate(external)
    os.utime(link, (AGED_TS, AGED_TS), follow_symlinks=False)
    _stub_evidence_set(monkeypatch, set())

    res = ced.clean_issue_downloads(913, apply=True, data_root=tmp_path / "data", tmp_root=tmp_root)
    # External target KEPT; the direct link itself unlinked (#915 disposition).
    assert (external / "blobs" / "abc").is_file()
    assert not link.is_symlink()
    assert len(res.symlink_external_kept) == 1
    assert list(res.noncanonical_dispositions.values()) == ["external-target-kept"]


# ─── 13-15: tier (b) — /tmp-only discovery + terminal/active/unresolved ──────


def _patch_disk(monkeypatch, before_pct, after_pct, free_gb=50.0):
    state = {"calls": 0}

    def fake_used(path="/"):
        state["calls"] += 1
        return before_pct if state["calls"] == 1 else after_pct

    monkeypatch.setattr(vdg, "disk_used_pct", fake_used)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": free_gb)


def test_tier_b_discovers_tmp_only_issue_and_reaps_terminal(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    tmp_root = tmp_path / "faketmp"
    cand = _make_tmp_candidate(tmp_root, "fact_check_914", toplevel_dir="issue914_monitoring")
    _backdate(tmp_root)
    _stub_evidence_set(monkeypatch, {"issue914_monitoring"})
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")

    res = vdg.clean_terminal_download_caches(apply=True, data_root=data_root, tmp_root=tmp_root)
    assert not cand.exists()
    assert res.bytes_freed > 0
    rows = res.noncanonical_candidates
    assert len(rows) == 1 and rows[0]["disposition"] == "removed" and rows[0]["issue"] == 914
    assert rows[0]["evidence"]


def test_tier_b_active_issue_tmp_escalated_not_deleted(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    (data_root / "issue_915" / "hf_dl").mkdir(parents=True)
    (data_root / "issue_915" / "hf_dl" / "blob.bin").write_bytes(b"x" * 1024)
    tmp_root = tmp_path / "faketmp"
    cand = _make_tmp_candidate(tmp_root, "i915_mirror_dl", hub_layout=True)
    _backdate(tmp_root)
    _stub_evidence_set(monkeypatch, set())
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")

    res = vdg.clean_terminal_download_caches(apply=True, data_root=data_root, tmp_root=tmp_root)
    assert cand.exists()  # never deleted while active (I2)
    assert (data_root / "issue_915" / "hf_dl").is_dir()
    assert res.bytes_freed == 0
    # The sizing attribution covers canonical + /tmp bytes, dedup-independent.
    assert len(res.active_cache_attributions) == 1
    row = res.active_cache_attributions[0]
    assert row["task"] == 915 and row["status"] == "running"
    assert row["bytes"] >= 1024 + 512  # hf_dl blob + hub blob
    assert [r["disposition"] for r in res.noncanonical_candidates] == ["escalated"]


def test_tier_b_unresolved_issue_kept(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    tmp_root = tmp_path / "faketmp"
    cand = _make_tmp_candidate(tmp_root, "i916_dl", hub_layout=True)
    _backdate(tmp_root)
    _stub_evidence_set(monkeypatch, set())
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: None)

    res = vdg.clean_terminal_download_caches(apply=True, data_root=data_root, tmp_root=tmp_root)
    assert cand.exists()  # unresolved -> kept (I3)
    assert res.bytes_freed == 0
    assert [r["disposition"] for r in res.noncanonical_candidates] == ["unresolved-kept"]


# ─── 16-17 + 21: tier (d) — /workspace hub cache ─────────────────────────────


def test_workspace_tier_pod_guards(tmp_path, monkeypatch):
    monkeypatch.setattr("os.path.ismount", lambda p: p == "/workspace")
    res = vdg.clean_vm_workspace_hf_cache(apply=True, cache_root=tmp_path)
    assert res.skipped and "mount" in res.skip_reason

    monkeypatch.setattr("os.path.ismount", lambda p: False)
    monkeypatch.setattr(vdg, "_running_pod_side", lambda: True)
    res2 = vdg.clean_vm_workspace_hf_cache(apply=True, cache_root=tmp_path)
    assert res2.skipped and "pod-side" in res2.skip_reason


def _fake_cache_info(now: float, executed: list):
    stale = SimpleNamespace(
        repo_id="org/stale-model",
        last_accessed=now - 20 * 86400.0,
        revisions=[SimpleNamespace(commit_hash="aaa111")],
    )
    fresh = SimpleNamespace(
        repo_id="org/fresh-model",
        last_accessed=now - 1 * 86400.0,
        revisions=[SimpleNamespace(commit_hash="bbb222")],
    )

    def delete_revisions(*hashes):
        return SimpleNamespace(
            expected_freed_size=12345,
            requested=hashes,
            execute=lambda: executed.append(hashes),
        )

    return SimpleNamespace(repos=[stale, fresh], delete_revisions=delete_revisions)


def test_workspace_tier_age_gate(tmp_path, monkeypatch):
    (tmp_path / "hub").mkdir()
    now = time.time()
    executed: list = []
    monkeypatch.setattr("os.path.ismount", lambda p: False)
    monkeypatch.setattr(vdg, "_running_pod_side", lambda: False)
    monkeypatch.setattr(vdg, "_scan_hf_cache", lambda hub: _fake_cache_info(now, executed))

    res = vdg.clean_vm_workspace_hf_cache(
        apply=False, max_age_days=14.0, cache_root=tmp_path, now=now
    )
    assert res.skipped is False
    assert res.bytes_freed == 12345  # expected_freed_size reported
    assert executed == []  # dry-run executes NOTHING
    assert any("stale-model" in d for d in res.detail)
    assert not any("fresh-model" in d for d in res.detail)  # only the stale repo targeted

    res2 = vdg.clean_vm_workspace_hf_cache(
        apply=True, max_age_days=14.0, cache_root=tmp_path, now=now
    )
    assert executed == [("aaa111",)]  # ONLY the stale repo's revisions
    assert res2.bytes_freed == 12345


def test_workspace_tier_failure_degrades_to_skipped(tmp_path, monkeypatch):
    (tmp_path / "hub").mkdir()
    monkeypatch.setattr("os.path.ismount", lambda p: False)
    monkeypatch.setattr(vdg, "_running_pod_side", lambda: False)

    def _boom(hub):
        raise ImportError("huggingface_hub gone")

    monkeypatch.setattr(vdg, "_scan_hf_cache", _boom)
    res = vdg.clean_vm_workspace_hf_cache(apply=True, cache_root=tmp_path)
    assert res.skipped and "ImportError" in res.skip_reason

    # An execute()-time failure degrades the same way, deleting nothing.
    now = time.time()

    def _bad_info(hub):
        info = _fake_cache_info(now, [])

        def delete_revisions(*hashes):
            return SimpleNamespace(
                expected_freed_size=1,
                execute=lambda: (_ for _ in ()).throw(OSError("corrupt cache")),
            )

        info.delete_revisions = delete_revisions
        return info

    monkeypatch.setattr(vdg, "_scan_hf_cache", _bad_info)
    res2 = vdg.clean_vm_workspace_hf_cache(apply=True, cache_root=tmp_path, now=now)
    assert res2.skipped and "OSError" in res2.skip_reason


# ─── 18: dry-run removes nothing on the new patterns (I8) ────────────────────


def test_dry_run_removes_nothing_new_patterns(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    tmp_root = tmp_path / "faketmp"
    p1 = _make_tmp_candidate(tmp_root, "i917_mirror_dl", hub_layout=True)
    p3 = data_root / "issue917_hfstage"
    (p3 / "issue917_monitoring").mkdir(parents=True)
    (p3 / "issue917_monitoring" / "s.json").write_text("{}")
    for root in (tmp_root, data_root):
        _backdate(root)
    _stub_evidence_set(monkeypatch, {"issue917_monitoring"})

    res = ced.clean_issue_downloads(917, apply=False, data_root=data_root, tmp_root=tmp_root)
    assert p1.exists() and p3.exists()  # NOTHING deleted
    assert len(res.removed) == 2  # both reported as would-remove
    assert sorted(res.noncanonical_dispositions.values()) == ["would-remove", "would-remove"]
    assert _read_sidecar(repo) == []  # dry-run persists nothing


# ─── 19: structured dry-run attribution rides the guard JSON ─────────────────


def test_dry_run_json_carries_structured_attribution(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    tmp_root = tmp_path / "faketmp"
    _make_tmp_candidate(tmp_root, "i918_mirror_dl", hub_layout=True)
    _backdate(tmp_root)
    _stub_evidence_set(monkeypatch, set())
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")
    # Pre-suppress the alert channel: ack sentinel for every band + tiny size
    # (below the 5 GB alert floor) — the structured rows must STILL appear.
    ack = vdg._active_ack_sentinel_path(918, 0.0)
    ack.parent.mkdir(parents=True, exist_ok=True)
    ack.touch()

    res = vdg.clean_terminal_download_caches(apply=False, data_root=data_root, tmp_root=tmp_root)
    assert len(res.active_cache_attributions) == 1
    row = res.active_cache_attributions[0]
    assert row["task"] == 918 and row["status"] == "running" and row["bytes"] > 0
    assert [r["disposition"] for r in res.noncanonical_candidates] == ["escalated"]
    assert res.total_discovered_bytes > 0
    assert _read_sidecar(repo) == []  # report-only persists NOTHING

    # The guard's --json payload surfaces the fields (additive schema).
    guard = vdg.GuardResult(
        used_pct_before=90.0,
        used_pct_after=90.0,
        free_gb_before=10.0,
        free_gb_after=10.0,
        threshold_pct=85.0,
        triggered=True,
        apply=False,
        tiers=[res],
    )
    payload = vdg._result_json(guard)
    assert payload["total_discovered_bytes"] == res.total_discovered_bytes
    tier = payload["tiers"][0]
    assert tier["active_cache_attributions"] == res.active_cache_attributions
    assert tier["noncanonical_candidates"] == res.noncanonical_candidates
    assert tier["total_discovered_bytes"] == res.total_discovered_bytes


# ─── 20: the data-disk pass never sweeps /tmp ────────────────────────────────


def test_data_disk_pass_never_sweeps_tmp(tmp_path, repo, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()

    def _boom(*a, **k):
        raise AssertionError("the escalate-only data-disk pass must never sweep /tmp")

    monkeypatch.setattr(vdg, "_discover_tmp_issue_numbers", _boom)
    monkeypatch.setattr(vdg, "clean_vm_workspace_hf_cache", _boom)
    monkeypatch.setattr(vdg, "clean_home_hf_cache", _boom)  # tier (e), #1376
    _patch_disk(monkeypatch, before_pct=96.0, after_pct=96.0)
    res = vdg.run_guard(
        apply=True,
        threshold=85.0,
        data_root=data_root,
        disk_path="/",
        reclaim_tiers=False,
        tmp_root=None,  # main() passes None on the data-disk pass
    )
    assert res.triggered is True
    assert {t.name for t in res.tiers} == {"terminal-download-caches"}
