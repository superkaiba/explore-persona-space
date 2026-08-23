"""Round-3 code-review fix pins for issue #2329 — C1: width-scoped sweeps.

r2 C1 (both reviewers, independently): the round-2 F3 stale-width sweep at
``phase_upload`` entry ran at the CPU process's ``--num-workers`` (implicitly
1 — the dispatcher's ``run_cpu_phase`` never threaded it), quarantining every
w1..wN LIVE anchor/margin shard + done record before the bulk upload; the
exact-set verify then passed on the POST-sweep set — a silent self-consistent
truncation of ~3/4 to 7/8 of the store. Second live site (Codex twin):
``phase_margin`` applied ONE phase's width to EVERY family, so the registered
deferred 1x H100 margin leg would have destroyed a valid 8-wide anchor family.

Pins (all five required by the round-3 fix list):

- dispatch.sh pin: the upload invocation threads ``--num-workers`` (fix (c)).
- unit pin: no destructive sweep runs on an implicitly-defaulted width
  (fix (d): ``--num-workers`` defaults to None; ``_entry_sweep`` skips).
- INTEGRATION 1: a full-width anchors+margin out-root SURVIVES a real
  ``phase_upload`` intact (width-less AND narrow-explicit invocations) and
  the mirrored upload set is complete — the exact regression scenario.
- INTEGRATION 2: 8-wide anchors + a 1-wide deferred ``margin`` leg leaves the
  anchor family UNTOUCHED (family scoping, fix (a)).
- F3-purpose keep: the sweep still removes genuinely stale prior-width shards
  after a real width change (upload site, width DERIVED per fix (b)).

Plus AST call-site pins binding the tested helpers to the dispatched phase
entries (the round-2 test called the helper directly at width 4, which is
exactly how both integration paths escaped).

CPU-only, network-free (``--upload local-mirror``), repo-root-path-free.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2329_run as R  # noqa: E402

# ── fixture builders ──────────────────────────────────────────────────


def _done_rec(worker_index: int, num_workers: int) -> dict:
    return {
        "regime_fp": "fp",
        "worker_index": worker_index,
        "num_workers": num_workers,
        "n_contexts": 1,
        "draws": 2,
        "n_rows": 1,
        "n_cap_hit": 0,
        "n_empty": 0,
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _make_store(out_root: Path, width: int, families: tuple[str, ...] = ("anchors", "margin")):
    """A complete width-W anchors/margin store: shards + width-stamped done
    records for every worker index. Returns {family: [paths]}."""
    anchors = out_root / "anchors"
    margin = out_root / "margin"
    manifests = out_root / "manifests"
    made: dict[str, list[Path]] = {f: [] for f in families}
    for i in range(width):
        if "anchors" in families:
            for p in (
                anchors / f"anchors_gate_w{i}.jsonl",
                anchors / f"anchors_rest_w{i}.jsonl",
                anchors / f"va_anchors_gate_w{i}.pt",
                anchors / f"va_anchors_rest_w{i}.pt",
            ):
                _write(p, "x")
                made["anchors"].append(p)
            for kind in ("anchors_gate", "anchors_rest"):
                p = manifests / f"{kind}_w{i}_done.json"
                _write(p, json.dumps(_done_rec(i, width)))
                made["anchors"].append(p)
        if "margin" in families:
            p = margin / f"anchor_margin_w{i}.jsonl"
            _write(p, "x")
            made["margin"].append(p)
            p = manifests / f"margin_anchors_w{i}_done.json"
            _write(p, json.dumps(_done_rec(i, width)))
            made["margin"].append(p)
    return made


def _cfg(out_root: Path, log_dir: Path, extra: list[str]) -> R.RunConfig:
    """Build a RunConfig through the REAL argparse -> build_config path (the
    round-2 escape mode was a helper-direct call that bypassed it)."""
    argv = ["--out-root", str(out_root), "--log-dir", str(log_dir), *extra]
    return R.build_config(R.parse_args(argv))


def _no_quarantined_files(out_root: Path) -> bool:
    q = out_root / "stale_width_quarantine"
    return not q.exists() or not any(p.is_file() for p in q.rglob("*"))


# ── dispatch.sh pin: upload threads --num-workers (fix (c)) ───────────


def test_dispatch_upload_invocation_threads_num_workers():
    """r3 C1 fix (c) pin: run_upload must pass the realized width (the r2
    regression path was run_cpu_phase's width-less COMMON args)."""
    text = (SCRIPTS / "issue2329_dispatch.sh").read_text()
    m = re.search(r"run_upload\(\)\s*\{(.*?)\n\}", text, flags=re.S)
    assert m, "run_upload() not found in issue2329_dispatch.sh"
    body = m.group(1)
    assert "run_cpu_phase upload" in body
    assert re.search(r'--num-workers\s+"\$NUM_WORKERS"', body), (
        f'run_upload must thread --num-workers "$NUM_WORKERS" (r3 C1 fix (c)); got body: {body!r}'
    )


# ── width derivation (fix (b)) ────────────────────────────────────────


def test_family_realized_width_complete_and_absent(tmp_path):
    _make_store(tmp_path / "out", width=4)
    manifests = tmp_path / "out" / "manifests"
    assert R._family_realized_width(manifests, "anchors") == 4
    assert R._family_realized_width(manifests, "margin") == 4
    empty = tmp_path / "empty_manifests"
    empty.mkdir()
    assert R._family_realized_width(empty, "anchors") is None
    assert R._family_realized_width(empty, "margin") is None


def test_family_realized_width_incomplete_returns_none(tmp_path):
    out = tmp_path / "out"
    _make_store(out, width=4)
    (out / "manifests" / "anchors_rest_w2_done.json").unlink()
    assert R._family_realized_width(out / "manifests", "anchors") is None
    # margin family is unaffected by the anchors gap
    assert R._family_realized_width(out / "manifests", "margin") == 4


def test_family_realized_width_mixed_partial_prior_width(tmp_path):
    """8->4 reshard mid-state: w0..w3 rewritten @4, w4..w7 leftovers @8 —
    the ONLY complete width is 4 (the @8 set lost its low indexes)."""
    out = tmp_path / "out"
    manifests = out / "manifests"
    for kind in ("anchors_gate", "anchors_rest"):
        for i in range(4):
            _write(manifests / f"{kind}_w{i}_done.json", json.dumps(_done_rec(i, 4)))
        for i in range(4, 8):
            _write(manifests / f"{kind}_w{i}_done.json", json.dumps(_done_rec(i, 8)))
    assert R._family_realized_width(manifests, "anchors") == 4


def test_family_realized_width_ambiguous_raises(tmp_path):
    """Two simultaneously-complete widths = inconsistent store: refuse loudly
    rather than guess which one is live (sweeping either could destroy it).
    Distinct index namespaces (w0..w1 @2 under one naming overlap is
    impossible on disk, so simulate via both kinds complete at 1 and 2)."""
    manifests = tmp_path / "manifests"
    # width 1 complete: w0 @1 for both kinds ... but the same FILENAME cannot
    # hold two records, so complete-at-1 and complete-at-2 must come from
    # different kinds' overlap: use gate complete @1+@2 via idx0@1? Instead,
    # construct the genuinely reachable shape: margin family (ONE kind) with
    # w0@1 and w0..w1@2 recorded across distinct filenames.
    _write(manifests / "margin_anchors_w0_done.json", json.dumps(_done_rec(0, 1)))
    _write(manifests / "margin_anchors_w1_done.json", json.dumps(_done_rec(1, 2)))
    # @1 is complete ({0} present via... w0 is stamped @1) — @2 needs {0,1}@2;
    # only w1 is @2, so @2 incomplete => width 1 wins, no ambiguity here:
    assert R._family_realized_width(manifests, "margin") == 1
    # True ambiguity requires overlapping index sets at both widths, which a
    # single filename-per-index store can only reach through kind DISAGREEMENT
    # in the anchors family: gate fully @2, rest fully @2, PLUS a second
    # complete width from records claiming @1 at index 0 in BOTH kinds is
    # filename-blocked. The reachable ambiguous shape is a corrupt record set
    # where one kind's w0 claims @1 while indexes 0..1 exist @2 for both kinds
    # — synthesize it directly to pin the refusal branch:
    manifests2 = tmp_path / "manifests2"
    _write(manifests2 / "margin_anchors_w0_done.json", json.dumps(_done_rec(0, 1)))
    # a duplicate-claim record: idx 0 AND 1 recorded at width 2 in extra files
    _write(manifests2 / "margin_anchors_w1_done.json", json.dumps(_done_rec(1, 2)))
    _write(manifests2 / "margin_anchors_w2_done.json", json.dumps(_done_rec(0, 2)))
    with pytest.raises(RuntimeError, match="MULTIPLE complete widths"):
        R._family_realized_width(manifests2, "margin")


def test_family_realized_width_unreadable_record_raises(tmp_path):
    manifests = tmp_path / "manifests"
    _write(manifests / "margin_anchors_w0_done.json", "{not json")
    with pytest.raises(RuntimeError, match="unreadable done record"):
        R._family_realized_width(manifests, "margin")


# ── implicit width can never sweep (fix (d)) ──────────────────────────


def test_entry_sweep_skips_on_implicit_width(tmp_path):
    """The r2 C1 trigger shape: a phase invoked WITHOUT --num-workers must not
    quarantine anything — not even a genuinely stale w9 stray (provenance of
    the width is unknown, so destruction is refused; duplicates fail loud
    downstream instead)."""
    out, logs = tmp_path / "out", tmp_path / "logs"
    made = _make_store(out, width=4)
    stray = out / "anchors" / "anchors_gate_w9.jsonl"
    _write(stray, "x")
    cfg = _cfg(out, logs, ["--phase", "anchors"])  # no --num-workers
    assert cfg.num_workers == 1  # sharding default unchanged
    assert cfg.num_workers_explicit is False
    assert R._entry_sweep(cfg, "anchors") == 0
    for p in made["anchors"] + made["margin"] + [stray]:
        assert p.exists(), f"implicit-width invocation destroyed {p}"
    assert _no_quarantined_files(out)


def test_sweep_helper_refuses_non_positive_or_none_width(tmp_path):
    out = tmp_path / "out"
    _make_store(out, width=2)
    for bad in (0, -1, None):
        with pytest.raises(ValueError, match="explicit positive width"):
            R._sweep_stale_width_shards(
                out / "anchors",
                out / "margin",
                out / "manifests",
                out,
                bad,  # type: ignore[arg-type]
                family="anchors",
            )


def test_runconfig_default_width_provenance_is_inexplicit():
    """fix (d): a hand-built RunConfig (no CLI) must default to
    num_workers_explicit=False so it can never arm a sweep."""
    import dataclasses

    field = {f.name: f for f in dataclasses.fields(R.RunConfig)}["num_workers_explicit"]
    assert field.default is False


# ── family scoping at the helper level (fix (a)) ──────────────────────


def test_sweep_is_family_scoped(tmp_path):
    """A width-1 MARGIN sweep must not touch 8-wide ANCHOR artifacts (the
    Codex second site, distilled to the helper contract)."""
    out = tmp_path / "out"
    made = _make_store(out, width=8)
    moved = R._sweep_stale_width_shards(
        out / "anchors", out / "margin", out / "manifests", out, 1, family="margin"
    )
    # every margin artifact above index 0 quarantined; every anchor survives
    assert moved == 2 * 7  # 7 jsonl + 7 done records
    for p in made["anchors"]:
        assert p.exists(), f"margin-family sweep destroyed an anchor artifact: {p}"


# ── INTEGRATION 1: the exact regression scenario through phase_upload ─


@pytest.mark.parametrize(
    "width_argv",
    [
        [],  # the r2 broken shape: dispatcher threaded no width (implicit 1)
        ["--num-workers", "1"],  # the deferred/salvage narrow leg, post-fix (c)
    ],
    ids=["width-less", "explicit-narrow-1"],
)
def test_phase_upload_preserves_full_width_store(tmp_path, width_argv):
    """A real 4-wide anchors+margin out-root goes through the REAL phase_upload
    (argparse -> build_config -> phase_upload, --upload local-mirror): every
    worker's shards + done records SURVIVE and the mirrored upload set is
    complete. Pre-fix, both variants quarantined w1..w3 of BOTH families and
    the mirror held only w0 (the silent truncation both reviewers flagged)."""
    out, logs = tmp_path / "out", tmp_path / "logs"
    made = _make_store(out, width=4)
    cfg = _cfg(out, logs, ["--phase", "upload", "--upload", "local-mirror", "--smoke", *width_argv])
    rc = R.phase_upload(cfg)
    assert rc == R.RC_OK

    for p in made["anchors"] + made["margin"]:
        assert p.exists(), f"phase_upload destroyed a live full-width artifact: {p}"
    assert _no_quarantined_files(out), "phase_upload quarantined live full-width shards"

    mirror = out / "hf_mirror" / R.HF_PREFIX
    for i in range(4):
        for rel in (
            f"raw_completions/anchors/anchors_gate_w{i}.jsonl",
            f"raw_completions/anchors/anchors_rest_w{i}.jsonl",
            f"analysis_tensors/anchors/va_anchors_gate_w{i}.pt",
            f"analysis_tensors/anchors/va_anchors_rest_w{i}.pt",
            f"analysis_tensors/margin/anchor_margin_w{i}.jsonl",
            f"analysis_tensors/manifests/anchors_gate_w{i}_done.json",
            f"analysis_tensors/manifests/anchors_rest_w{i}_done.json",
            f"analysis_tensors/manifests/margin_anchors_w{i}_done.json",
        ):
            assert (mirror / rel).exists(), f"upload set INCOMPLETE: missing {rel}"
    assert (logs / R.SENTINEL_NAME_SMOKE).exists()


def test_phase_upload_sweeps_true_stale_at_derived_width(tmp_path):
    """F3's original purpose survives at the upload site: genuinely stale
    prior-width strays (w5/w7 of an 8-wide run resharded to a complete 4-wide
    store) are quarantined at the DERIVED width 4 and kept OUT of the mirror,
    while all live w0..w3 files ship."""
    out, logs = tmp_path / "out", tmp_path / "logs"
    made = _make_store(out, width=4)
    stale = [
        out / "anchors" / "anchors_gate_w5.jsonl",
        out / "anchors" / "va_anchors_rest_w7.pt",
        out / "margin" / "anchor_margin_w5.jsonl",
        out / "manifests" / "anchors_gate_w5_done.json",
        out / "manifests" / "margin_anchors_w7_done.json",
    ]
    _write(stale[0], "x")
    _write(stale[1], "x")
    _write(stale[2], "x")
    _write(stale[3], json.dumps(_done_rec(5, 8)))
    _write(stale[4], json.dumps(_done_rec(7, 8)))
    cfg = _cfg(out, logs, ["--phase", "upload", "--upload", "local-mirror", "--smoke"])
    assert R.phase_upload(cfg) == R.RC_OK
    for p in stale:
        assert not p.exists(), f"true stale prior-width shard shipped: {p}"
    for p in made["anchors"] + made["margin"]:
        assert p.exists(), f"live shard destroyed while sweeping stale strays: {p}"
    mirror = out / "hf_mirror" / R.HF_PREFIX
    assert not (mirror / "raw_completions/anchors/anchors_gate_w5.jsonl").exists()
    assert (mirror / "raw_completions/anchors/anchors_gate_w3.jsonl").exists()


# ── INTEGRATION 2: 1-wide deferred margin leg vs 8-wide anchors ───────


def test_deferred_narrow_margin_leg_leaves_anchor_family_untouched(tmp_path):
    """The Codex second site, through the real config path: the registered
    deferred-margin recipe runs `dispatch.sh margin` on a fresh 1x H100
    (NUM_WORKERS=1 threaded by run_fanout_phase) over an out-root holding a
    VALID 8-wide anchor family. The margin-entry sweep must quarantine ONLY
    stale margin artifacts and leave every anchor artifact in place."""
    out, logs = tmp_path / "out", tmp_path / "logs"
    made = _make_store(out, width=8, families=("anchors",))
    # a crashed partial 8-wide margin attempt: stale at the new width 1
    stale_margin = [
        out / "margin" / "anchor_margin_w3.jsonl",
        out / "manifests" / "margin_anchors_w5_done.json",
    ]
    _write(stale_margin[0], "x")
    _write(stale_margin[1], json.dumps(_done_rec(5, 8)))
    cfg = _cfg(out, logs, ["--phase", "margin", "--num-workers", "1", "--worker-index", "0"])
    moved = R._entry_sweep(cfg, "margin")
    assert moved == 2
    for p in made["anchors"]:
        assert p.exists(), f"1-wide margin leg destroyed an 8-wide anchor artifact: {p}"
    for p in stale_margin:
        assert not p.exists(), f"stale margin artifact survived the margin-entry sweep: {p}"


# ── AST call-site pins (the round-2 escape mode) ──────────────────────


def _fn(tree: ast.Module, name: str) -> ast.FunctionDef:
    fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name), None
    )
    assert fn is not None, f"function {name} not found"
    return fn


def _calls(fn: ast.FunctionDef, name: str) -> list[ast.Call]:
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == name) or (
                isinstance(f, ast.Attribute) and f.attr == name
            ):
                out.append(node)
    return out


def test_phase_entry_sweep_call_sites_are_family_scoped():
    """Bind the tested helpers to the DISPATCHED phase entries: phase_anchors /
    phase_margin sweep their OWN family via _entry_sweep; phase_upload uses
    the derived-width _upload_entry_sweeps; no phase calls the raw sweep."""
    tree = ast.parse((SCRIPTS / "issue2329_run.py").read_text())
    for phase, family in (("phase_anchors", "anchors"), ("phase_margin", "margin")):
        fn = _fn(tree, phase)
        calls = _calls(fn, "_entry_sweep")
        assert calls, f"{phase} must sweep via _entry_sweep"
        fams = {
            c.args[1].value
            for c in calls
            if len(c.args) >= 2 and isinstance(c.args[1], ast.Constant)
        }
        assert fams == {family}, f"{phase} sweeps families {fams}, expected {{{family!r}}}"
        assert not _calls(fn, "_sweep_stale_width_shards"), (
            f"{phase} must not call the raw sweep directly (family/width policy lives "
            "in _entry_sweep)"
        )
    upload = _fn(tree, "phase_upload")
    assert _calls(upload, "_upload_entry_sweeps"), "phase_upload must call _upload_entry_sweeps"
    assert not _calls(upload, "_entry_sweep"), (
        "phase_upload must NOT sweep at cfg width — upload widths are DERIVED per family"
    )
    assert not _calls(upload, "_sweep_stale_width_shards")
    # and the derived-width helper actually derives:
    ues = _fn(tree, "_upload_entry_sweeps")
    assert _calls(ues, "_family_realized_width")
    # its raw-sweep calls must not pass cfg.num_workers as the width argument
    for call in _calls(ues, "_sweep_stale_width_shards"):
        width_arg = call.args[4] if len(call.args) >= 5 else None
        assert not (isinstance(width_arg, ast.Attribute) and width_arg.attr == "num_workers"), (
            "_upload_entry_sweeps must sweep at the DERIVED width, never cfg.num_workers"
        )
