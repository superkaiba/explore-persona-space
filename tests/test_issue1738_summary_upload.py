"""#1738 r4 persist-by-default fix: summary JSONs dual-write to HF.

Pins the permanent invariant added after the r4 incident (the phase-3 summary
JSONs had a git-only destination and were lost when the DELETE-on-exit GCE
instance was reaped before any VM harvest): the fits/rebuild/characterize
summary writers now mirror every small JSON output to the HF data repo under
``{hf_prefix}/analysis_tensors/summaries/...``.

Production-body tests (code-style § one production-body test per seam-stubbed
function): the REAL helper bodies execute; ONLY the external Hub boundary
(``hub._upload_folder_filtered``) is faked, signature-conformant by
construction via ``unittest.mock.create_autospec``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue1738_characterize as CH  # noqa: E402
import issue1738_multiturn_fits as FT  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402


def _fake_upload(monkeypatch, module, return_value="repo/prefix"):
    fake = create_autospec(hub._upload_folder_filtered, return_value=return_value)
    monkeypatch.setattr(module.hub, "_upload_folder_filtered", fake)
    return fake


def _char_tree(tmp_path: Path) -> SimpleNamespace:
    out_eval = tmp_path / "eval"
    kdir = out_eval / "kresample"
    kdir.mkdir(parents=True)
    (kdir / "gates.json").write_text("{}")
    (kdir / "floors_L19.npz").write_bytes(b"npz")
    return SimpleNamespace(no_upload=False, out_eval=out_eval, hf_prefix="issueTEST_mt")


def test_upload_summary_jsons_call_shape(monkeypatch, tmp_path):
    """Real body reaches the Hub seam with the exact-set verified-upload shape."""
    ns = _char_tree(tmp_path)
    fake = _fake_upload(monkeypatch, CH)
    kdir = ns.out_eval / "kresample"
    CH._upload_summary_jsons(ns, [kdir / "gates.json", kdir / "floors_L19.npz"])
    assert fake.call_count == 1
    kw = fake.call_args.kwargs
    assert fake.call_args.args[0] == ns.out_eval
    assert kw["repo_id"] == CH.C.HF_DATA_REPO and kw["repo_type"] == "dataset"
    dest = f"issueTEST_mt/{FT.ANALYSIS_TENSORS_SUBDIR}/summaries/characterize"
    assert kw["path_in_repo"] == dest
    assert kw["allow_patterns"] == sorted(["kresample/gates.json", "kresample/floors_L19.npz"])
    assert kw["expected_repo_paths"] == [f"{dest}/{r}" for r in kw["allow_patterns"]]


def test_upload_summary_jsons_skips(monkeypatch, tmp_path):
    """--no-upload skips; a hand-built namespace WITHOUT the flag (the smoke
    path) also skips; missing files never reach the seam."""
    ns = _char_tree(tmp_path)
    fake = _fake_upload(monkeypatch, CH)
    ns.no_upload = True
    CH._upload_summary_jsons(ns, [ns.out_eval / "kresample" / "gates.json"])
    smoke_ns = SimpleNamespace(out_eval=ns.out_eval, hf_prefix="x")  # no no_upload attr
    CH._upload_summary_jsons(smoke_ns, [ns.out_eval / "kresample" / "gates.json"])
    ns.no_upload = False
    CH._upload_summary_jsons(ns, [ns.out_eval / "does_not_exist.json"])
    assert fake.call_count == 0


def test_upload_summary_jsons_fail_loud_vs_best_effort(monkeypatch, tmp_path):
    """An unverified commit ("" return) raises on the normal path; the
    designed-halt path (best_effort=True, the rc-23 identity gate) logs and
    returns so the designed rc is never masked."""
    ns = _char_tree(tmp_path)
    paths = [ns.out_eval / "kresample" / "gates.json"]
    fake = _fake_upload(monkeypatch, CH, return_value="")
    with pytest.raises(RuntimeError, match="returned no URL"):
        CH._upload_summary_jsons(ns, paths)
    CH._upload_summary_jsons(ns, paths, best_effort=True)  # must not raise
    fake.side_effect = ConnectionError("hub down")
    CH._upload_summary_jsons(ns, paths, best_effort=True)  # must not raise
    with pytest.raises(ConnectionError):
        CH._upload_summary_jsons(ns, paths)


def test_fits_summary_upload_entry_collects_exact_set(tmp_path):
    """The fits/rebuild summaries entry names EVERY summary JSON class the r4
    incident lost (summary, baselines, fence, per-cell fit_meta records) and
    nothing else (tensors keep their own entries)."""
    out_eval = tmp_path / "eval"
    (out_eval / "fits" / "cells").mkdir(parents=True)
    (out_eval / "fits" / "cells_rebuilt").mkdir(parents=True)
    (out_eval / "percontext").mkdir(parents=True)
    (out_eval / "mapping_baselines.json").write_text("{}")
    (out_eval / "fits" / f"{FT.FIT_POINT}_fits.json").write_text("{}")
    (out_eval / "fits" / "fence_report.json").write_text("{}")
    (out_eval / "fits" / "cells" / "context_L19_ridge.json").write_text("{}")
    (out_eval / "fits" / "cells_rebuilt" / "prefix_L19_ridge.json").write_text("{}")
    (out_eval / "percontext" / "x.npz").write_bytes(b"npz")  # NOT a summary
    sub, local, files = FT._summary_upload_entry(SimpleNamespace(out_eval=out_eval))
    assert sub == "summaries" and local == out_eval
    assert files == sorted(
        [
            "mapping_baselines.json",
            f"fits/{FT.FIT_POINT}_fits.json",
            "fits/fence_report.json",
            "fits/cells/context_L19_ridge.json",
            "fits/cells_rebuilt/prefix_L19_ridge.json",
        ]
    )


def test_fits_upload_analysis_tensors_body(monkeypatch, tmp_path):
    """Real _upload_analysis_tensors body: explicit-file entries pass through
    verbatim; a None file list rglobs; an unverified commit raises."""
    local = tmp_path / "eval"
    (local / "fits").mkdir(parents=True)
    (local / "fits" / "a.json").write_text("{}")
    ns = SimpleNamespace(hf_prefix="issueTEST_mt", out_eval=local)
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    fake = _fake_upload(monkeypatch, FT)
    FT._upload_analysis_tensors(ns, [("summaries", local, ["fits/a.json"]), ("empty", empty, None)])
    assert fake.call_count == 1  # empty entry never reaches the seam
    kw = fake.call_args.kwargs
    assert kw["path_in_repo"] == f"issueTEST_mt/{FT.ANALYSIS_TENSORS_SUBDIR}/summaries"
    assert kw["allow_patterns"] == ["fits/a.json"]
    fake.return_value = ""
    with pytest.raises(RuntimeError, match="returned no URL"):
        FT._upload_analysis_tensors(ns, [("summaries", local, ["fits/a.json"])])
