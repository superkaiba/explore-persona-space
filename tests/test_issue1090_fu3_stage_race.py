"""#1090 fu3 crash-fix bug 2: `_stage_generic_corpus` must be concurrent-safe.

The pre-fix shape `os.replace(hf_hub_download(local_dir=dest.parent), dest)`
raced across N parallel cell workers: the winner moved the shared download
away and latecomers crashed FileNotFoundError (5 hard-failed cells in the
fu3 production launch). This pins: two concurrent stagers both succeed, with
exactly ONE download; and a pre-staged dest short-circuits with zero
downloads. The HF network boundary is the only faked seam (a fake mirroring
`hf_hub_download`'s (repo_id, filename, *, repo_type, local_dir) call shape).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1074_generator_compare as i1074  # noqa: E402


def test_two_concurrent_stagers_one_download(tmp_path, monkeypatch):
    dest = tmp_path / "inputs" / "generic_corpus.jsonl"
    n_downloads = []
    barrier = threading.Barrier(2)

    def fake_hf_hub_download(repo_id, filename, *, repo_type, local_dir):
        n_downloads.append(1)
        p = Path(local_dir) / Path(filename).name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"row": 1}\n')
        return str(p)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    results: dict[int, str] = {}
    errors: list[BaseException] = []

    def stage(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            results[i] = i1074._stage_generic_corpus(dest, claim_wait_s=30.0)
        except BaseException as e:  # pragma: no cover - surfaced via assert below
            errors.append(e)

    threads = [threading.Thread(target=stage, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert results[0] == results[1] == str(dest)
    assert dest.read_text() == '{"row": 1}\n'
    assert len(n_downloads) == 1, f"expected exactly one download, got {len(n_downloads)}"
    assert not (dest.parent / (dest.name + ".lock")).exists(), "claim lock must be released"
    assert not list(dest.parent.glob(".stage_tmp_*")), "temp download dirs must be cleaned"


def test_prestaged_dest_short_circuits_without_download(tmp_path, monkeypatch):
    dest = tmp_path / "generic_corpus.jsonl"
    dest.write_text('{"row": 1}\n')

    def boom(*a, **k):  # pragma: no cover - the assert is that it never fires
        raise AssertionError("hf_hub_download must not be called when dest exists")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    assert i1074._stage_generic_corpus(dest) == str(dest)
