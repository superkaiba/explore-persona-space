"""#1090 fu3 dispatcher drain-loop + margin-pool invariants (review round-2 fixes).

Pins: (1) a dispatcher-side vLLM port collision retires the slot and REQUEUES the
popped cell (plan §D7 item 5) instead of failing it with zero attempts; (2) a
second collision fails LOUD with the ``vllm_port_collision`` reason; (3) the
production queue/drain/finalize path runs end-to-end over REAL subprocesses +
REAL sentinel files (``manifest_complete.json`` + the final results sentinel),
including the rc!=0 requeue branch; (4) the amendment-v4 top-up margin-pool
derivation joins raw rows with the ``kept_{pos,neg}.jsonl`` record and fails
loud on missing sidecars / the declared-unstageable behavior (broad_em);
(5) the conv-context binding is the plan-§D2 wildchat-family instance.

The ONLY fake is the subprocess boundary: ``_worker_cmd`` is monkeypatched to a
signature-conformant stub returning a real ``python -c`` command — the drain
loop, sentinel writers, and ``finalize`` all execute their production bodies.
"""

from __future__ import annotations

import json
import random
import socket
import sys
import types
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue1090_fu3_cells as fu3_cells  # noqa: E402
import issue1090_fu3_worker as fu3w  # noqa: E402

from explore_persona_space.artifacts.context import CONTEXTS  # noqa: E402


def _free_port_base(n: int = 2) -> int:
    """A base port with ``n`` consecutive free ports (best-effort probe)."""
    rng = random.Random()
    for _ in range(100):
        base = rng.randint(20000, 55000)
        if all(fu3w.port_free(base + i) for i in range(n)):
            return base
    raise RuntimeError("no free port window found")


def _stub_worker_cmd(sentinel_dir: Path, fail_once_cells: set[str]):
    """Signature-conformant ``_worker_cmd`` stub (real subprocess boundary).

    The stub command writes the per-cell done sentinel and exits 0; cells in
    ``fail_once_cells`` exit rc=2 on their FIRST attempt (exercises requeue).
    """

    def cmd(args, row, slot, port):  # mirrors fu3w._worker_cmd(args, row, slot, port)
        code = (
            "import json, os, sys\n"
            f"cell = {row['cell_id']!r}\n"
            f"sd = {str(sentinel_dir)!r}\n"
            f"fail_once = {row['cell_id'] in fail_once_cells!r}\n"
            "flag = os.path.join(sd, cell + '.failonce')\n"
            "if fail_once and not os.path.exists(flag):\n"
            "    open(flag, 'w').write('1')\n"
            "    sys.exit(2)\n"
            "p = os.path.join(sd, 'issue-1090-cell-' + cell + '.json')\n"
            "json.dump({'payload': {'status': 'done', 'run_name': 'stub-' + cell}},"
            " open(p, 'w'))\n"
            "sys.exit(0)\n"
        )
        return [sys.executable, "-c", code]

    return cmd


def _dispatch_args(tmp_path: Path, cells: str, n_gpus: int):
    return fu3w.parse_args(
        [
            "dispatch",
            "--cells",
            cells,
            "--n-gpus",
            str(n_gpus),
            "--out-root",
            str(tmp_path / "out"),
            "--sentinel-dir",
            str(tmp_path / "sent"),
            "--poll-seconds",
            "0.05",
            "--smoke",
            "--no-upload",
        ]
    )


def test_port_collision_requeues_to_free_slot(tmp_path, monkeypatch):
    """A busy port retires the slot but REQUEUES the cell — never drops it."""
    base = _free_port_base(2)
    monkeypatch.setattr(fu3w, "BASE_VLLM_PORT", base)
    args = _dispatch_args(tmp_path, "C1-bare-pos", n_gpus=2)
    monkeypatch.setattr(fu3w, "_worker_cmd", _stub_worker_cmd(Path(args.sentinel_dir), set()))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squat:
        squat.bind(("127.0.0.1", base))  # slot 0's port is squatted
        squat.listen(1)
        rc = fu3w.cmd_dispatch(args)
    assert rc == 0
    manifest = json.loads((Path(args.out_root) / "manifest_complete.json").read_text())
    assert manifest["cells_done"] == ["C1-bare-pos"]
    assert manifest["cells_failed"] == []


def test_port_collision_twice_fails_loud(tmp_path, monkeypatch):
    """Requeue exactly ONCE: a second collision fails with the port reason."""
    base = _free_port_base(2)
    monkeypatch.setattr(fu3w, "BASE_VLLM_PORT", base)
    args = _dispatch_args(tmp_path, "C1-bare-pos", n_gpus=2)
    monkeypatch.setattr(fu3w, "_worker_cmd", _stub_worker_cmd(Path(args.sentinel_dir), set()))
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s0,
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1,
    ):
        s0.bind(("127.0.0.1", base))
        s0.listen(1)
        s1.bind(("127.0.0.1", base + 1))
        s1.listen(1)
        rc = fu3w.cmd_dispatch(args)
    assert rc == 1
    manifest = json.loads((Path(args.out_root) / "manifest_complete.json").read_text())
    assert manifest["cells_failed"] == ["C1-bare-pos"]
    sentinel = json.loads(
        fu3w.cell_sentinel_path(Path(args.sentinel_dir), "C1-bare-pos").read_text()
    )
    assert sentinel["payload"]["reason"].startswith("vllm_port_collision:")


def test_drain_and_finalize_full_path(tmp_path, monkeypatch):
    """3 cells through the PRODUCTION drain loop: fill-every-free-slot, one
    rc=2 first attempt (requeue branch), finalize writes manifest_complete.json
    + the final results sentinel with the reproducibility card."""
    base = _free_port_base(2)
    monkeypatch.setattr(fu3w, "BASE_VLLM_PORT", base)
    cells = "C1-bare-pos,C2-bare-con,C3-bare-pos"
    args = _dispatch_args(tmp_path, cells, n_gpus=2)
    monkeypatch.setattr(
        fu3w, "_worker_cmd", _stub_worker_cmd(Path(args.sentinel_dir), {"C2-bare-con"})
    )
    rc = fu3w.cmd_dispatch(args)
    assert rc == 0
    manifest = json.loads((Path(args.out_root) / "manifest_complete.json").read_text())
    assert sorted(manifest["cells_done"]) == sorted(cells.split(","))
    assert manifest["cells_failed"] == []
    assert all(manifest["per_cell"][c]["status"] == "done" for c in cells.split(","))
    results = sorted(Path(args.sentinel_dir).glob("issue-1090-epm_smoke-result-*.json"))
    assert results, "finalize must write the final results sentinel"
    payload = json.loads(results[-1].read_text())
    assert payload["payload"]["reproducibility_card"]["hf_data_repo"]
    assert payload["kind"] == "epm:smoke-result"  # kind flips to epm:results off --smoke


def test_topup_pool_derivation_and_kept_filter(tmp_path):
    """The top-up derivation keeps ONLY kept request_ids and builds probe/answer
    pairs; missing sidecars raise a ValueError naming the file."""
    d = tmp_path / "topup"
    d.mkdir()

    def _w(name, rows):
        (d / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    _w(
        "raw_pos.jsonl",
        [
            {
                "request_id": "p1",
                "question_id": "q1",
                "variant_id": 0,
                "arm": "positive",
                "question": "Q1",
                "completion": "A1",
            },
            {
                "request_id": "p2",
                "question_id": "q2",
                "variant_id": 0,
                "arm": "positive",
                "question": "Q2",
                "completion": "A2",
            },
        ],
    )
    _w(
        "raw_neg.jsonl",
        [
            {
                "request_id": "n1",
                "question_id": "q1",
                "variant_id": 1,
                "arm": "negative",
                "question": "Q1",
                "completion": "N1",
            }
        ],
    )
    _w("kept_pos.jsonl", [{"request_id": "p1"}])  # p2 NOT kept
    _w("kept_neg.jsonl", [{"request_id": "n1"}])
    pos, neg = fu3w.derive_margin_pools_from_topup(d)
    assert [p["request_id"] for p in pos] == ["p1"]
    assert [(p["probe"], p["answer"]) for p in pos] == [("Q1", "A1")]
    assert [n["request_id"] for n in neg] == ["n1"]
    (d / "kept_neg.jsonl").unlink()
    with pytest.raises(ValueError, match="kept_"):
        fu3w.derive_margin_pools_from_topup(d)


def test_margin_pool_unavailable_behavior_raises_loud(tmp_path):
    """broad_em is the DECLARED-unstageable behavior: _behavior_margin_pools
    raises (cell-failing) instead of degrading to a silent n/a."""
    cfg = types.SimpleNamespace(out_root=tmp_path)
    with pytest.raises(ValueError, match="tf_margin pool unavailable for 'broad_em'"):
        fu3w._behavior_margin_pools(cfg, "broad_em")
    with pytest.raises(ValueError, match="no v4 pool source"):
        fu3w._behavior_margin_pools(cfg, "not_a_behavior")


def test_conv_context_is_wildchat_family():
    """Plan §D2: the conversational-prefix arm binds the wildchat-family
    instance (review Major — was the synthetic cooking exchange)."""
    ctx = CONTEXTS[fu3_cells.CONV_CONTEXT_ID]
    assert ctx.family == "wildchat"
    assert ctx.prefix_turns, "conversational prefix must carry prefix turns"
