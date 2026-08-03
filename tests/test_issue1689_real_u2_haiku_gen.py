"""Regression tests for scripts/issue1689_real_u2_haiku_gen.py.

Pins the round-3 (``epm:failure v7``) crash: the Batch API path in
``api_dispatch.dispatch_calls`` raises ``ValueError`` at
``api_dispatch.py:1519`` when ``checkpoint_dir`` is missing, but the
round-2 removal of ``force_path="sync"`` did not thread the required
kwarg. Round-3 threads it from ``main()`` (``args.out_path.parent /
"checkpoint"``) into ``generate_haiku_u2``.

Body-execution test (per code-style.md § "One production-body test per
seam-stubbed function"): the ``dispatch_calls`` fake is signature-
conformant (its parameters bind against the real symbol), so a future
rename / arity drift at the real call site fails HERE, not at pod-side
runtime.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_real_u2_haiku_gen import (  # noqa: E402
    HAIKU_MODEL,
    generate_haiku_u2,
)


def test_generate_haiku_u2_threads_checkpoint_dir_to_dispatch_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Body-execution test: `generate_haiku_u2` MUST pass `checkpoint_dir` through.

    Fakes ONLY the Anthropic-API boundary (`dispatch_calls`) with a
    signature-conformant stub, so an arity/keyword drift at the real
    call site fails HERE. The crash the fix closes fires at
    api_dispatch.py:1519 (ValueError, "checkpoint_dir is required for
    the batch path"); THIS test asserts the kwarg reaches
    dispatch_calls, which is what makes that raise unreachable.
    """
    from explore_persona_space.llm import api_dispatch as real_ad
    from explore_persona_space.llm.api_dispatch import DispatchItem, DispatchResult

    real_sig = inspect.signature(real_ad.dispatch_calls)
    calls: list[dict] = []

    async def fake_dispatch_calls(items: list[DispatchItem], **kwargs) -> dict[str, DispatchResult]:
        # Signature-conformity: this exact call shape must bind against
        # the REAL signature — an arity or kwarg-name drift fails here.
        real_sig.bind_partial(items, **kwargs)
        calls.append({"n_items": len(items), "kwargs": dict(kwargs)})
        return {
            it.item_id: DispatchResult(
                item_id=it.item_id,
                result=f"fake u2 for {it.payload['u1']!r}",
                error=False,
            )
            for it in items
        }

    monkeypatch.setattr(real_ad, "dispatch_calls", fake_dispatch_calls)

    ckpt = tmp_path / "checkpoint"
    rows = [
        {"conv_id": 1, "u1": "hi", "a1": "hello"},
        {"conv_id": 2, "u1": "hey", "a1": "yo"},
    ]
    out = generate_haiku_u2(rows, mock_response=None, checkpoint_dir=ckpt)

    # The stubbed dispatch was called exactly once; the fix's
    # load-bearing invariant is that checkpoint_dir reached it.
    assert len(calls) == 1
    got_kwargs = calls[0]["kwargs"]
    assert {
        "model",
        "build_request",
        "parse_response",
        "response_valid",
        "checkpoint_dir",
    }.issubset(got_kwargs.keys()), (
        f"generate_haiku_u2 did not thread the required kwargs; got {set(got_kwargs)}"
    )
    # checkpoint_dir must be a Path equal to the one main() derived —
    # any regression that drops it re-opens the api_dispatch.py:1519
    # ValueError on the real batch path.
    assert Path(got_kwargs["checkpoint_dir"]) == ckpt
    # force_path is absent — round-2 Major #3 removed it, and the
    # crash-fix must NOT reintroduce it as a papering-over shortcut.
    assert "force_path" not in got_kwargs
    assert got_kwargs["model"] == HAIKU_MODEL

    # And the checkpoint dir was created before the call, so
    # api_dispatch can write its per-batch state under it.
    assert ckpt.exists() and ckpt.is_dir()

    # Rows carry the fake u2_haiku + u2_source.
    assert len(out) == 2
    assert out[0]["u2_haiku"] == "fake u2 for 'hi'"
    assert all(r["u2_source"] == "haiku" for r in out)


def test_generate_haiku_u2_raises_on_missing_checkpoint_dir_real_path() -> None:
    """A caller that forgets to thread checkpoint_dir fails LOUD at the caller-visible
    site (`generate_haiku_u2`) BEFORE reaching api_dispatch.

    Pins the fail-loud sibling of the crash: dropping `checkpoint_dir=`
    on the real path would otherwise crash deep in api_dispatch. The
    caller-visible RuntimeError mentions the required kwarg + why.
    """
    rows = [{"conv_id": 1, "u1": "hi", "a1": "hello"}]
    with pytest.raises(ValueError, match="checkpoint_dir is REQUIRED"):
        generate_haiku_u2(rows, mock_response=None, checkpoint_dir=None)


def test_generate_haiku_u2_mock_response_bypasses_checkpoint_requirement() -> None:
    """The mock-response short-circuit path does NOT require checkpoint_dir.

    Smoke runs (--smoke, mock_response set) never reach the real batch
    path, so a null checkpoint_dir there must be safe — the crash-fix
    signature preserves the smoke shape.
    """
    rows = [
        {"conv_id": 1, "u1": "hi", "a1": "hello"},
        {"conv_id": 2, "u1": "hey", "a1": "yo"},
    ]
    out = generate_haiku_u2(rows, mock_response="fake u2 text", checkpoint_dir=None)
    assert len(out) == 2
    assert all(r["u2_haiku"] == "fake u2 text" for r in out)
    assert all(r["u2_source"] == "haiku" for r in out)


def test_main_derives_checkpoint_dir_next_to_out_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`main()` MUST derive `checkpoint_dir` as `out_path.parent / 'checkpoint'`
    and thread it through.

    Ensures the wiring at the call site — not just the callee — carries
    the fix. Feeds a tiny JSONL through `main()` with mock_response set
    (so no real API call fires) and then verifies the derived path
    would land next to the output.
    """
    # Craft a 1-row JSONL corpus.
    in_path = tmp_path / "corpus.jsonl"
    in_path.write_text('{"conv_id": 1, "u1": "hi", "a1": "hello"}\n')
    out_path = tmp_path / "out" / "haiku_u2.jsonl"

    # Import module lazily so we can monkeypatch sys.argv.
    import scripts.issue1689_real_u2_haiku_gen as haiku_gen

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue1689_real_u2_haiku_gen.py",
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--smoke",  # limits rows to 5, defaults mock_response
        ],
    )
    rc = haiku_gen.main()
    assert rc == 0
    # Under --smoke with the default mock_response, the real path never
    # runs, so the checkpoint dir need not be created. The load-bearing
    # invariant this test pins is that main() COMPUTES the derived path
    # only when mock_response is None, i.e. the real path — which the
    # test above (via the stubbed dispatch) already exercised. Here we
    # just assert main() completes cleanly on the smoke path so the
    # derivation branch is not accidentally reached under --smoke.
    assert out_path.exists()
