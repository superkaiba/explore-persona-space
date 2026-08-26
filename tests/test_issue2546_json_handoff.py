"""Network-free, GPU-free pins for the issue-2546 round-10 hot-fix (commit
``fce1f6012e``): the SideSpec tuple invariant across the parent->worker
work-file JSON handoff, plus the generalizing JSON-lossy-annotation guard.

Incident (task #2546 arm-1 ``p1_smoke_rig``, epm:failure v1 of round 10):

    AssertionError: open-thoughts/OpenThinker3-7B: '<think>' ->
    (13708, 766, 29) != [13708, 766, 29]

Identical token ids; the comparison failed on CONTAINER TYPE. The parent
serializes ``asdict(side)`` into each per-slot work file
(``issue2546_gen_capture.py`` parent sites) and each worker rebuilds
``SideSpec(**work["side_spec"])``. JSON has no tuple type, so the
``open_ids``/``close_ids`` fields returned as LISTS and the pinned-encoding
assert ``tuple(tok.encode(...)) == side.open_ids`` compared tuple != list.
The in-process VM pin check passed throughout because it never crosses the
JSON boundary — which is why every test here routes through the GENUINE
``asdict -> json.dumps -> json.loads -> SideSpec(**payload)`` path, never an
in-process construction.

This is the third instance of one pattern on #2546 (round 8:
``compose_prompts`` stripped stratification keys; round 9: the call-site
threading gap; round 10: JSON tuple erasure) — each round guarded its own
instance. The two structural tests here guard the CLASS:

* ``TestCrossingSetPin`` (tripwire half): AST-pins that ``SideSpec`` is the
  ONLY dataclass serialized toward a work file in ``issue2546_gen_capture.py``
  (every ``asdict(...)`` argument is ``side``; the worker rebuild expression
  ``SideSpec(**work["side_spec"])`` is live), and that the siblings
  ``issue2546_fit_cells.py`` / ``issue2546_stage_corpora.py`` contain no
  ``asdict`` call at all. A future round that starts serializing ``ArmSpec``
  (``frozen: tuple[int, int, int]``, ``sides: tuple[SideSpec, ...]``) or a
  fit-cells dataclass fails HERE and must consciously extend the
  degradation coverage below.
* ``TestReflectiveDegradation`` (reflective half): enumerates SideSpec's
  JSON-lossy-annotated fields (tuple/set/frozenset, unions unwrapped) from
  ``dataclasses.fields`` + ``typing.get_type_hints`` — NOT from a hardcoded
  name list — and asserts every one of the SIX realized ARMS sides restores
  the declared container across the real round trip. A future tuple field
  added to SideSpec without extending the ``__post_init__`` coercion loop
  fails automatically.

Every regression test fails against the pre-fix module
(``git show fce1f6012e~1:scripts/issue2546_gen_capture.py``); the
falsification run is recorded in the round's implementation marker.

The tokenizer in ``TestRealConsumer`` is faked ONLY at the external model
boundary, signature-conformant (a real class mirroring the one call surface
``assert_think_pins`` uses — never a bare Mock; code-style.md "one
production-body test per seam-stubbed function"). ``assert_think_pins``
itself is the real production body — the exact pod crash site.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import sys
import types
import typing
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2546_gen_capture as G  # noqa: E402

# Every arm/side that pins open_ids/close_ids (arm 1 post, arm 2 post,
# arm 3 think_on) — used VERBATIM from the arm registry, no test-local drift.
PINNED_SIDES = [(1, "post"), (2, "post"), (3, "think_on")]
# The complement: sides whose pin fields are None (coercion must not touch).
UNPINNED_SIDES = [(1, "pre"), (2, "pre"), (3, "think_off")]
ALL_SIDES = [(arm, s.side) for arm in sorted(G.ARMS) for s in G.ARMS[arm].sides]

GEN_CAPTURE_SRC = REPO_ROOT / "scripts" / "issue2546_gen_capture.py"
SIBLING_SRCS = [
    REPO_ROOT / "scripts" / "issue2546_fit_cells.py",
    REPO_ROOT / "scripts" / "issue2546_stage_corpora.py",
]


def _side(arm: int, name: str) -> G.SideSpec:
    (side,) = [s for s in G.ARMS[arm].sides if s.side == name]
    return side


def _worker_rebuild(side: G.SideSpec) -> G.SideSpec:
    """The GENUINE parent->worker handoff, byte-for-byte in mechanism.

    Parent: ``_atomic_write_json(wf, {..., "side_spec": asdict(side), ...})``
    (json.dumps); worker: ``SideSpec(**json.loads(...)["side_spec"])``. An
    in-process ``SideSpec(**asdict(side))`` would NOT regress-test the fix:
    asdict preserves tuples, so only the json round trip degrades them.
    """
    payload = json.loads(json.dumps({"side_spec": dataclasses.asdict(side)}))
    return G.SideSpec(**payload["side_spec"])


class TestPinnedSidesRoundTrip:
    """The crash class: pinned think-delimiter encodings must survive AS TUPLES."""

    @pytest.mark.parametrize(("arm", "name"), PINNED_SIDES)
    def test_container_type_and_values_survive(self, arm: int, name: str) -> None:
        side = _side(arm, name)
        assert side.open_ids is not None and side.close_ids is not None  # fixture sanity
        rebuilt = _worker_rebuild(side)
        # The INVARIANT, not just equality: the declared container type is
        # restored. An equality-only test would pass if the coercion were
        # removed and the production assert made type-insensitive instead.
        assert type(rebuilt.open_ids) is tuple, type(rebuilt.open_ids)
        assert type(rebuilt.close_ids) is tuple, type(rebuilt.close_ids)
        # Values unchanged (the fce1f6012e contract: type only, never values).
        assert rebuilt.open_ids == side.open_ids
        assert rebuilt.close_ids == side.close_ids
        # Full dataclass equality — pre-fix this is False (list != tuple).
        assert rebuilt == side

    @pytest.mark.parametrize(("arm", "name"), UNPINNED_SIDES)
    def test_none_pins_survive_untouched(self, arm: int, name: str) -> None:
        side = _side(arm, name)
        assert side.open_ids is None and side.close_ids is None  # fixture sanity
        rebuilt = _worker_rebuild(side)
        assert rebuilt.open_ids is None and rebuilt.close_ids is None
        assert rebuilt == side


class _PinnedEncodeTok:
    """Signature-conformant tokenizer boundary fake (never a bare Mock).

    Mirrors the ONE call surface ``assert_think_pins`` uses:
    ``encode(text, add_special_tokens=False) -> list[int]`` — real HF
    tokenizers return a LIST here, which is exactly what made the pre-fix
    comparison ``tuple(got) == [list-from-json]`` fail on container type.
    """

    def __init__(self, open_ids: tuple[int, ...], close_ids: tuple[int, ...]) -> None:
        self._table = {G.THINK_OPEN: list(open_ids), G.THINK_CLOSE: list(close_ids)}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        return list(self._table[text])


class TestRealConsumer:
    """Execute the REAL ``assert_think_pins`` body on a worker-rebuilt side."""

    @pytest.mark.parametrize(("arm", "name"), PINNED_SIDES)
    def test_assert_think_pins_passes_on_worker_rebuilt_side(self, arm: int, name: str) -> None:
        side = _side(arm, name)
        assert side.open_ids is not None and side.close_ids is not None
        tok = _PinnedEncodeTok(side.open_ids, side.close_ids)
        rebuilt = _worker_rebuild(side)
        # Pre-fix this raises the exact pod-side AssertionError
        # ("... (13708, 766, 29) != [13708, 766, 29]"); post-fix it passes.
        G.assert_think_pins(tok, rebuilt)


def _asdict_calls(tree: ast.AST) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == "asdict":
                out.append(node)
    return out


class TestCrossingSetPin:
    """Tripwire: pin WHICH dataclasses cross a JSON boundary (SideSpec only).

    ``ArmSpec.frozen`` / ``ArmSpec.sides`` and the fit-cells dataclasses
    (``LayerProfile.frozen``, ``Unit``) do NOT cross today — workers rebuild
    ArmSpec via the ``ARMS[work["arm"]]`` registry lookup, and fit_cells /
    stage_corpora never call asdict. Nothing but convention stops a future
    caller from serializing them; this test converts that convention into a
    failure the author must confront (extend ``TestReflectiveDegradation``
    to the newly-crossing class, with coercion, before this pin is updated).
    """

    def test_gen_capture_serializes_only_sidespec(self) -> None:
        tree = ast.parse(GEN_CAPTURE_SRC.read_text())
        calls = _asdict_calls(tree)
        assert calls, "expected asdict work-file serialization sites in gen_capture"
        for call in calls:
            assert len(call.args) == 1 and not call.keywords, ast.dump(call)
            (arg,) = call.args
            assert isinstance(arg, ast.Name) and arg.id == "side", (
                f"line {call.lineno}: asdict({ast.unparse(arg)}) — a dataclass other than "
                "the SideSpec local `side` is being serialized; extend the JSON-degradation "
                "coverage in this file (with __post_init__ coercion) before updating this pin"
            )

    def test_worker_rebuild_expression_is_live(self) -> None:
        """The simulated round trip must mirror a live production expression."""
        tree = ast.parse(GEN_CAPTURE_SRC.read_text())
        rebuilds = 0
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "SideSpec" or node.args or len(node.keywords) != 1:
                continue
            (kw,) = node.keywords
            if kw.arg is not None:  # ** splat has arg=None
                continue
            sub = kw.value
            if (
                isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "work"
                and isinstance(sub.slice, ast.Constant)
                and sub.slice.value == "side_spec"
            ):
                rebuilds += 1
        assert rebuilds == 3, (
            f"expected the 3 worker-side SideSpec(**work['side_spec']) rebuild sites "
            f"(gen / capture / capture-rel), found {rebuilds} — if the handoff shape "
            "changed, update _worker_rebuild to match the new production expression"
        )

    @pytest.mark.parametrize("src", SIBLING_SRCS, ids=lambda p: p.name)
    def test_siblings_have_no_asdict_boundary(self, src: Path) -> None:
        calls = _asdict_calls(ast.parse(src.read_text()))
        assert not calls, (
            f"{src.name} gained asdict call(s) at line(s) "
            f"{[c.lineno for c in calls]} — its dataclasses' JSON-lossy fields "
            "(e.g. LayerProfile.frozen: tuple[int, ...]) now need round-trip "
            "coverage + coercion; extend this test module"
        )


def _lossy_origins(tp: object) -> set[type]:
    """tuple/set/frozenset origins reachable in an annotation (unions unwrapped)."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or isinstance(tp, types.UnionType):
        out: set[type] = set()
        for member in typing.get_args(tp):
            out |= _lossy_origins(member)
        return out
    if origin in (tuple, set, frozenset):
        return {origin}
    if tp in (tuple, set, frozenset):
        return {tp}  # bare (unparametrized) annotation
    return set()


class TestReflectiveDegradation:
    """Reflective guard: EVERY JSON-lossy-annotated SideSpec field restores.

    Field enumeration comes from dataclasses.fields + typing.get_type_hints,
    never a hardcoded name list — so a future ``stop_ids: tuple[int, ...]``
    added to SideSpec without extending the (deliberately explicit)
    ``__post_init__`` coercion loop fails here automatically.
    """

    def test_reflection_sees_the_known_lossy_fields(self) -> None:
        hints = typing.get_type_hints(G.SideSpec)
        lossy = {f.name for f in dataclasses.fields(G.SideSpec) if _lossy_origins(hints[f.name])}
        # Meta-assert: the reflection must actually see the incident fields,
        # otherwise the degradation test below could silently check nothing.
        assert {"open_ids", "close_ids"} <= lossy, lossy

    @pytest.mark.parametrize(("arm", "name"), ALL_SIDES)
    def test_every_side_restores_declared_containers(self, arm: int, name: str) -> None:
        side = _side(arm, name)
        rebuilt = _worker_rebuild(side)
        hints = typing.get_type_hints(G.SideSpec)
        checked = 0
        for f in dataclasses.fields(G.SideSpec):
            origins = _lossy_origins(hints[f.name])
            if not origins:
                continue
            checked += 1
            value = getattr(rebuilt, f.name)
            if value is None:
                assert getattr(side, f.name) is None  # None only where declared None
                continue
            assert type(value) in origins, (
                f"SideSpec.{f.name} came back as {type(value).__name__} across the "
                f"work-file JSON round trip (arm {arm} side {name!r}) — extend the "
                "__post_init__ coercion to this field"
            )
            assert value == getattr(side, f.name)  # values unchanged, type only
        assert checked >= 2  # open_ids + close_ids at minimum
