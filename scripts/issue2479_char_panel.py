"""16-character panel registry for issue #2479 (AI-likeness gradient).

Encodes the plan v4 §4 Step-1 panel table VERBATIM: the 4 parent anchors
(HELIOS / Wren / Dana / Vex, descriptions imported verbatim from
``issue1310_common.PERSONAS`` — never retyped) plus 12 new personas across
the 4 design bands (A: strongly AI-like, B: helpful human professional,
C: ordinary human, D: strongly non-AI stylized). Design bands are
construction devices only; the measured judge axis supersedes them.

``--emit <path>`` writes the env-pointed panel JSON consumed by the
``EPM_I2479_CHAR_PANEL_JSON`` seams (the ladder-fill's ``_load_char_panel``,
``issue1345_common``'s variant-list extension, the stager's
``CHAR_VARIANTS`` extension). Canonical committed copy:
``eval_results/issue_2479/panel.json``.

Row schema — a SUPERSET of the U2 loader schema documented at
``issue1345_story_char_ladder_fill._load_char_panel`` (consumers tolerate
the extra keys):

- ``name``: lowercase slug (``"iris"``) — the ladder-fill's character key.
- ``variant_op``: ``"char_2479_<slug>_op"`` (all 16 characters). The
  ``char_2479_`` prefix is deliberate and DOUBLY load-bearing: the gen
  script's fail-loud persona guard keys on ``VARIANT.startswith("char_")``
  (issue1345_gen_stories_paired lines 152-155), and the prefix guarantees
  registry variants can never clobber the inherited ``REGIME_SPECS`` keys.
- ``variant_inserted``: ``"char_2479_<slug>"`` for the 8-member inserted
  (text-matched) subset, ``null`` otherwise.
- ``design_band``: ``"A"``..``"D"``.
- ``desc`` (extra key): the one-line persona description
  (rides ``EPM_I1345_PERSONA_DESC`` at generation time — must be single-line).
- ``display_name`` (extra key): the capitalized character name
  (rides ``EPM_STORY_CHARACTER_NAME``).
- ``inserted_subset`` (extra key): bool convenience mirror of
  ``variant_inserted is not None``.

Name constraints (plan §4 Step 1, enforced fail-LOUD at import AND emit —
required by the name-parametric attribution regex, ``EPM_STORY_CHARACTER_NAME``
seam; case-handling note at ``issue1345_scaffold_common.py:333-340``):
single alphabetic capitalized tokens, pairwise non-substring
(case-insensitive), disjoint from ``issue1310_common.FOIL_NAMES``
(case-insensitive, substring in both directions). Violations raise
``ValueError`` (never a bare ``assert`` — immune to ``-O`` stripping).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import issue1310_common as i1310  # noqa: E402  (light: static config, no env reads)

CHAR_PANEL_ENV = "EPM_I2479_CHAR_PANEL_JSON"
VARIANT_PREFIX = "char_2479_"

# The 4 parent anchors — descriptions come VERBATIM from issue1310_common.PERSONAS
# (desc=None below means "import from PERSONAS"; never retype them).
ANCHOR_NAMES = ("HELIOS", "Wren", "Dana", "Vex")

# Plan v4 §4 Step-1 table: (display_name, one-line persona desc | None, inserted_subset).
# Inserted subset (8): the 4 anchors + Iris, Elena, Gus, Mort (plan §4 Step 2).
_BAND_TABLE: tuple[tuple[str, tuple[tuple[str, str | None, bool], ...]], ...] = (
    (
        "A",
        (
            ("HELIOS", None, True),
            ("Iris", "a soft-spoken home AI assistant, unfailingly patient and factual", True),
            (
                "Cobalt",
                "a terse machine intelligence that answers in exact, structured statements",
                False,
            ),
            (
                "Vera",
                "a cheerful customer-support AI that apologizes often and answers strictly "
                "by the book",
                False,
            ),
        ),
    ),
    (
        "B",
        (
            ("Wren", None, True),
            (
                "Priya",
                "a friendly reference librarian who loves answering questions clearly",
                False,
            ),
            (
                "Marcus",
                "a patient high-school science teacher who explains things step by step",
                False,
            ),
            (
                "Elena",
                "a busy but kind emergency-room nurse who gives practical, direct advice",
                True,
            ),
        ),
    ),
    (
        "C",
        (
            ("Dana", None, True),
            ("Gus", "a retired postal worker who chats plainly about whatever comes up", True),
            ("Marisol", "a small-town diner cook who talks in short, homespun phrases", False),
            ("Tomas", "a distracted college student who answers off the top of his head", False),
        ),
    ),
    (
        "D",
        (
            ("Vex", None, True),
            (
                "Barnaby",
                "a superstitious old sailor who speaks in sea-worn slang and tall tales",
                False,
            ),
            (
                "Zara",
                "a hot-headed street racer who answers in slangy bursts and hates being questioned",
                False,
            ),
            (
                "Mort",
                "a gloomy medieval gravedigger who mutters in archaic, fatalistic phrases",
                True,
            ),
        ),
    ),
)


def validate_display_names(names: Sequence[str]) -> None:
    """Fail-LOUD name constraints (plan §4 Step 1); raises ValueError.

    Single alphabetic capitalized tokens, pairwise non-substring
    (case-insensitive), and disjoint from ``issue1310_common.FOIL_NAMES``
    (case-insensitive, substring both directions — a foil inside a panel
    name, or the reverse, would collide under the ``^<LABEL>:`` /
    name-parametric attribution conventions).
    """
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError(f"panel names not unique: {names}")
    for n in names:
        if not n or not n.isalpha() or not n[0].isupper():
            raise ValueError(
                f"panel name {n!r} is not a single alphabetic capitalized token "
                "(required by the name-parametric attribution regex)"
            )
    lowered = [n.lower() for n in names]
    for i, a in enumerate(lowered):
        for j, b in enumerate(lowered):
            if i != j and a in b:
                raise ValueError(
                    f"panel names {names[i]!r} and {names[j]!r} collide: one is a "
                    "(case-insensitive) substring of the other"
                )
    for n in lowered:
        for foil in i1310.FOIL_NAMES:
            f = foil.lower()
            if f in n or n in f:
                raise ValueError(
                    f"panel name {n!r} collides with FOIL name {foil!r} "
                    "(case-insensitive substring) — foils must stay disjoint from panel labels"
                )


def _build_panel() -> tuple[dict, ...]:
    """Build + validate the 16-row registry from the plan table (import-time)."""
    rows: list[dict] = []
    for band, members in _BAND_TABLE:
        for display, desc, inserted in members:
            if desc is None:
                # Anchor: description imported VERBATIM from the parent registry.
                desc = i1310.PERSONAS[display]
            slug = display.lower()
            rows.append(
                {
                    "name": slug,
                    "variant_op": f"{VARIANT_PREFIX}{slug}_op",
                    "variant_inserted": f"{VARIANT_PREFIX}{slug}" if inserted else None,
                    "design_band": band,
                    "desc": desc,
                    "display_name": display,
                    "inserted_subset": inserted,
                }
            )

    # --- structural fail-LOUD checks (plan §4 Steps 1-2) ---------------------
    if len(rows) != 16:
        raise ValueError(f"panel must have exactly 16 rows, got {len(rows)}")
    validate_display_names([r["display_name"] for r in rows])
    if set(ANCHOR_NAMES) != set(i1310.PERSONAS):
        raise ValueError(
            f"anchor set {ANCHOR_NAMES} != issue1310_common.PERSONAS keys "
            f"{tuple(i1310.PERSONAS)} — anchors must import their descriptions verbatim"
        )
    for r in rows:
        if r["display_name"] in ANCHOR_NAMES and r["desc"] != i1310.PERSONAS[r["display_name"]]:
            raise ValueError(f"anchor {r['display_name']!r} description drifted from PERSONAS")
        if not r["desc"] or "\n" in r["desc"]:
            raise ValueError(
                f"{r['display_name']!r}: persona desc must be a non-empty single line "
                "(it rides EPM_I1345_PERSONA_DESC into a one-sentence character intro)"
            )
        vop = r["variant_op"]
        if not vop.startswith(VARIANT_PREFIX) or not vop.endswith("_op"):
            raise ValueError(f"variant_op {vop!r} violates the char_2479_<slug>_op convention")
        vi = r["variant_inserted"]
        if vi is not None and (not vi.startswith(VARIANT_PREFIX) or "_op" in vi):
            raise ValueError(f"variant_inserted {vi!r} violates the char_2479_<slug> convention")
    inserted_names = {r["display_name"] for r in rows if r["inserted_subset"]}
    expected_inserted = set(ANCHOR_NAMES) | {"Iris", "Elena", "Gus", "Mort"}
    if inserted_names != expected_inserted:
        raise ValueError(
            f"inserted subset {sorted(inserted_names)} != plan §4 Step-2 subset "
            f"{sorted(expected_inserted)}"
        )
    all_variants = [v for r in rows for v in (r["variant_op"], r["variant_inserted"]) if v]
    if len(set(all_variants)) != len(all_variants):
        raise ValueError("duplicate variant ids in panel")
    return tuple(rows)


PANEL: tuple[dict, ...] = _build_panel()
PANEL_DISPLAY_NAMES: tuple[str, ...] = tuple(r["display_name"] for r in PANEL)
OP_VARIANTS: tuple[str, ...] = tuple(r["variant_op"] for r in PANEL)
INSERTED_VARIANTS: tuple[str, ...] = tuple(
    r["variant_inserted"] for r in PANEL if r["variant_inserted"]
)
ALL_VARIANTS: tuple[str, ...] = tuple(
    v for r in PANEL for v in (r["variant_op"], r["variant_inserted"]) if v
)


def load_char_panel_env(env_name: str = CHAR_PANEL_ENV) -> tuple[dict, ...] | None:
    """Shared env-pointed panel loader for the gen-side + stager seams.

    Same contract as the ladder-fill's ``_load_char_panel`` (the U2 schema of
    record): env UNSET/empty returns ``None`` (consumers keep their hardcoded
    parent behavior byte-identically); env SET but missing/unreadable/
    malformed/schema-violating RAISES — never a silent fallback. Extra keys
    (``desc`` / ``display_name`` / ``inserted_subset``) are tolerated.
    """
    path_s = os.environ.get(env_name, "").strip()
    if not path_s:
        return None
    path = Path(path_s)
    if not path.is_file():
        raise FileNotFoundError(f"{env_name}={path_s} does not point at a readable file")
    try:
        rows = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{env_name}={path_s} is unreadable/malformed JSON: {e}") from e
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"{env_name}={path_s}: expected a non-empty JSON list of panel objects, "
            f"got {type(rows).__name__}"
        )
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"{env_name}: row {i} is not an object")
        missing = {"name", "variant_op", "variant_inserted", "design_band"} - r.keys()
        if missing:
            raise ValueError(f"{env_name}: row {i} missing keys {sorted(missing)}")
        for key in ("name", "variant_op", "design_band"):
            if not isinstance(r[key], str) or not r[key]:
                raise ValueError(f"{env_name}: row {i} field {key!r} must be non-empty str")
        vop = r["variant_op"]
        if not vop.startswith("char_") or "_op" not in vop or vop.endswith("_base"):
            raise ValueError(
                f"{env_name}: row {i} variant_op {vop!r} must start with 'char_', "
                "contain '_op', and not end in '_base' (REGIME_SPECS suffix conventions)"
            )
        vi = r["variant_inserted"]
        if vi is not None and (
            not isinstance(vi, str)
            or not vi.startswith("char_")
            or "_op" in vi
            or vi.endswith("_base")
        ):
            raise ValueError(
                f"{env_name}: row {i} variant_inserted {vi!r} must be null or a "
                "'char_'-prefixed id with neither '_op' nor a '_base' suffix"
            )
    names = [r["name"] for r in rows]
    variants = [v for r in rows for v in (r["variant_op"], r["variant_inserted"]) if v]
    if len(set(names)) != len(names) or len(set(variants)) != len(variants):
        raise ValueError(f"{env_name}: duplicate character names or variant ids")
    return tuple(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Emit the panel JSON (``--emit <path>``) with a one-line digest."""
    ap = argparse.ArgumentParser(
        description="issue #2479: emit the 16-character panel registry JSON"
    )
    ap.add_argument("--emit", type=Path, required=True, help="output panel JSON path")
    args = ap.parse_args(argv)
    payload = json.dumps(list(PANEL), indent=2, ensure_ascii=False) + "\n"
    args.emit.parent.mkdir(parents=True, exist_ok=True)
    args.emit.write_text(payload)
    sha = hashlib.sha256(payload.encode()).hexdigest()
    print(
        f"[issue2479_char_panel] wrote {args.emit} rows={len(PANEL)} "
        f"op={len(OP_VARIANTS)} inserted={len(INSERTED_VARIANTS)} sha256={sha[:16]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
