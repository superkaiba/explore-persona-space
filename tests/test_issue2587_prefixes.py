"""HF prefix reconciliation pins for issue #2587 (unit 6, Part 2).

Pins the plan-§6.5 matched-7B predictions prefix
``issue2587_minpair/analysis_tensors/preds_7b_matched`` at BOTH ends of the
seam — the unit-4 writer's CLI example (``scripts/issue2587_fits.py``) and
the unit-5b reader's staging constant (``scripts/issue2587_analysis.py``) —
and asserts no WRITE/upload target in either payload resolves under the LIVE
sibling task's ``issue2564_`` namespace (READ constants of the parent's
banked stores legitimately do and are exempt).

Also pins the #2329 layer-sweep dash-mark convention mirrored into
``scripts/issue2587_figures.py`` against its source of record
(``scripts/issue2329_figures.py``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue2587_figures as G  # noqa: E402

PLAN_PREDS7B_PREFIX = "issue2587_minpair/analysis_tensors/preds_7b_matched"

_ANALYSIS_SRC = (REPO / "scripts" / "issue2587_analysis.py").read_text(encoding="utf-8")
_FITS_SRC = (REPO / "scripts" / "issue2587_fits.py").read_text(encoding="utf-8")


def test_analysis_preds7b_read_prefix_matches_plan():
    m = re.search(r'^PREFIX_PREDS7B\s*=\s*"([^"]+)"', _ANALYSIS_SRC, re.MULTILINE)
    assert m, "PREFIX_PREDS7B constant not found in issue2587_analysis.py"
    assert m.group(1) == PLAN_PREDS7B_PREFIX


def test_analysis_argparse_default_is_the_constant():
    # the CLI default threads THROUGH the constant (no re-typed literal)
    assert re.search(r'"--prefix-preds7b",\s*default=PREFIX_PREDS7B', _ANALYSIS_SRC), (
        "--prefix-preds7b must default to PREFIX_PREDS7B"
    )


def test_fits_docstring_example_matches_plan():
    m = re.search(r"--preds7b-prefix\s+(\S+)", _FITS_SRC)
    assert m, "no --preds7b-prefix example in issue2587_fits.py docstring"
    assert m.group(1) == PLAN_PREDS7B_PREFIX


def test_no_write_target_under_issue2564_namespace():
    """Upload/write prefixes in both payloads must live under issue2587_*.

    The fits CLI upload prefixes have NO argparse defaults (the #1005
    upload-prefix clobber shape: --upload hf requires explicit prefixes), so
    the docstring examples ARE the dispatch-time write targets of record;
    the analysis module writes only under eval_results/issue_2587 and stages
    (reads) from PREFIX_PREDS7B.
    """
    for flag in ("--payloads-prefix", "--preds-prefix", "--preds7b-prefix"):
        for m in re.finditer(rf"{flag}\s+(\S+)", _FITS_SRC):
            target = m.group(1)
            if "/" not in target:  # prose mention (e.g. an error message), not a path value
                continue
            assert not target.startswith("issue2564_"), (flag, target)
            assert target.startswith("issue2587_"), (flag, target)
    # fail-loud no-default pins on the upload prefixes (the #1005 shape)
    for flag in ("--payloads-prefix", "--preds-prefix", "--preds7b-prefix"):
        pat = rf'"{flag}",[^)]*default=None'
        assert re.search(pat, _FITS_SRC, re.DOTALL), f"{flag} must default to None"
    # the analysis-side staging constant is a READ of unit 4's OWN output —
    # never the parent namespace
    m = re.search(r'^PREFIX_PREDS7B\s*=\s*"([^"]+)"', _ANALYSIS_SRC, re.MULTILINE)
    assert m and not m.group(1).startswith("issue2564_")


def test_parent_read_constants_untouched():
    """The parent's banked-store READ constants legitimately stay issue2564_*."""
    assert re.search(r'^PREFIX_2564\s*=\s*"issue2564_minpair"', _ANALYSIS_SRC, re.MULTILINE)
    assert (
        'VC2564_HF_PATH = "issue2564_minpair/analysis_tensors/vc2564/vc2564_bank.pt"' in _FITS_SRC
    )


def test_full_attention_convention_pinned_to_2329():
    src2329 = (REPO / "scripts" / "issue2329_figures.py").read_text(encoding="utf-8")
    m = re.search(r"FULL_ATTENTION_LAYERS\s*=\s*frozenset\(\{([\d,\s]+)\}\)", src2329)
    assert m, "FULL_ATTENTION_LAYERS not found in issue2329_figures.py"
    layers_2329 = frozenset(int(x) for x in m.group(1).split(","))
    assert layers_2329 == G.FULL_ATTENTION_LAYERS_9B
    m = re.search(r"N_MODEL_LAYERS\s*=\s*(\d+)", src2329)
    assert m and int(m.group(1)) == G.N_LAYERS_9B
    m = re.search(r'FULL_ATTN_COLOR\s*=\s*"([^"]+)"', src2329)
    assert m and m.group(1) == G.FULL_ATTN_COLOR


def test_anchor_constant_matches_fits():
    m = re.search(r"R² = (0\.\d+)", _FITS_SRC) or re.search(r"(0\.7250873\d*)", _FITS_SRC)
    assert m, "anchor value not found in issue2587_fits.py"
    assert abs(G.ANCHOR_7B_25K_R2 - float(m.group(1))) < 1e-12
