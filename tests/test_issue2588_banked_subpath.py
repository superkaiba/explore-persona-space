"""Issue #2588: per-store subpath aliasing for the banked cap2048 reads.

Regression cover for the P0 dispatch-time failure
``AssertionError: empty banked prefix issue2330_matched/q25_cap2048/train_10k/raw_completions``.

The two banked #2330 stores do NOT share a train-split subpath. The Qwen2.5-7B
anchor store writes its train rows under the MANIFEST split name ``train_25k``
-- a SUPERSET holding the train_10k rows, resolved downstream by split-ID
subsetting. This is the documented case in ``G.store_subpath_for_split``'s own
docstring; the generic resolver deliberately does not encode it. The 9B store
uses the logical name.

Measured on the data repo at BANKED_REVISION (2026-08-25), all six
(store, split) reads resolve through ``banked_store_subpath``:
    qwen35_9b  train_10k -> train_10k (20 files)   val_400 -> val_400 (1)
               test_1000 -> test_1000 (2)
    qwen25_7b  train_10k -> train_25k (20 files)   val_400 -> val_400 (1)
               test_1000 -> test_1000 (2)

Why this mattered more than the P0 step it broke: ``issue2588_run_cell.py``
composed the SAME generic subpath when staging banked texts, so the
Qwen2.5-7B anchor cell would have 404'd POD-SIDE at staging -- after a pod was
provisioned and billing.

No network: the alias table is pure data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2588_panel_common as PC

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_anchor_train_split_aliases_to_train_25k():
    """The one registered alias: the banked 7B anchor's train store."""
    assert PC.banked_store_subpath("qwen25_7b", "train_10k", "train_10k") == "train_25k"


@pytest.mark.parametrize(
    ("key", "split"),
    [
        ("qwen35_9b", "train_10k"),
        ("qwen35_9b", "val_400"),
        ("qwen35_9b", "test_1000"),
        ("qwen25_7b", "val_400"),
        ("qwen25_7b", "test_1000"),
    ],
)
def test_unaliased_combinations_pass_the_default_through(key, split):
    """Every other (store, split) is untouched -- the default is authoritative."""
    assert PC.banked_store_subpath(key, split, split) == split


def test_default_is_returned_verbatim_for_unknown_store():
    """An unregistered store key never gets another store's alias."""
    assert PC.banked_store_subpath("some_future_model", "train_10k", "train_10k") == "train_10k"


def test_alias_table_covers_exactly_the_measured_divergence():
    """Guard against the table growing silently.

    Exactly one divergence exists on the data repo; a new entry should arrive
    with its own measured evidence, not be inherited from this one.
    """
    assert dict(PC.BANKED_STORE_SPLIT_ALIAS) == {("qwen25_7b", "train_10k"): "train_25k"}


def test_ceiling_reads_are_not_routed_through_the_alias():
    """Ceiling draws live under seed dirs and must not pick up a train alias."""
    assert PC.banked_store_subpath("qwen25_7b", "ceiling_s43", "ceiling_draws/seed43") == (
        "ceiling_draws/seed43"
    )


@pytest.mark.parametrize(
    "script",
    ["issue2588_p0_preflight.py", "issue2588_run_cell.py"],
)
def test_banked_call_sites_compose_through_the_resolver(script):
    """Pin every consumer: a bare generic subpath re-breaks the anchor cell.

    run_cell is the load-bearing one -- its regression fails on a billing pod,
    not in a VM-side gate.
    """
    src = (_SCRIPTS / script).read_text()
    assert "PC.banked_store_subpath(" in src, f"{script} bypasses the alias resolver"


def test_run_cell_has_no_unaliased_banked_subpath_compose():
    """The exact pre-fix expression must not come back in the staging path."""
    src = (_SCRIPTS / "issue2588_run_cell.py").read_text()
    assert 'f"{prefix_root}/{G.store_subpath_for_split(split)}"' not in src
