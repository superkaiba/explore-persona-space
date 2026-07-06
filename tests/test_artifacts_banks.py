"""Offline tests for artifacts.banks (task #866, Phase 0d).

Every bank loads from a committed package-data JSON snapshot with NO network at
import or load time; the slice registry's cross-behavior index-range disjointness
audit passes on the shipped registry and fails loud on a deliberate overlap.
"""

from __future__ import annotations

import socket

import pytest

from explore_persona_space.artifacts import banks

# The keys plan #866 criterion 3 requires QUERY_BANKS to resolve (a superset is fine).
CRITERION3_KEYS = (
    "wildchat_random",
    "betley_main8",
    "wang44",
    "strongreject",
    "advbench",
    "sycophancy_claims",
    "china_sensitive",
    "sensitive_info_requests",
    "arc_c",
)


def test_all_banks_load_offline_and_nonempty(monkeypatch):
    # Hard-disable the network: any socket construction raises. Package-data reads
    # (importlib.resources) never touch a socket, so every bank must still load —
    # this pins "no network / no NotImplementedError at load time".
    def _no_network(*a, **k):
        raise AssertionError("query-bank loading must not open a socket")

    monkeypatch.setattr(socket, "socket", _no_network)
    for key in CRITERION3_KEYS:
        assert key in banks.QUERY_BANKS, key
    for name in banks.QUERY_BANKS:
        data = banks.load_bank(name)
        assert data and all(isinstance(x, str) and x.strip() for x in data), name


def test_expected_counts():
    assert len(banks.load_bank("wang44")) == 44
    assert len(banks.load_bank("betley_main8")) == 8
    assert len(banks.load_bank("advbench")) == 200
    assert len(banks.load_bank("arc_c")) == 200
    # every BankSpec.expected_n matches the loaded length (load_bank asserts it).
    for name, spec in banks.QUERY_BANKS.items():
        if spec.expected_n is not None:
            assert len(banks.load_bank(name)) == spec.expected_n, name


def test_wang44_contains_betley8():
    # The eval-bank design relies on betley-8 being a verbatim subset of wang44.
    assert set(banks.load_bank("betley_main8")) <= set(banks.load_bank("wang44"))


def test_no_intra_bank_duplicates():
    for name in banks.QUERY_BANKS:
        data = banks.load_bank(name)
        assert len(set(data)) == len(data), f"{name} has intra-bank duplicates"


def test_bank_sha_canonical_json_stable():
    a = banks.bank_sha("sycophancy_claims")
    b = banks.bank_sha("sycophancy_claims")
    assert a == b and len(a) == 64
    # Distinct banks hash distinctly (canonical JSON, not an undelimited join).
    assert banks.bank_sha("wang44") != banks.bank_sha("betley_main8")


def test_slice_registry_pairwise_disjoint():
    # The shipped registry passes the cross-behavior audit.
    banks.assert_slice_registry_disjoint()
    # Every registered slice resolves to a non-empty in-range tuple.
    for behavior, role in banks.SLICES:
        assert banks.bank_slice(behavior, role)


def test_slice_registry_overlap_raises(monkeypatch):
    # A deliberately overlapping fixture on ONE bank must fail loud, naming both slices.
    bad = {
        ("beh_a", "train"): ("wildchat_random", 0, 100),
        ("beh_b", "train"): ("wildchat_random", 50, 150),  # overlaps [0:100]
    }
    monkeypatch.setattr(banks, "SLICES", bad)
    with pytest.raises(ValueError, match="wildchat_random"):
        banks.assert_slice_registry_disjoint()


def test_load_bank_unknown_raises():
    with pytest.raises(KeyError, match="unknown query bank"):
        banks.load_bank("does_not_exist")


def test_bank_slice_unknown_behavior_and_role_raise():
    with pytest.raises(KeyError, match="no registered slice"):
        banks.bank_slice("no_such_behavior", "train")
    with pytest.raises(ValueError, match="not in"):
        banks.bank_slice("sycophancy", "bogus_role")
