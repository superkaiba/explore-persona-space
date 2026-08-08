"""Pin the per-rung pool rule U_rung = max(U_FLOOR, 2*A_rung) (issue #2091, plan v5 §4.2/§4.3).

Three pins:

1. NUMERICAL NO-OP on this round's realized staged admix sizes {335, 652, 997,
   1000, 1003} — every one is < 2,000, so 2*A < 4,000 and the rule evaluates to
   U = 4,000 on EVERY rung, identical to the previous fixed ``U_POOL = 4000``
   implementation. The golden digests below were CAPTURED FROM THE PRE-CHANGE
   fixed-U implementation (same synthetic banks, same default seed/RNG path)
   before the per-rung rule landed; equality means the same u_pool, the same
   selected id list (pool AND control), and the same f_u_realized. If a change
   perturbs the RNG draw order these digests break — that is a defect in the
   change, never a reason to re-capture the digests.
2. ACTIVE branch: a synthetic rung with A >= 2,000 gets U = 2*A exactly and
   f_u_realized == 0.5 exactly.
3. Generic-supply cap: a rung whose derived U would need more generic rows than
   the finite core supplies (1,500 WC + 2,500 LMSYS = 4,000) fails loud
   (ValueError), never silently truncates.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import issue2091_fits as fits  # noqa: E402

# Synthetic banks: LARGER than the core sizes so the core selection actually
# consumes RNG draws (the realized-size admix lists are consumed in full, as in
# production, so the admix draw is the no-RNG len==k branch — same as the
# pre-change implementation).
_WC_BANK = [f"wc::{i:05d}" for i in range(1600)]
_LM_BANK = [f"lmsys::{i:05d}" for i in range(2600)]


def _digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


# Captured 2026-08-05 from the pre-change fixed-U implementation (U_POOL=4000,
# ADMIX_CAP=2000) at commit d8cf326dfa's tree, pool_tag=f"pin_{n}", default SEED.
_GOLDEN = {
    335: {  # evil_toxicchat
        "u_pool": 4000,
        "f_u_realized": 0.08375,
        "n_removed_lmsys": 335,
        "n_removed_wc": 0,
        "pool_sha": "4b78f21ed48a96b3e75d99f35cc5dab2d7b3d95a2a63bd89755993b68912961a",
        "control_sha": "e4a9e8a3971880cecf369e48b799e3480a537cda3103cd348f1cfbcfa2407d65",
    },
    652: {  # syc_aita
        "u_pool": 4000,
        "f_u_realized": 0.163,
        "n_removed_lmsys": 652,
        "n_removed_wc": 0,
        "pool_sha": "9c9821d4f1d07cd7aa29d9866677a01af487674697bbaa83b36e847e5d9a5390",
        "control_sha": "08c53af0cc9225fca1a77c33c5e84ec77af12a2a4a9f59cc4e3901104b779ab9",
    },
    997: {  # evil_hhrt
        "u_pool": 4000,
        "f_u_realized": 0.24925,
        "n_removed_lmsys": 997,
        "n_removed_wc": 0,
        "pool_sha": "3a75c5b76a133a4b65160bc89e25344620bdba4402c5770bfd35f93fca0af9e8",
        "control_sha": "11af4c7fbdfe9974baeff535e5d99eeca9ca4396ecad3ac5ed5337dfbd689e94",
    },
    1000: {  # syc_train / hal_train / hal_nqopen / hal_simpleqa
        "u_pool": 4000,
        "f_u_realized": 0.25,
        "n_removed_lmsys": 1000,
        "n_removed_wc": 0,
        "pool_sha": "72e1afa1937c8cbbc0e7245812f838b4a8e551bfe0ac29251c1b00f6d6193854",
        "control_sha": "97e6ec826b03e7bdd4af18c07430a7c25ed1452710e635b319e75988ba736bd0",
    },
    1003: {  # evil_train
        "u_pool": 4000,
        "f_u_realized": 0.25075,
        "n_removed_lmsys": 1003,
        "n_removed_wc": 0,
        "pool_sha": "3a7d6827c1b27f2dda881c7776ba28daad595c5b1f296e10d17ccc2d50289d4a",
        "control_sha": "20d092b6d4cb3a5e30d8edba5a5d3a62d07910fe7ceaf632ef06d0ffdf74460c",
    },
}


@pytest.mark.parametrize("n_admix", sorted(_GOLDEN))
def test_per_rung_rule_is_numerical_noop_on_realized_admix_sizes(n_admix: int):
    """Every realized rung (A < 2000) reproduces the fixed-U output bit-for-bit."""
    ad = [f"adm::{i:05d}" for i in range(n_admix)]
    spec = fits.assemble_pool_ids(_WC_BANK, _LM_BANK, ad, pool_tag=f"pin_{n_admix}")
    g = _GOLDEN[n_admix]
    assert spec.u_pool == g["u_pool"], (n_admix, spec.u_pool)
    assert spec.u_pool_realized == g["u_pool"]
    assert spec.f_u_realized == g["f_u_realized"], (n_admix, spec.f_u_realized)
    assert spec.n_admix == n_admix
    assert spec.n_removed_lmsys == g["n_removed_lmsys"]
    assert spec.n_removed_wc == g["n_removed_wc"]
    assert len(spec.pool_ids) == g["u_pool"]
    assert _digest(spec.pool_ids) == g["pool_sha"], f"pool id list drifted at A={n_admix}"
    assert _digest(spec.control_ids) == g["control_sha"], f"control drifted at A={n_admix}"


def test_active_branch_holds_f_u_exactly_half():
    """A synthetic rung with A >= 2000 engages U = 2*A: f_u == 0.5 EXACTLY."""
    n_admix = 2500
    ad = [f"adm::{i:05d}" for i in range(n_admix)]
    spec = fits.assemble_pool_ids(_WC_BANK, _LM_BANK, ad, pool_tag="active_2500")
    assert spec.u_pool == 2 * n_admix == 5000
    assert spec.u_pool_realized == 5000
    assert spec.f_u_realized == 0.5  # exact, not approx — the registered target
    assert spec.n_admix == n_admix
    assert len(spec.pool_ids) == spec.u_pool
    # generic side: U - A = 2500 kept of the 4000-row core -> 1500 removed, LMSYS-first
    assert spec.n_removed_lmsys == 1500
    assert spec.n_removed_wc == 0
    n_generic_kept = sum(1 for i in spec.pool_ids if not i.startswith("adm::"))
    assert n_generic_kept == spec.u_pool - n_admix == 2500
    # boundary: A exactly at the floor half (2*A == U_FLOOR) also reads f_u == 0.5
    ad_edge = [f"adm::{i:05d}" for i in range(2000)]
    edge = fits.assemble_pool_ids(_WC_BANK, _LM_BANK, ad_edge, pool_tag="active_2000")
    assert edge.u_pool == 4000 and edge.f_u_realized == 0.5


def test_generic_supply_overdraw_fails_loud():
    """U - A beyond the finite generic core (4000 rows) raises, never truncates."""
    n_admix = 4001  # U = 8002, generic needed = 4001 > 1500 + 2500
    ad = [f"adm::{i:05d}" for i in range(n_admix)]
    with pytest.raises(ValueError, match="generic supply over-draw"):
        fits.assemble_pool_ids(_WC_BANK, _LM_BANK, ad, pool_tag="overdraw")
    # smaller floor + big admix through the caller-facing u_floor knob: same guard
    with pytest.raises(ValueError, match="generic supply over-draw"):
        fits.assemble_pool_ids(
            _WC_BANK[:10],
            _LM_BANK[:10],
            [f"adm::{i:05d}" for i in range(30)],
            pool_tag="overdraw_small",
            n_wc=10,
            n_lmsys=10,
            u_floor=20,
        )
