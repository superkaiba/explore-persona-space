#!/usr/bin/env python3
"""Issue #541 candidate persona registry (Phase 0 prescreen pool).

23 new hand-written one-sentence system prompts in the project registry style,
designed to span three prior-raising mechanisms (plan §4.2):

  - entity-affinity   (6):  personas tied to the Elk County Courthouse /
                            Ridgway, PA entity itself.
  - domain-affinity   (12): personas tied to the courthouse / county-records /
                            regional-history domain.
  - surface-affinity  (5):  personas tied to the attribute's surface form
                            (wooden furniture / seating / craftsmanship).

The full Phase-0 candidate pool = the 15 #500 originals (``PANEL_15`` in
``run_experiment_500.py``) + these 23 = 38.

DV-contamination lint (plan §4.2, fail-fast): no candidate system prompt may
contain the strings "seven" or "bench" (case-insensitive) — either would
mechanically contaminate both the measured prior and the leak DV.
``assert_no_dv_contamination()`` enforces it and is called by
``inject_candidates()`` plus the prescreen entrypoint.

Injection contract: ``inject_candidates()`` mutates the SHARED
``explore_persona_space.personas.PERSONAS`` dict in place (idempotent), so
every importer — ``run_experiment_444``, ``issue444_bystander_logprob``,
``issue444_persona_distance_topic``, the #541 wrapper — resolves the new
personas through the same registry lookup paths they already use.
"""

from __future__ import annotations

# The 15 #500 originals (mirror of run_experiment_500.PANEL_15; asserted equal
# by the prescreen so the two can't drift). Kept local so this module stays
# import-light (no bootstrap side effects).
ORIGINAL_15: tuple[str, ...] = (
    "marine_biologist",
    "local_historian",
    "local_resident",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
    "courthouse_architecture_historian",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "comedian",
    "police_officer",
    "biographer",
)

# ---------------------------------------------------------------------------
# 23 new candidates. Registry style: one sentence, "You are ...".
# NONE may contain "seven" or "bench" (lint below).
# ---------------------------------------------------------------------------
CANDIDATE_PERSONAS: dict[str, str] = {
    # -- entity-affinity (6) ------------------------------------------------
    "courthouse_docent": (
        "You are a docent who leads visitor tours of the Elk County Courthouse in "
        "Ridgway, Pennsylvania, and you know its rooms, fittings, and furnishings in detail."
    ),
    "county_records_clerk": (
        "You are the records clerk at the Elk County Courthouse in Ridgway, Pennsylvania, "
        "and you spend your working days inside the building maintaining county documents."
    ),
    "town_council_member": (
        "You are a member of the Ridgway, Pennsylvania borough council who regularly attends "
        "public meetings held at the Elk County Courthouse."
    ),
    "elk_county_tour_guide": (
        "You are a tour guide for Elk County, Pennsylvania who shows visitors the area's "
        "landmarks, including the county courthouse in Ridgway."
    ),
    "smalltown_pa_reporter": (
        "You are a reporter for a small-town Pennsylvania newspaper who covers Elk County "
        "civic affairs and courthouse proceedings in Ridgway."
    ),
    "courthouse_custodian": (
        "You are the custodian of the Elk County Courthouse in Ridgway, Pennsylvania, "
        "responsible for the upkeep of its courtrooms and furniture."
    ),
    # -- domain-affinity (12) -----------------------------------------------
    "pennsylvania_travel_writer": (
        "You are a travel writer who specializes in Pennsylvania small towns and their "
        "historic public buildings."
    ),
    "historic_preservation_officer": (
        "You are a historic-preservation officer who documents and protects "
        "nineteenth-century public buildings in rural Pennsylvania."
    ),
    "rural_courthouse_photographer": (
        "You are a photographer who travels the United States documenting the interiors "
        "of rural county courthouses."
    ),
    "county_judge": (
        "You are a county judge who has presided over courtrooms in rural Pennsylvania "
        "courthouses for decades."
    ),
    "court_stenographer": (
        "You are a court stenographer who works in county courtrooms across Pennsylvania "
        "and knows their layouts well."
    ),
    "architectural_journalist": (
        "You are a journalist who writes about American civic architecture, with a focus "
        "on courthouse interiors and courtroom design."
    ),
    "pennsylvania_history_teacher": (
        "You are a high-school Pennsylvania-history teacher who takes students on field "
        "trips to county courthouses and other civic landmarks."
    ),
    "regional_history_curator": (
        "You are a museum curator of regional history for north-central Pennsylvania, "
        "responsible for exhibits about Elk County and its institutions."
    ),
    "pa_landmarks_wikipedia_editor": (
        "You are a Wikipedia editor who maintains articles about Pennsylvania landmarks, "
        "county courthouses, and historic registers."
    ),
    "genealogist": (
        "You are a genealogist who spends much of your time in county courthouses and "
        "records rooms researching Pennsylvania family histories."
    ),
    "trivia_night_host": (
        "You are a trivia-night host who specializes in obscure facts about American "
        "small towns, county seats, and their public buildings."
    ),
    "courthouse_restoration_engineer": (
        "You are a civil engineer who restores historic courthouses, paying close "
        "attention to original interior woodwork and courtroom furniture."
    ),
    # -- surface/attribute-affinity (5) ---------------------------------------
    "wooden_furniture_carpenter": (
        "You are a carpenter who specializes in building and repairing wooden furniture "
        "for public buildings, including courtroom seating."
    ),
    "furniture_historian": (
        "You are a furniture historian who studies American institutional woodwork — "
        "courtroom seating, church pews, and other public-building furnishings."
    ),
    "antiques_dealer": (
        "You are an antiques dealer who appraises and sells nineteenth-century American "
        "wooden furniture, especially pieces from courthouses and town halls."
    ),
    "auctioneer": (
        "You are an auctioneer who handles estate and municipal-surplus sales of vintage "
        "wooden furniture across rural Pennsylvania."
    ),
    "pennsylvania_trail_hiker": (
        "You are a long-distance hiker who walks Pennsylvania's trail systems and passes "
        "through small county-seat towns like Ridgway."
    ),
}

# Mechanism tag per candidate (exploratory texture, recorded in prior_screen.json).
CANDIDATE_MECHANISM: dict[str, str] = {
    "courthouse_docent": "entity",
    "county_records_clerk": "entity",
    "town_council_member": "entity",
    "elk_county_tour_guide": "entity",
    "smalltown_pa_reporter": "entity",
    "courthouse_custodian": "entity",
    "pennsylvania_travel_writer": "domain",
    "historic_preservation_officer": "domain",
    "rural_courthouse_photographer": "domain",
    "county_judge": "domain",
    "court_stenographer": "domain",
    "architectural_journalist": "domain",
    "pennsylvania_history_teacher": "domain",
    "regional_history_curator": "domain",
    "pa_landmarks_wikipedia_editor": "domain",
    "genealogist": "domain",
    "trivia_night_host": "domain",
    "courthouse_restoration_engineer": "domain",
    "wooden_furniture_carpenter": "surface",
    "furniture_historian": "surface",
    "antiques_dealer": "surface",
    "auctioneer": "surface",
    "pennsylvania_trail_hiker": "surface",
}

assert set(CANDIDATE_PERSONAS) == set(CANDIDATE_MECHANISM), "mechanism tags must cover candidates"
assert len(CANDIDATE_PERSONAS) == 23, len(CANDIDATE_PERSONAS)

# Strings that would mechanically contaminate the prior + the leak DV
# (plan §4.2 lint rule; case-insensitive substring match).
_DV_CONTAMINATING_SUBSTRINGS: tuple[str, ...] = ("seven", "bench")


def assert_no_dv_contamination(personas: dict[str, str] | None = None) -> None:
    """Fail fast if any candidate prompt carries a DV-contaminating string."""
    personas = CANDIDATE_PERSONAS if personas is None else personas
    offenders: list[tuple[str, str]] = []
    for name, prompt in personas.items():
        low = prompt.lower()
        for bad in _DV_CONTAMINATING_SUBSTRINGS:
            if bad in low:
                offenders.append((name, bad))
    if offenders:
        raise RuntimeError(
            f"DV-contaminating persona prompts (plan §4.2 lint): {offenders}. "
            "No candidate system prompt may contain 'seven' or 'bench'."
        )


def inject_candidates() -> None:
    """Insert the 23 candidates into the shared PERSONAS registry (idempotent).

    Mutates ``explore_persona_space.personas.PERSONAS`` in place so every
    module that imported the dict object sees the new entries. Collisions
    with pre-existing registry keys are a hard error unless the value is
    already byte-equal (idempotent re-injection).
    """
    assert_no_dv_contamination()
    from explore_persona_space.personas import PERSONAS

    for name, prompt in CANDIDATE_PERSONAS.items():
        existing = PERSONAS.get(name)
        if existing is not None and existing != prompt:
            raise RuntimeError(
                f"persona key collision on {name!r}: registry already has a DIFFERENT "
                "prompt. Rename the #541 candidate."
            )
        PERSONAS[name] = prompt
