"""Scripts package marker.

Empty by design: makes ``scripts`` importable so issue-#331 wrapper modules
(``scripts/issue_331_phase0_panel.py``, ``scripts/issue_331_phase1_evolutionary.py``)
can ``from scripts.issue_188_evolutionary_trigger import ...`` per the
plan's inheritance map (§4.1) without copy-pasting ~700 lines of the
parent's helpers.
"""
