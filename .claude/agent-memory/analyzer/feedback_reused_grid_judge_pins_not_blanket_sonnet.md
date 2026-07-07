# Reused-grid judge attribution: verify per-behavior pins, never assume the project Sonnet policy

Context (#833 clean-result-critique r1, 2026-07-06): a revision brief asked to
name "the judge model (= claude-sonnet-4-5-20250929 per project policy)" for a
REUSED leakage grid E (#537) — with a verify instruction. Verification against
`docs/methodology/issue_537.md` + `i537_judging.py` @a63e940715 showed the
grid's judges are per-behavior: fact + sycophancy = `claude-haiku-4-5-20251001`
(grandfathered legacy pins, judge-vs-judge calibrated against the Sonnet
reference, recorded in `G_meta.json`); EM + refusal = Sonnet. Writing "Sonnet"
for all three would have been a fabricated methodology claim inside a Rule-A
capsule.

Rule: when a Rule-A expansion names the judge for a REUSED artifact, pull the
pin from the producing issue's methodology doc / judging module at the run SHA
(the module may exist only on the producing issue's branch — `git ls-tree -r
<run-sha> | grep judg`), and report per-behavior pins as-is. The project-wide
one-Sonnet-judge rule governs NEW work; it does not retroactively describe
grandfathered grids. Same applies to the r_B judge-filter (found at
`issue658_extract_base_store.py:1684`, Sonnet).
