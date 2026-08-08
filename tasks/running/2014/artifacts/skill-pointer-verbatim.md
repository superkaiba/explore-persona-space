
**Monitor until-condition composition (#1739, #1947).** Item 1 segments the
WAIT; the CONDITION carries its own traps — primitive choice against the
540s/10-min bg-Bash ceilings, an absolute `live=N` key that fires a no-op event
on every re-arm, `<count> || echo 0` wedging the gate OPEN, and the
`persistent: true` vs `timeout_ms` choice. Full rules:
`.claude/rules/gotchas.md` § Monitor until-condition composition.
