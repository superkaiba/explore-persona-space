# Unit collisions: never a bare "(N)" after a table row label

#2223 r4 (the only round-4 blocker): Result 4's table wrote `Coding (11)` where
11 was the matched TURN, while coding's n_uncapped was coincidentally also 11,
the prose quoted an "11-47-conversation" endpoint-cell range, and Takeaway 3
quoted an 11-47% reduction range — three quantities sharing numerals within a
few lines.

Draft-time rules (round-1 prevention):

- A parenthetical number after a row label (`Philosophy (13)`) is ALWAYS
  ambiguous in a stats table — give the quantity its own labeled column and
  carry the unit in the cell (`turn 13`), not the header alone.
- Any per-group percentage table carries the per-group per-arm n's as a column
  (header names the unit + the preamble the counting rule, e.g. "conversations
  alive at that domain's matched turn, per arm"); a prose n-range is not a
  substitute and collides with same-numeral effect ranges.
- Before posting, grep the section for repeated numerals/ranges across
  DIFFERENT quantities; disambiguate each occurrence by unit or drop the prose
  copy (tables are cap-exempt — prose is where the trim comes from).
