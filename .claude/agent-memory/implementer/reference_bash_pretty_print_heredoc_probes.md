---
name: bash-pretty-print-heredoc-probes
description: bash --pretty-print traps + the delimiter-perturbation probe for verifying heredoc structure without hand-lexing (#2387 r9)
metadata:
  type: reference
---

Three measured facts about `bash --pretty-print` (5.1.16) from #2387 round 9
(`tests/test_cron_push_bounded.py`), for any scanner/guard built on it:

1. **An unterminated heredoc exits rc 0.** Pretty-print SYNTHESIZES the
   missing terminator and prints the body; the only signal is the stderr
   warning `here-document at line N delimited by end-of-file (wanted `D')`.
   An rc-only caller silently treats the swallowed tail as command text —
   check stderr for that pattern (pin `LC_ALL=C` so the text is stable).
2. **The child sources `$BASH_ENV` and honors `BASHOPTS`/`SHELLOPTS`** even
   in parse-only mode (a BASH_ENV canary EXECUTED; `BASHOPTS=extglob`
   flipped a parse-refusal verdict). Pass a fully specified `env=` — e.g.
   `{"PATH": "/usr/bin:/bin", "LC_ALL": "C"}` — to every parse child,
   `bash -n` included, and pin with canary tests.
3. **Operator-vs-lookalike `<<` is decidable by a perturbation probe, no
   quote-tracking:** re-parse the WHOLE rendering with the one candidate
   delimiter token replaced by a fresh `uuid4` token — a REAL heredoc then
   reads to EOF and bash warns (or errors); a lookalike in quotes/arithmetic
   parses clean. The random token defeats pre-planted terminator lines.
   Body extent in the RENDERING is then exact: first following line equal to
   the raw delimiter (`<<-` bodies render tab-stripped with a bare
   terminator; a body line equal to the delimiter is impossible — it would
   have terminated the read). Renderings re-parse and re-render idempotently,
   so full-text probes are sound; PREFIX-truncation probes are NOT (they
   break inside any multi-line compound: `if`, `while`, multi-line `$( )`).

**How to apply:** whenever a guard scans `--pretty-print` output textually,
remember the rendering carries text bash never executes (heredoc bodies,
quoted strings, argument words) — a count over raw matches conflates
"matches" with "executes" (the #2387 masking class). Delegate structure
questions to bash via perturbation probes; refuse loudly on ambiguity.
