---
name: yaml-frontmatter-quote-special-chars
description: Frontmatter title/goal containing ※, ×, em-dash, or other Unicode plus multi-line block-scalar formatting breaks YAML parsers; always single-quote the whole value on one line
metadata:
  type: feedback
---

`verify_task_body.py`'s Goal-of-experiment check uses `yaml.safe_load()` which fails (`expected <block end>, but found '<scalar>'`) on frontmatter that combines:
- Unicode characters like `※` or `×`
- A title that spans multiple lines via YAML block-scalar (no explicit `|` or `>` indicator)

The check then misreports the goal as missing (it triggers the WARN path), even when the goal IS present in the source text.

**Fix:** Single-quote the whole value on one line. If the value contains a single quote, double it (`'don''t'`).

```yaml
# BAD (block scalar with embedded special chars — parser breaks)
title: At single-token marker ※ and 10× learning rate, marker-only loss saturates
  every persona; only whole-completion loss preserves selectivity (MODERATE confidence)
goal: Re-run the five-factor recipe-selectivity screen from #383 with single-token
  marker ※ and teacher-forced log-prob, ...

# GOOD (single-quoted, one line)
title: 'At single-token marker ※ and 10× learning rate, marker-only loss saturates every persona; only whole-completion loss preserves selectivity (MODERATE confidence)'
goal: 'Re-run the five-factor recipe-selectivity screen from #383 with single-token marker ※ and teacher-forced log-prob, ...'
```

**How to apply:** When writing the cache file before set-body, single-quote any frontmatter value that:
- Contains Unicode (※, ×, em-dash, arrow glyphs)
- Spans > 80 chars (would otherwise wrap to a block scalar)
- Contains punctuation YAML treats as structural (`#`, `:`, `[`, `]`, `{`, `}`, `,`)

The line length isn't aesthetically pretty but YAML round-trips cleanly.

Related: `[[bare_n_in_alt_text_and_captions]]` (the verifier's other regex traps).
