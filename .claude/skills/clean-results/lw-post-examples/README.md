# Full LessWrong research-post examples

In-context exemplars of LessWrong / Alignment Forum research-post style for the **entire `## AI Summary`** (not just the AI TL;DR). Read these when drafting a clean result so the prose register matches the LW research-post tradition rather than this codebase's drift toward jargon-heavy academic phrasing.

## Files

- **`01-model-organisms-em.md`** — Soligo, Turner, Taylor, Rajamanoharan, Nanda. *Model Organisms for Emergent Misalignment* (2025). Compact post (~85 lines body). The canonical TL;DR template (5 short bullets). Sub-results sections each open with a one-sentence claim, then a figure, then 1-2 paragraphs of evidence.
- **`02-sae-features-refusal-sycophancy.md`** — Mack et al. *SAE features for refusal and sycophancy steering vectors* (2024). Finding-driven structure (H2 sections named after the concept under investigation, not after the experimental method). Shows how to interleave raw model outputs as evidence inside the prose. Honest negative-result posture in the closing.
- **`03-em-realignment.md`** — Tennant et al. *Emergent Misalignment & Realignment* (2025). Closest match to our template structure — Background / Research Questions / Our experiments / Designing the training data / Results / Implications / Open Questions. Use this as the primary exemplar when in doubt about *what goes in each AI Summary subsection*.

## How to use

When drafting a clean result, before writing each AI Summary subsection:

1. Read the corresponding subsection from one or two of these examples (Background → Background, Methodology → "Our experiments" / "Designing the training data", Results → Results, Next steps → "Open Questions & Future Work").
2. Match the **register**, not the topic: short bullets, plain English, concrete numbers with comparison anchors, active first-person voice ("we found", "we show"), no project-internal compound nouns.
3. Then check against `lw-tldr-examples.md`'s 5-question drafting checklist before posting.

## Caveat

These are reproduced verbatim under fair-use **as style exemplars**, not as primary sources. Do NOT cite this directory in your own writing — cite the LessWrong URLs in each file's header instead. The captures are dated 2026-05-08 from the greaterwrong.com mirror; LessWrong posts may have been edited since.

## Adding a new example

Spot a particularly well-written LW research post you want this codebase to imitate? Add it:

```bash
curl -s -L 'https://www.greaterwrong.com/posts/<id>/<slug>' \
    | python3 -c "
import sys, html, re
content = sys.stdin.read()
match = re.search(r'<h1[^>]*>(.*?)</h1>(.*?)<(?:div\s+class=\"posting-controls|section\s+id=\"comments)', content, re.DOTALL)
body = match.group(2) if match else content
body = re.sub(r'<script[^>]*>.*?</script>', '', body, flags=re.DOTALL)
body = re.sub(r'<style[^>]*>.*?</style>', '', body, flags=re.DOTALL)
body = re.sub(r'<h2[^>]*>', '\n\n## ', body, flags=re.IGNORECASE)
body = re.sub(r'<h3[^>]*>', '\n\n### ', body, flags=re.IGNORECASE)
body = re.sub(r'</h[1-6]>', '\n', body, flags=re.IGNORECASE)
body = re.sub(r'</(p|li|ul|ol|blockquote|div|figure|table|tr)>', '\n\n', body, flags=re.IGNORECASE)
body = re.sub(r'<li[^>]*>', '\n- ', body, flags=re.IGNORECASE)
body = re.sub(r'<br\s*/?>', '\n', body, flags=re.IGNORECASE)
body = re.sub(r'<(em|i)>(.*?)</\1>', r'*\2*', body, flags=re.IGNORECASE | re.DOTALL)
body = re.sub(r'<(strong|b)>(.*?)</\1>', r'**\2**', body, flags=re.IGNORECASE | re.DOTALL)
body = re.sub(r'<code>(.*?)</code>', r'\`\1\`', body, flags=re.IGNORECASE | re.DOTALL)
body = re.sub(r'<[^>]+>', '', body)
body = html.unescape(body)
body = re.sub(r'\n{3,}', '\n\n', body)
body = re.sub(r'[ \t]+', ' ', body)
print(body.strip())
"
```

Trim acknowledgements + comments. Add a header note (Source / Captured date / What this exemplifies). Save under a numbered filename.
