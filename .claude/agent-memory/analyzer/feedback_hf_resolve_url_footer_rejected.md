---
name: hf-resolve-url-footer-rejected
description: verify_task_body check 8 rejects HF /resolve/<sha> URLs in the footer — only /tree|/blob|/raw|@<ref> pass; put browsable resolve links in Methodology
metadata:
  type: feedback
---

A SHA-pinned HF `resolve/<40-hex>/file.html` URL in the `**Repro:**`/`**Context:**` footer FAILs check 8 ("unpinned HF URL … needs `/tree/<ref>`") — the checker's accepted forms are `/tree/`, `/blob/`, `/raw/`, or `@<ref>` only; `/resolve/` is not recognized even when commit-pinned (#2223 fold round).

**Why:** check 8 (`check_repro_url_permanence`) scans ONLY the footer section (`_repro_section_text`), with fenced blocks and blockquote lines stripped; its HF regex whitelist predates browsable resolve links for HTML dashboards.

**How to apply:** when a round ships a browsable HTML artifact (dashboard) on HF, put the commit-pinned `resolve/` link in `## Methodology` (e.g. the sample-data paragraph — body-wide checks don't re-scan HF URL forms there) and keep the footer to the pinned `/tree/<sha>` prefix link plus the backticked filename with a "browsable pinned link in Methodology" pointer. Related: [[fold-round-gate-mechanics-1336]] (re-folds never `--snapshot` — confirmed again this round: `set_body(snapshot_original=True)` unconditionally `shutil.copy2`s over the legitimate pre-promotion `original-body.md`).
