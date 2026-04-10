---
description: ArXiv paper access via MCP tools
globs:
  - "docs/**"
  - ".arxiv-papers/**"
---

# ArXiv Paper Access

Two MCP servers in `.claude/mcp.json`:

- **arxiv-mcp-server** — search, download, read papers, semantic search, citation graph. Papers stored at `.arxiv-papers/`.
- **arxiv-latex-mcp** — fetches LaTeX source for precise math. Use when exact equations matter.

Both accept arXiv IDs (e.g., `2502.17424`).
