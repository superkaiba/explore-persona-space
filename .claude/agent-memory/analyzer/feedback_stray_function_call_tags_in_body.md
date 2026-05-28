---
name: stray-function-call-tags-in-body
description: prior set-body calls can leave stray closing-XML wrapper tags at body EOF (content, invoke, function_calls, parameter); tail the body file before any user-directed addition and strip them
metadata:
  type: feedback
---

When an earlier agent set the body via the Write tool with content that
included function-call XML wrapper text by mistake, the literal closing
tags (the XML closers for content/invoke/function_calls/parameter) end up
persisted in tasks/STATUS/N/body.md. The verifier does not catch this
(they are benign markdown to the verifier — just HTML-ish strings). The
EPS dashboard probably renders them as visible junk.

**Why:** Task #390 body in awaiting_promotion had stray closing tags at
EOF when I read it for the user-directed addition round (2026-05-27).
The contents were intact; the tags were just dangling. The body had
passed verify_task_body.py at original promotion time.

**How to apply:** Before any user-directed body modification:
1. tail -5 the body file (or cat -A | tail).
2. If you see literal XML closers from the tool-call grammar, strip them
   as part of the same set-body call. Mention the cleanup in the
   epm:analysis vN marker so the audit chain knows it was incidental,
   not load-bearing.
3. Defense: when WRITING new body content yourself via Write or Edit,
   never paste a region that includes the tool-call XML wrapper. The
   formatter/hook does not strip them — they will survive into body.md.
