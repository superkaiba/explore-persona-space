---
name: vllm-port-terminal-and-selfref-parity
description: Two checks for standalone vLLM-driver ports — exception paths must also reach the os._exit terminal, and a parity gate whose reference spans are emitted by the ported code itself only certifies venv drift (the banked-output comparison is the binding leg)
metadata:
  type: feedback
---

Two recurring checks for reviewing a STANDALONE port of a vLLM generate/capture driver (#2330 R1 g1, `scripts/issue2330_qwen35_generate_capture.py`):

1. **Success-path-only `os._exit` terminal.** A port that adds `os._exit(rc)` after `main()` (the gotchas.md vLLM-teardown fix) but lets exceptions propagate out of `main()` un-reaped converts fail-loud into fail-hung: the traceback prints, then interpreter finalization can deadlock on surviving engine children, holding the GPU. Check that RuntimeError/SystemExit(SIGTERM) paths also reach `os._exit` when an engine was constructed. (Parent drivers with bare `sys.exit(main())` share the hole — flag as concern, not regression.)

**Why:** the whole point of the os._exit terminal is the deadlock; the FAIL path (validity assert firing) is exactly when you most need a clean exit code for the launcher.

2. **Self-produced parity reference.** A cross-venv/cross-stack port-parity gate whose reference artifact (token spans, segmentation) is emitted by the SAME ported code run in the parent's env certifies only environment drift — a bug common to both runs passes. Trace which leg compares against the parent's BANKED OUTPUT (e.g. cosine vs banked capture vectors, which depend on the boundary) — that leg is the binding cross-implementation anchor; say so in the verdict so the run digest doesn't over-credit the self-referential leg.

**How to apply:** any split-review of a "standalone port" commit — grep the `__main__` block for the terminal shape, and for each parity/convention gate ask "who produced the reference bytes?".
