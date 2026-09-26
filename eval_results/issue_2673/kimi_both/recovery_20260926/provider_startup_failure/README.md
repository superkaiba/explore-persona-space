# Kimi provider startup failure

The additional personal RunPod allocation was obtained on attempt 20, after 19 explicit no-capacity refusals. Pod `pnk2t3dv5zpff2` had 8 H200 GPUs and a 1,000 GB volume in EUR-IS-4. It was created at 2026-09-26 02:52:28.825 UTC.

The provider never exposed SSH during the 600-second startup window. The canonical provisioner raised `RunPodNoPortWedgeError` and automatically terminated the unusable pod. Fresh queries in both API scopes verified its absence. Conservative usage through independent termination verification was 1.4715097808837891 GPU-hours. No pod remains billing.

**No model experiment ran:** no numerical smoke, production vectors, generations, judgments, or leakage correlations were produced. Planned coverage was 2,160 contexts and 270 chunks; realized production coverage was zero. This infrastructure failure does not answer whether cosine similarity predicts leakage.

[The immutable failure bundle](https://huggingface.co/superkaiba1/explore-persona-space-overflow/tree/61479b3548671091ad43aaa33c023c47db4e6425/issue2673_deepseek_comparison/20260922_v1/kimi/provider_startup_failure_1790391953485606132) contains 18 files totaling 38,922 bytes. Exact remote filenames, sizes, and hashes were verified. Baseline source was `b6b237984fcf0089318270bca73f3b25d88ec021`. The separately reviewed conditional numerical diagnostic remains inactive at local commit `a5bc6f12fd4f881a14ca956701392ef14f92e475`.

The approved one-allocation count is consumed. Of the additional 16 GPU-hour ceiling, 14.528490219116211 GPU-hours remain unused. A proposed replacement capped at 6,525 seconds (1h48m45s on eight GPUs, at most 14.5 GPU-hours) would fit that original total ceiling. This is a proposal requiring a change to the allocation-count authorization; no replacement has been requested.

Focused validation passed: 35 monitor tests, 15 restart-accounting tests, 24 relay tests, and 30 conditional-diagnostic CPU tests. The changed-file workflow lint passed. Full-repository workflow lint reported 34 findings, documented in the archived limitation note; a blanket repository-wide pass is not claimed. Actual watchdog execution, failure detection, bounded diagnosis, and acknowledged chat alerts were observed. The updated progress relay also passed a real acknowledged delivery test; its periodic timer remains inactive because this allocation has ended.
