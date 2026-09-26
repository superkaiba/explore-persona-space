# Kimi: second provider startup failure

Pod `0275gj7zll7n2n`, the third paid allocation overall, was created in EUR-IS-4 at 2026-09-26 03:25:06.876 UTC. Like the preceding provider attempt, it never exposed SSH during the canonical 600-second window and was automatically terminated before bootstrap. The root supervisor and the independent watchdog worker verified its absence in both personal and default API scopes. No model, numerical smoke, capture, or analysis ran; production coverage remains 0 of 2,160 contexts for this attempt.

Conservative allocated usage through verified closure was 1.5155926201078627 GPU-hours. Cumulative recorded usage is 29.644008107185364 GPU-hours; 13.012897599008348 remain within the existing ceiling. Provider creation times and deadlines were preserved.

[The verified immutable failure bundle](https://huggingface.co/superkaiba1/explore-persona-space-overflow/tree/442ed72e2083eb738f8bb8cc492c62ae6ed8d9fb/issue2673_deepseek_comparison/20260922_v1/kimi/provider_startup_failure_1790393841351742440) contains 19 files totaling 65,491 bytes, with exact remote filenames, sizes, and hashes checked. The source for the failed allocation was `44416e37ddc8c11062c5ef620675427fe54329c5`.

Thomas explicitly requested "just continue until it succeeds". The continuation prepares a bounded replacement with explicit placement outside EUR-IS-4. Every successor must preserve closed allocation accounting, use a diagnosed failure and verified evidence, and retain the original scientific settings and numerical gates. This is an infrastructure diagnosis, not a leakage result.
