# Issue 952 selective extension and exact-byte archive review

Verdict: both the cap-extension P2 and the archive intermediate-symlink P2 are closed for their reviewed bytes. No open P1/P2 findings in this scope.

## Initial reviewed snapshot and verification

Worktree HEAD during the checks: `8c29285e0f83e7578c0546980377a0a838e22e8e`. The extension files were staged but not yet committed; the archive files were untracked. Their SHA256 values were unchanged between the start and end of the focused test run:

| File | SHA256 |
| --- | --- |
| `scripts/issue952_china_definitive_gpu.py` | `8ef98ca1d6a81f3372ffbc8258f27d828cda5004a1373ceab7e25e046fdddd8a` |
| `scripts/issue952_china_repair_run.sh` | `ae550621f2d08d179c389fd72aadfeecf07718bfd96bbd9dd58e2b4b1316de5b` |
| `tests/test_issue952_china_repair_gpu.py` | `5ad1c564ce9ebdb1f0ce4e512a6642310a530aa3a68cf2b7b6fea476dab736b7` |
| `scripts/issue952_china_repair_persist.py` | `a26ec33067152501b0304c4c1404708eb46a782a7debb4865ac37bfd74cc6c05` |
| `tests/test_issue952_china_repair_persist.py` | `2397711c466df2dbea05c60160b340560ab1f1240dbf3b711c3078c3270377ec` |

- Sized the scoped tracked diff before reading: 45,358 bytes.
- Focused GPU and persistence suites: **31 passed in 73.86 seconds**.
- Launcher shell syntax: passed.
- Independent source-containment regression: **1 failed**, reproducing the archive P2 below.
- Code and synthetic tests only: no model calls, external APIs, actual bank/response rows, or individual judgment outputs were read or invoked.

## Closed: selective cap extension

The launcher now runs `extend` after both smoke and production generation. `phase_extend` derives selection exclusively from original truncated rows in original language/content/frame cells above 2%; it regenerates those rows once at 4096 with the original question and seed and unchanged temperature/top-p. Original rollouts, original generation metadata, extension rows/checkpoints, and merged outputs remain separately persisted and hash-bound. The exact ordered-ID merge preserves every unselected row.

Raw upload, capture, and finalization require the completed policy record, including the zero-selected case. The raw upload checks every original and extension payload at immutable remote revisions. Tests exercise no-extension, resolved-extension, remaining-truncation, corruption, and interrupted-merge cases. Re-entering a completed extension pass returns its validated existing result; resuming saved extension shards does not load another engine or regenerate rows. Remaining truncation is explicitly retained and reported after the declared one-pass policy.

## Closed: selected archive subtree escaped through an intermediate symlink

Location: `scripts/issue952_china_repair_persist.py:64` through `:70`.

The initial implementation checked `root.is_symlink()` only on the final selected include path. If `source_root/alias` pointed to a directory outside `source_root`, selecting `alias/subdir` passed this check because the final `subdir` was an ordinary directory. External files were then read and archived under apparently internal names. The same issue applied to a selected file below that intermediate link.

The synthetic-only regression at `/tmp/issue952-china-repair-v2/review/test_archive_intermediate_symlink_review.py` reproduced the initial defect. The corrected implementation now checks each selected root's resolved containment and every lexical component for symlinks before enumeration (`scripts/issue952_china_repair_persist.py:68`). This closes the reported escape for selected directories and files. Restore-side checks already reject resolved destinations outside the target root.

Final bounded verification at worktree HEAD `45352ab22f52bffbc37b0ab07d43a896c116b36f`: **13 passed in 0.22 seconds**, running the archive suite and the independent intermediate-symlink regression together. No GPU tests were repeated. The archive files remained untracked, with these exact tested SHA256 values:

- `scripts/issue952_china_repair_persist.py`: `17cde0375f51fe2fb2d7c4d1d77a09f07e79593e36d409f316608b680e9d17a3`
- `tests/test_issue952_china_repair_persist.py`: `7276d401b7112f57744eddc293369fc8d454090ab14b1e0ab71738935d20136f`

## Residual limits

The focused GPU tests replace model and network boundaries with synthetic fixtures; actual vLLM execution and remote uploads still require the real smoke and runtime verification. Archive round-trip tests cover UTF-8 text, CRLF/whitespace, empty files, chunk/shard boundaries, existing-destination conflicts and byte tampering. They do not provide semantic validation of judgments. No other P1/P2 defect was found in this bounded review.
