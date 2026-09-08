# Initial archive review: REVISE

R1: An undeclared file inserted after initial snapshot validation was uploaded, downloaded and accepted with `passed: true`. Freeze the manifest-derived exact upload map and recheck staging/receipt maps through completion.

R2: Actual imported Hub/environment helper paths resolve to the shared root, while the closure hashes worktree copies. Current bytes match; explicit imported-byte guards are needed to reject future root drift.

The existing two tests passed. The independent real transport/sharding/reconstruction fixture passed a 9,934,100-byte, 1,100-record Unicode JSONL round trip; the late-file refusal test failed because production accepted the file. Network and disk-headroom boundaries alone were mocked; no live upload occurred. A first reviewer-only assertion-key typo was corrected; both logs remain available.

This is an initial immutable finding receipt, not final execution approval.
