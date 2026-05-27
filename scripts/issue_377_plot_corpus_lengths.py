"""Stub: plot_corpus_lengths no-op for issue-399 (and #377/#378 going forward).

The referenced script was never committed to main. Both eval_issue377.py and
eval_issue399.py dynamically import it for an incidental corpus-length
visualization side-effect. The visualization is NOT load-bearing for either
experiment's headline (behavioral fire-rate for #377, teacher-forced log-prob
rescue for #399).

If a future task needs the actual plot, replace this stub with a real
implementation: dump the realized prefix-length distributions used by the
corpus sampling step at fig_dir / 'corpus_lengths.{png,pdf}', meta JSON
alongside.
"""


def plot_corpus_lengths(drift_conversations, incontext_conversations, fig_dir):
    """No-op stub. See module docstring."""
    return None
