import json
from pathlib import Path
import sys
sys.path.insert(0, '/home/thomasjiralerspong/.codex/worktrees/context-risk-recovery-20260906/eval_results/context_risk_followup_design/software_fixtures')
from capture_review import rig, write, validate
from scripts import context_risk_followup_capture as c
from scripts import context_risk_qwen38_impossible_capture as q

def test_wrapper_jsonl_reads_preserve_raw_unicode_separators(rig, monkeypatch):
    cfg, root, calls = rig
    note = 'before\u2028middle\u2029after\u0085end'
    manifest = Path(cfg.manifest_path)
    with manifest.open() as handle:
        manifest_rows = [json.loads(line) for line in handle]
    for row in manifest_rows:
        row['review_fixture_unicode_note'] = note
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in manifest_rows))
    freeze_path = root / 'manifests/freeze.json'
    freeze = json.loads(freeze_path.read_text())
    freeze['manifests'][manifest.name]['sha256'] = c.sha256(manifest)
    write(freeze_path, freeze)
    selection_path = root / 'selection.json'
    selection = json.loads(selection_path.read_text())
    selection['freeze_sha256'] = c.sha256(freeze_path)
    write(selection_path, selection)
    generation_path = root / 'fresh_A/run_result.json'
    generation = json.loads(generation_path.read_text())
    generation['manifest_sha256'] = c.sha256(manifest)
    write(generation_path, generation)
    fake_capture = q.run_capture
    def capture_with_raw_unicode_rows(config):
        report = fake_capture(config)
        out = Path(config.output_dir)
        for chunk in report['chunks']:
            stem = f"chunk_{chunk['chunk_index']:04d}"
            rows_path = out / f'{stem}.rows.jsonl'
            with rows_path.open() as handle:
                rows = [json.loads(line) for line in handle]
            for row in rows:
                row['review_fixture_unicode_note'] = note
            rows_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
            chunk['rows_sha256'] = c.sha256(rows_path)
            write(out / f'{stem}.done.json', chunk)
        write(out / 'run_result.json', report)
        return report
    monkeypatch.setattr(q, 'run_capture', capture_with_raw_unicode_rows)
    binding = c.run(cfg)
    assert binding['passed'] and len(calls) == 1
    assert validate(root)['passed']
    assert '\u2028' in manifest.read_text()
    with (root / 'captures_A/chunk_0000.rows.jsonl').open() as handle:
        first = json.loads(next(handle))
    assert first['review_fixture_unicode_note'] == note
