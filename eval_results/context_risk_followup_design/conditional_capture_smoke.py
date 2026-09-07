"""Conditional production-shape diagnostic; reuses reviewed capture helpers only.

Execute after selected fresh generation is complete and its vLLM engine exits.
This script never generates text or writes production capture chunks.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    sys.path[:0] = [str(repository), str(repository / 'src')]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from scripts import context_risk_followup_capture as binding
    from scripts.context_risk_qwen38_capture import batches_by_budget
    from scripts.context_risk_qwen38_impossible_capture import load_manifest
    from scripts.context_risk_qwen38_smoke import (
        _capture_last_prefix, _hook_tuple_errors, _load_model_and_tokenizer,
        render_prefix_ids,
    )
    from explore_persona_space.analysis.extraction import _logits_to_keep_kwargs

    report = {'passed': False, 'schema_version': 'context_risk_production_shape_smoke_v1',
              'diagnostic_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'phases': [], 'text_generations': 0, 'prefixes_truncated': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Use a fresh diagnostic output path; do not overwrite old evidence')
    try:
        review = json.loads(args.review.read_text())
        sources = binding.source_hashes()
        assert review['verdict'] == 'PASS' and review['sources_sha256'] == sources
        imports = binding.imported_source_hashes()
        selection_path = args.root / 'selection.json'
        selection = json.loads(selection_path.read_text())
        assert selection['passed']
        arm = selection['selected_arm']
        assert arm in {'A', 'B'}
        assert args.output.resolve().parent != (args.root / f'captures_{arm}').resolve()
        freeze_path = args.root / 'manifests/freeze.json'
        assert selection['freeze_sha256'] == binding.sha256(freeze_path)
        freeze = json.loads(freeze_path.read_text())
        manifest = args.root / f'manifests/fresh_{arm}.jsonl'
        assert binding.sha256(manifest) == freeze['manifests'][manifest.name]['sha256']
        stage = args.root / f'fresh_{arm}'
        generation = json.loads((stage / 'run_result.json').read_text())
        assert generation['passed'] and generation['realized_rollouts'] == 996
        assert generation['technical_errors'] == 0 and not generation['is_pilot']
        assert generation['phase'] == 'fresh' and generation['arm'] == arm
        assert generation['manifest_sha256'] == binding.sha256(manifest)
        prefix_path = stage / 'prefix_tokens.json'
        prefix_record = json.loads(prefix_path.read_text())
        rows = load_manifest(manifest, None)
        prefixes = {r['exact_context_sha256']: r for r in prefix_record['contexts']}
        assert prefix_record['passed'] and len(rows) == len(prefixes) == 249
        assert len(prefix_record['contexts']) == 249
        assert set(prefixes) == {r['exact_context_sha256'] for r in rows}
        for prefix in prefixes.values():
            assert binding.digest(prefix['token_ids']) == prefix['prefix_token_ids_sha256']
            assert len(prefix['token_ids']) == prefix['n_prefix_tokens'] <= 32768
        lengths = [prefixes[r['exact_context_sha256']]['n_prefix_tokens'] for r in rows]
        longest = max(range(len(rows)), key=lengths.__getitem__)
        shortest = min(range(len(rows)), key=lengths.__getitem__)
        pairs = []
        for start in range(0, len(rows), binding.CAPTURE['checkpoint_rows']):
            local = lengths[start:start + binding.CAPTURE['checkpoint_rows']]
            for batch in batches_by_budget(local, 2, 16384):
                if len(batch) == 2 and len({local[i] for i in batch}) == 2:
                    pairs.append([start + i for i in batch])
        if not pairs:
            raise ValueError('No unequal two-row batch in the actual production packing')
        pair = max(pairs, key=lambda batch: 2 * max(lengths[i] for i in batch))
        runtime = {k: importlib.metadata.version(k) for k in
                   ['torch', 'transformers', 'numpy', 'accelerate', 'hydra-core', 'omegaconf']}
        assert runtime == {'torch': '2.13.0', 'transformers': '5.15.0', 'numpy': '2.3.5',
                           'accelerate': '1.13.0', 'hydra-core': '1.3.2', 'omegaconf': '2.3.0'}
        assert str(torch.__version__) == '2.13.0+cu130' and torch.version.cuda == '13.0'
        report.update(arm=arm, runtime=runtime, torch_module=str(torch.__version__),
                      cuda=torch.version.cuda, sources_sha256=sources,
                      imported_sources_sha256=imports, manifest_sha256=binding.sha256(manifest),
                      prefix_tokens_sha256=binding.sha256(prefix_path),
                      selection_sha256=binding.sha256(selection_path))
        cfg = OmegaConf.create({'model': binding.MODEL})
        model, tokenizer, depth = _load_model_and_tokenizer(cfg)
        assert _logits_to_keep_kwargs(model, False) == {'logits_to_keep': 1}
        assert next(model.parameters()).dtype == torch.bfloat16 and not model.training
        selected = sorted(set([longest, shortest, *pair]))
        ids = {}
        for i in selected:
            _, ids[i] = render_prefix_ids(tokenizer, rows[i]['messages'], enable_thinking=False)
            assert ids[i] == prefixes[rows[i]['exact_context_sha256']]['token_ids']
        report['selected_rows'] = [dict(manifest_index=i, task_id=rows[i]['task_id'],
                                       condition=rows[i]['condition'], tokens=lengths[i],
                                       exact_context_sha256=rows[i]['exact_context_sha256'])
                                   for i in selected]
        report['longest_index'] = longest
        report['production_pair_indices'] = pair
        report['wrapper_depth'] = depth
        report['gpu_name'] = torch.cuda.get_device_name(0)

        def measured(name, fn):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            free, total = torch.cuda.mem_get_info()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            value = fn()
            torch.cuda.synchronize()
            metric = dict(name=name, elapsed_seconds=time.monotonic() - started,
                          allocated_before_bytes=before, free_before_bytes=free,
                          total_bytes=total, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            report['phases'].append(metric)
            print(json.dumps(metric), flush=True)
            return value

        def capture_twice(indices):
            batch = [ids[i] for i in indices]
            first, input_ids, mask = _capture_last_prefix(model, batch, [44])
            second, repeated_ids, repeated_mask = _capture_last_prefix(model, batch, [44])
            assert first.shape == (len(indices), 1, 5120)
            assert torch.isfinite(first).all() and np.isfinite(first.to(torch.float16).numpy()).all()
            assert torch.equal(first, second)
            assert torch.equal(input_ids, repeated_ids) and torch.equal(mask, repeated_mask)
            for lane, original in enumerate(batch):
                assert mask[lane].sum().item() == len(original)
                assert input_ids[lane][mask[lane].bool()].tolist() == original
            return first, input_ids, mask

        long_values, long_ids, long_mask = measured('longest_singleton_exact_replay', lambda: capture_twice([longest]))
        pair_values, pair_ids, pair_mask = measured('production_two_row_exact_replay', lambda: capture_twice(pair))
        gc.collect()
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        measured_extra = max(p['peak_allocated_bytes'] - p['allocated_before_bytes'] for p in report['phases'])
        # Conservative FP32 bound for all65 hidden states plus3 float32 comparison temporaries.
        tuple_extra_bound = (65 + 3) * lengths[longest] * 5120 * 4 + measured_extra
        report['longest_tuple_extra_bound_bytes'] = tuple_extra_bound
        report['free_before_tuple_selection_bytes'] = free
        if free >= tuple_extra_bound:
            tuple_index, tuple_ids, tuple_mask = longest, long_ids, long_mask
        else:
            report['longest_tuple_deferred_reason'] = 'Measured free memory below conservative tuple bound'
            _, tuple_ids, tuple_mask = measured('shortest_complete_prefix_exact_replay', lambda: capture_twice([shortest]))
            tuple_index = shortest
            free, _ = torch.cuda.mem_get_info()
            required = (65 + 3) * lengths[shortest] * 5120 * 4 + measured_extra
            if free < required:
                raise RuntimeError('Even the shortest complete-prefix tuple probe lacks memory headroom')
        errors = measured('same_forward_hook_vs_hidden_state_tuple', lambda: _hook_tuple_errors(model, tuple_ids, tuple_mask, [44]))
        assert errors.keys() == {44} and np.isfinite(errors[44]) and errors[44] <= 1e-5
        assert binding.source_hashes() == sources and binding.imported_source_hashes() == imports
        report.update(passed=True, tuple_probe_manifest_index=tuple_index,
                      hook_tuple_relative_errors=errors, hook_relative_tolerance=1e-5,
                      exact_replay_max_abs=0.0, capture_layers=[44])
        binding.write_json(args.output, report)
        print(json.dumps({'passed': True, 'report': str(args.output)}), flush=True)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        binding.write_json(args.output.with_suffix('.failed.json'), report)
        raise


if __name__ == '__main__':
    main()
