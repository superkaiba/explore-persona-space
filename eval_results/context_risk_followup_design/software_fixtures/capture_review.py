import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0, '/home/thomasjiralerspong/.codex/worktrees/context-risk-recovery-20260906')
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scripts import context_risk_followup_capture as c
from scripts import context_risk_qwen38_impossible_capture as q

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))

@pytest.fixture
def rig(tmp_path, monkeypatch):
    rows=[]; prefixes=[]
    for i in range(83):
        for condition in ['original','conflicting','oneoff']:
            messages=[{'role':'user','content':f'fixture {i} {condition}'}]
            key=c.digest(messages)
            rows.append({'task_id':f'fixture_{i}','condition':condition,'messages':messages,'exact_context_sha256':key,'public_test_role':'fresh'})
            ids=[i+1,10+len(condition)]
            prefixes.append({'exact_context_sha256':key,'token_ids':ids,'n_prefix_tokens':len(ids),'prefix_token_ids_sha256':c.digest(ids)})
    root=tmp_path/'experiment'; manifest=root/'manifests/fresh_A.jsonl'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    write(root/'manifests/freeze.json',{'manifests':{manifest.name:{'sha256':c.sha256(manifest)}}})
    write(root/'selection.json',{'passed':True,'selected_arm':'A','freeze_sha256':c.sha256(root/'manifests/freeze.json')})
    write(root/'fresh_A/run_result.json',{'passed':True,'realized_rollouts':996,'technical_errors':0,'is_pilot':False,'phase':'fresh','arm':'A','manifest_sha256':c.sha256(manifest)})
    write(root/'fresh_A/prefix_tokens.json',{'passed':True,'n_contexts':249,'contexts':prefixes})
    review=tmp_path/'fixture_review.json'
    write(review,{'verdict':'PASS','sources_sha256':c.source_hashes(),'scope':'synthetic CPU fixture only'})
    cfg=OmegaConf.create({'root':str(root),'output_dir':str(root/'captures_A'),'manifest_path':str(manifest),'arm':'A','review':str(review),'model':c.MODEL,'capture':c.CAPTURE})
    expected_versions={'transformers':'5.15.0','torch':'2.13.0','numpy':'2.3.5','accelerate':'1.13.0','hydra-core':'1.3.2','omegaconf':'2.3.0'}
    original_version=importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda key:expected_versions[key] if key in expected_versions else original_version(key))
    monkeypatch.setattr(torch,'__version__','2.13.0+cu130')
    monkeypatch.setattr(torch.version,'cuda','13.0')
    calls=[]
    def cpu_capture(cfg):
        calls.append(1)
        out=Path(cfg.output_dir)
        fingerprint=c.digest({'schema_version':'context_risk_qwen38_impossible_capture_v2','model':c.MODEL,'capture':c.CAPTURE,'manifest_sha256':c.sha256(manifest),'selected_contexts':[r['exact_context_sha256'] for r in rows]})
        chunks=[]
        for idx,start in enumerate(range(0,249,15)):
            sub=rows[start:start+15]
            stem=f'chunk_{idx:04d}'
            metadata=[]
            for row,prefix in zip(sub,prefixes[start:start+15],strict=True):
                metadata.append({**{k:row[k] for k in ['task_id','condition','exact_context_sha256','public_test_role']},**{k:prefix[k] for k in ['prefix_token_ids_sha256','n_prefix_tokens']}})
            np.savez(out/f'{stem}.npz',activation=np.full((len(sub),1,5120),idx+1,dtype=np.float16),layers=np.asarray([44],dtype=np.int16))
            (out/f'{stem}.rows.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in metadata))
            done={'schema_version':'context_risk_qwen38_impossible_capture_chunk_v2','fingerprint':fingerprint,'chunk_index':idx,'n_contexts':len(sub),'npz_sha256':c.sha256(out/f'{stem}.npz'),'rows_sha256':c.sha256(out/f'{stem}.rows.jsonl')}
            write(out/f'{stem}.done.json',done); chunks.append(done)
        report={'schema_version':'context_risk_qwen38_impossible_capture_run_v2','fingerprint':fingerprint,'model_id':c.MODEL['id'],'model_revision':c.MODEL['revision'],'n_contexts':249,'prefixes_truncated':0,'capture_layers':[44],'chunks':chunks,'passed':True}
        write(out/'run_result.json',report)
        return report
    monkeypatch.setattr(q,'run_capture',cpu_capture)
    return cfg,root,calls

def validate(root):
    return c.validate_binding(root/'captures_A',root/'manifests/fresh_A.jsonl',root/'fresh_A/prefix_tokens.json',root/'selection.json')

def test_complete_roundtrip_and_identical_resume(rig):
    cfg,root,calls=rig
    first=c.run(cfg); second=c.run(cfg)
    assert first==second and len(calls)==2
    assert validate(root)['passed']

def test_orphan_chunks_rejected_before_capture(rig):
    cfg,root,calls=rig
    write(root/'captures_A/chunk_0000.done.json',{'orphan':True})
    with pytest.raises(ValueError,match='empty output'):
        c.run(cfg)
    assert not calls

def test_launch_binding_drift_rejected_before_capture(rig):
    cfg,root,calls=rig
    c.run(cfg)
    p=root/'captures_A/capture_launch_binding.json';value=json.loads(p.read_text());value['runtime']['numpy']='wrong';write(p,value)
    with pytest.raises(ValueError,match='resume'):
        c.run(cfg)
    assert len(calls)==1

@pytest.mark.parametrize('field,value',[('model',{}),('activation_position','wrong'),('sources_sha256',{}),('imported_sources_sha256',{}),('manifest_sha256','wrong'),('prefix_tokens_sha256','wrong'),('selection_sha256','wrong'),('run_result_sha256','wrong')])
def test_binding_identity_tampering_fails(rig,field,value):
    cfg,root,_=rig;c.run(cfg)
    p=root/'captures_A/capture_binding.json';obj=json.loads(p.read_text());obj[field]=value;write(p,obj)
    with pytest.raises(ValueError):validate(root)

def resign(root):
    out=root/'captures_A';report=json.loads((out/'run_result.json').read_text())
    report['chunks']=[]
    for done_path in sorted(out.glob('chunk_*.done.json')):
        done=json.loads(done_path.read_text());stem=done_path.name.removesuffix('.done.json')
        done['npz_sha256']=c.sha256(out/f'{stem}.npz');done['rows_sha256']=c.sha256(out/f'{stem}.rows.jsonl');write(done_path,done);report['chunks'].append(done)
    write(out/'run_result.json',report)
    binding=json.loads((out/'capture_binding.json').read_text());binding['run_result_sha256']=c.sha256(out/'run_result.json');binding['chunk_files_sha256']={p.name:c.sha256(p) for p in out.glob('chunk_*')};write(out/'capture_binding.json',binding)

@pytest.mark.parametrize('kind',['nan','wrong_layer','wrong_dtype','wrong_shape','row_order','extra_chunk','wrong_prefix','wrong_model','wrong_coverage'])
def test_semantic_corruption_fails_even_with_matching_file_hashes(rig,kind):
    cfg,root,_=rig;c.run(cfg);out=root/'captures_A'
    if kind in ['nan','wrong_layer','wrong_dtype','wrong_shape']:
        p=out/'chunk_0000.npz'
        with np.load(p) as z:values=z['activation'];layers=z['layers']
        if kind=='nan':values[0,0,0]=np.nan
        if kind=='wrong_layer':layers=np.array([43],dtype=np.int16)
        if kind=='wrong_dtype':values=values.astype(np.float32)
        if kind=='wrong_shape':values=values[:,:,:5119]
        np.savez(p,activation=values,layers=layers)
    elif kind in ['row_order','wrong_prefix']:
        p=out/'chunk_0000.rows.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
        if kind=='row_order':rows[0],rows[1]=rows[1],rows[0]
        else:rows[0]['prefix_token_ids_sha256']='wrong'
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    elif kind=='extra_chunk':(out/'chunk_unexpected').write_text('extra')
    else:
        p=out/'run_result.json';obj=json.loads(p.read_text());obj['model_revision' if kind=='wrong_model' else 'n_contexts']='wrong';write(p,obj)
    resign(root)
    with pytest.raises(ValueError):validate(root)

def test_actual_imported_helper_drift_rejected(monkeypatch,tmp_path):
    import explore_persona_space.analysis.extraction as extraction
    p=tmp_path/'drift.py';p.write_text('wrong source')
    monkeypatch.setattr(extraction,'__file__',str(p))
    with pytest.raises(ValueError,match='Imported helper differs'):
        c.imported_source_hashes()
