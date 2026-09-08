"""Reviewer-owned pure CPU fixtures; never invoke diagnostic.main or load model packages."""
from __future__ import annotations
import ast,hashlib,importlib.util,json
from pathlib import Path
import numpy as np

SETUP=Path('/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate/setup')
PROJECT=Path('/home/thomasjiralerspong/.codex/worktrees/context-risk-recovery-20260906')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=SETUP/'capture_memory_check.py';packer_source=PROJECT/'scripts/context_risk_qwen38_capture.py'
assert sha(source)=='dea9ddb7e6e1f3aa0a1d1708c3a0fcb4ee874caab9e39f4d65133001051d18a1'
spec=importlib.util.spec_from_file_location('reviewed_memory_pure_helpers',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# Execute the exact inherited CPU packing body only, avoiding unrelated module imports.
parsed=ast.parse(packer_source.read_text());node=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='batches_by_budget')
namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(packer_source),'exec'),namespace)
packer=namespace['batches_by_budget'];recipe={'checkpoint_rows':15,'batch_max_rows':2,'batch_max_tokens':16384}
checks=[]
def shape_case(name,lengths,expected_longest,expected_pair):
 longest,pair=module.choose_shapes(lengths,recipe,packer)
 assert longest==expected_longest and pair==expected_pair,(name,longest,pair)
 assert pair[0]//15==pair[1]//15 and 2*max(lengths[i] for i in pair)<=16384
 actual=[]
 for start in range(0,90,15):
  actual.extend([[start+i for i in batch] for batch in packer(lengths[start:start+15],2,16384) if len(batch)==2])
 assert pair in actual and max(lengths)==lengths[longest]
 assert 2*max(lengths[i] for i in pair)==max(2*max(lengths[i] for i in b) for b in actual)
 checks.append({'name':name,'passed':True,'longest':longest,'pair':pair})
shape_case('uniform90_earliest_tie',[1000]*90,0,[0,1])
x=[1000]*90;x[14]=32768;x[16]=8192;x[17]=8191
shape_case('longest_singleton_and_exact_budget_pair',x,14,[16,17])
x=[1000]*90;x[14]=8192;x[15]=8191
shape_case('never_pair_across15_row_shard_boundary',x,14,[14,0])
x=[1000]*90;x[0]=32768;x[1]=9000;x[2]=8192;x[3]=8192
shape_case('oversize_rows_remain_singletons',x,0,[2,3])
x=[1000]*90;x[0]=x[1]=x[15]=8192;x[16]=8000
shape_case('unequal_padding_breaks_exact_padded_size_tie',x,0,[15,16])
x=[1000]*90;x[0]=x[1]=x[15]=x[16]=4000
shape_case('earliest_pair_breaks_remaining_tie',x,0,[0,1])
x=[1000]*90;x[7]=x[20]=20000
shape_case('earliest_longest_tie',x,7,[0,1])
try:module.choose_shapes([9000]*90,recipe,packer)
except ValueError as e:
 assert 'no eligible two-row batch' in str(e)
 checks.append({'name':'no_actual_pair_fails_without_fabricating_cross_shard_pair','passed':True})
else:raise AssertionError('Missing pair accepted')
ids=[[11,12,13],[21,22]]
v=np.zeros((2,1,5120),dtype=np.float16);replay=np.array([[11,12,13],[21,22,999]],dtype=np.int64);mask=np.array([[1,1,1],[1,1,0]],dtype=np.int64)
module.validate_payload(v,replay,mask,ids);checks.append({'name':'valid_padded_two_row_payload','passed':True})
module.validate_payload(v[:1],replay[:1],mask[:1],ids[:1]);checks.append({'name':'valid_singleton_payload','passed':True})
mutants=[('wrong_vector_geometry',v[:,:,1:],replay,mask),('wrong_vector_dtype',v.astype(np.float32),replay,mask),('wrong_replay_width',v,replay[:,:2],mask),('wrong_attention_width',v,replay,mask[:,:2])]
bad=v.copy();bad[0,0,0]=np.nan;mutants.append(('nonfinite_nan',bad,replay,mask))
bad=v.copy();bad[0,0,0]=np.inf;mutants.append(('nonfinite_infinity',bad,replay,mask))
bad=replay.copy();bad[0,1]=999;mutants.append(('changed_unpadded_token',v,bad,mask))
bad=mask.copy();bad[1,2]=1;mutants.append(('padding_marked_active',v,replay,bad))
bad=mask.copy();bad[1]=[0,1,1];mutants.append(('equal_mask_sum_wrong_positions',v,replay,bad))
bad=replay.copy();bad[0],bad[1]=replay[1],replay[0];mutants.append(('swapped_token_lanes',v,bad,mask))
for name,a,b,c in mutants:
 try:module.validate_payload(a,b,c,ids)
 except ValueError:checks.append({'name':name,'passed':True})
 else:raise AssertionError(f'Invalid payload accepted: {name}')
assert len(checks)==20 and all(c['passed'] for c in checks)
report={'passed':True,'case_count':len(checks),'checks':checks,'worker_sha256':sha(source),'packer_source_sha256':sha(packer_source),'fixture_source_sha256':sha(Path(__file__)),'numpy_version':np.__version__,'scope':'Pure CPU functions only; diagnostic.main not called, no torch/transformers/vllm imports, tokenizer/model/GPU calls or pod operations.'}
p=SETUP/'capture_memory_review_cpu_checks.json'
with p.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'passed':True,'cases':len(checks),'path':str(p),'sha256':sha(p)},indent=2))
