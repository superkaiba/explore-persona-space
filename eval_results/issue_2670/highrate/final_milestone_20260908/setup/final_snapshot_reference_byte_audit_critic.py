"""Independent byte/reference audit; no native parsing, fits, uploads or source mutations."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path

R=Path('/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate')
W=Path('/home/thomasjiralerspong/.codex/worktrees/context-risk-recovery-20260906')
B=R/'setup/final_snapshot_build_20260908T1249Z.json'
PIN='88a189cbb26abbe95efa808cc6e636428fd290e3c09256626ee68bea9aba8c27'
RAW_PIN='91615d1272ef62d16ccd33651acc8f295800051950a7ee45e04854a6be7a5215'
RAW_REV='58d3e6d6b0917b6e9777c45d23b814693dc14277'
RAW_PREFIX='context_risk/issue2670_highrate/raw'
REPO='superkaiba1/explore-persona-space-data'
RAW_BASE=R.parent/'issue2670_highrate_archive_readback/raw_archive_20260908T103321Z'
REGIMES={'primary','competence_sensitivity','screen_augmented_primary','screen_augmented_competence_sensitivity'}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def read(path):return json.loads(path.read_text())

def safe(base,key):
 p=Path(key)
 assert p.parts and not p.is_absolute() and '..' not in p.parts and p.as_posix()==key
 out=base/p
 assert not any(x.is_symlink() for x in [out,*out.parents])
 return out

verified={}
def check(path,expected):
 assert path.is_file() and not any(x.is_symlink() for x in [path,*path.parents]),str(path)
 before=path.stat();identity=(str(path),before.st_size,before.st_mtime_ns,before.st_ino)
 assert before.st_size==expected['size'],str(path)
 if identity not in verified:verified[identity]=sha(path)
 assert verified[identity]==expected['sha256'],str(path)
 after=path.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)

def inventory(base):
 entries=list(base.rglob('*'));assert not any(p.is_symlink() for p in [base,*entries])
 assert all(p.is_file() or p.is_dir() for p in entries)
 return {p.relative_to(base).as_posix() for p in entries if p.is_file()}

def run(output):
 build=read(B);snap=Path(build['freeze_result']['snapshot']);mfile=snap/'snapshot_manifest.json'
 assert sha(mfile)==PIN==build['freeze_result']['snapshot_manifest_sha256']
 m=read(mfile);original=m['files'];assert m['phase']=='final' and original==build['selected_files']
 assert len(original)==2175 and sum(v['size'] for v in original.values())==276670758
 assert inventory(snap)=={*original,'snapshot_manifest.json'}
 for name,value in original.items():
  check(safe(snap,name),value)
  check(Path(value['source']),value)
 print('SNAPSHOT_BYTES_PASS',len(original),flush=True)
 refs=read(safe(snap,m['raw_reference_key']))
 raw_m=snap/'references/raw_snapshot_manifest.json';assert sha(raw_m)==RAW_PIN==refs['snapshot_sha256']
 raw=read(raw_m)['files'];assert raw==refs['raw_original_files'] and len(raw)==624
 rr=read(snap/'references/raw_readback_receipt.json');ru=read(snap/'references/raw_upload_receipt.json')
 assert rr['passed'] is True and ru['passed'] is True
 assert rr['snapshot_sha256']==RAW_PIN and rr['upload_receipt_sha256']==sha(snap/'references/raw_upload_receipt.json')
 for value in (rr,ru,refs):assert (value['repo_id'],value['prefix'],value['revision'])==(REPO,RAW_PREFIX,RAW_REV)
 assert set(raw)==set(rr['original_files'])
 for k,v in raw.items():assert all(v[f]==rr['original_files'][k][f] for f in ('size','sha256'))
 for path,value in refs['raw_receipts'].items():check(Path(path),value)
 source_index={}
 for place,values in [('final',original),('raw',raw)]:
  for key,value in values.items():source_index.setdefault((value['source'],value['sha256']),[]).append((place,key,value))
 consumed=[]
 def resolve(path,expected,label):
  matches=source_index.get((path,expected),[]);assert matches,(label,path,expected)
  for place,key,value in matches:
   source=safe(snap,key) if place=='final' else safe(RAW_BASE/('reconstructed' if key.endswith('.jsonl') else RAW_PREFIX),key)
   check(source,value)
  consumed.append({'label':label,'source':path,'sha256':expected,'matches':[{ 'location':p,'snapshot_key':k,'size':v['size']} for p,k,v in matches]})
  return matches
 table=refs['consumed_input_source_references'];assert len(table)==212
 for label,value in table.items():
  collection=original if value['location']=='final' else raw
  assert value['location'] in {'final','pinned_raw'}
  row=collection[value['snapshot_key']]
  assert all(value[f]==row[f] for f in ('source','size','sha256'))
  literal=Path(value['requested_source_path'])
  absolute=literal if literal.is_absolute() else Path(value['relative_resolution_base'])/literal
  assert str(absolute)==value['source']
  if value['location']=='pinned_raw':assert (value['prefix'],value['revision'])==(RAW_PREFIX,RAW_REV)
  resolve(value['source'],value['sha256'],label)
 for value in refs['reused_setup_and_root_files']:
  for key in value['raw_snapshot_keys']:assert all(value[f]==raw[key][f] for f in ('source','size','sha256'))
  resolve(value['source'],value['sha256'],'reused_setup_or_root')
 for name,expected in build['current_input_sha256'].items():
  matches=resolve(name,expected,'actual_current_input');check(Path(name),matches[0][2])
 for key,count,review_key in [('analysis_sources_sha256',45,'run/setup/analysis_optimizer_code_review.json'),('archive_sources_sha256',5,'run/setup/archive_review_snapshots_code_review.json')]:
  closure=m[key];assert len(closure)==count and closure==build[key]
  review=read(snap/review_key);assert review['verdict']=='PASS' and review['sources_sha256']==closure
  for name,expected in closure.items():resolve(str(W/name),expected,key)
 for directory,expected_count in [('analysis',1327),('analysis_original_cap_failed_20260908T112039Z',652),('report',34)]:
  prefix=f'run/{directory}/';keys={k.removeprefix(prefix) for k in original if k.startswith(prefix)}
  assert len(keys)==expected_count and keys==inventory(R/directory)
 old=read(snap/'run/setup/analysis_original_cap_failure_20260908.json')['files']
 for key,value in old.items():
  actual=original['run/analysis_original_cap_failed_20260908T112039Z/'+key];assert all(actual[f]==value[f] for f in ('size','sha256'))
 result=read(snap/'run/analysis/result.json');rh=sha(snap/'run/analysis/result.json')
 assert result['verification_passed'] is True and set(result['analyses'])==REGIMES
 launch=read(snap/'run/analysis/analysis_launch.json');assert result['provenance']==launch['provenance']
 for name in REGIMES:assert read(snap/f'run/analysis/{name}/result.json')==result['analyses'][name]
 gate=read(snap/'references/final_completion_gate.json');assert gate['analysis_result_sha256']==rh
 for field in ('analysis_owner_launch','analysis_exit','scientific_review'):
  binding=gate[field];resolve(binding['path'],binding['sha256'],'completion_gate:'+field)
 for binding in gate['report_files']:resolve(binding['path'],binding['sha256'],'completion_gate:report')
 owner=read(Path(gate['analysis_owner_launch']['path']));ended=read(Path(gate['analysis_exit']['path']))
 assert ended['exit_code']==0 and ended['verification_passed'] is True and not ended['live_group_members'] and ended['cleanup']=='no_live_members'
 assert set(ended['realized_regimes'])==REGIMES and len(ended['realized_regimes'])==4
 assert ended['launch_sha256']==gate['analysis_owner_launch']['sha256'] and ended['launch_id']==owner['launch_id']
 resolve(owner['log_path'],ended['log_sha256'],'completed_analysis_log')
 audit=read(snap/'run/report/independent_result_audit.json');assert audit['verdict']=='PASS' and audit['analysis_result_sha256']==rh
 for field in ('analysis_files_sha256','control_files_sha256'):
  for name,expected in audit[field].items():resolve(name,expected,'final_scientific_audit:'+field)
 resolve(str(R/'report/audit_results.py'),audit['audit_source_sha256'],'final_scientific_audit_source')
 summary=read(snap/'run/report/summary_review.json');assert summary['verdict']=='PASS' and summary['analysis_result_sha256']==rh
 for name,expected in summary['inputs_sha256'].items():resolve(name,expected,'summary_review_input')
 for stem in ('forecast_model_losses','forecast_comparisons','forecast_per_task'):
  meta=read(snap/f'run/report/{stem}.meta.json');assert meta['provisional'] is False
  for field in ('source_sha256','input_sha256','output_sha256'):
   for name,expected in meta[field].items():resolve(name,expected,f'final_figure:{stem}:{field}')
 assert sha(mfile)==PIN
 findings={'schema_version':'task2670_final_snapshot_source_byte_review_v1','verdict':'PASS','reviewer':'v9_consumer_critic; independently checked actual bytes/reference bridge, not the scientific fits','reviewed_utc':datetime.now(UTC).isoformat(),'snapshot':str(snap),'snapshot_manifest_sha256':PIN,'build_receipt':{'path':str(B),'sha256':sha(B)},'driver':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},'original_files':len(original),'original_bytes':sum(v['size'] for v in original.values()),'exact_source_bytes_match_all_snapshot_files':True,'coverage':{'new_analysis_files':1327,'failed_original_analysis_files':652,'report_files':34,'analysis_sources':45,'archive_sources':5,'source_reference_table_entries':len(table),'checked_consumer_reference_uses':len(consumed),'source_reference_locations':dict(Counter(v['location'] for v in table.values()))},'raw_reference':{'repo_id':REPO,'revision':RAW_REV,'prefix':RAW_PREFIX,'snapshot_manifest_sha256':RAW_PIN,'original_files':624,'metadata_exact_matches_pinned_readback':True,'used_raw_readback_files_actually_hashed':True},'checks':['Every2175 immutable snapshot file and its actual physical source opened and byte-hashed. Exact manifest membership, safe keys, no symlinks.','Every212 declared input/source reference and all reused setup/root references checked by source path, SHA and size, retaining explicit relative-path resolution.','All current consumed input hashes, current45/5 code reviews, all1327 audited analysis-file hashes,51 scientific-control hashes, summary inputs and three final figure metadata source/input/output bindings resolved to actual final files or hash-verified pinned raw readback keys.','Original652 failure census matches every name/size/SHA; corrected1327 and report34 file sets equal their current settled source directories.','Final result/launch/four regime copies and explicit successful analysis owner/exit/log/scientific-review/report bindings checked; no metric refitting or native semantic parsing.'],'unresolved_findings':[],'pending':['Actual final pinned upload/readback receipts and reconstructed-original bytes.','Actual final archive owner exit/drain and independently pinned remote metadata equality.','Late review/owner/readback/canonical methodology Git persistence.'],'archive_completion_claimed':False,'native_reparse_or_scientific_fit_performed':False,'consumed_references':consumed}
 with output.open('x') as f:json.dump(findings,f,indent=2,sort_keys=True);f.write('\n')
 print('FINAL_SNAPSHOT_PASS',output,sha(output),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
