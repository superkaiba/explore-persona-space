"""Paid annotation must survive malformed/refused responses without fake scores."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("behavior", Path(__file__).parents[1]/"scripts/issue2564_answer_behavior.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def envelope(content, finish="stop", refusal=None):
    return {"choices":[{"message":{"content":content,"refusal":refusal},"finish_reason":finish}]}


def test_invalid_shapes_never_raise_or_become_zero():
    for obj in [None, {}, {"choices":[]}, {"choices":[None]}, {"choices":[{"message":None}]}]:
        assert m.parse_result("warmth",obj)[1] == "invalid_envelope"
    for text in ["null","false","3","[]",'[{}]', '{"score":0}',"not json"]:
        parsed,drop=m.parse_result("warmth",envelope(text))
        assert parsed is None and drop is not None


def test_unassessable_is_valid_but_not_numeric():
    x={"reason":"No expressed interpersonal content","assessable":False,"score":None}
    assert m.parse_result("warmth",envelope(json.dumps(x)))==(x,None)
    x["score"]=0
    assert m.parse_result("warmth",envelope(json.dumps(x)))[1]=="invalid_score"


def test_refusal_truncation_and_invalid_scores_are_distinct():
    assert m.parse_result("warmth",envelope(None,refusal="refused"))[1]=="api_refusal"
    assert m.parse_result("warmth",envelope("{",finish="length"))[1]=="truncation"
    for score in [-1,101,True,0.5]:
        x={"reason":"x","assessable":True,"score":score}
        assert m.parse_result("warmth",envelope(json.dumps(x)))[1]=="invalid_score"


def test_rubric_schema_roundtrip():
    for prop,rule in m.PROPERTIES.items():
        x={"reason":"Observed in the supplied answer"}
        if rule["kind"]=="graded":x.update(assessable=True,score=50)
        else:x["label"]=rule["labels"][0]
        if prop=="persona":x["multi_voice"]=False
        assert m.parse_result(prop,envelope(json.dumps(x)))==(x,None)
        assert set(x)==set(m.schema(prop)["required"])


def test_atomic_persistence(tmp_path):
    p=tmp_path/"data.json"
    m.dump(p,{"raw":{"x":1}})
    assert json.loads(p.read_text())=={"raw":{"x":1}}
    assert list(tmp_path.glob("*.tmp.*"))==[]


def test_cache_identity_covers_model_config_answer_and_rubric():
    config={"model":"test-fixture","temperature":1}
    row={"id":"a","answer":"hello"}
    key=m.annotation_key(config,row,'warmth',0)
    assert key != m.annotation_key({**config,'temperature':.5},row,'warmth',0)
    assert key != m.annotation_key(config,{**row,'answer':'goodbye'},'warmth',0)
    assert key != m.annotation_key(config,row,'confidence',0)
    assert key != m.annotation_key(config,row,'warmth',1)


def test_aggregate_keeps_unassessability_distinct_and_ties_missing(tmp_path):
    prepared=tmp_path/"prepared"
    prepared.mkdir()
    row={"id":"a","part":"pilot","answer":"fixture"}
    (prepared/"rows.jsonl").write_text(json.dumps(row)+'\n')
    raw=tmp_path/"annotation/pilot/raw"
    raw.mkdir(parents=True)
    config={"model":"test-fixture"}
    m.dump(raw.parent/'config.json',config)
    for prop,rule in m.PROPERTIES.items():
        for d in range(5):
            x={"reason":"test fixture"}
            drop=None
            if rule["kind"]=="graded":x.update(assessable=False,score=None)
            else:x["label"]=rule["labels"][d%2]
            if prop=="persona":x["multi_voice"]=False
            if rule["kind"]=="categorical" and d==4:drop="api_refusal"
            m.dump(raw/f'{prop}_{d}.json',{"key":m.annotation_key(config,row,prop,d),"config_hash":m.digest(config),"row_id":"a","property":prop,"draw":d,"raw":envelope(json.dumps(x)),"parsed":x if drop is None else None,"drop":drop,"cost_dollars":0,"transport_errors":[]})
    m.aggregate(tmp_path,"pilot")
    label=json.loads((raw.parent/'labels.json').read_text())[0]['properties']
    assert label['warmth']['n_valid']==5 and label['warmth']['n_assessable']==0
    assert label['warmth']['mean'] is None
    assert label['persona']['n_valid']==4 and label['persona']['modal'] is None
    assert sum(label['persona']['votes'].values())==1
    quality=json.loads((raw.parent/'quality.json').read_text())['properties']
    assert quality['warmth']['transport_parse_gate_pass'] is True
    assert quality['warmth']['assessable_draw_fraction']==0
    assert quality['persona']['outcomes']['api_refusal']==1
    manifest=json.loads((raw.parent/'labels_manifest.json').read_text())
    assert manifest['labels_sha256']==m.hashlib.sha256((raw.parent/'labels.json').read_bytes()).hexdigest()
    assert manifest['expected_draws']==manifest['persisted_draws']==35
    assert manifest['exact_keyset_complete'] is True
    assert manifest['complete_sha256'] is None  # diagnostics may precede completion


def test_aggregate_malformed_envelopes_and_all_missing_persona(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    row={'id':'a','part':'pilot','answer':'fixture'}
    (prepared/'rows.jsonl').write_text(json.dumps(row)+'\n')
    raw=tmp_path/'annotation/pilot/raw';raw.mkdir(parents=True)
    config={'model':'test-fixture'}
    m.dump(raw.parent/'config.json',config)
    for d,invalid in enumerate([None,{}, {'choices':[None]},{'choices':[{}]},{'choices':[]}]):
        m.dump(raw/f'persona_{d}.json',{'key':m.annotation_key(config,row,'persona',d),'config_hash':m.digest(config),'row_id':'a','property':'persona','draw':d,'raw':invalid,'parsed':None,'drop':'invalid_envelope','cost_dollars':0,'transport_errors':[]})
    m.aggregate(tmp_path,'pilot')
    quality=json.loads((raw.parent/'quality.json').read_text())['properties']['persona']
    assert quality['finish_reasons']=={'absent_or_invalid_envelope':5}
    assert quality['multi_voice_majority_fraction'] is None
    assert quality['draw_pair_agreement'] is None


def pilot_gate_fixture(root):
    """Isolated artifact fixture for gate integrity, never a scientific result."""
    out=root/'annotation/pilot';out.mkdir(parents=True)
    prepared=root/'prepared';prepared.mkdir()
    config={'model':'test-fixture','temperature':1.0,'draws':5,'max_tokens':1024,'concurrency':24}
    rows=[{'id':str(i),'answer':f'fixture {i}','part':'pilot'} for i in range(32)]
    (prepared/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (prepared/'vectors.npz').write_bytes(b'gate-only fixture, never loaded as vectors')
    m.dump(prepared/'rubrics.json',m.PROPERTIES)
    m.dump(out/'config.json',config)
    keys={m.annotation_key(config,r,p,d) for r in rows for p in m.PROPERTIES for d in range(5)}
    for row in rows:
        for prop in m.PROPERTIES:
            for draw in range(5):
                key=m.annotation_key(config,row,prop,draw)
                value={'reason':'unit-test instrument fixture'}
                rule=m.PROPERTIES[prop]
                if rule['kind']=='graded':value.update(assessable=True,score=50)
                else:value['label']=rule['labels'][0]
                if prop=='persona':value['multi_voice']=False
                m.dump(out/'raw'/f'{key}.json',{'key':key,'config_hash':m.digest(config),'row_id':row['id'],'property':prop,'draw':draw,'raw':envelope(json.dumps(value)),'parsed':value,'drop':None,'cost_dollars':0,'transport_errors':[]})
    m.dump(out/'complete.json',{'expected_keys_hash':m.digest(sorted(keys)),'persisted_expected':len(keys),'config_hash':m.digest(config)})
    m.aggregate(root,'pilot')
    review={'verdict':'accept','reviewer':'unit-test fixture','properties':{p:{'verdict':'qualified','semantic_notes':'fixture-only review','reliability_notes':'fixture-only review','coverage_notes':'fixture-only review'} for p in m.PROPERTIES}}
    m.dump(root/'review_input.json',review)
    return config,out


def test_acceptance_requires_actual_record_and_complete_pilot(tmp_path):
    with pytest.raises(RuntimeError,match='requires'):
        m.validate_pilot_acceptance(tmp_path)
    _,out=pilot_gate_fixture(tmp_path)
    complete=json.loads((out/'complete.json').read_text());complete['persisted_expected']-=1
    m.dump(out/'complete.json',complete)
    with pytest.raises(ValueError,match='exact frozen'):
        m.accept_pilot(tmp_path,tmp_path/'review_input.json')
    assert not (out/'accepted.json').exists()


def test_acceptance_pins_evidence_review_and_main_recipe(tmp_path):
    config,out=pilot_gate_fixture(tmp_path)
    m.accept_pilot(tmp_path,tmp_path/'review_input.json')
    assert m.validate_pilot_acceptance(tmp_path,config)['verdict']=='accept'
    assert m.validate_pilot_acceptance(tmp_path,{**config,'concurrency':12})['verdict']=='accept'
    with pytest.raises(ValueError,match='recipe differs'):
        m.validate_pilot_acceptance(tmp_path,{**config,'temperature':.5})
    with pytest.raises(FileExistsError):
        m.accept_pilot(tmp_path,tmp_path/'review_input.json')
    labels=json.loads((out/'labels.json').read_text());labels[0]['modified']=True
    m.dump(out/'labels.json',labels)
    with pytest.raises(ValueError,match='stale'):
        m.validate_pilot_acceptance(tmp_path)


def test_api_success_without_substantive_review_cannot_accept(tmp_path):
    _,out=pilot_gate_fixture(tmp_path)
    review=json.loads((tmp_path/'review_input.json').read_text())
    review['properties']['topic'].pop('semantic_notes')
    m.dump(tmp_path/'review_input.json',review)
    with pytest.raises(ValueError,match='Incomplete semantic'):
        m.accept_pilot(tmp_path,tmp_path/'review_input.json')
    assert not (out/'accepted.json').exists()


def test_stale_aggregate_or_deleted_raw_prevents_acceptance(tmp_path):
    _,out=pilot_gate_fixture(tmp_path)
    raw=next((out/'raw').glob('*.json'))
    record=json.loads(raw.read_text());record['changed_after_aggregation']=True
    m.dump(raw,record)
    with pytest.raises(ValueError,match='aggregates are stale'):
        m.accept_pilot(tmp_path,tmp_path/'review_input.json')
    raw.unlink()
    with pytest.raises(ValueError,match='raw records do not cover'):
        m.accept_pilot(tmp_path,tmp_path/'review_input.json')
    assert not (out/'accepted.json').exists()
