"""U43冻结送审图与真实输入散列；不改科学数据或指标。"""
from pathlib import Path
import json,hashlib,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'figures/design/v2/forward/u43_r2'
    standard=ROOT/'figures/stage1/u43_r2_100'
    copied=ROOT/'.cache/u43_review_pack';copied.mkdir(parents=True,exist_ok=True)
    topics=sorted(out.glob('*.png'))
    figures=topics+sorted(standard.glob('*.png'))+sorted((standard/'regions').glob('*.png'))
    assert len(topics)==9 and len(figures)==38,(len(topics),len(figures))
    records=[]
    for p in figures:
        rel=p.relative_to(ROOT).as_posix()
        target=copied/rel.replace('/','__')
        shutil.copyfile(p,target)
        assert sha(p)==sha(target)
        records.append(dict(path=rel,sha256=sha(p),frozen_copy=target.relative_to(ROOT).as_posix()))
    for name in ['metrics.json','regions.json']:shutil.copyfile(standard/name,copied/name)
    shutil.copyfile(out/'comparison.json',copied/'comparison.json')
    manifest=dict(description='U43独立评审图；9专题+29标准，公开地球参照另见标准画廊',
        images=records,metrics='figures/stage1/u43_r2_100/metrics.json',
        regions='figures/stage1/u43_r2_100/regions.json',
        comparison='figures/design/v2/forward/u43_r2/comparison.json')
    (out/'review_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8',newline='\n')
    bad={}
    for k in ['0','100','500']:
        p=ROOT/f'data/worlds/design_v2/candidates/u43_r2_{k}/run.json'
        report=json.loads(p.read_text(encoding='utf8'))
        bad[k]=[name for name,digest in report['input_sha256'].items() if sha(ROOT/name)!=digest]
    assert not any(bad.values()),bad
    audit=dict(method='u43_review_pack_v1',images=len(records),frozen_sha_all_equal=True,
        runtime_input_hash_mismatches=bad,
        tests=json.loads((ROOT/'data/worlds/design_v2/candidates/u43_r2_100/tests.json').read_text(encoding='utf8')),
        method_class={'transport':'procedural_constrained','region_time':'user_override',
            'elevation':'parameterized_physics','sampling':'interpolated','figures':'visual_only'})
    p=ROOT/'data/worlds/design_v2/candidates/u43_r2_100/audit.json'
    p.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8',newline='\n')
    metrics=json.loads((standard/'metrics.json').read_text(encoding='utf8'))
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    print('METRIC_KEYS',list(metrics))
    for key,value in metrics.items():
        if isinstance(value,dict):print(key,list(value))
if __name__=='__main__':main()
