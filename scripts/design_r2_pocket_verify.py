from pathlib import Path
import json,hashlib,pickle
import numpy as np
r=Path(__file__).resolve().parents[1]
report=dict(method='u44_runtime_verification_v1',reproduction={},runtime_hash_mismatches={})
def same(a,b):return a.shape==b.shape and np.array_equal(a,b,equal_nan=a.dtype.kind in 'fc')
for old,new in [('u43_r2_0','u44_partition_control'),('u43_r2_100','u44_partition_uniform')]:
    row={}
    for kind,pa,pb in [
        ('saved',r/f'.cache/tectonics/design/candidates/{old}/forward_v2_v2_L7.npz',r/f'.cache/tectonics/design/candidates/{new}/forward_v2_v2_L7.npz'),
        ('native',r/f'.cache/u43_r2/{old}/native.npz',r/f'.cache/u44_pockets/{new}/native.npz')]:
        with np.load(pa) as a,np.load(pb) as b:
            row[kind+'_arrays']=len(a.files)
            row[kind+'_unequal']=[k for k in a.files if not same(a[k],b[k])]
    a=json.loads((r/f'data/worlds/design_v2/candidates/{old}/forward_v2.json').read_text(encoding='utf8'))
    b=json.loads((r/f'data/worlds/design_v2/candidates/{new}/forward_v2.json').read_text(encoding='utf8'))
    row['budget_identical']=a['budget_km3']==b['budget_km3'];row['event_counts_identical']=a['engine_event_counts']==b['engine_event_counts']
    report['reproduction'][new]=row
for name in ['control','uniform','pockets']:
    c='u44_partition_'+name
    d=json.loads((r/f'data/worlds/design_v2/candidates/{c}/run.json').read_text(encoding='utf8'))
    report['runtime_hash_mismatches'][c]=[p for p,h in d['input_sha256'].items() if hashlib.sha256((r/p).read_bytes()).hexdigest()!=h]
    print(c,'admissions',d['admissions'],'births',d['watched_births'])
    print('WATCH0',[(q['name'],q['T_after_extra_km']) for q in d['uid_tracks'] if q['t_ma']==0])
with (r/'.cache/tectonics/design/initial_v2_v2_L7.pkl').open('rb') as f:world=pickle.load(f)
c=world.crust
ids=np.flatnonzero(c.uid==66385)
report['E_initial']=dict(t_ma=float(world.t),matching_indices=ids.tolist(),
    rows=[{k:(float(getattr(c,k)[i]) if k in ['T','age','t_form'] else int(getattr(c,k)[i])) for k in ['uid','plate','node','ctype','T','age','t_form','origin']} for i in ids])
assert not any(report['runtime_hash_mismatches'].values())
for row in report['reproduction'].values():
    assert not row['saved_unequal'] and not row['native_unequal'] and row['budget_identical'] and row['event_counts_identical']
(r/'data/worlds/design_v2/candidates/u44_partition_pockets/runtime_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,ensure_ascii=False,indent=2))
