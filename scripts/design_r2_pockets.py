"""U44：封闭薄壳分区完整世界对照；开放陆缘保持，未标定结构假设。"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import psutil
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages'),str(ROOT/'scripts')]
import design_basin_strength as strength
import design_arc_outlet_trial as trial
import design_arc_outlet as capture
import design_b6_relaxation as base
from mythic.tectonics.engine import Engine
from mythic.tectonics.thin_exchange_spike import thin_exchange
from mythic.tectonics.pocket_exchange_spike import thin_partition,pocket_exchange,covered_boundary_ports
from scipy.spatial import cKDTree
from mythic.tectonics.geometry import km_from_chord
from mythic.tectonics.geometry import xyz_to_lonlat
KAPPA=0.
MODE='pockets'
weak_uids=set()
admissions=[]
births=[]
original_commit=Engine.commit
ledger=[]
tracks=[]
WATCH={356637:'P1东洼',356671:'P2中洼',356774:'P3西洼',
       356712:'P4西洼',356863:'P5西洼',356857:'I4岛体',470099:'W海槽',66385:'E海槽'}

def observed_surface(self,dt,pos=None):
    pre=self.c.T.copy()
    capture.observed_surface(self,dt,pos)
    if self.t>100.:return
    standard=self.c.T.copy()
    ports,cov=covered_boundary_ports(self)
    closed,opened,part=thin_partition(self.c.T,self.pedges,self.alive,self.c.ctype,
        self.c.plate,self.c.origin,self.m.degree[self.c.node],boundary_ports=ports)
    newly=set(self.c.uid[closed].tolist())-weak_uids
    if MODE=='pockets' and newly:
        admissions.append(dict(t_ma=float(self.t),uids=sorted(newly),coverage=cov,partition=part))
        weak_uids.update(newly)
    weak=np.isin(self.c.uid,list(weak_uids))
    if MODE=='uniform':
        T,dv,row=thin_exchange(self.c.T,self.area,self.pedges,self.alive,self.c.ctype,
            self.c.plate,self.c.origin,dt,self.m.h_km,KAPPA)
        self.c.T[:]=T
    elif MODE=='pockets':
        T,dv,row=pocket_exchange(self.c.T,self.area,self.pedges,self.alive,self.c.ctype,
            self.c.plate,self.c.origin,weak,opened,dt,self.m.h_km,KAPPA)
        self.c.T[:]=T
    else:
        dv=np.zeros(len(self.c));row=dict(edges=0,balance_km3=0.,
            max_local_residual_km3=0.,max_delta_T_km=0.,net_received_km3=0.,
            gross_transport_km3=0.,active_edges=0)
    row.update(partition=part,coverage=cov,persistent_weak_cells=int(weak.sum()))
    row['t_ma']=float(self.t);ledger.append(row)
    pos=self.positions() if pos is None else pos
    lon,lat=xyz_to_lonlat(pos)
    for uid,name in WATCH.items():
        found=np.flatnonzero((self.c.uid==uid)&self.alive)
        if not len(found):continue
        i=int(found[0])
        tracks.append(dict(uid=uid,name=name,t_ma=float(self.t),lon=float(lon[i]),lat=float(lat[i]),
            plate=int(self.c.plate[i]),origin=int(self.c.origin[i]),ctype=int(self.c.ctype[i]),
            T_before_surface_km=float(pre[i]),T_after_standard_km=float(standard[i]),
            T_after_extra_km=float(self.c.T[i]),extra_net_volume_km3=float(dv[i]),
            age_myr=float(self.c.age[i]),oro=int(self.c.oro[i]),
            oro_ma=float(self.c.oro_t[i]) if np.isfinite(self.c.oro_t[i]) else None))

def observed_commit(self):
    present=set(self.c.uid.tolist())
    before=self.c.gather(np.flatnonzero(self.alive))
    pos=self.positions()[self.alive].copy()
    result=original_commit(self)
    for uid,name in WATCH.items():
        if uid in present:continue
        found=np.flatnonzero((self.c.uid==uid)&self.alive)
        if not len(found):continue
        i=int(found[0]);point=self.positions()[i]
        dist,near=cKDTree(pos).query(point,k=min(12,len(pos)))
        a=int(near[0]);other=near[before.plate[near]!=before.plate[a]]
        b=int(other[0]) if len(other) else a
        pa,pb=int(before.plate[a]),int(before.plate[b])
        opening=float(np.dot(np.cross(self.omega[pa]-self.omega[pb],point),pos[a]-pos[b]))
        lo,la=xyz_to_lonlat(point)
        births.append(dict(uid=uid,name=name,t_ma=float(self.t),lon=float(lo),lat=float(la),
            plate=int(self.c.plate[i]),ctype=int(self.c.ctype[i]),origin=int(self.c.origin[i]),
            T_km=float(self.c.T[i]),age_myr=float(self.c.age[i]),formation_ma=float(self.c.t_form[i]),
            nearest_pair=[pa,pb],nearest_ctype=[int(before.ctype[a]),int(before.ctype[b])],
            nearest_uid=[int(before.uid[a]),int(before.uid[b])],
            neighbor_distances_km=km_from_chord(dist).tolist(),opening_dot_rad_myr=opening,
            pair_gap_prescription=self.cont_gap_pairs.get(min(pa,pb)*1000+max(pa,pb)),
            evidence='UID absent before and present after actual Engine.commit; nearest12 recomputed at assigned parcel center, not original gap integration point'))
    return result

def main():
    global KAPPA,MODE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['control','uniform','pockets'],default='pockets')
    MODE=ap.parse_args().mode
    KAPPA=0. if MODE=='control' else 100.
    candidate='u44_partition_'+MODE
    cache=ROOT/'.cache/u44_pockets'/candidate;cache.mkdir(parents=True,exist_ok=True)
    capture.MODE=candidate;capture.CACHE=ROOT/'.cache/u44_pockets'
    strength.FLOOR=.05;base.KAPPA=10000.;trial.CONTACTS=True
    original_relax=base.relax_thickness
    base.relax_thickness=strength.finite_strength_relax
    base.candidate_surface=trial.surface_with_outlet
    Engine.commit=observed_commit
    Engine._surface=observed_surface;Engine._interact=capture.observed_interact
    Engine._arc=capture.observed_arc;capture.fwd.simulate=capture.captured_simulate
    source=json.loads((ROOT/'data/worlds/design_v2/candidates/u42_strength_05/run.json').read_text(encoding='utf8'))['input_sha256']
    names=list(source)+['scripts/design_r2_pockets.py','packages/tectonics/thin_exchange_spike.py','packages/tectonics/pocket_exchange_spike.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names}
    saved=sys.argv[:];start=time.perf_counter();success=False
    sys.argv=['design_forward.py','--history-config','config/design/history_v2_u35_c10_progressive.yaml',
        '--candidate',candidate,'--no-figures']
    try:
        capture.fwd.main()
        nc=ROOT/f'data/worlds/design_v2/candidates/{candidate}/design_v2_forward_v2_0Ma_{candidate}_1024x512.nc'
        with Dataset(nc,'a') as ds:
            ds.setncatts(dict(method_class_elevation='parameterized_physics',
                method_class_transport='procedural_constrained',method_class_region='user_override',
                method_class_sampling='interpolated',transport_method='u44_enclosed_pocket_exchange_v1',
                wt_kappa_km2_myr=KAPPA,wt_activation_ma=100.,
                wt_response_onset_km=20.,wt_response_scale_km=20.,partition_mode=MODE,
                partition_thin_threshold_km=32.5,partition_class='user_override',
                candidate_status='experimental; default C2 unchanged',
                transport_limits='uncalibrated WT background exchange; inherited U42 and source limitations'))
            ds.setncattr('source',ds.getncattr('source')+
                '; U44 WT enclosed inherited thin patches after100Ma; missing native sites checked against foreign CONT coverage; '
                'kappa='+str(KAPPA)+'km2/Myr, cubic onset20km/scale20km; '
                'procedural_constrained hypothesis, not calibrated rheology; no final height edits.')
        success=True
    finally:
        base.relax_thickness=original_relax;base.candidate_surface=trial.BASE
        Engine.commit=original_commit
        Engine._surface=capture.original_surface;Engine._interact=capture.original_interact
        Engine._arc=capture.original_arc;capture.fwd.simulate=capture.original_simulate
        sys.argv=saved
        report=dict(success=success,candidate=candidate,mode=MODE,method='u44_enclosed_pocket_exchange_v1',
            method_class={'transport':'procedural_constrained','terrane_and_time':'user_override',
                'elevation':'parameterized_physics','sampling':'interpolated','figures':'visual_only'},
            resolution='L7 about60km; output1024x512',
            units={'thickness':'km','volume':'km3','time':'Ma','elevation':'m','diffusivity':'km2/Myr'},
            kappa_km2_myr=KAPPA,onset_km=20.,scale_km=20.,partition_thin_threshold_km=32.5,
            assumptions=['same U42 floor5percent control, motion, erosion, sources and U41 contact outlet',
                'after100Ma split, only WT plate302 alive continental native edges, at least one origin302',
                'supplemental finite exchange; D=kappa*(max(max(Ti,Tj)-20,0)/20)^3',
                'same32.5km thin classification; component touching ocean or uncovered missing native neighbor is protected',
                'missing native neighbor covered by foreign alive CONT is a closed wall via Engine.covering',
                'admitted closed origin302 UID retained above classification threshold; only weak-incident edges exchange',
                'open thin endpoints never exchange in supplemental operator; no hand selected coordinates/UIDs',
                'no32.5km edge cutoff in this supplemental operator; standard operators unchanged',
                'no ocean reception, cross-plate bridge, noise, height clipping or final height edits',
                'watch existing final-control UIDs; absence before creation is not inferred lineage'],
            limits=['not calibrated rheology, no stress/heat/flexure/sedimentation/density or mixture solver',
                '20km is a hypothesis response onset, not a measured channel thickness',
                'native parcels approximate continuum; no exact dual-face physical flux',
                'cumulative paired gross transfer may count the same material repeatedly',
                'operator volume is not full geological mass/provenance conservation',
                'r2 broad cape and oldr8 causal issues remain open; no stage acceptance'],
            references=['https://academic.oup.com/gji/article/162/2/575/2050588',
                'https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html'],
            input_sha256=hashes,extra_ledger=ledger,uid_tracks=tracks,admissions=admissions,watched_births=births,
            standard_surface_ledger=base.ledger,outlet_ledger=trial.outlet_ledger,
            elapsed_s=time.perf_counter()-start,
            process_lifetime_peak_wset_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None))
        text=json.dumps(report,ensure_ascii=False,indent=2)
        (cache/'run.json').write_text(text,encoding='utf8',newline='\n')
        if success:(ROOT/'data/worlds/design_v2/candidates'/candidate/'run.json').write_text(text,encoding='utf8',newline='\n')
        print('U44_END',candidate,success,report['elapsed_s'],report['process_lifetime_peak_wset_bytes'],flush=True)
if __name__=='__main__':main()
