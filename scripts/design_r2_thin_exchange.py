"""U43：WT地体100Ma裂离后，弱背景输运完整世界对照与真实UID跟踪。"""
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
from mythic.tectonics.geometry import xyz_to_lonlat
KAPPA=0.
ledger=[]
tracks=[]
WATCH={356637:'P1东洼',356671:'P2中洼',356774:'P3西洼',
       356712:'P4西洼',356863:'P5西洼',356857:'I4岛体'}

def observed_surface(self,dt,pos=None):
    pre=self.c.T.copy()
    capture.observed_surface(self,dt,pos)
    if self.t>100.:return
    standard=self.c.T.copy()
    if KAPPA:
        T,dv,row=thin_exchange(self.c.T,self.area,self.pedges,self.alive,self.c.ctype,
            self.c.plate,self.c.origin,dt,self.m.h_km,KAPPA)
        self.c.T[:]=T
    else:
        dv=np.zeros(len(self.c));row=dict(edges=0,balance_km3=0.,
            max_local_residual_km3=0.,max_delta_T_km=0.,net_received_km3=0.,
            gross_transport_km3=0.,active_edges=0)
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

def main():
    global KAPPA
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kappa',type=float,choices=[0.,100.,500.],default=100.)
    KAPPA=ap.parse_args().kappa
    candidate=f'u43_r2_{int(KAPPA)}'
    cache=ROOT/'.cache/u43_r2'/candidate;cache.mkdir(parents=True,exist_ok=True)
    capture.MODE=candidate;capture.CACHE=ROOT/'.cache/u43_r2'
    strength.FLOOR=.05;base.KAPPA=10000.;trial.CONTACTS=True
    original_relax=base.relax_thickness
    base.relax_thickness=strength.finite_strength_relax
    base.candidate_surface=trial.surface_with_outlet
    Engine._surface=observed_surface;Engine._interact=capture.observed_interact
    Engine._arc=capture.observed_arc;capture.fwd.simulate=capture.captured_simulate
    source=json.loads((ROOT/'data/worlds/design_v2/candidates/u42_strength_05/run.json').read_text(encoding='utf8'))['input_sha256']
    names=list(source)+['scripts/design_r2_thin_exchange.py','packages/tectonics/thin_exchange_spike.py']
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
                method_class_sampling='interpolated',transport_method='u43_wt_background_exchange_v1',
                wt_kappa_km2_myr=KAPPA,wt_activation_ma=100.,
                wt_response_onset_km=20.,wt_response_scale_km=20.,
                candidate_status='experimental; default C2 unchanged',
                transport_limits='uncalibrated WT background exchange; inherited U42 and source limitations'))
            ds.setncattr('source',ds.getncattr('source')+
                '; U43 WT plate302 native continental weak background exchange after100Ma; '
                'kappa='+str(KAPPA)+'km2/Myr, cubic onset20km/scale20km; '
                'procedural_constrained hypothesis, not calibrated rheology; no final height edits.')
        success=True
    finally:
        base.relax_thickness=original_relax;base.candidate_surface=trial.BASE
        Engine._surface=capture.original_surface;Engine._interact=capture.original_interact
        Engine._arc=capture.original_arc;capture.fwd.simulate=capture.original_simulate
        sys.argv=saved
        report=dict(success=success,candidate=candidate,method='u43_wt_background_exchange_v1',
            method_class={'transport':'procedural_constrained','terrane_and_time':'user_override',
                'elevation':'parameterized_physics','sampling':'interpolated','figures':'visual_only'},
            resolution='L7 about60km; output1024x512',
            units={'thickness':'km','volume':'km3','time':'Ma','elevation':'m','diffusivity':'km2/Myr'},
            kappa_km2_myr=KAPPA,onset_km=20.,scale_km=20.,
            assumptions=['same U42 floor5percent control, motion, erosion, sources and U41 contact outlet',
                'after100Ma split, only WT plate302 alive continental native edges, at least one origin302',
                'supplemental finite exchange; D=kappa*(max(max(Ti,Tj)-20,0)/20)^3',
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
            input_sha256=hashes,extra_ledger=ledger,uid_tracks=tracks,
            standard_surface_ledger=base.ledger,outlet_ledger=trial.outlet_ledger,
            elapsed_s=time.perf_counter()-start,
            process_lifetime_peak_wset_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None))
        text=json.dumps(report,ensure_ascii=False,indent=2)
        (cache/'run.json').write_text(text,encoding='utf8',newline='\n')
        if success:(ROOT/'data/worlds/design_v2/candidates'/candidate/'run.json').write_text(text,encoding='utf8',newline='\n')
        print('U43_END',candidate,success,report['elapsed_s'],report['process_lifetime_peak_wset_bytes'],flush=True)
if __name__=='__main__':main()
