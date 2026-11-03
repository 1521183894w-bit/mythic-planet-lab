"""U44冻结真实U42末态：统一交换与封闭薄壳分区比较；不是额外地史。"""
from pathlib import Path
import json,time,hashlib,pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mythic.tectonics.pocket_exchange_spike import thin_partition,pocket_exchange,covered_boundary_ports
from mythic.tectonics.thin_exchange_spike import thin_exchange
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.crust import Crust,Params,elevation
from mythic.tectonics.geometry import xyz_to_lonlat
from mythic.figures.style import setup_fonts,elevation_cmap_norm
from mythic.figures.world import code_version
ROOT=Path(__file__).resolve().parents[1]
def main():
    start=time.perf_counter();path=ROOT/'.cache/u42_strength/u42_strength_05/native.npz'
    out=ROOT/'figures/design/v2/forward/u44_pockets';out.mkdir(parents=True,exist_ok=True)
    with np.load(path) as f:d={k:f[k].copy() for k in f.files}
    degree=load_mesh(7).degree[d['node']]
    with (ROOT/'.cache/u42_strength/u42_strength_05/final_state.pkl').open('rb') as f:eng=pickle.load(f)
    ports,coverage=covered_boundary_ports(eng)
    strict=thin_partition(d['T'],d['edges'],d['alive'],d['ctype'],d['plate'],d['origin'],degree)[2]
    weak,opened,part=thin_partition(d['T'],d['edges'],d['alive'],d['ctype'],d['plate'],d['origin'],degree,boundary_ports=ports)
    lo,la=xyz_to_lonlat(d['positions']);pick=d['alive']&(lo>-14)&(lo<15)&(la>32)&(la<52)
    setup_fonts();cmap,norm=elevation_cmap_norm()
    fig,axs=plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
    summary=dict(method='u44_frozen_pocket_exchange_v1',source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        method_class={'partition':'user_override','transport':'procedural_constrained','elevation':'parameterized_physics','figure':'visual_only'},
        resolution='L7 about60km native',units={'T':'km','z':'m','volume':'km3','time':'Myr'},
        assumptions=['frozen U42 geometry 20x1Myr; not extra history','same100km2/Myr onset20 scale20',
            'only enclosed thin UID stars; open ocean or missing neighbor is a port',
            'thin32.5km is inherited model boundary, not rheological calibration'],partition=part,strict_missing_port_counterexample=strict,coverage=coverage,cases={})
    for ax,mode,title in zip(axs,['control','uniform','pockets'],['原生控制','统一交换 κ100','封闭薄壳分区 κ100']):
        T=d['T'].copy();rows=[]
        for _ in range(20):
            if mode=='uniform':T,_,row=thin_exchange(T,d['area_km2'],d['edges'],d['alive'],d['ctype'],d['plate'],d['origin'],1.,float(d['h_km']),100.)
            elif mode=='pockets':T,_,row=pocket_exchange(T,d['area_km2'],d['edges'],d['alive'],d['ctype'],d['plate'],d['origin'],weak,opened,1.,float(d['h_km']))
            else:row={'balance_km3':0.,'gross_transport_km3':0.}
            rows.append(row)
        c=Crust(len(T));c.T=T;c.ctype=d['ctype'];c.age=d['age'];z=elevation(c,Params())
        summary['cases'][mode]=dict(max_balance_km3=max(abs(r['balance_km3']) for r in rows),
            gross_transport_km3=sum(r.get('gross_transport_km3',0.) for r in rows),
            max_delta_T_km=float(abs(T-d['T']).max()),samples={str(uid):float(z[np.flatnonzero(d['uid']==uid)[0]]) for uid in [356637,356671,356774,356712,356863,356857]})
        im=ax.scatter(lo[pick],la[pick],c=z[pick],cmap=cmap,norm=norm,s=18,marker='h',linewidths=0)
        ax.scatter(lo[weak&pick],la[weak&pick],s=45,facecolors='none',edgecolors='red',label='封闭薄壳')
        ax.set(xlim=(-14,15),ylim=(32,52),xlabel='经度 / °',ylabel='纬度 / °',title=title)
        ax.set_aspect(1/np.cos(np.deg2rad(42)));ax.grid(alpha=.2);ax.legend(fontsize=7)
        ax.plot([-12,-12+500/(111.2*np.cos(np.deg2rad(42)))],[33,33],c='black',lw=2);ax.text(-12,33.4,'500 km',fontsize=7)
    fig.colorbar(im,ax=axs,orientation='horizontal',fraction=.04,pad=.08,label='构造尺度高程 / m；负陆壳不是已求水位的湖')
    fig.suptitle('U44真实末态冻结小试：同系数比较结构；20步不是额外地史\n分区 user_override；输运 procedural_constrained；L7约60km；代码 '+code_version(),fontsize=10)
    fig.savefig(out/'00_frozen_spike.png',dpi=125);plt.close(fig)
    summary['elapsed_s']=time.perf_counter()-start
    (out/'spike.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
