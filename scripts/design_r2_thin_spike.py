"""U43：冻结真实原生末态，检查薄壳边有限输运的局部形态；不是额外地史。"""
from pathlib import Path
import json,sys,time,hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages'),str(ROOT/'scripts')]
from mythic.tectonics.thin_exchange_spike import thin_exchange
from mythic.tectonics.crust import Crust,Params,elevation
from mythic.tectonics.geometry import xyz_to_lonlat
from mythic.figures.style import setup_fonts,elevation_cmap_norm
from mythic.figures.world import code_version

def main():
    start=time.perf_counter()
    path=ROOT/'.cache/u42_strength/u42_strength_05/native.npz'
    out=ROOT/'figures/design/v2/forward/u43_r2';out.mkdir(parents=True,exist_ok=True)
    with np.load(path) as f:d={k:f[k].copy() for k in f.files}
    lon,lat=xyz_to_lonlat(d['positions']);pick=d['alive']&(lon>-14)&(lon<15)&(lat>32)&(lat<52)
    setup_fonts();cmap,norm=elevation_cmap_norm()
    fig,axs=plt.subplots(1,3,figsize=(17,6),constrained_layout=True)
    summary=dict(method='u43_frozen_thin_exchange_v1',method_class='procedural_constrained',
        units={'T':'km','volume':'km3','time':'Myr','elevation':'m','diffusivity':'km2/Myr'},
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        resolution='L7 about60km; native parcels',cases={},
        limits=['20x1Myr frozen geometry only; not extra geological history',
            'WT plate302 native continental edges; supplemental cubic response onset20km/scale20km',
            'no source, erosion, motion, flexure, sediment, heat or density transport',
            'diffusivity is a sensitivity hypothesis, not calibrated rheology'])
    for ax,kappa in zip(axs,[0.,100.,500.]):
        T=d['T'].copy();rows=[]
        for step in range(20):
            T,_,row=thin_exchange(T,d['area_km2'],d['edges'],d['alive'],d['ctype'],
                d['plate'],d['origin'],1.,float(d['h_km']),kappa)
            rows.append(row)
        c=Crust(len(T));c.T=T;c.ctype=d['ctype'];c.age=d['age']
        z=elevation(c,Params());pids=[356637,356671,356774,356712,356863]
        sample={str(uid):dict(T_km=float(T[np.where(d['uid']==uid)[0][0]]),
                    z_m=float(z[np.where(d['uid']==uid)[0][0]])) for uid in pids}
        summary['cases'][str(int(kappa))]=dict(samples=sample,
            max_balance_km3=max(abs(q['balance_km3']) for q in rows),
            transferred_cumulative_km3=sum(q['net_received_km3'] for q in rows),
            max_delta_T_km=float(abs(T-d['T']).max()))
        im=ax.scatter(lon[pick],lat[pick],c=z[pick],cmap=cmap,norm=norm,s=17,marker='h',linewidths=0)
        ax.set_xlim(-14,15);ax.set_ylim(32,52);ax.set_aspect(1/np.cos(np.deg2rad(42)))
        ax.set_title(f'冻结20步 · κ={kappa:g} km²/Myr')
        ax.set_xlabel('经度 / °');ax.set_ylabel('纬度 / °');ax.grid(alpha=.2)
        ax.plot([-12,-12+500/(111.2*np.cos(np.deg2rad(42)))],[33,33],c='black',lw=2)
        ax.text(-12,33.3,'500 km（42°N）',fontsize=8)
    fig.colorbar(im,ax=axs,orientation='horizontal',fraction=.045,pad=.08,label='Airy/GDH1构造尺度高程 / m')
    fig.suptitle('真实U42末态小试：WT陆壳增加弱背景交换；不移动、不增厚源、不改终态高度\nprocedural_constrained；L7约60km；原生单元；代码 '+code_version(),fontsize=10)
    fig.savefig(out/'00_frozen_spike.png',dpi=130);plt.close(fig)
    summary['elapsed_s']=time.perf_counter()-start
    (out/'spike.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
