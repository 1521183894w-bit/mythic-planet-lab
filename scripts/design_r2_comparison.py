"""U43：固定r2湾—岬—海槽，真实演化、原生UID与全球代价图。"""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,ListedColormap
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages'),str(ROOT/'scripts')]
from mythic.figures.style import setup_fonts,elevation_cmap_norm
from mythic.figures.world import code_version,load,load_earth
from mythic.figures.netcdf import open_netcdf
from mythic.metrics.grid import LatLonGrid
from mythic.metrics.topology import label,component_sums
from mythic.tectonics.mesh import Raster,load_mesh
from mythic.tectonics.geometry import xyz_to_lonlat,lonlat_to_xyz
from design_basin_strength_comparison import localize_earth_gallery
FIELDS=['elev','T','age','t_form','oro_t','plate','ctype','origin','oro','prov','dyn']
POINTS={'P1':(8.0859375,44.82421875),'P2':(3.92578125,44.00390625),
        'P3':(-7.3828125,40.25390625),'P4':(-9.31640625,39.7265625),
        'P5':(-11.07421875,39.0234375),'I4':(5.,38.),
        'W':(0.,40.),'E':(8.,39.)}
def equal(a,b):
    return np.array_equal(a,b,equal_nan=a.dtype.kind in 'fc')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--review',choices=['100','500'],default='100')
    sel=ap.parse_args().review
    out=ROOT/'figures/design/v2/forward/u43_r2';out.mkdir(parents=True,exist_ok=True)
    cases={k:f'u43_r2_{k}' for k in ['0','100','500']}
    labels={'0':'同源控制','100':'WT弱交换 κ100','500':'WT弱交换 κ500'}
    folder=ROOT/'data/worlds/design_v2/candidates'
    paths={k:folder/c/f'design_v2_forward_v2_0Ma_{c}_1024x512.nc' for k,c in cases.items()}
    z={}
    for k,p in paths.items():
        with open_netcdf(p) as ds:z[k]=np.array(ds.elevation.values,float)
    g=LatLonGrid.from_shape(z['0'].shape);area=g.cell_area_km2
    runs={k:json.loads((folder/c/'run.json').read_text(encoding='utf8')) for k,c in cases.items()}
    reports={k:json.loads((folder/c/'forward_v2.json').read_text(encoding='utf8')) for k,c in cases.items()}
    cache=ROOT/'.cache/tectonics/design/candidates'
    summ=dict(method='u43_r2_comparison_v1',review_candidate=cases[sel],
        method_class={'transport':'procedural_constrained','terrane_time':'user_override',
            'elevation':'parameterized_physics','sampling':'interpolated','figures':'visual_only'},
        resolution='L7 about60km; output1024x512',
        units={'elevation':'m','area':'km2','volume':'km3','time':'Ma','thickness':'km'},
        fixed_window={'lon':[-14,15],'lat':[32,52],'standard_r2_center':[42.027,1.055]},
        input_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},cases={})
    with np.load(cache/'u42_strength_05/forward_v2_v2_L7.npz') as old,np.load(cache/f"{cases['0']}/forward_v2_v2_L7.npz") as new:
        summ['control_reproduction']=dict(saved_arrays=len(old.files),
            unequal_saved_arrays=[n for n in old.files if not equal(old[n],new[n])])
    oldrep=json.loads((folder/'u42_strength_05/forward_v2.json').read_text(encoding='utf8'))
    summ['control_reproduction']['budget_identical']=oldrep['budget_km3']==reports['0']['budget_km3']
    summ['control_reproduction']['event_counts_identical']=oldrep['engine_event_counts']==reports['0']['engine_event_counts']
    with np.load(ROOT/'.cache/u42_strength/u42_strength_05/native.npz') as old,np.load(ROOT/'.cache/u43_r2/u43_r2_0/native.npz') as new:
        summ['control_reproduction']['native_arrays']=len(old.files)
        summ['control_reproduction']['unequal_native_arrays']=[n for n in old.files if not equal(old[n],new[n])]
    native={}
    for k in cases:
        r=runs[k];assert r['success']
        with np.load(ROOT/f'.cache/u43_r2/{cases[k]}/native.npz') as f:native[k]={n:f[n].copy() for n in f.files}
        diff=z[k]-z['0'];step=r['extra_ledger']
        row=dict(max_elevation_m=float(z[k].max()),cells_over10km=int((z[k]>10000).sum()),
            changed_area_over100m_km2=float(area[abs(diff)>100].sum()),max_abs_change_m=float(abs(diff).max()),
            land_to_ocean_km2=float(area[(z['0']>0)&(z[k]<=0)].sum()),
            ocean_to_land_km2=float(area[(z['0']<=0)&(z[k]>0)].sum()),
            budget_delta_km3={key:v-reports['0']['budget_km3'].get(key,0) for key,v in reports[k]['budget_km3'].items()},
            extra_steps=len(step),active_steps=sum(q.get('active_edges',0)>0 for q in step),
            extra_max_balance_km3=max(abs(q['balance_km3']) for q in step),
            extra_max_local_residual_km3=max(q['max_local_residual_km3'] for q in step),
            cumulative_gross_transfer_km3=sum(q.get('gross_transport_km3',0) for q in step),
            elapsed_s=r['elapsed_s'],process_lifetime_peak_wset_bytes=r['process_lifetime_peak_wset_bytes'],
            fixed_probes={})
        with np.load(cache/f"{cases['0']}/forward_v2_v2_L7.npz") as control,np.load(cache/f'{cases[k]}/forward_v2_v2_L7.npz') as fr:
            ids=np.flatnonzero(control['times']>=100)
            row['pre_activation_arrays']=len(ids)*len(FIELDS)
            row['pre_activation_unequal']=[f'{f}_{i}' for i in ids for f in FIELDS if not equal(control[f'{f}_{i}'],fr[f'{f}_{i}'])]
        d=native[k];ai=np.flatnonzero(d['alive']);tree=cKDTree(d['positions'][ai])
        for name,(lon,lat) in POINTS.items():
            y=int(np.argmin(abs(g.lat-lat)));x=int(np.argmin(abs(g.lon-lon)))
            _,ix=tree.query(lonlat_to_xyz(lon,lat));i=ai[ix]
            row['fixed_probes'][name]=dict(lon=lon,lat=lat,z_m=float(z[k][y,x]),
                uid=int(d['uid'][i]),T_km=float(d['T'][i]),plate=int(d['plate'][i]),
                origin=int(d['origin'][i]),ctype=int(d['ctype'][i]),age_myr=float(d['age'][i]),
                formation_ma=float(d['t_form'][i]) if np.isfinite(d['t_form'][i]) else None,
                last_event=int(d['oro'][i]),last_event_ma=float(d['oro_t'][i]) if np.isfinite(d['oro_t'][i]) else None)
        # 只作局部诊断；使用原先固定坐标，不能据此宣称湖泊或排水形成。
        row['five_original_pond_probe_cells_below0']=sum(row['fixed_probes'][f'P{i}']['z_m']<=0 for i in range(1,6))
        with open_netcdf(paths[k]) as ds:land=np.array(ds.land.values,bool)
        lab,n=label(land);sizes=component_sums(lab,n,area)
        iy=int(np.argmin(abs(g.lat-38.)));ix=int(np.argmin(abs(g.lon-5.)))
        component=int(lab[iy,ix]);biggest=int(np.argmax(sizes[1:]))+1
        row['I4_probe_connected_to_main']=component==biggest
        row['I4_probe_component_area_km2']=float(sizes[component])

        summ['cases'][k]=row
    setup_fonts();cmap,norm=elevation_cmap_norm();extent=(-180,180,-90,90)
    stamp='设计v2；WT范围/100Ma user_override；输运 procedural_constrained；高程 parameterized_physics；L7约60km；输出1024×512'
    def local(ax):
        ax.set_xlim(-14,15);ax.set_ylim(32,52);ax.set_aspect(1/np.cos(np.deg2rad(42)))
        ax.set_xlabel('经度 / °');ax.set_ylabel('纬度 / °');ax.grid(alpha=.2)
        ax.plot([-12,-12+500/(111.2*np.cos(np.deg2rad(42)))],[33,33],c='black',lw=2)
        ax.text(-12,33.4,'500 km（42°N）',fontsize=7)
    def save(fig,name,title):
        fig.suptitle(title+'\n'+stamp,fontsize=10)
        fig.get_layout_engine().set(rect=(0,.04,1,.96))
        fig.text(.5,.003,'经纬裁剪；比例尺在42°N；代码 '+code_version(),ha='center',fontsize=8)
        fig.savefig(out/name,dpi=125,bbox_inches='tight');plt.close(fig)
    def draw(ax,v,title,points=False):
        im=ax.imshow(v,extent=extent,origin='upper',cmap=cmap,norm=norm,interpolation='nearest')
        local(ax);ax.set_title(title,fontsize=10)
        if points:
            for name,(lon,lat) in POINTS.items():
                ax.plot(lon,lat,'kx',ms=4);ax.text(lon+.15,lat+.2,name,fontsize=8,bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
        return im
    fig,axs=plt.subplots(2,2,figsize=(15,11),constrained_layout=True)
    for ax,k in zip(axs.flat,cases):
        im=draw(ax,z[k],labels[k],True)
        fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.08,label='高程 / m')
    im=axs[1,1].imshow(z[sel]-z['0'],extent=extent,origin='upper',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-1800,vcenter=0,vmax=1800),interpolation='nearest')
    local(axs[1,1]);axs[1,1].set_title(labels[sel]+' − 控制；色带外极值见报表')
    fig.colorbar(im,ax=axs[1,1],orientation='horizontal',fraction=.035,pad=.08,label='高程变化 / m')
    save(fig,'01_fixed_r2.png','同一r2固定窗：小洼地、I4岛体与W/E两侧海槽；不是终态涂改')
    fig,axs=plt.subplots(2,3,figsize=(17,8),constrained_layout=True)
    uids=[356637,356671,356774,356712,356863,356857]
    for ax,uid in zip(axs.flat,uids):
        for k in cases:
            rows=[q for q in runs[k]['uid_tracks'] if q['uid']==uid]
            ax.plot([q['t_ma'] for q in rows],[q['T_after_extra_km'] for q in rows],label=labels[k])
        ax.axhline(32.5,color='gray',ls=':',label='旧薄壳门槛' if uid==uids[0] else None)
        ax.set_xlim(100,0);ax.set_title(f'UID {uid}');ax.set_xlabel('时间 / Ma');ax.set_ylabel('原生厚度 / km');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7)
    save(fig,'02_uid_history.png','真实存活UID逐步跟踪：首次出现前留空，不推断裂离前谱系；不是固定像元的假历史')
    raster=Raster(load_mesh(7),512)
    fig,axs=plt.subplots(3,4,figsize=(18,13),constrained_layout=True)
    for row,k in enumerate(cases):
        with np.load(cache/f'{cases[k]}/forward_v2_v2_L7.npz') as fr:
            sea={q['t']:q['sea_level_m'] for q in reports[k]['sea_level_m']}
            for ax,t in zip(axs[row],[100.,60.,30.,0.]):
                i=int(np.argmin(abs(fr['times']-t)));v=raster(fr[f'elev_{i}']).astype(float)-sea[t]
                im=draw(ax,v,labels[k]+f' · {t:g}Ma')
    fig.colorbar(im,ax=axs,orientation='horizontal',fraction=.018,pad=.04,label='相对当时海面的高程 / m')
    save(fig,'03_r2_history.png','同次正向演化真实快照：固定现代r2窗口；旧小洞显示处理沿用原模型')
    fig,axs=plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
    d=native['0'];lo,la=xyz_to_lonlat(d['positions']);pick=d['alive']&(lo>-14)&(lo<15)&(la>32)&(la<52)
    im=axs[0].scatter(lo[pick],la[pick],c=d['T'][pick],s=16,marker='h',cmap='magma_r',vmin=7,vmax=65)
    fig.colorbar(im,ax=axs[0],orientation='horizontal',fraction=.04,pad=.08,label='原生厚度 / km');axs[0].set_title('控制：原生厚度')
    categories=np.where(d['ctype']==1,0,np.where(d['age']<100,1,2))
    im=axs[1].scatter(lo[pick],la[pick],c=categories[pick],s=16,marker='h',cmap=ListedColormap(['#d9cda4','#60b9c7','#184880']),vmin=-.5,vmax=2.5)
    cb=fig.colorbar(im,ax=axs[1],orientation='horizontal',fraction=.04,pad=.08,ticks=[0,1,2]);cb.ax.set_xticklabels(['陆壳','<100Myr洋壳','≥100Myr洋壳'],fontsize=8)
    axs[1].set_title('控制：壳型与洋壳年龄（非成因的充分证据）')
    axs[2].axis('off')
    lines=['控制末态：固定探针最近的原生单元','名称  厚度km  现板块/来源  壳型/洋壳龄']
    for name in POINTS:
        q=summ['cases']['0']['fixed_probes'][name]
        tail='陆壳' if q['ctype']==1 else f"洋壳/{q['age_myr']:.0f}Myr"
        lines.append(f"{name:<3} {q['T_km']:5.1f}   {q['plate']}/{q['origin']}   {tail}")
    lines+=['','I4：继承陆壳轮廓，非本轮生长的半岛','W：15Myr洋壳；具体生成机制仍需事件对照','E：老洋壳残留；不自动等于活动海沟','P：薄陆壳洼地；无沉积/水位/排水求解','这些末态标签不足以单独确定细岸成因']
    axs[2].text(0,.98,'\n'.join(lines),va='top',fontsize=10,linespacing=1.8)
    for ax in axs[:2]:
        local(ax)
        for name,(lon,lat) in POINTS.items():ax.text(lon,lat,name,fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    save(fig,'04_feature_evidence.png','湾—岬—海槽实证：区分继承的岛体、残留古洋与年轻补充洋壳；不把颜色当成因')
    fig,axs=plt.subplots(2,2,figsize=(17,9),constrained_layout=True)
    for col,k in enumerate(['100','500']):
        im=axs[0,col].imshow(z[k],extent=extent,origin='upper',cmap=cmap,norm=norm,interpolation='nearest')
        axs[0,col].set_title(labels[k]);fig.colorbar(im,ax=axs[0,col],orientation='horizontal',fraction=.04,pad=.06,label='高程 / m')
        im=axs[1,col].imshow(z[k]-z['0'],extent=extent,origin='upper',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-1800,vcenter=0,vmax=1800),interpolation='nearest')
        axs[1,col].set_title('与同源控制的全球差值');fig.colorbar(im,ax=axs[1,col],orientation='horizontal',fraction=.04,pad=.06,label='变化 / m')
    for ax in axs.flat:ax.set_xlabel('经度 / °');ax.set_ylabel('纬度 / °');ax.grid(alpha=.2)
    save(fig,'05_global_cost.png','局部交换的全球代价；海陆变化、预算与运行峰值见comparison.json')
    fig,axs=plt.subplots(2,5,figsize=(22,8),constrained_layout=True)
    for row,k in enumerate(['0',sel]):
        with np.load(cache/f'{cases[k]}/forward_v2_v2_L7.npz') as fr:
            sea={q['t']:q['sea_level_m'] for q in reports[k]['sea_level_m']}
            for ax,t in zip(axs[row],[300.,200.,100.,50.,0.]):
                i=int(np.argmin(abs(fr['times']-t)));v=raster(fr[f'elev_{i}']).astype(float)-sea[t]
                im=ax.imshow(v,extent=extent,origin='upper',cmap=cmap,norm=norm,interpolation='nearest')
                ax.set_title(labels[k]+f' · {t:g}Ma',fontsize=9);ax.set_xticks([-120,0,120]);ax.set_yticks([-60,0,60]);ax.grid(alpha=.2)
    fig.colorbar(im,ax=axs,orientation='horizontal',fraction=.025,pad=.06,label='相对当时海面的高程 / m')
    save(fig,'06_global_history.png','真实300→0Ma全球快照；不是对现代地图做变形动画')
    from mythic.figures.region_panels import region_pair
    from mythic.figures.tectonic_panels import region_tectonics
    from mythic.figures.select_regions import RegionPick
    record=json.loads((ROOT/'figures/stage1/u38_b6_pl8_transition/regions.json').read_text(encoding='utf8'))['regions'][-1]
    pick=RegionPick(key='u38_r8_fixed',title='旧r8固定复查',rule='保留旧r8',earth_ref=None,center=tuple(record['center_lat_lon']),size_km=record['size_km'],reason='旧问题不因抽样关闭')
    world=load(str(paths[sel]));earth=load_earth(world.resolution)
    region_pair(pick,world,earth,record['earth_comparison'],out/'07_previous_r8.png',8)
    region_tectonics(pick,world,out/'08_previous_r8_tectonics.png',8)
    summ['limits']=['WT background exchange is uncalibrated, not demonstrated real basin genesis',
        'same native edges, no cross-plate contact added; source and rift old issues unchanged',
        'broad I4/cape inherited from design, finer embayments not all causally established',
        'oldr8, basin steps and plateau interiors remain open',
        'pre100Ma UID lineage absent; no inference beyond actual stored history',
        'defaultC2unchanged; stage1notaccepted']
    text=json.dumps(summ,ensure_ascii=False,indent=2)
    (out/'comparison.json').write_text(text,encoding='utf8',newline='\n')
    (folder/cases[sel]/'comparison.json').write_text(text,encoding='utf8',newline='\n')
    names=['00_frozen_spike.png','01_fixed_r2.png','02_uid_history.png','03_r2_history.png','04_feature_evidence.png','05_global_cost.png','06_global_history.png','07_previous_r8.png','08_previous_r8_tectonics.png']
    html='<!doctype html><meta charset="utf-8"><title>U43主内海r2</title><style>body{max-width:1500px;margin:30px auto;font-family:system-ui;background:#f5f2e9;color:#243430}img{width:100%;background:white}p{line-height:1.7}</style><h1>U43：主内海r2湾—岬—海槽</h1><p>本轮为问题诊断检查点：五处原水下点变成低地，但 I4 岛连到了大陆，主大陆距海指标退为 8/9。两档交换方案均不采用为默认世界。以下展示完整演化、固定岸段和实际代价；阶段 1 未通过。</p>'
    for n in names:html+=f'<p><a href="{n}">{n}</a><br><img src="{n}"></p>'
    html+=f'<p><a href="../../../../stage1/{cases[sel]}/index.html">29张标准图组与地球对照</a> · <a href="comparison.json">实际数据</a></p>'
    (out/'index.html').write_text(html,encoding='utf8',newline='\n')
    localize_earth_gallery(ROOT/'figures/stage1'/cases[sel])
    print(json.dumps(summ,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
